"""Tests of E82's protocol reading: the kernel weights of kc_remapping_harness (`kernel_window_weights`,
`kernel_weighted_counts`) against their limits, a brute-force integral and the planner's numbers; E82_flies_protocol's
kernel helpers and readings list; the gates on synthetic readings. The last tests read E80's saved runs and pulse.csv
(skipped when they are not on this machine): the counts reading equals E77's Scorer numbers in pulse.csv, and the step
kernel reproduces a 0.5 s run's counts and statistics exactly. No simulation."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kc_remapping_harness as H
import E82_flies_protocol as E

WINDOW = (2.0, 8.1)


def _brute(s, rise, decay, window=WINDOW, n=400001):
    """The kernel's area inside the window for a spike at s, by the trapezoid rule."""
    t = np.linspace(window[0], window[1], n)
    u = t - s
    k = np.where(u > 0, (1 - np.exp(-np.maximum(u, 0) / rise)) if rise > 0 else 1.0, 0.0)
    if np.isfinite(decay):
        k = k * np.exp(-np.maximum(u, 0) / decay)
    return np.trapezoid(k, t) if hasattr(np, 'trapezoid') else np.trapz(k, t)


def _spikes(n_kc=40, n_odor=5, n=600, t_max=1.9, seed=0):
    rng = np.random.default_rng(seed)
    st = pd.DataFrame(dict(kc=rng.integers(0, n_kc, n).astype(np.int32), odor=rng.integers(0, n_odor, n).astype(np.int32),
                           time_s=np.round(rng.uniform(0.0, t_max, n), 4)))
    st['step'] = np.round(st.time_s / 0.0005).astype(np.int32) + 4000
    idx = pd.MultiIndex.from_arrays([np.arange(n_kc) + 1000, ['ab'] * n_kc], names=['kc_id', 'kc_type'])
    cols = pd.Index([f'o{j} @ -3' for j in range(n_odor)], name='odor')
    counts = np.zeros((n_kc, n_odor))
    np.add.at(counts, (st.kc.to_numpy(), st.odor.to_numpy()), 1.0)
    return st, idx, cols, pd.DataFrame(counts, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# the harness's weights

def test_planner_weights():
    """A spike 50 ms after onset weighs 0.86, 0.62 and 0.17 of one at the pulse's end (2 s) at decays 5, 3 and 1 s with
    rises 1, 0.5 and 0.2 s (the E82 entry)."""
    got = [float(H.kernel_window_weights([0.05], r, d)[0]) for r, d in ((1.0, 5.0), (0.5, 3.0), (0.2, 1.0))]
    assert np.round(got, 2).tolist() == [0.86, 0.62, 0.17]


@pytest.mark.parametrize('rise,decay', [(0.2, 1.0), (0.5, 3.0), (1.0, 5.0), (0.0, 3.0), (0.5, np.inf), (0.0, np.inf),
                                        (1e-4, 1e4)])
def test_weights_match_brute_force(rise, decay):
    times = np.array([-0.3, 0.0, 0.05, 0.7, 1.99, 2.0, 2.4, 5.0, 8.05, 8.1, 9.0])
    w = H.kernel_window_weights(times, rise, decay)
    ref = _brute(2.0, rise, decay)
    want = np.array([_brute(s, rise, decay) for s in times]) / ref
    assert np.allclose(w, want, rtol=2e-5, atol=1e-6)
    assert w[times >= WINDOW[1]].max() == 0.0


def test_weights_tend_to_one():
    """For a zero rise and an infinite decay (a step) every spike before the window weighs exactly 1, and the weights tend
    to 1 as the rise goes to 0 and the decay to infinity."""
    times = np.linspace(-0.5, 1.999, 500)
    assert np.array_equal(H.kernel_window_weights(times, 0.0, np.inf), np.ones(len(times)))
    for eps in (1e-3, 1e-5, 1e-7):
        w = H.kernel_window_weights(times, eps, 1.0 / eps)
        assert np.abs(w - 1).max() < 20 * eps
    # inside the window a step counts the window's remaining length
    assert np.allclose(H.kernel_window_weights([2.0, 5.05, 8.1], 0.0, np.inf), [1.0, 0.5, 0.0])


def test_weights_monotone_and_reference():
    # (the weight rises with the spike's time until the kernel's value at the window's start equals its value at the
    # end, then falls: a spike up to about half a second before the pulse's end can outweigh one at it slightly, since
    # that one's kernel is still rising when the window opens; at most 1.8 % on the grid, at rise 1 s, decay 5 s)
    times = np.append(np.linspace(-0.5, 1.0, 200), 2.0)
    for r, d in E.KERNELS:
        w = H.kernel_window_weights(times, r, d)
        assert np.all(np.diff(w[:-1]) > 0) and w[-1] == 1.0
        assert H.kernel_window_weights(np.linspace(-0.5, 2.0, 2001), r, d).max() < 1.02
        # ref_time moves the unit only
        w2 = H.kernel_window_weights(times, r, d, ref_time=0.5)
        assert np.allclose(w2 / w, w2[0] / w[0])
        assert H.kernel_window_weights([0.5], r, d, ref_time=0.5)[0] == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError):
        H.kernel_window_weights([0.0], 0.5, 3.0, ref_time=9.0)
    with pytest.raises(ValueError):
        H.kernel_window_weights([0.0], -1.0, 3.0)


def test_weighted_counts_reproduce_counts_for_a_step():
    st, idx, cols, counts = _spikes()
    x = H.kernel_weighted_counts(st, idx, cols, 0.0, np.inf)
    assert x.index.equals(idx) and x.columns.equals(cols)
    assert np.array_equal(x.to_numpy(), counts.to_numpy())
    # the limit kernel is close, not equal
    y = H.kernel_weighted_counts(st, idx, cols, 1e-6, 1e6)
    assert np.allclose(y.to_numpy(), counts.to_numpy(), rtol=1e-4) and not np.array_equal(y.to_numpy(), counts.to_numpy())


@pytest.mark.parametrize('rise,decay', E.KERNELS)
def test_train_at_one_time(rise, decay):
    """A train of n spikes at one time returns n times that time's weight."""
    idx = pd.Index(range(3), name='kc_id')
    cols = pd.Index(['a', 'b'], name='odor')
    st = pd.DataFrame(dict(kc=[1] * 7 + [2] * 3, odor=[0] * 7 + [1] * 3, time_s=[0.3] * 7 + [1.2] * 3))
    x = H.kernel_weighted_counts(st, idx, cols, rise, decay)
    w = H.kernel_window_weights([0.3, 1.2], rise, decay)
    assert x.iloc[1, 0] == pytest.approx(7 * w[0], rel=1e-14) and x.iloc[2, 1] == pytest.approx(3 * w[1], rel=1e-14)
    assert x.to_numpy().sum() == pytest.approx(7 * w[0] + 3 * w[1], rel=1e-14)
    assert (x.iloc[0] == 0).all()


def test_weighted_counts_onset_and_bounds():
    st, idx, cols, _ = _spikes()
    a = H.kernel_weighted_counts(st, idx, cols, 0.5, 3.0)
    shifted = st.assign(time_s=st.time_s + 0.25)
    b = H.kernel_weighted_counts(shifted, idx, cols, 0.5, 3.0, stim_onset=0.25)
    assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12)
    with pytest.raises(ValueError):
        H.kernel_weighted_counts(st.assign(kc=st.kc + 1000), idx, cols, 0.5, 3.0)


# ---------------------------------------------------------------------------
# E82's helpers

def test_readings_list():
    labs = [lab for lab, _ in E.readings()]
    assert labs[:2] == ['counts', 'step'] and len(labs) == 2 + len(E.KERNELS) == 11
    assert E.klabel(E.PRIMARY) == 'r0.5_d3' and E.klabel(None) == 'counts' and E.klabel((0.0, np.inf)) == 'step'
    assert set(E.DIAGONAL) <= set(E.KERNELS) and E.PRIMARY in E.KERNELS


@pytest.mark.parametrize('rise,decay', [(0.2, 1.0), (0.5, 3.0), (1.0, 5.0), (0.0, 2.0), (0.3, np.inf)])
def test_trace_kernel_and_integral(rise, decay):
    u = np.linspace(0, 8, 160001)
    k = E._kernel(u, rise, decay)
    num = np.concatenate([[0.0], np.cumsum((k[1:] + k[:-1]) / 2 * np.diff(u))])
    assert np.allclose(E._kernel_integral(u, rise, decay), num, atol=1e-6 if rise > 0 else 5e-5)
    assert E._kernel(np.array([-1.0, 0.0]), rise, decay).tolist() == [0.0, 0.0]
    # the harness's weight is the integral's difference over the window, over a spike at the window's start
    s = np.array([0.05, 1.0, 3.0])
    G = lambda x: E._kernel_integral(x, rise, decay)
    want = (G(WINDOW[1] - s) - G(WINDOW[0] - s)) / G(WINDOW[1] - WINDOW[0])
    assert np.allclose(H.kernel_window_weights(s, rise, decay), want, rtol=1e-10)


# ---------------------------------------------------------------------------
# the gates on synthetic readings

def _gate_inputs():
    pairs16 = [tuple(sorted(p)) for p in H.pair_list(H.APL16_ODORS)]
    pidx = pd.MultiIndex.from_tuples(sorted(pairs16), names=['odor_a', 'odor_b'])
    rng = np.random.default_rng(3)
    c = pd.Series(rng.normal(0, 0.05, len(pidx)), index=pidx)
    c = c - c.mean()
    P = E.klabel(E.PRIMARY)
    runs = pd.DataFrame([
        dict(source='E80', point_id='fit0_pulse0.5_fixed', model='fit0', pulse=0.5, held=False, identity=True, arm='', ref_point=''),
        dict(source='E80', point_id='fit0_pulse2_held', model='fit0', pulse=2.0, held=True, identity=False, arm='', ref_point=''),
        dict(source='E80', point_id='C2_pulse2_held', model='C2', pulse=2.0, held=True, identity=False, arm='', ref_point=''),
        dict(source='E82B', point_id='B2_fit0_tau3', model='fit0', pulse=2.0, held=True, identity=False, arm='B2', ref_point=''),
        dict(source='E82B', point_id='B2_fit0_tau10', model='fit0', pulse=2.0, held=True, identity=False, arm='B2', ref_point=''),
        dict(source='E82B', point_id='B5_C2_taum0.25', model='C2', pulse=2.0, held=True, identity=False, arm='B5', ref_point=''),
        dict(source='E82B', point_id='B1_fit0_pulse0.5', model='fit0', pulse=0.5, held=False, identity=True, arm='B1', ref_point=''),
    ])
    base = dict(raw_mean=0.15, cal_mean=0.15, cal_rho=0.5, raw_pattern=0.05, rate_on=0.068, ratio=2.2, realism_flag=np.nan,
                realism_flag_zcrit=np.nan)
    vals = {  # (raw_loss counts, raw_loss primary, cal_loss, cal_pattern primary, max on, max off, change as a multiple of c)
        'fit0_pulse0.5_fixed': (0.0503, 0.0505, 0.0503, 0.0497, 8, 20, 0.0),
        'fit0_pulse2_held': (0.0612, 0.0705, 0.0700, 0.0550, 10, 24, 0.0),
        'C2_pulse2_held': (0.146, 0.158, 0.058, 0.057, 124, 236, 0.0),
        'B2_fit0_tau3': (0.0540, 0.0560, 0.0560, 0.0510, 9, 22, -0.5),
        'B2_fit0_tau10': (0.0520, 0.0530, 0.0530, 0.0505, 9, 30, -0.5),
        'B5_C2_taum0.25': (0.150, 0.160, 0.060, 0.0560, 120, 230, 0.2),
        'B1_fit0_pulse0.5': (0.0503, 0.0505, 0.0503, 0.0497, 8, 20, 0.0),
    }
    rd, pr = [], []
    for r in runs.itertuples():
        rc, rp, cl, cp, mon, moff, f = vals[r.point_id]
        for k, rl in (('counts', rc), (P, rp)):
            rd.append(dict(base, source=r.source, point_id=r.point_id, kernel=k, raw_loss=rl, cal_loss=cl, cal_pattern=cp,
                           max_count_on=mon, max_count_off=moff))
            eff = 0.18 + f * c
            pr.append(pd.DataFrame(dict(source=r.source, point_id=r.point_id, kernel=k, readout='cal', odor_a=pidx.get_level_values(0),
                                        odor_b=pidx.get_level_values(1), on=0.8, off=0.8 - eff.to_numpy(), effect=eff.to_numpy())))
    return pd.DataFrame(rd), pd.concat(pr, ignore_index=True), runs, c


def test_gate_rows(monkeypatch):
    rd, pairs, runs, c = _gate_inputs()
    monkeypatch.setattr(E, 'data', lambda: {'common': {'cal': c}, 'c6': {'cal': c}})
    g = E.gate_rows(rd, pairs, runs).set_index('point_id')
    assert set(g.index) == {'B2_fit0_tau3', 'B2_fit0_tau10', 'B5_C2_taum0.25'}
    t3, t10, b5 = g.loc['B2_fit0_tau3'], g.loc['B2_fit0_tau10'], g.loc['B5_C2_taum0.25']
    # the protocol gate: 0.0503 + 0.006 = 0.0563; maxima 10 / 24
    assert t3.pg_limit == pytest.approx(0.0563) and t3.pg_counts and t3.pg_primary and t3.pg_max and t3.protocol_gate
    assert t10.pg_counts and t10.pg_primary and not t10.pg_max and not t10.protocol_gate
    assert np.isnan(b5.protocol_gate)
    # E77's gate against the re-held 2 s point of the same model
    assert t3.ref_point == 'E80:fit0_pulse2_held' and b5.ref_point == 'E80:C2_pulse2_held'
    assert t3.pattern_gain == pytest.approx(0.004) and t3.transfer_all == pytest.approx(0.75) and t3.e77_c2
    assert t3.e77_c1 and t3.e77_gate
    # B5: no pattern gain, a change along the common error (transfer negative)
    assert not b5.e77_c1 and b5.transfer_all < 0 and not b5.e77_gate


def test_gate_rows_bands(monkeypatch):
    rd, pairs, runs, c = _gate_inputs()
    rd['realism_flag'] = rd['realism_flag'].astype(object)
    rd.loc[rd.point_id == 'B2_fit0_tau3', 'realism_flag'] = 'APL-on response rate 0.100 (band 0.045-0.085)'
    monkeypatch.setattr(E, 'data', lambda: {'common': {'cal': c}, 'c6': {'cal': c}})
    g = E.gate_rows(rd, pairs, runs).set_index('point_id')
    assert not g.loc['B2_fit0_tau3'].inside_bands and not g.loc['B2_fit0_tau3'].protocol_gate
    assert not g.loc['B2_fit0_tau3'].e77_gate


# ---------------------------------------------------------------------------
# on E80's saved runs

def _e80_available():
    try:
        p = H.results_dir(E.E80) / 'pulse.csv'
        return p.exists() and E._find(E._run_dirs(E.E80, 'pulse'), 'fit0_pulse0.5_fixed') is not None
    except Exception:
        return False


@pytest.mark.skipif(not _e80_available(), reason="E80's saved pulse runs not on this machine")
def test_e80_counts_and_step_reading():
    """fit 0 at the 0.5 s pulse: the counts reading equals E80's pulse.csv (E77's Scorer) and the step reading the
    counts' to the last digit; the primary reading differs."""
    d = E._find(E._run_dirs(E.E80, 'pulse'), 'fit0_pulse0.5_fixed')
    o, st = E.load_run(d, 'fit0_pulse0.5_fixed')
    resp = {c: o[c][0] for c in o}
    xc, xs, xp = (E.weighted(o, st, k) for k in (None, E.STEP, E.PRIMARY))
    assert all(np.array_equal(xs[c].to_numpy(), xc[c].to_numpy()) for c in ('on', 'off'))
    rc, pc = E.score_reading(resp, xc)
    rs, ps = E.score_reading(resp, xs)
    rp, _ = E.score_reading(resp, xp)
    assert rc == rs or all((rc[k] == rs[k]) or (np.isnan(rc[k]) and np.isnan(rs[k])) for k in rc)
    assert pc.equals(ps)
    ref = pd.read_csv(H.results_dir(E.E80) / 'pulse.csv').set_index('point_id').loc['fit0_pulse0.5_fixed']
    for k in ('raw_loss', 'raw_mean', 'raw_pattern', 'raw_rho', 'cal_loss', 'cal_pattern', 'raw_err_ester', 'raw_proj_common_six'):
        assert rc[k] == pytest.approx(float(ref[k]), abs=1e-12), k
    assert rc['raw_n_pairs'] == 120 and rc['raw_z6_n_pairs'] == 120 and rc['raw_noise_n_pairs'] == 120
    assert rp['raw_loss'] != rc['raw_loss']
