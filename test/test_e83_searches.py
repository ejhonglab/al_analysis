#!/usr/bin/env python3
"""Tests for E83's searches (scripts/claude_pair_remapping_experiments/E83_searches.py; parts B and C of the input and
APL's glomerulus profile fitted on several wirings at once): the evolution strategy's draws and step size, the APL scale
steps, the rate penalty, the candidates' objective path, the linear start, the input tables; and, with simulations, the
identities the searches start from: zero scores and factor 1 reproduce each searched wiring's saved counts exactly, the
population objective at identity is 0.0569 (half A) and 0.0552 (half B), and the real wiring's profile factors are
E75's. Needs the shared directory (the saved fits and the data); skipped without it. The simulation tests take a few
minutes (run with OMP_NUM_THREADS / OMP_THREAD_LIMIT set)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
H = pytest.importorskip('kc_remapping_harness')
try:
    H.data_targets_path().read_bytes()
except (FileNotFoundError, OSError):
    pytest.skip('the shared directory (data-target registry) is not available', allow_module_level=True)
X = pytest.importorskip('E83_cross_wiring_fit')
SR = pytest.importorskip('E83_searches')


def _have_fits() -> bool:
    try:
        return all(X.saved_fit(w)['saved'] is not None for w in X.S_WIRINGS)
    except Exception:
        return False


needs_fits = pytest.mark.skipif(not _have_fits(), reason="the four searched wirings' saved fits are not available")


# ---------------------------------------------------------------------------
# no simulation

def test_es_offspring_is_e77s_draw():
    """The arm-I offspring are E77 part E's: exp(clip(log(parent) + sigma N, log 0.5, log 2)), one draw per generation."""
    parent = np.exp(np.random.default_rng(5).normal(0, 0.3, 36))
    lo, hi = np.log(SR.I_BOUNDS[0]), np.log(SR.I_BOUNDS[1])
    a = np.exp(SR.es_offspring(np.random.default_rng(0), np.log(parent), 0.3, 32, lo, hi))
    r = np.random.default_rng(0)
    b = np.exp(np.clip(np.log(parent)[None, :] + 0.3 * r.standard_normal((32, 36)), lo, hi))
    assert np.array_equal(a, b)
    assert a.shape == (32, 36) and a.min() >= 0.5 - 1e-12 and a.max() <= 2.0 + 1e-12
    s = SR.es_offspring(np.random.default_rng(1), np.zeros(36), 5.0, 16, -np.log(4), np.log(4))
    assert np.abs(s).max() <= np.log(4) + 1e-12 and (np.abs(s) == np.log(4)).any()


def test_es_sigma():
    assert SR.es_sigma(0.1, True) == pytest.approx(0.15)
    assert SR.es_sigma(0.1, False) == pytest.approx(0.08)
    assert SR.es_sigma(0.035, False) == pytest.approx(0.03)
    assert SR.es_sigma(0.9, True) == pytest.approx(1.0)


def _counter(n0: int, e: float, calls: list, size=(400, 16)):
    """run_on(x) whose response count is round(n0 x^e), as a responses frame (and the same as spike counts)."""
    def run_on(x):
        calls.append(x)
        n = int(round(n0 * x ** e))
        r = np.zeros(size[0] * size[1])
        r[:n] = 1.0
        r = pd.DataFrame(r.reshape(size))
        return r, 2 * r
    return run_on


def test_step_apl_scale_converges():
    calls = []
    # the parent's factor gives 1100 responses for a target of 1000; the true elasticity -1.4, the guess -1
    res = SR.step_apl_scale(_counter(1100, -1.4, calls), 1000, 1.0, -1.0)
    assert res['rel_dev'] <= SR.APL_TOL and res['penalty'] == 0.0
    assert 2 <= res['n_sims'] <= 1 + SR.APL_MAX_STEPS and res['n_sims'] == len(calls)
    assert res['n'] == int((res['out'][0].to_numpy() > 0.5).sum())
    # within tolerance at the start: one run
    calls = []
    res = SR.step_apl_scale(_counter(1005, -1.4, calls), 1000, 1.0, -1.0)
    assert res['n_sims'] == 1 and res['x'] == 1.0
    # `first` is the run at x0: no run needed when it is within tolerance
    calls = []
    first = _counter(995, -1.4, [])(1.0)
    res = SR.step_apl_scale(_counter(995, -1.4, calls), 1000, 1.0, -1.0, first=first)
    assert res['n_sims'] == 0 and not calls


def test_step_apl_scale_penalizes_what_is_left():
    # a count that barely moves: three steps, then the closest run and a penalty for the rest
    calls = []
    res = SR.step_apl_scale(_counter(1500, -0.01, calls), 1000, 1.0, -1.0)
    assert res['n_sims'] == 1 + SR.APL_MAX_STEPS
    assert res['n'] == min(int(round(1500 * x ** -0.01)) for x in calls)
    assert res['penalty'] == pytest.approx(SR.APL_PENALTY * (res['rel_dev'] - SR.APL_TOL))
    # every step at most APL_MAX_STEP times the last, inside the bracket
    xs = [1.0] + calls[1:]
    assert all(abs(np.log(b / a)) <= np.log(SR.APL_MAX_STEP) + 1e-12 for a, b in zip(xs, xs[1:]))
    assert all(X.APL_BRACKET[0] <= x <= X.APL_BRACKET[1] for x in calls)
    with pytest.raises(ValueError):
        SR.step_apl_scale(_counter(1000, -1, []), 1000, 1.0, 0.5)


def test_rate_penalty():
    base = {'a': (0.1, 0.2), 'b': (0.05, 0.0)}
    assert SR.rate_penalty({'a': (0.11, 0.18), 'b': (0.055, 0.3)}, base) == 0.0
    # a: on +20 % (5 % beyond), off -25 % (10 % beyond); b: on -30 % (15 % beyond), off has base 0
    p = SR.rate_penalty({'a': (0.12, 0.15), 'b': (0.035, 0.5)}, base)
    assert p == pytest.approx(SR.I_PENALTY * (0.05 + 0.10 + 0.15))


def test_drop_other_frames(tmp_path):
    for tag in ('start', 'g0', 'g3'):
        for w in ('real', 'P1'):
            (tmp_path / f'{w}__{tag}_on_spike_counts.parquet').write_bytes(b'')
    (tmp_path / 'checkpoint.json').write_text('{}')
    SR._drop_other_frames(tmp_path, 'g3')
    assert sorted(p.name for p in tmp_path.glob('*.parquet')) == ['P1__g3_on_spike_counts.parquet', 'real__g3_on_spike_counts.parquet']
    SR._reset_state(tmp_path)
    assert not list(tmp_path.iterdir())


@needs_fits
@pytest.mark.parametrize('half', ['A', 'B'])
def test_population_objective_at_identity(half):
    """S's pattern part under the readout at the fits: 0.0569 against half A, 0.0552 against half B; the per-pair error
    the linear start uses gives the objective's squared sum."""
    obj = X.PopulationObjective(X.half_target(half), X.S_WIRINGS)
    fits = {w: X.saved_fit(w) for w in X.S_WIRINGS}
    on = {w: f['saved']['on'][1] for w, f in fits.items()}
    off = {w: f['saved']['off'][1] for w, f in fits.items()}
    o = obj(on)
    assert round(o['rms'], 4) == X.IDENTITY_OBJECTIVE[half] and o['n_pairs'] == 120
    assert obj(on=on, off=off)['sse'] == o['sse'] == obj()['sse']
    err = SR.population_error(on, off, {w: f['K'] for w, f in fits.items()}, X.half_target(half))
    assert len(err) == 120 and abs(float((err ** 2).sum()) - o['sse']) < 1e-12


@needs_fits
def test_candidate_objectives_identity_path():
    """Candidates whose odor column is the incumbent's own give the objective at identity and no penalty; a candidate
    with the column changed moves the objective."""
    obj = X.PopulationObjective(X.half_target('A'), X.S_WIRINGS)
    fits = {w: X.saved_fit(w) for w in X.S_WIRINGS}
    inc = {w: {c: f['saved'][c] for c in ('on', 'off')} for w, f in fits.items()}
    odor = 'eb'
    oc = SR._odor_col(inc['real']['on'][1], odor)
    res = {}
    for w in X.S_WIRINGS:
        cols = [inc[w][c][k][oc].to_numpy(dtype=float) for c in ('on', 'off') for k in (0, 1)]
        res[w] = tuple(np.stack([v, v], axis=1) for v in cols)
    # the second candidate: APL-on counts of the odor doubled on the real wiring
    res['real'][1][:, 1] *= 2
    base = {w: (float((res[w][0][:, 0] > 0.5).mean()), float((res[w][2][:, 0] > 0.5).mean())) for w in X.S_WIRINGS}
    rows = SR.candidate_objectives(inc, odor, res, obj, base)
    o0 = obj({w: inc[w]['on'][1] for w in X.S_WIRINGS})
    assert rows[0]['sse'] == pytest.approx(o0['sse'], abs=1e-15) and rows[0]['penalty'] == 0.0
    assert rows[1]['sse'] != pytest.approx(o0['sse'], abs=1e-9)
    # the incumbents are not changed by the scoring
    assert inc['real']['on'][1][oc].equals(fits['real']['saved']['on'][1][oc])


@needs_fits
@pytest.mark.parametrize('half', ['A', 'B'])
def test_linear_start(half):
    fits = {w: X.saved_fit(w) for w in X.S_WIRINGS}
    on = {w: f['saved']['on'][1] for w, f in fits.items()}
    off = {w: f['saved']['off'][1] for w, f in fits.items()}
    Ks = {w: f['K'] for w, f in fits.items()}
    s = SR.linear_start(half, on, off, Ks)
    import E75_glomerulus_apl_profile as E75
    M = E75.map_matrix(H.results_dir(E75.EXPERIMENT), 'cell')
    assert list(s) == list(M.columns) and len(s) == 36
    v = np.array(list(s.values()))
    assert np.abs(v).max() <= SR.P_START_STRENGTH * np.log(4) + 1e-12 and np.abs(v).max() > 0
    assert s == SR.linear_start(half, on, off, Ks)


def test_write_input_tables(tmp_path):
    orn = H.apl16_orn_deltas()
    rng = np.random.default_rng(3)
    done = {'eb': np.exp(rng.uniform(np.log(0.5), np.log(2), len(orn.index))).tolist(),
            'ms': np.exp(rng.uniform(np.log(0.5), np.log(2), len(orn.index))).tolist()}
    fac = SR.write_input_tables(tmp_path, 'A', orn, done)
    assert list(fac.columns) == ['half', 'odor', 'glomerulus', 'orn_delta', 'factor', 'log2_factor', 'searched_delta']
    assert list(dict.fromkeys(fac.odor)) == ['eb', 'ms']
    F = X.read_factors(tmp_path / 'input_factors_A.csv')
    ab = list(H.abbrev_columns(orn).columns)
    for o in ab:
        want = np.asarray(done[o]) if o in done else np.ones(len(orn.index))
        # read_factors parses the CSV with pandas' default float parser (equal to within an ulp)
        assert np.allclose(F[o].to_numpy(), want, rtol=1e-14, atol=0)
    from E10_joint_fit import read_orn_input
    got = read_orn_input(str(tmp_path / 'inputs' / 'searched_halfA.csv'), orn)
    want = X.input_at(F, 1.0, orn)
    assert np.allclose(got.to_numpy(dtype=float), want.to_numpy(dtype=float), rtol=1e-14, atol=0)
    for o in ab:
        if o not in done:
            j = ab.index(o)
            assert np.array_equal(got.iloc[:, j].to_numpy(dtype=float), orn.iloc[:, j].to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# simulations: the identities the searches start from

@pytest.fixture(scope='module')
def models():
    return {w: X.WiringModel(w, check=False) for w in X.S_WIRINGS}


@needs_fits
@pytest.mark.parametrize('w', X.S_WIRINGS)
def test_zero_scores_reproduce_the_fit(models, w):
    """Arm P's identity: every glomerulus's score 0 at the APL factor 1 gives the wiring's saved APL-on run exactly."""
    import E75_glomerulus_apl_profile as E75
    m = models[w]
    gl = list(E75.map_matrix(H.results_dir(E75.EXPERIMENT), 'cell').columns)
    m.set_profile(SR.P_SITE, {g: 0.0 for g in gl})
    try:
        assert np.array_equal(m.current_wAPLKC().to_numpy(), m.wAPLKC.to_numpy())
        r, s = m.run_on(1.0)
    finally:
        m.set_profile(None, {})
    assert np.array_equal(s.to_numpy(dtype=float), m.saved['on'][1].to_numpy(dtype=float))
    assert np.array_equal(r.to_numpy(dtype=float), m.saved['on'][0].to_numpy(dtype=float))
    assert s.index.equals(m.saved['on'][1].index)


@needs_fits
def test_real_profile_factors_are_e75s(models):
    """The real wiring's profile (E75's ridge cell profile at half strength) puts E75's factors on E75's breadth-weighted
    APL->KC weights, and its archived point (at E75's APL factor) is reproduced exactly."""
    import E10_joint_fit as J
    import E72_wiring_population as E72
    import E75_glomerulus_apl_profile as E75
    out75 = H.results_dir(E75.EXPERIMENT)
    sc = X.e75_profile('ridge_x0.5')
    _, wnorm, W, breadth = J.bouton_base(H.apl16_orn_deltas(), H.outputs_dir(E75.EXPERIMENT), None)
    wk75 = J.breadth_weights(wnorm['wAPLKC'], breadth, 2.0)
    f75 = E75.profile_factors('cell', sc, W, wk75, wnorm['wAPLPN'])
    m = models['real']
    assert W.equals(m.W) and wk75.equals(m.wAPLKC)
    f = E75.profile_factors('cell', sc, m.W, m.wAPLKC, m.wAPLPN)
    assert f.equals(f75)
    m.set_profile('cell', sc)
    try:
        assert np.array_equal(m.current_wAPLKC().to_numpy(), (wk75 * f75).to_numpy())
        pt = pd.read_csv(out75 / 'sensitivity_map.csv').set_index('point_id').loc['D2s1:cell:ridge:x0.5']
        r, s = m.run_on(float(pt.factor))
    finally:
        m.set_profile(None, {})
    saved = E72.load_fit(E75.EXPERIMENT, 'C2_D2s1_cell_ridge_x0.5')
    assert saved is not None
    assert np.array_equal(s.to_numpy(dtype=float), saved['on'][1].to_numpy(dtype=float))
    assert int((r.to_numpy(dtype=float) > 0.5).sum()) == int(pt.n_responses)


@needs_fits
@pytest.mark.parametrize('w', X.S_WIRINGS)
def test_factor_one_reproduces_the_fit(w):
    """Arm I's identity: the first searched odor as a pseudo-odor at factor 1, the lab input's breadth held, gives the
    wiring's saved APL-on and APL-off counts for that odor exactly."""
    from E64_input_oracle import candidate_orn
    P = X.wiring_parts(w)
    odor = SR.I_ODORS[0]
    sim, _ = X.build_simulator(w, orn=candidate_orn(P['orn'], odor, np.ones((1, len(P['orn'].index)))), breadth=P['breadth'])
    on, off = sim(dict(P['p'])), sim(dict(P['p'], wAPLKC_scale=0.0, wAPLPN_scale=0.0))
    oc = SR._odor_col(P['saved']['on'][1], odor)
    for c, o in (('on', on), ('off', off)):
        assert np.array_equal(o[1].iloc[:, 0].to_numpy(dtype=float), P['saved'][c][1][oc].to_numpy(dtype=float))
        assert np.array_equal(o[0].iloc[:, 0].to_numpy(dtype=float), P['saved'][c][0][oc].to_numpy(dtype=float))


@needs_fits
def test_input_pool_one_wiring():
    """The pool's worker path: two candidates at factor 1 on the real wiring give its saved counts for the odor."""
    pool = SR.InputPool(['real'], threads=4)
    try:
        P = X.wiring_parts('real')
        res = pool.run('ms', np.ones((2, len(P['orn'].index))))
    finally:
        pool.close()
    oc = SR._odor_col(P['saved']['on'][1], 'ms')
    r_on, s_on, r_off, s_off = res['real']
    for i in range(2):
        assert np.array_equal(s_on[:, i].astype(float), P['saved']['on'][1][oc].to_numpy(dtype=float))
        assert np.array_equal(s_off[:, i].astype(float), P['saved']['off'][1][oc].to_numpy(dtype=float))
        assert np.array_equal(r_on[:, i].astype(float), P['saved']['on'][0][oc].to_numpy(dtype=float))
