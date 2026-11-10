#!/usr/bin/env python3
"""Tests for the low end of the readout in the KC-remapping harness (E68:
scripts/claude_pair_remapping_experiments/kc_remapping_harness.py): the saturating readout's one-spike
floor, the expected cosine under measurement noise, the pseudo-recordings, the closed-form low-end
statistics, the conversion of readout units to z, the joint calibration, the three-term split of a cosine
and the added readings of the KC data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
H = pytest.importorskip('kc_remapping_harness')

ODORS = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep']


def _counts(seed=0, n_units=400, odors=ODORS, p_resp=0.08):
    """Spike-count-like matrix: mostly 0, some 1, responders with geometric counts."""
    rng = np.random.default_rng(seed)
    u = rng.random((n_units, len(odors)))
    n = np.where(u < p_resp, 2 + rng.geometric(0.15, u.shape), np.where(u < p_resp + 0.05, 1, 0))
    return pd.DataFrame(n.astype(float), columns=odors)


def _shared_available():
    try:
        H.shared_dir()
        return True
    except FileNotFoundError:
        return False


needs_shared = pytest.mark.skipif(not _shared_available(), reason='shared directory not mounted')


def test_floor_formula_and_default():
    c = _counts()
    K = 7.0
    plain = H.readout(c, 'saturating', K)
    assert np.array_equal(plain.to_numpy(), (c / (c + K)).to_numpy())
    assert np.array_equal(H.readout(c, 'saturating', K, floor=0.0).to_numpy(), plain.to_numpy())
    for b in (-1.0, -0.5, 0.5, 3.0):
        r = H.readout(c, 'saturating', K, floor=b).to_numpy()
        x = c.to_numpy()
        expect = np.where(x > 0, (x + b) / (x + b + K), 0.0)
        assert np.allclose(r, expect)
        assert (r[x == 0] == 0).all()
    assert (H.readout(c, 'saturating', K, floor=-1.0).to_numpy()[c.to_numpy() == 1] == 0).all()
    with pytest.raises(ValueError):
        H.readout(c, 'power', 0.5, floor=1.0)
    with pytest.raises(ValueError):
        H.readout(c, 'saturating', K, floor=-2.0)


def test_calibrate_readout_default_unchanged():
    c = _counts(1)
    k0 = H.calibrate_readout(c, 'saturating', 0.05, responders_min=2.0)
    k1 = H.calibrate_readout(c, 'saturating', 0.05, responders_min=2.0, floor=0.0)
    assert k0 == k1
    assert H.readout_stat(c, 'saturating', k0, 'participation_ratio', 2.0) == \
        H.readout_stat(c, 'saturating', k0, 'participation_ratio', 2.0, floor=0.0)


@needs_shared
def test_defaults_reproduce_archived_e65_values():
    """The calibrated saturating readout of the reference model, as the compressive readout experiment
    (E65) saved it, is reproduced bit for bit by the defaults (no floor)."""
    shared = H.shared_dir()
    table = shared / 'results' / 'E65_spike_count_readout' / 'by_variant_readout.csv'
    folder, tag = 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'
    on_p = shared / 'data' / 'model_outputs' / folder / f'{tag}_on_spike_counts.parquet'
    off_p = shared / 'data' / 'model_outputs' / folder / f'{tag}_acute_off_spike_counts.parquet'
    if not (table.exists() and on_p.exists() and off_p.exists()):
        pytest.skip('E65 table or archived counts missing')
    e65 = pd.read_csv(table, low_memory=False)
    row = e65[(e65.folder == folder) & (e65.tag == tag) & (e65.readout == 'saturating_cal_pr')].iloc[0]
    on, off = pd.read_parquet(on_p), pd.read_parquet(off_p)
    target = H.data_target('ctrl_participation_ratio_resp')
    K = H.calibrate_readout(on, 'saturating', target, stat='participation_ratio', responders_min=2.0)
    assert K == pytest.approx(row['param'], rel=1e-12)
    st = H.apl_effect_stats(H.readout_pair_distances(on, 'saturating', K), H.readout_pair_distances(off, 'saturating', K),
        H.yang_apl_effect())
    assert st['apl_effect_rmse'] == pytest.approx(row['apl_effect_rmse'], rel=1e-12)
    assert st['apl_effect_mean'] == pytest.approx(row['apl_effect_mean'], rel=1e-12)
    d0 = H.readout_pair_distances(on, 'saturating', K)
    d1 = H.readout_pair_distances(on, 'saturating', K, floor=0.0)
    assert np.array_equal(d0.to_numpy(), d1.to_numpy())


def test_expected_noisy_distances():
    c = H.readout(_counts(2, n_units=2000), 'saturating', 5.0) * 4.0
    d0 = H.expected_pair_distances_noisy(c, 0.0)
    assert np.allclose(d0.to_numpy(), H.pair_distances(c, 'cosine').to_numpy())
    s = 0.4
    exp = H.expected_pair_distances_noisy(c, s, n_trials=3)
    means = H.sample_pseudo_recordings(c, s, n_recordings=200, n_trials=3, seed=1)
    sampled = np.mean([H.pair_distances(pd.DataFrame(m, columns=c.columns), 'cosine').to_numpy() for m in means], axis=0)
    assert np.allclose(exp.to_numpy(), sampled, atol=3e-3)
    assert (exp.to_numpy() > d0.to_numpy()).all()


def test_sample_pseudo_recordings_shapes_and_seed():
    c = _counts(3)
    m = H.sample_pseudo_recordings(c, 0.3, n_recordings=4, n_trials=3, seed=5)
    t = H.sample_pseudo_recordings(c, 0.3, n_recordings=4, n_trials=3, seed=5, return_trials=True)
    assert m.shape == (4,) + c.shape and t.shape == (4, 3) + c.shape
    assert np.allclose(t.mean(axis=1), m)
    trials = np.repeat(c.to_numpy()[None, None], 3, axis=1).repeat(2, axis=0)
    t2 = H.sample_pseudo_recordings(trials, 0.0, return_trials=True)
    assert np.array_equal(t2, trials)


def test_noise_corrected_matches_analyse():
    NF = pytest.importorskip('noise_floor_trial_distances')
    rng = np.random.default_rng(4)
    base = _counts(4).to_numpy()
    trials = base[None] + rng.normal(0, 0.5, (3,) + base.shape)
    by_odor = {o: trials[:, :, j] for j, o in enumerate(ODORS)}
    _, pr = NF.analyse(by_odor, 'graded')
    mine = H.noise_corrected_pair_distances(trials, ODORS)
    ref = H.canonical_pairs(pr['corrected'])
    a, b = H.align_pairs(mine, ref)
    assert np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12)


def test_low_end_stats_closed_form_vs_sampling():
    rng = np.random.default_rng(6)
    mu = np.where(rng.random((5, 3000)) < 0.1, rng.gamma(4, 0.5, (5, 3000)), np.where(rng.random((5, 3000)) < 0.1, 0.7, 0.0))
    sd = 0.25
    cf = H.low_end_stats(mu, sd)
    draws = mu[None] + rng.normal(0, sd, (400,) + mu.shape)
    rate = (draws > 1).mean()
    band = ((draws > 0.5) & (draws <= 1)).mean()
    x2 = np.where(draws > 1, draws ** 2, 0).sum(axis=2).mean(axis=0)
    x4 = np.where(draws > 1, draws ** 4, 0).sum(axis=2).mean(axis=0)
    pr = np.mean(x2 ** 2 / (mu.shape[1] * x4))
    assert cf['rate'] == pytest.approx(rate, rel=0.01)
    assert cf['band_share'] == pytest.approx(band, rel=0.02)
    assert cf['participation_ratio_resp'] == pytest.approx(pr, rel=0.01)
    # grouping identical entries gives the same statistics
    u, cnt = np.unique(mu[0], return_counts=True)
    g = H.low_end_stats(u[None], sd, weights=cnt[None], n_units=np.array([mu.shape[1]]))
    one = H.low_end_stats(mu[:1], sd)
    for k in g:
        assert g[k] == pytest.approx(one[k], rel=1e-10)
    nf = H.low_end_stats(mu, 0.0)
    assert nf['rate'] == pytest.approx((mu > 1).mean())


def test_z_scale_hits_target_rate():
    r = H.readout(_counts(7, n_units=1500), 'saturating', 8.0)
    c = H.z_scale(r, 0.068, 0.39, n_trials=3)
    st = H.low_end_stats(c * r.to_numpy().T, 0.39 / np.sqrt(3))
    assert st['rate'] == pytest.approx(0.068, abs=1e-9)


def test_calibrate_low_end_reaches_targets():
    counts = _counts(8, n_units=1500, p_resp=0.075)
    targets = dict(target_rate=0.068, target_band=0.04, target_pr=0.05)
    cal = H.calibrate_low_end(counts, noise_sd=0.39, K0=10.0, **targets)
    assert cal['converged']
    assert cal['rate'] == pytest.approx(0.068, abs=1e-6)
    assert cal['band_share'] == pytest.approx(0.04, abs=1e-5)
    assert cal['participation_ratio_resp'] == pytest.approx(0.05, abs=1e-5)
    r = H.readout(counts, 'saturating', cal['K'], floor=cal['floor'])
    st = H.low_end_stats(cal['c'] * r.to_numpy().T, 0.39 / np.sqrt(3))
    assert st['band_share'] == pytest.approx(0.04, abs=1e-5)
    # b held at 0: only K and c
    cal0 = H.calibrate_low_end(counts, noise_sd=0.39, fit_floor=False, **targets)
    assert cal0['floor'] == 0.0 and cal0['participation_ratio_resp'] == pytest.approx(0.05, abs=1e-5)
    # per-trial counts: identical trials give the deterministic calibration
    trials = np.repeat(counts.to_numpy()[None, None], 3, axis=1)
    cal_t = H.calibrate_low_end(trials, noise_sd=0.39, K0=10.0, **targets)
    assert cal_t['K'] == pytest.approx(cal['K'], rel=1e-6)
    assert cal_t['floor'] == pytest.approx(cal['floor'], abs=1e-6)


def test_cosine_terms_add_up():
    rng = np.random.default_rng(9)
    m = pd.DataFrame(rng.normal(0, 0.5, (300, len(ODORS))) + np.where(rng.random((300, len(ODORS))) < 0.1, 2.0, 0.0),
        columns=ODORS)
    t = H.cosine_terms(m)
    assert np.allclose(t[['both', 'one', 'neither']].sum(axis=1), t['cosine'])
    assert np.allclose(1 - t['cosine'].to_numpy(), H.pair_distances(m, 'cosine').to_numpy())
    pos = m.clip(lower=0)
    tz = H.cosine_terms(pos.where(pos > 1, 0.0))
    assert np.allclose(tz['one'], 0) and np.allclose(tz['neither'], 0)


@needs_shared
def test_data_readings():
    sd = H.kc_trial_noise_sd('TNT3c')
    assert 0.3 < sd < 0.5
    assert H.kc_noise_floor() == pytest.approx(2 * sd / np.sqrt(3))
    mats = H.yang_recording_matrices('TNT3c')
    nonneg = H.yang_recording_rdms_values('TNT3c', 'nonneg')
    manual = H.recording_rdms_from_matrices({k: m.clip(lower=0) for k, m in mats.items()})
    assert np.allclose(nonneg.to_numpy(), manual.loc[nonneg.index, nonneg.columns].to_numpy())
    fl = H.yang_recording_rdms_values('TNT3c', 'floor')
    f = H.kc_noise_floor()
    manual_f = H.recording_rdms_from_matrices({k: m.where(m > f, 0.0) for k, m in mats.items()})
    assert np.allclose(fl.to_numpy(), manual_f.loc[fl.index, fl.columns].to_numpy())
    for v in ('nonneg', 'floor'):
        e = H.yang_apl_effect(values=v)
        assert len(e) == 120 and np.isfinite(e).all()


def test_noise_corrected_min_reliability():
    rng = np.random.default_rng(10)
    base = _counts(10).to_numpy().copy()
    base[:, 0] = 0.0                                   # an odor with no signal
    trials = base[None] + rng.normal(0, 0.5, (3,) + base.shape)
    d = H.noise_corrected_pair_distances(trials, ODORS, min_reliability=0.1)
    with_odor0 = [p[0] == ODORS[0] or p[1] == ODORS[0] for p in d.index]
    assert d[with_odor0].isna().all() and d[[not w for w in with_odor0]].notna().all()
    d0 = H.noise_corrected_pair_distances(trials, ODORS)
    assert d0.notna().all()


def test_calibrate_low_end_unreachable_rate():
    """A model with fewer active entries than the target rate: c at its bound, no exception."""
    counts = _counts(11, n_units=1500, p_resp=0.01)
    cal = H.calibrate_low_end(counts, noise_sd=0.39, target_rate=0.2, target_band=0.05, target_pr=0.05)
    assert cal['c_at_bound'] and cal['floor_at_bound']
