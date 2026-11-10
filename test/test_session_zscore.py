#!/usr/bin/env python3
"""Tests for the per-KC session z-score reading in the KC-remapping harness (E73:
scripts/claude_pair_remapping_experiments/kc_remapping_harness.py `session_zscore`, `zcell_pair_distances`,
`zcell_stats`, `calibrate_session_zscore`) and for the edge statistics of E73_per_kc_normalization.py:
synthetic cells z-scored over their session sit under an edge at 1 / q and the estimators recover q; a
fixed divisor for all KCs reproduces the present readings; the calibration reaches its targets."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
H = pytest.importorskip('kc_remapping_harness')

ODORS = list(H.APL16_ODORS)


def _counts(seed=0, n_units=800, p_resp=0.07):
    """Spike-count-like matrix over the 16 APL16 odors: mostly 0, some 1, responders with geometric counts."""
    rng = np.random.default_rng(seed)
    u = rng.random((n_units, len(ODORS)))
    n = np.where(u < p_resp, 2 + rng.geometric(0.12, u.shape), np.where(u < p_resp + 0.05, 1, 0))
    return pd.DataFrame(n.astype(float), columns=ODORS)


def test_session_zscore_formula():
    c = _counts()
    q, noise = 0.2, 1.5
    z = H.session_zscore(c, q=q, noise=noise, noise_ratio=0.4)
    x = c.to_numpy()
    n_stim = len(ODORS) + 1
    mu = q * x.sum(axis=1) / n_stim
    S = np.sqrt(q * (x ** 2).sum(axis=1) / n_stim - mu ** 2 + noise ** 2)
    assert np.allclose(z['center'][0], x / S[:, None])
    assert np.allclose(z['trial_sd'][0], 0.4 * noise / S)
    assert np.allclose(z['blank'][0], 0.0)
    zc = H.session_zscore(c, q=q, noise=noise, noise_ratio=0.4, center=True)
    assert np.allclose(zc['center'][0], (x - mu[:, None]) / S[:, None])
    assert np.allclose(zc['blank'][0], -mu / S)
    # the expected mean square: the trials' squares plus the measurement noise, the blank included
    assert np.allclose(z['mean_sq'][0], (z['center'][0] ** 2).sum(axis=1) / n_stim + z['trial_sd'][0] ** 2)
    # centered and noise-free: under the edge 1 / q
    z0 = H.session_zscore(c, q=q, noise=1e-9, noise_ratio=0.0, center=True)
    assert z0['mean_sq'][0].max() <= 1 / q + 1e-6


def test_noise_free_edge_at_inv_q():
    """A noise-free KC's session z puts its mean of squared trial values at 1 / q (minus q mean^2 / (1 - q) for
    the centered form; 1 / (q (1 - q p)) for a KC responding to a share p of the trials with the baseline
    subtracted)."""
    q = 0.17
    x = np.zeros((3, len(ODORS)))
    x[0, 0] = 10.0                  # one odor
    x[1, :8] = 5.0                  # eight odors
    x[2, :] = np.arange(1, 17)      # every odor, graded
    c = pd.DataFrame(x, columns=ODORS)
    zc = H.session_zscore(c, q=q, noise=0.0, center=True)
    n_stim = len(ODORS) + 1
    mean_r = (zc['center'][0].sum(axis=1) + zc['blank'][0]) / n_stim
    assert np.allclose(q * zc['mean_sq'][0] + (1 - q) * (q * mean_r / (1 - q)) ** 2, 1.0)
    z = H.session_zscore(c, q=q, noise=0.0)
    p = np.array([1, 8, 16]) / n_stim
    assert np.isclose(z['mean_sq'][0][0], 1 / (q * (1 - q * p[0])))
    assert np.isclose(z['mean_sq'][0][1], 1 / (q * (1 - q * p[1])))


def test_fixed_divisor_reproduces_present_readings():
    c, c2 = _counts(1), _counts(2)
    for amp, kind, K in (('linear', 'raw', None), ('saturating', 'saturating', 6.0)):
        present = H.readout_pair_distances(c, kind, K)
        fixed = H.zcell_pair_distances(H.session_zscore(c, q=0.2, noise=0.0, amp=amp, K=K, divisor=2.5))
        assert np.allclose(present.to_numpy(), fixed.reindex(present.index).to_numpy(), atol=1e-12)
        l_p = H.apl_effect_stats(present, H.readout_pair_distances(c2, kind, K), data=present * 0 + 0.1)['apl_effect_rmse']
        l_f = H.apl_effect_stats(fixed, H.zcell_pair_distances(H.session_zscore(c2, q=0.2, noise=0.0, amp=amp, K=K, divisor=2.5)),
            data=present * 0 + 0.1)['apl_effect_rmse']
        assert np.isclose(l_p, l_f)
        # with measurement noise: the closed form of expected_pair_distances_noisy
        sig = H.readout(c, kind, K)
        cz, s = 0.7, 0.385
        noisy = H.expected_pair_distances_noisy(cz * sig, s, 3)
        fz = H.zcell_pair_distances(H.session_zscore(c, q=1.0, noise=s / cz, amp=amp, K=K, noise_ratio=1.0, divisor=1 / cz))
        assert np.allclose(noisy.to_numpy(), fz.reindex(noisy.index).to_numpy(), atol=1e-12)


def test_readings_through_noise_match_sampling():
    """The closed-form readings (graded, thresholded, binary) and the response rate equal the mean over many
    sampled pseudo-recordings (ratio of expectations: close for many units)."""
    c = _counts(3, n_units=3000)
    z = H.session_zscore(c, q=0.2, noise=1.0, noise_ratio=0.5)
    rng = np.random.default_rng(0)
    sd = (z['trial_sd'][0] / np.sqrt(3))[:, None]
    n_draw = 60
    acc = {r: [] for r in H.ZCELL_READINGS}
    rates = []
    for _ in range(n_draw):
        m = z['center'][0] + rng.normal(0, 1, z['center'][0].shape) * sd
        rates.append((m > 1).mean())
        for r, x in (('graded', m), ('thresholded', np.where(m > 1, m, 0.0)), ('binary', (m > 1).astype(float))):
            acc[r].append(H.pair_distances(pd.DataFrame(x, columns=ODORS), 'cosine').to_numpy())
    st = H.zcell_stats(z)
    assert abs(st['rate'] - np.mean(rates)) < 2e-3
    for r in H.ZCELL_READINGS:
        closed = H.zcell_pair_distances(z, r).to_numpy()
        assert np.nanmax(np.abs(closed - np.nanmean(acc[r], axis=0))) < 0.02, r


def test_calibration_reaches_targets():
    c = _counts(4, n_units=1500, p_resp=0.09)
    for amp in H.ZCELL_MAPS:
        cal = H.calibrate_session_zscore(c, q=0.18, amp=amp, target_rate=0.068, target_amp=2.0 if amp == 'saturating' else None,
            target_silent_sd=0.385)
        assert cal['converged'] and not cal['noise_at_bound']
        z = H.session_zscore(c, q=0.18, noise=cal['noise'], amp=amp, K=cal['K'], noise_ratio=cal['noise_ratio'])
        st = H.zcell_stats(z)
        assert abs(st['rate'] - 0.068) < 1e-6
        assert abs(st['nonresp_trial_sd'] - 0.385) < 1e-3
        if amp == 'saturating':
            assert not cal['K_at_bound']
            assert abs(st['resp_amp_mean'] - 2.0) < 1e-4


def test_trials_input():
    """Per-trial counts (recordings, trials, units, odors): identical trials give the 2-D reading."""
    c = _counts(5)
    arr = np.broadcast_to(c.to_numpy(), (2, 3) + c.shape)
    z2 = H.session_zscore(c, q=0.2, noise=1.0)
    z4 = H.session_zscore(arr, q=0.2, noise=1.0)
    assert np.allclose(z4['center'][0], z2['center'][0]) and np.allclose(z4['center'][1], z2['center'][0])
    d2, d4 = H.zcell_pair_distances(z2), H.zcell_pair_distances(z4)
    assert np.allclose(d2.to_numpy(), d4.to_numpy())


def test_synthetic_cells_edge_and_q():
    """Synthetic cells z-scored over the whole recording sit under the edge 1 / q and the three estimators
    recover q; against a pre-odor baseline there is no edge and the estimators give about 0."""
    E = pytest.importorskip('E73_per_kc_normalization')
    stim = np.repeat(ODORS + ['pfo @ 0'], 3)
    rng = np.random.default_rng(0)
    breadth = rng.choice([0, 1, 2, 3, 5, 8, 12], size=1500, p=[0.4, 0.2, 0.15, 0.1, 0.08, 0.05, 0.02])
    q = 0.17
    X = E.synthetic_recording(1500, stim, q=q, how='session', seed=1, breadth=breadth, gain=3.0)
    s, _ = E.edge_stats(X, stim)
    assert s['ms_max'] < 1.25 / q
    for est in ('q_edge', 'q_single', 'q_noise'):
        assert abs(s[est] - q) < 0.05, (est, s[est])
    Xb = E.synthetic_recording(1500, stim, q=q, how='baseline', seed=1, breadth=breadth, gain=3.0)
    sb, _ = E.edge_stats(Xb, stim)
    assert sb['ms_max'] > 2 / q
    assert abs(sb['q_noise']) < 0.05


def test_drift_sets_units_only():
    """A drift term changes no reading once the noise is calibrated (it sets the units: S / noise is the
    whole-trace SD over the baseline-noise SD, sqrt(1 + drift^2) for a silent KC)."""
    c = _counts(6, n_units=1200, p_resp=0.09)
    kw = dict(q=0.128, amp='saturating', target_rate=0.068, target_amp=2.0, target_silent_sd=0.385)
    c0 = H.calibrate_session_zscore(c, **kw)
    c1 = H.calibrate_session_zscore(c, drift=1.0, **kw)
    z0 = H.session_zscore(c, q=0.128, noise=c0['noise'], amp='saturating', K=c0['K'], noise_ratio=c0['noise_ratio'])
    z1 = H.session_zscore(c, q=0.128, noise=c1['noise'], amp='saturating', K=c1['K'], noise_ratio=c1['noise_ratio'], drift=1.0)
    assert np.isclose(c0['K'], c1['K'], rtol=1e-5)
    assert np.allclose(z0['center'], z1['center'], atol=1e-6) and np.allclose(z0['trial_sd'], z1['trial_sd'], atol=1e-6)
    silent = c.to_numpy().sum(axis=1) == 0
    assert np.allclose(z1['scale_ratio'][0][silent], np.sqrt(2.0))


def test_trace_simulation_matches_closed_form():
    """A KC trace with Yang's trial structure (baseline, response window, inter-trial time), z-scored over the
    whole trace, with the response the window mean minus the baseline mean, reads as the closed form does."""
    E = pytest.importorskip('E73_per_kc_normalization')
    c = _counts(7, n_units=3000, p_resp=0.08)
    noise, drift = 0.6, 1.0
    X, stim, T = E.trace_reading(c, noise=noise, drift=drift, amp='linear', K=None, seed=3)
    X0, _, _ = E.trace_reading(c * 0, noise=noise, drift=drift, amp='linear', K=None, seed=3)
    nr = float(X0.std() * np.sqrt(1 + drift ** 2))
    z = H.session_zscore(c, q=51 * 18 / T, noise=noise, amp='linear', noise_ratio=nr, drift=drift)
    names, tm, _, _ = E._per_stimulus(X, stim)
    tm = tm[:, [names.index(o) for o in c.columns]]
    st = H.zcell_stats(z)
    assert abs((tm > 1).mean() - st['rate']) < 0.004
    assert abs(tm[tm > 1].mean() - st['resp_amp_mean']) < 0.05
    d_sim = H.canonical_pairs(H.pair_distances(pd.DataFrame(tm, columns=c.columns), 'cosine'))
    d_cf = H.zcell_pair_distances(z)
    assert float((d_sim - d_cf.reindex(d_sim.index)).abs().mean()) < 0.01
