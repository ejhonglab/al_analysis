#!/usr/bin/env python3
"""Tests for the repeat statistics of the KC-remapping harness (the flies' variability across the three
repeats, E81: scripts/claude_pair_remapping_experiments/kc_remapping_harness.py): single-trial distances by
kind of trial pair, the same-repeat gap, the noise-free distances from products of different repeats and the
parts of a response's variability, on synthetic recordings whose structure is known."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
H = pytest.importorskip('kc_remapping_harness')

ODORS = list(H.APL16_ODORS)


def _signal(seed=0, n_units=3000, n_odors=16, p_resp=0.12):
    """A KC-like noise-free signal (units x odors): sparse positive responses, odors sharing responders through
    a few latent groups, so that the pair distances span a range."""
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 4, n_odors)
    unit_pref = rng.random((n_units, 4)) ** 6
    p = np.clip(p_resp * unit_pref[:, groups] / unit_pref.mean(), 0, 1)   # odors of a group share their responders
    resp = rng.random((n_units, n_odors)) < p
    amp = 2.0 + rng.gamma(2.0, 0.6, (n_units, n_odors))
    return np.where(resp, amp, 0.0)


def _true_distances(m, odors=ODORS):
    return H.pair_distances(pd.DataFrame(m, columns=odors), 'cosine')


def _recording(m, *, seed=1, n_rep=3, sd_indep=0.6, sd_shared=0.0, sd_gain=0.0, return_parts=False):
    """Trials (n_rep, units, odors): x = g[r, o] * m + u[r, k] + e[r, k, o], with g a gain per presentation
    (1 + sd_gain * N(0, 1)), u a state shared by one unit's odors within a repeat, e independent noise; with
    `return_parts` also (g, u, e) as drawn."""
    rng = np.random.default_rng(seed)
    U, O = m.shape
    g = 1.0 + sd_gain * rng.standard_normal((n_rep, 1, O))
    u = sd_shared * rng.standard_normal((n_rep, U, 1))
    e = sd_indep * rng.standard_normal((n_rep, U, O))
    X = g * m[None] + u + e
    return (X, g, u, e) if return_parts else X


def _dev_var(x):
    """Per-trial variance of the deviations from the mean over repeats (axis 0), unbiased."""
    return ((x - x.mean(axis=0)) ** 2).sum(axis=0) / (x.shape[0] - 1)


def test_trial_values_and_cosines():
    X = np.array([[[0.5, 2.0], [3.0, -1.0], [0.0, 1.5]]])           # 1 repeat, 3 units, 2 odors
    assert np.array_equal(H.trial_values(X, 'thresholded', 1.0), [[[0, 2.0], [3.0, 0], [0, 1.5]]])
    assert np.array_equal(H.trial_values(X, 'binary', 1.0), [[[0, 1.0], [1.0, 0], [0, 1.0]]])
    C = H.trial_cosines(X)
    assert C.shape == (1, 2, 1, 2)
    a, b = X[0, :, 0], X[0, :, 1]
    assert np.isclose(C[0, 0, 0, 1], a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    Z = np.zeros((2, 3, 2))
    Z[0, 0, 0] = 1.0
    C = H.trial_cosines(Z)
    assert np.isnan(C[1, 0, 0, 0]) and np.isclose(C[0, 0, 0, 0], 1.0)


def test_distances_by_kind_layout_and_kinds():
    m = _signal(n_units=500)
    X = _recording(m, sd_indep=0.5)
    bk = H.distances_by_kind(X, ODORS)
    assert list(bk.columns) == list(H.REPEAT_KINDS)
    assert len(bk) == 120 + 16
    same = bk.index.get_level_values(0) == bk.index.get_level_values(1)
    assert bk.loc[same, 'same_repeat'].isna().all()
    assert bk.loc[~same].notna().all().all()
    # by hand, one pair: different repeats = mean of the six ordered (i != j) single-trial distances
    i_a, i_b = ODORS.index('2h'), ODORS.index('eb')
    cos = lambda x, y: x @ y / np.linalg.norm(x) / np.linalg.norm(y)
    diff = np.mean([1 - cos(X[i, :, i_a], X[j, :, i_b]) for i in range(3) for j in range(3) if i != j])
    adj = np.mean([1 - cos(X[i, :, i_a], X[j, :, i_b]) for i in range(3) for j in range(3) if abs(i - j) == 1])
    same_r = np.mean([1 - cos(X[i, :, i_a], X[i, :, i_b]) for i in range(3)])
    assert np.isclose(bk.loc[('2h', 'eb'), 'different_repeat'], diff)
    assert np.isclose(bk.loc[('2h', 'eb'), 'adjacent'], adj)
    assert np.isclose(bk.loc[('2h', 'eb'), 'same_repeat'], same_r)
    assert np.isclose(H.same_repeat_gap(bk, [('eb', '2h')]), diff - same_r)
    # two repeats: no first-against-third
    bk2 = H.distances_by_kind(X[:2], ODORS)
    assert bk2['first_third'].isna().all()


def test_independent_noise_only_agrees_with_correction_in_use():
    """With noise independent between odors and repeats, the same-repeat gap is about 0, and the noise-free
    estimate, the correction in use and the true distances agree within sampling error."""
    m = _signal(seed=2)
    true = _true_distances(m)
    X = _recording(m, seed=3, sd_indep=1.0)
    tab = H.repeat_pair_table(X, ODORS)
    gap = H.same_repeat_gap(H.distances_by_kind(X, ODORS))
    assert abs(gap) < 0.005
    # the noise inflates the trial-mean distances well beyond the sampling error
    assert (tab['trial_mean'] - true).mean() > 0.025
    for col in ('noise_free', 'corrected', 'noise_free_adjacent', 'noise_free_first_third'):
        err = tab[col] - true.reindex(tab.index)
        assert abs(err.mean()) < 0.01, col
        assert np.sqrt((err ** 2).mean()) < 0.03, col
    assert abs((tab['noise_free'] - tab['corrected']).mean()) < 0.01
    assert np.sqrt(((tab['noise_free'] - tab['corrected']) ** 2).mean()) < 0.03
    nf = H.noise_free_pair_distances(X, ODORS)
    assert np.allclose(nf.to_numpy(), tab['noise_free'].to_numpy())


def test_shared_state_only_new_estimate_returns_true_distances():
    """With a state shared by one unit's odors within a repeat, two odors of the same repeat are closer than
    two odors of different repeats (a positive gap), the correction in use is biased low, and the noise-free
    estimate from products of different repeats still returns the true distances."""
    m = _signal(seed=4)
    true = _true_distances(m)
    X = _recording(m, seed=5, sd_indep=0.5, sd_shared=0.5)
    tab = H.repeat_pair_table(X, ODORS)
    gap = H.same_repeat_gap(H.distances_by_kind(X, ODORS))
    assert gap > 0.02
    err_nf = tab['noise_free'] - true.reindex(tab.index)
    err_c = tab['corrected'] - true.reindex(tab.index)
    assert abs(err_nf.mean()) < 0.01
    assert np.sqrt((err_nf ** 2).mean()) < 0.03
    assert err_c.mean() < -0.03                                  # the correction makes the odors too alike
    # a gain per presentation, which a cosine does not see, changes neither estimate's agreement
    Xg = _recording(m, seed=5, sd_indep=0.5, sd_shared=0.5, sd_gain=0.2)
    err_g = H.noise_free_pair_distances(Xg, ODORS) - true
    assert abs(err_g.mean()) < 0.01


def test_drift_cancels_in_noise_free_ratio():
    """A drift of each odor's response pattern over the repeats (a fixed step per repeat, its own direction for
    each odor) acts on the same-odor and the different-odor products alike: the noise-free estimates from
    adjacent repeats and from the first against the third agree."""
    rng = np.random.default_rng(6)
    m = _signal(seed=6)
    U, O = m.shape
    # each odor's response pattern changes by a fixed step per repeat (a random direction per odor)
    step = 0.15 * np.where(m > 0, 1.0, 0.0) * rng.standard_normal((U, O))
    X = np.stack([m + r * step for r in (-1, 0, 1)]) + 0.5 * rng.standard_normal((3, U, O))
    tab = H.repeat_pair_table(X, ODORS)
    assert abs((tab['noise_free_adjacent'] - tab['noise_free_first_third']).mean()) < 0.02


def test_variance_parts_return_known_shares():
    """A synthetic recording with a gain per presentation, a state shared by a unit's odors within a repeat and
    independent noise, of known sizes: the parts come back."""
    m = _signal(seed=7, n_units=6000)
    sd_g, sd_u, sd_e = 0.15, 0.3, 0.5
    X, g, u, e = _recording(m, seed=8, n_rep=3, sd_indep=sd_e, sd_shared=sd_u, sd_gain=sd_g, return_parts=True)
    vp = H.repeat_variance_parts(X, ODORS, thresh=1.0, n_amp_bins=3)
    assert {'all', 'responders', 'silent', 'responders_amp1', 'responders_amp3'} <= set(vp.index)
    # the parts as drawn (48 gains, 18000 shared states: the realized sizes differ from sd_g, sd_u by sampling)
    v_gain = _dev_var(g * m[None])                               # per entry
    v_u = float(_dev_var(u).mean())
    v_e = float(_dev_var(e).mean())
    s = vp.loc['silent']
    assert np.isclose(s.var_shared, v_u, rtol=0.1)
    assert np.isclose(s.var_total, v_u + v_e, rtol=0.05)
    assert s.share_gain < 0.02
    assert np.isclose(s.corr_shared, v_u / (v_u + v_e), rtol=0.1)
    r = vp.loc['responders']
    resp = X.mean(axis=0) > 1.0
    assert np.isclose(r.var_gain, v_gain[resp].mean(), rtol=0.1)
    assert np.isclose(r.var_shared, v_u, rtol=0.15)
    assert np.isclose(r.var_total, v_gain[resp].mean() + v_u + v_e, rtol=0.05)
    assert np.isclose(r.share_gain + r.share_shared + r.share_rest, 1.0)
    # the gain part scales with the amplitude, the shared part does not
    assert vp.loc['responders_amp3', 'var_gain'] > 2 * vp.loc['responders_amp1', 'var_gain']
    assert np.isclose(vp.loc['responders_amp3', 'var_shared'], vp.loc['responders_amp1', 'var_shared'], rtol=0.3)
    # without a shared state, the shared part is about 0
    vp0 = H.repeat_variance_parts(_recording(m, seed=9, sd_indep=sd_e), ODORS, thresh=1.0)
    assert abs(vp0.loc['silent', 'corr_shared']) < 0.02
    assert abs(vp0.loc['responders', 'corr_shared']) < 0.05


def test_coresponder_deviation_corr():
    m = _signal(seed=10, n_units=6000)
    sd_u, sd_e = 0.4, 0.4
    X = _recording(m, seed=11, sd_indep=sd_e, sd_shared=sd_u)
    c = H.coresponder_deviation_corr(X, ODORS, thresh=1.0)
    assert len(c) == 120
    assert np.isclose(np.nanmedian(c), sd_u ** 2 / (sd_u ** 2 + sd_e ** 2), atol=0.08)
    c0 = H.coresponder_deviation_corr(_recording(m, seed=12, sd_indep=sd_e), ODORS, thresh=1.0)
    assert abs(np.nanmedian(c0)) < 0.08
    # a gain common to the whole repeat correlates co-responders' deviations; removing the gain part removes it
    rng = np.random.default_rng(13)
    g = (1 + 0.3 * rng.standard_normal(3))[:, None, None]
    Xg = g * m[None] + sd_e * rng.standard_normal((3,) + m.shape)
    assert np.nanmedian(H.coresponder_deviation_corr(Xg, ODORS, thresh=1.0)) > 0.3
    assert abs(np.nanmedian(H.coresponder_deviation_corr(Xg, ODORS, thresh=1.0, remove_gain=True))) < 0.1


def test_repeat_response_stats():
    X = np.zeros((3, 10, 2))
    X[0, :4, 0] = 2.0
    X[2, :1, 1] = 3.0
    st = H.repeat_response_stats(X, ['2h', 'eb'], thresh=1.0)
    assert st.loc[(0, '2h'), 'rate'] == 0.4 and st.loc[(0, '2h'), 'amplitude'] == 2.0
    assert st.loc[(2, 'eb'), 'n_resp'] == 1 and np.isnan(st.loc[(1, 'eb'), 'amplitude'])
