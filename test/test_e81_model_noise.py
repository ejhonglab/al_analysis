#!/usr/bin/env python3
"""Tests for E81 parts D and E (scripts/claude_pair_remapping_experiments/E81_model_noise.py, the model's variability with
the flies' shares): the repeat statistics against brute-force loops, the noise correction in use and the harness's
functions; the interpolation of the jitter to a target gap on synthetic curves; and the calibration of the independent
part on synthetic counts (it returns the target same-odor distance and gap); the search step of part D's check
(`pres_next`). Synthetic data only, except the flies' side test, which needs the shared directory's KC data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
H = pytest.importorskip('kc_remapping_harness')
M = pytest.importorskip('E81_model_noise')

ODORS = M.ODORS


def _counts(seed=0, n_kc=600, p=0.08):
    """KC-like spike counts (KCs x 16 odors): sparse, odors sharing responders through four latent groups."""
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 4, len(ODORS))
    pref = rng.random((n_kc, 4)) ** 5
    pr = np.clip(p * pref[:, groups] / pref.mean(), 0, 1)
    resp = rng.random((n_kc, len(ODORS))) < pr
    return np.where(resp, 2 + rng.poisson(4.0, (n_kc, len(ODORS))), 0).astype(float)


def _cos(a, b):
    return a @ b / np.linalg.norm(a) / np.linalg.norm(b)


def test_kind_arrays_brute_force():
    rng = np.random.default_rng(3)
    Y = rng.gamma(1.0, 1.0, (2, 3, 50, len(ODORS))) * (rng.random((2, 3, 50, len(ODORS))) < 0.4)
    k = M.kind_arrays(Y)
    r, T = 1, 3
    a, b = M.PAIRS.index(('2h', 'pa')), None
    ia, ib = ODORS.index('2h'), ODORS.index('pa')
    same_rep = np.mean([_cos(Y[r, t, :, ia], Y[r, t, :, ib]) for t in range(T)])
    diff = np.mean([_cos(Y[r, i, :, ia], Y[r, j, :, ib]) for i in range(T) for j in range(T) if i != j])
    adj = np.mean([_cos(Y[r, i, :, ia], Y[r, j, :, ib]) for i in range(T) for j in range(T) if abs(i - j) == 1])
    ft = np.mean([_cos(Y[r, i, :, ia], Y[r, j, :, ib]) for i in range(T) for j in range(T) if abs(i - j) == 2])
    caa = np.mean([_cos(Y[r, i, :, ia], Y[r, j, :, ia]) for i in range(T) for j in range(T) if i != j])
    cbb = np.mean([_cos(Y[r, i, :, ib], Y[r, j, :, ib]) for i in range(T) for j in range(T) if i != j])
    assert np.isclose(k['same_rep'][r, a], 1 - same_rep)
    assert np.isclose(k['diff_rep'][r, a], 1 - diff)
    assert np.isclose(k['adjacent'][r, a], 1 - adj)
    assert np.isclose(k['first_third'][r, a], 1 - ft)
    assert np.isclose(k['same_odor'][r, ia], 1 - caa)
    assert np.isclose(k['noise_free'][r, a], 1 - diff / np.sqrt(caa * cbb))
    xb = Y[r].mean(axis=0)
    assert np.isclose(k['trial_mean'][r, a], 1 - _cos(xb[:, ia], xb[:, ib]))
    # the noise correction in use, as noise_floor_trial_distances.analyse computes it for the data
    NF = pytest.importorskip('noise_floor_trial_distances')
    _, pr = NF.analyse({o: Y[r, :, :, j] for j, o in enumerate(ODORS)}, 'graded')
    cor = H.canonical_pairs(pr['corrected'])
    got = pd.Series(k['corrected'][r], index=M.PAIR_INDEX)
    a_, b_ = H.align_pairs(got, cor)
    assert np.allclose(a_, b_)


def test_gap_same_as_kind_arrays():
    """The calibration's light statistics are `kind_arrays`'s, also with all-zero vectors (binarized trials)."""
    rng = np.random.default_rng(7)
    Y = rng.gamma(1.0, 1.0, (4, 3, 60, len(ODORS))) * (rng.random((4, 3, 60, len(ODORS))) < 0.3)
    Y[1, 2, :, 3] = 0.0
    k = M.kind_arrays(Y)
    gap = np.nanmean(np.nanmean(k['diff_rep'] - k['same_rep'], axis=1))
    same = np.nanmean(np.nanmean(k['same_odor'], axis=1))
    g, s = M.gap_same(Y)
    assert np.isclose(g, gap) and np.isclose(s, same)


def test_kind_arrays_against_harness():
    """The numpy statistics against the harness's distances_by_kind / same_repeat_gap / noise_free_pair_distances
    (skipped until their bodies exist)."""
    rng = np.random.default_rng(5)
    X = rng.gamma(1.0, 1.0, (3, 80, len(ODORS))) * (rng.random((3, 80, len(ODORS))) < 0.4)
    try:
        bk = H.distances_by_kind(X, ODORS)
        nf = H.noise_free_pair_distances(X, ODORS)
    except NotImplementedError:
        pytest.skip('the harness repeat statistics are stubs')
    k = M.kind_arrays(X[None])
    two = bk[bk.index.get_level_values(0) != bk.index.get_level_values(1)]
    for col, name in (('same_repeat', 'same_rep'), ('different_repeat', 'diff_rep'), ('adjacent', 'adjacent'), ('first_third', 'first_third')):
        a, b = H.align_pairs(H.canonical_pairs(two[col]), pd.Series(k[name][0], index=M.PAIR_INDEX))
        assert np.allclose(a, b), col
    same = bk[bk.index.get_level_values(0) == bk.index.get_level_values(1)]['different_repeat']
    assert np.allclose(same.to_numpy(), k['same_odor'][0][[ODORS.index(o) for o in same.index.get_level_values(0)]])
    a, b = H.align_pairs(H.canonical_pairs(nf), pd.Series(k['noise_free'][0], index=M.PAIR_INDEX))
    assert np.allclose(a, b)
    assert np.isclose(H.same_repeat_gap(bk), M.gap_same(X[None])[0])


def test_interp_sigma_quadratic():
    """On a gap that grows as sigma^2 (the saved calibration trials' shape), linear interpolation in sigma^2 recovers the
    sigma of any target exactly, between any two levels, the deterministic level 0 included."""
    sig = np.array([0.0, 0.05, 0.1, 0.15])
    a = 6.4
    gaps = a * sig ** 2
    for target in (0.004, 0.025, 0.07, 0.1):
        ip = M.interp_sigma(sig, gaps, target)
        assert ip['bracketed']
        assert np.isclose(ip['sigma'], np.sqrt(target / a))
        lo, hi, w = ip['lo'], ip['hi'], ip['w']
        assert np.isclose((1 - w) * gaps[lo] + w * gaps[hi], target)
        assert sig[lo] <= ip['sigma'] <= sig[hi]
    ip = M.interp_sigma(sig, gaps, 0.5)
    assert not ip['bracketed'] and ip['lo'] == ip['hi'] == 3
    # only two levels, 0 and one jitter: still exact for a quadratic curve
    ip = M.interp_sigma([0.0, 0.126], [0.0, a * 0.126 ** 2], 0.025)
    assert np.isclose(ip['sigma'], np.sqrt(0.025 / a))
    # a curve that is not monotone: the first crossing
    ip = M.interp_sigma([0.0, 0.1, 0.2, 0.3], [0.0, 0.05, 0.02, 0.08], 0.03)
    assert ip['lo'] == 0 and ip['hi'] == 1


def _synthetic_mix(det, levels, seed=0, gsd=0.3):
    """A ModelMix on synthetic counts: `levels` {sigma: counts (R, T, KCs, odors)} beside the deterministic `det`, the
    conversion fixed at 1 and the silent entries' noise `gsd` in both conditions."""
    mm = object.__new__(M.ModelMix)
    mm.label = mm.name = 'synthetic'
    R, T = M.N_REC, M.N_TRIALS
    mm.det = {c: pd.DataFrame(det, columns=ODORS) for c in M.CONDS}
    sig = [0.0] + sorted(levels)
    mm.levels = [dict(sigma=s, source='synthetic') for s in sig]
    mm.sigmas = np.array(sig)
    mm.counts = [{c: np.broadcast_to(det, (R, T) + det.shape).copy() for c in M.CONDS}]
    mm.counts += [{c: levels[s] for c in M.CONDS} for s in sorted(levels)]
    mm.rate_spk = [{c: 0.0 for c in M.CONDS} for _ in sig]
    rng = np.random.default_rng(seed)
    mm.eps = {c: rng.standard_normal((R, T) + det.shape) for c in M.CONDS}
    mm.eta = {c: rng.standard_normal((R, T) + det.shape) for c in M.CONDS}
    mm.K = {u: 5.0 for u in M.UNITS}
    mm.gsd = {u: {c: gsd for c in M.CONDS} for u in M.UNITS}
    mm._c = {(ro, u, i): 1.0 for ro in M.READOUTS for u in M.UNITS for i in range(len(sig))}
    mm._cache = {}
    mm.matched = {}
    return mm


def _shared_level(det, sigma, seed):
    """Trials whose variability is all shared by a KC's odors within a trial: counts x (1 + sigma xi[r, t, KC])."""
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((M.N_REC, M.N_TRIALS, det.shape[0], 1))
    return det[None, None] * np.clip(1 + 4 * sigma * xi, 0, None)


def test_mixing_returns_target_same_odor_distance():
    """With the deterministic level alone (no shared part), the spread of the lognormal factor is set so that the
    same-odor distance between two trials is the target, and the result is the statistic at that spread."""
    det = _counts()
    mm = _synthetic_mix(det, {})
    target = 0.40
    mm.flies = {u: {g: dict(gap=0.0, same_odor=target) for g in ('TNT3c', 'TNT')} for u in M.UNITS}
    cal = mm.calibrate('raw', 'z', 'on')
    assert 0 < cal['s'] < M.S_MAX
    assert set(cal['flags']) <= {'gap_not_bracketed'}
    fin = mm.final(cal)
    assert abs(fin['same_odor'] - target) < 1e-4
    # independently: the same-odor distance of freshly mixed trials at that spread
    assert abs(M.gap_same(mm.Y(0, 'on', 'raw', 'z', cal['s']))[1] - target) < 1e-4
    # a larger target needs a larger spread
    mm2 = _synthetic_mix(det, {})
    mm2.flies = {u: {g: dict(gap=0.0, same_odor=0.5) for g in ('TNT3c', 'TNT')} for u in M.UNITS}
    assert mm2.calibrate('raw', 'z', 'on')['s'] > cal['s']


def test_joint_calibration_matches_gap_and_same_odor():
    """With levels of shared variability (a KC's odors moving together within a trial), the joint calibration reaches
    both targets: the interpolated gap and the same-odor distance at the calibrated jitter and spread."""
    det = _counts(seed=1)
    levels = {s: _shared_level(det, s, seed=int(s * 1000)) for s in (0.05, 0.1, 0.15)}
    mm = _synthetic_mix(det, levels, seed=2)
    g = [M.gap_same(mm.Y(i, 'on', 'raw', 'z', 0.0))[0] for i in range(4)]
    assert g[0] < g[1] < g[2] < g[3], g
    g_t = 0.5 * (g[1] + g[2])
    d_t = max(M.gap_same(mm.Y(i, 'on', 'raw', 'z', 0.0))[1] for i in range(4)) + 0.1
    mm.flies = {u: {gg: dict(gap=g_t, same_odor=d_t) for gg in ('TNT3c', 'TNT')} for u in M.UNITS}
    cal = mm.calibrate('raw', 'z', 'on')
    assert cal['bracketed'] and cal['converged'], cal['flags']
    fin = mm.final(cal)
    assert abs(fin['gap'] - g_t) < 1e-3
    assert abs(fin['same_odor'] - d_t) < 1e-3
    # the independent part lowers the gap at a given jitter, so the joint jitter is above the gap-only one
    assert cal['sigma'] >= cal['sigma_gap_only'] - 1e-9


def test_six_pairs_as_e78():
    src = (EXP_DIR / 'E78_claw_knee.py').read_text()
    assert f"SIX = {M.SIX!r}" in src


def test_flies_side_reproduces_registry():
    """The flies' same-odor distance (z) is the registry's single-trial floor and the silent entries' sd the harness's
    kc_trial_noise_sd (needs the shared directory)."""
    try:
        H.data_target('ctrl_single_trial_floor')
        recs = M.flies_trials('TNT3c', 'z')
    except (FileNotFoundError, OSError, KeyError):
        pytest.skip('the shared directory is not available')
    k = [M.kind_arrays(X[None]) for X in recs.values()]
    same = np.mean([np.nanmean(x['same_odor']) for x in k])
    assert np.isclose(same, H.data_target('ctrl_single_trial_floor'), atol=1e-9)
    assert np.isclose(M.silent_sd(recs, 1.0), H.kc_trial_noise_sd('TNT3c'), atol=1e-12)


def test_presentation_search_step():
    # the same-odor distance linear in sigma_p^2: the secant through two points lands on the target
    d = lambda sp: 0.2 + 18.0 * sp ** 2
    target = 0.41
    hist = [dict(sigma_p=0.08, same=d(0.08)), dict(sigma_p=0.13, same=d(0.13))]
    assert np.isclose(M.pres_next(hist, target, slope0=5.0), np.sqrt((target - 0.2) / 18.0))
    # the first step uses the given slope, the step kept within a factor of 2
    assert np.isclose(M.pres_next(hist[:1], target, slope0=18.0), np.sqrt((target - 0.2) / 18.0))
    assert np.isclose(M.pres_next(hist[:1], 10.0, slope0=18.0), 0.16)


def test_presentation_scoring_and_table(tmp_path):
    """pres_score and the table section on synthetic pseudo-recordings (no model): every readout, unit, kind and condition
    is scored, and the section renders from the written tables."""
    from types import SimpleNamespace
    flies = {u: M.flies_side(u) for u in M.UNITS}
    rng = np.random.default_rng(5)
    base = _counts(seed=2, n_kc=400)
    det = {'on': base, 'acute_off': base * 1.6 + (rng.random(base.shape) < 0.05) * 3}

    def trials(m, n):
        return np.stack([np.where(rng.random(m.shape) < 0.85, m, 0) * rng.lognormal(0, 0.3, m.shape) for _ in range(n)])
    X = {c: trials(det[c], 9).reshape(3, 3, *base.shape) for c in M.CONDS}
    read = lambda x, ro, u: x if ro == 'raw' else x / (x + 5.0)
    mm = SimpleNamespace(read=read, det_pairs=lambda ro, u: {c: H.canonical_pairs(H.pair_distances(
        pd.DataFrame(read(det[c], ro, u), columns=ODORS), 'cosine')) for c in M.CONDS})
    run = SimpleNamespace(name='stub', label='C2', mm=mm, out=tmp_path, flies=flies, conv=lambda on, ro, u: 0.6,
        gsd={u: {c: float(flies[u][M.GENOTYPE[c]]['silent_sd']) for c in M.CONDS} for u in M.UNITS})
    best = {c: dict(sigma_s=0.07, sigma_p=0.09, fixed_thr=110.0, fixed_thr_ratio=1.08, converged=True, same=0.44,
        arrs={'on_spk': trials(det['on'], 6)}) for c in M.CONDS}
    sc, ck = M.pres_score(run, best, X, {'on': 0.07, 'acute_off': 0.15}, M._flies_pairs(flies))
    sc, ck = pd.DataFrame(sc), pd.DataFrame(ck)
    assert len(sc) == len(M.READOUTS) * len(M.UNITS) * len(M.PRES_KINDS) and len(ck) == len(M.READOUTS) * len(M.UNITS) * 2
    assert sc[['loss', 'pattern', 'mean_effect', 'rho', 'six_effect', 'rate_trial_on', 'rate_trial_off']].notna().all().all()
    assert ck[['same_odor', 'gap', 'bin_same', 'bin_gap', 'flies_bin_gap', 'inflation']].notna().all().all()
    assert np.isclose(ck['shared_share'].iloc[0], 0.07 ** 2 / (0.07 ** 2 + 0.09 ** 2))
    sc.to_csv(tmp_path / 'per_presentation.csv', index=False)
    ck.to_csv(tmp_path / 'per_presentation_checks.csv', index=False)
    text = '\n'.join(M._presentation_section(tmp_path))
    assert "## Part D's check" in text and '| C2 | APL off |' in text and 'three-trial means, per-presentation jitter' in text
