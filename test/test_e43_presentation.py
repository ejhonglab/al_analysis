#!/usr/bin/env python3
"""Tests for the per-presentation threshold jitter of E43_model_trial_noise.py (`--presentation-jitter`, E81 part D's
check): the thresholds with SD 0 are part B's, the shared part of the variance is sigma^2 / (sigma^2 + sigma_p^2), the
presentation draws are a stream of their own, and a trial assembled from one run per odor (in this process and in forked
workers) is the KC layer at each presentation's thresholds. The KC layer is a stub (no olfsysm model is built); the
forked-worker test needs olfsysm imported only for its OpenMP runtime."""

import sys
from pathlib import Path

import numpy as np
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
E43 = pytest.importorskip('E43_model_trial_noise')

ODORS = E43.ODORS


class StubModel(E43.ExcitabilityModel):
    """ExcitabilityModel without olfsysm: a KC layer whose spike count is floor((drive - thr) / 2), at least 0, the
    drive 1.5 times larger with APL off; KCs independent, so a run of one odor is that odor's column."""

    def __init__(self, n_kcs: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.name = 'ref40'
        self.n_kcs = self.draw_size = n_kcs
        self.spont_in = rng.random((n_kcs, 1))
        self.kc_thr_factor = None
        self.f0 = 10.0
        self.pn_idx = list(range(len(ODORS)))
        self.pn_summary = dict(pn_win=np.zeros((len(ODORS), 4), np.float32), pn_pre_sd=np.zeros((len(ODORS), 4), np.float32))
        self.drive = rng.gamma(2.0, 7.0, (n_kcs, len(ODORS)))
        self._pool = None
        self.det = {c: self.run_kc(c, self.f0, self.thr0(self.f0)) for c in E43.CONDS}

    def run_kc(self, cond, fixed_thr, thr):
        d = self.drive * (1.0 if cond == 'on' else 1.5)
        s = np.clip(np.floor((d - np.asarray(thr).reshape(-1, 1)) / 2.0), 0, None).astype(np.uint16)
        return s >= 2, s

    def run_kc_odor(self, cond, fixed_thr, thr, j):
        r, s = self.run_kc(cond, fixed_thr, thr)
        return r[:, j].copy(), s[:, j].copy()


@pytest.mark.parametrize('jitter', ['normal', 'lognormal'])
def test_sd0_is_part_b(jitter, monkeypatch):
    monkeypatch.setattr(E43.ExcitabilityModel, 'thr_jitter', jitter)
    m = StubModel()
    rng = np.random.default_rng(1)
    xi, zeta = rng.standard_normal(m.n_kcs), rng.standard_normal(m.n_kcs)
    a, b = m.thr_presentation(m.f0, 0.13, xi, 0.0, zeta), m.thr_trial(m.f0, 0.13, xi)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1] and np.array_equal(a[2], b[2])
    t, _, f = m.thr_presentation(m.f0, 0.13, xi, 0.1, zeta)
    assert not np.array_equal(t, b[0])
    assert np.allclose(t.ravel(), m.thr0(m.f0).ravel() * f)


def test_shared_share_of_variance():
    m = StubModel(n_kcs=200_000)
    rng = np.random.default_rng(2)
    xi = rng.standard_normal(m.n_kcs)
    s, sp = 0.07, 0.1
    f1 = m.thr_presentation(m.f0, s, xi, sp, rng.standard_normal(m.n_kcs))[2]
    f2 = m.thr_presentation(m.f0, s, xi, sp, rng.standard_normal(m.n_kcs))[2]
    assert np.isclose(f1.std(), np.hypot(s, sp), rtol=0.01)
    assert np.isclose(np.corrcoef(f1, f2)[0, 1], s ** 2 / (s ** 2 + sp ** 2), atol=0.01)


def test_presentation_draws():
    m = StubModel()
    z = m.p_draws(43, 'rec', 0, 3)
    assert z.shape == (3, len(ODORS), m.n_kcs)
    # (the first trials' draws do not depend on how many are drawn; other reps and kinds differ)
    assert np.array_equal(m.p_draws(43, 'rec', 0, 6)[:3], z)
    assert not np.allclose(m.p_draws(43, 'rec', 1, 3), z)
    assert not np.allclose(m.p_draws(43, 'tune', 0, 3), z)
    # (the shared draws are part B's, untouched by the presentation stream)
    xi = E43.b_draws(43, m.name, 'rec', 0, 3, m.n_kcs)
    assert not np.allclose(xi, z[:, 0, :])
    ss = np.random.SeedSequence(43, spawn_key=(E43.seed_index(m.name), E43.DRAW_KINDS['rec'], 0))
    assert np.array_equal(xi, np.random.default_rng(ss).standard_normal((3, m.n_kcs)))


def test_assembly_against_part_b_and_per_odor_thresholds(monkeypatch):
    m = StubModel()
    xis = E43.b_draws(43, m.name, 'rec', 0, 3, m.n_kcs)
    zetas = m.p_draws(43, 'rec', 0, 3)
    f = 10.5
    # SD 0 through the single-odor path is part B's trials (the all-odor path)
    part_b = m.trials_b(0.12, xis, f)
    monkeypatch.setattr(E43.ExcitabilityModel, 'presentation', True)
    pres0 = m.trials_b(0.12, xis, f, zetas=zetas, sigma_p=0.0)
    for c in E43.CONDS:
        assert np.array_equal(pres0[f'{c}_spk'], part_b[f'{c}_spk'])
        assert np.array_equal(pres0[f'{c}_resp'], part_b[f'{c}_resp'])
    assert np.array_equal(pres0['n_clipped'], part_b['n_clipped'] * len(ODORS))
    # with SD > 0 each odor's column is the KC layer at that presentation's own thresholds
    pres = m.trials_b(0.12, xis, f, zetas=zetas, sigma_p=0.2)
    assert pres['on_spk'].shape == (3, m.n_kcs, len(ODORS)) and float(pres['sigma_p']) == 0.2
    for t in (0, 2):
        for j in (0, 7, 15):
            thr = m.thr_presentation(f, 0.12, xis[t], 0.2, zetas[t, j])[0]
            for c in E43.CONDS:
                assert np.array_equal(pres[f'{c}_spk'][t, :, j], m.run_kc(c, f, thr)[1][:, j])
    assert not np.array_equal(pres['on_spk'], part_b['on_spk'])
    # only the conditions asked for
    on_only = m.trials_b(0.12, xis, f, conds=('on',), zetas=zetas, sigma_p=0.2)
    assert 'acute_off_spk' not in on_only and np.array_equal(on_only['on_spk'], pres['on_spk'])
    assert m.check_presentation() == {c: dict(responses_mismatch=0, spike_counts_mismatch=0) for c in E43.CONDS}
    with pytest.raises(ValueError):
        m.trials_b(0.12, xis, f, sigma_p=0.2)


def test_forked_workers_match_this_process(monkeypatch):
    # (the workers' OpenMP runtime is looked up among the mapped libraries: olfsysm's)
    pytest.importorskip('olfsysm')
    monkeypatch.setattr(E43.ExcitabilityModel, 'presentation', True)
    m = StubModel()
    xis = E43.b_draws(43, m.name, 'tune', 0, 2, m.n_kcs)
    zetas = m.p_draws(43, 'tune', 0, 2)
    here = m.trials_b(0.1, xis, 10.0, zetas=zetas, sigma_p=0.15)
    m.workers = 3
    try:
        there = m.trials_b(0.1, xis, 10.0, zetas=zetas, sigma_p=0.15)
        assert m._pool is not None
    finally:
        m.close_pool()
    assert m._pool is None and E43._POOL_MODEL is None
    for k in ('on_spk', 'acute_off_spk', 'on_resp', 'acute_off_resp', 'factor', 'n_clipped'):
        assert np.array_equal(here[k], there[k]), k


def test_heavy_tail_per_presentation_factors():
    m = StubModel()
    a = dict(on_spk=np.ones((2, m.n_kcs, len(ODORS)), np.uint16), factor=np.ones((2, m.n_kcs, len(ODORS)), np.float32))
    a['factor'][0, :10, 3] = 0.3
    out = E43.heavy_tail(a, 'on')
    assert np.isclose(out['spike_share_thr_below_half'], 10 / a['on_spk'].size)


def test_b_key():
    assert E43.b_key() == E43.B_KEY == 'sigma'
    assert E43.b_key(0.1) == 'p0.1_sigma' and E43.b_key(0.0) == 'p0_sigma'

