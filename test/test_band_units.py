#!/usr/bin/env python3
"""Tests for the realism bands' data side in baseline-noise units (Tom, 2026-11-10 16:10;
scripts/claude_pair_remapping_experiments): the registry's noise_* entries (data_targets.py `criterion_stats`,
`noise_unit_criterion`), the flags of apl_effect_scores.py under both units and E73_per_kc_normalization.fixed_unit_flags
(E10_joint_fit.make_loss's `band_units`: test_e10_band_units.py). Needs the shared directory (the data-target registry, the KC data and
viki's per-KC scales); skipped without it."""

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
    H.data_target('noise_threshold')
except (FileNotFoundError, OSError):
    pytest.skip('the shared directory (data-target registry) is not available', allow_module_level=True)
except KeyError:
    pytest.skip('the registry has no noise_* entries yet (data_targets.py --add-new)', allow_module_level=True)
D = pytest.importorskip('data_targets')
S = pytest.importorskip('apl_effect_scores')

ODORS = list(H.APL16_ODORS)
LABELS = [f'{o} @ -3' for o in ODORS]
BINS = D.OPK_BIN_NAMES


def _close(a, b, rtol=1e-12):
    return bool(np.isclose(a, b, rtol=rtol, atol=1e-15))


@pytest.fixture(scope='module')
def z_mats():
    return D.kc_response_matrices()


@pytest.fixture(scope='module')
def noise():
    return D.noise_unit_criterion()


def test_scale_one_threshold_one_reproduces_z_entries(z_mats):
    """`criterion_stats` with every KC's scale 1 and threshold 1 gives the registry's z-unit entries."""
    ones = D.kc_scale_ratios() * 0 + 1.0
    mats = D.kc_response_matrices(ones)
    for g in mats:
        for k, m in mats[g].items():
            assert m.equals(z_mats[g][k])
    st = D.criterion_stats(mats, 1.0)
    dt = H.data_target
    for name in ('ctrl', 'tnt'):
        assert _close(st[name]['frac'], dt(f'{name}_frac'))
        for b in BINS:
            assert _close(st[name]['opk'][b], dt(f'{name}_odors_per_kc_{b}'))
        assert st[name]['opk_tv_max'] == dt(f'{name}_odors_per_kc_tv_max')
        assert st[name]['opk_tv_median'] == dt(f'{name}_odors_per_kc_tv_median')
    assert _close(st['rate_ratio'], dt('rate_ratio_fraction'))
    for a, b in zip(st['rate_ratio_ci90'], H.data_target_ci('rate_ratio_fraction')):
        assert _close(a, b)
    # the per-odor fractions are the response fraction table's (harness.yang_response_fractions)
    for name, g in (('ctrl', 'TNT3c'), ('tnt', 'TNT')):
        ref = H.yang_response_fractions(g)[ODORS]
        assert np.allclose(st[name]['frac_per_recording'].to_numpy(), ref.to_numpy(), rtol=1e-12, atol=1e-15)


def test_kc_scales_cover_every_kc(z_mats):
    s = D.kc_scale_ratios()
    n = sum(len(m) for mats in z_mats.values() for m in mats.values())
    assert len(s) == n == 38640
    assert (s > 0).all()


def test_noise_unit_entries(noise):
    """The registry's noise_* entries are `noise_unit_criterion`'s, and reproduce viki's summary.json (the threshold to
    the olfdb values' rounding, the ratio exactly) and the planning probe's values."""
    thr, nu, zu = noise
    dt = H.data_target
    assert dt('noise_threshold') == thr
    summ = json.loads((H.shared_dir() / 'notes' / 'yang_response_normalization' / 'summary.json').read_text())
    assert abs(thr - summ['baseline_threshold']) < 1e-6
    assert _close(nu['rate_ratio'], summ['rate_ratio']['baseline-noise units, > 1.97'])
    for name in ('ctrl', 'tnt'):
        assert dt(f'noise_{name}_frac') == nu[name]['frac']
        assert dt(f'noise_{name}_odors_per_kc_tv_max') == nu[name]['opk_tv_max']
        for b in BINS:
            assert dt(f'noise_{name}_odors_per_kc_{b}') == nu[name]['opk'][b]
        for o in ODORS:
            assert dt(f'noise_{name}_frac_{o}') == nu[name]['frac_per_odor'][o]
    assert dt('noise_rate_ratio_fraction') == nu['rate_ratio']
    assert list(H.data_target_ci('noise_rate_ratio_fraction')) == nu['rate_ratio_ci90']
    # the threshold holds the control flies at their rate under Yang's criterion (to one KC x odor entry)
    assert abs(nu['ctrl']['frac'] - dt('ctrl_frac')) < 1e-5
    # the probe (notes/response_criterion_decision.md)
    assert round(nu['tnt']['frac'], 3) == 0.156 and round(nu['rate_ratio'], 2) == 2.28
    assert [round(v, 1) for v in nu['rate_ratio_ci90']] == [2.0, 2.7]
    assert round(nu['ctrl']['opk_tv_max'], 3) == 0.155 and round(nu['tnt']['opk_tv_max'], 3) == 0.223
    assert round(dt('noise_ctrl_odors_per_kc_tv_vs_z'), 3) == 0.064 and round(dt('noise_tnt_odors_per_kc_tv_vs_z'), 3) == 0.078


def test_criterion_threshold_bisection():
    rng = np.random.default_rng(0)
    mats = {'TNT3c': {(str(i), 1, 'L'): pd.DataFrame(rng.normal(size=(500, 16)), columns=ODORS) for i in range(3)}}
    t = D.criterion_threshold(mats, 0.1)
    r = np.mean([float((m.to_numpy() > t).mean()) for m in mats['TNT3c'].values()])
    assert abs(r - 0.1) < 1e-3
    assert abs(t - 1.2816) < 0.05


def test_band_data_by_units():
    assert np.array_equal(S.opk_data('on'), [H.data_target(f'ctrl_odors_per_kc_{b}') for b in BINS])
    assert np.array_equal(S.opk_data('off', 'noise'), [H.data_target(f'noise_tnt_odors_per_kc_{b}') for b in BINS])
    assert S.opk_band('on') == H.data_target('ctrl_odors_per_kc_tv_max')
    assert S.opk_band('off', 'noise') == H.data_target('noise_tnt_odors_per_kc_tv_max')
    assert S.frac_data('on').equals(H.yang_response_fractions('TNT3c')[ODORS].mean())
    assert list(S.frac_data('off', 'noise').index) == ODORS
    assert S.data_rate_ratio('noise')[0] == H.data_target('noise_rate_ratio_fraction')


def test_realism_flag_units():
    """Only the odors-per-KC bands depend on the units; the defaults are Yang's z (as before)."""
    tv_on = 0.5 * (H.data_target('noise_ctrl_odors_per_kc_tv_max') + H.data_target('ctrl_odors_per_kc_tv_max'))
    tv_off = 0.5 * (H.data_target('noise_tnt_odors_per_kc_tv_max') + H.data_target('tnt_odors_per_kc_tv_max'))
    z = S.realism_flag(0.068, 0.15, 0.6, tv_on, tv_off)
    assert z == S.realism_flag(0.068, 0.15, 0.6, tv_on, tv_off, units='z')
    n = S.realism_flag(0.068, 0.15, 0.6, tv_on, tv_off, units='noise')
    # the APL-on band is tighter in noise units (0.155 < 0.167), the APL-off one looser (0.223 > 0.196)
    assert 'APL on' in n and 'APL off' not in n
    assert 'APL on' not in z and 'APL off' in z
    assert S.realism_flag(0.068, 0.15, 0.6, units='noise') == ''
    assert 'APL-on response rate' in S.realism_flag(0.10, 0.2, 0.6, units='noise')
    assert 'response rate ρ' in S.realism_flag(0.068, 0.15, 0.2, units='noise')


def test_archive_opk_both_units(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    folder = tmp_path / 'E0_test'
    folder.mkdir()
    resp = {}
    for c, p in (('on', 0.07), ('acute_off', 0.15)):
        r = pd.DataFrame((rng.random((800, 16)) < p).astype(float), columns=LABELS)
        r.to_parquet(folder / f'v_{c}_responses.parquet')
        resp[c] = r
    monkeypatch.setattr(S, 'ARCHIVE', tmp_path)
    out = S.archive_opk(folder, 'v')
    for c, name in (('on', 'on'), ('off', 'acute_off')):
        d = S.odors_per_kc(resp[name])
        assert out[f'opk_{c}_tv'] == pytest.approx(0.5 * np.abs(d - S.opk_data(c, 'noise')).sum(), rel=1e-12)
        assert out[f'opk_{c}_tv_zcrit'] == pytest.approx(0.5 * np.abs(d - S.opk_data(c, 'z')).sum(), rel=1e-12)
        per_odor = resp[name].mean()
        per_odor.index = ODORS
        assert out[f'rate_rho_{c}'] == pytest.approx(H.spearman(per_odor, S.frac_data(c, 'noise')), rel=1e-12)
    assert S.archive_opk(tmp_path / 'E0_none', 'v') == {}


# ------------------------------------------------------------------- E73's fixed-unit columns
def test_e73_fixed_unit_flags(monkeypatch):
    E73 = pytest.importorskip('E73_per_kc_normalization')
    rng = np.random.default_rng(2)
    counts = {}
    rows = []
    for i, (p_on, p_off) in enumerate(((0.068, 0.15), (0.12, 0.2), (0.068, 0.25))):
        for c, p in (('on', p_on), ('acute_off', p_off)):
            counts[(f'v{i}', c)] = pd.DataFrame(np.where(rng.random((600, 16)) < p, 3.0, 0.0), columns=ODORS)
        resp = {c: counts[(f'v{i}', name)] >= 2 for c, name in (('on', 'on'), ('off', 'acute_off'))}
        row = dict(folder='E0_test', tag=f'v{i}', reading_def='present_raw', reading='graded', rate_on=float(resp['on'].mean().mean()),
            rate_off=float(resp['off'].mean().mean()))
        for c in ('on', 'off'):
            row.update({f'opk_{c}_{b}': float(v) for b, v in zip(BINS, S.odors_per_kc(resp[c].astype(float)))})
        rows.append(row)
        rows.append(dict(row, reading_def='zcell_saturating'))
    monkeypatch.setattr(E73, 'load_counts', lambda folder, tag, cond: counts[(tag, cond)])
    fx = E73.fixed_unit_flags(pd.DataFrame(rows)).set_index('tag')
    assert list(fx.index) == ['v0', 'v1', 'v2']
    for r in rows[::2]:
        f = fx.loc[r['tag']]
        po = (counts[(r['tag'], 'on')] >= 2).astype(float).mean()
        assert f.fixed_rate_rho_on == pytest.approx(H.spearman(po, S.frac_data('on', 'noise')), rel=1e-12)
        d = np.array([r[f'opk_off_{b}'] for b in BINS])
        assert f.fixed_opk_off_tv == pytest.approx(0.5 * np.abs(d - S.opk_data('off', 'noise')).sum(), rel=1e-12)
        assert f.fixed_realism_flag == S.realism_flag(r['rate_on'], r['rate_off'], f.fixed_rate_rho_on, f.fixed_opk_on_tv,
            f.fixed_opk_off_tv, units='noise')
    assert 'APL-on response rate' in fx.loc['v1', 'fixed_realism_flag']
