#!/usr/bin/env python3
"""Tests for the data side of the main target (E76, scripts/claude_pair_remapping_experiments): the per-KC scale helpers
for the raw and F-normalized units (data_targets.kc_scales_raw / kc_scales_fnorm, beside kc_scale_ratios), E76's
matrices in fixed units against them and against the registry, its resampling against the registry's apl_effect_*
entries, its half-splits against E75's, and E10_joint_fit.make_loss's `apl_data_units`. Needs the shared directory (the
data-target registry, the KC data and data/yang_apl16_kc_noise_units/); skipped without it."""

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
E76 = pytest.importorskip('E76_data_side_target')
if not (E76.NOISE_DIR / 'trialmean_long.parquet').exists():
    pytest.skip('data/yang_apl16_kc_noise_units/ is not available', allow_module_level=True)


def _max_abs(a, b):
    return max(float(np.max(np.abs(a[g][k].to_numpy() - b[g][k].reindex(a[g][k].index).to_numpy()))) for g in a for k in a[g])


def _rec_keys(mats):
    return {g: {E76._rec(k): m for k, m in recs.items()} for g, recs in mats.items()}


@pytest.fixture(scope='module')
def z_mats():
    return _rec_keys(D.kc_response_matrices())


def test_scale_one_reproduces_the_registry_effect(z_mats):
    """Every KC's scale 1 gives Yang's z matrices, and their per-pair APL effect is the registry's (to rounding)."""
    ones = D.kc_scales_raw() * 0 + 1.0
    mats = _rec_keys(D.kc_response_matrices(ones))
    for g in mats:
        for k, m in mats[g].items():
            assert m.equals(z_mats[g][k])
    R = {g: H.recording_rdms_from_matrices(mats[g]) for g in E76.GENOTYPES}
    e = H.apl_effect(R['TNT3c'].mean(), R['TNT'].mean())
    a, b = H.align_pairs(e, H.yang_apl_effect())
    assert float((a - b).abs().max()) < 1e-12
    assert abs(float(e.mean()) - H.data_target('apl_effect_mean')) < 1e-12


def test_effect_resamples_in_z_are_the_registry_entries():
    """E76's resampling (the registry's draws, seed 0) reproduces apl_effect_mean's interval, the floor and the split-half
    ρ in z."""
    r = E76.effect_resamples(E76.unit_recording_rdms('z'), n_boot=1000, n_split=500, seed=0)
    lo, hi = H.data_target_ci('apl_effect_mean')
    assert r['mean'] == pytest.approx(H.data_target('apl_effect_mean'), abs=1e-12)
    assert (r['mean_lo'], r['mean_hi']) == pytest.approx((lo, hi), abs=1e-10)
    assert r['rmse_floor'] == pytest.approx(H.data_target('apl_effect_rmse_floor'), abs=1e-10)
    assert r['split_half_rho'] == pytest.approx(H.data_target('apl_effect_rho_split_half'), abs=1e-10)
    assert r['rho_ceiling'] == pytest.approx(H.data_target('apl_effect_rho_ceiling'), abs=1e-10)


def test_noise_unit_matrices_are_the_bands_own():
    """hal's baseline-noise trial means (data/yang_apl16_kc_noise_units/) are Yang's z times kc_scale_ratios (the bands'
    matrices), and at noise_threshold they give the registry's noise_ctrl_frac / noise_tnt_frac."""
    mine = E76.unit_matrices('noise')
    bands = _rec_keys(D.kc_response_matrices(D.kc_scale_ratios()))
    assert _max_abs(mine, bands) < 1e-9
    st = D.criterion_stats(bands, H.data_target('noise_threshold'))
    assert st['ctrl']['frac'] == pytest.approx(H.data_target('noise_ctrl_frac'), rel=1e-12)
    assert st['tnt']['frac'] == pytest.approx(H.data_target('noise_tnt_frac'), rel=1e-12)


def test_raw_scale_is_the_change_in_fc(z_mats):
    """kc_scales_raw (each KC's whole-trace SD) times Yang's z is viki's un-normalized change in Fc (README: to 6.6e-4 a.u.)."""
    raw = _rec_keys(D.kc_response_matrices(D.kc_scales_raw()))
    assert _max_abs(E76.unit_matrices('dfc'), raw) < 1e-3
    s = D.kc_scales_raw()
    assert (s > 0).all() and s.index.equals(D.kc_scale_ratios().index)


@pytest.mark.parametrize('denominator,unit', [('F_mean', 'dff_fmean'), ('Fc_mean', 'dff_fcmean')])
def test_fnorm_scales_match_the_f_normalized_units(denominator, unit):
    """kc_scales_fnorm gives E76's F-normalized units (the change in Fc over the KC's mean F or mean Fc, floored at the 5th
    percentile of its positive values) from Yang's z, with the same floor."""
    scale, floor = D.kc_scales_fnorm(denominator)
    mats = _rec_keys(D.kc_response_matrices(scale))
    mine = E76.unit_matrices(unit)
    assert floor == pytest.approx(E76.FLOORS[unit], rel=1e-12)
    # (the change in Fc from viki's file and z times sd_full differ by at most 6.6e-4 a.u., over a floored denominator)
    assert _max_abs(mine, mats) < 1e-3 / floor
    ks = D.kc_scales_table()
    d = ks['F_mean'] if denominator == 'F_mean' else ks['F_mean'] - D.NEUROPIL_COEF * ks['Fneu_mean']
    assert floor == pytest.approx(float(d[d > 0].quantile(0.05)), rel=1e-12)
    assert np.allclose(scale.to_numpy(), (ks['sd_full'] / d.clip(lower=floor)).to_numpy(), rtol=1e-15, atol=0)
    with pytest.raises(ValueError):
        D.kc_scales_fnorm('F0')


def test_half_splits_start_with_e75s(tmp_path):
    """split_halves.csv's first rows are E75 D1's half-splits (its `_split_halves`, seed 0), and the stratum counts the two
    strongest recordings with APL output blocked (TNT)."""
    df = E76.split_halves(tmp_path, n_splits=100)
    assert E76._check_e75_splits(df, n=100)
    two = {E76.rec_label(t) for t in E76._actual_two('TNT')}
    for _, r in df.head(20).iterrows():
        n_a = len(two & set(r['tnt_a'].split(';')))
        assert r['strongest_in_a'] == n_a and r['strongest_together'] == (n_a in (0, 2))


def test_unit_thresholds_keep_the_control_rate():
    """Each fixed unit's criterion keeps the control flies' response rate at the one under Yang's criterion."""
    target = H.data_target('ctrl_frac')
    for u in ('dfc', 'dff_fmean'):
        rates = E76.unit_rates(u)
        assert rates['TNT3c'] == pytest.approx(target, abs=2e-4)


def test_e10_apl_data_units():
    """make_loss's `apl_data_units`: 'z' (the default) keeps the floor, the calibration target and the data as before;
    'noise' takes them in baseline-noise units; 'noise' needs variant 'apl' and is refused with the zcell readout."""
    E10 = pytest.importorskip('E10_joint_fit')
    try:
        H.data_target('noise_apl_effect_rmse_floor')
    except KeyError:
        pytest.skip('the registry has no noise_apl_effect_* entries yet (data_targets.py --add-new)')
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')}}
    rates = {g: H.yang_response_fractions(g).mean() for g in ('TNT3c', 'TNT')}
    kw = dict(orn_pairs=H.orn16_consensus_pairs(), drive=H.consensus_orn_drive(), ref=H.reference_pairs(), structure=True,
        variant='apl', apl_readout='both', rate_target=H.data_target('ctrl_frac'))
    z = E10.make_loss(data, rates, **kw)
    z2 = E10.make_loss(data, rates, apl_data_units='z', **kw)
    assert z.apl == z2.apl and 'apl_data_units' not in z.apl
    assert z.apl['floor'] == H.data_target('apl_effect_rmse_floor')
    assert z.apl['cal_target'] == H.data_target('ctrl_participation_ratio_resp')
    n = E10.make_loss(data, rates, apl_data_units='noise', **kw)
    assert n.apl['apl_data_units'] == 'noise'
    assert n.apl['floor'] == H.data_target('noise_apl_effect_rmse_floor')
    assert n.apl['cal_target'] == H.data_target('noise_apl_effect_cal_target')
    assert n.apl['data_mean'] == pytest.approx(H.data_target('noise_apl_effect_mean'), abs=1e-12)
    with pytest.raises(ValueError):
        E10.make_loss(data, rates, apl_data_units='noise', **dict(kw, variant='structure', apl_readout='cal'))
    with pytest.raises(ValueError):
        E10.make_loss(data, rates, apl_data_units='noise', **dict(kw, apl_readout='zcell'))
    with pytest.raises(ValueError):
        E10.make_loss(data, rates, apl_data_units='dfc', **kw)
