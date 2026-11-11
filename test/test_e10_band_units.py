#!/usr/bin/env python3
"""Tests for E10_joint_fit.make_loss's `band_units` (scripts/claude_pair_remapping_experiments): 'z' (the default)
leaves every term and statistic as before; 'noise' takes the KC data of the two-spike rate terms in baseline-noise
units (the registry's noise_* entries, the realism bands' data side since Tom's decision of 2026-11-10 16:10). Needs
the shared directory (the data-target registry and the KC data); skipped without it."""

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
S = pytest.importorskip('apl_effect_scores')
E10 = pytest.importorskip('E10_joint_fit')

ODORS = list(H.APL16_ODORS)
LABELS = [f'{o} @ -3' for o in ODORS]
BINS = ('0', '1', '2_3', '4_7', '8plus')


def _outputs(seed=0, n_units=900, p_on=0.07, p_off=0.15):
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.6, 1.0, (n_units, len(ODORS))) * rng.gamma(2.0, 0.5, (n_units, 1))
    out = {}
    for c, p in (('on', p_on), ('off', p_off)):
        thr = np.quantile(base, 1 - p)
        n = np.where(base > thr, np.rint(2 + 6 * (base - thr) / base.std()), (base > 0.8 * thr).astype(float))
        s = pd.DataFrame(n, columns=LABELS)
        out[c] = ((s >= 2).astype(float), s)
    return out


@pytest.fixture(scope='module')
def ctx():
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g).mean() for g in ('TNT3c', 'TNT')}
    return dict(data=data, data_rates=data_rates, orn_pairs=H.orn16_consensus_pairs(), drive=H.consensus_orn_drive(),
        ref=H.reference_pairs(), rate=H.data_target('ctrl_frac'))


def _loss(ctx, **kw):
    kw.setdefault('variant', 'apl')
    if kw['variant'] == 'apl':
        kw.setdefault('apl_readout', 'cal')
    return E10.make_loss(ctx['data'], ctx['data_rates'], ctx['orn_pairs'], ctx['drive'], ref=ctx['ref'], structure=True,
        rate_target=ctx['rate'], **kw)


def test_band_units_z_unchanged(ctx):
    o = _outputs()
    for kw in (dict(apl_opk_term=True), dict(apl_readout='both'), dict(variant='structure')):
        a, lf = _loss(ctx, **kw), _loss(ctx, band_units='z', **kw)
        assert a(o) == lf(o)
        assert a.ratio_target == lf.ratio_target and lf.band_units == 'z'
        assert lf.apl is None or 'band_units' not in lf.apl


def test_band_units_noise_terms(ctx):
    o = _outputs()
    w = E10.loss_weights('apl')
    z_lf, n_lf = _loss(ctx, apl_opk_term=True), _loss(ctx, apl_opk_term=True, band_units='noise')
    z, n = z_lf(o), n_lf(o)
    fr = {g: S.frac_data(c, 'noise') for c, g in (('on', 'TNT3c'), ('off', 'TNT'))}
    assert n_lf.ratio_target == pytest.approx(fr['TNT'].mean() / fr['TNT3c'].mean(), rel=1e-12)
    assert n_lf.ratio_target == pytest.approx(H.data_target('noise_rate_ratio_fraction'), rel=1e-12)
    assert n_lf.apl['ratio_wall'] == list(H.data_target_ci('noise_rate_ratio_fraction'))
    assert n_lf.apl['opk_data'] == {b: H.data_target(f'noise_tnt_odors_per_kc_{b}') for b in BINS}
    assert n_lf.apl['band_units'] == 'noise'
    # the model's statistics are the same; the data side of the rate terms moves
    for k in ('rate_on', 'rate_off', 'ratio', 'opk_off_0', 'apl_raw_rmse', 'apl_cal_rmse', 'on_spearman'):
        assert n[k] == z[k]
    po = H.response_fractions(o['on'][0]).reindex(ODORS)
    assert n['po_on_spearman'] == pytest.approx(H.spearman(po, fr['TNT3c']), rel=1e-12)
    d = np.array([n[f'opk_off_{b}'] for b in BINS])
    assert n['opk_off_dist'] == pytest.approx(0.5 * np.abs(d - S.opk_data('off', 'noise')).sum(), rel=1e-12)
    assert n['term_ratio'] == pytest.approx(w['ratio'] * ((np.log(n['ratio']) - np.log(n_lf.ratio_target)) / E10.RATIO_LOG_SCALE) ** 2)
    assert n['term_opk'] == pytest.approx(w['opk'] * (n['opk_off_dist'] / E10.OPK_SCALE) ** 2)
    assert n['term_rate_on'] == z['term_rate_on']
    assert n['loss'] == pytest.approx(sum(v for k, v in n.items() if k.startswith('term_')))
    # (the non-APL variants switch the per-odor rates and the ratio target too)
    s = _loss(ctx, variant='structure', band_units='noise')
    assert s.ratio_target == n_lf.ratio_target


def test_band_units_validation(ctx):
    with pytest.raises(ValueError):
        _loss(ctx, band_units='dff')
    with pytest.raises(ValueError):
        _loss(ctx, apl_readout='zcell', rate_criterion='zcell', band_units='noise')
