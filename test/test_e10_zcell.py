#!/usr/bin/env python3
"""Tests for the per-KC session z-score options of the joint fit (E73 part C:
scripts/claude_pair_remapping_experiments/E10_joint_fit.py `zcell_terms` and make_loss's `apl_readout='zcell'`,
`rate_criterion='zcell'`, `zcell`): the reading per evaluation is E73 part B's (`zcell_rows`), the rate terms take
the reading's criterion only when asked, the options off leave every term and statistic as before, and the 'start'
calibration is held; E74 part C's class term (`apl_class_term`) is off by default and adds only its own term, in
either form (`apl_class_form`). Needs the shared directory (the data-target registry and the KC data); skipped without it."""

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
E10 = pytest.importorskip('E10_joint_fit')
E73 = pytest.importorskip('E73_per_kc_normalization')

ODORS = list(H.APL16_ODORS)
# the simulator's odor labels, as the fit's spike counts carry them
LABELS = [f'{o} @ -3' for o in ODORS]


def _outputs(seed=0, n_units=900, p_on=0.07, p_off=0.15):
    """{'on', 'off'}: (responses, spike counts) of a synthetic model, more and stronger responders with APL off."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.6, 1.0, (n_units, len(ODORS))) * rng.gamma(2.0, 0.5, (n_units, 1))
    out = {}
    for c, p in (('on', p_on), ('off', p_off)):
        thr = np.quantile(base, 1 - p)
        n = np.where(base > thr, np.rint(2 + 6 * (base - thr) / base.std()), (base > 0.8 * thr).astype(float))
        n = n + rng.binomial(1, 0.01, n.shape) * (n == 0)
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
    kw.setdefault('apl_readout', 'cal')
    return E10.make_loss(ctx['data'], ctx['data_rates'], ctx['orn_pairs'], ctx['drive'], ref=ctx['ref'], structure=True,
        variant='apl', rate_target=ctx['rate'], **kw)


def test_zcell_terms_is_part_b_reading():
    o = _outputs()
    q = H.data_target('zcell_q_window')
    st, cal = E10.zcell_terms(o['on'][1], o['off'][1], dict(q=q, amp='saturating'))
    on, off = (H.abbrev_columns(o[c][1])[ODORS] for c in ('on', 'off'))
    row = E73.zcell_rows({}, on, off, q=q, a_map='saturating', readings=('graded',))[0][0]
    for k, c in (('rmse', 'rmse'), ('mean', 'mean'), ('pattern', 'rmse_centered'), ('per_odor', 'per_odor'),
            ('pair_specific', 'pair_specific'), ('rho', 'spearman')):
        assert st[f'apl_zcell_{k}'] == row[c]
    for c in ('rate_on', 'rate_off', 'rate_ratio', 'rate_rho_on', 'rate_rho_off', 'opk_off_tv', 'K', 'noise', 'noise_ratio'):
        assert st[f'zcell_{c}'] == row[c]
    assert st['zcell_K'] == cal['K'] and st['zcell_noise'] == cal['noise']
    # the calibration reaches the control flies' rate under the data's criterion
    assert np.isclose(st['zcell_rate_on'], H.data_target('ctrl_frac'), rtol=1e-6)
    assert all(np.isscalar(v) and isinstance(v, float) for v in st.values())
    # a given calibration is applied, not re-made
    o2 = _outputs(seed=1)
    st2, cal2 = E10.zcell_terms(o2['on'][1], o2['off'][1], dict(q=q, amp='saturating'), cal=cal)
    assert cal2 is cal and st2['zcell_noise'] == cal['noise']
    assert not np.isclose(st2['zcell_rate_on'], H.data_target('ctrl_frac'), rtol=1e-6)


def test_options_off_unchanged(ctx):
    o = _outputs()
    for ro in ('raw', 'cal', 'both'):
        a = _loss(ctx, apl_readout=ro)(o)
        b = _loss(ctx, apl_readout=ro, rate_criterion='spikes', zcell=None)(o)
        assert a == b
        assert not any(k.startswith(('zcell_', 'apl_zcell_')) for k in a)
    assert _loss(ctx).zcell is None


def test_apl_readout_zcell_only_the_apl_term(ctx):
    o = _outputs()
    cal = _loss(ctx, apl_readout='cal', apl_opk_term=True)(o)
    z = _loss(ctx, apl_readout='zcell', apl_opk_term=True)(o)
    floor = H.data_target('apl_effect_rmse_floor')
    assert np.isclose(z['term_apl'], 2 * z['apl_zcell_rmse'] / floor)
    # the rate terms keep the two-spike criterion
    for t in ('rate_on', 'ratio', 'po_on', 'po_off', 'ratio_wall', 'po_wall', 'opk'):
        assert z[f'term_{t}'] == cal[f'term_{t}']
    pat = _loss(ctx, apl_readout='zcell', apl_pattern_only=True)(o)
    assert np.isclose(pat['term_apl'], 2 * pat['apl_zcell_pattern'] / floor)


def test_rate_criterion_zcell_terms(ctx):
    o = _outputs()
    lf = _loss(ctx, apl_readout='zcell', rate_criterion='zcell', apl_opk_term=True)
    z = lf(o)
    w = E10.loss_weights('apl')
    ratio_target = lf.ratio_target
    assert z['term_rate_on'] < 1e-8 * w['rate_on']      # the calibration holds the APL-on rate at the target
    assert np.isclose(z['term_ratio'], w['ratio'] * ((np.log(z['zcell_rate_ratio']) - np.log(ratio_target))
        / E10.RATIO_LOG_SCALE) ** 2)
    assert np.isclose(z['term_po_on'], w['po_on'] * (1 - z['zcell_rate_rho_on']))
    assert np.isclose(z['term_po_off'], w['po_off'] * (1 - z['zcell_rate_rho_off']))
    assert np.isclose(z['term_opk'], w['opk'] * (z['zcell_opk_off_tv'] / E10.OPK_SCALE) ** 2)
    lo, hi = lf.apl['ratio_wall']
    ex = max(0.0, lo - z['zcell_rate_ratio'], z['zcell_rate_ratio'] - hi)
    assert np.isclose(z['term_ratio_wall'], w['ratio_wall'] * (ex / E10.RATIO_WALL_SCALE) ** 2)
    # the two-spike statistics are still recorded under their own names
    assert z['rate_on'] == _loss(ctx)(o)['rate_on']
    assert np.isclose(z['loss'], sum(v for k, v in z.items() if k.startswith('term_')))


def test_calibrate_start_held(ctx):
    o, o2 = _outputs(), _outputs(seed=1)
    every = _loss(ctx, apl_readout='zcell', rate_criterion='zcell')
    start = _loss(ctx, apl_readout='zcell', rate_criterion='zcell', zcell=dict(calibrate='start'))
    e1, s1 = every(o), start(o)
    assert e1 == s1          # the first evaluation calibrates either way
    e2, s2 = every(o2), start(o2)
    assert s2['zcell_noise'] == s1['zcell_noise'] and s2['zcell_K'] == s1['zcell_K']
    assert e2['zcell_noise'] != e1['zcell_noise']
    assert s2['term_rate_on'] > e2['term_rate_on']      # the held measurement lets the operating point term bind
    start.zcell_held.clear()
    assert start(o2) == e2


def test_validation(ctx):
    with pytest.raises(ValueError):
        E10.make_loss(ctx['data'], ctx['data_rates'], ctx['orn_pairs'], ctx['drive'], ref=ctx['ref'], structure=True,
            variant='apl', rate_target=0.05, apl_readout='zcell', rate_criterion='zcell')
    with pytest.raises(ValueError):
        E10.make_loss(ctx['data'], ctx['data_rates'], ctx['orn_pairs'], ctx['drive'], ref=ctx['ref'], structure=True,
            variant='structure', rate_criterion='zcell')
    with pytest.raises(ValueError):
        _loss(ctx, zcell=dict(q=0.2))
    with pytest.raises(ValueError):
        _loss(ctx, apl_readout='zcell', zcell=dict(amp='log'))
    # fit 4's form needs no particular rate target
    E10.make_loss(ctx['data'], ctx['data_rates'], ctx['orn_pairs'], ctx['drive'], ref=ctx['ref'], structure=True,
        variant='apl', rate_target=0.05, apl_readout='zcell')


def test_apl_class_term(ctx):
    """E74 part C: off by default; on, it adds its term (apl_class_mse over the square of the registry's floor, weight
    APL_CLASS_WEIGHT) and its apl_class_* statistics and changes nothing else; it needs the reading's APL term."""
    import json
    if 'apl_class_rmse_floor' not in json.loads(H.data_targets_path().read_text()):
        pytest.skip('the registry has no apl_class_rmse_floor yet')
    o = _outputs()
    off = _loss(ctx, apl_readout='zcell', rate_criterion='zcell')
    assert off.apl_class_term is False and 'class_term' not in off.apl
    a = off(o)
    assert 'term_apl_class' not in a and not any(k.startswith('apl_class_') for k in a)
    on = _loss(ctx, apl_readout='zcell', rate_criterion='zcell', apl_class_term=True)
    b = on(o)
    floor = H.data_target('apl_class_rmse_floor')
    assert np.isclose(b['term_apl_class'], E10.APL_CLASS_WEIGHT * b['apl_class_mse'] / floor ** 2)
    for k, v in a.items():
        if k != 'loss':
            assert b[k] == v, k
    assert np.isclose(b['loss'], a['loss'] + b['term_apl_class'])
    assert on(o) == b          # the same pseudo-recordings at every evaluation
    with pytest.raises(ValueError):
        _loss(ctx, apl_readout='cal', apl_class_term=True)
    with pytest.raises(ValueError):
        E10.make_loss(ctx['data'], ctx['data_rates'], ctx['orn_pairs'], ctx['drive'], ref=ctx['ref'], structure=True,
            variant='structure', apl_class_term=True)


def test_apl_class_form(ctx):
    """E74 part C: the class term's form. 'sq' (the default) is the squared form; 'lin' is apl_class_rmse over the
    floor, the same statistics and every other term unchanged; a form other than 'sq' needs the class term."""
    import json
    if 'apl_class_rmse_floor' not in json.loads(H.data_targets_path().read_text()):
        pytest.skip('the registry has no apl_class_rmse_floor yet')
    o = _outputs()
    sq = _loss(ctx, apl_readout='zcell', rate_criterion='zcell', apl_class_term=True)
    lin = _loss(ctx, apl_readout='zcell', rate_criterion='zcell', apl_class_term=True, apl_class_form='lin')
    assert sq.apl['class_term']['form'] == 'sq' and lin.apl['class_term']['form'] == 'lin'
    a, b = sq(o), lin(o)
    floor = H.data_target('apl_class_rmse_floor')
    assert np.isclose(a['term_apl_class'], E10.APL_CLASS_WEIGHT * a['apl_class_mse'] / floor ** 2)
    assert np.isclose(b['term_apl_class'], E10.APL_CLASS_WEIGHT * b['apl_class_rmse'] / floor)
    assert np.isclose(b['apl_class_rmse'] ** 2, b['apl_class_mse'])
    for k, v in a.items():
        if k not in ('loss', 'term_apl_class'):
            assert b[k] == v, k
    assert np.isclose(b['loss'], a['loss'] - a['term_apl_class'] + b['term_apl_class'])
    assert lin(o) == b
    with pytest.raises(ValueError):
        _loss(ctx, apl_readout='zcell', rate_criterion='zcell', apl_class_form='lin')
    with pytest.raises(ValueError):
        _loss(ctx, apl_readout='zcell', rate_criterion='zcell', apl_class_term=True, apl_class_form='log')
    with pytest.raises(ValueError):
        E10.make_loss(ctx['data'], ctx['data_rates'], ctx['orn_pairs'], ctx['drive'], ref=ctx['ref'], structure=True,
            variant='structure', apl_class_form='lin')
