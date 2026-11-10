#!/usr/bin/env python3
"""Tests for the per-pair APL effect split by the number of odors each KC responds to (E74:
scripts/claude_pair_remapping_experiments/kc_remapping_harness.py `class_cosine_terms`, `apl_effect_by_class`;
E74_apl_effect_by_odors_per_kc.py's class-resolved loss, pattern-error shares, readings and the class term of the joint
fit): the classes add up to the cosine and to the APL effect per pair, one class gives `apl_effect` / `apl_effect_stats`,
a naive per-unit sum and a synthetic population with a known split are recovered, the per-condition table reads the
levels and the claw-count groups' shares of the added responses add up. The tests of E74's module need the shared
directory (the KC data and the data-target registry) and are skipped without it; the joint fit's options (`--apl-class-term`, `--apl-class-form`) are tested in test_e10_zcell.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
H = pytest.importorskip('kc_remapping_harness')

ODORS = list(H.APL16_ODORS)
CLASSES = list(H.OPK_CLASSES)


def _graded(seed=0, n_units=600, p_resp=0.12):
    """z-score-like values: noise around 0 (negatives included) plus responders well above 1."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.4, (n_units, len(ODORS)))
    resp = rng.random(x.shape) < p_resp * rng.gamma(2.0, 0.5, (n_units, 1))
    return pd.DataFrame(np.where(resp, 1.2 + rng.gamma(2.0, 1.0, x.shape), x), columns=ODORS)


def _counts(seed=0, n_units=800, p_resp=0.07):
    rng = np.random.default_rng(seed)
    u = rng.random((n_units, len(ODORS))) * rng.gamma(3.0, 1 / 3.0, (n_units, 1))
    n = np.where(u < p_resp, 2 + rng.geometric(0.12, u.shape), np.where(u < p_resp + 0.05, 1, 0))
    return pd.DataFrame(n.astype(float), columns=ODORS)


def test_classes_and_terms_add_up_to_the_cosine():
    X = _graded()
    t = H.class_cosine_terms(X)
    assert list(t.columns.get_level_values('class').unique()) == CLASSES + ['all']
    tot = t.xs('total', axis=1, level='term')
    cos = 1 - H.pair_distances(X, 'cosine')
    assert np.allclose(tot[CLASSES].sum(axis=1), tot['all'], rtol=0, atol=1e-12)
    assert np.allclose(tot['all'], cos.reindex(t.index), rtol=0, atol=1e-12)
    for c in CLASSES + ['all']:
        assert np.allclose(t[(c, 'both')] + t[(c, 'one')] + t[(c, 'neither')], t[(c, 'total')], rtol=0, atol=1e-12)
    # class 'all' split into terms is cosine_terms
    ct = H.cosine_terms(X)
    for term in ('both', 'one', 'neither'):
        assert np.allclose(t[('all', term)], ct[term], rtol=0, atol=1e-12)


def test_naive_per_unit_sum():
    X = _graded(seed=3, n_units=60)
    t = H.class_cosine_terms(X, thresh=1.0)
    x = X.to_numpy()
    R = x > 1.0
    n_kc = R.sum(axis=1)
    norms = np.sqrt((x ** 2).sum(axis=0))
    i, j = ODORS.index('IaA'), ODORS.index('ms')
    for c, (lo, hi) in zip(CLASSES, H.OPK_EDGES):
        both = one = neither = 0.0
        for u in range(len(x)):
            if not (lo <= n_kc[u] <= (16 if hi is None else hi)):
                continue
            p = x[u, i] * x[u, j]
            if R[u, i] and R[u, j]:
                both += p
            elif R[u, i] or R[u, j]:
                one += p
            else:
                neither += p
        den = norms[i] * norms[j]
        for term, v in (('both', both), ('one', one), ('neither', neither)):
            assert np.isclose(t.loc[('IaA', 'ms'), (c, term)], v / den, rtol=1e-12, atol=1e-15)


def test_effect_classes_add_up_and_match_apl_effect():
    on = [_graded(s) for s in (0, 1, 2)]
    off = [_graded(s, p_resp=0.25) for s in (3, 4)]
    e = H.apl_effect_by_class(on, off)
    eff = e['effect'].xs('total', axis=1, level='term')
    assert np.allclose(eff[CLASSES].sum(axis=1), eff['all'], rtol=0, atol=1e-12)
    d_on = pd.concat([H.pair_distances(m) for m in on], axis=1).mean(axis=1)
    d_off = pd.concat([H.pair_distances(m) for m in off], axis=1).mean(axis=1)
    ref = H.apl_effect(d_on, d_off)
    assert np.allclose(eff['all'], ref.reindex(eff.index), rtol=0, atol=1e-12)
    # the blocks: each condition's mean contribution, the effect off minus on
    assert np.allclose(e['effect'], e['off'] - e['on'])
    for c in ('on', 'off'):
        assert np.isclose(e.attrs['class_shares'][c].sum(), 1.0)


def test_one_class_equals_apl_effect_stats():
    on, off = _counts(1), _counts(2, p_resp=0.14)
    e = H.apl_effect_by_class(on, off, resp_on=on >= 2, resp_off=off >= 2, edges=((0, None),))
    assert list(e['effect'].columns.get_level_values('class').unique()) == ['0plus', 'all']
    d_on, d_off = H.pair_distances(on), H.pair_distances(off)
    eff = e['effect'][('0plus', 'total')]
    assert np.allclose(eff, H.apl_effect(d_on, d_off).reindex(eff.index), rtol=0, atol=1e-12)
    data = 0.8 * H.apl_effect(d_on, d_off) + 0.01 * np.sin(np.arange(120))
    st = H.apl_effect_stats(d_on, d_off, data)
    err = eff - data.reindex(eff.index)
    assert np.isclose(st['apl_effect_rmse'], np.sqrt(np.mean(err ** 2)), rtol=1e-12)
    assert np.isclose(st['apl_effect_rmse_centered'], np.sqrt(np.mean((err - err.mean()) ** 2)), rtol=1e-12)


def _known_population(n_broad: int, n_mid: int, a=3.0, b=2.0, c=2.5, s=0.5, n_narrow=40):
    """Units by construction: n_narrow per odor responding to that odor only (class '1'), n_broad responding to all 16
    odors with value b (class '8plus'), n_mid responding to the first five odors with value c and sub-threshold s at
    the others (class '4_7'). Returns (matrix, the expected per-pair class contributions, keyed (class, term))."""
    rows = []
    for o in range(16):
        r = np.zeros((n_narrow, 16))
        r[:, o] = a
        rows.append(r)
    rows.append(np.full((n_broad, 16), b))
    mid = np.full((n_mid, 16), s)
    mid[:, :5] = c
    rows.append(mid)
    X = np.vstack(rows)
    sq = np.array([n_narrow * a ** 2 + n_broad * b ** 2 + n_mid * (c ** 2 if o < 5 else s ** 2) for o in range(16)])
    exp = {}
    for i in range(16):
        for j in range(i + 1, 16):
            den = np.sqrt(sq[i] * sq[j])
            pair = (ODORS[i], ODORS[j])
            exp[pair] = {('8plus', 'both'): n_broad * b ** 2 / den, ('1', 'total'): 0.0,
                ('4_7', 'both'): (n_mid * c ** 2 / den) if j < 5 else 0.0,
                ('4_7', 'one'): (n_mid * c * s / den) if (i < 5 <= j) else 0.0,
                ('4_7', 'neither'): (n_mid * s ** 2 / den) if i >= 5 else 0.0}
    return pd.DataFrame(X, columns=ODORS), exp


def test_known_split_is_recovered():
    on, e_on = _known_population(n_broad=2, n_mid=10)
    off, e_off = _known_population(n_broad=12, n_mid=30)
    e = H.apl_effect_by_class(on, off)
    for pair in [('2h', 'IaA'), ('2h', 'ms'), ('ms', 'B-myr'), ('eb', 'ep')]:
        for key in e_on[pair]:
            assert np.isclose(e.loc[pair, ('effect',) + key], e_off[pair][key] - e_on[pair][key], rtol=1e-12, atol=1e-15), (pair, key)
        # nothing in the classes the construction leaves empty
        for c in ('0', '2_3'):
            assert e.loc[pair, ('effect', c, 'total')] == 0.0
    sh = e.attrs['class_shares']
    n_on = 16 * 40 + 2 + 10
    assert np.isclose(sh['on']['1'], 16 * 40 / n_on) and np.isclose(sh['on']['8plus'], 2 / n_on)
    assert np.isclose(sh['on']['4_7'], 10 / n_on)


def test_responders_aligned_by_label():
    X = _counts(5)
    resp = X >= 2
    a = H.class_cosine_terms(X, resp=resp)
    perm = list(reversed(ODORS))
    b = H.class_cosine_terms(X[perm], resp=resp[perm])
    c = H.class_cosine_terms(X, resp=resp[perm])             # a DataFrame is aligned by its labels
    pd.testing.assert_frame_equal(a, b.loc[a.index])
    pd.testing.assert_frame_equal(a, c)
    with pytest.raises(ValueError):
        H.class_cosine_terms(X, resp=resp.iloc[:10])


# ---------------------------------------------------------------------------
# E74's module and the joint fit's option (need the shared directory)
def _shared():
    try:
        H.data_targets_path().read_bytes()
        H.yang_recording_matrices('TNT3c')
    except (FileNotFoundError, OSError):
        pytest.skip('the shared directory (KC data, data-target registry) is not available')
    return pytest.importorskip('E74_apl_effect_by_odors_per_kc')


def _need_class_floor():
    import json
    if 'apl_class_rmse_floor' not in json.loads(H.data_targets_path().read_text()):
        pytest.skip('the registry has no apl_class_rmse_floor yet')


def test_class_loss_and_pattern_shares():
    E = _shared()
    data = E.data_effect_array()
    assert np.allclose(data[:, :E.ALL, E.TOTAL].sum(axis=1), data[:, E.ALL, E.TOTAL], atol=1e-12)
    assert np.isclose(data[:, E.ALL, E.TOTAL].mean(), H.data_target('apl_effect_mean'), rtol=1e-9)
    assert E.class_loss(data, data) == 0.0
    rng = np.random.default_rng(0)
    model = data + rng.normal(0, 0.02, data.shape)
    model[:, E.ALL] = model[:, :E.ALL].sum(axis=1)
    sh = E.pattern_error_shares(model, data)
    assert np.isclose(sh.sum(), 1.0)
    d = model[:, :E.ALL, E.TOTAL] - data[:, :E.ALL, E.TOTAL]
    assert np.isclose(E.class_loss(model, data), np.sqrt(np.mean(d ** 2)))
    # the DataFrame form gives the same
    assert np.isclose(E.class_loss(E._effect_frame(model), E.data_class_effect()), E.class_loss(model, data))


def test_read_cal_total_is_the_readout_effect():
    E = _shared()
    on, off = _counts(1), _counts(2, p_resp=0.14)
    K = 4.0
    blk = E.read_cal(on.to_numpy(), off.to_numpy(), 2, K)
    ref = H.apl_effect(H.readout_pair_distances(on, 'saturating', K), H.readout_pair_distances(off, 'saturating', K))
    assert np.allclose(blk['effect'][:, E.ALL, E.TOTAL], ref.reindex(E.PAIR_INDEX).to_numpy(), atol=1e-12)


def test_pseudo_noise_fixed():
    E = _shared()
    a_on, a_off = E.pseudo_noise(123)
    b_on, b_off = E.pseudo_noise(123)
    assert a_on is b_on and a_off is b_off
    E._NOISE.clear()
    c_on, c_off = E.pseudo_noise(123)
    assert np.array_equal(a_on, c_on) and np.array_equal(a_off, c_off)
    assert a_on.shape == (E.N_RECORDINGS, 123, 16) and not np.array_equal(a_on, a_off)
    # the trial means of three standard normal draws
    assert np.isclose(a_on.std(), 1 / np.sqrt(E.N_TRIALS), rtol=0.05)


def test_class_term_stats_is_the_reading():
    E = _shared()
    _need_class_floor()
    on, off = _counts(1), _counts(2, p_resp=0.14)
    labels = [f'{o} @ -3' for o in ODORS]
    s_on, s_off = on.set_axis(labels, axis=1), off.set_axis(labels, axis=1)
    zc = dict(q=0.128, amp='saturating', drift=0.98)
    cal = dict(K=5.0, noise=0.05, noise_ratio=0.4)
    st = E.class_term_stats(s_on, s_off, zc, cal)
    blk = E.read_zcell(on.to_numpy(), off.to_numpy(), dict(zc, **cal), noise=True)
    data = E.data_effect_array()
    d = blk['effect'][:, :E.ALL, E.TOTAL] - data[:, :E.ALL, E.TOTAL]
    assert st['apl_class_mse'] == float(np.mean(d ** 2))
    assert np.isclose(st['apl_class_rmse'] ** 2, st['apl_class_mse'])
    assert st['apl_class_floor'] == H.data_target('apl_class_rmse_floor')
    assert E.class_term_stats(s_on, s_off, zc, cal) == st          # the same draws at every evaluation


def test_per_condition_table():
    """The per-condition table: a model whose class arrays are the data's reproduces the data's levels and effects, a
    level moved above the data's interval is marked so, and the 1-to-3 class is the sum of its parts when a by_variant
    table has no column for it."""
    E = _shared()
    tables = E.data_tables(50, seed=0)
    d = E.data_class_terms()
    blk = dict(on=d['arrays']['TNT3c'].mean(axis=0), off=d['arrays']['TNT'].mean(axis=0))
    blk['effect'] = blk['off'] - blk['on']
    st = E._variant_stats(blk, E.data_effect_array())
    (s, _, f, t), (s2, _, f2, t2) = E.KEY_MODELS[:2]
    moved = dict(st, level_on_4_7=st['level_on_4_7'] + 1.0)
    df = pd.DataFrame([dict(folder=f, tag=t, reading='cal2', **st), dict(folder=f2, tag=t2, reading='zcell', **moved)])
    pc = E.per_condition_table(df, tables)
    dat = pc[pc.reading == 'data'].set_index('class')
    m = pc[pc.model == s].set_index('class')
    for cls in E.PC_CLASSES:
        for q in ('level_on', 'level_off', 'effect'):
            assert np.isclose(m.loc[cls, q], dat.loc[cls, q], rtol=0, atol=1e-12), (cls, q)
            assert dat.loc[cls, f'{q}_lo'] <= dat.loc[cls, f'{q}_hi']
    m2 = pc[pc.model == s2].set_index('class')
    assert (m2.reading == 'zcell').all()
    assert m2.loc['4_7', 'level_on_outside'] == 'above'
    assert np.isclose(m2.loc['4_7', 'effect'], dat.loc['4_7', 'effect'] - 1.0)
    pc2 = E.per_condition_table(df.drop(columns=['level_on_1_3', 'level_off_1_3']), tables)
    assert np.allclose(pc2.level_on.to_numpy(float), pc.level_on.to_numpy(float), equal_nan=True)
    assert np.allclose(pc2.level_off.to_numpy(float), pc.level_off.to_numpy(float), equal_nan=True)
    md = E.per_condition_md(pc)
    assert sum(l.startswith('| KC data |') for l in md) == 2 and any('↑' in l for l in md)


def test_claw_release():
    """Who is released by claw count: the groups' shares of the added responses add to 1 and are each group's added
    responses over all KCs'; the group sizes, odors per KC and silent shares are the group's."""
    E = _shared()
    rng = np.random.default_rng(0)
    claws = rng.integers(1, 13, 500).astype(float)
    n_on = rng.poisson(1.0, 500).astype(float)
    n_off = n_on + rng.poisson(0.2 * claws)
    rows = {r['group']: r for r in E.claw_release(n_on, n_off, claws)}
    assert list(rows) == [g for g, *_ in E.CLAW_GROUPS]
    assert np.isclose(sum(r['share_added'] for r in rows.values()), 1.0)
    m = claws >= 9
    assert rows['9+']['n_kcs'] == m.sum()
    assert np.isclose(rows['9+']['share_added'], (n_off[m] - n_on[m]).sum() / (n_off - n_on).sum())
    assert np.isclose(rows['1-4']['opk_off'], n_off[claws <= 4].mean())
    assert np.isclose(rows['5-6']['silent_on'], (n_on[(claws >= 5) & (claws <= 6)] == 0).mean())
    # more claws, more added responses here: the 9+ group's share exceeds its share of KCs
    assert rows['9+']['share_added'] > m.mean()
