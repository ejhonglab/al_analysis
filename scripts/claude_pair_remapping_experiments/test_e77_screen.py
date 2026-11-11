"""Tests of E77's screen (E77_input_screen): the L1 / L2 per-odor weight factors, the static handles' transforms at their
identity settings, the point list. No simulation, except `test_simulated_identity` (opt-in: E77_SIM_TEST=1; it builds C2,
several minutes at 4 threads)."""
import os

import numpy as np
import pandas as pd
import pytest

import E77_input_screen as S


def _activity(n_units=7, n_odors=4, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 50.0, (n_units, n_odors))
    a[0, 0] = 0.0
    return pd.DataFrame(a, index=pd.Index([f'u{i}' for i in range(n_units)]), columns=[f'o{j}' for j in range(n_odors)])


def _toy_wiring():
    # 6 claws; glomeruli A (2 boutons), B, C (1 each), D (no ORN input); one bouton per claw
    claws = pd.MultiIndex.from_tuples([(10, 0), (10, 1), (11, 0), (11, 1), (11, 2), (12, 0)], names=['kc_id', 'claw_id'])
    boutons = pd.MultiIndex.from_tuples([('A', 1, 0), ('A', 1, 1), ('B', 2, 0), ('C', 3, 0), ('D', 4, 0)],
                                        names=['glomerulus', 'pn_id', 'bouton_id'])
    W = pd.DataFrame(0.0, index=claws, columns=boutons)
    for claw, b in zip(claws, [0, 1, 0, 2, 3, 4]):
        W.loc[claw, boutons[b]] = 1.0
    return W


def test_power_zero_is_bit_identical():
    a = _activity()
    f = S.odor_weight_factors(a, 0.0)
    assert (f.to_numpy() == 1.0).all()
    w = np.random.default_rng(1).uniform(0.1, 2.0, len(a))
    assert np.array_equal(w[:, None] * f.to_numpy(), np.repeat(w[:, None], a.shape[1], axis=1))
    # with a permutation too (a permutation of ones)
    assert (S.odor_weight_factors(a, 0.0, permute_seed=3).to_numpy() == 1.0).all()


@pytest.mark.parametrize('power', [0.5, 1.0, 2.0])
def test_factors_mean_one_per_odor(power):
    a = _activity()
    f = S.odor_weight_factors(a, power)
    assert np.allclose(f.mean(axis=0).to_numpy(), 1.0, rtol=0, atol=1e-12)
    # monotone in the activity within each odor, and proportional to activity ** power
    for j in range(a.shape[1]):
        r = a.iloc[:, j].to_numpy() ** power
        assert np.allclose(f.iloc[:, j].to_numpy(), r / r.mean(), rtol=1e-12, atol=0)
    assert f.iloc[0, 0] == 0.0


def test_permutation_keeps_each_odors_factors():
    a = _activity(n_units=50, n_odors=5)
    f = S.odor_weight_factors(a, 2.0)
    g = S.odor_weight_factors(a, 2.0, permute_seed=1)
    for j in range(a.shape[1]):
        assert np.array_equal(np.sort(f.iloc[:, j].to_numpy()), np.sort(g.iloc[:, j].to_numpy()))
    assert not np.array_equal(f.to_numpy(), g.to_numpy())
    assert np.allclose(g.mean(axis=0).to_numpy(), 1.0, rtol=0, atol=1e-12)
    # the same seed gives the same permutation
    assert np.array_equal(g.to_numpy(), S.odor_weight_factors(a, 2.0, permute_seed=1).to_numpy())


def test_factors_reject_bad_activity():
    a = _activity()
    a.iloc[1, 1] = -1.0
    with pytest.raises(ValueError):
        S.odor_weight_factors(a, 1.0)
    a = _activity()
    a.iloc[:, 2] = 0.0
    with pytest.raises(ValueError):
        S.odor_weight_factors(a, 1.0)


def test_orn_power_one_is_bit_identical():
    rng = np.random.default_rng(2)
    orn = pd.DataFrame(rng.normal(30.0, 40.0, (36, 16)))
    out = S.orn_power(orn, 1.0)
    assert np.array_equal(out.to_numpy(), orn.to_numpy()) and out.index.equals(orn.index) and out.columns.equals(orn.columns)


@pytest.mark.parametrize('power', [0.5, 0.75, 1.33, 2.0])
def test_orn_power_keeps_sums_and_negatives(power):
    rng = np.random.default_rng(3)
    x = rng.normal(30.0, 40.0, (36, 16))
    orn = pd.DataFrame(x)
    y = S.orn_power(orn, power).to_numpy()
    pos = x > 0
    assert np.array_equal(y[~pos], x[~pos])
    assert np.allclose(np.where(pos, y, 0).sum(axis=0), np.where(pos, x, 0).sum(axis=0), rtol=1e-12, atol=0)
    assert np.allclose(y.sum(axis=0), x.sum(axis=0), rtol=1e-12, atol=1e-9)
    for j in range(x.shape[1]):
        # each odor's positive deltas keep their order and are proportional to x ** power
        p = pos[:, j]
        r = x[p, j] ** power
        assert np.allclose(y[p, j], r * x[p, j].sum() / r.sum(), rtol=1e-12, atol=0)


def test_claw_count_factors_identity_and_values():
    W = _toy_wiring()
    n = S.glomerulus_claw_counts(W)
    assert n.to_dict() == {'A': 3, 'B': 1, 'C': 1, 'D': 1}
    f0 = S.claw_count_factors(W, ['A', 'B', 'C'], 0.0)
    assert (f0.to_numpy() == 1.0).all()
    assert np.array_equal(S.weighted_wiring(W, f0).to_numpy(), W.to_numpy())
    f1 = S.claw_count_factors(W, ['A', 'B', 'C'], 1.0)
    # median claw count of the driven glomeruli (3, 1, 1) is 1: A's weights x 1/3; the undriven D keeps 1
    assert f1['A'] == pytest.approx(1 / 3) and f1['B'] == 1.0 and f1['C'] == 1.0 and f1['D'] == 1.0
    Wf = S.weighted_wiring(W, f1)
    per_glom = pd.Series(Wf.to_numpy().sum(axis=0), index=W.columns.get_level_values('glomerulus')).groupby(level=0).sum()
    assert per_glom.loc[['A', 'B', 'C']].to_numpy() == pytest.approx([1.0, 1.0, 1.0])
    fh = S.claw_count_factors(W, ['A', 'B', 'C'], 0.5)
    assert fh['A'] == pytest.approx(3 ** -0.5)


def test_claw_count_factor_permutation():
    rng = np.random.default_rng(4)
    gl = [f'g{i}' for i in range(12)]
    boutons = pd.MultiIndex.from_tuples([(g, i, 0) for i, g in enumerate(gl)], names=['glomerulus', 'pn_id', 'bouton_id'])
    claws = pd.MultiIndex.from_tuples([(k, 0) for k in range(200)], names=['kc_id', 'claw_id'])
    W = pd.DataFrame(0.0, index=claws, columns=boutons)
    for c, b in enumerate(rng.integers(0, len(gl), len(claws))):
        W.iloc[c, b] = 1.0
    driven = gl[:10]
    f = S.claw_count_factors(W, driven, 1.0)
    g = S.claw_count_factors(W, driven, 1.0, permute_seed=1)
    assert np.array_equal(np.sort(f[driven].to_numpy()), np.sort(g[driven].to_numpy()))
    assert (g[gl[10:]] == 1.0).all()


def test_e20_gain_identity():
    from E20_glomerulus_gain import gained_wPNKC
    W = _toy_wiring()
    peaks = pd.Series({'A': 120.0, 'B': 60.0, 'C': 200.0, 'D': 40.0})
    Wg, g = gained_wPNKC(W, peaks, 0.0, ['A', 'B', 'C'])
    assert np.array_equal(Wg.to_numpy(), W.to_numpy()) and (g == 1.0).all()


def test_point_specs():
    pts = S.point_specs()
    ids = [p['point_id'] for p in pts]
    assert len(ids) == len(set(ids))
    import re
    assert all(re.fullmatch(r'[A-Za-z0-9_.-]+', i) for i in ids)
    # every family's identity point comes before its other points
    for fam in ('S1', 'S2', 'S3', 'S5', 'L'):
        fam_pts = [i for i, p in enumerate(pts) if S.family(p) == fam]
        ident = [i for i in fam_pts if pts[i]['identity']]
        assert len(ident) == 1 and ident[0] == fam_pts[0]
    # controls come after their candidates
    pos = {p['point_id']: i for i, p in enumerate(pts)}
    for p in pts:
        for c in p.get('candidates') or ():
            assert pos[c] < pos[p['point_id']]
    # S6 last
    assert all(p['handle'] == 'S6' for p in pts[-2:])
    s1 = [p for p in pts if p['handle'] == 'S1' and not p['identity']]
    assert [p['kws']['pn_inhadd'] for p in s1] == list(S.S1_INHADD)
    assert all(p['kws']['pn_inhsc'] > 368.6631 for p in s1)


def test_resolve_takes_the_better_candidate():
    pt = next(p for p in S.point_specs() if p['point_id'] == 'L1_perm')
    rows = pd.DataFrame([dict(point_id='L1_p1', value=1.0, cal_loss=0.06), dict(point_id='L1_p2', value=2.0, cal_loss=0.05)])
    q = S._resolve(pt, rows)
    assert q['power'] == 2.0 and q['setting'] == 'perm_p2' and q['chosen_from'] == 'L1_p2'
    assert S._resolve(pt, rows.iloc[:1]) is None
    pt = next(p for p in S.point_specs() if p['point_id'] == 'S5_claws_perm')
    rows = pd.DataFrame([dict(point_id='S5_claws_e0.5', value=0.5, cal_loss=0.05), dict(point_id='S5_claws_e1', value=1.0, cal_loss=0.06)])
    q = S._resolve(pt, rows)
    assert q['wiring']['exponent'] == 0.5 and q['wiring']['permute_seed'] == S.PERM_SEED


@pytest.mark.skipif(os.environ.get('E77_SIM_TEST') != '1', reason='builds C2 (several minutes); set E77_SIM_TEST=1')
def test_simulated_identity():
    P = S.c2_parts()
    res = S.hold_rates(lambda: S.build_c2_simulator(dict(pn_inhadd=31.4088, pn_inhsc=368.6631)), label='test identity')
    assert res['thr_factor'] == 1.0 and res['apl_factor'] == 1.0
    for c in ('on', 'off'):
        assert np.array_equal(res[c][1].to_numpy(dtype=float), P['saved'][c][1].to_numpy(dtype=float))
