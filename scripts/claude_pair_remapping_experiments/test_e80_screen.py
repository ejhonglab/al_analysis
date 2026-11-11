"""Tests of E80's screen (E80_screen): the shared-input transform and its control, the per-KC threshold factors and
their within-type permutation, the point list, the controls' resolution and the gate. No simulation, except the tests
marked by E80_SIM_TEST=1 (opt-in: they build C2, a few minutes at 8 threads), which check that every family's identity
reproduces C2's saved counts exactly."""
import os
import re

import numpy as np
import pandas as pd
import pytest

import E80_screen as S


def _orn(n_glom=36, n_odors=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.gamma(0.6, 40.0, (n_glom, n_odors))
    x[rng.random(x.shape) < 0.04] *= -0.2
    return pd.DataFrame(x, index=pd.Index([f'g{i}' for i in range(n_glom)], name='glomerulus'),
                        columns=pd.Index([f'o{j}' for j in range(n_odors)], name='odor'))


def _kcs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    types = rng.choice(['ab', 'g', "a'b'"], size=n, p=[0.46, 0.35, 0.19])
    return pd.MultiIndex.from_arrays([np.arange(n) + 1000, types], names=['kc_id', 'kc_type'])


def test_shared_input_zero_is_bit_identical():
    orn = _orn()
    out = S.shared_input(orn, 0.0)
    assert np.array_equal(out.to_numpy(), orn.to_numpy()) and out.index.equals(orn.index) and out.columns.equals(orn.columns)
    assert not S.shared_input_drive(orn, 0.0).any()
    z = S.scaled_profile(orn, np.zeros(orn.shape[1]))
    assert np.array_equal(z.to_numpy(), orn.to_numpy())


@pytest.mark.parametrize('eps', list(S.B2_EPS))
def test_shared_input_adds_eps_mean_positive(eps):
    orn = _orn()
    x = orn.to_numpy()
    m = np.where(x > 0, x, 0).mean(axis=0)
    out = S.shared_input(orn, eps).to_numpy()
    assert np.allclose(out - x, eps * m[None, :], rtol=0, atol=1e-12)
    # the added summed drive is eps x the odor's summed positive delta
    assert np.allclose(S.shared_input_drive(orn, eps), eps * np.where(x > 0, x, 0).sum(axis=0), rtol=1e-12)


@pytest.mark.parametrize('eps', list(S.B2_EPS))
def test_scaled_profile_matches_summed_drive(eps):
    orn = _orn()
    x = orn.to_numpy()
    shared = S.shared_input(orn, eps).to_numpy()
    ctrl = S.scaled_profile(orn, S.shared_input_drive(orn, eps)).to_numpy()
    # the same summed delta per odor as the shared input, through the odor's own profile
    assert np.allclose(ctrl.sum(axis=0), shared.sum(axis=0), rtol=1e-12, atol=1e-9)
    assert np.allclose(ctrl.sum(axis=0) - x.sum(axis=0), S.shared_input_drive(orn, eps), rtol=1e-12, atol=1e-9)
    pos = x > 0
    assert np.array_equal(ctrl[~pos], x[~pos])
    # each odor's positive deltas scaled by one factor, 1 + eps (the drive added is eps x the summed positive delta)
    assert np.allclose(ctrl[pos], (x * (1 + eps))[pos], rtol=1e-12)


def test_scaled_profile_rejects_bad_input():
    orn = _orn()
    with pytest.raises(ValueError):
        S.scaled_profile(orn, np.ones(3))
    o = orn.copy()
    o.iloc[:, 0] = -1.0
    with pytest.raises(ValueError):
        S.scaled_profile(o, np.ones(o.shape[1]))


def test_activity_factors():
    idx = _kcs()
    act = pd.Series(np.random.default_rng(1).poisson(3.0, len(idx)).astype(float), index=idx)
    f0 = S.activity_factors(act, 0.0)
    assert (f0.to_numpy() == 1.0).all() and f0.index.equals(idx)
    f = S.activity_factors(act, 0.5)
    assert np.allclose(f.to_numpy(), 1 + 0.5 * act.to_numpy() / act.mean(), rtol=1e-14)
    assert f.mean() == pytest.approx(1.5)
    with pytest.raises(ValueError):
        S.activity_factors(act * 0, 1.0)


def test_permutation_keeps_each_types_factors():
    idx = _kcs()
    act = pd.Series(np.random.default_rng(2).poisson(3.0, len(idx)).astype(float), index=idx)
    f = S.activity_factors(act, 1.0)
    g = S.permute_within_type(f, seed=S.PERM_SEED)
    assert g.index.equals(f.index)
    types = idx.get_level_values('kc_type')
    for t in set(types):
        m = types == t
        assert np.array_equal(np.sort(f[m].to_numpy()), np.sort(g[m].to_numpy()))
    assert not np.array_equal(f.to_numpy(), g.to_numpy())
    # the same seed gives the same permutation, another seed another
    assert np.array_equal(g.to_numpy(), S.permute_within_type(f, seed=S.PERM_SEED).to_numpy())
    assert not np.array_equal(g.to_numpy(), S.permute_within_type(f, seed=S.PERM_SEED + 1).to_numpy())


def test_type_factors():
    idx = _kcs()
    one = S.type_factors(idx, 'ab', 1.0)
    assert (one.to_numpy() == 1.0).all()
    f = S.type_factors(idx, "a'b'", 1.5)
    m = idx.get_level_values('kc_type') == "a'b'"
    assert (f[m] == 1.5).all() and (f[~m] == 1.0).all()
    with pytest.raises(ValueError):
        S.type_factors(idx, 'ab_p', 2.0)


def test_point_specs():
    pts = S.point_specs()
    ids = [p['point_id'] for p in pts]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r'[A-Za-z0-9_.-]+', i) for i in ids)
    handles = list(dict.fromkeys(p['handle'] for p in pts))
    assert handles == ['B1', 'B1_fit0', 'B2', 'B3', 'B4_beta', 'B4_type', 'B5']
    for h in handles:
        hp = [i for i, p in enumerate(pts) if p['handle'] == h]
        ident = [i for i in hp if pts[i]['identity']]
        assert len(ident) == 1 and ident[0] == hp[0]
        assert len({pts[i]['base'] for i in hp}) == 1
    pos = {p['point_id']: i for i, p in enumerate(pts)}
    for p in pts:
        for c in p.get('candidates') or ():
            assert pos[c] < pos[p['point_id']]
    n = {h: sum(1 for p in pts if p['handle'] == h and not p['identity']) for h in handles}
    # the entry's counts: B1 8 (4 on C2, 4 on fit 0), B2 5, B3 2 per unit (plus the APL-off-only reading), B4 11 (plus the
    # opposite direction of the best type point), B5 6
    assert n == {'B1': 4, 'B1_fit0': 4, 'B2': 5, 'B3': 6, 'B4_beta': 5, 'B4_type': 7, 'B5': 6}
    # every identity setting is the identity
    by = {p['point_id']: p for p in pts}
    assert by['B1_identity']['c'] == 1.0 and by['B1_fit0_identity']['c'] == 1.0 and by['B2_identity']['eps'] == 0.0
    assert by['B4_beta_identity']['beta'] == 0.0 and by['B4_type_identity']['factor'] == 1.0
    assert by['B5_identity']['factor'] == 1.0 and by['B3_identity']['arm'] == 'identity'
    assert all(p['base'] == 'fit0' for p in pts if p['handle'] == 'B1_fit0')
    assert all(p['base'] == 'C2' for p in pts if p['handle'] != 'B1_fit0')


def test_resolve_takes_the_best_candidate():
    by = {p['point_id']: p for p in S.point_specs()}
    rows = pd.DataFrame([dict(point_id=f'B2_eps{e:g}', value=e, cal_pattern=0.06 - 0.001 * (e == 0.2), kc_type=None)
                         for e in S.B2_EPS])
    q = S.resolve(by['B2_scaled'], rows)
    assert q['eps'] == 0.2 and q['chosen_from'] == 'B2_eps0.2' and q['setting'] == 'scaled_eps0.2'
    assert S.resolve(by['B2_scaled'], rows.iloc[:2]) is None
    rows = pd.DataFrame([dict(point_id=f'B4_beta{b:g}', value=b, cal_pattern=0.05 + 0.001 * b, kc_type=None) for b in S.B4_BETAS])
    q = S.resolve(by['B4_beta_perm'], rows)
    assert q['beta'] == 0.25 and q['permute_seed'] == S.PERM_SEED
    rows = pd.DataFrame([dict(point_id=f'B4_type_{S.kc_type_label(t)}_x{f:g}', value=f, kc_type=t,
                              cal_pattern=0.05 - 0.002 * (t == 'g' and f == 1.5)) for t in S.KC_TYPES for f in S.B4_TYPE_FACTORS])
    q = S.resolve(by['B4_type_opp'], rows)
    assert q['kc_type'] == 'g' and q['factor'] == pytest.approx(1 / 1.5) and q['chosen_from'] == 'B4_type_g_x1.5'
    assert S.resolve(by['B5_ab_x2'], rows) is by['B5_ab_x2']


def test_controls():
    specs = S.point_specs()
    assert S.control_of('B2_eps0.1', specs) == 'B2_scaled'
    assert S.control_of('B4_beta0.5', specs) == 'B4_beta_perm'
    assert S.control_of('B4_type_ab_x1.25', specs) == 'B4_type_opp'
    assert S.control_of('B5_apbp_x0.5', specs) == 'B5_apbp_x2' and S.control_of('B5_g_x2', specs) == 'B5_g_x0.5'
    for pid in ('B3_thr_noise', 'B2_scaled', 'B4_beta_perm', 'B2_identity', 'B1_c1.5'):
        assert S.control_of(pid, specs) is None


def _screen_rows():
    specs = S.point_specs()
    rows = [dict(point_id='base', handle='', identity=True, base_model='C2', cal_pattern=0.056, raw_loss=0.093,
                 cal_transfer_all=0.0, cal_transfer_six=0.0, realism_flag='', realism_flag_zcrit='')]
    for p in specs:
        rows.append(dict(point_id=p['point_id'], handle=p['handle'], identity=p['identity'], base_model=p['base'],
                         setting=p['setting'], kind=p['kind'], cal_pattern=0.06, raw_loss=0.09, cal_transfer_all=0.0,
                         cal_transfer_six=0.0, realism_flag='', realism_flag_zcrit=''))
    s = pd.DataFrame(rows).set_index('point_id')
    # B2 eps 0.2 passes and its control does not; B5 g x2 would pass but its opposite gains as much
    s.loc['B2_eps0.2', ['cal_pattern', 'cal_transfer_all', 'cal_transfer_six']] = [0.050, 0.30, 0.20]
    s.loc['B2_scaled', 'cal_pattern'] = 0.055
    s.loc['B5_g_x2', ['cal_pattern', 'cal_transfer_all', 'cal_transfer_six']] = [0.050, 0.30, 0.20]
    s.loc['B5_g_x0.5', 'cal_pattern'] = 0.051
    # a B3 point that gains but is outside the bands
    s.loc['B3_thrapl_noise', ['cal_pattern', 'cal_transfer_all', 'cal_transfer_six', 'realism_flag']] = [0.045, 0.4, 0.3, 'x']
    # B1 is never gated
    s.loc['B1_c1.1', ['cal_pattern', 'cal_transfer_all', 'cal_transfer_six']] = [0.040, 0.5, 0.5]
    return s, specs


def test_gate_table():
    s, specs = _screen_rows()
    g = S.gate_table(s, specs).set_index('point_id')
    assert not g.index.str.startswith('B1').any() and not g.index.str.endswith('identity').any()
    assert bool(g.loc['B2_eps0.2', 'best']) and bool(g.loc['B2_eps0.2', 'passes']) and bool(g.loc['B2_eps0.2', 'goes_on'])
    assert g.loc['B2_eps0.2', 'control_gain'] == pytest.approx(0.001)
    assert not bool(g.loc['B2_scaled', 'best'])
    assert bool(g.loc['B5_g_x2', 'best']) and not bool(g.loc['B5_g_x2', 'c3']) and not bool(g.loc['B5_g_x2', 'goes_on'])
    assert bool(g.loc['B3_thrapl_noise', 'best']) and not bool(g.loc['B3_thrapl_noise', 'passes'])
    assert set(g.index[g.goes_on]) == {'B2_eps0.2'}


# ---------------------------------------------------------------------------
# opt-in: the identities simulated on C2

SIM = os.environ.get('E80_SIM_TEST') == '1'


@pytest.fixture(scope='module')
def c2():
    return S.Base('C2')


def _exact(o, saved):
    for c in ('on', 'off'):
        assert o[c][1].index.equals(saved[c][1].index) and o[c][1].columns.equals(saved[c][1].columns)
        assert np.array_equal(o[c][1].to_numpy(dtype=float), saved[c][1].to_numpy(dtype=float))
        assert np.array_equal(o[c][0].to_numpy(dtype=float), saved[c][0].to_numpy(dtype=float))


@pytest.mark.skipif(not SIM, reason='builds C2 (a few minutes); set E80_SIM_TEST=1')
def test_identity_b1_c1(c2):
    o, res = S.simulate_b1(c2, 1.0, 'test B1 identity')
    assert res['apl_factor'] == 1.0
    _exact(o, c2.saved)


@pytest.mark.skipif(not SIM, reason='builds C2 (a few minutes); set E80_SIM_TEST=1')
def test_identity_b4_beta0_and_type1(c2):
    idx = c2.type_index()
    act = pd.Series(c2.saved['off'][1].to_numpy(dtype=float).sum(axis=1), index=idx)
    for f in (S.activity_factors(act, 0.0), S.type_factors(idx, 'g', 1.0)):
        o, res = S.simulate_off_thr(c2, f, 'test B4 identity')
        assert res['thr_factor'] == 1.0
        _exact(o, c2.saved)


@pytest.mark.skipif(not SIM, reason='builds C2 (a few minutes); set E80_SIM_TEST=1')
def test_identity_b5_type1(c2):
    w0 = c2.info['cells']['wAPLKC_unscaled'].cell_contents.copy()
    o, res = S.simulate_b5(c2, "a'b'", 1.0, 'test B5 identity')
    assert res['apl_factor'] == 1.0 and res['n_weights_scaled'] == 0
    _exact(o, c2.saved)
    assert np.array_equal(c2.info['cells']['wAPLKC_unscaled'].cell_contents, w0)
    # a factor other than 1 is restored after the run, and scales only that type's weights
    f = S.aplkc_weight_factors(c2, "a'b'", 2.0)
    assert 0 < int((f == 2.0).sum()) < f.size and set(np.unique(f)) == {1.0, 2.0}


@pytest.mark.skipif(not SIM, reason='builds C2 (a few minutes); set E80_SIM_TEST=1')
def test_identity_b3_one_odor_runs(c2):
    pool = S.OdorPool(1, c2)
    n = len(pool.cols)
    outs, recs = pool.run(dict(thr=[1.0] * n, apl=[1.0] * n, label='test B3 identity'))
    _exact({'on': outs['on1'], 'off': outs['off']}, c2.saved)
    assert all(r['thr'] == 1.0 and r['apl'] == 1.0 for r in recs)


@pytest.mark.skipif(not SIM, reason='builds C2 (a few minutes); set E80_SIM_TEST=1')
def test_identity_b2_eps0(c2):
    P = S.S77.c2_parts()
    o, res = S.simulate_static_orn(S.shared_input(P['orn'], 0.0), 'test B2 identity')
    assert res['thr_factor'] == 1.0 and res['apl_factor'] == 1.0
    _exact(o, c2.saved)
