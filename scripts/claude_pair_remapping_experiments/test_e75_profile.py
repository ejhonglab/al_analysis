"""Tests of E75's glomerulus profile weight builder (E75_glomerulus_apl_profile.profile_factors / apply_profile)."""
import numpy as np
import pandas as pd
import pytest

import E75_glomerulus_apl_profile as E75


def _toy():
    # 3 KCs (2, 3 and 1 claws), 3 glomeruli with 2, 1 and 1 boutons; one bouton per claw
    claws = pd.MultiIndex.from_tuples([(10, 0), (10, 1), (11, 0), (11, 1), (11, 2), (12, 0)], names=['kc_id', 'claw_id'])
    boutons = pd.MultiIndex.from_tuples([('A', 1, 0), ('A', 1, 1), ('B', 2, 0), ('C', 3, 0)], names=['glomerulus', 'pn_id', 'bouton_id'])
    W = pd.DataFrame(0.0, index=claws, columns=boutons)
    for claw, b in zip(claws, [0, 2, 1, 2, 3, 0]):
        W.loc[claw, boutons[b]] = 1.0
    wAPLKC = pd.Series(np.linspace(0.5, 1.5, len(claws)), index=claws)
    wAPLPN = pd.Series([0.2, 0.4, 0.6, 0.8], index=boutons)
    return W, wAPLKC, wAPLPN


@pytest.mark.parametrize('site', E75.SITES)
def test_zero_scores_are_bit_identical(site):
    W, kc, pn = _toy()
    a, b = E75.apply_profile(site, {'A': 0.0, 'B': 0.0}, W, kc, pn)
    assert np.array_equal(a.to_numpy(), kc.to_numpy()) and np.array_equal(b.to_numpy(), pn.to_numpy())
    a, b = E75.apply_profile(site, {}, W, kc, pn)
    assert np.array_equal(a.to_numpy(), kc.to_numpy()) and np.array_equal(b.to_numpy(), pn.to_numpy())


@pytest.mark.parametrize('site', E75.SITES)
def test_factor_mean_is_one(site):
    W, kc, pn = _toy()
    f = E75.profile_factors(site, {'A': np.log(2), 'C': -np.log(3)}, W, kc, pn)
    assert f.mean() == pytest.approx(1.0)


def test_one_glomerulus_touches_only_its_claws_boutons_or_cells():
    W, kc, pn = _toy()
    g = E75.claw_glomeruli(W)
    # claw: only claws from B differ from the rest (before the mean normalization, all others share one value)
    f = E75.profile_factors('claw', {'B': np.log(2)}, W, kc, pn)
    assert np.allclose(f[g == 'B'] / f[g != 'B'].iloc[0], 2.0) and f[g != 'B'].nunique() == 1
    # bouton: only B's bouton
    f = E75.profile_factors('bouton', {'B': np.log(2)}, W, kc, pn)
    isB = f.index.get_level_values('glomerulus') == 'B'
    assert np.allclose(f[isB] / f[~isB].iloc[0], 2.0) and f[~isB].nunique() == 1
    # cell: every claw of a KC gets 2 ** (its number of claws from B); KC 11 has one claw from B, KC 10 one, KC 12 none
    f = E75.profile_factors('cell', {'B': np.log(2)}, W, kc, pn)
    kcs = f.index.get_level_values('kc_id')
    per_kc = f.groupby(kcs).agg(['min', 'max'])
    assert (per_kc['min'] == per_kc['max']).all()
    assert per_kc.loc[10, 'min'] / per_kc.loc[12, 'min'] == pytest.approx(2.0)
    assert per_kc.loc[11, 'min'] / per_kc.loc[12, 'min'] == pytest.approx(2.0)
