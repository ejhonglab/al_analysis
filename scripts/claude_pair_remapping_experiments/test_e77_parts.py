"""Tests of E77's split of the APL-effect error (kc_remapping_harness.apl_effect_inherited_split)."""
import numpy as np
import pandas as pd

import kc_remapping_harness as H


def _pairs(n_odors=6, seed=0):
    rng = np.random.default_rng(seed)
    odors = [f'o{i}' for i in range(n_odors)]
    idx = pd.MultiIndex.from_tuples([(a, b) for i, a in enumerate(odors) for b in odors[i + 1:]], names=['odor_a', 'odor_b'])
    off = pd.Series(rng.uniform(0.2, 0.9, len(idx)), index=idx)
    on = pd.Series(np.clip(off.to_numpy() + rng.uniform(0.0, 0.3, len(idx)), 0, 1.2), index=idx)
    return on, off


def test_terms_add_to_the_error():
    m_on, m_off = _pairs(seed=1)
    d_on, d_off = _pairs(seed=2)
    s = H.apl_effect_inherited_split(m_on, m_off, d_on, d_off)
    total = s.inherited + s.curve + s.pair_specific
    assert np.allclose(total, s.error, atol=1e-12, rtol=0)
    assert np.allclose(s.inherited_c + s.curve_c + s.pair_specific_c, s.error_c, atol=1e-12, rtol=0)
    sh = H.inherited_shares(s)
    assert abs(sh['share_inherited'] + sh['share_curve'] + sh['share_pair_specific'] - 1) < 1e-12


def test_terms_add_with_per_pair_curves():
    m_on, m_off = _pairs(seed=3)
    d_on, d_off = _pairs(seed=4)
    n = len(m_on)
    curves = np.column_stack([np.linspace(0.8, 1.2, n), np.linspace(0.9, 1.3, n)])
    s = H.apl_effect_inherited_split(m_on, m_off, d_on, d_off, data_curve=curves)
    assert np.allclose(s.inherited + s.curve + s.pair_specific, s.error, atol=1e-12, rtol=0)


def test_identical_model_and_data_give_zero():
    d_on, d_off = _pairs(seed=5)
    s = H.apl_effect_inherited_split(d_on.copy(), d_off.copy(), d_on, d_off)
    for c in ('error', 'inherited', 'curve', 'pair_specific'):
        assert np.allclose(s[c], 0, atol=1e-12)


def test_inherited_is_zero_when_apl_off_distances_match():
    d_on, d_off = _pairs(seed=6)
    m_on, _ = _pairs(seed=7)
    s = H.apl_effect_inherited_split(m_on, d_off.copy(), d_on, d_off)
    assert np.allclose(s.inherited, 0, atol=1e-12)
