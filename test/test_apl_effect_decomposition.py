#!/usr/bin/env python3
"""Tests for the APL-effect loss decomposition in the KC-remapping harness
(scripts/claude_pair_remapping_experiments/kc_remapping_harness.py)."""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'))
H = pytest.importorskip('kc_remapping_harness')

ODORS = ['a', 'b', 'c', 'd', 'e', 'f']


def _pairs():
    return pd.MultiIndex.from_tuples(list(combinations(ODORS, 2)), names=['odor_a', 'odor_b'])


def _series(values, name='distance'):
    return pd.Series(np.asarray(values, dtype=float), index=_pairs(), name=name)


def test_odor_shares_recovers_additive_effect():
    shares = np.array([0.1, 0.2, 0.05, 0.3, 0.0, 0.15])
    effect = _series([shares[ODORS.index(a)] + shares[ODORS.index(b)] for a, b in _pairs()])
    s, r2 = H.odor_shares(effect)
    assert np.allclose(s.reindex(ODORS).to_numpy(), shares)
    assert r2 == pytest.approx(1.0)


def test_decomposition_squares_add_and_parts():
    rng = np.random.default_rng(0)
    n = len(_pairs())
    data = _series(rng.normal(0.18, 0.05, n), name='apl_effect')
    off = _series(rng.uniform(0.4, 0.9, n))
    # model effect = data + a constant + an additive (per-odor) error: no pair-specific error
    add = np.array([0.03, -0.02, 0.01, 0.0, -0.01, 0.02])
    per_odor_err = np.array([add[ODORS.index(a)] + add[ODORS.index(b)] for a, b in _pairs()])
    on = off + data.to_numpy() - 0.05 + per_odor_err
    stats, shares = H.apl_effect_decomposition(on, off, data)
    total = stats['apl_effect_mean_error'] ** 2 + stats['apl_effect_per_odor'] ** 2 + stats['apl_effect_pair_specific'] ** 2
    assert stats['apl_effect_rmse'] ** 2 == pytest.approx(total)
    assert stats['apl_effect_pair_specific'] == pytest.approx(0.0, abs=1e-12)
    assert stats['apl_effect_per_odor'] > 0
    assert list(shares.columns) == ['model_share', 'data_share', 'share_error']
    # a purely pair-specific error (orthogonal to the additive space) has no per-odor part
    X, _ = H.pair_design(_pairs(), ODORS)
    noise = rng.normal(0, 0.02, n)
    noise -= X @ np.linalg.lstsq(X, noise, rcond=None)[0]
    stats2, _ = H.apl_effect_decomposition(off + data.to_numpy() + noise, off, data)
    assert stats2['apl_effect_per_odor'] == pytest.approx(0.0, abs=1e-12)
    assert stats2['apl_effect_pair_specific'] == pytest.approx(float(np.sqrt(np.mean(noise ** 2))))
    parts = H.apl_effect_pair_parts(off + data.to_numpy() + noise, off, data)
    assert np.allclose(parts['per_odor'] + parts['pair_specific'], parts['error'])
    assert np.allclose(parts['pair_specific'].to_numpy(), noise)
