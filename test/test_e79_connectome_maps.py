#!/usr/bin/env python3
"""Tests for E79's connectome analyses (E79_connectome_maps.py in scripts/claude_pair_remapping_experiments) on synthetic
geometry: the glomerulus co-localization on two- and three-bouton layouts, the posterior-view projection, the 12 x 12 grid
and its flips on a synthetic point set, the masked Pearson r, the proxy's invariance when the local field is uniform and
the leave-one-odor-out share. No connectome tables, measured maps or simulations needed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
E = pytest.importorskip('E79_connectome_maps')


def test_colocalization_two_glomeruli():
    lam = 4.0
    d = 6.0
    b = pd.DataFrame({'glomerulus': ['A', 'B'], 'n_tbars': [10, 30], 'R': [0.0, d], 'D': [0.0, 0.0], 'A': [0.0, 0.0]})
    C = E.colocalization(b, lam, ['A', 'B'])
    assert C.loc['A', 'A'] == pytest.approx(1.0)
    assert C.loc['B', 'B'] == pytest.approx(1.0)
    assert C.loc['A', 'B'] == pytest.approx(np.exp(-d ** 2 / (2 * lam ** 2)))
    assert np.allclose(C.to_numpy(), C.to_numpy().T)


def test_colocalization_weighted_mean():
    lam = 2.0
    # A has two boutons (weights 1 and 3) at 0 and 4 um along D; B one bouton at 0: C_AB = (1 * 1 + 3 * k(4)) / 4
    b = pd.DataFrame({'glomerulus': ['A', 'A', 'B'], 'n_tbars': [1, 3, 5], 'R': [0.0, 0.0, 0.0], 'D': [0.0, 4.0, 0.0],
                      'A': [0.0, 0.0, 0.0]})
    C = E.colocalization(b, lam, ['A', 'B'])
    k4 = np.exp(-16 / (2 * lam ** 2))
    assert C.loc['A', 'B'] == pytest.approx((1 + 3 * k4) / 4)
    assert C.loc['A', 'A'] == pytest.approx((1 + 9 + 2 * 3 * k4) / 16)


def test_to_rda_axes():
    # hemibrain: +y is anterior, -z dorsal, -x the fly's right; FlyWire: x right, -y dorsal, -z anterior
    assert np.allclose(E.to_rda('hemibrain', 0, 1000, 0), [[0, 0, 1]])
    assert np.allclose(E.to_rda('hemibrain', 0, 0, -1000), [[0, 1, 0]])
    assert np.allclose(E.to_rda('flywire', 1000, 0, 0), [[1, 0, 0]])
    assert np.allclose(E.to_rda('flywire', 0, 0, -1000), [[0, 0, 1]])
    assert np.allclose(E.to_rda('banc', 0, 1000, 1000), [[0, 1, -1]])


def _disk(n=4000, radius=20.0, seed=0):
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n))
    t = 2 * np.pi * rng.random(n)
    return pd.DataFrame({'R': r * np.cos(t), 'D': r * np.sin(t), 'A': rng.uniform(-3, 3, n)})


def test_projection_grid_and_flips():
    apl = _disk()
    frame = E.calyx_frame(apl)
    # one glomerulus's boutons at the dorsal right of the disk (posterior view: right of the image, top)
    pts = pd.DataFrame({'R': [12.0] * 5, 'D': [12.0] * 5, 'A': [0.0] * 5, 'col': [0] * 5, 'w': [1.0] * 5})
    got = E.slab_column_grids(pts, 1, apl, frame, 0.0)
    assert got is not None
    grids, inside, mask = got
    g = grids[2.0][0].reshape(E.GRID_N, E.GRID_N)
    iy, ix = np.unravel_index(np.argmax(g), g.shape)
    assert iy < E.GRID_N // 2 and ix >= E.GRID_N // 2
    # the whole calyx (no slab) puts the same points in the same place
    whole = E.slab_column_grids(pts, 1, apl, frame, None)[0][2.0][0].reshape(E.GRID_N, E.GRID_N)
    assert np.unravel_index(np.argmax(whole), whole.shape) == (iy, ix)
    # the outline is about the disk: corners of the grid are outside, the center inside
    ins = inside.reshape(E.GRID_N, E.GRID_N)
    assert not ins[0, 0] and ins[E.GRID_N // 2, E.GRID_N // 2]
    for flip, (want_top, want_right) in {'none': (True, True), 'horizontal': (True, False),
                                         'vertical': (False, True), 'both': (False, False)}.items():
        f = E.flip_grid(g, flip)
        fy, fx = np.unravel_index(np.argmax(f), f.shape)
        assert (fy < E.GRID_N // 2) == want_top and (fx >= E.GRID_N // 2) == want_right
        # flattened grids flip the same way
        assert np.allclose(E.flip_grid(g.ravel(), flip).reshape(E.GRID_N, E.GRID_N), f)


def test_cell_matrix_means():
    mask = np.zeros((24, 36), bool)
    mask[:, :] = True
    M, inside = E.cell_matrix(mask, n=12)
    assert inside.all()
    img = np.arange(mask.size, dtype=float).reshape(mask.shape)
    cells = np.asarray(M.T @ img.ravel()).ravel()
    # cell (0, 0) covers rows 0-1 and columns 0-2
    assert cells[0] == pytest.approx(img[:2, :3].mean())
    assert cells[-1] == pytest.approx(img[-2:, -3:].mean())


def test_masked_pearson():
    rng = np.random.default_rng(1)
    pred = rng.normal(size=(5, 144))
    meas = rng.normal(size=144)
    inside = rng.random(144) > 0.3
    r = E.masked_pearson(pred, meas, inside)
    for k in range(5):
        assert r[k] == pytest.approx(np.corrcoef(pred[k, inside], meas[inside])[0, 1])


def test_proxy_uniform_field_is_global():
    rng = np.random.default_rng(2)
    X = rng.random((10, 6)) * 50
    C = np.ones((10, 10))
    for form, strength in E.PROXY_SETTINGS:
        assert np.allclose(E.proxy_change(X, C, form, strength), 0.0, atol=1e-12)
    # a local field that differs between glomeruli changes the distances, and permutation axes broadcast
    C2 = np.eye(10) + 0.1
    d = E.proxy_change(X, np.stack([C2, C2]), 'sub', 1.0)
    assert d.shape == (2, 15) and np.abs(d).max() > 0


def test_loo_share():
    pairs = E._pairs(6)
    rng = np.random.default_rng(3)
    err = rng.normal(size=len(pairs))
    err -= err.mean()
    # a change exactly opposite to the error removes all of it (k = 1 fitted on every training set)
    assert E.loo_share(-err, err, pairs, 6) == pytest.approx(1.0)
    assert E.in_sample_share(-2 * err, err) == pytest.approx(1.0)
    # with k constrained to be non-negative, a change along the error removes nothing
    assert E.loo_share(err, err, pairs, 6, nonneg=True) == pytest.approx(0.0)
    noise = rng.normal(size=(200, len(pairs)))
    assert np.median(E.loo_share(noise, err, pairs, 6)) < 0.05
