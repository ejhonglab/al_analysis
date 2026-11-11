#!/usr/bin/env python3
"""Tests for the data-side analyses of the APL calyx maps (E79, scripts/claude_pair_remapping_experiments/
E79_apl_calyx_constraints.py) on synthetic inputs: apl_calyx_maps.py's ImageJ ROI reader and mask, the paraffin oil block
and the position-matched blank, the three baselines of the stimulus maps, the first singular vector's share, the
autocorrelation length, the profile shares, the matched pair and the fly bootstrap. Needs no shared directory."""

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
E79 = pytest.importorskip('E79_apl_calyx_constraints')
ACM = E79.ACM


def _imagej_polygon_roi(xs, ys) -> bytes:
    """An ImageJ polygon ROI (type 0) with the given integer vertices."""
    xs, ys = np.asarray(xs, int), np.asarray(ys, int)
    top, left, bottom, right = ys.min(), xs.min(), ys.max(), xs.max()
    header = bytearray(64)
    header[:4] = b'Iout'
    header[4:6] = struct.pack('>h', 227)
    header[6] = 0
    header[8:18] = struct.pack('>hhhhh', top, left, bottom, right, len(xs))
    body = struct.pack(f'>{len(xs)}h', *(xs - left)) + struct.pack(f'>{len(ys)}h', *(ys - top))
    return bytes(header) + body


def test_roi_reader_and_mask(tmp_path):
    p = tmp_path / 'square.roi'
    p.write_bytes(_imagej_polygon_roi([10, 30, 30, 10], [5, 5, 25, 25]))
    poly = ACM.read_imagej_polygon(p)
    assert np.array_equal(poly, [[10, 5], [30, 5], [30, 25], [10, 25]])
    mask = ACM.roi_mask(poly, (40, 50))
    # pixel centers (x + 0.5, y + 0.5) inside [10, 30] x [5, 25]: columns 10-29, rows 5-24
    assert mask.sum() == 20 * 20
    assert mask[5, 10] and mask[24, 29] and not mask[25, 10] and not mask[5, 30]


def _frames(pfo_block: int = 16, n_blocks: int = 17) -> pd.DataFrame:
    """frames.csv rows of a synthetic recording: paraffin oil first in each block, then one odor x concentration twice,
    except the block `pfo_block` of three paraffin oil presentations; the odor of block 12 repeats block 2's."""
    stims = [('ms', -2.0 - 0.5 * i) for i in range(5)] + [('pa', -4.0 - 0.5 * i) for i in range(6)] + \
        [('paa', -3.0 - 0.5 * i) for i in range(5)]
    rows, k = [], 0
    for b in range(n_blocks):
        if b == pfo_block:
            rows += [dict(block=b, odor='pfo', log10_conc=np.nan)] * 3
            continue
        o, c = stims[k % len(stims)] if b != 12 else stims[2]
        k += 1
        rows += [dict(block=b, odor='pfo', log10_conc=np.nan), dict(block=b, odor=o, log10_conc=c), dict(block=b, odor=o, log10_conc=c)]
    f = pd.DataFrame(rows)
    f.index.name = 'frame'
    return E79.add_positions(f)


@pytest.mark.parametrize('pfo_block', [16, 11])
def test_positions_and_blank(pfo_block):
    f = _frames(pfo_block)
    assert list(f['position'][:6]) == [1, 2, 3, 1, 2, 3]
    b = E79.blank_frames(f)
    assert b['pfo_pos2'] == [3 * pfo_block + 1] and b['pfo_pos3'] == [3 * pfo_block + 2]
    assert len(b['first_in_block']) == 17 and len(b['all']) == 19
    # block 12 repeats block 2's stimulus: the later presentations are doubtful
    assert f['doubtful'].sum() == 2 and set(f.loc[f['doubtful'], 'block']) == {12}
    rng = np.random.default_rng(0)
    stack = rng.normal(size=(len(f), 8, 9))
    # first-in-block frames all 1, the paraffin oil block's 2nd and 3rd frames 4 and 6: the blank is 5 - 1
    stack[b['first_in_block']] = 1.0
    stack[b['pfo_pos2']] = 4.0
    stack[b['pfo_pos3']] = 6.0
    assert np.allclose(E79.position_matched_blank(stack, f), 4.0)


def test_stimulus_map_baselines():
    f = _frames(11)
    b = E79.blank_frames(f)
    stack = np.zeros((len(f), 6, 7))
    stack[b['first_in_block']] = 1.0
    stack[b['pfo_pos2']] = 2.0
    stack[b['pfo_pos3']] = 3.0
    odor = f[(f['odor'] != 'pfo')]
    stack[odor.index] = 10.0
    om = E79.stimulus_maps(stack, f, 'order_matched')
    fb = E79.stimulus_maps(stack, f, 'first_in_block')
    ap = E79.stimulus_maps(stack, f, 'all_pfo')
    k = ('ms', -2.0)
    assert len(om[k][1]) == 2
    # presentation at position 2 minus the paraffin oil block's 2nd (2), at position 3 minus its 3rd (3)
    assert np.allclose(om[k][1][0], 8.0) and np.allclose(om[k][1][1], 7.0) and np.allclose(om[k][0], 7.5)
    assert np.allclose(fb[k][0], 9.0)
    assert np.allclose(ap[k][0], 10.0 - (17 * 1.0 + 2.0 + 3.0) / 19)
    # the doubtful repeat (block 12) is left out: two presentations of block 2's stimulus remain
    k2 = tuple(f.loc[f['block'] == 2, ['odor', 'log10_conc']].iloc[1])
    assert len(om[(k2[0], float(k2[1]))][1]) == 2


def test_svd_shared():
    rng = np.random.default_rng(1)
    v = rng.random(50) + 1
    X = np.outer([1.0, 2.0, 0.5], v)
    share, v1 = E79.svd_shared(X, demean=False)
    assert share == pytest.approx(1.0)
    assert abs(abs(v1 @ v) / np.linalg.norm(v)) == pytest.approx(1.0)
    share_d, _ = E79.svd_shared(X + rng.normal(scale=0.01, size=X.shape), demean=True)
    assert share_d > 0.95
    # residuals of three maps, weighted by the first left singular vector, sum to zero
    Y = rng.random((3, 40)) + np.outer([1, 1, 1], v[:40])
    _, v1 = E79.svd_shared(Y, demean=False)
    R = np.stack([E79._residual(y, v1, False) for y in Y])
    u1 = np.linalg.svd(Y, full_matrices=False)[0][:, 0]
    assert np.allclose(u1 @ R, 0, atol=1e-9)


@pytest.mark.parametrize('sigma', [3.0, 6.0])
def test_autocorr_length_white_noise(sigma):
    rng = np.random.default_rng(2)
    img = gaussian_filter(rng.normal(size=(200, 220)), sigma)
    mask = np.zeros(img.shape, bool)
    mask[10:190, 10:210] = True
    # Gaussian-blurred white noise: autocorrelation exp(-r^2 / (4 sigma^2)), 1/e at 2 sigma
    assert E79.autocorr_length(img, mask) == pytest.approx(2 * sigma, rel=0.12)
    # independent noise added to two copies does not move the cross-correlation length
    a = img + rng.normal(scale=img.std(), size=img.shape)
    b = img + rng.normal(scale=img.std(), size=img.shape)
    assert E79.autocorr_length(a, mask, b) == pytest.approx(2 * sigma, rel=0.15)
    # but shortens the autocorrelation length of one copy: with noise of the signal's variance, the normalized
    # autocorrelation is exp(-r^2 / (4 sigma^2)) / 2 off zero lag, 1/e at 2 sigma sqrt(1 - ln 2)
    assert E79.autocorr_length(a, mask) == pytest.approx(2 * sigma * np.sqrt(1 - np.log(2)), rel=0.15)


def test_profile_shares():
    mask = np.zeros((40, 30), bool)
    mask[10:30, 5:25] = True
    m = np.zeros(mask.shape)
    m[10:12, 5:25] = 1.0
    py = E79.profile_shares(m, mask, 'y', n_bins=10, blur=0)
    assert py.sum() == pytest.approx(1.0) and py[0] == pytest.approx(1.0)
    px = E79.profile_shares(m, mask, 'x', n_bins=10, blur=0)
    assert np.allclose(px, 0.1)
    assert np.isnan(E79.profile_shares(-m, mask, 'y', blur=0)).all()


def test_matched_pair_and_bootstrap():
    means = {('pa', -5.0): 0.10, ('pa', -4.0): 0.30, ('ms', -3.0): 0.29, ('ms', -2.0): 0.11, ('ms', -6.0): 0.001}
    reliab = {k: 0.9 for k in means}
    # (pa -5, ms -2) has |log ratio| 0.095 and (pa -4, ms -3) 0.034: the latter wins; ms -6 is near zero and unreliable
    reliab[('ms', -6.0)] = 0.1
    assert E79.matched_pair(means, reliab, 'pa', 'ms') == (('pa', -4.0), ('ms', -3.0))
    reliab[('ms', -3.0)] = 0.2
    assert E79.matched_pair(means, reliab, 'pa', 'ms') == (('pa', -5.0), ('ms', -2.0))
    assert E79.matched_pair(means, {k: 0.0 for k in means}, 'pa', 'ms') is None
    x = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    mean, lo, hi = E79.fly_bootstrap(x, n_boot=2000)
    assert np.allclose(mean, [2.0, 0.0]) and 1.0 <= lo[0] < 2.0 < hi[0] <= 3.0 and lo[1] == hi[1] == 0.0
