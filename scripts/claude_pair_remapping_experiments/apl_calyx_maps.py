#!/usr/bin/env python3
"""Renders and summarizes Yang's single-plane APL calyx response maps (data/yang_apl_calyx_maps/ in the shared directory) for
the question whether APL's spatial response patterns can constrain the APL models (Tom, 2026-11-10 23:25). No model runs.

Per recording (a TIFF of 51 dF/F peak maps, one per stimulus presentation, its ImageJ calyx-outline ROI and frames.csv's
stimulus labels): each odor x concentration map is the mean of its presentations minus the mean of the recording's paraffin
oil presentations, with pixels outside the calyx ROI masked. Writes to results/apl_calyx_maps/ in the shared directory:
- maps_<recording>.png: the maps (rows ms, pa, paa; columns concentrations), raw dF/F on a common scale and shape-normalized
  (each map over its mean inside the calyx);
- magnitudes.csv: mean dF/F inside the calyx per recording, odor, concentration and presentation;
- correlations.csv: Pearson correlation over calyx pixels (after a Gaussian blur, sigma 2 px) between maps: the two
  presentations of one stimulus (the reliability), and every pair of stimuli within a recording;
- centroids.csv and grid.csv: the response-weighted centroid in coordinates normalized to the ROI's bounding box (0-1, image
  axes as recorded: x left to right, y top to bottom; the anatomical orientation is not in the files), and the share of the
  response in each cell of a 3 x 3 grid over that box;
- population.png: per odor, the mean over recordings of the shape-normalized maps resampled to the ROI's bounding box
  (48 x 48), for the strongest concentration with a reliable response, with right-hemisphere recordings shown as recorded and
  mirrored (the mirroring assumes left-right symmetry about the midline, unverified).
"""
from __future__ import annotations

import struct
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter, map_coordinates

SHARED = Path('/mnt/tensor/Tom/claude_kc_remapping')
DATA = SHARED / 'data' / 'yang_apl_calyx_maps' / '2021_APL_18um'
OUT = SHARED / 'results' / 'apl_calyx_maps'
ODORS = ['ms', 'pa', 'paa']
BLUR = 2.0
GRID_N = 48


def read_imagej_polygon(path: Path) -> np.ndarray:
    """Vertices (x, y) of an ImageJ polygon / freehand / traced ROI (integer coordinates), or the rectangle of others."""
    b = path.read_bytes()
    assert b[:4] == b'Iout', path
    roi_type = b[6]
    top, left, bottom, right, n = struct.unpack('>hhhhh', b[8:18])
    if roi_type in (0, 7, 8):
        xs = np.array(struct.unpack(f'>{n}h', b[64:64 + 2 * n]), float) + left
        ys = np.array(struct.unpack(f'>{n}h', b[64 + 2 * n:64 + 4 * n]), float) + top
        return np.c_[xs, ys]
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], float)


def roi_mask(poly: np.ndarray, shape) -> np.ndarray:
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return MplPath(poly).contains_points(np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]).reshape(shape)


def stimulus_maps(stack: np.ndarray, frames: pd.DataFrame):
    """(odor, conc) -> (mean map minus the paraffin oil mean, list of per-presentation maps minus it)."""
    blank = stack[frames.index[frames['odor'] == 'pfo']].mean(axis=0)
    out = {}
    for (odor, conc), g in frames[frames['odor'] != 'pfo'].groupby(['odor', 'log10_conc']):
        reps = [stack[i] - blank for i in g.index]
        out[(odor, float(conc))] = (np.mean(reps, axis=0), reps)
    return out


def corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a = gaussian_filter(a, BLUR)[mask]
    b = gaussian_filter(b, BLUR)[mask]
    return float(np.corrcoef(a, b)[0, 1])


def box_resample(m: np.ndarray, mask: np.ndarray, flip: bool) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    gy, gx = np.meshgrid(np.linspace(y0, y1, GRID_N), np.linspace(x0, x1, GRID_N), indexing='ij')
    mm = np.where(mask, m, np.nan)
    r = map_coordinates(np.nan_to_num(mm), [gy, gx], order=1)
    inside = map_coordinates(mask.astype(float), [gy, gx], order=1) > 0.5
    r[~inside] = np.nan
    return r[:, ::-1] if flip else r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = pd.read_csv(DATA / 'recordings.csv')
    frames_all = pd.read_csv(DATA / 'frames.csv')
    mags, cors, cents, grids, pop = [], [], [], [], {}
    for _, rec in recs.iterrows():
        name = rec['file'].split('_moco')[0]
        stack = tifffile.imread(DATA / rec['file']).astype(float)
        frames = frames_all[frames_all['recording'] == name].set_index('frame').sort_index()
        assert len(frames) == stack.shape[0], name
        mask = roi_mask(read_imagej_polygon(DATA / rec['roi']), stack.shape[1:])
        maps = stimulus_maps(stack, frames)
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
        for (odor, conc), (m, reps) in maps.items():
            for k, r in enumerate(reps):
                mags.append(dict(recording=name, side=rec['side'], odor=odor, conc=conc, presentation=k, mean_dff=float(r[mask].mean())))
            mags.append(dict(recording=name, side=rec['side'], odor=odor, conc=conc, presentation='mean', mean_dff=float(m[mask].mean())))
            if len(reps) >= 2:
                cors.append(dict(recording=name, kind='repeat', a=f'{odor} {conc}', b=f'{odor} {conc}', r=corr(reps[0], reps[1], mask)))
            w = np.clip(gaussian_filter(m, BLUR), 0, None) * mask
            if w.sum() > 0:
                cents.append(dict(recording=name, side=rec['side'], odor=odor, conc=conc, mean_dff=float(m[mask].mean()),
                                  cx=float(((xx * w).sum() / w.sum() - x0) / (x1 - x0)),
                                  cy=float(((yy * w).sum() / w.sum() - y0) / (y1 - y0))))
                g = {}
                for i in range(3):
                    for j in range(3):
                        cell = mask & (yy >= y0 + i * (y1 - y0) / 3) & (yy <= y0 + (i + 1) * (y1 - y0) / 3) \
                            & (xx >= x0 + j * (x1 - x0) / 3) & (xx <= x0 + (j + 1) * (x1 - x0) / 3)
                        g[f'share_r{i}c{j}'] = float(w[cell].sum() / w.sum())
                grids.append(dict(recording=name, side=rec['side'], odor=odor, conc=conc, **g))
        keys = sorted(maps)
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                cors.append(dict(recording=name, kind='same odor' if ka[0] == kb[0] else 'different odors',
                                 a=f'{ka[0]} {ka[1]}', b=f'{kb[0]} {kb[1]}', r=corr(maps[ka][0], maps[kb][0], mask)))
        # figure: rows odors, columns concentrations (low to high), raw and shape-normalized
        concs = {o: sorted(c for oo, c in keys if oo == o) for o in ODORS}
        ncol = max(len(v) for v in concs.values())
        vmax = np.nanpercentile(np.concatenate([maps[k][0][mask] for k in keys]), 99)
        fig, axes = plt.subplots(2 * len(ODORS), ncol, figsize=(2.0 * ncol, 2.0 * 2 * len(ODORS)), squeeze=False)
        for r, odor in enumerate(ODORS):
            for c in range(ncol):
                for half, ax in enumerate((axes[r, c], axes[len(ODORS) + r, c])):
                    ax.set_xticks([]); ax.set_yticks([])
                    if c >= len(concs[odor]):
                        ax.axis('off'); continue
                    m = maps[(odor, concs[odor][c])][0]
                    mm = np.where(mask, gaussian_filter(m, 1.0), np.nan)
                    if half == 0:
                        ax.imshow(mm, cmap='magma', vmin=0, vmax=vmax)
                        ax.set_title(f'{odor} {concs[odor][c]:g}\nmean {m[mask].mean():.2f}', fontsize=7)
                    else:
                        mu = m[mask].mean()
                        ax.imshow(mm / mu if mu > 0 else mm * np.nan, cmap='viridis', vmin=0, vmax=3)
                        ax.set_title(f'{odor} {concs[odor][c]:g}\n(over its mean)', fontsize=7)
                    ax.contour(mask, levels=[0.5], colors='w', linewidths=0.5)
        fig.suptitle(f'{name} ({rec["side"]}): APL dF/F peak maps minus paraffin oil (top, 0 to {vmax:.2f}); '
                     'each over its calyx mean (bottom, 0 to 3); image axes as recorded', fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f'maps_{name}.png', dpi=110)
        plt.close(fig)
        # population input: per odor, the strongest concentration's shape-normalized map
        for odor in ODORS:
            if not concs[odor]:
                continue
            k = (odor, concs[odor][-1])
            m = maps[k][0]
            mu = m[mask].mean()
            if mu <= 0:
                continue
            for flip_r in (False, True):
                pop.setdefault((odor, flip_r), []).append(box_resample(gaussian_filter(m, BLUR) / mu, mask, flip_r and rec['side'] == 'R'))
    pd.DataFrame(mags).to_csv(OUT / 'magnitudes.csv', index=False)
    pd.DataFrame(cors).to_csv(OUT / 'correlations.csv', index=False)
    pd.DataFrame(cents).to_csv(OUT / 'centroids.csv', index=False)
    pd.DataFrame(grids).to_csv(OUT / 'grid.csv', index=False)
    fig, axes = plt.subplots(2, len(ODORS), figsize=(3.2 * len(ODORS), 6.4), squeeze=False)
    for j, odor in enumerate(ODORS):
        for i, flip_r in enumerate((False, True)):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            ms = pop.get((odor, flip_r), [])
            if not ms:
                ax.axis('off'); continue
            ax.imshow(np.nanmean(ms, axis=0), cmap='viridis', vmin=0, vmax=2.5)
            ax.set_title(f'{odor}, strongest concentration, n={len(ms)}\n'
                         + ('right hemispheres mirrored' if flip_r else 'as recorded'), fontsize=8)
    fig.suptitle('Mean over recordings of each map over its calyx mean, resampled to the ROI bounding box', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'population.png', dpi=110)
    plt.close(fig)
    print(f'{len(recs)} recordings -> {OUT}')


if __name__ == '__main__':
    main()
