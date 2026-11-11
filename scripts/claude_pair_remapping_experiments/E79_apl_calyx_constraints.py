#!/usr/bin/env python3
"""The APL calyx maps against the connectome (E79), the data-side analyses 1, 2, 3 and 5 (analyses 4 and 6, the
connectome-predicted maps and glomerulus co-localization, are in E79_connectome_maps.py). No model fits.

Inputs: Yang's 2021 single-plane APL calyx maps (data/yang_apl_calyx_maps/2021_APL_18um/ in the shared directory: per
recording a TIFF of 51 dF/F peak maps, one per presentation, in 17 blocks of three: paraffin oil first, then an odor x
concentration twice; one block, the last in 11 of 12 recordings and block 11 in 20210827_3FRfun4, is three paraffin oil
presentations), the ImageJ calyx-outline ROIs, frames.csv and
recordings.csv; the ROI reader and mask of apl_calyx_maps.py (atlas's first pass, imported, not changed); for analysis 5
the local APL-unit model's saved unit potentials (results/E2_local_apl/unit_potentials_*_coupling*.csv) and
`mb_model.load_apl_points()`.

Analyses (outputs to results/E79_apl_calyx_constraints/ in the shared directory):
1. Blank control: the position-matched blank (the paraffin oil block's 2nd and 3rd presentations minus the mean of
   every first-in-block paraffin oil presentation), its calyx mean and its Pearson r (2 px blur) with each odor map; every
   odor x concentration map re-made with an order-matched baseline (odor presentations are 2nd and 3rd in their block, so
   each presentation minus the paraffin oil block's presentation at the same position) and with the first-in-block
   baseline, and atlas's correlations, centroids and 3 x 3 grid shares redone with both.
2. The shared and odor-specific parts per recording, at Gaussian blur 2, 5, 10 and 20 px: the share of the sum of squares in
   the first singular vector of the stimulus x pixel matrix (maps inside the calyx; means kept, and each map's mean
   removed), and the residual after removing it: its repeat reliability, between-odor r and spatial autocorrelation length.
3. The dorsal-ventral profile: response share in 10 bins along image y (and x) inside the ROI's bounding box; pa minus ms and
   paa minus ms at matched calyx mean dF/F, per fly (hemispheres averaged), with a bootstrap 90 % interval over flies.
5. The local APL-unit model's maps (E2): posterior-view projections of the unit potentials for pa, ms and paa at each saved
   coupling, with the statistics of 2 and 3, against the measured values.

Image axes are as recorded (x left to right, y top to bottom); the anatomical orientation and pixel size are not in the files
(Yang to confirm), so lengths are given in pixels and in um at an assumed 0.3 um per pixel.

Run from this folder: `nice -n 10 ../../venv/bin/python E79_apl_calyx_constraints.py [--analyses 1,2,3,5]`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, gaussian_filter

import apl_calyx_maps as ACM

HERE = Path(__file__).resolve().parent
SHARED = ACM.SHARED
DATA = ACM.DATA
OUT = SHARED / 'results' / 'E79_apl_calyx_constraints'
E2_RESULTS = SHARED / 'results' / 'E2_local_apl'
ODORS = ACM.ODORS
BLURS_PX: Tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)
CORR_BLUR_PX = 2.0
N_PROFILE_BINS = 10
ASSUMED_UM_PER_PX = 0.3
N_BOOT = 10000
SEED = 0
BASELINES: Tuple[str, ...] = ('all_pfo', 'order_matched', 'first_in_block')
MAIN_BASELINE = 'order_matched'
MIN_MATCH_RELIABILITY = 0.5
# the E2 model's inputs (the APL16 panel's odors as E2 ran them: megamat pa and ms at -3, validation2 paa at -4)
MODEL_ODOR_ROWS: Dict[str, Tuple[str, float]] = {'pa': ('pa @ -3', -3.0), 'ms': ('ms @ -3', -3.0), 'paa': ('paa @ -4', -4.0)}
MODEL_PIXEL_UM = 0.5
MODEL_RENDER_UM = 1.0
SLAB_UM = 6.0
SLAB_FRACS: Dict[str, float] = {'slab_posterior': 0.25, 'slab_middle': 0.5, 'slab_anterior': 0.75}

Stim = Tuple[str, float]


@dataclass
class Recording:
    """One recording: its stack of 51 dF/F maps, frames.csv rows (indexed by frame) and calyx mask."""
    name: str
    side: str
    fly: str
    stack: np.ndarray
    frames: pd.DataFrame
    mask: np.ndarray


def stim_label(k: Stim) -> str:
    return f'{k[0]} {k[1]:g}'


def load_recordings() -> List[Recording]:
    """Reads every recording of recordings.csv (the TIFF, its frames.csv rows with the presentation's position in its
    block, and the calyx mask from its ImageJ ROI)."""
    import tifffile
    recs = pd.read_csv(DATA / 'recordings.csv')
    frames_all = pd.read_csv(DATA / 'frames.csv')
    out = []
    for _, rec in recs.iterrows():
        name = rec['file'].split('_moco')[0]
        stack = tifffile.imread(DATA / rec['file']).astype(float)
        frames = add_positions(frames_all[frames_all['recording'] == name].set_index('frame').sort_index())
        assert len(frames) == stack.shape[0], name
        mask = ACM.roi_mask(ACM.read_imagej_polygon(DATA / rec['roi']), stack.shape[1:])
        out.append(Recording(name=name, side=rec['side'], fly=f"{rec['date']}_fly{rec['fly']}", stack=stack, frames=frames,
            mask=mask))
    return out


def add_positions(frames: pd.DataFrame) -> pd.DataFrame:
    """frames.csv rows of one recording with `position` (1-3, the presentation's place in its block), `pfo_block` (the
    block of three paraffin oil presentations: the last block, except block 11 in 20210827_3FRfun4) and
    `doubtful` (a presentation whose odor x concentration was already presented in an earlier block, as pa -6.771 in
    blocks 2 and 12 of 20210805_3FRfun1 and 20210823_4FRfun2, where the yaml's pin 43 in block 12 is probably a typo for 53,
    ms -2: kept out of the stimulus maps and checked separately)."""
    f = frames.sort_index().copy()
    f['position'] = f.groupby('block').cumcount() + 1
    all_pfo = f.groupby('block')['odor'].transform(lambda s: bool((s == 'pfo').all()))
    f['pfo_block'] = all_pfo.astype(bool)
    first_block = f[f['odor'] != 'pfo'].groupby(['odor', 'log10_conc'])['block'].transform('min')
    f['doubtful'] = False
    f.loc[first_block.index, 'doubtful'] = (f.loc[first_block.index, 'block'] != first_block).to_numpy()
    return f


def blank_frames(frames: pd.DataFrame) -> Dict[str, List[int]]:
    """Frame indices of the paraffin oil presentations: 'first_in_block' (position 1 of every block, the paraffin oil
    block's included), 'pfo_pos2' / 'pfo_pos3' (the paraffin oil block's 2nd and 3rd) and 'all' (every one)."""
    pfo = frames[frames['odor'] == 'pfo']
    blk = pfo[pfo['pfo_block']]
    out = dict(first_in_block=list(pfo.index[pfo['position'] == 1]), pfo_pos2=list(blk.index[blk['position'] == 2]),
        pfo_pos3=list(blk.index[blk['position'] == 3]), all=list(pfo.index))
    assert len(out['pfo_pos2']) == 1 and len(out['pfo_pos3']) == 1, 'not exactly one block of three paraffin oil presentations'
    return out


def position_matched_blank(stack: np.ndarray, frames: pd.DataFrame) -> np.ndarray:
    """The paraffin oil block's 2nd and 3rd maps' mean minus the mean of every first-in-block paraffin oil map."""
    b = blank_frames(frames)
    return stack[b['pfo_pos2'] + b['pfo_pos3']].mean(axis=0) - stack[b['first_in_block']].mean(axis=0)


def stimulus_maps(stack: np.ndarray, frames: pd.DataFrame, baseline: str) -> Dict[Stim, Tuple[np.ndarray, List[np.ndarray]]]:
    """(odor, log10 conc) -> (mean map, per-presentation maps), each presentation minus its baseline:
    'all_pfo' (atlas's: the mean of every paraffin oil map), 'order_matched' (the paraffin oil block's map at the
    presentation's own position, 2nd or 3rd, so the two presentations' baselines are independent) or 'first_in_block'
    (the mean of every first-in-block paraffin oil map). Doubtful presentations (`add_positions`) are left out."""
    b = blank_frames(frames)
    if baseline == 'all_pfo':
        common = stack[b['all']].mean(axis=0)
    elif baseline == 'first_in_block':
        common = stack[b['first_in_block']].mean(axis=0)
    elif baseline == 'order_matched':
        common = None
        by_pos = {2: stack[b['pfo_pos2'][0]], 3: stack[b['pfo_pos3'][0]]}
    else:
        raise ValueError(f'unknown baseline {baseline!r}')
    odor = frames[(frames['odor'] != 'pfo') & ~frames['doubtful']]
    out = {}
    for (o, c), g in odor.groupby(['odor', 'log10_conc']):
        reps = []
        for i, pos in zip(g.index, g['position']):
            if common is not None:
                reps.append(stack[i] - common)
            else:
                assert pos in by_pos, f'odor presentation at position {pos}'
                reps.append(stack[i] - by_pos[pos])
        out[(o, float(c))] = (np.mean(reps, axis=0), reps)
    return dict(sorted(out.items()))


def blurred_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray, blur: float = CORR_BLUR_PX) -> float:
    """Pearson r over calyx pixels after a Gaussian blur (sigma `blur` px)."""
    return float(np.corrcoef(gaussian_filter(a, blur)[mask], gaussian_filter(b, blur)[mask])[0, 1])


def top_concs(maps: Dict[Stim, tuple], n: int = 1) -> Dict[str, List[float]]:
    """Each odor's `n` highest concentrations among the maps."""
    return {o: sorted(c for oo, c in maps if oo == o)[-n:] for o in ODORS if any(oo == o for oo, _ in maps)}


def centroid_grid(m: np.ndarray, mask: np.ndarray, blur: float = CORR_BLUR_PX) -> Optional[dict]:
    """atlas's response-weighted centroid (0-1 in the ROI's bounding box, image axes) and 3 x 3 grid shares, of the
    blurred map clipped at 0."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    w = np.clip(gaussian_filter(m, blur), 0, None) * mask
    if not w.sum() > 0:
        return None
    d = dict(cx=float(((xx * w).sum() / w.sum() - x0) / (x1 - x0)), cy=float(((yy * w).sum() / w.sum() - y0) / (y1 - y0)))
    for i in range(3):
        for j in range(3):
            cell = mask & (yy >= y0 + i * (y1 - y0) / 3) & (yy <= y0 + (i + 1) * (y1 - y0) / 3) \
                & (xx >= x0 + j * (x1 - x0) / 3) & (xx <= x0 + (j + 1) * (x1 - x0) / 3)
            d[f'share_r{i}c{j}'] = float(w[cell].sum() / w.sum())
    return d


def _sd_in(m: np.ndarray, mask: np.ndarray, blur: float = CORR_BLUR_PX) -> float:
    return float(gaussian_filter(m, blur)[mask].std())


def analysis_blank(recs: Sequence[Recording], out: Path) -> Dict[str, pd.DataFrame]:
    """Analysis 1: the blank's calyx mean and r with each odor map, the baselines' noise, and correlations, centroids and
    3 x 3 grid shares under each baseline. Writes E79_blank_*.csv, E79_baseline_*.csv and a figure."""
    blank_rows, noise_rows, cor_rows, cent_rows, grid_rows, cross_rows, doubt_rows = [], [], [], [], [], [], []
    for rec in recs:
        st, fr, mask = rec.stack, rec.frames, rec.mask
        bf = blank_frames(fr)
        blank = position_matched_blank(st, fr)
        first_mean = st[bf['first_in_block']].mean(axis=0)
        halves = [st[bf['pfo_pos2'][0]] - first_mean, st[bf['pfo_pos3'][0]] - first_mean]
        maps = {bl: stimulus_maps(st, fr, bl) for bl in BASELINES}
        top = top_concs(maps['first_in_block'])
        top2 = top_concs(maps['first_in_block'], 2)
        strong = [(o, c) for o, cs in top2.items() for c in cs]
        for k, (m, _) in maps['first_in_block'].items():
            blank_rows.append(dict(recording=rec.name, side=rec.side, fly=rec.fly, stimulus=stim_label(k), odor=k[0], conc=k[1],
                top_conc=k[1] == top[k[0]][-1], blank_mean=float(blank[mask].mean()), blank_sd=_sd_in(blank, mask),
                odor_mean_first_in_block=float(m[mask].mean()), odor_sd_first_in_block=_sd_in(m, mask),
                r_blank_vs_odor_first_in_block=blurred_corr(blank, m, mask),
                r_blank_vs_odor_all_pfo=blurred_corr(blank, maps['all_pfo'][k][0], mask)))
        # baseline noise: the single-frame noise from the paraffin oil block's 2nd and 3rd maps, the presentation
        # noise from the odors' two presentations, and the first-in-block frames' scatter around their mean
        frame_noise = _sd_in(st[bf['pfo_pos2'][0]] - st[bf['pfo_pos3'][0]], mask) / np.sqrt(2)
        rep_noise = float(np.median([_sd_in(r[0] - r[1], mask) / np.sqrt(2) for _, r in maps['first_in_block'].values()]))
        dev = [gaussian_filter(st[i] - first_mean, CORR_BLUR_PX)[mask] for i in bf['first_in_block']]
        n_first = len(bf['first_in_block'])
        first_noise = float(np.sqrt(np.mean([d.var() for d in dev]) * n_first / (n_first - 1)))
        signal_sd = float(np.median([_sd_in(maps['first_in_block'][k][0], mask) for k in strong]))
        signal_mean = float(np.median([maps['first_in_block'][k][0][mask].mean() for k in strong]))
        noise_rows.append(dict(recording=rec.name, side=rec.side, pfo_block=int(fr.loc[bf['pfo_pos2'][0], 'block']),
            n_blocks=int(fr['block'].nunique()), n_first_in_block=n_first,
            frame_noise_sd_pfo_block=frame_noise, frame_noise_sd_odor_repeats=rep_noise, frame_noise_sd_first_in_block=first_noise,
            baseline_noise_sd_order_matched_mean=frame_noise / np.sqrt(2), baseline_noise_sd_first_in_block=first_noise / np.sqrt(n_first),
            strong_signal_sd=signal_sd, strong_signal_mean=signal_mean,
            noise_to_signal_order_matched=frame_noise / np.sqrt(2) / signal_sd,
            noise_to_signal_first_in_block=first_noise / np.sqrt(n_first) / signal_sd,
            blank_halves_r=blurred_corr(halves[0], halves[1], mask), blank_mean=float(blank[mask].mean()),
            blank_sd=_sd_in(blank, mask), pfo_first_in_block_mean=float(first_mean[mask].mean()),
            pfo_block_pos2_mean=float(st[bf['pfo_pos2'][0]][mask].mean()), pfo_block_pos3_mean=float(st[bf['pfo_pos3'][0]][mask].mean())))
        for bl, mp in maps.items():
            for k, (m, reps) in mp.items():
                cor_rows.append(dict(recording=rec.name, baseline=bl, kind='repeat', a=stim_label(k), b=stim_label(k),
                    a_top=k[1] == top[k[0]][-1], b_top=k[1] == top[k[0]][-1], r=blurred_corr(reps[0], reps[1], mask)))
                cg = centroid_grid(m, mask)
                if cg is not None:
                    base = dict(recording=rec.name, side=rec.side, baseline=bl, odor=k[0], conc=k[1], top_conc=k[1] == top[k[0]][-1],
                        mean_dff=float(m[mask].mean()))
                    cent_rows.append(dict(base, cx=cg['cx'], cy=cg['cy']))
                    grid_rows.append(dict(base, **{c: v for c, v in cg.items() if c.startswith('share')}))
            keys = list(mp)
            for i, ka in enumerate(keys):
                for kb in keys[i + 1:]:
                    cor_rows.append(dict(recording=rec.name, baseline=bl, kind='same odor' if ka[0] == kb[0] else 'different odors',
                        a=stim_label(ka), b=stim_label(kb), a_top=ka[1] == top[ka[0]][-1], b_top=kb[1] == top[kb[0]][-1],
                        r=blurred_corr(mp[ka][0], mp[kb][0], mask)))
        for k in maps['order_matched']:
            cross_rows.append(dict(recording=rec.name, stimulus=stim_label(k), top_conc=k[1] == top[k[0]][-1],
                r_order_matched_vs_all_pfo=blurred_corr(maps['order_matched'][k][0], maps['all_pfo'][k][0], mask),
                r_order_matched_vs_first_in_block=blurred_corr(maps['order_matched'][k][0], maps['first_in_block'][k][0], mask),
                mean_order_matched=float(maps['order_matched'][k][0][mask].mean()),
                mean_first_in_block=float(maps['first_in_block'][k][0][mask].mean()),
                mean_all_pfo=float(maps['all_pfo'][k][0][mask].mean())))
        # the doubtful presentations: which stimulus their map is most like
        dbt = fr[fr['doubtful']]
        if len(dbt):
            for (o, c, blk), g in dbt.groupby(['odor', 'log10_conc', 'block']):
                dm = np.mean([st[i] - first_mean for i in g.index], axis=0)
                rr = {stim_label(k): blurred_corr(dm, m, mask) for k, (m, _) in maps['first_in_block'].items()}
                best = max(rr, key=rr.get)
                doubt_rows.append(dict(recording=rec.name, label=f'{o} {c:g}', block=int(blk), mean_dff=float(dm[mask].mean()),
                    repeat_r=blurred_corr(st[g.index[0]] - first_mean, st[g.index[1]] - first_mean, mask),
                    most_similar=best, r_most_similar=rr[best],
                    r_same_label_first_block=rr[f'{o} {c:g}'],
                    r_best_ms=max(v for s, v in rr.items() if s.startswith('ms ')),
                    r_best_pa=max(v for s, v in rr.items() if s.startswith('pa ')),
                    r_best_paa=max(v for s, v in rr.items() if s.startswith('paa '))))
    tabs = dict(blank=pd.DataFrame(blank_rows), noise=pd.DataFrame(noise_rows), correlations=pd.DataFrame(cor_rows),
        centroids=pd.DataFrame(cent_rows), grid=pd.DataFrame(grid_rows), baseline_cross=pd.DataFrame(cross_rows),
        doubtful=pd.DataFrame(doubt_rows))
    tabs['blank'].to_csv(out / 'E79_blank_vs_odor_maps.csv', index=False)
    tabs['noise'].to_csv(out / 'E79_baseline_noise.csv', index=False)
    tabs['correlations'].to_csv(out / 'E79_correlations_by_baseline.csv', index=False)
    tabs['centroids'].to_csv(out / 'E79_centroids_by_baseline.csv', index=False)
    tabs['grid'].to_csv(out / 'E79_grid_by_baseline.csv', index=False)
    tabs['baseline_cross'].to_csv(out / 'E79_maps_across_baselines.csv', index=False)
    tabs['doubtful'].to_csv(out / 'E79_doubtful_presentations.csv', index=False)
    tabs['summary'] = blank_summary(tabs)
    tabs['summary'].to_csv(out / 'E79_baseline_summary.csv', index=False)
    plot_blank(recs, out)
    return tabs


def blank_summary(tabs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per baseline: atlas's headline numbers (median repeat r at each odor's top concentration, ms vs pa r at the top
    concentrations, same-odor and different-odor r, recordings with pa's centroid lower in the image than ms's)."""
    cor, cen = tabs['correlations'], tabs['centroids']
    rows = []
    for bl in BASELINES:
        c = cor[cor.baseline == bl]
        rep_top = c[(c.kind == 'repeat') & c.a_top]
        diff_top = c[(c.kind == 'different odors') & c.a_top & c.b_top]
        ms_pa = diff_top[diff_top.a.str.startswith('ms ') & diff_top.b.str.startswith('pa ')]
        e = cen[(cen.baseline == bl) & cen.top_conc].pivot_table(index='recording', columns='odor', values='cy')
        rows.append(dict(baseline=bl, repeat_r_top_median=rep_top.r.median(), repeat_r_top_min=rep_top.r.min(),
            repeat_r_all_median=c[c.kind == 'repeat'].r.median(),
            same_odor_r_median=c[c.kind == 'same odor'].r.median(), different_odor_r_median=c[c.kind == 'different odors'].r.median(),
            different_odor_r_top_median=diff_top.r.median(), ms_vs_pa_r_top_median=ms_pa.r.median(),
            ms_vs_pa_r_top_min=ms_pa.r.min(), ms_vs_pa_r_top_max=ms_pa.r.max(),
            n_recordings_ms_vs_pa_top_below_0p8=int((ms_pa.r < 0.8).sum()),
            n_recordings_pa_centroid_below_ms_top=int((e['pa'] > e['ms']).sum()),
            n_recordings_paa_centroid_below_ms_top=int((e['paa'] > e['ms']).sum()), n_recordings=int(len(e))))
    return pd.DataFrame(rows)


def plot_blank(recs: Sequence[Recording], out: Path) -> None:
    """E79_blank_control.png: per recording (columns) the position-matched blank and each odor's top-concentration map
    (order-matched baseline), on one scale per recording."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, len(recs), figsize=(1.6 * len(recs), 1.6 * 4 + 0.8), squeeze=False)
    for j, rec in enumerate(recs):
        blank = position_matched_blank(rec.stack, rec.frames)
        maps = stimulus_maps(rec.stack, rec.frames, MAIN_BASELINE)
        top = top_concs(maps)
        panels = [('blank', blank)] + [(f'{o} {top[o][-1]:g}', maps[(o, top[o][-1])][0]) for o in ODORS]
        vmax = max(np.percentile(gaussian_filter(m, CORR_BLUR_PX)[rec.mask], 99) for _, m in panels)
        for i, (lab, m) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(np.where(rec.mask, gaussian_filter(m, CORR_BLUR_PX), np.nan), cmap='magma', vmin=0, vmax=vmax)
            ax.contour(rec.mask, levels=[0.5], colors='w', linewidths=0.4)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{lab}\nmean {m[rec.mask].mean():.2f}', fontsize=6)
        axes[0, j].set_ylabel(f'{rec.name}\n({rec.side})', fontsize=6)
    fig.suptitle('Position-matched blank (paraffin oil at the 2nd and 3rd positions minus 1st) and each odor\'s top '
        'concentration (order-matched baseline); dF/F, 0 to the recording\'s 99th percentile; image axes as recorded',
        fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'E79_blank_control.png', dpi=110)
    plt.close(fig)


def svd_shared(X: np.ndarray, demean: bool) -> Tuple[float, np.ndarray]:
    """Share of the sum of squares in the first singular vector of the stimulus x pixel matrix `X` (each row's mean
    removed first if `demean`), and that unit pixel vector."""
    X = np.asarray(X, float)
    if demean:
        X = X - X.mean(axis=1, keepdims=True)
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    return float(s[0] ** 2 / (s ** 2).sum()), vt[0]


def _residual(x: np.ndarray, v: np.ndarray, demean: bool) -> np.ndarray:
    if demean:
        x = x - x.mean()
    return x - (x @ v) * v


def _radial_profile(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Radially averaged, mask-normalized cross-correlation (autocorrelation if `a` is `b`) of the mean-removed images
    inside `mask`, at integer lags 0 .. half the smaller side of the mask's bounding box; normalized to lag 0."""
    H, W = mask.shape
    shape = (2 * H, 2 * W)
    m = mask.astype(float)
    am = np.where(mask, a - a[mask].mean(), 0.0)
    bm = np.where(mask, b - b[mask].mean(), 0.0)
    fa, fb, fm = np.fft.rfft2(am, shape), np.fft.rfft2(bm, shape), np.fft.rfft2(m, shape)
    num = np.fft.fftshift(np.fft.irfft2(fa * np.conj(fb), shape))
    den = np.fft.fftshift(np.fft.irfft2(fm * np.conj(fm), shape))
    yy, xx = np.mgrid[-H:H, -W:W]
    r = np.rint(np.hypot(yy, xx)).astype(int)
    ys, xs = np.nonzero(mask)
    rmax = int(min(ys.max() - ys.min(), xs.max() - xs.min()) // 2)
    valid = (den > 0.25 * m.sum()) & (r <= rmax)
    c = np.where(valid, num / np.where(valid, den, 1.0), 0.0)
    sums = np.bincount(r[valid], weights=c[valid], minlength=rmax + 1)
    counts = np.bincount(r[valid], minlength=rmax + 1)
    prof = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)[:rmax + 1]
    return np.arange(rmax + 1), prof


def autocorr_length(img: np.ndarray, mask: np.ndarray, other: Optional[np.ndarray] = None) -> float:
    """Lag (px) at which the radially averaged, mask-normalized autocorrelation of `img` inside `mask` (or its
    cross-correlation with `other`, which leaves out noise independent between the two) falls to 1/e of its value at 0
    (linear interpolation between lags; NaN if it never does within half the mask's extent)."""
    lags, prof = _radial_profile(img, img if other is None else other, mask)
    if not prof[0] > 0:
        return float('nan')
    prof = prof / prof[0]
    thr = np.exp(-1)
    below = np.nonzero(prof <= thr)[0]
    if not len(below):
        return float('nan')
    i = below[0]
    if i == 0:
        return 0.0
    p0, p1 = prof[i - 1], prof[i]
    return float(lags[i - 1] + (p0 - thr) / (p0 - p1) * (lags[i] - lags[i - 1]))


def shared_stats(maps: Dict[Stim, Tuple[np.ndarray, List[np.ndarray]]], mask: np.ndarray, blur: float,
        resid_stims: Sequence[Stim], all_stims: Optional[Sequence[Stim]] = None, px_um: float = 1.0) -> List[dict]:
    """Analysis 2's statistics for one recording (or model) and blur: per mode (means kept / removed) the first singular
    vector's share over `all_stims`, and the residual's repeat r (where there are two presentations), between-odor and
    same-odor r and autocorrelation lengths over `resid_stims` (lengths times `px_um`)."""
    all_stims = list(maps) if all_stims is None else list(all_stims)
    bl = {k: gaussian_filter(maps[k][0], blur) for k in set(all_stims) | set(resid_stims)}
    X = np.stack([bl[k][mask] for k in all_stims])
    rows = []
    for demean in (False, True):
        share, v = svd_shared(X, demean)
        res = {k: _residual(bl[k][mask], v, demean) for k in resid_stims}
        img = {}
        for k, rv in res.items():
            im = np.zeros(mask.shape)
            im[mask] = rv
            img[k] = im
        rep_r, cross_len = [], []
        for k in resid_stims:
            reps = maps[k][1]
            if len(reps) >= 2:
                r0, r1 = (_residual(gaussian_filter(reps[i], blur)[mask], v, demean) for i in (0, 1))
                rep_r.append(float(np.corrcoef(r0, r1)[0, 1]))
                i0, i1 = np.zeros(mask.shape), np.zeros(mask.shape)
                i0[mask], i1[mask] = r0, r1
                cross_len.append(autocorr_length(i0, mask, i1))
        diff_r, same_r = [], []
        ks = list(resid_stims)
        for i, ka in enumerate(ks):
            for kb in ks[i + 1:]:
                rr = float(np.corrcoef(res[ka], res[kb])[0, 1])
                (same_r if ka[0] == kb[0] else diff_r).append(rr)
        raw_diff = [float(np.corrcoef(bl[ka][mask], bl[kb][mask])[0, 1]) for i, ka in enumerate(ks) for kb in ks[i + 1:]
            if ka[0] != kb[0]]
        auto_len = [autocorr_length(img[k], mask) for k in resid_stims]
        cv = [bl[k][mask].std() / bl[k][mask].mean() for k in ks if bl[k][mask].mean() > 0]
        rows.append(dict(blur_px=blur, mode='means removed' if demean else 'means kept', n_stimuli=len(all_stims),
            n_resid_stimuli=len(ks), share_first_sv=share, cv_median=float(np.median(cv)) if cv else np.nan,
            resid_ss_share=float(np.mean([(res[k] ** 2).sum() / ((bl[k][mask] - (bl[k][mask].mean() if demean else 0)) ** 2).sum()
                for k in ks])),
            resid_repeat_r_median=float(np.median(rep_r)) if rep_r else np.nan,
            resid_between_odor_r_median=float(np.median(diff_r)) if diff_r else np.nan,
            resid_between_odor_r_mean=float(np.mean(diff_r)) if diff_r else np.nan,
            resid_same_odor_r_median=float(np.median(same_r)) if same_r else np.nan,
            raw_between_odor_r_median=float(np.median(raw_diff)) if raw_diff else np.nan,
            resid_autocorr_len=float(np.nanmedian(auto_len)) * px_um if np.isfinite(auto_len).any() else np.nan,
            resid_crossrep_len=float(np.nanmedian(cross_len)) * px_um if cross_len and np.isfinite(cross_len).any() else np.nan,
            noise_floor_len=2 * blur * px_um))
    return rows


def model_matched_stims(maps: Dict[Stim, tuple]) -> List[Stim]:
    """For each odor, the concentration closest (in log10) to the E2 model's input (pa -3, ms -3, paa -4)."""
    out = []
    for o in ODORS:
        cs = [c for oo, c in maps if oo == o]
        out.append((o, min(cs, key=lambda c: (abs(c - MODEL_ODOR_ROWS[o][1]), -c))))
    return out


def analysis_shared(recs: Sequence[Recording], out: Path, baseline: str = MAIN_BASELINE) -> Dict[str, pd.DataFrame]:
    """Analysis 2 per recording and blur: the first singular vector's share (means kept / removed), the residual's
    repeat reliability, between-odor r and autocorrelation length. Writes E79_shared_*.csv and a figure."""
    rows = []
    for rec in recs:
        maps = stimulus_maps(rec.stack, rec.frames, baseline)
        top2 = top_concs(maps, 2)
        strong = [(o, c) for o, cs in top2.items() for c in cs]
        m3 = model_matched_stims(maps)
        for blur in BLURS_PX:
            for set_name, all_stims, resid in (('all stimuli (residuals: top two per odor)', list(maps), strong),
                    ('model-matched three (one per odor)', m3, m3)):
                ys, xs = np.nonzero(rec.mask)
                for r in shared_stats(maps, rec.mask, blur, resid, all_stims):
                    rows.append(dict(recording=rec.name, side=rec.side, fly=rec.fly, stimulus_set=set_name,
                        resid_stimuli='; '.join(stim_label(k) for k in resid), roi_width_px=int(xs.max() - xs.min() + 1),
                        roi_height_px=int(ys.max() - ys.min() + 1), **r))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'E79_shared_specific_per_recording.csv', index=False)
    num = [c for c in df.columns if c.startswith(('share', 'resid_', 'raw_', 'noise_', 'cv_')) and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(['stimulus_set', 'mode', 'blur_px'])[num]
    summ = g.median().add_suffix('_median').join(g.min().add_suffix('_min')).join(g.max().add_suffix('_max')).reset_index()
    summ.to_csv(out / 'E79_shared_specific_summary.csv', index=False)
    plot_shared(df, out)
    return dict(per_recording=df, summary=summ)


def plot_shared(df: pd.DataFrame, out: Path) -> None:
    """E79_shared_specific.png: per recording (lines) against blur, the first singular vector's share, the residual's
    repeat r and between-odor r, and its autocorrelation lengths."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = df[df.stimulus_set.str.startswith('all')]
    cols = [('share_first_sv', 'share of sum of squares\nin first singular vector'), ('resid_repeat_r_median', 'residual: repeat r'),
        ('resid_between_odor_r_median', 'residual: between-odor r'), ('resid_crossrep_len', 'residual: 1/e length (px)\n(cross-presentation)')]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.2 * len(cols), 5.6), squeeze=False)
    for i, mode in enumerate(('means kept', 'means removed')):
        dm = d[d['mode'] == mode]
        for j, (c, lab) in enumerate(cols):
            ax = axes[i, j]
            for _, g in dm.groupby('recording'):
                ax.plot(g.blur_px, g[c], color='0.6', lw=0.8, marker='o', ms=2)
            med = dm.groupby('blur_px')[c].median()
            ax.plot(med.index, med.values, color='k', lw=2, marker='o', label='median')
            if c == 'resid_crossrep_len':
                ax.plot(BLURS_PX, [2 * b for b in BLURS_PX], color='tab:red', ls=':', label='blurred white noise')
                ax.legend(fontsize=6)
            ax.set_xscale('log')
            ax.minorticks_off()
            ax.set_xticks(BLURS_PX)
            ax.set_xticklabels([f'{b:g}' for b in BLURS_PX])
            ax.set_xlabel('Gaussian blur sigma (px)')
            ax.set_ylabel(lab, fontsize=8)
            ax.set_title(mode, fontsize=8)
    fig.suptitle('Shared and odor-specific parts of the APL calyx maps, per recording (grey) and median (black); residual '
        'statistics over each odor\'s two highest concentrations', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'E79_shared_specific.png', dpi=110)
    plt.close(fig)


def profile_shares(m: np.ndarray, mask: np.ndarray, axis: str, n_bins: int = N_PROFILE_BINS,
        blur: float = CORR_BLUR_PX) -> np.ndarray:
    """Share of the (blurred, clipped at 0) response in `n_bins` equal bins along image 'y' (top to bottom) or 'x' (left
    to right) of the ROI's bounding box."""
    w = np.clip(gaussian_filter(m, blur) if blur else m, 0, None) * mask
    ys, xs = np.nonzero(mask)
    coord = (ys, ys.min(), ys.max()) if axis == 'y' else (xs, xs.min(), xs.max()) if axis == 'x' else None
    if coord is None:
        raise ValueError(f'axis must be x or y, not {axis!r}')
    c, lo, hi = coord
    b = np.clip(((c - lo) / (hi + 1 - lo) * n_bins).astype(int), 0, n_bins - 1)
    tot = w[ys, xs].sum()
    if not tot > 0:
        return np.full(n_bins, np.nan)
    return np.bincount(b, weights=w[ys, xs], minlength=n_bins) / tot


def matched_pair(means: Dict[Stim, float], reliab: Dict[Stim, float], odor_a: str, odor_b: str,
        min_reliability: float = MIN_MATCH_RELIABILITY) -> Optional[Tuple[Stim, Stim]]:
    """The (odor_a, odor_b) concentration pair whose calyx means are closest (smallest |log ratio|) among stimuli with a
    positive mean and repeat r >= `min_reliability`; None if there is none."""
    ok = {k for k, m in means.items() if m > 0 and reliab.get(k, -1) >= min_reliability}
    best, best_d = None, np.inf
    for ka in sorted(k for k in ok if k[0] == odor_a):
        for kb in sorted(k for k in ok if k[0] == odor_b):
            d = abs(np.log(means[ka] / means[kb]))
            if d < best_d - 1e-12 or (abs(d - best_d) <= 1e-12 and means[ka] + means[kb] > means[best[0]] + means[best[1]]):
                best, best_d = (ka, kb), d
    return best


def fly_bootstrap(per_fly: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean over flies (rows) and the 5th / 95th percentiles of the mean over `n_boot` resamplings of flies."""
    x = np.asarray(per_fly, float)
    if x.ndim == 1:
        x = x[:, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = x[idx].mean(axis=1)
    return x.mean(axis=0), np.percentile(boots, 5, axis=0), np.percentile(boots, 95, axis=0)


def analysis_profiles(recs: Sequence[Recording], out: Path, baseline: str = MAIN_BASELINE) -> Dict[str, pd.DataFrame]:
    """Analysis 3: per-stimulus profiles along y and x, the matched pa / ms and paa / ms pairs, their difference profiles
    per fly and the bootstrap intervals. Writes E79_profile_*.csv and a figure."""
    prof_rows, pair_rows = [], []
    diffs: Dict[Tuple[str, str], Dict[str, list]] = {}
    for rec in recs:
        maps = stimulus_maps(rec.stack, rec.frames, baseline)
        means = {k: float(m[rec.mask].mean()) for k, (m, _) in maps.items()}
        reliab = {k: blurred_corr(r[0], r[1], rec.mask) for k, (_, r) in maps.items()}
        prof = {}
        for k, (m, _) in maps.items():
            py, px = profile_shares(m, rec.mask, 'y'), profile_shares(m, rec.mask, 'x')
            # left hemispheres mirrored, so that x runs as in the right hemisphere (the hemibrain calyx of analysis 5)
            pxm = px[::-1] if rec.side == 'L' else px
            prof[k] = dict(y=py, x=px, x_mirrored=pxm)
            cg = centroid_grid(m, rec.mask)
            for ax, p in (('y', py), ('x', px), ('x_mirrored', pxm)):
                prof_rows.append(dict(recording=rec.name, side=rec.side, fly=rec.fly, odor=k[0], conc=k[1], mean_dff=means[k],
                    repeat_r=reliab[k], axis=ax, **{f'bin{i}': v for i, v in enumerate(p)},
                    centroid=(cg['cy'] if ax == 'y' else cg['cx'] if ax == 'x' else 1 - cg['cx'] if rec.side == 'L' else cg['cx'])
                    if cg else np.nan))
        for oa in ('pa', 'paa'):
            pr = matched_pair(means, reliab, oa, 'ms')
            if pr is None:
                pair_rows.append(dict(recording=rec.name, side=rec.side, fly=rec.fly, pair=f'{oa} - ms', matched=False))
                continue
            ka, kb = pr
            pair_rows.append(dict(recording=rec.name, side=rec.side, fly=rec.fly, pair=f'{oa} - ms', matched=True,
                conc_a=ka[1], conc_ms=kb[1], mean_a=means[ka], mean_ms=means[kb], log_ratio=float(np.log(means[ka] / means[kb])),
                repeat_r_a=reliab[ka], repeat_r_ms=reliab[kb],
                **{f'dcentroid_{ax}': float(np.sum((np.arange(N_PROFILE_BINS) + 0.5) / N_PROFILE_BINS * (prof[ka][ax] - prof[kb][ax])))
                    for ax in ('y', 'x', 'x_mirrored')}))
            for ax in ('y', 'x', 'x_mirrored'):
                diffs.setdefault((f'{oa} - ms', ax), {}).setdefault(rec.fly, []).append(prof[ka][ax] - prof[kb][ax])
    profiles = pd.DataFrame(prof_rows)
    pairs = pd.DataFrame(pair_rows)
    profiles.to_csv(out / 'E79_profile_per_stimulus.csv', index=False)
    pairs.to_csv(out / 'E79_profile_matched_pairs.csv', index=False)
    fly_rows, boot_rows = [], []
    for (pair, ax), by_fly in diffs.items():
        flies = sorted(by_fly)
        per_fly = np.stack([np.mean(by_fly[f], axis=0) for f in flies])
        for f, v in zip(flies, per_fly):
            fly_rows.append(dict(pair=pair, axis=ax, fly=f, n_hemispheres=len(by_fly[f]), **{f'bin{i}': x for i, x in enumerate(v)}))
        mean, lo, hi = fly_bootstrap(per_fly)
        for i in range(N_PROFILE_BINS):
            boot_rows.append(dict(pair=pair, axis=ax, bin=i, bin_center=(i + 0.5) / N_PROFILE_BINS, n_flies=len(flies),
                mean_diff=mean[i], lo90=lo[i], hi90=hi[i], n_flies_positive=int((per_fly[:, i] > 0).sum())))
    fly_df, boot_df = pd.DataFrame(fly_rows), pd.DataFrame(boot_rows)
    fly_df.to_csv(out / 'E79_profile_difference_per_fly.csv', index=False)
    boot_df.to_csv(out / 'E79_profile_difference_bootstrap.csv', index=False)
    # the centroid shifts per fly, with the bootstrap
    cent_rows = []
    pm = pairs[pairs.matched.astype(bool)] if len(pairs) else pairs
    for pair, g in pm.groupby('pair'):
        for ax in ('y', 'x', 'x_mirrored'):
            per_fly = g.groupby('fly')[f'dcentroid_{ax}'].mean()
            mean, lo, hi = fly_bootstrap(per_fly.to_numpy())
            cent_rows.append(dict(pair=pair, axis=ax, n_flies=len(per_fly), mean_shift=float(mean[0]), lo90=float(lo[0]),
                hi90=float(hi[0]), n_flies_positive=int((per_fly > 0).sum())))
    cent_df = pd.DataFrame(cent_rows)
    cent_df.to_csv(out / 'E79_profile_centroid_shift.csv', index=False)
    mag, adj, off = centroid_vs_magnitude(profiles)
    mag.to_csv(out / 'E79_profile_centroid_vs_magnitude.csv', index=False)
    adj.to_csv(out / 'E79_profile_centroid_adjusted.csv', index=False)
    off.to_csv(out / 'E79_profile_centroid_adjusted_per_recording.csv', index=False)
    plot_profiles(fly_df, boot_df, out)
    return dict(profiles=profiles, pairs=pairs, per_fly=fly_df, bootstrap=boot_df, centroid_shift=cent_df, magnitude=mag,
        adjusted=adj)


def centroid_vs_magnitude(profiles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """How each odor's y and x centroid moves with its calyx mean across concentrations (per recording and odor: Spearman
    and the slope on log mean, over maps with a positive mean and repeat r >= MIN_MATCH_RELIABILITY), and the odor
    offsets at equal magnitude: per recording, least squares of the centroid on log mean (one slope) and odor (ms the
    reference) over those maps; pa and paa's offsets per fly (hemispheres averaged) with the bootstrap 90 % interval.
    Returns the per-odor table, the summary and the per-recording fits."""
    from scipy.stats import spearmanr
    p = profiles[(profiles.repeat_r >= MIN_MATCH_RELIABILITY) & (profiles.mean_dff > 0)]
    mag_rows, off_rows = [], []
    for ax in ('y', 'x_mirrored'):
        pa_ = p[p.axis == ax]
        for (rec, odor), g in pa_.groupby(['recording', 'odor']):
            if len(g) >= 3:
                mag_rows.append(dict(axis=ax, recording=rec, odor=odor, n_maps=len(g), spearman=float(spearmanr(g.mean_dff, g.centroid)[0]),
                    slope_per_log_mean=float(np.polyfit(np.log(g.mean_dff), g.centroid, 1)[0])))
        for rec, g in pa_.groupby('recording'):
            if not all((g.odor == o).sum() >= 1 for o in ODORS) or len(g) < 5:
                continue
            X = np.c_[np.ones(len(g)), np.log(g.mean_dff), (g.odor == 'pa').astype(float), (g.odor == 'paa').astype(float)]
            beta = np.linalg.lstsq(X, g.centroid.to_numpy(), rcond=None)[0]
            off_rows.append(dict(axis=ax, recording=rec, fly=g.fly.iloc[0], n_maps=len(g), slope_per_log_mean=beta[1],
                offset_pa_minus_ms=beta[2], offset_paa_minus_ms=beta[3]))
    mag, off = pd.DataFrame(mag_rows), pd.DataFrame(off_rows)
    rows = []
    for ax, g in off.groupby('axis'):
        for col in ('slope_per_log_mean', 'offset_pa_minus_ms', 'offset_paa_minus_ms'):
            per_fly = g.groupby('fly')[col].mean()
            mean, lo, hi = fly_bootstrap(per_fly.to_numpy())
            rows.append(dict(axis=ax, quantity=col, n_flies=len(per_fly), mean=float(mean[0]), lo90=float(lo[0]), hi90=float(hi[0]),
                n_flies_positive=int((per_fly > 0).sum())))
    for (ax, odor), g in mag.groupby(['axis', 'odor']):
        rows.append(dict(axis=ax, quantity=f'spearman centroid vs calyx mean, {odor} (median over recordings)', n_flies=np.nan,
            mean=float(g.spearman.median()), lo90=float(g.spearman.min()), hi90=float(g.spearman.max()),
            n_flies_positive=int((g.spearman > 0).sum())))
    return mag, pd.DataFrame(rows), off


def plot_profiles(fly_df: pd.DataFrame, boot_df: pd.DataFrame, out: Path,
        model: Optional[pd.DataFrame] = None) -> None:
    """E79_profiles.png: pa minus ms and paa minus ms response share along image y and x (left hemispheres mirrored),
    per fly (grey), mean over flies with the bootstrap 90 % interval (black); the unit model's profiles if given."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    axes_ = [('y', 'image y (top to bottom)'), ('x_mirrored', 'image x (left hemispheres mirrored)')]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), squeeze=False)
    centers = (np.arange(N_PROFILE_BINS) + 0.5) / N_PROFILE_BINS
    for i, pair in enumerate(('pa - ms', 'paa - ms')):
        for j, (ax_name, lab) in enumerate(axes_):
            ax = axes[i, j]
            f = fly_df[(fly_df.pair == pair) & (fly_df.axis == ax_name)]
            for _, r in f.iterrows():
                ax.plot(centers, [r[f'bin{k}'] for k in range(N_PROFILE_BINS)], color='0.7', lw=0.8)
            b = boot_df[(boot_df.pair == pair) & (boot_df.axis == ax_name)].sort_values('bin')
            if len(b):
                ax.fill_between(b.bin_center, b.lo90, b.hi90, color='k', alpha=0.15, lw=0)
                ax.plot(b.bin_center, b.mean_diff, color='k', lw=2, label=f'mean over {int(b.n_flies.iloc[0])} flies, 90 % interval')
            if model is not None:
                for _, r in model[(model.pair == pair) & (model.axis == ax_name)].iterrows():
                    ax.plot(centers, [r[f'bin{k}'] for k in range(N_PROFILE_BINS)], lw=1, ls='--',
                        label=f"unit model, coupling {r['coupling']:g} ({'dorsal to ventral' if ax_name == 'y' else 'medial to lateral'})")
            ax.axhline(0, color='0.4', lw=0.5)
            ax.set_xlabel(f'position along {lab},\n0-1 in the ROI bounding box', fontsize=8)
            ax.set_ylabel(f'{pair}: response share per bin', fontsize=8)
            ax.legend(fontsize=6)
    fig.suptitle('APL calyx response profiles at matched calyx mean dF/F (order-matched baseline); anatomical orientation '
        'of the images unknown\n(the unit model\'s posterior-view projection drawn as if image top = dorsal and image x = '
        'medial to lateral of a right calyx)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'E79_profiles.png', dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# analysis 5: the local APL-unit model's maps
def calyx_apl_nodes() -> pd.DataFrame:
    """`load_apl_points()` restricted to nodes whose synapses are mostly in CA(R), the hemibrain calyx."""
    from al_analysis.mb_model import apl_synapses_path, load_apl_points
    pts = load_apl_points()
    syns = pd.read_parquet(apl_synapses_path, columns=['node_id', 'roi'])
    n = syns.dropna(subset=['roi']).groupby(['node_id', 'roi']).size().rename('n').reset_index()
    roi = n.sort_values('n', ascending=False, kind='stable').drop_duplicates('node_id').set_index('node_id')['roi']
    pts = pts.join(roi.rename('roi'))
    return pts[pts['roi'] == 'CA(R)']


def axes_check() -> pd.DataFrame:
    """The posterior-view convention checked against landmarks: the median position of APL synapses in each hemibrain ROI
    relative to the calyx's, in `load_apl_points` coordinates (+right, +anterior, +dorsal), with the direction anatomy
    expects (PLP lateral and ventral of the calyx, PED anterior and ventral, SLP anterior and dorsal; for the right
    hemisphere lateral is +right), and Yang's Claude's hemibrain posterior view (view_right_um, view_dorsal_um) against
    (-x, -z) of the native voxels."""
    from al_analysis.mb_model import apl_point_coord_cols, apl_synapses_path, hemibrain_voxel_um
    s = pd.read_parquet(apl_synapses_path)
    c = s[['x', 'y', 'z']].to_numpy() * hemibrain_voxel_um * np.array([-1, 1, -1])
    d = pd.DataFrame(c, columns=apl_point_coord_cols)
    d['roi'] = s['roi'].to_numpy()
    med = d.groupby('roi')[apl_point_coord_cols].median()
    rel = (med - med.loc['CA(R)']).drop('CA(R)')
    rel['n_synapses'] = d.groupby('roi').size()
    expect = {'PLP(R)': dict(right_um=1, dorsal_um=-1), 'PED(R)': dict(anterior_um=1, dorsal_um=-1),
        'SLP(R)': dict(anterior_um=1, dorsal_um=1), 'ICL(R)': dict(anterior_um=1, dorsal_um=-1)}
    rel['expected'] = [', '.join(f"{'+' if v > 0 else '-'}{k.split('_')[0]}" for k, v in expect.get(r, {}).items()) for r in rel.index]
    rel['as_expected'] = [all(np.sign(rel.loc[r, k]) == v for k, v in expect[r].items()) if r in expect else np.nan for r in rel.index]
    rel = rel.reset_index()
    yc = Path('/mnt/tensor/Tom/claude_APL_modeling_YZ4TC/Claude_APL_analysis/apl_synapses/odor_weighted/'
        'PN_APL_synapses_odor_weighted.csv')
    if yc.exists():
        y = pd.read_csv(yc, usecols=['dataset', 'x_nm', 'z_nm', 'view_right_um', 'view_dorsal_um'])
        y = y[y.dataset == 'hemibrain']
        if len(y):
            r_right = float(np.corrcoef(-y.x_nm, y.view_right_um)[0, 1])
            r_dorsal = float(np.corrcoef(-y.z_nm, y.view_dorsal_um)[0, 1])
            rel = pd.concat([rel, pd.DataFrame([dict(roi="Yang's Claude hemibrain view vs (-x, -z)", right_um=r_right,
                dorsal_um=r_dorsal, n_synapses=len(y), expected='r = +1 on both', as_expected=r_right > 0.999 and r_dorsal > 0.999)])])
    return rel


def _render(nodes_xy: np.ndarray, values: Dict[str, np.ndarray], extent: Tuple[float, float, float, float],
        pixel_um: float, render_um: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Normalized-convolution images (rows dorsal to ventral, columns left to right in the posterior view) of node values,
    and the mask of pixels inside the filled outline of the node density."""
    r0, r1, d0, d1 = extent
    nx = int(np.ceil((r1 - r0) / pixel_um))
    ny = int(np.ceil((d1 - d0) / pixel_um))
    col = np.clip(((nodes_xy[:, 0] - r0) / pixel_um).astype(int), 0, nx - 1)
    row = np.clip(((d1 - nodes_xy[:, 1]) / pixel_um).astype(int), 0, ny - 1)
    flat = row * nx + col
    counts = np.bincount(flat, minlength=nx * ny).reshape(ny, nx).astype(float)
    s = render_um / pixel_um
    den = gaussian_filter(counts, s)
    dens2 = gaussian_filter(counts, 2.0 / pixel_um)
    mask = binary_fill_holes(dens2 >= 0.1 * np.percentile(dens2[dens2 > 0], 99))
    out = {}
    for k, v in values.items():
        num = gaussian_filter(np.bincount(flat, weights=v, minlength=nx * ny).reshape(ny, nx), s)
        out[k] = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return out, mask


def model_maps(potentials: pd.DataFrame, odors: Sequence[str], pixel_um: float = MODEL_PIXEL_UM,
        render_um: float = MODEL_RENDER_UM, slab: Optional[Tuple[float, float]] = None,
        nodes: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Posterior-view maps (fly's right to the right, dorsal up; rows from dorsal to ventral like an image) of the APL
    units' potentials (`potentials`: odor x APL unit): every calyx APL node of `load_apl_points()` takes its unit's
    potential, and each pixel is the Gaussian-weighted mean over nodes (normalized convolution, sigma `render_um`),
    optionally only nodes within `slab` (anterior um). Returns odor -> map and the mask of the calyx outline (pixels with
    nodes nearby, holes filled)."""
    from al_analysis.mb_model import APL_UNIT
    nodes = calyx_apl_nodes() if nodes is None else nodes
    extent = (nodes.right_um.min() - 3, nodes.right_um.max() + 3, nodes.dorsal_um.min() - 3, nodes.dorsal_um.max() + 3)
    if slab is not None:
        nodes = nodes[(nodes.anterior_um >= slab[0]) & (nodes.anterior_um < slab[1])]
    pot = potentials.copy()
    pot.columns = pot.columns.astype(int)
    vals = {o: pot.loc[o].reindex(nodes[APL_UNIT].to_numpy()).to_numpy(float) for o in odors}
    assert all(np.isfinite(v).all() for v in vals.values()), 'nodes whose unit has no potential'
    return _render(nodes[['right_um', 'dorsal_um']].to_numpy(), vals, extent, pixel_um, render_um)


def _masked_blur(m: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    num = gaussian_filter(np.where(mask, m, 0.0), sigma)
    den = gaussian_filter(mask.astype(float), sigma)
    return np.where(mask, num / np.where(den > 0, den, 1.0), 0.0)


def load_unit_potentials() -> Dict[Tuple[str, float], pd.DataFrame]:
    """(base, coupling) -> odor x APL unit mean potentials from E2's saved unit_potentials_<base>_coupling<c>.csv."""
    out = {}
    for p in sorted(E2_RESULTS.glob('unit_potentials_*_coupling*.csv')):
        m = re.match(r'unit_potentials_(.+)_coupling([0-9.]+)\.csv$', p.name)
        if m:
            df = pd.read_csv(p).drop(columns='panel').set_index('odor')
            out[(m.group(1), float(m.group(2)))] = df
    return dict(sorted(out.items(), key=lambda kv: (kv[0][0], kv[0][1])))


def analysis_model(measured: Dict[str, Dict[str, pd.DataFrame]], out: Path) -> Dict[str, pd.DataFrame]:
    """Analysis 5: the statistics of analyses 2 and 3 on the E2 unit model's maps of pa, ms and paa at each saved coupling,
    beside the measured values. Writes E79_model_*.csv and a figure."""
    pots = load_unit_potentials()
    if not pots:
        raise FileNotFoundError(f'no unit_potentials_*_coupling*.csv in {E2_RESULTS}')
    nodes = calyx_apl_nodes()
    ax_check = axes_check()
    ax_check.to_csv(out / 'E79_model_axes_check.csv', index=False)
    a0, a1 = nodes.anterior_um.min(), nodes.anterior_um.max()
    views = {'projection': None}
    for name, f in SLAB_FRACS.items():
        c = a0 + f * (a1 - a0)
        views[name] = (c - SLAB_UM / 2, c + SLAB_UM / 2)
    rows, prof_rows, map_store = [], [], {}
    for (base, coupling), pot in pots.items():
        p = pot.loc[[MODEL_ODOR_ROWS[o][0] for o in ODORS]]
        p.index = list(ODORS)
        for view, slab in views.items():
            maps, mask = model_maps(p, ODORS, slab=slab, nodes=nodes)
            map_store[(base, coupling, view)] = (maps, mask)
            ys, xs = np.nonzero(mask)
            for blur_px in BLURS_PX:
                blur_um = blur_px * ASSUMED_UM_PER_PX
                bl = {o: _masked_blur(maps[o], mask, blur_um / MODEL_PIXEL_UM) for o in ODORS}
                # shared_stats blurs again with gaussian_filter; pass the masked-blurred maps with blur 0
                st_maps = {(o, MODEL_ODOR_ROWS[o][1]): (bl[o], [bl[o]]) for o in ODORS}
                stims = list(st_maps)
                for r in shared_stats(st_maps, mask, 0.0, stims, stims, px_um=MODEL_PIXEL_UM):
                    r = dict(r, blur_px=blur_px, blur_um=blur_um, noise_floor_len=np.nan)
                    rows.append(dict(base=base, coupling=coupling, view=view, n_nodes=int(len(nodes) if slab is None else
                        ((nodes.anterior_um >= slab[0]) & (nodes.anterior_um < slab[1])).sum()),
                        mask_width_um=float((xs.max() - xs.min() + 1) * MODEL_PIXEL_UM),
                        mask_height_um=float((ys.max() - ys.min() + 1) * MODEL_PIXEL_UM),
                        **{f'mean_potential_{o}': float(maps[o][mask].mean()) for o in ODORS},
                        **{f'cv_{o}': float(maps[o][mask].std() / maps[o][mask].mean()) for o in ODORS},
                        **{f'resid_r_{a}_{b}': _pair_resid_r(bl, mask, a, b, r['mode']) for a, b in (('pa', 'ms'), ('paa', 'ms'), ('pa', 'paa'))},
                        **r))
            # profiles at the measured profiles' blur (2 px at the assumed scale)
            pr = {o: {ax: profile_shares(_masked_blur(maps[o], mask, CORR_BLUR_PX * ASSUMED_UM_PER_PX / MODEL_PIXEL_UM), mask, ax[0],
                blur=0.0) for ax in ('y', 'x')} for o in ODORS}
            for o in ODORS:
                for ax in ('y', 'x'):
                    prof_rows.append(dict(base=base, coupling=coupling, view=view, kind='odor', pair=o, axis=ax,
                        **{f'bin{i}': v for i, v in enumerate(pr[o][ax])}))
            for oa in ('pa', 'paa'):
                for ax in ('y', 'x'):
                    d = pr[oa][ax] - pr['ms'][ax]
                    prof_rows.append(dict(base=base, coupling=coupling, view=view, kind='difference', pair=f'{oa} - ms', axis=ax,
                        dcentroid=float(np.sum((np.arange(N_PROFILE_BINS) + 0.5) / N_PROFILE_BINS * d)),
                        **{f'bin{i}': v for i, v in enumerate(d)}))
    stats = pd.DataFrame(rows)
    profs = pd.DataFrame(prof_rows)
    stats.to_csv(out / 'E79_model_map_stats.csv', index=False)
    profs.to_csv(out / 'E79_model_profiles.csv', index=False)
    comp = model_vs_measured(stats, profs, measured)
    comp.to_csv(out / 'E79_model_vs_measured.csv', index=False)
    plot_model_maps(map_store, out)
    if 'profiles' in measured:
        mp = profs[(profs.kind == 'difference') & (profs.view == 'projection')].copy()
        mp['axis'] = mp['axis'].map({'y': 'y', 'x': 'x_mirrored'})
        plot_profiles(measured['profiles']['per_fly'], measured['profiles']['bootstrap'], out, model=mp)
    return dict(stats=stats, profiles=profs, comparison=comp, axes_check=ax_check)


def _pair_resid_r(bl: Dict[str, np.ndarray], mask: np.ndarray, a: str, b: str, mode: str) -> float:
    demean = mode == 'means removed'
    X = np.stack([bl[o][mask] for o in ODORS])
    _, v = svd_shared(X, demean)
    return float(np.corrcoef(_residual(bl[a][mask], v, demean), _residual(bl[b][mask], v, demean))[0, 1])


def model_vs_measured(stats: pd.DataFrame, profs: pd.DataFrame, measured: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """One row per statistic, blur and mode: the measured median [min, max] over recordings (the model-matched three
    stimuli, and all stimuli), and the model's value per coupling and view."""
    rows = []
    sh = measured.get('shared', {}).get('per_recording')
    stat_cols = [('cv_median', 'spatial CV of the maps (SD / mean over the calyx; median over maps)'),
        ('share_first_sv', 'share of sum of squares in the first singular vector'),
        ('resid_between_odor_r_mean', 'residual between-odor r (mean over odor pairs; about -1/(n-1) by construction)'),
        ('raw_between_odor_r_median', 'between-odor r of the maps (median over odor pairs)'),
        ('resid_autocorr_len', 'residual autocorrelation 1/e length (measured: of the mean map, noise included)'),
        ('resid_crossrep_len', 'residual 1/e length (measured: cross-presentation, noise left out; model: its autocorrelation)')]
    for blur_px in BLURS_PX:
        for mode in ('means kept', 'means removed'):
            for col, lab in stat_cols:
                row = dict(blur_px=blur_px, blur_um_assumed=blur_px * ASSUMED_UM_PER_PX, mode=mode, statistic=lab, column=col)
                if sh is not None:
                    for set_name, tag in (('model-matched three (one per odor)', 'measured_3'), ('all stimuli (residuals: top two per odor)', 'measured_all')):
                        v = sh[(sh.stimulus_set == set_name) & (sh['mode'] == mode) & (sh.blur_px == blur_px)][col]
                        scale = ASSUMED_UM_PER_PX if col.endswith('_len') else 1.0
                        row[f'{tag}_median'] = float(v.median()) * scale
                        row[f'{tag}_min'] = float(v.min()) * scale
                        row[f'{tag}_max'] = float(v.max()) * scale
                        if col.endswith('_len'):
                            row[f'{tag}_median_px'] = float(v.median())
                s = stats[(stats.blur_px == blur_px) & (stats['mode'] == mode)]
                for _, r in s.iterrows():
                    row[f"model_{r['view']}_c{r['coupling']:g}"] = r['resid_autocorr_len' if col == 'resid_crossrep_len' else col]
                rows.append(row)
    # profiles: the measured mean pa - ms / paa - ms (over flies) against the model's, read with image top = dorsal and
    # with image top = ventral (the orientation is unknown), and along x
    pf = measured.get('profiles', {}).get('bootstrap')
    if pf is not None:
        for pair in ('pa - ms', 'paa - ms'):
            for ax_meas, ax_model in (('y', 'y'), ('x_mirrored', 'x')):
                mv = pf[(pf.pair == pair) & (pf.axis == ax_meas)].sort_values('bin').mean_diff.to_numpy()
                for _, r in profs[(profs.kind == 'difference') & (profs.pair == pair) & (profs.axis == ax_model)].iterrows():
                    md = np.array([r[f'bin{i}'] for i in range(N_PROFILE_BINS)])
                    rows.append(dict(statistic=f'{pair} profile: r(measured mean over flies, model)', column='profile_r',
                        axis=f'measured {ax_meas} vs model {"dorsal-ventral" if ax_model == "y" else "left-right"}',
                        view=r['view'], coupling=r['coupling'], r_same_direction=float(np.corrcoef(mv, md)[0, 1]),
                        r_reversed=float(np.corrcoef(mv, md[::-1])[0, 1]),
                        model_amplitude=float(np.abs(md).sum() / 2), measured_amplitude=float(np.abs(mv).sum() / 2)))
    return pd.DataFrame(rows)


def plot_model_maps(store: Dict[tuple, tuple], out: Path) -> None:
    """E79_model_maps.png: the unit model's projection maps of pa, ms and paa (each over its calyx mean) and pa minus ms,
    paa minus ms (over the mean of the two), one row per coupling; posterior view, fly's right to the right, dorsal up."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [k for k in store if k[2] == 'projection']
    fig, axes = plt.subplots(len(keys), 5, figsize=(13, 2.6 * len(keys)), squeeze=False, layout='constrained')
    for i, k in enumerate(keys):
        maps, mask = store[k]
        rel = {o: _masked_blur(maps[o], mask, 1.5 / MODEL_PIXEL_UM) / maps[o][mask].mean() for o in ODORS}
        panels = [(o, rel[o], 'viridis', 0.0, 2.0) for o in ODORS] + [(f'{a} - ms', rel[a] - rel['ms'], 'RdBu_r', -0.6, 0.6) for a in ('pa', 'paa')]
        for j, (lab, m, cmap, lo, hi) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(np.where(mask, m, np.nan), cmap=cmap, vmin=lo, vmax=hi)
            ax.contour(mask, levels=[0.5], colors='0.3', linewidths=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{k[0]}, coupling {k[1]:g}: {lab}', fontsize=7)
            if j in (2, 4):
                fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle('Local APL-unit model (E2): unit potentials projected over the calyx, posterior view (fly\'s right to the '
        'right, dorsal up), each over its calyx mean; blur 1.5 um', fontsize=9)
    fig.savefig(out / 'E79_model_maps.png', dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# provenance and the comparison table
def _git(*args) -> str:
    return subprocess.run(['git', *args], cwd=HERE, capture_output=True, text=True).stdout.strip()


def _sha(p: Path) -> Optional[str]:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def write_provenance(out: Path, extra: dict) -> None:
    """provenance.json: script, al_analysis commit and dirty state, inputs with sha256."""
    inputs = sorted(DATA.glob('*')) + [HERE / 'apl_calyx_maps.py'] + sorted(E2_RESULTS.glob('unit_potentials_*.csv'))
    try:
        from al_analysis.mb_model import apl_synapses_path
        inputs.append(Path(apl_synapses_path))
    except Exception:
        pass
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script=Path(__file__).name,
        host=socket.gethostname(), al_analysis_commit=_git('rev-parse', 'HEAD'),
        al_analysis_dirty=bool(_git('status', '--porcelain', '--', Path(__file__).name, 'apl_calyx_maps.py')),
        al_analysis_diff=_git('diff', 'HEAD', '--', Path(__file__).name, 'apl_calyx_maps.py'),
        al_analysis_dirty_files_this_folder=_git('status', '--porcelain', '--', '.'),
        inputs={str(p): _sha(p) for p in inputs if p.is_file()},
        settings=dict(blurs_px=BLURS_PX, corr_blur_px=CORR_BLUR_PX, n_profile_bins=N_PROFILE_BINS,
            assumed_um_per_px=ASSUMED_UM_PER_PX, n_boot=N_BOOT, seed=SEED, baselines=BASELINES, main_baseline=MAIN_BASELINE,
            min_match_reliability=MIN_MATCH_RELIABILITY, model_odor_rows=MODEL_ODOR_ROWS, model_pixel_um=MODEL_PIXEL_UM,
            model_render_um=MODEL_RENDER_UM, slab_um=SLAB_UM, slab_fracs=SLAB_FRACS),
        **extra)
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')


def _f(x, spec='.2f') -> str:
    try:
        return '-' if x is None or not np.isfinite(x) else format(x, spec)
    except TypeError:
        return str(x)


def write_comparison_table(out: Path, tables: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """comparison_table.md: one section per analysis with the key numbers."""
    L = ['# The APL calyx maps against the connectome (E79): analyses 1, 2, 3 and 5', '',
        f'Generated by E79_apl_calyx_constraints.py ({datetime.now().strftime("%Y-%m-%d %H:%M")}); image axes as recorded '
        '(x left to right, y top to bottom; anatomical orientation unknown); medians over the 12 recordings unless stated.', '']
    if 'blank' in tables:
        t = tables['blank']
        n = t['noise']
        L += ['## 1. Blank control and baselines', '',
            '| recording | side | paraffin oil, 1st in block: calyx mean | paraffin oil block 2nd / 3rd: calyx mean | position-matched blank: '
            'calyx mean | blank halves r | blank r with top odor maps (median) | order-matched baseline noise / signal SD | '
            'first-in-block baseline noise / signal SD |', '|---|---|---|---|---|---|---|---|---|']
        b = t['blank']
        for _, r in n.iterrows():
            rb = b[(b.recording == r.recording) & b.top_conc]
            L.append(f"| {r.recording} | {r.side} | {_f(r.pfo_first_in_block_mean, '.3f')} | {_f(r.pfo_block_pos2_mean, '.3f')} / "
                f"{_f(r.pfo_block_pos3_mean, '.3f')} | {_f(r.blank_mean, '.3f')} | {_f(r.blank_halves_r)} | "
                f"{_f(rb.r_blank_vs_odor_first_in_block.median())} | {_f(r.noise_to_signal_order_matched)} | "
                f"{_f(r.noise_to_signal_first_in_block)} |")
        L += ['', 'Odor maps under the three baselines (correlations at 2 px blur; "top" = each odor\'s highest concentration):', '',
            '| statistic | all paraffin oil (atlas) | order-matched | first in block |', '|---|---|---|---|']
        s = t['summary'].set_index('baseline')
        for col, lab, spec in (('repeat_r_top_median', 'repeat r, top concentrations (median)', '.2f'),
                ('repeat_r_all_median', 'repeat r, all stimuli (median)', '.2f'),
                ('same_odor_r_median', 'same odor, different concentrations r (median)', '.2f'),
                ('different_odor_r_top_median', 'different odors r, top concentrations (median)', '.2f'),
                ('ms_vs_pa_r_top_median', 'ms vs pa r, top concentrations (median)', '.2f'),
                ('ms_vs_pa_r_top_min', 'ms vs pa r, top concentrations (min)', '.2f'),
                ('n_recordings_ms_vs_pa_top_below_0p8', 'recordings with ms vs pa r < 0.8', 'd'),
                ('n_recordings_pa_centroid_below_ms_top', "recordings with pa's centroid lower than ms's (top conc.)", 'd'),
                ('n_recordings_paa_centroid_below_ms_top', "recordings with paa's centroid lower than ms's (top conc.)", 'd')):
            L.append(f'| {lab} | ' + ' | '.join(_f(s.loc[bl, col], spec) for bl in BASELINES) + ' |')
        x = t['baseline_cross']
        L += ['', f"Same stimulus across baselines (r at 2 px blur, median over recordings and stimuli): order-matched vs all paraffin "
            f"oil {_f(x.r_order_matched_vs_all_pfo.median())} (top concentrations {_f(x[x.top_conc].r_order_matched_vs_all_pfo.median())}), "
            f"order-matched vs first in block {_f(x.r_order_matched_vs_first_in_block.median())} (top "
            f"{_f(x[x.top_conc].r_order_matched_vs_first_in_block.median())}).", '']
        d = t['doubtful']
        if len(d):
            L += ['Presentations whose label repeats an earlier block (left out of the maps):', '',
                '| recording | label | block | calyx mean | repeat r | most similar map (r) | best ms r | best pa r | best paa r |',
                '|---|---|---|---|---|---|---|---|---|']
            for _, r in d.iterrows():
                L.append(f'| {r.recording} | {r.label} | {r.block} | {_f(r.mean_dff, ".3f")} | {_f(r.repeat_r)} | {r.most_similar} '
                    f'({_f(r.r_most_similar)}) | {_f(r.r_best_ms)} | {_f(r.r_best_pa)} | {_f(r.r_best_paa)} |')
            L.append('')
    if 'shared' in tables:
        s = tables['shared']['summary']
        L += ['## 2. Shared and odor-specific parts (order-matched baseline)', '',
            'Median [min, max] over recordings. Residual statistics over each odor\'s two highest concentrations (all stimuli) '
            'or the three model-matched maps; lengths in px (x 0.3 um per px assumed); "cross-presentation" length from the '
            'cross-correlation of the two presentations\' residuals, which leaves out their independent noise; blurred white '
            'noise falls to 1/e at 2 sigma.', '',
            '| stimulus set | mode | blur (px) | first singular vector share | residual repeat r | residual between-odor r | '
            'residual same-odor r | residual 1/e length (px) | cross-presentation 1/e length (px) | white-noise floor (px) |',
            '|---|---|---|---|---|---|---|---|---|---|']
        for _, r in s.iterrows():
            def mm(c, spec='.2f'):
                return f"{_f(r[c + '_median'], spec)} [{_f(r[c + '_min'], spec)}, {_f(r[c + '_max'], spec)}]"
            L.append(f"| {r.stimulus_set} | {r['mode']} | {r.blur_px:g} | {mm('share_first_sv', '.3f')} | {mm('resid_repeat_r_median')} | "
                f"{mm('resid_between_odor_r_median')} | {mm('resid_same_odor_r_median')} | {mm('resid_autocorr_len', '.1f')} | "
                f"{mm('resid_crossrep_len', '.1f')} | {_f(r['noise_floor_len_median'], '.0f')} |")
        L.append('')
    if 'profiles' in tables:
        p = tables['profiles']
        L += ['## 3. Dorsal-ventral profiles at matched calyx mean dF/F', '',
            f'Matched pairs: per recording, the concentrations of the two odors whose calyx means are closest (|log ratio|), '
            f'among maps with a positive mean and repeat r >= {MIN_MATCH_RELIABILITY}.', '',
            '| recording | side | pair | concentrations | calyx means | y centroid shift (bins, + = lower in the image) |',
            '|---|---|---|---|---|---|']
        for _, r in p['pairs'].iterrows():
            if not r.get('matched', False):
                L.append(f'| {r.recording} | {r.side} | {r.pair} | no eligible pair | - | - |')
                continue
            L.append(f"| {r.recording} | {r.side} | {r.pair} | {r.conc_a:g} / {r.conc_ms:g} | {_f(r.mean_a, '.3f')} / "
                f"{_f(r.mean_ms, '.3f')} | {_f(r.dcentroid_y, '+.3f')} |")
        L += ['', 'Per fly (hemispheres averaged), mean over flies with the bootstrap 90 % interval (10,000 resamplings of flies):', '',
            '| pair | axis | flies | centroid shift (0-1 of the box) | 90 % interval | flies with a positive shift |', '|---|---|---|---|---|---|']
        for _, r in p['centroid_shift'].iterrows():
            L.append(f'| {r.pair} | {r.axis} | {r.n_flies} | {_f(r.mean_shift, "+.3f")} | [{_f(r.lo90, "+.3f")}, {_f(r.hi90, "+.3f")}] | '
                f'{r.n_flies_positive} |')
        a = p['adjusted']
        L += ['', 'Centroid against magnitude (maps with a positive mean and repeat r >= '
            f'{MIN_MATCH_RELIABILITY}): Spearman between each odor\'s calyx mean and its centroid across concentrations '
            '(median [min, max] over recordings; negative on y = stronger responses sit higher in the image), and the odor '
            'offsets at equal magnitude from one fit per recording (centroid on log calyx mean plus odor), per fly with the '
            'bootstrap 90 % interval:', '',
            '| axis | quantity | value | interval / range | flies (or recordings) positive |', '|---|---|---|---|---|']
        for _, r in a.iterrows():
            L.append(f'| {r.axis} | {r.quantity} | {_f(r["mean"], "+.3f")} | [{_f(r.lo90, "+.3f")}, {_f(r.hi90, "+.3f")}] | '
                f'{r.n_flies_positive} |')
        b = p['bootstrap']
        L += ['', 'Difference profile along image y (top to bottom), mean over flies [90 % interval], per tenth of the box:', '',
            '| pair | ' + ' | '.join(f'bin {i}' for i in range(N_PROFILE_BINS)) + ' |', '|---|' + '---|' * N_PROFILE_BINS]
        for pair in ('pa - ms', 'paa - ms'):
            bb = b[(b.pair == pair) & (b.axis == 'y')].sort_values('bin')
            L.append(f'| {pair} | ' + ' | '.join(f'{_f(r.mean_diff, "+.3f")} [{_f(r.lo90, "+.3f")}, {_f(r.hi90, "+.3f")}]'
                for _, r in bb.iterrows()) + ' |')
        L.append('')
    if 'model' in tables:
        m = tables['model']
        st = m['stats']
        ac = m['axes_check']
        L += ['## 5. The local APL-unit model (E2) against the maps', '',
            'Model: E2\'s saved unit potentials (claw-dynamics base at 5 %, mean potential over 0-0.5 s; the only base and '
            'couplings with saved potentials), for pa @ -3, ms @ -3 and paa @ -4 (E2\'s inputs). Maps: every calyx APL node '
            '(load_apl_points, nodes with synapses mostly in CA(R)) takes its unit\'s potential; posterior view (fly\'s right '
            'to the right, dorsal up), 0.5 um pixels, normalized convolution at 1 um, then each blur at 0.3 um per measured px '
            '(assumed). "projection" = all calyx depths; slabs 6 um thick at 25 / 50 / 75 % of the calyx\'s anterior-posterior '
            'extent. Measured: the three model-matched maps per recording (pa -4.386, ms -3.431, paa -4.386), order-matched '
            'baseline.', '',
            'Axis check (APL synapses per hemibrain ROI, median position relative to the calyx, um):', '',
            '| ROI | right | anterior | dorsal | synapses | expected | as expected |', '|---|---|---|---|---|---|---|']
        for _, r in ac.iterrows():
            L.append(f"| {r.roi} | {_f(r.right_um, '+.1f')} | {_f(r.get('anterior_um', np.nan), '+.1f')} | {_f(r.dorsal_um, '+.1f')} | "
                f"{int(r.n_synapses)} | {r.expected} | {r.as_expected} |")
        c = m['comparison']
        if 'shared' in tables:
            roi = tables['shared']['per_recording'].drop_duplicates('recording')
            L += ['', 'Implied pixel size, if the imaged plane spans the calyx\'s posterior-view extent (a rough check of the '
                'assumed 0.3 um per px; the model outline includes about 1-3 um of rendering margin):', '',
                '| model view | outline width x height (um) | um per px from widths: median [min, max] | from heights |',
                '|---|---|---|---|']
            for view, g in st.groupby('view'):
                w, h = float(g.mask_width_um.iloc[0]), float(g.mask_height_um.iloc[0])
                fw, fh = w / roi.roi_width_px, h / roi.roi_height_px
                L.append(f'| {view} | {w:.1f} x {h:.1f} | {fw.median():.3f} [{fw.min():.3f}, {fw.max():.3f}] | '
                    f'{fh.median():.3f} [{fh.min():.3f}, {fh.max():.3f}] |')
        L += ['', 'Statistics against coupling (model, projection and middle slab) beside the measured median [min, max] over '
            'recordings (three model-matched maps, and all maps with the residuals of each odor\'s two highest concentrations); '
            'lengths in um (measured at the assumed 0.3 um per px). The residual between-odor r is close to -1/(n-1) by '
            'construction (the residuals of the n maps, weighted by the first left singular vector, sum to zero), so with '
            'three maps it is near -0.5 in any data and does not discriminate; the between-odor r of the maps themselves and '
            'the share with means removed do.', '']
        couplings = sorted(st.coupling.unique())
        hdr = ['statistic', 'mode', 'blur (px / um)', 'measured (3 maps)', 'measured (all; residuals top two)'] + \
            [f'model c={x:g} projection' for x in couplings] + [f'model c={x:g} middle slab' for x in couplings]
        L += ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
        cc = c[c.column != 'profile_r']
        for _, r in cc.iterrows():
            spec = '.1f' if r.column.endswith('_len') else '.3f' if r.column.startswith('share') else '.2f'
            vals = [f"{_f(r.get('measured_3_median'), spec)} [{_f(r.get('measured_3_min'), spec)}, {_f(r.get('measured_3_max'), spec)}]",
                f"{_f(r.get('measured_all_median'), spec)} [{_f(r.get('measured_all_min'), spec)}, {_f(r.get('measured_all_max'), spec)}]"]
            vals += [_f(r.get(f'model_projection_c{x:g}'), spec) for x in couplings]
            vals += [_f(r.get(f'model_slab_middle_c{x:g}'), spec) for x in couplings]
            L.append(f"| {r.statistic} | {r['mode']} | {r.blur_px:g} / {r.blur_um_assumed:.1f} | " + ' | '.join(vals) + ' |')
        pr = c[c.column == 'profile_r']
        if len(pr):
            L += ['', 'Difference profiles (10 bins): r between the measured mean over flies and the model, with the image read '
                'top = dorsal ("same direction") and top = ventral ("reversed"); amplitudes = half the summed |difference| '
                '(share moved):', '',
                '| pair | axes | view | coupling | r same direction | r reversed | model amplitude | measured amplitude |',
                '|---|---|---|---|---|---|---|---|']
            for _, r in pr.iterrows():
                L.append(f"| {r.statistic.split(' profile')[0]} | {r.axis} | {r.view} | {r.coupling:g} | {_f(r.r_same_direction)} | "
                    f"{_f(r.r_reversed)} | {_f(r.model_amplitude, '.3f')} | {_f(r.measured_amplitude, '.3f')} |")
        cv = st[(st.view == 'projection') & (st.blur_px == BLURS_PX[0]) & (st['mode'] == 'means kept')]
        L += ['', 'Model maps\' spatial CV (SD / mean over the calyx outline, projection, blur 0.6 um):', '',
            '| coupling | pa | ms | paa | mean potential pa / ms / paa |', '|---|---|---|---|---|']
        for _, r in cv.iterrows():
            L.append(f"| {r.coupling:g} | {_f(r.cv_pa)} | {_f(r.cv_ms)} | {_f(r.cv_paa)} | {_f(r.mean_potential_pa, '.1f')} / "
                f"{_f(r.mean_potential_ms, '.1f')} / {_f(r.mean_potential_paa, '.1f')} |")
        L.append('')
    (out / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--analyses', default='1,2,3,5', help='comma-separated subset of 1,2,3,5')
    args = p.parse_args()
    which = {a.strip() for a in args.analyses.split(',') if a.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    recs = load_recordings()
    tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    if '1' in which:
        tables['blank'] = analysis_blank(recs, OUT)
        print('analysis 1 done', flush=True)
    if '2' in which or '5' in which:
        tables['shared'] = analysis_shared(recs, OUT)
        print('analysis 2 done', flush=True)
    if '3' in which or '5' in which:
        tables['profiles'] = analysis_profiles(recs, OUT)
        print('analysis 3 done', flush=True)
    if '5' in which:
        tables['model'] = analysis_model(tables, OUT)
        print('analysis 5 done', flush=True)
    write_comparison_table(OUT, tables)
    write_provenance(OUT, dict(analyses=sorted(which), runtime_s=(datetime.now() - t0).total_seconds(),
        recordings=[r.name for r in recs]))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
