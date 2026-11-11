#!/usr/bin/env python3
"""The APL calyx maps against the connectome (E79), the data-side analyses 1, 2, 3 and 5 (analyses 4 and 6, the
connectome-predicted maps and glomerulus co-localization, are in E79_connectome_maps.py). No model fits.

Inputs: Yang's 2021 single-plane APL calyx maps (data/yang_apl_calyx_maps/2021_APL_18um/ in the shared directory: per
recording a TIFF of 51 dF/F peak maps, one per presentation, in 17 blocks of three: paraffin oil first, then an odor x
concentration twice; the last block is three paraffin oil presentations), the ImageJ calyx-outline ROIs, frames.csv and
recordings.csv; the ROI reader and mask of apl_calyx_maps.py (atlas's first pass, imported, not changed); for analysis 5
the local APL-unit model's saved unit potentials (results/E2_local_apl/unit_potentials_*_coupling*.csv) and
`mb_model.load_apl_points()`.

Analyses (outputs to results/E79_apl_calyx_constraints/ in the shared directory):
1. Blank control: the position-matched blank (the last block's 2nd and 3rd paraffin oil presentations minus the mean of
   every first-in-block paraffin oil presentation), its calyx mean and its Pearson r (2 px blur) with each odor map; every
   odor x concentration map re-made with an order-matched baseline (odor presentations are 2nd and 3rd in their block, so
   each presentation minus the last block's paraffin oil presentation at the same position) and with the first-in-block
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import apl_calyx_maps as ACM

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


def load_recordings() -> List[Recording]:
    """Reads every recording of recordings.csv (the TIFF, its frames.csv rows with the presentation's position in its
    block, and the calyx mask from its ImageJ ROI)."""
    raise NotImplementedError


def add_positions(frames: pd.DataFrame) -> pd.DataFrame:
    """frames.csv rows of one recording with `position` (1-3, the presentation's place in its block), `last_block` and
    `doubtful` (a presentation whose odor x concentration was already presented in an earlier block, as pa -6.771 in
    blocks 2 and 12 of 20210805_3FRfun1 and 20210823_4FRfun2, where the yaml's pin 43 in block 12 is probably a typo for 53,
    ms -2: kept out of the stimulus maps and checked separately)."""
    raise NotImplementedError


def blank_frames(frames: pd.DataFrame) -> Dict[str, List[int]]:
    """Frame indices of the paraffin oil presentations: 'first_in_block' (position 1 of every block, the last block's
    included), 'last_pos2' / 'last_pos3' (the last block's 2nd and 3rd) and 'all' (every one)."""
    raise NotImplementedError


def position_matched_blank(stack: np.ndarray, frames: pd.DataFrame) -> np.ndarray:
    """The last block's 2nd and 3rd paraffin oil maps' mean minus the mean of every first-in-block paraffin oil map."""
    raise NotImplementedError


def stimulus_maps(stack: np.ndarray, frames: pd.DataFrame, baseline: str) -> Dict[Stim, Tuple[np.ndarray, List[np.ndarray]]]:
    """(odor, log10 conc) -> (mean map, per-presentation maps), each presentation minus its baseline:
    'all_pfo' (atlas's: the mean of every paraffin oil map), 'order_matched' (the last block's paraffin oil map at the
    presentation's own position, 2nd or 3rd, so the two presentations' baselines are independent) or 'first_in_block'
    (the mean of every first-in-block paraffin oil map). Doubtful presentations (`add_positions`) are left out."""
    raise NotImplementedError


def blurred_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray, blur: float = CORR_BLUR_PX) -> float:
    """Pearson r over calyx pixels after a Gaussian blur (sigma `blur` px)."""
    raise NotImplementedError


def analysis_blank(recs: Sequence[Recording], out: Path) -> Dict[str, pd.DataFrame]:
    """Analysis 1: the blank's calyx mean and r with each odor map, the baselines' noise, and correlations, centroids and
    3 x 3 grid shares under each baseline. Writes E79_blank_*.csv, E79_baseline_*.csv and a figure."""
    raise NotImplementedError


def svd_shared(X: np.ndarray, demean: bool) -> Tuple[float, np.ndarray]:
    """Share of the sum of squares in the first singular vector of the stimulus x pixel matrix `X` (each row's mean
    removed first if `demean`), and that unit pixel vector."""
    raise NotImplementedError


def autocorr_length(img: np.ndarray, mask: np.ndarray, other: Optional[np.ndarray] = None) -> float:
    """Lag (px) at which the radially averaged, mask-normalized autocorrelation of `img` inside `mask` (or its
    cross-correlation with `other`, which leaves out noise independent between the two) falls to 1/e of its value at 0
    (linear interpolation between lags; NaN if it never does within half the mask's extent)."""
    raise NotImplementedError


def analysis_shared(recs: Sequence[Recording], out: Path, baseline: str = 'order_matched') -> Dict[str, pd.DataFrame]:
    """Analysis 2 per recording and blur: the first singular vector's share (means kept / removed), the residual's
    repeat reliability, between-odor r and autocorrelation length. Writes E79_shared_*.csv and a figure."""
    raise NotImplementedError


def profile_shares(m: np.ndarray, mask: np.ndarray, axis: str, n_bins: int = N_PROFILE_BINS,
        blur: float = CORR_BLUR_PX) -> np.ndarray:
    """Share of the (blurred, clipped at 0) response in `n_bins` equal bins along image 'y' (top to bottom) or 'x' (left
    to right) of the ROI's bounding box."""
    raise NotImplementedError


def matched_pair(means: Dict[Stim, float], reliab: Dict[Stim, float], odor_a: str, odor_b: str,
        min_reliability: float = 0.5) -> Optional[Tuple[Stim, Stim]]:
    """The (odor_a, odor_b) concentration pair whose calyx means are closest (smallest |log ratio|) among stimuli with a
    positive mean and repeat r >= `min_reliability`; None if there is none."""
    raise NotImplementedError


def fly_bootstrap(per_fly: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean over flies (rows) and the 5th / 95th percentiles of the mean over `n_boot` resamplings of flies."""
    raise NotImplementedError


def analysis_profiles(recs: Sequence[Recording], out: Path, baseline: str = 'order_matched') -> Dict[str, pd.DataFrame]:
    """Analysis 3: per-stimulus profiles along y and x, the matched pa / ms and paa / ms pairs, their difference profiles
    per fly and the bootstrap intervals. Writes E79_profile_*.csv and a figure."""
    raise NotImplementedError


def model_maps(potentials: pd.DataFrame, odors: Sequence[str], pixel_um: float, blur_um: float,
        slab: Optional[Tuple[float, float]] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Posterior-view maps (fly's right to the right, dorsal up; rows from dorsal to ventral like an image) of the APL
    units' potentials: every calyx APL node of `load_apl_points()` takes its unit's potential, and each pixel is the
    Gaussian-weighted mean over nodes (normalized convolution, sigma `blur_um`), optionally only nodes within `slab`
    (anterior um). Returns odor -> map and the mask of pixels with nodes nearby."""
    raise NotImplementedError


def analysis_model(recs_stats: Dict[str, pd.DataFrame], out: Path) -> Dict[str, pd.DataFrame]:
    """Analysis 5: the statistics of analyses 2 and 3 on the E2 unit model's maps of pa, ms and paa at each saved coupling,
    beside the measured values. Writes E79_model_*.csv and a figure."""
    raise NotImplementedError


def write_provenance(out: Path, extra: dict) -> None:
    """provenance.json: script, al_analysis commit and dirty state, inputs with sha256."""
    raise NotImplementedError


def write_comparison_table(out: Path, tables: Dict[str, pd.DataFrame]) -> None:
    """comparison_table.md: one section per analysis with the key numbers."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--analyses', default='1,2,3,5', help='comma-separated subset of 1,2,3,5')
    args = p.parse_args()
    raise NotImplementedError(args)


if __name__ == '__main__':
    main()
