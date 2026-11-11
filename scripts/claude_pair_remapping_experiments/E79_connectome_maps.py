#!/usr/bin/env python3
"""The APL calyx maps against the connectome (E79), the connectome analyses 4 and 6 (the data-side analyses 1, 2, 3 and 5
are in E79_apl_calyx_constraints.py). No model fits.

Inputs:
- Yang's own Claude's connectome tables (separate work, in progress; tensor Tom/claude_APL_modeling_YZ4TC/
  Claude_APL_analysis/apl_synapses/): `<dataset>_PN_boutons_derived.csv.gz` (bouton centroids in nm, n_tbars, pn_type),
  `<dataset>_KC_claws_derived.csv.gz` and `<dataset>_APL_MB_synapses.csv.gz` for hemibrain, FlyWire, male CNS and BANC.
  The seven calyces are FlyWire L / R, male CNS L / R, BANC L / R and hemibrain R; the eighth, `model_hemibrain`, is the
  model's own hemibrain boutons (Pratyush's v5 bouton segmentation, as `prat_boutons=True` uses it).
- The model's PN rates per glomerulus: the APL16 ORN input through the PN stage of the best family (the breadth-2
  APL-effect fit, C2, E67's `rate0.068_aplboth_k2_cap3`, built by `E77_input_screen.build_c2_simulator`), evoked rates
  (the 0-0.5 s mean minus the pre-odor mean), cached in connectome_pn_rates.csv.
- C2's APL-on spike counts per KC (the model-output archive) for the spike-driven alternative.
- The measured maps (data/yang_apl_calyx_maps/, read with apl_calyx_maps.py's ROI reader and stimulus maps).
- The error common to all wirings (results/E72_wiring_population/common_error_pairs.csv, column `common`).

Analyses (outputs to results/E79_apl_calyx_constraints/ in the shared directory, files connectome_* and coloc_*, figures
E79_connectome_* / E79_coloc_*, provenance_connectome.json, connectome_table.md):
4. Connectome-predicted APL maps per calyx: per odor, sum over boutons of n_tbars x the glomerulus's PN rate at the bouton
   centroid, in slabs 6 um thick along the anterior-posterior axis in 3 um steps, Gaussian blur 2, 4 and 8 um, posterior
   view, scaled to the slab's calyx outline (APL calyx synapses in the slab) bounding box. Alternatives: claw-driven (claw
   centroids x their PN synapses x the claw glomerulus's rate), spike-driven (KC->APL calyx synapses x C2's APL-on spike
   counts of the KC; hemibrain only) and anatomy only (unweighted PN<->APL calyx synapses). pa minus ms and paa minus ms
   (and the single-odor maps) scored against the measured maps on a 12 x 12 grid over each outline's bounding box (Pearson
   r over cells inside both outlines) for the four flips, every slab, calyx and recording; null: 1,000 permutations of the
   glomerulus labels of the rate vector (KC labels for the spike-driven maps).
6. Glomerulus co-localization C_gh(lambda) (n_tbars-weighted mean over bouton pairs of exp(-d^2 / 2 lambda^2), lambda 2,
   4, 8, 16 um) per calyx; agreement between calyces; whether the block glomeruli are nearer each other than random sets
   of eight; each APL16 odor's spatial concentration x'Cx / (sum x)^2; the proxy (the change in the PN-rate cosine distance
   per odor pair when each glomerulus's inhibition follows its local field Cx instead of the global mean, subtractive and
   divisive) against the common error, leave one odor out, against a null of C permuted by glomerulus.

Run from this folder: `OMP_NUM_THREADS=6 OMP_THREAD_LIMIT=6 nice -n 10 ../../venv/bin/python E79_connectome_maps.py
[--analyses 4,6] [--n-perm 1000]`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import apl_calyx_maps as ACM

SHARED = ACM.SHARED
OUT = SHARED / 'results' / 'E79_apl_calyx_constraints'
YANG_APL = Path('/mnt/tensor/Tom/claude_APL_modeling_YZ4TC/Claude_APL_analysis/apl_synapses')
DATASETS: Tuple[str, ...] = ('hemibrain', 'flywire', 'malecns', 'banc')
MAP_ODORS: Tuple[str, ...] = ('pa', 'paa', 'ms')
DIFFS: Tuple[Tuple[str, str], ...] = (('pa', 'ms'), ('paa', 'ms'))
SLAB_THICK_UM = 6.0
SLAB_STEP_UM = 3.0
BLURS_UM: Tuple[float, ...] = (2.0, 4.0, 8.0)
GRID_N = 12
FLIPS: Tuple[str, ...] = ('none', 'horizontal', 'vertical', 'both')
LAMBDAS_UM: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
BLOCK_GLOMERULI: Tuple[str, ...] = ('D', 'DC2', 'DL1', 'DM1', 'DM5', 'DP1m', 'VA7m', 'VC1')
N_PERM = 1000
SEED = 0
GATE_SHARE = 0.15


def to_rda(dataset: str, x, y, z) -> np.ndarray:
    """Native nm -> (fly's right, dorsal, anterior) in um (posterior view: R to the right, D up; A positive anterior)."""
    raise NotImplementedError


def axis_checks() -> pd.DataFrame:
    """Per calyx, the landmark checks of to_rda's axes (lobes anterior of the calyx, the vertical lobe's tip dorsal of the
    peduncle, the medial lobe's tip medial of the calyx, the right MB at larger R than the left)."""
    raise NotImplementedError


def glomerulus_of_type(pn_type: str, dataset: str) -> Optional[str]:
    """The model's glomerulus name of a uniglomerular PN type (None for multiglomerular, bi-glomerular or untyped PNs)."""
    raise NotImplementedError


def load_calyces() -> Dict[str, Dict[str, pd.DataFrame]]:
    """calyx name -> {'boutons', 'claws', 'apl'}: tables with R, D, A (um), glomerulus and weights per point."""
    raise NotImplementedError


def model_pn_rates(rebuild: bool = False) -> pd.DataFrame:
    """Glomerulus x odor (16 APL16 odors, abbreviations) evoked PN rates of C2's PN stage; cached."""
    raise NotImplementedError


def c2_spike_counts() -> pd.DataFrame:
    """C2's APL-on spike counts, KC body id x odor abbreviation."""
    raise NotImplementedError


def measured_maps() -> Dict[str, dict]:
    """recording -> {'grid': {name: 12 x 12 map}, 'inside': 12 x 12 bool, 'concs': chosen concentrations, 'side'}, for
    the single-odor maps (each over its calyx mean) and pa minus ms, paa minus ms at matched mean dF/F."""
    raise NotImplementedError


def slab_glomerulus_grids(points: pd.DataFrame, outline_points: pd.DataFrame, blur_um: float,
                          columns: Sequence[str]) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """Per slab, each column's (glomerulus's or KC's) blurred map sampled on the 12 x 12 grid over the slab outline's
    bounding box: (slab centers, array slabs x columns x 144, inside mask slabs x 144)."""
    raise NotImplementedError


def flip_grid(g: np.ndarray, flip: str) -> np.ndarray:
    """A 12 x 12 grid (or ... x 12 x 12) flipped horizontally, vertically, both or not at all."""
    raise NotImplementedError


def masked_pearson(pred: np.ndarray, meas: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Pearson r over `inside` cells between each row of `pred` (n x 144) and `meas` (144,)."""
    raise NotImplementedError


def run_maps(n_perm: int = N_PERM) -> Dict[str, pd.DataFrame]:
    """Analysis 4: writes connectome_map_scores.csv (every calyx, alternative, blur, slab, flip, recording, comparison,
    with the null percentile), connectome_map_best.csv, connectome_flip_consistency.csv and figures."""
    raise NotImplementedError


def colocalization(boutons: pd.DataFrame, lam_um: float, glomeruli: Sequence[str]) -> pd.DataFrame:
    """C_gh: the n_tbars-weighted mean over bouton pairs (one of g, one of h) of exp(-d^2 / 2 lambda^2)."""
    raise NotImplementedError


def proxy_change(X: np.ndarray, C: np.ndarray, form: str, strength: float) -> np.ndarray:
    """Per odor pair, the cosine distance of the PN-rate vectors with each glomerulus inhibited by its local field Cx minus
    that with the global mean (one scale fitted so that the mean inhibition is equal); `form` 'sub' or 'div'."""
    raise NotImplementedError


def loo_share(delta: np.ndarray, err: np.ndarray, pairs: Sequence[Tuple[int, int]], n_odors: int) -> np.ndarray:
    """Share of err's squared sum removed by the best linear scaling of delta fitted on the pairs of 15 odors and scored
    on the 15 pairs of the left-out odor (pooled over the 16 odors); vectorized over leading axes of delta."""
    raise NotImplementedError


def run_coloc(n_perm: int = N_PERM) -> Dict[str, pd.DataFrame]:
    """Analysis 6: writes coloc_matrices_*.csv, coloc_agreement.csv, coloc_block.csv, coloc_concentration.csv,
    coloc_proxy.csv and figures."""
    raise NotImplementedError


def write_provenance(extra: dict) -> None:
    """provenance_connectome.json: script, al_analysis and olfsysm commits and dirty state, inputs with sha256."""
    raise NotImplementedError


def write_table(tables: Dict[str, pd.DataFrame]) -> None:
    """connectome_table.md: the key numbers of analyses 4 and 6."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--analyses', default='4,6', help='comma-separated subset of 4,6')
    p.add_argument('--n-perm', type=int, default=N_PERM)
    args = p.parse_args()
    raise NotImplementedError(args)


if __name__ == '__main__':
    main()
