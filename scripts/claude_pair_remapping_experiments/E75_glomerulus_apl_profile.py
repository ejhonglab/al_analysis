#!/usr/bin/env python3
"""E75: where APL acts, by glomerulus. The profile of APL inhibition over glomeruli that the per-pair APL effect asks for,
at the cell, the claw and the bouton, on the breadth-2 APL-effect fit (C2, E67) across wirings and against the connectomes.

The error the APL-effect fit on re-drawn wirings (E72) left is common to every wiring
(results/E72_wiring_population/common_error_pairs.csv): the model's per-pair APL effect is too small among 2h, IaA, pa,
eb, ep and ms and too large among 1-8ol, 1-5ol, 1-6ol and 6al. A wiring draw keeps each glomerulus's claws per KC type,
so a rule written per glomerulus is the one wiring-independent way to say which KCs APL acts on.

A profile is a score s_g per glomerulus applied at one site:
- 'cell': every claw's APL->KC weight of a KC times exp(sum over g of s_g x its number of claws from g) (E15's / E17's
  log-linear form), on top of the breadth weighting;
- 'claw': the APL->KC weight of each claw from g times exp(s_g);
- 'bouton': the APL->bouton weight of each bouton of g times exp(s_g).
The factor is divided by its mean (over claws, or over boutons), so the fitted scales keep their meaning.

Parts (CLI `--part`):
- A: run by `E43_model_trial_noise.py` (committed code; runner outputs/E75_glomerulus_apl_profile/run_viki_e75_partA.py);
  `--part A-score` scores it as E72's part E (`E72_wiring_population.part_e_score` on this experiment's partB/).
- B: no simulations: every archived variant's pattern error against E72's common error (`archive_vs_common`).
- C: the sensitivity map on C2 at its fitted parameters on the real wiring (`sensitivity_map`): known sets, then one
  glomerulus at a time at each site, the APL->KC scale bisected to C2's number of APL-on responses.
- D1: the linear profile from the map, cross-validated over odors and over recording halves, with the gate (`linear_profile`).
- D2: the chosen profiles simulated (`simulate_profiles`).
- F: the profile against the connectomes' APL densities and the glomeruli's drive (`anatomy`).
Fits with a profile (part D3) go through `E10_joint_fit.py --apl-glom-profile FILE`.

Outputs: results/E75_glomerulus_apl_profile/ (archive_vs_common_error.csv, sensitivity_map.csv and sensitivity_pairs.csv,
profiles.csv, profile_cv.csv, anatomy.csv, figures, provenance); simulated outputs under outputs/E75_glomerulus_apl_profile/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E75_glomerulus_apl_profile'
SITES = ('cell', 'claw', 'bouton')
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
E15_EIGHT = ('DM2', 'VM2', 'DM1', 'DM6', 'VM7d', 'DM4', 'VM3', 'VA2')
C2_FOUR = ('DM3', 'DM2', 'DM5', 'DM6')
ESTER_GROUP = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL_GROUP = ('1-8ol', '1-5ol', '1-6ol', '6al')


def claw_glomeruli(W: pd.DataFrame) -> pd.Series:
    """Each claw's glomerulus (from the claw x bouton wiring `W`), indexed as W's rows."""
    raise NotImplementedError


def profile_factors(site: str, scores: Dict[str, float], W: pd.DataFrame, wAPLKC: pd.Series, wAPLPN: pd.Series) -> pd.Series:
    """The multiplicative factor a profile puts on the APL->KC weights (site 'cell' or 'claw', indexed as `wAPLKC`) or on
    the APL->bouton weights (site 'bouton', indexed as `wAPLPN`), divided by its mean. All scores 0 gives exactly 1."""
    raise NotImplementedError


def apply_profile(site: str, scores: Dict[str, float], W: pd.DataFrame, wAPLKC: pd.Series, wAPLPN: pd.Series
                  ) -> Tuple[pd.Series, pd.Series]:
    """(APL->KC weights, APL->bouton weights) with the profile applied at `site`; the other set unchanged."""
    raise NotImplementedError


def common_error(readout: str = 'cal') -> pd.Series:
    """E72's per-pair error common to the eight wirings (model minus data, means removed), by (odor_a, odor_b)."""
    raise NotImplementedError


def archive_vs_common(out: Path) -> pd.DataFrame:
    """Part B: for every archived variant with both conditions, its per-pair pattern error on raw counts (and under the
    calibrated readout where spike counts are archived) against the common error: correlation, projection, the ester
    and alcohol groups' errors. Writes archive_vs_common_error.csv."""
    raise NotImplementedError


def sensitivity_map(out: Path, points: Optional[Sequence[dict]] = None) -> pd.DataFrame:
    """Part C: each point (site, scores) simulated on C2 at its fitted parameters on the real wiring, the APL->KC scale
    (with KC->APL and bouton->APL tied) bisected until the number of APL-on responses equals C2's; per point the per-pair
    APL effect (raw, calibrated with K re-calibrated, binarized), the loss and its parts, the projection on the common
    error, the group errors, the response rates and ρ, odors per KC, the class split. Appends to sensitivity_map.csv and
    sensitivity_pairs.csv (resumable)."""
    raise NotImplementedError


def linear_profile(out: Path) -> pd.DataFrame:
    """Part D1: per site, the ridge-penalized score vector whose linear prediction (the map's per-pair changes) cancels
    the common error, |s_g| <= ln 4; a two-site and a sparse version; leave-one-odor-out and split-half cross-validation;
    the gate. Writes profiles.csv and profile_cv.csv."""
    raise NotImplementedError


def simulate_profiles(out: Path) -> pd.DataFrame:
    """Part D2: the chosen profiles at 0.5, 1 and 1.5 times their strength, the leave-one-odor-out profiles on their
    held-out pairs, permuted controls; scored as part C."""
    raise NotImplementedError


def anatomy(out: Path) -> pd.DataFrame:
    """Part F: the scores against the hemibrain's and the male CNS's per-glomerulus APL densities and their ratio, each
    glomerulus's drive, E17's co-drive score, E15's eight, the gain sets. Writes anatomy.csv."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['A-score', 'B', 'C', 'D1', 'D2', 'F'])
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    {'B': archive_vs_common, 'C': sensitivity_map, 'D1': linear_profile, 'D2': simulate_profiles, 'F': anatomy}[args.part](out)


if __name__ == '__main__':
    main()
