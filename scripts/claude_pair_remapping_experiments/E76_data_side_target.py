#!/usr/bin/env python3
"""E76: the data side of the main target. The per-pair APL effect in fixed units of the recordings and across the
recordings with APL output blocked (TNT), every archived model against it, and the firing-ceiling family fitted to the
fixed-unit effect.

No mechanism is changed. The experiment measures how much of the APL-effect loss, and of each of its two parts (the mean
APL effect and the pattern part, `kc_remapping_harness.apl_effect_stats`), depends on two choices on the data's side:
the unit in which a KC's responses enter the cosine distance, and which recordings make up the APL-off mean.

The data in fixed units (data/yang_apl16_kc_noise_units/ in the shared directory, used as written, not re-derived): hal's
trial means in baseline-noise units (trialmean_long.parquet: the KC's change in Fc over the SD of its Fc within the 51
pre-odor windows, each window's mean removed; = Yang's z times the KC's `scale_ratio`), viki's un-normalized trial means
(trialmean_long_unnormalized.parquet: the change in Fc, raw suite2p units, = Yang's z times the KC's whole-trace SD) and
per-trial values with each trial's pre-odor Fc (per_trial_long_unnormalized.parquet), and the KC's mean F and Fneu from
viki's notes/yang_response_normalization/kc_scales.csv (`data_targets.INPUTS['yang_norm_kc_scales']`). Units (`UNITS`):

- 'z': Yang's whole-recording z (the target as defined; olfdb's values);
- 'noise': baseline-noise units (the realism bands' data side since Tom's decision of 2026-11-10 16:10);
- 'dfc': the raw change in Fc;
- 'dff_fmean': the change in Fc over the KC's mean F;
- 'dff_fcmean': the change in Fc over the KC's mean Fc (F - 0.7 Fneu), the denominator floored at the 5th percentile of
  the positive values over all KCs of both genotypes;
- 'dff_fctrial': each trial's change in Fc over that trial's pre-odor Fc (floored the same way, over trials), then the
  mean over the KC's three trials.

Parts (CLI `--part`):

- D (no simulations; first, since the glomerulus profile experiment's (E75) gate is being read): per recording of both
  genotypes the mean distance and its noise-corrected value, the response rate in z and in baseline-noise units, the
  ester and alcohol groups' distances, the number of KCs, the induction time, the median scale ratio and drift, the
  control recordings' label group (recordings.csv). For the recordings with APL output blocked (TNT): each one's
  effect against the control flies' mean; the six typical ones and the two strongest (the two with the lowest mean
  distance); every split of the eight into six and two (28) as the null for that split, and every split of the eleven
  control recordings into nine and two (55) for the same steps among the controls. The tests, in z, in baseline-noise
  units and noise-corrected (axis_tests.csv): ρ between the step from the control flies to the six and the step from the
  six to the two; the agreement of the two recordings' steps with each other; the same among the control recordings,
  each with its rank among the splits. Then every archived variant (results/E*/model_pair_distances_long.csv, raw spike
  counts) against the six's effect, the two's and the eight's (loss, pattern part, ρ); E72's common error against each;
  the floor and ρ ceiling of the six alone. For E75 (written, not run here): recording_subsets.csv (per pair the effect
  against the six, the two and the eight, in z and in baseline-noise units, and E72's common error against each) and
  split_halves.csv (2000 half-splits within genotype, E75's `_split_halves` procedure and seed, so that its first 100
  rows are E75 D1's splits, keyed by whether the two strongest fall in one half).
- B (no simulations): the unit. Per unit the two mean matrices and the per-pair APL effect; the mean effect with 90 %
  intervals over recordings and over flies; the sd over pairs; the floor of the loss and of its pattern part; the
  split-half ρ and its ceiling; the ester and alcohol groups' effects relative to the mean with intervals; ρ and the
  pattern difference against the z-based effect with intervals; the same with sub-threshold entries set to 0 and
  binarized (1 z; `noise_threshold`, 1.97 baseline-noise SDs, in baseline-noise units; each other unit at the threshold
  that keeps the control flies' response rate). The split by odors per KC (`kc_remapping_harness.apl_effect_by_class`)
  in baseline-noise units at 1.97, beside E74's in z. Registry entries `noise_apl_effect_*` (`registry_entries`, added
  by `data_targets.py --add-new`).
- C (no simulations): the archive. Every archived variant against the fixed-unit effects on raw counts, under E65's
  calibrated readout with K re-calibrated to the control flies' responders-only participation ratio taken in the same
  unit, and binarized against the data binarized in the same unit; rank correlations with the present scores overall
  and inside the bands (results/apl_effect/by_variant.csv's `realism_flag`); the best unflagged variant per family;
  E72's eight wirings and their mean with the common error recomputed against the baseline-noise effect; the saved
  pipeline rows through the trial noise (E72 part E, E75 part A) where their pair distances exist.
- E (fixed scales, simulations): the model's two axes on C2 (E67) and on E69's fit 2, each first re-run to reproduce its
  archived counts: a graded block (APL's output scales, APL->KC and APL->bouton, times 0.75, 0.5, 0.25, 0.125; the inputs
  to APL unchanged) and excitability (the somatic threshold times 0.8, 0.9, 1.1, 1.2, APL on and off). Per point the
  pair distances on raw counts and under the readout (K held at the model's own calibration), the response rates and
  the odors-per-KC distributions (model_axes.csv); read as step ρs against the data's two steps, the ranges each axis
  covers, the population reading (populations.csv) and each state against its own group's effect.
- F (fits; E10_joint_fit.py, not this script): fit 0 (the present target, `--band-units noise`), fit 1 (`--apl-data-units
  noise`), fit 2 (from E69's fit 3, only if part C confirms the reference set with the ceiling ahead of the breadth-2
  family against the fixed-unit effect).

Outputs in results/E76_data_side_target/: recordings.csv, splits_tnt.csv, splits_ctrl.csv, axis_tests.csv,
subsets_vs_archive.csv, common_error_subsets.csv, recording_subsets.csv, split_halves.csv (part D); data_units.csv,
effect_by_unit_pairs.csv, class_split_units.csv (part B); by_variant_units.csv, common_error_units.csv,
rank_correlations_units.csv, best_by_family_units.csv, pipeline_units.csv (part C); model_axes.csv, populations.csv
(part E); comparison_table.md; figures; provenance_<part>.json.
Usage: ./venv/bin/python E76_data_side_target.py --part D | B | C | E [--threads N]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E76_data_side_target'
NOISE_DIR = H.shared_dir() / 'data' / 'yang_apl16_kc_noise_units'
UNITS = ('z', 'noise', 'dfc', 'dff_fmean', 'dff_fcmean', 'dff_fctrial')
ESTER_GROUP = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL_GROUP = ('1-8ol', '1-5ol', '1-6ol', '6al')
GENOTYPES = ('TNT3c', 'TNT')
# the floor on the F-normalized units' denominators: this percentile of the positive values over all KCs (or trials)
DENOM_FLOOR_PCT = 5.0
SPLIT_SEED = 0
N_HALF_SPLITS = 2000


def unit_matrices(unit: str) -> Dict[str, Dict[tuple, pd.DataFrame]]:
    """{genotype: {(date, fly_num, hemibrain): KCs x the 16 odors (APL16_ODORS order) trial means in `unit`}}, recordings
    in sorted order, KCs indexed by olfdb's roi (as `data_targets.kc_response_matrices`)."""
    raise NotImplementedError


def unit_recording_rdms(unit: str, values: str = 'graded') -> Dict[str, pd.DataFrame]:
    """{genotype: per-recording cosine pair distances (rows recordings, columns canonical pairs)} in `unit`; `values`
    'graded', 'thresholded' (sub-threshold entries set to 0) or 'binary', at `unit_threshold(unit)`."""
    raise NotImplementedError


def unit_threshold(unit: str) -> float:
    """The response criterion in `unit`: 1 in z, the registry's noise_threshold in baseline-noise units, else the
    threshold that keeps the control flies' mean response rate at the one under Yang's criterion."""
    raise NotImplementedError


def strongest_recordings(tnt: pd.DataFrame, n: int = 2) -> List[tuple]:
    """The `n` recordings with APL output blocked (TNT) with the lowest mean distance in Yang's z (per-recording pair
    distances `tnt`)."""
    raise NotImplementedError


def group_effects(effect: pd.Series) -> Dict[str, float]:
    """The ester and alcohol groups' mean effect relative to the mean over all pairs (`ester_rel`, `alcohol_rel`), and
    their absolute means."""
    raise NotImplementedError


def recordings_table() -> pd.DataFrame:
    """Part D: one row per recording of both genotypes (see the module docstring)."""
    raise NotImplementedError


def split_stats(recs: Dict[str, pd.DataFrame], two: Sequence[tuple], genotype: str = 'TNT') -> Dict[str, float]:
    """The steps of one split of `genotype`'s recordings into `two` and the rest: ρ between the step from the control
    flies to the rest (or, among the controls, the full APL effect) and the step from the rest to the two, the agreement
    of the two recordings' own steps, the steps' means and sd over pairs, and the groups' relative steps."""
    raise NotImplementedError


def split_null(recs: Dict[str, pd.DataFrame], genotype: str = 'TNT') -> pd.DataFrame:
    """`split_stats` of every split of `genotype`'s recordings into two and the rest, with the actual split (the two
    strongest, or the two lowest-level controls) marked and ranked."""
    raise NotImplementedError


def axis_tests(out: Path) -> pd.DataFrame:
    """Part D's tests in each reading (z, baseline-noise units, noise-corrected): writes axis_tests.csv, splits_tnt.csv
    and splits_ctrl.csv."""
    raise NotImplementedError


def archive_pairs() -> Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]]:
    """{(folder, tag): (APL-on pairs, acute APL-off pairs)} of every archived variant (raw spike counts), as
    `apl_effect_scores.model_pairs`, with at least 100 pairs."""
    raise NotImplementedError


def score_against(pairs: Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]], targets: Dict[str, pd.Series]) -> pd.DataFrame:
    """One row per variant: for each named target effect its loss, pattern part, mean error and ρ (`<name>_loss`,
    `<name>_pattern`, `<name>_mean_error`, `<name>_rho`), and the model's mean effect."""
    raise NotImplementedError


def common_error_against(targets: Dict[str, pd.Series], readout: str = 'cal') -> pd.DataFrame:
    """E72's common error (the pattern error of the mean over its eight wirings of the pair distances, means removed)
    recomputed against each named target effect, one column per target, with the group means and the correlation with
    the present one."""
    raise NotImplementedError


def recording_subsets(out: Path) -> pd.DataFrame:
    """recording_subsets.csv for E75: per pair the effect against the six, the two and the eight in z and in
    baseline-noise units, and E72's common error (calibrated readout) against each."""
    raise NotImplementedError


def split_halves(out: Path, n_splits: int = N_HALF_SPLITS, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """split_halves.csv for E75: `n_splits` half-splits within genotype drawn as E75's `_split_halves` (one generator
    seeded `seed`; per split a permutation of the 11 control recordings, then of the 8 with APL output blocked (TNT);
    half A the first n // 2), the recordings in each half, the stratum (`strongest_together`: both strongest
    recordings in one half), ρ between the halves' effects, the pattern difference and the groups' differences."""
    raise NotImplementedError


def part_d(out: Path) -> None:
    """Part D: the recordings (see the module docstring)."""
    raise NotImplementedError


def effect_resamples(recs: Dict[str, pd.DataFrame], effect_z: Optional[pd.Series] = None, n_boot: int = 1000,
                     n_split: int = 500, seed: int = 0) -> Dict[str, object]:
    """The per-pair APL effect of per-recording distances `recs` with its 90 % intervals over recordings and over flies
    (`n_boot` resamples each), the floor of the loss and of its pattern part, the split-half ρ and its ceiling
    (`n_split` splits), the groups' effects with intervals and, given `effect_z`, ρ and the pattern difference against
    it with intervals over the same resamples."""
    raise NotImplementedError


def class_split_units(out: Path) -> pd.DataFrame:
    """The split of the per-pair APL effect by odors per KC (`apl_effect_by_class`) in baseline-noise units at
    noise_threshold, beside the same in z at 1 z (E74's)."""
    raise NotImplementedError


def part_b(out: Path) -> None:
    """Part B: the unit (data_units.csv, effect_by_unit_pairs.csv, class_split_units.csv)."""
    raise NotImplementedError


def registry_entries(n_boot: int = 1000, seed: int = 0) -> List[Tuple[str, float, str, Optional[Tuple[float, float]]]]:
    """The data-target registry's entries of this experiment (data_targets.py, prefix `noise_apl_effect_`): (name, value,
    definition, 90 % interval or None)."""
    raise NotImplementedError


def part_c(out: Path) -> None:
    """Part C: the archive against the fixed-unit effects (by_variant_units.csv, rank_correlations_units.csv,
    best_by_family_units.csv, common_error_units.csv, pipeline_units.csv)."""
    raise NotImplementedError


def part_e(out: Path, threads: int = 12, only: Optional[Sequence[str]] = None, check_only: bool = False) -> None:
    """Part E: the fixed-scale runner on E69's model wrapper (model_axes.csv, populations.csv); each base re-run first to
    reproduce its archived counts. `only` restricts to points by id (a smoke test); `check_only` stops after the
    reproduction checks."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=('D', 'B', 'C', 'E'))
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--only', nargs='*', default=None, help='part E: the point ids to run')
    ap.add_argument('--check-only', action='store_true', help='part E: only the reproduction checks')
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'D':
        part_d(out)
    elif args.part == 'B':
        part_b(out)
    elif args.part == 'C':
        part_c(out)
    else:
        part_e(out, threads=args.threads, only=args.only, check_only=args.check_only)


if __name__ == '__main__':
    main()
