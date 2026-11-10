#!/usr/bin/env python3
"""E72: the breadth-2 APL-effect fit (C2, the APL-effect fits E67) on re-drawn wirings and as a population of flies.

The fits are E10_joint_fit.py runs (part B: C2's family under `--loss apl --apl-readout both` on four within-type claw
permutations P1-P4 and the co-input-preserving shuffle's (E59) three within-type matched wirings M1-M3; part C: the
no-gain 40 Hz base on P1-P4), saved to results/E72_wiring_population/ and outputs/E72_wiring_population/. This script
scores them and builds the population:

- part A (no simulations): every family the archive holds on several wirings, each wiring's APL-effect loss with its
  mean, per-odor and pair-specific parts and ρ on the per-pair APL effect on raw counts, under the calibrated readout
  (K per wiring) and binarized; the same for the mean over wirings of the pair distances per condition, against the
  number of wirings averaged; the pair-by-pair correlation of the pattern error between wirings and between families;
  the split of each wiring's pattern error into the part common to the wirings and its own remainder;
- part B/C scoring: the same for the new fits, beside C2 and the KC data, with the realism bands, the block, the two ρ's
  on odor-odor distances, the natural-source alignment, the largest spike counts and the fitted parameters; the
  population rows (the real wiring with P1-P4, with M1-M3, all eight, the seven draws) in population.csv; the common-
  error pairs; the real wiring's rank; a two-wiring yardstick;
- part D: one parameter set fitted to four wirings' population (the real one, P1, P2, M1) and tested on the four held
  out (P3, P4, M2, M3), in this script, reproducing C2's saved loss with one wiring.

Outputs (results/E72_wiring_population/): stats_variants.csv (single-wiring fits, harness scoring columns),
population.csv, archive_wirings.csv (part A), common_error_pairs.csv, comparison_table.md, figures, provenance.json.
Usage: ./venv/bin/python E72_wiring_population.py [--part A|score|D]
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E72_wiring_population'


def load_fit(experiment: str, tag: str) -> Dict[str, tuple]:
    """{'on': (responses, spike counts), 'off': ...} of one saved fit (outputs dir, else the archive)."""
    raise NotImplementedError


def score_one(outputs: Dict[str, tuple]) -> Dict[str, float]:
    """Every statistic of one wiring's fit (raw, calibrated with its own K, binarized; realism; block; parameters)."""
    raise NotImplementedError


def population_distances(fits: Sequence[Dict[str, tuple]], readout: str = 'raw') -> Dict[str, pd.Series]:
    """{'on', 'off'}: the mean over wirings of the pair distances per condition under `readout` ('raw', 'cal' with K
    per wiring, 'binary'), as the data's are the mean over recordings."""
    raise NotImplementedError


def score_population(fits: Sequence[Dict[str, tuple]]) -> Dict[str, float]:
    """The population's APL-effect loss with its parts under the three readings, its response rate ρ (wiring-mean
    per-odor fractions), odors-per-KC shares averaged over wirings, and the block of the mean matrices."""
    raise NotImplementedError


def curve_over_n(fits: Sequence[Dict[str, tuple]], readout: str = 'raw') -> pd.DataFrame:
    """The population's pattern part and loss against the number of wirings averaged (all subsets: mean and SD)."""
    raise NotImplementedError


def common_error(fits: Sequence[Dict[str, tuple]], readout: str = 'cal') -> pd.DataFrame:
    """Per pair: the pattern error common to the wirings (the error of their mean) and each wiring's remainder."""
    raise NotImplementedError


def two_wiring_yardstick(fits: Sequence[Dict[str, tuple]]) -> pd.DataFrame:
    """APL on from one wiring and APL off from another (all ordered pairs), scored against the data's mean."""
    raise NotImplementedError


def part_a() -> pd.DataFrame:
    """The archive's families on several wirings (E44 / E46 / E50 / E59; E10 / E25 / E30; E42 / E46): single wirings
    against their mean, pattern-error correlations."""
    raise NotImplementedError


def population_fit(wirings: Sequence[str], held_out: Sequence[str], maxfev: int = 150) -> Dict[str, float]:
    """Part D: one set of the four parameters fitted to the population of `wirings` under E67's 'both' loss computed
    on the population, then run once on `held_out`."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', default='score', choices=['A', 'score', 'D'])
    ap.parse_args()
    raise NotImplementedError('E72: interface only so far')


if __name__ == '__main__':
    main()
