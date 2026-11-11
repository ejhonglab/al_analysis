#!/usr/bin/env python3
"""E81 parts D and E: the model's trial-to-trial variability re-specified with the flies' shares, and what the state
shared by the odors of a repeat could be in the model (the flies' variability across the three repeats, E81).

Stub (interface first); the bodies follow.

Part D (`--part D`, no simulations): for each model with pipeline trials (E43_model_trial_noise.py part B: per-KC
threshold jitter, one draw per KC and trial shared by the 16 odors, ten pseudo-recordings of three trials), the shared
part of the variability is the jitter, interpolated between the saved levels (and the deterministic run as level 0),
at which the model's same-repeat gap is the flies'; the independent part is added to the saved trial counts (a
lognormal factor per KC x odor x trial, and Gaussian noise of the flies' silent entries' size in the readout's units,
E68's conversion); the factor's spread is set so that the same-odor distance between two trials is the flies'. Checks
not used in the calibration, scores (single wirings and populations of four, raw counts and the calibrated readout,
Yang's z and baseline-noise units) and the re-reads of the block through the noise.

Part E (`--part E`, simulations; `--part E-tables` reads them): a threshold factor common to all KCs as the shared
state, on C2 (E67) and E69's fit 1, APL on and off; and the run-down placed on the model's two axes of E76 part E.

`--part table` writes model_noise_table.md.
Usage (from this folder): ../../venv/bin/python -u E81_model_noise.py --part D | E-tables | table
       OMP_NUM_THREADS=12 OMP_THREAD_LIMIT=12 nice -n 10 ../../venv/bin/python -u E81_model_noise.py --part E --threads 12
           [--smoke]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E81_repeat_structure'


def trial_kind_stats(X: np.ndarray) -> Dict[str, object]:
    """Single-trial cosine statistics of one recording, X (trials, units, odors): same-odor distances across repeats,
    two-odor distances within and across repeats, the same-repeat gap, the noise-free distances and the distances of
    the trial means."""
    raise NotImplementedError


def flies_targets(unit: str) -> Dict[str, object]:
    """The flies' targets and checks in `unit` ('z' or 'noise'), per genotype."""
    raise NotImplementedError


def model_levels(model: str) -> List[dict]:
    """The saved levels of `model` (the deterministic run as level 0, every pipeline level found)."""
    raise NotImplementedError


def mix_trials(counts: np.ndarray, s: float, gauss_sd: float, readout: str, K: Optional[float], eps: np.ndarray,
               eta: np.ndarray) -> np.ndarray:
    """The independent part added to saved trial counts: counts x exp(s eps - s^2 / 2), read out, plus gauss_sd x eta."""
    raise NotImplementedError


def calibrate(model: str, readout: str, unit: str, cond: str) -> dict:
    """The jitter (interpolated between levels) and the lognormal spread at which the model's same-repeat gap and
    same-odor distance are the flies'."""
    raise NotImplementedError


def part_d(out: Path, args) -> None:
    raise NotImplementedError


def part_e(out: Path, threads: int = 12, smoke: bool = False) -> None:
    raise NotImplementedError


def part_e_tables(out: Path) -> None:
    raise NotImplementedError


def part_table(out: Path) -> None:
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', required=True, choices=('D', 'E', 'E-tables', 'table'))
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    {'D': lambda: part_d(out, args), 'E': lambda: part_e(out, args.threads, args.smoke), 'E-tables': lambda: part_e_tables(out),
     'table': lambda: part_table(out)}[args.part]()


if __name__ == '__main__':
    main()
