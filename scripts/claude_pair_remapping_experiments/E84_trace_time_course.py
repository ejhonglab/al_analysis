#!/usr/bin/env python3
"""E84, the time course of the APL effect in the flies' traces (no simulations). The design is the E84 entry in
experiment_priorities.md (planned by a Fable subagent, claude-fable-5-2, spawned from viki 2026-11-11 10:08).

Data permission (Yang, 2026-11-11 10:07, notes/yang_drive_data_agreement.md item 2): this script reads, read-only and
on viki only, the suite2p output of the 19 recordings of olfdb datasets 19 / 20 (`yang_response_normalization.RECORDINGS`)
by their exact names, through `yang_response_normalization.read_suite2p` (the reader that script uses too). It accepts
no other recording name and lists nothing on the drives. The per-trial excerpts are cached in
outputs/E84_trace_time_course/cache/ (mode 700, Yang's data: never archived or copied). What goes to the shared
directory (results/E84_trace_time_course/) are statistics over many KCs only: per-window distances and effects with
their intervals, per-recording kinetics measures, mean traces over several hundred KC x odor entries or more, the
records as recorded and summary counts of the presentation order (the per-recording order stays in
outputs/E84_trace_time_course/local/ until Yang releases it).

Units. A KC's signal is Fc = F - c Fneu (c = 0.7, Yang's; part E also 0.5 and 0.9) for the ROIs kept as cells. A trial's
excerpt runs from 5 s before the onset to 40 s after it (the next onset is 47 s later), its 5 s pre-onset mean
subtracted; it is read in three units: Yang's z (over the KC's whole-trace SD, `sd_full`), baseline-noise units (over
the SD of Fc within the 51 pre-odor windows, each window's mean removed, `sd_base`; the realism bands' units) and
unscaled (the change in Fc). A window [a, b) s is the volumes round(a v) .. round(b v) - 1 after the onset volume
(v = 2.964 volumes / s), so the target's window, 2.0 to 8.1 s, is volumes 6 to 23, as Yang's 2-8 s.

- (A) the reader and the records: the cache; the reproduction check (the target's window reproduces olfdb's per-trial
  values to 1e-5 in every recording and the target's two matrices to 1e-6; the noise-free distances of that window
  reproduce E81's registry entries; the baseline-noise values reproduce data/yang_apl16_kc_noise_units/); the records
  as recorded in each recording's YAML file (genotype, label note, induction time, rates) and the order of the 51
  presentations (records_summary.csv, order_summary.csv; reproduction_checks.json).
- (B) the target by window: per window (1 s bins from -2 to 20 s; the pulse 0-2 s, 2-4 s, 4-8.1 s, the target's window,
  0-8.1 s, 8.1-10.1 s, 10.1-12.2 s) and recording, the cosine distances of the trial means and the noise-free ones
  (`kc_remapping_harness.noise_free_pair_distances`); per window the mean distances, the per-pair APL effect (mean,
  ester / alcohol groups, the six pairs with near-identical ORN input, the block), ρ on the per-pair APL effect with
  the target's, the floor of the loss and the ceiling of ρ, with 90 % intervals over resamples of recordings and of
  flies, in Yang's z and in baseline-noise units, for every subset (effect_by_window.csv, pairs_by_window.csv); the
  eight wirings' mean (E72's common error) and the populations of four against each window (common_error_by_window.csv);
  every archived variant against the pulse's and the 4-8.1 s effect (variants_by_window.csv); response rates per window
  at a criterion matched on the solvent's presentations (rates_by_window.csv).
- (C) the shape of the responses: mean traces of responders (chosen on two repeats, averaged on the third), entries
  below the criterion and all entries, per genotype, recording, odor group and class of KC by odors per KC
  (traces.csv); per trace the time of half rise and peak, the value at 2, 8.1, 20 and 40 s over the peak, the late /
  early signal and a decay time constant, also at matched amplitude (kinetics_by_recording.csv).
- (D) the kernel and the models: kernels fitted to the responders' mean traces under an onset burst, a steady drive
  over the pulse and each saved model's spike-time histogram (kernel_fits.csv); E82's readings redone through the
  fitted kernels after E82's primary kernel reproduces protocol_readings.csv exactly
  (protocol_readings_measured_kernel.csv); the models' predicted APL-off / APL-on signal by window against the flies'
  (trace_ratio_tests.csv); a recovery test on synthetic traces (kernel_recovery.csv).
- (E) checks: the carry-over from the previous presentation (carry_over.csv) and the neuropil coefficient
  (neuropil_by_window.csv).

Subsets in every table: all 19; APL output blocked (TNT) without 2024-07-17 fly 3 L (its YAML says "APL not labeled");
the six typical and the two strongest (E76); the control flies by label group as recorded in the YAML files (7 / 2 / 2)
and as in the queue's notes (6 / 3 / 2; which side of 2024-07-10 fly 2 is the notes' third TNTin-nolabel recording is
not known, so both readings).

Usage (from this folder; one thread, no simulations):
    OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 nice -n 10 ../../venv/bin/python -u E84_trace_time_course.py --part A
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
import yang_response_normalization as YRN

EXPERIMENT = 'E84_trace_time_course'
ODORS: List[str] = list(H.APL16_ODORS)
SOLVENT = 'pfo'
GENOS = ('TNT3c', 'TNT')
GLAB = {'TNT3c': 'control flies', 'TNT': 'APL output blocked (TNT)'}
NEUCOEFF = 0.7
NEUCOEFFS = (0.5, 0.7, 0.9)
EXCERPT_S = (5.0, 40.0)
UNITS = ('z', 'noise', 'dfc')
ESTIMATORS = ('measured', 'noise_free')
BINS: Tuple[Tuple[float, float], ...] = tuple((float(a), float(a + 1)) for a in range(-2, 20))
NAMED_WINDOWS: Dict[str, Tuple[float, float]] = {
    'pulse': (0.0, 2.0), '2-4': (2.0, 4.0), '4-8.1': (4.0, 8.1), 'target': (2.0, 8.1), '0-8.1': (0.0, 8.1),
    '8.1-10.1': (8.1, 10.1), '10.1-12.2': (10.1, 12.2)}
STRONG_TNT = (('2024-07-17', 3, 'L'), ('2024-07-25', 1, 'L'))
UNLABELED_TNT = ('2024-07-17', 3, 'L')
N_BOOT = 1000


def out_dir() -> Path:
    """results/E84_trace_time_course/ in the shared directory (statistics over KCs only)."""
    return H.results_dir(EXPERIMENT)


def local_dir() -> Path:
    """outputs/E84_trace_time_course/local/ (mode 700): tables that stay on viki (the presentation order)."""
    raise NotImplementedError


def cache_dir() -> Path:
    """outputs/E84_trace_time_course/cache/ (mode 700): the per-trial excerpts, Yang's data."""
    raise NotImplementedError


def rec_key(genotype: str, rec: str) -> Tuple[str, int, str]:
    """(date, fly_num, hemibrain) of a recording name, as olfdb keys it."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# part A: the reader, the cache and the records

def excerpt_recording(genotype: str, rec: str, drive_root: Path) -> dict:
    """One recording read (`YRN.read_suite2p`) and cut into per-trial excerpts: dfc (trials, KCs, samples) float32 at
    c = 0.7, dfneu (the same of Fneu), sd_full / sd_base per c in NEUCOEFFS, base_level (each trial's pre-onset mean
    of Fc and Fneu), the odor and repeat of each trial in time order, the onset volumes, the volume rate, the records
    from the YAML, median F / Fneu."""
    raise NotImplementedError


def build_cache(drive_root: Path, rebuild: bool = False) -> None:
    """Writes one cache file per recording (`excerpt_recording`), one recording in memory at a time."""
    raise NotImplementedError


def load_cache(genotype: str, rec: str) -> dict:
    """A recording's cached excerpts and metadata (`excerpt_recording`'s dict)."""
    raise NotImplementedError


def trial_array(c: dict, window: Tuple[float, float], units: str = 'z', neucoeff: float = NEUCOEFF,
                odors: Sequence[str] = ODORS) -> np.ndarray:
    """(repeats, KCs, odors) of a recording's single-trial responses in `window` (seconds) and `units`."""
    raise NotImplementedError


def part_a(drive_root: Path, rebuild: bool = False) -> dict:
    """Part A: the cache, the reproduction checks (RuntimeError on failure), records_summary.csv, order_summary.csv,
    reproduction_checks.json; the per-recording order to local_dir()."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# part B: the target by window

def windows() -> List[Tuple[str, float, float]]:
    """(name, start, end) of every window of part B: the 1 s bins, then NAMED_WINDOWS."""
    raise NotImplementedError


def recording_tables() -> Dict[str, pd.DataFrame]:
    """{genotype: rows recordings (date, fly_num, hemibrain), columns (units, estimator, window, odor_a, odor_b)} of
    per-recording pair distances by window, from the cache (cached in outputs/ as a parquet)."""
    raise NotImplementedError


def subsets() -> Dict[str, Dict[str, list]]:
    """{subset name: {genotype: [recording keys]}} of the subsets every table is given for."""
    raise NotImplementedError


def part_b(n_boot: int = N_BOOT) -> None:
    """Part B: effect_by_window.csv, pairs_by_window.csv, common_error_by_window.csv, variants_by_window.csv,
    rates_by_window.csv."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# part C: the shape of the responses

def part_c() -> None:
    """Part C: traces.csv, kinetics_by_recording.csv."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# part D: the kernel and the models

def part_d() -> None:
    """Part D: kernel_fits.csv, protocol_readings_measured_kernel.csv (after the identity check against E82's
    protocol_readings.csv), trace_ratio_tests.csv, kernel_recovery.csv."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# part E: checks

def part_e(n_boot: int = N_BOOT) -> None:
    """Part E: carry_over.csv, neuropil_by_window.csv."""
    raise NotImplementedError


def tables() -> str:
    """comparison_table.md from the parts' CSVs."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=('A', 'B', 'C', 'D', 'E', 'table'), required=True)
    ap.add_argument('--drive-root', type=Path, default=Path('/media/yang'))
    ap.add_argument('--rebuild-cache', action='store_true')
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    args = ap.parse_args()
    raise NotImplementedError


if __name__ == '__main__':
    main()
