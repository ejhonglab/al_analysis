#!/usr/bin/env python3
"""E70: the high-firing KCs that survive APL. What in the real wiring holds the mean APL effect down,
and how few cells a model's APL-effect loss rests on.

Definitions (fixed before any simulation; part A repeats its tables at 10 spikes as a check): the
high-firing set is the KCs with 20 or more spikes to some odor with APL off (acute); the survivors are
those of them that also reach 20 spikes to some odor with APL on; "left out" removes KCs from both
conditions' spike-count matrices after the fact, with no simulation. Scores: the APL-effect loss
(`kc_remapping_harness.apl_effect_decomposition`: loss, mean, per-odor and pair-specific parts, ρ on the
per-pair APL effect) on raw spike counts and under E65's calibrated saturating readout (n / (n + K), K set
on the APL-on counts of the KCs kept so that the responders-only participation ratio equals the control
flies'; `calibrate_readout`).

Part A (no simulations; the model-output archive in <shared>/data/model_outputs/):
  A1 how few cells: per variant, the loss and its parts after leaving out the k highest-firing KCs (largest
     APL-on count), k = 1, 2, 5, 10, 20, 50, and after leaving out the survivors, the rest of the
     high-firing set and the whole set; the largest change of the loss at k <= 2 and k <= 20.
  A2 the classes on each wiring (key models and their shuffles): each condition's mean similarity and each
     pair's APL effect split by KC class; the five worst pairs of the per-odor profile of the APL effect
     (E66) and each class's share of their error; the survivors' tuning and wiring against the shuffles'
     and the non-surviving high-firing KCs'; the block and the natural-source alignment with the
     survivors left out; survivors.csv (KC, type, claws by glomerulus, largest counts).
  A3 the classes against the odors-per-KC band, and the wiring features separating the KCs silent with
     APL off from those responding to 8 or more odors (also the KC-KC-held fits, E53, against their seed).
  A4 across the archive, at 6.8 % and at 5 %: survivors and their share of the squared APL-on vectors
     against the mean APL effect (both definitions), the pattern part, the rate ratio, the KC-KC scale and
     the odors-per-KC distances; rank and partial rank correlations.
  A5 the single-fly yardstick: one control recording with one APL-blocked recording against the mean of
     the others (88 pairs), beside the model as one draw and as the mean over a class's draws.
Part B (fixed-scale simulations, the runner of E57 / E54's `run_variants`): at E44's 40 Hz and 70 Hz
optima, the survivors' / the high-firing set's claws re-drawn on the real wiring
(`al_analysis.wpnkc_shuffles.redraw_kc_subset`), type- and claw-count-matched controls, the survivors'
real claws put back into E47's within-type claw-permutation draws (`restore_kc_subset`), and an oracle
with the survivors' APL->KC weights x2 and x4; three seeds per arm; somatic threshold re-tuned to 6.8 %.
Part C (only if part A finds E67's best fit moves by >= 0.01 with its survivors left out): the same arms
at that fit's scales.

Outputs: <shared>/results/E70_high_firing_kcs/ (leave_out.csv, classes.csv, class_similarity.csv,
worst_pairs.csv, wiring_features.csv, opk_classes.csv, survivors.csv, archive_relations.csv,
yardstick.csv, stats_variants.csv, comparison_table.md, figures, provenance.json) and
./outputs/E70_high_firing_kcs/ (the re-drawn wirings, swaps, per-variant responses / spike counts).
Usage: OMP_NUM_THREADS=2 nice -n 10 ./venv/bin/python E70_high_firing_kcs.py --part A
       OMP_NUM_THREADS=12 nice -n 10 ./venv/bin/python E70_high_firing_kcs.py --part B --models thr40_gain1.5
       ./venv/bin/python E70_high_firing_kcs.py --part B --plan     (the wirings only, no simulation)
       ./venv/bin/python E70_high_firing_kcs.py --summary
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E70_high_firing_kcs'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
RESULTS = H.shared_dir() / 'results'
HIGH_SPIKES = 20.0
CHECK_SPIKES = 10.0
K_LEAVE_OUT = (1, 2, 5, 10, 20, 50)
CLASSES = ('survivor', 'high_firing_other', 'responder', 'silent')
READOUTS = ('raw', 'cal')


def load_counts(folder: str, tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A variant's archived APL-on and acute APL-off spike counts (rows KCs, (kc_id, kc_type) index;
    columns odor abbreviations in APL16 order)."""
    raise NotImplementedError


def classify_kcs(on: pd.DataFrame, off: pd.DataFrame, spikes: float = HIGH_SPIKES) -> pd.Series:
    """Each KC's class: 'survivor' (>= `spikes` to some odor with APL off and with APL on),
    'high_firing_other' (>= `spikes` with APL off only), 'responder' (responds, >= 2 spikes, to some odor
    in either condition, otherwise), 'silent'."""
    raise NotImplementedError


def score(on: pd.DataFrame, off: pd.DataFrame, readout: str = 'raw', data: Optional[pd.Series] = None) -> Dict[str, float]:
    """The APL-effect loss with its parts (`apl_effect_decomposition`), the mean distances with APL on and
    off, and K for the calibrated readout ('cal'; calibrated on `on`, i.e. on the KCs kept)."""
    raise NotImplementedError


def leave_out_rows(folder: str, tag: str, spikes: float = HIGH_SPIKES) -> List[dict]:
    """A1 for one variant: one row per (readout, leave-out) with the scores and the change from the full
    set."""
    raise NotImplementedError


def class_similarity(on: pd.DataFrame, off: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    """Each pair's cosine similarity split into the KC classes' terms (each class's part of the dot
    product over the product of the two odors' lengths), in both conditions, and each class's part of the
    per-pair APL effect."""
    raise NotImplementedError


def wiring_features(W: pd.DataFrame, kcs: pd.Index, gains: Optional[pd.Series] = None,
    apl: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per KC: claws, distinct glomeruli, claws per glomerulus, claws on the eight gained glomeruli and on
    DM5, DM1, DP1m, KC type, APL weights (connectome and breadth)."""
    raise NotImplementedError


def yardstick() -> pd.DataFrame:
    """A5: the KC data's APL-effect loss and parts for each (control recording, APL-blocked recording)
    pair against the mean of the other recordings."""
    raise NotImplementedError


def part_a(spikes: Sequence[float] = (HIGH_SPIKES, CHECK_SPIKES)) -> None:
    raise NotImplementedError


def part_b_sets(mkey: str) -> Dict[str, pd.Index]:
    """The KC sets of one reference model on the real wiring at its fitted scales: survivors, high-firing
    set, and the matched controls."""
    raise NotImplementedError


def part_b_jobs(models: Sequence[str], seeds: Sequence[int], out_dir: Path, res_dir: Path) -> List[dict]:
    """The arms of part B as `E54_gain_and_lesion.run_variants` jobs."""
    raise NotImplementedError


def summary(res_dir: Path) -> None:
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=['A', 'B', 'C'], default='A')
    ap.add_argument('--models', nargs='+', default=['thr40_gain1.5', 'thr70_gain1.5'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--plan', action='store_true', help='part B / C: build the wirings and the job list only')
    ap.add_argument('--summary', action='store_true')
    ap.parse_args()
    raise NotImplementedError


if __name__ == '__main__':
    main()
