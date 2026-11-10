#!/usr/bin/env python3
"""E74: who carries the APL effect. Each pair's APL effect split by the number of odors a KC responds to,
in the recordings and in every archived model (part A, no simulations); the class-resolved term for the
joint fit (part C, `class_term_stats`, called from E10_joint_fit.py's `--apl-class-term`).

A pair's cosine similarity is a sum over KCs. Each KC is put, in each condition, in a class by the number
of the 16 odors it responds to in that condition (0, 1, 2-3, 4-7, 8 or more: the classes of the
odors-per-KC realism band, `kc_remapping_harness.OPK_EDGES`), and the per-pair APL effect (the distance
with APL on minus with APL off, i.e. the similarity with APL off minus with APL on) becomes a sum of
per-class effects: each condition's class contributions (averaged over its recordings) with APL off minus
with APL on (`kc_remapping_harness.apl_effect_by_class`). A KC changes class between conditions, so a
class's effect says how much of each condition's similarity sits in KCs of that tuning width; no pairing
of cells across flies is needed. Each class's contribution is also split into the responders' products
('both'), the responder x sub-threshold products ('one', E68's term) and the products of sub-threshold
entries ('neither').

Part A (no simulations):

- The data (Yang's APL16 recordings, trial means, 11 control and 8 APL output blocked (TNT) recordings):
  per recording, then the mean over recordings per genotype, at the data's criterion (trial mean > 1 z)
  and at 0.75 and 1.25 z. Per class: the mean effect over the 120 pairs and its share of the effect's
  variance over pairs (cov(class effect, total effect) / var(total effect); the shares add to 1), each with
  a 90 % recording-bootstrap interval (recordings drawn with replacement within each genotype), each
  condition's level, the class sizes (share of KCs per class and condition), the sub-threshold part per
  class. The class-resolved loss (root mean square over the 5 x 120 class-by-pair entries of model minus
  data, lower is better) gets its floor from resampled data sets against the measured one (mean over the
  resamples, as `apl_effect_rmse_floor`) and a single-recording-pair yardstick (each control recording
  with each APL-blocked recording, against the mean of the other recordings, as E70's A5). Registry
  entries in data_targets.py (`apl_class_*`).
- The models: every variant of results/apl_effect/by_variant.csv with both conditions archived
  (data/model_outputs/<folder>/<tag>_<on|acute_off>_spike_counts.parquet):
  'cal<n>': E65's calibrated readout n / (n + K) (K set on the APL-on counts so that the responders'
  participation ratio is the control flies') with an n-spike criterion (n = 2, also 1 and 3), against the
  data at 1 z; 'zcell': E73's reading (per-KC session z-score of the saturated counts, at the calibration
  E73's by_variant row records for the variant: K, noise, noise ratio, q, drift) on 10 pseudo-recordings of
  3 trials drawn with the reading's measurement noise (seed 43: one generator, the APL-on draws then the
  APL-off draws), the data's criterion (trial mean > 1 z) on each pseudo-recording, class contributions
  averaged over the pseudo-recordings, so that class membership carries the data's noise; 'zcell_free':
  the same without the noise (the planning probe's emulation). The key models with saved jittered trials
  (E73's TRIAL_SETS) are also read from their trials ('zcell_trials'). Per variant and reading: the five
  per-class effects (and per term), the class-resolved loss, the APL-effect loss with its mean and
  pattern parts under the same reading, each class's share of the pattern error's variance
  (cov(class error, total error) / var(total error), errors with their means removed), the sub-threshold
  part of the mean (the mean effect in the 'one' and 'neither' terms; and, under the reading, the graded
  mean minus the mean with sub-threshold values set to 0), the growth of the 0.5-1.5 z band (share of
  entries with a trial mean in (0.5, 1.5] z with APL off minus with APL on), and the realism bands under
  the two-spike criterion (results/apl_effect/by_variant.csv) and under the data's criterion through the
  reading (E73's row).
- Tables (comparison_table.md): the key models beside the data (C2, E69's fits 1 and 2, E71's three fits,
  E73 part C's fits if archived, the reference model, the first joint fit); E72's eight wirings singly and
  as their mean (each condition's class contributions averaged over the wirings); across variants the
  Spearman correlation of the class-resolved loss with the APL-effect loss and with its pattern part; the
  archive's sweeps as series on the 4-7 class's effect (series.csv); the variant with the lowest
  class-resolved loss inside the bands; part C's condition (the data's 4-7 interval excludes the best
  fits' values, and the floor of the class-resolved loss is at most half the best fits' class-resolved
  loss).

Part C support: `class_term_stats` gives E10_joint_fit.py's `--apl-class-term` (the mean over the five
classes and 120 pairs of the squared difference between the model's and the data's per-class effect under
the reading, over the square of the class-resolved loss's floor; the pseudo-recordings' noise drawn once,
seed 43, the same at every evaluation).

Outputs in <shared>/results/E74_apl_effect_by_odors_per_kc/: data_by_class.csv (per class, criterion and
term: the mean effect, variance share and levels with 90 % intervals, class sizes), data_by_recording.csv,
data_pairs_by_class.csv (the data's per-pair class effects), floor.csv, yardstick.csv, by_variant.csv (one
row per variant x reading), pairs_key_models.csv, wirings.csv, rank_correlations.csv, series.csv,
comparison_table.md, E74_classes.png, E74_series.png, provenance.json (with H.model_provenance).
Usage: nice -n 10 ./venv/bin/python -u E74_apl_effect_by_odors_per_kc.py [--jobs 4] [--only-key] [--boot 2000]
       [--summarize-only]
"""
from __future__ import annotations

import os

# (only when run as a script: E10_joint_fit.py imports this module for the class term and must keep the
# thread count of its simulations; the worker processes inherit the setting)
if __name__ == '__main__':
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(_v, '1')

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E74_apl_effect_by_odors_per_kc'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
RESULTS = H.shared_dir() / 'results'
CLASSES = ('0', '1', '2_3', '4_7', '8plus')   # the odors-per-KC classes (H.OPK_EDGES)
CRITERIA_Z = (0.75, 1.0, 1.25)
SPIKE_CRITERIA = (1, 2, 3)
N_RECORDINGS, N_TRIALS, SEED = 10, 3, 43


def data_matrices() -> Dict[str, Dict[tuple, pd.DataFrame]]:
    """{genotype: {recording: KCs x odors trial means (APL16 order)}} of Yang's APL16 recordings."""
    raise NotImplementedError


def data_class_terms(thresh: float = H.KC_RESPONSE_Z) -> Dict[str, List[pd.DataFrame]]:
    """{genotype: per recording, `H.class_cosine_terms` at the criterion trial mean > `thresh`}."""
    raise NotImplementedError


def data_class_effect(thresh: float = H.KC_RESPONSE_Z) -> pd.DataFrame:
    """The data's per-pair APL effect by class (`H.apl_effect_by_class`'s 'effect' block: pairs x (class,
    term)), each genotype's class contributions averaged over its recordings."""
    raise NotImplementedError


def class_summary(effect: pd.DataFrame) -> pd.DataFrame:
    """Per class (and 'all'): the mean effect over pairs and its share of the total's variance over pairs."""
    raise NotImplementedError


def data_tables(n_boot: int = 2000, seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Part A's data side: data_by_class (per criterion, class and term: observed, 90 % recording-bootstrap
    interval), data_by_recording, the floor of the class-resolved loss and the yardstick."""
    raise NotImplementedError


def class_loss(model: pd.DataFrame, data: pd.DataFrame) -> float:
    """The class-resolved loss: root mean square over the 5 classes x 120 pairs of model minus data per-class
    effect (the 'total' term of each class; pairs aligned)."""
    raise NotImplementedError


def pattern_error_shares(model: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
    """Each class's share of the variance over pairs of the pattern error (model minus data total effect,
    each with its mean removed): cov(class error, total error) / var(total error); the shares add to 1."""
    raise NotImplementedError


def read_cal(on: pd.DataFrame, off: pd.DataFrame, n_spikes: float = 2.0, K: Optional[float] = None
    ) -> Tuple[pd.DataFrame, dict]:
    """A model under E65's calibrated readout with an n-spike criterion: (`H.apl_effect_by_class` output,
    extras such as K and the class sizes)."""
    raise NotImplementedError


def pseudo_noise(n_units: int, n_odors: int = 16, n_recordings: int = N_RECORDINGS, n_trials: int = N_TRIALS,
    seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """(APL-on, APL-off) standard normal draws (n_recordings, n_trials, n_units, n_odors), one generator
    seeded `seed` (the APL-on draws first); cached per shape."""
    raise NotImplementedError


def read_zcell(on, off, cal: dict, noise: bool = True) -> Tuple[pd.DataFrame, dict]:
    """A model under E73's reading at the calibration `cal` (K, noise, noise_ratio, q, drift), on pseudo-
    recordings through the measurement noise (or noise-free): (`H.apl_effect_by_class` output, extras: the
    graded and thresholded mean effects, the 0.5-1.5 z band shares, the class sizes). `on` / `off`: (KCs x
    odors) counts or (recordings, trials, KCs, odors) jittered trials."""
    raise NotImplementedError


def class_term_stats(s_on: pd.DataFrame, s_off: pd.DataFrame, cal: dict) -> Dict[str, float]:
    """The hook for E10_joint_fit.py's `--apl-class-term`: the model's per-class effects under E73's reading
    at the calibration `cal` (the one the fit's evaluation used) on the fixed pseudo-recordings, against the
    data's at 1 z. Returns apl_class_mse (the term before division by the floor's square), apl_class_rmse,
    apl_class_floor and apl_class_<class> (the model's mean per-class effects)."""
    raise NotImplementedError


def score_variant(job: tuple) -> List[dict]:
    """One archived variant under every reading (rows; an exception is returned as one row with `error`)."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == '__main__':
    main()
