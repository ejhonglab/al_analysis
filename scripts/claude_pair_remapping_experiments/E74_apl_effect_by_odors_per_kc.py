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
  loss); the per-condition table (per_condition.csv): each class's similarity level with APL on and with
  APL off for the key models beside the data's (control flies, APL output blocked (TNT)) with intervals,
  under both readings, marking the model values outside the data's interval; who is released in the key
  models by claw count (released_by_claws.csv, E71's groups on the bouton-base wiring's claws per KC: each
  group's share of the responses blocking APL adds, under the two-spike criterion and through the reading).

Part C support: `class_term_stats` gives E10_joint_fit.py's `--apl-class-term` (the mean over the five
classes and 120 pairs of the squared difference between the model's and the data's per-class effect under
the reading, over the square of the class-resolved loss's floor, or with `--apl-class-form lin` its root over
the floor; the pseudo-recordings' noise drawn once, seed 43, the same at every evaluation). `--check-e10
<off|on|lin>` runs E73 part C's fit 2 for one evaluation at its optimum with the class term off, on in the
squared form or on in the linear form (outputs/E74_apl_effect_by_odors_per_kc/e10_check/option_<...>/).

Outputs in <shared>/results/E74_apl_effect_by_odors_per_kc/: data_by_class.csv (per class, criterion and
term: the mean effect, variance share and levels with 90 % intervals, class sizes), data_by_recording.csv,
data_pairs_by_class.csv (the data's per-pair class effects), floor.csv, yardstick.csv, by_variant.csv (one
row per variant x reading), pairs_key_models.csv, per_condition.csv, released_by_claws.csv, wirings.csv,
rank_correlations.csv, series.csv, comparison_table.md, E74_classes.png, E74_series.png, provenance.json (with
H.model_provenance).
Usage: nice -n 10 ./venv/bin/python -u E74_apl_effect_by_odors_per_kc.py [--jobs 4] [--only-key] [--boot 2000]
       [--summarize-only] [--check-e10 off|on|lin]
"""
from __future__ import annotations

import os

# (only when run as a script: E10_joint_fit.py imports this module for the class term and must keep the
# thread count of its simulations; the worker processes inherit the setting)
if __name__ == '__main__':
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(_v, '1')


import argparse
import json
import pickle
import socket
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E74_apl_effect_by_odors_per_kc'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
RESULTS = H.shared_dir() / 'results'
CLASSES = tuple(H.OPK_CLASSES)        # '0', '1', '2_3', '4_7', '8plus'
CLASSES_ALL = CLASSES + ('all',)
CLASS_LABEL = {'0': 'none', '1': '1', '2_3': '2 to 3', '1_3': '1 to 3', '4_7': '4 to 7', '8plus': '8 or more', 'all': 'all KCs'}
TERMS = tuple(H.CLASS_TERMS)          # 'both', 'one', 'neither', 'total'
ALL, TOTAL = len(CLASSES), TERMS.index('total')
ONE, NEITHER = TERMS.index('one'), TERMS.index('neither')
CRITERIA_Z = (0.75, 1.0, 1.25)
SPIKE_CRITERIA = (1, 2, 3)
N_RECORDINGS, N_TRIALS, SEED = 10, 3, 43
BAND = (0.5, 1.5)
GENOTYPES = ('TNT3c', 'TNT')
GENO_LABEL = {'TNT3c': 'control flies', 'TNT': 'APL output blocked (TNT)'}
ODORS = list(H.APL16_ODORS)
PAIRS = H.pair_list(ODORS)
PAIR_INDEX = pd.MultiIndex.from_tuples(PAIRS, names=['odor_a', 'odor_b'])
IU = np.triu_indices(len(ODORS), 1)
READINGS = ('cal2', 'cal1', 'cal3', 'zcell', 'zcell_free', 'zcell_trials')
READING_LABEL = {'cal1': 'calibrated readout, one-spike criterion', 'cal2': 'calibrated readout, two-spike criterion',
    'cal3': 'calibrated readout, three-spike criterion',
    'zcell': "per-KC reading (E73) through the measurement noise (10 pseudo-recordings of 3 trials), the data's criterion",
    'zcell_free': "per-KC reading (E73) without the noise, the data's criterion",
    'zcell_trials': "per-KC reading (E73) of the saved jittered trials through the measurement noise"}
# key models: (short, label, results / archive folder, tag); those not archived are skipped
KEY_MODELS = [
    ('C2', "C2: breadth exponent 2, 6.8 %, APL-effect term under the calibrated readout (E67)", 'E67_apl_effect_fits',
     'rate0.068_aplcal_k2_cap3'),
    ('C2 both', "C2's family refitted on both readings (E67)", 'E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3'),
    ('ceiling 1', "C2's family with the firing ceiling at strength 1 (E69's fit 1)", 'E69_kc_firing_ceiling',
     'rate0.068_aplboth_refr3_adapt0.04142t1000_k2'),
    ('ceiling 2', "the same at strength 2 (E69's fit 2)", 'E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.1114t1000_k2'),
    ('scaling 0.5', "C2's family with the input-scaled threshold at 0.5 (E71)", 'E71_input_scaled_threshold',
     'rate0.068_aplboth_thrin0.5_k2'),
    ('ceiling + scaling', "the ceiling at strength 1 with the input-scaled threshold at 0.25 (E71's fit 2)",
     'E71_input_scaled_threshold', 'rate0.068_aplboth_refr3_adapt0.04142t1000_thrin0.25_k2'),
    ('ref. + scaling 0.75', 'the reference set with the input-scaled threshold at 0.75 (E71)', 'E71_input_scaled_threshold',
     'thr40_gain1.5_rate0.068_aplboth_thrin0.75_k2'),
    ('reading C2', "C2's family fitted under the per-KC reading (E73 part C)", 'E73_per_kc_normalization',
     'zcaplrate_rate0.068_aplzcell_k2_cap3'),
    ('reading ceiling', 'the ceiling family fitted under the per-KC reading (E73 part C)', 'E73_per_kc_normalization',
     'zcaplrate_rate0.068_aplzcell_refr3_adapt0.04142t1000_k2_cap3'),
    ('fit S', 'the ceiling family with the input-scaled threshold at 0.25 fitted under the per-KC reading (E74 fit S)', EXPERIMENT,
     'zcaplrate_rate0.068_aplzcell_refr3_adapt0.04142t1000_thrin0.25_k2'),
    ('part C fit 1', 'the ceiling family fitted under the per-KC reading with the class term, linear form (E74 part C, fit C1)',
     EXPERIMENT, 'zcaplrate_rate0.068_aplzcellclslin_refr3_adapt0.04142t1000_k2_cap3'),
    ('part C fit 2', 'the same with the input-scaled threshold at 0.25, from fit S (E74 part C, fit C2)', EXPERIMENT,
     'zcaplrate_rate0.068_aplzcellclslin_refr3_adapt0.04142t1000_thrin0.25_k2'),
    ('ref.', 'the reference model (eight glomeruli at 1.5, 40 Hz per-claw threshold, 6.8 %; E44)', 'E44_operating_point',
     'thr40_gain1.5_rate0.068_k2_cap3'),
    ('first joint fit', 'the first joint fit (breadth exponent 2 with KC-KC inhibition, 5 %; E10)', 'E10_joint_fit', 'k2'),
]
# the fits part C's condition is checked on (as archived)
BEST_FITS = ('C2', 'C2 both', 'ceiling 1', 'ceiling 2', 'ceiling + scaling', 'reading C2', 'reading ceiling', 'fit S')
# E72's wirings (the real one and seven re-drawn: P = claw permutation within type, M = the co-input-preserving matched
# curveball): (group, label, [(wiring, folder, tag)]); the population fit (one parameter set) on all eight, C2's family
# refitted on each wiring, and the ceiling at strength 1 refitted on P1, P2 and M1 (E72 part G)
WIRING_FOLDER = 'E72_wiring_population'
_DRAW_TAG = {'P1': 'wiring-claw_permutation_within_type-seed1_', 'P2': 'wiring-claw_permutation_within_type-seed2_',
    'P3': 'wiring-claw_permutation_within_type-seed3_', 'P4': 'wiring-claw_permutation_within_type-seed4_',
    'M1': 'wiring-curveball_within_type_matched-seed1_', 'M2': 'wiring-curveball_within_type_matched-seed2_',
    'M3': 'wiring-curveball_within_type_matched-seed3_'}
WIRING_NAMES = ('real', 'P1', 'P2', 'P3', 'P4', 'M1', 'M2', 'M3')
WIRING_GROUPS = [
    ('popfit', 'the population fit (one parameter set) on each wiring', [(w, WIRING_FOLDER, f'popfit_{w}') for w in WIRING_NAMES]),
    ('c2', "C2's family refitted on each wiring", [('real', 'E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')]
        + [(w, WIRING_FOLDER, _DRAW_TAG[w] + 'rate0.068_aplboth_k2_cap3') for w in WIRING_NAMES[1:]]),
    ('ceiling', 'the ceiling at strength 1 refitted on each wiring (E72 part G)', [('real', 'E69_kc_firing_ceiling',
        'rate0.068_aplboth_refr3_adapt0.04142t1000_k2')] + [(w, WIRING_FOLDER, _DRAW_TAG[w] + 'rate0.068_aplboth_refr3_adapt0.04142t1000_k2_cap3')
        for w in ('P1', 'P2', 'M1')]),
]
WIRINGS = [(w, t) for w, f, t in WIRING_GROUPS[0][2]]
# the archive's sweeps read as series: (handle, series label, folder, tag pattern with the parameter as group x, base tags
# with their parameter value)
SERIES = [
    ('breadth exponent', 'E19, 40 Hz claw threshold, KC-KC 0', 'E19_thr_breadth_kckc', r'^thr40_k(?P<x>[-\d.]+)_kckc0$', {}),
    ('breadth exponent', 'E19, 40 Hz claw threshold, KC-KC 30', 'E19_thr_breadth_kckc', r'^thr40_k(?P<x>[-\d.]+)_kckc30$', {}),
    ('breadth exponent', 'E19, 40 Hz claw threshold, KC-KC 100', 'E19_thr_breadth_kckc', r'^thr40_k(?P<x>[-\d.]+)_kckc100$', {}),
    ('breadth exponent', 'E19, 70 Hz claw threshold, KC-KC 30', 'E19_thr_breadth_kckc', r'^thr70_k(?P<x>[-\d.]+)_kckc30$', {}),
    ('breadth exponent', 'E12, breadth by spikes', 'E12_breadth_apl', r'^spikes_k(?P<x>[-\d.]+)$', {'k0': 0.0}),
    ('breadth exponent', 'E12, breadth by odors', 'E12_breadth_apl', r'^n_odors_k(?P<x>[-\d.]+)$', {'k0': 0.0}),
    ('KC-KC scale', 'E8, plain', 'E8_kc_kc_inhibition', r'^plain_tau0\.05_kckc(?P<x>[\d.]+)$', {}),
    ('KC-KC scale', 'E8, breadth exponent 3', 'E8_kc_kc_inhibition', r'^breadth3_tau0\.05_kckc(?P<x>[\d.]+)$', {}),
    ('KC-KC scale', 'E19, 40 Hz, breadth exponent 2', 'E19_thr_breadth_kckc', r'^thr40_k2_kckc(?P<x>[\d.]+)$', {}),
    ('KC-KC scale', 'E19, 70 Hz, breadth exponent 2', 'E19_thr_breadth_kckc', r'^thr70_k2_kckc(?P<x>[\d.]+)$', {}),
    ('per-claw threshold (Hz)', 'E16, bouton model', 'E16_claw_saturation', r'^bouton_thr(?P<x>\d+)$', {'bouton_nocap': 0.0}),
    ('per-claw threshold (Hz)', 'E16, claw-dynamics model', 'E16_claw_saturation', r'^claw-dynamics_thr(?P<x>\d+)$',
     {'claw-dynamics_nocap': 0.0}),
    ('ceiling strength', "E69, C2's calibrated fit", 'E69_kc_firing_ceiling', r'^e67cal_r3_a(?P<x>\d+)_t1000$', {'e67cal_base': 0.0}),
    ('ceiling strength', "E69, C2's raw-count fit", 'E69_kc_firing_ceiling', r'^e67raw_r3_a(?P<x>\d+)_t1000$', {'e67raw_base': 0.0}),
    ('ceiling strength', 'E69, the reference model', 'E69_kc_firing_ceiling', r'^ref40_r3_a(?P<x>\d+)_t1000$', {'ref40_base': 0.0}),
    ('scaling exponent', 'E71, C2', 'E71_input_scaled_threshold', r'^c2_a(?P<x>-?[\d.]+)$', {'c2_base': 0.0}),
    ('scaling exponent', 'E71, C2 with the ceiling', 'E71_input_scaled_threshold', r'^c2ceil_a(?P<x>-?[\d.]+)$', {'c2ceil_base': 0.0}),
    ('scaling exponent', 'E71, the reference model', 'E71_input_scaled_threshold', r'^ref40_a(?P<x>-?[\d.]+)$', {'ref40_base': 0.0}),
    ('shared-input APL exponent', 'E15, breadth exponent 1', 'E15_breadth_shared_apl', r'^claw-dynamics_coinc3_sp0\.05_k1_m(?P<x>\d+)$', {}),
    ('shared-input APL exponent', 'E15, breadth exponent 2', 'E15_breadth_shared_apl', r'^claw-dynamics_coinc3_sp0\.05_k2_m(?P<x>\d+)$', {}),
    ('shared-input APL exponent', 'E15, breadth exponent 3', 'E15_breadth_shared_apl', r'^claw-dynamics_coinc3_sp0\.05_k3_m(?P<x>\d+)$', {}),
    ('APL->bouton scale', 'E3, bouton model', 'E3_drive_dependent_gain', r'^bouton_pnapl_x(?P<x>[\d.]+)$', {}),
    ('co-responder oracle factor', 'E14, claw-dynamics', 'E14_oracle_coresponders', r'^claw-dynamics_sp0\.05_f(?P<x>\d+)$',
     {'claw-dynamics_sp0.05_base': 1.0}),
    ('co-responder oracle factor', 'E14, claw-dynamics with coincidence', 'E14_oracle_coresponders',
     r'^claw-dynamics_coinc3_sp0\.05_f(?P<x>\d+)$', {'claw-dynamics_coinc3_sp0.05_base': 1.0}),
]
NOT_IN_ARCHIVE_NOTE = ('local APL units (E2) have no archived variant with both conditions in results/apl_effect/by_variant.csv, so '
    'they are not read here')


# ---------------------------------------------------------------------------
# small helpers
def _rms(x) -> float:
    return float(np.sqrt(np.nanmean(np.square(x))))


def _spearman(a, b) -> float:
    from scipy import stats
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(stats.spearmanr(a[ok], b[ok])[0]) if ok.sum() > 2 else np.nan


def _cos_pairs(X: np.ndarray) -> np.ndarray:
    """Cosine similarity of every odor pair of a (units x odors) array, pairs in `PAIRS` order."""
    n = np.sqrt((X ** 2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return ((X.T @ X) / np.outer(n, n))[IU]


def _nanmean0(arrs: List[np.ndarray]) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.stack(arrs), axis=0) if len(arrs) > 1 else arrs[0]


def class_stats(eff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(means over pairs, shares of the total's variance over pairs) of an effect array (pairs, classes + 1,
    terms), both (classes + 1, terms): share = cov(class-term effect, total effect) / var(total effect)."""
    tot = eff[:, ALL, TOTAL]
    v = float(np.var(tot))
    means = eff.mean(axis=0)
    shares = ((eff - means) * (tot - tot.mean())[:, None, None]).mean(axis=0) / v if v > 0 else np.full(means.shape, np.nan)
    return means, shares


def class_loss(model: np.ndarray, data: np.ndarray) -> float:
    """The class-resolved loss: root mean square over the 5 classes x 120 pairs of model minus data per-class
    effect (each an array (pairs, classes + 1, terms), or a DataFrame from `H.apl_effect_by_class`'s 'effect'
    block; the 'total' term of each class)."""
    m = _as_array(model)
    d = _as_array(data)
    return _rms(m[:, :ALL, TOTAL] - d[:, :ALL, TOTAL])


def pattern_error_shares(model, data) -> pd.Series:
    """Each class's share of the variance over pairs of the pattern error (model minus data total effect, each
    with its mean removed): cov(class error, total error) / var(total error); the shares add to 1."""
    m, d = _as_array(model), _as_array(data)
    ce = m[:, :ALL, TOTAL] - d[:, :ALL, TOTAL]
    err = ce.sum(axis=1)
    v = float(np.var(err))
    sh = ((ce - ce.mean(axis=0)) * (err - err.mean())[:, None]).mean(axis=0) / v if v > 0 else np.full(len(CLASSES), np.nan)
    return pd.Series(sh, index=list(CLASSES))


def _as_array(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        x = x.reindex(PAIR_INDEX)
        return x.to_numpy(dtype=float).reshape(len(x), len(CLASSES_ALL), len(TERMS))
    return np.asarray(x, dtype=float)


def _effect_frame(eff: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(eff.reshape(len(PAIRS), -1), index=PAIR_INDEX,
        columns=pd.MultiIndex.from_product([list(CLASSES_ALL), list(TERMS)], names=['class', 'term']))


# ---------------------------------------------------------------------------
# the data
_MATS: Dict[str, Dict[tuple, np.ndarray]] = {}
_DATA: Dict[float, dict] = {}


def data_matrices() -> Dict[str, Dict[tuple, np.ndarray]]:
    """{genotype: {recording: KCs x odors trial means (APL16 order)}} of Yang's APL16 recordings (cached)."""
    if not _MATS:
        for g in GENOTYPES:
            _MATS[g] = {rec: H.abbrev_columns(m)[ODORS].to_numpy(dtype=float) for rec, m in H.yang_recording_matrices(g).items()}
    return _MATS


def data_class_terms(thresh: float = H.KC_RESPONSE_Z) -> Dict[str, object]:
    """Per genotype, per recording, `H.class_cosine_terms_array` at the criterion trial mean > `thresh` (cached):
    {'arrays': {genotype: (recordings, pairs, classes + 1, terms)}, 'shares': {genotype: (recordings, classes)},
    'band': {genotype: (recordings,) share of entries in (0.5, 1.5] z}, 'recordings': {genotype: keys}}."""
    thresh = float(thresh)
    if thresh not in _DATA:
        out = dict(arrays={}, shares={}, band={}, recordings={})
        for g, mats in data_matrices().items():
            arrs, sh, band = [], [], []
            for X in mats.values():
                R = X > thresh
                arrs.append(H.class_cosine_terms_array(X, R))
                sh.append(H.opk_class_shares(R))
                band.append(float(np.mean((X > BAND[0]) & (X <= BAND[1]))))
            out['arrays'][g], out['shares'][g], out['band'][g] = np.stack(arrs), np.stack(sh), np.array(band)
            out['recordings'][g] = list(mats)
        _DATA[thresh] = out
    return _DATA[thresh]


def data_effect_array(thresh: float = H.KC_RESPONSE_Z) -> np.ndarray:
    """The data's per-pair APL effect by class, (pairs, classes + 1, terms): the APL output blocked (TNT)
    recordings' mean class contributions minus the control flies'."""
    d = data_class_terms(thresh)['arrays']
    return d['TNT'].mean(axis=0) - d['TNT3c'].mean(axis=0)


def data_class_effect(thresh: float = H.KC_RESPONSE_Z) -> pd.DataFrame:
    """The data's per-pair APL effect by class as a DataFrame (pairs x (class, term)), each genotype's class
    contributions averaged over its recordings (`H.apl_effect_by_class`'s 'effect' block)."""
    return _effect_frame(data_effect_array(thresh))


def class_summary(effect) -> pd.DataFrame:
    """Per class (and 'all') and term: the mean effect over pairs ('mean') and its share of the total's variance
    over pairs ('var_share')."""
    means, shares = class_stats(_as_array(effect))
    idx = pd.MultiIndex.from_product([list(CLASSES_ALL), list(TERMS)], names=['class', 'term'])
    return pd.DataFrame({'mean': means.ravel(), 'var_share': shares.ravel()}, index=idx)


def _with_combined(a: np.ndarray, axis: int) -> np.ndarray:
    """Appends the class '1_3' (classes '1' and '2_3' added) along the class axis, after 'all'."""
    i1, i2 = CLASSES.index('1'), CLASSES.index('2_3')
    add = np.take(a, [i1], axis=axis) + np.take(a, [i2], axis=axis)
    return np.concatenate([a, add], axis=axis)


CLASSES_OUT = CLASSES_ALL + ('1_3',)


def data_tables(n_boot: int = 2000, seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Part A's data side: data_by_class (per criterion, quantity, class and term: observed, the mean and 90 %
    interval over `n_boot` resamples of recordings drawn with replacement within each genotype, one generator
    seeded `seed` per criterion), data_by_recording, data_pairs_by_class, floor (the class-resolved loss and the
    APL-effect loss of a resampled data set against the measured one) and yardstick (each control recording
    with each APL output blocked (TNT) recording against the mean of the other recordings, at 1 z)."""
    rows, floor_rows, rec_rows, pair_rows = [], [], [], []
    for t in CRITERIA_Z:
        d = data_class_terms(t)
        A, B = d['arrays']['TNT3c'], d['arrays']['TNT']
        sA, sB = d['shares']['TNT3c'], d['shares']['TNT']
        on, off = A.mean(axis=0), B.mean(axis=0)
        eff = off - on
        means, vsh = class_stats(eff)
        obs = dict(effect=means, var_share=vsh, level_on=on.mean(axis=0), level_off=off.mean(axis=0))
        obs_sh = dict(kc_share_on=sA.mean(axis=0), kc_share_off=sB.mean(axis=0))
        boot = {k: [] for k in obs}
        boot_sh = {k: [] for k in obs_sh}
        cl, tl = [], []
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            ia, ib = rng.integers(0, len(A), len(A)), rng.integers(0, len(B), len(B))
            on_b, off_b = A[ia].mean(axis=0), B[ib].mean(axis=0)
            e_b = off_b - on_b
            m_b, v_b = class_stats(e_b)
            for k, v in (('effect', m_b), ('var_share', v_b), ('level_on', on_b.mean(axis=0)), ('level_off', off_b.mean(axis=0))):
                boot[k].append(v)
            boot_sh['kc_share_on'].append(sA[ia].mean(axis=0))
            boot_sh['kc_share_off'].append(sB[ib].mean(axis=0))
            cl.append(class_loss(e_b, eff))
            tl.append(_rms(e_b[:, ALL, TOTAL] - eff[:, ALL, TOTAL]))
        for k, o in obs.items():
            o2 = _with_combined(o, 0)
            b2 = _with_combined(np.stack(boot[k]), 1)
            for c, cls in enumerate(CLASSES_OUT):
                for ti, term in enumerate(TERMS):
                    if k == 'var_share' and cls == 'all' and term == 'total':
                        continue
                    x = b2[:, c, ti]
                    rows.append(dict(criterion=t, quantity=k, **{'class': cls}, term=term, observed=float(o2[c, ti]),
                        boot_mean=float(np.nanmean(x)), lo=float(np.nanpercentile(x, 5)), hi=float(np.nanpercentile(x, 95))))
        for k, o in obs_sh.items():
            o2 = _with_combined(o, 0)
            b2 = _with_combined(np.stack(boot_sh[k]), 1)
            for c, cls in enumerate([c for c in CLASSES_OUT if c != 'all']):
                x = b2[:, c]
                rows.append(dict(criterion=t, quantity=k, **{'class': cls}, term='', observed=float(o2[c]),
                    boot_mean=float(np.mean(x)), lo=float(np.percentile(x, 5)), hi=float(np.percentile(x, 95))))
        band_on, band_off = d['band']['TNT3c'], d['band']['TNT']
        rows.append(dict(criterion=t, quantity='band_05_15_on', **{'class': 'all'}, term='', observed=float(band_on.mean())))
        rows.append(dict(criterion=t, quantity='band_05_15_off', **{'class': 'all'}, term='', observed=float(band_off.mean())))
        cl, tl = np.array(cl), np.array(tl)
        floor_rows.append(dict(criterion=t, n_boot=n_boot, class_rmse_floor=float(cl.mean()), class_rmse_floor_median=float(np.median(cl)),
            class_rmse_floor_lo=float(np.percentile(cl, 5)), class_rmse_floor_hi=float(np.percentile(cl, 95)),
            apl_rmse_floor=float(tl.mean()), apl_rmse_floor_median=float(np.median(tl))))
        for g, arr, sh in (('TNT3c', A, sA), ('TNT', B, sB)):
            for r, rec in enumerate(d['recordings'][g]):
                row = dict(criterion=t, genotype=g, recording='/'.join(map(str, rec)))
                row.update({f'kc_share_{c}': float(sh[r, i]) for i, c in enumerate(CLASSES)})
                m = arr[r].mean(axis=0)
                row.update({f'level_{c}_{term}': float(m[i, ti]) for i, c in enumerate(CLASSES_ALL) for ti, term in enumerate(TERMS)})
                row['band_05_15'] = float(d['band'][g][r])
                rec_rows.append(row)
        for p, (a, b) in enumerate(PAIRS):
            for c, cls in enumerate(CLASSES_ALL):
                pair_rows.append(dict(criterion=t, odor_a=a, odor_b=b, **{'class': cls}, effect=float(eff[p, c, TOTAL]),
                    effect_both=float(eff[p, c, 0]), effect_one=float(eff[p, c, ONE]), effect_neither=float(eff[p, c, NEITHER]),
                    level_on=float(on[p, c, TOTAL]), level_off=float(off[p, c, TOTAL])))
    return dict(data_by_class=pd.DataFrame(rows), floor=pd.DataFrame(floor_rows), data_by_recording=pd.DataFrame(rec_rows),
        data_pairs_by_class=pd.DataFrame(pair_rows), yardstick=yardstick())


def yardstick() -> pd.DataFrame:
    """Each (control recording, APL output blocked (TNT) recording) pair's per-class effect at 1 z against the
    mean of the other recordings (the other APL-blocked recordings' class contributions minus the other control
    recordings'): the class-resolved loss and the APL-effect loss with its mean and pattern parts (as E70's A5)."""
    d = data_class_terms(H.KC_RESPONSE_Z)
    A, B = d['arrays']['TNT3c'], d['arrays']['TNT']
    rows = []
    for ia, ra in enumerate(d['recordings']['TNT3c']):
        for ib, rb in enumerate(d['recordings']['TNT']):
            e1 = B[ib] - A[ia]
            ref = np.delete(B, ib, axis=0).mean(axis=0) - np.delete(A, ia, axis=0).mean(axis=0)
            err = e1[:, ALL, TOTAL] - ref[:, ALL, TOTAL]
            rows.append(dict(control_recording='/'.join(map(str, ra)), blocked_recording='/'.join(map(str, rb)),
                class_rmse=class_loss(e1, ref), apl_rmse=_rms(err), apl_mean=float(e1[:, ALL, TOTAL].mean()),
                apl_pattern=_rms(err - err.mean()), eff_4_7=float(e1[:, CLASSES.index('4_7'), TOTAL].mean()),
                eff_8plus=float(e1[:, CLASSES.index('8plus'), TOTAL].mean())))
    return pd.DataFrame(rows)


def data_reference() -> Dict[str, float]:
    """The data's values of the per-variant statistics, at 1 z: per-class means, the sub-threshold terms' part
    of the mean, the graded minus thresholded mean effect, the 0.5-1.5 z band in each condition and its growth,
    the class shares."""
    d = data_class_terms(H.KC_RESPONSE_Z)
    eff = data_effect_array()
    out = _variant_stats(dict(on=d['arrays']['TNT3c'].mean(axis=0), off=d['arrays']['TNT'].mean(axis=0), effect=eff,
        share_on=d['shares']['TNT3c'].mean(axis=0), share_off=d['shares']['TNT'].mean(axis=0)), eff)
    thr = H.yang_apl_effect(values='thresholded')
    out['mean_thresholded'] = float(thr.mean())
    out['sub_thresholded'] = out['apl_mean'] - out['mean_thresholded']
    out['band_on'], out['band_off'] = float(d['band']['TNT3c'].mean()), float(d['band']['TNT'].mean())
    out['band_growth'] = out['band_off'] - out['band_on']
    for c in ('on', 'off'):
        out[f'rate_{c}'] = float(np.mean([(X > H.KC_RESPONSE_Z).mean() for X in data_matrices()['TNT3c' if c == 'on' else 'TNT'].values()]))
    return out


def registry_entries(n_boot: int = 2000, seed: int = 0) -> List[Tuple[str, float, str, Optional[Tuple[float, float]]]]:
    """The data-target registry's entries of this experiment (data_targets.py): (name, value, definition, 90 %
    interval or None), from `data_tables(n_boot, seed)`."""
    t = data_tables(n_boot, seed)
    dbc = t['data_by_class']
    fl = t['floor'].set_index('criterion')
    base = ('the per-pair APL effect on odor-odor distances (control flies\' cosine distance minus the distance with APL output '
        'blocked (TNT), each condition\'s class contributions to the cosine similarity averaged over its recordings) split by the '
        'number of the 16 odors a KC responds to in that condition (trial mean > {z} z): ')
    out = []

    def get(crit, q, cls, term='total'):
        r = dbc[(dbc.criterion == crit) & (dbc.quantity == q) & (dbc['class'] == cls) & (dbc.term == term)].iloc[0]
        return float(r.observed), (float(r.lo), float(r.hi))

    for cls in ('0', '1', '2_3', '1_3', '4_7', '8plus'):
        v, ci = get(1.0, 'effect', cls)
        out.append((f'apl_class_effect_{cls}', v, base.format(z=1) + f'the mean over the 120 pairs of the effect carried by KCs '
            f'responding to {CLASS_LABEL[cls]} odors; interval over {n_boot} resamples of recordings', ci))
        v, ci = get(1.0, 'var_share', cls)
        out.append((f'apl_class_var_share_{cls}', v, base.format(z=1) + f'the share of the per-pair effect\'s variance over pairs '
            f'carried by KCs responding to {CLASS_LABEL[cls]} odors (cov(class effect, effect) / var(effect))', ci))
        v, ci = get(1.0, 'effect', cls, 'one')
        out.append((f'apl_class_one_{cls}', v, base.format(z=1) + f'the part of the mean effect of KCs responding to {CLASS_LABEL[cls]} '
            'odors in products of a responder with a sub-threshold entry', ci))
    for z in (0.75, 1.25):
        for cls in ('4_7', '8plus'):
            v, ci = get(z, 'effect', cls)
            out.append((f'apl_class_effect_{cls}_z{z:g}', v, base.format(z=z) + f'the mean effect carried by KCs responding to '
                f'{CLASS_LABEL[cls]} odors, at the criterion {z} z', ci))
    f1 = fl.loc[1.0]
    out.append(('apl_class_rmse_floor', float(f1.class_rmse_floor), 'floor of the class-resolved loss (root mean square over the 5 '
        'odors-per-KC classes x 120 pairs of the per-class APL effect, model minus data, classes at 1 z): the mean over '
        f'{n_boot} resampled data sets (recordings drawn with replacement within each genotype) of its value against the '
        'measured one', (float(f1.class_rmse_floor_lo), float(f1.class_rmse_floor_hi))))
    y = t['yardstick']
    out.append(('apl_class_rmse_yardstick', float(y.class_rmse.median()), 'the class-resolved loss of one control recording paired '
        'with one APL output blocked (TNT) recording against the mean of the other recordings, median over the 88 pairs',
        (float(y.class_rmse.quantile(0.05)), float(y.class_rmse.quantile(0.95)))))
    return out


# ---------------------------------------------------------------------------
# the models
_NOISE: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}


def pseudo_noise(n_units: int, n_odors: int = 16, n_recordings: int = N_RECORDINGS, n_trials: int = N_TRIALS,
    seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """(APL-on, APL-off) trial means of standard normal draws, each (n_recordings, n_units, n_odors): the draws
    (n_recordings, n_trials, n_units, n_odors) from one generator, numpy default_rng(seed), the APL-on draws first;
    cached per shape (the same draws for every variant of a size and every evaluation of a fit). A reading's
    measurement noise on a trial mean is the unit's trial sd times these."""
    key = (int(n_units), int(n_odors), int(n_recordings), int(n_trials), int(seed))
    if key not in _NOISE:
        rng = np.random.default_rng(int(seed))
        shape = (int(n_recordings), int(n_trials), int(n_units), int(n_odors))
        _NOISE[key] = tuple(rng.standard_normal(shape).mean(axis=1) for _ in range(2))
        if len(_NOISE) > 8:
            _NOISE.pop(next(iter(_NOISE)))
    return _NOISE[key]


def read_cal(on: np.ndarray, off: np.ndarray, n_spikes: float = 2.0, K: Optional[float] = None) -> Dict[str, object]:
    """A model under E65's calibrated readout n / (n + K) with an n-spike criterion (KCs x odors spike counts in
    APL16 order): {'on', 'off', 'effect'} class arrays, the class shares and response rates, K. K defaults to the
    calibration (`H.calibrate_readout`, the control flies' responders-only participation ratio)."""
    on, off = np.asarray(on, dtype=float), np.asarray(off, dtype=float)
    if K is None:
        K = H.calibrate_readout(pd.DataFrame(on, columns=ODORS), 'saturating', H.data_target('ctrl_participation_ratio_resp'),
            stat='participation_ratio', responders_min=2.0)
    out = dict(K=float(K))
    for c, n in (('on', on), ('off', off)):
        R = n >= n_spikes
        out[c] = H.class_cosine_terms_array(n / (n + K), R)
        out[f'share_{c}'] = H.opk_class_shares(R)
        out[f'rate_{c}'] = float(R.mean())
    out['effect'] = out['off'] - out['on']
    return out


def read_zcell(on, off, cal: dict, noise: bool = True) -> Dict[str, object]:
    """A model under E73's reading at the calibration `cal` (K, noise, noise_ratio, q, drift; amp, default
    saturating): each condition's `H.session_zscore` trial means, on 10 pseudo-recordings of 3 trials through the
    measurement noise (`pseudo_noise`, seed 43) or noise-free, the data's criterion (trial mean > 1 z) on each,
    class contributions averaged over the pseudo-recordings. `on` / `off`: (KCs x odors) counts or (recordings,
    trials, KCs, odors) jittered trials, odors in APL16 order. Returns {'on', 'off', 'effect'} class arrays, the
    class shares, the response rates, the responses per KC (`n_resp_on` / `n_resp_off`, mean over the
    pseudo-recordings), the 0.5-1.5 z band shares and the graded and thresholded mean effects."""
    kw = dict(q=float(cal['q']), noise=float(cal['noise']), amp=cal.get('amp', 'saturating'),
        K=None if cal.get('K') is None or not np.isfinite(cal.get('K')) else float(cal['K']),
        noise_ratio=float(cal['noise_ratio']), drift=float(cal['drift']))
    out = {}
    thr_sim = {}
    for c, counts in (('on', on), ('off', off)):
        counts = np.asarray(counts, dtype=float)
        z = H.session_zscore(counts, **kw)
        C, sd = z['center'], z['trial_sd']
        if noise:
            n_rec = N_RECORDINGS if counts.ndim == 2 else C.shape[0]
            n_tr = N_TRIALS if counts.ndim == 2 else counts.shape[1]
            xi = pseudo_noise(C.shape[1], C.shape[2], n_rec, n_tr)[0 if c == 'on' else 1]
            tm = C + sd[:, :, None] * xi
        else:
            tm = C
        R = tm > H.KC_RESPONSE_Z
        # responses per KC (odors above the criterion), mean over the (pseudo-)recordings
        out[f'n_resp_{c}'] = R.sum(axis=2).mean(axis=0)
        out[c] = _nanmean0([H.class_cosine_terms_array(tm[r], R[r]) for r in range(len(tm))])
        out[f'share_{c}'] = np.mean([H.opk_class_shares(R[r]) for r in range(len(tm))], axis=0)
        out[f'rate_{c}'] = float(R.mean())
        out[f'band_{c}'] = float(np.mean((tm > BAND[0]) & (tm <= BAND[1])))
        thr_sim[c] = _nanmean0([_cos_pairs(np.where(R[r], tm[r], 0.0)) for r in range(len(tm))])
    out['effect'] = out['off'] - out['on']
    out['mean_thresholded'] = float(np.nanmean(thr_sim['off'] - thr_sim['on']))
    return out


def _variant_stats(blk: dict, data: np.ndarray) -> Dict[str, float]:
    """The per-variant statistics of a reading's class arrays against the data's (pairs, classes + 1, terms)."""
    eff = blk['effect']
    out = {}
    means, vsh = class_stats(eff)
    for c, cls in enumerate(CLASSES_ALL):
        out[f'eff_{cls}'] = float(means[c, TOTAL])
        for ti, term in enumerate(TERMS[:3]):
            out[f'eff_{cls}_{term}'] = float(means[c, ti])
        if cls != 'all':
            out[f'var_share_{cls}'] = float(vsh[c, TOTAL])
        out[f'level_on_{cls}'] = float(np.nanmean(blk['on'][:, c, TOTAL]))
        out[f'level_off_{cls}'] = float(np.nanmean(blk['off'][:, c, TOTAL]))
    out['eff_1_3'] = out['eff_1'] + out['eff_2_3']
    i1, i2 = CLASSES.index('1'), CLASSES.index('2_3')
    for c in ('on', 'off'):
        out[f'level_{c}_1_3'] = float(np.nanmean(blk[c][:, i1, TOTAL] + blk[c][:, i2, TOTAL]))
    out['class_rmse'] = class_loss(eff, data)
    ce = eff[:, :ALL, TOTAL] - data[:, :ALL, TOTAL]
    out['class_rmse_centered'] = _rms(ce - np.nanmean(ce, axis=0))
    for cls, v in pattern_error_shares(eff, data).items():
        out[f'perr_share_{cls}'] = float(v)
    for cls in CLASSES:
        out[f'class_err_{cls}'] = float(np.nanmean(ce[:, CLASSES.index(cls)]))
    m, d = eff[:, ALL, TOTAL], data[:, ALL, TOTAL]
    err = m - d
    out.update(apl_rmse=_rms(err), apl_mean=float(np.nanmean(m)), apl_mean_error=float(np.nanmean(err)),
        apl_pattern=_rms(err - np.nanmean(err)), apl_rho=_spearman(m, d))
    out['sub_terms'] = float(means[ALL, ONE] + means[ALL, NEITHER])
    for c in ('on', 'off'):
        if f'share_{c}' in blk:
            out.update({f'kc_share_{c}_{cls}': float(v) for cls, v in zip(CLASSES, blk[f'share_{c}'])})
        for k in ('rate', 'band'):
            if f'{k}_{c}' in blk:
                out[f'{k}_{c}'] = blk[f'{k}_{c}']
    if 'band_on' in blk:
        out['band_growth'] = blk['band_off'] - blk['band_on']
    if 'mean_thresholded' in blk:
        out['mean_thresholded'] = blk['mean_thresholded']
        out['sub_thresholded'] = out['apl_mean'] - blk['mean_thresholded']
    if 'K' in blk:
        out['K'] = blk['K']
    return out


_WORKER: dict = {}


def _init_worker(data: np.ndarray) -> None:
    import warnings as _w
    _w.simplefilter('ignore', RuntimeWarning)
    _WORKER['data'] = data


def _load_counts(folder: str, tag: str, cond: str) -> np.ndarray:
    df = pd.read_parquet(ARCHIVE / folder / f'{tag}_{cond}_spike_counts.parquet')
    return H.abbrev_columns(df.astype(float))[ODORS].to_numpy(dtype=float)


def _calibrate_zcell(on: np.ndarray) -> dict:
    """E73's calibration of the reading on APL-on counts (for variants E73's table does not have)."""
    import E73_per_kc_normalization as E73
    t = E73.refs()['targets']
    q = E73.q_values()[0]
    drift = E73.refs()['drift']
    cal = H.calibrate_session_zscore(on, q=q, amp='saturating', target_rate=t['rate'], target_amp=t['amp'],
        target_silent_sd=t['silent_sd'], drift=drift)
    return dict(K=cal['K'], noise=cal['noise'], noise_ratio=cal['noise_ratio'], q=q, drift=drift)


def score_variant(job: tuple) -> Tuple[List[dict], Optional[dict]]:
    """One archived variant under every reading. job: (folder, tag, K of the calibrated readout or None, E73's
    calibration of the reading or None, the trial set's glob pattern or None, its calibration, keep the class
    arrays). Returns (rows, {reading: class arrays} or None); an exception is returned as one row with `error`."""
    folder, tag, K, zcal, trial_pattern, tcal, keep = job
    data = _WORKER.get('data')
    if data is None:
        data = data_effect_array()
    base = dict(folder=folder, tag=tag)
    rows, kept = [], {}
    try:
        on, off = _load_counts(folder, tag, 'on'), _load_counts(folder, tag, 'acute_off')
        base['n_kcs'] = on.shape[0]
        for n in SPIKE_CRITERIA:
            blk = read_cal(on, off, n, K)
            K = blk['K']
            rows.append(dict(base, reading=f'cal{n}', **_variant_stats(blk, data)))
            if keep and n == 2:
                kept['cal2'] = {k: blk[k] for k in ('on', 'off', 'share_on', 'share_off')}
        zsrc = 'E73 by_variant'
        if zcal is None:
            zcal, zsrc = _calibrate_zcell(on), 'calibrated here (as E73)'
        for rd, noise in (('zcell', True), ('zcell_free', False)):
            blk = read_zcell(on, off, zcal, noise)
            rows.append(dict(base, reading=rd, zcal_source=zsrc, z_K=zcal['K'], z_noise=zcal['noise'],
                z_noise_ratio=zcal['noise_ratio'], z_q=zcal['q'], z_drift=zcal['drift'], **_variant_stats(blk, data)))
            if keep:
                kept[rd] = {k: blk[k] for k in ('on', 'off', 'share_on', 'share_off')}
        if trial_pattern is not None and tcal is not None:
            import E73_per_kc_normalization as E73
            t_on, t_off = E73.load_trials(trial_pattern)
            blk = read_zcell(t_on, t_off, tcal, True)
            rows.append(dict(base, reading='zcell_trials', zcal_source='E73 by_variant (trials)', z_K=tcal['K'], z_noise=tcal['noise'],
                z_noise_ratio=tcal['noise_ratio'], n_trial_recordings=t_on.shape[0], **_variant_stats(blk, data)))
            if keep:
                kept['zcell_trials'] = {k: blk[k] for k in ('on', 'off', 'share_on', 'share_off')}
        return rows, (kept if keep else None)
    except Exception as e:
        import traceback
        return [dict(base, error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())], None


# ---------------------------------------------------------------------------
# part C: the class term of the joint fit (E10_joint_fit.py --apl-class-term)
def class_term_stats(s_on: pd.DataFrame, s_off: pd.DataFrame, zcell: dict, cal: dict) -> Dict[str, float]:
    """The hook for E10_joint_fit.py's `--apl-class-term`: the model's per-class effects under E73's reading at
    the calibration `cal` (the one the evaluation's `zcell_terms` used: noise, K, noise_ratio) with the reading's
    settings `zcell` (q, amp, drift), on the fixed pseudo-recordings (`pseudo_noise`, seed 43: drawn once per
    KC count, the same at every evaluation), the data's criterion (trial mean > 1 z), against the data's per-class
    effects at 1 z.

    Returns apl_class_mse (the mean over the 5 classes x 120 pairs of the squared difference, the term before
    division by the floor's square), apl_class_rmse (its root, the class-resolved loss), apl_class_floor (the
    registry's apl_class_rmse_floor), apl_class_eff_<class> (the model's mean per-class effects, classes and
    'all'), apl_class_err_<class> (the mean per-class error), apl_class_noisy_rmse / _mean / _pattern (the
    APL-effect loss of the pseudo-recordings' total effect and its parts) and apl_class_rate_on / _off (the
    pseudo-recordings' response rates)."""
    on, off = (H.abbrev_columns(s.astype(float))[ODORS].to_numpy(dtype=float) for s in (s_on, s_off))
    zc = dict(q=zcell['q'], amp=zcell.get('amp', 'saturating'), drift=zcell['drift'], K=cal.get('K'), noise=cal['noise'],
        noise_ratio=cal['noise_ratio'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        blk = read_zcell(on, off, zc, noise=True)
    data = data_effect_array()
    d = blk['effect'][:, :ALL, TOTAL] - data[:, :ALL, TOTAL]
    mse = float(np.mean(d ** 2))
    out = dict(apl_class_mse=mse, apl_class_rmse=float(np.sqrt(mse)), apl_class_floor=float(H.data_target('apl_class_rmse_floor')))
    means = blk['effect'].mean(axis=0)
    out.update({f'apl_class_eff_{c}': float(means[i, TOTAL]) for i, c in enumerate(CLASSES_ALL)})
    out.update({f'apl_class_err_{c}': float(d[:, i].mean()) for i, c in enumerate(CLASSES)})
    err = blk['effect'][:, ALL, TOTAL] - data[:, ALL, TOTAL]
    out.update(apl_class_noisy_rmse=_rms(err), apl_class_noisy_mean=float(blk['effect'][:, ALL, TOTAL].mean()),
        apl_class_noisy_pattern=_rms(err - err.mean()), apl_class_rate_on=blk['rate_on'], apl_class_rate_off=blk['rate_off'])
    return out


# ---------------------------------------------------------------------------
# part A: scoring every archived variant, the tables and figures
def _git(*args) -> str:
    return subprocess.run(['git', *args], cwd=H.HERE, capture_output=True, text=True).stdout.strip()


def _key_archived() -> List[Tuple[str, str, str, str]]:
    return [(s, lab, f, t) for s, lab, f, t in KEY_MODELS
        if (ARCHIVE / f / f'{t}_on_spike_counts.parquet').exists() and (ARCHIVE / f / f'{t}_acute_off_spike_counts.parquet').exists()]


def variant_jobs(only_key: bool = False) -> List[tuple]:
    """Every variant of results/apl_effect/by_variant.csv with both conditions archived (E73's `variant_list`),
    plus the archived key models and E72's wirings; each with E73's calibrations where E73's table has them (the
    reading is calibrated here, as E73 does, for the others)."""
    import E73_per_kc_normalization as E73
    keep = {(f, t) for _, _, f, t in _key_archived()} | {(f, t) for _, _, mem in WIRING_GROUPS for _, f, t in mem}
    have = E73.variant_list(False)
    have += [(f, t) for f, t in sorted(keep) if (f, t) not in set(have)]
    if only_key:
        have = [h for h in have if h in keep]
    have = [h for h in have if (ARCHIVE / h[0] / f'{h[1]}_on_spike_counts.parquet').exists()]
    byv = pd.read_csv(RESULTS / 'E73_per_kc_normalization' / 'by_variant.csv', low_memory=False)
    byv['tag'] = byv['tag'].astype(str)
    g = byv[byv.reading == 'graded'] if 'reading' in byv else byv
    Kmap = {(r.folder, r.tag): float(r.K) for r in byv[byv.reading_def == 'present_cal'].itertuples()}
    zmap = {(r.folder, r.tag): dict(K=float(r.K), noise=float(r.noise), noise_ratio=float(r.noise_ratio), q=float(r.q), drift=float(r.drift))
        for r in g[g.reading_def == 'zcell_saturating'].itertuples()}
    tmap = {(r.folder, r.tag): dict(K=float(r.K), noise=float(r.noise), noise_ratio=float(r.noise_ratio), q=float(r.q), drift=float(r.drift))
        for r in g[g.reading_def == 'zcell_saturating_trials'].itertuples()}
    return [(f, t, Kmap.get((f, t)), zmap.get((f, t)), E73.TRIAL_SETS.get((f, t)), tmap.get((f, t)), (f, t) in keep) for f, t in have]


def merge_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the realism bands: under the two-spike criterion (results/apl_effect/by_variant.csv: flag_spikes,
    rate_on_spk, rate_off_spk, rate_ratio_spk, rate_rho_on_spk, opk_on_tv_spk, opk_off_tv_spk) and under the data's criterion
    through the reading (E73's zcell_saturating graded row: flag_reading, z_rate_on, z_rate_ratio, z_rate_rho_on,
    z_opk_on_tv, z_opk_off_tv) and E73's losses under its readings (e73_cal_rmse, e73_zcell_rmse, e73_zcell_pattern,
    e73_zcell_mean, in expectation over the noise)."""
    bv = pd.read_csv(RESULTS / 'apl_effect' / 'by_variant.csv').drop_duplicates(['folder', 'tag'])
    bv['tag'] = bv['tag'].astype(str)
    bv = bv[['folder', 'tag', 'exp', 'realism_flag', 'rate_on', 'rate_off', 'spearman_rate_on_vs_data', 'opk_on_tv', 'opk_off_tv']].rename(
        columns={'realism_flag': 'flag_spikes', 'rate_on': 'rate_on_spk', 'rate_off': 'rate_off_spk',
        'spearman_rate_on_vs_data': 'rate_rho_on_spk', 'opk_on_tv': 'opk_on_tv_spk', 'opk_off_tv': 'opk_off_tv_spk'})
    byv = pd.read_csv(RESULTS / 'E73_per_kc_normalization' / 'by_variant.csv', low_memory=False)
    byv['tag'] = byv['tag'].astype(str)
    z = byv[(byv.reading_def == 'zcell_saturating') & (byv.reading == 'graded')].drop_duplicates(['folder', 'tag'])
    z = z[['folder', 'tag', 'realism_flag', 'rate_on', 'rate_ratio', 'rate_rho_on', 'opk_on_tv', 'opk_off_tv', 'rmse', 'rmse_centered',
        'mean']].rename(columns={'realism_flag': 'flag_reading', 'rate_on': 'z_rate_on', 'rate_ratio': 'z_rate_ratio',
        'rate_rho_on': 'z_rate_rho_on', 'opk_on_tv': 'z_opk_on_tv', 'opk_off_tv': 'z_opk_off_tv', 'rmse': 'e73_zcell_rmse',
        'rmse_centered': 'e73_zcell_pattern', 'mean': 'e73_zcell_mean'})
    c = byv[byv.reading_def == 'present_cal'].drop_duplicates(['folder', 'tag'])[['folder', 'tag', 'rmse']].rename(columns={'rmse': 'e73_cal_rmse'})
    out = df.merge(bv, on=['folder', 'tag'], how='left').merge(z, on=['folder', 'tag'], how='left').merge(c, on=['folder', 'tag'], how='left')
    for col in ('flag_spikes', 'flag_reading'):
        out[col] = out[col].fillna('')
    out['rate_ratio_spk'] = out['rate_off_spk'] / out['rate_on_spk']
    # (scored = a band was read for the variant: archived responses or E73's reading)
    out['in_bands_spikes'] = out['exp'].notna() & (out['flag_spikes'] == '')
    out['in_bands_reading'] = out['z_rate_on'].notna() & (out['flag_reading'] == '')
    return out


def key_label(folder: str, tag: str) -> Optional[str]:
    for s, _, f, t in KEY_MODELS:
        if (f, t) == (folder, tag):
            return s
    for g, _, mem in WIRING_GROUPS:
        for w, f, t in mem:
            if (f, t) == (folder, tag):
                return f'{g} {w}'
    return None


def wiring_rows(blocks: Dict[tuple, dict], data: np.ndarray) -> pd.DataFrame:
    """E72's wirings as their mean (each condition's class contributions averaged over the wirings), per group of
    `WIRING_GROUPS` and reading; for the groups with fewer wirings also the same wirings' mean on the real wiring's
    partner group (none here: each group is its own)."""
    rows = []
    for g, lab, mem in WIRING_GROUPS:
        for rd in ('cal2', 'zcell', 'zcell_free'):
            have = [(w, blocks[(f, t)][rd]) for w, f, t in mem if (f, t) in blocks and rd in blocks[(f, t)]]
            if not have:
                continue
            bl = [b for _, b in have]
            blk = dict(on=np.mean([b['on'] for b in bl], axis=0), off=np.mean([b['off'] for b in bl], axis=0),
                share_on=np.mean([b['share_on'] for b in bl], axis=0), share_off=np.mean([b['share_off'] for b in bl], axis=0))
            blk['effect'] = blk['off'] - blk['on']
            rows.append(dict(group=g, model=f'{lab}: mean of {len(have)} wirings', wirings=','.join(w for w, _ in have), reading=rd,
                n_wirings=len(have), **_variant_stats(blk, data)))
    return pd.DataFrame(rows)


def rank_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Across variants, per reading: Spearman of the class-resolved loss with the APL-effect loss and with its
    pattern part (all variants, and those inside the bands)."""
    rows = []
    for rd, g in df.groupby('reading'):
        for subset, m in (('all variants', np.ones(len(g), dtype=bool)), ('inside the two-spike bands', g.in_bands_spikes.to_numpy()),
                ('inside both criteria\'s bands', (g.in_bands_spikes & g.in_bands_reading).to_numpy())):
            s = g[m]
            rows.append(dict(reading=rd, subset=subset, n=int(len(s)), rho_class_vs_loss=_spearman(s.class_rmse, s.apl_rmse),
                rho_class_vs_pattern=_spearman(s.class_rmse, s.apl_pattern), rho_4_7_err_vs_pattern=_spearman(s.class_err_4_7.abs(), s.apl_pattern)))
    return pd.DataFrame(rows)


def series_table(df: pd.DataFrame) -> pd.DataFrame:
    """The archive's sweeps (`SERIES`) as series on the per-class effects, per reading."""
    import re
    rows = []
    for handle, label, folder, pat, base in SERIES:
        g = df[df.folder == folder]
        for r in g.itertuples():
            m = re.match(pat, str(r.tag))
            x = float(m.group('x')) if m else base.get(str(r.tag))
            if x is None:
                continue
            rows.append(dict(handle=handle, series=label, folder=folder, tag=r.tag, x=x, reading=r.reading, eff_0=r.eff_0, eff_1_3=r.eff_1_3,
                eff_4_7=r.eff_4_7, eff_8plus=r.eff_8plus, eff_all=r.eff_all, class_rmse=r.class_rmse, apl_rmse=r.apl_rmse,
                apl_pattern=r.apl_pattern, in_bands_spikes=r.in_bands_spikes, in_bands_reading=r.in_bands_reading,
                flag_spikes=r.flag_spikes, flag_reading=r.flag_reading))
    return pd.DataFrame(rows).sort_values(['handle', 'series', 'reading', 'x']) if rows else pd.DataFrame()


def part_c_condition(df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Part C's condition, per reading (cal2, zcell): the data's 4-7 class effect has a 90 % interval that excludes
    every best fit's value, and the floor of the class-resolved loss is at most half the best fits' lowest
    class-resolved loss. Only the best fits inside the realism bands under both criteria (the two-spike one and the
    data's through the per-KC reading) count (planning decision of 2026-11-10 15:16: fit S meets the 4-7 class only
    outside three two-spike bands); the others are listed under `outside_bands`."""
    d = tables['data_by_class']
    r = d[(d.criterion == 1.0) & (d.quantity == 'effect') & (d['class'] == '4_7') & (d.term == 'total')].iloc[0]
    floor = float(tables['floor'].set_index('criterion').loc[1.0, 'class_rmse_floor'])
    out = dict(data_4_7=float(r.observed), data_4_7_lo=float(r.lo), data_4_7_hi=float(r.hi), class_rmse_floor=floor, readings={})
    for rd in ('cal2', 'zcell'):
        fits, outside = [], []
        for s, _, f, t in KEY_MODELS:
            if s not in BEST_FITS:
                continue
            row = df[(df.folder == f) & (df.tag == t) & (df.reading == rd)]
            if len(row):
                row = row.iloc[0]
                rec = dict(model=s, eff_4_7=float(row.eff_4_7), excluded=bool(row.eff_4_7 < r.lo or row.eff_4_7 > r.hi),
                    class_rmse=float(row.class_rmse), apl_rmse=float(row.apl_rmse))
                if bool(row.get('in_bands_spikes', True)) and bool(row.get('in_bands_reading', True)):
                    fits.append(rec)
                else:
                    outside.append(rec)
        best = min(f['class_rmse'] for f in fits) if fits else np.nan
        out['readings'][rd] = dict(fits=fits, outside_bands=outside, all_excluded=all(f['excluded'] for f in fits),
            lowest_class_rmse=best, floor_at_most_half=bool(floor <= 0.5 * best),
            holds=bool(fits and all(f['excluded'] for f in fits) and floor <= 0.5 * best))
    out['holds'] = all(v['holds'] for v in out['readings'].values())
    return out


PC_CLASSES = ('0', '1', '2_3', '1_3', '4_7', '8plus', 'all')
PC_READINGS = ('cal2', 'zcell', 'zcell_free', 'zcell_trials')


def per_condition_table(df: pd.DataFrame, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The per-condition table: each class's similarity level with APL on and with APL off (the mean over the 120 pairs
    of the class's contribution to the pair's cosine similarity, `level_on_<class>` / `level_off_<class>`) and the
    effect (APL off minus on), for the KC data at 1 z (control flies as APL on, APL output blocked (TNT) as APL off;
    observed with the 90 % recording-bootstrap interval of data_by_class) and the archived key models under each
    reading of `PC_READINGS`. Model rows also carry their difference from the data's value and whether it lies outside
    the data's interval ('above' / 'below' / ''). One row per (model, reading, class)."""
    d = tables['data_by_class']
    d = d[(d.criterion == 1.0) & (d.term == 'total')]
    data = {}
    for q in ('level_on', 'level_off', 'effect'):
        s = d[d.quantity == q]
        for cls, o, lo, hi in zip(s['class'].astype(str), s.observed, s.lo, s.hi):
            data[(q, cls)] = (float(o), float(lo), float(hi))
    rows = []
    for cls in PC_CLASSES:
        row = dict(model='KC data', label="KC data at 1 z: control flies (APL on), APL output blocked (TNT) (APL off)",
            folder='', tag='', reading='data', **{'class': cls})
        for q in ('level_on', 'level_off', 'effect'):
            v, lo, hi = data.get((q, cls), (np.nan, np.nan, np.nan))
            row.update({q: v, f'{q}_lo': lo, f'{q}_hi': hi})
        rows.append(row)
    for s, lab, f, t in KEY_MODELS:
        g = df[(df.folder == f) & (df.tag == t)]
        for rd in PC_READINGS:
            r = g[g.reading == rd]
            if not len(r):
                continue
            r = r.iloc[0]
            for cls in PC_CLASSES:
                row = dict(model=s, label=lab, folder=f, tag=t, reading=rd, **{'class': cls})
                for c in ('on', 'off'):
                    col = f'level_{c}_{cls}'
                    if col in r.index and pd.notna(r[col]):
                        v = float(r[col])
                    elif cls == '1_3':
                        v = float(r[f'level_{c}_1'] + r[f'level_{c}_2_3'])
                    else:
                        v = np.nan
                    row[f'level_{c}'] = v
                row['effect'] = row['level_off'] - row['level_on']
                for q in ('level_on', 'level_off', 'effect'):
                    dv, lo, hi = data.get((q, cls), (np.nan, np.nan, np.nan))
                    row[f'{q}_minus_data'] = row[q] - dv
                    row[f'{q}_outside'] = 'above' if row[q] > hi else ('below' if row[q] < lo else '')
                rows.append(row)
    return pd.DataFrame(rows)


def per_condition_md(pc: pd.DataFrame, readings: Sequence[str] = ('cal2', 'zcell')) -> List[str]:
    """comparison_table.md's section of the per-condition table (`per_condition_table`), one table per reading."""
    show = ('0', '1_3', '4_7', '8plus', 'all')
    dat = pc[pc.reading == 'data'].set_index('class')

    def dcell(cls, q):
        r = dat.loc[cls]
        return f'{f3(r[q], ".3f")} ({f3(r[q + "_lo"], ".3f")} to {f3(r[q + "_hi"], ".3f")})'

    def mcell(r, q):
        mark = {'above': ' ↑', 'below': ' ↓'}.get(r[f'{q}_outside'], '')
        return f3(r[q], '.3f') + mark

    L = ['## Per condition: the similarity each class carries with APL on and with APL off', '',
        'Each class\'s level is the mean over the 120 pairs of its contribution to the pair\'s cosine similarity; the per-pair APL '
        'effect of the class is the level with APL off minus the level with APL on. KC data at 1 z, with 90 % intervals over '
        'resamples of recordings: APL on = control flies, APL off = APL output blocked (TNT). A model value above the data\'s '
        'interval is marked ↑, below it ↓. All values in per_condition.csv (also the 1 and 2-to-3 classes, the reading without '
        'the noise and the saved jittered trials).', '']
    for rd in readings:
        g = pc[pc.reading == rd]
        if not len(g):
            continue
        L += [f'### {READING_LABEL[rd]}', '',
            '| model | ' + ' | '.join(f'{CLASS_LABEL[c]} {{span=2}}' for c in show) + ' |',
            '| | ' + ' | '.join('APL on | APL off' for _ in show) + ' |',
            '|---|' + '---:|' * (2 * len(show))]
        L.append('| KC data | ' + ' | '.join(f'{dcell(c, "level_on")} | {dcell(c, "level_off")}' for c in show) + ' |')
        for s in dict.fromkeys(g.model):
            m = g[g.model == s].set_index('class')
            L.append(f'| {s} | ' + ' | '.join(f'{mcell(m.loc[c], "level_on")} | {mcell(m.loc[c], "level_off")}' for c in show) + ' |')
        L.append('')
    return L


# who is released, by claw count (as E71's release_stats): the standard bouton-base wiring's claws per KC, the groups of E71
CLAW_WIRING = ARCHIVE / 'inputs' / 'wPNKC_bouton_base.parquet'
CLAW_GROUPS = (('1-4', 1, 4), ('5-6', 5, 6), ('7-8', 7, 8), ('9+', 9, 10 ** 6))
CLAW_READINGS = ('cal2', 'zcell', 'zcell_free')


def claw_release(n_on: np.ndarray, n_off: np.ndarray, claws: np.ndarray) -> List[dict]:
    """Per claw-count group of `CLAW_GROUPS`: the KCs, their mean claw count, the responses with APL on and off (summed
    over the group's KCs), the group's share of the responses blocking APL adds (its added responses over all KCs'),
    the mean odors per KC and the share of silent KCs in each condition. `n_on` / `n_off`: responses per KC (odors
    above the criterion; a mean over recordings may be fractional), `claws`: claws per KC, all in the same KC order."""
    n_on, n_off, claws = (np.asarray(x, dtype=float) for x in (n_on, n_off, claws))
    added = float(n_off.sum() - n_on.sum())
    rows = []
    for lab, lo, hi in CLAW_GROUPS:
        m = (claws >= lo) & (claws <= hi)
        rows.append(dict(group=lab, n_kcs=int(m.sum()), mean_claws=float(claws[m].mean()) if m.any() else np.nan,
            resp_on=float(n_on[m].sum()), resp_off=float(n_off[m].sum()),
            share_added=float((n_off[m].sum() - n_on[m].sum()) / added) if added else np.nan,
            opk_on=float(n_on[m].mean()) if m.any() else np.nan, opk_off=float(n_off[m].mean()) if m.any() else np.nan,
            silent_on=float((n_on[m] == 0).mean()) if m.any() else np.nan,
            silent_off=float((n_off[m] == 0).mean()) if m.any() else np.nan, added_all=added))
    return rows


def released_by_claws(df: pd.DataFrame) -> pd.DataFrame:
    """Who is released in the archived key models, by claw-count group (`claw_release`, the claws of `CLAW_WIRING`, the
    real wiring all key models are on): under the calibrated readout's two-spike criterion ('cal2', spike count >= 2),
    and under E73's reading with the data's criterion through the measurement noise ('zcell', responses per KC averaged
    over the 10 pseudo-recordings) and without it ('zcell_free'), at the calibration the variant's zcell row of `df`
    records. Models whose KCs are not all in the wiring are skipped. One row per (model, reading, group)."""
    if not CLAW_WIRING.exists():
        return pd.DataFrame()
    W = pd.read_parquet(CLAW_WIRING)
    # (claw rows with any input, per KC: E71's n_claws_per_kc)
    claws = (W != 0).any(axis='columns').groupby(level=H.KC_ID_LEVEL).sum().astype(float)
    rows = []
    for s, lab, f, t in _key_archived():
        frames = {c: pd.read_parquet(ARCHIVE / f / f'{t}_{c}_spike_counts.parquet') for c in ('on', 'acute_off')}
        if H.KC_ID_LEVEL not in (frames['on'].index.names or []):
            continue
        ids = frames['on'].index.get_level_values(H.KC_ID_LEVEL)
        c = claws.reindex(ids).to_numpy()
        if np.isnan(c).any() or not frames['acute_off'].index.get_level_values(H.KC_ID_LEVEL).equals(ids):
            continue
        on, off = (H.abbrev_columns(frames[k].astype(float))[ODORS].to_numpy(dtype=float) for k in ('on', 'acute_off'))
        per = {'cal2': ((on >= 2).sum(axis=1), (off >= 2).sum(axis=1))}
        z = df[(df.folder == f) & (df.tag == t) & (df.reading == 'zcell')]
        if len(z) and pd.notna(z.iloc[0].get('z_noise', np.nan)):
            z = z.iloc[0]
            cal = dict(K=z.z_K, noise=z.z_noise, noise_ratio=z.z_noise_ratio)
            if pd.notna(z.get('z_q', np.nan)):
                cal.update(q=z.z_q, drift=z.z_drift)
            else:
                cal = _calibrate_zcell(on)
            for rd, noise in (('zcell', True), ('zcell_free', False)):
                blk = read_zcell(on, off, cal, noise)
                per[rd] = (blk['n_resp_on'], blk['n_resp_off'])
        for rd, (n_on, n_off) in per.items():
            for g in claw_release(n_on, n_off, c):
                rows.append(dict(model=s, folder=f, tag=t, reading=rd, **g))
    return pd.DataFrame(rows)


def released_md(rel: pd.DataFrame) -> List[str]:
    """comparison_table.md's section of `released_by_claws`: per model and reading, each claw-count group's share of the
    responses blocking APL adds, the added responses per recording and the share of KCs silent with APL off."""
    L = ['## Who is released, by claw count', '',
        'Each claw-count group\'s share of the KC x odor responses that blocking APL adds (APL off minus APL on; the claws per '
        'KC of the standard bouton-base wiring, the groups of E71: 1-4 claws, the third of KCs with the fewest; 9+, the 14 % '
        'with the most); under the reading the responses per KC are averaged over the 10 pseudo-recordings. Values in '
        'released_by_claws.csv (also the noise-free reading, the groups\' sizes, odors per KC and silent shares).', '',
        '| model | reading | 1-4 | 5-6 | 7-8 | 9+ | added responses (per recording) | silent with APL off |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for (s, rd), g in rel[rel.reading.isin(['cal2', 'zcell'])].groupby(['model', 'reading'], sort=False):
        g = g.set_index('group')
        silent = float((g.silent_off * g.n_kcs).sum() / g.n_kcs.sum())
        L.append(f'| {s} | {rd} | ' + ' | '.join(f3(g.loc[k, "share_added"], ".3f") for k, *_ in CLAW_GROUPS)
            + f' | {g.added_all.iloc[0]:.0f} | {silent:.3f} |')
    return L + ['']


def f3(x, spec: str = '+.3f') -> str:
    try:
        return '–' if x is None or not np.isfinite(float(x)) else format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def write_table(out: Path, df: pd.DataFrame, tables: Dict[str, pd.DataFrame], wires: pd.DataFrame, rc: pd.DataFrame,
    ser: pd.DataFrame, cond: dict, ref: Dict[str, float], runtime: Optional[float], pc: Optional[pd.DataFrame] = None,
    rel: Optional[pd.DataFrame] = None) -> None:
    d = tables['data_by_class']
    L = ['# Who carries the APL effect: the per-pair APL effect split by odors per KC (E74 part A)', '',
        'A pair\'s cosine similarity is a sum over KCs; each KC is put, in each condition, in a class by the number of the 16 odors '
        'it responds to in that condition, and the per-pair APL effect (APL-on minus APL-off distance = APL-off minus APL-on '
        'similarity) splits into per-class effects. KC data: Yang\'s APL16 recordings, each genotype\'s class contributions '
        'averaged over its recordings (11 control flies, 8 APL output blocked (TNT)); 90 % intervals over resamples of recordings. '
        'paa, euc and -aPine enter the models with ORN input at substituted concentrations, as everywhere.', '']
    if runtime is not None:
        L += [f'Runtime of the scoring: {runtime / 60:.1f} min ({df[["folder", "tag"]].drop_duplicates().shape[0]} variants).', '']
    L += ['## The KC data', '', '| statistic | criterion | none | 1 | 2 to 3 | 1 to 3 | 4 to 7 | 8 or more | all KCs |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']

    def cell(crit, q, cls, term='total', ci=True, spec='+.3f'):
        r = d[(d.criterion == crit) & (d.quantity == q) & (d['class'] == cls) & (d.term == term)]
        if not len(r):
            return '–'
        r = r.iloc[0]
        return f3(r.observed, spec) + (f' ({f3(r.lo, spec)} to {f3(r.hi, spec)})' if ci and 'lo' in r and np.isfinite(r.lo) else '')
    cls_cols = ('0', '1', '2_3', '1_3', '4_7', '8plus', 'all')
    for crit in CRITERIA_Z:
        L.append(f'| mean per-pair APL effect | {crit:g} z | ' + ' | '.join(cell(crit, 'effect', c, ci=crit == 1.0) for c in cls_cols) + ' |')
    L.append('| share of the effect\'s variance over pairs | 1 z | ' + ' | '.join(cell(1.0, 'var_share', c, spec='.2f') if c != 'all' else '1'
        for c in cls_cols) + ' |')
    L.append('| responder x sub-threshold part of the mean | 1 z | ' + ' | '.join(cell(1.0, 'effect', c, 'one') for c in cls_cols) + ' |')
    L.append('| similarity level, control flies | 1 z | ' + ' | '.join(cell(1.0, 'level_on', c, ci=False) for c in cls_cols) + ' |')
    L.append('| similarity level, APL output blocked (TNT) | 1 z | ' + ' | '.join(cell(1.0, 'level_off', c, ci=False) for c in cls_cols) + ' |')
    for q, lab in (('kc_share_on', 'share of KCs, control flies'), ('kc_share_off', 'share of KCs, APL output blocked (TNT)')):
        L.append(f'| {lab} | 1 z | ' + ' | '.join(cell(1.0, q, c, '', ci=False, spec='.3f') if c != 'all' else '' for c in cls_cols) + ' |')
    fl = tables['floor']
    y = tables['yardstick']
    L += ['', 'The class-resolved loss (root mean square over the 5 classes x 120 pairs of model minus data per-class effect, lower is '
        'better): its floor, a resampled data set against the measured one (mean over the resamples, 5th to 95th percentile), '
        'beside the same for the APL-effect loss; and the single-recording-pair yardstick (one control recording with one APL '
        'output blocked (TNT) recording against the mean of the others, 88 pairs).', '',
        '| criterion | class-resolved loss floor | APL-effect loss floor (this resampling) |', '|---|---:|---:|']
    for r in fl.itertuples():
        L.append(f'| {r.criterion:g} z | {r.class_rmse_floor:.4f} ({r.class_rmse_floor_lo:.4f} to {r.class_rmse_floor_hi:.4f}) | {r.apl_rmse_floor:.4f} |')
    L += ['', f'Yardstick: class-resolved loss median {y.class_rmse.median():.3f} (range {y.class_rmse.min():.3f} to {y.class_rmse.max():.3f}); '
        f'APL-effect loss median {y.apl_rmse.median():.3f} (range {y.apl_rmse.min():.3f} to {y.apl_rmse.max():.3f}).', '']
    # key models
    cols = [('eff_0', 'none'), ('eff_1_3', '1 to 3'), ('eff_4_7', '4 to 7'), ('eff_8plus', '8 or more'), ('eff_all', 'all KCs')]
    for rd in ('cal2', 'zcell', 'zcell_free', 'cal1', 'cal3', 'zcell_trials'):
        g = df[df.reading == rd]
        if not len(g):
            continue
        L += [f'## Key models, {READING_LABEL[rd]}', '',
            '| model | ' + ' | '.join(c for _, c in cols) + ' | class-resolved loss | APL-effect loss | mean | pattern part | share of the '
            'pattern error: none / 1-3 / 4-7 / 8+ | responder x sub-threshold part / with the sub-threshold products | bands (two-spike / reading) |', '|---|' + '---:|' * (len(cols) + 6) + '---|']
        L.append('| KC data (1 z) | ' + ' | '.join(f3(ref[k]) for k, _ in cols) + f' | floor {cond["class_rmse_floor"]:.3f} | floor '
            f'{H.data_target("apl_effect_rmse_floor"):.3f} | {ref["apl_mean"]:+.3f} | – | – | {ref["eff_all_one"]:+.3f} / {ref["sub_terms"]:+.3f} | |')
        for s, lab, f, t in KEY_MODELS:
            r = g[(g.folder == f) & (g.tag == t)]
            if not len(r):
                continue
            r = r.iloc[0]
            shares = ' / '.join(f3(r[f'perr_share_{c}'], '.2f') for c in ('0',)) + ' / ' + f3(r['perr_share_1'] + r['perr_share_2_3'], '.2f') + \
                ' / ' + f3(r['perr_share_4_7'], '.2f') + ' / ' + f3(r['perr_share_8plus'], '.2f')
            bands = ('inside' if r.in_bands_spikes else (r.flag_spikes or 'not scored')) + ' / ' + (
                'inside' if r.in_bands_reading else (r.flag_reading or 'not scored'))
            L.append(f'| {s} | ' + ' | '.join(f3(r[k]) for k, _ in cols) + f' | {r.class_rmse:.3f} | {r.apl_rmse:.3f} | {r.apl_mean:+.3f} | '
                f'{r.apl_pattern:.3f} | {shares} | {r.eff_all_one:+.3f} / {r.sub_terms:+.3f} | {bands} |')
        L.append('')
    L += ['Key models: ' + '; '.join(f'{s}: {lab} ({f} {t})' for s, lab, f, t in KEY_MODELS) + '.', '']
    if pc is not None and len(pc):
        L += per_condition_md(pc)
    if rel is not None and len(rel):
        L += released_md(rel)
    # under the reading: the sub-threshold part and the band
    g = df[df.reading == 'zcell']
    if len(g):
        L += ['## Under the reading through the noise: the part of the mean below the criterion and the 0.5-1.5 z band', '',
            '| model | graded mean | sub-threshold values set to 0 | graded minus thresholded | entries in (0.5, 1.5] z, APL on -> off | growth |',
            '|---|---:|---:|---:|---|---:|',
            f'| KC data | {ref["apl_mean"]:+.3f} | {ref["mean_thresholded"]:+.3f} | {ref["sub_thresholded"]:+.3f} | {100 * ref["band_on"]:.1f} % -> '
            f'{100 * ref["band_off"]:.1f} % | {100 * ref["band_growth"]:+.1f} points |']
        for s, lab, f, t in KEY_MODELS:
            r = g[(g.folder == f) & (g.tag == t)]
            if len(r):
                r = r.iloc[0]
                L.append(f'| {s} | {r.apl_mean:+.3f} | {r.mean_thresholded:+.3f} | {r.sub_thresholded:+.3f} | {100 * r.band_on:.1f} % -> '
                    f'{100 * r.band_off:.1f} % | {100 * r.band_growth:+.1f} points |')
        L.append('')
    # wirings
    if len(wires):
        L += ['## E72\'s wirings, singly and as their mean', '', 'Each group\'s mean averages each condition\'s class contributions over '
            'its wirings (as E72 averages the distances). real = the hemibrain wiring; P1-P4 claw permutations within type; M1-M3 the '
            'co-input-preserving matched shuffles.', '']
        for g, lab, mem in WIRING_GROUPS:
            L += [f'### {lab}', '', '| wiring | reading | ' + ' | '.join(c for _, c in cols) + ' | class-resolved loss | APL-effect loss | '
                'pattern part | bands (two-spike / reading) |', '|---|---|' + '---:|' * (len(cols) + 3) + '---|']
            for rd in ('cal2', 'zcell'):
                for w, f, t in mem:
                    r = df[(df.folder == f) & (df.tag == t) & (df.reading == rd)]
                    if len(r):
                        r = r.iloc[0]
                        bands = ('inside' if r.in_bands_spikes else (r.flag_spikes or 'not scored')) + ' / ' + (
                            'inside' if r.in_bands_reading else (r.flag_reading or 'not scored'))
                        L.append(f'| {w} | {rd} | ' + ' | '.join(f3(r[k]) for k, _ in cols) + f' | {r.class_rmse:.3f} | {r.apl_rmse:.3f} | '
                            f'{r.apl_pattern:.3f} | {bands} |')
                r = wires[(wires.group == g) & (wires.reading == rd)]
                if len(r):
                    r = r.iloc[0]
                    L.append(f'| **mean of {r.n_wirings}** | {rd} | ' + ' | '.join(f3(r[k]) for k, _ in cols) + f' | {r.class_rmse:.3f} | '
                        f'{r.apl_rmse:.3f} | {r.apl_pattern:.3f} | |')
            L.append('')
    # rank correlations
    L += ['## Across variants: the class-resolved loss against the APL-effect loss and its pattern part (Spearman)', '',
        '| reading | variants | n | with the APL-effect loss | with the pattern part | 4-7 class error (absolute) with the pattern part |',
        '|---|---|---:|---:|---:|---:|']
    for r in rc.itertuples():
        L.append(f'| {r.reading} | {r.subset} | {r.n} | {f3(r.rho_class_vs_loss, ".2f")} | {f3(r.rho_class_vs_pattern, ".2f")} | '
            f'{f3(r.rho_4_7_err_vs_pattern, ".2f")} |')
    L.append('')
    # series
    if len(ser):
        L += ['## The archive\'s sweeps as series on the 4-7 class\'s effect', '',
            f'KC data: {ref["eff_4_7"]:+.3f}. Per series: the range of the 4-7 class effect over the sweep, and its largest value among the '
            'variants inside the two-spike bands (under the calibrated readout) and inside both criteria\'s bands (under the reading). '
            + NOT_IN_ARCHIVE_NOTE + '.', '',
            '| handle | series | values | 4-7 effect, calibrated readout (two spikes): first -> last (min, max) | largest inside the bands '
            '(at; class-resolved loss / APL-effect loss) | 4-7 effect, reading through the noise | largest inside the bands (at; class-resolved '
            'loss / APL-effect loss) |', '|---|---|---|---|---|---|---|']

        def rng(q):
            q = q[q.eff_4_7.notna()]
            return (f'{q.eff_4_7.iloc[0]:+.3f} -> {q.eff_4_7.iloc[-1]:+.3f} (min {q.eff_4_7.min():+.3f}, max {q.eff_4_7.max():+.3f})'
                if len(q) else '–')

        def best(q):
            q = q[q.eff_4_7.notna()]
            if not len(q):
                return 'none inside'
            r = q.loc[q.eff_4_7.idxmax()]
            return f'{r.eff_4_7:+.3f} (at {r.x:g}; {r.class_rmse:.3f} / {r.apl_rmse:.3f})'
        for (h, sr), g in ser.groupby(['handle', 'series'], sort=False):
            a, b = g[g.reading == 'cal2'], g[g.reading == 'zcell']
            xs = ', '.join(f'{x:g}' for x in sorted(g.x.unique()))
            L.append(f'| {h} | {sr} | {xs} | {rng(a)} | {best(a[a.in_bands_spikes])} | {rng(b)} | '
                f'{best(b[b.in_bands_spikes & b.in_bands_reading])} |')
        reach = ser[(ser.eff_4_7 >= 0.03) & ser.in_bands_spikes & ((ser.reading == 'cal2') | ser.in_bands_reading)
            & ser.reading.isin(['cal2', 'zcell'])]
        L += ['', 'Handles that take the 4-7 class\'s effect to +0.03 or more inside the bands (outcome (b)): ' + ('; '.join(
            f'{h} ({", ".join(sorted(set(f"{r.series}, {r.reading}" for r in g.itertuples())))})' for h, g in reach.groupby('handle'))
            if len(reach) else 'none') + '.', '']
    # best inside the bands
    L += ['## The variant with the lowest class-resolved loss inside the bands', '', '| reading | bands | variant | class-resolved loss | '
        'APL-effect loss | pattern part | 4-7 | 8 or more |', '|---|---|---|---:|---:|---:|---:|---:|']
    for rd in ('cal2', 'zcell'):
        g = df[df.reading == rd]
        for lab, m in (('two-spike', g.in_bands_spikes), ('both criteria', g.in_bands_spikes & g.in_bands_reading), ('none (all variants)', g.class_rmse.notna())):
            s = g[m]
            if len(s):
                r = s.loc[s.class_rmse.idxmin()]
                L.append(f'| {rd} | {lab} | {r.folder} {r.tag} | {r.class_rmse:.3f} | {r.apl_rmse:.3f} | {r.apl_pattern:.3f} | {r.eff_4_7:+.3f} | {r.eff_8plus:+.3f} |')
    L.append('')
    # part C's condition
    L += ['## Part C\'s condition', '', f'The data\'s 4-7 class effect: {cond["data_4_7"]:+.3f} ({cond["data_4_7_lo"]:+.3f} to {cond["data_4_7_hi"]:+.3f}); '
        f'the floor of the class-resolved loss {cond["class_rmse_floor"]:.4f}.', '',
        '| reading | fit | 4-7 class effect | outside the data\'s interval | class-resolved loss | APL-effect loss |', '|---|---|---:|---|---:|---:|']
    for rd, v in cond['readings'].items():
        for fr in v['fits']:
            L.append(f'| {rd} | {fr["model"]} | {fr["eff_4_7"]:+.3f} | {"yes" if fr["excluded"] else "no"} | {fr["class_rmse"]:.3f} | {fr["apl_rmse"]:.3f} |')
    L.append('')
    for rd, v in cond['readings'].items():
        L.append(f'- {rd}: every best fit outside the interval: {"yes" if v["all_excluded"] else "no"}; the floor at most half the best fits\' '
            f'lowest class-resolved loss ({v["lowest_class_rmse"]:.4f}, half {0.5 * v["lowest_class_rmse"]:.4f}): '
            f'{"yes" if v["floor_at_most_half"] else "no"}; condition {"holds" if v["holds"] else "does not hold"}.')
    L += ['', f'Part C\'s condition {"holds" if cond["holds"] else "does not hold"} under both readings.' if cond['holds'] else
        'Part C\'s condition does not hold under both readings (see above).', '']
    # checks
    g = df[df.reading == 'cal2']
    dc = (g.apl_rmse - g.e73_cal_rmse).abs()
    gz = df[df.reading == 'zcell']
    dz = (gz.apl_rmse - gz.e73_zcell_rmse).abs()
    gf = df[df.reading == 'zcell_free']
    dpf = (gf.apl_pattern - gf.e73_zcell_pattern).abs()
    L += ['## Checks', '',
        f'- The data\'s classes add up to the per-pair APL effect: mean {ref["eff_all"]:+.4f} against the registry\'s apl_effect_mean '
        f'{H.data_target("apl_effect_mean"):+.4f}.',
        f'- Under the calibrated readout the classes\' total reproduces E73\'s present_cal APL-effect loss: largest absolute difference '
        f'{dc.max():.2e} over {dc.notna().sum()} variants.',
        f'- Through the noise, the pseudo-recordings\' total APL-effect loss against E73\'s reading (in expectation over the noise): '
        f'median absolute difference {dz.median():.4f}, 95th percentile {dz.quantile(0.95):.4f} ({dz.notna().sum()} variants).',
        f'- Without the noise (the planning probe\'s emulation), the pattern part against E73\'s: median absolute difference '
        f'{dpf.median():.4f}, 95th percentile {dpf.quantile(0.95):.4f}.', '']
    (out / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def figures(out: Path, df: pd.DataFrame, tables: Dict[str, pd.DataFrame], ser: pd.DataFrame, ref: Dict[str, float]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
    ink, muted = '#0b0b0b', '#52514e'
    d = tables['data_by_class']
    show = ('0', '1_3', '4_7', '8plus')
    xl = [CLASS_LABEL[c] for c in show]
    models = [m for m in ('C2 both', 'ceiling 1', 'ceiling + scaling', 'ref.') if any(s == m for s, *_ in _key_archived())]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, rd, title in ((axes[0], 'cal2', 'calibrated readout\n(two-spike criterion)'), (axes[1], 'zcell', 'per-KC reading through the noise\n(data\'s criterion, 1 z)')):
        n = len(models) + 1
        w = 0.8 / n
        x = np.arange(len(show))
        dd = d[(d.criterion == 1.0) & (d.quantity == 'effect') & (d.term == 'total')].set_index('class')
        obs = np.array([dd.loc[c, 'observed'] for c in show])
        err = np.array([[obs[i] - dd.loc[c, 'lo'], dd.loc[c, 'hi'] - obs[i]] for i, c in enumerate(show)]).T
        ax.bar(x - 0.4 + w / 2, obs, w * 0.9, color=muted, label='KCs (control minus APL output blocked (TNT))', yerr=err,
            error_kw=dict(ecolor=ink, lw=1, capsize=2))
        for j, m in enumerate(models):
            _, _, f, t = next(k for k in KEY_MODELS if k[0] == m)
            r = df[(df.folder == f) & (df.tag == t) & (df.reading == rd)]
            if len(r):
                ax.bar(x - 0.4 + w * (j + 1.5), [r.iloc[0][f'eff_{c}'] for c in show], w * 0.9, color=colors[j], label=m)
        ax.axhline(0, color=muted, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xl)
        ax.set_xlabel('odors a KC responds to (in each condition)')
        ax.set_ylabel('mean per-pair APL effect carried by the class')
        ax.set_title(f'APL effect by odors per KC:\n{title}', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, loc='upper left')
    ax = axes[2]
    x = np.arange(len(show))
    w = 0.8 / max(len(models), 1)
    for j, m in enumerate(models):
        _, _, f, t = next(k for k in KEY_MODELS if k[0] == m)
        r = df[(df.folder == f) & (df.tag == t) & (df.reading == 'zcell')]
        if len(r):
            r = r.iloc[0]
            v = [r['perr_share_0'], r['perr_share_1'] + r['perr_share_2_3'], r['perr_share_4_7'], r['perr_share_8plus']]
            ax.bar(x - 0.4 + w * (j + 0.5), v, w * 0.9, color=colors[j], label=m)
    ax.axhline(0, color=muted, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xl)
    ax.set_xlabel('odors a KC responds to (in each condition)')
    ax.set_ylabel('share of the pattern error\'s variance over pairs')
    ax.set_title('Where the models\' pattern error sits\n(per-KC reading through the noise)', fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / 'E74_classes.png', dpi=130)
    plt.close(fig)
    for rd, fname, what in (('cal2', 'E74_series.png', 'calibrated readout, two-spike criterion'),
            ('zcell', 'E74_series_reading.png', 'per-KC reading through the noise, the data\'s criterion')):
        if not len(ser):
            break
        handles = list(dict.fromkeys(ser.handle))
        nc = 4
        nr = int(np.ceil(len(handles) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.4 * nr), squeeze=False)
        dd = d[(d.criterion == 1.0) & (d.quantity == 'effect') & (d.term == 'total') & (d['class'] == '4_7')].iloc[0]
        for ax, h in zip(axes.ravel(), handles):
            g = ser[(ser.handle == h) & (ser.reading == rd)]
            ax.axhspan(dd.lo, dd.hi, color='#9ec5f4', alpha=0.4, lw=0)
            ax.axhline(dd.observed, color=ink, ls='--', lw=1, label='KCs (90 % interval)')
            for j, (s, q) in enumerate(g.groupby('series', sort=False)):
                q = q.sort_values('x')
                c = colors[j % len(colors)]
                ax.plot(q.x, q.eff_4_7, color=c, lw=2, label=s)
                ins = q.in_bands_spikes.astype(bool) & (q.in_bands_reading.astype(bool) if rd == 'zcell' else True)
                ax.scatter(q.x[ins], q.eff_4_7[ins], s=30, color=c, zorder=3)
                ax.scatter(q.x[~ins], q.eff_4_7[~ins], s=30, facecolor='white', edgecolor=c, zorder=3)
            ax.axhline(0, color=muted, lw=0.6)
            if len(g) and (g.x > 0).all() and g.x.max() / g.x.min() > 50:
                ax.set_xscale('log')
            ax.set_xlabel(h)
            ax.set_ylabel('4-7 class APL effect')
            ax.legend(fontsize=6, frameon=False, loc='best')
            ax.spines[['top', 'right']].set_visible(False)
        for ax in axes.ravel()[len(handles):]:
            ax.axis('off')
        fig.suptitle(f'The archive\'s sweeps on the 4-7 class\'s APL effect ({what}; open markers outside the realism bands)', fontsize=10)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    floor = float(tables['floor'].set_index('criterion').loc[1.0, 'class_rmse_floor'])
    for ax, rd in zip(axes, ('cal2', 'zcell')):
        g = df[df.reading == rd]
        ins = g.in_bands_spikes & (g.in_bands_reading if rd == 'zcell' else True)
        ax.scatter(g.apl_pattern[~ins], g.class_rmse[~ins], s=10, facecolor='white', edgecolor=muted, lw=0.6, label='outside the bands')
        ax.scatter(g.apl_pattern[ins], g.class_rmse[ins], s=10, color=colors[0], label='inside the bands')
        ax.axhline(floor, color=ink, ls='--', lw=1, label='floor of the class-resolved loss')
        ax.set_xlabel('pattern part of the APL-effect loss')
        ax.set_ylabel('class-resolved loss')
        ax.set_title(READING_LABEL[rd].split(',')[0] if rd == 'cal2' else 'per-KC reading through the noise', fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / 'E74_class_loss_vs_pattern.png', dpi=120)
    plt.close(fig)


def write_provenance(out: Path, extra: dict) -> None:
    def sha(p: Path) -> Optional[str]:
        import hashlib
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    inputs = {str(p): sha(p) for p in (RESULTS / 'apl_effect' / 'by_variant.csv', RESULTS / 'E73_per_kc_normalization' / 'by_variant.csv',
        H.yang_data_dir() / 'computed' / 'trialmean_responses_long.parquet', H.data_targets_path(), CLAW_WIRING)}
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script=Path(__file__).name, part='A',
        host=socket.gethostname(), al_analysis_commit=_git('rev-parse', 'HEAD'), al_analysis_dirty=bool(_git('status', '--porcelain', '--', '.')),
        inputs=inputs, archive=str(ARCHIVE), classes=list(CLASSES), criteria_z=list(CRITERIA_Z), spike_criteria=list(SPIKE_CRITERIA),
        pseudo_recordings=dict(n_recordings=N_RECORDINGS, n_trials=N_TRIALS, seed=SEED, note='numpy default_rng(seed): the APL-on '
            'draws (n_recordings, n_trials, KCs, odors) first, then the APL-off draws; a trial mean\'s noise is the KC\'s trial sd under '
            'the reading times the mean of its three draws'), key_models=KEY_MODELS, wiring_groups=WIRING_GROUPS, series=[s[:4] for s in SERIES],
        model_provenance=H.model_provenance(), **extra)
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str, ensure_ascii=False) + '\n')


def summarize(out: Path, df: pd.DataFrame, blocks: Dict[tuple, dict], tables: Dict[str, pd.DataFrame], runtime: Optional[float]) -> dict:
    data = data_effect_array()
    ref = data_reference()
    wires = wiring_rows(blocks, data)
    wires.to_csv(out / 'wirings.csv', index=False)
    rc = rank_correlations(df)
    rc.to_csv(out / 'rank_correlations.csv', index=False)
    ser = series_table(df)
    ser.to_csv(out / 'series.csv', index=False)
    cond = part_c_condition(df, tables)
    (out / 'part_c_condition.json').write_text(json.dumps(cond, indent=1, default=str) + '\n')
    rows = []
    for (f, t), b in blocks.items():
        lab = key_label(f, t)
        for rd, blk in b.items():
            eff = blk['off'] - blk['on']
            for p, (a, bb) in enumerate(PAIRS):
                rows.append(dict(model=lab, folder=f, tag=t, reading=rd, odor_a=a, odor_b=bb,
                    **{f'eff_{c}': float(eff[p, i, TOTAL]) for i, c in enumerate(CLASSES_ALL)}))
    pd.DataFrame(rows).to_csv(out / 'pairs_key_models.csv', index=False)
    pc = per_condition_table(df, tables)
    pc.to_csv(out / 'per_condition.csv', index=False)
    rel = released_by_claws(df)
    rel.to_csv(out / 'released_by_claws.csv', index=False)
    write_table(out, df, tables, wires, rc, ser, cond, ref, runtime, pc, rel)
    try:
        figures(out, df, tables, ser, ref)
    except Exception as e:
        print('figures failed:', repr(e)[:300], flush=True)
    return cond


# the check of E10_joint_fit.py's option (part C): E73 part C's fit 2 (the ceiling family under the per-KC reading, hal),
# started exactly at its optimum for one evaluation with the option off must reproduce its best evaluation
E10_CHECK_START = ('E73_per_kc_normalization', 'zcaplrate_rate0.068_aplzcell_refr3_adapt0.04142t1000_k2_cap3')
E10_CHECK_ARGS = ['-k', '2', '--aplpn-max', '3', '--rate-target', 'ctrl_frac', '--loss', 'apl', '--apl-readout', 'zcell',
    '--rate-criterion', 'zcell', '--kc-refractory-period', '0.003', '--kc-adapt-inc', '0.041417845143644', '--kc-adapt-tau', '1.0',
    '--maxfev', '1', '--init-from-exact', '--tag', '_cap3']
E10_PARAM_COLS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')
# --check-e10's settings: the class term off, on in the squared form (the default), on in the linear form
E10_CHECK_EXTRA = {'off': [], 'on': ['--apl-class-term'], 'lin': ['--apl-class-term', '--apl-class-form', 'lin']}


def check_e10(name: str, extra: Sequence[str] = ()) -> dict:
    """E10_joint_fit.py with the options of E73 part C's fit 2 (`E10_CHECK_ARGS`, plus `extra`), started exactly at that
    fit's optimum, one evaluation, run through E73_zcell_fit_checks.py's --run-e10 (its scratch experiment's results
    under outputs/E74_apl_effect_by_odors_per_kc/e10_check/<name>/, nothing written to the shared results); the
    evaluation compared with the fit's best evaluation in its saved history, every column (`eval` and `time_s` aside):
    identical statistics, the columns that differ (with their difference in units in the last place), missing and
    extra columns. Writes and returns <name>/check.json."""
    import shutil
    exp, tag = E10_CHECK_START
    root = H.outputs_dir(EXPERIMENT) / 'e10_check' / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    args = [*E10_CHECK_ARGS, '--init-from', tag, '--init-from-experiment', exp, '--experiment', 'E73_zcell_check', *extra]
    cmd = [sys.executable, '-u', str(H.HERE / 'E73_zcell_fit_checks.py'),
        '--run-e10', str(root), '--', *args]
    t0 = time.time()
    with open(root / 'log.txt', 'w') as log:
        rc = subprocess.run(cmd, cwd=H.HERE, stdout=log, stderr=subprocess.STDOUT).returncode
    hist = sorted((root / 'results').glob('history_*.csv'))
    out = dict(name=name, args=args, returncode=rc, seconds=time.time() - t0, host=socket.gethostname(),
        commit=_git('rev-parse', 'HEAD'), dirty=bool(_git('status', '--porcelain', '--', '.')))
    if rc or not hist:
        out['error'] = 'run failed; see log.txt'
    else:
        new = pd.read_csv(hist[0], float_precision='round_trip').iloc[0]
        old = pd.read_csv(H.results_dir(exp) / f'history_{tag}.csv', float_precision='round_trip')
        row = old.loc[old['loss'].idxmin()]
        cols = [c for c in old.columns if c not in ('eval', 'time_s', 'Unnamed: 0')]
        diff = {}
        for c in cols:
            if c in new.index and not (new[c] == row[c] or (pd.isna(new[c]) and pd.isna(row[c]))):
                try:
                    diff[c] = dict(new=float(new[c]), saved=float(row[c]), abs=float(abs(new[c] - row[c])),
                        ulps=float(abs(new[c] - row[c]) / np.spacing(abs(float(row[c])))) if row[c] != 0 else float('inf'))
                except (TypeError, ValueError):
                    diff[c] = dict(new=str(new[c]), saved=str(row[c]))
        stat_diff = {c: v for c, v in diff.items() if c not in E10_PARAM_COLS}
        out.update(saved_eval=int(row['eval']) if 'eval' in row else None, n_columns=len(cols),
            statistics_identical=not stat_diff and all(c in new.index for c in cols),
            differing_statistics=stat_diff, differing_parameters={c: v for c, v in diff.items() if c in E10_PARAM_COLS},
            missing=[c for c in cols if c not in new.index], extra=[c for c in new.index if c not in old.columns and c not in ('eval', 'time_s')],
            loss_new=float(new['loss']), loss_saved=float(row['loss']),
            key_values={c: (float(new[c]), float(row[c])) for c in ('loss', 'apl_zcell_rmse', 'apl_zcell_mean', 'apl_zcell_pattern',
                'zcell_rate_on', 'zcell_rate_off', 'term_apl') if c in new.index and c in row.index},
            class_term={c: float(new[c]) for c in new.index if c.startswith(('apl_class_', 'term_apl_class'))})
    (root / 'check.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--jobs', type=int, default=2, help='worker processes for the variant loop')
    ap.add_argument('--only-key', action='store_true', help='only the key models and E72\'s wirings')
    ap.add_argument('--boot', type=int, default=2000, help='resamples of recordings for the data\'s intervals and floor')
    ap.add_argument('--summarize-only', action='store_true', help='the tables and figures again from the saved by_variant.csv')
    ap.add_argument('--check-e10', choices=list(E10_CHECK_EXTRA), default=None, help='part C\'s check of E10_joint_fit.py instead '
        'of part A: E73 part C\'s fit 2 started at its optimum for one evaluation, with --apl-class-term off (must reproduce the '
        'saved best evaluation), on (the squared form) or on in the linear form (lin); set OMP_NUM_THREADS for the simulations')
    args = ap.parse_args()
    if args.check_e10:
        r = check_e10(f'option_{args.check_e10}', E10_CHECK_EXTRA[args.check_e10])
        print(json.dumps(r, indent=1, default=str), flush=True)
        return
    warnings.simplefilter('ignore', RuntimeWarning)
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    local = H.outputs_dir(EXPERIMENT)
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tables = data_tables(args.boot, seed=0)
    for k, v in tables.items():
        v.to_csv(out / f'{k}.csv', index=False)
    print(f'data tables ({time.time() - t0:.0f} s)', flush=True)
    name = 'by_variant_key.csv' if args.only_key else 'by_variant.csv'
    blocks_path = local / ('blocks_key.pkl' if args.only_key else 'blocks.pkl')
    runtime = None
    if args.summarize_only:
        df = pd.read_csv(out / name, low_memory=False)
        df['tag'] = df['tag'].astype(str)
        blocks = pickle.loads(blocks_path.read_bytes())
    else:
        jobs = variant_jobs(args.only_key)
        print(f'{len(jobs)} variants, {args.jobs} workers', flush=True)
        t1 = time.time()
        rows, errors, blocks = [], [], {}
        data = data_effect_array()
        with ProcessPoolExecutor(max_workers=max(1, args.jobs), initializer=_init_worker, initargs=(data,)) as ex:
            for i, (res, kept) in enumerate(ex.map(score_variant, jobs, chunksize=4)):
                for r in res:
                    (errors if 'error' in r else rows).append(r)
                if kept is not None:
                    blocks[(jobs[i][0], jobs[i][1])] = kept
                if (i + 1) % 100 == 0:
                    print(f'{i + 1} / {len(jobs)} variants ({time.time() - t1:.0f} s)', flush=True)
        runtime = time.time() - t1
        print(f'scored in {runtime:.0f} s', flush=True)
        df = merge_bands(pd.DataFrame(rows))
        df['key'] = [key_label(f, t) for f, t in zip(df.folder, df.tag)]
        df.to_csv(out / name, index=False)
        blocks_path.write_bytes(pickle.dumps(blocks))
        if errors:
            pd.DataFrame(errors).to_csv(out / 'errors.csv', index=False)
            print(f'{len(errors)} variants failed: see errors.csv', flush=True)
    if 'in_bands_spikes' not in df:
        df = merge_bands(df)
    df['in_bands_spikes'] = df['in_bands_spikes'].astype(bool)
    df['in_bands_reading'] = df['in_bands_reading'].astype(bool)
    for col in ('flag_spikes', 'flag_reading'):
        df[col] = df[col].fillna('')
    cond = summarize(out, df, blocks, tables, runtime)
    write_provenance(out, dict(n_variants=int(df[['folder', 'tag']].drop_duplicates().shape[0]), n_rows=int(len(df)), n_boot=args.boot,
        runtime_s=runtime, jobs=args.jobs, part_c_condition_holds=cond['holds']))
    print(f'part C condition holds: {cond["holds"]}; -> {out}', flush=True)


if __name__ == '__main__':
    main()
