#!/usr/bin/env python3
"""E81 parts D and E: the model's trial-to-trial variability re-specified with the flies' shares, and what the state
shared by the odors of a repeat could be in the model (the flies' variability across the three repeats, E81).

The pipeline's variability (E43_model_trial_noise.py part B) is one threshold draw per KC and trial, used for all 16
odors of the trial: all of it is shared by the odors of a repeat (a trial stands in for a repeat; the model's trials are
exchangeable). The flies' variability is mostly independent between the presentations of a repeat, with a smaller part
shared by them (the same-repeat gap: two odors of one repeat are closer than two odors of different repeats). Part D
gives the model both parts, from saved trials only.

Part D (`--part D`, no simulations). Per model (the breadth-2 family's four wirings, C2, P1, P2, M1; the firing-ceiling
family's four, E69's fit 1 and E72 part G's three; and, for the re-reads, every model with pipeline trials in the archive
folders of the block through the noise (E48), the noise-robust block (E51) and the data side of the main target's fits
(E76)), per readout ('raw' spike counts; 'cal', E65's saturating readout n / (n + K), K set on the model's deterministic
APL-on counts to the control flies' responders-only participation ratio in the unit), per unit of the flies' data ('z',
Yang's whole-recording z; 'noise', baseline-noise units) and per condition (APL on against the control flies, APL off
against the flies with APL output blocked (TNT)):
- the levels: every complete set of pipeline trials found (ten pseudo-recordings of three trials; E81's own part C
  first, then the source experiment's, local outputs before the shared archive) and the deterministic run as level 0
  (three identical trials, ten copies);
- the independent part, added to each level's saved trial counts: each KC x odor x trial count times exp(s e - s^2 / 2)
  (e standard normal, so the factor has mean 1), read out, converted to the unit (c, E68's conversion: the factor at
  which the model's APL-on response rate under the data's criterion, a trial mean above 1 z or above the registry's
  noise_threshold in baseline-noise units, through the Gaussian noise below, is the control flies'; set per level on
  APL on and used unchanged for APL off), plus Gaussian noise of the silent entries' size (the flies' trial-to-trial
  sd of the entries whose trial mean is at or below the criterion, `silent_sd`, per genotype); draws seeded per model
  and condition, the same at every level and spread (common random numbers);
- the shared part: the jitter at which the model's same-repeat gap (the mean over the 120 pairs of the single-trial
  distance between two odors of different trials minus that between two odors of the same trial) is the flies',
  interpolated linearly in sigma^2 between the two levels that bracket it (the gap grows as sigma^2 in the saved
  calibration trials); every statistic at that jitter is the same interpolation of the statistic at the two levels
  (flag `bracket_from_deterministic` when the bracket runs from the deterministic run to a level more than 1.5 times
  the matched jitter, so that the sigma^2 shape carries the interpolation: E81 part C's `lo` level is there to avoid it;
  `gap_not_bracketed` when no saved level reaches the flies' gap, the nearest level then standing in);
- the calibration is joint: the gap is matched with the independent part in place and the spread s so that the
  same-odor distance between two single trials is the flies' (0.444 / 0.394 in z), alternated to a fixed point
  (`calibrate`); the jitter matched with s = 0 is kept as `sigma_gap_only`;
- checks not used in the calibration: how far the means of three trials exceed the noise-free distances (all pairs and
  the six pairs with near-identical ORN input), and the same-odor distance and the same-repeat gap with the single
  trials binarized at the criterion, each beside the flies' value from the same function; `bin_check_miss` when either
  binarized value misses the flies' by more than a third (the entry then asks for a per-presentation threshold jitter,
  not written here);
- scores (mix_scores.csv, one row per model or population, readout, unit and kind of distance), as E72 part E scores:
  'mix' (the three-trial means through the re-specified noise) against the flies' effect as measured; 'mix_noise_free'
  (the model's noise-free distances from products of different trials) against the flies' noise-free effect;
  'mix_corrected' (the noise correction in use, noise assumed independent between odors) against the flies' corrected
  effect; 'shared_only' (the jitter alone at the matched level, no independent part) and 'det' (the deterministic run)
  against the effect as measured; 'present_match_on' / 'present_match_off' (the present pipeline: both conditions at
  the model's matched level, no independent part; on raw counts in z these are E72 part E's and E75 part A's rows,
  checked). Per row: the APL-effect loss with its mean, per-odor and pair-specific parts, ρ on the per-pair APL effect,
  the ester and alcohol groups' pattern errors and their errors in each condition, the six pairs in both conditions,
  the block (the fruity x small-ester isotonic residual), the mean distances and the response rates (the model's
  per-trial rate at two spikes, interpolated; under the data's criterion after the mixing, single trials and means of
  three). Populations: the mean over the models of their pair distances per condition (each model at its own
  calibration): the breadth-2 four, the firing-ceiling four and all eight.

Part D's check, the independent part as a threshold jitter drawn per presentation (`--part D-presentation`, simulations;
the entry's alternative when the binarized checks miss by more than a third). For C2 and F1 (E69's fit 1), on E43's
pipeline model with its per-presentation jitter (E43_model_trial_noise.py, `ExcitabilityModel.trials_presentation`):
each presentation's threshold is thr0 * (1 + sigma_s xi + sigma_p zeta), xi the KC's draw for the trial (shared by the 16
odors; E43's draws at seed 43, so the pseudo-recordings' shared draws are part C's) and zeta its own for the odor, each
trial assembled from one run per odor, in `--workers` processes forked from this one. Per condition (APL on, APL off):
- the shared jitter sigma_s is part D's level for that condition (mix_levels.json, raw spike counts, Yang's z), and the
  per-presentation SD sigma_p is fitted so that the same-odor distance between two single trials is the flies' (0.444 /
  0.394 in z), with the conversion and the silent entries' Gaussian noise of part D in place (in place of its lognormal
  factor); at every sigma_p the mean threshold is re-tuned so that the per-trial APL-on response rate is the
  deterministic run's (E43's tuning), and the APL-off runs of a level use the threshold tuned there. The search starts
  from part D's saved levels (a presentation's threshold has relative SD sqrt(sigma_s^2 + sigma_p^2) under either noise,
  and the same-odor distance involves single presentations only, so the saved shared-jitter levels give the starting
  sigma_p and the slope), and runs on `--search-recordings` pseudo-recordings (E43's tuning draws) to within `--same-tol`
  or `--search-max` evaluations;
- then `--n-recordings` pseudo-recordings of three trials at the fitted levels (APL on at the APL-on level, APL off at
  the APL-off level), converted to the flies' unit with the conversion set on the APL-on runs of each level (E68's, as
  part D), plus the Gaussian noise of the silent entries, under both readouts and both units;
- scored as part D scores the re-specified noise: per_presentation.csv (kinds 'presentation', the three-trial means,
  against the flies' effect as measured; 'presentation_noise_free'; 'presentation_corrected'; 'det'), with the loss and
  its parts, ρ on the per-pair APL effect, the group errors, the six pairs, the block and the response rates;
  per_presentation_checks.csv (the same-odor distance, the same-repeat gap and the means of three against the
  noise-free distances, graded and binarized, beside the flies' and part D's lognormal re-specification);
  per_presentation_search.csv; provenance_D_presentation.json. Checks: the model's single-odor path reproduces its
  deterministic run, and one trial of part C's saved match_on level re-run through it with sigma_p = 0 reproduces the
  saved trial entry for entry. `--smoke`: C2, the checks, one search evaluation and one pseudo-recording with APL on,
  timed (files with _smoke).

Part E (`--part E`, simulations; `--part E-tables` reads them; no gate): what the shared state could be.
- Twelve simulations per model (C2, E67; E69's fit 1, the firing ceiling at the literature setting), on E69's
  fixed-scale model wrapper (`E69_kc_firing_ceiling.Model`), each first re-run with APL on and off to reproduce its
  archived spike counts exactly: the somatic threshold (fixed_thr, common to all KCs) times 0.8, 0.9, 1.1 and 1.2, APL
  on and APL off (C2's points also checked against E76 part E's saved runs where present). `--part E-tables`: each
  triplet of factors (0.9, 1, 1.1 and 0.8, 1, 1.2) read as the three repeats of one recording: the same-repeat gap from
  a state common to all KCs, with APL on and off, and its ratio between the conditions, against the per-KC jitter's
  (the saved trials at the matched level) and the flies' (shared_state.csv);
- no simulations: the run-down (the response rate in each repeat of the session over the session's mean, both
  genotypes, by the bands' criterion in baseline-noise units and by 1 z) placed on the model's two axes of E76 part E
  (a graded block of APL's output; the threshold, APL on and off) for C2 and E69's fit 2, and on the threshold axis of
  E69's fit 1 from part E's runs: the threshold factor that gives the APL-off run-down, the APL-on change that factor
  alone gives, and the change of APL's strength the rest needs (log rates taken as additive in the two axes; a factor
  above 1, APL stronger than fitted, extrapolated from the block axis's last segment and flagged) (rundown_axes.csv).

`--part table` writes model_noise_table.md and the figures (E81_mix_calibration.png, E81_mix_scores.png,
E81_shared_state.png) from the tables.

The repeat statistics are computed here in numpy (`kind_arrays`), to the definitions of the harness functions
(`kc_remapping_harness.distances_by_kind`, `same_repeat_gap`, `noise_free_pair_distances`; test/test_e81_model_noise.py
checks them against each other once the harness's bodies exist): a calibration evaluates them thousands of times.

Outputs in results/E81_repeat_structure/: mix_levels.json, mix_scores.csv, mix_pairs.csv, mix_checks.csv,
populations.csv, provenance_D.json (part D); per_presentation.csv, per_presentation_checks.csv,
per_presentation_search.csv, per_presentation_build_checks.json, provenance_D_presentation.json (part D's check);
shared_state_runs.csv, shared_state.csv, rundown_axes.csv, provenance_E.json (part E); model_noise_table.md and the
figures (`--part table`). Part E's counts in outputs/E81_repeat_structure/partE/, part D's check's trials in
outputs/E81_repeat_structure/partD_presentation/.
Usage (from this folder):
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ../../venv/bin/python -u E81_model_noise.py --part D [--jobs 2]
           [--models C2 ...] [--no-rereads]
       OMP_NUM_THREADS=12 OMP_THREAD_LIMIT=12 nice -n 10 ../../venv/bin/python -u E81_model_noise.py --part E --threads 12
           [--smoke]
       OMP_NUM_THREADS=12 OMP_THREAD_LIMIT=12 nice -n 10 ../../venv/bin/python -u E81_model_noise.py --part D-presentation
           [--workers 10] [--models C2 F1] [--search-recordings 2] [--n-recordings 3] [--smoke]
       ../../venv/bin/python -u E81_model_noise.py --part E-tables | table
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E81_repeat_structure'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
ODORS = list(H.APL16_ODORS)
PAIRS = H.pair_list(ODORS)
PAIR_INDEX = pd.MultiIndex.from_tuples(PAIRS, names=['odor_a', 'odor_b'])
PA = np.array([ODORS.index(a) for a, _ in PAIRS])
PB = np.array([ODORS.index(b) for _, b in PAIRS])
CONDS = ('on', 'acute_off')
GENOTYPE = {'on': 'TNT3c', 'acute_off': 'TNT'}
UNITS = ('z', 'noise')
READOUTS = ('raw', 'cal')
# the six pairs with near-identical ORN input (E78_claw_knee.SIX)
SIX = (('eb', 'ep'), ('6al', 't2h'), ('IaA', 'pa'), ('2h', 'pa'), ('1-5ol', '1-6ol'), ('2h', 'IaA'))
ESTER_GROUP = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL_GROUP = ('1-8ol', '1-5ol', '1-6ol', '6al')
N_REC, N_TRIALS = 10, 3
SEED = 81
S_MAX = 3.0
# the probe's values in the entry (Yang's z), shown beside this script's own computation of the flies' side
PROBE = dict(bin_same=(0.49, 0.47), bin_gap=(0.026, 0.064), inflation=(0.029, 0.036), inflation_six=(0.086, 0.063),
             gap=(0.025, 0.070))

# the eight models of part C: label -> (pipeline model name, family); the sources searched for their trials, in order
FAMILIES = {'breadth2': 'the breadth-2 family (C2, E67, on four wirings)',
            'ceiling': "the firing-ceiling family (E69's fit 1 at the literature setting, on four wirings)"}
_W = {'P1': 'wiring-claw_permutation_within_type-seed1', 'P2': 'wiring-claw_permutation_within_type-seed2',
      'M1': 'wiring-curveball_within_type_matched-seed1'}
MODELS = {
    'C2': ('E67-rate0.068_aplboth_k2_cap3', 'breadth2'),
    **{k: (f'E72-{w}_rate0.068_aplboth_k2_cap3', 'breadth2') for k, w in _W.items()},
    'F1': ('E69-rate0.068_aplboth_refr3_adapt0.04142t1000_k2', 'ceiling'),
    **{f'F1-{k}': (f'E72-{w}_rate0.068_aplboth_refr3_adapt0.04142t1000_k2_cap3', 'ceiling') for k, w in _W.items()},
}
FAMILY_SOURCES = {'breadth2': (EXPERIMENT, 'E72_wiring_population'), 'ceiling': (EXPERIMENT, 'E75_glomerulus_apl_profile')}
POPULATIONS = {'breadth-2 four': ['C2', 'P1', 'P2', 'M1'], 'firing-ceiling four': ['F1', 'F1-P1', 'F1-P2', 'F1-M1'],
               'all eight': ['C2', 'P1', 'P2', 'M1', 'F1', 'F1-P1', 'F1-P2', 'F1-M1']}
# the re-reads: every model with pipeline trials in these experiments' partB folders
REREAD_SOURCES = ('E48_block_target_noise', 'E51_noise_robust_block', 'E76_data_side_target')
KINDS = ('mix', 'mix_noise_free', 'mix_corrected', 'shared_only', 'det', 'present_match_on', 'present_match_off')
# which flies' distances each kind of model distance is scored against
DATA_KIND = {'mix': 'trial_mean', 'mix_noise_free': 'noise_free', 'mix_corrected': 'corrected', 'shared_only': 'trial_mean',
             'det': 'trial_mean', 'present_match_on': 'trial_mean', 'present_match_off': 'trial_mean'}


# ---------------------------------------------------------------------------
# the repeat statistics (numpy; the definitions of the harness's distances_by_kind / noise_free_pair_distances)
def kind_arrays(Y: np.ndarray) -> Dict[str, np.ndarray]:
    """Single-trial cosine statistics of pseudo-recordings (or recordings) Y (R, T, units, odors), per recording r:

    - 'same_odor' (R, O): 1 - the mean cosine between odor o in trial i and in trial j, over i != j;
    - 'same_rep' (R, P): 1 - the mean over trials i of the cosine between odors a and b of trial i (P = the 120 canonical
      pairs, `PAIRS`);
    - 'diff_rep' (R, P): 1 - the mean over ordered trial pairs i != j of the cosine between a in i and b in j;
    - 'adjacent', 'first_third' (R, P): the same over |i - j| == 1 and == 2 (first_third NaN with fewer than 3 trials);
    - 'same_odor_adjacent', 'same_odor_first_third' (R, O);
    - 'noise_free' (R, P): 1 - C_ab / sqrt(C_aa C_bb), C_ab the mean cosine between a in i and b in j over i != j, C_aa the
      same for a with itself;
    - 'trial_mean' (R, P): the cosine distance of the two odors' trial means;
    - 'corrected' (R, P): the noise correction in use (`noise_floor_trial_distances.analyse`, graded): 1 - xbar_a.xbar_b /
      sqrt(P_a P_b), P = |xbar|^2 - (trial-to-trial variance summed over units) / T, floored at 1e-12.
    NaN cosines (an all-zero vector) are skipped in every mean."""
    Y = np.asarray(Y, dtype=float)
    R, T, N, O = Y.shape
    nrm = np.sqrt((Y ** 2).sum(axis=2))                                   # R, T, O
    with np.errstate(invalid='ignore', divide='ignore'):
        U = Y / nrm[:, :, None, :]
    A = U.transpose(0, 1, 3, 2).reshape(R, T * O, N)
    C = np.matmul(A, A.transpose(0, 2, 1)).reshape(R, T, O, T, O)
    ii, jj = np.meshgrid(np.arange(T), np.arange(T), indexing='ij')
    off = ii != jj
    sel = {'diff': off, 'adjacent': np.abs(ii - jj) == 1, 'first_third': np.abs(ii - jj) == 2}
    out = {}
    with np.errstate(invalid='ignore'), _quiet():
        same_odor_cos = {}
        for k, m in sel.items():
            ti, tj = ii[m], jj[m]
            if not len(ti):
                same_odor_cos[k] = np.full((R, O), np.nan)
                out[{'diff': 'diff_rep'}.get(k, k)] = np.full((R, len(PAIRS)), np.nan)
                continue
            # C[r, ti, o, tj, o] and C[r, ti, a, tj, b] (both orders of the pair)
            so = np.stack([C[:, a, :, b, :][:, np.arange(O), np.arange(O)] for a, b in zip(ti, tj)], axis=1)
            same_odor_cos[k] = np.nanmean(so, axis=1)
            ab = np.stack([C[:, a, PA, b, PB] for a, b in zip(ti, tj)] + [C[:, a, PB, b, PA] for a, b in zip(ti, tj)], axis=1)
            out[{'diff': 'diff_rep'}.get(k, k)] = 1 - np.nanmean(ab, axis=1)
        out['same_odor'] = 1 - same_odor_cos['diff']
        out['same_odor_adjacent'] = 1 - same_odor_cos['adjacent']
        out['same_odor_first_third'] = 1 - same_odor_cos['first_third']
        sr = np.stack([C[:, t, PA, t, PB] for t in range(T)], axis=1)
        out['same_rep'] = 1 - np.nanmean(sr, axis=1)
        caa = same_odor_cos['diff']
        out['noise_free'] = 1 - (1 - out['diff_rep']) / np.sqrt(caa[:, PA] * caa[:, PB])
        xbar = Y.mean(axis=1)                                             # R, N, O
        n2 = (xbar ** 2).sum(axis=1)                                      # R, O
        dots = np.einsum('rna,rnb->rab', xbar, xbar)
        with np.errstate(divide='ignore'):
            tm = 1 - dots / np.sqrt(n2[:, :, None] * n2[:, None, :])
        out['trial_mean'] = tm[:, PA, PB]
        if T > 1:
            s2 = Y.var(axis=1, ddof=1).sum(axis=1)                        # R, O
            sig = np.maximum(n2 - s2 / T, 1e-12)
            cor = 1 - dots / np.sqrt(sig[:, :, None] * sig[:, None, :])
            out['corrected'] = cor[:, PA, PB]
        else:
            out['corrected'] = np.full_like(out['trial_mean'], np.nan)
    return out


class _quiet:
    """Silences numpy's 'mean of empty slice' warnings inside `kind_arrays`."""

    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter('ignore', category=RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


SIX_IDX = np.array([PAIRS.index(p) if p in PAIRS else PAIRS.index(p[::-1]) for p in SIX])


def gap_same(Y: np.ndarray) -> Tuple[float, float]:
    """(the same-repeat gap over the 120 pairs, the same-odor single-trial distance), each the mean over (pseudo-)
    recordings of the recording's mean over pairs / odors: the two calibration statistics, as `kind_arrays` gives them
    (computed from the cosines alone, for the calibration's inner loop)."""
    Y = np.asarray(Y, dtype=float)
    R, T, N, O = Y.shape
    nrm = np.sqrt((Y ** 2).sum(axis=2))
    with np.errstate(invalid='ignore', divide='ignore'):
        U = Y / nrm[:, :, None, :]
    A = U.transpose(0, 1, 3, 2).reshape(R, T * O, N)
    C = np.matmul(A, A.transpose(0, 2, 1)).reshape(R, T, O, T, O)
    ii, jj = np.nonzero(~np.eye(T, dtype=bool))
    with _quiet():
        so = np.nanmean(np.stack([C[:, i, np.arange(O), j, np.arange(O)] for i, j in zip(ii, jj)], axis=1), axis=1)
        diff = np.nanmean(np.stack([C[:, i, PA, j, PB] for i, j in zip(ii, jj)], axis=1), axis=1)
        same = np.nanmean(np.stack([C[:, t, PA, t, PB] for t in range(T)], axis=1), axis=1)
        gap = np.nanmean(same - diff, axis=1)
        return float(np.nanmean(gap)), float(np.nanmean(1 - np.nanmean(so, axis=1)))


def summary_stats(k: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Recording means (over pairs / odors) then means over recordings of `kind_arrays` output, and the per-pair means
    over recordings of the pair distances."""
    def m(a):
        return float(np.nanmean(a))
    gap = k['diff_rep'] - k['same_rep']
    infl = k['trial_mean'] - k['noise_free']
    out = dict(same_odor=m(np.nanmean(k['same_odor'], axis=1)), same_odor_adjacent=m(np.nanmean(k['same_odor_adjacent'], axis=1)),
        same_odor_first_third=m(np.nanmean(k['same_odor_first_third'], axis=1)),
        same_rep=m(np.nanmean(k['same_rep'], axis=1)), diff_rep=m(np.nanmean(k['diff_rep'], axis=1)),
        gap=m(np.nanmean(gap, axis=1)), gap_six=m(np.nanmean(gap[:, SIX_IDX], axis=1)),
        inflation=m(np.nanmean(infl, axis=1)), inflation_six=m(np.nanmean(infl[:, SIX_IDX], axis=1)))
    for name in ('trial_mean', 'noise_free', 'corrected', 'same_rep', 'diff_rep'):
        with _quiet():
            out[f'pairs_{name}'] = pd.Series(np.nanmean(k[name], axis=0), index=PAIR_INDEX)
    return out


# ---------------------------------------------------------------------------
# the flies' side
def flies_trials(genotype: str, unit: str) -> Dict[tuple, np.ndarray]:
    """{recording: (repeats, KCs, odors)} single-trial values of one genotype in `unit`: the harness's
    `yang_recording_trials` once implemented, else olfdb's long files (z; noise_floor_trial_distances) or hal's
    per-trial file (baseline-noise units; E76_data_side_target), repeats in olfdb's `repeat` order."""
    try:
        return {k: np.asarray(v, dtype=float) for k, v in H.yang_recording_trials(genotype, unit).items()}
    except NotImplementedError:
        pass
    if unit == 'z':
        import noise_floor_trial_distances as NF
        recs = NF.recording_arrays(NF.load_trials(genotype))
    else:
        import E76_data_side_target as E76
        recs = E76._trial_arrays_noise(genotype)
    return {(str(r[0]), int(r[1]), str(r[2])): np.stack([by[o] for o in ODORS], axis=-1) for r, by in recs.items()}


def unit_threshold(unit: str) -> float:
    return float(H.KC_RESPONSE_Z if unit == 'z' else H.data_target('noise_threshold'))


def unit_ctrl_rate(unit: str) -> float:
    return float(H.data_target('ctrl_frac' if unit == 'z' else 'noise_ctrl_frac'))


def silent_sd(recs: Dict[tuple, np.ndarray], thresh: float) -> float:
    """The trial-to-trial sd of the silent entries (trial mean at or below `thresh`): the root of the mean over those
    entries of their variance across the repeats (ddof 1), pooled over recordings (`kc_trial_noise_sd`'s definition)."""
    v = []
    for X in recs.values():
        var = X.var(axis=0, ddof=1)
        v.append(var[X.mean(axis=0) <= thresh])
    return float(np.sqrt(np.concatenate(v).mean()))


def flies_side(unit: str) -> Dict[str, Dict[str, object]]:
    """Per genotype in `unit`: the repeat statistics of the flies (`summary_stats` over recordings, graded and with the
    single trials binarized at the criterion), the silent entries' sd, the response rates (single trials and means of
    three above the criterion) and each recording's gap."""
    thr = unit_threshold(unit)
    out = {}
    for g in ('TNT3c', 'TNT'):
        recs = flies_trials(g, unit)
        per = [kind_arrays(X[None]) for X in recs.values()]
        k = {name: np.concatenate([p[name] for p in per]) for name in per[0]}
        st = summary_stats(k)
        perb = [kind_arrays((X > thr).astype(float)[None]) for X in recs.values()]
        kb = {name: np.concatenate([p[name] for p in perb]) for name in perb[0]}
        stb = summary_stats(kb)
        st.update(bin_same=stb['same_odor'], bin_gap=stb['gap'], bin_gap_six=stb['gap_six'],
            silent_sd=silent_sd(recs, thr), n_recordings=len(recs),
            rate_trial=float(np.mean([(X > thr).mean() for X in recs.values()])),
            rate_mean=float(np.mean([(X.mean(axis=0) > thr).mean() for X in recs.values()])),
            gap_by_recording={'_'.join(map(str, r)): float(np.nanmean(p['diff_rep'] - p['same_rep'])) for r, p in zip(recs, per)})
        out[g] = st
    return out


# ---------------------------------------------------------------------------
# the model's saved levels
def _dirs(exp: str) -> List[Path]:
    return [H.outputs_dir(exp) / 'partB', ARCHIVE / exp / 'partB']


_TRIAL_RE = re.compile(r'_sigma([0-9.eE+-]+)_n(\d+)_rep(\d+)\.npz$')


def model_levels(name: str, exps: Sequence[str]) -> Tuple[List[dict], Optional[Path]]:
    """The saved levels of pipeline model `name`: every sigma with all `N_REC` pseudo-recordings of `N_TRIALS` trials
    in one folder (the first folder of `exps`' partB folders, local outputs before the archive, that has it complete),
    sorted by sigma; and the folder holding its deterministic run (None if none does)."""
    levels: Dict[float, dict] = {}
    det_dir = None
    for exp in exps:
        for d in _dirs(exp):
            if det_dir is None and all((d / f'{name}_{c}_spike_counts.parquet').exists() for c in CONDS):
                det_dir = d
            td = d / 'trials'
            if not td.is_dir():
                continue
            found: Dict[str, Dict[int, Path]] = {}
            for p in td.glob(f'{name}_sigma*_n{N_TRIALS}_rep*.npz'):
                m = _TRIAL_RE.search(p.name)
                if m is None or p.name[:m.start()] != name:
                    continue
                found.setdefault(m.group(1), {})[int(m.group(3))] = p
            for tag, reps in found.items():
                sig = float(tag)
                if sig in levels or not all(r in reps for r in range(N_REC)):
                    continue
                levels[sig] = dict(sigma=sig, tag=tag, source=exp, dir=str(d), paths=[reps[r] for r in range(N_REC)])
    return [levels[s] for s in sorted(levels)], det_dir


def matched_levels(name: str, exps: Sequence[str]) -> Dict[str, float]:
    """The model's matched sigmas ({'on': ..., 'acute_off': ...}) from the first of `exps`' results/partB/noise_levels.json
    that has it."""
    for exp in exps:
        p = H.results_dir(exp) / 'partB' / 'noise_levels.json'
        if p.exists():
            js = json.loads(p.read_text())
            if name in js and 'matched' in js[name]:
                return {c: float(js[name]['matched'][c]['sd']) for c in CONDS}
    return {}


def reread_models() -> Dict[str, Tuple[str, Tuple[str, ...]]]:
    """{label: (pipeline model name, (source experiment,))} of every model with complete pipeline trials in the archive
    folders of `REREAD_SOURCES`."""
    out = {}
    for exp in REREAD_SOURCES:
        names = set()
        for d in _dirs(exp):
            td = d / 'trials'
            if td.is_dir():
                for p in td.glob(f'*_sigma*_n{N_TRIALS}_rep00.npz'):
                    m = _TRIAL_RE.search(p.name)
                    if m:
                        names.add(p.name[:m.start()])
        for n in sorted(names):
            out[f'{exp.split("_")[0]}:{n}'] = (n, (exp,))
    return out


def _seed_key(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


class ModelMix:
    """One model's saved levels with the independent part added: `Y` gives a level's mixed trials (R, T, KCs, odors) in
    the flies' unit; `calibrate` sets the jitter and the spread; `final` the statistics at the calibration."""

    def __init__(self, label: str, name: str, exps: Sequence[str], flies: Dict[str, Dict[str, Dict[str, object]]]):
        self.label, self.name, self.exps = label, name, tuple(exps)
        lv, det_dir = model_levels(name, exps)
        if det_dir is None:
            raise FileNotFoundError(f'{name}: no deterministic run in {exps}')
        self.det_dir = det_dir
        self.det = {}
        self.det_rate = {}
        for c in CONDS:
            s = pd.read_parquet(det_dir / f'{name}_{c}_spike_counts.parquet')
            s = H.abbrev_columns(s.astype(float))[ODORS]
            r = H.abbrev_columns(pd.read_parquet(det_dir / f'{name}_{c}_responses.parquet').astype(float))[ODORS]
            self.det[c] = s
            self.det_rate[c] = float(r.to_numpy().mean())
        n_kc = len(self.det['on'])
        self.levels = [dict(sigma=0.0, tag='0', source='deterministic', dir=str(det_dir), paths=[])] + lv
        self.counts, self.rate_spk, self.fixed_thr = [], [], []
        for L in self.levels:
            if not L['paths']:
                self.counts.append({c: np.broadcast_to(self.det[c].to_numpy(dtype=float), (N_REC, N_TRIALS, n_kc, len(ODORS))).copy()
                    for c in CONDS})
                self.rate_spk.append(dict(self.det_rate))
                self.fixed_thr.append(np.nan)
                continue
            arrs = []
            for p in L['paths']:
                with np.load(p) as z:
                    arrs.append({k: z[k] for k in ('on_spk', 'acute_off_spk', 'on_resp', 'acute_off_resp', 'fixed_thr')})
            if arrs[0]['on_spk'].shape[1] != n_kc:
                raise ValueError(f'{name} sigma {L["sigma"]}: {arrs[0]["on_spk"].shape[1]} KCs, deterministic run {n_kc}')
            self.counts.append({c: np.stack([a[f'{c}_spk'] for a in arrs]).astype(float) for c in CONDS})
            self.rate_spk.append({c: float(np.mean([a[f'{c}_resp'].mean() for a in arrs])) for c in CONDS})
            self.fixed_thr.append(float(arrs[0]['fixed_thr']))
        self.sigmas = np.array([L['sigma'] for L in self.levels])
        shape = (N_REC, N_TRIALS, n_kc, len(ODORS))
        self.eps, self.eta = {}, {}
        for i, c in enumerate(CONDS):
            ss = np.random.SeedSequence(SEED, spawn_key=(_seed_key(name), i))
            g = np.random.default_rng(ss)
            self.eps[c] = g.standard_normal(shape)
            self.eta[c] = g.standard_normal(shape)
        import E76_data_side_target as E76
        self.K = {u: H.calibrate_readout(self.det['on'], 'saturating', E76.cal_target(u), stat='participation_ratio',
            responders_min=2.0) for u in UNITS}
        self.flies = flies
        self.gsd = {u: {c: float(flies[u][GENOTYPE[c]]['silent_sd']) for c in CONDS} for u in UNITS}
        self._c: Dict[tuple, float] = {}
        self._cache: Dict[tuple, Tuple[float, float]] = {}
        self.matched = matched_levels(name, exps)

    def read(self, x: np.ndarray, readout: str, unit: str) -> np.ndarray:
        if readout == 'raw':
            return x
        if readout == 'cal':
            return x / (x + self.K[unit])
        raise ValueError(readout)

    def conv(self, readout: str, unit: str, i: int) -> float:
        """E68's conversion of readout units to `unit` at level i: the factor at which the APL-on response rate under the
        data's criterion (trial mean of three trials above the criterion, through the control flies' silent-entry noise)
        is the control flies'; set on the level's APL-on trials without the lognormal factor."""
        key = (readout, unit, i)
        if key not in self._c:
            sig = self.read(self.counts[i]['on'], readout, unit)
            self._c[key] = H.z_scale(sig, unit_ctrl_rate(unit), self.gsd[unit]['on'], thresh=unit_threshold(unit))
        return self._c[key]

    def Y(self, i: int, cond: str, readout: str, unit: str, s: float, gauss: bool = True) -> np.ndarray:
        """Level i's trials of `cond` with the independent part: counts x exp(s eps - s^2 / 2), read out, times the
        conversion, plus the silent entries' Gaussian noise (in the flies' unit)."""
        x = self.counts[i][cond]
        if s > 0:
            x = x * np.exp(s * self.eps[cond] - 0.5 * s * s)
        y = self.conv(readout, unit, i) * self.read(x, readout, unit)
        if gauss:
            y = y + self.gsd[unit][cond] * self.eta[cond]
        return y

    def gs(self, i: int, cond: str, readout: str, unit: str, s: float) -> Tuple[float, float]:
        key = (i, cond, readout, unit, round(float(s), 10))
        if key not in self._cache:
            self._cache[key] = gap_same(self.Y(i, cond, readout, unit, s))
        return self._cache[key]

    def calibrate(self, readout: str, unit: str, cond: str, tol: float = 1e-4, max_iter: int = 30) -> dict:
        """The jitter (interpolated in sigma^2 between the levels) at which the model's same-repeat gap is the flies' and
        the lognormal spread s at which its same-odor single-trial distance is the flies', with the independent part in
        place for the gap and the jitter in place for the spread, alternated to a fixed point."""
        from scipy.optimize import brentq
        fl = self.flies[unit][GENOTYPE[cond]]
        g_t, d_t = float(fl['gap']), float(fl['same_odor'])
        s, sig_prev, hist = 0.0, None, []
        first = None
        flags = set()
        for it in range(1, max_iter + 1):
            gaps = [self.gs(i, cond, readout, unit, s)[0] for i in range(len(self.levels))]
            ip = interp_sigma(self.sigmas, gaps, g_t)
            if first is None:
                first = ip
            if not ip['bracketed']:
                flags.add('gap_not_bracketed')

            def d_at(x, ip=ip):
                return (1 - ip['w']) * self.gs(ip['lo'], cond, readout, unit, x)[1] + ip['w'] * self.gs(ip['hi'], cond, readout, unit, x)[1]
            d0, d1 = d_at(0.0), d_at(S_MAX)
            if d0 >= d_t:
                s_new = 0.0
                flags.add('same_odor_reached_without_lognormal')
            elif d1 < d_t:
                s_new = S_MAX
                flags.add('s_at_bound')
            else:
                s_new = float(brentq(lambda x: d_at(x) - d_t, 0.0, S_MAX, xtol=1e-6))
            hist.append(dict(iteration=it, sigma=ip['sigma'], w=ip['w'], lo=int(ip['lo']), hi=int(ip['hi']), s=s_new,
                same=d_at(s_new)))
            done = sig_prev is not None and abs(ip['sigma'] - sig_prev) < tol and abs(s_new - s) < tol
            sig_prev, s = ip['sigma'], s_new
            if done:
                break
        gaps = [self.gs(i, cond, readout, unit, s)[0] for i in range(len(self.levels))]
        ip_final = interp_sigma(self.sigmas, gaps, g_t)
        # (interpolating from the deterministic run to a level far above the match leans on the sigma^2 shape alone)
        if ip_final['bracketed'] and self.sigmas[ip_final['lo']] == 0 and self.sigmas[ip_final['hi']] > 1.5 * ip_final['sigma']:
            flags.add('bracket_from_deterministic')
        return dict(model=self.label, name=self.name, readout=readout, unit=unit, condition=cond, sigma=ip_final['sigma'],
            w=ip_final['w'], lo=int(ip_final['lo']), hi=int(ip_final['hi']), sigma_lo=float(self.sigmas[ip_final['lo']]),
            sigma_hi=float(self.sigmas[ip_final['hi']]), bracketed=bool(ip_final['bracketed']), s=s,
            sigma_gap_only=first['sigma'], gap_only_bracketed=bool(first['bracketed']), gap_target=g_t, same_target=d_t,
            gauss_sd=self.gsd[unit][cond], c_lo=self.conv(readout, unit, ip_final['lo']), c_hi=self.conv(readout, unit, ip_final['hi']),
            K=self.K[unit] if readout == 'cal' else None, iterations=len(hist), converged=len(hist) < max_iter,
            flags=sorted(flags), history=hist, levels=[dict(sigma=L['sigma'], source=L['source']) for L in self.levels],
            level_gaps=gaps)

    def final(self, cal: dict) -> Dict[str, object]:
        """Every statistic at a calibration: the interpolation (weight w in sigma^2) between the two bracketing levels of
        the statistic at the calibrated spread; also the jitter alone ('shared_only', s = 0 and no Gaussian noise) and
        the jitter with the Gaussian noise only (s = 0), and the binarized trials."""
        readout, unit, cond, s, w = cal['readout'], cal['unit'], cal['condition'], cal['s'], cal['w']
        thr = unit_threshold(unit)
        parts = []
        for i in (cal['lo'], cal['hi']):
            Y = self.Y(i, cond, readout, unit, s)
            st = summary_stats(kind_arrays(Y))
            stb = summary_stats(kind_arrays((Y > thr).astype(float)))
            sh = summary_stats(kind_arrays(self.Y(i, cond, readout, unit, 0.0, gauss=False)))
            g0 = gap_same(self.Y(i, cond, readout, unit, 0.0))
            st.update(bin_same=stb['same_odor'], bin_gap=stb['gap'], bin_gap_six=stb['gap_six'],
                rate_trial=float((Y > thr).mean()), rate_mean=float((Y.mean(axis=1) > thr).mean()),
                rate_spikes=self.rate_spk[i][cond], same_shared_only=sh['same_odor'], gap_shared_only=sh['gap'],
                same_gauss_only=g0[1], gap_gauss_only=g0[0], pairs_shared_only=sh['pairs_trial_mean'])
            parts.append(st)
        out = {}
        for k in parts[0]:
            a, b = parts[0][k], parts[1][k]
            out[k] = (1 - w) * a + w * b
        return out

    def present(self, readout: str, unit: str, which: str) -> Optional[Dict[str, pd.Series]]:
        """The present pipeline at the matched level `which` ('on' or 'acute_off'): both conditions' three-trial-mean
        distances at that level, no independent part (None if the level is not loaded)."""
        sig = self.matched.get(which)
        if sig is None:
            return None
        hit = [i for i, x in enumerate(self.sigmas) if np.isclose(x, sig, rtol=0, atol=1e-9)]
        if not hit:
            return None
        i = hit[0]
        out = {}
        for c in CONDS:
            k = kind_arrays(self.read(self.counts[i][c], readout, unit))
            with _quiet():
                out[c] = pd.Series(np.nanmean(k['trial_mean'], axis=0), index=PAIR_INDEX)
        return out

    def det_pairs(self, readout: str, unit: str) -> Dict[str, pd.Series]:
        return {c: H.canonical_pairs(H.pair_distances(self.read(self.det[c], readout, unit), 'cosine')) for c in CONDS}


def interp_sigma(sigmas: Sequence[float], values: Sequence[float], target: float) -> Dict[str, object]:
    """The sigma at which the piecewise-linear interpolant in sigma^2 of `values` (one per level, `sigmas` ascending)
    first reaches `target`: {'sigma', 'lo', 'hi' (the bracketing levels' indices), 'w' (the weight of 'hi', linear in
    sigma^2), 'bracketed'}. When no segment brackets the target, the level whose value is nearest it ('lo' == 'hi',
    w 0, bracketed False)."""
    x = np.asarray(sigmas, dtype=float) ** 2
    y = np.asarray(values, dtype=float)
    for k in range(len(x) - 1):
        y0, y1 = y[k], y[k + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == target:
            return dict(sigma=float(np.sqrt(x[k])), lo=k, hi=k, w=0.0, bracketed=True)
        if (y0 - target) * (y1 - target) < 0 or y1 == target:
            w = (target - y0) / (y1 - y0)
            return dict(sigma=float(np.sqrt(x[k] + w * (x[k + 1] - x[k]))), lo=k, hi=k + 1, w=float(w), bracketed=True)
    j = int(np.nanargmin(np.abs(y - target)))
    return dict(sigma=float(np.sqrt(x[j])), lo=j, hi=j, w=0.0, bracketed=False)


# ---------------------------------------------------------------------------
# scoring
def pair_mask(idx: pd.Index, group: Sequence[str]) -> np.ndarray:
    a, b = idx.get_level_values(0), idx.get_level_values(1)
    return np.asarray(a.isin(group) & b.isin(group))


def kind_mean(s: pd.Series, pairs=SIX) -> float:
    s = H.canonical_pairs(s)
    vals = [s.get(p, s.get(p[::-1], np.nan)) for p in pairs]
    return float(np.nanmean(vals))


def score_row(on: pd.Series, off: pd.Series, d_on: pd.Series, d_off: pd.Series) -> Dict[str, float]:
    """A model's pair distances (APL on, APL off) against the flies' (control flies, APL output blocked (TNT)): the
    APL-effect loss with its parts, ρ on the per-pair APL effect, the ester and alcohol groups' pattern errors and their
    errors in each condition, the six pairs, the block, the mean distances."""
    on, off, d_on, d_off = (H.canonical_pairs(x) for x in (on, off, d_on, d_off))
    data_eff = H.apl_effect(d_on, d_off)
    dec, _ = H.apl_effect_decomposition(on, off, data_eff)
    parts = H.apl_effect_pair_parts(on, off, data_eff)
    err = parts['error']
    est, alc = pair_mask(err.index, ESTER_GROUP), pair_mask(err.index, ALCOHOL_GROUP)
    a_on, a_don = H.align_pairs(on, d_on)
    a_off, a_doff = H.align_pairs(off, d_off)
    row = dict(loss=dec['apl_effect_rmse'], pattern=dec['apl_effect_rmse_centered'], mean_effect=dec['apl_effect_mean'],
        mean_error=dec['apl_effect_mean_error'], per_odor=dec['apl_effect_per_odor'], pair_specific=dec['apl_effect_pair_specific'],
        rho=dec['apl_effect_spearman'], data_mean_effect=float(data_eff.mean()),
        err_ester=float(err[est].mean()), err_alcohol=float(err[alc].mean()),
        err_ester_on=float((a_on - a_don)[pair_mask(a_on.index, ESTER_GROUP)].mean()),
        err_ester_off=float((a_off - a_doff)[pair_mask(a_off.index, ESTER_GROUP)].mean()),
        err_alcohol_on=float((a_on - a_don)[pair_mask(a_on.index, ALCOHOL_GROUP)].mean()),
        err_alcohol_off=float((a_off - a_doff)[pair_mask(a_off.index, ALCOHOL_GROUP)].mean()),
        six_on=kind_mean(on), six_off=kind_mean(off), six_effect=kind_mean(on) - kind_mean(off),
        data_six_on=kind_mean(d_on), data_six_off=kind_mean(d_off), data_six_effect=kind_mean(d_on) - kind_mean(d_off),
        mean_on=float(on.mean()), mean_off=float(off.mean()), data_mean_on=float(d_on.mean()), data_mean_off=float(d_off.mean()))
    for k, (x, y) in (('block', (on, off)), ('data_block', (d_on, d_off))):
        try:
            row[k] = float(H.compare_on_off(x, y)[0]['block_fruity_small_iso'])
        except Exception:
            row[k] = np.nan
    return row


def _f(x, spec: str = '.3f') -> str:
    try:
        return '-' if x is None or not np.isfinite(float(x)) else format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


# ---------------------------------------------------------------------------
# part D
def _model_job(job: Tuple[str, str, Tuple[str, ...], dict]) -> dict:
    """One model: every calibration (readout x unit x condition), its final statistics and the deterministic and
    present-pipeline distances. Returns a picklable dict."""
    label, name, exps, flies = job
    t0 = time.time()
    try:
        mm = ModelMix(label, name, exps, flies)
    except FileNotFoundError as e:
        return dict(label=label, name=name, error=str(e))
    cals, finals, det, present = {}, {}, {}, {}
    for ro in READOUTS:
        for u in UNITS:
            for c in CONDS:
                cal = mm.calibrate(ro, u, c)
                fin = mm.final(cal)
                cals[(ro, u, c)] = cal
                finals[(ro, u, c)] = fin
            det[(ro, u)] = mm.det_pairs(ro, u)
            for which in CONDS:
                p = mm.present(ro, u, which)
                if p is not None:
                    present[(ro, u, which)] = p
            print(f'{label} {ro} {u}: sigma {cals[(ro, u, "on")]["sigma"]:.4f} / {cals[(ro, u, "acute_off")]["sigma"]:.4f} '
                  f'(gap only {cals[(ro, u, "on")]["sigma_gap_only"]:.4f} / {cals[(ro, u, "acute_off")]["sigma_gap_only"]:.4f}), '
                  f's {cals[(ro, u, "on")]["s"]:.3f} / {cals[(ro, u, "acute_off")]["s"]:.3f}, binarized same '
                  f'{finals[(ro, u, "on")]["bin_same"]:.3f} / {finals[(ro, u, "acute_off")]["bin_same"]:.3f}, gap '
                  f'{finals[(ro, u, "on")]["bin_gap"]:.4f} / {finals[(ro, u, "acute_off")]["bin_gap"]:.4f} '
                  f'({time.time() - t0:.0f} s)', flush=True)
    return dict(label=label, name=name, exps=list(exps), levels=[dict(sigma=L['sigma'], source=L['source'], dir=L['dir'])
        for L in mm.levels], det_dir=str(mm.det_dir), matched=mm.matched, K=mm.K, cals=cals, finals=finals, det=det,
        present=present, fixed_thr=mm.fixed_thr, rate_spk=mm.rate_spk, seconds=time.time() - t0)


def _flies_pairs(flies: dict) -> Dict[Tuple[str, str], Dict[str, pd.Series]]:
    """{(unit, data kind): {'on': control flies' pairs, 'acute_off': APL output blocked (TNT)'s}}."""
    out = {}
    for u in UNITS:
        for dk in ('trial_mean', 'noise_free', 'corrected'):
            out[(u, dk)] = {c: flies[u][GENOTYPE[c]][f'pairs_{dk}'] for c in CONDS}
    return out


def _model_pairs(res: dict, ro: str, u: str) -> Dict[str, Dict[str, pd.Series]]:
    """{kind: {'on': pairs, 'acute_off': pairs}} of one model result under readout `ro` in unit `u`."""
    f = {c: res['finals'][(ro, u, c)] for c in CONDS}
    out = {'mix': {c: f[c]['pairs_trial_mean'] for c in CONDS}, 'mix_noise_free': {c: f[c]['pairs_noise_free'] for c in CONDS},
           'mix_corrected': {c: f[c]['pairs_corrected'] for c in CONDS}, 'shared_only': {c: f[c]['pairs_shared_only'] for c in CONDS},
           'det': res['det'][(ro, u)]}
    for which, k in (('on', 'present_match_on'), ('acute_off', 'present_match_off')):
        if (ro, u, which) in res['present']:
            out[k] = res['present'][(ro, u, which)]
    return out


def _rates(res: dict, ro: str, u: str) -> Dict[str, float]:
    out = {}
    for c, s in (('on', 'on'), ('acute_off', 'off')):
        f = res['finals'][(ro, u, c)]
        out.update({f'rate_spikes_{s}': f['rate_spikes'], f'rate_trial_{s}': f['rate_trial'], f'rate_mean_{s}': f['rate_mean']})
    return out


def check_present(results: Dict[str, dict]) -> Dict[str, object]:
    """The present-pipeline distances (raw counts, matched levels) against every source's pairs_noise_long.csv ('raw',
    graded): the largest absolute difference per model, level and source (with E81 part C's trials loaded first, the
    comparison with E72's / E75's tables is the check that part C's match_on trials reproduce theirs)."""
    out = {}
    for label, res in results.items():
        if 'error' in res:
            continue
        for exp in res['exps']:
            p = H.results_dir(exp) / 'partB' / 'pairs_noise_long.csv'
            if not p.exists():
                continue
            d = pd.read_csv(p)
            d = d[(d['model'] == res['name']) & (d['transform'] == 'graded')]
            if not len(d):
                continue
            for which, lab in (('on', 'match_on'), ('acute_off', 'match_off')):
                pr = res['present'].get(('raw', 'z', which))
                if pr is None:
                    continue
                dd = d[d['level'].isin([lab, 'match_on+match_off'])]
                if not len(dd):
                    continue
                diffs = []
                for c in CONDS:
                    s = dd[dd['condition'] == c].set_index(['odor_a', 'odor_b'])['raw']
                    a, b = H.align_pairs(H.canonical_pairs(pr[c]), H.canonical_pairs(s))
                    diffs.append(float((a - b).abs().max()))
                out[f'{label} {lab} vs {exp}'] = max(diffs)
    return out


def part_d(out: Path, args) -> None:
    """Part D: every model's calibrations, checks and scores (see the module docstring)."""
    from multiprocessing import Pool
    t0 = datetime.now(timezone.utc)
    flies = {u: flies_side(u) for u in UNITS}
    fpairs = _flies_pairs(flies)
    import E76_data_side_target as E76
    data_checks = dict(
        z_trial_mean_vs_yang=float((H.apl_effect(*[fpairs[('z', 'trial_mean')][c] for c in CONDS]) - H.yang_apl_effect()).abs().max()),
        noise_trial_mean_vs_e76=float((H.apl_effect(*[fpairs[('noise', 'trial_mean')][c] for c in CONDS]) - E76.data_effect('noise')).abs().max()),
        z_same_odor_vs_registry={g: float(flies['z'][g]['same_odor'] - H.data_target(f'{p}_single_trial_floor'))
            for g, p in (('TNT3c', 'ctrl'), ('TNT', 'tnt'))},
        z_silent_sd_vs_harness={g: float(flies['z'][g]['silent_sd'] - H.kc_trial_noise_sd(g)) for g in ('TNT3c', 'TNT')})
    print('flies:', {u: {g: {k: round(float(flies[u][g][k]), 4) for k in ('same_odor', 'gap', 'gap_six', 'bin_same', 'bin_gap',
        'inflation', 'inflation_six', 'silent_sd', 'rate_trial', 'rate_mean')} for g in ('TNT3c', 'TNT')} for u in UNITS}, flush=True)
    print('data checks:', data_checks, flush=True)
    models = {k: (v[0], FAMILY_SOURCES[v[1]]) for k, v in MODELS.items()}
    if not args.no_rereads:
        models.update(reread_models())
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models or v[0] in args.models}
    flies_small = {u: {g: {k: v for k, v in flies[u][g].items() if not k.startswith('pairs_') and k != 'gap_by_recording'}
        for g in flies[u]} for u in UNITS}
    jobs = [(k, n, exps, flies_small) for k, (n, exps) in models.items()]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            res_list = pool.map(_model_job, jobs, chunksize=1)
    else:
        res_list = [_model_job(j) for j in jobs]
    results = {r['label']: r for r in res_list}
    for r in res_list:
        if 'error' in r:
            print(f"{r['label']}: skipped ({r['error']})", flush=True)
    results = {k: v for k, v in results.items() if 'error' not in v}
    # mix_levels.json
    levels_js = {}
    for label, res in results.items():
        levels_js[label] = dict(name=res['name'], sources=res['exps'], levels=res['levels'], det_dir=res['det_dir'],
            matched=res['matched'], K=res['K'], fixed_thr=res['fixed_thr'],
            calibration={f'{ro}/{u}/{c}': {k: v for k, v in res['cals'][(ro, u, c)].items() if k not in ('model', 'name')}
                for ro in READOUTS for u in UNITS for c in CONDS})
    (out / 'mix_levels.json').write_text(json.dumps(dict(models=levels_js, flies={u: {g: {k: v for k, v in flies[u][g].items()
        if not k.startswith('pairs_')} for g in flies[u]} for u in UNITS}), indent=1, default=float) + '\n')
    # mix_checks.csv
    chk = []
    for label, res in results.items():
        for ro in READOUTS:
            for u in UNITS:
                for c in CONDS:
                    cal, fin, fl = res['cals'][(ro, u, c)], res['finals'][(ro, u, c)], flies[u][GENOTYPE[c]]
                    row = dict(model=label, name=res['name'], family=MODELS.get(label, (None, 'reread'))[1], readout=ro, unit=u,
                        condition=c, sigma=cal['sigma'], sigma_gap_only=cal['sigma_gap_only'], bracketed=cal['bracketed'],
                        sigma_lo=cal['sigma_lo'], sigma_hi=cal['sigma_hi'], s=cal['s'], gauss_sd=cal['gauss_sd'],
                        c_lo=cal['c_lo'], c_hi=cal['c_hi'], flags=';'.join(cal['flags']), iterations=cal['iterations'])
                    for k in ('gap', 'same_odor', 'gap_six', 'bin_same', 'bin_gap', 'bin_gap_six', 'inflation', 'inflation_six',
                              'same_odor_adjacent', 'same_rep', 'diff_rep', 'rate_trial', 'rate_mean'):
                        row[k] = fin[k]
                        row[f'flies_{k}'] = fl[k]
                    for k in ('rate_spikes', 'same_shared_only', 'gap_shared_only', 'same_gauss_only', 'gap_gauss_only'):
                        row[k] = fin[k]
                    row['share_same_shared_only'] = fin['same_shared_only'] / fl['same_odor']
                    miss = [abs(fin[k] - fl[k]) / abs(fl[k]) for k in ('bin_same', 'bin_gap')]
                    row['bin_same_miss'], row['bin_gap_miss'] = miss
                    row['bin_check_miss'] = bool(max(miss) > 1 / 3)
                    chk.append(row)
    chk = pd.DataFrame(chk)
    chk.to_csv(out / 'mix_checks.csv', index=False)
    # mix_pairs.csv and mix_scores.csv
    pairs_rows, score_rows = [], []
    for u in UNITS:
        for dk in ('trial_mean', 'noise_free', 'corrected'):
            for c in CONDS:
                s = H.canonical_pairs(fpairs[(u, dk)][c])
                pairs_rows.append(pd.DataFrame({'group': 'data', 'readout': '-', 'unit': u, 'kind': dk, 'condition': c,
                    'odor_a': s.index.get_level_values(0), 'odor_b': s.index.get_level_values(1), 'distance': s.to_numpy()}))
    groups = {k: [k] for k in results}
    for pname, members in POPULATIONS.items():
        if all(m in results for m in members):
            groups[pname] = members
    for ro in READOUTS:
        for u in UNITS:
            mp = {k: _model_pairs(results[k], ro, u) for k in results}
            for gname, members in groups.items():
                for kind in KINDS:
                    if not all(kind in mp[m] for m in members):
                        continue
                    on = pd.concat([H.canonical_pairs(mp[m][kind]['on']) for m in members], axis=1).mean(axis=1)
                    off = pd.concat([H.canonical_pairs(mp[m][kind]['acute_off']) for m in members], axis=1).mean(axis=1)
                    dpairs = fpairs[(u, DATA_KIND[kind])]
                    row = dict(group=gname, members=','.join(members), n_models=len(members), readout=ro, unit=u, kind=kind,
                        data_kind=DATA_KIND[kind], **score_row(on, off, dpairs['on'], dpairs['acute_off']))
                    rates = [_rates(results[m], ro, u) for m in members]
                    row.update({k: float(np.mean([r[k] for r in rates])) for k in rates[0]})
                    if len(members) == 1:
                        cal_on, cal_off = results[members[0]]['cals'][(ro, u, 'on')], results[members[0]]['cals'][(ro, u, 'acute_off')]
                        row.update(sigma_on=cal_on['sigma'], sigma_off=cal_off['sigma'], s_on=cal_on['s'], s_off=cal_off['s'],
                            name=results[members[0]]['name'])
                    score_rows.append(row)
                    for c, s in (('on', on), ('acute_off', off)):
                        pairs_rows.append(pd.DataFrame({'group': gname, 'readout': ro, 'unit': u, 'kind': kind, 'condition': c,
                            'odor_a': s.index.get_level_values(0), 'odor_b': s.index.get_level_values(1), 'distance': s.to_numpy()}))
    scores = pd.DataFrame(score_rows)
    scores.to_csv(out / 'mix_scores.csv', index=False)
    scores[scores['n_models'] > 1].to_csv(out / 'populations.csv', index=False)
    pd.concat(pairs_rows, ignore_index=True).to_csv(out / 'mix_pairs.csv', index=False)
    pres = check_present(results)
    print('present pipeline vs saved pairs_noise_long.csv (largest abs difference):', pres, flush=True)
    show = scores[(scores['kind'].isin(['mix', 'present_match_on', 'det', 'mix_noise_free'])) & (scores['n_models'] > 1)]
    print(show[['group', 'readout', 'unit', 'kind', 'loss', 'pattern', 'mean_effect', 'data_mean_effect', 'rho', 'block']].round(4).to_string(), flush=True)
    print(chk[['model', 'readout', 'unit', 'condition', 'sigma', 'sigma_gap_only', 's', 'bin_same', 'flies_bin_same', 'bin_gap',
        'flies_bin_gap', 'bin_check_miss', 'flags']].round(4).to_string(), flush=True)
    _provenance(out, 'D', dict(started=t0.isoformat(timespec='seconds'), models={k: dict(name=v[0], sources=list(v[1])) for k, v in models.items()},
        populations=POPULATIONS, seed=SEED, s_max=S_MAX, n_recordings=N_REC, n_trials=N_TRIALS, jobs=args.jobs,
        interpolation='linear in sigma^2 between the two levels bracketing the flies\' gap; the deterministic run is level 0',
        conversion="E68's z_scale on each level's APL-on trials (rate under the data's criterion, trial mean of 3 above "
                   "the unit's threshold, through the control flies' silent-entry noise, equal to the control flies')",
        units={u: dict(threshold=unit_threshold(u), ctrl_rate=unit_ctrl_rate(u)) for u in UNITS},
        data_checks=data_checks, present_pipeline_check=pres,
        level_sources={k: v['levels'] for k, v in results.items()}, seconds={k: v['seconds'] for k, v in results.items()}))


# ---------------------------------------------------------------------------
# part D's check: the independent part as a threshold jitter drawn per presentation
PRES_MODELS = {'C2': 'E67_apl_effect_fits:rate0.068_aplboth_k2_cap3',
               'F1': 'E69_kc_firing_ceiling:rate0.068_aplboth_refr3_adapt0.04142t1000_k2'}
# E43's seed, so that the shared draws of the pseudo-recordings are part C's (E43 `b_draws`, kind 'rec', reps 0, 1, ...)
PRES_SEED = 43
PRES_KINDS = ('presentation', 'presentation_noise_free', 'presentation_corrected', 'det')
PRES_DATA_KIND = {'presentation': 'trial_mean', 'presentation_noise_free': 'noise_free',
                  'presentation_corrected': 'corrected', 'det': 'trial_mean'}
# the Gaussian draws of the silent entries: part D's streams are spawn_key (model, condition); these are their own
PRES_ETA_STREAM = {'search': 3, 'final': 2}


def _pres_trials(a: dict, cond: str) -> np.ndarray:
    """A trial set's counts (trials, KCs, odors) as pseudo-recordings of `N_TRIALS` trials (R, T, KCs, odors)."""
    x = np.asarray(a[f'{cond}_spk'], dtype=float)
    if x.shape[0] % N_TRIALS:
        raise ValueError(f'{x.shape[0]} trials are not pseudo-recordings of {N_TRIALS}')
    return x.reshape(x.shape[0] // N_TRIALS, N_TRIALS, *x.shape[1:])


def _pres_eta(name: str, cond: str, which: str, shape) -> np.ndarray:
    ss = np.random.SeedSequence(SEED, spawn_key=(_seed_key(name), CONDS.index(cond), PRES_ETA_STREAM[which]))
    return np.random.default_rng(ss).standard_normal(shape)


def pres_guess(mm: 'ModelMix', cond: str, sigma_s: float, target: float) -> Dict[str, float]:
    """The search's start from part D's saved levels (the shared jitter alone, with the silent entries' Gaussian noise in
    Yang's z, raw counts): a presentation's threshold has relative SD sqrt(sigma_s^2 + sigma_p^2) under the shared jitter
    and under the per-presentation one alike, and the same-odor distance involves single presentations only, so the
    level sigma_tot at which the saved levels' same-odor distance is `target` (interpolated in sigma^2) gives sigma_p =
    sqrt(sigma_tot^2 - sigma_s^2) (at least sigma_s / 4); 'slope', the distance's change per unit sigma^2 there."""
    d = [mm.gs(i, cond, 'raw', 'z', 0.0)[1] for i in range(len(mm.levels))]
    x = np.asarray(mm.sigmas, dtype=float) ** 2
    ip = interp_sigma(mm.sigmas, d, target)
    lo, hi = ip['lo'], ip['hi']
    if not ip['bracketed'] or lo == hi:
        # (not reached: extrapolated from the top segment)
        lo, hi = len(x) - 2, len(x) - 1
    slope = float((d[hi] - d[lo]) / (x[hi] - x[lo]))
    tot2 = ip['sigma'] ** 2 if ip['bracketed'] else x[hi] + (target - d[hi]) / slope
    sp = float(np.sqrt(max(tot2 - sigma_s ** 2, (sigma_s / 4) ** 2)))
    return dict(sigma_tot=float(np.sqrt(max(tot2, 0.0))), sigma_p=sp, slope=slope, bracketed=bool(ip['bracketed']),
        level_same=[float(v) for v in d], level_sigmas=[float(s) for s in mm.sigmas])


def pres_next(hist: List[dict], target: float, slope0: float) -> float:
    """The next per-presentation SD of the search: secant in sigma_p^2 through the two evaluations nearest the target
    (the first step with `slope0`, part D's slope), the step kept within a factor of 2 of the last sigma_p."""
    last = hist[-1]
    pts = sorted(hist, key=lambda h: abs(h['same'] - target))[:2]
    slope = slope0
    if len(pts) == 2 and pts[0]['sigma_p'] != pts[1]['sigma_p']:
        s = (pts[1]['same'] - pts[0]['same']) / (pts[1]['sigma_p'] ** 2 - pts[0]['sigma_p'] ** 2)
        if np.isfinite(s) and s > 0:
            slope = s
    ref = pts[0]
    sp2 = ref['sigma_p'] ** 2 + (target - ref['same']) / slope
    return float(np.sqrt(min(max(sp2, (last['sigma_p'] / 2) ** 2), (2 * last['sigma_p']) ** 2)))


def _pres_tuned_curve(name: str, exps: Sequence[str]) -> Dict[float, float]:
    """{sigma: the mean threshold part B tuned there} (its calibration points, from the first of `exps`'
    noise_levels.json that has them; empty if none), the start of the re-tuning at the matching total SD."""
    for exp in exps:
        p = H.results_dir(exp) / 'partB' / 'noise_levels.json'
        if p.exists():
            js = json.loads(p.read_text())
            if name in js and js[name].get('tuned'):
                return {float(k): float(v) for k, v in js[name]['tuned'].items()}
    return {}


class PresentationRun:
    """One model's part of `--part D-presentation`: E43's pipeline model (built, checked, with its forked workers), the
    search per condition and the pseudo-recordings at the fitted levels. Trial sets are saved to (and read back from)
    `odir`, so an interrupted run resumes."""

    def __init__(self, label: str, out: Path, odir: Path, args, flies: dict, log_rows: list):
        import E43_model_trial_noise as E43
        self.E43, self.label, self.out, self.odir, self.args, self.flies = E43, label, out, odir, args, flies
        self.spec = PRES_MODELS[label]
        self.name = E43.model_from(self.spec)
        if self.name != MODELS[label][0]:
            raise RuntimeError(f'{label}: {self.spec} is {self.name}, part D read {MODELS[label][0]}')
        self.exps = FAMILY_SOURCES[MODELS[label][1]]
        self.mm = ModelMix(label, self.name, self.exps, {u: flies[u] for u in UNITS})
        js = json.loads((out / 'mix_levels.json').read_text())['models'][label]
        self.part_d = {c: js['calibration'][f'raw/z/{c}'] for c in CONDS}
        self.sigma_s = {c: E43.r3(self.part_d[c]['sigma']) for c in CONDS}
        self.gsd = {u: {c: float(flies[u][GENOTYPE[c]]['silent_sd']) for c in CONDS} for u in UNITS}
        self.log_rows = log_rows
        E43.ExcitabilityModel.presentation = True
        E43.ExcitabilityModel.sigma_p = 0.0
        E43.ExcitabilityModel.workers = int(args.workers)
        E43.ExcitabilityModel.thr_min_frac = 0.01
        E43.ExcitabilityModel.thr_jitter = 'normal'
        self.orn = H.apl16_orn_deltas()
        self.checks_path = out / f'per_presentation_build_checks{args.sfx}.json'
        self.prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E81_model_noise.py '
            '--part D-presentation (E43_model_trial_noise.SessionB)', models=[self.name])
        self.sess = E43.SessionB(self.name, self.orn, odir, {}, self.checks_path, self.prov, odir / 'session_provenance.json')
        t0 = time.time()
        self.m = self.sess.get()
        self.build_seconds = time.time() - t0
        self.rate_target = float(self.m.det['on'][0].mean())
        self.f0 = float(self.m.f0)

    def log(self, s: str) -> None:
        print(f'{self.label}: {s}', flush=True)

    def identity_check(self) -> dict:
        """Trial 0 of part C's saved pseudo-recording 0 at the model's match_on level, re-run through the single-odor path
        with sigma_p = 0 (its shared draw from E43's draws, its threshold from the file): equal entry for entry."""
        E43 = self.E43
        sig = self.mm.matched.get('on')
        hit = [L for L in self.mm.levels if L['paths'] and sig is not None and np.isclose(L['sigma'], sig)]
        if not hit:
            return dict(skipped=f'no saved match_on level ({sig}) for {self.name}')
        p = hit[0]['paths'][0]
        with np.load(p) as z:
            saved = {k: z[k] for k in z.files}
        if int(saved['seed']) != PRES_SEED:
            return dict(skipped=f'{p} has seed {int(saved["seed"])}, not {PRES_SEED}')
        xi = E43.b_draws(PRES_SEED, self.name, 'rec', 0, N_TRIALS, self.m.draw_size)[:1]
        t0 = time.time()
        tr = self.m.trials_presentation(float(saved['sigma']), xi, np.zeros((1, len(ODORS), self.m.n_kcs)),
            float(saved['fixed_thr']), sigma_p=0.0)
        res = dict(file=str(p), sigma=float(saved['sigma']), fixed_thr=float(saved['fixed_thr']), seconds=time.time() - t0,
            **{f'{c}_{k}_mismatch': int((tr[f'{c}_{k}'][0] != saved[f'{c}_{k}'][0]).sum()) for c in CONDS for k in ('resp', 'spk')})
        res['reproduced'] = all(v == 0 for k, v in res.items() if k.endswith('_mismatch'))
        self.log(f'part C\'s saved trial through the single-odor path at sigma_p = 0: {res}')
        return res

    def conv(self, on_counts: np.ndarray, readout: str, unit: str) -> float:
        """E68's conversion on APL-on trials (R, T, KCs, odors), as part D's `ModelMix.conv`."""
        return H.z_scale(self.mm.read(on_counts, readout, unit), unit_ctrl_rate(unit), self.gsd[unit]['on'],
            thresh=unit_threshold(unit))

    def evaluate(self, cond: str, sp: float, xis, zetas, f_guess) -> dict:
        """One search point: the mean threshold re-tuned at (sigma_s, sp) on the search draws (APL on), the APL-off runs
        there for `cond` APL off, and the same-odor distance and gap (raw counts converted to Yang's z, Gaussian noise of
        the silent entries in place)."""
        E43, m, a = self.E43, self.m, self.args
        n = len(xis)
        sig = self.sigma_s[cond]
        path = self.odir / f'{self.label}_{cond}_search_s{E43.sd_tag(sig)}_p{E43.sd_tag(sp)}_n{n}.npz'
        t0 = time.time()
        if path.exists():
            arrs = E43.load_npz(path)
            if int(arrs['seed']) != PRES_SEED or float(arrs['rate_target']) != self.rate_target:
                raise RuntimeError(f'{path}: another seed or rate target; move it away')
            fresh = False
        else:
            res = m.tune(sig, xis, self.rate_target, f_guess, a.rate_tol, a.tune_max_evals, zetas=zetas, sigma_p=sp,
                log=lambda s: self.log(f'{cond}, sigma_p {sp:g}:{s}'))
            arrs = {k: v for k, v in res['arrs'].items() if k.startswith('on_') or k in ('n_clipped', 'n_draws_per_trial')}
            if cond == 'acute_off':
                arrs.update({k: v for k, v in m.trials_b(sig, xis, res['fixed_thr'], conds=('acute_off',), zetas=zetas,
                    sigma_p=sp).items() if k.startswith('acute_off_')})
            arrs.update(fixed_thr=np.asarray(res['fixed_thr']), rate_tuned=np.asarray(res['rate']),
                converged=np.asarray(res['converged']), n_evals=np.asarray(res['n_evals']), tune_history=res['history'],
                sigma=np.asarray(sig), sigma_p=np.asarray(sp), seed=np.asarray(PRES_SEED), rate_target=np.asarray(self.rate_target))
            E43.save_npz(path, arrs)
            fresh = True
        on = _pres_trials(arrs, 'on')
        c = self.conv(on, 'raw', 'z')
        X = _pres_trials(arrs, cond)
        Y = c * X + self.gsd['z'][cond] * _pres_eta(self.name, cond, 'search', X.shape)
        gap, same = gap_same(Y)
        row = dict(model=self.label, name=self.name, condition=cond, sigma_s=sig, sigma_p=sp, n_trials=n,
            fixed_thr=float(arrs['fixed_thr']), fixed_thr_ratio=float(arrs['fixed_thr']) / self.f0,
            rate_tuned=float(arrs['rate_tuned']), rate_target=self.rate_target, tune_converged=bool(arrs['converged']),
            tune_evals=int(arrs['n_evals']), conv=c, same=same, gap=gap, same_target=float(self.flies['z'][GENOTYPE[cond]]['same_odor']),
            gap_target=float(self.flies['z'][GENOTYPE[cond]]['gap']), seconds=time.time() - t0 if fresh else np.nan,
            frac_clipped=float(np.asarray(arrs['n_clipped']).mean()) / float(arrs['n_draws_per_trial']), file=path.name)
        self.log(f"{cond}: sigma_s {sig:g}, sigma_p {sp:g}: fixed_thr {row['fixed_thr']:.4f} ({row['fixed_thr_ratio']:.3f} x "
                 f"deterministic), APL-on rate {row['rate_tuned']:.5f} (target {self.rate_target:.5f}); same-odor distance "
                 f"{same:.4f} (flies {row['same_target']:.4f}), same-repeat gap {gap:.4f} (flies {row['gap_target']:.4f})"
                 + (f' ({row["seconds"]:.0f} s)' if fresh else ' (saved)'))
        self.log_rows.append(row)
        return dict(row, arrs=arrs)

    def search(self, cond: str) -> dict:
        """The per-presentation SD at which the same-odor distance is the flies', for condition `cond`."""
        E43, a = self.E43, self.args
        n = N_TRIALS * a.search_recordings
        xis = E43.b_draws(PRES_SEED, self.name, 'tune', 0, n, self.m.draw_size)
        zetas = self.m.p_draws(PRES_SEED, 'tune', 0, n)
        target = float(self.flies['z'][GENOTYPE[cond]]['same_odor'])
        g = pres_guess(self.mm, cond, self.sigma_s[cond], target)
        self.log(f"{cond}: start from part D's levels: total SD {g['sigma_tot']:.4f}, sigma_p {g['sigma_p']:.4f} (shared "
                 f"{self.sigma_s[cond]:g}), slope {g['slope']:.3f} per sigma^2")
        # (the threshold's start: part B's tuned threshold at the same total SD per presentation, which holds the same
        # per-presentation response rate in expectation; after an evaluation, scaled by what the last one needed)
        curve = _pres_tuned_curve(self.name, self.exps)
        sig_s = self.sigma_s[cond]

        def f_part_b(sp_):
            return E43.b_guess(curve, float(np.hypot(sig_s, sp_))) if curve else None
        sp = E43.r3(g['sigma_p'])
        f_g = f_part_b(sp)
        hist: List[dict] = []
        for _ in range(a.search_max):
            ev = self.evaluate(cond, sp, xis, zetas, f_g)
            hist.append(ev)
            if abs(ev['same'] - target) <= a.same_tol or a.smoke:
                break
            new = E43.r3(pres_next(hist, target, g['slope']))
            if any(h['sigma_p'] == new for h in hist):
                break
            f_g = None if f_part_b(sp) is None else f_part_b(new) * ev['fixed_thr'] / f_part_b(sp)
            sp = new
        best = min(hist, key=lambda h: abs(h['same'] - target))
        best = dict(best, converged=abs(best['same'] - target) <= a.same_tol, guess=g, n_evaluations=len(hist))
        return best

    def final(self, cond: str, best: dict) -> Tuple[np.ndarray, float]:
        """`--n-recordings` pseudo-recordings of `cond` at the fitted level (R, T, KCs, odors), and the per-trial response
        rate at two spikes."""
        E43, m = self.E43, self.m
        sig, sp, f = best['sigma_s'], best['sigma_p'], best['fixed_thr']
        X, resp = [], []
        for rep in range(self.args.n_recordings):
            path = self.odir / f'{self.label}_{cond}_final_s{E43.sd_tag(sig)}_p{E43.sd_tag(sp)}_n{N_TRIALS}_rep{rep:02d}.npz'
            if path.exists():
                arrs = E43.load_npz(path)
                if float(arrs['fixed_thr']) != float(f) or int(arrs['seed']) != PRES_SEED:
                    raise RuntimeError(f'{path}: another threshold or seed; move it away')
            else:
                t0 = time.time()
                arrs = m.trials_b(sig, E43.b_draws(PRES_SEED, self.name, 'rec', rep, N_TRIALS, m.draw_size), f, conds=(cond,),
                    zetas=m.p_draws(PRES_SEED, 'rec', rep, N_TRIALS), sigma_p=sp)
                arrs['seed'] = np.asarray(PRES_SEED)
                E43.save_npz(path, arrs)
                self.log(f'{cond}: pseudo-recording {rep} at sigma_s {sig:g}, sigma_p {sp:g} in {time.time() - t0:.0f} s')
            X.append(np.asarray(arrs[f'{cond}_spk'], dtype=float))
            resp.append(np.asarray(arrs[f'{cond}_resp'], dtype=float).mean())
        return np.stack(X), float(np.mean(resp))

    def close(self) -> None:
        self.sess.finish()
        self.m = None
        gc.collect()


def pres_score(run: PresentationRun, best: Dict[str, dict], X: Dict[str, np.ndarray], rate_spk: Dict[str, float],
               fpairs: dict) -> Tuple[List[dict], List[dict]]:
    """Scores and checks of one model's pseudo-recordings at the fitted levels, per readout and unit, as part D's."""
    scores, checks = [], []
    name, mm = run.name, run.mm
    eta = {c: _pres_eta(name, c, 'final', X[c].shape) for c in CONDS}
    mix_chk = pd.read_csv(run.out / 'mix_checks.csv') if (run.out / 'mix_checks.csv').exists() else pd.DataFrame()
    for ro in READOUTS:
        for u in UNITS:
            thr = unit_threshold(u)
            fin, pairs = {}, {}
            for c in CONDS:
                conv = run.conv(_pres_trials(best[c]['arrs'], 'on'), ro, u)
                Y = conv * mm.read(X[c], ro, u) + run.gsd[u][c] * eta[c]
                st = summary_stats(kind_arrays(Y))
                stb = summary_stats(kind_arrays((Y > thr).astype(float)))
                st.update(bin_same=stb['same_odor'], bin_gap=stb['gap'], bin_gap_six=stb['gap_six'], conv=conv,
                    rate_trial=float((Y > thr).mean()), rate_mean=float((Y.mean(axis=1) > thr).mean()), rate_spikes=rate_spk[c])
                fin[c] = st
                fl = run.flies[u][GENOTYPE[c]]
                row = dict(model=run.label, name=name, readout=ro, unit=u, condition=c, sigma_s=best[c]['sigma_s'],
                    sigma_p=best[c]['sigma_p'], sigma_total=float(np.hypot(best[c]['sigma_s'], best[c]['sigma_p'])),
                    shared_share=best[c]['sigma_s'] ** 2 / (best[c]['sigma_s'] ** 2 + best[c]['sigma_p'] ** 2),
                    fixed_thr=best[c]['fixed_thr'], fixed_thr_ratio=best[c]['fixed_thr_ratio'],
                    search_converged=best[c]['converged'], search_same=best[c]['same'], conv=conv)
                for k in ('same_odor', 'gap', 'gap_six', 'bin_same', 'bin_gap', 'bin_gap_six', 'inflation', 'inflation_six',
                          'same_odor_adjacent', 'same_rep', 'diff_rep', 'rate_trial', 'rate_mean'):
                    row[k] = st[k]
                    row[f'flies_{k}'] = fl[k]
                row['rate_spikes'] = rate_spk[c]
                miss = [abs(st[k] - fl[k]) / abs(fl[k]) for k in ('bin_same', 'bin_gap')]
                row['bin_same_miss'], row['bin_gap_miss'] = miss
                row['bin_check_miss'] = bool(max(miss) > 1 / 3)
                if len(mix_chk):
                    mc = mix_chk[(mix_chk['model'] == run.label) & (mix_chk['readout'] == ro) & (mix_chk['unit'] == u) &
                                 (mix_chk['condition'] == c)]
                    if len(mc):
                        for k in ('same_odor', 'gap', 'bin_same', 'bin_gap', 'inflation', 'inflation_six', 'sigma', 's'):
                            row[f'mix_{k}'] = float(mc[k].iloc[0])
                checks.append(row)
            det = mm.det_pairs(ro, u)
            kinds = {'presentation': {c: fin[c]['pairs_trial_mean'] for c in CONDS},
                     'presentation_noise_free': {c: fin[c]['pairs_noise_free'] for c in CONDS},
                     'presentation_corrected': {c: fin[c]['pairs_corrected'] for c in CONDS}, 'det': det}
            for kind, pr in kinds.items():
                d = fpairs[(u, PRES_DATA_KIND[kind])]
                row = dict(group=run.label, members=run.label, n_models=1, name=name, readout=ro, unit=u, kind=kind,
                    data_kind=PRES_DATA_KIND[kind], **score_row(pr['on'], pr['acute_off'], d['on'], d['acute_off']))
                for c, s in (('on', 'on'), ('acute_off', 'off')):
                    row.update({f'rate_spikes_{s}': rate_spk[c], f'rate_trial_{s}': fin[c]['rate_trial'],
                        f'rate_mean_{s}': fin[c]['rate_mean'], f'sigma_s_{s}': best[c]['sigma_s'], f'sigma_p_{s}': best[c]['sigma_p'],
                        f'fixed_thr_{s}': best[c]['fixed_thr']})
                scores.append(row)
    return scores, checks


def part_d_presentation(out: Path, args) -> None:
    """Part D's check (see the module docstring)."""
    t_start = datetime.now(timezone.utc)
    args.sfx = '_smoke' if args.smoke else ''
    odir = H.outputs_dir(EXPERIMENT) / f'partD_presentation{args.sfx}'
    odir.mkdir(parents=True, exist_ok=True)
    flies = {u: flies_side(u) for u in UNITS}
    fpairs = _flies_pairs(flies)
    labels = ['C2'] if args.smoke else [m for m in (args.models or list(PRES_MODELS)) if m in PRES_MODELS]
    if not labels:
        raise ValueError(f'--models: none of {list(PRES_MODELS)}')
    search_rows, scores, checks, info = [], [], [], {}
    for label in labels:
        t0 = time.time()
        run = PresentationRun(label, out, odir, args, flies, search_rows)
        rec = dict(name=run.name, spec=run.spec, build_seconds=run.build_seconds, rate_target=run.rate_target, f0=run.f0,
            sigma_s=run.sigma_s, part_d_sigma={c: run.part_d[c]['sigma'] for c in CONDS}, workers=args.workers)
        try:
            rec['identity'] = run.identity_check()
            if rec['identity'].get('reproduced') is False:
                raise RuntimeError(f'{label}: the single-odor path did not reproduce part C\'s saved trial: {rec["identity"]}')
            best, X, rate_spk = {}, {}, {}
            for c in (['on'] if args.smoke else list(CONDS)):
                t1 = time.time()
                best[c] = run.search(c)
                rec[f'search_seconds_{c}'] = time.time() - t1
                t1 = time.time()
                X[c], rate_spk[c] = run.final(c, best[c])
                rec[f'final_seconds_{c}'] = time.time() - t1
                rec[f'fitted_{c}'] = {k: v for k, v in best[c].items() if k != 'arrs'}
            if not args.smoke:
                sc, ck = pres_score(run, best, X, rate_spk, fpairs)
                scores += sc
                checks += ck
        finally:
            run.close()
        rec['seconds'] = time.time() - t0
        info[label] = rec
        print(f'{label}: done in {rec["seconds"]:.0f} s', flush=True)
    pd.DataFrame(search_rows).to_csv(out / f'per_presentation_search{args.sfx}.csv', index=False)
    if scores:
        sc = pd.DataFrame(scores)
        sc.to_csv(out / 'per_presentation.csv', index=False)
        ck = pd.DataFrame(checks)
        ck.to_csv(out / 'per_presentation_checks.csv', index=False)
        show = sc[(sc['readout'] == 'raw') & (sc['unit'] == 'z')]
        print(show[['group', 'kind', 'loss', 'pattern', 'mean_effect', 'data_mean_effect', 'rho', 'block', 'six_effect',
            'data_six_effect']].round(4).to_string(), flush=True)
        print(ck[(ck['readout'] == 'raw') & (ck['unit'] == 'z')][['model', 'condition', 'sigma_s', 'sigma_p', 'same_odor',
            'flies_same_odor', 'gap', 'flies_gap', 'bin_same', 'flies_bin_same', 'bin_gap', 'flies_bin_gap',
            'bin_check_miss']].round(4).to_string(), flush=True)
    import E43_model_trial_noise as E43
    _provenance(out, f'D_presentation{args.sfx}', dict(started=t_start.isoformat(timespec='seconds'), models=info,
        seed=PRES_SEED, eta_seed=SEED, workers=args.workers, search_recordings=args.search_recordings, search_max=args.search_max,
        same_tol=args.same_tol, n_recordings=args.n_recordings, rate_tol=args.rate_tol, tune_max_evals=args.tune_max_evals,
        smoke=args.smoke, noise=E43.THR_NOISE_NOTE + E43.PRES_NOISE_NOTE, omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
        omp_thread_limit=os.environ.get('OMP_THREAD_LIMIT'),
        model=H.model_provenance(orn_deltas=H.apl16_orn_deltas())))
    if scores:
        part_table(out)


# ---------------------------------------------------------------------------
# part E
E_MODELS = {
    'C2': ('e67cal', ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3'), 'C2, the breadth-2 APL-effect fit (E67)'),
    'F1': ('e67cal', ('E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.04142t1000_k2'),
           "E69's fit 1, the firing ceiling at the literature setting (strength 1)"),
}
E_FACTORS = (0.8, 0.9, 1.1, 1.2)
TRIPLETS = ((0.9, 1.0, 1.1), (0.8, 1.0, 1.2))


def e_points() -> List[Tuple[str, float, str]]:
    """(point id, threshold factor, condition): the base with APL on and off, then each factor of `E_FACTORS` with APL on
    and off (twelve simulations per model, the base's two being its reproduction check)."""
    return [('base_on', 1.0, 'on'), ('base_off', 1.0, 'off')] + [(f'thr{f:g}_{c}', f, c) for f in E_FACTORS for c in ('on', 'off')]


def part_e(out: Path, threads: int = 12, smoke: bool = False) -> None:
    """Part E's simulations: per model of `E_MODELS` (C2 only with `smoke`), E69's fixed-scale wrapper, the base re-run with
    APL on and off checked against its archived spike counts (entry for entry), then each threshold factor with APL on
    and off (`smoke`: only thr0.9_on). Counts and responses to outputs/E81_repeat_structure/partE[_smoke]/; one row per
    point to shared_state_runs[_smoke].csv; checks and the model provenance to provenance_E[_smoke].json."""
    import E69_kc_firing_ceiling as E69
    import E76_data_side_target as E76
    sfx = '_smoke' if smoke else ''
    odir = H.outputs_dir(EXPERIMENT) / f'partE{sfx}'
    odir.mkdir(parents=True, exist_ok=True)
    runs_p = out / f'shared_state_runs{sfx}.csv'
    e76_dir = H.outputs_dir('E76_data_side_target') / 'partE'
    checks, provs = {}, {}
    t_all = time.time()
    for label in (['C2'] if smoke else list(E_MODELS)):
        key, (exp, tag), desc = E_MODELS[label]
        t0 = time.time()
        m = E69.Model(key, ref=(exp, tag))
        provs[label] = H.model_provenance(orn_deltas=m.orn)
        on, off = m.sim(m.p), m.sim(m.off_params())
        saved = E76._archived_counts(exp, tag)
        ck = {}
        for cond, o in (('on', on), ('acute_off', off)):
            a = saved[cond]
            ck[f'{cond}_archived'] = a is not None
            if a is not None:
                ck[f'{cond}_index_equal'] = bool(a.index.equals(o[1].index))
                ck[f'{cond}_mismatch'] = int((a.to_numpy() != o[1].to_numpy(dtype=float)).sum())
        ok = all(ck.get(f'{c}_archived') and ck.get(f'{c}_index_equal') and ck.get(f'{c}_mismatch') == 0 for c in CONDS)
        checks[label] = dict(experiment=exp, tag=tag, reproduced=ok, build_and_check_seconds=time.time() - t0, params=m.p, **ck)
        print(f'{label}: {exp}:{tag} re-run {"reproduces" if ok else "does NOT reproduce"} its archived counts {ck} '
              f'({time.time() - t0:.0f} s)', flush=True)
        if not ok:
            continue
        done = set()
        if runs_p.exists():
            r0 = pd.read_csv(runs_p)
            done = set(r0.loc[r0['model'] == label, 'point'])
        e76_match = {}
        for pid, f, cond in e_points():
            if (smoke and pid not in ('base_on', 'base_off', 'thr0.9_on')) or pid in done:
                continue
            t1 = time.time()
            if pid == 'base_on':
                r, sc = on
                p = dict(m.p)
            elif pid == 'base_off':
                r, sc = off
                p = m.off_params()
            else:
                p = dict(m.p) if cond == 'on' else m.off_params()
                p['fixed_thr'] = p['fixed_thr'] * f
                r, sc = m.sim(p)
            sc = sc.astype(float)
            r.to_parquet(odir / f'{label}_{pid}_responses.parquet')
            sc.to_parquet(odir / f'{label}_{pid}_spike_counts.parquet')
            q = e76_dir / f'c2_{pid}_spike_counts.parquet'
            if label == 'C2' and q.exists():
                e76_match[pid] = int((pd.read_parquet(q).to_numpy(dtype=float) != sc.to_numpy()).sum())
            row = dict(model=label, experiment=exp, tag=tag, point=pid, factor=f, condition=cond,
                rate=float(r.to_numpy(dtype=float).mean()), max_count=float(sc.to_numpy().max()), seconds=time.time() - t1,
                threads=threads, **{k: float(v) for k, v in p.items()})
            pd.DataFrame([row]).to_csv(runs_p, mode='a', header=not runs_p.exists(), index=False)
            print(f"{label} {pid}: rate {row['rate']:.4f}, largest count {row['max_count']:.0f} ({row['seconds']:.0f} s)", flush=True)
        checks[label]['e76_partE_mismatches'] = e76_match
        del m
        gc.collect()
    prev_p = out / f'provenance_E{sfx}.json'
    prev = json.loads(prev_p.read_text()) if prev_p.exists() else {}
    runs = prev.get('runs', []) + [dict(time=datetime.now(timezone.utc).isoformat(timespec='seconds'), smoke=smoke, threads=threads,
        omp_num_threads=os.environ.get('OMP_NUM_THREADS'), omp_thread_limit=os.environ.get('OMP_THREAD_LIMIT'),
        seconds=time.time() - t_all)]
    _provenance(out, f'E{sfx}', dict(checks=dict(prev.get('checks', {}), **checks), runs=runs, factors=E_FACTORS,
        triplets=TRIPLETS, models={k: list(v[1]) for k, v in E_MODELS.items()}, model=dict(prev.get('model', {}), **provs)))


def _e_counts(label: str, pid: str, sfx: str = '') -> Optional[pd.DataFrame]:
    p = H.outputs_dir(EXPERIMENT) / f'partE{sfx}' / f'{label}_{pid}_spike_counts.parquet'
    if not p.exists():
        return None
    return H.abbrev_columns(pd.read_parquet(p).astype(float))[ODORS]


def flies_rundown(unit: str) -> pd.DataFrame:
    """Per genotype, recording and repeat: the response rate (single-trial value above the criterion in `unit`, over
    the KCs and the 16 odors) and the responders' mean amplitude."""
    thr = unit_threshold(unit)
    rows = []
    for g in ('TNT3c', 'TNT'):
        for rec, X in flies_trials(g, unit).items():
            for t in range(X.shape[0]):
                resp = X[t] > thr
                rows.append(dict(genotype=g, recording='_'.join(map(str, rec)), repeat=t, rate=float(resp.mean()),
                    amplitude=float(X[t][resp].mean()) if resp.any() else np.nan))
    return pd.DataFrame(rows)


def _axis_factor(f: np.ndarray, rate: np.ndarray, target_ratio: float, base_rate: float) -> Tuple[float, bool]:
    """The factor along an axis (points `f`, response rates `rate`) at which the rate is `target_ratio` x `base_rate`:
    linear interpolation of log rate in the factor; outside the axis's range, the end segment extended (flagged)."""
    o = np.argsort(f)
    f, lr = np.asarray(f, dtype=float)[o], np.log(np.asarray(rate, dtype=float)[o])
    y = np.log(target_ratio * base_rate)
    for k in range(len(f) - 1):
        if (lr[k] - y) * (lr[k + 1] - y) <= 0 and lr[k] != lr[k + 1]:
            return float(f[k] + (y - lr[k]) * (f[k + 1] - f[k]) / (lr[k + 1] - lr[k])), True
    k = 0 if abs(y - lr[0]) < abs(y - lr[-1]) else len(f) - 2
    slope = (lr[k + 1] - lr[k]) / (f[k + 1] - f[k])
    return float(f[k] + (y - lr[k]) / slope), False


def rundown_axes(out: Path) -> pd.DataFrame:
    """The run-down on the model's axes (rundown_axes.csv): per base and unit of the flies' criterion, for each repeat k
    the threshold factor f_k at which the model's APL-off rate over its base rate is the flies' APL-output-blocked rate
    in repeat k over their session mean; the APL-on rate that factor alone gives; and the factor a_k on APL's output
    (APL->KC and APL->bouton scales) the rest of the control flies' run-down needs, with log rates additive in the two
    axes (a_k > 1, APL stronger than fitted, extrapolated from the block axis's segment 0.75 to 1)."""
    axes = []
    p76 = H.results_dir('E76_data_side_target') / 'model_axes_runs.csv'
    if p76.exists():
        r = pd.read_csv(p76)
        # (a base label can hold runs of two fits, e.g. E76's ceiling2 replaced by --base2: one axis per fit, its last
        # run of each point)
        for (b, exp, tag), g in r.groupby(['base', 'experiment', 'tag'], sort=False):
            g = g.drop_duplicates('point', keep='last')
            if (g['axis'] == 'thr').any():
                axes.append((f'E76:{b}', exp, tag, g[['point', 'axis', 'factor', 'condition', 'rate']].copy()))
    runs_p = out / 'shared_state_runs.csv'
    if runs_p.exists():
        r = pd.read_csv(runs_p)
        for (b, exp, tag), g in r.groupby(['model', 'experiment', 'tag'], sort=False):
            g = g.drop_duplicates('point', keep='last').copy()
            g['axis'] = np.where(g['point'].str.startswith('base'), 'base', 'thr')
            axes.append((f'E81:{b}', exp, tag, g[['point', 'axis', 'factor', 'condition', 'rate']]))
    rows = []
    for unit in UNITS:
        rd = flies_rundown(unit)
        per = rd.groupby(['genotype', 'repeat'])['rate'].mean()
        for base, exp, tag, g in axes:
            if not {'base_on', 'base_off'} <= set(g['point']):
                continue
            base_on = float(g.loc[g['point'] == 'base_on', 'rate'].iloc[0])
            base_off = float(g.loc[g['point'] == 'base_off', 'rate'].iloc[0])
            thr_off = g[(g['axis'].isin(['thr', 'base'])) & (g['condition'] == 'off')]
            thr_on = g[(g['axis'].isin(['thr', 'base'])) & (g['condition'] == 'on')]
            blk = g[(g['axis'].isin(['block'])) | (g['point'].isin(['base_on', 'base_off']))].copy()
            blk['a'] = np.where(blk['point'] == 'base_off', 0.0, np.where(blk['point'] == 'base_on', 1.0, blk['factor']))
            for k in sorted(rd['repeat'].unique()):
                r_on = float(per[('TNT3c', k)] / per.loc['TNT3c'].mean())
                r_off = float(per[('TNT', k)] / per.loc['TNT'].mean())
                f_k, f_in = _axis_factor(thr_off['factor'].to_numpy(), thr_off['rate'].to_numpy(), r_off, base_off)
                o = np.argsort(thr_on['factor'].to_numpy())
                fo, lro = thr_on['factor'].to_numpy()[o], np.log(thr_on['rate'].to_numpy()[o])
                on_from_thr = float(np.exp(np.interp(f_k, fo, lro)) / base_on) if fo.min() <= f_k <= fo.max() else np.nan
                row = dict(unit=unit, base=base, experiment=exp, tag=tag, repeat=k, flies_ratio_on=r_on, flies_ratio_off=r_off,
                    flies_rate_ctrl=float(per[('TNT3c', k)]), flies_rate_tnt=float(per[('TNT', k)]), thr_factor=f_k,
                    thr_factor_in_range=f_in, on_ratio_from_thr=on_from_thr)
                if len(blk) >= 3 and np.isfinite(on_from_thr):
                    need = r_on / on_from_thr
                    bb = blk[blk['a'] > 0]
                    a_k, a_in = _axis_factor(bb['a'].to_numpy(), bb['rate'].to_numpy(), need, base_on)
                    row.update(apl_on_needed_ratio=need, apl_factor=a_k, apl_factor_in_range=a_in)
                rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        first, last = df['repeat'].min(), df['repeat'].max()
        for (u, b, t), g in df.groupby(['unit', 'base', 'tag']):
            g1, g3 = g[g['repeat'] == first].iloc[0], g[g['repeat'] == last].iloc[0]
            m = (df['unit'] == u) & (df['base'] == b) & (df['tag'] == t)
            df.loc[m, 'thr_change_first_to_last'] = g3['thr_factor'] / g1['thr_factor']
            if 'apl_factor' in g:
                df.loc[m, 'apl_change_first_to_last'] = g3.get('apl_factor', np.nan) / g1.get('apl_factor', np.nan)
    df.to_csv(out / 'rundown_axes.csv', index=False)
    return df


def shared_state(out: Path, flies: Optional[dict] = None) -> pd.DataFrame:
    """shared_state.csv: per model, triplet, readout and condition, the triplet's three threshold factors read as the
    three repeats of one recording (all KCs and odors): the same-repeat gap (all pairs, the six pairs), the two-odor
    distances within and across repeats, the same-odor distance across repeats; the off / on ratio of the gap; beside
    the per-KC jitter's (the model's saved trials at its matched APL-on level, the gap with APL off over that with APL
    on) and the flies' (APL output blocked (TNT) over control, Yang's z)."""
    if flies is None:
        flies = {'z': flies_side('z')}
    rows = []
    for label, (key, (exp, tag), desc) in E_MODELS.items():
        base_on = _e_counts(label, 'base_on')
        if base_on is None:
            continue
        import E76_data_side_target as E76
        K = H.calibrate_readout(base_on, 'saturating', E76.cal_target('z'), stat='participation_ratio', responders_min=2.0)
        for trip in TRIPLETS:
            for c, cn in (('on', 'on'), ('off', 'acute_off')):
                mats = []
                for f in trip:
                    pid = f'base_{c}' if f == 1.0 else f'thr{f:g}_{c}'
                    mats.append(_e_counts(label, pid))
                if any(m is None for m in mats):
                    continue
                X = np.stack([m.to_numpy(dtype=float) for m in mats])
                for ro, Y in (('raw', X), ('cal', X / (X + K)), ('binary', (X >= 2).astype(float))):
                    st = summary_stats(kind_arrays(Y[None]))
                    rows.append(dict(model=label, triplet='/'.join(f'{f:g}' for f in trip), readout=ro, condition=cn,
                        gap=st['gap'], gap_six=st['gap_six'], same_rep=st['same_rep'], diff_rep=st['diff_rep'],
                        same_odor=st['same_odor'], trial_mean=float(st['pairs_trial_mean'].mean()),
                        noise_free=float(st['pairs_noise_free'].mean()), rates='/'.join(f'{(m >= 2).to_numpy().mean():.4f}' for m in mats)))
    df = pd.DataFrame(rows)
    if len(df):
        on = df[df['condition'] == 'on'].set_index(['model', 'triplet', 'readout'])
        off = df[df['condition'] == 'acute_off'].set_index(['model', 'triplet', 'readout'])
        ratio = (off['gap'] / on['gap']).rename('gap_ratio_off_on')
        ratio_six = (off['gap_six'] / on['gap_six']).rename('gap_six_ratio_off_on')
        df = df.merge(ratio.reset_index(), on=['model', 'triplet', 'readout'], how='left').merge(ratio_six.reset_index(),
            on=['model', 'triplet', 'readout'], how='left')
    # the per-KC jitter's ratio (saved pipeline trials at the matched APL-on level) and the flies'
    ref = []
    for label, name in (('C2', MODELS['C2'][0]), ('F1', MODELS['F1'][0])):
        exps = FAMILY_SOURCES[MODELS[label][1]]
        lv, _ = model_levels(name, exps)
        mt = matched_levels(name, exps)
        for L in lv:
            if mt and np.isclose(L['sigma'], mt.get('on', np.nan)):
                g = {}
                for c in CONDS:
                    X = []
                    for p in L['paths']:
                        with np.load(p) as z:
                            X.append(z[f'{c}_spk'].astype(float))
                    g[c] = gap_same(np.stack(X))[0]
                ref.append(dict(model=label, triplet=f'per-KC jitter, sigma {L["sigma"]:g} (matched APL on)', readout='raw',
                    condition='both', gap_on=g['on'], gap_off=g['acute_off'], gap_ratio_off_on=g['acute_off'] / g['on']))
                break
    fz = flies['z']
    ref.append(dict(model='flies', triplet="the three repeats (Yang's z)", readout='graded', condition='both',
        gap_on=fz['TNT3c']['gap'], gap_off=fz['TNT']['gap'], gap_ratio_off_on=fz['TNT']['gap'] / fz['TNT3c']['gap']))
    df = pd.concat([df, pd.DataFrame(ref)], ignore_index=True)
    df.to_csv(out / 'shared_state.csv', index=False)
    return df


def part_e_tables(out: Path) -> None:
    flies = {'z': flies_side('z')}
    ss = shared_state(out, flies)
    rd = rundown_axes(out)
    print(ss.round(4).to_string(), flush=True)
    print(rd.round(4).to_string(), flush=True)


# ---------------------------------------------------------------------------
# the table and figures
def _scores_table(sc: pd.DataFrame, groups: Sequence[str], kinds: Sequence[str], readout: str, unit: str) -> List[str]:
    L = ['| group | distances (against the flies\') | APL-effect loss | pattern part | mean APL effect (data) | ρ on the per-pair '
         'APL effect | block (data) | mean distance APL on / off (data) | six pairs: APL effect (data) |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        for k in kinds:
            r = sc[(sc['group'] == g) & (sc['kind'] == k) & (sc['readout'] == readout) & (sc['unit'] == unit)]
            if not len(r):
                continue
            r = r.iloc[0]
            L.append(f"| {g} | {KIND_LABEL[k]} | {_f(r['loss'])} | {_f(r['pattern'])} | {_f(r['mean_effect'], '+.3f')} "
                     f"({_f(r['data_mean_effect'], '+.3f')}) | {_f(r['rho'], '.2f')} | {_f(r['block'])} ({_f(r['data_block'])}) | "
                     f"{_f(r['mean_on'])} / {_f(r['mean_off'])} ({_f(r['data_mean_on'])} / {_f(r['data_mean_off'])}) | "
                     f"{_f(r['six_effect'], '+.3f')} ({_f(r['data_six_effect'], '+.3f')}) |")
    return L


KIND_LABEL = {'mix': 'three-trial means, re-specified noise (as measured)',
    'mix_noise_free': 'noise-free, re-specified noise (noise-free)', 'mix_corrected': 'correction in use, re-specified noise (corrected)',
    'shared_only': 'three-trial means, the jitter alone at the re-specified level (as measured)',
    'det': 'deterministic (as measured)', 'present_match_on': 'present pipeline, level matched on APL on (as measured)',
    'present_match_off': 'present pipeline, level matched on APL off (as measured)'}
READOUT_LABEL = {'raw': 'raw spike counts', 'cal': 'the calibrated readout'}
UNIT_LABEL = {'z': "Yang's z", 'noise': 'baseline-noise units'}


def part_table(out: Path) -> None:
    """model_noise_table.md (parts D and E) and the figures, from the tables."""
    L = ['# E81 parts D and E: the model\'s variability with the flies\' shares, and the shared state', '',
         f'Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} by E81_model_noise.py --part table (tables in this folder).', '']
    chk_p, sc_p = out / 'mix_checks.csv', out / 'mix_scores.csv'
    if chk_p.exists() and sc_p.exists():
        chk, sc = pd.read_csv(chk_p), pd.read_csv(sc_p)
        js = json.loads((out / 'mix_levels.json').read_text())
        fl = js['flies']
        L += ['## Part D: the model\'s variability re-specified with the flies\' shares', '',
              'The flies\' side (computed here from the per-trial values with the same functions as the model\'s; the probe\'s '
              'values in the entry in brackets, Yang\'s z):', '',
              '| statistic | control flies, z | APL output blocked (TNT), z | control flies, baseline-noise units | APL output blocked (TNT), baseline-noise units |',
              '|---|---:|---:|---:|---:|']
        for k, lab in (('same_odor', 'same odor, two repeats (calibration target)'), ('gap', 'same-repeat gap (calibration target)'),
                       ('gap_six', 'same-repeat gap, six pairs'), ('bin_same', 'same odor, two repeats, binarized (check)'),
                       ('bin_gap', 'same-repeat gap, binarized (check)'), ('inflation', 'means of three minus noise-free (check)'),
                       ('inflation_six', 'the same, six pairs (check)'), ('silent_sd', "silent entries' sd per trial (the Gaussian part)"),
                       ('rate_trial', 'response rate, single trials'), ('rate_mean', 'response rate, means of three')):
            pr = PROBE.get(k)
            vals = [fl[u][g][k] for u in UNITS for g in ('TNT3c', 'TNT')]
            cells = [_f(v, '.4f') for v in vals]
            if pr:
                cells[0] += f' [{pr[0]}]'
                cells[1] += f' [{pr[1]}]'
            L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
        L += ['', '### Calibration per model (raw spike counts, Yang\'s z)', '',
              '| model | APL on: jitter (gap only) | spread s | APL off: jitter (gap only) | spread s | levels bracketing on / off | flags |',
              '|---|---:|---:|---:|---:|---|---|']
        for m, g in chk[(chk['readout'] == 'raw') & (chk['unit'] == 'z')].groupby('model', sort=False):
            on, off = g[g['condition'] == 'on'].iloc[0], g[g['condition'] == 'acute_off'].iloc[0]
            fl_ = ';'.join(x for x in {str(on['flags']), str(off['flags'])} if x and x != 'nan')
            L.append(f"| {m} | {_f(on['sigma'], '.4f')} ({_f(on['sigma_gap_only'], '.4f')}) | {_f(on['s'])} | "
                     f"{_f(off['sigma'], '.4f')} ({_f(off['sigma_gap_only'], '.4f')}) | {_f(off['s'])} | "
                     f"{_f(on['sigma_lo'], '.3g')}-{_f(on['sigma_hi'], '.3g')} / {_f(off['sigma_lo'], '.3g')}-{_f(off['sigma_hi'], '.3g')} | {fl_ or '-'} |")
        L += ['', '### Checks not used in the calibration (raw spike counts, Yang\'s z; flies in brackets)', '',
              '| model | condition | same odor, binarized | same-repeat gap, binarized | means of three minus noise-free | six pairs | '
              'share of the same-odor distance from the jitter alone | miss > 1/3 |', '|---|---|---:|---:|---:|---:|---:|---|']
        for _, r in chk[(chk['readout'] == 'raw') & (chk['unit'] == 'z')].iterrows():
            L.append(f"| {r['model']} | {'APL on' if r['condition'] == 'on' else 'APL off'} | {_f(r['bin_same'])} ({_f(r['flies_bin_same'])}) | "
                     f"{_f(r['bin_gap'], '.4f')} ({_f(r['flies_bin_gap'], '.4f')}) | {_f(r['inflation'], '+.3f')} ({_f(r['flies_inflation'], '+.3f')}) | "
                     f"{_f(r['inflation_six'], '+.3f')} ({_f(r['flies_inflation_six'], '+.3f')}) | {_f(r['share_same_shared_only'], '.2f')} | "
                     f"{'yes' if r['bin_check_miss'] else 'no'} |")
        pops = [g for g in POPULATIONS if g in set(sc['group'])]
        for ro in READOUTS:
            for u in UNITS:
                L += ['', f'### Populations and C2: {READOUT_LABEL[ro]}, {UNIT_LABEL[u]}', '']
                L += _scores_table(sc, pops + ['C2', 'F1'], ['present_match_on', 'present_match_off', 'mix', 'shared_only', 'det',
                    'mix_noise_free', 'mix_corrected'], ro, u)
        rr = sorted(g for g in sc['group'].unique() if ':' in g)
        if rr:
            L += ['', '### Re-reads: the block through the noise (raw spike counts, Yang\'s z)', '',
                  '| model | present, matched on APL on | present, matched on APL off | re-specified noise | noise-free | deterministic | data (as measured / noise-free) |',
                  '|---|---:|---:|---:|---:|---:|---|']
            for g in rr:
                s = sc[(sc['group'] == g) & (sc['readout'] == 'raw') & (sc['unit'] == 'z')].set_index('kind')
                get = lambda k: _f(s.loc[k, 'block']) if k in s.index else '-'
                dm = f"{_f(s.loc['mix', 'data_block'])} / {_f(s.loc['mix_noise_free', 'data_block'])}" if 'mix' in s.index else '-'
                L.append(f"| {g} | {get('present_match_on')} | {get('present_match_off')} | {get('mix')} | {get('mix_noise_free')} | {get('det')} | {dm} |")
    L += _presentation_section(out)
    ss_p, rd_p = out / 'shared_state.csv', out / 'rundown_axes.csv'
    if ss_p.exists():
        ss = pd.read_csv(ss_p)
        L += ['', '## Part E: a threshold change common to all KCs as the shared state', '',
              '| model | triplet (threshold factors as three repeats) | readout | gap APL on | gap APL off | off / on |', '|---|---|---|---:|---:|---:|']
        for (m, t, ro), g in ss[ss['condition'] != 'both'].groupby(['model', 'triplet', 'readout'], sort=False):
            on = g[g['condition'] == 'on']
            off = g[g['condition'] == 'acute_off']
            if len(on) and len(off):
                L.append(f"| {m} | {t} | {ro} | {_f(on['gap'].iloc[0], '.4f')} | {_f(off['gap'].iloc[0], '.4f')} | {_f(on['gap_ratio_off_on'].iloc[0], '.2f')} |")
        for _, r in ss[ss['condition'] == 'both'].iterrows():
            L.append(f"| {r['model']} | {r['triplet']} | {r['readout']} | {_f(r['gap_on'], '.4f')} | {_f(r['gap_off'], '.4f')} | {_f(r['gap_ratio_off_on'], '.2f')} |")
    if rd_p.exists():
        rd = pd.read_csv(rd_p)
        L += ['', '## Part E: the run-down on the model\'s axes', '',
              '| unit | base | repeat | flies: rate over session mean, control / APL output blocked (TNT) | threshold factor | APL-on rate from it (over base) | APL strength needed |',
              '|---|---|---:|---|---:|---:|---:|']
        for _, r in rd.iterrows():
            L.append(f"| {r['unit']} | {r['base']} | {int(r['repeat']) + 1} | {_f(r['flies_ratio_on'])} / {_f(r['flies_ratio_off'])} | "
                     f"{_f(r['thr_factor'])}{'' if r['thr_factor_in_range'] else ' (extrapolated)'} | {_f(r['on_ratio_from_thr'])} | "
                     f"{_apl_cell(r)} |")
    (out / 'model_noise_table.md').write_text('\n'.join(L) + '\n')
    figures(out)
    print(f"-> {out / 'model_noise_table.md'}", flush=True)


PRES_KIND_LABEL = {'presentation': 'three-trial means, per-presentation jitter (as measured)',
    'presentation_noise_free': 'noise-free, per-presentation jitter (noise-free)',
    'presentation_corrected': 'correction in use, per-presentation jitter (corrected)'}


def _presentation_section(out: Path) -> List[str]:
    """model_noise_table.md's section on part D's check (per_presentation.csv, per_presentation_checks.csv), with part D's
    lognormal re-specification (mix_scores.csv, mix_checks.csv) beside it."""
    pp, pc = out / 'per_presentation.csv', out / 'per_presentation_checks.csv'
    if not (pp.exists() and pc.exists()):
        return []
    ps, pk = pd.read_csv(pp), pd.read_csv(pc)
    mix = pd.read_csv(out / 'mix_scores.csv') if (out / 'mix_scores.csv').exists() else pd.DataFrame()
    cond_l = {'on': 'APL on', 'acute_off': 'APL off'}
    L = ['', "## Part D's check: the independent part as a threshold jitter drawn per presentation", '',
         "Each presentation's threshold is thr0 (1 + σ_s ξ + σ_p ζ): ξ the KC's draw for the trial, shared by the 16 odors, at "
         "part D's level for the condition; ζ the presentation's own, its SD σ_p fitted so that the same-odor distance between "
         "two single trials is the flies' (raw spike counts converted to Yang's z, the silent entries' Gaussian noise in place, "
         "in place of part D's lognormal factor); each trial assembled from one run per odor; the mean threshold re-tuned at "
         "each level so that the per-trial APL-on response rate is the deterministic run's. APL on is read at the APL-on "
         "level, APL off at the APL-off level. Flies in parentheses, part D's lognormal re-specification in brackets.", '',
         "### Levels (fitted on raw spike counts in Yang's z)", '',
         '| model | condition | shared jitter σ_s (part D) | per-presentation SD σ_p | SD per presentation | share of its variance '
         'shared by the odors | mean threshold over the deterministic run\'s | same-odor distance, search / pseudo-recordings '
         '(flies) | search converged |', '|---|---|---:|---:|---:|---:|---:|---:|---|']
    rz = pk[(pk['readout'] == 'raw') & (pk['unit'] == 'z')]
    for _, r in rz.iterrows():
        L.append(f"| {r['model']} | {cond_l[r['condition']]} | {_f(r['sigma_s'], '.4f')} | {_f(r['sigma_p'], '.4f')} | "
                 f"{_f(r['sigma_total'], '.4f')} | {_f(r['shared_share'], '.2f')} | {_f(r['fixed_thr_ratio'])} | "
                 f"{_f(r['search_same'])} / {_f(r['same_odor'])} ({_f(r['flies_same_odor'])}) | "
                 f"{'yes' if bool(r['search_converged']) else 'no'} |")

    def mx(r, k, spec='.3f'):
        return f" [{_f(r[f'mix_{k}'], spec)}]" if f'mix_{k}' in r and pd.notna(r.get(f'mix_{k}')) else ''
    for ro in READOUTS:
        for u in UNITS:
            sub = pk[(pk['readout'] == ro) & (pk['unit'] == u)]
            if not len(sub):
                continue
            L += ['', f'### Checks not used in the fit: {READOUT_LABEL[ro]}, {UNIT_LABEL[u]}', '',
                  '| model | condition | same odor (graded) | same-repeat gap (graded) | same odor, binarized | same-repeat gap, '
                  'binarized | means of three minus noise-free | the same, six pairs | response rate: single trials / means of '
                  'three | binarized miss > 1/3 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
            for _, r in sub.iterrows():
                L.append(f"| {r['model']} | {cond_l[r['condition']]} | {_f(r['same_odor'])} ({_f(r['flies_same_odor'])}){mx(r, 'same_odor')} | "
                         f"{_f(r['gap'], '.4f')} ({_f(r['flies_gap'], '.4f')}){mx(r, 'gap', '.4f')} | "
                         f"{_f(r['bin_same'])} ({_f(r['flies_bin_same'])}){mx(r, 'bin_same')} | "
                         f"{_f(r['bin_gap'], '.4f')} ({_f(r['flies_bin_gap'], '.4f')}){mx(r, 'bin_gap', '.4f')} | "
                         f"{_f(r['inflation'], '+.3f')} ({_f(r['flies_inflation'], '+.3f')}){mx(r, 'inflation', '+.3f')} | "
                         f"{_f(r['inflation_six'], '+.3f')} ({_f(r['flies_inflation_six'], '+.3f')}){mx(r, 'inflation_six', '+.3f')} | "
                         f"{_f(r['rate_trial'], '.4f')} / {_f(r['rate_mean'], '.4f')} ({_f(r['flies_rate_trial'], '.4f')} / "
                         f"{_f(r['flies_rate_mean'], '.4f')}) | {'yes' if bool(r['bin_check_miss']) else 'no'} |")
    for ro in READOUTS:
        for u in UNITS:
            L += ['', f'### Scores: {READOUT_LABEL[ro]}, {UNIT_LABEL[u]}', '',
                  '| model | distances (against the flies\') | APL-effect loss | pattern part | mean APL effect (data) | ρ on the '
                  'per-pair APL effect | block (data) | mean distance APL on / off (data) | six pairs: APL effect (data) | '
                  'response rate APL on / off, single trials |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for g in [x for x in PRES_MODELS if x in set(ps['group'])]:
                rows = []
                for k in ('presentation', 'presentation_noise_free', 'presentation_corrected'):
                    r = ps[(ps['group'] == g) & (ps['kind'] == k) & (ps['readout'] == ro) & (ps['unit'] == u)]
                    if len(r):
                        rows.append((PRES_KIND_LABEL[k], r.iloc[0]))
                if len(mix):
                    for k in ('mix', 'mix_noise_free', 'present_match_on', 'det'):
                        r = mix[(mix['group'] == g) & (mix['kind'] == k) & (mix['readout'] == ro) & (mix['unit'] == u)]
                        if len(r):
                            rows.append((f'part D: {KIND_LABEL[k]}', r.iloc[0]))
                for lab, r in rows:
                    L.append(f"| {g} | {lab} | {_f(r['loss'])} | {_f(r['pattern'])} | {_f(r['mean_effect'], '+.3f')} "
                             f"({_f(r['data_mean_effect'], '+.3f')}) | {_f(r['rho'], '.2f')} | {_f(r['block'])} ({_f(r['data_block'])}) | "
                             f"{_f(r['mean_on'])} / {_f(r['mean_off'])} ({_f(r['data_mean_on'])} / {_f(r['data_mean_off'])}) | "
                             f"{_f(r['six_effect'], '+.3f')} ({_f(r['data_six_effect'], '+.3f')}) | "
                             f"{_f(r.get('rate_trial_on'), '.4f')} / {_f(r.get('rate_trial_off'), '.4f')} |")
    return L


def _apl_cell(r: pd.Series) -> str:
    a = r.get('apl_factor', np.nan)
    if a is None or not np.isfinite(float(a)):
        return '-'
    return _f(a) + ('' if bool(r.get('apl_factor_in_range', False)) else ' (extrapolated)')


def figures(out: Path) -> None:
    """E81_mix_calibration.png (per model, the same-repeat gap against the jitter at the saved levels, without and with
    the independent part, and the flies' gap); E81_mix_scores.png (the populations' APL-effect loss and its pattern part
    under the present pipeline and the re-specified noise); E81_shared_state.png (part E's gaps)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, ink2, grid = '#0b0b0b', '#52514e', '#e6e5e1'
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#8a5cd0']
    lv_p = out / 'mix_levels.json'
    if lv_p.exists():
        js = json.loads(lv_p.read_text())
        models = [m for m in MODELS if m in js['models']]
        if models:
            fig, axes = plt.subplots(2, len(models), figsize=(2.6 * len(models), 5.2), squeeze=False, sharey='row')
            for j, m in enumerate(models):
                for i, c in enumerate(CONDS):
                    ax = axes[i, j]
                    cal = js['models'][m]['calibration'][f'raw/z/{c}']
                    sig = np.array([L['sigma'] for L in cal['levels']])
                    gp = np.array(cal['level_gaps'], dtype=float)
                    # the interpolant (linear in sigma^2 between levels), so the matched jitter sits on its crossing
                    xs = np.linspace(0, sig.max(), 200)
                    ax.plot(xs, np.interp(xs ** 2, sig ** 2, gp), '-', color=cols[0], lw=1.2,
                            label='with the independent part\n(interpolated in sigma^2)')
                    ax.plot(sig, gp, 'o', color=cols[0], ms=3)
                    ax.axhline(cal['gap_target'], color=ink2, ls=':', lw=1, label="flies' gap")
                    ax.axvline(cal['sigma'], color=cols[1], lw=1, label='matched jitter')
                    ax.set_title(f"{m}, {'APL on' if c == 'on' else 'APL off'}", fontsize=8, color=ink)
                    ax.grid(color=grid, lw=0.5)
                    ax.tick_params(labelsize=7)
                    if j == 0:
                        ax.set_ylabel('same-repeat gap\n(cosine distance)', fontsize=8)
                    if i == 1:
                        ax.set_xlabel('threshold jitter (sigma)', fontsize=8)
            axes[0, 0].legend(fontsize=6, frameon=False)
            fig.suptitle("The jitter at which the model's same-repeat gap is the flies' (raw spike counts, Yang's z)", fontsize=9)
            fig.tight_layout()
            fig.savefig(out / 'E81_mix_calibration.png', dpi=150)
            plt.close(fig)
    sc_p = out / 'mix_scores.csv'
    if sc_p.exists():
        sc = pd.read_csv(sc_p)
        pops = [g for g in POPULATIONS if g in set(sc['group'])]
        kinds = [k for k in ('present_match_on', 'mix', 'det') if k in set(sc['kind'])]
        if pops and kinds:
            fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
            for ax, (ro, u) in zip(axes, [(ro, u) for ro in READOUTS for u in UNITS]):
                x = np.arange(len(pops))
                wdt = 0.8 / len(kinds)
                for k_i, k in enumerate(kinds):
                    vals = [sc[(sc['group'] == g) & (sc['kind'] == k) & (sc['readout'] == ro) & (sc['unit'] == u)]['loss'] for g in pops]
                    vals = [float(v.iloc[0]) if len(v) else np.nan for v in vals]
                    pat = [sc[(sc['group'] == g) & (sc['kind'] == k) & (sc['readout'] == ro) & (sc['unit'] == u)]['pattern'] for g in pops]
                    pat = [float(v.iloc[0]) if len(v) else np.nan for v in pat]
                    ax.bar(x + (k_i - (len(kinds) - 1) / 2) * wdt, vals, wdt * 0.95, color=cols[k_i], label=KIND_LABEL[k].split(' (')[0])
                    ax.scatter(x + (k_i - (len(kinds) - 1) / 2) * wdt, pat, s=10, color=ink, zorder=3, label='pattern part' if k_i == 0 else None)
                ax.axhline(H.data_target('apl_effect_rmse_floor'), color=ink2, ls=':', lw=1, label="the data's floor")
                ax.set_xticks(x)
                ax.set_xticklabels(pops, fontsize=7, rotation=15)
                ax.set_title(f'{READOUT_LABEL[ro]}, {UNIT_LABEL[u]}', fontsize=8)
                ax.grid(axis='y', color=grid, lw=0.5)
                ax.tick_params(labelsize=7)
            axes[0].set_ylabel('APL-effect loss (bars), pattern part (dots)', fontsize=8)
            axes[0].legend(fontsize=6, frameon=False)
            fig.tight_layout()
            fig.savefig(out / 'E81_mix_scores.png', dpi=150)
            plt.close(fig)
    ss_p = out / 'shared_state.csv'
    if ss_p.exists():
        ss = pd.read_csv(ss_p)
        s2 = ss[(ss['condition'] != 'both') & (ss['readout'] == 'raw')]
        if len(s2):
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            labels, on, off = [], [], []
            for (m, t), g in s2.groupby(['model', 'triplet'], sort=False):
                labels.append(f'{m}\n{t}')
                on.append(float(g.loc[g['condition'] == 'on', 'gap'].iloc[0]) if (g['condition'] == 'on').any() else np.nan)
                off.append(float(g.loc[g['condition'] == 'acute_off', 'gap'].iloc[0]) if (g['condition'] == 'acute_off').any() else np.nan)
            for _, r in ss[ss['condition'] == 'both'].iterrows():
                labels.append(f"{r['model']}\n{str(r['triplet'])[:22]}")
                on.append(r['gap_on'])
                off.append(r['gap_off'])
            x = np.arange(len(labels))
            ax.bar(x - 0.2, on, 0.38, color=cols[0], label='APL on / control flies')
            ax.bar(x + 0.2, off, 0.38, color=cols[1], label='APL off / APL output blocked (TNT)')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=6)
            ax.set_ylabel('same-repeat gap (cosine distance)', fontsize=8)
            ax.grid(axis='y', color=grid, lw=0.5)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, frameon=False)
            fig.tight_layout()
            fig.savefig(out / 'E81_shared_state.png', dpi=150)
            plt.close(fig)


# ---------------------------------------------------------------------------
def _provenance(out: Path, part: str, extra: dict) -> None:
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', Path(__file__).name, 'kc_remapping_harness.py',
        'E43_model_trial_noise.py', 'E69_kc_firing_ceiling.py', 'E76_data_side_target.py'], cwd=H.HERE, capture_output=True,
        text=True).stdout.strip())
    prov = dict(script=f'{Path(__file__).name} --part {part}', generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        host=subprocess.run(['hostname', '-s'], capture_output=True, text=True).stdout.strip(), al_analysis_commit=commit,
        dirty=dirty, registry=str(H.data_targets_path()), **extra)
    (out / f'provenance_{part}.json').write_text(json.dumps(prov, indent=1, ensure_ascii=False, default=str) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--part', required=True, choices=('D', 'D-presentation', 'E', 'E-tables', 'table'))
    ap.add_argument('--threads', type=int, default=12,
        help='parts E and D-presentation: OMP threads (OMP_NUM_THREADS and OMP_THREAD_LIMIT must match)')
    ap.add_argument('--smoke', action='store_true', help='part E: C2 only, its reproduction check and one point (thr0.9_on); '
        'part D-presentation: C2, the checks, one search evaluation and one pseudo-recording with APL on, timed')
    ap.add_argument('--jobs', type=int, default=2, help='part D: models in parallel (one process each)')
    ap.add_argument('--models', nargs='*', default=None, help='parts D and D-presentation: only these models (labels or, part '
        'D, pipeline names)')
    ap.add_argument('--workers', type=int, default=10,
        help='part D-presentation: processes forked from this one for the single-odor runs (one OpenMP thread each)')
    ap.add_argument('--search-recordings', type=int, default=2,
        help='part D-presentation: pseudo-recordings of three trials the per-presentation SD is fitted on')
    ap.add_argument('--search-max', type=int, default=4, help='part D-presentation: search evaluations per condition, at most')
    ap.add_argument('--same-tol', type=float, default=0.004,
        help="part D-presentation: the search stops within this of the flies' same-odor distance")
    ap.add_argument('--n-recordings', type=int, default=3,
        help='part D-presentation: pseudo-recordings of three trials at the fitted levels')
    ap.add_argument('--rate-tol', type=float, default=0.01,
        help='part D-presentation: relative tolerance of the re-tuned APL-on response rate (E43\'s)')
    ap.add_argument('--tune-max-evals', type=int, default=16, help='part D-presentation: evaluations per re-tuning, at most')
    ap.add_argument('--no-rereads', action='store_true', help="part D: only the eight models of part C, not the re-reads")
    ap.add_argument('--out', type=Path, default=None, help='write the tables here instead of results/E81_repeat_structure (tests)')
    args = ap.parse_args()
    out = args.out or H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part in ('E', 'D-presentation'):
        for v in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT'):
            if os.environ.get(v) != str(args.threads):
                ap.error(f'part {args.part}: run with OMP_NUM_THREADS={args.threads} OMP_THREAD_LIMIT={args.threads} in the '
                         f'environment (importing mb_model resets OpenMP to all cores otherwise)')
    if args.part == 'E':
        part_e(out, args.threads, args.smoke)
    elif args.part == 'D-presentation':
        if args.smoke:
            args.search_recordings, args.n_recordings = 1, 1
        part_d_presentation(out, args)
    elif args.part == 'D':
        part_d(out, args)
    elif args.part == 'E-tables':
        part_e_tables(out)
    else:
        part_table(out)


if __name__ == '__main__':
    main()
