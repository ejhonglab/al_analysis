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
subsets_vs_archive.csv, common_error_subsets.csv (and _summary), subset_floors.csv, recording_subsets.csv,
split_halves.csv (part D); data_units.csv, effect_by_unit_pairs.csv, class_split_units.csv (part B);
by_variant_units.csv, common_error_units.csv (and _summary), rank_correlations_units.csv, best_by_family_units.csv,
pipeline_units.csv (part C); model_axes_runs.csv and model_axes_pairs.csv (part E's runs), model_axes.csv,
populations.csv, recording_ranges.csv (`--part E-tables`); comparison_table.md (`--part table`); provenance_<part>.json.
The check of E10_joint_fit.py's --apl-data-units (`--part check-e10`) writes outputs/E76_data_side_target/e10_check/.
Usage (from this folder): ../../venv/bin/python -u E76_data_side_target.py --part D | B | C [--jobs 3]
       OMP_NUM_THREADS=12 OMP_THREAD_LIMIT=12 nice -n 10 ../../venv/bin/python -u E76_data_side_target.py --part E --threads 12
           [--bases c2 ceiling2] [--base2 EXP:TAG] [--only POINT ...] [--check-only]
       ../../venv/bin/python -u E76_data_side_target.py --part check-e10 [--check off noise]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
from datetime import datetime, timezone
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


_CACHE: Dict[tuple, object] = {}
# the denominators' floors of the F-normalized units (raw fluorescence units), set when the matrices are built
FLOORS: Dict[str, float] = {}
UNIT_LABEL = {'z': "Yang's z (the target as defined)",
    'noise': 'baseline-noise units (z times whole-trace SD over baseline-noise SD)',
    'dfc': 'the raw change in Fc (z times whole-trace SD)',
    'dff_fmean': "the change in Fc over the KC's mean F",
    'dff_fcmean': "the change in Fc over the KC's mean Fc (denominator floored at its 5th percentile)",
    'dff_fctrial': "each trial's change in Fc over its pre-odor Fc (floored at its 5th percentile), mean over trials"}
# the readings of part D's tests: graded cosine distances in z and in baseline-noise units, and each recording's cosine
# corrected for trial-to-trial noise (noise_floor_trial_distances.analyse) in the same two units
READINGS = ('z', 'noise', 'nc_z', 'nc_noise')
READING_LABEL = {'z': "Yang's z", 'noise': 'baseline-noise units', 'nc_z': "noise-corrected, Yang's z",
    'nc_noise': 'noise-corrected, baseline-noise units'}
# the control recordings' label groups (notes/yang-summary.md, from Yang's slide 28 by date): TNT-nolabel (TeTx genotype,
# APL not expressing it), TNTin-nolabel and TNTin-label; for 2024-07-10 fly 2 and 2024-08-09 fly 3 one side is TNTin-nolabel
# and the other TNTin-label, and which is which is not known
LABEL_GROUP = {('2024-08-02', 5, 'R'): 'TNT-nolabel', ('2024-08-30', 3, 'R'): 'TNT-nolabel',
    ('2024-08-14', 1, 'L'): 'TNTin-nolabel', ('2024-07-12', 2, 'L'): 'TNTin-label', ('2024-07-26', 4, 'R'): 'TNTin-label',
    ('2024-08-01', 3, 'L'): 'TNTin-label', ('2024-08-01', 3, 'R'): 'TNTin-label'}
LABEL_AMBIGUOUS = 'TNTin, label or nolabel (side not known)'


def _rec(rec) -> tuple:
    return (str(rec[0]), int(rec[1]), str(rec[2]))


def rec_label(rec) -> str:
    return f'{rec[0]} fly {rec[1]} {rec[2]}'


def _long_to_matrices(df: pd.DataFrame, col: str) -> Dict[str, Dict[tuple, pd.DataFrame]]:
    df = df[df['odor'].isin(H.APL16_ODORS)]
    out: Dict[str, Dict[tuple, pd.DataFrame]] = {}
    for g in GENOTYPES:
        out[g] = {}
        for rec, sub in df[df['genotype'] == g].groupby(H.RECORDING_KEYS, sort=True):
            m = sub.pivot(index='roi', columns='odor', values=col)[list(H.APL16_ODORS)].astype(float)
            assert not m.isna().any().any(), (g, rec)
            out[g][_rec(rec)] = m
    return out


def _noise_file(name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    key = ('file', name, tuple(columns or ()))
    if key not in _CACHE:
        _CACHE[key] = pd.read_parquet(NOISE_DIR / name, columns=columns)
    return _CACHE[key]


def _per_kc(mats: Dict[str, Dict[tuple, pd.DataFrame]], s: pd.Series, divide: bool) -> Dict[str, Dict[tuple, pd.DataFrame]]:
    out: Dict[str, Dict[tuple, pd.DataFrame]] = {}
    for g, recs in mats.items():
        out[g] = {}
        for rec, m in recs.items():
            v = s.loc[(g, *rec)].reindex(m.index)
            assert not v.isna().any(), (g, rec)
            out[g][rec] = m.div(v.to_numpy(), axis=0) if divide else m.mul(v.to_numpy(), axis=0)
    return out


def unit_matrices(unit: str) -> Dict[str, Dict[tuple, pd.DataFrame]]:
    """{genotype: {(date, fly_num, hemibrain): KCs x the 16 odors (APL16_ODORS order) trial means in `unit`}}, recordings
    in sorted order, KCs indexed by olfdb's roi (as `data_targets.kc_response_matrices`)."""
    import data_targets as DT
    key = ('mats', unit)
    if key in _CACHE:
        return _CACHE[key]
    if unit == 'z':
        out = {g: {_rec(k): m.astype(float) for k, m in recs.items()} for g, recs in DT.kc_response_matrices().items()}
    elif unit == 'noise':
        out = _long_to_matrices(_noise_file('trialmean_long.parquet'), 'value')
    elif unit == 'dfc':
        out = _long_to_matrices(_noise_file('trialmean_long_unnormalized.parquet'), 'value')
    elif unit in ('dff_fmean', 'dff_fcmean'):
        ks = DT.kc_scales_table()
        d = ks['F_mean'].astype(float)
        if unit == 'dff_fcmean':
            d = d - DT.NEUROPIL_COEF * ks['Fneu_mean'].astype(float)
        FLOORS[unit] = float(d[d > 0].quantile(DENOM_FLOOR_PCT / 100.0))
        out = _per_kc(unit_matrices('dfc'), d.clip(lower=FLOORS[unit]), divide=True)
    elif unit == 'dff_fctrial':
        t = _noise_file('per_trial_long_unnormalized.parquet', ['genotype', 'date', 'fly_num', 'hemibrain', 'roi', 'odor',
            'repeat', 'value', 'fc_baseline']).copy()
        t['odor'] = t['odor'].map(H.odor_abbrev)
        t = t[t['odor'].isin(H.APL16_ODORS)]
        b = t['fc_baseline']
        FLOORS[unit] = float(b[b > 0].quantile(DENOM_FLOOR_PCT / 100.0))
        t['dff'] = t['value'] / b.clip(lower=FLOORS[unit])
        tm = t.groupby(['genotype'] + H.RECORDING_KEYS + ['roi', 'odor'], sort=True)['dff'].mean().reset_index()
        out = _long_to_matrices(tm, 'dff')
    else:
        raise ValueError(f'{unit=} not in {UNITS}')
    _CACHE[key] = out
    return out


def _rate_threshold(mats: Dict[str, Dict[tuple, pd.DataFrame]], target: float) -> float:
    """The threshold at which the control flies' mean response rate over recordings comes down to `target`: geometric
    bisection on [1e-4, 1e5] (wide enough for raw fluorescence units), 80 steps."""
    lo, hi = 1e-4, 1e5
    arrs = [m.to_numpy(dtype=float) for m in mats['TNT3c'].values()]
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        r = np.mean([float((a > mid).mean()) for a in arrs])
        lo, hi = (mid, hi) if r > target else (lo, mid)
    return float(np.sqrt(lo * hi))


def unit_threshold(unit: str) -> float:
    """The response criterion in `unit`: 1 in z, the registry's noise_threshold in baseline-noise units, else the
    threshold that keeps the control flies' mean response rate at the one under Yang's criterion."""
    key = ('thr', unit)
    if key not in _CACHE:
        if unit == 'z':
            v = H.KC_RESPONSE_Z
        elif unit == 'noise':
            v = H.data_target('noise_threshold')
        else:
            target = float(np.mean([float((m.to_numpy() > 1.0).mean()) for m in unit_matrices('z')['TNT3c'].values()]))
            v = _rate_threshold(unit_matrices(unit), target)
        _CACHE[key] = float(v)
    return _CACHE[key]


def unit_recording_rdms(unit: str, values: str = 'graded') -> Dict[str, pd.DataFrame]:
    """{genotype: per-recording cosine pair distances (rows recordings, columns canonical pairs)} in `unit`; `values`
    'graded', 'thresholded' (sub-threshold entries set to 0) or 'binary', at `unit_threshold(unit)`."""
    key = ('rdms', unit, values)
    if key not in _CACHE:
        thr = unit_threshold(unit)
        _CACHE[key] = {g: H.recording_rdms_from_matrices(m, values=values, thresh=thr) for g, m in unit_matrices(unit).items()}
    return _CACHE[key]


def _trial_arrays_noise(genotype: str) -> Dict[tuple, Dict[str, np.ndarray]]:
    """{recording: {odor: trials x KCs}} in baseline-noise units, from hal's per-trial file (KCs in sorted roi order, as
    noise_floor_trial_distances.recording_arrays)."""
    t = _noise_file('per_trial_long.parquet', ['genotype', 'date', 'fly_num', 'hemibrain', 'roi', 'odor', 'repeat', 'value'])
    t = t[t['genotype'] == genotype].copy()
    t['abbrev'] = t['odor'].map(H.odor_abbrev)
    t = t[t['abbrev'].isin(H.APL16_ODORS)]
    out = {}
    for rec, sub in t.groupby(H.RECORDING_KEYS, sort=True):
        rois = np.array(sorted(sub['roi'].unique()))
        by_odor = {}
        for o, so in sub.groupby('abbrev'):
            w = so.pivot(index='repeat', columns='roi', values='value').reindex(columns=rois)
            assert not w.isna().any().any(), (rec, o)
            by_odor[o] = w.to_numpy(dtype=float)
        out[_rec(rec)] = by_odor
    return out


def reading_rdms(reading: str) -> Dict[str, pd.DataFrame]:
    """{genotype: per-recording pair distances} in one of part D's `READINGS`."""
    key = ('reading', reading)
    if key not in _CACHE:
        if reading in ('z', 'noise'):
            v = unit_recording_rdms(reading)
        elif reading == 'nc_z':
            v = {g: H.yang_recording_rdms_values(g, 'noise_corrected') for g in GENOTYPES}
        elif reading == 'nc_noise':
            from noise_floor_trial_distances import analyse
            v = {}
            for g in GENOTYPES:
                rows = {rec: H.canonical_pairs(analyse(by_odor, 'graded')[1]['corrected'])
                    for rec, by_odor in _trial_arrays_noise(g).items()}
                v[g] = H._stack_recordings(rows)
        else:
            raise ValueError(f'{reading=} not in {READINGS}')
        # one index type for every reading (date str, fly_num int, hemibrain str), canonical pair columns
        for g in v:
            v[g] = v[g].copy()
            v[g].index = pd.MultiIndex.from_tuples([_rec(r) for r in v[g].index], names=H.RECORDING_KEYS)
        _CACHE[key] = v
    return _CACHE[key]


def pair_mask(idx: pd.Index, group: Sequence[str]) -> np.ndarray:
    a, b = idx.get_level_values(0), idx.get_level_values(1)
    return np.asarray(a.isin(group) & b.isin(group))


def group_effects(effect: pd.Series) -> Dict[str, float]:
    """The ester and alcohol groups' mean effect relative to the mean over all pairs (`ester_rel`, `alcohol_rel`), and
    their absolute means (`ester`, `alcohol`)."""
    e = effect.dropna()
    est, alc = e[pair_mask(e.index, ESTER_GROUP)].mean(), e[pair_mask(e.index, ALCOHOL_GROUP)].mean()
    return dict(ester=float(est), alcohol=float(alc), ester_rel=float(est - e.mean()), alcohol_rel=float(alc - e.mean()))


def strongest_recordings(tnt: pd.DataFrame, n: int = 2) -> List[tuple]:
    """The `n` recordings with APL output blocked (TNT) with the lowest mean distance in Yang's z (per-recording pair
    distances `tnt`)."""
    return [_rec(r) for r in tnt.mean(axis=1).sort_values().index[:n]]


def _std(x: pd.Series) -> float:
    # the sd over pairs as the pattern part counts it (ddof 0): the pattern part of an effect with the right mean and no
    # pattern
    return float(np.std(np.asarray(x, dtype=float)))


def _pattern_diff(a: pd.Series, b: pd.Series) -> float:
    a, b = H.align_pairs(a, b)
    d = (a - a.mean()) - (b - b.mean())
    return float(np.sqrt(np.mean(d ** 2)))


def _recording_meta() -> pd.DataFrame:
    """viki's recordings.csv (notes/yang_response_normalization), keyed (genotype, date, fly_num, hemibrain)."""
    import data_targets as DT
    r = pd.read_csv(DT.INPUTS['yang_norm_recordings'])
    day, unit = (r['recording'].str.split('_', expand=True)[i] for i in (0, 1))
    r.index = pd.MultiIndex.from_arrays([r['genotype'], day.str[:4] + '-' + day.str[4:6] + '-' + day.str[6:8],
        unit.str[0].astype(int), unit.str[2]], names=['genotype'] + H.RECORDING_KEYS)
    return r


def recordings_table() -> pd.DataFrame:
    """Part D: one row per recording of both genotypes (see the module docstring)."""
    meta = _recording_meta()
    zm, nm = unit_matrices('z'), unit_matrices('noise')
    thr_n = unit_threshold('noise')
    two = strongest_recordings(reading_rdms('z')['TNT'])
    rows = []
    for g in GENOTYPES:
        for rec in reading_rdms('z')[g].index:
            rec = _rec(rec)
            row = dict(genotype=g, date=rec[0], fly_num=rec[1], hemibrain=rec[2], recording=rec_label(rec),
                n_kcs=int(len(zm[g][rec])), rate_z=float((zm[g][rec].to_numpy() > 1.0).mean()),
                rate_noise=float((nm[g][rec].to_numpy() > thr_n).mean()))
            for rd in READINGS:
                d = reading_rdms(rd)[g].loc[rec]
                gr = group_effects(d)
                row.update({f'mean_dist_{rd}': float(d.mean()), f'ester_dist_{rd}': gr['ester'], f'alcohol_dist_{rd}': gr['alcohol']})
            m = meta.loc[(g, *rec)]
            row.update(induction_hours=float(m['induction_hours']), median_scale_ratio=float(m['median_scale_ratio']),
                median_drift=float(m['median_drift']), n_kcs_meta=int(m['n_kcs']))
            row['label_group'] = (LABEL_GROUP.get(rec, LABEL_AMBIGUOUS) if g == 'TNT3c' else 'TNT-label')
            if g == 'TNT':
                row['strongest'] = rec in two
                for rd in READINGS:
                    R = reading_rdms(rd)
                    e = R['TNT3c'].mean() - R['TNT'].loc[rec]
                    gr = group_effects(e)
                    row.update({f'effect_mean_{rd}': float(e.mean()), f'effect_sd_{rd}': _std(e),
                        f'effect_ester_rel_{rd}': gr['ester_rel'], f'effect_alcohol_rel_{rd}': gr['alcohol_rel']})
            rows.append(row)
    df = pd.DataFrame(rows)
    assert (df['n_kcs'] == df['n_kcs_meta']).all()
    return df.drop(columns='n_kcs_meta')


def _steps(recs: Dict[str, pd.DataFrame], two: Sequence[tuple], genotype: str) -> Tuple[pd.Series, pd.Series, List[pd.Series]]:
    """(step 1, step 2, the two recordings' own steps) of one split: for 'TNT' step 1 is the control flies' mean minus the
    rest's (the rest's APL effect) and step 2 the rest's mean minus the two's; for 'TNT3c' step 1 is the rest's mean minus
    the APL-blocked mean (the APL effect with the rest as the control side) and step 2 the rest's mean minus the two's."""
    two = [tuple(t) for t in two]
    X = recs[genotype]
    rest = X.drop(index=two)
    if genotype == 'TNT':
        s1 = recs['TNT3c'].mean() - rest.mean()
    else:
        s1 = rest.mean() - recs['TNT'].mean()
    s2 = rest.mean() - X.loc[two].mean()
    own = [rest.mean() - X.loc[t] for t in two]
    return s1, s2, own


def split_stats(recs: Dict[str, pd.DataFrame], two: Sequence[tuple], genotype: str = 'TNT') -> Dict[str, float]:
    """The steps of one split of `genotype`'s recordings into `two` and the rest (`_steps`): ρ between step 1 and step 2
    (`rho_steps`, and Pearson), the agreement of the two recordings' own steps (`rho_own`), their mean ρ with step 1
    (`rho_own_step1`), the steps' means and sd over pairs, and step 2's groups relative to its mean."""
    s1, s2, own = _steps(recs, two, genotype)
    g2 = group_effects(s2)
    return dict(rho_steps=H.spearman(s1, s2), pearson_steps=H.pearson(s1, s2), rho_own=H.spearman(own[0], own[1]),
        rho_own_step1=float(np.mean([H.spearman(o, s1) for o in own])), step1_mean=float(s1.mean()), step1_sd=_std(s1),
        step2_mean=float(s2.mean()), step2_sd=_std(s2), step2_ester_rel=g2['ester_rel'], step2_alcohol_rel=g2['alcohol_rel'])


def _actual_two(genotype: str) -> List[tuple]:
    """The actual split's two recordings: the two strongest with APL output blocked (TNT), or the two control recordings
    with the lowest mean distance (both in Yang's z)."""
    return strongest_recordings(reading_rdms('z')[genotype])


def split_null(recs: Dict[str, pd.DataFrame], genotype: str = 'TNT') -> pd.DataFrame:
    """`split_stats` of every split of `genotype`'s recordings into two and the rest, with the actual split (`_actual_two`)
    marked (`actual`) and each statistic's rank among the splits (`rank_<stat>`, 1 the lowest)."""
    actual = set(_actual_two(genotype))
    rows = []
    for two in itertools.combinations([_rec(r) for r in recs[genotype].index], 2):
        rows.append(dict(two=' + '.join(rec_label(t) for t in two), actual=set(two) == actual, **split_stats(recs, two, genotype)))
    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c not in ('two', 'actual')]:
        df[f'rank_{c}'] = df[c].rank(method='min').astype(int)
    return df


def _step_rho_interval(recs: Dict[str, pd.DataFrame], two: Sequence[tuple], genotype: str, n_boot: int = 1000,
                       seed: int = 0) -> Dict[str, float]:
    """90 % intervals of `rho_steps` and `rho_own` over resamples of the recordings: the control recordings and the rest
    drawn with replacement within their groups, the two kept as they are (the split's subject)."""
    rng = np.random.default_rng(seed)
    two = [tuple(t) for t in two]
    X = recs[genotype]
    other = 'TNT' if genotype == 'TNT3c' else 'TNT3c'
    rest = X.drop(index=two)
    r_steps, r_own = [], []
    for _ in range(n_boot):
        o = recs[other].iloc[rng.integers(0, len(recs[other]), len(recs[other]))]
        r = rest.iloc[rng.integers(0, len(rest), len(rest))]
        s1 = (o.mean() - r.mean()) if genotype == 'TNT' else (r.mean() - o.mean())
        s2 = r.mean() - X.loc[two].mean()
        own = [r.mean() - X.loc[t] for t in two]
        r_steps.append(H.spearman(s1, s2))
        r_own.append(H.spearman(own[0], own[1]))
    lo, hi = np.nanpercentile(r_steps, [5, 95])
    lo2, hi2 = np.nanpercentile(r_own, [5, 95])
    return dict(rho_steps_lo=float(lo), rho_steps_hi=float(hi), rho_own_lo=float(lo2), rho_own_hi=float(hi2))


def axis_tests(out: Path) -> pd.DataFrame:
    """Part D's tests in each reading (`READINGS`): writes splits_tnt.csv, splits_ctrl.csv (every split, per reading) and
    axis_tests.csv (per reading and genotype: the actual split's statistics, their intervals over recordings, their ranks
    among the splits and the splits' median and 5th / 95th percentiles), and returns the latter."""
    nulls = {'TNT': [], 'TNT3c': []}
    rows = []
    for rd in READINGS:
        recs = reading_rdms(rd)
        for g in GENOTYPES:
            df = split_null(recs, g)
            df.insert(0, 'reading', rd)
            nulls[g].append(df)
            a = df[df['actual']].iloc[0]
            row = dict(reading=rd, genotype=g, n_splits=len(df), two=a['two'])
            for c in ('rho_steps', 'pearson_steps', 'rho_own', 'rho_own_step1', 'step1_mean', 'step2_mean', 'step2_sd',
                      'step2_ester_rel', 'step2_alcohol_rel'):
                row[c] = float(a[c])
                row[f'rank_{c}'] = int(a[f'rank_{c}'])
                row[f'null_median_{c}'] = float(df[c].median())
                row[f'null_lo_{c}'], row[f'null_hi_{c}'] = (float(v) for v in df[c].quantile([0.05, 0.95]))
            row.update(_step_rho_interval(recs, _actual_two(g), g))
            # the TNT six-to-two step against the controls' own step for the same kind of split
            if g == 'TNT3c':
                s_t = _steps(recs, _actual_two('TNT'), 'TNT')[1]
                s_c = _steps(recs, _actual_two('TNT3c'), 'TNT3c')[1]
                row['rho_ctrl_step2_vs_tnt_step2'] = H.spearman(s_c, s_t)
            rows.append(row)
    pd.concat(nulls['TNT'], ignore_index=True).to_csv(out / 'splits_tnt.csv', index=False)
    pd.concat(nulls['TNT3c'], ignore_index=True).to_csv(out / 'splits_ctrl.csv', index=False)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'axis_tests.csv', index=False)
    return res


def archive_pairs() -> Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]]:
    """{(folder, tag): (APL-on pairs, acute APL-off pairs)} of every archived variant (raw spike counts), as
    `apl_effect_scores.model_pairs`, with at least 100 pairs."""
    if 'archive' not in _CACHE:
        import re
        import apl_effect_scores as A
        out = {}
        for folder in sorted(A.RESULTS.glob('E*_*')):
            if not re.match(r'E\d+_', folder.name) or not folder.is_dir():
                continue
            for tag, (on, off) in A.model_pairs(folder).items():
                if len(H.apl_effect(on, off).dropna()) >= 100:
                    out[(folder.name, tag)] = (on, off)
        _CACHE['archive'] = out
    return _CACHE['archive']


def score_against(pairs: Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]], targets: Dict[str, pd.Series]) -> pd.DataFrame:
    """One row per variant: for each named target effect its loss, pattern part, mean error and ρ (`<name>_loss`,
    `<name>_pattern`, `<name>_mean_error`, `<name>_rho`), and the model's mean effect (`model_mean`)."""
    rows = []
    for (folder, tag), (on, off) in pairs.items():
        row = dict(folder=folder, tag=tag)
        for name, t in targets.items():
            st = H.apl_effect_stats(on, off, t)
            row.update({f'{name}_loss': st['apl_effect_rmse'], f'{name}_pattern': st['apl_effect_rmse_centered'],
                f'{name}_mean_error': st['apl_effect_mean_error'], f'{name}_rho': st['apl_effect_spearman']})
            row['model_mean'] = st['apl_effect_mean']
        rows.append(row)
    return pd.DataFrame(rows)


def with_bands(df: pd.DataFrame) -> pd.DataFrame:
    """`df` (folder, tag) with the response rates and the realism flags of results/apl_effect/by_variant.csv
    (`realism_flag`: the bands with the KC data in baseline-noise units; `realism_flag_zcrit`: in Yang's z)."""
    import apl_effect_scores as A
    bv = pd.read_csv(A.OUT / 'by_variant.csv', keep_default_na=False, na_values=[''])
    cols = ['folder', 'tag', 'rate_on', 'rate_off', 'realism_flag', 'realism_flag_zcrit']
    bv = bv[cols].copy()
    bv['tag'] = bv['tag'].astype(str)
    for c in ('realism_flag', 'realism_flag_zcrit'):
        bv[c] = bv[c].fillna('').astype(str)
    out = df.merge(bv, on=['folder', 'tag'], how='left')
    out['scored_in_bands'] = out['realism_flag'].notna()
    for c in ('realism_flag', 'realism_flag_zcrit'):
        out[c] = out[c].fillna('not scored')
    return out


# key models of the entry's tables: (short, label, folder, tag)
KEY_MODELS = [
    ('C2', 'C2, the breadth-2 APL-effect fit (E67)', 'E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3'),
    ('ceiling 1', 'the same family with the firing ceiling at strength 1 (E69 fit 1)', 'E69_kc_firing_ceiling',
     'rate0.068_aplboth_refr3_adapt0.04142t1000_k2'),
    ('ceiling 2', 'with the ceiling at strength 2 (E69 fit 2)', 'E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.1114t1000_k2'),
    ('reading ceiling', 'the ceiling fitted under the per-KC reading (E73 part C fit 2)', 'E73_per_kc_normalization',
     'zcaplrate_rate0.068_aplzcell_refr3_adapt0.04142t1000_k2_cap3'),
    ('ref. ceiling 2', 'the reference set with the ceiling at strength 2, fitted (E69 fit 3)', 'E69_kc_firing_ceiling',
     'thr40_gain1.5_rate0.068_aplboth_refr3_adapt0.1114t1000_k2'),
    ('ref. ceiling fixed', 'the reference model with the strongest ceiling at fixed scales (E69 part A)', 'E69_kc_firing_ceiling',
     'ref40_r3_a3_t1000'),
    ('ref.', 'the reference model (E44)', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'),
]


def subset_targets(units: Sequence[str] = ('z', 'noise')) -> Dict[str, pd.Series]:
    """The per-pair APL effect against the eight recordings with APL output blocked (TNT), the six typical ones and the two
    strongest, in each of `units` (`eight_z`, `six_z`, `two_z`, ...)."""
    out = {}
    two = _actual_two('TNT')
    for u in units:
        R = reading_rdms(u)
        C, T = R['TNT3c'].mean(), R['TNT']
        out[f'eight_{u}'] = H.apl_effect(C, T.mean())
        out[f'six_{u}'] = H.apl_effect(C, T.drop(index=two).mean())
        out[f'two_{u}'] = H.apl_effect(C, T.loc[two].mean())
    return out


def e72_distances(readout: str = 'cal', cal_target: Optional[float] = None) -> Tuple[List[str], List[Dict[str, pd.Series]]]:
    """E72's eight wirings (the real one's fit and the seven re-drawn ones, `E72_wiring_population.new_fit_members`):
    (labels, per-wiring {'on', 'off'} pair distances under `readout`); 'cal' calibrates each wiring's K on its own APL-on
    counts to `cal_target` (default the control flies' responders-only participation ratio in z, as E72)."""
    import E72_wiring_population as E72
    key = ('e72', readout, cal_target)
    if key not in _CACHE:
        labels, dists = [], []
        for lab, folder, tag in E72.new_fit_members():
            o = E72.load_fit(folder, tag)
            if o is None:
                continue
            if readout == 'cal' and cal_target is not None:
                s_on, s_off = o['on'][1], o['off'][1]
                K = H.calibrate_readout(s_on, 'saturating', cal_target, stat='participation_ratio', responders_min=2.0)
                d = {'on': H.readout_pair_distances(s_on, 'saturating', K), 'off': H.readout_pair_distances(s_off, 'saturating', K)}
            else:
                d = E72.distances(o, readout)
            labels.append(lab)
            dists.append(d)
        _CACHE[key] = (labels, dists)
    return _CACHE[key]


def common_error_against(targets: Dict[str, pd.Series], readout: str = 'cal', cal_target: Optional[float] = None,
                         reference: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """E72's common error (the pattern error of the mean over its eight wirings of the pair distances, model minus data,
    means removed) recomputed against each named target effect: (per pair, one column per target, `common_<name>`, and
    the population's effect `population`; per target its rms, the ester and alcohol groups' mean errors and its
    correlation with the common error against `reference`)."""
    import E72_wiring_population as E72
    labels, dists = e72_distances(readout, cal_target)
    md = E72.mean_distances(dists)
    cols = {f'common_{n}': H.apl_effect_pair_parts(md['on'], md['off'], t)['error'] for n, t in targets.items()}
    per_pair = pd.DataFrame(cols)
    per_pair['population'] = H.apl_effect(md['on'], md['off']).reindex(per_pair.index)
    ref = per_pair[f'common_{reference}'] if reference else None
    rows = []
    for n in targets:
        c = per_pair[f'common_{n}']
        rows.append(dict(target=n, readout=readout, n_wirings=len(labels), rms=float(np.sqrt(np.mean(c ** 2))),
            err_ester=float(c[pair_mask(c.index, ESTER_GROUP)].mean()), err_alcohol=float(c[pair_mask(c.index, ALCOHOL_GROUP)].mean()),
            corr_with_reference=np.nan if ref is None else float(np.corrcoef(c, ref)[0, 1])))
    return per_pair, pd.DataFrame(rows)


def recording_subsets(out: Path) -> pd.DataFrame:
    """recording_subsets.csv for E75: per pair the effect against the six, the two and the eight in z and in
    baseline-noise units, the population's effect (E72's eight wirings, calibrated readout) and E72's common error
    against each (calibrated readout; K in z, as E72's)."""
    t = subset_targets(('z', 'noise'))
    per_pair, _ = common_error_against(t, 'cal')
    eff = pd.DataFrame({f'effect_{n}': v for n, v in t.items()})
    df = eff.join(per_pair, how='outer')
    df.index.names = ['odor_a', 'odor_b']
    df.to_csv(out / 'recording_subsets.csv')
    return df


def split_halves(out: Path, n_splits: int = N_HALF_SPLITS, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """split_halves.csv for E75: `n_splits` half-splits within genotype drawn as E75's `_split_halves` (one generator
    seeded `seed`; per split a permutation of the 11 control recordings, then of the 8 with APL output blocked (TNT); half
    A the first n // 2, in the row order of `yang_recording_rdms_values`), the recordings in each half, the stratum
    (`strongest_together`: both strongest recordings in one half), ρ between the halves' effects (z), the pattern
    difference, and the groups' relative effects in each half and their differences."""
    rng = np.random.default_rng(seed)
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    two = set(_actual_two('TNT'))
    lab_c = [rec_label(_rec(r)) for r in ctrl.index]
    lab_t = [rec_label(_rec(r)) for r in tnt.index]
    rows = []
    for k in range(n_splits):
        ic, it = rng.permutation(len(ctrl)), rng.permutation(len(tnt))
        halves = []
        for sel_c, sel_t in ((ic[:len(ic) // 2], it[:len(it) // 2]), (ic[len(ic) // 2:], it[len(it) // 2:])):
            halves.append((sel_c, sel_t, ctrl.iloc[sel_c].mean() - tnt.iloc[sel_t].mean()))
        (ca, ta, ea), (cb, tb, eb) = halves
        n_a = sum(_rec(tnt.index[i]) in two for i in ta)
        ga, gb = group_effects(ea), group_effects(eb)
        rows.append(dict(split=k, seed=seed, strongest_in_a=int(n_a), strongest_together=n_a in (0, 2),
            ctrl_a=';'.join(lab_c[i] for i in ca), ctrl_b=';'.join(lab_c[i] for i in cb),
            tnt_a=';'.join(lab_t[i] for i in ta), tnt_b=';'.join(lab_t[i] for i in tb),
            rho_halves=H.spearman(ea, eb), pattern_diff=_pattern_diff(ea, eb), mean_a=float(ea.mean()), mean_b=float(eb.mean()),
            ester_rel_a=ga['ester_rel'], ester_rel_b=gb['ester_rel'], alcohol_rel_a=ga['alcohol_rel'], alcohol_rel_b=gb['alcohol_rel'],
            ester_rel_diff=abs(ga['ester_rel'] - gb['ester_rel']), alcohol_rel_diff=abs(ga['alcohol_rel'] - gb['alcohol_rel'])))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'split_halves.csv', index=False)
    return df


def _check_e75_splits(df: pd.DataFrame, n: int = 100) -> bool:
    """Whether the first `n` rows of split_halves.csv are E75 D1's splits (its `_split_halves`, seed 0): the same halves'
    effects."""
    import E75_glomerulus_apl_profile as E75
    ref = E75._split_halves(n_splits=n, seed=SPLIT_SEED)
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    lab_c = {rec_label(_rec(r)): i for i, r in enumerate(ctrl.index)}
    lab_t = {rec_label(_rec(r)): i for i, r in enumerate(tnt.index)}
    for k in range(n):
        r = df.iloc[k]
        ea = ctrl.iloc[[lab_c[x] for x in r['ctrl_a'].split(';')]].mean() - tnt.iloc[[lab_t[x] for x in r['tnt_a'].split(';')]].mean()
        ea.index = E75._canon(ea.index)
        if not np.allclose(ea.sort_index().to_numpy(), ref[k][0].to_numpy(), rtol=0, atol=1e-12):
            return False
    return True


def subset_floors(n_boot: int = 1000, n_split: int = 500, seed: int = 0) -> pd.DataFrame:
    """The floor of the loss and of its pattern part and the split-half ρ and ceiling of the effect against the six
    typical recordings with APL output blocked (TNT), the two strongest and the eight, in z and baseline-noise units."""
    two = _actual_two('TNT')
    rows = []
    for u in ('z', 'noise'):
        R = reading_rdms(u)
        for name, T in (('eight', R['TNT']), ('six', R['TNT'].drop(index=two)), ('two', R['TNT'].loc[two])):
            r = effect_resamples({'TNT3c': R['TNT3c'], 'TNT': T}, n_boot=n_boot, n_split=n_split if len(T) >= 4 else 0, seed=seed)
            rows.append(dict(unit=u, subset=name, n_tnt=len(T), **{k: v for k, v in r.items() if not isinstance(v, (pd.Series, dict))}))
    return pd.DataFrame(rows)


def _f(x, spec: str = '.3f') -> str:
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def part_d(out: Path) -> None:
    """Part D: the recordings (see the module docstring)."""
    t0 = datetime.now(timezone.utc)
    rec = recordings_table()
    rec.to_csv(out / 'recordings.csv', index=False)
    print(rec[['genotype', 'recording', 'n_kcs', 'mean_dist_z', 'mean_dist_nc_z', 'mean_dist_noise', 'rate_z', 'rate_noise',
               'ester_dist_z', 'alcohol_dist_z', 'induction_hours', 'median_scale_ratio', 'label_group']].round(3).to_string(), flush=True)
    two = _actual_two('TNT')
    print('the two strongest recordings with APL output blocked (TNT):', [rec_label(t) for t in two], flush=True)
    tnt = rec[rec['genotype'] == 'TNT']
    for u in ('z', 'noise'):
        R = reading_rdms(u)
        C, T = R['TNT3c'].mean(), R['TNT']
        e6, e2, e8 = H.apl_effect(C, T.drop(index=two).mean()), H.apl_effect(C, T.loc[two].mean()), H.apl_effect(C, T.mean())
        g6, g2, g8 = group_effects(e6), group_effects(e2), group_effects(e8)
        print(f'{u}: effect against the six {e6.mean():+.3f} (sd {_std(e6):.3f}; ester / alcohol rel {g6["ester_rel"]:+.3f} / '
              f'{g6["alcohol_rel"]:+.3f}), the two {e2.mean():+.3f} (sd {_std(e2):.3f}; {g2["ester_rel"]:+.3f} / {g2["alcohol_rel"]:+.3f}), '
              f'the eight {e8.mean():+.3f} (sd {_std(e8):.3f}; {g8["ester_rel"]:+.3f} / {g8["alcohol_rel"]:+.3f}); ρ(six, eight) '
              f'{H.spearman(e6, e8):.2f}, ρ(two, eight) {H.spearman(e2, e8):.2f}, ρ(six, two) {H.spearman(e6, e2):.2f}', flush=True)
    tests = axis_tests(out)
    print(tests[['reading', 'genotype', 'rho_steps', 'rho_steps_lo', 'rho_steps_hi', 'rank_rho_steps', 'n_splits',
                 'null_median_rho_steps', 'rho_own', 'rho_own_lo', 'rho_own_hi', 'rank_rho_own', 'null_median_rho_own',
                 'step2_mean', 'rank_step2_mean']].round(3).to_string(), flush=True)
    # every archived variant against the six's, the two's and the eight's effect
    targets = subset_targets(('z', 'noise'))
    sc = with_bands(score_against(archive_pairs(), targets))
    sc.to_csv(out / 'subsets_vs_archive.csv', index=False)
    ok = sc[sc['realism_flag'] == '']
    print(f'{len(sc)} archived variants ({len(ok)} inside the bands, baseline-noise units)', flush=True)
    for n in ('six_z', 'two_z', 'eight_z', 'six_noise', 'two_noise'):
        b = ok.sort_values(f'{n}_loss').head(5)
        print(f'best unflagged against {n}:', '; '.join(f"{r.folder}:{r.tag} {r[n + '_loss']:.3f}/{r[n + '_pattern']:.3f}/{r[n + '_rho']:.2f}"
              for _, r in b.iterrows()), flush=True)
    key = sc.set_index(['folder', 'tag'])
    for short, lab, f, tg in KEY_MODELS:
        if (f, tg) in key.index:
            r = key.loc[(f, tg)]
            print(f'{short:20s} mean {r.model_mean:+.3f} | ' + ' | '.join(f"{n} {r[n + '_loss']:.3f}/{r[n + '_pattern']:.3f}/{r[n + '_rho']:.2f}"
                  for n in ('eight_z', 'six_z', 'two_z', 'eight_noise', 'six_noise', 'two_noise')), flush=True)
    # E72's common error against each subset (calibrated readout, K as E72's)
    per_pair, ce = common_error_against(targets, 'cal', reference='eight_z')
    per_pair.index.names = ['odor_a', 'odor_b']
    per_pair.to_csv(out / 'common_error_subsets.csv')
    ce.to_csv(out / 'common_error_subsets_summary.csv', index=False)
    print(ce.round(3).to_string(), flush=True)
    floors = subset_floors()
    floors.to_csv(out / 'subset_floors.csv', index=False)
    print(floors.round(3).to_string(), flush=True)
    rs = recording_subsets(out)
    sh = split_halves(out)
    same = _check_e75_splits(sh)
    st = sh.groupby('strongest_together').agg(n=('split', 'size'), rho_median=('rho_halves', 'median'),
        rho_below_045=('rho_halves', lambda s: float((s < 0.45).mean())), pattern_diff_median=('pattern_diff', 'median'),
        alcohol_rel_diff_median=('alcohol_rel_diff', 'median'), ester_rel_diff_median=('ester_rel_diff', 'median'))
    print(st.round(3).to_string(), flush=True)
    print(f'first 100 half-splits are E75 D1\'s: {same}', flush=True)
    _provenance(out, 'D', dict(strongest=[rec_label(t) for t in two], n_variants=len(sc), split_seed=SPLIT_SEED,
        n_half_splits=N_HALF_SPLITS, first_100_are_e75_d1_splits=same, strata=st.reset_index().to_dict(orient='records'),
        started=t0.isoformat(timespec='seconds')))


def _provenance(out: Path, part: str, extra: dict) -> None:
    import data_targets as DT
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', Path(__file__).name, 'kc_remapping_harness.py', 'data_targets.py'],
        cwd=H.HERE, capture_output=True, text=True).stdout.strip())
    files = {str(NOISE_DIR / n): None for n in ('trialmean_long.parquet', 'trialmean_long_unnormalized.parquet',
        'per_trial_long.parquet', 'per_trial_long_unnormalized.parquet')}
    files[str(DT.INPUTS['yang_norm_kc_scales'])] = None
    files[str(DT.INPUTS['yang_norm_recordings'])] = None
    for f in files:
        p = Path(f)
        files[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    prov = dict(script=f'{Path(__file__).name} --part {part}', generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        host=subprocess.run(['hostname', '-s'], capture_output=True, text=True).stdout.strip(), al_analysis_commit=commit,
        dirty=dirty, inputs=files, unit_thresholds={u: _CACHE.get(('thr', u)) for u in UNITS}, denominator_floors=dict(FLOORS),
        registry=str(H.data_targets_path()), **extra)
    (out / f'provenance_{part}.json').write_text(json.dumps(prov, indent=1, ensure_ascii=False, default=str) + '\n')


def _fly_positions(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Row positions of one draw of flies (date, fly_num) with replacement, all hemispheres of each drawn fly kept (as
    `kc_remapping_harness.resample_recording_means(unit='fly')`)."""
    flies = list(dict.fromkeys((r[0], r[1]) for r in df.index))
    drawn = [flies[i] for i in rng.integers(0, len(flies), len(flies))]
    pos = {f: [i for i, r in enumerate(df.index) if (r[0], r[1]) == f] for f in flies}
    return np.concatenate([pos[f] for f in drawn])


def _canon_array(df: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """(recordings x pairs) array of a per-recording pair frame with its columns in the order of the canonical pair index
    `index` (each pair found in either orientation; NaN where a pair is missing)."""
    pos = {tuple(c): i for i, c in enumerate(df.columns)}
    X = df.to_numpy(dtype=float)
    out = np.full((X.shape[0], len(index)), np.nan)
    for j, (a, b) in enumerate(index):
        i = pos.get((a, b), pos.get((b, a)))
        if i is not None:
            out[:, j] = X[:, i]
    return out


def _nanmean_rows(X: np.ndarray) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(X, axis=0)


def _spearman_arr(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats
    ok = np.isfinite(a) & np.isfinite(b)
    return float(stats.spearmanr(a[ok], b[ok])[0]) if ok.sum() > 2 else np.nan


def _pattern_diff_arr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a[ok] - a[ok].mean()) - (b[ok] - b[ok].mean())
    return float(np.sqrt(np.mean(d ** 2)))


def effect_resamples(recs: Dict[str, pd.DataFrame], ref: Optional[Dict[str, pd.DataFrame]] = None, n_boot: int = 1000,
                     n_split: int = 500, seed: int = 0) -> Dict[str, object]:
    """The per-pair APL effect of per-recording distances `recs` ({'TNT3c', 'TNT'}, in this order) with its statistics
    over resamples of the recordings, and, given `ref` (the same recordings in another unit, e.g. Yang's z), ρ and the
    pattern difference (rms of the difference with both means removed) against `ref`'s effect, over the same draws.

    One generator seeded `seed` draws `n_boot` resamples of recordings (with replacement within each genotype), then
    `n_split` random halves (each genotype split in two), in the order of data_targets.apl_effect_resamples, so that in z
    the mean, interval, floor and split-half ρ are the registry's apl_effect_* entries (to rounding); the intervals over
    flies come from a second generator seeded `seed + 1` (flies drawn with replacement within each genotype, a fly's
    hemispheres kept). Means over recordings and pairs skip missing values, as the pandas computation does.

    Returns: `effect` (Series), `mean`, `mean_lo` / `mean_hi` (90 % over recordings), `mean_fly_lo` / `mean_fly_hi`, `sd`
    (over pairs, ddof 0: what an effect with the right mean and no pattern scores as the pattern part), `rmse_floor` (the
    mean over resamples of the loss of a resampled data set against the measured effect) with its 5th / 95th percentiles,
    `pattern_floor` (the same for the pattern part), `split_half_rho` (mean over splits) with its interval and
    `rho_ceiling` (sqrt of the Spearman-Brown-corrected split-half ρ), the groups' relative effects with intervals over
    recordings, and, with `ref`, `rho_ref`, `pattern_diff_ref` with intervals."""
    recs = {g: recs[g] for g in GENOTYPES}
    effect = H.apl_effect(recs['TNT3c'].mean(), recs['TNT'].mean())
    idx = effect.index
    gr = group_effects(effect)
    out: Dict[str, object] = dict(effect=effect, mean=float(effect.mean()), sd=_std(effect), ester=gr['ester'], alcohol=gr['alcohol'],
        ester_rel=gr['ester_rel'], alcohol_rel=gr['alcohol_rel'])
    X = {g: _canon_array(recs[g], idx) for g in GENOTYPES}
    e0 = effect.to_numpy(dtype=float)
    est, alc = pair_mask(idx, ESTER_GROUP), pair_mask(idx, ALCOHOL_GROUP)

    def groups(e):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            m = np.nanmean(e)
            return float(np.nanmean(e[est]) - m), float(np.nanmean(e[alc]) - m)

    Xr = None
    if ref is not None:
        ref = {g: ref[g].reindex(recs[g].index) for g in GENOTYPES}
        assert not any(ref[g].isna().all(axis=1).any() for g in GENOTYPES)
        e_ref = H.apl_effect(ref['TNT3c'].mean(), ref['TNT'].mean())
        out.update(rho_ref=H.spearman(effect, e_ref), pattern_diff_ref=_pattern_diff(effect, e_ref))
        Xr = {g: _canon_array(ref[g], idx) for g in GENOTYPES}
    rng = np.random.default_rng(seed)
    b = {k: [] for k in ('mean', 'rmse', 'pattern', 'ester_rel', 'alcohol_rel', 'rho_ref', 'pattern_diff_ref')}
    for _ in range(n_boot):
        pos = {g: rng.integers(0, len(X[g]), len(X[g])) for g in GENOTYPES}
        e_b = _nanmean_rows(X['TNT3c'][pos['TNT3c']]) - _nanmean_rows(X['TNT'][pos['TNT']])
        e_b = np.where(np.isfinite(e0), e_b, np.nan)
        err = e_b - e0
        ok = np.isfinite(err)
        b['mean'].append(float(np.nanmean(e_b)))
        b['rmse'].append(float(np.sqrt(np.mean(err[ok] ** 2))))
        b['pattern'].append(float(np.sqrt(np.mean((err[ok] - err[ok].mean()) ** 2))))
        g_e, g_a = groups(e_b)
        b['ester_rel'].append(g_e)
        b['alcohol_rel'].append(g_a)
        if Xr is not None:
            r_b = _nanmean_rows(Xr['TNT3c'][pos['TNT3c']]) - _nanmean_rows(Xr['TNT'][pos['TNT']])
            b['rho_ref'].append(_spearman_arr(e_b, r_b))
            b['pattern_diff_ref'].append(_pattern_diff_arr(e_b, r_b))
    pct = lambda v: [float(x) for x in np.nanpercentile(v, [5, 95])]
    out['mean_lo'], out['mean_hi'] = pct(b['mean'])
    out['rmse_floor'] = float(np.mean(b['rmse']))
    out['rmse_floor_lo'], out['rmse_floor_hi'] = pct(b['rmse'])
    out['pattern_floor'] = float(np.mean(b['pattern']))
    out['pattern_floor_lo'], out['pattern_floor_hi'] = pct(b['pattern'])
    for k in ('ester_rel', 'alcohol_rel'):
        out[f'{k}_lo'], out[f'{k}_hi'] = pct(b[k])
    if Xr is not None:
        for k in ('rho_ref', 'pattern_diff_ref'):
            out[f'{k}_lo'], out[f'{k}_hi'] = pct(b[k])
    sh = []
    for _ in range(n_split):
        halves = [{}, {}]
        for g in GENOTYPES:
            i = rng.permutation(len(X[g]))
            halves[0][g] = _nanmean_rows(X[g][i[:len(i) // 2]])
            halves[1][g] = _nanmean_rows(X[g][i[len(i) // 2:]])
        sh.append(_spearman_arr(halves[0]['TNT3c'] - halves[0]['TNT'], halves[1]['TNT3c'] - halves[1]['TNT']))
    if sh:
        r = float(np.nanmean(sh))
        out['split_half_rho'] = r
        out['split_half_lo'], out['split_half_hi'] = pct(sh)
        out['rho_ceiling'] = float(np.sqrt(2 * r / (1 + r))) if r > 0 else np.nan
    rng_f = np.random.default_rng(seed + 1)
    bf = []
    for _ in range(n_boot):
        pos = {g: _fly_positions(recs[g], rng_f) for g in GENOTYPES}
        e_b = _nanmean_rows(X['TNT3c'][pos['TNT3c']]) - _nanmean_rows(X['TNT'][pos['TNT']])
        bf.append(float(np.nanmean(np.where(np.isfinite(e0), e_b, np.nan))))
    out['mean_fly_lo'], out['mean_fly_hi'] = pct(bf)
    return out


def _effect_resamples_pandas(recs: Dict[str, pd.DataFrame], ref: Optional[Dict[str, pd.DataFrame]] = None, n_boot: int = 1000,
                     n_split: int = 500, seed: int = 0) -> Dict[str, object]:
    """(The pandas form of `effect_resamples`, kept to check it.) The per-pair APL effect of per-recording distances `recs` ({'TNT3c', 'TNT'}, in this order) with its statistics
    over resamples of the recordings, and, given `ref` (the same recordings in another unit, e.g. Yang's z), ρ and the
    pattern difference (rms of the difference with both means removed) against `ref`'s effect, over the same draws.

    One generator seeded `seed` draws `n_boot` resamples of recordings (with replacement within each genotype), then
    `n_split` random halves (each genotype split in two), in the order of data_targets.apl_effect_resamples, so that in z
    the mean, interval, floor and split-half ρ are the registry's apl_effect_* entries; the intervals over flies come from
    a second generator seeded `seed + 1` (flies drawn with replacement within each genotype, a fly's hemispheres kept).

    Returns: `effect` (Series), `mean`, `mean_lo` / `mean_hi` (90 % over recordings), `mean_fly_lo` / `mean_fly_hi`, `sd`
    (over pairs, ddof 0: what an effect with the right mean and no pattern scores as the pattern part), `rmse_floor` (the
    mean over resamples of the loss of a resampled data set against the measured effect) with its 5th / 95th percentiles,
    `pattern_floor` (the same for the pattern part), `split_half_rho` (mean over splits) with its interval and
    `rho_ceiling` (sqrt of the Spearman-Brown-corrected split-half ρ), the groups' relative effects with intervals over
    recordings, and, with `ref`, `rho_ref`, `pattern_diff_ref` with intervals."""
    recs = {g: recs[g] for g in GENOTYPES}
    effect = H.apl_effect(recs['TNT3c'].mean(), recs['TNT'].mean())
    gr = group_effects(effect)
    out: Dict[str, object] = dict(effect=effect, mean=float(effect.mean()), sd=_std(effect), ester=gr['ester'], alcohol=gr['alcohol'],
        ester_rel=gr['ester_rel'], alcohol_rel=gr['alcohol_rel'])
    e_ref = None
    if ref is not None:
        ref = {g: ref[g].reindex(recs[g].index) for g in GENOTYPES}
        assert not any(ref[g].isna().all(axis=1).any() for g in GENOTYPES)
        e_ref = H.apl_effect(ref['TNT3c'].mean(), ref['TNT'].mean())
        out.update(rho_ref=H.spearman(effect, e_ref), pattern_diff_ref=_pattern_diff(effect, e_ref))
    rng = np.random.default_rng(seed)
    b = {k: [] for k in ('mean', 'rmse', 'pattern', 'ester_rel', 'alcohol_rel', 'rho_ref', 'pattern_diff_ref')}
    for _ in range(n_boot):
        pos = {g: rng.integers(0, len(df), len(df)) for g, df in recs.items()}
        e_b = H.apl_effect(recs['TNT3c'].iloc[pos['TNT3c']].mean(), recs['TNT'].iloc[pos['TNT']].mean()).reindex(effect.index)
        err = e_b - effect
        g_b = group_effects(e_b)
        b['mean'].append(float(e_b.mean()))
        b['rmse'].append(float(np.sqrt(np.mean(err ** 2))))
        b['pattern'].append(float(np.sqrt(np.mean((err - err.mean()) ** 2))))
        b['ester_rel'].append(g_b['ester_rel'])
        b['alcohol_rel'].append(g_b['alcohol_rel'])
        if ref is not None:
            r_b = H.apl_effect(ref['TNT3c'].iloc[pos['TNT3c']].mean(), ref['TNT'].iloc[pos['TNT']].mean())
            b['rho_ref'].append(H.spearman(e_b, r_b))
            b['pattern_diff_ref'].append(_pattern_diff(e_b, r_b))
    pct = lambda v: [float(x) for x in np.nanpercentile(v, [5, 95])]
    out['mean_lo'], out['mean_hi'] = pct(b['mean'])
    out['rmse_floor'] = float(np.mean(b['rmse']))
    out['rmse_floor_lo'], out['rmse_floor_hi'] = pct(b['rmse'])
    out['pattern_floor'] = float(np.mean(b['pattern']))
    out['pattern_floor_lo'], out['pattern_floor_hi'] = pct(b['pattern'])
    for k in ('ester_rel', 'alcohol_rel'):
        out[f'{k}_lo'], out[f'{k}_hi'] = pct(b[k])
    if ref is not None:
        for k in ('rho_ref', 'pattern_diff_ref'):
            out[f'{k}_lo'], out[f'{k}_hi'] = pct(b[k])
    sh = []
    for _ in range(n_split):
        halves = [{}, {}]
        for g, df in recs.items():
            idx = rng.permutation(len(df))
            halves[0][g] = df.iloc[idx[:len(df) // 2]].mean()
            halves[1][g] = df.iloc[idx[len(df) // 2:]].mean()
        a = H.apl_effect(halves[0]['TNT3c'], halves[0]['TNT'])
        c = H.apl_effect(halves[1]['TNT3c'], halves[1]['TNT']).reindex(a.index)
        sh.append(H.spearman(a, c))
    if sh:
        r = float(np.nanmean(sh))
        out['split_half_rho'] = r
        out['split_half_lo'], out['split_half_hi'] = pct(sh)
        out['rho_ceiling'] = float(np.sqrt(2 * r / (1 + r))) if r > 0 else np.nan
    rng_f = np.random.default_rng(seed + 1)
    bf = []
    for _ in range(n_boot):
        pos = {g: _fly_positions(df, rng_f) for g, df in recs.items()}
        bf.append(float(H.apl_effect(recs['TNT3c'].iloc[pos['TNT3c']].mean(), recs['TNT'].iloc[pos['TNT']].mean()).mean()))
    out['mean_fly_lo'], out['mean_fly_hi'] = pct(bf)
    return out


VALUES = ('graded', 'thresholded', 'binary')
VALUES_LABEL = {'graded': 'graded', 'thresholded': 'sub-threshold entries set to 0', 'binary': 'binarized'}


def data_effect(unit: str = 'z', values: str = 'graded') -> pd.Series:
    """The KC data's per-pair APL effect (control flies minus APL output blocked (TNT), cosine, each the mean over
    recordings) with each KC's trial means in `unit`; `values` as `unit_recording_rdms`. In z, graded, this is
    `kc_remapping_harness.yang_apl_effect()` (to rounding); in 'noise' it is the target of E10_joint_fit.py's
    `--apl-data-units noise`."""
    R = unit_recording_rdms(unit, values)
    return H.apl_effect(R['TNT3c'].mean(), R['TNT'].mean())


def cal_target(unit: str = 'z') -> float:
    """E65's calibration target in `unit`: the control flies' responders-only participation ratio ((sum x^2)^2 /
    (n sum x^4) of each odor's response vector across KCs, trial means at or below `unit_threshold(unit)` set to 0),
    mean over odors and recordings; in z the registry's ctrl_participation_ratio_resp."""
    key = ('cal_target', unit)
    if key not in _CACHE:
        thr = unit_threshold(unit)
        _CACHE[key] = float(np.mean([float(H.amplitude_stats(m.where(m > thr, 0.0))['participation_ratio'].mean())
            for m in unit_matrices(unit)['TNT3c'].values()]))
    return _CACHE[key]


def unit_rates(unit: str) -> Dict[str, float]:
    """Each genotype's response rate in `unit` (entries above `unit_threshold(unit)`, mean over the 16 odors and the
    recordings)."""
    thr = unit_threshold(unit)
    return {g: float(np.mean([float((m.to_numpy() > thr).mean()) for m in unit_matrices(unit)[g].values()])) for g in GENOTYPES}


CLASS_N_BOOT = 1000


def class_split_units(out: Optional[Path], n_boot: int = CLASS_N_BOOT, seed: int = 0) -> pd.DataFrame:
    """The split of the per-pair APL effect by odors per KC (`kc_remapping_harness.apl_effect_by_class`: each condition's
    class contributions to the cosine similarity averaged over its recordings, the classes set by that condition's
    responses) in baseline-noise units at noise_threshold, beside the same in z at 1 z (E74's): per unit, class and term
    the mean effect over pairs, its share of the effect's variance over pairs (cov(class effect, effect) / var(effect)),
    each condition's level and share of KCs, with 90 % intervals over `n_boot` resamples of recordings (one generator
    seeded `seed` per unit, the control recordings drawn first). Writes class_split_units.csv to `out` (if given)."""
    import E74_apl_effect_by_odors_per_kc as E74
    rows = []
    checks = {}
    for unit in ('z', 'noise'):
        thr = unit_threshold(unit)
        mats = unit_matrices(unit)
        obs = H.apl_effect_by_class(list(mats['TNT3c'].values()), list(mats['TNT'].values()), thresh_on=thr, thresh_off=thr)
        arrs, shares = {}, {}
        for g in GENOTYPES:
            a, sh = [], []
            for m in mats[g].values():
                X, _, _ = H._ordered_matrix(m)
                R = X > thr
                a.append(H.class_cosine_terms_array(X, R))
                sh.append(H.opk_class_shares(R))
            arrs[g], shares[g] = np.stack(a), np.stack(sh)
        eff = np.nanmean(arrs['TNT'], axis=0) - np.nanmean(arrs['TNT3c'], axis=0)
        checks[unit] = float(np.nanmax(np.abs(eff.reshape(len(eff), -1) - obs['effect'].to_numpy())))
        n_cls = eff.shape[1]
        labels = list(H.opk_class_labels()) + ['all']

        def stats_of(on, off):
            e = off - on
            tot = e[:, -1, -1]
            v = float(np.var(tot))
            means = e.mean(axis=0)
            vsh = ((e - means) * (tot - tot.mean())[:, None, None]).mean(axis=0) / v
            return dict(effect=means, var_share=vsh, level_on=on.mean(axis=0), level_off=off.mean(axis=0))

        o = stats_of(np.nanmean(arrs['TNT3c'], axis=0), np.nanmean(arrs['TNT'], axis=0))
        rng = np.random.default_rng(seed)
        boot = {k: [] for k in o}
        bsh = {'kc_share_on': [], 'kc_share_off': []}
        for _ in range(n_boot):
            ia = rng.integers(0, len(arrs['TNT3c']), len(arrs['TNT3c']))
            ib = rng.integers(0, len(arrs['TNT']), len(arrs['TNT']))
            b = stats_of(np.nanmean(arrs['TNT3c'][ia], axis=0), np.nanmean(arrs['TNT'][ib], axis=0))
            for k in boot:
                boot[k].append(b[k])
            bsh['kc_share_on'].append(shares['TNT3c'][ia].mean(axis=0))
            bsh['kc_share_off'].append(shares['TNT'][ib].mean(axis=0))
        for k, v in o.items():
            B = np.stack(boot[k])
            for c in range(n_cls):
                for t, term in enumerate(H.CLASS_TERMS):
                    if k == 'var_share' and labels[c] == 'all' and term == 'total':
                        continue
                    x = B[:, c, t]
                    rows.append(dict(unit=unit, threshold=thr, quantity=k, **{'class': labels[c]}, term=term, observed=float(v[c, t]),
                        lo=float(np.nanpercentile(x, 5)), hi=float(np.nanpercentile(x, 95))))
        for k, cond in (('kc_share_on', 'TNT3c'), ('kc_share_off', 'TNT')):
            B = np.stack(bsh[k])
            for c in range(n_cls - 1):
                rows.append(dict(unit=unit, threshold=thr, quantity=k, **{'class': labels[c]}, term='', observed=float(shares[cond].mean(axis=0)[c]),
                    lo=float(np.percentile(B[:, c], 5)), hi=float(np.percentile(B[:, c], 95))))
    df = pd.DataFrame(rows)
    # the z rows against E74's data_by_class.csv (observed, criterion 1 z)
    p = H.results_dir(E74.EXPERIMENT) / 'data_by_class.csv'
    if p.exists():
        e74 = pd.read_csv(p, dtype={'class': str})
        e74 = e74[(e74['criterion'] == 1.0) & e74['quantity'].isin(['effect', 'var_share'])].set_index(['quantity', 'class', 'term'])['observed']
        mine = df[df['unit'] == 'z'].set_index(['quantity', 'class', 'term'])['observed']
        common = mine.index.intersection(e74.index)
        checks['z_vs_e74_max_abs'] = float((mine.reindex(common) - e74.reindex(common)).abs().max())
    df.attrs['checks'] = checks
    if out is not None:
        df.to_csv(out / 'class_split_units.csv', index=False)
    return df


def part_b(out: Path) -> None:
    """Part B: the unit (data_units.csv, effect_by_unit_pairs.csv, class_split_units.csv)."""
    t0 = datetime.now(timezone.utc)
    ref = unit_recording_rdms('z', 'graded')
    rows, pairs = [], {}
    for unit in UNITS:
        rates = unit_rates(unit)
        for values in VALUES:
            R = unit_recording_rdms(unit, values)
            r = effect_resamples(R, ref=ref)
            e = r.pop('effect')
            same = data_effect('z', values)
            row = dict(unit=unit, values=values, unit_label=UNIT_LABEL[unit], threshold=unit_threshold(unit),
                denominator_floor=FLOORS.get(unit, np.nan), mean_dist_ctrl=float(R['TNT3c'].mean().mean()),
                mean_dist_tnt=float(R['TNT'].mean().mean()), rate_ctrl=rates['TNT3c'], rate_tnt=rates['TNT'],
                rate_ratio=rates['TNT'] / rates['TNT3c'], cal_target=cal_target(unit) if values == 'graded' else np.nan,
                rho_vs_z_same_values=H.spearman(e, same), pattern_diff_vs_z_same_values=_pattern_diff(e, same),
                effect_min=float(e.min()), effect_max=float(e.max()), sd_ddof1=float(e.std()),
                **{k: v for k, v in r.items() if not isinstance(v, (pd.Series, dict))})
            rows.append(row)
            pairs[f'effect__{unit}__{values}'] = e
            if values == 'graded':
                pairs[f'ctrl__{unit}'] = H.canonical_pairs(R['TNT3c'].mean())
                pairs[f'tnt__{unit}'] = H.canonical_pairs(R['TNT'].mean())
            print(f"{unit:12s} {values:11s} thr {row['threshold']:.4g}: dist {row['mean_dist_ctrl']:.3f} / {row['mean_dist_tnt']:.3f}, "
                  f"effect {row['mean']:+.3f} [{row['mean_lo']:+.3f}, {row['mean_hi']:+.3f}] (flies [{row['mean_fly_lo']:+.3f}, "
                  f"{row['mean_fly_hi']:+.3f}]), sd {row['sd']:.3f}, floor {row['rmse_floor']:.3f} (pattern {row['pattern_floor']:.3f}), "
                  f"split-half ρ {row.get('split_half_rho', np.nan):.3f} (ceiling {row.get('rho_ceiling', np.nan):.3f}), ρ with z "
                  f"{row['rho_ref']:.3f} [{row['rho_ref_lo']:.3f}, {row['rho_ref_hi']:.3f}], pattern diff {row['pattern_diff_ref']:.3f} "
                  f"[{row['pattern_diff_ref_lo']:.3f}, {row['pattern_diff_ref_hi']:.3f}], ester / alcohol {row['ester_rel']:+.3f} "
                  f"[{row['ester_rel_lo']:+.3f}, {row['ester_rel_hi']:+.3f}] / {row['alcohol_rel']:+.3f} [{row['alcohol_rel_lo']:+.3f}, "
                  f"{row['alcohol_rel_hi']:+.3f}]", flush=True)
    # each recording's cosine corrected for trial-to-trial noise, in z and in baseline-noise units
    for unit, rd in (('z', 'nc_z'), ('noise', 'nc_noise')):
        R = reading_rdms(rd)
        r = effect_resamples(R, ref=ref)
        e = r.pop('effect')
        rows.append(dict(unit=unit, values='noise_corrected', unit_label=UNIT_LABEL[unit], threshold=np.nan,
            mean_dist_ctrl=float(R['TNT3c'].mean().mean()), mean_dist_tnt=float(R['TNT'].mean().mean()),
            rho_vs_z_same_values=H.spearman(e, H.apl_effect(reading_rdms('nc_z')['TNT3c'].mean(), reading_rdms('nc_z')['TNT'].mean())),
            effect_min=float(e.min()), effect_max=float(e.max()), sd_ddof1=float(e.std()),
            **{k: v for k, v in r.items() if not isinstance(v, (pd.Series, dict))}))
        pairs[f'effect__{unit}__noise_corrected'] = e
        print(f"{unit:12s} noise-corrected: effect {rows[-1]['mean']:+.3f} [{rows[-1]['mean_lo']:+.3f}, {rows[-1]['mean_hi']:+.3f}], "
              f"ρ with z {rows[-1]['rho_ref']:.3f}", flush=True)
    du = pd.DataFrame(rows)
    du.to_csv(out / 'data_units.csv', index=False)
    ep = pd.DataFrame(pairs)
    ep.index.names = ['odor_a', 'odor_b']
    ep.to_csv(out / 'effect_by_unit_pairs.csv')
    cs = class_split_units(out)
    show = cs[(cs['quantity'] == 'effect') & (cs['term'] == 'total')]
    print(show.pivot_table(index='class', columns='unit', values='observed').round(4).to_string(), flush=True)
    print('class split checks:', cs.attrs['checks'], flush=True)
    # z, graded against the registry (the same draws: the mean's interval, the floor and the split-half ρ)
    z = du[(du['unit'] == 'z') & (du['values'] == 'graded')].iloc[0]
    reg = dict(apl_effect_mean=(z['mean'], H.data_target('apl_effect_mean')),
        apl_effect_rmse_floor=(z['rmse_floor'], H.data_target('apl_effect_rmse_floor')),
        apl_effect_rho_split_half=(z['split_half_rho'], H.data_target('apl_effect_rho_split_half')),
        apl_effect_rho_ceiling=(z['rho_ceiling'], H.data_target('apl_effect_rho_ceiling')),
        apl_effect_sd=(z['sd_ddof1'], H.data_target('apl_effect_sd')),
        ctrl_participation_ratio_resp=(z['cal_target'], H.data_target('ctrl_participation_ratio_resp')))
    lo, hi = H.data_target_ci('apl_effect_mean')
    reg['apl_effect_mean_ci'] = ((z['mean_lo'], z['mean_hi']), (lo, hi))
    print('z against the registry:', {k: (round(float(np.ravel(a)[0]), 6), round(float(np.ravel(b)[0]), 6)) for k, (a, b) in reg.items()}, flush=True)
    _provenance(out, 'B', dict(class_split_checks=cs.attrs['checks'], class_n_boot=CLASS_N_BOOT,
        z_vs_registry={k: [np.ravel(a).tolist(), np.ravel(b).tolist()] for k, (a, b) in reg.items()},
        cal_targets={u: cal_target(u) for u in UNITS}, started=t0.isoformat(timespec='seconds')))


def registry_entries(n_boot: int = 1000, seed: int = 0) -> List[Tuple[str, float, str, Optional[Tuple[float, float]]]]:
    """The data-target registry's entries of this experiment (data_targets.py, prefix `noise_apl_effect_`): (name, value,
    definition, 90 % interval or None). The per-pair APL effect with each KC's trial means in baseline-noise units (hal's
    data/yang_apl16_kc_noise_units/trialmean_long.parquet), with the statistics the registry's apl_effect_* entries have in
    Yang's z (`effect_resamples`, the same draws), the groups' effects, the effect's agreement with the z-based one, the
    thresholded and binarized versions (at noise_threshold), the noise-corrected mean, the two mean distances, the
    calibrated readout's target in the same unit and the split by odors per KC."""
    unit = 'noise'
    ref = unit_recording_rdms('z', 'graded')
    pre = 'noise_apl_effect_'
    lab = ("the per-pair APL effect (control flies' cosine distance minus the distance with APL output blocked (TNT), each "
        'the mean over recordings) with each KC\'s trial means in units of its baseline noise (Yang\'s z times the KC\'s '
        'whole-trace SD of Fc over its SD within the pre-odor windows)')
    iv = f'90 % interval over {n_boot} resamples of recordings (seed {seed}, as apl_effect_mean)'
    out = []
    g = effect_resamples(unit_recording_rdms(unit, 'graded'), ref=ref, n_boot=n_boot, seed=seed)
    e = g['effect']
    out += [(f'{pre}mean', g['mean'], f'{lab}: mean over the 120 pairs; {iv}', (g['mean_lo'], g['mean_hi'])),
        (f'{pre}mean_fly', g['mean'], f'{lab}: mean over the 120 pairs; 90 % interval over {n_boot} resamples of flies (a fly\'s '
         'hemispheres kept together)', (g['mean_fly_lo'], g['mean_fly_hi'])),
        (f'{pre}sd', float(e.std()), f'{lab}: standard deviation over the 120 pairs (ddof 1, as apl_effect_sd)', None),
        (f'{pre}sd_pattern', g['sd'], f'{lab}: root mean square over the 120 pairs of the effect minus its mean (ddof 0): the '
         'pattern part of a model with the right mean and no pattern', None),
        (f'{pre}min', float(e.min()), f'{lab}: smallest per-pair effect', None),
        (f'{pre}max', float(e.max()), f'{lab}: largest per-pair effect', None),
        (f'{pre}rmse_floor', g['rmse_floor'], f'floor of the APL-effect loss against {lab}: the mean over resampled data sets '
         '(recordings drawn with replacement within each genotype) of their loss against the measured effect; '
         'interval the 5th to 95th percentiles', (g['rmse_floor_lo'], g['rmse_floor_hi'])),
        (f'{pre}pattern_floor', g['pattern_floor'], f'floor of the APL-effect loss\'s pattern part (both means removed) against '
         f'{lab}, as {pre}rmse_floor', (g['pattern_floor_lo'], g['pattern_floor_hi'])),
        (f'{pre}rho_split_half', g['split_half_rho'], f'ρ on {lab} between random halves of the recordings (each genotype split '
         'in two; 500 splits)', (g['split_half_lo'], g['split_half_hi'])),
        (f'{pre}rho_ceiling', g['rho_ceiling'], f'ceiling on ρ on {lab}: sqrt of the Spearman-Brown-corrected split-half ρ', None),
        (f'{pre}ester_rel', g['ester_rel'], f'{lab}: the mean over the 15 pairs among 2h, IaA, pa, eb, ep and ms minus the mean '
         f'over all pairs; {iv}', (g['ester_rel_lo'], g['ester_rel_hi'])),
        (f'{pre}alcohol_rel', g['alcohol_rel'], f'{lab}: the mean over the 6 pairs among 1-8ol, 1-5ol, 1-6ol and 6al minus the '
         f'mean over all pairs; {iv}', (g['alcohol_rel_lo'], g['alcohol_rel_hi'])),
        (f'{pre}rho_vs_z', g['rho_ref'], f'ρ over the 120 pairs between {lab} and the same effect in Yang\'s z; {iv} (the same '
         'draws in both units)', (g['rho_ref_lo'], g['rho_ref_hi'])),
        (f'{pre}pattern_diff_vs_z', g['pattern_diff_ref'], f'root mean square over the 120 pairs of the difference between {lab} '
         f'and the same effect in Yang\'s z, both means removed; {iv}', (g['pattern_diff_ref_lo'], g['pattern_diff_ref_hi']))]
    R = unit_recording_rdms(unit, 'graded')
    out += [(f'{pre}ctrl_mean_dist', float(R['TNT3c'].mean().mean()), 'KCs, control flies: mean pair distance over the 120 pairs '
         'with each KC\'s trial means in baseline-noise units (1 - cosine per recording, mean over recordings)', None),
        (f'{pre}tnt_mean_dist', float(R['TNT'].mean().mean()), 'KCs, APL output blocked (TNT): the same', None),
        (f'{pre}cal_target', cal_target(unit), 'KCs, control flies: participation ratio of each odor\'s response vector across '
         'KCs, (sum x^2)^2 / (n sum x^4), on responders only (trial means in baseline-noise units at or below noise_threshold set '
         'to 0), mean over odors and recordings: the calibrated readout\'s target in baseline-noise units (as '
         'ctrl_participation_ratio_resp in Yang\'s z)', None)]
    for values in ('thresholded', 'binary'):
        v = effect_resamples(unit_recording_rdms(unit, values), n_boot=n_boot, seed=seed)
        vl = f'{lab}, {VALUES_LABEL[values]} at noise_threshold'
        out += [(f'{pre}{values}_mean', v['mean'], f'{vl}: mean over the 120 pairs; {iv}', (v['mean_lo'], v['mean_hi'])),
            (f'{pre}{values}_rmse_floor', v['rmse_floor'], f'floor of the APL-effect loss against {vl} (as {pre}rmse_floor)',
             (v['rmse_floor_lo'], v['rmse_floor_hi'])),
            (f'{pre}{values}_rho_ceiling', v['rho_ceiling'], f'ceiling on ρ on {vl} (as {pre}rho_ceiling)', None)]
    nc = reading_rdms('nc_noise')
    out.append((f'{pre}noise_corrected_mean', float(H.apl_effect(nc['TNT3c'].mean(), nc['TNT'].mean()).mean()), f'{lab}, from '
        'each recording\'s cosine corrected for trial-to-trial noise (noise_floor_trial_distances.analyse on the per-trial '
        'values in baseline-noise units): mean over the 120 pairs', None))
    cs = class_split_units(None, seed=seed)
    cs = cs[(cs['unit'] == unit) & (cs['term'] == 'total')]
    for cls in H.opk_class_labels():
        r = cs[(cs['quantity'] == 'effect') & (cs['class'] == cls)].iloc[0]
        out.append((f'{pre}class_{cls}', float(r['observed']), f'{lab}, split by the number of the 16 odors a KC responds to in each '
            f'condition (above noise_threshold; harness.apl_effect_by_class): the mean over the 120 pairs of the effect carried by '
            f'KCs of class {cls}; 90 % interval over {CLASS_N_BOOT} resamples of recordings', (float(r['lo']), float(r['hi']))))
        r = cs[(cs['quantity'] == 'var_share') & (cs['class'] == cls)].iloc[0]
        out.append((f'{pre}class_var_share_{cls}', float(r['observed']), f'{lab}, split as {pre}class_{cls}: the share of the '
            'per-pair effect\'s variance over pairs carried by that class (cov(class effect, effect) / var(effect))',
            (float(r['lo']), float(r['hi']))))
    return out


ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
# the units part C scores the archive against on raw counts (all of them), under the calibrated readout and binarized
# (Yang's z, the target as defined, and baseline-noise units)
C_UNITS_FIXED = ('z', 'noise')
_WORKER: Dict[str, object] = {}


def family_of(folder: str, tag: str) -> str:
    """A model family for "the best unflagged variant per family", from the tag (first rule that matches): the reference set
    (eight glomeruli at 1.5 with a per-claw threshold) with the firing ceiling (E69's refractory period and adaptation,
    or E69 part A's ref40_r* settings); a family without the gain with the ceiling; the reference set; a per-claw threshold
    without the gain; otherwise the experiment folder."""
    import re
    ceiling = 'refr' in tag or 'adapt' in tag or (folder == 'E69_kc_firing_ceiling' and tag.startswith('ref40_r'))
    gain = 'gain' in tag or tag.startswith('ref40')
    if ceiling and gain:
        return 'the reference set with the firing ceiling'
    if ceiling:
        return 'breadth-weighted APL, no gain, with the firing ceiling'
    if gain and re.search(r'thr\d', tag):
        return 'the reference set (glomerulus gain and per-claw threshold)'
    if re.search(r'(^|_)thr\d', tag):
        return 'a per-claw threshold without the gain'
    return f'other: {folder}'


def _init_worker(targets: dict) -> None:
    _WORKER.update(targets)


def _score_counts(job: Tuple[str, str]) -> dict:
    """Part C's readouts of one archived variant with spike counts: the calibrated readout (K set on the APL-on counts to
    each unit's `cal_target`, applied unchanged to APL off) against the data in that unit, and binarized (two spikes)
    against the data binarized in that unit."""
    folder, tag = job
    d = ARCHIVE / folder
    ps = [d / f'{tag}_{c}_spike_counts.parquet' for c in ('on', 'acute_off')]
    row = dict(folder=folder, tag=tag, has_counts=all(p.exists() for p in ps))
    if not row['has_counts']:
        return row
    try:
        s_on, s_off = (pd.read_parquet(p).astype(float) for p in ps)
        b_on, b_off = H.readout_pair_distances(s_on, 'binary', 2.0), H.readout_pair_distances(s_off, 'binary', 2.0)
        for u in C_UNITS_FIXED:
            K = H.calibrate_readout(s_on, 'saturating', _WORKER['cal'][u], stat='participation_ratio', responders_min=2.0)
            got = H.readout_stat(s_on, 'saturating', K, 'participation_ratio', 2.0)
            on, off = H.readout_pair_distances(s_on, 'saturating', K), H.readout_pair_distances(s_off, 'saturating', K)
            st = H.apl_effect_stats(on, off, _WORKER['graded'][u])
            row.update({f'cal_{u}_K': float(K), f'cal_{u}_K_at_bound': bool(abs(got - _WORKER['cal'][u]) > 0.02 * _WORKER['cal'][u]),
                f'cal_{u}_loss': st['apl_effect_rmse'], f'cal_{u}_pattern': st['apl_effect_rmse_centered'],
                f'cal_{u}_mean': st['apl_effect_mean'], f'cal_{u}_mean_error': st['apl_effect_mean_error'], f'cal_{u}_rho': st['apl_effect_spearman']})
            st = H.apl_effect_stats(b_on, b_off, _WORKER['binary'][u])
            row.update({f'bin_{u}_loss': st['apl_effect_rmse'], f'bin_{u}_pattern': st['apl_effect_rmse_centered'],
                f'bin_{u}_mean': st['apl_effect_mean'], f'bin_{u}_mean_error': st['apl_effect_mean_error'], f'bin_{u}_rho': st['apl_effect_spearman']})
    except Exception as e:
        row['error'] = repr(e)[:200]
    return row


def _rank_corr(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations across variants between the scores against Yang's z (the target as defined) and against each
    fixed unit, per readout and part, overall and among the variants inside the bands (baseline-noise units)."""
    rows = []
    ok = df['realism_flag'] == ''
    for readout, units in (('raw', UNITS[1:]), ('cal', C_UNITS_FIXED[1:]), ('bin', C_UNITS_FIXED[1:])):
        for u in units:
            for part in ('loss', 'pattern', 'rho'):
                a, b = f'{readout}_z_{part}', f'{readout}_{u}_{part}'
                if a not in df or b not in df:
                    continue
                for who, m in (('all', np.ones(len(df), dtype=bool)), ('inside the bands', ok.to_numpy())):
                    x = df.loc[m, [a, b]].dropna()
                    rows.append(dict(readout=readout, unit=u, part=part, variants=who, n=len(x),
                        spearman=float(x[a].corr(x[b], method='spearman')) if len(x) > 2 else np.nan))
    return pd.DataFrame(rows)


def _best(df: pd.DataFrame, by: str, col: str) -> pd.DataFrame:
    ok = df[(df['realism_flag'] == '') & df[col].notna()]
    idx = ok.groupby(by)[col].idxmin()
    return ok.loc[idx.to_numpy()].sort_values(col)


def common_error_units(out: Path) -> pd.DataFrame:
    """E72's eight wirings and their mean against the effect in baseline-noise units: per wiring and for the population
    (the mean over the wirings of the pair distances per condition) the loss, its parts and ρ on raw counts and under the
    calibrated readout (K at each unit's target), and the common error (the population's pattern error) recomputed against
    the baseline-noise effect beside the present one (z, K in z), with their correlation. Writes common_error_units.csv
    (per pair) and common_error_units_summary.csv."""
    import E72_wiring_population as E72
    rows, cols = [], {}
    for readout in ('raw', 'cal'):
        for u in C_UNITS_FIXED:
            labels, dists = e72_distances(readout, None if (readout == 'raw' or u == 'z') else cal_target(u))
            data = data_effect(u)
            for lab, d in zip(labels, dists):
                st = H.apl_effect_stats(d['on'], d['off'], data)
                rows.append(dict(readout=readout, unit=u, wiring=lab, loss=st['apl_effect_rmse'], pattern=st['apl_effect_rmse_centered'],
                    mean=st['apl_effect_mean'], rho=st['apl_effect_spearman']))
            md = E72.mean_distances(dists)
            st = H.apl_effect_stats(md['on'], md['off'], data)
            err = H.apl_effect_pair_parts(md['on'], md['off'], data)['error']
            cols[f'common_{readout}_{u}'] = err
            rows.append(dict(readout=readout, unit=u, wiring='population', n_wirings=len(labels), loss=st['apl_effect_rmse'],
                pattern=st['apl_effect_rmse_centered'], mean=st['apl_effect_mean'], rho=st['apl_effect_spearman'],
                common_rms=float(np.sqrt(np.mean(err ** 2))), err_ester=float(err[pair_mask(err.index, ESTER_GROUP)].mean()),
                err_alcohol=float(err[pair_mask(err.index, ALCOHOL_GROUP)].mean())))
    pp = pd.DataFrame(cols)
    pp.index.names = ['odor_a', 'odor_b']
    # E72's saved common error (calibrated readout, z) as the check of the recomputation
    saved = pd.read_csv(H.results_dir('E72_wiring_population') / 'common_error_pairs.csv').set_index(['odor_a', 'odor_b'])['common']
    a, b = H.align_pairs(pp['common_cal_z'], saved)
    summ = pd.DataFrame(rows)
    for readout in ('raw', 'cal'):
        m = (summ['readout'] == readout) & (summ['wiring'] == 'population')
        summ.loc[m, 'corr_with_z'] = [float(np.corrcoef(pp[f'common_{readout}_{u}'], pp[f'common_{readout}_z'])[0, 1])
            for u in summ.loc[m, 'unit']]
    summ.attrs['check_vs_saved_max_abs'] = float((a - b).abs().max())
    pp.to_csv(out / 'common_error_units.csv')
    summ.to_csv(out / 'common_error_units_summary.csv', index=False)
    return summ


def pipeline_units(out: Path) -> pd.DataFrame:
    """The saved pipeline rows through the trial noise (E72 part E and E75 part A, each experiment's
    partB/pairs_noise_long.csv; graded transform) re-scored as E72's part E scores them: each model alone and their
    population, the noisy three-trial means ('raw') and the deterministic distances ('det') against the effect from the
    data's raw distances, the noise-corrected ones ('corrected') against the effect from the data's corrected ones; in
    Yang's z (the data columns saved with the rows) and in baseline-noise units (this experiment's graded and
    noise-corrected effects). Writes pipeline_units.csv."""
    rows = []
    for exp in ('E72_wiring_population', 'E75_glomerulus_apl_profile'):
        p = H.results_dir(exp) / 'partB' / 'pairs_noise_long.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d['transform'] == 'graded']
        both = d[d['level'] == 'match_on+match_off']
        d = pd.concat([d[d['level'] != 'match_on+match_off'], both.assign(level='match_on'), both.assign(level='match_off')], ignore_index=True)
        noise_eff = {'raw': data_effect('noise'), 'det': data_effect('noise'),
            'corrected': H.apl_effect(reading_rdms('nc_noise')['TNT3c'].mean(), reading_rdms('nc_noise')['TNT'].mean())}
        for level, g in d.groupby('level'):
            models = sorted(g['model'].unique())
            groups = {m: [m] for m in models}
            if len(models) > 1:
                groups['population'] = models
            for gname, ms in groups.items():
                gg = g[g['model'].isin(ms)]
                on = gg[gg['condition'] == 'on'].groupby(['odor_a', 'odor_b']).mean(numeric_only=True)
                off = gg[gg['condition'] == 'acute_off'].groupby(['odor_a', 'odor_b']).mean(numeric_only=True)
                for col, dcol in (('raw', 'data_raw'), ('corrected', 'data_corrected'), ('det', 'data_raw')):
                    for u, data in (('z', H.apl_effect(on[dcol], off[dcol])), ('noise', noise_eff[col])):
                        st = H.apl_effect_stats(on[col], off[col], data)
                        rows.append(dict(experiment=exp, group=gname, n_models=len(ms), level=level, distances=col, unit=u,
                            data_effect_mean=float(data.mean()), loss=st['apl_effect_rmse'], pattern=st['apl_effect_rmse_centered'],
                            mean=st['apl_effect_mean'], rho=st['apl_effect_spearman']))
    res = pd.DataFrame(rows)
    res.to_csv(out / 'pipeline_units.csv', index=False)
    return res


def part_c(out: Path, jobs: int = 3) -> None:
    """Part C: the archive against the fixed-unit effects (by_variant_units.csv, rank_correlations_units.csv,
    best_by_family_units.csv, common_error_units.csv, pipeline_units.csv)."""
    from multiprocessing import Pool
    t0 = datetime.now(timezone.utc)
    pairs = archive_pairs()
    raw = score_against(pairs, {u: data_effect(u) for u in UNITS})
    raw = raw.rename(columns={**{c: f'raw_{c}' for c in raw.columns if c not in ('folder', 'tag', 'model_mean')},
        'model_mean': 'raw_mean'})
    targets = dict(cal={u: cal_target(u) for u in C_UNITS_FIXED}, graded={u: data_effect(u) for u in C_UNITS_FIXED},
        binary={u: data_effect(u, 'binary') for u in C_UNITS_FIXED})
    with Pool(jobs, initializer=_init_worker, initargs=(targets,)) as pool:
        cnt = pd.DataFrame(pool.map(_score_counts, list(pairs), chunksize=8))
    df = with_bands(raw.merge(cnt, on=['folder', 'tag'], how='left'))
    df.insert(2, 'family', [family_of(f, t) for f, t in zip(df['folder'], df['tag'])])
    df.to_csv(out / 'by_variant_units.csv', index=False)
    rc = _rank_corr(df)
    rc.to_csv(out / 'rank_correlations_units.csv', index=False)
    best = []
    for col in ('raw_z_loss', 'raw_noise_loss', 'raw_noise_pattern', 'cal_z_loss', 'cal_noise_loss', 'bin_z_loss', 'bin_noise_loss'):
        for by in ('family', 'folder'):
            b = _best(df, by, col)
            b = b.assign(ranked_by=col, group_by=by)
            best.append(b)
    bb = pd.concat(best, ignore_index=True)
    bb.to_csv(out / 'best_by_family_units.csv', index=False)
    ce = common_error_units(out)
    pl = pipeline_units(out)
    ok = df[df['realism_flag'] == '']
    print(f"{len(df)} variants ({int(df['has_counts'].fillna(False).sum())} with spike counts; {len(ok)} inside the bands)", flush=True)
    print(rc[rc['part'] != 'rho'].round(3).to_string(), flush=True)
    show = ['folder', 'tag', 'raw_mean', 'raw_z_loss', 'raw_z_pattern', 'raw_noise_loss', 'raw_noise_pattern', 'raw_noise_rho',
        'cal_noise_loss', 'cal_noise_pattern', 'rate_on', 'rate_off']
    for col in ('raw_noise_loss', 'cal_noise_loss', 'bin_noise_loss'):
        b = bb[(bb['ranked_by'] == col) & (bb['group_by'] == 'family')]
        print(f'best unflagged per family, by {col}:', flush=True)
        print(b[['family'] + show].round(3).to_string(index=False, max_colwidth=60), flush=True)
    print(ce.round(3).to_string(), flush=True)
    print('common error recomputed vs saved (z, calibrated):', ce.attrs['check_vs_saved_max_abs'], flush=True)
    if len(pl):
        print(pl[(pl['group'] == 'population') & pl['distances'].isin(['raw', 'det'])].round(3).to_string(), flush=True)
    _provenance(out, 'C', dict(n_variants=len(df), cal_targets=targets['cal'], jobs=jobs,
        common_error_check_vs_saved_max_abs=ce.attrs['check_vs_saved_max_abs'], started=t0.isoformat(timespec='seconds')))


# part E: the bases ((E69 base key for the bouton base, (experiment, tag) of the fit), label) and the axes
E_BASES = {
    'c2': ('e67cal', ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3'), 'C2, the breadth-2 APL-effect fit (E67)'),
    'ceiling2': ('e67cal', ('E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.1114t1000_k2'),
        "the same family with the firing ceiling at strength 2 (E69's fit 2)"),
}
BLOCK_FACTORS = (0.75, 0.5, 0.25, 0.125)
THR_FACTORS = (0.8, 0.9, 1.1, 1.2)


def e_points() -> List[Tuple[str, str, float, str]]:
    """Part E's points per base: (point id, axis, factor, condition): the base with APL on and off; the graded block (APL's
    output scales, APL->KC and APL->bouton, times each of BLOCK_FACTORS, the inputs to APL unchanged); excitability (the
    somatic threshold times each of THR_FACTORS, with APL on and with APL off)."""
    pts = [('base_on', 'base', 1.0, 'on'), ('base_off', 'base', 1.0, 'off')]
    pts += [(f'block{f:g}', 'block', f, 'on') for f in BLOCK_FACTORS]
    pts += [(f'thr{f:g}_{c}', 'thr', f, c) for f in THR_FACTORS for c in ('on', 'off')]
    return pts


def _point_params(model, axis: str, factor: float, cond: str) -> dict:
    p = dict(model.p)
    if cond == 'off':
        p = model.off_params(p)
    if axis == 'block':
        p.update(wAPLKC_scale=p['wAPLKC_scale'] * factor, wAPLPN_scale=p['wAPLPN_scale'] * factor)
    elif axis == 'thr':
        p['fixed_thr'] = p['fixed_thr'] * factor
    return p


def _archived_counts(exp: str, tag: str) -> Dict[str, Optional[pd.DataFrame]]:
    """The fit's saved spike counts per condition ('on', 'acute_off'): this host's outputs, else the shared archive."""
    out = {}
    for cond in ('on', 'acute_off'):
        out[cond] = None
        for d in (H.HERE / 'outputs' / exp, ARCHIVE / exp):
            q = d / f'{tag}_{cond}_spike_counts.parquet'
            if q.exists():
                out[cond] = H.abbrev_columns(pd.read_parquet(q).astype(float))[list(H.APL16_ODORS)]
                break
    return out


def part_e(out: Path, threads: int = 12, only: Optional[Sequence[str]] = None, check_only: bool = False,
           bases: Optional[Sequence[str]] = None, base2: Optional[str] = None) -> None:
    """Part E: the fixed-scale runner on E69's model wrapper (`E69_kc_firing_ceiling.Model` with `ref`, the fit at its fitted
    parameters with its own firing ceiling). Per base (`E_BASES`; `base2` 'EXP:TAG' replaces the ceiling base by another
    fit of the same base, e.g. fit 0's optimum if it moved): the base re-run with APL on and off must reproduce its archived
    spike counts exactly (else the base is skipped), then each of `e_points()` (`only`: the point ids to run, a smoke
    test). Each point's responses and spike counts go to outputs/E76_data_side_target/partE/<base>_<point>_*.parquet,
    one row per point to model_axes_runs.csv (results), with its absolute parameters, response rate, odors-per-KC
    distribution and mean distances; per-pair distances (raw counts, and under the calibrated readout with K held at the
    base's own calibration in z and in baseline-noise units) to model_axes_pairs.csv. `part_e_tables` then reads them."""
    import time
    import E69_kc_firing_ceiling as E69
    from apl_effect_scores import odors_per_kc, OPK_BINS
    odir = H.outputs_dir(EXPERIMENT) / 'partE'
    odir.mkdir(parents=True, exist_ok=True)
    spec = dict(E_BASES)
    if base2:
        exp, _, tag = base2.partition(':')
        spec['ceiling2'] = (spec['ceiling2'][0], (exp, tag), f'the ceiling family at strength 2 ({exp}:{tag})')
    runs_p, pairs_p = out / 'model_axes_runs.csv', out / 'model_axes_pairs.csv'
    checks = {}
    for b in (bases or list(spec)):
        key, (exp, tag), label = spec[b]
        t0 = time.time()
        m = E69.Model(key, ref=(exp, tag))
        on, off = m.sim(m.p), m.sim(m.off_params())
        saved = _archived_counts(exp, tag)
        ck = {}
        for cond, o in (('on', on), ('acute_off', off)):
            a = saved[cond]
            if a is None:
                ck[f'{cond}_archived'] = False
                continue
            ck[f'{cond}_archived'] = True
            ck[f'{cond}_index_equal'] = bool(a.index.equals(o[1].index))
            ck[f'{cond}_mismatch'] = int((a.to_numpy() != o[1].to_numpy(dtype=float)).sum())
        ok = all(ck.get(f'{c}_archived') and ck.get(f'{c}_index_equal') and ck.get(f'{c}_mismatch') == 0 for c in ('on', 'acute_off'))
        checks[b] = dict(experiment=exp, tag=tag, reproduced=ok, seconds=time.time() - t0, threads=threads, params=m.p,
            kc=m.kws.get('kc_refractory_period'), **ck)
        print(f'{b}: {exp}:{tag} re-run {"reproduces" if ok else "does NOT reproduce"} its archived counts {ck} '
              f'({time.time() - t0:.0f} s)', flush=True)
        if not ok or check_only:
            continue
        s_on = on[1].astype(float)
        K = {u: H.calibrate_readout(s_on, 'saturating', cal_target(u), stat='participation_ratio', responders_min=2.0)
            for u in C_UNITS_FIXED}
        done = set()
        if runs_p.exists():
            r0 = pd.read_csv(runs_p)
            done = set(r0.loc[(r0['base'] == b) & (r0['experiment'] == exp) & (r0['tag'] == tag), 'point'])
        for pid, axis, factor, cond in e_points():
            # (the base points are simulated by the reproduction check anyway, so `only` keeps them)
            if (only and pid not in only and not pid.startswith('base_')) or pid in done:
                continue
            t1 = time.time()
            if pid == 'base_on':
                r, sc = on
            elif pid == 'base_off':
                r, sc = off
            else:
                r, sc = m.sim(_point_params(m, axis, factor, cond))
            sc = sc.astype(float)
            r.to_parquet(odir / f'{b}_{pid}_responses.parquet')
            sc.to_parquet(odir / f'{b}_{pid}_spike_counts.parquet')
            pp = _point_params(m, axis, factor, cond)
            d_raw = H.readout_pair_distances(sc, 'raw')
            row = dict(base=b, experiment=exp, tag=tag, point=pid, axis=axis, factor=factor, condition=cond,
                rate=float(r.to_numpy(dtype=float).mean()), mean_dist_raw=float(d_raw.mean()), max_count=float(sc.to_numpy().max()),
                seconds=time.time() - t1, threads=threads, **{f'K_{u}': float(v) for u, v in K.items()},
                **{f'opk_{bb}': float(v) for bb, v in zip(OPK_BINS, odors_per_kc(r))}, **{k: float(v) for k, v in pp.items()})
            prs = [d_raw.rename('distance').to_frame().assign(readout='raw')]
            for u, k in K.items():
                d = H.readout_pair_distances(sc, 'saturating', k)
                row[f'mean_dist_cal_{u}'] = float(d.mean())
                prs.append(d.rename('distance').to_frame().assign(readout=f'cal_{u}'))
            pr = pd.concat(prs).reset_index().assign(base=b, experiment=exp, tag=tag, point=pid)
            pr.to_csv(pairs_p, mode='a', header=not pairs_p.exists(), index=False)
            pd.DataFrame([row]).to_csv(runs_p, mode='a', header=not runs_p.exists(), index=False)
            print(f"{b} {pid}: rate {row['rate']:.4f}, mean distance raw {row['mean_dist_raw']:.3f}, largest count "
                  f"{row['max_count']:.0f} ({row['seconds']:.0f} s)", flush=True)
    # (the reproduction checks of earlier runs are kept, keyed by base, beside this run's settings)
    prev = json.loads((out / 'provenance_E.json').read_text()) if (out / 'provenance_E.json').exists() else {}
    all_checks = dict(prev.get('checks', {}), **checks)
    runs = prev.get('runs', []) + [dict(time=datetime.now(timezone.utc).isoformat(timespec='seconds'), bases=list(bases or spec),
        only=list(only) if only else None, check_only=check_only, threads=threads, omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
        omp_thread_limit=os.environ.get('OMP_THREAD_LIMIT'))]
    _provenance(out, 'E', dict(checks=all_checks, runs=runs, block_factors=BLOCK_FACTORS, thr_factors=THR_FACTORS,
        bases={k: list(v[1]) for k, v in spec.items()}))


def _interp_pairs(points: pd.DataFrame, pairs: Dict[str, pd.Series], x: float) -> Tuple[pd.Series, float, bool]:
    """The pair distances at response-rate ratio `x` along an axis: `points` (point, ratio) sorted by ratio, `pairs`
    {point: distances}; linear interpolation between the two neighbouring points, or the nearest end when `x` is outside
    the axis's range. Returns (distances, the ratio used, whether `x` was reachable)."""
    pts = points.sort_values('ratio')
    r = pts['ratio'].to_numpy()
    if x <= r[0] or x >= r[-1]:
        i = 0 if x <= r[0] else len(r) - 1
        return pairs[pts['point'].iloc[i]], float(r[i]), bool(np.isclose(x, r[i]))
    j = int(np.searchsorted(r, x))
    w = (x - r[j - 1]) / (r[j] - r[j - 1])
    a, b = H.align_pairs(pairs[pts['point'].iloc[j - 1]], pairs[pts['point'].iloc[j]])
    return (1 - w) * a + w * b, float(x), True


def part_e_tables(out: Path) -> Dict[str, pd.DataFrame]:
    """Part E's reading from model_axes_runs.csv and model_axes_pairs.csv (no simulations): model_axes.csv (per base, axis
    point and readout: the response rate and its ratio to the base's APL-on rate, the mean distance, the effect from the
    base's APL-on distances against the data's eight, six and two (loss, pattern part, ρ) in z and in baseline-noise units,
    and the step from the previous point of the axis with its ρ against the data's two steps, the control flies to the six
    and the six to the two); populations.csv (the population reading: the APL-off side as six pseudo-flies at the point of
    the axis whose APL-off / APL-on response-rate ratio is the six's and two at the two's, linearly interpolated between
    points, or the axis's nearest end when out of reach; on the excitability axis the APL-on side also over the control
    recordings' rates; its loss with parts against the eight's effect beside the homogeneous model's, and each state
    against its own group's effect). The readouts are raw counts and the calibrated readout with K held at the base's own
    calibration ('cal_z', 'cal_noise'), each scored against the data in the matching unit (raw against both)."""
    runs = pd.read_csv(out / 'model_axes_runs.csv')
    prs = pd.read_csv(out / 'model_axes_pairs.csv')
    rec = pd.read_csv(out / 'recordings.csv')
    two = [rec_label(t) for t in _actual_two('TNT')]
    targets = subset_targets(('z', 'noise'))
    steps_data = {}
    for u in ('z', 'noise'):
        R = reading_rdms(u)
        six = R['TNT'].drop(index=_actual_two('TNT')).mean()
        steps_data[u] = dict(step1=H.canonical_pairs(R['TNT3c'].mean() - six), step2=H.canonical_pairs(six - R['TNT'].loc[_actual_two('TNT')].mean()))
    # the data's APL-off / APL-on response-rate ratios: the six's and the two's mean rate over the control flies' mean
    ctrl = rec[rec['genotype'] == 'TNT3c']
    tnt = rec[rec['genotype'] == 'TNT']
    ratio_data = {}
    for u in ('z', 'noise'):
        c = float(ctrl[f'rate_{u}'].mean())
        ratio_data[u] = dict(six=float(tnt.loc[~tnt['recording'].isin(two), f'rate_{u}'].mean()) / c,
            two=float(tnt.loc[tnt['recording'].isin(two), f'rate_{u}'].mean()) / c, ctrl=(ctrl[f'rate_{u}'] / c).tolist())
    rows, pops = [], []
    for (b, exp, tag), g in runs.groupby(['base', 'experiment', 'tag'], sort=False):
        P = prs[(prs['base'] == b) & (prs['experiment'] == exp) & (prs['tag'] == tag)]
        rate_on = float(g.loc[g['point'] == 'base_on', 'rate'].iloc[0])
        g = g.assign(ratio=g['rate'] / rate_on)
        for readout in sorted(P['readout'].unique()):
            dd = {pt: H.canonical_pairs(x.set_index(['odor_a', 'odor_b'])['distance']) for pt, x in P[P['readout'] == readout].groupby('point')}
            if 'base_on' not in dd or 'base_off' not in dd:
                continue
            units = ('z', 'noise') if readout == 'raw' else (readout.split('_', 1)[1],)
            # the axes as ordered sequences of points: the block from APL on (factor 1) to APL off (factor 0); the
            # excitability axis with APL off (and with APL on) from the highest threshold to the lowest
            seqs = {'block': ['base_on'] + [f'block{f:g}' for f in BLOCK_FACTORS] + ['base_off'],
                'thr_off': [f'thr{f:g}_off' for f in sorted(THR_FACTORS, reverse=True) if f > 1] + ['base_off']
                    + [f'thr{f:g}_off' for f in sorted(THR_FACTORS, reverse=True) if f < 1],
                'thr_on': [f'thr{f:g}_on' for f in sorted(THR_FACTORS, reverse=True) if f > 1] + ['base_on']
                    + [f'thr{f:g}_on' for f in sorted(THR_FACTORS, reverse=True) if f < 1]}
            for axis, seq in seqs.items():
                seq = [x for x in seq if x in dd]
                for i, pt in enumerate(seq):
                    gr = g[g['point'] == pt].iloc[0]
                    row = dict(base=b, experiment=exp, tag=tag, readout=readout, axis=axis, point=pt, factor=float(gr['factor']),
                        rate=float(gr['rate']), ratio=float(gr['ratio']), mean_dist=float(dd[pt].mean()),
                        **{f'opk_{k}': float(gr[f'opk_{k}']) for k in ('0', '1', '2_3', '4_7', '8plus')})
                    e = H.apl_effect(dd['base_on'], dd[pt])
                    for u in units:
                        for name in ('eight', 'six', 'two'):
                            st = H.apl_effect_stats(dd['base_on'], dd[pt], targets[f'{name}_{u}'])
                            row.update({f'{name}_{u}_loss': st['apl_effect_rmse'], f'{name}_{u}_pattern': st['apl_effect_rmse_centered'],
                                f'{name}_{u}_rho': st['apl_effect_spearman']})
                        if i:
                            stp = H.canonical_pairs(dd[seq[i - 1]] - dd[pt])
                            row[f'step_mean'] = float(stp.mean())
                            for k, v in steps_data[u].items():
                                row[f'step_rho_{k}_{u}'] = H.spearman(stp, v)
                    row['effect_mean'] = float(e.mean())
                    rows.append(row)
            # the population reading
            for u in units:
                rd = ratio_data[u]
                pts_off = pd.DataFrame([dict(point=x, ratio=float(g.loc[g['point'] == x, 'ratio'].iloc[0])) for x in seqs['block'] if x in dd])
                pts_thr = pd.DataFrame([dict(point=x, ratio=float(g.loc[g['point'] == x, 'ratio'].iloc[0])) for x in seqs['thr_off'] if x in dd])
                pts_thr_on = pd.DataFrame([dict(point=x, ratio=float(g.loc[g['point'] == x, 'ratio'].iloc[0])) for x in seqs['thr_on'] if x in dd])
                homog = {name: H.apl_effect_stats(dd['base_on'], dd['base_off'], targets[f'{name}_{u}']) for name in ('eight', 'six', 'two')}
                for axis, pts in (('block', pts_off), ('thr', pts_thr)):
                    if len(pts) < 2:
                        continue
                    d6, x6, ok6 = _interp_pairs(pts, dd, rd['six'])
                    d2, x2, ok2 = _interp_pairs(pts, dd, rd['two'])
                    a6, a2 = H.align_pairs(d6, d2)
                    off_pop = (6 * a6 + 2 * a2) / 8
                    on_sides = {'homogeneous APL on': dd['base_on']}
                    if axis == 'thr' and len(pts_thr_on) >= 2:
                        ons = [_interp_pairs(pts_thr_on, dd, x)[0] for x in rd['ctrl']]
                        on_sides['APL on over the control recordings\' rates'] = pd.concat(ons, axis=1).mean(axis=1)
                    for on_name, on in on_sides.items():
                        st = H.apl_effect_stats(on, off_pop, targets[f'eight_{u}'])
                        pops.append(dict(base=b, experiment=exp, tag=tag, readout=readout, unit=u, axis=axis, apl_on_side=on_name,
                            ratio_six=rd['six'], ratio_two=rd['two'], ratio_used_six=x6, ratio_used_two=x2, reachable_six=ok6, reachable_two=ok2,
                            loss=st['apl_effect_rmse'], pattern=st['apl_effect_rmse_centered'], mean=st['apl_effect_mean'], rho=st['apl_effect_spearman'],
                            homog_loss=homog['eight']['apl_effect_rmse'], homog_pattern=homog['eight']['apl_effect_rmse_centered'],
                            homog_mean=homog['eight']['apl_effect_mean'], homog_rho=homog['eight']['apl_effect_spearman'],
                            six_state_vs_six_loss=H.apl_effect_stats(on, d6, targets[f'six_{u}'])['apl_effect_rmse'],
                            six_state_vs_six_rho=H.apl_effect_stats(on, d6, targets[f'six_{u}'])['apl_effect_spearman'],
                            two_state_vs_two_loss=H.apl_effect_stats(on, d2, targets[f'two_{u}'])['apl_effect_rmse'],
                            two_state_vs_two_rho=H.apl_effect_stats(on, d2, targets[f'two_{u}'])['apl_effect_spearman'],
                            homog_vs_six_loss=homog['six']['apl_effect_rmse'], homog_vs_two_loss=homog['two']['apl_effect_rmse']))
    ma, po = pd.DataFrame(rows), pd.DataFrame(pops)
    ma.to_csv(out / 'model_axes.csv', index=False)
    po.to_csv(out / 'populations.csv', index=False)
    # the recordings' ranges beside the axes'
    ranges = rec.groupby(rec['genotype'].where(rec['genotype'] == 'TNT3c', np.where(rec['recording'].isin(two), 'TNT two', 'TNT six'))).agg(
        mean_dist_z_min=('mean_dist_z', 'min'), mean_dist_z_max=('mean_dist_z', 'max'), mean_dist_noise_min=('mean_dist_noise', 'min'),
        mean_dist_noise_max=('mean_dist_noise', 'max'), rate_noise_min=('rate_noise', 'min'), rate_noise_max=('rate_noise', 'max'),
        rate_z_min=('rate_z', 'min'), rate_z_max=('rate_z', 'max'))
    ranges.to_csv(out / 'recording_ranges.csv')
    return dict(model_axes=ma, populations=po, ranges=ranges)


def _ci(r, k, spec='+.3f') -> str:
    lo, hi = r.get(f'{k}_lo', np.nan), r.get(f'{k}_hi', np.nan)
    return f"{_f(r[k], spec)} [{_f(lo, spec)}, {_f(hi, spec)}]"


def write_table(out: Path) -> None:
    """comparison_table.md from the parts' tables (whichever exist): the unit (part B), the recordings and the step tests
    (part D), the subsets with the key models and E72's common error, the half-split strata, the archive in fixed units
    (part C). Numbers as computed; the experiment's note interprets them."""
    L = ['# E76: the data side of the main target (tables)', '']
    p = out / 'data_units.csv'
    if p.exists():
        du = pd.read_csv(p)
        L += ['## The unit (part B)', '', 'The KC data\'s per-pair APL effect with each KC\'s trial means in each unit; intervals 90 % over 1000 '
              'resamples of recordings (flies: of flies); floors and the split-half ρ as the registry\'s apl_effect_* entries.', '',
              '| unit | values | mean distance, control flies / APL output blocked (TNT) | mean APL effect [recordings] | [flies] | sd over pairs '
              '| floor of the loss (pattern part) | split-half ρ (ceiling) | ρ with the z-based effect | pattern difference from it '
              '| ester group / alcohol group, relative |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for _, r in du.iterrows():
            L.append(f"| {UNIT_LABEL.get(r['unit'], r['unit'])} | {r['values']} | {_f(r['mean_dist_ctrl'])} / {_f(r['mean_dist_tnt'])} | "
                f"{_ci(r, 'mean')} | [{_f(r['mean_fly_lo'], '+.3f')}, {_f(r['mean_fly_hi'], '+.3f')}] | {_f(r['sd'])} | "
                f"{_f(r['rmse_floor'])} ({_f(r['pattern_floor'])}) | {_f(r.get('split_half_rho'))} ({_f(r.get('rho_ceiling'))}) | "
                f"{_ci(r, 'rho_ref', '.3f')} | {_ci(r, 'pattern_diff_ref', '.3f')} | {_ci(r, 'ester_rel')} / {_ci(r, 'alcohol_rel')} |")
        L.append('')
    p = out / 'class_split_units.csv'
    if p.exists():
        cs = pd.read_csv(p, dtype={'class': str})
        t = cs[(cs['term'] == 'total') & (cs['quantity'].isin(['effect', 'var_share']))]
        L += ['## The effect by odors per KC (part B)', '', '| class (odors per KC) | Yang\'s z: mean effect | share of the variance '
              '| baseline-noise units: mean effect | share of the variance |', '|---|---:|---:|---:|---:|']
        for cls in list(H.opk_class_labels()) + ['all']:
            cells = []
            for u in ('z', 'noise'):
                for q in ('effect', 'var_share'):
                    x = t[(t['unit'] == u) & (t['quantity'] == q) & (t['class'] == cls)]
                    cells.append('–' if not len(x) else f"{x.iloc[0]['observed']:+.3f} [{x.iloc[0]['lo']:+.3f}, {x.iloc[0]['hi']:+.3f}]")
            L.append(f'| {cls} | ' + ' | '.join(cells) + ' |')
        L.append('')
    p = out / 'recordings.csv'
    if p.exists():
        rec = pd.read_csv(p)
        L += ['## The recordings (part D)', '', '| genotype | recording | KCs | mean distance (noise-corrected) | in baseline-noise units '
              '| response rate, z / baseline-noise units | ester group / alcohol group distance | effect against the control flies\' mean '
              '(sd over pairs; ester / alcohol relative) | induction (h) | median scale ratio | label group |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for _, r in rec.iterrows():
            eff = '' if r['genotype'] != 'TNT' else (f"{r['effect_mean_z']:+.3f} ({r['effect_sd_z']:.3f}; {r['effect_ester_rel_z']:+.3f} / "
                f"{r['effect_alcohol_rel_z']:+.3f})" + (' strongest' if r.get('strongest') in (True, 'True') else ''))
            L.append(f"| {'control flies' if r['genotype'] == 'TNT3c' else 'APL output blocked (TNT)'} | {r['recording']} | {r['n_kcs']} | "
                f"{r['mean_dist_z']:.3f} ({r['mean_dist_nc_z']:.3f}) | {r['mean_dist_noise']:.3f} | {r['rate_z']:.3f} / {r['rate_noise']:.3f} | "
                f"{r['ester_dist_z']:.3f} / {r['alcohol_dist_z']:.3f} | {eff} | {r['induction_hours']:g} | {r['median_scale_ratio']:.2f} | {r['label_group']} |")
        L.append('')
    p = out / 'axis_tests.csv'
    if p.exists():
        a = pd.read_csv(p)
        L += ['## The step tests (part D)', '', 'With APL output blocked (TNT): step 1 the control flies\' mean minus the six typical recordings\' '
              'mean, step 2 the six\'s mean minus the two strongest\'s; among the control recordings step 1 is the APL effect with the other '
              'nine as the control side and step 2 the nine minus the two control recordings with the lowest mean distance. Rank among '
              'every split into two and the rest (1 the lowest; 28 splits with APL output blocked, 55 among the controls); the null\'s '
              'median and 5th / 95th percentiles over those splits; intervals over recordings (the two kept).', '',
              '| reading | recordings | ρ between the steps [recordings] | rank | the splits\' median [5 %, 95 %] | the two\'s own steps, ρ '
              '[recordings] | rank | the splits\' median | step 2, mean | rank |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for _, r in a.iterrows():
            L.append(f"| {READING_LABEL[r['reading']]} | {'APL output blocked (TNT)' if r['genotype'] == 'TNT' else 'control flies'} | "
                f"{_ci(r, 'rho_steps', '.3f')} | {r['rank_rho_steps']}/{r['n_splits']} | {r['null_median_rho_steps']:.3f} "
                f"[{r['null_lo_rho_steps']:.3f}, {r['null_hi_rho_steps']:.3f}] | {_ci(r, 'rho_own', '.3f')} | {r['rank_rho_own']}/{r['n_splits']} | "
                f"{r['null_median_rho_own']:.3f} | {r['step2_mean']:+.3f} | {r['rank_step2_mean']}/{r['n_splits']} |")
        L.append('')
    p, q = out / 'subset_floors.csv', out / 'subsets_vs_archive.csv'
    if p.exists() and q.exists():
        fl = pd.read_csv(p)
        sc = pd.read_csv(q, keep_default_na=False, na_values=['']).set_index(['folder', 'tag'])
        L += ['## The six typical and the two strongest recordings with APL output blocked (TNT) (part D)', '',
              '| unit | APL-off side | mean APL effect [recordings] | sd over pairs | ester / alcohol relative | floor (pattern part) '
              '| split-half ρ (ceiling) |', '|---|---|---:|---:|---:|---:|---:|']
        for _, r in fl.iterrows():
            L.append(f"| {r['unit']} | {r['subset']} | {_ci(r, 'mean')} | {r['sd']:.3f} | {r['ester_rel']:+.3f} / {r['alcohol_rel']:+.3f} | "
                f"{r['rmse_floor']:.3f} ({r['pattern_floor']:.3f}) | {_f(r.get('split_half_rho'))} ({_f(r.get('rho_ceiling'))}) |")
        L += ['', 'Key models on raw spike counts against each side (loss / pattern part / ρ on the per-pair APL effect):', '',
              '| model | mean APL effect | eight, z | six, z | two, z | eight, baseline-noise units | six | two |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for short, lab, f, tg in KEY_MODELS:
            if (f, tg) in sc.index:
                r = sc.loc[(f, tg)]
                L.append(f'| {lab} | {r.model_mean:+.3f} | ' + ' | '.join(f"{r[n + '_loss']:.3f} / {r[n + '_pattern']:.3f} / {r[n + '_rho']:.2f}"
                    for n in ('eight_z', 'six_z', 'two_z', 'eight_noise', 'six_noise', 'two_noise')) + ' |')
        ok = sc[sc['realism_flag'] == ''].reset_index()
        L += ['', f'Best inside the bands (baseline-noise units) of {len(sc)} archived variants ({len(ok)} inside):', '']
        for n in ('eight_z', 'six_z', 'two_z', 'eight_noise', 'six_noise', 'two_noise'):
            b = ok.sort_values(f'{n}_loss').head(3)
            L.append(f'- against {n}: ' + '; '.join(f"{r.folder}:{r.tag} {r[n + '_loss']:.3f} / {r[n + '_pattern']:.3f} / {r[n + '_rho']:.2f}"
                for _, r in b.iterrows()))
        L.append('')
    p = out / 'common_error_subsets_summary.csv'
    if p.exists():
        ce = pd.read_csv(p)
        L += ['E72\'s common error (the eight wirings\' mean, calibrated readout) against each side: rms, the ester and alcohol groups\' '
              'mean errors, its correlation with the common error against the eight in z:', '', '| target | rms | ester | alcohol | correlation |',
              '|---|---:|---:|---:|---:|']
        L += [f"| {r['target']} | {r['rms']:.3f} | {r['err_ester']:+.3f} | {r['err_alcohol']:+.3f} | {r['corr_with_reference']:.3f} |" for _, r in ce.iterrows()]
        L.append('')
    p = out / 'split_halves.csv'
    if p.exists():
        sh = pd.read_csv(p)
        L += ['## Half-splits for E75 (part D)', '', f'{len(sh)} half-splits (seed {int(sh["seed"].iloc[0])}; the first 100 are E75 D1\'s):', '',
              '| both strongest in one half | splits | of E75\'s 100 | ρ between the halves, median | share below 0.45 | pattern difference, '
              'median | alcohol group\'s relative effect, difference between halves, median |', '|---|---:|---:|---:|---:|---:|---:|']
        for k, g in sh.groupby('strongest_together'):
            L.append(f"| {'yes' if k else 'no'} | {len(g)} | {int((g['split'] < 100).sum())} | {g['rho_halves'].median():.3f} | "
                f"{(g['rho_halves'] < 0.45).mean():.2f} | {g['pattern_diff'].median():.3f} | {g['alcohol_rel_diff'].median():.3f} |")
        L.append('')
    p = out / 'by_variant_units.csv'
    if p.exists():
        bv = pd.read_csv(p, keep_default_na=False, na_values=[''])
        L += ['## The archive in fixed units (part C)', '']
        rc = pd.read_csv(out / 'rank_correlations_units.csv')
        rc = rc[(rc['part'] != 'rho')]
        L += ['Spearman correlation across variants between the score against Yang\'s z and against the unit:', '',
              '| readout | unit | part | all variants | inside the bands |', '|---|---|---|---:|---:|']
        for (ro, u, part), g in rc.groupby(['readout', 'unit', 'part'], sort=False):
            v = g.set_index('variants')['spearman']
            L.append(f"| {ro} | {u} | {part} | {v.get('all', np.nan):.3f} | {v.get('inside the bands', np.nan):.3f} |")
        bb = pd.read_csv(out / 'best_by_family_units.csv', keep_default_na=False, na_values=[''])
        for col in ('raw_noise_loss', 'cal_noise_loss', 'bin_noise_loss'):
            b = bb[(bb['ranked_by'] == col) & (bb['group_by'] == 'family')].sort_values(col)
            ro = col.split('_')[0]
            L += ['', f'Best inside the bands per family, ranked by {col} (loss / pattern part / ρ against the effect in baseline-noise units; '
                  'the same against Yang\'s z):', '', '| family | variant | mean APL effect | baseline-noise units | Yang\'s z | response rates on / off |',
                  '|---|---|---:|---:|---:|---:|']
            for _, r in b.iterrows():
                L.append(f"| {r['family']} | {r['folder']}:{r['tag']} | {_f(r.get(ro + '_mean' if ro != 'raw' else 'raw_mean'), '+.3f')} | "
                    f"{_f(r[ro + '_noise_loss'])} / {_f(r[ro + '_noise_pattern'])} / {_f(r[ro + '_noise_rho'], '.2f')} | "
                    f"{_f(r[ro + '_z_loss'])} / {_f(r[ro + '_z_pattern'])} / {_f(r[ro + '_z_rho'], '.2f')} | {_f(r['rate_on'])} / {_f(r['rate_off'])} |")
        L.append('')
        p2 = out / 'common_error_units_summary.csv'
        if p2.exists():
            ce = pd.read_csv(p2)
            ce = ce[ce['wiring'] == 'population']
            L += ['E72\'s eight wirings\' mean (the population) against each unit:', '', '| readout | unit | loss | pattern part | mean APL effect '
                  '| ρ | common error rms | ester / alcohol | correlation with z\'s |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
            L += [f"| {r['readout']} | {r['unit']} | {r['loss']:.3f} | {r['pattern']:.3f} | {r['mean']:+.3f} | {r['rho']:.2f} | {r['common_rms']:.3f} | "
                  f"{r['err_ester']:+.3f} / {r['err_alcohol']:+.3f} | {_f(r.get('corr_with_z'))} |" for _, r in ce.iterrows()]
            L.append('')
    p = out / 'populations.csv'
    if p.exists():
        po = pd.read_csv(p)
        L += ['## The model\'s axes: the population reading (part E)', '', '| base | readout | unit | axis | APL-on side | ratio six / two (used) '
              '| loss / pattern / mean / ρ | homogeneous | six\'s state vs the six | two\'s state vs the two |', '|---|---|---|---|---|---:|---:|---:|---:|---:|']
        for _, r in po.iterrows():
            L.append(f"| {r['base']} | {r['readout']} | {r['unit']} | {r['axis']} | {r['apl_on_side']} | {r['ratio_six']:.2f} / {r['ratio_two']:.2f} "
                f"({r['ratio_used_six']:.2f} / {r['ratio_used_two']:.2f}) | {r['loss']:.3f} / {r['pattern']:.3f} / {r['mean']:+.3f} / {r['rho']:.2f} | "
                f"{r['homog_loss']:.3f} / {r['homog_pattern']:.3f} / {r['homog_mean']:+.3f} / {r['homog_rho']:.2f} | {r['six_state_vs_six_loss']:.3f} "
                f"(ρ {r['six_state_vs_six_rho']:.2f}) | {r['two_state_vs_two_loss']:.3f} (ρ {r['two_state_vs_two_rho']:.2f}) |")
        L.append('')
    (out / 'comparison_table.md').write_text('\n'.join(L) + '\n')


# the check of E10_joint_fit.py's --apl-data-units: E69's fit 2 (the firing ceiling at strength 2 on C2's family), started
# exactly at its optimum for one evaluation; with the option off it must reproduce the fit's best evaluation to the last digit
E10_CHECK_START = ('E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.1114t1000_k2')
E10_CHECK_ARGS = ['-k', '2', '--aplpn-max', '3', '--rate-target', 'ctrl_frac', '--loss', 'apl', '--apl-readout', 'both',
    '--kc-refractory-period', '0.003', '--kc-adapt-inc', '0.1113973859994802', '--kc-adapt-tau', '1.0', '--maxfev', '1',
    '--init-from-exact']
E10_PARAM_COLS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')
E10_CHECK_EXTRA = {'off': [], 'noise': ['--apl-data-units', 'noise']}


def check_e10(name: str, extra: Sequence[str] = ()) -> dict:
    """E10_joint_fit.py with the options of E69's fit 2 (`E10_CHECK_ARGS`, plus `extra`), started exactly at that fit's
    optimum, one evaluation, through E73_zcell_fit_checks.py's --run-e10 (its scratch experiment's results under
    outputs/E76_data_side_target/e10_check/<name>/, nothing written to the shared results); the evaluation compared with
    the fit's best evaluation in its saved history, every column (`eval` and `time_s` aside): identical statistics, the
    columns that differ (with their difference in units in the last place), missing and extra columns. Writes and returns
    <name>/check.json."""
    import shutil
    import socket
    import sys
    import time
    exp, tag = E10_CHECK_START
    root = H.outputs_dir(EXPERIMENT) / 'e10_check' / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    args = [*E10_CHECK_ARGS, '--init-from', tag, '--init-from-experiment', exp, '--experiment', 'E73_zcell_check', *extra]
    cmd = [sys.executable, '-u', str(H.HERE / 'E73_zcell_fit_checks.py'), '--run-e10', str(root), '--', *args]
    t0 = time.time()
    with open(root / 'log.txt', 'w') as log:
        rc = subprocess.run(cmd, cwd=H.HERE, stdout=log, stderr=subprocess.STDOUT).returncode
    hist = sorted((root / 'results').glob('history_*.csv'))
    git = lambda *a: subprocess.run(['git', *a], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    out = dict(name=name, args=args, returncode=rc, seconds=time.time() - t0, host=socket.gethostname(),
        commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain', '--', '.')))
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
        out.update(history=str(hist[0]), saved_eval=int(row['eval']) if 'eval' in row else None, n_columns=len(cols),
            statistics_identical=not stat_diff and all(c in new.index for c in cols),
            differing_statistics=stat_diff, differing_parameters={c: v for c, v in diff.items() if c in E10_PARAM_COLS},
            missing=[c for c in cols if c not in new.index], extra=[c for c in new.index if c not in old.columns and c not in ('eval', 'time_s')],
            key_values={c: (float(new[c]), float(row[c])) for c in ('loss', 'apl_raw_rmse', 'apl_raw_mean', 'apl_raw_pattern', 'apl_cal_rmse',
                'apl_cal_mean', 'apl_cal_pattern', 'apl_cal_K', 'rate_on', 'rate_off', 'term_apl_raw', 'term_apl_cal') if c in new.index and c in row.index})
    (root / 'check.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=('D', 'B', 'C', 'E', 'E-tables', 'table', 'check-e10'))
    ap.add_argument('--check', nargs='*', default=['off'], help=f'with --part check-e10: the checks to run, of {list(E10_CHECK_EXTRA)}')
    ap.add_argument('--jobs', type=int, default=3, help='part C: worker processes')
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--only', nargs='*', default=None, help='part E: the point ids to run')
    ap.add_argument('--check-only', action='store_true', help='part E: only the reproduction checks')
    ap.add_argument('--bases', nargs='*', default=None, help=f'part E: the bases to run, of {list(E_BASES)}')
    ap.add_argument('--base2', default=None, metavar='EXP:TAG', help="part E: the ceiling base replaced by this fit (e.g. fit 0's optimum)")
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'D':
        part_d(out)
    elif args.part == 'B':
        part_b(out)
    elif args.part == 'C':
        part_c(out, jobs=args.jobs)
    elif args.part == 'table':
        write_table(out)
    elif args.part == 'E-tables':
        t = part_e_tables(out)
        print(t['populations'].round(3).to_string(), flush=True)
    elif args.part == 'check-e10':
        for name in args.check:
            r = check_e10(name, E10_CHECK_EXTRA[name])
            print(json.dumps({k: v for k, v in r.items() if k in ('name', 'returncode', 'seconds', 'statistics_identical', 'differing_statistics',
                'differing_parameters', 'missing', 'extra', 'key_values', 'error')}, indent=1, default=str), flush=True)
    else:
        # (the harness imports olfsysm, so the OpenMP thread count is the caller's environment: set both)
        if os.environ.get('OMP_THREAD_LIMIT') != str(args.threads) or os.environ.get('OMP_NUM_THREADS') != str(args.threads):
            ap.error(f'part E: run with OMP_NUM_THREADS={args.threads} OMP_THREAD_LIMIT={args.threads} in the environment '
                '(the harness loads olfsysm on import, so they cannot be set from here)')
        part_e(out, threads=args.threads, only=args.only, check_only=args.check_only, bases=args.bases, base2=args.base2)


if __name__ == '__main__':
    main()
