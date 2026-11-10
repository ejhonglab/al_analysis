#!/usr/bin/env python3
"""E68: the low end of the readout. Measurement noise below the response criterion, the weight of the
weakest responses, and (part B) a sub-threshold signal from the membrane potential.

The calibrated readout of the compressive readout experiment (E65) maps a model's spike counts n onto
n / (n + K), K set per model on its APL-on counts so that the responders-only participation ratio equals
the control flies'. At its low end a KC without a spike reads exactly 0, while the KC data (z-scored
trial means of 3 trials) carry measurement noise, weak positive and reproducible negative values there.
This script asks whether a like-for-like low end changes the APL-effect loss or the ranking of the
models, and how the KC data's own APL effect depends on how their low end is read.

Part A (no simulations):

A1, the data's low end (Yang's per-trial APL16 recordings, `noise_floor_trial_distances.load_trials`;
11 control, 8 APL output blocked (TNT)), per recording, mean over recordings per genotype: the
amplitude histogram of the trial means and the trial-to-trial sd by amplitude; the reproducible share of
the low bands, cross-validated over trials (entries selected on the mean of two trials, read on the
third); every pair's cosine split into three terms (`kc_remapping_harness.cosine_terms`: both of a KC's
entries responders, i.e. trial mean > 1 z; one a responder and the other sub-threshold; both
sub-threshold) and the per-pair APL effect split the same way into mean, per-odor and pair-specific
parts (recording-bootstrap intervals); the responder x sub-threshold term split by the sub-threshold
entry (an unreliable response: some trial above 1.5 z; a weak positive; a negative); the term between
sub-threshold entries computed between single trials of the same and of different repeat numbers (noise
shared within a block of trials) and against the solvent (pfo), to tell a per-KC baseline offset from
weak broad tuning; whether the negatives follow the number of responders to the odor.

A2, readouts with a low end, on the archived APL-on and acute APL-off spike counts of the key models
(E65's KEY_MODELS), the APL-effect fits (E67) if archived, and every archived variant without a realism
flag. Every low-end parameter is calibrated on the model's APL-on counts against control-fly statistics
that involve neither the APL effect nor any odor-odor distance (`kc_remapping_harness.calibrate_low_end`,
closed-form expectations through the noise, iterated to a fixed point) and applied unchanged to APL off:
the half-saturation K (responders-only participation ratio), the one-spike floor b ((n + b) / (n + b +
K) for n >= 1; the share of entries with a trial mean between 0.5 and 1.5 z, see below), the conversion c
of readout units to z (the APL-on response rate under the data's criterion, trial mean > 1 z, equals the
control flies' 6.8 %); the noise is the control flies' trial-to-trial sd of non-responding entries
(`kc_trial_noise_sd`, 0.39 z per trial). Configurations (CONFIGS): 'cal' (E65's calibrated readout as now:
its K, no floor, no noise; c only places the z-unit readings), 'cal_noise' (the same with noise), 'noise'
(K and c recalibrated through the noise, no floor), 'floor' (K, b and c jointly, scored without noise),
'floor_noise' (the same with noise), and 'floor_b1' / 'floor_b1_noise' (b set on the share between 0.5
and 1 z, as the entry literally proposed). With noise, ten pseudo-recordings of three trials are sampled
(seeded, the same draws for every variant); the graded reading uses the closed form
(`expected_pair_distances_noisy`), the others the pseudo-recordings. Each model reading is scored against
the same reading of the data (READINGS: graded, thresholded = trial means <= 1 z set to 0, binary =
trial mean > 1 z, noise_corrected, nonneg = negative values set to 0, floor = values at or below the
noise floor `kc_noise_floor()` set to 0): the APL-effect loss with its mean, per-odor and pair-specific
parts (`apl_effect_decomposition`), the mean distances, the two ρ's on odor-odor distances against the
data's matrices in the same reading, the fruity x small-ester block and the natural-source alignment
(APL on / off / on minus off); per configuration also K, b, c, the amplitude statistics in both
conditions and the three cosine terms. The one-spike weight table (key models, K held at E65's value):
one-spike entries read as 0, as now, as two spikes, and a sweep of b.

Why b is set on 0.5-1.5 z rather than the proposal's 0.5-1 z: the share between 0.5 and 1 z barely moves
with b once c holds the response rate (in models tuned below 6.8 % c pins the one-spike entries just
below 1 z; in the 6.8 % models the share stays below the control flies' 6.1 %), so the fixed point drifts
to the bounds (an affine readout); the share between 0.5 and 1.5 z (the proposal's first table) rises
monotonically with b and identifies it. Both are run ('floor' and 'floor_b1').

A3, the same through the archived jittered trials of the models that have them (data/model_outputs/
<experiment>/partB/trials/: ten pseudo-recordings of three trials with per-KC threshold jitter at the
level matched on the control flies' single-trial distance, 'match_on'): readout and measurement noise per
trial, the data's processing after; calibrations on the trials' APL-on counts ('cal_trials': E65's K;
'noise_trials': K and c; 'floor_trials': K, b and c); the single-trial same-odor distances beside the
data's.

Part B (`--part B`, simulations) is not implemented yet.

Outputs, <shared>/results/E68_readout_low_end/: data_*.csv (A1), calibration.csv, by_variant_low_end.csv
(one row per variant x configuration x reading), variant_amplitude.csv, key_models.csv,
key_model_histograms.csv, one_spike_weight.csv, medians.csv, trials_low_end.csv, trials_calibration.csv,
comparison_table.md, the figures, provenance.json.
Usage: nice -n 10 ./venv/bin/python E68_readout_low_end.py [--jobs 8] [--only-key] [--boot 1000]
"""
from __future__ import annotations

import os
import sys

if '--part' not in sys.argv or sys.argv[sys.argv.index('--part') + 1] != 'B':
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(_v, '1')

import argparse
import hashlib
import json
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E65_spike_count_readout import KEY_MODELS

EXPERIMENT = 'E68_readout_low_end'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
RESULTS = H.shared_dir() / 'results'
E65_TABLE = RESULTS / 'E65_spike_count_readout' / 'by_variant_readout.csv'
BY_VARIANT = RESULTS / 'apl_effect' / 'by_variant.csv'
ODORS = list(H.APL16_ODORS)
PAIRS = H.pair_list(ODORS)
PAIR_INDEX = pd.MultiIndex.from_tuples(PAIRS, names=['odor_a', 'odor_b'])
IU = np.triu_indices(len(ODORS), 1)
N_TRIALS = 3
N_RECORDINGS = 10
SEED_ON, SEED_OFF = 11, 12
# the noise-corrected reading of a pseudo-recording leaves out pairs with an odor whose signal power is
# below this share of its trial mean's power (`noise_corrected_pair_distances`); the data's are all > 0.5
MIN_RELIABILITY = 0.1
READINGS = ('graded', 'thresholded', 'binary', 'noise_corrected', 'nonneg', 'floor')
READING_LABEL = {'graded': 'graded (the target as defined)', 'thresholded': 'sub-threshold values set to 0',
    'binary': 'binarized', 'noise_corrected': 'corrected for trial-to-trial noise', 'nonneg': 'negative values set to 0',
    'floor': 'values below the noise floor set to 0', 'graded_sampled': 'graded, sampled (check of the closed form)'}
BAND = (0.5, 1.5)          # the band b is calibrated on (see the docstring)
BAND_LITERAL = (0.5, 1.0)  # the proposal's band
FLOOR_BOUNDS = (-1.0, 10.0)
# (name, how K / b / c are set, scored with noise)
CONFIGS = [('cal', 'e65', False), ('cal_noise', 'e65', True), ('noise', 'joint_no_floor', True),
    ('floor', 'joint_floor', False), ('floor_noise', 'joint_floor', True),
    ('floor_b1', 'joint_floor_literal', False), ('floor_b1_noise', 'joint_floor_literal', True)]
CONFIG_LABEL = {'cal': 'calibrated readout as now', 'cal_noise': 'as now, with measurement noise',
    'noise': 'K and c recalibrated through the noise', 'floor': 'one-spike floor (K, b, c jointly), no noise',
    'floor_noise': 'one-spike floor with measurement noise', 'floor_b1': 'floor on 0.5-1 z, no noise',
    'floor_b1_noise': 'floor on 0.5-1 z, with noise', 'cal_trials': 'jittered trials, as now',
    'noise_trials': 'jittered trials, K and c through the noise', 'floor_trials': 'jittered trials, one-spike floor'}
HIST_EDGES = np.array([-np.inf, -1, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, np.inf])
TRIAL_FOLDERS = ['E48_block_target_noise', 'E51_noise_robust_block', 'E53_kckc_held_block', 'E58_gain_noise_block', 'E63_rho_fit']
GENO_LABEL = {'TNT3c': 'control flies', 'TNT': 'APL output blocked (TNT)'}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cos_dist(X: np.ndarray) -> np.ndarray:
    """Cosine distances (PAIRS order) between the odor columns of a (units x odors) array."""
    n = np.linalg.norm(X, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = (X.T @ X) / np.outer(n, n)
    return 1 - S[IU]


def as_pairs(v: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(v, dtype=float), index=PAIR_INDEX)


def hist_shares(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return np.histogram(x, bins=HIST_EDGES)[0] / len(x)


def hist_labels() -> List[str]:
    e = HIST_EDGES
    return [f'({e[i]:g}, {e[i + 1]:g}]' for i in range(len(e) - 1)]


def amplitude_summary(x: np.ndarray, floor_z: float) -> Dict[str, float]:
    """Low-end statistics of z values (KC x odor trial means, pooled): shares of entries per band, the
    response rate (> 1 z) and the responders' mean amplitude."""
    x = np.asarray(x, dtype=float).ravel()
    sh = lambda m: float(m.mean())
    return dict(share_05_15=sh((x > 0.5) & (x <= 1.5)), share_05_1=sh((x > 0.5) & (x <= 1.0)), share_1_15=sh((x > 1.0) & (x <= 1.5)),
        share_025_1=sh((x > 0.25) & (x <= 1.0)), share_above_4=sh(x > 4), share_below_m025=sh(x < -0.25), share_negative=sh(x < 0),
        share_below_floor=sh(x <= floor_z), rate_z=sh(x > 1), resp_amp_mean=float(x[x > 1].mean()) if (x > 1).any() else np.nan)


def trialmean_band_shares(genotype: str) -> Dict[str, float]:
    """The KC data's low-end shares (`amplitude_summary` of the 16 odors' trial means per recording, mean
    over recordings): the calibration targets of the low end and the registry entries."""
    mats = H.yang_recording_matrices(genotype)
    rows = [amplitude_summary(m[ODORS].to_numpy(dtype=float), H.kc_noise_floor()) for m in mats.values()]
    return pd.DataFrame(rows).mean().to_dict()


# ---------------------------------------------------------------------------
# A1: the data's low end
def data_trials(genotype: str) -> Dict[tuple, Dict[str, np.ndarray]]:
    """{recording: {'X': trials x KCs x 16 odors, 'pfo': trials x KCs or None, 'repeat': repeat numbers}}."""
    import noise_floor_trial_distances as NF
    p = H.yang_data_dir() / genotype / f'dataset_{NF.DATASETS[genotype]}_long.csv'
    d = pd.read_csv(p)
    d['abbrev'] = d['odor'].map(H.odor_abbrev)
    d = d[d.abbrev.isin(ODORS + [H.SOLVENT])]
    out = {}
    for rec, by_odor in NF.recording_arrays(d).items():
        out[rec] = dict(X=np.stack([by_odor[o] for o in ODORS], axis=-1), pfo=by_odor.get(H.SOLVENT))
    return out


def split_parts(e: np.ndarray) -> Dict[str, float]:
    """Mean, per-odor (additive projection of the centered values) and pair-specific (residual) parts of a
    per-pair quantity (PAIRS order), as root mean squares, with the per-odor and pair-specific vectors."""
    X, _ = H.pair_design(PAIR_INDEX, ODORS)
    c = e - e.mean()
    coef, *_ = np.linalg.lstsq(X, c, rcond=None)
    proj = X @ coef
    return dict(mean=float(e.mean()), per_odor=float(np.sqrt(np.mean(proj ** 2))), pair_specific=float(np.sqrt(np.mean((c - proj) ** 2))),
        _proj=proj, _resid=c - proj, _c=c)


def recording_terms(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Per pair (PAIRS order): the three cosine terms of the trial means and the responder x sub-threshold
    term split by the sub-threshold entry's class (unreliable: a trial above 1.5 z; weak positive: mean >
    0; negative: mean <= 0)."""
    m = X.mean(axis=0)
    R = m > H.KC_RESPONSE_Z
    S = ~R
    unrel = S & (X > 1.5).any(axis=0)
    weak = S & ~unrel & (m > 0)
    neg = S & ~unrel & (m <= 0)
    n = np.linalg.norm(m, axis=0)
    den = np.outer(n, n)
    mR, mS = m * R, m * S
    out = dict(both=(mR.T @ mR) / den, one=(mR.T @ mS + mS.T @ mR) / den, neither=(mS.T @ mS) / den)
    for k, C in (('one_unreliable', unrel), ('one_weak', weak), ('one_negative', neg)):
        mC = m * C
        out[k] = (mR.T @ mC + mC.T @ mR) / den
    return {k: v[IU] for k, v in out.items()}


def repeat_terms(X: np.ndarray, pfo: Optional[np.ndarray]) -> Dict[str, float]:
    """The term between sub-threshold entries (trial means <= 1 z) from single trials of the same and of
    different repeat numbers (each single-trial vector normalized by its own length; mean over pairs and
    trial combinations), the full single-trial cosine likewise, and the solvent comparisons: the mean
    Pearson r across sub-threshold KCs between two odors' trial means and between an odor's and pfo's."""
    m = X.mean(axis=0)
    S = m <= H.KC_RESPONSE_Z
    T = X.shape[0]
    out = {}
    for name, same in (('same_repeat', True), ('different_repeat', False)):
        sub, full = [], []
        for r in range(T):
            for q in range(T):
                if (r == q) != same:
                    continue
                A, B = X[r], X[q]
                na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
                den = np.outer(na, nb)
                sub.append((((A * S).T @ (B * S)) / den)[IU])
                full.append(((A.T @ B) / den)[IU])
        out[f'neither_{name}'] = float(np.mean(sub))
        out[f'cosine_{name}'] = float(np.mean(full))
    rs = []
    for i, j in zip(*IU):
        k = S[:, i] & S[:, j]
        if k.sum() > 10:
            rs.append(np.corrcoef(m[k, i], m[k, j])[0, 1])
    out['sub_r_odor_odor'] = float(np.nanmean(rs))
    if pfo is not None:
        p = pfo.mean(axis=0)
        sp = p <= H.KC_RESPONSE_Z
        rp = []
        for i in range(m.shape[1]):
            k = S[:, i] & sp
            if k.sum() > 10:
                rp.append(np.corrcoef(m[k, i], p[k])[0, 1])
        out['sub_r_odor_pfo'] = float(np.nanmean(rp))
        # the 'neither' cosine term between each odor and pfo, against the odor-odor mean
        mm = np.column_stack([m, p])
        Sm = np.column_stack([S, sp])
        n = np.linalg.norm(mm, axis=0)
        nt = ((mm * Sm).T @ (mm * Sm)) / np.outer(n, n)
        out['neither_odor_pfo'] = float(nt[:-1, -1].mean())
        out['pfo_mean'] = float(p.mean())
        out['pfo_share_negative'] = float((p < 0).mean())
    out['neither_odor_odor'] = float(recording_terms(X)['neither'].mean())
    return out


def part_a1(n_boot: int, seed: int = 0) -> Dict[str, pd.DataFrame]:
    floor_z = H.kc_noise_floor()
    rec_rows, hist_rows, sd_rows, cv_rows, rep_rows, neg_rows = [], [], [], [], [], []
    terms = {}
    for g in ('TNT3c', 'TNT'):
        terms[g] = []
        for rec, a in data_trials(g).items():
            X = a['X']
            m = X.mean(axis=0)
            rid = dict(genotype=g, recording='_'.join(map(str, rec)), n_kcs=X.shape[1])
            rec_rows.append(dict(rid, **amplitude_summary(m, floor_z)))
            hist_rows.append(dict(rid, **dict(zip(hist_labels(), hist_shares(m)))))
            v = X.var(axis=0, ddof=1)
            for i in range(len(HIST_EDGES) - 1):
                k = (m > HIST_EDGES[i]) & (m <= HIST_EDGES[i + 1])
                sd_rows.append(dict(rid, bin=hist_labels()[i], n=int(k.sum()), trial_sd=float(np.sqrt(v[k].mean())) if k.any() else np.nan))
            # cross-validated over trials: select on the mean of two trials, read the third
            cv_bins = [(-np.inf, -0.3), (-0.3, 0), (0, 0.3), (0.3, 1), (1, 1.5), (1.5, np.inf)]
            for held in range(X.shape[0]):
                sel = np.delete(X, held, axis=0).mean(axis=0)
                t = X[held]
                for lo, hi in cv_bins:
                    k = (sel > lo) & (sel <= hi)
                    if k.any():
                        cv_rows.append(dict(rid, held_out=held, band=f'({lo:g}, {hi:g}]', n=int(k.sum()), selected_mean=float(sel[k].mean()),
                            held_out_mean=float(t[k].mean()), any_trial_above_15=float((X[:, k] > 1.5).any(axis=0).mean())))
            rep_rows.append(dict(rid, **repeat_terms(X, a['pfo'])))
            # negatives against the number of responders, per odor
            rate = (m > 1).mean(axis=0)
            negs = (m < -0.25).mean(axis=0)
            negpow = np.array([(m[:, j][m[:, j] < 0] ** 2).sum() / (m[:, j] ** 2).sum() for j in range(m.shape[1])])
            for j, o in enumerate(ODORS):
                neg_rows.append(dict(rid, odor=o, rate=float(rate[j]), share_below_m025=float(negs[j]), negative_power_share=float(negpow[j]),
                    mean_negative=float(m[:, j][m[:, j] < 0].mean())))
            terms[g].append(recording_terms(X))
    out = dict(rec=pd.DataFrame(rec_rows), hist=pd.DataFrame(hist_rows), sd=pd.DataFrame(sd_rows), cv=pd.DataFrame(cv_rows),
        rep=pd.DataFrame(rep_rows), neg=pd.DataFrame(neg_rows))
    # the cosine terms per pair and the APL effect split into them, with recording-bootstrap intervals
    keys = list(terms['TNT3c'][0].keys())
    arr = {g: {k: np.stack([t[k] for t in terms[g]]) for k in keys} for g in terms}

    def effect_parts(sel):
        mean = {g: {k: arr[g][k][sel[g]].mean(axis=0) for k in keys} for g in arr}
        total = -(sum(mean['TNT3c'][k] for k in ('both', 'one', 'neither')) - sum(mean['TNT'][k] for k in ('both', 'one', 'neither')))
        tot = split_parts(total)
        row = dict(effect_mean=tot['mean'], effect_per_odor=tot['per_odor'], effect_pair_specific=tot['pair_specific'])
        var_c = float(np.mean(tot['_c'] ** 2))
        for k in keys:
            e = -(mean['TNT3c'][k] - mean['TNT'][k])
            p = split_parts(e)
            row.update({f'{k}__ctrl': float(mean['TNT3c'][k].mean()), f'{k}__tnt': float(mean['TNT'][k].mean()), f'{k}__effect_mean': p['mean'],
                f'{k}__effect_per_odor': p['per_odor'], f'{k}__effect_pair_specific': p['pair_specific'],
                # share of the effect's pattern (variance over pairs) the term carries: cov(term, total) / var(total)
                f'{k}__pattern_share': float(np.mean(p['_c'] * tot['_c']) / var_c),
                f'{k}__per_odor_share': float(np.mean(p['_proj'] * tot['_proj']) / np.mean(tot['_proj'] ** 2)),
                f'{k}__pair_specific_share': float(np.mean(p['_resid'] * tot['_resid']) / np.mean(tot['_resid'] ** 2))})
        return row
    full = {g: np.arange(len(terms[g])) for g in terms}
    obs = effect_parts(full)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(effect_parts({g: rng.integers(0, len(terms[g]), len(terms[g])) for g in terms}))
    b = pd.DataFrame(boots)
    out['terms'] = pd.DataFrame({'observed': pd.Series(obs), 'lo': b.quantile(0.05), 'hi': b.quantile(0.95)})
    pair_rows = []
    for k in keys:
        for g in arr:
            for (a_, b_), v in zip(PAIRS, arr[g][k].mean(axis=0)):
                pair_rows.append(dict(term=k, genotype=g, odor_a=a_, odor_b=b_, value=float(v)))
    out['term_pairs'] = pd.DataFrame(pair_rows)
    return out


def data_low_end_tables(a1: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Per-genotype means of the A1 tables."""
    num = lambda df: df.drop(columns=[c for c in ('recording',) if c in df]).groupby('genotype').mean(numeric_only=True)
    out = dict(summary=num(a1['rec']).T, hist=num(a1['hist']).T, rep=num(a1['rep']).T)
    sd = a1['sd'].groupby(['genotype', 'bin'], sort=False).apply(lambda g: pd.Series(dict(trial_sd=float(np.sqrt(np.nanmean(g.trial_sd ** 2))),
        n=int(g.n.sum())))).unstack('genotype')
    out['sd'] = sd
    cv = a1['cv'].groupby(['genotype', 'band'], sort=False)[['n', 'selected_mean', 'held_out_mean', 'any_trial_above_15']].mean()
    cv['reproducible_share'] = cv['held_out_mean'] / cv['selected_mean']
    out['cv'] = cv
    neg = a1['neg']
    from scipy import stats
    rows = []
    for g, gg in neg.groupby('genotype'):
        per_rec = [stats.spearmanr(r.rate, r.share_below_m025)[0] for _, r in gg.groupby('recording')]
        rows.append(dict(genotype=g, rho_within_recording_mean=float(np.nanmean(per_rec)),
            rho_pooled=float(stats.spearmanr(gg.rate, gg.share_below_m025)[0]),
            rho_power_within_recording_mean=float(np.nanmean([stats.spearmanr(r.rate, r.negative_power_share)[0] for _, r in gg.groupby('recording')])),
            rho_odor_means=float(stats.spearmanr(gg.groupby('odor').rate.mean(), gg.groupby('odor').share_below_m025.mean())[0])))
    out['neg'] = pd.DataFrame(rows).set_index('genotype')
    out['neg_odor'] = neg.groupby(['genotype', 'odor'])[['rate', 'share_below_m025', 'negative_power_share', 'mean_negative']].mean().unstack('genotype')
    return out


# ---------------------------------------------------------------------------
# A2 / A3: the models
_REFS: Dict[str, object] = {}


def refs() -> Dict[str, object]:
    """The data side, cached per process: per reading the per-pair effect and the two genotypes' mean
    matrices; the 66 reference pairs; the noise, the floor and the calibration targets."""
    if not _REFS:
        for v in READINGS:
            if v == 'graded':
                eff, on, off = H.yang_apl_effect(), H.yang_mean_rdm('TNT3c'), H.yang_mean_rdm('TNT')
            elif v == 'binary':
                eff, on, off = H.yang_apl_effect(binarize=1.0), H.yang_mean_rdm('TNT3c', binarize=1.0), H.yang_mean_rdm('TNT', binarize=1.0)
            else:
                eff = H.yang_apl_effect(values=v)
                on, off = H.yang_recording_rdms_values('TNT3c', v).mean(), H.yang_recording_rdms_values('TNT', v).mean()
            _REFS[v] = dict(effect=eff, on=H.canonical_pairs(on), off=H.canonical_pairs(off))
        _REFS['graded_sampled'] = _REFS['graded']
        _REFS['ref66'] = H.reference_pairs()
        _REFS['noise_sd'] = H.kc_trial_noise_sd('TNT3c')
        _REFS['floor_z'] = H.kc_noise_floor()
        dt = H.data_target
        _REFS['targets'] = dict(rate=dt('ctrl_frac'), pr=dt('ctrl_participation_ratio_resp'), band=dt('ctrl_share_z_05_15'),
            band_literal=dt('ctrl_share_z_05_1'))
    return _REFS


_VCF: Dict[str, np.ndarray] = {}


def vcf_alignment(d: pd.Series) -> float:
    """The natural-source alignment of a pair Series in PAIRS order, as
    `compare_to_references(d)['spearman_resid_vs_vcf_hyp']` computes it (Spearman of the residual
    around the least-squares line on the ORN distances with the hyperbolic natural-source distances,
    over the 66 megamat pairs), without its per-call pandas overhead."""
    from scipy import stats
    if not _VCF:
        ref = refs()['ref66']
        pos = {p: i for i, p in enumerate(PAIRS)}
        _VCF['idx'] = np.array([pos[p] for p in ref.index])
        _VCF['orn'], _VCF['vcf'] = ref['orn'].to_numpy(dtype=float), ref['vcf_hyp'].to_numpy(dtype=float)
    y = d.to_numpy(dtype=float)[_VCF['idx']]
    x = _VCF['orn']
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    b, a = np.polyfit(x[ok], y[ok], 1)
    return float(stats.spearmanr(y[ok] - (a + b * x[ok]), _VCF['vcf'][ok])[0])


def score_pair(d_on: pd.Series, d_off: pd.Series, reading: str) -> Dict[str, float]:
    """Everything scored for one model reading against the same reading of the data."""
    r = refs()
    st, _ = H.apl_effect_decomposition(d_on, d_off, r[reading]['effect'])
    st = {k.replace('apl_effect_', ''): v for k, v in st.items()}
    st['mean_dist_on'], st['mean_dist_off'] = float(d_on.mean()), float(d_off.mean())
    st['rho_on'] = H.compare_model_to_data(d_on, r[reading]['on'])['spearman']
    st['rho_off'] = H.compare_model_to_data(d_off, r[reading]['off'])['spearman']
    oo, _ = H.compare_on_off(d_on, d_off)
    st['block'] = oo['block_fruity_small_iso']
    st['vcf_on'], st['vcf_off'] = vcf_alignment(d_on), vcf_alignment(d_off)
    st['vcf_diff'] = st['vcf_on'] - st['vcf_off']
    return st


def readings_noise_free(z: np.ndarray) -> Dict[str, np.ndarray]:
    floor_z = refs()['floor_z']
    g = cos_dist(z)
    return dict(graded=g, thresholded=cos_dist(np.where(z > 1, z, 0.0)), binary=cos_dist((z > 1).astype(float)), noise_corrected=g,
        nonneg=cos_dist(np.clip(z, 0, None)), floor=cos_dist(np.where(z > floor_z, z, 0.0)))


def readings_noisy(trials: np.ndarray, z_closed: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Readings of noisy pseudo-recordings (recordings x trials x units x odors), each the mean over
    recordings of the per-recording distances; 'graded' from the closed form when `z_closed` (the
    noise-free z signal) is given, else sampled."""
    floor_z, s = refs()['floor_z'], refs()['noise_sd']
    means = trials.mean(axis=1)
    acc = {k: [] for k in ('graded_sampled', 'thresholded', 'binary', 'noise_corrected', 'nonneg', 'floor')}
    for r in range(means.shape[0]):
        m = means[r]
        acc['graded_sampled'].append(cos_dist(m))
        acc['thresholded'].append(cos_dist(np.where(m > 1, m, 0.0)))
        acc['binary'].append(cos_dist((m > 1).astype(float)))
        acc['nonneg'].append(cos_dist(np.clip(m, 0, None)))
        acc['floor'].append(cos_dist(np.where(m > floor_z, m, 0.0)))
        acc['noise_corrected'].append(H.noise_corrected_pair_distances(trials[r], ODORS, min_reliability=MIN_RELIABILITY).to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = {k: np.nanmean(np.stack(v), axis=0) for k, v in acc.items()}
    # pseudo-recordings x pairs the noise correction left out (an odor with almost no signal)
    out['_n_corrected_dropped'] = int(np.isnan(np.stack(acc['noise_corrected'])).sum())
    if z_closed is not None:
        out['graded'] = H.expected_pair_distances_noisy(pd.DataFrame(z_closed, columns=ODORS), s, trials.shape[1]).to_numpy()
    else:
        out['graded'] = out['graded_sampled']
    return out


def terms_of(z_list: List[np.ndarray]) -> Dict[str, float]:
    """Mean over pairs (and pseudo-recordings) of the three cosine terms."""
    acc = {'both': [], 'one': [], 'neither': [], 'cosine': []}
    for z in z_list:
        t = H.cosine_terms(pd.DataFrame(z, columns=ODORS))
        for k in acc:
            acc[k].append(float(t[k].mean()))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def saturating(counts: np.ndarray, K: float, b: float) -> np.ndarray:
    """`H.readout(..., 'saturating', K, floor=b)` on an array (the same expressions)."""
    x = counts.astype(float)
    if not b:
        return x / (x + K)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(x > 0, (x + b) / (x + b + K), 0.0)


def calibrate(on: np.ndarray, how: str, K_e65: float) -> Dict[str, object]:
    """K, b and c of one configuration (`CONFIGS`), on APL-on counts (units x odors, or recordings x trials
    x units x odors)."""
    r = refs()
    t, s = r['targets'], r['noise_sd']
    if how == 'e65':
        trial_sig = saturating(on, K_e65, 0.0)
        c = H.z_scale(trial_sig, t['rate'], s, n_trials=N_TRIALS)
        sig = trial_sig.mean(axis=1) if trial_sig.ndim == 4 else trial_sig
        mu = np.concatenate([x.T for x in sig]) if sig.ndim == 3 else sig.T
        st = H.low_end_stats(c * mu, s / np.sqrt(N_TRIALS), band=BAND)
        return dict(K=K_e65, floor=0.0, c=c, iterations=0, converged=True, K_at_bound=False, floor_at_bound=False, c_at_bound=False, **st)
    kw = dict(noise_sd=s, target_rate=t['rate'], target_pr=t['pr'], n_trials=N_TRIALS, K0=K_e65 if np.isfinite(K_e65) else 10.0,
        floor_bounds=FLOOR_BOUNDS)
    if how == 'joint_no_floor':
        cal = H.calibrate_low_end(on, target_band=t['band'], fit_floor=False, band=BAND, **kw)
    elif how == 'joint_floor':
        cal = H.calibrate_low_end(on, target_band=t['band'], band=BAND, **kw)
    elif how == 'joint_floor_literal':
        cal = H.calibrate_low_end(on, target_band=t['band_literal'], band=BAND_LITERAL, **kw)
    else:
        raise ValueError(how)
    cal = dict(cal)
    cal.pop('history')
    if how == 'joint_floor_literal':
        # report the 0.5-1.5 z share too, as for the other configurations
        sig = saturating(on, cal['K'], cal['floor'])
        sig = sig.mean(axis=1) if sig.ndim == 4 else sig
        mu = np.concatenate([x.T for x in sig]) if sig.ndim == 3 else sig.T
        cal['band_share_literal'] = cal.pop('band_share')
        cal['band_share'] = H.low_end_stats(cal['c'] * mu, s / np.sqrt(N_TRIALS), band=BAND)['band_share']
    return cal


def one_spike_share(r: np.ndarray, counts: np.ndarray) -> float:
    sq = r ** 2
    return float(sq[counts == 1].sum() / sq.sum())


def load_counts(folder: str, tag: str, cond: str) -> np.ndarray:
    df = pd.read_parquet(ARCHIVE / folder / f'{tag}_{cond}_spike_counts.parquet')
    return H.abbrev_columns(df.astype(float))[ODORS].to_numpy()


def score_variant(job: tuple) -> Tuple[List[dict], List[dict], List[dict], Dict[str, object]]:
    """All configurations and readings of one archived variant: (score rows, calibration rows, amplitude
    rows, extras for key models: histograms and one-spike-weight rows)."""
    folder, tag, K_e65, key = job
    on, off = load_counts(folder, tag, 'on'), load_counts(folder, tag, 'acute_off')
    s = refs()['noise_sd']
    floor_z = refs()['floor_z']
    rows, cal_rows, amp_rows = [], [], []
    extras = dict(hist=[], one_spike=[])
    cals = {}
    for name, how, noisy in CONFIGS:
        if how not in cals:
            cals[how] = calibrate(on, how, K_e65)
            cal_rows.append(dict(folder=folder, tag=tag, calibration=how, K_e65=K_e65, **cals[how]))
        cal = cals[how]
        K, b, c = cal['K'], cal['floor'], cal['c']
        z = {cond: c * saturating(x, K, b) for cond, x in (('on', on), ('off', off))}
        if noisy:
            trials = {cond: H.sample_pseudo_recordings(z[cond], s, n_recordings=N_RECORDINGS, n_trials=N_TRIALS,
                seed=SEED_ON if cond == 'on' else SEED_OFF, return_trials=True) for cond in z}
            dist = {cond: readings_noisy(trials[cond], z[cond]) for cond in z}
            zs = {cond: list(trials[cond].mean(axis=1)) for cond in z}
        else:
            dist = {cond: readings_noise_free(z[cond]) for cond in z}
            zs = {cond: [z[cond]] for cond in z}
        base = dict(folder=folder, tag=tag, config=name, K=K, floor=b, c=c, one_spike_z=c * saturating(np.array([1.0]), K, b)[0],
            two_spike_z=c * saturating(np.array([2.0]), K, b)[0])
        amp = dict(base)
        for cond in z:
            pooled = np.concatenate([x.ravel() for x in zs[cond]])
            amp.update({f'{k}_{cond}': v for k, v in amplitude_summary(pooled, floor_z).items()})
            amp.update({f'term_{k}_{cond}': v for k, v in terms_of(zs[cond]).items()})
            if key:
                extras['hist'].append(dict(base, condition=cond, **dict(zip(hist_labels(), hist_shares(pooled)))))
        amp_rows.append(amp)
        if noisy:
            amp['n_corrected_dropped_on'], amp['n_corrected_dropped_off'] = dist['on']['_n_corrected_dropped'], dist['off']['_n_corrected_dropped']
        for reading in list(READINGS) + (['graded_sampled'] if noisy else []):
            st = score_pair(as_pairs(dist['on'][reading]), as_pairs(dist['off'][reading]), reading)
            rows.append(dict(base, reading=reading, **st))
    if key:
        # the one-spike weight with K held at E65's value: one-spike entries read as 0, as now, as two
        # spikes, and a sweep of the floor b
        variants = [('one spike read as 0', None, 0.0), ('as now, 1 / (1 + K)', None, 'now'), ('one spike read as two', None, 'two')]
        variants += [(f'floor b = {bb:g}', bb, None) for bb in (-1.0, -0.5, 0.25, 0.5, 1.0, 2.0)]
        variants += [('calibrated floor (K, b, c jointly)', 'cal', None)]
        for label, bb, mode in variants:
            if bb == 'cal':
                Kc, bc = cals['joint_floor']['K'], cals['joint_floor']['floor']
                r_on, r_off = saturating(on, Kc, bc), saturating(off, Kc, bc)
            elif bb is not None:
                r_on, r_off = saturating(on, K_e65, bb), saturating(off, K_e65, bb)
            else:
                r_on, r_off = saturating(on, K_e65, 0.0), saturating(off, K_e65, 0.0)
                if mode != 'now':
                    v = 0.0 if mode == 0.0 else 2.0 / (2.0 + K_e65)
                    r_on, r_off = np.where(on == 1, v, r_on), np.where(off == 1, v, r_off)
            st = score_pair(as_pairs(cos_dist(r_on)), as_pairs(cos_dist(r_off)), 'graded')
            extras['one_spike'].append(dict(folder=folder, tag=tag, weight=label, K=Kc if bb == 'cal' else K_e65,
                floor=bc if bb == 'cal' else (bb if bb is not None else 0.0), one_spike_share_on=one_spike_share(r_on, on),
                one_spike_share_off=one_spike_share(r_off, off), **st))
    return rows, cal_rows, amp_rows, extras


def score_variant_safe(job: tuple):
    """`score_variant`, with an exception reported as (None, message) instead of stopping the run."""
    try:
        return score_variant(job)
    except Exception as e:
        return None, f'{job[0]} {job[1]}: {type(e).__name__}: {e}'


def trial_sets() -> List[Tuple[str, str, float, str]]:
    """(folder, model, sigma, level) of every archived jittered trial set at the level matched on the
    control flies' single-trial distance ('match_on', or 'match_on+match_off')."""
    out, seen = [], set()
    for folder in TRIAL_FOLDERS:
        lv = RESULTS / folder / 'partB' / 'noise_levels.json'
        tdir = ARCHIVE / folder / 'partB' / 'trials'
        if not lv.exists() or not tdir.exists():
            continue
        for model, v in json.loads(lv.read_text()).items():
            levels = v.get('levels', v)
            lev = next((k for k in ('match_on', 'match_on+match_off') if k in levels), None)
            if lev is None:
                continue
            sig = levels[lev]
            sig = sig.get('sigma', sig) if isinstance(sig, dict) else sig
            files = sorted(tdir.glob(f'{model}_sigma{sig:g}_n3_rep*.npz'))
            if len(files) and model not in seen:
                seen.add(model)
                out.append((folder, model, float(sig), lev))
    return out


def score_trials(job: tuple) -> Tuple[List[dict], List[dict]]:
    """A3: one jittered trial set: per configuration, readout and measurement noise per trial, then the
    data's processing; plus the single-trial same-odor distances."""
    folder, model, sigma, level = job
    tdir = ARCHIVE / folder / 'partB' / 'trials'
    files = sorted(tdir.glob(f'{model}_sigma{sigma:g}_n3_rep*.npz'))
    arrs = [np.load(f) for f in files]
    spk = {cond: np.stack([a[f'{cond}_spk'].astype(float) for a in arrs]) for cond in ('on', 'acute_off')}   # recs x trials x KCs x odors
    det_on = ARCHIVE / folder / 'partB' / f'{model}_on_spike_counts.parquet'
    K_e65 = H.calibrate_readout(H.abbrev_columns(pd.read_parquet(det_on).astype(float))[ODORS], 'saturating',
        refs()['targets']['pr'], stat='participation_ratio', responders_min=2.0)
    s = refs()['noise_sd']
    rows, cal_rows = [], []
    for name, how in (('cal_trials', 'e65'), ('noise_trials', 'joint_no_floor'), ('floor_trials', 'joint_floor')):
        cal = calibrate(spk['on'], how, K_e65)
        cal_rows.append(dict(folder=folder, model=model, sigma=sigma, level=level, config=name, K_det_e65=K_e65, **cal))
        K, b, c = cal['K'], cal['floor'], cal['c']
        base = dict(folder=folder, model=model, sigma=sigma, level=level, config=name, K=K, floor=b, c=c)
        dist, single = {}, {}
        for cond, seed0 in (('on', 1000), ('acute_off', 2000)):
            sig = c * saturating(spk[cond], K, b)
            noisy = np.stack([H.sample_pseudo_recordings(sig[r][None], s, seed=seed0 + r, return_trials=True)[0] for r in range(sig.shape[0])])
            dist[cond] = readings_noisy(noisy, None)
            st = []
            for r in range(noisy.shape[0]):
                for i in range(noisy.shape[1]):
                    for j in range(i + 1, noisy.shape[1]):
                        A, B = noisy[r, i], noisy[r, j]
                        st.append(1 - (A * B).sum(axis=0) / (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0)))
            single[cond] = float(np.nanmean(st))
            pooled = noisy.mean(axis=1).ravel()
            base.update({f'{k}_{cond}': v for k, v in amplitude_summary(pooled, refs()['floor_z']).items()})
            base.update({f'term_{k}_{cond}': v for k, v in terms_of(list(noisy.mean(axis=1))).items()})
        base['single_trial_dist_on'], base['single_trial_dist_off'] = single['on'], single['acute_off']
        base['n_corrected_dropped_on'], base['n_corrected_dropped_off'] = dist['on']['_n_corrected_dropped'], dist['acute_off']['_n_corrected_dropped']
        for reading in list(READINGS) + ['graded_sampled']:
            st = score_pair(as_pairs(dist['on'][reading]), as_pairs(dist['acute_off'][reading]), reading)
            rows.append(dict(base, reading=reading, **st))
    return rows, cal_rows


def medians(df: pd.DataFrame) -> pd.DataFrame:
    """Per configuration and reading, over the variants without a realism flag: medians of the loss, its
    parts, the mean effect and the ρ's; the share of variants whose loss falls below their 'cal' loss in
    the same reading; the Spearman of the variants' order by loss against 'cal'."""
    real = df[df.realism_flag.fillna('') == '']
    base = real[real.config == 'cal'].set_index(['folder', 'tag', 'reading'])['rmse']
    out = []
    for (cfg, reading), g in real.groupby(['config', 'reading'], sort=False):
        g = g.set_index(['folder', 'tag', 'reading'])
        b0 = base.reindex(g.index)
        best = g['rmse'].idxmin()
        out.append(dict(config=cfg, reading=reading, n=len(g), loss=g.rmse.median(), mean_effect=g['mean'].median(),
            pattern_part=g.rmse_centered.median(), per_odor=g.per_odor.median(), pair_specific=g.pair_specific.median(),
            rho_effect=g.spearman.median(), rho_on=g.rho_on.median(), rho_off=g.rho_off.median(), block=g.block.median(),
            K=g.K.median(), floor=g.floor.median(), c=g.c.median(), share_improved_vs_cal=float((g.rmse < b0).mean()),
            order_rho_vs_cal=H.spearman(g.rmse, b0), best=f'{best[0]} {best[1]}', best_loss=float(g.loc[best, 'rmse'])))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', default='A', choices=['A', 'B'])
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--only-key', action='store_true', help='A2 on the key models only (a quick check)')
    ap.add_argument('--boot', type=int, default=1000, help='recording-bootstrap draws for A1')
    args = ap.parse_args()
    if args.part == 'B':
        raise NotImplementedError('part B is built after part A')
    out = H.results_dir(EXPERIMENT)
    t0 = datetime.now()
    # A1
    a1 = part_a1(args.boot)
    for k in ('rec', 'hist', 'sd', 'cv', 'rep', 'neg', 'term_pairs'):
        a1[k].to_csv(out / f'data_{k}.csv', index=False)
    a1['terms'].to_csv(out / 'data_effect_terms.csv')
    dtab = data_low_end_tables(a1)
    for k, v in dtab.items():
        v.to_csv(out / f'data_{k}_by_genotype.csv')
    print(dtab['summary'].round(4).to_string(), flush=True)
    print(a1['terms'].round(4).to_string(), flush=True)
    # A2
    e65 = pd.read_csv(E65_TABLE, low_memory=False)
    k_e65 = e65[e65.readout == 'saturating_cal_pr'].set_index(['folder', 'tag'])['param']
    bv = pd.read_csv(BY_VARIANT)
    have = [(r.folder, r.tag) for r in bv.itertuples() if (ARCHIVE / r.folder / f'{r.tag}_on_spike_counts.parquet').exists()
        and (ARCHIVE / r.folder / f'{r.tag}_acute_off_spike_counts.parquet').exists()]
    keys = {(f, t) for _, f, t in KEY_MODELS}
    e67 = sorted({(r.folder, r.tag) for r in bv.itertuples() if str(r.folder).startswith('E67')})
    flag = bv.set_index(['folder', 'tag'])['realism_flag'].fillna('')
    chosen = [h for h in have if h in keys or h in e67 or (not args.only_key and flag.get(h, 'x') == '')]
    print(f'{len(chosen)} variants ({len(have)} archived; E67 archived: {len(e67)})', flush=True)
    jobs = []
    for f, t in chosen:
        K = float(k_e65.get((f, t), np.nan))
        if not np.isfinite(K):
            on_df = H.abbrev_columns(pd.read_parquet(ARCHIVE / f / f'{t}_on_spike_counts.parquet').astype(float))[ODORS]
            K = H.calibrate_readout(on_df, 'saturating', refs()['targets']['pr'], stat='participation_ratio', responders_min=2.0)
        jobs.append((f, t, K, (f, t) in keys))
    rows, cal_rows, amp_rows, hist_rows, os_rows, failed = [], [], [], [], [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, res in enumerate(ex.map(score_variant_safe, jobs, chunksize=2)):
            if res[0] is None:
                failed.append(res[1])
                print('failed:', res[1], flush=True)
                continue
            r, c, a, x = res
            rows.extend(r); cal_rows.extend(c); amp_rows.extend(a); hist_rows.extend(x['hist']); os_rows.extend(x['one_spike'])
            if (i + 1) % 50 == 0:
                print(f'{i + 1} / {len(jobs)} variants ({(datetime.now() - t0).seconds} s)', flush=True)
        tjobs = trial_sets()
        print(f'{len(tjobs)} jittered trial sets', flush=True)
        trows, tcal = [], []
        for r, c in ex.map(score_trials, tjobs):
            trows.extend(r); tcal.extend(c)
    flags = bv.set_index(['folder', 'tag'])[['exp', 'rate_on', 'rate_off', 'spearman_rate_on_vs_data', 'realism_flag']]
    df = pd.DataFrame(rows).join(flags, on=['folder', 'tag'])
    df.to_csv(out / 'by_variant_low_end.csv', index=False)
    pd.DataFrame(cal_rows).join(flags, on=['folder', 'tag']).to_csv(out / 'calibration.csv', index=False)
    amp = pd.DataFrame(amp_rows).join(flags, on=['folder', 'tag'])
    amp.to_csv(out / 'variant_amplitude.csv', index=False)
    hist = pd.DataFrame(hist_rows)
    hist.to_csv(out / 'key_model_histograms.csv', index=False)
    one = pd.DataFrame(os_rows)
    one.to_csv(out / 'one_spike_weight.csv', index=False)
    med = medians(df)
    med.to_csv(out / 'medians.csv', index=False)
    tr = pd.DataFrame(trows)
    tr.to_csv(out / 'trials_low_end.csv', index=False)
    pd.DataFrame(tcal).to_csv(out / 'trials_calibration.csv', index=False)
    label = {(f, t): lab for lab, f, t in KEY_MODELS}
    key = df[[(f, t) in label for f, t in zip(df.folder, df.tag)]].copy()
    key.insert(0, 'model', [label[(f, t)] for f, t in zip(key.folder, key.tag)])
    key.to_csv(out / 'key_models.csv', index=False)
    write_table(out, dtab, a1, df, key, amp, one, med, tr, pd.DataFrame(tcal))
    figures(out, dtab, a1, key, hist, amp)
    git = lambda *a: subprocess.run(['git', *a], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    import noise_floor_trial_distances as NF
    inputs = [BY_VARIANT, E65_TABLE, H.data_targets_path(), H.yang_data_dir() / 'computed' / 'trialmean_responses_long.parquet',
        H.yang_data_dir() / 'computed' / 'rdm_per_recording_long.csv'] + [H.yang_data_dir() / g / f'dataset_{i}_long.csv' for g, i in NF.DATASETS.items()]
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E68_readout_low_end.py --part A',
        host=subprocess.run(['hostname', '-s'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=git('rev-parse', 'HEAD'),
        al_analysis_dirty=bool(git('status', '--porcelain', '--untracked-files=no')),
        script_and_harness_dirty=bool(git('status', '--porcelain', '--', 'E68_readout_low_end.py', 'kc_remapping_harness.py', 'data_targets.py')),
        inputs_sha256={str(p): sha256(p) for p in inputs if Path(p).exists()},
        n_variants=len(jobs), failed_variants=failed, n_trial_sets=len(tjobs), configs=[c[0] for c in CONFIGS], readings=list(READINGS), band=BAND,
        band_literal=BAND_LITERAL, floor_bounds=FLOOR_BOUNDS, n_recordings=N_RECORDINGS, n_trials=N_TRIALS, seeds=dict(on=SEED_ON, off=SEED_OFF,
        trials_on='1000 + rep', trials_off='2000 + rep', a1_bootstrap=0), n_boot=args.boot, noise_sd=refs()['noise_sd'],
        floor_z=refs()['floor_z'], targets=refs()['targets'], e67_archived=[list(x) for x in e67],
        runtime_s=(datetime.now() - t0).seconds)
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n')
    print(med[med.reading == 'graded'].round(3).to_string())
    print('->', out)


# ---------------------------------------------------------------------------
# tables and figures
def f3(x, spec='.3f') -> str:
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def pct(x) -> str:
    return '' if not np.isfinite(x) else f'{100 * x:.1f} %'


def write_table(out, dtab, a1, df, key, amp, one, med, tr, tcal) -> None:
    dt = H.data_target
    s = dtab['summary']
    ref_f, ref_t = KEY_MODELS[0][1], KEY_MODELS[0][2]
    L = ['# E68 part A: the low end of the readout (tables)', '']
    # 1: the data's low end, beside the reference model under three configurations
    L += ['## The low end of the KC data and of the reference model', '',
        f"Noise: the control flies' trial-to-trial sd of non-responding entries {refs()['noise_sd']:.3f} z per trial "
        f"(APL output blocked (TNT): {H.kc_trial_noise_sd('TNT'):.3f}); noise floor of the 'floor' reading {refs()['floor_z']:.3f} z.", '']
    ra = amp[(amp.folder == ref_f) & (amp.tag == ref_t)].set_index('config')
    cfgs = [c for c in ('cal', 'noise', 'floor_noise') if c in ra.index]
    tr_ref = tr[(tr.model == f'E44-{ref_t}') & (tr.reading == 'graded')].set_index('config') if len(tr) else pd.DataFrame()
    head = '| statistic | control flies | APL output blocked (TNT) | ' + ' | '.join(f'reference model, {CONFIG_LABEL[c]}, APL on | APL off' for c in cfgs)
    if 'floor_trials' in tr_ref.index:
        head += ' | reference model, jittered trials with the floor, APL on | APL off'
    L += [head + ' |', '|---|' + '---:|' * (2 + 2 * len(cfgs) + (2 if 'floor_trials' in tr_ref.index else 0))]
    stats_rows = [('entries with a trial mean of 0.5 to 1.5 z', 'share_05_15', pct), ('entries 0.5 to 1 z', 'share_05_1', pct),
        ('entries 0.25 to 1 z', 'share_025_1', pct), ('entries above 4 z', 'share_above_4', pct), ('entries below -0.25 z', 'share_below_m025', pct),
        ('response rate (trial mean > 1 z)', 'rate_z', pct), ('mean amplitude of the responders (z)', 'resp_amp_mean', lambda x: f3(x, '.2f'))]
    for lab, col, fmt in stats_rows:
        cells = [fmt(s.loc[col, 'TNT3c']), fmt(s.loc[col, 'TNT'])]
        for c in cfgs:
            cells += [fmt(ra.loc[c, f'{col}_on']), fmt(ra.loc[c, f'{col}_off'])]
        if 'floor_trials' in tr_ref.index:
            cells += [fmt(tr_ref.loc['floor_trials', f'{col}_on']), fmt(tr_ref.loc['floor_trials', f'{col}_acute_off'])]
        L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    terms = a1['terms']['observed']
    for lab, k in (('mean cosine similarity between two odors', 'cosine'), ('of which: both of a KC\'s entries responders', 'both'),
            ('of which: one a responder, the other sub-threshold', 'one'), ('of which: both sub-threshold', 'neither')):
        if k == 'cosine':
            dc = sum(terms[f'{t}__ctrl'] for t in ('both', 'one', 'neither')); dtn = sum(terms[f'{t}__tnt'] for t in ('both', 'one', 'neither'))
        else:
            dc, dtn = terms[f'{k}__ctrl'], terms[f'{k}__tnt']
        cells = [f3(dc), f3(dtn)]
        for c in cfgs:
            cells += [f3(ra.loc[c, f'term_{k}_on']), f3(ra.loc[c, f'term_{k}_off'])]
        if 'floor_trials' in tr_ref.index:
            cells += [f3(tr_ref.loc['floor_trials', f'term_{k}_on']), f3(tr_ref.loc['floor_trials', f'term_{k}_acute_off'])]
        L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    if 'floor_trials' in tr_ref.index:
        L.append(f"| distance between two single trials of one odor | {dt('ctrl_single_trial_floor'):.2f} | {dt('tnt_single_trial_floor'):.2f} | "
            + ' | ' * (2 * len(cfgs) - 1) + f" | {tr_ref.loc['floor_trials', 'single_trial_dist_on']:.2f} | {tr_ref.loc['floor_trials', 'single_trial_dist_off']:.2f} |")
    L += ['', 'Model columns: z units through the conversion c, pooled over the ten pseudo-recordings where noise is added; the KC data: mean over '
        'recordings. The cosine terms: each pair\'s cosine similarity split by KC (responder = trial mean above 1 z), mean over the 120 pairs.', '']
    # 2: the APL effect split into the three terms (data)
    L += ['## The data\'s per-pair APL effect split into the three cosine terms (90 % recording-bootstrap intervals)', '',
        '| term | control flies | APL output blocked (TNT) | mean APL effect | per-odor part | pair-specific part | share of the pattern | share of the per-odor part | share of the pair-specific part |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    T = a1['terms']
    for k in ('both', 'one', 'neither', 'one_unreliable', 'one_weak', 'one_negative'):
        cell = lambda c, sp='.3f': f"{T.loc[c, 'observed']:{sp}} [{T.loc[c, 'lo']:{sp}}, {T.loc[c, 'hi']:{sp}}]"
        L.append(f'| {k} | {cell(f"{k}__ctrl")} | {cell(f"{k}__tnt")} | {cell(f"{k}__effect_mean", "+.3f")} | {f3(T.loc[f"{k}__effect_per_odor", "observed"])} | '
            f'{f3(T.loc[f"{k}__effect_pair_specific", "observed"])} | {f3(T.loc[f"{k}__pattern_share", "observed"], ".2f")} | '
            f'{f3(T.loc[f"{k}__per_odor_share", "observed"], ".2f")} | {f3(T.loc[f"{k}__pair_specific_share", "observed"], ".2f")} |')
    L.append(f"| total (the data's effect) | | | {T.loc['effect_mean', 'observed']:+.3f} | {T.loc['effect_per_odor', 'observed']:.3f} | "
        f"{T.loc['effect_pair_specific', 'observed']:.3f} | 1 | 1 | 1 |")
    L += ['', 'The APL effect of a term is minus its control-minus-blocked change (a larger cosine is a smaller distance); the three terms add up to '
        'the effect. one_unreliable / one_weak / one_negative split the responder x sub-threshold term by the sub-threshold entry: some trial above '
        '1.5 z; otherwise trial mean above 0; otherwise. Shares: each term\'s covariance with the total over the 120 pairs (centered), relative to the '
        "total's variance; they add up to 1.", '']
    # 3: cross-validated bands, repeats, pfo, negatives
    cv = dtab['cv']
    L += ['## The data\'s low bands across trials (selected on the mean of two trials, read on the third)', '',
        '| band of the two-trial mean (z) | control flies: entries per recording | selected mean | third trial | any trial above 1.5 z | '
        'APL output blocked (TNT): entries | selected mean | third trial | any trial above 1.5 z |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for band in cv.loc['TNT3c'].index:
        c_, t_ = cv.loc[('TNT3c', band)], cv.loc[('TNT', band)]
        L.append(f'| {band} | {c_.n:.0f} | {c_.selected_mean:+.2f} | {c_.held_out_mean:+.2f} | {pct(c_.any_trial_above_15)} | {t_.n:.0f} | '
            f'{t_.selected_mean:+.2f} | {t_.held_out_mean:+.2f} | {pct(t_.any_trial_above_15)} |')
    rep = dtab['rep']
    L += ['', '| between sub-threshold entries (data) | control flies | APL output blocked (TNT) |', '|---|---:|---:|']
    for lab, k in (('the both-sub-threshold term, trial means', 'neither_odor_odor'), ('the same from single trials, same repeat number', 'neither_same_repeat'),
            ('the same, different repeat numbers', 'neither_different_repeat'), ('single-trial cosine (all KCs), same repeat number', 'cosine_same_repeat'),
            ('single-trial cosine, different repeat numbers', 'cosine_different_repeat'),
            ('Pearson r across sub-threshold KCs, two odors', 'sub_r_odor_odor'), ('Pearson r across sub-threshold KCs, odor and pfo', 'sub_r_odor_pfo'),
            ('the both-sub-threshold term between an odor and pfo', 'neither_odor_pfo'), ('pfo: mean trial mean (z)', 'pfo_mean'),
            ('pfo: share of KCs below 0', 'pfo_share_negative')):
        if k in rep.index:
            L.append(f'| {lab} | {f3(rep.loc[k, "TNT3c"])} | {f3(rep.loc[k, "TNT"])} |')
    ng = dtab['neg']
    L += ['', '| do the negatives follow the responders? (Spearman across odors) | control flies | APL output blocked (TNT) |', '|---|---:|---:|']
    for lab, k in (('share below -0.25 z vs response rate, within a recording (mean)', 'rho_within_recording_mean'),
            ('share of the squared vector in negatives vs response rate, within a recording', 'rho_power_within_recording_mean'),
            ('the same on the odors\' means over recordings', 'rho_odor_means')):
        L.append(f'| {lab} | {f3(ng.loc["TNT3c", k], ".2f")} | {f3(ng.loc["TNT", k], ".2f")} |')
    # 4: readings x models
    dmean = {v: float(refs()[v]['effect'].mean()) for v in READINGS}
    L += ['', '## The key models under each reading (mean APL effect, and in parentheses the APL-effect loss against the same reading of the data)', '']
    show = ['cal', 'cal_noise', 'noise', 'floor', 'floor_noise', 'floor_b1_noise']
    L += ['| reading, both sides alike | KC data | ' + ' | '.join(CONFIG_LABEL[c] for c in show) + ' |',
        '|---|---:|' + '---:|' * len(show)]
    for label, f, t in KEY_MODELS:
        g = key[(key.folder == f) & (key.tag == t)]
        if g.empty:
            continue
        L.append(f'| **{label}** | | ' + ' | ' * (len(show) - 1) + ' |')
        for v in READINGS:
            cells = []
            for c in show:
                r = g[(g.config == c) & (g.reading == v)]
                cells.append(f"{r['mean'].iloc[0]:+.3f} ({r['rmse'].iloc[0]:.3f})" if len(r) else '')
            L.append(f'| {READING_LABEL[v]} | {dmean[v]:+.3f} | ' + ' | '.join(cells) + ' |')
    L += ['', f"The data's loss floor, graded: {dt('apl_effect_rmse_floor'):.3f}; thresholded {dt('apl_effect_thresholded_rmse_floor'):.3f}; binarized "
        f"{dt('apl_effect_binary_rmse_floor'):.3f}; nonneg {dt('apl_effect_nonneg_rmse_floor'):.3f}; floor {dt('apl_effect_floor_rmse_floor'):.3f}.", '']
    # 5: loss parts, graded, key models
    L += ['## The loss and its parts against the graded data (key models)', '',
        '| model | configuration | K | b | c (z per unit) | one spike (z) | loss | mean effect | per-odor part | pair-specific part | ρ on the per-pair APL effect | ρ on odor-odor distances, APL on / off | block | natural-source alignment, APL on / off / on minus off |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, f, t in KEY_MODELS:
        g = key[(key.folder == f) & (key.tag == t) & (key.reading == 'graded')].set_index('config')
        for c in [x[0] for x in CONFIGS]:
            if c not in g.index:
                continue
            r = g.loc[c]
            L.append(f'| {label} | {CONFIG_LABEL[c]} | {r.K:.2f} | {r.floor:+.2f} | {r.c:.2f} | {r.one_spike_z:.2f} | {r.rmse:.3f} | {r["mean"]:+.3f} | '
                f'{r.per_odor:.3f} | {r.pair_specific:.3f} | {r.spearman:.2f} | {r.rho_on:.2f} / {r.rho_off:.2f} | {r.block:+.3f} | '
                f'{r.vcf_on:.2f} / {r.vcf_off:.2f} / {r.vcf_diff:+.2f} |')
    L += ['', f"KC data: mean effect {dmean['graded']:+.3f}, loss floor {dt('apl_effect_rmse_floor'):.3f}, ρ ceiling {dt('apl_effect_rho_ceiling'):.2f}; "
        f"block {dt('block'):+.3f}; natural-source alignment {dt('ctrl_vcf'):.2f} / {dt('tnt_vcf'):.2f} / {dt('vcf_apl_effect'):+.2f}.", '']
    # 6: one-spike weight
    L += ['## The weight of the one-spike entries (K held at the calibrated readout\'s value; graded, no noise)', '',
        '| one-spike entries read as | ' + ' | '.join(lab for lab, _, _ in KEY_MODELS) + ' |', '|---|' + '---:|' * len(KEY_MODELS)]
    for w in one.weight.unique():
        cells = []
        for lab, f, t in KEY_MODELS:
            r = one[(one.folder == f) & (one.tag == t) & (one.weight == w)]
            cells.append(f"{r.rmse.iloc[0]:.3f} ({r.rmse_centered.iloc[0]:.3f}; {r['mean'].iloc[0]:+.3f})" if len(r) else '')
        L.append(f'| {w} | ' + ' | '.join(cells) + ' |')
    cells = []
    for lab, f, t in KEY_MODELS:
        r = one[(one.folder == f) & (one.tag == t) & (one.weight == 'as now, 1 / (1 + K)')]
        cells.append(f'{pct(r.one_spike_share_on.iloc[0])} / {pct(r.one_spike_share_off.iloc[0])}' if len(r) else '')
    L.append('| share of the squared readout vector in one-spike entries, APL on / APL off | ' + ' | '.join(cells) + ' |')
    L += ['', 'Cells: loss against the graded data (pattern part; mean APL effect). Lower loss is better; floor '
        f"{dt('apl_effect_rmse_floor'):.3f}, the data's mean effect {dmean['graded']:+.3f}.", '']
    # 7: calibration and medians
    L += ['## Medians over the variants without a realism flag', '',
        '| configuration | reading | variants | loss | mean effect | pattern part | per-odor | pair-specific | ρ on the per-pair APL effect | '
        'ρ on odor-odor distances, APL on / off | K | b | c | improved vs as now | ρ with the as-now order | lowest loss (variant) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in med.itertuples():
        L.append(f'| {r.config} | {r.reading} | {r.n} | {r.loss:.3f} | {r.mean_effect:+.3f} | {r.pattern_part:.3f} | {r.per_odor:.3f} | {r.pair_specific:.3f} | '
            f'{r.rho_effect:.2f} | {r.rho_on:.2f} / {r.rho_off:.2f} | {r.K:.1f} | {r.floor:+.2f} | {r.c:.1f} | {pct(r.share_improved_vs_cal)} | {r.order_rho_vs_cal:.2f} | '
            f'{r.best_loss:.3f} ({r.best}) |')
    cal = pd.read_csv(out / 'calibration.csv')
    L += ['', '## The calibrations', '', '| calibration | variants | at a bound (K / b / c) | not converged | iterations (median, max) | K (median) | b (median) | c (median) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for how, g in cal.groupby('calibration', sort=False):
        L.append(f'| {how} | {len(g)} | {int(g.K_at_bound.sum())} / {int(g.floor_at_bound.sum())} / {int(g.c_at_bound.sum())} | {int((~g.converged.astype(bool)).sum())} | '
            f'{g.iterations.median():.0f}, {g.iterations.max():.0f} | {g.K.median():.2f} | {g.floor.median():+.2f} | {g.c.median():.2f} |')
    # 8: A3
    if len(tr):
        L += ['', '## Through the jittered trials (graded and the other readings against the same reading of the data)', '',
            '| model (trial set) | configuration | K | b | c | single-trial distance, APL on / off | ' + ' | '.join(READING_LABEL[v] for v in READINGS) + ' |',
            '|---|---|---:|---:|---:|---:|' + '---:|' * len(READINGS)]
        for (model, cfg), g in tr.groupby(['model', 'config'], sort=False):
            g = g.set_index('reading')
            r0 = g.iloc[0]
            L.append(f'| {model} ({r0.folder}) | {cfg} | {r0.K:.2f} | {r0.floor:+.2f} | {r0.c:.2f} | {r0.single_trial_dist_on:.2f} / {r0.single_trial_dist_off:.2f} | '
                + ' | '.join(f"{g.loc[v, 'mean']:+.3f} ({g.loc[v, 'rmse']:.3f})" for v in READINGS) + ' |')
        L += ['', f"KC data: single-trial distance {dt('ctrl_single_trial_floor'):.2f} (control flies) / {dt('tnt_single_trial_floor'):.2f} (APL output "
            'blocked (TNT)); mean effect per reading: ' + ', '.join(f'{v} {dmean[v]:+.3f}' for v in READINGS) + '.', '']
    (out / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def figures(out, dtab, a1, key, hist, amp) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = hist_labels()
    x = np.arange(len(labels))
    dh = dtab['hist']
    ref_f, ref_t = KEY_MODELS[0][1], KEY_MODELS[0][2]
    j_f, j_t = KEY_MODELS[3][1], KEY_MODELS[3][2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, (cond, g) in zip(axes, (('on', 'TNT3c'), ('off', 'TNT'))):
        ax.bar(x - 0.3, dh[g].reindex(labels), 0.2, color='k', label=f'KC data, {GENO_LABEL[g]}')
        series = [(ref_f, ref_t, 'noise', '#9ecae1', 'the reference model, K and c through the noise'),
            (ref_f, ref_t, 'floor_noise', '#2171b5', 'the reference model, one-spike floor with noise'),
            (j_f, j_t, 'floor_noise', '#d9480f', 'the first joint fit, one-spike floor with noise')]
        for i, (f, t, c, col, lab) in enumerate(series):
            h = hist[(hist.folder == f) & (hist.tag == t) & (hist.config == c) & (hist.condition == cond)]
            if len(h):
                ax.bar(x - 0.1 + 0.2 * i, h[labels].iloc[0].to_numpy(dtype=float), 0.2, color=col, label=f'{lab}, APL {cond}')
        ax.set_yscale('log'); ax.set_ylim(1e-4, 1)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7.5)
        ax.set_xlabel('trial mean (z)'); ax.set_title(f'APL {cond} / {GENO_LABEL[g]}', fontsize=9.5)
        ax.legend(fontsize=7, frameon=False); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('share of KC x odor entries')
    fig.suptitle('Amplitude histograms of the trial means: the KC data against the models\' readout in z units (log scale)', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'E68_amplitude_histograms.png', dpi=130); plt.close(fig)
    # loss parts per reading, key models, two configurations
    cfgs = ['cal', 'noise', 'floor_noise']
    fig, axes = plt.subplots(1, len(READINGS), figsize=(18, 5.5), sharey=True)
    models = [m for m in KEY_MODELS if ((key.folder == m[1]) & (key.tag == m[2])).any()]
    cols = {'mean': '#bdbdbd', 'per_odor': '#6baed6', 'pair_specific': '#08519c'}
    for ax, v in zip(axes, READINGS):
        for i, (lab, f, t) in enumerate(models):
            for j, c in enumerate(cfgs):
                r = key[(key.folder == f) & (key.tag == t) & (key.config == c) & (key.reading == v)]
                if not len(r):
                    continue
                r = r.iloc[0]
                parts = {'mean': r.mean_error ** 2, 'per_odor': r.per_odor ** 2, 'pair_specific': r.pair_specific ** 2}
                y = i * (len(cfgs) + 1) + j
                left = 0
                for k, val in parts.items():
                    ax.barh(y, val, left=left, color=cols[k], edgecolor='white', lw=0.3, label=k.replace('_', '-') + ' part' if (i == 0 and j == 0 and v == READINGS[0]) else None)
                    left += val
        ticks = [i * (len(cfgs) + 1) + j for i in range(len(models)) for j in range(len(cfgs))]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{m[0][:38]} / {c}' for m in models for c in cfgs], fontsize=6)
        ax.invert_yaxis(); ax.set_title(READING_LABEL[v], fontsize=8.5); ax.set_xlabel('squared loss (parts add up)', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False, loc='lower right')
    fig.suptitle('The APL-effect loss squared, split into its mean, per-odor and pair-specific parts, per reading (key models; each against the same reading of the data)', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E68_loss_parts_by_reading.png', dpi=130); plt.close(fig)
    # the data's cosine terms and their APL effect
    T = a1['terms']
    ks = ['both', 'one', 'neither', 'one_unreliable', 'one_weak', 'one_negative']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xx = np.arange(len(ks))
    for i, (g, col) in enumerate((('ctrl', H.CONDITION_COLORS['control']), ('tnt', H.CONDITION_COLORS['TNT']))):
        o = np.array([T.loc[f'{k}__{g}', 'observed'] for k in ks])
        lo = np.array([T.loc[f'{k}__{g}', 'lo'] for k in ks]); hi = np.array([T.loc[f'{k}__{g}', 'hi'] for k in ks])
        axes[0].bar(xx - 0.2 + 0.4 * i, o, 0.4, yerr=[o - lo, hi - o], color=col, capsize=2,
            label='KCs, ' + ('control flies' if g == 'ctrl' else 'APL output blocked (TNT)'))
    o = np.array([T.loc[f'{k}__effect_mean', 'observed'] for k in ks])
    lo = np.array([T.loc[f'{k}__effect_mean', 'lo'] for k in ks]); hi = np.array([T.loc[f'{k}__effect_mean', 'hi'] for k in ks])
    axes[1].bar(xx, o, 0.6, yerr=[o - lo, hi - o], color='0.4', capsize=2)
    axes[1].axhline(0, color='k', lw=0.6)
    for ax, yl in zip(axes, ('mean term of the cosine similarity', 'its share of the mean APL effect')):
        ax.set_xticks(xx); ax.set_xticklabels(['both responders', 'one responder', 'both sub-threshold', 'one: unreliable', 'one: weak positive', 'one: negative'],
            rotation=25, ha='right', fontsize=8)
        ax.set_ylabel(yl, fontsize=8.5); ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle('The KC data\'s cosine similarity split by KC (responder = trial mean above 1 z), mean over the 120 pairs, 90 % recording-bootstrap intervals', fontsize=9)
    fig.tight_layout(); fig.savefig(out / 'E68_data_cosine_terms.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
