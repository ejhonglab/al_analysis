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

Part B (`--part B`, simulations at fixed scales; OMP_NUM_THREADS 16 or 8): the reference model, the eight at
1.5 at 70 Hz, the first joint fits with breadth exponent 2 and 3, and the best APL-effect fit (E67) under each
definition (`e67_best`), rebuilt with the trial-noise experiment's (E43) model code and re-run at their fitted
scales with APL on and acute APL off and the KC membrane potential recorded (olfsysm mp.kc.save_vm_sims; spike
counts checked against the archived ones, entry for entry), each KC x odor trace reduced in the worker to its
mean and peak over the odor window relative to the last 100 ms before onset and as a fraction of the distance
to the KC's threshold ('sims'). The readout adds, for entries without a spike, wp x the depolarization (at
most one spike's readout) and, for every non-responding entry (< 2 spikes), minus wm x the hyperpolarization;
wp is set on the control flies' share of entries between 0.25 and 1 z and wm on their share below -0.25 z,
through the noise, jointly with K, b and c (`calibrate_vm`), all on APL on. Scored as predictions: the APL-off
histogram, the three cosine terms with APL off, the order of the readings and the loss with its parts under
every reading. Controls (VM_CONFIGS): each weight alone, the peak instead of the mean depolarization, the
potential permuted across the non-spiking KCs within each odor and condition, the APL-on potential in both
conditions. For the reference model, thirty jittered trials with the per-KC threshold jitter re-calibrated
under the readout to the control flies' single-trial distance ('trials'). Then 'score' writes partB_*.csv,
comparison_table_partB.md, E68_partB_histograms.png and provenance_partB.json (with the model provenance);
the reduced potentials stay in ./outputs/E68_readout_low_end/partB/.

Outputs, <shared>/results/E68_readout_low_end/: data_*.csv (A1), calibration.csv, by_variant_low_end.csv
(one row per variant x configuration x reading), variant_amplitude.csv, key_models.csv,
key_model_histograms.csv, one_spike_weight.csv, medians.csv, trials_low_end.csv, trials_calibration.csv,
comparison_table.md, the figures, provenance.json.
Usage: nice -n 10 ./venv/bin/python E68_readout_low_end.py [--jobs 8] [--only-key] [--boot 1000]
       OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E68_readout_low_end.py --part B [--b-step sims|trials|score]
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
    ap.add_argument('--b-step', default='all', choices=['all', 'sims', 'trials', 'score'], help='part B: which step')
    args = ap.parse_args()
    if args.part == 'B':
        return main_b(args)
    out = H.results_dir(EXPERIMENT)
    t0 = datetime.now()
    # the code state at the start of the run (the working tree may change while it runs)
    git = lambda *a: subprocess.run(['git', *a], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    git_state = dict(al_analysis_commit=git('rev-parse', 'HEAD'), al_analysis_dirty=bool(git('status', '--porcelain', '--untracked-files=no')),
        script_and_harness_dirty=bool(git('status', '--porcelain', '--', 'E68_readout_low_end.py', 'kc_remapping_harness.py', 'data_targets.py')))
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
    e67 = sorted({(f, t) for f, t in have if str(f).startswith('E67')})
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
    import noise_floor_trial_distances as NF
    inputs = [BY_VARIANT, E65_TABLE, H.data_targets_path(), H.yang_data_dir() / 'computed' / 'trialmean_responses_long.parquet',
        H.yang_data_dir() / 'computed' / 'rdm_per_recording_long.csv'] + [H.yang_data_dir() / g / f'dataset_{i}_long.csv' for g, i in NF.DATASETS.items()]
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E68_readout_low_end.py --part A',
        host=subprocess.run(['hostname', '-s'], capture_output=True, text=True).stdout.strip(), **git_state,
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


# ---------------------------------------------------------------------------
# part B: a sub-threshold signal from the membrane potential (simulations at fixed scales)
B_SUB = 'partB'
# (key, label, experiment, tag)
B_MODELS = [('ref40', 'the reference model (eight at 1.5, 40 Hz, 6.8 %)', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'),
    ('thr70', 'eight at 1.5, 70 Hz, 6.8 %', 'E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3'),
    ('k2', 'the first joint fit (k 2, 5 %)', 'E10_joint_fit', 'k2'),
    ('k3', 'the joint fit with k 3 (5 %)', 'E10_joint_fit', 'k3')]
E67_FOLDER = 'E67_apl_effect_fits'
VM_PRE_S = 0.1             # the pre-odor baseline: the last 100 ms before odor onset
# (name, which potential terms are calibrated, the depolarization measure, the potential used)
VM_CONFIGS = [('floor_noise', (), 'mean', 'own'), ('vm', ('wp', 'wm'), 'mean', 'own'), ('vm_peak', ('wp', 'wm'), 'peak', 'own'),
    ('vm_pos_only', ('wp',), 'mean', 'own'), ('vm_neg_only', ('wm',), 'mean', 'own'),
    ('vm_permuted', 'vm', 'mean', 'permuted'), ('vm_on_potential', 'vm', 'mean', 'apl_on')]
VM_LABEL = {'floor_noise': 'one-spike floor with noise (part A)', 'vm': 'membrane potential, both terms',
    'vm_peak': 'both terms, peak depolarization', 'vm_pos_only': 'depolarization term alone', 'vm_neg_only': 'hyperpolarization term alone',
    'vm_permuted': 'control: potential permuted across non-spiking KCs', 'vm_on_potential': 'control: APL-on potential in both conditions'}
PARAM_BOUNDS = {'K': (1e-2, 1e3), 'b': FLOOR_BOUNDS, 'wp': (0.0, 1.0), 'wm': (0.0, 50.0)}
PARAM_TARGET = {'b': 'band_share', 'K': 'participation_ratio_resp', 'wp': 'band025', 'wm': 'below'}
JITTER_N_CAL = 6           # trials per evaluation of the jitter search
JITTER_N_TUNE = 3


def b_dir() -> Path:
    d = H.outputs_dir(EXPERIMENT) / B_SUB
    d.mkdir(parents=True, exist_ok=True)
    return d


def e67_best() -> List[Tuple[str, str, str, str, str]]:
    """The APL-effect fits (E67) with the lowest loss under each definition (raw counts; the calibrated
    readout), among the saved fits inside the realism bands (apl_effect_scores.realism_flag on the
    fits' response rates and, where scored, their response rate ρ)."""
    from apl_effect_scores import realism_flag
    p = RESULTS / E67_FOLDER / 'fits.csv'
    if not p.exists():
        return []
    f = pd.read_csv(p)
    bv = pd.read_csv(BY_VARIANT).set_index(['folder', 'tag'])
    rho = [bv['spearman_rate_on_vs_data'].get((E67_FOLDER, t), np.nan) for t in f.tag]
    f['flag'] = [realism_flag(a, b, r) for a, b, r in zip(f.rate_on, f.rate_off, rho)]
    f = f[f.flag == '']
    out = []
    for col, d in (('fit_apl_raw_rmse', 'raw'), ('fit_apl_cal_rmse', 'cal')):
        if col in f and f[col].notna().any():
            r = f.loc[f[col].idxmin()]
            out.append((f'e67_{d}', f'the best APL-effect fit, {"raw counts" if d == "raw" else "calibrated readout"} definition ({r.tag})',
                E67_FOLDER, str(r.tag), d))
    return out


def b_models() -> List[Tuple[str, str, str, str]]:
    return B_MODELS + [m[:4] for m in e67_best()]


def make_vm_model_class():
    """A subclass of the trial-noise experiment's (E43) excitability model that can record the KC
    membrane potential of a run and reduce it per KC and odor (imported lazily: E43 imports the model
    code)."""
    import E43_model_trial_noise as E43
    from al_analysis.mb_model import get_time_index

    class VmModel(E43.ExcitabilityModel):
        def __init__(self, name, orn, out_dir):
            super().__init__(name, orn, out_dir)
            t = np.asarray(get_time_index(self.mp, self.rv), dtype=float)
            self.vpre = (t >= self.mp.time_stim_start - VM_PRE_S) & (t < self.mp.time_stim_start)
            if not self.vpre.any() or not self.win.any():
                raise RuntimeError('empty baseline or response window')

        def run_vm(self, cond: str, fixed_thr: float, thr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
            """One KC run with the membrane potential recorded: (responses, spike counts, reduced
            potential: per KC and odor (ODORS order) the pre-odor baseline, the mean and the peak over the
            odor window, and the KC's threshold)."""
            self.mp.kc.save_vm_sims = True
            try:
                resp, spk = self.run_kc(cond, fixed_thr, thr)
                vm = self.rv.kc.vm_sims
                n = spk.shape[0]
                red = {k: np.empty((n, len(ODORS)), dtype=np.float32) for k in ('base', 'mean', 'peak')}
                for j, i in enumerate(self.pn_idx):
                    V = np.asarray(vm[i])
                    if V.shape != (n, len(self.win)):
                        raise RuntimeError(f'unexpected vm_sims shape {V.shape}')
                    red['base'][:, j] = V[:, self.vpre].mean(axis=1)
                    red['mean'][:, j] = V[:, self.win].mean(axis=1)
                    red['peak'][:, j] = V[:, self.win].max(axis=1)
                del vm, V
                red['thr'] = np.asarray(thr, dtype=np.float32).reshape(-1)
            finally:
                self.mp.kc.save_vm_sims = False
            return resp, spk, red
    return VmModel, E43


def potentials(red: Dict[str, np.ndarray], measure: str = 'mean') -> Tuple[np.ndarray, np.ndarray]:
    """(depolarization, hyperpolarization) per KC and odor as fractions of the distance from the pre-odor
    baseline to the KC's threshold: (window mean or peak minus baseline) / (threshold minus baseline),
    and (baseline minus window mean, if positive) / the same. The last axis of 'thr' broadcast over odors."""
    base = red['base'].astype(float)
    thr = red['thr'].astype(float)[..., None]
    dist = np.maximum(thr - base, 1e-9)
    v = red[measure].astype(float)
    return (v - base) / dist, np.maximum(base - red['mean'].astype(float), 0.0) / dist


def vm_readout(n: np.ndarray, u: np.ndarray, h: np.ndarray, K: float, b: float, wp: float, wm: float) -> np.ndarray:
    """The saturating readout with the one-spike floor (part A), plus for entries without a spike wp x
    the depolarization (clipped to [0, 1], so a KC at its threshold reads at most what one spike reads,
    wp <= 1) x one spike's readout, minus for the non-responding entries (< 2 spikes) wm x the
    hyperpolarization x one spike's readout."""
    r = saturating(n, K, b)
    r1 = (1.0 + b) / (1.0 + b + K)
    if wp:
        r = r + np.where(n == 0, wp * r1 * np.clip(u, 0.0, 1.0), 0.0)
    if wm:
        r = r - np.where(n < 2, wm * r1 * h, 0.0)
    return r


def vm_stats(mu: np.ndarray, sd: float) -> Dict[str, float]:
    """`H.low_end_stats` on the 0.5-1.5 z band plus the shares between 0.25 and 1 z and below -0.25 z (the
    potential terms' targets), in expectation through noise of sd `sd` on a trial mean."""
    from scipy.special import ndtr
    st = H.low_end_stats(mu, sd, band=BAND)
    st['band025'] = float(np.mean(ndtr((1.0 - mu) / sd) - ndtr((0.25 - mu) / sd)))
    st['below'] = float(np.mean(ndtr((-0.25 - mu) / sd)))
    return st


def calibrate_vm(n: np.ndarray, u: np.ndarray, h: np.ndarray, *, fit: Tuple[str, ...], start: Dict[str, float],
        max_iter: int = 40, tol: float = 1e-4) -> Dict[str, object]:
    """K, b, c and the potential weights jointly on the APL-on arrays (KCs x odors, or recordings x trials
    x KCs x odors): c on the response rate, b on the share between 0.5 and 1.5 z, K on the responders-only
    participation ratio, wp on the share between 0.25 and 1 z, wm on the share below -0.25 z (the control
    flies'), through the measurement noise; the parameters in `fit` (plus K and b) iterated to a fixed
    point, each the root nearest its current value among a grid's sign changes (c re-solved at every
    point), the others held at `start`."""
    from scipy.optimize import brentq
    r = refs()
    tg = dict(r['targets'], band025=H.data_target('ctrl_share_z_025_1'), below=H.data_target('ctrl_share_z_below_m025'))
    tg = {'band_share': tg['band'], 'participation_ratio_resp': tg['pr'], 'band025': tg['band025'], 'below': tg['below'], 'rate': tg['rate']}
    trials = n.ndim == 4
    n_tr = n.shape[1] if trials else N_TRIALS
    sd = r['noise_sd'] / np.sqrt(n_tr)

    def mu_unit(p):
        sig = vm_readout(n, u, h, p['K'], p['b'], p['wp'], p['wm'])
        if trials:
            sig = sig.mean(axis=1)
            return np.concatenate([x.T for x in sig])
        return sig.T

    from scipy.special import ndtr

    def solve_c(m):
        f = lambda lc: float(np.mean(ndtr((np.exp(lc) * m - H.KC_RESPONSE_Z) / sd))) - tg['rate']
        lo, hi = np.log(1e-2), np.log(1e3)
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return float(np.exp(lo if abs(flo) < abs(fhi) else hi))
        return float(np.exp(brentq(f, lo, hi, xtol=1e-10)))

    def err(p, name):
        m = mu_unit(p)
        c = solve_c(m)
        return vm_stats(c * m, sd)[PARAM_TARGET[name]] - tg[PARAM_TARGET[name]]

    def solve(p, name, first):
        lo, hi = PARAM_BOUNDS[name]
        logk = name == 'K'
        tr = (np.log, np.exp) if logk else (lambda x: x, lambda x: x)
        f = lambda x: err(dict(p, **{name: tr[1](x)}), name)
        x0 = tr[0](p[name])
        if not first:
            # after the first round, a bracket grown outward from the current value (the root moves little)
            a, fa = x0, f(x0)
            step = 0.02 * (tr[0](hi) - tr[0](lo))
            for direction in (1, -1):
                xb, fb = a, fa
                for _ in range(8):
                    xn = min(max(xb + direction * step * 2 ** _, tr[0](lo)), tr[0](hi))
                    fn = f(xn)
                    if np.isfinite(fa) and np.isfinite(fn) and fa * fn <= 0:
                        return float(tr[1](brentq(f, min(a, xn), max(a, xn), xtol=1e-10))), False
                    if xn in (tr[0](lo), tr[0](hi)):
                        break
        grid = np.linspace(tr[0](lo), tr[0](hi), 25)
        vals = np.array([f(x) for x in grid])
        ok = np.isfinite(vals)
        br = [i for i in range(len(grid) - 1) if ok[i] and ok[i + 1] and vals[i] * vals[i + 1] <= 0]
        if not br:
            return float(tr[1](grid[int(np.nanargmin(np.abs(np.where(ok, vals, np.nan))))])), True
        roots = [float(brentq(f, grid[i], grid[i + 1], xtol=1e-10)) if vals[i] != 0 else float(grid[i]) for i in br]
        return float(tr[1](min(roots, key=lambda x: abs(x - x0)))), False

    p = {k: float(start.get(k, 0.0)) for k in ('K', 'b', 'wp', 'wm')}
    order = ['b', 'K'] + [k for k in ('wp', 'wm') if k in fit]
    at_bound, hist = {}, []
    converged = False
    for it in range(1, max_iter + 1):
        prev = dict(p)
        for name in order:
            p[name], at_bound[name] = solve(p, name, it == 1)
        m = mu_unit(p)
        c = solve_c(m)
        st = vm_stats(c * m, sd)
        hist.append(dict(iteration=it, c=c, **p, **st))
        if it > 1 and abs(np.log(p['K']) - np.log(prev['K'])) < tol and all(abs(p[k] - prev[k]) < tol for k in ('b', 'wp', 'wm')):
            converged = True
            break
    return dict(p, c=c, iterations=it, converged=converged, **{f'{k}_at_bound': v for k, v in at_bound.items()}, **st,
        history=hist, targets=tg)


def permute_potentials(n: np.ndarray, u: np.ndarray, h: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """The (depolarization, hyperpolarization) pairs permuted across the KCs without a spike, within each
    odor (KCs x odors arrays): the same distribution per odor, no structure."""
    rng = np.random.default_rng(seed)
    u2, h2 = u.copy(), h.copy()
    for j in range(n.shape[1]):
        idx = np.flatnonzero(n[:, j] == 0)
        perm = rng.permutation(idx)
        u2[idx, j], h2[idx, j] = u[perm, j], h[perm, j]
    return u2, h2


def b_sims(args) -> None:
    """The deterministic APL-on and acute APL-off runs of every part B model with the membrane potential
    recorded, each checked against the archived spike counts (or, for an APL-effect fit, against its
    fits.csv rates and saved distances), reduced and saved to outputs/E68_readout_low_end/partB/<key>_vm.npz."""
    VmModel, E43 = make_vm_model_class()
    orn = H.apl16_orn_deltas()
    log = []
    for key, label, exp, tag in b_models():
        path = b_dir() / f'{key}_vm.npz'
        if path.exists():
            print(f'{key}: {path.name} exists', flush=True)
            continue
        t0 = datetime.now()
        name = E43.model_from(f'{exp}:{tag}')
        m = VmModel(name, orn, b_dir())
        checks = {}
        arch = ARCHIVE / exp
        for cond in E43.CONDS:
            p = arch / f'{tag}_{cond}_spike_counts.parquet'
            if p.exists():
                a = H.abbrev_columns(pd.read_parquet(p).astype(float))[ODORS].to_numpy()
                checks[f'{cond}_archived_mismatch'] = int((a != m.det[cond][1]).sum())
        if exp == E67_FOLDER:
            row = pd.read_csv(RESULTS / exp / 'fits.csv').set_index('tag').loc[tag]
            checks['rate_on_vs_fit'] = float(m.det['on'][0].mean() - row['rate_on'])
            checks['rate_off_vs_fit'] = float(m.det['acute_off'][0].mean() - row['rate_off'])
            pd_long = RESULTS / exp / 'model_pair_distances_long.csv'
            if pd_long.exists():
                d = pd.read_csv(pd_long)
                d = d[(d.tag == tag) & (d.rdm == 'spike_counts_cosine')] if 'rdm' in d else d[d.tag == tag]
                for cond, gcol in (('on', 'on'), ('acute_off', 'acute_off')):
                    dd = d[d.condition == gcol] if 'condition' in d else d.iloc[0:0]
                    if len(dd):
                        mine = as_pairs(cos_dist(m.det[cond][1].astype(float)))
                        theirs = H.canonical_pairs(dd.set_index(['odor_a', 'odor_b'])['distance'])
                        a_, b_ = H.align_pairs(mine, theirs)
                        checks[f'{cond}_distance_max_abs_diff_vs_fit'] = float(np.abs(a_ - b_).max())
        if any(v != 0 for k, v in checks.items() if k.endswith('mismatch')) or \
                any(abs(v) > 1e-9 for k, v in checks.items() if k.startswith('rate_')) or \
                any(v > 1e-9 for k, v in checks.items() if k.endswith('_vs_fit') and 'distance' in k):
            raise RuntimeError(f'{key}: the re-simulated model differs from its saved outputs: {checks}')
        arrs = {}
        for cond in E43.CONDS:
            resp, spk, red = m.run_vm(cond, m.f0, m.thr0(m.f0))
            checks[f'{cond}_vm_run_spike_mismatch'] = int((spk != m.det[cond][1]).sum())
            if checks[f'{cond}_vm_run_spike_mismatch']:
                raise RuntimeError(f'{key}: recording the potential changed the spike counts ({checks})')
            arrs[f'{cond}_spk'] = spk
            arrs.update({f'{cond}_{k}': v for k, v in red.items()})
        info = dict(key=key, label=label, experiment=exp, tag=tag, name=name, checks=checks, model_info=m.info,
            params={c: {k: float(v) for k, v in p.items()} for c, p in m.params.items()}, fixed_thr=m.f0,
            seconds=(datetime.now() - t0).seconds)
        np.savez_compressed(path, info=json.dumps(info), **arrs)
        print(f'{key}: {label}: checks {checks}; {info["seconds"]} s -> {path.name}', flush=True)
        log.append(info)
        m.close()
    (b_dir() / 'sims_log.json').write_text(json.dumps(log, indent=1, default=str) + '\n')


def load_vm(key: str) -> Dict[str, object]:
    z = np.load(b_dir() / f'{key}_vm.npz')
    out = {k: z[k] for k in z.files if k != 'info'}
    out['info'] = json.loads(str(z['info']))
    return out


def part_a_start(exp: str, tag: str) -> Dict[str, float]:
    """Part A's joint calibration (K, b) of an archived variant, the start of part B's."""
    p = H.results_dir(EXPERIMENT) / 'calibration.csv'
    if p.exists():
        c = pd.read_csv(p)
        r = c[(c.folder == exp) & (c.tag == tag) & (c.calibration == 'joint_floor')]
        if len(r):
            return dict(K=float(r.K.iloc[0]), b=float(r.floor.iloc[0]))
    return dict(K=10.0, b=0.0)


def single_trial_distance(trials: np.ndarray) -> float:
    """Mean over recordings, odors and trial pairs of 1 - cosine between two single trials of one odor
    ((recordings, trials, units, odors) array)."""
    vals = []
    for r in range(trials.shape[0]):
        for i in range(trials.shape[1]):
            for j in range(i + 1, trials.shape[1]):
                A, B = trials[r, i], trials[r, j]
                with np.errstate(invalid='ignore', divide='ignore'):
                    vals.append(1 - (A * B).sum(axis=0) / (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0)))
    return float(np.nanmean(vals))


def b_trials(args) -> None:
    """For the reference model: the per-KC threshold jitter (as the trial-to-trial variability experiment's
    part B, E43: thr0 x (1 + sigma xi), fixed_thr re-tuned to hold the APL-on response rate) re-calibrated
    so that the APL-on single-trial same-odor distance, under the readout with both potential terms
    (calibrated on the deterministic run) and measurement noise per trial, equals the control flies';
    then thirty trials (ten pseudo-recordings of three) with APL on and off and the potential recorded."""
    VmModel, E43 = make_vm_model_class()
    orn = H.apl16_orn_deltas()
    key, label, exp, tag = B_MODELS[0]
    out = b_dir() / f'{key}_trials.npz'
    if out.exists():
        print(f'{out.name} exists', flush=True)
        return
    det = load_vm(key)
    n0, (u0, h0) = det['on_spk'].astype(float), potentials({k[3:]: det[k] for k in ('on_base', 'on_mean', 'on_peak', 'on_thr')})
    cal = calibrate_vm(n0, u0, h0, fit=('wp', 'wm'), start=part_a_start(exp, tag))
    pr = {k: cal[k] for k in ('K', 'b', 'wp', 'wm', 'c')}
    print(f'{key}: deterministic calibration {pr}', flush=True)
    name = E43.model_from(f'{exp}:{tag}')
    m = VmModel(name, orn, b_dir())
    target_rate = float(m.det['on'][0].mean())
    target = H.data_target('ctrl_single_trial_floor')
    s = refs()['noise_sd']
    seed = 0
    history, tuned = [], {}

    def run_trials(sigma, fixed_thr, kind, rep, n, conds):
        xis = E43.b_draws(seed, name, kind, rep, n, m.draw_size)
        res = {c: dict(spk=[], base=[], mean=[], peak=[], thr=[]) for c in conds}
        for xi in xis:
            thr, _, _ = m.thr_trial(fixed_thr, sigma, xi)
            for c in conds:
                _, spk, red = m.run_vm(c, fixed_thr, thr)
                res[c]['spk'].append(spk)
                for k in ('base', 'mean', 'peak', 'thr'):
                    res[c][k].append(red[k])
        return {c: {k: np.stack(v) for k, v in d.items()} for c, d in res.items()}

    def evaluate(sigma):
        t0 = datetime.now()
        guess = None
        if tuned:
            near = min(tuned, key=lambda x: abs(x - sigma))
            guess = tuned[near]
        tn = m.tune(sigma, E43.b_draws(seed, name, 'tune', 0, JITTER_N_TUNE, m.draw_size), target_rate, guess, 0.01, 10)
        tuned[sigma] = tn['fixed_thr']
        tr = run_trials(sigma, tn['fixed_thr'], 'rec', 99, JITTER_N_CAL, ('on',))['on']
        u, h = potentials(tr)
        sig = pr['c'] * vm_readout(tr['spk'].astype(float), u, h, pr['K'], pr['b'], pr['wp'], pr['wm'])
        noisy = H.sample_pseudo_recordings(sig[None], s, seed=500, return_trials=True)[0]
        d = single_trial_distance(noisy[None])
        history.append(dict(sigma=sigma, fixed_thr=tn['fixed_thr'], rate=tn['rate'], tune_converged=tn['converged'],
            single_trial_dist_on=d, seconds=(datetime.now() - t0).seconds))
        print(f'{key}: sigma {sigma:.4f}: fixed_thr {tn["fixed_thr"]:.4g} (rate {tn["rate"]:.4f}, target {target_rate:.4f}); '
            f'single-trial distance {d:.3f} (target {target:.3f}); {history[-1]["seconds"]} s', flush=True)
        return d - target

    # regula falsi on sigma between no jitter and the level the trial-noise experiments matched on raw counts
    lo, hi = 0.0, 0.207
    f_lo, f_hi = evaluate(lo), evaluate(hi)
    sigma = hi
    if f_lo > 0:
        sigma = lo
    elif f_hi < 0:
        sigma = hi
    else:
        for _ in range(5):
            sigma = float(lo - f_lo * (hi - lo) / (f_hi - f_lo))
            f = evaluate(sigma)
            if abs(f) < 0.005:
                break
            if f > 0:
                hi, f_hi = sigma, f
            else:
                lo, f_lo = sigma, f
    best = min(history, key=lambda r: abs(r['single_trial_dist_on'] - target))
    sigma, fixed_thr = best['sigma'], best['fixed_thr']
    print(f'{key}: jitter sigma {sigma:.4f}, fixed_thr {fixed_thr:.6g}: 30 trials with APL on and off', flush=True)
    recs = [run_trials(sigma, fixed_thr, 'rec', rep, N_TRIALS, E43.CONDS) for rep in range(N_RECORDINGS)]
    arrs = {}
    for c in E43.CONDS:
        for k in ('spk', 'base', 'mean', 'peak', 'thr'):
            arrs[f'{c}_{k}'] = np.stack([r[c][k] for r in recs])
    info = dict(key=key, sigma=sigma, fixed_thr=fixed_thr, fixed_thr_det=m.f0, rate_target=target_rate, history=history,
        det_calibration=pr, seed=seed, single_trial_target=target)
    np.savez_compressed(out, info=json.dumps(info), **arrs)
    pd.DataFrame(history).to_csv(H.results_dir(EXPERIMENT) / 'partB_jitter_search.csv', index=False)
    m.close()
    print('->', out, flush=True)


def score_vm_config(n_on, n_off, pot_on, pot_off, cal, trials: bool, seeds=(SEED_ON, SEED_OFF)) -> Tuple[List[dict], dict, dict]:
    """Readout with the potential terms, measurement noise, every reading against the same reading of the
    data; (rows per reading, amplitude / cosine-term row, histograms per condition)."""
    s = refs()['noise_sd']
    out = {}
    amp, hists = {}, {}
    for cond, n, (u, h), seed in (('on', n_on, pot_on, seeds[0]), ('off', n_off, pot_off, seeds[1])):
        sig = cal['c'] * vm_readout(n, u, h, cal['K'], cal['b'], cal['wp'], cal['wm'])
        if trials:
            noisy = np.stack([H.sample_pseudo_recordings(sig[r][None], s, seed=seed * 1000 + r, return_trials=True)[0]
                for r in range(sig.shape[0])])
            dist = readings_noisy(noisy, None)
            amp[f'single_trial_dist_{cond}'] = single_trial_distance(noisy)
        else:
            noisy = H.sample_pseudo_recordings(sig, s, n_recordings=N_RECORDINGS, n_trials=N_TRIALS, seed=seed, return_trials=True)
            dist = readings_noisy(noisy, sig)
        means = noisy.mean(axis=1)
        pooled = means.ravel()
        amp.update({f'{k}_{cond}': v for k, v in amplitude_summary(pooled, refs()['floor_z']).items()})
        amp.update({f'term_{k}_{cond}': v for k, v in terms_of(list(means)).items()})
        hists[cond] = hist_shares(pooled)
        out[cond] = dist
    rows = []
    for reading in list(READINGS) + ['graded_sampled']:
        rows.append(dict(reading=reading, **score_pair(as_pairs(out['on'][reading]), as_pairs(out['off'][reading]), reading)))
    return rows, amp, hists


def b_score(args) -> None:
    """Calibrates and scores every part B model under VM_CONFIGS (and the jittered trials of the reference
    model when saved); writes the part B tables and figure."""
    res = H.results_dir(EXPERIMENT)
    rows, cal_rows, amp_rows, hist_rows, cal_hist = [], [], [], [], []
    models = [(k, l, e, t) for k, l, e, t in b_models() if (b_dir() / f'{k}_vm.npz').exists()]
    for key, label, exp, tag in models:
        d = load_vm(key)
        n = {c: d[f'{c}_spk'].astype(float) for c in ('on', 'acute_off')}
        red = {c: {k: d[f'{c}_{k}'] for k in ('base', 'mean', 'peak', 'thr')} for c in ('on', 'acute_off')}
        start = part_a_start(exp, tag)
        cals = {}
        for name, fit, measure, which in VM_CONFIGS:
            pot = {c: potentials(red[c], measure) for c in n}
            if isinstance(fit, str):
                cal = cals[fit]
            else:
                cal = calibrate_vm(n['on'], *pot['on'], fit=fit, start=start)
                cals[name] = cal
                cal_rows.append(dict(model=key, label=label, config=name, **{k: v for k, v in cal.items() if k not in ('history', 'targets')}))
                cal_hist += [dict(model=key, config=name, **h) for h in cal['history']]
            if which == 'permuted':
                pot = {c: permute_potentials(n[c], *pot[c], seed=7) for c in n}
            elif which == 'apl_on':
                pot = {'on': pot['on'], 'acute_off': pot['on']}
            r, amp, hists = score_vm_config(n['on'], n['acute_off'], pot['on'], pot['acute_off'], cal, False)
            base = dict(model=key, label=label, config=name, K=cal['K'], floor=cal['b'], c=cal['c'], wp=cal['wp'], wm=cal['wm'])
            rows += [dict(base, **x) for x in r]
            amp_rows.append(dict(base, **amp))
            for cond, hh in hists.items():
                hist_rows.append(dict(base, condition=cond, **dict(zip(hist_labels(), hh))))
            print(f"{key} {name}: K {cal['K']:.2f} b {cal['b']:+.2f} c {cal['c']:.2f} wp {cal['wp']:.3f} wm {cal['wm']:.3f}; graded loss "
                f"{[x for x in r if x['reading'] == 'graded'][0]['rmse']:.3f}", flush=True)
    # the jittered trials of the reference model
    tp = b_dir() / f'{B_MODELS[0][0]}_trials.npz'
    if tp.exists():
        z = np.load(tp)
        info = json.loads(str(z['info']))
        n = {c: z[f'{c}_spk'].astype(float) for c in ('on', 'acute_off')}
        red = {c: {k: z[f'{c}_{k}'] for k in ('base', 'mean', 'peak')} for c in n}
        for c in n:
            red[c]['thr'] = z[f'{c}_thr']
        pot = {c: potentials(red[c]) for c in n}
        start = part_a_start(B_MODELS[0][2], B_MODELS[0][3])
        for name, fit in (('floor_trials', ()), ('vm_trials', ('wp', 'wm'))):
            cal = calibrate_vm(n['on'], *pot['on'], fit=fit, start=start)
            cal_rows.append(dict(model='ref40_trials', label=f"the reference model, thirty jittered trials (sigma {info['sigma']:.3f})", config=name,
                **{k: v for k, v in cal.items() if k not in ('history', 'targets')}))
            cal_hist += [dict(model='ref40_trials', config=name, **h) for h in cal['history']]
            r, amp, hists = score_vm_config(n['on'], n['acute_off'], pot['on'], pot['acute_off'], cal, True)
            base = dict(model='ref40_trials', label='the reference model, jittered trials', config=name, K=cal['K'], floor=cal['b'], c=cal['c'],
                wp=cal['wp'], wm=cal['wm'], sigma=info['sigma'])
            rows += [dict(base, **x) for x in r]
            amp_rows.append(dict(base, **amp))
            for cond, hh in hists.items():
                hist_rows.append(dict(base, condition=cond, **dict(zip(hist_labels(), hh))))
    df, cal_df, amp_df, hist_df = pd.DataFrame(rows), pd.DataFrame(cal_rows), pd.DataFrame(amp_rows), pd.DataFrame(hist_rows)
    df.to_csv(res / 'partB_by_model.csv', index=False)
    cal_df.to_csv(res / 'partB_calibration.csv', index=False)
    pd.DataFrame(cal_hist).to_csv(res / 'partB_calibration_history.csv', index=False)
    amp_df.to_csv(res / 'partB_amplitude.csv', index=False)
    hist_df.to_csv(res / 'partB_histograms.csv', index=False)
    write_table_b(res, df, cal_df, amp_df)
    figure_b(res, hist_df, df)
    sims = json.loads((b_dir() / 'sims_log.json').read_text()) if (b_dir() / 'sims_log.json').exists() else []
    vm_infos = {k: load_vm(k)['info'] for k, *_ in models}
    git = lambda *a: subprocess.run(['git', *a], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E68_readout_low_end.py --part B',
        host=subprocess.run(['hostname', '-s'], capture_output=True, text=True).stdout.strip(), al_analysis_commit=git('rev-parse', 'HEAD'),
        al_analysis_dirty=bool(git('status', '--porcelain', '--untracked-files=no')),
        model_provenance=H.model_provenance(orn_deltas=H.apl16_orn_deltas()), models=vm_infos, sims_log=sims,
        trials=json.loads(str(np.load(tp)['info'])) if tp.exists() else None, vm_configs=[list(map(str, c)) for c in VM_CONFIGS],
        baseline_window_s=[-VM_PRE_S, 0.0], response_window='mp.time_stim_start to mp.time_stim_end (as the trial-noise experiment, E43)',
        e67_selection=[list(x) for x in e67_best()], targets=dict(refs()['targets'], band025=H.data_target('ctrl_share_z_025_1'),
        below=H.data_target('ctrl_share_z_below_m025')), noise_sd=refs()['noise_sd'],
        outputs_local=str(b_dir()))
    (res / 'provenance_partB.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('->', res)


def write_table_b(res: Path, df: pd.DataFrame, cal: pd.DataFrame, amp: pd.DataFrame) -> None:
    dt = H.data_target
    dmean = {v: float(refs()[v]['effect'].mean()) for v in READINGS}
    s = pd.read_csv(res / 'data_summary_by_genotype.csv', index_col=0)
    T = pd.read_csv(res / 'data_effect_terms.csv', index_col=0)['observed']
    L = ['# E68 part B: a sub-threshold signal from the membrane potential (tables)', '']
    L += ['## Calibrated parameters (APL-on counts and potential, control-fly targets, through the noise)', '',
        '| model | configuration | K | b | c | wp | wm | iterations | converged | at a bound | rate | 0.5-1.5 z | 0.25-1 z | below -0.25 z |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|']
    for r in cal.itertuples():
        ab = ', '.join(k.replace('_at_bound', '') for k in cal.columns if k.endswith('_at_bound') and pd.notna(getattr(r, k)) and bool(getattr(r, k)))
        L.append(f'| {r.model} | {r.config} | {r.K:.2f} | {r.b:+.2f} | {r.c:.2f} | {r.wp:.3f} | {r.wm:.3f} | {r.iterations} | {r.converged} | {ab} | '
            f'{pct(r.rate)} | {pct(r.band_share)} | {pct(r.band025)} | {pct(r.below)} |')
    L += ['', f"Targets (control flies): rate {pct(dt('ctrl_frac'))}, 0.5-1.5 z {pct(dt('ctrl_share_z_05_15'))}, 0.25-1 z "
        f"{pct(dt('ctrl_share_z_025_1'))}, below -0.25 z {pct(dt('ctrl_share_z_below_m025'))}, responders-only participation ratio "
        f"{pct(dt('ctrl_participation_ratio_resp'))}.", '']
    L += ['## Predictions with APL off (the low end the potential terms add) beside the KC data', '',
        '| model | configuration | 0.5-1.5 z, APL on / off | 0.25-1 z | below -0.25 z | above 4 z | responders\' mean amplitude | both responders, APL on / off | one responder | both sub-threshold |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    L.append(f"| KC data | control flies / APL output blocked (TNT) | {pct(s.loc['share_05_15', 'TNT3c'])} / {pct(s.loc['share_05_15', 'TNT'])} | "
        f"{pct(s.loc['share_025_1', 'TNT3c'])} / {pct(s.loc['share_025_1', 'TNT'])} | {pct(s.loc['share_below_m025', 'TNT3c'])} / {pct(s.loc['share_below_m025', 'TNT'])} | "
        f"{pct(s.loc['share_above_4', 'TNT3c'])} / {pct(s.loc['share_above_4', 'TNT'])} | {s.loc['resp_amp_mean', 'TNT3c']:.2f} / {s.loc['resp_amp_mean', 'TNT']:.2f} | "
        f"{T['both__ctrl']:.3f} / {T['both__tnt']:.3f} | {T['one__ctrl']:.3f} / {T['one__tnt']:.3f} | {T['neither__ctrl']:.3f} / {T['neither__tnt']:.3f} |")
    for r in amp.itertuples():
        g = lambda k: getattr(r, k)
        L.append(f"| {r.model} | {r.config} | {pct(g('share_05_15_on'))} / {pct(g('share_05_15_off'))} | {pct(g('share_025_1_on'))} / {pct(g('share_025_1_off'))} | "
            f"{pct(g('share_below_m025_on'))} / {pct(g('share_below_m025_off'))} | {pct(g('share_above_4_on'))} / {pct(g('share_above_4_off'))} | "
            f"{g('resp_amp_mean_on'):.2f} / {g('resp_amp_mean_off'):.2f} | {g('term_both_on'):.3f} / {g('term_both_off'):.3f} | "
            f"{g('term_one_on'):.3f} / {g('term_one_off'):.3f} | {g('term_neither_on'):.3f} / {g('term_neither_off'):.3f} |")
    L += ['', '## Mean APL effect (loss) under each reading, against the same reading of the data', '',
        '| model | configuration | ' + ' | '.join(READING_LABEL[v] for v in READINGS) + ' | order of the readings as the data\'s |',
        '|---|---|' + '---:|' * len(READINGS) + '---|']
    L.append('| KC data | | ' + ' | '.join(f'{dmean[v]:+.3f}' for v in READINGS) + ' | thresholded < graded < noise-corrected |')
    for (model, cfg), g in df.groupby(['model', 'config'], sort=False):
        g = g.set_index('reading')
        m = {v: g.loc[v, 'mean'] for v in READINGS}
        order = m['thresholded'] < m['graded'] < m['noise_corrected']
        L.append(f'| {model} | {cfg} | ' + ' | '.join(f"{m[v]:+.3f} ({g.loc[v, 'rmse']:.3f})" for v in READINGS) + f" | {'yes' if order else 'no'} |")
    L += ['', '## The loss and its parts against the graded data', '',
        '| model | configuration | loss | mean effect | per-odor part | pair-specific part | ρ on the per-pair APL effect | ρ on odor-odor distances, APL on / off | block | natural-source alignment, APL on / off / on minus off |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in df[df.reading == 'graded'].itertuples():
        L.append(f"| {r.model} | {r.config} | {r.rmse:.3f} | {r.mean:+.3f} | {r.per_odor:.3f} | {r.pair_specific:.3f} | {r.spearman:.2f} | {r.rho_on:.2f} / {r.rho_off:.2f} | "
            f"{r.block:+.3f} | {r.vcf_on:.2f} / {r.vcf_off:.2f} / {r.vcf_diff:+.2f} |")
    L += ['', f"KC data: mean effect {dmean['graded']:+.3f}, loss floor {dt('apl_effect_rmse_floor'):.3f}, ρ ceiling {dt('apl_effect_rho_ceiling'):.2f}; "
        f"block {dt('block'):+.3f}; natural-source alignment {dt('ctrl_vcf'):.2f} / {dt('tnt_vcf'):.2f} / {dt('vcf_apl_effect'):+.2f}.", '']
    if 'single_trial_dist_on' in amp:
        tr = amp[amp.single_trial_dist_on.notna()]
        L += ['## The jittered trials of the reference model', '', '| configuration | single-trial distance, APL on / off |', '|---|---:|']
        L += [f'| {r.config} | {r.single_trial_dist_on:.3f} / {r.single_trial_dist_off:.3f} |' for r in tr.itertuples()]
        L += ['', f"KC data: {dt('ctrl_single_trial_floor'):.3f} (control flies, the jitter's target) / {dt('tnt_single_trial_floor'):.3f} (APL output blocked (TNT), a prediction).", '']
    (res / 'comparison_table_partB.md').write_text('\n'.join(L) + '\n')


def figure_b(res: Path, hist: pd.DataFrame, df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = hist_labels()
    x = np.arange(len(labels))
    dh = pd.read_csv(res / 'data_hist_by_genotype.csv', index_col=0)
    models = list(dict.fromkeys(hist.model))
    fig, axes = plt.subplots(len(models), 2, figsize=(14, 3.2 * len(models)), sharey=True, squeeze=False)
    cfgs = [('floor_noise', '#9ecae1'), ('vm', '#2171b5'), ('vm_permuted', '#fd8d3c'), ('vm_on_potential', '#a1d99b'),
        ('floor_trials', '#9ecae1'), ('vm_trials', '#2171b5')]
    for i, mdl in enumerate(models):
        for j, (cond, g) in enumerate((('on', 'TNT3c'), ('off', 'TNT'))):
            ax = axes[i, j]
            ax.bar(x - 0.36, dh[g].reindex(labels), 0.18, color='k', label=f'KC data, {GENO_LABEL[g]}')
            k = 0
            for cfg, col in cfgs:
                h = hist[(hist.model == mdl) & (hist.config == cfg) & (hist.condition == cond)]
                if len(h):
                    ax.bar(x - 0.18 + 0.18 * k, h[labels].iloc[0].to_numpy(dtype=float), 0.18, color=col, label=f'{VM_LABEL.get(cfg, cfg)}, APL {cond}')
                    k += 1
            ax.set_yscale('log'); ax.set_ylim(1e-4, 1)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            ax.set_title(f'{mdl}: APL {cond} / {GENO_LABEL[g]}', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            if i == 0:
                ax.legend(fontsize=6.5, frameon=False)
        axes[i, 0].set_ylabel('share of KC x odor entries')
    fig.suptitle('Amplitude histograms of the trial means (z) with the membrane-potential terms: the APL-off column is a prediction', fontsize=10)
    fig.tight_layout(); fig.savefig(res / 'E68_partB_histograms.png', dpi=120); plt.close(fig)


def main_b(args) -> None:
    steps = ['sims', 'trials', 'score'] if args.b_step == 'all' else [args.b_step]
    for st in steps:
        print(f'part B: {st} ({datetime.now():%Y-%m-%d %H:%M})', flush=True)
        {'sims': b_sims, 'trials': b_trials, 'score': b_score}[st](args)


if __name__ == '__main__':
    main()
