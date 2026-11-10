#!/usr/bin/env python3
"""E73: the recordings' per-KC normalization on the model's side. Whether Yang's KC values are z-scored
cell by cell over the whole recording, what that does to the amplitudes and the response criterion, and
what the models score when read the same way (no simulations).

Part A, the data (`--part A`): per recording of the APL16 panel (19 recordings: 11 control flies, 8 APL
output blocked (TNT)), the low-concentration diagnostic mixtures (diag_mix, 19) and the dilute natural
mixtures (natmix_dilute, 9), from Yang's per-trial values: each KC's mean of squared trial values over
the whole recording (every odor, the solvent included) with its upper edge (maximum, 99.9th and 99th
percentiles, the gap between the maximum and the 99.9th percentile), the largest single-trial value and
trial mean, and the same split by the number of odors the KC responds to (trial mean > 1 z); the mean of
each KC's non-responding entries against its responses (a z-score over the whole recording without a
baseline subtraction puts them below 0 by the KC's own mean). Synthetic cells with each recording's trial
structure and a heavy-tailed amplitude distribution are normalized three ways and compared with the data:
a z-score over the whole recording (an edge at 1 / q, q the share of the recording inside the response
windows), a z-score per trial (an edge on every single trial) and a z-score against a pre-odor baseline
(no edge). q with an interval from the edge and, independently, from the largest trial means of KCs that
respond to a single odor. Writes data_edge_per_recording.csv, data_edge_by_breadth.csv, synthetic_*.csv,
q_estimates.csv, the figures E73_edge.png and E73_breadth.png, provenance_A.json.

Yang confirmed the premise on 2026-11-10 (add_to_reports/yang_response_definition.md): each KC's Fc is
z-scored over the whole recording and a trial's response is the mean z 2-8 s after onset minus the mean over
the 5 s before; part A is kept as the independent check from the per-trial values.

Part B, the models (`--part B`): every archived variant with both conditions saved
(data/model_outputs/<folder>/<tag>_<on|acute_off>_spike_counts.parquet), read as the recordings appear to
be (`kc_remapping_harness.session_zscore`: per model KC a session of the panel's 16 odors and the solvent,
3 trials each, its spike counts through an amplitude map, linear or E65's saturating n / (n + K), the
response confined to its window, which holds the share q = 51 x 18 / 7194 = 0.128 of the recording as in
Yang's APL16 recordings (`zcell_q_window`; sensitivity at part A's footprint 0.18 and its interval), one
baseline-noise term for all KCs and a slow drift (the silent KCs' whole-trace SD is 1.40 baseline SDs),
z-scored over the KC's session, the response relative to the pre-odor baseline; trial means, the data's criterion (above 1 z) and the data's cosine on all values; in expectation
over the measurement noise), calibrated on the APL-on counts alone (`calibrate_session_zscore`: the noise
term, and K for the saturating map, so that the APL-on response rate under the data's criterion is the
control flies' 6.8 % and the responders' mean amplitude 2.00 z; the noise of a silent KC's trial set to
the control flies' 0.385 z) and applied unchanged to APL off. Scored against the same reading of the data
(graded, sub-threshold values set to 0, binarized): the APL-effect loss with its mean, per-odor and
pair-specific parts and ρ on the per-pair APL effect; the mean distances; under the data's criterion the
response rates, their ratio, the response rate ρ in both conditions, the odors-per-KC distributions with
their distances from the flies' (bands 0.167 / 0.196), the silent share and the share at eight or more
odors; the responders' mean amplitude and coefficient of variation, the entries above 4 z and in the 0.5
to 1.5 z band; the block; the two ρ's on odor-odor distances; the natural-source alignment (APL on / off /
on minus off); beside the present readings (raw counts and E65's calibrated readout, two-spike criterion).
Controls: a fixed divisor for all KCs reproduces the present readings; q over part A's interval; the reading
applied to a model shows part A's edge; a simulated whole-recording trace with Yang's trial structure reads as
the closed form (`trace_reading`); the breadth checks (whole-trace over baseline-noise SD, and the responders'
amplitude in baseline-noise units, by breadth) against viki's measurements. Also the reading of E49's low-concentration panels within their own
session. Writes by_variant.csv (one row per variant x reading definition x reading, with the calibrated
parameters and both conditions' statistics under that reading's criterion), controls.csv, low_conc.csv,
provenance_B.json (with the model provenance).

`--part summarize`: comparison_table.md (the key models, the order of the variants under the reading
against the present order, the best-models columns, the firing ceiling's gain, the wiring population and
the input-scaled threshold's settings, as archived; the low-concentration panels), key_models.csv, order.csv,
best_models_columns.csv, map_choice.csv (the map kept: the one whose APL-on amplitude statistics are nearer the
control flies'), E73_loss_present_vs_reading.png, E73_key_models.png, provenance_summary.json.

Outputs in <shared>/results/E73_per_kc_normalization/. Nothing is simulated.
Usage: nice -n 10 ./venv/bin/python -u E73_per_kc_normalization.py --part A|B|summarize [--jobs 16] [--only-key]
       [--folders E71_input_scaled_threshold E72_wiring_population] (part B: add folders archived later, then --part summarize)
"""
from __future__ import annotations

import os

# (only when run as a script: E10_joint_fit.py and E43_model_trial_noise.py import this module for the reading,
# and must keep the thread count of their simulations; the worker processes inherit the setting)
if __name__ == '__main__':
    for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(_v, '1')

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E73_per_kc_normalization'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
RESULTS = H.shared_dir() / 'results'
WEAK = H.shared_dir() / 'data' / 'yang_weak_odor_kc'
PANELS = ('APL16', 'diag_mix', 'natmix_dilute')


CONTROL_STIMULI = ('pfo @ 0', 'air', 'pfo')
GENO_LABEL = {'TNT3c': 'control flies', 'TNT': 'APL output blocked (TNT)', 'control': 'control flies',
    'apl_output_blocked': 'APL output blocked (TNT)'}
# breadth bins (number of non-control stimuli with a trial mean > 1 z)
BREADTH_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 10), (11, None))
WEAK_Z = 0.3            # entries with |trial mean| below this give a KC's own measurement noise
TOP_EDGE_SHARE = 0.01   # "the edge": the top 1 % of a recording's KCs by mean square
TOP_SINGLE = 10         # the single-odor responders with the largest trial means, per recording
NORMALIZATIONS = ('session', 'session_centered', 'trial', 'baseline')
NORM_LABEL = {'session': 'z over the whole recording, response relative to the pre-odor frames',
    'session_centered': 'z over the whole recording, window mean (no baseline subtracted)',
    'trial': 'z within each trial', 'baseline': 'against the pre-odor frames (sd of the baseline)'}
SYNTH_Q = (0.10, 0.17, 0.25)
SYNTH_W = 7             # response-window frames (white frame noise: a window mean's sd 1 / sqrt(7) = 0.38 of the frame sd)


def _breadth_label(lo, hi) -> str:
    return f'{lo}' if hi == lo else (f'{lo}+' if hi is None else f'{lo}-{hi}')


def load_panel_trials(panel: str) -> Dict[tuple, dict]:
    """Per-trial values of one panel, per recording: {recording: dict(X=trials x KCs array of every trial of
    the recording (stimuli in sorted order, repeats in order), stim=per-trial stimulus labels,
    genotype ('TNT3c' / 'TNT' for APL16; 'control' / 'apl_output_blocked' for the low-concentration panels),
    n_kcs)}. KCs with a missing trial are dropped (none in these files)."""
    if panel == 'APL16':
        parts = []
        for g, ds in (('TNT3c', 20), ('TNT', 19)):
            d = pd.read_csv(H.yang_data_dir() / g / f'dataset_{ds}_long.csv')
            d['genotype'] = g
            d['recording'] = d['date'].astype(str) + '_' + d['fly_num'].astype(str) + d['hemibrain'].astype(str)
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
    else:
        d = pd.read_csv(WEAK / panel / f'{panel}_long.csv')
        d['genotype'] = d['genotype_group']
    out = {}
    for rec, s in d.groupby('recording', sort=True):
        w = s.pivot_table(index='roi', columns=['odor', 'repeat'], values='value', aggfunc='first').sort_index(axis=1)
        w = w.dropna(axis=0)
        out[(panel, rec)] = dict(X=w.to_numpy(dtype=float).T, stim=np.asarray(w.columns.get_level_values(0)),
            genotype=str(s['genotype'].iloc[0]), n_kcs=int(w.shape[0]))
    return out


def _per_stimulus(X: np.ndarray, stim: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """(stimuli, trial means KCs x stimuli, trial variances (ddof 1, NaN for one trial), trials per stimulus)."""
    names = list(dict.fromkeys(stim))
    tm = np.stack([X[stim == o].mean(axis=0) for o in names], axis=1)
    var = np.stack([X[stim == o].var(axis=0, ddof=1) if (stim == o).sum() > 1 else np.full(X.shape[1], np.nan)
        for o in names], axis=1)
    n = np.array([(stim == o).sum() for o in names])
    return names, tm, var, n


def q_from_budget(M, R, t2, r0) -> np.ndarray:
    """Per KC, the share q of the session inside the response windows implied by its variance budget under
    a z-score over the whole recording (responses relative to the pre-odor frames): the session variance in
    z units is 1 = q (M - t^2) - q^2 R^2 + t^2 / r0^2 (M the mean of the KC's squared trial values, R their
    mean, t its trial-to-trial noise sd, r0 a silent KC's, so t^2 / r0^2 is the frame noise's share of the
    session variance); the smaller root, NaN where there is none."""
    M, R, t2 = (np.asarray(v, dtype=float) for v in (M, R, t2))
    a, b, c = R ** 2, -(M - t2), 1 - t2 / r0 ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        disc = b ** 2 - 4 * a * c
        q = np.where(a > 1e-12, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), c / -b)
    return np.where((disc >= 0) & (-b > 0), q, np.nan)


def edge_stats(X: np.ndarray, stim: np.ndarray) -> Tuple[dict, pd.DataFrame]:
    """One recording's edge statistics.

    Args:
        X: trials x KCs per-trial values (z).
        stim: per-trial stimulus labels (`CONTROL_STIMULI` are the solvent / air).

    Returns (summary dict, per-KC DataFrame). Per KC: `ms` (mean of the squared trial values over every
    trial), `mean` (their mean), `breadth` (non-control stimuli with a trial mean > 1 z), `t2` (the KC's
    own measurement noise: mean trial-to-trial variance of its entries with |trial mean| < `WEAK_Z`,
    solvent included), `max_tm`, `nonresp_mean` (mean trial mean of its non-control entries <= 1 z),
    `blank_mean` (the solvent's trial mean), `amp` (its responders' mean trial mean). Summary: the mean
    square's maximum, 99.9th / 99th percentiles, median and the gap between the maximum and the 99.9th
    percentile; the largest single trial and trial mean (and of the KCs responding to one odor); r0 (the
    root mean measurement-noise variance of the KCs responding to nothing); q from the noise budget (`q_noise`:
    the KCs' measurement noise against their mean square, least squares t^2 = r0^2 - r0^2 q (ms - t^2), so q
    = -slope / intercept), from the edge (`q_edge`: `q_from_budget` of the top 1 % by mean square, median)
    and from the single-odor responders (`q_single`: the same of the `TOP_SINGLE` with the largest trial
    means); and the bounds 1 / max mean square (`q_edge_bound`) and the largest single-odor trial mean's
    (`q_single_bound`: a noise-free KC responding to a share p of the trials reads at most
    1 / sqrt(q p (1 - q p)))."""
    names, tm, var, n = _per_stimulus(X, stim)
    ctrl = np.array([o in CONTROL_STIMULI for o in names])
    resp = (tm > 1.0) & ~ctrl[None]
    breadth = resp.sum(axis=1)
    weak = np.abs(tm) < WEAK_Z
    with np.errstate(invalid='ignore'):
        t2 = np.nanmean(np.where(weak, var, np.nan), axis=1)
        nonresp_mean = np.nanmean(np.where(~resp & ~ctrl[None], tm, np.nan), axis=1)
        amp = np.nanmean(np.where(resp, tm, np.nan), axis=1)
    ms = (X ** 2).mean(axis=0)
    mean = X.mean(axis=0)
    blank = tm[:, ctrl].mean(axis=1) if ctrl.any() else np.full(X.shape[1], np.nan)
    max_tm = np.where(ctrl[None], -np.inf, tm).max(axis=1)
    kc = pd.DataFrame(dict(ms=ms, mean=mean, breadth=breadth, t2=t2, max_tm=max_tm, nonresp_mean=nonresp_mean,
        blank_mean=blank, amp=amp))
    silent = (breadth == 0) & np.isfinite(t2)
    r0 = float(np.sqrt(np.mean(t2[silent])))
    ok = np.isfinite(t2)
    x, y = ms[ok] - t2[ok], t2[ok]
    slope, icpt = np.polyfit(x, y, 1)
    kc['q_kc'] = q_from_budget(ms, mean, t2, r0)
    top = kc.ms >= np.quantile(ms, 1 - TOP_EDGE_SHARE)
    single = kc[(kc.breadth == 1) & np.isfinite(kc.t2)].nlargest(TOP_SINGLE, 'max_tm')
    p_single = np.median(n[~ctrl]) / len(stim)
    m1 = float(kc.loc[kc.breadth == 1, 'max_tm'].max()) if (kc.breadth == 1).any() else np.nan
    with np.errstate(invalid='ignore'):
        qp = (1 - np.sqrt(max(1 - 4 / m1 ** 2, 0.0))) / 2 if np.isfinite(m1) else np.nan
    q99 = float(np.quantile(ms, 0.999))
    summ = dict(n_kcs=X.shape[1], n_trials=X.shape[0], n_stimuli=len(names), ms_max=float(ms.max()), ms_p999=q99,
        ms_p99=float(np.quantile(ms, 0.99)), ms_median=float(np.median(ms)), edge_gap=float(ms.max() - q99),
        max_trial=float(X.max()), max_trial_mean=float(max_tm.max()), max_trial_mean_single=m1,
        rate=float(resp[:, ~ctrl].mean()), resp_amp_mean=float(tm[resp].mean()), r0=r0,
        q_noise=float(-slope / icpt), r0_fit=float(np.sqrt(icpt)) if icpt > 0 else np.nan,
        q_edge=float(np.nanmedian(kc.loc[top, 'q_kc'])), q_single=float(np.nanmedian(single['q_kc'])) if len(single) else np.nan,
        q_edge_bound=float(1 / ms.max()), q_single_bound=float(qp / p_single) if np.isfinite(qp) else np.nan,
        share_trials_single_odor=float(p_single), blank_mean=float(np.nanmean(blank)),
        ms_max_over_inv_q=np.nan)
    return summ, kc


def breadth_table(kc: pd.DataFrame) -> pd.DataFrame:
    """Per breadth bin: KCs, the mean square's median and maximum, the responders' mean amplitude, the mean
    of the non-responding entries and of the solvent, the KCs' measurement noise (root mean variance of
    the weak entries) and q_kc's median among KCs with half or more of their session variance from signal."""
    rows = []
    for lo, hi in BREADTH_BINS:
        m = (kc.breadth >= lo) & ((kc.breadth <= hi) if hi is not None else True)
        g = kc[m]
        if not len(g):
            continue
        rows.append(dict(breadth=_breadth_label(lo, hi), n=len(g), ms_median=g.ms.median(), ms_max=g.ms.max(),
            amp=g.amp.mean(), nonresp_mean=g.nonresp_mean.mean(), blank_mean=g.blank_mean.mean(),
            noise_sd=float(np.sqrt(g.t2.mean())), q_kc_median=float(g.q_kc.median())))
    return pd.DataFrame(rows)


def ms_band_table(kc: pd.DataFrame) -> pd.DataFrame:
    """The KCs' measurement noise against their mean square (bins of the mean square): the noise-budget
    test, flat under a normalization by the baseline, falling as sqrt(1 - q (ms - t^2)) under a z-score
    over the whole recording."""
    rows = []
    for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, np.inf)):
        g = kc[(kc.ms > lo) & (kc.ms <= hi)]
        if len(g) >= 3:
            rows.append(dict(ms_bin=f'({lo:g}, {hi:g}]', n=len(g), ms_mean=g.ms.mean(), noise_sd=float(np.sqrt(g.t2.mean())),
                breadth_mean=g.breadth.mean()))
    return pd.DataFrame(rows)


def synthetic_recording(n_kcs: int, stim: np.ndarray, *, q: float, how: str, seed: int, breadth: np.ndarray,
    gain: float, gain_sd: float = 0.6, odor_sd: float = 0.5, trial_sd: float = 0.3, w: int = SYNTH_W) -> np.ndarray:
    """Per-trial values (trials x KCs) of synthetic cells with a recording's trial structure (`stim`, the
    per-trial labels; `CONTROL_STIMULI` evoke nothing).

    Each trial is T = w / q frames: a pre-odor baseline (the first (T - w) // 2 frames), a response window
    of w frames and a tail; the session's response windows thus hold the share q of its frames. Cell i
    responds to `breadth[i]` of the non-control stimuli (drawn at random), with amplitude gain x g_i x u_io
    x jitter in the window (lognormal: g_i over cells with sd `gain_sd`, u_io over its stimuli with sd
    `odor_sd`, the jitter over trials with sd `trial_sd`, all in log units: a heavy-tailed amplitude
    distribution), 0 elsewhere, plus white frame noise of sd 1. Normalized as `how` says
    (`NORMALIZATIONS`): 'session' (z over the whole recording, the window mean minus the pre-odor mean),
    'session_centered' (the same z, the window mean), 'trial' (z within each trial, window minus pre-odor
    mean) or 'baseline' (window minus pre-odor mean over the sd of the cell's pre-odor frames)."""
    rng = np.random.default_rng(seed)
    T = max(int(round(w / q)), w + 4)
    n_pre = (T - w) // 2
    names = [o for o in dict.fromkeys(stim) if o not in CONTROL_STIMULI]
    col = {o: j for j, o in enumerate(names)}
    sig = np.zeros((n_kcs, len(names)))
    g = gain * np.exp(rng.normal(0, gain_sd, n_kcs))
    for i in range(n_kcs):
        k = int(min(breadth[i], len(names)))
        if k:
            j = rng.choice(len(names), k, replace=False)
            sig[i, j] = g[i] * np.exp(rng.normal(0, odor_sd, k))
    n_tr = len(stim)
    a = np.zeros((n_kcs, n_tr))
    for t, o in enumerate(stim):
        if o in col:
            a[:, t] = sig[:, col[o]] * np.exp(rng.normal(0, trial_sd, n_kcs))
    F = rng.normal(0, 1, (n_kcs, n_tr, T)).astype(np.float32)
    F[:, :, n_pre:n_pre + w] += a[:, :, None].astype(np.float32)
    win, pre = slice(n_pre, n_pre + w), slice(0, n_pre)
    if how in ('session', 'session_centered'):
        flat = F.reshape(n_kcs, -1)
        z = ((flat - flat.mean(axis=1, keepdims=True)) / flat.std(axis=1, keepdims=True)).reshape(F.shape)
        r = z[:, :, win].mean(axis=2) - (z[:, :, pre].mean(axis=2) if how == 'session' else 0.0)
    elif how == 'trial':
        z = (F - F.mean(axis=2, keepdims=True)) / F.std(axis=2, keepdims=True)
        r = z[:, :, win].mean(axis=2) - z[:, :, pre].mean(axis=2)
    elif how == 'baseline':
        b = F[:, :, pre]
        sd = (b - b.mean(axis=2, keepdims=True)).reshape(n_kcs, -1).std(axis=1, ddof=1)
        r = (F[:, :, win].mean(axis=2) - b.mean(axis=2)) / sd[:, None]
    else:
        raise ValueError(f'unknown normalization {how!r} (one of {NORMALIZATIONS})')
    return r.T.astype(float)


def _tune_gain(stim, breadth, *, q, how, target_amp, seed) -> float:
    """The amplitude scale at which the synthetic responders' mean trial mean (> 1 z) is `target_amp`
    (bisection in log gain on 600 cells)."""
    lo, hi = np.log(0.05), np.log(50.0)
    sub = breadth[:600]

    def amp(lg):
        X = synthetic_recording(len(sub), stim, q=q, how=how, seed=seed, breadth=sub, gain=float(np.exp(lg)))
        _, tm, _, _ = _per_stimulus(X, stim)
        ctrl = np.array([o in CONTROL_STIMULI for o in dict.fromkeys(stim)])
        v = tm[:, ~ctrl]
        return float(v[v > 1].mean()) if (v > 1).any() else 0.0

    flo, fhi = amp(lo) - target_amp, amp(hi) - target_amp
    if flo * fhi > 0:
        return float(np.exp(lo if abs(flo) < abs(fhi) else hi))
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        fm = amp(mid) - target_amp
        if fm * flo <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return float(np.exp(0.5 * (lo + hi)))


def _synth_job(job: tuple) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    rec_key, stim, n_kcs, breadth, q, how, target_amp, genotype, seed = job
    gain = _tune_gain(stim, breadth, q=q, how=how, target_amp=target_amp, seed=seed)
    X = synthetic_recording(n_kcs, stim, q=q, how=how, seed=seed + 1, breadth=breadth, gain=gain)
    summ, kc = edge_stats(X, stim)
    base = dict(panel=rec_key[0], recording=rec_key[1], genotype=genotype, normalization=how, q_true=q, gain=gain)
    bt = breadth_table(kc).assign(**base)
    mt = ms_band_table(kc).assign(**base)
    return dict(base, **summ), bt, mt


def estimate_q(per_rec: pd.DataFrame) -> pd.DataFrame:
    """q with an interval from each estimator (noise budget, edge, single-odor responders, and the two
    bounds), over the recordings of each panel and genotype group, and over the APL16 panel's 19
    recordings: median, the 5th and 95th percentiles over recordings, and the range."""
    rows = []
    groups = [((p, g), d) for (p, g), d in per_rec.groupby(['panel', 'genotype'], sort=False)]
    groups += [((p, 'all'), d) for p, d in per_rec.groupby('panel', sort=False)]
    for (panel, geno), d in groups:
        for est in ('q_noise', 'q_edge', 'q_single', 'q_edge_bound', 'q_single_bound'):
            v = d[est].dropna()
            if not len(v):
                continue
            rows.append(dict(panel=panel, genotype=geno, estimator=est, n_recordings=len(v), median=float(v.median()),
                p5=float(v.quantile(0.05)), p95=float(v.quantile(0.95)), min=float(v.min()), max=float(v.max())))
    return pd.DataFrame(rows)


def part_a(args) -> None:
    from concurrent.futures import ProcessPoolExecutor
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    per_rec, by_breadth, by_ms, kc_all = [], [], [], []
    synth_jobs = []
    for panel in PANELS:
        recs = load_panel_trials(panel)
        print(f'{panel}: {len(recs)} recordings', flush=True)
        for key, r in recs.items():
            summ, kc = edge_stats(r['X'], r['stim'])
            base = dict(panel=panel, recording=key[1], genotype=r['genotype'])
            per_rec.append(dict(base, **summ))
            by_breadth.append(breadth_table(kc).assign(**base))
            by_ms.append(ms_band_table(kc).assign(**base))
            kc_all.append(kc.assign(**base))
            if panel == 'APL16':
                amp_target = summ['resp_amp_mean']
                for q in SYNTH_Q:
                    for how in NORMALIZATIONS:
                        synth_jobs.append((key, r['stim'], r['n_kcs'], kc['breadth'].to_numpy(), q, how, amp_target,
                            r['genotype'], 1000 + len(synth_jobs)))
    per_rec = pd.DataFrame(per_rec)
    per_rec.to_csv(out / 'data_edge_per_recording.csv', index=False)
    pd.concat(by_breadth).to_csv(out / 'data_edge_by_breadth.csv', index=False)
    pd.concat(by_ms).to_csv(out / 'data_noise_by_mean_square.csv', index=False)
    kcs = pd.concat(kc_all, ignore_index=True)
    kcs.to_parquet(out / 'data_per_kc.parquet')
    est = estimate_q(per_rec)
    print(per_rec[['panel', 'recording', 'genotype', 'ms_max', 'ms_p999', 'edge_gap', 'max_trial', 'max_trial_mean',
        'max_trial_mean_single', 'r0', 'q_noise', 'q_edge', 'q_single', 'q_edge_bound', 'q_single_bound']].round(3).to_string(), flush=True)
    print(est.round(3).to_string(), flush=True)
    # synthetic cells: every APL16 recording's trial structure and breadths, three q, four normalizations
    print(f'synthetic: {len(synth_jobs)} recordings', flush=True)
    s_rows, s_bt, s_mt = [], [], []
    with ProcessPoolExecutor(max_workers=min(args.jobs, 16)) as ex:
        for r, bt, mt in ex.map(_synth_job, synth_jobs, chunksize=1):
            s_rows.append(r); s_bt.append(bt); s_mt.append(mt)
    synth = pd.DataFrame(s_rows)
    synth.to_csv(out / 'synthetic_per_recording.csv', index=False)
    pd.concat(s_bt).to_csv(out / 'synthetic_by_breadth.csv', index=False)
    pd.concat(s_mt).to_csv(out / 'synthetic_noise_by_mean_square.csv', index=False)
    s_est = []
    for (how, q), d in synth.groupby(['normalization', 'q_true']):
        e = estimate_q(d)
        s_est.append(e[e.genotype == 'all'].assign(normalization=how, q_true=q))
    s_est = pd.concat(s_est)
    s_est.to_csv(out / 'synthetic_q_estimates.csv', index=False)
    est.to_csv(out / 'q_estimates.csv', index=False)
    print(synth.groupby(['normalization', 'q_true'])[['ms_max', 'ms_p999', 'edge_gap', 'max_trial', 'max_trial_mean',
        'max_trial_mean_single', 'resp_amp_mean', 'rate', 'q_noise', 'q_edge', 'q_single']].median().round(3).to_string(), flush=True)
    a_figures(out, per_rec, pd.concat(by_breadth), pd.concat(by_ms), synth, pd.concat(s_mt), kcs)
    write_provenance(out, 'A', dict(n_recordings=per_rec.groupby('panel').size().to_dict(), synthetic_q=SYNTH_Q,
        normalizations=NORMALIZATIONS, weak_z=WEAK_Z, top_edge_share=TOP_EDGE_SHARE, top_single=TOP_SINGLE))
    print('->', out, flush=True)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def a_figures(out: Path, per_rec, bt, mt, synth, s_mt, kcs) -> None:
    """E73_edge.png (the per-KC mean square's upper tail per APL16 recording, with synthetic cells under the
    four normalizations; the measurement-noise budget; the q estimates per recording) and E73_breadth.png
    (amplitude, solvent and non-responding entries against the KC's breadth, data and synthetic)."""
    plt = _plt()
    col = {'TNT3c': '#2a78d6', 'TNT': '#c8553d', 'control': '#2a78d6', 'apl_output_blocked': '#c8553d'}
    q_main = float(np.median(per_rec.loc[per_rec.panel == 'APL16', 'q_edge']))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    for (rec, g), d in kcs[kcs.panel == 'APL16'].groupby(['recording', 'genotype']):
        v = np.sort(d.ms.to_numpy())[::-1]
        ax.plot(np.arange(1, len(v) + 1) / len(v), v, color=col[g], lw=0.8, alpha=0.7)
    ax.axhline(1 / q_main, color='k', ls='--', lw=1)
    ax.text(1e-3, 1 / q_main, f' 1 / q = {1 / q_main:.1f} (q {q_main:.3f}, from the edge)', fontsize=7.5, va='bottom')
    q_win = q_values()[0]
    ax.axhline(1 / q_win, color='0.5', ls=':', lw=1)
    ax.text(1e-3, 1 / q_win, f' 1 / q = {1 / q_win:.1f} (q {q_win:.3f}, the response windows\' share of the recording)', fontsize=7.5, va='bottom')
    ax.set_ylim(0, 1 / q_win + 0.6)
    ax.set_xscale('log'); ax.set_xlim(3e-4, 1)
    ax.set_xlabel('share of the recording\'s KCs at or above (rank / KCs)'); ax.set_ylabel('per-KC mean of squared trial values (z²)')
    ax.set_title('KCs: the mean square\'s upper tail, one line per APL16 recording\n(blue control flies, red APL output blocked (TNT))', fontsize=9)
    ax = axes[0, 1]
    styles = {'session': ('k', '-'), 'session_centered': ('0.45', '--'), 'trial': ('#7b3294', ':'), 'baseline': ('#e08214', '-.')}
    s17 = synth[synth.q_true == 0.17]
    for how in NORMALIZATIONS:
        d = s17[s17.normalization == how]
        ax.scatter(d.ms_max, d.max_trial, s=14, marker='o', facecolor='none', edgecolor=styles[how][0], label=f'synthetic, {NORM_LABEL[how]}')
    d = per_rec[per_rec.panel == 'APL16']
    for g, dd in d.groupby('genotype'):
        ax.scatter(dd.ms_max, dd.max_trial, s=22, color=col[g], label=f'KCs, {GENO_LABEL[g]}')
    ax.axvline(1 / 0.17, color='k', ls='--', lw=0.8)
    ax.set_xscale('log'); ax.set_xlabel('largest per-KC mean square in the recording (z²)')
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 4, 6, 10, 20, 40])); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.set_ylabel('largest single-trial value (z)')
    ax.set_title('Per recording: the largest mean square against the largest single trial\n(synthetic cells at q = 0.17, dashed 1 / q)', fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc='upper left')
    ax = axes[1, 0]
    for g, d in mt[mt.panel == 'APL16'].groupby('genotype'):
        m = d.groupby('ms_bin', sort=False)[['ms_mean', 'noise_sd']].mean()
        ax.plot(m.ms_mean, m.noise_sd, 'o-', color=col[g], label=f'KCs, {GENO_LABEL[g]}')
    for how in ('session', 'baseline'):
        d = s_mt[(s_mt.normalization == how) & (s_mt.q_true == 0.17)]
        m = d.groupby('ms_bin', sort=False)[['ms_mean', 'noise_sd']].mean()
        ax.plot(m.ms_mean, m.noise_sd, 's:', color=styles[how][0], ms=4, label=f'synthetic, {NORM_LABEL[how]}, q 0.17')
    r0 = float(per_rec.loc[per_rec.panel == 'APL16', 'r0'].mean())
    x = np.linspace(0.1, 5, 100)
    for qq, ls in ((0.12, ':'), (q_main, '-'), (0.25, '--')):
        tt = r0 * np.ones_like(x)
        for _ in range(20):
            tt = r0 * np.sqrt(np.clip(1 - qq * (x - tt ** 2), 0, None))
        ax.plot(x, tt, color='0.3', ls=ls, lw=0.9, label=f'z over the recording, q {qq:.2f}: r0 sqrt(1 - q (ms - t²))')
    ax.set_xlabel('per-KC mean of squared trial values (z²)'); ax.set_ylabel('KC\'s own measurement noise, sd of a trial (z)')
    ax.set_title('The noise budget: a KC\'s trial noise falls as its responses fill its session variance\n(entries with |trial mean| < 0.3 z and the solvent)', fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 1]
    est = ('q_edge', 'q_single', 'q_noise')
    lab = {'q_edge': 'from the edge (top 1 % by mean square)', 'q_single': 'from the single-odor responders', 'q_noise': 'from the noise budget (all KCs)'}
    mk = {'q_edge': 'o', 'q_single': '^', 'q_noise': 's'}
    order = per_rec.sort_values(['panel', 'genotype', 'recording']).reset_index(drop=True)
    for e in est:
        ax.scatter(order.index, order[e], marker=mk[e], s=20, c=[col[g] for g in order.genotype], label=lab[e], edgecolor='k', linewidth=0.3)
    ax.axhline(q_main, color='k', lw=0.8)
    for pnl in PANELS:
        idx = order.index[order.panel == pnl]
        ax.text(idx.min(), -0.12, pnl, fontsize=8)
        ax.axvline(idx.max() + 0.5, color='0.8', lw=0.6)
    ax.set_ylim(-0.15, 0.55); ax.set_xticks([])
    ax.set_xlabel('recordings (blue control flies, red APL output blocked (TNT))'); ax.set_ylabel('q, share of the recording in the response windows')
    ax.set_title('q per recording, three estimators', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    for a in axes.ravel():
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(out / 'E73_edge.png', dpi=120); plt.close(fig)
    # breadth
    sb = pd.read_csv(out / 'synthetic_by_breadth.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    order_b = [_breadth_label(lo, hi) for lo, hi in BREADTH_BINS]
    xb = np.arange(len(order_b))
    for ax, colname, ylabel in zip(axes, ('amp', 'blank_mean', 'nonresp_mean'), ('responders\' mean trial mean (z)',
            'the solvent\'s trial mean (z)', 'mean of the non-responding entries (z)')):
        for g, d in bt[bt.panel == 'APL16'].groupby('genotype'):
            m = d.groupby('breadth')[colname].mean().reindex(order_b)
            ax.plot(xb, m.to_numpy(), 'o-', color=col[g], label=f'KCs, {GENO_LABEL[g]}')
        for how in ('session', 'session_centered', 'baseline'):
            d = sb[(sb.normalization == how) & (sb.q_true == 0.17)]
            m = d.groupby('breadth')[colname].mean().reindex(order_b)
            ax.plot(xb, m.to_numpy(), 's:', color=styles[how][0], ms=4, label=f'synthetic, {NORM_LABEL[how]}')
        ax.axhline(0, color='0.8', lw=0.6)
        ax.set_xticks(xb); ax.set_xticklabels(order_b); ax.set_xlabel('odors the KC responds to (trial mean > 1 z)')
        ax.set_ylabel(ylabel); ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle('APL16 KCs by breadth, against synthetic cells (q 0.17; the synthetic amplitudes tuned to each recording\'s responder amplitude)', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E73_breadth.png', dpi=120); plt.close(fig)


def _git(*args, cwd=None) -> str:
    import subprocess
    return subprocess.run(['git', *args], cwd=cwd or H.HERE, capture_output=True, text=True).stdout.strip()


def write_provenance(out: Path, part: str, extra: dict, models: bool = False, orn_deltas=None) -> None:
    """provenance_<part>.json: script, commit and dirty state, host, inputs, settings; with `models`, also
    `H.model_provenance()` (olfsysm build, al_analysis / hong2p commits, package versions)."""
    import json
    import socket
    from datetime import datetime, timezone
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script=Path(__file__).name, part=part,
        host=socket.gethostname(), al_analysis_commit=_git('rev-parse', 'HEAD'),
        al_analysis_dirty=bool(_git('status', '--porcelain', '--', '.')), **extra)
    if models:
        prov['model_provenance'] = H.model_provenance(orn_deltas=orn_deltas)
    (out / f'provenance_{part}.json').write_text(json.dumps(prov, indent=1, default=str, ensure_ascii=False) + '\n')


# ---------------------------------------------------------------------------
# part B: the models read the same way
MAPS = ('linear', 'saturating')
READINGS = H.ZCELL_READINGS
READING_LABEL = {'graded': 'graded (the target as defined)', 'thresholded': 'sub-threshold values set to 0',
    'binary': 'binarized'}
MAP_LABEL = {'linear': 'per-KC session z of the counts', 'saturating': 'per-KC session z of the saturated counts'}
# key models: (short, label, folder, tag, present reading the summary quotes it under)
KEY_MODELS = [
    ('C2', 'C2: the first joint fit\'s family, breadth exponent 2, no gain, no per-claw threshold, 6.8 % (E67, APL-effect term under the calibrated readout)',
     'E67_apl_effect_fits', 'rate0.068_aplcal_k2_cap3', 'cal'),
    ('C2 both', 'C2 refit on both readings (E67 rate0.068_aplboth_k2_cap3)', 'E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3', 'cal'),
    ('fit 1', 'E69 fit 1: C2\'s family with the firing ceiling at strength 1', 'E69_kc_firing_ceiling',
     'rate0.068_aplboth_refr3_adapt0.04142t1000_k2', 'raw'),
    ('fit 2', 'E69 fit 2: the same at strength 2', 'E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.1114t1000_k2', 'raw'),
    ('fit 3', 'E69 fit 3: the reference set with the firing ceiling at strength 2', 'E69_kc_firing_ceiling',
     'thr40_gain1.5_rate0.068_aplboth_refr3_adapt0.1114t1000_k2', 'raw'),
    ('ref.', 'the reference model (eight glomeruli at 1.5, 40 Hz per-claw threshold, 6.8 %; E44)', 'E44_operating_point',
     'thr40_gain1.5_rate0.068_k2_cap3', 'cal'),
]
# the key models' jittered trials at the noise level matched on the control flies' single-trial distance
TRIAL_SETS = {
    ('E67_apl_effect_fits', 'rate0.068_aplcal_k2_cap3'): 'E67_apl_effect_fits/partB/trials/E67-rate0.068_aplcal_k2_cap3_sigma0.126_n3_rep*.npz',
    ('E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.04142t1000_k2'):
        'E69_kc_firing_ceiling/trial_noise/trials/E69-rate0.068_aplboth_refr3_adapt0.04142t1000_k2_sigma0.162_n3_rep*.npz',
    ('E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.1114t1000_k2'):
        'E69_kc_firing_ceiling/trial_noise/trials/E69-rate0.068_aplboth_refr3_adapt0.1114t1000_k2_sigma0.18_n3_rep*.npz',
    ('E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'):
        'E48_block_target_noise/partB/trials/E44-thr40_gain1.5_rate0.068_k2_cap3_sigma0.207_n3_rep*.npz',
}
CRIT_KEYS = ('rate', 'resp_amp_mean', 'resp_amp_cv', 'share_above_4', 'share_05_15', 'silent_share', 'opk_8plus',
    'nonresp_trial_sd', 'mean_sq_max', 'mean_sq_p999', 'edge_gap', 'max_trial_mean')
_REFS: Dict[str, object] = {}


def refs() -> Dict[str, object]:
    """The data side, cached per process: per reading the per-pair APL effect and the two genotypes' mean
    matrices; the 66 reference pairs; the per-odor response fractions; the calibration targets; q."""
    if not _REFS:
        import warnings
        warnings.simplefilter('ignore', RuntimeWarning)
        for v in READINGS:
            if v == 'graded':
                eff, on, off = H.yang_apl_effect(), H.yang_mean_rdm('TNT3c'), H.yang_mean_rdm('TNT')
            elif v == 'binary':
                eff, on, off = H.yang_apl_effect(binarize=1.0), H.yang_mean_rdm('TNT3c', binarize=1.0), H.yang_mean_rdm('TNT', binarize=1.0)
            else:
                eff = H.yang_apl_effect(values=v)
                on, off = H.yang_recording_rdms_values('TNT3c', v).mean(), H.yang_recording_rdms_values('TNT', v).mean()
            _REFS[v] = dict(effect=eff, on=H.canonical_pairs(on), off=H.canonical_pairs(off))
        _REFS['ref66'] = H.reference_pairs()
        _REFS['frac'] = {c: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for c, g in (('on', 'TNT3c'), ('off', 'TNT'))}
        dt = H.data_target
        _REFS['targets'] = dict(rate=dt('ctrl_frac'), amp=dt('ctrl_resp_amp_mean_z'), silent_sd=dt('ctrl_trial_noise_sd'),
            pr=dt('ctrl_participation_ratio_resp'))
        _REFS['noise_sd'] = H.kc_trial_noise_sd('TNT3c')
        _REFS['drift'] = drift_ratio()
    return _REFS


# Yang's APL16 recordings (add_to_reports/yang_response_definition.md, notes/yang_response_normalization/): 51 response
# windows of 18 volumes (2-8 s after onset) and 15 baseline volumes before each onset, in about 7,194 volumes
Q_WINDOW_DEFAULT = 51 * 18 / 7194
SCALE_RATIO_SILENT_DEFAULT = 1.403
YANG_NORM = H.shared_dir() / 'notes' / 'yang_response_normalization'


def q_values() -> Tuple[float, List[float]]:
    """(q, sensitivity q's). q is the share of the recording inside the 51 response windows of Yang's APL16
    recordings (registry `zcell_q_window`), the model's response taken as confined to its window; the
    sensitivity check runs at part A's estimate of a response's variance footprint (`zcell_q`, from the edge,
    0.18: a GCaMP response spreads beyond the 2-8 s window) and the ends of its interval over recordings."""
    try:
        qw = H.data_target('zcell_q_window')
    except KeyError:
        qw = Q_WINDOW_DEFAULT
    try:
        qe = H.data_target('zcell_q')
        lo, hi = H.data_target_ci('zcell_q')
    except KeyError:
        e = pd.read_csv(H.results_dir(EXPERIMENT) / 'q_estimates.csv')
        r = e[(e.panel == 'APL16') & (e.genotype == 'all') & (e.estimator == 'q_edge')].iloc[0]
        qe, lo, hi = r['median'], r['p5'], r['p95']
    return float(qw), [float(lo), float(qe), float(hi)]


def drift_ratio() -> float:
    """The slow drift's sd relative to the baseline noise, from the KCs that respond to no odor: their
    whole-trace SD over baseline-noise SD (registry `kc_scale_ratio_0`, 1.40) is sqrt(1 + drift^2)."""
    try:
        r = H.data_target('kc_scale_ratio_0')
    except KeyError:
        r = SCALE_RATIO_SILENT_DEFAULT
    return float(np.sqrt(max(r ** 2 - 1, 0.0)))


def vcf_alignment(d: pd.Series) -> float:
    """The natural-source alignment of a pair Series (as `compare_to_references(d)['spearman_resid_vs_vcf_hyp']`:
    Spearman of the residual around the least-squares line on the ORN distances with the hyperbolic
    natural-source distances, over the 66 megamat pairs)."""
    from scipy import stats
    ref = refs()['ref66']
    y = H.canonical_pairs(d).reindex(ref.index).to_numpy(dtype=float)
    x, v = ref['orn'].to_numpy(dtype=float), ref['vcf_hyp'].to_numpy(dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    b, a = np.polyfit(x[ok], y[ok], 1)
    return float(stats.spearmanr(y[ok] - (a + b * x[ok]), v[ok])[0])


def score_distances(d_on: pd.Series, d_off: pd.Series, reading: str) -> dict:
    """Everything scored for one model reading against the same reading of the data: the APL-effect loss with
    its mean, per-odor and pair-specific parts and ρ on the per-pair APL effect, the mean distances, the two
    ρ's on odor-odor distances, the block and the natural-source alignment (APL on / off / on minus off)."""
    r = refs()[reading]
    st, _ = H.apl_effect_decomposition(d_on, d_off, r['effect'])
    out = {k.replace('apl_effect_', ''): v for k, v in st.items()}
    out['data_mean'] = float(r['effect'].mean())
    out['mean_dist_on'], out['mean_dist_off'] = float(d_on.mean()), float(d_off.mean())
    out['rho_on'] = H.compare_model_to_data(d_on, r['on'])['spearman']
    out['rho_off'] = H.compare_model_to_data(d_off, r['off'])['spearman']
    oo, _ = H.compare_on_off(d_on, d_off)
    out['block'] = oo['block_fruity_small_iso']
    out['vcf_on'], out['vcf_off'] = vcf_alignment(d_on), vcf_alignment(d_off)
    out['vcf_diff'] = out['vcf_on'] - out['vcf_off']
    return out


BREADTH_CHECK = (0, 1, 2, 3, 4, 6, 8, 11)


def criterion_stats(s_on: dict, s_off: dict, breadth_checks: bool = True) -> dict:
    """The response-criterion statistics of both conditions (`zcell_stats` dicts), the response rate ρ against
    the KC data's per-odor fractions, the odors-per-KC distances from the flies' and the realism flag."""
    from apl_effect_scores import OPK_BINS, opk_band, opk_data, realism_flag
    out = {}
    for c, s in (('on', s_on), ('off', s_off)):
        out.update({f'{k}_{c}': s[k] for k in CRIT_KEYS if k in s})
        out[f'rate_rho_{c}'] = H.spearman(s['rate_per_odor'].rename(H.odor_abbrev), refs()['frac'][c])
        out.update({f'opk_{c}_{b}': float(v) for b, v in zip(OPK_BINS, s['opk'])})
        out[f'opk_{c}_tv'] = float(0.5 * np.abs(np.asarray(s['opk']) - opk_data(c)).sum())
        if breadth_checks and 'scale_ratio_by_breadth' in s:
            out.update({f'scale_ratio_b{b}_{c}': s['scale_ratio_by_breadth'].get(b, np.nan) for b in BREADTH_CHECK})
            out.update({f'amp_base_b{b}_{c}': s['amp_base_by_breadth'].get(b, np.nan) for b in BREADTH_CHECK if b})
    out['rate_ratio'] = out['rate_off'] / out['rate_on'] if out['rate_on'] > 0 else np.nan
    out['realism_flag'] = realism_flag(out['rate_on'], out['rate_off'], out['rate_rho_on'], out['opk_on_tv'], out['opk_off_tv'])
    return out


def _z_fixed(counts: pd.DataFrame, a_map: str, K: Optional[float], c: float, noise_sd: float) -> Dict[str, object]:
    """The present reading as a `session_zscore` with a fixed divisor 1 / c for every KC and measurement noise
    of sd `noise_sd` per trial (E68's conversion of readout units to z)."""
    return H.session_zscore(counts, q=1.0, noise=noise_sd / c, amp=a_map, K=K, noise_ratio=1.0, divisor=1.0 / c)


def present_rows(base: dict, on: pd.DataFrame, off: pd.DataFrame) -> List[dict]:
    """The present readings: raw counts and E65's calibrated saturating readout (graded, against the graded
    data) and the counts binarized at two spikes (against the binarized data); response statistics under the
    two-spike criterion, amplitudes through E68's conversion to z (c set so that the APL-on response rate under
    the data's criterion, through the data's measurement noise, is the control flies')."""
    from apl_effect_scores import odors_per_kc
    t = refs()['targets']
    s = refs()['noise_sd']
    K = H.calibrate_readout(on, 'saturating', t['pr'], stat='participation_ratio', responders_min=2.0)
    crit = {}
    for c, m in (('on', on), ('off', off)):
        resp = (m >= 2).astype(float)
        opk = odors_per_kc(resp)
        crit[c] = dict(rate=float(resp.mean().mean()), rate_per_odor=resp.mean(), opk=opk, silent_share=float(opk[0]), opk_8plus=float(opk[-1]))
    rows = []
    for name, kind, param, a_map, reading in (('present_raw', 'raw', None, 'linear', 'graded'),
            ('present_cal', 'saturating', K, 'saturating', 'graded'), ('present_binary', 'binary', 2.0, 'linear', 'binary')):
        d_on, d_off = H.readout_pair_distances(on, kind, param), H.readout_pair_distances(off, kind, param)
        row = dict(base, reading_def=name, reading=reading, q=np.nan, K=K if a_map == 'saturating' else np.nan)
        row.update(score_distances(d_on, d_off, reading))
        sig = H.readout(on, 'saturating', K) if a_map == 'saturating' else on.astype(float)
        cz = H.z_scale(sig, t['rate'], s, n_trials=3)
        amp = {}
        for cnd, m in (('on', on), ('off', off)):
            st = H.zcell_stats(_z_fixed(m, a_map, K if a_map == 'saturating' else None, cz, s))
            amp[cnd] = st
        st_on = dict(amp['on'], **crit['on'])
        st_off = dict(amp['off'], **crit['off'])
        row.update(criterion_stats(st_on, st_off, breadth_checks=False))
        row['c_z'] = cz
        rows.append(row)
    return rows


def zcell_rows(base: dict, on, off, *, q: float, a_map: str, center: bool = False, readings=READINGS, tag: str = '',
    cal: Optional[dict] = None) -> Tuple[List[dict], dict]:
    """The session z-score readings of one variant (one map, one q): calibration on APL on, both conditions
    read, every reading scored. `on` / `off` (units x odors) DataFrames or (recordings, trials, units, odors)
    arrays of jittered trials."""
    t = refs()['targets']
    drift = refs()['drift']
    if cal is None:
        cal = H.calibrate_session_zscore(on, q=q, amp=a_map, target_rate=t['rate'],
            target_amp=t['amp'] if a_map == 'saturating' else None, target_silent_sd=t['silent_sd'], center=center, drift=drift)
    kw = dict(q=q, noise=cal['noise'], amp=a_map, K=cal['K'], noise_ratio=cal['noise_ratio'], center=center, drift=drift)
    z_on, z_off = H.session_zscore(on, **kw), H.session_zscore(off, **kw)
    crit = criterion_stats(H.zcell_stats(z_on), H.zcell_stats(z_off))
    name = f'zcell_{a_map}' + tag
    rows = []
    for rd in readings:
        row = dict(base, reading_def=name, reading=rd, q=q, K=cal['K'] if a_map == 'saturating' else np.nan, noise=cal['noise'],
            noise_ratio=cal['noise_ratio'], cal_noise_at_bound=cal['noise_at_bound'], cal_K_at_bound=cal['K_at_bound'],
            cal_converged=cal['converged'], center=center, drift=drift)
        row.update(score_distances(H.zcell_pair_distances(z_on, rd), H.zcell_pair_distances(z_off, rd), rd))
        row.update(crit)
        rows.append(row)
    return rows, cal


def load_counts(folder: str, tag: str, cond: str) -> pd.DataFrame:
    df = pd.read_parquet(ARCHIVE / folder / f'{tag}_{cond}_spike_counts.parquet')
    return H.abbrev_columns(df.astype(float))[H.APL16_ODORS]


def load_trials(pattern: str) -> Tuple[np.ndarray, np.ndarray]:
    """(APL-on, APL-off) jittered spike counts (recordings, trials, KCs, odors in APL16 order) of a trial set."""
    files = sorted(ARCHIVE.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    arrs = [np.load(f) for f in files]
    return tuple(np.stack([a[f'{c}_spk'].astype(float) for a in arrs]) for c in ('on', 'acute_off'))


def q_tag(q: float) -> str:
    return f'_q{q:.3f}'


def score_variant(job: tuple) -> List[dict]:
    """One archived variant: the present readings and the session z-score readings (both maps, three readings
    each, at q; the saturating map also at the sensitivity q's (rows `zcell_saturating_q<q>`) and with the
    session mean subtracted), each scored against the same reading of the data; and, for the key models with
    saved jittered trials, the reading of their trials (`_trials`, both maps at q, the saturating map also at
    part A's footprint). Returns rows (an exception is returned as one row with `error`)."""
    folder, tag, q, q_sens = job
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    base = dict(folder=folder, tag=tag)
    try:
        on, off = load_counts(folder, tag, 'on'), load_counts(folder, tag, 'acute_off')
        rows = present_rows(base, on, off)
        for a_map in MAPS:
            rows += zcell_rows(base, on, off, q=q, a_map=a_map)[0]
        for qq in q_sens:
            rows += zcell_rows(base, on, off, q=qq, a_map='saturating', tag=q_tag(qq))[0]
        rows += zcell_rows(base, on, off, q=q, a_map='saturating', center=True, tag='_centered')[0]
        if (folder, tag) in TRIAL_SETS:
            t_on, t_off = load_trials(TRIAL_SETS[(folder, tag)])
            tb = dict(base, n_trial_recordings=t_on.shape[0])
            for a_map in MAPS:
                rows += zcell_rows(tb, t_on, t_off, q=q, a_map=a_map, tag='_trials')[0]
            rows += zcell_rows(tb, t_on, t_off, q=q_sens[1], a_map='saturating', tag='_trials' + q_tag(q_sens[1]))[0]
        return rows
    except Exception as e:
        import traceback
        return [dict(base, error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())]


def controls(q: float) -> pd.DataFrame:
    """The controls on the key models: the reading with a fixed divisor for every KC and no noise reproduces
    the raw-count and calibrated-readout distances and losses; with the data's measurement noise it equals
    `expected_pair_distances_noisy`; the reading's per-KC mean square stays under the edge 1 / q (with the
    pre-odor baseline subtracted, a KC responding to a share p of the trials can reach 1 / (q (1 - q p)))."""
    t = refs()['targets']
    s = refs()['noise_sd']
    rows = []
    for short, _, folder, tag, _ in KEY_MODELS:
        on, off = load_counts(folder, tag, 'on'), load_counts(folder, tag, 'acute_off')
        K = H.calibrate_readout(on, 'saturating', t['pr'], stat='participation_ratio', responders_min=2.0)
        for a_map, kind, param in (('linear', 'raw', None), ('saturating', 'saturating', K)):
            present = {c: H.readout_pair_distances(m, kind, param) for c, m in (('on', on), ('off', off))}
            fixed = {c: H.zcell_pair_distances(H.session_zscore(m, q=q, noise=0.0, amp=a_map, K=param, divisor=3.7))
                for c, m in (('on', on), ('off', off))}
            diff = max(float((present[c] - fixed[c].reindex(present[c].index)).abs().max()) for c in present)
            l_p = H.apl_effect_stats(present['on'], present['off'])['apl_effect_rmse']
            l_f = H.apl_effect_stats(fixed['on'], fixed['off'])['apl_effect_rmse']
            sig = H.readout(on, kind, param)
            cz = H.z_scale(sig, t['rate'], s, n_trials=3)
            noisy = H.expected_pair_distances_noisy(cz * sig, s, 3)
            fixed_n = H.zcell_pair_distances(_z_fixed(on, a_map, param, cz, s))
            diff_n = float((noisy - fixed_n.reindex(noisy.index)).abs().max())
            rows.append(dict(model=short, folder=folder, tag=tag, map=a_map, check='fixed divisor = present reading',
                max_abs_distance_diff=diff, loss_present=l_p, loss_fixed_divisor=l_f, max_abs_distance_diff_noisy=diff_n))
        for a_map in MAPS:
            cal = H.calibrate_session_zscore(on, q=q, amp=a_map, target_rate=t['rate'], target_amp=t['amp'] if a_map == 'saturating' else None,
                target_silent_sd=t['silent_sd'], drift=refs()['drift'])
            for c, m in (('on', on), ('off', off)):
                z = H.session_zscore(m, q=q, noise=cal['noise'], amp=a_map, K=cal['K'], noise_ratio=cal['noise_ratio'], drift=refs()['drift'])
                st = H.zcell_stats(z)
                rows.append(dict(model=short, folder=folder, tag=tag, map=a_map, check=f'edge of the reading, APL {c}',
                    mean_sq_max=st['mean_sq_max'], mean_sq_p999=st['mean_sq_p999'], edge_gap=st['edge_gap'], inv_q=1 / q,
                    rate=st['rate'], resp_amp_mean=st['resp_amp_mean'], nonresp_trial_sd=st['nonresp_trial_sd'],
                    noise=cal['noise'], K=cal['K'], noise_ratio=cal['noise_ratio']))
            if a_map == 'saturating' and short in ('C2', 'ref.'):
                rows += trace_check(short, folder, tag, on, off, cal)
    return pd.DataFrame(rows)


def trace_reading(counts: pd.DataFrame, *, noise: float, drift: float, amp: str, K: Optional[float], seed: int = 0,
    n_trials: int = 3, n_volumes: int = 7194, base: int = 15, gap: int = 6, window: int = 18, lead: int = 5,
    drift_tau: float = 300.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """Per-trial responses of a model's KCs read from a simulated whole-recording trace with the structure of Yang's
    APL16 recordings: the 16 odors and a blank, `n_trials` blocks in random order, each trial n_volumes // 51
    volumes; the odor onset `lead` volumes after the `base` baseline volumes; the response (the count through
    the amplitude map) a boxcar over volumes gap .. gap + window - 1 after onset; white noise of sd `noise` per
    volume and an AR(1) drift of sd drift x noise (time constant `drift_tau` volumes); each KC's trace
    z-scored over all its volumes (ddof 0); a trial's response the mean z over the window minus the mean over
    the `base` volumes before onset. Returns (trials x KCs responses, per-trial stimulus labels, volumes)."""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    names = list(counts.columns) + ['pfo @ 0']
    a = H._zcell_amplitude(counts.to_numpy(dtype=float), amp, K).astype(np.float32)
    U, n_st = a.shape[0], len(names)
    L = n_volumes // (n_st * n_trials)
    order = np.concatenate([rng.permutation(n_st) for _ in range(n_trials)])
    T = L * len(order)
    onset = np.arange(len(order)) * L + base + lead
    F = rng.normal(0, noise, (U, T)).astype(np.float32)
    if drift > 0:
        rho = np.exp(-1.0 / drift_tau)
        e = rng.normal(0, 1, (U, T + int(5 * drift_tau))).astype(np.float32)
        F += (lfilter([np.sqrt(1 - rho ** 2)], [1, -rho], e, axis=1)[:, int(5 * drift_tau):] * (drift * noise)).astype(np.float32)
    for j, st in enumerate(order):
        if st < a.shape[1]:
            F[:, onset[j] + gap:onset[j] + gap + window] += a[:, st][:, None]
    z = (F - F.mean(axis=1, keepdims=True)) / F.std(axis=1, keepdims=True)
    iw = onset[:, None] + gap + np.arange(window)
    ib = onset[:, None] - base + np.arange(base)
    r = z[:, iw].mean(axis=2) - z[:, ib].mean(axis=2)
    return r.T, np.array([names[st] for st in order]), T


def trace_check(short: str, folder: str, tag: str, on: pd.DataFrame, off: pd.DataFrame, cal: dict) -> List[dict]:
    """The closed form against the simulated trace: the model's calibrated noise and K, Yang's trial structure,
    the trace reading's own window share and noise ratio (from the same trace with no responses) in the closed
    form; the response rate, the responders' amplitude, the per-KC mean square's maximum, the silent KCs' whole-
    trace SD over baseline-noise SD, and the graded distances (largest difference over the 120 pairs) and the
    APL-effect loss of the two."""
    drift = refs()['drift']
    rows = []
    sims = {}
    for c, m in (('on', on), ('off', off)):
        X, stim, T = trace_reading(m, noise=cal['noise'], drift=drift, amp='saturating', K=cal['K'], seed=7)
        X0, _, _ = trace_reading(m * 0, noise=cal['noise'], drift=drift, amp='saturating', K=cal['K'], seed=7)
        nr = float(X0.std() * np.sqrt(1 + drift ** 2))
        qw = 51 * 18 / T
        names, tm, _, _ = _per_stimulus(X, stim)
        odor_cols = [names.index(o) for o in m.columns]
        tm = tm[:, odor_cols]
        z = H.session_zscore(m, q=qw, noise=cal['noise'], amp='saturating', K=cal['K'], noise_ratio=nr, drift=drift)
        st = H.zcell_stats(z)
        d_cf = H.zcell_pair_distances(z, 'graded')
        d_sim = H.canonical_pairs(H.pair_distances(pd.DataFrame(tm, columns=m.columns), 'cosine'))
        sims[c] = (d_sim, d_cf)
        resp = tm > 1
        rows.append(dict(model=short, folder=folder, tag=tag, map='saturating', check=f'trace simulation = closed form, APL {c}', q_window=qw,
            noise_ratio_trace=nr, rate_trace=float(resp.mean()), rate_closed=st['rate'], amp_trace=float(tm[resp].mean()),
            amp_closed=st['resp_amp_mean'], ms_max_trace=float((X ** 2).mean(axis=0).max()), ms_max_closed=st['mean_sq_max'],
            silent_scale_ratio_trace=float(np.sqrt(1 + drift ** 2)), max_abs_distance_diff=float((d_sim - d_cf.reindex(d_sim.index)).abs().max())))
    l_sim = H.apl_effect_stats(sims['on'][0], sims['off'][0])
    l_cf = H.apl_effect_stats(sims['on'][1], sims['off'][1])
    rows.append(dict(model=short, folder=folder, tag=tag, map='saturating', check='trace simulation = closed form, APL effect',
        loss_trace=l_sim['apl_effect_rmse'], loss_closed=l_cf['apl_effect_rmse'], mean_trace=l_sim['apl_effect_mean'], mean_closed=l_cf['apl_effect_mean']))
    return rows


# the low-concentration panels in E49's archive (26 odors: the 16 APL16 odors, then these, named as in olfdb)
LOW_CONC_NAMES = {'2h+ma @ -7': '2h @ -7 + ma @ -7', '2h-7+farn-3 @ 0': '2h @ -7 + farn @ -3', 'ma-7+farn-3 @ 0': 'ma @ -7 + farn @ -3'}
LOW_CONC_PANELS = {'diag_mix': (('2h @ -7', 'ma @ -7', 'ma @ -8', 'farn @ -3', '2h @ -7 + ma @ -7', '2h @ -7 + farn @ -3',
    'ma @ -7 + farn @ -3'), 6), 'natmix_dilute': (('2h @ -7', 'ma @ -7', '2h @ -7 + ma @ -7', '2h @ -5', 'ea @ -5', 'ea @ -7'), 3)}


def low_conc(q: float) -> pd.DataFrame:
    """The low-concentration response fractions of E49's archived models (each an earlier model at its fitted
    parameters on a 26-odor input), read within each low-concentration panel's own session: the reading
    calibrated on the model's APL-on counts of the 16 APL16 odors (both maps), then each panel's odors that
    the model has, plus a blank, z-scored over that session (diag_mix: 6 trials per odor; natmix_dilute: 3),
    and the fraction of KCs with a trial mean above 1 z, beside the two-spike criterion and the KC data
    (E49's data_fractions.csv). The model's session lacks the panel's odors it was not given (diag_mix: 2h
    @ -8, farn @ -4, farn @ -2.5 and their mixtures), so its session sd is an approximation."""
    t = refs()['targets']
    d_frac = pd.read_csv(RESULTS / 'E49_low_conc_kc' / 'data_fractions.csv')
    rows = []
    a = ARCHIVE / 'E49_low_conc_kc'
    for f in sorted(a.glob('*_on_spike_counts.parquet')):
        tag = f.name[:-len('_on_spike_counts.parquet')]
        if tag.endswith('_base') or not (a / f'{tag}_acute_off_spike_counts.parquet').exists():
            continue
        cnt = {c: pd.read_parquet(a / f'{tag}_{c}_spike_counts.parquet').astype(float).rename(columns=LOW_CONC_NAMES)
            for c in ('on', 'acute_off')}
        apl16 = H.abbrev_columns(cnt['on'].iloc[:, :16])[H.APL16_ODORS]
        for a_map in MAPS:
            cal = H.calibrate_session_zscore(apl16, q=q, amp=a_map, target_rate=t['rate'], target_amp=t['amp'] if a_map == 'saturating' else None,
                target_silent_sd=t['silent_sd'], drift=refs()['drift'])
            for panel, (odors, n_tr) in LOW_CONC_PANELS.items():
                cols = [o for o in odors if o in cnt['on'].columns]
                for c, m in cnt.items():
                    z = H.session_zscore(m[cols], q=q, noise=cal['noise'], amp=a_map, K=cal['K'], noise_ratio=cal['noise_ratio'], n_trials=n_tr,
                        drift=refs()['drift'])
                    sd = (z['trial_sd'] / np.sqrt(n_tr))[:, :, None]
                    P = H._tail_moments(z['center'], sd, 1.0)[0].mean(axis=(0, 1))
                    group = 'control' if c == 'on' else 'apl_output_blocked'
                    for o, pz in zip(cols, P):
                        dd = d_frac[(d_frac.panel == panel) & (d_frac.group == group) & (d_frac.odor == o)]
                        rows.append(dict(model=tag, map=a_map, panel=panel, condition=c, odor=o, frac_zcell=float(pz),
                            frac_two_spikes=float((m[o] >= 2).mean()), frac_data=float(dd.frac.iloc[0]) if len(dd) else np.nan,
                            noise=cal['noise'], K=cal['K'], noise_ratio=cal['noise_ratio'], q=q))
    return pd.DataFrame(rows)


def variant_list(only_key: bool, folders: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """Every (folder, tag) of results/apl_effect/by_variant.csv with both conditions archived (only those of
    `folders` when given)."""
    bv = pd.read_csv(RESULTS / 'apl_effect' / 'by_variant.csv')
    have = [(r.folder, str(r.tag)) for r in bv.itertuples()
        if (ARCHIVE / r.folder / f'{r.tag}_on_spike_counts.parquet').exists() and (ARCHIVE / r.folder / f'{r.tag}_acute_off_spike_counts.parquet').exists()]
    if only_key:
        keys = {(f, t) for _, _, f, t, _ in KEY_MODELS}
        have = [h for h in have if h in keys]
    if folders:
        have = [h for h in have if h[0] in folders]
    return list(dict.fromkeys(have))


def part_b(args) -> None:
    from concurrent.futures import ProcessPoolExecutor
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    q, q_sens = q_values()
    if args.q:
        q = args.q
    have = variant_list(args.only_key, args.folders)
    print(f'part B: {len(have)} variants, q {q:.4f} (sensitivity {", ".join(f"{x:.4f}" for x in q_sens)}), drift {refs()["drift"]:.3f}', flush=True)
    jobs = [(f, t, q, q_sens) for f, t in have]
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=min(args.jobs, 16)) as ex:
        for i, res in enumerate(ex.map(score_variant, jobs, chunksize=2)):
            for r in res:
                (errors if 'error' in r else rows).append(r)
            if (i + 1) % 50 == 0:
                print(f'{i + 1} / {len(jobs)} variants', flush=True)
    df = pd.DataFrame(rows)
    bv = pd.read_csv(RESULTS / 'apl_effect' / 'by_variant.csv').drop_duplicates(['folder', 'tag'])
    bv['tag'] = bv['tag'].astype(str)
    df = df.merge(bv[['folder', 'tag', 'exp', 'realism_flag']].rename(columns={'realism_flag': 'realism_flag_by_variant'}),
        on=['folder', 'tag'], how='left')
    name = 'by_variant_key.csv' if args.only_key else 'by_variant.csv'
    if args.folders and (out / name).exists():
        # scores of the named folders replace their rows in the existing table
        old = pd.read_csv(out / name)
        df = pd.concat([old[~old.folder.isin(args.folders)], df], ignore_index=True)
    df.to_csv(out / name, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out / 'errors.csv', index=False)
        print(f'{len(errors)} variants failed: see errors.csv', flush=True)
    ctl = controls(q)
    ctl.to_csv(out / 'controls.csv', index=False)
    lc = low_conc(q)
    lc.to_csv(out / 'low_conc.csv', index=False)
    print(ctl.round(4).to_string(), flush=True)
    write_provenance(out, 'B', dict(q=q, q_sensitivity=q_sens, drift=refs()['drift'], n_variants=len(have), n_rows=len(df), n_errors=len(errors),
        maps=MAPS, readings=READINGS, key_models=KEY_MODELS, trial_sets=list(TRIAL_SETS.values()),
        calibration_targets=refs()['targets'], data_noise_sd=refs()['noise_sd'],
        inputs={'by_variant': str(RESULTS / 'apl_effect' / 'by_variant.csv'), 'archive': str(ARCHIVE)}), models=True)
    print('->', out / name, flush=True)


# ---------------------------------------------------------------------------
# summary tables and figures
AMP_STATS = ('resp_amp_cv', 'share_above_4', 'max_trial_mean', 'mean_sq_max')


def map_choice(df: pd.DataFrame) -> pd.DataFrame:
    """Per map, how near the reading's APL-on amplitude statistics (the responders' coefficient of variation,
    the share above 4 z, the largest trial mean, the per-KC edge) are to the control flies' (mean relative
    absolute deviation), per variant; the map kept is the one with the lower median over the variants
    without a realism flag under the present definitions (the same variants for both maps). No distance
    between odors and no APL effect enters."""
    pa = pd.read_csv(H.results_dir(EXPERIMENT) / 'data_edge_per_recording.csv')
    ctrl = pa[(pa.panel == 'APL16') & (pa.genotype == 'TNT3c')]
    dt = H.data_target
    target = dict(resp_amp_cv=dt('ctrl_resp_amp_cv'), share_above_4=dt('ctrl_share_z_above_4'),
        max_trial_mean=float(ctrl.max_trial_mean.mean()), mean_sq_max=float(ctrl.ms_max.mean()))
    rows = []
    for a_map in MAPS:
        g = df[(df.reading_def == f'zcell_{a_map}') & (df.reading == 'graded')]
        dev = np.mean([np.abs(g[f'{k}_on'] - v) / abs(v) for k, v in target.items()], axis=0)
        flags = g['realism_flag_by_variant'].fillna('') if 'realism_flag_by_variant' in g else g['realism_flag']
        for f, t, flag, d in zip(g.folder, g.tag, flags, dev):
            rows.append(dict(folder=f, tag=t, map=a_map, amp_deviation=float(d), realistic=flag == ''))
    out = pd.DataFrame(rows)
    out.attrs['target'] = target
    return out


def summarize(args) -> None:
    out = H.results_dir(EXPERIMENT)
    df = pd.read_csv(out / ('by_variant_key.csv' if args.only_key else 'by_variant.csv'))
    df['tag'] = df['tag'].astype(str)
    df['realism_flag'] = df['realism_flag'].fillna('')
    q, q_sens = q_values()
    mc = map_choice(df)
    med = mc[mc.realistic].groupby('map').amp_deviation.median() if mc.realistic.any() else mc.groupby('map').amp_deviation.median()
    chosen = str(med.idxmin())
    mc.to_csv(out / 'map_choice.csv', index=False)
    print('map choice (median relative deviation of the APL-on amplitude statistics from the control flies\'):', med.round(3).to_dict(),
        '->', chosen, flush=True)
    if not (out / 'low_conc.csv').exists():
        low_conc(q).to_csv(out / 'low_conc.csv', index=False)
    write_tables(out, df, chosen, med, mc.attrs['target'], q, q_sens)
    b_figures(out, df, chosen)
    write_provenance(out, 'summary', dict(q=q, q_sensitivity=q_sens, map_kept=chosen, map_deviation=med.to_dict(),
        amplitude_targets=mc.attrs['target']))
    print('->', out, flush=True)


def f3(x, spec: str = '.3f') -> str:
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def _row(df, folder, tag, reading_def, reading='graded'):
    g = df[(df.folder == folder) & (df.tag == tag) & (df.reading_def == reading_def) & (df.reading == reading)]
    return None if g.empty else g.iloc[0]


def _template_columns() -> List[Tuple[str, str, str]]:
    """(label, folder, tag) of every model column of the summary's best-models table (its matrices buttons)."""
    from apl_effect_scores import TEMPLATE, button_targets
    out = []
    for name, label in button_targets(TEMPLATE.read_text()).items():
        if name == 'data':
            continue
        folder, _, tag = name.partition('__')
        out.append((label, folder, tag))
    return out


def _fmt_rates(r) -> str:
    return f"{f3(r['rate_on'])} / {f3(r['rate_off'])} ({f3(r['rate_ratio'], '.2f')})"


def _key_line(label: str, reading_label: str, g: pd.Series, thr: Optional[pd.Series], binr: Optional[pd.Series]) -> str:
    flag = g['realism_flag'] if isinstance(g['realism_flag'], str) and g['realism_flag'] else ''
    return (f"| {label} | {reading_label} | {f3(g['rmse'])} | {f3(g['mean'], '+.3f')} | {f3(g['rmse_centered'])} | "
        f"{f3(g['spearman'], '.2f')} | "
        + (f"{f3(thr['rmse'])} ({f3(thr['mean'], '+.3f')})" if thr is not None else '–') + ' | '
        + (f"{f3(binr['rmse'])} ({f3(binr['mean'], '+.3f')})" if binr is not None else '–') + ' | '
        f"{_fmt_rates(g)} | {f3(g['rate_rho_on'], '.2f')} / {f3(g['rate_rho_off'], '.2f')} | "
        f"{f3(g['resp_amp_mean_on'], '.2f')} / {f3(g['resp_amp_mean_off'], '.2f')} | {100 * g['share_above_4_off']:.1f} % | "
        f"{100 * g['silent_share_off']:.0f} % / {100 * g['opk_8plus_off']:.0f} % | {f3(g['opk_on_tv'], '.2f')} / {f3(g['opk_off_tv'], '.2f')} | "
        f"{'yes: ' + flag if flag else 'no'} |")


def write_tables(out: Path, df: pd.DataFrame, chosen: str, med: pd.Series, amp_target: dict, q: float, q_sens: List[float]) -> None:
    """comparison_table.md and the summary CSVs (key_models.csv, order.csv, best_models_columns.csv)."""
    dt = H.data_target
    other = [m for m in MAPS if m != chosen][0]
    pa = pd.read_csv(out / 'data_edge_per_recording.csv')
    apl = pa[pa.panel == 'APL16']
    est = pd.read_csv(out / 'q_estimates.csv')
    synth = pd.read_csv(out / 'synthetic_per_recording.csv')
    L = [f'# E73: the per-KC normalization of the recordings (parts A and B)', '',
         f'Generated by E73_per_kc_normalization.py. Yang\'s responses (her answer of 2026-11-10, checked against her pipeline by viki: '
         'add_to_reports/yang_response_definition.md): each KC\'s Fc z-scored over the whole recording (about 7,194 volumes), a trial\'s '
         'response the mean z 2-8 s after onset (18 volumes) minus the mean over the 15 volumes before. The model\'s session is built the '
         f'same way: q = {q:.3f}, the share of the recording inside the 51 response windows (51 x 18 / 7,194), the response confined to its '
         f'window; a slow drift of {refs()["drift"]:.2f} times the baseline noise (the silent KCs\' whole-trace SD over baseline-noise SD, '
         '1.40); sensitivity at part A\'s estimate of a response\'s variance footprint, q = ' + ', '.join(f'{x:.3f}' for x in q_sens)
         + ' (the edge estimate and its 5th to 95th percentiles over recordings). Readings: "as now" are the present readings (raw spike '
         'counts, or E65\'s calibrated saturating readout, with the two-spike response criterion; amplitudes through E68\'s conversion to z); '
         f'"{MAP_LABEL["linear"]}" and "{MAP_LABEL["saturating"]}" are the per-KC session z-score reading (responses relative to the '
         'pre-odor baseline, in units of the KC\'s SD over the whole session), calibrated on APL on (6.8 % of entries above 1 z, a silent '
         'KC\'s trial sd 0.385 z and, for the saturating map, the responders\' mean amplitude 2.00 z) and applied unchanged to APL off, '
         'scored in expectation over the measurement noise; "trials" reads the model\'s saved jittered trials the same way. Every '
         'reading is scored against the same reading of the KC data.', '']
    # part A
    L += ['## Part A: the edge in the recordings', '',
          '| panel | group | recordings | largest per-KC mean square (z²), range over recordings | max minus 99.9th percentile | '
          'largest single trial (z) | largest trial mean (z) | q from the edge | q from the single-odor responders | q from the noise budget |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (panel, g), d in pa.groupby(['panel', 'genotype'], sort=False):
        L.append(f"| {panel} | {GENO_LABEL.get(g, g)} | {len(d)} | {d.ms_max.min():.2f} to {d.ms_max.max():.2f} | "
            f"{d.edge_gap.min():.2f} to {d.edge_gap.max():.2f} | {d.max_trial.min():.1f} to {d.max_trial.max():.1f} | "
            f"{d.max_trial_mean.min():.1f} to {d.max_trial_mean.max():.1f} | {d.q_edge.median():.3f} | {d.q_single.median():.3f} | {d.q_noise.median():.3f} |")
    s17 = synth[synth.q_true == 0.17]
    for how in NORMALIZATIONS:
        d = s17[s17.normalization == how]
        L.append(f"| synthetic, q 0.17 | {NORM_LABEL[how]} | {len(d)} | {d.ms_max.min():.2f} to {d.ms_max.max():.2f} | "
            f"{d.edge_gap.min():.2f} to {d.edge_gap.max():.2f} | {d.max_trial.min():.1f} to {d.max_trial.max():.1f} | "
            f"{d.max_trial_mean.min():.1f} to {d.max_trial_mean.max():.1f} | {d.q_edge.median():.3f} | {d.q_single.median():.3f} | {d.q_noise.median():.3f} |")
    n_edge = int((apl.edge_gap <= 0.6).sum())
    L += ['', f'The edge (the 99.9th percentile of the per-KC mean square within 0.6 z² of the recording\'s maximum) is in {n_edge} of the '
          f'{len(apl)} APL16 recordings. q estimates over recordings:', '',
          '| panel | group | estimator | recordings | median | 5th to 95th percentile |', '|---|---|---|---:|---:|---:|']
    for r in est.itertuples():
        L.append(f'| {r.panel} | {GENO_LABEL.get(r.genotype, r.genotype)} | {r.estimator} | {r.n_recordings} | {r.median:.3f} | {r.p5:.3f} to {r.p95:.3f} |')
    sq = pd.read_csv(out / 'synthetic_q_estimates.csv')
    L += ['', 'The estimators on synthetic cells with a known q (median over 19 synthetic recordings):', '',
          '| normalization | q | from the edge | from the single-odor responders | from the noise budget |', '|---|---:|---:|---:|---:|']
    for (how, qt), d in sq.groupby(['normalization', 'q_true']):
        v = d.set_index('estimator')['median']
        L.append(f"| {NORM_LABEL[how]} | {qt:.2f} | {v.get('q_edge', np.nan):.3f} | {v.get('q_single', np.nan):.3f} | {v.get('q_noise', np.nan):.3f} |")
    bt = pd.read_csv(out / 'data_edge_by_breadth.csv')
    sb = pd.read_csv(out / 'synthetic_by_breadth.csv')
    order_b = [_breadth_label(lo, hi) for lo, hi in BREADTH_BINS]
    L += ['', 'The solvent\'s trial mean by the KC\'s breadth (z; the session mean subtracted would put broad KCs\' solvent entries below 0):', '',
          '| | ' + ' | '.join(order_b) + ' |', '|---|' + '---:|' * len(order_b)]
    for g, d in bt[bt.panel == 'APL16'].groupby('genotype'):
        m = d.groupby('breadth')['blank_mean'].mean().reindex(order_b)
        L.append(f'| KCs, {GENO_LABEL[g]} | ' + ' | '.join(f3(v, '+.2f') for v in m) + ' |')
    for how in ('session', 'session_centered', 'baseline'):
        m = sb[(sb.normalization == how) & (sb.q_true == 0.17)].groupby('breadth')['blank_mean'].mean().reindex(order_b)
        L.append(f'| synthetic, {NORM_LABEL[how]} | ' + ' | '.join(f3(v, '+.2f') for v in m) + ' |')
    # Yang's answer and the known window share
    L += ['', f'Confirmed by Yang (2026-11-10): a z-score over the whole recording, the response relative to the 5 s before onset; part A '
          'found this from the per-trial values alone (the edge, the noise budget, the solvent\'s trial mean rising with breadth, which a '
          'subtraction of the session mean would push below 0). The 51 response windows hold a share '
          f'{q:.3f} of the recording, so a response confined to its window puts the edge at 1 / q = {1 / q:.1f} z²; the recordings\' '
          f'largest per-KC mean squares are {apl.ms_max.min():.1f} to {apl.ms_max.max():.1f} z², and part A\'s three estimators put the share '
          f'of a KC\'s session variance that a unit of squared response takes at {q_sens[1]:.3f} ({q_sens[0]:.3f} to {q_sens[2]:.3f}): a '
          'response\'s variance footprint is larger than its window (the GCaMP response rises before 2 s and decays after 8 s, outside the '
          'window but inside the z-score), so the main reading (q the window share) and the sensitivity reading (q the footprint) bracket '
          'the model\'s side.', '']
    # the breadth checks: the whole-trace SD over the baseline-noise SD, and the responders' amplitude in baseline-noise units
    ks = pd.read_csv(YANG_NORM / 'kc_scales.csv')
    ks['b'] = ks.n_odors_z1.clip(upper=11)
    ab = pd.read_csv(YANG_NORM / 'amplitude_by_breadth.csv')
    ab['b'] = ab.n_odors.clip(upper=11)
    lab_b = [str(b) if b < 11 else '11+' for b in BREADTH_CHECK]
    L += ['### The breadth checks (data: viki\'s whole-trace and baseline-noise SDs, notes/yang_response_normalization/)', '',
          'Whole-trace SD over baseline-noise SD, median over the KCs that respond (trial mean above 1 z) to the given number of odors '
          '(models: each KC at its expected number, the reading\'s S / noise):', '',
          '| | ' + ' | '.join(lab_b) + ' |', '|---|' + '---:|' * len(lab_b)]
    L.append('| KC data, both genotypes | ' + ' | '.join(f3(ks.loc[ks.b == b, 'scale_ratio'].median(), '.2f') for b in BREADTH_CHECK) + ' |')
    chk_defs = [(f'zcell_{chosen}', f'q {q:.3f}'), (f'zcell_saturating{q_tag(q_sens[1])}', f'q {q_sens[1]:.3f}'),
        (f'zcell_{other}', f'{other} map, q {q:.3f}')]
    for short, _, folder, tag, _ in KEY_MODELS:
        if short in ('C2 both', 'fit 3'):
            continue
        for rd, qlab in chk_defs:
            g = _row(df, folder, tag, rd)
            if g is None:
                continue
            for c, cl in (('on', 'APL on'), ('off', 'APL off')):
                L.append(f'| {short}, {cl}, {qlab} | ' + ' | '.join(f3(g.get(f"scale_ratio_b{b}_{c}", np.nan), '.2f') for b in BREADTH_CHECK) + ' |')
    L += ['', 'Responders\' mean amplitude in baseline-noise units (z times the KC\'s whole-trace SD over baseline-noise SD), by breadth:', '',
          '| | ' + ' | '.join(lab_b[1:]) + ' |', '|---|' + '---:|' * (len(lab_b) - 1)]
    for g_, gl in (('TNT3c', 'KC data, control flies'), ('TNT', 'KC data, APL output blocked (TNT)')):
        d = ab[ab.genotype == g_]
        L.append(f'| {gl} | ' + ' | '.join(f3(np.average(d.loc[d.b == b, 'amp_base'], weights=d.loc[d.b == b, 'n_kcs']) if (d.b == b).any() else np.nan, '.1f')
            for b in BREADTH_CHECK[1:]) + ' |')
    for short, _, folder, tag, _ in KEY_MODELS:
        if short in ('C2 both', 'fit 3'):
            continue
        for rd, qlab in chk_defs:
            g = _row(df, folder, tag, rd)
            if g is None:
                continue
            for c, cl in (('on', 'APL on'), ('off', 'APL off')):
                L.append(f'| {short}, {cl}, {qlab} | ' + ' | '.join(f3(g.get(f"amp_base_b{b}_{c}", np.nan), '.1f') for b in BREADTH_CHECK[1:]) + ' |')
    # the data under the two normalizations, beside the models
    nr = pd.read_csv(YANG_NORM / 'normalization_rates.csv')
    L += ['', '### The KC data in baseline-noise units (viki: one threshold, 1.97 baseline SDs, holds the control flies at 0.068)', '',
          '| | response rate, control flies / APL output blocked (TNT) (ratio) | KCs responding to no odor, control / TNT | KCs at 8 or more odors, control / TNT |',
          '|---|---:|---:|---:|']
    for norm, d in nr.groupby('normalization', sort=False):
        c, t_ = d[d.genotype == 'TNT3c'].iloc[0], d[d.genotype == 'TNT'].iloc[0]
        L.append(f"| KC data, {norm} | {c.response_rate:.3f} / {t_.response_rate:.3f} ({t_.response_rate / c.response_rate:.2f}) | "
            f"{c.odors_per_kc_0_0:.2f} / {t_.odors_per_kc_0_0:.2f} | {c.odors_per_kc_8_99:.3f} / {t_.odors_per_kc_8_99:.3f} |")
    for short, _, folder, tag, pres in KEY_MODELS:
        if short in ('C2 both', 'fit 3'):
            continue
        for rd, rl in ((f'present_{pres}', 'as now, two spikes'), (f'zcell_{chosen}', f'the reading, q {q:.3f}'),
                (f'zcell_saturating{q_tag(q_sens[1])}', f'the reading, q {q_sens[1]:.3f}')):
            g = _row(df, folder, tag, rd)
            if g is None:
                continue
            L.append(f"| {short}, {rl} | {g['rate_on']:.3f} / {g['rate_off']:.3f} ({g['rate_ratio']:.2f}) | {g['silent_share_on']:.2f} / "
                f"{g['silent_share_off']:.2f} | {g['opk_8plus_on']:.3f} / {g['opk_8plus_off']:.3f} |")
    L += ['', 'The two-spike criterion counts a model KC\'s response in fixed units, as baseline-noise units count the flies\'; the reading '
          'applies the flies\' criterion in the flies\' units.', '']
    # map choice
    L += ['', '## Part B: the models read the same way', '',
          f'Map kept: **{chosen}** (median over the variants without a realism flag of the mean relative deviation of the APL-on '
          'amplitude statistics from the control flies\': ' + ', '.join(f'{m} {v:.3f}' for m, v in med.items()) + '; statistics: '
          + ', '.join(f'{k} {v:.3g}' for k, v in amp_target.items()) + ').', '']
    # key models
    data_row = (f"| KC data | | floor {dt('apl_effect_rmse_floor'):.3f} | {dt('apl_effect_mean'):+.3f} | | ceiling {dt('apl_effect_rho_ceiling'):.2f} | "
        f"floor {dt('apl_effect_thresholded_rmse_floor'):.3f} ({dt('apl_effect_thresholded_mean'):+.3f}) | "
        f"floor {dt('apl_effect_binary_rmse_floor'):.3f} ({dt('apl_effect_binary_mean'):+.3f}) | "
        f"{dt('ctrl_frac'):.3f} / {dt('tnt_frac'):.3f} ({dt('rate_ratio_fraction'):.2f}) | ceiling | "
        f"{dt('ctrl_resp_amp_mean_z'):.2f} / {dt('tnt_resp_amp_mean_z'):.2f} | {100 * dt('tnt_share_z_above_4'):.1f} % | "
        f"{100 * dt('tnt_odors_per_kc_0'):.0f} % / {100 * dt('tnt_odors_per_kc_8plus'):.0f} % | "
        f"bands {dt('ctrl_odors_per_kc_tv_max'):.3f} / {dt('tnt_odors_per_kc_tv_max'):.3f} | |")
    head = ('| model | reading | APL-effect loss, graded | mean APL effect | pattern part | ρ on the per-pair APL effect | '
            'loss, sub-threshold set to 0 (mean) | loss, binarized (mean) | response rates, APL on / off (ratio) | response rate ρ, APL on / off | '
            'responders\' amplitude, APL on / off (z) | entries above 4 z, APL off | KCs silent / at 8 or more odors, APL off | '
            'odors per KC, distance from the flies, APL on / off | realism flag |')
    L += [head, '|---|---|' + '---:|' * 12 + '---|']
    key_rows = []
    for short, label, folder, tag, pres in KEY_MODELS:
        for rd_def, rd_label in ((f'present_{pres}', 'as now: ' + ('raw counts' if pres == 'raw' else 'calibrated readout') + ', two-spike criterion'),
                ('present_raw' if pres == 'cal' else 'present_cal', 'as now: ' + ('raw counts' if pres == 'cal' else 'calibrated readout') + ', two-spike criterion'),
                (f'zcell_{chosen}', MAP_LABEL[chosen]), (f'zcell_{other}', MAP_LABEL[other]),
                (f'zcell_{chosen}_trials', MAP_LABEL[chosen] + ', jittered trials'),
                *[(f'zcell_saturating{q_tag(x)}', MAP_LABEL['saturating'] + f', q {x:.3f}') for x in q_sens],
                ('zcell_saturating_trials' + q_tag(q_sens[1]), MAP_LABEL['saturating'] + f', jittered trials, q {q_sens[1]:.3f}'),
                ('zcell_saturating_centered', MAP_LABEL['saturating'] + ', session mean subtracted')):
            g = _row(df, folder, tag, rd_def)
            if g is None:
                continue
            binr = _row(df, folder, tag, 'present_binary', 'binary') if rd_def.startswith('present') else _row(df, folder, tag, rd_def, 'binary')
            thr = None if rd_def.startswith('present') else _row(df, folder, tag, rd_def, 'thresholded')
            L.append(_key_line(short, rd_label, g, thr, binr))
            key_rows.append(dict(model=short, folder=folder, tag=tag, reading_def=rd_def, **{k: g[k] for k in g.index if k not in ('folder', 'tag', 'reading_def')},
                loss_thresholded=np.nan if thr is None else thr['rmse'], mean_thresholded=np.nan if thr is None else thr['mean'],
                loss_binary=np.nan if binr is None else binr['rmse'], mean_binary=np.nan if binr is None else binr['mean']))
    L.append(data_row)
    pd.DataFrame(key_rows).to_csv(out / 'key_models.csv', index=False)
    L += ['', 'Lower loss is better; each loss against the same reading of the KC data; "pattern part" the loss with both means removed. '
          'The "as now" rows\' amplitudes are E68\'s conversion of the readout to z (c set so that 6.8 % of entries are above 1 z through '
          'the data\'s measurement noise), their response statistics under the two-spike criterion; the reading\'s rows under the data\'s '
          'criterion (trial mean above 1 z). C2 is quoted under the calibrated readout and E69\'s fits under raw counts.', '']
    # part C condition
    cond = []
    for short, folder, tag, pres in (('C2', 'E67_apl_effect_fits', 'rate0.068_aplcal_k2_cap3', 'cal'),
            ('fit 1', 'E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.04142t1000_k2', 'raw')):
        p0, z1 = _row(df, folder, tag, f'present_{pres}'), _row(df, folder, tag, f'zcell_{chosen}')
        if p0 is not None and z1 is not None:
            cond.append((short, p0['mean'], z1['mean'], z1['mean'] - p0['mean']))
        z2 = _row(df, folder, tag, f'zcell_saturating{q_tag(q_sens[1])}')
        if p0 is not None and z2 is not None:
            cond.append((short + f' (q {q_sens[1]:.3f})', p0['mean'], z2['mean'], z2['mean'] - p0['mean']))
    moved = any(abs(c[3]) >= 0.03 for c in cond)
    L += ['### Part C\'s condition', '',
          f'- the edge in every APL16 recording: {"yes" if n_edge == len(apl) else "no"} ({n_edge} of {len(apl)}; gap 0.6 z² or less)',
          *[f'- {s}: mean APL effect {a:+.3f} as now, {b:+.3f} under the reading ({MAP_LABEL[chosen]}), change {d:+.3f} (KC data {dt("apl_effect_mean"):+.3f})'
            for s, a, b, d in cond],
          f'- condition met: {"yes" if (n_edge == len(apl) and moved) else "no"}', '']
    # order of the variants
    real = lambda d: d[d.realism_flag_by_variant.fillna('') == ''] if 'realism_flag_by_variant' in d else d
    rows_o = []
    for rd in READINGS:
        z = df[(df.reading_def == f'zcell_{chosen}') & (df.reading == rd)].set_index(['folder', 'tag'])
        for pres in ('present_raw', 'present_cal', 'present_binary'):
            p = df[(df.reading_def == pres)].set_index(['folder', 'tag'])
            if pres != 'present_binary' and rd == 'binary':
                continue
            if pres == 'present_binary' and rd != 'binary':
                continue
            common = z.index.intersection(p.index)
            zr = real(z.loc[common].reset_index()).set_index(['folder', 'tag'])
            rows_o.append(dict(reading=rd, present=pres, n_variants=len(common), rho_all=H.spearman(z.loc[common, 'rmse'], p.loc[common, 'rmse']),
                n_realistic=len(zr), rho_realistic=H.spearman(zr['rmse'], p.loc[zr.index, 'rmse']),
                rho_mean_all=H.spearman(z.loc[common, 'mean'], p.loc[common, 'mean']),
                median_change_loss=float((z.loc[common, 'rmse'] - p.loc[common, 'rmse']).median()),
                median_change_mean=float((z.loc[common, 'mean'] - p.loc[common, 'mean']).median())))
    order = pd.DataFrame(rows_o)
    order.to_csv(out / 'order.csv', index=False)
    L += ['### The order of the variants under the reading against the present order', '',
          '| reading | against | variants | Spearman of the losses, all / without a realism flag (present definitions) | Spearman of the mean APL effects | '
          'median change of the loss | median change of the mean APL effect |', '|---|---|---:|---:|---:|---:|---:|']
    for r in order.itertuples():
        L.append(f'| {READING_LABEL[r.reading]} | {r.present} | {r.n_variants} | {r.rho_all:.2f} / {r.rho_realistic:.2f} | {r.rho_mean_all:.2f} | '
            f'{r.median_change_loss:+.3f} | {r.median_change_mean:+.3f} |')
    # best-models columns
    cols = _template_columns()
    bm = []
    for label, folder, tag in cols:
        p, pc, z = _row(df, folder, tag, 'present_raw'), _row(df, folder, tag, 'present_cal'), _row(df, folder, tag, f'zcell_{chosen}')
        if p is None or z is None:
            bm.append(dict(column=label, folder=folder, tag=tag, missing=True))
            continue
        bm.append(dict(column=label, folder=folder, tag=tag, loss_raw=p['rmse'], mean_raw=p['mean'], loss_cal=pc['rmse'], mean_cal=pc['mean'],
            loss_zcell=z['rmse'], mean_zcell=z['mean'], rho_zcell=z['spearman'], rate_on_zcell=z['rate_on'], rate_ratio_zcell=z['rate_ratio'],
            flag_present=p['realism_flag'], flag_zcell=z['realism_flag']))
    bm = pd.DataFrame(bm)
    if 'loss_raw' in bm:
        ok = bm.loss_raw.notna()
        bm.loc[ok, 'rank_raw'] = bm.loc[ok, 'loss_raw'].rank()
        bm.loc[ok, 'rank_cal'] = bm.loc[ok, 'loss_cal'].rank()
        bm.loc[ok, 'rank_zcell'] = bm.loc[ok, 'loss_zcell'].rank()
    bm.to_csv(out / 'best_models_columns.csv', index=False)
    L += ['', '### The best-models columns', '',
          '| column | loss (mean APL effect), raw counts | rank | calibrated readout | rank | under the reading | rank | response rates under the data\'s criterion, APL on (ratio) | realism flag under the reading |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in bm.itertuples():
        if getattr(r, 'missing', False) is True:
            L.append(f'| {r.column} | not archived | | | | | | | |')
            continue
        L.append(f'| {r.column} | {r.loss_raw:.3f} ({r.mean_raw:+.3f}) | {r.rank_raw:.0f} | {r.loss_cal:.3f} ({r.mean_cal:+.3f}) | {r.rank_cal:.0f} | '
            f'{r.loss_zcell:.3f} ({r.mean_zcell:+.3f}) | {r.rank_zcell:.0f} | {r.rate_on_zcell:.3f} ({r.rate_ratio_zcell:.2f}) | {r.flag_zcell if r.flag_zcell else "no"} |')
    if 'loss_raw' in bm and bm.loss_raw.notna().sum() > 2:
        L += ['', f"Spearman of the columns' order, the reading against raw counts {H.spearman(bm.loss_zcell, bm.loss_raw):.2f}, against the calibrated readout "
              f"{H.spearman(bm.loss_zcell, bm.loss_cal):.2f}."]
    # experiment groups
    for title, folders in (('The firing ceiling\'s gain (E69\'s fits against C2)', ['E67_apl_effect_fits', 'E69_kc_firing_ceiling']),
            ('The wiring population (E72, as archived)', ['E72_wiring_population']),
            ('The input-scaled threshold (E71, as archived)', ['E71_input_scaled_threshold'])):
        g = df[df.folder.isin(folders) & (df.reading == 'graded') & df.reading_def.isin(['present_raw', 'present_cal', f'zcell_{chosen}', f'zcell_{other}'])]
        if g.empty:
            continue
        if folders[0] == 'E67_apl_effect_fits':
            keep = {t for _, _, f, t, _ in KEY_MODELS if f in folders}
            g = g[g.tag.isin(keep)]
        w = g.pivot_table(index=['folder', 'tag'], columns='reading_def', values=['rmse', 'mean'], aggfunc='first')
        rr = g[g.reading_def == f'zcell_{chosen}'].set_index(['folder', 'tag'])
        L += ['', f'### {title}', '', f'| variant | loss (mean APL effect): raw counts | calibrated readout | {MAP_LABEL[chosen]} | {MAP_LABEL[other]} | '
              'response rates under the data\'s criterion (ratio) | realism flag under the reading |', '|---|---:|---:|---:|---:|---:|---|']
        for idx in w.index:
            cells = []
            for rd in ('present_raw', 'present_cal', f'zcell_{chosen}', f'zcell_{other}'):
                try:
                    cells.append(f"{w.loc[idx, ('rmse', rd)]:.3f} ({w.loc[idx, ('mean', rd)]:+.3f})")
                except KeyError:
                    cells.append('–')
            r = rr.loc[idx] if idx in rr.index else None
            L.append(f'| {idx[1]} | ' + ' | '.join(cells) + ' | ' + (_fmt_rates(r) if r is not None else '–') + ' | '
                + ((r['realism_flag'] or 'no') if r is not None else '–') + ' |')
        if folders[0] == 'E72_wiring_population' and len(w) > 1:
            L.append('| mean over the archived variants | ' + ' | '.join(
                f"{w[('rmse', rd)].mean():.3f} ({w[('mean', rd)].mean():+.3f})" if ('rmse', rd) in w else '–'
                for rd in ('present_raw', 'present_cal', f'zcell_{chosen}', f'zcell_{other}')) + ' | | |')
    # medians over the variants
    L += ['', '### Over every archived variant (medians; variants without a realism flag under the present definitions)', '',
          '| reading | variants | loss, graded | mean APL effect | pattern part | ρ on the per-pair APL effect | APL-off / APL-on response rate | '
          'flagged under this reading\'s criterion |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for rd_def in ('present_raw', 'present_cal', f'zcell_{chosen}', f'zcell_{other}', *[f'zcell_saturating{q_tag(x)}' for x in q_sens], 'zcell_saturating_centered'):
        g = real(df[(df.reading_def == rd_def) & (df.reading == 'graded')])
        if g.empty:
            continue
        L.append(f"| {rd_def} | {len(g)} | {g.rmse.median():.3f} | {g['mean'].median():+.3f} | {g.rmse_centered.median():.3f} | {g.spearman.median():.2f} | "
            f"{g.rate_ratio.median():.2f} | {int((g.realism_flag != '').sum())} |")
    lc = pd.read_csv(out / 'low_conc.csv')
    lc = lc[lc['map'] == chosen]
    if len(lc):
        models = list(dict.fromkeys(lc.model))
        L += ['', f'### The low-concentration panels within their own session (E49\'s archived models; {MAP_LABEL[chosen]}, calibrated on the '
              'APL16 odors)', '', 'Fraction of KCs responding: two-spike criterion / the data\'s criterion under the reading; E72 part F\'s fits (C2 among '
              'them) are not archived, so C2\'s low-concentration fractions are not read here.', '',
              '| panel | condition | odor | KC data | ' + ' | '.join(models) + ' |', '|---|---|---|---:|' + '---:|' * len(models)]
        for (panel, cond, odor), g in lc.groupby(['panel', 'condition', 'odor'], sort=False):
            g = g.set_index('model')
            dv = g.frac_data.iloc[0]
            L.append(f"| {panel} | {'APL on / control flies' if cond == 'on' else 'APL off / APL output blocked (TNT)'} | {odor} | {f3(dv)} | "
                + ' | '.join(f"{g.loc[m, 'frac_two_spikes']:.3f} / {g.loc[m, 'frac_zcell']:.3f}" if m in g.index else '–' for m in models) + ' |')
    (out / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def b_figures(out: Path, df: pd.DataFrame, chosen: str) -> None:
    """E73_loss_present_vs_reading.png: every variant's graded APL-effect loss and mean APL effect, the
    present readings against the reading."""
    plt = _plt()
    dt = H.data_target
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    z = df[(df.reading_def == f'zcell_{chosen}') & (df.reading == 'graded')].set_index(['folder', 'tag'])
    for j, pres in enumerate(('present_raw', 'present_cal')):
        p = df[(df.reading_def == pres) & (df.reading == 'graded')].set_index(['folder', 'tag'])
        common = z.index.intersection(p.index)
        flag = (z.loc[common, 'realism_flag'].fillna('') != '').to_numpy()
        for i, (col, lab, ref) in enumerate((('rmse', 'APL-effect loss, graded (lower is better)', dt('apl_effect_rmse_floor')),
                ('mean', 'mean APL effect', dt('apl_effect_mean')))):
            ax = axes[i, j]
            x, y = p.loc[common, col].to_numpy(), z.loc[common, col].to_numpy()
            ax.scatter(x[~flag], y[~flag], s=8, c='#2b6cb0', label='inside the bands under the data\'s criterion')
            ax.scatter(x[flag], y[flag], s=8, c='0.7', marker='x', lw=0.6, label='realism flag under the data\'s criterion')
            lim = [min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))]
            ax.plot(lim, lim, ':', color='0.5')
            ax.axhline(ref, color='k', ls='--', lw=0.8)
            ax.text(lim[0], ref, ' KC data' + (' floor' if col == 'rmse' else ''), fontsize=7, va='bottom')
            ax.set_xlabel(f"{lab}, {'raw counts' if pres == 'present_raw' else 'calibrated readout'} (as now)", fontsize=8.5)
            ax.set_ylabel(f'{lab}, {MAP_LABEL[chosen]}', fontsize=8.5)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(True, color='0.93', lw=0.6)
    axes[0, 0].legend(fontsize=7, frameon=False)
    for short, _, folder, tag, _ in KEY_MODELS:
        if (folder, tag) in z.index:
            for j, pres in enumerate(('present_raw', 'present_cal')):
                p = _row(df, folder, tag, pres)
                for i, col in enumerate(('rmse', 'mean')):
                    axes[i, j].annotate(short, (p[col], z.loc[(folder, tag), col]), fontsize=7, color='#c8553d')
    fig.suptitle('Every archived variant: the per-pair APL effect read as the recordings appear to be against the present readings', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'E73_loss_present_vs_reading.png', dpi=120); plt.close(fig)
    # the key models: loss and mean APL effect under each reading
    other = [m for m in MAPS if m != chosen][0]
    defs = [('present_raw', 'raw counts (as now)', '0.75'), ('present_cal', 'calibrated readout (as now)', '0.45'),
        (f'zcell_{other}', MAP_LABEL[other], '#9ecae1'), (f'zcell_{chosen}', MAP_LABEL[chosen], '#2b6cb0'),
        (f'zcell_{chosen}_trials', MAP_LABEL[chosen] + ', jittered trials', '#c8553d')]
    keys = [k for k in KEY_MODELS if _row(df, k[2], k[3], 'present_raw') is not None]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.8 / len(defs)
    for i, (rd, lab, colr) in enumerate(defs):
        for ax, col in zip(axes, ('rmse', 'mean')):
            vals = [(_row(df, f, t, rd)[col] if _row(df, f, t, rd) is not None else np.nan) for _, _, f, t, _ in keys]
            ax.bar(np.arange(len(keys)) + (i - len(defs) / 2 + 0.5) * w, vals, w, color=colr, label=lab)
    axes[0].axhline(dt('apl_effect_rmse_floor'), color='k', ls='--', lw=0.8); axes[0].text(-0.4, dt('apl_effect_rmse_floor'), 'KC data floor', fontsize=7, va='bottom')
    axes[1].axhline(dt('apl_effect_mean'), color='k', ls='--', lw=0.8); axes[1].text(-0.4, dt('apl_effect_mean'), 'KC data', fontsize=7, va='bottom')
    for ax, ylab in zip(axes, ('APL-effect loss, graded (lower is better)', 'mean APL effect')):
        ax.set_xticks(np.arange(len(keys))); ax.set_xticklabels([k[0] for k in keys]); ax.set_ylabel(ylab)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Key models under the present readings and read as the recordings appear to be (each against the same reading of the KC data)', fontsize=10)
    fig.tight_layout(); fig.savefig(out / 'E73_key_models.png', dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['A', 'B', 'summarize'])
    ap.add_argument('--jobs', type=int, default=16, help='worker processes (part B; at most 16)')
    ap.add_argument('--only-key', action='store_true', help='part B on the key models only (a quick check)')
    ap.add_argument('--q', type=float, default=None, help='part B: q (default: part A\'s estimate)')
    ap.add_argument('--folders', nargs='+', default=None, help='part B: score only these results folders (e.g. once E71 / E72 are '
        'archived) and replace their rows in by_variant.csv')
    args = ap.parse_args()
    {'A': part_a, 'B': part_b, 'summarize': summarize}[args.part](args)


if __name__ == '__main__':
    main()
