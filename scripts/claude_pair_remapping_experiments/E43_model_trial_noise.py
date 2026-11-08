#!/usr/bin/env python3
"""E43: trial-to-trial variability in the model (part A: PN noise; B: KC excitability noise; C: input gain noise).

Every other experiment compares one deterministic model run with KC matrices built from trial
means of three noisy trials per odor, averaged over recordings. Here two models get
trial-to-trial variability, and their trials go through the data's own pipeline
(noise_floor_trial_distances.py: `analyse` per recording, then the mean over recordings), so the
model's floors, raw and noise-corrected mean distances, ρ and response rates are measured as the
data's are, and the noisy model's grand mean can be compared with the deterministic run.

Noise. olfsysm adds N(0, sd) to dPN/dt of every glomerulus at every time step (sim_PN_layer,
`mp.pn.noise.sd`, 0 in every other experiment), so the PN rate carries an Ornstein-Uhlenbeck-like
fluctuation with the PN time constant (10 ms), of stationary SD 0.16 * sd at dt = 0.5 ms (before
the rectification at 0; the increment is not scaled by sqrt(dt), so the same sd is a different
fluctuation at another time step). It is shared by every bouton and claw of a glomerulus and
independent across glomeruli, odors and trials. The generator is a thread_local std::mt19937
seeded from std::random_device (build_wPNKC reseeds it only when it draws a random wiring, not for
these preset ones), so draws are independent across trials, processes and resumed runs, but not
reproducible. A trial: set the sd, `osm.run_PN_sims(mp, rv)`, then the KC layer at fixed
parameters through `make_fixed_param_kc_simulator`'s simulate (`run_KC_sims(mp, rv, False)`:
wiring, thresholds and APL weights fixed), with APL on and with APL off acutely (APL->KC and
APL->bouton scales 0) on the same noisy PN trial. The simulator's mp / rv are captured by wrapping
mb_model.fit_mb_model while the simulator is built (no library change). Spike counts are over the
whole simulated period (-0.5 to 0.75 s around the 0.5 s odor), as in every experiment.

Models. base70: the claw-driven bouton base with the 70 Hz per-claw threshold at its 5 % tuned
parameters (E16's bouton_thr70, E19's thr70_k0_kckc0 row). ref40: the reference model, E34's
thr40_gain1.5_k2_cap3 at its fitted parameters (results/E34_gain_bounded/fits.csv), built as E39
builds it (E10_joint_fit.bouton_base with the gain, breadth_weights, its KC-KC tau). Built-in
checks: a trial at sd = 0 reproduces the deterministic run exactly (asserted for APL on and off
when a model is built, and again after its noisy trials), and the deterministic run equals E16's /
E34's saved outputs (asserted when those files are present).

Calibration. Per model, `--cal-trials` trials at each sd of `--cal-grid`, then `--cal-refine`
rounds of trials at the interpolated estimates. The floor (the single-trial same-odor distance of
graded spike counts, as the data's) is not monotone in sd: it rises from 0, peaks, and falls again
once the rectified noise drives every KC on every trial (the base model at sd 1000 responds in
99.8 % of KC x odor trials, and its trials are near-identical). The grid is extended (by its last
ratio, up to `--cal-max-extend` points) upward only while a floor below its target is still rising
at the top, and downward while the lowest point is above the target. The matched sd is interpolated
linearly in log sd at the first upward crossing: APL on matched to the control flies' floor
(data_targets ctrl_single_trial_floor), APL off to APL output blocked (TNT)'s
(tnt_single_trial_floor); when a floor never reaches its target, refinement locates its peak
instead, and the peak's sd stands in (flagged bracketed = false in noise_levels.json). Levels: the
two matched sds, `--bracket`[0] x the smaller, and `--bracket`[1] x the larger, unless that lies
beyond a floor's peak, in which case `--bracket`[0]^2 x the smaller (defaults 0.5 and 2): four
levels, three if the two matched sds coincide.

Pseudo-recordings. At each level, `--n-recordings` pseudo-recordings (default 10) of `--n-trials`
trials (default 3) of all 16 odors: all model KCs of the one wiring (pseudo-recordings differ only
in noise); a trial's value is the KC's spike count in it (the model's primary readout; the data's
is z-scored fluorescence). Transforms as for the data: 'graded' (spike counts) and 'thresholded'
(counts <= 1 set to 0: only trials with a response, 2 spikes, keep their count).

Readouts per model x level x condition x transform (the KC data through the same functions):
floors (single trial, trial mean, reliability); the mean pair distance of the 3-trial means
averaged over pseudo-recordings (raw) and noise-corrected, against the deterministic run and the
data; the grand mean over all trials against the deterministic run (response rate, per-odor
response fractions, the weak odors, spike counts, distances: the bias from noise passing through
the thresholds); ρ on odor-odor distances of the noisy, corrected, grand-mean and deterministic
matrices with the KC data (raw and corrected) and with the deterministic model, and the model's own
split-half ρ over pseudo-recordings (its measurement ceiling, computed as the data's); response
fractions under four definitions (the per-trial response averaged over trials; a response in at
least 2 of 3 trials; trial-mean spike count >= 1; trial-mean spike count >= 2, the per-trial
criterion applied to the trial mean) with their response rate ρ with the data; the block, the
isotonic R² and ρ between the APL-on and APL-off matrices (raw, corrected, grand mean,
deterministic; the data likewise); and the PN noise in Hz (pre-odor temporal SD, trial-to-trial SD
of the odor-window mean rate), to judge how plausible the matched noise is.

Outputs: <shared>/results/E43_model_trial_noise/: stats_noise.csv (one row per model x level x
condition x transform; deliberately not named stats_variants.csv, so the progress plot does not
take a noise-driven value as a best model), onoff_noise.csv, per_odor_noise.csv,
pairs_noise_long.csv, calibration.csv, noise_levels.json, checks.json, provenance.json (code
provenance taken at the start of each session, under "model"; earlier sessions kept under
"previous_sessions") and E43_model_trial_noise.png. ./outputs/E43_model_trial_noise/: the
deterministic runs (<model>_{on,acute_off}_{responses,spike_counts}.parquet), <model>_meta.json,
<model>_det_pn.npz, and the trials (calibration/ and trials/: npz with per-trial spike counts,
responses and PN summaries). Resumable: finished trial files are skipped, and a model is only built
when something of it is missing.

Part B (`--part B`): trial-level KC excitability noise at a fixed operating point (the amendment
to E43's entry; part A's PN noise cannot reach the data's floors). No PN noise and no PN re-runs:
the PN sims stay the deterministic ones, and a trial is only the KC layer (`run_KC_sims(regen=False)`
through the simulator, APL on and acute APL off). What is jittered is rv.kc.thr itself, the per-KC
threshold compared with Vm at every step: fit_mb_model (add_fixed_thr_to_spont) and the simulator
set it to fixed_thr + 2 * kc.spont_in, so thr0 includes the spontaneous-input offset, and a trial
uses max(thr0 * (1 + sigma * xi), thr_min_frac * thr0) (`--thr-min-frac`, default 0.01, i.e. clipped
above 0), xi standard normal per KC and trial, the same xi for the APL-on and the APL-off run. (For
the two models here spont_in is reported in checks.json; where it is 0, thr0 = fixed_thr.) The
override is a proxy for mb_model's `osm` during the simulator call, so simulate still sets every
weight, and rv.kc.thr is checked unchanged after each run. At each sigma, fixed_thr is re-tuned
(APL weights fixed) on `--tune-trials` trials with fixed draws, so that the per-trial APL-on
response fraction averaged over those trials equals the deterministic run's (0.0472 for base70,
tuned at 5 %): a bracketing bisection (multiplicative steps from the threshold interpolated
between the sigmas tuned so far, until the target is bracketed, then splits at the log-rate interpolation, kept inside
the middle 80 % of the bracket) to `--rate-tol` (relative, default 1 %). This holds the operating
point fixed, which part A confounds with the noise. The tuning trials (APL on and off at the tuned
threshold) are the calibration point at that sigma; calibration (grid `--sigma-grid`, extension
and refinement as part A) matches the APL-on floor to the control flies' and the APL-off floor to
APL output blocked (TNT)'s, and the levels are the two matched sigmas plus `--below` (0.5, 0.25) x
the smaller; each level is re-tuned, then `--n-recordings` pseudo-recordings of `--n-trials`
trials, analysed exactly as part A (same functions; noise column `sigma`; extra columns fixed_thr
(re-tuned), fixed_thr_det, fixed_thr_ratio, tune_rate, frac_clipped, and the low tail's weight:
max_spike_count and spike_share_thr_below_half, the share of spikes from KCs whose threshold was
jittered below half its mean; a KC near zero threshold fires on almost every step to any input, so
at sigma >= ~0.3 a few such KCs can carry much of a trial's graded spike counts). `--thr-jitter
lognormal` replaces the factor by exp(sigma * xi - sigma^2 / 2) (mean 1, no low tail; folders
partB_lognormal), as an alternative if that tail dominates. Draws are reproducible:
numpy default_rng(SeedSequence(`--seed`, spawn_key=(model, kind, rep))), the same draws at every
sigma. sigma = 0 through the jitter path (with a real draw) reproduces the deterministic run
exactly (asserted at build and after the noisy trials).

Part C (`--part C`): slow input variability instead, a per-trial gain on each glomerulus x odor
ORN input delta (mp.orn.data.delta x max(1 + sigma * eta, 0), eta per glomerulus, odor and trial,
constant within the trial), re-running the ORN / LN, PN and KC layers per trial (APL on and off on
the same input), with KC thresholds unjittered and fixed_thr re-tuned per sigma as in part B;
everything else as part B (draw kinds 3-5).

Parts B / C write to <results>/partB|partC and <outputs>/partB|partC (same file names as part A;
trial files named by sigma), with their own provenance.json (code provenance at the start, under
"model"), so part A's files are never touched.

Models from other experiments' joint fits (`--model-from EXPERIMENT:TAG`, repeatable; E48): the model
of row TAG of <shared>/results/EXPERIMENT/fits.csv (a fit of E10_joint_fit.py) at its fitted
parameters, built as ref40 is from E34's row (`fit_model`, which builds ref40 too): bouton_base at the
row's claw_thr, with its gain on gain_glomeruli (empty: the oracle set, as in E10_joint_fit), its
re-drawn wiring (wiring, wiring_seed) and its ORN input (orn_input, read with
E10_joint_fit.read_orn_input); breadth weights with exponent k from the APL-off run of that wiring at
the base's tuned parameters; the fitted fixed_thr, APL scales and kc_kc_inh (tau kc_kc_tau). The
APL->bouton bound (aplpn_max) and the targets of the loss do not matter at fixed parameters; rows
with APL driven by KC spikes (apl_drive spikes / spikes_only, E36 / E37) or from a --base-params
base are refused. Name: E<n>-TAG, E<n> the experiment's ID (e.g. E45-thr70_gain1.25_rate0.068_k2_cap3;
the whole folder name if it has no E<n>_ prefix); file names, tables and noise_levels.json /
checks.json use it. Checks, as for ref40: the deterministic run equals the fit's saved outputs
(./outputs/EXPERIMENT, else the shared archive data/model_outputs/EXPERIMENT; when neither has them, a
warning and `missing` in checks.json), its response rates equal the row's rate_on / rate_off, and
sigma = 0 reproduces it. These models draw the noise of ref40 (draw index 1), so all models defined
by fits share their draws (common random numbers across models). With --model-from, a run includes
only those models plus any named by --models (without --model-from the default is base70 and ref40).
`--experiment NAME` puts the folders under results/NAME and outputs/NAME instead of E43's (part A in
a partA subfolder there, parts B / C in partB / partC as above).

Parts B / C only: `--sigma-from MODEL` skips the calibration and takes the matched sigmas (APL on and
APL off) of MODEL from an existing noise_levels.json of the same noise (`--sigma-from-dir`; default
this run's results folder if it holds MODEL, else E43's own folder for the part, e.g.
results/E43_model_trial_noise/partB); each level is still re-tuned to the model's own rate, and those
tuning trials are its calibration rows (kind 'sigma_from'). `--levels match_on ...` (all parts)
simulates pseudo-recordings only at the levels with one of those labels (a model's levels in
noise_levels.json are those of its last run). The tables are rewritten by every analysis, but a model
whose trial files are not on this host keeps its rows from the existing tables, as its entries in
noise_levels.json, checks.json and calibration.csv are kept.

Usage: OMP_NUM_THREADS=12 nice -n 10 ./venv/bin/python E43_model_trial_noise.py [--models base70 ref40]
       [--n-recordings 10] [--n-trials 3] [--cal-grid 25 50 100 200 400 800] [--cal-trials 3]
       [--analyse-only] [--figure-only]
       ... --part B [--sigma-grid 0.05 0.1 0.15 0.2 0.3] [--tune-trials 3] [--seed 43] [--rate-tol 0.01]
       [--below 0.5 0.25] [--cal-refine 1] [--cal-max-extend 3]
       ... --part C [--sigma-grid 0.1 0.2 0.3 0.45 0.6] (other options as part B)
       ... [--model-from EXPERIMENT:TAG ...] [--models ...] [--experiment NAME] [--levels LABEL ...]
       [--sigma-from MODEL [--sigma-from-dir DIR]]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

import kc_remapping_harness as H
import noise_floor_trial_distances as NF
from E10_joint_fit import TAU, base_kws, bouton_base, breadth_weights, e19_row, read_orn_input
from E26_claw_power import WEAK
from E27_glomerulus_gain import ORACLE

EXPERIMENT = 'E43_model_trial_noise'
# (--model-from adds its models to MODELS and FIT_MODELS)
MODELS = {
    'base70': 'claw-driven bouton base, 70 Hz per-claw threshold, tuned at 5 %',
    'ref40': 'reference model (thr40_gain1.5_k2_cap3 at its fitted parameters)',
}
REF_FIT = 'thr40_gain1.5_k2_cap3'
REF_EXPERIMENT = 'E34_gain_bounded'
# models defined by a row of a joint fit's fits.csv: name -> (experiment, tag)
FIT_MODELS = {'ref40': (REF_EXPERIMENT, REF_FIT)}
BASE_CLAW_THR = 70.0
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
CONDS = ('on', 'acute_off')
GENOTYPE = {'on': 'TNT3c', 'acute_off': 'TNT'}
DATA_PREFIX = {'on': 'ctrl', 'acute_off': 'tnt'}
TRANSFORMS = ('graded', 'thresholded')
ODORS = list(H.APL16_ODORS)
FRACTION_DEFS = ('trial', '2of3', 'count1', 'mean_resp')
# (det_pn) PN entries counted as driven for the coefficient of variation of the window mean
PN_ACTIVE_HZ = 20.0
NOISE_NOTE = ('olfsysm sim_PN_layer adds N(mp.pn.noise.mean = 0, mp.pn.noise.sd) to dPN/dt of every glomerulus at every '
    'time step; PN rate += dPNdt * dt / pn.taum, then rectified at 0. Stationary SD of the fluctuation (unrectified) = '
    'sd * (dt / taum) / sqrt(1 - (1 - dt / taum)^2) = 0.160 * sd at dt 0.5 ms, taum 10 ms; correlation time taum; not '
    'scaled by sqrt(dt). Generator: thread_local std::mt19937 seeded from std::random_device (not reseeded for preset '
    'wirings): trials independent, not reproducible. APL on and APL off share each noisy PN trial.')


# ---------------------------------------------------------------------------
# small helpers
def r3(x: float) -> float:
    """`x` rounded to 3 significant digits (noise sds are kept at this precision)."""
    return float(f'{float(x):.3g}')


def sd_tag(sd: float) -> str:
    return format(float(sd), '.4g').replace('+', '')


def cal_path(out_dir: Path, name: str, sd: float, n: int, key: str = 'sd') -> Path:
    """`key` names the noise level in the file name ('sd' part A, 'sigma' part B)."""
    return out_dir / 'calibration' / f'{name}_{key}{sd_tag(sd)}_n{n}.npz'


def trials_path(out_dir: Path, name: str, sd: float, n: int, rep: int, key: str = 'sd') -> Path:
    return out_dir / 'trials' / f'{name}_{key}{sd_tag(sd)}_n{n}_rep{rep:02d}.npz'


def save_npz(path: Path, arrs: dict) -> None:
    """Written to a temporary name first, so an interrupted run never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.savez_compressed(f, **arrs)
    os.replace(tmp, path)


def load_npz(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str) + '\n')


def read_json(path: Path, default=None):
    """`path`'s JSON (`default` if missing), read again after a pause if another process is writing it."""
    for i in range(5):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if i == 4:
                raise
            time.sleep(2.0)


def update_json(path: Path, key: str, value) -> dict:
    """Sets one model's entry of a JSON file keyed by model (noise_levels.json), re-reading it first so
    the entries another process wrote meanwhile (other models into the same folder) are kept; returns
    the whole content."""
    cur = read_json(path, {})
    cur[key] = value
    write_json(path, cur)
    return cur


# provenance.json: entries keyed by model, and the keys of a session's record in previous_sessions
PROV_PER_MODEL = ('bases', 'params', 'model_info', 'rate_targets', 'model_sources', 'levels', 'tuned_fixed_thr')
PROV_SESSION_KEYS = ('generated', 'host', 'models', 'settings', 'n_trials_simulated', 'model')


def write_prov(path: Path, prov: dict) -> None:
    """Writes this session's provenance.json (`prov`, updated in place), keeping what another process
    running other models into the same folder wrote meanwhile: the per-model entries of models not in
    this session (`prov['models']`), and that process's session among previous_sessions."""
    cur = read_json(path)
    if cur:
        mine = set(prov.get('models') or [])
        for k in PROV_PER_MODEL:
            if k in cur or k in prov:
                theirs = cur.get(k) or {}
                prov[k] = dict(theirs, **{m: v for m, v in (prov.get(k) or {}).items() if m in mine or m not in theirs})
        me = (prov.get('generated'), prov.get('host'))
        sessions = list(cur.get('previous_sessions') or []) + list(prov.get('previous_sessions') or [])
        if (cur.get('generated'), cur.get('host')) != me:
            sessions.append({k: cur.get(k) for k in PROV_SESSION_KEYS})
        uniq = {}
        for sess in sessions:
            key = (sess.get('generated'), sess.get('host'))
            if key != me:
                uniq[key] = sess
        if uniq or 'previous_sessions' in prov:
            prov['previous_sessions'] = sorted(uniq.values(), key=lambda x: str(x.get('generated')))
    write_json(path, prov)


def to_arrays(responses: pd.DataFrame, spike_counts: pd.DataFrame):
    """(responses as bool, spike counts as uint16), KCs x odors in ODORS order."""
    r = H.abbrev_columns(responses.astype(float))[ODORS].to_numpy()
    s = H.abbrev_columns(spike_counts.astype(float))[ODORS].to_numpy()
    if not np.isin(r, (0.0, 1.0)).all():
        raise ValueError('responses not 0 / 1')
    if not np.array_equal(s, np.rint(s)) or s.min() < 0 or s.max() >= 2 ** 16:
        raise ValueError('spike counts not non-negative integers < 2^16')
    return r.astype(bool), s.astype(np.uint16)


def pairs_of(mat: np.ndarray) -> pd.Series:
    """Cosine pair distances of a KCs x odors array (ODORS order)."""
    return H.pair_distances(pd.DataFrame(np.asarray(mat, dtype=float), columns=ODORS), 'cosine')


def spearman_odors(a: pd.Series, b: pd.Series) -> float:
    a, b = a.reindex(ODORS), b.reindex(ODORS)
    ok = a.notna() & b.notna()
    return float(sstats.spearmanr(a[ok], b[ok])[0]) if ok.sum() > 2 else np.nan


@contextmanager
def captured_olfsysm_vars(store: dict):
    """Within the block, `fit_mb_model(..., return_olfsysm_vars=True)` also puts its mp / rv in
    `store` (make_fixed_param_kc_simulator keeps them only in its closure)."""
    import al_analysis.mb_model as mbm
    orig = mbm.fit_mb_model

    def wrapper(*args, **kwargs):
        ret = orig(*args, **kwargs)
        if kwargs.get('return_olfsysm_vars'):
            store['mp'], store['rv'] = ret[3]['mp'], ret[3]['rv']
        return ret

    mbm.fit_mb_model = wrapper
    try:
        yield store
    finally:
        mbm.fit_mb_model = orig


# ---------------------------------------------------------------------------
# the data's pipeline, applied to KC data and model trials alike
def summarize(recs: dict, how: str) -> dict:
    """noise_floor_trial_distances' per-recording `analyse`, aggregated as its main() does: floors
    as the mean over odors within a recording, then over recordings; pair distances (raw and
    noise-corrected) as the mean over recordings.

    Args:
        recs: {recording: {odor: trials x KCs array}}.
        how: 'graded' or 'thresholded' (NF.transform).
    """
    floors, pairs = [], []
    for rec, by_odor in recs.items():
        if len(by_odor) < 2:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            po, pr = NF.analyse(by_odor, how)
        po = po.reset_index()
        po.insert(0, 'recording', str(rec))
        floors.append(po)
        pairs.append(pr)
    fl = pd.concat(floors, ignore_index=True)
    by_rec = fl.groupby('recording')
    return dict(
        floor1=float(by_rec.single_trial_floor.mean().mean()),
        floor_mean=float(by_rec.trial_mean_floor.mean().mean()),
        reliability=float(by_rec.reliability.mean().mean()),
        raw=pd.concat([p['raw'] for p in pairs], axis=1).mean(axis=1),
        corrected=pd.concat([p['corrected'] for p in pairs], axis=1).mean(axis=1),
        rec_raw=pd.DataFrame([p['raw'] for p in pairs]).reset_index(drop=True),
        per_odor=fl.groupby('odor')[['single_trial_floor', 'trial_mean_floor', 'reliability']].mean().reindex(ODORS),
        n_recordings=len(pairs),
        # (odor, recording) whose noise estimate reached its trial-mean power (signal power clipped
        # at 1e-12: trial-mean floor 1, corrected distances of its pairs unbounded), and corrected
        # pair distances outside [0, 2] in any recording
        n_no_signal=int((fl.reliability <= 1e-9).sum()),
        n_corrected_out_of_range=int(sum(((p['corrected'] < 0) | (p['corrected'] > 2)).sum() for p in pairs)),
    )


def recordings(arr_list: list, cond: str, how: str):
    """{pseudo-recording: {odor: trials x KCs spike counts}} from trial files, without odors whose
    transformed trials are all zero in that pseudo-recording (their cosine is undefined, and
    `analyse` would divide by zero); also returns how many (recording, odor) were dropped."""
    recs, n_silent = {}, 0
    for i, a in enumerate(arr_list):
        X = a[f'{cond}_spk'].astype(float)
        by_odor = {}
        for j, o in enumerate(ODORS):
            if not NF.transform(X[:, :, j], how).any():
                n_silent += 1
                continue
            by_odor[o] = X[:, :, j]
        recs[i] = by_odor
    return recs, n_silent


def data_side() -> dict:
    """The KC data through `summarize` (both genotypes and transforms), their split-half ρ, and the
    per-odor response fractions (trial mean > 1, mean over recordings)."""
    out = {}
    for g in ('TNT3c', 'TNT'):
        recs = NF.recording_arrays(NF.load_trials(g))
        for how in TRANSFORMS:
            s = summarize(recs, how)
            s['split_half'] = H.split_half_reliability(s['rec_raw'], n_splits=200, seed=0)
            out[(g, how)] = s
        out[(g, 'fractions')] = H.yang_response_fractions(g)[ODORS].mean()
    return out


def trial_set_floors(arrs: dict, cond: str, how: str = 'graded') -> dict:
    recs, n_silent = recordings([arrs], cond, how)
    s = summarize(recs, how)
    return dict(floor1=s['floor1'], floor_mean=s['floor_mean'], n_silent=n_silent)


def pn_stats(arr_list: list, det_pn: dict) -> dict:
    """PN noise in Hz from the trials' PN summaries (odors x glomeruli per trial)."""
    win = [a['pn_win'].astype(float) for a in arr_list]
    det_win = det_pn['pn_win'].astype(float)   # odors x glomeruli (the sd = 0 trial)
    active = det_win >= PN_ACTIVE_HZ
    n_tr = win[0].shape[0]
    tsd = [w.std(axis=0, ddof=1) for w in win] if n_tr > 1 else []
    return dict(
        pn_window_mean_hz_det=float(det_win.mean()),
        pn_window_mean_hz=float(np.mean([w.mean() for w in win])),
        pn_window_trial_sd_hz=float(np.mean(tsd)) if tsd else np.nan,
        pn_window_trial_cv_driven=float(np.mean([(s[active] / det_win[active]).mean() for s in tsd])) if tsd and active.any() else np.nan,
        pn_pre_temporal_sd_hz_det=float(det_pn['pn_pre_sd'].astype(float).mean()),
        pn_pre_temporal_sd_hz=float(np.mean([a['pn_pre_sd'].astype(float).mean() for a in arr_list])),
    )


def fractions(arr_list: list, cond: str, n_spikes_for_response: int) -> dict:
    """Per-odor response fractions of the noisy trials under the four definitions (FRACTION_DEFS),
    each a mean over pseudo-recordings: 'trial' the per-trial response averaged over trials,
    '2of3' a response in at least ceil(2 T / 3) of the T trials, 'count1' trial-mean spike count
    >= 1, 'mean_resp' trial-mean spike count >= the per-trial criterion (2 spikes)."""
    resp = [a[f'{cond}_resp'].astype(float) for a in arr_list]
    spk = [a[f'{cond}_spk'].astype(float) for a in arr_list]
    k = math.ceil(2 * resp[0].shape[0] / 3)
    vals = dict(
        trial=np.concatenate(resp).mean(axis=(0, 1)),
        **{'2of3': np.mean([(r.sum(axis=0) >= k).mean(axis=0) for r in resp], axis=0)},
        count1=np.mean([(s.mean(axis=0) >= 1).mean(axis=0) for s in spk], axis=0),
        mean_resp=np.mean([(s.mean(axis=0) >= n_spikes_for_response).mean(axis=0) for s in spk], axis=0),
    )
    return {d: pd.Series(vals[d], index=ODORS) for d in FRACTION_DEFS}


def onoff_cols(st: dict) -> dict:
    return dict(block=st['block_fruity_small_iso'], block_other_between=st['block_other_between_iso'],
        iso_r2=st['iso_r2'], lin_r2=st['lin_r2'], rho_on_off=st['spearman_on_off'], mean_on=st['mean_on'],
        mean_off=st['mean_off'])


# ---------------------------------------------------------------------------
# the models
def exp_id(experiment: str) -> str:
    """'E45' for 'E45_gain_at_ctrl_frac' (the whole name if it has no E<n>_ prefix)."""
    m = re.match(r'(E\d+)_', experiment)
    return m.group(1) if m else experiment


def fit_row(experiment: str, tag: str) -> pd.Series:
    """Row `tag` of <shared>/results/<experiment>/fits.csv (E10_joint_fit.py's table of fits)."""
    path = H.shared_dir() / 'results' / experiment / 'fits.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path}: no fits.csv for experiment {experiment!r}')
    f = pd.read_csv(path)
    m = f[f['tag'] == tag]
    if len(m) != 1:
        raise ValueError(f'{path}: {len(m)} rows with tag {tag!r} (need exactly one)')
    return m.iloc[0]


def model_from(spec: str) -> str:
    """Registers the model of a `--model-from EXPERIMENT:TAG` spec (MODELS, FIT_MODELS) and returns
    its name, '<E<n>>-<TAG>'."""
    experiment, sep, tag = spec.partition(':')
    if not (sep and experiment and tag):
        raise ValueError(f'--model-from {spec!r}: expected EXPERIMENT:TAG')
    fit_row(experiment, tag)
    name = f'{exp_id(experiment)}-{tag}'
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError(f'--model-from {spec!r}: the model name {name!r} is not file-safe')
    if FIT_MODELS.get(name, (experiment, tag)) != (experiment, tag):
        raise ValueError(f'--model-from {spec!r}: the name {name} already stands for {FIT_MODELS[name]}')
    FIT_MODELS[name] = (experiment, tag)
    MODELS[name] = f'the {experiment} fit at its fitted parameters'
    return name


def model_source(name: str) -> dict:
    """{'fit_experiment', 'fit_tag'} of a model defined by a fits.csv row (empty for base70)."""
    return dict(fit_experiment=FIT_MODELS[name][0], fit_tag=FIT_MODELS[name][1]) if name in FIT_MODELS else {}


def _given(row: pd.Series, col: str):
    """The row's value in `col`, or None when the column is missing or the cell is empty."""
    v = row.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip()):
        return None
    return v


def breadth_cache_name(claw_thr, gain, gloms, wiring, wiring_seed, input_tag):
    """Name of the APL-off run that E10_joint_fit.bouton_base computes (and caches in its out_dir)
    for the breadth; None when it reads the base's saved run (no gain, wiring or other input)."""
    if gain is None and wiring is None and input_tag is None:
        return None
    name = ('' if claw_thr is None else f'thr{claw_thr:g}_') + 'base_acute_off_spike_counts.parquet'
    if input_tag is not None:
        name = f'input-{input_tag}_' + name
    if wiring is not None:
        name = f'wiring-{wiring}-seed{int(wiring_seed)}_' + name
    if gain is not None:
        name = f'gain{gain:g}_' + ('' if sorted(gloms) == sorted(ORACLE) else '-'.join(sorted(gloms)) + '_') + name
    return name


def fit_model(name: str, orn: pd.DataFrame, out_dir: Path) -> dict:
    """The model of a joint fit's fits.csv row (FIT_MODELS[name]) at its fitted parameters, built as
    E10_joint_fit.py built it during the fit (E39 rebuilds it the same way): bouton_base with the
    row's claw_thr, gain, gain glomeruli, re-drawn wiring and ORN input; breadth weights (exponent k)
    from the APL-off run of that wiring at the base's tuned parameters; the fitted scales, KC-KC
    inhibition with tau kc_kc_tau.

    Returns:
        dict: 'orn' (the model's ORN input), 'p_on' (the fitted parameters, wKCKC_scale =
        kc_kc_inh), 'kws' (make_fixed_param_kc_simulator kwargs), 'saved' ({condition: [responses,
        spike counts paths]} of the fit's own outputs, local copy else the shared archive), 'rates'
        (the row's rate_on, rate_off) and 'info'.
    """
    experiment, tag = FIT_MODELS[name]
    fit = fit_row(experiment, tag)
    drive = _given(fit, 'apl_drive') or 'claws'
    if drive != 'claws':
        raise NotImplementedError(f'{experiment}:{tag}: APL driven by {drive!r} (E36 / E37 rows); only fits with '
            'claw-driven APL can be rebuilt here')
    if _given(fit, 'base_params') is not None:
        raise NotImplementedError(f'{experiment}:{tag}: fits from a --base-params base are not supported')
    if name == 'ref40' and sorted(str(fit['gain_glomeruli']).split(',')) != sorted(ORACLE):
        raise ValueError(f"{REF_FIT}: gain glomeruli {fit['gain_glomeruli']} are not the oracle set {ORACLE}")
    claw_thr = None if _given(fit, 'claw_thr') is None else float(fit['claw_thr'])
    k = float(fit['k'])
    gain = None if _given(fit, 'gain') is None or float(fit['gain']) == 1.0 else float(fit['gain'])
    gloms = []
    if gain is not None:
        g = _given(fit, 'gain_glomeruli')
        # (as E10_joint_fit: no glomeruli given = the oracle set of the glomerulus-gain experiment, E27)
        gloms = list(ORACLE) if g is None or sorted(str(g).split(',')) == sorted(ORACLE) else str(g).split(',')
    wiring = _given(fit, 'wiring')
    wiring_seed = int(float(fit['wiring_seed'])) if wiring is not None else 0
    orn_input = _given(fit, 'orn_input')
    input_tag, extra = None, {}
    if wiring is not None:
        extra.update(wiring=str(wiring), wiring_seed=wiring_seed)
    if orn_input is not None:
        orn = read_orn_input(str(orn_input), orn)
        input_tag = Path(str(orn_input)).stem
        p = Path(str(orn_input))
        p = p if p.is_absolute() else H.shared_dir() / p
        extra.update(orn_input=str(orn_input), orn_input_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    tau = float(fit['kc_kc_tau']) if _given(fit, 'kc_kc_tau') is not None else TAU
    p_on = {kk: float(fit[kk]) for kk in PARAM_KEYS}
    p_on['wKCKC_scale'] = float(fit['kc_kc_inh'])
    _, wnorm, W, breadth = bouton_base(orn, out_dir, claw_thr=claw_thr, gain=gain, gain_gloms=gloms or None,
        wiring=None if wiring is None else str(wiring), wiring_seed=wiring_seed, input_tag=input_tag)
    kws = dict(base_kws(claw_thr))
    if gain is not None or wiring is not None:
        # (E10_joint_fit passes the wiring only when it differs from the base's)
        kws['_wPNKC'] = W
    kws.update(_wAPLKC=breadth_weights(wnorm['wAPLKC'], breadth, k), _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'],
        _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=p_on['wKCKC_scale'], kc_kc_tau=tau)
    # the fit's own outputs: this host's copy, else the shared archive
    dirs = (H.HERE / 'outputs' / experiment, H.shared_dir() / 'data' / 'model_outputs' / experiment)
    saved_dir = next((d for d in dirs if (d / f'{tag}_on_responses.parquet').exists()), dirs[0])
    saved = {c: [saved_dir / f'{tag}_{c}_{kind}.parquet' for kind in ('responses', 'spike_counts')] for c in CONDS}
    info = dict(source=f'results/{experiment}/fits.csv row {tag}', claw_thr=claw_thr, k=k, gain=1.0 if gain is None else gain,
        gain_glomeruli=gloms, kc_kc_inh=p_on['wKCKC_scale'], kc_kc_tau=tau, fit_experiment=experiment, fit_tag=tag, **extra)
    # the APL-off run behind the breadth against the fit's
    off_name = breadth_cache_name(claw_thr, gain, gloms, wiring, wiring_seed, input_tag)
    if off_name is not None:
        theirs = next((d / off_name for d in (saved_dir, *dirs) if (d / off_name).exists()), None)
        if theirs is not None and (out_dir / off_name).exists():
            info[f'breadth_run_max_abs_diff_vs_{exp_id(experiment).lower()}'] = float(np.abs(
                pd.read_parquet(theirs).astype(float).to_numpy() - pd.read_parquet(out_dir / off_name).astype(float).to_numpy()).max())
    return dict(orn=orn, p_on=p_on, kws=kws, saved=saved, rates=(float(fit['rate_on']), float(fit['rate_off'])), info=info)


class NoisyModel:
    """One model at fixed parameters, re-run trial by trial with PN noise.

    Builds the model (`make_fixed_param_kc_simulator`, keeping its olfsysm mp / rv), runs the
    deterministic APL-on and acute APL-off outputs (`det`), and gives `trial(sd)` /
    `trials(sd, n)`.
    """

    def __init__(self, name: str, orn: pd.DataFrame, out_dir: Path):
        from al_analysis.mb_model import get_time_index, make_fixed_param_kc_simulator
        self.name = name
        self.info: dict = {}
        # (models from fits.csv rows: the row's rate_on / rate_off, checked against the deterministic run)
        self.fit_rates = None
        if name == 'base70':
            r = e19_row(BASE_CLAW_THR, 0.0, 0.0)
            p_on = {k: float(r[k]) for k in PARAM_KEYS}
            kws = base_kws(BASE_CLAW_THR)
            hal = H.shared_dir() / 'data' / 'hal_outputs' / 'E16_claw_saturation'
            self.saved = {c: [hal / f'bouton_thr{BASE_CLAW_THR:g}_{c}_{kind}.parquet' for kind in ('responses', 'spike_counts')]
                for c in CONDS}
            self.info.update(source='E19 stats row thr70_k0_kckc0 (E16 bouton_thr70, tuned at 5 % with APL on)',
                claw_thr=BASE_CLAW_THR, k=0.0, gain=1.0, kc_kc_inh=0.0)
        elif name in FIT_MODELS:
            # ref40 (E34's thr40_gain1.5_k2_cap3) and the --model-from models
            fm = fit_model(name, orn, out_dir)
            orn, p_on, kws, self.saved, self.fit_rates = fm['orn'], fm['p_on'], fm['kws'], fm['saved'], fm['rates']
            self.info.update(fm['info'])
        else:
            raise ValueError(name)
        self.kws_scalar = {k: v for k, v in kws.items() if not k.startswith('_')}
        store: dict = {}
        with captured_olfsysm_vars(store):
            self.sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p_on, **kws)
        self.mp, self.rv = store['mp'], store['rv']
        self.params = {'on': dict(p_on), 'acute_off': dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0)}
        t = np.asarray(get_time_index(self.mp, self.rv), dtype=float)
        self.win = (t >= self.mp.time_stim_start) & (t < self.mp.time_stim_end)
        self.pre = (t >= self.mp.time_start) & (t < self.mp.time_stim_start)
        inputs = [H.odor_abbrev(c) for c in orn.columns]
        self.pn_idx = [inputs.index(o) for o in ODORS]
        self.n_spikes_for_response = int(self.mp.kc.n_spikes_for_response)
        if float(self.mp.pn.noise.sd) != 0.0 or float(self.mp.pn.noise.mean) != 0.0:
            raise RuntimeError('the model was built with PN noise')
        self.det_frames = {cond: self.sim(p) for cond, p in self.params.items()}
        self.det = {cond: to_arrays(*rs) for cond, rs in self.det_frames.items()}

    def trial(self, sd: float) -> dict:
        """One trial at noise sd: PN layer with noise, then the KC layer with APL on and APL off."""
        import olfsysm as osm
        from al_analysis.mb_model import get_sim_var
        self.mp.pn.noise.sd = float(sd)
        try:
            if float(self.mp.pn.noise.sd) != float(sd):
                raise RuntimeError('could not set mp.pn.noise.sd')
            osm.run_PN_sims(self.mp, self.rv)
            out = {}
            for cond, p in self.params.items():
                out[f'{cond}_resp'], out[f'{cond}_spk'] = to_arrays(*self.sim(p))
            pn = np.asarray(get_sim_var(self.rv.pn, 'pn_sims'), dtype=float)
            if pn.shape[0] != len(self.pn_idx) or pn.shape[2] != len(self.win):
                raise RuntimeError(f'unexpected PN sims shape {pn.shape}')
            pn = pn[self.pn_idx]
            out['pn_win'] = pn[:, :, self.win].mean(axis=2).astype(np.float32)
            out['pn_pre_sd'] = pn[:, :, self.pre].std(axis=2).astype(np.float32)
        finally:
            self.mp.pn.noise.sd = 0.0
        return out

    def trials(self, sd: float, n: int) -> dict:
        trs = [self.trial(sd) for _ in range(n)]
        out = {k: np.stack([tr[k] for tr in trs]) for k in trs[0]}
        out['noise_sd'] = np.asarray(float(sd))
        return out

    def check_sd0(self) -> tuple:
        """A trial at sd = 0 against the deterministic run: mismatching entries per condition."""
        tr = self.trial(0.0)
        res = {cond: dict(responses_mismatch=int((tr[f'{cond}_resp'] != self.det[cond][0]).sum()),
            spike_counts_mismatch=int((tr[f'{cond}_spk'] != self.det[cond][1]).sum())) for cond in CONDS}
        return res, tr

    def close(self) -> None:
        del self.sim, self.mp, self.rv, self.det_frames
        gc.collect()


class Session:
    """One model's simulations in this process; the model is built on first need, with its
    deterministic outputs saved / compared and the sd = 0 checks asserted and recorded."""
    model_cls = NoisyModel

    def __init__(self, name: str, orn: pd.DataFrame, out_dir: Path, checks: dict, checks_path: Path, prov: dict,
            prov_path: Path):
        self.name, self.orn, self.out_dir = name, orn, out_dir
        self.checks, self.checks_path, self.prov, self.prov_path = checks, checks_path, prov, prov_path
        self.model = None
        self.n_noisy = 0
        self.record: dict = {}
        self.sim_seconds = 0.0
        self.n_trials_run = 0

    def get(self) -> NoisyModel:
        if self.model is not None:
            return self.model
        meta_path = self.out_dir / f'{self.name}_meta.json'
        if self.name in FIT_MODELS and meta_path.exists():
            prev = json.loads(meta_path.read_text())
            if 'fit_experiment' in prev and (prev['fit_experiment'], prev.get('fit_tag')) != FIT_MODELS[self.name]:
                raise RuntimeError(f'{meta_path}: {self.name} was {prev["fit_experiment"]}:{prev.get("fit_tag")} here, not '
                    f'{":".join(FIT_MODELS[self.name])}')
        t0 = time.time()
        m = self.model_cls(self.name, self.orn, self.out_dir)
        build_s = time.time() - t0
        rec = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), build_seconds=build_s, **m.info)
        # the deterministic run: saved on first build, compared exactly on later builds
        det_vs_previous = {}
        for cond, (r, s) in m.det_frames.items():
            rp, sp = (self.out_dir / f'{self.name}_{cond}_{kind}.parquet' for kind in ('responses', 'spike_counts'))
            if rp.exists() and sp.exists():
                pr, ps = to_arrays(pd.read_parquet(rp), pd.read_parquet(sp))
                det_vs_previous[cond] = int((pr != m.det[cond][0]).sum() + (ps != m.det[cond][1]).sum())
            else:
                r.to_parquet(rp)
                s.to_parquet(sp)
        rec['det_mismatch_vs_previous_session'] = det_vs_previous
        # against the experiment's own saved outputs (E16 / E34)
        saved = {}
        for cond, (rp, sp) in m.saved.items():
            if rp.exists() and sp.exists():
                pr, ps = to_arrays(pd.read_parquet(rp), pd.read_parquet(sp))
                saved[cond] = dict(file=str(rp), responses_mismatch=int((pr != m.det[cond][0]).sum()),
                    spike_counts_mismatch=int((ps != m.det[cond][1]).sum()))
            else:
                saved[cond] = dict(file=str(rp), missing=True)
        rec['det_vs_saved'] = saved
        missing = [v['file'] for v in saved.values() if v.get('missing')]
        if missing:
            print(f'{self.name}: WARNING no saved outputs to compare the deterministic run with ({", ".join(missing)} not '
                'found); recorded as missing in checks.json', flush=True)
        res0, tr0 = m.check_sd0()
        rec['sd0_trial_vs_det_at_build'] = res0
        rec['rate_det'] = {cond: float(m.det[cond][0].mean()) for cond in CONDS}
        if m.fit_rates is not None:
            rec['rate_det_minus_fits_csv'] = {c: rec['rate_det'][c] - r for c, r in zip(CONDS, m.fit_rates)}
        rec['n_kcs'] = int(m.det['on'][0].shape[0])
        self.record = rec
        self._save_checks()
        print(f"{self.name}: built in {build_s:.0f}s; {rec['n_kcs']} KCs; deterministic response rate on "
            f"{rec['rate_det']['on']:.4f} off {rec['rate_det']['acute_off']:.4f}; vs saved {saved}; sd = 0 trial vs "
            f"deterministic {res0}", flush=True)
        bad = [c for c, v in res0.items() if v['responses_mismatch'] or v['spike_counts_mismatch']]
        assert not bad, f'{self.name}: the sd = 0 trial did not reproduce the deterministic run ({res0})'
        assert not any(det_vs_previous.values()), f'{self.name}: deterministic run differs from the previous session ({det_vs_previous})'
        bad = [c for c, v in saved.items() if not v.get('missing') and (v['responses_mismatch'] or v['spike_counts_mismatch'])]
        assert not bad, f'{self.name}: deterministic run differs from the saved outputs ({saved})'
        # (fits.csv's rates are means of per-odor means: equal up to rounding)
        assert all(abs(d) < 1e-9 for d in rec.get('rate_det_minus_fits_csv', {}).values()), \
            f'{self.name}: deterministic response rates differ from the fits.csv row ({rec["rate_det_minus_fits_csv"]})'
        save_npz(self.out_dir / f'{self.name}_det_pn.npz', {k: v for k, v in tr0.items() if k.startswith('pn_')})
        meta = dict(model=self.name, description=MODELS[self.name], kws=m.kws_scalar, params=m.params,
            n_spikes_for_response=m.n_spikes_for_response, n_kcs=rec['n_kcs'], n_pn_glomeruli=int(tr0['pn_win'].shape[-1]), **m.info)
        write_json(self.out_dir / f'{self.name}_meta.json', meta)
        self.prov.setdefault('bases', {})[self.name] = m.kws_scalar
        self.prov.setdefault('params', {})[self.name] = m.params
        self.prov.setdefault('model_info', {})[self.name] = m.info
        write_prov(self.prov_path, self.prov)
        self.model = m
        return m

    def _save_checks(self) -> None:
        # (re-read first: another process may have added other models' checks meanwhile)
        cur = read_json(self.checks_path, {})
        sessions = cur.setdefault(self.name, [])
        if sessions and sessions[-1].get('generated') == self.record.get('generated'):
            sessions[-1] = self.record
        else:
            sessions.append(self.record)
        write_json(self.checks_path, cur)
        self.checks.clear()
        self.checks.update(cur)

    def ensure(self, path: Path, sd: float, n: int) -> dict:
        """The n trials at noise sd stored at `path`, simulated first if missing."""
        if path.exists():
            return load_npz(path)
        m = self.get()
        t0 = time.time()
        arrs = m.trials(sd, n)
        dt = time.time() - t0
        save_npz(path, arrs)
        self.sim_seconds += dt
        self.n_trials_run += n
        self.n_noisy += n if sd > 0 else 0
        print(f'{self.name}: {n} trials at sd {sd:g} in {dt:.0f}s ({dt / n:.1f}s per trial) -> {path.name}', flush=True)
        return arrs

    def finish(self) -> None:
        if self.model is None:
            return
        if self.n_noisy:
            res, _ = self.model.check_sd0()
            self.record['sd0_trial_vs_det_after_noisy_trials'] = res
            self.record['n_noisy_trials_this_session'] = self.n_noisy
            self.record['seconds_per_trial'] = self.sim_seconds / max(self.n_trials_run, 1)
            self._save_checks()
            print(f'{self.name}: sd = 0 trial after {self.n_noisy} noisy trials vs deterministic {res}', flush=True)
            bad = [c for c, v in res.items() if v['responses_mismatch'] or v['spike_counts_mismatch']]
            assert not bad, f'{self.name}: after the noisy trials, the sd = 0 trial no longer reproduces the deterministic run ({res})'
        self.model.close()
        self.model = None


# ---------------------------------------------------------------------------
# calibration
def interpolate_sd(points: dict, cond: str, target: float) -> dict:
    """sd at which the floor of `cond` crosses `target` (first upward crossing, linear in log sd),
    or the closest point if no pair of points brackets it."""
    sds = sorted(points)
    f = [points[s][cond] for s in sds]
    for i in range(len(sds) - 1):
        if f[i] <= target <= f[i + 1] and f[i + 1] > f[i]:
            x0, x1 = np.log(sds[i]), np.log(sds[i + 1])
            sd = float(np.exp(x0 + (target - f[i]) / (f[i + 1] - f[i]) * (x1 - x0)))
            return dict(sd=r3(sd), bracketed=True, between=[sds[i], sds[i + 1]], target=target)
    j = int(np.argmin(np.abs(np.asarray(f) - target)))
    return dict(sd=sds[j], bracketed=False, between=None, target=target)


def calibrate(sess: Session, grid: list, n_cal: int, max_extend: int, refine: int, targets: dict, det_pn_path: Path):
    """Floors at the grid (extended / refined as needed) and the sds matching the data's floors.

    Returns (calibration rows, {'points', 'matched', 'step', ...})."""
    pts, rows = {}, []
    grid = sorted(r3(g) for g in grid)

    def measure(sd: float, kind: str) -> None:
        sd = r3(sd)
        if sd in pts:
            return
        arrs = sess.ensure(cal_path(sess.out_dir, sess.name, sd, n_cal), sd, n_cal)
        pts[sd] = {}
        pn = pn_stats([arrs], load_npz(det_pn_path)) if det_pn_path.exists() else {}
        for cond in CONDS:
            fg = trial_set_floors(arrs, cond, 'graded')
            ft = trial_set_floors(arrs, cond, 'thresholded')
            pts[sd][cond] = fg['floor1']
            rows.append(dict(model=sess.name, noise_sd=sd, kind=kind, condition=cond, n_trials=n_cal,
                single_trial_floor=fg['floor1'], trial_mean_floor=fg['floor_mean'], n_silent_odors=fg['n_silent'],
                single_trial_floor_thresholded=ft['floor1'], target_single_trial_floor=targets[cond],
                rate_trial=float(arrs[f'{cond}_resp'].mean()), spikes_per_kc_odor=float(arrs[f'{cond}_spk'].astype(float).mean()), **pn))
        print(f"{sess.name} calibration sd {sd:g} ({kind}): single-trial floor APL on {pts[sd]['on']:.3f} "
            f"(control flies {targets['on']:.3f}), APL off {pts[sd]['acute_off']:.3f} (APL output blocked (TNT) "
            f"{targets['acute_off']:.3f})", flush=True)

    def peak(c: str) -> float:
        return max(pts, key=lambda s: pts[s][c])

    for sd in grid:
        measure(sd, 'grid')
    step = grid[-1] / grid[-2] if len(grid) > 1 and grid[-1] > grid[-2] else 2.0
    # The floor is not monotone in sd: it rises from 0, peaks, then falls as heavy noise makes every
    # KC fire on every trial (rate -> 1, trials alike). Extend up only while a floor below its target
    # is still rising at the top of the points; down while the lowest point is above the target.
    for _ in range(max_extend):
        if not any(pts[peak(c)][c] < targets[c] and peak(c) == max(pts) for c in CONDS):
            break
        measure(max(pts) * step, 'extension')
    for _ in range(max_extend):
        if not any(pts[min(pts)][c] > targets[c] for c in CONDS):
            break
        measure(min(pts) / step, 'extension')
    for _ in range(refine):
        new = set()
        for c in CONDS:
            m = interpolate_sd(pts, c, targets[c])
            if m['bracketed']:
                new.add(m['sd'])
            else:   # never reached: locate the floor's peak more closely instead
                new |= {r3(peak(c) * step ** 0.5), r3(peak(c) / step ** 0.5)}
        new = sorted(new - set(pts))
        if not new:
            break
        for sd in new:
            measure(sd, 'refine')
    matched = {c: interpolate_sd(pts, c, targets[c]) for c in CONDS}
    peaks = {c: dict(sd=peak(c), floor=pts[peak(c)][c]) for c in CONDS}
    for c, m in matched.items():
        if not m['bracketed']:
            print(f'{sess.name}: WARNING the {c} single-trial floor never reached {targets[c]:.3f} over sd {min(pts):g}-{max(pts):g} '
                f'(largest {peaks[c]["floor"]:.3f} at sd {peaks[c]["sd"]:g}); using the closest point, sd {m["sd"]:g}', flush=True)
    return rows, dict(grid=grid, step=step, n_cal_trials=n_cal, points={str(k): v for k, v in sorted(pts.items())}, matched=matched,
        peaks=peaks)


def make_levels(matched: dict, peaks: dict, bracket) -> dict:
    """label -> sd: the matched sds first, then `bracket[0]` x the smaller and `bracket[1]` x the
    larger; when the larger bracket lies beyond the sd of either floor's peak (where more noise only
    makes trials more alike), `bracket[0]`^2 x the smaller instead ('lo2'). Labels are joined with
    '+' where two levels coincide."""
    m_on, m_off = matched['on']['sd'], matched['acute_off']['sd']
    lv: dict = {}

    def add(label, sd):
        sd = r3(sd)
        for k, v in list(lv.items()):
            if v == sd:
                lv[f'{k}+{label}'] = lv.pop(k)
                return
        lv[label] = sd

    add('match_on', m_on)
    add('match_off', m_off)
    add('lo', bracket[0] * min(m_on, m_off))
    hi = bracket[1] * max(m_on, m_off)
    if hi <= min(p['sd'] for p in peaks.values()):
        add('hi', hi)
    else:
        add('lo2', bracket[0] ** 2 * min(m_on, m_off))
    return lv


# ---------------------------------------------------------------------------
# analysis
def load_det(out_dir: Path, name: str) -> dict:
    return {cond: tuple(a.astype(float) for a in to_arrays(*(pd.read_parquet(out_dir / f'{name}_{cond}_{kind}.parquet')
        for kind in ('responses', 'spike_counts')))) for cond in CONDS}


def merge_kept(new: pd.DataFrame, path: Path, analysed: set) -> pd.DataFrame:
    """`new` plus the rows of the existing table at `path` of models not analysed in this call (their
    trial files are on another host), so an analysis on one host never drops another host's models."""
    if not path.exists():
        return new
    # (round_trip: the kept rows are written back unchanged)
    old = pd.read_csv(path, float_precision='round_trip')
    if 'model' not in old.columns:
        return new
    keep = old[~old['model'].isin(set(analysed) | {'data'})]
    if not len(keep):
        return new
    print(f'{path.name}: rows of {sorted(keep["model"].unique())} kept from the existing table (no trial files of theirs '
        'analysed here)', flush=True)
    return pd.concat([new, keep], ignore_index=True) if len(new) else keep.reset_index(drop=True)


def analyse_all(levels_all: dict, n_rec: int, n_tr: int, out_dir: Path, res_dir: Path, data: dict, key: str = 'sd',
        ncol: str = 'noise_sd', extra=None) -> pd.DataFrame:
    """The readouts of every model x level (tables written to `res_dir`; models without trial files
    here keep their rows of the existing tables, `merge_kept`).

    Args:
        key: the noise level's name in the trial files (`trials_path`).
        ncol: the noise level's column in the tables ('noise_sd' part A, 'sigma' part B).
        extra: optional function (model, level label, level, condition, trial arrays) -> dict of
            columns added to (or replacing) each stats row (part B: the re-tuned threshold).
    """
    stats, odor_rows, pair_rows, onoff_rows = [], [], [], []
    for how in TRANSFORMS:
        for kind in ('raw', 'corrected'):
            st, _ = H.compare_on_off(data[('TNT3c', how)][kind], data[('TNT', how)][kind])
            onoff_rows.append(dict(model='data', level='-', **{ncol: np.nan}, transform=how, matrix=kind, **onoff_cols(st)))
    analysed = set()
    for name, lv in levels_all.items():
        meta_path = out_dir / f'{name}_meta.json'
        if not meta_path.exists():
            print(f'{name}: no {meta_path.name}; skipped in the analysis', flush=True)
            continue
        meta = json.loads(meta_path.read_text())
        det = load_det(out_dir, name)
        det_pn = load_npz(out_dir / f'{name}_det_pn.npz')
        det_pairs = {(c, how): pairs_of(NF.transform(det[c][1], how)) for c in CONDS for how in TRANSFORMS}
        # (the deterministic rows go in only if the model has trial files here)
        det_onoff, at, n_stats = [], len(onoff_rows), len(stats)
        for how in TRANSFORMS:
            st, _ = H.compare_on_off(det_pairs[('on', how)], det_pairs[('acute_off', how)])
            det_onoff.append(dict(model=name, level='deterministic', **{ncol: 0.0}, transform=how, matrix='deterministic',
                **onoff_cols(st)))
        for label, sd in lv['levels'].items():
            arrs = [load_npz(p) for p in (trials_path(out_dir, name, sd, n_tr, r, key) for r in range(n_rec)) if p.exists()]
            if not arrs:
                continue
            pn = pn_stats(arrs, det_pn)
            mats = {}
            for cond in CONDS:
                g = GENOTYPE[cond]
                matched_to = [c for c in CONDS if lv['matched'][c]['sd'] == sd]
                spk_all = np.concatenate([a[f'{cond}_spk'] for a in arrs]).astype(float)
                fr = fractions(arrs, cond, int(meta['n_spikes_for_response']))
                det_fr = pd.Series(det[cond][0].mean(axis=0), index=ODORS)
                data_fr = data[(g, 'fractions')]
                rc = dict(rate_det=float(det_fr.mean()), data_rate=float(data_fr.mean()), weak_frac_det=float(det_fr[WEAK].mean()),
                    data_weak_frac=float(data_fr[WEAK].mean()), rate_rho_vs_data_det=spearman_odors(det_fr, data_fr))
                for d, f in fr.items():
                    rc[f'rate_{d}'] = float(f.mean())
                    rc[f'weak_frac_{d}'] = float(f[WEAK].mean())
                    rc[f'rate_rho_vs_data_{d}'] = spearman_odors(f, data_fr)
                rc['rate_bias_trial'] = rc['rate_trial'] - rc['rate_det']
                rc['rate_rho_trial_vs_det'] = spearman_odors(fr['trial'], det_fr)
                rc['spikes_per_kc_odor_det'] = float(det[cond][1].mean())
                rc['spikes_per_kc_odor'] = float(spk_all.mean())
                rc['spike_ratio_noisy_over_det'] = rc['spikes_per_kc_odor'] / max(rc['spikes_per_kc_odor_det'], 1e-12)
                params = meta['params'][cond]
                for how in TRANSFORMS:
                    recs, n_silent = recordings(arrs, cond, how)
                    s = summarize(recs, how)
                    d = data[(g, how)]
                    grand = np.mean([NF.transform(x, how) for x in spk_all], axis=0)
                    grand_p, det_p = pairs_of(grand), det_pairs[(cond, how)]
                    mats[(cond, how)] = dict(raw=s['raw'], corrected=s['corrected'], grand=grand_p)
                    sh = H.split_half_reliability(s['rec_raw'], n_splits=200, seed=0) if s['n_recordings'] >= 4 else {}
                    raw_a, det_a = H.align_pairs(s['raw'], det_p)
                    cor_a, det_b = H.align_pairs(s['corrected'], det_p)
                    row = dict(tag=f'{name}__{label}__{cond}__{how}', variant=f'{name}__{label}', model=name, base='bouton',
                        fit_experiment=meta.get('fit_experiment'), fit_tag=meta.get('fit_tag'),
                        level=label, **{ncol: float(sd)}, matched_to='+'.join(matched_to), condition=cond, transform=how,
                        n_recordings=s['n_recordings'], n_trials=int(arrs[0][f'{cond}_spk'].shape[0]),
                        n_silent_odor_recordings=n_silent, n_no_signal_odor_recordings=s['n_no_signal'],
                        n_corrected_pairs_out_of_range=s['n_corrected_out_of_range'],
                        claw_thr=meta.get('claw_thr'), k=meta.get('k'), gain=meta.get('gain'), kc_kc_tau=meta.get('kc_kc_tau', np.nan),
                        kc_kc_inh=params.get('wKCKC_scale', 0.0), **{kk: params[kk] for kk in PARAM_KEYS},
                        single_trial_floor=s['floor1'], trial_mean_floor=s['floor_mean'], reliability=s['reliability'],
                        data_single_trial_floor=d['floor1'], data_trial_mean_floor=d['floor_mean'], data_reliability=d['reliability'],
                        mean_dist_raw=float(s['raw'].mean()), mean_dist_corrected=float(s['corrected'].mean()),
                        mean_dist_det=float(det_p.mean()), mean_dist_grand=float(grand_p.mean()),
                        data_mean_dist_raw=float(d['raw'].mean()), data_mean_dist_corrected=float(d['corrected'].mean()),
                        raw_minus_det=float(s['raw'].mean() - det_p.mean()), corrected_minus_det=float(s['corrected'].mean() - det_p.mean()),
                        grand_minus_det=float(grand_p.mean() - det_p.mean()),
                        pair_mae_raw_vs_det=float((raw_a - det_a).abs().mean()), pair_mae_corrected_vs_det=float((cor_a - det_b).abs().mean()),
                        rho_raw_vs_data=H.spearman(s['raw'], d['raw']), rho_corrected_vs_data=H.spearman(s['corrected'], d['raw']),
                        rho_corrected_vs_data_corrected=H.spearman(s['corrected'], d['corrected']),
                        rho_grand_vs_data=H.spearman(grand_p, d['raw']),
                        rho_det_vs_data=H.spearman(det_p, d['raw']), rho_det_vs_data_corrected=H.spearman(det_p, d['corrected']),
                        rho_raw_vs_det=H.spearman(s['raw'], det_p), rho_corrected_vs_det=H.spearman(s['corrected'], det_p),
                        rho_grand_vs_det=H.spearman(grand_p, det_p),
                        model_split_half_rho=sh.get('split_half', np.nan), model_ceiling=sh.get('ceiling', np.nan),
                        data_split_half_rho=d['split_half']['split_half'], data_ceiling=d['split_half']['ceiling'],
                        **rc, **pn)
                    if extra is not None:
                        row.update(extra(name, label, sd, cond, arrs))
                    stats.append(row)
                    pdf = pd.DataFrame({'raw': s['raw'], 'corrected': s['corrected'], 'det': det_p, 'grand': grand_p,
                        'data_raw': d['raw'], 'data_corrected': d['corrected']})
                    pdf.index.names = ['odor_a', 'odor_b']
                    pdf = pdf.reset_index()
                    for col, val in (('transform', how), ('condition', cond), (ncol, float(sd)), ('level', label), ('model', name)):
                        pdf.insert(0, col, val)
                    pair_rows.append(pdf)
                    if how == 'graded':
                        po, dpo = s['per_odor'], d['per_odor']
                        det_spk_o = det[cond][1].mean(axis=0)
                        spk_o = spk_all.mean(axis=(0, 1))
                        for j, o in enumerate(ODORS):
                            odor_rows.append(dict(model=name, level=label, **{ncol: float(sd)}, condition=cond, odor=o, weak=o in WEAK,
                                det_frac=float(det_fr[o]), **{f'frac_{dd}': float(fr[dd][o]) for dd in FRACTION_DEFS},
                                data_frac=float(data_fr[o]), single_trial_floor=float(po.at[o, 'single_trial_floor']),
                                trial_mean_floor=float(po.at[o, 'trial_mean_floor']),
                                data_single_trial_floor=float(dpo.at[o, 'single_trial_floor']),
                                data_trial_mean_floor=float(dpo.at[o, 'trial_mean_floor']),
                                spikes_per_kc_det=float(det_spk_o[j]), spikes_per_kc=float(spk_o[j])))
            for how in TRANSFORMS:
                for kind in ('raw', 'corrected', 'grand'):
                    st, _ = H.compare_on_off(mats[('on', how)][kind], mats[('acute_off', how)][kind])
                    onoff_rows.append(dict(model=name, level=label, **{ncol: float(sd)}, transform=how, matrix=kind, **onoff_cols(st)))
        if len(stats) > n_stats:
            analysed.add(name)
            onoff_rows[at:at] = det_onoff
    st = merge_kept(pd.DataFrame(stats), res_dir / 'stats_noise.csv', analysed)
    st.to_csv(res_dir / 'stats_noise.csv', index=False)
    merge_kept(pd.DataFrame(onoff_rows), res_dir / 'onoff_noise.csv', analysed).to_csv(res_dir / 'onoff_noise.csv', index=False)
    for rows, fname in ((pd.DataFrame(odor_rows), 'per_odor_noise.csv'),
            (pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame(), 'pairs_noise_long.csv')):
        rows = merge_kept(rows, res_dir / fname, analysed)
        if len(rows):
            rows.to_csv(res_dir / fname, index=False)
    return st


def print_summary(st: pd.DataFrame, res_dir: Path, ncol: str = 'noise_sd', extra_cols: tuple = ()) -> None:
    if not len(st):
        return
    g = st[st['transform'] == 'graded'].copy()
    cols = ['model', 'level', ncol, *extra_cols, 'condition', 'n_recordings', 'single_trial_floor', 'data_single_trial_floor',
        'trial_mean_floor', 'data_trial_mean_floor', 'mean_dist_raw', 'data_mean_dist_raw', 'mean_dist_corrected',
        'data_mean_dist_corrected', 'mean_dist_det', 'rho_raw_vs_data', 'rho_det_vs_data', 'model_ceiling', 'data_ceiling',
        'rate_det', 'rate_trial', 'rate_2of3', 'rate_count1', 'rate_mean_resp', 'data_rate', 'weak_frac_det', 'weak_frac_trial',
        'pn_pre_temporal_sd_hz', 'pn_window_trial_sd_hz']
    pd.set_option('display.width', 250)
    print(g[[c for c in cols if c in g]].round(3).to_string(index=False))
    oo = pd.read_csv(res_dir / 'onoff_noise.csv')
    print(oo[oo['transform'] == 'graded'].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
# figure
COLORS = {'on': '#2a78d6', 'acute_off': '#eb6834'}
COND_LABEL = {'on': 'APL on', 'acute_off': 'APL off'}
DATA_LABEL = {'on': 'control flies', 'acute_off': 'APL output blocked (TNT)'}
KIND_COLORS = {'raw': '#1baf7a', 'corrected': '#6250d6', 'grand': '#eda100'}


def make_figure(res_dir: Path, ncol: str = 'noise_sd', xlabel: str = 'PN noise sd (mp.pn.noise.sd)', suptitle=None) -> None:
    """The summary figure from the tables in `res_dir` (`ncol`: the noise level's column;
    `suptitle`: a format string with {n_rec} / {n_tr}, part A's title by default)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    st = pd.read_csv(res_dir / 'stats_noise.csv')
    cal = pd.read_csv(res_dir / 'calibration.csv') if (res_dir / 'calibration.csv').exists() else pd.DataFrame()
    oo = pd.read_csv(res_dir / 'onoff_noise.csv')
    lv = json.loads((res_dir / 'noise_levels.json').read_text())
    g = st[st['transform'] == 'graded']
    # (models registered in this process first, then any other model of the tables, e.g. from --model-from in another run)
    in_table = list(dict.fromkeys(g.model))
    models = [m for m in MODELS if m in in_table] + [m for m in in_table if m not in MODELS]
    if not models:
        return
    desc = {m: MODELS.get(m) or lv.get(m, {}).get('description', '') for m in models}
    fig, axes = plt.subplots(len(models), 7, figsize=(31, 4.8 * len(models)), squeeze=False)

    def first(s, col):
        return float(s[col].iloc[0]) if len(s) else np.nan

    for i, m in enumerate(models):
        ax = axes[i]
        gm = g[g.model == m]
        per = {c: gm[gm.condition == c].sort_values(ncol) for c in CONDS}
        # 1: calibration
        a = ax[0]
        for c in CONDS:
            if len(cal):
                cc = cal[(cal.model == m) & (cal.condition == c)].sort_values(ncol)
                a.plot(cc[ncol], cc.single_trial_floor, 'o-', color=COLORS[c], ms=4, lw=1, label=f'{COND_LABEL[c]}, calibration trials')
            a.plot(per[c][ncol], per[c].single_trial_floor, 'D', color=COLORS[c], ms=8, mfc='none', mew=1.5,
                label=f'{COND_LABEL[c]}, pseudo-recordings')
            a.axhline(first(per[c], 'data_single_trial_floor'), color=COLORS[c], ls=':', lw=1.2, label=f'KCs, {DATA_LABEL[c]}')
            if m in lv:
                a.axvline(lv[m]['matched'][c]['sd'], color=COLORS[c], lw=0.8, alpha=0.5)
        a.set_title('same-odor distance between single trials (floor)\n(dotted: KC data; vertical: matched noise)', fontsize=9)
        a.set_ylabel(f'{m}:\n{desc[m]}', fontsize=8)
        # 2: mean pair distance
        a = ax[1]
        for c in CONDS:
            a.plot(per[c][ncol], per[c].mean_dist_raw, 'o-', color=COLORS[c], ms=5, lw=1.3, label=f'{COND_LABEL[c]}, 3-trial means')
            a.plot(per[c][ncol], per[c].mean_dist_corrected, 's--', color=COLORS[c], ms=5, lw=1.3, label=f'{COND_LABEL[c]}, noise-corrected')
            a.axhline(first(per[c], 'mean_dist_det'), color=COLORS[c], lw=1, alpha=0.6, label=f'{COND_LABEL[c]}, deterministic')
            a.axhline(first(per[c], 'data_mean_dist_raw'), color='0.45', ls=':', lw=1)
            a.axhline(first(per[c], 'data_mean_dist_corrected'), color='0.45', ls='-.', lw=0.8)
        a.set_title('mean pair distance (1 - cosine)\n(KC data: dotted raw {:.2f} / {:.2f}, dash-dot noise-corrected {:.2f} / {:.2f})'.format(
            first(per['on'], 'data_mean_dist_raw'), first(per['acute_off'], 'data_mean_dist_raw'),
            first(per['on'], 'data_mean_dist_corrected'), first(per['acute_off'], 'data_mean_dist_corrected')), fontsize=9)
        # 3: rho with the KC data
        a = ax[2]
        for c in CONDS:
            a.plot(per[c][ncol], per[c].rho_raw_vs_data, 'o-', color=COLORS[c], ms=5, lw=1.3, label=f'{COND_LABEL[c]}, 3-trial means')
            a.plot(per[c][ncol], per[c].rho_corrected_vs_data_corrected, 's--', color=COLORS[c], ms=5, lw=1.3,
                label=f'{COND_LABEL[c]}, noise-corrected (vs corrected KC data)')
            a.axhline(first(per[c], 'rho_det_vs_data'), color=COLORS[c], lw=1, alpha=0.6, label=f'{COND_LABEL[c]}, deterministic')
        a.set_title('ρ on odor-odor distances with the KC data\n(measurement ceilings {:.2f} APL on / {:.2f} APL off)'.format(
            first(per['on'], 'data_ceiling'), first(per['acute_off'], 'data_ceiling')), fontsize=9)
        # 4: response rate
        a = ax[3]
        for c in CONDS:
            for d, sty, lab in (('trial', 'o-', 'per trial'), ('2of3', 's--', 'in 2 of 3 trials'), ('count1', '^:', 'trial-mean count >= 1')):
                a.plot(per[c][ncol], per[c][f'rate_{d}'], sty, color=COLORS[c], ms=5, lw=1.2, label=f'{COND_LABEL[c]}, {lab}')
            a.axhline(first(per[c], 'rate_det'), color=COLORS[c], lw=1, alpha=0.6, label=f'{COND_LABEL[c]}, deterministic')
            a.axhline(first(per[c], 'data_rate'), color='0.45', ls=':', lw=1)
        a.set_title('response rate (mean over the 16 odors)\n(dotted: KC data {:.3f} / {:.3f})'.format(
            first(per['on'], 'data_rate'), first(per['acute_off'], 'data_rate')), fontsize=9)
        # 5: weak odors
        a = ax[4]
        for c in CONDS:
            a.plot(per[c][ncol], per[c].weak_frac_trial, 'o-', color=COLORS[c], ms=5, lw=1.3, label=f'{COND_LABEL[c]}, per trial')
            a.axhline(first(per[c], 'weak_frac_det'), color=COLORS[c], lw=1, alpha=0.6, label=f'{COND_LABEL[c]}, deterministic')
            a.axhline(first(per[c], 'data_weak_frac'), color='0.45', ls=':', lw=1)
        a.set_title('weak odors\' mean response fraction ({})\n(dotted: KC data {:.3f} / {:.3f})'.format(
            ', '.join(WEAK), first(per['on'], 'data_weak_frac'), first(per['acute_off'], 'data_weak_frac')), fontsize=9)
        # 6, 7: block and isotonic R2 between the APL-on and APL-off matrices
        og = oo[oo['transform'] == 'graded']
        dat = og[og.model == 'data'].set_index('matrix')
        detm = og[(og.model == m) & (og.matrix == 'deterministic')]
        for a, col, title in ((ax[5], 'block', 'fruity x small-ester block'), (ax[6], 'iso_r2', 'isotonic R² of APL on on APL off')):
            for kind, sty, lab in (('raw', 'o-', '3-trial means'), ('corrected', 's--', 'noise-corrected'), ('grand', '^:', 'grand mean of all trials')):
                s = og[(og.model == m) & (og.matrix == kind)].sort_values(ncol)
                a.plot(s[ncol], s[col], sty, color=KIND_COLORS[kind], ms=5, lw=1.2, label=lab)
            if len(detm):
                a.axhline(float(detm[col].iloc[0]), color='0.2', lw=1, label='deterministic')
            a.axhline(float(dat.at['raw', col]), color='0.45', ls=':', lw=1)
            a.axhline(float(dat.at['corrected', col]), color='0.45', ls='-.', lw=0.8)
            a.set_title(f'{title}\n(KC data: dotted raw {dat.at["raw", col]:.3f}, dash-dot corrected {dat.at["corrected", col]:.3f})', fontsize=9)
        for a in ax:
            a.set_xscale('log')
            lo, hi = a.get_xlim()
            if 0 < lo and hi / lo < 2:
                # (one or two close levels, e.g. a --levels match_on run: a readable range around them)
                a.set_xlim(np.sqrt(lo * hi) / 1.6, np.sqrt(lo * hi) * 1.6)
            a.set_xlabel(xlabel, fontsize=8)
            a.legend(fontsize=6, frameon=False)
            a.spines[['top', 'right']].set_visible(False)
            a.grid(True, color='0.9', lw=0.6)
    n_rec, n_tr = int(g.n_recordings.max()), int(g.n_trials.max())
    if suptitle is None:
        suptitle = (f'E43: trial-to-trial variability in the model (Gaussian noise on the PN rate; {n_rec} pseudo-recordings of {n_tr} '
            'trials per noise level; graded spike counts through the KC data\'s pipeline; APL on compared with control flies, '
            'APL off with APL output blocked (TNT))')
    else:
        suptitle = suptitle.format(n_rec=n_rec, n_tr=n_tr)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(res_dir / 'E43_model_trial_noise.png', dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# parts B and C: trial-level noise at a re-tuned operating point. B: per-trial jitter of each KC's
# somatic threshold. C: per-trial gain jitter of each glomerulus x odor ORN input.
B_KEY = 'sigma'
MODEL_SEED_IDX = {'base70': 0, 'ref40': 1}
DRAW_KINDS = {'tune': 0, 'rec': 1, 'check': 2, 'tune_c': 3, 'rec_c': 4, 'check_c': 5}
THR_NOISE_NOTE = ('Part B: per trial, each KC\'s somatic threshold (rv.kc.thr, compared with Vm at every step; '
    'fixed_thr + 2 * kc.spont_in, as fit_mb_model / make_fixed_param_kc_simulator set it) is replaced by '
    'thr0 * (1 + sigma * xi), xi standard normal per KC and trial, clipped below at thr_min_frac * thr0 (> 0); the same '
    'draw for the APL-on and the APL-off run of a trial; PN sims unchanged (deterministic, no PN noise); only '
    'run_KC_sims(regen=False) per trial. At each sigma fixed_thr is re-tuned (bracketing bisection on the tuning draws) so '
    'the per-trial APL-on response fraction, averaged over trials, equals the deterministic run\'s; APL weights fixed. '
    'Draws: numpy default_rng(SeedSequence(seed, spawn_key=(model index, kind, rep))), kind 0 tuning, 1 pseudo-recording '
    'rep, 2 the sigma = 0 check; the same draws at every sigma (common random numbers across levels).')
GAIN_NOISE_NOTE = ('Part C: per trial, each glomerulus x odor ORN input delta (mp.orn.data.delta, the ORN deltas of the '
    'model input) is scaled by max(1 + sigma * eta, 0), eta standard normal per glomerulus, odor and trial (each odor '
    'presentation is its own trial), constant over the trial; then run_ORN_LN_sims, run_PN_sims and the KC layer with APL '
    'on and with APL off on the same input. ORN spontaneous rates and KC thresholds unjittered (rv.kc.thr = fixed_thr + 2 * '
    'spont_in), fixed_thr re-tuned at each sigma as in part B; no PN-rate noise. Draws as part B, with kinds 3 (tuning), 4 '
    '(pseudo-recording rep) and 5 (the sigma = 0 check).')


def seed_index(name: str) -> int:
    """The model's index in the draws' spawn key: base70 0, ref40 1, and every other model defined by a
    fits.csv row (--model-from) ref40's, so all those models get the same draws (common random numbers)."""
    if name in MODEL_SEED_IDX:
        return MODEL_SEED_IDX[name]
    if name in FIT_MODELS:
        return MODEL_SEED_IDX['ref40']
    raise KeyError(name)


def b_draws(seed: int, name: str, kind: str, rep: int, n: int, size: int) -> np.ndarray:
    """n x size standard normal draws for one tuning set or pseudo-recording (reproducible, and the
    first trials' draws do not depend on n)."""
    ss = np.random.SeedSequence(int(seed), spawn_key=(seed_index(name), DRAW_KINDS[kind], int(rep)))
    return np.random.default_rng(ss).standard_normal((int(n), int(size)))


@contextmanager
def kc_threshold_override(thr: np.ndarray):
    """Within the block, a `make_fixed_param_kc_simulator` simulate call runs the KC layer with
    rv.kc.thr = `thr` (simulate sets thr from fixed_thr just before `osm.run_KC_sims`, which is
    looked up in mb_model's namespace, so a proxy there sets `thr` last; the APL / KC-KC weights
    are still set by simulate)."""
    import al_analysis.mb_model as mbm
    real = mbm.osm

    class _Osm:
        def __getattr__(self, k):
            return getattr(real, k)

        @staticmethod
        def run_KC_sims(mp, rv, regen):
            rv.kc.thr = thr
            return real.run_KC_sims(mp, rv, regen)

    mbm.osm = _Osm()
    try:
        yield
    finally:
        mbm.osm = real


class ExcitabilityModel(NoisyModel):
    """One model at fixed APL weights, re-run trial by trial with each KC's threshold jittered.

    rv.kc.thr = thr0 = fixed_thr + 2 * spont_in (KCs x 1); a trial uses
    max(thr0 * (1 + sigma * xi), thr_min_frac * thr0) with one xi per KC, for the APL-on and the
    APL-off run alike. `fixed_thr` (the mean threshold's free part) is the deterministic run's
    unless given (part B re-tunes it per sigma).
    """
    thr_min_frac = 0.01
    thr_jitter = 'normal'
    draw_kinds = dict(tune='tune', rec='rec', check='check')

    def __init__(self, name: str, orn: pd.DataFrame, out_dir: Path):
        super().__init__(name, orn, out_dir)
        if int(self.mp.kc.n_claws_active_to_spike) >= 0:
            raise NotImplementedError('thresholds are per claw (n_claws_active_to_spike >= 0)')
        self.n_kcs = self.draw_size = int(self.det['on'][0].shape[0])
        self.spont_in = np.array(self.rv.kc.spont_in, dtype=float, copy=True)
        if self.spont_in.size != self.n_kcs:
            raise RuntimeError(f'spont_in has {self.spont_in.size} entries for {self.n_kcs} KCs')
        self.f0 = float(self.params['on']['fixed_thr'])
        thr_now = np.array(self.rv.kc.thr, dtype=float)
        if thr_now.shape != self.spont_in.shape or not np.array_equal(thr_now, self.thr0(self.f0)):
            raise RuntimeError('rv.kc.thr is not fixed_thr + 2 * spont_in')
        if not (self.thr0(self.f0) > 0).all():
            raise RuntimeError('non-positive deterministic thresholds')
        self.pn_summary = self._pn_now()
        t0 = self.thr0(self.f0)
        self.info.update(thr_layout=f'rv.kc.thr {thr_now.shape} = fixed_thr + 2 * spont_in', fixed_thr_det=self.f0,
            thr0_mean_det=float(t0.mean()), thr0_min_det=float(t0.min()), thr0_max_det=float(t0.max()),
            spont_in_mean=float(self.spont_in.mean()), spont_in_max=float(self.spont_in.max()),
            frac_kcs_spont_in_positive=float((self.spont_in > 0).mean()), thr_min_frac=self.thr_min_frac)

    def _pn_now(self) -> dict:
        """Odor-window mean and pre-odor temporal SD of the PN rates now in rv (odors x glomeruli)."""
        from al_analysis.mb_model import get_sim_var
        pn = np.asarray(get_sim_var(self.rv.pn, 'pn_sims'), dtype=float)[self.pn_idx]
        return dict(pn_win=pn[:, :, self.win].mean(axis=2).astype(np.float32),
            pn_pre_sd=pn[:, :, self.pre].std(axis=2).astype(np.float32))

    def thr0(self, fixed_thr: float) -> np.ndarray:
        # (same expression as make_fixed_param_kc_simulator's simulate, so sigma = 0 is exact)
        return fixed_thr + self.spont_in * 2.0

    def thr_trial(self, fixed_thr: float, sigma: float, xi: np.ndarray):
        """(jittered thresholds, how many were clipped, the factors thr / thr0). 'normal':
        thr0 * (1 + sigma * xi) clipped below at thr_min_frac * thr0; 'lognormal': thr0 *
        exp(sigma * xi - sigma^2 / 2) (mean thr0, never clipped)."""
        t0 = self.thr0(fixed_thr)
        xi = np.asarray(xi, dtype=float).reshape(t0.shape)
        if self.thr_jitter == 'lognormal':
            t = t0 * np.exp(sigma * xi - 0.5 * sigma ** 2)
        elif self.thr_jitter == 'normal':
            t = t0 * (1.0 + sigma * xi)
        else:
            raise ValueError(f'{self.thr_jitter=}')
        lo = self.thr_min_frac * t0
        clipped = t < lo
        t = np.where(clipped, lo, t)
        return t, int(clipped.sum()), (t / t0).ravel()

    def run_kc(self, cond: str, fixed_thr: float, thr: np.ndarray):
        with kc_threshold_override(thr):
            r, s = self.sim(dict(self.params[cond], fixed_thr=float(fixed_thr)))
        if not np.array_equal(np.asarray(self.rv.kc.thr), thr):
            raise RuntimeError('rv.kc.thr changed during the KC simulation')
        return to_arrays(r, s)

    def trial(self, sd: float, xi=None, fixed_thr=None, conds=CONDS) -> dict:
        """One trial at jitter `sd` (sigma) with draws `xi` (zeros if None) and `fixed_thr` (the
        deterministic run's if None)."""
        f = self.f0 if fixed_thr is None else float(fixed_thr)
        thr, n_clip, factor = self.thr_trial(f, float(sd), np.zeros(self.n_kcs) if xi is None else xi)
        out = {}
        for cond in conds:
            out[f'{cond}_resp'], out[f'{cond}_spk'] = self.run_kc(cond, f, thr)
        out.update(self.pn_summary)
        out['n_clipped'] = np.asarray(n_clip)
        out['factor'] = factor.astype(np.float32)
        return out

    def trials_b(self, sigma: float, xis: np.ndarray, fixed_thr: float, conds=CONDS) -> dict:
        trs = [self.trial(sigma, xi, fixed_thr, conds) for xi in xis]
        out = {k: np.stack([tr[k] for tr in trs]) for k in trs[0]}
        out.update(sigma=np.asarray(float(sigma)), fixed_thr=np.asarray(float(fixed_thr)),
            n_draws_per_trial=np.asarray(self.draw_size))
        return out

    def check_sd0(self) -> tuple:
        """A sigma = 0 trial (through the jitter path, with a real draw) against the deterministic run."""
        xi = b_draws(0, self.name, self.draw_kinds['check'], 0, 1, self.draw_size)[0]
        tr = self.trial(0.0, xi, self.f0)
        res = {cond: dict(responses_mismatch=int((tr[f'{cond}_resp'] != self.det[cond][0]).sum()),
            spike_counts_mismatch=int((tr[f'{cond}_spk'] != self.det[cond][1]).sum())) for cond in CONDS}
        return res, tr

    def tune(self, sigma: float, xis: np.ndarray, target: float, guess=None, tol: float = 0.01, max_evals: int = 16,
            log=None) -> dict:
        """fixed_thr at which the APL-on response fraction of the trials with draws `xis`, averaged
        over trials, is `target` within relative tolerance `tol`.

        The fraction falls with fixed_thr (same draws at every evaluation), so: step from `guess`
        (multiplicatively, doubling the step) until the target is bracketed, then split the
        bracket (at the point where log fraction interpolates to the target, kept within the
        middle 80 % of the bracket; the midpoint when that is undefined) until within `tol`, the
        bracket is < 1e-6 relative, or `max_evals` evaluations. Returns the evaluated point closest
        to the target, with its APL-on trial arrays and the history.
        """
        evals: dict = {}

        def ev(f: float) -> float:
            f = float(f)
            if f not in evals:
                a = self.trials_b(sigma, xis, f, conds=('on',))
                evals[f] = (float(a['on_resp'].mean()), a)
                if log:
                    log(f'  tune sigma {sigma:g}: fixed_thr {f:.6g} -> APL-on rate {evals[f][0]:.5f} (target {target:.5f})')
            return evals[f][0]

        def ok(r: float) -> bool:
            return abs(r / target - 1) <= tol

        f_a = float(self.f0 if guess is None else guess)
        r_a = ev(f_a)
        lo = hi = None   # lo: rate above the target; hi: rate below
        if not ok(r_a):
            up = r_a > target
            step = 0.05
            while len(evals) < max_evals:
                f_b = f_a * (1 + step) if up else f_a / (1 + step)
                r_b = ev(f_b)
                if ok(r_b) or (r_b > target) != up:
                    lo, hi = (f_a, f_b) if up else (f_b, f_a)
                    break
                f_a, step = f_b, step * 2
            while lo is not None and len(evals) < max_evals:
                r_lo, r_hi = evals[lo][0], evals[hi][0]
                if ok(r_lo) or ok(r_hi) or (hi - lo) / lo < 1e-6:
                    break
                w = hi - lo
                if r_lo > 0 and r_hi > 0 and r_lo != r_hi:
                    x = lo + (np.log(r_lo) - np.log(target)) / (np.log(r_lo) - np.log(r_hi)) * w
                    x = min(max(x, lo + 0.1 * w), hi - 0.1 * w)
                else:
                    x = lo + 0.5 * w
                r = ev(x)
                if ok(r):
                    break
                if r > target:
                    lo = x
                else:
                    hi = x
        best = min(evals, key=lambda f: abs(evals[f][0] - target))
        hist = np.array(sorted((f, r) for f, (r, _) in evals.items()), dtype=float)
        return dict(fixed_thr=best, rate=evals[best][0], converged=ok(evals[best][0]), n_evals=len(evals), history=hist,
            arrs=evals[best][1])


class SessionB(Session):
    """Session for part B: tuned calibration points and pseudo-recordings at re-tuned thresholds."""
    model_cls = ExcitabilityModel

    def tuned(self, sigma: float, n_tune: int, seed: int, target: float, guess, tol: float, max_evals: int) -> dict:
        """The re-tuned threshold at `sigma` with its tuning trials (APL on and off at the tuned
        threshold), from the file if present."""
        path = cal_path(self.out_dir, self.name, sigma, n_tune, B_KEY)
        if path.exists():
            a = load_npz(path)
            if float(a['rate_target']) != float(target) or int(a['seed']) != int(seed):
                raise RuntimeError(f'{path}: tuned for rate target {float(a["rate_target"])} / seed {int(a["seed"])}, not '
                    f'{target} / {seed}; move it away to re-tune')
            return a
        m = self.get()
        xis = b_draws(seed, self.name, m.draw_kinds['tune'], 0, n_tune, m.draw_size)
        t0 = time.time()
        if sigma == 0:
            res = dict(fixed_thr=m.f0, rate=float(m.det['on'][0].mean()), converged=True, n_evals=0,
                history=np.empty((0, 2)), arrs=m.trials_b(0.0, xis, m.f0, conds=('on',)))
        else:
            res = m.tune(sigma, xis, target, guess, tol, max_evals, log=lambda s: print(f'{self.name}{s}', flush=True))
        arrs = dict(res['arrs'])
        arrs.update({k: v for k, v in m.trials_b(sigma, xis, res['fixed_thr'], conds=('acute_off',)).items()
            if k.startswith('acute_off_')})
        arrs.update(fixed_thr=np.asarray(res['fixed_thr']), fixed_thr_det=np.asarray(m.f0), rate_target=np.asarray(float(target)),
            rate_tuned=np.asarray(res['rate']), converged=np.asarray(res['converged']), n_evals=np.asarray(res['n_evals']),
            tune_history=res['history'], seed=np.asarray(int(seed)), sigma=np.asarray(float(sigma)))
        save_npz(path, arrs)
        dt = time.time() - t0
        n_runs = res['n_evals'] * n_tune + n_tune
        self.sim_seconds += dt
        self.n_trials_run += n_tune
        self.n_noisy += n_tune if sigma > 0 else 0
        print(f'{self.name}: sigma {sigma:g} tuned in {dt:.0f}s ({res["n_evals"]} evaluations, {n_runs} KC runs, '
            f'{dt / max(n_runs, 1):.1f}s per run): fixed_thr {res["fixed_thr"]:.6g} (deterministic {m.f0:.6g}), APL-on rate '
            f'{res["rate"]:.5f} (target {target:.5f}){"" if res["converged"] else " NOT CONVERGED"} -> {path.name}', flush=True)
        return arrs

    def ensure_b(self, path: Path, sigma: float, n: int, fixed_thr: float, seed: int, rep: int) -> dict:
        """The n trials of pseudo-recording `rep` at `sigma` and `fixed_thr`, simulated first if missing."""
        if path.exists():
            a = load_npz(path)
            if float(a['fixed_thr']) != float(fixed_thr) or int(a['seed']) != int(seed):
                raise RuntimeError(f'{path}: fixed_thr {float(a["fixed_thr"])} / seed {int(a["seed"])}, expected {fixed_thr} / '
                    f'{seed} (a different tuning); move it away to re-run')
            return a
        m = self.get()
        t0 = time.time()
        arrs = m.trials_b(sigma, b_draws(seed, self.name, m.draw_kinds['rec'], rep, n, m.draw_size), fixed_thr)
        arrs['seed'] = np.asarray(int(seed))
        dt = time.time() - t0
        save_npz(path, arrs)
        self.sim_seconds += dt
        self.n_trials_run += n
        self.n_noisy += n if sigma > 0 else 0
        print(f'{self.name}: {n} trials at sigma {sigma:g} in {dt:.0f}s ({dt / n:.1f}s per trial) -> {path.name}', flush=True)
        return arrs


class InputGainModel(ExcitabilityModel):
    """Part C: one model at fixed APL weights, re-run trial by trial with each glomerulus x odor ORN
    input delta scaled by max(1 + sigma * eta, 0) (one eta per glomerulus, odor and trial), through
    the ORN / LN, PN and KC layers; KC thresholds unjittered (thr0(fixed_thr)).
    """
    draw_kinds = dict(tune='tune_c', rec='rec_c', check='check_c')

    def __init__(self, name: str, orn: pd.DataFrame, out_dir: Path):
        super().__init__(name, orn, out_dir)
        self.delta0 = np.array(self.mp.orn.data.delta, dtype=float, copy=True)
        if self.delta0.ndim != 2 or self.delta0.shape[1] != len(orn.columns):
            raise RuntimeError(f'unexpected mp.orn.data.delta shape {self.delta0.shape}')
        self.draw_size = int(self.delta0.size)
        self.info.update(orn_delta_shape=list(self.delta0.shape), gain='max(1 + sigma * eta, 0) per glomerulus x odor x trial')

    def trial(self, sd: float, xi=None, fixed_thr=None, conds=CONDS) -> dict:
        """One trial at gain jitter `sd` (sigma) with draws `xi` (zeros if None) and `fixed_thr`
        (the deterministic run's if None); mp.orn.data.delta is restored afterwards."""
        import olfsysm as osm
        f = self.f0 if fixed_thr is None else float(fixed_thr)
        eta = np.zeros(self.draw_size) if xi is None else np.asarray(xi, dtype=float)
        g = 1.0 + float(sd) * eta.reshape(self.delta0.shape)
        self.mp.orn.data.delta = self.delta0 * np.maximum(g, 0.0)
        try:
            osm.run_ORN_LN_sims(self.mp, self.rv)
            osm.run_PN_sims(self.mp, self.rv)
            thr = self.thr0(f)
            out = {}
            for cond in conds:
                out[f'{cond}_resp'], out[f'{cond}_spk'] = self.run_kc(cond, f, thr)
            out.update(self._pn_now())
        finally:
            self.mp.orn.data.delta = self.delta0
        out['n_clipped'] = np.asarray(int((g < 0).sum()))
        out['factor'] = np.maximum(g, 0.0).ravel().astype(np.float32)
        return out


class SessionC(SessionB):
    model_cls = InputGainModel


def b_guess(tuned: dict, sigma: float):
    """fixed_thr at `sigma` interpolated (linearly in sigma; extrapolated from the two nearest
    points) from {sigma: tuned fixed_thr}; None without points."""
    if not tuned:
        return None
    s = np.array(sorted(tuned), dtype=float)
    f = np.array([tuned[x] for x in s], dtype=float)
    if len(s) == 1:
        return float(f[0])
    if s[0] <= sigma <= s[-1]:
        return float(np.interp(sigma, s, f))
    i = [0, 1] if sigma < s[0] else [-2, -1]
    return float(f[i[0]] + (sigma - s[i[0]]) * (f[i[1]] - f[i[0]]) / (s[i[1]] - s[i[0]]))


def calibrate_b(sess: SessionB, grid: list, n_tune: int, max_extend: int, refine: int, targets: dict, rate_target: float,
        seed: int, tol: float, max_evals: int, f0: float, fixed_matched=None, grid_kind: str = 'grid'):
    """Floors of the tuning trials at each sigma of the grid (extended / refined as needed), each at
    its re-tuned threshold, and the sigmas matching the data's floors. Returns (calibration rows,
    info dict as part A's plus 'tuned' {sigma: fixed_thr}).

    With `fixed_matched` (--sigma-from: another model's 'matched' entry) only the grid is measured
    (its rows labelled `grid_kind`), and the matched sigmas are `fixed_matched`."""
    pts, rows, tuned = {}, [], {0.0: f0}
    grid = sorted(r3(g) for g in grid)

    def measure(sigma: float, kind: str) -> None:
        sigma = r3(sigma)
        if sigma in pts:
            return
        arrs = sess.tuned(sigma, n_tune, seed, rate_target, b_guess(tuned, sigma), tol, max_evals)
        tuned[sigma] = float(arrs['fixed_thr'])
        pts[sigma] = {}
        det_pn_path = sess.out_dir / f'{sess.name}_det_pn.npz'
        pn = pn_stats([arrs], load_npz(det_pn_path)) if det_pn_path.exists() else {}
        for cond in CONDS:
            fg = trial_set_floors(arrs, cond, 'graded')
            ft = trial_set_floors(arrs, cond, 'thresholded')
            pts[sigma][cond] = fg['floor1']
            rows.append({'model': sess.name, B_KEY: sigma, 'kind': kind, 'condition': cond, 'n_trials': n_tune,
                'single_trial_floor': fg['floor1'], 'trial_mean_floor': fg['floor_mean'], 'n_silent_odors': fg['n_silent'],
                'single_trial_floor_thresholded': ft['floor1'], 'target_single_trial_floor': targets[cond],
                'rate_trial': float(arrs[f'{cond}_resp'].mean()), 'rate_target_on': rate_target,
                'spikes_per_kc_odor': float(arrs[f'{cond}_spk'].astype(float).mean()), 'fixed_thr': tuned[sigma],
                'fixed_thr_det': f0, 'tune_converged': bool(arrs['converged']), 'tune_evals': int(arrs['n_evals']),
                'frac_clipped': float(arrs['n_clipped'].mean()) / float(arrs['n_draws_per_trial']),
                **heavy_tail(arrs, cond), **pn})
        print(f"{sess.name} calibration sigma {sigma:g} ({kind}): fixed_thr {tuned[sigma]:.6g} ({tuned[sigma] / f0:.3f} x "
            f"deterministic); APL-on rate {pts_rate(arrs, 'on'):.4f}, APL-off {pts_rate(arrs, 'acute_off'):.4f}; single-trial "
            f"floor APL on {pts[sigma]['on']:.3f} (control flies {targets['on']:.3f}), APL off {pts[sigma]['acute_off']:.3f} "
            f"(APL output blocked (TNT) {targets['acute_off']:.3f})", flush=True)

    def peak(c: str) -> float:
        return max(pts, key=lambda s: pts[s][c])

    for sigma in grid:
        measure(sigma, grid_kind)
    step = grid[-1] / grid[-2] if len(grid) > 1 and grid[-1] > grid[-2] else 2.0
    if fixed_matched is not None:
        return rows, dict(grid=grid, step=step, n_tune_trials=n_tune, points={str(k): v for k, v in sorted(pts.items())},
            matched=fixed_matched, peaks={c: dict(sd=peak(c), floor=pts[peak(c)][c]) for c in CONDS},
            tuned={str(k): v for k, v in sorted(tuned.items())})
    for _ in range(max_extend):
        if not any(pts[peak(c)][c] < targets[c] and peak(c) == max(pts) for c in CONDS):
            break
        measure(max(pts) * step, 'extension')
    for _ in range(max_extend):
        if not any(pts[min(pts)][c] > targets[c] for c in CONDS):
            break
        measure(min(pts) / step, 'extension')
    for _ in range(refine):
        new = set()
        for c in CONDS:
            m = interpolate_sd(pts, c, targets[c])
            if m['bracketed']:
                new.add(m['sd'])
        new = sorted(new - set(pts))
        if not new:
            break
        for sigma in new:
            measure(sigma, 'refine')
    matched = {c: interpolate_sd(pts, c, targets[c]) for c in CONDS}
    peaks = {c: dict(sd=peak(c), floor=pts[peak(c)][c]) for c in CONDS}
    for c, m in matched.items():
        if not m['bracketed']:
            print(f'{sess.name}: WARNING the {c} single-trial floor never crossed {targets[c]:.3f} over sigma {min(pts):g}-'
                f'{max(pts):g} (largest {peaks[c]["floor"]:.3f} at sigma {peaks[c]["sd"]:g}); using the closest point, sigma '
                f'{m["sd"]:g}', flush=True)
    return rows, dict(grid=grid, step=step, n_tune_trials=n_tune, points={str(k): v for k, v in sorted(pts.items())},
        matched=matched, peaks=peaks, tuned={str(k): v for k, v in sorted(tuned.items())})


def pts_rate(arrs: dict, cond: str) -> float:
    return float(arrs[f'{cond}_resp'].mean())


def heavy_tail(arrs, cond: str) -> dict:
    """How much the low tail of the noise factors carries (one trial-set dict or a list of them):
    the largest spike count of a KC to an odor in a trial, and (part B, one factor per KC) the share
    of all spikes from KCs whose threshold was jittered below half its mean."""
    arrs = arrs if isinstance(arrs, list) else [arrs]
    spk = [a[f'{cond}_spk'].astype(float) for a in arrs]
    out = dict(max_spike_count=float(max(s.max() for s in spk)), spike_share_thr_below_half=np.nan)
    if all('factor' in a and a['factor'].shape[-1] == s.shape[1] for a, s in zip(arrs, spk)):
        low = sum(float(s[a['factor'] < 0.5].sum()) for a, s in zip(arrs, spk))
        out['spike_share_thr_below_half'] = low / max(sum(float(s.sum()) for s in spk), 1e-12)
    return out


def make_levels_b(matched: dict, below) -> dict:
    """label -> sigma: the two matched sigmas (APL on to the control flies' floor, APL off to APL
    output blocked (TNT)'s), then `below` factors x the smaller; labels joined with '+' where two
    levels coincide (at 3 significant digits)."""
    m_on, m_off = matched['on']['sd'], matched['acute_off']['sd']
    lv: dict = {}

    def add(label, sigma):
        sigma = r3(sigma)
        for k, v in list(lv.items()):
            if v == sigma:
                lv[f'{k}+{label}'] = lv.pop(k)
                return
        lv[label] = sigma

    add('match_on', m_on)
    add('match_off', m_off)
    for i, b in enumerate(below):
        add('lo' if i == 0 else f'lo{i + 1}', b * min(m_on, m_off))
    return lv


def select_levels(levels: dict, labels) -> dict:
    """`levels` restricted to those with one of `labels` among their '+'-joined labels (--levels;
    all of them when `labels` is empty)."""
    if not labels:
        return levels
    out = {lab: s for lab, s in levels.items() if set(lab.split('+')) & set(labels)}
    if not out:
        raise ValueError(f'--levels {" ".join(labels)}: none of the levels {list(levels)}')
    return out


def sigma_source(model: str, res_dir: Path, sub: str, part: str, thr_jitter: str, thr_min_frac: float, src_dir=None) -> dict:
    """--sigma-from: the matched sigmas of `model` in an existing noise_levels.json of the same noise:
    `src_dir`'s, else `res_dir`'s when it holds `model`, else E43's own folder for this part
    (<shared>/results/E43_model_trial_noise/<sub>). The folder's provenance.json must name the same
    part (and for part B the same jitter and clipping)."""
    cands = [Path(src_dir)] if src_dir else [res_dir, H.shared_dir() / 'results' / EXPERIMENT / sub]
    for d in cands:
        path = d / 'noise_levels.json'
        lv = json.loads(path.read_text()) if path.exists() else {}
        if model in lv:
            break
    else:
        raise FileNotFoundError(f'--sigma-from {model}: no entry in the noise_levels.json of {[str(c) for c in cands]}')
    pv = d / 'provenance.json'
    if not pv.exists():
        raise FileNotFoundError(f'--sigma-from {model}: {pv} missing, so the kind of noise of {path} cannot be checked')
    prov = json.loads(pv.read_text())
    st = prov.get('settings', {})
    theirs = (prov.get('part'), st.get('thr_jitter'), st.get('thr_min_frac')) if part == 'B' else (prov.get('part'),)
    mine = (part, thr_jitter, thr_min_frac) if part == 'B' else (part,)
    if theirs != mine:
        raise ValueError(f'--sigma-from {model}: {path} is noise {theirs} (part, jitter, clipping), this run {mine}')
    e = lv[model]
    return dict(model=model, file=str(path), matched=e['matched'], source_levels=e.get('levels'),
        source_rate_target=e.get('rate_target'), source_n_tune_trials=e.get('n_tune_trials'))


PARTS = {
    'B': dict(dir='partB', session=SessionB, note=THR_NOISE_NOTE, grid=[0.05, 0.1, 0.15, 0.2, 0.3],
        xlabel='σ, relative SD of the per-trial jitter of each KC\'s threshold',
        title=('E43 part B: trial-level KC excitability noise (each KC\'s threshold x (1 + σ ξ) per trial, the same draw '
            'APL on and off; mean threshold re-tuned at each σ so the per-trial APL-on response rate stays at the deterministic '
            'run\'s;\n{n_rec} pseudo-recordings of {n_tr} trials per level; graded spike counts through the KC data\'s pipeline; '
            'APL on compared with control flies, APL off with APL output blocked (TNT))')),
    'C': dict(dir='partC', session=SessionC, note=GAIN_NOISE_NOTE, grid=[0.1, 0.2, 0.3, 0.45, 0.6],
        xlabel='σ, relative SD of the per-trial gain of each glomerulus x odor ORN input',
        title=('E43 part C: slow input variability (each glomerulus x odor ORN input x max(1 + σ η, 0) per trial, constant '
            'over the trial, through the ORN / LN, PN and KC layers; threshold re-tuned at each σ so the per-trial APL-on response '
            'rate stays at the deterministic run\'s;\n{n_rec} pseudo-recordings of {n_tr} trials per level; graded spike counts '
            'through the KC data\'s pipeline; APL on compared with control flies, APL off with APL output blocked (TNT))')),
}


def main_b(args) -> None:
    """Parts B and C: calibration with re-tuned thresholds, pseudo-recordings at each level, and
    the same analysis as part A (tables / figure in <results>/part<X>, trials in
    <outputs>/part<X>)."""
    part = PARTS[args.part]
    # (the lognormal threshold jitter is a different noise: its own folders)
    sub = part['dir'] + ('_lognormal' if args.part == 'B' and args.thr_jitter == 'lognormal' else '')
    res_dir = Path(args.results_dir) if args.results_dir else H.results_dir(args.experiment) / sub
    out_dir = Path(args.outputs_dir) if args.outputs_dir else H.outputs_dir(args.experiment) / sub
    res_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ExcitabilityModel.thr_min_frac = float(args.thr_min_frac)
    ExcitabilityModel.thr_jitter = args.thr_jitter
    grid = args.sigma_grid if args.sigma_grid else part['grid']
    fig_kw = dict(ncol=B_KEY, xlabel=part['xlabel'], suptitle=part['title'])
    if args.figure_only:
        make_figure(res_dir, **fig_kw)
        return
    targets = {c: H.data_target(f'{DATA_PREFIX[c]}_single_trial_floor') for c in CONDS}
    levels_path, checks_path, cal_csv = res_dir / 'noise_levels.json', res_dir / 'checks.json', res_dir / 'calibration.csv'
    levels_all = json.loads(levels_path.read_text()) if levels_path.exists() else {}
    settings = dict(n_recordings=args.n_recordings, n_trials=args.n_trials, sigma_grid=grid,
        tune_trials=args.tune_trials, seed=args.seed, rate_tol=args.rate_tol, tune_max_evals=args.tune_max_evals,
        below=args.below, max_extend=args.cal_max_extend, refine=args.cal_refine, levels=args.levels,
        **({'thr_min_frac': args.thr_min_frac, 'thr_jitter': args.thr_jitter} if args.part == 'B' else {}))
    if not args.analyse_only:
        src_sigma = None
        if args.sigma_from:
            if args.sigma_from in args.models:
                raise ValueError(f'--sigma-from {args.sigma_from}: it is also a model of this run')
            src_sigma = sigma_source(args.sigma_from, res_dir, sub, args.part, args.thr_jitter, float(args.thr_min_frac),
                args.sigma_from_dir)
            print(f'matched sigmas from {args.sigma_from} ({src_sigma["file"]}): APL on {src_sigma["matched"]["on"]["sd"]:g}, '
                f'APL off {src_sigma["matched"]["acute_off"]["sd"]:g}', flush=True)
        orn = H.apl16_orn_deltas()
        checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
        prov_path = res_dir / 'provenance.json'
        prev = json.loads(prov_path.read_text()) if prov_path.exists() else None
        prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E43_model_trial_noise.py',
            part=args.part, host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), models=args.models,
            model_descriptions={m: MODELS[m] for m in args.models}, ref_fit=REF_FIT, fits_source='results/E34_gain_bounded/fits.csv',
            model_sources={m: dict(experiment=FIT_MODELS[m][0], tag=FIT_MODELS[m][1]) for m in args.models if m in FIT_MODELS},
            experiment=args.experiment, sigma_from=src_sigma, settings=settings, floor_targets=targets, noise=part['note'],
            approx_conc_note=H.APPROX_CONC_NOTE, model=H.model_provenance(orn_deltas=orn))
        if prev is not None:
            prov['bases'], prov['params'] = prev.get('bases', {}), prev.get('params', {})
            prov['model_info'] = prev.get('model_info', {})
            prov['rate_targets'] = prev.get('rate_targets', {})
            prov['model_sources'] = dict(prev.get('model_sources', {}), **prov['model_sources'])
            prov['previous_sessions'] = prev.get('previous_sessions', []) + [{k: prev.get(k) for k in ('generated', 'host', 'models',
                'settings', 'n_trials_simulated', 'model')}]
        write_prov(prov_path, prov)
        cal_new = []
        t_all = time.time()
        for name in args.models:
            sess = part['session'](name, orn, out_dir, checks, checks_path, prov, prov_path)
            try:
                # the deterministic run's APL-on response rate and threshold (building the model if not saved here)
                if all((out_dir / f'{name}_{c}_{k}.parquet').exists() for c in CONDS for k in ('responses', 'spike_counts')) and \
                        (out_dir / f'{name}_meta.json').exists():
                    meta = json.loads((out_dir / f'{name}_meta.json').read_text())
                    rate_target = float(load_det(out_dir, name)['on'][0].mean())
                    f0 = float(meta['params']['on']['fixed_thr'])
                else:
                    m = sess.get()
                    rate_target, f0 = float(m.det['on'][0].mean()), m.f0
                prov.setdefault('rate_targets', {})[name] = rate_target
                write_prov(prov_path, prov)
                if src_sigma is None:
                    rows, info = calibrate_b(sess, grid, args.tune_trials, args.cal_max_extend, args.cal_refine, targets,
                        rate_target, args.seed, args.rate_tol, args.tune_max_evals, f0)
                    levels = select_levels(make_levels_b(info['matched'], args.below), args.levels)
                else:
                    # another model's matched sigmas: tuned (and floors measured) only at this run's levels
                    levels = select_levels(make_levels_b(src_sigma['matched'], args.below), args.levels)
                    rows, info = calibrate_b(sess, sorted(set(levels.values())), args.tune_trials, 0, 0, targets, rate_target,
                        args.seed, args.rate_tol, args.tune_max_evals, f0, fixed_matched=src_sigma['matched'],
                        grid_kind='sigma_from')
                cal_new += rows
                tuned = {float(k): v for k, v in info['tuned'].items()}
                lvl_thr, lvl_tune = {}, {}
                for label, sigma in levels.items():
                    a = sess.tuned(sigma, args.tune_trials, args.seed, rate_target, b_guess(tuned, sigma), args.rate_tol,
                        args.tune_max_evals)
                    lvl_thr[label] = float(a['fixed_thr'])
                    lvl_tune[label] = dict(rate_tuned=float(a['rate_tuned']), converged=bool(a['converged']),
                        n_evals=int(a['n_evals']))
                levels_all = update_json(levels_path, name, dict(info, levels=levels, tuned_fixed_thr=lvl_thr, tuning=lvl_tune,
                    fixed_thr_det=f0, rate_target=rate_target, n_recordings=args.n_recordings, n_trials=args.n_trials,
                    seed=args.seed, noise_level=(f'sigma ({part["xlabel"]}; the "sd" keys of matched / peaks hold sigma)'),
                    description=MODELS[name], **model_source(name),
                    **({} if src_sigma is None else {'sigma_from': {k: src_sigma[k] for k in ('model', 'file')}})))
                print(f'{name}: matched sigma APL on {info["matched"]["on"]["sd"]:g}, APL off '
                    f'{info["matched"]["acute_off"]["sd"]:g}; levels {levels}; re-tuned fixed_thr {lvl_thr}', flush=True)
                for label, sigma in levels.items():
                    for rep in range(args.n_recordings):
                        sess.ensure_b(trials_path(out_dir, name, sigma, args.n_trials, rep, B_KEY), sigma, args.n_trials,
                            lvl_thr[label], args.seed, rep)
            finally:
                sess.finish()
                prov['n_trials_simulated'] = prov.get('n_trials_simulated', 0) + sess.n_trials_run
            # (re-read: another process may have added other models' rows meanwhile)
            cal_cur = pd.read_csv(cal_csv, float_precision='round_trip') if cal_csv.exists() else pd.DataFrame()
            keep = cal_cur[~cal_cur.model.isin({r['model'] for r in cal_new})] if len(cal_cur) else cal_cur
            pd.concat([keep, pd.DataFrame(cal_new)], ignore_index=True).to_csv(cal_csv, index=False)
        prov['levels'] = {m: v['levels'] for m, v in levels_all.items()}
        prov['tuned_fixed_thr'] = {m: v['tuned_fixed_thr'] for m, v in levels_all.items()}
        prov['finished'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        prov['seconds'] = time.time() - t_all
        write_prov(prov_path, prov)

    def extra(name, label, sigma, cond, arrs):
        lv = levels_all[name]
        return dict(part=args.part, fixed_thr=lv['tuned_fixed_thr'][label], fixed_thr_det=lv['fixed_thr_det'],
            fixed_thr_ratio=lv['tuned_fixed_thr'][label] / lv['fixed_thr_det'], rate_target_on=lv['rate_target'],
            tune_rate=lv['tuning'][label]['rate_tuned'], tune_converged=lv['tuning'][label]['converged'],
            frac_clipped=float(np.mean([a['n_clipped'].mean() / float(a['n_draws_per_trial']) for a in arrs])), seed=lv['seed'],
            sigma_from=(lv.get('sigma_from') or {}).get('model'), **heavy_tail(arrs, cond))

    # (again: other processes may have added models to this folder meanwhile)
    levels_all = read_json(levels_path, {})
    data = data_side()
    st = analyse_all(levels_all, args.n_recordings, args.n_trials, out_dir, res_dir, data, key=B_KEY, ncol=B_KEY, extra=extra)
    print_summary(st, res_dir, ncol=B_KEY, extra_cols=('fixed_thr_ratio', 'frac_clipped', 'max_spike_count',
        'spike_share_thr_below_half'))
    if len(st):
        make_figure(res_dir, **fig_kw)
    print('done ->', res_dir)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--models', nargs='*', default=None, choices=list(MODELS),
        help='built-in models of this run (default base70 ref40, or none when --model-from is given)')
    ap.add_argument('--model-from', action='append', default=[], metavar='EXPERIMENT:TAG',
        help='also run the model of row TAG of <shared>/results/EXPERIMENT/fits.csv at its fitted parameters, named '
        'E<n>-TAG (repeatable; e.g. E45_gain_at_ctrl_frac:thr70_gain1.25_rate0.068_k2_cap3)')
    ap.add_argument('--experiment', default=EXPERIMENT,
        help='results / outputs folder name (default E43_model_trial_noise; with another name part A goes in its partA '
        'subfolder, parts B / C in partB / partC as usual)')
    ap.add_argument('--levels', nargs='+', default=None, metavar='LABEL',
        help='pseudo-recordings only at the levels with one of these labels (e.g. match_on; part A: match_on match_off lo '
        'hi lo2, parts B / C: match_on match_off lo lo2 ...)')
    ap.add_argument('--n-recordings', type=int, default=10, help='pseudo-recordings per noise level')
    ap.add_argument('--n-trials', type=int, default=3, help='trials per pseudo-recording')
    ap.add_argument('--cal-grid', nargs='+', type=float, default=[25.0, 50.0, 100.0, 200.0, 400.0, 800.0],
        help='noise sds of the calibration grid')
    ap.add_argument('--cal-trials', type=int, default=3, help='trials per calibration point')
    ap.add_argument('--cal-max-extend', type=int, default=3, help='points the grid may be extended by, either way')
    ap.add_argument('--cal-refine', type=int, default=1, help='rounds of trials at the interpolated matched sds')
    ap.add_argument('--bracket', nargs=2, type=float, default=[0.5, 2.0],
        help='factors of the smaller / larger matched sd for the bracketing levels')
    ap.add_argument('--results-dir', default=None, help='default: <shared>/results/E43_model_trial_noise')
    ap.add_argument('--outputs-dir', default=None, help='default: ./outputs/E43_model_trial_noise')
    ap.add_argument('--analyse-only', action='store_true', help='no simulation: tables and figure from the stored trials')
    ap.add_argument('--figure-only', action='store_true')
    ap.add_argument('--part', choices=('A', 'B', 'C'), default='A',
        help='A: PN-rate noise (mp.pn.noise.sd); B: per-trial KC threshold jitter; C: per-trial gain jitter of each glomerulus '
        'x odor ORN input; B and C at a re-tuned threshold, outputs in the partB / partC subfolders')
    ap.add_argument('--sigma-grid', nargs='+', type=float, default=None,
        help='parts B / C: sigmas of the calibration grid (default B 0.05 0.1 0.15 0.2 0.3, C 0.1 0.2 0.3 0.45 0.6)')
    ap.add_argument('--tune-trials', type=int, default=3,
        help='parts B / C: trials (fixed draws) the threshold is re-tuned on; their floors are the calibration points')
    ap.add_argument('--seed', type=int, default=43, help='parts B / C: seed of the noise draws')
    ap.add_argument('--rate-tol', type=float, default=0.01,
        help='parts B / C: relative tolerance of the re-tuned APL-on response rate')
    ap.add_argument('--tune-max-evals', type=int, default=16, help='parts B / C: rate evaluations per re-tuning, at most')
    ap.add_argument('--below', nargs='*', type=float, default=[0.5, 0.25],
        help='parts B / C: factors of the smaller matched sigma for the levels below it (none: only the matched levels)')
    ap.add_argument('--thr-min-frac', type=float, default=0.01,
        help='part B: jittered thresholds are clipped below at this fraction of the unjittered threshold')
    ap.add_argument('--thr-jitter', choices=('normal', 'lognormal'), default='normal',
        help='part B: thr0 x (1 + sigma xi) clipped (normal, as in the entry), or thr0 x exp(sigma xi - sigma^2 / 2) '
        '(lognormal, mean thr0, no low tail; folders partB_lognormal)')
    ap.add_argument('--sigma-from', default=None, metavar='MODEL',
        help="parts B / C: no calibration; the matched sigmas are MODEL's from an existing noise_levels.json of the same "
        "noise (e.g. ref40), each level still re-tuned to this model's rate")
    ap.add_argument('--sigma-from-dir', default=None, metavar='DIR',
        help="the results folder whose noise_levels.json --sigma-from reads (default: this run's if it holds MODEL, else "
        "<shared>/results/E43_model_trial_noise/<part folder>)")
    args = ap.parse_args()
    if args.n_trials < 2 or args.cal_trials < 2 or args.tune_trials < 2:
        ap.error('floors need at least 2 trials')
    if args.sigma_from and args.part == 'A':
        ap.error('--sigma-from is for parts B / C')
    try:
        fit_names = [model_from(s) for s in args.model_from]
    except (ValueError, FileNotFoundError) as e:
        ap.error(str(e))
    if args.models is None:
        args.models = [] if fit_names else ['base70', 'ref40']
    args.models = list(dict.fromkeys(list(args.models) + fit_names))
    if not args.models and not (args.analyse_only or args.figure_only):
        ap.error('no models to run')
    if args.part in ('B', 'C'):
        main_b(args)
        return
    sub_a = '' if args.experiment == EXPERIMENT else 'partA'
    res_dir = Path(args.results_dir) if args.results_dir else H.results_dir(args.experiment) / sub_a
    out_dir = Path(args.outputs_dir) if args.outputs_dir else H.outputs_dir(args.experiment) / sub_a
    res_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        make_figure(res_dir)
        return
    targets = {c: H.data_target(f'{DATA_PREFIX[c]}_single_trial_floor') for c in CONDS}
    levels_path, checks_path, cal_csv = res_dir / 'noise_levels.json', res_dir / 'checks.json', res_dir / 'calibration.csv'
    levels_all = json.loads(levels_path.read_text()) if levels_path.exists() else {}
    if not args.analyse_only:
        orn = H.apl16_orn_deltas()
        checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
        prov_path = res_dir / 'provenance.json'
        prev = json.loads(prov_path.read_text()) if prov_path.exists() else None
        # code provenance at the start: the checkout can move (a pull) while the run goes on
        prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E43_model_trial_noise.py',
            host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), models=args.models,
            model_descriptions={m: MODELS[m] for m in args.models}, ref_fit=REF_FIT, fits_source='results/E34_gain_bounded/fits.csv',
            model_sources={m: dict(experiment=FIT_MODELS[m][0], tag=FIT_MODELS[m][1]) for m in args.models if m in FIT_MODELS},
            experiment=args.experiment,
            settings=dict(n_recordings=args.n_recordings, n_trials=args.n_trials, cal_grid=args.cal_grid, cal_trials=args.cal_trials,
                cal_max_extend=args.cal_max_extend, cal_refine=args.cal_refine, bracket=args.bracket, levels=args.levels),
            floor_targets=targets, noise=NOISE_NOTE, approx_conc_note=H.APPROX_CONC_NOTE,
            model=H.model_provenance(orn_deltas=orn))
        if prev is not None:
            prov['bases'], prov['params'] = prev.get('bases', {}), prev.get('params', {})
            prov['model_info'] = prev.get('model_info', {})
            prov['model_sources'] = dict(prev.get('model_sources', {}), **prov['model_sources'])
            prov['previous_sessions'] = prev.get('previous_sessions', []) + [{k: prev.get(k) for k in ('generated', 'host', 'models',
                'settings', 'n_trials_simulated', 'model')}]
        write_prov(prov_path, prov)
        cal_new = []
        t_all = time.time()
        for name in args.models:
            sess = Session(name, orn, out_dir, checks, checks_path, prov, prov_path)
            try:
                rows, info = calibrate(sess, args.cal_grid, args.cal_trials, args.cal_max_extend, args.cal_refine, targets,
                    out_dir / f'{name}_det_pn.npz')
                cal_new += rows
                levels = select_levels(make_levels(info['matched'], info['peaks'], args.bracket), args.levels)
                levels_all = update_json(levels_path, name, dict(info, levels=levels, n_recordings=args.n_recordings,
                    n_trials=args.n_trials, description=MODELS[name], **model_source(name)))
                print(f'{name}: matched sd APL on {info["matched"]["on"]["sd"]:g}, APL off {info["matched"]["acute_off"]["sd"]:g}; '
                    f'levels {levels}', flush=True)
                for label, sd in levels.items():
                    for rep in range(args.n_recordings):
                        sess.ensure(trials_path(out_dir, name, sd, args.n_trials, rep), sd, args.n_trials)
            finally:
                sess.finish()
                prov['n_trials_simulated'] = prov.get('n_trials_simulated', 0) + sess.n_trials_run
            # calibration points of models re-calibrated here replace their earlier rows
            # (re-read: another process may have added other models' rows meanwhile)
            cal_cur = pd.read_csv(cal_csv, float_precision='round_trip') if cal_csv.exists() else pd.DataFrame()
            keep = cal_cur[~cal_cur.model.isin({r['model'] for r in cal_new})] if len(cal_cur) else cal_cur
            pd.concat([keep, pd.DataFrame(cal_new)], ignore_index=True).to_csv(cal_csv, index=False)
        prov['levels'] = {m: v['levels'] for m, v in levels_all.items()}
        prov['finished'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        prov['seconds'] = time.time() - t_all
        write_prov(prov_path, prov)
    # (again: other processes may have added models to this folder meanwhile)
    levels_all = read_json(levels_path, {})
    data = data_side()
    for c in CONDS:
        g = GENOTYPE[c]
        for key, val in (('single_trial_floor', data[(g, 'graded')]['floor1']), ('trial_mean_floor', data[(g, 'graded')]['floor_mean'])):
            reg = H.data_target(f'{DATA_PREFIX[c]}_{key}')
            if abs(val - reg) > 1e-6:
                print(f'WARNING: KC data {DATA_PREFIX[c]}_{key} recomputed here {val:.6f}, registry {reg:.6f}', flush=True)
    st = analyse_all(levels_all, args.n_recordings, args.n_trials, out_dir, res_dir, data)
    print_summary(st, res_dir)
    if len(st):
        make_figure(res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
