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

Part A (no simulations; the model-output archive in <shared>/data/model_outputs/, every variant of
results/apl_effect/by_variant.csv with archived APL-on and acute APL-off spike counts):
  A1 how few cells: per variant, the loss and its parts after leaving out the k highest-firing KCs (largest
     APL-on count), k = 1, 2, 5, 10, 20, 50, and after leaving out the survivors, the rest of the
     high-firing set and the whole set (both definitions); the largest change of the loss at k <= 2 and
     k <= 20 (leave_out.csv, per_variant.csv).
  A2 the classes on each wiring (the key models, and the three reference models on E47's 20 shuffled draws
     each): each condition's mean similarity and the mean APL effect split by KC class (class_parts.csv);
     the five worst pairs of the per-odor profile of the APL effect (E66's: largest pair-specific part)
     with each class's part of the model's effect and the error with the class left out (worst_pairs.csv);
     the survivors' tuning and wiring against the non-surviving high-firing KCs' and against the
     shuffles' survivors (wiring_by_class.csv, wiring_features.csv); the block and the natural-source
     alignment with the survivors left out, and the survivors' part of the block pairs' APL effect
     (block_vcf.csv); survivors.csv (KC, type, claws by glomerulus, largest counts, odors kept).
  A3 each class's distribution over the odors-per-KC groups in both conditions (opk_classes.csv) and the
     wiring features separating the KCs silent with APL off from those responding to 8 or more odors
     (opk_features.csv), the KC-KC-held fits (E53) beside their seed.
  A4 across the archive, at 6.8 % and at 5 % separately: survivors and their share of the squared APL-on
     vectors against the mean APL effect (both definitions), the pattern part, the response-rate ratio,
     the KC-KC scale and the odors-per-KC distances; rank correlations, and partial ones given the ratio
     (archive_relations.csv).
  A5 the single-fly yardstick: each (control recording, APL-blocked recording) pair's APL-effect loss and
     parts against the mean of the other recordings (88 pairs), beside the models as one wiring draw and
     as the mean over a class's five draws (yardstick.csv, yardstick_models.csv). Single pseudo-recordings
     through the matched noise: the trial-noise pipeline runs saved only their means, so none are scored.
Part B (fixed-scale simulations, E54's `run_variants` on E57's pattern): at E44's 40 Hz and 70 Hz optima
(APL and KC-KC scales fixed, somatic threshold re-tuned to 6.8 % by bisection), with the sets defined on
the real wiring at the fitted scales (A_<model>_real_fit of E47 = E44's fit): (i) the survivors' claws
re-drawn (`al_analysis.wpnkc_shuffles.redraw_kc_subset`, within type); (ii) the whole high-firing set's;
(iii) controls: as many high-firing non-survivors, and as many KCs from outside the high-firing set, both
matched to the survivors in KC type and number of claws (the survivors, resp. the high-firing set,
protected), and for (ii) as many KCs from outside the set matched in type; (iv) on E47's within-type claw
permutation draws (seeds 1-3) the survivors' real claws put back (`restore_kc_subset`); (v) the
survivors' APL->KC weights x2 and x4 on the real wiring at the fitted threshold; the real wiring at the
fitted threshold once (must reproduce the archived counts). Three seeds per arm; 21 runs per model.
Part C (only if part A finds E67's best fit under a definition moves by >= 0.01 with its survivors left
out under that definition): arm (i), the first control and three untreated within-type claw-permutation
draws at that fit's scales, three seeds each, plus its real wiring at the fitted threshold.

Outputs: <shared>/results/E70_high_firing_kcs/ (part A's tables above, comparison_table.md, figures,
provenance.json; parts B / C: wirings.csv, stats_variants.csv with the harness scoring columns,
per_odor_response_fractions.csv, model_pair_distances_long.csv, partB_scores.csv) and
./outputs/E70_high_firing_kcs/ (the re-drawn wirings and swaps, per-variant responses / spike counts,
partC/ caches of the bases' APL-off runs).
Usage: OMP_NUM_THREADS=2 nice -n 10 ./venv/bin/python E70_high_firing_kcs.py --part A [--jobs 2]
       ./venv/bin/python E70_high_firing_kcs.py --part B --plan          (the wirings and jobs only)
       OMP_NUM_THREADS=12 nice -n 10 ./venv/bin/python E70_high_firing_kcs.py --part B --models thr40_gain1.5
       ./venv/bin/python E70_high_firing_kcs.py --part B --summary       (part B's tables only)
       OMP_NUM_THREADS=12 nice -n 10 ./venv/bin/python E70_high_firing_kcs.py --part C --fit-tags <E67 tag> ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
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
RESP_MIN = 2.0
K_LEAVE_OUT = (1, 2, 5, 10, 20, 50)
CLASSES = ('survivor', 'high_firing_other', 'responder', 'silent')
CLASS_LABEL = {'survivor': 'survivors', 'high_firing_other': 'the rest of the high-firing set',
    'responder': 'the other responders', 'silent': 'silent KCs'}
READOUTS = ('raw', 'cal')
READOUT_LABEL = {'raw': 'raw spike counts', 'cal': 'calibrated readout'}
WORST_N = 5
FOCUS_GLOMS = ('DM5', 'DM1', 'DP1m')
OPK_BINS = ('0', '1', '2_3', '4_7', '8plus')
E47 = 'E47_wiring_features'
SHUFFLES = ('claw_permutation', 'claw_permutation_within_type', 'curveball', 'curveball_within_type')
SHUFFLE_LABEL = {'real': 'real wiring', 'claw_permutation': 'claw permutation',
    'claw_permutation_within_type': 'claw permutation within type', 'curveball': 'Curveball',
    'curveball_within_type': 'Curveball within type'}
# the key models: key -> (label, archive folder, tag, model key of E47's draws or None)
KEY_MODELS: Dict[str, Tuple[str, str, str, Optional[str]]] = {
    'ref40': ('the reference model (eight at 1.5, 40 Hz, 6.8 %)', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3', 'thr40_gain1.5'),
    'ref70': ('eight at 1.5, 70 Hz, 6.8 %', 'E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3', 'thr70_gain1.5'),
    'g125': ('eight at 1.25, 70 Hz, 6.8 % (the seed of the batched-proxy fit)', 'E45_gain_at_ctrl_frac',
        'thr70_gain1.25_rate0.068_k2_cap3', 'thr70_gain1.25'),
    'kckc20': ('the same with KC-KC held at 20', 'E53_kckc_held_block', 'thr70_gain1.25_rate0.068_kckc20_k2_cap3', None),
    'kckc45': ('the same with KC-KC held at 45', 'E53_kckc_held_block', 'thr70_gain1.25_rate0.068_kckc45_k2_cap3', None),
    'joint': ('the first joint fit, breadth exponent 2, 5 %', 'E10_joint_fit', 'k2', None),
    'dm1': ('the 70 Hz base with the gain on DM1 alone, 5 %', 'E39_gain_decomposition', 'thr70__only_DM1', None),
}
# the probe draws of the entry: the reference model's E47 draws, three each of three classes
PROBE_SHUFFLES = ('claw_permutation_within_type', 'curveball_within_type', 'claw_permutation')
E67 = 'E67_apl_effect_fits'
E60_FOLDERS = ('E60_proxy_fit', 'E60_noise_proxy_batched')


# ---------------------------------------------------------------------------- loading

def load_counts(folder: str, tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A variant's archived APL-on and acute APL-off spike counts (rows KCs, (kc_id, kc_type) index;
    columns odor abbreviations in APL16 order)."""
    out = []
    for cond in ('on', 'acute_off'):
        m = H.abbrev_columns(pd.read_parquet(ARCHIVE / folder / f'{tag}_{cond}_spike_counts.parquet').astype(float))
        out.append(m[H.odor_order(m.columns)])
    if not out[0].index.equals(out[1].index):
        raise ValueError(f'{folder}/{tag}: APL-on and APL-off KCs differ')
    return out[0], out[1]


def kc_ids(m: pd.DataFrame) -> pd.Index:
    return pd.Index(m.index.get_level_values(H.KC_ID_LEVEL))


def archive_variants() -> pd.DataFrame:
    """Every scored variant (results/apl_effect/by_variant.csv) with archived APL-on and acute APL-off
    spike counts, with its response rates, odors-per-KC distances and realism flag, and its KC-KC
    scale, target rate, per-claw threshold and gain from the archive's params.csv where listed."""
    bv = pd.read_csv(RESULTS / 'apl_effect' / 'by_variant.csv')
    ok = [(ARCHIVE / r.folder / f'{r.tag}_on_spike_counts.parquet').exists()
        and (ARCHIVE / r.folder / f'{r.tag}_acute_off_spike_counts.parquet').exists() for r in bv.itertuples()]
    bv = bv[ok].copy()
    params = []
    for folder in bv.folder.unique():
        p = ARCHIVE / folder / 'params.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        if 'tag' not in df.columns:
            continue
        df = df.drop_duplicates('tag', keep='last')
        keep = [c for c in ('tag', 'kc_kc_inh', 'wKCKC_scale', 'target_rate', 'claw_thr', 'gain', 'gain_glomeruli', 'glomeruli',
            'wiring', 'wiring_seed', 'k') if c in df.columns]
        params.append(df[keep].assign(folder=folder))
    if params:
        pr = pd.concat(params, ignore_index=True)
        pr['tag'] = pr['tag'].astype(str)
        bv['tag'] = bv['tag'].astype(str)
        bv = bv.merge(pr, on=['folder', 'tag'], how='left', suffixes=('', '_params'))
    if 'glomeruli' in bv.columns:
        bv['gain_glomeruli'] = bv.get('gain_glomeruli', pd.Series(np.nan, index=bv.index)).fillna(bv['glomeruli'])
    if 'kc_kc_inh' not in bv.columns:
        bv['kc_kc_inh'] = np.nan
    if 'wKCKC_scale' in bv.columns:
        bv['kc_kc_inh'] = bv['kc_kc_inh'].fillna(bv['wKCKC_scale'])
    bv['kc_kc_inh'] = bv['kc_kc_inh'].fillna(0.0)
    # the operating point: the target rate where listed, else the APL-on rate's nearer of 5 % and 6.8 %
    tr = pd.to_numeric(bv.get('target_rate', pd.Series(np.nan, index=bv.index)), errors='coerce')
    ref = tr.where(tr.notna(), bv.rate_on)
    bv['operating_point'] = np.where((ref - 0.068).abs() < (ref - 0.05).abs(), '6.8 %', '5 %')
    return bv.reset_index(drop=True)


# ---------------------------------------------------------------------------- classes and scores

def classify_kcs(on: pd.DataFrame, off: pd.DataFrame, spikes: float = HIGH_SPIKES) -> pd.Series:
    """Each KC's class: 'survivor' (>= `spikes` to some odor with APL off and with APL on),
    'high_firing_other' (>= `spikes` with APL off only), 'responder' (responds, >= 2 spikes, to some odor
    in either condition, otherwise), 'silent'. Indexed like `on`."""
    hf = off.max(axis=1) >= spikes
    surv = hf & (on.max(axis=1) >= spikes)
    resp = (on.max(axis=1) >= RESP_MIN) | (off.max(axis=1) >= RESP_MIN)
    cls = np.where(surv, 'survivor', np.where(hf, 'high_firing_other', np.where(resp, 'responder', 'silent')))
    return pd.Series(cls, index=on.index, name='kc_class')


def top_kcs(on: pd.DataFrame, off: pd.DataFrame, k: int) -> pd.Index:
    """The k highest-firing KCs: largest APL-on count, ties broken by the largest APL-off count, then the
    total APL-on count, then the order of the rows."""
    df = pd.DataFrame({'a': on.max(axis=1).to_numpy(), 'b': off.max(axis=1).to_numpy(), 'c': on.sum(axis=1).to_numpy(),
        'pos': np.arange(len(on))})
    order = df.sort_values(['a', 'b', 'c', 'pos'], ascending=[False, False, False, True]).pos.to_numpy()
    return on.index[order[:k]]


_REFS: dict = {}


def refs() -> dict:
    """The KC data's per-pair APL effect, the calibration target of the readout (the control flies'
    responders-only participation ratio), the data's two mean RDMs and the reference pairs."""
    if not _REFS:
        _REFS['effect'] = H.yang_apl_effect()
        _REFS['cal_target'] = H.data_target('ctrl_participation_ratio_resp')
        _REFS['ctrl'] = H.yang_mean_rdm('TNT3c')
        _REFS['tnt'] = H.yang_mean_rdm('TNT')
        _REFS['ref'] = H.reference_pairs()
    return _REFS


def calibrated_k(on: pd.DataFrame, bounds: Tuple[float, float] = (1e-3, 1e3), iters: int = 60) -> float:
    """K of the calibrated saturating readout on these APL-on counts: the same bisection as
    `H.calibrate_readout(on, 'saturating', target, stat='participation_ratio', responders_min=2)` (log
    scale, the same bounds, iterations and bracket rule), on numpy arrays (about 50 times faster)."""
    X = on.to_numpy(dtype=float)
    mask = X >= RESP_MIN
    n = X.shape[0]
    target = refs()['cal_target']

    def f(K: float) -> float:
        r = np.where(mask, X / (X + K), 0.0)
        sq = r ** 2
        tot = sq.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            pr = tot ** 2 / (n * (sq ** 2).sum(axis=0))
        pr = pr[np.isfinite(pr)]
        return float(pr.mean()) if len(pr) else np.nan

    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    flo, fhi = f(np.exp(lo)) - target, f(np.exp(hi)) - target
    if flo * fhi > 0:
        return float(np.exp(lo) if abs(flo) < abs(fhi) else np.exp(hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(np.exp(mid)) - target
        if fm * flo <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return float(np.exp(0.5 * (lo + hi)))


def distances(on: pd.DataFrame, off: pd.DataFrame, readout: str = 'raw', K: Optional[float] = None
    ) -> Tuple[pd.Series, pd.Series, float]:
    """The model's pair distances in both conditions under a readout ('raw', or 'cal': K calibrated on
    `on` unless given), and K (NaN for raw)."""
    if readout == 'raw':
        return H.readout_pair_distances(on), H.readout_pair_distances(off), np.nan
    K = calibrated_k(on) if K is None else K
    return H.readout_pair_distances(on, 'saturating', K), H.readout_pair_distances(off, 'saturating', K), float(K)


_FAST: dict = {}


def _fast_setup(odors: Tuple[str, ...]):
    """For `fast_score`: the canonical pairs' odor positions, the data's effect on them and the
    projection onto the additive space over odors."""
    if odors not in _FAST:
        pl = H.pair_list(list(odors))
        if list(H.odor_order(list(odors))) != list(odors):
            raise ValueError('odors must be in odor_order')
        idx = {o: i for i, o in enumerate(odors)}
        mi = pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b'])
        dv = refs()['effect'].reindex(mi).to_numpy(dtype=float)
        X, _ = H.pair_design(mi)
        P = X @ np.linalg.pinv(X)
        _FAST[odors] = (np.array([idx[a] for a, _ in pl]), np.array([idx[b] for _, b in pl]), dv, P)
    return _FAST[odors]


def _cos_dist(M: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        S = (M.T @ M) / np.outer(norms, norms)
    return 1 - S[ia, ib]


def fast_score(on: pd.DataFrame, off: pd.DataFrame, readout: str, K: Optional[float] = None) -> Optional[Dict[str, float]]:
    """`score` against the KC data's effect on numpy arrays (the same distances, decomposition and ρ as the
    harness's, checked equal to 1e-12); None where a distance is NaN (an odor with no response left), for
    the harness's path to handle."""
    from scipy import stats
    ia, ib, dv, P = _fast_setup(tuple(on.columns))
    if readout == 'cal':
        K = calibrated_k(on) if K is None else K
        f = lambda M: M / (M + K)
    else:
        K = np.nan
        f = lambda M: M
    d_on = _cos_dist(f(on.to_numpy(dtype=float)), ia, ib)
    d_off = _cos_dist(f(off.to_numpy(dtype=float)), ia, ib)
    if not (np.isfinite(d_on).all() and np.isfinite(d_off).all() and np.isfinite(dv).all()):
        return None
    m = d_on - d_off
    err = m - dv
    me = float(err.mean())
    errc = (m - m.mean()) - (dv - dv.mean())
    proj = P @ errc
    return dict(rmse=float(np.sqrt(np.mean(err ** 2))), rmse_centered=float(np.sqrt(np.mean((err - me) ** 2))), mean=float(m.mean()),
        mean_error=me, per_odor=float(np.sqrt(np.mean(proj ** 2))), pair_specific=float(np.sqrt(np.mean((errc - proj) ** 2))),
        spearman=float(stats.spearmanr(m, dv)[0]), mean_dist_on=float(d_on.mean()), mean_dist_off=float(d_off.mean()), K=K)


def score(on: pd.DataFrame, off: pd.DataFrame, readout: str = 'raw', data: Optional[pd.Series] = None,
    K: Optional[float] = None) -> Dict[str, float]:
    """The APL-effect loss with its parts (`apl_effect_decomposition`; keys without the `apl_effect_`
    prefix: rmse, rmse_centered (the pattern part), mean, mean_error, per_odor, pair_specific, spearman),
    the mean distances with APL on and off, and K for the calibrated readout ('cal'; calibrated on `on`,
    i.e. on the KCs kept)."""
    if data is None:
        out = fast_score(on, off, readout, K)
        if out is not None:
            return out
    d_on, d_off, K = distances(on, off, readout, K)
    st, _ = H.apl_effect_decomposition(d_on, d_off, refs()['effect'] if data is None else data)
    out = {k.replace('apl_effect_', ''): v for k, v in st.items() if k in ('apl_effect_rmse', 'apl_effect_rmse_centered',
        'apl_effect_mean', 'apl_effect_mean_error', 'apl_effect_per_odor', 'apl_effect_pair_specific', 'apl_effect_spearman')}
    out.update(mean_dist_on=float(d_on.mean()), mean_dist_off=float(d_off.mean()), K=K)
    return out


def sq_share(m: pd.DataFrame, mask: pd.Series) -> float:
    """The KCs in `mask`'s share of each odor's squared response vector, mean over odors."""
    sq = m.to_numpy() ** 2
    tot = sq.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return float(np.nanmean(sq[mask.to_numpy()].sum(axis=0) / tot))


def leave_out_sets(on: pd.DataFrame, off: pd.DataFrame) -> Dict[str, pd.Index]:
    sets = {'none': on.index[:0]}
    for k in K_LEAVE_OUT:
        sets[f'top{k}'] = top_kcs(on, off, k)
    for spikes, sfx in ((HIGH_SPIKES, ''), (CHECK_SPIKES, '_10')):
        c = classify_kcs(on, off, spikes)
        sets[f'survivors{sfx}'] = on.index[(c == 'survivor').to_numpy()]
        sets[f'hf_rest{sfx}'] = on.index[(c == 'high_firing_other').to_numpy()]
        sets[f'hf_all{sfx}'] = on.index[c.isin(['survivor', 'high_firing_other']).to_numpy()]
    return sets


def variant_rows(job: Tuple[str, str]) -> Tuple[dict, List[dict]]:
    """A1 for one variant: (summary row, one row per readout x leave-out with the scores and the
    change from the full set)."""
    folder, tag = job
    on, off = load_counts(folder, tag)
    sets = leave_out_sets(on, off)
    rows = []
    for readout in READOUTS:
        base = None
        for name, kcs in sets.items():
            keep = ~on.index.isin(kcs)
            st = score(on[keep], off[keep], readout)
            if name == 'none':
                base = st
            rows.append(dict(folder=folder, tag=tag, readout=readout, leave_out=name, n_left_out=int(len(kcs)),
                n_kept=int(keep.sum()), **st, d_rmse=st['rmse'] - base['rmse'], d_mean=st['mean'] - base['mean'],
                d_rmse_centered=st['rmse_centered'] - base['rmse_centered']))
    summ = dict(folder=folder, tag=tag, n_kcs=len(on), max_count_on=float(on.to_numpy().max()),
        max_count_off=float(off.to_numpy().max()))
    for spikes, sfx in ((HIGH_SPIKES, ''), (CHECK_SPIKES, '_10')):
        c = classify_kcs(on, off, spikes)
        hf = c.isin(['survivor', 'high_firing_other'])
        summ[f'n_high_firing{sfx}'] = int(hf.sum())
        summ[f'n_survivors{sfx}'] = int((c == 'survivor').sum())
        summ[f'survivor_share_sq_on{sfx}'] = sq_share(on, c == 'survivor')
        summ[f'high_firing_share_sq_off{sfx}'] = sq_share(off, hf)
    df = pd.DataFrame(rows)
    for readout in READOUTS:
        d = df[df.readout == readout].set_index('leave_out')
        for col in ('rmse', 'mean', 'rmse_centered', 'per_odor', 'pair_specific', 'spearman', 'K', 'mean_dist_on', 'mean_dist_off'):
            summ[f'{readout}__{col}'] = d.loc['none', col]
        for lo in ('top1', 'top2', 'survivors', 'hf_rest', 'hf_all', 'survivors_10', 'hf_all_10'):
            summ[f'{readout}__{lo}__rmse'] = d.loc[lo, 'rmse']
            summ[f'{readout}__{lo}__mean'] = d.loc[lo, 'mean']
            summ[f'{readout}__{lo}__rmse_centered'] = d.loc[lo, 'rmse_centered']
        for kmax in (2, 20):
            sub = d.loc[[f'top{k}' for k in K_LEAVE_OUT if k <= kmax], 'd_rmse']
            j = sub.abs().idxmax()
            summ[f'{readout}__max_change_k{kmax}'] = float(sub[j])
            summ[f'{readout}__max_change_k{kmax}_at'] = int(j[3:])
    return summ, rows


def _worker(job):
    try:
        return variant_rows(job)
    except Exception as e:  # a broken archive file loses one variant, not the run
        return dict(folder=job[0], tag=job[1], error=repr(e)), []


def part_a1(variants: pd.DataFrame, res_dir: Path, jobs: int) -> pd.DataFrame:
    todo = list(zip(variants.folder, variants.tag.astype(str)))
    summ, rows = [], []
    t0 = time.time()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    if jobs <= 1:
        it = map(_worker, todo)
        ex = None
    else:
        ex = ProcessPoolExecutor(max_workers=jobs)
        it = ex.map(_worker, todo, chunksize=4)
    for i, (s, r) in enumerate(it):
        summ.append(s); rows.extend(r)
        if (i + 1) % 50 == 0:
            print(f'A1: {i + 1} / {len(todo)} variants ({time.time() - t0:.0f} s)', flush=True)
    if ex is not None:
        ex.shutdown()
    per = pd.DataFrame(summ).drop_duplicates(['folder', 'tag'])
    meta_cols = ['folder', 'tag', 'exp', 'rate_on', 'rate_off', 'spearman_rate_on_vs_data', 'opk_on_tv', 'opk_off_tv', 'realism_flag',
        'kc_kc_inh', 'target_rate', 'claw_thr', 'gain', 'operating_point']
    meta = variants[[c for c in meta_cols if c in variants.columns]].copy()
    meta['tag'] = meta['tag'].astype(str)
    per = meta.merge(per, on=['folder', 'tag'], how='right')
    per['rate_ratio'] = per.rate_off / per.rate_on
    per.to_csv(res_dir / 'per_variant.csv', index=False)
    pd.DataFrame(rows).to_csv(res_dir / 'leave_out.csv', index=False)
    print(f'A1: {len(per)} variants in {time.time() - t0:.0f} s', flush=True)
    return per


# ---------------------------------------------------------------------------- wiring

_W: dict = {}


def real_wiring() -> pd.DataFrame:
    """The bouton base's claw x bouton wiring (0/1; the shared inputs copy, E52's `base_wiring`)."""
    if 'real' not in _W:
        _W['real'] = pd.read_parquet(ARCHIVE / 'inputs' / 'wPNKC_bouton_base.parquet')
    return _W['real']


def shuffled_wiring(shuffle: str, seed: int) -> pd.DataFrame:
    """E47's draw (`bouton_base(wiring=shuffle, wiring_seed=seed)` shuffles the same matrix)."""
    key = (shuffle, int(seed))
    if key not in _W:
        from al_analysis.wpnkc_shuffles import NAMED_SHUFFLES, shuffle_wPNKC
        method, within = NAMED_SHUFFLES[shuffle]
        _W[key] = shuffle_wPNKC(real_wiring(), method, within_kc_type=within, seed=int(seed))
    return _W[key]


def gained_glomeruli() -> List[str]:
    from E27_glomerulus_gain import ORACLE
    return list(ORACLE)


_STATIC: dict = {}


def static_kc_weights() -> Tuple[pd.Series, pd.DataFrame]:
    """Per KC, the connectome APL->KC weight (summed over its claws, normalized to mean 1 over KCs; a
    property of the claws, unchanged by a claw shuffle) and the normalized hemibrain KC->KC weights
    (post x pre, mean incoming weight 1), as the model uses them."""
    if not _STATIC:
        from al_analysis.mb_model import connectome_APL_weights, connectome_KCKC_weights
        W = real_wiring()
        wA, *_ = connectome_APL_weights(prat_claws=True, prat_boutons=True, wPNKC=W)
        per_kc = wA.groupby(level=H.KC_ID_LEVEL, sort=False).sum()
        ids = pd.unique(W.index.get_level_values(H.KC_ID_LEVEL))
        per_kc = per_kc.reindex(ids).fillna(0.0)
        _STATIC['apl'] = per_kc / per_kc.mean()
        _STATIC['kckc'] = connectome_KCKC_weights(ids)
    return _STATIC['apl'], _STATIC['kckc']


def base_breadth(claw_thr, gain, gain_gloms: Optional[Sequence[str]], wiring: Optional[str] = None,
    seed: Optional[int] = None) -> Optional[pd.Series]:
    """The per-KC APL-off total spike count of a bouton base (the breadth of the breadth weights), from
    the copy of the base's cached run in the model-output archive (named as `E10_joint_fit.bouton_base`
    names it); None where no copy is archived."""
    thr_part = 'base_acute_off_spike_counts.parquet' if claw_thr is None or (isinstance(claw_thr, float) and np.isnan(claw_thr)) \
        else f'thr{float(claw_thr):g}_base_acute_off_spike_counts.parquet'
    name = thr_part
    if wiring is not None:
        name = f'wiring-{wiring}-seed{int(seed)}_' + name
    if gain is not None and not (isinstance(gain, float) and np.isnan(gain)) and float(gain) != 1.0:
        gset = '' if gain_gloms is None or sorted(gain_gloms) == sorted(gained_glomeruli()) else '-'.join(sorted(gain_gloms)) + '_'
        name = f'gain{float(gain):g}_{gset}' + name
    elif wiring is None:
        hal = H.shared_dir() / 'data' / 'hal_outputs'
        p = hal / 'E16_claw_saturation' / f'bouton_thr{float(claw_thr):g}_acute_off_spike_counts.parquet' if 'thr' in thr_part \
            else hal / 'E3_drive_dependent_gain' / 'bouton_pnapl_x1_acute_off_spike_counts.parquet'
        if p.exists():
            off = pd.read_parquet(p)
            return pd.Series(off.astype(float).sum(axis=1).to_numpy(), index=off.index.get_level_values(H.KC_ID_LEVEL))
    for p in sorted(ARCHIVE.glob(f'*/{name}')):
        off = pd.read_parquet(p)
        return pd.Series(off.astype(float).sum(axis=1).to_numpy(), index=off.index.get_level_values(H.KC_ID_LEVEL))
    return None


def wiring_features(W: pd.DataFrame, gain: Optional[float] = None, gain_gloms: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Per KC (indexed by KC id): KC type, number of claws, distinct glomeruli, the most claws from one
    glomerulus, claws per glomerulus, claws on the eight gained glomeruli and on DM5, DM1 and DP1m, the
    total claw weight with the gain, the connectome APL->KC weight and the structural KC-KC inhibition
    received and given (the normalized hemibrain weights' row and column sums)."""
    glom = W.columns.get_level_values('glomerulus')
    by_glom = W.T.groupby(glom).sum().T.groupby(level=[H.KC_ID_LEVEL, H.KC_TYPE_LEVEL], sort=False).sum()
    ids = by_glom.index.get_level_values(H.KC_ID_LEVEL)
    gl8 = [g for g in gained_glomeruli() if g in by_glom.columns]
    gg = list(gain_gloms) if gain_gloms is not None else gl8
    g = 1.0 if gain is None or (isinstance(gain, float) and np.isnan(gain)) else float(gain)
    n = by_glom.sum(axis=1).to_numpy()
    out = pd.DataFrame({'kc_type': by_glom.index.get_level_values(H.KC_TYPE_LEVEL), 'n_claws': n,
        'n_glomeruli': (by_glom > 0).sum(axis=1).to_numpy(), 'max_claws_one_glomerulus': by_glom.max(axis=1).to_numpy()},
        index=pd.Index(ids, name=H.KC_ID_LEVEL))
    out['claws_per_glomerulus'] = out.n_claws / out.n_glomeruli.where(out.n_glomeruli > 0)
    out['claws_gained8'] = by_glom[gl8].sum(axis=1).to_numpy()
    for gname in FOCUS_GLOMS:
        out[f'claws_{gname}'] = by_glom[gname].to_numpy() if gname in by_glom.columns else 0
    out['claw_weight'] = n + (g - 1.0) * by_glom[[x for x in gg if x in by_glom.columns]].sum(axis=1).to_numpy()
    apl, kckc = static_kc_weights()
    out['apl_weight_connectome'] = apl.reindex(out.index).to_numpy()
    out['kckc_in_structural'] = kckc.sum(axis=1).reindex(out.index).to_numpy()
    out['kckc_out_structural'] = kckc.sum(axis=0).reindex(out.index).to_numpy()
    return out


def model_kc_features(feat: pd.DataFrame, on: pd.DataFrame, off: pd.DataFrame, kc_kc: float, breadth: Optional[pd.Series],
    k: float = 2.0) -> pd.DataFrame:
    """`wiring_features` plus the model's: the breadth weight (((breadth + 1) / mean) ** k, the factor on
    the connectome APL->KC weight) and the resulting APL->KC weight (normalized to mean 1), the KC-KC
    inhibition received and given at the model's scale, structurally and weighted by the APL-off spike
    counts (summed over odors), and the counts themselves."""
    f = feat.copy()
    ids = kc_ids(on)
    f = f.reindex(ids)
    if breadth is not None:
        b = breadth.reindex(ids).astype(float)
        fac = ((b + 1.0) / (b + 1.0).mean()) ** k
        f['breadth'] = b.to_numpy()
        f['breadth_factor'] = fac.to_numpy()
        w = f.apl_weight_connectome * f.breadth_factor
        f['apl_weight_breadth'] = (w / w.mean()).to_numpy()
    else:
        f['breadth'] = f['breadth_factor'] = f['apl_weight_breadth'] = np.nan
    _, kckc = static_kc_weights()
    kk = kckc.reindex(index=ids, columns=ids).fillna(0.0).to_numpy()
    tot_off = off.sum(axis=1).to_numpy()
    tot_on = on.sum(axis=1).to_numpy()
    f['kckc_in_scaled'] = kc_kc * f.kckc_in_structural
    f['kckc_out_scaled'] = kc_kc * f.kckc_out_structural
    f['kckc_received_off'] = kc_kc * (kk @ tot_off)
    f['kckc_given_off'] = kc_kc * kk.sum(axis=0) * tot_off
    f['kckc_received_on'] = kc_kc * (kk @ tot_on)
    f['max_count_on'] = on.max(axis=1).to_numpy()
    f['max_count_off'] = off.max(axis=1).to_numpy()
    f['n_odors_on'] = (on >= RESP_MIN).sum(axis=1).to_numpy()
    f['n_odors_off'] = (off >= RESP_MIN).sum(axis=1).to_numpy()
    return f


FEATURE_COLS = ('n_claws', 'n_glomeruli', 'max_claws_one_glomerulus', 'claws_per_glomerulus', 'claws_gained8', 'claws_DM5',
    'claws_DM1', 'claws_DP1m', 'claw_weight', 'apl_weight_connectome', 'breadth', 'apl_weight_breadth', 'kckc_in_scaled',
    'kckc_out_scaled', 'kckc_received_off', 'kckc_given_off', 'n_odors_on', 'n_odors_off', 'max_count_on', 'max_count_off')
FEATURE_LABEL = {'n_claws': 'claws', 'n_glomeruli': 'distinct glomeruli', 'max_claws_one_glomerulus': 'most claws from one glomerulus',
    'claws_per_glomerulus': 'claws per glomerulus', 'claws_gained8': 'claws on the eight gained glomeruli', 'claws_DM5': 'claws on DM5',
    'claws_DM1': 'claws on DM1', 'claws_DP1m': 'claws on DP1m', 'claw_weight': 'total claw weight (with the gain)',
    'apl_weight_connectome': 'APL->KC weight, connectome (mean 1)', 'breadth': 'breadth (APL-off spike total of the base)',
    'apl_weight_breadth': 'APL->KC weight with the breadth weight (mean 1)', 'kckc_in_scaled': 'KC-KC weight received (at the scale)',
    'kckc_out_scaled': 'KC-KC weight given (at the scale)', 'kckc_received_off': 'KC-KC inhibition received, APL off (weight x spikes)',
    'kckc_given_off': 'KC-KC inhibition given, APL off (weight x spikes)', 'n_odors_on': 'odors responded to, APL on',
    'n_odors_off': 'odors responded to, APL off', 'max_count_on': 'largest spike count, APL on', 'max_count_off': 'largest spike count, APL off'}


def claws_string(W: pd.DataFrame, ids: Sequence) -> Dict[int, str]:
    glom = W.columns.get_level_values('glomerulus')
    sub = W[W.index.get_level_values(H.KC_ID_LEVEL).isin(list(ids))]
    by = sub.T.groupby(glom).sum().T.groupby(level=H.KC_ID_LEVEL).sum()
    return {kc: ', '.join(f'{g} {int(v)}' for g, v in r[r > 0].sort_values(ascending=False).items()) for kc, r in by.iterrows()}


# ---------------------------------------------------------------------------- A2 / A3

def class_parts(on: pd.DataFrame, off: pd.DataFrame, classes: pd.Series, readout: str) -> pd.DataFrame:
    """Per pair, each class's term of the cosine similarity in each condition (the class's part of the dot
    product over the product of the two odors' lengths over all KCs, so the terms add up to the
    similarity) and its part of the per-pair APL effect (APL-off term minus APL-on term, as the effect is
    the APL-on distance minus the APL-off distance)."""
    K = calibrated_k(on) if readout == 'cal' else None
    X = {c: (H.readout(m, 'saturating', K) if K is not None else m).to_numpy() for c, m in (('on', on), ('off', off))}
    odors = list(on.columns)
    pl = H.pair_list(odors)
    idx = {o: i for i, o in enumerate(odors)}
    ia = np.array([idx[a] for a, _ in pl]); ib = np.array([idx[b] for _, b in pl])
    rows = {}
    for cond, M in X.items():
        norms = np.linalg.norm(M, axis=0)
        den = norms[ia] * norms[ib]
        for c in CLASSES:
            mk = (classes == c).to_numpy()
            prod = (M[mk][:, ia] * M[mk][:, ib]).sum(axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                rows[(cond, c)] = prod / den
    df = pd.DataFrame({f'sim_{cond}__{c}': v for (cond, c), v in rows.items()},
        index=pd.MultiIndex.from_tuples(pl, names=['odor_a', 'odor_b']))
    for c in CLASSES:
        df[f'effect__{c}'] = df[f'sim_off__{c}'] - df[f'sim_on__{c}']
    return df


def worst_pair_rows(label: str, on: pd.DataFrame, off: pd.DataFrame, classes: pd.Series, readout: str) -> List[dict]:
    """The five pairs with the largest pair-specific part of the APL-effect error (E66's worst pairs), each
    class's part of the model's effect on them and the error with the class left out (K re-calibrated on
    the KCs kept)."""
    d_on, d_off, K = distances(on, off, readout)
    parts = H.apl_effect_pair_parts(d_on, d_off, refs()['effect'])
    top = parts.reindex(parts.pair_specific.abs().sort_values(ascending=False).index[:WORST_N])
    cp = class_parts(on, off, classes, readout)
    without = {}
    for c in ('survivor', 'high_firing_other'):
        keep = (classes != c).to_numpy()
        o2, f2, _ = distances(on[keep], off[keep], readout)
        without[c] = H.apl_effect_pair_parts(o2, f2, refs()['effect'])
    rows = []
    for rank, ((a, b), r) in enumerate(top.iterrows(), 1):
        row = dict(model=label, readout=readout, rank=rank, odor_a=a, odor_b=b, model_effect=r.model, data_effect=r.data,
            error=r.model - r.data, pattern_error=r.error, pair_specific=r.pair_specific)
        for c in CLASSES:
            row[f'part__{c}'] = float(cp.loc[(a, b), f'effect__{c}'])
        for c, pw in without.items():
            row[f'error_without__{c}'] = float(pw.loc[(a, b), 'model'] - pw.loc[(a, b), 'data'])
            row[f'pair_specific_without__{c}'] = float(pw.loc[(a, b), 'pair_specific'])
        rows.append(row)
    return rows


def opk_distribution(m: pd.DataFrame) -> np.ndarray:
    n = (m.to_numpy() >= RESP_MIN).sum(axis=1)
    return np.array([np.mean(n == 0), np.mean(n == 1), np.mean((n >= 2) & (n <= 3)), np.mean((n >= 4) & (n <= 7)), np.mean(n >= 8)]) \
        if len(n) else np.full(5, np.nan)


def block_vcf(on: pd.DataFrame, off: pd.DataFrame, readout: str) -> Dict[str, float]:
    d_on, d_off, _ = distances(on, off, readout)
    oo, _ = H.compare_on_off(d_on, d_off)
    v_on = H.compare_to_references(d_on, refs()['ref'])['spearman_resid_vs_vcf_hyp']
    v_off = H.compare_to_references(d_off, refs()['ref'])['spearman_resid_vs_vcf_hyp']
    return dict(block=oo['block_fruity_small_iso'], vcf_on=v_on, vcf_off=v_off, vcf_on_minus_off=v_on - v_off,
        rho_on=H.compare_model_to_data(d_on, refs()['ctrl'])['spearman'], rho_off=H.compare_model_to_data(d_off, refs()['tnt'])['spearman'])


def key_variants() -> List[dict]:
    """The A2 / A3 list: each key model on its wiring, the three reference models on E47's 20 draws each,
    E67's archived fits and E60's first fit if archived."""
    out = []
    p = pd.read_csv(RESULTS / E47 / 'stats_variants.csv') if (RESULTS / E47 / 'stats_variants.csv').exists() else None
    for key, (label, folder, tag, e47) in KEY_MODELS.items():
        out.append(dict(key=key, label=label, folder=folder, tag=tag, wiring='real', seed=None, model=key))
        if e47 is None:
            continue
        for sh in SHUFFLES:
            for seed in range(1, 6):
                t = f'A_{e47}_{sh}-seed{seed}_retuned'
                if (ARCHIVE / E47 / f'{t}_on_spike_counts.parquet').exists():
                    out.append(dict(key=f'{key}__{sh}-seed{seed}', label=f'{label}, {SHUFFLE_LABEL[sh]} seed {seed}', folder=E47, tag=t,
                        wiring=sh, seed=seed, model=key))
    fits = RESULTS / E67 / 'fits.csv'
    if fits.exists():
        for r in pd.read_csv(fits).itertuples():
            if (ARCHIVE / E67 / f'{r.tag}_on_spike_counts.parquet').exists():
                out.append(dict(key=f'e67__{r.tag}', label=f'APL-effect fit {r.tag}', folder=E67, tag=r.tag, wiring='real', seed=None,
                    model=f'e67__{r.tag}'))
    for f in E60_FOLDERS:
        for p_on in sorted((ARCHIVE / f).glob('*_on_spike_counts.parquet')) if (ARCHIVE / f).exists() else []:
            t = p_on.name[:-len('_on_spike_counts.parquet')]
            out.append(dict(key=f'e60__{t}', label=f'batched-proxy fit {t}', folder=f, tag=t, wiring='real', seed=None, model=f'e60__{t}'))
    return out


def model_meta(variants: pd.DataFrame, folder: str, tag: str) -> dict:
    r = variants[(variants.folder == folder) & (variants.tag.astype(str) == str(tag))]
    if not len(r):
        return dict(kc_kc_inh=0.0, claw_thr=np.nan, gain=np.nan, gain_glomeruli=None, k=2.0)
    r = r.iloc[0]
    gg = r.get('gain_glomeruli')
    gg = [x.strip() for x in str(gg).split(',')] if isinstance(gg, str) and gg.strip() else None
    kv = r.get('k', 2.0)
    return dict(kc_kc_inh=float(r.get('kc_kc_inh', 0.0) or 0.0), claw_thr=r.get('claw_thr', np.nan), gain=r.get('gain', np.nan),
        gain_glomeruli=gg, k=2.0 if pd.isna(kv) else float(kv))


def part_a2(variants: pd.DataFrame, res_dir: Path) -> Dict[str, pd.DataFrame]:
    """A2 and A3 over `key_variants`."""
    feats_cache: Dict[tuple, pd.DataFrame] = {}
    cls_rows, part_rows, worst_rows, bv_rows, feat_rows, surv_rows, opk_rows, sep_rows, kc_rows = [], [], [], [], [], [], [], [], []
    t0 = time.time()
    for v in key_variants():
        try:
            on, off = load_counts(v['folder'], v['tag'])
        except Exception as e:
            print(f"A2: {v['key']}: {e!r}", flush=True)
            continue
        meta = model_meta(variants, v['folder'], v['tag'])
        if v['wiring'] == 'real':
            W = real_wiring()
            bkey = (meta['claw_thr'], meta['gain'], tuple(meta['gain_glomeruli'] or ()), None, None)
        else:
            W = shuffled_wiring(v['wiring'], v['seed'])
            bkey = (meta['claw_thr'], meta['gain'], tuple(meta['gain_glomeruli'] or ()), v['wiring'], v['seed'])
        fk = (v['wiring'], v['seed'], meta['gain'] if not pd.isna(meta['gain']) else None, tuple(meta['gain_glomeruli'] or ()))
        if fk not in feats_cache:
            feats_cache[fk] = wiring_features(W, None if pd.isna(meta['gain']) else float(meta['gain']), meta['gain_glomeruli'])
        breadth = base_breadth(None if pd.isna(meta['claw_thr']) else float(meta['claw_thr']),
            None if pd.isna(meta['gain']) else float(meta['gain']), meta['gain_glomeruli'], *( (v['wiring'], v['seed']) if v['wiring'] != 'real' else (None, None)))
        f = model_kc_features(feats_cache[fk], on, off, meta['kc_kc_inh'], breadth, meta['k'])
        base = dict(key=v['key'], model=v['model'], label=v['label'], folder=v['folder'], tag=v['tag'], wiring=v['wiring'], seed=v['seed'])
        for spikes, sfx in ((HIGH_SPIKES, ''), (CHECK_SPIKES, '_10')):
            c = classify_kcs(on, off, spikes)
            cv = c.to_numpy()
            cls_rows.append(dict(base, definition=f'{spikes:g} spikes', **{f'n_{x}': int((cv == x).sum()) for x in CLASSES},
                survivor_share_sq_on=sq_share(on, c == 'survivor'),
                high_firing_share_sq_off=sq_share(off, c.isin(['survivor', 'high_firing_other']))))
            if sfx:
                continue
            for readout in READOUTS:
                cp = class_parts(on, off, c, readout)
                part_rows.append(dict(base, readout=readout, **{col: float(cp[col].mean()) for col in cp.columns}))
                if v['wiring'] == 'real' or v['key'].endswith('seed1'):
                    worst_rows.extend([dict(base, **r) for r in worst_pair_rows(v['label'], on, off, c, readout)])
                keep = (cv != 'survivor')
                full = block_vcf(on, off, readout)
                wo = block_vcf(on[keep], off[keep], readout)
                blk = [(a, b) for a, b in cp.index if H.block_label(a, b) == H.FRUITY_SMALL_BLOCK]
                eff_blk = cp.loc[blk, [f'effect__{x}' for x in CLASSES]].sum(axis=1).mean()
                bv_rows.append(dict(base, readout=readout, **{f'{k}': val for k, val in full.items()},
                    **{f'{k}__survivors_out': val for k, val in wo.items()},
                    block_pairs_effect=float(eff_blk), block_pairs_effect_survivors=float(cp.loc[blk, 'effect__survivor'].mean()),
                    block_pairs_survivor_share=float(cp.loc[blk, 'effect__survivor'].mean() / eff_blk) if eff_blk else np.nan,
                    n_block_pairs=len(blk)))
            fc = f.assign(kc_class=cv)
            for x in CLASSES:
                g = fc[fc.kc_class == x]
                feat_rows.append(dict(base, kc_class=x, n=len(g), **{col: float(g[col].mean()) if len(g) else np.nan for col in FEATURE_COLS},
                    **{f'type_{t}': int((g.kc_type == t).sum()) for t in sorted(f.kc_type.dropna().unique())}))
                for cond, m in (('on', on), ('off', off)):
                    d = opk_distribution(m[(cv == x)])
                    opk_rows.append(dict(base, kc_class=x, condition=cond, n=int((cv == x).sum()), **{f'opk_{b}': float(val) for b, val in zip(OPK_BINS, d)}))
            # A3: what separates the KCs silent with APL off from those responding to 8 or more odors
            n_off = (off >= RESP_MIN).sum(axis=1).to_numpy()
            silent, broad = fc[n_off == 0], fc[n_off >= 8]
            for col in FEATURE_COLS:
                if col in ('n_odors_on', 'n_odors_off', 'max_count_on', 'max_count_off'):
                    continue
                a, b = silent[col].dropna().astype(float), broad[col].dropna().astype(float)
                auc = np.nan
                if len(a) and len(b):
                    from scipy.stats import mannwhitneyu
                    try:
                        auc = float(mannwhitneyu(b, a).statistic / (len(a) * len(b)))
                    except ValueError:
                        auc = np.nan
                sep_rows.append(dict(base, feature=col, n_silent_off=len(a), n_8plus_off=len(b), mean_silent_off=float(a.mean()) if len(a) else np.nan,
                    mean_8plus_off=float(b.mean()) if len(b) else np.nan, auc_8plus_vs_silent=auc))
            if v['wiring'] == 'real':
                kc_rows.append(fc.assign(**{kk: vv for kk, vv in base.items() if kk not in ('label',)}))
            sv = fc[fc.kc_class == 'survivor']
            if len(sv):
                cl = claws_string(W, sv.index)
                for kc, r in sv.iterrows():
                    o_on, o_off = on.loc[on.index.get_level_values(H.KC_ID_LEVEL) == kc].iloc[0], off.loc[off.index.get_level_values(H.KC_ID_LEVEL) == kc].iloc[0]
                    surv_rows.append(dict(base, kc_id=kc, kc_type=r.kc_type, claws=cl.get(kc, ''), n_claws=r.n_claws,
                        claws_gained8=r.claws_gained8, apl_weight_connectome=r.apl_weight_connectome, apl_weight_breadth=r.apl_weight_breadth,
                        kckc_received_off=r.kckc_received_off, max_count_on=o_on.max(), max_count_off=o_off.max(),
                        top_odor_on=o_on.idxmax(), top_odor_off=o_off.idxmax(),
                        odors_kept_on=', '.join(o for o in on.columns if o_on[o] >= RESP_MIN),
                        odors_20_on=', '.join(o for o in on.columns if o_on[o] >= HIGH_SPIKES),
                        n_odors_on=int((o_on >= RESP_MIN).sum()), n_odors_off=int((o_off >= RESP_MIN).sum())))
        print(f"A2: {v['key']} ({time.time() - t0:.0f} s)", flush=True)
    out = dict(classes=pd.DataFrame(cls_rows), class_parts=pd.DataFrame(part_rows), worst_pairs=pd.DataFrame(worst_rows),
        block_vcf=pd.DataFrame(bv_rows), wiring_by_class=pd.DataFrame(feat_rows), survivors=pd.DataFrame(surv_rows),
        opk_classes=pd.DataFrame(opk_rows), opk_features=pd.DataFrame(sep_rows))
    for name, df in out.items():
        df.to_csv(res_dir / f'{name}.csv', index=False)
    if kc_rows:
        kcs = pd.concat(kc_rows)
        kcs.reset_index().to_csv(res_dir / 'wiring_features.csv', index=False)
    return out


# ---------------------------------------------------------------------------- A4

def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Spearman correlation of x and y given z (Pearson on the ranks' residuals after regressing out z's
    ranks)."""
    d = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna().rank()
    if len(d) < 5:
        return np.nan
    Z = np.column_stack([np.ones(len(d)), d.z.to_numpy()])
    rx = d.x.to_numpy() - Z @ np.linalg.lstsq(Z, d.x.to_numpy(), rcond=None)[0]
    ry = d.y.to_numpy() - Z @ np.linalg.lstsq(Z, d.y.to_numpy(), rcond=None)[0]
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


A4_X = ('n_survivors', 'survivor_share_sq_on', 'n_survivors_10', 'survivor_share_sq_on_10')
A4_Y = ('raw__mean', 'cal__mean', 'raw__rmse', 'cal__rmse', 'raw__rmse_centered', 'cal__rmse_centered', 'rate_ratio', 'kc_kc_inh',
    'opk_on_tv', 'opk_off_tv')


def part_a4(per: pd.DataFrame, res_dir: Path) -> pd.DataFrame:
    rows = []
    per = per[per.get('error', pd.Series(np.nan, index=per.index)).isna()] if 'error' in per.columns else per
    for op, g in per.groupby('operating_point'):
        for x in A4_X:
            for y in A4_Y:
                d = g[list(dict.fromkeys([x, y, 'rate_ratio']))].apply(pd.to_numeric, errors='coerce').dropna()
                rho = float(d[x].rank().corr(d[y].rank())) if len(d) > 4 else np.nan
                part = partial_spearman(d[x], d[y], d['rate_ratio']) if y != 'rate_ratio' else np.nan
                rows.append(dict(operating_point=op, x=x, y=y, n=len(d), spearman=rho, partial_spearman_given_ratio=part))
    out = pd.DataFrame(rows)
    out.to_csv(res_dir / 'archive_relations.csv', index=False)
    return out


# ---------------------------------------------------------------------------- A5

def yardstick() -> pd.DataFrame:
    """A5: the KC data's APL-effect loss and parts for each (control recording, APL-blocked recording) pair
    against the mean of the other recordings (the other control recordings' mean minus the other
    APL-blocked recordings' mean)."""
    ctrl, tnt = H.yang_recording_rdms('TNT3c'), H.yang_recording_rdms('TNT')
    rows = []
    for rc in ctrl.index:
        for rt in tnt.index:
            on, off = H.canonical_pairs(ctrl.loc[rc]), H.canonical_pairs(tnt.loc[rt])
            data = H.apl_effect(ctrl.drop(index=rc).mean(), tnt.drop(index=rt).mean())
            st, _ = H.apl_effect_decomposition(on, off, data)
            rows.append(dict(control_recording='/'.join(map(str, rc)), blocked_recording='/'.join(map(str, rt)),
                **{k.replace('apl_effect_', ''): v for k, v in st.items() if k in ('apl_effect_rmse', 'apl_effect_rmse_centered', 'apl_effect_mean',
                'apl_effect_mean_error', 'apl_effect_per_odor', 'apl_effect_pair_specific', 'apl_effect_spearman')}))
    return pd.DataFrame(rows)


def yardstick_models(variants: pd.DataFrame) -> pd.DataFrame:
    """The key models scored the same way as one wiring draw (their real-wiring fit), and the three
    reference models as the mean over each E47 class's five draws (each condition's distances averaged
    over the draws, K calibrated per draw), beside the per-draw scores' median and range."""
    rows = []
    for key, (label, folder, tag, e47) in KEY_MODELS.items():
        try:
            on, off = load_counts(folder, tag)
        except Exception:
            continue
        for readout in READOUTS:
            rows.append(dict(model=key, label=label, what='one wiring draw (the real wiring)', readout=readout, n_draws=1, **score(on, off, readout)))
        if e47 is None:
            continue
        for sh in SHUFFLES:
            for readout in READOUTS:
                ds_on, ds_off, per, per_c = [], [], [], []
                for seed in range(1, 6):
                    t = f'A_{e47}_{sh}-seed{seed}_retuned'
                    try:
                        o, f = load_counts(E47, t)
                    except Exception:
                        continue
                    d_on, d_off, _ = distances(o, f, readout)
                    ds_on.append(d_on); ds_off.append(d_off)
                    sc_ = score(o, f, readout)
                    per.append(sc_['rmse']); per_c.append(sc_['rmse_centered'])
                if not ds_on:
                    continue
                st, _ = H.apl_effect_decomposition(pd.concat(ds_on, axis=1).mean(axis=1), pd.concat(ds_off, axis=1).mean(axis=1), refs()['effect'])
                rows.append(dict(model=key, label=label, what=f'mean over the {SHUFFLE_LABEL[sh]} draws', readout=readout, n_draws=len(ds_on),
                    **{k.replace('apl_effect_', ''): v for k, v in st.items() if k in ('apl_effect_rmse', 'apl_effect_rmse_centered', 'apl_effect_mean',
                    'apl_effect_mean_error', 'apl_effect_per_odor', 'apl_effect_pair_specific', 'apl_effect_spearman')},
                    per_draw_rmse_median=float(np.median(per)), per_draw_rmse_min=float(np.min(per)), per_draw_rmse_max=float(np.max(per)),
                    per_draw_pattern_mean=float(np.mean(per_c)), per_draw_pattern_median=float(np.median(per_c))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- tables and figures

def _f(x, fmt: str) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return '-'
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return '-'


def part_a_tables(per: pd.DataFrame, a2: Dict[str, pd.DataFrame], a4: pd.DataFrame, yard: pd.DataFrame, ymod: pd.DataFrame,
    res_dir: Path) -> str:
    out = []
    dt = H.data_target
    floor = dt('apl_effect_rmse_floor')
    data_mean = float(refs()['effect'].mean())
    # the headline table (the entry's probe table)
    pv = per.set_index(['folder', 'tag'])
    rows = ['**How few cells the APL-effect loss rests on: the key models with their survivors (20 or more spikes to some odor with APL '
        'on and off) or their highest-firing KCs left out after the fact (each cell: APL-effect loss; mean APL effect)**', '',
        '| model | high-firing KCs / survivors | survivors\' share of each odor\'s squared APL-on response vector | raw counts | raw counts, '
        'survivors left out | raw counts, the one highest-firing KC left out | raw counts, the two left out | largest change with up to 2 / up to 20 '
        'left out, raw counts | calibrated readout | calibrated readout, survivors left out | calibrated readout, the two highest-firing left out | '
        'largest change with up to 2 / up to 20 left out, calibrated readout |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']

    def cell(r, ro, lo=None):
        if lo is None:
            return f"{_f(r[f'{ro}__rmse'], '.3f')}; {_f(r[f'{ro}__mean'], '+.3f')}"
        return f"{_f(r[f'{ro}__{lo}__rmse'], '.3f')}; {_f(r[f'{ro}__{lo}__mean'], '+.3f')}"

    def add_row(label, r):
        rows.append(f"| {label} | {int(r.n_high_firing)} / {int(r.n_survivors)} | {_f(r.survivor_share_sq_on, '.3f')} | {cell(r, 'raw')} | "
            f"{cell(r, 'raw', 'survivors')} | {cell(r, 'raw', 'top1')} | {cell(r, 'raw', 'top2')} | {_f(r['raw__max_change_k2'], '+.3f')} / "
            f"{_f(r['raw__max_change_k20'], '+.3f')} | {cell(r, 'cal')} | {cell(r, 'cal', 'survivors')} | {cell(r, 'cal', 'top2')} | "
            f"{_f(r['cal__max_change_k2'], '+.3f')} / {_f(r['cal__max_change_k20'], '+.3f')} |")

    for key, (label, folder, tag, e47) in KEY_MODELS.items():
        if (folder, tag) in pv.index:
            add_row(label, pv.loc[(folder, tag)])
        if key == 'ref40':
            sub = per[(per.folder == E47) & per.tag.astype(str).str.match(r'A_thr40_gain1\.5_(%s)-seed[123]_retuned' % '|'.join(PROBE_SHUFFLES))]
            if len(sub):
                rng = lambda c, fmt: f"{_f(sub[c].min(), fmt)} to {_f(sub[c].max(), fmt)}"
                rows.append(f"| the same scales on nine shuffled draws (three each of the claw permutation within type, Curveball within type and "
                    f"the claw permutation) | {int(sub.n_high_firing.min())} to {int(sub.n_high_firing.max())} / {int(sub.n_survivors.min())} to "
                    f"{int(sub.n_survivors.max())} | {rng('survivor_share_sq_on', '.3f')} | {rng('raw__rmse', '.3f')}; {rng('raw__mean', '+.3f')} | "
                    f"{rng('raw__survivors__rmse', '.3f')}; {rng('raw__survivors__mean', '+.3f')} | {rng('raw__top1__rmse', '.3f')}; "
                    f"{rng('raw__top1__mean', '+.3f')} | {rng('raw__top2__rmse', '.3f')}; {rng('raw__top2__mean', '+.3f')} | - | "
                    f"{rng('cal__rmse', '.3f')}; {rng('cal__mean', '+.3f')} | {rng('cal__survivors__rmse', '.3f')}; {rng('cal__survivors__mean', '+.3f')} | "
                    f"{rng('cal__top2__rmse', '.3f')}; {rng('cal__top2__mean', '+.3f')} | - |")
    e67 = per[per.folder == E67]
    for r in e67.itertuples():
        add_row(f'APL-effect fit {r.tag}', pv.loc[(r.folder, r.tag)])
    rows.append(f"| KC data | | | floor {floor:.3f}; {data_mean:+.3f} | | | | | floor {floor:.3f}; {data_mean:+.3f} | | | |")
    out.append('\n'.join(rows))
    # the whole high-firing set left out: real against shuffled
    rows = ['**The whole high-firing set left out: the reference model on the real wiring and on the nine probe draws (loss / pattern part)**', '',
        '| | raw counts {span=2} | calibrated readout {span=2} |', '| wiring | loss | pattern part | loss | pattern part |', '|---|---:|---:|---:|---:|']
    r0 = pv.loc[(KEY_MODELS['ref40'][1], KEY_MODELS['ref40'][2])] if (KEY_MODELS['ref40'][1], KEY_MODELS['ref40'][2]) in pv.index else None
    if r0 is not None:
        rows.append(f"| real wiring | {_f(r0['raw__hf_all__rmse'], '.3f')} | {_f(r0['raw__hf_all__rmse_centered'], '.3f')} | "
            f"{_f(r0['cal__hf_all__rmse'], '.3f')} | {_f(r0['cal__hf_all__rmse_centered'], '.3f')} |")
        sub = per[(per.folder == E47) & per.tag.astype(str).str.match(r'A_thr40_gain1\.5_(%s)-seed[123]_retuned' % '|'.join(PROBE_SHUFFLES))]
        if len(sub):
            rows.append(f"| nine shuffled draws (range) | {_f(sub['raw__hf_all__rmse'].min(), '.3f')} to {_f(sub['raw__hf_all__rmse'].max(), '.3f')} | "
                f"{_f(sub['raw__hf_all__rmse_centered'].min(), '.3f')} to {_f(sub['raw__hf_all__rmse_centered'].max(), '.3f')} | "
                f"{_f(sub['cal__hf_all__rmse'].min(), '.3f')} to {_f(sub['cal__hf_all__rmse'].max(), '.3f')} | "
                f"{_f(sub['cal__hf_all__rmse_centered'].min(), '.3f')} to {_f(sub['cal__hf_all__rmse_centered'].max(), '.3f')} |")
        out.append('\n'.join(rows))
    # the yardstick
    if len(yard):
        q = lambda c: f"{yard[c].median():.3f} ({yard[c].min():.3f} to {yard[c].max():.3f})"
        rows = ['**The single-fly yardstick: one control recording paired with one APL-blocked recording, against the mean of the other '
            'recordings (88 pairs; median, range)**', '', '| statistic | single pairs of recordings | KC data (mean over recordings): floor |', '|---|---:|---:|',
            f"| APL-effect loss | {q('rmse')} | {floor:.3f} |", f"| pattern part | {q('rmse_centered')} | |",
            f"| per-odor part | {q('per_odor')} | |", f"| pair-specific part | {q('pair_specific')} | |",
            f"| ρ on the per-pair APL effect | {yard.spearman.median():.3f} ({yard.spearman.min():.3f} to {yard.spearman.max():.3f}) | |"]
        out.append('\n'.join(rows))
    if len(ymod):
        rows = ['**The models scored as one wiring draw and as the mean over a class\'s five draws (APL-effect loss / pattern part)**', '',
            '| model | what | raw counts: loss / pattern part | calibrated readout: loss / pattern part | per-draw loss, raw counts: median (range) | '
            'per-draw pattern part, mean: raw counts / calibrated readout |', '|---|---|---:|---:|---:|---:|']
        for (m, what), g in ymod.groupby(['model', 'what'], sort=False):
            gr, gc = g[g.readout == 'raw'], g[g.readout == 'cal']
            pd_ = f"{_f(gr.per_draw_rmse_median.iloc[0], '.3f')} ({_f(gr.per_draw_rmse_min.iloc[0], '.3f')} to {_f(gr.per_draw_rmse_max.iloc[0], '.3f')})" \
                if 'per_draw_rmse_median' in gr.columns and len(gr) and pd.notna(gr.per_draw_rmse_median.iloc[0]) else '-'
            pc = (f"{_f(gr.per_draw_pattern_mean.iloc[0], '.3f')} / {_f(gc.per_draw_pattern_mean.iloc[0], '.3f')}"
                if 'per_draw_pattern_mean' in gr.columns and len(gr) and pd.notna(gr.per_draw_pattern_mean.iloc[0]) else '-')
            rows.append(f"| {g.label.iloc[0]} | {what} | {_f(gr.rmse.iloc[0], '.3f')} / {_f(gr.rmse_centered.iloc[0], '.3f')} | "
                f"{_f(gc.rmse.iloc[0], '.3f')} / {_f(gc.rmse_centered.iloc[0], '.3f')} | {pd_} | {pc} |")
        out.append('\n'.join(rows))
    # A2: wiring by class, the reference models
    wb = a2['wiring_by_class']
    for key in ('ref40', 'ref70'):
        sub = wb[wb.model == key]
        if sub.empty:
            continue
        real = sub[sub.wiring == 'real'].set_index('kc_class')
        sh = sub[(sub.wiring != 'real') & (sub.kc_class == 'survivor')]
        lab = KEY_MODELS[key][0]
        rows = [f"**{lab[0].upper() + lab[1:]}: the KC classes' wiring and activity (means; survivors on the shuffled draws: mean over the 20 "
            f"draws of each draw's survivors; the breadth and the breadth-weighted APL->KC weight on the draws only where the draw's base run "
            f"is archived, seed 2)**", '',
            '| | real wiring {span=4} | shuffled draws {span=1} |',
            f"| statistic | survivors (n = {int(real.loc['survivor', 'n']) if 'survivor' in real.index else 0}) | the rest of the high-firing set "
            f"(n = {int(real.loc['high_firing_other', 'n']) if 'high_firing_other' in real.index else 0}) | other responders | silent KCs | survivors |",
            '|---|---:|---:|---:|---:|---:|']
        for col in FEATURE_COLS:
            cells = [_f(real.loc[c, col], '.2f') if c in real.index else '-' for c in CLASSES]
            rows.append(f"| {FEATURE_LABEL[col]} | " + ' | '.join(cells) + f" | {_f(sh[col].mean(), '.2f')} |")
        types = [c for c in real.columns if c.startswith('type_')]
        if types and 'survivor' in real.index:
            rows.append('| KC types (survivors) | ' + ', '.join(f"{c[5:]} {int(real.loc['survivor', c])}" for c in types if real.loc['survivor', c] > 0)
                + ' | ' + ', '.join(f"{c[5:]} {int(real.loc['high_firing_other', c])}" for c in types if real.loc['high_firing_other', c] > 0) + ' | | | |')
        out.append('\n'.join(rows))
    # A2: class parts
    cp = a2['class_parts']
    if len(cp):
        rows = ['**Each class\'s part of the mean similarity and of the mean APL effect (sums over classes give the model\'s; real wiring)**', '',
            '| model | readout | ' + ' | '.join(f'{CLASS_LABEL[c]}: similarity APL on / off / part of the APL effect' for c in CLASSES) + ' |',
            '|---|---|' + '---:|' * len(CLASSES)]
        for r in cp[cp.wiring == 'real'].itertuples():
            d = r._asdict()
            rows.append(f"| {r.label} | {READOUT_LABEL[r.readout]} | " + ' | '.join(
                f"{_f(d[f'sim_on__{c}'], '.3f')} / {_f(d[f'sim_off__{c}'], '.3f')} / {_f(d[f'effect__{c}'], '+.3f')}" for c in CLASSES) + ' |')
        out.append('\n'.join(rows))
    # block and natural-source alignment with the survivors left out
    bvc = a2['block_vcf']
    if len(bvc):
        rows = ['**The block and the natural-source alignment with the survivors left out (real wiring)**', '',
            '| model | readout | block: all KCs / survivors left out | natural-source alignment, APL on / off / on minus off: all KCs | '
            'the same, survivors left out | survivors\' part of the block pairs\' mean APL effect (share) |', '|---|---|---:|---:|---:|---:|']
        for r in bvc[bvc.wiring == 'real'].itertuples():
            d = r._asdict()
            rows.append(f"| {r.label} | {READOUT_LABEL[r.readout]} | {_f(r.block, '+.3f')} / {_f(d['block__survivors_out'], '+.3f')} | "
                f"{_f(r.vcf_on, '.2f')} / {_f(r.vcf_off, '.2f')} / {_f(r.vcf_on_minus_off, '+.2f')} | {_f(d['vcf_on__survivors_out'], '.2f')} / "
                f"{_f(d['vcf_off__survivors_out'], '.2f')} / {_f(d['vcf_on_minus_off__survivors_out'], '+.2f')} | "
                f"{_f(r.block_pairs_effect_survivors, '+.3f')} ({_f(r.block_pairs_survivor_share, '.2f')}) |")
        rows.append(f"| KC data | | {dt('block'):+.3f} | {dt('ctrl_vcf'):.2f} / {dt('tnt_vcf'):.2f} / {dt('vcf_apl_effect'):+.2f} | | |")
        out.append('\n'.join(rows))
    # worst pairs
    wp = a2['worst_pairs']
    if len(wp):
        rows = ['**The five pairs with the largest pair-specific error (E66\'s worst pairs), each class\'s part of the model\'s APL effect on them, '
            'and the error with the survivors / the rest of the high-firing set left out (real wiring)**', '',
            '| model | readout | pair | model / data effect | error | survivors\' part | rest of the high-firing set\'s part | others\' part | '
            'error without the survivors | error without the rest of the high-firing set |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in wp[wp.wiring == 'real'].itertuples():
            d = r._asdict()
            rows.append(f"| {r.model} | {READOUT_LABEL[r.readout]} | {r.odor_a}-{r.odor_b} | {_f(r.model_effect, '+.3f')} / {_f(r.data_effect, '+.3f')} | "
                f"{_f(r.error, '+.3f')} | {_f(d['part__survivor'], '+.3f')} | {_f(d['part__high_firing_other'], '+.3f')} | "
                f"{_f(d['part__responder'] + d['part__silent'], '+.3f')} | {_f(d['error_without__survivor'], '+.3f')} | "
                f"{_f(d['error_without__high_firing_other'], '+.3f')} |")
        out.append('\n'.join(rows))
    # A3
    oc = a2['opk_classes']
    if len(oc):
        rows = ['**The KC classes against the odors-per-KC groups (share of each class responding to 0 / 1 / 2-3 / 4-7 / 8 or more odors; real wiring)**', '',
            '| model | class | n | APL on | APL off |', '|---|---|---:|---:|---:|']
        for (key, x), g in oc[oc.wiring == 'real'].groupby(['key', 'kc_class'], sort=False):
            on_r, off_r = g[g.condition == 'on'].iloc[0], g[g.condition == 'off'].iloc[0]
            rows.append(f"| {g.label.iloc[0]} | {CLASS_LABEL[x]} | {int(on_r.n)} | " + ' / '.join(_f(on_r[f'opk_{b}'], '.2f') for b in OPK_BINS) + ' | '
                + ' / '.join(_f(off_r[f'opk_{b}'], '.2f') for b in OPK_BINS) + ' |')
        out.append('\n'.join(rows))
    of = a2['opk_features']
    if len(of):
        rows = ['**What separates the KCs silent with APL off from those responding to 8 or more odors with APL off (real wiring; AUC 0.5 = no '
            'separation, above 0.5 = larger in the broadly responding KCs)**', '', '| model | feature | silent: n, mean | 8 or more: n, mean | AUC |',
            '|---|---|---:|---:|---:|']
        for r in of[of.wiring == 'real'].itertuples():
            if r.key not in ('ref40', 'ref70', 'g125', 'kckc20', 'kckc45'):
                continue
            rows.append(f"| {r.label} | {FEATURE_LABEL[r.feature]} | {r.n_silent_off}, {_f(r.mean_silent_off, '.2f')} | {r.n_8plus_off}, "
                f"{_f(r.mean_8plus_off, '.2f')} | {_f(r.auc_8plus_vs_silent, '.2f')} |")
        out.append('\n'.join(rows))
    # A4
    if len(a4):
        rows = ['**Across the archive: the survivors against the APL effect and the realism statistics (Spearman; partial given the APL-off / APL-on '
            'response-rate ratio)**', '', '| operating point | survivors (20 spikes) or their share of the squared APL-on vectors | statistic | n | ρ | '
            'partial ρ given the ratio |', '|---|---|---|---:|---:|---:|']
        for r in a4[a4.x.isin(['n_survivors', 'survivor_share_sq_on'])].itertuples():
            rows.append(f"| {r.operating_point} | {r.x} | {r.y} | {r.n} | {_f(r.spearman, '+.2f')} | {_f(r.partial_spearman_given_ratio, '+.2f')} |")
        out.append('\n'.join(rows))
    md = '\n\n'.join(out) + '\n'
    (res_dir / 'comparison_table.md').write_text(md)
    return md


def part_a_figures(per: pd.DataFrame, lo: pd.DataFrame, yard: pd.DataFrame, ymod: pd.DataFrame, res_dir: Path) -> None:
    plt = H._plt()
    floor = H.data_target('apl_effect_rmse_floor')
    data_mean = float(refs()['effect'].mean())
    # 1. loss and mean effect against k for the key models
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.get_cmap('tab10')
    for i, (key, (label, folder, tag, _)) in enumerate(KEY_MODELS.items()):
        g = lo[(lo.folder == folder) & (lo.tag.astype(str) == tag)]
        for j, readout in enumerate(READOUTS):
            s = g[g.readout == readout].set_index('leave_out')
            if s.empty:
                continue
            ks = [0] + list(K_LEAVE_OUT)
            names = ['none'] + [f'top{k}' for k in K_LEAVE_OUT]
            x = np.array(ks) + 0.5
            for row, col in ((0, 'rmse'), (1, 'mean')):
                ax = axes[row, j]
                ax.plot(x, s.loc[names, col], '-o', ms=3, color=colors(i), label=label)
                if 'survivors' in s.index:
                    ax.scatter([s.loc['survivors', 'n_left_out'] + 0.5], [s.loc['survivors', col]], marker='*', s=90, color=colors(i), zorder=4)
    for j, readout in enumerate(READOUTS):
        for row, (col, lab, ref) in enumerate((('rmse', 'APL-effect loss (lower is better)', floor), ('mean', 'mean APL effect', data_mean))):
            ax = axes[row, j]
            ax.set_xscale('log')
            ax.axhline(ref, color='k', ls='--', lw=1)
            ax.set_xlabel('KCs left out + 0.5 (the highest-firing by APL-on count; star: the survivors)')
            ax.set_ylabel(lab)
            ax.set_title(f'{READOUT_LABEL[readout]}: {lab}', fontsize=10)
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, loc='best')
    fig.suptitle('How few cells the APL-effect loss rests on: the key models with their highest-firing KCs left out after the fact '
        '(dashed: KC data, floor of the loss / mean effect)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(res_dir / 'E70_leave_out.png', dpi=120)
    plt.close(fig)
    # 2. across the archive: the survivors' share against the mean APL effect
    p = per.dropna(subset=['survivor_share_sq_on'])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (col, lab) in zip(axes, (('raw__mean', 'mean APL effect, raw spike counts'), ('cal__mean', 'mean APL effect, calibrated readout'),
            ('raw__max_change_k2', 'largest change of the raw-count loss\nwith up to 2 KCs left out'))):
        for op, c in (('6.8 %', '#1f5fbf'), ('5 %', '#d1662a')):
            q = p[p.operating_point == op]
            ax.scatter(q.survivor_share_sq_on, q[col], s=9, alpha=0.5, color=c, label=f'tuned at {op} ({len(q)} variants)')
        if col != 'raw__max_change_k2':
            ax.axhline(data_mean, color='k', ls='--', lw=1, label='KC data')
        ax.set_xlabel("survivors' share of each odor's squared APL-on response vector")
        ax.set_ylabel(lab); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Across the model-output archive: the survivors (20 or more spikes to some odor with APL on and off) against the mean APL effect',
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(res_dir / 'E70_archive_relations.png', dpi=120)
    plt.close(fig)
    # 3. the yardstick
    if len(yard):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for ax, (col, lab) in zip(axes, (('rmse', 'APL-effect loss'), ('rmse_centered', 'pattern part of the APL-effect loss'))):
            ax.hist(yard[col], bins=25, color='0.6', label='KC data: one control with one APL-output-blocked (TNT) recording\nagainst the mean of the others (88 pairs)')
            if col == 'rmse':
                ax.axvline(floor, color='k', ls='--', lw=1, label='KC data: floor (mean over recordings)')
            if len(ymod):
                for i, (key, g) in enumerate(ymod[(ymod.readout == 'raw') & (ymod.n_draws == 1)].groupby('model', sort=False)):
                    ax.axvline(g[col].iloc[0], color=plt.get_cmap('tab10')(i), lw=1.5, label=f'{g.label.iloc[0]} (raw counts)')
            ax.set_xlabel(lab); ax.set_ylabel('pairs of recordings')
        axes[0].legend(fontsize=6.5)
        fig.suptitle('The single-fly yardstick', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(res_dir / 'E70_yardstick.png', dpi=120)
        plt.close(fig)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()


def part_a(res_dir: Path, jobs: int, only_key: bool = False) -> None:
    t0 = time.time()
    variants = archive_variants()
    if only_key:
        keys = {(f, t) for _, f, t, _ in KEY_MODELS.values()}
        sel = [(f, str(t)) in keys or f in (E47, E67) for f, t in zip(variants.folder, variants.tag)]
        variants_a1 = variants[sel]
    else:
        variants_a1 = variants
    print(f'part A: {len(variants_a1)} variants with archived spike counts', flush=True)
    per = part_a1(variants_a1, res_dir, jobs)
    a2 = part_a2(variants, res_dir)
    a4 = part_a4(per, res_dir)
    yard = yardstick()
    yard.to_csv(res_dir / 'yardstick.csv', index=False)
    ymod = yardstick_models(variants)
    ymod.to_csv(res_dir / 'yardstick_models.csv', index=False)
    lo = pd.read_csv(res_dir / 'leave_out.csv')
    md = part_a_tables(per, a2, a4, yard, ymod, res_dir)
    part_a_figures(per, lo, yard, ymod, res_dir)
    e60 = [f for f in E60_FOLDERS if (ARCHIVE / f).exists()]
    prov_path = res_dir / 'provenance.json'
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov['part_A'] = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E70_high_firing_kcs.py --part A',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), al_analysis_commit=git_commit(),
        al_analysis_dirty=bool(subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()),
        n_variants=len(per), definitions=dict(high_firing=f'>= {HIGH_SPIKES:g} spikes to some odor with APL off (acute)',
        survivors=f'high-firing and >= {HIGH_SPIKES:g} spikes to some odor with APL on', check=f'the same at {CHECK_SPIKES:g} spikes',
        responder=f'>= {RESP_MIN:g} spikes'), leave_out_k=list(K_LEAVE_OUT),
        calibrated_readout='saturating n / (n + K), K = calibrate_readout(APL-on counts of the KCs kept, participation_ratio, responders_min 2) '
            f"to data_target('ctrl_participation_ratio_resp') = {refs()['cal_target']:.4f}",
        data_effect='harness.yang_apl_effect() (control flies minus APL output blocked (TNT), cosine, mean over recordings)',
        key_models={k: v[1:3] for k, v in KEY_MODELS.items()}, e60_archived=e60 or 'not archived yet (rows to be added when it is)',
        e67_archived=sorted(per[per.folder == E67].tag.astype(str).tolist()),
        pseudo_recordings='not scored: the trial-noise pipeline runs saved only the means over pseudo-recordings',
        inputs_sha256={str(RESULTS / 'apl_effect' / 'by_variant.csv'): _sha(RESULTS / 'apl_effect' / 'by_variant.csv'),
            str(ARCHIVE / 'inputs' / 'wPNKC_bouton_base.parquet'): _sha(ARCHIVE / 'inputs' / 'wPNKC_bouton_base.parquet')},
        approx_conc_note=H.APPROX_CONC_NOTE, seconds=round(time.time() - t0))
    prov_path.write_text(json.dumps(prov, indent=1, default=str, ensure_ascii=False) + '\n')
    print(md)
    print(f'part A done in {time.time() - t0:.0f} s -> {res_dir}', flush=True)


# ---------------------------------------------------------------------------- part B / C

def fit_sets(folder: str, tag: str) -> Dict[str, pd.Index]:
    """The survivors and the high-firing set (KC ids) of an archived variant at the definitions."""
    on, off = load_counts(folder, tag)
    c = classify_kcs(on, off, HIGH_SPIKES)
    ids = kc_ids(on)
    return dict(survivors=ids[(c == 'survivor').to_numpy()], high_firing=ids[c.isin(['survivor', 'high_firing_other']).to_numpy()],
        high_firing_other=ids[(c == 'high_firing_other').to_numpy()], all=ids)


def matched_control(target: pd.Index, pool: pd.Index, feat: pd.DataFrame, rng: np.random.Generator, by_claws: bool = True) -> pd.Index:
    """As many KCs from `pool` as in `target`, each matched to one target KC in KC type and (with
    `by_claws`) number of claws (nearest number where none is equal), drawn without replacement."""
    pool = pd.Index(pool)
    types = feat.loc[pool, 'kc_type'].astype(str).to_numpy()
    claws = feat.loc[pool, 'n_claws'].to_numpy(dtype=float)
    free = np.ones(len(pool), dtype=bool)
    out = []
    for kc in rng.permutation(np.asarray(target)):
        t_type, t_claws = str(feat.at[kc, 'kc_type']), float(feat.at[kc, 'n_claws'])
        cand = np.flatnonzero(free & (types == t_type))
        if not len(cand):
            raise ValueError(f'no KC of type {t_type} left in the control pool')
        if by_claws:
            d = np.abs(claws[cand] - t_claws)
            cand = cand[d == d.min()]
        j = cand[rng.integers(len(cand))]
        out.append(pool[j])
        free[j] = False
    return pd.Index(sorted(out))


def build_arm_wirings(mkey: str, sets: Dict[str, pd.Index], seeds: Sequence[int], out_dir: Path, arms: Sequence[str]) -> List[dict]:
    """The re-drawn / restored wirings of one model's arms (cached in out_dir as parquet with their swaps
    and KC sets), with each draw's statistics."""
    from al_analysis.wpnkc_shuffles import redraw_kc_subset, restore_kc_subset, wPNKC_shuffle_summary
    W = real_wiring()
    feat = wiring_features(W)
    surv, hf = sets['survivors'], sets['high_firing']
    out = []
    for arm in arms:
        for seed in seeds:
            name = f'{mkey}_{arm}-seed{seed}'
            path = out_dir / f'wPNKC_{name}.parquet'
            rng = np.random.default_rng(1000 + int(seed))
            base_W = W
            if arm == 'surv_redraw':
                kcs, protect = surv, None
            elif arm == 'hf_redraw':
                kcs, protect = hf, None
            elif arm == 'ctl_hf_nonsurv':
                kcs, protect = matched_control(surv, sets['high_firing_other'], feat, rng), surv
            elif arm == 'ctl_outside':
                kcs, protect = matched_control(surv, sets['all'].difference(hf), feat, rng), hf
            elif arm == 'ctl_outside_hf':
                kcs, protect = matched_control(hf, sets['all'].difference(hf), feat, rng, by_claws=False), hf
            elif arm in ('restore', 'cpwt_untreated'):
                kcs, protect = surv, None
                base_W = shuffled_wiring('claw_permutation_within_type', seed)
            else:
                raise ValueError(arm)
            if path.exists():
                L = pd.read_parquet(path)
                swaps = pd.read_csv(out_dir / f'swaps_{name}.csv')
            else:
                if arm in ('surv_redraw', 'hf_redraw', 'ctl_hf_nonsurv', 'ctl_outside', 'ctl_outside_hf'):
                    L, swaps = redraw_kc_subset(W, kcs, seed=int(seed), within_kc_type=True, protect_kcs=protect)
                elif arm == 'restore':
                    L, swaps = restore_kc_subset(base_W, W, kcs, seed=int(seed), within_kc_type=True)
                else:
                    L, swaps = base_W, pd.DataFrame(columns=['kc', 'partner_kc', 'claw_row', 'partner_row', 'column_before', 'column_after'])
                L.to_parquet(path)
                swaps.to_csv(out_dir / f'swaps_{name}.csv', index=False)
                (out_dir / f'kcs_{name}.json').write_text(json.dumps(dict(kcs=[int(x) for x in kcs],
                    protect=[int(x) for x in protect] if protect is not None else [])))
            s_real = wPNKC_shuffle_summary(W, L)
            moved = int((swaps.column_before != swaps.column_after).sum()) if len(swaps) else 0
            in_set = L.index.get_level_values(H.KC_ID_LEVEL).isin(list(surv))
            out.append(dict(name=name, model=mkey, arm=arm, seed=int(seed), wiring=L, n_kcs_set=len(kcs), n_swaps=len(swaps), n_claws_moved=moved,
                frac_claws_changed_vs_real=float(s_real['frac_claws_changed']), coinput_corr_vs_real=float(s_real['coinput_corr']),
                coinput_resid_corr_vs_real=float(s_real['coinput_resid_corr']), kc_n_claws_preserved=bool(s_real['kc_n_claws_preserved']),
                max_abs_glom_n_claws_change=int(s_real['max_abs_glom_n_claws_change']),
                survivor_claws_as_real=float((L[in_set].to_numpy() == W[in_set].to_numpy()).all(axis=1).mean()),
                set_kcs=','.join(str(int(x)) for x in kcs)))
    return out


def precopy_base(out_dir: Path, folder: str, claw_thr, gain, gain_gloms: Optional[Sequence[str]]) -> None:
    """Copies the fit's own APL-off base run (its breadth) from the archive into the runner's base cache
    (out_dir/partC, under the name `E10_joint_fit.bouton_base` reads), so the real-wiring runs use the
    breadth the fit used and do not simulate it again."""
    import shutil
    name = 'base_acute_off_spike_counts.parquet' if claw_thr is None else f'thr{float(claw_thr):g}_base_acute_off_spike_counts.parquet'
    if gain is not None and float(gain) != 1.0:
        gset = '' if gain_gloms is None or sorted(gain_gloms) == sorted(gained_glomeruli()) else '-'.join(sorted(gain_gloms)) + '_'
        name = f'gain{float(gain):g}_{gset}' + name
    else:
        return   # the base without gain: bouton_base reads hal's saved run from the shared directory
    (out_dir / 'partC').mkdir(exist_ok=True)
    for nm in (name, name.replace('spike_counts', 'responses')):
        if (ARCHIVE / folder / nm).exists() and not (out_dir / 'partC' / nm).exists():
            shutil.copyfile(ARCHIVE / folder / nm, out_dir / 'partC' / nm)


B_ARMS = ('surv_redraw', 'hf_redraw', 'ctl_hf_nonsurv', 'ctl_outside', 'ctl_outside_hf', 'restore')
C_ARMS = ('surv_redraw', 'ctl_hf_nonsurv', 'cpwt_untreated')
APL_FACTORS = (2.0, 4.0)


def part_b_jobs(models: Sequence[str], seeds: Sequence[int], out_dir: Path, res_dir: Path) -> List[dict]:
    """The arms of part B as `E54_gain_and_lesion.run_variants` jobs (the real wiring at the fitted
    threshold first, then the oracle, then the wirings), and wirings.csv."""
    from E47_wiring_features import MODELS
    jobs, wrows = [], []
    for mkey in models:
        spec = MODELS[mkey]
        sets = fit_sets(E47, f'A_{mkey}_real_fit')
        precopy_base(out_dir, spec[0], spec[2], spec[4], None)
        common = dict(model=mkey, spec=spec, gain=spec[4])
        jobs.append(dict(common, tag=f'B_{mkey}_real_fit', part='B', wiring=None, wiring_tag=None, shuffle='real_fit', seed=None, retune=False))
        for fac in APL_FACTORS:
            f = pd.Series(fac, index=sets['survivors'])
            jobs.append(dict(common, tag=f'B_{mkey}_surv_apl_x{fac:g}', part='B', wiring=None, wiring_tag=None, shuffle=f'surv_apl_x{fac:g}', seed=None,
                retune=False, wAPLKC_factor=f))
        for w in build_arm_wirings(mkey, sets, seeds, out_dir, B_ARMS):
            jobs.append(dict(common, tag=f"B_{w['name']}_retuned", part='B', wiring=w['wiring'], wiring_tag=w['name'], shuffle=w['arm'], seed=w['seed']))
            wrows.append({k: v for k, v in w.items() if k != 'wiring'})
    if wrows:
        p = res_dir / 'wirings.csv'
        old = pd.read_csv(p) if p.exists() else pd.DataFrame()
        new = pd.DataFrame(wrows)
        if len(old):
            old = old[~old.name.isin(new.name)]
        pd.concat([old, new], ignore_index=True).to_csv(p, index=False)
    return jobs


def best_e67_fits(res_dir: Path) -> Dict[str, str]:
    """E67's best fit under each definition (lowest APL-effect loss on raw counts, and under the calibrated
    readout, among its archived fits, from this experiment's per_variant.csv)."""
    per = pd.read_csv(res_dir / 'per_variant.csv')
    e = per[per.folder == E67]
    if e.empty:
        return {}
    return {'raw': str(e.loc[e['raw__rmse'].idxmin(), 'tag']), 'cal': str(e.loc[e['cal__rmse'].idxmin(), 'tag'])}


def part_c_check(res_dir: Path) -> pd.DataFrame:
    """Part C's condition: E67's best fit under each definition and its loss change with its survivors left
    out under that definition (part C runs where it is 0.01 or more)."""
    per = pd.read_csv(res_dir / 'per_variant.csv')
    per['tag'] = per['tag'].astype(str)
    per = per.set_index(['folder', 'tag'])
    rows = []
    for ro, tag in best_e67_fits(res_dir).items():
        r = per.loc[(E67, tag)]
        d = float(r[f'{ro}__survivors__rmse'] - r[f'{ro}__rmse'])
        rows.append(dict(definition=ro, tag=tag, loss=float(r[f'{ro}__rmse']), loss_survivors_out=float(r[f'{ro}__survivors__rmse']),
            change=d, n_survivors=int(r.n_survivors), runs_part_c=bool(abs(d) >= 0.01)))
    return pd.DataFrame(rows)


def part_c_jobs(tags: Sequence[str], seeds: Sequence[int], out_dir: Path, res_dir: Path) -> List[dict]:
    """Part C's arms at the scales of E67 fits (`tags`): arm (i), the first control and the untreated
    within-type claw-permutation draws, plus the fit's real wiring at its fitted threshold."""
    fits = pd.read_csv(RESULTS / E67 / 'fits.csv').set_index('tag')
    jobs, wrows = [], []
    for tag in tags:
        r = fits.loc[tag]
        thr = None if pd.isna(r.claw_thr) else float(r.claw_thr)
        gain = None if pd.isna(r.gain) else float(r.gain)
        gg = [x.strip() for x in str(r.gain_glomeruli).split(',')] if isinstance(r.gain_glomeruli, str) and r.gain_glomeruli.strip() else None
        spec = (E67, tag, thr, float(r.k), gain)
        mkey = f'e67-{tag}'
        sets = fit_sets(E67, tag)
        precopy_base(out_dir, E67, thr, gain, gg)
        common = dict(model=mkey, spec=spec, gain=gain, gain_gloms=gg)
        jobs.append(dict(common, tag=f'C_{mkey}_real_fit', part='C', wiring=None, wiring_tag=None, shuffle='real_fit', seed=None, retune=False))
        for w in build_arm_wirings(mkey, sets, seeds, out_dir, C_ARMS):
            jobs.append(dict(common, tag=f"C_{w['name']}_retuned", part='C', wiring=w['wiring'], wiring_tag=w['name'], shuffle=w['arm'], seed=w['seed']))
            wrows.append({k: v for k, v in w.items() if k != 'wiring'})
    if wrows:
        p = res_dir / 'wirings.csv'
        old = pd.read_csv(p) if p.exists() else pd.DataFrame()
        new = pd.DataFrame(wrows)
        if len(old):
            old = old[~old.name.isin(new.name)]
        pd.concat([old, new], ignore_index=True).to_csv(p, index=False)
    return jobs


def score_run(on: pd.DataFrame, off: pd.DataFrame) -> dict:
    """Part B / C's readouts of one run beyond the harness's: the APL-effect loss with its parts on raw
    counts and under the calibrated readout, the new wiring's own high-firing set and survivors and the
    loss with them left out, the largest counts, odors per KC against the bands, the block and the
    natural-source alignment under the readout."""
    from apl_effect_scores import opk_data, opk_band
    row = {}
    c = classify_kcs(on, off, HIGH_SPIKES)
    keep = (c != 'survivor').to_numpy()
    row.update(n_high_firing=int(c.isin(['survivor', 'high_firing_other']).sum()), n_survivors=int((c == 'survivor').sum()),
        survivor_share_sq_on=sq_share(on, c == 'survivor'), max_count_on=float(on.to_numpy().max()), max_count_off=float(off.to_numpy().max()))
    for ro in READOUTS:
        st = score(on, off, ro)
        row.update({f'{ro}__{k}': v for k, v in st.items()})
        sw = score(on[keep], off[keep], ro)
        row.update({f'{ro}__survivors_out__{k}': sw[k] for k in ('rmse', 'mean', 'rmse_centered')})
        row.update({f'{ro}__{k}': v for k, v in block_vcf(on, off, ro).items()})
    for cond, m in (('on', on), ('off', off)):
        d = opk_distribution(m)
        row.update({f'opk_{cond}_{b}': float(v) for b, v in zip(OPK_BINS, d)})
        row[f'opk_{cond}_tv'] = float(0.5 * np.abs(d - opk_data(cond)).sum())
        row[f'opk_{cond}_band'] = opk_band(cond)
    return row


B_GROUP_LABEL = {'real_fit': 'real wiring, fitted threshold', 'surv_redraw': '(i) survivors re-drawn', 'hf_redraw': '(ii) high-firing set re-drawn',
    'ctl_hf_nonsurv': '(iii) matched high-firing non-survivors re-drawn', 'ctl_outside': '(iii) matched KCs outside the high-firing set re-drawn',
    'ctl_outside_hf': '(iii) as many KCs outside the set (type-matched) re-drawn', 'restore': '(iv) survivors\' real claws into within-type claw permutations',
    'cpwt_untreated': 'within-type claw permutations, untreated', 'surv_apl_x2': '(v) survivors\' APL->KC weights x2', 'surv_apl_x4': '(v) survivors\' APL->KC weights x4'}


def summary_b(res_dir: Path, out_dir: Path) -> None:
    """partB_scores.csv (one row per run: stats_variants' harness columns plus `score_run`'s), the check of the
    real-wiring run against the archive, and part B's tables in comparison_table_B.md."""
    from apl_effect_scores import realism_flag
    sp = res_dir / 'stats_variants.csv'
    if not sp.exists():
        print('no part B / C rows yet'); return
    st = pd.read_csv(sp)
    if 'error' in st.columns:
        st = st[~(st.error.astype(str).str.len() > 3)]
    rows = []
    for r in st.itertuples():
        p_on = out_dir / f'{r.tag}_on_spike_counts.parquet'
        p_off = out_dir / f'{r.tag}_acute_off_spike_counts.parquet'
        if not (p_on.exists() and p_off.exists()):
            continue
        on = H.abbrev_columns(pd.read_parquet(p_on).astype(float)); off = H.abbrev_columns(pd.read_parquet(p_off).astype(float))
        o = H.odor_order(on.columns)
        row = dict(tag=r.tag, part=r.part, model=r.model, arm=r.shuffle, seed=r.seed, rate_on=r.rate_on, rate_off=r.rate_off,
            rate_ratio_off_over_on=r.rate_off / r.rate_on, spearman_rate_on_vs_data=getattr(r, 'spearman_rate_on_vs_data', np.nan),
            spearman_rate_off_vs_data=getattr(r, 'spearman_rate_off_vs_data', np.nan), thr_factor_vs_fit=r.thr_factor_vs_fit, **score_run(on[o], off[o]))
        row['realism_flag'] = realism_flag(row['rate_on'], row['rate_off'], row['spearman_rate_on_vs_data'], row['opk_on_tv'], row['opk_off_tv'])
        if r.shuffle == 'real_fit':
            mk = str(r.model)
            ref = (E47, f'A_{mk}_real_fit') if not mk.startswith('e67-') else (E67, mk[4:])
            try:
                a_on, a_off = load_counts(*ref)
                row['max_abs_diff_vs_archive'] = float(max(np.abs(a_on.to_numpy() - on[o].to_numpy()).max(), np.abs(a_off.to_numpy() - off[o].to_numpy()).max()))
            except Exception as e:
                row['max_abs_diff_vs_archive'] = repr(e)
        rows.append(row)
    df = pd.DataFrame(rows)
    w = res_dir / 'wirings.csv'
    if w.exists():
        wr = pd.read_csv(w).drop(columns=['wiring', 'set_kcs'], errors='ignore')
        df['name'] = df.tag.str.replace(r'^[BC]_', '', regex=True).str.replace('_retuned$', '', regex=True)
        df = df.merge(wr.drop(columns=['model', 'arm', 'seed'], errors='ignore'), on='name', how='left')
    df.to_csv(res_dir / 'partB_scores.csv', index=False)
    out = []
    data_mean = float(refs()['effect'].mean()); floor = H.data_target('apl_effect_rmse_floor')
    stat_rows = [('raw__rmse', 'APL-effect loss, raw counts', '.3f', f'floor {floor:.3f}'), ('raw__mean', 'mean APL effect, raw counts', '+.3f', f'{data_mean:+.3f}'),
        ('raw__rmse_centered', 'pattern part, raw counts', '.3f', ''), ('raw__per_odor', 'per-odor part, raw counts', '.3f', ''),
        ('raw__pair_specific', 'pair-specific part, raw counts', '.3f', ''), ('raw__spearman', 'ρ on the per-pair APL effect, raw counts', '.2f', ''),
        ('cal__rmse', 'APL-effect loss, calibrated readout', '.3f', f'floor {floor:.3f}'), ('cal__mean', 'mean APL effect, calibrated readout', '+.3f', f'{data_mean:+.3f}'),
        ('cal__rmse_centered', 'pattern part, calibrated readout', '.3f', ''), ('cal__K', 'K of the calibrated readout', '.1f', ''),
        ('raw__mean_dist_on', 'mean distance, APL on (raw counts)', '.3f', f"{H.data_target('ctrl_mean_dist'):.3f}"),
        ('raw__mean_dist_off', 'mean distance, APL off (raw counts)', '.3f', f"{H.data_target('tnt_mean_dist'):.3f}"),
        ('n_high_firing', 'high-firing KCs (the run\'s own)', '.0f', ''), ('n_survivors', 'survivors (the run\'s own)', '.0f', ''),
        ('survivor_share_sq_on', 'survivors\' share of the squared APL-on vectors', '.3f', ''),
        ('raw__survivors_out__rmse', 'loss with the run\'s survivors left out, raw counts', '.3f', ''),
        ('raw__survivors_out__mean', 'mean APL effect with them left out, raw counts', '+.3f', ''),
        ('max_count_on', 'largest spike count, APL on', '.0f', ''), ('max_count_off', 'largest spike count, APL off', '.0f', ''),
        ('opk_on_tv', 'odors per KC, APL on: distance from the flies (band)', '.2f', f"{H.data_target('ctrl_odors_per_kc_tv_max'):.2f}"),
        ('opk_off_tv', 'odors per KC, APL off: distance from the flies (band)', '.2f', f"{H.data_target('tnt_odors_per_kc_tv_max'):.2f}"),
        ('rate_on', 'response rate, APL on', '.4f', f"{H.data_target('ctrl_frac'):.4f}"), ('rate_ratio_off_over_on', 'response rate, APL off / APL on', '.2f', ''),
        ('spearman_rate_on_vs_data', 'response rate ρ, APL on', '.2f', ''), ('spearman_rate_off_vs_data', 'response rate ρ, APL off', '.2f', ''),
        ('raw__block', 'fruity × small-ester block, raw counts', '+.3f', f"{H.data_target('block'):+.3f}"), ('cal__block', 'fruity × small-ester block, calibrated readout', '+.3f', ''),
        ('raw__rho_on', 'ρ on odor-odor distances, APL on', '.2f', f"ceiling {H.data_target('ctrl_ceiling'):.2f}"),
        ('raw__rho_off', 'ρ on odor-odor distances, APL off', '.2f', f"ceiling {H.data_target('tnt_ceiling'):.2f}"),
        ('raw__vcf_on', 'natural-source alignment, APL on', '.2f', f"{H.data_target('ctrl_vcf'):.2f}"),
        ('raw__vcf_off', 'natural-source alignment, APL off', '.2f', f"{H.data_target('tnt_vcf'):.2f}"),
        ('raw__vcf_on_minus_off', 'natural-source alignment, APL on minus off', '+.2f', f"{H.data_target('vcf_apl_effect'):+.2f}"),
        ('thr_factor_vs_fit', 'somatic threshold (× the fitted one)', '.3f', ''), ('n_claws_moved', 'claws moved', '.0f', ''),
        ('coinput_corr_vs_real', 'co-input correlation with the real wiring', '.3f', '')]
    for mk, g in df.groupby('model', sort=False):
        arms = [a for a in B_GROUP_LABEL if (g.arm == a).any()]
        rows_ = [f'**{mk}: the arms at the fitted scales (mean ± sd over seeds)**', '', '| statistic | KC data | ' + ' | '.join(
            f'{B_GROUP_LABEL[a]} (n = {int((g.arm == a).sum())})' for a in arms) + ' |', '|---|---:|' + '---:|' * len(arms)]
        for col, lab, fmt, dv in stat_rows:
            if col not in g.columns:
                continue
            cells = []
            for a in arms:
                v = pd.to_numeric(g[g.arm == a][col], errors='coerce').dropna()
                cells.append(f'{format(v.mean(), fmt)} ± {format(v.std(ddof=1), fmt.lstrip("+"))}' if len(v) > 1 else (format(float(v.iloc[0]), fmt) if len(v) else '-'))
            rows_.append(f'| {lab} | {dv} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows_))
    md = '\n\n'.join(out) + '\n'
    (res_dir / 'comparison_table_B.md').write_text(md)
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=['A', 'B', 'C'], default='A')
    ap.add_argument('--models', nargs='+', default=['thr40_gain1.5', 'thr70_gain1.5'], help="part B: E47's model keys")
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--fit-tags', nargs='+', default=None, help="part C: E67 fit tags (default: those part A's check selects)")
    ap.add_argument('--jobs', type=int, default=2, help='part A: worker processes for A1')
    ap.add_argument('--only-key', action='store_true', help='part A: A1 on the key models, E47 and E67 only (a quick check)')
    ap.add_argument('--plan', action='store_true', help='parts B / C: build the wirings and the job list only, no simulation')
    ap.add_argument('--summary', action='store_true', help='parts B / C: only (re)write their tables')
    ap.add_argument('--local-test', metavar='DIR', default=None,
        help='write results and outputs under DIR instead of the shared / outputs folders (a smoke test)')
    args = ap.parse_args()
    if args.local_test:
        res_dir = Path(args.local_test) / 'results'; out_dir = Path(args.local_test) / 'outputs'
        res_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    else:
        res_dir = H.results_dir(EXPERIMENT)
        out_dir = H.outputs_dir(EXPERIMENT)
    if args.part == 'A':
        part_a(res_dir, args.jobs, args.only_key)
        return
    if args.summary:
        summary_b(res_dir, out_dir)
        return
    if args.part == 'B':
        jobs = part_b_jobs(args.models, args.seeds, out_dir, res_dir)
    else:
        chk = part_c_check(res_dir)
        print(chk.to_string(index=False))
        tags = args.fit_tags if args.fit_tags else chk[chk.runs_part_c].tag.unique().tolist()
        if not tags:
            print('part C: no E67 fit moves by 0.01 or more with its survivors left out; nothing to run')
            return
        jobs = part_c_jobs(tags, args.seeds, out_dir, res_dir)
    pd.set_option('display.width', 250)
    print(f'{len(jobs)} runs:', flush=True)
    for j in jobs:
        print(f"  {j['tag']}: {'fitted threshold' if j.get('retune') is False else 'threshold re-tuned'}"
            f"{', APL->KC x' + format(float(j['wAPLKC_factor'].iloc[0]), 'g') + ' on ' + str(len(j['wAPLKC_factor'])) + ' KCs' if j.get('wAPLKC_factor') is not None else ''}",
            flush=True)
    w = res_dir / 'wirings.csv'
    if w.exists():
        print(pd.read_csv(w).drop(columns=['set_kcs'], errors='ignore').to_string(index=False))
    if args.plan:
        return
    from E54_gain_and_lesion import run_variants
    run_variants(jobs, res_dir, out_dir)
    # run_variants records its provenance under 'runs'; keep each invocation's under its own key
    prov_path = res_dir / 'provenance.json'
    prov = json.loads(prov_path.read_text())
    key = f"runs_part{args.part}_" + '_'.join(args.models if args.part == 'B' else [j['model'] for j in jobs[:1]])
    prov[key] = prov.pop('runs')
    prov_path.write_text(json.dumps(prov, indent=1, default=str, ensure_ascii=False) + '\n')
    summary_b(res_dir, out_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
