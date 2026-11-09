#!/usr/bin/env python3
"""E55: a margin proxy for the block through the noise.

For every model that has been run through the trial-noise pipeline (E43 part B at its APL-on-matched
level: results/E*/partB/), the deterministic per-KC x odor threshold margins by E51 part A's method
(`E51_block_carriers.simulate_model`'s reconstruction of each KC's drive from the deterministic run and
its critical threshold factor f*, the factor on its threshold at which its response to an odor appears or
disappears), and from the same reconstructed drives a prediction of what the pipeline measures without
simulating the network's response to the jitter:

- The replay model. Each KC's drive from the deterministic run (its summed claw input after the per-claw
  knee and the APL inhibition, minus the KC-KC inhibition from the other KCs' deterministic spikes; its own
  self-inhibition replayed with its own spikes) is replayed through olfsysm's integration at the pipeline's
  own jittered thresholds: the same draws (seed 43, kind 'rec', one standard normal per KC and trial, the
  draws the pipeline used for its pseudo-recordings of this model) and the re-tuned fixed_thr of the matched
  level, thr0 (1 + sigma xi) clipped below at 0.01 thr0 as the pipeline does. So the replay model is the
  pipeline with the rest of the network (the other KCs' spikes, APL, the PN layer) held as in the
  deterministic run: what the margins alone say the jitter does. Its trials are processed exactly as the
  pipeline processes model trials (E43's `recordings` / `summarize`: ten pseudo-recordings of three trials,
  the single-trial floors, the raw pair distances as the mean over recordings of the three-trial-mean cosine
  distances, the grand mean over all trials; the isotonic excess of the nine block pairs and the block).
- The margin statistics. Each entry's probability of changing on a jittered trial, the rest of the network
  held fixed, Phi((f* / ratio - 1) / sigma) for a non-response and one minus that for a response (ratio the
  re-tuned mean threshold over the deterministic one), the fraction of responses within one sigma of
  threshold, and the APL-made carriers' return probability (E51 part A's classes).

Then the comparison with the pipeline's own rows for the same models (the block through the noise as
three-trial means and as the grand mean, the floors, the per-trial response rates, the mean distances, the
per-pair excess) across models and per pair, with Spearman rho, bias and mean absolute error, and the cost
of the margin computation per model (what a margin term in the loss would add to an evaluation).

Models: every model with a matched level in a results/E*/partB/onoff_noise.csv ('match_on', or
'match_on+match_off' where the two matched sigmas coincided), the first table that names a model giving its
sigma, re-tuned threshold and draws; ref40 and base70 are E43's own, every other name 'E<n>-TAG[_ev<k>]' is
the fits.csv row TAG of results/E<n>_*/ (or evaluation k of its history), registered with E43's
`model_from`. --models restricts the set.

Outputs: <shared>/results/E55_margin_proxy/ (per_model.csv, per_pair.csv, margin_stats.csv,
comparison_table.md, agreement.csv, E55_predicted_vs_observed.png, E55_pairs.png, provenance.json);
./outputs/E55_margin_proxy/ (per model: <model>_kc_odor.parquet, the per-KC x odor x condition table with
the margins and change probabilities; <model>_replay.npz, the replay model's trials; <model>_checks.json).
With the cached files present a model is not rebuilt (--resim rebuilds).
Usage: OMP_NUM_THREADS=12 nice -n 10 ./venv/bin/python E55_margin_proxy.py [--models NAME ...] [--seed 43]
  [--resim] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

import kc_remapping_harness as H
import noise_floor_trial_distances as NF
import E43_model_trial_noise as E43
import E51_block_carriers as E51C

EXPERIMENT = 'E55_margin_proxy'
RES = H.shared_dir() / 'results'
ODORS, CONDS, BLOCK_PAIRS = E51C.ODORS, E51C.CONDS, E51C.BLOCK_PAIRS
COND_LABEL = E51C.COND_LABEL
HOWS = ('graded', 'thresholded')
MATCHED = ('match_on', 'match_on+match_off')
MATRICES = ('raw', 'grand')
OWN_MODELS = ('ref40', 'base70')


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# the models and the pipeline's rows
def matched_level(df: pd.DataFrame, model: str):
    lv = set(df.loc[df['model'] == model, 'level'].unique())
    for cand in MATCHED:
        if cand in lv:
            return cand
    return None


def pipeline_models() -> dict:
    """{model: info} for every model with a matched level in a part-B table: the part-B folder,
    the level, sigma, the re-tuned and deterministic fixed_thr, the number of pseudo-recordings and
    trials (the first table naming a model wins; a model re-run elsewhere has the same rows)."""
    out = {}
    for f in sorted(RES.glob('E*/partB/onoff_noise.csv')):
        lv_path = f.parent / 'noise_levels.json'
        if not lv_path.exists():
            continue
        df = pd.read_csv(f)
        levels = json.loads(lv_path.read_text())
        for m in df['model'].unique():
            if m == 'data' or m in out or m not in levels:
                continue
            level = matched_level(df, m)
            if level is None:
                continue
            l = levels[m]
            out[m] = dict(partb=f.parent, partb_experiment=f.parent.parent.name, level=level,
                sigma=float(l['levels'][level]), tuned_fixed_thr=float(l['tuned_fixed_thr'][level]),
                fixed_thr_det=float(l['fixed_thr_det']), n_recordings=int(l.get('n_recordings', 10)),
                n_trials=int(l.get('n_trials', 3)), seed=int(l.get('seed', 43)))
    return out


def register(name: str) -> dict:
    """Makes E43 know the model: ref40 / base70 are its own; 'E<n>-TAG[_ev<k>]' is the fits.csv row
    TAG of results/E<n>_*/ (evaluation k of its history with _ev<k>). Returns the fit's source."""
    if name in OWN_MODELS:
        return dict(fit_experiment=None, fit_tag=None, fit_eval=None)
    m = re.fullmatch(r'(E\d+)-(.+?)(?:_ev(\d+))?', name)
    if not m:
        raise ValueError(f'cannot place {name!r}: expected E<n>-TAG or E<n>-TAG_ev<k>')
    folders = [d for d in sorted(RES.glob(f'{m.group(1)}_*')) if (d / 'fits.csv').exists()]
    if len(folders) != 1:
        raise FileNotFoundError(f'{name}: {len(folders)} results folders {m.group(1)}_* with a fits.csv')
    spec = f'{folders[0].name}:{m.group(2)}' + ('' if m.group(3) is None else f'@{m.group(3)}')
    got = E43.model_from(spec)
    if got != name:
        raise RuntimeError(f'{spec} registers as {got}, not {name}')
    return dict(fit_experiment=folders[0].name, fit_tag=m.group(2), fit_eval=None if m.group(3) is None else int(m.group(3)))


def label_of(name: str) -> str:
    """A short description of a model from its name (the notes name experiments and then the ID)."""
    if name == 'ref40':
        return 'the reference model at 5 % (E34; eight at 1.5, 40 Hz)'
    if name == 'base70':
        return 'the bouton base at 70 Hz, 5 %, no gain (E16)'
    exp, _, tag = name.partition('-')
    thr = re.search(r'thr(\d+)', tag)
    gain = re.search(r'gain([\d.]+)', tag)
    parts = [f'eight at {gain.group(1)}' if gain else 'no gain', f'{thr.group(1)} Hz' if thr else '']
    base = ', '.join(p for p in parts if p)
    if exp in ('E44', 'E45'):
        kind = 'deterministic fit at 6.8 %'
    elif exp == 'E48':
        b = re.search(r'block([\d.]+)', tag)
        kind = f'target +{float(b.group(1)):.3f}'.replace('+0.100', '+0.10') + (', exact start' if tag.endswith('_exact') else '')
    elif exp == 'E51':
        ev = re.search(r'_ev(\d+)$', tag)
        kind = 'noise-aware fit' + (f', vertex (evaluation {ev.group(1)})' if ev else '')
    elif exp == 'E53':
        kk = re.search(r'kckc([\d.]+)', tag)
        kind = f'KC-KC held at {kk.group(1)}' if kk else 'KC-KC held'
    else:
        kind = exp
    return f'{base}, {kind} ({exp})'


def pair_excess(pairs: pd.DataFrame, col: str) -> pd.Series:
    """Per pair, the APL-on distance's residual from the isotonic fit on the APL-off distance."""
    def series(p):
        s = p.set_index(['odor_a', 'odor_b'])[col]
        s.index.names = [None, None]
        return s
    on, off = series(pairs[pairs['condition'] == 'on']), series(pairs[pairs['condition'] == 'acute_off'])
    resid, _ = H.isotonic_resid(on, off)
    return resid.reindex(BLOCK_PAIRS)


def observed(name: str, info: dict, inputs: dict) -> dict:
    """The pipeline's rows for the model at its matched level (and its deterministic block)."""
    tabs = {k: info['partb'] / f'{k}.csv' for k in ('stats_noise', 'onoff_noise', 'pairs_noise_long')}
    for p in tabs.values():
        inputs[str(p)] = sha256(p)
    st, on, pl = (pd.read_csv(p) for p in tabs.values())
    level = info['level']
    out = {}
    # (bracket access throughout: 'transform' is also a DataFrame method)
    for how in HOWS:
        for cond in CONDS:
            r = st[(st['model'] == name) & (st['level'] == level) & (st['condition'] == cond) & (st['transform'] == how)]
            assert len(r) == 1, (name, level, cond, how, len(r))
            r = r.iloc[0]
            out[(how, cond)] = dict(floor1=float(r['single_trial_floor']), floor_mean=float(r['trial_mean_floor']),
                rate_trial=float(r['rate_trial']), rate_det=float(r['rate_det']), mean_dist_raw=float(r['mean_dist_raw']),
                mean_dist_grand=float(r['mean_dist_grand']), mean_dist_det=float(r['mean_dist_det']))
        for matrix in MATRICES:
            r = on[(on['model'] == name) & (on['level'] == level) & (on['matrix'] == matrix) & (on['transform'] == how)]
            assert len(r) == 1, (name, level, matrix, how, len(r))
            out[(how, matrix, 'block')] = float(r.iloc[0]['block'])
        r = on[(on['model'] == name) & (on['level'] == 'deterministic') & (on['matrix'] == 'deterministic') & (on['transform'] == how)]
        assert len(r) == 1, (name, how, len(r))
        out[(how, 'det', 'block')] = float(r.iloc[0]['block'])
        p = pl[(pl['model'] == name) & (pl['level'] == level) & (pl['transform'] == how)]
        for col in ('raw', 'grand', 'det'):
            out[(how, col, 'pairs_excess')] = pair_excess(p, col)
    return out


# ---------------------------------------------------------------------------
# the margins and the replay model
def simulate(name: str, info: dict, orn: pd.DataFrame, loc: Path) -> tuple:
    """Builds the model (E43's ExcitabilityModel: the pipeline's class), checks its deterministic run,
    reconstructs each KC's drive per odor and condition (E51 part A), finds its critical threshold
    factor, and replays the drive at the pipeline's own jittered thresholds. Returns (per-KC x odor x
    condition table, {rep: trial arrays}, checks)."""
    from scipy.signal import lfilter
    from al_analysis.mb_model import get_time_index
    t0 = time.time()
    m = E43.ExcitabilityModel(name, orn, loc)
    checks = dict(model=name, label=label_of(name), build_seconds=time.time() - t0, **m.info)
    # the deterministic run against the fit's saved outputs where they exist (a history evaluation has none)
    saved = {}
    for cond, (rp, sp) in m.saved.items():
        if rp.exists() and sp.exists():
            pr, ps = E43.to_arrays(pd.read_parquet(rp), pd.read_parquet(sp))
            saved[cond] = dict(file=str(rp), responses_mismatch=int((pr != m.det[cond][0]).sum()),
                spike_counts_mismatch=int((ps != m.det[cond][1]).sum()))
        else:
            saved[cond] = dict(file=str(rp), missing=True)
    checks['det_vs_saved'] = saved
    bad = [c for c, v in saved.items() if v.get('responses_mismatch') or v.get('spike_counts_mismatch')]
    assert not bad, f'{name}: the deterministic run differs from the saved outputs ({saved})'
    checks['rate_det'] = {c: float(m.det[c][0].mean()) for c in CONDS}
    if m.fit_rates is not None:
        checks['rate_det_minus_fits_csv'] = {c: checks['rate_det'][c] - r for c, r in zip(CONDS, m.fit_rates)}
        assert all(abs(d) < 1e-9 for d in checks['rate_det_minus_fits_csv'].values()), checks['rate_det_minus_fits_csv']
    print(f'{name}: built in {checks["build_seconds"]:.0f}s; deterministic vs saved {saved}', flush=True)
    mp, rv = m.mp, m.rv
    if float(mp.kc.ves_p) != 0.0:
        raise NotImplementedError('KC vesicle depletion (mp.kc.ves_p != 0)')
    if not bool(mp.kc.pn_claw_to_apl):
        raise NotImplementedError('APL driven by KC spikes (pn_claw_to_apl False): the no-spike run would change APL')
    taum, dt = float(mp.kc.taum), float(mp.time_dt)
    k = dt / taum
    kckc = bool(mp.kc.use_kc_kc_inh)
    tau = float(mp.kc.kc_kc_tau) if kckc else 1.0
    decay = 1.0 - dt / tau
    W = rv.kc.wKCKC.tocsr() if kckc else None
    w_self = W.diagonal().copy() if kckc else np.zeros(m.n_kcs)
    t = np.asarray(get_time_index(mp, rv), dtype=float)
    i0 = int(np.searchsorted(t, mp.time_start - dt / 2)) - 2
    n_resp = m.n_spikes_for_response
    checks.update(taum=taum, dt=dt, n_steps=int(len(t)), kc_kc_inh=kckc, kc_kc_tau=tau, n_spikes_for_response=n_resp)
    # the pipeline's own draws and jittered thresholds for its pseudo-recordings of this model
    sigma, f_tuned = info['sigma'], info['tuned_fixed_thr']
    n_rec, n_tr, seed = info['n_recordings'], info['n_trials'], info['seed']
    assert abs(m.f0 - info['fixed_thr_det']) < 1e-6, (m.f0, info['fixed_thr_det'])
    thr_list, n_clipped = [], 0
    for rep in range(n_rec):
        xis = E43.b_draws(seed, name, m.draw_kinds['rec'], rep, n_tr, m.draw_size)
        for xi in xis:
            th, nc, _ = m.thr_trial(f_tuned, sigma, xi)
            thr_list.append(th.ravel())
            n_clipped += nc
    checks.update(replay=dict(seed=seed, sigma=sigma, tuned_fixed_thr=f_tuned, n_recordings=n_rec, n_trials=n_tr,
        n_clipped=int(n_clipped), draws='E43.b_draws(seed, model, "rec", rep, n_trials, n_kcs): the pipeline\'s own'))
    spk = {cond: np.zeros((n_rec * n_tr, m.n_kcs, len(ODORS)), dtype=np.int32) for cond in CONDS}
    rows = []
    t_margin = t_replay = 0.0
    mp.kc.save_vm_sims = True
    try:
        for cond, p in m.params.items():
            t1 = time.time()
            r0, s0 = E43.to_arrays(*m.sim(dict(p, fixed_thr=1e12)))
            assert s0.sum() == 0
            vms = rv.kc.vm_sims
            if len(vms) != len(m.pn_idx) or vms[0].shape[1] != len(t):
                raise RuntimeError(f'unexpected vm_sims layout: {len(vms)} x {vms[0].shape}')
            d_free = {}
            for j, o in enumerate(ODORS):
                i = m.pn_idx[j]
                V = np.asarray(vms[i])
                vms[i] = None
                assert not V[:, :i0].any(), 'Vm nonzero before the KC loop'
                d_free[o] = E51C.drive_from_vm(V[:, i0:], taum, dt)
            del vms
            r, s = E43.to_arrays(*m.sim(p))
            mism = int((r != m.det[cond][0]).sum() + (s != m.det[cond][1]).sum())
            checks[f'vm_run_mismatch_vs_det_{cond}'] = mism
            assert mism == 0, f'{name} {cond}: the run with the voltages saved differs from the deterministic run ({mism})'
            thr = np.asarray(rv.kc.thr, dtype=float).ravel()
            vms = rv.kc.vm_sims
            n_bad_spikes, max_dev, n_bad_replay = 0, 0.0, 0
            for j, o in enumerate(ODORS):
                i = m.pn_idx[j]
                V = np.asarray(vms[i])[:, i0:]
                vms[i] = None
                prev, cur = V[:, :-1], V[:, 1:]
                spike = (cur == 0) & (prev != 0)
                n_bad_spikes += int((spike.sum(axis=1) != s[:, j]).sum())
                D = d_free.pop(o)
                if kckc:
                    a = lfilter([0.0, 1.0 / tau], [1.0, -decay], spike.astype(float), axis=1)
                    D = D - W @ a
                obs = ~spike & ~((cur == 0) & (prev == 0))
                dev = np.abs(D - E51C.drive_from_vm(V, taum, dt))[obs]
                max_dev = max(max_dev, float(dev.max() / thr.min()) if dev.size else 0.0)
                D_ext = D + w_self[:, None] * a if kckc else D
                n1 = E51C.replay_counts(D_ext, w_self, thr, k, decay, tau)
                n_bad_replay += int((n1 != s[:, j]).sum())
                t2 = time.time()
                f_crit = E51C.critical_factor(D_ext, w_self, thr, n_resp, k, decay, tau)
                t_margin += time.time() - t2
                t2 = time.time()
                for q, th in enumerate(thr_list):
                    spk[cond][q, :, j] = E51C.replay_counts(D_ext, w_self, th, k, decay, tau)
                t_replay += time.time() - t2
                rows.append(pd.DataFrame(dict(model=name, condition=cond, kc=np.arange(len(thr)), odor=o,
                    spikes=s[:, j].astype(int), response=r[:, j], thr=thr, crit_factor=f_crit, replay_spikes=n1)))
            del vms
            checks[f'n_kc_odor_spike_count_from_vm_resets_mismatch_{cond}'] = n_bad_spikes
            checks[f'drive_reconstruction_max_abs_dev_over_thr_{cond}'] = max_dev
            checks[f'n_kc_odor_replay_spike_count_mismatch_{cond}'] = n_bad_replay
            assert n_bad_spikes == 0, f'{name} {cond}: spike counts from the Vm resets differ from the model\'s in {n_bad_spikes} entries'
            assert max_dev < 1e-6, f'{name} {cond}: the reconstructed drive deviates from the Vm-derived one by {max_dev} x thr'
            assert n_bad_replay <= 5, f'{name} {cond}: the replay at the model\'s thresholds misses {n_bad_replay} spike counts'
            print(f'{name}: {COND_LABEL[cond]} reconstructed, margins and {len(thr_list)} replayed trials in {time.time() - t1:.0f}s '
                f'(resets vs spikes: {n_bad_spikes} mismatches; drive deviation {max_dev:.2g} x thr; replay at f = 1: '
                f'{n_bad_replay} mismatches)', flush=True)
    finally:
        mp.kc.save_vm_sims = False
    r, s = E43.to_arrays(*m.sim(m.params['on']))
    checks['after_mismatch_on'] = int((r != m.det['on'][0]).sum() + (s != m.det['on'][1]).sum())
    assert checks['after_mismatch_on'] == 0
    m.close()
    checks.update(margin_seconds=t_margin, replay_seconds=t_replay, total_seconds=time.time() - t0)
    kc = pd.concat(rows, ignore_index=True)
    kc['margin'] = kc['crit_factor'] - 1
    ratio = f_tuned / info['fixed_thr_det']
    kc['p_resp_trial'] = sstats.norm.cdf((kc['crit_factor'] / ratio - 1) / sigma)
    kc['p_change'] = np.where(kc['response'], 1 - kc['p_resp_trial'], kc['p_resp_trial'])
    kc['within_sigma'] = np.abs(kc['crit_factor'] / ratio - 1) < sigma
    checks['thr_ratio'] = ratio
    checks['n_kc_odor_margin_sign_vs_response_mismatch'] = int((kc['response'] != (kc['margin'] > 0)).sum())
    trials = {}
    for rep in range(n_rec):
        sl = slice(rep * n_tr, (rep + 1) * n_tr)
        a = {}
        for cond in CONDS:
            a[f'{cond}_spk'] = spk[cond][sl]
            a[f'{cond}_resp'] = spk[cond][sl] >= n_resp
        trials[rep] = a
    return kc, trials, checks


def process(trials: dict, n_resp: int) -> dict:
    """The pipeline's statistics from the replay model's trials (E43's recordings / summarize, the grand
    mean as E43 builds it, the on/off comparison as E43's onoff rows)."""
    arrs = [trials[r] for r in sorted(trials)]
    out = {}
    for how in HOWS:
        pairs = {}
        for cond in CONDS:
            recs, n_silent = E43.recordings(arrs, cond, how)
            s = E43.summarize(recs, how)
            spk_all = [tr for a in arrs for tr in a[f'{cond}_spk']]
            grand = np.mean([NF.transform(x.astype(float), how) for x in spk_all], axis=0)
            gp = E43.pairs_of(grand)
            rate = float(np.mean([(x >= n_resp).mean() for x in spk_all]))
            out[(how, cond)] = dict(floor1=s['floor1'], floor_mean=s['floor_mean'], rate_trial=rate,
                mean_dist_raw=float(s['raw'].mean()), mean_dist_grand=float(gp.mean()), n_silent=n_silent)
            pairs[cond] = dict(raw=s['raw'], grand=gp)
        for matrix in MATRICES:
            st, tabs = H.compare_on_off(pairs['on'][matrix], pairs['acute_off'][matrix])
            out[(how, matrix, 'block')] = st['block_fruity_small_iso']
            out[(how, matrix, 'pairs_excess')] = tabs['pairs']['iso_resid'].reindex(BLOCK_PAIRS)
    return out


def det_block(kc: pd.DataFrame) -> tuple:
    dp = {c: E43.pairs_of(kc[kc.condition == c].pivot(index='kc', columns='odor', values='spikes')[ODORS].to_numpy())
        for c in CONDS}
    st, tabs = H.compare_on_off(dp['on'], dp['acute_off'])
    return st['block_fruity_small_iso'], tabs['pairs']['iso_resid'].reindex(BLOCK_PAIRS)


def margin_stats(name: str, kc: pd.DataFrame, sigma: float, excess_det: pd.Series) -> dict:
    """The margin statistics of E51 part A for one model: per condition the fraction of responses
    within one sigma of threshold and the mean change probability of responses and non-responses; the
    carriers' and the APL-made carriers' classes (E51's decompose)."""
    out = {}
    for cond in CONDS:
        g = kc[kc.condition == cond]
        resp, non = g[g['response']], g[~g['response']]
        out[f'frac_resp_within_sigma_{cond}'] = float(resp['within_sigma'].mean())
        out[f'mean_p_lose_{cond}'] = float(resp['p_change'].mean())
        out[f'mean_p_gain_{cond}'] = float(non['p_change'].mean())
        out[f'expected_lost_responses_{cond}'] = float(resp['p_change'].sum())
        out[f'expected_gained_responses_{cond}'] = float(non['p_change'].sum())
        out[f'n_responses_{cond}'] = int(len(resp))
    kc = kc.copy()
    kc['pct_rank'] = E51C.pct_rank_within_odor(kc)
    _, sets = E51C.decompose(name, kc, sigma, excess_det)
    pch = kc[kc.condition == 'on'].set_index(['kc', 'odor'])['p_change']
    mar = kc[kc.condition == 'on'].set_index(['kc', 'odor'])['margin']
    for key, lab in (('carrier_resp', 'carriers_resp'), ('carrier_other', 'carriers_other'), ('apl_made_resp', 'apl_made_resp'),
            ('apl_made_other', 'apl_made_other'), ('both', 'both_responders')):
        idx = list(sets['on'][key])
        if idx and not isinstance(idx[0], tuple):
            raise RuntimeError(f'unexpected class index type {type(idx[0])}')
        out[f'n_{lab}_on'] = len(idx)
        out[f'mean_p_change_{lab}_on'] = float(pch.loc[idx].mean()) if idx else np.nan
        out[f'median_margin_{lab}_on'] = float(mar.loc[idx].median()) if idx else np.nan
    return out


# ---------------------------------------------------------------------------
# tables and figures
def agreement(df: pd.DataFrame, pred: str, obs: str) -> dict:
    d = df[[pred, obs]].dropna()
    if len(d) < 3:
        return dict(n=len(d))
    return dict(n=int(len(d)), spearman=float(sstats.spearmanr(d[pred], d[obs]).statistic),
        pearson=float(np.corrcoef(d[pred], d[obs])[0, 1]), bias=float((d[pred] - d[obs]).mean()),
        mae=float((d[pred] - d[obs]).abs().mean()), rmse=float(np.sqrt(((d[pred] - d[obs]) ** 2).mean())))


def fmt(x, spec='+.3f', missing=''):
    return missing if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def write_table(per_model: pd.DataFrame, out: Path, agree: pd.DataFrame) -> None:
    head = ('| model | block, deterministic | block through the noise, three-trial means: pipeline / replay model | grand mean of '
        '30 trials: pipeline / replay model | single-trial floor, APL on: pipeline / replay | response rate per trial, APL on / off: '
        'pipeline | the same, replay model | APL-on responses within one σ | APL-made carriers\' return probability | margins (s) | '
        'replay (s) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    rows = []
    for _, r in per_model.iterrows():
        rows.append(f"| {r['label']} | {fmt(r['block_det'])} | {fmt(r['obs_block_raw'])} / {fmt(r['pred_block_raw'])} | "
            f"{fmt(r['obs_block_grand'])} / {fmt(r['pred_block_grand'])} | {fmt(r['obs_floor1_on'], '.2f')} / {fmt(r['pred_floor1_on'], '.2f')} | "
            f"{fmt(r['obs_rate_on'], '.3f')} / {fmt(r['obs_rate_off'], '.3f')} | {fmt(r['pred_rate_on'], '.3f')} / {fmt(r['pred_rate_off'], '.3f')} | "
            f"{fmt(100 * r['frac_resp_within_sigma_on'], '.0f')} % | {fmt(r['mean_p_change_apl_made_other_on'], '.2f')} | "
            f"{fmt(r['margin_seconds'], '.0f')} | {fmt(r['replay_seconds'], '.0f')} |")
    txt = head + '\n'.join(rows) + '\n\nAgreement across models (replay model against the pipeline):\n\n| statistic | n | Spearman ρ | Pearson r | bias (replay minus pipeline) | mean absolute error |\n|---|---:|---:|---:|---:|---:|\n'
    for _, a in agree.iterrows():
        txt += f"| {a['statistic']} | {int(a['n'])} | {fmt(a.get('spearman'), '.2f')} | {fmt(a.get('pearson'), '.2f')} | {fmt(a.get('bias'), '+.4f')} | {fmt(a.get('mae'), '.4f')} |\n"
    (out / 'comparison_table.md').write_text(txt)


def make_figures(per_model: pd.DataFrame, per_pair: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    for ax, matrix, title in ((axes[0], 'raw', 'three-trial means over ten pseudo-recordings'), (axes[1], 'grand', 'grand mean over all 30 trials')):
        x, y = per_model[f'obs_block_{matrix}'], per_model[f'pred_block_{matrix}']
        lim = (min(0, x.min(), y.min()) - 0.005, max(x.max(), y.max()) + 0.008)
        ax.plot(lim, lim, ':', color='0.5', lw=1)
        ax.scatter(x, y, s=34, color='#2b6cb0', zorder=3)
        for _, r in per_model.iterrows():
            ax.annotate(r['short'], (r[f'obs_block_{matrix}'], r[f'pred_block_{matrix}']), fontsize=6.5,
                textcoords='offset points', xytext=(4, 3))
        ax.set(xlim=lim, ylim=lim, xlabel=f'block through the noise, pipeline ({title})',
            ylabel='block, replay model (margins only; same draws)', title=f'fruity × small-ester block: {title}')
        ax.grid(alpha=0.3)
    ax = axes[2]
    x = per_model['block_det'] - per_model['obs_block_raw']
    y = per_model['frac_resp_within_sigma_on']
    ax.scatter(x, y, s=34, color='#d9480f', zorder=3)
    for _, r in per_model.iterrows():
        ax.annotate(r['short'], (r['block_det'] - r['obs_block_raw'], r['frac_resp_within_sigma_on']), fontsize=6.5,
            textcoords='offset points', xytext=(4, 3))
    ax.set(xlabel='block lost through the noise (deterministic minus three-trial means), pipeline',
        ylabel='fraction of APL-on responses within one σ of threshold', title='a margin statistic against the block lost')
    ax.grid(alpha=0.3)
    fig.suptitle('The margin proxy: the replay model (each KC replayed at the pipeline\'s own jittered thresholds, the network held '
        'as in the deterministic run) against the pipeline', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'E55_predicted_vs_observed.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    pairs = sorted(per_pair['pair'].unique())
    cmap = plt.get_cmap('tab10')
    lim = (per_pair[['obs_excess_raw', 'pred_excess_raw']].min().min() - 0.01, per_pair[['obs_excess_raw', 'pred_excess_raw']].max().max() + 0.01)
    ax.plot(lim, lim, ':', color='0.5', lw=1)
    for i, pr in enumerate(pairs):
        g = per_pair[per_pair['pair'] == pr]
        ax.scatter(g['obs_excess_raw'], g['pred_excess_raw'], s=22, color=cmap(i % 10), label=pr, zorder=3)
    ax.set(xlim=lim, ylim=lim, xlabel='pair excess through the noise, pipeline (three-trial means)',
        ylabel='pair excess, replay model', title='The nine block pairs, every model: excess of the APL-on distance over the isotonic trend')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'E55_pairs.png', dpi=150)
    plt.close(fig)


def short_label(name: str) -> str:
    s = label_of(name)
    s = s.replace('deterministic fit at 6.8 %', 'det. fit').replace('noise-aware fit', 'noise-aware').replace(', exact start', ' exact')
    s = re.sub(r'\s*\(E\d+\)$', '', s)
    s = s.replace('the reference model at 5 %; eight at 1.5, 40 Hz', 'reference 5 %')
    return s


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--models', nargs='+', default=None, help='model names as the part-B tables name them (default: every model with a matched level)')
    ap.add_argument('--seed', type=int, default=None, help="the pipeline's draw seed (default: the table's, 43)")
    ap.add_argument('--resim', action='store_true', help='rebuild the models even if the cached tables exist')
    ap.add_argument('--out', type=Path, default=None, help='results folder (default <shared>/results/E55_margin_proxy)')
    args = ap.parse_args()
    out = args.out or H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    loc = H.outputs_dir(EXPERIMENT)
    loc.mkdir(parents=True, exist_ok=True)
    orn = H.apl16_orn_deltas()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    infos = pipeline_models()
    names = args.models or list(infos)
    missing = [n for n in names if n not in infos]
    if missing:
        raise SystemExit(f'no matched pipeline rows for {missing}; known: {sorted(infos)}')
    inputs, checks_all, pm_rows, pp_rows, ms_rows = {}, {}, [], [], []
    for n in names:
        info = dict(infos[n])
        if args.seed is not None:
            info['seed'] = args.seed
        src = register(n)
        obs = observed(n, info, inputs)
        cache, npz, ck = loc / f'{n}_kc_odor.parquet', loc / f'{n}_replay.npz', loc / f'{n}_checks.json'
        if cache.exists() and npz.exists() and ck.exists() and not args.resim:
            kc = pd.read_parquet(cache)
            a = E43.load_npz(npz)
            n_rec = int(a['n_recordings'])
            trials = {rep: {key: a[f'{key}_rep{rep:02d}'] for key in (f'{c}_{k}' for c in CONDS for k in ('spk', 'resp'))} for rep in range(n_rec)}
            checks = json.loads(ck.read_text())
            checks['from_cache'] = str(cache)
            print(f'{n}: from {cache}', flush=True)
        else:
            kc, trials, checks = simulate(n, info, orn, loc)
            kc.to_parquet(cache)
            flat = {f'{key}_rep{rep:02d}': arr for rep, a in trials.items() for key, arr in a.items()}
            flat.update(n_recordings=np.asarray(len(trials)), seed=np.asarray(info['seed']), sigma=np.asarray(info['sigma']),
                tuned_fixed_thr=np.asarray(info['tuned_fixed_thr']))
            E43.save_npz(npz, flat)
            E43.write_json(ck, checks)
        inputs[str(cache)] = sha256(cache)
        checks['source'] = src
        n_resp = int(checks['n_spikes_for_response'])
        bdet, excess_det = det_block(kc)
        checks['block_det'] = bdet
        checks['block_det_minus_pipeline'] = bdet - obs[('graded', 'det', 'block')]
        assert abs(checks['block_det_minus_pipeline']) < 1e-6, (n, bdet, obs[('graded', 'det', 'block')])
        pred = process(trials, n_resp)
        ms = margin_stats(n, kc, info['sigma'], excess_det)
        ms_rows.append(dict(model=n, label=label_of(n), sigma=info['sigma'], thr_ratio=checks['thr_ratio'], **ms))
        row = dict(model=n, label=label_of(n), short=short_label(n), partb_experiment=info['partb_experiment'], level=info['level'],
            sigma=info['sigma'], thr_ratio=checks['thr_ratio'], seed=info['seed'], n_recordings=info['n_recordings'],
            n_trials=info['n_trials'], block_det=bdet, **src)
        for how in HOWS:
            suf = '' if how == 'graded' else '_thresholded'
            for matrix in MATRICES:
                row[f'obs_block_{matrix}{suf}'] = obs[(how, matrix, 'block')]
                row[f'pred_block_{matrix}{suf}'] = pred[(how, matrix, 'block')]
            for cond, cl in (('on', 'on'), ('acute_off', 'off')):
                for key in ('floor1', 'floor_mean', 'rate_trial', 'mean_dist_raw', 'mean_dist_grand'):
                    k2 = key.replace('rate_trial', 'rate')
                    row[f'obs_{k2}_{cl}{suf}'] = obs[(how, cond)][key]
                    row[f'pred_{k2}_{cl}{suf}'] = pred[(how, cond)][key]
        row.update({k: ms[k] for k in ('frac_resp_within_sigma_on', 'frac_resp_within_sigma_acute_off', 'mean_p_lose_on',
            'mean_p_change_apl_made_other_on', 'mean_p_change_carriers_resp_on')})
        row.update(margin_seconds=checks['margin_seconds'], replay_seconds=checks['replay_seconds'],
            build_seconds=checks['build_seconds'], total_seconds=checks['total_seconds'])
        pm_rows.append(row)
        for (a, b) in BLOCK_PAIRS:
            pp_rows.append(dict(model=n, label=label_of(n), pair=f'{a} / {b}', odor_a=a, odor_b=b,
                excess_det=float(excess_det[(a, b)]), obs_excess_raw=float(obs[('graded', 'raw', 'pairs_excess')][(a, b)]),
                pred_excess_raw=float(pred[('graded', 'raw', 'pairs_excess')][(a, b)]),
                obs_excess_grand=float(obs[('graded', 'grand', 'pairs_excess')][(a, b)]),
                pred_excess_grand=float(pred[('graded', 'grand', 'pairs_excess')][(a, b)])))
        checks_all[n] = checks
        per_model, per_pair = pd.DataFrame(pm_rows), pd.DataFrame(pp_rows)
        per_model.to_csv(out / 'per_model.csv', index=False)
        per_pair.to_csv(out / 'per_pair.csv', index=False)
        pd.DataFrame(ms_rows).to_csv(out / 'margin_stats.csv', index=False)
        print(f"{n}: block det {bdet:+.3f}; pipeline raw {row['obs_block_raw']:+.3f} grand {row['obs_block_grand']:+.3f}; replay model raw "
            f"{row['pred_block_raw']:+.3f} grand {row['pred_block_grand']:+.3f}; floors on {row['obs_floor1_on']:.3f} / {row['pred_floor1_on']:.3f}; "
            f"rates on {row['obs_rate_on']:.4f} / {row['pred_rate_on']:.4f} off {row['obs_rate_off']:.4f} / {row['pred_rate_off']:.4f}; "
            f"margins {checks['margin_seconds']:.0f}s, replay {checks['replay_seconds']:.0f}s", flush=True)
    per_model, per_pair = pd.DataFrame(pm_rows), pd.DataFrame(pp_rows)
    ag_rows = []
    for stat, pred_c, obs_c in (('block, three-trial means', 'pred_block_raw', 'obs_block_raw'),
            ('block, grand mean', 'pred_block_grand', 'obs_block_grand'),
            ('block, three-trial means (thresholded)', 'pred_block_raw_thresholded', 'obs_block_raw_thresholded'),
            ('single-trial floor, APL on', 'pred_floor1_on', 'obs_floor1_on'), ('single-trial floor, APL off', 'pred_floor1_off', 'obs_floor1_off'),
            ('response rate per trial, APL on', 'pred_rate_on', 'obs_rate_on'), ('response rate per trial, APL off', 'pred_rate_off', 'obs_rate_off'),
            ('mean pair distance, three-trial means, APL on', 'pred_mean_dist_raw_on', 'obs_mean_dist_raw_on'),
            ('mean pair distance, three-trial means, APL off', 'pred_mean_dist_raw_off', 'obs_mean_dist_raw_off')):
        ag_rows.append(dict(statistic=stat, **agreement(per_model, pred_c, obs_c)))
    ag_rows.append(dict(statistic='pair excess, three-trial means (all pairs x models)', **agreement(per_pair, 'pred_excess_raw', 'obs_excess_raw')))
    ag_rows.append(dict(statistic='block lost (det minus three-trial means) against the APL-on fraction within one sigma',
        **agreement(per_model.assign(lost=per_model['block_det'] - per_model['obs_block_raw']), 'frac_resp_within_sigma_on', 'lost')))
    agree = pd.DataFrame(ag_rows)
    agree.to_csv(out / 'agreement.csv', index=False)
    write_table(per_model, out, agree)
    make_figures(per_model, per_pair, out)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E55_margin_proxy.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), al_analysis_commit=commit,
        model=H.model_provenance(orn_deltas=orn), approx_conc_note=H.APPROX_CONC_NOTE, models=names,
        pipeline_info={n: {k: (str(v) if isinstance(v, Path) else v) for k, v in infos[n].items()} for n in names},
        margin_note=E51C.MARGIN_NOTE, replay_note=('replay model: each KC\'s reconstructed drive (the other KCs\' spikes, APL and the '
        'PN layer as in the deterministic run) integrated as olfsysm does at the pipeline\'s own jittered thresholds (the same '
        'b_draws, the re-tuned fixed_thr of the matched level, thr0 (1 + sigma xi) clipped at 0.01 thr0), its own KC-KC '
        'self-inhibition from its replayed spikes; trials processed as E43 part B processes model trials'),
        inputs=inputs, checks=checks_all)
    E43.write_json(out / 'provenance.json', prov)
    print(f'-> {out}', flush=True)


if __name__ == '__main__':
    main()
