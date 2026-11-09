#!/usr/bin/env python3
"""The network-aware margin proxy, batched (the follow-up of E56 proposed 2026-11-09).

E56_network_margin_proxy.py replays each KC's reconstructed drive at the trial-noise pipeline's own jittered
thresholds, trial by trial, with each KC's KC-KC inhibition taken from the other KCs' spikes of the previous
pass (pass 1: the deterministic run's), and converges onto the pipeline in about five passes at about
95 s per pass and condition. This module does the same with the thirty trials integrated together (one
pass of the time loop per odor and condition, arrays of trials x KCs), the KC-KC inhibition computed at
each step from the previous pass's spikes of the KCs that spiked at all in that trial and odor (the
product restricted to them), and an optional relaxation step (the inhibition fed to pass k + 1 from the
mean of the filtered spike trains of passes k and k - 1, which damps the oscillation E56 saw between its
passes 2 and 4). Pass 1 reproduces E56's pass 1 exactly; without relaxation every pass reproduces E56's.

`reconstruct(model)` builds a model with E43's ExcitabilityModel, checks it against its saved outputs,
and returns per condition the drive of every KC without KC-KC inhibition (from a run with the thresholds
out of reach, the voltages saved), the deterministic spike trains, the thresholds, the KC-KC weights and
the pipeline's jittered thresholds for its trials; `run_passes(rec, n_pass, relax)` returns each pass's
trial spike counts in the pipeline's layout and the convergence between passes; `score(trials, n_resp)`
is E55's processing (the pipeline's statistics). As a script it runs the three on E56's models and
compares each pass with the pipeline and with E56's per-trial numbers, timing every pass:
results/<out>/ (per_model_pass.csv, agreement_pass.csv, timing.csv, comparison_table.md).
Usage: OMP_NUM_THREADS=12 nice -n 10 ./venv/bin/python noise_proxy.py [--models NAME ...] [--passes 3]
  [--relax] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H
import E43_model_trial_noise as E43
import E51_block_carriers as E51C
import E55_margin_proxy as E55

EXPERIMENT_DEFAULT = 'noise_proxy_batched'
ODORS, CONDS = E55.ODORS, E55.CONDS
COND_LABEL = E55.COND_LABEL


class Reconstruction:
    """One model's drives and the pipeline's trial thresholds (see `reconstruct`)."""

    def __init__(self, name, info, d_free, spikes_det, thr, W, thr_trials, k, tau, decay, n_resp, checks, cond_seconds):
        self.name, self.info = name, info
        self.d_free = d_free            # {cond: {odor: KCs x steps float32}}, the drive without KC-KC inhibition
        self.spikes_det = spikes_det    # {cond: {odor: KCs x steps bool}}, the deterministic run's spikes
        self.thr = thr                  # KCs, the deterministic thresholds
        self.W = W                      # csr KCs x KCs (float32) or None
        self.thr_trials = thr_trials    # trials x KCs, the pipeline's jittered thresholds
        self.k, self.tau, self.decay, self.n_resp = k, tau, decay, n_resp
        self.checks, self.cond_seconds = checks, cond_seconds
        self.n_kcs = len(thr)
        self.n_trials = thr_trials.shape[0]


def reconstruct(name: str, info: dict, orn: pd.DataFrame, loc: Path) -> Reconstruction:
    """E56's reconstruction without the passes: the model, its checks, the drives without KC-KC
    inhibition and the deterministic spike trains per condition and odor, the pipeline's jittered
    thresholds for its trials."""
    from al_analysis.mb_model import get_time_index
    t0 = time.time()
    m = E43.ExcitabilityModel(name, orn, loc)
    checks = dict(model=name, label=E55.label_of(name), build_seconds=time.time() - t0, **m.info)
    mp, rv = m.mp, m.rv
    if float(mp.kc.ves_p) != 0.0 or not bool(mp.kc.pn_claw_to_apl):
        raise NotImplementedError('vesicle depletion or spike-driven APL')
    taum, dt = float(mp.kc.taum), float(mp.time_dt)
    k = dt / taum
    kckc = bool(mp.kc.use_kc_kc_inh)
    tau = float(mp.kc.kc_kc_tau) if kckc else 1.0
    decay = 1.0 - dt / tau
    W = rv.kc.wKCKC.tocsr().astype(np.float32) if kckc else None
    if kckc and np.any(W.diagonal() != 0):
        raise NotImplementedError('wKCKC has a nonzero diagonal')
    t = np.asarray(get_time_index(mp, rv), dtype=float)
    i0 = int(np.searchsorted(t, mp.time_start - dt / 2)) - 2
    n_resp = m.n_spikes_for_response
    checks.update(taum=taum, dt=dt, n_steps=int(len(t)), kc_kc_inh=kckc, kc_kc_tau=tau, n_spikes_for_response=n_resp,
        wKCKC_nnz=int(W.nnz) if kckc else 0)
    sigma, f_tuned = info['sigma'], info['tuned_fixed_thr']
    n_rec, n_tr, seed = info['n_recordings'], info['n_trials'], info['seed']
    assert abs(m.f0 - info['fixed_thr_det']) < 1e-6
    thr_trials = []
    for rep in range(n_rec):
        for xi in E43.b_draws(seed, name, m.draw_kinds['rec'], rep, n_tr, m.draw_size):
            thr_trials.append(m.thr_trial(f_tuned, sigma, xi)[0].ravel())
    thr_trials = np.asarray(thr_trials)
    d_free, spikes_det, cond_seconds = {}, {}, {}
    thr = None
    mp.kc.save_vm_sims = True
    try:
        for cond, p in m.params.items():
            t1 = time.time()
            r0, s0 = E43.to_arrays(*m.sim(dict(p, fixed_thr=1e12)))
            assert s0.sum() == 0
            vms = rv.kc.vm_sims
            d_free[cond] = {}
            for j, o in enumerate(ODORS):
                V = np.asarray(vms[m.pn_idx[j]])
                vms[m.pn_idx[j]] = None
                d_free[cond][o] = E51C.drive_from_vm(V[:, i0:], taum, dt).astype(np.float32)
            del vms
            r, s = E43.to_arrays(*m.sim(p))
            mism = int((r != m.det[cond][0]).sum() + (s != m.det[cond][1]).sum())
            assert mism == 0, f'{name} {cond}: the run with the voltages saved differs from the deterministic run ({mism})'
            thr = np.asarray(rv.kc.thr, dtype=float).ravel()
            vms = rv.kc.vm_sims
            spikes_det[cond] = {}
            n_bad = 0
            for j, o in enumerate(ODORS):
                V = np.asarray(vms[m.pn_idx[j]])[:, i0:]
                vms[m.pn_idx[j]] = None
                spike = (V[:, 1:] == 0) & (V[:, :-1] != 0)
                n_bad += int((spike.sum(axis=1) != s[:, j]).sum())
                spikes_det[cond][o] = spike
            del vms
            assert n_bad == 0, f'{name} {cond}: spike counts from the Vm resets differ in {n_bad} entries'
            cond_seconds[cond] = time.time() - t1
    finally:
        mp.kc.save_vm_sims = False
    r, s = E43.to_arrays(*m.sim(m.params['on']))
    checks['after_mismatch_on'] = int((r != m.det['on'][0]).sum() + (s != m.det['on'][1]).sum())
    assert checks['after_mismatch_on'] == 0
    m.close()
    checks['reconstruct_seconds'] = time.time() - t0
    return Reconstruction(name, info, d_free, spikes_det, thr, W, thr_trials, k, tau, decay, n_resp, checks, cond_seconds)


def filtered_batch(S: np.ndarray, tau: float, decay: float) -> np.ndarray:
    """olfsysm's kc_kc_activity for a trials x KCs x steps spike array (float32)."""
    from scipy.signal import lfilter
    return lfilter([0.0, 1.0 / tau], [1.0, -decay], S.astype(np.float32), axis=2).astype(np.float32)


def replay_batch(D0: np.ndarray, inh_t: np.ndarray | None, thr_trials: np.ndarray, k: float) -> np.ndarray:
    """Spike trains (trials x KCs x steps, bool) of the KCs integrated as olfsysm does for every trial at once:
    drive D0 (KCs x steps, shared by the trials) minus the per-trial inhibition `inh_t` (steps x trials x KCs,
    the layout `inhibition_from` returns, or None), thresholds per trial. The loop starts at the first step
    with any positive drive (before it every membrane potential stays at zero and no KC can spike)."""
    n_tr, n_kc = thr_trials.shape
    n_t = D0.shape[1]
    D0_t = np.ascontiguousarray(D0.T)                     # steps x KCs
    pos = (D0_t > 0).any(axis=1)
    t_start = int(np.argmax(pos)) if pos.any() else n_t
    v = np.zeros((n_tr, n_kc), dtype=np.float32)
    S_t = np.zeros((n_t, n_tr, n_kc), dtype=bool)
    th = thr_trials.astype(np.float32)
    for t in range(t_start, n_t):
        d = D0_t[t][None, :] if inh_t is None else D0_t[t][None, :] - inh_t[t]
        v += (d - v) * k
        sp = v > th
        S_t[t] = sp
        v[sp] = 0.0
    return S_t.transpose(1, 2, 0)


def inhibition_from(S_prev: np.ndarray, W, tau: float, decay: float) -> np.ndarray:
    """The KC-KC inhibition every KC receives in every trial (steps x trials x KCs, float32: contiguous per
    step for the replay loop) from the previous pass's spikes, the product restricted to the KCs that
    spiked at all in some trial."""
    n_tr, n_kc, n_t = S_prev.shape
    act = np.flatnonzero(S_prev.any(axis=(0, 2)))
    out = np.zeros((n_t, n_tr, n_kc), dtype=np.float32)
    if len(act) == 0:
        return out
    A = filtered_batch(np.ascontiguousarray(S_prev[:, act, :]), tau, decay)   # trials x act x steps
    W_act = W[:, act].toarray().astype(np.float32)                           # KCs x act (dense; a few hundred columns)
    for q in range(n_tr):
        out[:, q, :] = (W_act @ A[q]).T
    return out


def run_passes(rec: Reconstruction, n_pass: int, relax: bool = False, conds=CONDS) -> tuple:
    """Every pass's trial spike counts ({pass: {cond: trials x KCs x odors}}), the convergence rows and
    the seconds per pass (`conds`: the condition names of rec.d_free / rec.spikes_det)."""
    spk = {p: {cond: np.zeros((rec.n_trials, rec.n_kcs, len(ODORS)), dtype=np.int32) for cond in conds} for p in range(1, n_pass + 1)}
    conv, secs = [], {p: 0.0 for p in range(1, n_pass + 1)}
    kckc = rec.W is not None
    for cond in conds:
        for j, o in enumerate(ODORS):
            D0 = rec.d_free[cond][o]
            S_det = np.broadcast_to(rec.spikes_det[cond][o][None], (rec.n_trials, rec.n_kcs, D0.shape[1]))
            prev, prev2 = None, None
            for pss in range(1, n_pass + 1):
                t2 = time.time()
                if not kckc:
                    inh = None
                elif pss == 1:
                    inh = inhibition_from(np.ascontiguousarray(S_det), rec.W, rec.tau, rec.decay)
                else:
                    src = prev if (not relax or prev2 is None) else None
                    if src is not None:
                        inh = inhibition_from(prev, rec.W, rec.tau, rec.decay)
                    else:
                        inh = 0.5 * (inhibition_from(prev, rec.W, rec.tau, rec.decay) + inhibition_from(prev2, rec.W, rec.tau, rec.decay))
                S = replay_batch(D0, inh, rec.thr_trials, rec.k)
                del inh
                spk[pss][cond][:, :, j] = S.sum(axis=2)
                secs[pss] += time.time() - t2
                if pss > 1:
                    a, b = spk[pss - 1][cond][:, :, j] >= rec.n_resp, spk[pss][cond][:, :, j] >= rec.n_resp
                    conv.append(dict(model=rec.name, condition=cond, odor=o, pass_from=pss - 1, pass_to=pss,
                        frac_response_changed=float((a != b).mean()), relax=relax))
                prev2, prev = prev, S
    return spk, conv, secs


def pipeline_pairs(spk_by_cond: dict, n_rec: int, n_tr: int, n_resp: int) -> dict:
    """From one pass's trial spike counts ({cond: trials x KCs x odors}, cond 'on' / 'acute_off'), the
    pipeline's raw pair distances per condition (the mean over the pseudo-recordings of the three-trial-mean
    cosine distances; Series indexed by (odor_a, odor_b)), its grand-mean pair distances, the single-trial
    floors and the per-trial response rates: {cond: dict(raw=Series, grand=Series, floor1=float, rate=float)}."""
    import noise_floor_trial_distances as NF
    arrs = [a for _, a in sorted(as_trials(spk_by_cond, n_rec, n_tr, n_resp).items())]
    out = {}
    for cond in CONDS:
        recs, _ = E43.recordings(arrs, cond, 'graded')
        s = E43.summarize(recs, 'graded')
        spk_all = [tr for a in arrs for tr in a[f'{cond}_spk']]
        grand = E43.pairs_of(np.mean([NF.transform(x.astype(float), 'graded') for x in spk_all], axis=0))
        out[cond] = dict(raw=s['raw'], grand=grand, floor1=float(s['floor1']),
            rate=float(np.mean([(x >= n_resp).mean() for x in spk_all])))
    return out


def make_block_proxy_hook(store: dict, orn: pd.DataFrame, sigma: float, thr_ratio: float, n_rec: int, n_tr: int,
        seed: int, n_pass: int, relax: bool, condition_overrides: dict, thr_min_frac: float = 0.01):
    """A `condition_hooks` callable for `optimize_mb_model_params`: at the parameters being evaluated it
    runs, per condition, the deterministic simulation with the voltages saved and one with the thresholds
    out of reach, reconstructs every KC's drive, and runs `n_pass` batched replay passes at the trial-noise
    pipeline's own jittered thresholds (`sigma`, the mean threshold `thr_ratio` x the current one, the
    pipeline's draws for the fits.csv models: seed, kind 'rec', `n_rec` pseudo-recordings of `n_tr`
    trials, so a later pipeline run of the fitted model uses the same draws). Returns {condition: trials
    x KCs x odors spike counts} of the last pass, for `pipeline_pairs`. `store` is
    E43.captured_olfsysm_vars' dict (the simulator's mp / rv once built); `condition_overrides` the
    fit's {'on': {}, 'off': {...}}. The hook leaves mp.kc.save_vm_sims False."""
    from al_analysis.mb_model import get_time_index
    state: dict = {}

    def hook(simulate, params):
        mp, rv = store['mp'], store['rv']
        if 'xi' not in state:
            inputs = [H.odor_abbrev(c) for c in orn.columns]
            state['pn_idx'] = [inputs.index(o) for o in ODORS]
            n_kc = int(np.asarray(rv.kc.spont_in).size)
            state['spont_in'] = np.asarray(rv.kc.spont_in, dtype=float).ravel()
            state['xi'] = np.concatenate([E43.b_draws(seed, 'ref40', 'rec', rep, n_tr, n_kc) for rep in range(n_rec)])
            t = np.asarray(get_time_index(mp, rv), dtype=float)
            dt = float(mp.time_dt)
            state['i0'] = int(np.searchsorted(t, mp.time_start - dt / 2)) - 2
            state['taum'], state['dt'] = float(mp.kc.taum), dt
            kckc = bool(mp.kc.use_kc_kc_inh)
            state['tau'] = float(mp.kc.kc_kc_tau) if kckc else 1.0
            state['decay'] = 1.0 - dt / state['tau']
            state['W'] = rv.kc.wKCKC.tocsr().astype(np.float32) if kckc else None
            if kckc and np.any(state['W'].diagonal() != 0):
                raise NotImplementedError('wKCKC has a nonzero diagonal')
            state['n_resp'] = int(mp.kc.n_spikes_for_response)
        # the pipeline re-tunes fixed_thr alone (thr0 = tuned fixed_thr + 2 spont_in), so the ratio scales fixed_thr
        thr0 = thr_ratio * float(params['fixed_thr']) + 2.0 * state['spont_in']
        thr_trials = np.maximum(thr0[None, :] * (1.0 + sigma * state['xi']), thr_min_frac * thr0[None, :])
        taum, dt, i0 = state['taum'], state['dt'], state['i0']
        k = dt / taum
        d_free, spikes_det = {}, {}
        mp.kc.save_vm_sims = True
        try:
            for cond, ov in condition_overrides.items():
                c_params = {**params, **ov}
                _, s0 = E43.to_arrays(*simulate(dict(c_params, fixed_thr=1e12)))
                if s0.sum() != 0:
                    raise RuntimeError('KCs spike with the thresholds out of reach')
                vms = rv.kc.vm_sims
                d_free[cond] = {}
                for j, o in enumerate(ODORS):
                    V = np.asarray(vms[state['pn_idx'][j]])
                    vms[state['pn_idx'][j]] = None
                    d_free[cond][o] = E51C.drive_from_vm(V[:, i0:], taum, dt).astype(np.float32)
                del vms
                _, s = E43.to_arrays(*simulate(c_params))
                vms = rv.kc.vm_sims
                spikes_det[cond] = {}
                for j, o in enumerate(ODORS):
                    V = np.asarray(vms[state['pn_idx'][j]])[:, i0:]
                    vms[state['pn_idx'][j]] = None
                    spike = (V[:, 1:] == 0) & (V[:, :-1] != 0)
                    if int((spike.sum(axis=1) != s[:, j]).sum()) != 0:
                        raise RuntimeError('spike counts from the voltage resets differ from the simulation\'s')
                    spikes_det[cond][o] = spike
                del vms
        finally:
            mp.kc.save_vm_sims = False
        rec = Reconstruction('fit', {}, d_free, spikes_det, thr0, state['W'], thr_trials, k, state['tau'], state['decay'],
            state['n_resp'], {}, {})
        rec.conds = tuple(condition_overrides)
        spk, _, _ = run_passes(rec, n_pass, relax, conds=tuple(condition_overrides))
        return {('acute_off' if c == 'off' else c): spk[n_pass][c] for c in condition_overrides}

    return hook


def as_trials(spk_pass: dict, n_rec: int, n_tr: int, n_resp: int) -> dict:
    out = {}
    for rep in range(n_rec):
        sl = slice(rep * n_tr, (rep + 1) * n_tr)
        a = {}
        for cond in CONDS:
            a[f'{cond}_spk'] = spk_pass[cond][sl]
            a[f'{cond}_resp'] = spk_pass[cond][sl] >= n_resp
        out[rep] = a
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--models', nargs='+', default=None, help="model names (default: E56's eleven, from its per_model_pass.csv)")
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--relax', action='store_true', help='relaxation: the inhibition for pass k + 1 from the mean of passes k and k - 1')
    ap.add_argument('--out', type=Path, default=None, help=f'results folder (default <shared>/results/{EXPERIMENT_DEFAULT})')
    args = ap.parse_args()
    out = args.out or H.results_dir(EXPERIMENT_DEFAULT)
    out.mkdir(parents=True, exist_ok=True)
    loc = H.outputs_dir(EXPERIMENT_DEFAULT)
    loc.mkdir(parents=True, exist_ok=True)
    orn = H.apl16_orn_deltas()
    infos = E55.pipeline_models()
    e56 = H.results_dir('E56_network_margin_proxy') / 'per_model_pass.csv'
    e56 = pd.read_csv(e56) if e56.exists() else None
    names = args.models or (list(dict.fromkeys(e56['model'])) if e56 is not None else list(infos))
    rows, conv_all, timing, checks_all = [], [], [], {}
    for n in names:
        info = infos[n]
        src = E55.register(n)
        obs = E55.observed(n, info, {})
        rec = reconstruct(n, info, orn, loc)
        spk, conv, secs = run_passes(rec, args.passes, args.relax)
        conv_all += conv
        kckc = E55.kc_kc_scale(n, src)
        for pss in range(1, args.passes + 1):
            pred = E55.process(as_trials(spk[pss], info['n_recordings'], info['n_trials'], rec.n_resp), rec.n_resp)
            row = dict(model=n, label=E55.label_of(n), short=E55.short_label(n), **{'pass': pss}, relax=args.relax, kc_kc_scale=kckc,
                fixed_thr_det=info['fixed_thr_det'], kckc_over_thr=kckc / info['fixed_thr_det'], block_det=obs[('graded', 'det', 'block')],
                pass_seconds=secs[pss], reconstruct_seconds=rec.checks['reconstruct_seconds'])
            for matrix in E55.MATRICES:
                row[f'obs_block_{matrix}'] = obs[('graded', matrix, 'block')]
                row[f'pred_block_{matrix}'] = pred[('graded', matrix, 'block')]
            for cond, cl in (('on', 'on'), ('acute_off', 'off')):
                for key in ('floor1', 'rate_trial', 'mean_dist_raw'):
                    k2 = key.replace('rate_trial', 'rate')
                    row[f'obs_{k2}_{cl}'] = obs[('graded', cond)][key]
                    row[f'pred_{k2}_{cl}'] = pred[('graded', cond)][key]
            if e56 is not None:
                g = e56[(e56.model == n) & (e56['pass'] == pss)]
                if len(g) == 1:
                    row['e56_pred_block_raw'] = float(g['pred_block_raw'].iloc[0])
                    row['e56_pred_floor1_on'] = float(g['pred_floor1_on'].iloc[0])
                    row['e56_pass_seconds'] = float(g['pass_seconds'].iloc[0])
            rows.append(row)
        checks_all[n] = rec.checks
        timing.append(dict(model=n, reconstruct_seconds=rec.checks['reconstruct_seconds'], build_seconds=rec.checks['build_seconds'],
            **{f'pass{p}_seconds': secs[p] for p in secs}, total_pass_seconds=sum(secs.values())))
        pm = pd.DataFrame(rows)
        pm.to_csv(out / 'per_model_pass.csv', index=False)
        pd.DataFrame(conv_all).to_csv(out / 'convergence.csv', index=False)
        pd.DataFrame(timing).to_csv(out / 'timing.csv', index=False)
        g = pm[pm.model == n].set_index('pass')
        print(f"{n}: pipeline block {g.loc[1, 'obs_block_raw']:+.3f}, passes " + ' / '.join(f"{g.loc[p, 'pred_block_raw']:+.3f}" for p in g.index)
            + (f" (E56: " + ' / '.join(f"{g.loc[p, 'e56_pred_block_raw']:+.3f}" for p in g.index if 'e56_pred_block_raw' in g and pd.notna(g.loc[p, 'e56_pred_block_raw'])) + ')' if e56 is not None else '')
            + f"; floor on {g.loc[1, 'obs_floor1_on']:.3f}, passes " + ' / '.join(f"{g.loc[p, 'pred_floor1_on']:.3f}" for p in g.index)
            + f"; seconds: reconstruct {rec.checks['reconstruct_seconds']:.0f}, passes " + ' / '.join(f"{secs[p]:.0f}" for p in secs), flush=True)
        del rec
    pm = pd.DataFrame(rows)
    ag = []
    for pss in range(1, args.passes + 1):
        g = pm[pm['pass'] == pss]
        for key, stat, pc, oc in (('block_raw', 'block, three-trial means', 'pred_block_raw', 'obs_block_raw'),
                ('block_grand', 'block, grand mean', 'pred_block_grand', 'obs_block_grand'),
                ('floor_on', 'single-trial floor, APL on', 'pred_floor1_on', 'obs_floor1_on'),
                ('floor_off', 'single-trial floor, APL off', 'pred_floor1_off', 'obs_floor1_off'),
                ('rate_off', 'response rate per trial, APL off', 'pred_rate_off', 'obs_rate_off')):
            ag.append(dict(key=key, statistic=stat, **{'pass': pss}, relax=args.relax, **E55.agreement(g, pc, oc)))
    agree = pd.DataFrame(ag)
    agree.to_csv(out / 'agreement_pass.csv', index=False)
    lines = ['| model | KC-KC / threshold | block: pipeline / ' + ' / '.join(f'pass {p}' for p in range(1, args.passes + 1))
        + ' | APL-on floor: pipeline / ' + ' / '.join(f'pass {p}' for p in range(1, args.passes + 1)) + ' | seconds per pass (E56 per-trial) |',
        '|---|---:|---:|---:|---:|']
    for n, g in pm.groupby('model', sort=False):
        g = g.set_index('pass')
        r1 = g.loc[1]
        lines.append(f"| {r1['label']} | {E55.fmt(r1['kc_kc_scale'], '.0f')} / {E55.fmt(r1['fixed_thr_det'], '.0f')} | {E55.fmt(r1['obs_block_raw'])} / "
            + ' / '.join(E55.fmt(g.loc[p, 'pred_block_raw']) for p in g.index) + f" | {E55.fmt(r1['obs_floor1_on'], '.2f')} / "
            + ' / '.join(E55.fmt(g.loc[p, 'pred_floor1_on'], '.2f') for p in g.index) + ' | '
            + ' / '.join(E55.fmt(g.loc[p, 'pass_seconds'], '.0f') + (f" ({E55.fmt(g.loc[p, 'e56_pass_seconds'], '.0f')})" if 'e56_pass_seconds' in g and pd.notna(g.loc[p, 'e56_pass_seconds']) else '') for p in g.index) + ' |')
    lines += ['', 'Agreement with the pipeline per pass:', '', '| statistic | pass | n | Spearman ρ | bias | mean absolute error |', '|---|---:|---:|---:|---:|---:|']
    for _, a in agree.iterrows():
        lines.append(f"| {a['statistic']} | {int(a['pass'])} | {int(a['n'])} | {E55.fmt(a.get('spearman'), '.2f')} | {E55.fmt(a.get('bias'), '+.4f')} | {E55.fmt(a.get('mae'), '.4f')} |")
    (out / 'comparison_table.md').write_text('\n'.join(lines) + '\n')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    E43.write_json(out / 'provenance.json', dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='noise_proxy.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), al_analysis_commit=commit,
        model=H.model_provenance(orn_deltas=orn), models=names, passes=args.passes, relax=args.relax, checks=checks_all))
    print(f'-> {out}', flush=True)


if __name__ == '__main__':
    main()
