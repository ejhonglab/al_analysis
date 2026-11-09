#!/usr/bin/env python3
"""E56: the network-aware margin proxy.

The margin proxy (E55, `E55_margin_proxy.py`) replayed each KC's reconstructed drive at the trial-noise
pipeline's own jittered thresholds with the rest of the network held at the deterministic run, and found
that what it misses of the pipeline grows with the KC-KC inhibition's weight relative to the threshold:
the other KCs' jittered spikes change each KC's inhibition, and the replay held those spikes fixed. This
adds the network's response by iteration. Pass 1 is E55's replay model (each KC's drive without KC-KC
inhibition, minus the KC-KC weights applied to the low-pass-filtered spike trains of the deterministic
run, replayed at the jittered thresholds). Pass k + 1 replaces the deterministic spike trains by the
spike trains of pass k in the same trial: each KC's drive = its drive without KC-KC inhibition - W a_k,
a_k the filtered spikes of all KCs from pass k (the diagonal of W is zero for these models, so there is
no self term), replayed at the same jittered thresholds. The passes are a synchronous (Jacobi) iteration
of the KC-KC loop over whole trials; olfsysm integrates the loop step by step, so convergence of the
passes to olfsysm's trials is the question, not a given.

Per model and pass: the trials processed exactly as the pipeline processes model trials (E55's
`process`: ten pseudo-recordings of three trials, the single-trial floors, per-trial response rates,
mean distances, the block as three-trial means and as the grand mean, the per-pair excess); the
convergence between passes (the fraction of KC x odor x trial responses that change, the change in the
block); and, across models, each pass's agreement with the pipeline (Spearman rho, bias, mean absolute
error) and the network's share remaining after each pass against KC-KC / threshold.

Models: E55's set (every model with pipeline rows at its matched level; `E55_margin_proxy.pipeline_models`),
or --models. Default passes: 1, 2, 3 (--passes).

Outputs: <shared>/results/E56_network_margin_proxy/ (per_model_pass.csv, per_pair_pass.csv,
convergence.csv, agreement_pass.csv, comparison_table.md, E56_passes.png, E56_network_share.png,
provenance.json); ./outputs/E56_network_margin_proxy/ (per model: <model>_passes.npz with every pass's
trial spike counts, <model>_checks.json). With the cached files present a model is not rebuilt (--resim).
Usage: OMP_NUM_THREADS=12 nice -n 10 ./venv/bin/python E56_network_margin_proxy.py [--models NAME ...]
  [--passes 3] [--resim] [--out DIR]
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

EXPERIMENT = 'E56_network_margin_proxy'
ODORS, CONDS, BLOCK_PAIRS = E55.ODORS, E55.CONDS, E55.BLOCK_PAIRS
COND_LABEL = E55.COND_LABEL


def replay_spikes(D: np.ndarray, th: np.ndarray, k: float) -> np.ndarray:
    """Spike matrix (KCs x steps, bool) of independent KCs integrated as olfsysm does
    (Vm += (D - Vm) dt / taum, spike and reset to 0 above `th`), for models without KC-KC
    self-inhibition (the diagonal of wKCKC is zero)."""
    n_kc, n_t = D.shape
    v = np.zeros(n_kc)
    S = np.zeros((n_kc, n_t), dtype=bool)
    for t in range(n_t):
        v = v + (D[:, t] - v) * k
        sp = v > th
        S[:, t] = sp
        v[sp] = 0.0
    return S


def filtered(S: np.ndarray, tau: float, decay: float) -> np.ndarray:
    """olfsysm's kc_kc_activity: a(t) = decay a(t-1) + spikes(t-1) / tau (float32 to save memory)."""
    from scipy.signal import lfilter
    return lfilter([0.0, 1.0 / tau], [1.0, -decay], S.astype(np.float32), axis=1).astype(np.float32)


def simulate_passes(name: str, info: dict, orn: pd.DataFrame, loc: Path, n_pass: int) -> tuple:
    """Builds the model, reconstructs the drives (E55 / E51 part A) and runs the replay passes.
    Returns ({pass: {rep: trial arrays}}, convergence rows, checks)."""
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
        raise NotImplementedError('wKCKC has a nonzero diagonal (KC self-inhibition)')
    t = np.asarray(get_time_index(mp, rv), dtype=float)
    i0 = int(np.searchsorted(t, mp.time_start - dt / 2)) - 2
    n_resp = m.n_spikes_for_response
    checks.update(taum=taum, dt=dt, n_steps=int(len(t)), kc_kc_inh=kckc, kc_kc_tau=tau, n_spikes_for_response=n_resp,
        wKCKC_nnz=int(W.nnz) if kckc else 0)
    sigma, f_tuned = info['sigma'], info['tuned_fixed_thr']
    n_rec, n_tr, seed = info['n_recordings'], info['n_trials'], info['seed']
    assert abs(m.f0 - info['fixed_thr_det']) < 1e-6
    thr_list = []
    for rep in range(n_rec):
        for xi in E43.b_draws(seed, name, m.draw_kinds['rec'], rep, n_tr, m.draw_size):
            thr_list.append(m.thr_trial(f_tuned, sigma, xi)[0].ravel())
    n_trials = len(thr_list)
    spk = {p: {cond: np.zeros((n_trials, m.n_kcs, len(ODORS)), dtype=np.int32) for cond in CONDS} for p in range(1, n_pass + 1)}
    conv_rows = []
    t_pass = {p: 0.0 for p in range(1, n_pass + 1)}
    mp.kc.save_vm_sims = True
    try:
        for cond, p in m.params.items():
            t1 = time.time()
            r0, s0 = E43.to_arrays(*m.sim(dict(p, fixed_thr=1e12)))
            assert s0.sum() == 0
            vms = rv.kc.vm_sims
            d_free = {}
            for j, o in enumerate(ODORS):
                i = m.pn_idx[j]
                V = np.asarray(vms[i])
                vms[i] = None
                d_free[o] = E51C.drive_from_vm(V[:, i0:], taum, dt)
            del vms
            r, s = E43.to_arrays(*m.sim(p))
            mism = int((r != m.det[cond][0]).sum() + (s != m.det[cond][1]).sum())
            assert mism == 0, f'{name} {cond}: the run with the voltages saved differs from the deterministic run ({mism})'
            thr = np.asarray(rv.kc.thr, dtype=float).ravel()
            vms = rv.kc.vm_sims
            n_bad_spikes, n_bad_replay = 0, 0
            for j, o in enumerate(ODORS):
                i = m.pn_idx[j]
                V = np.asarray(vms[i])[:, i0:]
                vms[i] = None
                prev, cur = V[:, :-1], V[:, 1:]
                spike = (cur == 0) & (prev != 0)
                n_bad_spikes += int((spike.sum(axis=1) != s[:, j]).sum())
                D0 = d_free.pop(o)
                a_det = filtered(spike, tau, decay) if kckc else None
                D_det = D0 - (W @ a_det) if kckc else D0
                n1 = replay_spikes(D_det, thr, k).sum(axis=1)
                n_bad_replay += int((n1 != s[:, j]).sum())
                # pass 1 (E55's replay model), then the passes with the replayed spikes feeding the inhibition
                S_prev = [None] * n_trials
                for pss in range(1, n_pass + 1):
                    t2 = time.time()
                    S_new = []
                    for q, th in enumerate(thr_list):
                        if pss == 1:
                            D = D_det
                        else:
                            D = D0 - (W @ filtered(S_prev[q], tau, decay)) if kckc else D0
                        S = replay_spikes(D, th, k)
                        spk[pss][cond][q, :, j] = S.sum(axis=1)
                        S_new.append(S)
                    t_pass[pss] += time.time() - t2
                    if pss > 1:
                        prev_resp = spk[pss - 1][cond][:, :, j] >= n_resp
                        new_resp = spk[pss][cond][:, :, j] >= n_resp
                        conv_rows.append(dict(model=name, condition=cond, odor=o, pass_from=pss - 1, pass_to=pss,
                            frac_response_changed=float((prev_resp != new_resp).mean()),
                            frac_count_changed=float((spk[pss - 1][cond][:, :, j] != spk[pss][cond][:, :, j]).mean()),
                            n_entries=int(prev_resp.size)))
                    S_prev = S_new
                del S_prev, S_new
            del vms
            checks[f'n_kc_odor_spike_count_from_vm_resets_mismatch_{cond}'] = n_bad_spikes
            checks[f'n_kc_odor_replay_spike_count_mismatch_{cond}'] = n_bad_replay
            assert n_bad_spikes == 0 and n_bad_replay <= 5, (n_bad_spikes, n_bad_replay)
            print(f'{name}: {COND_LABEL[cond]} reconstructed and {n_pass} passes of {n_trials} trials in {time.time() - t1:.0f}s '
                f'(replay at f = 1: {n_bad_replay} mismatches)', flush=True)
    finally:
        mp.kc.save_vm_sims = False
    r, s = E43.to_arrays(*m.sim(m.params['on']))
    checks['after_mismatch_on'] = int((r != m.det['on'][0]).sum() + (s != m.det['on'][1]).sum())
    assert checks['after_mismatch_on'] == 0
    m.close()
    checks.update(pass_seconds={str(p): v for p, v in t_pass.items()}, total_seconds=time.time() - t0, n_passes=n_pass,
        replay=dict(seed=seed, sigma=sigma, tuned_fixed_thr=f_tuned, n_recordings=n_rec, n_trials=n_tr))
    trials = {}
    for pss in range(1, n_pass + 1):
        trials[pss] = {}
        for rep in range(n_rec):
            sl = slice(rep * n_tr, (rep + 1) * n_tr)
            a = {}
            for cond in CONDS:
                a[f'{cond}_spk'] = spk[pss][cond][sl]
                a[f'{cond}_resp'] = spk[pss][cond][sl] >= n_resp
            trials[pss][rep] = a
    return trials, conv_rows, checks


def load_cached(npz: Path) -> dict:
    a = E43.load_npz(npz)
    n_rec, n_pass = int(a['n_recordings']), int(a['n_passes'])
    return {p: {rep: {key: a[f'p{p}_{key}_rep{rep:02d}'] for key in (f'{c}_{k}' for c in CONDS for k in ('spk', 'resp'))}
        for rep in range(n_rec)} for p in range(1, n_pass + 1)}


def make_figures(pm: pd.DataFrame, out: Path, n_pass: int) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {1: '#2b6cb0', 2: '#d9480f', 3: '#6a4c93', 4: '#2f9e44'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, title in ((axes[0], 'block_raw', 'block through the noise, three-trial means'), (axes[1], 'floor1_on', 'single-trial floor, APL on')):
        lim = (min(pm[f'obs_{col}'].min(), pm[f'pred_{col}'].min()) - 0.01, max(pm[f'obs_{col}'].max(), pm[f'pred_{col}'].max()) + 0.01)
        ax.plot(lim, lim, ':', color='0.5', lw=1)
        for pss in range(1, n_pass + 1):
            g = pm[pm['pass'] == pss]
            ax.scatter(g[f'obs_{col}'], g[f'pred_{col}'], s=30, color=colors.get(pss, 'k'), label=f'pass {pss}' + (' (E55\'s replay model)' if pss == 1 else ''), zorder=3)
        g = pm[pm['pass'] == n_pass]
        for _, r in g.iterrows():
            ax.annotate(r['short'], (r[f'obs_{col}'], r[f'pred_{col}']), fontsize=6.5, textcoords='offset points', xytext=(4, 3))
        ax.set(xlim=lim, ylim=lim, xlabel=f'{title}, pipeline', ylabel=f'{title}, replay passes', title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('The network-aware margin proxy: each replay pass against the pipeline (pass k + 1 feeds each KC the inhibition from the other KCs\' pass-k spikes)', fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / 'E56_passes.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, col, title in ((axes[0], 'floor1_on', 'the network\'s share of the APL-on floor (pipeline minus pass)'),
            (axes[1], 'block_raw', 'the network\'s share of the block (pipeline minus pass)')):
        for pss in range(1, n_pass + 1):
            g = pm[pm['pass'] == pss]
            ax.scatter(g['kckc_over_thr'], g[f'obs_{col}'] - g[f'pred_{col}'], s=30, color=colors.get(pss, 'k'), label=f'pass {pss}', zorder=3)
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set(xlabel='KC-KC scale / fixed threshold', ylabel='pipeline minus pass', title=title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'E56_network_share.png', dpi=150)
    plt.close(fig)


def write_table(pm: pd.DataFrame, agree: pd.DataFrame, conv: pd.DataFrame, out: Path, n_pass: int) -> None:
    passes = list(range(1, n_pass + 1))
    head = ('| model | KC-KC scale / fixed threshold | block, deterministic | block through the noise, three-trial means: pipeline / '
        + ' / '.join(f'pass {p}' for p in passes) + ' | grand mean of 30 trials: pipeline / ' + ' / '.join(f'pass {p}' for p in passes)
        + ' | single-trial floor, APL on: pipeline / ' + ' / '.join(f'pass {p}' for p in passes)
        + ' | response rate per trial, APL off: pipeline / ' + ' / '.join(f'pass {p}' for p in passes)
        + ' | responses changed, pass 1 to 2 / 2 to 3 (APL on) | seconds per pass |\n|---|' + '---:|' * 8 + '\n')
    rows = []
    for mname, g in pm.groupby('model', sort=False):
        g = g.set_index('pass')
        r1 = g.loc[1]
        c = conv[(conv.model == mname) & (conv.condition == 'on')].groupby('pass_to')['frac_response_changed'].mean()
        rows.append(f"| {r1['label']} | {E55.fmt(r1['kc_kc_scale'], '.1f')} / {E55.fmt(r1['fixed_thr_det'], '.0f')} | {E55.fmt(r1['block_det'])} | "
            f"{E55.fmt(r1['obs_block_raw'])} / " + ' / '.join(E55.fmt(g.loc[p, 'pred_block_raw']) for p in passes) + ' | '
            f"{E55.fmt(r1['obs_block_grand'])} / " + ' / '.join(E55.fmt(g.loc[p, 'pred_block_grand']) for p in passes) + ' | '
            f"{E55.fmt(r1['obs_floor1_on'], '.2f')} / " + ' / '.join(E55.fmt(g.loc[p, 'pred_floor1_on'], '.2f') for p in passes) + ' | '
            f"{E55.fmt(r1['obs_rate_off'], '.3f')} / " + ' / '.join(E55.fmt(g.loc[p, 'pred_rate_off'], '.3f') for p in passes) + ' | '
            + ' / '.join(E55.fmt(100 * c.get(p, np.nan), '.1f') + ' %' for p in passes[1:]) + ' | '
            + ' / '.join(E55.fmt(g.loc[p, 'pass_seconds'], '.0f') for p in passes) + ' |')
    txt = head + '\n'.join(rows) + '\n\nAgreement with the pipeline across models, per pass:\n\n| statistic | pass | n | Spearman ρ | Pearson r | bias (pass minus pipeline) | mean absolute error |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for _, a in agree.iterrows():
        txt += f"| {a['statistic']} | {int(a['pass'])} | {int(a['n'])} | {E55.fmt(a.get('spearman'), '.2f')} | {E55.fmt(a.get('pearson'), '.2f')} | {E55.fmt(a.get('bias'), '+.4f')} | {E55.fmt(a.get('mae'), '.4f')} |\n"
    (out / 'comparison_table.md').write_text(txt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--models', nargs='+', default=None, help="model names as the part-B tables name them (default: E55's set)")
    ap.add_argument('--passes', type=int, default=3, help='replay passes (pass 1 is the margin-only replay)')
    ap.add_argument('--resim', action='store_true')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    out = args.out or H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    loc = H.outputs_dir(EXPERIMENT)
    loc.mkdir(parents=True, exist_ok=True)
    orn = H.apl16_orn_deltas()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    infos = E55.pipeline_models()
    names = args.models or list(infos)
    missing = [n for n in names if n not in infos]
    if missing:
        raise SystemExit(f'no matched pipeline rows for {missing}')
    inputs, checks_all, pm_rows, pp_rows, conv_all = {}, {}, [], [], []
    e55 = pd.read_csv(H.results_dir('E55_margin_proxy') / 'per_model.csv').set_index('model') if (H.results_dir('E55_margin_proxy') / 'per_model.csv').exists() else None
    for n in names:
        info = dict(infos[n])
        src = E55.register(n)
        obs = E55.observed(n, info, inputs)
        npz, ck = loc / f'{n}_passes.npz', loc / f'{n}_checks.json'
        cached_passes = int(json.loads(ck.read_text()).get('n_passes', 0)) if ck.exists() else 0
        if npz.exists() and ck.exists() and not args.resim and cached_passes >= args.passes:
            trials = load_cached(npz)
            checks = json.loads(ck.read_text())
            conv = pd.read_csv(loc / f'{n}_convergence.csv').to_dict('records')
            print(f'{n}: from {npz}', flush=True)
        else:
            trials, conv, checks = simulate_passes(n, info, orn, loc, args.passes)
            flat = {f'p{p}_{key}_rep{rep:02d}': arr for p, byrep in trials.items() for rep, a in byrep.items() for key, arr in a.items()}
            flat.update(n_recordings=np.asarray(len(trials[1])), n_passes=np.asarray(len(trials)), seed=np.asarray(info['seed']))
            E43.save_npz(npz, flat)
            E43.write_json(ck, checks)
            pd.DataFrame(conv).to_csv(loc / f'{n}_convergence.csv', index=False)
        checks['source'] = src
        conv_all += conv
        n_resp = int(checks['n_spikes_for_response'])
        kckc = E55.kc_kc_scale(n, src)
        bdet = obs[('graded', 'det', 'block')]
        for pss, byrep in trials.items():
            pred = E55.process(byrep, n_resp)
            row = dict(model=n, label=E55.label_of(n), short=E55.short_label(n), **{'pass': pss}, kc_kc_scale=kckc,
                fixed_thr_det=info['fixed_thr_det'], kckc_over_thr=kckc / info['fixed_thr_det'], sigma=info['sigma'], block_det=bdet,
                pass_seconds=float(checks['pass_seconds'][str(pss)]), **src)
            for matrix in E55.MATRICES:
                row[f'obs_block_{matrix}'] = obs[('graded', matrix, 'block')]
                row[f'pred_block_{matrix}'] = pred[('graded', matrix, 'block')]
            for cond, cl in (('on', 'on'), ('acute_off', 'off')):
                for key in ('floor1', 'rate_trial', 'mean_dist_raw'):
                    k2 = key.replace('rate_trial', 'rate')
                    row[f'obs_{k2}_{cl}'] = obs[('graded', cond)][key]
                    row[f'pred_{k2}_{cl}'] = pred[('graded', cond)][key]
            if pss == 1 and e55 is not None and n in e55.index:
                row['pass1_minus_e55_block_raw'] = row['pred_block_raw'] - float(e55.loc[n, 'pred_block_raw'])
            pm_rows.append(row)
            for (a, b) in BLOCK_PAIRS:
                pp_rows.append(dict(model=n, **{'pass': pss}, pair=f'{a} / {b}', odor_a=a, odor_b=b,
                    obs_excess_raw=float(obs[('graded', 'raw', 'pairs_excess')][(a, b)]),
                    pred_excess_raw=float(pred[('graded', 'raw', 'pairs_excess')][(a, b)])))
        checks_all[n] = checks
        pm = pd.DataFrame(pm_rows)
        pm.to_csv(out / 'per_model_pass.csv', index=False)
        pd.DataFrame(pp_rows).to_csv(out / 'per_pair_pass.csv', index=False)
        pd.DataFrame(conv_all).to_csv(out / 'convergence.csv', index=False)
        g = pm[pm.model == n].set_index('pass')
        print(f"{n}: block det {bdet:+.3f}; pipeline raw {g.loc[1, 'obs_block_raw']:+.3f}, passes "
            + ' / '.join(f"{g.loc[p, 'pred_block_raw']:+.3f}" for p in g.index) + f"; floor on pipeline {g.loc[1, 'obs_floor1_on']:.3f}, passes "
            + ' / '.join(f"{g.loc[p, 'pred_floor1_on']:.3f}" for p in g.index) + '; pass seconds ' + ' / '.join(f"{g.loc[p, 'pass_seconds']:.0f}" for p in g.index), flush=True)
    pm, pp, conv = pd.DataFrame(pm_rows), pd.DataFrame(pp_rows), pd.DataFrame(conv_all)
    n_pass = int(pm['pass'].max())
    ag_rows = []
    for pss in range(1, n_pass + 1):
        g = pm[pm['pass'] == pss]
        for key, stat, pc, oc in (('block_raw', 'block, three-trial means', 'pred_block_raw', 'obs_block_raw'),
                ('block_grand', 'block, grand mean', 'pred_block_grand', 'obs_block_grand'),
                ('floor_on', 'single-trial floor, APL on', 'pred_floor1_on', 'obs_floor1_on'),
                ('floor_off', 'single-trial floor, APL off', 'pred_floor1_off', 'obs_floor1_off'),
                ('rate_on', 'response rate per trial, APL on', 'pred_rate_on', 'obs_rate_on'),
                ('rate_off', 'response rate per trial, APL off', 'pred_rate_off', 'obs_rate_off'),
                ('mean_dist_on', 'mean pair distance, three-trial means, APL on', 'pred_mean_dist_raw_on', 'obs_mean_dist_raw_on')):
            ag_rows.append(dict(key=key, statistic=stat, **{'pass': pss}, **E55.agreement(g, pc, oc)))
        ag_rows.append(dict(key='pair_excess_raw', statistic='pair excess, three-trial means (all pairs x models)', **{'pass': pss},
            **E55.agreement(pp[pp['pass'] == pss], 'pred_excess_raw', 'obs_excess_raw')))
        ag_rows.append(dict(key='net_floor_vs_kckc', statistic="the network's share of the APL-on floor remaining against KC-KC / threshold", **{'pass': pss},
            **E55.agreement(g.assign(net=g['obs_floor1_on'] - g['pred_floor1_on']), 'kckc_over_thr', 'net')))
    agree = pd.DataFrame(ag_rows)
    agree.to_csv(out / 'agreement_pass.csv', index=False)
    write_table(pm, agree, conv, out, n_pass)
    make_figures(pm, out, n_pass)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E56_network_margin_proxy.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), al_analysis_commit=commit,
        model=H.model_provenance(orn_deltas=orn), approx_conc_note=H.APPROX_CONC_NOTE, models=names, n_passes=n_pass,
        pipeline_info={n: {k: (str(v) if isinstance(v, Path) else v) for k, v in infos[n].items()} for n in names},
        note=('pass 1: E55\'s replay model (each KC\'s drive without KC-KC inhibition minus W applied to the filtered spike trains of the '
        'deterministic run, replayed at the pipeline\'s own jittered thresholds); pass k + 1: the same with the filtered spike trains of '
        'pass k in the same trial (a synchronous iteration of the KC-KC loop over whole trials); trials processed as E43 part B processes '
        'model trials (E55.process)'), inputs=inputs, checks=checks_all)
    E43.write_json(out / 'provenance.json', prov)
    print(f'-> {out}', flush=True)


if __name__ == '__main__':
    main()
