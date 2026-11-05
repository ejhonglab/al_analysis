#!/usr/bin/env python3
"""E10: joint fit of one parameter set to the control (APL on) and TNT (APL off) matrices.

Family (the general one, per E8 / E12 / E17, not E15's block-specific rule): bouton base
(E3's wiring; connectome per-claw APL weights, PN-bouton <-> APL) with each KC's APL->KC
weight multiplied by ((breadth + 1) / mean(breadth + 1))^k (E12's breadth weighting, breadth
= total APL-off spike count of the KC in E3's tuned bouton model, regenerated locally), k on a
grid, plus olfsysm's connectome-weighted KC-KC lateral inhibition (E8; `kc_kc_inh`, tau
0.05 s). Per k, Nelder-Mead (`optimize_mb_model_params`, two conditions: 'on' = the fitted
parameters, 'off' = the same with wAPLKC_scale = wAPLPN_scale = 0) over fixed_thr,
wAPLKC_scale (wKCAPL / wPNAPL tied), wAPLPN_scale and wKCKC_scale (= kc_kc_inh).

Loss (each term ~1 when bad, 0 when matched; weights in LOSS_WEIGHTS):
  on_mat    1 - Spearman(model APL on, control; 120 pairs, 1 - cosine over spike counts)
  off_mat   1 - Spearman(model APL off, TNT)
  block     ((block residual - 0.055) / 0.05)^2; E7's on-vs-off fruity x small-ester block
            (isotonic residual), data +0.055 [0.035, 0.075]: the discriminating target
  rate_on   ((mean APL-on response rate - 0.05) / 0.01)^2 (the usual 5 % tuning target)
  ratio     ((log(off / on rate ratio) - log(data ratio)) / log 1.5)^2; data = ratio of the
            mean response fractions of TNT and TNT3c over the 16 odors (2.2; E8 quoted 1.9)
  iso       max(0, 0.94 - isotonic R2 of off vs on) / 0.05 (data 0.94; monotone on/off relation)
  po_on     1 - Spearman(per-odor APL-on rates, TNT3c per-odor fractions)
  po_off    1 - Spearman(per-odor APL-off rates, TNT per-odor fractions)
Reported but not fitted: E7's density-null excess profile and E4's full score of each best fit.

Outputs: <shared>/results/E10_joint_fit/ (fits.csv, history_k<k>.csv, best-fit score rows,
figure) and ./outputs/E10_joint_fit/ (responses / spike counts of each best fit).
Usage: OMP_NUM_THREADS=6 nice -n 10 ./venv/bin/python E10_joint_fit.py -k 3 1 [--maxfev 150]
       [--kckc0 60] [--method Nelder-Mead|direct|differential_evolution] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON, per_odor_extras, rate
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import e3_bouton_params, density_excess

EXPERIMENT = 'E10_joint_fit'
TAU = 0.05
BLOCK_TARGET, BLOCK_SCALE = 0.055, 0.05
RATE_TARGET, RATE_SCALE = 0.05, 0.01
RATIO_LOG_SCALE = np.log(1.5)
ISO_TARGET, ISO_SCALE = 0.94, 0.05
LOSS_WEIGHTS = dict(on_mat=1.0, off_mat=1.0, block=1.0, rate_on=1.0, ratio=0.5, po_on=0.5,
    po_off=0.5, iso=0.25)
# E12's KC<->APL factor matching the breadth-weighted model's response count to k = 0's
# (spikes_k<k> rows of its stats): used only as the initial wAPLKC_scale per k
E12_FACTORS = {0.0: 1.0, 0.5: 0.82, 1.0: 0.65, 2.0: 0.36, 3.0: 0.18, 4.0: 0.085}


def bouton_base(orn: pd.DataFrame, out_dir: Path):
    """E3's bouton base at its tuned parameters: normalized APL weights, wPNKC, and the
    per-KC APL-off breadth (total spike count), regenerated here if not saved."""
    from al_analysis.mb_model import fit_mb_model
    initial = e3_bouton_params()
    _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'],
        wAPLKC=initial['wAPLKC_scale'], wKCAPL=initial['wKCAPL_scale'],
        wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **BOUTON,
        make_plots=False, silent=True
    )
    wnorm = {k: pd0[k] / pd0[f'{k}_scale'] for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
    # hal's E3 APL-off run (the breadth E12 / E8 used), copied to the shared directory;
    # regenerated locally only if it is not there
    hal_path = (H.shared_dir() / 'data' / 'hal_outputs' / 'E3_drive_dependent_gain' /
        'bouton_pnapl_x1_acute_off_spike_counts.parquet')
    off_path = out_dir / 'base_acute_off_spike_counts.parquet'
    if hal_path.exists():
        off_spk = pd.read_parquet(hal_path)
        print(f'breadth from {hal_path}', flush=True)
    elif off_path.exists():
        off_spk = pd.read_parquet(off_path)
    else:
        r_off, off_spk, _, _ = fit_mb_model(orn, fixed_thr=initial['fixed_thr'], wAPLKC=0.0,
            wKCAPL=initial['wKCAPL_scale'], wAPLPN=0.0, wPNAPL=initial['wPNAPL_scale'],
            **BOUTON, make_plots=False, silent=True
        )
        off_spk.to_parquet(off_path)
        r_off.to_parquet(out_dir / 'base_acute_off_responses.parquet')
        print(f'base APL-off rate {rate(r_off):.4f} (E3: 0.0978)', flush=True)
    breadth = pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(),
        index=off_spk.index.get_level_values(H.KC_ID_LEVEL)
    )
    return initial, wnorm, wPNKC, breadth


def breadth_weights(wAPLKC0: pd.Series, breadth: pd.Series, k: float) -> pd.Series:
    bp1 = breadth + 1.0
    f = ((bp1 / bp1.mean()) ** k).reindex(wAPLKC0.index.get_level_values(H.KC_ID_LEVEL))
    return wAPLKC0 * f.fillna(1.0).to_numpy()


def make_loss(data: dict, data_rates: Dict[str, pd.Series], orn_pairs: pd.Series,
    drive: pd.Series):
    """Joint loss over {'on': (responses, spike_counts), 'off': ...} (see module doc)."""
    ratio_target = float(data_rates['TNT'].reindex(H.APL16_ODORS).mean() /
        data_rates['TNT3c'].reindex(H.APL16_ODORS).mean())
    ctrl = data['raw']['TNT3c']
    tnt = data['raw']['TNT']
    v = H.PRIMARY_MODEL_RDM

    def loss_fn(outputs) -> Dict[str, float]:
        r_on, s_on = outputs['on']
        r_off, s_off = outputs['off']
        on = H.model_rdms(r_on, s_on)[v]
        off = H.model_rdms(r_off, s_off)[v]
        out = {}
        out['on_spearman'] = H.spearman(on, ctrl)
        out['off_spearman'] = H.spearman(off, tnt)
        try:
            st, _ = H.compare_on_off(on, off, orn=orn_pairs, drive=drive)
            out['block'] = float(st['block_fruity_small_iso'])
            out['iso_r2'] = float(st['iso_r2'])
        except Exception:
            out['block'], out['iso_r2'] = np.nan, np.nan
        out['rate_on'] = rate(r_on)
        out['rate_off'] = rate(r_off)
        out['ratio'] = out['rate_off'] / max(out['rate_on'], 1e-9)
        po_on = H.response_fractions(r_on).reindex(H.APL16_ODORS)
        po_off = H.response_fractions(r_off).reindex(H.APL16_ODORS)
        out['po_on_spearman'] = float(stats.spearmanr(po_on,
            data_rates['TNT3c'].reindex(H.APL16_ODORS))[0])
        out['po_off_spearman'] = float(stats.spearmanr(po_off,
            data_rates['TNT'].reindex(H.APL16_ODORS))[0])
        terms = dict(
            on_mat=1.0 - out['on_spearman'],
            off_mat=1.0 - out['off_spearman'],
            block=((np.nan_to_num(out['block'], nan=-0.2) - BLOCK_TARGET) / BLOCK_SCALE) ** 2,
            rate_on=((out['rate_on'] - RATE_TARGET) / RATE_SCALE) ** 2,
            ratio=((np.log(max(out['ratio'], 1e-3)) - np.log(ratio_target)) / RATIO_LOG_SCALE) ** 2,
            po_on=1.0 - np.nan_to_num(out['po_on_spearman'], nan=0.0),
            po_off=1.0 - np.nan_to_num(out['po_off_spearman'], nan=0.0),
            iso=max(0.0, ISO_TARGET - np.nan_to_num(out['iso_r2'], nan=0.0)) / ISO_SCALE,
        )
        for name, t in terms.items():
            out[f'term_{name}'] = float(LOSS_WEIGHTS[name] * t)
        out['loss'] = float(sum(out[f'term_{n}'] for n in terms))
        return {kk: float(vv) for kk, vv in out.items()}

    loss_fn.ratio_target = ratio_target
    return loss_fn


def fit_one(k: float, orn, initial, wnorm, breadth, loss_fn, *, kckc0: float, maxfev: int,
    method: str, thr0_factor: float, verbose: bool = True):
    from al_analysis.mb_model import optimize_mb_model_params
    wAPLKC = breadth_weights(wnorm['wAPLKC'], breadth, k)
    f0 = E12_FACTORS.get(k, float(np.exp(np.interp(k, sorted(E12_FACTORS),
        np.log([E12_FACTORS[x] for x in sorted(E12_FACTORS)])))))
    init = dict(fixed_thr=initial['fixed_thr'] * thr0_factor,
        wAPLKC_scale=initial['wAPLKC_scale'] * f0, wKCAPL_scale=initial['wKCAPL_scale'] * f0,
        wAPLPN_scale=initial['wAPLPN_scale'], wPNAPL_scale=initial['wPNAPL_scale'],
        wKCKC_scale=kckc0,
    )
    bounds = {
        'fixed_thr': (initial['fixed_thr'] / 4, initial['fixed_thr'] * 4),
        'wAPLKC_scale': (init['wAPLKC_scale'] / 30, init['wAPLKC_scale'] * 30),
        'wAPLPN_scale': (init['wAPLPN_scale'] / 30, init['wAPLPN_scale'] * 30),
        'wKCKC_scale': (0.0, 300.0),
    }
    opt_kws = {}
    if method == 'differential_evolution':
        opt_kws = dict(popsize=6, seed=0, polish=False, init='sobol')
    res = optimize_mb_model_params(orn, loss_fn=loss_fn,
        condition_param_overrides={'on': {}, 'off': {'wAPLKC_scale': 0.0, 'wAPLPN_scale': 0.0}},
        params_to_optimize=['fixed_thr', 'wAPLKC_scale', 'wAPLPN_scale', 'wKCKC_scale'],
        initial_params=init, bounds=bounds, tie_apl_weights=True, method=method,
        maxfev=maxfev, optimizer_kws=opt_kws, verify=False, verbose=verbose, **BOUTON,
        _wAPLKC=wAPLKC, _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'],
        _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=kckc0, kc_kc_tau=TAU, make_plots=False,
        silent=True
    )
    return res, init, bounds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-k', '--exponents', nargs='+', type=float, default=[3.0, 2.0, 1.0, 0.0])
    ap.add_argument('--maxfev', type=int, default=150)
    ap.add_argument('--kckc0', type=float, default=60.0, help='initial KC-KC scale')
    ap.add_argument('--thr0-factor', type=float, default=0.7,
        help='initial threshold as a factor of E3 bouton threshold (E8: 0.6-0.77 at KC-KC 30-100)')
    ap.add_argument('--method', default='Nelder-Mead')
    ap.add_argument('--tag', default='', help='suffix for this run (e.g. a second start)')
    ap.add_argument('--smoke', action='store_true', help='time a few evaluations and exit')
    args = ap.parse_args()

    res_dir = H.results_dir(EXPERIMENT)
    out_dir = H.outputs_dir(EXPERIMENT)
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {
        'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')},
    }
    data_rates = {g: H.yang_response_fractions(g).mean() for g in ('TNT3c', 'TNT')}
    loss_fn = make_loss(data, data_rates, orn_pairs, drive)
    print(f'data off/on rate ratio target {loss_fn.ratio_target:.3f}', flush=True)

    t0 = time.time()
    initial, wnorm, wPNKC, breadth = bouton_base(orn, out_dir)
    print(f'bouton base ready ({time.time() - t0:.0f}s); E3 params {initial}', flush=True)

    fits_path = res_dir / 'fits.csv'
    for k in args.exponents:
        tag = f'k{k:g}{args.tag}'
        t0 = time.time()
        res, init, bounds = fit_one(k, orn, initial, wnorm, breadth, loss_fn, kckc0=args.kckc0,
            maxfev=5 if args.smoke else args.maxfev, method=args.method,
            thr0_factor=args.thr0_factor
        )
        hist = res['history']
        hist.to_csv(res_dir / f'history_{tag}.csv')
        if args.smoke:
            print(hist[['loss', 'time_s']].describe().T, flush=True)
            print(hist.iloc[-1].to_dict(), flush=True)
            return
        r_on, s_on = res['condition_outputs']['on']
        r_off, s_off = res['condition_outputs']['off']
        for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
            r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
            sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        p = res['params']
        meta = dict(variant=tag, tag=tag, base='bouton', seed=None, k=k, method=args.method,
            maxfev=args.maxfev, n_evals=len(hist), kckc0=args.kckc0, thr0_factor=args.thr0_factor,
            fixed_thr=p['fixed_thr'], thr_factor=p['fixed_thr'] / initial['fixed_thr'],
            wAPLKC_scale=p['wAPLKC_scale'], wKCAPL_scale=p['wKCAPL_scale'],
            wAPLPN_scale=p['wAPLPN_scale'], wPNAPL_scale=p['wPNAPL_scale'],
            kc_kc_inh=p['wKCKC_scale'], kc_kc_tau=TAU,
            apl_factor_vs_e3=p['wAPLKC_scale'] / initial['wAPLKC_scale'],
            aplpn_factor_vs_e3=p['wAPLPN_scale'] / initial['wAPLPN_scale'],
            loss=res['loss'], initial_loss=res['initial_losses']['loss'],
            n_kcs=len(r_on), n_glomeruli=wPNKC.shape[1], rate_on=rate(r_on),
            rate_off=rate(r_off), rate_ratio_off_over_on=rate(r_off) / max(rate(r_on), 1e-9),
            seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
        )
        meta.update({f'fit_{kk}': vv for kk, vv in res['losses'].items() if kk != 'loss'})
        row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs, drive,
            data, ref
        )
        row.update(per_odor_extras(r_on, r_off, on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM],
            drive, data_rates
        ))
        for cond, sp in (('on', s_on), ('off', s_off)):
            row.update({f'{cond}_{kk}': vv for kk, vv in density_excess(sp).items()})
        # re-read right before writing: two processes (different k) share this file
        rows = list(pd.read_csv(fits_path).to_dict('records')) if fits_path.exists() else []
        rows = [r for r in rows if r.get('tag') != tag] + [row]
        pd.DataFrame(rows).to_csv(fits_path, index=False)
        pd.concat({(tag, 'on'): on[H.PRIMARY_MODEL_RDM], (tag, 'acute_off'): off[H.PRIMARY_MODEL_RDM]},
            names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index(
            ).to_csv(res_dir / 'model_pair_distances_long.csv', index=False, mode='a',
            header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
        v = H.PRIMARY_MODEL_RDM
        print(f"{tag} done: loss {res['loss']:.3f} (initial {meta['initial_loss']:.3f}, "
            f"{len(hist)} evals, {meta['seconds']/60:.0f} min); thr x{meta['thr_factor']:.3f}, "
            f"APL x{meta['apl_factor_vs_e3']:.3f}, APL>PN x{meta['aplpn_factor_vs_e3']:.3f}, "
            f"KC-KC {meta['kc_kc_inh']:.1f}; rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f} "
            f"(x{meta['rate_ratio_off_over_on']:.2f}); block {row[v + '__block_fruity_small_iso']:+.3f}; "
            f"vs ctrl {row[v + '__on_spearman']:.3f} dev {row[v + '__on_spearman_dev']:+.2f}; off vs TNT "
            f"{row[v + '__off_spearman']:.3f}; iso R2 {row[v + '__iso_r2']:.3f}; excess on "
            f"{row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f}; per-odor rates "
            f"{row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}", flush=True
        )
        prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            script='E10_joint_fit.py', host=subprocess.run(['hostname'], capture_output=True,
                text=True).stdout.strip(),
            al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
                capture_output=True, text=True).stdout.strip(),
            olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
                cwd=Path('~/src/olfsysm').expanduser(), capture_output=True, text=True).stdout.strip(),
            base=BOUTON, e3_params=initial, loss_weights=LOSS_WEIGHTS, block_target=BLOCK_TARGET,
            block_scale=BLOCK_SCALE, rate_target=RATE_TARGET, ratio_target=loss_fn.ratio_target,
            ratio_log_scale=RATIO_LOG_SCALE, iso_target=ISO_TARGET, iso_scale=ISO_SCALE,
            tau=TAU, bounds={kk: list(vv) for kk, vv in bounds.items()}, initial=init,
            approx_conc_note=H.APPROX_CONC_NOTE,
        )
        (res_dir / f'provenance_{tag}.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
