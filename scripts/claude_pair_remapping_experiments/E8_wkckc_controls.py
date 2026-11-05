#!/usr/bin/env python3
"""E8 addendum: does the connectome's KC-KC weight structure matter? Shuffled and uniform
controls at E19's operating points.

Reruns E19's balanced (claw_thr 40, k 2, kc_kc_inh 30) and block-first (claw_thr 70, k 2,
kc_kc_inh 100) variants with the KC-KC weight matrix replaced by (a) the connectome matrix
with rows and columns permuted by the same random permutation (degree structure kept, KC
identities scrambled; two seeds), (b) a uniform all-to-all matrix with the same mean
incoming weight (diagonal zero). The connectome matrix itself is rerun as the reference.
Scored as E19.

Outputs: <shared>/results/E8_kc_kc_inhibition/wkckc_controls.csv and
./outputs/E8_kc_kc_inhibition/controls/. Usage: OMP_NUM_THREADS=16 nice -n 10
./venv/bin/python E8_wkckc_controls.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E1_apl_knockout import bisect_threshold
from E3_drive_dependent_gain import BOUTON, match_kc_apl_scale, per_odor_extras, rate
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess

E16_OUT = H.HERE / 'outputs' / 'E16_claw_saturation'
TAU = 0.05
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
POINTS = [(40.0, 2.0, 30.0), (70.0, 2.0, 100.0)]


def main() -> None:
    from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator, connectome_KCKC_weights
    res_dir = H.results_dir('E8_kc_kc_inhibition')
    out_dir = H.outputs_dir('E8_kc_kc_inhibition') / 'controls'
    out_dir.mkdir(exist_ok=True)
    stats_path = res_dir / 'wkckc_controls.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    done = {r['tag'] for r in rows}
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
    tuned = json.loads((E16_OUT / 'tuned_params.json').read_text())

    for thr, k, s in POINTS:
        src = f'bouton_thr{thr:g}'
        initial = {kk: float(tuned[src][kk]) for kk in PARAM_KEYS}
        thr_kws = dict(BOUTON, make_plots=False, silent=True, claw_thr=thr)
        _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'],
            wAPLKC=initial['wAPLKC_scale'], wKCAPL=initial['wKCAPL_scale'],
            wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **thr_kws
        )
        wnorm = {kk: pd0[kk] / pd0[f'{kk}_scale'] for kk in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
        claw_kc = wnorm['wAPLKC'].index.get_level_values(H.KC_ID_LEVEL)
        kc_ids = list(dict.fromkeys(claw_kc))
        off_spk = pd.read_parquet(E16_OUT / f'{src}_acute_off_spike_counts.parquet')
        b = pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(),
            index=off_spk.index.get_level_values(H.KC_ID_LEVEL)) + 1.0
        f_claw = ((b / b.mean()) ** k).reindex(claw_kc).fillna(1.0).to_numpy()
        on_resp0 = pd.read_parquet(E16_OUT / f'{src}_on_responses.parquet')
        n_target = int((on_resp0.astype(float) != 0).to_numpy().sum())
        rate_target = rate(on_resp0)
        W = connectome_KCKC_weights(kc_ids)
        Wm = W.to_numpy(dtype=float)
        mats = {'connectome': W}
        for seed in (0, 1):
            perm = np.random.default_rng(seed).permutation(len(kc_ids))
            mats[f'shuffled{seed}'] = pd.DataFrame(Wm[np.ix_(perm, perm)], index=W.index, columns=W.columns)
        U = np.full_like(Wm, Wm.sum() / (Wm.shape[0] * (Wm.shape[0] - 1)))
        np.fill_diagonal(U, 0.0)
        mats['uniform'] = pd.DataFrame(U, index=W.index, columns=W.columns)
        p_k = None
        for name, mat in mats.items():
            tag = f'thr{thr:g}_k{k:g}_kckc{s:g}_{name}'
            if tag in done:
                continue
            t0 = time.time()
            sim, _ = make_fixed_param_kc_simulator(orn, initial_params=initial, **thr_kws,
                _wAPLKC=wnorm['wAPLKC'] * f_claw, _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'],
                _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=0.0, kc_kc_tau=TAU, _wKCKC=mat
            )
            if p_k is None:
                f, _, _ = match_kc_apl_scale(sim, dict(initial, wKCKC_scale=0.0), n_target)
                p_k = dict(initial, wAPLKC_scale=initial['wAPLKC_scale'] * f,
                    wKCAPL_scale=initial['wKCAPL_scale'] * f, wKCKC_scale=0.0)
            extra = {kk: p_k[kk] for kk in ('wAPLKC_scale', 'wKCAPL_scale')}
            extra['wKCKC_scale'] = s
            new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_k['fixed_thr'], rate_target, extra=extra)
            p_on = dict(p_k, fixed_thr=new_thr, wKCKC_scale=s)
            r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
            del sim
            for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
                r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
                sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
            meta = dict(variant=tag, base='bouton', seed=None, target_rate=0.05, tag=tag, claw_thr=thr, k=k,
                kc_kc_inh=s, wkckc=name, thr_factor=thr_f, n_kcs=len(r_on), n_glomeruli=wPNKC.shape[1],
                rate_on=rate(r_on), rate_off=rate(r_off), seconds=time.time() - t0,
                mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
                wkckc_mean_in=float(np.asarray(mat, dtype=float).sum(axis=1).mean()),
                wkckc_cv_in=float(np.std(np.asarray(mat, dtype=float).sum(axis=1)) / np.mean(np.asarray(mat, dtype=float).sum(axis=1))),
            )
            meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
            row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs, drive, data, ref)
            row.update(per_odor_extras(r_on, r_off, on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM], drive, data_rates))
            for cond, sp in (('on', s_on), ('off', s_off)):
                row.update({f'{cond}_{kk}': v for kk, v in density_excess(sp).items()})
            rows.append(row)
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            v = H.PRIMARY_MODEL_RDM
            print(f"{tag}: thr x{thr_f:.3f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f}; mean dist on "
                f"{row[v + '__on_mean_model']:.3f}; block {row[v + '__block_fruity_small_iso']:+.3f}; excess on "
                f"{row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f}; vs ctrl {row[v + '__on_spearman']:.3f} "
                f"dev {row[v + '__on_spearman_dev']:+.2f}; off vs TNT {row[v + '__off_spearman']:.3f}; rate vs data "
                f"{row['spearman_rate_on_vs_data']:.2f}; {meta['seconds']:.0f}s", flush=True
            )
    print('done ->', stats_path)


if __name__ == '__main__':
    main()
