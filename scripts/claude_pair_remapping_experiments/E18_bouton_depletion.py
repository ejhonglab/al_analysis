#!/usr/bin/env python3
"""E18: rate-based vesicle depletion at PN boutons (dynamic, bouton-shared saturation).

Each bouton's output is its post-inhibition PN rate times an available-pool fraction n_b,
dn/dt = (1 - n) / tau_r - p r n (olfsysm `mp.pn.bouton_ves_p`, `bouton_tau_r`; steady state
1 / (1 + p tau_r r)), shared by the ~25 claws a bouton contacts and present with APL
blocked. E16 found a static compressive transfer moves the model away from the coincidence
models, so the expectation is negative unless the dynamics (undepressed onset, bouton
sharing) change that. Bouton base tuned with APL on at 5 % for each (p, tau_r), acute APL
off; scored as E16 (E13 statistics plus the Spearman of each model's pair distances with the
coincidence-2 / -3 models', E16's claw_thr 70 model and the undepleted base). Bouton rates are ~44 Hz
spontaneous (median) and up to ~220 Hz during odors; sustained output saturates at
1 / (p tau_r) Hz, so p 0.003-0.03 is mild to moderate compression, p >= 0.1 strong.

Outputs: <shared>/results/E18_bouton_depletion/ and ./outputs/E18_bouton_depletion/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E18_bouton_depletion.py
    [--p 0.003 0.01 0.03 0.1] [--tau 0.05 0.2 1.0] [--figure-only]
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
from E3_drive_dependent_gain import BOUTON, TARGET, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess
from E16_claw_saturation import reference_pairs, spearman_with_refs

EXPERIMENT = 'E18_bouton_depletion'


def all_refs() -> dict:
    refs = reference_pairs('bouton')
    long = pd.read_csv(H.shared_dir() / 'results' / 'E16_claw_saturation' / 'model_pair_distances_long.csv')
    for cond in ('on', 'acute_off'):
        sel = long[(long.tag == 'bouton_thr70') & (long.condition == cond)]
        refs[f'thr70_{cond}'] = sel.set_index(['odor_a', 'odor_b'])['distance']
    return refs


def make_figure(stats: pd.DataFrame, res_dir: Path) -> None:
    """Statistics vs release probability p, one line per recovery time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    pre = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    panels = [
        ('Spearman with coincidence-3 (APL on)', 'spearman_vs_coinc3_on', None),
        ('Spearman with claw_thr 70 (APL on)', 'spearman_vs_thr70_on', None),
        ('mean pair distance, APL on (data 0.84, graded cosine)', pre + 'on_mean_model', 0.840),
        ('excess sharing, block pairs, APL on (data -0.08)', 'on_excess_block', -0.080),
        ('Spearman with control (APL on)', pre + 'on_spearman', None),
        ('Spearman with TNT (APL off)', pre + 'off_spearman', None),
        ('fruity x small-ester block (data +0.055)', pre + 'block_fruity_small_iso', 0.055),
        ('per-odor rate vs data (APL on)', 'spearman_rate_on_vs_data', None),
        ('APL-off residual vs VCF (data 0.36)', pre + 'off_spearman_resid_vs_vcf_hyp', 0.36),
    ]
    colors = {0.05: '#2b6cb0', 0.2: '#d9480f', 1.0: '#2f855a'}
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.5))
    base = stats[stats.bouton_ves_p.isna()]
    dep = stats[stats.bouton_ves_p.notna()]
    for ax, (title, col, ref) in zip(axes.ravel(), panels):
        if len(base) and col in base and pd.notna(base[col].iloc[0]):
            ax.axhline(float(base[col].iloc[0]), color='0.3', lw=1, ls=':', label='no depletion')
        for tau, ss in dep.groupby('bouton_tau_r'):
            ss = ss.sort_values('bouton_ves_p')
            ax.plot(ss['bouton_ves_p'], ss[col], '-o', color=colors.get(float(tau), '0.5'), ms=5,
                lw=1.3, label=f'tau_r {tau:g} s')
        if ref is not None:
            ax.axhline(ref, color='0.45', lw=1, ls=(0, (1, 2)))
        ax.set_xscale('log')
        ax.set_xticks([0.003, 0.01, 0.03, 0.1])
        ax.set_xticklabels(['0.003', '0.01', '0.03', '0.1'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel('release probability p (per PN spike)')
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, color='0.9', lw=0.6)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('E18: bouton vesicle depletion on the bouton base (5 %, thresholds re-tuned)', fontsize=11)
    fig.tight_layout()
    fig.savefig(res_dir / 'E18_bouton_depletion.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    # bouton rates are ~20-65 Hz spontaneous (median 44) and up to ~220 Hz during odors;
    # sustained output saturates at 1 / (p tau_r) Hz, so p 0.003-0.03 is mild to moderate
    # compression and p >= 0.1 the strong regime
    ap.add_argument('--p', nargs='+', type=float, default=[0.003, 0.01, 0.03, 0.1])
    ap.add_argument('--tau', nargs='+', type=float, default=[0.05, 0.2, 1.0])
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()

    res_dir = H.results_dir(EXPERIMENT)
    out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        make_figure(pd.DataFrame(rows), res_dir)
        return
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
    refs = all_refs()
    params_path = out_dir / 'tuned_params.json'
    tuned = json.loads(params_path.read_text()) if params_path.exists() else {}
    pairs = {}

    jobs = [(None, None)] + [(p, tau) for tau in args.tau for p in args.p]
    for p, tau in jobs:
        tag = 'bouton_base' if p is None else f'bouton_p{p:g}_tau{tau:g}'
        if tag in done:
            continue
        kws = dict(BOUTON)
        if p is not None:
            kws.update(bouton_ves_p=p, bouton_tau_r=tau)
        t0 = time.time()
        try:
            on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag,
                with_pn_contrast=False
            )
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, base='bouton', bouton_ves_p=p, bouton_tau_r=tau,
                error=repr(e)[:300]))
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            continue
        tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items()
            if k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')}
        params_path.write_text(json.dumps(tuned, indent=1))
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag,
            bouton_ves_p=p, bouton_tau_r=tau,
            steady_pool_spont=(None if p is None else 1.0 / (1.0 + p * tau * 44.0)),
            steady_pool_peak=(None if p is None else 1.0 / (1.0 + p * tau * 220.0)),
            onset_tau_s=(None if p is None else 1.0 / (p * 220.0 + 1.0 / tau)),
            n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1],
            rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
            **extra, **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale',
                'wAPLPN_scale', 'wPNAPL_scale') if k in params},
        )
        meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        if p is None:
            refs.update({'base_on': on_p, 'base_acute_off': off_p})
        row.update(spearman_with_refs(on_p, off_p, refs))
        rows.append(row)
        pairs[(tag, 'on')] = on_p
        pairs[(tag, 'acute_off')] = off_p
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        v = H.PRIMARY_MODEL_RDM
        print(f"{tag}: thr {meta.get('fixed_thr_mean', float('nan')):.0f}, rates on {meta['rate_on']:.4f} off "
            f"{meta['rate_off']:.4f}; mean dist on {row[v + '__on_mean_model']:.3f}; block "
            f"{row[v + '__block_fruity_small_iso']:+.3f}; excess on {row['on_excess_block']:+.3f}/"
            f"{row['on_excess_other_between']:+.3f} off {row['off_excess_block']:+.3f}/{row['off_excess_other_between']:+.3f}; "
            f"vs ctrl {row[v + '__on_spearman']:.3f} dev {row[v + '__on_spearman_dev']:+.2f}; off vs TNT "
            f"{row[v + '__off_spearman']:.3f}; vs coinc3/thr70 on {row.get('spearman_vs_coinc3_on', float('nan')):.2f}/"
            f"{row.get('spearman_vs_thr70_on', float('nan')):.2f}; rate vs data {row['spearman_rate_on_vs_data']:.2f}; "
            f"{meta['seconds']:.0f}s", flush=True
        )
        done.add(tag)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
            ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
            mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
        pairs.clear()

    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E18_bouton_depletion.py', host=subprocess.run(['hostname'], capture_output=True,
            text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=H.olfsysm_dir(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, p=args.p, tau=args.tau, approx_conc_note=H.APPROX_CONC_NOTE,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
