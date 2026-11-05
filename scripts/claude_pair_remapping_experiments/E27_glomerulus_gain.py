#!/usr/bin/env python3
"""E27: oracle gain on the weak-input odors' top glomeruli (threshold base), and the anatomical
version (per-claw PN->KC synapse counts as claw strength).

The weak-odor analyses (E21, E23, E24, E26) agree that ms, paa, B-myr, euc and 1-8ol are not
recruited by the threshold model because at the PN level they drive two or three glomeruli at
rates no higher than the esters' five to seven, so every rule applied alike to claws favors the
esters. The in-model option left is glomerulus-specific: claws from their top glomeruli
counting for more. Variants on the bouton base with the 70 Hz per-claw threshold (thresholds
tuned at 5 % with APL on, acute APL off):
  none          the threshold model (control);
  oracle_g<g>   PN->claw weights of the weak odors' top glomeruli (VA7m, DL1, D for ms; DM1 for
                B-myr and paa; DP1m, VC1 for euc and -aPine; DC2, DM5 for 1-8ol) scaled by g in
                {1.5, 2, 3} (E20's column scaling of wPNKC, so a true PN->claw gain);
  random_g2_s<k> the same gain on as many randomly drawn ORN-driven glomeruli outside that set
                (control for "any extra drive helps");
  synapses      the anatomical version: each claw weighted by its PN->KC synapse count
                (fit_mb_model claw_weight='synapses'), normalized to mean 1 over claws so the
                70 Hz knee keeps its meaning, passed as _wPNKC.
Scored as E26 (E13 statistics, E7 excess by block, per-odor response fractions, E16 reference
ranks, E21's APL-off sharing deficit by pair type incl. the terpene x fruity-ester subset).

Outputs: <shared>/results/E27_glomerulus_gain/ and ./outputs/E27_glomerulus_gain/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E27_glomerulus_gain.py [--gains 1.5 2 3] [--thr 70]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON, TARGET, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess, e3_bouton_params
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E26_claw_power import sharing_deficit, deficit_table, WEAK

EXPERIMENT = 'E27_glomerulus_gain'
TOP_GLOMS = {'ms': ['VA7m', 'DL1', 'D'], 'paa': ['DM1'], 'B-myr': ['DM1'], 'euc': ['DP1m', 'VC1'], '-aPine': ['VC1'], '1-8ol': ['DC2', 'DM5']}
ORACLE = sorted({g for gs in TOP_GLOMS.values() for g in gs})
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')


def gained(wPNKC: pd.DataFrame, gloms, g: float) -> pd.DataFrame:
    col = wPNKC.columns.get_level_values('glomerulus').isin(gloms)
    out = wPNKC.astype(float).copy()
    out.loc[:, col] = out.loc[:, col] * g
    return out


def synapse_weighted_wPNKC(orn: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The bouton base's wiring with each claw weighted by its PN->KC synapse count
    (claw_weight='synapses'), normalized to mean 1 over the claws that have a weight."""
    from al_analysis.mb_model import fit_mb_model
    init = e3_bouton_params()
    _, _, w, _ = fit_mb_model(orn, fixed_thr=init['fixed_thr'], wAPLKC=init['wAPLKC_scale'], wKCAPL=init['wKCAPL_scale'],
        wAPLPN=init['wAPLPN_scale'], wPNAPL=init['wPNAPL_scale'], **BOUTON, claw_weight='synapses', make_plots=False, silent=True)
    vals = w.to_numpy(dtype=float)
    nz = vals[vals > 0]
    info = dict(n_claws=int((vals > 0).sum()), mean_synapses=float(nz.mean()), median_synapses=float(np.median(nz)), cv=float(nz.std() / nz.mean()),
        min=float(nz.min()), max=float(nz.max()))
    return w.astype(float) / nz.mean(), info


def make_figure(stats: pd.DataFrame, fracs: pd.DataFrame, res_dir) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    tags = list(stats.tag); x = np.arange(len(tags))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]; wdt = 0.13
    for k, (o, c) in enumerate(zip(WEAK, ['#1f1f1f', '#8a6d00', '#2b6cb0', '#d9480f', '#2f855a', '#6b46c1'])):
        ax.bar(x + (k - 2.5) * wdt, [fracs.loc[o, f'{t}_acute_off'] for t in tags], wdt, color=c, label=o)
        ax.axhline(fracs.loc[o, 'data_TNT'], color=c, lw=0.8, ls=':')
    ax.set_title('weak odors: APL-off response fraction (dotted: data)', fontsize=9); ax.legend(fontsize=7, frameon=False, ncol=3)
    panels = [('APL-off sharing deficit (0 = data)', [('deficit_terp_x_fruity', 'terpene / 1-8ol x fruity ester'), ('deficit_terp_pairs', 'terpene / 1-8ol pairs'), ('deficit_ms_paa_pairs', 'ms / paa pairs'), ('deficit_other_pairs', 'other pairs')], [0.0]),
        ('rate rank agreement with the data (ceiling 0.98 / 0.97)', [('spearman_rate_on_vs_data', 'APL on'), ('spearman_rate_off_vs_data', 'APL off')], []),
        ('rank agreement with control (on) / TNT (off)', [(p + 'on_spearman', 'APL on vs control'), (p + 'off_spearman', 'APL off vs TNT')], []),
        ('mean pair distance (data 0.84 on / 0.66 off)', [(p + 'on_mean_model', 'APL on'), (p + 'off_mean_model', 'APL off')], [0.840, 0.661]),
        ('block (data +0.055) and APL-on block-pair excess (data -0.08)', [(p + 'block_fruity_small_iso', 'block'), ('on_excess_block', 'APL-on block-pair excess')], [0.055, -0.08])]
    for ax, (title, cols, refs) in zip(list(axes.ravel())[1:], panels):
        for i, (col, lab) in enumerate(cols):
            ax.plot(x, stats[col].to_numpy(), ['-o', '--s', ':^', '-.d'][i], color=['#2b6cb0', '#2b6cb0', '#d9480f', '#2f855a'][i], ms=5, lw=1.2, label=lab)
        for r in refs:
            ax.axhline(r, color='0.45', lw=1, ls=':')
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha='right', fontsize=8); ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    fig.suptitle("E27: gain on the weak odors' top glomeruli, and synapse-count claw weights, on the bouton base with the 70 Hz per-claw threshold (5 %)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(res_dir / 'E27_glomerulus_gain.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--gains', nargs='+', type=float, default=[1.5, 2.0, 3.0])
    ap.add_argument('--thr', type=float, default=70.0)
    ap.add_argument('--random-seeds', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--no-synapses', action='store_true')
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        make_figure(pd.DataFrame(rows), pd.read_csv(fracs_path, index_col=0), res_dir); return
    done = {r['tag'] for r in rows}
    orn = H.apl16_orn_deltas(); orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    refs = all_refs()
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned = json.loads((out_dir / 'tuned_params.json').read_text()) if (out_dir / 'tuned_params.json').exists() else {}
    driven = [g for g in orn.index]
    variants = [('none', None, None)] + [(f'oracle_g{g:g}', ORACLE, g) for g in args.gains]
    rng_sets = {}
    for seed in args.random_seeds:
        rng = np.random.default_rng(seed)
        pool = [g for g in driven if g not in ORACLE]
        rng_sets[seed] = sorted(rng.choice(pool, size=len(ORACLE), replace=False).tolist())
        variants.append((f'random_g2_s{seed}', rng_sets[seed], 2.0))
    if not args.no_synapses:
        variants.append(('synapses', 'synapses', None))
    (res_dir / 'glomerulus_sets.json').write_text(json.dumps(dict(oracle=ORACLE, top_glomeruli=TOP_GLOMS, random={str(k): v for k, v in rng_sets.items()}), indent=1) + '\n')
    wPNKC0 = None; syn_info = {}
    pairs = {}; deficits = []
    for tag, gloms, g in variants:
        if tag in done:
            continue
        kws = dict(BOUTON, claw_thr=args.thr)
        if gloms == 'synapses':
            W, syn_info = synapse_weighted_wPNKC(orn); kws['_wPNKC'] = W
            (res_dir / 'synapse_weights_info.json').write_text(json.dumps(syn_info, indent=1) + '\n')
        elif gloms is not None:
            if wPNKC0 is None:
                raise RuntimeError('the control must run first (it provides the base wiring)')
            kws['_wPNKC'] = gained(wPNKC0, gloms, g)
        t0 = time.time()
        try:
            on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag, with_pn_contrast=False)
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, claw_thr=args.thr, gain=g, error=repr(e)[:300])); pd.DataFrame(rows).to_csv(stats_path, index=False); continue
        if tag == 'none':
            wPNKC0 = wPNKC
        tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
        tuned[tag].update(claw_thr=args.thr, gain=g, glomeruli=(gloms if isinstance(gloms, list) else gloms))
        (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1))
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag, claw_thr=args.thr, gain=g if g is not None else 1.0,
            glomeruli=','.join(gloms) if isinstance(gloms, list) else (gloms or ''),
            n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()), **extra,
            **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale') if k in params})
        meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        d = deficit_table(off_out[1], data_pairs); d.insert(0, 'tag', tag); deficits.append(d)
        rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index); fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        v = H.PRIMARY_MODEL_RDM
        wk = ' '.join(f"{o}:{fracs.loc[o, f'{tag}_acute_off']:.3f}" for o in WEAK)
        print(f"{tag}: thr {meta.get('fixed_thr_mean', float('nan')):.0f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f}; weak odors off [{wk}] (data "
            f"{' '.join(f'{o}:{fracs.loc[o, 'data_TNT']:.3f}' for o in WEAK)}); deficit terp-x-fruity/terp/ms-paa/other {row['deficit_terp_x_fruity']:+.3f}/{row['deficit_terp_pairs']:+.3f}/"
            f"{row['deficit_ms_paa_pairs']:+.3f}/{row['deficit_other_pairs']:+.3f}; rate rank on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; mean dist on/off "
            f"{row[v + '__on_mean_model']:.3f}/{row[v + '__off_mean_model']:.3f}; block {row[v + '__block_fruity_small_iso']:+.3f}; on excess block/other {row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f}; "
            f"vs ctrl {row[v + '__on_spearman']:.3f}; off vs TNT {row[v + '__off_spearman']:.3f}; vcf resid on/off {row.get(v + '__on_spearman_resid_vs_vcf_hyp', float('nan')):.2f}/{row.get(v + '__off_spearman_resid_vs_vcf_hyp', float('nan')):.2f}; {meta['seconds']:.0f}s", flush=True)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
        done.add(tag)
    if deficits:
        pd.concat(deficits).to_csv(res_dir / 'pair_deficits_long.csv', index=False, mode='a', header=not (res_dir / 'pair_deficits_long.csv').exists())
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E27_glomerulus_gain.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=(H.HERE / '../../../olfsysm').resolve(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, claw_thr=args.thr, gains=args.gains, oracle_glomeruli=ORACLE, top_glomeruli=TOP_GLOMS, random_sets={str(k): v for k, v in rng_sets.items()},
        synapse_weights=syn_info, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), fracs, res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
