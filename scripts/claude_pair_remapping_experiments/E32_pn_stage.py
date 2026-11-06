#!/usr/bin/env python3
"""E32, stage 1: the ORN->PN stage swept on the threshold base (can a uniform PN-level change
lift the weak odors?).

olfsysm's PN layer normalizes each glomerulus's input by the total ORN activity
(PN gain = pn_inhsc / (pn_inhadd + 0.25 inhA + 0.75 inhB), inside the PN saturation); at the
defaults (pn_inhsc 368.7, pn_inhadd 31.4) the divisive term dominates, so raising pn_inhadd
weakens the normalization. Variants on the bouton base with the 70 Hz per-claw threshold,
thresholds tuned at 5 % with APL on, acute APL off:
  default                      the threshold model (control);
  inhadd<x>                    pn_inhadd in E3's list (100, 314, 1000, 3141, 10000) with pn_inhsc
                               compensated as E3 did (gain at a typical inhibition kept);
  inhsc<f>x                    pn_inhsc at 0.5x / 2x / 4x its default at the default pn_inhadd
                               (the overall PN gain alone).
Scored as E27 (E13 statistics incl. natural-source alignment, E7 excess by block, per-odor
response fractions, E16 reference ranks, E21's APL-off sharing deficit by pair type) plus the
per-glomerulus PN rates of every variant (0-0.5 s mean, saved) and, in the row, the PN rate of
each weak odor's top glomerulus and its number of PNs above the knee.

Outputs: <shared>/results/E32_pn_stage/ and ./outputs/E32_pn_stage/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E32_pn_stage.py [--thr 70]
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
from E3_drive_dependent_gain import (BOUTON, TARGET, DEFAULT_INHADD, PN_INHADD_DEFAULT, PN_INHSC_DEFAULT, TYPICAL_INH,
    per_odor_extras, rate, tune_and_knockout)
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E26_claw_power import sharing_deficit, deficit_table, WEAK
from E27_glomerulus_gain import TOP_GLOMS

EXPERIMENT = 'E32_pn_stage'
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')


def glomerulus_pn_rates(orn, kws, params) -> pd.DataFrame:
    """Glomerulus x odor PN rates (0-0.5 s mean) of the tuned model."""
    from al_analysis.mb_model import fit_mb_model, get_time_index, get_sim_var
    _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=params['fixed_thr'], wAPLKC=params['wAPLKC_scale'], wKCAPL=params['wKCAPL_scale'],
        wAPLPN=params['wAPLPN_scale'], wPNAPL=params['wPNAPL_scale'], **kws, return_olfsysm_vars=True)
    mp, rv = pd0['mp'], pd0['rv']
    t = np.asarray(get_time_index(mp, rv)); win = (t >= 0.0) & (t <= 0.5)
    pn = get_sim_var(rv.pn, 'pn_sims')
    gloms = list(dict.fromkeys(wPNKC.columns.get_level_values('glomerulus')))
    return pd.DataFrame({H.odor_abbrev(c): np.asarray(pn[i])[:, win].mean(axis=1) for i, c in enumerate(orn.columns)}, index=gloms)


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
    panels = [('top-glomerulus PN rate of the weak odors (Hz; knee 70)', [(f'pn_top_{o}', o) for o in ['ms', 'paa', 'B-myr', 'euc', '1-8ol']], [70.0]),
        ('APL-off sharing deficit (0 = data)', [('deficit_terp_x_fruity', 'terpene / 1-8ol x fruity ester'), ('deficit_ms_paa_pairs', 'ms / paa pairs'), ('deficit_other_pairs', 'other pairs')], [0.0]),
        ('structure beyond the ORN input (data 0.44 on / 0.36 off)', [(p + 'on_spearman_resid_vs_vcf_hyp', 'APL on'), (p + 'off_spearman_resid_vs_vcf_hyp', 'APL off')], [0.44, 0.36]),
        ('rank agreement with control (on) / TNT (off); rate rank on / off', [(p + 'on_spearman', 'vs control'), (p + 'off_spearman', 'vs TNT'), ('spearman_rate_on_vs_data', 'rate rank, on'), ('spearman_rate_off_vs_data', 'rate rank, off')], []),
        ('mean pair distance (data 0.84 / 0.66) and block (data +0.055)', [(p + 'on_mean_model', 'mean distance, APL on'), (p + 'off_mean_model', 'mean distance, APL off'), (p + 'block_fruity_small_iso', 'block')], [0.84, 0.66, 0.055])]
    for ax, (title, cols, refs) in zip(list(axes.ravel())[1:], panels):
        for i, (col, lab) in enumerate(cols):
            if col in stats:
                ax.plot(x, stats[col].to_numpy(), ['-o', '--s', ':^', '-.d', '-v'][i % 5], color=['#2b6cb0', '#d9480f', '#2f855a', '#6b46c1', '#8a6d00'][i % 5], ms=5, lw=1.2, label=lab)
        for r in refs:
            ax.axhline(r, color='0.45', lw=1, ls=':')
        ax.set_title(title, fontsize=8.5); ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha='right', fontsize=8); ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    fig.suptitle('E32 stage 1: the PN normalization swept on the bouton base with the 70 Hz per-claw threshold (5 %)', fontsize=10.5)
    fig.tight_layout(); fig.savefig(res_dir / 'E32_pn_stage.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--thr', type=float, default=70.0)
    ap.add_argument('--inhadd', nargs='+', type=float, default=[v for v in DEFAULT_INHADD if v != PN_INHADD_DEFAULT])
    ap.add_argument('--inhsc-factors', nargs='+', type=float, default=[0.5, 2.0, 4.0])
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        make_figure(pd.DataFrame(rows), pd.read_csv(fracs_path, index_col=0), res_dir); return
    done = {r['tag'] for r in rows}
    orn = H.apl16_orn_deltas(); orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs(); refs = all_refs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned = json.loads((out_dir / 'tuned_params.json').read_text()) if (out_dir / 'tuned_params.json').exists() else {}
    variants = [('default', PN_INHADD_DEFAULT, PN_INHSC_DEFAULT)]
    for a in args.inhadd:
        variants.append((f'inhadd{a:g}', a, PN_INHSC_DEFAULT * (a + TYPICAL_INH) / (PN_INHADD_DEFAULT + TYPICAL_INH)))
    for f in args.inhsc_factors:
        variants.append((f'inhsc{f:g}x', PN_INHADD_DEFAULT, PN_INHSC_DEFAULT * f))
    pairs = {}; deficits = []
    for tag, inhadd, inhsc in variants:
        if tag in done:
            continue
        kws = dict(BOUTON, claw_thr=args.thr, pn_inhadd=float(inhadd), pn_inhsc=float(inhsc))
        t0 = time.time()
        try:
            on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag, with_pn_contrast=False)
            pn_rates = glomerulus_pn_rates(orn, dict(kws, make_plots=False, silent=True), params)
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, claw_thr=args.thr, pn_inhadd=inhadd, pn_inhsc=inhsc, error=repr(e)[:300])); pd.DataFrame(rows).to_csv(stats_path, index=False); continue
        pn_rates.rename_axis('glomerulus').to_csv(res_dir / f'glomerulus_pn_rates_{tag}.csv')
        tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
        tuned[tag].update(claw_thr=args.thr, pn_inhadd=inhadd, pn_inhsc=inhsc)
        (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1))
        driven = [g for g in pn_rates.index if g in set(orn.index)]
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag, claw_thr=args.thr, pn_inhadd=inhadd, pn_inhsc=inhsc,
            n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()), **extra,
            **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale') if k in params},
            pn_max_rate=float(pn_rates.loc[driven].to_numpy().max()), pn_median_peak=float(pn_rates.loc[driven].max(axis=1).median()))
        for o in WEAK:
            col = pn_rates[o].loc[driven]
            meta[f'pn_top_{o}'] = float(col.max()); meta[f'pn_n_above_knee_{o}'] = int((col > args.thr).sum())
        for o in ['2h', 'pa', 'eb']:
            col = pn_rates[o].loc[driven]; meta[f'pn_top_{o}'] = float(col.max()); meta[f'pn_n_above_knee_{o}'] = int((col > args.thr).sum())
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
        pk = ' '.join(f"{o}:{meta[f'pn_top_{o}']:.0f}/{meta[f'pn_n_above_knee_{o}']}" for o in ['ms', 'paa', 'B-myr', 'euc', '1-8ol', '2h', 'pa'])
        print(f"{tag} (inhadd {inhadd:g}, inhsc {inhsc:.0f}): thr {meta.get('fixed_thr_mean', float('nan')):.0f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f}; PN max {meta['pn_max_rate']:.0f}, "
            f"top PN / n>knee [{pk}]; weak odors off [{wk}]; deficit terp-x-fruity/ms-paa/other {row['deficit_terp_x_fruity']:+.3f}/{row['deficit_ms_paa_pairs']:+.3f}/{row['deficit_other_pairs']:+.3f}; "
            f"rate rank on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; mean dist on/off {row[v + '__on_mean_model']:.3f}/{row[v + '__off_mean_model']:.3f}; block {row[v + '__block_fruity_small_iso']:+.3f}; "
            f"vs ctrl {row[v + '__on_spearman']:.3f}; off vs TNT {row[v + '__off_spearman']:.3f}; vcf resid on/off {row.get(v + '__on_spearman_resid_vs_vcf_hyp', float('nan')):.2f}/{row.get(v + '__off_spearman_resid_vs_vcf_hyp', float('nan')):.2f}; {meta['seconds']:.0f}s", flush=True)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
        done.add(tag)
    if deficits:
        pd.concat(deficits).to_csv(res_dir / 'pair_deficits_long.csv', index=False, mode='a', header=not (res_dir / 'pair_deficits_long.csv').exists())
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E32_pn_stage.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=(H.HERE / '../../../olfsysm').resolve(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, claw_thr=args.thr, variants=[dict(tag=t, pn_inhadd=a, pn_inhsc=s) for t, a, s in variants], typical_inh=TYPICAL_INH, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), fracs, res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
