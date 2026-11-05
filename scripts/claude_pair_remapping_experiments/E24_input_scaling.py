#!/usr/bin/env python3
"""E24: input scaling for the terpenes and 1-octanol (sensitivity run).

The weak-odor responder analysis (E23) found that eucalyptol, alpha-pinene, beta-myrcene and
1-octanol are recruited in the fly through ordinary broad KC populations whose co-tuning
follows the odors' ORN profiles, so the model has the right glomeruli for them but too
little drive. Here their ORN responses (all glomeruli) are scaled by a common factor f in
{1, 1.5, 2, 3} on the bouton base with the 70 Hz per-claw threshold (E16), thresholds tuned
at 5 % with APL on, acute APL off; methyl salicylate and phenylacetaldehyde are unchanged.
Scored as E22 (E13 statistics, E7 excess by block, per-odor response fractions, E21's
APL-off sharing deficit split into the pairs that involve a scaled odor and the rest).

Outputs: <shared>/results/E24_input_scaling/ and ./outputs/E24_input_scaling/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E24_input_scaling.py [--factors 1 1.5 2 3]
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
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E21_apl_off_sharing import pair_excess

EXPERIMENT = 'E24_input_scaling'
SCALED = ['euc', '-aPine', 'B-myr', '1-8ol']
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')


def sharing_deficit(spk_off: pd.DataFrame, data_pairs: pd.DataFrame) -> dict:
    ex = pair_excess(spk_off)
    d = []
    for (a, b), v in ex.items():
        t = data_pairs['TNT_excess'].get((a, b), data_pairs['TNT_excess'].get((b, a), np.nan))
        d.append(dict(deficit=v - t, scaled=int(a in SCALED or b in SCALED), msp=int(a in ('ms', 'paa') or b in ('ms', 'paa'))))
    d = pd.DataFrame(d)
    return dict(deficit_all=float(d.deficit.mean()), deficit_scaled_pairs=float(d[d.scaled == 1].deficit.mean()),
        deficit_ms_paa_pairs=float(d[(d.msp == 1) & (d.scaled == 0)].deficit.mean()),
        deficit_other_pairs=float(d[(d.scaled == 0) & (d.msp == 0)].deficit.mean()))


def make_figure(stats: pd.DataFrame, fracs: pd.DataFrame, res_dir) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    stats = stats.sort_values('factor')
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    ax = axes[0, 0]
    for o, c in zip(SCALED, ['#2b6cb0', '#d9480f', '#2f855a', '#6b46c1']):
        ys = [fracs.loc[o, f"f{f:g}_acute_off"] for f in stats.factor]
        ax.plot(stats.factor, ys, '-o', color=c, ms=5, label=o); ax.axhline(fracs.loc[o, 'data_TNT'], color=c, lw=1, ls=':')
    ax.set_title('scaled odors: APL-off response fraction (dotted = data)', fontsize=9); ax.set_xlabel('input factor f'); ax.legend(fontsize=7, frameon=False)
    panels = [('APL-off sharing deficit: scaled-odor pairs / ms-paa pairs / other', ['deficit_scaled_pairs', 'deficit_ms_paa_pairs', 'deficit_other_pairs'], 0.0),
        ('rate rank agreement on / off (ceiling 0.98 / 0.97)', ['spearman_rate_on_vs_data', 'spearman_rate_off_vs_data'], None),
        ('rank agreement with control (on) / TNT (off)', [p + 'on_spearman', p + 'off_spearman'], None),
        ('mean pair distance on / off (data 0.84 / 0.66)', [p + 'on_mean_model', p + 'off_mean_model'], 0.840),
        ('fruity x small-ester block (data +0.055)', [p + 'block_fruity_small_iso'], 0.055)]
    for ax, (title, cols, ref) in zip(list(axes.ravel())[1:], panels):
        for i, col in enumerate(cols):
            ax.plot(stats.factor, stats[col], ['-o', '--s', ':^'][i], color='#2b6cb0', ms=5, lw=1.3, label=col.replace(p, ''))
        if ref is not None:
            ax.axhline(ref, color='0.45', lw=1, ls=':')
        ax.set_title(title, fontsize=9); ax.set_xlabel('input factor f'); ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    fig.suptitle('E24: ORN input of euc / -aPine / B-myr / 1-8ol scaled by f (bouton base, claw_thr 70, 5 %)', fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / 'E24_input_scaling.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--factors', nargs='+', type=float, default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument('--thr', type=float, default=70.0)
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        make_figure(pd.DataFrame(rows), pd.read_csv(fracs_path, index_col=0), res_dir); return
    done = {r['tag'] for r in rows}
    orn0 = H.apl16_orn_deltas()
    abbr = {c: H.odor_abbrev(c) for c in orn0.columns}
    orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    refs = all_refs()
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned = json.loads((out_dir / 'tuned_params.json').read_text()) if (out_dir / 'tuned_params.json').exists() else {}
    pairs = {}
    for f in args.factors:
        tag = f'f{f:g}'
        if tag in done:
            continue
        orn = orn0.copy()
        for c in orn.columns:
            if abbr[c] in SCALED:
                orn[c] = orn[c] * f
        kws = dict(BOUTON, claw_thr=args.thr)
        t0 = time.time()
        try:
            on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag, with_pn_contrast=False)
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, factor=f, claw_thr=args.thr, error=repr(e)[:300])); pd.DataFrame(rows).to_csv(stats_path, index=False); continue
        tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
        (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1))
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag, factor=f, claw_thr=args.thr, scaled_odors=','.join(SCALED),
            n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()), **extra,
            **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale') if k in params})
        meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
        # scored against the UNSCALED ORN input (the data's input is what it is; only the model's drive changed)
        row, on, off = score(meta, on_out, off_out, wPNKC, orn0, orn_pairs, drive, data, ref)
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index); fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        v = H.PRIMARY_MODEL_RDM
        sc = ' '.join(f"{o}:{fracs.loc[o, f'{tag}_acute_off']:.3f}" for o in SCALED)
        print(f"{tag}: thr {meta.get('fixed_thr_mean', float('nan')):.0f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f}; scaled odors off [{sc}] (data "
            f"{' '.join(f'{o}:{fracs.loc[o, 'data_TNT']:.3f}' for o in SCALED)}); deficit scaled/ms-paa/other {row['deficit_scaled_pairs']:+.3f}/{row['deficit_ms_paa_pairs']:+.3f}/"
            f"{row['deficit_other_pairs']:+.3f}; rate rank on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; mean dist on/off "
            f"{row[v + '__on_mean_model']:.3f}/{row[v + '__off_mean_model']:.3f}; block {row[v + '__block_fruity_small_iso']:+.3f}; vs ctrl {row[v + '__on_spearman']:.3f}; "
            f"off vs TNT {row[v + '__off_spearman']:.3f}; {meta['seconds']:.0f}s", flush=True)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
        done.add(tag)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E24_input_scaling.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=(H.HERE / '../../../olfsysm').resolve(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, claw_thr=args.thr, factors=args.factors, scaled_odors=SCALED, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), fracs, res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
