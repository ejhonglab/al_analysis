#!/usr/bin/env python3
"""E26: supralinear per-claw transfer (a single strongly driven claw can fire a KC).

After the per-claw threshold (E16) each claw contributes ref * (y / ref)^p of its rectified
input y (olfsysm p.kc.claw_pow / p.kc.claw_pow_ref), so for p > 1 one claw driven far above
the knee can carry a KC on its own while moderately driven claws still need coincidence. Asked
for by the weak-odor responder analysis (E23: methyl salicylate is recruited through one claw)
and the input-scaling run (E24: stronger input gives the terpenes / 1-octanol the right number
of responders but not the fly's overlap with the fruity-ester responders). Bouton base with
claw_thr 70, p in {1, 1.5, 2, 3} at ref 50 Hz, thresholds tuned at 5 % with APL on, acute APL
off; optionally other thresholds (--thr). Scored as E24: E13 statistics, E7 excess by block,
per-odor response fractions, E16 reference ranks, E21's APL-off sharing deficit decomposed by
pair type (among the weak odors, weak x other by the partner's odor group, the rest).

Outputs: <shared>/results/E26_claw_power/ and ./outputs/E26_claw_power/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E26_claw_power.py [--pows 1 1.5 2 3] [--thr 70] [--ref 50]
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

EXPERIMENT = 'E26_claw_power'
WEAK = ['ms', 'paa', 'euc', '-aPine', 'B-myr', '1-8ol']
TERP = ['euc', '-aPine', 'B-myr', '1-8ol']
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')


def tnt_excess(data_pairs: pd.DataFrame, a: str, b: str) -> float:
    return data_pairs['TNT_excess'].get((a, b), data_pairs['TNT_excess'].get((b, a), np.nan))


def deficit_table(spk_off: pd.DataFrame, data_pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-pair APL-off sharing deficit (model excess - TNT excess; positive = the model shares
    fewer KCs than the fly) with the pair's type: 'ms_paa' (involves ms or paa), 'terp'
    (involves euc / -aPine / B-myr / 1-8ol but not ms / paa) or 'other', and, for weak x
    other pairs, the unscaled partner's odor group.
    """
    ex = pair_excess(spk_off)
    rows = []
    for (a, b), v in ex.items():
        msp = a in ('ms', 'paa') or b in ('ms', 'paa')
        terp = a in TERP or b in TERP
        kind = 'ms_paa' if msp else ('terp' if terp else 'other')
        partner = None
        if (a in WEAK) ^ (b in WEAK):
            partner = H.ODOR_GROUPS.get(b if a in WEAK else a, b if a in WEAK else a)
        rows.append(dict(a=a, b=b, model=v, data=tnt_excess(data_pairs, a, b), deficit=v - tnt_excess(data_pairs, a, b),
            kind=kind, partner_group=partner, both_weak=int(a in WEAK and b in WEAK)))
    return pd.DataFrame(rows)


def sharing_deficit(spk_off: pd.DataFrame, data_pairs: pd.DataFrame) -> dict:
    d = deficit_table(spk_off, data_pairs)
    out = dict(deficit_all=float(d.deficit.mean()), deficit_ms_paa_pairs=float(d[d.kind == 'ms_paa'].deficit.mean()),
        deficit_terp_pairs=float(d[d.kind == 'terp'].deficit.mean()), deficit_other_pairs=float(d[d.kind == 'other'].deficit.mean()),
        deficit_among_weak=float(d[d.both_weak == 1].deficit.mean()))
    for g in ('fruity', 'small_ester', 'ald', 'alc'):
        sub = d[(d.partner_group == g) & (d.kind == 'terp')]
        out[f'deficit_terp_x_{g}'] = float(sub.deficit.mean()) if len(sub) else np.nan
        sub = d[(d.partner_group == g) & (d.kind == 'ms_paa')]
        out[f'deficit_ms_paa_x_{g}'] = float(sub.deficit.mean()) if len(sub) else np.nan
    return out


def make_figure(stats: pd.DataFrame, fracs: pd.DataFrame, res_dir, thr: float) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    stats = stats[stats.claw_thr == thr].sort_values('claw_pow')
    if not len(stats):
        return
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    ax = axes[0, 0]
    for o, c in zip(WEAK, ['#1f1f1f', '#8a6d00', '#2b6cb0', '#d9480f', '#2f855a', '#6b46c1']):
        ys = [fracs.loc[o, f"{t}_acute_off"] for t in stats.tag]
        ax.plot(stats.claw_pow, ys, '-o', color=c, ms=5, label=o); ax.axhline(fracs.loc[o, 'data_TNT'], color=c, lw=1, ls=':')
    ax.set_title('weak odors: APL-off response fraction (dotted = data)', fontsize=9); ax.set_xlabel('claw power p'); ax.legend(fontsize=7, frameon=False, ncol=2)
    panels = [('APL-off sharing deficit (model excess - TNT excess; 0 = data)', ['deficit_terp_x_fruity', 'deficit_terp_pairs', 'deficit_ms_paa_pairs', 'deficit_other_pairs'], 0.0,
            ['terpene / 1-8ol x fruity ester', 'all terpene / 1-8ol pairs', 'pairs with ms / paa', 'other pairs']),
        ('rate rank agreement with the data (ceiling 0.98 / 0.97)', ['spearman_rate_on_vs_data', 'spearman_rate_off_vs_data'], None, ['APL on', 'APL off']),
        ('rank agreement with control (on) / TNT (off)', [p + 'on_spearman', p + 'off_spearman'], None, ['APL on vs control', 'APL off vs TNT']),
        ('mean pair distance (data 0.84 on / 0.66 off)', [p + 'on_mean_model', p + 'off_mean_model'], 0.840, ['APL on', 'APL off']),
        ('block (data +0.055) and APL-on block-pair excess (data -0.08)', [p + 'block_fruity_small_iso', 'on_excess_block'], 0.055, ['fruity x small-ester block', 'APL-on block-pair excess'])]
    for ax, (title, cols, ref, labels) in zip(list(axes.ravel())[1:], panels):
        for i, col in enumerate(cols):
            ax.plot(stats.claw_pow, stats[col], ['-o', '--s', ':^', '-.d'][i], color=['#2b6cb0', '#2b6cb0', '#d9480f', '#2f855a'][i], ms=5, lw=1.3, label=labels[i])
        if ref is not None:
            ax.axhline(ref, color='0.45', lw=1, ls=':')
        ax.set_title(title, fontsize=9); ax.set_xlabel('claw power p'); ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    fig.suptitle(f'E26: supralinear per-claw transfer ref x (y / ref)^p on the bouton base, claw_thr {thr:g}, 5 %', fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / f'E26_claw_power_thr{thr:g}.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--pows', nargs='+', type=float, default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument('--thr', type=float, default=70.0)
    ap.add_argument('--ref', type=float, default=50.0)
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        make_figure(pd.DataFrame(rows), pd.read_csv(fracs_path, index_col=0), res_dir, args.thr); return
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
    pairs = {}; deficits = []
    for pw in args.pows:
        tag = f'thr{args.thr:g}_pow{pw:g}'
        if tag in done:
            continue
        kws = dict(BOUTON, claw_thr=args.thr)
        if pw != 1.0:
            kws.update(claw_pow=pw, claw_pow_ref=args.ref)
        t0 = time.time()
        try:
            on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag, with_pn_contrast=False)
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, claw_thr=args.thr, claw_pow=pw, claw_pow_ref=args.ref, error=repr(e)[:300])); pd.DataFrame(rows).to_csv(stats_path, index=False); continue
        tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
        tuned[tag].update(claw_thr=args.thr, claw_pow=pw, claw_pow_ref=args.ref)
        (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1))
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag, claw_thr=args.thr, claw_pow=pw, claw_pow_ref=args.ref,
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
            f"vs ctrl {row[v + '__on_spearman']:.3f}; off vs TNT {row[v + '__off_spearman']:.3f}; {meta['seconds']:.0f}s", flush=True)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
        done.add(tag)
    if deficits:
        pd.concat(deficits).to_csv(res_dir / 'pair_deficits_long.csv', index=False, mode='a', header=not (res_dir / 'pair_deficits_long.csv').exists())
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E26_claw_power.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=(H.HERE / '../../../olfsysm').resolve(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, claw_thr=args.thr, pows=args.pows, claw_pow_ref=args.ref, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), fracs, res_dir, args.thr)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
