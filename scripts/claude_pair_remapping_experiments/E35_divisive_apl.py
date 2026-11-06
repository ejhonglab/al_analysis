#!/usr/bin/env python3
"""E35: divisive APL inhibition (shunting at the claws / KCs, gain reduction at the boutons).

In the KC data blocking APL output multiplies every odor's response fraction by about the
same factor (1.65-2.90 across the 16 odors, median 2.2, the spread at the bootstrap noise of
the recordings), whereas every model so far removes the sparse odors' responders
preferentially (notes/gain_rate_cost.md in the shared directory). The model's inhibition is
subtractive at both sites; this sweep replaces each subtraction with a division (olfsysm
`p.kc.apl_inh_divisive`: claw / KC drive = drive / (1 + wAPLKC x APL); `p.pn.bouton_inh_divisive`:
bouton rate = rate / (1 + wAPLPN x APL)) on the bouton base with connectome APL weights, tuned
to 5 % with APL on, acute APL off: forms {sub (reference), kcdiv, boutondiv, bothdiv} x
per-claw threshold {none, 70 Hz}. Scored as E24 / E26 (E13 statistics, E7 excess by block,
per-odor response fractions, E16 reference ranks) plus the suppression uniformity: the sd
across odors of log(APL-off fraction / APL-on fraction) (data 0.15, bootstrap noise 0.13), its
rank correlation with the APL-on fraction (data about 0) and the ratio by odor group.

Outputs: <shared>/results/E35_divisive_apl/ and ./outputs/E35_divisive_apl/. Each run writes
stats_variants_<thrtag>.csv / per_odor_response_fractions_<thrtag>.csv so that runs with
different --thr can go in parallel; --figure-only merges them into stats_variants.csv and
per_odor_response_fractions.csv and draws the figure.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E35_divisive_apl.py --thr none [--forms sub kcdiv boutondiv bothdiv]
       OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E35_divisive_apl.py --thr 70
       ./venv/bin/python E35_divisive_apl.py --figure-only
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sstats

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON, TARGET, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs

EXPERIMENT = 'E35_divisive_apl'
FORMS = {'sub': (False, False), 'kcdiv': (True, False), 'boutondiv': (False, True), 'bothdiv': (True, True)}
ESTERS = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep']
WEAK = ['ms', 'paa', 'B-myr', 'euc', '1-8ol']
SPARSE = ['ms', 'paa', 'B-myr']
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
DATA_UNIFORMITY_SD = 0.148
DATA_UNIFORMITY_NOISE = 0.128


def group(odor: str) -> str:
    return 'esters' if odor in ESTERS else ('weak' if odor in WEAK else 'other')


def suppression_stats(r_on: pd.Series, r_off: pd.Series, data_rates: dict, floor: float = 0.5 / 1732) -> dict:
    """Per-odor APL-off / APL-on response-fraction ratio: its spread across odors (sd of the
    log ratio), its rank correlation with the APL-on fraction, its correlation with the
    data's ratio, the ratio by odor group and for the three sparse odors. An odor without
    APL-on responders counts as having half a KC (`floor`), so it is not dropped.
    """
    r_on = r_on.reindex(H.APL16_ODORS); r_off = r_off.reindex(H.APL16_ODORS)
    ratio = r_off / r_on.clip(lower=floor)
    d_ratio = (data_rates['TNT'] / data_rates['TNT3c']).reindex(H.APL16_ODORS)
    out = dict(suppression_sd_log=float(np.log(ratio).std(ddof=1)), suppression_median=float(ratio.median()),
        suppression_max=float(ratio.max()), n_odors_no_on_responders=int((r_on == 0).sum()),
        spearman_suppression_vs_on_frac=float(sstats.spearmanr(ratio, r_on)[0]),
        spearman_suppression_vs_data=float(sstats.spearmanr(ratio, d_ratio)[0]))
    for g in ('esters', 'other', 'weak'):
        sel = [o for o in H.APL16_ODORS if group(o) == g]
        out[f'suppression_{g}'] = float(r_off[sel].mean() / r_on[sel].mean())
    for o in SPARSE:
        out[f'suppression_{o}'] = float(ratio[o])
        out[f'on_frac_{o}'] = float(r_on[o])
    return out


def thr_tag(thr) -> str:
    return 'nothr' if thr is None else f'thr{thr:g}'


def make_figure(stats: pd.DataFrame, fracs: pd.DataFrame, data_rates: dict, res_dir) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    stats = stats.copy()
    stats['thr_label'] = stats.claw_thr.map(lambda t: 'no threshold' if pd.isna(t) else f'{t:g} Hz')
    order = [f'{f}_{thr_tag(t)}' for t in (None, 70.0) for f in FORMS]
    stats = stats.set_index('tag').reindex([t for t in order if t in set(stats.tag)]).reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = {'sub': '#6b7280', 'kcdiv': '#1d4ed8', 'boutondiv': '#c2410c', 'bothdiv': '#15803d'}
    d_on = data_rates['TNT3c'].reindex(H.APL16_ODORS); d_ratio = (data_rates['TNT'] / data_rates['TNT3c']).reindex(H.APL16_ODORS)
    ax = axes[0, 0]
    ax.scatter(d_on, d_ratio, s=30, color='black', label='KC data', zorder=5)
    for _, r in stats.iterrows():
        on = fracs[f"{r.tag}_on"].reindex(H.APL16_ODORS); off = fracs[f"{r.tag}_acute_off"].reindex(H.APL16_ODORS)
        ratio = off / on.where(on > 0)
        ax.scatter(on, ratio, s=18, color=colors[r.form], marker='o' if r.thr_label == 'no threshold' else '^', alpha=0.8,
            label=f'{r.form}, {r.thr_label}')
    ax.set_yscale('log'); ax.set_xlabel('APL-on response fraction (model) / control flies (data)'); ax.set_ylabel('APL off / on response fraction')
    ax.set_title('per-odor APL suppression vs the odor\'s APL-on fraction', fontsize=9); ax.legend(fontsize=6.5, frameon=False, ncol=2)
    x = np.arange(len(stats)); labels = [f'{f}\n{t}' for f, t in zip(stats.form, stats.thr_label)]
    panels = [('suppression uniformity: sd across odors of log(off / on) (data 0.15, noise 0.13)', ['suppression_sd_log'], DATA_UNIFORMITY_SD, ['model']),
        ('rate rank agreement with the data (ceiling 0.98 / 0.97)', ['spearman_rate_on_vs_data', 'spearman_rate_off_vs_data'], None, ['APL on', 'APL off']),
        ('rank agreement with control (on) / TNT (off)', [p + 'on_spearman', p + 'off_spearman'], None, ['APL on vs control', 'APL off vs TNT']),
        ('mean pair distance (data 0.84 on / 0.66 off)', [p + 'on_mean_model', p + 'off_mean_model'], 0.840, ['APL on', 'APL off']),
        ('block (data +0.055) and APL-on block-pair excess (data -0.08)', [p + 'block_fruity_small_iso', 'on_excess_block'], 0.055, ['fruity x small-ester block', 'APL-on block-pair excess'])]
    for ax, (title, cols, ref, lab) in zip(list(axes.ravel())[1:], panels):
        w = 0.8 / len(cols)
        for i, col in enumerate(cols):
            ax.bar(x + (i - (len(cols) - 1) / 2) * w, stats[col], w, color=[colors[f] for f in stats.form], alpha=1.0 if i == 0 else 0.45, label=lab[i])
        if ref is not None:
            ax.axhline(ref, color='0.3', lw=1, ls=':')
        if cols[0] == 'suppression_sd_log':
            ax.axhline(DATA_UNIFORMITY_NOISE, color='0.6', lw=1, ls='--')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7); ax.set_title(title, fontsize=9)
        if len(cols) > 1:
            ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    fig.suptitle('E35: divisive APL inhibition on the bouton base (5 % with APL on, acute APL off); circles / first row: no per-claw threshold, triangles: 70 Hz', fontsize=10)
    fig.tight_layout(); fig.savefig(res_dir / 'E35_divisive_apl.png', dpi=130); plt.close(fig)


def merge(res_dir, data_rates: dict) -> tuple:
    """Merges the per-threshold tables and recomputes the suppression statistics from the
    per-odor fractions (so every row uses the current definition)."""
    stats = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(res_dir / 'stats_variants_*.csv')))], ignore_index=True)
    fr = [pd.read_csv(f, index_col=0) for f in sorted(glob.glob(str(res_dir / 'per_odor_response_fractions_*.csv')))]
    fracs = fr[0]
    for f in fr[1:]:
        fracs = fracs.join(f[[c for c in f.columns if c not in fracs.columns]])
    for i, r in stats.iterrows():
        if f"{r.tag}_on" in fracs:
            for kk, vv in suppression_stats(fracs[f"{r.tag}_on"], fracs[f"{r.tag}_acute_off"], data_rates).items():
                stats.loc[i, kk] = vv
    stats.to_csv(res_dir / 'stats_variants.csv', index=False)
    fracs.to_csv(res_dir / 'per_odor_response_fractions.csv')
    return stats, fracs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--thr', default='70', help="per-claw threshold in Hz, or 'none'")
    ap.add_argument('--forms', nargs='+', default=list(FORMS), choices=list(FORMS))
    ap.add_argument('--apl-drive', default='claws', choices=['claws', 'spikes'],
        help="what drives APL: 'claws' (the bouton base: summed claw input plus bouton rates, "
        "pn_claw_to_apl=True) or 'spikes' (KC spikes plus bouton rates, pn_claw_to_apl=False)")
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    thr = None if args.thr == 'none' else float(args.thr)
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    if args.figure_only:
        stats, fracs = merge(res_dir, data_rates); make_figure(stats, fracs, data_rates, res_dir); print('figure ->', res_dir); return
    tt = thr_tag(thr) + ('_spk' if args.apl_drive == 'spikes' else '')
    stats_path = res_dir / f'stats_variants_{tt}.csv'; fracs_path = res_dir / f'per_odor_response_fractions_{tt}.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}
    orn = H.apl16_orn_deltas(); orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    refs = all_refs()
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned_path = out_dir / f'tuned_params_{tt}.json'
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    pairs = {}
    for form in args.forms:
        kc_div, bouton_div = FORMS[form]
        tag = f'{form}_{tt}'
        if tag in done:
            continue
        kws = dict(BOUTON)
        if args.apl_drive == 'spikes':
            kws['pn_claw_to_apl'] = False
        if thr is not None:
            kws['claw_thr'] = thr
        if kc_div:
            kws['apl_inh_divisive'] = True
        if bouton_div:
            kws['bouton_inh_divisive'] = True
        if kc_div or bouton_div:
            # the divisive forms tune to APL scales 5-15x smaller than the subtractive ones,
            # and the scale search needs more (mostly unsimulated) iterations to get there
            kws['max_iters'] = 1000
        t0 = time.time()
        try:
            on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag, with_pn_contrast=False)
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, form=form, claw_thr=thr, apl_drive=args.apl_drive, apl_inh_divisive=kc_div, bouton_inh_divisive=bouton_div, error=repr(e)[:300]))
            pd.DataFrame(rows).to_csv(stats_path, index=False); continue
        tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
        tuned[tag].update(claw_thr=thr, form=form, apl_drive=args.apl_drive, apl_inh_divisive=kc_div, bouton_inh_divisive=bouton_div)
        tuned_path.write_text(json.dumps(tuned, indent=1))
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag, form=form, claw_thr=thr, apl_drive=args.apl_drive, apl_inh_divisive=kc_div,
            bouton_inh_divisive=bouton_div, n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]),
            seconds=time.time() - t0, mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()), **extra,
            **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale') if k in params})
        meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs))
        r_on = H.response_fractions(on_out[0]); r_off = H.response_fractions(off_out[0])
        row.update(suppression_stats(r_on, r_off, data_rates))
        rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = r_on.reindex(fracs.index); fracs[f'{tag}_acute_off'] = r_off.reindex(fracs.index)
        fracs.to_csv(fracs_path)
        v = H.PRIMARY_MODEL_RDM
        print(f"{tag}: thr {meta.get('fixed_thr_mean', float('nan')):.0f}, APL->KC {meta.get('wAPLKC_scale', float('nan')):.3g}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f} "
            f"(x{meta['rate_ratio_off_over_on']:.2f}); suppression sd(log) {row['suppression_sd_log']:.2f} (data 0.15), rho vs on-frac {row['spearman_suppression_vs_on_frac']:+.2f}, "
            f"esters/other/weak x{row['suppression_esters']:.1f}/{row['suppression_other']:.1f}/{row['suppression_weak']:.1f}, ms/paa/B-myr on {row['on_frac_ms']:.3f}/{row['on_frac_paa']:.3f}/{row['on_frac_B-myr']:.3f}; "
            f"rate rank on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; mean dist on/off {row[v + '__on_mean_model']:.3f}/{row[v + '__off_mean_model']:.3f}; "
            f"block {row[v + '__block_fruity_small_iso']:+.3f}; on excess block/other {row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f}; vs ctrl {row[v + '__on_spearman']:.3f}; "
            f"off vs TNT {row[v + '__off_spearman']:.3f}; {meta['seconds']:.0f}s", flush=True)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
        done.add(tag)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E35_divisive_apl.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), base=BOUTON, claw_thr=thr, apl_drive=args.apl_drive, forms={f: FORMS[f] for f in args.forms},
        approx_conc_note=H.APPROX_CONC_NOTE)
    try:
        from al_analysis.mb_model import model_provenance
        prov.update(model_provenance(orn_deltas=orn))
    except Exception as e:  # keep the commit hashes at least
        prov.update(al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
            olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=(H.HERE / '../../../olfsysm').resolve(), capture_output=True, text=True).stdout.strip(),
            model_provenance_error=repr(e)[:200])
    (res_dir / f'provenance_{tt}.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
