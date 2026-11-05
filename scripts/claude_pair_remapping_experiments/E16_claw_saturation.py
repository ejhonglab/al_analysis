#!/usr/bin/env python3
"""E16: per-claw saturation as the linear-summation form of the claw-coincidence rule.

`n_claws_active_to_spike` (E5, E13) is one of the two manipulations that moved anything,
but it replaces somatic integration by counting thresholded claw events. The analysis in
add_to_reports/coincidence_rule_qna.md says its effective action is to bound each claw's
contribution. Here each claw's drive x enters the sum across claws as
`claw_sat * tanh(x / claw_sat)` (or `min(x, claw_sat)`, hard) (olfsysm `mp.kc.claw_sat`,
`claw_sat_hard`), with the somatic threshold tuned as usual (APL on, 5 %), then acute APL
knockout. Bases: bouton (E3) and claw-dynamics (E1/E5). Caps are swept around the
coincidence models' tuned per-claw thresholds (bouton: 106 / 68 for 2 / 3 claws; claw
dynamics: 65 / 38; somatic thresholds 254 / 170). Scored as E13 (E4 statistics, per-odor
readouts, E7 excess by block) plus the Spearman of each capped model's pair distances
with the matching coincidence-2 / -3 models' (E13; E5 hal probe) and with the uncapped
base. Optionally (--breadth) the best caps are combined with E12's breadth-weighted APL
(k = 3; the capped model's tuned thresholds, KC<->APL scales re-matched to its count).

Outputs: <shared>/results/E16_claw_saturation/ and ./outputs/E16_claw_saturation/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E16_claw_saturation.py
    [--bases bouton claw-dynamics] [--caps 30 50 70 100 150 250] [--forms tanh hard]
    [--breadth CAP ...]   (bouton base only; run after the sweep)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H
from E3_drive_dependent_gain import BASES, BOUTON, TARGET, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess
from E3_drive_dependent_gain import match_kc_apl_scale

EXPERIMENT = 'E16_claw_saturation'
E3_DIR = H.HERE / 'outputs' / 'E3_drive_dependent_gain'
BASE_KWS = {'bouton': dict(BOUTON), 'claw-dynamics': dict(BASES['claw-dynamics'])}
# (results dir, tag of the coincidence-N model, tag of the uncapped base) per base
COINC_REFS = {
    'bouton': ('E13_bouton_coincidence', 'bouton_coincidence-{n}', 'bouton_base'),
    'claw-dynamics': ('E5_kc_heterogeneity/hal_probe', 'coincidence-{n}_sp0.05', None),
}
BREADTH_K = 3.0


def reference_pairs(base: str) -> Dict[str, pd.Series]:
    """Pair distances (odor_a, odor_b -> distance) of the coincidence-2 / -3 models (and
    the uncapped base, if saved) of `base`, APL on and acute off."""
    sub, coinc, base_tag = COINC_REFS[base]
    long = pd.read_csv(H.shared_dir() / 'results' / sub / 'model_pair_distances_long.csv')
    out = {}
    for name, tag in [('coinc2', coinc.format(n=2)), ('coinc3', coinc.format(n=3)),
            ('base', base_tag)]:
        if tag is None:
            continue
        for cond in ('on', 'acute_off'):
            sel = long[(long.tag == tag) & (long.condition == cond)]
            if len(sel):
                out[f'{name}_{cond}'] = sel.set_index(['odor_a', 'odor_b'])['distance']
    return out


def spearman_with_refs(on_pairs: pd.Series, off_pairs: pd.Series, refs: Dict[str, pd.Series]
    ) -> Dict[str, float]:
    out = {}
    for key, ref in refs.items():
        model = on_pairs if key.endswith('_on') else off_pairs
        m = model.reindex(ref.index)
        if m.isna().any():
            # pairs may be stored in the other order
            m = model.reindex(pd.MultiIndex.from_arrays(
                [ref.index.get_level_values(1), ref.index.get_level_values(0)])
            )
            m.index = ref.index
        # (a reference model may have NaN distances, e.g. an odor with no responders)
        ok = ~(m.isna() | ref.isna())
        out[f'spearman_vs_{key}'] = float(stats.spearmanr(m[ok], ref[ok])[0])
    return out


def row_summary(tag: str, meta: dict, row: dict) -> str:
    p = H.PRIMARY_MODEL_RDM
    return (f"{tag}: thr {meta.get('fixed_thr_mean', float('nan')):.0f}, rates on "
        f"{meta['rate_on']:.4f} off {meta['rate_off']:.4f}; mean dist on "
        f"{row[p + '__on_mean_model']:.3f}; block {row[p + '__block_fruity_small_iso']:+.3f}; "
        f"excess on {row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f} off "
        f"{row['off_excess_block']:+.3f}/{row['off_excess_other_between']:+.3f}; vs ctrl "
        f"{row[p + '__on_spearman']:.3f} dev {row[p + '__on_spearman_dev']:+.2f}; off vs TNT "
        f"{row[p + '__off_spearman']:.3f}; vs coinc2/3 on {row.get('spearman_vs_coinc2_on', float('nan')):.2f}/"
        f"{row.get('spearman_vs_coinc3_on', float('nan')):.2f}; {meta['seconds']:.0f}s")


def make_figure(stats: pd.DataFrame, res_dir: Path) -> None:
    """Statistics vs cap: hue = base, line style = transfer (tanh / hard / offset + tanh),
    dotted horizontal = the uncapped base, grey dotted = data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    stats = stats[stats.get('breadth_k', pd.Series(index=stats.index, dtype=float)).isna()]
    panels = [
        ('Spearman with the coincidence-3 model (APL on)', 'spearman_vs_coinc3_on', None),
        ('mean pair distance, APL on (data ctrl 0.86)', p + 'on_mean_model', 0.855),
        ('excess sharing, block pairs, APL on (data -0.08)', 'on_excess_block', -0.080),
        ('Spearman with control (APL on)', p + 'on_spearman', None),
        ('Spearman with TNT (APL off)', p + 'off_spearman', None),
        ('fruity x small-ester block (data +0.055)', p + 'block_fruity_small_iso', 0.055),
    ]
    colors = {'bouton': '#2b6cb0', 'claw-dynamics': '#d9480f'}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6))
    for ax, (title, col, ref) in zip(axes.ravel(), panels):
        for base, sub in stats.groupby('base'):
            c = colors.get(base, '0.3')
            nocap = sub[sub.claw_sat.isna()]
            if len(nocap) and col in nocap:
                ax.axhline(float(nocap[col].iloc[0]), color=c, lw=1, ls=':')
            capped = sub[sub.claw_sat.notna()]
            thr_col = capped['claw_thr'] if 'claw_thr' in capped else pd.Series(index=capped.index, dtype=float)
            groups = capped.assign(_thr=thr_col.fillna(-1)).groupby(['form', '_thr'])
            markers = ['o', 's', '^', 'D', 'v']
            for (form, thr), ss in groups:
                ss = ss.sort_values('claw_sat')
                x = ss['claw_sat'].replace(0, 10).to_numpy()
                if thr < 0:
                    lab, ls, mfc, mk = f'{base}, {form}', ('-' if form == 'tanh' else '--'), (c if form == 'tanh' else 'white'), 'o'
                else:
                    thrs = sorted(set(thr_col.dropna()))
                    mk = markers[(thrs.index(thr) + 1) % len(markers)]
                    lab, ls, mfc = f'{base}, offset {thr:g} + tanh', '-', c
                ax.plot(x, ss[col].to_numpy(), ls, color=c, marker=mk, ms=5, mfc=mfc, lw=1.4, label=lab)
        if ref is not None:
            ax.axhline(ref, color='0.45', lw=1, ls=(0, (1, 2)))
        ax.set_xscale('log')
        ax.set_xticks([10, 30, 50, 70, 100, 150, 250])
        ax.set_xticklabels(['off.', '30', '50', '70', '100', '150', '250'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel('per-claw cap (claw_sat; off. = offset only)')
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, color='0.9', lw=0.6)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle('E16: per-claw saturation (and offset) of the claw>soma transfer; dotted colored = uncapped base', fontsize=11)
    fig.tight_layout()
    fig.savefig(res_dir / 'E16_claw_saturation.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bases', nargs='+', choices=list(BASE_KWS), default=['bouton', 'claw-dynamics'])
    ap.add_argument('--caps', nargs='+', type=float, default=[30.0, 50.0, 70.0, 100.0, 150.0, 250.0])
    ap.add_argument('--forms', nargs='+', choices=['tanh', 'hard'], default=['tanh', 'hard'])
    ap.add_argument('--thr', nargs='+', type=float, default=None,
        help='per-claw rectified offsets (olfsysm claw_thr) to cross with --caps (tanh); '
        'tags <base>_thr<t>_tanh<cap>; a cap of 0 means offset only')
    ap.add_argument('--breadth', nargs='*', type=float, default=None,
        help='caps (tanh, bouton base) to combine with E12 breadth-weighted APL (k = 3)')
    ap.add_argument('--no-sweep', action='store_true', help='only the --breadth combinations')
    ap.add_argument('--figure-only', action='store_true', help='only redraw the figure from the saved stats')
    ap.add_argument('--rescore-refs', action='store_true',
        help='recompute the spearman_vs_* columns of every saved row from the saved pair distances')
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
    stats_path = res_dir / 'stats_variants.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}
    if args.figure_only:
        make_figure(pd.read_csv(stats_path), res_dir)
        return
    if args.rescore_refs:
        long = pd.read_csv(res_dir / 'model_pair_distances_long.csv')
        for r in rows:
            base = r['base']
            refs = reference_pairs(base)
            sel = {c: long[(long.tag == f'{base}_nocap') & (long.condition == c)] for c in ('on', 'acute_off')}
            for c, d in sel.items():
                if len(d):
                    refs[f'base_{c}'] = d.set_index(['odor_a', 'odor_b'])['distance']
            mine = {c: long[(long.tag == r['tag']) & (long.condition == c)].set_index(
                ['odor_a', 'odor_b'])['distance'] for c in ('on', 'acute_off')}
            if len(mine['on']) and len(mine['acute_off']):
                r.update(spearman_with_refs(mine['on'], mine['acute_off'], refs))
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        print('rescored', len(rows), 'rows')
        return
    params_path = out_dir / 'tuned_params.json'
    tuned = json.loads(params_path.read_text()) if params_path.exists() else {}
    pairs = {}

    def finish(tag, meta, row, on, off):
        rows.append(row)
        pairs[(tag, 'on')] = on
        pairs[(tag, 'acute_off')] = off
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        print(row_summary(tag, meta, row), flush=True)

    jobs = []
    if not args.no_sweep:
        for base in args.bases:
            jobs.append((base, None, 'tanh', None))
            for form in args.forms:
                for cap in args.caps:
                    jobs.append((base, cap, form, None))
    for base in args.bases:
        for thr in (args.thr or []):
            for cap in args.caps:
                jobs.append((base, cap, 'tanh', thr))
    for base, cap, form, thr in jobs:
        if thr is not None:
            tag = f'{base}_thr{thr:g}_tanh{cap:g}' if cap > 0 else f'{base}_thr{thr:g}'
        else:
            tag = f'{base}_nocap' if cap is None else f'{base}_{form}{cap:g}'
        if tag in done:
            continue
        refs = reference_pairs(base)
        kws = dict(BASE_KWS[base])
        if cap is not None and cap > 0:
            kws.update(claw_sat=cap, claw_sat_hard=(form == 'hard'))
        if thr is not None:
            kws.update(claw_thr=thr)
        t0 = time.time()
        try:
            on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag,
                with_pn_contrast=False
            )
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, base=base, claw_sat=cap, form=form,
                claw_thr=thr, error=repr(e)[:300]))
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            continue
        # (fixed_thr may be per-KC and wAPLKC / wKCAPL are per-claw Series: keep means)
        tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items()
            if k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale',
            'wAPLKC', 'wKCAPL')}
        params_path.write_text(json.dumps(tuned, indent=1))
        meta = dict(variant=tag, base=base, seed=None, target_rate=TARGET, tag=tag,
            claw_sat=cap, form=(form if cap is not None else None), claw_thr=thr,
            n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1],
            rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
            **extra, **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale',
                'wAPLPN_scale', 'wPNAPL_scale') if k in params},
        )
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        if f'{base}_nocap' in done or cap is None:
            # uncapped base of this run (E5's probe did not save the claw-dynamics base)
            if cap is None:
                refs.update({'base_on': on_p, 'base_acute_off': off_p})
            elif 'base_on' not in refs:
                long = pd.read_csv(res_dir / 'model_pair_distances_long.csv')
                for cond in ('on', 'acute_off'):
                    sel = long[(long.tag == f'{base}_nocap') & (long.condition == cond)]
                    refs[f'base_{cond}'] = sel.set_index(['odor_a', 'odor_b'])['distance']
        row.update(spearman_with_refs(on_p, off_p, refs))
        finish(tag, meta, row, on_p, off_p)
        done.add(tag)
        if pairs:
            pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
                ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
                mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
            )
            pairs.clear()

    # --- breadth-weighted APL (E12, k = 3) on top of a capped bouton model ---
    for cap in (args.breadth or []):
        src = f'bouton_tanh{cap:g}'
        tag = f'{src}_breadth{BREADTH_K:g}'
        if tag in done:
            continue
        if src not in tuned:
            print(f'{tag}: run the sweep first ({src} not tuned)', flush=True)
            continue
        from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator
        refs = reference_pairs('bouton')
        t0 = time.time()
        try:
            initial = {k: tuned[src][k] for k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale',
                'wAPLPN_scale', 'wPNAPL_scale')}
            kws = dict(BOUTON, claw_sat=cap, claw_sat_hard=False, make_plots=False, silent=True)
            _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'],
                wAPLKC=initial['wAPLKC_scale'], wKCAPL=initial['wKCAPL_scale'],
                wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **kws
            )
            wnorm = {k: pd0[k] / pd0[f'{k}_scale'] for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
            claw_kc = wnorm['wAPLKC'].index.get_level_values(H.KC_ID_LEVEL)
            # breadth (spikes summed over odors) from this capped model's own APL-off run
            off_spk = pd.read_parquet(out_dir / f'{src}_acute_off_spike_counts.parquet')
            b = pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(),
                index=off_spk.index.get_level_values(H.KC_ID_LEVEL)) + 1.0
            f_claw = ((b / b.mean()) ** BREADTH_K).reindex(claw_kc).fillna(1.0).to_numpy()
            sim, _ = make_fixed_param_kc_simulator(orn, initial_params=initial, **kws,
                _wAPLKC=wnorm['wAPLKC'] * f_claw, _wKCAPL=wnorm['wKCAPL'],
                _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL']
            )
            on_resp = pd.read_parquet(out_dir / f'{src}_on_responses.parquet')
            n_target = int((on_resp.astype(float) != 0).to_numpy().sum())
            f, r_on, s_on = match_kc_apl_scale(sim, dict(initial), n_target)
            p_on = dict(initial, wAPLKC_scale=initial['wAPLKC_scale'] * f,
                wKCAPL_scale=initial['wKCAPL_scale'] * f)
            r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
            del sim
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, base='bouton', claw_sat=cap, form='tanh',
                breadth_k=BREADTH_K, error=repr(e)[:300]))
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            continue
        for cond, (r, s) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
            r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
            s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag,
            claw_sat=cap, form='tanh', breadth_k=BREADTH_K, kc_apl_factor=f,
            fixed_thr_mean=initial['fixed_thr'], n_kcs=len(r_on), n_glomeruli=wPNKC.shape[1],
            rate_on=rate(r_on), rate_off=rate(r_off), seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
            **{k: float(v) for k, v in p_on.items()},
        )
        row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs, drive,
            data, ref
        )
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(r_on, r_off, on_p, off_p, drive, data_rates))
        for cond, s in (('on', s_on), ('off', s_off)):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs))
        finish(tag, meta, row, on_p, off_p)
        done.add(tag)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
            ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
            mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
        pairs.clear()

    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E16_claw_saturation.py', host=subprocess.run(['hostname'],
            capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        bases=BASE_KWS, caps=args.caps, forms=args.forms, thr=args.thr, breadth=args.breadth,
        approx_conc_note=H.APPROX_CONC_NOTE,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.read_csv(stats_path), res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
