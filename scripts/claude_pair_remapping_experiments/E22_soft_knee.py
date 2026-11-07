#!/usr/bin/env python3
"""E22: a softer per-claw knee (smooth threshold) on the bouton base.

The per-claw threshold that reproduces the coincidence rule (E16, olfsysm `claw_thr`) is a
hard knee, max(x - thr, 0): an odor recruits KCs only through claws driven clearly above
40-70 Hz, and odors with narrow or weak input (ms, B-myr, euc, paa, 1-8ol) lose their
responders while the fly recruits 10-14 % of KCs for them with APL blocked (E20, E21). Here
the knee is the softplus w * log(1 + exp((x - thr) / w)) (olfsysm `claw_thr_width` = w; the
hard knee as w -> 0), at claw_thr 70 on the bouton base, w in {0, 10, 20, 40} Hz, thresholds
tuned at 5 % with APL on, acute APL off; scored as E16 / E20 (E13 statistics, E7 excess by
block, rank agreement with the coincidence-3 and claw_thr 70 models' matrices, per-odor
response fractions) plus E21's per-pair APL-off sharing deficit vs the TNT data. `--combo W`
runs the three-mechanism grid's block-first point (claw_thr 70, breadth 2, KC-KC 100; E19)
with that width.

Outputs: <shared>/results/E22_soft_knee/ and ./outputs/E22_soft_knee/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E22_soft_knee.py [--widths 0 10 20 40]
    [--thr 70] [--combo W ...] [--figure-only]
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
from E1_apl_knockout import bisect_threshold
from E3_drive_dependent_gain import BOUTON, TARGET, match_kc_apl_scale, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E21_apl_off_sharing import pair_excess, SUBST

EXPERIMENT = 'E22_soft_knee'
E16_OUT = H.HERE / 'outputs' / 'E16_claw_saturation'
TAU = 0.05
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
WEAK = ['ms', 'B-myr', 'euc', 'paa', '-aPine', '1-8ol']


def sharing_deficit(spk_off: pd.DataFrame, data_pairs: pd.DataFrame) -> dict:
    """E21's per-pair deficit (model APL-off excess - TNT excess) summarized."""
    ex = pair_excess(spk_off)
    d = []
    for (a, b), v in ex.items():
        t = data_pairs['TNT_excess'].get((a, b), data_pairs['TNT_excess'].get((b, a), np.nan))
        d.append(dict(a=a, b=b, deficit=v - t, subst=int(a in SUBST or b in SUBST), weak=int(a in WEAK or b in WEAK),
            block=H.block_label(a, b)))
    d = pd.DataFrame(d)
    return dict(deficit_all=float(d.deficit.mean()), deficit_weak_pairs=float(d[d.weak == 1].deficit.mean()),
        deficit_other_pairs=float(d[d.weak == 0].deficit.mean()),
        deficit_block=float(d[d.block == H.FRUITY_SMALL_BLOCK].deficit.mean()),
        deficit_abs_mean=float(d.deficit.abs().mean()))


def make_figure(stats: pd.DataFrame, res_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    sweep = stats[stats.get('k', pd.Series(index=stats.index, dtype=float)).isna()] if 'k' in stats else stats
    sweep = sweep.sort_values('claw_thr_width')
    panels = [
        ('per-odor rate rank agreement, APL on / off (ceiling 0.98 / 0.97)', ['spearman_rate_on_vs_data', 'spearman_rate_off_vs_data'], None),
        ('mean pair distance, APL on (data 0.84, graded cosine)', [p + 'on_mean_model'], 0.840),
        ('excess sharing, block pairs, APL on (data -0.08)', ['on_excess_block'], -0.080),
        ('APL-off sharing deficit: weak-odor pairs / other pairs (0 = data)', ['deficit_weak_pairs', 'deficit_other_pairs'], 0.0),
        ('rank agreement with control (on) / TNT (off)', [p + 'on_spearman', p + 'off_spearman'], None),
        ('fruity x small-ester block (data +0.055)', [p + 'block_fruity_small_iso'], 0.055),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (title, cols, ref) in zip(axes.ravel(), panels):
        for i, col in enumerate(cols):
            if col in sweep:
                ax.plot(sweep['claw_thr_width'], sweep[col], '-o' if i == 0 else '--s', color='#2b6cb0', ms=5, lw=1.3, label=col.replace(p, ''))
        if ref is not None:
            ax.axhline(ref, color='0.45', lw=1, ls=':')
        ax.set_xlabel('knee width w (Hz; 0 = hard)'); ax.set_title(title, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
        if len(cols) > 1:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle('E22: softer per-claw knee at claw_thr 70 (bouton base, 5 %)', fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / 'E22_soft_knee.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--widths', nargs='+', type=float, default=[0.0, 10.0, 20.0, 40.0])
    ap.add_argument('--thr', type=float, default=70.0)
    ap.add_argument('--combo', nargs='*', type=float, default=None)
    ap.add_argument('--no-sweep', action='store_true')
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()

    res_dir = H.results_dir(EXPERIMENT)
    out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        make_figure(pd.DataFrame(rows), res_dir); return
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
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    refs = all_refs()
    params_path = out_dir / 'tuned_params.json'
    tuned = json.loads(params_path.read_text()) if params_path.exists() else {}
    fracs_path = res_dir / 'per_odor_response_fractions.csv'
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    pairs = {}

    def record(tag, meta, row, on_p, off_p, r_on, r_off):
        rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(r_on).reindex(fracs.index); fracs[f'{tag}_acute_off'] = H.response_fractions(r_off).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        v = H.PRIMARY_MODEL_RDM
        weak_off = ' '.join(f"{o}:{fracs.loc[o, f'{tag}_acute_off']:.3f}" for o in WEAK if o in fracs.index)
        print(f"{tag}: thr {meta.get('fixed_thr_mean', float('nan')):.0f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f}; "
            f"rate rank on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; mean dist on "
            f"{row[v + '__on_mean_model']:.3f}; block {row[v + '__block_fruity_small_iso']:+.3f}; excess on {row['on_excess_block']:+.3f}/"
            f"{row['on_excess_other_between']:+.3f}; vs ctrl {row[v + '__on_spearman']:.3f}; off vs TNT {row[v + '__off_spearman']:.3f}; "
            f"deficit weak/other {row['deficit_weak_pairs']:+.3f}/{row['deficit_other_pairs']:+.3f}; weak odors off [{weak_off}]; {meta['seconds']:.0f}s", flush=True)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists())
        pairs.clear()

    if not args.no_sweep:
        for w in args.widths:
            tag = f'thr{args.thr:g}_w{w:g}'
            if tag in done:
                continue
            kws = dict(BOUTON, claw_thr=args.thr)
            if w > 0:
                kws['claw_thr_width'] = w
            t0 = time.time()
            try:
                on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag, with_pn_contrast=False)
            except Exception as e:
                print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                rows.append(dict(variant=tag, tag=tag, claw_thr=args.thr, claw_thr_width=w, error=repr(e)[:300]))
                pd.DataFrame(rows).to_csv(stats_path, index=False); continue
            tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
            params_path.write_text(json.dumps(tuned, indent=1))
            meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag, claw_thr=args.thr, claw_thr_width=w,
                n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
                mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()), **extra,
                **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale') if k in params})
            meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
            row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
            on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
            row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
            for cond, s in (('on', on_out[1]), ('off', off_out[1])):
                row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
            row.update(spearman_with_refs(on_p, off_p, refs))
            row.update(sharing_deficit(off_out[1], data_pairs))
            record(tag, meta, row, on_p, off_p, on_out[0], off_out[0]); done.add(tag)

    for w in (args.combo or []):
        src = f'thr{args.thr:g}_w{w:g}'; tag = f'{src}_k2_kckc100'
        if tag in done:
            continue
        if src not in tuned:
            print(f'{tag}: {src} not tuned yet', flush=True); continue
        from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator
        initial = {k: tuned[src][k] for k in PARAM_KEYS}
        thr_kws = dict(BOUTON, claw_thr=args.thr, make_plots=False, silent=True)
        if w > 0:
            thr_kws['claw_thr_width'] = w
        t0 = time.time()
        try:
            _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'], wAPLKC=initial['wAPLKC_scale'], wKCAPL=initial['wKCAPL_scale'],
                wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **thr_kws)
            wnorm = {k: pd0[k] / pd0[f'{k}_scale'] for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
            claw_kc = wnorm['wAPLKC'].index.get_level_values(H.KC_ID_LEVEL)
            off_spk = pd.read_parquet(out_dir / f'{src}_acute_off_spike_counts.parquet')
            b = pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(), index=off_spk.index.get_level_values(H.KC_ID_LEVEL)) + 1.0
            f_claw = ((b / b.mean()) ** 2.0).reindex(claw_kc).fillna(1.0).to_numpy()
            on_resp0 = pd.read_parquet(out_dir / f'{src}_on_responses.parquet')
            n_target = int((on_resp0.astype(float) != 0).to_numpy().sum()); rate_target = rate(on_resp0)
            sim, _ = make_fixed_param_kc_simulator(orn, initial_params=initial, **thr_kws, _wAPLKC=wnorm['wAPLKC'] * f_claw, _wKCAPL=wnorm['wKCAPL'],
                _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=0.0, kc_kc_tau=TAU)
            f, _, _ = match_kc_apl_scale(sim, dict(initial, wKCKC_scale=0.0), n_target)
            p_k = dict(initial, wAPLKC_scale=initial['wAPLKC_scale'] * f, wKCAPL_scale=initial['wKCAPL_scale'] * f, wKCKC_scale=0.0)
            extra = {kk: p_k[kk] for kk in ('wAPLKC_scale', 'wKCAPL_scale')}; extra['wKCKC_scale'] = 100.0
            new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_k['fixed_thr'], rate_target, extra=extra)
            p_on = dict(p_k, fixed_thr=new_thr, wKCKC_scale=100.0)
            r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0)); del sim
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, claw_thr=args.thr, claw_thr_width=w, k=2.0, kc_kc_inh=100.0, error=repr(e)[:300]))
            pd.DataFrame(rows).to_csv(stats_path, index=False); continue
        for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
            r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet'); sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag, claw_thr=args.thr, claw_thr_width=w, k=2.0, kc_kc_inh=100.0,
            kc_apl_factor=f, thr_factor=thr_f, fixed_thr_mean=float(np.mean(new_thr)), n_kcs=len(r_on), n_glomeruli=wPNKC.shape[1],
            rate_on=rate(r_on), rate_off=rate(r_off), seconds=time.time() - t0, mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
            **{kk: float(np.mean(v)) for kk, v in p_on.items() if kk != 'fixed_thr'})
        meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
        row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs, drive, data, ref)
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(r_on, r_off, on_p, off_p, drive, data_rates))
        for cond, sp in (('on', s_on), ('off', s_off)):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(sp).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(s_off, data_pairs))
        record(tag, meta, row, on_p, off_p, r_on, r_off); done.add(tag)

    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E22_soft_knee.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.olfsysm_dir(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, thr=args.thr, widths=args.widths, combo=args.combo, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
