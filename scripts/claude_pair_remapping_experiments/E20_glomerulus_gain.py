#!/usr/bin/env python3
"""E20: per-glomerulus drive normalization ahead of the per-claw threshold.

The per-claw threshold that reproduces the coincidence rule (E16, olfsysm `claw_thr`) is
absolute in PN-rate units, so odors driving only weakly firing glomeruli lose responders and
the per-odor rate rank agreement with the KC data falls (bouton model with connectome APL
weights and no threshold 0.68 -> 0.57 at a 70 Hz threshold; ceiling 0.98). Here each
glomerulus's PN->claw weights are scaled by g = (median peak / glomerulus peak)^alpha, where
the peak is that glomerulus's bouton output (odor-window mean, max over the 16 odors) in the
base model, so every glomerulus's dynamic range straddles the threshold at alpha = 1. Sweep
alpha in {0, 0.5, 1} x claw_thr in {0, 40, 70} on the bouton base, thresholds tuned at 5 %,
acute APL off; scored as E16 / E18 plus the per-odor response fractions. `--combo ALPHA`
then runs E19's balanced point (claw_thr 40, breadth k 2, KC-KC 30) with that gain.

Outputs: <shared>/results/E20_glomerulus_gain/ and ./outputs/E20_glomerulus_gain/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E20_glomerulus_gain.py
    [--alpha 0 0.5 1] [--thr 0 40 70] [--combo ALPHA ...] [--figure-only]
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
from E12_breadth_apl import density_excess, e3_bouton_params
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs

EXPERIMENT = 'E20_glomerulus_gain'
TAU = 0.05
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')


def glomerulus_peaks(orn: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Base bouton model (E3 parameters, APL on): its PN->claw weights and each glomerulus's
    peak bouton output over the panel (odor-window mean per odor, max over odors, median
    over the glomerulus's boutons). Cached in out_dir."""
    cache = out_dir / 'glomerulus_peaks.csv'
    from al_analysis.mb_model import fit_mb_model, get_time_index, get_sim_var
    init = e3_bouton_params()
    _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=init['fixed_thr'], wAPLKC=init['wAPLKC_scale'],
        wKCAPL=init['wKCAPL_scale'], wAPLPN=init['wAPLPN_scale'], wPNAPL=init['wPNAPL_scale'],
        **BOUTON, make_plots=False, silent=True, return_olfsysm_vars=True
    )
    if cache.exists():
        peaks = pd.read_csv(cache, index_col=0).iloc[:, 0]
        return wPNKC, peaks
    mp = pd0['mp']; rv = pd0['rv']
    t = np.asarray(get_time_index(mp, rv)); win = (t >= 0.0) & (t <= 0.5)
    bs = get_sim_var(rv.pn, 'bouton_sims')
    per_bouton = np.max([np.asarray(bs[i])[:, win].mean(axis=1) for i in range(len(bs))], axis=0)
    assert len(per_bouton) == wPNKC.shape[1], (len(per_bouton), wPNKC.shape)
    gl = wPNKC.columns.get_level_values('glomerulus')
    peaks = pd.Series(per_bouton, index=gl).groupby(level=0).median().rename('peak_bouton_rate')
    peaks.to_csv(cache)
    return wPNKC, peaks


def gained_wPNKC(wPNKC: pd.DataFrame, peaks: pd.Series, alpha: float, orn_glomeruli
    ) -> tuple[pd.DataFrame, pd.Series]:
    """Per-glomerulus gain (median peak / peak)^alpha on the PN->claw weights, for the
    glomeruli that have ORN input for the panel only (the model wiring also has 18
    glomeruli with boutons but no ORN data, which receive spontaneous PN input only; their
    peak is their spontaneous rate and must not be boosted)."""
    pk = peaks.reindex([g for g in peaks.index if g in set(orn_glomeruli)])
    ref = float(pk.median())
    g = ((ref / pk) ** alpha).reindex(peaks.index).fillna(1.0)
    col_gain = g.reindex(wPNKC.columns.get_level_values('glomerulus')).to_numpy()
    return wPNKC * col_gain[None, :], g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--alpha', nargs='+', type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument('--thr', nargs='+', type=float, default=[0.0, 40.0, 70.0])
    ap.add_argument('--combo', nargs='*', type=float, default=None,
        help='alphas to run at E19\'s balanced point (claw_thr 40, k 2, KC-KC 30)')
    ap.add_argument('--no-sweep', action='store_true')
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
    data_fracs = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_rates = data_fracs
    refs = all_refs()
    wPNKC0, peaks = glomerulus_peaks(orn, out_dir)
    print('glomerulus peak bouton rates (Hz):', peaks.round(0).sort_values().to_dict(), flush=True)
    print('gain applied to', sum(g in set(orn.index) for g in peaks.index), 'glomeruli with ORN input; the other',
        sum(g not in set(orn.index) for g in peaks.index), 'keep gain 1', flush=True)
    params_path = out_dir / 'tuned_params.json'
    tuned = json.loads(params_path.read_text()) if params_path.exists() else {}
    fracs_path = res_dir / 'per_odor_response_fractions.csv'
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame(
        {'data_TNT3c': data_fracs['TNT3c'], 'data_TNT': data_fracs['TNT']})
    pairs = {}

    def record(tag, meta, row, on_p, off_p, r_on, r_off):
        rows.append(row)
        pairs[(tag, 'on')] = on_p
        pairs[(tag, 'acute_off')] = off_p
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(r_on).reindex(fracs.index)
        fracs[f'{tag}_acute_off'] = H.response_fractions(r_off).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        v = H.PRIMARY_MODEL_RDM
        print(f"{tag}: thr {meta.get('fixed_thr_mean', float('nan')):.0f}, rates on {meta['rate_on']:.4f} off "
            f"{meta['rate_off']:.4f}; rate rank agreement on/off {row['spearman_rate_on_vs_data']:.2f}/"
            f"{row['spearman_rate_off_vs_data']:.2f}; mean dist on {row[v + '__on_mean_model']:.3f}; block "
            f"{row[v + '__block_fruity_small_iso']:+.3f}; excess on {row['on_excess_block']:+.3f}/"
            f"{row['on_excess_other_between']:+.3f} off {row['off_excess_block']:+.3f}/{row['off_excess_other_between']:+.3f}; "
            f"vs ctrl {row[v + '__on_spearman']:.3f} dev {row[v + '__on_spearman_dev']:+.2f}; off vs TNT "
            f"{row[v + '__off_spearman']:.3f}; vs coinc3/thr70 {row.get('spearman_vs_coinc3_on', float('nan')):.2f}/"
            f"{row.get('spearman_vs_thr70_on', float('nan')):.2f}; {meta['seconds']:.0f}s", flush=True
        )
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
            ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
            mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
        pairs.clear()

    if not args.no_sweep:
        for alpha in args.alpha:
            W, g = gained_wPNKC(wPNKC0, peaks, alpha, orn.index)
            for thr in args.thr:
                tag = f'a{alpha:g}_thr{thr:g}'
                if tag in done:
                    continue
                kws = dict(BOUTON, _wPNKC=W)
                if thr > 0:
                    kws['claw_thr'] = thr
                t0 = time.time()
                try:
                    on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag,
                        with_pn_contrast=False)
                except Exception as e:
                    print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                    rows.append(dict(variant=tag, tag=tag, alpha=alpha, claw_thr=thr, error=repr(e)[:300]))
                    pd.DataFrame(rows).to_csv(stats_path, index=False)
                    continue
                tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
                params_path.write_text(json.dumps(tuned, indent=1))
                meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag, alpha=alpha,
                    claw_thr=thr, gain_min=float(g.min()), gain_max=float(g.max()),
                    n_kcs=len(on_out[0]), n_glomeruli=len(peaks), rate_on=rate(on_out[0]), rate_off=rate(off_out[0]),
                    seconds=time.time() - t0, mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
                    **extra, **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale') if k in params},
                )
                meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
                row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
                on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
                row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
                for cond, s in (('on', on_out[1]), ('off', off_out[1])):
                    row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
                if tag == 'a0_thr0':
                    refs.update({'base_on': on_p, 'base_acute_off': off_p})
                row.update(spearman_with_refs(on_p, off_p, refs))
                record(tag, meta, row, on_p, off_p, on_out[0], off_out[0])
                done.add(tag)

    # --- E19's balanced point with the gain ---
    for alpha in (args.combo or []):
        src = f'a{alpha:g}_thr40'
        tag = f'{src}_k2_kckc30'
        if tag in done:
            continue
        if src not in tuned:
            print(f'{tag}: {src} not tuned yet', flush=True); continue
        from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator
        W, g = gained_wPNKC(wPNKC0, peaks, alpha, orn.index)
        initial = {k: tuned[src][k] for k in PARAM_KEYS}
        thr_kws = dict(BOUTON, _wPNKC=W, claw_thr=40.0, make_plots=False, silent=True)
        t0 = time.time()
        try:
            _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'], wAPLKC=initial['wAPLKC_scale'],
                wKCAPL=initial['wKCAPL_scale'], wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **thr_kws)
            wnorm = {k: pd0[k] / pd0[f'{k}_scale'] for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
            claw_kc = wnorm['wAPLKC'].index.get_level_values(H.KC_ID_LEVEL)
            off_spk = pd.read_parquet(out_dir / f'{src}_acute_off_spike_counts.parquet')
            b = pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(),
                index=off_spk.index.get_level_values(H.KC_ID_LEVEL)) + 1.0
            f_claw = ((b / b.mean()) ** 2.0).reindex(claw_kc).fillna(1.0).to_numpy()
            on_resp0 = pd.read_parquet(out_dir / f'{src}_on_responses.parquet')
            n_target = int((on_resp0.astype(float) != 0).to_numpy().sum())
            rate_target = rate(on_resp0)
            sim, _ = make_fixed_param_kc_simulator(orn, initial_params=initial, **thr_kws,
                _wAPLKC=wnorm['wAPLKC'] * f_claw, _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'],
                _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=0.0, kc_kc_tau=TAU)
            f, _, _ = match_kc_apl_scale(sim, dict(initial, wKCKC_scale=0.0), n_target)
            p_k = dict(initial, wAPLKC_scale=initial['wAPLKC_scale'] * f, wKCAPL_scale=initial['wKCAPL_scale'] * f, wKCKC_scale=0.0)
            extra = {kk: p_k[kk] for kk in ('wAPLKC_scale', 'wKCAPL_scale')}; extra['wKCKC_scale'] = 30.0
            new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_k['fixed_thr'], rate_target, extra=extra)
            p_on = dict(p_k, fixed_thr=new_thr, wKCKC_scale=30.0)
            r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
            del sim
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, alpha=alpha, claw_thr=40.0, k=2.0, kc_kc_inh=30.0, error=repr(e)[:300]))
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            continue
        for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
            r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
            sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=TARGET, tag=tag, alpha=alpha, claw_thr=40.0,
            k=2.0, kc_kc_inh=30.0, kc_apl_factor=f, thr_factor=thr_f, fixed_thr_mean=float(np.mean(new_thr)),
            n_kcs=len(r_on), n_glomeruli=len(peaks), rate_on=rate(r_on), rate_off=rate(r_off), seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
            **{kk: float(np.mean(v)) for kk, v in p_on.items() if kk != 'fixed_thr'})
        meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
        row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs, drive, data, ref)
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(r_on, r_off, on_p, off_p, drive, data_rates))
        for cond, sp in (('on', s_on), ('off', s_off)):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(sp).items()})
        row.update(spearman_with_refs(on_p, off_p, refs))
        record(tag, meta, row, on_p, off_p, r_on, r_off)
        done.add(tag)

    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E20_glomerulus_gain.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path('~/src/olfsysm').expanduser(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, alpha=args.alpha, thr=args.thr, combo=args.combo, peaks=peaks.round(1).to_dict(),
        approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), res_dir)
    print('done ->', res_dir)


def make_figure(stats: pd.DataFrame, res_dir: Path) -> None:
    """Top: statistics vs alpha per claw_thr. Bottom: per-odor response fractions, model vs
    data, for the untouched base, the 70 Hz threshold without and with full normalization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    sweep = stats[stats.get('k', pd.Series(index=stats.index, dtype=float)).isna()] if 'k' in stats else stats
    panels = [
        ('per-odor rate rank agreement, APL on (ceiling 0.98)', 'spearman_rate_on_vs_data', 0.98),
        ('per-odor rate rank agreement, APL off (ceiling 0.97)', 'spearman_rate_off_vs_data', 0.97),
        ('mean pair distance, APL on (data 0.84, graded cosine)', p + 'on_mean_model', 0.840),
        ('excess sharing, block pairs, APL on (data -0.08)', 'on_excess_block', -0.080),
        ('rank agreement with control (APL on)', p + 'on_spearman', None),
        ('rank agreement with TNT (APL off)', p + 'off_spearman', None),
    ]
    colors = {0.0: '0.35', 40.0: '#2b6cb0', 70.0: '#d9480f'}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, (title, col, ref_) in zip(axes.ravel()[:6], panels):
        for thr, ss in sweep.groupby('claw_thr'):
            ss = ss.sort_values('alpha')
            ax.plot(ss['alpha'], ss[col], '-o', color=colors.get(float(thr), '0.5'), ms=5, lw=1.3, label=f'claw_thr {thr:g}')
        if ref_ is not None:
            ax.axhline(ref_, color='0.45', lw=1, ls=':')
        ax.set_xlabel('normalization strength alpha')
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, color='0.9', lw=0.6)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fr_path = res_dir / 'per_odor_response_fractions.csv'
    if fr_path.exists():
        fr = pd.read_csv(fr_path, index_col=0)
        for ax, (col, label, cond, dcol) in zip(axes.ravel()[6:], [
                ('a0_thr0_on', 'untouched base, APL on', 'on', 'data_TNT3c'),
                ('a0_thr70_on', 'claw_thr 70, no normalization, APL on', 'on', 'data_TNT3c'),
                ('a1_thr70_on', 'claw_thr 70, full normalization, APL on', 'on', 'data_TNT3c')]):
            if col not in fr:
                ax.set_visible(False); continue
            x = np.arange(len(fr.index))
            ax.bar(x - 0.2, fr[dcol], width=0.4, color='0.6', label='KC data (control)')
            ax.bar(x + 0.2, fr[col], width=0.4, color='#2b6cb0', label='model')
            ax.set_xticks(x); ax.set_xticklabels(fr.index, rotation=90, fontsize=7)
            ax.set_title(f'response fraction per odor: {label}', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
        axes[2, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('E20: per-glomerulus drive normalization ahead of the per-claw threshold (bouton base, 5 %)', fontsize=11)
    fig.tight_layout()
    fig.savefig(res_dir / 'E20_glomerulus_gain.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
