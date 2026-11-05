#!/usr/bin/env python3
"""E19: per-claw threshold x breadth-weighted APL x KC-KC inhibition on the bouton base.

E16 made the coincidence rule a parameter (`claw_thr`, a rectified per-claw offset that keeps
somatic integration); E8 found breadth-weighted APL (E12, k = 3) with connectome KC-KC
inhibition (scale 30-100, tau 50 ms) gives the fruity x small-ester block and the control's
excess profile without an oracle, but not the decorrelation level or the APL-off excess,
which the per-claw threshold fixes. Here the three are combined: bouton base at the E16
operating points claw_thr in {40, 70} (tuned threshold and APL scales from E16's
tuned_params.json), x breadth exponent k in {0, 1, 2, 3} (breadth = that model's own APL-off
spike counts summed over odors; KC<->APL scales re-matched to the k = 0 response count, as
E12) x kc_kc_inh in {0, 30, 100} at tau 50 ms (APL-on rate re-matched by threshold
re-tuning, as E8); acute APL off = APL output weights zero. Scored as E8.

Outputs: <shared>/results/E19_thr_breadth_kckc/ and ./outputs/E19_thr_breadth_kckc/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E19_thr_breadth_kckc.py
    [--thr 40 70] [--exponents 0 1 2 3] [--scales 0 30 100] [--figure-only]
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
from E3_drive_dependent_gain import BOUTON, match_kc_apl_scale, per_odor_extras, rate
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess

EXPERIMENT = 'E19_thr_breadth_kckc'
E16_OUT = H.HERE / 'outputs' / 'E16_claw_saturation'
TAU = 0.05
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')


def make_figure(stats: pd.DataFrame, res_dir: Path) -> None:
    """Statistics vs breadth exponent k: hue = claw_thr, marker = KC-KC scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    panels = [
        ('fruity x small-ester block (data +0.055)', p + 'block_fruity_small_iso', 0.055),
        ('Spearman with control (APL on)', p + 'on_spearman', None),
        ('Spearman with TNT (APL off)', p + 'off_spearman', None),
        ('excess sharing, block pairs, APL on (data -0.08)', 'on_excess_block', -0.080),
        ('excess sharing, block pairs, APL off (data -0.24)', 'off_excess_block', -0.239),
        ('mean pair distance, APL on (data 0.86)', p + 'on_mean_model', 0.855),
        ('APL-off / APL-on response rate (data 1.9)', 'rate_ratio_off_over_on', 1.9),
        ('per-odor rate vs data, APL on', 'spearman_rate_on_vs_data', None),
        ('isotonic R2 on vs off (data 0.75)', p + 'iso_r2', 0.75),
    ]
    colors = {40.0: '#2b6cb0', 70.0: '#d9480f', 0.0: '0.35'}
    markers = {0.0: 'o', 30.0: 's', 100.0: '^'}
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.5))
    for ax, (title, col, ref) in zip(axes.ravel(), panels):
        for (thr, s), ss in stats.groupby(['claw_thr', 'kc_kc_inh']):
            ss = ss.sort_values('k')
            ax.plot(ss['k'], ss[col], '-', color=colors.get(float(thr), '0.3'),
                marker=markers.get(float(s), 'o'), ms=5, lw=1.3,
                label=f'claw_thr {thr:g}, KC-KC {s:g}')
        if ref is not None:
            ax.axhline(ref, color='0.45', lw=1, ls=':')
        ax.set_xlabel('breadth exponent k')
        ax.set_xticks(sorted(stats['k'].unique()))
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, color='0.9', lw=0.6)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle('E19: per-claw threshold x breadth-weighted APL x KC-KC inhibition (bouton base, 5 %)', fontsize=11)
    fig.tight_layout()
    fig.savefig(res_dir / 'E19_thr_breadth_kckc.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--thr', nargs='+', type=float, default=[40.0, 70.0])
    ap.add_argument('--exponents', nargs='+', type=float, default=[0.0, 1.0, 2.0, 3.0])
    ap.add_argument('--scales', nargs='+', type=float, default=[0.0, 30.0, 100.0])
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
    tuned = json.loads((E16_OUT / 'tuned_params.json').read_text())
    from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator
    pairs = {}

    for thr in args.thr:
        src = f'bouton_thr{thr:g}' if thr > 0 else 'bouton_nocap'
        initial = {k: float(tuned[src][k]) for k in PARAM_KEYS}
        thr_kws = dict(BOUTON, make_plots=False, silent=True)
        if thr > 0:
            thr_kws['claw_thr'] = thr
        # normalized APL weights of this base (hardcoded _w* overrides are used unnormalized)
        _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'],
            wAPLKC=initial['wAPLKC_scale'], wKCAPL=initial['wKCAPL_scale'],
            wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **thr_kws
        )
        wnorm = {k: pd0[k] / pd0[f'{k}_scale'] for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
        claw_kc = wnorm['wAPLKC'].index.get_level_values(H.KC_ID_LEVEL)
        off_spk = pd.read_parquet(E16_OUT / f'{src}_acute_off_spike_counts.parquet')
        b = pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(),
            index=off_spk.index.get_level_values(H.KC_ID_LEVEL)) + 1.0
        on_resp0 = pd.read_parquet(E16_OUT / f'{src}_on_responses.parquet')
        n_target = int((on_resp0.astype(float) != 0).to_numpy().sum())
        rate_target = rate(on_resp0)
        mean_claws = float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean())
        for k in args.exponents:
            f_claw = ((b / b.mean()) ** k).reindex(claw_kc).fillna(1.0).to_numpy()
            sim = None
            p_k = None
            for s in args.scales:
                tag = f'thr{thr:g}_k{k:g}_kckc{s:g}'
                if tag in done:
                    continue
                t0 = time.time()
                try:
                    if sim is None:
                        sim, _ = make_fixed_param_kc_simulator(orn, initial_params=initial,
                            **thr_kws, _wAPLKC=wnorm['wAPLKC'] * f_claw, _wKCAPL=wnorm['wKCAPL'],
                            _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=0.0,
                            kc_kc_tau=TAU
                        )
                    if p_k is None:
                        if k == 0.0:
                            f = 1.0
                        else:
                            f, _, _ = match_kc_apl_scale(sim, dict(initial, wKCKC_scale=0.0), n_target)
                        p_k = dict(initial, wAPLKC_scale=initial['wAPLKC_scale'] * f,
                            wKCAPL_scale=initial['wKCAPL_scale'] * f, wKCKC_scale=0.0)
                    if s == 0.0:
                        p_on = dict(p_k)
                        thr_f = 1.0
                        r_on, s_on = sim(p_on)
                    else:
                        extra = {kk: p_k[kk] for kk in ('wAPLKC_scale', 'wKCAPL_scale')}
                        extra['wKCKC_scale'] = s
                        new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_k['fixed_thr'],
                            rate_target, extra=extra)
                        p_on = dict(p_k, fixed_thr=new_thr, wKCKC_scale=s)
                    r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                except Exception as e:
                    print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                    rows.append(dict(variant=tag, tag=tag, claw_thr=thr, k=k, kc_kc_inh=s,
                        error=repr(e)[:300]))
                    pd.DataFrame(rows).to_csv(stats_path, index=False)
                    continue
                for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
                    r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
                    sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
                meta = dict(variant=tag, base='bouton', seed=None, target_rate=0.05, tag=tag,
                    claw_thr=thr, k=k, kc_kc_inh=s, kc_kc_tau=TAU, kc_apl_factor=p_k['wAPLKC_scale'] / initial['wAPLKC_scale'],
                    thr_factor=thr_f, n_kcs=len(r_on), n_glomeruli=wPNKC.shape[1],
                    rate_on=rate(r_on), rate_off=rate(r_off), seconds=time.time() - t0,
                    mean_claws_per_kc=mean_claws,
                    **{kk: float(np.mean(v)) for kk, v in p_on.items()},
                )
                meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
                row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs,
                    drive, data, ref
                )
                row.update(per_odor_extras(r_on, r_off, on[H.PRIMARY_MODEL_RDM],
                    off[H.PRIMARY_MODEL_RDM], drive, data_rates
                ))
                for cond, sp in (('on', s_on), ('off', s_off)):
                    row.update({f'{cond}_{kk}': v for kk, v in density_excess(sp).items()})
                rows.append(row)
                pairs[(tag, 'on')] = on[H.PRIMARY_MODEL_RDM]
                pairs[(tag, 'acute_off')] = off[H.PRIMARY_MODEL_RDM]
                pd.DataFrame(rows).to_csv(stats_path, index=False)
                v = H.PRIMARY_MODEL_RDM
                print(f"{tag}: apl x{meta['kc_apl_factor']:.3f} thr x{thr_f:.3f}, rates on {meta['rate_on']:.4f} "
                    f"off {meta['rate_off']:.4f} (x{meta['rate_ratio_off_over_on']:.2f}); mean dist on "
                    f"{row[v + '__on_mean_model']:.3f}; block {row[v + '__block_fruity_small_iso']:+.3f}; "
                    f"excess on {row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f} off "
                    f"{row['off_excess_block']:+.3f}/{row['off_excess_other_between']:+.3f}; vs ctrl "
                    f"{row[v + '__on_spearman']:.3f} dev {row[v + '__on_spearman_dev']:+.2f}; off vs TNT "
                    f"{row[v + '__off_spearman']:.3f}; iso R2 {row[v + '__iso_r2']:.3f}; rate vs data "
                    f"{row['spearman_rate_on_vs_data']:.2f}; {meta['seconds']:.0f}s", flush=True
                )
                if pairs:
                    pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
                        ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
                        mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
                    )
                    pairs.clear()
            del sim

    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E19_thr_breadth_kckc.py', host=subprocess.run(['hostname'], capture_output=True,
            text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=Path('~/src/olfsysm').expanduser(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, thr=args.thr, exponents=args.exponents, scales=args.scales, tau=TAU,
        approx_conc_note=H.APPROX_CONC_NOTE,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
