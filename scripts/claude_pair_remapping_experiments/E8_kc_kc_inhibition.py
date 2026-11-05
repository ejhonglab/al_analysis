#!/usr/bin/env python3
"""E8: APL-independent KC-KC lateral inhibition (hemibrain KC->KC synapses) on the bouton base.

olfsysm now has a KC-KC term (`p.kc.use_kc_kc_inh`, `rv.kc.wKCKC`; `fit_mb_model(kc_kc_inh=,
kc_kc_tau=)`): each KC's spike output is low-pass filtered (time constant `kc_kc_tau`) and
`wKCKC @ activity` is subtracted from every KC's dVm/dt at the soma, with weights from the
hemibrain KC-claw -> KC-claw synapse counts normalized so the mean incoming weight is 1
(1 unit ~ 17.5 synapses) times `kc_kc_inh`. It acts with and without APL, so it is a
candidate for the APL-independent (TNT) structure and, being pair-specific (KCs that touch
each other inhibit each other), for the fruity x small-ester block that no per-KC APL
weighting removed (E12 / E14 / E15).

Design: bouton base at E3's tuned parameters, (a) plain and (b) with E12's k = 3 breadth
weighting of the APL->KC weights (the most data-like bouton model); KC-KC scale in
{0, 10, 30, 100, 300} with tau 0.05 s, and {30, 100} with tau 0.3 s (GPCR-like, Manoim et
al. 2022). For each: APL on with the threshold re-tuned (APL at its base scale) to the no-KC-KC model's
response rate; APL off (both APL output weights zero) with the KC-KC term kept. Scored as
E12 (E4 statistics, per-odor drive / density readouts, E7's density-null excess by block),
plus the APL-off / APL-on rate ratio (data 2.2; KC-KC inhibition can only lower it).

Outputs: <shared>/results/E8_kc_kc_inhibition/ and ./outputs/E8_kc_kc_inhibition/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E8_kc_kc_inhibition.py
    [--scales 0 10 30 100 300] [--slow-scales 30 100] [--configs plain breadth3]
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
from E3_drive_dependent_gain import BOUTON, per_odor_extras, rate
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import e3_bouton_params, density_excess

EXPERIMENT = 'E8_kc_kc_inhibition'
E3_DIR = H.HERE / 'outputs' / 'E3_drive_dependent_gain'
TAU_FAST, TAU_SLOW = 0.05, 0.3
BREADTH_K = 3.0


def make_figure(stats: pd.DataFrame, res_dir: Path) -> None:
    """Statistics vs KC-KC scale, per config (hue) and tau (marker fill)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    stats = stats[~stats.get('error', pd.Series(index=stats.index, dtype=object)).notna()]
    data_ctrl, data_tnt = -0.080, -0.239   # E7 excess (block) in the data, control / TNT
    panels = [
        ('Spearman with data', [(p + 'on_spearman', 'APL on vs control'), (p + 'off_spearman', 'APL off vs TNT')], None),
        ('fruity x small-ester block (data +0.055)', [(p + 'block_fruity_small_iso', '')], 0.055),
        ('APL-off residual vs VCF (data 0.36)', [(p + 'off_spearman_resid_vs_vcf_hyp', '')], 0.36),
        ('excess sharing, block pairs (APL on)', [('on_excess_block', '')], data_ctrl),
        ('isotonic R2 on vs off (data 0.75)', [(p + 'iso_r2', '')], 0.75),
        ('APL-off / APL-on response rate (data 1.9)', [('rate_ratio_off_over_on', '')], 1.9),
    ]
    colors = {'plain': '#2b6cb0', 'breadth3': '#d9480f'}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, (title, series, ref) in zip(axes.ravel(), panels):
        for cfg, sub in stats.groupby('config'):
            for tau, ss in sub.groupby('kc_kc_tau'):
                ss = ss.sort_values('kc_kc_inh')
                x = ss['kc_kc_inh'].to_numpy()
                for i, (col, label) in enumerate(series):
                    if col not in ss:
                        continue
                    ls = '-' if i == 0 else '--'
                    mfc = colors[cfg] if tau < 0.1 else 'white'
                    lab = f"{cfg}, tau {tau:g}" + (f" ({label})" if label else '')
                    ax.plot(x, ss[col].to_numpy(), ls, color=colors[cfg], marker='o', ms=5,
                        mfc=mfc, lw=1.5, label=lab)
        if ref is not None:
            ax.axhline(ref, color='0.4', lw=1, ls=':')
        ax.set_xscale('symlog', linthresh=10)
        ax.set_xticks([0, 10, 30, 100, 300])
        ax.set_xticklabels(['0', '10', '30', '100', '300'])
        ax.set_xlim(-3, 420)
        ax.set_xlabel('KC-KC inhibition scale (0 = off)')
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, color='0.9', lw=0.6)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('E8: KC-KC lateral inhibition on the bouton base (threshold re-tuned to 5 %)', fontsize=11)
    fig.tight_layout()
    fig.savefig(res_dir / 'E8_kc_kc_inhibition.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--scales', nargs='+', type=float, default=[0.0, 10.0, 30.0, 100.0, 300.0])
    ap.add_argument('--slow-scales', nargs='+', type=float, default=[30.0, 100.0])
    ap.add_argument('--configs', nargs='+', choices=['plain', 'breadth3'], default=['plain', 'breadth3'])
    ap.add_argument('--figure-only', action='store_true', help='only redraw the figure from the saved stats')
    args = ap.parse_args()
    from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator

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
    if args.figure_only:
        make_figure(pd.read_csv(stats_path), res_dir)
        return
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}
    pairs = {}

    initial = e3_bouton_params()
    # the bouton base's wiring and normalized APL weights (as in E12)
    _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'],
        wAPLKC=initial['wAPLKC_scale'], wKCAPL=initial['wKCAPL_scale'],
        wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **BOUTON,
        make_plots=False, silent=True
    )
    wnorm = {k: pd0[k] / pd0[f'{k}_scale'] for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
    claw_kc = wnorm['wAPLKC'].index.get_level_values(H.KC_ID_LEVEL)
    off_spk = pd.read_parquet(E3_DIR / 'bouton_pnapl_x1_acute_off_spike_counts.parquet')
    breadth = pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(),
        index=off_spk.index.get_level_values(H.KC_ID_LEVEL)
    )
    bp1 = breadth + 1.0
    f3 = ((bp1 / bp1.mean()) ** BREADTH_K).reindex(claw_kc).fillna(1.0).to_numpy()
    configs = {'plain': wnorm['wAPLKC'], 'breadth3': wnorm['wAPLKC'] * f3}
    # E12 matched the k = 3 model's response count to the plain model's by scaling the
    # KC<->APL weights (kc_apl_factor, 0.18 for spikes_k3; the breadth factors raise the
    # mean APL weight); with the plain scales the breadth config runs at ~1 % instead of
    # 5 %. Apply that factor so the breadth3, no-KC-KC model is E12's spikes_k3 model.
    e12 = pd.read_csv(H.shared_dir() / 'results' / 'E12_breadth_apl' / 'stats_variants.csv')
    f12 = float(e12.loc[e12.tag == f'spikes_k{BREADTH_K:g}', 'kc_apl_factor'].iloc[0])
    inits = {'plain': dict(initial), 'breadth3': dict(initial,
        wAPLKC_scale=initial['wAPLKC_scale'] * f12, wKCAPL_scale=initial['wKCAPL_scale'] * f12)}

    jobs = []
    for cfg in args.configs:
        for s in args.scales:
            jobs.append((cfg, TAU_FAST, s))
        for s in args.slow_scales:
            jobs.append((cfg, TAU_SLOW, s))
    sims = {}
    rate_target = {}
    for cfg, tau, s in jobs:
        tag = f'{cfg}_tau{tau:g}_kckc{s:g}'
        if tag in done:
            continue
        t0 = time.time()
        try:
            key = (cfg, tau)
            init = inits[cfg]
            if key not in sims:
                sims[key], _ = make_fixed_param_kc_simulator(orn, initial_params=init,
                    **BOUTON, _wAPLKC=configs[cfg], _wKCAPL=wnorm['wKCAPL'],
                    _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=0.0,
                    kc_kc_tau=tau, make_plots=False, silent=True
                )
                # APL-on reference (no KC-KC term, APL weights as given) sets the target count
                # for the plain config; the breadth config matches its own k = 3, no-KC-KC model
                r_ref, _ = sims[key](dict(init, wKCKC_scale=0.0))
                rate_target[key] = rate(r_ref)
            sim = sims[key]
            p = dict(init, wKCKC_scale=s)
            # the KC-KC term lowers the rate at fixed thresholds (for scales >= 30 below
            # the target even with the KC<->APL scales at 0.002x), so match the APL-on
            # rate by re-tuning the threshold with APL at its base scale: the term is
            # present in both genotypes and thresholds adapt
            thr, f, r_on, s_on = bisect_threshold(sim, init['fixed_thr'], rate_target[key],
                extra={'wKCKC_scale': s}
            )
            p_on = dict(p, fixed_thr=thr)
            r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
        except Exception as e:
            print(f'{tag} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag, tag=tag, config=cfg, kc_kc_tau=tau, kc_kc_inh=s, error=repr(e)[:300]))
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            continue
        for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
            r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
            sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        meta = dict(variant=tag, base='bouton', seed=None, target_rate=0.05, tag=tag, config=cfg,
            kc_kc_tau=tau, kc_kc_inh=s, thr_factor=f, n_kcs=len(r_on),
            n_glomeruli=wPNKC.shape[1], rate_on=rate(r_on), rate_off=rate(r_off),
            rate_ratio_off_over_on=rate(r_off) / max(rate(r_on), 1e-9), seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
        )
        row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs, drive,
            data, ref
        )
        row.update(per_odor_extras(r_on, r_off, on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM],
            drive, data_rates
        ))
        for cond, sp in (('on', s_on), ('off', s_off)):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(sp).items()})
        rows.append(row)
        pairs[(tag, 'on')] = on[H.PRIMARY_MODEL_RDM]
        pairs[(tag, 'acute_off')] = off[H.PRIMARY_MODEL_RDM]
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        v = H.PRIMARY_MODEL_RDM
        print(f"{tag}: thr x{f:.3f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f} "
            f"(x{meta['rate_ratio_off_over_on']:.2f}); block {row[v + '__block_fruity_small_iso']:+.3f}; "
            f"excess on {row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f} off "
            f"{row['off_excess_block']:+.3f}/{row['off_excess_other_between']:+.3f}; vs ctrl "
            f"{row[v + '__on_spearman']:.3f} dev {row[v + '__on_spearman_dev']:+.2f}; off vs TNT "
            f"{row[v + '__off_spearman']:.3f} dev {row[v + '__off_spearman_dev']:+.2f}; iso R2 "
            f"{row[v + '__iso_r2']:.3f}; {meta['seconds']:.0f}s", flush=True
        )

    if pairs:
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
            ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
            mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E8_kc_kc_inhibition.py', host=subprocess.run(['hostname'], capture_output=True,
            text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=Path('~/src/olfsysm').expanduser(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, initial_params=initial, scales=args.scales, slow_scales=args.slow_scales,
        taus=[TAU_FAST, TAU_SLOW], breadth_k=BREADTH_K, approx_conc_note=H.APPROX_CONC_NOTE,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.read_csv(stats_path), res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
