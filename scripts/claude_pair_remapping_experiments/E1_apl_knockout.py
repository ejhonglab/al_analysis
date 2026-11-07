#!/usr/bin/env python3
"""E1: APL knockout in the existing models, acute and homeostatic, scored against Yang's data.

For each base model (chapter-3 claw model with and without claw dynamics, and the paper
hemibrain wd20 model) tuned to 5% and 10% mean response rate with APL on:
- `on`: the tuned model
- `acute_off`: same thresholds, `wAPLKC_scale=0` (mimics TNT with no compensation)
- `homeo_full_off`: APL off, thresholds re-tuned (scalar bisection) so the mean response
  rate equals the APL-on rate (full homeostatic compensation)
- `homeo_data_off`: APL off, thresholds re-tuned so the rate is 0.150 / 0.068 = 2.2x the
  APL-on rate (the ratio in Yang's data)

Each is scored with `kc_remapping_harness`: model APL-on vs TNT3c (control), APL-off vs
TNT, the on/off relation (Spearman, linear / isotonic R2, fruity x small-ester block
residual, per-odor APL effect vs ORN drive, ORN agreement) next to the data's, and the
66-pair reference comparisons (claws, VCF, ...). All model RDM variants (spike counts /
binarized responses x cosine / Pearson) are saved; the primary one is cosine over spike
counts (the data recipe is cosine over all KCs).

Outputs: compact results (stats CSVs, matrices, figures, provenance) to
<shared>/results/E1_apl_knockout/; full responses / spike counts to
./outputs/E1_apl_knockout/.

Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E1_apl_knockout.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E1_apl_knockout'
DATA_RATE_RATIO = 0.150 / 0.068

BASES: Dict[str, dict] = {
    'claw-dynamics': dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True, pn_claw_to_apl=True, claw_dynamics=True,
        n_spikes_for_response=2,
    ),
    'claw': dict(one_row_per_claw=True, prat_claws=True, use_connectome_APL_weights=True,
        n_spikes_for_response=2,
    ),
    # the paper's hemibrain model (weight divisor 20, keeps '+' glomeruli)
    'hemibrain-wd20': dict(pn2kc_connections='hemibrain', weight_divisor=20,
        _drop_glom_with_plus=False,
    ),
}
TARGET_RATES = (0.05, 0.10)


def rate(responses: pd.DataFrame) -> float:
    return float(responses.astype(float).mean().mean())


def bisect_threshold(sim: Callable, thr0, target: float, *, extra: dict, tol: float = 0.02,
    max_iter: int = 16) -> Tuple[object, float, pd.DataFrame, pd.DataFrame]:
    """Finds a threshold scale giving mean response rate within `tol` (relative) of target.

    `thr0` is the tuned threshold (scalar or per-KC vector); candidates are `thr0 * k`.
    Rate decreases with k. Returns (threshold, k, responses, spike_counts).
    """
    thr0 = np.asarray(thr0, dtype=float)

    def run(k):
        r, s = sim({'fixed_thr': thr0 * k if thr0.ndim else float(thr0 * k), **extra})
        return rate(r), r, s

    lo, hi = 0.25, 4.0
    r_lo, *_ = run(lo)
    r_hi, *_ = run(hi)
    assert r_lo > target > r_hi, (r_lo, target, r_hi)
    best = None
    for _ in range(max_iter):
        k = np.sqrt(lo * hi)
        rk, r, s = run(k)
        if best is None or abs(rk - target) < abs(best[1] - target):
            best = (k, rk, r, s)
        if abs(rk - target) / target < tol:
            break
        if rk > target:
            lo = k
        else:
            hi = k
    k, rk, r, s = best
    return (thr0 * k if thr0.ndim else float(thr0 * k)), k, r, s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--quick', action='store_true', help='one base, one rate')
    args = ap.parse_args()

    from al_analysis.mb_model import make_fixed_param_kc_simulator

    res_dir = H.results_dir(EXPERIMENT)
    out_dir = H.outputs_dir(EXPERIMENT)
    (res_dir / 'matrices').mkdir(exist_ok=True)
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs()
    orn_input_pairs = H.orn_input_pairs(orn)
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()

    data = {
        'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')},
    }
    data_rates = {g: H.yang_response_fractions(g).mean() for g in ('TNT3c', 'TNT')}
    # which data target each model RDM variant is scored against
    variant_target = {'spike_counts_cosine': 'raw', 'responses_cosine': 'binarized',
        'spike_counts_pearson': 'pearson', 'spike_counts_pearson_nosilent': 'pearson',
        'responses_pearson': 'pearson', 'responses_pearson_nosilent': 'pearson',
    }

    bases = BASES if not args.quick else {'claw-dynamics': BASES['claw-dynamics']}
    rates = TARGET_RATES if not args.quick else (0.05,)
    log = []
    mvd_rows, onoff_rows, ref_rows, rate_rows, meta_rows = [], [], [], [], []
    saved_pairs = {}
    seed_check = {}

    for base, kws in bases.items():
        for target in rates:
            tag = f'{base}_sp{target:g}'
            t0 = time.time()
            sim, params = make_fixed_param_kc_simulator(orn, **kws, target_sparsity=target,
                make_plots=False, silent=True
            )
            thr0 = params['fixed_thr']
            conds = {}
            conds['on'] = sim({})
            conds['acute_off'] = sim({'wAPLKC_scale': 0.0})
            r_on = rate(conds['on'][0])
            thr_full, k_full, r_f, s_f = bisect_threshold(sim, thr0, r_on,
                extra={'wAPLKC_scale': 0.0}
            )
            conds['homeo_full_off'] = (r_f, s_f)
            thr_data, k_data, r_d, s_d = bisect_threshold(sim, thr0, r_on * DATA_RATE_RATIO,
                extra={'wAPLKC_scale': 0.0}
            )
            conds['homeo_data_off'] = (r_d, s_d)
            if base == 'claw-dynamics' and target == 0.05:
                # does the seed matter for connectome models? (expect: no)
                sim2, params2 = make_fixed_param_kc_simulator(orn, **kws,
                    target_sparsity=target, make_plots=False, silent=True, seed=1
                )
                r2, s2 = sim2({})
                seed_check = dict(identical_responses=bool(
                    conds['on'][0].to_numpy().astype(bool).__eq__(r2.to_numpy().astype(bool)).all()),
                    identical_spike_counts=bool(np.allclose(conds['on'][1].to_numpy(),
                        s2.to_numpy())),
                    thr_seed12345=float(np.mean(thr0)), thr_seed1=float(np.mean(params2['fixed_thr'])),
                )
                del sim2
            elapsed = time.time() - t0
            meta_rows.append(dict(base=base, target_rate=target, tag=tag,
                n_kcs=len(conds['on'][0]), fixed_thr_mean=float(np.mean(thr0)),
                fixed_thr_is_vector=bool(np.ndim(thr0)),
                wAPLKC_scale=float(params.get('wAPLKC_scale', np.nan)),
                wKCAPL_scale=float(params.get('wKCAPL_scale', np.nan)),
                thr_scale_homeo_full=k_full, thr_scale_homeo_data=k_data,
                seconds=elapsed, **{f'rate_{c}': rate(v[0]) for c, v in conds.items()},
            ))
            log.append(f'{tag}: {len(conds["on"][0])} KCs, thr {np.mean(thr0):.1f}, '
                f'rates ' + ', '.join(f'{c} {rate(v[0]):.4f}' for c, v in conds.items())
                + f'; {elapsed:.0f}s'
            )
            print(log[-1], flush=True)

            rdms = {}
            for cond, (resp, spk) in conds.items():
                resp.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
                spk.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
                rdms[cond] = H.model_rdms(resp, spk)
                fr = H.response_fractions(resp)
                for odor, v in fr.items():
                    rate_rows.append(dict(base=base, target_rate=target, condition=cond,
                        odor=odor, response_fraction=v
                    ))
                for variant, pairs in rdms[cond].items():
                    saved_pairs[(tag, cond, variant)] = pairs
                    if variant == H.PRIMARY_MODEL_RDM:
                        H.square_from_pairs(pairs).to_csv(
                            res_dir / 'matrices' / f'{tag}_{cond}_{variant}.csv'
                        )
                    geno = 'TNT3c' if cond == 'on' else 'TNT'
                    tgt = data[variant_target[variant]][geno]
                    st = H.compare_model_to_data(pairs, tgt, orn=orn_pairs)
                    # also against the other genotype, to see which the model is closer to
                    other = 'TNT' if geno == 'TNT3c' else 'TNT3c'
                    st_other = H.compare_model_to_data(pairs, data[variant_target[variant]][other])
                    mvd_rows.append(dict(base=base, target_rate=target, condition=cond,
                        variant=variant, data_genotype=geno, **st,
                        spearman_other_genotype=st_other['spearman'],
                        rmse_other_genotype=st_other['rmse'],
                    ))
                    if variant in ('spike_counts_cosine', 'responses_cosine',
                            'spike_counts_pearson'):
                        rr = H.compare_to_references(pairs, ref)
                        ref_rows.append(dict(base=base, target_rate=target, condition=cond,
                            variant=variant, **rr
                        ))

            for off_cond in ('acute_off', 'homeo_full_off', 'homeo_data_off'):
                for variant in ('spike_counts_cosine', 'responses_cosine',
                        'spike_counts_pearson'):
                    st, tabs = H.compare_on_off(rdms['on'][variant], rdms[off_cond][variant],
                        orn=orn_pairs, drive=drive
                    )
                    onoff_rows.append(dict(base=base, target_rate=target, off_condition=off_cond,
                        variant=variant, **st
                    ))
                    if variant == H.PRIMARY_MODEL_RDM:
                        tabs['pairs'].to_csv(res_dir / f'pairs_{tag}_on_vs_{off_cond}.csv')
                        tabs['blocks'].to_csv(res_dir / f'blocks_{tag}_on_vs_{off_cond}.csv')
                        tabs['per_odor'].to_csv(res_dir / f'per_odor_{tag}_on_vs_{off_cond}.csv')
            del sim

    # data-vs-data rows for reference, in the same tables
    for dvar, (mv, gon, goff) in {'raw': ('spike_counts_cosine', 'TNT3c', 'TNT'),
            'binarized': ('responses_cosine', 'TNT3c', 'TNT'),
            'pearson': ('spike_counts_pearson', 'TNT3c', 'TNT')}.items():
        st, _ = H.compare_on_off(data[dvar][gon], data[dvar][goff], orn=orn_pairs, drive=drive)
        onoff_rows.append(dict(base='DATA', target_rate=np.nan, off_condition='TNT',
            variant=mv, **st
        ))
        for g in (gon, goff):
            rr = H.compare_to_references(data[dvar][g], ref)
            ref_rows.append(dict(base='DATA', target_rate=np.nan, condition=g, variant=mv, **rr))
    for g in ('TNT3c', 'TNT'):
        for odor, v in data_rates[g].items():
            rate_rows.append(dict(base='DATA', target_rate=np.nan, condition=g, odor=odor,
                response_fraction=v
            ))

    pd.DataFrame(meta_rows).to_csv(res_dir / 'models.csv', index=False)
    pd.DataFrame(mvd_rows).to_csv(res_dir / 'stats_model_vs_data.csv', index=False)
    pd.DataFrame(onoff_rows).to_csv(res_dir / 'stats_on_vs_off.csv', index=False)
    pd.DataFrame(ref_rows).to_csv(res_dir / 'stats_references_66pairs.csv', index=False)
    pd.DataFrame(rate_rows).to_csv(res_dir / 'response_fractions_per_odor.csv', index=False)
    for (tag, cond, variant), pairs in saved_pairs.items():
        pass
    pairs_long = pd.concat({k: v for k, v in saved_pairs.items()}, names=['tag', 'condition',
        'variant', 'odor_a', 'odor_b']).rename('distance').reset_index()
    pairs_long.to_csv(res_dir / 'model_pair_distances_long.csv', index=False)
    orn_sanity = dict(spearman_orn_input_vs_consensus=H.spearman(orn_input_pairs, orn_pairs),
        pearson_orn_input_vs_consensus=H.pearson(orn_input_pairs, orn_pairs),
    )
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E1_apl_knockout.py', host=subprocess.run(['hostname'], capture_output=True,
            text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=H.olfsysm_dir(), capture_output=True, text=True).stdout.strip(),
        bases=bases, target_rates=list(rates), data_rate_ratio=DATA_RATE_RATIO,
        seed_check=seed_check, orn_sanity=orn_sanity, approx_conc_note=H.APPROX_CONC_NOTE,
        log=log,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')

    make_figures(res_dir, saved_pairs, data, drive, orn_pairs, meta_rows, rate_rows)
    print('\nORN sanity:', orn_sanity)
    print('seed check:', seed_check)
    print('done ->', res_dir)


def make_figures(res_dir: Path, saved_pairs: dict, data: dict, drive: pd.Series,
    orn_pairs: pd.Series, meta_rows: list, rate_rows: list) -> None:
    """One summary figure per base model / rate, plus a heatmap sheet."""
    plt = H._plt()
    tags = sorted({k[0] for k in saved_pairs})
    rates_df = pd.DataFrame(rate_rows)
    for tag in tags:
        on = saved_pairs[(tag, 'on', 'spike_counts_cosine')]
        fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
        for ax, off_cond in zip(axs[0], ('acute_off', 'homeo_full_off', 'homeo_data_off')):
            off = saved_pairs[(tag, off_cond, 'spike_counts_cosine')]
            H.plot_on_vs_off(ax, on, off, label=f'model ({off_cond})',
                data_on=data['raw']['TNT3c'], data_off=data['raw']['TNT']
            )
            st, _ = H.compare_on_off(on, off, orn=orn_pairs, drive=drive)
            ax.set_title(f'{tag}: on vs {off_cond}\nSpearman {st["spearman_on_off"]:.2f}, '
                f'iso R2 {st["iso_r2"]:.2f}, block {st["block_fruity_small_iso"]:+.3f}',
                fontsize=9
            )
        ax = axs[1, 0]
        H.plot_model_vs_data(ax, on, data['raw']['TNT3c'], label='APL on vs TNT3c',
            color=H.CONDITION_COLORS['APL on']
        )
        H.plot_model_vs_data(ax, saved_pairs[(tag, 'acute_off', 'spike_counts_cosine')],
            data['raw']['TNT'], label='acute off vs TNT', color=H.CONDITION_COLORS['APL off']
        )
        ax.set_title('model vs data (cosine over spike counts / z-scores)', fontsize=9)
        ax = axs[1, 1]
        _, tabs_m = H.compare_on_off(on, saved_pairs[(tag, 'acute_off', 'spike_counts_cosine')],
            orn=orn_pairs, drive=drive
        )
        _, tabs_d = H.compare_on_off(data['raw']['TNT3c'], data['raw']['TNT'], orn=orn_pairs,
            drive=drive
        )
        H.plot_per_odor_vs_drive(ax, tabs_d['per_odor'], color=H.CONDITION_COLORS['data'],
            label='data'
        )
        H.plot_per_odor_vs_drive(ax, tabs_m['per_odor'], color=H.CONDITION_COLORS['model'],
            label='model (acute off)'
        )
        ax.legend(fontsize=7, frameon=False)
        ax.set_title('per-odor APL effect vs ORN drive', fontsize=9)
        ax = axs[1, 2]
        r = rates_df[(rates_df.base == tag.split('_sp')[0]) &
            (rates_df.target_rate == float(tag.split('_sp')[1]))
        ]
        piv = r.pivot(index='odor', columns='condition', values='response_fraction')
        d = rates_df[rates_df.base == 'DATA'].pivot(index='odor', columns='condition',
            values='response_fraction'
        )
        piv = piv.loc[[o for o in H.APL16_ODORS if o in piv.index]]
        x = np.arange(len(piv))
        ax.plot(x, d.loc[piv.index, 'TNT3c'], 'o', color=H.CONDITION_COLORS['control'],
            label='data TNT3c', ms=4
        )
        ax.plot(x, d.loc[piv.index, 'TNT'], 'o', color=H.CONDITION_COLORS['TNT'],
            label='data TNT', ms=4
        )
        ax.plot(x, piv['on'], 's', mfc='none', color=H.CONDITION_COLORS['control'],
            label='model on', ms=5
        )
        ax.plot(x, piv['acute_off'], 's', mfc='none', color=H.CONDITION_COLORS['TNT'],
            label='model acute off', ms=5
        )
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=90, fontsize=7)
        ax.set_ylabel('fraction of KCs responding')
        ax.legend(fontsize=7, frameon=False)
        ax.set_title('per-odor response fractions (model: >= n spikes; data: > 1)', fontsize=9)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        fig.suptitle(f'E1 {tag}. {H.APPROX_CONC_NOTE[:120]}...', fontsize=8)
        fig.tight_layout()
        fig.savefig(res_dir / f'E1_{tag}_summary.png', dpi=130)
        plt.close(fig)

    # heatmaps: data and the claw-dynamics 5% model
    fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    H.plot_square(axs[0, 0], data['raw']['TNT3c'], 'data: control (TNT3c)')
    H.plot_square(axs[0, 1], data['raw']['TNT'], 'data: TNT (APL blocked)')
    H.plot_square(axs[0, 2], data['raw']['TNT3c'] - data['raw']['TNT'], 'data: control - TNT',
        vmin=-0.4, vmax=0.4, cmap='RdBu_r', cbar_label='difference'
    )
    t = 'claw-dynamics_sp0.05' if 'claw-dynamics_sp0.05' in tags else tags[0]
    on = saved_pairs[(t, 'on', 'spike_counts_cosine')]
    off = saved_pairs[(t, 'acute_off', 'spike_counts_cosine')]
    H.plot_square(axs[1, 0], on, f'model {t}: APL on')
    H.plot_square(axs[1, 1], off, f'model {t}: acute APL off')
    H.plot_square(axs[1, 2], on - off, 'model: on - off', vmin=-0.4, vmax=0.4, cmap='RdBu_r',
        cbar_label='difference'
    )
    fig.tight_layout()
    fig.savefig(res_dir / 'E1_heatmaps.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
