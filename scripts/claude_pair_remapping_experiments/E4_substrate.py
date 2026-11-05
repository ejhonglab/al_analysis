#!/usr/bin/env python3
"""E4: substrate structure. PN->KC connectivity variants with APL on and off, on APL16.

For each connectivity variant (uniform and hemidraw random draws over seeds; hemibrain
with weight divisors None / 10 / 20 / 40; hemibrain claws (Pratyush's segmentation) with
and without claw dynamics and with distance weighting; FAFB left / right), the model is
tuned with APL on to the target response rate, then re-run with the same thresholds and
APL output removed (acute knockout). Shuffled hemibrain matrices (degree-preserving,
KC-type-preserving, glomerulus-frequency-preserving) are added through `--shuffles` once
`al_analysis.wpnkc_shuffles` exists (atlas is writing it).

Readouts (kc_remapping_harness): for APL on vs Yang's control and APL off vs TNT, the
Spearman between model and measured deviations from the linear ORN prediction
(`spearman_dev`, 120 pairs; the report's 0.60 for wd20 was on the 66 megamat pairs vs
Remy's wild-type somata, reported here as `spearman_resid_vs_remap_wt`), plain Spearman,
the on/off relation (isotonic R2, fruity x small-ester block, per-odor effect vs drive),
and a linear "claw input" readout (wPNKC @ ORN input, no threshold) against the 66-pair
KC-claw distances.

Outputs: stats to <shared>/results/E4_substrate/, responses / spike counts / wPNKC to
./outputs/E4_substrate/.

Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E4_substrate.py [-v VARIANT ...]
    [--seeds N] [--rates 0.05 0.1] [--shuffles ...]
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

import kc_remapping_harness as H

EXPERIMENT = 'E4_substrate'
UNIFORM_N_CLAWS = 7

CLAWS = dict(one_row_per_claw=True, prat_claws=True, use_connectome_APL_weights=True,
    n_spikes_for_response=2
)
# name -> (fit_mb_model kwargs, uses random seed)
VARIANTS: Dict[str, tuple] = {
    # n_claws: hemibrain claws per KC average 5.7 (Pratyush's segmentation) and 6-7 in
    # the literature; 7 as in the lab's uniform model
    'uniform': (dict(pn2kc_connections='uniform', n_claws=UNIFORM_N_CLAWS), True),
    'hemidraw': (dict(pn2kc_connections='hemidraw', n_claws=UNIFORM_N_CLAWS), True),
    'hemibrain-wdNone': (dict(pn2kc_connections='hemibrain', weight_divisor=None,
        _drop_glom_with_plus=False), False),
    'hemibrain-wd10': (dict(pn2kc_connections='hemibrain', weight_divisor=10,
        _drop_glom_with_plus=False), False),
    'hemibrain-wd20': (dict(pn2kc_connections='hemibrain', weight_divisor=20,
        _drop_glom_with_plus=False), False),
    'hemibrain-wd40': (dict(pn2kc_connections='hemibrain', weight_divisor=40,
        _drop_glom_with_plus=False), False),
    'hemibrain-claws': (dict(CLAWS), False),
    'hemibrain-claws-dyn': (dict(CLAWS, pn_claw_to_apl=True, claw_dynamics=True), False),
    'hemibrain-claws-distw-pct': (dict(CLAWS, dist_weight='percentile'), False),
    'hemibrain-claws-distw-raw': (dict(CLAWS, dist_weight='raw'), False),
    'fafb-left-wd12': (dict(pn2kc_connections='fafb-left', weight_divisor=12), False),
    'fafb-right-wd12': (dict(pn2kc_connections='fafb-right', weight_divisor=12), False),
    # APL weight type crossed with wiring: do heterogeneous per-claw connectome APL
    # weights (used by the claw models) remove pair structure the wd models keep?
    'hemibrain-wdNone-cAPL': (dict(pn2kc_connections='hemibrain', weight_divisor=None,
        _drop_glom_with_plus=False, use_connectome_APL_weights=True), False),
    'hemibrain-wd20-cAPL': (dict(pn2kc_connections='hemibrain', weight_divisor=20,
        _drop_glom_with_plus=False, use_connectome_APL_weights=True), False),
    'hemibrain-claws-uAPL': (dict(CLAWS, use_connectome_APL_weights=False), False),
    'hemibrain-claws-1spk': (dict(CLAWS, n_spikes_for_response=None), False),
}


def rate(responses: pd.DataFrame) -> float:
    return float(responses.astype(float).mean().mean())


def kc_by_glomerulus(wPNKC: pd.DataFrame) -> pd.DataFrame:
    """Sums a (claw x glomerulus) matrix to (KC x glomerulus); passes KC x glomerulus through."""
    if H.KC_ID_LEVEL in (wPNKC.index.names or []) and len(wPNKC.index.names) > 1:
        return wPNKC.groupby(level=H.KC_ID_LEVEL).sum()
    return wPNKC


def claw_input_pairs(wPNKC: pd.DataFrame, orn: pd.DataFrame) -> pd.Series:
    """1 - Pearson between odors of the linear claw input wPNKC @ ORN deltas (across KCs)."""
    w = kc_by_glomerulus(wPNKC)
    orn_a = H.abbrev_columns(orn)
    gl = [g for g in w.columns if g in orn_a.index]
    x = w[gl].to_numpy(dtype=float) @ orn_a.loc[gl].to_numpy(dtype=float)
    mat = pd.DataFrame(x, columns=orn_a.columns)
    mat = mat.loc[(mat != 0).any(axis='columns')]
    return H.pair_distances(mat, 'pearson')


def run_variant(name: str, kws: dict, seed: Optional[int], target: float, orn: pd.DataFrame,
    out_dir: Path, wPNKC_override: Optional[pd.DataFrame] = None):
    """Tunes APL on, then acute APL off with the same thresholds. Saves outputs; returns them."""
    from al_analysis.mb_model import fit_mb_model
    common = dict(kws, make_plots=False, silent=True)
    if seed is not None:
        common['seed'] = seed
    if wPNKC_override is not None:
        common['_wPNKC'] = wPNKC_override
    tag = f'{name}_sp{target:g}' + (f'_seed{seed}' if seed is not None else '')
    t0 = time.time()
    resp_on, spk_on, wPNKC, params = fit_mb_model(orn, target_sparsity=target, **common)
    thr = params['fixed_thr']
    resp_off, spk_off, _, params_off = fit_mb_model(orn, fixed_thr=thr, wAPLKC=0.0, **common)
    for cond, (r, s) in (('on', (resp_on, spk_on)), ('acute_off', (resp_off, spk_off))):
        r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
        s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
    wPNKC.to_parquet(out_dir / f'{tag}_wPNKC.parquet')
    meta = dict(variant=name, seed=seed, target_rate=target, tag=tag, n_kcs=len(resp_on),
        n_glomeruli=wPNKC.shape[1], fixed_thr_mean=float(np.mean(thr)),
        wAPLKC=float(np.mean(params.get('wAPLKC_scale', params.get('wAPLKC', np.nan)))),
        rate_on=rate(resp_on), rate_off=rate(resp_off), seconds=time.time() - t0,
        mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
    )
    print(f'{tag}: {meta["n_kcs"]} KCs, rates on {meta["rate_on"]:.4f} off '
        f'{meta["rate_off"]:.4f}; {meta["seconds"]:.0f}s', flush=True
    )
    return meta, (resp_on, spk_on), (resp_off, spk_off), wPNKC


def score(meta: dict, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref) -> dict:
    row = dict(meta)
    on = H.model_rdms(*on_out)
    off = H.model_rdms(*off_out)
    for variant in ('spike_counts_cosine', 'responses_cosine', 'spike_counts_pearson'):
        tgt = data['binarized' if variant == 'responses_cosine' else
            ('pearson' if variant.endswith('pearson') else 'raw')]
        for cond, rd, geno in (('on', on, 'TNT3c'), ('off', off, 'TNT')):
            m = H.compare_model_to_data(rd[variant], tgt[geno], orn=orn_pairs)
            for k in ('spearman', 'pearson', 'rmse', 'spearman_dev', 'dev_sd_model',
                    'mean_model'):
                row[f'{variant}__{cond}_{k}'] = m[k]
            rr = H.compare_to_references(rd[variant], ref)
            for k in ('spearman_orn', 'spearman_kc_claw', 'spearman_kc_soma',
                    'spearman_vcf_hyp', 'spearman_resid_vs_vcf_hyp',
                    'spearman_resid_vs_remap_wt', 'spearman_resid_vs_remap_ctrl',
                    'spearman_resid_vs_remap_tnt', 'resid_orn_sd'):
                row[f'{variant}__{cond}_{k}'] = rr[k]
        st, _ = H.compare_on_off(on[variant], off[variant], orn=orn_pairs, drive=drive)
        for k in ('spearman_on_off', 'iso_r2', 'block_fruity_small_iso',
                'spearman_per_odor_iso_vs_drive', 'spearman_orn_on', 'spearman_orn_off',
                'gap_closed'):
            row[f'{variant}__{k}'] = st[k]
    ci = claw_input_pairs(wPNKC, orn)
    rr = H.compare_to_references(ci, ref)
    row['claw_input__spearman_kc_claw'] = rr['spearman_kc_claw']
    row['claw_input__spearman_orn'] = rr['spearman_orn']
    row['claw_input__spearman_resid_vs_remap_wt'] = rr['spearman_resid_vs_remap_wt']
    row['claw_input__spearman_vs_ctrl'] = H.spearman(ci, data['raw']['TNT3c'])
    return row, on, off


def make_figure(stats: pd.DataFrame, res_dir: Path, data_stats: dict) -> None:
    plt = H._plt()
    v = 'spike_counts_cosine'
    panels = [
        (f'{v}__on_spearman_dev', 'APL on: Spearman of deviations from ORN, model vs control data (120 pairs)'),
        (f'{v}__off_spearman_dev', 'APL off: same, vs TNT data'),
        (f'{v}__on_spearman_resid_vs_remap_wt', 'APL on: deviations vs Remy WT somata (66 pairs)'),
        (f'{v}__on_spearman', 'APL on: Spearman with control data'),
        (f'{v}__block_fruity_small_iso', 'on vs off: fruity x small-ester block residual'),
        (f'{v}__iso_r2', 'on vs off: isotonic R2'),
        (f'{v}__spearman_per_odor_iso_vs_drive', 'on vs off: per-odor effect vs ORN drive'),
        ('claw_input__spearman_kc_claw', 'linear claw input vs KC-claw data (66 pairs)'),
    ]
    variants = list(dict.fromkeys(stats.variant))
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (col, title) in zip(axs.flat, panels):
        for i, var in enumerate(variants):
            sub = stats[stats.variant == var]
            for rate_, mk in zip(sorted(sub.target_rate.unique()), ('o', 's', '^')):
                s2 = sub[sub.target_rate == rate_]
                ax.scatter([i] * len(s2), s2[col], marker=mk, s=22, color='#2a78d6',
                    alpha=0.8, edgecolors='none', label=f'rate {rate_:g}' if i == 0 else None
                )
        key = col.split('__')[-1]
        if key in data_stats:
            ax.axhline(data_stats[key], color='#999999', ls='--', lw=1, label='data')
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=90, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.axhline(0, color='#dddddd', lw=0.8)
        ax.legend(fontsize=6, frameon=False)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle('E4: connectivity variants (APL16 input; ' + H.APPROX_CONC_NOTE[:90] + '...)',
        fontsize=8
    )
    fig.tight_layout()
    fig.savefig(res_dir / 'E4_variants.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-v', '--variants', nargs='+', choices=list(VARIANTS),
        default=list(VARIANTS)
    )
    ap.add_argument('--seeds', type=int, default=3, help='seeds for random variants')
    ap.add_argument('--rates', nargs='+', type=float, default=[0.05])
    ap.add_argument('--shuffles', nargs='*', default=[],
        help='shuffle names from al_analysis.wpnkc_shuffles (applied to hemibrain-claws)'
    )
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
    stats_path = res_dir / 'stats_variants.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    # failed runs (rows with an error) are retried
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}
    pairs = {}

    jobs = []
    for name in args.variants:
        kws, random = VARIANTS[name]
        seeds = list(range(args.seeds)) if random else [None]
        for target in args.rates:
            for seed in seeds:
                jobs.append((name, kws, seed, target, None))
    for shuf in args.shuffles:
        import al_analysis.wpnkc_shuffles as WS  # provided by atlas
        base_kws, _ = VARIANTS['hemibrain-claws']
        for target in args.rates:
            for seed in range(args.seeds):
                jobs.append((f'shuffle-{shuf}', base_kws, seed, target, (WS, shuf)))

    for name, kws, seed, target, shuffle in jobs:
        tag = f'{name}_sp{target:g}' + (f'_seed{seed}' if seed is not None else '')
        if tag in done:
            continue
        wPNKC_override = None
        if shuffle is not None:
            WS, shuf = shuffle
            wPNKC_override = WS.shuffled_wPNKC(shuf, seed=seed, **kws)
        try:
            meta, on_out, off_out, wPNKC = run_variant(name, kws, seed, target, orn, out_dir,
                wPNKC_override
            )
        except Exception as e:  # keep going; record the failure
            print(f'{tag} FAILED: {e!r}', flush=True)
            rows.append(dict(variant=name, seed=seed, target_rate=target, tag=tag,
                error=repr(e)
            ))
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            continue
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
        rows.append(row)
        pairs[(tag, 'on')] = on[H.PRIMARY_MODEL_RDM]
        pairs[(tag, 'acute_off')] = off[H.PRIMARY_MODEL_RDM]
        pd.DataFrame(rows).to_csv(stats_path, index=False)

    stats = pd.DataFrame(rows)
    if pairs:
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
            ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
            mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
    dst, _ = H.compare_on_off(data['raw']['TNT3c'], data['raw']['TNT'], orn=orn_pairs,
        drive=drive
    )
    dref = H.compare_to_references(data['raw']['TNT3c'], ref)
    data_stats = {k: dst[k] for k in ('iso_r2', 'block_fruity_small_iso',
        'spearman_per_odor_iso_vs_drive')}
    data_stats['spearman_resid_vs_remap_wt'] = dref['spearman_resid_vs_remap_wt']
    data_stats['spearman_kc_claw'] = dref['spearman_kc_claw']
    make_figure(stats[stats.get('error', pd.Series(dtype=object)).isna()] if 'error' in stats
        else stats, res_dir, data_stats)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E4_substrate.py', host=subprocess.run(['hostname'], capture_output=True,
            text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        variants={n: VARIANTS[n][0] for n in args.variants}, seeds=args.seeds,
        rates=args.rates, shuffles=args.shuffles, data_stats=data_stats,
        approx_conc_note=H.APPROX_CONC_NOTE,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
