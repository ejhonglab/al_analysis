#!/usr/bin/env python3
"""E2: local APL (coupled APL units) on the APL16 panel, scored against Yang's data.

Runs `apl_coupling_sweep.run_base` (the 'fixed' and 'matched' sweeps over
`apl_unit_coupling`) on the 16-odor APL16 input for each base model, then scores every
matched model (APL split into spatial units, APL weights scaled so the number of KC-odor
responses equals the single-APL model's) with `kc_remapping_harness`:
- the on/off relation against the acute APL-off model (same thresholds, wAPLKC_scale=0):
  isotonic R2, fruity x small-ester block residual, per-odor APL effect vs ORN drive
- agreement with Yang's control matrix, and the 66-pair references (claws, VCF)
- how much coupling changes the APL-on pair structure (Spearman vs the single-APL model)

Prediction (mechanism 2): weaker coupling (more local inhibition) separates groups that
share glomerular input (fruity vs small esters) more than groups that do not, giving a
positive block residual and a stronger per-odor drive dependence, up to the point where
self-inhibition dominates.

Outputs: run_base's outputs regularized to the archive layout under ./outputs/E2_local_apl/
(`<base>_<variant>_<cond>_{responses,spike_counts}.parquet`, `<base>_sweep.csv`, merged `sweep.csv`;
bases whose runs exist are skipped on re-run);
stats, figures and provenance to <shared>/results/E2_local_apl/.

Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E2_local_apl.py [-b BASE ...]
    [-k COUPLING ...] [-d N_DYNAMICS_ODORS]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H

sys.path.insert(0, str(H.REPO / 'scripts'))
import apl_coupling_sweep as ACS  # noqa: E402

EXPERIMENT = 'E2_local_apl'

BASES = {
    'claw-dynamics_5pct': (dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True, pn_claw_to_apl=True, claw_dynamics=True,
        n_spikes_for_response=2), 0.05),
    'claw_5pct': (dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True, n_spikes_for_response=2), 0.05),
    'claw-dynamics_10pct': (dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True, pn_claw_to_apl=True, claw_dynamics=True,
        n_spikes_for_response=2), 0.10),
}
DEFAULT_COUPLINGS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0]


def yang_loss_fn():
    """`make_kc_data_loss` on Yang's control (TNT3c) per-odor rates and mean correlations."""
    from al_analysis.mb_model import make_kc_data_loss
    rates = H.yang_response_fractions('TNT3c').mean()
    rates = rates[[o for o in H.APL16_ODORS]]
    corr = 1 - H.square_from_pairs(H.yang_mean_rdm('TNT3c', 'cosine'))
    corr = corr.loc[H.APL16_ODORS, H.APL16_ODORS]
    return make_kc_data_loss(target_response_rates=rates, target_corr=corr)


def flat_name(base: str, mode: str, coupling, cond: str = 'on') -> str:
    """Variant name in the archive layout: `<base>_single`, `<base>_fixed_coupling<c>` or
    `<base>_matched_coupling<c>` (file: `<variant>_<cond>_{responses,spike_counts}.parquet`)."""
    variant = 'single' if mode == 'single' else f'{mode}_coupling{coupling:g}'
    return f'{base}_{variant}_{cond}'


def regularize_layout(out_root: Path, base: str) -> None:
    """Moves `apl_coupling_sweep.run_base`'s outputs for `base` (`<base>/responses/
    {responses,spike_counts}_<name>.parquet`, `<base>/sweep.csv`) into the flat archive
    layout `<base>_<variant>_<cond>_<kind>.parquet` and `<base>_sweep.csv`, and merges the
    per-base sweep tables into `sweep.csv` (tag column = variant name). No-op once done."""
    import re
    import shutil
    bdir = out_root / base
    rdir = bdir / 'responses'
    if rdir.exists():
        for f in sorted(rdir.glob('*.parquet')):
            m = re.fullmatch(r'(responses|spike_counts)_(.+)\.parquet', f.name)
            kind, name = m.group(1), m.group(2)
            if name == 'acute_off':
                target = f'{flat_name(base, "single", None, "acute_off")}_{kind}.parquet'
            elif name == 'single':
                target = f'{flat_name(base, "single", None)}_{kind}.parquet'
            else:
                mm = re.fullmatch(r'(fixed|matched)_coupling_([0-9.]+)', name)
                target = f'{flat_name(base, mm.group(1), float(mm.group(2)))}_{kind}.parquet'
            shutil.move(str(f), str(out_root / target))
        rdir.rmdir()
    if (bdir / 'sweep.csv').exists():
        shutil.move(str(bdir / 'sweep.csv'), str(out_root / f'{base}_sweep.csv'))
    if bdir.exists() and not any(bdir.iterdir()):
        bdir.rmdir()
    sweeps = []
    for sp in sorted(out_root.glob('*_sweep.csv')):
        sw = pd.read_csv(sp)
        if 'tag' not in sw:
            sw.insert(0, 'tag', [flat_name(r['base'], r['mode'], r['coupling'])[:-3] for _, r in sw.iterrows()])
        sweeps.append(sw)
    if sweeps:
        pd.concat(sweeps, ignore_index=True).to_csv(out_root / 'sweep.csv', index=False)


def load_run(out_root: Path, base: str, mode: str, coupling=None, cond: str = 'on'):
    stem = flat_name(base, mode, coupling, cond)
    return (pd.read_parquet(out_root / f'{stem}_responses.parquet'),
        pd.read_parquet(out_root / f'{stem}_spike_counts.parquet'))


def score_base(base: str, orn: pd.DataFrame, couplings, out_root: Path, res_dir: Path,
    data: dict, orn_pairs: pd.Series, drive: pd.Series, ref: pd.DataFrame) -> pd.DataFrame:
    """Scores the saved runs of one base; returns one row per (mode, coupling)."""
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    kws, _ = BASES[base]
    regularize_layout(out_root, base)
    sweep = pd.read_csv(out_root / f'{base}_sweep.csv')
    single = sweep[sweep['mode'] == 'single'].iloc[0]
    # acute APL off with the single-APL model's thresholds
    initial = {k: single[k] for k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale')
        if k in single and pd.notna(single[k])
    }
    sim, _ = make_fixed_param_kc_simulator(orn, initial_params=initial, **kws,
        make_plots=False, silent=True
    )
    off_resp, off_spk = sim({'wAPLKC_scale': 0.0})
    del sim
    off_stem = flat_name(base, 'single', None, 'acute_off')
    off_resp.to_parquet(out_root / f'{off_stem}_responses.parquet')
    off_spk.to_parquet(out_root / f'{off_stem}_spike_counts.parquet')
    off = H.model_rdms(off_resp, off_spk)
    single_on = H.model_rdms(*load_run(out_root, base, 'single'))

    rows = []
    pair_rows = {}
    runs = [('single', None)] + [(m, c) for c in couplings for m in ('fixed', 'matched')]
    for mode, coupling in runs:
        name = mode if coupling is None else f'{mode}_coupling_{coupling:g}'
        resp, spk = load_run(out_root, base, mode, coupling)
        on = H.model_rdms(resp, spk)
        row = dict(base=base, mode=mode, coupling=coupling,
            response_rate=float(resp.astype(float).mean().mean()),
            rate_off=float(off_resp.astype(float).mean().mean()),
        )
        for variant in ('spike_counts_cosine', 'responses_cosine', 'spike_counts_pearson'):
            st, tabs = H.compare_on_off(on[variant], off[variant], orn=orn_pairs, drive=drive)
            for k in ('mean_on', 'mean_off', 'spearman_on_off', 'lin_r2', 'iso_r2',
                    'block_fruity_small_iso', 'block_fruity_small_lin',
                    'block_other_between_iso', 'spearman_orn_on', 'spearman_orn_off',
                    'gap_closed', 'spearman_per_odor_iso_vs_drive',
                    'spearman_per_odor_delta_vs_drive'):
                row[f'{variant}__{k}'] = st[k]
            tgt = data['binarized' if variant == 'responses_cosine' else
                ('pearson' if variant.endswith('pearson') else 'raw')]
            mvd = H.compare_model_to_data(on[variant], tgt['TNT3c'], orn=orn_pairs)
            mvd_off = H.compare_model_to_data(off[variant], tgt['TNT'], orn=orn_pairs)
            row[f'{variant}__spearman_vs_ctrl'] = mvd['spearman']
            row[f'{variant}__spearman_dev_vs_ctrl'] = mvd['spearman_dev']
            row[f'{variant}__rmse_vs_ctrl'] = mvd['rmse']
            row[f'{variant}__off_spearman_vs_tnt'] = mvd_off['spearman']
            row[f'{variant}__spearman_vs_single'] = H.spearman(on[variant], single_on[variant])
            rr = H.compare_to_references(on[variant], ref)
            for k in ('spearman_kc_claw', 'spearman_kc_soma', 'spearman_vcf_hyp',
                    'spearman_resid_vs_vcf_hyp', 'spearman_resid_vs_vcf_raw',
                    'spearman_resid_vs_remap_wt', 'spearman_resid_vs_remap_ctrl'):
                row[f'{variant}__{k}'] = rr[k]
            if variant == H.PRIMARY_MODEL_RDM:
                pair_rows[(base, name)] = on[variant]
                tabs['blocks'].to_csv(res_dir / f'blocks_{base}_{name}.csv')
                tabs['per_odor'].to_csv(res_dir / f'per_odor_{base}_{name}.csv')
        rows.append(row)
        print(f'{base} {name}: rate {row["response_rate"]:.4f}, iso R2 '
            f'{row["spike_counts_cosine__iso_r2"]:.3f}, block '
            f'{row["spike_counts_cosine__block_fruity_small_iso"]:+.3f}, per-odor vs drive '
            f'{row["spike_counts_cosine__spearman_per_odor_iso_vs_drive"]:.2f}, vs ctrl '
            f'{row["spike_counts_cosine__spearman_vs_ctrl"]:.3f}', flush=True
        )
    pairs_long = pd.concat(pair_rows, names=['base', 'run', 'odor_a', 'odor_b']
        ).rename('distance').reset_index()
    pairs_long.to_csv(res_dir / f'model_pair_distances_{base}.csv', index=False)
    return pd.DataFrame(rows)


def make_figures(stats: pd.DataFrame, res_dir: Path, data_stats: dict) -> None:
    plt = H._plt()
    v = 'spike_counts_cosine'
    metrics = [('iso_r2', 'isotonic R2 (on from off)'),
        ('block_fruity_small_iso', 'fruity x small-ester block residual'),
        ('spearman_per_odor_iso_vs_drive', 'per-odor APL effect vs ORN drive (Spearman)'),
        ('spearman_vs_ctrl', 'Spearman with control data (TNT3c)'),
        ('spearman_resid_vs_vcf_hyp', 'ORN-residual vs VCF (hyperbolic), 66 pairs'),
        ('spearman_vs_single', 'Spearman with single-APL model'),
    ]
    bases = list(stats.base.unique())
    colors = ['#2a78d6', '#c8553d', '#4f9d69', '#8c6bb1']
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (m, title) in zip(axs.flat, metrics):
        col = f'{v}__{m}'
        for i, base in enumerate(bases):
            sub = stats[(stats.base == base) & (stats['mode'] == 'matched')].sort_values('coupling')
            x = sub.coupling.replace(0, 0.03)  # log axis; 0 drawn at the left
            ax.plot(x, sub[col], 'o-', color=colors[i % len(colors)], ms=4, lw=1.5,
                label=f'{base} matched'
            )
            single = stats[(stats.base == base) & (stats['mode'] == 'single')]
            if len(single):
                ax.axhline(single[col].iloc[0], color=colors[i % len(colors)], lw=0.8,
                    ls=':', label=f'{base} single APL'
                )
        if m in data_stats:
            ax.axhline(data_stats[m], color='#999999', lw=1.2, ls='--', label='data')
        ax.set_xscale('log')
        ax.set_xlabel('apl_unit_coupling (0 drawn at 0.03)')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, frameon=False)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.suptitle('E2: APL units, response counts matched across couplings. '
        + H.APPROX_CONC_NOTE[:110] + '...', fontsize=8
    )
    fig.tight_layout()
    fig.savefig(res_dir / 'E2_metrics_vs_coupling.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-b', '--bases', nargs='+', choices=list(BASES), default=list(BASES))
    ap.add_argument('-k', '--couplings', nargs='+', type=float, default=DEFAULT_COUPLINGS)
    ap.add_argument('-d', '--n-dynamics-odors', type=int, default=0,
        help='odors to compute APL unit potentials on (0 skips; slow)'
    )
    ap.add_argument('--score-only', action='store_true', help='skip simulations')
    args = ap.parse_args()

    res_dir = H.results_dir(EXPERIMENT)
    out_root = H.outputs_dir(EXPERIMENT)
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {
        'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')},
    }
    ACS.BASES = BASES
    loss_fn = yang_loss_fn()
    if not args.score_only:
        for base in args.bases:
            if (out_root / f'{flat_name(base, "single", None)}_responses.parquet').exists():
                print(f'=== {base}: runs already saved (flat layout); skipping simulations', flush=True)
                continue
            print(f'=== {base}', flush=True)
            # run_base writes <base>/responses/... and <base>/sweep.csv; regularized below
            ACS.run_base(base, orn, loss_fn, args.couplings, out_root,
                n_dynamics_odors=args.n_dynamics_odors
            )
            regularize_layout(out_root, base)
    stats = pd.concat([score_base(b, orn, args.couplings, out_root, res_dir, data, orn_pairs,
        drive, ref) for b in args.bases], ignore_index=True)
    stats.to_csv(res_dir / 'stats_vs_coupling.csv', index=False)
    dst, _ = H.compare_on_off(data['raw']['TNT3c'], data['raw']['TNT'], orn=orn_pairs,
        drive=drive
    )
    dref = H.compare_to_references(data['raw']['TNT3c'], ref)
    data_stats = {k: dst[k] for k in ('iso_r2', 'block_fruity_small_iso',
        'spearman_per_odor_iso_vs_drive')}
    data_stats['spearman_resid_vs_vcf_hyp'] = dref['spearman_resid_vs_vcf_hyp']
    make_figures(stats, res_dir, data_stats)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E2_local_apl.py', host=subprocess.run(['hostname'], capture_output=True,
            text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        bases={b: BASES[b] for b in args.bases}, couplings=args.couplings,
        n_dynamics_odors=args.n_dynamics_odors, data_stats=data_stats,
        approx_conc_note=H.APPROX_CONC_NOTE,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
