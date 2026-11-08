#!/usr/bin/env python3
"""E47: which feature of the hemibrain claw wiring carries the fruity x small-ester block.

Every re-drawn wiring loses most of the block the real hemibrain claw wiring makes (the
within-KC-type claw permutations of E30, E33, E42 and E46), so the block depends on
something a claw permutation destroys. The four shuffles in `al_analysis.wpnkc_shuffles`
keep different things: `claw_permutation` keeps each KC's number of claws and each
glomerulus' number of claws; `curveball` keeps each KC's number of distinct glomeruli and
each glomerulus' number of KCs (the degree structure of the binary KC x glomerulus
matrix); the `_within_type` variants keep those per KC type as well. All four destroy
which glomeruli share a KC beyond what those margins imply.

Part A (no fit): the reference models at the control flies' 6.8 % (E44's eight-glomerulus
gain of 1.5 at 70 and 40 Hz, E45's eight at 1.25 at 70 Hz) at their fitted APL and KC-KC
scales, with only the somatic threshold re-tuned to 6.8 % (bisection, as E44 part C), on
the real wiring (also at the fitted threshold, which must reproduce the fit) and on seeds
1-5 of each shuffle (seed 2 of `claw_permutation_within_type` is E30's draw). Each variant
is scored as E44 (E4's statistics, E7's excess profile, per-odor response fractions, E21's
sharing deficit, E35's suppression uniformity, the weak odors' fractions, the structure
loss at both rate targets), and each draw gets `wPNKC_shuffle_summary`'s statistics
(claws changed, degrees preserved, co-input correlation with the real wiring).
Part B (fits) runs through `E10_joint_fit.py --experiment E47_wiring_features --wiring
curveball[_within_type] --wiring-seed 2 --rate-target ctrl_frac ...` (outputs/E47_wiring_features/run.sh
on the host that ran them); its rows land in this experiment's fits.csv and --summary puts
them beside E46's claw-permutation refits and the real-wiring fits.

Outputs: <shared>/results/E47_wiring_features/ (stats_variants.csv, shuffle_summary.csv,
per_odor_response_fractions.csv, comparison.csv, comparison_table.md, provenance.json,
E47_wiring_features.png) and ./outputs/E47_wiring_features/ (responses / spike counts per
variant, tuned_params.json; partA/ holds the bases' cache files).
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E47_wiring_features.py [--models ...] [--shuffles ...] [--seeds 1 2 3 4 5]
       ./venv/bin/python E47_wiring_features.py --summary   (tables and figure only)
       ./venv/bin/python E47_wiring_features.py --local-test DIR --models thr70_gain1.5 --shuffles claw_permutation --seeds 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E1_apl_knockout import bisect_threshold
from E3_drive_dependent_gain import per_odor_extras, rate
from E4_substrate import score, kc_by_glomerulus
from E10_joint_fit import TAU, base_kws, bouton_base, breadth_weights, make_loss
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E26_claw_power import sharing_deficit
from E35_divisive_apl import suppression_stats
from E44_operating_point import fraction_stats, losses

EXPERIMENT = 'E47_wiring_features'
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
SCALE_KEYS = ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')
CTRL_FRAC = H.data_target('ctrl_frac')
# model key -> (experiment, tag in its fits.csv, per-claw threshold, breadth exponent, gain)
MODELS: Dict[str, Tuple[str, str, float, float, float]] = {
    'thr70_gain1.5': ('E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3', 70.0, 2.0, 1.5),
    'thr40_gain1.5': ('E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3', 40.0, 2.0, 1.5),
    'thr70_gain1.25': ('E45_gain_at_ctrl_frac', 'thr70_gain1.25_rate0.068_k2_cap3', 70.0, 2.0, 1.25),
}
SHUFFLES = ('claw_permutation', 'claw_permutation_within_type', 'curveball', 'curveball_within_type')
CLASS_LABEL = {'real': 'real wiring', 'claw_permutation': 'claw permutation',
    'claw_permutation_within_type': 'claw permutation within type', 'curveball': 'Curveball',
    'curveball_within_type': 'Curveball within type'}
V = H.PRIMARY_MODEL_RDM
# statistics in the summary table and figure: column in stats_variants.csv, label, format, data-target key
STATS = [
    (f'{V}__block_fruity_small_iso', 'fruity × small-ester block', '+.3f', 'block'),
    ('loss_rate_ctrl', 'extended loss at 6.8 % (lower is better)', '.2f', None),
    (f'{V}__on_spearman', 'ρ on odor-odor distances, APL on', '.2f', 'ctrl_ceiling'),
    (f'{V}__off_spearman', 'ρ on odor-odor distances, APL off', '.2f', 'tnt_ceiling'),
    ('spearman_rate_on_vs_data', 'response rate ρ, APL on', '.2f', 'ctrl_rate_rho_ceiling'),
    ('spearman_rate_off_vs_data', 'response rate ρ, APL off', '.2f', 'tnt_rate_rho_ceiling'),
    (f'{V}__on_spearman_resid_vs_remap_wt', 'structure vs wild-type somata', '.2f', 'ctrl_struct_wt'),
    (f'{V}__off_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL off', '.2f', 'tnt_vcf'),
    ('rate_ratio_off_over_on', 'response rate, APL off / APL on', '.2f', 'rate_ratio'),
    ('weak_share_off', "weak odors' share of the APL-off response rate", '.2f', 'tnt_weak_share'),
    ('thr_factor_vs_fit', 'somatic threshold (× the fitted one)', '.2f', None),
]


def fitted_params(exp: str, tag: str) -> Tuple[Dict[str, float], pd.Series]:
    """The fitted parameters of one row of an experiment's fits.csv (absolute values), and
    the row."""
    fits = pd.read_csv(H.shared_dir() / 'results' / exp / 'fits.csv')
    fr = fits[fits.tag == tag]
    if len(fr) != 1:
        raise ValueError(f'{tag} not found once in {exp}/fits.csv')
    fr = fr.iloc[0]
    p = {kk: float(fr[kk]) for kk in PARAM_KEYS}
    p['wKCKC_scale'] = float(fr['kc_kc_inh'])
    return p, fr


def wiring_summary(real_W: pd.DataFrame, shuffle: str, seed: int) -> Dict[str, float]:
    """`wPNKC_shuffle_summary` of the shuffle (the same draw `bouton_base` makes, before any
    gain) against the real wiring."""
    from al_analysis.wpnkc_shuffles import NAMED_SHUFFLES, shuffle_wPNKC, wPNKC_shuffle_summary
    method, within = NAMED_SHUFFLES[shuffle]
    shuf = shuffle_wPNKC(real_W, method, within_kc_type=within, seed=int(seed))
    s = wPNKC_shuffle_summary(real_W, shuf)
    return {k: (float(v) if not isinstance(v, (bool, np.bool_)) else bool(v)) for k, v in s.items()}


def data_values() -> Dict[str, float]:
    d = {}
    for key in ('block', 'ctrl_ceiling', 'tnt_ceiling', 'ctrl_rate_rho_ceiling', 'tnt_rate_rho_ceiling',
            'ctrl_struct_wt', 'tnt_vcf', 'tnt_weak_share', 'iso_r2'):
        try:
            d[key] = H.data_target(key)
        except Exception as e:   # a missing registry key only loses a reference line
            print('no data target', key, repr(e))
    rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    d['rate_ratio'] = float(rates['TNT'].mean() / rates['TNT3c'].mean())
    return d


def summarize(res_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """comparison.csv (part A rows with the shuffle statistics merged in, plus any part B refit
    rows from fits.csv) and the per-class means and sds."""
    stats = pd.read_csv(res_dir / 'stats_variants.csv')
    stats = stats[~(stats.get('error', pd.Series(index=stats.index, dtype=object)).astype(str).str.len() > 3)] \
        if 'error' in stats.columns else stats
    stats = stats.copy()
    stats['weak_share_off'] = stats.weak_frac_off / stats.rate_off
    shp = res_dir / 'shuffle_summary.csv'
    if shp.exists():
        sh = pd.read_csv(shp)
        stats = stats.merge(sh, on=['shuffle', 'seed'], how='left')
    rows = []
    fits_path = res_dir / 'fits.csv'
    if fits_path.exists():
        for r in pd.read_csv(fits_path).to_dict('records'):
            rows.append(dict(part='B', tag=r['tag'], model=f"thr{r['claw_thr']:g}_gain{r['gain']:g}",
                shuffle=r.get('wiring'), seed=r.get('wiring_seed'), claw_thr=r['claw_thr'], gain=r['gain'],
                loss_rate_ctrl=r['loss'], **{f'{V}__block_fruity_small_iso': r['fit_block'],
                f'{V}__on_spearman': r['fit_on_spearman'], f'{V}__off_spearman': r['fit_off_spearman'],
                'spearman_rate_on_vs_data': r['fit_po_on_spearman'], 'spearman_rate_off_vs_data': r['fit_po_off_spearman'],
                f'{V}__on_spearman_resid_vs_remap_wt': r['fit_wt_dev_on'],
                f'{V}__off_spearman_resid_vs_vcf_hyp': r['fit_vcf_off'],
                'rate_ratio_off_over_on': r['rate_ratio_off_over_on'], 'thr_factor_vs_fit': np.nan}))
    comp = pd.concat([stats, pd.DataFrame(rows)], ignore_index=True) if rows else stats
    comp.to_csv(res_dir / 'comparison.csv', index=False)
    a = stats[stats.part == 'A']
    cols = [c for c, *_ in STATS if c in a.columns]
    per_class = a.groupby(['model', 'shuffle'])[cols].agg(['mean', 'std', 'count'])
    per_class.columns = [f'{c}__{s}' for c, s in per_class.columns]
    per_class = per_class.reset_index()
    per_class.to_csv(res_dir / 'per_class.csv', index=False)
    return comp, per_class


def markdown_tables(stats: pd.DataFrame, data: Dict[str, float]) -> str:
    """One table per model: statistic | KC data | real wiring (fitted threshold, re-tuned) |
    each shuffle class as mean ± sd over seeds."""
    out = []
    a = stats[stats.part == 'A']
    for mkey, (exp, tag, thr, k, gain) in MODELS.items():
        sub = a[a.model == mkey]
        if sub.empty:
            continue
        classes = [s for s in SHUFFLES if (sub.shuffle == s).any()]
        real_fit = sub[(sub.shuffle == 'real') & (sub.variant == 'fit')]
        real_ret = sub[(sub.shuffle == 'real') & (sub.variant == 'retuned')]
        head1 = ('| | | real wiring {span=2} | ' + ' | '.join(f'{CLASS_LABEL[s]} (n = {int((sub.shuffle == s).sum())})' for s in classes) + ' |')
        head2 = '| statistic | KC data | at the fitted threshold | threshold re-tuned | ' + ' | '.join('mean ± sd' for _ in classes) + ' |'
        sep = '|---|---|---|---|' + '---|' * len(classes)
        rows = [f'**The eight glomeruli at {gain:g}, {thr:g} Hz per-claw threshold, fitted at 6.8 % on the real wiring ({exp.split("_")[0]}); '
            f'APL and KC-KC scales fixed, somatic threshold re-tuned to 6.8 % per wiring**', '', head1, head2, sep]
        for col, label, fmt, dkey in STATS:
            if col not in sub.columns:
                continue
            cells = []
            for r in (real_fit, real_ret):
                cells.append(format(float(r[col].iloc[0]), fmt) if len(r) == 1 and pd.notna(r[col].iloc[0]) else '-')
            for s in classes:
                v = sub[sub.shuffle == s][col].astype(float)
                cells.append(f'{format(v.mean(), fmt)} ± {format(v.std(ddof=1), fmt.lstrip("+"))}' if v.notna().sum() > 1
                    else (format(float(v.iloc[0]), fmt) if v.notna().sum() == 1 else '-'))
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
    # the shuffles' own statistics
    sh = stats[(stats.part == 'A') & stats.shuffle.isin(SHUFFLES)].drop_duplicates(['shuffle', 'seed'])
    if 'coinput_corr' in sh.columns and not sh.empty:
        rows = ['**What each shuffle keeps of the real wiring (mean over seeds)**', '',
            '| shuffle | claws moved | KC claw counts kept | KC glomerulus counts kept | max change in a glomerulus\' claws | max change in a glomerulus\' KCs | co-input correlation | co-input residual correlation |',
            '|---|---|---|---|---|---|---|---|']
        for s in SHUFFLES:
            g = sh[sh.shuffle == s]
            if g.empty:
                continue
            rows.append(f'| {CLASS_LABEL[s]} | {g.frac_claws_changed.mean():.2f} | {"yes" if g.kc_n_claws_preserved.all() else "no"} | '
                f'{"yes" if g.kc_n_glomeruli_preserved.all() else "no"} | {g.max_abs_glom_n_claws_change.mean():.0f} | '
                f'{g.max_abs_glom_n_kcs_change.mean():.0f} | {g.coinput_corr.mean():.2f} | {g.coinput_resid_corr.mean():.2f} |')
        out.append('\n'.join(rows))
    return '\n\n'.join(out) + '\n'


def make_figure(stats: pd.DataFrame, data: Dict[str, float], res_dir: Path) -> None:
    plt = H._plt()
    from matplotlib.lines import Line2D
    a = stats[stats.part == 'A']
    classes = ['real'] + [s for s in SHUFFLES if (a.shuffle == s).any()]
    colors = {'thr70_gain1.5': '#1f5fbf', 'thr40_gain1.5': '#d1662a', 'thr70_gain1.25': '#2a9d5c'}
    offs = {'thr70_gain1.5': -0.22, 'thr40_gain1.5': 0.0, 'thr70_gain1.25': 0.22}
    panels = [s for s in STATS if s[0] in a.columns][:10]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8.8))
    for ax, (col, label, fmt, dkey) in zip(axes.flat, panels):
        for mkey in MODELS:
            sub = a[a.model == mkey]
            if sub.empty:
                continue
            c = colors[mkey]
            for i, s in enumerate(classes):
                g = sub[sub.shuffle == s]
                if s == 'real':
                    g = g[g.variant == 'retuned']
                v = g[col].astype(float).dropna()
                if v.empty:
                    continue
                x = i + offs[mkey]
                ax.scatter(np.full(len(v), x) + np.linspace(-0.04, 0.04, len(v)), v, s=14, color=c, alpha=0.6, zorder=3)
                if len(v) > 1:
                    ax.errorbar([x], [v.mean()], yerr=[v.std(ddof=1)], fmt='_', color=c, ms=14, mew=2, capsize=3, zorder=4)
                else:
                    ax.scatter([x], v, marker='_', s=180, color=c, linewidths=2, zorder=4)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='k', ls='--', lw=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([CLASS_LABEL[s] for s in classes], rotation=25, ha='right', fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
    handles = [Line2D([], [], marker='o', ls='', color=colors[m], label={'thr70_gain1.5': 'eight at 1.5, 70 Hz',
        'thr40_gain1.5': 'eight at 1.5, 40 Hz', 'thr70_gain1.25': 'eight at 1.25, 70 Hz'}[m]) for m in MODELS]
    handles.append(Line2D([], [], ls='--', color='k', label='KC data (target or ceiling)'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle('The reference models at their fitted scales (threshold re-tuned to 6.8 %) on the real wiring and on four '
        'kinds of shuffle of it (points: seeds; bar: mean ± sd)', fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(res_dir / f'{EXPERIMENT}.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    ap.add_argument('--shuffles', nargs='+', choices=list(SHUFFLES), default=list(SHUFFLES))
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--skip-real', action='store_true', help='do not run the real-wiring variants')
    ap.add_argument('--summary', action='store_true', help='only (re)write the comparison tables and the figure')
    ap.add_argument('--local-test', metavar='DIR', default=None,
        help='write results and outputs under DIR instead of the shared / outputs folders (a smoke test)')
    args = ap.parse_args()
    if args.local_test:
        res_dir = Path(args.local_test) / 'results'; out_dir = Path(args.local_test) / 'outputs'
        res_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    else:
        res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    data = data_values()
    if args.summary:
        comp, per_class = summarize(res_dir)
        (res_dir / 'comparison_table.md').write_text(markdown_tables(comp, data))
        make_figure(comp, data, res_dir)
        pd.set_option('display.width', 250)
        print(per_class.round(3).to_string(index=False))
        return
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    shuf_path = res_dir / 'shuffle_summary.csv'
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs()
    data_mats = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    loss5 = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=0.05)
    loss_ctrl = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=CTRL_FRAC)
    # recorded at the start, so a pull during the run cannot change the commit it names
    model_prov = H.model_provenance(orn_deltas=orn)
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}
    refs = all_refs()
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame(
        {'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned_path = out_dir / 'tuned_params.json'
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    shuf_rows = list(pd.read_csv(shuf_path).to_dict('records')) if shuf_path.exists() else []
    shuf_done = {(r['shuffle'], int(r['seed'])) for r in shuf_rows}
    base_dir = out_dir / 'partA'; base_dir.mkdir(exist_ok=True)
    real_W = None

    def finish(tag, meta, on_out, off_out, wPNKC):
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data_mats, ref)
        on_p, off_p = on[V], off[V]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        row.update(suppression_stats(H.response_fractions(on_out[0]), H.response_fractions(off_out[0]), data_rates))
        row.update(fraction_stats(on_out[0], off_out[0], data_rates))
        row.update(losses(loss5, loss_ctrl, on_out, off_out))
        rows.append(row)
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index)
        fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        pd.concat({(tag, 'on'): on_p, (tag, 'acute_off'): off_p}, names=['tag', 'condition', 'odor_a', 'odor_b']).rename(
            'distance').reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False, mode='a',
            header=not (res_dir / 'model_pair_distances_long.csv').exists())
        print(f"{tag}: rates on {row['rate_on']:.4f} off {row['rate_off']:.4f} (x{row['rate_ratio_off_over_on']:.2f}); "
            f"rho ctrl {row[V + '__on_spearman']:.3f} tnt {row[V + '__off_spearman']:.3f}; block {row[V + '__block_fruity_small_iso']:+.3f}; "
            f"struct wt {row[V + '__on_spearman_resid_vs_remap_wt']:+.2f}; vcf off {row[V + '__off_spearman_resid_vs_vcf_hyp']:+.2f}; "
            f"rate rho {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; weak off {row['weak_frac_off']:.3f} "
            f"(data {row['data_weak_frac_off']:.3f}); loss @ctrl {row['loss_rate_ctrl']:.3f}; thr x{meta['thr_factor_vs_fit']:.3f}; "
            f"{meta['seconds']:.0f}s", flush=True)
        done.add(tag)

    wirings: List[Tuple[str, Optional[int]]] = ([] if args.skip_real else [('real', None)]) + \
        [(s, seed) for s in args.shuffles for seed in args.seeds]
    for mkey in args.models:
        exp, ftag, thr, k, gain = MODELS[mkey]
        p_fit, fr = fitted_params(exp, ftag)
        for shuffle, seed in wirings:
            wname = 'real' if shuffle == 'real' else f'{shuffle}-seed{seed}'
            variants = ['fit', 'retuned'] if shuffle == 'real' else ['retuned']
            todo = [v for v in variants if f'A_{mkey}_{wname}_{v}' not in done]
            if not todo and (shuffle == 'real' or (shuffle, seed) in shuf_done):
                continue
            t0 = time.time()
            try:
                initial, wnorm, W, breadth = bouton_base(orn, base_dir, claw_thr=thr, gain=gain,
                    wiring=None if shuffle == 'real' else shuffle, wiring_seed=0 if seed is None else seed)
                if shuffle != 'real' and (shuffle, seed) not in shuf_done:
                    if real_W is None:
                        _, _, real_W, _ = bouton_base(orn, base_dir, claw_thr=thr)
                    srow = dict(shuffle=shuffle, seed=seed, **wiring_summary(real_W, shuffle, seed))
                    shuf_rows.append(srow); shuf_done.add((shuffle, seed))
                    pd.DataFrame(shuf_rows).to_csv(shuf_path, index=False)
                    print(f'{wname}: claws moved {srow["frac_claws_changed"]:.2f}, co-input corr {srow["coinput_corr"]:.2f} '
                        f'(resid {srow["coinput_resid_corr"]:.2f})', flush=True)
                if not todo:
                    continue
                sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p_fit, **base_kws(thr), _wPNKC=W,
                    _wAPLKC=breadth_weights(wnorm['wAPLKC'], breadth, k), _wKCAPL=wnorm['wKCAPL'],
                    _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
                for variant in todo:
                    tag = f'A_{mkey}_{wname}_{variant}'
                    t1 = time.time()
                    if variant == 'fit':
                        p_on, thr_f = dict(p_fit), 1.0
                        r_on, s_on = sim(p_on)
                    else:
                        extra = {kk: p_fit[kk] for kk in SCALE_KEYS}
                        new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_fit['fixed_thr'], CTRL_FRAC, extra=extra)
                        p_on = dict(p_fit, fixed_thr=float(new_thr))
                    r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                    for cond, (r_, s_) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
                        r_.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
                        s_.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
                    tuned[tag] = dict(p_on, claw_thr=thr, k=k, gain=gain, ref_fit=ftag, ref_experiment=exp,
                        wiring=shuffle, wiring_seed=seed, thr_factor_vs_fit=float(thr_f))
                    tuned_path.write_text(json.dumps(tuned, indent=1))
                    meta = dict(variant=variant, tag=tag, part='A', model=mkey, shuffle=shuffle, seed=seed, base='bouton',
                        target_rate=(CTRL_FRAC if variant == 'retuned' else float(fr['rate_on'])), claw_thr=thr,
                        k=k, kc_kc_inh=p_fit['wKCKC_scale'], gain=gain, ref_fit=ftag, ref_experiment=exp,
                        ref_fit_loss=float(fr['loss']), thr_factor_vs_fit=float(thr_f),
                        n_kcs=len(r_on), n_glomeruli=W.shape[1], rate_on=rate(r_on), rate_off=rate(r_off),
                        seconds=time.time() - t1, mean_claws_per_kc=float(kc_by_glomerulus(W).sum(axis='columns').mean()),
                        **p_on)
                    meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
                    finish(tag, meta, (r_on, s_on), (r_off, s_off), W)
                del sim
            except Exception as e:  # keep going; record the failure
                print(f'A_{mkey}_{wname} FAILED after {time.time() - t0:.0f}s: {e!r}', flush=True)
                rows.append(dict(variant='retuned', tag=f'A_{mkey}_{wname}_retuned', part='A', model=mkey,
                    shuffle=shuffle, seed=seed, error=repr(e)))
                pd.DataFrame(rows).to_csv(stats_path, index=False)

    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E47_wiring_features.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), models=args.models,
        shuffles=args.shuffles, seeds=args.seeds, model_sources={m: MODELS[m][:2] for m in args.models},
        ctrl_frac=CTRL_FRAC, kc_kc_tau=TAU, loss_rate_targets=[0.05, CTRL_FRAC], approx_conc_note=H.APPROX_CONC_NOTE,
        model=model_prov)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    comp, per_class = summarize(res_dir)
    (res_dir / 'comparison_table.md').write_text(markdown_tables(comp, data))
    make_figure(comp, data, res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
