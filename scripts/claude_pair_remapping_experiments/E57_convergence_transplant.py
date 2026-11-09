#!/usr/bin/env python3
"""E57: building the natural-source alignment back, the food-glomerulus convergence transplanted
into shuffled wirings (the sufficiency test of E52's and E54's two carriers).
E52 and E54 found the fitted models' natural-source alignment to have two wiring carriers, the
KC-type organization of the glomerular input (kept by the within-type shuffles) and the
food-glomerulus convergence (the 32 over-shared pairs with ORN data, which have to be removed as a
whole, and carry the alignment only through the gain), both shown by removal. Here the convergence
is put back: `al_analysis.wpnkc_shuffles.lesion_coinput` with targets above the current counts
raises each of the 32 pairs' shared-KC counts to the real wiring's by margin-preserving claw swaps
(a KC with one glomerulus of the pair takes a claw of the other from a same-type KC that has the
other and not the first, giving a claw of a third glomerulus in exchange).
(A) Within-type Curveball draws (E47's seeds 1-3; type structure kept, convergence destroyed) with
    the 32 pairs raised to the real counts.
(B) The same on plain Curveball draws (type structure destroyed too).
(C) The control: on the same draws, the count-matched control pairs of E54's 32 raised by the same
    amounts (each control pair by what its matched over-shared pair was raised).
The three reference models of E47's part A (the eight glomeruli at 1.5 at 40 and 70 Hz, the eight
at 1.25 at 70 Hz) at their fitted scales with the somatic threshold re-tuned to 6.8 % on each
wiring, scored as E47 / E52 / E54 score their variants; --summary puts them beside E47's rows for
the untreated draws and the real wiring.
Outputs: <shared>/results/E57_convergence_transplant/ (transplant_pairs.csv, transplant_outcome.csv,
stats_variants.csv, comparison.csv, comparison_table.md, per_class.csv, paired_differences.csv, E57_convergence_transplant.png,
provenance.json) and ./outputs/E57_convergence_transplant/ (the transplanted wirings, swaps,
per-variant responses / spike counts, partC/ caches).
Usage: ./venv/bin/python E57_convergence_transplant.py --plan            (the wirings only)
       OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E57_convergence_transplant.py [--seeds 1 2 3]
       ./venv/bin/python E57_convergence_transplant.py --summary
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E47_wiring_features import MODELS, STATS, V, CLASS_LABEL, data_values
from E52_glomerulus_pairs import base_wiring, EXPERIMENT as E52
from E54_gain_and_lesion import run_variants, MODEL_LABEL, EXPERIMENT as E54

EXPERIMENT = 'E57_convergence_transplant'
E47 = 'E47_wiring_features'
CLASSES = ('curveball_within_type', 'curveball')
KINDS = ('transplant', 'control')
GROUP_LABEL = {'real': 'real wiring',
    'curveball_within_type': 'Curveball within type, untreated (E47)', 'curveball': 'Curveball, untreated (E47)',
    'curveball_within_type_transplant': 'Curveball within type + the 32 pairs\' convergence',
    'curveball_within_type_control': 'Curveball within type + the same raises on control pairs',
    'curveball_transplant': 'Curveball + the 32 pairs\' convergence',
    'curveball_control': 'Curveball + the same raises on control pairs'}


def pairs_and_controls() -> pd.DataFrame:
    """The 32 over-shared pairs with ORN data and their matched control pairs (E54's split plan,
    both halves), with the real wiring's shared-KC counts."""
    plan = pd.read_csv(H.shared_dir() / 'results' / E54 / 'lesion_pairs.csv')
    tgt = plan[plan.kind.isin(['lesion_gained', 'lesion_ungained'])].copy()
    ctl = plan[plan.kind.isin(['control_gained', 'control_ungained'])].copy()
    tgt['key'] = tgt.glom_a + '/' + tgt.glom_b
    ctl = ctl.set_index('matched_to')
    rows = []
    for r in tgt.itertuples():
        c = ctl.loc[r.key]
        rows.append(dict(glom_a=r.glom_a, glom_b=r.glom_b, real_count=int(r.observed), control_a=c.glom_a, control_b=c.glom_b,
            control_real_count=int(c.observed)))
    return pd.DataFrame(rows)


def make_wirings(seeds: List[int], out_dir: Path, res_dir: Path) -> Dict[Tuple[str, str, int], pd.DataFrame]:
    from al_analysis.wpnkc_shuffles import NAMED_SHUFFLES, shuffle_wPNKC, lesion_coinput, coinput_counts, wPNKC_shuffle_summary
    W = base_wiring()
    pairs = pairs_and_controls()
    pairs.to_csv(res_dir / 'transplant_pairs.csv', index=False)
    co_real = coinput_counts(W)
    out: Dict[Tuple[str, str, int], pd.DataFrame] = {}
    oc_path = res_dir / 'transplant_outcome.csv'
    outcome = list(pd.read_csv(oc_path).to_dict('records')) if oc_path.exists() else []
    outcome = [r for r in outcome if int(r['seed']) not in seeds]
    for cls in CLASSES:
        method, within = NAMED_SHUFFLES[cls]
        for seed in seeds:
            S = shuffle_wPNKC(W, method, within_kc_type=within, seed=seed)
            co_s = coinput_counts(S)
            tp = {(a, b): int(co_real.loc[a, b]) for a, b in zip(pairs.glom_a, pairs.glom_b)}
            # each control pair raised by what its matched over-shared pair is raised (floored at 0)
            cp = {}
            for r in pairs.itertuples():
                delta = max(int(co_real.loc[r.glom_a, r.glom_b]) - int(co_s.loc[r.glom_a, r.glom_b]), 0)
                cp[(r.control_a, r.control_b)] = int(co_s.loc[r.control_a, r.control_b]) + delta
            for kind, targets, protect in (('transplant', tp, list(cp)), ('control', cp, list(tp))):
                path = out_dir / f'wPNKC_{cls}-seed{seed}_{kind}.parquet'
                if path.exists():
                    L = pd.read_parquet(path); swaps = pd.read_csv(out_dir / f'swaps_{cls}-seed{seed}_{kind}.csv')
                else:
                    t0 = time.time()
                    L, swaps = lesion_coinput(S, targets, seed=100 + seed, within_kc_type=True, protect=protect, strict=False,
                        passes=4)
                    L.to_parquet(path); swaps.to_csv(out_dir / f'swaps_{cls}-seed{seed}_{kind}.csv', index=False)
                    print(f'{cls} seed {seed} {kind}: {len(swaps)} claw swaps in {time.time() - t0:.0f}s', flush=True)
                out[(cls, kind, seed)] = L
                co = coinput_counts(L); s = wPNKC_shuffle_summary(S, L); s_real = wPNKC_shuffle_summary(W, L)
                outcome.append(dict(shuffle=cls, kind=kind, seed=seed, n_pairs=len(targets), n_swaps=len(swaps),
                    frac_claws_changed_vs_draw=float(s['frac_claws_changed']), coinput_corr_vs_real=float(s_real['coinput_corr']),
                    coinput_resid_corr_vs_real=float(s_real['coinput_resid_corr']),
                    coinput_corr_draw_vs_real=float(wPNKC_shuffle_summary(W, S)['coinput_corr']),
                    kc_n_claws_preserved=bool(s['kc_n_claws_preserved']), max_abs_glom_n_claws_change=float(s['max_abs_glom_n_claws_change']),
                    sum_raised_listed=int(sum(int(co.loc[a, b]) - int(co_s.loc[a, b]) for a, b in targets)),
                    sum_planned=int(sum(t - int(co_s.loc[a, b]) for (a, b), t in targets.items())),
                    over_shared_total_real=int(sum(int(co_real.loc[a, b]) for a, b in tp)),
                    over_shared_total_draw=int(sum(int(co_s.loc[a, b]) for a, b in tp)),
                    over_shared_total_after=int(sum(int(co.loc[a, b]) for a, b in tp)),
                    protected_changed=int(sum(abs(int(co.loc[a, b]) - int(co_s.loc[a, b])) for a, b in protect)),
                    reached=json.dumps({f'{a}/{b}': int(co.loc[a, b]) for a, b in targets})))
                pd.DataFrame(outcome).to_csv(oc_path, index=False)
    return out


def jobs_for(wirings, res_dir: Path, out_dir: Path) -> List[dict]:
    jobs = []
    for mkey, spec in MODELS.items():
        for (cls, kind, seed), L in wirings.items():
            group = f'{cls}_{kind}'
            jobs.append(dict(tag=f'T_{mkey}_{group}-seed{seed}_retuned', part='T', model=mkey, spec=spec, gain=spec[4], wiring=L,
                wiring_tag=f'{group}-seed{seed}', shuffle=group, seed=seed))
    return jobs


def summary(res_dir: Path) -> None:
    stats_path = res_dir / 'stats_variants.csv'
    if not stats_path.exists():
        print('no rows yet'); return
    mine = pd.read_csv(stats_path)
    if 'error' in mine.columns:
        mine = mine[~(mine.error.astype(str).str.len() > 3)]
    e47 = pd.read_csv(H.shared_dir() / 'results' / E47 / 'stats_variants.csv')
    if 'error' in e47.columns:
        e47 = e47[~(e47.error.astype(str).str.len() > 3)]
    e47 = e47[(e47.part == 'A') & (e47.shuffle.isin(['real'] + list(CLASSES)))]
    e47 = e47[(e47.shuffle != 'real') | (e47.variant == 'retuned')]
    comp = pd.concat([e47, mine], ignore_index=True)
    comp['vcf_on_minus_off'] = comp[f'{V}__on_spearman_resid_vs_vcf_hyp'] - comp[f'{V}__off_spearman_resid_vs_vcf_hyp']
    comp['weak_share_off'] = comp.weak_frac_off / comp.rate_off
    comp.to_csv(res_dir / 'comparison.csv', index=False)
    cols = [c for c, *_ in STATS if c in comp.columns]
    grp = comp[comp.shuffle != 'real'].groupby(['model', 'shuffle'])[cols].agg(['mean', 'std', 'count'])
    grp.columns = [f'{c}__{s}' for c, s in grp.columns]
    grp.reset_index().to_csv(res_dir / 'per_class.csv', index=False)
    paired = paired_differences(comp)
    paired.to_csv(res_dir / 'paired_differences.csv', index=False)
    data = data_values()
    out = []
    groups = ['curveball_within_type', 'curveball_within_type_transplant', 'curveball_within_type_control',
        'curveball', 'curveball_transplant', 'curveball_control']
    for mkey in MODELS:
        sub = comp[comp.model == mkey]
        if sub[sub.shuffle.str.endswith('transplant')].empty:
            continue
        present = [g for g in groups if (sub.shuffle == g).any()]
        real = sub[sub.shuffle == 'real']
        rows = [f'**The {MODEL_LABEL[mkey]} at its fitted scales (threshold re-tuned to 6.8 %) on the real wiring, on untreated Curveball draws '
            '(E47, five draws) and on draws with the 32 over-shared pairs\' convergence transplanted or the same raises on control pairs**', '',
            '| | | real wiring | ' + ' | '.join(f'{GROUP_LABEL[g]} (n = {int((sub.shuffle == g).sum())})' for g in present) + ' |',
            '| statistic | KC data | threshold re-tuned | ' + ' | '.join('mean ± sd' for _ in present) + ' |',
            '|---|---|---|' + '---|' * len(present)]
        for col, label, fmt, dkey in STATS:
            if col not in sub.columns:
                continue
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            cells = [format(float(real[col].iloc[0]), fmt) if len(real) == 1 and pd.notna(real[col].iloc[0]) else '-']
            for g in present:
                v = sub[sub.shuffle == g][col].astype(float).dropna()
                cells.append(f'{format(v.mean(), fmt)} ± {format(v.std(ddof=1), fmt.lstrip("+"))}' if len(v) > 1
                    else (format(float(v.iloc[0]), fmt) if len(v) == 1 else '-'))
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
    out.extend(paired_tables(paired, comp, data))
    oc_path = res_dir / 'transplant_outcome.csv'
    if oc_path.exists():
        oc = pd.read_csv(oc_path)
        rows = ['**The transplants (per draw)**', '',
            '| draw | treatment | pairs | shared KCs raised on the listed pairs (planned) | claw swaps | claws moved (fraction of the draw\'s) | '
            'co-input correlation with the real wiring: draw -> treated | over-shared pairs\' shared KCs: real / draw / after | protected pairs changed by |',
            '|---|---|---|---|---|---|---|---|---|']
        for r in oc.itertuples():
            rows.append(f'| {CLASS_LABEL[r.shuffle]} seed {r.seed} | {r.kind} | {r.n_pairs} | {r.sum_raised_listed} ({r.sum_planned}) | {r.n_swaps} | '
                f'{r.frac_claws_changed_vs_draw:.3f} | {r.coinput_corr_draw_vs_real:.3f} -> {r.coinput_corr_vs_real:.3f} | '
                f'{r.over_shared_total_real} / {r.over_shared_total_draw} / {r.over_shared_total_after} | {r.protected_changed} |')
        out.append('\n'.join(rows))
    (res_dir / 'comparison_table.md').write_text('\n\n'.join(out) + '\n')
    figure(comp, data, res_dir, groups)
    print('\n\n'.join(out))


PAIRED_COLS = (f'{V}__on_spearman_resid_vs_vcf_hyp', f'{V}__off_spearman_resid_vs_vcf_hyp', 'vcf_on_minus_off',
    f'{V}__block_fruity_small_iso', f'{V}__on_spearman_resid_vs_remap_wt', 'spearman_rate_on_vs_data', 'loss_rate_ctrl')


def paired_differences(comp: pd.DataFrame) -> pd.DataFrame:
    """Each treated draw against the same seed's untreated draw (E47's row for that draw): the treatment's
    effect per draw and its mean and sd over draws, for the statistics in PAIRED_COLS (long format).

    The group means of the comparison tables set three treated draws against E47's five untreated ones;
    this takes the draw-to-draw variation out by differencing within a seed.
    """
    rows = []
    for mkey in MODELS:
        sub = comp[comp.model == mkey]
        for cls in CLASSES:
            base = sub[sub.shuffle == cls].set_index('seed')
            for kind in KINDS:
                t = sub[sub.shuffle == f'{cls}_{kind}'].set_index('seed')
                seeds = sorted(set(t.index) & set(base.index))
                if not seeds:
                    continue
                for col in PAIRED_COLS:
                    if col not in sub.columns:
                        continue
                    tv = t.loc[seeds, col].astype(float).values; bv = base.loc[seeds, col].astype(float).values
                    d = tv - bv
                    rows.append(dict(model=mkey, shuffle=cls, kind=kind, stat=col, n=len(seeds), mean_diff=float(d.mean()),
                        sd_diff=float(d.std(ddof=1)) if len(seeds) > 1 else float('nan'), untreated_mean=float(bv.mean()),
                        treated_mean=float(tv.mean()), per_seed=json.dumps({int(sd): round(float(x), 4) for sd, x in zip(seeds, d)})))
    return pd.DataFrame(rows)


def paired_tables(paired: pd.DataFrame, comp: pd.DataFrame, data: Dict[str, float]) -> List[str]:
    """One markdown table per model: the KC data, the real wiring, and per shuffle class the untreated draws
    (seeds 1 to 3) and each treatment's effect on the same draws (mean ± sd of the paired differences)."""
    out = []
    for mkey in MODELS:
        p = paired[paired.model == mkey]
        if p.empty:
            continue
        real = comp[(comp.model == mkey) & (comp.shuffle == 'real')]
        rows = [f'**{MODEL_LABEL[mkey]} at its fitted scales: each treated draw against the same seed\'s untreated draw '
            '(the treatment\'s effect, mean ± sd over the three draws)**', '',
            '| | | | Curveball within type {span=3} | Curveball {span=3} |',
            '| statistic | KC data | real wiring | untreated (seeds 1 to 3) | + the 32 pairs\' convergence, minus untreated | '
            '+ the same raises on control pairs, minus untreated | untreated (seeds 1 to 3) | + the 32 pairs\' convergence, minus '
            'untreated | + the same raises on control pairs, minus untreated |',
            '|---|---|---|---|---|---|---|---|---|']
        for col, label, fmt, dkey in STATS:
            if col not in PAIRED_COLS or col not in comp.columns:
                continue
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            cells = [format(float(real[col].iloc[0]), fmt) if len(real) == 1 and pd.notna(real[col].iloc[0]) else '-']
            dfmt = '+' + fmt.lstrip('+'); sfmt = fmt.lstrip('+')
            for cls in CLASSES:
                q = p[(p.shuffle == cls) & (p.stat == col)].set_index('kind')
                if q.empty:
                    cells += ['-', '-', '-']; continue
                cells.append(format(float(q.untreated_mean.iloc[0]), fmt))
                for kind in KINDS:
                    if kind in q.index:
                        r = q.loc[kind]
                        cells.append(f'{format(r.mean_diff, dfmt)} ± {format(r.sd_diff, sfmt)}' if r.n > 1 else format(r.mean_diff, dfmt))
                    else:
                        cells.append('-')
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
    return out


def figure(comp: pd.DataFrame, data: Dict[str, float], res_dir: Path, groups: List[str]) -> None:
    plt = H._plt()
    from matplotlib.lines import Line2D
    panels = [(f'{V}__off_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL off', 'tnt_vcf'),
        (f'{V}__on_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL on', 'ctrl_vcf'),
        (f'{V}__block_fruity_small_iso', 'fruity × small-ester block', 'block'),
        ('spearman_rate_on_vs_data', 'response rate ρ, APL on', 'ctrl_rate_rho_ceiling'),
        ('loss_rate_ctrl', 'extended loss at 6.8 % (lower is better)', None)]
    xs = ['real'] + [g for g in groups if (comp.shuffle == g).any()]
    colors = {'thr70_gain1.5': '#1f5fbf', 'thr40_gain1.5': '#d1662a', 'thr70_gain1.25': '#2a9d5c'}
    offs = {'thr70_gain1.5': -0.2, 'thr40_gain1.5': 0.0, 'thr70_gain1.25': 0.2}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 6.8))
    for ax, (col, label, dkey) in zip(axes, panels):
        for m in MODELS:
            sub = comp[comp.model == m]
            for i, g in enumerate(xs):
                v = sub[sub.shuffle == g][col].astype(float).dropna()
                if v.empty:
                    continue
                x = i + offs[m]
                ax.scatter(np.full(len(v), x) + np.linspace(-0.03, 0.03, len(v)), v, s=14, color=colors[m], alpha=0.6, zorder=3)
                if len(v) > 1:
                    ax.errorbar([x], [v.mean()], yerr=[v.std(ddof=1)], fmt='_', color=colors[m], ms=12, mew=2, capsize=3, zorder=4)
                else:
                    ax.scatter([x], v, marker='_', s=150, color=colors[m], linewidths=2, zorder=4)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='k', ls='--', lw=1)
        ax.axvline(3.5, color='0.7', lw=0.8)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([GROUP_LABEL[g] for g in xs], rotation=35, ha='right', fontsize=7)
        ax.set_title(label, fontsize=9); ax.grid(axis='y', alpha=0.25)
    handles = [Line2D([], [], marker='o', ls='', color=colors[m], label=MODEL_LABEL[m]) for m in MODELS]
    handles.append(Line2D([], [], ls='--', color='k', label='KC data (target or ceiling)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle('Building the natural-source alignment back: the reference models at fixed scales on Curveball draws (within type / plain) '
        'with the 32 over-shared pairs\' convergence transplanted, or the same raises on control pairs (points: draws; bar: mean ± sd)', fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(res_dir / f'{EXPERIMENT}.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3], help='the Curveball draws (E47\'s seeds)')
    ap.add_argument('--plan', action='store_true', help='only build the transplanted wirings and their table')
    ap.add_argument('--summary', action='store_true', help='only (re)write the comparison tables and the figure')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    if args.summary:
        summary(res_dir); return
    wirings = make_wirings(args.seeds, out_dir, res_dir)
    pd.set_option('display.width', 250)
    print(pd.read_csv(res_dir / 'transplant_outcome.csv').drop(columns=['reached']).to_string(index=False))
    if args.plan:
        return
    jobs = jobs_for(wirings, res_dir, out_dir)
    print(f'{len(jobs)} variants', flush=True)
    run_variants(jobs, res_dir, out_dir)
    summary(res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
