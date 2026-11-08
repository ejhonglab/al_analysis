#!/usr/bin/env python3
"""E42: the reduced gain sets on a re-drawn wiring; comparison table and figure.

The fits themselves are `E10_joint_fit.py` runs (structure loss, breadth exponent 2, the
APL->bouton scale capped at 3x, 150 evaluations) on E30 / E33's second within-KC-type
claw-permutation draw of the bouton wiring (`--wiring claw_permutation_within_type
--wiring-seed 2`), for the capped eight-glomerulus gain (the reference model, E34), the gain
on DC2 alone and the DC2 + DM5 + DM1 core with Hallem's ms input (E40's models), at the 70
and 40 Hz per-claw thresholds, each seeded from its real-wiring optimum (the exact command
list is `outputs/E42_gain_set_redraw/run_all.sh` on the host that ran them; the Status line of
the experiment's entry has it too). Their rows land in results/E42_gain_set_redraw/fits.csv.

This script puts those rows next to the same models on the real wiring (E34's and E40's
fits.csv), the uncapped eight-glomerulus fits on the same draw (E33) and the threshold-only
fits on the draw and the real wiring (E30, E25), reads each fit's APL-off response fractions
for the weak-input odors from its saved responses (E42's from ./outputs/, the others from the
model-output archive in the shared directory), and writes
results/E42_gain_set_redraw/{comparison.csv, E42_gain_set_redraw.png}.

Usage: ./venv/bin/python E42_gain_set_redraw.py
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E42_gain_set_redraw'
WEAK = ['ms', 'paa', 'B-myr', '1-8ol', 'euc']
CORE3 = {'DC2', 'DM5', 'DM1'}
# the oracle set of the glomerulus-gain experiment (E27), the reference model's eight glomeruli
ORACLE = {'DC2', 'DL1', 'DM1', 'DM5', 'D', 'DP1m', 'VA7m', 'VC1'}
# experiment -> (its fits.csv rows to include: a predicate on the row, or None for all)
SOURCES = {
    'E42_gain_set_redraw': None,
    'E40_reduced_gain_set': None,
    'E34_gain_bounded': lambda r: r.gain == 1.5,
    'E33_gain_redraw': None,
    'E30_wiring_replication': lambda r: r.wiring_seed == 2 and r.claw_thr in (40.0, 70.0),
    'E25_joint_fit_structure': lambda r: r.k == 2.0 and r.claw_thr in (40.0, 70.0),
}
STATS = [
    ('loss', 'extended loss (lower is better)', None),
    ('fit_block', 'fruity x small-ester block', 'block'),
    ('fit_on_spearman', 'rho on odor-odor distances, APL on (control flies)', 'ctrl_ceiling'),
    ('fit_off_spearman', 'rho on odor-odor distances, APL off (APL output blocked)', 'tnt_ceiling'),
    ('fit_po_on_spearman', 'response rate rho, APL on', 'ctrl_rate_rho_ceiling'),
    ('fit_po_off_spearman', 'response rate rho, APL off', 'tnt_rate_rho_ceiling'),
    ('fit_wt_dev_on', 'structure vs wild-type somata', 'ctrl_struct_wt'),
    ('fit_vcf_off', 'natural-source alignment, APL off', 'tnt_vcf'),
    ('rate_ratio_off_over_on', 'response rate, APL off / APL on', 'rate_ratio'),
    ('off_frac_ms', 'ms response fraction, APL off', 'tnt_frac_ms'),
]


def gain_set(r: pd.Series) -> str:
    """Which gain set a fits.csv row has: 'none', 'eight', 'DC2', 'core3'."""
    if pd.isna(r.get('gain')):
        return 'none'
    g = r.get('gain_glomeruli')
    if isinstance(g, str) and g.strip():
        try:
            gl = set(ast.literal_eval(g))
        except (ValueError, SyntaxError):
            gl = set(g.replace(',', ' ').split())
        if gl == {'DC2'}:
            return 'DC2'
        if gl == CORE3:
            return 'core3'
        if gl == ORACLE:
            return 'eight'
        return '+'.join(sorted(gl))
    return 'eight'


def model_name(r: pd.Series) -> str:
    gs = gain_set(r)
    inp = 'Hallem-ms ' if isinstance(r.get('orn_input'), str) and 'hallem' in r['orn_input'] else ''
    cap = '' if pd.isna(r.get('aplpn_max')) else ' (capped)'
    start2 = ' start 2' if 'start2' in str(r.get('tag', '')) else ''
    if gs == 'none':
        return 'no gain' + start2
    if gs == 'eight':
        return 'eight glomeruli' + cap + start2
    return f'{inp}{gs}' + cap + start2


def wiring_name(r: pd.Series) -> str:
    if isinstance(r.get('wiring'), str) and r['wiring']:
        return f"draw {int(r['wiring_seed'])}"
    return 'real'


def responses_path(exp: str, tag: str) -> Optional[Path]:
    local = H.outputs_dir(exp) / f'{tag}_acute_off_responses.parquet'
    if local.exists():
        return local
    arch = H.shared_dir() / 'data' / 'model_outputs' / exp / f'{tag}_acute_off_responses.parquet'
    return arch if arch.exists() else None


def weak_fractions(exp: str, tag: str) -> Dict[str, float]:
    p = responses_path(exp, tag)
    if p is None:
        return {f'off_frac_{o}': np.nan for o in WEAK}
    fr = H.response_fractions(H.abbrev_columns(pd.read_parquet(p).astype(float)))
    return {f'off_frac_{o}': float(fr.get(o, np.nan)) for o in WEAK}


def load_rows() -> pd.DataFrame:
    rows = []
    for exp, pred in SOURCES.items():
        f = H.shared_dir() / 'results' / exp / 'fits.csv'
        if not f.exists():
            print('no fits.csv for', exp)
            continue
        fits = pd.read_csv(f)
        for _, r in fits.iterrows():
            if pred is not None and not pred(r):
                continue
            row = dict(experiment=exp, tag=r.tag, model=model_name(r), wiring=wiring_name(r),
                claw_thr=float(r.claw_thr) if pd.notna(r.get('claw_thr')) else 0.0,
                k=float(r.k), n_evals=r.get('n_evals'), seconds=r.get('seconds'))
            for c in ['loss', 'initial_loss', 'fit_block', 'fit_on_spearman', 'fit_off_spearman', 'fit_po_on_spearman',
                    'fit_po_off_spearman', 'fit_wt_dev_on', 'fit_vcf_off', 'fit_iso_r2', 'rate_on', 'rate_off',
                    'rate_ratio_off_over_on', 'thr_factor', 'apl_factor_vs_e3', 'aplpn_factor_vs_e3', 'kc_kc_inh',
                    'fit_term_on_mat', 'fit_term_off_mat', 'fit_term_block', 'fit_term_rate_on', 'fit_term_ratio',
                    'fit_term_po_on', 'fit_term_po_off', 'fit_term_iso', 'fit_term_wt_dev', 'fit_term_vcf_off',
                    'fit_term_excess_on', 'spike_counts_cosine__on_mean_model', 'spike_counts_cosine__off_mean_model']:
                row[c] = r.get(c, np.nan)
            row.update(weak_fractions(exp, r.tag))
            rows.append(row)
    df = pd.DataFrame(rows)
    order = {'eight glomeruli (capped)': 0, 'eight glomeruli': 1, 'DC2 (capped)': 2, 'DC2 (capped) start 2': 2.5, 'core3 (capped)': 3,
        'Hallem-ms core3 (capped)': 4, 'no gain': 5}
    df['model_order'] = df.model.map(order).fillna(9)
    return df.sort_values(['claw_thr', 'model_order', 'wiring', 'experiment']).drop(columns='model_order')


def data_values() -> Dict[str, float]:
    d = {}
    for key in ('block', 'ctrl_ceiling', 'tnt_ceiling', 'ctrl_rate_rho_ceiling', 'tnt_rate_rho_ceiling',
            'ctrl_struct_wt', 'tnt_vcf'):
        try:
            d[key] = H.data_target(key)
        except Exception as e:   # a missing registry key only loses a reference line
            print('no data target', key, repr(e))
    rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    d['rate_ratio'] = float(rates['TNT'].mean() / rates['TNT3c'].mean())
    for o in WEAK:
        d[f'tnt_frac_{o}'] = float(rates['TNT'][o])
    return d


def make_figure(df: pd.DataFrame, data: Dict[str, float], res_dir: Path) -> None:
    plt = H._plt()
    models = [m for m in ['eight glomeruli (capped)', 'eight glomeruli', 'DC2 (capped)', 'DC2 (capped) start 2', 'core3 (capped)',
        'Hallem-ms core3 (capped)', 'no gain'] if m in set(df.model)]
    colors = {70.0: '#1f5fbf', 40.0: '#d1662a'}
    markers = {'real': 'o', 'draw 2': 's', 'draw 1': '^'}
    fig, axes = plt.subplots(2, 5, figsize=(21, 8.6))
    for ax, (col, title, dkey) in zip(axes.flat, STATS):
        for _, r in df.iterrows():
            if r.model not in models or pd.isna(r[col]):
                continue
            x = models.index(r.model) + (-0.12 if r.wiring == 'real' else 0.12)
            ax.scatter([x], [r[col]], marker=markers.get(r.wiring, 'x'), s=46, color=colors.get(r.claw_thr, '#777777'),
                facecolors='none' if r.wiring != 'real' else colors.get(r.claw_thr, '#777777'), linewidths=1.4, zorder=3)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='k', ls='--', lw=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='', color=colors[70.0], label='70 Hz, real wiring'),
        Line2D([], [], marker='s', ls='', markerfacecolor='none', color=colors[70.0], label='70 Hz, draw 2'),
        Line2D([], [], marker='o', ls='', color=colors[40.0], label='40 Hz, real wiring'),
        Line2D([], [], marker='s', ls='', markerfacecolor='none', color=colors[40.0], label='40 Hz, draw 2'),
        Line2D([], [], ls='--', color='k', label='KC data (target or ceiling)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, frameon=False)
    fig.suptitle('The reduced gain sets re-fitted on the second wiring draw, against the same models on the real wiring '
        '(capped structure fits, breadth exponent 2; filled = real wiring, open = draw 2)', fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(res_dir / 'E42_gain_set_redraw.png', dpi=130)
    plt.close(fig)


TABLE_ROWS = [
    ('loss', 'extended loss (lower is better)', '.2f', None),
    ('fit_block', 'fruity × small-ester block', '+.3f', 'block'),
    ('fit_on_spearman', 'ρ on odor-odor distances, APL on', '.2f', 'ctrl_ceiling'),
    ('fit_off_spearman', 'ρ on odor-odor distances, APL off', '.2f', 'tnt_ceiling'),
    ('fit_po_on_spearman', 'response rate ρ, APL on', '.2f', 'ctrl_rate_rho_ceiling'),
    ('fit_po_off_spearman', 'response rate ρ, APL off', '.2f', 'tnt_rate_rho_ceiling'),
    ('fit_wt_dev_on', 'structure vs wild-type somata', '.2f', 'ctrl_struct_wt'),
    ('fit_vcf_off', 'natural-source alignment, APL off', '.2f', 'tnt_vcf'),
    ('rate_ratio_off_over_on', 'response rate, APL off / APL on', '.2f', 'rate_ratio'),
    ('off_frac_ms', 'ms response fraction, APL off', '.3f', 'tnt_frac_ms'),
    ('off_frac_paa', 'paa response fraction, APL off', '.3f', 'tnt_frac_paa'),
    ('thr_factor', 'somatic threshold (× the base)', '.2f', None),
    ('apl_factor_vs_e3', 'APL↔KC scale (× the base)', '.2f', None),
    ('aplpn_factor_vs_e3', 'APL→bouton scale (× the base)', '.2f', None),
    ('kc_kc_inh', 'KC–KC scale', '.1f', None),
    ('n_evals', 'evaluations', '.0f', None),
]
MODELS = ['eight glomeruli (capped)', 'DC2 (capped)', 'DC2 (capped) start 2', 'Hallem-ms core3 (capped)']
SHORT = {'eight glomeruli (capped)': 'eight glomeruli', 'DC2 (capped)': 'DC2 alone', 'DC2 (capped) start 2': 'DC2 alone, second start',
    'Hallem-ms core3 (capped)': 'Hallem-ms core'}


def markdown_tables(df: pd.DataFrame, data: Dict[str, float]) -> str:
    """One table per threshold: statistic | KC data | each model on the real wiring and on draw 2."""
    out = []
    for thr in (70.0, 40.0):
        sub = df[(df.claw_thr == thr) & df.model.isin(MODELS)]
        models = [m for m in MODELS if (sub.model == m).any()]
        # a second start exists only on the draw: one column for it
        wirings = {m: (('draw 2',) if m.endswith('start 2') else ('real', 'draw 2')) for m in models}
        head1 = '| | | ' + ' | '.join(f'{SHORT[m]} {{span={len(wirings[m])}}}' if len(wirings[m]) > 1 else SHORT[m] for m in models) + ' |'
        head2 = '| statistic | KC data | ' + ' | '.join(' | '.join('real wiring' if w == 'real' else w for w in wirings[m]) for m in models) + ' |'
        sep = '|---|---|' + '---|' * sum(len(wirings[m]) for m in models)
        rows = [f'**{thr:g} Hz per-claw threshold**', '', head1, head2, sep]
        for col, label, fmt, dkey in TABLE_ROWS:
            cells = []
            for m in models:
                for w in wirings[m]:
                    r = sub[(sub.model == m) & (sub.wiring == w)]
                    cells.append(format(float(r[col].iloc[0]), fmt) if len(r) == 1 and pd.notna(r[col].iloc[0]) else '-')
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
    return '\n\n'.join(out) + '\n'


def main() -> None:
    res_dir = H.results_dir(EXPERIMENT)
    df = load_rows()
    data = data_values()
    df.to_csv(res_dir / 'comparison.csv', index=False)
    (res_dir / 'comparison_table.md').write_text(markdown_tables(df, data))
    pd.set_option('display.width', 250)
    cols = ['experiment', 'model', 'wiring', 'claw_thr', 'loss', 'fit_block', 'fit_on_spearman', 'fit_off_spearman',
        'fit_po_on_spearman', 'fit_po_off_spearman', 'fit_wt_dev_on', 'fit_vcf_off', 'rate_ratio_off_over_on', 'off_frac_ms']
    print(df[cols].round(3).to_string(index=False))
    print('data:', {k: round(v, 3) for k, v in data.items()})
    make_figure(df, data, res_dir)
    print('->', res_dir)


if __name__ == '__main__':
    main()
