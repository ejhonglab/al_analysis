#!/usr/bin/env python3
"""E46: the operating-point fits on the second wiring draw; comparison tables and figure.

The fits themselves are `E10_joint_fit.py` runs (structure loss, breadth exponent 2, the
APL->bouton scale capped at 3x, the response-rate target at the control flies' fraction,
150 evaluations) on E30 / E33 / E42's second within-KC-type claw-permutation draw of the
bouton wiring (`--wiring claw_permutation_within_type --wiring-seed 2`), for the two best
fits of the glomerulus gain at the data's operating point (E45: the eight glomeruli at 1.25
at 70 Hz, the DC2 + DM5 + DM1 core with Hallem's ms input at 1.5 at 40 Hz) and for the
reference model (the eight at 1.5, E44's refits) at both thresholds, each seeded from its
real-wiring optimum (the command list is `outputs/E46_operating_point_redraw/run.sh` on the
host that ran them). Their rows land in results/E46_operating_point_redraw/fits.csv.

This script puts those rows next to the same models on the real wiring at 6.8 % (E45's and
E44's fits.csv), and next to the same models at the 5 % convention on both wirings (E42's
draw-2 fits; E34's and E40's real-wiring fits), reads each fit's APL-off response fractions
for the weak-input odors from its saved responses (E46's from ./outputs/, the others from
the model-output archive in the shared directory), and writes
results/E46_operating_point_redraw/{comparison.csv, comparison_table.md,
E46_operating_point_redraw.png}. Losses are the fits' own; `loss_no_rate` (the loss without
its APL-on response-rate term) is the number comparable across the two operating points.

Usage: ./venv/bin/python E46_operating_point_redraw.py
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E26_claw_power import WEAK

EXPERIMENT = 'E46_operating_point_redraw'
CORE3 = {'DC2', 'DM5', 'DM1'}
# the oracle set of the glomerulus-gain experiment (E27), the reference model's eight glomeruli
ORACLE = {'DC2', 'DL1', 'DM1', 'DM5', 'D', 'DP1m', 'VA7m', 'VC1'}
CTRL_FRAC = H.data_target('ctrl_frac')
# experiment -> predicate on its fits.csv rows (None: all rows)
SOURCES = {
    'E46_operating_point_redraw': None,
    'E45_gain_at_ctrl_frac': lambda r: pd.notna(r.get('gain')),
    'E44_operating_point': None,
    'E42_gain_set_redraw': lambda r: 'DC2_cap3' not in str(r.tag),
    'E34_gain_bounded': lambda r: r.gain == 1.5,
    'E40_reduced_gain_set': lambda r: isinstance(r.get('orn_input'), str) and 'hallem' in r['orn_input'],
}
COLUMNS = ['loss', 'initial_loss', 'fit_block', 'fit_on_spearman', 'fit_off_spearman', 'fit_po_on_spearman',
    'fit_po_off_spearman', 'fit_wt_dev_on', 'fit_vcf_off', 'fit_iso_r2', 'rate_on', 'rate_off',
    'rate_ratio_off_over_on', 'thr_factor', 'apl_factor_vs_e3', 'aplpn_factor_vs_e3', 'kc_kc_inh',
    'fit_term_on_mat', 'fit_term_off_mat', 'fit_term_block', 'fit_term_rate_on', 'fit_term_ratio',
    'fit_term_po_on', 'fit_term_po_off', 'fit_term_iso', 'fit_term_wt_dev', 'fit_term_vcf_off',
    'fit_term_excess_on', 'spike_counts_cosine__on_mean_model', 'spike_counts_cosine__off_mean_model',
    'fixed_thr', 'wAPLKC_scale', 'wAPLPN_scale']
STATS = [
    ('loss', 'extended loss (lower is better)', None),
    ('fit_block', 'fruity x small-ester block', 'block'),
    ('fit_on_spearman', 'rho on odor-odor distances, APL on (control flies)', 'ctrl_ceiling'),
    ('fit_off_spearman', 'rho on odor-odor distances, APL off (APL output blocked)', 'tnt_ceiling'),
    ('fit_po_on_spearman', 'response rate rho, APL on', 'ctrl_rate_rho_ceiling'),
    ('fit_po_off_spearman', 'response rate rho, APL off', 'tnt_rate_rho_ceiling'),
    ('fit_wt_dev_on', 'structure vs wild-type somata', 'ctrl_struct_wt'),
    ('fit_vcf_on', 'natural-source alignment, APL on', 'ctrl_vcf'),
    ('fit_vcf_off', 'natural-source alignment, APL off', 'tnt_vcf'),
    ('vcf_on_minus_off', 'natural-source alignment, APL on minus off', 'vcf_apl_effect'),
    ('rate_ratio_off_over_on', 'response rate, APL off / APL on', 'rate_ratio'),
    ('weak_share_off', "weak odors' share of the APL-off response rate", 'tnt_weak_share'),
]
TABLE_ROWS = [
    ('loss', 'extended loss (lower is better)', '.2f', None),
    ('loss_no_rate', 'extended loss without its response-rate term', '.2f', None),
    ('fit_block', 'fruity × small-ester block', '+.3f', 'block'),
    ('fit_on_spearman', 'ρ on odor-odor distances, APL on', '.2f', 'ctrl_ceiling'),
    ('fit_off_spearman', 'ρ on odor-odor distances, APL off', '.2f', 'tnt_ceiling'),
    ('fit_po_on_spearman', 'response rate ρ, APL on', '.2f', 'ctrl_rate_rho_ceiling'),
    ('fit_po_off_spearman', 'response rate ρ, APL off', '.2f', 'tnt_rate_rho_ceiling'),
    ('fit_wt_dev_on', 'structure vs wild-type somata', '.2f', 'ctrl_struct_wt'),
    ('fit_vcf_on', 'natural-source alignment, APL on', '.2f', 'ctrl_vcf'),
    ('fit_vcf_off', 'natural-source alignment, APL off', '.2f', 'tnt_vcf'),
    ('vcf_on_minus_off', 'natural-source alignment, APL on minus off', '+.2f', 'vcf_apl_effect'),
    ('fit_iso_r2', 'isotonic R² of the APL-on vs APL-off relation', '.2f', 'iso_r2'),
    ('rate_ratio_off_over_on', 'response rate, APL off / APL on', '.2f', 'rate_ratio'),
    ('weak_frac_off', "weak odors' response rate, APL off", '.3f', 'tnt_weak_frac'),
    ('weak_share_off', "the same, relative to the mean over the 16 odors", '.2f', 'tnt_weak_share'),
    ('off_frac_ms', 'ms response fraction, APL off', '.3f', 'tnt_frac_ms'),
    ('off_frac_paa', 'paa response fraction, APL off', '.3f', 'tnt_frac_paa'),
    ('thr_factor', 'somatic threshold (× the base)', '.2f', None),
    ('apl_factor_vs_e3', 'APL↔KC scale (× the base)', '.2f', None),
    ('aplpn_factor_vs_e3', 'APL→bouton scale (× the base)', '.2f', None),
    ('kc_kc_inh', 'KC–KC scale', '.1f', None),
    ('n_evals', 'evaluations', '.0f', None),
]
# statistics in the draw-cost table (draw 2 minus real wiring)
DELTA_ROWS = [(c, l, f) for c, l, f, _ in TABLE_ROWS if c in ('loss_no_rate', 'fit_block', 'fit_on_spearman',
    'fit_off_spearman', 'fit_po_on_spearman', 'fit_po_off_spearman', 'fit_wt_dev_on', 'fit_vcf_on', 'fit_vcf_off',
    'vcf_on_minus_off')]
DELTA_ROWS.append(('weak_share_off', "weak odors' share of the APL-off response rate", '.2f'))
MODELS = ['eight glomeruli at 1.25', 'eight glomeruli at 1.5', 'Hallem-ms core at 1.5']


def gain_set(r: pd.Series) -> str:
    """Which gain set a fits.csv row has: 'none', 'eight', 'DC2', 'core3' (or the sorted set)."""
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
    if gs == 'none':
        return 'no gain'
    inp = 'Hallem-ms ' if isinstance(r.get('orn_input'), str) and 'hallem' in r['orn_input'] else ''
    name = 'eight glomeruli' if gs == 'eight' else ('core' if gs == 'core3' else gs)
    return f'{inp}{name} at {float(r.gain):g}'


def wiring_name(r: pd.Series) -> str:
    if isinstance(r.get('wiring'), str) and r['wiring']:
        return f"draw {int(r['wiring_seed'])}"
    return 'real'


def operating_point(r: pd.Series) -> str:
    t = r.get('rate_target')
    if pd.notna(t) and abs(float(t) - CTRL_FRAC) < 1e-6:
        return '6.8 %'
    return '5 %'


def responses_path(exp: str, tag: str) -> Optional[Path]:
    local = H.outputs_dir(exp) / f'{tag}_acute_off_responses.parquet'
    if local.exists():
        return local
    arch = H.shared_dir() / 'data' / 'model_outputs' / exp / f'{tag}_acute_off_responses.parquet'
    return arch if arch.exists() else None


def weak_fractions(exp: str, tag: str) -> Dict[str, float]:
    """The weak-input odors' APL-off response fractions, their mean and its share of the
    16-odor mean, from the fit's saved APL-off responses."""
    p = responses_path(exp, tag)
    out = {f'off_frac_{o}': np.nan for o in WEAK}
    out.update(weak_frac_off=np.nan, weak_share_off=np.nan)
    if p is None:
        return out
    fr = H.response_fractions(H.abbrev_columns(pd.read_parquet(p).astype(float)))
    for o in WEAK:
        out[f'off_frac_{o}'] = float(fr.get(o, np.nan))
    weak = float(fr.reindex(WEAK).mean())
    out.update(weak_frac_off=weak, weak_share_off=weak / float(fr.reindex(H.APL16_ODORS).mean()))
    return out


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
                operating_point=operating_point(r),
                claw_thr=float(r.claw_thr) if pd.notna(r.get('claw_thr')) else 0.0,
                k=float(r.k), n_evals=r.get('n_evals'), seconds=r.get('seconds'))
            for c in COLUMNS:
                row[c] = r.get(c, np.nan)
            row['loss_no_rate'] = row['loss'] - row['fit_term_rate_on']
            # the natural-source alignment with APL on (the loss only scores it with APL off), and APL on minus off
            row['fit_vcf_on'] = r.get('spike_counts_cosine__on_spearman_resid_vs_vcf_hyp', np.nan)
            row['vcf_on_minus_off'] = row['fit_vcf_on'] - row['fit_vcf_off']
            row.update(weak_fractions(exp, r.tag))
            rows.append(row)
    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODELS)}
    df['model_order'] = df.model.map(order).fillna(9)
    df['op_order'] = df.operating_point.map({'6.8 %': 0, '5 %': 1})
    return df.sort_values(['claw_thr', 'model_order', 'op_order', 'wiring', 'experiment']).drop(
        columns=['model_order', 'op_order'])


def data_values() -> Dict[str, float]:
    d = {}
    for key in ('block', 'ctrl_ceiling', 'tnt_ceiling', 'ctrl_rate_rho_ceiling', 'tnt_rate_rho_ceiling',
            'ctrl_struct_wt', 'ctrl_vcf', 'tnt_vcf', 'vcf_apl_effect', 'tnt_weak_frac', 'tnt_weak_share', 'iso_r2'):
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
    from matplotlib.lines import Line2D
    models = [m for m in MODELS if m in set(df.model)]
    colors = {70.0: '#1f5fbf', 40.0: '#d1662a'}
    fig, axes = plt.subplots(2, 5, figsize=(21, 8.6))
    for ax, (col, title, dkey) in zip(axes.flat, STATS):
        for _, r in df.iterrows():
            if r.model not in models or pd.isna(r[col]):
                continue
            at68 = r.operating_point == '6.8 %'
            x = models.index(r.model) + (-0.18 if r.wiring == 'real' else 0.18) + (0 if at68 else 0.07)
            c = colors.get(r.claw_thr, '#777777')
            ax.scatter([x], [r[col]], marker='o' if r.wiring == 'real' else 's', s=52 if at68 else 26, color=c,
                facecolors=c if r.wiring == 'real' else 'none', linewidths=1.4, alpha=1.0 if at68 else 0.4, zorder=3)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='k', ls='--', lw=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
    handles = [Line2D([], [], marker='o', ls='', color=colors[70.0], label='70 Hz, real wiring'),
        Line2D([], [], marker='s', ls='', markerfacecolor='none', color=colors[70.0], label='70 Hz, draw 2'),
        Line2D([], [], marker='o', ls='', color=colors[40.0], label='40 Hz, real wiring'),
        Line2D([], [], marker='s', ls='', markerfacecolor='none', color=colors[40.0], label='40 Hz, draw 2'),
        Line2D([], [], marker='o', ls='', color='#777777', alpha=0.4, markersize=4, label='small, faint: the same at 5 %'),
        Line2D([], [], ls='--', color='k', label='KC data (target or ceiling)')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9, frameon=False)
    fig.suptitle('The operating-point fits re-fitted on the second wiring draw, against the same models on the real wiring '
        '(capped structure fits at 6.8 %, breadth exponent 2; filled = real wiring, open = draw 2)', fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(res_dir / f'{EXPERIMENT}.png', dpi=130)
    plt.close(fig)


def _cell(sub: pd.DataFrame, model: str, wiring: str, op: str, col: str, fmt: str) -> str:
    r = sub[(sub.model == model) & (sub.wiring == wiring) & (sub.operating_point == op)]
    if len(r) == 1 and pd.notna(r[col].iloc[0]):
        return format(float(r[col].iloc[0]), fmt)
    return '-'


def markdown_tables(df: pd.DataFrame, data: Dict[str, float]) -> str:
    """Per threshold: (1) the 6.8 % fits, statistic | KC data | each model on the real wiring and
    on draw 2; (2) the draw's cost (draw 2 minus real wiring) at 5 % and at 6.8 % per model."""
    out = []
    for thr in (70.0, 40.0):
        sub = df[df.claw_thr == thr]
        s68 = sub[sub.operating_point == '6.8 %']
        models = [m for m in MODELS if (s68.model == m).any()]
        wirings = {m: tuple(w for w in ('real', 'draw 2') if ((s68.model == m) & (s68.wiring == w)).any()) for m in models}
        head1 = '| | | ' + ' | '.join(f'{m} {{span={len(wirings[m])}}}' if len(wirings[m]) > 1 else m for m in models) + ' |'
        head2 = '| statistic | KC data | ' + ' | '.join(' | '.join('real wiring' if w == 'real' else w for w in wirings[m]) for m in models) + ' |'
        sep = '|---|---|' + '---|' * sum(len(wirings[m]) for m in models)
        rows = [f'**{thr:g} Hz per-claw threshold, fitted at the control flies\' 6.8 %**', '', head1, head2, sep]
        for col, label, fmt, dkey in TABLE_ROWS:
            cells = [_cell(s68, m, w, '6.8 %', col, fmt) for m in models for w in wirings[m]]
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
        # the draw's cost at both operating points
        pairs = []
        for m in models:
            for op in ('5 %', '6.8 %'):
                have = {w for w in ('real', 'draw 2') if ((sub.model == m) & (sub.wiring == w) & (sub.operating_point == op)).any()}
                if have == {'real', 'draw 2'}:
                    pairs.append((m, op))
        if not pairs:
            continue
        head1 = '| | ' + ' | '.join(f'{m} {{span={n}}}' if n > 1 else m for m, n in
            [(m, sum(1 for mm, _ in pairs if mm == m)) for m in dict.fromkeys(mm for mm, _ in pairs)]) + ' |'
        head2 = '| statistic, draw 2 minus real wiring | ' + ' | '.join(f'at {op}' for _, op in pairs) + ' |'
        sep = '|---|' + '---|' * len(pairs)
        rows = [f'**{thr:g} Hz per-claw threshold: what the draw costs each model**', '', head1, head2, sep]
        for col, label, fmt in DELTA_ROWS:
            cells = []
            for m, op in pairs:
                a = sub[(sub.model == m) & (sub.wiring == 'real') & (sub.operating_point == op)][col]
                b = sub[(sub.model == m) & (sub.wiring == 'draw 2') & (sub.operating_point == op)][col]
                cells.append(format(float(b.iloc[0] - a.iloc[0]), '+' + fmt.lstrip('+')) if len(a) == 1 and len(b) == 1
                    and pd.notna(a.iloc[0]) and pd.notna(b.iloc[0]) else '-')
            rows.append(f'| {label} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
    return '\n\n'.join(out) + '\n'


def main() -> None:
    res_dir = H.results_dir(EXPERIMENT)
    df = load_rows()
    data = data_values()
    df.to_csv(res_dir / 'comparison.csv', index=False)
    (res_dir / 'comparison_table.md').write_text(markdown_tables(df, data))
    pd.set_option('display.width', 250)
    cols = ['experiment', 'model', 'wiring', 'operating_point', 'claw_thr', 'loss', 'loss_no_rate', 'fit_block',
        'fit_on_spearman', 'fit_off_spearman', 'fit_po_on_spearman', 'fit_po_off_spearman', 'fit_wt_dev_on',
        'fit_vcf_on', 'fit_vcf_off', 'vcf_on_minus_off', 'rate_ratio_off_over_on', 'weak_share_off']
    print(df[cols].round(3).to_string(index=False))
    print('data:', {k: round(v, 3) for k, v in data.items()})
    make_figure(df, data, res_dir)
    print('->', res_dir)


if __name__ == '__main__':
    main()
