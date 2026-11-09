#!/usr/bin/env python3
"""E50: the fitted block on more shuffle draws.

The wiring-feature experiment (E47) re-fitted the 40 Hz reference model (the eight glomeruli
at 1.5, at the control flies' 6.8 %) on one draw (seed 2) of each shuffle class and found the
classes ranked by what they keep of the fruity x small-ester block: the plain claw permutation
nearly kept it (+0.055), the within-type Curveball lost most of it (+0.027). One draw per class
cannot tell the class from the draw. This experiment re-fits the same model, from the same
start (E44's 40 Hz optimum, `--init-from`), on seeds 1, 3 and 4 of the plain claw permutation
and of the within-type Curveball; the six fits run through

    E10_joint_fit.py --experiment E50_block_more_draws --loss structure -k 2 --aplpn-max 3
        --rate-target ctrl_frac --maxfev 150 --claw-thr 40 --glom-gain 1.5 --wiring <S>
        --wiring-seed <n> --init-from thr40_gain1.5_rate0.068_k2_cap3
        --init-from-experiment E44_operating_point --tag _cap3

(outputs/E50_block_more_draws/run_hal.sh on the host that ran them) and land in this
experiment's fits.csv. This script simulates nothing: it puts the four draws per class (seed 2
from E47's fits.csv) beside the real-wiring fit (E44) and E47 part A's fixed-scale means
(the same model at its fitted scales with only the threshold re-tuned, five draws per class),
tests whether the two classes differ beyond their draw-to-draw spread, and draws the figure.

Outputs (<shared>/results/E50_block_more_draws/): comparison.csv (one row per re-fit, every
draw of both classes plus E47's Curveball seed 2 and the real-wiring fit, with the statistics
under the names E47's tables use), per_class.csv (mean, sd, min, max, n per class),
class_contrast.csv (the two classes' means, their difference and a Welch t-test per statistic),
comparison_table.md and E50_block_more_draws.png.
Usage: ./venv/bin/python E50_block_more_draws.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sps

import kc_remapping_harness as H
from E47_wiring_features import CLASS_LABEL, MODELS, STATS as E47_STATS, V, data_values

EXPERIMENT = 'E50_block_more_draws'
E47 = 'E47_wiring_features'
MODEL = 'thr40_gain1.5'
REAL_EXP, REAL_TAG = MODELS[MODEL][:2]
CLASSES = ('claw_permutation', 'curveball_within_type')
# statistics: column (E47's names), label, format, data-target key
STATS = [s for s in E47_STATS if s[0] != 'thr_factor_vs_fit'] + [
    ('fixed_thr', 'fitted somatic threshold', '.0f', None),
    ('wAPLKC_scale', 'fitted APL→KC scale', '.3f', None),
    ('wAPLPN_scale', 'fitted APL→bouton scale', '.2f', None),
    ('kc_kc_inh', 'fitted KC–KC inhibition', '.1f', None),
    ('n_evals', 'loss evaluations', '.0f', None),
]
FIT_ONLY = {'fixed_thr', 'wAPLKC_scale', 'wAPLPN_scale', 'kc_kc_inh', 'n_evals'}


def fit_rows(fits: pd.DataFrame, source: str) -> List[Dict]:
    """fits.csv rows as E47's comparison rows (part B naming), plus the fitted parameters."""
    rows = []
    for r in fits.to_dict('records'):
        vcf_on = r.get('spike_counts_cosine__on_spearman_resid_vs_vcf_hyp', np.nan)
        wiring = r.get('wiring')
        wiring = 'real' if not isinstance(wiring, str) else wiring
        rows.append(dict(part='B', source=source, tag=r['tag'], model=f"thr{r['claw_thr']:g}_gain{r['gain']:g}",
            shuffle=wiring, seed=(np.nan if wiring == 'real' else int(r['wiring_seed'])), claw_thr=r['claw_thr'],
            gain=r['gain'], loss_rate_ctrl=r['loss'], **{f'{V}__block_fruity_small_iso': r['fit_block'],
            f'{V}__on_spearman': r['fit_on_spearman'], f'{V}__off_spearman': r['fit_off_spearman'],
            'spearman_rate_on_vs_data': r['fit_po_on_spearman'], 'spearman_rate_off_vs_data': r['fit_po_off_spearman'],
            f'{V}__on_spearman_resid_vs_remap_wt': r['fit_wt_dev_on'], f'{V}__on_spearman_resid_vs_vcf_hyp': vcf_on,
            f'{V}__off_spearman_resid_vs_vcf_hyp': r['fit_vcf_off'], 'vcf_on_minus_off': vcf_on - r['fit_vcf_off'],
            'rate_ratio_off_over_on': r['rate_ratio_off_over_on'], 'weak_share_off': np.nan},
            **{k: r[k] for k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale',
                'kc_kc_inh', 'n_evals', 'initial_loss', 'seconds', 'init_from')}))
    return rows


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(res_dir: Path) -> pd.DataFrame:
    """This experiment's re-fits, E47's seed-2 re-fits of the same model and the real-wiring fit."""
    shared = H.shared_dir() / 'results'
    rows = []
    own = res_dir / 'fits.csv'
    if own.exists():
        rows += fit_rows(pd.read_csv(own), EXPERIMENT)
    e47 = pd.read_csv(shared / E47 / 'fits.csv')
    rows += fit_rows(e47[e47.claw_thr == MODELS[MODEL][2]], E47)
    real = pd.read_csv(shared / REAL_EXP / 'fits.csv')
    rows += fit_rows(real[real.tag == REAL_TAG], REAL_EXP)
    comp = pd.DataFrame(rows)
    comp = comp[comp.model == MODEL].reset_index(drop=True)
    return comp


def per_class(comp: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c, *_ in STATS if c in comp.columns]
    fits = comp[comp.shuffle != 'real']
    g = fits.groupby('shuffle')[cols].agg(['mean', 'std', 'count', 'min', 'max'])
    g.columns = [f'{c}__{s}' for c, s in g.columns]
    g = g.reset_index()
    g.insert(1, 'seeds', [' '.join(str(int(s)) for s in sorted(fits[fits.shuffle == sh].seed)) for sh in g.shuffle])
    return g


def class_contrast(comp: pd.DataFrame) -> pd.DataFrame:
    """The two classes' means per statistic, their difference (claw permutation minus within-type
    Curveball), whether the draws' ranges overlap, and a Welch t-test (tiny n: descriptive)."""
    rows = []
    a = comp[comp.shuffle == CLASSES[0]]; b = comp[comp.shuffle == CLASSES[1]]
    for col, label, fmt, dkey in STATS:
        if col not in comp.columns:
            continue
        x = a[col].astype(float).dropna(); y = b[col].astype(float).dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        t, p = sps.ttest_ind(x, y, equal_var=False)
        rows.append(dict(statistic=col, label=label, n_claw_permutation=len(x), n_curveball_within_type=len(y),
            mean_claw_permutation=x.mean(), sd_claw_permutation=x.std(ddof=1), mean_curveball_within_type=y.mean(),
            sd_curveball_within_type=y.std(ddof=1), difference=x.mean() - y.mean(),
            ranges_overlap=bool(max(x.min(), y.min()) <= min(x.max(), y.max())), welch_t=float(t), welch_p=float(p),
            pooled_sd=float(np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2))))
    return pd.DataFrame(rows)


def fmt_ms(v: pd.Series, fmt: str) -> str:
    v = v.astype(float).dropna()
    if len(v) > 1:
        return f'{format(v.mean(), fmt)} ± {format(v.std(ddof=1), fmt.lstrip("+"))}'
    return format(float(v.iloc[0]), fmt) if len(v) == 1 else '-'


def markdown_tables(comp: pd.DataFrame, e47_classes: pd.DataFrame, data: Dict[str, float]) -> str:
    out = []
    real = comp[comp.shuffle == 'real']
    n = {s: int((comp.shuffle == s).sum()) for s in CLASSES}
    a_n = {s: int(e47_classes[e47_classes.shuffle == s][f'{V}__block_fruity_small_iso__count'].iloc[0]) for s in CLASSES}
    head0 = ('| | | | ' + ' | '.join(f'{CLASS_LABEL[s]} {{span=2}}' for s in CLASSES) + ' |')
    head1 = ('| statistic | KC data | real wiring, fitted | ' +
        ' | '.join(f'fixed scales, threshold re-tuned (n = {a_n[s]}) | re-fitted per draw (n = {n[s]})' for s in CLASSES) + ' |')
    rows = ['**The eight glomeruli at 1.5, 40 Hz per-claw threshold, at 6.8 %: the real-wiring fit, the same model at its '
        'fitted scales on five draws of each class (threshold re-tuned; the wiring-feature experiment\'s part A), and '
        'the model re-fitted on each draw (seeds 1-4; mean ± sd over draws)**', '', head0, head1, '|---|---|---|---|---|---|---|']
    for col, label, fmt, dkey in STATS:
        if col not in comp.columns:
            continue
        d = '-'
        if dkey and dkey in data:
            d = format(data[dkey], fmt)
            if 'ceiling' in dkey:
                d = f'ceiling {d}'
        cells = [fmt_ms(real[col], fmt)]
        for s in CLASSES:
            ec = e47_classes[e47_classes.shuffle == s]
            if col in FIT_ONLY or ec.empty or f'{col}__mean' not in ec.columns or pd.isna(ec[f'{col}__mean'].iloc[0]):
                cells.append('-')
            else:
                cells.append(f"{format(float(ec[f'{col}__mean'].iloc[0]), fmt)} ± {format(float(ec[f'{col}__std'].iloc[0]), fmt.lstrip('+'))}")
            cells.append(fmt_ms(comp[comp.shuffle == s][col], fmt))
        rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
    out.append('\n'.join(rows))
    # one row per re-fit
    draws = comp[comp.shuffle != 'real'].copy()
    draws['order'] = draws.shuffle.map({'claw_permutation': 0, 'curveball_within_type': 1, 'curveball': 2})
    draws = draws.sort_values(['order', 'seed'])
    rows = ['**Each re-fit (the real-wiring fit first)**', '',
        '| wiring | seed | fruity × small-ester block | extended loss | ρ on odor-odor distances, APL on | APL off | response rate ρ, APL on | APL off | structure vs wild-type somata | natural-source alignment, APL on / off / on minus off | somatic threshold | APL→KC scale | APL→bouton scale | KC–KC inhibition | loss evaluations |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in pd.concat([real, draws]).to_dict('records'):
        rows.append(f"| {CLASS_LABEL.get(r['shuffle'], r['shuffle'])} | {'-' if pd.isna(r['seed']) else int(r['seed'])} | "
            f"{r[f'{V}__block_fruity_small_iso']:+.3f} | {r['loss_rate_ctrl']:.2f} | {r[f'{V}__on_spearman']:.2f} | {r[f'{V}__off_spearman']:.2f} | "
            f"{r['spearman_rate_on_vs_data']:.2f} | {r['spearman_rate_off_vs_data']:.2f} | {r[f'{V}__on_spearman_resid_vs_remap_wt']:.2f} | "
            f"{r[f'{V}__on_spearman_resid_vs_vcf_hyp']:.2f} / {r[f'{V}__off_spearman_resid_vs_vcf_hyp']:.2f} / {r['vcf_on_minus_off']:+.2f} | "
            f"{r['fixed_thr']:.0f} | {r['wAPLKC_scale']:.3f} | {r['wAPLPN_scale']:.2f} | {r['kc_kc_inh']:.1f} | {int(r['n_evals'])} |")
    out.append('\n'.join(rows))
    return '\n\n'.join(out) + '\n'


def make_figure(comp: pd.DataFrame, e47_classes: pd.DataFrame, data: Dict[str, float], res_dir: Path) -> None:
    plt = H._plt()
    from matplotlib.lines import Line2D
    classes = ['real'] + [s for s in (*CLASSES, 'curveball') if (comp.shuffle == s).any()]
    panels = [s for s in STATS if s[0] in comp.columns]
    fig, axes = plt.subplots(3, 5, figsize=(22, 12.5))
    for ax, (col, label, fmt, dkey) in zip(axes.flat, panels):
        for i, s in enumerate(classes):
            g = comp[comp.shuffle == s]
            v = g[col].astype(float).dropna()
            if s == 'real':
                if not v.empty:
                    ax.scatter([i], v, marker='_', s=220, color='k', linewidths=2.5, zorder=4)
            elif not v.empty:
                ax.scatter(np.full(len(v), i + 0.12) + np.linspace(-0.05, 0.05, len(v)), v, s=22, color='#d1662a', zorder=3)
                for x_, y_, sd_ in zip(np.full(len(v), i + 0.12) + np.linspace(-0.05, 0.05, len(v)), v, g.loc[v.index, 'seed']):
                    ax.annotate(f'{int(sd_)}', (x_, y_), textcoords='offset points', xytext=(5, -2), fontsize=6.5, color='#d1662a')
                if len(v) > 1:
                    ax.errorbar([i + 0.12], [v.mean()], yerr=[v.std(ddof=1)], fmt='_', color='#d1662a', ms=14, mew=2, capsize=3, zorder=4)
            if s in CLASSES and col not in FIT_ONLY:
                ec = e47_classes[e47_classes.shuffle == s]
                if not ec.empty and f'{col}__mean' in ec.columns and pd.notna(ec[f'{col}__mean'].iloc[0]):
                    ax.errorbar([i - 0.12], [float(ec[f'{col}__mean'].iloc[0])], yerr=[float(ec[f'{col}__std'].iloc[0])], fmt='o',
                        mfc='none', color='#6b6b6b', ms=7, mew=1.5, capsize=3, zorder=3)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='k', ls='--', lw=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([CLASS_LABEL[s] for s in classes], rotation=25, ha='right', fontsize=8)
        ax.set_xlim(-0.6, len(classes) - 0.4)
        ax.set_title(label, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    handles = [Line2D([], [], marker='_', ls='', color='k', ms=14, mew=2.5, label='real wiring, fitted'),
        Line2D([], [], marker='o', ls='', color='#d1662a', label='re-fitted on the draw (points: seeds; bar: mean ± sd)'),
        Line2D([], [], marker='o', ls='', mfc='none', color='#6b6b6b', label='fixed scales, threshold re-tuned (mean ± sd over five draws)'),
        Line2D([], [], ls='--', color='k', label='KC data (target or ceiling)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle('The eight glomeruli at 1.5, 40 Hz per-claw threshold, at 6.8 %: re-fitted on four draws of each shuffle class', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(res_dir / f'{EXPERIMENT}.png', dpi=130)
    plt.close(fig)


def main() -> None:
    res_dir = H.results_dir(EXPERIMENT)
    shared = H.shared_dir() / 'results'
    data = data_values()
    comp = load_rows(res_dir)
    e47_pc = pd.read_csv(shared / E47 / 'per_class.csv')
    e47_classes = e47_pc[e47_pc.model == MODEL]
    comp.to_csv(res_dir / 'comparison.csv', index=False)
    pc = per_class(comp)
    pc.to_csv(res_dir / 'per_class.csv', index=False)
    cc = class_contrast(comp)
    cc.to_csv(res_dir / 'class_contrast.csv', index=False)
    (res_dir / 'comparison_table.md').write_text(markdown_tables(comp, e47_classes, data))
    make_figure(comp, e47_classes, data, res_dir)
    sources = {str(p.relative_to(shared)): sha(p) for p in (res_dir / 'fits.csv', shared / E47 / 'fits.csv',
        shared / E47 / 'per_class.csv', shared / REAL_EXP / 'fits.csv') if p.exists()}
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E50_block_more_draws.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), model=MODEL,
        real_fit=[REAL_EXP, REAL_TAG], classes=list(CLASSES), sources=sources,
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
            cwd=Path(__file__).parent).stdout.strip(),
        note='summary only; the fits are E10_joint_fit.py runs with their own provenance_<tag>.json in this folder')
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n')
    pd.set_option('display.width', 250)
    print((res_dir / 'comparison_table.md').read_text())
    if not cc.empty:
        print(cc[['label', 'mean_claw_permutation', 'mean_curveball_within_type', 'difference', 'ranges_overlap', 'welch_p']]
            .round(3).to_string(index=False))
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
