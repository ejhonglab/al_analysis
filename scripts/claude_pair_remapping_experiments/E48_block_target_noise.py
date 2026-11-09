#!/usr/bin/env python3
"""E48: the block target through the noise. Comparison table and figures.

Reads the fits (results/E48_block_target_noise/fits.csv: the structure fits at the control flies'
6.8 % response rate with the loss's block target at the data's noise-corrected +0.072, and the +0.10
probe; their +0.055-target counterparts in E44_operating_point / E45_gain_at_ctrl_frac) and the
trial-noise pipeline's results on every one of them (results/E48_block_target_noise/partB/, made by
E43_model_trial_noise.py --part B --model-from ...: per-trial KC threshold jitter at the sigma that
matches the data's single-trial same-odor floor, 10 pseudo-recordings of 3 trials, processed as the
KC data), plus E43's run of the 5 % reference model (results/E43_model_trial_noise/partB/, model
ref40), and writes:
- comparison.csv / comparison_table.md: per model, the deterministic block and the block through the
  noise (the three-trial means, and the grand mean over all 30 trials), the matched sigma and the
  floors reached, the per-trial response rate, rho on odor-odor distances and the fitted scales;
- E48_block_vs_processed.png: the processed block against the deterministic block per model;
- E48_block_pairs.png: for the nine fruity x small-ester pairs, each pair's excess of the APL-on
  distance over the isotonic trend from the APL-off distance, deterministic against processed, per
  model, so the pairs whose block the noise removes can be seen.
Models whose pipeline results are not in yet are listed as pending and left out of the figures.
Nothing is simulated.
Usage: ./venv/bin/python E48_block_target_noise.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E48_block_target_noise'
V = 'spike_counts_cosine__'
# (label, experiment, fits.csv tag, pipeline model name, part-B folder)
E43B = H.results_dir('E43_model_trial_noise') / 'partB'
E48B = H.results_dir(EXPERIMENT) / 'partB'
MODELS = [
    ('the reference model (5 %)', 'E34_gain_bounded', 'thr40_gain1.5_k2_cap3', 'ref40', E43B),
    ('eight at 1.5, 40 Hz, target +0.055', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3', 'E44-thr40_gain1.5_rate0.068_k2_cap3', E48B),
    ('eight at 1.5, 40 Hz, target +0.072', EXPERIMENT, 'thr40_gain1.5_rate0.068_block0.072_k2_cap3', 'E48-thr40_gain1.5_rate0.068_block0.072_k2_cap3', E48B),
    ('eight at 1.5, 70 Hz, target +0.055', 'E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3', 'E44-thr70_gain1.5_rate0.068_k2_cap3', E48B),
    ('eight at 1.5, 70 Hz, target +0.072', EXPERIMENT, 'thr70_gain1.5_rate0.068_block0.072_k2_cap3', 'E48-thr70_gain1.5_rate0.068_block0.072_k2_cap3', E48B),
    ('eight at 1.25, 70 Hz, target +0.055', 'E45_gain_at_ctrl_frac', 'thr70_gain1.25_rate0.068_k2_cap3', 'E45-thr70_gain1.25_rate0.068_k2_cap3', E48B),
    ('eight at 1.25, 70 Hz, target +0.072', EXPERIMENT, 'thr70_gain1.25_rate0.068_block0.072_k2_cap3', 'E48-thr70_gain1.25_rate0.068_block0.072_k2_cap3', E48B),
    ('eight at 1.25, 70 Hz, target +0.10 (probe)', EXPERIMENT, 'thr70_gain1.25_rate0.068_block0.100_k2_cap3', 'E48-thr70_gain1.25_rate0.068_block0.100_k2_cap3', E48B),
    # the clean repeat (2026-11-08): the same re-fits started exactly at their +0.055-target seeds (--init-from-exact;
    # the --init-from seeding had reset the bouton->APL input scale, unresolved_issues.md)
    ('eight at 1.5, 40 Hz, target +0.072, exact start', EXPERIMENT, 'thr40_gain1.5_rate0.068_block0.072_k2_cap3_exact', 'E48-thr40_gain1.5_rate0.068_block0.072_k2_cap3_exact', E48B),
    ('eight at 1.5, 70 Hz, target +0.072, exact start', EXPERIMENT, 'thr70_gain1.5_rate0.068_block0.072_k2_cap3_exact', 'E48-thr70_gain1.5_rate0.068_block0.072_k2_cap3_exact', E48B),
    ('eight at 1.25, 70 Hz, target +0.072, exact start', EXPERIMENT, 'thr70_gain1.25_rate0.068_block0.072_k2_cap3_exact', 'E48-thr70_gain1.25_rate0.068_block0.072_k2_cap3_exact', E48B),
]
BLOCK_PAIRS = [(a, b) for a in ('2h', 'IaA', 'pa') for b in ('2-but', 'eb', 'ep')]


def level_of(df: pd.DataFrame, model: str) -> str | None:
    """The APL-on-matched level of a model in a pipeline table ('match_on', or the joint label
    when the two matched sigmas coincided), or None if the model is not in the table."""
    lv = df.loc[df['model'] == model, 'level'].unique()
    for cand in ('match_on', 'match_on+match_off'):
        if cand in lv:
            return cand
    return None


def pair_series(pairs: pd.DataFrame, col: str) -> pd.Series:
    s = pairs.set_index(['odor_a', 'odor_b'])[col]
    s.index.names = [None, None]
    return s


def pair_excess(pairs: pd.DataFrame, col: str) -> pd.Series:
    """Per pair, the APL-on distance's residual from the isotonic fit on the APL-off distance."""
    on = pair_series(pairs[pairs['condition'] == 'on'], col)
    off = pair_series(pairs[pairs['condition'] == 'acute_off'], col)
    resid, _ = H.isotonic_resid(on, off)
    return resid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=None, help='output folder (default: the results folder)')
    args = ap.parse_args()
    out = args.out or H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    dt = H.data_target
    inputs = {}
    fits = {}
    for exp in sorted({m[1] for m in MODELS}):
        f = H.results_dir(exp) / 'fits.csv'
        inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
        fits[exp] = pd.read_csv(f).set_index('tag')
    tables = {}
    for folder in (E43B, E48B):
        t = {}
        for name in ('stats_noise', 'onoff_noise', 'pairs_noise_long'):
            f = folder / f'{name}.csv'
            if f.exists():
                inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
                t[name] = pd.read_csv(f)
        tables[folder] = t
    rows = []
    excess = {}
    for label, exp, tag, mname, folder in MODELS:
        fit = fits[exp].loc[tag] if tag in fits[exp].index else None
        row = dict(model=label, experiment=exp, tag=tag, pipeline_model=mname,
            block_target=(float(fit['block_target']) if fit is not None and 'block_target' in fit.index and pd.notna(fit['block_target']) else dt('block')),
            loss_recorded=(float(fit['loss']) if fit is not None else np.nan))
        if fit is not None:
            row.update({k: float(fit[k]) for k in ('fit_block', 'fit_on_spearman', 'fit_off_spearman', 'fit_po_on_spearman', 'fit_po_off_spearman',
                'fit_wt_dev_on', 'fit_vcf_off', 'thr_factor', 'apl_factor_vs_e3', 'aplpn_factor_vs_e3', 'kc_kc_inh', 'fixed_thr') if k in fit.index})
            row['vcf_on'] = float(fit[V + 'on_spearman_resid_vs_vcf_hyp']) if V + 'on_spearman_resid_vs_vcf_hyp' in fit.index else np.nan
            # the (unfitted) balance between the APL's bouton and KC inputs, which the --init-from seeding changed
            if 'wPNAPL_scale' in fit.index and 'wKCAPL_scale' in fit.index and float(fit['wKCAPL_scale']) != 0:
                row['wPNAPL_over_wKCAPL'] = float(fit['wPNAPL_scale']) / float(fit['wKCAPL_scale'])
        t = tables[folder]
        lv = level_of(t['stats_noise'], mname) if 'stats_noise' in t else None
        if lv is None:
            row['pipeline'] = 'pending'
            rows.append(row)
            continue
        row['pipeline'] = 'done'
        s = t['stats_noise']; o = t['onoff_noise']; p = t['pairs_noise_long']
        g = s[(s['model'] == mname) & (s['level'] == lv) & (s['transform'] == 'graded')].set_index('condition')
        oo = o[(o['model'] == mname) & (o['transform'] == 'graded')].set_index(['level', 'matrix'])
        row.update(sigma=float(g.at['on', 'sigma']), floor_on=float(g.at['on', 'single_trial_floor']), floor_off=float(g.at['acute_off', 'single_trial_floor']),
            trial_mean_floor_on=float(g.at['on', 'trial_mean_floor']), rate_trial_on=float(g.at['on', 'rate_trial']), rate_trial_off=float(g.at['acute_off', 'rate_trial']),
            rate_det_on=float(g.at['on', 'rate_det']), thr_retuned_x=float(g.at['on', 'fixed_thr_ratio']) if 'fixed_thr_ratio' in g else np.nan,
            block_det=float(oo.at[('deterministic', 'deterministic'), 'block']), block_raw=float(oo.at[(lv, 'raw'), 'block']),
            block_grand=float(oo.at[(lv, 'grand'), 'block']), block_corrected=float(oo.at[(lv, 'corrected'), 'block']),
            iso_r2_raw=float(oo.at[(lv, 'raw'), 'iso_r2']), iso_r2_det=float(oo.at[('deterministic', 'deterministic'), 'iso_r2']),
            mean_dist_raw_on=float(g.at['on', 'mean_dist_raw']), mean_dist_det_on=float(g.at['on', 'mean_dist_det']),
            rho_raw_on=float(g.at['on', 'rho_raw_vs_data']), rho_raw_off=float(g.at['acute_off', 'rho_raw_vs_data']),
            rho_det_on=float(g.at['on', 'rho_det_vs_data']), rho_det_off=float(g.at['acute_off', 'rho_det_vs_data']),
            rate_rho_trial_on=float(g.at['on', 'rate_rho_vs_data_trial']) if 'rate_rho_vs_data_trial' in g else np.nan,
            n_corrected_out_of_range_on=int(g.at['on', 'n_corrected_pairs_out_of_range']) if 'n_corrected_pairs_out_of_range' in g else -1)
        pm = p[(p['model'] == mname) & (p['level'] == lv) & (p['transform'] == 'graded')]
        excess[label] = pd.DataFrame({'deterministic': pair_excess(pm, 'det'), 'three-trial means': pair_excess(pm, 'raw'), 'grand mean': pair_excess(pm, 'grand')})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'comparison.csv', index=False)
    # data row for the pair excess (from the data columns of any pipeline table)
    anyp = tables[E48B].get('pairs_noise_long')
    if anyp is not None and len(anyp):
        m0 = anyp['model'].iloc[0]; lv0 = level_of(anyp, m0)
        pm = anyp[(anyp['model'] == m0) & (anyp['level'] == lv0) & (anyp['transform'] == 'graded')]
        excess['KC data'] = pd.DataFrame({'measured': pair_excess(pm, 'data_raw'), 'noise-corrected': pair_excess(pm, 'data_corrected')})
    # markdown table
    data_block_corr = None
    o48 = tables[E48B].get('onoff_noise')
    if o48 is not None:
        d = o48[(o48['model'] == 'data') & (o48['transform'] == 'graded')].set_index('matrix')
        data_block_corr = float(d.at['corrected', 'block'])
    lines = ['| model (structure fits at 6.8 %, APL->bouton cap 3; the 5 % reference model for comparison) | block target | extended loss (as fitted) | bouton->APL / KC->APL input scale ratio | block, deterministic | block through the noise, three-trial means | block, grand mean of 30 trials | block, noise-corrected | matched σ | single-trial floor reached, APL on / off | response rate per trial, APL on / off | ρ on odor-odor distances, APL on / off (noisy) | response rate ρ, APL on (noisy) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in df.iterrows():
        ratio = f"{r['wPNAPL_over_wKCAPL']:.2f}" if 'wPNAPL_over_wKCAPL' in r.index and pd.notna(r['wPNAPL_over_wKCAPL']) else ''
        if r['pipeline'] != 'done':
            lines.append(f"| {r['model']} | {r['block_target']:+.3f} | {r['loss_recorded']:.2f} | {ratio} | {r.get('fit_block', np.nan):+.3f} | (pending) | | | | | | | |")
            continue
        lines.append(f"| {r['model']} | {r['block_target']:+.3f} | {r['loss_recorded']:.2f} | {ratio} | {r['block_det']:+.3f} | {r['block_raw']:+.3f} | {r['block_grand']:+.3f} | "
            f"{r['block_corrected']:+.3f} | {r['sigma']:.3f} | {r['floor_on']:.2f} / {r['floor_off']:.2f} | {r['rate_trial_on']:.3f} / {r['rate_trial_off']:.3f} | "
            f"{r['rho_raw_on']:.2f} / {r['rho_raw_off']:.2f} | {r['rate_rho_trial_on']:.2f} |")
    lines.append(f"| KC data | | | | {dt('block'):+.3f} measured{'' if data_block_corr is None else f', {data_block_corr:+.3f} noise-corrected'} | | | | | "
        f"{dt('ctrl_single_trial_floor'):.2f} / {dt('tnt_single_trial_floor'):.2f} | {dt('ctrl_frac'):.3f} / {dt('tnt_frac'):.3f} | ceiling {dt('ctrl_ceiling'):.2f} / {dt('tnt_ceiling'):.2f} | ceiling {dt('ctrl_rate_rho_ceiling'):.2f} |")
    pend = [r['model'] for _, r in df.iterrows() if r['pipeline'] != 'done']
    if pend:
        lines.append('')
        lines.append('Pipeline results still pending: ' + '; '.join(pend) + '.')
    (out / 'comparison_table.md').write_text('\n'.join(lines) + '\n')
    make_figures(df, excess, out, data_block_corr)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E48_block_target_noise.py',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        inputs_sha256=inputs, models=[dict(label=m[0], experiment=m[1], tag=m[2], pipeline_model=m[3]) for m in MODELS],
        pending=pend)
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n')
    print('\n'.join(lines))
    print('->', out)


def make_figures(df: pd.DataFrame, excess: dict, out: Path, data_block_corr) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    done = df[df['pipeline'] == 'done']
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for _, r in done.iterrows():
        ax.plot([r['block_det']], [r['block_raw']], 'o', color='#2b6cb0', ms=6)
        ax.plot([r['block_det']], [r['block_grand']], 's', color='#d9480f', ms=5, mfc='none')
        ax.annotate(r['model'].replace(', target ', '\ntarget '), (r['block_det'], r['block_raw']), fontsize=6.5, xytext=(4, 3), textcoords='offset points')
    ax.plot([0, 0.08], [0, 0.08], ':', color='0.5', lw=1)
    ax.axhline(dt('block'), color='k', lw=1, ls='--')
    ax.text(0.001, dt('block') + 0.001, 'KC data, measured (three-trial means)', fontsize=7)
    if data_block_corr is not None:
        ax.axvline(data_block_corr, color='0.4', lw=1, ls='-.')
        ax.text(data_block_corr + 0.001, 0.002, 'KC data,\nnoise-corrected', fontsize=7)
    ax.plot([], [], 'o', color='#2b6cb0', label='three-trial means (as the data are measured)')
    ax.plot([], [], 's', color='#d9480f', mfc='none', label='grand mean over 30 trials')
    ax.set_xlabel('fruity × small-ester block, deterministic model'); ax.set_ylabel('block after trial-to-trial noise, processed as the data')
    ax.set_xlim(0, 0.08); ax.set_ylim(-0.005, 0.08); ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    ax.set_title('The block through the noise: each fit at its matched KC threshold jitter', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E48_block_vs_processed.png', dpi=130); plt.close(fig)
    # per-pair excess
    labels = [k for k in excess if k != 'KC data']
    if not labels:
        return
    panels = labels + ['KC data']
    ncols = min(len(panels), 6)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 4.2 * nrows), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    pair_names = [f'{a} / {b}' for a, b in BLOCK_PAIRS]
    for ax, lab in zip(axes, panels):
        e = excess[lab].reindex(BLOCK_PAIRS)
        x = np.arange(len(BLOCK_PAIRS))
        for i, (col, c, mk) in enumerate(zip(e.columns, ['0.2', '#2b6cb0', '#d9480f'], ['o', 's', '^'])):
            ax.plot(x, e[col].to_numpy(), mk, color=c, ms=5, label=col, mfc=('none' if i == 2 else None))
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(pair_names, rotation=70, fontsize=7)
        ax.set_title(lab.replace(', target ', '\ntarget '), fontsize=8); ax.legend(fontsize=6, frameon=False)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
    for ax in axes[::ncols]:
        ax.set_ylabel('excess of the APL-on distance over the\nisotonic trend from the APL-off distance')
    fig.suptitle('The nine fruity × small-ester pairs: deterministic against processed through the matched noise', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E48_block_pairs.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
