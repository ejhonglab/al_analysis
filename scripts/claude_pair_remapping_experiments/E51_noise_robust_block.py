#!/usr/bin/env python3
"""E51: the noise-robust block. Comparison table and figures (part B / C write-up).

Reads the noise-aware fits (results/E51_noise_robust_block/fits.csv: E10_joint_fit.py with
--jitter-k 9 --jitter-trials 3 --jitter-thr-scale matched --init-from-exact, the structure loss at
the control flies' 6.8 % response rate whose block term is computed on jittered runs, three
pseudo-recordings of three trials with fixed per-KC threshold-jitter draws), their
deterministic-fitted counterparts (E44_operating_point: the eight glomeruli at 1.5 at the 40 and
70 Hz per-claw thresholds; E45_gain_at_ctrl_frac: the eight at 1.25 at 70 Hz) and the trial-noise
pipeline's results on all of them (results/E51_noise_robust_block/partB/, made by
E43_model_trial_noise.py --part B --model-from <experiment>:<tag> --cal-trials 6 --experiment
E51_noise_robust_block: per-trial KC threshold jitter at the sigma that matches the data's
single-trial same-odor floor, independent draws from the fits', 10 pseudo-recordings of 3 trials,
processed as the KC data), and writes:
- comparison.csv / comparison_table.md: per model, the deterministic block, the block on the fit's
  own jitter draws (the fitted criterion, from fits.csv), the block through the pipeline's
  independent noise (three-trial means; the grand mean over all 30 trials), the matched sigma and
  the floors reached, the per-trial response rates, rho on odor-odor distances and the response
  rate rho (noisy), the natural-source alignment with APL on / off / on minus off (deterministic,
  from fits.csv) and the fitted parameters;
- E51_block_vs_processed.png: the processed block against the deterministic block, each
  noise-fitted model joined to its deterministic-fitted counterpart, with the fit's own-draw block;
- E51_block_pairs.png: for the nine fruity x small-ester pairs, each pair's excess of the APL-on
  distance over the isotonic trend from the APL-off distance, deterministic against processed, per
  model, and in the data;
- E51_fit_histories.png: for each noise-aware fit, the loss and the deterministic and jittered
  blocks over the evaluations (from history_<tag>.csv), i.e. how the optimizer traded the two.
Models whose fits or pipeline results are not in yet are listed as pending and left out of the
figures. Nothing is simulated.
Usage: ./venv/bin/python E51_noise_robust_block.py [--out DIR] [--partb DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E51_noise_robust_block'
V = 'spike_counts_cosine__'
# the deterministic-fitted counterparts: (label, experiment, fits.csv tag, claw_thr, gain)
COUNTERPARTS = [
    ('eight at 1.5, 40 Hz', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3', 40, 1.5),
    ('eight at 1.5, 70 Hz', 'E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3', 70, 1.5),
    ('eight at 1.25, 70 Hz', 'E45_gain_at_ctrl_frac', 'thr70_gain1.25_rate0.068_k2_cap3', 70, 1.25),
]
BLOCK_PAIRS = [(a, b) for a in ('2h', 'IaA', 'pa') for b in ('2-but', 'eb', 'ep')]
FIT_COLS = ('fit_block', 'fit_on_spearman', 'fit_off_spearman', 'fit_po_on_spearman', 'fit_po_off_spearman',
    'fit_wt_dev_on', 'fit_vcf_off', 'fit_rate_on', 'fit_rate_off', 'fit_jit_block', 'fit_jit_rate_on',
    'fit_jit_rate_off', 'fit_jit_on_spearman', 'fit_jit_off_spearman', 'fit_jit_vcf_off', 'thr_factor',
    'apl_factor_vs_e3', 'aplpn_factor_vs_e3', 'kc_kc_inh', 'fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale',
    'wAPLPN_scale', 'wPNAPL_scale', 'n_evals', 'time_s')
JIT_COLS = ('jitter_k', 'jitter_sigma', 'jitter_sigma_source', 'jitter_seed', 'jitter_terms', 'jitter_trials',
    'jitter_thr_scale', 'jitter_thr_scale_source')


def exp_id(experiment: str) -> str:
    return experiment.split('_', 1)[0]


def level_of(df: pd.DataFrame, model: str) -> Optional[str]:
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


def noise_fits_for(fits: pd.DataFrame, claw_thr: int, gain: float) -> list:
    """Tags of the noise-aware fits of one base in E51's fits.csv (tag prefix thr<T>_gain<G>_rate0.068_jitK)."""
    prefix = f'thr{claw_thr}_gain{gain:g}_rate0.068_jitK'
    return [t for t in fits.index if str(t).startswith(prefix)]


def criterion_label(fit: pd.Series) -> str:
    """'jittered block, K 9 in 3 x 3 trials, threshold x1.066 (block term only)' from a fits.csv row."""
    k = int(fit['jitter_k']); tr = int(fit['jitter_trials']) if pd.notna(fit.get('jitter_trials', np.nan)) else k
    sc = float(fit['jitter_thr_scale']) if pd.notna(fit.get('jitter_thr_scale', np.nan)) else 1.0
    terms = str(fit.get('jitter_terms', 'block'))
    s = f'jittered block, K {k} in {k // tr} x {tr} trials' if tr < k else f'jittered block, K {k} as one recording'
    if abs(sc - 1.0) > 1e-9:
        s += f', threshold x{sc:.3f}'
    s += ' (every term)' if terms == 'all' else ' (block term only)'
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=None, help='output folder (default: the results folder)')
    ap.add_argument('--partb', type=Path, default=None,
        help="the pipeline's part-B folder (default: results/E51_noise_robust_block/partB)")
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT)
    out = args.out or res_dir
    out.mkdir(parents=True, exist_ok=True)
    partb = args.partb or (res_dir / 'partB')
    dt = H.data_target
    inputs = {}
    fits = {}
    for exp in sorted({c[1] for c in COUNTERPARTS} | {EXPERIMENT}):
        f = H.results_dir(exp) / 'fits.csv'
        if not f.exists():
            fits[exp] = pd.DataFrame(columns=['tag']).set_index('tag')
            continue
        inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
        fits[exp] = pd.read_csv(f).set_index('tag')
    tables = {}
    for name in ('stats_noise', 'onoff_noise', 'pairs_noise_long'):
        f = partb / f'{name}.csv'
        if f.exists():
            inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
            tables[name] = pd.read_csv(f)
    # the model list: each counterpart, then its noise-aware fits
    # (label, key, experiment, tag, pipeline model name, counterpart label or None); `key` is a
    # stable row key for the summary template (thr<T>_gain<G>_det / _noise[_<i>])
    models = []
    for label, exp, tag, thr, gain in COUNTERPARTS:
        base_key = f'thr{thr}_gain{gain:g}'
        models.append((label + ', deterministic fit', base_key + '_det', exp, tag, f'{exp_id(exp)}-{tag}', None))
        nf = noise_fits_for(fits[EXPERIMENT], thr, gain)
        if not nf:
            models.append((label + ', noise-aware fit (pending)', base_key + '_noise', EXPERIMENT, None, None,
                label + ', deterministic fit'))
        for i, t in enumerate(nf):
            models.append((label + ', noise-aware fit' + (f' ({i + 1})' if len(nf) > 1 else ''),
                base_key + '_noise' + (f'_{i + 1}' if len(nf) > 1 else ''), EXPERIMENT, t,
                f'{exp_id(EXPERIMENT)}-{t}', label + ', deterministic fit'))
    rows = []
    excess = {}
    for label, key, exp, tag, mname, counterpart in models:
        fit = fits[exp].loc[tag] if (tag is not None and tag in fits[exp].index) else None
        row = dict(model=label, key=key, experiment=exp, tag=tag, pipeline_model=mname, counterpart=counterpart,
            fit=('done' if fit is not None else 'pending'))
        if fit is not None:
            row.update(loss_recorded=float(fit['loss']),
                block_target=(float(fit['block_target']) if 'block_target' in fit.index and pd.notna(fit['block_target']) else dt('block')),
                criterion=(criterion_label(fit) if 'jitter_k' in fit.index and pd.notna(fit['jitter_k']) and int(fit['jitter_k']) > 0
                    else 'deterministic block'))
            row.update({k: float(fit[k]) for k in FIT_COLS if k in fit.index and pd.notna(fit[k])})
            row.update({k: fit[k] for k in JIT_COLS if k in fit.index and pd.notna(fit[k])})
            row['vcf_on'] = float(fit[V + 'on_spearman_resid_vs_vcf_hyp']) if V + 'on_spearman_resid_vs_vcf_hyp' in fit.index else np.nan
            row['vcf_on_minus_off'] = row['vcf_on'] - row['fit_vcf_off'] if 'fit_vcf_off' in row else np.nan
            if 'wPNAPL_scale' in row and row.get('wKCAPL_scale', 0) > 0:
                row['wPNAPL_over_wKCAPL'] = row['wPNAPL_scale'] / row['wKCAPL_scale']
        lv = level_of(tables['stats_noise'], mname) if ('stats_noise' in tables and mname is not None) else None
        if lv is None:
            row['pipeline'] = 'pending'
            rows.append(row)
            continue
        row['pipeline'] = 'done'
        s = tables['stats_noise']; o = tables['onoff_noise']; p = tables['pairs_noise_long']
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
    data_block_corr = None
    if 'pairs_noise_long' in tables and len(tables['pairs_noise_long']):
        anyp = tables['pairs_noise_long']
        m0 = anyp['model'].iloc[0]; lv0 = level_of(anyp, m0)
        pm = anyp[(anyp['model'] == m0) & (anyp['level'] == lv0) & (anyp['transform'] == 'graded')]
        excess['KC data'] = pd.DataFrame({'measured': pair_excess(pm, 'data_raw'), 'noise-corrected': pair_excess(pm, 'data_corrected')})
    if 'onoff_noise' in tables:
        o = tables['onoff_noise']
        d = o[(o['model'] == 'data') & (o['transform'] == 'graded')].set_index('matrix')
        if 'corrected' in d.index:
            data_block_corr = float(d.at['corrected', 'block'])
    lines = ['| model (structure fits at 6.8 %, APL->bouton cap 3) | fitting criterion | extended loss (as fitted) | block, deterministic | block on the fit\'s own draws | block through independent noise, three-trial means | block, grand mean of 30 trials | matched σ | single-trial floor reached, APL on / off | response rate per trial, APL on / off | ρ on odor-odor distances, APL on / off (noisy) | response rate ρ, APL on (noisy) | natural-source alignment, APL on / off / on minus off (deterministic) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']

    def fmt(x, spec):
        return '' if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)

    for _, r in df.iterrows():
        if r['fit'] != 'done':
            lines.append(f"| {r['model']} | | | | | | | | | | | | |")
            continue
        vcf = f"{fmt(r.get('vcf_on'), '.2f')} / {fmt(r.get('fit_vcf_off'), '.2f')} / {fmt(r.get('vcf_on_minus_off'), '+.2f')}"
        own = fmt(r.get('fit_jit_block'), '+.3f')
        if r['pipeline'] != 'done':
            # a noise-aware fit records its own-draw response rates; a deterministic fit has none to show here
            own_rates = (f"{fmt(r.get('fit_jit_rate_on'), '.3f')} / {fmt(r.get('fit_jit_rate_off'), '.3f')} (fit's own draws)"
                if pd.notna(r.get('fit_jit_rate_on', np.nan)) else '')
            lines.append(f"| {r['model']} | {r['criterion']} | {r['loss_recorded']:.2f} | {fmt(r.get('fit_block'), '+.3f')} | {own} | (pending) | | | | "
                f"{own_rates} | | | {vcf} |")
            continue
        lines.append(f"| {r['model']} | {r['criterion']} | {r['loss_recorded']:.2f} | {r['block_det']:+.3f} | {own} | {r['block_raw']:+.3f} | {r['block_grand']:+.3f} | "
            f"{r['sigma']:.3f} | {r['floor_on']:.2f} / {r['floor_off']:.2f} | {r['rate_trial_on']:.3f} / {r['rate_trial_off']:.3f} | "
            f"{r['rho_raw_on']:.2f} / {r['rho_raw_off']:.2f} | {fmt(r.get('rate_rho_trial_on'), '.2f')} | {vcf} |")
    lines.append(f"| KC data | | | {dt('block'):+.3f} measured{'' if data_block_corr is None else f', {data_block_corr:+.3f} noise-corrected'} | | | | | "
        f"{dt('ctrl_single_trial_floor'):.2f} / {dt('tnt_single_trial_floor'):.2f} | {dt('ctrl_frac'):.3f} / {dt('tnt_frac'):.3f} | "
        f"ceiling {dt('ctrl_ceiling'):.2f} / {dt('tnt_ceiling'):.2f} | ceiling {dt('ctrl_rate_rho_ceiling'):.2f} | "
        f"{dt('ctrl_vcf'):.2f} / {dt('tnt_vcf'):.2f} / {dt('vcf_apl_effect'):+.2f} |")
    pend_fit = [r['model'] for _, r in df.iterrows() if r['fit'] != 'done']
    pend_pipe = [r['model'] for _, r in df.iterrows() if r['fit'] == 'done' and r['pipeline'] != 'done']
    if pend_fit:
        lines += ['', 'Fits still pending: ' + '; '.join(pend_fit) + '.']
    if pend_pipe:
        lines += ['', 'Pipeline results still pending: ' + '; '.join(pend_pipe) + '.']
    (out / 'comparison_table.md').write_text('\n'.join(lines) + '\n')
    make_figures(df, excess, out, data_block_corr)
    hist_files = make_history_figure(df, res_dir, out)
    for f in hist_files:
        inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E51_noise_robust_block.py',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        inputs_sha256=inputs, partb=str(partb),
        models=[dict(label=m[0], key=m[1], experiment=m[2], tag=m[3], pipeline_model=m[4], counterpart=m[5]) for m in models],
        pending_fits=pend_fit, pending_pipeline=pend_pipe)
    (out / 'provenance_comparison.json').write_text(json.dumps(prov, indent=1) + '\n')
    print('\n'.join(lines))
    print('->', out)


def _short(label: str) -> str:
    return label.replace(', deterministic fit', '\ndeterministic fit').replace(', noise-aware fit', '\nnoise-aware fit')


def make_figures(df: pd.DataFrame, excess: dict, out: Path, data_block_corr) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    done = df[(df['fit'] == 'done') & (df['pipeline'] == 'done')]
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    by_label = {r['model']: r for _, r in done.iterrows()}
    for _, r in done.iterrows():
        noise_fit = r['criterion'] != 'deterministic block'
        c = '#d9480f' if noise_fit else '#2b6cb0'
        ax.plot([r['block_det']], [r['block_raw']], 'o', color=c, ms=6)
        ax.plot([r['block_det']], [r['block_grand']], 's', color=c, ms=5, mfc='none')
        if noise_fit and pd.notna(r.get('fit_jit_block', np.nan)):
            ax.plot([r['block_det']], [r['fit_jit_block']], '^', color=c, ms=6, mfc='none')
        ax.annotate(_short(r['model']), (r['block_det'], r['block_raw']), fontsize=6.5, xytext=(4, 3), textcoords='offset points')
        cp = r.get('counterpart')
        if noise_fit and isinstance(cp, str) and cp in by_label:
            c0 = by_label[cp]
            ax.annotate('', xy=(r['block_det'], r['block_raw']), xytext=(c0['block_det'], c0['block_raw']),
                arrowprops=dict(arrowstyle='->', color='0.45', lw=0.9))
    ax.plot([0, 0.08], [0, 0.08], ':', color='0.5', lw=1)
    ax.axhline(dt('block'), color='k', lw=1, ls='--')
    ax.text(0.001, dt('block') + 0.001, 'KC data, measured (three-trial means)', fontsize=7)
    if data_block_corr is not None:
        ax.axvline(data_block_corr, color='0.4', lw=1, ls='-.')
        ax.text(data_block_corr + 0.001, 0.002, 'KC data,\nnoise-corrected', fontsize=7)
    ax.plot([], [], 'o', color='#2b6cb0', label='deterministic fit: three-trial means through independent noise')
    ax.plot([], [], 'o', color='#d9480f', label='noise-aware fit: three-trial means through independent noise')
    ax.plot([], [], 's', color='0.3', mfc='none', label='grand mean over 30 trials')
    ax.plot([], [], '^', color='#d9480f', mfc='none', label="noise-aware fit: block on the fit's own draws (the criterion)")
    ax.set_xlabel('fruity × small-ester block, deterministic model'); ax.set_ylabel('block after trial-to-trial noise, processed as the data')
    ax.set_xlim(0, 0.08); ax.set_ylim(-0.005, 0.08); ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    ax.set_title('Fitting the block through the noise: each fit at its matched KC threshold jitter', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E51_block_vs_processed.png', dpi=130); plt.close(fig)
    labels = [k for k in excess if k != 'KC data']
    if not labels:
        return
    fig, axes = plt.subplots(1, len(labels) + 1, figsize=(2.6 * (len(labels) + 1), 4.4), sharey=True)
    pair_names = [f'{a} / {b}' for a, b in BLOCK_PAIRS]
    for ax, lab in zip(np.atleast_1d(axes), labels + ['KC data']):
        e = excess[lab].reindex(BLOCK_PAIRS)
        x = np.arange(len(BLOCK_PAIRS))
        for i, (col, c, mk) in enumerate(zip(e.columns, ['0.2', '#2b6cb0', '#d9480f'], ['o', 's', '^'])):
            ax.plot(x, e[col].to_numpy(), mk, color=c, ms=5, label=col, mfc=('none' if i == 2 else None))
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(pair_names, rotation=70, fontsize=7)
        ax.set_title(_short(lab), fontsize=8); ax.legend(fontsize=6, frameon=False)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel('excess of the APL-on distance over the\nisotonic trend from the APL-off distance')
    fig.suptitle('The nine fruity × small-ester pairs: deterministic against processed through the matched noise', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E51_block_pairs.png', dpi=130); plt.close(fig)


def make_history_figure(df: pd.DataFrame, res_dir: Path, out: Path) -> list:
    """Loss and the two blocks over the evaluations of each noise-aware fit; returns the history files read."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    fits = df[(df['fit'] == 'done') & (df['experiment'] == EXPERIMENT)]
    files = [(r['model'], res_dir / f"history_{r['tag']}.csv") for _, r in fits.iterrows()]
    files = [(m, f) for m, f in files if f.exists()]
    if not files:
        return []
    fig, axes = plt.subplots(2, len(files), figsize=(4.2 * len(files), 6.2), squeeze=False)
    for j, (label, f) in enumerate(files):
        h = pd.read_csv(f)
        ax = axes[0, j]
        ax.plot(h.index, h['loss'], color='0.75', lw=0.7, label='loss at each evaluation')
        ax.plot(h.index, h['loss'].cummin(), color='k', lw=1.2, label='best so far')
        ax.set(yscale='log', xlabel='evaluation', ylabel='extended loss', title=_short(label).replace('\n', ', '))
        ax.title.set_fontsize(8.5); ax.legend(fontsize=7, frameon=False)
        ax = axes[1, j]
        if 'block' in h and 'jit_block' in h:
            ax.plot(h.index, h['block'], color='#2b6cb0', lw=0.9, label='deterministic block')
            ax.plot(h.index, h['jit_block'], color='#d9480f', lw=0.9, label="block on the fit's draws (the criterion)")
        ax.axhline(dt('block'), color='k', lw=1, ls='--', label='KC data, measured')
        ax.set(xlabel='evaluation', ylabel='fruity × small-ester block'); ax.legend(fontsize=7, frameon=False)
        for a in axes[:, j]:
            a.spines[['top', 'right']].set_visible(False); a.grid(True, color='0.92', lw=0.6)
    fig.suptitle('The noise-aware fits: loss, and the deterministic against the fitted (jittered) block over the evaluations', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E51_fit_histories.png', dpi=130); plt.close(fig)
    return [f for _, f in files]


if __name__ == '__main__':
    main()
