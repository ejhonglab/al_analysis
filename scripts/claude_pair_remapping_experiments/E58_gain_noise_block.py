#!/usr/bin/env python3
"""E58: the glomerulus gain as the carrier of a noise-robust block. Comparison table and figures.

Reads, per per-claw threshold (40 and 70 Hz), the gain axis of the structure fits at the control
flies' 6.8 %: E45's no-gain and gain-1.25 fits, E44's gain-1.5 fits (the operating point) and E58's
gain-1.75 and gain-2 fits (results/E58_gain_noise_block/fits.csv: E10_joint_fit.py --glom-gain 1.75 /
2, warm-started from the gain-1.5 fit of the same threshold), together with the trial-noise
pipeline's results on all of them (E43_model_trial_noise.py --part B --model-from <experiment>:<tag>
--tune-trials 3 with E48's settings; a model's rows come from the first part-B folder that has
them: E58's own, then E48's, E51's and E53's), and writes to results/E58_gain_noise_block/:
- comparison.csv / comparison_table.md: per model the gain, the fitted KC-KC scale, threshold and
  APL scales, the loss, the deterministic block, the block through independent noise (three-trial
  means and the grand mean over 30 trials), the matched sigma and the floors, the response rate per
  trial with APL on and off, the deterministic response rate rho with APL on and off, rho on
  odor-odor distances through the noise and the natural-source alignment with APL on / off / on
  minus off (deterministic);
- E58_block_vs_gain.png: per threshold, against the gain, the block through the noise (with the
  deterministic block and the grand mean), the fitted KC-KC scale, the APL-off response rate per
  trial and the loss;
- E58_block_pairs.png: the nine fruity x small-ester pairs' excess, deterministic against processed,
  per model and in the data;
- E58_fit_histories.png: loss, block and APL-off response rate over the evaluations of E58's fits;
- with --per-odor, per_odor_rates.csv and E58_per_odor_rates.png: each model's deterministic
  per-odor response rates with APL on and off against the KC data's.
Models whose fit or pipeline result is not in yet are listed as pending and left out of the figures.
Nothing is simulated.
Usage: ./venv/bin/python E58_gain_noise_block.py [--out DIR] [--per-odor]
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
import E51_noise_robust_block as E51
import E53_kckc_held_block as E53

EXPERIMENT = 'E58_gain_noise_block'
V = 'spike_counts_cosine__'
# the gain axis per threshold: (gain, experiment, fits.csv tag); gain 1 = no gain
AXIS = {
    40: [(1.0, 'E45_gain_at_ctrl_frac', 'thr40_rate0.068_k2_cap3'), (1.25, 'E45_gain_at_ctrl_frac', 'thr40_gain1.25_rate0.068_k2_cap3'),
         (1.5, 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'), (1.75, EXPERIMENT, 'thr40_gain1.75_rate0.068_k2_cap3'),
         (2.0, EXPERIMENT, 'thr40_gain2_rate0.068_k2_cap3')],
    70: [(1.0, 'E45_gain_at_ctrl_frac', 'thr70_rate0.068_k2_cap3'), (1.25, 'E45_gain_at_ctrl_frac', 'thr70_gain1.25_rate0.068_k2_cap3'),
         (1.5, 'E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3'), (1.75, EXPERIMENT, 'thr70_gain1.75_rate0.068_k2_cap3'),
         (2.0, EXPERIMENT, 'thr70_gain2_rate0.068_k2_cap3')],
}
THR_LABEL = {40: 'the eight at the 40 Hz per-claw threshold (the reference base)', 70: 'the eight at the 70 Hz per-claw threshold'}


def model_label(thr: int, gain: float) -> str:
    return f'{thr} Hz, ' + ('no gain' if gain == 1.0 else f'gain {gain:g}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=None, help='output folder (default: the results folder)')
    ap.add_argument('--per-odor', action='store_true', help='only the per-odor response-rate diagnostic')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT)
    out = args.out or res_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.per_odor:
        per_odor_rates(out)
        return
    dt = H.data_target
    inputs, fits = {}, {}
    for exp in sorted({m[1] for ax in AXIS.values() for m in ax}):
        f = H.results_dir(exp) / 'fits.csv'
        if not f.exists():
            fits[exp] = pd.DataFrame(columns=['tag']).set_index('tag')
            continue
        inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
        fits[exp] = pd.read_csv(f).set_index('tag')
    folders = [res_dir / 'partB'] + [H.results_dir(e) / 'partB' for e in ('E48_block_target_noise', 'E51_noise_robust_block', 'E53_kckc_held_block')]
    tables, pipeline_source, tin = E53.load_tables(folders)
    inputs.update(tin)
    rows, excess = [], {}
    for thr, axis in AXIS.items():
        for gain, exp, tag in axis:
            fit = fits[exp].loc[tag] if tag in fits[exp].index else None
            mname = f'{E51.exp_id(exp)}-{tag}'
            label = model_label(thr, gain)
            row = dict(model=label, key=f'thr{thr}_gain{gain:g}', thr=thr, gain=gain, experiment=exp, tag=tag, pipeline_model=mname,
                fit=('done' if fit is not None else 'pending'))
            if fit is not None:
                row['loss_recorded'] = float(fit['loss'])
                row.update({k: float(fit[k]) for k in E53.FIT_COLS if k in fit.index and pd.notna(fit[k])})
                row['vcf_on'] = float(fit[V + 'on_spearman_resid_vs_vcf_hyp']) if V + 'on_spearman_resid_vs_vcf_hyp' in fit.index else np.nan
                row['vcf_on_minus_off'] = row['vcf_on'] - row['fit_vcf_off'] if 'fit_vcf_off' in row else np.nan
            pr = E53.pipeline_row(tables, mname)
            if pr is None:
                row['pipeline'] = 'pending'
                rows.append(row)
                continue
            row['pipeline'] = 'done'
            row['pipeline_source'] = pipeline_source.get(mname)
            row.update(pr)
            excess[label] = E53.pair_excess_frame(tables, mname, pr['level'])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'comparison.csv', index=False)
    data_block_corr = None
    if 'pairs_noise_long' in tables and len(tables['pairs_noise_long']):
        anyp = tables['pairs_noise_long']
        m0 = anyp['model'].iloc[0]; lv0 = E51.level_of(anyp, m0)
        pm = anyp[(anyp['model'] == m0) & (anyp['level'] == lv0) & (anyp['transform'] == 'graded')]
        excess['KC data'] = pd.DataFrame({'measured': E51.pair_excess(pm, 'data_raw'), 'noise-corrected': E51.pair_excess(pm, 'data_corrected')})
    if 'onoff_noise' in tables:
        o = tables['onoff_noise']
        d = o[(o['model'] == 'data') & (o['transform'] == 'graded')].set_index('matrix')
        if 'corrected' in d.index:
            data_block_corr = float(d.at['corrected', 'block'])

    def fmt(x, spec, missing=''):
        return missing if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)

    lines = ['| model (structure fits at 6.8 %, APL->bouton cap 3) | gain | KC-KC (fitted) | fixed threshold | APL->KC / APL->bouton scale | loss | block, deterministic | block through independent noise, three-trial means | block, grand mean of 30 trials | matched σ | single-trial floor reached, APL on / off | response rate per trial, APL on / off | response rate ρ, APL on / off (deterministic) | ρ on odor-odor distances, APL on / off (noisy) | natural-source alignment, APL on / off / on minus off (deterministic) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in df.iterrows():
        if r['fit'] != 'done':
            lines.append(f"| {r['model']} (fit pending) | {r['gain']:g} | | | | | | | | | | | | | |")
            continue
        vcf = f"{fmt(r.get('vcf_on'), '.2f', 'n/a')} / {fmt(r.get('fit_vcf_off'), '.2f', 'n/a')} / {fmt(r.get('vcf_on_minus_off'), '+.2f', 'n/a')}"
        po = f"{fmt(r.get('fit_po_on_spearman'), '.2f')} / {fmt(r.get('fit_po_off_spearman'), '.2f')}"
        head = (f"| {r['model']} | {r['gain']:g} | {fmt(r.get('kc_kc_inh'), '.1f')} | {fmt(r.get('fixed_thr'), '.1f')} | "
            f"{fmt(r.get('wAPLKC_scale'), '.3f')} / {fmt(r.get('wAPLPN_scale'), '.3f')} | {r['loss_recorded']:.2f} | ")
        if r['pipeline'] != 'done':
            lines.append(head + f"{fmt(r.get('fit_block'), '+.3f')} | (pending) | | | | {fmt(r.get('fit_rate_on'), '.3f')} / {fmt(r.get('fit_rate_off'), '.3f')} (deterministic) | {po} | | {vcf} |")
            continue
        lines.append(head + f"{r['block_det']:+.3f} | {r['block_raw']:+.3f} | {r['block_grand']:+.3f} | {r['sigma']:.3f} | "
            f"{r['floor_on']:.2f} / {r['floor_off']:.2f} | {r['rate_trial_on']:.3f} / {r['rate_trial_off']:.3f} | {po} | "
            f"{r['rho_raw_on']:.2f} / {r['rho_raw_off']:.2f} | {vcf} |")
    lines.append(f"| KC data | | | | | | {dt('block'):+.3f} measured{'' if data_block_corr is None else f', {data_block_corr:+.3f} noise-corrected'} | | | | "
        f"{dt('ctrl_single_trial_floor'):.2f} / {dt('tnt_single_trial_floor'):.2f} | {dt('ctrl_frac'):.3f} / {dt('tnt_frac'):.3f} | "
        f"ceiling {dt('ctrl_rate_rho_ceiling'):.2f} | ceiling {dt('ctrl_ceiling'):.2f} / {dt('tnt_ceiling'):.2f} | "
        f"{dt('ctrl_vcf'):.2f} / {dt('tnt_vcf'):.2f} / {dt('vcf_apl_effect'):+.2f} |")
    pend_fit = [r['model'] for _, r in df.iterrows() if r['fit'] != 'done']
    pend_pipe = [r['model'] for _, r in df.iterrows() if r['fit'] == 'done' and r['pipeline'] != 'done']
    if pend_fit:
        lines += ['', 'Fits still pending: ' + '; '.join(pend_fit) + '.']
    if pend_pipe:
        lines += ['', 'Pipeline results still pending: ' + '; '.join(pend_pipe) + '.']
    (out / 'comparison_table.md').write_text('\n'.join(lines) + '\n')
    make_figures(df, excess, out)
    hist_files = make_history_figure(df, res_dir, out)
    for f in hist_files:
        inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E58_gain_noise_block.py',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        inputs_sha256=inputs, partb_folders=[str(d) for d in folders],
        pipeline_source={m: pipeline_source.get(m) for m in sorted(df['pipeline_model'])},
        models=[dict(label=r['model'], key=r['key'], experiment=r['experiment'], tag=r['tag'], pipeline_model=r['pipeline_model'], gain=r['gain']) for _, r in df.iterrows()],
        pending_fits=pend_fit, pending_pipeline=pend_pipe)
    (out / 'provenance_comparison.json').write_text(json.dumps(prov, indent=1) + '\n')
    print('\n'.join(lines))
    print('->', out)


def make_figures(df: pd.DataFrame, excess: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    done = df[(df['fit'] == 'done') & (df['pipeline'] == 'done')]
    thrs = list(AXIS)
    fig, axes = plt.subplots(4, len(thrs), figsize=(5.2 * len(thrs), 11.5), squeeze=False, sharex='col')
    panels = [('block_raw', 'fruity × small-ester block\nthrough the matched noise (three-trial means)', dt('block'), 'KC data, measured'),
        ('kc_kc_inh', 'fitted KC-KC scale', None, None),
        ('rate_trial_off', 'response rate per trial, APL off', dt('tnt_frac'), 'APL output blocked (TNT)'),
        ('loss_recorded', 'extended loss', None, None)]
    for j, thr in enumerate(thrs):
        d = done[done['thr'] == thr].sort_values('gain')
        for i, (col, ylabel, ref, refname) in enumerate(panels):
            ax = axes[i, j]
            c = '#2b6cb0' if thr == 40 else '#6a4c93'
            ax.plot(d['gain'], d[col], 'o-', color=c, ms=6, lw=1)
            if i == 0:
                ax.plot(d['gain'], d['block_grand'], 'v', color=c, ms=5, mfc='none', label='grand mean over 30 trials')
                ax.plot(d['gain'], d['block_det'], '_', color=c, ms=9, mew=1.5, label='deterministic block')
                ax.plot([], [], 'o-', color=c, label='three-trial means through independent noise')
                ax.legend(fontsize=7, frameon=False, loc='best')
            if ref is not None:
                ax.axhline(ref, color='k', lw=1, ls='--')
                ax.text(ax.get_xlim()[0], ref, f' {refname}', fontsize=7, va='bottom')
            ax.set_ylabel(ylabel, fontsize=8)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
            if i == 0:
                ax.set_title(THR_LABEL[thr], fontsize=9)
            if i == len(panels) - 1:
                ax.set_xlabel('eight-glomerulus gain (1 = no gain)')
    fig.suptitle('The glomerulus gain and the block through the noise: the structure fits at 6.8 % along the gain axis', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E58_block_vs_gain.png', dpi=130); plt.close(fig)
    labels = [k for k in excess if k != 'KC data']
    if not labels:
        return
    panels2 = labels + ['KC data']
    ncols = min(len(panels2), 5)
    nrows = int(np.ceil(len(panels2) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 4.4 * nrows), sharey=True, squeeze=False)
    pair_names = [f'{a} / {b}' for a, b in E51.BLOCK_PAIRS]
    for ax in axes.flat[len(panels2):]:
        ax.set_visible(False)
    for ax, lab in zip(axes.flat, panels2):
        e = excess[lab].reindex(E51.BLOCK_PAIRS)
        x = np.arange(len(E51.BLOCK_PAIRS))
        for i, (col, c, mk) in enumerate(zip(e.columns, ['0.2', '#2b6cb0', '#d9480f'], ['o', 's', '^'])):
            ax.plot(x, e[col].to_numpy(), mk, color=c, ms=5, label=col, mfc=('none' if i == 2 else None))
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(pair_names, rotation=70, fontsize=7)
        ax.set_title(lab, fontsize=8); ax.legend(fontsize=6, frameon=False)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
    for row_axes in axes:
        row_axes[0].set_ylabel('excess of the APL-on distance over the\nisotonic trend from the APL-off distance')
    fig.suptitle('The nine fruity × small-ester pairs: deterministic against processed through the matched noise', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E58_block_pairs.png', dpi=130); plt.close(fig)


def make_history_figure(df: pd.DataFrame, res_dir: Path, out: Path) -> list:
    """Loss, block and APL-off response rate over the evaluations of E58's fits; returns the files read."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    own = df[(df['fit'] == 'done') & (df['experiment'] == EXPERIMENT)]
    files = [(r['model'], r['tag'], res_dir / f"history_{r['tag']}.csv") for _, r in own.iterrows()]
    files = [(m, t, f) for m, t, f in files if f.exists()]
    if not files:
        return []
    fig, axes = plt.subplots(2, len(files), figsize=(max(4.2 * len(files), 8.4), 6.2), squeeze=False)
    for j, (label, tag, f) in enumerate(files):
        h = pd.read_csv(f)
        ax = axes[0, j]
        ax.plot(h.index, h['loss'], color='0.75', lw=0.7, label='loss at each evaluation')
        ax.plot(h.index, h['loss'].cummin(), color='k', lw=1.2, label='best so far')
        ax.set(yscale='log', xlabel='evaluation', ylabel='extended loss', title=label)
        ax.title.set_fontsize(8.5); ax.legend(fontsize=7, frameon=False)
        ax = axes[1, j]
        ax.plot(h.index, h['block'], color='#2b6cb0', lw=0.9, label='deterministic block')
        if 'rate_off' in h:
            ax2 = ax.twinx()
            ax2.plot(h.index, h['rate_off'], color='#d9480f', lw=0.9)
            ax2.axhline(dt('tnt_frac'), color='#d9480f', lw=0.8, ls=':')
            ax2.set_ylabel('APL-off response rate (deterministic)', color='#d9480f', fontsize=8)
            ax2.spines[['top']].set_visible(False)
        ax.axhline(dt('block'), color='k', lw=1, ls='--', label='KC data, measured')
        ax.set(xlabel='evaluation', ylabel='fruity × small-ester block'); ax.legend(fontsize=7, frameon=False, loc='lower left')
        for a in axes[:, j]:
            a.spines[['top', 'right']].set_visible(False); a.grid(True, color='0.92', lw=0.6)
    fig.suptitle('The gain-1.75 and gain-2 fits: loss, the deterministic block and the APL-off response rate over the evaluations', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E58_fit_histories.png', dpi=130); plt.close(fig)
    return [f for _, _, f in files]


def per_odor_rates(out: Path) -> None:
    """Per threshold, each model's deterministic per-odor response rates with APL on and off from its
    saved outputs, against the KC data's (control flies for APL on, APL output blocked (TNT) for APL
    off); per_odor_rates.csv and E58_per_odor_rates.png."""
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = {'on': H.yang_response_fractions('TNT3c').mean().reindex(H.APL16_ODORS),
        'acute_off': H.yang_response_fractions('TNT').mean().reindex(H.APL16_ODORS)}
    rows, series = [], {}
    for thr, axis in AXIS.items():
        for gain, exp, tag in axis:
            label = model_label(thr, gain)
            for cond in ('on', 'acute_off'):
                r = E53.saved_responses(exp, tag, cond)
                if r is None:
                    continue
                fr = H.response_fractions(r).reindex(H.APL16_ODORS)
                series[(thr, label, cond)] = fr
                rho = float(stats.spearmanr(fr, data[cond])[0])
                for odor in H.APL16_ODORS:
                    rows.append(dict(thr=thr, model=label, experiment=exp, tag=tag, gain=gain, condition=cond, odor=odor,
                        model_rate=float(fr[odor]), data_rate=float(data[cond][odor]), spearman_vs_data=rho))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'per_odor_rates.csv', index=False)
    thrs = list(AXIS)
    fig, axes = plt.subplots(2, len(thrs), figsize=(6.4 * len(thrs), 7.2), squeeze=False, sharey='row')
    x = np.arange(len(H.APL16_ODORS))
    cmap = plt.get_cmap('viridis')
    for j, thr in enumerate(thrs):
        for i, cond in enumerate(('on', 'acute_off')):
            ax = axes[i, j]
            ax.plot(x, data[cond].to_numpy(), 'k--', lw=1.2, label=('KC data: control flies' if cond == 'on' else 'KC data: APL output blocked (TNT)'))
            keys = [k for k in series if k[0] == thr and k[2] == cond]
            for n, key in enumerate(keys):
                fr = series[key]
                rho = float(stats.spearmanr(fr, data[cond])[0])
                ax.plot(x, fr.to_numpy(), 'o-', color=cmap(n / max(len(keys) - 1, 1)), ms=4, lw=1, label=f'{key[1].split(", ", 1)[1]} (ρ={rho:.2f})')
            ax.set_xticks(x); ax.set_xticklabels(H.APL16_ODORS, rotation=70, fontsize=7)
            ax.set_ylabel('fraction of KCs responding' + (', APL on' if cond == 'on' else ', APL off'), fontsize=8)
            ax.legend(fontsize=6.5, frameon=False)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
            if i == 0:
                ax.set_title(THR_LABEL[thr], fontsize=9)
    fig.suptitle('Per-odor response rates (deterministic) along the gain axis against the KC data', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E58_per_odor_rates.png', dpi=130); plt.close(fig)
    print(df.groupby(['model', 'condition'])['spearman_vs_data'].first().unstack('condition').round(3).to_string())
    print('->', out / 'per_odor_rates.csv')


if __name__ == '__main__':
    main()
