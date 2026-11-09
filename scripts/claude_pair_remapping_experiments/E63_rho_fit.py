#!/usr/bin/env python3
"""E63: the two-rho fit. Comparison table and figures.

Reads the two-rho fits (results/E63_rho_fit/fits.csv: E10_joint_fit.py --loss rho, the two matrix
terms, one minus the Spearman rho on odor-odor distances with the control flies' KC matrix (APL on)
and with the APL-output-blocked (TNT) matrix (APL off), at weight 4 each, the APL-on response rate,
the APL-off / APL-on rate ratio and the per-odor response-rate terms as constraints at their usual
weights, the block, the monotone relation and the structure statistics recorded but not optimized;
each started exactly at its seed's optimum), their seeds (E44's gain-1.5 fits at 40 and 70 Hz, E45's
no-gain and gain-1.25 fits at 40 Hz; the structure loss at the same operating point) and the
trial-noise pipeline's results on all of them (E43_model_trial_noise.py --part B --model-from
<experiment>:<tag> --tune-trials 3 with E48's settings; a model's rows from the first part-B folder
that has them: E63's own, then E58's, E48's and E51's), and writes to results/E63_rho_fit/:
- comparison.csv / comparison_table.md: per model, rho on odor-odor distances with APL on and off,
  deterministic and through the noise, the response rate rho with APL on and off, the response
  rates (deterministic and per trial), the block deterministic and through the noise, the
  natural-source alignment (APL on / off / on minus off), structure vs wild-type somata, the fitted
  parameters, and both losses (the fit's own criterion and the structure loss recomputed from the
  recorded terms where available);
- E63_rho_vs_seed.png: per base, the two rho's (deterministic and through the noise) of the seed and
  the two-rho fit, with the data's ceilings and the ORN input's values;
- E63_block_pairs.png and E63_fit_histories.png as the other write-ups' figures;
- with --per-odor, per_odor_rates.csv and E63_per_odor_rates.png.
Nothing is simulated.
Usage: ./venv/bin/python E63_rho_fit.py [--out DIR] [--per-odor]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H
import E51_noise_robust_block as E51
import E53_kckc_held_block as E53

EXPERIMENT = 'E63_rho_fit'
V = 'spike_counts_cosine__'
# the bases: (label, seed experiment, seed tag, claw_thr, gain or None)
BASES = [
    ('eight at 1.5, 40 Hz (the reference model)', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3', 40, 1.5),
    ('eight at 1.5, 70 Hz', 'E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3', 70, 1.5),
    ('no gain, 40 Hz', 'E45_gain_at_ctrl_frac', 'thr40_rate0.068_k2_cap3', 40, None),
    ('eight at 1.25, 40 Hz', 'E45_gain_at_ctrl_frac', 'thr40_gain1.25_rate0.068_k2_cap3', 40, 1.25),
]
FIT_COLS = E53.FIT_COLS + ('fit_term_vcf_off', 'loss_variant')


def rho_fits_for(fits: pd.DataFrame, thr: int, gain) -> list:
    """Tags of the two-rho fits of one base in E63's fits.csv (thr<T>[_gain<G>]_rate0.068_rho<W>_...)."""
    g = '' if gain is None else f'_gain{gain:g}'
    return [t for t in fits.index if re.match(rf'thr{thr}{g}_rate0\.068_rho[\d.]+_', str(t))]


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
    for exp in sorted({b[1] for b in BASES} | {EXPERIMENT}):
        f = H.results_dir(exp) / 'fits.csv'
        if not f.exists():
            fits[exp] = pd.DataFrame(columns=['tag']).set_index('tag')
            continue
        inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
        fits[exp] = pd.read_csv(f).set_index('tag')
    folders = [res_dir / 'partB'] + [H.results_dir(e) / 'partB' for e in ('E58_gain_noise_block', 'E48_block_target_noise', 'E51_noise_robust_block')]
    tables, pipeline_source, tin = E53.load_tables(folders)
    inputs.update(tin)
    models = []
    for label, exp, tag, thr, gain in BASES:
        bk = f'thr{thr}_' + ('nogain' if gain is None else f'gain{gain:g}')
        models.append((label + ', the seed (structure loss)', bk + '_seed', exp, tag, f'{E51.exp_id(exp)}-{tag}', 'seed', label))
        rf = rho_fits_for(fits[EXPERIMENT], thr, gain)
        if not rf:
            models.append((label + ', two-ρ fit (pending)', bk + '_rho', EXPERIMENT, None, None, 'rho', label))
        for i, t in enumerate(rf):
            models.append((label + ', two-ρ fit' + (f' ({i + 1})' if len(rf) > 1 else ''), bk + '_rho' + (f'_{i + 1}' if len(rf) > 1 else ''), EXPERIMENT, t, f'{E51.exp_id(EXPERIMENT)}-{t}', 'rho', label))
    rows, excess = [], {}
    for label, key, exp, tag, mname, kind, base in models:
        fit = fits[exp].loc[tag] if (tag is not None and tag in fits[exp].index) else None
        row = dict(model=label, key=key, kind=kind, base=base, experiment=exp, tag=tag, pipeline_model=mname, fit=('done' if fit is not None else 'pending'))
        if fit is not None:
            row['loss_recorded'] = float(fit['loss'])
            row.update({k: (float(fit[k]) if k != 'loss_variant' else str(fit[k])) for k in FIT_COLS if k in fit.index and pd.notna(fit[k])})
            row['vcf_on'] = float(fit[V + 'on_spearman_resid_vs_vcf_hyp']) if V + 'on_spearman_resid_vs_vcf_hyp' in fit.index else np.nan
            row['vcf_on_minus_off'] = row['vcf_on'] - row['fit_vcf_off'] if 'fit_vcf_off' in row else np.nan
        pr = E53.pipeline_row(tables, mname) if mname is not None else None
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

    lines = ['| model (fits at 6.8 %, APL->bouton cap 3) | criterion | loss (its own criterion) | ρ on odor-odor distances, APL on / off (deterministic) | ρ on odor-odor distances, APL on / off (through the noise) | response rate ρ, APL on / off | response rates, APL on / off (deterministic) | block, deterministic | block through independent noise, three-trial means / grand mean | natural-source alignment, APL on / off / on minus off | structure vs wild-type somata | fixed threshold | APL->KC / APL->bouton scale | KC-KC |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in df.iterrows():
        if r['fit'] != 'done':
            lines.append(f"| {r['model']} | | | | | | | | | | | | | |")
            continue
        crit = 'structure loss' if r['kind'] == 'seed' else f"two-ρ loss ({r.get('loss_variant', 'rho')})"
        vcf = f"{fmt(r.get('vcf_on'), '.2f', 'n/a')} / {fmt(r.get('fit_vcf_off'), '.2f', 'n/a')} / {fmt(r.get('vcf_on_minus_off'), '+.2f', 'n/a')}"
        noisy = f"{r['rho_raw_on']:.2f} / {r['rho_raw_off']:.2f}" if r['pipeline'] == 'done' else '(pending)'
        blk = f"{r['block_raw']:+.3f} / {r['block_grand']:+.3f}" if r['pipeline'] == 'done' else '(pending)'
        lines.append(f"| {r['model']} | {crit} | {r['loss_recorded']:.2f} | {fmt(r.get('fit_on_spearman'), '.3f')} / {fmt(r.get('fit_off_spearman'), '.3f')} | {noisy} | "
            f"{fmt(r.get('fit_po_on_spearman'), '.2f')} / {fmt(r.get('fit_po_off_spearman'), '.2f')} | {fmt(r.get('fit_rate_on'), '.3f')} / {fmt(r.get('fit_rate_off'), '.3f')} | "
            f"{fmt(r.get('fit_block'), '+.3f')} | {blk} | {vcf} | {fmt(r.get('fit_wt_dev_on'), '.2f')} | {fmt(r.get('fixed_thr'), '.1f')} | "
            f"{fmt(r.get('wAPLKC_scale'), '.3f')} / {fmt(r.get('wAPLPN_scale'), '.3f')} | {fmt(r.get('kc_kc_inh'), '.1f')} |")
    lines.append(f"| KC data | | | ceiling {dt('ctrl_ceiling'):.2f} / {dt('tnt_ceiling'):.2f} | ceiling {dt('ctrl_ceiling'):.2f} / {dt('tnt_ceiling'):.2f} | ceiling {dt('ctrl_rate_rho_ceiling'):.2f} | {dt('ctrl_frac'):.3f} / {dt('tnt_frac'):.3f} | "
        f"{dt('block'):+.3f} measured{'' if data_block_corr is None else f', {data_block_corr:+.3f} noise-corrected'} | | {dt('ctrl_vcf'):.2f} / {dt('tnt_vcf'):.2f} / {dt('vcf_apl_effect'):+.2f} | {dt('ctrl_struct_wt'):.2f} | | | |")
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
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E63_rho_fit.py',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        inputs_sha256=inputs, partb_folders=[str(d) for d in folders],
        pipeline_source={m: pipeline_source.get(m) for m in sorted({mm[4] for mm in models if mm[4] is not None})},
        models=[dict(label=m[0], key=m[1], experiment=m[2], tag=m[3], pipeline_model=m[4], kind=m[5], base=m[6]) for m in models],
        pending_fits=pend_fit, pending_pipeline=pend_pipe)
    (out / 'provenance_comparison.json').write_text(json.dumps(prov, indent=1) + '\n')
    print('\n'.join(lines))
    print('->', out)


def make_figures(df: pd.DataFrame, excess: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    done = df[df['fit'] == 'done']
    bases = [b[0] for b in BASES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, cond, col_d, col_n, ceil, orn in [(axes[0], 'APL on (control flies)', 'fit_on_spearman', 'rho_raw_on', dt('ctrl_ceiling'), None),
                                              (axes[1], 'APL off (APL output blocked, TNT)', 'fit_off_spearman', 'rho_raw_off', dt('tnt_ceiling'), None)]:
        x = np.arange(len(bases))
        for i, base in enumerate(bases):
            d = done[done['base'] == base]
            for _, r in d.iterrows():
                c = '#2b6cb0' if r['kind'] == 'seed' else '#d9480f'
                off = -0.15 if r['kind'] == 'seed' else 0.15
                ax.plot([i + off], [r[col_d]], 'o', color=c, ms=7)
                if r['pipeline'] == 'done':
                    ax.plot([i + off], [r[col_n]], 's', color=c, ms=6, mfc='none')
        ax.axhline(ceil, color='k', lw=1, ls='--')
        ax.text(-0.4, ceil, ' KC data, ceiling', fontsize=7, va='bottom')
        ax.set_xticks(x); ax.set_xticklabels([b.replace(' (the reference model)', '\n(the reference model)') for b in bases], fontsize=7.5)
        ax.set_ylabel(f'ρ on odor-odor distances, {cond}', fontsize=8.5)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
    axes[0].plot([], [], 'o', color='#2b6cb0', label='the seed (structure loss), deterministic')
    axes[0].plot([], [], 'o', color='#d9480f', label='the two-ρ fit, deterministic')
    axes[0].plot([], [], 's', color='0.3', mfc='none', label='through the matched noise (processed as the data)')
    axes[0].legend(fontsize=7, frameon=False, loc='lower left')
    fig.suptitle('The two-ρ fits against their seeds: ρ on odor-odor distances with the two KC matrices', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E63_rho_vs_seed.png', dpi=130); plt.close(fig)
    labels = [k for k in excess if k != 'KC data']
    if not labels:
        return
    panels = labels + ['KC data']
    ncols = min(len(panels), 5)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 4.4 * nrows), sharey=True, squeeze=False)
    pair_names = [f'{a} / {b}' for a, b in E51.BLOCK_PAIRS]
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, lab in zip(axes.flat, panels):
        e = excess[lab].reindex(E51.BLOCK_PAIRS)
        x = np.arange(len(E51.BLOCK_PAIRS))
        for i, (col, c, mk) in enumerate(zip(e.columns, ['0.2', '#2b6cb0', '#d9480f'], ['o', 's', '^'])):
            ax.plot(x, e[col].to_numpy(), mk, color=c, ms=5, label=col, mfc=('none' if i == 2 else None))
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(pair_names, rotation=70, fontsize=7)
        ax.set_title(lab.replace(', ', ',\n'), fontsize=7.5); ax.legend(fontsize=6, frameon=False)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
    for row_axes in axes:
        row_axes[0].set_ylabel('excess of the APL-on distance over the\nisotonic trend from the APL-off distance')
    fig.suptitle('The nine fruity × small-ester pairs: deterministic against processed through the matched noise', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E63_block_pairs.png', dpi=130); plt.close(fig)


def make_history_figure(df: pd.DataFrame, res_dir: Path, out: Path) -> list:
    """Loss and the two rho's over the evaluations of each two-rho fit; returns the files read."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
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
        ax.set(yscale='log', xlabel='evaluation', ylabel='two-ρ loss', title=label.replace(', two-ρ fit', '\ntwo-ρ fit'))
        ax.title.set_fontsize(8.5); ax.legend(fontsize=7, frameon=False)
        ax = axes[1, j]
        if 'on_spearman' in h and 'off_spearman' in h:
            ax.plot(h.index, h['on_spearman'], color='#2b6cb0', lw=0.9, label='ρ on odor-odor distances, APL on')
            ax.plot(h.index, h['off_spearman'], color='#d9480f', lw=0.9, label='ρ on odor-odor distances, APL off')
        if 'block' in h:
            ax2 = ax.twinx()
            ax2.plot(h.index, h['block'], color='0.4', lw=0.8, ls=':')
            ax2.set_ylabel('deterministic block', color='0.4', fontsize=8)
            ax2.spines[['top']].set_visible(False)
        ax.set(xlabel='evaluation', ylabel='ρ on odor-odor distances'); ax.legend(fontsize=7, frameon=False, loc='lower right')
        for a in axes[:, j]:
            a.spines[['top', 'right']].set_visible(False); a.grid(True, color='0.92', lw=0.6)
    fig.suptitle('The two-ρ fits over their evaluations: the loss, the two ρ\'s and the deterministic block', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E63_fit_histories.png', dpi=130); plt.close(fig)
    return [f for _, _, f in files]


def per_odor_rates(out: Path) -> None:
    """Each seed's and two-rho fit's deterministic per-odor response rates with APL on and off against
    the KC data's; per_odor_rates.csv and E63_per_odor_rates.png."""
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = {'on': H.yang_response_fractions('TNT3c').mean().reindex(H.APL16_ODORS),
        'acute_off': H.yang_response_fractions('TNT').mean().reindex(H.APL16_ODORS)}
    fits = pd.read_csv(H.results_dir(EXPERIMENT) / 'fits.csv').set_index('tag') if (H.results_dir(EXPERIMENT) / 'fits.csv').exists() else pd.DataFrame()
    rows, series = [], {}
    for label, exp, tag, thr, gain in BASES:
        models = [(label + ', the seed', exp, tag)] + [(label + ', two-ρ fit', EXPERIMENT, t) for t in rho_fits_for(fits, thr, gain)]
        for mlabel, mexp, mtag in models:
            for cond in ('on', 'acute_off'):
                r = E53.saved_responses(mexp, mtag, cond)
                if r is None:
                    continue
                fr = H.response_fractions(r).reindex(H.APL16_ODORS)
                series[(label, mlabel, cond)] = fr
                rho = float(stats.spearmanr(fr, data[cond])[0])
                for odor in H.APL16_ODORS:
                    rows.append(dict(base=label, model=mlabel, experiment=mexp, tag=mtag, condition=cond, odor=odor,
                        model_rate=float(fr[odor]), data_rate=float(data[cond][odor]), spearman_vs_data=rho))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'per_odor_rates.csv', index=False)
    bases = [b[0] for b in BASES]
    fig, axes = plt.subplots(2, len(bases), figsize=(5.4 * len(bases), 7.2), squeeze=False, sharey='row')
    x = np.arange(len(H.APL16_ODORS))
    for j, base in enumerate(bases):
        for i, cond in enumerate(('on', 'acute_off')):
            ax = axes[i, j]
            ax.plot(x, data[cond].to_numpy(), 'k--', lw=1.2, label=('KC data: control flies' if cond == 'on' else 'KC data: APL output blocked (TNT)'))
            for key, c in [(k, ('#2b6cb0' if 'seed' in k[1] else '#d9480f')) for k in series if k[0] == base and k[2] == cond]:
                fr = series[key]
                rho = float(stats.spearmanr(fr, data[cond])[0])
                ax.plot(x, fr.to_numpy(), 'o-', color=c, ms=4, lw=1, label=f"{key[1].split(', ')[-1]} (ρ={rho:.2f})")
            ax.set_xticks(x); ax.set_xticklabels(H.APL16_ODORS, rotation=70, fontsize=7)
            ax.set_ylabel('fraction of KCs responding' + (', APL on' if cond == 'on' else ', APL off'), fontsize=8)
            ax.legend(fontsize=6.5, frameon=False)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
            if i == 0:
                ax.set_title(base, fontsize=9)
    fig.suptitle('Per-odor response rates (deterministic): the seeds and the two-ρ fits against the KC data', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E63_per_odor_rates.png', dpi=130); plt.close(fig)
    print(df.groupby(['model', 'condition'])['spearman_vs_data'].first().unstack('condition').round(3).to_string())
    print('->', out / 'per_odor_rates.csv')


if __name__ == '__main__':
    main()
