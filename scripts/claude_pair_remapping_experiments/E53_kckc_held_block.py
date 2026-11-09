#!/usr/bin/env python3
"""E53: KC-KC lateral inhibition as the carrier of a noise-robust block. Comparison table and
figures (the write-up of the KC-KC-held fits).

Reads the KC-KC-held fits (results/E53_kckc_held_block/fits.csv: E10_joint_fit.py --kckc-fixed S
--init-from-exact, the deterministic structure loss at the control flies' 6.8 % response rate with
the KC-KC scale held at S and the other three parameters fitted, started exactly at the seed's
optimum with its KC-KC replaced), their seeds (E45_gain_at_ctrl_frac: the eight glomeruli at 1.25
at the 70 Hz per-claw threshold; E44_operating_point: the eight at 1.5 at 40 Hz), the simplex
vertices of the noise-robust block experiment's (E51) 1.25 fit that motivated E53 (KC-KC 27 and 33
at threshold 60; their deterministic statistics from E51's history, their pipeline rows from E51's
part-B folder) and the trial-noise pipeline's results on all of them (E43_model_trial_noise.py
--part B --model-from <experiment>:<tag> --tune-trials 3 with E48's settings: per-trial KC threshold
jitter at the sigma matching the data's single-trial same-odor floor, the mean threshold re-tuned to
the model's own APL-on response rate, 10 pseudo-recordings of 3 trials processed as the KC data), and
writes to results/E53_kckc_held_block/:
- comparison.csv / comparison_table.md: per model the KC-KC scale, the fitted threshold and APL
  scales, the loss, the deterministic block, the block through independent noise (three-trial means
  and the grand mean over 30 trials), the matched sigma and the floors reached, the response rate
  per trial with APL on and off, the deterministic response rate rho with APL on and off (the
  per-odor pattern), rho on odor-odor distances through the noise, and the natural-source alignment
  with APL on / off / on minus off (deterministic);
- E53_block_vs_kckc.png: per base, against the KC-KC scale, the block through the noise, the
  APL-off response rate per trial and the loss, the seed, the vertices and the held fits on one axis;
- E53_block_pairs.png: the nine fruity x small-ester pairs' excess of the APL-on distance over the
  isotonic trend from the APL-off distance, deterministic against processed, per model and in the
  data;
- E53_fit_histories.png: loss and block over the evaluations of each held fit.
A model's pipeline rows come from the first part-B folder that has them: E53's own, then E51's
(the vertices), then E48's (the seeds, already run under the same settings); comparison.csv and the
provenance say which. Models whose fit or pipeline result is not in yet are listed as pending and left
out of the figures. Nothing is simulated.
Usage: ./venv/bin/python E53_kckc_held_block.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import kc_remapping_harness as H
import E51_noise_robust_block as E51

EXPERIMENT = 'E53_kckc_held_block'
V = 'spike_counts_cosine__'
# the bases: (label, seed experiment, seed fits.csv tag, claw_thr, gain)
BASES = [
    ('eight at 1.25, 70 Hz', 'E45_gain_at_ctrl_frac', 'thr70_gain1.25_rate0.068_k2_cap3', 70, 1.25),
    ('eight at 1.5, 40 Hz', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3', 40, 1.5),
]
# the E51 vertex rows shown beside each base (E51 fit tag -> (claw_thr, gain))
E51_VERTEX_FITS = {'thr70_gain1.25_rate0.068_jitK9t3x1.041_k2_cap3': (70, 1.25)}
FIT_COLS = ('fit_block', 'fit_on_spearman', 'fit_off_spearman', 'fit_po_on_spearman', 'fit_po_off_spearman',
    'fit_wt_dev_on', 'fit_vcf_off', 'fit_rate_on', 'fit_rate_off', 'fit_ratio', 'fit_iso_r2', 'fit_jit_block',
    'fit_term_on_mat', 'fit_term_off_mat', 'fit_term_block', 'fit_term_rate_on', 'fit_term_ratio', 'fit_term_po_on',
    'fit_term_po_off', 'fit_term_iso', 'fit_term_wt_dev', 'fit_term_excess_on', 'thr_factor', 'apl_factor_vs_e3',
    'aplpn_factor_vs_e3', 'kc_kc_inh', 'fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale',
    'n_evals', 'seconds', 'time_s')


def held_fits_for(fits: pd.DataFrame, claw_thr: int, gain: float) -> list:
    """(tag, held KC-KC scale) of the KC-KC-held fits of one base in E53's fits.csv, by held value."""
    out = []
    for t in fits.index:
        m = re.fullmatch(rf'thr{claw_thr}_gain{gain:g}_rate0\.068_kckc(\d+(?:\.\d+)?)_k2_cap3', str(t))
        if m:
            out.append((str(t), float(m.group(1))))
    return sorted(out, key=lambda x: x[1])


def load_tables(folders: list) -> tuple:
    """The pipeline's three tables with each model's rows taken from the first folder that has
    them; returns (tables, {model: folder}, {file: sha256})."""
    tables, source, inputs = {}, {}, {}
    for name in ('stats_noise', 'onoff_noise', 'pairs_noise_long'):
        frames, have = [], set()
        for d in folders:
            f = d / f'{name}.csv'
            if not f.exists():
                continue
            t = pd.read_csv(f)
            t = t[~t['model'].isin(have)]
            if not len(t):
                continue
            inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
            frames.append(t)
            have |= set(t['model'])
            if name == 'stats_noise':
                source.update({m: str(d) for m in t['model'].unique()})
        if frames:
            tables[name] = pd.concat(frames, ignore_index=True)
    return tables, source, inputs


def pipeline_row(tables: dict, mname: str) -> Optional[dict]:
    """The pipeline statistics of model `mname` at its APL-on-matched level, or None if not run."""
    if 'stats_noise' not in tables:
        return None
    lv = E51.level_of(tables['stats_noise'], mname)
    if lv is None:
        return None
    s = tables['stats_noise']; o = tables['onoff_noise']
    g = s[(s['model'] == mname) & (s['level'] == lv) & (s['transform'] == 'graded')].set_index('condition')
    oo = o[(o['model'] == mname) & (o['transform'] == 'graded')].set_index(['level', 'matrix'])
    return dict(level=lv, sigma=float(g.at['on', 'sigma']), floor_on=float(g.at['on', 'single_trial_floor']),
        floor_off=float(g.at['acute_off', 'single_trial_floor']), trial_mean_floor_on=float(g.at['on', 'trial_mean_floor']),
        rate_trial_on=float(g.at['on', 'rate_trial']), rate_trial_off=float(g.at['acute_off', 'rate_trial']),
        rate_det_on=float(g.at['on', 'rate_det']), rate_det_off=float(g.at['acute_off', 'rate_det']),
        block_det=float(oo.at[('deterministic', 'deterministic'), 'block']), block_raw=float(oo.at[(lv, 'raw'), 'block']),
        block_grand=float(oo.at[(lv, 'grand'), 'block']), block_corrected=float(oo.at[(lv, 'corrected'), 'block']),
        iso_r2_raw=float(oo.at[(lv, 'raw'), 'iso_r2']), iso_r2_det=float(oo.at[('deterministic', 'deterministic'), 'iso_r2']),
        rho_raw_on=float(g.at['on', 'rho_raw_vs_data']), rho_raw_off=float(g.at['acute_off', 'rho_raw_vs_data']),
        rho_det_on=float(g.at['on', 'rho_det_vs_data']), rho_det_off=float(g.at['acute_off', 'rho_det_vs_data']),
        rate_rho_trial_on=float(g.at['on', 'rate_rho_vs_data_trial']) if 'rate_rho_vs_data_trial' in g else np.nan)


def pair_excess_frame(tables: dict, mname: str, lv: str) -> pd.DataFrame:
    p = tables['pairs_noise_long']
    pm = p[(p['model'] == mname) & (p['level'] == lv) & (p['transform'] == 'graded')]
    return pd.DataFrame({'deterministic': E51.pair_excess(pm, 'det'), 'three-trial means': E51.pair_excess(pm, 'raw'),
        'grand mean': E51.pair_excess(pm, 'grand')})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=None, help='output folder (default: the results folder)')
    ap.add_argument('--per-odor', action='store_true',
        help="only the per-odor response-rate diagnostic (per_odor_rates.csv and E53_per_odor_rates.png): each model's "
        "deterministic per-odor response rates with APL on and off, from its saved outputs, against the KC data's")
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT)
    out = args.out or res_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.per_odor:
        per_odor_rates(out)
        return
    dt = H.data_target
    inputs = {}
    fits = {}
    for exp in sorted({b[1] for b in BASES} | {EXPERIMENT, E51.EXPERIMENT}):
        f = H.results_dir(exp) / 'fits.csv'
        if not f.exists():
            fits[exp] = pd.DataFrame(columns=['tag']).set_index('tag')
            continue
        inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
        fits[exp] = pd.read_csv(f).set_index('tag')
    folders = [res_dir / 'partB', H.results_dir(E51.EXPERIMENT) / 'partB', H.results_dir('E48_block_target_noise') / 'partB']
    tables, pipeline_source, tin = load_tables(folders)
    inputs.update(tin)
    # the model list: per base the seed, E51's vertices (where that fit has any in the pipeline's
    # tables) and the held fits, lowest held value first
    # (label, key, experiment, tag, pipeline model name, kind, evaluation or None, held value or None)
    models = []
    for label, exp, tag, thr, gain in BASES:
        bk = f'thr{thr}_gain{gain:g}'
        models.append((label + ', the seed (KC-KC fitted)', bk + '_det', exp, tag, f'{E51.exp_id(exp)}-{tag}', 'seed', None, None))
        if 'stats_noise' in tables:
            for vt, ev, mname in E51.vertex_models(tables['stats_noise'], fits[E51.EXPERIMENT]):
                if E51_VERTEX_FITS.get(vt) == (thr, gain):
                    models.append((label + f', E51 noise-aware fit, simplex vertex (evaluation {ev})', bk + f'_noise_ev{ev}',
                        E51.EXPERIMENT, vt, mname, 'vertex', ev, None))
        hf = held_fits_for(fits[EXPERIMENT], thr, gain)
        if not hf:
            models.append((label + ', KC-KC held at 30 (pending)', bk + '_kckc30', EXPERIMENT, None, None, 'held', None, 30.0))
        for t, s in hf:
            models.append((label + f', KC-KC held at {s:g}', bk + f'_kckc{s:g}', EXPERIMENT, t, f'{E51.exp_id(EXPERIMENT)}-{t}', 'held', None, s))
    rows, excess = [], {}
    for label, key, exp, tag, mname, kind, ev, held in models:
        fit = fits[exp].loc[tag] if (tag is not None and tag in fits[exp].index) else None
        if fit is not None and ev is not None:
            fit = E51.history_as_fit(H.results_dir(E51.EXPERIMENT), tag, ev, fit)
        row = dict(model=label, key=key, kind=kind, experiment=exp, tag=tag, pipeline_model=mname,
            eval=(np.nan if ev is None else ev), held_kckc=(np.nan if held is None else held),
            fit=('done' if fit is not None else 'pending'))
        if fit is not None:
            row['loss_recorded'] = float(fit['loss'])
            row['criterion'] = ('deterministic structure loss, KC-KC fitted' if kind == 'seed' else
                f'deterministic structure loss, KC-KC held at {held:g}' if kind == 'held' else
                f'evaluation {ev} of E51\'s noise-aware fit (block term on the fit\'s own jitter draws; a simplex vertex, not an optimum)')
            row.update({k: float(fit[k]) for k in FIT_COLS if k in fit.index and pd.notna(fit[k])})
            if ev is None:
                row['vcf_on'] = float(fit[V + 'on_spearman_resid_vs_vcf_hyp']) if V + 'on_spearman_resid_vs_vcf_hyp' in fit.index else np.nan
            else:
                al = E51.saved_alignment(mname)
                row['vcf_on'] = al['vcf_on']
                row['alignment_source'] = al['source']
            row['vcf_on_minus_off'] = row['vcf_on'] - row['fit_vcf_off'] if 'fit_vcf_off' in row else np.nan
        pr = pipeline_row(tables, mname) if mname is not None else None
        if pr is None:
            row['pipeline'] = 'pending'
            rows.append(row)
            continue
        row['pipeline'] = 'done'
        row['pipeline_source'] = pipeline_source.get(mname)
        row.update(pr)
        excess[label] = pair_excess_frame(tables, mname, pr['level'])
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

    lines = ['| model (structure fits at 6.8 %, APL->bouton cap 3) | KC-KC | fixed threshold | APL->KC / APL->bouton scale | loss (deterministic criterion; the vertices under E51\'s jittered one) | block, deterministic | block through independent noise, three-trial means | block, grand mean of 30 trials | matched σ | single-trial floor reached, APL on / off | response rate per trial, APL on / off | response rate ρ, APL on / off (deterministic) | ρ on odor-odor distances, APL on / off (noisy) | natural-source alignment, APL on / off / on minus off (deterministic) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in df.iterrows():
        if r['fit'] != 'done':
            lines.append(f"| {r['model']} | {fmt(r.get('held_kckc'), 'g')} | | | | | | | | | | | | |")
            continue
        vcf = f"{fmt(r.get('vcf_on'), '.2f', 'n/a')} / {fmt(r.get('fit_vcf_off'), '.2f', 'n/a')} / {fmt(r.get('vcf_on_minus_off'), '+.2f', 'n/a')}"
        head = (f"| {r['model']} | {fmt(r.get('kc_kc_inh'), '.1f')} | {fmt(r.get('fixed_thr'), '.1f')} | "
            f"{fmt(r.get('wAPLKC_scale'), '.3f')} / {fmt(r.get('wAPLPN_scale'), '.3f')} | {r['loss_recorded']:.2f} | ")
        po = f"{fmt(r.get('fit_po_on_spearman'), '.2f')} / {fmt(r.get('fit_po_off_spearman'), '.2f')}"
        if r['pipeline'] != 'done':
            lines.append(head + f"{fmt(r.get('fit_block'), '+.3f')} | (pending) | | | | {fmt(r.get('fit_rate_on'), '.3f')} / {fmt(r.get('fit_rate_off'), '.3f')} (deterministic) | {po} | | {vcf} |")
            continue
        lines.append(head + f"{r['block_det']:+.3f} | {r['block_raw']:+.3f} | {r['block_grand']:+.3f} | {r['sigma']:.3f} | "
            f"{r['floor_on']:.2f} / {r['floor_off']:.2f} | {r['rate_trial_on']:.3f} / {r['rate_trial_off']:.3f} | {po} | "
            f"{r['rho_raw_on']:.2f} / {r['rho_raw_off']:.2f} | {vcf} |")
    lines.append(f"| KC data | | | | | {dt('block'):+.3f} measured{'' if data_block_corr is None else f', {data_block_corr:+.3f} noise-corrected'} | | | | "
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
    make_figures(df, excess, out, data_block_corr)
    hist_files = make_history_figure(df, res_dir, out)
    for f in hist_files:
        inputs[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E53_kckc_held_block.py',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        inputs_sha256=inputs, partb_folders=[str(d) for d in folders],
        pipeline_source={m: pipeline_source.get(m) for m in sorted({mm[4] for mm in models if mm[4] is not None})},
        models=[dict(label=m[0], key=m[1], experiment=m[2], tag=m[3], pipeline_model=m[4], kind=m[5], eval=m[6], held_kckc=m[7]) for m in models],
        pending_fits=pend_fit, pending_pipeline=pend_pipe)
    (out / 'provenance_comparison.json').write_text(json.dumps(prov, indent=1) + '\n')
    print('\n'.join(lines))
    print('->', out)


KIND_STYLE = {'seed': ('#2b6cb0', 'o', 'the seed (KC-KC fitted)'), 'vertex': ('#6a4c93', 'D', "E51's simplex vertices (not optima)"),
    'held': ('#d9480f', 's', 'KC-KC held, the rest fitted')}


def make_figures(df: pd.DataFrame, excess: dict, out: Path, data_block_corr) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    done = df[(df['fit'] == 'done') & (df['pipeline'] == 'done')]
    bases = [b[0] for b in BASES]
    fig, axes = plt.subplots(3, len(bases), figsize=(5.2 * len(bases), 9.6), squeeze=False, sharex='col')
    for j, base in enumerate(bases):
        d = done[done['model'].str.startswith(base)]
        for i, (col, ylabel, ref, refname) in enumerate([
                ('block_raw', 'fruity × small-ester block\nthrough the matched noise (three-trial means)', dt('block'), 'KC data, measured'),
                ('rate_trial_off', 'response rate per trial, APL off', dt('tnt_frac'), 'APL output blocked (TNT)'),
                ('loss_recorded', 'extended loss (deterministic criterion;\nvertices under E51\'s jittered one)', None, None)]):
            ax = axes[i, j]
            for _, r in d.iterrows():
                c, mk, _ = KIND_STYLE[r['kind']]
                ax.plot([r['kc_kc_inh']], [r[col]], mk, color=c, ms=7, mfc=('none' if r['kind'] == 'vertex' else None))
                if i == 0:
                    ax.plot([r['kc_kc_inh']], [r['block_grand']], 'v', color=c, ms=5, mfc='none')
                    ax.plot([r['kc_kc_inh']], [r['block_det']], '_', color=c, ms=9, mew=1.5)
            if ref is not None:
                ax.axhline(ref, color='k', lw=1, ls='--')
                ax.text(ax.get_xlim()[0], ref, f' {refname}', fontsize=7, va='bottom')
            ax.set_ylabel(ylabel, fontsize=8)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
            if i == 0:
                ax.set_title(base, fontsize=9.5)
            if i == 2:
                ax.set_xlabel('KC-KC scale (kc_kc_inh)')
    for kind, (c, mk, lab) in KIND_STYLE.items():
        axes[0, 0].plot([], [], mk, color=c, ms=7, label=lab, mfc=('none' if kind == 'vertex' else None))
    axes[0, 0].plot([], [], 'v', color='0.3', ms=5, mfc='none', label='grand mean over 30 trials')
    axes[0, 0].plot([], [], '_', color='0.3', ms=9, mew=1.5, label='deterministic block')
    axes[0, 0].legend(fontsize=7, frameon=False, loc='best')
    fig.suptitle('KC-KC lateral inhibition and the block through the noise: the seed, E51\'s vertices and the KC-KC-held fits', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E53_block_vs_kckc.png', dpi=130); plt.close(fig)
    labels = [k for k in excess if k != 'KC data']
    if not labels:
        return
    panels = labels + ['KC data']
    ncols = min(len(panels), 4)
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 4.4 * nrows), sharey=True, squeeze=False)
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
        ax.set_title(lab.replace(', ', ',\n').replace(' (evaluation', '\n(evaluation'), fontsize=7); ax.legend(fontsize=6, frameon=False)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
    for row_axes in axes:
        row_axes[0].set_ylabel('excess of the APL-on distance over the\nisotonic trend from the APL-off distance')
    fig.suptitle('The nine fruity × small-ester pairs: deterministic against processed through the matched noise', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E53_block_pairs.png', dpi=130); plt.close(fig)


def make_history_figure(df: pd.DataFrame, res_dir: Path, out: Path) -> list:
    """Loss and block over the evaluations of each KC-KC-held fit; returns the history files read."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    held = df[(df['fit'] == 'done') & (df['kind'] == 'held')]
    files = [(r['model'], r['tag'], res_dir / f"history_{r['tag']}.csv") for _, r in held.iterrows()]
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
            ax2.plot(h.index, h['rate_off'], color='#d9480f', lw=0.9, label='APL-off response rate')
            ax2.axhline(dt('tnt_frac'), color='#d9480f', lw=0.8, ls=':')
            ax2.set_ylabel('APL-off response rate (deterministic)', color='#d9480f', fontsize=8)
            ax2.spines[['top']].set_visible(False)
        ax.axhline(dt('block'), color='k', lw=1, ls='--', label='KC data, measured')
        ax.set(xlabel='evaluation', ylabel='fruity × small-ester block'); ax.legend(fontsize=7, frameon=False, loc='lower left')
        for a in axes[:, j]:
            a.spines[['top', 'right']].set_visible(False); a.grid(True, color='0.92', lw=0.6)
    fig.suptitle('The KC-KC-held fits: loss, the deterministic block and the APL-off response rate over the evaluations', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E53_fit_histories.png', dpi=130); plt.close(fig)
    return [f for _, _, f in files]


def saved_responses(experiment: str, tag: str, cond: str) -> Optional[pd.DataFrame]:
    """The saved KC x odor response matrix of fit `tag` of `experiment` with APL on ('on') or off
    ('acute_off'): from the experiment's local outputs folder, else the shared archive; None if absent."""
    for d in (H.outputs_dir(experiment), H.shared_dir() / 'data' / 'model_outputs' / experiment):
        f = d / f'{tag}_{cond}_responses.parquet'
        if f.exists():
            return pd.read_parquet(f)
    return None


def per_odor_rates(out: Path) -> None:
    """The per-odor response-rate diagnostic: for each base's seed and KC-KC-held fits, the deterministic
    fraction of KCs responding to each odor with APL on and off (from the saved outputs), beside the KC
    data's per-odor response fractions (control flies for APL on, APL output blocked (TNT) for APL off;
    the means over recordings), with each model's Spearman ρ against the data; per_odor_rates.csv (long)
    and E53_per_odor_rates.png."""
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = {'on': H.yang_response_fractions('TNT3c').mean().reindex(H.APL16_ODORS),
        'acute_off': H.yang_response_fractions('TNT').mean().reindex(H.APL16_ODORS)}
    fits = pd.read_csv(H.results_dir(EXPERIMENT) / 'fits.csv').set_index('tag') if (H.results_dir(EXPERIMENT) / 'fits.csv').exists() else pd.DataFrame()
    rows, series = [], {}
    for label, exp, tag, thr, gain in BASES:
        models = [(label + ', the seed', exp, tag, float('nan'))]
        models += [(label + f', KC-KC held at {s:g}', EXPERIMENT, t, s) for t, s in held_fits_for(fits, thr, gain)]
        for mlabel, mexp, mtag, held in models:
            for cond in ('on', 'acute_off'):
                r = saved_responses(mexp, mtag, cond)
                if r is None:
                    continue
                fr = H.response_fractions(r).reindex(H.APL16_ODORS)
                series[(mlabel, cond)] = fr
                rho = float(stats.spearmanr(fr, data[cond])[0])
                for odor in H.APL16_ODORS:
                    rows.append(dict(base=label, model=mlabel, experiment=mexp, tag=mtag, held_kckc=held, condition=cond,
                        odor=odor, model_rate=float(fr[odor]), data_rate=float(data[cond][odor]), spearman_vs_data=rho))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'per_odor_rates.csv', index=False)
    bases = [b[0] for b in BASES]
    fig, axes = plt.subplots(2, len(bases), figsize=(6.4 * len(bases), 7.2), squeeze=False, sharey='row')
    x = np.arange(len(H.APL16_ODORS))
    for j, base in enumerate(bases):
        for i, cond in enumerate(('on', 'acute_off')):
            ax = axes[i, j]
            ax.plot(x, data[cond].to_numpy(), 'k--', lw=1.2, label=('KC data: control flies' if cond == 'on' else 'KC data: APL output blocked (TNT)'))
            labels = [k for k in series if k[0].startswith(base) and k[1] == cond]
            cmap = plt.get_cmap('viridis')
            for n, key in enumerate(labels):
                fr = series[key]
                rho = float(stats.spearmanr(fr, data[cond])[0])
                ax.plot(x, fr.to_numpy(), 'o-', color=cmap(n / max(len(labels) - 1, 1)), ms=4, lw=1,
                    label=f"{key[0].split(', ', 2)[-1]} (ρ={rho:.2f})")
            ax.set_xticks(x); ax.set_xticklabels(H.APL16_ODORS, rotation=70, fontsize=7)
            ax.set_ylabel('fraction of KCs responding' + (', APL on' if cond == 'on' else ', APL off'), fontsize=8)
            ax.legend(fontsize=6.5, frameon=False)
            ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
            if i == 0:
                ax.set_title(base, fontsize=9.5)
    fig.suptitle('Per-odor response rates (deterministic) of the seed and the KC-KC-held fits against the KC data', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E53_per_odor_rates.png', dpi=130); plt.close(fig)
    summary = df.groupby(['model', 'condition'])['spearman_vs_data'].first().unstack('condition')
    print(summary.round(3).to_string())
    print('->', out / 'per_odor_rates.csv')


if __name__ == '__main__':
    main()
