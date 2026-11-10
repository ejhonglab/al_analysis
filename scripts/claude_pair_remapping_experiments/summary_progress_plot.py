#!/usr/bin/env python3
"""How close the best model has come to the KC data over the run, per statistic, as an
interactive plot for the end of the summary report.

Reads every experiment's result table under <shared>/results/E<n>_*/ (the harness scoring
columns, in either naming scheme), takes each experiment's best variant per statistic
(closest to the data's value / measurement ceiling; APL-on response rate within
RATE_RANGE where recorded; oracle variants excluded, see EXCLUDE / EXCLUDE_VARIANT), dates it
by the experiment's completion time from the experiment index in experiment_priorities.md,
and writes:

    <shared>/results/summary_progress/progress_points.csv   one row per experiment x statistic
    <shared>/results/summary_progress/progress_best.csv     the running best per statistic
    <shared>/results/summary_progress/progress_plot.html    the plot (inline SVG + script, no
                                                            external files), inserted into the
                                                            summary page by md_to_artifact.py's
                                                            `%%% html <path>` line

The y axis is the fraction of the gap between the first measured model and the KC data that
the best model so far has closed (0 = the first model, 1 = the data's value or ceiling); the
hover text gives the raw values. Legend entries toggle their series. Re-run after every
completed experiment, before rebuilding the summary page:

    ./venv/bin/python summary_progress_plot.py
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H

OUT = H.shared_dir() / 'results' / 'summary_progress'
YEAR = 2026
RATE_RANGE = (0.035, 0.075)   # APL-on response rate: comparable models only (tuned near 5 %)

# experiments whose variants use information from the data they are scored against; E39 splits the
# reference model's oracle gain into glomerulus subsets at the full set's fitted parameters (its
# full-set variant is E34's fit), so its best subset per statistic would be a post-hoc selection
# E62 (the per-odor rate oracle) and E64 (the bounded-input oracle) set each odor's input from the KC data they
# are scored against
EXCLUDE = {'E14', 'E15', 'E39', 'E62', 'E64'}
# variants excluded by tag / variant prefix: oracle gains, the block-specific glomerulus set,
# and random / shuffled / re-drawn wirings (nulls, many seeds each: the best of twenty random
# draws is a selection on noise, not a model)
# E38's level-matched input controls (the lab's profiles scaled to Hallem's level: not candidate
# inputs, a control separating profile from level) are excluded too
EXCLUDE_VARIANT = re.compile(r'^(oracle|blockset|uniform|hemidraw|shuffle-|A_lab_scaled)|_g1\.25$|_g1\.5$|_g2$|_g3$|random_g|_seed\d+', re.I)
# experiments without a comparable model table (data-only, readout-only, or no on/off scoring)
SKIP_FOLDERS = {'E1_apl_knockout', 'E2_local_apl', 'E6_readout_window', 'E7_confound_controls',
    'E21_apl_off_sharing', 'E23_weak_odor_responders', 'E29_oracle_substrate'}

# statistic -> (label, data-target registry entry of its target (the data's value, or its
# measurement ceiling), format of the target in labels, candidate columns [standard names,
# early-atlas names])
METRIC_DEFS = {
    'ctrl': ('ρ on odor-odor distances, APL on', 'ctrl_ceiling', '.2f',
        ['spike_counts_cosine__on_spearman', 'spearman_vs_TNT3c_on']),
    'tnt': ('ρ on odor-odor distances, APL off', 'tnt_ceiling', '.2f',
        ['spike_counts_cosine__off_spearman', 'spearman_vs_TNT_off']),
    'block': ('fruity x small-ester block', 'block', '+.3f',
        ['spike_counts_cosine__block_fruity_small_iso', 'block_fruity_small_iso']),
    'rate_rank': ('per-odor response rate ρ, APL on', 'ctrl_rate_rho_ceiling', '.2f',
        ['spearman_rate_on_vs_data', 'per_odor_rate_spearman_vs_TNT3c_on']),
    'wt_dev': ('structure beyond the ORN input vs wild-type somata (APL on)', 'ctrl_struct_wt', '.2f',
        ['spike_counts_cosine__on_spearman_resid_vs_remap_wt', 'spearman_resid_vs_remap_wt_on']),
    'vcf_on': ('natural-source alignment of the APL-on residual', 'ctrl_vcf', '.2f',
        ['spike_counts_cosine__on_spearman_resid_vs_vcf_hyp', 'spearman_resid_vs_vcf_hyp_on']),
    'vcf_off': ('natural-source alignment of the APL-off residual', 'tnt_vcf', '.2f',
        ['spike_counts_cosine__off_spearman_resid_vs_vcf_hyp', 'spearman_resid_vs_vcf_hyp_off']),
    # the main target since 2026-11-09 17:55 (CLAUDE.md "The main goal for new experiments"): the per-pair
    # APL effect (APL-on minus APL-off distance); its loss against the KC data's (lower is better, the
    # target is the data's own floor) and the ρ across pairs; scored for every saved variant by
    # apl_effect_scores.py (results/apl_effect/by_variant.csv, merged into each experiment's table by tag)
    'apl_loss': ('APL-effect loss (per-pair APL on minus APL off; lower is better)', 'apl_effect_rmse_floor', '.3f',
        ['spike_counts_cosine__apl_effect_rmse']),
    'apl_rho': ('ρ on the per-pair APL effect', 'apl_effect_rho_ceiling', '.2f',
        ['spike_counts_cosine__apl_effect_spearman']),
}
# statistic -> (label, target, candidate columns), the target read from the registry
METRICS = {k: (label, H.data_target(entry), cols) for k, (label, entry, _, cols) in METRIC_DEFS.items()}
TARGET_LABELS = {k: format(METRICS[k][1], fmt) for k, (_, _, fmt, _) in METRIC_DEFS.items()}
# (the APL-off / APL-on rate ratio is not plotted: the first measured model already sat at the
# data's 2.2, as most tuned acute-knockout models do, so a gap fraction is undefined for it)
# statistic -> (light, dark) color: the reference palette's categorical hues (dataviz skill). The
# two rank-agreement lines share the slot-1 blue on purpose, as a related pair: full strength for the
# control matrix, a lighter, less saturated step of the same OKLCH hue (L 0.75 / 0.78, C 0.10) for
# the APL-blocked one (normal-vision Delta E to the full blue 18.6 light / 16.9 dark); the two
# natural-source alignment lines are paired the same way in the dark green (full for APL on, a
# lighter step for APL off)
SERIES_COLORS = {
    'ctrl': ('#2a78d6', '#3987e5'), 'tnt': ('#83b1ed', '#8cbaf7'),
    'block': ('#1baf7a', '#199e70'), 'rate_rank': ('#eda100', '#c98500'),
    'wt_dev': ('#e87ba4', '#d55181'), 'vcf_on': ('#008300', '#2a9d2a'), 'vcf_off': ('#7fc27f', '#8ccf8c'),
    # the APL-effect pair in violet: full strength for the loss, a lighter step for its ρ
    'apl_loss': ('#8250df', '#a475f9'), 'apl_rho': ('#c2a5f0', '#c9b0fb'),
}
APL_EFFECT = H.shared_dir() / 'results' / 'apl_effect' / 'by_variant.csv'
_APL_EFFECT_COLS = ['spike_counts_cosine__apl_effect_rmse', 'spike_counts_cosine__apl_effect_spearman',
    'spike_counts_cosine__apl_effect_mean']
TABLE_NAMES = ('stats_variants.csv', 'fits.csv', 'wkckc_controls.csv')

# parameter columns shown in the hover text of a best model, in this order (label, format)
PARAM_LABELS = [
    ('base', 'base'), ('config', 'configuration'), ('kind', 'rule'),
    ('claw_thr', 'per-claw threshold (Hz)'), ('claw_thr_width', 'knee width (Hz)'),
    ('claw_thr_subgain', 'sub-threshold gain'), ('claw_sat', 'per-claw cap'), ('form', 'cap form'),
    ('claw_pow', 'per-claw power'),
    ('n_claws_active_to_spike', 'claws to spike (coincidence)'), ('coinput_factor', 'co-input factor'),
    ('k', 'breadth exponent'), ('breadth_k', 'breadth exponent'), ('m', 'shared-input exponent'),
    ('kc_kc_inh', 'KC-KC scale'), ('wKCKC_scale', 'KC-KC scale'), ('kc_kc_tau', 'KC-KC tau (s)'),
    ('fixed_thr', 'somatic threshold (Hz)'), ('thr_mean', 'somatic threshold (Hz)'),
    ('wAPLKC_scale', 'APL->KC scale'), ('wKCAPL_scale', 'KC->APL scale'),
    ('wAPLPN_scale', 'APL->bouton scale'), ('wPNAPL_scale', 'bouton->APL scale'),
    ('apl_factor_vs_e3', 'APL->KC x E3 base'), ('aplpn_factor_vs_e3', 'APL->bouton x E3 base'),
    ('kc_apl_factor', 'KC-APL factor'), ('apl_unit_coupling', 'APL unit coupling'), ('coupling', 'APL unit coupling'),
    ('bouton_ves_p', 'bouton release p'), ('bouton_tau_r', 'bouton recovery tau (s)'),
    ('alpha', 'normalization exponent'), ('gain_min', 'gain min'), ('gain_max', 'gain max'),
    ('gain', 'glomerulus gain'), ('pn_inhadd', 'PN inhadd'), ('pn_inhsc', 'PN inhsc'), ('inhadd', 'PN inhadd'),
    ('ext_inh_scale', 'external inhibition scale'), ('multibouton_claws', 'multibouton claws'),
    ('claw_weight', 'claw weight'), ('wiring', 'wiring'), ('seed', 'seed'),
    ('target_rate', 'target rate'), ('target', 'target rate'),
]
# statistics of the model's responses (measured, not set or tuned), shown after the parameters
STAT_LABELS = [
    ('rate_on', 'APL-on response rate'), ('rate_off', 'APL-off response rate'),
    ('rate_ratio_off_over_on', 'APL-off / APL-on rate ratio'),
]


def describe_params(row: pd.Series, labels: list = PARAM_LABELS) -> str:
    """'label value; label value; ...' for the columns in `labels` the row has values for.

    Args:
        row: one variant's row of a result table.
        labels: ordered (column, label) pairs; PARAM_LABELS by default, STAT_LABELS for
            `describe_stats`. Columns the row lacks, and empty / NaN values, are skipped.
    """
    parts, seen = [], set()
    for col, label in labels:
        if col not in row.index or label in seen:
            continue
        v = row[col]
        if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() in ('', 'nan', 'None')):
            continue
        if col in ('claw_thr', 'claw_sat', 'kc_kc_inh', 'wKCKC_scale', 'coupling', 'apl_unit_coupling', 'gain') \
                and isinstance(v, (int, float, np.integer, np.floating)) and float(v) in (0.0, 1.0) and col == 'gain':
            continue   # gain 1 = no gain
        if col in ('claw_thr', 'claw_sat', 'kc_kc_inh', 'wKCKC_scale') and isinstance(v, (int, float, np.integer, np.floating)) and float(v) == 0.0:
            continue   # 0 = the mechanism is off
        if isinstance(v, (float, np.floating)):
            v = f'{float(v):.3g}' if abs(float(v)) < 1000 else f'{float(v):.0f}'
        elif isinstance(v, (bool, np.bool_)):
            v = 'yes' if v else 'no'
        parts.append(f'{label} {v}')
        seen.add(label)
    return '; '.join(parts)


def describe_stats(row: pd.Series) -> str:
    """As `describe_params`, for the response statistics in STAT_LABELS."""
    return describe_params(row, STAT_LABELS)


def experiment_index() -> pd.DataFrame:
    """ID, name and completion time of every completed experiment, from the index table."""
    text = (H.shared_dir() / 'experiment_priorities.md').read_text()
    rows = []
    for ln in text.splitlines():
        if not re.match(r'\| E\d+ \|', ln):
            continue
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        if len(cells) < 8 or not cells[2].startswith('completed'):
            continue
        m = re.search(r'(\d\d)-(\d\d) ~?(\d\d):(\d\d)', cells[7])
        if not m:
            continue
        mo, d, hh, mm = (int(x) for x in m.groups())
        rows.append(dict(exp=cells[0], name=cells[1], completed=datetime(YEAR, mo, d, hh, mm)))
    return pd.DataFrame(rows).sort_values('completed')


def load_tables(folder: Path) -> pd.DataFrame:
    frames = []
    for name in TABLE_NAMES:
        p = folder / name
        if p.exists():
            frames.append(pd.read_csv(p))
    for p in folder.glob('E*_summary.csv'):
        frames.append(pd.read_csv(p))
    if not frames:
        return merge_apl_effect(folder, pd.DataFrame())
    df = pd.concat(frames, ignore_index=True, sort=False)
    if 'tag' not in df.columns:
        df['tag'] = df['variant'] if 'variant' in df.columns else [f'row{i}' for i in range(len(df))]
    df['tag'] = df['tag'].astype(str)
    df = merge_apl_effect(folder, df)
    if '__ratio__' not in df.columns and {'rate_on', 'rate_off'} <= set(df.columns):
        df['__ratio__'] = df['rate_off'] / df['rate_on'].replace(0, np.nan)
    return df


def merge_apl_effect(folder: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Adds the per-pair APL-effect columns of `folder`'s variants (apl_effect_scores.py) to its
    result table by tag; variants the table lacks are appended with their response rates."""
    if not APL_EFFECT.exists():
        return df
    ae = pd.read_csv(APL_EFFECT)
    ae = ae[ae.folder == folder.name]
    if ae.empty:
        return df
    ae = ae[['tag'] + _APL_EFFECT_COLS + ['rate_on', 'rate_off']].copy()
    ae['tag'] = ae['tag'].astype(str)
    if df.empty:
        return ae
    df = df.drop(columns=[c for c in _APL_EFFECT_COLS if c in df.columns]).merge(ae[['tag'] + _APL_EFFECT_COLS], on='tag', how='left')
    extra = ae[~ae.tag.isin(set(df.tag))]
    return pd.concat([df, extra], ignore_index=True, sort=False) if len(extra) else df


def best_rows(exp: str, df: pd.DataFrame) -> list:
    if 'rate_on' in df.columns:
        df = df[df['rate_on'].between(*RATE_RANGE) | df['rate_on'].isna()]
    keep = ~df['tag'].str.contains(EXCLUDE_VARIANT)
    if 'variant' in df.columns:
        keep &= ~df['variant'].astype(str).str.contains(EXCLUDE_VARIANT)
    if 'gain' in df.columns:
        keep &= df['gain'].isna() | (df['gain'] == 1.0)
    if 'kind' in df.columns:
        keep &= ~df['kind'].astype(str).str.contains('oracle|blockset', case=False)
    df = df[keep]
    out = []
    for key, (label, target, cols) in METRICS.items():
        d = realistic(df) if key in REALISM_METRICS else df
        col = next((c for c in cols if c in d.columns), None)
        if col is None:
            continue
        vals = pd.to_numeric(d[col], errors='coerce')
        if vals.notna().sum() == 0:
            continue
        i = (vals - target).abs().idxmin()
        out.append(dict(exp=exp, metric=key, value=float(vals[i]), target=target, variant=str(d.loc[i, 'tag']),
            params=describe_params(d.loc[i]), stats=describe_stats(d.loc[i]), n_variants=int(vals.notna().sum())))
    return out


# the APL-effect series count only variants inside the realism bands (CLAUDE.md, "The main goal for new
# experiments"; apl_effect_scores.py): the loss's mean part can be bought by moving the response rates
REALISM_METRICS = {'apl_loss', 'apl_rho'}


def realistic(df: pd.DataFrame) -> pd.DataFrame:
    """The rows inside the realism bands: APL-on response rate and the APL-off / APL-on ratio in
    their bands, and the APL-on response rate ρ not below its minimum where recorded."""
    from apl_effect_scores import RATE_ON_BAND, RATE_RATIO_BAND, RATE_RHO_MIN
    if not {'rate_on', 'rate_off'} <= set(df.columns):
        return df.iloc[0:0]
    on, off = pd.to_numeric(df['rate_on'], errors='coerce'), pd.to_numeric(df['rate_off'], errors='coerce')
    ok = on.between(*RATE_ON_BAND) & (off / on).between(*RATE_RATIO_BAND)
    if 'spearman_rate_on_vs_data' in df.columns:
        rr = pd.to_numeric(df['spearman_rate_on_vs_data'], errors='coerce')
        ok &= rr.isna() | (rr >= RATE_RHO_MIN)
    return df[ok]


def collect() -> pd.DataFrame:
    idx = experiment_index()
    rows = []
    for folder in sorted((H.shared_dir() / 'results').glob('E*')):
        exp = folder.name.split('_')[0]
        if folder.name in SKIP_FOLDERS or exp in EXCLUDE or exp not in set(idx.exp):
            continue
        df = load_tables(folder)
        if df.empty:
            continue
        rows += best_rows(exp, df)
    pts = pd.DataFrame(rows).merge(idx, on='exp').sort_values(['metric', 'completed', 'exp'])
    # gap closed relative to the first measured model of each statistic
    pts['baseline'] = pts.groupby('metric')['value'].transform('first')
    pts['gap_closed'] = 1 - (pts['value'] - pts['target']).abs() / (pts['baseline'] - pts['target']).abs()
    pts['running_best'] = pts.groupby('metric')['gap_closed'].cummax()
    pts['new_best'] = pts['gap_closed'] >= pts['running_best'] - 1e-12
    return pts


def plot_html(pts: pd.DataFrame) -> str:
    keys = [k for k in METRICS if k in set(pts.metric)]
    series = []
    for k in keys:
        sub = pts[pts.metric == k]
        label, target, _ = METRICS[k]
        series.append(dict(key=k, label=label, target=target, target_label=TARGET_LABELS[k], light=SERIES_COLORS[k][0], dark=SERIES_COLORS[k][1],
            baseline=float(sub.baseline.iloc[0]),
            points=[dict(t=r.completed.isoformat(timespec='minutes'), exp=r.exp, name=r['name'], variant=r.variant,
                params=r.params, stats=r['stats'], value=round(float(r.value), 4), gap=round(float(r.gap_closed), 4),
                best=round(float(r.running_best), 4), new=bool(r.new_best)) for _, r in sub.iterrows()]))
    data_json = json.dumps(series).replace('&', '&amp;').replace('</', '<\\/')
    # the table view (the running best per statistic, with the model that set it)
    best_tbl = ['<table><tr><th>statistic</th><th>KC data</th><th class="num">first model</th><th class="num">best so far</th><th>set by</th></tr>']
    for s in series:
        last = max(s['points'], key=lambda p: (p['best'], p['t']))
        setter = next(p for p in s['points'] if abs(p['best'] - last['best']) < 1e-9 and p['new'])
        best_tbl.append(f"<tr><td>{html.escape(s['label'])}</td><td class=\"num\">{s['target_label']}</td>"
            f"<td class=\"num\">{s['baseline']:.3g}</td><td class=\"num\">{setter['value']:.3g} ({100 * last['best']:.0f} % of the gap)</td>"
            f"<td>{html.escape(setter['exp'] + ': ' + setter['name'])}, {html.escape(setter['variant'])}, {setter['t'].replace('T', ' ')}</td></tr>")
    best_tbl.append('</table>')
    css = """
<style>
.prog { position: relative; --p-ink: var(--ink, #1f2328); --p-muted: var(--muted, #57606a); --p-line: var(--line, #d8dee4); --p-bg: var(--bg, #fff); margin: 14px 0 6px; }
.prog svg { width: 100%; height: auto; display: block; font: 12px -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
.prog .grid line { stroke: var(--p-line); stroke-width: 1; }
.prog .axis text { fill: var(--p-muted); }
.prog .axis line, .prog .axis path { stroke: var(--p-line); }
.prog .step { fill: none; stroke-width: 2; stroke-linejoin: round; }
.prog .target { stroke: var(--p-ink); stroke-width: 1.5; stroke-dasharray: 2,4; }
.prog .legend .item.fixed { cursor: default; } .prog .legend .sw.target { background: none; border: 0; height: 0; width: 22px; border-top: 2px dotted var(--p-ink); border-radius: 0; }
.prog .pt { stroke: var(--p-bg); stroke-width: 1.5; cursor: pointer; }
.prog .pt.hit { fill: transparent; stroke: none; cursor: pointer; }
.prog .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 8px 0 0; font-size: 0.86em; color: var(--p-ink); }
.prog .legend .item { display: flex; align-items: center; gap: 7px; cursor: pointer; user-select: none; }
.prog .legend .sw { width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.15); }
.prog .legend .item.off { opacity: 0.4; } .prog .legend .item.off .sw { background: var(--p-line) !important; }
.prog .tip { position: absolute; pointer-events: none; background: var(--p-bg); color: var(--p-ink); border: 1px solid var(--p-line); border-radius: 6px; padding: 7px 9px; font-size: 0.82em; line-height: 1.35; box-shadow: 0 2px 8px rgba(0,0,0,0.12); max-width: 360px; display: none; z-index: 5; }
.prog .tip b { font-weight: 600; }
.prog .tip .params { color: var(--p-muted); }
.prog .tip.pinned { pointer-events: auto; user-select: text; cursor: auto; border-color: var(--p-muted); box-shadow: 0 4px 16px rgba(0,0,0,0.22); padding-right: 26px; }
.prog .tip .tip-x { position: absolute; top: 3px; right: 4px; border: 0; background: none; color: var(--p-muted); font: 16px/1 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; padding: 2px 5px; cursor: pointer; border-radius: 4px; }
.prog .tip .tip-x:hover { color: var(--p-ink); background: var(--p-line); }
.prog .step.hit { cursor: pointer; }
.prog .note { color: var(--p-muted); font-size: 0.86em; margin-top: 6px; }
.prog details { margin-top: 8px; font-size: 0.9em; } .prog details summary { cursor: pointer; color: var(--p-muted); }
@media (prefers-color-scheme: dark) { .prog .legend .sw { border-color: rgba(255,255,255,0.2); } }
</style>"""
    script = """
<script>
(function () {
  var root = document.getElementById('progress-plot');
  var data = JSON.parse(root.getAttribute('data-series'));
  var dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  var W = 920, Hh = 420, m = {l: 58, r: 18, t: 16, b: 46};
  var iw = W - m.l - m.r, ih = Hh - m.t - m.b;
  var times = [];
  data.forEach(function (s) { s.points.forEach(function (p) { p.time = new Date(p.t).getTime(); times.push(p.time); }); });
  var t0 = Math.min.apply(null, times), t1 = Math.max.apply(null, times);
  var pad = (t1 - t0) * 0.03 || 3600000; t0 -= pad; t1 += pad;
  var yMin = -0.5, yMax = 1.0;
  function x(t) { return m.l + (t - t0) / (t1 - t0) * iw; }
  function y(v) { v = Math.max(yMin, Math.min(yMax, v)); return m.t + (yMax - v) / (yMax - yMin) * ih; }
  var NS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs, parent) { var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
  var svg = el('svg', {viewBox: '0 0 ' + W + ' ' + Hh, role: 'img', 'aria-label': 'Fraction of the gap to the KC data closed by the best model so far, per statistic, against completion time'}, root.querySelector('.chart'));
  var grid = el('g', {'class': 'grid'}, svg), axis = el('g', {'class': 'axis'}, svg);
  var ticks = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1];
  ticks.forEach(function (v) {
    if (v !== 1) el('line', {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), 'stroke-dasharray': v === 0 ? '' : '2,4'}, grid);
    var tx = el('text', {x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end'}, axis); tx.textContent = (v * 100).toFixed(0) + ' %';
  });
  // the KC data (each statistic's value or measurement ceiling) is 100 % by construction
  el('line', {x1: m.l, x2: W - m.r, y1: y(1), y2: y(1), 'class': 'target'}, svg);
  var lab = el('text', {x: 14, y: m.t + ih / 2, transform: 'rotate(-90 14 ' + (m.t + ih / 2) + ')', 'text-anchor': 'middle'}, axis);
  lab.textContent = 'gap to the KC data closed by the best model so far';
  // time ticks every 6 h on the local clock (00, 06, 12, 18); the date on the first tick and at midnight
  var d0 = new Date(t0); d0.setMinutes(0, 0, 0); d0.setHours(Math.ceil(d0.getHours() / 6) * 6);
  var firstTick = true;
  for (var d = new Date(d0); d.getTime() <= t1; d.setHours(d.getHours() + 6)) {
    var t = d.getTime(), hh = d.getHours();
    el('line', {x1: x(t), x2: x(t), y1: m.t + ih, y2: m.t + ih + 5}, axis);
    var tt = el('text', {x: x(t), y: m.t + ih + 18, 'text-anchor': 'middle'}, axis);
    tt.textContent = (hh === 0 || firstTick) ? (d.getMonth() + 1) + '/' + d.getDate() + ' ' + ('0' + hh).slice(-2) + ':00' : ('0' + hh).slice(-2) + ':00';
    firstTick = false;
  }
  var xl = el('text', {x: m.l + iw / 2, y: Hh - 8, 'text-anchor': 'middle'}, axis); xl.textContent = 'experiment completion time (2026)';
  el('line', {x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih}, axis);
  var tip = root.querySelector('.tip');
  // a left click on a hovered point or step pins its popup (selectable, with a close control) until
  // the x, Esc or a click anywhere outside it; hovering other marks does not replace a pinned popup
  var pinned = false, pinEvent = null, downInTip = false;
  // the model's parameters (set or tuned), then statistics of its responses (measured), each
  // de-emphasized and only when present
  function modelText(p) {
    var h = p.params ? '<br>model parameters: <span class="params">' + p.params + '</span>' : '';
    if (p.stats) h += '<br>model stats:<br><span class="params">' + p.stats + '</span>';
    return h;
  }
  function showTip(evt, s, p) {
    if (pinned) return;
    var pct = Math.round(p.gap * 100);
    tip.innerHTML = '<b>' + p.exp + ': ' + p.name + '</b><br>completed ' + p.t.replace('T', ' ') + '<br>' + s.label + ': <b>' + p.value + '</b> (KC data ' + s.target_label + '; first model ' + s.baseline.toFixed(3) + ')<br>best variant: ' + p.variant + modelText(p) + '<br>' + pct + ' % of the gap closed' + (p.new ? ' (a new best at the time)' : '');
    placeTip(evt);
  }
  function showSegTip(evt, s, p, untilText) {
    if (pinned) return;
    tip.innerHTML = '<b>best so far on: ' + s.label + '</b><br>from ' + p.t.replace('T', ' ') + ' ' + untilText + '<br>' + p.exp + ': ' + p.name + ', variant ' + p.variant + '<br>' + s.label + ' <b>' + p.value + '</b> (KC data ' + s.target_label + '; ' + Math.round(p.best * 100) + ' % of the gap closed)' + modelText(p);
    placeTip(evt);
  }
  function placeTip(evt) {
    tip.style.display = 'block';
    var r = root.getBoundingClientRect();
    var left = evt.clientX - r.left + 14, top = evt.clientY - r.top - 10;
    if (left + 430 > r.width) left = Math.max(0, evt.clientX - r.left - 440);
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
    // keep the popup over the chart rather than the legend: flip it above the cursor when it
    // would run past the chart's bottom edge
    var chart = root.querySelector('.chart').getBoundingClientRect();
    var h = tip.offsetHeight;
    if (top + h > chart.height) tip.style.top = Math.max(0, evt.clientY - r.top - h - 12) + 'px';
  }
  function hideTip() { if (!pinned) tip.style.display = 'none'; }
  function unpin() { pinned = false; pinEvent = null; tip.classList.remove('pinned'); tip.style.display = 'none'; }
  function pin(evt, show) {
    if (evt.button !== 0) return;
    unpin(); show(evt);
    pinned = true; pinEvent = evt; tip.classList.add('pinned');
    var btn = document.createElement('button'); btn.type = 'button'; btn.className = 'tip-x';
    btn.setAttribute('aria-label', 'close'); btn.title = 'close (Esc)'; btn.innerHTML = '&times;';
    btn.addEventListener('click', unpin);
    tip.appendChild(btn);
  }
  document.addEventListener('mousedown', function (evt) { downInTip = tip.contains(evt.target); });
  document.addEventListener('click', function (evt) {
    // (a drag that selects popup text and ends outside it is not a click outside)
    if (!pinned || evt === pinEvent || evt.button !== 0 || downInTip || tip.contains(evt.target)) return;
    unpin();
  });
  document.addEventListener('keydown', function (evt) { if (pinned && (evt.key === 'Escape' || evt.key === 'Esc')) unpin(); });
  var groups = {};
  data.forEach(function (s, j) {
    var color = dark ? s.dark : s.light;
    var g = el('g', {'data-key': s.key}, svg); groups[s.key] = g;
    // running best as a staircase (step after): one segment per model that set the best, so
    // hovering a segment names the model (and its parameters) that held the best over that span
    var setters = s.points.filter(function (p) { return p.new; });
    setters.forEach(function (p, i) {
      var next = setters[i + 1];
      var xStart = x(p.time), xEnd = next ? x(next.time) : W - m.r, yy = y(p.best);
      var d = 'M' + xStart + ' ' + (i === 0 ? yy : y(setters[i - 1].best)) + ' V' + yy + ' H' + xEnd;
      el('path', {d: d, 'class': 'step', stroke: color}, g);
      var hit = el('path', {d: d, 'class': 'step hit', stroke: 'transparent', 'stroke-width': 12}, g);
      var until = next ? 'until ' + next.t.replace('T', ' ') : 'to now';
      var showSeg = function (evt) { showSegTip(evt, s, p, until); };
      hit.addEventListener('mousemove', showSeg);
      hit.addEventListener('mouseleave', hideTip);
      hit.addEventListener('click', function (evt) { pin(evt, showSeg); });
    });
    s.points.forEach(function (p) {
      var c = el('circle', {cx: x(p.time), cy: y(p.gap), r: p.new ? 5 : 4, fill: color, 'fill-opacity': p.new ? 1 : 0.55, 'class': 'pt'}, g);
      var hit = el('circle', {cx: x(p.time), cy: y(p.gap), r: 11, 'class': 'pt hit'}, g);
      var showPt = function (evt) { showTip(evt, s, p); };
      [c, hit].forEach(function (e) {
        e.addEventListener('mousemove', showPt);
        e.addEventListener('mouseleave', hideTip);
        e.addEventListener('click', function (evt) { pin(evt, showPt); });
      });
    });
  });
  var legend = root.querySelector('.legend');
  var fixed = document.createElement('span'); fixed.className = 'item fixed';
  var fsw = document.createElement('span'); fsw.className = 'sw target';
  var ftxt = document.createElement('span'); ftxt.textContent = 'KC data: the target of every statistic (its value, or its measurement ceiling where the data set one)';
  fixed.appendChild(fsw); fixed.appendChild(ftxt); legend.appendChild(fixed);
  data.forEach(function (s) {
    var color = dark ? s.dark : s.light;
    var item = document.createElement('span'); item.className = 'item';
    var sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = color;
    var txt = document.createElement('span'); txt.textContent = s.label + ' (KC data ' + s.target_label + ')';
    item.appendChild(sw); item.appendChild(txt); legend.appendChild(item);
    item.addEventListener('click', function () {
      var off = item.classList.toggle('off');
      groups[s.key].style.display = off ? 'none' : '';
    });
  });
})();
</script>"""
    return (css + '\n<figure class="prog" id="progress-plot" data-series=\'' + data_json.replace("'", '&#39;') + "'>\n"
        '<div class="chart" style="position:relative"></div><div class="tip"></div>\n<div class="legend"></div>\n'
        '<div class="note">Each point is one experiment\'s best model on that statistic (filled and larger when it was the best so far; points below -50 % are drawn at -50 %); '
        'the staircase is the best so far. Hover a point for the experiment, the model variant, its parameters, its response rates and the raw value, or a step of the staircase for the model that held the best over that span, and click it to keep that popup open (close it with its x, Esc or a click elsewhere); click a legend entry to hide or show its line. '
        'Oracle variants (co-responder suppression, the oracle glomerulus gain, the block-specific glomerulus set) are not counted.</div>\n'
        '<details><summary>Table: the best so far on each statistic</summary>' + ''.join(best_tbl) + '</details>\n'
        '</figure>' + script + '\n')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pts = collect()
    pts.to_csv(OUT / 'progress_points.csv', index=False)
    best = pts[pts.new_best][['metric', 'exp', 'name', 'completed', 'variant', 'value', 'target', 'gap_closed']]
    best.to_csv(OUT / 'progress_best.csv', index=False)
    (OUT / 'progress_plot.html').write_text(plot_html(pts))
    pd.set_option('display.width', 200)
    summ = pts.groupby('metric').agg(experiments=('exp', 'nunique'), first=('baseline', 'first'), best_value=('value', lambda v: v.iloc[(v - pts.loc[v.index, 'target']).abs().argmin()]),
        best_gap=('running_best', 'max'))
    print(summ.round(3).to_string())
    print(f"wrote {OUT / 'progress_plot.html'} ({(OUT / 'progress_plot.html').stat().st_size // 1024} KB), {len(pts)} points over {pts.exp.nunique()} experiments")


if __name__ == '__main__':
    main()
