#!/usr/bin/env python3
"""The per-pair APL effect on odor-odor distances for every saved model variant, and the
distance-matrix figures the summary's best-models table opens from its buttons.

The main target since 2026-11-09 17:55 (Tom; CLAUDE.md "The main goal for new experiments"): for each
of the 120 odor pairs, the distance with APL on minus the distance with APL off; in the KC data the
control flies' cosine distance minus the distance with APL output blocked (TNT), each the mean over
recordings, in a model its APL-on minus its acute APL-off distance (cosine on spike counts). Scored
by `kc_remapping_harness.apl_effect_stats` (the APL-effect loss, the root mean squared difference
from the data's per-pair effect over the 120 pairs, with its mean and pattern parts).

Reads every <shared>/results/E<n>_*/model_pair_distances_long.csv (the spike-count cosine rows where
the table has a `variant` column; conditions `on` and `acute_off`), takes each variant's response
rates from the experiment's result tables (by tag or variant; else from the model-output archive's
response matrices), flags variants outside the realism bands (CLAUDE.md), and writes

    <shared>/results/apl_effect/by_variant.csv           one row per experiment folder x tag
    <shared>/results/apl_effect/best_by_experiment.csv   each experiment's lowest APL-effect loss,
                                                         among variants with no realism flag and the
                                                         lowest overall
    <shared>/results/apl_effect/matrices/<name>.png      for every `!button[...](results/apl_effect/
                                                         matrices/<name>.png)` in the summary template:
                                                         `data` (the KC data's APL-on, APL-off and
                                                         difference matrices) or `<folder>__<tag>`
                                                         (the data's row above the model's)
    <shared>/results/apl_effect/provenance.json

The bands' response criterion (Tom, 2026-11-10 16:10): two or more spikes on the model's side; on the KC data's
side each KC's trial-mean response in units of its baseline noise, above the registry's noise_threshold (1.97),
which keeps the control flies at their rate under Yang's criterion. `realism_flag` / `realism` use it, with
`opk_<on|off>_tv` (the odors-per-KC distances, against the registry's noise_<ctrl|tnt>_odors_per_kc_*) and
`rate_rho_<on|off>` (the response rate ρ, the model's per-odor rates from the archived responses against the
noise_<ctrl|tnt>_frac_<odor> entries); `realism_flag_zcrit` / `realism_zcrit` keep the bands as they were until
then (the KC data in Yang's whole-recording z, above 1 z: `opk_<on|off>_tv_zcrit`, and the response rate ρ
of the experiment's own table, `spearman_rate_on_vs_data`), so that earlier write-ups can be checked.

build_summary.py runs it before the progress plot. Usage: ./venv/bin/python apl_effect_scores.py
[--no-figures] [--only-figures]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

RESULTS = H.shared_dir() / 'results'
OUT = RESULTS / 'apl_effect'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
TEMPLATE = H.HERE / 'reports' / 'summary_template.md'
V = 'spike_counts_cosine'
STAT_KEYS = ('apl_effect_rmse', 'apl_effect_rmse_centered', 'apl_effect_mean', 'apl_effect_mean_error',
    'apl_effect_spearman', 'apl_effect_pearson', 'apl_effect_n_pairs')
# the realism bands (CLAUDE.md, "The main goal for new experiments"): APL-on response rate (the control
# flies' 0.068; models tuned at 0.05 or 0.068), the APL-off / APL-on response rate ratio (the data's 2.28 in
# baseline-noise units, 90 % interval 2.0-2.7 over recordings; 2.2 and 1.9-2.5 in Yang's z; a ratio band rather
# than an APL-off rate band, so that models tuned at 5 % are judged by the same proportion), and the APL-on
# response rate ρ no more than 0.10 below the reference model's
RATE_ON_BAND = (0.045, 0.085)
RATE_RATIO_BAND = (1.6, 2.9)
RATE_RHO_MIN = 0.47 - 0.10
# odors per KC (Tom, 2026-11-09 23:40): the total variation distance between the model's distribution of the number of
# odors each KC responds to (0, 1, 2-3, 4-7, 8 or more of the 16) and the flies' mean distribution, with APL on against
# the control flies and with APL off against APL output blocked (TNT), at most the most atypical recorded fly's distance
# from the other flies' mean (registry noise_ctrl_ / noise_tnt_odors_per_kc_tv_max; ctrl_ / tnt_odors_per_kc_tv_max in
# Yang's z)
OPK_BINS = ('0', '1', '2_3', '4_7', '8plus')
TABLE_GLOBS = ('stats_variants.csv', 'fits.csv', 'stage2_fits.csv', 'wkckc_controls.csv', 'E*_summary.csv', 'comparison.csv')
BUTTON = re.compile(r'!button\[([^\]]+)\]\(results/apl_effect/matrices/([^)\s]+)\.png\)')


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pair_series(g: pd.DataFrame) -> pd.Series:
    return H.canonical_pairs(g.set_index(['odor_a', 'odor_b'])['distance'])


def model_pairs(folder: Path) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """{tag: (APL-on pairs, acute APL-off pairs)} of a results folder's saved distances."""
    p = folder / 'model_pair_distances_long.csv'
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if 'variant' in df.columns:
        df = df[df['variant'].astype(str) == V]
    df = df[df['tag'].astype(str) != 'tag']
    out = {}
    for tag, g in df.groupby(df['tag'].astype(str), sort=False):
        on, off = g[g.condition == 'on'], g[g.condition == 'acute_off']
        if len(on) and len(off):
            out[tag] = (pair_series(on), pair_series(off))
    return out


def rate_lookup(folder: Path) -> Dict[str, dict]:
    """{tag or variant: dict(rate_on, rate_off, spearman_rate_on_vs_data, source)} from the
    experiment's result tables (the first table that has the key wins)."""
    out: Dict[str, dict] = {}
    for pat in TABLE_GLOBS:
        for p in sorted(folder.glob(pat)):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            if 'rate_on' not in df.columns:
                continue
            keys = [c for c in ('tag', 'variant') if c in df.columns]
            for _, r in df.iterrows():
                vals = dict(rate_on=r.get('rate_on'), rate_off=r.get('rate_off'),
                    spearman_rate_on_vs_data=r.get('spearman_rate_on_vs_data', np.nan), source=p.name)
                names = [str(r[k]) for k in keys]
                if {'base', 'variant'} <= set(df.columns):
                    names.append(f"{r['base']}__{r['variant']}")
                for n in names:
                    out.setdefault(n, vals)
    return out


def archive_rates(folder: Path, tag: str) -> Optional[dict]:
    d = ARCHIVE / folder.name
    on, off = d / f'{tag}_on_responses.parquet', d / f'{tag}_acute_off_responses.parquet'
    if not (on.exists() and off.exists()):
        return None
    rate = lambda p: float(pd.read_parquet(p).astype(float).mean().mean())
    return dict(rate_on=rate(on), rate_off=rate(off), spearman_rate_on_vs_data=np.nan, source='model-output archive')


def odors_per_kc(responses: pd.DataFrame) -> np.ndarray:
    """Fractions of KCs responding to 0, 1, 2-3, 4-7 and 8 or more of the odors (`OPK_BINS`), as
    E10_joint_fit.odors_per_kc."""
    n = (responses.to_numpy(dtype=float) > 0.5).sum(axis=1)
    return np.array([np.mean(n == 0), np.mean(n == 1), np.mean((n >= 2) & (n <= 3)), np.mean((n >= 4) & (n <= 7)),
        np.mean(n >= 8)])


_OPK_DATA: Dict[Tuple[str, str], np.ndarray] = {}
# the KC data's units for the bands: 'noise' (each KC's responses in units of its baseline noise, above the registry's
# noise_threshold; the bands' data side since Tom's decision of 2026-11-10 16:10) or 'z' (Yang's whole-recording z, above
# 1 z; the bands until then, kept for the record as realism_flag_zcrit, and the data side of E73's per-KC session z-score
# reading). The model side is two or more spikes under both.
BAND_UNITS = ('z', 'noise')
_PREFIX = {'z': '', 'noise': 'noise_'}


def _geno(cond: str) -> str:
    return 'ctrl' if cond == 'on' else 'tnt'


def opk_data(cond: str, units: str = 'z') -> np.ndarray:
    """The flies' mean odors-per-KC distribution: control flies for 'on', APL output blocked (TNT) for 'off', in
    `units` (`BAND_UNITS`)."""
    if (cond, units) not in _OPK_DATA:
        _OPK_DATA[(cond, units)] = np.array([H.data_target(f'{_PREFIX[units]}{_geno(cond)}_odors_per_kc_{b}') for b in OPK_BINS])
    return _OPK_DATA[(cond, units)]


def opk_band(cond: str, units: str = 'z') -> float:
    """The realism band on odors per KC (the most atypical recording's distance from the others) in `units`."""
    return H.data_target(f'{_PREFIX[units]}{_geno(cond)}_odors_per_kc_tv_max')


_FRAC_DATA: Dict[Tuple[str, str], pd.Series] = {}


def frac_data(cond: str, units: str = 'z') -> pd.Series:
    """The flies' per-odor response fractions (mean over recordings, `APL16_ODORS` order): control flies for 'on',
    APL output blocked (TNT) for 'off', in `units` ('z': `kc_remapping_harness.yang_response_fractions`; 'noise':
    the registry's noise_<ctrl|tnt>_frac_<odor>)."""
    if (cond, units) not in _FRAC_DATA:
        if units == 'z':
            s = H.yang_response_fractions('TNT3c' if cond == 'on' else 'TNT')[H.APL16_ODORS].mean()
        elif units == 'noise':
            s = pd.Series({o: H.data_target(f'noise_{_geno(cond)}_frac_{o}') for o in H.APL16_ODORS})
        else:
            raise ValueError(f'{units=} not in {BAND_UNITS}')
        _FRAC_DATA[(cond, units)] = s
    return _FRAC_DATA[(cond, units)]


def data_rate_ratio(units: str = 'z') -> Tuple[float, Tuple[float, float]]:
    """The flies' APL-off / APL-on response rate ratio and its 90 % interval over recordings in `units`."""
    name = f'{_PREFIX[units]}rate_ratio_fraction'
    return H.data_target(name), H.data_target_ci(name)


def archive_opk(folder: Path, tag: str) -> dict:
    """From the model-output archive's response matrices (two or more spikes): opk_<on|off>_<bin>, opk_<on|off>_tv
    (total variation distance from the flies' mean distribution in baseline-noise units), opk_<on|off>_tv_zcrit (the
    same against Yang's z units) and rate_rho_<on|off> (the response rate ρ: Spearman of the model's per-odor
    response rates with the flies' in baseline-noise units); empty where the archive lacks them."""
    out = {}
    for cond, name in (('on', 'on'), ('off', 'acute_off')):
        p = ARCHIVE / folder.name / f'{tag}_{name}_responses.parquet'
        if not p.exists():
            continue
        r = pd.read_parquet(p)
        d = odors_per_kc(r)
        out.update({f'opk_{cond}_{b}': float(v) for b, v in zip(OPK_BINS, d)})
        out[f'opk_{cond}_tv'] = float(0.5 * np.abs(d - opk_data(cond, 'noise')).sum())
        out[f'opk_{cond}_tv_zcrit'] = float(0.5 * np.abs(d - opk_data(cond, 'z')).sum())
        per_odor = H.abbrev_columns(r.astype(float) > 0.5).mean().reindex(H.APL16_ODORS)
        out[f'rate_rho_{cond}'] = H.spearman(per_odor, frac_data(cond, 'noise'))
    return out


def realism_flag(rate_on, rate_off, rate_rho, opk_on_tv=np.nan, opk_off_tv=np.nan, units: str = 'z') -> str:
    """'' when inside every band, else the reasons ('; '-joined). `rate_rho` and the odors-per-KC distances must be
    against the KC data in `units` (`BAND_UNITS`), which sets the odors-per-KC bands."""
    why = []
    f = lambda x: x is not None and not (isinstance(x, float) and np.isnan(x))
    if f(rate_on) and not RATE_ON_BAND[0] <= float(rate_on) <= RATE_ON_BAND[1]:
        why.append(f'APL-on response rate {float(rate_on):.3f} (band {RATE_ON_BAND[0]}-{RATE_ON_BAND[1]})')
    if f(rate_on) and f(rate_off) and float(rate_on) > 0:
        ratio = float(rate_off) / float(rate_on)
        if not RATE_RATIO_BAND[0] <= ratio <= RATE_RATIO_BAND[1]:
            why.append(f'APL-off / APL-on response rate {ratio:.2f} (band {RATE_RATIO_BAND[0]}-{RATE_RATIO_BAND[1]})')
    if f(rate_rho) and float(rate_rho) < RATE_RHO_MIN:
        why.append(f'response rate ρ, APL on, {float(rate_rho):.2f} (below {RATE_RHO_MIN:.2f})')
    for cond, tv, who in (('on', opk_on_tv, 'APL on'), ('off', opk_off_tv, 'APL off')):
        if f(tv) and float(tv) > opk_band(cond, units):
            why.append(f'odors per KC, {who}, {float(tv):.2f} from the flies (band {opk_band(cond, units):.2f})')
    return '; '.join(why)


def score_all(data: pd.Series) -> pd.DataFrame:
    rows = []
    for folder in sorted(RESULTS.glob('E*_*')):
        m = re.match(r'(E\d+)_', folder.name)
        if not m or not folder.is_dir():
            continue
        pairs = model_pairs(folder)
        if not pairs:
            continue
        rates = rate_lookup(folder)
        for tag, (on, off) in pairs.items():
            st = H.apl_effect_stats(on, off, data)
            if st['apl_effect_n_pairs'] < 100:
                continue
            rr = rates.get(tag) or archive_rates(folder, tag) or {}
            row = dict(exp=m.group(1), folder=folder.name, tag=tag, **{f'{V}__{k}': v for k, v in st.items()},
                rate_on=rr.get('rate_on', np.nan), rate_off=rr.get('rate_off', np.nan),
                spearman_rate_on_vs_data=rr.get('spearman_rate_on_vs_data', np.nan), rate_source=rr.get('source', ''))
            row.update(archive_opk(folder, tag))
            missing = np.isnan(row.get('opk_on_tv', np.nan)) or np.isnan(row.get('opk_off_tv', np.nan))
            # the bands as decided on 2026-11-10 16:10: the model's two-spike responses against the KC data in
            # baseline-noise units (the response rate ρ from the archived responses)
            row['realism_flag'] = realism_flag(row['rate_on'], row['rate_off'], row.get('rate_rho_on', np.nan),
                row.get('opk_on_tv', np.nan), row.get('opk_off_tv', np.nan), units='noise')
            row['realism'] = 'inside the bands' if not row['realism_flag'] else 'outside: ' + row['realism_flag']
            if missing:
                row['realism'] += ' (odors per KC and the response rate ρ not scored: no archived responses)'
            # the bands until then, against Yang's z units (the response rate ρ from the experiment's own table)
            row['realism_flag_zcrit'] = realism_flag(row['rate_on'], row['rate_off'], row['spearman_rate_on_vs_data'],
                row.get('opk_on_tv_zcrit', np.nan), row.get('opk_off_tv_zcrit', np.nan), units='z')
            row['realism_zcrit'] = 'inside the bands' if not row['realism_flag_zcrit'] else 'outside: ' + row['realism_flag_zcrit']
            if missing:
                row['realism_zcrit'] += ' (odors per KC not scored: no archived responses)'
            rows.append(row)
    return pd.DataFrame(rows)


def best_by_experiment(df: pd.DataFrame) -> pd.DataFrame:
    col = f'{V}__apl_effect_rmse'
    out = []
    for (exp, folder), g in df.groupby(['exp', 'folder'], sort=False):
        best = g.loc[g[col].idxmin()]
        ok = g[g.realism_flag == '']
        best_ok = ok.loc[ok[col].idxmin()] if len(ok) else None
        out.append(dict(exp=exp, folder=folder, n_variants=len(g),
            best_tag=best.tag, best_loss=best[col], best_flag=best.realism_flag,
            best_realistic_tag=None if best_ok is None else best_ok.tag,
            best_realistic_loss=np.nan if best_ok is None else best_ok[col],
            best_realistic_mean=np.nan if best_ok is None else best_ok[f'{V}__apl_effect_mean'],
            best_realistic_rho=np.nan if best_ok is None else best_ok[f'{V}__apl_effect_spearman']))
    return pd.DataFrame(out).sort_values('best_realistic_loss')


# ---------------------------------------------------------------------------- figures

def button_targets(template: str) -> Dict[str, str]:
    """{figure name: label} for every matrices button in the template; the label is the column's
    header in the best-models table (parameters joined by ', ') where the button sits there."""
    labels: Dict[str, str] = {}
    lines = template.split('\n')
    header: Optional[List[str]] = None
    for ln in lines:
        if ln.startswith('## '):
            header = None
        if ln.startswith('| statistic |'):
            header = [c.strip() for c in ln.strip().strip('|').split('|')]
        if '!button[' not in ln:
            continue
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        for k, cell in enumerate(cells):
            for _, name in BUTTON.findall(cell):
                lab = header[k] if header and k < len(header) else name
                labels[name] = re.sub(r'\s*<br>\s*', ', ', re.sub(r'\{span=\d+\}', '', lab)).strip()
    return labels


def square(pairs: pd.Series, order: List[str]) -> np.ndarray:
    sq = np.array(H.square_from_pairs(pairs).reindex(index=order, columns=order).to_numpy(dtype=float))
    np.fill_diagonal(sq, np.nan)
    return sq


def draw(name: str, label: str, data_on: pd.Series, data_off: pd.Series, model: Optional[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = H.odor_order(H.APL16_ODORS)
    d_eff = H.apl_effect(data_on, data_off)
    rows = [('KCs', square(data_on, order), square(data_off, order), square(d_eff, order), None)]
    if model is not None:
        m_eff = H.apl_effect(model['on'], model['off'])
        rows.append(('model', square(model['on'], order), square(model['off'], order), square(m_eff, order), model))
    lim = max(np.nanmax(np.abs(r[3])) for r in rows)
    cmap_d, cmap_e = 'viridis', H.oklab_diverging_cmap()
    fig, axes = plt.subplots(len(rows), 3, figsize=(17, 4.9 * len(rows) + 1.0), squeeze=False,
        gridspec_kw=dict(hspace=0.32, wspace=0.32))
    for i, (who, on, off, eff, mod) in enumerate(rows):
        if who == 'KCs':
            titles = ['KCs, control flies\n(APL on)', 'KCs, APL output blocked (TNT)\n(APL off)', 'KCs: APL on minus APL off\n(mean {:+.3f})'.format(np.nanmean(eff[np.triu_indices(len(order), 1)]))]
        else:
            titles = ['model, APL on', 'model, APL off (acute)',
                'model: APL on minus APL off\n(loss {:.3f}, mean {:+.3f}, ρ {:.2f} with the KCs\')'.format(
                    mod['stats']['apl_effect_rmse'], mod['stats']['apl_effect_mean'], mod['stats']['apl_effect_spearman'])]
        for j, (mat, title) in enumerate(zip((on, off, eff), titles)):
            ax = axes[i, j]
            if j < 2:
                im = ax.imshow(mat, cmap=cmap_d, vmin=0, vmax=1)
                cb_label = 'cosine distance across KC responses' if who == 'KCs' else 'cosine distance across model KC spike counts'
            else:
                im = ax.imshow(mat, cmap=cmap_e, vmin=-lim, vmax=lim)
                cb_label = 'APL on minus APL off: cosine distance'
            ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=8)
            ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
            ax.set_title(title, fontsize=10)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label(cb_label, fontsize=8); cb.ax.tick_params(labelsize=7)
    sup = label
    if model is not None and model.get('rates'):
        r = model['rates']
        f = lambda x: '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.3f}'
        sup += (f"\nresponse rate, APL on / off: {f(r.get('rate_on'))} / {f(r.get('rate_off'))} "
            f"(KCs {H.data_target('noise_ctrl_frac'):.3f} / {H.data_target('noise_tnt_frac'):.3f}, in baseline-noise units)")
        if model.get('flag'):
            sup += f"\nrealism flag: {model['flag']}"
    fig.suptitle(sup, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=95, bbox_inches='tight')
    plt.close(fig)


def make_figures(df: pd.DataFrame, data_on: pd.Series, data_off: pd.Series) -> List[str]:
    labels = button_targets(TEMPLATE.read_text())
    made = []
    for name, label in labels.items():
        out = OUT / 'matrices' / f'{name}.png'
        if name == 'data':
            draw(name, 'KC data (Yang\'s APL16 recordings): the control flies (APL on), APL output blocked (TNT) (APL off) and their difference',
                data_on, data_off, None, out)
            made.append(name)
            continue
        folder, _, tag = name.partition('__')
        pairs = model_pairs(RESULTS / folder).get(tag)
        if pairs is None:
            raise KeyError(f'button {name!r}: no saved distances for tag {tag!r} in results/{folder}')
        row = df[(df.folder == folder) & (df.tag == tag)]
        r = row.iloc[0] if len(row) else None
        model = dict(on=pairs[0], off=pairs[1], stats=H.apl_effect_stats(*pairs, H.apl_effect(data_on, data_off)),
            rates=None if r is None else dict(rate_on=r.rate_on, rate_off=r.rate_off),
            flag=r.realism_flag if r is not None and isinstance(r.realism_flag, str) else '')
        draw(name, f'{label}  ({folder}, {tag})', data_on, data_off, model, out)
        made.append(name)
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--only-figures', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data_on, data_off = H.yang_mean_rdm('TNT3c'), H.yang_mean_rdm('TNT')
    data = H.apl_effect(data_on, data_off)
    if args.only_figures:
        df = pd.read_csv(OUT / 'by_variant.csv')
    else:
        df = score_all(data)
        df.to_csv(OUT / 'by_variant.csv', index=False)
        best = best_by_experiment(df)
        best.to_csv(OUT / 'best_by_experiment.csv', index=False)
        new, old = df.realism_flag != '', df.realism_flag_zcrit != ''
        print(f'{len(df)} variants from {df.folder.nunique()} result folders; {int(new.sum())} flagged (against the KC data in '
              f'baseline-noise units; {int(old.sum())} against Yang\'s z: {int((new & ~old).sum())} newly flagged, '
              f'{int((old & ~new).sum())} cleared); {int(df.rate_on.isna().sum())} without response rates', flush=True)
        print(best.head(12).to_string(index=False, max_colwidth=45), flush=True)
    made = [] if args.no_figures else make_figures(df, data_on, data_off)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=H.HERE).stdout.strip()
    (OUT / 'provenance.json').write_text(json.dumps(dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='apl_effect_scores.py', host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=commit, statistic='kc_remapping_harness.apl_effect_stats (spike-count cosine; APL on minus acute APL off)',
        data='control flies (TNT3c) minus APL output blocked (TNT), cosine, mean over recordings (harness.yang_mean_rdm)',
        realism_bands=dict(rate_on=RATE_ON_BAND, rate_ratio_off_over_on=RATE_RATIO_BAND, rate_rho_min=RATE_RHO_MIN,
            odors_per_kc_tv_max=dict(on=opk_band('on', 'noise'), off=opk_band('off', 'noise')),
            data_units='baseline-noise units, above the registry\'s noise_threshold (realism_flag); Yang\'s z, above 1 z '
                '(realism_flag_zcrit)', noise_threshold=H.data_target('noise_threshold'),
            data_rate_ratio=dict(zip(('value', 'ci90'), data_rate_ratio('noise'))),
            zcrit_odors_per_kc_tv_max=dict(on=opk_band('on', 'z'), off=opk_band('off', 'z'))),
        figures=made, template=str(TEMPLATE.relative_to(H.REPO)), template_sha256=_sha(TEMPLATE)), indent=1, ensure_ascii=False) + '\n')
    print(f'-> {OUT} ({len(made)} figures)', flush=True)


if __name__ == '__main__':
    main()
