#!/usr/bin/env python3
"""E45: the glomerulus gain at the data's operating point; comparison table and figure.

The fits themselves are `E10_joint_fit.py` runs (structure loss, breadth exponent 2, the
APL->bouton scale capped at 3x, 150 evaluations) with the loss's APL-on response-rate target
at the control flies' fraction (`--rate-target ctrl_frac`, 6.8 %; E44) instead of 5 %, at the
70 and 40 Hz per-claw thresholds, written to results/E45_gain_at_ctrl_frac/fits.csv:
  (1) no gain                      thr<T>_rate0.068_k2_cap3
  (2) the eight glomeruli at 1.25  thr<T>_gain1.25_rate0.068_k2_cap3
  (3) the Hallem-ms core at 1.5    thr<T>_input-hallem_shared_ms_gain1.5_rate0.068_k2_core3_cap3
  (4) the Hallem-ms core at 1.25   thr<T>_input-hallem_shared_ms_gain1.25_rate0.068_k2_core3_cap3
(the core is DC2 + DM5 + DM1 with E38's hallem_shared_ms ORN input; the exact command lines
are in the experiment's entry in experiment_priorities.md). The eight glomeruli at 1.5 at
6.8 % are E44's part-B refits (results/E44_operating_point/fits.csv), not re-run.

This script puts those rows next to the same models fitted at 5 %: E34's capped eight-glomerulus
fits at 1.25 and 1.5, E40's Hallem-ms core at 1.5, and for no gain E31's capped fit (70 Hz
only) and E25's uncapped fits (both thresholds; the table uses E25 only where no capped fit
exists). No 5 % fit of the Hallem-ms core at 1.25 exists. Fits not in their fits.csv yet are
reported as missing and left out, so the script can run before every E45 fit has landed.

Each fit's extended loss is recomputed from its saved outputs (./outputs/<experiment>/ if
present, else the model-output archive in the shared directory) with the current data targets,
at the 5 % and at the control flies' rate target, as E44 did: older fits were scored with
hand-typed targets (unresolved_issues.md), so their recorded losses differ slightly. Where no
outputs are found, the recorded loss is used (exact for fits made since 2026-11-07) and the
response rates of the weak-input odors are left empty. The loss without its rate term is the
same at both targets and compares fits made at different operating points on everything else.
The weak odors are E26's six (ms, paa, euc, -aPine, B-myr, 1-8ol), as in the data-target
registry; their share is their mean response rate over the mean of all 16 odors'. Results
involving paa, euc or -aPine depend on ORN input at a substituted concentration (within
1 log10 unit; see H.APPROX_CONC_NOTE).

Outputs (in results/E45_gain_at_ctrl_frac/, or --out): comparison.csv (one row per fit found),
comparison_table.md (one table per threshold, with the KC data from the data-target registry),
E45_gain_at_ctrl_frac.png, provenance.json (al_analysis commit at the start, the input files'
paths and sha256, the fits found and missing).

Usage: ./venv/bin/python E45_gain_at_ctrl_frac.py [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E10_joint_fit import LOSS_WEIGHTS, RATE_SCALE, RATE_TARGET, make_loss
from E26_claw_power import WEAK

EXPERIMENT = 'E45_gain_at_ctrl_frac'
CTRL_FRAC = H.data_target('ctrl_frac')
# E10_joint_fit.py's tag prefix for --rate-target ctrl_frac
RATE_TAG = f'rate{CTRL_FRAC:.3f}_'
HALLEM = 'input-hallem_shared_ms_'
THRS = (70.0, 40.0)
V = H.PRIMARY_MODEL_RDM
CORE3 = {'DC2', 'DM5', 'DM1'}
# the oracle set of the glomerulus-gain experiment (E27), the reference model's eight glomeruli
ORACLE = {'DC2', 'DL1', 'DM1', 'DM5', 'D', 'DP1m', 'VA7m', 'VC1'}

MODELS = {
    'none': 'no gain',
    'eight1.25': 'eight glomeruli<br>gain 1.25',
    'eight1.5': 'eight glomeruli<br>gain 1.5',
    'core1.25': 'Hallem-ms core<br>gain 1.25',
    'core1.5': 'Hallem-ms core<br>gain 1.5',
}
# (gain, gain set, Hallem-ms input) each model's fits must have
MODEL_SETTINGS = {'none': (None, None, False), 'eight1.25': (1.25, ORACLE, False),
    'eight1.5': (1.5, ORACLE, False), 'core1.25': (1.25, CORE3, True), 'core1.5': (1.5, CORE3, True)}
OPS = {'5': '5 %', 'ctrl': f'{100 * CTRL_FRAC:.1f} %'}
# the figure's tick labels (the table's are too wide)
FIG_LABELS = {'none': 'no gain', 'eight1.25': 'eight\ngain 1.25', 'eight1.5': 'eight\ngain 1.5',
    'core1.25': 'core\ngain 1.25', 'core1.5': 'core\ngain 1.5'}


class Arm(NamedTuple):
    model: str
    op: str             # '5' (the 5 % convention) or 'ctrl' (the control flies' fraction)
    experiment: str
    tag: str            # format string, {T} = per-claw threshold
    thrs: Tuple[float, ...] = THRS
    capped: bool = True
    note: str = ''


# in table preference order within each (model, operating point)
ARMS = [
    Arm('none', '5', 'E31_aplpn_bound', 'thr{T}_k2_aplpnmax3', thrs=(70.0,)),
    Arm('none', '5', 'E25_joint_fit_structure', 'thr{T}_k2_ext', capped=False, note='uncapped'),
    Arm('none', 'ctrl', EXPERIMENT, 'thr{T}_' + RATE_TAG + 'k2_cap3'),
    Arm('eight1.25', '5', 'E34_gain_bounded', 'thr{T}_gain1.25_k2_cap3'),
    Arm('eight1.25', 'ctrl', EXPERIMENT, 'thr{T}_gain1.25_' + RATE_TAG + 'k2_cap3'),
    Arm('eight1.5', '5', 'E34_gain_bounded', 'thr{T}_gain1.5_k2_cap3'),
    Arm('eight1.5', 'ctrl', 'E44_operating_point', 'thr{T}_gain1.5_' + RATE_TAG + 'k2_cap3'),
    Arm('core1.25', 'ctrl', EXPERIMENT, 'thr{T}_' + HALLEM + 'gain1.25_' + RATE_TAG + 'k2_core3_cap3'),
    Arm('core1.5', '5', 'E40_reduced_gain_set', 'thr{T}_' + HALLEM + 'gain1.5_k2_core3_cap3'),
    Arm('core1.5', 'ctrl', EXPERIMENT, 'thr{T}_' + HALLEM + 'gain1.5_' + RATE_TAG + 'k2_core3_cap3'),
]

# comparison.csv column -> fits.csv column
FITS_COLS = {
    'rate_on': 'rate_on', 'rate_off': 'rate_off', 'rate_ratio': 'rate_ratio_off_over_on',
    'rho_ctrl': 'fit_on_spearman', 'rho_tnt': 'fit_off_spearman', 'block': 'fit_block',
    'iso_r2': 'fit_iso_r2', 'rate_rho_on': 'fit_po_on_spearman', 'rate_rho_off': 'fit_po_off_spearman',
    'struct_wt_on': 'fit_wt_dev_on', 'vcf_on': f'{V}__on_spearman_resid_vs_vcf_hyp', 'vcf_off': 'fit_vcf_off',
    'mean_dist_on': f'{V}__on_mean_model', 'mean_dist_off': f'{V}__off_mean_model',
    'loss_recorded': 'loss', 'initial_loss': 'initial_loss', 'rate_target_recorded': 'rate_target',
    'fixed_thr': 'fixed_thr', 'thr_factor': 'thr_factor', 'wAPLKC_scale': 'wAPLKC_scale',
    'wKCAPL_scale': 'wKCAPL_scale', 'wAPLPN_scale': 'wAPLPN_scale', 'wPNAPL_scale': 'wPNAPL_scale',
    'apl_factor_vs_e3': 'apl_factor_vs_e3', 'aplpn_factor_vs_e3': 'aplpn_factor_vs_e3',
    'kc_kc_inh': 'kc_kc_inh', 'k': 'k', 'gain': 'gain', 'gain_glomeruli': 'gain_glomeruli',
    'orn_input': 'orn_input', 'aplpn_max': 'aplpn_max', 'init_from': 'init_from', 'n_evals': 'n_evals',
    'seconds': 'seconds',
}
# the loss's own statistics, checked against the fits.csv row when recomputed from the outputs
CHECK = {'rho_ctrl': 'on_spearman', 'rho_tnt': 'off_spearman', 'block': 'block', 'iso_r2': 'iso_r2',
    'rate_rho_on': 'po_on_spearman', 'rate_rho_off': 'po_off_spearman', 'struct_wt_on': 'wt_dev_on',
    'vcf_off': 'vcf_off', 'rate_on': 'rate_on', 'rate_off': 'rate_off'}

# (column, label, format ('pct' = percent), data key)
TABLE_ROWS = [
    ('rate_on', 'response rate, APL on', 'pct', 'ctrl_frac'),
    ('rate_off', 'response rate, APL off', 'pct', 'tnt_frac'),
    ('rate_ratio', 'APL-off / APL-on response rate', '.2f', 'rate_ratio_fraction'),
    ('loss_own', 'extended loss at its own rate target', '.2f', None),
    ('loss_no_rate_term', 'extended loss without the rate term', '.2f', None),
    ('block', 'fruity × small-ester block', '+.3f', 'block'),
    ('rho_ctrl', 'ρ on odor-odor distances, APL on', '.2f', 'ctrl_ceiling'),
    ('rho_tnt', 'ρ on odor-odor distances, APL off', '.2f', 'tnt_ceiling'),
    ('iso_r2', 'isotonic R² of APL on vs APL off', '.2f', 'iso_r2'),
    ('rate_rho_on', 'response rate ρ, APL on', '.2f', 'ctrl_rate_rho_ceiling'),
    ('rate_rho_off', 'response rate ρ, APL off', '.2f', 'tnt_rate_rho_ceiling'),
    ('struct_wt_on', 'structure vs wild-type somata, APL on', '.2f', 'ctrl_struct_wt'),
    ('vcf_on', 'natural-source alignment, APL on', '.2f', 'ctrl_vcf'),
    ('vcf_off', 'natural-source alignment, APL off', '.2f', 'tnt_vcf'),
    ('mean_dist_on', 'mean cosine distance, APL on', '.2f', 'ctrl_mean_dist'),
    ('mean_dist_off', 'mean cosine distance, APL off', '.2f', 'tnt_mean_dist'),
    ('weak_frac_on', "weak odors' response rate, APL on", '.3f', 'ctrl_weak_frac'),
    ('weak_share_on', "weak odors' APL-on response rate / mean APL-on response rate", '.2f', 'ctrl_weak_share'),
    ('weak_frac_off', "weak odors' response rate, APL off", '.3f', 'tnt_weak_frac'),
    ('weak_share_off', "weak odors' APL-off response rate / mean APL-off response rate", '.2f', 'tnt_weak_share'),
    ('off_frac_ms', 'ms response rate, APL off', '.3f', 'tnt_frac_ms'),
    ('off_frac_paa', 'paa response rate, APL off', '.3f', 'tnt_frac_paa'),
    ('fixed_thr', 'somatic threshold (Hz)', '.0f', None),
    ('thr_factor', 'somatic threshold (× its base)', '.2f', None),
    ('apl_factor_vs_e3', 'APL↔KC scale (× its base)', '.2f', None),
    ('aplpn_factor_vs_e3', 'APL→bouton scale (× its base)', '.2f', None),
    ('kc_kc_inh', 'KC–KC scale', '.1f', None),
    ('n_evals', 'evaluations', '.0f', None),
]
DATA_KEYS = ('ctrl_frac', 'tnt_frac', 'rate_ratio_fraction', 'block', 'ctrl_ceiling', 'tnt_ceiling', 'iso_r2',
    'ctrl_rate_rho_ceiling', 'tnt_rate_rho_ceiling', 'ctrl_struct_wt', 'ctrl_vcf', 'tnt_vcf', 'ctrl_mean_dist',
    'tnt_mean_dist', 'ctrl_mean_dist_noise_corrected', 'tnt_mean_dist_noise_corrected', 'ctrl_weak_frac',
    'ctrl_weak_share', 'tnt_weak_frac', 'tnt_weak_share')


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    """al_analysis commit and whether its tracked files differ from it."""
    run = lambda *a: subprocess.run(['git', *a], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    dirty = run('status', '--porcelain', '--untracked-files=no')
    return dict(al_analysis_commit=run('rev-parse', 'HEAD'), al_analysis_dirty=bool(dirty),
        al_analysis_dirty_files=dirty.splitlines(), script_sha256=sha256(Path(__file__)))


def outputs_paths(exp: str, tag: str) -> Tuple[Optional[Dict[str, Path]], str]:
    """The fit's four saved outputs ({on,acute_off}_{responses,spike_counts}), from ./outputs/
    if all are there, else from the model-output archive; (None, 'none') if neither has them."""
    names = {f'{c}_{x}': f'{tag}_{c}_{x}.parquet' for c in ('on', 'acute_off') for x in ('responses', 'spike_counts')}
    for where, d in (('local', H.HERE / 'outputs' / exp), ('archive', H.shared_dir() / 'data' / 'model_outputs' / exp)):
        paths = {k: d / n for k, n in names.items()}
        if all(p.exists() for p in paths.values()):
            return paths, where
    return None, 'none'


def rate_term(rate_on: float, target: float) -> float:
    """The structure loss's APL-on response-rate term (E10_joint_fit.make_loss)."""
    return float(LOSS_WEIGHTS['rate_on'] * ((rate_on - target) / RATE_SCALE) ** 2)


def gain_set(g) -> Optional[set]:
    if not isinstance(g, str) or not g.strip():
        return None
    return set(g.replace('[', '').replace(']', '').replace("'", '').replace(',', ' ').split())


def check_settings(arm: Arm, thr: float, r: pd.Series) -> List[str]:
    """Differences between a fits.csv row and the settings its arm is meant to have."""
    out = []
    gain, gset, hallem = MODEL_SETTINGS[arm.model]
    if float(r.get('k', np.nan)) != 2.0:
        out.append(f"k {r.get('k')}")
    if float(r.get('claw_thr', np.nan)) != thr:
        out.append(f"claw_thr {r.get('claw_thr')}")
    cap = r.get('aplpn_max', np.nan)
    if arm.capped != (pd.notna(cap) and float(cap) == 3.0):
        out.append(f'aplpn_max {cap}')
    rg = r.get('gain', np.nan)
    if (gain is None) != pd.isna(rg) or (gain is not None and abs(float(rg) - gain) > 1e-9):
        out.append(f'gain {rg}')
    if gain is not None:
        gs = gain_set(r.get('gain_glomeruli')) or ORACLE
        if gs != gset:
            out.append(f"gain_glomeruli {r.get('gain_glomeruli')}")
    if hallem != ('hallem_shared_ms' in str(r.get('orn_input', ''))):
        out.append(f"orn_input {r.get('orn_input')}")
    rt = r.get('rate_target', np.nan)
    want = CTRL_FRAC if arm.op == 'ctrl' else RATE_TARGET
    if not (pd.isna(rt) and arm.op == '5') and not (pd.notna(rt) and abs(float(rt) - want) < 1e-6):
        out.append(f'rate_target {rt}')
    return out


def load_rows(loss5, loss_ctrl, inputs: Dict[str, str]) -> Tuple[pd.DataFrame, List[dict]]:
    """One row per fit found (statistics from fits.csv, losses and the weak odors' response
    rates from its saved outputs); and the arms' fits not found."""
    rows, missing, fits_cache = [], [], {}
    for arm in ARMS:
        f = H.shared_dir() / 'results' / arm.experiment / 'fits.csv'
        if arm.experiment not in fits_cache:
            fits_cache[arm.experiment] = pd.read_csv(f) if f.exists() else None
            if f.exists():
                inputs[str(f)] = sha256(f)
        fits = fits_cache[arm.experiment]
        for thr in arm.thrs:
            tag = arm.tag.format(T=f'{thr:g}')
            sel = fits[fits.tag == tag] if fits is not None else []
            if len(sel) != 1:
                missing.append(dict(model=arm.model, operating_point=OPS[arm.op], claw_thr=thr,
                    experiment=arm.experiment, tag=tag,
                    reason='no fits.csv' if fits is None else ('not in fits.csv' if not len(sel) else f'{len(sel)} rows')))
                continue
            r = sel.iloc[0]
            row = dict(model=arm.model, model_label=MODELS[arm.model].replace('<br>', ', '),
                operating_point=OPS[arm.op], op=arm.op, rate_target=(CTRL_FRAC if arm.op == 'ctrl' else RATE_TARGET),
                experiment=arm.experiment, tag=tag, claw_thr=thr, note=arm.note,
                **{k: r.get(c, np.nan) for k, c in FITS_COLS.items()})
            warnings = check_settings(arm, thr, r)
            paths, where = outputs_paths(arm.experiment, tag)
            row['outputs'] = where
            if paths is not None:
                for p in paths.values():
                    inputs[str(p)] = sha256(p)
                out = {c: (pd.read_parquet(paths[f'{c}_responses']), pd.read_parquet(paths[f'{c}_spike_counts']))
                    for c in ('on', 'acute_off')}
                outs = {'on': out['on'], 'off': out['acute_off']}
                l5, lc = loss5(outs), loss_ctrl(outs)
                for col, key in CHECK.items():
                    if pd.notna(row[col]) and abs(float(row[col]) - l5[key]) > 1e-6:
                        warnings.append(f'{col} {row[col]:.4f} in fits.csv, {l5[key]:.4f} from the outputs')
                row.update(loss_rate5=l5['loss'], loss_rate_ctrl=lc['loss'],
                    loss_no_rate_term=l5['loss'] - l5['term_rate_on'], loss_source='recomputed from the outputs',
                    **{f'loss_{k}': v for k, v in l5.items() if k.startswith('term_') and k != 'term_rate_on'})
                fr = {c: H.response_fractions(out[c][0]).reindex(H.APL16_ODORS) for c in ('on', 'acute_off')}
                for c, name in (('on', 'on'), ('acute_off', 'off')):
                    row[f'weak_frac_{name}'] = float(fr[c][WEAK].mean())
                    row[f'weak_share_{name}'] = float(fr[c][WEAK].mean() / fr[c].mean())
                    row.update({f'{name}_frac_{o}': float(fr[c][o]) for o in WEAK})
            else:
                nr = float(row['loss_recorded']) - float(r.get('fit_term_rate_on', np.nan))
                row.update(loss_rate5=nr + rate_term(row['rate_on'], RATE_TARGET),
                    loss_rate_ctrl=nr + rate_term(row['rate_on'], CTRL_FRAC), loss_no_rate_term=nr,
                    loss_source='recorded in fits.csv')
            row['loss_own'] = row['loss_rate_ctrl'] if arm.op == 'ctrl' else row['loss_rate5']
            row['warnings'] = '; '.join(warnings)
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df['model_order'] = df.model.map({m: i for i, m in enumerate(MODELS)})
        df = df.sort_values(['claw_thr', 'model_order', 'op'], ascending=[False, True, True], kind='stable').drop(
            columns='model_order')
    return df, missing


def data_values() -> Dict[str, float]:
    d = {}
    for key in DATA_KEYS:
        try:
            d[key] = H.data_target(key)
        except KeyError as e:   # a missing registry key only loses a reference value
            print('no data target', key, repr(e))
    tnt = H.yang_response_fractions('TNT')[H.APL16_ODORS].mean()
    for o in WEAK:
        d[f'tnt_frac_{o}'] = float(tnt[o])
    return d


def table_columns(df: pd.DataFrame, thr: float) -> List[Tuple[str, str, pd.Series]]:
    """(model, subheader, row) per table column: per model and operating point, the first arm
    in ARMS with a fit at this threshold."""
    cols = []
    for m in MODELS:
        for op in OPS:
            for arm in ARMS:
                if arm.model != m or arm.op != op:
                    continue
                r = df[(df.claw_thr == thr) & (df.experiment == arm.experiment) & (df.tag == arm.tag.format(T=f'{thr:g}'))]
                if len(r):
                    exp_id = arm.experiment.split('_')[0]
                    note = f'<br>{arm.note}' if arm.note else ''
                    cols.append((m, f'fit at {OPS[op]}{note}<br>({exp_id})', r.iloc[0]))
                    break
    return cols


def fmt(v, spec: str) -> str:
    if v is None or pd.isna(v):
        return '-'
    if spec == 'pct':
        return f'{100 * float(v):.1f} %'
    return format(float(v), spec)


def data_cell(data: Dict[str, float], key: Optional[str], spec: str) -> str:
    if not key or key not in data:
        return '-'
    s = fmt(data[key], spec)
    if 'ceiling' in key:
        return f'ceiling {s}'
    nc = f'{key}_noise_corrected'
    if nc in data:
        return f'{s} (noise-corrected {fmt(data[nc], spec)})'
    return s


def markdown_tables(df: pd.DataFrame, data: Dict[str, float], missing: List[dict]) -> str:
    out = []
    for thr in THRS:
        cols = table_columns(df, thr) if len(df) else []
        if not cols:
            continue
        groups = []
        for m, _, _ in cols:
            if groups and groups[-1][0] == m:
                groups[-1][1] += 1
            else:
                groups.append([m, 1])
        head1 = '| | | ' + ' | '.join(MODELS[m] + (f' {{span={n}}}' if n > 1 else '') for m, n in groups) + ' |'
        head2 = '| statistic | KC data | ' + ' | '.join(h for _, h, _ in cols) + ' |'
        rows = [f'**{thr:g} Hz per-claw threshold**', '', head1, head2, '|---|---|' + '---|' * len(cols)]
        for col, label, spec, dkey in TABLE_ROWS:
            rows.append(f'| {label} | {data_cell(data, dkey, spec)} | '
                + ' | '.join(fmt(r.get(col, np.nan), spec) for _, _, r in cols) + ' |')
        out.append('\n'.join(rows))
    notes = ['Losses are recomputed from each fit\'s saved outputs with the current data targets (fits made '
        'before 2026-11-07 recorded slightly different losses); "without the rate term" is the same at both rate '
        'targets. The weak odors are ms, paa, euc, -aPine, B-myr and 1-8ol (paa, euc and -aPine with ORN input at a '
        'substituted concentration). Scales "× its base" are relative to the base each fit started from (the gain '
        'and the ORN input change the base).']
    no_out = df[df.outputs == 'none'] if len(df) else df
    if len(no_out):
        notes.append('No saved outputs found (recorded loss used, per-odor response rates left empty): '
            + ', '.join(f'{r.experiment} {r.tag}' for r in no_out.itertuples()) + '.')
    warned = df[df.warnings.astype(bool)] if len(df) else df
    for r in warned.itertuples():
        notes.append(f'Check {r.experiment} {r.tag}: {r.warnings}.')
    if missing:
        notes.append('Not in fits.csv yet (left out): ' + '; '.join(
            f"{MODELS[m['model']].replace('<br>', ', ')} at {m['operating_point']}, {m['claw_thr']:g} Hz "
            f"({m['experiment']} {m['tag']})" for m in missing) + '.')
    return '\n\n'.join(out + notes) + '\n'


PANELS = [
    ('loss_no_rate_term', 'extended loss without the rate term\n(lower is better)', None),
    ('block', 'fruity × small-ester block', 'block'),
    ('rho_ctrl', 'ρ on odor-odor distances,\nAPL on vs control flies', 'ctrl_ceiling'),
    ('rho_tnt', 'ρ on odor-odor distances,\nAPL off vs APL output blocked (TNT)', 'tnt_ceiling'),
    ('iso_r2', 'isotonic R² of APL on vs APL off', 'iso_r2'),
    ('rate_rho_on', 'response rate ρ, APL on\nvs control flies', 'ctrl_rate_rho_ceiling'),
    ('rate_rho_off', 'response rate ρ, APL off\nvs APL output blocked (TNT)', 'tnt_rate_rho_ceiling'),
    ('struct_wt_on', 'structure beyond the ORN input\nvs wild-type somata, APL on', 'ctrl_struct_wt'),
    ('vcf_on', 'natural-source alignment, APL on', 'ctrl_vcf'),
    ('vcf_off', 'natural-source alignment, APL off', 'tnt_vcf'),
    ('rate_on', 'response rate, APL on', 'ctrl_frac'),
    ('rate_ratio', 'APL-off / APL-on response rate', 'rate_ratio_fraction'),
    ('weak_frac_off', "weak odors' response rate, APL off", 'tnt_weak_frac'),
    ('weak_share_off', "weak odors' APL-off response rate\n/ mean APL-off response rate", 'tnt_weak_share'),
    ('off_frac_ms', 'ms response rate, APL off', 'tnt_frac_ms'),
]


def make_figure(df: pd.DataFrame, data: Dict[str, float], path: Path) -> None:
    """Per statistic: each model's fit at 5 % (open) and at the control flies' fraction (filled),
    joined, at 70 Hz (blue circles) and 40 Hz (red squares); KC data dotted."""
    plt = H._plt()
    from matplotlib.lines import Line2D
    colors = {70.0: '#2a6fdb', 40.0: '#d1495b'}
    markers = {70.0: 'o', 40.0: 's'}
    thr_dx = {70.0: -0.17, 40.0: 0.17}
    op_dx = {'5': -0.06, 'ctrl': 0.06}
    models = list(MODELS)
    cols = {thr: table_columns(df, thr) if len(df) else [] for thr in THRS}
    fig, axs = plt.subplots(3, 5, figsize=(21, 12), constrained_layout=True)
    for ax, (col, title, dkey) in zip(axs.flat, PANELS):
        for thr in THRS:
            for m in models:
                pts = [(r['op'], r.get(col, np.nan)) for mm, _, r in cols[thr] if mm == m]
                pts = [(op, float(v)) for op, v in pts if pd.notna(v)]
                if not pts:
                    continue
                x0 = models.index(m) + thr_dx[thr]
                xs = [x0 + op_dx[op] for op, _ in pts]
                ys = [v for _, v in pts]
                if len(pts) == 2:
                    ax.plot(xs, ys, color=colors[thr], lw=1, zorder=2)
                for (op, v), x in zip(pts, xs):
                    ax.plot([x], [v], markers[thr], ms=8, mew=1.6, color=colors[thr],
                        mfc=colors[thr] if op == 'ctrl' else 'white', zorder=3)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='0.3', ls=':', lw=1.3, zorder=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([FIG_LABELS[m] for m in models], fontsize=8.5)
        ax.set_xlim(-0.5, len(models) - 0.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.25, lw=0.5)
    handles = [Line2D([], [], marker=markers[t], ls='', ms=8, mew=1.6, color=colors[t], mfc=mfc,
            label=f'{t:g} Hz per-claw threshold, fit at {OPS[op]}')
        for t in THRS for op, mfc in (('5', 'white'), ('ctrl', colors[t]))]
    handles.append(Line2D([], [], color='0.3', ls=':', lw=1.3, label='KC data (ρ panels: the measurement ceiling)'))
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("The glomerulus gain re-fitted at the control flies' response rate "
        f"({OPS['ctrl']}; filled) against the same models fitted at 5 % (open)\n"
        'eight: the eight oracle glomeruli; core: DC2 + DM5 + DM1 with Hallem\'s ms input; capped structure fits, '
        'breadth exponent 2 (no gain at 5 %, 40 Hz: uncapped)', fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=None,
        help=f'output folder (default: <shared>/results/{EXPERIMENT}/)')
    args = ap.parse_args()
    # recorded at the start, so a pull during the run cannot change the commit it names
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script=Path(__file__).name,
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), **git_state(),
        ctrl_frac=CTRL_FRAC, rate_tag=RATE_TAG, weak_odors=WEAK, approx_conc_note=H.APPROX_CONC_NOTE,
        arms=[a._asdict() for a in ARMS])
    res_dir = args.out if args.out is not None else H.results_dir(EXPERIMENT)
    res_dir.mkdir(parents=True, exist_ok=True)

    inputs: Dict[str, str] = {}
    yang = H.yang_data_dir() / 'computed'
    for p in (H.data_targets_path(), yang / 'rdm_per_recording_long.csv', yang / 'response_fraction_per_recording.csv',
            H.shared_dir() / 'data' / 'model_outputs' / 'inputs' / 'orn16_corrdist_olfdb_36glom.csv',
            H.CONSENSUS_CSV, H.APL_PAIR_DIR / 'shared12_pairs.csv'):
        inputs[str(p)] = sha256(p)
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    loss_args = (data, data_rates, H.orn16_consensus_pairs(), H.consensus_orn_drive(), H.reference_pairs())
    loss5 = make_loss(*loss_args, structure=True, rate_target=RATE_TARGET)
    loss_ctrl = make_loss(*loss_args, structure=True, rate_target=CTRL_FRAC)

    df, missing = load_rows(loss5, loss_ctrl, inputs)
    dvals = data_values()
    df.to_csv(res_dir / 'comparison.csv', index=False)
    (res_dir / 'comparison_table.md').write_text(markdown_tables(df, dvals, missing))
    make_figure(df, dvals, res_dir / f'{EXPERIMENT}.png')
    prov.update(found=df[['experiment', 'tag', 'outputs', 'loss_source', 'warnings']].to_dict('records') if len(df) else [],
        missing=missing, data_values=dvals, inputs=inputs)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')

    pd.set_option('display.width', 250)
    show = ['experiment', 'model', 'operating_point', 'claw_thr', 'loss_own', 'loss_no_rate_term', 'block', 'rho_ctrl',
        'rho_tnt', 'rate_rho_on', 'rate_rho_off', 'struct_wt_on', 'vcf_off', 'rate_ratio', 'weak_frac_off', 'weak_share_off']
    if len(df):
        print(df[show].round(3).to_string(index=False))
        for r in df[df.warnings.astype(bool)].itertuples():
            print(f'warning, {r.experiment} {r.tag}: {r.warnings}')
        for r in df[df.outputs == 'none'].itertuples():
            print(f'no saved outputs, {r.experiment} {r.tag}: recorded loss, no per-odor response rates')
    print('data:', {k: round(v, 3) for k, v in dvals.items()})
    for m in missing:
        print(f"missing: {m['model']} at {m['operating_point']}, {m['claw_thr']:g} Hz ({m['experiment']} {m['tag']}: {m['reason']})")
    print('->', res_dir)


if __name__ == '__main__':
    main()
