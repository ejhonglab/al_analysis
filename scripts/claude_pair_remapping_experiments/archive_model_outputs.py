#!/usr/bin/env python3
"""Copies an experiment's model outputs, with their parameters, to the shared archive.

The archive is <shared>/data/model_outputs/E<n>_<short-description>/ (same name as the
experiment's results/ and outputs/ folders), organized by experiment regardless of the host
that ran it. Per experiment it holds: every KC x odor matrix the run saved (parquet;
<variant>_<condition>_spike_counts.parquet = total spikes per KC per odor,
<variant>_<condition>_responses.parquet = at least 2 spikes), any other data files the run
wrote next to them (tuned_params.json, wPNKC matrices, sweep tables, run logs),
params.csv (the stats table's identifying and parameter columns: tag, variant, base, seed,
thresholds, APL / KC-KC scales, sweep parameters), provenance.json from results/, and a
README.md listing the variants. The top-level README.md gets one row per experiment.

    ./venv/bin/python archive_model_outputs.py E16_claw_saturation [E19_... ...]
    ./venv/bin/python archive_model_outputs.py --all          # every outputs/E* folder
    ./venv/bin/python archive_model_outputs.py --inputs       # ORN input + base wirings
    ./venv/bin/python archive_model_outputs.py --all --params-only   # rebuild every params.csv

File and folder names are ASCII only (CLAUDE.md); names with other characters are rejected.
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import kc_remapping_harness as H

ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
# identifying columns, every model parameter a script has put in its table (fit_mb_model kwargs,
# olfsysm parameters, input / wiring modifications), fit settings, and a few output rates. Add
# to this list whenever a script records a new parameter column; params.csv is what the
# archive README points at for regenerating a variant.
PARAM_COLS = ['tag', 'variant', 'base', 'config', 'kind', 'point', 'part', 'seed', 'target_rate', 'target',
    'fixed_thr', 'fixed_thr_mean', 'fixed_thr_is_vector', 'thr_factor', 'thr_scale_homeo_full',
    'thr_scale_homeo_data', 'stage0_fixed_thr', 'stage0_thr_factor', 'wAPLKC', 'wKCAPL',
    'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'kc_apl_factor', 'wKCKC_scale',
    'kc_kc_inh', 'kc_kc_tau', 'wkckc', 'claw_thr', 'claw_thr_width', 'claw_thr_subgain', 'claw_sat', 'form',
    'claw_pow', 'claw_pow_ref', 'bouton_ves_p', 'bouton_tau_r', 'alpha', 'gain_min', 'gain_max', 'gain', 'glomeruli',
    'gain_glomeruli', 'factor', 'scaled_odors', 'k', 'm', 'breadth', 'breadth_k', 'n_claws_active_to_spike',
    'coinput_factor', 'apl_unit_coupling', 'coupling', 'kc_set', 'kc_scale', 'bouton_scale', 'ext_inh_scale',
    'pn_inhadd', 'pn_inhsc', 'pn_apl_multiplier', 'multibouton_claws', 'claw_weight', 'wiring', 'wiring_seed',
    'apl_drive', 'pn_claw_to_apl', 'apl_inh_divisive', 'bouton_inh_divisive', 'base_params', 'breadth_def', 'apl_kc_factor',
    'arm', 'orn_input', 'multiplier_min', 'multiplier_max', 'multiplier_geomean', 'scale_min', 'scale_max', 'cal_readout_param',
    'init_params', 'aplpn_max', 'aplpn_fixed', 'aplkc_min',
    'init_from', 'init_thr', 'loss_variant', 'method', 'maxfev', 'kckc0', 'thr0_factor',
    'n_kcs', 'n_glomeruli', 'rate_on', 'rate_off', 'rate_acute_off', 'rate_homeo_full_off',
    'rate_homeo_data_off', 'error']
STATS_NAMES = ['stats_variants.csv', 'models.csv', 'wkckc_controls.csv', 'sweep.csv']
# archive folders whose params.csv (and file names / README) were rebuilt by hand, so that olfdb lists one
# row per saved model; re-archiving them would overwrite that with the stats tables' rows (and, for E6, copy
# back the old file names), so they are skipped unless --force-hand-built is passed. Each folder's
# params_migration_log.csv says what was done.
HAND_BUILT = {
    'E6_readout_window': 'files renamed <base>_<window>_<condition>_spike_counts.parquet, one row per base x window (2026-11-09)',
    'E36_spike_apl': 'rows spk_base_thr40 / spk_base_thr70 added from tuned_params.json (2026-11-09)',
    'E43_model_trial_noise': 'params.csv is the two models ref40 / base70; the noise statistics are in noise_statistics_rows.csv (2026-11-09)',
    'E49_low_conc_kc': 'params.csv written by hand: E49 has none of the tables params.csv is built from (2026-11-09)',
}


# condition names used in the output file names (longest first, so 'acute_off' wins over 'off')
CONDITIONS = ('homeo_full_off', 'homeo_data_off', 'acute_off', 'off', 'on')


def split_variant_condition(filename: str):
    """'<variant>_<condition>_spike_counts.parquet' -> (variant, condition); condition may
    contain underscores (acute_off), so it is matched against CONDITIONS."""
    stem = filename
    for suffix in ('_spike_counts.parquet', '_responses.parquet'):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    for c in CONDITIONS:
        if stem.endswith('_' + c):
            return stem[:-len(c) - 1], c
    return stem, ''


def ascii_only(name: str) -> None:
    if not name.isascii():
        raise ValueError(f'non-ASCII name: {name!r}')


def archive_experiment(name: str, host: str, params_only: bool = False, force_hand_built: bool = False) -> dict:
    """Copies outputs/<name>/ to the archive and (re)builds its params.csv and README.

    With `params_only`, only params.csv and the provenance files are refreshed (from the shared
    results/ tables and the tables already in the archive folder), so it works for experiments
    whose local outputs are on another host; the README row is left as it is.
    """
    src = H.HERE / 'outputs' / name
    res = H.results_dir(name) if (H.shared_dir() / 'results' / name).exists() else None
    ascii_only(name)
    if name in HAND_BUILT and not force_hand_built:
        print(f'!!! {name} SKIPPED: its params.csv was rebuilt by hand ({HAND_BUILT[name]}); re-archiving would overwrite it. '
            'Update it by hand (and its params_migration_log.csv), or pass --force-hand-built and redo the hand edits.', file=sys.stderr)
        return dict(name=name, host=host, n_files=0, n_variants=0, conditions='', skipped=True)
    dst = ARCHIVE / name
    if params_only:
        if not dst.exists():
            raise FileNotFoundError(f'{dst} (nothing archived yet; run without --params-only on the host that ran it)')
        src = dst
    elif not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for p in ([] if params_only else sorted(src.rglob('*'))):
        if not p.is_file() or p.suffix in ('.pyc', '.html'):
            continue   # .html: artifact pages live in reports/, not here
        rel = p.relative_to(src)
        ascii_only(rel.as_posix())
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != p.stat().st_size:
            shutil.copy2(p, target)
        n_files += 1
    # parameters: the stats tables' identifying / parameter columns
    frames = []
    # the fixed names plus any per-experiment summary / stats / fits table (atlas's
    # experiments write E<n>_summary.csv, window_stats.csv, fits.csv, ...)
    tables = []
    for d in ([res] if res else []) + [src]:
        for sname in STATS_NAMES:
            tables.append(d / sname)
        for pat in ('*_summary.csv', 'stats*.csv', 'window_stats.csv', 'fits.csv', 'variants.csv'):
            tables.extend(sorted(d.glob(pat)))
    seen = set()
    for sp in tables:
        if sp in seen:
            continue
        seen.add(sp)
        for _ in (0,):
            if sp.exists():
                df = pd.read_csv(sp)
                keep = [c for c in PARAM_COLS if c in df.columns]
                if keep:
                    d = df[keep].copy(); d.insert(0, 'source_table', sp.name); frames.append(d)
    params = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(params):
        params.to_csv(dst / 'params.csv', index=False)
    missing_params = not len(params) and any(dst.rglob('*_spike_counts.parquet'))
    if missing_params:
        # E49 (2026-11-08) was archived with spike counts and no params.csv, so olfdb listed none of its models
        print(f'!!! {name}: spike count files archived but NO params.csv built: none of the tables it is built from '
            f'({", ".join(STATS_NAMES)}, *_summary.csv, stats*.csv, window_stats.csv, fits.csv, variants.csv) exists in '
            f'results/{name}/ or outputs/{name}/. Write params.csv by hand (one row per saved model, tag = the file names\' '
            'variant) or save the run\'s table under one of those names, then re-run.', file=sys.stderr)
    if res:
        # provenance.json, plus the per-fit provenance_<tag>.json the joint-fit scripts write
        for pf in sorted(res.glob('provenance*.json')):
            shutil.copy2(pf, dst / pf.name)
    if params_only:
        return dict(name=name, host=host, n_files=len(list(dst.rglob('*'))), n_variants=params['tag'].nunique() if 'tag' in params.columns else len(params), conditions='', params_only=True,
            missing_params=missing_params)
    parsed = [split_variant_condition(p.name) for p in list(dst.rglob('*_spike_counts.parquet'))
        + list(dst.rglob('*_responses.parquet'))]
    variants = sorted({v for v, _ in parsed})
    conds = sorted({c for _, c in parsed if c})
    lines = [f'# {name}: model outputs', '',
        f'Copied {datetime.now():%Y-%m-%d %H:%M} from host {host} (`outputs/{name}/` of the al_analysis checkout). '
        f'{n_files} files; {len(variants)} variants; conditions: {", ".join(conds) if conds else "(see file names)"}.', '',
        'Files: `<variant>_<condition>_spike_counts.parquet` (KCs x odors, total spikes per KC per odor), '
        '`<variant>_<condition>_responses.parquet` (same shape, 1 = at least 2 spikes); rows indexed by hemibrain '
        'KC body ID (and type), columns the odor at its concentration. `params.csv`: identifying and parameter '
        'columns of the stats tables (tag = variant name used in the file names); `provenance.json`: script, '
        'commit hashes, base kwargs and sweep settings; run logs as written. Regenerate with the script named in '
        'provenance.json at the recorded commit, with the base kwargs there and the tuned parameters in params.csv.', '']
    if variants:
        lines += ['Variants:', ''] + [f'- `{v}`' for v in variants] + ['']
    (dst / 'README.md').write_text('\n'.join(lines))
    return dict(name=name, host=host, n_files=n_files, n_variants=len(variants), conditions=', '.join(conds), missing_params=missing_params)


def save_inputs() -> None:
    from E3_drive_dependent_gain import BOUTON, BASES
    from E4_substrate import CLAWS
    from al_analysis.mb_model import fit_mb_model
    dst = ARCHIVE / 'inputs'; dst.mkdir(parents=True, exist_ok=True)
    orn = H.apl16_orn_deltas()
    orn.to_csv(dst / 'orn_deltas_apl16.csv')
    bases = {'claw_base': dict(CLAWS), 'claw_dynamics_base': dict(BASES['claw-dynamics']), 'bouton_base': dict(BOUTON),
        'hemibrain_wd20_base': dict(BASES['hemibrain-wd20'])}
    meta = {}
    for bname, kws in bases.items():
        _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=100.0, wAPLKC=0.1, **kws, make_plots=False, silent=True)
        wPNKC.to_parquet(dst / f'wPNKC_{bname}.parquet')
        meta[bname] = dict(kwargs={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in kws.items()},
            shape=list(wPNKC.shape), index_levels=list(wPNKC.index.names), column_levels=list(wPNKC.columns.names))
    (dst / 'bases.json').write_text(json.dumps(meta, indent=1))
    (dst / 'README.md').write_text('# Inputs shared by all experiments\n\n'
        '- `orn_deltas_apl16.csv`: the ORN input matrix (glomeruli x the 16 APL16 odors; odor-evoked rate changes in Hz '
        'relative to the Hallem spontaneous rates olfsysm adds), as `kc_remapping_harness.apl16_orn_deltas()` returns it. '
        'paa, euc and -aPine are at substituted ORN concentrations (see the harness).\n'
        '- `wPNKC_<base>.parquet`: the PN->KC claw weight matrix of each standard base (rows: claws with KC id / type / '
        'coordinates; columns: glomeruli, or glomerulus / PN / bouton for the bouton base); `bases.json`: the fit_mb_model '
        'kwargs that build each. Experiments that modified the wiring saved their own matrices (substrate experiment E4; '
        'per-glomerulus gains of E20 are (median peak / peak)^alpha on the bouton base columns, peaks in its glomerulus_peaks.csv).\n')
    print('inputs saved:', sorted(p.name for p in dst.iterdir()))


def update_index(rows: list) -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    idx = ARCHIVE / 'README.md'
    head = ('# Model outputs of the KC-remapping experiments\n\n'
        'One folder per experiment, `E<n>_<short-description>` (the same name as its `results/` folder and its entry in '
        '`experiment_priorities.md`), regardless of the host that ran it. Each holds the KC x odor spike-count and response '
        'matrices of every model variant and condition, the variants\' parameters (`params.csv`, `tuned_params.json` where '
        'written), `provenance.json` (script, commit hashes, base kwargs, sweep settings), the run logs and a README listing '
        'the variants. `inputs/` holds what all experiments share: the ORN input matrix and the standard PN->KC wiring matrices. '
        'File and folder names are ASCII only. Add or refresh a folder with `archive_model_outputs.py` from the repo folder.\n\n'
        '| experiment | host | files | variants | conditions | copied |\n|---|---|---|---|---|---|\n')
    existing = {}
    if idx.exists():
        for line in idx.read_text().splitlines():
            if line.startswith('| E') or line.startswith('| inputs'):
                existing[line.split('|')[1].strip()] = line
    for r in rows:
        existing[r['name']] = f"| {r['name']} | {r['host']} | {r['n_files']} | {r['n_variants']} | {r['conditions']} | {datetime.now():%Y-%m-%d %H:%M} |"
    idx.write_text(head + '\n'.join(existing[k] for k in sorted(existing, key=lambda s: (len(s.split('_')[0]), s))) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('experiments', nargs='*')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--inputs', action='store_true')
    ap.add_argument('--params-only', action='store_true',
        help='only rebuild params.csv from the shared results tables (works for any host\'s experiments)')
    ap.add_argument('--force-hand-built', action='store_true',
        help=f'also re-archive the folders whose params.csv was rebuilt by hand ({", ".join(HAND_BUILT)}), overwriting it')
    args = ap.parse_args()
    host = socket.gethostname()
    if args.inputs:
        save_inputs()
    names = args.experiments
    if args.all:
        where = ARCHIVE if args.params_only else (H.HERE / 'outputs')
        names = sorted(p.name for p in where.glob('E*') if p.is_dir())
    rows = []
    for n in names:
        r = archive_experiment(n, host, params_only=args.params_only, force_hand_built=args.force_hand_built)
        if r.get('skipped'):
            continue
        rows.append(r)
        print(f"{n}: {r['n_files']} files, {r['n_variants']} variants -> {ARCHIVE / n}" + (' (params.csv only)' if args.params_only else ''))
    if rows and not args.params_only:
        update_index(rows)
    missing = [r['name'] for r in rows if r.get('missing_params')]
    if missing:
        sys.exit(f'!!! no params.csv for {", ".join(missing)} (see above); olfdb will list none of their models')


if __name__ == '__main__':
    main()
