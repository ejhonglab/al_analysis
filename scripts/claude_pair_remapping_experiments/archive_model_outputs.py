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

File and folder names are ASCII only (CLAUDE.md); names with other characters are rejected.
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
from datetime import datetime
from pathlib import Path

import pandas as pd

import kc_remapping_harness as H

ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
PARAM_COLS = ['tag', 'variant', 'base', 'config', 'seed', 'target_rate', 'fixed_thr', 'fixed_thr_mean',
    'fixed_thr_is_vector', 'thr_factor', 'thr_scale_homeo_full', 'thr_scale_homeo_data', 'wAPLKC', 'wKCAPL',
    'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'kc_apl_factor', 'wKCKC_scale',
    'kc_kc_inh', 'kc_kc_tau', 'wkckc', 'claw_thr', 'claw_sat', 'form', 'bouton_ves_p', 'bouton_tau_r',
    'alpha', 'gain_min', 'gain_max', 'k', 'breadth', 'breadth_k', 'n_claws_active_to_spike', 'coupling',
    'n_kcs', 'n_glomeruli', 'rate_on', 'rate_off', 'rate_acute_off', 'rate_homeo_full_off',
    'rate_homeo_data_off', 'error']
STATS_NAMES = ['stats_variants.csv', 'models.csv', 'wkckc_controls.csv', 'sweep.csv']


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


def archive_experiment(name: str, host: str) -> dict:
    src = H.HERE / 'outputs' / name
    res = H.results_dir(name) if (H.shared_dir() / 'results' / name).exists() else None
    if not src.exists():
        raise FileNotFoundError(src)
    ascii_only(name)
    dst = ARCHIVE / name
    dst.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for p in sorted(src.rglob('*')):
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
    if res and (res / 'provenance.json').exists():
        shutil.copy2(res / 'provenance.json', dst / 'provenance.json')
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
    return dict(name=name, host=host, n_files=n_files, n_variants=len(variants), conditions=', '.join(conds))


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
    args = ap.parse_args()
    host = socket.gethostname()
    if args.inputs:
        save_inputs()
    names = args.experiments
    if args.all:
        names = sorted(p.name for p in (H.HERE / 'outputs').glob('E*') if p.is_dir())
    rows = []
    for n in names:
        r = archive_experiment(n, host); rows.append(r)
        print(f"{n}: {r['n_files']} files, {r['n_variants']} variants -> {ARCHIVE / n}")
    if rows:
        update_index(rows)


if __name__ == '__main__':
    main()
