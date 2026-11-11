#!/usr/bin/env python3
"""Yang's APL16 KC data in units of each KC's baseline noise (the realism bands' data side since Tom's decision of
2026-11-10 16:10).

Reads every trial's response as olfdb has it (<shared>/data/yang_apl16_kc_olfdb/<genotype>/dataset_<id>_long.csv:
Yang's whole-recording z, a trial's response being the mean z 2-8 s after odor onset minus the mean over the 5 s
before) and viki's per-KC scales (<shared>/notes/yang_response_normalization/kc_scales.csv, written by
yang_response_normalization.py from Yang's suite2p output: `sd_full` the SD of the KC's Fc = F - 0.7 Fneu over the
whole recording, Yang's z unit; `sd_base` its SD within the 51 pre-odor windows, each window's mean removed), and
writes each trial's response in baseline-noise units, value_z * sd_full / sd_base (the trial's change in Fc over
sd_base), to <shared>/data/yang_apl16_kc_noise_units/ (README.md there describes each output), including one file per
genotype in olfdb's long upload layout, checked with olfdb's own loader when /opt/olfdb is on this host.

The script stops if a check fails: every KC of every recording has a scale; dividing by the scale gives olfdb's value
back; the control flies' and the APL-blocked flies' mean response rates over the 16 odors above the registry's
noise_threshold are the registry's noise_ctrl_frac and noise_tnt_frac.

Usage:
    ./venv/bin/python yang_apl16_kc_noise_units.py [--out DIR] [--olfdb-src /opt/olfdb]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H

HERE = Path(__file__).resolve().parent
SHARED = H.shared_dir()
DATASETS = {19: 'TNT', 20: 'TNT3c'}
LABEL = {'TNT3c': 'control flies', 'TNT': 'APL output blocked (TNT)'}
SOLVENT = 'pfo'
PANEL = 'APL16'
KEYS = ['genotype', 'date', 'fly_num', 'hemibrain', 'roi']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def kc_scales(path: Path) -> pd.DataFrame:
    """viki's kc_scales.csv keyed as olfdb's ROIs (genotype, date, fly_num, hemibrain, roi): `recording` is
    `<YYYYMMDD>_<fly><F><L|R>fun<n>` and `kc` the row order of the iscell ROIs, which is olfdb's `roi` within a
    recording (data_targets.kc_scale_ratios reads it the same way)."""
    ks = pd.read_csv(path)
    day, unit = (ks.recording.str.split('_', expand=True)[i] for i in (0, 1))
    out = pd.DataFrame({'genotype': ks.genotype, 'date': day.str[:4] + '-' + day.str[4:6] + '-' + day.str[6:8],
        'fly_num': unit.str[0].astype(int), 'hemibrain': unit.str[2], 'roi': ks.kc.astype(int), 'recording': ks.recording,
        'sd_full': ks.sd_full, 'sd_base': ks.sd_base, 'scale_ratio': ks.scale_ratio})
    assert not out.duplicated(KEYS).any()
    return out


def per_trial(scales: pd.DataFrame) -> pd.DataFrame:
    """Every trial of both datasets, as olfdb has them, with each KC's scale: value (baseline-noise units), value_z
    (olfdb's), dfc (the change in Fc, raw fluorescence units: value_z * sd_full)."""
    parts = []
    for ds, g in DATASETS.items():
        d = pd.read_csv(SHARED / 'data' / 'yang_apl16_kc_olfdb' / g / f'dataset_{ds}_long.csv')
        d.insert(0, 'genotype', g)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).rename(columns={'value': 'value_z'})
    assert set(d.panel) == {PANEL}, set(d.panel)
    d = d.merge(scales[KEYS + ['sd_full', 'scale_ratio']], on=KEYS, how='left', validate='many_to_one')
    missing = d[d.scale_ratio.isna()][KEYS].drop_duplicates()
    if len(missing):
        sys.exit(f'{len(missing)} KCs without a scale, e.g.\n{missing.head()}')
    d['value'] = d.value_z * d.scale_ratio
    d['dfc'] = d.value_z * d.sd_full
    back = d.value / d.scale_ratio
    assert np.allclose(back, d.value_z, rtol=1e-12, atol=0), float(np.abs(back - d.value_z).max())
    return d[KEYS + ['panel', 'odor', 'repeat', 'value', 'value_z', 'dfc', 'scale_ratio']]


def trial_means(d: pd.DataFrame) -> pd.DataFrame:
    """Each KC's mean over its trials of each odor, in both units (odor split into its abbreviation and log10
    concentration, as data/yang_apl16_kc_olfdb/computed/trialmean_responses_long has it)."""
    t = d.groupby(KEYS + ['odor'], sort=True).agg(n_trials=('repeat', 'size'), value=('value', 'mean'),
        value_z=('value_z', 'mean')).reset_index()
    sp = t.odor.str.split(' @ ', expand=True)
    t['odor'], t['log10_conc'] = sp[0], sp[1].astype(float)
    return t[KEYS + ['odor', 'log10_conc', 'n_trials', 'value', 'value_z']]


def response_fractions(t: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Per recording and odor (the solvent too), the fraction and number of KCs whose trial mean is above `thr`
    (baseline-noise units) and above 1 (Yang's z, the recordings' own criterion)."""
    t = t.assign(resp_noise=t.value > thr, resp_z=t.value_z > 1)
    return (t.groupby(['genotype', 'date', 'fly_num', 'hemibrain', 'odor'], sort=True)
        .agg(n_kcs=('roi', 'size'), n_resp_noise=('resp_noise', 'sum'), frac_noise=('resp_noise', 'mean'),
             n_resp_z=('resp_z', 'sum'), frac_z=('resp_z', 'mean')).reset_index())


def check_registry(t: pd.DataFrame, reg: dict, thr: float) -> dict:
    """The mean over recordings of each recording's share of KC x odor entries above `thr` over the 16 odors, against
    the registry's noise_ctrl_frac / noise_tnt_frac (data_targets.criterion_stats)."""
    out = {}
    t16 = t[t.odor.isin(H.APL16_ODORS)]
    assert t16.odor.nunique() == 16
    for g, key in (('TNT3c', 'noise_ctrl_frac'), ('TNT', 'noise_tnt_frac')):
        r = t16[t16.genotype == g].groupby(['date', 'fly_num', 'hemibrain']).value.apply(lambda v: float((v > thr).mean()))
        val, ref = float(r.mean()), float(reg[key]['value'])
        if abs(val - ref) > 1e-12:
            sys.exit(f'{g}: response rate {val!r} above {thr} against the registry\'s {key} {ref!r}')
        out[key] = {'here': val, 'registry': ref}
    return out


def olfdb_files(d: pd.DataFrame, out: Path, olfdb_src: Path) -> dict:
    """One parquet per genotype in olfdb's long upload layout (app/ingest.py: date, fly_num, roi, panel, odor, repeat,
    value, plus hemibrain as a further level of the ROIs, as Yang's datasets have it), each loaded with olfdb's own
    `load_file` when its source is on this host."""
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    loader = None
    if (olfdb_src / 'app' / 'ingest.py').exists():
        sys.path.insert(0, str(olfdb_src))
        try:
            from app.ingest import load_file as loader  # noqa: E402
        except Exception as e:  # its dependencies may be missing outside its container
            checks['loader'] = f'not checked: {type(e).__name__}: {e}'
    for ds, g in DATASETS.items():
        f = out / f'APL16_KCs_{g}_baseline_noise_units_long.parquet'
        sub = d[d.genotype == g][['date', 'fly_num', 'hemibrain', 'roi', 'panel', 'odor', 'repeat', 'value']]
        sub.to_parquet(f, index=False)
        rec = {'file': f.name, 'rows': len(sub), 'kcs': int(sub[['date', 'fly_num', 'hemibrain', 'roi']].drop_duplicates()
            .shape[0]), 'sha256': sha256(f), 'source_dataset': ds}
        if loader is not None:
            p = loader(f, original_name=f.name)
            vals = np.asarray(p.values, dtype=float)
            rec['olfdb_load_file'] = {'units': len(p.units), 'trials': len(p.trials), 'warnings': list(p.warnings),
                'n_values': int(np.isfinite(vals).sum()), 'sum_abs': float(np.nansum(np.abs(vals)))}
            assert len(p.units) == rec['kcs'] and len(p.trials) == 51, rec
            assert rec['olfdb_load_file']['n_values'] == len(sub), rec
            # olfdb keeps float32: the totals agree to its precision
            assert np.isclose(rec['olfdb_load_file']['sum_abs'], float(sub.value.abs().sum()), rtol=1e-5), rec
        checks[g] = rec
    return checks


README = """# Yang's APL16 KC data in baseline-noise units

Every trial of Yang's APL16 KC recordings (olfdb datasets 19, APL output blocked (TNT), and 20, control flies (TNT3c);
the same trials as data/yang_apl16_kc_olfdb/) with each response in units of the KC's own baseline noise, the KC data's
side of the realism bands since Tom's decision of 2026-11-10 16:10 (scripts/claude_pair_remapping_experiments/CLAUDE.md,
"The main goal for new experiments"; notes/response_criterion_decision.md). Written by
`al_analysis/scripts/claude_pair_remapping_experiments/yang_apl16_kc_noise_units.py` ({when}, al_analysis {commit});
re-run it to regenerate.

## Units

A KC's signal is suite2p's Fc = F - 0.7 Fneu. Yang z-scored it over the whole recording (about 7,194 volumes, 40 min)
and took a trial's response as the mean z 2-8 s after odor onset minus the mean over the 5 s before (the values olfdb
has). Its baseline noise, `sd_base`, is the SD of Fc within the 51 pre-odor windows (each window's own mean removed;
viki's yang_response_normalization.py, notes/yang_response_normalization/kc_scales.csv). A response in baseline-noise
units is the trial's change in Fc over `sd_base`, which is Yang's value times `scale_ratio` = `sd_full` / `sd_base`
(`sd_full` the whole-trace SD, her z unit). The realism bands count a trial mean above {thr:.4f} (the registry's
`noise_threshold`, at which the control flies keep their response rate by the recordings' own criterion, above 1 z:
{ctrl:.4f}) as a response; with APL output blocked (TNT) the rate is then {tnt:.4f}.

## Files

- `per_trial_long.parquet`: one row per KC, odor and trial (both genotypes, the solvent too): genotype, date, fly_num,
  hemibrain, roi (olfdb's, numbered within a recording), panel, odor (`<abbrev> @ <log10 conc>`), repeat, value
  (baseline-noise units), value_z (olfdb's value), dfc (the change in Fc, raw fluorescence units: value_z * sd_full),
  scale_ratio.
- `trialmean_long.parquet`, `trialmean_long.csv.gz`: each KC's mean over its three trials of each odor, in both units
  (genotype, date, fly_num, hemibrain, roi, odor, log10_conc, n_trials, value, value_z).
- `kc_scales.csv`: per KC, keyed as olfdb's ROIs: sd_full, sd_base, scale_ratio (from viki's table, whose `recording`
  and `kc` are kept), and the number of the 16 odors it responds to by each criterion (n_odors_noise above
  {thr:.4f}, n_odors_z above 1 z).
- `response_fraction_per_recording.csv`: per recording and odor (the solvent too), the number and fraction of KCs
  responding by each criterion.
- `olfdb/`: one file per genotype in olfdb's long upload layout (`app/ingest.py`: date, fly_num, hemibrain, roi, panel,
  odor, repeat, value; the value in baseline-noise units), parquet; `README_olfdb.md` there has the metadata for each
  upload. Checked with olfdb's own loader ({loader}).
- `provenance.json`: the inputs and their sha256, the commit, the threshold and the checks.

## Checks

Every KC of the 19 recordings has a scale; dividing by it gives olfdb's value back (relative error at most 1e-12);
the mean over recordings of each recording's share of KC x odor entries above the threshold, over the 16 odors, is the
registry's noise_ctrl_frac and noise_tnt_frac (to 1e-12). viki checked the scales' inputs against Yang's pipeline: the
z-scored traces and the per-trial responses recomputed from her suite2p output match her saved values and olfdb's to
at most 3e-6 in all 19 recordings (notes/yang_response_normalization/recordings.csv).

## In olfdb

Uploaded 2026-11-11 (olfdb ce400a5 and later keep several response calculations of the same recordings as datasets of their own): #22 (APL output blocked (TNT)) and #23 (control flies, TNT3c) in baseline-noise units, viki's unnormalized change in Fc as #24 / #25; #19 and #20 carry the response calculation "whole-recording z".

## Caveats

A baseline-noise unit is not a unit of activity: Fc is not divided by a resting fluorescence, so a response in these
units still depends on how bright a KC is. Only these 19 recordings have scales; the low-concentration panels and
Remy's recordings are in their own units. The flies' odor-odor distances (and the per-pair APL effect) everywhere else
are computed from the z values, not from these.
"""

README_OLFDB = """# Uploading these files to olfdb

One dataset per file, in olfdb's long layout (one row per value: date, fly_num, hemibrain, roi, panel, odor, repeat,
value), the same KCs, odors and trials as olfdb's datasets 19 and 20, with every value in units of the KC's baseline
noise instead of Yang's whole-recording z (../README.md). Metadata as datasets 19 and 20 (uploader Yang, MB247dbLexA >
GCaMP6s, KCs, 4-7 d, 925 nm, upstairs-left 2p, panel APL16), with:

| file | from dataset | title | rows | KCs |
|---|---|---|---:|---:|
{rows}

Notes for each (suggested; the dataset number from the table): "Responses of olfdb dataset <19 or 20> in units of each KC's baseline noise: a trial's change in Fc
(suite2p F - 0.7 Fneu) over the SD of Fc within the 51 pre-odor windows (each window's mean removed), i.e. Yang's
whole-recording z response times the KC's whole-trace SD over that baseline SD (viki's per-KC scales from Yang's suite2p
output, 2026-11-10). A trial mean above {thr:.4f} holds the control flies at their response rate by the 1 z criterion.
Made by al_analysis scripts/claude_pair_remapping_experiments/yang_apl16_kc_noise_units.py ({commit})."
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=SHARED / 'data' / 'yang_apl16_kc_noise_units')
    ap.add_argument('--olfdb-src', type=Path, default=Path('/opt/olfdb'))
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    scales_path = SHARED / 'notes' / 'yang_response_normalization' / 'kc_scales.csv'
    reg_path = SHARED / 'results' / 'data_targets' / 'data_targets.json'
    inputs = [scales_path, reg_path] + [SHARED / 'data' / 'yang_apl16_kc_olfdb' / g / f'dataset_{ds}_long.csv'
        for ds, g in DATASETS.items()]
    reg = json.loads(reg_path.read_text())
    thr = float(reg['noise_threshold']['value'])

    scales = kc_scales(scales_path)
    d = per_trial(scales)
    t = trial_means(d)
    checks = {'registry': check_registry(t, reg, thr)}

    n16 = (t[t.odor.isin(H.APL16_ODORS)].assign(noise=lambda x: x.value > thr, z=lambda x: x.value_z > 1)
        .groupby(KEYS).agg(n_odors_noise=('noise', 'sum'), n_odors_z=('z', 'sum')).reset_index())
    ks = scales.merge(n16, on=KEYS, how='inner', validate='one_to_one')
    assert len(ks) == len(n16) == d[KEYS].drop_duplicates().shape[0]
    checks['kcs'] = {g: int((ks.genotype == g).sum()) for g in DATASETS.values()}

    d.to_parquet(out / 'per_trial_long.parquet', index=False)
    t.to_parquet(out / 'trialmean_long.parquet', index=False)
    t.to_csv(out / 'trialmean_long.csv.gz', index=False)
    ks[['genotype', 'date', 'fly_num', 'hemibrain', 'roi', 'recording', 'sd_full', 'sd_base', 'scale_ratio',
        'n_odors_noise', 'n_odors_z']].to_csv(out / 'kc_scales.csv', index=False)
    response_fractions(t, thr).to_csv(out / 'response_fraction_per_recording.csv', index=False)

    checks['olfdb'] = olfdb_files(d, out / 'olfdb', args.olfdb_src)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=HERE, capture_output=True,
        text=True).stdout.strip())
    when = datetime.now().strftime('%Y-%m-%d %H:%M')
    loader = ('olfdb ' + ', '.join(f"{g}: {c['olfdb_load_file']['units']} KCs, {c['olfdb_load_file']['trials']} trials"
        for g, c in checks['olfdb'].items() if isinstance(c, dict) and 'olfdb_load_file' in c)) or checks['olfdb'].get(
        'loader', 'olfdb source not on this host: not checked')
    ctrl, tnt = (checks['registry'][k]['here'] for k in ('noise_ctrl_frac', 'noise_tnt_frac'))
    (out / 'README.md').write_text(README.format(when=when, commit=commit + (' (dirty)' if dirty else ''), thr=thr,
        ctrl=ctrl, tnt=tnt, loader=loader))
    rows = '\n'.join(f"| `{c['file']}` | {c['source_dataset']} | APL16: MB247dbLexA > GCaMP6s (KCs) [{g}], baseline-noise "
        f"units | {c['rows']} | {c['kcs']} |" for g, c in checks['olfdb'].items() if isinstance(c, dict) and 'file' in c)
    (out / 'olfdb' / 'README_olfdb.md').write_text(README_OLFDB.format(rows=rows, thr=thr, commit=commit))
    prov = {'script': Path(__file__).name, 'generated': when, 'al_analysis_commit': commit, 'dirty': dirty,
        'inputs': {str(p): sha256(p) for p in inputs}, 'noise_threshold': thr, 'checks': checks}
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print(json.dumps(checks, indent=1, default=str))


if __name__ == '__main__':
    main()
