#!/usr/bin/env python3
"""Yang's APL16 KC responses without any normalization: each trial's change in Fc, from her suite2p output.

The un-normalized counterpart of data/yang_apl16_kc_noise_units/ (hal's yang_apl16_kc_noise_units.py), written into
the same folder under separate names (Yang's request, 2026-11-10). For each of the 19 recordings in olfdb datasets 19 /
20 it reads, from Yang's data drive, suite2p's F.npy, Fneu.npy and iscell.npy (suite2p/combined of the
`<recording>_els_moco_s2p0` run), the recording's YAML (frame rate, frames per volume, baseline and response windows) and
the onset table (`<recording>_els_moco_thor_df_summary_n.csv`), and computes, as Yang's pipeline does but without her
whole-recording z-score: Fc = F - 0.7 Fneu for the ROIs with iscell == 1; per trial the mean Fc over volumes
round(2 v) .. round(8 v) - 1 after the onset volume minus the mean over the round(5 v) volumes before it (v the volume
rate). A response here is that change in Fc, in suite2p's raw fluorescence units (a.u.): not z-scored, not divided by a
resting fluorescence. The trial's pre-odor mean Fc is kept beside it (fc_baseline), so that dFc / Fc can be formed.

Rows are keyed as olfdb's: a recording's KCs in suite2p's iscell order, which is olfdb's `roi` (checked: the change in
Fc over the KC's whole-trace SD is olfdb's value, every trial of every recording; yang_response_normalization.py checked
the same order to 3e-6). Also checked against hal's `dfc` column (olfdb's value times the KC's whole-trace SD).

Outputs in <shared>/data/yang_apl16_kc_noise_units/ (`--out`):
- per_trial_long_unnormalized.parquet: genotype, date, fly_num, hemibrain, roi, panel, odor, repeat, value (the change in
  Fc, a.u.), fc_baseline (the trial's pre-odor mean Fc, a.u.), value_z (olfdb's), sd_full (the KC's whole-trace SD of Fc);
- trialmean_long_unnormalized.parquet / .csv.gz: each KC's mean over its trials of each odor (value, fc_baseline);
- olfdb/APL16_KCs_<TNT|TNT3c>_unnormalized_dFc_long.parquet: olfdb's long upload layout (date, fly_num, hemibrain, roi,
  panel, odor, repeat, value), value the change in Fc; olfdb/README_olfdb_unnormalized.md: the metadata for each upload;
- README_unnormalized.md, provenance_unnormalized.json.

Usage (on viki, where Yang's drive is mounted): ./venv/bin/python yang_apl16_kc_unnormalized.py [--drive-root /media/yang]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import kc_remapping_harness as H
from yang_response_normalization import ODOR_NAMES, RECORDINGS, find_one, olfdb_matrix

OUT = H.shared_dir() / 'data' / 'yang_apl16_kc_noise_units'
DATASETS = {'TNT': 19, 'TNT3c': 20}
PANEL = 'APL16'
KEYS = ['genotype', 'date', 'fly_num', 'hemibrain', 'roi']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def recording_rows(genotype: str, rec: str, drive_root: Path) -> tuple:
    """(per-trial rows of one recording, its check record)."""
    date = rec.split('_')[0]
    y = find_one(drive_root, f'YANGZHENG_0*/0_2P/{date}/{rec}.yaml')
    cfg = yaml.safe_load(y.read_text())
    day = y.parent.parent / f'{date}_fromHal'
    s2p = day / f'{rec}_els_moco_s2p0' / 'suite2p' / 'combined'
    pca = day / f'{rec}_els_moco_s2p0_PCA_result'
    F = np.load(s2p / 'F.npy').astype(np.float64)
    Fneu = np.load(s2p / 'Fneu.npy').astype(np.float64)
    iscell = np.load(s2p / 'iscell.npy')
    Fc = (F - 0.7 * Fneu)[iscell[:, 0] == 1]
    del F, Fneu
    sd_full = Fc.std(axis=1)
    vr = cfg['frame_rate'] / cfg['frame_per_volume']
    B, S, P = round(cfg['basePeriodLen'] * vr), round(cfg['peak_start'] * vr), round(cfg['peak_end'] * vr)
    onsets = pd.read_csv(pca / f'{rec}_els_moco_thor_df_summary_n.csv', index_col=0)['VC'].to_numpy(dtype=int)
    pins = np.array(cfg['odor_pin'])[np.array(cfg['odor_stim_seq']) - 1]
    names = np.array([ODOR_NAMES[p] for p in pins])
    K = len(onsets)
    rep = np.array([int((names[:i] == names[i]).sum()) for i in range(K)])
    base = np.stack([Fc[:, n - B:n].mean(axis=1) for n in onsets], axis=1)
    dfc = np.stack([Fc[:, n + S:n + P].mean(axis=1) for n in onsets], axis=1) - base
    n_kc = Fc.shape[0]
    fly, hemi = int(rec.split('_')[1][0]), rec.split('_')[1][2]
    d = pd.DataFrame(dict(genotype=genotype, date=f'{date[:4]}-{date[4:6]}-{date[6:8]}', fly_num=fly, hemibrain=hemi,
                          roi=np.repeat(np.arange(n_kc), K), panel=PANEL, odor=np.tile(names, n_kc), repeat=np.tile(rep, n_kc),
                          value=dfc.ravel(), fc_baseline=base.ravel(), sd_full=np.repeat(sd_full, K)))
    # olfdb's values: the change in Fc over the KC's whole-trace SD
    om = olfdb_matrix(genotype, rec)
    oz = om.stack([0, 1]).rename('value_z').reset_index().rename(columns={'level_1': 'odor', 'level_2': 'repeat'})
    oz.columns = ['roi', 'odor', 'repeat', 'value_z']
    d = d.merge(oz, on=['roi', 'odor', 'repeat'], how='left', validate='one_to_one')
    if d.value_z.isna().any():
        raise RuntimeError(f'{rec}: {int(d.value_z.isna().sum())} trials without an olfdb value')
    err = np.abs(d.value / d.sd_full - d.value_z)
    check = dict(recording=rec, genotype=genotype, n_kcs=n_kc, n_trials=K, volume_rate=vr, window=[S, P - 1], baseline_volumes=B,
                 max_abs_diff_z=float(err.max()), olfdb_kcs=int(len(om)))
    if len(om) != n_kc or check['max_abs_diff_z'] > 1e-5:
        raise RuntimeError(f'{rec}: does not match olfdb ({check})')
    return d, check


def olfdb_files(d: pd.DataFrame, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rec = {}
    for g, ds in DATASETS.items():
        f = out / f'APL16_KCs_{g}_unnormalized_dFc_long.parquet'
        sub = d[d.genotype == g][['date', 'fly_num', 'hemibrain', 'roi', 'panel', 'odor', 'repeat', 'value']]
        sub.to_parquet(f, index=False)
        rec[g] = dict(file=f.name, rows=len(sub), kcs=int(sub[['date', 'fly_num', 'hemibrain', 'roi']].drop_duplicates().shape[0]),
                      sha256=sha256(f), source_dataset=ds)
    return rec


def readme(out: Path, checks: list, hal: dict, olf: dict, commit: str) -> None:
    n = {g: sum(c['n_kcs'] for c in checks if c['genotype'] == g) for g in DATASETS}
    txt = f"""# Yang's APL16 KC responses without normalization (the change in Fc)

The un-normalized counterpart of the baseline-noise-unit files in this folder (README.md), at Yang's request
(2026-11-10). Written by `al_analysis/scripts/claude_pair_remapping_experiments/yang_apl16_kc_unnormalized.py`
({datetime.now():%Y-%m-%d %H:%M}, viki, al_analysis {commit}) from Yang's suite2p output on her data drive; re-run it there to
regenerate.

## What a value is

suite2p's F (the ROI's weighted pixel mean) and Fneu (the surrounding neuropil ring), for the ROIs kept as cells; Fc = F -
0.7 Fneu; per trial, the mean Fc over the 18 volumes from 2.0 to 8.1 s after the onset volume minus the mean over the 15
volumes (5.1 s) before it, at 59.273 frames/s and 20 frames per volume, as in Yang's pipeline but without her whole-recording
z-score. Units: suite2p's raw fluorescence (a.u.). Not divided by a resting fluorescence: `fc_baseline` (the trial's pre-odor
mean Fc) is given so that dFc / Fc can be formed; Fc includes the fixed 0.7 neuropil subtraction, so it is not a pure somatic
signal. Not comparable between KCs or recordings without some normalization (brightness, expression and depth differ).

These values are Yang's z responses times each KC's whole-trace SD of Fc (`sd_full`), since the z-score is linear: dividing
`value` by `sd_full` gives olfdb's value back to at most {max(c['max_abs_diff_z'] for c in checks):.1e} in every trial of the 19
recordings; and they equal the `dfc` column of per_trial_long.parquet (hal's, olfdb's value times `sd_full`) to at most
{hal['max_abs_diff']:.2e} a.u. (largest |value| {hal['max_abs_value']:.0f}).

## Files

- `per_trial_long_unnormalized.parquet`: one row per KC, odor and trial (both genotypes, the solvent too; {sum(n.values())} KCs):
  genotype, date, fly_num, hemibrain, roi (olfdb's), panel, odor (`<abbrev> @ <log10 conc>`), repeat, value (the change in Fc),
  fc_baseline, value_z (olfdb's), sd_full.
- `trialmean_long_unnormalized.parquet`, `.csv.gz`: each KC's mean over its trials of each odor (genotype, date, fly_num,
  hemibrain, roi, odor, log10_conc, n_trials, value, fc_baseline).
- `olfdb/APL16_KCs_TNT_unnormalized_dFc_long.parquet` ({olf['TNT']['kcs']} KCs) and
  `olfdb/APL16_KCs_TNT3c_unnormalized_dFc_long.parquet` ({olf['TNT3c']['kcs']} KCs): olfdb's long upload layout (date, fly_num,
  hemibrain, roi, panel, odor, repeat, value), value the change in Fc; `olfdb/README_olfdb_unnormalized.md` has the metadata.
- `provenance_unnormalized.json`: the per-recording checks, the inputs' locations, the commit.
"""
    (out / 'README_unnormalized.md').write_text(txt)
    olf_txt = f"""# Uploading the un-normalized files to olfdb

One dataset per file, in olfdb's long layout (one row per value: date, fly_num, hemibrain, roi, panel, odor, repeat, value),
the same KCs, odors and trials as olfdb's datasets 19 and 20, with every value the trial's change in Fc (suite2p's raw
fluorescence units) instead of Yang's whole-recording z (../README_unnormalized.md). Metadata as datasets 19 and 20
(uploader Yang, MB247dbLexA > GCaMP6s, KCs, 4-7 d, 925 nm, upstairs-left 2p, panel APL16), with:

| file | from dataset | title | rows | KCs |
|---|---|---|---:|---:|
| `{olf['TNT']['file']}` | 19 | APL16: MB247dbLexA > GCaMP6s (KCs) [TNT], unnormalized (dFc) | {olf['TNT']['rows']} | {olf['TNT']['kcs']} |
| `{olf['TNT3c']['file']}` | 20 | APL16: MB247dbLexA > GCaMP6s (KCs) [TNT3c], unnormalized (dFc) | {olf['TNT3c']['rows']} | {olf['TNT3c']['kcs']} |

Notes for each (suggested; the dataset number from the table): "Responses of olfdb dataset <19 or 20> without normalization: a
trial's mean Fc (suite2p F - 0.7 Fneu) 2-8 s after odor onset minus its mean over the 5 s before, in raw fluorescence units,
from Yang's suite2p output (the same responses as the dataset's, before her whole-recording z-score; value / the KC's
whole-trace SD of Fc gives the dataset's value). Made by al_analysis scripts/claude_pair_remapping_experiments/
yang_apl16_kc_unnormalized.py ({commit})." Not checked with olfdb's own loader on viki (/opt/olfdb is on hal); the layout,
column names and types are those of the baseline-noise-unit files, which were.
"""
    (out / 'olfdb' / 'README_olfdb_unnormalized.md').write_text(olf_txt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--drive-root', type=Path, default=Path('/media/yang'))
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    parts, checks = [], []
    for g, rec in RECORDINGS:
        d, c = recording_rows(g, rec, args.drive_root)
        parts.append(d)
        checks.append(c)
        print(f"{g} {rec}: {c['n_kcs']} KCs x {c['n_trials']} trials; value / sd_full vs olfdb max |diff| {c['max_abs_diff_z']:.1e}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    # against hal's dfc (olfdb's value times sd_full)
    h = pd.read_parquet(args.out / 'per_trial_long.parquet', columns=KEYS + ['odor', 'repeat', 'dfc'])
    m = d.merge(h, on=KEYS + ['odor', 'repeat'], how='left', validate='one_to_one')
    if m.dfc.isna().any():
        sys.exit(f'{int(m.dfc.isna().sum())} trials missing from per_trial_long.parquet')
    hal = dict(max_abs_diff=float(np.abs(m.value - m.dfc).max()), max_abs_value=float(np.abs(m.value).max()))
    d = d[KEYS + ['panel', 'odor', 'repeat', 'value', 'fc_baseline', 'value_z', 'sd_full']]
    d.to_parquet(args.out / 'per_trial_long_unnormalized.parquet', index=False)
    t = d.groupby(KEYS + ['odor'], sort=True).agg(n_trials=('repeat', 'size'), value=('value', 'mean'),
                                                  fc_baseline=('fc_baseline', 'mean')).reset_index()
    sp = t.odor.str.split(' @ ', expand=True)
    t['odor'], t['log10_conc'] = sp[0], sp[1].astype(float)
    t = t[KEYS + ['odor', 'log10_conc', 'n_trials', 'value', 'fc_baseline']]
    t.to_parquet(args.out / 'trialmean_long_unnormalized.parquet', index=False)
    t.to_csv(args.out / 'trialmean_long_unnormalized.csv.gz', index=False)
    olf = olfdb_files(d, args.out / 'olfdb')
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    readme(args.out, checks, hal, olf, commit)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='yang_apl16_kc_unnormalized.py',
                al_analysis_commit=commit, drive_root=str(args.drive_root), checks=checks, against_hal_dfc=hal, olfdb=olf,
                files={p.name: sha256(p) for p in (args.out / 'per_trial_long_unnormalized.parquet',
                                                   args.out / 'trialmean_long_unnormalized.parquet',
                                                   args.out / 'trialmean_long_unnormalized.csv.gz')})
    (args.out / 'provenance_unnormalized.json').write_text(json.dumps(prov, indent=1) + '\n')
    print(json.dumps(dict(against_hal_dfc=hal, olfdb=olf), indent=1))


if __name__ == '__main__':
    main()
