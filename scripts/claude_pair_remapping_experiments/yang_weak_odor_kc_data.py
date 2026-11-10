#!/usr/bin/env python3
"""Export Yang's diagnostic-odor mixture and dilute natural-mixture KC data, as APL16's.

Reads Yang's two other KC datasets from al_analysis/data/from_yang/20260415_YZtransfer/
(committed in this repository):
- `diag_mix`: the diagnostic-odor binary mixtures (2h, ma, farn and their pairs; WT,
  TNTin and TNT flies, 2025-01-29 to 03-26), per-trial z-scored responses in
  `20250129_0326_KC_odormix_resp_good.csv`
- `natmix_dilute`: the dilute natural-mixture components (2h + ma, 2h + 1o3ol, ea + eb;
  WT, 2026-01-16 to 02-10), one per-trial file per recording in `20260116_20260210/`

and writes <shared>/data/yang_weak_odor_kc/<panel>/ with the per-trial responses in
olfdb's long format (plus recording, genotype columns) and, under computed/, the same
derived files as yang_apl16_kc_data.py makes for the APL16 data: trial means, response
fractions (trial mean > 1, and Yang's > 1 on at least half the trials), per-recording
odor x odor RDMs (1 - Pearson, 1 - cosine over all KCs) and their recording and fly
means. Every recording and genotype is kept, both hemispheres of a fly as separate
recordings. Yang's binarized files and response rates are reproduced from the per-trial
data (checks.txt). Also writes, at the top level, which ORN measurements could stand in
as model input for each odor (orn_input_availability*.csv), from the committed ORN spike
deltas and olfdb's pebbled glomerulus datasets, and README.md describing everything.

Usage:
    ./venv/bin/python yang_weak_odor_kc_data.py [--shared-dir DIR] [--olfdb URL]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from yang_apl16_kc_data import rdm_long

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RAW_DIR = REPO / 'al_analysis' / 'data' / 'from_yang' / '20260415_YZtransfer'
ORN_DELTA_DIR = REPO / 'al_analysis' / 'data' / 'internal'
OUT_NAME = 'yang_weak_odor_kc'
DEFAULT_OLFDB = 'http://hal:8080'

RESPONSE_THRESH = 1.0
# Yang's binarization ('respthresh1_trialfr0.5'): > RESPONSE_THRESH on at least this
# fraction of the recording's trials of the odor
YANG_TRIAL_FRAC = 0.5
# Tom's rule for ORN data at another concentration standing in (log10 v/v, inclusive)
ORN_SUBSTITUTE_MAX_LOG10 = 1.0

DIAG_CSV = '20250129_0326_KC_odormix_resp_good.csv'
DIAG_BINARY_SUFFIX = '_resp_binary_respthresh1_trialfr0p5_wmixpred.csv'
DIAG_RESPRATE_CSV = 'All_resprate_respthresh1_trialfr0p5_wmixpred.csv'
DILUTE_SUBDIR = '20260116_20260210'
DILUTE_RESP_SUFFIX = (
    '_els_moco_s2p_mean_curated_Fc_zscored_full_cell_resp_df_odor_sorted.csv'
)
DILUTE_BINARY_SUFFIX = '_responsive_binlabel_df_respthresh1_trialfr0.5.csv'
DILUTE_RESPRATE_SUFFIX = '_resp_rate_respthresh1trialfr0p5_wmixpred.csv'

# Yang's odor names -> olfdb abbreviations (olfdb's odors table, /api/odors)
ABBREV = {'2h': '2h', 'ma': 'ma', 'farn': 'farn', 'ea': 'ea', 'eb': 'eb',
    '1o3ol': '1o3ol', 'Iaa': 'IaA'}
# Yang's labels that carry no concentration. pfo is written as olfdb writes it. olfdb
# has no 'air' and no 'Bmix28'; banana is olfdb's 'ban', but Yang's file gives no
# concentration, so none is written (README).
NO_CONC = {'PFO': 'pfo @ 0', 'air': 'air', 'banana': 'ban', 'Bmix28': 'Bmix28'}
CONTROL_ODORS = ('pfo @ 0', 'air')
COMPLEX_ODORS = ('ban', 'Bmix28')
OLFDB_ABBREVS_USED = sorted(set(ABBREV.values()) | {'pfo', 'ban'})

# Yang's labels in the order used for every output (the set is checked against the data)
PANEL_ODORS = {
    'diag_mix': ['2h(-8)', '2h(-7)', 'ma(-8)', 'ma(-7)', 'farn(-4)', 'farn(-3)',
        'farn(-2.5)', '2h(-7)+ma(-7)', '2h(-7)+farn(-3)', '2h(-7)+farn(-2.5)',
        'ma(-7)+farn(-3)', 'ma(-7)+farn(-2.5)', 'PFO'],
    'natmix_dilute': ['2h(-7)', 'ma(-7)', '2h(-7)+ma(-7)', '2h(-5)', '1o3ol(-5)',
        '2h(-5)+1o3ol(-5)', '2h(-7)+1o3ol(-5)', 'ea(-7)', 'eb(-7)', 'ea(-7)+eb(-7)',
        'ea(-5)', 'eb(-5)', 'ea(-5)+eb(-5)', 'Iaa(-2)', 'banana', 'Bmix28', 'PFO', 'air'],
}

# Yang's exp_type -> genotype group. TNT / TNTin flies carry Flp-out UAS-TeTx (active)
# or UAS-TeTxin (inactive) in APL; Flp is stochastic, so each hemisphere's APL either
# expresses it ('label') or not ('nolabel'). Only TNT_label blocks APL output.
EXP_TYPE2GROUP = {'WT': 'control', 'TNTin_label': 'control', 'TNTin_nolabel': 'control',
    'TNT_nolabel': 'control', 'TNT_label': 'apl_output_blocked'}
EXP_TYPE2GENOTYPE = {'WT': 'WT', 'TNTin_label': 'TNTin', 'TNTin_nolabel': 'TNTin',
    'TNT_nolabel': 'TNT', 'TNT_label': 'TNT'}
GROUP_LABEL = {'control': 'control flies', 'apl_output_blocked': 'APL output blocked (TNT)'}

OLFDB_ORN_DATASETS = (1, 4, 5, 6, 21)
ORN_DELTA_CSVS = ('kiwi-control_mean_est_spike_deltas.csv',
    'megamat_mean_est_spike_deltas.csv', 'validation2_mean_est_spike_deltas.csv')
# air mixtures in the kiwi-control spike deltas, named there without their components;
# the components are as olfdb dataset 5 (the same flies) names the same mixtures
ORN_DELTA_AIR_MIXES = {
    'ea+eb (air mix) @ 0': 'ea @ -4.2 + eb @ -3.5',
    '1o3ol+2h (air mix) @ 0': '1o3ol @ -3 + 2h @ -5',
}

RECORDING_COLS = ['recording', 'exp_type', 'genotype_group', 'date', 'fly_num',
    'hemibrain']
LONG_COLS = ['date', 'fly_num', 'hemibrain', 'roi', 'panel', 'odor', 'repeat', 'value',
    'recording', 'recording_num', 'sex', 'exp_type', 'genotype_group']
MEASURES = ['one_minus_pearson', 'one_minus_cosine']
ODOR_SETS = ['pure_no_pfo', 'pure_with_pfo', 'all_no_pfo', 'all_with_pfo']

_COMP_RE = re.compile(r'^([A-Za-z0-9-]+)\((-?\d+(?:\.\d+)?)\)$')
_BRAIN_RE = re.compile(r'^(\d{4})(\d{2})(\d{2})_(\d+)([FM])([LR])fun(\d+)$')
_TRIAL_COL_RE = re.compile(r'^(.*?)(?:\.(\d+))?$')


class Checks:
    """Collects named pass / fail checks, written to a file before failing on any."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.failed: List[str] = []

    def section(self, title: str) -> None:
        self.lines.append(f'\n## {title}')

    def check(self, name: str, ok: bool, detail: str = '') -> bool:
        line = f'{"PASS" if ok else "FAIL"}: {name}' + (f' ({detail})' if detail else '')
        self.lines.append(line)
        if not ok:
            self.failed.append(line)
            print(line, file=sys.stderr)
        return ok

    def note(self, text: str) -> None:
        self.lines.append(f'NOTE: {text}')


def default_shared_dir() -> Path:
    """The shared directory as kc_remapping_harness.shared_dir() finds it (without its
    model imports)."""
    cands = [os.environ.get('KC_REMAPPING_SHARED')] + [
        f'/{root}/tensor/Tom/claude_kc_remapping' for root in ('local', 'mnt', 'media')
    ]
    for c in cands:
        if c and Path(c).is_dir():
            return Path(c)
    raise FileNotFoundError('shared directory not found; set KC_REMAPPING_SHARED or pass '
        '--shared-dir')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_json(url: str, cache: Path):
    """GETs url as JSON, refreshing a cached copy; falls back to the cache if offline."""
    try:
        with urlopen(url, timeout=120) as r:
            data = r.read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    except (URLError, OSError) as e:
        if not cache.exists():
            raise
        print(f'warning: could not fetch {url} ({e}); using cached {cache}')
        data = cache.read_bytes()
    return json.loads(data)


def yang2olfdb(label: str) -> str:
    """Yang's odor label -> olfdb odor string.

    '2h(-7)' -> '2h @ -7', '2h(-7)+farn(-2.5)' -> '2h @ -7 + farn @ -2.5' (an air mixture,
    components in Yang's order, as olfdb writes air mixtures), 'Iaa(-2)' -> 'IaA @ -2',
    'PFO' -> 'pfo @ 0', and the labels without a concentration as in `NO_CONC`.
    """
    if label in NO_CONC:
        return NO_CONC[label]
    parts = []
    for comp in label.split('+'):
        m = _COMP_RE.match(comp)
        if m is None:
            raise ValueError(f'unexpected odor label {label!r}')
        name, conc = m.groups()
        parts.append(f'{ABBREV[name]} @ {float(conc):g}')
    return ' + '.join(parts)


def odor_kind(odor: str) -> str:
    """'control', 'air mixture', 'complex' (single stimulus of many chemicals) or
    'monomolecular'."""
    if odor in CONTROL_ODORS:
        return 'control'
    if ' + ' in odor:
        return 'air mixture'
    if odor in COMPLEX_ODORS:
        return 'complex'
    return 'monomolecular'


def split_odor(odor: str) -> Tuple[str, Optional[float]]:
    """'2h @ -7' -> ('2h', -7.0); 'ban' -> ('ban', None)."""
    if ' @ ' not in odor:
        return odor, None
    name, conc = odor.split(' @ ')
    return name, float(conc)


def parse_brain(brain: str) -> dict:
    """Yang's recording id, e.g. '20250129_1FLfun1' (date, fly 1, female, left
    hemisphere, functional recording 1) -> metadata dict."""
    m = _BRAIN_RE.match(brain)
    if m is None:
        raise ValueError(f'unexpected recording id {brain!r}')
    y, mo, d, fly, sex, side, fun = m.groups()
    return dict(recording=brain, date=f'{y}-{mo}-{d}', fly_num=int(fly), sex=sex,
        hemibrain=side, recording_num=int(fun))


def trial_columns(labels: Sequence[str], suffixed: bool) -> pd.MultiIndex:
    """Per-trial column names -> (Yang label, repeat) MultiIndex.

    Args:
        labels: column names in file order
        suffixed: True if repeats are written as pandas-style suffixes ('2h(-7)',
            '2h(-7).1', ...), False if the names repeat ('2h(-7)', '2h(-7)', ...); in the
            latter case repeats are numbered in column order.
    """
    out, counts = [], {}
    for c in labels:
        if suffixed:
            base, rep = _TRIAL_COL_RE.match(c).groups()
            rep = int(rep or 0)
        else:
            base = c
            rep = counts.get(c, 0)
            counts[c] = rep + 1
        out.append((base, rep))
    idx = pd.MultiIndex.from_tuples(out, names=['yang_label', 'repeat'])
    assert not idx.duplicated().any()
    return idx


def finish_wide(wide: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Drops odor-trial columns the recording does not have (all NaN), checks the rest are
    complete with repeats 0..n-1, and sorts columns into panel odor order."""
    wide = wide.loc[:, wide.notna().any()]
    assert wide.notna().all().all(), 'partially missing trial'
    labels = wide.columns.get_level_values('yang_label')
    unknown = set(labels) - set(order)
    assert not unknown, f'unexpected odors {unknown}'
    for lab in set(labels):
        reps = sorted(wide.columns[labels == lab].get_level_values('repeat'))
        assert reps == list(range(len(reps))), (lab, reps)
    key = {lab: i for i, lab in enumerate(order)}
    cols = sorted(wide.columns, key=lambda c: (key[c[0]], c[1]))
    wide = wide[cols]
    wide.index.name = 'roi'
    return wide


def load_diag_mix(raw_dir: Path) -> List[Tuple[dict, pd.DataFrame]]:
    """Per recording: (metadata, KC x (Yang label, repeat) per-trial responses)."""
    df = pd.read_csv(raw_dir / DIAG_CSV, index_col=0)
    trial_cols = [c for c in df.columns if c not in ('brain', 'exp_type')]
    assert not df[['brain', 'exp_type']].isna().any().any()
    assert (df.groupby('brain').exp_type.nunique() == 1).all()
    out = []
    for brain, sub in df.groupby('brain', sort=False):
        meta = parse_brain(brain)
        meta['exp_type'] = sub.exp_type.iloc[0]
        assert (sub.index == np.arange(len(sub))).all(), f'{brain}: KC ids not 0..n-1'
        wide = sub[trial_cols].astype(float)
        wide.columns = trial_columns(trial_cols, suffixed=True)
        out.append((meta, finish_wide(wide, PANEL_ODORS['diag_mix'])))
    return out


def read_dup_header_csv(path: Path) -> Tuple[List[str], pd.DataFrame]:
    """Reads a CSV whose header repeats names (one column per trial), keeping the names."""
    with open(path, newline='') as f:
        header = next(csv.reader(f))
    df = pd.read_csv(path, header=None, skiprows=1, index_col=0)
    assert len(df.columns) == len(header) - 1
    return header[1:], df


def load_natmix_dilute(raw_dir: Path) -> List[Tuple[dict, pd.DataFrame]]:
    """Per recording: (metadata, KC x (Yang label, repeat) per-trial responses)."""
    out = []
    for p in sorted((raw_dir / DILUTE_SUBDIR).glob(f'*{DILUTE_RESP_SUFFIX}')):
        brain = p.name[:-len(DILUTE_RESP_SUFFIX)]
        meta = parse_brain(brain)
        # all WT (Tom's loader and Yang's notes)
        meta['exp_type'] = 'WT'
        labels, wide = read_dup_header_csv(p)
        assert (wide.index == np.arange(len(wide))).all(), f'{brain}: KC ids not 0..n-1'
        wide = wide.astype(float)
        wide.columns = trial_columns(labels, suffixed=False)
        out.append((meta, finish_wide(wide, PANEL_ODORS['natmix_dilute'])))
    return out


def yang_binarize(wide: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """KC x label: > RESPONSE_THRESH on at least YANG_TRIAL_FRAC of the trials (more than,
    if strict)."""
    above = (wide > RESPONSE_THRESH).T.groupby(level='yang_label', sort=False).sum().T
    n = wide.notna().T.groupby(level='yang_label', sort=False).sum().T
    frac = above / n
    return frac > YANG_TRIAL_FRAC if strict else frac >= YANG_TRIAL_FRAC


def trial_mean(wide: pd.DataFrame) -> pd.DataFrame:
    """KC x label trial means."""
    return wide.T.groupby(level='yang_label', sort=False).mean().T


def mix_components(label: str) -> List[str]:
    """Yang's mixture label -> its component labels ('2h(-7)+ma(-7)' -> ['2h(-7)',
    'ma(-7)'])."""
    return label.split('+')


def to_long(meta: dict, wide: pd.DataFrame, panel: str) -> pd.DataFrame:
    """Per-trial responses in olfdb's long format plus recording / genotype columns."""
    n_kc, n_col = wide.shape
    labels = wide.columns.get_level_values('yang_label')
    odors = np.array([yang2olfdb(x) for x in labels], dtype=object)
    df = pd.DataFrame({
        'date': meta['date'],
        'fly_num': meta['fly_num'],
        'hemibrain': meta['hemibrain'],
        'roi': np.repeat(wide.index.to_numpy(), n_col),
        'panel': panel,
        'odor': np.tile(odors, n_kc),
        'repeat': np.tile(wide.columns.get_level_values('repeat').to_numpy(), n_kc),
        'value': wide.to_numpy().ravel(),
        'recording': meta['recording'],
        'recording_num': meta['recording_num'],
        'sex': meta['sex'],
        'exp_type': meta['exp_type'],
        'genotype_group': meta['genotype_group'],
    })
    return df[LONG_COLS]


def check_diag_yang_files(recs: List[Tuple[dict, pd.DataFrame]], raw_dir: Path,
    checks: Checks) -> List[str]:
    """Compares Yang's binary and response rate files with the per-trial data.

    Returns:
        recording ids with a binary file but no per-trial data.
    """
    checks.section('diag_mix: Yang\'s files reproduced from the per-trial data')
    by_id = {m['recording']: (m, w) for m, w in recs}
    bin_files = {p.name[:-len(DIAG_BINARY_SUFFIX)]: p
        for p in sorted(raw_dir.glob(f'*{DIAG_BINARY_SUFFIX}'))}
    binary_only = sorted(set(bin_files) - set(by_id))
    checks.check('every recording in the per-trial file has a binary file',
        set(by_id) <= set(bin_files), f'{len(by_id)} recordings, {len(bin_files)} binary '
        f'files')
    checks.note(f'{len(binary_only)} binary files have no per-trial data (not in '
        f'{DIAG_CSV} or {DIAG_RESPRATE_CSV}; not exported): {", ".join(binary_only)}')

    n_kc_ok, bin_ok, pred_ok = [], [], []
    n_cells, n_mismatch = 0, 0
    for rid, p in bin_files.items():
        b = pd.read_csv(p, index_col=0)
        assert (b.experiment == rid).all(), rid
        b = b.drop(columns='experiment')
        pred_cols = [c for c in b.columns if c.endswith('_predicted')]
        # her '_predicted' columns: union of the component responders
        for c in pred_cols:
            union = b[mix_components(c[:-len('_predicted')])].any(axis=1)
            pred_ok.append(bool((union == b[c]).all()))
        if rid not in by_id:
            continue
        meta, wide = by_id[rid]
        n_kc_ok.append(len(b) == len(wide) and (b.index == wide.index).all())
        mine = yang_binarize(wide)
        odor_cols = [c for c in b.columns if c not in pred_cols]
        same_odors = set(odor_cols) == set(mine.columns)
        bin_ok.append(same_odors)
        if same_odors:
            mism = int((mine[odor_cols].to_numpy() != b[odor_cols].to_numpy()).sum())
            n_mismatch += mism
            n_cells += b[odor_cols].size
            bin_ok[-1] = mism == 0
    checks.check('number of KCs per recording equals the rows of Yang\'s binary file',
        all(n_kc_ok), f'{sum(n_kc_ok)} of {len(n_kc_ok)} recordings')
    checks.check('Yang\'s binary files reproduced by > 1 on at least 50 % of trials',
        all(bin_ok) and n_mismatch == 0, f'{n_mismatch} of {n_cells} KC x odor cells '
        f'differ, {sum(bin_ok)} of {len(bin_ok)} recordings identical')
    checks.check('Yang\'s mixture _predicted columns equal the union of the component '
        'responders', all(pred_ok), f'{sum(pred_ok)} of {len(pred_ok)} columns, '
        f'{len(bin_files)} files incl. the binary-only recordings')

    rr = pd.read_csv(raw_dir / DIAG_RESPRATE_CSV, index_col=0)
    checks.check(f'{DIAG_RESPRATE_CSV} covers exactly the per-trial recordings',
        set(rr.experiment) == set(by_id), f'{rr.experiment.nunique()} recordings')
    diffs = []
    for rid, sub in rr.groupby('experiment', sort=False):
        mine = yang_binarize(by_id[rid][1])
        for odor, rate in zip(sub.odor, sub.response_rate):
            if odor.endswith('_predicted'):
                v = mine[mix_components(odor[:-len('_predicted')])].any(axis=1).mean()
            else:
                v = mine[odor].mean()
            diffs.append(abs(v - rate))
    checks.check(f'{DIAG_RESPRATE_CSV} response rates (and _predicted rates) reproduced',
        max(diffs) < 1e-12, f'{len(diffs)} rows, max |diff| {max(diffs):.2g}')
    return binary_only


def check_dilute_yang_files(recs: List[Tuple[dict, pd.DataFrame]], raw_dir: Path,
    checks: Checks) -> Dict[str, str]:
    """Compares Yang's per-recording binary and response rate files with the per-trial data.

    Returns:
        dict recording id -> which rule her response rate file matches.
    """
    checks.section('natmix_dilute: Yang\'s files reproduced from the per-trial data')
    sub = raw_dir / DILUTE_SUBDIR
    files = [p for p in sub.glob('*.csv') if p.is_file()]
    checks.check('the folder holds exactly 3 files per recording',
        len(files) == 3 * len(recs), f'{len(files)} files, {len(recs)} recordings')
    n_kc_ok, bin_ok, rules = [], [], {}
    n_cells, n_mismatch = 0, 0
    for meta, wide in recs:
        rid = meta['recording']
        b = pd.read_csv(sub / f'{rid}{DILUTE_BINARY_SUFFIX}', index_col=0)
        n_kc_ok.append(len(b) == len(wide) and (b.index == wide.index).all())
        mine = yang_binarize(wide)
        same = set(b.columns) == set(mine.columns)
        if same:
            mism = int((mine[list(b.columns)].to_numpy() != b.to_numpy()).sum())
            n_mismatch += mism
            n_cells += b.size
            same = mism == 0
        bin_ok.append(same)

        rr = pd.read_csv(sub / f'{rid}{DILUTE_RESPRATE_SUFFIX}', index_col=0).set_index(
            'odor').percent_cells_responding
        matched = None
        for rule, strict in (('>= 50 % of trials', False), ('> 50 % of trials', True)):
            m = yang_binarize(wide, strict=strict)
            vals = []
            for odor, pct in rr.items():
                if odor.endswith('_predicted'):
                    v = m[mix_components(odor[:-len('_predicted')])].any(axis=1).mean()
                else:
                    v = m[odor].mean()
                vals.append(abs(100 * v - pct))
            if set(rr.index) - {o for o in rr.index if o.endswith('_predicted')} == set(
                m.columns) and max(vals) < 1e-9:
                matched = rule
                break
        rules[rid] = matched
    checks.check('number of KCs per recording equals the rows of Yang\'s binary file',
        all(n_kc_ok), f'{sum(n_kc_ok)} of {len(n_kc_ok)} recordings')
    checks.check('Yang\'s binary files reproduced by > 1 on at least 50 % of trials',
        all(bin_ok) and n_mismatch == 0, f'{n_mismatch} of {n_cells} KC x odor cells '
        f'differ, {sum(bin_ok)} of {len(bin_ok)} recordings identical')
    n_main = sum(r == '>= 50 % of trials' for r in rules.values())
    checks.check('Yang\'s per-recording response rate files (incl. _predicted = union of '
        'component responders) reproduced', all(r is not None for r in rules.values()),
        f'{n_main} of {len(rules)} by > 1 on at least 50 % of trials')
    walls = {m['recording']: w for m, w in recs}
    for rid, rule in rules.items():
        if rule != '>= 50 % of trials':
            n_tr = sorted(set(walls[rid].columns.to_frame(index=False).groupby(
                'yang_label').size()))
            checks.note(f'{rid} ({n_tr} trials per odor): its response rate file matches '
                f'{rule} (> 1 on every trial), while its binary file (dated later) uses at '
                'least 50 %; the outputs here use at least 50 %')
    return rules


def recording_tables(panel: str, recs: List[Tuple[dict, pd.DataFrame]]) -> dict:
    """Trial means, response fractions and RDMs for one panel's recordings."""
    order = [yang2olfdb(x) for x in PANEL_ODORS[panel]]
    tm_rows, frac_rows, pair_rows, rec_rows = [], [], [], []
    for meta, wide in recs:
        rec_vals = {k: meta[k] for k in RECORDING_COLS}
        tm = trial_mean(wide)
        yb = yang_binarize(wide)
        n_tr = wide.columns.to_frame(index=False).groupby('yang_label', sort=False).size()
        labels = list(tm.columns)
        odors = [yang2olfdb(x) for x in labels]
        tm.columns = yb.columns = odors
        n_tr.index = [yang2olfdb(x) for x in n_tr.index]

        rec_rows.append({**rec_vals, 'recording_num': meta['recording_num'],
            'sex': meta['sex'], 'genotype': EXP_TYPE2GENOTYPE[meta['exp_type']],
            'n_kcs': len(wide), 'n_odors': len(odors),
            **{f'n_trials {o}': n_tr.get(o, 0) for o in order}})

        t = tm.stack().rename('value').reset_index()
        t.columns = ['roi', 'odor', 'value']
        t['kind'] = t.odor.map(odor_kind)
        t['log10_conc'] = t.odor.map(lambda o: np.nan if ' + ' in o else
            split_odor(o)[1])
        t['n_trials'] = t.odor.map(n_tr).astype(int)
        for k in reversed(RECORDING_COLS):
            t.insert(0, k, rec_vals[k])
        tm_rows.append(t[RECORDING_COLS + ['roi', 'odor', 'kind', 'log10_conc',
            'n_trials', 'value']])

        resp_tm = tm > RESPONSE_THRESH
        non_control = [o for o in odors if odor_kind(o) != 'control']
        any_tm = resp_tm[non_control].any(axis=1).mean()
        any_y = yb[non_control].any(axis=1).mean()
        for lab, o in zip(labels, odors):
            row = {**rec_vals, 'odor': o, 'kind': odor_kind(o),
                'n_trials': int(n_tr[o]), 'n_kcs': len(wide),
                'n_trialmean_gt1': int(resp_tm[o].sum()),
                'frac_trialmean_gt1': resp_tm[o].mean(),
                'n_yang': int(yb[o].sum()), 'frac_yang': yb[o].mean(),
                'frac_union_components_trialmean': np.nan,
                'frac_union_components_yang': np.nan,
                'frac_kcs_responding_any_odor_trialmean': any_tm,
                'frac_kcs_responding_any_odor_yang': any_y}
            if '+' in lab:
                comps = [yang2olfdb(c) for c in mix_components(lab)]
                row['frac_union_components_trialmean'] = resp_tm[comps].any(axis=1).mean()
                row['frac_union_components_yang'] = yb[comps].any(axis=1).mean()
            frac_rows.append(row)

        pr = rdm_long(tm.T, odors)
        for k in reversed(RECORDING_COLS):
            pr.insert(0, k, rec_vals[k])
        pair_rows.append(pr)

    pairs_all = pd.concat(pair_rows, ignore_index=True)
    kinds = {o: odor_kind(o) for o in order}
    sets = {
        'pure_no_pfo': {o for o, k in kinds.items() if k == 'monomolecular'},
        'all_no_pfo': {o for o, k in kinds.items() if k != 'control'},
    }
    sets['pure_with_pfo'] = sets['pure_no_pfo'] | set(CONTROL_ODORS)
    sets['all_with_pfo'] = set(order)
    per_set = []
    for name in ODOR_SETS:
        s = sets[name]
        p = pairs_all[pairs_all.odor_a.isin(s) & pairs_all.odor_b.isin(s)].copy()
        p.insert(len(RECORDING_COLS), 'odor_set', name)
        per_set.append(p)
    pairs = pd.concat(per_set, ignore_index=True)

    recordings = pd.DataFrame(rec_rows)
    # drop odors this panel never has (none, but keeps the columns honest)
    recordings = recordings.loc[:, (recordings != 0).any() | ~recordings.columns.str
        .startswith('n_trials ')]
    return dict(order=order, sets=sets, recordings=recordings,
        trialmeans=pd.concat(tm_rows, ignore_index=True),
        fracs=pd.DataFrame(frac_rows), pairs=pairs)


def mean_tables(pairs: pd.DataFrame, fracs: pd.DataFrame) -> dict:
    """Recording- and fly-mean RDMs, and recording means of the response fractions."""
    keys = ['genotype_group', 'odor_set', 'odor_a', 'odor_b']
    rec_mean = pairs.groupby(keys, sort=False).agg(
        **{m: (m, 'mean') for m in MEASURES},
        n_recordings=('recording', 'nunique'),
        n_flies=('date', lambda s: len(set(zip(s, pairs.loc[s.index, 'fly_num']))))
    ).reset_index()
    by_fly = pairs.groupby(keys[:2] + ['date', 'fly_num'] + keys[2:], sort=False)[
        MEASURES].mean().reset_index()
    fly_mean = by_fly.groupby(keys, sort=False).agg(**{m: (m, 'mean') for m in MEASURES},
        n_flies=(MEASURES[0], 'size')).reset_index()

    vals = ['frac_trialmean_gt1', 'frac_yang', 'frac_union_components_trialmean',
        'frac_union_components_yang']
    out = []
    for grouping in ('genotype_group', 'exp_type'):
        g = fracs.groupby([grouping, 'odor'], sort=False)
        m = g[vals].mean().add_suffix('_mean')
        s = g[vals].sem().add_suffix('_sem')
        n = g.size().rename('n_recordings')
        t = pd.concat([n, m, s], axis=1).reset_index().rename(columns={grouping: 'group'})
        t.insert(0, 'grouping', grouping)
        out.append(t)
    frac_mean = pd.concat(out, ignore_index=True)
    frac_mean = frac_mean[['grouping', 'group', 'odor', 'n_recordings'] + [
        f'{v}_{x}' for v in vals for x in ('mean', 'sem')]]
    frac_mean = frac_mean.dropna(axis='columns', how='all')
    return dict(recording_mean=rec_mean, fly_mean=fly_mean, frac_mean=frac_mean)


def square(long_df: pd.DataFrame, col: str, odors: Sequence[str]) -> pd.DataFrame:
    """Symmetric odor x odor matrix (zero diagonal, NaN for pairs never recorded together)
    from a long pair table."""
    m = pd.DataFrame(np.nan, index=list(odors), columns=list(odors))
    for o in odors:
        m.loc[o, o] = 0.0
    for a, b, v in long_df[['odor_a', 'odor_b', col]].itertuples(index=False):
        m.loc[a, b] = m.loc[b, a] = v
    return m


def write_panel(panel: str, recs: List[Tuple[dict, pd.DataFrame]], out_dir: Path) -> dict:
    """Writes one panel's long CSV, odor key and computed/ files. Returns its tables."""
    pdir = out_dir / panel
    comp = pdir / 'computed'
    sq_dir = comp / 'rdm_square'
    sq_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.concat([to_long(m, w, panel) for m, w in recs], ignore_index=True)
    long_df.to_csv(pdir / f'{panel}_long.csv', index=False)
    print(f'{panel}: {len(long_df)} long rows')

    t = recording_tables(panel, recs)
    recs_df = t['recordings']
    key_rows = []
    for lab in PANEL_ODORS[panel]:
        o = yang2olfdb(lab)
        col = f'n_trials {o}'
        have = recs_df[recs_df[col] > 0] if col in recs_df else recs_df.iloc[:0]
        key_rows.append(dict(yang_label=lab, odor=o, kind=odor_kind(o),
            components=' ; '.join(yang2olfdb(c) for c in mix_components(lab))
                if '+' in lab else '',
            conc_stated=lab not in NO_CONC or lab == 'PFO',
            n_recordings=len(have),
            n_recordings_control=int((have.genotype_group == 'control').sum()),
            n_recordings_apl_output_blocked=int(
                (have.genotype_group == 'apl_output_blocked').sum()),
            trials_per_recording='/'.join(str(x) for x in sorted(set(have[col])))
                if len(have) else ''))
    odor_key = pd.DataFrame(key_rows)
    odor_key.to_csv(pdir / f'{panel}_odor_key.csv', index=False)

    recs_df.to_csv(comp / 'recordings.csv', index=False)
    t['trialmeans'].to_parquet(comp / 'trialmean_responses_long.parquet', index=False)
    t['trialmeans'].to_csv(comp / 'trialmean_responses_long.csv.gz', index=False)
    t['fracs'].to_csv(comp / 'response_fraction_per_recording.csv', index=False)
    t['pairs'].to_csv(comp / 'rdm_per_recording_long.csv', index=False)

    means = mean_tables(t['pairs'], t['fracs'])
    means['frac_mean'].to_csv(comp / 'response_fraction_recording_mean.csv', index=False)
    for how in ('recording_mean', 'fly_mean'):
        m = means[how]
        m.to_csv(comp / f'rdm_{how}_long.csv', index=False)
        for (group, oset), sub in m.groupby(['genotype_group', 'odor_set']):
            odors = [o for o in t['order'] if o in t['sets'][oset]
                and (o in set(sub.odor_a) or o in set(sub.odor_b))]
            for col in MEASURES:
                square(sub, col, odors).to_csv(
                    sq_dir / f'rdm_{how}_{group}_{oset}_{col}.csv')
    return dict(t, **means, odor_key=odor_key, n_long_rows=len(long_df))


def orn_measurements(cache_dir: Path, olfdb: str, checks: Checks) -> Tuple[pd.DataFrame,
    pd.DataFrame]:
    """Single-odor and air-mixture ORN measurements available as possible model input.

    Returns:
        (singles, mixtures): singles has one row per (source, panel, odor) with columns
        source_kind ('spike deltas' for the committed estimated ORN spike deltas, the
        model's input; 'olfdb dF/F' for olfdb's pebbled glomerulus datasets), source,
        panel, odor, abbrev, log10_conc; mixtures has the air mixtures (source_kind,
        source, panel, odor, components).
    """
    singles, mixes = [], []
    for name in ORN_DELTA_CSVS:
        with open(ORN_DELTA_DIR / name, newline='') as f:
            r = csv.reader(f)
            panels, odors = next(r), next(r)
        assert panels[0] == 'panel' and odors[0] == 'odor'
        for panel, odor in zip(panels[1:], odors[1:]):
            row = dict(source_kind='spike deltas', source=name, panel=panel, odor=odor)
            if odor in ORN_DELTA_AIR_MIXES:
                mixes.append(dict(row, components=ORN_DELTA_AIR_MIXES[odor]))
                continue
            abbrev, conc = split_odor(odor)
            if '+' in abbrev or 'mix' in abbrev:
                continue
            singles.append(dict(row, abbrev=abbrev, log10_conc=conc))
    for ds in OLFDB_ORN_DATASETS:
        meta = fetch_json(f'{olfdb}/api/datasets/{ds}',
            cache_dir / f'dataset_{ds}_metadata.json')
        for p in meta['summary']['panels']:
            for odor in p['odors']:
                row = dict(source_kind='olfdb dF/F', source=f'olfdb dataset {ds}',
                    panel=p['panel'], odor=odor)
                if ' + ' in odor:
                    mixes.append(dict(row, components=odor))
                    continue
                abbrev, conc = split_odor(odor)
                if '+' in abbrev or 'mix' in abbrev:
                    continue
                singles.append(dict(row, abbrev=abbrev, log10_conc=conc))
    singles = pd.DataFrame(singles)
    mixes = pd.DataFrame(mixes)
    olfdb_mixes = set(mixes[mixes.source_kind == 'olfdb dF/F'].odor)
    checks.section('ORN sources')
    checks.check('olfdb writes air mixtures of components at different concentrations as '
        '"<a> @ <c1> + <b> @ <c2>" (dataset 5), and the kiwi-control spike deltas\' air '
        'mixes are those', set(ORN_DELTA_AIR_MIXES.values()) <= olfdb_mixes,
        ', '.join(sorted(olfdb_mixes)))
    return singles, mixes


def closest(singles: pd.DataFrame, abbrev: str, conc: Optional[float]) -> dict:
    """Closest measurement of one odor in `singles` (ties: higher concentration first)."""
    s = singles[singles.abbrev == abbrev]
    if len(s) == 0:
        return dict(closest='', sources='', log10_distance=np.nan, status='none',
            available='')
    avail = ', '.join(f'{c:g}' for c in sorted(s.log10_conc.unique()))
    if conc is None:
        return dict(closest='', sources='', log10_distance=np.nan,
            status='unknown concentration', available=avail)
    d = (s.log10_conc - conc).abs()
    best = s.assign(d=d).sort_values(['d', 'log10_conc'], ascending=[True, False])
    c = best.log10_conc.iloc[0]
    dist = float(best.d.iloc[0])
    at = s[s.log10_conc == c]
    srcs = '; '.join(sorted(set(f'{x.source} ({x.panel})' for x in at.itertuples())))
    if dist < 1e-9:
        status = 'exact'
    elif dist <= ORN_SUBSTITUTE_MAX_LOG10 + 1e-9:
        status = 'substitute'
    else:
        status = 'none'
    return dict(closest=f'{abbrev} @ {c:g}', sources=srcs, log10_distance=round(dist, 3),
        status=status, available=avail)


def orn_availability(panel_keys: Dict[str, pd.DataFrame], singles: pd.DataFrame,
    mixes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per Yang odor / mixture component, and per mixture: the closest ORN measurements."""
    uses: Dict[str, dict] = {}
    for panel, key in panel_keys.items():
        for r in key.itertuples():
            if r.kind == 'control':
                comps, as_odor = [r.odor], True
            elif r.kind == 'air mixture':
                comps, as_odor = r.components.split(' ; '), False
            else:
                comps, as_odor = [r.odor], True
            for c in comps:
                u = uses.setdefault(c, dict(panels=set(), as_odor=set(), in_mixtures=[]))
                u['panels'].add(panel)
                if as_odor:
                    u['as_odor'].add(panel)
                else:
                    u['in_mixtures'].append(f'{r.odor} ({panel})')
    rows = []
    for odor, u in uses.items():
        abbrev, conc = split_odor(odor)
        row = dict(odor=odor, abbrev=abbrev, log10_conc=conc,
            panels=', '.join(sorted(u['panels'])),
            as_odor_in=', '.join(sorted(u['as_odor'])),
            in_mixtures='; '.join(u['in_mixtures']), kind=odor_kind(odor))
        for kind, prefix in (('spike deltas', 'spike_deltas'), ('olfdb dF/F', 'olfdb_dff')):
            c = closest(singles[singles.source_kind == kind], abbrev, conc)
            for k, v in c.items():
                row[f'{prefix}_{k}'] = v
        if odor in CONTROL_ODORS:
            row['status'] = 'control (no odor input)'
        else:
            st = [row['spike_deltas_status'], row['olfdb_dff_status']]
            for s in ('exact', 'substitute', 'unknown concentration', 'none'):
                if s in st:
                    row['status'] = s
                    break
        rows.append(row)
    avail = pd.DataFrame(rows)

    mix_rows = []
    by_odor = avail.set_index('odor')
    for panel, key in panel_keys.items():
        for r in key[key.kind == 'air mixture'].itertuples():
            comps = r.components.split(' ; ')
            row = dict(panel=panel, mixture=r.odor)
            for i, c in enumerate(comps, 1):
                row[f'component_{i}'] = c
                row[f'component_{i}_spike_deltas'] = (
                    f'{by_odor.loc[c, "spike_deltas_status"]}'
                    + (f' ({by_odor.loc[c, "spike_deltas_closest"]})'
                       if by_odor.loc[c, 'spike_deltas_closest'] else ''))
                row[f'component_{i}_any_source'] = (f'{by_odor.loc[c, "status"]}')
            ok = lambda col: all(by_odor.loc[c, col] in ('exact', 'substitute')
                for c in comps)
            row['both_components_spike_deltas'] = ok('spike_deltas_status')
            row['both_components_any_source'] = ok('status')
            # air mixtures of the same two odors measured in ORNs (at any concentrations)
            names = sorted(split_odor(c)[0] for c in comps)
            found = []
            for m in mixes.itertuples():
                mc = m.components.split(' + ')
                if sorted(split_odor(c)[0] for c in mc) != names:
                    continue
                cmap = {split_odor(c)[0]: split_odor(c)[1] for c in mc}
                dmax = max(abs(cmap[split_odor(c)[0]] - split_odor(c)[1]) for c in comps)
                label = m.odor if m.odor == m.components else f'{m.odor} = {m.components}'
                found.append(f'{label} [{m.source} ({m.panel}); max component distance '
                    f'{dmax:g}]')
            row['measured_orn_air_mixture_same_odors'] = '; '.join(found)
            mix_rows.append(row)
    return avail, pd.DataFrame(mix_rows)


def md_table(df: pd.DataFrame, floatfmt: str = '.3f') -> List[str]:
    """Markdown table lines from a DataFrame."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return '' if np.isnan(v) else format(v, floatfmt)
        return str(v)
    lines = ['| ' + ' | '.join(str(c) for c in df.columns) + ' |',
        '|' + '|'.join('---' for _ in df.columns) + '|']
    for row in df.itertuples(index=False):
        lines.append('| ' + ' | '.join(fmt(v) for v in row) + ' |')
    return lines


def panel_summary(panel: str, res: dict) -> dict:
    r = res['recordings']
    groups = r.groupby('genotype_group', sort=False)
    by_type = r.groupby('exp_type', sort=False).recording.size()
    return dict(
        panel=panel,
        recordings=len(r),
        flies=len(set(zip(r.date, r.fly_num))),
        KCs=int(r.n_kcs.sum()),
        dates=f'{r.date.min()} to {r.date.max()}',
        **{GROUP_LABEL[g]: f'{len(s)} rec., {len(set(zip(s.date, s.fly_num)))} flies, '
            f'{int(s.n_kcs.sum())} KCs' for g, s in groups},
        exp_types=', '.join(f'{k} {v}' for k, v in by_type.items()),
    )


def write_readme(out_dir: Path, results: dict, avail: pd.DataFrame,
    mix_avail: pd.DataFrame, checks: Checks, binary_only: List[str],
    dilute_rules: Dict[str, str]) -> None:
    """Writes README.md from the computed tables."""
    L: List[str] = []
    a = L.append
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    a("# Yang's diagnostic-odor mixture and dilute natural-mixture KC data\n")
    a(f'Made {now} by `al_analysis/scripts/claude_pair_remapping_experiments/'
        'yang_weak_odor_kc_data.py` from the files Yang sent Tom in April 2026 '
        '(`al_analysis/data/from_yang/20260415_YZtransfer/`, committed in al_analysis). '
        'Re-run that script to regenerate everything here (it overwrites this folder). '
        'The layout follows `data/yang_apl16_kc_olfdb/` (Yang\'s APL16 TNT / TNT3c data '
        'from olfdb), so the same code can read both; these two datasets are not in olfdb.'
        '\n')
    a('Both panels are MB247 > GCaMP KC imaging (as the APL16 data), one recording = one '
        'hemisphere (`hemibrain` L / R) of one fly (`date`, `fly_num`), KC ids (`roi`) '
        'numbered within a recording. Every recording Yang included in her per-trial files '
        'is kept, whatever its genotype, and both hemispheres of a fly are separate '
        'recordings (as in the APL16 export). Tom\'s loader (`load_yang_kc_data` in '
        '`scripts/model_yang_mixtures.py`) instead drops TNT_label, keeps one recording '
        'per fly and drops odors that are not mixture components.\n')
    a('## Panels\n')
    summ = pd.DataFrame([panel_summary(p, r) for p, r in results.items()]).fillna('-')
    L += md_table(summ)
    a('')
    dr = results['diag_mix']['recordings']
    first = dr['n_trials farn @ -4'] > 0
    second = dr['n_trials farn @ -2.5'] > 0
    assert not (first & second).any() and (first | second).all()
    blocked = dr.genotype_group == 'apl_output_blocked'
    assert not (blocked & second).any()
    a(f'- `diag_mix`: diagnostic-odor binary mixtures, recorded {dr.date.min()} to '
        f'{dr.date.max()} (Tom\'s loader calls it "diag-binaries"). 6 trials per odor '
        '(pfo 3), except 20250321_3FRfun1 (4, pfo 2). It is two sub-panels: recordings to '
        f'{dr[first].date.max()} have 2h @ -8 / -7, ma @ -8 / -7, farn @ -4 / -3 and the '
        f'three mixtures with farn @ -3; recordings from {dr[second].date.min()} drop the '
        '-8 / -4 concentrations and add farn @ -2.5 with its two mixtures. All '
        f'{blocked.sum()} APL output blocked (TNT) recordings '
        f'({len(set(zip(dr[blocked].date, dr[blocked].fly_num)))} flies) are in the first '
        'sub-panel, so the blocked vs control comparison exists only for its odors '
        f'({(first & ~blocked).sum()} control recordings there, {(second & ~blocked).sum()} '
        'in the second).')
    a('- `natmix_dilute`: the top-2 components of natural mixtures at low concentrations, '
        'all WT (Tom\'s loader: "natmix-top2-dilute"), 2026-01-16 to 02-10. 2-4 trials per '
        'odor (one recording 2, two 4, the rest 3). Also two sub-panels: recordings to '
        '2026-01-21 have 2h @ -5 + 1o3ol @ -5 and ea / eb only at -5; recordings from '
        '2026-02-02 have 2h @ -7 + 1o3ol @ -5 (no 2h @ -5) and ea / eb at -7 and -5. Two '
        'recordings (2026-02-03) add IaA @ -2, banana and Bmix28; one (2026-01-16) has air. '
        '20260121_2FL has two functional recordings of the same hemisphere (`fun1`, 4 '
        'trials; `fun2`, 3 trials), kept as two recordings (`recording_num`); fly means '
        'average them with the fly\'s other recordings.\n')

    a('## Odors and their names\n')
    a('Odor strings are written as olfdb writes them, `<abbrev> @ <log10 conc>`, with '
        'olfdb\'s abbreviations (its odors table, `/api/odors`: 2h, ma, farn, ea, eb, '
        '1o3ol, IaA, pfo, ban all exist there). Yang\'s mixtures (`2h(-7)+farn(-3)`) are '
        'written `2h @ -7 + farn @ -3`, components in Yang\'s order: that is how olfdb '
        'writes air mixtures (two vials mixed in the air stream) of components at different '
        'concentrations (dataset 5: `ea @ -4.2 + eb @ -3.5`, `1o3ol @ -3 + 2h @ -5`). '
        'That assumes Yang\'s mixtures are air mixtures, which her labels (one '
        'concentration per component) suggest but nothing states; olfdb would write an '
        'in-vial mixture as an odor of its own (e.g. `ea+eb @ 0`). Names that are not '
        'olfdb\'s: Yang\'s `Iaa` is olfdb\'s `IaA`; `PFO` is `pfo @ 0`; `banana` is written '
        '`ban` (olfdb\'s abbreviation for banana) without a concentration, since Yang\'s '
        'label gives none (olfdb has Sam\'s ORN ramp of ban at -4 to 0, which may or may not '
        'be the same stimulus); `Bmix28` (not in olfdb; from the name, possibly a mixture '
        'of 28 banana components, composition and concentration unknown) and `air` (not '
        'in olfdb) are '
        'kept as Yang wrote them. These three have no ` @ ` and `log10_conc` is empty for '
        'them. `<panel>_odor_key.csv` maps every label of Yang\'s to its string here.\n')
    for panel, res in results.items():
        a(f'`{panel}`:\n')
        k = res['odor_key'].drop(columns=['components', 'conc_stated'])
        k = k.rename(columns={'n_recordings_control': 'n rec. control flies',
            'n_recordings_apl_output_blocked': 'n rec. APL output blocked (TNT)',
            'n_recordings': 'n rec.'})
        L += md_table(k)
        a('')

    a('## Genotype groups\n')
    a('`exp_type` is Yang\'s label, per recording. `genotype_group` pools it into the '
        'two conditions of the APL16 data: `control` = WT, TNTin_label, TNTin_nolabel '
        '(and TNT_nolabel, absent here); `apl_output_blocked` = TNT_label. Yang\'s TNT '
        'design (notes/yang-summary.md, from her slides): Flp-out UAS-TeTx (TNT, active) '
        'or UAS-TeTxin (TNTin, inactive) in APL, with Flp stochastic, so each hemisphere\'s '
        'APL either expresses the toxin ("label") or not ("nolabel"). Only an active-toxin, '
        'labelled hemisphere has APL output blocked; TNTin hemispheres (labelled or not) '
        'and WT keep APL output, and in the APL16 data Yang pooled TNTin_label, '
        'TNTin_nolabel and TNT_nolabel as the "TNT3c" control. The label is per '
        'hemisphere: fly 2025-03-21 #3 is TNTin_label on the left and TNTin_nolabel on the '
        'right. WT is presumably the GCaMP genotype without the TeTx transgenes (not stated '
        'in the files). `recordings.csv` also has `genotype` (WT / TNTin / TNT).\n')

    a('## Files\n')
    a('- `README.md` (this file), `provenance.json` (script, commit, input file hashes, '
        'thresholds, odor names), `checks.txt` (the built-in checks, below)')
    a('- `orn_input_availability.csv`, `orn_input_availability_mixtures.csv`: which ORN '
        'measurements could stand in as model input (below)')
    a('- `olfdb_metadata/`: olfdb\'s odors table and the metadata of the ORN datasets, as '
        'fetched (used for the abbreviations and the ORN availability)')
    a('- `<panel>/<panel>_long.csv`: per-trial responses, olfdb long columns (date, '
        'fly_num, hemibrain, roi, panel, odor, repeat, value) plus `recording` (Yang\'s '
        'recording id, e.g. 20250129_1FLfun1 = date, fly 1, female, left hemisphere, '
        'functional recording 1), `recording_num`, `sex` (F / M, from the id), '
        '`exp_type`, `genotype_group`. Values as Yang wrote them. Odor-trials a recording '
        'does not have are absent (no NaN rows). Use `recording` (not date, fly_num, '
        'hemibrain) to identify a recording: 20260121_2FL has two.')
    a('- `<panel>/<panel>_odor_key.csv`: Yang\'s label -> odor string, kind (monomolecular, '
        'air mixture, complex, control), components, recordings and trials per recording')
    a('- `<panel>/computed/`, as for the APL16 data:')
    a('  - `recordings.csv`: recording, exp_type, genotype_group, date, fly_num, hemibrain, '
        'recording_num, sex, genotype, n_kcs, n_odors and `n_trials <odor>` per odor (0: '
        'not presented)')
    a('  - `trialmean_responses_long.{parquet,csv.gz}`: trial-mean response of each KC to '
        'each odor (recording columns, roi, odor, kind, log10_conc (empty for mixtures and '
        'odors without one), n_trials, value)')
    a('  - `response_fraction_per_recording.csv`: per recording and odor, the number and '
        'fraction of KCs responding by the program\'s definition (`trialmean_gt1`: trial '
        'mean > 1, as for the APL16 data) and by Yang\'s (`yang`: > 1 on at least 50 % of '
        'the trials, her `respthresh1_trialfr0.5`); for mixtures, the fraction responding '
        'to either component (`frac_union_components_*`, Yang\'s `_predicted`); and the '
        'fraction of KCs responding to any non-control odor of that recording (depends on '
        'the recording\'s odor set). `response_fraction_recording_mean.csv`: means and SEM '
        'over recordings by `genotype_group` and by `exp_type` (`grouping` column).')
    a('  - `rdm_per_recording_long.csv`: per recording, each odor pair\'s 1 - Pearson r and '
        '1 - cosine similarity between the two odors\' trial-mean response vectors over all '
        'KCs of the recording. `odor_set`: `pure_no_pfo` (monomolecular odors), '
        '`pure_with_pfo` (plus pfo and air), `all_no_pfo` (plus air mixtures, ban, '
        'Bmix28), `all_with_pfo` (everything). A pair\'s values do not depend on the set; '
        'the sets only filter pairs.')
    a('  - `rdm_recording_mean_long.csv`, `rdm_fly_mean_long.csv`: averaged over recordings '
        '(each weight 1) or over flies (a fly\'s recordings averaged first), per '
        'genotype_group, over the recordings that have both odors (`n_recordings`, '
        '`n_flies`)')
    a('  - `rdm_square/rdm_<recording_mean|fly_mean>_<genotype_group>_<odor_set>_<measure>'
        '.csv`: those as square odor x odor matrices, NaN for pairs no recording of the '
        'group had together (the two sub-panels of each panel share only some odors)\n')

    a('## Units\n')
    a('As the APL16 data: neuropil-corrected suite2p fluorescence (`Fc`) of curated ROIs, '
        'z-scored (file suffix `..._s2p_mean_curated_Fc_zscored_full_cell_resp_df...`), '
        'one response value per KC and trial: each KC\'s Fc (F - 0.7 Fneu) is z-scored over the whole '
        'recording, and a trial\'s response is the mean z 2-8 s after odor onset minus the mean '
        'over the 5 s before (Yang, 2026-11-10; add_to_reports/yang_response_definition.md, checked '
        'on the APL16 recordings). Absolute z-score levels are not '
        'comparable to dF/F (or to the ORN dF/F data), and depend on each KC\'s noise; the '
        '> 1 response thresholds are Yang\'s choice, in these units.\n')

    a('## What differs from the APL16 export\n')
    a('- Not from olfdb: read from Yang\'s files in the repository, so no '
        '`dataset_<id>_*` downloads; the long CSV has the extra recording and genotype '
        'columns.')
    a('- 6 trials per odor in `diag_mix` (3 for pfo; one recording 4 / 2), 2-4 in '
        '`natmix_dilute`, instead of 3. Trial means and the program\'s > 1 threshold on '
        'them average over more (or fewer) trials, so their noise differs from the APL16 '
        'data; Yang\'s at-least-half-the-trials definition is also given.')
    a('- Both genotype groups are in one file (`diag_mix`), per recording, instead of one '
        'olfdb dataset per genotype; the group is a column (`genotype_group`), and the '
        'square matrices are per group.')
    a('- Recordings differ in their odor sets (sub-panels), so recording and fly means are '
        'over the recordings that have each pair, and square matrices can have NaN.')
    a('- No comparison to the report\'s matrices (they are APL16 only). Square matrices are '
        'in `computed/rdm_square/` instead of `computed/`.\n')

    a('## Checks\n')
    a('`checks.txt` lists every check (the script fails if any does). In short:\n')
    for line in checks.lines:
        if line.startswith(('PASS', 'FAIL', 'NOTE')):
            a(f'- {line}')
    a('')

    a('## Response fractions\n')
    a('Fraction of KCs responding, mean over recordings (n recordings), by the program\'s '
        'definition (trial mean > 1) and Yang\'s (> 1 on at least half the trials). Control '
        'means pool both sub-panels where an odor is in both.\n')
    for panel, res in results.items():
        fm = res['frac_mean']
        fm = fm[fm.grouping == 'genotype_group']
        groups = [g for g in ('control', 'apl_output_blocked') if g in set(fm.group)]
        rows = []
        for o in res['order']:
            row = {'odor': o}
            for g in groups:
                s = fm[(fm.group == g) & (fm.odor == o)]
                lab = GROUP_LABEL[g]
                if len(s):
                    s = s.iloc[0]
                    row[f'{lab}: n'] = int(s.n_recordings)
                    row[f'{lab}: trial mean > 1'] = s.frac_trialmean_gt1_mean
                    row[f'{lab}: Yang'] = s.frac_yang_mean
                    if ' + ' in o:
                        row[f'{lab}: Yang, either component'] = \
                            s.frac_union_components_yang_mean
                else:
                    row[f'{lab}: n'] = 0
            rows.append(row)
        t = pd.DataFrame(rows)
        cols = ['odor'] + [c for g in groups for c in t.columns
            if c.startswith(GROUP_LABEL[g] + ':')]
        a(f'`{panel}`:\n')
        L += md_table(t[cols])
        a('')

    a('Recordings in which a mixture has fewer responders (Yang\'s definition) than each of '
        'its components alone:\n')
    rows = []
    for panel, res in results.items():
        fr = res['fracs']
        for (mix, group), sub in fr[fr.kind == 'air mixture'].groupby(['odor',
            'genotype_group'], sort=False):
            n_below = 0
            for rec in sub.recording:
                s = fr[fr.recording == rec].set_index('odor').frac_yang
                n_below += int(s[mix] < min(s[c] for c in mix.split(' + ')))
            rows.append({'panel': panel, 'mixture': mix, 'group': GROUP_LABEL[group],
                'recordings below both components': f'{n_below} of {len(sub)}'})
    L += md_table(pd.DataFrame(rows))
    a('')

    a('## ORN input availability\n')
    a('The model\'s input is ORN spike deltas. Two kinds of ORN data could supply them: '
        'the estimated spike deltas already committed in al_analysis '
        '(`al_analysis/data/internal/{kiwi-control,megamat,validation2}_mean_est_spike_'
        'deltas.csv`, Tom\'s pebbled dF/F converted by the dF/F-to-spike-rate fit; '
        '`mb_model.natmix_orn_deltas()` loads the first), and olfdb\'s pebbled glomerulus '
        'dF/F datasets 1, 4, 5 (Tom; the same flies as those three files), 6 (George) and '
        '21 (Sam, with concentration ramps), which would first need that conversion. For '
        'each odor and mixture component, the closest measured concentration of the same '
        'odor in each, the log10 distance, and the status under Tom\'s rule (ORN data within '
        '1 log10 unit, inclusive, may stand in): exact, substitute, none. '
        '`orn_input_availability.csv` has the sources and all measured concentrations.\n')
    t = avail[avail.kind != 'control'].copy()
    t['used as'] = [', '.join(x for x in [
        ('odor in ' + r.as_odor_in) if r.as_odor_in else '',
        (f'component of {len(r.in_mixtures.split("; "))} mixture(s)')
            if r.in_mixtures else ''] if x) for r in t.itertuples()]
    t['spike deltas'] = [f'{r.spike_deltas_status}' + (f': {r.spike_deltas_closest} '
        f'(distance {r.spike_deltas_log10_distance:g})' if r.spike_deltas_closest else '')
        for r in t.itertuples()]
    t['olfdb dF/F'] = [f'{r.olfdb_dff_status}' + (f': {r.olfdb_dff_closest} '
        f'(distance {r.olfdb_dff_log10_distance:g}; '
        f'{r.olfdb_dff_sources.replace("olfdb dataset ", "ds ")})'
        if r.olfdb_dff_closest else (f' (measured at {r.olfdb_dff_available})'
        if r.olfdb_dff_available else '')) for r in t.itertuples()]
    L += md_table(t[['odor', 'used as', 'spike deltas', 'olfdb dF/F', 'status']])
    a('')
    for s in ('exact', 'substitute', 'unknown concentration', 'none'):
        odors = list(t[t.status == s].odor)
        sd = list(t[(t.status == s) & (t.spike_deltas_status == s)].odor) if s in (
            'exact', 'substitute') else []
        a(f'- {s}: {", ".join(odors) if odors else "-"}' + (
            f' (in the spike deltas: {", ".join(sd) if sd else "none"})'
            if s in ('exact', 'substitute') else ''))
    a('- pfo and air are controls (no odor input).\n')
    a('Mixtures (both components covered = each exact or a substitute; how a mixture '
        'becomes model input is not decided here):\n')
    mt = mix_avail.copy()
    mt['components'] = mt.component_1 + ' ; ' + mt.component_2
    mt['both, spike deltas'] = mt.both_components_spike_deltas.map({True: 'yes',
        False: 'no'})
    mt['both, any source'] = mt.both_components_any_source.map({True: 'yes', False: 'no'})
    mt['same two odors measured as an ORN air mixture'] = \
        mt.measured_orn_air_mixture_same_odors.replace('', '-')
    L += md_table(mt[['panel', 'mixture', 'both, spike deltas', 'both, any source',
        'same two odors measured as an ORN air mixture']])
    a('')

    a('## Ambiguities and notes\n')
    a('- Sex: every recording id has F except 20250214_3MLfun1 (`diag_mix`, '
        'TNTin_label, control), a male; kept, with `sex` = M.')
    a('- farn: `diag_mix` has farn @ -4 and -3 (first sub-panel) and -3 and -2.5 (second); '
        'every recording has -3. The mixtures with farn use -3 or -2.5 (second sub-panel '
        'both).')
    a('- Missing trials: 20250321_3FRfun1 (`diag_mix`, TNTin_nolabel) has 4 trials per odor '
        'and 2 of pfo instead of 6 and 3; `natmix_dilute` has 2-4 trials (20260116_1FLfun1 '
        'has 2). No KC is missing a trial its recording has.')
    a(f'- Binary-only recordings: Yang\'s folder has binarized files for '
        f'{len(binary_only)} `diag_mix`-panel recordings that are not in her "good" '
        f'per-trial file or in her response-rate summary ({", ".join(binary_only)}); '
        'presumably excluded by her, and not exported here (no per-trial data).')
    odd = {k: v for k, v in dilute_rules.items() if v != '>= 50 % of trials'}
    if odd:
        a('- Yang\'s response rate file for ' + ', '.join(odd) + ' (2 trials) counts a '
            'KC only if it is > 1 on both trials, while her binary file for it (file dated '
            'a week later) and every other file count at least half the trials (1 of 2). The '
            'outputs here use at least half throughout.')
    a('- Extra odors in `natmix_dilute`: IaA @ -2, ban and Bmix28 (2026-02-03, both '
        'hemispheres) and air (2026-01-16); kept (Tom\'s loader drops them).')
    a('- The mixtures\' delivery (air mixture of two vials vs one vial) and the nature of '
        'banana and Bmix28 are not stated in Yang\'s files; worth asking her.')
    (out_dir / 'README.md').write_text('\n'.join(L) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--shared-dir', type=Path, default=None,
        help='the shared directory (default: as kc_remapping_harness.shared_dir() finds '
        'it: $KC_REMAPPING_SHARED, /local/tensor/..., /mnt/tensor/...)')
    ap.add_argument('--olfdb', default=DEFAULT_OLFDB)
    args = ap.parse_args()
    shared = args.shared_dir or default_shared_dir()
    out_dir = shared / 'data' / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / 'olfdb_metadata'
    checks = Checks()

    checks.section('olfdb names')
    odors_tab = fetch_json(f'{args.olfdb}/api/odors', cache / 'odors.json')['odors']
    olfdb_abbrevs = {o['abbrev'] for o in odors_tab}
    missing = [x for x in OLFDB_ABBREVS_USED if x not in olfdb_abbrevs]
    checks.check('every abbreviation used is in olfdb\'s odors table', not missing,
        f'{", ".join(OLFDB_ABBREVS_USED)}' + (f'; missing {missing}' if missing else ''))

    panels = {'diag_mix': load_diag_mix(RAW_DIR), 'natmix_dilute': load_natmix_dilute(
        RAW_DIR)}
    for panel, recs in panels.items():
        seen = {lab for _, w in recs for lab in w.columns.get_level_values('yang_label')}
        checks.check(f'{panel}: odors are the expected ones',
            seen == set(PANEL_ODORS[panel]), ', '.join(PANEL_ODORS[panel]))
        for meta, _ in recs:
            meta['genotype_group'] = EXP_TYPE2GROUP[meta['exp_type']]
        flies = {}
        for meta, _ in recs:
            flies.setdefault((meta['date'], meta['fly_num']), set()).add(
                (meta['sex'], EXP_TYPE2GENOTYPE[meta['exp_type']]))
        checks.check(f'{panel}: one sex and one genotype (WT / TNTin / TNT) per fly',
            all(len(v) == 1 for v in flies.values()), f'{len(flies)} flies')
        keys = [(m['date'], m['fly_num'], m['hemibrain'], m['recording_num'])
            for m, _ in recs]
        checks.check(f'{panel}: recording ids unique', len(set(keys)) == len(keys))
    binary_only = check_diag_yang_files(panels['diag_mix'], RAW_DIR, checks)
    dilute_rules = check_dilute_yang_files(panels['natmix_dilute'], RAW_DIR, checks)

    results = {}
    for panel, recs in panels.items():
        results[panel] = write_panel(panel, recs, out_dir)

    singles, mixes = orn_measurements(cache, args.olfdb, checks)
    avail, mix_avail = orn_availability({p: r['odor_key'] for p, r in results.items()},
        singles, mixes)
    avail.to_csv(out_dir / 'orn_input_availability.csv', index=False)
    mix_avail.to_csv(out_dir / 'orn_input_availability_mixtures.csv', index=False)

    (out_dir / 'checks.txt').write_text(
        f'Checks run by yang_weak_odor_kc_data.py, {datetime.now():%Y-%m-%d %H:%M}\n'
        + '\n'.join(checks.lines) + '\n')
    write_readme(out_dir, results, avail, mix_avail, checks, binary_only, dilute_rules)

    git = lambda *a: subprocess.run(['git', *a], cwd=HERE, capture_output=True,
        text=True).stdout.strip()
    inputs = sorted([RAW_DIR / DIAG_CSV, RAW_DIR / DIAG_RESPRATE_CSV]
        + list(RAW_DIR.glob(f'*{DIAG_BINARY_SUFFIX}'))
        + [p for p in (RAW_DIR / DILUTE_SUBDIR).glob('*.csv')]
        + [ORN_DELTA_DIR / x for x in ORN_DELTA_CSVS])
    meta = {
        'generated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'script': str(Path(__file__).resolve().relative_to(HERE.parents[2])),
        'al_analysis_commit': git('rev-parse', 'HEAD'),
        'script_uncommitted_changes': bool(git('status', '--porcelain', '--',
            Path(__file__).name)),
        'olfdb': args.olfdb,
        'olfdb_orn_datasets': list(OLFDB_ORN_DATASETS),
        'response_thresh': RESPONSE_THRESH,
        'yang_trial_frac': YANG_TRIAL_FRAC,
        'orn_substitute_max_log10': ORN_SUBSTITUTE_MAX_LOG10,
        'exp_type_to_genotype_group': EXP_TYPE2GROUP,
        'panels': {p: {'odors': {lab: yang2olfdb(lab) for lab in PANEL_ODORS[p]},
            'n_recordings': len(panels[p]), 'n_long_rows': results[p]['n_long_rows']}
            for p in panels},
        'inputs': {str(p.relative_to(REPO)): sha256(p) for p in inputs},
        'checks_failed': checks.failed,
    }
    (out_dir / 'provenance.json').write_text(json.dumps(meta, indent=1) + '\n')

    if checks.failed:
        raise AssertionError('checks failed:\n' + '\n'.join(checks.failed))
    print(f'wrote {out_dir}')


if __name__ == '__main__':
    main()
