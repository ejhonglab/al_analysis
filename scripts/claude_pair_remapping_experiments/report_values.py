#!/usr/bin/env python3
"""Resolve report placeholders to values with recorded provenance.

A placeholder in report markdown is `{{key ~ spec}}` (spec: a Python format spec such as `.2f`
or `+.3f`; negative numbers are written with a Unicode minus; `|` is not used because it
separates table cells). Keys:

- `data:<name>` or `data:<name>.lo` / `.hi`: an entry of the data-target registry
  (results/data_targets/data_targets.json in the shared directory, written by
  data_targets.py: value, optional 90 % interval, definition, function, inputs, commit).
- `<EXP>[/<table.csv>]:<row>:<column>`: one cell of an experiment's results table
  (results/<EXP>_*/<table>.csv, or results/<EXP>/ when <EXP> names a folder; without a table
  name every csv with a `tag` or `variant` column is searched, and a key matching more than one
  table is an error). `<row>` is a tag (matched against `tag`, else `variant`), or
  `col=value[;col=value]` to select by other columns. `<column>` may list alternatives
  separated by commas; the first one the table has is used.
- `lit:<text> # <source>`: a hand-entered value, allowed only with its source after `#`.

Every resolution returns the value and a source record (file, row, column, sha256 of the
file, the al_analysis commit recorded in the folder's provenance if any), which
fill_report.py writes to the page's sidecar.
"""
from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import kc_remapping_harness as H

RESULTS = H.shared_dir() / 'results'
DATA_TARGETS = RESULTS / 'data_targets' / 'data_targets.json'


@dataclass
class Resolved:
    key: str
    value: Any
    source: Dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=None)
def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def _registry() -> Dict[str, Any]:
    return json.loads(DATA_TARGETS.read_text())


@lru_cache(maxsize=None)
def _table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _folders(exp: str) -> List[Path]:
    if (RESULTS / exp).is_dir():
        return [RESULTS / exp]
    return [Path(p) for p in sorted(glob.glob(str(RESULTS / f'{exp}_*'))) if Path(p).is_dir()]


@lru_cache(maxsize=None)
def _commit(folder: str) -> Optional[str]:
    for p in sorted(Path(folder).glob('provenance*.json')):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for k in ('al_analysis_commit', 'commit'):
            if isinstance(d.get(k), str):
                return d[k]
        mp = d.get('model_provenance') or {}
        if isinstance(mp, dict):
            c = (mp.get('al_analysis') or {}).get('commit') if isinstance(mp.get('al_analysis'), dict) else None
            if c:
                return c
    return None


def _select(df: pd.DataFrame, row: str) -> pd.DataFrame:
    if '=' in row:
        m = pd.Series(True, index=df.index)
        for cond in row.split(';'):
            col, val = cond.split('=', 1)
            col, val = col.strip(), val.strip()
            if col not in df.columns:
                return df.iloc[0:0]
            m &= df[col].astype(str) == val
        return df[m]
    for col in ('tag', 'variant'):
        if col in df.columns:
            return df[df[col].astype(str) == row]
    return df[df.iloc[:, 0].astype(str) == row]


def _resolve_table(key: str) -> Resolved:
    head, row, cols = key.split(':', 2)
    exp, _, table = head.partition('/')
    folders = _folders(exp)
    if not folders:
        raise KeyError(f'{key}: no results folder for {exp!r}')
    paths = [f / table for f in folders] if table else [Path(p) for f in folders for p in sorted(glob.glob(str(f / '*.csv')))]
    candidates = [c.strip() for c in cols.split(',')]
    hits = []
    for p in paths:
        if not p.exists():
            continue
        df = _table(str(p))
        if not table and not ({'tag', 'variant'} & set(df.columns)) and '=' not in row:
            continue
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            continue
        sel = _select(df, row)
        if len(sel) == 0:
            continue
        if len(sel) > 1:
            raise KeyError(f'{key}: {len(sel)} rows match in {p}')
        hits.append((p, col, sel.iloc[0][col]))
    if not hits:
        raise KeyError(f'{key}: no table row / column matches')
    if len(hits) > 1:
        vals = {round(float(v), 9) if isinstance(v, (int, float, np.floating)) else v for _, _, v in hits}
        if len(vals) > 1:
            raise KeyError(f'{key}: matches in {len(hits)} tables with different values: {[str(h[0]) for h in hits]}')
    p, col, v = hits[0]
    v = v.item() if hasattr(v, 'item') else v
    return Resolved(key, v, dict(kind='table', file=str(p.relative_to(H.shared_dir())), row=row, column=col,
        sha256=_sha256(str(p)), commit=_commit(str(p.parent))))


def resolve(key: str) -> Resolved:
    """The value of a placeholder key, with its source."""
    key = key.strip()
    if key.startswith('lit:'):
        text, sep, src = key[4:].partition('#')
        if not sep or not src.strip():
            raise KeyError(f'{key}: a literal needs its source after #')
        return Resolved(key, text.strip(), dict(kind='literal', source=src.strip()))
    if key.startswith('data:'):
        name = key[5:]
        part = None
        if name.endswith('.lo') or name.endswith('.hi'):
            name, part = name[:-3], name[-2:]
        reg = _registry()
        if name not in reg:
            raise KeyError(f'{key}: not in the data-target registry')
        e = reg[name]
        v = e['value'] if part is None else e['ci'][0 if part == 'lo' else 1]
        return Resolved(key, v, dict(kind='data', name=name, part=part, definition=e.get('definition'),
            function=e.get('function'), commit=e.get('commit'), file=str(DATA_TARGETS.relative_to(H.shared_dir())),
            sha256=_sha256(str(DATA_TARGETS))))
    return _resolve_table(key)


def format_value(value: Any, spec: Optional[str]) -> str:
    """`value` formatted with `spec` (Python format spec), Unicode minus for negatives; a value
    that rounds to zero carries no minus sign; strings (literals) without a spec are returned as
    written."""
    if isinstance(value, str):
        if not spec:
            return value
        try:
            value = float(value)
        except ValueError:
            return value
    if spec:
        s = format(float(value) if not spec.endswith('d') else int(round(float(value))), spec)
    else:
        s = str(value)
    if s.startswith('-') and float(s) == 0:
        s = s[1:]
    return s.replace('-', '−')


def tooltip(res: Resolved) -> str:
    """One-line description of a value's source, for the page's hover text."""
    s = res.source
    if s.get('kind') == 'literal':
        return f"hand-entered: {s['source']}"
    if s.get('kind') == 'data':
        part = f" ({'lower' if s['part'] == 'lo' else 'upper'} end of the 90 % interval)" if s.get('part') else ''
        return f"KC / ORN data: {s['name']}{part}: {s.get('definition') or ''} [data_targets.py {str(s.get('commit') or '')[:7]}]"
    return f"{s['file']}, row {s['row']}, column {s['column']}" + (f" [commit {s['commit'][:7]}]" if s.get('commit') else '')
