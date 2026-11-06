#!/usr/bin/env python3
"""Resolve report placeholders to values with recorded provenance.

A placeholder in report markdown is `{{key|spec}}` (spec: a Python format spec such as `.2f`
or `+.3f`; negative numbers are written with a Unicode minus). Keys:

- `data:<name>` or `data:<name>.lo` / `.hi`: an entry of the data-target registry
  (results/data_targets/data_targets.json in the shared directory, written by
  data_targets.py: value, optional 90 % interval, definition, function, inputs, commit).
- `<EXP>[/<table.csv>]:<row>:<column>`: one cell of an experiment's results table
  (results/<EXP>_*/<table>.csv; without a table name every csv with a `tag` or `variant`
  column is searched, and a key matching more than one table is an error). `<row>` is a
  tag, or `col=value[;col=value]` to select by other columns.
- `lit:<text> # <source>`: a hand-entered value, allowed only with its source after `#`.

Every resolution returns the value and a source record (file, row, column, sha256 of the
file, the commit that wrote it if the folder's provenance records one), which fill_report.py
writes to the page's sidecar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Resolved:
    key: str
    value: Any
    source: Dict[str, Any] = field(default_factory=dict)


def resolve(key: str) -> Resolved:
    """The value of a placeholder key, with its source."""
    raise NotImplementedError


def format_value(value: Any, spec: Optional[str]) -> str:
    """`value` formatted with `spec` (Python format spec), Unicode minus for negatives."""
    raise NotImplementedError


def tooltip(res: Resolved) -> str:
    """One-line description of a value's source, for the page's hover text."""
    raise NotImplementedError
