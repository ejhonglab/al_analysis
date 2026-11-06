#!/usr/bin/env python3
"""Fill a report template's `{{key|spec}}` placeholders (report_values.py), write the filled
markdown and a sidecar of every value's source, and optionally build the HTML page.

- fill: template -> filled markdown (+ `<out>.values.json` sidecar); with tooltips, each value is
  written as `[[pv:value|source]]`, which md_to_artifact.py renders as a span with hover text.
- --check SIDECAR: re-resolves every key in a sidecar and lists values whose sources changed.
- --compare CURRENT_MD: compares the filled markdown (tooltips stripped) with another markdown
  line by line and lists every differing value with its row, column, key and source.
- lint: in "KC data" rows and the best-models table, a number that is not a placeholder is an
  error (`{{lit:... # source}}` is the escape).
Usage: ./venv/bin/python fill_report.py reports/summary_template.md OUT.md [--html OUT.html]
       [--no-tooltips] [--compare CURRENT.md] | --check OUT.values.json
"""
from __future__ import annotations


def fill(template: str, *, tooltips: bool = True):
    """Returns (filled markdown, sidecar list of {key, spec, text, value, source})."""
    raise NotImplementedError


def lint(template: str) -> list:
    """Bare numbers in KC-data rows and the best-models table (list of messages)."""
    raise NotImplementedError


def compare(filled: str, current: str) -> list:
    """Differences between two markdown versions of a report, value by value."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == '__main__':
    main()
