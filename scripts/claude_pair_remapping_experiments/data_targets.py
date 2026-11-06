#!/usr/bin/env python3
"""The data-target registry: every KC- and ORN-data value the reports quote, computed in one
place with its definition and inputs.

Writes results/data_targets/data_targets.json in the shared directory: {name: {value, ci (optional
[lo, hi]), definition, function, inputs ({path: sha256}), commit, computed_at}}. Report templates
reference entries as `{{data:<name>|spec}}` (report_values.py); figure scripts and loss functions
should read their reference values from here instead of literals.
Usage: ./venv/bin/python data_targets.py
"""
from __future__ import annotations


def build() -> dict:
    """Computes every registry entry; returns {name: entry}."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == '__main__':
    main()
