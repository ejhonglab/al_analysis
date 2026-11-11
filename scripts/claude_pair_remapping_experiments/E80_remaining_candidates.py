#!/usr/bin/env python3
"""E80: the flies' pulse in the model and the remaining candidates for the common error.

The entry is in experiment_priorities.md (planned by a Fable subagent, claude-fable-5-2, spawned from viki 2026-11-10
23:41). Part A (the pulse) is in E80_pulse.py, parts B and C (the screen, the gate) in E80_screen.py; this script has part D (no simulations: the
response amplitude against a KC's breadth in the flies, in baseline-noise units, and in the archived fits of the best
families; the breadth-weighted reading of the fits against the effect in baseline-noise units and in Yang's z; each KC
type alone with recording-bootstrap intervals) and the tables. Only shared data are read.

    python E80_remaining_candidates.py --part {D,A,B,C,tables}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E80_remaining_candidates'


def amplitude_by_breadth(out: Path) -> pd.DataFrame:
    """Part D: per odors-per-KC class (1, 2-3, 4-7, 8 or more), the responders' mean amplitude and share of the summed
    squared response, in the flies (baseline-noise units) and in every archived fit of the best families (spike counts),
    with APL on and off. Writes amplitude_by_breadth.csv."""
    raise NotImplementedError


def breadth_weighted(out: Path) -> pd.DataFrame:
    """Part D: the fits' counts weighted by (odors per KC) ** p, the weight set on the control flies' curve, scored
    against the APL effect in baseline-noise units and in Yang's z. Writes breadth_weighted.csv."""
    raise NotImplementedError


def type_split(out: Path) -> pd.DataFrame:
    """Part D: each KC type alone (alpha'beta', alphabeta, gamma) for the eight wirings' mean, under the readout, with
    recording-bootstrap intervals. Writes type_split.csv."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['D', 'A', 'B', 'C', 'tables'])
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'D':
        for f in (amplitude_by_breadth, breadth_weighted, type_split):
            f(out)
    elif args.part in ('A', 'B', 'C'):
        if args.part == 'A':
            import E80_pulse as Pp
            Pp.run_pulse(out)
        else:
            import E80_screen as S
            {'B': S.screen, 'C': S.gate}[args.part](out)
    else:
        raise NotImplementedError(args.part)


if __name__ == '__main__':
    main()
