#!/usr/bin/env python3
"""E80 parts A to C: the flies' pulse in the model and the screen of the remaining candidates for the common error, on C2
(the breadth-2 APL-effect fit, E67) at fixed scales on the real wiring with both response rates held, on E77's runner
(E77_input_screen: hold_rates, build_c2_simulator, PerOdorRunner). The design is the E80 entry in experiment_priorities.md.

- (A) the pulse: C2 (and E76's fit 0, the firing-ceiling fit) with a 0.5 s (identity), 1 s and 2 s odor pulse, the
  simulation ending 0.25 s after it, at C2's parameters and with both rates re-held; the model's APL effect in windows of
  its own response from saved spike times. Needs mb_model's stimulus-end / simulation-end options.
- (B) the screen: B1 the threshold lower with APL on than off (c 1.1, 1.25, 1.5, 2; on C2 and fit 0); B2 a shared
  lateral input (epsilon x the odor's mean positive ORN delta added to every glomerulus; control: the odor's own profile
  scaled to the same summed drive); B3 each odor's response fractions put on the flies' (per-odor thresholds, then APL
  scales, in one-odor runs); B4 changes that act only with APL off (a per-KC threshold factor growing with the KC's
  APL-off activity, beta 0.25 to 2, with a within-type permutation; one type's threshold x 1.25 / 1.5); B5 APL->KC weights
  of one KC type x 0.5 / 2.
- (C) E77's gate on part B, and what follows for a handle that passes (re-drawn wirings, then re-fits).

Outputs under results/E80_remaining_candidates/ (pulse.csv, pulse_windows.csv, screen.csv, screen_pairs.csv,
screen_params.json, gate.csv); spike counts under outputs/E80_remaining_candidates/.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

EXPERIMENT = 'E80_remaining_candidates'


def pulse_points() -> List[dict]:
    """Part A's points: pulse lengths 0.5 (identity), 1 and 2 s, at C2's parameters and re-held; fit 0 at 2 s, re-held."""
    raise NotImplementedError


def run_pulse(out: Path, only: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Runs part A (resumable); writes pulse.csv and pulse_windows.csv."""
    raise NotImplementedError


def point_specs() -> List[dict]:
    """Part B's points (B1 to B5, each family with its identity point first)."""
    raise NotImplementedError


def screen(out: Path, only: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Runs part B (resumable) and scores each point as E77's screen does; writes screen.csv, screen_pairs.csv and
    screen_params.json."""
    raise NotImplementedError


def gate(out: Path) -> pd.DataFrame:
    """Part C: E77's gate on part B (E77_input_side_common_error.gate's conditions); writes gate.csv."""
    raise NotImplementedError
