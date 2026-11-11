#!/usr/bin/env python3
"""E80 part A: the flies' odor pulse in the model. C2 (the breadth-2 APL-effect fit, E67) and E76's fit 0 (the
firing-ceiling fit) with a 0.5 s (identity), 1 s and 2 s odor pulse, the simulation ending 0.25 s after it, spike counts
over the whole simulation: once at C2's parameters, once with both response rates re-held (E77_input_screen.hold_rates;
the breadth from the pre-pass at the same pulse length); the model's APL effect in windows of its own response (0-0.5,
0.5-1, 1-2 s) from saved spike times. Needs mb_model's stimulus-end / simulation-end options (defaults olfsysm's 0.5 and
0.75 s, outputs unchanged). The design is the E80 entry in experiment_priorities.md. Outputs under
results/E80_remaining_candidates/ (pulse.csv, pulse_windows.csv).
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
