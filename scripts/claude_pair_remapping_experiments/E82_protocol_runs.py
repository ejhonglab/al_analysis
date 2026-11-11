#!/usr/bin/env python3
"""E82 part B: runs at the flies' pulse with both response rates re-held (E77_input_screen.hold_rates; E80_pulse's
builders), spike times saved: B1 the breadth-2 and the firing-ceiling populations of four wirings; B2 the ceiling
adaptation's time constant; B3 bouton depletion; B4 the KC-KC time constant; B5 APL's time constant (needs mb_model's
apl_taum). Every model first at 0.5 s must reproduce its saved counts exactly. Design: the E82 entry in
experiment_priorities.md. Outputs: results/E82_flies_protocol/ runs.csv, runs_windows.csv, runs_pairs.csv,
runs_params.json; outputs/E82_flies_protocol/runs/.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

EXPERIMENT = 'E82_flies_protocol'


def point_specs() -> List[dict]:
    """Part B's points (B1 to B5, identities first)."""
    raise NotImplementedError


def run_points(out: Path, only: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Runs part B (resumable)."""
    raise NotImplementedError
