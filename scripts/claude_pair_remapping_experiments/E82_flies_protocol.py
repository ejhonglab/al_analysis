#!/usr/bin/env python3
"""E82: the flies' recording protocol in the model. The design is the E82 entry in experiment_priorities.md (planned by a
Fable subagent, claude-fable-5-2, spawned from viki 2026-11-11 02:48).

- (A) No simulations: saved runs with spike times (E80's at 0.5, 1 and 2 s; then parts B and D's) read through an
  indicator kernel (1 - exp(-t/rise)) exp(-t/decay) over the flies' window (2.0 to 8.1 s after onset), rise 0.2 / 0.5 / 1
  s by decay 1 / 3 / 5 s (primary: rise 0.5, decay 3), with and without E65's saturation; the late spikes by condition,
  250 ms bin, odor, KC type and claw count; the loss with the APL-off spikes after 0.5 s left out; the model's predicted
  traces. `kc_remapping_harness.kernel_weighted_counts`.
- (B) the runs at the flies' pulse: E82_protocol_runs.py.
- (C) the protocol gate and E77's gate.
- (D) fits at the flies' pulse (E10_joint_fit.py --stim-end / --sim-end / --count-kernel).

    python E82_flies_protocol.py --part {A,C,tables}
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPERIMENT = 'E82_flies_protocol'
KERNELS = [(r, d) for r in (0.2, 0.5, 1.0) for d in (1.0, 3.0, 5.0)]
PRIMARY = (0.5, 3.0)
WINDOW = (2.0, 8.1)


def protocol_readings(out: Path) -> pd.DataFrame:
    """Part A: every saved run with spike times read through each kernel; writes protocol_readings.csv,
    protocol_pairs.csv."""
    raise NotImplementedError


def late_spikes(out: Path) -> pd.DataFrame:
    """Part A: the late spikes by condition, 250 ms bin, odor, KC type and claw count, and the loss with the APL-off spikes
    after 0.5 s left out; writes late_spikes.csv."""
    raise NotImplementedError


def predicted_traces(out: Path) -> pd.DataFrame:
    """Part A: the responders' mean kernel-convolved signal per condition from onset to 8.1 s; writes
    predicted_traces.csv."""
    raise NotImplementedError


def gates(out: Path) -> pd.DataFrame:
    """Part C: the protocol gate and E77's gate on part B's points; writes gate.csv."""
    raise NotImplementedError
