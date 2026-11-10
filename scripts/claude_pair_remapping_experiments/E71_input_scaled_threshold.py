#!/usr/bin/env python3
"""E71: the input-scaled KC threshold. Which KCs are released when APL is blocked, and whether the APL
effect survives a release spread over the KCs as in the flies.

STUB (interface first; bodies raise NotImplementedError until implemented).

Every model KC had the same somatic threshold apart from a term proportional to its spontaneous input
(threshold = fixed_thr + 2 x spont_in), so a KC with ten claws needs the same depolarization as one with
three. Here each KC's threshold is multiplied by (W / mean W)^a, W the KC's summed PN->claw weight in the
wiring the model is given (the glomerulus gain included where the base has one; without a gain W is the
KC's number of claws), through the simulator's per-KC threshold factor (`kc_thr_factor`), identically in
the APL-on and the acute APL-off runs. a = 0 is the present model.

Part A (`--stage run`): the sweep at fixed scales. Bases (BASES): (1) 'c2', the APL-effect fits' (E67)
`rate0.068_aplboth_k2_cap3` as the firing ceiling experiment (E69) loaded it; (2) 'c2ceil', the same with
the firing ceiling at hal's E69 part-B setting (its fitted optimum if archived, else part A's
`e67cal_r3_a1_t1000`); (3) 'ref40', the reference model (E44_operating_point
`thr40_gain1.5_rate0.068_k2_cap3`). Exponents: c2 -0.5, 0.25, 0.5, 0.75, 1 (and 0.5, 1 with the base's
breadth kept); c2ceil 0.5, 1 and the two next to c2's chosen exponent; ref40 0.25, 0.5, 0.75, 1. Each
base is first re-run and must reproduce its archived spike counts exactly (`--stage check`). At every
setting (`--stage setting`, one process each): the breadth (the per-KC APL-off spike totals the APL->KC
weights are built from, E10's `bouton_base` pre-pass) is recomputed with the same factor (except for the
breadth-kept controls), both response rates are held at the base's by two bisections (the somatic
threshold factor on the APL-off response rate, then one common factor on APL->KC and APL->bouton on the
APL-on response rate; KC-KC and everything else as fitted) and the re-tuned absolute values recorded.
Readout per variant beside the base and the KC data (E69's `score` plus): who is released (responding
entries by claw-count group in both conditions, the rank correlation of W with the odors per KC), the
odors-per-KC distributions, silent and 8+ shares and the squared CV of the odors per KC against the
recordings' range, E70's high-firing set and survivors (the base's survivors cell by cell from
results/E70_high_firing_kcs/survivors.csv), binarized mean distances and the binarized APL-effect loss,
the share of each base's threshold already scaling with input through the spontaneous-input term. The
exponent of a base (`choose_exponent`): the smallest APL-off odors-per-KC distance among settings whose
APL-on distance is inside its band and whose APL-on response rate ρ is >= 0.37; within 0.01, the smaller
exponent.

Part B (`--stage partB-commands`): the E10_joint_fit.py commands for the re-fits (E67's `--loss apl
--apl-readout both`, k 2, cap 3, 6.8 %, `--kc-thr-input-exp A`, from part A's tuned_params.json).
Part C (`--stage partC-commands`): E43_model_trial_noise.py runs with the fixed factor composed with the
per-trial jitter, saving each pseudo-recording's pair distances; single-pair scoring (`--stage partC`).

Outputs: results/E71_input_scaled_threshold/ (bases.json, base_checks.json, settings_log.csv,
stats_variants.csv, model_pair_distances_long.csv, input_readouts.csv, released_by_group.csv,
tuned_params.json, comparison_table.md, figures, provenance.json); per-variant counts / responses in
outputs/E71_input_scaled_threshold/<tag>_<on|acute_off>_<spike_counts|responses>.parquet.

Usage (from this folder; one process at a time):
    ../../venv/bin/python E71_input_scaled_threshold.py --plan
    OMP_NUM_THREADS=12 OMP_THREAD_LIMIT=12 nice -n 10 ../../venv/bin/python E71_input_scaled_threshold.py --stage run
    ../../venv/bin/python E71_input_scaled_threshold.py --stage summarize
    ... --local-test DIR (results and outputs under DIR; a smoke test)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E71_input_scaled_threshold'


def kc_input_weight(wiring: pd.DataFrame, gain: Optional[pd.Series] = None) -> pd.Series:
    """W per KC: the summed PN->claw weight of each KC (rows of the KC x glomerulus wiring, times the
    per-glomerulus gain where the base has one)."""
    raise NotImplementedError


def input_thr_factor(W: pd.Series, a: float) -> np.ndarray:
    """(W / mean W)^a, one positive value per KC in W's (the model's output) row order."""
    raise NotImplementedError


def settings() -> List[dict]:
    """Every part-A setting: base, exponent a, whether the breadth is recomputed, tag."""
    raise NotImplementedError


def stage_check(key: str) -> None:
    """Re-runs base `key` and checks it reproduces its archived spike counts exactly; scores it."""
    raise NotImplementedError


def stage_setting(tag: str) -> None:
    """One setting: the factor, the breadth pre-pass, the two bisections, scored."""
    raise NotImplementedError


def choose_exponent(df: pd.DataFrame, base: str) -> Optional[float]:
    """The exponent rule of the entry (no odor-odor distance, no APL effect involved)."""
    raise NotImplementedError


def stage_run() -> None:
    """Every missing check and setting, each in its own process, then the summary."""
    raise NotImplementedError


def stage_summarize() -> None:
    """stats_variants.csv, tuned_params.json, comparison table, figures, provenance."""
    raise NotImplementedError


def partB_commands() -> List[str]:
    """E10_joint_fit.py command lines for part B's fits (only those whose conditions hold)."""
    raise NotImplementedError


def partC_commands() -> List[str]:
    """E43_model_trial_noise.py command lines for part C."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--stage', choices=['check', 'setting', 'run', 'summarize', 'partB-commands', 'partC-commands'])
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--local-test', metavar='DIR', default=None)
    ap.parse_args()
    raise NotImplementedError


if __name__ == '__main__':
    main()
