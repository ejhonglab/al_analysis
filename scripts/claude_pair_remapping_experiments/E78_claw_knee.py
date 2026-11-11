#!/usr/bin/env python3
"""E78: the claw's transfer under the main target. A high per-claw knee without a dead zone on the best family (C2, the
breadth-2 APL-effect fit, E67 rate0.068_aplboth_k2_cap3), for the odor pairs with near-identical input that the flies'
KCs hold apart between claw and soma.

The entry (experiment_priorities.md, item 3; planned by a Fable subagent, claude-fable-5-2, spawned from hal 2026-11-10
17:47) is the spec. Parts:

- A: the hard knee at 100 and 130 Hz in the best family, `E10_joint_fit.py` as it is (run by hal's session; fits.csv and
  histories in results/E78_claw_knee/).
- B (`--part B`, no simulations but one run per model for the time courses): the archive's knee variants on E77's kinds
  of pairs and on the six near pairs, each against its own base (archive_knee_variants.csv); the six pairs through the
  saved trial-noise rows (noise_six_pairs.csv); who selects the responders (who_selects.csv).
- C (`--part C`): fixed scales on C2 (and, for the five best points, on E69's fit 1 and E72's P1 fit), both response
  rates held as E77's screen holds them: the somatic threshold bisected to the start's APL-off response count, the
  breadth weights recomputed for the transfer, the APL->KC scale (KC->APL and bouton->APL tied to it) bisected to the
  start's APL-on response count; KC-KC and APL->bouton scales the start's. C1: knees 100 / 115 / 130 Hz with a
  sub-threshold gain g of 0 / 0.1 / 0.25 / 0.5, 70 Hz with g 0 / 0.25, a softplus width of 15 Hz at the two best;
  C2: the knee relative to the odor (the per-odor external inhibition at the boutons, E9's `ext_inh_bouton`, at 0.5 /
  0.7 / 0.85 times the odor's largest bouton rate, in both conditions). Writes sweep.csv, sweep_pairs.csv,
  sweep_params.json; `--part C-low` runs the five best at Yang's low concentrations (E49's script, --into).
- gate (`--part gate`): the entry's four conditions on part C's points (gate.json).
- D (`--part D`): the launch commands of the fits (E10_joint_fit.py with --claw-thr / --claw-thr-subgain /
  --claw-thr-width from the gate point's parameters, on the real wiring and P1, P2, M1), the held-out run on P3, P4, M2,
  M3, the paired and population tables.
- E (`--part E`): every fit and gate point under the per-KC reading (E73's scorer), the release by claw count.
- summarize (`--part summarize`): comparison_table.md and the figures.

Outputs: <shared>/results/E78_claw_knee/; large outputs (spike counts, responses, time courses) under
./outputs/E78_claw_knee/.
Usage (from this folder; at most two 16-thread processes on hal):
    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 nice -n 10 ../../venv/bin/python -u E78_claw_knee.py --part C [--only ID ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E78_claw_knee'
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
# the starts of part C (experiment, fit tag, wiring options as E10_joint_fit.py takes them)
STARTS = {
    'C2': (C2[0], C2[1], {}),
    'fit1': ('E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.04142t1000_k2', {}),
    'P1': ('E72_wiring_population', 'wiring-claw_permutation_within_type-seed1_rate0.068_aplboth_k2_cap3',
        dict(wiring='claw_permutation_within_type', wiring_seed=1)),
}
ESTER = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL = ('1-8ol', '1-5ol', '1-6ol', '6al')
# the six pairs with an ORN distance below 0.25 (the entry's table) and E77's kinds of ester pairs
SIX = (('eb', 'ep'), ('6al', 't2h'), ('IaA', 'pa'), ('2h', 'pa'), ('1-5ol', '1-6ol'), ('2h', 'IaA'))
NEAREST = (('2h', 'IaA'), ('2h', 'pa'), ('IaA', 'pa'), ('eb', 'ep'))
BLOCK = tuple((a, b) for a in ('2h', 'IaA', 'pa') for b in ('eb', 'ep'))
MS_PAIRS = tuple(('ms', o) for o in ('2h', 'IaA', 'pa', 'eb', 'ep'))
KNEES = (100.0, 115.0, 130.0)
SUBGAINS = (0.0, 0.1, 0.25, 0.5)
LINK_KNEE, LINK_SUBGAINS = 70.0, (0.0, 0.25)
SOFT_WIDTH = 15.0
REL_LEVELS = (0.5, 0.7, 0.85)
N_BEST = 5
# the gate (the entry's part D): (i) ester group's error reduced by a third or more and the six pairs' raw APL effect
# raised by 0.03 or more; (ii) readout loss within 0.004 of the start's (0.006 for the firing ceiling's points);
# (iii) every odor keeps half its start's response fraction in both conditions, response rate ρ with APL on >= 0.50;
# (iv) 2h at 10^-5 keeps half the fraction the start recruits
GATE = dict(ester_reduction=1 / 3, six_raise=0.03, loss_tol=0.004, loss_tol_ceiling=0.006, odor_frac_keep=0.5,
    rate_rho_min=0.50, low_conc_keep=0.5)


# ---------------------------------------------------------------------------
# part B
def archive_knee_variants(out: Path) -> pd.DataFrame:
    """The archive's knee variants (E16, E22, E26, E41, E32's PN-gain rows, the knee fits of E67 and E72) on E77's kinds
    of pairs (ms's, the four nearest, the block's six, the alcohol group) and on the six pairs, each against its own
    base, with the knee over that base's largest bouton rate; the ester and alcohol group errors and the projection on
    the common error (E75 `error_vs_common`). Writes archive_knee_variants.csv."""
    raise NotImplementedError


def noise_six_pairs(out: Path) -> pd.DataFrame:
    """The six pairs through the saved trial-noise rows (E72 part E, E75 part A, E73's), like for like with the flies' raw
    distances. Writes noise_six_pairs.csv."""
    raise NotImplementedError


def who_selects(out: Path, models: Sequence[str] = ('C2', 'A100', 'A130')) -> pd.DataFrame:
    """Who selects the responders: one run of C2 and of each part-A fit with the claw and bouton time courses; per odor
    and condition the share of claw inputs above the knee, the number of claws that carry a responder's drive (half of
    it), the somatic threshold over the median responder's drive, and APL's level at the claw against the odor's bouton
    rates. Writes who_selects.csv."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# part C
def largest_bouton_rates(start: str = 'C2') -> pd.Series:
    """Each odor's largest bouton rate in the start's model (the PN rate of the most driven glomerulus in the wiring, mean
    over the odor window, Hz), by odor abbreviation."""
    raise NotImplementedError


def odor_relative_inhibition(W: pd.DataFrame, orn: pd.DataFrame, top: pd.Series, level: float) -> pd.DataFrame:
    """The odor-relative knee as olfsysm's per-odor external inhibition at the boutons: a bouton x odor matrix (the
    wiring's bouton columns, the ORN input's odor columns) whose every row is `level` times the odor's largest bouton
    rate `top` (`largest_bouton_rates`)."""
    raise NotImplementedError


def point_specs() -> List[dict]:
    """Part C's points on C2: C1 (knee x sub-threshold gain, the 70 Hz link points; the softplus width at the two best
    of them, added once those exist) and C2 (the odor-relative knee at REL_LEVELS); each a dict with point_id, family,
    transfer kwargs (claw_thr, claw_thr_subgain, claw_thr_width, or rel)."""
    raise NotImplementedError


class KneeModel:
    """A start (STARTS) rebuilt at its fitted parameters (its spike counts checked equal to its saved ones), with a
    transfer set on top: the somatic threshold factor bisected to the start's APL-off response count, the breadth weights
    recomputed for the transfer (the plain base's APL-off run with the transfer, its threshold re-tuned to that run's
    response count; at the identity the start's breadth exactly), the APL->KC scale (KC->APL and bouton->APL tied)
    bisected to the start's APL-on response count."""

    def __init__(self, start: str, out_dir: Path):
        raise NotImplementedError

    def hold(self, transfer: dict, label: str) -> Tuple[dict, Dict[str, tuple], dict]:
        """(absolute parameters, {'on': (responses, spike counts), 'off': ...}, record of the bisections) of the start with
        `transfer`."""
        raise NotImplementedError


def score_point(o: Dict[str, tuple], ref: Dict[str, tuple]) -> Tuple[dict, List[dict]]:
    """A point's statistics: the per-pair APL effect on raw counts, under the calibrated readout (K re-calibrated on its
    APL-on counts) and binarized; the loss with its parts and ρ on the per-pair APL effect against all eight recordings
    with APL output blocked (TNT), the six typical ones and the effect in baseline-noise units; E77's kinds of pairs and
    the six pairs; the ester and alcohol group errors and the projection on the common error; the 16 odors' response
    fractions in both conditions with the response rate ρ; odors per KC; the class split (`H.apl_effect_by_class`); the
    largest spike counts; the realism bands under both rules. Returns (row, per-pair rows)."""
    raise NotImplementedError


def sweep(out: Path, start: str = 'C2', only: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Runs (resumably) part C's points on `start` and scores each (`score_point`); appends to sweep.csv and
    sweep_pairs.csv, the absolute parameters to sweep_params.json, provenance per point (H.model_provenance)."""
    raise NotImplementedError


def best_points(sweep_df: pd.DataFrame, n: int = N_BEST) -> List[str]:
    """The n best points of the sweep on C2 by the readout loss among those inside the bands (all points if fewer)."""
    raise NotImplementedError


def low_conc(out: Path, point_ids: Sequence[str]) -> pd.DataFrame:
    """The points at Yang's low concentrations (ma at 10^-7, 2h at 10^-5; E49's script with --into, deterministic), against
    C2's fractions and the flies'. Writes low_conc/."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# the gate
def gate_point(row: dict, ref: dict, ceiling: bool = False) -> Dict[str, object]:
    """The entry's four conditions for one point (`row`, a sweep.csv row) against its start (`ref`, the start's row):
    each condition's values and pass flag, the bands, and 'pass'."""
    raise NotImplementedError


def gate(out: Path) -> dict:
    """The gate on part C's points; which transfers go on to part D (at most two: the best knee with a gain, the
    odor-relative knee if it passes too). Writes gate.json."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# part D
def d_commands(g: dict) -> List[List[str]]:
    """E10_joint_fit.py argument lists of part D's fits for the transfers that passed the gate (real wiring, P1, P2, M1,
    from the gate point's parameters, written to results/E78_claw_knee/init_params.json)."""
    raise NotImplementedError


def d_launch(cmds: Sequence[Sequence[str]], max_jobs: int = 2) -> List[str]:
    """systemd-run --user units for the fits (nice 10, 16 threads, python -u, logs under outputs/E78_claw_knee/logs/);
    returns the unit names."""
    raise NotImplementedError


def d_tables(out: Path) -> pd.DataFrame:
    """The fits paired with C2 and with E72's part B fits per wiring, the population of four, the held-out four (the fits'
    mean parameters on P3, P4, M2, M3). Writes paired.csv, population.csv."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# part E and the summary
def e_score(out: Path) -> pd.DataFrame:
    """Every fit and gate point under the per-KC reading (E73_per_kc_normalization.py --part B --folders E78_claw_knee, once
    archived), split with E77's harness function when it exists; who is released when APL is blocked, by claw count
    (release.csv's form)."""
    raise NotImplementedError


def summarize(out: Path) -> None:
    """comparison_table.md and the figures (the sweep's loss, pattern part, six pairs' effect and ester error against
    the knee and the gain; the response fractions of the weak odors)."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['B', 'C', 'C-low', 'gate', 'D', 'E', 'summarize'])
    ap.add_argument('--start', default='C2', choices=list(STARTS), help='part C: the fit the points are set on')
    ap.add_argument('--only', nargs='+', default=None, help='part C: these point ids only')
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'B':
        for f in (archive_knee_variants, noise_six_pairs, who_selects):
            f(out)
    elif args.part == 'C':
        sweep(out, args.start, args.only)
    elif args.part == 'C-low':
        low_conc(out, best_points(pd.read_csv(out / 'sweep.csv')))
    elif args.part == 'gate':
        gate(out)
    elif args.part == 'D':
        d_launch(d_commands(gate(out)))
    elif args.part == 'E':
        e_score(out)
    else:
        summarize(out)


if __name__ == '__main__':
    main()
