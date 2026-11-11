#!/usr/bin/env python3
"""E83: the error common to all wirings, fitted on several wirings at once: a profile of APL's weight over the glomeruli
(arm P) and each odor's input (arm I), searched on four wirings and one half of the recordings, tested on four other
wirings and on the other half.

The entry is item E83 of experiment_priorities.md (planned by a Fable subagent, claude-fable-5-2, spawned from hal
2026-11-11 03:15). C2's family throughout (the breadth-2 APL-effect fit of E67: breadth exponent 2, no glomerulus gain, no
per-claw threshold, 6.8 %), each wiring at its own fit of E72's part B (C2 on the real wiring), the 0.5 s pulse.
- S, the four searched wirings: real, P1, P2, M1. T, the four held out: P3, P4, M2, M3 (E72 part D's division).
- Halves A and B of the recordings: E77's (`E77_input_side_common_error.recording_halves()`, seed 0).
- A population is the mean over its wirings of the pair distances per condition (E72).

This module holds the readings every part uses (a profile of APL's weight at the cell site, E75's `profile_factors`, or an
ORN input with per-odor, per-glomerulus factors, on any of the eight wirings, each at its own fit; the targets; the
population objective of the searches; the populations, the two-by-two tables, the gate) and part A. The searches
(parts B and C) are in E83_searches.py (`part_b(out, half)`, `part_c(out, half)`), built on this interface; `--part B` /
`--part C` here call them.

Parts (CLI `--part`):
- identity: each wiring's simulator against its saved fit (factor 1, zero scores), E75's two archived points on the real
  wiring, the population objective at identity (also test/test_e83_cross_wiring.py).
- A1: E75's cell-site profiles (results/E75_glomerulus_apl_profile/profiles.csv) on the eight wirings at their own fits:
  the ridge profile at 0.5 and 1 times its scores, the sparse profile at 0.5, C2's four glomeruli at x2, one label
  permutation (E75's seed 1) of the ridge profile at 0.5; the APL->KC scale (KC->APL and bouton->APL tied) re-bisected to
  each wiring's APL-on count; the APL-off run is the saved fit.
- A2: E77's two searched inputs (results/E77_input_side_common_error/oracle_factors_<A|B>.csv) at 0.6 of their log
  factors on all eight wirings and at full size on T (the four of S at full size are E77's archived runs, read, not
  rerun); breadth recomputed for the input, both response rates re-held (E77's reading, `hold_rates`).
- A-score: existing_changes.csv, readings.csv, pairs.csv, the part-A section of comparison_table.md.
- B, C: the searches (E83_searches.py).
- D: each half's searched profile / input at 0.6 of its log size on all eight wirings and at full size on T; the
  two-by-two tables, the gate, the agreement between the halves.
- E: only for an arm that passes (not written).

Outputs: results/E83_cross_wiring_fit/ (existing_changes.csv, runs.csv (absolute parameters, factors, realism per run),
readings.csv (per run and population, the scores against every target), pairs.csv (per-pair APL effect per run and
population), two_by_two.csv, gate.csv, agreement.csv, comparison_table.md, model_pair_distances_long.csv, variants.csv,
provenance_<part>.json (H.model_provenance per point)); the searches' files (search_history_<arm>_<half>.csv,
profile_scores_<half>.csv: glomerulus, score; input_factors_<half>.csv: odor, glomerulus, orn_delta, factor,
log2_factor, searched_delta, as E77's oracle_factors; inputs/ in E38's layout). Spike counts and responses under
outputs/E83_cross_wiring_fit/<part>/ (tag <part>_<change>_<wiring>, conditions on / acute_off).

Usage (from this folder):
    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 nice -n 10 ../../venv/bin/python -u E83_cross_wiring_fit.py --part A1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E83_cross_wiring_fit'
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
S_WIRINGS = ('real', 'P1', 'P2', 'M1')
T_WIRINGS = ('P3', 'P4', 'M2', 'M3')
ALL_WIRINGS = S_WIRINGS + T_WIRINGS
POPULATIONS = {'S': S_WIRINGS, 'T': T_WIRINGS, 'all8': ALL_WIRINGS}
BREADTH_K = 2.0
# E75's / E77's APL scale tie (ProfileModel.run_on, E77_input_screen.TIED_ON) and bisection
TIED_ON = ('wAPLKC_scale', 'wKCAPL_scale', 'wPNAPL_scale')
APL_BRACKET, APL_STEP = (0.1, 10.0), 1.25
# the gate's size: the share of a searched change's log size at which it is read (fixed before any run)
GATE_SIZE = 0.6
# targets of every reading: E77's halves, all recordings (the main target), the six typical TNT recordings, E81's
# noise-free effect (Yang's z), E76's baseline-noise units
TARGETS = ('halfA', 'halfB', 'z8', 'z6', 'noise_free', 'noise8')
# the probe values of the population objective at identity (S's pattern part against the halves, E77's oracle_scores)
IDENTITY_OBJECTIVE = {'A': 0.0569, 'B': 0.0552}
# part A1's profiles: name -> (E75 profiles.csv (profile, variant), multiplier, permutation seed or None, or fixed scores)
A1_POINTS = {
    'ridge_x0.5': (('cell', 'ridge'), 0.5, None),
    'ridge_x1': (('cell', 'ridge'), 1.0, None),
    'sparse_x0.5': (('cell', 'sparse'), 0.5, None),
    'C2_four_x2': (None, 1.0, None),
    'ridge_perm1_x0.5': (('cell', 'ridge'), 0.5, 1),
}
# E75's archived points on the real wiring that A1 must reproduce exactly: A1 name -> E75 tag
E75_ARCHIVED = {'ridge_x0.5': 'C2_D2s1_cell_ridge_x0.5', 'C2_four_x2': 'C2_cell_C2_four_x2'}
# E75's sensitivity_map point of each A1 name on the real wiring (factor and response count to match)
E75_POINT_IDS = {'ridge_x0.5': 'D2s1:cell:ridge:x0.5', 'ridge_x1': 'D2s1:cell:ridge:x1', 'sparse_x0.5': 'D2s1:cell:sparse:x0.5',
                 'C2_four_x2': 'cell:C2_four:x2', 'ridge_perm1_x0.5': 'D2s1:cell:perm1:x0.5'}
A2_HALVES = ('A', 'B')


# ---------------------------------------------------------------------------
# wirings and their fits

def members() -> Dict[str, Tuple[str, str]]:
    """{label: (folder, tag)} of E72's eight wirings of C2's family at their own fits (E77's `members`, E72's
    `new_fit_members`: the lower of a draw's two starts), in ALL_WIRINGS order."""
    raise NotImplementedError


def wiring_kws(label: str) -> dict:
    """The `E10_joint_fit.bouton_base` wiring kwargs of a wiring label, as E77's `wiring_simulator` passes them: {} for
    'real', `wiring='claw_permutation_within_type', wiring_seed=n` for Pn, and E59's matched curveball matrix with its
    `wiring_tag` for Mn."""
    raise NotImplementedError


def wiring_parts(label: str) -> dict:
    """A wiring's pieces for the lab's input, computed once per process: 'label', 'folder', 'tag' (its fit), 'p' (the
    fit's absolute parameters, E72's `fitted_params`), 'wnorm' (normalized APL weights), 'W' (claw x bouton wiring),
    'breadth' (the lab input's, E10_joint_fit.bouton_base), 'saved' (the fit's {'on', 'off'}: (responses, spike counts)),
    'target_on' / 'target_off' (its response counts, two or more spikes), 'size' (KCs x odors), 'K' (the calibrated
    readout's K on the fit's APL-on counts, E72's `distances`), 'orn' (the lab's input)."""
    raise NotImplementedError


def build_simulator(label: str, orn: Optional[pd.DataFrame] = None, breadth: Optional[pd.Series] = None,
                    params: Optional[Dict[str, float]] = None, input_tag: Optional[str] = None) -> Tuple:
    """C2's family on one wiring as a fixed-parameter simulator (`make_fixed_param_kc_simulator`, one simulation to
    build), at the wiring's fit's parameters, with the ORN input `orn` (default the lab's).

    Args:
        label: one of ALL_WIRINGS.
        orn: the ORN input (the lab's layout; for candidates searched as pseudo-odors, E64's `candidate_orn`).
        breadth: the breadth behind the APL->KC weights. None: the lab input's (`wiring_parts`) when `orn` is None,
            otherwise recomputed for `orn` (`E10_joint_fit.bouton_base(orn, ..., input_tag=input_tag)`, which needs
            `input_tag`; the APL-off breadth run is cached under outputs/E83_cross_wiring_fit/ by that tag, so a tag
            must name one input only: `input_tag_for` makes one from the input's hash). Pass the lab's breadth
            (`wiring_parts(label)['breadth']`) to hold it, as the searches of part C do.
        params: starting parameters (default the fit's).
        input_tag: see `breadth`.

    Returns:
        (simulate, info): info has 'params', 'breadth', 'wAPLKC' (breadth-weighted APL->KC weights), 'wAPLPN', 'W',
        'orn', 'cells' (the simulator's closure cells: 'wAPLKC_unscaled' / 'wAPLPN_unscaled' for in-place profile
        changes, 'mp', 'rv', 'odor_order').
    """
    raise NotImplementedError


def input_tag_for(orn: pd.DataFrame, prefix: str = 'E83') -> str:
    """A cache tag naming one ORN input: `prefix` and the first 12 hex digits of its `E77_input_screen.frame_sha256`."""
    raise NotImplementedError


class WiringModel:
    """One wiring at its fit with the lab's input, for profiles of APL's weight (E75's ProfileModel on any wiring): a
    fixed-parameter simulator whose APL->KC and APL->bouton weights a profile changes in place. `run_on(x)` runs APL on
    with the APL->KC, KC->APL and bouton->APL scales times x (TIED_ON; APL->bouton and the somatic threshold unchanged);
    the APL-off run does not depend on APL's weights and is the saved fit's. With `check`, the constructor verifies that
    at factor 1 the simulator reproduces the saved APL-on and APL-off spike counts exactly (raises otherwise).

    Attributes: label, folder, tag, p, saved, target_on, target_off, size, K, W, wAPLKC, wAPLPN, sim, info."""

    def __init__(self, label: str, check: bool = True):
        raise NotImplementedError

    def set_profile(self, site: Optional[str], scores: Mapping[str, float]) -> None:
        """Set the APL weights to the fit's times a profile (E75's `profile_factors` at `site`, 'cell' / 'claw' /
        'bouton'; each wiring's own claws); None or no scores restores the fit's weights exactly."""
        raise NotImplementedError

    def run_on(self, x: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """(responses, spike counts) with APL on at the fit's APL scales (TIED_ON) times `x`, the current profile."""
        raise NotImplementedError

    def outputs(self, on: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict[str, tuple]:
        """{'on': `on`, 'off': the saved fit's APL-off run}."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# changes: profiles and inputs, at a given log size

def e75_profile(name: str) -> Dict[str, float]:
    """Part A1's profile `name` (A1_POINTS) as scores {glomerulus: score}: E75's profiles.csv scores times the
    multiplier, permuted over the glomeruli as E75's D2 stage 1 permutes them (seed), or C2's four glomeruli at ln 2."""
    raise NotImplementedError


def scale_scores(scores: Mapping[str, float], size: float) -> Dict[str, float]:
    """A profile at `size` times its log size (scores times size)."""
    raise NotImplementedError


def read_factors(path: Path) -> pd.DataFrame:
    """An input's factors as glomeruli x odors (abbreviated, the lab input's order), 1 where none is given, from a long
    table with columns odor, glomerulus, factor (E77's oracle_factors_<half>.csv, E83's input_factors_<half>.csv)."""
    raise NotImplementedError


def input_at(factors: pd.DataFrame, size: float, orn: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The ORN input with each odor's per-glomerulus factors at `size` of their log size: orn x factor ** size (orn the
    lab's by default, `H.apl16_orn_deltas()` layout); size 0 gives the lab's input exactly."""
    raise NotImplementedError


def profile_reading(model: 'WiringModel', scores: Mapping[str, float], site: str = 'cell', label: str = '') -> dict:
    """A profile on one wiring at its fit: the APL scale (TIED_ON) bisected (E69's `bisect_factor`, APL_BRACKET,
    APL_STEP, from factor 1) to the wiring's APL-on response count; the APL-off run is the saved fit's. The model's
    profile is reset on return.

    Returns:
        dict with 'on' / 'off' ((responses, spike counts)), 'apl_factor', 'thr_factor' (1), 'n_on', 'n_off',
        'target_on', 'target_off', 'at_bound_apl', 'n_sims', 'params' (absolute, APL on), 'history'."""
    raise NotImplementedError


def input_reading(label: str, orn: pd.DataFrame, input_tag: Optional[str] = None) -> dict:
    """An ORN input on one wiring at its fit, E77's reading: the breadth recomputed for the input
    (`build_simulator(label, orn, input_tag=...)`), the somatic threshold bisected to the fit's APL-off response count,
    then the APL scale (TIED_ON) to its APL-on count (`E77_input_screen.hold_rates`).

    Returns:
        hold_rates' dict ('on', 'off', 'params', 'params_off', 'thr_factor', 'apl_factor', 'n_on', 'n_off',
        'at_bound_thr', 'at_bound_apl', 'n_sims', 'history', 'breadth_sha256', ...) plus 'input_tag'."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# targets, distances, populations, scores

def targets() -> Dict[str, pd.Series]:
    """The data's per-pair APL effect (canonical pair order) for every name in TARGETS: 'halfA' / 'halfB' (E77's halves,
    `half_effect`), 'z8' (all recordings, Yang's z: the main target), 'z6' (all control recordings, the six typical TNT
    recordings), 'noise_free' (E81's noise_free_pairs.csv, units z, effect_noise_free), 'noise8' (E76's
    effect_by_unit_pairs.csv in baseline-noise units)."""
    raise NotImplementedError


def half_target(half: str) -> pd.Series:
    """The per-pair APL effect of half 'A' or 'B' of the recordings (E77's `recording_halves()` and `half_effect`)."""
    raise NotImplementedError


def readout_K(o: Dict[str, tuple]) -> float:
    """The calibrated readout's K on a run's APL-on spike counts (as E72's `distances` sets it)."""
    raise NotImplementedError


def pair_distances(o: Dict[str, tuple], readout: str, K: Optional[float] = None) -> Dict[str, pd.Series]:
    """{'on', 'off'} pair distances (canonical order) of a run: readout 'raw' (cosine on counts), 'cal' (the calibrated
    readout, K re-calibrated on this run's APL-on counts unless `K` is given: the readings' default, as E75's and E77's)
    or 'calK' (needs `K`, e.g. the wiring's fit's: the searches' objective)."""
    raise NotImplementedError


def population(ds: Sequence[Dict[str, pd.Series]]) -> Dict[str, pd.Series]:
    """The mean over wirings of the pair distances per condition (E72's `mean_distances`), canonical order."""
    raise NotImplementedError


def effect_scores(d: Dict[str, pd.Series], tg: Optional[Dict[str, pd.Series]] = None) -> dict:
    """The scores of a run's or a population's distances against every target: per target '<t>_loss' (rms of model
    minus data per-pair effect), '<t>_pattern' (the same with both means removed), '<t>_rho' (ρ on the per-pair APL
    effect), '<t>_err_ester' / '<t>_err_alcohol' (mean pattern error over E75's groups' pairs), and 'mean_effect'."""
    raise NotImplementedError


class PopulationObjective:
    """The searches' objective: the squared pattern error of a population's per-pair APL effect under the calibrated
    readout, each wiring's K held at its fit's (`wiring_parts(label)['K']`), against a target (a half's effect). The
    APL-off distances of a wiring not given are its saved fit's (computed once).

    Usage: `obj = PopulationObjective(half_target('A'))`; `obj({'real': s_on, ...})` with APL-on spike counts per wiring
    (arm P), or `obj(on={...}, off={...})` (arm I). Returns dict(sse=..., rms=sqrt(sse / n_pairs), n_pairs=...). At the
    fits (no change) rms is IDENTITY_OBJECTIVE[half] to four decimals (0.0569 against half A, 0.0552 against half B)."""

    def __init__(self, target: pd.Series, wirings: Sequence[str] = S_WIRINGS):
        raise NotImplementedError

    def __call__(self, on: Mapping[str, pd.DataFrame], off: Optional[Mapping[str, pd.DataFrame]] = None) -> dict:
        raise NotImplementedError


def realism(o: Dict[str, tuple]) -> dict:
    """A run's response rates, their ratio, the response rate ρ and the odors-per-KC distances under both rules, the
    largest spike counts and the flags (E77's `realism_of`)."""
    raise NotImplementedError


def save_run(part: str, tag: str, o: Dict[str, tuple]) -> None:
    """Spike counts and responses of a run under outputs/E83_cross_wiring_fit/<part>/ (<tag>_<on|acute_off>_...)."""
    raise NotImplementedError


def load_run(part: str, tag: str) -> Optional[Dict[str, tuple]]:
    """A saved run of this experiment ({'on', 'off'}), or None."""
    raise NotImplementedError


def score_runs(runs: Mapping[Tuple[str, str], Dict[str, tuple]], change_meta: Mapping[str, dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Readings: for every (change, wiring) run and every population of POPULATIONS whose wirings all have a run of the
    change (an unchanged wiring's run is its saved fit, change 'none'), `effect_scores` under 'cal', 'calK' (each
    wiring's fit's K) and 'raw', with the paired change against the same wirings unchanged.

    Returns:
        (readings, pairs): readings one row per (change, unit, readout) with unit a wiring or a population name;
        pairs the per-pair APL effect per (change, unit, readout)."""
    raise NotImplementedError


def two_by_two(readings: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Per arm and direction (searched on A, scored on B; searched on B, scored on A): S and T against the half searched
    on and the half never seen, unchanged and with the change at GATE_SIZE and at full size, and each single wiring as a
    paired difference against its own unchanged fit."""
    raise NotImplementedError


def gate(readings: pd.DataFrame, arm: str) -> pd.DataFrame:
    """The entry's gate per arm at GATE_SIZE in both directions: (i) T's pattern part against the unseen half lower by
    0.003 or more; (ii) lower on at least six of T's eight wiring-by-direction cells; (iii) every point inside the four
    bands, T's raw-count loss within 0.005 of its unchanged value, its mean APL effect under the readout within 0.01 of
    the flies'; (iv) T's squared pattern error against all recordings 15 % lower (mean of the directions), half that
    against the six typical recordings; (v) the label-permuted controls (when run) gain less than half as much on (i);
    and the borderline rule (i between 0.0015 and 0.0045)."""
    raise NotImplementedError


def agreement(out: Path, arm: str) -> pd.DataFrame:
    """Agreement between the halves: the correlation of the two halves' scores over the glomeruli (arm P) or per odor of
    the log factors over glomeruli (arm I) with the share of the factors' variance a per-glomerulus and a per-odor term
    explain; the correlation of the two halves' per-pair changes of T's APL effect."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# parts

def identity(out: Path) -> dict:
    """The identity checks: every wiring's simulator reproduces its saved fit at factor 1 and zero scores; A1's two
    archived E75 points on the real wiring reproduced exactly; the population objective at identity. Writes
    identity.json."""
    raise NotImplementedError


def part_a1(out: Path, wirings: Sequence[str] = ALL_WIRINGS, points: Sequence[str] = tuple(A1_POINTS)) -> pd.DataFrame:
    """Part A1 (see the module docstring); resumable (runs saved under outputs/.../A1/, rows in runs.csv)."""
    raise NotImplementedError


def part_a2(out: Path, halves: Sequence[str] = A2_HALVES) -> pd.DataFrame:
    """Part A2 (see the module docstring); resumable."""
    raise NotImplementedError


def score_part_a(out: Path) -> pd.DataFrame:
    """existing_changes.csv (A1 and A2 per change and unit: the pattern part against every target unchanged and with the
    change, the paired change on single wirings, the outcome's conditions (a1) to (a3)), readings and pairs rows, and the
    part-A section of comparison_table.md."""
    raise NotImplementedError


def part_d(out: Path, arm: str) -> pd.DataFrame:
    """Part D for one arm ('P' or 'I'): each half's searched change at GATE_SIZE on all eight wirings and at full size on
    T; the two-by-two tables, the gate, the agreement."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['identity', 'A1', 'A2', 'A-score', 'B', 'C', 'D', 'E'])
    ap.add_argument('--half', choices=['A', 'B'], help='parts B and C: the half of the recordings searched against')
    ap.add_argument('--arm', choices=['P', 'I'], help='part D: the arm read')
    ap.add_argument('--wirings', nargs='+', default=list(ALL_WIRINGS), help='part A1: the wirings to run')
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    if args.part == 'identity':
        identity(out)
    elif args.part == 'A1':
        part_a1(out, wirings=args.wirings)
    elif args.part == 'A2':
        part_a2(out)
    elif args.part == 'A-score':
        score_part_a(out)
    elif args.part in ('B', 'C'):
        import E83_searches as SR
        if args.half is None:
            ap.error(f'--part {args.part} needs --half A or B')
        (SR.part_b if args.part == 'B' else SR.part_c)(out, args.half)
    elif args.part == 'D':
        if args.arm is None:
            ap.error('--part D needs --arm P or I')
        part_d(out, args.arm)
    else:
        raise NotImplementedError(f'part {args.part}')


if __name__ == '__main__':
    main()
