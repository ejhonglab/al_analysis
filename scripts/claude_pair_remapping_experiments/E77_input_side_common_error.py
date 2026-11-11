#!/usr/bin/env python3
"""E77: the input side of the error common to all wirings (what reaches the KCs for the ester and the alcohol pairs).

The entry is in experiment_priorities.md (planned by a Fable subagent, claude-fable-5-2, spawned from viki 2026-11-10
17:55). The best family is C2, the breadth-2 APL-effect fit of E67 (rate0.068_aplboth_k2_cap3); its error common to all
wirings is E72's (E75's `common_error`). Parts:

- A: the best family re-fitted with another ORN input, by `E10_joint_fit.py` as it is (fit H: E38's hallem_shared.csv;
  fit S: E64's searched input for the weak odors, written in E38's layout by `--part A-input`); `--part A-score` reads
  both against C2 on the same wiring.
- B: no simulations, the anatomy of the common error: the kinds of ester pairs, the per-odor and pair-specific split
  with recording-bootstrap intervals, the split into what a model inherits from its APL-off distances, its on-against-off
  curve and a pair-specific part for every archived variant (`kc_remapping_harness.apl_effect_inherited_split`), the
  archive's input-side variants under their right names, the model's PN-level distances.
- C: the screen at fixed scales on C2 with both response rates held, and the APL-on-only emulations
  (`E77_input_screen.py`); `--part gate` reads it.
- D, E, F: re-fits of a handle that passes, the bounded input oracle, what it is (not written yet).

Outputs under results/E77_input_side_common_error/ (see the entry); large outputs under
outputs/E77_input_side_common_error/.

    python E77_input_side_common_error.py --part {A-input,A-score,B,C,gate,D,E,F}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E77_input_side_common_error'
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
ESTER = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL = ('1-8ol', '1-5ol', '1-6ol', '6al')
# the three kinds of ester pairs (the entry's table)
NEAREST = (('2h', 'IaA'), ('2h', 'pa'), ('IaA', 'pa'), ('eb', 'ep'))
BLOCK = tuple((a, b) for a in ('2h', 'IaA', 'pa') for b in ('eb', 'ep'))
MS_PAIRS = tuple(('ms', o) for o in ('2h', 'IaA', 'pa', 'eb', 'ep'))
FITS = {'H': 'input-hallem_shared_rate0.068_aplboth_k2_cap3', 'S': 'input-searched_weak_rate0.068_aplboth_k2_cap3'}


def searched_input(out: Path) -> Path:
    """Part A's fit S input: E64's pass-1 searched ORN input for the weak odors (orn_input_searched_pass1.csv, two-row
    panel / odor header) rewritten in E38's layout (glomeruli x abbreviated odors) as inputs/searched_weak.csv, values
    equal to the last digit. Returns the path relative to the shared directory, for `E10_joint_fit.py --orn-input`."""
    raise NotImplementedError


def score_fits(out: Path) -> pd.DataFrame:
    """Part A-score: fits H and S against C2 on the real wiring (the APL-effect loss with its parts on raw counts and
    under the calibrated readout, the ms pairs, the 1-8ol pairs, the group errors, the per-odor part, the bands under
    both rules). Writes partA_fits.csv."""
    raise NotImplementedError


def pair_kinds(out: Path) -> pd.DataFrame:
    """Part B: per kind of ester pair (ms's, the four nearest, the block's six) and the alcohol group, the KC data's
    distances and APL effect, C2's, the common error and its share of the squared sum; for C2, each of the eight wirings
    and their mean, raw and under the readout, against all eight recordings with APL output blocked (TNT), the six
    typical ones and in baseline-noise units (E76's tables). Writes pair_kinds.csv."""
    raise NotImplementedError


def per_odor_parts(out: Path, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Part B: the common error split into per-odor shares and a pair-specific remainder (two-way additive fit over
    the 120 pairs), with recording-bootstrap intervals, and each odor's share error against the model-minus-data gain
    of its response fraction when APL is blocked. Writes per_odor_parts.csv."""
    raise NotImplementedError


def inherited_parts(out: Path) -> pd.DataFrame:
    """Part B: for every archived variant with both conditions, its pattern error split into what it inherits from its
    APL-off distances, its on-against-off curve and a pair-specific part (`H.apl_effect_inherited_split`), the data's
    curve fitted in sample, leaving one odor out and on halves of the recordings. Writes inherited_parts.csv."""
    raise NotImplementedError


def archive_input_variants(out: Path) -> pd.DataFrame:
    """Part B: the archive's input-side variants (E3 part A's pn_inhadd sweep on each base, E3's bouton arm, E32,
    E38, E4's random wirings, E16's per-claw threshold and saturation variants) under their right names, each against
    its own base (projection on the common error, pattern error). Writes archive_input_variants.csv."""
    raise NotImplementedError


def pn_level(out: Path) -> pd.DataFrame:
    """Part B: the model's PN totals per odor and its PN-level odor-odor distances against the ORN input's and against
    Remy's PN-bouton and KC-claw matrices (the 66 megamat pairs), per fly where the data allow. Writes pn_level.csv."""
    raise NotImplementedError


def gate(out: Path) -> pd.DataFrame:
    """The gate on part C's screen (the entry's three conditions, inside the four bands). Writes gate.csv."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['A-input', 'A-score', 'B', 'C', 'gate', 'D', 'E', 'F'])
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'A-input':
        print(searched_input(out))
    elif args.part == 'A-score':
        score_fits(out)
    elif args.part == 'B':
        for f in (pair_kinds, per_odor_parts, inherited_parts, archive_input_variants, pn_level):
            f(out)
    elif args.part == 'C':
        import E77_input_screen as S
        S.screen(out)
    elif args.part == 'gate':
        gate(out)
    else:
        raise NotImplementedError(f'part {args.part}')


if __name__ == '__main__':
    main()
