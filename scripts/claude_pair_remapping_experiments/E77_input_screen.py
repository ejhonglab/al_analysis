"""E77 part C: the screen of the model's input handles on C2 (the breadth-2 APL-effect fit, E67) at fixed scales on the
real wiring, both response rates held, and the APL-on-only emulations (L1 / L2). Used by E77_input_side_common_error.py
(--part C); the design is the E77 entry's part C in experiment_priorities.md.

Static handles (S1-S6) act in both conditions: for each point the somatic threshold is bisected to C2's APL-off response
count, the breadth weights are recomputed from that run, and the APL->KC scale (KC->APL and bouton->APL with it) is
bisected to C2's APL-on response count (1852); KC-KC and APL->bouton scales unchanged. Emulations act only with APL on:
the APL-off run is C2's own and only the APL->KC scale is bisected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import kc_remapping_harness as H


def point_specs() -> List[dict]:
    """The screen's points: S1 (pn_inhadd 100, 314, 1000, 3141, pn_inhsc raised as E3 / E32), S2 (Hallem's input for the
    four odors together and one at a time, lab_scaled_ms, the searched input), S3 (each odor's positive ORN deltas to the
    power 0.5, 0.75, 1.33, 2, rescaled to the odor's sum), S4 (a per-claw tanh cap at 100, 150, 250 Hz, no per-claw
    threshold), S5 (glomerulus weights x (median claw count / its claw count)^0.5 and ^1, one permuted control; E20's
    drive normalization at 0.5 and 1), S6 (E18's bouton depletion at its two strongest settings, if its kwargs run on
    C2's simulator), L1 / L2 (APL->bouton / APL->KC weights x the bouton's / claw's own input in the odor, over its mean,
    to the power 1 and 2, mean 1; the better power's factors permuted within each odor as control)."""
    raise NotImplementedError


def odor_weight_factors(activity: pd.DataFrame, power: float, permute_seed: Optional[int] = None) -> pd.DataFrame:
    """Per-odor multiplicative factors for L1 / L2: (activity / its mean over units) ** power per odor, renormalized to
    mean 1 per odor; power 0 gives exactly 1; with `permute_seed` each odor's factors are permuted across units (the
    odor's set of factors kept)."""
    raise NotImplementedError


def screen(out: Path, only: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Runs (resumably) the screen's points on C2 and scores each as the entry's part C lists; writes screen.csv,
    screen_pairs.csv (per-pair effects, both readouts) and screen_params.json (each point's absolute parameters)."""
    raise NotImplementedError
