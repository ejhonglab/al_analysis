#!/usr/bin/env python3
"""E9: structured centrifugal input to the calyx (LHCENT / MB-C1 inhibition of KCs and boutons).

Interface stub (hal, 2026-11-05); bodies follow. Builds, from the hemibrain v1.2.1 tables in
al_analysis/data/hemibrain_neuprint/ (pull_neuprint_connectivity.py):
  1. a lateral-horn-neuron layer: LHN rate per odor = rectified sum over glomeruli of the
     PN -> LHN synapse count (uniglomerular PNs, summed per glomerulus) times the odor-evoked
     change of that glomerulus's PN rate in the base model (E21's glomerulus_rates_per_odor);
  2. centrifugal rates: LHCENT* and MB-C1 rate per odor = rectified sum over their LHN inputs
     (conn_to_centrifugal_all.csv) of synapse count times LHN rate, normalized per cell;
  3. their projection onto the model's KCs (ext_inh_kc: KC x odor from the LHCENT / MB-C1 ->
     KC synapse counts) and boutons (ext_inh_bouton: bouton x odor from their -> PN counts
     split evenly over each PN's boutons), handed to fit_mb_model(ext_inh_kc=, ext_inh_bouton=,
     ext_inh_scale=) (olfsysm rv.kc.ext_inh / rv.pn.bouton_ext_inh, step over the odor window).
Variants: bouton base (and the 70 Hz per-claw threshold base), centrifugal gain in
{0, small, medium, large} for KC targets, bouton targets and both, thresholds tuned at 5 %
with APL on, acute APL off. Scored as E24 / E26 plus the APL-independent pair effects the
report named (ep-6al, eb-t2h, paa-B-myr) and the APL-on excess profile.

Outputs: <shared>/results/E9_centrifugal_input/ and ./outputs/E9_centrifugal_input/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E9_centrifugal_input.py [--gains ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E9_centrifugal_input'
HEMIBRAIN = (H.HERE.parents[1] / 'al_analysis' / 'data' / 'hemibrain_neuprint').resolve()


def pn_to_lhn_weights(min_weight: int = 3) -> pd.DataFrame:
    """Glomerulus x LHN matrix of PN -> LHN synapse counts in LH(R) (uniglomerular PNs summed
    per glomerulus; connections below `min_weight` synapses dropped).
    """
    raise NotImplementedError


def lhn_rates(glom_rates: pd.DataFrame, w_pn_lhn: pd.DataFrame, spont_col=None) -> pd.DataFrame:
    """LHN x odor rates: rectified weighted sum of the odor-evoked PN rate changes."""
    raise NotImplementedError


def centrifugal_rates(lhn: pd.DataFrame, min_weight: int = 3) -> pd.DataFrame:
    """Centrifugal cell (LHCENT*, MB-C1 bodyIds) x odor rates from their LHN inputs."""
    raise NotImplementedError


def calyx_projection(cent: pd.DataFrame, wPNKC: pd.DataFrame, min_weight: int = 3):
    """(ext_inh_kc: KC x odor, ext_inh_bouton: bouton x odor) from the centrifugal cells'
    synapse counts onto the model's KCs and PNs (per-PN counts split over its boutons).
    """
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == '__main__':
    main()
