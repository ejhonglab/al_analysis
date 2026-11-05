#!/usr/bin/env python3
"""Pulls the hemibrain connectivity the centrifugal-input experiment (E9) needs from neuPrint.

Queries hemibrain v1.2.1 (neuprint.janelia.org) for
  1. PN -> lateral-horn-neuron connections in the lateral horn (LH(R)): sources every neuron
     whose type contains "PN" (uniglomerular types such as DM1_lPN, DL1_adPN, with the
     glomerulus parsed from the type; multiglomerular M_* types kept and flagged), targets
     every type starting with "LH";
  2. lateral-horn centrifugal neurons (types LHCENT*) -> calyx (CA(R)) connections onto KCs,
     PNs (boutons) and APL;
  3. MB-C1 -> calyx connections onto the same targets;
  4. the inputs to the LHCENT* cells and MB-C1 from every traced neuron, in every ROI (what
     drives the centrifugal cells: lateral-horn neurons, PNs, others; `conn_to_centrifugal_all.csv`,
     with the presynaptic type and the ROI, so the experiment can build the LHN -> centrifugal
     weights and see what else converges on them);
plus the neuron tables (bodyId, type, instance, status, synapse counts) for the PNs, LHNs,
LHCENT neurons, MB-C1 and APL. Per-ROI synapse counts (weight) are saved unthresholded;
the experiment chooses its threshold (the model's claw data used >= 4-5 synapses).

Outputs (CSV + provenance.json) go to al_analysis/data/hemibrain_neuprint/, which is
committed, so the experiment itself needs no neuPrint access.

Credentials: neuprint-python reads the token from the environment variable
NEUPRINT_APPLICATION_CREDENTIALS (a token from neuprint.janelia.org, account menu > Auth
Token; set it in ~/.bash.local). The token is never written or printed by this script.

Usage: ./venv/bin/python scripts/claude_pair_remapping_experiments/pull_neuprint_connectivity.py [--out DIR] [--only NAME ...]
(--only restricts the pull to the named output files, e.g. --only conn_to_centrifugal_all.csv;
the provenance's row counts are merged with the existing file's.)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SERVER = 'https://neuprint.janelia.org'
DATASET = 'hemibrain:v1.2.1'
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[1] / 'al_analysis' / 'data' / 'hemibrain_neuprint'

PN_TYPE = r'.*PN.*'
LHN_TYPE = r'^LH.*'
LHCENT_TYPE = r'^LHCENT.*'
MBC1_TYPE = r'^MB-C1$'
CALYX_TARGET_TYPE = r'^KC.*|.*PN.*|^APL$'
ROI_LH, ROI_CA = 'LH(R)', 'CA(R)'


def token_check() -> None:
    tok = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS', '')
    if not tok:
        sys.exit('NEUPRINT_APPLICATION_CREDENTIALS is not set. Get a token at '
            f'{SERVER} (account menu > Auth Token) and put\n'
            "  export NEUPRINT_APPLICATION_CREDENTIALS='<token>'\n"
            'in ~/.bash.local (then open a new shell, or source it).')


def glomerulus_of(pn_type: str):
    """'DM1_lPN' -> 'DM1'; 'VA1d_adPN' -> 'VA1d'; 'M_adPNm4' (multiglomerular) -> None."""
    if not isinstance(pn_type, str) or pn_type.startswith('M_') or '_' not in pn_type:
        return None
    glom = pn_type.split('_')[0]
    return glom if re.match(r'^[A-Z]{1,3}[0-9]*[a-z]*(\+.*)?$', glom) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--only', nargs='+', default=None, help='only (re)pull these output files')
    args = ap.parse_args()
    token_check()
    import neuprint
    from neuprint import Client, NeuronCriteria as NC, fetch_neurons, fetch_adjacencies
    try:
        client = Client(SERVER, dataset=DATASET)
        server_version = client.fetch_version()
    except Exception as e:
        sys.exit(f'could not connect to neuPrint ({type(e).__name__}: {str(e)[:200]}). If this '
            'is an authentication error the token in NEUPRINT_APPLICATION_CREDENTIALS is '
            f'missing, expired or wrong: get a new one at {SERVER} (account menu > Auth Token) '
            'and update ~/.bash.local.')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    if args.only and (out / 'provenance.json').exists():
        counts = json.loads((out / 'provenance.json').read_text()).get('row_counts', {})

    def wanted(name: str) -> bool:
        return not args.only or name in args.only

    def save(df: pd.DataFrame, name: str) -> None:
        df.to_csv(out / name, index=False)
        counts[name] = len(df)
        print(f'{name}: {len(df)} rows', flush=True)

    neuron_cols = ['bodyId', 'type', 'instance', 'status', 'pre', 'post', 'size']
    if wanted('neurons_pn.csv'):
        pns, _ = fetch_neurons(NC(type=PN_TYPE, regex=True), client=client)
        pns = pns[[c for c in neuron_cols if c in pns.columns]].copy()
        pns['glomerulus'] = pns['type'].map(glomerulus_of)
        pns['uniglomerular'] = pns['glomerulus'].notna()
        save(pns.sort_values(['glomerulus', 'type', 'bodyId'], na_position='last'), 'neurons_pn.csv')
    if wanted('neurons_lhn.csv'):
        lhns, _ = fetch_neurons(NC(type=LHN_TYPE, regex=True), client=client)
        lhns = lhns[[c for c in neuron_cols if c in lhns.columns]].copy()
        lhns['centrifugal'] = lhns['type'].str.match(LHCENT_TYPE)
        save(lhns.sort_values(['type', 'bodyId']), 'neurons_lhn.csv')
    if wanted('neurons_centrifugal.csv'):
        cent, _ = fetch_neurons(NC(type=f'{LHCENT_TYPE}|{MBC1_TYPE}|^APL$', regex=True), client=client)
        cent = cent[[c for c in neuron_cols if c in cent.columns]].copy()
        save(cent.sort_values(['type', 'bodyId']), 'neurons_centrifugal.csv')

    def adjacencies(src_type, tgt_type, roi, name):
        if not wanted(name):
            return
        neurons, conn = fetch_adjacencies(NC(type=src_type, regex=True), NC(type=tgt_type, regex=True),
            rois=[roi], min_total_weight=1, client=client)
        types = neurons.set_index('bodyId')['type']
        conn = conn[conn['roi'] == roi].copy()
        conn['type_pre'] = conn['bodyId_pre'].map(types)
        conn['type_post'] = conn['bodyId_post'].map(types)
        conn = conn[['bodyId_pre', 'type_pre', 'bodyId_post', 'type_post', 'roi', 'weight']]
        save(conn.sort_values(['type_pre', 'bodyId_pre', 'weight'], ascending=[True, True, False]), name)

    adjacencies(PN_TYPE, LHN_TYPE, ROI_LH, 'conn_pn_to_lhn_LH.csv')
    adjacencies(LHCENT_TYPE, CALYX_TARGET_TYPE, ROI_CA, 'conn_lhcent_to_calyx_CA.csv')
    adjacencies(MBC1_TYPE, CALYX_TARGET_TYPE, ROI_CA, 'conn_mbc1_to_calyx_CA.csv')
    if wanted('conn_to_centrifugal_all.csv'):
        # every traced input to the centrifugal cells, all ROIs (sources=None: any neuron)
        neurons, conn = fetch_adjacencies(None, NC(type=f'{LHCENT_TYPE}|{MBC1_TYPE}', regex=True),
            min_total_weight=1, client=client)
        types = neurons.set_index('bodyId')['type']
        conn = conn.copy()
        conn['type_pre'] = conn['bodyId_pre'].map(types)
        conn['type_post'] = conn['bodyId_post'].map(types)
        conn = conn[['bodyId_pre', 'type_pre', 'bodyId_post', 'type_post', 'roi', 'weight']]
        save(conn.sort_values(['type_post', 'bodyId_post', 'weight'], ascending=[True, True, False]),
            'conn_to_centrifugal_all.csv')

    def git(args, cwd):
        try:
            return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), host=platform.node(),
        server=SERVER, dataset=DATASET, server_version=server_version,
        neuprint_python=getattr(neuprint, '__version__', None), script='pull_neuprint_connectivity.py',
        al_analysis_commit=git(['rev-parse', 'HEAD'], HERE),
        queries=dict(pn_types=PN_TYPE, lhn_types=LHN_TYPE, lhcent_types=LHCENT_TYPE, mbc1_type=MBC1_TYPE,
            calyx_targets=CALYX_TARGET_TYPE, rois=dict(pn_to_lhn=ROI_LH, centrifugal_to_calyx=ROI_CA,
                inputs_to_centrifugal='all ROIs'),
            weights='per-ROI synapse counts, unthresholded (min_total_weight=1)'),
        row_counts=counts,
    )
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n')
    print('done ->', out)


if __name__ == '__main__':
    main()
