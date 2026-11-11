#!/usr/bin/env python3
"""The APL calyx maps against the connectome (E79), the connectome analyses 4 and 6 (the data-side analyses 1, 2, 3 and 5
are in E79_apl_calyx_constraints.py). No model fits.

Inputs:
- Yang's own Claude's connectome tables (separate work, in progress; tensor Tom/claude_APL_modeling_YZ4TC/
  Claude_APL_analysis/apl_synapses/): `<dataset>_PN_boutons_derived.csv.gz` (bouton centroids in nm, n_tbars, pn_type),
  `<dataset>_KC_claws_derived.csv.gz` and `<dataset>_APL_MB_synapses.csv.gz` for hemibrain, FlyWire, male CNS and BANC.
  The seven calyces are FlyWire L / R, male CNS L / R, BANC L / R and hemibrain R; the eighth, `model_hemibrain_R`, is
  the model's own hemibrain boutons (Pratyush's v5 bouton segmentation, which `prat_boutons=True` uses: every PN T-bar in
  CA(R) with its bouton), with the hemibrain's APL synapses for its outline.
- The model's PN rates per glomerulus: the APL16 ORN input through the PN stage of the best family (the breadth-2
  APL-effect fit, C2, E67's `rate0.068_aplboth_k2_cap3`, built by `E77_input_screen.build_c2_simulator`, APL off; the PN
  stage does not depend on APL), evoked rates (the 0-0.5 s mean of olfsysm's pn_sims minus the -0.5-0 s mean), cached in
  connectome_pn_rates.csv. The ORN input is at the APL16 panel's concentrations (pa, ms -3; paa at the substituted -4),
  not at the APL maps' (pa -6.8 to -4.4, paa -6.3 to -3.0, ms -7.7 to -2); pa's map concentrations are all more than one
  log10 unit from the input's, so every pa comparison is outside the 1-log rule.
- C2's APL-on spike counts per KC (the model-output archive) for the spike-driven alternative.
- The measured maps (data/yang_apl_calyx_maps/, apl_calyx_maps.py's ROI reader and stimulus maps: each stimulus's mean
  map minus the mean of all the recording's paraffin oil presentations).
- The error common to all wirings (results/E72_wiring_population/common_error_pairs.csv, column `common`: per odor pair,
  the APL effect of the mean over E72's eight wirings minus the data's, both means removed, calibrated readout).

Axis conventions (native nm to fly's right R, dorsal D, anterior A in um), Yang's Claude's posterior-view axes
(geom.to_rda) with the anterior-posterior sign set from the lobes: hemibrain (-x, -z, +y), FlyWire (x, -y, -z) with sides
from FlyWire's labels, male CNS (-x, -y, -z), BANC (-x, +z, -y). `axis_checks` verifies them per mushroom body from the
APL synapse table's ROIs (lobes anterior of the calyx, the vertical lobe's tip and the calyx dorsal of the peduncle, the
medial lobe's tip medial of the calyx, the right mushroom body at larger R) and writes connectome_axis_checks.csv.
Posterior view: R to the right, D up; a left calyx compared with a right-hemisphere recording (and the reverse) is mirrored
(R -> -R), so a flip of the measured image means the same thing for both hemispheres (left-right symmetry assumed).

Analyses (outputs to results/E79_apl_calyx_constraints/ in the shared directory, files connectome_* and coloc_*, figures
E79_connectome_* / E79_coloc_*, provenance_connectome.json, connectome_table.md):
4. Connectome-predicted APL maps per calyx: per odor, sum over boutons of n_tbars x the glomerulus's evoked PN rate at the
   bouton centroid, in slabs 6 um thick along the anterior-posterior axis in 3 um steps, Gaussian blur 2, 4 and 8 um, on
   a 1 um raster in posterior view; the slab's calyx outline is the region holding 97.5 % of its APL calyx synapses
   (their density blurred 2 um, holes filled, largest component); maps are sampled on a 12 x 12 grid over the outline's
   bounding box (cell = mean over its raster pixels inside the outline; a cell counts as inside when at least half its
   pixels are); for the differences each odor's predicted map is divided by its mean inside the outline first, as the
   measured maps are. Alternatives: claw-driven (claw centroids x the claw's PN synapses x its main PN's glomerulus's
   rate),
   spike-driven (KC->APL calyx synapses x C2's APL-on spike count of the KC; hemibrain only) and anatomy only (unweighted
   PN<->APL calyx synapses; scored on the single-odor maps only, since it predicts no difference between odors).
   Measured: pa minus ms and paa minus ms at matched calyx-mean dF/F (among the pairs of concentrations with both
   responses at least a quarter of the odor's largest in that recording, the strongest pair whose means are within 1.25x
   of each other, else the pair closest in ratio; each map over its mean, so the difference is the shape difference at
   exactly matched means), and each odor's map at its highest concentration,
   blurred 2 px, on the same 12 x 12 grid over the ROI's bounding box. Score: Pearson r over the cells inside both
   outlines, for the four flips of the measured image, every slab (and the whole calyx projected, slab_depth_um -1),
   blur, calyx and recording; null: 1,000 permutations of
   the glomerulus labels of the rate vector (one set per calyx, the same for every slab, blur and recording; KC labels for
   the spike-driven maps), giving each score's percentile, the percentile of the best score (over slabs, blurs and flips)
   against the same maximum in each permutation, and a pooled test over recordings at a common flip.
6. Glomerulus co-localization C_gh(lambda), the n_tbars-weighted mean over bouton pairs (one of g, one of h) of
   exp(-d^2 / 2 lambda^2), lambda 2, 4, 8, 16 um (3D distances), per calyx, on the glomeruli with ORN input (the 36 of
   the model's input) present in every calyx; agreement between calyces (Pearson r and Spearman rho of the off-diagonal entries); whether the oracle
   glomeruli (D, DC2, DL1, DM1, DM5, DP1m, VA7m, VC1) and two ester-group sets are nearer each other than random sets of
   the same size (10,000 draws); each APL16 odor's spatial concentration x'Cx / (sum x)^2 (x its evoked PN rates, negatives
   set to 0). The proxy: per odor pair, the cosine distance of the PN-rate vectors when each glomerulus's inhibition
   follows its local field Cx minus that when it follows the global mean, subtractive (relu(x - I)) and divisive (x / (1
   + I / mu), mu the mean rate over odors and glomeruli), two strengths each, the local field's scale fitted so that the
   mean inhibition over odors and glomeruli is the same; correlated over the 120 pairs with the common error; the share of
   the common error's squared sum removed by the best linear scaling of the (centered) change, fitted on the pairs of 15
   odors and scored on the 15 pairs of the left-out one (pooled over the 16); null: C permuted by glomerulus (rows and
   columns together, the same permutation for every calyx and setting), per setting and for the best setting. The entry's
   gate for outcome (a): 15 % or more removed on left-out odors beyond the null.

Run from this folder: `OMP_NUM_THREADS=6 OMP_THREAD_LIMIT=6 nice -n 10 ../../venv/bin/python E79_connectome_maps.py
[--analyses 4,6] [--n-perm 1000]`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import binary_fill_holes, gaussian_filter, label

import apl_calyx_maps as ACM

HERE = Path(__file__).resolve().parent
SHARED = ACM.SHARED
OUT = SHARED / 'results' / 'E79_apl_calyx_constraints'
YANG_APL = Path('/mnt/tensor/Tom/claude_APL_modeling_YZ4TC/Claude_APL_analysis/apl_synapses')
PRAT_DIR = HERE.parents[1] / 'al_analysis' / 'data' / 'from_pratyush' / '2025-12-05'
PRAT_PN_SYNAPSES = PRAT_DIR / 'PN_Synapse_Assignments_Hemibrain_2025-12-05-13-39-22.parquet'
PRAT_PN_KC = PRAT_DIR / 'PNbouton-to-KCclaw_Connectivity_2025-12-09-19-47-36.parquet'
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
C2_ON_COUNTS = SHARED / 'data' / 'model_outputs' / C2[0] / f'{C2[1]}_on_spike_counts.parquet'
COMMON_ERROR = SHARED / 'results' / 'E72_wiring_population' / 'common_error_pairs.csv'
# the model's ORN input (kc_remapping_harness.apl16_orn_deltas() as archived); its glomeruli are the driven ones
ORN_INPUT = SHARED / 'data' / 'model_outputs' / 'inputs' / 'orn_deltas_apl16.csv'
DATASETS: Tuple[str, ...] = ('hemibrain', 'flywire', 'malecns', 'banc')
MAP_ODORS: Tuple[str, ...] = ('pa', 'paa', 'ms')
DIFFS: Tuple[Tuple[str, str], ...] = (('pa', 'ms'), ('paa', 'ms'))
COMPARISONS: Tuple[str, ...] = ('pa-ms', 'paa-ms', 'pa', 'paa', 'ms')
# the model's ORN input concentration of each map odor (log10), for the 1-log rule
INPUT_CONC = {'pa': -3.0, 'ms': -3.0, 'paa': -4.0}
SLAB_THICK_UM = 6.0
SLAB_STEP_UM = 3.0
SLAB_MIN_SHARE = 0.02
# slab_depth_um of the whole calyx projected (every A), scored beside the slabs
WHOLE_CALYX = -1.0
BLURS_UM: Tuple[float, ...] = (2.0, 4.0, 8.0)
PX_UM = 1.0
OUTLINE_BLUR_UM = 2.0
OUTLINE_KEEP = 0.975
GRID_N = 12
CELL_INSIDE = 0.5
MEAS_BLUR_PX = ACM.BLUR
MATCH_MIN_FRACTION = 0.25
MATCH_MAX_RATIO = 1.25
FLIPS: Tuple[str, ...] = ('none', 'horizontal', 'vertical', 'both')
ALTERNATIVES: Tuple[str, ...] = ('bouton', 'claw', 'spike', 'anatomy')
LAMBDAS_UM: Tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
GLOMERULUS_SETS: Dict[str, Tuple[str, ...]] = {
    # the oracle gain's eight glomeruli (the weak-input odors' most-driven glomeruli; the summary's "oracle glomeruli")
    'oracle_eight': ('D', 'DC2', 'DL1', 'DM1', 'DM5', 'DP1m', 'VA7m', 'VC1'),
    # C2's four co-responder glomeruli of the ester pairs (E75)
    'c2_coresponders': ('DM2', 'DM3', 'DM5', 'DM6'),
    # the ester group's shared glomeruli (the E79 entry)
    'ester_shared': ('DM1', 'DM2', 'DM3', 'VM2'),
}
N_SET_DRAWS = 10000
PROXY_SETTINGS: Tuple[Tuple[str, float], ...] = (('sub', 0.5), ('sub', 1.0), ('div', 1.0), ('div', 4.0))
N_PERM = 1000
SEED = 0
GATE_SHARE = 0.15
GATE_PCT = 95.0
# FlyWire / male CNS / BANC glomerulus names (Schlegel et al. 2024) of hemibrain VC3m and VC5, into the model's names (the
# model merges hemibrain VC3l and VC3m into VC3, mb_model.glomerulus_renames)
FLYWIRE_TO_MODEL = {'VC5': 'VC3', 'VM6': 'VC5', 'VM6v': 'VC5', 'VM6m': 'VC5', 'VM6l': 'VC5'}
HEMIBRAIN_TO_MODEL = {'VC3l': 'VC3', 'VC3m': 'VC3'}


# ---------------------------------------------------------------------------
# geometry and tables

def to_rda(dataset: str, x, y, z) -> np.ndarray:
    """Native nm -> (fly's right, dorsal, anterior) in um (posterior view: R to the right, D up; A positive anterior)."""
    x, y, z = np.asarray(x, float) / 1e3, np.asarray(y, float) / 1e3, np.asarray(z, float) / 1e3
    return {'flywire': np.c_[x, -y, -z], 'hemibrain': np.c_[-x, -z, y], 'malecns': np.c_[-x, -y, -z],
            'banc': np.c_[-x, z, -y]}[dataset]


def _side_of(a: pd.DataFrame, dataset: str) -> pd.Series:
    if dataset == 'hemibrain':
        return pd.Series('R', index=a.index)
    return a['apl_side'].astype(str).str[0].str.upper()


def _apl_table(dataset: str, regions: Optional[Sequence[str]] = ('CA',)) -> pd.DataFrame:
    """The dataset's APL MB synapses with side, partner kind ('PN', 'KC', 'other'), partner id, region and the synapse's
    position (midpoint of the pre and post sites) as R, D, A; synapses whose ROI side differs from the APL's dropped."""
    cols = ['direction', 'apl_side', 'partner_id', 'partner_class', 'roi_official', 'mb_region',
            'pre_x_nm', 'pre_y_nm', 'pre_z_nm', 'post_x_nm', 'post_y_nm', 'post_z_nm']
    a = pd.read_csv(YANG_APL / f'{dataset}_APL_MB_synapses.csv.gz', usecols=cols, dtype={'partner_id': str},
                    low_memory=False)
    side = _side_of(a, dataset)
    roi_side = a['roi_official'].astype(str).str.extract(r'(?:_|\()([LR])\)?$')[0]
    keep = (roi_side == side).to_numpy().copy()
    if regions is not None:
        keep &= a['mb_region'].isin(regions).to_numpy()
    a, side = a[keep], side[keep]
    P = to_rda(dataset, (a.pre_x_nm + a.post_x_nm) / 2, (a.pre_y_nm + a.post_y_nm) / 2, (a.pre_z_nm + a.post_z_nm) / 2)
    cls = a['partner_class'].astype(str).str.lower()
    kind = np.where(cls.isin(['alpn', 'antennal_lobe_projection_neuron']), 'PN',
                    np.where(cls.str.contains('kenyon'), 'KC', 'other'))
    return pd.DataFrame({'side': side.to_numpy(), 'kind': kind, 'partner_id': a['partner_id'].to_numpy(),
                         'apl_input': a['direction'].str.startswith('APL_input').to_numpy(),
                         'region': a['mb_region'].to_numpy(), 'R': P[:, 0], 'D': P[:, 1], 'A': P[:, 2]})


def axis_checks() -> pd.DataFrame:
    """Per mushroom body, the landmark checks of to_rda's axes, from each dataset's APL MB synapses: the lobes anterior of
    the calyx, the vertical lobe's tip (the 2 % of its synapses farthest from the peduncle's centroid) and the calyx
    dorsal of the peduncle, the medial lobe's tip medial of the calyx, the right mushroom body at larger R."""
    rows = []
    for ds in DATASETS:
        t = _apl_table(ds, regions=None)
        t['lobe'] = t['region'].replace({'aL': 'VL', "a'L": 'VL', 'bL': 'ML', "b'L": 'ML', 'gL': 'ML'})
        ca_r = {}
        for side, g in t.groupby('side'):
            c = {r: g.loc[g['lobe'] == r, ['R', 'D', 'A']].mean().to_numpy() for r in ('CA', 'PED', 'VL', 'ML')}

            def tip(r):
                P = g.loc[g['lobe'] == r, ['R', 'D', 'A']].to_numpy()
                d = np.linalg.norm(P - c['PED'], axis=1)
                return P[d >= np.quantile(d, 0.98)].mean(axis=0)
            vt, mt = tip('VL'), tip('ML')
            medial = -1.0 if side == 'R' else 1.0
            ca_r[side] = c['CA'][0]
            rows.append(dict(dataset=ds, side=side, n_synapses=len(g),
                             lobes_minus_calyx_A=float(np.mean([c['VL'][2], c['ML'][2]]) - c['CA'][2]),
                             vl_tip_minus_ped_D=float(vt[1] - c['PED'][1]), calyx_minus_ped_D=float(c['CA'][1] - c['PED'][1]),
                             ml_tip_medial_of_calyx_um=float(medial * (mt[0] - c['CA'][0]))))
        for r in rows:
            if r['dataset'] == ds:
                r['right_minus_left_calyx_R'] = float(ca_r['R'] - ca_r['L']) if len(ca_r) == 2 else np.nan
    df = pd.DataFrame(rows)
    # the same transform as Yang's Claude's posterior-view maps: R, D against its view_right_um / view_dorsal_um
    view = YANG_APL / 'odor_weighted' / 'PN_APL_synapses_odor_weighted.csv'
    if view.exists():
        v = pd.read_csv(view, usecols=['dataset', 'x_nm', 'y_nm', 'z_nm', 'view_right_um', 'view_dorsal_um'])
        for ds, g in v.groupby('dataset'):
            P = to_rda(ds, g.x_nm, g.y_nm, g.z_nm)
            df.loc[df.dataset == ds, 'max_abs_diff_vs_yang_view_R_um'] = float(np.abs(P[:, 0] - g.view_right_um).max())
            df.loc[df.dataset == ds, 'max_abs_diff_vs_yang_view_D_um'] = float(np.abs(P[:, 1] - g.view_dorsal_um).max())
    df['passed'] = ((df.lobes_minus_calyx_A > 0) & (df.vl_tip_minus_ped_D > 0) & (df.calyx_minus_ped_D > 0)
                    & (df.ml_tip_medial_of_calyx_um > 0) & ~(df.right_minus_left_calyx_R <= 0))
    return df


def glomerulus_of_type(pn_type, dataset: str) -> Optional[str]:
    """The model's glomerulus name of a uniglomerular PN type ('DM1_lPN' -> 'DM1'; hemibrain VC3l / VC3m -> VC3; FlyWire
    names VC5 -> VC3 and VM6 -> VC5), None for multiglomerular (M_), bi-glomerular ('+'), untyped or non-PN types."""
    if not isinstance(pn_type, str) or '_' not in pn_type:
        return None
    g = pn_type.split('_')[0]
    if '+' in g or g in ('M', 'MZ', '') or g.startswith('WED'):
        return None
    ren = HEMIBRAIN_TO_MODEL if dataset in ('hemibrain', 'model_hemibrain') else FLYWIRE_TO_MODEL
    return ren.get(g, g)


def _assign_sides(P: np.ndarray, centers: Dict[str, float]) -> np.ndarray:
    sides = list(centers)
    d = np.abs(P[:, [0]] - np.array([centers[s] for s in sides])[None, :])
    return np.array(sides)[d.argmin(axis=1)]


def _prat_boutons() -> pd.DataFrame:
    """The model's own hemibrain boutons: Pratyush's v5 PN T-bars in CA(R) grouped by (PN, anatomical bouton); PN types
    from his PN->KC table."""
    s = pd.read_parquet(PRAT_PN_SYNAPSES)
    s = s[(s['type'] == 'pre') & (s['roi'] == 'CA(R)') & (s['anatomical_bouton'] >= 0)]
    types = pd.read_parquet(PRAT_PN_KC, columns=['bodyId_pre', 'type_pre']).drop_duplicates('bodyId_pre')
    types = types.set_index('bodyId_pre')['type_pre']
    s = s.assign(pn_type=s['neuron'].map(types))
    P = to_rda('hemibrain', s['x'] * 8.0, s['y'] * 8.0, s['z'] * 8.0)
    s = s.assign(R=P[:, 0], D=P[:, 1], A=P[:, 2])
    b = s.groupby(['neuron', 'anatomical_bouton']).agg(n_tbars=('R', 'size'), R=('R', 'mean'), D=('D', 'mean'),
                                                       A=('A', 'mean'), pn_type=('pn_type', 'first')).reset_index()
    b['bouton_id'] = b['neuron'].astype(str) + '_' + b['anatomical_bouton'].astype(int).astype(str)
    b['pn_id'] = b['neuron'].astype(str)
    b['glomerulus'] = [glomerulus_of_type(t, 'model_hemibrain') for t in b['pn_type']]
    return b[['bouton_id', 'pn_id', 'pn_type', 'glomerulus', 'n_tbars', 'R', 'D', 'A']]


def load_calyces() -> Dict[str, Dict[str, pd.DataFrame]]:
    """calyx name ('<dataset>_<side>', and 'model_hemibrain_R') -> {'boutons', 'claws', 'apl'}: tables with R, D, A (um)
    and, for boutons and claws, the model's glomerulus name and a weight (n_tbars; the claw's PN synapses)."""
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for ds in DATASETS:
        apl = _apl_table(ds)
        centers = apl.groupby('side')['R'].mean().to_dict()
        b = pd.read_csv(YANG_APL / f'{ds}_PN_boutons_derived.csv.gz', dtype={'pn_id': str, 'bouton_id': str})
        P = to_rda(ds, b.centroid_x_nm, b.centroid_y_nm, b.centroid_z_nm)
        b = b.assign(R=P[:, 0], D=P[:, 1], A=P[:, 2], side=_assign_sides(P, centers),
                     glomerulus=[glomerulus_of_type(t, ds) for t in b['pn_type']])
        c = pd.read_csv(YANG_APL / f'{ds}_KC_claws_derived.csv.gz', dtype={'kc_id': str, 'claw_id': str})
        P = to_rda(ds, c.centroid_x_nm, c.centroid_y_nm, c.centroid_z_nm)
        c = c.assign(R=P[:, 0], D=P[:, 1], A=P[:, 2], side=_assign_sides(P, centers),
                     glomerulus=[glomerulus_of_type(t, ds) for t in c['main_PN_type']])
        for side in sorted(centers):
            out[f'{ds}_{side}'] = dict(boutons=b[b['side'] == side].reset_index(drop=True),
                                       claws=c[c['side'] == side].reset_index(drop=True),
                                       apl=apl[apl['side'] == side].reset_index(drop=True))
    out['model_hemibrain_R'] = dict(boutons=_prat_boutons(), claws=None, apl=out['hemibrain_R']['apl'])
    return out


def calyx_side(calyx: str) -> str:
    return calyx[-1]


# ---------------------------------------------------------------------------
# model inputs

def model_pn_rates(rebuild: bool = False) -> pd.DataFrame:
    """Glomerulus x odor (16 APL16 odors, abbreviations) evoked PN rates of C2's PN stage (Hz; the 0-0.5 s mean of
    pn_sims minus the -0.5-0 s mean); cached in connectome_pn_rates.csv (with the total rates in
    connectome_pn_rates_total.csv)."""
    path = OUT / 'connectome_pn_rates.csv'
    if path.exists() and not rebuild:
        return pd.read_csv(path, index_col=0)
    import kc_remapping_harness as H
    import E77_input_screen as S
    from al_analysis.mb_model import get_sim_var, get_time_index
    sim, info = S.build_c2_simulator()
    r, _ = sim(dict(info['params'], wAPLKC_scale=0.0, wAPLPN_scale=0.0))
    runner = S.PerOdorRunner(sim, info)
    t = np.asarray(get_time_index(runner.mp, runner.rv))
    pn = get_sim_var(runner.rv.pn, 'pn_sims')
    gl = list(dict.fromkeys(runner.W.columns.get_level_values('glomerulus')))
    win = (t >= S.PN_WINDOW[0]) & (t <= S.PN_WINDOW[1])
    pre = (t >= -0.5) & (t < 0.0)
    tot = np.stack([np.asarray(pn[int(j)])[:, win].mean(axis=1) for j in runner.order], axis=1)
    base = np.stack([np.asarray(pn[int(j)])[:, pre].mean(axis=1) for j in runner.order], axis=1)
    if np.asarray(pn[0]).shape[0] != len(gl):
        raise RuntimeError(f'{np.asarray(pn[0]).shape[0]} PN rows for {len(gl)} glomeruli in the wiring')
    odors = list(H.abbrev_columns(r).columns)
    total = pd.DataFrame(tot, index=pd.Index(gl, name='glomerulus'), columns=odors)
    evoked = pd.DataFrame(tot - base, index=total.index, columns=odors)
    OUT.mkdir(parents=True, exist_ok=True)
    total.to_csv(OUT / 'connectome_pn_rates_total.csv')
    evoked.to_csv(path)
    return evoked


def c2_spike_counts() -> pd.DataFrame:
    """C2's APL-on spike counts, KC body id (str) x odor abbreviation."""
    s = pd.read_parquet(C2_ON_COUNTS)
    s.index = s.index.get_level_values('kc_id').astype(str)
    s.columns = [c.split(' @ ')[0] for c in s.columns]
    return s.astype(float)


def common_error() -> pd.Series:
    """The error common to all wirings per odor pair (canonical sorted pairs), E72's `common` column."""
    e = pd.read_csv(COMMON_ERROR)
    idx = pd.MultiIndex.from_tuples([tuple(sorted(p)) for p in zip(e['odor_a'], e['odor_b'])], names=['odor_a', 'odor_b'])
    return pd.Series(e['common'].to_numpy(dtype=float), index=idx)


# ---------------------------------------------------------------------------
# rasters and grids

def cell_matrix(mask: np.ndarray, n: int = GRID_N) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """(pixels x n*n averaging matrix over the mask's pixels in each cell of the n x n grid over the mask's bounding box,
    cells-inside flags: at least CELL_INSIDE of the cell's pixels in the mask). Rows index the flattened image."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cy = np.minimum((yy - y0) * n // (y1 - y0), n - 1)
    cx = np.minimum((xx - x0) * n // (x1 - x0), n - 1)
    cell = (cy * n + cx).ravel()
    pix = (yy * mask.shape[1] + xx).ravel()
    inm = mask[yy, xx].ravel()
    n_all = np.bincount(cell, minlength=n * n)
    n_in = np.bincount(cell[inm], minlength=n * n)
    M = sparse.csr_matrix((np.ones(int(inm.sum())), (pix[inm], cell[inm])), shape=(mask.size, n * n))
    M = M @ sparse.diags(1.0 / np.maximum(n_in, 1))
    return M.tocsr(), (n_in >= CELL_INSIDE * n_all) & (n_in > 0)


def rasterize(R: np.ndarray, D: np.ndarray, cols: np.ndarray, w: np.ndarray, n_cols: int, frame: dict) -> np.ndarray:
    """n_cols x H x W image stack: the weights `w` summed per raster pixel for each column index in `cols` (row 0 = most
    dorsal, column 0 = most left in posterior view)."""
    H_, W_ = frame['shape']
    iy = np.clip(((frame['D1'] - D) / PX_UM).astype(int), 0, H_ - 1)
    ix = np.clip(((R - frame['R0']) / PX_UM).astype(int), 0, W_ - 1)
    flat = np.bincount(cols * (H_ * W_) + iy * W_ + ix, weights=w, minlength=n_cols * H_ * W_)
    return flat.reshape(n_cols, H_, W_)


def calyx_frame(apl: pd.DataFrame, pad_um: float = 3 * max(BLURS_UM)) -> dict:
    R0, R1 = apl['R'].min() - pad_um, apl['R'].max() + pad_um
    D0, D1 = apl['D'].min() - pad_um, apl['D'].max() + pad_um
    return dict(R0=R0, D1=D1, shape=(int(np.ceil((D1 - D0) / PX_UM)) + 1, int(np.ceil((R1 - R0) / PX_UM)) + 1))


def outline_mask(R: np.ndarray, D: np.ndarray, frame: dict) -> np.ndarray:
    """The region holding OUTLINE_KEEP of the points: their density blurred OUTLINE_BLUR_UM, thresholded at the density
    of the (1 - OUTLINE_KEEP) quantile point, holes filled, the largest connected component."""
    img = gaussian_filter(rasterize(R, D, np.zeros(len(R), int), np.ones(len(R)), 1, frame)[0], OUTLINE_BLUR_UM / PX_UM)
    H_, W_ = frame['shape']
    iy = np.clip(((frame['D1'] - D) / PX_UM).astype(int), 0, H_ - 1)
    ix = np.clip(((R - frame['R0']) / PX_UM).astype(int), 0, W_ - 1)
    thr = np.quantile(img[iy, ix], 1 - OUTLINE_KEEP)
    m = binary_fill_holes(img >= thr)
    lab, n = label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        m = lab == sizes.argmax()
    return m


def slab_centers(apl: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """(slab centers along A, the calyx's posterior edge: the 0.5th percentile of its APL synapses' A)."""
    a0, a1 = np.quantile(apl['A'], [0.005, 0.995])
    return np.arange(a0 + SLAB_THICK_UM / 2, a1 - SLAB_THICK_UM / 2 + 1e-9, SLAB_STEP_UM), float(a0)


def flip_grid(g: np.ndarray, flip: str) -> np.ndarray:
    """A 12 x 12 grid (or ... x 12 x 12; or flattened ... x 144) flipped horizontally, vertically, both or not at all."""
    flat = g.shape[-1] == GRID_N * GRID_N
    x = g.reshape(g.shape[:-1] + (GRID_N, GRID_N)) if flat else g
    if flip in ('horizontal', 'both'):
        x = x[..., :, ::-1]
    if flip in ('vertical', 'both'):
        x = x[..., ::-1, :]
    return x.reshape(g.shape) if flat else x


def masked_pearson(pred: np.ndarray, meas: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Pearson r over `inside` cells between each row of `pred` (n x 144) and `meas` (144,)."""
    X = pred[:, inside]
    y = meas[inside]
    X = X - X.mean(axis=1, keepdims=True)
    y = y - y.mean()
    den = np.sqrt((X ** 2).sum(axis=1) * (y ** 2).sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, X @ y / den, np.nan)


# ---------------------------------------------------------------------------
# measured maps

def measured_maps(baseline: str = 'all_pfo') -> Dict[str, dict]:
    """recording -> {'grid': {comparison: 144 map}, 'inside': 144 bool, 'concs': {comparison: chosen concentrations},
    'means': ..., 'side'}: the single-odor maps at their highest concentration (each over its calyx mean), and pa minus
    ms, paa minus ms at matched calyx-mean dF/F (each map over its mean), blurred MEAS_BLUR_PX, on the 12 x 12 grid over
    the ROI's bounding box (image axes as recorded). `baseline` 'all_pfo': apl_calyx_maps.py's maps (each stimulus minus
    the mean of all paraffin oil presentations); any other value is passed to E79_apl_calyx_constraints.stimulus_maps
    (e.g. 'order_matched', which also leaves out its doubtful presentations)."""
    import tifffile
    recs = pd.read_csv(ACM.DATA / 'recordings.csv')
    frames_all = pd.read_csv(ACM.DATA / 'frames.csv')
    out = {}
    for _, rec in recs.iterrows():
        name = rec['file'].split('_moco')[0]
        stack = tifffile.imread(ACM.DATA / rec['file']).astype(float)
        frames = frames_all[frames_all['recording'] == name].set_index('frame').sort_index()
        mask = ACM.roi_mask(ACM.read_imagej_polygon(ACM.DATA / rec['roi']), stack.shape[1:])
        if baseline == 'all_pfo':
            maps = ACM.stimulus_maps(stack, frames)
        else:
            import E79_apl_calyx_constraints as DS
            maps = DS.stimulus_maps(stack, DS.add_positions(frames), baseline)
        M, inside = cell_matrix(mask)
        means = {k: float(v[0][mask].mean()) for k, v in maps.items()}

        def grid(k):
            m = gaussian_filter(maps[k][0], MEAS_BLUR_PX) / means[k]
            return np.asarray(M.T @ m.ravel()).ravel()
        g, concs, used = {}, {}, {}
        for odor in MAP_ODORS:
            k = max((kk for kk in maps if kk[0] == odor), key=lambda kk: kk[1])
            g[odor], concs[odor], used[odor] = grid(k), {odor: k[1]}, {odor: means[k]}
        for a, b in DIFFS:
            ka = [k for k in maps if k[0] == a]
            kb = [k for k in maps if k[0] == b]
            ma = max(means[k] for k in ka)
            mb = max(means[k] for k in kb)
            cand = [(abs(np.log(means[x] / means[y])), min(means[x], means[y]), x, y) for x in ka for y in kb
                    if means[x] >= MATCH_MIN_FRACTION * ma and means[y] >= MATCH_MIN_FRACTION * mb
                    and means[x] > 0 and means[y] > 0]
            close = [c for c in cand if c[0] <= np.log(MATCH_MAX_RATIO)]
            # the strongest pair within the ratio, else the closest in ratio
            _, _, x, y = max(close, key=lambda c: c[1]) if close else min(cand)
            g[f'{a}-{b}'] = grid(x) - grid(y)
            concs[f'{a}-{b}'] = {a: x[1], b: y[1]}
            used[f'{a}-{b}'] = {a: means[x], b: means[y]}
        out[name] = dict(grid=g, inside=inside, concs=concs, means=used, side=rec['side'], mask=mask,
                         maps={k: v[0] for k, v in maps.items()})
    return out


# ---------------------------------------------------------------------------
# analysis 4

def _alternative_points(calyx: str, parts: dict, alt: str, rates: pd.DataFrame,
                        counts: Optional[pd.DataFrame]) -> Optional[Tuple[pd.DataFrame, List[str], np.ndarray]]:
    """(points with R, D, A, a column index 'col' and weight 'w'; column names; columns x odor weight matrix for the
    map odors) for an alternative, or None when the calyx does not have it."""
    if alt == 'bouton':
        p = parts['boutons'].dropna(subset=['glomerulus']).assign(w=lambda d: d['n_tbars'].astype(float))
        key = 'glomerulus'
    elif alt == 'claw':
        if parts['claws'] is None:
            return None
        p = parts['claws'].dropna(subset=['glomerulus']).assign(w=lambda d: d['n_PN_synapses'].astype(float))
        key = 'glomerulus'
    elif alt == 'spike':
        if calyx != 'hemibrain_R' or counts is None:
            return None
        a = parts['apl']
        p = a[(a['kind'] == 'KC') & a['apl_input'] & a['partner_id'].isin(counts.index)].assign(w=1.0)
        key = 'partner_id'
    elif alt == 'anatomy':
        a = parts['apl']
        p = a[a['kind'] == 'PN'].assign(w=1.0, col_name='all')
        key = 'col_name'
    else:
        raise ValueError(alt)
    names = sorted(p[key].unique())
    p = p.assign(col=pd.Categorical(p[key], categories=names).codes.astype(np.int64))
    if alt in ('bouton', 'claw'):
        Wt = rates.reindex(names).fillna(0.0)[list(MAP_ODORS)].to_numpy(dtype=float)
    elif alt == 'spike':
        Wt = counts.loc[names, list(MAP_ODORS)].to_numpy(dtype=float)
    else:
        Wt = np.ones((1, len(MAP_ODORS)))
    return p[['R', 'D', 'A', 'col', 'w']], names, Wt


def slab_column_grids(points: pd.DataFrame, n_cols: int, apl: pd.DataFrame, frame: dict, center: Optional[float]
                      ) -> Optional[Tuple[Dict[float, np.ndarray], np.ndarray, np.ndarray]]:
    """For one slab (`center` None: the whole calyx): ({blur: columns x 144 grid of each column's blurred map}, inside
    flags, outline mask), or None when the slab holds no points."""
    lo, hi = ((-np.inf, np.inf) if center is None else (center - SLAB_THICK_UM / 2, center + SLAB_THICK_UM / 2))
    a = apl[(apl['A'] >= lo) & (apl['A'] < hi)]
    p = points[(points['A'] >= lo) & (points['A'] < hi)]
    if len(p) == 0 or len(a) < 50:
        return None
    mask = outline_mask(a['R'].to_numpy(), a['D'].to_numpy(), frame)
    M, inside = cell_matrix(mask)
    used = np.unique(p['col'].to_numpy())
    remap = np.full(n_cols, -1)
    remap[used] = np.arange(len(used))
    img = rasterize(p['R'].to_numpy(), p['D'].to_numpy(), remap[p['col'].to_numpy()], p['w'].to_numpy(), len(used),
                    frame)
    grids = {}
    for blur in BLURS_UM:
        bl = gaussian_filter(img, (0, blur / PX_UM, blur / PX_UM))
        g = np.zeros((n_cols, GRID_N * GRID_N))
        g[used] = np.asarray((M.T @ bl.reshape(len(used), -1).T).T)
        grids[blur] = g
    return grids, inside, mask


def predicted_comparisons(Wt: np.ndarray, perms: np.ndarray, G: np.ndarray, inside: np.ndarray,
                          diffs: bool = True) -> Dict[str, np.ndarray]:
    """Per comparison, the predicted grids (permutations x 144) from the column x odor weights `Wt`, each permutation's
    column order and the columns' grids G (columns x 144): the single-odor maps, and for the differences each odor's map
    over its mean inside the outline minus the other's (the shape difference, as the measured differences are)."""
    maps = {o: Wt[:, k][perms] @ G for k, o in enumerate(MAP_ODORS)}
    out = dict(maps)
    if diffs:
        for a, b in DIFFS:
            ma = maps[a][:, inside].mean(axis=1, keepdims=True)
            mb = maps[b][:, inside].mean(axis=1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                out[f'{a}-{b}'] = maps[a] / ma - maps[b] / mb
    return out


def run_maps(n_perm: int = N_PERM, calyces: Optional[Dict[str, dict]] = None, rates: Optional[pd.DataFrame] = None,
             meas: Optional[Dict[str, dict]] = None, baseline: str = 'all_pfo') -> Dict[str, pd.DataFrame]:
    """Analysis 4: writes connectome_map_scores.csv.gz (every calyx, alternative, blur, slab, flip, recording and
    comparison, with the null percentile), connectome_map_best.csv (per calyx, alternative, recording and comparison the
    best over slabs, blurs and flips, with the percentile against the permutations' maxima), connectome_map_flips.csv
    (per recording the best flip), connectome_map_pooled.csv (the mean over recordings of each recording's best score at
    one flip, against the same mean in each permutation), connectome_measured.csv (the chosen concentrations) and
    figures; with another measured baseline than 'all_pfo' the file names end in _<baseline>."""
    calyces = load_calyces() if calyces is None else calyces
    rates = model_pn_rates() if rates is None else rates
    meas = measured_maps(baseline) if meas is None else meas
    sfx = '' if baseline == 'all_pfo' else f'_{baseline}'
    counts = c2_spike_counts()
    rec_names = list(meas)
    rng = np.random.default_rng(SEED)
    rows, best_rows, keep_maps = [], [], {}
    # per (calyx, alt, comparison): recordings x flips x (perm + 1) running maxima over slabs and blurs
    maxima: Dict[Tuple[str, str, str], np.ndarray] = {}
    for calyx, parts in calyces.items():
        frame = calyx_frame(parts['apl'])
        centers, a0 = slab_centers(parts['apl'])
        n_apl = len(parts['apl'])
        for alt in ALTERNATIVES:
            got = _alternative_points(calyx, parts, alt, rates, counts)
            if got is None:
                continue
            points, names, Wt = got
            n_cols = len(names)
            comps = [c for c in COMPARISONS if alt != 'anatomy' or '-' not in c]
            if alt == 'anatomy':
                perms = np.arange(n_cols)[None, :]
            else:
                perms = np.vstack([np.arange(n_cols)] + [rng.permutation(n_cols) for _ in range(n_perm)])
            for comp in comps:
                maxima[(calyx, alt, comp)] = np.full((len(rec_names), len(FLIPS), len(perms)), -np.inf)
            for center in [None] + list(centers):
                if center is not None:
                    lo, hi = center - SLAB_THICK_UM / 2, center + SLAB_THICK_UM / 2
                    if ((parts['apl']['A'] >= lo) & (parts['apl']['A'] < hi)).sum() < SLAB_MIN_SHARE * n_apl:
                        continue
                got = slab_column_grids(points, n_cols, parts['apl'], frame, center)
                if got is None:
                    continue
                grids, inside_p, _ = got
                depth = WHOLE_CALYX if center is None else float(center - a0)
                for blur, G in grids.items():
                    preds = predicted_comparisons(Wt, perms, G, inside_p, diffs=alt != 'anatomy')
                    for comp, pred in preds.items():
                        keep_maps[(calyx, alt, comp, blur, depth)] = (pred[0], inside_p)
                        for ri, rec in enumerate(rec_names):
                            m = meas[rec]
                            # an opposite-side calyx is mirrored into the recording's hemisphere
                            pr = flip_grid(pred, 'horizontal') if calyx_side(calyx) != m['side'] else pred
                            ins_p = (flip_grid(inside_p, 'horizontal') if calyx_side(calyx) != m['side'] else inside_p)
                            for fi, flip in enumerate(FLIPS):
                                y = flip_grid(m['grid'][comp], flip)
                                ins = ins_p & flip_grid(m['inside'], flip)
                                r = masked_pearson(pr, y, ins)
                                mx = maxima[(calyx, alt, comp)]
                                mx[ri, fi] = np.fmax(mx[ri, fi], r)
                                null = r[1:]
                                pct = float((null <= r[0]).mean() * 100) if len(null) else np.nan
                                rows.append((calyx, alt, comp, blur, depth, flip, rec, m['side'], float(r[0]), pct,
                                             int(ins.sum())))
            print(f'{time.strftime("%H:%M:%S")} {calyx} {alt}: {len(names)} columns, {len(centers)} slab centers',
                  flush=True)
    scores = pd.DataFrame(rows, columns=['calyx', 'alternative', 'comparison', 'blur_um', 'slab_depth_um', 'flip',
                                         'recording', 'side', 'r', 'null_pct', 'n_cells'])
    # best per (calyx, alt, comparison, recording) over slabs, blurs and flips, against the permutations' maxima
    for (calyx, alt, comp), mx in maxima.items():
        for ri, rec in enumerate(rec_names):
            sub = scores[(scores.calyx == calyx) & (scores.alternative == alt) & (scores.comparison == comp)
                         & (scores.recording == rec)]
            if sub.empty:
                continue
            b = sub.loc[sub['r'].idxmax()]
            over = mx[ri].max(axis=0)
            null = over[1:]
            best_rows.append(dict(calyx=calyx, alternative=alt, comparison=comp, recording=rec, side=b['side'],
                                  r=float(b['r']), flip=b['flip'], blur_um=float(b['blur_um']),
                                  slab_depth_um=float(b['slab_depth_um']), null_pct_of_cell=float(b['null_pct']),
                                  null_pct_of_best=float((null <= over[0]).mean() * 100) if len(null) else np.nan,
                                  null_best_p95=float(np.quantile(null, 0.95)) if len(null) else np.nan))
    best = pd.DataFrame(best_rows)
    # pooled over recordings at one flip, and the best flip per recording
    pooled_rows, flip_rows = [], []
    for (calyx, alt, comp), mx in maxima.items():
        per_flip = mx.max(axis=0)
        if not np.isfinite(mx).all():
            continue
        mean_rec = mx.mean(axis=0)
        for fi, flip in enumerate(FLIPS):
            null = mean_rec[fi, 1:]
            pooled_rows.append(dict(calyx=calyx, alternative=alt, comparison=comp, flip=flip,
                                    mean_best_r=float(mean_rec[fi, 0]),
                                    null_pct=float((null <= mean_rec[fi, 0]).mean() * 100) if len(null) else np.nan,
                                    null_p95=float(np.quantile(null, 0.95)) if len(null) else np.nan))
        best_flip_mean = mean_rec.max(axis=0)
        null = best_flip_mean[1:]
        pooled_rows.append(dict(calyx=calyx, alternative=alt, comparison=comp, flip='best',
                                mean_best_r=float(best_flip_mean[0]),
                                null_pct=float((null <= best_flip_mean[0]).mean() * 100) if len(null) else np.nan,
                                null_p95=float(np.quantile(null, 0.95)) if len(null) else np.nan))
        for ri, rec in enumerate(rec_names):
            fr = mx[ri, :, 0]
            flip_rows.append(dict(calyx=calyx, alternative=alt, comparison=comp, recording=rec,
                                  **{f'r_{f}': float(fr[i]) for i, f in enumerate(FLIPS)},
                                  best_flip=FLIPS[int(np.argmax(fr))]))
        del per_flip
    pooled = pd.DataFrame(pooled_rows)
    flips = pd.DataFrame(flip_rows)
    measured = pd.DataFrame([dict(recording=rec, side=m['side'], comparison=comp,
                                  **{f'conc_{o}': c for o, c in m['concs'][comp].items()},
                                  **{f'mean_dff_{o}': v for o, v in m['means'][comp].items()},
                                  outside_1log=any(abs(c - INPUT_CONC[o]) > 1.0 for o, c in m['concs'][comp].items()))
                             for rec, m in meas.items() for comp in COMPARISONS])
    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT / f'connectome_map_scores{sfx}.csv.gz', index=False)
    best.to_csv(OUT / f'connectome_map_best{sfx}.csv', index=False)
    pooled.to_csv(OUT / f'connectome_map_pooled{sfx}.csv', index=False)
    flips.to_csv(OUT / f'connectome_map_flips{sfx}.csv', index=False)
    measured.to_csv(OUT / f'connectome_measured{sfx}.csv', index=False)
    plot_maps(scores, pooled, keep_maps, meas, sfx)
    return dict(scores=scores, best=best, pooled=pooled, flips=flips, measured=measured)


def profiles(calyces: Optional[Dict[str, dict]] = None, rates: Optional[pd.DataFrame] = None, blur_um: float = 2.0,
             n_bins: int = 10) -> Dict[str, pd.DataFrame]:
    """The predicted maps' dorsal-ventral and lateral-medial profiles, for comparison with the measured profiles of
    E79_apl_calyx_constraints.py's analysis 3: per calyx, alternative (bouton, claw), the whole calyx (every bouton) and
    each slab, the share of each odor's predicted map (blurred `blur_um`, clipped at 0) in `n_bins` equal bins of the
    outline's bounding box, top (dorsal) to bottom in posterior view (`y`, as image y with the image dorsal up) and lateral
    to medial (`x_lateral_medial`, right calyces mirrored), and the centroid shift of pa and paa against ms in the same
    normalized coordinates (positive y: the first odor's response lower, i.e. more ventral). Writes connectome_profiles.csv
    and connectome_centroid_shift.csv; and connectome_glomerulus_positions.csv, each glomerulus's T-bar-weighted bouton
    centroid per calyx, 0 to 1 in the calyx's bounding box (dorsal: 0 ventral, 1 dorsal; medial: 0 lateral, 1 medial;
    anterior: 0 posterior edge, 1 anterior edge)."""
    calyces = load_calyces() if calyces is None else calyces
    rates = model_pn_rates() if rates is None else rates
    prof_rows, shift_rows = [], []
    for calyx, parts in calyces.items():
        frame = calyx_frame(parts['apl'])
        centers, a0 = slab_centers(parts['apl'])
        n_apl = len(parts['apl'])
        for alt in ('bouton', 'claw'):
            got = _alternative_points(calyx, parts, alt, rates, None)
            if got is None:
                continue
            points, names, Wt = got
            slabs = [(WHOLE_CALYX, -np.inf, np.inf)] + [(float(c - a0), c - SLAB_THICK_UM / 2, c + SLAB_THICK_UM / 2)
                                                    for c in centers]
            for depth, lo, hi in slabs:
                a = parts['apl'][(parts['apl']['A'] >= lo) & (parts['apl']['A'] < hi)]
                p = points[(points['A'] >= lo) & (points['A'] < hi)]
                if len(a) < SLAB_MIN_SHARE * n_apl or len(p) == 0:
                    continue
                mask = outline_mask(a['R'].to_numpy(), a['D'].to_numpy(), frame)
                ys, xs = np.nonzero(mask)
                img = rasterize(p['R'].to_numpy(), p['D'].to_numpy(), p['col'].to_numpy(), p['w'].to_numpy(),
                                len(names), frame)
                img = gaussian_filter(img, (0, blur_um / PX_UM, blur_um / PX_UM))
                cent = {}
                for k, odor in enumerate(MAP_ODORS):
                    m = np.clip(np.tensordot(Wt[:, k], img, axes=1), 0, None)[ys, xs]
                    for axis, c, lo_, hi_ in (('y', ys, ys.min(), ys.max()), ('x_lateral_medial', xs, xs.min(), xs.max())):
                        u = (c - lo_) / max(hi_ - lo_, 1)
                        if axis == 'x_lateral_medial' and calyx_side(calyx) == 'R':
                            u = 1 - u
                        b = np.clip((u * n_bins).astype(int), 0, n_bins - 1)
                        sh = np.bincount(b, weights=m, minlength=n_bins) / m.sum()
                        cent[(odor, axis)] = float((u * m).sum() / m.sum())
                        prof_rows.append(dict(calyx=calyx, alternative=alt, slab_depth_um=depth, odor=odor, axis=axis,
                                              **{f'bin{i}': float(v) for i, v in enumerate(sh)}))
                for a_, b_ in DIFFS:
                    for axis in ('y', 'x_lateral_medial'):
                        shift_rows.append(dict(calyx=calyx, alternative=alt, slab_depth_um=depth, pair=f'{a_} - {b_}',
                                               axis=axis, shift=cent[(a_, axis)] - cent[(b_, axis)]))
    prof = pd.DataFrame(prof_rows)
    shift = pd.DataFrame(shift_rows)
    # each glomerulus's T-bar-weighted bouton centroid in the calyx's APL bounding box (0.5th to 99.5th percentiles)
    pos_rows = []
    for calyx, parts in calyces.items():
        a = parts['apl']
        lo = {k: np.quantile(a[k], 0.005) for k in 'RDA'}
        hi = {k: np.quantile(a[k], 0.995) for k in 'RDA'}
        b = parts['boutons'].dropna(subset=['glomerulus'])
        for g, bb in b.groupby('glomerulus'):
            w = bb['n_tbars'].to_numpy(dtype=float)
            c = {k: float((bb[k].to_numpy() * w).sum() / w.sum()) for k in 'RDA'}
            u = (c['R'] - lo['R']) / (hi['R'] - lo['R'])
            pos_rows.append(dict(calyx=calyx, glomerulus=g, n_boutons=len(bb), n_tbars=int(w.sum()),
                                 dorsal=(c['D'] - lo['D']) / (hi['D'] - lo['D']),
                                 medial=1 - u if calyx_side(calyx) == 'R' else u,
                                 anterior=(c['A'] - lo['A']) / (hi['A'] - lo['A'])))
    pos = pd.DataFrame(pos_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    prof.to_csv(OUT / 'connectome_profiles.csv', index=False)
    shift.to_csv(OUT / 'connectome_centroid_shift.csv', index=False)
    pos.to_csv(OUT / 'connectome_glomerulus_positions.csv', index=False)
    return dict(profiles=prof, centroid_shift=shift, positions=pos)


def plot_maps(scores: pd.DataFrame, pooled: pd.DataFrame, keep_maps: dict, meas: Dict[str, dict], sfx: str = '') -> None:
    """E79_connectome_scores.png: per calyx, the best r over flips and blurs against slab depth for pa minus ms and paa
    minus ms (bouton-driven), median over recordings with the range; E79_connectome_maps.png: the bouton-driven
    predicted difference maps at each calyx's best slab (by the pooled score), beside the measured ones."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sub = scores[(scores.alternative == 'bouton') & scores.comparison.isin(['pa-ms', 'paa-ms'])]
    cal = list(dict.fromkeys(sub.calyx))
    fig, axes = plt.subplots(2, len(cal), figsize=(2.3 * len(cal), 5.0), sharey=True, squeeze=False)
    for j, c in enumerate(cal):
        for i, comp in enumerate(['pa-ms', 'paa-ms']):
            ax = axes[i, j]
            s = sub[(sub.calyx == c) & (sub.comparison == comp)]
            b = s[s.slab_depth_um >= 0].groupby(['slab_depth_um', 'recording'])['r'].max().unstack()
            ax.fill_between(b.index, b.min(axis=1), b.max(axis=1), color='0.85', lw=0)
            ax.plot(b.index, b.median(axis=1), color='C0', lw=1.5)
            w = s[s.slab_depth_um == WHOLE_CALYX].groupby('recording')['r'].max()
            if len(w):
                ax.errorbar([-4.0], [w.median()], yerr=[[w.median() - w.min()], [w.max() - w.median()]], fmt='o',
                            color='C1', ms=3, lw=0.8)
                ax.annotate('whole', (-4.0, w.max()), fontsize=5, ha='center', va='bottom')
            ax.axhline(0, color='0.5', lw=0.5)
            ax.set_title(f'{c}\n{comp} (median, range)', fontsize=7)
            if i == 1:
                ax.set_xlabel('slab depth from posterior edge (um)', fontsize=7)
            if j == 0:
                ax.set_ylabel('best r over flips and blurs', fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle('Bouton-driven predicted difference maps against the 12 measured ones, by slab depth', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f'E79_connectome_scores{sfx}.png', dpi=120)
    plt.close(fig)
    # maps: per calyx the slab and blur with the best median r at the best pooled flip
    fig, axes = plt.subplots(2, len(cal) + 2, figsize=(1.9 * (len(cal) + 2), 4.2), squeeze=False)
    for i, comp in enumerate(['pa-ms', 'paa-ms']):
        for j, c in enumerate(cal):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            s = sub[(sub.calyx == c) & (sub.comparison == comp)]
            med = s.groupby(['blur_um', 'slab_depth_um', 'flip'])['r'].median()
            blur, depth, flip = med.idxmax()
            g, ins = keep_maps[(c, 'bouton', comp, blur, depth)]
            img = np.where(ins, g, np.nan).reshape(GRID_N, GRID_N)
            v = np.nanmax(np.abs(img))
            ax.imshow(img, cmap='RdBu_r', vmin=-v, vmax=v)
            ax.set_title(f'{c} {comp}\n{depth:.0f} um, blur {blur:g}, median r {med.max():.2f}\n(posterior view; '
                         f'best flip {flip})', fontsize=5.5)
        for k, rec in enumerate(list(meas)[:2]):
            ax = axes[i, len(cal) + k]
            ax.set_xticks([]); ax.set_yticks([])
            m = meas[rec]
            img = np.where(m['inside'], m['grid'][comp], np.nan).reshape(GRID_N, GRID_N)
            v = np.nanmax(np.abs(img))
            ax.imshow(img, cmap='RdBu_r', vmin=-v, vmax=v)
            ax.set_title(f'measured {rec} ({m["side"]})\n{comp}, image axes', fontsize=5.5)
    fig.suptitle('Predicted (bouton-driven, 12 x 12 grid over the outline) and measured difference maps; red = more '
                 'response to the first odor', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f'E79_connectome_maps{sfx}.png', dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# analysis 6

def colocalization(boutons: pd.DataFrame, lam_um: float, glomeruli: Sequence[str]) -> pd.DataFrame:
    """C_gh: the n_tbars-weighted mean over bouton pairs (one of g, one of h; a bouton paired with itself included) of
    exp(-d^2 / 2 lambda^2), d the 3D distance between bouton centroids (um)."""
    b = boutons[boutons['glomerulus'].isin(glomeruli)]
    P = b[['R', 'D', 'A']].to_numpy(dtype=float)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    K = np.exp(-d2 / (2 * lam_um ** 2))
    idx = pd.Categorical(b['glomerulus'], categories=list(glomeruli)).codes
    Wg = np.zeros((len(b), len(glomeruli)))
    Wg[np.arange(len(b)), idx] = b['n_tbars'].to_numpy(dtype=float)
    T = Wg.sum(axis=0)
    C = (Wg.T @ K @ Wg) / np.outer(T, T)
    return pd.DataFrame(C, index=list(glomeruli), columns=list(glomeruli))


def _pairs(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def cosine_distances(Y: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """1 - cosine similarity of the odor columns of Y (... x glomeruli x odors) for each pair: ... x n_pairs."""
    nrm = np.sqrt((Y ** 2).sum(axis=-2))
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    dot = (Y[..., :, a] * Y[..., :, b]).sum(axis=-2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 1.0 - dot / (nrm[..., a] * nrm[..., b])


def proxy_change(X: np.ndarray, C: np.ndarray, form: str, strength: float,
                 pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """Per odor pair, the cosine distance of the PN-rate vectors (X: glomeruli x odors, >= 0) with each glomerulus's
    inhibition following its local field Cx, minus that with the global mean. Global inhibition I = strength x the odor's
    mean rate over glomeruli; local I_g = s x (Cx)_g, s set so that the mean of I over glomeruli and odors is the same.
    `form` 'sub': relu(x - I); 'div': x / (1 + I / mu), mu the mean rate over glomeruli and odors (the global divisive
    version scales each odor uniformly, so its distances are X's own). C may carry leading permutation axes (... x G x G);
    returns ... x n_pairs."""
    pairs = _pairs(X.shape[1]) if pairs is None else pairs
    m = X.mean(axis=0)
    glob = strength * np.broadcast_to(m, X.shape)
    field = C @ X
    s = glob.sum() / field.sum(axis=(-2, -1), keepdims=True)
    loc = s * field
    if form == 'sub':
        y_loc, y_glob = np.maximum(X - loc, 0.0), np.maximum(X - glob, 0.0)
    elif form == 'div':
        mu = X.mean()
        y_loc, y_glob = X / (1 + loc / mu), X / (1 + glob / mu)
    else:
        raise ValueError(form)
    return cosine_distances(y_loc, pairs) - cosine_distances(y_glob, pairs)


def loo_share(delta: np.ndarray, err: np.ndarray, pairs: Sequence[Tuple[int, int]], n_odors: int,
              nonneg: bool = False) -> np.ndarray:
    """Share of err's squared sum removed by the best linear scaling k of delta (err + k delta), k fitted on the pairs of
    15 odors and scored on the 15 pairs of the left-out one, pooled over the n_odors left-out odors; delta (... x n_pairs)
    centered over the pairs first. `nonneg`: k >= 0 (the direction in which local inhibition would add to the model's
    APL effect where the change in distance is positive). Vectorized over leading axes of delta."""
    d = delta - np.nanmean(delta, axis=-1, keepdims=True)
    d = np.nan_to_num(d)
    num = den = 0.0
    for o in range(n_odors):
        test = np.array([o in p for p in pairs])
        dd, ee = d[..., ~test], err[~test]
        k = -(dd * ee).sum(axis=-1) / np.maximum((dd ** 2).sum(axis=-1), 1e-300)
        if nonneg:
            k = np.maximum(k, 0.0)
        res = err[test] + k[..., None] * d[..., test]
        num = num + (res ** 2).sum(axis=-1)
        den = den + (err[test] ** 2).sum()
    return 1.0 - num / den


def in_sample_share(delta: np.ndarray, err: np.ndarray) -> np.ndarray:
    d = delta - np.nanmean(delta, axis=-1, keepdims=True)
    k = -(d * err).sum(axis=-1) / np.maximum((d ** 2).sum(axis=-1), 1e-300)
    return 1.0 - ((err + k[..., None] * d) ** 2).sum(axis=-1) / (err ** 2).sum()


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - np.nanmean(a, axis=-1, keepdims=True)
    b = b - b.mean()
    return (a * b).sum(axis=-1) / np.sqrt((a ** 2).sum(axis=-1) * (b ** 2).sum())


def run_coloc(n_perm: int = N_PERM, calyces: Optional[Dict[str, dict]] = None,
              rates: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """Analysis 6: writes coloc_matrix_<calyx>_lambda<l>.csv, coloc_agreement.csv, coloc_sets.csv,
    coloc_concentration.csv, coloc_proxy.csv, coloc_proxy_pairs.csv and figures."""
    calyces = load_calyces() if calyces is None else calyces
    rates = model_pn_rates() if rates is None else rates
    err = common_error()
    odors = list(rates.columns)
    pairs = _pairs(len(odors))
    pair_index = pd.MultiIndex.from_tuples([tuple(sorted((odors[i], odors[j]))) for i, j in pairs],
                                           names=['odor_a', 'odor_b'])
    e = err.reindex(pair_index).to_numpy(dtype=float)
    if np.isnan(e).any():
        raise RuntimeError('common error lacks some of the 120 pairs')
    with_input = set(pd.read_csv(ORN_INPUT, index_col=0, header=[0, 1]).index.dropna())
    driven = [g for g in rates.index if g in with_input]
    glomeruli = [g for g in driven if all(g in set(p['boutons']['glomerulus'].dropna()) for p in calyces.values())]
    missing = sorted(set(driven) - set(glomeruli))
    X = rates.loc[glomeruli, odors].clip(lower=0).to_numpy(dtype=float)
    OUT.mkdir(parents=True, exist_ok=True)
    Cs: Dict[Tuple[str, float], np.ndarray] = {}
    for calyx, parts in calyces.items():
        for lam in LAMBDAS_UM:
            C = colocalization(parts['boutons'], lam, glomeruli)
            C.to_csv(OUT / f'coloc_matrix_{calyx}_lambda{lam:g}.csv')
            Cs[(calyx, lam)] = C.to_numpy()
    dataset_calyces = [c for c in calyces if not c.startswith('model_')]
    for lam in LAMBDAS_UM:
        Cs[('consensus', lam)] = np.mean([Cs[(c, lam)] for c in dataset_calyces], axis=0)
    names = list(calyces) + ['consensus']
    G = len(glomeruli)
    iu = np.triu_indices(G, 1)
    # agreement between calyces
    agree = []
    from scipy.stats import spearmanr
    for lam in LAMBDAS_UM:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                va, vb = Cs[(a, lam)][iu], Cs[(b, lam)][iu]
                agree.append(dict(lambda_um=lam, calyx_a=a, calyx_b=b, pearson_r=float(np.corrcoef(va, vb)[0, 1]),
                                  spearman_rho=float(spearmanr(va, vb)[0])))
    agree = pd.DataFrame(agree)
    # glomerulus sets nearer each other than random sets
    rng = np.random.default_rng(SEED)
    set_rows = []
    for set_name, gs in GLOMERULUS_SETS.items():
        gi = [glomeruli.index(g) for g in gs if g in glomeruli]
        k = len(gi)
        draws = np.array([rng.choice(G, k, replace=False) for _ in range(N_SET_DRAWS)])
        for calyx in names:
            for lam in LAMBDAS_UM:
                C = Cs[(calyx, lam)]

                def mean_off(ix):
                    sub = C[np.ix_(ix, ix)]
                    return (sub.sum() - np.trace(sub)) / (k * (k - 1))
                obs = mean_off(gi)
                sub = C[draws[:, :, None], draws[:, None, :]]
                null = (sub.sum(axis=(1, 2)) - np.trace(sub, axis1=1, axis2=2)) / (k * (k - 1))
                set_rows.append(dict(set=set_name, glomeruli=' '.join(gs), n_in_universe=k, calyx=calyx,
                                     lambda_um=lam, mean_C=float(obs), random_median=float(np.median(null)),
                                     pct=float((null < obs).mean() * 100)))
    sets = pd.DataFrame(set_rows)
    # spatial concentration
    conc_rows = []
    tot = X.sum(axis=0)
    for calyx in names:
        for lam in LAMBDAS_UM:
            C = Cs[(calyx, lam)]
            q = np.einsum('go,gh,ho->o', X, C, X) / tot ** 2
            for o, v in zip(odors, q):
                conc_rows.append(dict(calyx=calyx, lambda_um=lam, odor=o, concentration=float(v)))
    concentration = pd.DataFrame(conc_rows)
    # the proxy against the common error
    perms = np.vstack([np.arange(G)] + [rng.permutation(G) for _ in range(n_perm)])
    proxy_rows, pair_rows = [], []
    all_loo = []
    for calyx in names:
        for lam in LAMBDAS_UM:
            C = Cs[(calyx, lam)]
            Cp = C[perms[:, :, None], perms[:, None, :]]
            for form, strength in PROXY_SETTINGS:
                delta = proxy_change(X, Cp, form, strength, pairs)
                r = _corr(np.nan_to_num(delta), e)
                loo = loo_share(delta, e, pairs, len(odors))
                loo_pos = loo_share(delta, e, pairs, len(odors), nonneg=True)
                ins = in_sample_share(np.nan_to_num(delta), e)
                all_loo.append(loo)
                proxy_rows.append(dict(calyx=calyx, lambda_um=lam, form=form, strength=strength,
                                       corr_with_common_error=float(r[0]), corr_null_pct=float((r[1:] <= r[0]).mean() * 100),
                                       in_sample_share=float(ins[0]), loo_share=float(loo[0]),
                                       loo_null_p95=float(np.quantile(loo[1:], 0.95)),
                                       loo_null_pct=float((loo[1:] < loo[0]).mean() * 100),
                                       loo_share_k_nonneg=float(loo_pos[0]),
                                       loo_nonneg_null_pct=float((loo_pos[1:] < loo_pos[0]).mean() * 100),
                                       delta_sd=float(np.nanstd(delta[0]))))
                for (a, b), dv, ev in zip(pair_index, delta[0], e):
                    pair_rows.append(dict(calyx=calyx, lambda_um=lam, form=form, strength=strength, odor_a=a,
                                          odor_b=b, delta=float(dv), common_error=float(ev)))
    proxy = pd.DataFrame(proxy_rows)
    allv = np.array(all_loo)
    best_null = allv[:, 1:].max(axis=0)
    proxy['loo_null_pct_of_best'] = [float((best_null < v).mean() * 100) for v in proxy['loo_share']]
    proxy['gate'] = (proxy['loo_share'] >= GATE_SHARE) & (proxy['loo_null_pct'] >= GATE_PCT)
    proxy['gate_best_null'] = (proxy['loo_share'] >= GATE_SHARE) & (proxy['loo_null_pct_of_best'] >= GATE_PCT)
    meta = dict(glomeruli=glomeruli, driven_missing_from_some_calyx=missing, best_null_p95=float(np.quantile(best_null, 0.95)))
    (OUT / 'coloc_meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    agree.to_csv(OUT / 'coloc_agreement.csv', index=False)
    sets.to_csv(OUT / 'coloc_sets.csv', index=False)
    concentration.to_csv(OUT / 'coloc_concentration.csv', index=False)
    proxy.to_csv(OUT / 'coloc_proxy.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / 'coloc_proxy_pairs.csv.gz', index=False)
    plot_coloc(Cs, glomeruli, names, proxy, pd.DataFrame(pair_rows))
    return dict(agreement=agree, sets=sets, concentration=concentration, proxy=proxy, meta=pd.Series(meta))


def plot_coloc(Cs: dict, glomeruli: List[str], names: List[str], proxy: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """E79_coloc_matrices.png (C at 8 um per calyx, glomeruli ordered by the consensus's hierarchical clustering) and
    E79_coloc_proxy.png (the best setting's change against the common error)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    lam = 8.0
    cons = Cs[('consensus', lam)]
    order = leaves_list(linkage(cons, 'average'))
    n = len(names)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.0 * ((n + 1) // 2), 6.4), squeeze=False)
    for k, c in enumerate(names):
        ax = axes.flat[k]
        C = Cs[(c, lam)][np.ix_(order, order)]
        ax.imshow(np.log10(np.maximum(C, 1e-6)), cmap='viridis', vmin=-4, vmax=0)
        ax.set_title(f'{c}', fontsize=7)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([glomeruli[i] for i in order], rotation=90, fontsize=3.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([glomeruli[i] for i in order], fontsize=3.5)
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle(f'Glomerulus co-localization C (log10), lambda {lam:g} um; glomeruli in the consensus\'s cluster order',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'E79_coloc_matrices.png', dpi=150)
    plt.close(fig)
    b = proxy.loc[proxy['loo_share'].idxmax()]
    s = pairs[(pairs.calyx == b.calyx) & (pairs.lambda_um == b.lambda_um) & (pairs.form == b.form)
              & (pairs.strength == b.strength)]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.scatter(s['delta'] - s['delta'].mean(), s['common_error'], s=10, color='C0')
    ax.axhline(0, color='0.6', lw=0.5)
    ax.axvline(0, color='0.6', lw=0.5)
    ax.set_xlabel('change in PN-rate cosine distance, local minus global (centered)', fontsize=7)
    ax.set_ylabel('error common to all wirings', fontsize=7)
    ax.set_title(f'best setting: {b.calyx}, lambda {b.lambda_um:g} um, {b.form} {b.strength:g}\n'
                 f'r = {b.corr_with_common_error:.2f}, left-out share {b.loo_share:.3f}', fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(OUT / 'E79_coloc_proxy.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# provenance and tables

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path) -> dict:
    def run(*a):
        return subprocess.run(['git', '-C', str(repo), *a], capture_output=True, text=True).stdout.strip()
    diff = run('diff', 'HEAD')
    status = run('status', '--porcelain', '-uno')
    return dict(path=str(repo), commit=run('rev-parse', 'HEAD'), dirty=bool(status),
                dirty_files=[ln[3:] for ln in status.splitlines()],
                diff_sha256=hashlib.sha256(diff.encode()).hexdigest() if diff else None)


def write_provenance(extra: dict) -> None:
    """provenance_connectome.json: script, al_analysis and olfsysm commits and dirty state, inputs with sha256, and
    `runs`: every run's analyses, settings and commit (kept across runs that redo part of the analyses)."""
    inputs = [YANG_APL / f'{ds}_{k}.csv.gz' for ds in DATASETS
              for k in ('PN_boutons_derived', 'KC_claws_derived', 'APL_MB_synapses')]
    inputs += [PRAT_PN_SYNAPSES, PRAT_PN_KC, C2_ON_COUNTS, COMMON_ERROR, ORN_INPUT, ACM.DATA / 'recordings.csv',
               ACM.DATA / 'frames.csv']
    inputs += sorted(ACM.DATA.glob('*.tif')) + sorted(ACM.DATA.glob('*.roi'))
    pn = OUT / 'connectome_pn_rates.csv'
    if pn.exists():
        inputs.append(pn)
    try:
        import olfsysm
        build = olfsysm.build_info()
    except Exception as ex:
        build = repr(ex)
    prov = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S %z'), script=Path(__file__).name,
                al_analysis=_git(HERE), olfsysm=_git(HERE.parents[2] / 'olfsysm'), olfsysm_build=build,
                c2=dict(experiment=C2[0], tag=C2[1]),
                settings=dict(slab_thick_um=SLAB_THICK_UM, slab_step_um=SLAB_STEP_UM, slab_min_share=SLAB_MIN_SHARE,
                              blurs_um=list(BLURS_UM), px_um=PX_UM, outline_blur_um=OUTLINE_BLUR_UM,
                              outline_keep=OUTLINE_KEEP, grid_n=GRID_N, cell_inside=CELL_INSIDE,
                              meas_blur_px=MEAS_BLUR_PX, match_min_fraction=MATCH_MIN_FRACTION,
                              match_max_ratio=MATCH_MAX_RATIO, flips=list(FLIPS),
                              lambdas_um=list(LAMBDAS_UM), glomerulus_sets={k: list(v) for k, v in GLOMERULUS_SETS.items()},
                              n_set_draws=N_SET_DRAWS, proxy_settings=[list(s) for s in PROXY_SETTINGS], seed=SEED,
                              gate_share=GATE_SHARE, gate_pct=GATE_PCT),
                inputs={str(p): _sha256(p) for p in inputs if p.exists()}, **extra)
    path = OUT / 'provenance_connectome.json'
    runs = []
    if path.exists():
        try:
            old = json.loads(path.read_text())
            runs = old.get('runs', [])
        except (ValueError, OSError):
            pass
    runs.append(dict(generated=prov['generated'], al_analysis=prov['al_analysis'], **extra))
    prov['runs'] = runs
    path.write_text(json.dumps(prov, indent=1, default=str) + '\n')


def _md(df: pd.DataFrame, floatfmt: str = '.3f') -> str:
    cols = list(df.columns)
    out = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


TABLE_FILES = {'axes': 'connectome_axis_checks.csv', 'calyces': 'connectome_calyces.csv',
               'pooled': 'connectome_map_pooled.csv', 'flips': 'connectome_map_flips.csv',
               'best': 'connectome_map_best.csv', 'measured': 'connectome_measured.csv',
               'centroid_shift': 'connectome_centroid_shift.csv', 'positions': 'connectome_glomerulus_positions.csv',
               'agreement': 'coloc_agreement.csv',
               'sets': 'coloc_sets.csv', 'concentration': 'coloc_concentration.csv', 'proxy': 'coloc_proxy.csv'}


def write_table(tables: Dict[str, pd.DataFrame]) -> None:
    """connectome_table.md: the key numbers of analyses 4 and 6 (tables not computed in this run are read from their
    CSVs in OUT)."""
    tables = dict(tables)
    for k, f in TABLE_FILES.items():
        if k not in tables and (OUT / f).exists():
            tables[k] = pd.read_csv(OUT / f)
    lines = ['# E79 connectome analyses (4 and 6): key tables', '',
             f'Written by E79_connectome_maps.py, {time.strftime("%Y-%m-%d %H:%M")}. Details in the CSVs named in each '
             'section; the axis checks in connectome_axis_checks.csv.', '',
             'Axes: native nm to (fly\'s right R, dorsal D, anterior A) in um: hemibrain (-x, -z, +y), FlyWire (x, -y, -z; '
             'sides from FlyWire\'s labels), male CNS (-x, -y, -z), BANC (-x, +z, -y). R and D are exactly Yang\'s Claude\'s '
             'posterior-view axes (view_right_um, view_dorsal_um of odor_weighted/PN_APL_synapses_odor_weighted.csv; '
             'maximum difference in the axis-check table); the A sign is set so the lobes are anterior of the calyx. '
             'Posterior view: R to the right, D up (flip "none" = the measured image is a posterior view with dorsal up and '
             'the fly\'s right on the right); a calyx of the other side than the recording is mirrored first.', '',
             'Model PN rates: C2\'s PN stage on the APL16 ORN input (pa and ms at -3, paa at -4 substituted), evoked rates '
             '(connectome_pn_rates.csv). The APL maps used pa -6.8 to -4.4, paa -6.3 to -3.0, ms -7.7 to -2: every pa map is '
             'more than one log10 unit from the model\'s input (outside the 1-log rule), and so are the paa maps at -2.954 '
             'and below -5 and the ms maps below -4; see the measured-maps table.', '',
             'Measured maps: apl_calyx_maps.py\'s (each stimulus minus the mean of every paraffin oil presentation), unless '
             'the run\'s baseline says otherwise (provenance_connectome.json, runs).', '']
    if 'axes' in tables:
        lines += ['## Axis checks (um; every check should be positive)', '', _md(tables['axes'], '.1f'), '']
    if 'calyces' in tables:
        lines += ['## Calyces (connectome_calyces.csv; partners_per_tbar is the bouton tables\' n_postsynaptic_partners '
                  'over n_tbars)', '', _md(tables['calyces'], '.3f'), '']
    if 'pooled' in tables:
        p = tables['pooled']
        lines += ['## Analysis 4: pooled over the 12 recordings (connectome_map_pooled.csv)', '',
                  'Mean over recordings of each recording\'s best r (over slabs and blurs) at one flip, and at the best '
                  'flip; null: the same mean in 1,000 label permutations (percentile, and its 95th percentile).', '']
        for comp in COMPARISONS:
            s = p[(p.comparison == comp)].pivot_table(index=['alternative', 'calyx'], columns='flip',
                                                       values=['mean_best_r', 'null_pct'])
            if s.empty:
                continue
            s.columns = [f'{a} {b}' for a, b in s.columns]
            lines += [f'### {comp}', '', _md(s.reset_index(), '.2f'), '']
    for bl in ('order_matched', 'first_in_block'):
        f = OUT / f'connectome_map_pooled_{bl}.csv'
        if not f.exists():
            continue
        q = pd.read_csv(f)
        q = q[q.comparison.isin(['pa-ms', 'paa-ms']) & (q.flip == 'best')]
        q = q.pivot_table(index=['alternative', 'calyx'], columns='comparison', values=['mean_best_r', 'null_pct'])
        q.columns = [f'{a} {b}' for a, b in q.columns]
        lines += [f'### pa-ms and paa-ms with the measured maps\' {bl} baseline (E79_apl_calyx_constraints.py\'s '
                  f'stimulus_maps; connectome_map_*_{bl}.csv), best flip', '', _md(q.reset_index(), '.2f'), '']
    if 'flips' in tables:
        f = tables['flips']
        f = f[f.comparison.isin(['pa-ms', 'paa-ms'])]
        cnt = f.groupby(['alternative', 'comparison', 'calyx'])['best_flip'].value_counts().unstack(fill_value=0)
        lines += ['## Analysis 4: recordings won by each flip (best over slabs and blurs; connectome_map_flips.csv)', '',
                  _md(cnt.reset_index(), '.0f'), '']
    if 'best' in tables:
        b = tables['best']
        b = b[(b.alternative == 'bouton') & b.comparison.isin(['pa-ms', 'paa-ms'])]
        s = b.groupby(['comparison', 'calyx']).agg(median_r=('r', 'median'), max_r=('r', 'max'),
                                                   n_rec_pct95=('null_pct_of_best', lambda v: int((v >= 95).sum())),
                                                   median_depth_um=('slab_depth_um', 'median'))
        lines += ['## Analysis 4: per recording, the best slab, blur and flip (bouton-driven; connectome_map_best.csv)', '',
                  'n_rec_pct95: recordings whose best score is at or above the 95th percentile of the permutations\' '
                  'best scores.', '', _md(s.reset_index(), '.2f'), '']
    if 'centroid_shift' in tables:
        c = tables['centroid_shift']
        c = c[c['slab_depth_um'] == WHOLE_CALYX].pivot_table(index=['alternative', 'calyx'], columns=['pair', 'axis'],
                                                      values='shift')
        c.columns = [f'{a} {b}' for a, b in c.columns]
        lines += ['## Predicted centroid shifts, whole calyx (connectome_centroid_shift.csv)', '',
                  'In the outline\'s bounding box, 0 to 1: y top (dorsal) to bottom, positive = the first odor\'s '
                  'response more ventral; x lateral to medial, positive = more medial. Blur 2 um, maps clipped at 0.', '',
                  _md(c.reset_index(), '.3f'), '']
        mp = OUT / 'E79_profile_centroid_shift.csv'
        if mp.exists():
            lines += ['Measured (E79_apl_calyx_constraints.py, E79_profile_centroid_shift.csv; image axes as recorded, '
                      'y top to bottom):', '', _md(pd.read_csv(mp), '.3f'), '']
    if 'positions' in tables:
        q = tables['positions']
        # pa's, ms's and paa's most driven glomeruli in the model's PN rates, and the block's
        show = ['DM3', 'DM2', 'DM6', 'DC2', 'VM5d', 'DL1', 'VA7m', 'D', 'DM1', 'VL2a', 'DL2v', 'VM2']
        q = q[q.glomerulus.isin(show)].pivot_table(index='calyx', columns='glomerulus', values='dorsal')[show]
        lines += ['## Height of glomeruli in the calyx (T-bar-weighted bouton centroid, 0 ventral to 1 dorsal; '
                  'connectome_glomerulus_positions.csv)', '',
                  'pa drives DM3, DM2, DM6, DC2, VM5d most in the model\'s PN rates at -3; ms DL1, VA7m, D; paa DM1, VL2a, '
                  'DL2v.', '', _md(q.reset_index(), '.2f'), '']
    if 'measured' in tables:
        m = tables['measured']
        lines += ['## Measured maps: concentrations used (log10; outside_1log: more than one log unit from the model\'s '
                  'ORN input, pa -3, ms -3, paa -4)', '', _md(m, '.3f'), '']
    if 'agreement' in tables:
        a = tables['agreement']
        a = a[(a.calyx_a != 'consensus') & (a.calyx_b != 'consensus')].copy()
        ds = lambda c: c.replace('model_', '').rsplit('_', 1)[0]
        a['kind'] = np.where(a.calyx_a.map(ds) == a.calyx_b.map(ds), 'same dataset (L-R, or the two hemibrain bouton sets)',
                             'different datasets')
        s = a.groupby(['lambda_um', 'kind']).agg(n_pairs=('pearson_r', 'size'), median_r=('pearson_r', 'median'),
                                                 min_r=('pearson_r', 'min'), median_rho=('spearman_rho', 'median'))
        lines += ['## Analysis 6: agreement of C between calyces (off-diagonal entries; coloc_agreement.csv)', '',
                  _md(s.reset_index(), '.2f'), '']
    if 'sets' in tables:
        s = tables['sets'].pivot_table(index=['set', 'calyx'], columns='lambda_um', values='pct')
        s.columns = [f'pct lambda {c:g}' for c in s.columns]
        lines += ['## Analysis 6: glomerulus sets nearer each other than random sets (percentile of the mean '
                  'off-diagonal C among 10,000 random sets; coloc_sets.csv)', '', _md(s.reset_index(), '.1f'), '']
    if 'concentration' in tables:
        c = tables['concentration']
        c = c[c.calyx == 'consensus'].pivot_table(index='odor', columns='lambda_um', values='concentration')
        c.columns = [f'lambda {x:g}' for x in c.columns]
        lines += ['## Analysis 6: spatial concentration x\'Cx / (sum x)^2, consensus C (coloc_concentration.csv)', '',
                  _md(c.reset_index(), '.4f'), '']
    if 'proxy' in tables:
        p = tables['proxy'].sort_values('loo_share', ascending=False)
        cols = ['calyx', 'lambda_um', 'form', 'strength', 'corr_with_common_error', 'corr_null_pct', 'in_sample_share',
                'loo_share', 'loo_null_p95', 'loo_null_pct', 'loo_null_pct_of_best', 'loo_share_k_nonneg', 'gate',
                'gate_best_null']
        lines += ['## Analysis 6: the proxy against the error common to all wirings (coloc_proxy.csv; top 15 settings '
                  'by the left-out share)', '', _md(p[cols].head(15), '.3f'), '',
                  f'Settings passing the gate (left-out share >= {GATE_SHARE:.2f} and >= {GATE_PCT:g}th percentile of its '
                  f'null): {int(p["gate"].sum())} of {len(p)}; against the best-of-all-settings null: '
                  f'{int(p["gate_best_null"].sum())}.', '']
    (OUT / 'connectome_table.md').write_text('\n'.join(lines) + '\n')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--analyses', default='4,6', help="comma-separated subset of 4, 6 and 'profiles' (part of 4); "
                   "'table' only rewrites connectome_table.md from the CSVs")
    p.add_argument('--n-perm', type=int, default=N_PERM)
    p.add_argument('--rebuild-pn-rates', action='store_true', help="recompute C2's PN rates (runs the PN stage)")
    p.add_argument('--baseline', default='all_pfo', help="the measured maps' baseline (see measured_maps)")
    args = p.parse_args()
    which = set(args.analyses.split(','))
    OUT.mkdir(parents=True, exist_ok=True)
    if which == {'table'}:
        write_table({})
        return
    t0 = time.time()
    rates = model_pn_rates(rebuild=args.rebuild_pn_rates)
    calyces = load_calyces()
    tables: Dict[str, pd.DataFrame] = {}
    axes = axis_checks()
    axes.to_csv(OUT / 'connectome_axis_checks.csv', index=False)
    tables['axes'] = axes
    if not axes['passed'].all():
        raise RuntimeError(f'axis checks failed:\n{axes}')
    summary = pd.DataFrame([dict(calyx=c, n_boutons=len(p['boutons']), n_tbars=int(p['boutons']['n_tbars'].sum()),
                                 share_tbars_uniglomerular=float(p['boutons'].dropna(subset=['glomerulus'])['n_tbars'].sum()
                                                                 / p['boutons']['n_tbars'].sum()),
                                 partners_per_tbar=(float(p['boutons']['n_postsynaptic_partners'].sum()
                                                          / p['boutons']['n_tbars'].sum())
                                                    if 'n_postsynaptic_partners' in p['boutons'] else np.nan),
                                 n_claws=len(p['claws']) if p['claws'] is not None else 0, n_apl_calyx=len(p['apl']),
                                 apl_A_extent_um=float(np.diff(np.quantile(p['apl']['A'], [0.005, 0.995]))[0]))
                            for c, p in calyces.items()])
    summary.to_csv(OUT / 'connectome_calyces.csv', index=False)
    tables['calyces'] = summary
    extra = dict(n_perm=args.n_perm, analyses=sorted(which), baseline=args.baseline)
    if '4' in which:
        res = run_maps(args.n_perm, calyces, rates, baseline=args.baseline)
        if args.baseline == 'all_pfo':
            tables.update(res)
    if '4' in which or 'profiles' in which:
        tables.update(profiles(calyces, rates))
    if '6' in which:
        tables.update(run_coloc(args.n_perm, calyces, rates))
    write_table(tables)
    write_provenance(extra)
    print(f'done in {time.time() - t0:.0f} s -> {OUT}')


if __name__ == '__main__':
    main()
