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
# populations: S, T, all eight, and the seven re-drawn wirings (part A1's outcome (a1))
POPULATIONS = {'S': S_WIRINGS, 'T': T_WIRINGS, 'all8': ALL_WIRINGS, 'draws7': ALL_WIRINGS[1:]}
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

_FITS: Dict[str, dict] = {}
_PARTS: Dict[str, dict] = {}
_TARGETS: Dict[str, pd.Series] = {}


def _cs(s: pd.Series) -> pd.Series:
    """A pair Series in canonical (sorted) pair order, as E77's."""
    s = s.copy()
    s.index = pd.MultiIndex.from_tuples([tuple(sorted(x)) for x in s.index], names=['odor_a', 'odor_b'])
    return s.sort_index()


def _n(responses: pd.DataFrame) -> int:
    return int((responses.to_numpy(dtype=float) > 0.5).sum())


def _rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def members() -> Dict[str, Tuple[str, str]]:
    """{label: (folder, tag)} of E72's eight wirings of C2's family at their own fits (E77's `members`, E72's
    `new_fit_members`: the lower of a draw's two starts), in ALL_WIRINGS order."""
    import E77_input_side_common_error as E77
    m = {lab: (f, t) for lab, f, t in E77.members()}
    if set(m) != set(ALL_WIRINGS):
        raise RuntimeError(f'expected the wirings {ALL_WIRINGS}, found {sorted(m)}')
    return {lab: m[lab] for lab in ALL_WIRINGS}


def wiring_kws(label: str) -> dict:
    """The `E10_joint_fit.bouton_base` wiring kwargs of a wiring label, as E77's `wiring_simulator` passes them: {} for
    'real', `wiring='claw_permutation_within_type', wiring_seed=n` for Pn, and E59's matched curveball matrix with its
    `wiring_tag` for Mn."""
    import E72_wiring_population as E72
    if label == 'real':
        return {}
    prefix = dict(E72.DRAWS)[label]
    seed = int(prefix.split('seed')[1].rstrip('_'))
    if label.startswith('P'):
        return dict(wiring='claw_permutation_within_type', wiring_seed=seed)
    m59 = H.shared_dir() / 'data' / 'model_outputs' / 'E59_coinput_shuffle'
    return dict(wiring_matrix=pd.read_parquet(m59 / f'wPNKC_curveball_within_type-seed{seed}_matched.parquet'),
                wiring_tag=f'curveball_within_type_matched-seed{seed}')


def saved_fit(label: str) -> dict:
    """A wiring's fit without any simulation, cached: 'label', 'folder', 'tag', 'p' (absolute parameters, E72's
    `fitted_params`), 'saved' ({'on', 'off'}: (responses, spike counts)), 'target_on' / 'target_off' (response counts,
    two or more spikes), 'size' (KCs x odors), 'K' (the calibrated readout's K on the fit's APL-on counts)."""
    if label not in _FITS:
        import E72_wiring_population as E72
        folder, tag = members()[label]
        saved = E72.load_fit(folder, tag)
        if saved is None:
            raise RuntimeError(f'{label}: no saved outputs for {folder}:{tag}')
        _FITS[label] = dict(label=label, folder=folder, tag=tag, p=E72.fitted_params(folder, tag), saved=saved,
                            target_on=_n(saved['on'][0]), target_off=_n(saved['off'][0]), size=int(saved['on'][0].size),
                            K=readout_K(saved))
    return _FITS[label]


def wiring_parts(label: str) -> dict:
    """A wiring's pieces for the lab's input, computed once per process (one simulation of the bouton base, plus its
    APL-off breadth run the first time, cached under outputs/E83_cross_wiring_fit/): `saved_fit`'s keys and 'wnorm'
    (normalized APL weights), 'W' (claw x bouton wiring), 'breadth' (E10_joint_fit.bouton_base), 'orn' (the lab's
    input)."""
    if label not in _PARTS:
        import E10_joint_fit as J
        orn = H.apl16_orn_deltas()
        _, wnorm, W, breadth = J.bouton_base(orn, H.outputs_dir(EXPERIMENT), None, **wiring_kws(label))
        _PARTS[label] = dict(saved_fit(label), wnorm=wnorm, W=W, breadth=breadth, orn=orn)
    return _PARTS[label]


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
        (simulate, info): info has 'label', 'params', 'breadth', 'wAPLKC' (breadth-weighted APL->KC weights),
        'wAPLPN', 'W', 'orn', 'cells' (the simulator's closure cells: 'wAPLKC_unscaled' / 'wAPLPN_unscaled' for
        in-place profile changes, 'mp', 'rv', 'odor_order').
    """
    import E10_joint_fit as J
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    P = wiring_parts(label)
    p = dict(P['p'] if params is None else params)
    wnorm, W = P['wnorm'], P['W']
    if orn is None:
        orn = P['orn']
        if breadth is None:
            breadth = P['breadth']
    elif breadth is None:
        if input_tag is None:
            raise ValueError('another ORN input with its breadth recomputed needs input_tag (input_tag_for(orn))')
        _, wnorm, W, breadth = J.bouton_base(orn, H.outputs_dir(EXPERIMENT), None, input_tag=input_tag, **wiring_kws(label))
        if not W.equals(P['W']):
            raise RuntimeError(f'{label}: the wiring built with input {input_tag} differs from the lab input\'s')
    wAPLKC = J.breadth_weights(wnorm['wAPLKC'], breadth, BREADTH_K)
    sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p, **J.base_kws(None), _wPNKC=W, _wAPLKC=wAPLKC,
                                           _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'],
                                           kc_kc_inh=p['wKCKC_scale'], kc_kc_tau=J.TAU)
    cells = dict(zip(sim.__code__.co_freevars, sim.__closure__))
    if not (np.array_equal(cells['wAPLKC_unscaled'].cell_contents.ravel(), wAPLKC.to_numpy())
            and np.array_equal(cells['wAPLPN_unscaled'].cell_contents.ravel(), wnorm['wAPLPN'].to_numpy())):
        raise RuntimeError("the simulator's unscaled APL weights are not the ones it was built with")
    return sim, dict(label=label, params=p, breadth=breadth, wAPLKC=wAPLKC, wAPLPN=wnorm['wAPLPN'], W=W, orn=orn,
                     cells=cells)


def input_tag_for(orn: pd.DataFrame, prefix: str = 'E83') -> str:
    """A cache tag naming one ORN input: `prefix` and the first 12 hex digits of its `E77_input_screen.frame_sha256`."""
    from E77_input_screen import frame_sha256
    return f'{prefix}_{frame_sha256(orn)[:12]}'


def _check_saved(label: str, sim, p: Dict[str, float], saved: Dict[str, tuple]) -> None:
    """At the fit's parameters the simulator reproduces the saved APL-on and APL-off responses and spike counts."""
    for c, q in (('on', dict(p)), ('off', dict(p, wAPLKC_scale=0.0, wAPLPN_scale=0.0))):
        r, s = sim(q)
        rs, ss = saved[c]
        ns = int((s.to_numpy(dtype=float) != ss.to_numpy(dtype=float)).sum())
        nr = int((r.to_numpy(dtype=float) != rs.to_numpy(dtype=float)).sum())
        if ns or nr or not s.index.equals(ss.index) or not s.columns.equals(ss.columns):
            raise RuntimeError(f'{label}: the simulator differs from the saved {c} run in {ns} counts and {nr} responses')


class WiringModel:
    """One wiring at its fit with the lab's input, for profiles of APL's weight (E75's ProfileModel on any wiring): a
    fixed-parameter simulator whose APL->KC and APL->bouton weights a profile changes in place. `run_on(x)` runs APL on
    with the APL->KC, KC->APL and bouton->APL scales times x (TIED_ON; APL->bouton and the somatic threshold unchanged);
    the APL-off run does not depend on APL's weights and is the saved fit's. With `check`, the constructor verifies that
    at factor 1 the simulator reproduces the saved APL-on and APL-off spike counts exactly (raises otherwise).

    Attributes: label, folder, tag, p, saved, target_on, target_off, size, K, W, wAPLKC, wAPLPN, sim, info."""

    def __init__(self, label: str, check: bool = True):
        P = wiring_parts(label)
        self.label = label
        for k in ('folder', 'tag', 'p', 'saved', 'target_on', 'target_off', 'size', 'K', 'W'):
            setattr(self, k, P[k])
        self.sim, self.info = build_simulator(label)
        self.wAPLKC, self.wAPLPN = self.info['wAPLKC'], self.info['wAPLPN']
        cells = self.info['cells']
        self._kc = cells['wAPLKC_unscaled'].cell_contents
        self._pn = cells['wAPLPN_unscaled'].cell_contents
        self._kc0, self._pn0 = self._kc.copy(), self._pn.copy()
        self.site, self.scores = None, {}
        if check:
            _check_saved(label, self.sim, self.p, self.saved)

    def set_profile(self, site: Optional[str], scores: Mapping[str, float]) -> None:
        """Set the APL weights to the fit's times a profile (E75's `profile_factors` at `site`, 'cell' / 'claw' /
        'bouton', or several joined by '+' with scores keyed 'site:glomerulus'; each wiring's own claws); None or no
        scores restores the fit's weights exactly."""
        import E75_glomerulus_apl_profile as E75
        self._kc[...] = self._kc0
        self._pn[...] = self._pn0
        self.site, self.scores = site, dict(scores)
        if site is None or not scores:
            return
        fkc = np.ones(self._kc0.size)
        fpn = np.ones(self._pn0.size)
        for st, sc in E75.split_scores(site, dict(scores)).items():
            f = E75.profile_factors(st, sc, self.W, self.wAPLKC, self.wAPLPN).to_numpy()
            if st == 'bouton':
                fpn = fpn * f
            else:
                fkc = fkc * f
        self._pn[...] = self._pn0 * fpn.reshape(self._pn0.shape)
        self._kc[...] = self._kc0 * fkc.reshape(self._kc0.shape)

    def current_wAPLKC(self) -> pd.Series:
        """The APL->KC weights (unscaled) the simulator holds now, indexed as `wAPLKC`."""
        return pd.Series(self._kc.ravel().copy(), index=self.wAPLKC.index)

    def run_on(self, x: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """(responses, spike counts) with APL on at the fit's APL scales (TIED_ON) times `x`, the current profile."""
        return self.sim(dict(self.p, **{k: self.p[k] * x for k in TIED_ON}))

    def outputs(self, on: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict[str, tuple]:
        """{'on': `on`, 'off': the saved fit's APL-off run}."""
        return {'on': on, 'off': self.saved['off']}


# ---------------------------------------------------------------------------
# changes: profiles and inputs, at a given log size

def e75_profile(name: str) -> Dict[str, float]:
    """Part A1's profile `name` (A1_POINTS) as scores {glomerulus: score}: E75's profiles.csv scores times the
    multiplier, permuted over the glomeruli as E75's D2 stage 1 permutes them (seed), or C2's four glomeruli at ln 2."""
    import json
    import E75_glomerulus_apl_profile as E75
    src, mult, seed = A1_POINTS[name]
    if src is None:
        return {g: float(np.log(2)) for g in E75.C2_FOUR}
    prof = pd.read_csv(H.results_dir(E75.EXPERIMENT) / 'profiles.csv')
    row = prof[(prof.profile == src[0]) & (prof.variant == src[1])]
    if len(row) != 1:
        raise RuntimeError(f'E75 profiles.csv: {len(row)} rows for {src}')
    sc = json.loads(row.iloc[0].scores)
    if seed is None:
        return {g: v * mult for g, v in sc.items()}
    keys = list(sc)
    perm = np.random.default_rng(seed).permutation(len(keys))
    return {keys[i]: sc[keys[j]] * mult for i, j in enumerate(perm)}


def scale_scores(scores: Mapping[str, float], size: float) -> Dict[str, float]:
    """A profile at `size` times its log size (scores times size)."""
    return {g: float(v) * float(size) for g, v in scores.items()}


def read_factors(path: Path) -> pd.DataFrame:
    """An input's factors as glomeruli x odors (abbreviated, the lab input's order), 1 where none is given, from a long
    table with columns odor, glomerulus, factor (E77's oracle_factors_<half>.csv, E83's input_factors_<half>.csv)."""
    orn = H.apl16_orn_deltas()
    f = pd.read_csv(path)
    w = f.pivot(index='glomerulus', columns='odor', values='factor')
    ab = list(H.abbrev_columns(orn).columns)
    extra = (set(w.index) - set(orn.index)) | (set(w.columns) - set(ab))
    if extra:
        raise ValueError(f'{path}: glomeruli / odors not in the lab input: {sorted(extra)}')
    return w.reindex(index=orn.index, columns=ab).fillna(1.0).astype(float)


def input_at(factors: pd.DataFrame, size: float, orn: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The ORN input with each odor's per-glomerulus factors at `size` of their log size: orn x factor ** size (orn the
    lab's by default, `H.apl16_orn_deltas()` layout); size 0 gives the lab's input exactly."""
    orn = H.apl16_orn_deltas() if orn is None else orn
    out = orn.astype(float).copy()
    if float(size) == 0.0:
        return out
    ab = list(H.abbrev_columns(orn).columns)
    F = factors.reindex(index=orn.index, columns=ab).fillna(1.0).to_numpy(dtype=float)
    out[:] = out.to_numpy() * np.power(F, float(size))
    return out


def profile_reading(model: 'WiringModel', scores: Mapping[str, float], site: str = 'cell', label: str = '') -> dict:
    """A profile on one wiring at its fit: the APL scale (TIED_ON) bisected (E69's `bisect_factor`, APL_BRACKET,
    APL_STEP, from factor 1) to the wiring's APL-on response count; the APL-off run is the saved fit's. The model's
    profile is reset on return.

    Returns:
        dict with 'on' / 'off' ((responses, spike counts)), 'apl_factor', 'thr_factor' (1), 'n_on', 'n_off',
        'target_on', 'target_off', 'at_bound_apl', 'n_sims', 'params' (absolute, APL on), 'history', 'wAPLKC' (the
        profiled APL->KC weights, unscaled), 'seconds'."""
    import time
    import E69_kc_firing_ceiling as E69
    t0 = time.time()
    model.set_profile(site, scores)
    try:
        wk = model.current_wAPLKC()
        x, on, hist, at_bound = E69.bisect_factor(model.run_on, model.target_on, model.size, APL_BRACKET, APL_STEP,
                                                  f'{label} APL scale')
    finally:
        model.set_profile(None, {})
    p = model.p
    return dict(on=on, off=model.saved['off'], apl_factor=float(x), thr_factor=1.0, n_on=_n(on[0]),
                n_off=_n(model.saved['off'][0]), target_on=model.target_on, target_off=model.target_off,
                at_bound_apl=bool(at_bound), at_bound_thr=False, n_sims=len(hist), history=hist,
                params=dict(p, **{k: p[k] * x for k in TIED_ON}), wAPLKC=wk, seconds=time.time() - t0)


def input_reading(label: str, orn: pd.DataFrame, input_tag: Optional[str] = None) -> dict:
    """An ORN input on one wiring at its fit, E77's reading: the breadth recomputed for the input
    (`build_simulator(label, orn, input_tag=...)`), the somatic threshold bisected to the fit's APL-off response count,
    then the APL scale (TIED_ON) to its APL-on count (`E77_input_screen.hold_rates`).

    Returns:
        hold_rates' dict ('on', 'off', 'params', 'params_off', 'thr_factor', 'apl_factor', 'n_on', 'n_off',
        'at_bound_thr', 'at_bound_apl', 'n_sims', 'history', 'breadth_sha256', ...) plus 'input_tag'."""
    import E77_input_screen as S
    P = saved_fit(label)
    tag = input_tag or input_tag_for(orn)
    res = S.hold_rates(lambda: build_simulator(label, orn, input_tag=tag), target_off=P['target_off'],
                       target_on=P['target_on'], params=P['p'], label=f'{label} {tag}')
    res['input_tag'] = tag
    return res


# ---------------------------------------------------------------------------
# targets, distances, populations, scores

def targets() -> Dict[str, pd.Series]:
    """The data's per-pair APL effect (canonical pair order) for every name in TARGETS: 'halfA' / 'halfB' (E77's halves,
    `half_effect`), 'z8' (all recordings, Yang's z: the main target), 'z6' (all control recordings, the six typical TNT
    recordings), 'noise_free' (E81's noise_free_pairs.csv, units z, effect_noise_free), 'noise8' (E76's
    effect_by_unit_pairs.csv in baseline-noise units)."""
    if not _TARGETS:
        import E77_input_side_common_error as E77
        halves = E77.recording_halves()
        t = {f'half{h}': _cs(E77.half_effect(halves[h])) for h in ('A', 'B')}
        for k in ('z8', 'z6', 'noise8'):
            dd = E77.data_distances(k)
            t[k] = _cs(H.apl_effect(dd['on'], dd['off']))
        nf = pd.read_csv(H.shared_dir() / 'results' / 'E81_repeat_structure' / 'noise_free_pairs.csv')
        nf = nf[nf.units == 'z'].set_index(['odor_a', 'odor_b'])['effect_noise_free']
        t['noise_free'] = _cs(nf)
        for k, v in t.items():
            if len(v) != 120 or v.isna().any():
                raise RuntimeError(f'target {k}: {len(v)} pairs, {int(v.isna().sum())} missing')
        _TARGETS.update({k: t[k] for k in TARGETS})
    return dict(_TARGETS)


def half_target(half: str) -> pd.Series:
    """The per-pair APL effect of half 'A' or 'B' of the recordings (E77's `recording_halves()` and `half_effect`)."""
    return targets()[f'half{half}']


def readout_K(o: Dict[str, tuple]) -> float:
    """The calibrated readout's K on a run's APL-on spike counts (as E72's `distances` sets it)."""
    from E67_apl_effect_fits import refs
    return float(H.calibrate_readout(o['on'][1], 'saturating', refs()['cal_target'], stat='participation_ratio',
                                     responders_min=2.0))


def pair_distances(o: Dict[str, tuple], readout: str, K: Optional[float] = None) -> Dict[str, pd.Series]:
    """{'on', 'off'} pair distances (canonical order) of a run: readout 'raw' (cosine on counts), 'cal' (the calibrated
    readout, K re-calibrated on this run's APL-on counts unless `K` is given: the readings' default, as E75's and E77's)
    or 'calK' (needs `K`, e.g. the wiring's fit's: the searches' objective)."""
    s_on, s_off = o['on'][1], o['off'][1]
    if readout == 'raw':
        d = {'on': H.readout_pair_distances(s_on, 'raw'), 'off': H.readout_pair_distances(s_off, 'raw')}
    elif readout in ('cal', 'calK'):
        if K is None:
            if readout == 'calK':
                raise ValueError("readout 'calK' needs K")
            K = readout_K(o)
        d = {'on': H.readout_pair_distances(s_on, 'saturating', K), 'off': H.readout_pair_distances(s_off, 'saturating', K)}
    else:
        raise ValueError(readout)
    return {c: _cs(v) for c, v in d.items()}


def population(ds: Sequence[Dict[str, pd.Series]]) -> Dict[str, pd.Series]:
    """The mean over wirings of the pair distances per condition (E72's `mean_distances`), canonical order."""
    return {c: _cs(pd.concat([_cs(d[c]) for d in ds], axis=1).mean(axis=1)) for c in ('on', 'off')}


def pattern_error(d: Dict[str, pd.Series], target: pd.Series) -> pd.Series:
    """Model minus data per-pair APL effect, each with its mean removed (canonical order)."""
    me = _cs(H.apl_effect(d['on'], d['off']))
    de = _cs(target).reindex(me.index)
    return (me - me.mean()) - (de - de.mean())


def effect_scores(d: Dict[str, pd.Series], tg: Optional[Dict[str, pd.Series]] = None) -> dict:
    """The scores of a run's or a population's distances against every target: per target '<t>_loss' (rms of model
    minus data per-pair effect), '<t>_pattern' (the same with both means removed), '<t>_rho' (ρ on the per-pair APL
    effect), '<t>_err_ester' / '<t>_err_alcohol' (mean pattern error over E75's groups' pairs), and 'mean_effect'."""
    import E77_input_side_common_error as E77
    tg = targets() if tg is None else tg
    me = _cs(H.apl_effect(d['on'], d['off']))
    masks = E77.kind_masks(me.index)
    row = {'mean_effect': float(me.mean())}
    for t, de0 in tg.items():
        de = _cs(de0).reindex(me.index)
        err = (me - me.mean()) - (de - de.mean())
        row[f'{t}_loss'] = _rms((me - de).to_numpy())
        row[f'{t}_pattern'] = _rms(err.to_numpy())
        row[f'{t}_rho'] = H.spearman(me, de)
        row[f'{t}_err_ester'] = float(err[masks['ester']].mean())
        row[f'{t}_err_alcohol'] = float(err[masks['alcohol']].mean())
    return row


def _counts(x) -> pd.DataFrame:
    return x[1] if isinstance(x, tuple) else x


class PopulationObjective:
    """The searches' objective: the squared pattern error of a population's per-pair APL effect under the calibrated
    readout, each wiring's K held at its fit's (`saved_fit(label)['K']`), against a target (a half's effect). The
    distances of a wiring or a condition not given are its saved fit's (computed once).

    Usage: `obj = PopulationObjective(half_target('A'))`; `obj({'real': s_on, ...})` with APL-on spike counts (or
    (responses, spike counts)) per wiring (arm P), or `obj(on={...}, off={...})` (arm I). Returns dict(sse=...,
    rms=sqrt(sse / n_pairs), n_pairs=...). At the fits (no change) rms is IDENTITY_OBJECTIVE[half] to four decimals
    (0.0569 against half A, 0.0552 against half B)."""

    def __init__(self, target: pd.Series, wirings: Sequence[str] = S_WIRINGS):
        self.target = _cs(target)
        self.wirings = tuple(wirings)
        self.K = {w: saved_fit(w)['K'] for w in self.wirings}
        self.base = {w: pair_distances(saved_fit(w)['saved'], 'calK', self.K[w]) for w in self.wirings}

    def distances(self, on: Optional[Mapping[str, object]] = None, off: Optional[Mapping[str, object]] = None
                  ) -> Dict[str, pd.Series]:
        """The population's mean distances with the given runs (APL-on / APL-off spike counts per wiring)."""
        on, off = on or {}, off or {}
        ds = []
        for w in self.wirings:
            d = dict(self.base[w])
            for c, given in (('on', on), ('off', off)):
                if w in given:
                    d[c] = _cs(H.readout_pair_distances(_counts(given[w]), 'saturating', self.K[w]))
            ds.append(d)
        return population(ds)

    def __call__(self, on: Optional[Mapping[str, object]] = None, off: Optional[Mapping[str, object]] = None) -> dict:
        err = pattern_error(self.distances(on, off), self.target)
        sse = float((err.to_numpy() ** 2).sum())
        return dict(sse=sse, rms=float(np.sqrt(sse / len(err))), n_pairs=int(len(err)))


def realism(o: Dict[str, tuple]) -> dict:
    """A run's response rates, their ratio, the response rate ρ and the odors-per-KC distances under both rules, the
    largest spike counts and the flags (E77's `realism_of`)."""
    import E77_input_side_common_error as E77
    return E77.realism_of(o)


def _run_dir(part: str) -> Path:
    d = H.outputs_dir(EXPERIMENT) / part
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(part: str, tag: str, o: Dict[str, tuple]) -> None:
    """Spike counts and responses of a run under outputs/E83_cross_wiring_fit/<part>/ (<tag>_<on|acute_off>_...)."""
    d = _run_dir(part)
    for c, cc in (('on', 'on'), ('off', 'acute_off')):
        o[c][0].to_parquet(d / f'{tag}_{cc}_responses.parquet')
        o[c][1].to_parquet(d / f'{tag}_{cc}_spike_counts.parquet')


def _load_pair(d: Path, tag: str) -> Optional[Dict[str, tuple]]:
    out = {}
    for c, cc in (('on', 'on'), ('off', 'acute_off')):
        ps = [d / f'{tag}_{cc}_{k}.parquet' for k in ('responses', 'spike_counts')]
        if not all(p.exists() for p in ps):
            return None
        out[c] = tuple(pd.read_parquet(p) for p in ps)
    return out


def load_run(part: str, tag: str) -> Optional[Dict[str, tuple]]:
    """A saved run of this experiment ({'on', 'off'}): the local outputs, else the model-output archive; or None."""
    for d in (H.HERE / 'outputs' / EXPERIMENT / part, H.shared_dir() / 'data' / 'model_outputs' / EXPERIMENT / part):
        o = _load_pair(d, tag)
        if o is not None:
            return o
    return None


def e77_oracle_run(half: str, label: str) -> Optional[Dict[str, tuple]]:
    """E77's archived reading of its half-`half` searched input at full size on wiring `label` (S only)."""
    tag = f'oracle_half{half}_{label}'
    for d in (H.HERE / 'outputs' / 'E77_input_side_common_error' / 'oracle',
              H.shared_dir() / 'data' / 'model_outputs' / 'E77_input_side_common_error' / 'oracle'):
        o = _load_pair(d, tag)
        if o is not None:
            return o
    return None


def score_runs(runs: Mapping[Tuple[str, str], Dict[str, tuple]], change_meta: Optional[Mapping[str, dict]] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Readings: for every (change, wiring) run and every population of POPULATIONS whose wirings all have a run of the
    change, `effect_scores` under 'cal', 'calK' (each wiring's fit's K) and 'raw', with the paired change against the
    same unit unchanged (change 'none': the saved fits, added here if absent).

    Returns:
        (readings, pairs): readings one row per (change, unit, readout) with unit a wiring or a population name (and
        the change's metadata from `change_meta`); pairs the per-pair APL effect per (change, unit, readout)."""
    runs = dict(runs)
    for w in ALL_WIRINGS:
        runs.setdefault(('none', w), saved_fit(w)['saved'])
    change_meta = dict(change_meta or {})
    tg = targets()
    changes = list(dict.fromkeys(c for c, _ in runs))
    rows, prs = [], []
    for ch in changes:
        dist = {ro: {} for ro in ('cal', 'calK', 'raw')}
        for w in ALL_WIRINGS:
            if (ch, w) not in runs:
                continue
            o = runs[(ch, w)]
            dist['cal'][w] = pair_distances(o, 'cal')
            dist['calK'][w] = pair_distances(o, 'calK', saved_fit(w)['K'])
            dist['raw'][w] = pair_distances(o, 'raw')
        units = [(w, 'wiring', (w,)) for w in ALL_WIRINGS if w in dist['cal']]
        units += [(name, 'population', ws) for name, ws in POPULATIONS.items() if all(w in dist['cal'] for w in ws)]
        for unit, kind, ws in units:
            for ro in ('cal', 'calK', 'raw'):
                d = dist[ro][unit] if kind == 'wiring' else population([dist[ro][w] for w in ws])
                row = dict(change=ch, unit=unit, kind=kind, n_wirings=len(ws), readout=ro, **change_meta.get(ch, {}),
                           **effect_scores(d, tg))
                rows.append(row)
                me = _cs(H.apl_effect(d['on'], d['off']))
                prs.append(pd.DataFrame(dict(change=ch, unit=unit, readout=ro, odor_a=me.index.get_level_values(0),
                                             odor_b=me.index.get_level_values(1), on=d['on'].reindex(me.index).to_numpy(),
                                             off=d['off'].reindex(me.index).to_numpy(), effect=me.to_numpy())))
    res = pd.DataFrame(rows)
    base = res[res.change == 'none'].set_index(['unit', 'readout'])
    score_cols = [c for c in res.columns if c.endswith(('_loss', '_pattern', '_rho')) or c == 'mean_effect']
    for c in score_cols:
        res[f'd_{c}'] = [v - base[c].get((u, ro), np.nan) for u, ro, v in zip(res.unit, res.readout, res[c])]
    return res, pd.concat(prs, ignore_index=True)


def two_by_two(readings: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Per arm and direction (searched on A, scored on B; searched on B, scored on A): S and T against the half searched
    on and the half never seen, unchanged and with the change at GATE_SIZE and at full size, and each single wiring as a
    paired difference against its own unchanged fit. Changes are named '<arm>_half<X>_s<size>' (part A2's arm 'A2',
    part D's 'DP' and 'DI'); readout 'cal'."""
    r = readings[readings.readout == 'cal']
    base = r[r.change == 'none'].set_index('unit')
    rows = []
    for X in ('A', 'B'):
        Y = 'B' if X == 'A' else 'A'
        for size in (GATE_SIZE, 1.0):
            ch = f'{arm}_half{X}_s{size:g}'
            g = r[r.change == ch].set_index('unit')
            if g.empty:
                continue
            for unit in list(POPULATIONS) + list(ALL_WIRINGS):
                if unit not in g.index:
                    continue
                for which, t in (('searched', f'half{X}'), ('unseen', f'half{Y}'), ('all', 'z8'), ('six', 'z6')):
                    u0, u1 = float(base.loc[unit, f'{t}_pattern']), float(g.loc[unit, f'{t}_pattern'])
                    rows.append(dict(arm=arm, searched_on=X, size=size, unit=unit, kind=g.loc[unit, 'kind'], against=which,
                                     target=t, unchanged=u0, changed=u1, gain=u0 - u1,
                                     sq_reduction=1.0 - (u1 ** 2) / (u0 ** 2)))
    return pd.DataFrame(rows, columns=['arm', 'searched_on', 'size', 'unit', 'kind', 'against', 'target', 'unchanged',
                                       'changed', 'gain', 'sq_reduction'])


GATE_GAIN, GATE_BORDER = 0.003, (0.0015, 0.0045)
GATE_RAW_SLACK, GATE_MEAN_SLACK, GATE_SQ_ALL = 0.005, 0.01, 0.15


def gate(readings: pd.DataFrame, arm: str, runs: Optional[pd.DataFrame] = None,
         controls: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The entry's gate per arm at GATE_SIZE in both directions: (i) T's pattern part against the unseen half lower by
    0.003 or more; (ii) lower on at least six of T's eight wiring-by-direction cells; (iii) every point inside the four
    bands (`runs`' realism_flag, the rule of 16:10), T's raw-count loss no more than 0.005 above its unchanged value
    (E77's reading of the slack), its mean
    APL effect under the readout within 0.01 of the flies'; (iv) T's squared pattern error against all recordings 15 %
    lower (mean of the directions), half that against the six typical recordings; (v) the label-permuted controls
    (`controls`, a two_by_two of the permuted changes, when run) gain less than half as much on (i); and the
    borderline rule (i between 0.0015 and 0.0045). One row per direction and one 'both' row with the verdict."""
    tb = two_by_two(readings, arm)
    tb = tb[np.isclose(tb['size'], GATE_SIZE)] if len(tb) else tb
    flies_mean = float(targets()['z8'].mean())
    rows = []
    for X in ('A', 'B'):
        t = tb[(tb.searched_on == X) & (tb.unit == 'T')].set_index('against')
        if t.empty:
            continue
        singles = tb[(tb.searched_on == X) & tb.unit.isin(T_WIRINGS) & (tb.against == 'unseen')]
        ch = f'{arm}_half{X}_s{GATE_SIZE:g}'
        rawT = readings[(readings.change == ch) & (readings.unit == 'T') & (readings.readout == 'raw')]
        raw0 = readings[(readings.change == 'none') & (readings.unit == 'T') & (readings.readout == 'raw')]
        calT = readings[(readings.change == ch) & (readings.unit == 'T') & (readings.readout == 'cal')]
        flags = None
        if runs is not None:
            rr = runs[runs.change == ch]
            flags = int((rr.realism_flag.fillna('').astype(str).str.strip() != '').sum()) if len(rr) else None
        row = dict(arm=arm, searched_on=X, gain_unseen=float(t.loc['unseen', 'gain']), n_singles_lower=int((singles.gain > 0).sum()),
                   n_singles=int(len(singles)), raw_loss_change=float(rawT.z8_loss.iloc[0] - raw0.z8_loss.iloc[0]) if len(rawT) else np.nan,
                   mean_effect_cal=float(calT.mean_effect.iloc[0]) if len(calT) else np.nan, flies_mean=flies_mean,
                   sq_reduction_all=float(t.loc['all', 'sq_reduction']), sq_reduction_six=float(t.loc['six', 'sq_reduction']),
                   n_flagged=flags)
        row['i'] = row['gain_unseen'] >= GATE_GAIN
        row['borderline'] = GATE_BORDER[0] <= row['gain_unseen'] <= GATE_BORDER[1]
        if controls is not None and len(controls):
            c = controls[(controls.searched_on == X) & (controls.unit == 'T') & (controls.against == 'unseen')]
            row['control_gain_max'] = float(c.gain.max()) if len(c) else np.nan
            row['v'] = bool(len(c) and (c.gain < 0.5 * row['gain_unseen']).all())
        rows.append(row)
    res = pd.DataFrame(rows)
    if len(res) == 2:
        both = dict(arm=arm, searched_on='both', gain_unseen=float(res.gain_unseen.min()),
                    n_singles_lower=int(res.n_singles_lower.sum()), n_singles=int(res.n_singles.sum()),
                    raw_loss_change=float(res.raw_loss_change.max()),
                    mean_effect_cal=float(res.mean_effect_cal.iloc[np.argmax(np.abs(res.mean_effect_cal - flies_mean))]),
                    flies_mean=flies_mean, sq_reduction_all=float(res.sq_reduction_all.mean()),
                    sq_reduction_six=float(res.sq_reduction_six.mean()),
                    n_flagged=None if res.n_flagged.isna().any() else int(res.n_flagged.sum()))
        both['i'] = bool(res['i'].all())
        both['ii'] = both['n_singles_lower'] >= 6
        both['iii'] = bool((both['n_flagged'] == 0) and both['raw_loss_change'] <= GATE_RAW_SLACK
                           and abs(both['mean_effect_cal'] - flies_mean) <= GATE_MEAN_SLACK)
        both['iv'] = both['sq_reduction_all'] >= GATE_SQ_ALL and both['sq_reduction_six'] >= 0.5 * GATE_SQ_ALL
        both['borderline'] = bool(res['borderline'].any())
        if 'v' in res.columns:
            both['v'] = bool(res['v'].all())
        both['passes_i_to_iv'] = bool(both['i'] and both['ii'] and both['iii'] and both['iv'])
        res = pd.concat([res, pd.DataFrame([both])], ignore_index=True)
    return res


# ---------------------------------------------------------------------------
# parts

def _git_head() -> str:
    import subprocess
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()


def _append_row(path: Path, row: dict) -> None:
    pd.DataFrame([row]).to_csv(path, index=False, mode='a', header=not path.exists())


def _read_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _provenance_add(path: Path, part: str, tag: str, prov: dict) -> None:
    import json
    d = json.loads(path.read_text()) if path.exists() else dict(part=part, script='E83_cross_wiring_fit.py', points={})
    d['points'][tag] = prov
    path.write_text(json.dumps(d, indent=1, default=str) + '\n')


PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')


def _run_row(part: str, change: str, w: str, tag: str, res: dict, o: Dict[str, tuple], **extra) -> dict:
    P = saved_fit(w)
    row = dict(part=part, change=change, wiring=w, tag=tag, fit_folder=P['folder'], fit_tag=P['tag'], **extra,
               thr_factor=res['thr_factor'], apl_factor=res['apl_factor'], n_on=res['n_on'], n_off=res['n_off'],
               target_on=res['target_on'], target_off=res['target_off'], at_bound_thr=res['at_bound_thr'],
               at_bound_apl=res['at_bound_apl'], n_sims=res['n_sims'], seconds=res.get('seconds'),
               **{k: res['params'][k] for k in PARAM_KEYS}, K_fit=P['K'], K_run=readout_K(o),
               breadth_sha256=res.get('breadth_sha256'), al_analysis_commit=_git_head())
    row.update(realism(o))
    return row


def _e75_archived(tag: str) -> Dict[str, tuple]:
    import E72_wiring_population as E72
    o = E72.load_fit('E75_glomerulus_apl_profile', tag)
    if o is None:
        raise RuntimeError(f"E75's archived point {tag} not found")
    return o


def _same(a: Dict[str, tuple], b: Dict[str, tuple]) -> Tuple[int, int]:
    """(differing spike counts, differing responses) between two runs, both conditions."""
    ns = nr = 0
    for c in ('on', 'off'):
        if not (a[c][1].index.equals(b[c][1].index) and a[c][1].columns.equals(b[c][1].columns)):
            return -1, -1
        ns += int((a[c][1].to_numpy(dtype=float) != b[c][1].to_numpy(dtype=float)).sum())
        nr += int((a[c][0].to_numpy(dtype=float) != b[c][0].to_numpy(dtype=float)).sum())
    return ns, nr


def identity(out: Path, wirings: Sequence[str] = ALL_WIRINGS) -> dict:
    """The identity checks: every wiring's simulator reproduces its saved fit at factor 1 and zero scores; A1's two
    archived E75 points on the real wiring reproduced exactly; the population objective at identity. Writes
    identity.json."""
    import json
    res = dict(al_analysis_commit=_git_head(), wirings={}, e75_points={}, objective={})
    for h in ('A', 'B'):
        v = PopulationObjective(half_target(h))({})
        res['objective'][h] = dict(rms=v['rms'], probe=IDENTITY_OBJECTIVE[h], ok=round(v['rms'], 4) == IDENTITY_OBJECTIVE[h])
    for w in wirings:
        m = WiringModel(w, check=True)
        m.set_profile('cell', {g: 0.0 for g in set(m.W.columns.get_level_values('glomerulus'))})
        r, s = m.run_on(1.0)
        ns = int((s.to_numpy(dtype=float) != m.saved['on'][1].to_numpy(dtype=float)).sum())
        res['wirings'][w] = dict(fit=f'{m.folder}:{m.tag}', saved_reproduced=True, zero_scores_count_diffs=ns)
        if w == 'real':
            for name, tag in E75_ARCHIVED.items():
                rd = profile_reading(m, e75_profile(name), 'cell', label=name)
                ns, nr = _same({'on': rd['on'], 'off': rd['off']}, _e75_archived(tag))
                res['e75_points'][tag] = dict(apl_factor=rd['apl_factor'], count_diffs=ns, response_diffs=nr)
        del m
        print(json.dumps({w: res['wirings'][w]}), flush=True)
    (out / 'identity.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


def part_a1(out: Path, wirings: Sequence[str] = ALL_WIRINGS, points: Sequence[str] = tuple(A1_POINTS)) -> pd.DataFrame:
    """Part A1 (see the module docstring); resumable (runs saved under outputs/.../A1/, rows in runs_A1.csv, provenance
    per point in provenance_A1.json). On the real wiring every point's APL factor and response count must equal E75's
    (sensitivity_map.csv) and the two archived points' outputs E75's exactly (raises otherwise)."""
    import gc
    import hashlib
    import json
    import E75_glomerulus_apl_profile as E75
    rp, pp = out / 'runs_A1.csv', out / 'provenance_A1.json'
    done = set(_read_rows(rp).get('tag', pd.Series(dtype=str)))
    m75 = pd.read_csv(H.results_dir(E75.EXPERIMENT) / 'sensitivity_map.csv').drop_duplicates('point_id', keep='last').set_index('point_id')
    for w in wirings:
        todo = [n for n in points if f'A1_{n}_{w}' not in done]
        if not todo:
            continue
        model = WiringModel(w, check=True)
        print(f'{w}: the simulator reproduces {model.folder}:{model.tag}', flush=True)
        for n in todo:
            tag = f'A1_{n}_{w}'
            sc = e75_profile(n)
            res = profile_reading(model, sc, 'cell', label=tag)
            o = {'on': res['on'], 'off': res['off']}
            if w == 'real':
                e = m75.loc[E75_POINT_IDS[n]]
                if json.loads(e.scores) != sc:
                    raise RuntimeError(f'{tag}: the scores differ from E75 {E75_POINT_IDS[n]}')
                if res['apl_factor'] != float(e.factor) or res['n_on'] != int(e.n_responses):
                    raise RuntimeError(f'{tag}: APL factor {res["apl_factor"]!r} / {res["n_on"]} responses, E75 '
                                       f'{float(e.factor)!r} / {int(e.n_responses)}')
                if n in E75_ARCHIVED:
                    ns, nr = _same(o, _e75_archived(E75_ARCHIVED[n]))
                    if ns or nr:
                        raise RuntimeError(f'{tag}: differs from E75 {E75_ARCHIVED[n]} in {ns} counts and {nr} responses')
                    print(f'{tag}: reproduces E75 {E75_ARCHIVED[n]} exactly', flush=True)
            save_run('A1', tag, o)
            sj = json.dumps(sc, sort_keys=True)
            row = _run_row('A1', f'A1_{n}', w, tag, res, o, site='cell', multiplier=A1_POINTS[n][1],
                           scores=json.dumps(sc), scores_sha256=hashlib.sha256(sj.encode()).hexdigest())
            _append_row(rp, row)
            _provenance_add(pp, 'A1', tag, H.model_provenance(orn_deltas=model.info['orn'], _wPNKC=model.W, _wAPLKC=res['wAPLKC']))
            d0, d1 = pair_distances(model.saved, 'cal'), pair_distances(o, 'cal')
            print(f"{tag}: APL x{res['apl_factor']:.4f} ({res['n_on']} of {res['target_on']}); pattern under the readout against "
                  f"all recordings {_rms(pattern_error(d0, targets()['z8'])):.4f} -> {_rms(pattern_error(d1, targets()['z8'])):.4f}; "
                  f"{res['seconds']:.0f}s", flush=True)
        del model
        gc.collect()
    return _read_rows(rp)


def _write_input(out: Path, name: str, orn: pd.DataFrame) -> Path:
    """An ORN input in E38's layout (glomeruli x abbreviated odors) as results/E83_cross_wiring_fit/inputs/<name>.csv."""
    d = out / 'inputs'
    d.mkdir(exist_ok=True)
    lay = H.abbrev_columns(orn)
    lay.index.name = 'glomerulus'
    lay.to_csv(d / f'{name}.csv')
    return d / f'{name}.csv'


def a2_input(half: str, size: float) -> pd.DataFrame:
    """E77's half-`half` searched input at `size` of its log factors; at full size exactly E77's reading of it
    (`read_orn_input` of inputs/oracle_half<X>.csv, checked against the factors to 1e-9)."""
    from E10_joint_fit import read_orn_input
    e77 = H.shared_dir() / 'results' / 'E77_input_side_common_error'
    F = read_factors(e77 / f'oracle_factors_{half}.csv')
    orn_in = input_at(F, size)
    if float(size) == 1.0:
        exact = read_orn_input(str(e77 / 'inputs' / f'oracle_half{half}.csv'), H.apl16_orn_deltas())
        dev = float(np.max(np.abs(exact.to_numpy() - orn_in.to_numpy()) / np.maximum(1.0, np.abs(exact.to_numpy()))))
        if dev > 1e-9:
            raise RuntimeError(f'E77 half {half}: the factors do not give its input (relative deviation {dev:.2g})')
        return exact
    return orn_in


def part_a2(out: Path, halves: Sequence[str] = A2_HALVES, check_e77: bool = True) -> pd.DataFrame:
    """Part A2 (see the module docstring); resumable. First (with `check_e77`) every wiring's simulator against its saved
    fit, and E77's archived reading of half A's input on the real wiring re-simulated through this reading and compared
    exactly (tag A2_check_halfA_s1_real, not a point of the part)."""
    import gc
    rp, pp = out / 'runs_A2.csv', out / 'provenance_A2.json'
    done = set(_read_rows(rp).get('tag', pd.Series(dtype=str)))
    plan = [(X, size, w) for X in halves for size, ws in ((GATE_SIZE, ALL_WIRINGS), (1.0, T_WIRINGS)) for w in ws]
    if check_e77:
        for w in ALL_WIRINGS:
            if any(f'A2_half{X}_s{size:g}_{ww}' not in done for X, size, ww in plan if ww == w):
                sim, info = build_simulator(w)
                _check_saved(w, sim, info['params'], saved_fit(w)['saved'])
                del sim, info
                print(f'{w}: the simulator reproduces {saved_fit(w)["folder"]}:{saved_fit(w)["tag"]}', flush=True)
        if 'A2_check_halfA_s1_real' not in done:
            orn_in = a2_input('A', 1.0)
            res = input_reading('real', orn_in, input_tag=input_tag_for(orn_in, 'E83_A2_halfA_s1'))
            o = {'on': res['on'], 'off': res['off']}
            ns, nr = _same(o, e77_oracle_run('A', 'real'))
            if ns or nr:
                raise RuntimeError(f"half A's input on the real wiring differs from E77's archived run in {ns} counts and {nr} responses")
            print("half A's input at full size on the real wiring reproduces E77's archived run exactly", flush=True)
            _append_row(rp, _run_row('A2', 'A2_check_halfA_s1', 'real', 'A2_check_halfA_s1_real', res, o, half='A', size=1.0,
                                     input_tag=res['input_tag'], reproduces='E77 oracle_halfA_real'))
            del res, o
    for X, size, w in plan:
        tag = f'A2_half{X}_s{size:g}_{w}'
        if tag in done:
            continue
        orn_in = a2_input(X, size)
        name = f'A2_half{X}_s{size:g}'
        path = _write_input(out, name, orn_in)
        res = input_reading(w, orn_in, input_tag=input_tag_for(orn_in, f'E83_{name}'))
        o = {'on': res['on'], 'off': res['off']}
        save_run('A2', tag, o)
        _append_row(rp, _run_row('A2', name, w, tag, res, o, half=X, size=size, input_tag=res['input_tag'],
                                 orn_input=str(path.relative_to(H.shared_dir()))))
        _provenance_add(pp, 'A2', tag, H.model_provenance(orn_deltas=orn_in))
        d0, d1 = pair_distances(saved_fit(w)['saved'], 'cal'), pair_distances(o, 'cal')
        Y = 'B' if X == 'A' else 'A'
        print(f"{tag}: thr x{res['thr_factor']:.4f}, APL x{res['apl_factor']:.4f}; pattern under the readout against the unseen "
              f"half {Y} {_rms(pattern_error(d0, half_target(Y))):.4f} -> {_rms(pattern_error(d1, half_target(Y))):.4f}; "
              f"{res['seconds']:.0f}s", flush=True)
        del res, o
        gc.collect()
    return _read_rows(rp)


def _part_a_runs(out: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, tuple]], Dict[str, dict], pd.DataFrame]:
    """Part A's runs {(change, wiring): outputs}, the changes' metadata, and the runs table (runs.csv's rows of part A,
    E77's archived readings of S at full size included from its oracle_runs.csv)."""
    rows = [r for r in (_read_rows(out / 'runs_A1.csv'), _read_rows(out / 'runs_A2.csv')) if len(r)]
    rt = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    rt = rt[~rt.change.astype(str).str.startswith('A2_check')] if len(rt) else rt
    runs, meta = {}, {}
    for _, r in rt.iterrows():
        o = load_run(r.part, r.tag)
        if o is None:
            print(f'{r.tag}: outputs missing', flush=True)
            continue
        runs[(r.change, r.wiring)] = o
        meta[r.change] = dict(part=r.part, size=float(r.get('size')) if pd.notna(r.get('size', np.nan)) else
                              float(r.get('multiplier', np.nan)))
    e77 = pd.read_csv(H.shared_dir() / 'results' / 'E77_input_side_common_error' / 'oracle_runs.csv')
    extra = []
    for X in A2_HALVES:
        ch = f'A2_half{X}_s1'
        for w in S_WIRINGS:
            o = e77_oracle_run(X, w)
            if o is None:
                print(f"E77's oracle_half{X}_{w}: outputs missing", flush=True)
                continue
            runs[(ch, w)] = o
            meta[ch] = dict(part='A2', size=1.0)
            q = e77[(e77.input == f'oracle_half{X}') & (e77.wiring == w)].iloc[0].to_dict()
            P = saved_fit(w)
            extra.append(dict(part='A2', change=ch, wiring=w, tag=f'oracle_half{X}_{w}', source='E77 (archived, not rerun)',
                              fit_folder=P['folder'], fit_tag=P['tag'], half=X, size=1.0, K_fit=P['K'], K_run=readout_K(o),
                              **{k: q[k] for k in q if k not in ('input', 'wiring', 'fit_folder', 'fit_tag')}))
    if extra:
        rt = pd.concat([rt, pd.DataFrame(extra)], ignore_index=True)
    return runs, meta, rt


def _replace_rows(path: Path, new: pd.DataFrame, key: str, values) -> None:
    old = _read_rows(path)
    if len(old) and key in old.columns:
        old = old[~old[key].isin(set(values))]
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def _write_archive_tables(out: Path, rt: pd.DataFrame) -> None:
    """model_pair_distances_long.csv rows (raw spike counts, the harness's primary model RDM) and variants.csv rows
    (absolute parameters) for this experiment's own runs, for apl_effect_scores.py and the model-output archive."""
    own = rt[rt.get('source', pd.Series(index=rt.index, dtype=object)).isna()] if 'source' in rt.columns else rt
    parts, vrows = [], []
    for _, r in own.iterrows():
        o = load_run(r.part, r.tag)
        if o is None:
            continue
        parts.append(pd.concat({(r.tag, 'on'): H.model_rdms(*o['on'])[H.PRIMARY_MODEL_RDM],
                                (r.tag, 'acute_off'): H.model_rdms(*o['off'])[H.PRIMARY_MODEL_RDM]},
                               names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index())
        vrows.append(dict(tag=r.tag, variant=r.tag, part=r.part, change=r.change, wiring=r.wiring, base='bouton', k=BREADTH_K,
                          fit_folder=r.fit_folder, fit_tag=r.fit_tag, kc_kc_inh=r.wKCKC_scale,
                          **{k: r[k] for k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale',
                                               'thr_factor', 'apl_factor', 'rate_on', 'rate_off')},
                          site=r.get('site'), scores=r.get('scores'), orn_input=r.get('orn_input'),
                          subdir=r.part))
    if parts:
        new = pd.concat(parts, ignore_index=True)
        _replace_rows(out / 'model_pair_distances_long.csv', new, 'tag', set(new.tag))
    if vrows:
        v = pd.DataFrame(vrows)
        _replace_rows(out / 'variants.csv', v, 'tag', set(v.tag))


def _fmt(x, spec='.4f') -> str:
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def score_part_a(out: Path) -> pd.DataFrame:
    """existing_changes.csv (A1 and A2 per change and unit: the pattern part against every target unchanged and with the
    change, the paired change on single wirings), readings and pairs rows of part A, runs.csv (part A's rows), the
    archive tables, outcome.json (conditions (a1) to (a3)) and the part-A section of comparison_table.md."""
    import json
    runs, meta, rt = _part_a_runs(out)
    readings, pairs = score_runs(runs, meta)
    readings['part'] = readings['change'].map(lambda c: 'none' if c == 'none' else c.split('_')[0])
    _replace_rows(out / 'runs.csv', rt, 'part', {'A1', 'A2'})
    _replace_rows(out / 'readings.csv', readings, 'part', {'none', 'A1', 'A2'})
    pairs['part'] = pairs['change'].map(lambda c: 'none' if c == 'none' else c.split('_')[0])
    _replace_rows(out / 'pairs.csv', pairs, 'part', {'none', 'A1', 'A2'})
    _write_archive_tables(out, rt)
    cal = readings[readings.readout == 'cal']
    ex = []
    for _, r in cal[cal.change != 'none'].iterrows():
        b = cal[(cal.change == 'none') & (cal.unit == r.unit)].iloc[0]
        row = dict(change=r.change, unit=r.unit, kind=r.kind, n_wirings=r.n_wirings)
        for t in TARGETS:
            row[f'{t}_unchanged'], row[f'{t}_changed'] = float(b[f'{t}_pattern']), float(r[f'{t}_pattern'])
            row[f'{t}_gain'] = row[f'{t}_unchanged'] - row[f'{t}_changed']
        row['z8_loss_unchanged'], row['z8_loss_changed'] = float(b.z8_loss), float(r.z8_loss)
        row['mean_effect_unchanged'], row['mean_effect_changed'] = float(b.mean_effect), float(r.mean_effect)
        raw = readings[(readings.readout == 'raw') & (readings.unit == r.unit)]
        row['raw_loss_unchanged'] = float(raw[raw.change == 'none'].z8_loss.iloc[0])
        row['raw_loss_changed'] = float(raw[raw.change == r.change].z8_loss.iloc[0])
        ex.append(row)
    ex = pd.DataFrame(ex)
    ex.to_csv(out / 'existing_changes.csv', index=False)

    def gain(ch, unit, t):
        q = ex[(ex.change == ch) & (ex.unit == unit)]
        return float(q[f'{t}_gain'].iloc[0]) if len(q) else np.nan

    draws = ALL_WIRINGS[1:]
    oc = {}
    g = gain('A1_ridge_x0.5', 'draws7', 'z8')
    n_lower = int(sum(gain('A1_ridge_x0.5', w, 'z8') > 0 for w in draws))
    gp = gain('A1_ridge_perm1_x0.5', 'draws7', 'z8')
    oc['a1'] = dict(gain_draws7_all=g, n_draws_lower=n_lower, perm_gain_draws7_all=gp,
                    holds=bool(g >= GATE_GAIN and n_lower >= 5 and gp < 0.5 * g))
    oc['a2'] = dict(gain_draws7_all=g, holds=bool(g < 0.0015))
    a3 = {X: gain(f'A2_half{X}_s{GATE_SIZE:g}', 'T', f'half{"B" if X == "A" else "A"}') for X in A2_HALVES}
    oc['a3'] = dict(gain_T_unseen=a3, holds=bool(all(v >= GATE_GAIN for v in a3.values())))
    tb = two_by_two(readings, 'A2')
    tb.to_csv(out / 'two_by_two_A2.csv', index=False)
    (out / 'outcome_A.json').write_text(json.dumps(oc, indent=1, default=float) + '\n')
    _comparison_table_a(out, ex, readings, rt, oc, tb)
    print(json.dumps(oc, indent=1, default=float), flush=True)
    return ex


A1_LABELS = {'A1_ridge_x0.5': "E75's ridge profile at 0.5", 'A1_ridge_x1': 'the ridge profile at 1',
             'A1_sparse_x0.5': 'the sparse profile at 0.5', 'A1_C2_four_x2': "C2's four glomeruli (DM3, DM2, DM5, DM6) x2",
             'A1_ridge_perm1_x0.5': 'the ridge profile at 0.5, labels permuted (seed 1)'}


def _comparison_table_a(out: Path, ex: pd.DataFrame, readings: pd.DataFrame, rt: pd.DataFrame, oc: dict,
                        tb: pd.DataFrame) -> None:
    """The part-A section of comparison_table.md (replaced in place if the file has one)."""
    cal = readings[readings.readout == 'cal'].set_index(['change', 'unit'])
    raw = readings[readings.readout == 'raw'].set_index(['change', 'unit'])
    L = ['## Part A: what exists, on wirings it was never run on', '',
         'Pattern part of the APL-effect loss under the calibrated readout (lower is better), each wiring at its own fit '
         "(C2 on the real wiring), both response rates held; a population is the mean over its wirings of the pair "
         'distances per condition. S: real, P1, P2, M1; T: P3, P4, M2, M3; the seven re-drawn: all but the real wiring.', '',
         "### A1, E75's cell-site profiles (APL->KC scale re-bisected to each wiring's APL-on count)", '',
         '| change | S, all recordings | T, all recordings | the seven re-drawn, all recordings | all eight, all recordings | '
         'T, six typical | re-drawn wirings lower | all eight, raw-count loss | all eight, mean APL effect | runs flagged |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def v(df, ch, unit, col, spec='.4f'):
        return _fmt(float(df.loc[(ch, unit), col]), spec) if (ch, unit) in df.index else '–'

    def gains(ch, t='z8'):
        q = ex[(ex.change == ch) & ex.unit.isin(ALL_WIRINGS[1:])]
        return '–' if q.empty else f"{int((q[f'{t}_gain'] > 0).sum())} of {len(q)}"

    for ch in ['none'] + [c for c in A1_LABELS if any((c, w) in cal.index for w in ALL_WIRINGS)]:
        lab = 'unchanged (each wiring at its fit)' if ch == 'none' else A1_LABELS[ch]
        nl = '–' if ch == 'none' else gains(ch)
        fl = '–' if ch == 'none' else str(int((rt[rt.change == ch].realism_flag.fillna('').astype(str).str.strip() != '').sum()))
        L.append(f"| {lab} | {v(cal, ch, 'S', 'z8_pattern')} | {v(cal, ch, 'T', 'z8_pattern')} | {v(cal, ch, 'draws7', 'z8_pattern')} | "
                 f"{v(cal, ch, 'all8', 'z8_pattern')} | {v(cal, ch, 'T', 'z6_pattern')} | {nl} | {v(raw, ch, 'all8', 'z8_loss')} | "
                 f"{v(cal, ch, 'all8', 'mean_effect', '+.3f')} | {fl} |")
    L.append(f"| KC data | floor {_fmt(H.data_target('apl_effect_rmse_floor'), '.3f')} | | | | | | | "
             f"{_fmt(float(targets()['z8'].mean()), '+.3f')} | |")
    L += ['', 'Single wirings, the change of the pattern part against all recordings (changed minus unchanged; negative is better):', '',
          '| change | ' + ' | '.join(ALL_WIRINGS) + ' |', '|---|' + '---:|' * len(ALL_WIRINGS)]
    for ch in [c for c in A1_LABELS if any((c, w) in cal.index for w in ALL_WIRINGS)]:
        cells = [_fmt(-float(ex[(ex.change == ch) & (ex.unit == w)].z8_gain.iloc[0]), '+.4f') if ((ex.change == ch) & (ex.unit == w)).any()
                 else '–' for w in ALL_WIRINGS]
        L.append(f'| {A1_LABELS[ch]} | ' + ' | '.join(cells) + ' |')
    L += ['', "### A2, E77's searched inputs (breadth recomputed, both response rates re-held)", '',
          '| input | size | S against half A | S against half B | T against half A | T against half B | T, all recordings | '
          'T, six typical | T wirings lower against the unseen half (of 4) | T, raw-count loss | T, mean APL effect | runs flagged |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    rowsA2 = [('none', 'unchanged', '–')] + [(f'A2_half{X}_s{s:g}', f'searched on half {X}', f'{s:g}') for X in A2_HALVES
                                               for s in (GATE_SIZE, 1.0)]
    for ch, lab, sz in rowsA2:
        def c(unit, col):
            return _fmt(cal.loc[(ch, unit), col]) if (ch, unit) in cal.index else '–'
        if ch == 'none':
            nl = '–'
        else:
            X = ch.split('_')[1][-1]
            q = tb[(tb.searched_on == X) & np.isclose(tb['size'].astype(float), float(sz)) & tb.unit.isin(T_WIRINGS)
                   & (tb.against == 'unseen')]
            nl = str(int((q.gain > 0).sum())) if len(q) else '–'
        fl = '–' if ch == 'none' else str(int((rt[rt.change == ch].realism_flag.fillna('').astype(str).str.strip() != '').sum()))
        rl = _fmt(raw.loc[(ch, 'T'), 'z8_loss']) if (ch, 'T') in raw.index else '–'
        me = _fmt(cal.loc[(ch, 'T'), 'mean_effect'], '+.3f') if (ch, 'T') in cal.index else '–'
        L.append(f"| {lab} | {sz} | {c('S', 'halfA_pattern')} | {c('S', 'halfB_pattern')} | {c('T', 'halfA_pattern')} | "
                 f"{c('T', 'halfB_pattern')} | {c('T', 'z8_pattern')} | {c('T', 'z6_pattern')} | {nl} | {rl} | {me} | {fl} |")
    L += ['', "(S at full size: E77's archived readings, not rerun.)", '',
          '### Outcomes of part A (the entry\'s letters)', '',
          f"- (a1) the ridge profile at 0.5 lowers the seven re-drawn wirings' population by {_fmt(oc['a1']['gain_draws7_all'])} "
          f"against all recordings (needs 0.003), on {oc['a1']['n_draws_lower']} of 7 single re-drawn wirings (needs 5), the "
          f"permutation by {_fmt(oc['a1']['perm_gain_draws7_all'])} (needs less than half): "
          f"{'holds' if oc['a1']['holds'] else 'does not hold'}.",
          f"- (a2) the gain is below 0.0015: {'holds' if oc['a2']['holds'] else 'does not hold'}.",
          f"- (a3) E77's inputs at 0.6 lower T's pattern part against the unseen half by "
          f"{_fmt(oc['a3']['gain_T_unseen'].get('A'))} (half A's input, scored on B; unchanged {_fmt(cal.loc[('none', 'T'), 'halfB_pattern'])}) "
          f"and {_fmt(oc['a3']['gain_T_unseen'].get('B'))} (half B's, scored on A; unchanged {_fmt(cal.loc[('none', 'T'), 'halfA_pattern'])}); "
          f"(i) needs 0.003 in both: {'holds' if oc['a3']['holds'] else 'does not hold'}.", '']
    path = out / 'comparison_table.md'
    text = path.read_text() if path.exists() else '# E83, the input and APL\'s glomerulus profile fitted on several wirings at once\n\n'
    sec = '\n'.join(L)
    start = text.find('## Part A:')
    if start >= 0:
        nxt = text.find('\n## ', start + 1)
        text = text[:start] + sec + ('' if nxt < 0 else text[nxt + 1:])
    else:
        text = text.rstrip('\n') + '\n\n' + sec
    path.write_text(text.rstrip('\n') + '\n')


CONTROL_SEEDS = (1, 2)


def searched_profile(out: Path, half: str) -> Dict[str, float]:
    """Arm P's searched scores on half `half` (profile_scores_<half>.csv: glomerulus, score)."""
    d = pd.read_csv(out / f'profile_scores_{half}.csv')
    return {str(g): float(s) for g, s in zip(d.glomerulus, d.score)}


def searched_factors(out: Path, half: str) -> pd.DataFrame:
    """Arm I's searched factors on half `half` (input_factors_<half>.csv), glomeruli x odors."""
    return read_factors(out / f'input_factors_{half}.csv')


def permuted_profile(scores: Mapping[str, float], seed: int) -> Dict[str, float]:
    """Part D's control of arm P: the scores permuted over their glomeruli (the 36 driven ones), one generator per
    seed, keys in sorted order."""
    keys = sorted(scores)
    perm = np.random.default_rng(seed).permutation(len(keys))
    return {keys[i]: float(scores[keys[j]]) for i, j in enumerate(perm)}


def permuted_factors(factors: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Part D's control of arm I: each odor's factors permuted over that odor's responding glomeruli (nonzero lab
    response), one generator per seed, odors in the lab input's order."""
    orn = H.abbrev_columns(H.apl16_orn_deltas())
    rng = np.random.default_rng(seed)
    out = factors.copy()
    for o in out.columns:
        resp = orn.index[orn[o].to_numpy(dtype=float) != 0]
        v = out.loc[resp, o].to_numpy()
        out.loc[resp, o] = v[rng.permutation(len(v))]
    return out


def _d_plan(arm: str, controls: bool) -> List[Tuple[str, str, float, str, Optional[int]]]:
    """[(change, half, size, wiring, control seed)] of part D for an arm."""
    plan = []
    for X in ('A', 'B'):
        for size, ws in ((GATE_SIZE, ALL_WIRINGS), (1.0, T_WIRINGS)):
            plan += [(f'D{arm}_half{X}_s{size:g}', X, size, w, None) for w in ws]
        if controls:
            for seed in CONTROL_SEEDS:
                plan += [(f'D{arm}perm{seed}_half{X}_s{GATE_SIZE:g}', X, GATE_SIZE, w, seed) for w in T_WIRINGS]
    return plan


def run_part_d(out: Path, arm: str, controls: bool = False) -> pd.DataFrame:
    """Part D's simulations for an arm (resumable; runs_D<arm>.csv, outputs under outputs/.../D<arm>/): arm P, the
    searched scores at the size on each wiring's WiringModel, the APL scale re-bisected; arm I, the input at the size,
    breadth recomputed, both rates re-held. With `controls`, also the label-permuted controls on T at GATE_SIZE."""
    import gc
    import hashlib
    import json
    part = f'D{arm}'
    rp, pp = out / f'runs_{part}.csv', out / f'provenance_{part}.json'
    done = set(_read_rows(rp).get('tag', pd.Series(dtype=str)))
    plan = [x for x in _d_plan(arm, controls) if f'{x[0]}_{x[3]}' not in done]
    by_wiring: Dict[str, list] = {}
    for x in plan:
        by_wiring.setdefault(x[3], []).append(x)
    for w in [w for w in ALL_WIRINGS if w in by_wiring]:
        model = WiringModel(w, check=True) if arm == 'P' else None
        if arm == 'I':
            sim, info = build_simulator(w)
            _check_saved(w, sim, info['params'], saved_fit(w)['saved'])
            del sim, info
        for ch, X, size, _, seed in by_wiring[w]:
            tag = f'{ch}_{w}'
            if arm == 'P':
                sc = searched_profile(out, X)
                if seed is not None:
                    sc = permuted_profile(sc, seed)
                sc = scale_scores(sc, size)
                res = profile_reading(model, sc, 'cell', label=tag)
                o = {'on': res['on'], 'off': res['off']}
                extra = dict(site='cell', scores=json.dumps(sc),
                             scores_sha256=hashlib.sha256(json.dumps(sc, sort_keys=True).encode()).hexdigest())
                prov = H.model_provenance(orn_deltas=model.info['orn'], _wPNKC=model.W, _wAPLKC=res['wAPLKC'])
            else:
                F = searched_factors(out, X)
                if seed is not None:
                    F = permuted_factors(F, seed)
                orn_in = input_at(F, size)
                path = _write_input(out, ch, orn_in)
                res = input_reading(w, orn_in, input_tag=input_tag_for(orn_in, f'E83_{ch}'))
                o = {'on': res['on'], 'off': res['off']}
                extra = dict(input_tag=res['input_tag'], orn_input=str(path.relative_to(H.shared_dir())))
                prov = H.model_provenance(orn_deltas=orn_in)
            save_run(part, tag, o)
            _append_row(rp, _run_row(part, ch, w, tag, res, o, half=X, size=size, control_seed=seed, **extra))
            _provenance_add(pp, part, tag, prov)
            Y = 'B' if X == 'A' else 'A'
            d0, d1 = pair_distances(saved_fit(w)['saved'], 'cal'), pair_distances(o, 'cal')
            print(f"{tag}: thr x{res['thr_factor']:.4f}, APL x{res['apl_factor']:.4f}; pattern under the readout against the "
                  f"unseen half {Y} {_rms(pattern_error(d0, half_target(Y))):.4f} -> {_rms(pattern_error(d1, half_target(Y))):.4f}",
                  flush=True)
            del res, o
            gc.collect()
        del model
        gc.collect()
    return _read_rows(rp)


def score_part_d(out: Path, arm: str) -> pd.DataFrame:
    """Part D's tables for an arm: readings and pairs rows, two_by_two_D<arm>.csv, gate_D<arm>.csv (with the controls'
    two-by-two when run), agreement_D<arm>.csv, and the arm's section of comparison_table.md."""
    part = f'D{arm}'
    rt = _read_rows(out / f'runs_{part}.csv')
    runs, meta = {}, {}
    for _, r in rt.iterrows():
        o = load_run(part, r.tag)
        if o is not None:
            runs[(r.change, r.wiring)] = o
            meta[r.change] = dict(part=part, size=float(r['size']))
    readings, pairs = score_runs(runs, meta)
    readings['part'] = readings['change'].map(lambda c: 'none' if c == 'none' else part)
    pairs['part'] = pairs['change'].map(lambda c: 'none' if c == 'none' else part)
    _replace_rows(out / 'readings.csv', readings[readings.part == part], 'part', {part})
    _replace_rows(out / 'pairs.csv', pairs[pairs.part == part], 'part', {part})
    _replace_rows(out / 'runs.csv', rt, 'part', {part})
    _write_archive_tables(out, rt)
    main_r = readings[~readings.change.str.contains('perm')]
    tb = two_by_two(main_r, part)
    tb.to_csv(out / f'two_by_two_{part}.csv', index=False)
    ctl = None
    if readings.change.str.contains('perm').any():
        parts_ = []
        for seed in CONTROL_SEEDS:
            r2 = readings[readings.change.str.startswith(f'{part}perm{seed}_') | (readings.change == 'none')].copy()
            r2['change'] = r2['change'].str.replace(f'{part}perm{seed}_', f'{part}_', regex=False)
            parts_.append(two_by_two(r2, part).assign(control_seed=seed))
        ctl = pd.concat(parts_, ignore_index=True)
    g = gate(main_r, part, runs=rt[rt.control_seed.isna()] if 'control_seed' in rt.columns else rt, controls=ctl)
    g.to_csv(out / f'gate_{part}.csv', index=False)
    ag = agreement(out, arm)
    _comparison_table_d(out, arm, tb, g, ag)
    print(g.to_string(), flush=True)
    print(ag.to_string(), flush=True)
    return g


def _section(out: Path, heading: str, lines: List[str]) -> None:
    """Replace (or append) the comparison_table.md section starting with `heading` ('## ...')."""
    path = out / 'comparison_table.md'
    text = path.read_text() if path.exists() else '# E83, the input and APL\'s glomerulus profile fitted on several wirings at once\n\n'
    sec = '\n'.join([heading] + lines)
    start = text.find(heading)
    if start >= 0:
        nxt = text.find('\n## ', start + 1)
        text = text[:start] + sec + ('\n' if nxt >= 0 else '') + ('' if nxt < 0 else text[nxt + 1:])
    else:
        text = text.rstrip('\n') + '\n\n' + sec
    path.write_text(text.rstrip('\n') + '\n')


ARM_NAMES = {'P': "arm P, APL's weight by glomerulus (cell site)", 'I': "arm I, each odor's input"}


def _comparison_table_d(out: Path, arm: str, tb: pd.DataFrame, g: pd.DataFrame, ag: pd.DataFrame) -> None:
    """Part D's section of comparison_table.md for an arm: the two-by-two (S and T against the half searched on and the
    half never seen), T's single wirings as paired differences, the gate's conditions and the agreement."""
    part = f'D{arm}'
    L = ['', 'Pattern part of the APL-effect loss under the calibrated readout (lower is better), each wiring at its own fit, '
         'both response rates held; unchanged / with the change; S: real, P1, P2, M1; T: P3, P4, M2, M3.', '',
         '| searched on | size | population | against the half searched on | against the half never seen | against all recordings | against the six typical |',
         '|---|---:|---|---:|---:|---:|---:|']
    for X in ('A', 'B'):
        for size in (GATE_SIZE, 1.0):
            for unit in ('S', 'T'):
                q = tb[(tb.searched_on == X) & np.isclose(tb['size'], size) & (tb.unit == unit)].set_index('against')
                if q.empty:
                    continue
                cells = [f"{_fmt(q.loc[a, 'unchanged'])} / {_fmt(q.loc[a, 'changed'])}" for a in ('searched', 'unseen', 'all', 'six')]
                L.append(f'| half {X} | {size:g} | {unit} | ' + ' | '.join(cells) + ' |')
    L += ['', "T's single wirings against the half never seen, changed minus unchanged at the gate's size (negative is better):", '',
          '| searched on | ' + ' | '.join(T_WIRINGS) + ' |', '|---|' + '---:|' * len(T_WIRINGS)]
    for X in ('A', 'B'):
        q = tb[(tb.searched_on == X) & np.isclose(tb['size'], GATE_SIZE) & (tb.against == 'unseen')].set_index('unit')
        if len(q):
            L.append(f'| half {X} | ' + ' | '.join(_fmt(-float(q.loc[w, 'gain']), '+.4f') if w in q.index else '–'
                                                for w in T_WIRINGS) + ' |')
    if len(g):
        L += ['', 'The gate at the size 0.6:', '']
        for _, r in g.iterrows():
            if r.searched_on == 'both':
                conds = ', '.join(f'({k}) {"holds" if bool(r[k]) else "fails"}' for k in ('i', 'ii', 'iii', 'iv', 'v')
                                  if k in r.index and pd.notna(r[k]))
                L.append(f"- both directions: {conds}; T's singles lower in {int(r.n_singles_lower)} of {int(r.n_singles)}; "
                         f"squared pattern error against all recordings {_fmt(100 * r.sq_reduction_all, '.1f')} % lower "
                         f"(needs 15), against the six typical {_fmt(100 * r.sq_reduction_six, '.1f')} % (needs 7.5); "
                         f"borderline: {'yes' if r.borderline else 'no'}.")
            else:
                L.append(f"- searched on half {r.searched_on}: T against the unseen half lower by {_fmt(r.gain_unseen)} (needs 0.003); "
                         f"raw-count loss {_fmt(r.raw_loss_change, '+.4f')}; mean APL effect under the readout "
                         f"{_fmt(r.mean_effect_cal, '+.3f')} (the flies' {_fmt(r.flies_mean, '+.3f')}); runs flagged: "
                         f"{'–' if pd.isna(r.n_flagged) else int(r.n_flagged)}.")
    if len(ag):
        L += ['', 'Agreement between the halves:', '']
        for _, r in ag.iterrows():
            L.append(f"- {r.what}{(' (' + r.odor + ')') if isinstance(r.odor, str) and r.odor else ''}: {_fmt(r.value, '.2f')}")
    _section(out, f'## Part D, {ARM_NAMES[arm]}', L + [''])


def agreement(out: Path, arm: str) -> pd.DataFrame:
    """Agreement between the halves (agreement_<arm>.csv; arm 'P', 'I', or 'A2' for E77's inputs): the correlation of
    the two halves' scores over the glomeruli (arm P) or, per odor, of the log factors over all glomeruli (as E77's
    report_tables) and over the odor's responding ones, with the share of the factors' variance over the responding
    entries that a per-glomerulus and a per-odor term explain (arm I, A2); and, where both halves' readings at
    GATE_SIZE exist, the correlation of their per-pair changes of T's APL effect under the readout."""
    rows = []
    if arm == 'P':
        a, b = searched_profile(out, 'A'), searched_profile(out, 'B')
        g = sorted(set(a) | set(b))
        va, vb = np.array([a.get(x, 0.0) for x in g]), np.array([b.get(x, 0.0) for x in g])
        rows.append(dict(arm=arm, what='scores over glomeruli', odor='', value=float(np.corrcoef(va, vb)[0, 1]), n=len(g)))
    else:
        src = (H.shared_dir() / 'results' / 'E77_input_side_common_error' / 'oracle_factors_{}.csv') if arm == 'A2' else \
            (out / 'input_factors_{}.csv')
        F = {X: np.log2(read_factors(Path(str(src).format(X)))) for X in ('A', 'B')}
        orn = H.abbrev_columns(H.apl16_orn_deltas())
        searched = [o for o in F['A'].columns if (F['A'][o] != 0).any() or (F['B'][o] != 0).any()]
        for o in searched:
            resp = orn[o].to_numpy(dtype=float) != 0
            rows.append(dict(arm=arm, what='log factors over all glomeruli', odor=o,
                             value=float(np.corrcoef(F['A'][o], F['B'][o])[0, 1]), n=len(F['A'])))
            rows.append(dict(arm=arm, what='log factors over responding glomeruli', odor=o,
                             value=float(np.corrcoef(F['A'][o][resp], F['B'][o][resp])[0, 1]), n=int(resp.sum())))
        for X in ('A', 'B'):
            L = F[X][searched]
            mask = (orn[searched].to_numpy(dtype=float) != 0)
            y = L.to_numpy()[mask]
            gi = np.nonzero(mask)[0]
            oi = np.nonzero(mask)[1]
            tot = float(((y - y.mean()) ** 2).sum())
            gm = pd.Series(y).groupby(gi).transform('mean').to_numpy()
            om = pd.Series(y).groupby(oi).transform('mean').to_numpy()
            rows.append(dict(arm=arm, what=f'half {X}: variance share of a per-glomerulus term', odor='',
                             value=1.0 - float(((y - gm) ** 2).sum()) / tot, n=int(mask.sum())))
            rows.append(dict(arm=arm, what=f'half {X}: variance share of a per-odor term', odor='',
                             value=1.0 - float(((y - om) ** 2).sum()) / tot, n=int(mask.sum())))
            # both terms (additive two-way fit, alternating means)
            fit = np.zeros_like(y)
            g_eff, o_eff = np.zeros(gi.max() + 1), np.zeros(oi.max() + 1)
            mu = y.mean()
            for _ in range(200):
                g_eff = pd.Series(y - mu - o_eff[oi]).groupby(gi).mean().reindex(range(gi.max() + 1)).fillna(0).to_numpy()
                o_eff = pd.Series(y - mu - g_eff[gi]).groupby(oi).mean().reindex(range(oi.max() + 1)).fillna(0).to_numpy()
            fit = mu + g_eff[gi] + o_eff[oi]
            rows.append(dict(arm=arm, what=f'half {X}: variance share of per-glomerulus plus per-odor terms', odor='',
                             value=1.0 - float(((y - fit) ** 2).sum()) / tot, n=int(mask.sum())))
    pp = out / 'pairs.csv'
    if pp.exists():
        pr = pd.read_csv(pp)
        pr = pr[(pr.unit == 'T') & (pr.readout == 'cal')]
        part = 'A2' if arm == 'A2' else f'D{arm}'
        base = pr[pr.change == 'none'].set_index(['odor_a', 'odor_b'])['effect']
        ch = {X: pr[pr.change == f'{part}_half{X}_s{GATE_SIZE:g}'].set_index(['odor_a', 'odor_b'])['effect'] for X in ('A', 'B')}
        if all(len(v) for v in ch.values()):
            da, db = (ch[X] - base.reindex(ch[X].index) for X in ('A', 'B'))
            rows.append(dict(arm=arm, what="T's per-pair change of the APL effect, half A's change against half B's", odor='',
                             value=float(np.corrcoef(da.to_numpy(), db.reindex(da.index).to_numpy())[0, 1]), n=len(da)))
    res = pd.DataFrame(rows)
    res.to_csv(out / f'agreement_{arm}.csv', index=False)
    return res


def part_d(out: Path, arm: str, controls: Optional[bool] = None) -> pd.DataFrame:
    """Part D for one arm ('P' or 'I'): each half's searched change at GATE_SIZE on all eight wirings and at full size on
    T; the two-by-two tables, the gate, the agreement. The label-permuted controls run only when (i) to (iv) hold (or
    when `controls` is True), then the gate is read again with them."""
    run_part_d(out, arm, controls=bool(controls))
    g = score_part_d(out, arm)
    both = g[g.searched_on == 'both']
    if controls is None and len(both) and bool(both.passes_i_to_iv.iloc[0]):
        print(f'arm {arm}: (i) to (iv) hold; the controls run', flush=True)
        run_part_d(out, arm, controls=True)
        g = score_part_d(out, arm)
    return g


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
