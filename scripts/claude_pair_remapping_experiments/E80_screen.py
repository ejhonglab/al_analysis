#!/usr/bin/env python3
"""E80 parts B and C: the screen of the remaining candidates for the common error, on C2 (the breadth-2 APL-effect fit,
E67, rate0.068_aplboth_k2_cap3) at fixed scales on the real wiring, on E77's runner (E77_input_screen: c2_parts,
build_c2_simulator, hold_rates, handle_breadth, E69's bisect_factor and E77's Scorer for the readings). The design is
the E80 entry in experiment_priorities.md, "(B) The screen" and "(C) The gate and what follows".

- (A) the pulse is in E80_pulse.py; (D) in E80_remaining_candidates.py.
- (B) the screen. Every family starts with an identity point that must reproduce its base's saved APL-on and APL-off
  spike counts exactly (the screen stops otherwise). The arms without a probe run first.
  - B1, the uniform form: with APL on, every KC's threshold (fixed_thr + 2 x its spontaneous input, through the
    simulator's `kc_thr_factor`, 1 / c for all KCs) is c times lower than with APL off (c 1.1, 1.25, 1.5, 2), and the
    APL->KC scale (KC->APL and bouton->APL with it, E77's TIED_ON) is re-bisected to the base's APL-on count; the
    APL-off run is the base's own, KC-KC and APL->bouton scales at their absolute values. On C2 (handle B1) and on the
    firing-ceiling fit (E76's fit 0, handle B1_fit0, rebuilt as E43.fit_model / E69.Model rebuild a fit, with its
    refractory period and adaptation). Not gated: reported as a curve against c (b1_curve.csv). (On C2 the
    spontaneous part of the threshold is the larger one: fixed_thr 102, 2 x spontaneous input median 151, range 17 to
    377, so 1 / c on the whole threshold is a larger step than 1 / c on fixed_thr alone.)
  - B2, a shared lateral input: every one of the 36 glomeruli's ORN delta plus eps x the odor's mean positive delta
    (the mean over the 36 glomeruli of max(delta, 0), so the odor's summed drive grows by eps x its summed positive
    drive; eps 0.05, 0.1, 0.2, 0.4). Static: both rates re-held as E77's static handles (hold_rates: the threshold to
    C2's 4067 APL-off responses, the breadth recomputed for the input by handle_breadth, the APL scale to 1852). Control
    (B2_scaled): at the best eps, the odor's own positive deltas scaled so that its summed drive grows by the same amount.
  - B3, each odor's recruitment: one-odor runs (olfsysm's `mp.sim_only`, as E77's PerOdorRunner) spread over worker
    processes (`OdorPool`). With APL off, a per-odor threshold factor (every KC's threshold, `kc_thr_factor`) is bisected
    until each odor's response fraction is the flies' with APL output blocked (TNT); then, at those thresholds, a
    per-odor APL scale (TIED_ON) is bisected until each odor's APL-on fraction is the control flies'. Targets in
    baseline-noise units (the registry's noise_<ctrl|tnt>_frac_<odor>, the recording means of
    data/yang_apl16_kc_noise_units/response_fraction_per_recording.csv) and, beside them, in Yang's z. Three readings
    per unit: B3_thr_<u> (the per-odor thresholds in both conditions, C2's APL scales; like E66's 'scales_off' arm),
    B3_throff_<u> (the thresholds with APL off only, the APL-on run C2's own; no extra simulation) and B3_thrapl_<u>
    (thresholds and per-odor APL scales; like E66's 'both' arm). No control (an oracle).
  - B4, changes that act only with APL off: the APL-on run is C2's own; with APL off a per-KC threshold factor
    (`kc_thr_factor`) and the common threshold (fixed_thr) re-bisected to C2's 4067 APL-off responses. B4_beta: each
    KC's factor 1 + beta x its summed APL-off spike count in C2 / the mean over KCs (beta 0.25, 0.5, 1, 2); control
    B4_beta_perm, the best beta's factors permuted across KCs within each KC type (one seed). B4_type: one type's
    threshold x 1.25 and x 1.5; control B4_type_opp, the best type point's opposite direction (its type x 1 / factor).
  - B5, KC types: the APL->KC weights of one type x 0.5 and x 2 (in place in the simulator's unscaled weights, as
    E75 does); the APL-off run is C2's own, the APL scale bisected to 1852. Each point's control is its opposite
    direction.
  - Per point: E77's Scorer (the loss and its parts on raw counts, under the calibrated readout with K re-calibrated
    and binarized; the per-odor and pair-specific parts; the kinds of ester pairs and the alcohol group; the projection
    on the common error and the transfer reduction against all eight recordings and the six typical ones; the per-odor
    response fractions; the four bands under both rules; the 4-7 class), plus the APL effect against the effect in
    baseline-noise units (E76's effect_by_unit_pairs.csv, ctrl__noise - tnt__noise): on raw counts, and under the
    calibrated readout with K set to the noise-unit calibration target (`noise_scores`; `cal_noise_K_at_bound` where
    the readout cannot reach it, as for the firing-ceiling fit, whose K then sits at 1000, close to raw counts).
  - B3's worker processes hold one C2 simulator each (about 1.5 GB) and are closed once B3's points are done.
- (C) `gate`: E77's gate (E77_input_side_common_error.gate's conditions and constants) on every handle but B1, with
  E80's controls (B2's scaled profile, B4's permutation and opposite direction, B5's opposite direction); a handle's
  best point is its lowest pattern part under the readout among its non-identity, non-control points; every handle
  whose best point passes goes on (the entry has no limit on how many). Writes gate.csv and b1_curve.csv.

Outputs, results/E80_remaining_candidates/: screen.csv (one row per point; resumable), screen_pairs.csv (per point and
pair: distances APL on / off and the APL effect on raw counts, under the readout, binarized and under the noise-unit
readout), screen_params.json (each point's absolute parameters, factors and bisection histories), screen_errors.csv,
screen_provenance.json, gate.csv, b1_curve.csv. Spike counts and responses of every point under
outputs/E80_remaining_candidates/screen/ (tag <base>_<point_id>, base C2 or fit0; conditions on / acute_off), with each
point's per-KC threshold factors (<tag>_kc_thr_factor.parquet) where it has them.

Usage (from this folder):
    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 nice -n 10 ../../venv/bin/python -u E80_screen.py [--only ID ...] [--list]
        [--workers N] [--gate]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
import E77_input_screen as S77

EXPERIMENT = 'E80_remaining_candidates'
C2 = S77.C2
# E76's fit 0: the firing-ceiling family (E69's fit 2) re-fitted with the rate terms in baseline-noise units
FIT0 = ('E76_data_side_target', 'nu_rate0.068_aplboth_refr3_adapt0.1114t1000_k2')
BASES = ('C2', 'fit0')
TIED_ON = S77.TIED_ON
THR_BRACKET, THR_STEP = S77.THR_BRACKET, S77.THR_STEP
APL_BRACKET, APL_STEP = S77.APL_BRACKET, S77.APL_STEP
# B1's APL-on threshold is lowered by up to 2x, so its APL scale needs more room than E77's bracket
B1_APL_BRACKET = (0.1, 50.0)
# B3's per-odor factors: each odor moves from C2's fraction to the flies' (up to ~2.5x apart for some odors)
B3_THR_BRACKET, B3_THR_STEP = (0.05, 20.0), 1.1
B3_APL_BRACKET = (0.02, 50.0)

B1_C = (1.1, 1.25, 1.5, 2.0)
B2_EPS = (0.05, 0.1, 0.2, 0.4)
B3_UNITS = ('noise', 'z')
B3_ARMS = ('thr', 'throff', 'thrapl')
B4_BETAS = (0.25, 0.5, 1.0, 2.0)
B4_TYPE_FACTORS = (1.25, 1.5)
B5_FACTORS = (0.5, 2.0)
# the KC types as in the outputs' kc_type level, with file-safe labels
KC_TYPES = {'ab': 'ab', 'g': 'g', "a'b'": 'apbp'}
PERM_SEED = 1
E76_PAIRS = 'results/E76_data_side_target/effect_by_unit_pairs.csv'
# a uniform factor above this is not defensible without an acute-block measurement (the entry's realism section)
B1_DEFENSIBLE_C = 1.25
# outcome (b): the firing-ceiling fit's pattern part (raw counts against the effect in baseline-noise units) may be at
# most this much above its own at c <= B1_DEFENSIBLE_C
B1_PATTERN_SLACK = 0.006


# ---------------------------------------------------------------------------
# transforms and factors (no simulation)

def shared_input(orn: pd.DataFrame, eps: float) -> pd.DataFrame:
    """B2: every glomerulus's ORN delta plus `eps` x the odor's mean positive delta (the mean over all glomeruli of
    max(delta, 0)). eps 0 gives the input back bit for bit."""
    out = orn.astype(float).copy()
    if float(eps) == 0.0:
        return out
    x = out.to_numpy()
    m = np.where(x > 0, x, 0.0).mean(axis=0)
    out[:] = x + float(eps) * m[None, :]
    return out


def shared_input_drive(orn: pd.DataFrame, eps: float) -> np.ndarray:
    """The summed drive B2 adds to each odor (over glomeruli): n_glomeruli x eps x the odor's mean positive delta, which
    is eps x the odor's summed positive delta."""
    x = orn.to_numpy(dtype=float)
    return x.shape[0] * float(eps) * np.where(x > 0, x, 0.0).mean(axis=0)


def scaled_profile(orn: pd.DataFrame, added: np.ndarray) -> pd.DataFrame:
    """B2's control: each odor's positive deltas times 1 + added / (its summed positive delta), so that its summed delta
    over glomeruli grows by `added` (one value per odor) through its own profile; zero and negative deltas unchanged.
    Zero added gives the input back bit for bit."""
    out = orn.astype(float).copy()
    x = out.to_numpy()
    added = np.asarray(added, dtype=float)
    if added.shape != (x.shape[1],):
        raise ValueError(f'added has shape {added.shape}, expected ({x.shape[1]},)')
    if not added.any():
        return out
    pos = x > 0
    sp = np.where(pos, x, 0.0).sum(axis=0)
    if ((sp <= 0) & (added != 0)).any():
        raise ValueError('an odor without positive deltas cannot be scaled to a larger summed drive')
    s = 1.0 + np.divide(added, sp, out=np.zeros_like(added), where=sp > 0)
    out[:] = np.where(pos, x * s[None, :], x)
    return out


def activity_factors(activity: pd.Series, beta: float) -> pd.Series:
    """B4: each KC's threshold factor 1 + beta x activity / its mean over KCs (`activity` each KC's summed APL-off
    spike count in C2). beta 0 gives exactly 1."""
    a = activity.astype(float)
    if float(beta) == 0.0:
        return pd.Series(1.0, index=a.index)
    m = float(a.mean())
    if not m > 0 or (a < 0).any():
        raise ValueError('activity must be non-negative with a positive mean')
    return 1.0 + float(beta) * a / m


def permute_within_type(factors: pd.Series, seed: int = PERM_SEED, level: str = H.KC_TYPE_LEVEL) -> pd.Series:
    """`factors` (indexed by KC, with a KC-type level) permuted among the KCs of each type (one generator, types in
    sorted order), so each type keeps its multiset of factors."""
    rng = np.random.default_rng(seed)
    types = factors.index.get_level_values(level)
    out = factors.to_numpy(dtype=float).copy()
    for t in sorted(pd.unique(types)):
        idx = np.flatnonzero(types == t)
        out[idx] = out[idx[rng.permutation(len(idx))]]
    return pd.Series(out, index=factors.index)


def type_factors(index: pd.Index, kc_type: str, factor: float, level: str = H.KC_TYPE_LEVEL) -> pd.Series:
    """`factor` for the KCs of `kc_type`, 1 elsewhere (a factor of 1 gives exactly 1 everywhere)."""
    types = index.get_level_values(level)
    if kc_type not in set(types):
        raise ValueError(f'no KCs of type {kc_type!r} (types: {sorted(set(types))})')
    return pd.Series(np.where(types == kc_type, float(factor), 1.0), index=index)


def _arr_sha256(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=float), dtype='<f8').tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# the bases

class Base:
    """One base at its fitted parameters as a fixed-scale simulator, rebuilt and checked against its saved outputs:
    'C2' (E77's build_c2_simulator with C2's own breadth) or 'fit0' (E76's fit 0, rebuilt as E43.fit_model does for
    E69.Model, with its refractory period and adaptation). Attributes: sim (simulate(params, kc_thr_factor=...)), info
    (build_c2_simulator's, C2 only), p (APL-on parameters), saved ({'on', 'off'}: (responses, spike counts)),
    target_on / target_off (response counts), size, n_kc, kws (scalar kwargs), label."""

    def __init__(self, name: str):
        import E72_wiring_population as E72
        if name not in BASES:
            raise ValueError(name)
        self.name = name
        t0 = time.time()
        if name == 'C2':
            P = S77.c2_parts()
            self.sim, self.info = S77.build_c2_simulator(breadth=P['breadth'])
            self.p, self.saved, self.kws = dict(P['p']), P['saved'], self.info['kws']
            self.experiment, self.tag = C2
            self.label = 'C2, the breadth-2 APL-effect fit (E67)'
        else:
            import E43_model_trial_noise as E43
            from al_analysis.mb_model import make_fixed_param_kc_simulator
            self.experiment, self.tag = FIT0
            name43 = E43.model_from(f'{FIT0[0]}:{FIT0[1]}')
            d = H.outputs_dir(EXPERIMENT) / 'bases'
            d.mkdir(exist_ok=True)
            fm = E43.fit_model(name43, H.apl16_orn_deltas(), d)
            if fm.get('kc_thr_factor') is not None:
                raise NotImplementedError(f'{FIT0}: a fit with an input-scaled threshold')
            self.sim, _ = make_fixed_param_kc_simulator(fm['orn'], initial_params=fm['p_on'], **fm['kws'])
            self.info = dict(fit_info=fm['info'])
            self.p = dict(fm['p_on'])
            self.kws = {k: v for k, v in sorted(fm['kws'].items()) if not k.startswith('_')}
            self.saved = E72.load_fit(*FIT0)
            if self.saved is None:
                raise RuntimeError(f"fit 0's saved outputs ({FIT0}) not found")
            self.label = "the firing-ceiling fit (E76's fit 0)"
            # (the simulator's odors in the saved outputs' column order, if fit_mb_model returned them in another)
            cols = self.saved['on'][1].columns
            r0, _ = self.sim(dict(self.p))
            if not r0.columns.equals(cols):
                if set(r0.columns) != set(cols):
                    raise RuntimeError(f'{FIT0}: the simulator has odors {list(r0.columns)}, the saved outputs {list(cols)}')
                raw_sim = self.sim

                def ordered(params, *a, **kw):
                    r, s = raw_sim(params, *a, **kw)
                    return r[cols], s[cols]
                self.sim = ordered
        self.build_seconds = time.time() - t0
        self.target_on, self.target_off = S77._n(self.saved['on'][0]), S77._n(self.saved['off'][0])
        self.size = int(self.saved['on'][0].size)
        self.n_kc = int(self.saved['on'][0].shape[0])
        # the base rebuilt must reproduce its saved counts exactly (both conditions, same index and columns)
        for c, q in (('on', self.p), ('off', self.off_params())):
            r, s = self.sim(dict(q))
            n = int((s.to_numpy(dtype=float) != self.saved[c][1].to_numpy(dtype=float)).sum())
            if n or not s.index.equals(self.saved[c][1].index) or not s.columns.equals(self.saved[c][1].columns):
                raise RuntimeError(f'{name} rebuilt differs from its saved {c} run in {n} counts (or index / columns)')
        print(f"{name} ({self.experiment}:{self.tag}) rebuilt in {self.build_seconds:.0f} s, reproduces its saved APL-on and "
              f"APL-off spike counts exactly ({self.target_off} / {self.target_on} responses off / on)", flush=True)
        self._scorer = None

    def off_params(self, p: Optional[dict] = None) -> dict:
        return dict(p or self.p, wAPLKC_scale=0.0, wAPLPN_scale=0.0)

    def scorer(self) -> 'S77.Scorer':
        if self._scorer is None:
            self._scorer = S77.Scorer(self.saved)
        return self._scorer

    def type_index(self) -> pd.Index:
        """The KCs (kc_id, kc_type) in output row order."""
        return self.saved['off'][1].index


# ---------------------------------------------------------------------------
# the runners (each returns ({'on', 'off'}: (responses, spike counts)), res) with res as hold_rates')

def _tied(p: dict, x: float) -> dict:
    return dict(p, **{k: p[k] * x for k in TIED_ON})


def simulate_b1(base: Base, c: float, label: str = '') -> Tuple[Dict[str, tuple], dict]:
    """B1: the APL-on run with every KC's threshold x 1 / c and the APL scale (TIED_ON) bisected to the base's APL-on
    count; the APL-off run the base's own."""
    import E69_kc_firing_ceiling as E69
    t0 = time.time()
    f = np.full(base.n_kc, 1.0 / float(c))
    x, (r, s), hist, at_b = E69.bisect_factor(lambda v: base.sim(_tied(base.p, v), kc_thr_factor=f), base.target_on,
                                              base.size, B1_APL_BRACKET, APL_STEP, f'{label} APL scale')
    res = dict(params=_tied(base.p, x), params_off=base.off_params(), thr_factor=1.0, apl_factor=float(x),
               at_bound_apl=bool(at_b), n_sims=len(hist), history=hist, kc_thr_factor_on=1.0 / float(c),
               seconds=time.time() - t0)
    return {'on': (r, s), 'off': base.saved['off']}, res


def simulate_off_thr(base: Base, factors: pd.Series, label: str = '') -> Tuple[Dict[str, tuple], dict]:
    """B4: the APL-off run with the per-KC threshold factors `factors` (output row order) and the common threshold
    (fixed_thr) bisected to the base's APL-off count; the APL-on run the base's own."""
    import E69_kc_firing_ceiling as E69
    t0 = time.time()
    if not factors.index.equals(base.type_index()):
        raise ValueError("the factors are not in the base's KC order")
    f = factors.to_numpy(dtype=float)
    q = base.off_params()
    g, (r, s), hist, at_b = E69.bisect_factor(lambda v: base.sim(dict(q, fixed_thr=q['fixed_thr'] * v), kc_thr_factor=f),
                                              base.target_off, base.size, THR_BRACKET, THR_STEP, f'{label} threshold (APL off)')
    res = dict(params=dict(base.p), params_off=dict(q, fixed_thr=q['fixed_thr'] * g), thr_factor=float(g), apl_factor=1.0,
               at_bound_thr=bool(at_b), n_sims=len(hist), history=hist, seconds=time.time() - t0)
    return {'on': base.saved['on'], 'off': (r, s)}, res


def aplkc_weight_factors(base: Base, kc_type: str, factor: float) -> np.ndarray:
    """B5: the factor for each entry of the simulator's unscaled APL->KC weights (per claw or per KC, as the base's
    `wAPLKC` Series is indexed), `factor` for the entries of KCs of `kc_type`, 1 elsewhere."""
    w = base.info['wAPLKC']
    tmap = dict(zip(base.type_index().get_level_values(H.KC_ID_LEVEL), base.type_index().get_level_values(H.KC_TYPE_LEVEL)))
    ids = w.index.get_level_values(H.KC_ID_LEVEL)
    types = np.array([tmap.get(i) for i in ids], dtype=object)
    if pd.isna(types).any():
        raise ValueError(f'{int(pd.isna(types).sum())} APL->KC weights of KCs not in the outputs')
    if kc_type not in set(types):
        raise ValueError(f'no APL->KC weights of type {kc_type!r}')
    return np.where(types == kc_type, float(factor), 1.0)


def simulate_b5(base: Base, kc_type: str, factor: float, label: str = '') -> Tuple[Dict[str, tuple], dict]:
    """B5: the APL->KC weights of `kc_type`'s KCs x `factor` (in place in the simulator's unscaled weights, restored
    after), the APL scale (TIED_ON) bisected to the base's APL-on count; the APL-off run the base's own."""
    import E69_kc_firing_ceiling as E69
    t0 = time.time()
    cell = base.info['cells']['wAPLKC_unscaled'].cell_contents
    w0 = cell.copy()
    f = aplkc_weight_factors(base, kc_type, factor)
    try:
        cell[...] = w0 * f.reshape(cell.shape)
        x, (r, s), hist, at_b = E69.bisect_factor(lambda v: base.sim(_tied(base.p, v)), base.target_on, base.size,
                                                  APL_BRACKET, APL_STEP, f'{label} APL scale')
    finally:
        cell[...] = w0
    res = dict(params=_tied(base.p, x), params_off=base.off_params(), thr_factor=1.0, apl_factor=float(x),
               at_bound_apl=bool(at_b), n_sims=len(hist), history=hist, seconds=time.time() - t0,
               weight_factor_sha256=_arr_sha256(f), n_weights_scaled=int((f != 1.0).sum()))
    return {'on': (r, s), 'off': base.saved['off']}, res


def simulate_static_orn(orn: pd.DataFrame, label: str = '') -> Tuple[Dict[str, tuple], dict]:
    """B2: C2's family with another ORN input, both rates re-held (E77's hold_rates; the breadth from handle_breadth
    for the input)."""
    res = S77.hold_rates(lambda: S77.build_c2_simulator(orn=orn), label=label)
    return {'on': res['on'], 'off': res['off']}, res


# ---------------------------------------------------------------------------
# B3: one-odor runs, spread over worker processes

def _odor_task(sim: Callable, mp, order: np.ndarray, names: Sequence[str], p: dict, n_kc: int, task: dict) -> List[dict]:
    """One worker's share of a B3 task: for each of its odors (local index jj, output column jj, simulated alone with
    `mp.sim_only`), the APL-off run at a threshold factor (given, `thr`, or bisected to `target_off` responses), the
    APL-on run at that factor and C2's APL scales, and the APL-on run at a per-odor APL scale (given, `apl`, or bisected
    to `target_on` responses at that threshold factor). Every threshold factor goes through `kc_thr_factor` (all KCs)."""
    import E69_kc_firing_ceiling as E69
    p_off = dict(p, wAPLKC_scale=0.0, wAPLPN_scale=0.0)
    out = []
    for jj in range(len(order)):
        def one(params, t, jj=jj):
            mp.sim_only = [int(order[jj])]
            try:
                r, s = sim(params, kc_thr_factor=np.full(n_kc, float(t)))
            finally:
                mp.sim_only = []
            return r.iloc[:, [jj]], s.iloc[:, [jj]]

        lab = f"{task.get('label', '')} {names[jj]}"
        h_t, h_x, b_t, b_x = [], [], False, False
        if task.get('target_off') is not None:
            t, off, h_t, b_t = E69.bisect_factor(lambda f: one(p_off, f), int(task['target_off'][jj]), n_kc, B3_THR_BRACKET,
                                                 B3_THR_STEP, f'{lab} threshold (APL off)')
        else:
            t = float(task['thr'][jj])
            off = one(p_off, t)
        on1 = one(p, t)
        if task.get('target_on') is not None:
            x, on, h_x, b_x = E69.bisect_factor(lambda v: one(_tied(p, v), t), int(task['target_on'][jj]), n_kc,
                                                B3_APL_BRACKET, APL_STEP, f'{lab} APL scale')
        else:
            x = float(task['apl'][jj]) if task.get('apl') is not None else 1.0
            on = on1 if x == 1.0 else one(_tied(p, x), t)
        out.append(dict(odor=str(names[jj]), thr=float(t), apl=float(x), at_bound_thr=bool(b_t), at_bound_apl=bool(b_x),
                        history=h_t + h_x, n_sims=len(h_t) + len(h_x) + 2,
                        **{k: (v[0].to_numpy(), v[1].to_numpy()) for k, v in (('off', off), ('on1', on1), ('on', on))}))
    return out


def _odor_worker(conn, parts: dict, js: List[int]) -> None:
    """An OdorPool worker: C2's simulator on the ORN columns `js` of the lab's input, then `_odor_task` for each task
    received (its odors' entries), until None."""
    try:
        S77._PARTS.update(parts)
        sim, info = S77.build_c2_simulator(orn=parts['orn'].iloc[:, js], breadth=parts['breadth'])
        cells = info['cells']
        mp, order = cells['mp'].cell_contents, np.asarray(cells['odor_order'].cell_contents, dtype=int)
        names = [str(c) for c in cells['odor_index'].cell_contents]
        conn.send(('ready', names))
    except Exception as e:
        conn.send(('error', repr(e)))
        return
    n_kc = int(parts['saved']['on'][0].shape[0])
    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            conn.send(('ok', _odor_task(sim, mp, order, names, dict(parts['p']), n_kc, msg)))
        except Exception as e:
            conn.send(('error', repr(e)))


class OdorPool:
    """B3's one-odor runs over worker processes (spawned, one OpenMP thread each, each with C2's simulator on its share
    of the odors' ORN columns, as E77's EmulationPool); with `n_workers` 0 or 1, in this process on C2's simulator."""

    _TASK_KEYS = ('thr', 'target_off', 'apl', 'target_on')

    def __init__(self, n_workers: int, base: Base):
        import multiprocessing
        import os
        P = S77.c2_parts()
        self.base = base
        self.template = P['saved']
        n_odors = self.template['on'][0].shape[1]
        self.cols = [str(c) for c in self.template['on'][0].columns]
        self.local = n_workers <= 1
        self.groups = [list(map(int, g)) for g in np.array_split(np.arange(n_odors), max(1, min(n_workers, n_odors)))]
        self.procs, self.conns = [], []
        if self.local:
            cells = base.info['cells']
            self._local = (base.sim, cells['mp'].cell_contents, np.asarray(cells['odor_order'].cell_contents, dtype=int),
                           [str(c) for c in cells['odor_index'].cell_contents])
            if self._local[3] != self.cols:
                raise RuntimeError("C2's simulator's odors are not its saved outputs' columns")
            return
        parts = {k: P[k] for k in ('orn', 'p', 'init', 'wnorm', 'W', 'breadth', 'saved', 'target_on', 'target_off', 'size', 'driven')}
        ctx = multiprocessing.get_context('spawn')
        keep = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT')}
        os.environ['OMP_NUM_THREADS'] = os.environ['OMP_THREAD_LIMIT'] = '1'
        try:
            for js in self.groups:
                a, b = ctx.Pipe()
                pr = ctx.Process(target=_odor_worker, args=(b, parts, js), daemon=True)
                pr.start()
                self.procs.append(pr)
                self.conns.append(a)
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for conn, js in zip(self.conns, self.groups):
            status, v = conn.recv()
            if status != 'ready':
                self.close()
                raise RuntimeError(f'an odor worker failed to start: {v}')
            if v != [self.cols[j] for j in js]:
                self.close()
                raise RuntimeError(f'an odor worker has odors {v}, expected {[self.cols[j] for j in js]}')
        print(f'odor pool: {len(self.procs)} workers, odors {self.groups}', flush=True)

    def run(self, task: dict) -> Tuple[Dict[str, tuple], List[dict]]:
        """A task over all odors (per-odor lists in the outputs' column order for any of 'thr', 'target_off', 'apl',
        'target_on'; 'label'). Returns ({'off', 'on1', 'on'}: (responses, spike counts) over all odors, per-odor
        records in column order)."""
        def part(js):
            return dict({k: (None if task.get(k) is None else [task[k][j] for j in js]) for k in self._TASK_KEYS},
                        label=task.get('label', ''))
        if self.local:
            sim, mp, order, names = self._local
            recs = _odor_task(sim, mp, order, names, dict(self.base.p), self.base.n_kc, part(list(range(len(self.cols)))))
        else:
            for conn, js in zip(self.conns, self.groups):
                conn.send(part(js))
            recs = [None] * len(self.cols)
            for conn, js in zip(self.conns, self.groups):
                status, v = conn.recv()
                if status != 'ok':
                    raise RuntimeError(f'an odor worker failed: {v}')
                for j, rec in zip(js, v):
                    recs[j] = rec
        outs = {}
        for key in ('off', 'on1', 'on'):
            src = self.template['off' if key == 'off' else 'on']
            R, S = src[0].copy(), src[1].copy()
            for j, rec in enumerate(recs):
                if rec['odor'] != self.cols[j]:
                    raise RuntimeError(f"odor {j}: {rec['odor']} where {self.cols[j]} was expected")
                R.iloc[:, j] = rec[key][0][:, 0]
                S.iloc[:, j] = rec[key][1][:, 0]
            outs[key] = (R, S)
        return outs, recs

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(None)
            except Exception:
                pass
        for pr in self.procs:
            pr.join(timeout=30)
            if pr.is_alive():
                pr.terminate()
        self.procs, self.conns = [], []


def b3_targets(units: str, cols: Sequence[str], n_kc: int) -> Dict[str, List[int]]:
    """B3's per-odor response counts (round(fraction x KCs)): 'target_off' the flies' with APL output blocked (TNT),
    'target_on' the control flies', in `units` ('noise': the registry's noise_<tnt|ctrl>_frac_<odor>; 'z': Yang's z,
    `kc_remapping_harness.yang_response_fractions`), in the outputs' column order."""
    import apl_effect_scores as A
    ab = [H.odor_abbrev(c) for c in cols]
    return {f'target_{c}': [int(round(float(A.frac_data(c, units)[o]) * n_kc)) for o in ab] for c in ('off', 'on')}


# ---------------------------------------------------------------------------
# the APL effect against the flies in baseline-noise units

_NOISE: Dict[str, object] = {}


def noise_data() -> dict:
    """The flies' per-pair APL effect in baseline-noise units (E76's effect_by_unit_pairs.csv, ctrl__noise - tnt__noise,
    checked against its effect__noise__graded), its mean and the registry's 90 % interval of the mean
    (noise_apl_effect_mean), the noise-unit calibration target of the readout (noise_apl_effect_cal_target), and the
    mean of the effect in Yang's z (the target as defined)."""
    import E72_wiring_population as E72
    if not _NOISE:
        path = H.shared_dir() / E76_PAIRS
        df = pd.read_csv(path)
        eff = pd.Series((df['ctrl__noise'] - df['tnt__noise']).to_numpy(dtype=float),
                        index=pd.MultiIndex.from_arrays([df['odor_a'], df['odor_b']], names=['odor_a', 'odor_b']))
        dev = float(np.abs(eff.to_numpy() - df['effect__noise__graded'].to_numpy(dtype=float)).max())
        if dev > 1e-12:
            raise RuntimeError(f'{path}: ctrl__noise - tnt__noise differs from effect__noise__graded by {dev}')
        lo, hi = H.data_target_ci('noise_apl_effect_mean')
        _NOISE.update(effect=eff, mean=float(eff.mean()), lo=float(lo), hi=float(hi), registry_mean=H.data_target('noise_apl_effect_mean'),
                      cal_target=float(H.data_target('noise_apl_effect_cal_target')), z_mean=float(E72.refs()['graded'].mean()),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(), path=str(path))
    return _NOISE


def noise_scores(o: Dict[str, tuple]) -> Tuple[dict, pd.DataFrame]:
    """A point's APL effect against the flies' in baseline-noise units: on raw counts and under the calibrated readout
    with K set on its APL-on counts to the noise-unit calibration target (`raw_noise_*`, `cal_noise_*`: loss, mean error,
    pattern, per-odor and pair-specific parts, ρ); the raw-count mean's error against the effect in Yang's z and whether
    it is inside the noise-unit interval. Returns (row, per-pair distances under the noise-unit readout)."""
    import E72_wiring_population as E72
    N = noise_data()
    s_on, s_off = o['on'][1].astype(float), o['off'][1].astype(float)
    K = H.calibrate_readout(s_on, 'saturating', N['cal_target'], stat='participation_ratio', responders_min=2.0)
    dist = {'raw': E72.distances(o, 'raw'),
            'cal': {c: H.readout_pair_distances(s, 'saturating', K) for c, s in (('on', s_on), ('off', s_off))}}
    # (as E76's part C: the calibration is at its bound when the readout misses the target by more than 2 %; K then
    # sits at calibrate_readout's upper end, where the readout is nearly linear, i.e. close to raw counts)
    got = H.readout_stat(s_on, 'saturating', K, 'participation_ratio', 2.0)
    row = {'cal_noise_K': float(K), 'cal_noise_K_at_bound': bool(abs(got - N['cal_target']) > 0.02 * N['cal_target'])}
    for ro, d in dist.items():
        dec, _ = H.apl_effect_decomposition(d['on'], d['off'], N['effect'])
        row.update({f'{ro}_noise_loss': dec['apl_effect_rmse'], f'{ro}_noise_mean': dec['apl_effect_mean'],
                    f'{ro}_noise_mean_error': dec['apl_effect_mean_error'], f'{ro}_noise_pattern': dec['apl_effect_rmse_centered'],
                    f'{ro}_noise_per_odor': dec['apl_effect_per_odor'], f'{ro}_noise_pair_specific': dec['apl_effect_pair_specific'],
                    f'{ro}_noise_rho': dec['apl_effect_spearman']})
    m = row['raw_noise_mean']
    row.update(raw_mean_error_z=m - N['z_mean'], raw_mean_in_noise_interval=bool(N['lo'] <= m <= N['hi']))
    d = dist['cal']
    # (pairs in E77's canonical order, each pair's odors sorted, as its Scorer writes them)
    pairs = pd.DataFrame({'cal_noise_on': S77._canon_series(d['on']), 'cal_noise_off': S77._canon_series(d['off']),
                          'cal_noise_effect': S77._canon_series(H.apl_effect(d['on'], d['off']))}
                         ).rename_axis(['odor_a', 'odor_b']).reset_index()
    return row, pairs


# ---------------------------------------------------------------------------
# the points

def kc_type_label(t: str) -> str:
    return KC_TYPES[t]


def point_specs() -> List[dict]:
    """Part B's points, each family (handle) with its identity point first, controls after their candidates:
    B1 (C2: c 1 then 1.1, 1.25, 1.5, 2), B1_fit0 (the same on E76's fit 0), B2 (eps 0, 0.05, 0.1, 0.2, 0.4; B2_scaled),
    B3 (the one-odor runs at factors 1; per unit, noise then z: thr, throff, thrapl), B4_beta (beta 0, 0.25, 0.5, 1, 2;
    B4_beta_perm), B4_type (alpha-beta x 1; each type x 1.25 and x 1.5; B4_type_opp), B5 (alpha-beta x 1; each type x 0.5
    and x 2).

    A spec has point_id, handle, setting, value (the setting's number; 'best' for a control until resolved), kind
    ('b1', 'b2', 'b3', 'b4', 'b5'), base ('C2' or 'fit0'), identity, and per kind: c (b1); eps or control 'scaled' (b2);
    units and arm (b3; the identity has arm 'identity'); beta, or kc_type and factor, and permute_seed (b4); kc_type and
    factor (b5); candidates (the points whose best, by the pattern part under the readout, a control takes)."""
    pts: List[dict] = []

    def add(pid, handle, setting, value, kind, base='C2', **kw):
        pts.append(dict(point_id=pid, handle=handle, setting=setting, value=value, kind=kind, base=base,
                        identity=setting == 'identity', **kw))

    for base, h in (('C2', 'B1'), ('fit0', 'B1_fit0')):
        add(f'{h}_identity', h, 'identity', 1.0, 'b1', base=base, c=1.0)
        for c in B1_C:
            add(f'{h}_c{c:g}', h, f'c{c:g}', c, 'b1', base=base, c=c)
    add('B2_identity', 'B2', 'identity', 0.0, 'b2', eps=0.0)
    for e in B2_EPS:
        add(f'B2_eps{e:g}', 'B2', f'eps{e:g}', e, 'b2', eps=e)
    add('B2_scaled', 'B2', 'scaled', 'best', 'b2', eps='best', control='scaled', candidates=tuple(f'B2_eps{e:g}' for e in B2_EPS))
    add('B3_identity', 'B3', 'identity', 1.0, 'b3', units=None, arm='identity')
    for u in B3_UNITS:
        for arm in B3_ARMS:
            add(f'B3_{arm}_{u}', 'B3', f'{arm}_{u}', np.nan, 'b3', units=u, arm=arm)
    add('B4_beta_identity', 'B4_beta', 'identity', 0.0, 'b4', beta=0.0)
    for b in B4_BETAS:
        add(f'B4_beta{b:g}', 'B4_beta', f'beta{b:g}', b, 'b4', beta=b)
    add('B4_beta_perm', 'B4_beta', 'perm', 'best', 'b4', beta='best', permute_seed=PERM_SEED,
        candidates=tuple(f'B4_beta{b:g}' for b in B4_BETAS))
    types = list(KC_TYPES)
    add('B4_type_identity', 'B4_type', 'identity', 1.0, 'b4', kc_type=types[0], factor=1.0)
    for t in types:
        for f in B4_TYPE_FACTORS:
            add(f'B4_type_{kc_type_label(t)}_x{f:g}', 'B4_type', f'{kc_type_label(t)}_x{f:g}', f, 'b4', kc_type=t, factor=f)
    add('B4_type_opp', 'B4_type', 'opp', 'best', 'b4', kc_type='best', factor='best',
        candidates=tuple(f'B4_type_{kc_type_label(t)}_x{f:g}' for t in types for f in B4_TYPE_FACTORS))
    add('B5_identity', 'B5', 'identity', 1.0, 'b5', kc_type=types[0], factor=1.0)
    for t in types:
        for f in B5_FACTORS:
            add(f'B5_{kc_type_label(t)}_x{f:g}', 'B5', f'{kc_type_label(t)}_x{f:g}', f, 'b5', kc_type=t, factor=f)
    return pts


def resolve(pt: dict, rows: pd.DataFrame) -> Optional[dict]:
    """A control point with its setting taken from the best of its candidates (the lowest pattern part under the
    readout, cal_pattern, as the gate picks a handle's best point); None while they are not all done."""
    if not pt.get('candidates'):
        return pt
    have = rows[rows.point_id.isin(pt['candidates'])] if len(rows) else rows
    if len(have) < len(pt['candidates']):
        return None
    best = have.loc[have.cal_pattern.astype(float).idxmin()]
    v = float(best.value)
    q = dict(pt, chosen_from=str(best.point_id))
    if pt['kind'] == 'b2':
        q.update(eps=v, value=v, setting=f'scaled_eps{v:g}')
    elif 'beta' in pt:
        q.update(beta=v, value=v, setting=f'perm_beta{v:g}')
    else:
        t = str(best.kc_type)
        q.update(kc_type=t, factor=1.0 / v, value=1.0 / v, setting=f'opp_{kc_type_label(t)}_x{1.0 / v:.4g}')
    return q


def control_of(pid: str, specs: Sequence[dict]) -> Optional[str]:
    """The gate's control of a point: B2's scaled profile, B4_beta's permutation, B4_type's opposite direction (for every
    point of the handle, as E77 gives S5's permutation to both exponents), B5's opposite direction of the same type; None
    for B3 (an oracle, no control) and for the controls and identities themselves."""
    sp = {s['point_id']: s for s in specs}
    s = sp.get(pid)
    if s is None or s['identity'] or s.get('candidates'):
        return None
    if s['handle'] == 'B2':
        return 'B2_scaled'
    if s['handle'] == 'B4_beta':
        return 'B4_beta_perm'
    if s['handle'] == 'B4_type':
        return 'B4_type_opp'
    if s['handle'] == 'B5':
        other = [f for f in B5_FACTORS if f != s['factor']]
        return f"B5_{kc_type_label(s['kc_type'])}_x{other[0]:g}" if len(other) == 1 else None
    return None


# ---------------------------------------------------------------------------
# the screen

def _save_outputs(tag: str, o: Dict[str, tuple], kc_thr: Optional[pd.Series] = None) -> None:
    d = H.outputs_dir(EXPERIMENT) / 'screen'
    d.mkdir(exist_ok=True)
    for c, cc in (('on', 'on'), ('off', 'acute_off')):
        o[c][0].to_parquet(d / f'{tag}_{cc}_responses.parquet')
        o[c][1].to_parquet(d / f'{tag}_{cc}_spike_counts.parquet')
    if kc_thr is not None:
        kc_thr.rename('kc_thr_factor').to_frame().to_parquet(d / f'{tag}_kc_thr_factor.parquet')


def check_identity(pid: str, o: Dict[str, tuple], saved: Dict[str, tuple], factors: Sequence[float] = ()) -> None:
    """An identity point must reproduce `saved` (its base's saved spike counts) exactly in both conditions, at
    bisected factors of exactly 1."""
    for c in ('on', 'off'):
        n = int((o[c][1].to_numpy(dtype=float) != saved[c][1].to_numpy(dtype=float)).sum())
        nr = int((o[c][0].to_numpy(dtype=float) != saved[c][0].to_numpy(dtype=float)).sum())
        if n or nr or not o[c][1].index.equals(saved[c][1].index) or not o[c][1].columns.equals(saved[c][1].columns):
            raise RuntimeError(f"{pid}: the identity setting differs from its base's saved {c} run in {n} counts and {nr} responses")
    bad = [f for f in factors if f != 1.0]
    if bad:
        raise RuntimeError(f'{pid}: the identity setting needed factors {list(factors)}')
    print(f"{pid}: reproduces its base's saved APL-on and APL-off spike counts exactly", flush=True)


def write_provenance(out: Path, specs: Sequence[dict], bases: Dict[str, Base]) -> None:
    """screen_provenance.json: the model's provenance (`H.model_provenance`, on the lab's input and C2's wiring), each
    base's parameters and kwargs, the targets, the points, the brackets, the data behind the noise-unit readings."""
    P = S77.c2_parts()
    N = noise_data()
    reg = H.data_targets_path()
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E80_screen.py',
                al_analysis_commit=S77._git_head(), model=H.model_provenance(orn_deltas=P['orn'], _wPNKC=P['W']),
                bases={k: dict(experiment=b.experiment, tag=b.tag, label=b.label, params=b.p, kws=b.kws,
                               targets=dict(on=b.target_on, off=b.target_off, size=b.size), build_seconds=b.build_seconds,
                               info={kk: v for kk, v in b.info.items() if kk == 'fit_info'})
                       for k, b in bases.items()},
                bouton_base_params=P['init'], breadth_k=S77.BREADTH_K, tied_on=TIED_ON,
                brackets=dict(thr=THR_BRACKET, thr_step=THR_STEP, apl=APL_BRACKET, apl_step=APL_STEP, b1_apl=B1_APL_BRACKET,
                              b3_thr=B3_THR_BRACKET, b3_thr_step=B3_THR_STEP, b3_apl=B3_APL_BRACKET),
                points=list(specs), perm_seed=PERM_SEED,
                noise=dict(path=N['path'], sha256=N['sha256'], mean=N['mean'], interval=(N['lo'], N['hi']),
                           cal_target=N['cal_target'], z_mean=N['z_mean']),
                data_targets_sha256=hashlib.sha256(reg.read_bytes()).hexdigest() if reg.exists() else None,
                approx_conc_note=H.APPROX_CONC_NOTE,
                definitions=dict(
                    threshold_factor="every per-KC or per-odor threshold factor multiplies the KC's whole threshold "
                                     "(fixed_thr + 2 x its spontaneous input) through the simulator's kc_thr_factor; the "
                                     "common threshold re-bisections (B2 via hold_rates, B4) multiply fixed_thr",
                    b2="eps x the odor's mean over the 36 glomeruli of max(delta, 0) added to every glomerulus; control: "
                       "the odor's positive deltas scaled to the same added summed delta",
                    b3="per-odor threshold factors bisected with APL off to the TNT flies' fractions; thr: in both "
                       "conditions at C2's APL scales; throff: APL off only; thrapl: plus per-odor APL scales (TIED_ON) "
                       "bisected at those thresholds to the control flies' fractions",
                    controls='by the lowest cal_pattern among the candidates'))
    (out / 'screen_provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')


def _family_ok(pt: dict, specs: Sequence[dict], done: set) -> Optional[str]:
    """The identity point of `pt`'s family that has not passed yet, or None."""
    if pt['identity']:
        return None
    return next((s['point_id'] for s in specs if s['identity'] and s['handle'] == pt['handle'] and s['point_id'] not in done), None)


class _Screen:
    """The screen's state: the bases (built on first use, each checked), the rows written so far, the odor pool."""

    def __init__(self, out: Path, workers: Optional[int]):
        self.out, self.workers = out, workers
        self.sp, self.pp, self.jp, self.ep = (out / f for f in ('screen.csv', 'screen_pairs.csv', 'screen_params.json',
                                                                'screen_errors.csv'))
        self.rows = pd.read_csv(self.sp) if self.sp.exists() else pd.DataFrame()
        self.done = set(self.rows.point_id) if len(self.rows) else set()
        self.params = json.loads(self.jp.read_text()) if self.jp.exists() else {}
        self.bases: Dict[str, Base] = {}
        self.pool: Optional[OdorPool] = None
        self.b3: Dict[str, Tuple[Dict[str, tuple], List[dict]]] = {}

    def base(self, name: str) -> Base:
        if name not in self.bases:
            self.bases[name] = Base(name)
            pid = 'base' if name == 'C2' else f'base_{name}'
            if pid not in self.done:
                b = self.bases[name]
                self.record(dict(point_id=pid, handle='', setting=name, value=np.nan, kind='base', base=name, identity=True),
                            b.saved, dict(params=dict(b.p), params_off=b.off_params()),
                            dict(note=f'{b.label} as saved, rebuilt and checked exactly'))
        return self.bases[name]

    def record(self, pt: dict, o: Dict[str, tuple], res: dict, extra: dict, kc_thr: Optional[pd.Series] = None) -> None:
        b = self.bases[pt['base']]
        t0 = time.time()
        row, pairs = b.scorer().score(o)
        nrow, npairs = noise_scores(o)
        p_on, p_off = res['params'], res.get('params_off') or b.off_params(res['params'])
        meta = dict(point_id=pt['point_id'], handle=pt['handle'], setting=pt['setting'], value=pt['value'], kind=pt['kind'],
                    base_model=pt['base'], identity=bool(pt['identity']), kc_type=pt.get('kc_type'), units=pt.get('units'),
                    fixed_thr=p_on['fixed_thr'], fixed_thr_off=p_off['fixed_thr'],
                    **{k: p_on[k] for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')},
                    thr_factor=res.get('thr_factor', 1.0), apl_factor=res.get('apl_factor', 1.0),
                    kc_thr_factor_on=res.get('kc_thr_factor_on', 1.0), n_on=S77._n(o['on'][0]), n_off=S77._n(o['off'][0]),
                    target_on=b.target_on, target_off=b.target_off, at_bound_thr=res.get('at_bound_thr', False),
                    at_bound_apl=res.get('at_bound_apl', False), n_sims=res.get('n_sims', 0),
                    seconds=res.get('seconds', 0.0) + (time.time() - t0), note=extra.get('note', ''))
        self.rows = S77._write_rows(self.sp, self.rows, dict(meta, **row, **nrow))
        pairs = pairs.merge(npairs, on=['odor_a', 'odor_b'], how='left', validate='one_to_one')
        pairs.insert(0, 'point_id', pt['point_id'])
        pairs.to_csv(self.pp, index=False, mode='a', header=not self.pp.exists())
        self.params[pt['point_id']] = dict(pt, params=p_on, params_off=p_off, **extra, thr_factor=meta['thr_factor'],
                                           apl_factor=meta['apl_factor'], n_on=meta['n_on'], n_off=meta['n_off'],
                                           history=res.get('history', []))
        self.jp.write_text(json.dumps(self.params, indent=1, default=str) + '\n')
        _save_outputs(f"{pt['base']}_{pt['point_id']}", o, kc_thr)
        self.done.add(pt['point_id'])
        r = self.rows.iloc[-1]
        flag = r.realism_flag if isinstance(r.realism_flag, str) and r.realism_flag else 'inside the bands'
        print(f"{pt['point_id']}: thr x{meta['thr_factor']:.4f}, APL x{meta['apl_factor']:.4f} ({meta['n_off']} / {meta['n_on']} "
              f"responses off / on); readout loss {r.cal_loss:.4f} (mean {r.cal_mean:+.3f}, pattern {r.cal_pattern:.4f}), raw "
              f"{r.raw_loss:.4f} (mean {r.raw_mean:+.3f}; flies {noise_data()['z_mean']:+.3f} in z, {noise_data()['mean']:+.3f} in "
              f"noise units, raw pattern vs noise {r.raw_noise_pattern:.4f}); transfer {r.cal_transfer_all:+.3f} / six "
              f"{r.cal_transfer_six:+.3f}; per-odor {r.cal_per_odor:.4f}; realism: {flag}", flush=True)

    # one runner per kind: (o, res, extra, kc_thr factors or None)
    def run_b1(self, pt: dict):
        b = self.base(pt['base'])
        o, res = simulate_b1(b, float(pt['c']), pt['point_id'])
        note = 'APL-on threshold x 1/c (every KC); APL-off run the base\'s own'
        if float(pt['c']) > B1_DEFENSIBLE_C:
            note += f'; c above {B1_DEFENSIBLE_C:g} is not defensible without an acute-block measurement'
        return o, res, dict(note=note, c=float(pt['c'])), None

    def run_b2(self, pt: dict):
        P = S77.c2_parts()
        self.base('C2')
        if pt.get('control') == 'scaled':
            added = shared_input_drive(P['orn'], float(pt['eps']))
            orn = scaled_profile(P['orn'], added)
            note = f"control: each odor's own positive deltas scaled to B2 eps {float(pt['eps']):g}'s added summed drive"
        else:
            orn = shared_input(P['orn'], float(pt['eps']))
            added = shared_input_drive(P['orn'], float(pt['eps']))
            note = 'shared lateral input; both rates re-held, breadth for the input'
        o, res = simulate_static_orn(orn, pt['point_id'])
        extra = dict(note=note, orn_sha256=S77.frame_sha256(orn), sim_kws=res['kws'], breadth_sha256=res['breadth_sha256'],
                     added_drive={str(c): float(v) for c, v in zip(P['orn'].columns.get_level_values(-1), added)},
                     summed_drive={str(c): float(v) for c, v in zip(P['orn'].columns.get_level_values(-1), orn.to_numpy().sum(axis=0))})
        return o, res, extra, None

    def _pool(self) -> OdorPool:
        if self.pool is None:
            n = S77.default_workers() if self.workers is None else int(self.workers)
            self.pool = OdorPool(n, self.base('C2'))
        return self.pool

    def run_b3(self, pt: dict):
        b = self.base('C2')
        pool = self._pool()
        t0 = time.time()
        if pt['arm'] == 'identity':
            n = len(pool.cols)
            outs, recs = pool.run(dict(thr=[1.0] * n, apl=[1.0] * n, label=pt['point_id']))
            o = {'on': outs['on1'], 'off': outs['off']}
            check_identity(f"{pt['point_id']} (one-odor runs at factors 1)", o, b.saved)
        else:
            u = pt['units']
            if u not in self.b3:
                tg = b3_targets(u, pool.cols, b.n_kc)
                outs, recs = pool.run(dict(tg, label=f'B3 {u}'))
                self.b3[u] = (outs, recs, tg)
            outs, recs, tg = self.b3[u]
            on = {'thr': outs['on1'], 'throff': b.saved['on'], 'thrapl': outs['on']}[pt['arm']]
            o = {'on': on, 'off': outs['off']}
        per = {r['odor']: dict(thr=r['thr'], apl=r['apl'] if pt['arm'] == 'thrapl' else 1.0,
                               at_bound_thr=r['at_bound_thr'], at_bound_apl=r['at_bound_apl'] if pt['arm'] == 'thrapl' else False)
               for r in recs}
        thr_on = 'per odor (as APL off)' if pt['arm'] in ('thr', 'thrapl') else 1.0
        res = dict(params=dict(b.p), params_off=b.off_params(), thr_factor=1.0, apl_factor=1.0,
                   at_bound_thr=any(v['at_bound_thr'] for v in per.values()),
                   at_bound_apl=any(v['at_bound_apl'] for v in per.values()),
                   n_sims=0 if pt['arm'] == 'throff' else int(sum(r['n_sims'] for r in recs)),
                   history=[] if pt['arm'] == 'identity' else [dict(h, odor=r['odor']) for r in recs for h in r['history']],
                   seconds=time.time() - t0, kc_thr_factor_on=thr_on)
        note = {'identity': 'one-odor runs at factors 1',
                'thr': "per-odor thresholds on the TNT flies' fractions, both conditions, C2's APL scales",
                'throff': "per-odor thresholds on the TNT flies' fractions, APL off only; the APL-on run C2's own",
                'thrapl': "per-odor thresholds on the TNT flies' fractions, then per-odor APL scales on the control flies'"}[pt['arm']]
        if u_note := pt.get('units'):
            note += f" ({'baseline-noise units' if u_note == 'noise' else 'Yang z'}); oracle"
        extra = dict(note=note, per_odor=per)
        if pt['arm'] != 'identity':
            extra['targets'] = self.b3[pt['units']][2]
        return o, res, extra, None

    def run_b4(self, pt: dict):
        b = self.base('C2')
        idx = b.type_index()
        if 'beta' in pt:
            act = pd.Series(b.saved['off'][1].to_numpy(dtype=float).sum(axis=1), index=idx)
            f = activity_factors(act, float(pt['beta']))
            note = 'per-KC threshold x (1 + beta x summed APL-off count / mean), APL off only'
            if pt.get('permute_seed') is not None:
                f = permute_within_type(f, int(pt['permute_seed']))
                note += f"; permuted within KC type (seed {pt['permute_seed']})"
        else:
            f = type_factors(idx, pt['kc_type'], float(pt['factor']))
            note = f"{pt['kc_type']} KCs' threshold x {float(pt['factor']):g}, APL off only"
        o, res = simulate_off_thr(b, f, pt['point_id'])
        fa = f.to_numpy(dtype=float)
        extra = dict(note=note + '; oracle (stands against the one published test)', kc_thr_factor_sha256=_arr_sha256(fa),
                     kc_thr_factor_min=float(fa.min()), kc_thr_factor_max=float(fa.max()), kc_thr_factor_mean=float(fa.mean()))
        return o, res, extra, f

    def run_b5(self, pt: dict):
        b = self.base('C2')
        o, res = simulate_b5(b, pt['kc_type'], float(pt['factor']), pt['point_id'])
        note = f"{pt['kc_type']} KCs' APL->KC weights x {float(pt['factor']):g}; APL-off run C2's own"
        return o, res, dict(note=note, weight_factor_sha256=res['weight_factor_sha256'],
                            n_weights_scaled=res['n_weights_scaled']), None

    def run(self, specs: Sequence[dict], want: set) -> None:
        for spec in specs:
            pid = spec['point_id']
            if pid not in want or pid in self.done:
                continue
            pt = resolve(spec, self.rows)
            if pt is None:
                print(f"{pid}: waits for {list(spec['candidates'])}; skipped", flush=True)
                continue
            missing = _family_ok(pt, specs, self.done)
            if missing is not None:
                print(f'{pid}: its identity point {missing} has not passed; skipped', flush=True)
                continue
            if pt['kind'] != 'b3' and self.pool is not None:
                # (B3's points are consecutive: its workers' simulators are freed before the next family)
                self.close()
            print(f"--- {pid} ({pt['kind']}: {pt['setting']}, on {pt['base']})", flush=True)
            try:
                o, res, extra, kc_thr = getattr(self, f"run_{pt['kind']}")(pt)
                if pt['identity']:
                    check_identity(pid, o, self.bases[pt['base']].saved, [res.get('thr_factor', 1.0), res.get('apl_factor', 1.0)])
            except RuntimeError as e:
                if pt['identity'] or 'identity' in str(e) or 'rebuilt' in str(e):
                    raise
                self._error(pt, e)
                continue
            except Exception as e:
                if pt['identity']:
                    raise
                self._error(pt, e)
                continue
            self.record(pt, o, res, extra, kc_thr)
            del o, res
            gc.collect()

    def _error(self, pt: dict, e: Exception) -> None:
        print(f"{pt['point_id']} FAILED: {e!r}"[:500], flush=True)
        pd.DataFrame([dict(point_id=pt['point_id'], handle=pt['handle'], setting=pt['setting'],
                           time=datetime.now().isoformat(timespec='seconds'), error=repr(e)[:500])]).to_csv(
            self.ep, index=False, mode='a', header=not self.ep.exists())

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool = None


def screen(out: Path, only: Optional[Sequence[str]] = None, workers: Optional[int] = None) -> pd.DataFrame:
    """Runs part B (resumably) and scores each point as E77's screen does, plus the APL effect against the flies in
    baseline-noise units; writes screen.csv, screen_pairs.csv, screen_params.json (and screen_errors.csv,
    screen_provenance.json).

    Args:
        out: the results folder (results/E80_remaining_candidates/).
        only: point ids to run (default all); the identity points of their families are run first if not done.
        workers: B3's odor worker processes (`OdorPool`; default E77's `default_workers()`, half the OpenMP thread
            limit, at most 8).

    Returns screen.csv's rows. A point that fails is listed in screen_errors.csv and not marked done; an identity point
    that does not reproduce its base stops the screen (RuntimeError).
    """
    out.mkdir(parents=True, exist_ok=True)
    specs = point_specs()
    ids = [s['point_id'] for s in specs]
    want = set(ids) if only is None else set(only)
    unknown = want - set(ids)
    if unknown:
        raise ValueError(f'unknown points {sorted(unknown)}')
    if only is not None:
        fams = {s['handle'] for s in specs if s['point_id'] in want}
        want |= {s['point_id'] for s in specs if s['identity'] and s['handle'] in fams}
    st = _Screen(out, workers)
    try:
        bases_needed = {s['base'] for s in specs if s['point_id'] in want and s['point_id'] not in st.done}
        for b in sorted(bases_needed, key=BASES.index):
            st.base(b)
        if bases_needed or not (out / 'screen_provenance.json').exists():
            write_provenance(out, specs, st.bases)
        st.run(specs, want)
    finally:
        st.close()
    return st.rows


# ---------------------------------------------------------------------------
# part C: the gate

def gate_table(s: pd.DataFrame, specs: Sequence[dict]) -> pd.DataFrame:
    """E77's gate (E77_input_side_common_error.gate's conditions and constants) on the screen's rows `s` (indexed by
    point_id, with a 'base' row for C2), for every handle but B1 / B1_fit0: a point (1) lowers C2's pattern part under
    the readout by GATE_PATTERN_GAIN or more with the raw-count loss no more than GATE_RAW_SLACK above C2's, (2) has a
    transfer reduction of the common error of GATE_TRANSFER or more and, against the six typical TNT recordings, at least
    half that share, (3) where it has a control (`control_of`), the control gains less than half as much on the pattern
    part; passes if inside the four bands (the rule since 2026-11-10 16:10) and all three hold. A handle's best point is
    its lowest cal_pattern among its points that are not identities or controls; it goes on if it passes."""
    import E77_input_side_common_error as G
    base = s.loc['base']
    sp = {x['point_id']: x for x in specs}
    keep = [i for i in s.index if i in sp and not sp[i]['identity'] and sp[i]['handle'] not in ('B1', 'B1_fit0')]
    pts = s.loc[keep].copy()
    gain = base.cal_pattern - pts.cal_pattern.astype(float)
    pts['pattern_gain'] = gain
    flag = lambda col: pts[col].isna() | (pts[col].astype(str).str.strip().isin(['', 'nan']))
    pts['inside_bands'] = flag('realism_flag')
    pts['inside_bands_zcrit'] = flag('realism_flag_zcrit')
    pts['c1'] = (gain >= G.GATE_PATTERN_GAIN) & (pts.raw_loss.astype(float) <= base.raw_loss + G.GATE_RAW_SLACK)
    pts['c2'] = (pts.cal_transfer_all >= G.GATE_TRANSFER) & (pts.cal_transfer_six >= 0.5 * pts.cal_transfer_all)
    ctrl = [control_of(i, specs) for i in pts.index]
    pts['control'] = ctrl
    pts['control_gain'] = [float(gain.get(c, np.nan)) if c else np.nan for c in ctrl]
    pts['c3'] = [True if c is None else bool(np.isfinite(cg) and cg < 0.5 * g) for c, cg, g in zip(ctrl, pts.control_gain, gain)]
    pts['passes'] = pts.inside_bands & pts.c1 & pts.c2 & pts.c3
    is_control = np.array([bool(sp[i].get('candidates')) for i in pts.index])
    pts['is_control'] = is_control
    pts['best'] = False
    for h, g in pts[~is_control].groupby('handle'):
        if g.cal_pattern.notna().any():
            pts.loc[g.cal_pattern.astype(float).idxmin(), 'best'] = True
    pts['goes_on'] = pts.best & pts.passes
    pts['base_cal_pattern'] = base.cal_pattern
    pts['base_raw_loss'] = base.raw_loss
    cols = ['handle', 'setting', 'kind', 'value', 'kc_type', 'units', 'best', 'goes_on', 'passes', 'is_control',
            'inside_bands', 'inside_bands_zcrit', 'c1', 'c2', 'c3', 'cal_loss', 'cal_mean', 'cal_pattern', 'pattern_gain',
            'cal_per_odor', 'cal_pair_specific', 'raw_loss', 'raw_mean', 'raw_pattern', 'raw_noise_loss', 'raw_noise_mean_error',
            'raw_noise_pattern', 'cal_noise_pattern', 'cal_rho', 'cal_transfer_all', 'cal_transfer_six', 'control', 'control_gain',
            'cal_err_ester', 'cal_err_alcohol', 'cal_err_ms', 'cal_err_nearest', 'cal_err_block', 'cal_block', 'rate_on',
            'rate_off', 'ratio', 'rate_rho_on', 'opk_on_tv', 'opk_off_tv', 'realism_flag', 'realism_flag_zcrit', 'thr_factor',
            'apl_factor', 'at_bound_thr', 'at_bound_apl', 'note', 'base_cal_pattern', 'base_raw_loss']
    return pts[[c for c in cols if c in pts.columns]].rename_axis('point_id').reset_index()


def b1_curve(s: pd.DataFrame) -> pd.DataFrame:
    """B1 as a curve against c (c 1 is each base's identity), per base: the raw-count mean APL effect against the flies'
    in Yang's z and in baseline-noise units (with the noise-unit interval), the loss and pattern parts (readout, raw
    counts against z and against the noise-unit effect), the response rates and bands, the APL factor; and outcome (b):
    for the firing-ceiling fit, whether the mean reaches the noise-unit interval at c <= B1_DEFENSIBLE_C with the
    raw-count pattern part against the noise-unit effect within B1_PATTERN_SLACK of its own."""
    N = noise_data()
    rows = []
    for h, base_pid in (('B1', 'base'), ('B1_fit0', 'base_fit0')):
        g = s[s.handle == h]
        if not len(g) or base_pid not in s.index:
            continue
        b = s.loc[base_pid]
        for pid, r in g.iterrows():
            c = float(r.value)
            row = dict(point_id=pid, base_model=r.base_model, c=c, defensible=c <= B1_DEFENSIBLE_C,
                       raw_mean=r.raw_mean, flies_mean_z=N['z_mean'], flies_mean_noise=N['mean'], flies_noise_lo=N['lo'],
                       flies_noise_hi=N['hi'], raw_mean_error_z=r.raw_mean_error_z, raw_noise_mean_error=r.raw_noise_mean_error,
                       raw_mean_in_noise_interval=bool(r.raw_mean_in_noise_interval),
                       raw_loss=r.raw_loss, raw_pattern=r.raw_pattern, raw_noise_loss=r.raw_noise_loss,
                       raw_noise_pattern=r.raw_noise_pattern, raw_noise_pattern_change=float(r.raw_noise_pattern - b.raw_noise_pattern),
                       cal_loss=r.cal_loss, cal_mean=r.cal_mean, cal_pattern=r.cal_pattern,
                       cal_pattern_change=float(r.cal_pattern - b.cal_pattern), cal_noise_loss=r.cal_noise_loss,
                       cal_noise_pattern=r.cal_noise_pattern, rate_on=r.rate_on, rate_off=r.rate_off, ratio=r.ratio,
                       rate_rho_on=r.rate_rho_on, max_count_on=r.max_count_on, apl_factor=r.apl_factor,
                       at_bound_apl=r.at_bound_apl, realism_flag=r.realism_flag, realism_flag_zcrit=r.realism_flag_zcrit)
            row['outcome_b'] = bool(r.base_model == 'fit0' and row['defensible'] and row['raw_mean_in_noise_interval']
                                    and row['raw_noise_pattern_change'] <= B1_PATTERN_SLACK)
            rows.append(row)
    return pd.DataFrame(rows)


def gate(out: Path) -> pd.DataFrame:
    """Part C: E77's gate on part B (`gate_table`); writes gate.csv (every gated non-identity point with its conditions;
    `best` marks each handle's best point, `goes_on` the ones that go on) and b1_curve.csv (B1, not gated: the curve
    against c on C2 and on the firing-ceiling fit, with outcome (b))."""
    s = pd.read_csv(out / 'screen.csv').drop_duplicates('point_id', keep='last').set_index('point_id')
    specs = point_specs()
    res = gate_table(s, specs)
    res.to_csv(out / 'gate.csv', index=False)
    curve = b1_curve(s)
    curve.to_csv(out / 'b1_curve.csv', index=False)
    base = s.loc['base']
    show = ['point_id', 'best', 'passes', 'cal_loss', 'cal_pattern', 'pattern_gain', 'raw_loss', 'cal_transfer_all',
            'cal_transfer_six', 'control', 'control_gain', 'inside_bands']
    print(f"C2: pattern {base.cal_pattern:.4f}, raw loss {base.raw_loss:.4f}", flush=True)
    if len(res):
        print(res[[c for c in show if c in res.columns]].round(4).to_string(index=False), flush=True)
    on = list(res.loc[res.goes_on, 'point_id']) if len(res) else []
    print(f'goes on: {on or "none"}', flush=True)
    if len(curve):
        print(curve[['point_id', 'c', 'raw_mean', 'flies_mean_noise', 'raw_mean_in_noise_interval', 'raw_noise_pattern',
                     'cal_pattern', 'ratio', 'apl_factor', 'outcome_b']].round(4).to_string(index=False), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', nargs='+', help="point ids to run (their families' identity points run first)")
    ap.add_argument('--list', action='store_true', help='list the points and exit')
    ap.add_argument('--workers', type=int, help="B3's odor worker processes (default: half the OpenMP thread limit, at most 8)")
    ap.add_argument('--gate', action='store_true', help='part C: the gate on screen.csv (no simulations)')
    args = ap.parse_args()
    if args.list:
        for s in point_specs():
            print(s['point_id'], s['kind'], s['base'], s['setting'])
        return
    out = H.results_dir(EXPERIMENT)
    if args.gate:
        gate(out)
        return
    screen(out, only=args.only, workers=args.workers)


if __name__ == '__main__':
    main()
