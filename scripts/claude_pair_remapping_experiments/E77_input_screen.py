#!/usr/bin/env python3
"""E77 part C: the screen of the model's input handles on C2 (the breadth-2 APL-effect fit, E67) at fixed scales on the
real wiring, both response rates held, and the APL-on-only emulations (L1 / L2). Used by E77_input_side_common_error.py
(--part C); the design is the E77 entry's part C in experiment_priorities.md.

Static handles (S1-S6) act in both conditions: for each point the somatic threshold is bisected to C2's APL-off response
count, the breadth weights are recomputed for the handle, and the APL->KC scale (KC->APL and bouton->APL with it) is
bisected to C2's APL-on response count (1852); KC-KC and APL->bouton scales unchanged (`hold_rates`). Emulations act only
with APL on: the APL-off run is C2's own and only the APL->KC scale is bisected, each odor simulated on its own with its
own APL weights (`PerOdorRunner`).

The breadth weights of a handle are computed as E10_joint_fit.bouton_base computes them for another ORN input or wiring
(`handle_breadth`): each KC's total spike count in the bouton base's APL-off run at the base's own parameters (E3's,
somatic threshold 254, no KC-KC inhibition), with the handle applied. C2's own breadth comes from that run, not from
C2's APL-off run (threshold 102), so this is the definition under which a handle at its identity setting reproduces C2
exactly and under which E10's re-fits with a handle (part D) are built; `hold_rates(..., breadth_from='held')` takes
them from the threshold-held APL-off run instead.

Every handle family starts with its identity point (S1 the default pn_inhadd / pn_inhsc passed explicitly, S2 E38's
lab.csv through `read_orn_input`, S3 power 1, S5 exponent 0, L power 0 at both sites), which must reproduce C2's saved
APL-on and APL-off spike counts exactly at threshold and APL factors 1; the screen stops otherwise. Their rows are in
screen.csv with identity True.

Reusable pieces (also for other scripts, e.g. a per-claw knee on C2's family): `c2_parts`, `handle_breadth`,
`build_c2_simulator`, `hold_rates`, `PerOdorRunner`.

Outputs, results/E77_input_side_common_error/: screen.csv (one row per point; resumable), screen_pairs.csv (per point
and pair: distances APL on / off and the APL effect on raw counts, under the calibrated readout and binarized),
screen_params.json (each point's absolute parameters and inputs), screen_errors.csv (points that failed),
screen_provenance.json. Spike counts and responses of every point under outputs/E77_input_side_common_error/screen/
(tag C2_<point_id>, conditions on / acute_off).

Usage (from this folder):
    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 nice -n 10 ../../venv/bin/python -u E77_input_screen.py [--only ID ...] [--list]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E77_input_side_common_error'
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
BREADTH_K = 2.0
# ProfileModel.run_on's tie: the APL->KC scale with KC->APL and bouton->APL (APL->bouton unchanged)
TIED_ON = ('wAPLKC_scale', 'wKCAPL_scale', 'wPNAPL_scale')
THR_BRACKET, THR_STEP = (0.1, 10.0), 1.1
APL_BRACKET, APL_STEP = (0.1, 10.0), 1.25
PN_WINDOW = (0.0, 0.5)

E38_INPUTS = 'results/E38_hallem_input/inputs'
S2_INPUTS = ('hallem_shared', 'hallem_shared_ms', 'hallem_shared_1-8ol', 'hallem_shared_paa', 'hallem_shared_B-myr',
             'lab_scaled_ms')
SEARCHED_INPUT = f'results/{EXPERIMENT}/inputs/searched_weak.csv'
S1_INHADD = (100.0, 314.0, 1000.0, 3141.0)
S3_POWERS = (0.5, 0.75, 1.33, 2.0)
S4_CAPS = (100.0, 150.0, 250.0)
S5_EXPONENTS = (0.5, 1.0)
# E18's two strongest settings by p x tau_r (sustained output saturating at 1 / (p tau_r) = 10 and 33 Hz)
S6_DEPLETION = ((0.1, 1.0), (0.03, 1.0))
L_POWERS = (1.0, 2.0)
PERM_SEED = 1

ESTER = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL = ('1-8ol', '1-5ol', '1-6ol', '6al')
# the entry's three kinds of ester pairs and the alcohol group, in canonical (sorted) order
PAIR_KINDS = {k: tuple(tuple(sorted(p)) for p in v) for k, v in {
    'ms': tuple(('ms', o) for o in ('2h', 'IaA', 'pa', 'eb', 'ep')),
    'nearest': (('2h', 'IaA'), ('2h', 'pa'), ('IaA', 'pa'), ('eb', 'ep')),
    'block': tuple((a, b) for a in ('2h', 'IaA', 'pa') for b in ('eb', 'ep')),
    'alcohol': tuple((a, b) for i, a in enumerate(ALCOHOL) for b in ALCOHOL[i + 1:]),
}.items()}


# ---------------------------------------------------------------------------
# input transforms and weight factors (no simulation)

def orn_power(orn: pd.DataFrame, power: float) -> pd.DataFrame:
    """S3: each odor's positive ORN deltas to `power`, rescaled so that their sum over glomeruli is the odor's own sum
    of positive deltas; zero and negative deltas unchanged (the lab's matrix has 19 negative deltas). Power 1 gives the
    input back bit for bit."""
    out = orn.astype(float).copy()
    x = out.to_numpy()
    pos = x > 0
    y = np.where(pos, np.power(np.where(pos, x, 0.0), float(power)), x)
    s0 = np.where(pos, x, 0.0).sum(axis=0)
    s1 = np.where(pos, y, 0.0).sum(axis=0)
    scale = np.divide(s0, s1, out=np.ones_like(s0), where=s1 > 0)
    out[:] = np.where(pos, y * scale[None, :], x)
    return out


def glomerulus_claw_counts(W: pd.DataFrame) -> pd.Series:
    """Each glomerulus's number of claws in the claw x bouton wiring `W` (claws with a nonzero weight from one of its
    boutons)."""
    g = W.columns.get_level_values('glomerulus')
    return pd.Series((W.to_numpy() != 0).sum(axis=0), index=g).groupby(level=0, sort=False).sum()


def claw_count_factors(W: pd.DataFrame, driven: Sequence[str], exponent: float, permute_seed: Optional[int] = None
                       ) -> pd.Series:
    """S5: per glomerulus, (median claw count / its claw count) ** exponent, median and factors over the glomeruli of
    `driven` (those with ORN input for the panel; the others, which carry only spontaneous PN input, keep 1, as E20's
    drive normalization does). With `permute_seed`, the driven glomeruli's factors are permuted among them (the control).
    Exponent 0 gives exactly 1 everywhere."""
    n = glomerulus_claw_counts(W)
    f = pd.Series(1.0, index=n.index)
    if float(exponent) == 0.0:
        return f
    d = n[[g for g in n.index if g in set(driven) and n[g] > 0]].astype(float)
    fd = (float(d.median()) / d) ** float(exponent)
    if permute_seed is not None:
        fd = pd.Series(np.random.default_rng(permute_seed).permutation(fd.to_numpy()), index=fd.index)
    f.loc[fd.index] = fd.to_numpy()
    return f


def weighted_wiring(W: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    """`W` with each bouton's column (PN->claw weights) times its glomerulus's factor (1 where `factors` has none)."""
    col = factors.reindex(W.columns.get_level_values('glomerulus')).fillna(1.0).to_numpy(dtype=float)
    return W * col[None, :]


def odor_weight_factors(activity: pd.DataFrame, power: float, permute_seed: Optional[int] = None) -> pd.DataFrame:
    """Per-odor multiplicative factors for L1 / L2: (activity / its mean over units) ** power per odor, renormalized to
    mean 1 per odor; power 0 gives exactly 1; with `permute_seed` each odor's factors are permuted across units (the
    odor's set of factors kept).

    Args:
        activity: units (boutons or claws) x odors, non-negative (PN rates, claw inputs).
        power: the exponent; 0 for the identity.
        permute_seed: one seed for all odors (one generator, odors in column order), or None.
    """
    a = activity.to_numpy(dtype=float)
    if float(power) == 0.0:
        f = np.ones_like(a)
    else:
        if (a < 0).any() or not np.isfinite(a).all():
            raise ValueError('activity must be finite and non-negative')
        m = a.mean(axis=0)
        if (m <= 0).any():
            raise ValueError('an odor has no activity in any unit')
        f = (a / m[None, :]) ** float(power)
        f = f / f.mean(axis=0, keepdims=True)
    if permute_seed is not None:
        rng = np.random.default_rng(permute_seed)
        f = np.stack([f[rng.permutation(f.shape[0]), j] for j in range(f.shape[1])], axis=1)
    return pd.DataFrame(f, index=activity.index, columns=activity.columns)


# ---------------------------------------------------------------------------
# C2 and its fixed-scale runner

_PARTS: Dict[str, object] = {}


def _n(responses: pd.DataFrame) -> int:
    return int((responses.to_numpy(dtype=float) > 0.5).sum())


def c2_parts() -> dict:
    """C2's pieces, computed once per process: 'orn' (the lab's 16-odor input), 'p' (C2's fitted absolute parameters,
    E72's `fitted_params`), 'init' (the bouton base's parameters, E3's, at which breadth runs are simulated), 'wnorm'
    (normalized APL weights), 'W' (claw x bouton wiring) and 'breadth' (E10_joint_fit.bouton_base, real wiring, no
    per-claw threshold), 'saved' (C2's saved {'on', 'off'} responses and spike counts), 'target_on' / 'target_off' (its
    response counts, two or more spikes), 'size' (KCs x odors), 'driven' (glomeruli with ORN input, in wiring order)."""
    if not _PARTS:
        import E10_joint_fit as J
        import E72_wiring_population as E72
        from E12_breadth_apl import e3_bouton_params
        orn = H.apl16_orn_deltas()
        _, wnorm, W, breadth = J.bouton_base(orn, H.outputs_dir(EXPERIMENT), None)
        saved = E72.load_fit(*C2)
        if saved is None:
            raise RuntimeError(f"C2's saved outputs ({C2}) not found")
        gl = list(dict.fromkeys(W.columns.get_level_values('glomerulus')))
        _PARTS.update(orn=orn, p=E72.fitted_params(*C2), init=e3_bouton_params(), wnorm=wnorm, W=W, breadth=breadth,
                      saved=saved, target_on=_n(saved['on'][0]), target_off=_n(saved['off'][0]),
                      size=int(saved['on'][0].size), driven=[g for g in gl if g in set(orn.index)])
    return _PARTS


def _scalar_kws(kws: dict) -> dict:
    return {k: v for k, v in sorted(kws.items()) if not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))}


def frame_sha256(df: pd.DataFrame) -> str:
    """sha256 of a frame's values (little-endian float64) and its index and columns."""
    h = hashlib.sha256(np.ascontiguousarray(df.to_numpy(dtype=float), dtype='<f8').tobytes())
    h.update(repr(list(df.index)).encode())
    h.update(repr(list(df.columns)).encode())
    return h.hexdigest()


def handle_breadth(extra_kws: Optional[dict] = None, *, orn: Optional[pd.DataFrame] = None,
                   wPNKC: Optional[pd.DataFrame] = None) -> pd.Series:
    """The breadth (each KC's total APL-off spike count, indexed by KC id) of C2's family with a handle, as
    E10_joint_fit.bouton_base computes it for another ORN input or wiring: the bouton base at its own parameters (E3's;
    no KC-KC inhibition), APL off, with the handle's kwargs, ORN input and wiring. With no handle it is C2's own breadth.
    Runs are cached under outputs/E77_input_side_common_error/breadth/ by a hash of their inputs, the base's parameters
    and the olfsysm build's source hash."""
    import E10_joint_fit as J
    from al_analysis.mb_model import fit_mb_model, olfsysm_build_info
    P = c2_parts()
    if not extra_kws and orn is None and wPNKC is None:
        return P['breadth']
    kws = dict(J.base_kws(None), **(extra_kws or {}))
    orn = P['orn'] if orn is None else orn
    h = hashlib.sha256(frame_sha256(orn).encode())
    h.update(json.dumps(_scalar_kws(kws), default=str).encode())
    h.update(json.dumps(P['init'], sort_keys=True).encode())
    h.update(str(olfsysm_build_info().get('source_sha256')).encode())
    if wPNKC is not None:
        h.update(frame_sha256(wPNKC).encode())
        kws['_wPNKC'] = wPNKC
    d = H.outputs_dir(EXPERIMENT) / 'breadth'
    d.mkdir(exist_ok=True)
    path = d / f'breadth_{h.hexdigest()[:16]}_acute_off_spike_counts.parquet'
    if path.exists():
        off_spk = pd.read_parquet(path)
    else:
        init = P['init']
        r_off, off_spk, _, _ = fit_mb_model(orn, fixed_thr=init['fixed_thr'], wAPLKC=0.0, wKCAPL=init['wKCAPL_scale'],
                                            wAPLPN=0.0, wPNAPL=init['wPNAPL_scale'], **kws)
        off_spk.to_parquet(path)
        r_off.to_parquet(path.with_name(path.name.replace('spike_counts', 'responses')))
        path.with_name(path.name.replace('_acute_off_spike_counts.parquet', '.json')).write_text(json.dumps(
            dict(kws=_scalar_kws(kws), orn_sha256=frame_sha256(orn), wPNKC_sha256=None if wPNKC is None else frame_sha256(wPNKC),
                 base_params=init), indent=1, default=str) + '\n')
    return pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(), index=off_spk.index.get_level_values(H.KC_ID_LEVEL))


def build_c2_simulator(extra_kws: Optional[dict] = None, *, orn: Optional[pd.DataFrame] = None,
                       wPNKC: Optional[pd.DataFrame] = None, breadth: Optional[pd.Series] = None,
                       params: Optional[Dict[str, float]] = None) -> Tuple[Callable, dict]:
    """C2's fixed-parameter simulator, as E75's ProfileModel builds it (the bouton base's kwargs, the connectome APL
    weights normalized, breadth exponent 2 on the APL->KC weights, KC-KC inhibition at C2's scale with E10's tau), with
    a handle.

    Args:
        extra_kws: further `make_fixed_param_kc_simulator` / `fit_mb_model` kwargs, e.g. dict(claw_thr=100.0),
            dict(pn_inhadd=1000.0, pn_inhsc=...), dict(claw_sat=150.0, claw_sat_hard=False).
        orn: another ORN input (the lab's 16-odor layout, e.g. E10_joint_fit.read_orn_input's), default the lab's.
        wPNKC: another claw x bouton wiring (C2's index and columns), default the hemibrain's.
        breadth: the breadth behind the APL->KC weights; default `handle_breadth` of the same handle (C2's own when
            there is none). For a per-claw threshold, E10_joint_fit's fits with `--claw-thr T` take the breadth of
            another base instead (E19's thr<T> base row and hal's E16 run): pass
            `E10_joint_fit.bouton_base(H.apl16_orn_deltas(), out_dir, T)[3]` here to match them.
        params: the starting parameters (default C2's fitted ones; `fixed_thr`, the four APL scales, `wKCKC_scale`).

    Returns:
        (simulate, info): `simulate` as make_fixed_param_kc_simulator's; info has 'params', 'kws' (scalar kwargs),
        'breadth', 'wAPLKC' (the breadth-weighted APL->KC weights), 'orn', 'wPNKC' (the wiring used) and 'cells' (the
        simulator's closure cells by name: 'wAPLKC_unscaled' / 'wAPLPN_unscaled' for in-place weight changes as E75
        does, 'mp', 'rv', 'odor_order').
    """
    import E10_joint_fit as J
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    P = c2_parts()
    p = dict(P['p'] if params is None else params)
    if breadth is None:
        breadth = handle_breadth(extra_kws, orn=orn, wPNKC=wPNKC)
    wAPLKC = J.breadth_weights(P['wnorm']['wAPLKC'], breadth, BREADTH_K)
    kws = dict(J.base_kws(None), **(extra_kws or {}))
    scalar = _scalar_kws(kws)
    if wPNKC is not None:
        kws['_wPNKC'] = wPNKC
    kws.update(_wAPLKC=wAPLKC, _wKCAPL=P['wnorm']['wKCAPL'], _wAPLPN=P['wnorm']['wAPLPN'], _wPNAPL=P['wnorm']['wPNAPL'],
               kc_kc_inh=p['wKCKC_scale'], kc_kc_tau=J.TAU)
    o = P['orn'] if orn is None else orn
    sim, _ = make_fixed_param_kc_simulator(o, initial_params=p, **kws)
    cells = {k: c for k, c in zip(sim.__code__.co_freevars, sim.__closure__)}
    if not np.array_equal(cells['wAPLKC_unscaled'].cell_contents.ravel(), wAPLKC.to_numpy()):
        raise RuntimeError("the simulator's unscaled APL->KC weights are not the ones it was built with")
    return sim, dict(params=p, kws=scalar, breadth=breadth, wAPLKC=wAPLKC, orn=o, wPNKC=P['W'] if wPNKC is None else wPNKC,
                     cells=cells)


def hold_rates(build: Callable[[], Union[Callable, Tuple[Callable, dict]]], *, target_off: Optional[int] = None,
               target_on: Optional[int] = None, params: Optional[Dict[str, float]] = None, breadth_from: str = 'base',
               label: str = '', thr_bracket: Tuple[float, float] = THR_BRACKET,
               apl_bracket: Tuple[float, float] = APL_BRACKET) -> dict:
    """The fixed-scale runner with both response rates held, on the simulator `build()` returns (e.g.
    `lambda: build_c2_simulator(dict(claw_thr=100.0))`; a bare simulate function or build_c2_simulator's
    (simulate, info) pair). First the somatic threshold (fixed_thr) is bisected with APL off (APL->KC and APL->bouton
    scales 0) to `target_off` responses, then the APL->KC scale with KC->APL and bouton->APL tied (TIED_ON, as E75's
    ProfileModel.run_on) to `target_on` responses at that threshold; KC-KC and APL->bouton scales stay at `params`'.
    Both bisections are E69's `bisect_factor` (log scale, from factor 1; the number of responses falls with the factor).

    Args:
        build: builds the simulator (called once; the simulator is freed on return).
        target_off, target_on: response counts (two or more spikes) to hold; default C2's (APL off, APL on: 1852).
        params: the parameters the factors multiply; default C2's fitted ones.
        breadth_from: 'base' keeps the simulator's breadth weights (build_c2_simulator computes them as
            E10_joint_fit.bouton_base does for another input or wiring, `handle_breadth`); 'held' replaces them by the
            breadth of the threshold-held APL-off run (each KC's total spike count there, KC-KC inhibition included),
            written into the simulator's APL->KC weights in place before the APL scale is bisected (needs `build` to
            return build_c2_simulator's pair).
        label: printed with each bisection step.
        thr_bracket, apl_bracket: the factors' brackets.

    Returns:
        dict with 'params' (absolute, APL on), 'params_off' (the APL-off run's), 'thr_factor', 'apl_factor', 'n_on',
        'n_off', 'target_on', 'target_off', 'at_bound_thr', 'at_bound_apl', 'n_sims', 'history' (both bisections),
        'on' / 'off' ((responses, spike counts)), 'breadth' (the held run's, for 'held', else None), 'kws' (the
        simulator's scalar kwargs, if build returned info), 'breadth_sha256', 'seconds'.
    """
    import E69_kc_firing_ceiling as E69
    import E10_joint_fit as J
    if breadth_from not in ('base', 'held'):
        raise ValueError(f'{breadth_from=}')
    P = c2_parts()
    t0 = time.time()
    p = dict(P['p'] if params is None else params)
    target_off = P['target_off'] if target_off is None else int(target_off)
    target_on = P['target_on'] if target_on is None else int(target_on)
    built = build()
    sim, info = built if isinstance(built, tuple) else (built, None)
    size = P['size']
    off_q = dict(wAPLKC_scale=0.0, wAPLPN_scale=0.0)
    g, (r_off, s_off), h_off, b_off = E69.bisect_factor(lambda f: sim(dict(p, fixed_thr=p['fixed_thr'] * f, **off_q)),
                                                         target_off, size, thr_bracket, THR_STEP, f'{label} threshold')
    thr = p['fixed_thr'] * g
    breadth = None
    if breadth_from == 'held':
        if info is None:
            raise ValueError("breadth_from='held' needs build() to return build_c2_simulator's (simulate, info)")
        breadth = pd.Series(s_off.astype(float).sum(axis='columns').to_numpy(), index=s_off.index.get_level_values(H.KC_ID_LEVEL))
        w = J.breadth_weights(P['wnorm']['wAPLKC'], breadth, BREADTH_K)
        cell = info['cells']['wAPLKC_unscaled'].cell_contents
        cell[...] = w.to_numpy().reshape(cell.shape)
    x, (r_on, s_on), h_on, b_on = E69.bisect_factor(
        lambda f: sim(dict(p, fixed_thr=thr, **{k: p[k] * f for k in TIED_ON})), target_on, size, apl_bracket, APL_STEP,
        f'{label} APL scale')
    on_params = dict(p, fixed_thr=thr, **{k: p[k] * x for k in TIED_ON})
    bsha = None
    if info is not None:
        b = breadth if breadth is not None else info['breadth']
        bsha = hashlib.sha256(np.ascontiguousarray(b.to_numpy(dtype=float), dtype='<f8').tobytes()).hexdigest()
    res = dict(params=on_params, params_off=dict(p, fixed_thr=thr, **off_q), thr_factor=float(g), apl_factor=float(x),
               n_on=_n(r_on), n_off=_n(r_off), target_on=target_on, target_off=target_off, at_bound_thr=bool(b_off),
               at_bound_apl=bool(b_on), n_sims=len(h_off) + len(h_on), history=h_off + h_on, on=(r_on, s_on),
               off=(r_off, s_off), breadth=breadth, kws=None if info is None else info['kws'], breadth_sha256=bsha,
               seconds=time.time() - t0)
    del sim, info, built
    gc.collect()
    return res


class PerOdorRunner:
    """C2's APL-on run with per-odor APL weights (the L1 / L2 emulations): each odor simulated on its own (olfsysm's
    `mp.sim_only`; odors are independent at fixed scales, and at factors 1 the 16 one-odor runs reproduce C2's APL-on
    run exactly, which `check` verifies), its APL->bouton weights times `pn_f[:, j]` and APL->KC weights times
    `kc_f[:, j]`, in place in the simulator's own unscaled weight arrays (E75's pattern), restored after every run."""

    def __init__(self, sim: Callable, info: dict):
        cells = info['cells']
        self.sim, self.p, self.W = sim, dict(info['params']), info['wPNKC']
        self.mp, self.rv = cells['mp'].cell_contents, cells['rv'].cell_contents
        self.order = np.asarray(cells['odor_order'].cell_contents, dtype=int)
        self._kc = cells['wAPLKC_unscaled'].cell_contents
        self._pn = cells['wAPLPN_unscaled'].cell_contents
        self._kc0, self._pn0 = self._kc.copy(), self._pn.copy()

    def run_on(self, x: float, kc_f: Optional[np.ndarray] = None, pn_f: Optional[np.ndarray] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """APL on at the APL scales TIED_ON times `x`, odor j (output column j) with the factors' column j."""
        q = dict(self.p, **{k: self.p[k] * x for k in TIED_ON})
        R = S = None
        try:
            for j in range(len(self.order)):
                self._kc[...] = self._kc0 if kc_f is None else self._kc0 * np.asarray(kc_f)[:, j].reshape(self._kc0.shape)
                self._pn[...] = self._pn0 if pn_f is None else self._pn0 * np.asarray(pn_f)[:, j].reshape(self._pn0.shape)
                self.mp.sim_only = [int(self.order[j])]
                r, s = self.sim(q)
                if R is None:
                    R, S = r.copy(), s.copy()
                else:
                    R.iloc[:, j] = r.iloc[:, j].to_numpy()
                    S.iloc[:, j] = s.iloc[:, j].to_numpy()
        finally:
            self._kc[...] = self._kc0
            self._pn[...] = self._pn0
            self.mp.sim_only = []
        return R, S

    def check(self, saved_on: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """At factors 1 the one-odor runs must reproduce `saved_on` (C2's APL-on outputs) exactly."""
        r, s = self.run_on(1.0)
        n = int((s.to_numpy(dtype=float) != saved_on[1].to_numpy(dtype=float)).sum())
        nr = int((r.to_numpy(dtype=float) != saved_on[0].to_numpy(dtype=float)).sum())
        if n or nr or not s.index.equals(saved_on[1].index):
            raise RuntimeError(f"the one-odor runs differ from C2's APL-on run in {n} counts and {nr} responses")

    def pn_rates(self) -> pd.DataFrame:
        """Glomerulus x odor PN rates of C2's input (window PN_WINDOW mean of olfsysm's pn_sims; rows in the wiring's
        glomerulus order, columns the outputs' odors by position)."""
        from al_analysis.mb_model import get_sim_var, get_time_index
        t = np.asarray(get_time_index(self.mp, self.rv))
        win = (t >= PN_WINDOW[0]) & (t <= PN_WINDOW[1])
        pn = get_sim_var(self.rv.pn, 'pn_sims')
        gl = list(dict.fromkeys(self.W.columns.get_level_values('glomerulus')))
        if np.asarray(pn[0]).shape[0] != len(gl):
            raise RuntimeError(f'{np.asarray(pn[0]).shape[0]} PN rows for {len(gl)} glomeruli in the wiring')
        cols = [np.asarray(pn[int(self.order[j])])[:, win].mean(axis=1) for j in range(len(self.order))]
        return pd.DataFrame(np.stack(cols, axis=1), index=pd.Index(gl, name='glomerulus'))

    def activity(self, site: str) -> pd.DataFrame:
        """Units x odors activity: 'bouton', each bouton's PN rate (its glomerulus's; before APL's inhibition), in the
        APL->bouton weights' order; 'claw', each claw's input (its PN->claw weights times its boutons' PN rates), in the
        APL->KC weights' order."""
        rates = self.pn_rates()
        b = rates.reindex(self.W.columns.get_level_values('glomerulus')).to_numpy()
        if site == 'bouton':
            return pd.DataFrame(b, index=self.W.columns)
        if site == 'claw':
            return pd.DataFrame(self.W.to_numpy(dtype=float) @ b, index=self.W.index)
        raise ValueError(site)


def _pool_worker(conn, parts: dict, js: List[int]) -> None:
    """An EmulationPool worker: C2's simulator on the ORN columns `js` of the lab's input, then PerOdorRunner.run_on
    for each (x, kc_f, pn_f) received (factors for its odors only), until None."""
    try:
        _PARTS.update(parts)
        sim, info = build_c2_simulator(orn=parts['orn'].iloc[:, js], breadth=parts['breadth'])
        runner = PerOdorRunner(sim, info)
        conn.send(('ready', [str(c) for c in info['cells']['odor_index'].cell_contents]))
    except Exception as e:
        conn.send(('error', repr(e)))
        return
    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            r, s = runner.run_on(*msg)
            conn.send(('ok', (r.to_numpy(), s.to_numpy())))
        except Exception as e:
            conn.send(('error', repr(e)))


class EmulationPool:
    """PerOdorRunner.run_on spread over worker processes (spawned, one OpenMP thread each), each holding C2's simulator
    built on its share of the odors' ORN columns (olfsysm simulates odors independently; `check` verifies that at
    factors 1 the pool reproduces C2's APL-on run exactly). With `n_workers` 0 or 1, runs in this process on `sim`."""

    def __init__(self, n_workers: int, sim: Callable, info: dict):
        import multiprocessing
        import os
        P = c2_parts()
        self.template = P['saved']['on']
        n_odors = self.template[0].shape[1]
        self.local = PerOdorRunner(sim, info) if n_workers <= 1 else None
        self.groups = [list(map(int, g)) for g in np.array_split(np.arange(n_odors), max(1, min(n_workers, n_odors)))]
        self.procs, self.conns = [], []
        if self.local is not None:
            return
        parts = {k: P[k] for k in ('orn', 'p', 'init', 'wnorm', 'W', 'breadth', 'saved', 'target_on', 'target_off', 'size', 'driven')}
        ctx = multiprocessing.get_context('spawn')
        keep = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT')}
        os.environ['OMP_NUM_THREADS'] = os.environ['OMP_THREAD_LIMIT'] = '1'
        try:
            for js in self.groups:
                a, b = ctx.Pipe()
                p = ctx.Process(target=_pool_worker, args=(b, parts, js), daemon=True)
                p.start()
                self.procs.append(p)
                self.conns.append(a)
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        cols = [str(c) for c in self.template[0].columns]
        for conn, js in zip(self.conns, self.groups):
            status, v = conn.recv()
            if status != 'ready':
                self.close()
                raise RuntimeError(f'an emulation worker failed to start: {v}')
            if v != [cols[j] for j in js]:
                self.close()
                raise RuntimeError(f'an emulation worker has odors {v}, expected {[cols[j] for j in js]}')
        print(f'emulation pool: {len(self.procs)} workers, odors {self.groups}', flush=True)

    def run_on(self, x: float, kc_f: Optional[np.ndarray] = None, pn_f: Optional[np.ndarray] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.local is not None:
            return self.local.run_on(x, kc_f, pn_f)
        for conn, js in zip(self.conns, self.groups):
            conn.send((float(x), None if kc_f is None else np.asarray(kc_f)[:, js], None if pn_f is None else np.asarray(pn_f)[:, js]))
        R, S = self.template[0].copy(), self.template[1].copy()
        for conn, js in zip(self.conns, self.groups):
            status, v = conn.recv()
            if status != 'ok':
                raise RuntimeError(f'an emulation worker failed: {v}')
            R.iloc[:, js] = v[0]
            S.iloc[:, js] = v[1]
        return R, S

    def check(self) -> None:
        """At factors 1 the one-odor runs must reproduce C2's saved APL-on run exactly."""
        r, s = self.run_on(1.0)
        saved = self.template
        n = int((s.to_numpy(dtype=float) != saved[1].to_numpy(dtype=float)).sum())
        nr = int((r.to_numpy(dtype=float) != saved[0].to_numpy(dtype=float)).sum())
        if n or nr:
            raise RuntimeError(f"the one-odor runs differ from C2's APL-on run in {n} counts and {nr} responses")

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        self.procs, self.conns = [], []


def default_workers() -> int:
    """Emulation workers: half the OpenMP thread limit (or of the CPUs), at most 8 (each holds a C2 simulator)."""
    import os
    lim = os.environ.get('OMP_THREAD_LIMIT') or os.environ.get('OMP_NUM_THREADS') or os.cpu_count() or 2
    return int(max(1, min(8, int(lim) // 2)))


# ---------------------------------------------------------------------------
# the screen's points

def inhsc_for(inhadd: float) -> float:
    """pn_inhsc raised with pn_inhadd as E3's part A and E32 did (the PN gain at a typical inhibition kept)."""
    from E3_drive_dependent_gain import PN_INHADD_DEFAULT, PN_INHSC_DEFAULT, TYPICAL_INH
    return PN_INHSC_DEFAULT * (float(inhadd) + TYPICAL_INH) / (PN_INHADD_DEFAULT + TYPICAL_INH)


def point_specs() -> List[dict]:
    """The screen's points: S1 (pn_inhadd 100, 314, 1000, 3141, pn_inhsc raised as E3 / E32), S2 (Hallem's input for the
    four odors together and one at a time, lab_scaled_ms, the searched input), S3 (each odor's positive ORN deltas to the
    power 0.5, 0.75, 1.33, 2, rescaled to the odor's sum), S4 (a per-claw tanh cap at 100, 150, 250 Hz, no per-claw
    threshold), S5 (glomerulus weights x (median claw count / its claw count)^0.5 and ^1, one permuted control; E20's
    drive normalization at 0.5 and 1), S6 (E18's bouton depletion at its two strongest settings, if its kwargs run on
    C2's simulator), L1 / L2 (APL->bouton / APL->KC weights x the bouton's / claw's own input in the odor, over its mean,
    to the power 1 and 2, mean 1; the better power's factors permuted within each odor as control).

    Each family starts with its identity point. A spec has point_id, handle, setting, value (the setting's number),
    kind ('static' or 'emulation'), identity, and: kws (extra simulator kwargs), orn (None, ('file', path relative to
    the shared directory) or ('power', e)), wiring (None, dict(kind='claws', exponent, permute_seed) or
    dict(kind='e20', exponent)), site / power / permute_seed (emulations), candidates (the points whose better value,
    by the APL-effect loss under the calibrated readout, a control takes: value 'better'), optional (skipped while its
    input file is absent). S6 is last: a C++ error inside olfsysm's parallel region would end the process."""
    from E3_drive_dependent_gain import PN_INHADD_DEFAULT, PN_INHSC_DEFAULT
    pts: List[dict] = []

    def add(pid, handle, setting, value, kind, **kw):
        pts.append(dict(point_id=pid, handle=handle, setting=setting, value=value, kind=kind, identity=setting == 'identity',
                        kws=kw.pop('kws', {}), orn=kw.pop('orn', None), wiring=kw.pop('wiring', None), **kw))

    add('S1_identity', 'S1', 'identity', PN_INHADD_DEFAULT, 'static', kws=dict(pn_inhadd=PN_INHADD_DEFAULT, pn_inhsc=PN_INHSC_DEFAULT))
    for a in S1_INHADD:
        add(f'S1_inhadd{a:g}', 'S1', f'inhadd{a:g}', a, 'static', kws=dict(pn_inhadd=a, pn_inhsc=inhsc_for(a)))
    add('S2_identity', 'S2', 'identity', np.nan, 'static', orn=('file', f'{E38_INPUTS}/lab.csv'))
    for name in S2_INPUTS:
        add(f'S2_{name}', 'S2', name, np.nan, 'static', orn=('file', f'{E38_INPUTS}/{name}.csv'))
    add('S2_searched_weak', 'S2', 'searched_weak', np.nan, 'static', orn=('file', SEARCHED_INPUT), optional=True)
    add('S3_identity', 'S3', 'identity', 1.0, 'static', orn=('power', 1.0))
    for e in S3_POWERS:
        add(f'S3_pow{e:g}', 'S3', f'pow{e:g}', e, 'static', orn=('power', e))
    for c in S4_CAPS:
        add(f'S4_tanh{c:g}', 'S4', f'tanh{c:g}', c, 'static', kws=dict(claw_sat=c, claw_sat_hard=False))
    add('S5_identity', 'S5', 'identity', 0.0, 'static', wiring=dict(kind='claws', exponent=0.0))
    for e in S5_EXPONENTS:
        add(f'S5_claws_e{e:g}', 'S5', f'claws_e{e:g}', e, 'static', wiring=dict(kind='claws', exponent=e))
    add('S5_claws_perm', 'S5', 'claws_perm', 'better', 'static', wiring=dict(kind='claws', exponent='better', permute_seed=PERM_SEED),
        candidates=tuple(f'S5_claws_e{e:g}' for e in S5_EXPONENTS))
    for a in S5_EXPONENTS:
        add(f'S5_e20_a{a:g}', 'S5', f'e20_a{a:g}', a, 'static', wiring=dict(kind='e20', exponent=a))
    add('L_identity', 'L', 'identity', 0.0, 'emulation', site='bouton+claw', power=0.0)
    for site, h in (('bouton', 'L1'), ('claw', 'L2')):
        for q in L_POWERS:
            add(f'{h}_p{q:g}', h, f'p{q:g}', q, 'emulation', site=site, power=q)
        add(f'{h}_perm', h, 'perm', 'better', 'emulation', site=site, power='better', permute_seed=PERM_SEED,
            candidates=tuple(f'{h}_p{q:g}' for q in L_POWERS))
    for p, tau in S6_DEPLETION:
        add(f'S6_p{p:g}_tau{tau:g}', 'S6', f'p{p:g}_tau{tau:g}', p, 'static', kws=dict(bouton_ves_p=p, bouton_tau_r=tau))
    return pts


def orn_for(spec) -> Optional[pd.DataFrame]:
    """A point's ORN input (None: C2's own, the lab's)."""
    import E10_joint_fit as J
    if spec is None:
        return None
    kind, v = spec
    if kind == 'file':
        return J.read_orn_input(v, c2_parts()['orn'])
    if kind == 'power':
        return orn_power(c2_parts()['orn'], float(v))
    raise ValueError(spec)


_E20_PEAKS: Dict[str, pd.Series] = {}


def wiring_for(spec: Optional[dict]) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """A point's wiring (None: C2's own) and its record (the factor per glomerulus)."""
    if spec is None:
        return None, None
    P = c2_parts()
    if spec['kind'] == 'claws':
        f = claw_count_factors(P['W'], P['driven'], float(spec['exponent']), spec.get('permute_seed'))
    elif spec['kind'] == 'e20':
        from E20_glomerulus_gain import gained_wPNKC, glomerulus_peaks
        if 'peaks' not in _E20_PEAKS:
            d = H.outputs_dir(EXPERIMENT) / 'e20'
            d.mkdir(exist_ok=True)
            _E20_PEAKS['peaks'] = glomerulus_peaks(P['orn'], d)[1]
        W, g = gained_wPNKC(P['W'], _E20_PEAKS['peaks'], float(spec['exponent']), P['driven'])
        return W, dict(spec, factors={str(k): float(v) for k, v in g.items()}, wPNKC_sha256=frame_sha256(W))
    else:
        raise ValueError(spec)
    W = weighted_wiring(P['W'], f)
    return W, dict(spec, factors={str(k): float(v) for k, v in f.items()}, wPNKC_sha256=frame_sha256(W))


# ---------------------------------------------------------------------------
# scoring

def _canon_series(s: pd.Series) -> pd.Series:
    import E75_glomerulus_apl_profile as E75
    s = s.copy()
    s.index = E75._canon(s.index)
    return s.sort_index()


class Scorer:
    """The per-point statistics against C2 and the data: E75's `score_point` (the APL-effect loss with its parts on raw
    counts, under the calibrated readout with K re-calibrated and binarized; ρ on the per-pair APL effect; the projection
    on the common error; the class split), plus the kinds of ester pairs and the alcohol group, the transfer reduction
    of the common error against all eight recordings with APL output blocked (TNT) and against the six typical ones, the
    projection against the six, the per-odor response fractions, the four realism bands under both rules (as E75's
    `save_points`) and the class at 4-7 odors per KC."""

    def __init__(self, base: Dict[str, tuple]):
        import E72_wiring_population as E72
        import E74_apl_effect_by_odors_per_kc as E74
        import E75_glomerulus_apl_profile as E75
        self.base = base
        self.base_eff = {ro: H.apl_effect(*[E72.distances(base, ro)[c] for c in ('on', 'off')]) for ro in ('raw', 'cal')}
        self.base_eff_c = {ro: _canon_series(v) for ro, v in self.base_eff.items()}
        self.c = {ro: _canon_series(E75.common_error(ro)) for ro in ('raw', 'cal')}
        self.c6 = {ro: E75.six_target(E75.population_effect(ro)) for ro in ('raw', 'cal')}
        ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
        tnt = H.yang_recording_rdms_values('TNT', 'graded')
        keep = [r for r in tnt.index if (str(r[0]), int(r[1]), str(r[2])) not in E75.STRONG_TNT]
        self.d6 = _canon_series(ctrl.mean() - tnt.loc[keep].mean())
        self.class_data = E74.data_effect_array()

    def score(self, o: Dict[str, tuple]) -> Tuple[dict, pd.DataFrame]:
        import apl_effect_scores as A
        import E72_wiring_population as E72
        import E74_apl_effect_by_odors_per_kc as E74
        import E75_glomerulus_apl_profile as E75
        row, _ = E75.score_point(o, self.base_eff)
        # E67's full_row's realism keys are against the KC data in Yang's z: replaced below by both rules' (E75's save_points)
        for k in ('rate_rho_on', 'rate_rho_off', 'opk_on_dist', 'opk_off_dist', 'realism_flag', 'opk_off_0', 'opk_off_8plus'):
            row.pop(k, None)
        dist = {ro: E72.distances(o, ro) for ro in ('raw', 'cal', 'binary')}
        for ro in ('raw', 'cal'):
            err = _canon_series(E72.pattern_error(dist[ro], ro))
            for kind, pairs in PAIR_KINDS.items():
                row[f'{ro}_err_{kind}'] = float(err.reindex(list(pairs)).mean())
            eff = _canon_series(H.apl_effect(dist[ro]['on'], dist[ro]['off']))
            ch = eff - self.base_eff_c[ro].reindex(eff.index)
            row[f'{ro}_transfer_all'] = E75.reduction(self.c[ro], ch.reindex(self.c[ro].index))
            row[f'{ro}_transfer_six'] = E75.reduction(self.c6[ro], ch.reindex(self.c6[ro].index))
            e = eff.reindex(self.c6[ro].index)
            d = self.d6.reindex(self.c6[ro].index)
            e6 = (e - e.mean()) - (d - d.mean())
            c6 = self.c6[ro].to_numpy(dtype=float)
            row[f'{ro}_proj_common_six'] = float(e6.to_numpy(dtype=float) @ c6 / (c6 @ c6))
        rate_on = float(o['on'][0].to_numpy(dtype=float).mean())
        rate_off = float(o['off'][0].to_numpy(dtype=float).mean())
        row.update(rate_on=rate_on, rate_off=rate_off, ratio=rate_off / rate_on)
        for c in ('on', 'off'):
            d = A.odors_per_kc(o[c][0])
            row[f'opk_{c}_tv'] = float(0.5 * np.abs(d - A.opk_data(c, 'noise')).sum())
            row[f'opk_{c}_tv_zcrit'] = float(0.5 * np.abs(d - A.opk_data(c, 'z')).sum())
            per_odor = H.abbrev_columns(o[c][0].astype(float) > 0.5).mean().reindex(H.APL16_ODORS)
            row[f'rate_rho_{c}'] = H.spearman(per_odor, A.frac_data(c, 'noise'))
            row[f'rate_rho_{c}_zcrit'] = H.spearman(per_odor, A.frac_data(c, 'z'))
            row[f'opk_{c}_0'], row[f'opk_{c}_8plus'] = float(d[0]), float(d[-1])
            row[f'max_count_{c}'] = int(o[c][1].to_numpy(dtype=float).max())
            row.update({f'frac_{c}_{od}': float(v) for od, v in per_odor.items()})
        row['realism_flag'] = A.realism_flag(rate_on, rate_off, row['rate_rho_on'], row['opk_on_tv'], row['opk_off_tv'], units='noise')
        row['realism_flag_zcrit'] = A.realism_flag(rate_on, rate_off, row['rate_rho_on_zcrit'], row['opk_on_tv_zcrit'],
                                                   row['opk_off_tv_zcrit'], units='z')
        on, off = (H.abbrev_columns(o[c][1].astype(float))[E74.ODORS].to_numpy(dtype=float) for c in ('on', 'off'))
        st = E74._variant_stats(E74.read_cal(on, off), self.class_data)
        row.update({'class_' + (k[len('class_'):] if k.startswith('class_') else k): v for k, v in st.items()
                    if '4_7' in k and not k.startswith('eff_')})
        cols = {}
        for ro, d in dist.items():
            on_c, off_c = _canon_series(d['on']), _canon_series(d['off'])
            cols.update({f'{ro}_on': on_c, f'{ro}_off': off_c, f'{ro}_effect': _canon_series(H.apl_effect(d['on'], d['off']))})
        pairs = pd.DataFrame(cols).rename_axis(['odor_a', 'odor_b']).reset_index()
        return row, pairs


# ---------------------------------------------------------------------------
# the screen

def _resolve(pt: dict, rows: pd.DataFrame) -> Optional[dict]:
    """A control point with its value taken from the better of its candidates (lower cal_loss); None while they are
    not all done."""
    if not pt.get('candidates'):
        return pt
    have = rows[rows.point_id.isin(pt['candidates'])] if len(rows) else rows
    if len(have) < len(pt['candidates']):
        return None
    best = have.loc[have.cal_loss.astype(float).idxmin()]
    v = float(best.value)
    q = dict(pt, value=v, chosen_from=str(best.point_id))
    if q['kind'] == 'emulation':
        q['power'] = v
        q['setting'] = f'perm_p{v:g}'
    else:
        q['wiring'] = dict(q['wiring'], exponent=v)
        q['setting'] = f'claws_perm_e{v:g}'
    return q


def _write_rows(path: Path, rows: pd.DataFrame, new: dict) -> pd.DataFrame:
    rows = pd.concat([rows, pd.DataFrame([new])], ignore_index=True) if len(rows) else pd.DataFrame([new])
    rows.to_csv(path, index=False)
    return rows


def _git_head() -> str:
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()


def write_provenance(out: Path, specs: Sequence[dict]) -> None:
    """screen_provenance.json: the model's provenance (`H.model_provenance`, on the lab's input and C2's wiring), C2's
    parameters, the bouton base's, the targets, the points, the brackets, the hashes of E38's inputs."""
    P = c2_parts()
    inputs = {}
    for name in S2_INPUTS + ('lab',):
        f = H.shared_dir() / E38_INPUTS / f'{name}.csv'
        inputs[name] = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
    f = H.shared_dir() / SEARCHED_INPUT
    inputs['searched_weak'] = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E77_input_screen.py',
                al_analysis_commit=_git_head(), model=H.model_provenance(orn_deltas=P['orn'], _wPNKC=P['W']),
                c2=dict(experiment=C2[0], tag=C2[1], params=P['p']), bouton_base_params=P['init'], breadth_k=BREADTH_K,
                targets=dict(on=P['target_on'], off=P['target_off'], size=P['size']), tied_on=TIED_ON,
                thr_bracket=THR_BRACKET, thr_step=THR_STEP, apl_bracket=APL_BRACKET, apl_step=APL_STEP, pn_window=PN_WINDOW,
                points=list(specs), input_sha256=inputs, approx_conc_note=H.APPROX_CONC_NOTE,
                hallem_note=("S2's Hallem values are at 10^-2, within one log unit of Yang's ms, paa and 1-8ol: approximated "
                             'input'),
                breadth=('handle_breadth: the bouton base (E3 parameters, no KC-KC inhibition) APL off with the handle, '
                         'as E10_joint_fit.bouton_base for another input or wiring'))
    (out / 'screen_provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')


def _save_outputs(tag: str, o: Dict[str, tuple]) -> None:
    d = H.outputs_dir(EXPERIMENT) / 'screen'
    d.mkdir(exist_ok=True)
    for c, cc in (('on', 'on'), ('off', 'acute_off')):
        o[c][0].to_parquet(d / f'{tag}_{cc}_responses.parquet')
        o[c][1].to_parquet(d / f'{tag}_{cc}_spike_counts.parquet')


def _check_identity(pid: str, o: Dict[str, tuple], saved: Dict[str, tuple], factors: Sequence[float]) -> None:
    bad = [f for f in factors if f != 1.0]
    for c in ('on', 'off'):
        n = int((o[c][1].to_numpy(dtype=float) != saved[c][1].to_numpy(dtype=float)).sum())
        if n or not o[c][1].index.equals(saved[c][1].index):
            raise RuntimeError(f"{pid}: the identity setting differs from C2's saved {c} run in {n} counts")
    if bad:
        raise RuntimeError(f'{pid}: the identity setting needed factors {factors}')
    print(f"{pid}: reproduces C2's saved APL-on and APL-off spike counts exactly (threshold and APL factors 1)", flush=True)


def family(spec: dict) -> str:
    """A point's family for its identity check: its handle, 'L' for L1 / L2."""
    return 'L' if spec['handle'] in ('L', 'L1', 'L2') else spec['handle']


class _Screen:
    """The screen's state: C2 rebuilt and checked, the scorer, the rows written so far, the emulation pool."""

    def __init__(self, out: Path, workers: Optional[int]):
        self.P = P = c2_parts()
        self.out, self.workers = out, workers
        self.sp, self.pp, self.jp, self.ep = (out / f for f in ('screen.csv', 'screen_pairs.csv', 'screen_params.json',
                                                                'screen_errors.csv'))
        self.rows = pd.read_csv(self.sp) if self.sp.exists() else pd.DataFrame()
        self.done = set(self.rows.point_id) if len(self.rows) else set()
        self.params = json.loads(self.jp.read_text()) if self.jp.exists() else {}
        self.sim0, self.info0 = build_c2_simulator(breadth=P['breadth'])
        base = {'on': self.sim0(dict(P['p'])), 'off': self.sim0(dict(P['p'], wAPLKC_scale=0.0, wAPLPN_scale=0.0))}
        _check_identity('base (C2 rebuilt)', base, P['saved'], [])
        self.scorer = Scorer(P['saved'])
        self.act = PerOdorRunner(self.sim0, self.info0)
        self.pool: Optional[EmulationPool] = None

    def record(self, pt: dict, o: Dict[str, tuple], res: dict, extra: dict) -> None:
        P = self.P
        t0 = time.time()
        row, pairs = self.scorer.score(o)
        meta = dict(point_id=pt['point_id'], handle=pt['handle'], setting=pt['setting'], value=pt['value'], kind=pt['kind'],
                    identity=bool(pt['identity']), fixed_thr=res['params']['fixed_thr'],
                    **{k: res['params'][k] for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')},
                    thr_factor=res.get('thr_factor', 1.0), apl_factor=res.get('apl_factor', 1.0), n_on=_n(o['on'][0]),
                    n_off=_n(o['off'][0]), target_on=P['target_on'], target_off=P['target_off'],
                    at_bound_thr=res.get('at_bound_thr', False), at_bound_apl=res.get('at_bound_apl', False),
                    n_sims=res.get('n_sims', 0), seconds=res.get('seconds', 0.0) + (time.time() - t0),
                    note=extra.get('note', ''))
        self.rows = _write_rows(self.sp, self.rows, dict(meta, **row))
        pairs.insert(0, 'point_id', pt['point_id'])
        pairs.to_csv(self.pp, index=False, mode='a', header=not self.pp.exists())
        self.params[pt['point_id']] = dict({k: v for k, v in pt.items() if k != 'kws'}, kws=pt.get('kws', {}),
                                           params=res['params'], params_off=res.get('params_off'), **extra,
                                           thr_factor=meta['thr_factor'], apl_factor=meta['apl_factor'], n_on=meta['n_on'],
                                           n_off=meta['n_off'], history=res.get('history', []))
        self.jp.write_text(json.dumps(self.params, indent=1, default=str) + '\n')
        _save_outputs(f"C2_{pt['point_id']}", o)
        self.done.add(pt['point_id'])
        r = self.rows.iloc[-1]
        print(f"{pt['point_id']}: thr x{meta['thr_factor']:.4f}, APL x{meta['apl_factor']:.4f} ({meta['n_off']} / {meta['n_on']} "
              f"responses off / on); readout loss {r.cal_loss:.4f} (mean {r.cal_mean:+.3f}, pattern {r.cal_pattern:.4f}), raw "
              f"{r.raw_loss:.4f}; transfer {r.cal_transfer_all:+.3f} / six {r.cal_transfer_six:+.3f}; projection "
              f"{r.cal_proj_common:.3f}; ester ms / nearest / block {r.cal_err_ms:+.3f} / {r.cal_err_nearest:+.3f} / "
              f"{r.cal_err_block:+.3f}, alcohol {r.cal_err_alcohol:+.3f}; realism: "
              f"{r.realism_flag if isinstance(r.realism_flag, str) and r.realism_flag else 'inside the bands'}",
              flush=True)

    def run_static(self, pt: dict) -> Tuple[Dict[str, tuple], dict, dict]:
        orn = orn_for(pt['orn'])
        W, wrec = wiring_for(pt['wiring'])
        res = hold_rates(lambda: build_c2_simulator(pt['kws'] or None, orn=orn, wPNKC=W), label=pt['point_id'])
        note = ('approximated input (Hallem at 10^-2)' if pt['handle'] == 'S2' and pt['setting'] in S2_INPUTS else
                'oracle input (searched)' if pt['setting'] == 'searched_weak' else '')
        extra = dict(sim_kws=res['kws'], breadth_sha256=res['breadth_sha256'],
                     orn_sha256=None if orn is None else frame_sha256(orn), wiring_record=wrec, note=note)
        return {'on': res['on'], 'off': res['off']}, res, extra

    def run_emulation(self, pt: dict) -> Tuple[Dict[str, tuple], dict, dict]:
        import E69_kc_firing_ceiling as E69
        P = self.P
        if self.pool is None:
            pool = EmulationPool(default_workers() if self.workers is None else int(self.workers), self.sim0, self.info0)
            try:
                pool.check()
            except Exception:
                pool.close()
                raise
            print("one-odor runs at factors 1 reproduce C2's APL-on run exactly", flush=True)
            self.pool = pool
        t0 = time.time()
        kc_f = pn_f = None
        frec = {}
        for site in pt['site'].split('+'):
            f = odor_weight_factors(self.act.activity(site), float(pt['power']), pt.get('permute_seed'))
            a = f.to_numpy()
            frec[site] = dict(sha256=hashlib.sha256(np.ascontiguousarray(a, dtype='<f8').tobytes()).hexdigest(),
                              min=float(a.min()), max=float(a.max()), cv_mean=float((a.std(axis=0) / a.mean(axis=0)).mean()))
            if site == 'claw':
                kc_f = a
            else:
                pn_f = a
        x, (r, s), hist, at_b = E69.bisect_factor(lambda v: self.pool.run_on(v, kc_f, pn_f), P['target_on'], P['size'],
                                                  APL_BRACKET, APL_STEP, f"{pt['point_id']} APL scale")
        res = dict(params=dict(P['p'], **{k: P['p'][k] * x for k in TIED_ON}),
                   params_off=dict(P['p'], wAPLKC_scale=0.0, wAPLPN_scale=0.0), thr_factor=1.0, apl_factor=float(x),
                   at_bound_apl=bool(at_b), n_sims=len(hist), history=hist, seconds=time.time() - t0)
        return {'on': (r, s), 'off': P['saved']['off']}, res, dict(factors=frec, note="oracle emulation (APL on only; APL off is C2's own)")

    def run(self, specs: Sequence[dict], want: set) -> None:
        P = self.P
        if 'base' not in self.done:
            self.record(dict(point_id='base', handle='', setting='C2', value=np.nan, kind='base', identity=True), P['saved'],
                        dict(params=dict(P['p']), params_off=dict(P['p'], wAPLKC_scale=0.0, wAPLPN_scale=0.0)),
                        dict(note='C2 as saved (E67), rebuilt and checked exactly'))
        for spec in specs:
            pid = spec['point_id']
            if pid not in want or pid in self.done:
                continue
            if spec.get('optional') and spec['orn'] and not (H.shared_dir() / spec['orn'][1]).exists():
                print(f"{pid}: {spec['orn'][1]} not there yet; skipped", flush=True)
                continue
            pt = _resolve(spec, self.rows)
            if pt is None:
                print(f"{pid}: waits for {list(spec['candidates'])}; skipped", flush=True)
                continue
            fam_id = next((s['point_id'] for s in specs if s['identity'] and family(s) == family(pt)), None)
            if not pt['identity'] and fam_id is not None and fam_id not in self.done:
                print(f'{pid}: its identity point {fam_id} has not passed; skipped', flush=True)
                continue
            print(f"--- {pid} ({pt['kind']}: {pt['setting']})", flush=True)
            try:
                o, res, extra = (self.run_static if pt['kind'] == 'static' else self.run_emulation)(pt)
            except Exception as e:
                print(f'{pid} FAILED: {e!r}'[:500], flush=True)
                pd.DataFrame([dict(point_id=pid, handle=pt['handle'], setting=pt['setting'],
                                   time=datetime.now().isoformat(timespec='seconds'), error=repr(e)[:500])]).to_csv(
                    self.ep, index=False, mode='a', header=not self.ep.exists())
                continue
            if pt['identity']:
                _check_identity(pid, o, P['saved'], [res.get('thr_factor', 1.0), res.get('apl_factor', 1.0)])
            self.record(pt, o, res, extra)
            del o, res
            gc.collect()

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool = None


def screen(out: Path, only: Optional[Sequence[str]] = None, workers: Optional[int] = None) -> pd.DataFrame:
    """Runs (resumably) the screen's points on C2 and scores each as the entry's part C lists; writes screen.csv,
    screen_pairs.csv (per-pair effects, both readouts) and screen_params.json (each point's absolute parameters).

    Args:
        out: the results folder (results/E77_input_side_common_error/).
        only: point ids to run (default all); the identity point of each of their families is run first if not done.
        workers: emulation worker processes (`EmulationPool`; default `default_workers()`).

    Returns screen.csv's rows. A point that fails is listed in screen_errors.csv and not marked done; an identity point
    that does not reproduce C2 stops the screen (RuntimeError).
    """
    out.mkdir(parents=True, exist_ok=True)
    specs = point_specs()
    ids = [s['point_id'] for s in specs]
    want = set(ids) if only is None else set(only)
    unknown = want - set(ids)
    if unknown:
        raise ValueError(f'unknown points {sorted(unknown)}')
    if only is not None:
        fams = {family(s) for s in specs if s['point_id'] in want}
        want |= {s['point_id'] for s in specs if s['identity'] and family(s) in fams}
    write_provenance(out, specs)
    st = _Screen(out, workers)
    try:
        st.run(specs, want)
    finally:
        st.close()
    return st.rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', nargs='+', help='point ids to run (their families\' identity points run first)')
    ap.add_argument('--list', action='store_true', help='list the points and exit')
    ap.add_argument('--workers', type=int, help='emulation worker processes (default: half the OpenMP thread limit, at most 8)')
    args = ap.parse_args()
    if args.list:
        for s in point_specs():
            print(s['point_id'], s['kind'], s['setting'])
        return
    screen(H.results_dir(EXPERIMENT), only=args.only, workers=args.workers)


if __name__ == '__main__':
    main()
