#!/usr/bin/env python3
"""E83 parts B and C: the two searches on the four searched wirings (S: real, P1, P2, M1), each against one half of the
recordings. The entry is E83 in experiment_priorities.md; the readings, populations and part A are in
E83_cross_wiring_fit.py, whose interface this module builds on (`WiringModel`, `wiring_parts`, `build_simulator`,
`PopulationObjective`, `half_target`, `realism`, `save_run`).

- Part B, arm P: E75's cell-site profile of APL's weight (`profile_factors`: each KC's APL->KC weights times the
  exponential of the sum of its claws' glomerulus scores, divided by the mean), 36 scores with |s| <= ln 4, searched per
  half by E64's evolution strategy as (1 + 16) for 12 generations, sigma from 0.1. The start is that half's linear
  profile at half strength (E75's sensitivity map of the cell site and `fit_scores` at ridge 1e-4, against S's
  population pattern error with that half's recordings), or zero scores if those score better. Objective: the squared
  pattern error of S's population under the calibrated readout (each wiring's K held at its fit's) against the half,
  APL-on runs only (the APL-off runs are the saved fits). Per wiring the APL scale (APL->KC with KC->APL and bouton->APL
  tied) is stepped by secant steps on log count against log scale (`step_apl_scale`: from the parent's scale, at most
  three steps after the first run) to within 1 % of the wiring's APL-on response count; what is left is penalized.
- Part C, arm I: E77 part E's input search as it was (the ten odors in its order, factors in [0.5, 2] per glomerulus on
  the lab's response, (1 + 32) for eight generations per odor, sigma from 0.3, candidates as pseudo-odors, a penalty
  outside +-15 % of each wiring's response fractions, each wiring at its fit's parameters, breadth and K), with one
  change: the objective is the squared pattern error of S's population against the half, so every generation simulates
  the candidates on the four wirings (`InputPool`, one worker process per wiring).

Both searches check their identities before searching (zero scores and factor 1 reproduce each wiring's saved counts;
the objective at identity is E83_cross_wiring_fit.IDENTITY_OBJECTIVE), write their history every generation and resume
from a checkpoint (`--no-resume` starts over).

Outputs, results/E83_cross_wiring_fit/ (or `--out`):
- search_history_<P|I>_<half>.csv: per generation the best offspring's objective, the parent's, success, sigma, and S's
  pattern part (rms over the 120 pairs) of the parent against the searched half and the unseen half; arm P also the
  parent's APL factor and response count per wiring, arm I the best offspring's response fractions per wiring.
- search_candidates_<P|I>_<half>.csv: every candidate's objective, its squared pattern error and penalty (arm P with its
  scores, APL factors and counts per wiring).
- profile_scores_<half>.csv (arm P): glomerulus, score (the searched profile), start, linear_half_strength.
- input_factors_<half>.csv (arm I): half, odor, glomerulus, orn_delta, factor, log2_factor, searched_delta (E77's
  oracle_factors layout); inputs/searched_half<half>.csv, the searched input in E38's layout.
- search_runs_<P|I>_<half>.csv: per wiring the absolute parameters of the final point (arm P: the APL scales times the
  stepped factor), its response counts and realism (`realism`).
- provenance_search_<P|I>_<half>.json.
Spike counts and responses of each wiring's final point under outputs/E83_cross_wiring_fit/<B|C>/ (tag
<P|I>_half<half>_<wiring>), the checkpoints under outputs/E83_cross_wiring_fit/<B|C>/state_<half>/.

Usage (from this folder; arm P on one 16-thread lane, arm I on four 8-thread workers):
    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 nice -n 10 ../../venv/bin/python -u E83_searches.py --arm P --half A
    OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=8 nice -n 10 ../../venv/bin/python -u E83_searches.py --arm I --half A
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
import E83_cross_wiring_fit as X

EXPERIMENT = X.EXPERIMENT
S_WIRINGS = X.S_WIRINGS
HALVES = ('A', 'B')
N_PAIRS = 120

# E64's evolution strategy: sigma times 1.5 after a success, 0.8 after a failure, kept within [0.03, 1]
SUCCESS_UP, FAILURE_DOWN = 1.5, 0.8
SIGMA_MIN, SIGMA_MAX = 0.03, 1.0

# arm P (part B)
P_SITE = 'cell'
P_POP, P_GENS = 16, 12
P_SIGMA0 = 0.1
P_RIDGE = 1e-4
P_START_STRENGTH = 0.5
P_SEEDS = {'A': 830, 'B': 831}
APL_TOL = 0.01
APL_MAX_STEPS = 3
# a wiring's response count left more than APL_TOL from its target costs APL_PENALTY per unit of relative deviation
# beyond APL_TOL (E64's / E77's penalty per unit of relative response-fraction deviation beyond its tolerance)
APL_PENALTY = 10.0
# the scale at which each wiring's elasticity of the response count to the APL scale is probed (zero scores)
APL_PROBE = 1.1
# a secant step changes the APL scale by at most this factor
APL_MAX_STEP = 2.0

# arm I (part C): E77 part E's constants
I_ODORS = ('eb', '1-8ol', 'IaA', 'ep', '2h', '1-6ol', 'pa', '6al', '1-5ol', 'ms')
I_POP, I_GENS = 32, 8
I_BOUNDS = (0.5, 2.0)
I_RATE_TOL, I_PENALTY = 0.15, 10.0
I_SIGMA0 = 0.3
# E77's generator: seed 0 for half A, 1 for half B
I_SEEDS = {'A': 0, 'B': 1}
I_WORKERS_THREADS = 8


def _other(half: str) -> str:
    return 'B' if half == 'A' else 'A'


def _n(responses: pd.DataFrame) -> int:
    return int((responses.to_numpy(dtype=float) > 0.5).sum())


def _git_head(path: Path = H.HERE) -> str:
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=path, capture_output=True, text=True).stdout.strip()


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------
# the evolution strategy and the APL scale steps (no simulation)

def es_offspring(rng: np.random.Generator, parent: np.ndarray, sigma: float, pop: int, lo: float, hi: float) -> np.ndarray:
    """E64's offspring: `pop` vectors parent + sigma x N(0, 1) per component, clipped to [lo, hi] (all in log space; one
    standard_normal draw of shape (pop, len(parent)), as E64 and E77 draw them)."""
    return np.clip(np.asarray(parent, dtype=float)[None, :] + sigma * rng.standard_normal((pop, len(parent))), lo, hi)


def es_sigma(sigma: float, success: bool) -> float:
    """E64's step-size rule: times 1.5 after a success, 0.8 after a failure, within [SIGMA_MIN, SIGMA_MAX]."""
    return float(np.clip(sigma * (SUCCESS_UP if success else FAILURE_DOWN), SIGMA_MIN, SIGMA_MAX))


def step_apl_scale(run_on: Callable[[float], Tuple[pd.DataFrame, pd.DataFrame]], target_n: int, x0: float,
                   elasticity: float, tol: float = APL_TOL, max_steps: int = APL_MAX_STEPS,
                   bracket: Tuple[float, float] = X.APL_BRACKET, first: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None
                   ) -> dict:
    """The APL scale factor at which `run_on`'s response count is within `tol` of `target_n`, by secant steps on log
    count against log factor: run at `x0` (or take `first`, its outputs at `x0`), then at most `max_steps` steps, the
    first along `elasticity` (d log count / d log factor, negative), the later ones along the secant of the last two
    runs (the elasticity again when that slope is not negative); each step changes the factor by at most APL_MAX_STEP
    times and stays within `bracket`. Stops once a run is within `tol`.

    Returns:
        dict with 'x' (the factor of the run closest to the target, the first of equals), 'out' (its responses and
        spike counts), 'n', 'rel_dev' (|n / target_n - 1|), 'penalty' (APL_PENALTY x the deviation beyond `tol`),
        'n_sims' (runs made; 0 when `first` was the only one), 'steps' ([(factor, count)] in order).
    """
    if elasticity >= 0 or not np.isfinite(elasticity):
        raise ValueError(f'the elasticity must be negative, got {elasticity}')
    lo, hi = np.log(bracket[0]), np.log(bracket[1])
    lx = float(np.clip(np.log(x0), lo, hi))
    pts: List[Tuple[float, int, tuple]] = []
    n_sims = 0
    for k in range(max_steps + 1):
        x = float(np.exp(lx))
        if k == 0 and first is not None:
            out = first
        else:
            out = run_on(x)
            n_sims += 1
        n = _n(out[0])
        pts.append((x, n, out))
        if abs(n / target_n - 1.0) <= tol or k == max_steps:
            break
        slope = elasticity
        if len(pts) >= 2:
            (x1, n1, _), (x2, n2, _) = pts[-2], pts[-1]
            if n1 > 0 and n2 > 0 and n1 != n2 and x1 != x2:
                s = (np.log(n2) - np.log(n1)) / (np.log(x2) - np.log(x1))
                if s < 0:
                    slope = float(s)
        step = (np.log(target_n) - np.log(max(n, 1))) / slope
        step = float(np.clip(step, -np.log(APL_MAX_STEP), np.log(APL_MAX_STEP)))
        new = float(np.clip(lx + step, lo, hi))
        if new == lx:
            break
        lx = new
    x, n, out = min(pts, key=lambda p: abs(p[1] - target_n))
    rel = abs(n / target_n - 1.0)
    return dict(x=x, out=out, n=n, rel_dev=rel, penalty=APL_PENALTY * max(0.0, rel - tol), n_sims=n_sims,
                steps=[(p[0], p[1]) for p in pts])


def rate_penalty(fracs: Mapping[str, Tuple[float, float]], base: Mapping[str, Tuple[float, float]]) -> float:
    """E77 part E's penalty on one odor's response fractions, summed over the wirings: per wiring and condition (APL on,
    APL off), I_PENALTY x the relative deviation from the wiring's own fraction (`base`, its saved fit's) beyond
    I_RATE_TOL; a condition whose base fraction is 0 is not penalized."""
    pen = 0.0
    for w, (fo, ff) in fracs.items():
        for f, b in zip((fo, ff), base[w]):
            if b > 0:
                pen += I_PENALTY * max(0.0, abs(f / b - 1.0) - I_RATE_TOL)
    return float(pen)


# ---------------------------------------------------------------------------
# checkpoints

def _state_dir(part: str, half: str, out_dir: Optional[Path] = None) -> Path:
    d = (out_dir or H.outputs_dir(EXPERIMENT)) / part / f'state_{half}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float) + '\n')
    os.replace(tmp, path)


def _save_frames(d: Path, prefix: str, frames: Mapping[str, pd.DataFrame]) -> None:
    for name, df in frames.items():
        tmp = d / f'{prefix}_{name}.parquet.tmp'
        df.to_parquet(tmp)
        os.replace(tmp, d / f'{prefix}_{name}.parquet')


def _load_frame(d: Path, prefix: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(d / f'{prefix}_{name}.parquet')


def _drop_other_frames(d: Path, keep: str) -> None:
    """Remove saved frames whose set tag (the part after '__' in the name) is not `keep`."""
    for p in d.glob('*__*.parquet'):
        if p.name.split('__', 1)[1].split('_', 1)[0] != keep:
            p.unlink()


def _reset_state(d: Path) -> None:
    """Remove a search's checkpoint and saved frames (a fresh start)."""
    for p in list(d.glob('checkpoint.json*')) + list(d.glob('*.parquet')) + list(d.glob('*.parquet.tmp')):
        p.unlink()


def _rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def _rng_from(state: dict) -> np.random.Generator:
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


# ---------------------------------------------------------------------------
# pieces shared by the two arms

def population_error(on: Mapping[str, pd.DataFrame], off: Mapping[str, pd.DataFrame], Ks: Mapping[str, float],
                     target: pd.Series) -> pd.Series:
    """The pattern error per pair (canonical order) of the population of `on`'s wirings: model minus data per-pair APL
    effect, both means removed, the model's under the calibrated readout at each wiring's `Ks` (the mean over wirings of
    the pair distances per condition, as E72's `mean_distances`)."""
    pop = {c: pd.concat([H.readout_pair_distances(m[w], 'saturating', Ks[w]) for w in on], axis=1).mean(axis=1)
           for c, m in (('on', on), ('off', off))}
    e = H.apl_effect(pop['on'], pop['off'])
    e.index = pd.MultiIndex.from_tuples([tuple(sorted(x)) for x in e.index], names=['odor_a', 'odor_b'])
    e = e.sort_index()
    d = target.copy()
    d.index = pd.MultiIndex.from_tuples([tuple(sorted(x)) for x in d.index], names=['odor_a', 'odor_b'])
    d = d.reindex(e.index)
    if d.isna().any():
        raise ValueError('the target lacks pairs of the population')
    return (e - e.mean()) - (d - d.mean())


def provenance(out: Path, arm: str, half: str, extra: dict) -> None:
    """provenance_search_<arm>_<half>.json: the model's provenance (`H.model_provenance` on the lab's input and the real
    wiring), the commits, the wirings' fits with their absolute parameters and K, the halves' recordings, the settings."""
    import E77_input_side_common_error as E77
    parts = {w: X.wiring_parts(w) for w in S_WIRINGS}
    real = parts['real']
    hv = E77.recording_halves()
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E83_searches.py', arm=arm,
                half=half, al_analysis_commit=_git_head(),
                model=H.model_provenance(orn_deltas=real['orn'], _wPNKC=real['W']),
                wirings={w: dict(folder=p['folder'], tag=p['tag'], params=p['p'], K=p['K'], target_on=p['target_on'],
                                 target_off=p['target_off']) for w, p in parts.items()},
                halves={h: {g: [list(map(str, r)) for r in v] for g, v in hv[h].items()} for h in hv},
                tied_on=X.TIED_ON, approx_conc_note=H.APPROX_CONC_NOTE, **extra)
    _write_json(out / f'provenance_search_{arm}_{half}.json', prov)


def _append_csv(path: Path, rows: Sequence[dict]) -> None:
    """Append rows to a CSV, rewriting it with the union of the columns (the start row and the generations' rows differ)."""
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True, sort=False)
    tmp = path.with_suffix(path.suffix + '.tmp')
    new.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _trim_csv(path: Path, keep: Callable[[pd.DataFrame], pd.Series]) -> None:
    """Keep only the rows `keep` selects (a resumed search drops what it wrote after its last checkpoint)."""
    if path.exists():
        d = pd.read_csv(path)
        d[keep(d)].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# arm P (part B): APL's weight by glomerulus

def linear_start(half: str, saved_on: Mapping[str, pd.DataFrame], saved_off: Mapping[str, pd.DataFrame],
                 Ks: Mapping[str, float]) -> Dict[str, float]:
    """That half's linear profile at half strength: E75's cell-site map (the real wiring's per-pair change of the APL
    effect per unit score, under the calibrated readout) and `fit_scores` at ridge P_RIDGE against S's population
    pattern error with that half's recordings (at the fits), times P_START_STRENGTH. {glomerulus: score}."""
    import E75_glomerulus_apl_profile as E75
    c = population_error(saved_on, saved_off, Ks, X.half_target(half))
    M = E75.map_matrix(H.results_dir(E75.EXPERIMENT), P_SITE)
    M.index = pd.MultiIndex.from_tuples([tuple(sorted(x)) for x in M.index], names=['odor_a', 'odor_b'])
    M = M.reindex(c.index)
    if M.isna().any().any() or M.shape != (N_PAIRS, 36):
        raise RuntimeError(f"E75's cell map does not cover the 120 pairs and 36 glomeruli ({M.shape})")
    s = E75.fit_scores(M.to_numpy(), c.to_numpy(), P_RIDGE)
    return {str(g): float(P_START_STRENGTH * v) for g, v in zip(M.columns, s)}


class ProfileSearch:
    """S's four wirings at their fits for arm P: `evaluate(scores, xs)` sets the profile on every wiring, steps each
    wiring's APL scale from `xs` (`step_apl_scale`) and scores the population against the searched and the unseen half."""

    def __init__(self, half: str, check: bool = True):
        self.half = half
        self.models = {w: X.WiringModel(w, check=check) for w in S_WIRINGS}
        self.obj = X.PopulationObjective(X.half_target(half), S_WIRINGS)
        self.other = X.PopulationObjective(X.half_target(_other(half)), S_WIRINGS)
        self.saved_on = {w: m.saved['on'][1] for w, m in self.models.items()}
        self.saved_off = {w: m.saved['off'][1] for w, m in self.models.items()}
        self.Ks = {w: float(m.K) for w, m in self.models.items()}
        self.elasticity: Dict[str, float] = {}

    def identity(self) -> dict:
        """Zero scores at factor 1: each wiring's APL-on run must be its saved fit's exactly; the objective there."""
        for w, m in self.models.items():
            m.set_profile(P_SITE, {})
            r, s = m.run_on(1.0)
            if not (np.array_equal(s.to_numpy(dtype=float), m.saved['on'][1].to_numpy(dtype=float))
                    and np.array_equal(r.to_numpy(dtype=float), m.saved['on'][0].to_numpy(dtype=float))):
                raise RuntimeError(f'{w}: zero scores at factor 1 do not reproduce the saved APL-on run')
        o = self.obj(self.saved_on)
        ot = self.other(self.saved_on)
        want = X.IDENTITY_OBJECTIVE[self.half]
        if abs(o['rms'] - want) > 5e-5:
            raise RuntimeError(f"the objective at identity is {o['rms']:.5f} against half {self.half}, expected {want}")
        print(f"half {self.half}: zero scores reproduce the four saved fits; S's pattern part {o['rms']:.4f} against this "
              f"half, {ot['rms']:.4f} against the other", flush=True)
        return dict(sse=o['sse'], rms=o['rms'], rms_other=ot['rms'])

    def probe_elasticity(self) -> Dict[str, float]:
        """Each wiring's d log(APL-on count) / d log(APL scale factor) at zero scores, from factor 1 (the saved count)
        and APL_PROBE."""
        for w, m in self.models.items():
            m.set_profile(P_SITE, {})
            r, _ = m.run_on(APL_PROBE)
            n1, n0 = _n(r), m.target_on
            e = (np.log(n1) - np.log(n0)) / np.log(APL_PROBE) if n1 > 0 else np.nan
            if not (np.isfinite(e) and e < 0):
                raise RuntimeError(f'{w}: the APL-on count does not fall with the APL scale ({n0} at 1, {n1} at {APL_PROBE})')
            self.elasticity[w] = float(e)
        print('elasticity of the APL-on count to the APL scale: '
              + ', '.join(f'{w} {e:.3f}' for w, e in self.elasticity.items()), flush=True)
        return dict(self.elasticity)

    def evaluate(self, scores: Mapping[str, float], xs: Mapping[str, float]) -> dict:
        per = {}
        for w, m in self.models.items():
            m.set_profile(P_SITE, dict(scores))
            per[w] = step_apl_scale(m.run_on, m.target_on, xs[w], self.elasticity[w])
            m.set_profile(None, {})
        on = {w: p['out'][1] for w, p in per.items()}
        o = self.obj(on)
        pen = float(sum(p['penalty'] for p in per.values()))
        return dict(obj=o['sse'] + pen, sse=o['sse'], rms=o['rms'], penalty=pen, per=per,
                    rms_other=self.other(on)['rms'], n_sims=int(sum(p['n_sims'] for p in per.values())))

    def close(self) -> None:
        for m in self.models.values():
            m.set_profile(None, {})


def _p_row(gen: int, ev: dict, extra: dict) -> dict:
    row = dict(generation=gen, **extra, sse=ev['sse'], penalty=ev['penalty'], obj=ev['obj'],
               pattern_rms_half=ev['rms'], pattern_rms_other=ev['rms_other'], n_sims=ev['n_sims'])
    for w, p in ev['per'].items():
        row[f'apl_factor_{w}'] = p['x']
        row[f'n_on_{w}'] = p['n']
        row[f'rel_dev_{w}'] = p['rel_dev']
    return row


def part_b(out: Path, half: str, gens: int = P_GENS, pop: int = P_POP, resume: bool = True,
           out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Part B (arm P) against half `half`: the identity, the start, the (1 + `pop`) search for `gens` generations, the
    final point's runs and tables (see the module docstring). Resumes from the last generation's checkpoint."""
    if half not in HALVES:
        raise ValueError(half)
    t_start = time.time()
    sdir = _state_dir('B', half, out_dir)
    ck = sdir / 'checkpoint.json'
    hist_path = out / f'search_history_P_{half}.csv'
    cand_path = out / f'search_candidates_P_{half}.csv'
    S = ProfileSearch(half)
    ident = S.identity()
    glomeruli = None
    if resume and ck.exists():
        st = json.loads(ck.read_text())
        glomeruli = list(st['glomeruli'])
        S.elasticity = {w: float(v) for w, v in st['elasticity'].items()}
        parent = np.array(st['parent'], dtype=float)
        parent_obj, sigma, g0 = float(st['parent_obj']), float(st['sigma']), int(st['generation']) + 1
        xs = {w: float(v) for w, v in st['parent_x'].items()}
        rng = _rng_from(st['rng'])
        start_info = st['start']
        frames = st['frames']
        par = {w: (_load_frame(sdir, f'{w}__{frames}', 'on_responses'), _load_frame(sdir, f'{w}__{frames}', 'on_spike_counts'))
               for w in S_WIRINGS}
        _trim_csv(hist_path, lambda d: d.generation <= st['generation'])
        _trim_csv(cand_path, lambda d: d.generation <= st['generation'])
        # the parent's runs must still give its objective (same build, same profile and factors)
        o = S.obj({w: par[w][1] for w in S_WIRINGS})
        pen = sum(APL_PENALTY * max(0.0, abs(_n(par[w][0]) / S.models[w].target_on - 1.0) - APL_TOL) for w in S_WIRINGS)
        if abs(o['sse'] + pen - parent_obj) > 1e-12:
            raise RuntimeError(f'the checkpoint parent scores {o["sse"] + pen} now, {parent_obj} when saved')
        print(f'resuming half {half} at generation {g0} (parent objective {parent_obj:.5f}, sigma {sigma:.3f})', flush=True)
    else:
        for p in (hist_path, cand_path):
            if p.exists():
                p.unlink()
        _reset_state(sdir)
        S.probe_elasticity()
        lin = linear_start(half, S.saved_on, S.saved_off, S.Ks)
        glomeruli = list(lin)
        ones = {w: 1.0 for w in S_WIRINGS}
        ev_lin = S.evaluate(lin, ones)
        zero_obj = ident['sse']
        print(f"half {half}: linear start (half strength): objective {ev_lin['obj']:.5f} (pattern {ev_lin['rms']:.4f}, "
              f"penalty {ev_lin['penalty']:.4f}); zero scores {zero_obj:.5f} (pattern {ident['rms']:.4f})", flush=True)
        _append_csv(cand_path, [dict(_p_row(-1, ev_lin, dict(candidate='linear_start')), scores=json.dumps(lin))])
        if ev_lin['obj'] < zero_obj:
            parent = np.array([lin[g] for g in glomeruli])
            parent_obj = ev_lin['obj']
            xs = {w: p['x'] for w, p in ev_lin['per'].items()}
            par = {w: p['out'] for w, p in ev_lin['per'].items()}
            start = 'linear'
            row0 = _p_row(-1, ev_lin, dict(candidate='start'))
        else:
            parent = np.zeros(len(glomeruli))
            parent_obj = zero_obj
            xs = dict(ones)
            par = {w: m.saved['on'] for w, m in S.models.items()}
            start = 'zero'
            row0 = dict(generation=-1, candidate='start', sse=ident['sse'], penalty=0.0, obj=zero_obj,
                        pattern_rms_half=ident['rms'], pattern_rms_other=ident['rms_other'], n_sims=0,
                        **{f'apl_factor_{w}': 1.0 for w in S_WIRINGS},
                        **{f'n_on_{w}': S.models[w].target_on for w in S_WIRINGS}, **{f'rel_dev_{w}': 0.0 for w in S_WIRINGS})
        start_info = dict(chosen=start, linear=lin, linear_obj=ev_lin['obj'], linear_rms=ev_lin['rms'],
                          linear_penalty=ev_lin['penalty'], zero_obj=zero_obj, zero_rms=ident['rms'])
        _append_csv(hist_path, [dict(row0, best_obj=parent_obj, parent_obj=parent_obj, success=True, sigma=P_SIGMA0,
                                     start=start, seconds=time.time() - t_start)])
        sigma, g0 = P_SIGMA0, 0
        rng = np.random.default_rng(P_SEEDS[half])
        frames = 'start'
        for w in S_WIRINGS:
            _save_frames(sdir, f'{w}__{frames}', {'on_responses': par[w][0], 'on_spike_counts': par[w][1]})
        print(f'half {half}: starting from the {start} profile', flush=True)
    provenance(out, 'P', half, dict(
        site=P_SITE, pop=pop, generations=gens, sigma0=P_SIGMA0, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
        success_up=SUCCESS_UP, failure_down=FAILURE_DOWN, seed=P_SEEDS[half], score_bound=float(np.log(4)), ridge=P_RIDGE,
        start_strength=P_START_STRENGTH, apl_tol=APL_TOL, apl_max_steps=APL_MAX_STEPS, apl_penalty=APL_PENALTY,
        apl_max_step=APL_MAX_STEP, apl_probe=APL_PROBE, apl_bracket=X.APL_BRACKET, elasticity=S.elasticity,
        identity=ident, start=start_info, glomeruli=glomeruli,
        e75_map=str(H.results_dir('E75_glomerulus_apl_profile') / 'sensitivity_pairs.csv'),
        e75_map_sha256=hashlib.sha256((H.results_dir('E75_glomerulus_apl_profile') / 'sensitivity_pairs.csv').read_bytes()).hexdigest()))
    s_max = float(np.log(4))
    for g in range(g0, gens):
        t0 = time.time()
        offspring = es_offspring(rng, parent, sigma, pop, -s_max, s_max)
        evs, crow = [], []
        for i in range(pop):
            sc = {gl: float(v) for gl, v in zip(glomeruli, offspring[i])}
            ev = S.evaluate(sc, xs)
            evs.append(ev)
            crow.append(dict(_p_row(g, ev, dict(candidate=i)), scores=json.dumps(sc)))
            print(f"  half {half} gen {g} candidate {i}: objective {ev['obj']:.5f} (pattern {ev['rms']:.4f}, other "
                  f"{ev['rms_other']:.4f}, penalty {ev['penalty']:.4f}, {ev['n_sims']} runs)", flush=True)
        objs = np.array([e['obj'] for e in evs])
        b = int(np.argmin(objs))
        success = bool(objs[b] < parent_obj)
        if success:
            parent, parent_obj = offspring[b].copy(), float(objs[b])
            xs = {w: p['x'] for w, p in evs[b]['per'].items()}
            par = {w: p['out'] for w, p in evs[b]['per'].items()}
        sigma = es_sigma(sigma, success)
        po = S.obj({w: par[w][1] for w in S_WIRINGS})
        pt = S.other({w: par[w][1] for w in S_WIRINGS})
        hrow = dict(generation=g, candidate='parent', best_obj=float(objs[b]), parent_obj=parent_obj, success=success,
                    sigma=sigma, sse=po['sse'], penalty=parent_obj - po['sse'], obj=parent_obj, pattern_rms_half=po['rms'],
                    pattern_rms_other=pt['rms'], n_sims=int(sum(e['n_sims'] for e in evs)),
                    **{f'apl_factor_{w}': xs[w] for w in S_WIRINGS}, **{f'n_on_{w}': _n(par[w][0]) for w in S_WIRINGS},
                    **{f'rel_dev_{w}': abs(_n(par[w][0]) / S.models[w].target_on - 1.0) for w in S_WIRINGS},
                    score_rms=float(np.sqrt(np.mean(parent ** 2))), score_max_abs=float(np.max(np.abs(parent))),
                    seconds=time.time() - t0)
        _append_csv(cand_path, crow)
        if success:
            frames = f'g{g}'
            for w in S_WIRINGS:
                _save_frames(sdir, f'{w}__{frames}', {'on_responses': par[w][0], 'on_spike_counts': par[w][1]})
        _write_json(ck, dict(half=half, generation=g, frames=frames, glomeruli=glomeruli, parent=parent.tolist(), parent_obj=parent_obj,
                             sigma=sigma, parent_x=xs, rng=_rng_state(rng), elasticity=S.elasticity, start=start_info))
        _drop_other_frames(sdir, frames)
        _append_csv(hist_path, [hrow])
        print(f"half {half} gen {g}: best {objs[b]:.5f} (parent {parent_obj:.5f}), S's pattern part {po['rms']:.4f} against "
              f"this half, {pt['rms']:.4f} against the other, sigma {sigma:.3f}, {hrow['n_sims']} runs, "
              f"{time.time() - t0:.0f}s", flush=True)
    S.close()
    # the tables of the final point
    final = {gl: float(v) for gl, v in zip(glomeruli, parent)}
    sc = pd.DataFrame(dict(glomerulus=glomeruli, score=[final[g] for g in glomeruli],
                           start=[float(start_info['linear'][g]) if start_info['chosen'] == 'linear' else 0.0 for g in glomeruli],
                           linear_half_strength=[float(start_info['linear'][g]) for g in glomeruli]))
    sc['factor'] = np.exp(sc.score)
    sc.to_csv(out / f'profile_scores_{half}.csv', index=False)
    rows = []
    for w, m in S.models.items():
        o = {'on': par[w], 'off': m.saved['off']}
        X.save_run('B', f'P_half{half}_{w}', o)
        p_abs = dict(m.p, **{k: m.p[k] * xs[w] for k in X.TIED_ON})
        rows.append(dict(arm='P', half=half, wiring=w, tag=f'P_half{half}_{w}', fit_folder=m.folder, fit_tag=m.tag,
                         apl_factor=xs[w], thr_factor=1.0, n_on=_n(par[w][0]), target_on=m.target_on,
                         n_off=_n(m.saved['off'][0]), target_off=m.target_off, K=m.K, profile_sha256=_sha(final),
                         **{f'fit_{k}': v for k, v in m.p.items()}, **p_abs, **X.realism(o)))
    pd.DataFrame(rows).to_csv(out / f'search_runs_P_{half}.csv', index=False)
    print(pd.DataFrame(rows)[['wiring', 'apl_factor', 'n_on', 'target_on', 'rate_on', 'ratio', 'rate_rho_on',
                              'realism_flag']].to_string(index=False), flush=True)
    return pd.read_csv(hist_path)


# ---------------------------------------------------------------------------
# arm I (part C): each odor's input

def _input_worker(conn, label: str) -> None:
    """An InputPool worker: one wiring's pieces (`wiring_parts`), then for each (odor, factors) received the
    candidates simulated as pseudo-odors (E64's `candidate_orn` on the lab's input) on a simulator built for them at the
    wiring's fit, the lab input's breadth held, APL on and APL off; until None."""
    try:
        from E64_input_oracle import candidate_orn
        parts = X.wiring_parts(label)
        p = dict(parts['p'])
        conn.send(('ready', dict(label=label, n_kc=int(parts['saved']['on'][1].shape[0]),
                                 odors=[str(c) for c in parts['saved']['on'][1].columns])))
    except Exception as e:  # pragma: no cover - reported to the parent
        conn.send(('error', repr(e)))
        return
    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            odor, factors = msg
            cand = candidate_orn(parts['orn'], odor, np.asarray(factors, dtype=float))
            sim, _ = X.build_simulator(label, orn=cand, breadth=parts['breadth'])
            r_on, s_on = sim(dict(p))
            r_off, s_off = sim(dict(p, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
            del sim
            gc.collect()
            if not s_on.index.equals(parts['saved']['on'][1].index):
                raise RuntimeError(f'{label}: the candidates\' KCs are not the saved fit\'s')
            conn.send(('ok', (r_on.to_numpy(), s_on.to_numpy(), r_off.to_numpy(), s_off.to_numpy())))
        except Exception as e:
            conn.send(('error', repr(e)))


class InputPool:
    """One worker process per wiring (spawned, `threads` OpenMP threads each), each simulating a generation's candidates
    for one odor on its wiring (`_input_worker`); `run(odor, factors)` returns {wiring: (responses on, counts on,
    responses off, counts off)} as arrays (KCs x candidates)."""

    def __init__(self, wirings: Sequence[str] = S_WIRINGS, threads: int = I_WORKERS_THREADS):
        import multiprocessing
        ctx = multiprocessing.get_context('spawn')
        keep = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT')}
        os.environ['OMP_NUM_THREADS'] = os.environ['OMP_THREAD_LIMIT'] = str(int(threads))
        self.wirings = list(wirings)
        self.procs, self.conns = {}, {}
        try:
            for w in self.wirings:
                a, b = ctx.Pipe()
                p = ctx.Process(target=_input_worker, args=(b, w), daemon=True)
                p.start()
                self.procs[w], self.conns[w] = p, a
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for w in self.wirings:
            status, v = self.conns[w].recv()
            if status != 'ready':
                self.close()
                raise RuntimeError(f'the worker of {w} failed to start: {v}')
        print(f'input pool: {len(self.procs)} workers ({", ".join(self.wirings)}), {threads} threads each', flush=True)

    def run(self, odor: str, factors: np.ndarray) -> Dict[str, Tuple[np.ndarray, ...]]:
        for w in self.wirings:
            self.conns[w].send((odor, np.asarray(factors, dtype=float)))
        res, err = {}, []
        for w in self.wirings:
            status, v = self.conns[w].recv()
            if status != 'ok':
                err.append(f'{w}: {v}')
            else:
                res[w] = v
        if err:
            raise RuntimeError('input workers failed: ' + '; '.join(err))
        return res

    def close(self) -> None:
        for c in self.conns.values():
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs.values():
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
        self.procs, self.conns = {}, {}


def _odor_col(df: pd.DataFrame, odor: str):
    from E64_input_oracle import out_col
    return out_col(df, odor)


def _with_column(df: pd.DataFrame, col, values: np.ndarray) -> pd.DataFrame:
    d = df.copy()
    d[col] = values
    return d


def candidate_objectives(inc: Mapping[str, Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]], odor: str,
                         res: Mapping[str, Tuple[np.ndarray, ...]], obj, base_fracs: Mapping[str, Tuple[float, float]]
                         ) -> List[dict]:
    """Each candidate's objective: the odor's column of every wiring's incumbent (`inc[w]['on' / 'off']`, (responses,
    spike counts)) replaced by the candidate's (column i of `res[w]`), S's population pattern error (`obj`) plus
    `rate_penalty` on the odor's response fractions against `base_fracs`."""
    oc = _odor_col(next(iter(inc.values()))['on'][1], odor)
    n_cand = next(iter(res.values()))[1].shape[1]
    rows = []
    for i in range(n_cand):
        on = {w: _with_column(inc[w]['on'][1], oc, res[w][1][:, i]) for w in inc}
        off = {w: _with_column(inc[w]['off'][1], oc, res[w][3][:, i]) for w in inc}
        o = obj(on=on, off=off)
        fr = {w: (float((res[w][0][:, i] > 0.5).mean()), float((res[w][2][:, i] > 0.5).mean())) for w in inc}
        pen = rate_penalty(fr, base_fracs)
        rows.append(dict(obj=o['sse'] + pen, sse=o['sse'], rms=o['rms'], penalty=pen, fracs=fr))
    return rows


def part_c(out: Path, half: str, odors: Sequence[str] = I_ODORS, gens: int = I_GENS, pop: int = I_POP,
           threads: int = I_WORKERS_THREADS, resume: bool = True, out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Part C (arm I) against half `half`: E77 part E's search (`E77_input_side_common_error.input_oracle`) on S's
    population (see the module docstring). Resumes from the last generation's checkpoint."""
    if half not in HALVES:
        raise ValueError(half)
    sdir = _state_dir('C', half, out_dir)
    ck = sdir / 'checkpoint.json'
    hist_path = out / f'search_history_I_{half}.csv'
    cand_path = out / f'search_candidates_I_{half}.csv'
    fac_path = out / f'input_factors_{half}.csv'
    parts = {w: X.wiring_parts(w) for w in S_WIRINGS}
    orn = parts['real']['orn']
    glomeruli = list(orn.index)
    obj = X.PopulationObjective(X.half_target(half), S_WIRINGS)
    other = X.PopulationObjective(X.half_target(_other(half)), S_WIRINGS)
    saved = {w: parts[w]['saved'] for w in S_WIRINGS}
    oc0 = {o: _odor_col(saved['real']['on'][1], o) for o in odors}
    base_fracs = {o: {w: (float(saved[w]['on'][0][oc0[o]].astype(float).gt(0.5).mean()),
                          float(saved[w]['off'][0][oc0[o]].astype(float).gt(0.5).mean())) for w in S_WIRINGS} for o in odors}
    lo, hi = np.log(I_BOUNDS[0]), np.log(I_BOUNDS[1])
    o0 = obj(on={w: saved[w]['on'][1] for w in S_WIRINGS}, off={w: saved[w]['off'][1] for w in S_WIRINGS})
    ot0 = other(on={w: saved[w]['on'][1] for w in S_WIRINGS}, off={w: saved[w]['off'][1] for w in S_WIRINGS})
    if abs(o0['rms'] - X.IDENTITY_OBJECTIVE[half]) > 5e-5:
        raise RuntimeError(f"the objective at identity is {o0['rms']:.5f} against half {half}, expected {X.IDENTITY_OBJECTIVE[half]}")
    pool = InputPool(S_WIRINGS, threads)
    try:
        # the pseudo-odor path at factor 1 must give each wiring's saved counts for the odor
        res = pool.run(odors[0], np.ones((1, len(glomeruli))))
        for w in S_WIRINGS:
            for c, k in (('on', 1), ('off', 3)):
                if not np.array_equal(res[w][k][:, 0].astype(float), saved[w][c][1][oc0[odors[0]]].to_numpy(dtype=float)):
                    raise RuntimeError(f"{w}: the pseudo-odor run of {odors[0]} at factor 1 differs from the saved {c} counts")
        print(f"half {half}: the pseudo-odor runs at factor 1 reproduce the four saved fits; S's pattern part "
              f"{o0['rms']:.4f} against this half, {ot0['rms']:.4f} against the other", flush=True)
        done: Dict[str, List[float]] = {}
        inc = {w: {c: tuple(x.copy() for x in saved[w][c]) for c in ('on', 'off')} for w in S_WIRINGS}
        start_odor, start_gen = 0, 0
        parent, parent_obj, sigma = None, None, I_SIGMA0
        rng = np.random.default_rng(I_SEEDS[half])
        if resume and ck.exists():
            st = json.loads(ck.read_text())
            done = {o: list(v) for o, v in st['done'].items()}
            rng = _rng_from(st['rng'])
            oi, g = int(st['odor_index']), int(st['generation'])
            frames = st['frames']
            inc = {w: {c: (_load_frame(sdir, f'{w}__{frames}', f'{c}_responses'), _load_frame(sdir, f'{w}__{frames}', f'{c}_spike_counts'))
                       for c in ('on', 'off')} for w in S_WIRINGS}
            if g + 1 < gens:
                start_odor, start_gen = oi, g + 1
                parent, parent_obj, sigma = np.array(st['parent']), float(st['parent_obj']), float(st['sigma'])
            else:
                start_odor, start_gen = oi + 1, 0
            pos = oi * gens + g
            _trim_csv(hist_path, lambda d: d.odor.map({o: k for k, o in enumerate(odors)}) * gens + d.generation <= pos)
            _trim_csv(cand_path, lambda d: d.odor.map({o: k for k, o in enumerate(odors)}) * gens + d.generation <= pos)
            print(f'resuming half {half} at odor {start_odor} generation {start_gen}', flush=True)
        else:
            for p in (hist_path, cand_path, fac_path):
                if p.exists():
                    p.unlink()
            _reset_state(sdir)
            frames = 'start'
            for w in S_WIRINGS:
                _save_frames(sdir, f'{w}__{frames}', {'on_responses': inc[w]['on'][0], 'on_spike_counts': inc[w]['on'][1],
                                                      'off_responses': inc[w]['off'][0], 'off_spike_counts': inc[w]['off'][1]})
        provenance(out, 'I', half, dict(
            odors=list(odors), pop=pop, generations=gens, bounds=I_BOUNDS, rate_tol=I_RATE_TOL, penalty=I_PENALTY,
            sigma0=I_SIGMA0, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, success_up=SUCCESS_UP, failure_down=FAILURE_DOWN,
            seed=I_SEEDS[half], workers=len(S_WIRINGS), threads_per_worker=threads, identity=dict(sse=o0['sse'], rms=o0['rms'],
            rms_other=ot0['rms']), base_fracs=base_fracs,
            penalty_rule='summed over the four wirings and both conditions; each against the wiring\'s saved fraction',
            breadth="each wiring's lab-input breadth (wiring_parts) held for every candidate"))
        for oi in range(start_odor, len(odors)):
            odor = odors[oi]
            oc = _odor_col(inc['real']['on'][1], odor)
            if oi == start_odor and parent is not None:
                g_first = start_gen
            else:
                parent = np.ones(len(glomeruli))
                o_cur = obj(on={w: inc[w]['on'][1] for w in S_WIRINGS}, off={w: inc[w]['off'][1] for w in S_WIRINGS})
                parent_obj, sigma, g_first = float(o_cur['sse']), I_SIGMA0, 0
            for g in range(g_first, gens):
                t0 = time.time()
                offspring = np.exp(es_offspring(rng, np.log(parent), sigma, pop, lo, hi))
                res = pool.run(odor, offspring)
                rows = candidate_objectives(inc, odor, res, obj, base_fracs[odor])
                objs = np.array([r['obj'] for r in rows])
                b = int(np.argmin(objs))
                success = bool(objs[b] < parent_obj)
                if success:
                    parent, parent_obj = offspring[b].copy(), float(objs[b])
                    for w in S_WIRINGS:
                        r_on, s_on = inc[w]['on']
                        r_off, s_off = inc[w]['off']
                        inc[w] = {'on': (_with_column(r_on, oc, res[w][0][:, b]), _with_column(s_on, oc, res[w][1][:, b])),
                                  'off': (_with_column(r_off, oc, res[w][2][:, b]), _with_column(s_off, oc, res[w][3][:, b]))}
                sigma = es_sigma(sigma, success)
                on = {w: inc[w]['on'][1] for w in S_WIRINGS}
                off = {w: inc[w]['off'][1] for w in S_WIRINGS}
                ph, po = obj(on=on, off=off), other(on=on, off=off)
                hrow = dict(half=half, odor=odor, generation=g, best_obj=float(objs[b]), parent_obj=parent_obj,
                            success=success, sigma=sigma, best_sse=rows[b]['sse'], best_penalty=rows[b]['penalty'],
                            pattern_rms_half=ph['rms'], pattern_rms_other=po['rms'],
                            **{f'frac_on_{w}': rows[b]['fracs'][w][0] for w in S_WIRINGS},
                            **{f'frac_off_{w}': rows[b]['fracs'][w][1] for w in S_WIRINGS},
                            **{f'fit_frac_on_{w}': base_fracs[odor][w][0] for w in S_WIRINGS},
                            **{f'fit_frac_off_{w}': base_fracs[odor][w][1] for w in S_WIRINGS},
                            log2_factor_rms=float(np.sqrt(np.mean(np.log2(parent) ** 2))), seconds=time.time() - t0)
                _append_csv(cand_path, [dict(half=half, odor=odor, generation=g, candidate=i, obj=r['obj'], sse=r['sse'],
                                             pattern_rms_half=r['rms'], penalty=r['penalty']) for i, r in enumerate(rows)])
                if success:
                    frames = f'o{oi}g{g}'
                    for w in S_WIRINGS:
                        _save_frames(sdir, f'{w}__{frames}', {'on_responses': inc[w]['on'][0], 'on_spike_counts': inc[w]['on'][1],
                                                              'off_responses': inc[w]['off'][0], 'off_spike_counts': inc[w]['off'][1]})
                if g == gens - 1:
                    done[odor] = parent.tolist()
                _write_json(ck, dict(half=half, odor_index=oi, generation=g, frames=frames, parent=parent.tolist(),
                                     parent_obj=parent_obj, sigma=sigma, rng=_rng_state(rng), done=done))
                _drop_other_frames(sdir, frames)
                _append_csv(hist_path, [hrow])
                print(f"half {half} {odor} gen {g}: best {objs[b]:.5f} (parent {parent_obj:.5f}), S's pattern part "
                      f"{ph['rms']:.4f} against this half, {po['rms']:.4f} against the other, sigma {sigma:.2f}, "
                      f"{time.time() - t0:.0f}s", flush=True)
            parent = None
            write_input_tables(out, half, orn, done, odors)
        write_input_tables(out, half, orn, done, odors)
    finally:
        pool.close()
    rows = []
    for w in S_WIRINGS:
        o = {c: inc[w][c] for c in ('on', 'off')}
        X.save_run('C', f'I_half{half}_{w}', o)
        p = parts[w]
        rows.append(dict(arm='I', half=half, wiring=w, tag=f'I_half{half}_{w}', fit_folder=p['folder'], fit_tag=p['tag'],
                         apl_factor=1.0, thr_factor=1.0, n_on=_n(o['on'][0]), target_on=p['target_on'], n_off=_n(o['off'][0]),
                         target_off=p['target_off'], K=p['K'], **p['p'], **X.realism(o)))
    pd.DataFrame(rows).to_csv(out / f'search_runs_I_{half}.csv', index=False)
    print(pd.DataFrame(rows)[['wiring', 'n_on', 'target_on', 'n_off', 'target_off', 'rate_rho_on', 'realism_flag']].to_string(index=False),
          flush=True)
    return pd.read_csv(hist_path)


def write_input_tables(out: Path, half: str, orn: pd.DataFrame, done: Mapping[str, Sequence[float]],
                       odors: Sequence[str] = I_ODORS) -> pd.DataFrame:
    """input_factors_<half>.csv (E77's oracle_factors layout, the odors searched so far, in `odors`' order) and
    inputs/searched_half<half>.csv (the searched input, glomeruli x abbreviated odors, E38's layout; odors not searched
    are the lab's)."""
    ab = [H.odor_abbrev(c) for c in orn.columns]
    now = orn.astype(float).copy()
    rows = []
    for odor in odors:
        if odor not in done:
            continue
        f = np.asarray(done[odor], dtype=float)
        j = ab.index(odor)
        now.iloc[:, j] = orn.iloc[:, j].to_numpy(dtype=float) * f
        for gi, glom in enumerate(orn.index):
            rows.append(dict(half=half, odor=odor, glomerulus=glom, orn_delta=float(orn.iloc[gi, j]), factor=float(f[gi]),
                             log2_factor=float(np.log2(f[gi])), searched_delta=float(orn.iloc[gi, j] * f[gi])))
    fac = pd.DataFrame(rows)
    fac.to_csv(out / f'input_factors_{half}.csv', index=False)
    lay = H.abbrev_columns(now)
    lay.index.name = 'glomerulus'
    (out / 'inputs').mkdir(exist_ok=True)
    lay.to_csv(out / 'inputs' / f'searched_half{half}.csv')
    return fac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--arm', required=True, choices=['P', 'I'], help='P: the profile search (part B); I: the input search (part C)')
    ap.add_argument('--half', required=True, choices=list(HALVES))
    ap.add_argument('--generations', type=int, default=None, help='default 12 (P) / 8 per odor (I)')
    ap.add_argument('--pop', type=int, default=None, help='offspring per generation: default 16 (P) / 32 (I)')
    ap.add_argument('--odors', nargs='+', default=list(I_ODORS), help='arm I: the odors searched, in order')
    ap.add_argument('--threads', type=int, default=I_WORKERS_THREADS, help="arm I: each worker's OpenMP threads")
    ap.add_argument('--no-resume', action='store_true', help='start over (ignore the checkpoint)')
    ap.add_argument('--out', default=None, help='results folder (default results/E83_cross_wiring_fit/ in the shared directory)')
    ap.add_argument('--state-dir', default=None, help='folder of the checkpoints (default outputs/E83_cross_wiring_fit/)')
    args = ap.parse_args()
    out = Path(args.out) if args.out else H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    sd = Path(args.state_dir) if args.state_dir else None
    if args.arm == 'P':
        part_b(out, args.half, gens=args.generations or P_GENS, pop=args.pop or P_POP, resume=not args.no_resume, out_dir=sd)
    else:
        part_c(out, args.half, odors=args.odors, gens=args.generations or I_GENS, pop=args.pop or I_POP, threads=args.threads,
               resume=not args.no_resume, out_dir=sd)


if __name__ == '__main__':
    main()
