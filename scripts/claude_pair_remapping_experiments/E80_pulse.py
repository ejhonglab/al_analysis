#!/usr/bin/env python3
"""E80 part A: the flies' odor pulse in the model. C2 (the breadth-2 APL-effect fit, E67) and E76's fit 0 (the
firing-ceiling fit) with a 0.5 s (identity), 1 s and 2 s odor pulse, the simulation ending 0.25 s after it, spike counts
over the whole simulation: once at C2's parameters, once with both response rates re-held (E77_input_screen.hold_rates;
the breadth from the pre-pass at the same pulse length); the model's APL effect in windows of its own response (0-0.5,
0.5-1, 1-2 s) from saved spike times. Needs mb_model's stimulus-end / simulation-end options (`time_stim_end`,
`time_end`; defaults olfsysm's 0.5 and 0.75 s, outputs unchanged). The design is the E80 entry in
experiment_priorities.md.

The points (`pulse_points`):
- C2 at each pulse, fixed: C2's fitted parameters and C2's own APL->KC weights (its breadth from the bouton base's
  pre-pass at 0.5 s), only the pulse and the simulation's end changed: what the pulse alone does to the two response
  rates and their ratio.
- C2 at each pulse, re-held: the breadth from the bouton base's pre-pass at the same pulse (`handle_breadth` with the
  timing kwargs), then `hold_rates` (the somatic threshold bisected with APL off to C2's APL-off response count, then
  the APL->KC scale with KC->APL and bouton->APL tied to C2's APL-on count, 1852; KC-KC and APL->bouton scales kept).
- E76's fit 0 (results/E76_data_side_target/fits.csv's first row; the bouton base with breadth exponent 2, KC-KC
  inhibition, a 3 ms refractory period and spike-triggered adaptation), at 0.5 s (identity) and 2 s fixed, and at 2 s
  re-held to its own response counts. Its breadth pre-pass has no refractory period or adaptation, as E10_joint_fit /
  E43.fit_model build it.
The 0.5 s points pass the timing kwargs explicitly (olfsysm's defaults) and must reproduce the saved outputs of C2 and
of fit 0 exactly (and the re-held one with factors 1); the run stops otherwise.

Per point: E77_input_screen.Scorer's readings (against the model's own saved outputs at 0.5 s: C2's, or fit 0's),
the largest spike counts, and from the spikes' times (mb_model's `simulate(..., spike_times=True)`) the spike counts in
windows of the response (before onset, 0-0.5, 0.5-1, 1-2 s, 2 s to the end, and the whole simulation, each clipped
to the simulation) with the APL effect on raw counts in each.

Outputs, results/E80_remaining_candidates/: pulse.csv (one row per point; resumable), pulse_windows.csv (per point and
window), pulse_pairs.csv (per point and pair: distances APL on / off and the APL effect, raw, calibrated readout,
binarized), pulse_params.json (each point's absolute parameters, bisection histories, breadth hashes),
pulse_checks.json (`--check`: the last-digit checks of the timing options at their defaults), pulse_provenance.json.
Responses, spike counts and spike times of every point under outputs/E80_remaining_candidates/pulse/ (tag <point_id>,
conditions on / acute_off). The breadth pre-passes are cached by E77_input_screen.handle_breadth under
outputs/E77_input_side_common_error/breadth/.

Usage (from this folder):
    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 nice -n 10 ../../venv/bin/python -u E80_pulse.py [--check] [--only ID ...] [--list]
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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E80_remaining_candidates'
PULSES = (0.5, 1.0, 2.0)
IDENTITY_PULSE = 0.5
# the simulation ends this long after the pulse (olfsysm's default: 0.75 s for its 0.5 s pulse)
TAIL = 0.25
FIT0_FOLDER = 'E76_data_side_target'
KC_SPIKING = ('kc_refractory_period', 'kc_adapt_inc', 'kc_adapt_tau')
# APL off as E77 runs it (APL->KC and APL->bouton scales 0)
OFF = dict(wAPLKC_scale=0.0, wAPLPN_scale=0.0)
# (label, start, end) in seconds from odor onset; None: the simulation's start (olfsysm's time_start, -0.5 s) / end.
# Each is clipped to the simulation, and skipped when that leaves it empty.
WINDOWS = (('pre', None, 0.0), ('0-0.5', 0.0, 0.5), ('0.5-1', 0.5, 1.0), ('1-2', 1.0, 2.0), ('2-end', 2.0, None),
           ('all', None, None))
P1 = 'P1'


def timing_kws(pulse: float) -> Dict[str, float]:
    """mb_model's timing kwargs for a pulse of `pulse` seconds, the simulation ending TAIL after it."""
    return dict(time_stim_end=float(pulse), time_end=float(pulse) + TAIL)


def _n(responses: pd.DataFrame) -> int:
    return int((responses.to_numpy(dtype=float) > 0.5).sum())


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=float), dtype='<f8').tobytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# the two models

_FIT0: Dict[str, object] = {}


def fit0_parts() -> dict:
    """E76's fit 0 (the firing-ceiling fit): 'tag', 'p' (its fitted absolute parameters, E72's `fitted_params`), 'kc'
    (its refractory period and adaptation kwargs), 'saved' ({'on', 'off'} outputs), 'target_on' / 'target_off' (its
    response counts). Checks it is what E77_input_screen.build_c2_simulator builds with those kwargs added: the bouton
    base, no per-claw threshold, glomerulus gain, re-drawn wiring, other ORN input or input-scaled threshold, APL driven
    by the claws, breadth exponent 2, E10's KC-KC tau."""
    if not _FIT0:
        import E10_joint_fit as J
        import E72_wiring_population as E72
        import E77_input_screen as S
        f = pd.read_csv(H.results_dir(FIT0_FOLDER) / 'fits.csv')
        r = f.iloc[0]
        tag = str(r['tag'])
        if not (tag.startswith('nu_') and 'refr3_adapt' in tag):
            raise RuntimeError(f"{FIT0_FOLDER}'s fit 0 is {tag!r}, not the firing-ceiling fit (nu_..._refr3_adapt...)")
        for c in ('claw_thr', 'gain', 'gain_glomeruli', 'wiring', 'wiring_file', 'orn_input', 'kc_thr_input_exp',
                  'claw_thr_subgain', 'claw_thr_width', 'base_params'):
            if c in r.index and not pd.isna(r[c]) and str(r[c]) != '':
                raise RuntimeError(f'fit 0 ({tag}) has {c}={r[c]!r}; build_c2_simulator would not build it')
        if str(r.get('apl_drive', 'claws')) != 'claws' or float(r['k']) != S.BREADTH_K or float(r['kc_kc_tau']) != J.TAU:
            raise RuntimeError(f"fit 0 ({tag}): apl_drive {r.get('apl_drive')}, k {r['k']}, kc_kc_tau {r['kc_kc_tau']}")
        saved = E72.load_fit(FIT0_FOLDER, tag)
        if saved is None:
            raise RuntimeError(f"fit 0's saved outputs ({FIT0_FOLDER}, {tag}) not found")
        _FIT0.update(tag=tag, p=E72.fitted_params(FIT0_FOLDER, tag), kc={c: float(r[c]) for c in KC_SPIKING},
                     saved=saved, target_on=_n(saved['on'][0]), target_off=_n(saved['off'][0]))
    return _FIT0


def model_parts(model: str) -> dict:
    """'C2' or 'fit0': 'p' (the starting absolute parameters), 'extra' (kwargs beside the bouton base's: fit 0's
    refractory period and adaptation), 'saved', 'target_on', 'target_off', 'source' (experiment, tag)."""
    import E77_input_screen as S
    if model == 'C2':
        P = S.c2_parts()
        return dict(p=dict(P['p']), extra={}, saved=P['saved'], target_on=P['target_on'], target_off=P['target_off'],
                    source=S.C2)
    if model == 'fit0':
        F = fit0_parts()
        return dict(p=dict(F['p']), extra=dict(F['kc']), saved=F['saved'], target_on=F['target_on'],
                    target_off=F['target_off'], source=(FIT0_FOLDER, F['tag']))
    raise ValueError(model)


def breadth_for(pulse: Optional[float], breadth: str) -> pd.Series:
    """The breadth behind the APL->KC weights: 'own', C2's (the bouton base's pre-pass at olfsysm's defaults, which fit
    0 shares); 'pulse', the pre-pass with the timing kwargs of `pulse` (`handle_breadth`; no refractory period or
    adaptation in it, as for fit 0's own)."""
    import E77_input_screen as S
    if breadth == 'own' or pulse is None:
        return S.c2_parts()['breadth']
    if breadth == 'pulse':
        return S.handle_breadth(timing_kws(pulse))
    raise ValueError(breadth)


def build(model: str, pulse: Optional[float], breadth: str) -> Tuple[Callable, dict, pd.Series]:
    """(simulate, info, breadth) of `model` with the pulse `pulse` (None: no timing kwargs passed), as
    E77_input_screen.build_c2_simulator builds C2 (with fit 0's kwargs and parameters for 'fit0')."""
    import E77_input_screen as S
    m = model_parts(model)
    b = breadth_for(pulse, breadth)
    extra = dict(m['extra'], **({} if pulse is None else timing_kws(pulse)))
    sim, info = S.build_c2_simulator(extra or None, breadth=b, params=m['p'])
    return sim, info, b


def count_diffs(o: Dict[str, tuple], saved: Dict[str, tuple]) -> dict:
    """Number of differing spike counts and responses per condition between two {'on', 'off'} outputs, and whether
    their KC indices match."""
    d = {}
    for c in ('on', 'off'):
        d[f'{c}_counts_differ'] = int((o[c][1].to_numpy(dtype=float) != saved[c][1].to_numpy(dtype=float)).sum())
        d[f'{c}_responses_differ'] = int((o[c][0].to_numpy(dtype=float) != saved[c][0].to_numpy(dtype=float)).sum())
        d[f'{c}_index_equal'] = bool(o[c][1].index.equals(saved[c][1].index))
    d['exact'] = all(v == 0 for k, v in d.items() if k.endswith('differ')) and all(
        v for k, v in d.items() if k.endswith('index_equal'))
    return d


# ---------------------------------------------------------------------------
# the last-digit checks of the timing options at their defaults

def last_digit_checks(out: Path) -> dict:
    """With the timing options at their defaults (not passed), C2's best evaluation (E67's rate0.068_aplboth_k2_cap3,
    rebuilt as E77 rebuilds it), E72's part-B fit on P1 (E77_input_side_common_error.wiring_simulator, as score_oracle's
    lab check) and E76's fit 0 must reproduce their saved spike counts and responses exactly; so must C2 with the
    options passed at olfsysm's defaults and its breadth from a pre-pass with them (which must equal C2's breadth
    exactly), and C2's runs with `spike_times=True`. Writes pulse_checks.json; raises RuntimeError if any differs."""
    import E72_wiring_population as E72
    import E77_input_screen as S
    import E77_input_side_common_error as E77
    P = S.c2_parts()
    res: Dict[str, dict] = {}
    t0 = time.time()

    sim, info = S.build_c2_simulator(breadth=P['breadth'])
    on, off = sim(dict(P['p'])), sim(dict(P['p'], **OFF))
    res['C2_defaults'] = count_diffs({'on': on, 'off': off}, P['saved'])
    r_on, s_on, st_on = sim(dict(P['p']), spike_times=True)
    r_off, s_off, st_off = sim(dict(P['p'], **OFF), spike_times=True)
    d = count_diffs({'on': (r_on, s_on), 'off': (r_off, s_off)}, P['saved'])
    d['n_spikes_on'], d['n_spikes_off'] = int(len(st_on)), int(len(st_off))
    res['C2_defaults_spike_times'] = d
    # (the simulator keeps olfsysm's recordings allocated after a spike_times run: plain runs again, unchanged)
    res['C2_defaults_after_spike_times'] = count_diffs({'on': sim(dict(P['p'])), 'off': sim(dict(P['p'], **OFF))},
                                                       P['saved'])
    del sim, info
    gc.collect()
    print(f"C2, timing options not passed: {res['C2_defaults']} ({time.time() - t0:.0f} s)", flush=True)

    t1 = time.time()
    b = S.handle_breadth(timing_kws(IDENTITY_PULSE))
    breadth_equal = bool(b.index.equals(P['breadth'].index) and np.array_equal(b.to_numpy(), P['breadth'].to_numpy()))
    sim, info = S.build_c2_simulator(timing_kws(IDENTITY_PULSE), breadth=b)
    d = count_diffs({'on': sim(dict(P['p'])), 'off': sim(dict(P['p'], **OFF))}, P['saved'])
    d['breadth_equal'] = breadth_equal
    d['exact'] = d['exact'] and breadth_equal
    res['C2_explicit_defaults'] = d
    del sim, info
    gc.collect()
    print(f"C2, timing options at olfsysm's defaults (and its breadth pre-pass with them): {d} ({time.time() - t1:.0f} s)",
          flush=True)

    t1 = time.time()
    sim, p, (folder, tag) = E77.wiring_simulator(P1, P['orn'])
    saved = E72.load_fit(folder, tag)
    d = count_diffs({'on': sim(dict(p)), 'off': sim(dict(p, **OFF))}, saved)
    d.update(folder=folder, tag=tag)
    res['P1_defaults'] = d
    del sim
    gc.collect()
    print(f"P1 ({tag}), timing options not passed: {d} ({time.time() - t1:.0f} s)", flush=True)

    t1 = time.time()
    F = fit0_parts()
    sim, info, _ = build('fit0', None, 'own')
    d = count_diffs({'on': sim(dict(F['p'])), 'off': sim(dict(F['p'], **OFF))}, F['saved'])
    d.update(folder=FIT0_FOLDER, tag=F['tag'])
    res['fit0_defaults'] = d
    del sim, info
    gc.collect()
    print(f"fit 0 ({F['tag']}), timing options not passed: {d} ({time.time() - t1:.0f} s)", flush=True)

    import olfsysm
    rec = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E80_pulse.py --check',
               al_analysis_commit=_git_head(), al_analysis_dirty=bool(subprocess.run(
                   ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=H.HERE, capture_output=True,
                   text=True).stdout.strip()),
               olfsysm=olfsysm.build_info(), checks=res, seconds=time.time() - t0,
               all_exact=all(v['exact'] for v in res.values()))
    (out / 'pulse_checks.json').write_text(json.dumps(rec, indent=1, default=str) + '\n')
    if not rec['all_exact']:
        raise RuntimeError(f'last-digit checks failed: {[k for k, v in res.items() if not v["exact"]]}')
    print(f"all last-digit checks exact ({rec['seconds']:.0f} s)", flush=True)
    return rec


# ---------------------------------------------------------------------------
# the points

def pulse_points() -> List[dict]:
    """Part A's points: C2 at pulses 0.5 (identity), 1 and 2 s, fixed (C2's parameters and APL->KC weights) and re-held
    (breadth from the pre-pass at the pulse, both response rates held); fit 0 at 0.5 s (identity) and 2 s fixed, and at
    2 s re-held. Identity points first, per model."""
    pts = []

    def add(model: str, pulse: float, held: bool) -> None:
        pts.append(dict(point_id=f"{model}_pulse{pulse:g}_{'held' if held else 'fixed'}", model=model, pulse=float(pulse),
                        held=bool(held), identity=float(pulse) == IDENTITY_PULSE, breadth='pulse' if held else 'own',
                        **timing_kws(pulse)))

    for pulse in PULSES:
        for held in (False, True):
            add('C2', pulse, held)
    add('fit0', IDENTITY_PULSE, False)
    add('fit0', 2.0, False)
    add('fit0', 2.0, True)
    return pts


def run_point(pt: dict, sim: Callable, info: dict, m: dict) -> Tuple[Dict[str, tuple], dict, Dict[str, pd.DataFrame]]:
    """A point's outputs ({'on', 'off'}: (responses, spike counts)), its run record (as hold_rates', or the fixed
    parameters') and its spikes ({'on', 'off'}: `simulate(..., spike_times=True)`'s third output). For a re-held point
    the final parameters are run again with spike times, which must give hold_rates' outputs exactly."""
    import E77_input_screen as S
    t0 = time.time()
    if not pt['held']:
        p = dict(m['p'])
        r_on, s_on, st_on = sim(p, spike_times=True)
        r_off, s_off, st_off = sim(dict(p, **OFF), spike_times=True)
        res = dict(params=p, params_off=dict(p, **OFF), thr_factor=1.0, apl_factor=1.0, at_bound_thr=False,
                   at_bound_apl=False, n_sims=2, history=[], target_on=m['target_on'], target_off=m['target_off'])
    else:
        res = S.hold_rates(lambda: (sim, info), target_off=m['target_off'], target_on=m['target_on'], params=m['p'],
                           label=pt['point_id'])
        r_on, s_on, st_on = sim(res['params'], spike_times=True)
        r_off, s_off, st_off = sim(res['params_off'], spike_times=True)
        d = count_diffs({'on': (r_on, s_on), 'off': (r_off, s_off)}, {'on': res['on'], 'off': res['off']})
        if not d['exact']:
            raise RuntimeError(f"{pt['point_id']}: the spike-time runs differ from hold_rates' outputs: {d}")
        res = dict(res, n_sims=res['n_sims'] + 2)
    res['seconds'] = time.time() - t0
    return {'on': (r_on, s_on), 'off': (r_off, s_off)}, res, {'on': st_on, 'off': st_off}


def _steps(t: float) -> int:
    import olfsysm as osm
    mp = osm.ModelParams()
    return int(round((t - mp.time_pre_start) / mp.time_dt))


def window_counts(st: pd.DataFrame, like: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """KCs x odors spike counts (`like`'s index and columns) of the spikes `st` at steps lo <= step < hi."""
    w = st[(st.step >= lo) & (st.step < hi)]
    a = np.zeros(like.shape, dtype=float)
    np.add.at(a, (w.kc.to_numpy(), w.odor.to_numpy()), 1.0)
    return pd.DataFrame(a, index=like.index, columns=like.columns)


def window_rows(pid: str, pt: dict, o: Dict[str, tuple], spikes: Dict[str, pd.DataFrame]) -> List[dict]:
    """Per window (WINDOWS, clipped to the simulation), the spike counts and the APL effect on raw counts (cosine
    distances on the window's counts, against the KC data's effect; E72.effect_stats). The 'all' window must give the
    point's spike counts exactly."""
    import olfsysm as osm
    import E72_wiring_population as E72
    mp = osm.ModelParams()
    t_start, t_end = float(mp.time_start), float(pt['time_end'])
    tot = {c: float(o[c][1].to_numpy(dtype=float).sum()) for c in ('on', 'off')}
    rows = []
    for label, a, b in WINDOWS:
        a_ = t_start if a is None else max(float(a), t_start)
        b_ = t_end if b is None else min(float(b), t_end)
        if a_ >= b_:
            continue
        lo, hi = _steps(a_), _steps(b_)
        w = {c: window_counts(spikes[c], o[c][1], lo, hi) for c in ('on', 'off')}
        if label == 'all':
            for c in ('on', 'off'):
                if not np.array_equal(w[c].to_numpy(), o[c][1].to_numpy(dtype=float)):
                    raise RuntimeError(f'{pid}: the spike times do not add up to the {c} spike counts')
        row = dict(point_id=pid, window=label, t0=a_, t1=b_)
        for c in ('on', 'off'):
            x = w[c].to_numpy()
            row[f'n_spikes_{c}'] = int(x.sum())
            row[f'share_spikes_{c}'] = float(x.sum() / tot[c]) if tot[c] > 0 else np.nan
            row[f'frac_1plus_{c}'] = float((x >= 1).mean())
            row[f'frac_2plus_{c}'] = float((x >= 2).mean())
            row[f'max_count_{c}'] = int(x.max())
            row[f'n_odors_silent_{c}'] = int((x.sum(axis=0) == 0).sum())
        try:
            d = {c: H.readout_pair_distances(w[c], 'raw') for c in ('on', 'off')}
            row.update({f'raw_{k}': v for k, v in E72.effect_stats(d, 'raw').items()})
            row.update(mean_dist_on=float(d['on'].mean()), mean_dist_off=float(d['off'].mean()),
                       n_pairs_defined=int((d['on'].notna() & d['off'].notna()).sum()))
            row['note'] = ''
        except Exception as e:
            row['note'] = f'APL effect not computed: {e!r}'[:300]
        rows.append(row)
    return rows


def count_stats(o: Dict[str, tuple]) -> dict:
    """The largest spike counts: per condition the maximum, the 99th percentile and the mean over responding KC x odor
    entries (two or more spikes), and the number of entries with 10 or more spikes."""
    row = {}
    for c in ('on', 'off'):
        x = o[c][1].to_numpy(dtype=float)
        resp = x[o[c][0].to_numpy(dtype=float) > 0.5]
        row[f'max_count_{c}'] = int(x.max())
        row[f'p99_count_resp_{c}'] = float(np.percentile(resp, 99)) if resp.size else np.nan
        row[f'mean_count_resp_{c}'] = float(resp.mean()) if resp.size else np.nan
        row[f'n_count_10plus_{c}'] = int((x >= 10).sum())
    return row


# ---------------------------------------------------------------------------
# the run

def _save_outputs(pid: str, o: Dict[str, tuple], spikes: Dict[str, pd.DataFrame]) -> None:
    d = H.outputs_dir(EXPERIMENT) / 'pulse'
    d.mkdir(exist_ok=True)
    for c, cc in (('on', 'on'), ('off', 'acute_off')):
        o[c][0].to_parquet(d / f'{pid}_{cc}_responses.parquet')
        o[c][1].to_parquet(d / f'{pid}_{cc}_spike_counts.parquet')
        spikes[c].to_parquet(d / f'{pid}_{cc}_spike_times.parquet')


def write_provenance(out: Path, points: Sequence[dict]) -> None:
    """pulse_provenance.json: the model's provenance (`H.model_provenance`, on the lab's input and C2's wiring), the two
    models' parameters and sources, the points, the windows, the brackets of hold_rates."""
    import E77_input_screen as S
    P = S.c2_parts()
    F = fit0_parts()
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E80_pulse.py',
                al_analysis_commit=_git_head(), model=H.model_provenance(orn_deltas=P['orn'], _wPNKC=P['W']),
                c2=dict(experiment=S.C2[0], tag=S.C2[1], params=P['p'], target_on=P['target_on'],
                        target_off=P['target_off']),
                fit0=dict(experiment=FIT0_FOLDER, tag=F['tag'], params=F['p'], kc_spiking=F['kc'],
                          target_on=F['target_on'], target_off=F['target_off']),
                bouton_base_params=P['init'], breadth_k=S.BREADTH_K, tied_on=S.TIED_ON, thr_bracket=S.THR_BRACKET,
                thr_step=S.THR_STEP, apl_bracket=S.APL_BRACKET, apl_step=S.APL_STEP, tail=TAIL,
                windows=[list(w) for w in WINDOWS], points=list(points), approx_conc_note=H.APPROX_CONC_NOTE,
                breadth=("'own': C2's breadth (the bouton base's APL-off pre-pass at olfsysm's 0.5 / 0.75 s); 'pulse': "
                         'the same pre-pass with the timing kwargs (E77_input_screen.handle_breadth)'),
                scoring=('E77_input_screen.Scorer against the model\'s own saved outputs at 0.5 s (C2\'s for C2, fit '
                         "0's for fit 0); windows: raw-count cosine distances on the spikes in each window"))
    (out / 'pulse_provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')


def _filtered_csv(path: Path, done: set) -> None:
    """Drops rows of points not in `done` (left by an interrupted point) from a per-point table."""
    if path.exists():
        df = pd.read_csv(path)
        keep = df[df.point_id.isin(done)]
        if len(keep) < len(df):
            keep.to_csv(path, index=False)


def run_pulse(out: Path, only: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Runs part A (resumable); writes pulse.csv, pulse_windows.csv, pulse_pairs.csv and pulse_params.json.

    Args:
        out: the results folder (results/E80_remaining_candidates/).
        only: point ids to run (default all); the identity point of each of their models runs first if not done.

    Returns pulse.csv's rows. An identity point that does not reproduce its model's saved outputs exactly stops the run
    (RuntimeError).
    """
    import E77_input_screen as S
    out.mkdir(parents=True, exist_ok=True)
    pts = pulse_points()
    ids = [p['point_id'] for p in pts]
    want = set(ids) if only is None else set(only)
    unknown = want - set(ids)
    if unknown:
        raise ValueError(f'unknown points {sorted(unknown)}')
    models = {p['model'] for p in pts if p['point_id'] in want}
    want |= {p['point_id'] for p in pts if p['identity'] and p['model'] in models}
    write_provenance(out, pts)

    sp, wp, pp, jp = (out / f for f in ('pulse.csv', 'pulse_windows.csv', 'pulse_pairs.csv', 'pulse_params.json'))
    rows = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
    done = set(rows.point_id) if len(rows) else set()
    for f in (wp, pp):
        _filtered_csv(f, done)
    params = json.loads(jp.read_text()) if jp.exists() else {}
    scorers: Dict[str, object] = {}
    cur: Dict[str, object] = {}
    try:
        for pt in pts:
            pid = pt['point_id']
            if pid not in want or pid in done:
                continue
            if not pt['identity']:
                ident = next(q['point_id'] for q in pts if q['identity'] and q['model'] == pt['model'])
                if ident not in done:
                    print(f'{pid}: its identity point {ident} has not passed; skipped', flush=True)
                    continue
            print(f"--- {pid} (pulse {pt['pulse']:g} s, simulation to {pt['time_end']:g} s, "
                  f"{'both response rates re-held' if pt['held'] else 'fixed parameters'}, breadth {pt['breadth']})",
                  flush=True)
            m = model_parts(pt['model'])
            key = (pt['model'], pt['pulse'], pt['breadth'])
            if cur.get('key') != key:
                cur.clear()
                gc.collect()
                t0 = time.time()
                sim, info, b = build(*key)
                cur.update(key=key, sim=sim, info=info, breadth=b)
                del sim, info, b
                print(f'  built in {time.time() - t0:.0f} s', flush=True)
            o, res, spikes = run_point(pt, cur['sim'], cur['info'], m)
            if pt['identity']:
                d = count_diffs(o, m['saved'])
                if not d['exact'] or res['thr_factor'] != 1.0 or res['apl_factor'] != 1.0:
                    raise RuntimeError(f"{pid}: the identity point differs from {m['source']}'s saved outputs ({d}; "
                                       f"factors {res['thr_factor']}, {res['apl_factor']})")
                if pt['breadth'] == 'pulse' and not np.array_equal(cur['breadth'].to_numpy(),
                                                                   S.c2_parts()['breadth'].to_numpy()):
                    raise RuntimeError(f"{pid}: the breadth pre-pass at {pt['pulse']} s differs from C2's breadth")
                print(f"{pid}: reproduces {m['source'][1]}'s saved APL-on and APL-off outputs exactly", flush=True)
            if pt['model'] not in scorers:
                scorers[pt['model']] = S.Scorer(m['saved'])
            t1 = time.time()
            row, pairs = scorers[pt['model']].score(o)
            wrows = window_rows(pid, pt, o, spikes)
            rp = res['params']
            meta = dict(point_id=pid, model=pt['model'], source=f"{m['source'][0]}:{m['source'][1]}", pulse=pt['pulse'],
                        time_stim_end=pt['time_stim_end'], time_end=pt['time_end'], held=pt['held'],
                        identity=pt['identity'], breadth=pt['breadth'], breadth_sha256=_sha(cur['breadth'].to_numpy()),
                        fixed_thr=rp['fixed_thr'], fixed_thr_off=res['params_off']['fixed_thr'],
                        **{k: rp[k] for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale',
                                              'wKCKC_scale')},
                        **{k: v for k, v in m['extra'].items()}, thr_factor=res['thr_factor'],
                        apl_factor=res['apl_factor'], n_on=_n(o['on'][0]), n_off=_n(o['off'][0]),
                        target_on=m['target_on'], target_off=m['target_off'], at_bound_thr=res['at_bound_thr'],
                        at_bound_apl=res['at_bound_apl'], n_sims=res['n_sims'],
                        seconds=res['seconds'] + (time.time() - t1), **count_stats(o))
            # (Scorer's max_count_* equal count_stats'; kept once)
            row = {k: v for k, v in row.items() if k not in meta}
            pd.DataFrame(wrows).to_csv(wp, index=False, mode='a', header=not wp.exists())
            pairs.insert(0, 'point_id', pid)
            pairs.to_csv(pp, index=False, mode='a', header=not pp.exists())
            params[pid] = dict(pt, source=list(m['source']), params=rp, params_off=res['params_off'],
                               thr_factor=res['thr_factor'], apl_factor=res['apl_factor'], n_on=meta['n_on'],
                               n_off=meta['n_off'], history=res.get('history', []), breadth_sha256=meta['breadth_sha256'],
                               kc_spiking=m['extra'], sim_kws=cur['info']['kws'])
            jp.write_text(json.dumps(params, indent=1, default=str) + '\n')
            _save_outputs(pid, o, spikes)
            new = dict(meta, **row)
            rows = pd.concat([rows, pd.DataFrame([new])], ignore_index=True) if len(rows) else pd.DataFrame([new])
            rows.to_csv(sp, index=False)
            done.add(pid)
            r = rows.iloc[-1]
            win = {w['window']: w for w in wrows}
            print(f"{pid}: thr x{r.thr_factor:.4f}, APL x{r.apl_factor:.4f}; response rates {r.rate_on:.4f} / "
                  f"{r.rate_off:.4f} (ratio {r.ratio:.3f}; {r.n_on} / {r.n_off} responses); largest counts "
                  f"{r.max_count_on} / {r.max_count_off}; readout loss {r.cal_loss:.4f} (mean {r.cal_mean:+.3f}, pattern "
                  f"{r.cal_pattern:.4f}), raw {r.raw_loss:.4f} (mean {r.raw_mean:+.3f}, pattern {r.raw_pattern:.4f}); "
                  'windows (raw mean effect): ' + ', '.join(f"{k} {v.get('raw_mean', np.nan):+.3f}" for k, v in win.items())
                  + f"; realism: {r.realism_flag if isinstance(r.realism_flag, str) and r.realism_flag else 'inside the bands'}",
                  flush=True)
            del o, res, spikes
            gc.collect()
    finally:
        cur.clear()
        gc.collect()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true', help='the last-digit checks of the timing options at their '
                    'defaults (C2, P1, fit 0), then exit')
    ap.add_argument('--only', nargs='+', help="point ids to run (their models' identity points run first)")
    ap.add_argument('--list', action='store_true', help='list the points and exit')
    args = ap.parse_args()
    if args.list:
        for p in pulse_points():
            print(p['point_id'], p['model'], p['pulse'], 'held' if p['held'] else 'fixed', p['breadth'])
        return
    out = H.results_dir(EXPERIMENT)
    if args.check:
        last_digit_checks(out)
        return
    run_pulse(out, only=args.only)


if __name__ == '__main__':
    main()
