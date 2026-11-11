#!/usr/bin/env python3
"""E82: the flies' recording protocol in the model. The design is the E82 entry in experiment_priorities.md (planned by a
Fable subagent, claude-fable-5-2, spawned from viki 2026-11-11 02:48).

- (A) No simulations: saved runs with spike times (E80's at 0.5, 1 and 2 s; then parts B and D's) read through an
  indicator kernel (1 - exp(-t/rise)) exp(-t/decay) over the flies' window (2.0 to 8.1 s after onset), rise 0.2 / 0.5 / 1
  s by decay 1 / 3 / 5 s (primary: rise 0.5, decay 3), with and without E65's saturation; the late spikes by condition,
  250 ms bin, odor, KC type and claw count; the loss with the APL-off spikes after 0.5 s left out; the model's predicted
  traces. `kc_remapping_harness.kernel_weighted_counts`.
- (B) the runs at the flies' pulse: E82_protocol_runs.py.
- (C) the protocol gate and E77's gate.
- (D) fits at the flies' pulse (E10_joint_fit.py --stim-end / --sim-end / --count-kernel).

The readings (`protocol_readings`). Each spike of a run is weighted by the area of its kernel inside the window, divided
by the area of a spike at the window's start (the end of the flies' 2 s pulse), so the weights are in units of a spike at
the end of the flies' pulse whatever the model's pulse (a spike 50 ms after onset weighs 0.86, 0.62 and 0.17 at rise /
decay 1 / 5, 0.5 / 3 and 0.2 / 1 s); the KC x odor sums of the weights replace the spike counts. Two identity readings
beside the kernels: 'counts' (the spike counts themselves) and 'step' (rise 0, no decay: every spike before the window
weighs exactly 1, so for runs ending before 2 s the step reading is the counts' to the last digit, which is checked).
Which KCs respond stays the two-spike count over the whole simulation under every reading (the saved responses), so the
response rates and the four bands do not move with the kernel. Per reading:
- raw: cosine distances on the weighted counts;
- cal: E65's saturating readout x / (x + K), K calibrated on the weighted APL-on values of the APL-on responders to the
  control flies' responders-only participation ratio (E67.refs()['cal_target']), as E72_wiring_population.distances
  calibrates it on counts (there `responders_min=2` on the counts; here the responders are the responses, which are the
  same KCs on counts and stay the same KCs under a kernel, where a weighted sum of two spikes can be below 2), and used
  unchanged for APL off;
- cal_noise: the same with K calibrated to the noise-unit target (E80_screen.noise_scores' readout).
Each is scored as E77_input_screen.Scorer scores a point (E72's effect statistics: the APL-effect loss with its mean,
pattern, per-odor and pair-specific parts and ρ on the per-pair APL effect, against Yang's z over all eight TNT recordings;
E75's comparison with the common error: the ester and alcohol group errors and the projection; E77's kinds of ester
pairs and the projection against the six typical TNT recordings), plus the same effect statistics against the six
typical recordings (`_z6_`) and against the flies' effect in baseline-noise units (E76's effect_by_unit_pairs.csv,
`_noise_`, as E80_screen.noise_scores), and the six pairs with near-identical ORN input (E78's SIX: mean distance APL on
/ off and mean APL effect).

The late spikes (`late_spikes`, `late_leave_out`): spikes at or after 0.5 s from onset (olfsysm's step of 0.5 s on),
per condition: totals, the share in each 250 ms bin from onset, per odor, per KC type and per claw count (the real
wiring's claws per KC, which the re-drawn wirings keep), and the loss with the APL-off late spikes left out: all of
them, odor by odor and cumulatively in descending order of the odors' late APL-off spikes; on counts and under the
primary reading (the responses kept).

The predicted traces (`predicted_traces`): per run, kernel and condition the responders' mean kernel-convolved signal
(the sum over the spikes of a responding KC x odor entry of the kernel, in the kernel's own units, peak below 1;
averaged over the condition's responding entries) and the population mean (over all KC x odor entries) from onset to
8.1 s; and the APL-off / APL-on ratio of their means over 0 to 2, 2 to 4 and 4 to 8.1 s (predicted_ratios.csv).

The populations (`population_readings`, part C): part B1's two populations of four wirings (runs.csv's `population`,
'b2' and 'fc') at each pulse, the mean over the wirings of the pair distances per condition and readout (K per wiring,
as E72.population_row), under every reading.

The gates (`gates`, part C), on part B's points (runs.csv, written by E82_protocol_runs.py; the arm B1 to B5 from its
`part` column or the point id's prefix, the model from `model`, the reference from `reference`, the model's B1 point),
read from protocol_readings.csv:
- the protocol gate, for points of B2 to B5 on the firing-ceiling fit (model 'fit0', E76's fit 0): inside the four
  bands (the rule since 2026-11-10 16:10); the raw loss on counts at 2 s and under the primary reading within 0.006 of
  the fit's at the 0.5 s pulse (fit0_pulse0.5_fixed's counts reading, 0.050); the largest spike counts not above the
  fit's re-held 2 s point's (10 / 24);
- E77's gate, for every point of B2 to B5, against the model's own re-held 2 s point (the reference): under the primary
  reading the readout's pattern part lowered by 0.003 or more with the raw loss no more than 0.005 above the
  reference's, and a transfer reduction of the common error (of the change of the per-pair effect under the readout
  from the reference's) of 15 % or more, against the six typical recordings at least half that; inside the bands.

Outputs, results/E82_flies_protocol/: protocol_readings.csv (one row per run and reading), protocol_pairs.csv (per run,
reading, readout and pair: distances APL on / off and the APL effect), protocol_checks.json (the identity checks),
late_spikes.csv, late_leave_out.csv, predicted_traces.csv, predicted_ratios.csv, protocol_table.md (part A's tables,
from E80's runs), E82_partA.png, protocol_provenance_<part>.json; protocol_populations.csv and gate.csv (part C, which
reads every saved run again, E80's and part B's, and redoes the late spikes and the traces with them).

Usage (from this folder; no simulations, single-threaded):
    OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 ../../venv/bin/python -u E82_flies_protocol.py --part {A,C,tables}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E82_flies_protocol'
E80 = 'E80_remaining_candidates'
KERNELS = [(r, d) for r in (0.2, 0.5, 1.0) for d in (1.0, 3.0, 5.0)]
PRIMARY = (0.5, 3.0)
WINDOW = (2.0, 8.1)
# the planner's three kernels (rise with decay): decay 5 / 3 / 1 s at rise 1 / 0.5 / 0.2 s
DIAGONAL = ((1.0, 5.0), (0.5, 3.0), (0.2, 1.0))
# the identity readings: the spike counts, and the step kernel (rise 0, no decay)
COUNTS = 'counts'
STEP = (0.0, np.inf)
CONDS = (('on', 'on'), ('off', 'acute_off'))
LATE = 0.5
BIN = 0.25
TRACE_DT = 0.05
TRACE_END = WINDOW[1]
TRACE_WINDOWS = ((0.0, 2.0), (2.0, 4.0), (4.0, 8.1))
# E78's six pairs with near-identical ORN input (an ORN distance below 0.25)
SIX = (('eb', 'ep'), ('6al', 't2h'), ('IaA', 'pa'), ('2h', 'pa'), ('1-5ol', '1-6ol'), ('2h', 'IaA'))
# the claw-count groups of the input-scaled threshold experiment (E71.CLAW_GROUPS)
CLAW_GROUPS = (('1-4', 1, 4), ('5-6', 5, 6), ('7-8', 7, 8), ('9+', 9, 10 ** 6))
WIRING = H.shared_dir() / 'data' / 'model_outputs' / 'inputs' / 'wPNKC_bouton_base.parquet'
E76_DATA_UNITS = 'results/E76_data_side_target/data_units.csv'
# part C
FIRING_CEILING = 'fit0'
# (source, point id) in saved_runs' labels: E80's saved points of the firing-ceiling fit
FIT0_PULSE05 = ('E80', 'fit0_pulse0.5_fixed')
FIT0_PULSE2 = ('E80', 'fit0_pulse2_held')
PROTOCOL_SLACK = 0.006
GATED_ARMS = ('B2', 'B3', 'B4', 'B5')
# the readings' statistics checked against E80's pulse.csv (E77's Scorer) for the counts reading of E80's runs
CHECK_COLS = ('raw_loss', 'raw_mean', 'raw_pattern', 'raw_per_odor', 'raw_pair_specific', 'raw_rho', 'raw_corr_common',
              'raw_proj_common', 'raw_err_ester', 'raw_err_alcohol', 'raw_err_ms', 'raw_err_nearest', 'raw_err_block',
              'raw_proj_common_six', 'cal_loss', 'cal_mean', 'cal_pattern', 'cal_per_odor', 'cal_pair_specific', 'cal_rho',
              'cal_corr_common', 'cal_proj_common', 'cal_err_ester', 'cal_err_alcohol', 'cal_err_ms', 'cal_err_nearest',
              'cal_err_block', 'cal_proj_common_six', 'rate_on', 'rate_off', 'ratio', 'rate_rho_on', 'rate_rho_off',
              'rate_rho_on_zcrit', 'opk_on_tv', 'opk_off_tv', 'opk_on_tv_zcrit', 'opk_off_tv_zcrit', 'max_count_on',
              'max_count_off')


def klabel(k) -> str:
    """A reading's label: 'counts', 'step' or 'r<rise>_d<decay>'."""
    if k is None or k == COUNTS:
        return COUNTS
    r, d = k
    if r == 0 and np.isinf(d):
        return 'step'
    return f'r{r:g}_d{d:g}'


def readings() -> List[Tuple[str, Optional[Tuple[float, float]]]]:
    """(label, kernel) of every reading: the counts, the step kernel, then KERNELS."""
    return [(COUNTS, None), ('step', STEP)] + [(klabel(k), k) for k in KERNELS]


def _git(*args) -> str:
    return subprocess.run(['git', *args], cwd=H.HERE, capture_output=True, text=True).stdout.strip()


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _canon(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.MultiIndex.from_tuples([tuple(sorted(x)) for x in s.index], names=['odor_a', 'odor_b'])
    return s.sort_index()


# ---------------------------------------------------------------------------
# the saved runs

def _steps(t: float) -> int:
    """olfsysm's step of time `t` (s from onset), as E80_pulse._steps."""
    import E80_pulse as P
    return P._steps(t)


def _dt() -> float:
    import olfsysm as osm
    return float(osm.ModelParams().time_dt)


def _run_dirs(folder: str, sub: str) -> List[Path]:
    """The outputs dir's `sub`, then the archive's."""
    return [H.outputs_dir(folder) / sub, H.shared_dir() / 'data' / 'model_outputs' / folder / sub]


def _files(d: Path, pid: str) -> Dict[str, Path]:
    return {f'{c}_{kind}': d / f'{pid}_{cc}_{kind}.parquet' for c, cc in CONDS
            for kind in ('responses', 'spike_counts', 'spike_times')}


def _find(dirs: Sequence[Path], pid: str) -> Optional[Path]:
    for d in dirs:
        if all(p.exists() for p in _files(d, pid).values()):
            return d
    return None


def _arm(pid: str, m: dict) -> str:
    for c in ('arm', 'handle', 'block', 'part', 'group'):
        v = m.get(c)
        if isinstance(v, str) and re.match(r'^B[1-5]', v):
            return v[:2]
    g = re.match(r'^(B[1-5])', pid)
    return g.group(1) if g else ''


def saved_runs() -> pd.DataFrame:
    """Every saved run with spike times: E80's pulse points (results/E80_remaining_candidates/pulse.csv; files in
    outputs/E80_remaining_candidates/pulse/, else the archive), then part B's (outputs/E82_flies_protocol/runs/, metadata
    from results/E82_flies_protocol/runs.csv where it lists them) and part D's (outputs/E82_flies_protocol/fits/, by the
    same file naming, `<id>_<on|acute_off>_<responses|spike_counts|spike_times>.parquet`). Columns: source, point_id,
    model, pulse, time_stim_end, held, identity, arm, ref_point, dir."""
    rows = []
    p = pd.read_csv(H.results_dir(E80) / 'pulse.csv').drop_duplicates('point_id', keep='last')
    dirs = _run_dirs(E80, 'pulse')
    for r in p.itertuples():
        d = _find(dirs, r.point_id)
        if d is None:
            raise RuntimeError(f"E80's pulse point {r.point_id}: no spike times in {[str(x) for x in dirs]}")
        rows.append(dict(source='E80', point_id=r.point_id, model=r.model, pulse=float(r.pulse),
                         time_stim_end=float(r.time_stim_end), held=bool(r.held), identity=bool(r.identity), arm='',
                         ref_point='', dir=str(d)))
    for source, sub in (('E82B', 'runs'), ('E82D', 'fits')):
        dirs = _run_dirs(EXPERIMENT, sub)
        pids = set()
        for d in dirs:
            if d.exists():
                pids |= {f.name[:-len('_on_spike_times.parquet')] for f in d.glob('*_on_spike_times.parquet')}
        if not pids:
            continue
        meta = {}
        mp = H.results_dir(EXPERIMENT) / ('runs.csv' if sub == 'runs' else 'fits.csv')
        if mp.exists():
            m = pd.read_csv(mp)
            key = 'point_id' if 'point_id' in m.columns else ('tag' if 'tag' in m.columns else None)
            if key is not None:
                meta = {str(r[key]): r.to_dict() for _, r in m.drop_duplicates(key, keep='last').iterrows()}
        for pid in sorted(pids):
            d = _find(dirs, pid)
            if d is None:
                continue
            m = meta.get(pid, {})
            pulse = m.get('pulse', m.get('time_stim_end', np.nan))
            rows.append(dict(source=source, point_id=pid, model=str(m.get('model', '')), pulse=float(pulse),
                             time_stim_end=float(m.get('time_stim_end', pulse)), held=bool(m.get('held', False)),
                             identity=bool(m.get('identity', False)), arm=_arm(pid, m),
                             ref_point=next((m[c] for c in ('ref_point', 'reference') if isinstance(m.get(c), str)
                                             and m[c]), ''), dir=str(d)))
    return pd.DataFrame(rows)


def load_run(d: Path, pid: str) -> Tuple[Dict[str, tuple], Dict[str, pd.DataFrame]]:
    """({'on', 'off'}: (responses, spike counts), {'on', 'off'}: spike times) of one saved run. The spike times must add up
    to the spike counts exactly (RuntimeError otherwise)."""
    f = _files(Path(d), pid)
    o, st = {}, {}
    for c, _ in CONDS:
        r, s = pd.read_parquet(f[f'{c}_responses']), pd.read_parquet(f[f'{c}_spike_counts'])
        t = pd.read_parquet(f[f'{c}_spike_times'])
        a = np.zeros(s.shape, dtype=float)
        np.add.at(a, (t.kc.to_numpy(), t.odor.to_numpy()), 1.0)
        if not np.array_equal(a, s.to_numpy(dtype=float)):
            raise RuntimeError(f'{pid} ({c}): the spike times do not add up to the spike counts')
        if not (r.index.equals(s.index) and r.columns.equals(s.columns)):
            raise RuntimeError(f'{pid} ({c}): responses and spike counts have different labels')
        o[c], st[c] = (r, s), t
    return o, st


def weighted(o: Dict[str, tuple], st: Dict[str, pd.DataFrame], k) -> Dict[str, pd.DataFrame]:
    """{'on', 'off'} amplitudes of a run under reading `k` (None: the spike counts; else (rise, decay))."""
    if k is None or k == COUNTS:
        return {c: o[c][1].astype(float) for c in o}
    return {c: H.kernel_weighted_counts(st[c], o[c][1].index, o[c][1].columns, k[0], k[1], window=WINDOW) for c in o}


# ---------------------------------------------------------------------------
# scoring

_DATA: Dict[str, object] = {}


def data() -> dict:
    """The data side, once per process: 'z8' (the per-pair APL effect in Yang's z, all recordings), 'z6' (the same with
    the six typical TNT recordings, as E77's Scorer), 'noise' (E80_screen.noise_data: the effect in baseline-noise units
    and its calibration target), 'cal_target', 'common' / 'c6' (per readout, the common error and its version against
    the six typical recordings, canonical pairs), 'units' (E76's data_units.csv rows for z and noise, graded)."""
    if not _DATA:
        import E72_wiring_population as E72
        import E75_glomerulus_apl_profile as E75
        import E80_screen as S80
        R = E72.refs()
        ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
        tnt = H.yang_recording_rdms_values('TNT', 'graded')
        keep = [r for r in tnt.index if (str(r[0]), int(r[1]), str(r[2])) not in E75.STRONG_TNT]
        if len(keep) != len(tnt) - 2:
            raise RuntimeError(f'expected to drop 2 of {len(tnt)} TNT recordings, kept {len(keep)}')
        u = pd.read_csv(H.shared_dir() / E76_DATA_UNITS)
        u = u[u['values'] == 'graded'].set_index('unit')
        z6 = ctrl.mean() - tnt.loc[keep].mean()
        N = S80.noise_data()
        # (the decompositions align pairs in the harness's canonical order, the projections in E75's sorted one)
        _DATA.update(z8=H.canonical_pairs(R['graded']), z6=_canon(z6), z6h=H.canonical_pairs(z6), noise=N,
                     noise_h=H.canonical_pairs(N['effect']), cal_target=float(R['cal_target']),
                     common={ro: _canon(E75.common_error(ro)) for ro in ('raw', 'cal')},
                     c6={ro: E75.six_target(E75.population_effect(ro)) for ro in ('raw', 'cal')},
                     units={k: u.loc[k].to_dict() for k in ('z', 'noise')},
                     units_sha256=_sha_file(H.shared_dir() / E76_DATA_UNITS))
    return _DATA


def calibrate(x_on: pd.DataFrame, resp_on: pd.DataFrame, target: float) -> Tuple[float, bool]:
    """K of the saturating readout on the APL-on responders' amplitudes (`resp_on`, the two-spike responses), and whether
    it is at a bound (the readout misses the target by more than 2 %, as E67.full_row's K_at_bound)."""
    r = x_on.where(resp_on.to_numpy(dtype=float) > 0.5, 0.0)
    K = H.calibrate_readout(r, 'saturating', target, stat='participation_ratio')
    got = H.readout_stat(r, 'saturating', K, 'participation_ratio')
    return float(K), bool(abs(got - target) > 0.02 * target)


def _decomp(d: Dict[str, pd.Series], target: pd.Series, pre: str) -> dict:
    dec, _ = H.apl_effect_decomposition(d['on'], d['off'], target)
    return {f'{pre}loss': dec['apl_effect_rmse'], f'{pre}mean': dec['apl_effect_mean'],
            f'{pre}mean_error': dec['apl_effect_mean_error'], f'{pre}pattern': dec['apl_effect_rmse_centered'],
            f'{pre}per_odor': dec['apl_effect_per_odor'], f'{pre}pair_specific': dec['apl_effect_pair_specific'],
            f'{pre}rho': dec['apl_effect_spearman'], f'{pre}n_pairs': dec['apl_effect_n_pairs']}


def score_distances(dist: Dict[str, Dict[str, pd.Series]], full: bool = True) -> dict:
    """The statistics of one reading's pair distances (see the module docstring): `dist` has 'raw' and 'cal' (and
    'cal_noise' when `full`), each {'on', 'off'} pair distances in any pair order."""
    import E72_wiring_population as E72
    import E75_glomerulus_apl_profile as E75
    import E77_input_screen as S
    D = data()
    row = {}
    six = [tuple(sorted(p)) for p in SIX]
    for ro in ('raw', 'cal'):
        d = dist[ro]
        row.update(_decomp(d, D['z8'], f'{ro}_'))
        row.update({f'{ro}_{k}': v for k, v in E75.error_vs_common(E72.pattern_error(d, ro), ro).items()})
        err = _canon(E72.pattern_error(d, ro))
        for kind, pairs in S.PAIR_KINDS.items():
            row[f'{ro}_err_{kind}'] = float(err.reindex(list(pairs)).mean())
        eff = _canon(H.apl_effect(d['on'], d['off']))
        c6 = D['c6'][ro]
        e, z6 = eff.reindex(c6.index), D['z6'].reindex(c6.index)
        e6 = (e - e.mean()) - (z6 - z6.mean())
        cv = c6.to_numpy(dtype=float)
        row[f'{ro}_proj_common_six'] = float(e6.to_numpy(dtype=float) @ cv / (cv @ cv))
        on, off = _canon(d['on']), _canon(d['off'])
        row.update({f'{ro}_near6_on': float(on.reindex(six).mean()), f'{ro}_near6_off': float(off.reindex(six).mean()),
                    f'{ro}_near6_effect': float(eff.reindex(six).mean()), f'{ro}_mean_dist_on': float(on.mean()),
                    f'{ro}_mean_dist_off': float(off.mean())})
        if full:
            row.update(_decomp(d, D['z6h'], f'{ro}_z6_'))
            if ro == 'raw':
                row.update(_decomp(d, D['noise_h'], 'raw_noise_'))
    if full:
        row.update(_decomp(dist['cal_noise'], D['noise_h'], 'cal_noise_'))
    return row


def score_reading(resp: Dict[str, pd.DataFrame], x: Dict[str, pd.DataFrame], full: bool = True,
                  K: Optional[Tuple[float, bool]] = None) -> Tuple[dict, pd.DataFrame]:
    """One reading's statistics (`score_distances`; the readout's K calibrated here) and its per-pair distances.

    Args:
        resp: {'on', 'off'} responses (two or more spikes over the whole simulation).
        x: {'on', 'off'} amplitudes (spike counts, or their kernel-weighted sums), same labels.
        full: False for the raw and cal readouts against Yang's z only, with the group errors (the leave-out).
        K: the readout's (K, at_bound) to use instead of calibrating it (the leave-out keeps the APL-on values).

    Returns (row, pairs): pairs has odor_a, odor_b, readout, on, off, effect (pairs with their odors sorted, as E77's
    Scorer writes them).
    """
    D = data()
    K, K_bound = calibrate(x['on'], resp['on'], D['cal_target']) if K is None else K
    row = dict(K_cal=K, K_cal_at_bound=K_bound)
    dist = {'raw': {c: H.readout_pair_distances(x[c], 'raw') for c in ('on', 'off')},
            'cal': {c: H.readout_pair_distances(x[c], 'saturating', K) for c in ('on', 'off')}}
    if full:
        Kn, Kn_bound = calibrate(x['on'], resp['on'], D['noise']['cal_target'])
        row.update(K_noise=Kn, K_noise_at_bound=Kn_bound)
        dist['cal_noise'] = {c: H.readout_pair_distances(x[c], 'saturating', Kn) for c in ('on', 'off')}
    row.update(score_distances(dist, full))
    cols = []
    for ro, d in dist.items():
        on, off = _canon(d['on']), _canon(d['off'])
        cols.append(pd.DataFrame({'readout': ro, 'on': on, 'off': off, 'effect': on - off.reindex(on.index)}))
    pairs = pd.concat(cols).rename_axis(['odor_a', 'odor_b']).reset_index()
    return row, pairs


def band_row(o: Dict[str, tuple]) -> dict:
    """The response statistics of a run, as E77's Scorer computes them (the same under every reading): the response
    rates and their ratio, odors per KC against the flies (both rules), the response rate ρ, the four bands, the
    largest spike counts."""
    import apl_effect_scores as A
    rate_on = float(o['on'][0].to_numpy(dtype=float).mean())
    rate_off = float(o['off'][0].to_numpy(dtype=float).mean())
    row = dict(rate_on=rate_on, rate_off=rate_off, ratio=rate_off / rate_on)
    for c in ('on', 'off'):
        d = A.odors_per_kc(o[c][0])
        row[f'opk_{c}_tv'] = float(0.5 * np.abs(d - A.opk_data(c, 'noise')).sum())
        row[f'opk_{c}_tv_zcrit'] = float(0.5 * np.abs(d - A.opk_data(c, 'z')).sum())
        per_odor = H.abbrev_columns(o[c][0].astype(float) > 0.5).mean().reindex(H.APL16_ODORS)
        row[f'rate_rho_{c}'] = H.spearman(per_odor, A.frac_data(c, 'noise'))
        row[f'rate_rho_{c}_zcrit'] = H.spearman(per_odor, A.frac_data(c, 'z'))
        row[f'max_count_{c}'] = int(o[c][1].to_numpy(dtype=float).max())
        row[f'n_spikes_{c}'] = int(o[c][1].to_numpy(dtype=float).sum())
        row[f'responses_are_2plus_{c}'] = bool(np.array_equal(o[c][0].to_numpy(dtype=float) > 0.5,
                                                              o[c][1].to_numpy(dtype=float) >= 2))
    row['realism_flag'] = A.realism_flag(rate_on, rate_off, row['rate_rho_on'], row['opk_on_tv'], row['opk_off_tv'],
                                         units='noise')
    row['realism_flag_zcrit'] = A.realism_flag(rate_on, rate_off, row['rate_rho_on_zcrit'], row['opk_on_tv_zcrit'],
                                               row['opk_off_tv_zcrit'], units='z')
    return row


# ---------------------------------------------------------------------------
# part A

def _meta(r) -> dict:
    return dict(source=r.source, point_id=r.point_id, model=r.model, pulse=r.pulse, held=r.held, identity=r.identity,
                arm=r.arm)


def protocol_readings(out: Path) -> pd.DataFrame:
    """Part A: every saved run with spike times read through each reading (`readings`); writes protocol_readings.csv,
    protocol_pairs.csv and protocol_checks.json. Checks (RuntimeError on failure): the spike times add up to the
    counts; for E80's runs, the counts reading's statistics equal E80's pulse.csv (E77's Scorer) to 1e-9; for runs whose
    spikes all fall before the window, the step reading's amplitudes are the counts exactly and its statistics equal the
    counts reading's exactly."""
    out.mkdir(parents=True, exist_ok=True)
    runs = saved_runs()
    # (round_trip: pandas' default float parser can be one unit in the last place off)
    p80 = pd.read_csv(H.results_dir(E80) / 'pulse.csv', float_precision='round_trip').drop_duplicates(
        'point_id', keep='last').set_index('point_id')
    rows, pairs, checks = [], [], {}
    t0 = time.time()
    for r in runs.itertuples():
        o, st = load_run(Path(r.dir), r.point_id)
        resp = {c: o[c][0] for c in o}
        band = band_row(o)
        t_last = max(float(st[c].time_s.max()) if len(st[c]) else -np.inf for c in st)
        chk = dict(t_last_spike=t_last)
        by = {}
        for lab, k in readings():
            x = weighted(o, st, k)
            row, pr = score_reading(resp, x)
            meta = dict(_meta(r), kernel=lab, rise=np.nan if k is None else k[0], decay=np.nan if k is None else k[1],
                        primary=k == PRIMARY)
            for c in ('on', 'off'):
                a = x[c].to_numpy()
                meta[f'x_sum_{c}'], meta[f'x_max_{c}'] = float(a.sum()), float(a.max())
            row = dict(meta, **band, **row)
            rows.append(row)
            by[lab] = (x, row)
            pr.insert(0, 'kernel', lab)
            pr.insert(0, 'point_id', r.point_id)
            pr.insert(0, 'source', r.source)
            pairs.append(pr)
        if t_last < WINDOW[0]:
            same = all(np.array_equal(by['step'][0][c].to_numpy(), by[COUNTS][0][c].to_numpy()) for c in ('on', 'off'))
            num = [c for c, v in by[COUNTS][1].items() if isinstance(v, (float, int, np.floating, np.integer))
                   and not isinstance(v, bool) and c not in ('rise', 'decay')]
            diff = [c for c in num if not (by['step'][1][c] == by[COUNTS][1][c]
                                           or (np.isnan(by['step'][1][c]) and np.isnan(by[COUNTS][1][c])))]
            chk.update(step_equals_counts=bool(same), step_stats_differ=diff)
            if not same or diff:
                raise RuntimeError(f'{r.point_id}: the step reading differs from the counts ({diff})')
        else:
            w = {c: H.kernel_window_weights(st[c].time_s.to_numpy(), *STEP, window=WINDOW) for c in st}
            chk.update(step_equals_counts=False, step_min_weight=float(min(v.min() for v in w.values() if len(v))),
                       step_n_spikes_in_window=int(sum(int((st[c].time_s.to_numpy() >= WINDOW[0]).sum()) for c in st)))
        if r.source == 'E80':
            ref = p80.loc[r.point_id]
            row = by[COUNTS][1]
            dev = {c: float(abs(float(row[c]) - float(ref[c]))) for c in CHECK_COLS if c in ref.index}
            dev['K_cal_vs_E67_K'] = float(abs(row['K_cal'] - float(ref['K'])))
            flags = dict(realism_flag=(str(row['realism_flag']) if row['realism_flag'] else 'nan')
                         == (str(ref['realism_flag']) if isinstance(ref['realism_flag'], str) else 'nan'),
                         realism_flag_zcrit=(str(row['realism_flag_zcrit']) if row['realism_flag_zcrit'] else 'nan')
                         == (str(ref['realism_flag_zcrit']) if isinstance(ref['realism_flag_zcrit'], str) else 'nan'))
            chk.update(vs_pulse_csv_max_abs=max(dev.values()), vs_pulse_csv_worst=max(dev, key=dev.get),
                       vs_pulse_csv_exact=all(v == 0 for v in dev.values()), flags_equal=flags)
            if max(dev.values()) > 1e-9 or not all(flags.values()):
                raise RuntimeError(f"{r.point_id}: the counts reading differs from E80's pulse.csv ({dev}, {flags})")
        checks[f'{r.source}:{r.point_id}'] = chk
        print(f"{r.source}:{r.point_id}: {len(readings())} readings; counts raw / cal loss "
              f"{by[COUNTS][1]['raw_loss']:.4f} / {by[COUNTS][1]['cal_loss']:.4f}, primary "
              f"{by[klabel(PRIMARY)][1]['raw_loss']:.4f} / {by[klabel(PRIMARY)][1]['cal_loss']:.4f}; {chk}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'protocol_readings.csv', index=False)
    pd.concat(pairs, ignore_index=True).to_csv(out / 'protocol_pairs.csv', index=False)
    rec = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E82_flies_protocol.py',
               al_analysis_commit=_git('rev-parse', 'HEAD'), seconds=time.time() - t0, runs=checks,
               all_ok=True)
    (out / 'protocol_checks.json').write_text(json.dumps(rec, indent=1, default=str) + '\n')
    return res


def claw_counts() -> pd.Series:
    """Claws per KC (claw rows with any input, E71.n_claws_per_kc) of the real wiring (the archive's bouton-base wiring;
    the claw permutations and matched wirings keep them per KC), by KC id."""
    W = pd.read_parquet(WIRING)
    return (W != 0).any(axis='columns').groupby(level=H.KC_ID_LEVEL).sum().astype(int)


def _late_mask(st: pd.DataFrame) -> np.ndarray:
    return st.step.to_numpy() >= _steps(LATE)


def late_spikes(out: Path) -> pd.DataFrame:
    """Part A: per run and condition the late spikes (at or after 0.5 s from onset): totals (with the KC x odor entries
    and KCs that have them, the most per entry, the share of all spikes in the first 250 ms), every spike by 250 ms bin,
    the late spikes by odor, KC type and claw-count group; writes late_spikes.csv (one row per run, condition, `by`,
    `key`), and the loss with the APL-off late spikes left out (`late_leave_out`)."""
    runs = saved_runs()
    claws = claw_counts()
    s0, spb = _steps(0.0), int(round(BIN / _dt()))
    rows, lo_rows = [], []
    for r in runs.itertuples():
        o, st = load_run(Path(r.dir), r.point_id)
        m = _meta(r)
        idx = o['on'][1].index
        kc_type = idx.get_level_values(H.KC_TYPE_LEVEL).to_numpy()
        ncl = claws.reindex(idx.get_level_values(H.KC_ID_LEVEL)).to_numpy()
        if np.isnan(ncl.astype(float)).any():
            raise RuntimeError(f'{r.point_id}: KCs without a claw count')
        grp = np.full(len(ncl), '', dtype=object)
        for g, lo, hi in CLAW_GROUPS:
            grp[(ncl >= lo) & (ncl <= hi)] = g
        odors = [H.odor_abbrev(c) for c in o['on'][1].columns]
        for c, _ in CONDS:
            t = st[c]
            late = _late_mask(t)
            n, nl = len(t), int(late.sum())
            tl = t[late]
            # (late spikes per KC x odor entry that has any)
            _, ent = np.unique(tl.kc.to_numpy().astype(np.int64) * len(odors) + tl.odor.to_numpy(), return_counts=True)
            first = int(((t.step.to_numpy() >= s0) & (t.step.to_numpy() < s0 + spb)).sum())
            rows.append(dict(m, condition=c, by='total', key='', n_spikes=n, n_late=nl, share=nl / n if n else np.nan,
                             n_entries_late=int(len(ent)), n_kcs_late=int(tl.kc.nunique()) if nl else 0,
                             max_late_per_entry=int(ent.max()) if nl else 0,
                             share_first_bin=first / n if n else np.nan, n_kcs_group=len(idx)))
            b = (t.step.to_numpy() - s0) // spb
            for bi in np.unique(b):
                k = int((b == bi).sum())
                key = 'pre' if bi < 0 else f'{bi * BIN:.2f}-{(bi + 1) * BIN:.2f}'
                rows.append(dict(m, condition=c, by='bin', key=key, n_spikes=k, share=k / n if n else np.nan,
                                 t0=float(bi * BIN) if bi >= 0 else np.nan, late=bool(bi * BIN >= LATE)))
            ln = np.bincount(tl.odor.to_numpy(), minlength=len(odors)) if nl else np.zeros(len(odors), dtype=int)
            an = np.bincount(t.odor.to_numpy(), minlength=len(odors))
            for j, od in enumerate(odors):
                rows.append(dict(m, condition=c, by='odor', key=od, n_spikes=int(an[j]), n_late=int(ln[j]),
                                 share=ln[j] / nl if nl else np.nan,
                                 approx_conc=od in H.APPROX_CONC_ODORS))
            for by, lab in (('kc_type', kc_type), ('claws', grp)):
                at = lab[t.kc.to_numpy()]
                al = lab[tl.kc.to_numpy()] if nl else np.array([], dtype=object)
                for g in sorted(set(lab)):
                    nk = int((lab == g).sum())
                    k_all, k_late = int((at == g).sum()), int((al == g).sum())
                    rows.append(dict(m, condition=c, by=by, key=str(g), n_spikes=k_all, n_late=k_late,
                                     share=k_late / nl if nl else np.nan, n_kcs_group=nk, late_per_kc=k_late / nk,
                                     n_kcs_late=int(len(set(tl.kc.to_numpy()[al == g]))) if nl else 0))
        lo_rows += late_leave_out(r, o, st)
        print(f'{r.source}:{r.point_id}: late spikes on / off '
              + ' / '.join(str(int(_late_mask(st[c]).sum())) for c in ('on', 'off')), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'late_spikes.csv', index=False)
    pd.DataFrame(lo_rows).to_csv(out / 'late_leave_out.csv', index=False)
    return res


def late_leave_out(r, o: Dict[str, tuple], st: Dict[str, pd.DataFrame]) -> List[dict]:
    """The loss (raw and under the readout, against Yang's z; the readout's K kept from APL on, which is unchanged)
    with the APL-off spikes at or after 0.5 s left out: none (the reading itself), all, each odor's alone and the odors'
    cumulatively in descending order of their late APL-off spikes; on counts and under the primary reading. The
    responses are kept."""
    resp = {c: o[c][0] for c in o}
    t = st['off']
    late = _late_mask(t)
    odors = [H.odor_abbrev(c) for c in o['off'][1].columns]
    nl = np.bincount(t.odor.to_numpy()[late], minlength=len(odors))
    order = [j for j in np.argsort(-nl, kind='stable') if nl[j] > 0]
    sets = [('none', []), ('all', list(range(len(odors))))] + [(odors[j], [j]) for j in order]
    sets += [('top' + str(i) + ':' + '+'.join(odors[j] for j in order[:i]), order[:i]) for i in range(2, len(order))]
    rows = []
    for lab, k in ((COUNTS, None), (klabel(PRIMARY), PRIMARY)):
        x = weighted(o, st, k)
        K = calibrate(x['on'], resp['on'], data()['cal_target'])
        if k is None:
            w = np.ones(len(t))
        else:
            w = H.kernel_window_weights(t.time_s.to_numpy(), *k, window=WINDOW)
        for name, js in sets:
            sel = late & np.isin(t.odor.to_numpy(), js)
            a = np.zeros(x['off'].shape)
            np.add.at(a, (t.kc.to_numpy()[sel], t.odor.to_numpy()[sel]), w[sel])
            xo = {'on': x['on'], 'off': x['off'] - a}
            if k is None and name == 'none' and not np.array_equal(xo['off'].to_numpy(), x['off'].to_numpy()):
                raise RuntimeError('leave-out of nothing changed the counts')
            row, _ = score_reading(resp, xo, full=False, K=K)
            rows.append(dict(_meta(r), kernel=lab, left_out=name, n_removed=int(sel.sum()),
                             weight_removed=float(w[sel].sum()),
                             **{c: row[c] for c in row if c.startswith(('raw_', 'cal_', 'K_'))
                                and not c.endswith(('_near6_on', '_near6_off', 'mean_dist_on'))}))
    return rows


def _kernel(u: np.ndarray, rise: float, decay: float) -> np.ndarray:
    """The kernel (1 - exp(-u / rise)) exp(-u / decay) at u (0 for u <= 0)."""
    u = np.asarray(u, dtype=float)
    pos = u > 0
    k = np.ones_like(u) if rise == 0 else -np.expm1(-np.where(pos, u, 0.0) / rise)
    if not np.isinf(decay):
        k = k * np.exp(-np.where(pos, u, 0.0) / decay)
    return np.where(pos, k, 0.0)


def _kernel_integral(u: np.ndarray, rise: float, decay: float) -> np.ndarray:
    """The kernel's integral from 0 to u (u >= 0)."""
    u = np.maximum(np.asarray(u, dtype=float), 0.0)
    first = u if np.isinf(decay) else -decay * np.expm1(-u / decay)
    tau = rise if np.isinf(decay) else (rise * decay / (rise + decay) if rise > 0 else 0.0)
    return first if tau == 0 else first + tau * np.expm1(-u / tau)


def predicted_traces(out: Path) -> pd.DataFrame:
    """Part A: per run, kernel (KERNELS) and condition, the responders' mean kernel-convolved signal (summed kernels of
    the spikes of responding KC x odor entries over the number of such entries; in the kernel's units) and the
    population mean (all spikes over all entries), from onset to 8.1 s every 50 ms; writes predicted_traces.csv, and
    predicted_ratios.csv with the APL-off / APL-on ratio of each signal's mean over 0 to 2, 2 to 4 and 4 to 8.1 s (exact
    window means from the kernel's integral)."""
    runs = saved_runs()
    grid = np.round(np.arange(0.0, TRACE_END + 1e-9, TRACE_DT), 6)
    long, rat = [], []
    for r in runs.itertuples():
        o, st = load_run(Path(r.dir), r.point_id)
        m = _meta(r)
        n_ent = o['on'][0].size
        sel = {}
        for c in ('on', 'off'):
            R = o[c][0].to_numpy(dtype=float) > 0.5
            t = st[c]
            isr = R[t.kc.to_numpy(), t.odor.to_numpy()]
            # (unique spike times with their multiplicities, and the number of entries the signal is averaged over)
            sel[c] = {kind: np.unique(times, return_counts=True) + (n,) for kind, times, n in (
                ('resp', t.time_s.to_numpy()[isr], int(R.sum())), ('all', t.time_s.to_numpy(), n_ent))}
        for k in KERNELS:
            means = {}
            for c in ('on', 'off'):
                sig = {}
                for kind, (tu, cnt, n) in sel[c].items():
                    x = np.zeros(len(grid))
                    for i0 in range(0, len(tu), 2000):
                        x += (_kernel(grid[:, None] - tu[None, i0:i0 + 2000], *k) * cnt[i0:i0 + 2000]).sum(axis=1)
                    sig[kind] = x / max(n, 1)
                    for a, b in TRACE_WINDOWS:
                        area = (_kernel_integral(b - tu, *k) - _kernel_integral(a - tu, *k)) @ cnt
                        means[(c, kind, a, b)] = float(area / (b - a) / max(n, 1))
                long.append(pd.DataFrame(dict(m, kernel=klabel(k), rise=k[0], decay=k[1], condition=c,
                                              n_responding_entries=sel[c]['resp'][2], t=grid, resp_mean=sig['resp'],
                                              all_mean=sig['all'])))
            for kind in ('resp', 'all'):
                for a, b in TRACE_WINDOWS:
                    on, off = means[('on', kind, a, b)], means[('off', kind, a, b)]
                    rat.append(dict(m, kernel=klabel(k), rise=k[0], decay=k[1], signal=kind, t0=a, t1=b, mean_on=on,
                                    mean_off=off, ratio_off_on=off / on if on > 0 else np.nan))
        print(f'{r.source}:{r.point_id}: traces done', flush=True)
    res = pd.concat(long, ignore_index=True)
    res.to_csv(out / 'predicted_traces.csv', index=False)
    pd.DataFrame(rat).to_csv(out / 'predicted_ratios.csv', index=False)
    return res


def population_readings(out: Path) -> pd.DataFrame:
    """The populations of four wirings under every reading (part B1's 'b2' and 'fc' populations, runs.csv's
    `population` column, at each pulse where all four members are read): the mean over the wirings of the pair
    distances per condition and readout (the readout's K per wiring, as E72.population_row), scored by
    `score_distances`; writes protocol_populations.csv."""
    rp = H.results_dir(EXPERIMENT) / 'runs.csv'
    if not rp.exists():
        print('no runs.csv yet: no populations', flush=True)
        return pd.DataFrame()
    m = pd.read_csv(rp).drop_duplicates('point_id', keep='last')
    if 'population' not in m.columns:
        return pd.DataFrame()
    m = m[m.population.fillna('').astype(str).str.len() > 0]
    pairs = pd.read_csv(out / 'protocol_pairs.csv')
    pairs = pairs[pairs.source == 'E82B']
    rows = []
    for (popn, pulse), g in m.groupby(['population', 'pulse']):
        pids = sorted(g.point_id)
        pp = pairs[pairs.point_id.isin(pids)]
        if len(pids) != 4 or set(pp.point_id) != set(pids):
            continue
        for lab, _ in readings():
            dist = {}
            for ro in ('raw', 'cal', 'cal_noise'):
                q = pp[(pp.kernel == lab) & (pp.readout == ro)]
                mean = q.groupby(['odor_a', 'odor_b'])[['on', 'off']].mean()
                dist[ro] = {c: mean[c] for c in ('on', 'off')}
            rows.append(dict(population=popn, pulse=pulse, members=';'.join(pids), n_wirings=len(pids), kernel=lab,
                             primary=lab == klabel(PRIMARY), **score_distances(dist)))
    res = pd.DataFrame(rows)
    res.to_csv(out / 'protocol_populations.csv', index=False)
    if len(res):
        print(res[['population', 'pulse', 'kernel', 'raw_loss', 'raw_mean', 'raw_pattern', 'cal_loss', 'cal_pattern']]
              .round(4).to_string(index=False), flush=True)
    return res


# ---------------------------------------------------------------------------
# part C: the gates

def _reading(rd: pd.DataFrame, source: str, pid: str, kernel: str) -> pd.Series:
    x = rd[(rd.source == source) & (rd.point_id == pid) & (rd.kernel == kernel)]
    if len(x) != 1:
        raise KeyError(f'{source}:{pid} {kernel}: {len(x)} rows in protocol_readings.csv')
    return x.iloc[0]


def _effect(pairs: pd.DataFrame, source: str, pid: str, kernel: str, readout: str = 'cal') -> pd.Series:
    x = pairs[(pairs.source == source) & (pairs.point_id == pid) & (pairs.kernel == kernel) & (pairs.readout == readout)]
    return x.set_index(['odor_a', 'odor_b'])['effect'].sort_index()


def reference_point(pt, runs: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """A gated point's reference: its `ref_point` if given, else the run of the same model at the 2 s pulse, re-held, not
    in B2 to B5 and not an identity (part B's own before E80's)."""
    if isinstance(pt.ref_point, str) and pt.ref_point:
        hit = runs[runs.point_id == pt.ref_point]
        if len(hit):
            h = hit.sort_values('source', key=lambda s: s.map({'E82B': 0, 'E80': 1}).fillna(2)).iloc[0]
            return h.source, h.point_id
    c = runs[(runs.model == pt.model) & np.isclose(runs.pulse.astype(float), 2.0) & runs.held.astype(bool)
             & ~runs.arm.isin(GATED_ARMS) & ~runs.identity.astype(bool)]
    if not len(c):
        return None
    c = c.sort_values('source', key=lambda s: s.map({'E82B': 0, 'E80': 1}).fillna(2))
    return c.iloc[0].source, c.iloc[0].point_id


def gate_rows(rd: pd.DataFrame, pairs: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Both gates on every non-identity point of B2 to B5 (see the module docstring): one row per point with each
    condition, the numbers it was judged on and the references'."""
    import E77_input_side_common_error as G
    import E75_glomerulus_apl_profile as E75
    D = data()
    P = klabel(PRIMARY)
    f05 = _reading(rd, *FIT0_PULSE05, COUNTS)
    f2 = _reading(rd, *FIT0_PULSE2, COUNTS)
    lim = float(f05.raw_loss) + PROTOCOL_SLACK
    rows = []
    for pt in runs[runs.arm.isin(GATED_ARMS) & ~runs.identity.astype(bool)].itertuples():
        cnt, pri = _reading(rd, pt.source, pt.point_id, COUNTS), _reading(rd, pt.source, pt.point_id, P)
        flag = pri.realism_flag
        inside = not (isinstance(flag, str) and flag.strip() not in ('', 'nan'))
        row = dict(source=pt.source, point_id=pt.point_id, model=pt.model, arm=pt.arm, pulse=pt.pulse,
                   inside_bands=inside, realism_flag=flag, realism_flag_zcrit=pri.realism_flag_zcrit,
                   raw_loss_counts=cnt.raw_loss, raw_loss_primary=pri.raw_loss, cal_loss_counts=cnt.cal_loss,
                   cal_loss_primary=pri.cal_loss, cal_pattern_primary=pri.cal_pattern, raw_pattern_primary=pri.raw_pattern,
                   raw_mean_primary=pri.raw_mean, cal_mean_primary=pri.cal_mean, cal_rho_primary=pri.cal_rho,
                   max_count_on=pri.max_count_on, max_count_off=pri.max_count_off, rate_on=pri.rate_on, ratio=pri.ratio)
        # the protocol gate
        if pt.model == FIRING_CEILING:
            row.update(pg_ref_loss_05=float(f05.raw_loss), pg_limit=lim, pg_ref_max_on=int(f2.max_count_on),
                       pg_ref_max_off=int(f2.max_count_off),
                       pg_counts=bool(cnt.raw_loss <= lim), pg_primary=bool(pri.raw_loss <= lim),
                       pg_counts_cal=bool(cnt.cal_loss <= lim), pg_primary_cal=bool(pri.cal_loss <= lim),
                       pg_max=bool(pri.max_count_on <= f2.max_count_on and pri.max_count_off <= f2.max_count_off))
            row['protocol_gate'] = bool(inside and row['pg_counts'] and row['pg_primary'] and row['pg_max'])
        else:
            row['protocol_gate'] = np.nan
        # E77's gate against the model's own re-held 2 s point
        ref = reference_point(pt, runs)
        if ref is None:
            row.update(ref_point='', e77_gate=np.nan, note='no re-held 2 s point of this model')
        else:
            rp = _reading(rd, ref[0], ref[1], P)
            gain = float(rp.cal_pattern - pri.cal_pattern)
            ch = _effect(pairs, pt.source, pt.point_id, P) - _effect(pairs, ref[0], ref[1], P)
            c, c6 = D['common']['cal'], D['c6']['cal']
            tr_all = E75.reduction(c, ch.reindex(c.index))
            tr_six = E75.reduction(c6, ch.reindex(c6.index))
            row.update(ref_point=f'{ref[0]}:{ref[1]}', ref_cal_pattern_primary=rp.cal_pattern,
                       ref_raw_loss_primary=rp.raw_loss, pattern_gain=gain, transfer_all=tr_all, transfer_six=tr_six,
                       e77_c1=bool(gain >= G.GATE_PATTERN_GAIN and pri.raw_loss <= rp.raw_loss + G.GATE_RAW_SLACK),
                       e77_c2=bool(tr_all >= G.GATE_TRANSFER and tr_six >= 0.5 * tr_all))
            row['e77_gate'] = bool(inside and row['e77_c1'] and row['e77_c2'])
        rows.append(row)
    return pd.DataFrame(rows)


def gates(out: Path) -> pd.DataFrame:
    """Part C: the protocol gate and E77's gate on part B's points (`gate_rows`), from protocol_readings.csv and
    protocol_pairs.csv (run part A's readings first so that they include part B's runs); writes gate.csv."""
    rd = pd.read_csv(out / 'protocol_readings.csv')
    pairs = pd.read_csv(out / 'protocol_pairs.csv')
    runs = saved_runs()
    missing = [f'{r.source}:{r.point_id}' for r in runs.itertuples()
               if not ((rd.source == r.source) & (rd.point_id == r.point_id)).any()]
    if missing:
        raise RuntimeError(f'protocol_readings.csv lacks {missing}: run the readings first')
    res = gate_rows(rd, pairs, runs)
    res.to_csv(out / 'gate.csv', index=False)
    if len(res):
        show = [c for c in ('point_id', 'model', 'arm', 'inside_bands', 'raw_loss_counts', 'raw_loss_primary',
                            'cal_pattern_primary', 'pattern_gain', 'transfer_all', 'transfer_six', 'max_count_on',
                            'max_count_off', 'protocol_gate', 'e77_gate') if c in res.columns]
        print(res[show].round(4).to_string(index=False), flush=True)
    else:
        print('no points of B2 to B5 among the saved runs yet; gate.csv is empty', flush=True)
    return res


# ---------------------------------------------------------------------------
# tables

def _f(x, spec: str = '.3f') -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '-'
    return '-' if not np.isfinite(x) else format(x, spec)


def tables(out: Path) -> str:
    """protocol_table.md: part A's tables (the main target at the 0.5 s pulse and under the flies' protocol for C2 and
    the firing-ceiling fit, the common error and ρ, the late spikes, the predicted ratios, realism)."""
    rd = pd.read_csv(out / 'protocol_readings.csv')
    rd = rd[rd.source == 'E80'].set_index(['point_id', 'kernel'])
    lo = pd.read_csv(out / 'late_leave_out.csv')
    lo = lo[lo.source == 'E80']
    ls = pd.read_csv(out / 'late_spikes.csv')
    ls = ls[ls.source == 'E80']
    pr = pd.read_csv(out / 'predicted_ratios.csv')
    pr = pr[pr.source == 'E80']
    U = data()['units']
    z, nz = U['z'], U['noise']
    lines = ["# E82 part A: the flies' recording protocol read from E80's saved runs", '',
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by E82_flies_protocol.py (al_analysis "
             f"{_git('rev-parse', '--short', 'HEAD')}). KC data: Yang's z, all eight recordings with APL output blocked "
             "(TNT), unless a column says otherwise. C2: the breadth-2 APL-effect fit (E67); the firing-ceiling fit: fit 0 "
             "of the data side of the main target (E76). The 2 s points are E80's re-held ones (both response rates "
             "held). Lower loss is better. paa, euc and -aPine enter with ORN input at substituted concentrations.", '',
             '## The main target at the 0.5 s pulse and under the flies\' protocol', '',
             '| reading of the model | C2, no saturation: APL-effect loss / mean APL effect / pattern part | C2, calibrated '
             'saturation: loss / pattern part (K) | the firing-ceiling fit, no saturation: loss / mean / pattern part | the '
             'firing-ceiling fit, calibrated saturation: loss / pattern part (K) |',
             '|---|---:|---:|---:|---:|']

    def cells(c2: Optional[pd.Series], f0: Optional[pd.Series]) -> str:
        def a(r):
            return '-' if r is None else f"{_f(r.raw_loss)} / {_f(r.raw_mean, '+.3f')} / {_f(r.raw_pattern)}"

        def b(r):
            if r is None:
                return '-'
            K = 'K at its bound' if bool(r.K_cal_at_bound) else f"K {_f(r.K_cal, '.3g')}"
            return f"{_f(r.cal_loss)} / {_f(r.cal_pattern)} ({K})"
        return f'| {a(c2)} | {b(c2)} | {a(f0)} | {b(f0)} |'

    def get(pid, k):
        try:
            return rd.loc[(pid, k)]
        except KeyError:
            return None
    lines.append('| 0.5 s pulse, spike counts ' + cells(get('C2_pulse0.5_fixed', COUNTS), get('fit0_pulse0.5_fixed', COUNTS)))
    lines.append('| 2 s pulse, spike counts ' + cells(get('C2_pulse2_held', COUNTS), get('fit0_pulse2_held', COUNTS)))
    for r_, d_ in DIAGONAL:
        k = klabel((r_, d_))
        lines.append(f'| 2 s pulse, the flies\' window, rise {r_:g} s, decay {d_:g} s'
                     + (' (primary)' if (r_, d_) == PRIMARY else '') + ' '
                     + cells(get('C2_pulse2_held', k), get('fit0_pulse2_held', k)))
    for kl, name in ((COUNTS, 'spike counts'), (klabel(PRIMARY), 'the primary reading')):
        g = lo[(lo.kernel == kl) & (lo.left_out == 'all')].set_index('point_id')
        c2 = g.loc['C2_pulse2_held'] if 'C2_pulse2_held' in g.index else None
        f0 = g.loc['fit0_pulse2_held'] if 'fit0_pulse2_held' in g.index else None
        lines.append(f'| 2 s pulse, {name}, the APL-off spikes after 0.5 s left out ' + cells(c2, f0))
    lines.append(f"| 0.5 s pulse, the flies' window, primary " + cells(get('C2_pulse0.5_fixed', klabel(PRIMARY)),
                                                                     get('fit0_pulse0.5_fixed', klabel(PRIMARY))))
    kd = f"floor {_f(z['rmse_floor'])} / {_f(z['mean'], '+.3f')} / floor {_f(z['pattern_floor'])}"
    lines.append(f"| KC data | {kd} | floor {_f(z['rmse_floor'])} / floor {_f(z['pattern_floor'])} | {kd} | floor "
                 f"{_f(z['rmse_floor'])} / floor {_f(z['pattern_floor'])} |")
    lines += ['', 'Every kernel of the grid (rise x decay), the 2 s re-held points, APL-effect loss (no saturation / calibrated '
              'saturation):', '', '| rise \\ decay | C2: 1 s | 3 s | 5 s | the firing-ceiling fit: 1 s | 3 s | 5 s |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r_ in sorted({k[0] for k in KERNELS}):
        cs = []
        for pid in ('C2_pulse2_held', 'fit0_pulse2_held'):
            for d_ in sorted({k[1] for k in KERNELS}):
                x = get(pid, klabel((r_, d_)))
                cs.append('-' if x is None else f'{_f(x.raw_loss)} / {_f(x.cal_loss)}')
        lines.append(f'| {r_:g} s | ' + ' | '.join(cs) + ' |')

    lines += ['', '## The common error and ρ on the per-pair APL effect', '',
              '| reading | C2, saturation: ρ / ester / alcohol group error / projection on the common error | the '
              'firing-ceiling fit, no saturation: ρ / ester / alcohol / projection | C2, saturation, against baseline-noise '
              'units: loss / mean / pattern | the firing-ceiling fit, no saturation, against baseline-noise units: loss / '
              'mean / pattern | C2, saturation, against the six typical TNT recordings: loss / pattern / ρ |',
              '|---|---:|---:|---:|---:|---:|']
    sets = [('0.5 s, counts', 'C2_pulse0.5_fixed', 'fit0_pulse0.5_fixed', COUNTS),
            ('2 s, counts', 'C2_pulse2_held', 'fit0_pulse2_held', COUNTS)]
    sets += [(f'2 s, rise {r_:g} / decay {d_:g} s', 'C2_pulse2_held', 'fit0_pulse2_held', klabel((r_, d_)))
             for r_, d_ in DIAGONAL]
    for name, a, b, k in sets:
        x, y = get(a, k), get(b, k)
        if x is None or y is None:
            continue
        lines.append(f"| {name} | {_f(x.cal_rho, '.2f')} / {_f(x.cal_err_ester, '+.3f')} / {_f(x.cal_err_alcohol, '+.3f')}"
                     f" / {_f(x.cal_proj_common, '.2f')} | {_f(y.raw_rho, '.2f')} / {_f(y.raw_err_ester, '+.3f')} / "
                     f"{_f(y.raw_err_alcohol, '+.3f')} / {_f(y.raw_proj_common, '.2f')} | {_f(x.cal_noise_loss)} / "
                     f"{_f(x.cal_noise_mean, '+.3f')} / {_f(x.cal_noise_pattern)} | {_f(y.raw_noise_loss)} / "
                     f"{_f(y.raw_noise_mean, '+.3f')} / {_f(y.raw_noise_pattern)} | {_f(x.cal_z6_loss)} / "
                     f"{_f(x.cal_z6_pattern)} / {_f(x.cal_z6_rho, '.2f')} |")
    lines.append(f"| KC data | ceiling {_f(z['rho_ceiling'], '.2f')} | ceiling {_f(z['rho_ceiling'], '.2f')} | floor "
                 f"{_f(nz['rmse_floor'])} / {_f(nz['mean'], '+.3f')} / floor {_f(nz['pattern_floor'])} | floor "
                 f"{_f(nz['rmse_floor'])} / {_f(nz['mean'], '+.3f')} / floor {_f(nz['pattern_floor'])} | - |")

    lines += ['', '## The late spikes (at or after 0.5 s from onset), 2 s re-held points', '',
              '| model | condition | spikes | late | share | KC x odor entries / KCs with late spikes | most per entry | '
              'share in the first 250 ms |', '|---|---|---:|---:|---:|---:|---:|---:|']
    tot = ls[ls.by == 'total']
    for pid in ('C2_pulse2_held', 'fit0_pulse2_held'):
        for c in ('on', 'off'):
            g = tot[(tot.point_id == pid) & (tot.condition == c)]
            if not len(g):
                continue
            g = g.iloc[0]
            lines.append(f"| {pid.split('_')[0]} | APL {c} | {int(g.n_spikes)} | {int(g.n_late)} | {_f(g.share, '.1%')} | "
                         f"{int(g.n_entries_late)} / {int(g.n_kcs_late)} | {int(g.max_late_per_entry)} | "
                         f"{_f(g.share_first_bin, '.1%')} |")
    b = ls[(ls.by == 'bin') & (ls.condition == 'off') & (ls.point_id == 'fit0_pulse2_held')]
    if len(b):
        lines += ['', 'The firing-ceiling fit, APL off, spikes per 250 ms bin: '
                  + ', '.join(f'{k} s {int(n)}' for k, n in zip(b.key, b.n_spikes)) + '.']
    od = ls[(ls.by == 'odor') & (ls.condition == 'off') & (ls.point_id == 'fit0_pulse2_held')].sort_values(
        'n_late', ascending=False)
    if len(od):
        lines += ['', 'The firing-ceiling fit, APL off, late spikes by odor: '
                  + ', '.join(f'{k} {int(n)}' for k, n in zip(od.key, od.n_late)) + '.']
    for by in ('kc_type', 'claws'):
        g = ls[(ls.by == by) & (ls.condition == 'off') & (ls.point_id == 'fit0_pulse2_held')]
        if len(g):
            lines += ['', f'The firing-ceiling fit, APL off, late spikes by {by.replace("_", " ")} (late / KCs in the group): '
                      + ', '.join(f'{k} {int(n)} / {int(m)}' for k, n, m in zip(g.key, g.n_late, g.n_kcs_group)) + '.']
    lines += ['', 'The loss with the APL-off late spikes left out (2 s re-held; no saturation; the responses kept): the '
              'reading itself, all of them, the three odors with the most, then the odors cumulatively in that order:', '',
              '| model | left out | spikes removed | counts: loss / mean / pattern | primary reading: loss / mean / pattern |',
              '|---|---|---:|---:|---:|']
    for pid in ('C2_pulse2_held', 'fit0_pulse2_held'):
        g = lo[lo.point_id == pid]
        names = list(dict.fromkeys(g.left_out))
        singles = [n for n in names if n not in ('none', 'all') and not n.startswith('top')][:3]
        tops = [n for n in names if n.startswith('top')][:2]
        for name in ['none', 'all'] + singles + tops:
            cs = []
            for kl in (COUNTS, klabel(PRIMARY)):
                x = g[(g.kernel == kl) & (g.left_out == name)]
                cs.append('-' if not len(x) else f"{_f(x.iloc[0].raw_loss)} / {_f(x.iloc[0].raw_mean, '+.3f')} / "
                                                 f"{_f(x.iloc[0].raw_pattern)}")
            n = g[(g.kernel == COUNTS) & (g.left_out == name)]
            lines.append(f"| {pid.split('_')[0]} | {name} | {int(n.iloc[0].n_removed) if len(n) else '-'} | "
                         + ' | '.join(cs) + ' |')

    lines += ['', "## The model's prediction for the traces (primary kernel, 2 s re-held points)", '',
              '| model | signal | APL-off / APL-on ratio 0-2 s | 2-4 s | 4-8.1 s |', '|---|---|---:|---:|---:|']
    for pid in ('C2_pulse2_held', 'fit0_pulse2_held'):
        for sig, name in (('resp', "responders' mean"), ('all', 'population mean')):
            g = pr[(pr.point_id == pid) & (pr.kernel == klabel(PRIMARY)) & (pr.signal == sig)].sort_values('t0')
            if len(g):
                lines.append(f"| {pid.split('_')[0]} | {name} | " + ' | '.join(_f(v, '.2f') for v in g.ratio_off_on) + ' |')

    lines += ['', '## Realism (the same under every reading: the responses are the two-spike counts)', '',
              '| point | response rate APL on / off (ratio) | response rate ρ, APL on (noise units / Yang\'s z) | odors per '
              'KC, distance APL on / off (noise units) | largest spike count APL on / off | flag |',
              '|---|---:|---:|---:|---:|---|']
    for pid in dict.fromkeys(rd.index.get_level_values(0)):
        x = get(pid, COUNTS)
        flag = x.realism_flag if isinstance(x.realism_flag, str) and x.realism_flag else 'inside the bands'
        lines.append(f"| {pid} | {_f(x.rate_on)} / {_f(x.rate_off)} ({_f(x.ratio, '.2f')}) | {_f(x.rate_rho_on, '.2f')} / "
                     f"{_f(x.rate_rho_on_zcrit, '.2f')} | {_f(x.opk_on_tv)} / {_f(x.opk_off_tv)} | {int(x.max_count_on)} / "
                     f"{int(x.max_count_off)} | {flag} |")
    lines.append('| KC data (bands) | 0.068 / 0.156 (0.045 to 0.085 on; ratio 1.6 to 2.9) | (0.37 or more) | (0.155 / '
                 '0.223) | - | - |')
    text = '\n'.join(lines) + '\n'
    (out / 'protocol_table.md').write_text(text)
    return text


COLORS = dict(c2_raw='#2a78d6', c2_cal='#eb6834', fit0='#1baf7a', on='#52514e', off='#6250d6', ink='#0b0b0b',
              ink2='#52514e', grid='#e6e5e1', band='#f1f0ec')


def figures(out: Path) -> Path:
    """E82_partA.png: the APL-effect loss and the mean APL effect of C2 (no saturation, calibrated saturation) and the
    firing-ceiling fit (no saturation) from the 0.5 s pulse on counts to the 2 s pulse under the flies' window at decays
    5, 3 and 1 s, with the KC data's floor and mean; and the predicted traces (the responders' mean signal under the
    primary kernel, APL on and off) of both at the 2 s pulse."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': COLORS['ink2'], 'axes.labelcolor': COLORS['ink'],
                         'xtick.color': COLORS['ink2'], 'ytick.color': COLORS['ink2'], 'axes.spines.top': False,
                         'axes.spines.right': False})
    rd = pd.read_csv(out / 'protocol_readings.csv')
    rd = rd[rd.source == 'E80'].set_index(['point_id', 'kernel'])
    tr = pd.read_csv(out / 'predicted_traces.csv')
    z = data()['units']['z']
    steps = [('0.5 s pulse\ncounts', '0.5', COUNTS), ('2 s pulse\ncounts', '2', COUNTS)] + [
        (f'2 s, window\nrise {r:g} s\ndecay {d:g} s', '2', klabel((r, d))) for r, d in DIAGONAL]
    pid = {('C2', '0.5'): 'C2_pulse0.5_fixed', ('C2', '2'): 'C2_pulse2_held', ('fit0', '0.5'): 'fit0_pulse0.5_fixed',
           ('fit0', '2'): 'fit0_pulse2_held'}
    series = [('C2, no saturation', 'C2', 'raw', COLORS['c2_raw']), ('C2, calibrated saturation', 'C2', 'cal', COLORS['c2_cal']),
              ('firing-ceiling fit, no saturation', 'fit0', 'raw', COLORS['fit0'])]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1, 1]))
    x = np.arange(len(steps))
    for ax, stat, ylab, ref, reflab in ((axes[0], 'loss', 'APL-effect loss (lower is better)', z['rmse_floor'],
                                         'KC data: floor'),
                                        (axes[1], 'mean', 'mean APL effect', z['mean'], 'KC data')):
        ax.axhline(ref, color=COLORS['ink2'], ls=':', lw=1.2)
        ax.text(len(steps) - 0.6, ref, reflab, color=COLORS['ink2'], va='bottom', ha='right', fontsize=8)
        for lab, m, ro, col in series:
            y = [float(rd.loc[(pid[(m, pu)], k), f'{ro}_{stat}']) for _, pu, k in steps]
            ax.plot(x, y, '-o', color=col, lw=2, ms=6, mec='white', mew=1.5, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([s_[0] for s_ in steps], fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.grid(axis='y', color=COLORS['grid'], lw=0.8)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    axes[0].set_title('the main target under the flies\' protocol', fontsize=9, loc='left')
    axes[1].set_title('its mean part', fontsize=9, loc='left')
    P = klabel(PRIMARY)
    top = 0.0
    for ax, m, name in ((axes[2], 'C2_pulse2_held', 'C2'), (axes[3], 'fit0_pulse2_held', 'firing-ceiling fit')):
        ax.axvspan(WINDOW[0], WINDOW[1], color=COLORS['band'], lw=0)
        ax.axvspan(0.0, 2.0, ymin=0, ymax=0.03, color=COLORS['ink2'], lw=0)
        for c, col, lab in (('on', COLORS['on'], 'APL on'), ('off', COLORS['off'], 'APL off')):
            g = tr[(tr.source == 'E80') & (tr.point_id == m) & (tr.kernel == P) & (tr.condition == c)]
            ax.plot(g.t, g.resp_mean, color=col, lw=2, label=lab)
            top = max(top, float(g.resp_mean.max()))
        ax.set_xlabel('time from odor onset (s)')
        ax.set_title(f"{name}, 2 s pulse:\nresponders' mean signal\n(rise {PRIMARY[0]:g} s, decay {PRIMARY[1]:g} s)",
                     fontsize=9, loc='left')
        ax.grid(axis='y', color=COLORS['grid'], lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.5, loc='upper right')
        ax.text(WINDOW[0] + 0.1, top * 0.04, "the flies' window", color=COLORS['ink2'], fontsize=7, va='bottom')
        ax.text(1.0, top * 0.04, 'odor', color=COLORS['ink2'], fontsize=7, va='bottom', ha='center')
    axes[2].set_ylabel('kernel-convolved spikes per responding KC x odor')
    for ax in axes[2:]:
        ax.set_ylim(0, top * 1.08)
    fig.tight_layout()
    path = out / 'E82_partA.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_provenance(out: Path, part: str) -> None:
    runs = saved_runs()
    D = data()
    files = {}
    for r in runs.itertuples():
        for k, p in _files(Path(r.dir), r.point_id).items():
            if k.endswith('spike_times'):
                files[f'{r.source}:{r.point_id}:{k}'] = _sha_file(p)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E82_flies_protocol.py',
                part=part, al_analysis_commit=_git('rev-parse', 'HEAD'),
                al_analysis_dirty=_git('status', '--porcelain', '--untracked-files=no'),
                model=H.model_provenance(orn_deltas=H.apl16_orn_deltas()), kernels=[list(k) for k in KERNELS],
                primary=list(PRIMARY), window=list(WINDOW), late=LATE, bin=BIN, six_pairs=[list(p) for p in SIX],
                claw_groups=[list(g) for g in CLAW_GROUPS], wiring=str(WIRING), wiring_sha256=_sha_file(WIRING),
                noise_pairs=dict(path=D['noise']['path'], sha256=D['noise']['sha256']),
                data_units=dict(path=E76_DATA_UNITS, sha256=D['units_sha256']),
                runs=runs.to_dict(orient='records'), spike_times_sha256=files, approx_conc_note=H.APPROX_CONC_NOTE,
                weights=('each spike weighted by its kernel area inside the window over the area of a spike at the '
                         "window's start (kc_remapping_harness.kernel_window_weights)"),
                readout=("K calibrated on the weighted APL-on values of the APL-on responders (the two-spike "
                         "responses) to E67.refs()['cal_target'] (cal) or E80_screen.noise_data()['cal_target'] "
                         '(cal_noise), used unchanged for APL off'))
    (out / f'protocol_provenance_{part}.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')


def archive_prep(out: Path) -> pd.DataFrame:
    """Before archive_model_outputs.py: variants.csv with every part-B run (runs.csv; outputs under
    outputs/E82_flies_protocol/runs/, tag = point_id) and its absolute parameters and options, and their pair distances
    (raw spike counts, the harness's primary model RDM) added to model_pair_distances_long.csv (part D's fits write their
    own rows and fits.csv)."""
    import kc_remapping_harness as H
    odir = H.outputs_dir(EXPERIMENT) / 'runs'
    keep = ['model', 'wiring', 'option', 'value', 'time_stim_end', 'time_end', 'fixed_thr', 'fixed_thr_off', 'wAPLKC_scale',
            'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale', 'kc_kc_tau', 'thr_factor', 'apl_factor', 'rate_on',
            'rate_off', 'identity', 'part']
    r = pd.read_csv(out / 'runs.csv').drop_duplicates('point_id', keep='last')
    rows = [dict(tag=x.point_id, variant=x.point_id, base='bouton', k=2.0, kc_kc_inh=x.get('wKCKC_scale'),
                 **{c: x[c] for c in keep if c in r.columns}) for _, x in r.iterrows()]
    v = pd.DataFrame(rows)
    v.to_csv(out / 'variants.csv', index=False)
    lp = out / 'model_pair_distances_long.csv'
    old = pd.read_csv(lp) if lp.exists() else pd.DataFrame(columns=['tag', 'condition', 'odor_a', 'odor_b', 'distance'])
    new = []
    for tag in v.tag:
        if tag in set(old.tag):
            continue
        d = {}
        for c in ('on', 'acute_off'):
            f, g = odir / f'{tag}_{c}_spike_counts.parquet', odir / f'{tag}_{c}_responses.parquet'
            if not (f.exists() and g.exists()):
                break
            d[c] = H.model_rdms(pd.read_parquet(g), pd.read_parquet(f))[H.PRIMARY_MODEL_RDM]
        if len(d) == 2:
            new.append(pd.concat({(tag, c): x for c, x in d.items()}, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index())
    if new:
        pd.concat([old] + new, ignore_index=True).to_csv(lp, index=False)
    print(f'{len(v)} variants; {len(new)} added to model_pair_distances_long.csv', flush=True)
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=('A', 'C', 'tables', 'archive-prep'), required=True)
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.part == 'A':
        protocol_readings(out)
        late_spikes(out)
        predicted_traces(out)
        print(tables(out), flush=True)
        figures(out)
        write_provenance(out, 'A')
    elif args.part == 'C':
        protocol_readings(out)
        late_spikes(out)
        predicted_traces(out)
        population_readings(out)
        gates(out)
        write_provenance(out, 'C')
    elif args.part == 'archive-prep':
        archive_prep(out)
    else:
        print(tables(out), flush=True)
        figures(out)
    print(f'done in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
