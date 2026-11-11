#!/usr/bin/env python3
"""E84, the time course of the APL effect in the flies' traces (no simulations). The design is the E84 entry in
experiment_priorities.md (planned by a Fable subagent, claude-fable-5-2, spawned from viki 2026-11-11 10:08).

Data permission (Yang, 2026-11-11 10:07, notes/yang_drive_data_agreement.md item 2): this script reads, read-only and
on viki only, the suite2p output of the 19 recordings of olfdb datasets 19 / 20 (`yang_response_normalization.RECORDINGS`)
by their exact names, through `yang_response_normalization.read_suite2p` (the reader that script uses too). It accepts
no other recording name and lists nothing on the drives (the 12 recordings outside olfdb are on hold and never read).
The per-trial excerpts are cached in outputs/E84_trace_time_course/cache/ (mode 700, not archived). Yang allowed on
2026-11-11 (in viki's terminal) the sharing of the per-recording mean traces and the order of the 51 presentations
(about 10:52) and of single KCs' traces (about 10:55), for these 19 recordings: results/E84_trace_time_course/ holds the
statistics, the mean traces and presentation_order.csv, and `--part export` writes every KC's per-trial excerpts (Fc
units, with each KC's two scales) to data/yang_apl16_kc_traces/ in the shared directory.

Units. A KC's signal is Fc = F - c Fneu (c = 0.7, Yang's; part E also 0.5 and 0.9) for the ROIs kept as cells. A trial's
excerpt runs from 5 s before the onset to 40 s after it (the next onset is 47 s later), its 5 s pre-onset mean
subtracted; it is read in three units: Yang's z (over the KC's whole-trace SD, `sd_full`), baseline-noise units (over
the SD of Fc within the 51 pre-odor windows, each window's mean removed, `sd_base`; the realism bands' units) and
unscaled (the change in Fc). A window [a, b) s is the volumes round(a v) .. round(b v) - 1 after the onset volume
(v = 2.964 volumes / s), so the target's window, 2.0 to 8.1 s, is volumes 6 to 23, as Yang's 2-8 s; a volume k after
the onset is at k / v s.

- (A) the reader and the records: the cache; the reproduction check (the target's window reproduces olfdb's per-trial
  values to 1e-5 in every recording and the target's two matrices to 1e-6; the noise-free distances of that window
  reproduce E81's registry entries; the baseline-noise values and scales reproduce data/yang_apl16_kc_noise_units/);
  the records as recorded in each recording's YAML file (genotype, label note, induction time, rates) and the order of
  the 51 presentations (records_summary.csv, presentation_order.csv, order_summary.csv, succession_counts.csv;
  reproduction_checks.json).
- (B) the target by window: per window (1 s bins from -2 to 20 s; the pulse 0-2 s, 2-4 s, 4-8.1 s, the target's window,
  0-8.1 s, 8.1-10.1 s, 10.1-12.2 s) and recording, the cosine distances of the trial means ('measured') and the
  noise-free ones from products of different repeats (`kc_remapping_harness.noise_free_pair_distances`); per window the
  mean distances, the per-pair APL effect (mean, the ester and alcohol groups, the six pairs with near-identical ORN
  input, the block), ρ on the per-pair APL effect with the target's, the floor of the loss and the ceiling of ρ
  (data_targets.apl_effect_resamples, as the registry), with 90 % intervals over 1000 resamples of recordings and of
  flies, in Yang's z and in baseline-noise units, for every subset (effect_by_window.csv, pairs_by_window.csv); the
  eight wirings' mean (E72's common error) and the populations of four against each window
  (common_error_by_window.csv); every archived variant against the windows' effects (variants_by_window.csv,
  variants_by_window_summary.csv); response rates per window at a criterion matched on the solvent's presentations
  (rates_by_window.csv, rates_by_window_odors.csv); E82's models by spike window beside them (models_by_window.csv).
- (C) the shape of the responses: mean traces of responders (chosen on two repeats, averaged on the third), entries
  below the criterion, entries at matched amplitude (3 to 6 noise SDs) and all entries, per genotype subset,
  recording, odor group, class of KC by odors per KC and repeat (traces.csv); per trace the time of half rise and
  peak, the value at 2, 8.1, 20 and 40 s over the peak, the 4-8.1 s over the 0-2 s signal and a decay time constant
  from 8.1 s (kinetics_by_recording.csv, kinetics_by_group.csv, kinetics_relations.csv).
- (D) the kernel and the models: kernels fitted to the responders' mean traces under an onset burst, a steady drive
  over the pulse and each saved model's spike-time histogram, fitted on the control flies' trace and tested on the
  trace with APL output blocked (kernel_fits.csv, kernel_fits_models.csv); E82's readings redone through the fitted
  kernels after E82's primary kernel reproduces protocol_readings.csv exactly (protocol_readings_measured_kernel.csv,
  protocol_populations_measured_kernel.csv, gate_measured_kernel.csv); the models' predicted APL-off / APL-on signal by
  window against the flies' (trace_ratio_tests.csv); a recovery test on synthetic traces (kernel_recovery.csv).
- (E) checks: the carry-over from the previous presentation (carry_over.csv, carry_over_pairs.csv) and the neuropil
  coefficient (neuropil_by_window.csv).
- (export) every KC's per-trial excerpts of the 19 recordings from the cache, one parquet per recording, with a README
  and the check that the target's window of the written values reproduces olfdb's (data/yang_apl16_kc_traces/).

Subsets in every table: all 19; APL output blocked (TNT) without 2024-07-17 fly 3 L (its YAML says "APL not labeled");
the six typical and the two strongest (E76); the control flies by label group as recorded in the YAML files (7 / 2 / 2)
and as in the queue's notes (6 / 3 / 2; which side of 2024-07-10 fly 2 is the notes' third TNTin-nolabel recording is
not known, so both readings; the notes' 2024-08-09 fly 3 side is taken as L, the one its YAML calls not labeled).

Usage (from this folder; one thread, no simulations):
    OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 nice -n 10 ../../venv/bin/python -u E84_trace_time_course.py --part A
(then B, C, D, E, export; `--part table` rewrites comparison_table.md and the figures from the CSVs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
import yang_response_normalization as YRN

EXPERIMENT = 'E84_trace_time_course'
ODORS: List[str] = list(H.APL16_ODORS)
SOLVENT = 'pfo'
GENOS = ('TNT3c', 'TNT')
GLAB = {'TNT3c': 'control flies', 'TNT': 'APL output blocked (TNT)'}
DATASET = {'TNT': 19, 'TNT3c': 20}
NEUCOEFF = 0.7
NEUCOEFFS = (0.5, 0.7, 0.9)
EXCERPT_S = (5.0, 40.0)
UNITS = ('z', 'noise', 'dfc')
ESTIMATORS = ('measured', 'noise_free')
BINS: Tuple[Tuple[float, float], ...] = tuple((float(a), float(a + 1)) for a in range(-2, 20))
NAMED_WINDOWS: Dict[str, Tuple[float, float]] = {
    'pulse': (0.0, 2.0), '2-4': (2.0, 4.0), '4-8.1': (4.0, 8.1), 'target': (2.0, 8.1), '0-8.1': (0.0, 8.1),
    '8.1-10.1': (8.1, 10.1), '10.1-12.2': (10.1, 12.2)}
TARGET = NAMED_WINDOWS['target']
STRONG_TNT = (('2024-07-17', 3, 'L'), ('2024-07-25', 1, 'L'))
UNLABELED_TNT = ('2024-07-17', 3, 'L')
# the queue's notes (notes/yang-summary.md): TNT-nolabel 2024-08-02 fly 5 R and 2024-08-30 fly 3 R; TNTin-nolabel
# 2024-08-14 fly 1 L and one side each of 2024-07-10 fly 2 and 2024-08-09 fly 3 (the latter's side taken as L)
NOTES_NOLABEL = {'TNT-nolabel': (('2024-08-02', 5, 'R'), ('2024-08-30', 3, 'R')),
                 'TNTin-nolabel': (('2024-08-14', 1, 'L'), ('2024-08-09', 3, 'L'))}
ESTER = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL = ('1-8ol', '1-5ol', '1-6ol', '6al')
OPK_BINS = (('0', 0, 0), ('1', 1, 1), ('2_3', 2, 3), ('4_7', 4, 7), ('8plus', 8, 99))
N_BOOT = 1000
MATCHED_AMP = (3.0, 6.0)
LATE, EARLY = (4.0, 8.1), (0.0, 2.0)
TRACE_WINDOWS = ((0.0, 2.0), (2.0, 4.0), (4.0, 8.1))
POPULATIONS = ('eight', 'breadth2_four', 'ceiling_four')
COLORS = {'TNT3c': '#2a78d6', 'TNT': '#eb6834', 'six': '#1baf7a', 'two': '#6250d6', 'ink': '#0b0b0b',
          'ink2': '#52514e', 'grid': '#e6e5e1'}


# ---------------------------------------------------------------------------
# paths and small helpers

def out_dir() -> Path:
    """results/E84_trace_time_course/ in the shared directory (statistics over KCs only)."""
    d = H.results_dir(EXPERIMENT)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _private(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    os.chmod(p, 0o700)
    return p


def work_dir() -> Path:
    """outputs/E84_trace_time_course/ (mode 700)."""
    return _private(H.outputs_dir(EXPERIMENT))


def local_dir() -> Path:
    """outputs/E84_trace_time_course/local/ (mode 700): intermediates kept on viki (the per-recording response rates)."""
    work_dir()
    return _private(H.outputs_dir(EXPERIMENT) / 'local')


def cache_dir() -> Path:
    """outputs/E84_trace_time_course/cache/ (mode 700): the per-trial excerpts, Yang's data."""
    work_dir()
    return _private(H.outputs_dir(EXPERIMENT) / 'cache')


def rec_key(genotype: str, rec: str) -> Tuple[str, int, str]:
    """(date, fly_num, hemibrain) of a recording name, as olfdb keys it."""
    fly = rec.split('_')[1]
    return f'{rec[:4]}-{rec[4:6]}-{rec[6:8]}', int(fly[0]), fly[2]


def rec_id(key) -> str:
    return f'{key[0]} fly {key[1]} {key[2]}'


def _git(*args) -> str:
    return subprocess.run(['git', *args], cwd=H.HERE, capture_output=True, text=True).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_provenance(part: str, settings: dict) -> None:
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E84_trace_time_course.py',
                part=part, al_analysis_commit=_git('rev-parse', 'HEAD'),
                al_analysis_dirty=_git('status', '--porcelain', '--untracked-files=no'),
                recordings=[f'{g}:{r}' for g, r in YRN.RECORDINGS], neucoeff=NEUCOEFF, excerpt_s=list(EXCERPT_S),
                approx_conc_note=H.APPROX_CONC_NOTE, **settings)
    (out_dir() / f'provenance_{part}.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')


_PAIR_IDX = pd.MultiIndex.from_tuples([tuple(p) for p in H.PAIRS16], names=['odor_a', 'odor_b'])


def _within(group: Sequence[str]) -> List[Tuple[str, str]]:
    def canon(a, b):
        return (a, b) if ODORS.index(a) < ODORS.index(b) else (b, a)
    return [canon(a, b) for i, a in enumerate(group) for b in group[i + 1:]]


SIX = (('eb', 'ep'), ('6al', 't2h'), ('IaA', 'pa'), ('2h', 'pa'), ('1-5ol', '1-6ol'), ('2h', 'IaA'))
NEAREST = (('2h', 'IaA'), ('2h', 'pa'), ('IaA', 'pa'), ('eb', 'ep'))
BLOCK = tuple((a, b) for a in ('2h', 'IaA', 'pa') for b in ('eb', 'ep'))


def _mask(pairs) -> np.ndarray:
    want = {tuple(sorted(p)) for p in pairs}
    return np.array([tuple(sorted(p)) in want for p in _PAIR_IDX])


PAIR_MASKS: Dict[str, np.ndarray] = {'ester': _mask(_within(ESTER)), 'alcohol': _mask(_within(ALCOHOL)),
                                     'six': _mask(SIX), 'nearest': _mask(NEAREST), 'block': _mask(BLOCK)}


def _pairs_array(s: pd.Series) -> np.ndarray:
    """A pair Series as an array in PAIRS16 order (pairs in either order)."""
    s = H.canonical_pairs(s)
    return s.reindex(_PAIR_IDX).to_numpy(dtype=float)


def _rank_rows(X: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(X, axis=-1)


def _spearman_rows(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of X with y (rows broadcast against y)."""
    rx = _rank_rows(X)
    ry = _rank_rows(np.broadcast_to(y, X.shape))
    rx = rx - rx.mean(axis=-1, keepdims=True)
    ry = ry - ry.mean(axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (rx * ry).sum(-1) / np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))


# ---------------------------------------------------------------------------
# part A: the reader, the cache and the records

def label_as_recorded(note: str, hemi: str) -> str:
    """'label', 'nolabel' or 'unknown' for this hemisphere from the YAML's Note (e.g. "APL labeled", "Both labeled",
    "R weakly labeled", "Not labeled", "APL not labeled")."""
    n = str(note).lower()
    if 'not labeled' in n or 'not labelled' in n:
        return 'nolabel'
    if 'both labeled' in n or 'apl labeled' in n:
        return 'label'
    m = re.search(r'\b([lr])\s+(weakly\s+)?labell?ed', n)
    if m:
        return 'label' if m.group(1).upper() == hemi else 'nolabel'
    return 'unknown'


def construct_as_recorded(genotype_text: str) -> str:
    """'TNT' (UAS-TNT) or 'TNTin' (UAS-TNT-in, the inactive toxin) from the YAML's Genotype."""
    g = str(genotype_text)
    if re.search(r'UAS-TNT-?in', g):
        return 'TNTin'
    if 'UAS-TNT' in g:
        return 'TNT'
    return 'unknown'


YAML_FIELDS = ('Genotype', 'Note', 'Thirtytwo_since', 'Thirtytwo_till', 'Thirtytwo_hours', 'frame_rate',
               'frame_per_volume', 'nblock', 'nstim', 'odorPulseLen', 'odorStartInterval', 'interOdorInterval',
               'basePeriodLen', 'peak_start', 'peak_end', 'MFC', 'pockel', 'laserpower', 'gain_A')
OPS_FIELDS = ('suite2p_version', 'nframes', 'neucoeff', 'nplanes', 'fs')


def _sd_base(Fc: np.ndarray, onsets: np.ndarray, B: int) -> np.ndarray:
    """Each KC's SD of Fc within the pre-odor windows, each window's mean removed (yang_response_normalization)."""
    K = len(onsets)
    base = np.concatenate([Fc[:, n - B:n] - Fc[:, n - B:n].mean(axis=1, keepdims=True) for n in onsets], axis=1)
    return np.sqrt((base ** 2).sum(axis=1) / (base.shape[1] - K))


def excerpt_recording(genotype: str, rec: str, drive_root: Path) -> dict:
    """One recording read (`YRN.read_suite2p`) and cut into per-trial excerpts: dfc (trials, KCs, samples) float32 at
    c = 0.7, dfneu (the same of Fneu), sd_full / sd_base per c in NEUCOEFFS, base_fc / base_fneu (each trial's pre-onset
    mean of Fc and Fneu), the odor and repeat of each trial in time order, the onset volumes, the volume rate, the
    records from the YAML, each KC's mean F / Fneu."""
    r = YRN.read_suite2p(genotype, rec, drive_root)
    cfg, ops = r['cfg'], r['ops']
    v = float(cfg['frame_rate']) / float(cfg['frame_per_volume'])
    B = int(round(cfg['basePeriodLen'] * v))
    S, P = int(round(cfg['peak_start'] * v)), int(round(cfg['peak_end'] * v))
    n_after = int(round(EXCERPT_S[1] * v)) + 1
    F, Fn = r['F'], r['Fneu']
    onsets = np.asarray(r['onsets'], dtype=np.int64)
    sd_full, sd_base = {}, {}
    for c in NEUCOEFFS:
        Fc = F - c * Fn
        sd_full[c] = Fc.std(axis=1)
        sd_base[c] = _sd_base(Fc, onsets, B)
    Fc = F - 0.7 * Fn
    dfc = H.trace_excerpts(Fc, onsets, B, n_after).astype(np.float32)
    dfneu = H.trace_excerpts(Fn, onsets, B, n_after).astype(np.float32)
    base_fc = np.stack([Fc[:, n - B:n].mean(axis=1) for n in onsets])
    base_fneu = np.stack([Fn[:, n - B:n].mean(axis=1) for n in onsets])
    names = [YRN.ODOR_NAMES[p] for p in r['pins']]
    abbrev = [H.odor_abbrev(n) for n in names]
    rep = [int(sum(1 for m in abbrev[:i] if m == a)) for i, a in enumerate(abbrev)]
    meta = dict(genotype=genotype, recording=rec, key=list(rec_key(genotype, rec)), volume_rate=v, B=B, S=S, P=P,
                n_after=n_after, n_volumes=int(F.shape[1]), n_kcs=int(F.shape[0]), n_rois=r['n_rois'],
                odors=abbrev, odor_names=names, repeat=rep, onsets=onsets.tolist(),
                yaml={k: (cfg.get(k) if not isinstance(cfg.get(k), (list, dict)) else str(cfg.get(k))) for k in YAML_FIELDS},
                ops={k: (ops[k] if not isinstance(ops.get(k), (list, np.ndarray)) else str(ops[k])) for k in OPS_FIELDS
                     if k in ops})
    out = dict(dfc=dfc, dfneu=dfneu, base_fc=base_fc, base_fneu=base_fneu, F_mean=F.mean(axis=1), Fneu_mean=Fn.mean(axis=1),
               **{f'sd_full_{c:g}': sd_full[c] for c in NEUCOEFFS}, **{f'sd_base_{c:g}': sd_base[c] for c in NEUCOEFFS})
    out['meta'] = np.array(json.dumps(meta, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    return out


def _cache_path(genotype: str, rec: str) -> Path:
    return cache_dir() / f'{genotype}_{rec}.npz'


def build_cache(drive_root: Path, rebuild: bool = False) -> None:
    """Writes one cache file per recording (`excerpt_recording`), one recording in memory at a time."""
    for g, rec in YRN.RECORDINGS:
        p = _cache_path(g, rec)
        if p.exists() and not rebuild:
            continue
        t0 = time.time()
        c = excerpt_recording(g, rec, drive_root)
        tmp = p.with_suffix('.tmp.npz')
        np.savez(tmp, **c)
        os.chmod(tmp, 0o600)
        tmp.replace(p)
        print(f'cache {g} {rec}: {c["dfc"].shape} in {time.time() - t0:.0f} s', flush=True)
        del c


def load_cache(genotype: str, rec: str) -> dict:
    """A recording's cached excerpts and metadata (`excerpt_recording`'s dict, with 'meta' parsed)."""
    if (genotype, rec) not in YRN.RECORDINGS:
        raise ValueError(f'{genotype} {rec} is not one of the 19 recordings')
    with np.load(_cache_path(genotype, rec), allow_pickle=False) as z:
        c = {k: z[k] for k in z.files}
    c['meta'] = json.loads(str(c['meta']))
    return c


def window_values(c: dict, wins: Sequence[Tuple[float, float]], units: str = 'z', neucoeff: float = NEUCOEFF) -> np.ndarray:
    """(trials in time order, KCs, windows) of a recording's single-trial responses in each window (seconds) and units."""
    m = c['meta']
    if neucoeff == NEUCOEFF:
        X = c['dfc']
    else:
        X = c['dfc'].astype(np.float64) - (neucoeff - NEUCOEFF) * c['dfneu'].astype(np.float64)
    W = H.windowed_responses(X, wins, first_volume=-m['B'], volume_rate=m['volume_rate'])
    if units == 'dfc':
        return W
    sc = c[f'sd_full_{neucoeff:g}'] if units == 'z' else c[f'sd_base_{neucoeff:g}']
    if units not in ('z', 'noise'):
        raise ValueError(units)
    return W / sc[None, :, None]


def trial_index(c: dict, odors: Sequence[str] = ODORS) -> np.ndarray:
    """(repeats, odors) trial positions (in time order) of each odor's repeats."""
    m = c['meta']
    ab, rep = np.array(m['odors']), np.array(m['repeat'])
    nrep = int(rep.max()) + 1
    idx = np.full((nrep, len(odors)), -1, dtype=int)
    for j, o in enumerate(odors):
        for r in range(nrep):
            hit = np.flatnonzero((ab == o) & (rep == r))
            if len(hit) != 1:
                raise RuntimeError(f"{m['recording']}: {len(hit)} trials of {o} repeat {r}")
            idx[r, j] = hit[0]
    return idx


def arrange(W: np.ndarray, c: dict, odors: Sequence[str] = ODORS) -> np.ndarray:
    """(trials, KCs, ...) -> (repeats, KCs, odors, ...)."""
    idx = trial_index(c, odors)
    return np.moveaxis(W[idx], 2, 1)


def trial_array(c: dict, window: Tuple[float, float], units: str = 'z', neucoeff: float = NEUCOEFF,
                odors: Sequence[str] = ODORS) -> np.ndarray:
    """(repeats, KCs, odors) of a recording's single-trial responses in `window` (seconds) and `units`."""
    return arrange(window_values(c, [window], units, neucoeff), c, odors)[..., 0]


def times(c: dict) -> np.ndarray:
    """Each excerpt sample's time from the onset (s): volume k at k / v."""
    m = c['meta']
    return (np.arange(m['B'] + m['n_after']) - m['B']) / m['volume_rate']


def records_row(c: dict) -> dict:
    m = c['meta']
    y = m['yaml']
    key = tuple(m['key'])
    label = label_as_recorded(y.get('Note'), key[2])
    construct = construct_as_recorded(y.get('Genotype'))
    group = f'{construct}-{label}'
    agrees = (group == 'TNT-label') if m['genotype'] == 'TNT' else (group in ('TNTin-label', 'TNTin-nolabel', 'TNT-nolabel'))
    gaps = np.diff(m['onsets'])
    return dict(genotype=m['genotype'], dataset=DATASET[m['genotype']], recording=m['recording'], date=key[0],
                fly_num=key[1], hemibrain=key[2], yaml_genotype=y.get('Genotype'), yaml_note=y.get('Note'),
                construct_as_recorded=construct, label_as_recorded=label, group_as_recorded=group,
                agrees_with_dataset=bool(agrees), group_notes_0710L=notes_group(m['genotype'], key, 'L'),
                group_notes_0710R=notes_group(m['genotype'], key, 'R'),
                induction_hours=y.get('Thirtytwo_hours'), induction_since=y.get('Thirtytwo_since'),
                induction_till=y.get('Thirtytwo_till'), frame_rate=y.get('frame_rate'),
                frame_per_volume=y.get('frame_per_volume'), volume_rate=m['volume_rate'], n_volumes=m['n_volumes'],
                n_rois=m['n_rois'], n_kcs=m['n_kcs'], n_trials=len(m['onsets']), odor_pulse_s=y.get('odorPulseLen'),
                inter_odor_s=y.get('interOdorInterval'), base_s=y.get('basePeriodLen'), peak_start=y.get('peak_start'),
                peak_end=y.get('peak_end'), mfc=y.get('MFC'), pockel=y.get('pockel'), laserpower=y.get('laserpower'),
                gain_A=y.get('gain_A'), suite2p_version=m['ops'].get('suite2p_version'),
                suite2p_neucoeff=m['ops'].get('neucoeff'), onset_gap_min=int(gaps.min()), onset_gap_max=int(gaps.max()),
                median_F=float(np.median(c['F_mean'])), median_Fneu=float(np.median(c['Fneu_mean'])),
                median_sd_base=float(np.median(c['sd_base_0.7'])), median_sd_full=float(np.median(c['sd_full_0.7'])))


def notes_group(genotype: str, key, side_0710: str) -> str:
    """The recording's label group as the queue's notes give it (6 / 3 / 2), with 2024-07-10 fly 2 `side_0710` taken as
    the notes' TNTin-nolabel side."""
    key = tuple(key)
    if genotype == 'TNT':
        return 'TNT-label'
    if key in NOTES_NOLABEL['TNT-nolabel']:
        return 'TNT-nolabel'
    if key in NOTES_NOLABEL['TNTin-nolabel'] or key == ('2024-07-10', 2, side_0710):
        return 'TNTin-nolabel'
    return 'TNTin-label'


def _olfdb_noise(nz: pd.DataFrame, key) -> pd.DataFrame:
    d = nz[(nz.date == key[0]) & (nz.fly_num == key[1]) & (nz.hemibrain == key[2])]
    return d.pivot_table(index='roi', columns=['odor', 'repeat'], values='value')


def part_a(drive_root: Path, rebuild: bool = False) -> dict:
    """Part A: the cache, the reproduction checks (RuntimeError on failure, after reproduction_checks.json is written),
    records_summary.csv, presentation_order.csv, order_summary.csv, succession_counts.csv."""
    write_provenance('A', dict(drive_root=str(drive_root)))
    build_cache(drive_root, rebuild)
    nu = H.shared_dir() / 'data' / 'yang_apl16_kc_noise_units'
    nz = pd.read_parquet(nu / 'per_trial_long.parquet', columns=['genotype', 'date', 'fly_num', 'hemibrain', 'roi', 'odor',
                                                                  'repeat', 'value'])
    scales = pd.read_csv(nu / 'kc_scales.csv', dtype={'date': str, 'hemibrain': str})
    rdms = {g: H.yang_recording_rdms(g, 'cosine') for g in GENOS}
    checks, records, orders = {}, [], []
    ours_rec = {g: {} for g in GENOS}
    nf_rec = {g: {} for g in GENOS}
    for g, rec in YRN.RECORDINGS:
        c = load_cache(g, rec)
        m = c['meta']
        key = tuple(m['key'])
        if H.window_volumes(TARGET, m['volume_rate']) != (m['S'], m['P']):
            raise RuntimeError(f'{rec}: the 2.0-8.1 s window is not volumes {m["S"]} to {m["P"] - 1}')
        Wz = window_values(c, [TARGET], 'z')[..., 0]
        Wn = window_values(c, [TARGET], 'noise')[..., 0]
        cols = pd.MultiIndex.from_arrays([m['odor_names'], m['repeat']])
        om = YRN.olfdb_matrix(g, rec)
        ours = pd.DataFrame(Wz.T, columns=cols)
        common = [x for x in ours.columns if x in om.columns]
        d_olfdb = float(np.abs(om[common].to_numpy() - ours[common].to_numpy()).max()) if len(om) == len(ours) else np.inf
        on_ = _olfdb_noise(nz[nz.genotype == g], key)
        ourn = pd.DataFrame(Wn.T, columns=cols)
        common_n = [x for x in ourn.columns if x in on_.columns]
        d_noise = float((np.abs(on_[common_n].to_numpy() - ourn[common_n].to_numpy())
                         / np.maximum(1.0, np.abs(on_[common_n].to_numpy()))).max()) if len(on_) == len(ourn) else np.inf
        sc = scales[(scales.date == key[0]) & (scales.fly_num == key[1]) & (scales.hemibrain == key[2])].sort_values('roi')
        d_sdb = float(np.abs(sc.sd_base.to_numpy() / c['sd_base_0.7'] - 1).max()) if len(sc) == m['n_kcs'] else np.inf
        d_sdf = float(np.abs(sc.sd_full.to_numpy() / c['sd_full_0.7'] - 1).max()) if len(sc) == m['n_kcs'] else np.inf
        T = trial_array(c, TARGET, 'z')
        tm = pd.DataFrame(T.mean(axis=0), columns=ODORS)
        dist = H.pair_distances(tm, 'cosine')
        ref = H.canonical_pairs(rdms[g].loc[key])
        d_rdm = float(np.abs(_pairs_array(dist) - _pairs_array(ref)).max())
        ours_rec[g][key] = _pairs_array(dist)
        nf_rec[g][key] = _pairs_array(H.noise_free_pair_distances(T, ODORS))
        checks[f'{g}:{rec}'] = dict(olfdb_max_abs=d_olfdb, olfdb_trials=len(common), noise_units_max_rel=d_noise,
                                    sd_base_max_rel=d_sdb, sd_full_max_rel=d_sdf, rdm_max_abs=d_rdm,
                                    n_kcs=m['n_kcs'], volume_rate=m['volume_rate'], window_volumes=[m['S'], m['P'] - 1])
        records.append(records_row(c))
        for i, (o, rp) in enumerate(zip(m['odors'], m['repeat'])):
            nstim = int(m['yaml']['nstim'])
            orders.append(dict(recording=rec, genotype=g, block=i // nstim, position=i % nstim, odor=m['odor_names'][i],
                               date=key[0], fly_num=key[1], hemibrain=key[2], presentation=i, odor_abbrev=o, repeat=rp,
                               onset_volume=m['onsets'][i]))
        print(f"A {g} {rec}: olfdb {d_olfdb:.2g}, noise units {d_noise:.2g}, sd_base {d_sdb:.2g}, rdm {d_rdm:.2g}", flush=True)
        del c
    # the target's two matrices and E81's noise-free registry entries
    mean_rdm = {g: np.mean(list(ours_rec[g].values()), axis=0) for g in GENOS}
    d_mean = {g: float(np.abs(mean_rdm[g] - _pairs_array(H.yang_mean_rdm(g, 'cosine'))).max()) for g in GENOS}
    nf = {g: np.mean(list(nf_rec[g].values()), axis=0) for g in GENOS}
    eff = nf['TNT3c'] - nf['TNT']
    reg_vals = dict(noise_free_mean_dist_ctrl=float(np.nanmean(nf['TNT3c'])), noise_free_mean_dist_tnt=float(np.nanmean(nf['TNT'])),
                    noise_free_apl_effect_mean=float(eff.mean()), noise_free_apl_effect_six=float(eff[PAIR_MASKS['six']].mean()),
                    noise_free_apl_effect_ester=float(eff[PAIR_MASKS['ester']].mean()),
                    noise_free_apl_effect_alcohol=float(eff[PAIR_MASKS['alcohol']].mean()),
                    noise_free_six_ctrl=float(nf['TNT3c'][PAIR_MASKS['six']].mean()),
                    noise_free_six_tnt=float(nf['TNT'][PAIR_MASKS['six']].mean()))
    reg = {k: H.data_target(k) for k in reg_vals}
    d_reg = {k: abs(reg_vals[k] - reg[k]) for k in reg_vals}
    summary = dict(olfdb_max_abs=max(v['olfdb_max_abs'] for v in checks.values()),
                   noise_units_max_rel=max(v['noise_units_max_rel'] for v in checks.values()),
                   sd_base_max_rel=max(v['sd_base_max_rel'] for v in checks.values()),
                   rdm_max_abs=max(v['rdm_max_abs'] for v in checks.values()), mean_rdm_max_abs=d_mean,
                   registry=dict(ours=reg_vals, registry=reg, abs_diff=d_reg, max_abs_diff=max(d_reg.values())))
    ok = (summary['olfdb_max_abs'] <= 1e-5 and summary['noise_units_max_rel'] <= 1e-5 and summary['sd_base_max_rel'] <= 1e-9
          and max(d_mean.values()) <= 1e-6 and summary['rdm_max_abs'] <= 1e-6 and summary['registry']['max_abs_diff'] <= 1e-6)
    rec_out = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), al_analysis_commit=_git('rev-parse', 'HEAD'),
                   tolerances=dict(olfdb_max_abs=1e-5, noise_units_max_rel=1e-5, sd_base_max_rel=1e-9, rdm_max_abs=1e-6,
                                   mean_rdm_max_abs=1e-6, registry=1e-6),
                   all_ok=bool(ok), summary=summary, recordings=checks)
    (out_dir() / 'reproduction_checks.json').write_text(json.dumps(rec_out, indent=1, default=float) + '\n')
    pd.DataFrame(records).to_csv(out_dir() / 'records_summary.csv', index=False)
    od = pd.DataFrame(orders)
    od.to_csv(out_dir() / 'presentation_order.csv', index=False)
    order_summary(od.rename(columns={'odor': 'odor_name', 'odor_abbrev': 'odor'}))
    print(json.dumps(summary, indent=1, default=float), flush=True)
    if not ok:
        raise RuntimeError('part A: the reproduction check failed (reproduction_checks.json)')
    return rec_out


def order_summary(od: pd.DataFrame) -> None:
    """order_summary.csv (counts of distinct orders: of whole sessions, of blocks, blocks per recording that differ) and
    succession_counts.csv (for each odor, how many of its presentations follow each other odor, over all recordings)."""
    seq = {r: tuple(g.sort_values('presentation').odor) for r, g in od.groupby('recording')}
    blocks = {(r, b): tuple(g.sort_values('presentation').odor) for (r, b), g in od.groupby(['recording', 'block'])}
    per_rec_distinct = [len({blocks[(r, b)] for b in sorted({k[1] for k in blocks if k[0] == r})}) for r in seq]
    rows = [dict(statistic='recordings', value=len(seq)),
            dict(statistic='distinct session orders (51 presentations)', value=len(set(seq.values()))),
            dict(statistic='blocks', value=len(blocks)),
            dict(statistic='distinct block orders (17 presentations)', value=len(set(blocks.values()))),
            dict(statistic='recordings whose three blocks all differ', value=int(sum(1 for n in per_rec_distinct if n == 3))),
            dict(statistic='largest number of recordings sharing a session order',
                 value=int(pd.Series(list(seq.values())).value_counts().max()))]
    pd.DataFrame(rows).to_csv(out_dir() / 'order_summary.csv', index=False)
    succ = []
    for r, s in seq.items():
        for a, b in zip(s[:-1], s[1:]):
            succ.append((a, b))
    sc = pd.Series(succ).value_counts()
    n_next = od.groupby('odor').size()
    tab = pd.DataFrame([dict(previous=a, next=b, count=int(n), presentations_of_next=int(n_next[b]),
                             share=float(n / n_next[b])) for (a, b), n in sc.items()])
    tab = tab.sort_values(['count', 'next'], ascending=[False, True])
    tab.to_csv(out_dir() / 'succession_counts.csv', index=False)


# ---------------------------------------------------------------------------
# part B: the target by window

def windows() -> List[Tuple[str, float, float]]:
    """(name, start, end) of every window of part B: the 1 s bins, then NAMED_WINDOWS."""
    return [(f'{a:g} to {b:g} s', a, b) for a, b in BINS] + [(k, a, b) for k, (a, b) in NAMED_WINDOWS.items()]


def subsets() -> Dict[str, Dict[str, list]]:
    """{subset name: {genotype: [recording keys]}} of the subsets every table is given for."""
    keys = {g: [rec_key(g, r) for gg, r in YRN.RECORDINGS if gg == g] for g in GENOS}
    rec = pd.read_csv(out_dir() / 'records_summary.csv', dtype={'date': str, 'hemibrain': str})
    rk = {(r.date, int(r.fly_num), r.hemibrain): r for r in rec.itertuples()}
    ctrl, tnt = keys['TNT3c'], keys['TNT']
    out = {'all': dict(TNT3c=ctrl, TNT=tnt),
           'tnt_without_unlabeled': dict(TNT3c=ctrl, TNT=[k for k in tnt if k != UNLABELED_TNT]),
           'six_typical': dict(TNT3c=ctrl, TNT=[k for k in tnt if k not in STRONG_TNT]),
           'two_strongest': dict(TNT3c=ctrl, TNT=[k for k in tnt if k in STRONG_TNT])}
    for grp in ('TNTin-label', 'TNTin-nolabel', 'TNT-nolabel'):
        out[f'ctrl_{grp}_recorded'] = dict(TNT3c=[k for k in ctrl if rk[k].group_as_recorded == grp], TNT=tnt)
        for side in ('L', 'R'):
            out[f'ctrl_{grp}_notes_0710{side}'] = dict(TNT3c=[k for k in ctrl if getattr(rk[k], f'group_notes_0710{side}') == grp],
                                                       TNT=tnt)
    return out


def _fly_groups(keys: list) -> List[np.ndarray]:
    flies = {}
    for i, k in enumerate(keys):
        flies.setdefault((k[0], k[1]), []).append(i)
    return [np.array(v) for v in flies.values()]


def _boot(tabs: Dict[str, np.ndarray], keys: Dict[str, list], fn: Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]],
          n_boot: int, unit: str, seed: int = 0) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """{statistic: (observed, lo, hi)} of `fn` of the recording means of `tabs` ({genotype: (recordings, ...)}), 90 %
    intervals over `n_boot` resamples of recordings or of flies (all recordings of a drawn fly), each genotype apart."""
    obs = fn({g: t.mean(axis=0) for g, t in tabs.items()})
    rng = np.random.default_rng(seed)
    groups = {g: _fly_groups(keys[g]) for g in tabs}
    draws = {k: np.full((n_boot,) + np.shape(v), np.nan) for k, v in obs.items()}
    for b in range(n_boot):
        m = {}
        for g, t in tabs.items():
            if unit == 'recording':
                idx = rng.integers(0, len(t), len(t))
            else:
                gr = groups[g]
                idx = np.concatenate([gr[i] for i in rng.integers(0, len(gr), len(gr))])
            m[g] = t[idx].mean(axis=0)
        d = fn(m)
        for k in draws:
            draws[k][b] = d[k]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for k, v in obs.items():
            lo, hi = np.nanpercentile(draws[k], [5, 95], axis=0)
            out[k] = (np.asarray(v), lo, hi)
    return out


def recording_tables() -> Tuple[Dict[str, np.ndarray], Dict[str, list], List[tuple], pd.DataFrame]:
    """Per genotype, (recordings, rows, 120) pair distances by window: rows = (units, estimator, window) in `row_keys`
    order; and the per-recording response-rate rows. Cached in outputs/ (recording_tables.npz, rates_recordings.csv)."""
    wl = windows()
    row_keys = [(u, e, w[0]) for u in ('z', 'noise') for e in ESTIMATORS for w in wl]
    path = work_dir() / 'recording_tables.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            if meta['row_keys'] == [list(r) for r in row_keys]:
                tabs = {g: z[g] for g in GENOS}
                keys = {g: [tuple(k) for k in meta['keys'][g]] for g in GENOS}
                return tabs, keys, row_keys, recording_rates()
    tabs = {g: [] for g in GENOS}
    keys = {g: [] for g in GENOS}
    wins = [(a, b) for _, a, b in wl]
    for g, rec in YRN.RECORDINGS:
        t0 = time.time()
        c = load_cache(g, rec)
        key = tuple(c['meta']['key'])
        rows = []
        for u in ('z', 'noise'):
            W = window_values(c, wins, u)
            T = arrange(W, c, ODORS)                                       # (3, KCs, 16, nW)
            for e in ESTIMATORS:
                for wi in range(len(wins)):
                    X = T[..., wi]
                    if e == 'measured':
                        d = H.pair_distances(pd.DataFrame(X.mean(axis=0), columns=ODORS), 'cosine')
                    else:
                        d = H.noise_free_pair_distances(X, ODORS)
                    rows.append(_pairs_array(d))
        tabs[g].append(np.stack(rows))
        keys[g].append(key)
        print(f'B tables {g} {rec}: {time.time() - t0:.0f} s', flush=True)
        del c
    tabs = {g: np.stack(v) for g, v in tabs.items()}
    meta = dict(row_keys=[list(r) for r in row_keys], keys={g: [list(k) for k in v] for g, v in keys.items()})
    np.savez(path, meta=np.array(json.dumps(meta)), **tabs)
    return tabs, keys, row_keys, recording_rates()


RATE_CRITERIA = ('trialmean', 'single', 'trialmean_mad', 'single_mad', 'trialmean_fixed')


def _robust_sd(x: np.ndarray, axis: int = 0) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def recording_rates() -> pd.DataFrame:
    """Per recording, criterion and window, the response rates in baseline-noise units (local_dir()/rates_recordings.csv,
    cached). The bands' criterion (a trial mean above `noise_threshold`) holds at the target's window; elsewhere the
    threshold is, per recording: 'trialmean' / 'single' (trial means / single trials), the quantile of the solvent's
    presentations' values in that window that keeps their share above it at the target window's (which includes the
    KCs that respond to the solvent); '..._mad', the target's threshold scaled by the robust SD (1.4826 MAD) of the
    solvent's values in that window over the target window's (the same false-positive rate under Gaussian noise);
    'trialmean_fixed', the bands' threshold in every window."""
    rp = local_dir() / 'rates_recordings.csv'
    if rp.exists():
        d = pd.read_csv(rp, dtype={'date': str, 'hemibrain': str})
        if set(d.criterion) == set(RATE_CRITERIA):
            return d
    thr = float(H.data_target('noise_threshold'))
    wl = windows()
    wins = [(a, b) for _, a, b in wl]
    ti = [w[0] for w in wl].index('target')
    rate_rows = []
    for g, rec in YRN.RECORDINGS:
        c = load_cache(g, rec)
        key = tuple(c['meta']['key'])
        W = window_values(c, wins, 'noise')
        T = arrange(W, c, ODORS)                                           # (3, KCs, 16, nW)
        Tp = arrange(W, c, [SOLVENT])[:, :, 0, :]                          # (3, KCs, nW)
        for crit in RATE_CRITERIA:
            single = crit.startswith('single')
            blank = Tp.reshape(-1, Tp.shape[-1]) if single else Tp.mean(axis=0)
            fpr = float((blank[:, ti] > thr).mean())
            rsd = _robust_sd(blank, axis=0)
            for wi, (name, a, b) in enumerate(wl):
                if wi == ti or crit == 'trialmean_fixed':
                    th = thr
                elif crit.endswith('_mad'):
                    th = float(thr * rsd[wi] / rsd[ti])
                elif fpr > 0:
                    th = float(np.quantile(blank[:, wi], 1 - fpr))
                else:
                    th = float(blank[:, wi].max())
                X = T[..., wi]
                R = (X > th) if single else (X.mean(axis=0) > th)
                per_odor = R.mean(axis=tuple(i for i in range(R.ndim) if i != R.ndim - 1))
                row = dict(genotype=g, date=key[0], fly_num=key[1], hemibrain=key[2], criterion=crit, window=name, t0=a,
                           t1=b, threshold=th, blank_fpr_target=fpr, blank_robust_sd=float(rsd[wi]),
                           blank_rate=float((blank[:, wi] > th).mean()), rate=float(R.mean()),
                           **{f'rate_{o}': float(per_odor[j]) for j, o in enumerate(ODORS)})
                if not single:
                    n = R.sum(axis=1)
                    row.update({f'opk_{lab}': float(((n >= lo) & (n <= hi)).mean()) for lab, lo, hi in OPK_BINS})
                rate_rows.append(row)
        del c
    rates = pd.DataFrame(rate_rows)
    rates.to_csv(rp, index=False)
    return rates


_POP: Dict[str, Dict[str, np.ndarray]] = {}


def population_effects() -> Dict[str, Dict[str, np.ndarray]]:
    """{population: {readout ('raw', 'cal'): per-pair APL effect in PAIRS16 order}}: E72's eight wirings (whose error is
    the common error) and E81's populations of four (the breadth-2 family's four wirings through the pipeline and the
    firing-ceiling family's four). Cached in outputs/ (population_effects.csv)."""
    if _POP:
        return _POP
    path = work_dir() / 'population_effects.csv'
    if path.exists():
        d = pd.read_csv(path)
        for (p, ro), g in d.groupby(['population', 'readout']):
            _POP.setdefault(p, {})[ro] = _pairs_array(g.set_index(['odor_a', 'odor_b'])['effect'])
        return _POP
    import E72_wiring_population as E72
    import E81_repeat_structure as E81
    pops = E81.populations()
    members = {'eight': E72.new_fit_members(), 'breadth2_four': pops['breadth2_four'], 'ceiling_four': pops['ceiling_four']}
    rows = []
    for p, mem in members.items():
        fits = [E72.load_fit(f, t) for _, f, t in mem]
        if any(o is None for o in fits):
            print(f'population {p}: members without saved outputs; skipped', flush=True)
            continue
        for ro in ('raw', 'cal'):
            md = E72.mean_distances([E72.distances(o, ro) for o in fits])
            e = H.canonical_pairs(H.apl_effect(md['on'], md['off'])).reindex(_PAIR_IDX)
            _POP.setdefault(p, {})[ro] = e.to_numpy(dtype=float)
            rows += [dict(population=p, readout=ro, odor_a=a, odor_b=b, effect=float(v)) for (a, b), v in e.items()]
    pd.DataFrame(rows).to_csv(path, index=False)
    return _POP


def _stat_fn(row_keys: List[tuple], main_target: np.ndarray, pops: Dict[str, Dict[str, np.ndarray]]):
    """The statistics of every (units, estimator, window) row of the recording means {genotype: (rows, 120)}."""
    ref = np.array([row_keys.index((u, e, 'target')) for u, e, _ in row_keys])
    late = np.array([row_keys.index((u, e, '4-8.1')) for u, e, _ in row_keys])
    em, am = PAIR_MASKS['ester'], PAIR_MASKS['alcohol']

    def fn(m):
        C, T = m['TNT3c'], m['TNT']
        E = C - T
        out = dict(ctrl_mean=np.nanmean(C, 1), tnt_mean=np.nanmean(T, 1), effect_mean=np.nanmean(E, 1))
        for ps, mk in PAIR_MASKS.items():
            out[f'effect_{ps}'] = E[:, mk].mean(1)
            if ps in ('ester', 'alcohol', 'six'):
                out[f'ctrl_{ps}'] = C[:, mk].mean(1)
                out[f'tnt_{ps}'] = T[:, mk].mean(1)
        out['rho_main_target'] = _spearman_rows(E, main_target)
        out['rho_own_target'] = _spearman_rows(E, E[ref])
        for k in ('effect_mean', 'effect_ester', 'effect_alcohol', 'effect_six', 'ctrl_mean', 'tnt_mean'):
            out[f'{k}_minus_target'] = out[k] - out[k][ref]
            out[f'{k}_minus_late'] = out[k] - out[k][late]
        out['alcohol_minus_ester_change_from_late'] = out['effect_alcohol_minus_late'] - out['effect_ester_minus_late']
        for p, d in pops.items():
            for ro, P in d.items():
                err = P[None] - E
                me = err.mean(1)
                errc = err - me[:, None]
                pat = np.sqrt((errc ** 2).mean(1))
                pre = f'pop:{p}:{ro}:'
                out[pre + 'loss'] = np.sqrt((err ** 2).mean(1))
                out[pre + 'pattern'] = pat
                out[pre + 'mean_error'] = me
                out[pre + 'ester_error'] = errc[:, em].mean(1)
                out[pre + 'alcohol_error'] = errc[:, am].mean(1)
                out[pre + 'rho'] = _spearman_rows(E, P)
                out[pre + 'pattern_minus_target'] = pat - pat[ref]
        return out
    return fn


def floor_ceiling(tab_g: Dict[str, np.ndarray], seed: int = 0) -> Dict[str, np.ndarray]:
    """The floor of the APL-effect loss (its mean and 90 % interval over 1000 resamples of recordings: the RMS difference
    of a resampled effect from the measured one) and the ceiling of ρ (Spearman-Brown-corrected ρ between the effects of
    500 random split halves of the recordings) of every row of `tab_g` ({genotype: (recordings, rows, 120)}), with the
    random draws of data_targets.apl_effect_resamples in the same order from the same seed (they do not depend on the
    data, so every row gets the draws the registry's computation would give it). Returns {'floor', 'floor_lo',
    'floor_hi', 'ceiling'}: arrays over rows."""
    rng = np.random.default_rng(seed)
    C, T = tab_g['TNT3c'], tab_g['TNT']
    eff = C.mean(0) - T.mean(0)
    rmse = []
    for _ in range(1000):
        ic = rng.integers(0, len(C), len(C))
        it = rng.integers(0, len(T), len(T))
        e_b = C[ic].mean(0) - T[it].mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            rmse.append(np.sqrt(np.nanmean((e_b - eff) ** 2, axis=1)))
    rhos = []
    for _ in range(500):
        pc, pt = rng.permutation(len(C)), rng.permutation(len(T))
        a = C[pc[:len(C) // 2]].mean(0) - T[pt[:len(T) // 2]].mean(0)
        b = C[pc[len(C) // 2:]].mean(0) - T[pt[len(T) // 2:]].mean(0)
        rhos.append(_spearman_rows(a, b))
    rmse, rhos = np.array(rmse), np.array(rhos)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        r = np.nanmean(rhos, axis=0)
        lo, hi = np.nanpercentile(rmse, [5, 95], axis=0)
        return dict(floor=rmse.mean(0), floor_lo=lo, floor_hi=hi, ceiling=np.sqrt(2 * r / (1 + r)))


def floor_ceiling_check(tab_g: Dict[str, np.ndarray], row: int) -> Dict[str, float]:
    """data_targets.apl_effect_resamples itself on one row, against `floor_ceiling`'s values for it."""
    import data_targets as DT
    recs = {g: pd.DataFrame(t[:, row, :], columns=_PAIR_IDX) for g, t in tab_g.items()}
    _, _, boot_rmse, sh_rho = DT.apl_effect_resamples(recs)
    r = float(np.nanmean(sh_rho))
    fc = floor_ceiling({g: t[:, [row], :] for g, t in tab_g.items()})
    return dict(floor_dt=float(np.mean(boot_rmse)), ceiling_dt=float(np.sqrt(2 * r / (1 + r))),
                floor_vec=float(fc['floor'][0]), ceiling_vec=float(fc['ceiling'][0]))


def part_b(n_boot: int = N_BOOT) -> None:
    """Part B: effect_by_window.csv, pairs_by_window.csv, common_error_by_window.csv, variants_by_window.csv,
    variants_by_window_summary.csv, rates_by_window.csv, rates_by_window_odors.csv, models_by_window.csv."""
    write_provenance('B', dict(n_boot=n_boot, windows=windows(), subsets={k: {g: [rec_id(x) for x in v] for g, v in s.items()}
                                                                         for k, s in subsets().items()}))
    t0 = time.time()
    tabs, keys, row_keys, rates = recording_tables()
    main_target = _pairs_array(H.yang_apl_effect())
    pops = population_effects()
    # the common error (E72's common_error_pairs.csv) is the eight wirings' pattern error against the target
    ce = pd.read_csv(H.results_dir('E72_wiring_population') / 'common_error_pairs.csv')
    ce = _pairs_array(ce.set_index(['odor_a', 'odor_b'])['common'])
    err = pops['eight']['cal'] - main_target
    common_check = float(np.abs((err - err.mean()) - ce).max())
    print(f'B: the eight wirings\' calibrated pattern error against the target equals E72\'s common error to {common_check:.2g}',
          flush=True)
    fn = _stat_fn(row_keys, main_target, pops)
    wl = {w[0]: (w[1], w[2]) for w in windows()}
    eff_rows, pop_rows = [], []
    fc_check = dict(common_error_check=common_check)
    for sname, sub in subsets().items():
        sel = {g: [keys[g].index(k) for k in sub[g]] for g in GENOS}
        if any(len(v) == 0 for v in sel.values()):
            print(f'B: subset {sname} is empty; skipped', flush=True)
            continue
        tg = {g: tabs[g][sel[g]] for g in GENOS}
        kg = {g: [keys[g][i] for i in sel[g]] for g in GENOS}
        res = {u: _boot(tg, kg, fn, n_boot, u) for u in ('recording', 'fly')}
        for stat, (obs, lo, hi) in res['recording'].items():
            _, flo, fhi = res['fly'][stat]
            for i, (u, e, w) in enumerate(row_keys):
                r = dict(subset=sname, units=u, estimator=e, window=w, t0=wl[w][0], t1=wl[w][1], value=float(obs[i]),
                         lo_rec=float(lo[i]), hi_rec=float(hi[i]), lo_fly=float(flo[i]), hi_fly=float(fhi[i]),
                         n_ctrl=len(sel['TNT3c']), n_tnt=len(sel['TNT']))
                if stat.startswith('pop:'):
                    _, p, ro, s = stat.split(':')
                    pop_rows.append(dict(r, population=p, readout=ro, statistic=s))
                else:
                    eff_rows.append(dict(r, statistic=stat))
        if sname in ('all', 'six_typical'):
            fc = floor_ceiling(tg)
            if sname == 'all':
                chk = floor_ceiling_check(tg, row_keys.index(('z', 'measured', 'target')))
                fc_check.update(chk, registry_floor=H.data_target('apl_effect_rmse_floor'),
                                registry_ceiling=H.data_target('apl_effect_rho_ceiling'))
                print(f'B: floor / ceiling check {chk}', flush=True)
            for i, (u, e, w) in enumerate(row_keys):
                base = dict(subset=sname, units=u, estimator=e, window=w, t0=wl[w][0], t1=wl[w][1], n_ctrl=len(sel['TNT3c']),
                            n_tnt=len(sel['TNT']))
                eff_rows.append(dict(base, statistic='loss_floor', value=float(fc['floor'][i]), lo_rec=float(fc['floor_lo'][i]),
                                     hi_rec=float(fc['floor_hi'][i])))
                eff_rows.append(dict(base, statistic='rho_ceiling', value=float(fc['ceiling'][i])))
        print(f'B: subset {sname} done, {time.time() - t0:.0f} s', flush=True)
    ef = pd.DataFrame(eff_rows)
    ef.to_csv(out_dir() / 'effect_by_window.csv', index=False)
    (out_dir() / 'part_b_checks.json').write_text(json.dumps(fc_check, indent=1, default=float) + '\n')
    pd.DataFrame(pop_rows).to_csv(out_dir() / 'common_error_by_window.csv', index=False)
    # per pair, all and the six typical
    prs = []
    for sname in ('all', 'six_typical'):
        sub = subsets()[sname]
        sel = {g: [keys[g].index(k) for k in sub[g]] for g in GENOS}
        C, T = tabs['TNT3c'][sel['TNT3c']].mean(0), tabs['TNT'][sel['TNT']].mean(0)
        for i, (u, e, w) in enumerate(row_keys):
            for j, (a, b) in enumerate(_PAIR_IDX):
                prs.append(dict(subset=sname, units=u, estimator=e, window=w, odor_a=a, odor_b=b, ctrl=C[i, j], tnt=T[i, j],
                                effect=C[i, j] - T[i, j]))
    pd.DataFrame(prs).to_csv(out_dir() / 'pairs_by_window.csv', index=False)
    variants_by_window(tabs, keys, row_keys)
    rates_by_window(rates, n_boot)
    models_by_window()
    print(f'part B done in {time.time() - t0:.0f} s', flush=True)


VARIANT_WINDOWS = ('pulse', '2-4', '4-8.1', 'target', '0-8.1', '8.1-10.1')


def variants_by_window(tabs, keys, row_keys) -> None:
    """Every archived variant's raw-count APL effect (apl_effect_scores.model_pairs over results/E*/) against the flies'
    effect in VARIANT_WINDOWS (all recordings, Yang's z, both estimators): loss, mean error, pattern part, ρ; the rank
    correlation over variants of each window's loss with the target window's and with the present scores
    (results/apl_effect/by_variant.csv)."""
    import apl_effect_scores as AES
    effs, names = [], []
    folders = sorted(p for p in (H.shared_dir() / 'results').glob('E*_*') if p.is_dir() and re.match(r'E\d+_', p.name))
    for folder in folders:
        try:
            pairs = AES.model_pairs(folder)
        except Exception as e:
            print(f'variants: {folder.name}: {e!r}'[:200], flush=True)
            continue
        for tag, (on, off) in pairs.items():
            e = _pairs_array(H.apl_effect(on, off))
            if np.isfinite(e).sum() < 100:
                continue
            effs.append(e)
            names.append((folder.name, str(tag)))
    V = np.stack(effs)
    rows = {k: dict(folder=f, tag=t) for k, (f, t) in enumerate(names)}
    for e in ESTIMATORS:
        for w in VARIANT_WINDOWS:
            i = row_keys.index(('z', e, w))
            d = tabs['TNT3c'][:, i].mean(0) - tabs['TNT'][:, i].mean(0)
            err = V - d[None]
            ok = np.isfinite(err)
            me = np.nanmean(err, 1)
            loss = np.sqrt(np.nanmean(err ** 2, 1))
            pat = np.sqrt(np.nanmean((err - me[:, None]) ** 2, 1))
            rho = np.array([H.spearman(pd.Series(V[k][ok[k]]), pd.Series(d[ok[k]])) for k in range(len(V))])
            for k in range(len(V)):
                rows[k].update({f'{e}_{w}_loss': loss[k], f'{e}_{w}_mean_error': me[k], f'{e}_{w}_pattern': pat[k],
                                f'{e}_{w}_rho': rho[k]})
    df = pd.DataFrame(list(rows.values()))
    bv = H.shared_dir() / 'results' / 'apl_effect' / 'by_variant.csv'
    if bv.exists():
        b = pd.read_csv(bv)[['folder', 'tag', 'spike_counts_cosine__apl_effect_rmse']].rename(
            columns={'spike_counts_cosine__apl_effect_rmse': 'present_loss'})
        b['tag'] = b['tag'].astype(str)
        df = df.merge(b.drop_duplicates(['folder', 'tag']), on=['folder', 'tag'], how='left')
    df.to_csv(out_dir() / 'variants_by_window.csv', index=False)
    summ = []
    for e in ESTIMATORS:
        for w in VARIANT_WINDOWS:
            col = f'{e}_{w}_loss'
            summ.append(dict(estimator=e, window=w, n_variants=int(df[col].notna().sum()),
                             rho_with_target_measured=H.spearman(df[col], df['measured_target_loss']),
                             rho_with_present=H.spearman(df[col], df['present_loss']) if 'present_loss' in df else np.nan,
                             max_abs_diff_target_measured_vs_present=float(np.nanmax(np.abs(df['measured_target_loss']
                                                                                            - df['present_loss'])))
                             if 'present_loss' in df else np.nan,
                             median_loss=float(df[col].median()), best_loss=float(df[col].min()),
                             best=f"{df.loc[df[col].idxmin(), 'folder']}:{df.loc[df[col].idxmin(), 'tag']}"))
    pd.DataFrame(summ).to_csv(out_dir() / 'variants_by_window_summary.csv', index=False)


def rates_by_window(rates: pd.DataFrame, n_boot: int) -> None:
    """rates_by_window.csv (per subset, criterion and window: the mean response rate per genotype, the APL-off / APL-on
    ratio with its 90 % interval over resampled recordings, the odors-per-KC shares) and rates_by_window_odors.csv (the
    per-odor response rates, all recordings)."""
    rng = np.random.default_rng(0)
    out = []
    for sname, sub in subsets().items():
        if sname.startswith('ctrl_') and 'recorded' not in sname:
            continue
        for crit, g1 in rates.groupby('criterion'):
            for w, g2 in g1.groupby('window', sort=False):
                v = {}
                for g in GENOS:
                    want = set(sub[g])
                    x = g2[g2.genotype == g]
                    x = x[[(r.date, int(r.fly_num), r.hemibrain) in want for r in x.itertuples()]]
                    v[g] = x
                if any(len(x) == 0 for x in v.values()):
                    continue
                rc, rt = v['TNT3c'].rate.to_numpy(), v['TNT'].rate.to_numpy()
                bs = [rt[rng.integers(0, len(rt), len(rt))].mean() / rc[rng.integers(0, len(rc), len(rc))].mean()
                      for _ in range(n_boot)]
                row = dict(subset=sname, criterion=crit, window=w, t0=float(g2.t0.iloc[0]), t1=float(g2.t1.iloc[0]),
                           rate_ctrl=float(rc.mean()), rate_tnt=float(rt.mean()), ratio=float(rt.mean() / rc.mean()),
                           ratio_lo=float(np.percentile(bs, 5)), ratio_hi=float(np.percentile(bs, 95)),
                           threshold_ctrl=float(v['TNT3c'].threshold.mean()), threshold_tnt=float(v['TNT'].threshold.mean()),
                           blank_fpr=float(pd.concat(v.values()).blank_fpr_target.mean()))
                if not crit.startswith('single'):
                    for g in GENOS:
                        for lab, _, _ in OPK_BINS:
                            row[f'opk_{lab}_{g}'] = float(v[g][f'opk_{lab}'].mean())
                out.append(row)
    pd.DataFrame(out).to_csv(out_dir() / 'rates_by_window.csv', index=False)
    po = []
    for (crit, w, g), x in rates.groupby(['criterion', 'window', 'genotype'], sort=False):
        for o in ODORS:
            po.append(dict(criterion=crit, window=w, genotype=g, odor=o, rate=float(x[f'rate_{o}'].mean())))
    pd.DataFrame(po).to_csv(out_dir() / 'rates_by_window_odors.csv', index=False)


def models_by_window() -> None:
    """models_by_window.csv: E82's runs_windows.csv (the models' per-pair APL effect from the spikes in each window of
    the simulation at the flies' pulse), the columns the comparison needs."""
    p = H.results_dir('E82_flies_protocol') / 'runs_windows.csv'
    if not p.exists():
        return
    d = pd.read_csv(p)
    keep = [c for c in ('point_id', 'window', 't0', 't1', 'share_spikes_on', 'share_spikes_off', 'raw_loss', 'raw_mean',
                        'raw_pattern', 'raw_rho', 'mean_dist_on', 'mean_dist_off', 'n_pairs_defined') if c in d.columns]
    d[keep].to_csv(out_dir() / 'models_by_window.csv', index=False)


# ---------------------------------------------------------------------------
# part C: the shape of the responses

SELECTIONS = ('resp', 'below', 'matched', 'all')
ODOR_GROUPS = {'all': ODORS, 'ester': list(ESTER), 'alcohol': list(ALCOHOL),
               'other': [o for o in ODORS if o not in ESTER and o not in ALCOHOL]}
KC_CLASSES = ('all',) + tuple(lab for lab, _, _ in OPK_BINS)


def recording_traces(c: dict, neucoeff: float = NEUCOEFF) -> Dict[str, np.ndarray]:
    """Sums and counts of one recording's held-out traces (baseline-noise units): S (selection, odor, held-out repeat,
    KC class, samples), N (the same without samples: traces summed) and D (selection, odor, KC class: distinct KC x odor
    entries, any repeat). Responders are entries whose mean over the other two repeats in the target's window is above
    the bands' criterion; 'below' the rest; 'matched' those with that mean in MATCHED_AMP; 'all' every entry. The KC
    class is its number of odors above the criterion on the trial means of all three repeats."""
    thr = float(H.data_target('noise_threshold'))
    m = c['meta']
    X = c['dfc'] if neucoeff == NEUCOEFF else (c['dfc'].astype(np.float64) - (neucoeff - NEUCOEFF) * c['dfneu'])
    sb = c[f'sd_base_{neucoeff:g}']
    Wt = window_values(c, [TARGET], 'noise', neucoeff)[..., 0]                # (trials, KCs)
    idx = trial_index(c, ODORS)                                               # (3, 16)
    tm = Wt[idx].mean(axis=0)                                                 # (16, KCs)
    nodor = (tm > thr).sum(axis=0)
    cls = np.zeros(len(nodor), dtype=int)
    for j, (_, lo, hi) in enumerate(OPK_BINS):
        cls[(nodor >= lo) & (nodor <= hi)] = j + 1
    nsel, nod, nrep, ncl, nt = len(SELECTIONS), len(ODORS), idx.shape[0], len(KC_CLASSES), X.shape[-1]
    S = np.zeros((nsel, nod, nrep, ncl, nt))
    N = np.zeros((nsel, nod, nrep, ncl))
    D = np.zeros((nsel, nod, ncl))
    for j in range(nod):
        anyk = np.zeros((nsel, X.shape[1]), dtype=bool)
        for h in range(nrep):
            others = [r for r in range(nrep) if r != h]
            score = Wt[idx[others, j]].mean(axis=0)
            masks = (score > thr, score <= thr, (score >= MATCHED_AMP[0]) & (score < MATCHED_AMP[1]),
                     np.ones_like(score, dtype=bool))
            tr = X[idx[h, j]].astype(np.float64) / sb[:, None]
            for s, mk in enumerate(masks):
                anyk[s] |= mk
                for k in range(ncl):
                    mm = mk if k == 0 else (mk & (cls == k))
                    if mm.any():
                        S[s, j, h, k] = tr[mm].sum(axis=0)
                        N[s, j, h, k] = mm.sum()
        for s in range(nsel):
            for k in range(ncl):
                D[s, j, k] = (anyk[s] if k == 0 else (anyk[s] & (cls == k))).sum()
    return dict(S=S, N=N, D=D, t=times(c))


def _marginal(R: Dict[str, np.ndarray], sel: str, group: str = 'all', kc_class: str = 'all',
              repeat: Optional[int] = None) -> Tuple[np.ndarray, int, int]:
    """(mean trace, traces averaged, distinct entries) of one recording's sums."""
    s = SELECTIONS.index(sel)
    js = [ODORS.index(o) for o in ODOR_GROUPS[group]]
    k = KC_CLASSES.index(kc_class)
    hs = list(range(R['S'].shape[2])) if repeat is None else [repeat]
    S = R['S'][s][np.ix_(js, hs)][:, :, k].sum(axis=(0, 1))
    N = R['N'][s][np.ix_(js, hs)][:, :, k].sum()
    D = R['D'][s, js, k].sum() if repeat is None else N
    return (S / N if N > 0 else np.full(S.shape, np.nan)), int(N), int(D)


def trace_kinetics(t: np.ndarray, y: np.ndarray, volume_rate: float) -> Dict[str, float]:
    """The time of half rise and of the peak (0 to 20 s), the value at 2.0, 8.1, 20 and 40 s over the peak (linear
    interpolation), the mean over 4-8.1 s over the mean over 0-2 s (windows as volumes), the peak value and a decay
    time constant fitted from 8.1 s to the excerpt's end (a * exp(-(t - 8.1) / tau))."""
    from scipy.optimize import curve_fit
    out = {}
    pk = (t >= 0) & (t <= 20)
    i = int(np.flatnonzero(pk)[np.nanargmax(y[pk])])
    peak = float(y[i])
    out.update(peak=peak, peak_t=float(t[i]))
    pre = np.flatnonzero((t >= 0) & (t <= t[i]))
    above = pre[y[pre] >= 0.5 * peak]
    j = int(above[0]) if len(above) else i
    if j > 0 and y[j - 1] < 0.5 * peak:
        out['half_rise_t'] = float(t[j - 1] + (0.5 * peak - y[j - 1]) / (y[j] - y[j - 1]) * (t[j] - t[j - 1]))
    else:
        out['half_rise_t'] = float(t[j])
    for lab, tt in (('2', 2.0), ('8.1', 8.1), ('20', 20.0), ('40', 40.0)):
        out[f'v{lab}_over_peak'] = float(np.interp(min(tt, t[-1]), t, y) / peak)
    k0 = int(np.flatnonzero(t >= 0)[0])
    m = {}
    for (a, b) in (EARLY, LATE):
        lo, hi = H.window_volumes((a, b), volume_rate)
        m[(a, b)] = float(np.mean(y[k0 + lo:k0 + hi]))
    out['late_over_early'] = m[LATE] / m[EARLY]
    out['mean_0_2'], out['mean_4_8.1'] = m[EARLY], m[LATE]
    tail = t >= 8.05                                                          # from volume 24 (8.098 s), the window's end
    try:
        p, _ = curve_fit(lambda x, a, tau: a * np.exp(-(x - 8.1) / tau), t[tail], y[tail], p0=(float(y[tail][0]), 15.0),
                         bounds=([0, 0.5], [np.inf, 500]), maxfev=20000)
        out['decay_tau_from_8.1'] = float(p[1])
    except Exception:
        out['decay_tau_from_8.1'] = np.nan
    return out


def _traces_all() -> Dict[Tuple[str, Tuple[str, int, str]], Dict[str, np.ndarray]]:
    """Every recording's `recording_traces`, cached in outputs/ (recording_traces.npz)."""
    path = work_dir() / 'recording_traces.npz'
    out = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['meta']))
            for i, (g, k) in enumerate(meta['keys']):
                out[(g, tuple(k))] = {q: z[f'{q}_{i}'] for q in ('S', 'N', 'D', 't')}
        return out
    arrs, keys = {}, []
    for i, (g, rec) in enumerate(YRN.RECORDINGS):
        c = load_cache(g, rec)
        R = recording_traces(c)
        key = tuple(c['meta']['key'])
        out[(g, key)] = R
        keys.append([g, list(key)])
        for q, v in R.items():
            arrs[f'{q}_{i}'] = v
        print(f'C traces {g} {rec}', flush=True)
        del c
    np.savez(path, meta=np.array(json.dumps(dict(keys=keys))), **arrs)
    os.chmod(path, 0o600)
    return out


GENO_SUBSETS = {'control flies': ('TNT3c', None), 'APL output blocked (TNT), all eight': ('TNT', None),
                'APL output blocked (TNT), the six typical': ('TNT', 'six'),
                'APL output blocked (TNT), the two strongest': ('TNT', 'two'),
                'APL output blocked (TNT), without 2024-07-17 fly 3 L': ('TNT', 'wo_unlabeled')}


def _in_subset(g: str, key, which: Optional[str]) -> bool:
    if which is None:
        return True
    if which == 'six':
        return key not in STRONG_TNT
    if which == 'two':
        return key in STRONG_TNT
    if which == 'wo_unlabeled':
        return key != UNLABELED_TNT
    raise ValueError(which)


def subset_trace(TR: dict, subset: str, sel: str, group: str = 'all', kc_class: str = 'all', repeat: Optional[int] = None,
                 keys: Optional[list] = None) -> Tuple[np.ndarray, int, int, int]:
    """(mean over recordings of the recordings' mean traces, traces, distinct entries, recordings) of a genotype subset;
    `keys` restricts it to these recordings (a resample, with repeats)."""
    g, which = GENO_SUBSETS[subset]
    ks = [k for (gg, k) in TR if gg == g and _in_subset(g, k, which)] if keys is None else keys
    ys, n, d = [], 0, 0
    for k in ks:
        y, nn, dd = _marginal(TR[(g, k)], sel, group, kc_class, repeat)
        if nn > 0:
            ys.append(y)
            n += nn
            d += dd
    return (np.mean(ys, axis=0) if ys else np.nan), n, d, len(ys)


def part_c() -> None:
    """Part C: traces.csv (every mean trace, per recording and per genotype subset, with the number of traces and of
    distinct KC x odor entries averaged), kinetics_by_recording.csv, kinetics_by_group.csv, kinetics_relations.csv."""
    write_provenance('C', dict(matched_amplitude=list(MATCHED_AMP), selections=SELECTIONS,
                               odor_groups=ODOR_GROUPS, kc_classes=KC_CLASSES))
    TR = _traces_all()
    rec = pd.read_csv(out_dir() / 'records_summary.csv', dtype={'date': str, 'hemibrain': str})
    rk = {(r.date, int(r.fly_num), r.hemibrain): r for r in rec.itertuples()}
    t = next(iter(TR.values()))['t']
    v = float(rec.volume_rate.iloc[0])
    long, kin = [], []
    # per recording
    for (g, key), R in TR.items():
        r = rk[key]
        for sel in SELECTIONS:
            for group in ODOR_GROUPS:
                for kc_class in KC_CLASSES:
                    for repeat in (None, 0, 1, 2):
                        if repeat is not None and (group != 'all' or kc_class != 'all'):
                            continue
                        if group != 'all' and kc_class != 'all':
                            continue
                        y, n, d = _marginal(R, sel, group, kc_class, repeat)
                        if n == 0:
                            continue
                        meta = dict(level='recording', genotype=g, recording=rec_id(key), subset='', selection=sel,
                                    odor_group=group, kc_class=kc_class, repeat='all' if repeat is None else repeat + 1,
                                    n_traces=n, n_entries=d, n_recordings=1)
                        if sel != 'below':
                            try:
                                kin.append(dict(meta, date=key[0], group_as_recorded=r.group_as_recorded,
                                                strong=key in STRONG_TNT, unlabeled=key == UNLABELED_TNT,
                                                median_F=r.median_F, median_Fneu=r.median_Fneu, median_sd_base=r.median_sd_base,
                                                **trace_kinetics(t, y, v)))
                            except Exception as e:
                                print(f'kinetics {meta}: {e!r}', flush=True)
                        long.append(pd.DataFrame(dict(meta, t=t, value=y)))
    # per genotype subset (mean over recordings of the recordings' traces)
    grp = []
    for subset in GENO_SUBSETS:
        for sel in SELECTIONS:
            for group in ODOR_GROUPS:
                for kc_class in KC_CLASSES:
                    for repeat in (None, 0, 1, 2):
                        if repeat is not None and (group != 'all' or kc_class != 'all'):
                            continue
                        y, n, d, nr = subset_trace(TR, subset, sel, group, kc_class, repeat)
                        if nr == 0:
                            continue
                        meta = dict(level='subset', genotype=GENO_SUBSETS[subset][0], recording='', subset=subset,
                                    selection=sel, odor_group=group, kc_class=kc_class,
                                    repeat='all' if repeat is None else repeat + 1, n_traces=n, n_entries=d, n_recordings=nr)
                        long.append(pd.DataFrame(dict(meta, t=t, value=y)))
                        if sel != 'below':
                            try:
                                grp.append(dict(meta, **trace_kinetics(t, y, v)))
                            except Exception as e:
                                print(f'kinetics {meta}: {e!r}', flush=True)
    pd.concat(long, ignore_index=True).to_csv(out_dir() / 'traces.csv', index=False)
    kr = pd.DataFrame(kin)
    kr.to_csv(out_dir() / 'kinetics_by_recording.csv', index=False)
    pd.DataFrame(grp).to_csv(out_dir() / 'kinetics_by_group.csv', index=False)
    kinetics_relations(kr)


KIN_MEASURES = ('peak_t', 'half_rise_t', 'v2_over_peak', 'v8.1_over_peak', 'v20_over_peak', 'v40_over_peak', 'late_over_early',
                'decay_tau_from_8.1', 'peak')


def kinetics_relations(kr: pd.DataFrame) -> None:
    """kinetics_relations.csv: for the responders' and the matched-amplitude traces (all odors, all repeats), each
    measure's range over the 17 other recordings and where each of the two strongest lies; the Spearman correlation of
    each measure with brightness (median F, Fneu) and baseline noise over the 19 and over the 17; the measure by
    repeat (mean over recordings of each genotype)."""
    rows = []
    for sel in ('resp', 'matched'):
        x = kr[(kr.selection == sel) & (kr.odor_group == 'all') & (kr.kc_class == 'all') & (kr.repeat.astype(str) == 'all')]
        others = x[~x.strong]
        for meas in KIN_MEASURES:
            lo, hi = float(others[meas].min()), float(others[meas].max())
            r = dict(selection=sel, measure=meas, others_min=lo, others_max=hi,
                     control_mean=float(x[x.genotype == 'TNT3c'][meas].mean()),
                     six_typical_mean=float(x[(x.genotype == 'TNT') & ~x.strong][meas].mean()))
            for _, s in x[x.strong].iterrows():
                tag = 'strong_' + s.recording.replace(' ', '_')
                r[tag] = float(s[meas])
                r[tag + '_outside'] = bool(s[meas] < lo or s[meas] > hi)
            for cov in ('median_F', 'median_Fneu', 'median_sd_base'):
                r[f'rho_{cov}_19'] = H.spearman(x[meas], x[cov])
                r[f'rho_{cov}_17'] = H.spearman(others[meas], others[cov])
            rows.append(r)
        xr = kr[(kr.selection == sel) & (kr.odor_group == 'all') & (kr.kc_class == 'all') & (kr.repeat.astype(str) != 'all')]
        for (g, rep), q in xr.groupby(['genotype', 'repeat']):
            rows.append(dict(selection=sel, measure='by_repeat', genotype=g, repeat=rep,
                             **{f'{meas}_mean': float(q[meas].mean()) for meas in KIN_MEASURES}))
        for (gr), q in x.groupby('group_as_recorded'):
            rows.append(dict(selection=sel, measure='by_label_group', group_as_recorded=gr, n=len(q),
                             **{f'{meas}_mean': float(q[meas].mean()) for meas in KIN_MEASURES}))
        for (dt), q in x.groupby('date'):
            rows.append(dict(selection=sel, measure='by_date', date=dt, n=len(q),
                             **{f'{meas}_mean': float(q[meas].mean()) for meas in KIN_MEASURES}))
    pd.DataFrame(rows).to_csv(out_dir() / 'kinetics_relations.csv', index=False)


# ---------------------------------------------------------------------------
# part D: the kernel and the models

FIT_SUBSETS = ('control flies', 'APL output blocked (TNT), all eight', 'APL output blocked (TNT), the six typical',
               'APL output blocked (TNT), the two strongest')


def _round_kernel(rise: float, decay: float) -> Tuple[float, float]:
    """A fitted kernel at the precision it is read at (rise to 0.01 s, decay to 0.1 s)."""
    return float(np.round(rise, 2)), float(np.round(decay, 1))


def kernel_fits(TR: dict, t: np.ndarray) -> pd.DataFrame:
    """kernel_fits.csv: per subset and selection (responders, matched amplitude) and drive (onset, steady), the free
    fit; and the control flies' kernel tested on each subset with APL output blocked (TNT) (amplitude only); per
    recording, the responders' free fits."""
    rows = []
    fits = {}
    for sel in ('resp', 'matched'):
        for subset in FIT_SUBSETS:
            y, n, d, nr = subset_trace(TR, subset, sel)
            for drive in ('onset', 'steady'):
                f = H.fit_indicator_kernel(t, y, drive)
                fits[(sel, subset, drive)] = f
                rows.append(dict(level='subset', subset=subset, recording='', selection=sel, fit='free', kernel_from='',
                                 n_entries=d, n_recordings=nr, **f))
        for subset in FIT_SUBSETS[1:]:
            y, n, d, nr = subset_trace(TR, subset, sel)
            for drive in ('onset', 'steady'):
                k = fits[(sel, 'control flies', drive)]
                f = H.fit_indicator_kernel(t, y, drive, fixed=(k['rise'], k['decay']))
                rows.append(dict(level='subset', subset=subset, recording='', selection=sel, fit='kernel fixed',
                                 kernel_from='control flies', n_entries=d, n_recordings=nr, **f,
                                 rms_own_fit=fits[(sel, subset, drive)]['rms']))
    for (g, key), R in TR.items():
        y, n, d = _marginal(R, 'resp')
        for drive in ('onset', 'steady'):
            f = H.fit_indicator_kernel(t, y, drive)
            rows.append(dict(level='recording', subset=g, recording=rec_id(key), selection='resp', fit='free', kernel_from='',
                             n_entries=d, n_recordings=1, **f))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / 'kernel_fits.csv', index=False)
    return df


def _hist(times_s: np.ndarray, dt: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    t_end = max(float(times_s.max()) + dt, 2.0) if len(times_s) else 2.0
    edges = np.arange(0.0, t_end + dt, dt)
    w, _ = np.histogram(times_s, bins=edges)
    return edges, w.astype(float)


def model_kernel_fits(TR: dict, t: np.ndarray) -> pd.DataFrame:
    """kernel_fits_models.csv: per saved E82 run, the kernel fitted to the control flies' responders' trace with the run's
    APL-on responders' spike-time histogram as the drive, then (amplitude only) the trace with APL output blocked (TNT)
    (all eight; the six typical) with its APL-off responders' histogram."""
    import E82_flies_protocol as E82
    yc = subset_trace(TR, 'control flies', 'resp')[0]
    yt = {s: subset_trace(TR, s, 'resp')[0] for s in FIT_SUBSETS[1:3]}
    rows = []
    for r in E82.saved_runs().itertuples():
        o, st = E82.load_run(Path(r.dir), r.point_id)
        hs = {}
        for cnd in ('on', 'off'):
            R = o[cnd][0].to_numpy(dtype=float) > 0.5
            s = st[cnd]
            isr = R[s.kc.to_numpy(), s.odor.to_numpy()]
            hs[cnd] = _hist(s.time_s.to_numpy()[isr])
        f = H.fit_indicator_kernel(t, yc, 'histogram', edges=hs['on'][0], weights=hs['on'][1])
        row = dict(source=r.source, point_id=r.point_id, model=r.model, pulse=r.pulse, held=r.held, identity=r.identity,
                   arm=r.arm, rise=f['rise'], decay=f['decay'], rms_ctrl=f['rms'], rise_at_bound=f['rise_at_bound'],
                   decay_at_bound=f['decay_at_bound'],
                   on_share_after_0p5=float(hs['on'][1][hs['on'][0][:-1] >= 0.5].sum() / max(hs['on'][1].sum(), 1)),
                   off_share_after_0p5=float(hs['off'][1][hs['off'][0][:-1] >= 0.5].sum() / max(hs['off'][1].sum(), 1)))
        for s, y in yt.items():
            g = H.fit_indicator_kernel(t, y, 'histogram', edges=hs['off'][0], weights=hs['off'][1],
                                       fixed=(f['rise'], f['decay']))
            row[f'rms_{"tnt8" if "eight" in s else "tnt6"}'] = g['rms']
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / 'kernel_fits_models.csv', index=False)
    return df


def _norm(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        return v
    if v is None:
        return ''
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return '' if np.isnan(f) else f


def _same(a, b) -> bool:
    a, b = _norm(a), _norm(b)
    if isinstance(a, float) and isinstance(b, float):
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return str(a) == str(b)
    if a == '' and b == '':
        return True
    return a == b or str(a) == str(b)


def e82_reading(E82, r, o, st, k, primary: bool) -> Tuple[dict, pd.DataFrame]:
    """One run's protocol_readings.csv row and protocol_pairs.csv rows under kernel `k`, made as
    E82_flies_protocol.protocol_readings makes them."""
    resp = {c: o[c][0] for c in o}
    band = E82.band_row(o)
    x = E82.weighted(o, st, k)
    row, pr = E82.score_reading(resp, x)
    meta = dict(E82._meta(r), kernel=E82.klabel(k), rise=np.nan if k is None else k[0], decay=np.nan if k is None else k[1],
                primary=primary)
    for c in ('on', 'off'):
        a = x[c].to_numpy()
        meta[f'x_sum_{c}'], meta[f'x_max_{c}'] = float(a.sum()), float(a.max())
    pr.insert(0, 'kernel', E82.klabel(k))
    pr.insert(0, 'point_id', r.point_id)
    pr.insert(0, 'source', r.source)
    return dict(meta, **band, **row), pr


def predicted_window_means(o, st, k) -> Dict[Tuple[str, str, float, float], float]:
    """{(condition, kind, t0, t1): mean over the kind's KC x odor entries of the kernel-convolved signal over the window}
    for kinds 'resp' (the responses), 'below' (the other entries) and 'all', the windows TRACE_WINDOWS (as
    E82_flies_protocol.predicted_traces computes them for 'resp' and 'all')."""
    out = {}
    for c in ('on', 'off'):
        R = o[c][0].to_numpy(dtype=float) > 0.5
        s = st[c]
        tt = s.time_s.to_numpy()
        isr = R[s.kc.to_numpy(), s.odor.to_numpy()]
        for kind, sel, n in (('resp', isr, int(R.sum())), ('below', ~isr, int((~R).sum())), ('all', np.ones_like(isr), R.size)):
            tu, cnt = np.unique(tt[sel], return_counts=True)
            for a, b in TRACE_WINDOWS:
                area = (H.indicator_kernel_integral(b - tu, *k) - H.indicator_kernel_integral(a - tu, *k)) @ cnt
                out[(c, kind, a, b)] = float(area / (b - a) / max(n, 1))
    return out


def flies_ratio(TR: dict, n_boot: int = N_BOOT) -> pd.DataFrame:
    """The flies' signal with APL output blocked (TNT) relative to the control flies', 4-8.1 s over 0-2 s, per selection
    (responders, below, all entries) and TNT subset, with 90 % intervals over resampled recordings; also 2-4 s over
    0-2 s."""
    v = None
    rows = []
    ctrl_keys = [k for (g, k) in TR if g == 'TNT3c']
    rng = np.random.default_rng(0)
    for sel in ('resp', 'below', 'all'):
        def wm(keys, g):
            out = []
            for k in keys:
                y, n, d = _marginal(TR[(g, k)], sel)
                tt = TR[(g, k)]['t']
                k0 = int(np.flatnonzero(tt >= -1e-12)[0])
                rate = 1.0 / (tt[k0 + 1] - tt[k0])
                out.append([np.mean(y[k0 + a:k0 + b]) for a, b in (H.window_volumes(w, rate) for w in TRACE_WINDOWS)])
            return np.array(out)
        C = wm(ctrl_keys, 'TNT3c')
        for subset in ('APL output blocked (TNT), all eight', 'APL output blocked (TNT), the six typical',
                       'APL output blocked (TNT), the two strongest'):
            tk = [k for (g, k) in TR if g == 'TNT' and _in_subset('TNT', k, GENO_SUBSETS[subset][1])]
            T = wm(tk, 'TNT')

            def ratio(c, t_, w1, w0):
                return (t_[:, w1].mean() / t_[:, w0].mean()) / (c[:, w1].mean() / c[:, w0].mean())
            for lab, w1 in (('4-8.1 over 0-2', 2), ('2-4 over 0-2', 1)):
                obs = ratio(C, T, w1, 0)
                bs = [ratio(C[rng.integers(0, len(C), len(C))], T[rng.integers(0, len(T), len(T))], w1, 0) for _ in range(n_boot)]
                rows.append(dict(kind='flies', subset=subset, selection=sel, ratio=lab, value=float(obs),
                                 lo=float(np.percentile(bs, 5)), hi=float(np.percentile(bs, 95)),
                                 ctrl_late_over_early=float(C[:, w1].mean() / C[:, 0].mean()),
                                 tnt_late_over_early=float(T[:, w1].mean() / T[:, 0].mean())))
    return pd.DataFrame(rows)


def kernel_recovery(t: np.ndarray, fit_rows: pd.DataFrame, TR: dict, n_draws: int = 200) -> pd.DataFrame:
    """kernel_recovery.csv: synthetic responders' mean traces at the flies' sampling, with firing uniform over the first
    D s (D 0 = an onset burst, 0.5, 1, 2 = the steady drive) through the control flies' onset-fit kernel, at the
    amplitude of the control flies' trace and white noise of its pre-onset SD (per recording, and for the mean over the
    11 recordings), each fitted under every D; how often each D is chosen (lowest rms) and the recovered rise / decay."""
    rng = np.random.default_rng(0)
    f = fit_rows[(fit_rows.level == 'subset') & (fit_rows.subset == 'control flies') & (fit_rows.selection == 'resp')
                 & (fit_rows.fit == 'free') & (fit_rows.drive == 'onset')].iloc[0]
    pre_sd = []
    for (g, k), R in TR.items():
        if g == 'TNT3c':
            y, _, _ = _marginal(R, 'resp')
            pre_sd.append(float(np.std(y[t < 0])))
    noise_levels = {'one recording': float(np.median(pre_sd)), 'mean of the 11': float(np.median(pre_sd) / np.sqrt(len(pre_sd)))}
    Ds = (0.0, 0.5, 1.0, 2.0)

    def drive_kw(D):
        return dict(drive='onset') if D == 0 else dict(drive='steady', pulse=D)
    rows = []
    for nl, sd in noise_levels.items():
        for D in Ds:
            clean = f['amplitude'] * H.kernel_signal(t, f['rise'], f['decay'], **drive_kw(D))
            chosen = []
            rec = []
            for _ in range(n_draws):
                y = clean + sd * rng.standard_normal(len(t))
                fits = {Dc: H.fit_indicator_kernel(t, y, **drive_kw(Dc)) for Dc in Ds}
                best = min(fits, key=lambda q: fits[q]['rms'])
                chosen.append(best)
                rec.append((fits[D]['rise'], fits[D]['decay']))
            rec = np.array(rec)
            row = dict(noise=nl, noise_sd=sd, true_D=D, true_rise=f['rise'], true_decay=f['decay'], n_draws=n_draws,
                       rise_median=float(np.median(rec[:, 0])), rise_p5=float(np.percentile(rec[:, 0], 5)),
                       rise_p95=float(np.percentile(rec[:, 0], 95)), decay_median=float(np.median(rec[:, 1])),
                       decay_p5=float(np.percentile(rec[:, 1], 5)), decay_p95=float(np.percentile(rec[:, 1], 95)))
            for Dc in Ds:
                row[f"chosen_D{format(Dc, 'g').replace('.', 'p')}"] = float(np.mean(np.array(chosen) == Dc))
            rows.append(row)
            print(f'recovery {nl} D={D}: chosen {[round(row[f"chosen_D{format(q, chr(103)).replace(chr(46), chr(112))}"], 2) for q in Ds]}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / 'kernel_recovery.csv', index=False)
    return df


def part_d(n_boot: int = N_BOOT) -> None:
    """Part D: kernel_fits.csv, kernel_fits_models.csv, protocol_readings_measured_kernel.csv (after the identity check
    against E82's protocol_readings.csv), protocol_populations_measured_kernel.csv, gate_measured_kernel.csv,
    trace_ratio_tests.csv, kernel_recovery.csv; part_d_checks.json."""
    import E82_flies_protocol as E82
    TR = _traces_all()
    t = next(iter(TR.values()))['t']
    t0 = time.time()
    kf = kernel_fits(TR, t)
    ctrl = kf[(kf.level == 'subset') & (kf.subset == 'control flies') & (kf.selection == 'resp') & (kf.fit == 'free')]
    kernels = {f'measured_{d}': _round_kernel(float(r.rise), float(r.decay)) for d, r in ctrl.set_index('drive').iterrows()}
    write_provenance('D', dict(kernels=kernels, e82_primary=list(E82.PRIMARY), e82_window=list(E82.WINDOW)))
    print(f'D: kernels {kernels}, {time.time() - t0:.0f} s', flush=True)
    model_kernel_fits(TR, t)
    # E82's readings: the identity check, then the measured kernels
    e82 = H.results_dir('E82_flies_protocol')
    rd = pd.read_csv(e82 / 'protocol_readings.csv', float_precision='round_trip')
    pp = pd.read_csv(e82 / 'protocol_pairs.csv', float_precision='round_trip')
    pr_e82 = pd.read_csv(e82 / 'predicted_ratios.csv', float_precision='round_trip')
    plab = E82.klabel(E82.PRIMARY)
    runs = E82.saved_runs()
    rows, pairs, ratios, checks = [], [], [], {}
    worst = 0.0
    for r in runs.itertuples():
        o, st = E82.load_run(Path(r.dir), r.point_id)
        row, prs = e82_reading(E82, r, o, st, E82.PRIMARY, True)
        ref = rd[(rd.source == r.source) & (rd.point_id == r.point_id) & (rd.kernel == plab)]
        if len(ref) != 1:
            raise RuntimeError(f'{r.source}:{r.point_id}: {len(ref)} rows of the primary reading in protocol_readings.csv')
        ref = ref.iloc[0]
        diff = [c for c in rd.columns if c in row and not _same(row[c], ref[c])]
        missing = [c for c in rd.columns if c not in row]
        refp = pp[(pp.source == r.source) & (pp.point_id == r.point_id) & (pp.kernel == plab)].reset_index(drop=True)
        mine = prs.reset_index(drop=True)
        pairs_same = (len(refp) == len(mine)) and all(
            np.array_equal(refp[c].to_numpy(dtype=float), mine[c].to_numpy(dtype=float), equal_nan=True)
            if pd.api.types.is_numeric_dtype(refp[c]) else (refp[c].astype(str) == mine[c].astype(str)).all()
            for c in refp.columns)
        checks[f'{r.source}:{r.point_id}'] = dict(columns_differing=diff, columns_missing=missing, pairs_identical=bool(pairs_same))
        if diff or missing or not pairs_same:
            num = {c: abs(float(row[c]) - float(ref[c])) for c in diff if isinstance(_norm(row[c]), float)
                   and isinstance(_norm(ref[c]), float)}
            worst = max([worst] + list(num.values()))
            print(f'D identity {r.source}:{r.point_id}: differs in {diff[:8]} {num}, missing {missing}, pairs {pairs_same}',
                  flush=True)
        # E82's predicted ratios at its two kernels against this script's window means (identity)
        for k in ((0.5, 3.0), (1.0, 5.0)):
            mine_m = predicted_window_means(o, st, k)
            ref_r = pr_e82[(pr_e82.source == r.source) & (pr_e82.point_id == r.point_id) & (pr_e82.kernel == E82.klabel(k))]
            for q in ref_r.itertuples():
                dd = max(abs(mine_m[('on', q.signal, q.t0, q.t1)] - q.mean_on), abs(mine_m[('off', q.signal, q.t0, q.t1)] - q.mean_off))
                checks[f'{r.source}:{r.point_id}']['predicted_ratio_max_abs'] = max(
                    dd, checks[f'{r.source}:{r.point_id}'].get('predicted_ratio_max_abs', 0.0))
        for name, k in kernels.items():
            row, prs = e82_reading(E82, r, o, st, k, False)
            row['measured_kernel'] = name
            rows.append(row)
            pairs.append(prs)
        for name, k in list(kernels.items()) + [('e82_primary', E82.PRIMARY), ('e82_r1_d5', (1.0, 5.0))]:
            mm = predicted_window_means(o, st, k)
            for kind in ('resp', 'below', 'all'):
                rat = {}
                for a, b in TRACE_WINDOWS:
                    on, off = mm[('on', kind, a, b)], mm[('off', kind, a, b)]
                    rat[(a, b)] = off / on if on > 0 else np.nan
                for lab, w1 in (('4-8.1 over 0-2', (4.0, 8.1)), ('2-4 over 0-2', (2.0, 4.0))):
                    ratios.append(dict(kind='model', source=r.source, point_id=r.point_id, model=r.model, pulse=r.pulse,
                                       held=r.held, identity=r.identity, arm=r.arm, kernel=name, rise=k[0], decay=k[1],
                                       selection=kind, ratio=lab, value=rat[w1] / rat[(0.0, 2.0)]))
        print(f'D readings {r.source}:{r.point_id}: {time.time() - t0:.0f} s', flush=True)
    n_bad = sum(1 for v in checks.values() if v['columns_differing'] or v['columns_missing'] or not v['pairs_identical'])
    worst_ratio = max(v.get('predicted_ratio_max_abs', 0.0) for v in checks.values())
    summ = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), al_analysis_commit=_git('rev-parse', 'HEAD'),
                e82_protocol_readings_sha256=_sha(e82 / 'protocol_readings.csv'), n_runs=len(runs),
                primary_identical_runs=len(runs) - n_bad, primary_worst_abs_diff=worst,
                predicted_ratio_worst_abs_diff=worst_ratio, kernels=kernels, runs=checks)
    (out_dir() / 'part_d_checks.json').write_text(json.dumps(summ, indent=1, default=str) + '\n')
    if n_bad:
        raise RuntimeError(f"part D: E82's primary reading is not reproduced exactly for {n_bad} runs (part_d_checks.json)")
    if worst_ratio > 1e-9:
        raise RuntimeError(f"part D: E82's predicted traces are not reproduced ({worst_ratio:.2g}; part_d_checks.json)")
    res = pd.DataFrame(rows)
    res.to_csv(out_dir() / 'protocol_readings_measured_kernel.csv', index=False)
    allp = pd.concat(pairs, ignore_index=True)
    allp.to_csv(work_dir() / 'protocol_pairs_measured_kernel.csv', index=False)
    measured_populations(E82, allp, kernels)
    measured_gates(E82, rd, pp, res, allp, kernels)
    fr = flies_ratio(TR, n_boot)
    rt = pd.concat([fr, pd.DataFrame(ratios)], ignore_index=True)
    rt = ratio_verdicts(rt)
    rt.to_csv(out_dir() / 'trace_ratio_tests.csv', index=False)
    kernel_recovery(t, kf, TR)
    print(f'part D done in {time.time() - t0:.0f} s', flush=True)


def ratio_verdicts(rt: pd.DataFrame) -> pd.DataFrame:
    """Each model row (4-8.1 over 0-2) gets `outside_eight` / `outside_six` (outside the flies' 90 % interval for the same
    selection with all eight, with the six typical) and `excluded` (both)."""
    fl = rt[rt.kind == 'flies'].set_index(['subset', 'selection', 'ratio'])
    out = rt.copy()
    for col, sub in (('outside_eight', 'APL output blocked (TNT), all eight'), ('outside_six', 'APL output blocked (TNT), the six typical')):
        vals = []
        for r in out.itertuples():
            if r.kind != 'model':
                vals.append(np.nan)
                continue
            f = fl.loc[(sub, r.selection, r.ratio)]
            vals.append(bool(r.value < f.lo or r.value > f.hi))
        out[col] = vals
    out['excluded'] = [bool(a) and bool(b) if k == 'model' else np.nan for k, a, b in
                       zip(out.kind, out.outside_eight, out.outside_six)]
    return out


def measured_populations(E82, allp: pd.DataFrame, kernels: Dict[str, tuple]) -> None:
    """protocol_populations_measured_kernel.csv: E82's populations of four wirings (runs.csv's `population`, at each
    pulse) under the measured kernels, scored as E82_flies_protocol.population_readings scores them."""
    rp = H.results_dir('E82_flies_protocol') / 'runs.csv'
    if not rp.exists():
        return
    m = pd.read_csv(rp).drop_duplicates('point_id', keep='last')
    if 'population' not in m.columns:
        return
    m = m[m.population.fillna('').astype(str).str.len() > 0]
    q0 = allp[allp.source == 'E82B']
    rows = []
    for (popn, pulse), g in m.groupby(['population', 'pulse']):
        pids = sorted(g.point_id)
        pp = q0[q0.point_id.isin(pids)]
        if len(pids) != 4 or set(pp.point_id) != set(pids):
            continue
        for name, k in kernels.items():
            dist = {}
            for ro in ('raw', 'cal', 'cal_noise'):
                q = pp[(pp.kernel == E82.klabel(k)) & (pp.readout == ro)]
                mean = q.groupby(['odor_a', 'odor_b'])[['on', 'off']].mean()
                dist[ro] = {c: mean[c] for c in ('on', 'off')}
            rows.append(dict(population=popn, pulse=pulse, members=';'.join(pids), n_wirings=len(pids), kernel=E82.klabel(k),
                             measured_kernel=name, **E82.score_distances(dist)))
    pd.DataFrame(rows).to_csv(out_dir() / 'protocol_populations_measured_kernel.csv', index=False)


def measured_gates(E82, rd: pd.DataFrame, pp: pd.DataFrame, res: pd.DataFrame, allp: pd.DataFrame,
                   kernels: Dict[str, tuple]) -> None:
    """gate_measured_kernel.csv: E82's protocol gate and E77's gate (E82_flies_protocol.gate_rows) with each measured
    kernel as the primary reading (E82.PRIMARY set for the call in this process only; the file is not changed)."""
    out = []
    rd2 = pd.concat([rd, res.drop(columns=['measured_kernel'])], ignore_index=True)
    pp2 = pd.concat([pp, allp], ignore_index=True)
    runs = E82.saved_runs()
    saved = E82.PRIMARY
    try:
        for name, k in kernels.items():
            E82.PRIMARY = k
            g = E82.gate_rows(rd2, pp2, runs)
            g.insert(0, 'measured_kernel', name)
            g.insert(1, 'kernel', E82.klabel(k))
            out.append(g)
    finally:
        E82.PRIMARY = saved
    E82.PRIMARY = saved
    g0 = E82.gate_rows(rd, pp, runs)
    g0.insert(0, 'measured_kernel', 'e82_primary')
    g0.insert(1, 'kernel', E82.klabel(saved))
    pd.concat([g0] + out, ignore_index=True).to_csv(out_dir() / 'gate_measured_kernel.csv', index=False)


# ---------------------------------------------------------------------------
# part E: checks

TAIL_END = (35.0, 40.0)


def _tail_tau(y: np.ndarray, t: np.ndarray) -> float:
    """Decay time constant of a mean trace from 20 s to the excerpt's end (a * exp(-t / tau), tau within 2 to 200 s)."""
    from scipy.optimize import curve_fit
    m = t >= 20.0
    try:
        p, _ = curve_fit(lambda x, a, tau: a * np.exp(-(x - 20.0) / tau), t[m], y[m], p0=(max(float(y[m][0]), 1e-3), 15.0),
                         bounds=([0, 2.0], [np.inf, 200.0]), maxfev=20000)
        return float(p[1])
    except Exception:
        return np.nan


def carry_over() -> None:
    """carry_over.csv, carry_over_summary.csv and carry_over_pairs.csv: the carry-over of a response into the next
    presentation, per recording, in baseline-noise units at the target's window.

    - Observed: for KCs that responded to the previous presentation (single trial above the bands' criterion), the next
      presentation's baseline shift (its pre-onset mean of Fc minus the previous presentation's) and its window response;
      within entries (an entry's repeats preceded by a response of the KC against its repeats that were not), the
      difference of the window response; and the slope of an entry's deviation from its mean over repeats on its
      predecessor's (carry-over and any state shared by consecutive presentations together).
    - The tail correction: each KC's signal 35 to 40 s after the previous onset (its own excerpt) extrapolated with the
      recording's decay time constant (the responders' mean trace from 20 s on) over the next presentation's baseline
      and window (at the actual onset gap), the difference being the tail's part of the next response. Each single trial
      is corrected by it (in Yang's z for the distances) and the per-pair bias of the trial-mean and noise-free distances
      and of the APL effect is the corrected minus the measured.
    - The calibration: on entries silent on their other repeats (mean below 1 noise SD, where a gain shared by
      consecutive presentations has little to act on), the observed difference of the next window response after a
      responder and after a non-responder against the tail model's prediction for the same trials; their ratio, pooled
      per genotype and clipped to 0 to 1 (a negative ratio is an offset that is not a decaying tail), scales the tail
      correction in a second, calibrated reading ('_cal' columns).
    """
    thr = float(H.data_target('noise_threshold'))
    TR = _traces_all()
    rows, xs, ys = [], {g: [] for g in GENOS}, {g: [] for g in GENOS}
    dist = {g: {k: [] for k in ('measured', 'noise_free', 'measured_corr', 'noise_free_corr', 'measured_cal', 'noise_free_cal')}
            for g in GENOS}
    keep = {}
    silent_obs = {g: [0.0, 0.0] for g in GENOS}
    for g, rec in YRN.RECORDINGS:
        c = load_cache(g, rec)
        m = c['meta']
        key = tuple(m['key'])
        v, B = m['volume_rate'], m['B']
        Wn = window_values(c, [TARGET, TAIL_END], 'noise')                  # (51, KCs, 2)
        Wz = window_values(c, [TARGET, TAIL_END], 'z')
        Rn, En = Wn[..., 0], Wn[..., 1]
        Rz, Ez = Wz[..., 0], Wz[..., 1]
        sb = c['sd_base_0.7']
        y, _, _ = _marginal(TR[(g, key)], 'resp')
        tau = _tail_tau(y, TR[(g, key)]['t'])
        on = np.array(m['onsets'])
        # the tail's part of each presentation's response: (window mean - baseline mean) / end mean of exp(-t / tau)
        f = np.zeros(len(on))
        e_lo, e_hi = H.window_volumes(TAIL_END, v)
        t_end = np.arange(e_lo, e_hi) / v
        for i in range(1, len(on)):
            gap = int(on[i] - on[i - 1])
            tb = (gap + np.arange(-B, 0)) / v
            tw = (gap + np.arange(m['S'], m['P'])) / v
            f[i] = (np.exp(-tw / tau).mean() - np.exp(-tb / tau).mean()) / np.exp(-t_end / tau).mean()
        prev_n = np.vstack([np.zeros((1, Rn.shape[1])), Rn[:-1]])
        tail_n = np.vstack([np.zeros((1, Rn.shape[1])), En[:-1]]) * f[:, None]
        tail_z = np.vstack([np.zeros((1, Rz.shape[1])), Ez[:-1]]) * f[:, None]
        shift = np.vstack([np.full((1, Rn.shape[1]), np.nan), np.diff(c['base_fc'], axis=0)]) / sb[None]
        rp = prev_n > thr
        rp[0] = False
        oth = ~rp
        oth[0] = False
        idx = trial_index(c, ODORS)                                           # (3, 16)
        # within entries: repeats preceded by a response of the KC against those that were not
        R3, P3 = Rn[idx], rp[idx]                                             # (3, 16, KCs)
        ok3 = (idx > 0)[..., None] & np.ones_like(P3, dtype=bool)
        both = (P3 & ok3).any(0) & ((~P3) & ok3).any(0)
        with np.errstate(invalid='ignore'):
            a_ = np.where(P3 & ok3, R3, 0).sum(0) / (P3 & ok3).sum(0)
            b_ = np.where((~P3) & ok3, R3, 0).sum(0) / ((~P3) & ok3).sum(0)
        T3 = tail_n[idx]
        with np.errstate(invalid='ignore'):
            ta = np.where(P3 & ok3, T3, 0).sum(0) / (P3 & ok3).sum(0)
            tb_ = np.where((~P3) & ok3, T3, 0).sum(0) / ((~P3) & ok3).sum(0)
        # entries silent on their other repeats: observed and predicted offsets after a responder
        other = np.full(Rn.shape, np.nan)
        for h in range(idx.shape[0]):
            rest = [r for r in range(idx.shape[0]) if r != h]
            other[idx[h]] = Rn[idx[rest]].mean(axis=0)
        sil = np.isfinite(other) & (other < 1.0)
        sil[0] = False
        A, Bm = sil & rp, sil & oth
        d_obs = float(Rn[A].mean() - Rn[Bm].mean()) if A.any() and Bm.any() else np.nan
        d_pred = float(tail_n[A].mean() - tail_n[Bm].mean()) if A.any() and Bm.any() else np.nan
        if np.isfinite(d_obs):
            silent_obs[g][0] += d_obs * A.sum()
            silent_obs[g][1] += d_pred * A.sum()
        ym = R3 - R3.mean(axis=0, keepdims=True)
        x3 = prev_n[idx]
        xm = x3 - x3.mean(axis=0, keepdims=True)
        okm = np.broadcast_to((idx > 0)[..., None], xm.shape)
        xs[g].append(xm[okm])
        ys[g].append(ym[okm])
        rows.append(dict(genotype=g, recording=rec_id(key), tail_tau=tau, tail_factor_median=float(np.median(f[1:])),
                         prev_responders=int(rp.sum()), prev_response_mean=float(prev_n[rp].mean()),
                         tail_level_35_40_after_responders=float(np.vstack([np.zeros((1, En.shape[1])), En[:-1]])[rp].mean()),
                         baseline_shift_after_responders=float(np.nanmean(shift[rp])),
                         baseline_shift_after_others=float(np.nanmean(shift[oth])),
                         next_window_after_responders=float(Rn[rp].mean()), next_window_after_others=float(Rn[oth].mean()),
                         within_entry_diff=float(np.nanmean((a_ - b_)[both])), within_entry_entries=int(both.sum()),
                         within_entry_tail_predicted=float(np.nanmean((ta - tb_)[both])),
                         tail_bias_after_responders=float(tail_n[rp].mean()), tail_bias_all=float(tail_n[1:].mean()),
                         silent_after_responders_entries=int(A.sum()), silent_observed_offset=d_obs,
                         silent_predicted_offset=d_pred))
        keep[(g, key)] = (Rz, tail_z, idx)
        print(f"E carry-over {g} {rec}: tau {tau:.1f} s, tail factor {np.median(f[1:]):+.3f}", flush=True)
        del c
    beta = {g: float(np.concatenate(xs[g]) @ np.concatenate(ys[g]) / (np.concatenate(xs[g]) @ np.concatenate(xs[g])))
            for g in GENOS}
    scale_raw = {g: float(silent_obs[g][0] / silent_obs[g][1]) for g in GENOS}
    # (a negative ratio is an offset that is no decaying tail: no tail correction for that genotype)
    scale = {g: float(np.clip(v, 0.0, 1.0)) for g, v in scale_raw.items()}
    for (g, key), (Rz, tail_z, idx) in keep.items():
        for lab, W in (('', Rz), ('_corr', Rz - tail_z), ('_cal', Rz - scale[g] * tail_z)):
            T = np.moveaxis(W[idx], 2, 1)                                     # (3, KCs, 16)
            dist[g]['measured' + lab].append(_pairs_array(H.pair_distances(pd.DataFrame(T.mean(0), columns=ODORS), 'cosine')))
            dist[g]['noise_free' + lab].append(_pairs_array(H.noise_free_pair_distances(T, ODORS)))
    co = pd.DataFrame(rows)
    co['calibration_ratio_genotype'] = co.genotype.map(scale_raw)
    co['calibration_scale_genotype'] = co.genotype.map(scale)
    co['slope_on_previous_genotype'] = co.genotype.map(beta)
    prs, summ = [], []
    for est in ESTIMATORS:
        e0 = np.mean(dist['TNT3c'][est], 0) - np.mean(dist['TNT'][est], 0)
        e1 = np.mean(dist['TNT3c'][est + '_corr'], 0) - np.mean(dist['TNT'][est + '_corr'], 0)
        e2 = np.mean(dist['TNT3c'][est + '_cal'], 0) - np.mean(dist['TNT'][est + '_cal'], 0)
        bias = e1 - e0
        bias_cal = e2 - e0
        bc = np.mean(dist['TNT3c'][est + '_corr'], 0) - np.mean(dist['TNT3c'][est], 0)
        bt = np.mean(dist['TNT'][est + '_corr'], 0) - np.mean(dist['TNT'][est], 0)
        for j, (a, bb) in enumerate(_PAIR_IDX):
            prs.append(dict(estimator=est, odor_a=a, odor_b=bb, effect=e0[j], effect_corrected=e1[j], bias=bias[j],
                            bias_ctrl_distance=bc[j], bias_tnt_distance=bt[j], effect_corrected_cal=e2[j],
                            bias_cal=bias_cal[j]))
        summ.append(dict(estimator=est, slope_ctrl=beta['TNT3c'], slope_tnt=beta['TNT'], max_abs_bias=float(np.abs(bias).max()),
                         mean_bias=float(bias.mean()), rms_bias=float(np.sqrt((bias ** 2).mean())),
                         pairs_bias_ge_0p01=int((np.abs(bias) >= 0.01).sum()),
                         max_abs_bias_ctrl_distance=float(np.abs(bc).max()), max_abs_bias_tnt_distance=float(np.abs(bt).max()),
                         calibration_ratio_ctrl=scale_raw['TNT3c'], calibration_ratio_tnt=scale_raw['TNT'],
                         calibration_scale_ctrl=scale['TNT3c'], calibration_scale_tnt=scale['TNT'],
                         max_abs_bias_cal=float(np.abs(bias_cal).max()), mean_bias_cal=float(bias_cal.mean()),
                         rms_bias_cal=float(np.sqrt((bias_cal ** 2).mean())), pairs_bias_cal_ge_0p01=int((np.abs(bias_cal) >= 0.01).sum())))
    co.to_csv(out_dir() / 'carry_over.csv', index=False)
    pd.DataFrame(summ).to_csv(out_dir() / 'carry_over_summary.csv', index=False)
    pd.DataFrame(prs).to_csv(out_dir() / 'carry_over_pairs.csv', index=False)
    print(pd.DataFrame(summ).to_string(), flush=True)


def neuropil() -> None:
    """neuropil_by_window.csv: per Fneu coefficient (0.5, 0.7, 0.9) and named window, the mean distances and the APL
    effect (mean, groups) measured and noise-free in Yang's z; and the all-entries and responders' signal 4-8.1 s over
    0-2 s per genotype (baseline-noise units at that coefficient; responders chosen at that coefficient)."""
    wl = list(NAMED_WINDOWS.items())
    acc = {}
    sig = {}
    for g, rec in YRN.RECORDINGS:
        c = load_cache(g, rec)
        for cf in NEUCOEFFS:
            W = window_values(c, [w for _, w in wl], 'z', cf)
            T = arrange(W, c, ODORS)
            for wi, (name, _) in enumerate(wl):
                X = T[..., wi]
                acc.setdefault((cf, name, 'measured', g), []).append(
                    _pairs_array(H.pair_distances(pd.DataFrame(X.mean(0), columns=ODORS), 'cosine')))
                acc.setdefault((cf, name, 'noise_free', g), []).append(_pairs_array(H.noise_free_pair_distances(X, ODORS)))
            R = recording_traces(c, cf)
            for sel in ('resp', 'all'):
                y, _, _ = _marginal(R, sel)
                t = R['t']
                k0 = int(np.flatnonzero(t >= -1e-12)[0])
                v = c['meta']['volume_rate']
                ms = [np.mean(y[k0 + a:k0 + b]) for a, b in (H.window_volumes(w, v) for w in (EARLY, LATE))]
                sig.setdefault((cf, sel, g, tuple(c['meta']['key'])), ms)
        print(f'E neuropil {g} {rec}', flush=True)
        del c
    rows = []
    for cf in NEUCOEFFS:
        for name, (a, b) in wl:
            for est in ESTIMATORS:
                C, T = np.mean(acc[(cf, name, est, 'TNT3c')], 0), np.mean(acc[(cf, name, est, 'TNT')], 0)
                E = C - T
                rows.append(dict(neucoeff=cf, window=name, t0=a, t1=b, estimator=est, ctrl_mean=float(np.nanmean(C)),
                                 tnt_mean=float(np.nanmean(T)), effect_mean=float(np.nanmean(E)),
                                 effect_ester=float(E[PAIR_MASKS['ester']].mean()),
                                 effect_alcohol=float(E[PAIR_MASKS['alcohol']].mean()),
                                 effect_six=float(E[PAIR_MASKS['six']].mean())))
        for sel in ('resp', 'all'):
            for which, lab in ((None, 'all eight'), ('six', 'the six typical')):
                cm = np.array([v for (q, s, g, k), v in sig.items() if q == cf and s == sel and g == 'TNT3c'])
                tmm = np.array([v for (q, s, g, k), v in sig.items() if q == cf and s == sel and g == 'TNT'
                                and _in_subset('TNT', k, which)])
                r_c = cm[:, 1].mean() / cm[:, 0].mean()
                r_t = tmm[:, 1].mean() / tmm[:, 0].mean()
                rows.append(dict(neucoeff=cf, window='signal 4-8.1 over 0-2', selection=sel, tnt_subset=lab,
                                 ctrl_late_over_early=float(r_c), tnt_late_over_early=float(r_t), tnt_over_ctrl=float(r_t / r_c)))
    pd.DataFrame(rows).to_csv(out_dir() / 'neuropil_by_window.csv', index=False)


def part_e(n_boot: int = N_BOOT, carry_only: bool = False) -> None:
    """Part E: carry_over.csv, carry_over_summary.csv, carry_over_pairs.csv, neuropil_by_window.csv (not with
    `carry_only`)."""
    write_provenance('E' if not carry_only else 'E-carry', dict(neucoeffs=list(NEUCOEFFS), tail_end=list(TAIL_END)))
    carry_over()
    if not carry_only:
        neuropil()


# ---------------------------------------------------------------------------
# export: every KC's per-trial excerpts (Yang, 2026-11-11 about 10:55)

def traces_dir() -> Path:
    return H.shared_dir() / 'data' / 'yang_apl16_kc_traces'


MAX_EXPORT_BYTES = 5e9


def export_traces() -> None:
    """data/yang_apl16_kc_traces/ in the shared directory: one parquet per recording (<genotype>_<recording>.parquet), one
    row per KC and trial (olfdb's keys and roi, the trial's presentation, block, position, odor and repeat, the KC's two
    scales) with the excerpt's values as float32 columns v<k> (volume k after the onset volume, at k / volume_rate s),
    from the cache; volume_times.csv, README.md and provenance.json. Each written file is read back and its target's
    window checked against olfdb's per-trial values. Refuses to write (RuntimeError) if the values would exceed
    MAX_EXPORT_BYTES."""
    total, metas = 0, []
    for g, rec in YRN.RECORDINGS:
        p = _cache_path(g, rec)
        with np.load(p, allow_pickle=False) as z:
            sh = z['dfc'].shape
        total += int(np.prod(sh)) * 4
        metas.append((g, rec, sh))
    print(f'export: {total / 1e9:.2f} GB of float32 values in {len(metas)} recordings', flush=True)
    if total > MAX_EXPORT_BYTES:
        raise RuntimeError(f'export: {total / 1e9:.2f} GB is above the {MAX_EXPORT_BYTES / 1e9:.0f} GB limit; not written')
    d = traces_dir()
    d.mkdir(parents=True, exist_ok=True)
    checks, vt = {}, []
    for g, rec in YRN.RECORDINGS:
        c = load_cache(g, rec)
        m = c['meta']
        key = tuple(m['key'])
        K, n, T = c['dfc'].shape
        vols = np.arange(-m['B'], m['n_after'])
        nstim = int(m['yaml']['nstim'])
        vals = np.ascontiguousarray(c['dfc'].transpose(1, 0, 2).reshape(n * K, T), dtype=np.float32)
        df = pd.DataFrame(vals, columns=[f'v{k}' for k in vols])
        roi = np.repeat(np.arange(n), K)
        trial = np.tile(np.arange(K), n)
        front = pd.DataFrame(dict(date=key[0], fly_num=key[1], hemibrain=key[2], roi=roi, genotype=g, dataset=DATASET[g],
                                  presentation=trial, block=trial // nstim, position=trial % nstim,
                                  odor=np.array(m['odor_names'])[trial], odor_abbrev=np.array(m['odors'])[trial],
                                  repeat=np.array(m['repeat'])[trial], onset_volume=np.array(m['onsets'])[trial],
                                  sd_full=c['sd_full_0.7'][roi], sd_base=c['sd_base_0.7'][roi]))
        df = pd.concat([front, df], axis=1)
        path = d / f'{g}_{rec}.parquet'
        df.to_parquet(path, index=False)
        del df, vals
        # read back: the target's window over sd_full against olfdb's per-trial values
        back = pd.read_parquet(path)
        S, P = m['S'], m['P']
        w = back[[f'v{k}' for k in range(S, P)]].to_numpy(dtype=np.float64).mean(axis=1) / back.sd_full.to_numpy()
        ours = pd.DataFrame(dict(roi=back.roi, odor=back.odor, repeat=back.repeat, value=w)).pivot_table(
            index='roi', columns=['odor', 'repeat'], values='value')
        om = YRN.olfdb_matrix(g, rec)
        common = [x for x in ours.columns if x in om.columns]
        dmax = float(np.abs(om[common].to_numpy() - ours.loc[om.index, common].to_numpy()).max())
        checks[f'{g}:{rec}'] = dict(rows=len(back), kcs=n, trials=K, samples=T, olfdb_max_abs=dmax,
                                    olfdb_trials=len(common), bytes=path.stat().st_size, sha256=_sha(path))
        vt += [dict(recording=rec, genotype=g, volume=int(k), t_s=float(k / m['volume_rate'])) for k in vols]
        print(f'export {g} {rec}: {len(back)} rows, {path.stat().st_size / 1e6:.0f} MB, olfdb {dmax:.2g}', flush=True)
        del back, c
        if dmax > 1e-5:
            raise RuntimeError(f'export {rec}: the written values do not reproduce olfdb ({dmax:.2g})')
    pd.DataFrame(vt).to_csv(d / 'volume_times.csv', index=False)
    rec0 = load_cache(*YRN.RECORDINGS[0])['meta']
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E84_trace_time_course.py --part export',
                al_analysis_commit=_git('rev-parse', 'HEAD'), neucoeff=NEUCOEFF, volume_rate=rec0['volume_rate'],
                frame_rate=rec0['yaml']['frame_rate'], frame_per_volume=rec0['yaml']['frame_per_volume'], files=checks,
                total_bytes=int(sum(v['bytes'] for v in checks.values())),
                olfdb_max_abs=max(v['olfdb_max_abs'] for v in checks.values()))
    (d / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    (d / 'README.md').write_text(export_readme(prov))


def export_readme(prov: dict) -> str:
    v = prov['volume_rate']
    return f"""# Yang's APL16 KC traces: every KC's per-trial excerpts

The per-trial signal of every KC of the 19 APL16 recordings of olfdb datasets 19 (APL output blocked (TNT), 8 recordings)
and 20 (control flies, TNT3c, 11 recordings), around each of the 51 presentations (16 odors and the solvent, pfo, three
repeats). Yang allowed sharing these traces on 2026-11-11 (in viki's terminal, about 10:55), for these 19 recordings
only; the 12 APL16 recordings on her drives that are not in olfdb are on hold and are not here. Written by
`al_analysis/scripts/claude_pair_remapping_experiments/E84_trace_time_course.py --part export` ({prov['generated']},
al_analysis {prov['al_analysis_commit'][:7]}) from the suite2p output on Yang's drive, read by
`yang_response_normalization.read_suite2p`; re-run it on viki to regenerate.

## Files

- `<genotype>_<recording>.parquet` (19 files, {prov['total_bytes'] / 1e9:.2f} GB): one row per KC and presentation.
  - keys as olfdb's: `date`, `fly_num`, `hemibrain`, `roi` (the KC's position among the ROIs suite2p kept as cells,
    iscell == 1, which is olfdb's roi), `genotype`, `dataset` (olfdb's);
  - the presentation: `presentation` (0 to 50, in time order), `block` (0 to 2) and `position` (0 to 16 within the block),
    `odor` (olfdb's name, `<abbrev> @ <log10 conc>`), `odor_abbrev`, `repeat` (0 to 2, olfdb's: the odor's earlier
    presentations), `onset_volume` (the onset's volume in the recording);
  - the KC's scales: `sd_full` (the SD of its Fc over the whole recording, about 7,200 volumes or 40 min: Yang's z
    unit) and `sd_base` (the SD of Fc within the 51 pre-odor windows, each window's mean
    removed: the baseline-noise unit of the realism bands);
  - the excerpt: float32 columns `v-15` to `v119`, the value at volume k after the onset volume (time k / {v:.5f} s:
    -5.06 s to 40.15 s; `volume_times.csv` lists the times).
- `volume_times.csv`: recording, genotype, volume, t_s.
- `provenance.json`: the commit, the rates, each file's rows, size, sha256 and its check against olfdb.

## Units and window

A KC's signal is suite2p's Fc = F - 0.7 Fneu (raw fluorescence units, not divided by a resting fluorescence). Each
excerpt runs from the 15 volumes before the onset (5 s) to 119 volumes after it, with the mean of its 15 pre-onset
volumes subtracted (so the response in any window is the window's mean of these values). Frame rate
{prov['frame_rate']} frames / s, {prov['frame_per_volume']} frames per volume, {v:.5f} volumes / s. Yang's z: value /
sd_full; baseline-noise units: value / sd_base. Yang's per-trial response (olfdb's values) is the mean over volumes
6 to 23 (2.0 to 8.1 s; her 2-8 s window, round(2 v) to round(8 v) - 1) divided by sd_full.

## Check

Each file was read back after writing: the mean over volumes 6 to 23 over sd_full equals olfdb's per-trial values of
the recording (data/yang_apl16_kc_olfdb/) to at most {prov['olfdb_max_abs']:.2g} (float32 storage) in every recording and
trial. E84's part A checks the same window from the cache against olfdb, the target's two matrices and the
baseline-noise values of data/yang_apl16_kc_noise_units/ (results/E84_trace_time_course/reproduction_checks.json).

## Caveats

Not a unit of activity: Fc is not normalized by a resting fluorescence, so amplitudes depend on how bright a KC is
(sd_full and sd_base put them in the two units the queue uses). The order of the presentations differs between
recordings and between a recording's blocks (results/E84_trace_time_course/presentation_order.csv).
"""


# ---------------------------------------------------------------------------
# tables and figures

def _f(x, spec: str = '.3f') -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '-'
    return '-' if not np.isfinite(x) else format(x, spec)


def _ci(r, spec='+.3f', kind='rec') -> str:
    return f"{_f(r.value, spec)} ({_f(getattr(r, f'lo_{kind}'), spec)} to {_f(getattr(r, f'hi_{kind}'), spec)})"


def tables() -> str:
    """comparison_table.md from the parts' CSVs (the sections whose CSVs exist)."""
    o = out_dir()
    lines = [f'# E84: the time course of the APL effect in the flies\' traces',
             '', f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by E84_trace_time_course.py (al_analysis "
             f"{_git('rev-parse', '--short', 'HEAD')}). Cosine distances between trial-mean response vectors over all KCs of a "
             'recording (\'as measured\'), or noise-free from products of different repeats (E81), means over recordings; '
             'intervals 90 % over 1000 resamples of recordings (of flies in the effect table\'s last column). KC data in '
             "Yang's z unless a column says otherwise.", '']
    if (o / 'reproduction_checks.json').exists():
        rc = json.loads((o / 'reproduction_checks.json').read_text())
        s = rc['summary']
        lines += ['## Reproduction checks (part A)', '', '| check | largest difference | tolerance |', '|---|---:|---:|',
                  f"| the target's window against olfdb's per-trial values (19 recordings) | {s['olfdb_max_abs']:.2g} | 1e-5 |",
                  f"| baseline-noise values against data/yang_apl16_kc_noise_units (relative) | {s['noise_units_max_rel']:.2g} | 1e-5 |",
                  f"| baseline-noise scale sd_base against kc_scales.csv (relative) | {s['sd_base_max_rel']:.2g} | 1e-9 |",
                  f"| per-recording cosine distances against the target's | {s['rdm_max_abs']:.2g} | 1e-6 |",
                  f"| the target's two matrices (control flies / APL output blocked (TNT)) | {s['mean_rdm_max_abs']['TNT3c']:.2g} / "
                  f"{s['mean_rdm_max_abs']['TNT']:.2g} | 1e-6 |",
                  f"| E81's noise-free registry entries | {s['registry']['max_abs_diff']:.2g} | 1e-6 |", '']
    if (o / 'effect_by_window.csv').exists():
        ef = pd.read_csv(o / 'effect_by_window.csv')
        order = [w[0] for w in windows()]
        main = ['0 to 1 s', '1 to 2 s', '2-4', '4-8.1', '8.1-10.1', '10.1-12.2', 'pulse', 'target', '0-8.1']
        for units in ('z', 'noise'):
            for sname in ('all', 'six_typical', 'two_strongest', 'tnt_without_unlabeled'):
                x = ef[(ef.subset == sname) & (ef.units == units)]
                if not len(x):
                    continue
                lines += [f"## The target by window: {sname.replace('_', ' ')}, {'Yang z' if units == 'z' else 'baseline-noise units'}", '',
                          '| window | mean distance, noise-free: control flies / APL output blocked (TNT) | mean APL effect, as '
                          'measured | mean APL effect, noise-free | ester group, noise-free | alcohol group, noise-free | ρ on the '
                          'per-pair APL effect with the target\'s, as measured | eight wirings\' pattern error (calibrated), '
                          'noise-free | mean APL effect, noise-free, resampling flies |',
                          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
                cw = pd.read_csv(o / 'common_error_by_window.csv') if (o / 'common_error_by_window.csv').exists() else None
                for w in main:
                    def g(stat, est):
                        q = x[(x.window == w) & (x.statistic == stat) & (x.estimator == est)]
                        return q.iloc[0] if len(q) else None
                    a, b = g('ctrl_mean', 'noise_free'), g('tnt_mean', 'noise_free')
                    em, en = g('effect_mean', 'measured'), g('effect_mean', 'noise_free')
                    es, al = g('effect_ester', 'noise_free'), g('effect_alcohol', 'noise_free')
                    rho = g('rho_main_target', 'measured')
                    pe = '-'
                    if cw is not None:
                        q = cw[(cw.subset == sname) & (cw.units == units) & (cw.estimator == 'noise_free') & (cw.window == w)
                               & (cw.population == 'eight') & (cw.readout == 'cal') & (cw.statistic == 'pattern')]
                        if len(q):
                            pe = _ci(q.iloc[0], '.3f')
                    if a is None:
                        continue
                    lines.append(f"| {w} | {_f(a.value)} / {_f(b.value)} | {_ci(em)} | {_ci(en)} | {_ci(es)} | {_ci(al)} | "
                                 f"{_f(rho.value, '.2f')} | {pe} | {_ci(en, kind='fly')} |")
                c = x[(x.window == 'pulse') & (x.statistic == 'effect_mean_minus_late')]
                if len(c):
                    lines.append('')
                    for est in ESTIMATORS:
                        q = c[c.estimator == est]
                        if len(q):
                            lines.append(f"Pulse minus 4-8.1 s, mean APL effect, {est.replace('_', '-')}: {_ci(q.iloc[0])}; "
                                         f"resampling flies {_ci(q.iloc[0], kind='fly')}.")
                    q = x[(x.window == 'pulse') & (x.statistic == 'alcohol_minus_ester_change_from_late')
                          & (x.estimator == 'noise_free')]
                    if len(q):
                        lines.append(f'Alcohol minus ester group, change from 4-8.1 s to the pulse, noise-free: {_ci(q.iloc[0])}.')
                fl = x[(x.statistic == 'loss_floor') & (x.window.isin(['pulse', '4-8.1', 'target']))]
                if len(fl):
                    lines.append('Floor of the loss / ceiling of ρ by window (as measured; noise-free): ' + '; '.join(
                        f"{w}: {_f(fl[(fl.window == w) & (fl.estimator == 'measured')].value.iloc[0])} / "
                        f"{_f(x[(x.window == w) & (x.statistic == 'rho_ceiling') & (x.estimator == 'measured')].value.iloc[0], '.2f')}"
                        f" ({_f(fl[(fl.window == w) & (fl.estimator == 'noise_free')].value.iloc[0])} / "
                        f"{_f(x[(x.window == w) & (x.statistic == 'rho_ceiling') & (x.estimator == 'noise_free')].value.iloc[0], '.2f')})"
                        for w in ('pulse', '4-8.1', 'target') if len(fl[fl.window == w])) + '.')
                lines.append('')
    if (o / 'common_error_by_window.csv').exists():
        cw = pd.read_csv(o / 'common_error_by_window.csv')
        x = cw[(cw.subset == 'all') & (cw.units == 'z') & (cw.estimator == 'noise_free') & (cw.readout == 'cal')]
        lines += ['## The populations against each window (all recordings, noise-free, Yang\'s z, calibrated readout)', '',
                  '| population | window | pattern error | change from the target\'s window | ester group error | alcohol group error |',
                  '|---|---|---:|---:|---:|---:|']
        for p in POPULATIONS:
            for w in ['0 to 1 s', '1 to 2 s', '2 to 3 s', '3 to 4 s', '4 to 5 s', '5 to 6 s', '6 to 7 s', '7 to 8 s', 'pulse',
                      '2-4', '4-8.1', 'target']:
                def g(stat):
                    q = x[(x.population == p) & (x.window == w) & (x.statistic == stat)]
                    return q.iloc[0] if len(q) else None
                pt = g('pattern')
                if pt is None:
                    continue
                lines.append(f"| {p} | {w} | {_ci(pt, '.3f')} | {_ci(g('pattern_minus_target'))} | {_ci(g('ester_error'))} | "
                             f"{_ci(g('alcohol_error'))} |")
        lines.append('')
    if (o / 'rates_by_window.csv').exists():
        rt = pd.read_csv(o / 'rates_by_window.csv')
        x = rt[(rt.subset == 'all')]
        lines += ['## Response rates by window (baseline-noise units; the bands\' criterion at the target\'s window; elsewhere '
                  'trialmean / single: the solvent\'s share above it matched (it includes the KCs that respond to the solvent); '
                  '_mad: scaled by the solvent\'s robust SD; _fixed: the same threshold)', '',
                  '| criterion | window | control flies | APL output blocked (TNT) | APL-off / APL-on ratio | threshold (control flies) |',
                  '|---|---|---:|---:|---:|---:|']
        for crit in RATE_CRITERIA:
            for w in ('0 to 1 s', '1 to 2 s', 'pulse', '2-4', '4-8.1', 'target', '8.1-10.1'):
                q = x[(x.criterion == crit) & (x.window == w)]
                if len(q):
                    q = q.iloc[0]
                    lines.append(f"| {crit} | {w} | {_f(q.rate_ctrl, '.4f')} | {_f(q.rate_tnt, '.4f')} | {_f(q.ratio, '.2f')} "
                                 f"({_f(q.ratio_lo, '.2f')} to {_f(q.ratio_hi, '.2f')}) | {_f(q.threshold_ctrl, '.2f')} |")
        lines.append('')
    if (o / 'kinetics_by_recording.csv').exists():
        kr = pd.read_csv(o / 'kinetics_by_recording.csv')
        x = kr[(kr.selection == 'resp') & (kr.odor_group == 'all') & (kr.kc_class == 'all') & (kr.repeat.astype(str) == 'all')]
        lines += ['## The responders\' mean trace by recording (baseline-noise units; chosen on two repeats, averaged on the third)',
                  '', '| measure | control flies (11): mean (range) | APL output blocked (TNT), the six typical | '
                  '2024-07-17 fly 3 L | 2024-07-25 fly 1 L |', '|---|---:|---:|---:|---:|']
        for meas, spec in (('peak_t', '.2f'), ('half_rise_t', '.2f'), ('v2_over_peak', '.2f'), ('v8.1_over_peak', '.2f'),
                           ('v20_over_peak', '.2f'), ('v40_over_peak', '.2f'), ('late_over_early', '.2f'),
                           ('decay_tau_from_8.1', '.1f'), ('peak', '.2f')):
            c = x[x.genotype == 'TNT3c'][meas]
            s6 = x[(x.genotype == 'TNT') & ~x.strong][meas]
            s1 = x[x.recording == '2024-07-17 fly 3 L'][meas]
            s2 = x[x.recording == '2024-07-25 fly 1 L'][meas]
            lines.append(f"| {meas} | {_f(c.mean(), spec)} ({_f(c.min(), spec)} to {_f(c.max(), spec)}) | {_f(s6.mean(), spec)} "
                         f"({_f(s6.min(), spec)} to {_f(s6.max(), spec)}) | {_f(s1.iloc[0] if len(s1) else np.nan, spec)} | "
                         f"{_f(s2.iloc[0] if len(s2) else np.nan, spec)} |")
        n08 = int((x['v8.1_over_peak'] >= 0.6).sum())
        n079 = int((x['v8.1_over_peak'] >= 0.79).sum())
        lines += ['', f'Recordings whose responders\' trace is at 0.6 or more of its peak at 8.1 s: {n08} of {len(x)} '
                  f'(0.79 or more: {n079}).', '']
    if (o / 'kinetics_relations.csv').exists():
        rel = pd.read_csv(o / 'kinetics_relations.csv')
        x = rel[(rel.selection == 'resp') & rel.measure.isin(KIN_MEASURES)]
        st = [c for c in x.columns if c.startswith('strong_') and not c.endswith('_outside')]
        lines += ['## The two strongest recordings against the other 17 (responders)', '',
                  '| measure | range of the other 17 | ' + ' | '.join(c.replace('strong_', '').replace('_', ' ') for c in st)
                  + ' | ρ with median F (19) | ρ with baseline noise (19) |', '|---|---:|' + '---:|' * (len(st) + 2)]
        for _, r in x.iterrows():
            cells = [f"{_f(r[c], '.2f')}{' (outside)' if str(r[c + '_outside']) == 'True' else ''}" for c in st]
            lines.append(f"| {r['measure']} | {_f(r['others_min'], '.2f')} to {_f(r['others_max'], '.2f')} | " + ' | '.join(cells)
                         + f" | {_f(r['rho_median_F_19'], '.2f')} | {_f(r['rho_median_sd_base_19'], '.2f')} |")
        lines.append('')
    if (o / 'kernel_fits.csv').exists():
        kf = pd.read_csv(o / 'kernel_fits.csv')
        x = kf[kf.level == 'subset']
        lines += ['## Kernel fits to the responders\' mean traces (rise / decay, s; rms misfit in noise SDs)', '',
                  '| selection | trace | fit | drive | rise | decay | rms | rms of the trace\'s own fit |', '|---|---|---|---|---:|---:|---:|---:|']
        for r in x.itertuples():
            lines.append(f"| {r.selection} | {r.subset} | {r.fit}{' (' + r.kernel_from + ')' if isinstance(r.kernel_from, str) and r.kernel_from else ''} "
                         f"| {r.drive} | {_f(r.rise, '.2f')}{' (bound)' if r.rise_at_bound else ''} | {_f(r.decay, '.1f')}"
                         f"{' (bound)' if r.decay_at_bound else ''} | {_f(r.rms, '.3f')} | {_f(getattr(r, 'rms_own_fit', np.nan), '.3f')} |")
        lines.append('')
    if (o / 'kernel_recovery.csv').exists():
        kr = pd.read_csv(o / 'kernel_recovery.csv')
        lines += ['## Recovery test: synthetic traces with firing uniform over the first D s, how often each D is chosen', '',
                  '| noise | true D | D 0 (onset) | D 0.5 | D 1 | D 2 (steady) | rise recovered (median, 5-95 %) | decay recovered |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in kr.itertuples():
            lines.append(f"| {r.noise} ({_f(r.noise_sd, '.3f')}) | {r.true_D:g} | {_f(r.chosen_D0, '.2f')} | {_f(r.chosen_D0p5, '.2f')} "
                         f"| {_f(r.chosen_D1, '.2f')} | {_f(r.chosen_D2, '.2f')} | {_f(r.rise_median, '.2f')} ({_f(r.rise_p5, '.2f')} to "
                         f"{_f(r.rise_p95, '.2f')}) | {_f(r.decay_median, '.1f')} ({_f(r.decay_p5, '.1f')} to {_f(r.decay_p95, '.1f')}) |")
        lines.append('')
    if (o / 'trace_ratio_tests.csv').exists():
        rt = pd.read_csv(o / 'trace_ratio_tests.csv')
        fl = rt[(rt.kind == 'flies') & (rt.ratio == '4-8.1 over 0-2')]
        lines += ['## The signal with APL output blocked (TNT) relative to the control flies\', 4-8.1 s over 0-2 s', '',
                  '| | responders | entries below the criterion | all KC x odor entries |', '|---|---:|---:|---:|']
        for sub in fl.subset.unique():
            q = fl[fl.subset == sub].set_index('selection')
            lines.append(f"| flies, {sub} | " + ' | '.join(
                f"{_f(q.loc[s].value, '.2f')} ({_f(q.loc[s].lo, '.2f')} to {_f(q.loc[s].hi, '.2f')})" if s in q.index else '-'
                for s in ('resp', 'below', 'all')) + ' |')
        md = rt[(rt.kind == 'model') & (rt.ratio == '4-8.1 over 0-2')]
        for kern in md.kernel.unique():
            q = md[md.kernel == kern]
            for (src, pid), g in q.groupby(['source', 'point_id']):
                gg = g.set_index('selection')
                lines.append(f"| {src}:{pid} ({kern}) | " + ' | '.join(
                    f"{_f(gg.loc[s].value, '.2f')}{' excluded' if gg.loc[s].excluded is True or gg.loc[s].excluded == 'True' else ''}"
                    if s in gg.index else '-' for s in ('resp', 'below', 'all')) + ' |')
        lines.append('')
    if (o / 'protocol_readings_measured_kernel.csv').exists():
        pr = pd.read_csv(o / 'protocol_readings_measured_kernel.csv')
        e82 = pd.read_csv(H.results_dir('E82_flies_protocol') / 'protocol_readings.csv')
        lines += ['## E82\'s readings through the measured kernels (raw loss / mean / pattern; calibrated loss)', '',
                  '| point | counts | E82 primary (rise 0.5, decay 3) | ' + ' | '.join(sorted(pr.measured_kernel.unique())) + ' |',
                  '|---|---:|---:|' + '---:|' * pr.measured_kernel.nunique()]
        for (src, pid), g in pr.groupby(['source', 'point_id']):
            c = e82[(e82.source == src) & (e82.point_id == pid)].set_index('kernel')

            def cell(q):
                return f"{_f(q.raw_loss)} / {_f(q.raw_mean, '+.3f')} / {_f(q.raw_pattern)}; {_f(q.cal_loss)}"
            row = [cell(c.loc['counts']) if 'counts' in c.index else '-', cell(c.loc['r0.5_d3']) if 'r0.5_d3' in c.index else '-']
            row += [cell(g[g.measured_kernel == k].iloc[0]) for k in sorted(pr.measured_kernel.unique())]
            lines.append(f'| {src}:{pid} | ' + ' | '.join(row) + ' |')
        lines.append('')
    if (o / 'gate_measured_kernel.csv').exists():
        gm = pd.read_csv(o / 'gate_measured_kernel.csv')
        lines += ['## The gates under each reading (protocol gate / E77\'s gate)', '', '| point | ' + ' | '.join(gm.measured_kernel.unique())
                  + ' |', '|---|' + '---:|' * gm.measured_kernel.nunique()]
        for pid, g in gm.groupby('point_id'):
            gg = g.set_index('measured_kernel')
            lines.append(f'| {pid} | ' + ' | '.join(f"{gg.loc[k].protocol_gate} / {gg.loc[k].e77_gate}" if k in gg.index else '-'
                                                   for k in gm.measured_kernel.unique()) + ' |')
        lines.append('')
    if (o / 'carry_over_summary.csv').exists():
        cs = pd.read_csv(o / 'carry_over_summary.csv')
        co = pd.read_csv(o / 'carry_over.csv')
        lines += ['## The carry-over from the previous presentation (baseline-noise units)', '',
                  '| | control flies | APL output blocked (TNT) |', '|---|---:|---:|']
        for col in ('tail_tau', 'tail_factor_median', 'prev_response_mean', 'tail_level_35_40_after_responders',
                    'baseline_shift_after_responders', 'baseline_shift_after_others', 'next_window_after_responders',
                    'next_window_after_others', 'within_entry_diff', 'within_entry_tail_predicted', 'tail_bias_after_responders',
                    'tail_bias_all', 'silent_observed_offset', 'silent_predicted_offset', 'calibration_ratio_genotype',
                    'calibration_scale_genotype',
                    'slope_on_previous_genotype'):
            lines.append(f"| {col.replace('_', ' ')} | {_f(co[co.genotype == 'TNT3c'][col].mean())} | {_f(co[co.genotype == 'TNT'][col].mean())} |")
        for r in cs.itertuples():
            lines.append(f"| per-pair bias of the APL effect ({r.estimator}), tail model: largest / rms / pairs at 0.01 or more | "
                         f"{_f(r.max_abs_bias, '.4f')} / {_f(r.rms_bias, '.4f')} / {r.pairs_bias_ge_0p01} | |")
            lines.append(f"| the same, calibrated on silent entries | {_f(r.max_abs_bias_cal, '.4f')} / {_f(r.rms_bias_cal, '.4f')} / "
                         f"{r.pairs_bias_cal_ge_0p01} | |")
        lines.append('')
    if (o / 'neuropil_by_window.csv').exists():
        nb = pd.read_csv(o / 'neuropil_by_window.csv')
        x = nb[nb.window.isin(['pulse', '4-8.1', 'target']) & (nb.estimator == 'noise_free')]
        lines += ['## The neuropil coefficient (noise-free, Yang\'s z)', '', '| Fneu coefficient | window | mean distance control / TNT | '
                  'mean APL effect | ester | alcohol |', '|---:|---|---:|---:|---:|---:|']
        for r in x.itertuples():
            lines.append(f"| {r.neucoeff:g} | {r.window} | {_f(r.ctrl_mean)} / {_f(r.tnt_mean)} | {_f(r.effect_mean, '+.3f')} | "
                         f"{_f(r.effect_ester, '+.3f')} | {_f(r.effect_alcohol, '+.3f')} |")
        y = nb[nb.window == 'signal 4-8.1 over 0-2']
        lines += ['', '| Fneu coefficient | selection | TNT subset | control flies late / early | TNT late / early | TNT over control |',
                  '|---:|---|---|---:|---:|---:|']
        for r in y.itertuples():
            lines.append(f"| {r.neucoeff:g} | {r.selection} | {r.tnt_subset} | {_f(r.ctrl_late_over_early, '.2f')} | "
                         f"{_f(r.tnt_late_over_early, '.2f')} | {_f(r.tnt_over_ctrl, '.2f')} |")
        lines.append('')
    text = '\n'.join(lines) + '\n'
    (o / 'comparison_table.md').write_text(text)
    return text


def figures() -> None:
    """E84_effect_by_window.png (mean distances and the mean APL effect by 1 s bin, noise-free, with intervals) and
    E84_traces.png (the responders' and all entries' mean traces by genotype subset, with the kernel fits)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    o = out_dir()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    if (o / 'effect_by_window.csv').exists():
        ef = pd.read_csv(o / 'effect_by_window.csv')
        x = ef[(ef.subset == 'all') & (ef.units == 'z') & ef.window.str.endswith(' s')]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for est, ls in (('noise_free', '-'), ('measured', ':')):
            for stat, g, ax in (('ctrl_mean', 'TNT3c', axes[0]), ('tnt_mean', 'TNT', axes[0]), ('effect_mean', None, axes[1])):
                q = x[(x.statistic == stat) & (x.estimator == est)].sort_values('t0')
                mid = (q.t0 + q.t1) / 2
                col = COLORS[g] if g else COLORS['ink']
                lab = (GLAB[g] if g else 'mean APL effect') + (', noise-free' if est == 'noise_free' else ', as measured')
                ax.plot(mid, q.value, ls, color=col, lw=2 if est == 'noise_free' else 1.2, label=lab)
                if est == 'noise_free':
                    ax.fill_between(mid, q.lo_rec, q.hi_rec, color=col, alpha=0.15, lw=0)
        for ax in axes:
            ax.axvspan(0, 2, color=COLORS['grid'], alpha=0.6, lw=0)
            ax.axvspan(2.0, 8.1, color=COLORS['grid'], alpha=0.25, lw=0)
            ax.set_xlabel('time after the odor\'s onset (s), 1 s bins')
            ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel('mean pair distance (cosine distance)')
        axes[0].set_title('mean odor-odor distance by window')
        axes[1].set_ylabel('mean APL effect (control minus APL output blocked (TNT))')
        axes[1].set_title('the mean APL effect by window\n(shaded: the pulse; lighter: the target\'s window)')
        fig.tight_layout()
        fig.savefig(o / 'E84_effect_by_window.png', dpi=150)
        plt.close(fig)
    if (o / 'traces.csv').exists():
        tr = pd.read_csv(o / 'traces.csv')
        x = tr[(tr.level == 'subset') & (tr.odor_group == 'all') & (tr.kc_class == 'all') & (tr.repeat.astype(str) == 'all')]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        cols = {'control flies': COLORS['TNT3c'], 'APL output blocked (TNT), all eight': COLORS['TNT'],
                'APL output blocked (TNT), the six typical': COLORS['six'], 'APL output blocked (TNT), the two strongest': COLORS['two']}
        for ax, sel, ttl in ((axes[0], 'resp', 'responders (chosen on two repeats), over the peak'),
                             (axes[1], 'all', 'all KC x odor entries, over the 0-2 s mean')):
            for sub, col in cols.items():
                q = x[(x.subset == sub) & (x.selection == sel)].sort_values('t')
                if not len(q):
                    continue
                y = q.value.to_numpy()
                norm = y[(q.t >= 0) & (q.t <= 20)].max() if sel == 'resp' else y[(q.t >= 0) & (q.t < 2)].mean()
                ax.plot(q.t, y / norm, color=col, lw=1.8, label=f'{sub} (n {int(q.n_entries.iloc[0])})')
            ax.axvspan(0, 2, color=COLORS['grid'], alpha=0.6, lw=0)
            ax.axvline(8.1, color=COLORS['ink2'], ls=':', lw=1)
            ax.set_xlabel('time after the odor\'s onset (s)')
            ax.set_title(ttl)
            ax.legend(frameon=False, fontsize=7)
        axes[0].set_ylabel('signal (baseline-noise units), normalized')
        fig.tight_layout()
        fig.savefig(o / 'E84_traces.png', dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=('A', 'B', 'B-rates', 'C', 'D', 'E', 'E-carry', 'export', 'table'), required=True,
                    help='B-rates: only the response rates of part B (rates_by_window*.csv); E-carry: only the carry-over')
    ap.add_argument('--drive-root', type=Path, default=Path('/media/yang'))
    ap.add_argument('--rebuild-cache', action='store_true')
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    args = ap.parse_args()
    t0 = time.time()
    if args.part == 'A':
        part_a(args.drive_root, args.rebuild_cache)
    elif args.part == 'B':
        part_b(args.n_boot)
    elif args.part == 'B-rates':
        rates_by_window(recording_rates(), args.n_boot)
    elif args.part == 'C':
        part_c()
    elif args.part == 'D':
        part_d(args.n_boot)
    elif args.part in ('E', 'E-carry'):
        part_e(args.n_boot, carry_only=args.part == 'E-carry')
    elif args.part == 'export':
        export_traces()
        print(f'done in {time.time() - t0:.0f} s', flush=True)
        return
    print(tables(), flush=True)
    figures()
    print(f'done in {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
