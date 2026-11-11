#!/usr/bin/env python3
"""E81, the flies' variability across the three repeats: parts A and B (no simulations).

Yang's APL16 recordings hold three repeats of the 17 stimuli (45 s apart, about 13 min per repeat); olfdb's
`repeat` is taken as the repeat's number. Every statistic here is computed per recording from the single trials
(arrays repeats x KCs x odors, `kc_remapping_harness.yang_recording_trials`), averaged over recordings with equal
weight as the main target is, with 90 % intervals over 1000 resamples of the recordings (each genotype drawn with
replacement, seed 0). The repeat statistics themselves are harness functions (`distances_by_kind`,
`same_repeat_gap`, `noise_free_pair_distances`, `repeat_pair_table`, `repeat_response_stats`,
`repeat_variance_parts`, `coresponder_deviation_corr`), tested in test/test_e81_repeat_structure.py.

Part A, the flies' repeats (`--part A`), in Yang's z (the main target's units) and in baseline-noise units (the
realism bands' units), each read graded, with sub-threshold single-trial values set to 0 ('thresholded') and
binarized (above 1 z, or above the bands' noise_threshold in baseline-noise units):
- A1, the run-down (rundown.csv): per repeat, the single-trial response rate (a KC x odor single-trial value above
  the criterion: 1 and 1.5 z, and the bands' threshold in baseline-noise units), the responders' mean amplitude,
  the APL-off / APL-on response-rate ratio and the odors-per-KC distribution; per genotype, per odor and per
  recording, with APL output blocked (TNT) also for the six typical recordings and the two strongest (E77's
  STRONG_TNT) apart.
- A2, the drift and the shared state (distances_by_kind.csv): single-trial cosine distances by kind (same odor or
  two odors; same repeat, any two different repeats, adjacent repeats, first against third) for all pairs, the six
  pairs with near-identical ORN input (E78's SIX) and E77's kinds of pairs; the same-repeat gap (two odors of
  different repeats minus the same repeat), also per recording.
- A3, the parts of a response's variability (variance_parts.csv, coresponder_corr.csv): a gain per presentation
  (fitted from cross-repeat products), the part shared by one KC's odors within a repeat and the rest, for all
  entries, responders (trial mean above the criterion), silent entries and four amplitude bins of the responders;
  and, per pair, the correlation of the co-responders' deviations from their mean over repeats.
- A4, noise-free distances and the noise-free per-pair APL effect (noise_free_pairs.csv, noise_free_summary.csv):
  per recording and pair, one minus the mean single-trial cosine between different repeats over the geometric mean
  of the two odors' same-odor values, averaged over recordings; from adjacent repeats and from the first against
  the third as the check; the correction in use (noise_floor_trial_distances.py) and the trial means beside; the
  floor of the APL-effect loss and the ceiling of ρ on the per-pair APL effect by the resampling the present target
  uses (data_targets.apl_effect_resamples); the block, ρ on odor-odor distances between the two KC matrices and the
  isotonic R² of the on-against-off curve on each estimator's matrices.
- A5, the APL effect within each repeat (effect_by_repeat.csv): the control flies' single-trial distance minus that
  with APL output blocked (TNT), both genotypes' trials of the same repeat; its mean, the ester and alcohol groups,
  the six pairs, ρ and the projection on the error common to all wirings (E72's common error, model minus data,
  means removed; on the calibrated readout and on raw counts), the pattern's correlation between repeats, and the
  breadth-2 APL-effect fit's (C2, E67) loss and group errors against each repeat's effect.
- A6, the stimulus order: a stub (`--order-table`), until viki has the order of the 51 presentations.

Part B, the models against the noise-free effect and the saved trials (`--part B`):
- B1 (by_variant_noise_free.csv): every archived variant (each results/E*/model_pair_distances_long.csv, read as
  apl_effect_scores.py reads them) on raw counts, and under the calibrated readout where its spike counts are
  archived (E72's `distances(..., 'cal')`), against the flies' per-pair APL effect as measured and against the
  noise-free effect (Yang's z): loss, mean and pattern parts, ρ on the per-pair APL effect, each group's error by
  condition; the rank correlation of the two losses over variants; the populations (the breadth-2 family's four
  wirings through the pipeline, its eight, the firing-ceiling family's four, and the eight through the pipeline)
  with E77's split (`apl_effect_inherited_split`) against both.
- B2 (saved_trials_structure.csv): every model whose pipeline trials (E43_model_trial_noise.py part B) are in the
  model-output archive, per level and readout (graded counts, counts of 1 set to 0, two or more spikes): the
  same-repeat gap, the same-odor distances, the trial-mean, noise-free and corrected distances, the six pairs, the
  block, and the per-trial response rates; hal's noise_six_pairs.csv (E78) is read beside, not recomputed.
- B3 (jitter_at_flies_gap.csv, jitter_calibration_structure.csv): from the saved calibration trials (jitter 0.05 to
  0.3, three trials each), per model the jitter at which its same-repeat gap is the flies' (control flies with APL
  on, APL output blocked (TNT) with APL off), interpolated linearly in the jitter (sigma 0 gives gap 0), and the
  share of the flies' same-odor distance that jitter makes.

`--part table` rewrites this script's sections of comparison_table.md (between its markers, keeping the rest) from
the CSVs; `--part A` and `--part B` end with it. `registry_entries()` gives data_targets.py the new entries
(`repeat_*`, `noise_free_*`). Outputs: <shared>/results/E81_repeat_structure/ (the CSVs above, comparison_table.md,
E81_*.png, provenance_A.json / provenance_B.json, recorded at the start of each part).
Usage (light CPU): OMP_NUM_THREADS=2 nice -n 10 ../../venv/bin/python -u E81_repeat_structure.py --part A
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E81_repeat_structure'
ODORS: List[str] = list(H.APL16_ODORS)
GENOS = ('TNT3c', 'TNT')
GLAB = {'TNT3c': 'control flies', 'TNT': 'APL output blocked (TNT)'}
GKEY = {'TNT3c': 'ctrl', 'TNT': 'tnt'}
UNITS = ('z', 'noise')
N_BOOT = 1000
# E77's two recordings with APL output blocked (TNT) of the strongest responses (E76's open question)
STRONG_TNT = (('2024-07-17', 3, 'L'), ('2024-07-25', 1, 'L'))
ESTER = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL = ('1-8ol', '1-5ol', '1-6ol', '6al')
# E78's six pairs with an ORN distance below 0.25, and E77's kinds of ester pairs
SIX = (('eb', 'ep'), ('6al', 't2h'), ('IaA', 'pa'), ('2h', 'pa'), ('1-5ol', '1-6ol'), ('2h', 'IaA'))
NEAREST = (('2h', 'IaA'), ('2h', 'pa'), ('IaA', 'pa'), ('eb', 'ep'))
BLOCK = tuple((a, b) for a in ('2h', 'IaA', 'pa') for b in ('eb', 'ep'))
MS_PAIRS = tuple(('ms', o) for o in ('2h', 'IaA', 'pa', 'eb', 'ep'))
OPK_BINS = ('0', '1', '2_3', '4_7', '8plus')
ESTIMATORS = ('trial_mean', 'corrected', 'noise_free', 'noise_free_adjacent', 'noise_free_first_third')
# the KC data's same-odor single-trial distance (registry ctrl_ / tnt_single_trial_floor) and the bands' threshold
# are read from the registry where needed
COLORS = {'TNT3c': '#2a78d6', 'TNT': '#eb6834', 'c3': '#1baf7a', 'c4': '#eda100', 'ink': '#0b0b0b',
          'ink2': '#52514e', 'grid': '#e6e5e1'}
# the breadth-2 APL-effect fit on the real wiring (C2, E67) and the families' tags (E72)
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
TABLE_START = '<!-- E81_repeat_structure.py parts A and B: start -->'
TABLE_END = '<!-- E81_repeat_structure.py parts A and B: end -->'


def out_dir() -> Path:
    return H.results_dir(EXPERIMENT)


def canon(p: Tuple[str, str]) -> Tuple[str, str]:
    a, b = p
    return (a, b) if ODORS.index(a) < ODORS.index(b) else (b, a)


def _within(group: Sequence[str]) -> List[Tuple[str, str]]:
    return [canon((a, b)) for i, a in enumerate(group) for b in group[i + 1:]]


PAIRSETS: Dict[str, List[Tuple[str, str]]] = {
    'all': [tuple(p) for p in H.PAIRS16], 'six': [canon(p) for p in SIX], 'nearest': [canon(p) for p in NEAREST],
    'block': [canon(p) for p in BLOCK], 'ms_pairs': [canon(p) for p in MS_PAIRS], 'ester': _within(ESTER),
    'alcohol': _within(ALCOHOL)}


def resp_thresh(units: str) -> float:
    """The response criterion of a trial value: 1 z (Yang's) or the bands' threshold in baseline-noise units."""
    return H.KC_RESPONSE_Z if units == 'z' else float(H.data_target('noise_threshold'))


def rec_id(rec) -> str:
    return f'{rec[0]}_{rec[1]}{rec[2]}'


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_provenance(part: str, settings: dict, inputs: Sequence[Path]) -> None:
    """provenance_<part>.json, at the start of the part (the code that runs, not the checkout at the end)."""
    prov = dict(script='E81_repeat_structure.py', part=part, started=time.strftime('%Y-%m-%d %H:%M:%S%z'),
                settings=settings, inputs={str(p): _sha(p) for p in inputs if Path(p).exists()},
                code=H.model_provenance())
    (out_dir() / f'provenance_{part}.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')


def data_inputs() -> List[Path]:
    return [H.yang_data_dir() / g / f'dataset_{i}_long.csv' for g, i in H.YANG_DATASETS.items()] + [
        H.shared_dir() / 'data' / 'yang_apl16_kc_noise_units' / 'per_trial_long.parquet']


def boot(tables: Dict[str, pd.DataFrame], fn: Callable[[Dict[str, pd.Series]], dict], n_boot: int = N_BOOT,
         seed: int = 0) -> pd.DataFrame:
    """`fn` of the recording means of `tables` ({genotype: rows recordings}), with its 90 % interval over `n_boot`
    resamples of the recordings (each genotype with replacement). Returns DataFrame (value, lo, hi) by fn's keys."""
    rng = np.random.default_rng(seed)
    obs = fn({g: t.mean() for g, t in tables.items()})
    keys = list(obs)
    draws = np.full((n_boot, len(keys)), np.nan)
    for b in range(n_boot):
        means = {g: t.iloc[rng.integers(0, len(t), len(t))].mean() for g, t in tables.items()}
        d = fn(means)
        draws[b] = [d.get(k, np.nan) for k in keys]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        lo, hi = np.nanpercentile(draws, [5, 95], axis=0)
    return pd.DataFrame({'value': [obs[k] for k in keys], 'lo': lo, 'hi': hi},
                        index=pd.Index(keys, tupleize_cols=False, name='key'))


_PAIRSET_IDX = {k: pd.MultiIndex.from_tuples(v) for k, v in PAIRSETS.items()}


def pmean(s: pd.Series, name: str) -> float:
    """Mean over a pair set of a pair Series indexed (odor_a, odor_b) in canonical order."""
    return float(s.reindex(_PAIRSET_IDX[name]).mean())


# ---------------------------------------------------------------------------
# part A: per-recording tables
_TABLES: Dict[tuple, Dict[str, pd.DataFrame]] = {}


def pair_tables(units: str = 'z', values: str = 'graded') -> Dict[str, pd.DataFrame]:
    """{genotype: DataFrame} rows recordings (date, fly_num, hemibrain), columns (quantity, odor_a, odor_b): the
    pair distances of `H.repeat_pair_table` (two-odor pairs), 'same_<kind>' for the same-odor rows of
    `H.distances_by_kind` (odor_a == odor_b) and 'repeat<r>' (1-based) the single-trial distance of two odors within
    repeat r. Single trials read as `values` with the criterion of `units`. Cached."""
    key = (units, values)
    if key in _TABLES:
        return _TABLES[key]
    th = resp_thresh(units)
    pl = [tuple(p) for p in H.PAIRS16]
    idx = {o: i for i, o in enumerate(ODORS)}
    ia, ib = [idx[a] for a, _ in pl], [idx[b] for _, b in pl]
    out = {}
    for g in GENOS:
        rows = {}
        for rec, X in H.yang_recording_trials(g, units).items():
            Xv = H.trial_values(X, values, th)
            with np.errstate(invalid='ignore', divide='ignore'):
                tab = H.repeat_pair_table(Xv, ODORS)
                bk = H.distances_by_kind(Xv, ODORS)
                C = H.trial_cosines(Xv)
            parts = {q: tab[q] for q in tab.columns}
            same = bk[bk.index.get_level_values(0) == bk.index.get_level_values(1)]
            for kind in ('different_repeat', 'adjacent', 'first_third'):
                parts[f'same_{kind}'] = same[kind]
            for r in range(Xv.shape[0]):
                parts[f'repeat{r + 1}'] = pd.Series(1 - C[r, :, r, :][ia, ib], index=tab.index)
            rows[rec] = pd.concat(parts)
        df = pd.DataFrame(rows).T
        df.index = df.index.set_names(H.RECORDING_KEYS)
        out[g] = df
    _TABLES[key] = out
    return out


# ---------------------------------------------------------------------------
# A1, the run-down
def rundown_tables(units: str, thresh: float) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Per genotype, rows recordings: (summary columns: rate_r<r>, amp_r<r>, opk_<bin>_r<r>, rate_session (the
    single-trial rate over all three repeats), rate_trialmean (trial mean above `thresh`)), and (per-odor rate
    columns (odor, repeat))."""
    summ, per_odor = {}, {}
    for g in GENOS:
        srows, orows = {}, {}
        for rec, X in H.yang_recording_trials(g, units).items():
            st = H.repeat_response_stats(X, ODORS, thresh=thresh)
            R = X.shape[0]
            row = {}
            for r in range(R):
                s = st.xs(r, level='repeat')
                row[f'rate_r{r + 1}'] = float(s['rate'].mean())
                row[f'amp_r{r + 1}'] = float((s['amplitude'].fillna(0) * s['n_resp']).sum() / max(s['n_resp'].sum(), 1))
                n_k = (X[r] > thresh).sum(axis=1)
                fr = [np.mean(n_k == 0), np.mean(n_k == 1), np.mean((n_k >= 2) & (n_k <= 3)),
                      np.mean((n_k >= 4) & (n_k <= 7)), np.mean(n_k >= 8)]
                row.update({f'opk_{b}_r{r + 1}': float(v) for b, v in zip(OPK_BINS, fr)})
            row['rate_session'] = float(np.mean([row[f'rate_r{r + 1}'] for r in range(R)]))
            row['amp_session'] = float(np.nanmean([row[f'amp_r{r + 1}'] for r in range(R)]))
            row['rate_trialmean'] = float((X.mean(axis=0) > thresh).mean())
            srows[rec] = row
            orows[rec] = st['rate']                                                   # (repeat, odor)
        summ[g] = pd.DataFrame(srows).T
        per_odor[g] = pd.DataFrame(orows).T
    return summ, per_odor


def _rundown_fn(subset_tnt: str):
    def fn(m):
        out = {}
        c, t = m['TNT3c'], m['TNT']
        for gname, s in (('TNT3c', c), ('TNT', t)):
            for col in s.index:
                out[(gname, col)] = float(s[col])
            out[(gname, 'rate_change_r3_vs_r1')] = float(s['rate_r3'] / s['rate_r1'] - 1)
            out[(gname, 'amp_change_r3_vs_r1')] = float(s['amp_r3'] / s['amp_r1'] - 1)
        for r in ('r1', 'r2', 'r3', 'session'):
            out[('ratio', f'rate_{r}')] = float(t[f'rate_{r}'] / c[f'rate_{r}'])
        out[('ratio', 'rate_trialmean')] = float(t['rate_trialmean'] / c['rate_trialmean'])
        out[('ratio', 'r1_minus_session')] = out[('ratio', 'rate_r1')] - out[('ratio', 'rate_session')]
        out[('ratio', 'r3_minus_r1')] = out[('ratio', 'rate_r3')] - out[('ratio', 'rate_r1')]
        return out
    return fn


def part_a1(n_boot: int) -> pd.DataFrame:
    """A1: rundown.csv (long: units, criterion, threshold, genotype, subset, level, key, statistic, value, lo, hi)."""
    rows = []
    crits = [('z', '1 z', 1.0), ('z', '1.5 z', 1.5), ('noise', 'bands', resp_thresh('noise'))]
    for units, crit, th in crits:
        summ, per_odor = rundown_tables(units, th)
        tnt = summ['TNT']
        strong = [r for r in tnt.index if tuple(r) in STRONG_TNT]
        typical = [r for r in tnt.index if tuple(r) not in STRONG_TNT]
        for subset, rows_tnt in (('all', list(tnt.index)), ('typical6', typical), ('strong2', strong)):
            res = boot({'TNT3c': summ['TNT3c'], 'TNT': tnt.loc[rows_tnt]}, _rundown_fn(subset), n_boot=n_boot)
            for (gname, stat), r in res.iterrows():
                if subset != 'all' and gname == 'TNT3c':
                    continue
                rows.append(dict(units=units, criterion=crit, threshold=th, genotype=gname, subset=subset, level='genotype',
                                 key='all', statistic=stat, value=r.value, lo=r.lo, hi=r.hi))
        # per recording, and the number of recordings whose third repeat is below the first
        for g in GENOS:
            s = summ[g]
            for rec, rr in s.iterrows():
                for stat in ('rate_r1', 'rate_r2', 'rate_r3', 'amp_r1', 'amp_r2', 'amp_r3', 'rate_session', 'rate_trialmean'):
                    rows.append(dict(units=units, criterion=crit, threshold=th, genotype=g, subset='typical6' if
                                     (g == 'TNT' and tuple(rec) not in STRONG_TNT) else ('strong2' if g == 'TNT' else 'all'),
                                     level='recording', key=rec_id(rec), statistic=stat, value=float(rr[stat])))
                rows.append(dict(units=units, criterion=crit, threshold=th, genotype=g, subset='', level='recording', key=rec_id(rec),
                                 statistic='rate_change_r3_vs_r1', value=float(rr['rate_r3'] / rr['rate_r1'] - 1)))
            rows.append(dict(units=units, criterion=crit, threshold=th, genotype=g, subset='all', level='genotype', key='all',
                             statistic='n_recordings_r3_below_r1', value=float((s['rate_r3'] < s['rate_r1']).sum())))
            rows.append(dict(units=units, criterion=crit, threshold=th, genotype=g, subset='all', level='genotype', key='all',
                             statistic='n_recordings', value=float(len(s))))
            # per odor (mean over recordings)
            po = per_odor[g].mean()
            for (r, o), v in po.items():
                rows.append(dict(units=units, criterion=crit, threshold=th, genotype=g, subset='all', level='odor', key=o,
                                 statistic=f'rate_r{int(r) + 1}', value=float(v)))
        poc, pot = per_odor['TNT3c'].mean(), per_odor['TNT'].mean()
        for (r, o), v in (pot / poc).items():
            rows.append(dict(units=units, criterion=crit, threshold=th, genotype='ratio', subset='all', level='odor', key=o,
                             statistic=f'rate_r{int(r) + 1}', value=float(v)))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / 'rundown.csv', index=False)
    return df


# ---------------------------------------------------------------------------
# A2, the drift and the shared state
def _kinds_fn(m):
    out = {}
    for g, s in m.items():
        for ps in PAIRSETS:
            for kind in H.REPEAT_KINDS:
                out[(g, ps, kind)] = pmean(s[kind], ps)
            out[(g, ps, 'gap')] = out[(g, ps, 'different_repeat')] - out[(g, ps, 'same_repeat')]
            out[(g, ps, 'gap_adjacent')] = out[(g, ps, 'adjacent')] - out[(g, ps, 'same_repeat')]
            out[(g, ps, 'gap_first_third')] = out[(g, ps, 'first_third')] - out[(g, ps, 'same_repeat')]
        for kind in ('different_repeat', 'adjacent', 'first_third'):
            out[(g, 'same_odor', kind)] = float(s[f'same_{kind}'].mean())
    for ps in PAIRSETS:
        out[('ratio', ps, 'gap')] = out[('TNT', ps, 'gap')] / out[('TNT3c', ps, 'gap')]
    return out


def part_a2(n_boot: int) -> pd.DataFrame:
    """A2: distances_by_kind.csv (units, values, genotype, pairset, kind, value, lo, hi; level 'genotype' or
    'recording' for the per-recording gaps)."""
    rows = []
    for units in UNITS:
        for values in H.TRIAL_VALUES:
            T = pair_tables(units, values)
            res = boot(T, _kinds_fn, n_boot=n_boot)
            for (g, ps, kind), r in res.iterrows():
                rows.append(dict(units=units, values=values, genotype=g, level='genotype', recording='', pairset=ps, kind=kind,
                                 value=r.value, lo=r.lo, hi=r.hi))
            for g in GENOS:
                for rec, s in T[g].iterrows():
                    for ps in ('all', 'six'):
                        gap = pmean(s['different_repeat'], ps) - pmean(s['same_repeat'], ps)
                        rows.append(dict(units=units, values=values, genotype=g, level='recording', recording=rec_id(rec),
                                         pairset=ps, kind='gap', value=gap))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / 'distances_by_kind.csv', index=False)
    return df


# ---------------------------------------------------------------------------
# A3, the parts of the variability
VP_COLS = ('n_entries', 'n_pairs', 'mean_amplitude', 'var_total', 'var_gain', 'var_shared', 'sd_trial')


def part_a3(n_boot: int) -> pd.DataFrame:
    """A3: variance_parts.csv (per units, genotype and class: the recording means of the variances and the shares
    and correlation computed from them, with intervals; level 'recording' rows per recording) and
    coresponder_corr.csv (per units, genotype and pair: the co-responders' deviation correlation, mean over
    recordings, with and without the gain part)."""
    rows, crows = [], []
    for units in UNITS:
        th = resp_thresh(units)
        tabs = {}
        for g in GENOS:
            vrows, crs, crg = {}, {}, {}
            for rec, X in H.yang_recording_trials(g, units).items():
                vp = H.repeat_variance_parts(X, ODORS, thresh=th).astype(float)
                vrows[rec] = vp[list(VP_COLS)].stack()
                crs[rec] = H.coresponder_deviation_corr(X, ODORS, thresh=th)
                crg[rec] = H.coresponder_deviation_corr(X, ODORS, thresh=th, remove_gain=True)
                for cls, r in vp.iterrows():
                    rows.append(dict(units=units, genotype=g, level='recording', recording=rec_id(rec), cls=cls,
                                     **{k: r.get(k, np.nan) for k in ('n_entries', 'var_total', 'var_gain', 'var_shared',
                                                                      'share_gain', 'share_shared', 'share_rest', 'sd_trial',
                                                                      'corr_shared')}))
            tabs[g] = pd.DataFrame(vrows).T
            c, cg = pd.DataFrame(crs).mean(axis=1), pd.DataFrame(crg).mean(axis=1)
            for (a, b), v in c.items():
                crows.append(dict(units=units, genotype=g, odor_a=a, odor_b=b, corr=v, corr_gain_removed=cg[(a, b)],
                                  in_six=canon((a, b)) in PAIRSETS['six'],
                                  n_recordings=int(pd.DataFrame(crs).loc[(a, b)].notna().sum())))

        def fn(m):
            out = {}
            for g, s in m.items():
                for cls in s.index.get_level_values(0).unique():
                    vt, vg, vs = s[(cls, 'var_total')], s[(cls, 'var_gain')], s[(cls, 'var_shared')]
                    out[(g, cls, 'var_total')] = vt
                    out[(g, cls, 'var_gain')] = vg
                    out[(g, cls, 'var_shared')] = vs
                    out[(g, cls, 'share_gain')] = vg / vt
                    out[(g, cls, 'share_shared')] = vs / vt
                    out[(g, cls, 'share_rest')] = 1 - (vg + vs) / vt
                    out[(g, cls, 'corr_shared')] = vs / (vt - vg)
                    out[(g, cls, 'sd_trial')] = s[(cls, 'sd_trial')]
                    out[(g, cls, 'mean_amplitude')] = s[(cls, 'mean_amplitude')]
                    out[(g, cls, 'n_entries')] = s[(cls, 'n_entries')]
            return out
        res = boot(tabs, fn, n_boot=n_boot)
        for (g, cls, stat), r in res.iterrows():
            rows.append(dict(units=units, genotype=g, level='genotype', recording='', cls=cls, statistic=stat,
                             value=r.value, lo=r.lo, hi=r.hi))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / 'variance_parts.csv', index=False)
    pd.DataFrame(crows).to_csv(out_dir() / 'coresponder_corr.csv', index=False)
    return df


# ---------------------------------------------------------------------------
# A4, noise-free distances and the noise-free effect
def _on_off(on: pd.Series, off: pd.Series) -> dict:
    st, _ = H.compare_on_off(on, off)
    return dict(block=float(st['block_fruity_small_iso']), rho_ctrl_tnt=float(st['spearman_on_off']),
                iso_r2=float(st['iso_r2']))


def _a4_fn(with_onoff: bool = True, per_pair: bool = False):
    def fn(m):
        out = {}
        eff = {}
        for est in ESTIMATORS:
            c, t = m['TNT3c'][est], m['TNT'][est]
            e = c - t
            eff[est] = e
            out[('ctrl', est, 'mean_dist')] = float(c.mean())
            out[('tnt', est, 'mean_dist')] = float(t.mean())
            out[('effect', est, 'mean')] = float(e.mean())
            for ps in ('six', 'ester', 'alcohol', 'block', 'nearest'):
                out[('ctrl', est, f'{ps}_mean')] = pmean(c, ps)
                out[('tnt', est, f'{ps}_mean')] = pmean(t, ps)
                out[('effect', est, f'{ps}_mean')] = pmean(e, ps)
            if with_onoff and est in ('trial_mean', 'corrected', 'noise_free'):
                for k, v in _on_off(c, t).items():
                    out[('both', est, k)] = v
            if per_pair and est in ('trial_mean', 'corrected', 'noise_free'):
                for p in c.index:
                    out[('pair_ctrl', est, p)] = float(c[p])
                    out[('pair_tnt', est, p)] = float(t[p])
                    out[('pair_effect', est, p)] = float(e[p])
        for g, s in (('ctrl', m['TNT3c']), ('tnt', m['TNT'])):
            out[(g, 'inflation', 'mean')] = float((s['trial_mean'] - s['noise_free']).mean())
            out[(g, 'inflation', 'six_mean')] = pmean(s['trial_mean'] - s['noise_free'], 'six')
            out[(g, 'corrected_minus_noise_free', 'mean')] = float((s['corrected'] - s['noise_free']).mean())
            out[(g, 'adjacent_minus_first_third', 'mean')] = float((s['noise_free_adjacent'] - s['noise_free_first_third']).mean())
        out[('effect', 'noise_free', 'spearman_with_measured')] = H.spearman(eff['noise_free'], eff['trial_mean'])
        out[('effect', 'noise_free', 'pearson_with_measured')] = H.pearson(eff['noise_free'], eff['trial_mean'])
        out[('effect', 'noise_free', 'rms_diff_from_measured')] = float(np.sqrt(((eff['noise_free'] - eff['trial_mean']) ** 2).mean()))
        out[('effect', 'noise_free', 'spearman_with_corrected')] = H.spearman(eff['noise_free'], eff['corrected'])
        return out
    return fn


def recording_distances(units: str = 'z', est: str = 'noise_free', values: str = 'graded') -> Dict[str, pd.DataFrame]:
    """{genotype: rows recordings, columns pairs} of one estimator's per-recording distances (`pair_tables`)."""
    T = pair_tables(units, values)
    return {g: T[g][est] for g in GENOS}


def noise_free_data(units: str = 'z') -> Dict[str, pd.Series]:
    """The flies' noise-free mean pair distances ('on': control flies, 'off': APL output blocked (TNT)) and their
    per-pair APL effect ('effect'), canonical pair order, graded."""
    r = recording_distances(units, 'noise_free')
    on, off = r['TNT3c'].mean(), r['TNT'].mean()
    on.index.names = off.index.names = ['odor_a', 'odor_b']
    return dict(on=on, off=off, effect=H.apl_effect(on, off))


def part_a4(n_boot: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A4: noise_free_pairs.csv (per units and pair: each estimator's distances and effect, the noise-free ones with
    intervals) and noise_free_summary.csv (scalars with intervals: means, groups, six pairs, the block, ρ between the
    two KC matrices, the isotonic R², the floor of the loss and the ceiling of ρ on the per-pair APL effect)."""
    import data_targets as DT
    srows, prows = [], []
    for units in UNITS:
        for values in (('graded', 'thresholded', 'binary') if units == 'z' else ('graded',)):
            T = pair_tables(units, values)
            res = boot(T, _a4_fn(with_onoff=(values == 'graded'), per_pair=(values == 'graded')), n_boot=n_boot)
            for key, r in res.iterrows():
                if key[0].startswith('pair_'):
                    continue
                srows.append(dict(units=units, values=values, side=key[0], estimator=key[1], statistic=key[2],
                                  value=r.value, lo=r.lo, hi=r.hi))
            if values != 'graded':
                continue
            # floors and ceilings as the present target's (data_targets.apl_effect_resamples), per estimator
            for est in ('trial_mean', 'corrected', 'noise_free', 'noise_free_adjacent', 'noise_free_first_third'):
                recs = {g: T[g][est] for g in GENOS}
                effect, boot_mean, boot_rmse, sh_rho = DT.apl_effect_resamples(recs)
                r_sh = float(np.nanmean(sh_rho))
                srows += [dict(units=units, values=values, side='effect', estimator=est, statistic='rmse_floor',
                               value=float(np.mean(boot_rmse)), lo=float(np.percentile(boot_rmse, 5)), hi=float(np.percentile(boot_rmse, 95))),
                          dict(units=units, values=values, side='effect', estimator=est, statistic='rho_split_half',
                               value=r_sh, lo=float(np.nanpercentile(sh_rho, 5)), hi=float(np.nanpercentile(sh_rho, 95))),
                          dict(units=units, values=values, side='effect', estimator=est, statistic='rho_ceiling',
                               value=float(np.sqrt(2 * r_sh / (1 + r_sh))), lo=np.nan, hi=np.nan)]
            # per pair
            M = {g: T[g].mean() for g in GENOS}
            for p in [tuple(x) for x in H.PAIRS16]:
                row = dict(units=units, odor_a=p[0], odor_b=p[1], in_six=p in PAIRSETS['six'],
                           in_ester=p in PAIRSETS['ester'], in_alcohol=p in PAIRSETS['alcohol'])
                for est in ESTIMATORS:
                    c, t = float(M['TNT3c'][(est,) + p]), float(M['TNT'][(est,) + p])
                    row[f'ctrl_{est}'], row[f'tnt_{est}'], row[f'effect_{est}'] = c, t, c - t
                for est in ('trial_mean', 'corrected', 'noise_free'):
                    for side, col in (('pair_ctrl', 'ctrl'), ('pair_tnt', 'tnt'), ('pair_effect', 'effect')):
                        r = res.loc[[(side, est, p)]].iloc[0]
                        row[f'{col}_{est}_lo'], row[f'{col}_{est}_hi'] = float(r.lo), float(r.hi)
                prows.append(row)
    summ, pairs = pd.DataFrame(srows), pd.DataFrame(prows)
    summ.to_csv(out_dir() / 'noise_free_summary.csv', index=False)
    pairs.to_csv(out_dir() / 'noise_free_pairs.csv', index=False)
    return summ, pairs


# ---------------------------------------------------------------------------
# A5, the APL effect within each repeat
def common_errors() -> Dict[str, pd.Series]:
    """E72's per-pair error common to the eight wirings (model minus data, means removed), canonical order, on the
    calibrated readout ('cal') and on raw counts ('raw')."""
    out = {}
    for ro, name in (('cal', 'common_error_pairs.csv'), ('raw', 'common_error_pairs_raw.csv')):
        d = pd.read_csv(H.results_dir('E72_wiring_population') / name)
        out[ro] = H.canonical_pairs(d.set_index(['odor_a', 'odor_b'])['common'])
    return out


def model_pair_distances(folder: str, tag: str) -> Tuple[pd.Series, pd.Series]:
    """(APL on, acute APL off) raw-count cosine distances of one archived variant (apl_effect_scores.model_pairs)."""
    import apl_effect_scores as AES
    return AES.model_pairs(H.results_dir(folder))[tag]


def part_a5(n_boot: int) -> pd.DataFrame:
    """A5: effect_by_repeat.csv (statistic, repeat ('1', '2', '3', 'session' for the trial means, 'single' for the
    single-trial same-repeat distances of all repeats), value, lo, hi)."""
    T = pair_tables('z', 'graded')
    ce = common_errors()
    c2_on, c2_off = model_pair_distances(*C2)
    c2_on, c2_off = H.canonical_pairs(c2_on), H.canonical_pairs(c2_off)
    c2_eff = H.apl_effect(c2_on, c2_off)

    def fn(m):
        out = {}
        effs = {}
        for rep, q in (('1', 'repeat1'), ('2', 'repeat2'), ('3', 'repeat3'), ('session', 'trial_mean'),
                       ('single', 'same_repeat')):
            e = m['TNT3c'][q] - m['TNT'][q]
            effs[rep] = e
            out[('mean', rep)] = float(e.mean())
            for ps in ('ester', 'alcohol', 'six', 'block', 'nearest'):
                out[(f'{ps}_mean', rep)] = pmean(e, ps)
            out[('ctrl_mean_dist', rep)] = float(m['TNT3c'][q].mean())
            out[('tnt_mean_dist', rep)] = float(m['TNT'][q].mean())
            ec = e - e.mean()
            for ro, c in ce.items():
                cc = c.reindex(e.index)
                out[(f'rho_common_{ro}', rep)] = H.spearman(e, cc)
                out[(f'proj_common_{ro}', rep)] = float((ec.to_numpy() @ cc.to_numpy()) / (cc.to_numpy() @ cc.to_numpy()))
            st = H.apl_effect_stats(c2_on, c2_off, e)
            out[('c2_loss', rep)] = st['apl_effect_rmse']
            out[('c2_pattern', rep)] = st['apl_effect_rmse_centered']
            out[('c2_mean_error', rep)] = st['apl_effect_mean_error']
            out[('c2_rho', rep)] = st['apl_effect_spearman']
            err = c2_eff.reindex(e.index) - e
            errc = err - err.mean()
            for ps in ('ester', 'alcohol', 'six'):
                out[(f'c2_{ps}_error', rep)] = pmean(errc, ps)
        for a, b in (('1', '2'), ('1', '3'), ('2', '3'), ('1', 'session'), ('3', 'session')):
            out[(f'pearson_with_{b}', a)] = H.pearson(effs[a], effs[b])
            out[(f'spearman_with_{b}', a)] = H.spearman(effs[a], effs[b])
        for ro in ce:
            out[(f'proj_common_{ro}_ratio_3_over_1', 'all')] = out[(f'proj_common_{ro}', '3')] / out[(f'proj_common_{ro}', '1')]
        out[('mean_change_3_minus_1', 'all')] = out[('mean', '3')] - out[('mean', '1')]
        return out
    res = boot(T, fn, n_boot=n_boot)
    df = pd.DataFrame([dict(statistic=k[0], repeat=k[1], value=r.value, lo=r.lo, hi=r.hi) for k, r in res.iterrows()])
    df.to_csv(out_dir() / 'effect_by_repeat.csv', index=False)
    return df


# ---------------------------------------------------------------------------
# A6, the stimulus order (stub)
def part_a6(order_table: Optional[Path]) -> None:
    """A6, the stimulus order: with a table of each recording's 51 presentations in the order given (viki, from the
    recordings' YAML files; Yang's to release): that olfdb's `repeat` is the repeat's number, the same-repeat
    distances against the time between two presentations, and whether the order is the same in every recording (if
    so, a pair's place in the order is confounded with the pair). Not written yet: the table's layout is not known.
    Expected columns: date, fly_num, hemibrain, presentation (0-50, in time order), odor (`<abbrev> @ <conc>`),
    repeat (olfdb's)."""
    if order_table is None:
        print('A6: no stimulus-order table; part A runs without it')
        return
    raise NotImplementedError('A6 (the stimulus order) is a stub until the order table exists')


# ---------------------------------------------------------------------------
# part B
def _err_by_condition(m_on: pd.Series, m_off: pd.Series, d_on: pd.Series, d_off: pd.Series) -> dict:
    """Each group's pattern error of the per-pair APL effect split by condition: the group mean of (model minus data)
    with APL on, minus the mean over all pairs of the same, likewise with APL off; the effect's group error is
    on minus off."""
    eon = (H.canonical_pairs(m_on) - H.canonical_pairs(d_on)).dropna()
    eoff = (H.canonical_pairs(m_off) - H.canonical_pairs(d_off)).dropna()
    out = {}
    for ps in ('ester', 'alcohol', 'six'):
        a, b = pmean(eon, ps) - float(eon.mean()), pmean(eoff, ps) - float(eoff.mean())
        out[f'{ps}_err_on'], out[f'{ps}_err_off'], out[f'{ps}_err'] = a, b, a - b
    out['mean_err_on'], out['mean_err_off'] = float(eon.mean()), float(eoff.mean())
    return out


def score_against(on: pd.Series, off: pd.Series, targets: Dict[str, dict], prefix: str) -> dict:
    """APL-effect statistics of one model against each target ({name: dict(on, off, effect)})."""
    row = {}
    for tname, t in targets.items():
        st = H.apl_effect_stats(on, off, t['effect'])
        p = f'{prefix}_{tname}'
        row.update({f'{p}_loss': st['apl_effect_rmse'], f'{p}_pattern': st['apl_effect_rmse_centered'],
                    f'{p}_mean_error': st['apl_effect_mean_error'], f'{p}_rho': st['apl_effect_spearman']})
        row.update({f'{p}_{k}': v for k, v in _err_by_condition(on, off, t['on'], t['off']).items()})
    row[f'{prefix}_effect_mean'] = float(H.apl_effect(on, off).mean())
    row[f'{prefix}_mean_dist_on'] = float(H.canonical_pairs(on).mean())
    row[f'{prefix}_mean_dist_off'] = float(H.canonical_pairs(off).mean())
    return row


def targets_z() -> Dict[str, dict]:
    on, off = H.canonical_pairs(H.yang_mean_rdm('TNT3c')), H.canonical_pairs(H.yang_mean_rdm('TNT'))
    nf = noise_free_data('z')
    return {'measured': dict(on=on, off=off, effect=H.apl_effect(on, off)), 'noise_free': nf}


def populations() -> Dict[str, List[Tuple[str, str, str]]]:
    """The populations scored in B1: [(label, folder, tag)]."""
    import E72_wiring_population as E72
    b8 = E72.new_fit_members()
    b4 = [m for m in b8 if m[0] in (E72.REAL, 'P1', 'P2', 'M1')]
    g4 = E72.new_fit_members(prefix_tag=E72.G_SUFFIX, real_folder=E72.G_REAL[0], real_tag=E72.G_REAL[1], draws=E72.G_DRAWS)
    return {'breadth2_four': b4, 'breadth2_eight': b8, 'ceiling_four': g4, 'pipeline_eight': b4 + g4}


def part_b1(cal: bool = True) -> pd.DataFrame:
    """B1: by_variant_noise_free.csv, one row per archived variant (folder, tag) and per population."""
    import apl_effect_scores as AES
    import E72_wiring_population as E72
    tg = targets_z()
    rows = []
    folders = sorted(p for p in (H.shared_dir() / 'results').glob('E*_*') if p.is_dir() and re.match(r'E\d+_', p.name))
    t0 = time.time()
    for i, folder in enumerate(folders):
        pairs = AES.model_pairs(folder)
        for tag, (on, off) in pairs.items():
            on, off = H.canonical_pairs(on), H.canonical_pairs(off)
            if len(H.align_pairs(on, off)[0]) < 100:
                continue
            row = dict(folder=folder.name, tag=tag, kind='variant', **score_against(on, off, tg, 'raw'))
            if cal:
                try:
                    o = E72.load_fit(folder.name, tag)
                except Exception as e:
                    o = None
                    print(f'{folder.name}:{tag}: {e!r}'[:200], flush=True)
                if o is not None:
                    try:
                        d = E72.distances(o, 'cal')
                        row.update(score_against(d['on'], d['off'], tg, 'cal'))
                    except Exception as e:
                        print(f'{folder.name}:{tag} (cal): {e!r}'[:200], flush=True)
            rows.append(row)
        print(f'B1 {i + 1}/{len(folders)} {folder.name}: {len(rows)} variants, {time.time() - t0:.0f} s', flush=True)
    # the populations: the mean over wirings of the pair distances per condition
    for pname, members in populations().items():
        fits = [E72.load_fit(f, t) for _, f, t in members]
        if any(o is None for o in fits):
            print(f'B1: population {pname} has members without saved outputs; skipped', flush=True)
            continue
        row = dict(folder='population', tag=pname, kind='population', members=';'.join(f'{f}:{t}' for _, f, t in members))
        for ro in ('raw', 'cal'):
            md = E72.mean_distances([E72.distances(o, ro) for o in fits])
            on, off = H.canonical_pairs(md['on']), H.canonical_pairs(md['off'])
            row.update(score_against(on, off, tg, ro))
            for tname, t in tg.items():
                s = H.apl_effect_inherited_split(on, off, t['on'], t['off'])
                row.update({f'{ro}_{tname}_split_{k}': v for k, v in H.inherited_shares(s).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    # the present scores (apl_effect_scores.py's by_variant.csv) beside, as the check that raw_measured is them
    bv = H.shared_dir() / 'results' / 'apl_effect' / 'by_variant.csv'
    if bv.exists():
        b = pd.read_csv(bv)[['folder', 'tag', 'spike_counts_cosine__apl_effect_rmse', 'realism_flag']]
        b = b.rename(columns={'spike_counts_cosine__apl_effect_rmse': 'present_raw_loss'})
        b['tag'] = b['tag'].astype(str)
        df['tag'] = df['tag'].astype(str)
        df = df.merge(b, on=['folder', 'tag'], how='left')
    df.to_csv(out_dir() / 'by_variant_noise_free.csv', index=False)
    return df


TRIAL_FILE = re.compile(r'^(?P<model>.+)_sigma(?P<sigma>[0-9.e-]+)_n(?P<n>\d+)(?:_rep(?P<rep>\d+))?\.npz$')
COUNT_READOUTS = (('graded', 'graded'), ('thresholded', 'thresholded'), ('binary', 'binary'))


def _load_npz(p: Path) -> dict:
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def trial_set_stats(arr_list: Sequence[dict], cond: str, values: str) -> dict:
    """Part A's statistics for one condition of a model's pseudo-recordings (each a dict with '<cond>_spk' (trials,
    KCs, odors) and '<cond>_resp'), the counts read as `values` with a response at two or more spikes (thresh 1):
    means over pseudo-recordings of the pair tables (canonical pairs) and the same-odor kinds, and the per-trial
    response rate."""
    tabs, same, gaps, gaps6, rates = [], [], [], [], []
    for a in arr_list:
        X = H.trial_values(a[f'{cond}_spk'].astype(float), values, 1.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            tab = H.repeat_pair_table(X, ODORS)
            bk = H.distances_by_kind(X, ODORS)
        tabs.append(tab)
        s = bk[bk.index.get_level_values(0) == bk.index.get_level_values(1)]
        same.append(s[['different_repeat', 'adjacent', 'first_third']].mean())
        gaps.append(H.same_repeat_gap(bk))
        gaps6.append(H.same_repeat_gap(bk, PAIRSETS['six']))
        rates.append(float(np.mean(a[f'{cond}_resp'])))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        tab = pd.concat(tabs).groupby(level=[0, 1], sort=False).mean()
    same = pd.concat(same, axis=1).mean(axis=1)
    return dict(tab=tab, same_odor=float(same['different_repeat']), same_odor_adjacent=float(same['adjacent']),
                same_odor_first_third=float(same['first_third']), gap=float(np.nanmean(gaps)), gap_six=float(np.nanmean(gaps6)),
                rate=float(np.mean(rates)), n_recordings=len(arr_list))


def _levels_for(folder: str) -> Dict[str, Dict[float, str]]:
    """{model: {sigma: level label(s)}} from the experiment's partB/noise_levels.json."""
    p = H.results_dir(folder) / 'partB' / 'noise_levels.json'
    out = {}
    if p.exists():
        for model, d in json.loads(p.read_text()).items():
            lv = {}
            for lab, s in (d.get('levels') or {}).items():
                lv.setdefault(round(float(s), 6), []).append(lab)
            out[model] = {s: '+'.join(v) for s, v in lv.items()}
    return out


def part_b2() -> pd.DataFrame:
    """B2: saved_trials_structure.csv, one row per archive folder, model, jitter and readout."""
    tg = targets_z()
    archive = H.shared_dir() / 'data' / 'model_outputs'
    rows = []
    for tdir in sorted(archive.glob('E*/partB/trials')):
        folder = tdir.parent.parent.name
        levels = _levels_for(folder)
        groups: Dict[Tuple[str, float], List[Path]] = {}
        for p in sorted(tdir.glob('*.npz')):
            m = TRIAL_FILE.match(p.name)
            if m and m.group('rep') is not None:
                groups.setdefault((m.group('model'), float(m.group('sigma'))), []).append(p)
        for (model, sigma), paths in sorted(groups.items()):
            arrs = [_load_npz(p) for p in paths]
            for values, _ in COUNT_READOUTS:
                st = {c: trial_set_stats(arrs, c, values) for c in ('on', 'acute_off')}
                row = dict(folder=folder, model=model, sigma=sigma, level=levels.get(model, {}).get(round(sigma, 6), ''),
                           values=values, n_recordings=st['on']['n_recordings'],
                           fixed_thr=float(arrs[0].get('fixed_thr', np.nan)))
                for c, cs in (('on', 'on'), ('acute_off', 'off')):
                    s, tab = st[c], st[c]['tab']
                    row.update({f'{k}_{cs}': s[k] for k in ('same_odor', 'same_odor_adjacent', 'same_odor_first_third',
                                                            'gap', 'gap_six', 'rate')})
                    for est in ('trial_mean', 'corrected', 'noise_free', 'noise_free_adjacent', 'noise_free_first_third',
                                'same_repeat', 'different_repeat'):
                        row[f'{est}_{cs}'] = float(tab[est].mean())
                        row[f'six_{est}_{cs}'] = pmean(tab[est], 'six')
                    row[f'inflation_{cs}'] = row[f'trial_mean_{cs}'] - row[f'noise_free_{cs}']
                    row[f'six_inflation_{cs}'] = row[f'six_trial_mean_{cs}'] - row[f'six_noise_free_{cs}']
                row['gap_ratio_off_on'] = row['gap_off'] / row['gap_on'] if row['gap_on'] else np.nan
                ton, toff = st['on']['tab'], st['acute_off']['tab']
                for est in ('trial_mean', 'corrected', 'noise_free'):
                    on, off = ton[est], toff[est]
                    try:
                        row.update({f'{k}_{est}': v for k, v in _on_off(on, off).items()})
                    except Exception:
                        pass
                    e = H.apl_effect(on, off)
                    row[f'effect_mean_{est}'] = float(e.mean())
                    row[f'six_effect_{est}'] = pmean(e, 'six')
                # like for like: the model's trial means against the flies' trial means (the target as measured), its
                # noise-free distances against the flies' noise-free distances
                for est, tname in (('trial_mean', 'measured'), ('noise_free', 'noise_free'), ('corrected', 'noise_free')):
                    stt = H.apl_effect_stats(ton[est], toff[est], tg[tname]['effect'])
                    p = f'{est}_vs_{tname}'
                    row.update({f'{p}_loss': stt['apl_effect_rmse'], f'{p}_pattern': stt['apl_effect_rmse_centered'],
                                f'{p}_rho': stt['apl_effect_spearman']})
                rows.append(row)
            print(f'B2 {folder} {model} sigma {sigma}: {len(paths)} pseudo-recordings', flush=True)
    df = pd.DataFrame(rows)
    # hal's six-pairs table (E78), read beside: its six_raw (the trial means) must be this table's six_trial_mean
    e78 = H.results_dir('E78_claw_knee') / 'noise_six_pairs.csv'
    if e78.exists() and len(df):
        n6 = pd.read_csv(e78)
        n6 = n6[['folder', 'model', 'level', 'transform'] + [c for c in n6.columns if c.startswith('six_')]]
        n6 = n6.rename(columns={'transform': 'values', **{c: f'e78_{c}' for c in n6.columns if c.startswith('six_')}})
        df = df.merge(n6, on=['folder', 'model', 'level', 'values'], how='left')
    df.to_csv(out_dir() / 'saved_trials_structure.csv', index=False)
    return df


def _interp_cross(x: np.ndarray, y: np.ndarray, target: float) -> float:
    """The first x at which y (sorted by x) reaches `target`, linearly interpolated; NaN if never."""
    for i in range(1, len(x)):
        if (y[i - 1] - target) * (y[i] - target) <= 0 and y[i] != y[i - 1]:
            return float(x[i - 1] + (target - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return np.nan


def part_b3() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """B3: jitter_calibration_structure.csv (per folder, model, jitter and condition: the same-repeat gap and the
    same-odor distance of the calibration trials, graded) and jitter_at_flies_gap.csv (per folder and model: the
    jitter at which the gap is the flies', with APL on against the control flies and with APL off against APL output
    blocked (TNT), the same-odor distance there and its share of the flies')."""
    archive = H.shared_dir() / 'data' / 'model_outputs'
    T = pair_tables('z', 'graded')
    flies = {}
    for g, c in (('TNT3c', 'on'), ('TNT', 'off')):
        m = T[g].mean()
        flies[c] = dict(gap=pmean(m['different_repeat'], 'all') - pmean(m['same_repeat'], 'all'),
                        same_odor=float(m['same_different_repeat'].mean()))
    rows = []
    for cdir in sorted(archive.glob('E*/partB/calibration')):
        folder = cdir.parent.parent.name
        for p in sorted(cdir.glob('*.npz')):
            m = TRIAL_FILE.match(p.name)
            if not m or m.group('rep') is not None:
                continue
            a = _load_npz(p)
            for cond, cs in (('on', 'on'), ('acute_off', 'off')):
                if f'{cond}_spk' not in a:
                    continue
                st = trial_set_stats([a], cond, 'graded')
                rows.append(dict(folder=folder, model=m.group('model'), sigma=float(m.group('sigma')), condition=cs,
                                 gap=st['gap'], gap_six=st['gap_six'], same_odor=st['same_odor'], rate=st['rate'],
                                 trial_mean=float(st['tab']['trial_mean'].mean()),
                                 noise_free=float(st['tab']['noise_free'].mean())))
    cal = pd.DataFrame(rows).sort_values(['folder', 'model', 'condition', 'sigma'])
    cal.to_csv(out_dir() / 'jitter_calibration_structure.csv', index=False)
    srows = []
    for (folder, model), g in cal.groupby(['folder', 'model']):
        lv = json.loads((H.results_dir(folder) / 'partB' / 'noise_levels.json').read_text()).get(model, {}) \
            if (H.results_dir(folder) / 'partB' / 'noise_levels.json').exists() else {}
        row = dict(folder=folder, model=model)
        for cs in ('on', 'off'):
            gc = g[g.condition == cs].drop_duplicates('sigma').sort_values('sigma')
            x = np.r_[0.0, gc.sigma.to_numpy()]
            gap = np.r_[0.0, gc.gap.to_numpy()]
            so = np.r_[0.0, gc.same_odor.to_numpy()]
            s_gap = _interp_cross(x, gap, flies[cs]['gap'])
            so_at = float(np.interp(s_gap, x, so)) if np.isfinite(s_gap) else np.nan
            matched = ((lv.get('matched') or {}).get('on' if cs == 'on' else 'acute_off') or {}).get('sd', np.nan)
            row.update({f'sigma_at_flies_gap_{cs}': s_gap, f'same_odor_at_flies_gap_{cs}': so_at,
                        f'same_odor_share_{cs}': so_at / flies[cs]['same_odor'] if np.isfinite(so_at) else np.nan,
                        f'sigma_matched_{cs}': matched,
                        f'gap_at_matched_{cs}': float(np.interp(matched, x, gap)) if np.isfinite(matched) else np.nan,
                        f'flies_gap_{cs}': flies[cs]['gap'], f'flies_same_odor_{cs}': flies[cs]['same_odor'],
                        f'n_sigmas_{cs}': len(gc), f'sigma_max_{cs}': float(gc.sigma.max()) if len(gc) else np.nan})
        srows.append(row)
    summ = pd.DataFrame(srows)
    summ.to_csv(out_dir() / 'jitter_at_flies_gap.csv', index=False)
    return cal, summ


# ---------------------------------------------------------------------------
# registry entries (data_targets.py --add-new)
def registry_entries() -> List[Tuple[str, float, str, Optional[Tuple[float, float]]]]:
    """[(name, value, definition, 90 % interval or None)] of the repeat structure and the noise-free distances (Yang's
    z, graded), for data_targets.build: `repeat_*` (the same-repeat gap, the same-odor distances by repeat distance,
    the response rates and their ratio per repeat under the bands' criterion, the APL effect within each repeat) and
    `noise_free_*` (from products of different repeats; not the realism bands' `noise_*` entries)."""
    out = []
    T = pair_tables('z', 'graded')
    kinds = boot(T, _kinds_fn)
    for g in GENOS:
        k, lab = GKEY[g], GLAB[g]
        for ps, desc in (('all', 'the 120 pairs'), ('six', 'the six pairs with near-identical ORN input')):
            r = kinds.loc[[(g, ps, 'gap')]].iloc[0]
            out.append((f'repeat_gap{"_six" if ps == "six" else ""}_{k}', r.value,
                        f'KCs, {lab}: the same-repeat gap, the single-trial cosine distance between two odors of different '
                        f'repeats minus that of the same repeat, mean over {desc}, per recording then mean over recordings',
                        (r.lo, r.hi)))
        for kind in ('adjacent', 'first_third'):
            r = kinds.loc[[(g, 'same_odor', kind)]].iloc[0]
            out.append((f'repeat_same_odor_{kind}_{k}', r.value,
                        f'KCs, {lab}: single-trial cosine distance between two presentations of the same odor in '
                        f'{"adjacent repeats" if kind == "adjacent" else "the first and the third repeat"}, mean over odors '
                        'and recordings', (r.lo, r.hi)))
    r = kinds.loc[[('ratio', 'all', 'gap')]].iloc[0]
    out.append(('repeat_gap_ratio_tnt_ctrl', r.value, 'the same-repeat gap with APL output blocked (TNT) over that of the '
                'control flies (repeat_gap_tnt / repeat_gap_ctrl)', (r.lo, r.hi)))
    Tb = pair_tables('z', 'binary')
    kb = boot(Tb, _kinds_fn)
    for g in GENOS:
        r = kb.loc[[(g, 'all', 'gap')]].iloc[0]
        out.append((f'repeat_gap_binary_{GKEY[g]}', r.value, f'KCs, {GLAB[g]}: the same-repeat gap (as repeat_gap_'
                    f'{GKEY[g]}) of single trials binarized at 1 z', (r.lo, r.hi)))
    # noise-free distances and effect
    a4 = boot(T, _a4_fn(with_onoff=True))
    for g, k in (('ctrl', 'ctrl'), ('tnt', 'tnt')):
        lab = GLAB['TNT3c' if g == 'ctrl' else 'TNT']
        for est, name, desc in (('noise_free', '', 'all pairs of different repeats'),
                                ('noise_free_adjacent', '_adjacent', 'adjacent repeats'),
                                ('noise_free_first_third', '_first_third', 'the first against the third repeat')):
            r = a4.loc[[(g, est, 'mean_dist')]].iloc[0]
            out.append((f'noise_free_mean_dist{name}_{k}', r.value,
                        f'KCs, {lab}: noise-free mean pair distance over the 120 pairs: per recording and pair, one minus the '
                        f'mean single-trial cosine between the two odors over {desc}, over the geometric mean of the two '
                        "odors' same-odor values over the same repeats; mean over recordings", (r.lo, r.hi)))
        r = a4.loc[[(g, 'noise_free', 'six_mean')]].iloc[0]
        out.append((f'noise_free_six_{k}', r.value, f'KCs, {lab}: noise-free distance (as noise_free_mean_dist_{k}), mean over '
                    'the six pairs with near-identical ORN input', (r.lo, r.hi)))
        r = a4.loc[[(g, 'inflation', 'mean')]].iloc[0]
        out.append((f'noise_free_inflation_{k}', r.value, f'KCs, {lab}: mean pair distance of the trial means (the main '
                    f'target\'s) minus the noise-free one (noise_free_mean_dist_{k})', (r.lo, r.hi)))
    for stat, name, desc in (('mean', 'noise_free_apl_effect_mean', 'mean over the 120 pairs of the noise-free per-pair APL '
                              'effect: the control flies\' noise-free distance minus that with APL output blocked (TNT)'),
                             ('six_mean', 'noise_free_apl_effect_six', 'the noise-free per-pair APL effect, mean over the six '
                              'pairs with near-identical ORN input'),
                             ('ester_mean', 'noise_free_apl_effect_ester', 'the noise-free per-pair APL effect, mean over the '
                              'ester group\'s 15 pairs'),
                             ('alcohol_mean', 'noise_free_apl_effect_alcohol', 'the noise-free per-pair APL effect, mean over '
                              'the alcohol group\'s 6 pairs')):
        r = a4.loc[[('effect', 'noise_free', stat)]].iloc[0]
        out.append((name, r.value, desc, (r.lo, r.hi)))
    for stat, name, desc in (('block', 'noise_free_block', 'fruity x small-ester block on the noise-free matrices: mean '
                              'isotonic residual of the 9 pairs (control on APL output blocked)'),
                             ('rho_ctrl_tnt', 'noise_free_rho_ctrl_tnt', 'ρ on odor-odor distances between the two noise-free '
                              'KC matrices'),
                             ('iso_r2', 'noise_free_iso_r2', 'isotonic R² of the noise-free control matrix as a monotone '
                              'function of the noise-free APL output blocked (TNT) matrix')):
        r = a4.loc[[('both', 'noise_free', stat)]].iloc[0]
        out.append((name, r.value, desc, (r.lo, r.hi)))
    r = a4.loc[[('effect', 'noise_free', 'spearman_with_measured')]].iloc[0]
    out.append(('noise_free_apl_effect_rho_measured', r.value, 'ρ between the noise-free per-pair APL effect and the effect '
                'as measured (trial means; the main target)', (r.lo, r.hi)))
    import data_targets as DT
    effect, boot_mean, boot_rmse, sh_rho = DT.apl_effect_resamples({g: T[g]['noise_free'] for g in GENOS})
    out.append(('noise_free_apl_effect_rmse_floor', float(np.mean(boot_rmse)), 'floor of the APL-effect loss against the '
                'noise-free per-pair APL effect (as apl_effect_rmse_floor: resampled recordings against the measured ones)',
                tuple(np.percentile(boot_rmse, [5, 95]))))
    r_sh = float(np.nanmean(sh_rho))
    out.append(('noise_free_apl_effect_rho_ceiling', float(np.sqrt(2 * r_sh / (1 + r_sh))), 'ceiling on ρ on the noise-free '
                'per-pair APL effect (as apl_effect_rho_ceiling: Spearman-Brown-corrected split halves of the recordings)', None))
    # the run-down under the bands' criterion (single trials above noise_threshold, baseline-noise units)
    summ, _ = rundown_tables('noise', resp_thresh('noise'))
    rd = boot(summ, _rundown_fn('all'))
    for g in GENOS:
        for r_ in ('r1', 'r2', 'r3'):
            r = rd.loc[[(g, f'rate_{r_}')]].iloc[0]
            out.append((f'repeat_rate_{r_}_{GKEY[g]}', r.value, f'KCs, {GLAB[g]}: response rate in repeat {r_[1]}, single trials '
                        'above noise_threshold in baseline-noise units, mean over the 16 odors and recordings', (r.lo, r.hi)))
    for r_ in ('r1', 'r2', 'r3', 'session'):
        r = rd.loc[[('ratio', f'rate_{r_}')]].iloc[0]
        out.append((f'repeat_rate_ratio_{r_}', r.value, 'APL-off / APL-on response-rate ratio (APL output blocked (TNT) over '
                    f'control flies) of single trials above noise_threshold in baseline-noise units, '
                    f'{"repeat " + r_[1] if r_ != "session" else "all three repeats"}', (r.lo, r.hi)))
    # the APL effect within each repeat
    a5 = pd.read_csv(out_dir() / 'effect_by_repeat.csv') if (out_dir() / 'effect_by_repeat.csv').exists() else part_a5(N_BOOT)
    for rep in ('1', '2', '3'):
        r = a5[(a5.statistic == 'mean') & (a5.repeat.astype(str) == rep)].iloc[0]
        out.append((f'repeat_apl_effect_mean_r{rep}', r.value, f'mean over the 120 pairs of the per-pair APL effect within repeat '
                    f'{rep}: the control flies\' single-trial cosine distance minus that with APL output blocked (TNT), both '
                    f'from repeat {rep}', (r.lo, r.hi)))
    return [(n, float(v), d, None if ci is None else (float(ci[0]), float(ci[1]))) for n, v, d, ci in out]


# ---------------------------------------------------------------------------
# figures
def _style(ax):
    ax.grid(True, color=COLORS['grid'], lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(COLORS['ink2'])
    ax.tick_params(colors=COLORS['ink2'], labelsize=8)


def figures_a() -> None:
    plt = H._plt()
    out = out_dir()
    rd = pd.read_csv(out / 'rundown.csv')
    kinds = pd.read_csv(out / 'distances_by_kind.csv')
    nfp = pd.read_csv(out / 'noise_free_pairs.csv')
    ebr = pd.read_csv(out / 'effect_by_repeat.csv')
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    sel = rd[(rd.units == 'noise') & (rd.criterion == 'bands') & (rd.level == 'genotype') & (rd.subset == 'all')]
    for g in GENOS:
        s = sel[sel.genotype == g].set_index('statistic')
        for ax, stat in ((axs[0], 'rate'), (axs[1], 'amp')):
            v = [s.loc[f'{stat}_r{r}'] for r in (1, 2, 3)]
            ax.errorbar([1, 2, 3], [x.value for x in v], yerr=[[x.value - x.lo for x in v], [x.hi - x.value for x in v]],
                        color=COLORS[g], lw=2, marker='o', ms=6, capsize=3, label=GLAB[g])
        per = rd[(rd.units == 'noise') & (rd.criterion == 'bands') & (rd.level == 'recording') & (rd.genotype == g)]
        for rec, gg in per.groupby('key'):
            gg = gg.set_index('statistic')
            if 'rate_r1' in gg.index:
                axs[0].plot([1, 2, 3], [gg.loc[f'rate_r{r}', 'value'] for r in (1, 2, 3)], color=COLORS[g], lw=0.6, alpha=0.35)
    rs = sel[sel.genotype == 'ratio'].set_index('statistic')
    v = [rs.loc[f'rate_r{r}'] for r in (1, 2, 3)]
    axs[2].errorbar([1, 2, 3], [x.value for x in v], yerr=[[x.value - x.lo for x in v], [x.hi - x.value for x in v]],
                    color=COLORS['ink'], lw=2, marker='o', ms=6, capsize=3, label='all recordings')
    sel6 = rd[(rd.units == 'noise') & (rd.criterion == 'bands') & (rd.level == 'genotype') & (rd.subset == 'typical6') & (rd.genotype == 'ratio')]
    s6 = sel6.set_index('statistic')
    axs[2].plot([1, 2, 3], [s6.loc[f'rate_r{r}', 'value'] for r in (1, 2, 3)], color=COLORS['ink2'], lw=1.2, ls='--', marker='o',
                ms=5, label='six typical recordings with APL output blocked (TNT)')
    axs[2].axhspan(1.6, 2.9, color=COLORS['grid'], alpha=0.6, lw=0, label='realism band')
    axs[0].set_ylabel('response rate (single trials)')
    axs[1].set_ylabel("responders' mean amplitude\n(baseline-noise units)")
    axs[2].set_ylabel('APL-off / APL-on response-rate ratio')
    for ax, t in zip(axs, ('response rate by repeat', "responders' amplitude by repeat", 'response-rate ratio by repeat')):
        _style(ax)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel('repeat')
        ax.set_title(t, fontsize=9)
    axs[0].legend(fontsize=7, frameon=False)
    axs[2].legend(fontsize=7, frameon=False)
    fig.suptitle('The run-down over the three repeats (single trials above the bands\' threshold, baseline-noise units; 90 % '
                 'intervals over recordings)', fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'E81_rundown.png', dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    kk = kinds[(kinds.units == 'z') & (kinds['values'] == 'graded') & (kinds.level == 'genotype')]
    labels = [('same_odor', 'adjacent', 'same odor,\nadjacent repeats'), ('same_odor', 'first_third', 'same odor,\nfirst / third'),
              ('all', 'same_repeat', 'two odors,\nsame repeat'), ('all', 'different_repeat', 'two odors,\ndifferent repeats')]
    w = 0.38
    for i, g in enumerate(GENOS):
        s = kk[kk.genotype == g].set_index(['pairset', 'kind'])
        v = [s.loc[(ps, k)] for ps, k, _ in labels]
        x = np.arange(len(labels)) + (i - 0.5) * w
        axs[0].bar(x, [r.value for r in v], width=w * 0.92, color=COLORS[g], label=GLAB[g])
        axs[0].errorbar(x, [r.value for r in v], yerr=[[r.value - r.lo for r in v], [r.hi - r.value for r in v]], fmt='none',
                        ecolor=COLORS['ink'], lw=1, capsize=2)
    axs[0].set_xticks(np.arange(len(labels)))
    axs[0].set_xticklabels([lab for _, _, lab in labels], fontsize=7)
    axs[0].set_ylabel('single-trial cosine distance')
    axs[0].set_title('single-trial distances by kind (z)', fontsize=9)
    axs[0].legend(fontsize=7, frameon=False)
    per = kinds[(kinds.units == 'z') & (kinds['values'] == 'graded') & (kinds.level == 'recording')]
    for i, g in enumerate(GENOS):
        for j, ps in enumerate(('all', 'six')):
            v = per[(per.genotype == g) & (per.pairset == ps)].value.to_numpy()
            xj = j * 2.5 + i + np.linspace(-0.15, 0.15, len(v))
            axs[1].scatter(xj, v, s=18, color=COLORS[g], edgecolor='white', lw=0.5, label=GLAB[g] if j == 0 else None)
    axs[1].axhline(0, color=COLORS['ink2'], lw=0.8)
    axs[1].set_xticks([0.5, 3.0])
    axs[1].set_xticklabels(['all 120 pairs', 'the six pairs with\nnear-identical ORN input'], fontsize=7)
    axs[1].set_ylabel('same-repeat gap (per recording)')
    axs[1].set_title('two odors: different repeats minus same repeat', fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    z = nfp[nfp.units == 'z']
    for g, k in (('TNT3c', 'ctrl'), ('TNT', 'tnt')):
        axs[2].scatter(z[f'{k}_corrected'], z[f'{k}_noise_free'], s=12, color=COLORS[g], edgecolor='white', lw=0.4, label=GLAB[g])
    lim = [min(z.filter(like='_corrected').min().min(), z.filter(like='_noise_free').min().min()) - 0.02, 1.02]
    axs[2].plot(lim, lim, color=COLORS['ink2'], lw=0.8)
    axs[2].set_xlabel('the correction in use: cosine distance')
    axs[2].set_ylabel('noise-free (different repeats): cosine distance')
    axs[2].set_title('noise-free distances against the correction in use', fontsize=9)
    axs[2].legend(fontsize=7, frameon=False)
    for ax in axs:
        _style(ax)
    fig.tight_layout()
    fig.savefig(out / 'E81_distances_by_kind.png', dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].scatter(z['effect_trial_mean'], z['effect_noise_free'], s=14, color=COLORS['TNT3c'], edgecolor='white', lw=0.4,
                   label='noise-free')
    axs[0].scatter(z['effect_trial_mean'], z['effect_corrected'], s=14, color=COLORS['c4'], edgecolor='white', lw=0.4,
                   label='the correction in use')
    lim = [z[['effect_trial_mean', 'effect_noise_free', 'effect_corrected']].min().min() - 0.02,
           z[['effect_trial_mean', 'effect_noise_free', 'effect_corrected']].max().max() + 0.02]
    axs[0].plot(lim, lim, color=COLORS['ink2'], lw=0.8)
    axs[0].set_xlabel('per-pair APL effect, trial means (the main target)')
    axs[0].set_ylabel('per-pair APL effect')
    axs[0].set_title('the per-pair APL effect, noise-free and corrected', fontsize=9)
    axs[0].legend(fontsize=7, frameon=False)
    e = ebr.set_index(['statistic', 'repeat'])
    for stat, col, lab in (('mean', COLORS['ink'], 'all pairs'), ('ester_mean', COLORS['TNT3c'], 'ester group'),
                           ('alcohol_mean', COLORS['TNT'], 'alcohol group')):
        v = [e.loc[(stat, r)] for r in ('1', '2', '3')]
        axs[1].errorbar([1, 2, 3], [x.value for x in v], yerr=[[x.value - x.lo for x in v], [x.hi - x.value for x in v]], color=col,
                        lw=2, marker='o', ms=6, capsize=3, label=lab)
    axs[1].set_xticks([1, 2, 3])
    axs[1].set_xlabel('repeat')
    axs[1].set_ylabel('APL effect within the repeat\n(single-trial distances)')
    axs[1].set_title('the APL effect within each repeat', fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    for ax in axs:
        _style(ax)
    fig.tight_layout()
    fig.savefig(out / 'E81_effect.png', dpi=150)
    plt.close(fig)


def figures_b() -> None:
    plt = H._plt()
    out = out_dir()
    b1 = pd.read_csv(out / 'by_variant_noise_free.csv')
    v = b1[b1.kind == 'variant']
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, ro, lab in ((axs[0], 'raw', 'raw counts'), (axs[1], 'cal', 'calibrated readout')):
        a, b = f'{ro}_measured_loss', f'{ro}_noise_free_loss'
        if a not in v:
            continue
        vv = v[[a, b]].dropna()
        ax.scatter(vv[a], vv[b], s=8, color=COLORS['TNT3c'], alpha=0.5, lw=0)
        p = b1[b1.kind == 'population']
        for _, r in p.iterrows():
            if pd.notna(r.get(a)):
                ax.scatter(r[a], r[b], s=40, color=COLORS['TNT'], edgecolor='white', lw=0.8)
                ax.annotate(r['tag'].replace('_', ' '), (r[a], r[b]), fontsize=6, xytext=(4, -2), textcoords='offset points',
                            color=COLORS['ink2'])
        lim = [min(vv.min()) - 0.005, max(vv.max()) + 0.005]
        ax.plot(lim, lim, color=COLORS['ink2'], lw=0.8)
        rho = H.spearman(vv[a], vv[b])
        ax.set_title(f'{lab}: {len(vv)} variants, rank correlation {rho:.3f}', fontsize=9)
        ax.set_xlabel('APL-effect loss against the effect as measured')
        ax.set_ylabel('APL-effect loss against the noise-free effect')
        _style(ax)
    fig.tight_layout()
    fig.savefig(out / 'E81_variants_noise_free.png', dpi=150)
    plt.close(fig)
    cal = pd.read_csv(out / 'jitter_calibration_structure.csv')
    j = pd.read_csv(out / 'jitter_at_flies_gap.csv')
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, cs, g in ((axs[0], 'on', 'TNT3c'), (axs[1], 'off', 'TNT')):
        for (folder, model), gg in cal[cal.condition == cs].groupby(['folder', 'model']):
            gg = gg.drop_duplicates('sigma').sort_values('sigma')
            ax.plot(np.r_[0, gg.sigma], np.r_[0, gg.gap], color=COLORS['ink2'], lw=0.7, alpha=0.5)
        ax.axhline(j[f'flies_gap_{cs}'].iloc[0], color=COLORS[g], lw=1.5, ls='--', label=f'{GLAB[g]}')
        ax.set_xlabel('threshold jitter (sigma)')
        ax.set_title(f'APL {cs}: the models\' same-repeat gap', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
    axs[0].set_ylabel('same-repeat gap (calibration trials)')
    fig.tight_layout()
    fig.savefig(out / 'E81_jitter_gap.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# comparison_table.md
def _f(x, spec: str = '.3f') -> str:
    try:
        return '' if x is None or not np.isfinite(float(x)) else format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def _ci(r, spec: str = '.3f') -> str:
    if r is None:
        return ''
    lo, hi = r.get('lo', np.nan), r.get('hi', np.nan)
    s = _f(r['value'], spec)
    if pd.notna(lo) and pd.notna(hi):
        s += f' ({_f(lo, spec)} to {_f(hi, spec)})'
    return s


def table_a() -> List[str]:
    out = out_dir()
    L = ['## Parts A and B (E81_repeat_structure.py)', '', '### A1, the run-down', '']
    rd = pd.read_csv(out / 'rundown.csv')
    for units, crit, lab in (('noise', 'bands', "single trials above the bands' threshold (baseline-noise units)"),
                             ('z', '1.5 z', 'single trials above 1.5 z (the probe\'s criterion)')):
        s = rd[(rd.units == units) & (rd.criterion == crit) & (rd.level == 'genotype')]
        L += [f'Response rates, {lab}; 90 % intervals over 1000 resamples of the recordings.', '',
              '| statistic | repeat 1 | repeat 2 | repeat 3 | all three repeats |', '|---|---:|---:|---:|---:|']

        def get(g, sub, stat):
            x = s[(s.genotype == g) & (s.subset == sub) & (s.statistic == stat)]
            return x.iloc[0] if len(x) else None
        for g, sub, name in (('TNT3c', 'all', 'response rate, control flies'), ('TNT', 'all', 'response rate, APL output blocked (TNT)'),
                             ('TNT', 'typical6', 'response rate, APL output blocked (TNT), six typical recordings'),
                             ('TNT', 'strong2', 'response rate, APL output blocked (TNT), the two strongest recordings')):
            L.append(f'| {name} | ' + ' | '.join(_ci(get(g, sub, f'rate_{r}')) for r in ('r1', 'r2', 'r3', 'session')) + ' |')
        for g, name in (('TNT3c', "responders' amplitude, control flies"), ('TNT', "responders' amplitude, APL output blocked (TNT)")):
            L.append(f'| {name} | ' + ' | '.join(_ci(get(g, 'all', f'amp_{r}'), '.2f') for r in ('r1', 'r2', 'r3', 'session')) + ' |')
        for sub, name in (('all', 'APL-off / APL-on response-rate ratio'), ('typical6', 'the same ratio, six typical recordings')):
            L.append(f'| {name} | ' + ' | '.join(_ci(get('ratio', sub, f'rate_{r}'), '.2f') for r in ('r1', 'r2', 'r3', 'session')) + ' |')
        ch = {g: get(g, 'all', 'rate_change_r3_vs_r1') for g in GENOS}
        nb = {g: get(g, 'all', 'n_recordings_r3_below_r1') for g in GENOS}
        nn = {g: get(g, 'all', 'n_recordings') for g in GENOS}
        L += ['', f'Third repeat against the first: control flies {_ci(ch["TNT3c"], "+.0%")} ({_f(nb["TNT3c"]["value"], ".0f")} of '
              f'{_f(nn["TNT3c"]["value"], ".0f")} recordings lower); APL output blocked (TNT) {_ci(ch["TNT"], "+.0%")} '
              f'({_f(nb["TNT"]["value"], ".0f")} of {_f(nn["TNT"]["value"], ".0f")} lower); ratio of the first repeat minus that of '
              f'all three: {_ci(get("ratio", "all", "r1_minus_session"), "+.2f")}.', '']
        per = rd[(rd.units == units) & (rd.criterion == crit) & (rd.level == 'recording') & (rd.statistic == 'rate_change_r3_vs_r1')
                 & (rd.genotype == 'TNT')]
        L += ['Per recording with APL output blocked (TNT), third repeat against the first: ' + ', '.join(
            f'{k} {v:+.0%}' for k, v in zip(per.key, per.value)) + '.', '']
    # odors per KC by repeat (bands' criterion)
    s = rd[(rd.units == 'noise') & (rd.criterion == 'bands') & (rd.level == 'genotype') & (rd.subset == 'all')]
    L += ["Odors per KC by repeat (single trials above the bands' threshold): share of KCs responding to 0, 1, 2-3, 4-7, 8 or "
          'more of the 16 odors.', '', '| genotype and repeat | 0 | 1 | 2-3 | 4-7 | 8 or more |', '|---|---:|---:|---:|---:|---:|']
    for g in GENOS:
        for r in ('r1', 'r2', 'r3'):
            vals = [s[(s.genotype == g) & (s.statistic == f'opk_{b}_{r}')].value for b in OPK_BINS]
            L.append(f'| {GLAB[g]}, repeat {r[1]} | ' + ' | '.join(_f(v.iloc[0]) if len(v) else '' for v in vals) + ' |')
    L += ['', '### A2, single-trial distances by kind and the same-repeat gap', '']
    kk = pd.read_csv(out / 'distances_by_kind.csv')
    g_ = kk[(kk.level == 'genotype')]
    L += ["Yang's z; single-trial cosine distances, per recording then mean over recordings.", '',
          '| statistic | control flies | APL output blocked (TNT) | ratio |', '|---|---:|---:|---:|']

    def kv(units, values, g, ps, kind, spec='.3f'):
        x = g_[(g_.units == units) & (g_['values'] == values) & (g_.genotype == g) & (g_.pairset == ps) & (g_.kind == kind)]
        return _ci(x.iloc[0], spec) if len(x) else ''
    for ps, kind, name in (('same_odor', 'different_repeat', 'same odor, two repeats'), ('same_odor', 'adjacent', 'same odor, adjacent repeats'),
                           ('same_odor', 'first_third', 'same odor, first against third'), ('all', 'same_repeat', 'two odors, same repeat'),
                           ('all', 'different_repeat', 'two odors, different repeats'), ('all', 'adjacent', 'two odors, adjacent repeats'),
                           ('all', 'first_third', 'two odors, first against third')):
        L.append(f'| {name} | {kv("z", "graded", "TNT3c", ps, kind)} | {kv("z", "graded", "TNT", ps, kind)} | |')
    for ps, name in (('all', 'same-repeat gap, all pairs'), ('six', 'same-repeat gap, the six pairs'), ('nearest', 'same-repeat gap, the four nearest ester pairs'),
                     ('block', 'same-repeat gap, the block\'s six pairs'), ('ms_pairs', "same-repeat gap, ms's pairs"),
                     ('ester', 'same-repeat gap, ester group'), ('alcohol', 'same-repeat gap, alcohol group')):
        L.append(f'| {name} | {kv("z", "graded", "TNT3c", ps, "gap")} | {kv("z", "graded", "TNT", ps, "gap")} | '
                 f'{kv("z", "graded", "ratio", ps, "gap", ".1f")} |')
    for units, values, name in (('z', 'thresholded', 'same-repeat gap, sub-threshold set to 0 (1 z)'), ('z', 'binary', 'same-repeat gap, binarized at 1 z'),
                                ('noise', 'graded', 'same-repeat gap, baseline-noise units'),
                                ('noise', 'binary', "same-repeat gap, binarized at the bands' threshold")):
        L.append(f'| {name} | {kv(units, values, "TNT3c", "all", "gap")} | {kv(units, values, "TNT", "all", "gap")} | '
                 f'{kv(units, values, "ratio", "all", "gap", ".1f")} |')
    for values, name in (('binary', 'same odor, two repeats, binarized at 1 z'),):
        L.append(f'| {name} | {kv("z", values, "TNT3c", "same_odor", "different_repeat")} | {kv("z", values, "TNT", "same_odor", "different_repeat")} | |')
    per = kk[(kk.level == 'recording') & (kk.units == 'z') & (kk['values'] == 'graded') & (kk.pairset == 'all')]
    L += ['']
    for g in GENOS:
        v = per[per.genotype == g].value
        L.append(f'Per recording, {GLAB[g]}: gap {v.min():.3f} to {v.max():.3f}, positive in {int((v > 0).sum())} of {len(v)}.')
    L += ['', '### A3, the parts of the variability', '']
    vp = pd.read_csv(out / 'variance_parts.csv')
    vg = vp[vp.level == 'genotype']
    L += ['Per trial, recording means; shares of the deviation variance from the mean over repeats. Gain: common to all KCs of a '
          'presentation, proportional to the response (a cosine does not see it); shared: common to one KC\'s odors within a '
          'repeat; rest: independent between odors.', '',
          '| units, genotype, entries | SD per trial | gain | shared | rest | correlation within a repeat (gain removed) |',
          '|---|---:|---:|---:|---:|---:|']
    for units in UNITS:
        for g in GENOS:
            for cls in ('responders', 'silent', 'responders_amp1', 'responders_amp4'):
                x = vg[(vg.units == units) & (vg.genotype == g) & (vg.cls == cls)].set_index('statistic')
                if not len(x):
                    continue
                L.append(f'| {units}, {GLAB[g]}, {cls.replace("_", " ")} (mean {x.loc["mean_amplitude", "value"]:.2f}) | '
                         f'{_f(x.loc["sd_trial", "value"])} | {_ci(x.loc["share_gain"], ".2f")} | {_ci(x.loc["share_shared"], ".2f")} | '
                         f'{_ci(x.loc["share_rest"], ".2f")} | {_ci(x.loc["corr_shared"], ".2f")} |')
    cc = pd.read_csv(out / 'coresponder_corr.csv')
    L += ['']
    for units in UNITS:
        for g in GENOS:
            c = cc[(cc.units == units) & (cc.genotype == g)]
            L.append(f'Co-responders\' deviation correlation between two odors ({units}, {GLAB[g]}): median over pairs '
                     f'{c["corr"].median():.2f} (pairs {c["corr"].min():.2f} to {c["corr"].max():.2f}); with the gain part removed '
                     f'{c.corr_gain_removed.median():.2f}.')
    L += ['', '### A4, noise-free distances and the noise-free per-pair APL effect', '']
    ns = pd.read_csv(out / 'noise_free_summary.csv')

    def nv(units, side, est, stat, spec='.3f', values='graded'):
        x = ns[(ns.units == units) & (ns['values'] == values) & (ns.side == side) & (ns.estimator == est) & (ns.statistic == stat)]
        return _ci(x.iloc[0], spec) if len(x) else ''
    L += ["Yang's z, graded; 90 % intervals over resampled recordings.", '',
          '| statistic | trial means (the main target) | the correction in use | noise-free | noise-free, adjacent repeats | noise-free, first against third |',
          '|---|---:|---:|---:|---:|---:|']
    ests = ESTIMATORS
    for side, stat, name in (('ctrl', 'mean_dist', 'mean distance, control flies'), ('tnt', 'mean_dist', 'mean distance, APL output blocked (TNT)'),
                             ('effect', 'mean', 'mean APL effect'), ('ctrl', 'six_mean', 'six pairs, control flies'),
                             ('tnt', 'six_mean', 'six pairs, APL output blocked (TNT)'), ('effect', 'six_mean', 'six pairs, APL effect'),
                             ('effect', 'ester_mean', 'ester group, APL effect'), ('effect', 'alcohol_mean', 'alcohol group, APL effect'),
                             ('both', 'block', 'the block'), ('both', 'rho_ctrl_tnt', 'ρ on odor-odor distances between the two KC matrices'),
                             ('both', 'iso_r2', 'isotonic R² of the on-against-off curve'),
                             ('effect', 'rmse_floor', 'floor of the APL-effect loss'), ('effect', 'rho_ceiling', 'ceiling of ρ on the per-pair APL effect')):
        L.append(f'| {name} | ' + ' | '.join(nv('z', side, e, stat) for e in ests) + ' |')
    L += ['', f'Noise-free per-pair APL effect against the effect as measured: ρ {nv("z", "effect", "noise_free", "spearman_with_measured")}, '
          f'root mean squared difference {nv("z", "effect", "noise_free", "rms_diff_from_measured")}. Trial means minus noise-free: '
          f'control flies {nv("z", "ctrl", "inflation", "mean", "+.3f")} (six pairs {nv("z", "ctrl", "inflation", "six_mean", "+.3f")}), '
          f'APL output blocked (TNT) {nv("z", "tnt", "inflation", "mean", "+.3f")} (six pairs {nv("z", "tnt", "inflation", "six_mean", "+.3f")}). '
          f'Adjacent minus first against third: control flies {nv("z", "ctrl", "adjacent_minus_first_third", "mean", "+.3f")}, '
          f'APL output blocked (TNT) {nv("z", "tnt", "adjacent_minus_first_third", "mean", "+.3f")}.', '',
          f'In baseline-noise units: mean distances noise-free {nv("noise", "ctrl", "noise_free", "mean_dist")} / '
          f'{nv("noise", "tnt", "noise_free", "mean_dist")} (trial means {nv("noise", "ctrl", "trial_mean", "mean_dist")} / '
          f'{nv("noise", "tnt", "trial_mean", "mean_dist")}), mean APL effect noise-free {nv("noise", "effect", "noise_free", "mean")} '
          f'(trial means {nv("noise", "effect", "trial_mean", "mean")}, correction in use {nv("noise", "effect", "corrected", "mean")}).', '']
    L += ['### A5, the APL effect within each repeat', '']
    eb = pd.read_csv(out / 'effect_by_repeat.csv').set_index(['statistic', 'repeat'])

    def ev(stat, rep, spec='.3f'):
        try:
            return _ci(eb.loc[(stat, str(rep))], spec)
        except KeyError:
            return ''
    L += ['Single-trial distances of both genotypes from the same repeat; the trial means and all single trials of a repeat '
          'beside. Common error: E72\'s error common to all wirings (model minus data, means removed).', '',
          '| statistic | repeat 1 | repeat 2 | repeat 3 | single trials, same repeat | trial means |', '|---|---:|---:|---:|---:|---:|']
    for stat, name, spec in (('mean', 'mean APL effect', '+.3f'), ('ester_mean', 'ester group', '+.3f'), ('alcohol_mean', 'alcohol group', '+.3f'),
                             ('six_mean', 'six pairs', '+.3f'), ('rho_common_cal', 'ρ with the common error (calibrated readout)', '.2f'),
                             ('proj_common_cal', 'projection on the common error (calibrated readout)', '.2f'),
                             ('rho_common_raw', 'ρ with the common error (raw counts)', '.2f'),
                             ('proj_common_raw', 'projection on the common error (raw counts)', '.2f'),
                             ('c2_loss', 'C2 (raw counts): APL-effect loss', '.3f'), ('c2_pattern', 'C2: pattern part', '.3f'),
                             ('c2_mean_error', 'C2: mean error', '+.3f'), ('c2_ester_error', 'C2: ester group error', '+.3f'),
                             ('c2_alcohol_error', 'C2: alcohol group error', '+.3f')):
        L.append(f'| {name} | ' + ' | '.join(ev(stat, r, spec) for r in ('1', '2', '3', 'single', 'session')) + ' |')
    L += ['', f'Pattern between repeats (Pearson r): 1 with 2 {ev("pearson_with_2", "1", ".2f")}, 1 with 3 {ev("pearson_with_3", "1", ".2f")}, '
          f'2 with 3 {ev("pearson_with_3", "2", ".2f")}. Projection on the common error, third repeat over first: calibrated '
          f'{ev("proj_common_cal_ratio_3_over_1", "all", ".2f")}, raw {ev("proj_common_raw_ratio_3_over_1", "all", ".2f")}; mean effect, '
          f'third minus first {ev("mean_change_3_minus_1", "all", "+.3f")}.', '']
    return L


def table_b() -> List[str]:
    out = out_dir()
    L = ['### B1, the archived variants against the noise-free effect', '']
    b1 = pd.read_csv(out / 'by_variant_noise_free.csv')
    v = b1[b1.kind == 'variant']
    for ro, lab in (('raw', 'raw counts'), ('cal', 'the calibrated readout')):
        a, b = f'{ro}_measured_loss', f'{ro}_noise_free_loss'
        if a not in v:
            continue
        vv = v[[a, b]].dropna()
        L.append(f'On {lab}: {len(vv)} variants; rank correlation of the APL-effect loss against the effect as measured and against '
                 f'the noise-free effect {H.spearman(vv[a], vv[b]):.3f} (Pearson {H.pearson(vv[a], vv[b]):.3f}); the lowest ten by '
                 f'the measured effect keep {len(set(vv[a].nsmallest(10).index) & set(vv[b].nsmallest(10).index))} of their places '
                 'among the lowest ten by the noise-free one.')
    if 'present_raw_loss' in v:
        d = (v.raw_measured_loss - v.present_raw_loss).abs().max()
        L.append(f'Check: the raw loss against the effect as measured equals apl_effect_scores.py\'s by_variant.csv to {d:.1e}.')
    p = b1[b1.kind == 'population'].set_index('tag')
    L += ['', 'Populations (the mean over wirings of the pair distances per condition), no trial noise:', '',
          '| statistic | ' + ' | '.join(t.replace('_', ' ') for t in p.index) + ' |', '|---|' + '---:|' * len(p)]
    for ro in ('raw', 'cal'):
        for col, name, spec in ((f'{ro}_measured_loss', 'APL-effect loss, effect as measured', '.3f'),
                                (f'{ro}_noise_free_loss', 'APL-effect loss, noise-free effect', '.3f'),
                                (f'{ro}_measured_pattern', 'pattern part, as measured', '.3f'),
                                (f'{ro}_noise_free_pattern', 'pattern part, noise-free', '.3f'),
                                (f'{ro}_effect_mean', 'mean APL effect (model)', '+.3f'),
                                (f'{ro}_noise_free_rho', 'ρ on the per-pair APL effect, noise-free', '.2f'),
                                (f'{ro}_mean_dist_on', 'mean distance, APL on', '.3f'), (f'{ro}_mean_dist_off', 'mean distance, APL off', '.3f'),
                                (f'{ro}_noise_free_ester_err_on', 'ester group error, APL on (noise-free)', '+.3f'),
                                (f'{ro}_noise_free_ester_err_off', 'ester group error, APL off (noise-free)', '+.3f'),
                                (f'{ro}_noise_free_six_err', 'six pairs\' effect error (noise-free)', '+.3f'),
                                (f'{ro}_noise_free_split_share_inherited', 'inherited share of the pattern error (noise-free)', '.2f'),
                                (f'{ro}_measured_split_share_inherited', 'inherited share (as measured)', '.2f')):
            if col in p:
                L.append(f'| {ro}: {name} | ' + ' | '.join(_f(p.loc[t, col], spec) for t in p.index) + ' |')
    nf = noise_free_summary_row('z', 'effect', 'noise_free', 'mean')
    L += ['', f'The flies: mean APL effect as measured {_f(H.data_target("apl_effect_mean"), "+.3f")}, noise-free {nf}; mean distances '
          f'noise-free {noise_free_summary_row("z", "ctrl", "noise_free", "mean_dist")} / {noise_free_summary_row("z", "tnt", "noise_free", "mean_dist")}.', '']
    b2p = out / 'saved_trials_structure.csv'
    if b2p.exists():
        b2 = pd.read_csv(b2p)
        g = b2[b2['values'] == 'graded']
        L += ['### B2, the saved pipeline trials under part A\'s statistics', '',
              'Graded spike counts, means over the pseudo-recordings; the flies\' values in the first row (Yang\'s z).', '',
              '| model (folder) | sigma | level | same odor on / off | gap on / off | gap ratio off / on | trial means minus noise-free on / off | '
              'six pairs: trial means / noise-free / corrected, on | the same, off | per-trial rate on / off |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        T = pair_tables('z', 'graded')
        fm = {gg: T[gg].mean() for gg in GENOS}
        fl = {cs: dict(so=float(fm[gg]['same_different_repeat'].mean()),
                       gap=pmean(fm[gg]['different_repeat'], 'all') - pmean(fm[gg]['same_repeat'], 'all'),
                       infl=float((fm[gg]['trial_mean'] - fm[gg]['noise_free']).mean()),
                       six=[pmean(fm[gg][e], 'six') for e in ('trial_mean', 'noise_free', 'corrected')]) for cs, gg in (('on', 'TNT3c'), ('off', 'TNT'))}
        L.append(f'| the flies (control flies / APL output blocked (TNT)) | | | {fl["on"]["so"]:.3f} / {fl["off"]["so"]:.3f} | '
                 f'{fl["on"]["gap"]:.3f} / {fl["off"]["gap"]:.3f} | {fl["off"]["gap"] / fl["on"]["gap"]:.1f} | '
                 f'{fl["on"]["infl"]:+.3f} / {fl["off"]["infl"]:+.3f} | ' + ' / '.join(f'{x:.3f}' for x in fl['on']['six']) + ' | '
                 + ' / '.join(f'{x:.3f}' for x in fl['off']['six']) + ' | |')
        for _, r in g.sort_values(['folder', 'model', 'sigma']).iterrows():
            L.append(f'| {r.model} ({r.folder}) | {r.sigma:g} | {r.level if isinstance(r.level, str) else ""} | {r.same_odor_on:.3f} / {r.same_odor_off:.3f} | '
                     f'{r.gap_on:.3f} / {r.gap_off:.3f} | {_f(r.gap_ratio_off_on, ".1f")} | {r.inflation_on:+.3f} / {r.inflation_off:+.3f} | '
                     f'{r.six_trial_mean_on:.3f} / {r.six_noise_free_on:.3f} / {r.six_corrected_on:.3f} | '
                     f'{r.six_trial_mean_off:.3f} / {r.six_noise_free_off:.3f} / {r.six_corrected_off:.3f} | {r.rate_on:.4f} / {r.rate_off:.4f} |')
        if 'e78_six_raw_on' in g:
            gg = g.dropna(subset=['e78_six_raw_on'])
            if len(gg):
                d = max((gg.six_trial_mean_on - gg.e78_six_raw_on).abs().max(), (gg.six_trial_mean_off - gg.e78_six_raw_off).abs().max())
                L += ['', f'Check: the six pairs\' trial-mean distances equal hal\'s noise_six_pairs.csv (E78) six_raw to {d:.1e} '
                      f'({len(gg)} rows matched).']
        L.append('')
    b3p = out / 'jitter_at_flies_gap.csv'
    if b3p.exists():
        j = pd.read_csv(b3p)
        L += ['### B3, the jitter at which the model\'s same-repeat gap is the flies\'', '',
              f'The flies\' gap: control flies {j.flies_gap_on.iloc[0]:.3f}, APL output blocked (TNT) {j.flies_gap_off.iloc[0]:.3f}; '
              f'same-odor distance {j.flies_same_odor_on.iloc[0]:.3f} / {j.flies_same_odor_off.iloc[0]:.3f}. Calibration trials (three per '
              'jitter), interpolated linearly in the jitter, sigma 0 giving gap 0.', '',
              '| model (folder) | sigma at the flies\' gap, on / off | the matched sigma in use, on / off | same-odor distance there, on / off | '
              'its share of the flies\', on / off |', '|---|---:|---:|---:|---:|']
        for _, r in j.iterrows():
            L.append(f'| {r.model} ({r.folder}) | {_f(r.sigma_at_flies_gap_on)} / {_f(r.sigma_at_flies_gap_off)} | '
                     f'{_f(r.sigma_matched_on)} / {_f(r.sigma_matched_off)} | {_f(r.same_odor_at_flies_gap_on)} / {_f(r.same_odor_at_flies_gap_off)} | '
                     f'{_f(r.same_odor_share_on, ".2f")} / {_f(r.same_odor_share_off, ".2f")} |')
        L.append('')
    return L


def noise_free_summary_row(units, side, est, stat, spec='.3f') -> str:
    ns = pd.read_csv(out_dir() / 'noise_free_summary.csv')
    x = ns[(ns.units == units) & (ns['values'] == 'graded') & (ns.side == side) & (ns.estimator == est) & (ns.statistic == stat)]
    return _ci(x.iloc[0], spec) if len(x) else ''


def write_table() -> None:
    """Rewrites this script's block of comparison_table.md (between TABLE_START and TABLE_END), keeping the rest."""
    p = out_dir() / 'comparison_table.md'
    L = []
    if (out_dir() / 'rundown.csv').exists():
        L += table_a()
    if (out_dir() / 'by_variant_noise_free.csv').exists():
        L += table_b()
    block = TABLE_START + '\n' + '\n'.join(L).rstrip() + '\n' + TABLE_END + '\n'
    old = p.read_text() if p.exists() else '# E81, the flies\' variability across the three repeats: comparison tables\n\n'
    if TABLE_START in old and TABLE_END in old:
        new = old[:old.index(TABLE_START)] + block + old[old.index(TABLE_END) + len(TABLE_END) + 1:]
    else:
        new = old.rstrip('\n') + '\n\n' + block
    p.write_text(new)
    print('->', p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=('A', 'B', 'table', 'figures'), required=True)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--no-cal', action='store_true', help='B1: raw counts only (no calibrated readout from the archived counts)')
    ap.add_argument('--only', nargs='*', default=None, help='sub-parts to run (e.g. A1 A4, B2)')
    ap.add_argument('--order-table', type=Path, default=None, help='A6: the stimulus-order table (not available yet)')
    args = ap.parse_args()
    t0 = time.time()

    def want(x):
        return args.only is None or x in args.only
    if args.part == 'A':
        write_provenance('A', dict(n_boot=args.n_boot, only=args.only, units=UNITS, values=H.TRIAL_VALUES,
                                   thresholds={u: resp_thresh(u) for u in UNITS}), data_inputs())
        for name, fn in (('A1', part_a1), ('A2', part_a2), ('A3', part_a3), ('A4', part_a4), ('A5', part_a5)):
            if want(name):
                fn(args.n_boot)
                print(f'{name} done, {time.time() - t0:.0f} s', flush=True)
        part_a6(args.order_table)
        figures_a()
        write_table()
    elif args.part == 'B':
        write_provenance('B', dict(no_cal=args.no_cal, only=args.only), data_inputs() + [
            H.shared_dir() / 'results' / 'apl_effect' / 'by_variant.csv',
            H.results_dir('E72_wiring_population') / 'common_error_pairs.csv'])
        if want('B1'):
            part_b1(cal=not args.no_cal)
            print(f'B1 done, {time.time() - t0:.0f} s', flush=True)
        if want('B2'):
            part_b2()
            print(f'B2 done, {time.time() - t0:.0f} s', flush=True)
        if want('B3'):
            part_b3()
            print(f'B3 done, {time.time() - t0:.0f} s', flush=True)
        if (out_dir() / 'by_variant_noise_free.csv').exists() and (out_dir() / 'jitter_calibration_structure.csv').exists():
            figures_b()
        write_table()
    elif args.part == 'figures':
        if (out_dir() / 'rundown.csv').exists():
            figures_a()
        if (out_dir() / 'by_variant_noise_free.csv').exists() and (out_dir() / 'jitter_calibration_structure.csv').exists():
            figures_b()
    else:
        write_table()


if __name__ == '__main__':
    main()
