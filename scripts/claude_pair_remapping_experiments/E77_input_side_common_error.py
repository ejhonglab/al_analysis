#!/usr/bin/env python3
"""E77: the input side of the error common to all wirings (what reaches the KCs for the ester and the alcohol pairs).

The entry is in experiment_priorities.md (planned by a Fable subagent, claude-fable-5-2, spawned from viki 2026-11-10
17:55). The best family is C2, the breadth-2 APL-effect fit of E67 (rate0.068_aplboth_k2_cap3); its error common to all
wirings is E72's (E75's `common_error`). Parts:

- A: the best family re-fitted with another ORN input, by `E10_joint_fit.py` as it is (fit H: E38's hallem_shared.csv;
  fit S: E64's searched input for the weak odors, written in E38's layout by `--part A-input`); `--part A-score` reads
  both against C2 on the same wiring.
- B: no simulations, the anatomy of the common error: the kinds of ester pairs, the per-odor and pair-specific split
  with recording-bootstrap intervals, the split into what a model inherits from its APL-off distances, its on-against-off
  curve and a pair-specific part for every archived variant (`kc_remapping_harness.apl_effect_inherited_split`), the
  archive's input-side variants under their right names, the model's PN-level distances.
- C: the screen at fixed scales on C2 with both response rates held, and the APL-on-only emulations
  (`E77_input_screen.py`); `--part gate` reads it.
- D, E, F: re-fits of a handle that passes, the bounded input oracle, what it is (not written yet).

Outputs under results/E77_input_side_common_error/ (see the entry); large outputs under
outputs/E77_input_side_common_error/.

    python E77_input_side_common_error.py --part {A-input,A-score,B,C,gate,D,E,F}
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E77_input_side_common_error'
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
ESTER = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL = ('1-8ol', '1-5ol', '1-6ol', '6al')
# the three kinds of ester pairs (the entry's table)
NEAREST = (('2h', 'IaA'), ('2h', 'pa'), ('IaA', 'pa'), ('eb', 'ep'))
BLOCK = tuple((a, b) for a in ('2h', 'IaA', 'pa') for b in ('eb', 'ep'))
MS_PAIRS = tuple(('ms', o) for o in ('2h', 'IaA', 'pa', 'eb', 'ep'))
FITS = {'H': 'input-hallem_shared_rate0.068_aplboth_k2_cap3', 'S': 'input-searched_weak_rate0.068_aplboth_k2_cap3'}


STRONG_TNT = (('2024-07-17', 3, 'L'), ('2024-07-25', 1, 'L'))
TARGETS = ('z8', 'z6', 'noise8')
KINDS = {'ms_pairs': MS_PAIRS, 'nearest': NEAREST, 'block': BLOCK}


def _canon(idx) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([tuple(sorted(x)) for x in idx], names=['odor_a', 'odor_b'])


def _cs(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = _canon(s.index)
    return s.sort_index()


def kind_masks(pairs: pd.MultiIndex) -> Dict[str, np.ndarray]:
    """Boolean masks over `pairs` (canonical order) of the three kinds of ester pairs, the ester group's 15 pairs, the
    alcohol group's 6 pairs and all pairs."""
    keys = [tuple(x) for x in pairs]
    out = {k: np.array([x in {tuple(sorted(q)) for q in v} for x in keys]) for k, v in KINDS.items()}
    a, b = pairs.get_level_values(0), pairs.get_level_values(1)
    out['ester'] = np.asarray(a.isin(ESTER) & b.isin(ESTER))
    out['alcohol'] = np.asarray(a.isin(ALCOHOL) & b.isin(ALCOHOL))
    out['all'] = np.ones(len(pairs), dtype=bool)
    return out


def _tnt_keep(tnt: pd.DataFrame) -> List:
    keep = [r for r in tnt.index if (str(r[0]), int(r[1]), str(r[2])) not in STRONG_TNT]
    if len(keep) != len(tnt) - 2:
        raise RuntimeError(f'expected to drop 2 of {len(tnt)} TNT recordings, kept {len(keep)}')
    return keep


def data_distances(target: str) -> Dict[str, pd.Series]:
    """The KC data's mean pair distances (canonical order) with APL on (control flies) and off (APL output blocked
    (TNT)): 'z8' Yang's z, all recordings (the main target's); 'z6' the same with the six typical TNT recordings (without
    STRONG_TNT); 'noise8' baseline-noise units, all recordings (E76's effect_by_unit_pairs.csv)."""
    if target == 'noise8':
        d = pd.read_csv(H.results_dir('E76_data_side_target') / 'effect_by_unit_pairs.csv').set_index(['odor_a', 'odor_b'])
        return {'on': _cs(d['ctrl__noise']), 'off': _cs(d['tnt__noise'])}
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    if target == 'z6':
        tnt = tnt.loc[_tnt_keep(tnt)]
    return {'on': _cs(ctrl.mean()), 'off': _cs(tnt.mean())}


def members() -> List[tuple]:
    """E72's eight wirings of the best family: [(label, folder, tag)], the real wiring's first."""
    import E72_wiring_population as E72
    m = E72.new_fit_members()
    if len(m) != 8:
        raise RuntimeError(f'expected E72\'s eight wirings, found {len(m)}')
    return m


def member_distances(readout: str) -> Dict[str, Dict[str, pd.Series]]:
    """{label: {'on', 'off'}} pair distances (canonical order) of the eight wirings under `readout`, and 'mean8' (their
    mean, the population whose error is the common error)."""
    import E72_wiring_population as E72
    out = {}
    for lab, folder, tag in members():
        d = E72.distances(E72.load_fit(folder, tag), readout)
        out[lab] = {c: _cs(v) for c, v in d.items()}
    md = E72.mean_distances([out[k] for k in list(out)])
    out['mean8'] = {c: _cs(v) for c, v in md.items()}
    return out


def pattern_error(m: Dict[str, pd.Series], d: Dict[str, pd.Series]) -> pd.Series:
    """Model minus data per-pair APL effect, each with its mean removed (canonical order)."""
    me = _cs(H.apl_effect(m['on'], m['off']))
    de = _cs(H.apl_effect(d['on'], d['off'])).reindex(me.index)
    return (me - me.mean()) - (de - de.mean())


def _check_common(md: Dict[str, pd.Series], readout: str) -> None:
    """The population's pattern error against the z-based target must be E72's common error (to rounding)."""
    import E75_glomerulus_apl_profile as E75
    c = _cs(E75.common_error(readout))
    e = pattern_error(md, data_distances('z8')).reindex(c.index)
    if float(np.max(np.abs(e.to_numpy() - c.to_numpy()))) > 1e-9:
        raise RuntimeError(f'the population\'s error ({readout}) is not E72\'s common error')


def searched_input(out: Path) -> Path:
    """Part A's fit S input: E64's pass-1 searched ORN input for the weak odors (orn_input_searched_pass1.csv, two-row
    panel / odor header) rewritten in E38's layout (glomeruli x abbreviated odors) as inputs/searched_weak.csv, values
    equal to the last digit. Returns the path relative to the shared directory, for `E10_joint_fit.py --orn-input`."""
    orn = H.apl16_orn_deltas()
    src = H.results_dir('E64_input_oracle') / 'orn_input_searched_pass1.csv'
    s = pd.read_csv(src, header=[0, 1], index_col=0, float_precision='round_trip')
    s.index.name = 'glomerulus'
    s.columns = [str(o).split(' @ ')[0] for o in s.columns.get_level_values(1)]
    ab = H.abbrev_columns(orn)
    if list(s.columns) != list(ab.columns):
        raise ValueError(f'{src}: odors {list(s.columns)} differ from {list(ab.columns)}')
    s = s.reindex(orn.index)
    if s.isna().any().any():
        raise ValueError(f'{src}: glomeruli differ from apl16_orn_deltas()')
    dst = out / 'inputs' / 'searched_weak.csv'
    dst.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(dst)
    back = pd.read_csv(dst, index_col=0, float_precision='round_trip')
    if not np.array_equal(back.to_numpy(dtype=float), s.to_numpy(dtype=float)):
        raise RuntimeError(f'{dst} does not reproduce {src} to the last digit')
    changed = (np.abs(s.to_numpy(dtype=float) - ab.to_numpy(dtype=float)) > 1e-9)
    odors = [o for o, c in zip(s.columns, changed.any(axis=0)) if c]
    print(f'{dst}: {int(changed.sum())} of {changed.size} values differ from the lab input, in {odors}', flush=True)
    return dst.relative_to(H.shared_dir())


def score_fits(out: Path) -> pd.DataFrame:
    """Part A-score: fits H and S against C2 on the real wiring (the APL-effect loss with its parts on raw counts and
    under the calibrated readout, the ms pairs, the 1-8ol pairs, the group errors, the per-odor part, the bands under
    both rules). Writes partA_fits.csv (one row per fit and readout; C2's rows first) and partA_pairs.csv (each fit's
    per-pair pattern error against 'z8')."""
    import E72_wiring_population as E72
    fl = [('C2', C2[0], C2[1])] + [(k, EXPERIMENT, v) for k, v in FITS.items()]
    rows, prs = [], []
    for label, folder, tag in fl:
        o = E72.load_fit(folder, tag)
        if o is None:
            print(f'{label} ({tag}): no saved outputs yet', flush=True)
            continue
        real = realism_of(o)
        for readout in ('raw', 'cal'):
            d = {c: _cs(v) for c, v in E72.distances(o, readout).items()}
            st, _ = H.apl_effect_decomposition(d['on'], d['off'], E72.data_for(readout))
            row = dict(fit=label, folder=folder, tag=tag, readout=readout, **st, **real)
            for target in TARGETS:
                err = pattern_error(d, data_distances(target))
                masks = kind_masks(err.index)
                row[f'pattern_rms_{target}'] = float(np.sqrt(np.mean(err.to_numpy() ** 2)))
                for kind, mk in masks.items():
                    if kind != 'all':
                        row[f'err_{kind}_{target}'] = float(err[mk].mean())
                a, b = err.index.get_level_values(0), err.index.get_level_values(1)
                row[f'err_1-8ol_pairs_{target}'] = float(err[np.asarray((a == '1-8ol') | (b == '1-8ol'))].mean())
                row[f'err_eb_1-8ol_{target}'] = float(err.get(tuple(sorted(('eb', '1-8ol'))), np.nan))
                if target == 'z8':
                    prs.append(pd.DataFrame({'fit': label, 'readout': readout, 'odor_a': a, 'odor_b': b, 'error': err.to_numpy()}))
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'partA_fits.csv', index=False)
    if prs:
        pd.concat(prs, ignore_index=True).to_csv(out / 'partA_pairs.csv', index=False)
    cols = ['fit', 'readout', 'apl_effect_rmse', 'apl_effect_mean', 'apl_effect_rmse_centered', 'apl_effect_spearman',
            'apl_effect_per_odor', 'apl_effect_pair_specific', 'err_ms_pairs_z8', 'err_1-8ol_pairs_z8', 'err_ester_z8',
            'err_alcohol_z8', 'pattern_rms_z6', 'rate_on', 'ratio', 'rate_rho_on', 'realism_flag', 'realism_flag_zcrit']
    print(res[[c for c in cols if c in res.columns]].round(3).to_string(index=False), flush=True)
    return res


def realism_of(o: Dict[str, tuple]) -> dict:
    """A fit's response rates, their ratio, the response rate ρ and the odors-per-KC distances under both rules (the
    model's two spikes against the KC data in baseline-noise units, the bands since 2026-11-10 16:10, and against
    Yang's z), and the flags (as E75's save_points computes them)."""
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
    row['realism_flag'] = A.realism_flag(rate_on, rate_off, row['rate_rho_on'], row['opk_on_tv'], row['opk_off_tv'], units='noise')
    row['realism_flag_zcrit'] = A.realism_flag(rate_on, rate_off, row['rate_rho_on_zcrit'], row['opk_on_tv_zcrit'],
                                               row['opk_off_tv_zcrit'], units='z')
    return row


def pair_kinds(out: Path) -> pd.DataFrame:
    """Part B: per kind of ester pair (ms's, the four nearest, the block's six) and the alcohol group, the KC data's
    distances and APL effect, C2's, the common error and its share of the squared sum; for C2, each of the eight wirings
    and their mean, raw and under the readout, against all eight recordings with APL output blocked (TNT), the six
    typical ones and in baseline-noise units (E76's tables). Writes pair_kinds.csv."""
    rows = []
    for readout in ('raw', 'cal'):
        md = member_distances(readout)
        _check_common(md['mean8'], readout)
        for target in TARGETS:
            dd = data_distances(target)
            for lab, m in md.items():
                err = pattern_error(m, dd)
                masks = kind_masks(err.index)
                me = _cs(H.apl_effect(m['on'], m['off'])).reindex(err.index)
                de = _cs(H.apl_effect(dd['on'], dd['off'])).reindex(err.index)
                tot = float((err.to_numpy() ** 2).sum())
                for kind, mk in masks.items():
                    rows.append(dict(member=lab, readout=readout, target=target, kind=kind, n_pairs=int(mk.sum()),
                                     data_on=float(dd['on'].reindex(err.index)[mk].mean()), data_off=float(dd['off'].reindex(err.index)[mk].mean()),
                                     data_effect=float(de[mk].mean()), model_on=float(m['on'].reindex(err.index)[mk].mean()),
                                     model_off=float(m['off'].reindex(err.index)[mk].mean()), model_effect=float(me[mk].mean()),
                                     error=float(err[mk].mean()), share_sq=float((err[mk].to_numpy() ** 2).sum()) / tot))
    res = pd.DataFrame(rows)
    res.to_csv(out / 'pair_kinds.csv', index=False)
    print(res[(res.member == 'mean8') & (res.target == 'z8')].round(3).to_string(index=False), flush=True)
    return res


def per_odor_parts(out: Path, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Part B: the common error split into per-odor shares and a pair-specific remainder (two-way additive fit over
    the 120 pairs), with recording-bootstrap intervals, and each odor's share error against the model-minus-data gain
    of its response fraction when APL is blocked. Writes per_odor_parts.csv (per odor) and group_parts.csv (the ester
    and alcohol groups: error, per-odor and pair-specific means), for the population (mean8) under both readouts against
    'z8' and 'z6', with 90 % intervals over `n_boot` resamples of the recordings (each genotype with replacement)."""
    import E72_wiring_population as E72
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt8 = H.yang_recording_rdms_values('TNT', 'graded')
    tnts = {'z8': tnt8, 'z6': tnt8.loc[_tnt_keep(tnt8)]}
    # response fractions per odor: the population's (mean over the eight wirings, two spikes) and the flies' (z > 1)
    fr = {c: [] for c in ('on', 'off')}
    for _, folder, tag in members():
        o = E72.load_fit(folder, tag)
        for c in ('on', 'off'):
            fr[c].append(H.abbrev_columns(o[c][0]).gt(0.5).mean())
    m_on, m_off = (pd.concat(fr[c], axis=1).mean(axis=1) for c in ('on', 'off'))
    d_on = H.abbrev_columns(H.yang_response_fractions('TNT3c')).mean()
    d_off = H.abbrev_columns(H.yang_response_fractions('TNT')).mean()
    gain = (m_off - m_on) - (d_off - d_on).reindex(m_on.index)
    rng = np.random.default_rng(seed)
    orows, grows = [], []
    for readout in ('cal', 'raw'):
        md = member_distances(readout)['mean8']
        me = _cs(H.apl_effect(md['on'], md['off']))
        for target, tnt in tnts.items():
            def parts(c_df, t_df):
                pp = H.apl_effect_pair_parts(md['on'], md['off'], H.canonical_pairs(c_df.mean() - t_df.mean()))
                pp.index = _canon(pp.index)
                pp = pp.reindex(me.index)
                sh, _ = H.odor_shares(pp['error'])
                masks = kind_masks(pp.index)
                g = {(grp, k): float(pp[k][masks[grp]].mean()) for grp in ('ester', 'alcohol') for k in ('error', 'per_odor', 'pair_specific')}
                return sh, g
            sh0, g0 = parts(ctrl, tnt)
            bs, bg = [], []
            for _ in range(n_boot):
                c_i = rng.integers(0, len(ctrl), len(ctrl))
                t_i = rng.integers(0, len(tnt), len(tnt))
                s, g = parts(ctrl.iloc[c_i], tnt.iloc[t_i])
                bs.append(s)
                bg.append(g)
            bsd = pd.concat(bs, axis=1)
            for o in sh0.index:
                orows.append(dict(readout=readout, target=target, odor=o, share=float(sh0[o]),
                                  lo=float(bsd.loc[o].quantile(0.05)), hi=float(bsd.loc[o].quantile(0.95)),
                                  model_frac_on=float(m_on.get(o, np.nan)), model_frac_off=float(m_off.get(o, np.nan)),
                                  data_frac_on=float(d_on.get(o, np.nan)), data_frac_off=float(d_off.get(o, np.nan)),
                                  gain_diff=float(gain.get(o, np.nan))))
            for key, v in g0.items():
                vals = np.array([b[key] for b in bg])
                grows.append(dict(readout=readout, target=target, group=key[0], part=key[1], value=v,
                                  lo=float(np.quantile(vals, 0.05)), hi=float(np.quantile(vals, 0.95))))
    od, gr = pd.DataFrame(orows), pd.DataFrame(grows)
    od.to_csv(out / 'per_odor_parts.csv', index=False)
    gr.to_csv(out / 'group_parts.csv', index=False)
    for (readout, target), g in od.groupby(['readout', 'target']):
        r = np.corrcoef(g.share, g.gain_diff)[0, 1]
        print(f'{readout} {target}: odor share error vs response-fraction gain difference r = {r:.2f}', flush=True)
    print(gr.round(3).to_string(index=False), flush=True)
    return od


def inherited_parts(out: Path) -> pd.DataFrame:
    """Part B: for every archived variant with both conditions, its pattern error split into what it inherits from its
    APL-off distances, its on-against-off curve and a pair-specific part (`H.apl_effect_inherited_split`), the data's
    curve fitted in sample, leaving one odor out and on halves of the recordings. Writes inherited_parts.csv (one row
    per variant, readout, target and data-curve fit; the eight wirings and their mean under folder 'E72 members') and
    inherited_halves.csv (the population's shares with the data's curve fitted on one half of the recordings, 100
    splits)."""
    import E72_wiring_population as E72
    dds = {t: data_distances(t) for t in ('z8', 'z6')}
    # the data's curve in sample and leaving each odor out (a pair gets the mean of its two odors' left-out curves)
    curves = {}
    for t, dd in dds.items():
        on, off = dd['on'], dd['off'].reindex(dd['on'].index)
        curves[(t, 'insample')] = H.power_curve(1 - off.to_numpy(), 1 - on.to_numpy())
        odors = sorted(set(on.index.get_level_values(0)) | set(on.index.get_level_values(1)))
        loo = {}
        for o in odors:
            keep = ~(on.index.get_level_values(0).isin([o]) | on.index.get_level_values(1).isin([o]))
            loo[o] = H.power_curve(1 - off.to_numpy()[keep], 1 - on.to_numpy()[keep])
        pairs = list(on.index)
        curves[(t, 'loo')] = (pairs, np.array([[(loo[a][0] + loo[b][0]) / 2, (loo[a][1] + loo[b][1]) / 2] for a, b in pairs]))

    def split_rows(folder, tag, readout, m):
        rows = []
        for t, dd in dds.items():
            for mode in ('insample', 'loo'):
                cv = curves[(t, mode)]
                if mode == 'loo':
                    pairs, arr = cv
                    s0 = H.apl_effect_inherited_split(m['on'], m['off'], dd['on'], dd['off'], data_curve=curves[(t, 'insample')])
                    order = {p: i for i, p in enumerate(pairs)}
                    arr = arr[[order[tuple(sorted(x))] for x in s0.index]]
                    s = H.apl_effect_inherited_split(m['on'], m['off'], dd['on'], dd['off'], data_curve=arr,
                                                     model_curve=s0.attrs['model_curve'])
                    mc = s0.attrs['model_curve']
                    dc = (np.nan, np.nan)
                else:
                    s = H.apl_effect_inherited_split(m['on'], m['off'], dd['on'], dd['off'], data_curve=cv)
                    mc, dc = s.attrs['model_curve'], cv
                masks = kind_masks(_canon(s.index))
                sh = H.inherited_shares(s)
                row = dict(folder=folder, tag=tag, readout=readout, target=t, data_curve_fit=mode, n_pairs=len(s), **sh,
                           pattern_rms_without_inherited=float(np.sqrt(np.mean((s.error_c - s.inherited_c) ** 2))),
                           a_data=dc[0], b_data=dc[1], a_model=mc[0], b_model=mc[1])
                for grp in ('ester', 'alcohol'):
                    for k in ('error', 'inherited', 'curve', 'pair_specific'):
                        row[f'{grp}_{k}'] = float(s[f'{k}_c'][masks[grp]].mean())
                rows.append(row)
        return rows

    rows = []
    for readout in ('raw', 'cal'):
        for lab, m in member_distances(readout).items():
            rows += split_rows('E72 members', lab, readout, m)
    for folder in sorted(H.shared_dir().joinpath('results').glob('E*')):
        lp = folder / 'model_pair_distances_long.csv'
        if not lp.exists():
            continue
        d = pd.read_csv(lp)
        if not {'tag', 'condition', 'odor_a', 'odor_b', 'distance'} <= set(d.columns):
            continue
        d = d.drop_duplicates(['tag', 'condition', 'odor_a', 'odor_b'], keep='last')
        for tag, g in d.groupby('tag'):
            if not {'on', 'acute_off'} <= set(g.condition):
                continue
            m = {c: _cs(g[g.condition == k].set_index(['odor_a', 'odor_b'])['distance']) for c, k in (('on', 'on'), ('off', 'acute_off'))}
            try:
                rows += split_rows(folder.name, tag, 'raw', m)
            except Exception as e:
                print(f'{folder.name}:{tag} (raw): {e!r}'[:200], flush=True)
                continue
            o = E72.load_fit(folder.name, tag)
            if o is not None:
                try:
                    rows += split_rows(folder.name, tag, 'cal', {c: _cs(v) for c, v in E72.distances(o, 'cal').items()})
                except Exception as e:
                    print(f'{folder.name}:{tag} (cal): {e!r}'[:200], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'inherited_parts.csv', index=False)
    # the population with the data's curve fitted on half of the recordings
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    rng = np.random.default_rng(0)
    hrows = []
    for readout in ('cal', 'raw'):
        md = member_distances(readout)['mean8']
        for i in range(100):
            ic, it = rng.permutation(len(ctrl))[:len(ctrl) // 2], rng.permutation(len(tnt))[:len(tnt) // 2]
            on_h, off_h = _cs(ctrl.iloc[ic].mean()), _cs(tnt.iloc[it].mean()).reindex(_cs(ctrl.iloc[ic].mean()).index)
            cv = H.power_curve(1 - off_h.to_numpy(), 1 - on_h.to_numpy())
            s = H.apl_effect_inherited_split(md['on'], md['off'], dds['z8']['on'], dds['z8']['off'], data_curve=cv)
            hrows.append(dict(readout=readout, split=i, a_data=cv[0], b_data=cv[1], **H.inherited_shares(s)))
    hv = pd.DataFrame(hrows)
    hv.to_csv(out / 'inherited_halves.csv', index=False)
    pop = res[(res.folder == 'E72 members') & (res.tag == 'mean8')]
    print(pop[['readout', 'target', 'data_curve_fit', 'share_inherited', 'share_curve', 'share_pair_specific', 'pattern_rms',
               'pattern_rms_without_inherited', 'ester_inherited', 'alcohol_inherited']].round(3).to_string(index=False))
    print(hv.groupby('readout')[['share_inherited']].quantile([0.05, 0.5, 0.95]).round(3).to_string())
    arc = res[(res.folder != 'E72 members') & (res.data_curve_fit == 'insample') & (res.target == 'z8')]
    print(f'{arc.tag.nunique()} archived variants; median inherited share raw '
          f'{arc[arc.readout == "raw"].share_inherited.median():.2f}, cal {arc[arc.readout == "cal"].share_inherited.median():.2f}', flush=True)
    return res


def archive_input_variants(out: Path) -> pd.DataFrame:
    """Part B: the archive's input-side variants (E3 part A's pn_inhadd sweep on each base, E3's bouton arm, E32,
    E38, E4's random wirings, E16's per-claw threshold and saturation variants) under their right names, each against
    its own base (projection on the common error, pattern error; the inherited share where inherited_parts.csv has
    it). Writes archive_input_variants.csv."""
    import re
    a = pd.read_csv(H.results_dir('E75_glomerulus_apl_profile') / 'archive_vs_common_error.csv')
    rules = [  # (folder, tag regex, what it is, base tag or None)
        ('E3_drive_dependent_gain', r'^hemibrain-wd20_inhadd', "the antennal lobe's divisive normalization weakened (pn_inhadd), one-row-per-KC wd20 base", 'hemibrain-wd20_inhadd31.4088'),
        ('E3_drive_dependent_gain', r'^claw-dynamics_inhadd', "the antennal lobe's divisive normalization weakened (pn_inhadd), claw-dynamics base", 'claw-dynamics_inhadd31.4088'),
        ('E3_drive_dependent_gain', r'^claw-dynamics_noPNAPL', 'no PN->APL input, claw-dynamics base', 'claw-dynamics_inhadd31.4088'),
        ('E3_drive_dependent_gain', r'^bouton_pnapl_x', 'the APL->bouton weight scaled (APL inhibition of the PN boutons)', 'bouton_pnapl_x1'),
        ('E32_pn_stage', r'.', 'the ORN->PN stage (E32)', None),
        ('E38_hallem_input', r'.', "Hallem's receptor data as input (E38)", None),
        ('E4_substrate', r'^(uniform|hemidraw)_', 'random PN->KC wirings of the one-row-per-KC model (E4)', None),
        ('E16_claw_saturation', r'_thr\d', 'a per-claw threshold (with or without a cap)', None),
        ('E16_claw_saturation', r'^(bouton|claw-dynamics)_tanh\d+$', 'a per-claw saturation (tanh cap) alone', None),
    ]
    ip = out / 'inherited_parts.csv'
    inh = None
    if ip.exists():
        inh = pd.read_csv(ip)
        inh = inh[(inh.target == 'z8') & (inh.data_curve_fit == 'insample')]
    rows = []
    for folder, rx, what, base in rules:
        sub = a[(a.folder == folder) & a.tag.str.contains(rx, regex=True)]
        b = a[(a.folder == folder) & (a.tag == base)].iloc[0] if base is not None and ((a.folder == folder) & (a.tag == base)).any() else None
        for _, r in sub.iterrows():
            row = dict(folder=folder, tag=r.tag, what=what, cal_proj_common=r.cal_proj_common, cal_err_rms=r.cal_err_rms,
                       raw_proj_common=r.raw_proj_common, raw_err_rms=r.raw_err_rms, base_tag=base,
                       base_cal_proj_common=(b.cal_proj_common if b is not None else np.nan),
                       base_cal_err_rms=(b.cal_err_rms if b is not None else np.nan))
            if inh is not None:
                for ro in ('raw', 'cal'):
                    q = inh[(inh.folder == folder) & (inh.tag == r.tag) & (inh.readout == ro)]
                    row[f'{ro}_share_inherited'] = float(q.share_inherited.iloc[0]) if len(q) else np.nan
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(out / 'archive_input_variants.csv', index=False)
    print(res.groupby(['folder', 'what'])[['cal_proj_common', 'cal_err_rms']].agg(['min', 'median', 'max']).round(3).to_string(), flush=True)
    return res


def pn_level(out: Path) -> pd.DataFrame:
    """Part B: the model's PN totals per odor and its PN-level odor-odor distances against the ORN input's and against
    Remy's PN-bouton and KC-claw matrices (the 66 megamat pairs), per fly where the data allow. Writes pn_totals.csv
    (per odor: the ORN input's summed positive deltas and the model's summed PN rates, C2's PN stage, 0-0.5 s) and
    pn_level.csv (per pair: correlation distances across glomeruli of the ORN input and of the model's PN rates, and
    Remy's bouton and claw distances (1 - Pearson across ROIs, data/remy_rdm_timeseries/windows/, the 0.5-3 s and
    0.5-1 s windows), per fly and the fly mean; the megamat odors the APL16 panel shares)."""
    import json
    import E77_input_screen as S
    sim, info = S.build_c2_simulator()
    runner = S.PerOdorRunner(sim, info)
    r, _ = sim(dict(info['params'], wAPLKC_scale=0.0, wAPLPN_scale=0.0))
    rates = runner.pn_rates()
    rates.columns = list(H.abbrev_columns(r).columns)
    orn = H.abbrev_columns(H.apl16_orn_deltas().astype(float))
    tot = pd.DataFrame({'orn_sum_pos': orn.clip(lower=0).sum(), 'orn_sum': orn.sum(), 'pn_sum': rates.sum(),
                        'pn_mean': rates.mean(), 'pn_max': rates.max()})
    tot.index.name = 'odor'
    tot.to_csv(out / 'pn_totals.csv')

    def corrdist(m: pd.DataFrame) -> pd.Series:
        c = np.corrcoef(m.to_numpy(dtype=float).T)
        o = list(m.columns)
        return pd.Series({tuple(sorted((o[i], o[j]))): 1 - c[i, j] for i in range(len(o)) for j in range(i + 1, len(o))})

    wdir = H.shared_dir() / 'data' / 'remy_rdm_timeseries' / 'windows'
    shared = [o for o in json.loads((wdir.parent / 'odors.json').read_text()) if o in set(orn.columns)]
    cols = {'orn_input': corrdist(orn[shared]), 'model_pn': corrdist(rates[shared])}
    for stage, fname in (('remy_boutons', 'pn_boutons'), ('remy_claws', 'kc_claws')):
        for win in ('odor_0.5-3', 'early_0.5-1'):
            for f in sorted(wdir.glob(f'{fname}_*_corrdist_{win}.csv')):
                fly = f.name.split('_')[2]
                m = pd.read_csv(f, index_col=0)
                cols[f'{stage}_{fly}_{win}'] = pd.Series({tuple(sorted((a, b))): float(m.loc[a, b])
                                                          for i, a in enumerate(shared) for b in shared[i + 1:]})
    res = pd.DataFrame(cols)
    res.index = pd.MultiIndex.from_tuples(res.index, names=['odor_a', 'odor_b'])
    res.to_csv(out / 'pn_level.csv')
    near = [tuple(sorted(p)) for p in NEAREST]
    summ = pd.DataFrame({'mean_all': res.mean(), 'mean_nearest': res.loc[[p for p in near if p in res.index]].mean(),
                         'spearman_with_model_pn': res.corrwith(res['model_pn'], method='spearman'),
                         'spearman_with_orn_input': res.corrwith(res['orn_input'], method='spearman')})
    summ.to_csv(out / 'pn_level_summary.csv')
    print(tot.round(1).to_string(), flush=True)
    print(summ.round(3).to_string(), flush=True)
    return res


# the controls of the gate's third condition: a point's control is the opposite direction of S3 (power p against 1 / p,
# the nearest of the screen's powers) or the handle's permutation
S3_OPPOSITE = {'S3_pow0.5': 'S3_pow2', 'S3_pow2': 'S3_pow0.5', 'S3_pow0.75': 'S3_pow1.33', 'S3_pow1.33': 'S3_pow0.75'}
GATE_PATTERN_GAIN = 0.003
GATE_RAW_SLACK = 0.005
GATE_TRANSFER = 0.15


def _control_of(pid: str) -> Optional[str]:
    if pid in S3_OPPOSITE:
        return S3_OPPOSITE[pid]
    if pid.startswith('S5_claws_e'):
        return 'S5_claws_perm'
    for h in ('L1', 'L2'):
        if pid.startswith(f'{h}_p'):
            return f'{h}_perm'
    return None


def gate(out: Path) -> pd.DataFrame:
    """The gate on part C's screen, as the entry states it: a handle goes on if its best point (the lowest pattern part
    under the calibrated readout among its points that are not identities or controls) is inside the four bands (the
    rule since 2026-11-10 16:10; Yang's z reported beside) and (1) lowers C2's pattern part under the readout by 0.003 or
    more with the raw-count loss no more than 0.005 above C2's, (2) has a transfer reduction of the common error of 15 %
    or more and, against the six typical TNT recordings, at least half that share, and (3) where it has a control (S3's
    opposite direction, S5's permutation, L1's and L2's permutations), the control gains less than half as much on the
    pattern part. At most two go on: the best static handle and the best emulation, or the two best static handles of
    different kinds. Writes gate.csv (every non-identity point with its conditions; `best` marks each handle's best point,
    `goes_on` the ones that go on)."""
    s = pd.read_csv(out / 'screen.csv').drop_duplicates('point_id', keep='last').set_index('point_id')
    base = s.loc['base']
    pts = s[(s.index != 'base') & ~s['identity'].fillna(False).astype(bool)].copy()
    gain = base.cal_pattern - pts.cal_pattern
    pts['pattern_gain'] = gain
    pts['inside_bands'] = pts.realism_flag.isna() | (pts.realism_flag.astype(str).str.strip() == '')
    pts['inside_bands_zcrit'] = pts.realism_flag_zcrit.isna() | (pts.realism_flag_zcrit.astype(str).str.strip() == '')
    pts['c1'] = (gain >= GATE_PATTERN_GAIN) & (pts.raw_loss <= base.raw_loss + GATE_RAW_SLACK)
    pts['c2'] = (pts.cal_transfer_all >= GATE_TRANSFER) & (pts.cal_transfer_six >= 0.5 * pts.cal_transfer_all)
    ctrl = [_control_of(i) for i in pts.index]
    pts['control'] = ctrl
    pts['control_gain'] = [float(gain.get(c, np.nan)) if c else np.nan for c in ctrl]
    pts['c3'] = [True if c is None else bool(np.isfinite(cg) and cg < 0.5 * g) for c, cg, g in zip(ctrl, pts.control_gain, gain)]
    pts['passes'] = pts.inside_bands & pts.c1 & pts.c2 & pts.c3
    is_control = pts.index.isin([c for c in set(ctrl) if c]) & pts.index.str.contains('perm')
    pts['best'] = False
    cand = pts[~is_control]
    for h, g in cand.groupby('handle'):
        pts.loc[g.cal_pattern.idxmin(), 'best'] = True
    best = pts[pts.best & pts.passes].sort_values('cal_pattern')
    static = best[best.kind == 'static']
    emul = best[best.kind == 'emulation']
    on = []
    if len(static):
        on.append(static.index[0])
    if len(emul):
        on.append(emul.index[0])
    elif len(static) > 1:
        on.append(static.index[1])
    pts['goes_on'] = pts.index.isin(on)
    pts['base_cal_pattern'] = base.cal_pattern
    pts['base_raw_loss'] = base.raw_loss
    cols = ['handle', 'setting', 'kind', 'best', 'goes_on', 'passes', 'inside_bands', 'inside_bands_zcrit', 'c1', 'c2', 'c3',
            'cal_loss', 'cal_mean', 'cal_pattern', 'pattern_gain', 'raw_loss', 'cal_rho', 'cal_transfer_all', 'cal_transfer_six',
            'control', 'control_gain', 'cal_err_ester', 'cal_err_alcohol', 'cal_err_ms', 'cal_err_nearest', 'cal_err_block',
            'cal_block', 'rate_on', 'ratio', 'rate_rho_on', 'opk_on_tv', 'opk_off_tv', 'realism_flag', 'realism_flag_zcrit',
            'thr_factor', 'apl_factor', 'at_bound_thr', 'at_bound_apl', 'base_cal_pattern', 'base_raw_loss']
    res = pts[[c for c in cols if c in pts.columns]].reset_index()
    res.to_csv(out / 'gate.csv', index=False)
    show = ['point_id', 'best', 'passes', 'cal_loss', 'cal_pattern', 'pattern_gain', 'raw_loss', 'cal_transfer_all',
            'cal_transfer_six', 'control_gain', 'inside_bands']
    print(f"C2: pattern {base.cal_pattern:.4f}, raw loss {base.raw_loss:.4f}", flush=True)
    print(res[show].round(4).to_string(index=False), flush=True)
    print(f'goes on: {on or "none"}', flush=True)
    return res


# part E: the bounded input oracle (the entry's design; E64's evolution strategy on C2 at fixed parameters)
E_ODORS = ('eb', '1-8ol', 'IaA', 'ep', '2h', '1-6ol', 'pa', '6al', '1-5ol', 'ms')
E_POP, E_GENS = 32, 8
E_BOUNDS = (0.5, 2.0)
E_RATE_TOL, E_PENALTY = 0.15, 10.0
E_SIGMA0, E_SIGMA_MIN, E_SIGMA_MAX = 0.3, 0.03, 1.0


def recording_halves(seed: int = 0) -> Dict[str, Dict[str, list]]:
    """Two halves of the recordings for part E: each genotype's flies (a fly's hemispheres together) split in two, the
    two strong TNT recordings (STRONG_TNT) in different halves; {'A' / 'B': {'ctrl' / 'tnt': [recording keys]}}."""
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    rng = np.random.default_rng(seed)

    def split(idx, must_split=()):
        flies = sorted({(str(r[0]), int(r[1])) for r in idx})
        strong = {(d, f) for d, f, _ in must_split}
        for _ in range(1000):
            perm = [flies[i] for i in rng.permutation(len(flies))]
            a = set(perm[:len(perm) // 2])
            if len(strong) < 2 or len(strong & a) == 1:
                return ([r for r in idx if (str(r[0]), int(r[1])) in a], [r for r in idx if (str(r[0]), int(r[1])) not in a])
        raise RuntimeError('no split keeps the strong recordings apart')
    ca, cb = split(list(ctrl.index))
    ta, tb = split(list(tnt.index), STRONG_TNT)
    return {'A': {'ctrl': ca, 'tnt': ta}, 'B': {'ctrl': cb, 'tnt': tb}}


def half_effect(half: Dict[str, list]) -> pd.Series:
    """The data's per-pair APL effect (Yang's z, graded) from one half's recordings, canonical (sorted) pair order."""
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    return _cs(H.apl_effect(H.canonical_pairs(ctrl.loc[half['ctrl']].mean()), H.canonical_pairs(tnt.loc[half['tnt']].mean())))


def c2_readout_K() -> float:
    """The calibrated readout's K of C2 (set on its APL-on counts as E72's `distances` sets it), held fixed in part E."""
    import E72_wiring_population as E72
    from E67_apl_effect_fits import refs
    o = E72.load_fit(*C2)
    return float(H.calibrate_readout(o['on'][1], 'saturating', refs()['cal_target'], stat='participation_ratio', responders_min=2.0))


def _pattern_sse(s_on: pd.DataFrame, s_off: pd.DataFrame, K: float, target: pd.Series) -> float:
    on = H.readout_pair_distances(s_on, 'saturating', K)
    off = H.readout_pair_distances(s_off, 'saturating', K)
    e = _cs(H.apl_effect(on, off))
    d = target.reindex(e.index)
    err = (e - e.mean()) - (d - d.mean())
    return float((err.to_numpy() ** 2).sum())


def input_oracle(out: Path, half_name: str, odors: Sequence[str] = E_ODORS, pop: int = E_POP, gens: int = E_GENS,
                 seed: int = 0) -> pd.DataFrame:
    """Part E's search on one half of the recordings: on C2 at its fitted parameters and breadth on the real wiring,
    each odor of `odors` in turn gets a per-glomerulus factor vector in [0.5, 2] on the lab's response (glomeruli without
    a response stay without), searched by a (1 + pop) evolution strategy in log-factor space (E64's: pop candidates
    simulated as pseudo-odors, the incumbent's column replaced by each candidate's for the scoring) to minimize the
    squared pattern error of the per-pair APL effect under the readout (K held at C2's) against `half_name`'s effect,
    with a penalty when the odor's response fraction leaves +-15 % of C2's in either condition. Writes
    oracle_history_<half>.csv, oracle_factors_<half>.csv and inputs/oracle_half<half>.csv (E38's layout)."""
    import E77_input_screen as S
    from E64_input_oracle import candidate_orn, out_col
    rng = np.random.default_rng(seed + (0 if half_name == 'A' else 1))
    halves = recording_halves()
    target = half_effect(halves[half_name])
    other = half_effect(halves['B' if half_name == 'A' else 'A'])
    P = S.c2_parts()
    orn, params, breadth = P['orn'], dict(P['p']), P['breadth']
    off_params = dict(params, wAPLKC_scale=0.0, wAPLPN_scale=0.0)
    K = c2_readout_K()
    r_on, s_on = (x.copy() for x in P['saved']['on'])
    r_off, s_off = (x.copy() for x in P['saved']['off'])
    ab = [H.odor_abbrev(c) for c in orn.columns]
    f0 = {o: (float(r_on[out_col(r_on, o)].astype(float).mean()), float(r_off[out_col(r_off, o)].astype(float).mean())) for o in odors}
    lo, hi = np.log(E_BOUNDS[0]), np.log(E_BOUNDS[1])
    hist, frows = [], []
    orn_now = orn.astype(float).copy()
    # the pseudo-odor path at factor 1 must give C2's own counts for the odor
    sim, _ = S.build_c2_simulator(orn=candidate_orn(orn, odors[0], np.ones((1, len(orn.index)))), breadth=breadth)
    chk_on, chk_off = sim(dict(params)), sim(dict(off_params))
    del sim
    oc0 = out_col(s_on, odors[0])
    for c_, sv in ((chk_on[1], s_on), (chk_off[1], s_off)):
        if not np.array_equal(c_.iloc[:, 0].to_numpy(dtype=float), sv[oc0].to_numpy(dtype=float)):
            raise RuntimeError(f"the pseudo-odor run of {odors[0]} at factor 1 differs from C2's saved counts")
    sse0 = _pattern_sse(s_on, s_off, K, target)
    print(f'half {half_name}: C2 pattern rms {np.sqrt(sse0 / 120):.4f} against this half, '
          f'{np.sqrt(_pattern_sse(s_on, s_off, K, other) / 120):.4f} against the other', flush=True)
    for odor in odors:
        oc = out_col(s_on, odor)
        parent = np.ones(len(orn.index))
        parent_obj = _pattern_sse(s_on, s_off, K, target)
        sigma = E_SIGMA0
        for g in range(gens):
            t0 = time.time()
            offspring = np.exp(np.clip(np.log(parent)[None, :] + sigma * rng.standard_normal((pop, len(parent))), lo, hi))
            cand = candidate_orn(orn, odor, offspring)
            sim, _ = S.build_c2_simulator(orn=cand, breadth=breadth)
            c_on, c_off = sim(dict(params)), sim(dict(off_params))
            del sim
            objs = np.empty(pop)
            parts = []
            for i in range(pop):
                so, sf = s_on.copy(), s_off.copy()
                so[oc] = c_on[1].iloc[:, i].to_numpy()
                sf[oc] = c_off[1].iloc[:, i].to_numpy()
                fo, ff = float(c_on[0].iloc[:, i].astype(float).mean()), float(c_off[0].iloc[:, i].astype(float).mean())
                pen = sum(E_PENALTY * max(0.0, abs(f / b - 1.0) - E_RATE_TOL) for f, b in ((fo, f0[odor][0]), (ff, f0[odor][1])) if b > 0)
                sse = _pattern_sse(so, sf, K, target)
                objs[i] = sse + pen
                parts.append((sse, pen, fo, ff))
            b = int(np.argmin(objs))
            success = objs[b] < parent_obj
            if success:
                parent, parent_obj = offspring[b], float(objs[b])
                for df_, c_ in ((r_on, c_on[0]), (s_on, c_on[1]), (r_off, c_off[0]), (s_off, c_off[1])):
                    df_[oc] = c_.iloc[:, b].to_numpy()
            sigma = float(np.clip(sigma * (1.5 if success else 0.8), E_SIGMA_MIN, E_SIGMA_MAX))
            sse_other = _pattern_sse(s_on, s_off, K, other)
            hist.append(dict(half=half_name, odor=odor, generation=g, best_obj=float(objs[b]), parent_obj=parent_obj, success=bool(success),
                             sigma=sigma, pattern_rms_half=float(np.sqrt(_pattern_sse(s_on, s_off, K, target) / 120)),
                             pattern_rms_other=float(np.sqrt(sse_other / 120)), frac_on=parts[b][2], frac_off=parts[b][3],
                             c2_frac_on=f0[odor][0], c2_frac_off=f0[odor][1], seconds=time.time() - t0))
            pd.DataFrame(hist).to_csv(out / f'oracle_history_{half_name}.csv', index=False)
            print(f'half {half_name} {odor} gen {g}: best {objs[b]:.4f} (parent {parent_obj:.4f}), rms this half '
                  f'{hist[-1]["pattern_rms_half"]:.4f}, other {hist[-1]["pattern_rms_other"]:.4f}, sigma {sigma:.2f}, '
                  f'{time.time() - t0:.0f}s', flush=True)
        j = ab.index(odor)
        orn_now.iloc[:, j] = orn.iloc[:, j].to_numpy(dtype=float) * parent
        for gi, glom in enumerate(orn.index):
            frows.append(dict(half=half_name, odor=odor, glomerulus=glom, orn_delta=float(orn.iloc[gi, j]), factor=float(parent[gi]),
                              log2_factor=float(np.log2(parent[gi])), searched_delta=float(orn.iloc[gi, j] * parent[gi])))
        pd.DataFrame(frows).to_csv(out / f'oracle_factors_{half_name}.csv', index=False)
        lay = H.abbrev_columns(orn_now)
        lay.index.name = 'glomerulus'
        (out / 'inputs').mkdir(exist_ok=True)
        lay.to_csv(out / 'inputs' / f'oracle_half{half_name}.csv')
    return pd.DataFrame(hist)


E_WIRINGS = ('real', 'P1', 'P2', 'M1')


def wiring_simulator(label: str, orn: pd.DataFrame, input_tag: Optional[str] = None) -> Tuple:
    """C2's family on one of E72's wirings ('real' or a draw label) at that wiring's own fitted parameters (E72's
    part-B optimum), with the ORN input `orn` and its breadth recomputed for it (E10_joint_fit.bouton_base, as a fit with
    `--orn-input` and the wiring options builds it). Returns (simulate, params, (folder, tag) of the wiring's fit)."""
    import E10_joint_fit as J
    import E72_wiring_population as E72
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    mem = {lab: (f, tg) for lab, f, tg in members()}
    folder, tag = mem[label]
    p = E72.fitted_params(folder, tag)
    wk: dict = {}
    if label != 'real':
        prefix = dict(E72.DRAWS)[label]
        seed = int(prefix.split('seed')[1].rstrip('_'))
        if label.startswith('P'):
            wk = dict(wiring='claw_permutation_within_type', wiring_seed=seed)
        else:
            m59 = H.shared_dir() / 'data' / 'model_outputs' / 'E59_coinput_shuffle'
            wk = dict(wiring_matrix=pd.read_parquet(m59 / f'wPNKC_curveball_within_type-seed{seed}_matched.parquet'),
                      wiring_tag=f'curveball_within_type_matched-seed{seed}')
    _, wnorm, W, breadth = J.bouton_base(orn, H.outputs_dir(EXPERIMENT), None, input_tag=input_tag, **wk)
    wAPLKC = J.breadth_weights(wnorm['wAPLKC'], breadth, 2.0)
    sim, _ = make_fixed_param_kc_simulator(orn, initial_params=dict(p), **J.base_kws(None), _wPNKC=W, _wAPLKC=wAPLKC,
                                           _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'],
                                           kc_kc_inh=p['wKCKC_scale'], kc_kc_tau=J.TAU)
    return sim, p, (folder, tag)


def score_oracle(out: Path, wirings: Sequence[str] = E_WIRINGS) -> pd.DataFrame:
    """Part E's reading: each half's searched input (inputs/oracle_half<X>.csv) on C2's family on `wirings`, the breadth
    recomputed for the input and both response rates re-bisected to that wiring's fit's counts (E77_input_screen's
    `hold_rates`; the lab's input is each wiring's saved fit, after a check that the rebuilt simulator reproduces it),
    scored per wiring and as the population of the four (mean distances) against the half it was searched on, the half
    it never saw, all eight recordings and the six typical ones. Writes oracle_scores.csv, oracle_runs.csv (parameters,
    factors, realism), model_pair_distances_long.csv rows (tags oracle_half<X>_<wiring>) and the runs' outputs under
    outputs/E77_input_side_common_error/oracle/."""
    import E72_wiring_population as E72
    import E77_input_screen as S
    from E10_joint_fit import read_orn_input
    halves = recording_halves()
    targets = {'halfA': half_effect(halves['A']), 'halfB': half_effect(halves['B'])}
    for t in ('z8', 'z6'):
        dd = data_distances(t)
        targets[t] = _cs(H.apl_effect(dd['on'], dd['off']))
    orn = H.apl16_orn_deltas()
    odir = H.outputs_dir(EXPERIMENT) / 'oracle'
    odir.mkdir(parents=True, exist_ok=True)
    runs: Dict[Tuple[str, str], Dict[str, tuple]] = {}
    rrows, long_parts = [], []
    for w in wirings:
        sim, p, (folder, tag) = wiring_simulator(w, orn)
        saved = E72.load_fit(folder, tag)
        on, off = sim(dict(p)), sim(dict(p, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
        for c, o in (('on', on), ('off', off)):
            if not np.array_equal(o[1].to_numpy(dtype=float), saved[c][1].to_numpy(dtype=float)):
                raise RuntimeError(f'{w}: the rebuilt simulator does not reproduce the saved {c} counts of {tag}')
        del sim
        runs[('lab', w)] = saved
        rrows.append(dict(input='lab', wiring=w, fit_folder=folder, fit_tag=tag, thr_factor=1.0, apl_factor=1.0, **p, **realism_of(saved)))
        print(f'{w}: rebuilt simulator reproduces {tag}', flush=True)
    for hx in ('A', 'B'):
        path = out / 'inputs' / f'oracle_half{hx}.csv'
        if not path.exists():
            print(f'{path} missing; half {hx} skipped', flush=True)
            continue
        o_in = read_orn_input(str(path), orn)
        for w in wirings:
            saved = runs[('lab', w)]
            n_on, n_off = int((saved['on'][0].to_numpy(dtype=float) > 0.5).sum()), int((saved['off'][0].to_numpy(dtype=float) > 0.5).sum())
            sim, p, _ = wiring_simulator(w, o_in, input_tag=f'E77_oracle_half{hx}')
            res = S.hold_rates(lambda: sim, target_off=n_off, target_on=n_on, params=p, label=f'oracle {hx} {w}')
            del sim
            o = {'on': res['on'], 'off': res['off']}
            tg = f'oracle_half{hx}_{w}'
            runs[(f'oracle_half{hx}', w)] = o
            for c, cc in (('on', 'on'), ('off', 'acute_off')):
                o[c][0].to_parquet(odir / f'{tg}_{cc}_responses.parquet')
                o[c][1].to_parquet(odir / f'{tg}_{cc}_spike_counts.parquet')
            long_parts.append(pd.concat({(tg, 'on'): H.model_rdms(*o['on'])[H.PRIMARY_MODEL_RDM],
                                         (tg, 'acute_off'): H.model_rdms(*o['off'])[H.PRIMARY_MODEL_RDM]},
                                        names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index())
            rrows.append(dict(input=f'oracle_half{hx}', wiring=w, thr_factor=res['thr_factor'], apl_factor=res['apl_factor'],
                              n_on=res['n_on'], n_off=res['n_off'], at_bound_thr=res['at_bound_thr'], at_bound_apl=res['at_bound_apl'],
                              **res['params'], **realism_of(o)))
            pd.DataFrame(rrows).to_csv(out / 'oracle_runs.csv', index=False)
            print(f'oracle {hx} on {w}: thr x{res["thr_factor"]:.3f}, APL x{res["apl_factor"]:.3f}', flush=True)
    srows = []
    inputs = sorted({k[0] for k in runs})
    for inp in inputs:
        for readout in ('cal', 'raw'):
            ds = {w: {c: _cs(v) for c, v in E72.distances(runs[(inp, w)], readout).items()} for w in wirings if (inp, w) in runs}
            if len(ds) == len(wirings):
                ds['pop4'] = {c: _cs(v) for c, v in E72.mean_distances([ds[w] for w in wirings]).items()}
            for w, d in ds.items():
                me = _cs(H.apl_effect(d['on'], d['off']))
                for tn, te in targets.items():
                    de = te.reindex(me.index)
                    err = (me - me.mean()) - (de - de.mean())
                    masks = kind_masks(err.index)
                    srows.append(dict(input=inp, wiring=w, readout=readout, target=tn, mean_effect=float(me.mean()),
                                      loss=float(np.sqrt(np.mean((me - de).to_numpy() ** 2))),
                                      pattern_rms=float(np.sqrt(np.mean(err.to_numpy() ** 2))),
                                      err_ester=float(err[masks['ester']].mean()), err_alcohol=float(err[masks['alcohol']].mean())))
    sc = pd.DataFrame(srows)
    lab = sc[sc.input == 'lab'].set_index(['wiring', 'readout', 'target'])['pattern_rms']
    sc['pattern_gain_vs_lab'] = [float(lab.get((w, r, tn), np.nan)) - v for w, r, tn, v in zip(sc.wiring, sc.readout, sc.target, sc.pattern_rms)]
    sc.to_csv(out / 'oracle_scores.csv', index=False)
    if long_parts:
        lp = out / 'model_pair_distances_long.csv'
        new = pd.concat(long_parts, ignore_index=True)
        if lp.exists():
            old = pd.read_csv(lp)
            new = pd.concat([old[~old.tag.isin(set(new.tag))], new], ignore_index=True)
        new.to_csv(lp, index=False)
    show = sc[(sc.readout == 'cal') & sc.wiring.isin(['real', 'pop4'])]
    print(show.pivot_table(index=['input', 'wiring'], columns='target', values='pattern_rms').round(4).to_string(), flush=True)
    return sc


def archive_prep(out: Path) -> pd.DataFrame:
    """Before archive_model_outputs.py: variants.csv with every saved screen point (tag C2_<point_id>, outputs under
    outputs/E77_input_side_common_error/screen/) and part E run (tag oracle_half<X>_<wiring>, outputs under .../oracle/)
    and its absolute parameters, and their pair distances (raw spike counts, the harness's primary model RDM) added to
    model_pair_distances_long.csv, for the archive's params.csv and apl_effect_scores.py."""
    odir = H.outputs_dir(EXPERIMENT)
    rows, parts = [], []
    s = pd.read_csv(out / 'screen.csv').drop_duplicates('point_id', keep='last')
    keep = ['fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale', 'thr_factor', 'apl_factor']
    for _, r in s.iterrows():
        tag = f'C2_{r.point_id}'
        rows.append(dict(tag=tag, variant=tag, part='C', point=r.point_id, kind=r['kind'], handle=r.handle, setting=r.setting,
                         value=r.get('value'), base='bouton', k=2.0, kc_kc_inh=r.wKCKC_scale, **{c: r[c] for c in keep if c in r.index},
                         rate_on=r.rate_on, rate_off=r.rate_off, note=r.get('note')))
        parts.append(('screen', tag))
    if (out / 'oracle_runs.csv').exists():
        o = pd.read_csv(out / 'oracle_runs.csv')
        for _, r in o[o.input != 'lab'].iterrows():
            tag = f'{r.input}_{r.wiring}'
            rows.append(dict(tag=tag, variant=tag, part='E', point=r.input, kind='oracle', handle='E', setting=r.wiring,
                             base='bouton', k=2.0, kc_kc_inh=r.wKCKC_scale, wiring=r.wiring,
                             orn_input=f'results/{EXPERIMENT}/inputs/{r.input}.csv', **{c: r[c] for c in keep if c in r.index},
                             rate_on=r.rate_on, rate_off=r.rate_off))
            parts.append(('oracle', tag))
    v = pd.DataFrame(rows)
    v.to_csv(out / 'variants.csv', index=False)
    lp = out / 'model_pair_distances_long.csv'
    old = pd.read_csv(lp) if lp.exists() else pd.DataFrame(columns=['tag', 'condition', 'odor_a', 'odor_b', 'distance'])
    new = []
    for sub, tag in parts:
        if tag in set(old.tag):
            continue
        d = {}
        for c in ('on', 'acute_off'):
            f = odir / sub / f'{tag}_{c}_spike_counts.parquet'
            g = odir / sub / f'{tag}_{c}_responses.parquet'
            if not (f.exists() and g.exists()):
                break
            d[c] = H.model_rdms(pd.read_parquet(g), pd.read_parquet(f))[H.PRIMARY_MODEL_RDM]
        if len(d) == 2:
            new.append(pd.concat({(tag, c): x for c, x in d.items()}, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index())
    if new:
        pd.concat([old] + new, ignore_index=True).to_csv(lp, index=False)
    print(f'{len(v)} variants; {len(new)} added to model_pair_distances_long.csv', flush=True)
    return v


def report_tables(out: Path) -> pd.DataFrame:
    """report_scalars.csv (tag, value, definition): the values the summary's passage reads that are in no table (the
    inherited share's range over half-splits, the per-odor share against response-fraction gain correlations, the two
    halves' factor agreement in part E, the screen's highest transfer reductions)."""
    hv = pd.read_csv(out / 'inherited_halves.csv')
    hc = hv[hv.readout == 'cal'].share_inherited
    po = pd.read_csv(out / 'per_odor_parts.csv')
    rows = [('inherited_halves_cal_q05', float(hc.quantile(0.05)), "5th percentile of share_inherited over the 100 splits of inherited_halves.csv, readout=cal"),
            ('inherited_halves_cal_q95', float(hc.quantile(0.95)), 'the 95th percentile of the same')]
    for tg in ('z8', 'z6'):
        g = po[(po.readout == 'cal') & (po.target == tg)]
        rows.append((f'per_odor_share_gain_corr_cal_{tg}', float(np.corrcoef(g.share, g.gain_diff)[0, 1]),
                     f'Pearson over the 16 odors between share and gain_diff in per_odor_parts.csv, readout=cal, target={tg}'))
    f = pd.concat([pd.read_csv(out / f'oracle_factors_{h}.csv') for h in 'AB'])
    w = f.pivot_table(index=['odor', 'glomerulus'], columns='half', values='log2_factor')
    cors = pd.Series({o: float(np.corrcoef(g['A'], g['B'])[0, 1]) for o, g in w.groupby(level=0)})
    d = ('over the 10 searched odors, Pearson over the 36 glomeruli between log2_factor in oracle_factors_A.csv and '
         'oracle_factors_B.csv')
    rows += [('halves_factor_corr_min', float(cors.min()), f'{d}; the minimum ({cors.idxmin()})'),
             ('halves_factor_corr_median', float(cors.median()), f'{d}; the median'),
             ('halves_factor_corr_max', float(cors.max()), f'{d}; the maximum ({cors.idxmax()})')]
    gt = pd.read_csv(out / 'gate.csv')
    rows += [('gate_transfer_all_max', float(gt.cal_transfer_all.max()),
              f'highest cal_transfer_all over the {len(gt)} rows of gate.csv ({gt.loc[gt.cal_transfer_all.idxmax(), "point_id"]})'),
             ('gate_transfer_six_max', float(gt.cal_transfer_six.max()),
              f'highest cal_transfer_six over the same rows ({gt.loc[gt.cal_transfer_six.idxmax(), "point_id"]})')]
    sc = pd.DataFrame(rows, columns=['tag', 'value', 'definition'])
    sc.to_csv(out / 'report_scalars.csv', index=False)
    print(sc[['tag', 'value']].to_string(index=False), flush=True)
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['A-input', 'A-score', 'B', 'C', 'gate', 'D', 'E', 'F', 'archive-prep', 'report-tables'])
    ap.add_argument('--half', choices=['A', 'B'], help="part E's half of the recordings to search against")
    ap.add_argument('--score', action='store_true', help="part E: score the two searched inputs (score_oracle)")
    ap.add_argument('--odors', nargs='+', default=list(E_ODORS))
    ap.add_argument('--generations', type=int, default=E_GENS)
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'A-input':
        print(searched_input(out))
    elif args.part == 'A-score':
        score_fits(out)
    elif args.part == 'B':
        for f in (pair_kinds, per_odor_parts, inherited_parts, archive_input_variants, pn_level):
            f(out)
    elif args.part == 'C':
        import E77_input_screen as S
        S.screen(out)
    elif args.part == 'gate':
        gate(out)
    elif args.part == 'archive-prep':
        archive_prep(out)
    elif args.part == 'report-tables':
        report_tables(out)
    elif args.part == 'E' and args.score:
        score_oracle(out)
    elif args.part == 'E':
        if args.half is None:
            ap.error('--part E needs --half A or B')
        input_oracle(out, args.half, odors=args.odors, gens=args.generations)
    else:
        raise NotImplementedError(f'part {args.part}')


if __name__ == '__main__':
    main()
