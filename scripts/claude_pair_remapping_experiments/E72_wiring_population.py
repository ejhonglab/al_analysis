#!/usr/bin/env python3
"""E72: the breadth-2 APL-effect fit (C2, the APL-effect fits E67) on re-drawn wirings and as a population of flies.

The fits are E10_joint_fit.py runs (part B: C2's family under `--loss apl --apl-readout both` on four within-type claw
permutations P1-P4 and the co-input-preserving shuffle's (E59) three within-type matched wirings M1-M3; part C: the
no-gain 40 Hz base on P1-P4), saved to results/E72_wiring_population/ and outputs/E72_wiring_population/. This script
scores them and builds the population:

- part A (no simulations, `--part A`): every family the archive holds on several wirings (FAMILIES), each wiring's
  APL-effect loss with its mean, per-odor and pair-specific parts and ρ on the per-pair APL effect on raw counts, under
  the calibrated readout (K per wiring, on its own APL-on counts) and binarized against the binarized data; the same
  for the mean over the wirings of the pair distances per condition (the data are a mean over recordings), against the
  number of wirings averaged; the pair-by-pair correlation of the pattern error between wirings; the split of each
  wiring's pattern error into the part common to the wirings (the error of their mean) and its own remainder;
- the new fits (`--part score`): each wiring's full row (E67_apl_effect_fits.full_row: both definitions, realism bands,
  block, the two ρ's, the natural-source alignment, odors per KC, mean distances) with its parameters, in
  stats_variants.csv; the populations (the real wiring with P1-P4, with M1-M3, all eight, the seven draws) in
  population.csv; the common-error pairs; the real wiring's rank; a two-wiring yardstick;
- part D (`--part D`, not yet written): one parameter set fitted to four wirings' population.

Outputs in results/E72_wiring_population/: archive_wirings.csv, archive_population.csv, archive_curve.csv,
archive_error_corr.csv (part A); stats_variants.csv, population.csv, curve.csv, common_error_pairs.csv,
yardstick.csv, comparison_table.md, figures, provenance_score.json.
Usage: ./venv/bin/python E72_wiring_population.py --part A | score
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E67_apl_effect_fits import full_row, odors_per_kc, refs, OPK_BINS

EXPERIMENT = 'E72_wiring_population'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
REAL = 'real'
# families the archive holds on several wirings: name -> [(wiring label, folder, tag)], the real wiring first
_E50 = [(f'{s}', 'E50_block_more_draws', f'thr40_wiring-{s}_gain1.5_rate0.068_k2_cap3') for s in
        ('claw_permutation-seed1', 'claw_permutation-seed3', 'claw_permutation-seed4', 'curveball_within_type-seed1',
         'curveball_within_type-seed3', 'curveball_within_type-seed4')]
FAMILIES = {
    'the reference set at 40 Hz, structure loss, 6.8 %': [(REAL, 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'),
        ('claw_permutation_within_type-seed2', 'E46_operating_point_redraw', 'thr40_wiring-claw_permutation_within_type-seed2_gain1.5_rate0.068_k2_cap3')] + _E50,
    'the reference set at 70 Hz, structure loss, 6.8 %': [(REAL, 'E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3'),
        ('claw_permutation_within_type-seed2', 'E46_operating_point_redraw', 'thr70_wiring-claw_permutation_within_type-seed2_gain1.5_rate0.068_k2_cap3')],
    "the first joint fit's family, breadth 2, two-matrix loss, 5 %": [(REAL, 'E10_joint_fit', 'k2'),
        ('claw_permutation_within_type-seed1', 'E30_wiring_replication', 'wiring-claw_permutation_within_type-seed1_k2'),
        ('claw_permutation_within_type-seed2', 'E30_wiring_replication', 'wiring-claw_permutation_within_type-seed2_k2')],
    "the same family with C2 as the real wiring's fit": [(REAL, 'E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3'),
        ('claw_permutation_within_type-seed1', 'E30_wiring_replication', 'wiring-claw_permutation_within_type-seed1_k2'),
        ('claw_permutation_within_type-seed2', 'E30_wiring_replication', 'wiring-claw_permutation_within_type-seed2_k2')],
    "the first joint fit's family at 40 Hz, two-matrix loss, 5 %": [(REAL, 'E10_joint_fit', 'thr40_k2'),
        ('claw_permutation_within_type-seed1', 'E30_wiring_replication', 'thr40_wiring-claw_permutation_within_type-seed1_k2'),
        ('claw_permutation_within_type-seed2', 'E30_wiring_replication', 'thr40_wiring-claw_permutation_within_type-seed2_k2')],
    "the first joint fit's family at 70 Hz, two-matrix loss, 5 %": [(REAL, 'E10_joint_fit', 'thr70_k2_fromk2'),
        ('claw_permutation_within_type-seed1', 'E30_wiring_replication', 'thr70_wiring-claw_permutation_within_type-seed1_k2'),
        ('claw_permutation_within_type-seed2', 'E30_wiring_replication', 'thr70_wiring-claw_permutation_within_type-seed2_k2')],
}
DRAWS = [('P1', 'wiring-claw_permutation_within_type-seed1_'), ('M1', 'wiring-curveball_within_type_matched-seed1_'),
         ('P2', 'wiring-claw_permutation_within_type-seed2_'), ('M2', 'wiring-curveball_within_type_matched-seed2_'),
         ('P3', 'wiring-claw_permutation_within_type-seed3_'), ('M3', 'wiring-curveball_within_type_matched-seed3_'),
         ('P4', 'wiring-claw_permutation_within_type-seed4_')]
C2_TAG = 'rate0.068_aplboth_k2_cap3'


def load_fit(folder: str, tag: str) -> Optional[Dict[str, tuple]]:
    """{'on': (responses, spike counts), 'off': ...} of one saved fit: the experiment's outputs dir, else the archive."""
    for d in (H.outputs_dir(folder), ARCHIVE / folder):
        out = {}
        for c, f in (('on', 'on'), ('off', 'acute_off')):
            ps = [Path(d) / f'{tag}_{f}_{kind}.parquet' for kind in ('responses', 'spike_counts')]
            if not all(p.exists() for p in ps):
                break
            out[c] = tuple(pd.read_parquet(p) for p in ps)
        if len(out) == 2:
            return out
    return None


def distances(o: Dict[str, tuple], readout: str) -> Dict[str, pd.Series]:
    """{'on', 'off'} pair distances of one fit under `readout`: 'raw' (cosine on counts), 'cal' (E65's saturating readout,
    K set on this fit's APL-on counts to the control flies' responders-only participation ratio), 'binary' (2 spikes)."""
    s_on, s_off = o['on'][1], o['off'][1]
    if readout == 'raw':
        return {'on': H.readout_pair_distances(s_on, 'raw'), 'off': H.readout_pair_distances(s_off, 'raw')}
    if readout == 'binary':
        return {'on': H.readout_pair_distances(s_on, 'binary', 2.0), 'off': H.readout_pair_distances(s_off, 'binary', 2.0)}
    K = H.calibrate_readout(s_on, 'saturating', refs()['cal_target'], stat='participation_ratio', responders_min=2.0)
    return {'on': H.readout_pair_distances(s_on, 'saturating', K), 'off': H.readout_pair_distances(s_off, 'saturating', K)}


def mean_distances(ds: Sequence[Dict[str, pd.Series]]) -> Dict[str, pd.Series]:
    return {c: pd.concat([d[c] for d in ds], axis=1).mean(axis=1) for c in ('on', 'off')}


def data_for(readout: str) -> pd.Series:
    return refs()['binary'] if readout == 'binary' else refs()['graded']


def effect_stats(d: Dict[str, pd.Series], readout: str) -> Dict[str, float]:
    dec, _ = H.apl_effect_decomposition(d['on'], d['off'], data_for(readout))
    return dict(loss=dec['apl_effect_rmse'], mean=dec['apl_effect_mean'], pattern=dec['apl_effect_rmse_centered'],
        per_odor=dec['apl_effect_per_odor'], pair_specific=dec['apl_effect_pair_specific'], rho=dec['apl_effect_spearman'])


def pattern_error(d: Dict[str, pd.Series], readout: str) -> pd.Series:
    """Model minus data per pair, each with its mean removed (the pattern part's error)."""
    return H.apl_effect_pair_parts(d['on'], d['off'], data_for(readout))['error']


def population_row(fits: Sequence[Dict[str, tuple]], dist: Dict[str, List[Dict[str, pd.Series]]]) -> Dict[str, float]:
    """The population: the mean over wirings of the pair distances per condition, scored under the three readouts;
    the wiring-mean per-odor response fractions' ρ with the data's; odors-per-KC shares averaged over wirings; the block
    of the mean matrices."""
    R = refs()
    row = {'n_wirings': len(fits)}
    for ro in ('raw', 'cal', 'binary'):
        md = mean_distances(dist[ro])
        row.update({f'{ro}_{k}': v for k, v in effect_stats(md, ro).items()})
        if ro in ('raw', 'cal'):
            try:
                st, _ = H.compare_on_off(md['on'], md['off'])
                row[f'{ro}_block'] = float(st['block_fruity_small_iso'])
            except Exception:
                row[f'{ro}_block'] = np.nan
            row[f'{ro}_rho_on'] = H.spearman(md['on'], R['ctrl'])
            row[f'{ro}_rho_off'] = H.spearman(md['off'], R['tnt'])
    for c, g in (('on', 'TNT3c'), ('off', 'TNT')):
        frs = pd.concat([H.response_fractions(o[c][0]).reindex(H.APL16_ODORS) for o in fits], axis=1).mean(axis=1)
        row[f'rate_{c}'] = float(frs.mean())
        row[f'rate_rho_{c}'] = H.spearman(frs, R['rates'][g])
        opk = np.mean([odors_per_kc(o[c][0]) for o in fits], axis=0)
        row.update({f'opk_{c}_{b}': float(v) for b, v in zip(OPK_BINS, opk)})
        row[f'opk_{c}_dist'] = float(0.5 * np.abs(opk - R['opk'][g]).sum())
    row['ratio'] = row['rate_off'] / max(row['rate_on'], 1e-9)
    return row


def curve_over_n(dist: Dict[str, List[Dict[str, pd.Series]]], max_subsets: int = 70) -> pd.DataFrame:
    """The population's loss and pattern part against the number of wirings averaged (all subsets up to
    `max_subsets` per n, chosen by a fixed seed if there are more): mean and SD per n."""
    n_all = len(dist['raw'])
    rng = np.random.default_rng(0)
    rows = []
    for n in range(1, n_all + 1):
        subs = list(itertools.combinations(range(n_all), n))
        if len(subs) > max_subsets:
            subs = [subs[i] for i in sorted(rng.choice(len(subs), max_subsets, replace=False))]
        for ro in ('raw', 'cal'):
            vals = [effect_stats(mean_distances([dist[ro][i] for i in sub]), ro) for sub in subs]
            for k in ('loss', 'pattern', 'per_odor', 'pair_specific'):
                v = np.array([x[k] for x in vals])
                rows.append(dict(n=n, readout=ro, stat=k, mean=float(v.mean()), sd=float(v.std()), n_subsets=len(subs)))
    return pd.DataFrame(rows)


def common_error(labels: Sequence[str], dist: Dict[str, List[Dict[str, pd.Series]]], readout: str = 'cal') -> pd.DataFrame:
    """Per pair: the pattern error of the population (common to the wirings) and each wiring's own remainder."""
    errs = {lab: pattern_error(d, readout) for lab, d in zip(labels, dist[readout])}
    common = pattern_error(mean_distances(dist[readout]), readout).rename('common')
    out = pd.concat([common] + [(e - common).rename(f'own_{lab}') for lab, e in errs.items()], axis=1)
    out['abs_common'] = out['common'].abs()
    return out.sort_values('abs_common', ascending=False)


def error_corr(labels: Sequence[str], dist: Dict[str, List[Dict[str, pd.Series]]], readout: str) -> pd.DataFrame:
    errs = pd.concat({lab: pattern_error(d, readout) for lab, d in zip(labels, dist[readout])}, axis=1)
    return errs.corr()


def two_wiring_yardstick(labels: Sequence[str], dist: Dict[str, List[Dict[str, pd.Series]]]) -> pd.DataFrame:
    """APL on from one wiring and APL off from another (all ordered pairs, on raw counts and under the readout),
    scored against the data's mean: the model's counterpart of the data's single-fly yardstick."""
    rows = []
    for ro in ('raw', 'cal'):
        for i, j in itertools.permutations(range(len(labels)), 2):
            d = {'on': dist[ro][i]['on'], 'off': dist[ro][j]['off']}
            rows.append(dict(readout=ro, on_wiring=labels[i], off_wiring=labels[j], **effect_stats(d, ro)))
    return pd.DataFrame(rows)


def family_tables(name: str, members: Sequence[tuple]) -> tuple:
    """(per-wiring rows, population row, curve, error correlations, common error) of one family."""
    loaded = []
    for lab, folder, tag in members:
        o = load_fit(folder, tag)
        if o is None:
            print(f'{name}: no saved outputs for {folder}:{tag}')
            continue
        loaded.append((lab, folder, tag, o))
    fits = [x[3] for x in loaded]
    labels = [x[0] for x in loaded]
    dist = {ro: [distances(o, ro) for o in fits] for ro in ('raw', 'cal', 'binary')}
    rows = []
    for i, (lab, folder, tag, _) in enumerate(loaded):
        r = dict(family=name, wiring=lab, folder=folder, tag=tag)
        for ro in ('raw', 'cal', 'binary'):
            r.update({f'{ro}_{k}': v for k, v in effect_stats(dist[ro][i], ro).items()})
        rows.append(r)
    pop = dict(family=name, members=','.join(labels), **population_row(fits, dist))
    curve = curve_over_n(dist).assign(family=name)
    corr = {ro: error_corr(labels, dist, ro) for ro in ('raw', 'cal')}
    common = common_error(labels, dist, 'cal')
    return rows, pop, curve, corr, common


def part_a(out: Path) -> None:
    rows, pops, curves, corr_rows = [], [], [], []
    for name, members in FAMILIES.items():
        r, pop, curve, corr, _ = family_tables(name, members)
        rows += r; pops.append(pop); curves.append(curve)
        for ro, c in corr.items():
            vals = c.to_numpy()[np.triu_indices(len(c), 1)]
            corr_rows.append(dict(family=name, readout=ro, n=len(c), corr_min=float(vals.min()) if len(vals) else np.nan,
                corr_mean=float(vals.mean()) if len(vals) else np.nan, corr_max=float(vals.max()) if len(vals) else np.nan))
        print(f'{name}: {len(r)} wirings; pattern raw single {min(x["raw_pattern"] for x in r):.3f}-{max(x["raw_pattern"] for x in r):.3f}, '
              f'mean of all {pop["raw_pattern"]:.3f}; calibrated {min(x["cal_pattern"] for x in r):.3f}-{max(x["cal_pattern"] for x in r):.3f}, '
              f'{pop["cal_pattern"]:.3f}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'archive_wirings.csv', index=False)
    pd.DataFrame(pops).to_csv(out / 'archive_population.csv', index=False)
    pd.concat(curves).to_csv(out / 'archive_curve.csv', index=False)
    pd.DataFrame(corr_rows).to_csv(out / 'archive_error_corr.csv', index=False)
    # between families on the real wiring (E67's fits)
    fits67 = pd.read_csv(H.results_dir('E67_apl_effect_fits') / 'fits.csv').tag.tolist()
    members = [(t, 'E67_apl_effect_fits', t) for t in fits67]
    _, _, _, corr, _ = family_tables('E67 fits on the real wiring', members)
    for ro, c in corr.items():
        c.to_csv(out / f'archive_error_corr_E67_{ro}.csv')


def new_fit_members(prefix_tag: str = C2_TAG, real_folder: str = 'E67_apl_effect_fits', real_tag: str = C2_TAG,
                    draws: Sequence[tuple] = DRAWS) -> List[tuple]:
    """[(label, folder, tag)] of the real wiring's fit and the part-B (or part-C) fits that exist, the lower of a draw's
    two starts where it has two (by the calibrated APL-effect loss in fits.csv)."""
    members = [(REAL, real_folder, real_tag)]
    f = pd.read_csv(H.results_dir(EXPERIMENT) / 'fits.csv').set_index('tag') if (H.results_dir(EXPERIMENT) / 'fits.csv').exists() else None
    for lab, prefix in draws:
        cands = [t for t in (f'{prefix}{prefix_tag}', f'{prefix}{prefix_tag}_s2') if f is not None and t in f.index]
        if cands:
            best = min(cands, key=lambda t: f.loc[t, 'fit_apl_cal_rmse'])
            members.append((lab, EXPERIMENT, best))
    return members


def score(out: Path) -> None:
    members = new_fit_members()
    fits = [(lab, folder, tag, load_fit(folder, tag)) for lab, folder, tag in members]
    fits = [x for x in fits if x[3] is not None]
    rows = []
    fp = H.results_dir(EXPERIMENT) / 'fits.csv'
    params = pd.read_csv(fp).set_index('tag') if fp.exists() else pd.DataFrame()
    p67 = pd.read_csv(H.results_dir('E67_apl_effect_fits') / 'fits.csv').set_index('tag')
    for lab, folder, tag, o in fits:
        p = (p67 if folder == 'E67_apl_effect_fits' else params).loc[tag]
        rows.append(dict(wiring=lab, folder=folder, tag=tag, **full_row(o),
            **{c: p[c] for c in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'kc_kc_inh', 'thr_factor',
                                 'apl_factor_vs_e3', 'aplpn_factor_vs_e3', 'loss', 'n_evals') if c in p.index},
            input_balance=float(p['wPNAPL_scale'] / p['wKCAPL_scale'])))
    st = pd.DataFrame(rows)
    st.to_csv(out / 'stats_variants.csv', index=False)
    labels = [x[0] for x in fits]
    dist = {ro: [distances(x[3], ro) for x in fits] for ro in ('raw', 'cal', 'binary')}
    pops = []
    groups = {'all': labels, 'real+P': [l for l in labels if l == REAL or l.startswith('P')],
              'real+M': [l for l in labels if l == REAL or l.startswith('M')], 'draws': [l for l in labels if l != REAL]}
    for gname, glabels in groups.items():
        idx = [labels.index(l) for l in glabels]
        if len(idx) < 2:
            continue
        sub = {ro: [dist[ro][i] for i in idx] for ro in dist}
        pops.append(dict(group=gname, members=','.join(glabels), **population_row([fits[i][3] for i in idx], sub)))
    pd.DataFrame(pops).to_csv(out / 'population.csv', index=False)
    if len(fits) >= 2:
        curve_over_n(dist).to_csv(out / 'curve.csv', index=False)
        common_error(labels, dist, 'cal').to_csv(out / 'common_error_pairs.csv')
        common_error(labels, dist, 'raw').to_csv(out / 'common_error_pairs_raw.csv')
        two_wiring_yardstick(labels, dist).to_csv(out / 'yardstick.csv', index=False)
        for ro in ('raw', 'cal'):
            error_corr(labels, dist, ro).to_csv(out / f'error_corr_{ro}.csv')
    write_table(st, pd.DataFrame(pops), out)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E72_wiring_population.py --part score',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        members=members)
    (out / 'provenance_score.json').write_text(json.dumps(prov, indent=1) + '\n')
    print(st[['wiring', 'raw_loss', 'raw_pattern', 'cal_loss', 'cal_mean', 'cal_pattern', 'cal_block', 'rate_on', 'ratio', 'rate_rho_on',
              'opk_off_dist', 'kc_kc_inh', 'thr_factor', 'realism_flag']].round(3).to_string())
    if pops:
        print(pd.DataFrame(pops)[['group', 'n_wirings', 'raw_loss', 'raw_pattern', 'cal_loss', 'cal_mean', 'cal_pattern', 'binary_loss']].round(3).to_string())


def _f(x, sp):
    try:
        return '' if x is None or (isinstance(x, float) and np.isnan(x)) else format(float(x), sp)
    except (TypeError, ValueError):
        return str(x)


def write_table(st: pd.DataFrame, pops: pd.DataFrame, out: Path) -> None:
    dt = H.data_target
    L = ['| wiring | APL-effect loss<br>raw / calibrated | mean APL effect<br>raw / calibrated | pattern part<br>raw / calibrated | '
         'per-odor / pair-specific<br>calibrated | ρ on the per-pair APL effect<br>raw / calibrated | block<br>raw / calibrated | '
         'response rate APL on;<br>ratio | response rate ρ<br>APL on | odors per KC<br>APL on / off | KC–KC | somatic threshold<br>(x base) | realism |',
         '|---|' + '---:|' * 11 + '---|']
    for _, r in st.iterrows():
        L.append(f"| {r['wiring']} | {_f(r['raw_loss'], '.3f')} / {_f(r['cal_loss'], '.3f')} | {_f(r['raw_mean'], '+.3f')} / {_f(r['cal_mean'], '+.3f')} | "
                 f"{_f(r['raw_pattern'], '.3f')} / {_f(r['cal_pattern'], '.3f')} | {_f(r['cal_per_odor'], '.3f')} / {_f(r['cal_pair_specific'], '.3f')} | "
                 f"{_f(r['raw_rho'], '.2f')} / {_f(r['cal_rho'], '.2f')} | {_f(r['raw_block'], '.3f')} / {_f(r['cal_block'], '.3f')} | "
                 f"{_f(r['rate_on'], '.3f')}; {_f(r['ratio'], '.2f')} | {_f(r['rate_rho_on'], '.2f')} | {_f(r['opk_on_dist'], '.2f')} / {_f(r['opk_off_dist'], '.2f')} | "
                 f"{_f(r.get('kc_kc_inh'), '.0f')} | {_f(r.get('thr_factor'), '.2f')} | {r['realism_flag'] if isinstance(r['realism_flag'], str) else 'inside'} |")
    for _, r in pops.iterrows():
        L.append(f"| population: {r['group']} ({int(r['n_wirings'])}) | {_f(r['raw_loss'], '.3f')} / {_f(r['cal_loss'], '.3f')} | "
                 f"{_f(r['raw_mean'], '+.3f')} / {_f(r['cal_mean'], '+.3f')} | {_f(r['raw_pattern'], '.3f')} / {_f(r['cal_pattern'], '.3f')} | "
                 f"{_f(r['cal_per_odor'], '.3f')} / {_f(r['cal_pair_specific'], '.3f')} | {_f(r['raw_rho'], '.2f')} / {_f(r['cal_rho'], '.2f')} | "
                 f"{_f(r['raw_block'], '.3f')} / {_f(r['cal_block'], '.3f')} | {_f(r['rate_on'], '.3f')}; {_f(r['ratio'], '.2f')} | {_f(r['rate_rho_on'], '.2f')} | "
                 f"{_f(r['opk_on_dist'], '.2f')} / {_f(r['opk_off_dist'], '.2f')} | | | |")
    L.append(f"| KC data | floor {dt('apl_effect_rmse_floor'):.3f} | {dt('apl_effect_mean'):+.3f} | | floors 0.020 / 0.021 | ceiling {dt('apl_effect_rho_ceiling'):.2f} | "
             f"{dt('block'):.3f} | {dt('ctrl_frac'):.3f}; {dt('rate_ratio_fraction'):.2f} | ceiling {dt('ctrl_rate_rho_ceiling'):.2f} | bands "
             f"{dt('ctrl_odors_per_kc_tv_max'):.2f} / {dt('tnt_odors_per_kc_tv_max'):.2f} | | | |")
    (out / 'comparison_table.md').write_text('\n'.join(L) + '\n')


M_FILES = {f'M{s}': (H.shared_dir() / 'data' / 'model_outputs' / 'E59_coinput_shuffle' / f'wPNKC_curveball_within_type-seed{s}_matched.parquet',
                     f'curveball_within_type_matched-seed{s}') for s in (1, 2, 3)}
TIED = ('wKCAPL_scale', 'wPNAPL_scale')
FREE = ('fixed_thr', 'wAPLKC_scale', 'wAPLPN_scale', 'wKCKC_scale')


def wiring_simulator(label: str, p_fit: Dict[str, float], orn: pd.DataFrame, out_dir: Path):
    """A fixed-parameter simulator of C2's family (breadth exponent 2, no gain, no per-claw threshold) on one wiring,
    built as E10_joint_fit.py builds the fit's model (bouton_base of that wiring, its own breadth pre-pass, the connectome
    APL weights, KC-KC inhibition with tau TAU), started at `p_fit` (that wiring's fitted absolute parameters)."""
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    import E10_joint_fit as J
    if label == REAL:
        _, wnorm, W, breadth = J.bouton_base(orn, out_dir, None)
        override = {}
    elif label.startswith('P'):
        _, wnorm, W, breadth = J.bouton_base(orn, out_dir, None, wiring='claw_permutation_within_type', wiring_seed=int(label[1:]))
        override = {'_wPNKC': W}
    else:
        path, tag = M_FILES[label]
        _, wnorm, W, breadth = J.bouton_base(orn, out_dir, None, wiring_matrix=pd.read_parquet(path), wiring_tag=tag)
        override = {'_wPNKC': W}
    kws = dict(J.base_kws(None), **override, _wAPLKC=J.breadth_weights(wnorm['wAPLKC'], breadth, 2.0), _wKCAPL=wnorm['wKCAPL'],
        _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=J.TAU)
    sim, _ = make_fixed_param_kc_simulator(orn, initial_params=dict(p_fit), **kws)
    return sim


def fitted_params(folder: str, tag: str) -> Dict[str, float]:
    f = pd.read_csv(H.results_dir(folder) / 'fits.csv').set_index('tag').loc[tag]
    p = {k: float(f[k]) for k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')}
    p['wKCKC_scale'] = float(f['kc_kc_inh'])
    return p


class PopulationLoss:
    """E67's 'both' APL-effect loss (E10_joint_fit.make_loss variant 'apl', readout 'both') computed on a population of
    wirings: the two APL-effect terms (raw counts; the calibrated readout with K per wiring) and the two matrix terms
    on the mean over wirings of the pair distances per condition; the operating point, the rate ratio with its wall
    and the per-odor terms on the wiring-mean response rates and fractions. With one wiring it is E10's loss."""

    def __init__(self):
        import E10_joint_fit as J
        self.J = J
        self.w = J.loss_weights('apl')
        self.rate_target = float(H.data_target('ctrl_frac'))
        data_rates = {g: H.yang_response_fractions(g).mean() for g in ('TNT3c', 'TNT')}
        self.rates = {g: data_rates[g].reindex(H.APL16_ODORS) for g in data_rates}
        self.ratio_target = float(self.rates['TNT'].mean() / self.rates['TNT3c'].mean())
        self.ratio_lo, self.ratio_hi = H.data_target_ci('rate_ratio_fraction')
        self.floor = H.data_target('apl_effect_rmse_floor')
        self.cal_target = H.data_target('ctrl_participation_ratio_resp')
        self.ctrl, self.tnt = H.yang_mean_rdm('TNT3c', 'cosine'), H.yang_mean_rdm('TNT', 'cosine')
        self.effect = H.yang_apl_effect()

    def __call__(self, outs: Sequence[Dict[str, tuple]]) -> Dict[str, float]:
        from scipy import stats
        from E3_drive_dependent_gain import rate
        J, v = self.J, H.PRIMARY_MODEL_RDM
        raw = {c: pd.concat([H.model_rdms(*o[c])[v] for o in outs], axis=1).mean(axis=1) for c in ('on', 'off')}
        cal_on, cal_off = [], []
        for o in outs:
            K = H.calibrate_readout(o['on'][1], 'saturating', self.cal_target, stat='participation_ratio', responders_min=2.0)
            cal_on.append(H.readout_pair_distances(o['on'][1], 'saturating', K))
            cal_off.append(H.readout_pair_distances(o['off'][1], 'saturating', K))
        cal = {'on': pd.concat(cal_on, axis=1).mean(axis=1), 'off': pd.concat(cal_off, axis=1).mean(axis=1)}
        out = {}
        out['on_spearman'] = H.spearman(raw['on'], self.ctrl)
        out['off_spearman'] = H.spearman(raw['off'], self.tnt)
        out['rate_on'] = float(np.mean([rate(o['on'][0]) for o in outs]))
        out['rate_off'] = float(np.mean([rate(o['off'][0]) for o in outs]))
        out['ratio'] = out['rate_off'] / max(out['rate_on'], 1e-9)
        po = {c: pd.concat([H.response_fractions(o[c][0]).reindex(H.APL16_ODORS) for o in outs], axis=1).mean(axis=1) for c in ('on', 'off')}
        out['po_on_spearman'] = float(stats.spearmanr(po['on'], self.rates['TNT3c'])[0])
        out['po_off_spearman'] = float(stats.spearmanr(po['off'], self.rates['TNT'])[0])
        for name, d in (('raw', raw), ('cal', cal)):
            dec, _ = H.apl_effect_decomposition(d['on'], d['off'], self.effect)
            out.update({f'apl_{name}_rmse': dec['apl_effect_rmse'], f'apl_{name}_mean': dec['apl_effect_mean'],
                f'apl_{name}_pattern': dec['apl_effect_rmse_centered'], f'apl_{name}_rho': dec['apl_effect_spearman']})
        terms = dict(
            on_mat=1.0 - out['on_spearman'],
            off_mat=1.0 - out['off_spearman'],
            rate_on=((out['rate_on'] - self.rate_target) / J.RATE_SCALE) ** 2,
            ratio=((np.log(max(out['ratio'], 1e-3)) - np.log(self.ratio_target)) / J.RATIO_LOG_SCALE) ** 2,
            po_on=1.0 - np.nan_to_num(out['po_on_spearman'], nan=0.0),
            po_off=1.0 - np.nan_to_num(out['po_off_spearman'], nan=0.0),
            apl_raw=out['apl_raw_rmse'] / self.floor,
            apl_cal=out['apl_cal_rmse'] / self.floor,
            ratio_wall=(max(0.0, self.ratio_lo - out['ratio'], out['ratio'] - self.ratio_hi) / J.RATIO_WALL_SCALE) ** 2,
            po_wall=(max(0.0, J.PO_WALL - np.nan_to_num(out['po_on_spearman'], nan=0.0)) / J.PO_WALL_SCALE) ** 2)
        for name, t in terms.items():
            out[f'term_{name}'] = float(self.w[name] * t)
        out['loss'] = float(sum(out[f'term_{n}'] for n in terms))
        return out


def simulate_population(sims: Dict[str, object], ratios: Dict[str, tuple], theta: Dict[str, float]) -> List[Dict[str, tuple]]:
    """Every wiring at the shared parameters `theta` (FREE), its KC->APL and bouton->APL scales tied to APL->KC by that
    wiring's own fitted ratios; APL off: APL->KC and APL->bouton at 0."""
    outs = []
    for lab, sim in sims.items():
        r_kc, r_pn = ratios[lab]
        p = dict(theta, wKCAPL_scale=theta['wAPLKC_scale'] * r_kc, wPNAPL_scale=theta['wAPLKC_scale'] * r_pn)
        outs.append({'on': sim(p), 'off': sim(dict(p, wAPLKC_scale=0.0, wAPLPN_scale=0.0))})
    return outs


def population_fit(wirings: Sequence[str], held_out: Sequence[str], maxfev: int = 150, check_only: bool = False) -> Dict[str, float]:
    """Part D: one set of the four parameters (absolute: somatic threshold, APL->KC, APL->bouton, KC-KC) fitted to the
    population of `wirings` under E67's 'both' loss computed on the population (PopulationLoss), Nelder-Mead in
    log(parameter / start) from the median of the wirings' part-B optima, then run once on `held_out` and scored.
    First checks that with the real wiring alone at C2's parameters the loss is C2's saved loss to the last digit."""
    from scipy.optimize import minimize
    out_dir = H.outputs_dir(EXPERIMENT)
    res_dir = H.results_dir(EXPERIMENT)
    orn = H.apl16_orn_deltas()
    members = {lab: (folder, tag) for lab, folder, tag in new_fit_members()}
    loss = PopulationLoss()
    # the check: the real wiring alone at C2's fitted parameters reproduces E67's saved loss
    p_c2 = fitted_params('E67_apl_effect_fits', C2_TAG)
    sim_real = wiring_simulator(REAL, p_c2, orn, out_dir)
    o = {'on': sim_real(p_c2), 'off': sim_real(dict(p_c2, wAPLKC_scale=0.0, wAPLPN_scale=0.0))}
    got = loss([o])['loss']
    saved = float(pd.read_csv(H.results_dir('E67_apl_effect_fits') / 'fits.csv').set_index('tag').loc[C2_TAG, 'loss'])
    print(f'check: one wiring at C2 -> loss {got!r}, saved {saved!r}', flush=True)
    if got != saved:
        raise RuntimeError(f'the population loss with one wiring does not reproduce C2 ({got!r} vs {saved!r})')
    if check_only:
        return {'check_loss': got, 'saved_loss': saved}
    sims, ratios, starts = {}, {}, []
    for lab in list(wirings) + list(held_out):
        folder, tag = members[lab]
        p = fitted_params(folder, tag)
        sims[lab] = sim_real if lab == REAL else wiring_simulator(lab, p, orn, out_dir)
        ratios[lab] = (p['wKCAPL_scale'] / p['wAPLKC_scale'], p['wPNAPL_scale'] / p['wAPLKC_scale'])
        if lab in wirings:
            starts.append([p[k] for k in FREE])
    x0 = np.median(np.array(starts), axis=0)
    start = dict(zip(FREE, x0))
    fit_sims = {lab: sims[lab] for lab in wirings}
    hist = []

    def f(z):
        theta = {k: float(start[k] * np.exp(zz)) for k, zz in zip(FREE, z)}
        out = loss(simulate_population(fit_sims, ratios, theta))
        hist.append(dict(theta, **out))
        if len(hist) % 10 == 0:
            print(f"eval {len(hist)}: loss {out['loss']:.4f} (readout {out['apl_cal_rmse']:.4f}, raw {out['apl_raw_rmse']:.4f})", flush=True)
            pd.DataFrame(hist).to_csv(res_dir / 'population_fit_history.csv', index=False)
        return out['loss']

    simplex = np.vstack([np.zeros(len(FREE))] + [0.15 * np.eye(len(FREE))[i] for i in range(len(FREE))])
    res = minimize(f, np.zeros(len(FREE)), method='Nelder-Mead', options=dict(maxfev=maxfev, initial_simplex=simplex, xatol=1e-3, fatol=1e-4))
    pd.DataFrame(hist).to_csv(res_dir / 'population_fit_history.csv', index=False)
    best = min(hist, key=lambda h: h['loss'])
    theta = {k: best[k] for k in FREE}
    # held-out: run once at the fitted set, score the held-out population and each held-out wiring
    held = {lab: sims[lab] for lab in held_out}
    outs_held = simulate_population(held, ratios, theta)
    outs_fit = simulate_population(fit_sims, ratios, theta)
    for lab, oo in zip(list(fit_sims) + list(held), outs_fit + outs_held):
        for c, cc in (('on', 'on'), ('off', 'acute_off')):
            oo[c][0].to_parquet(out_dir / f'popfit_{lab}_{cc}_responses.parquet')
            oo[c][1].to_parquet(out_dir / f'popfit_{lab}_{cc}_spike_counts.parquet')
    dist_h = {ro: [distances(o, ro) for o in outs_held] for ro in ('raw', 'cal', 'binary')}
    dist_f = {ro: [distances(o, ro) for o in outs_fit] for ro in ('raw', 'cal', 'binary')}
    result = dict(start=start, theta=theta, n_evals=len(hist), loss=best['loss'], wirings=list(wirings), held_out=list(held_out),
        fitted_population={k: v for k, v in population_row(outs_fit, dist_f).items()},
        held_out_population={k: v for k, v in population_row(outs_held, dist_h).items()},
        held_out_loss=loss(outs_held), check_loss=got, ratios={k: list(v) for k, v in ratios.items()},
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        model=H.model_provenance(orn_deltas=orn))
    (res_dir / 'population_fit.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
    print(json.dumps({k: result[k] for k in ('theta', 'loss', 'n_evals')}, indent=1, default=float))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', default='score', choices=['A', 'score', 'D', 'D-check'])
    ap.add_argument('--maxfev', type=int, default=150)
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'A':
        part_a(out)
    elif args.part == 'score':
        score(out)
    else:
        population_fit([REAL, 'P1', 'P2', 'M1'], ['P3', 'P4', 'M2', 'M3'], maxfev=args.maxfev, check_only=(args.part == 'D-check'))


if __name__ == '__main__':
    main()
