#!/usr/bin/env python3
"""E65: the compressive readout of the model's spike counts. How much of the APL-effect loss is
the comparison of raw spike counts with a saturating calcium signal (no simulations).

The main target since 2026-11-09 17:55 is the per-pair APL effect on odor-odor distances (each pair's
APL-on minus APL-off distance) against the KC data's, scored by the APL-effect loss
(`kc_remapping_harness.apl_effect_stats`). The model's side is a cosine distance on raw spike counts,
the data's on z-scored nuclear GCaMP6s fluorescence (trial means), which compresses large responses.
This script re-scores the archived spike counts of every saved variant under readouts
(`kc_remapping_harness.readout`) and compares the response amplitudes' concentration in the model and
the data:

Part A, the data (Yang's APL16 recordings, `yang_trialmeans`): per recording and genotype, over the
16 odors, the concentration of each odor's response vector (`amplitude_stats`: the share of the
squared vector in the top 1 % / 5 % of KCs, the participation ratio), computed on all values (as the
cosine sees them) and on responders only (values above 1, others 0; like-for-like with the model's
counts, whose non-responders are 0 or 1 spike); the share of the squared vector carried by
sub-threshold (<= 1) and by negative values; the responders' amplitude distribution (mean, median,
coefficient of variation, 90th / 50th percentile); the number of odors each KC responds to.

Part B, the models (every variant with archived `<tag>_on_spike_counts.parquet` and
`<tag>_acute_off_spike_counts.parquet` in data/model_outputs/<folder>/; 729 of 733 scored variants on
2026-11-09): the same concentration statistics of the raw counts, and the APL-effect statistics under
each readout of READOUTS: raw counts (the target as defined); binarized at two spikes (the model's
`responses`), against the binarized data's effect and against the graded data's; fixed one-parameter
readouts (power n**p, saturating n / (n + k)) over a range of the parameter; and calibrated readouts
whose parameter is fixed per model on its APL-on counts alone, without reference to the APL effect,
then used unchanged for APL off: the power and saturating readouts calibrated to the control flies'
responder-only participation ratio and top-1 % share, and a rank-matching map onto the control flies'
responder amplitudes (`calibrate_readout`, `rank_match_map`). Under each readout also the two ρ's on
odor-odor distances with the two KC matrices, the mean distances, the fruity x small-ester block and
the natural-source alignment (APL on / off / on minus off).

Part C, the tables and figures, written to <shared>/results/E65_spike_count_readout/:
data_amplitude.csv (per recording), data_amplitude_summary.csv (per genotype), model_amplitude.csv (per
variant, raw counts), by_variant_readout.csv (one row per variant x readout), readout_summary.csv (per
readout, over the realistic variants), key_models.csv and comparison_table.md (the summary's key models
under each readout), and the figures E65_readout_params.png (loss and mean APL effect against the
readout parameter for the key models), E65_loss_raw_vs_calibrated.png (every variant), E65_concentration.png
(the data's and the models' concentration statistics), E65_odors_per_kc.png; provenance.json.

Realism (`apl_effect_scores.realism_flag`, CLAUDE.md) is a property of the model's response rates, which
no readout changes, so each variant keeps the flag apl_effect_scores.py gave it. Nothing is simulated.
Usage: ./venv/bin/python E65_spike_count_readout.py [--jobs N] [--only-key]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E65_spike_count_readout'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
RESULTS = H.shared_dir() / 'results'
# fixed readouts: (name, kind, param, data): data 'graded' (the KC data's effect) or 'binary' (the
# binarized data's effect)
FIXED = ([('raw', 'raw', None, 'graded'), ('binary', 'binary', 2.0, 'binary'), ('binary_vs_graded', 'binary', 2.0, 'graded'),
          ('sqrt', 'sqrt', None, 'graded')]
         + [(f'power_{p:g}', 'power', p, 'graded') for p in (0.25, 0.33, 0.75)]
         + [(f'saturating_{k:g}', 'saturating', k, 'graded') for k in (1, 2, 5, 10, 20, 50)])
# calibrated readouts: (name, kind, calibration target). The targets are the control flies' (part A):
# 'pr' and 'top1', the participation ratio and the top-1 % share on responders only (trial means <= 1
# set to 0), matched by the model's readout with its non-responders (< 2 spikes) zeroed; 'cv', the
# coefficient of variation of the responders' amplitudes; 'pr_all', the participation ratio on all
# values (as the cosine sees the data), matched by the model's readout on all its values
CALIBRATED = [(f'{kind}_cal_{t}', kind, t) for t in ('pr', 'top1', 'cv', 'pr_all') for kind in ('power', 'saturating')] + [
    ('rank_match', 'rank_match', None)]
CAL_STATS = {'pr': ('participation_ratio', 2.0), 'top1': ('top1_share', 2.0), 'cv': ('responder_cv', 2.0), 'pr_all': ('participation_ratio', None)}
# the data's per-pair APL effect, as the KC data are read: 'graded' (the target as defined), 'thresholded'
# (trial means <= 1 set to 0), 'noise_corrected' (each recording corrected for trial-to-trial noise),
# 'binary' (trial mean > 1); every readout is scored against the first three, the binary readout also
# against the last
DATA_VERSIONS = ('graded', 'thresholded', 'noise_corrected', 'binary')
# the summary's key models: (label, folder, tag)
KEY_MODELS = [
    ('the reference model (eight at 1.5, 40 Hz, 6.8 %)', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'),
    ('eight at 1.5, 70 Hz, 6.8 %', 'E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3'),
    ('the reference model until 2026-11-08 (40 Hz, 5 %)', 'E34_gain_bounded', 'thr40_gain1.5_k2_cap3'),
    ('the first joint fit (k 2, 5 %)', 'E10_joint_fit', 'k2'),
    ('the joint fit with k 3 (5 %)', 'E10_joint_fit', 'k3'),
    ('the gain on DM1 alone (70 Hz, 5 %)', 'E39_gain_decomposition', 'thr70__only_DM1'),
    ('breadth 3 with KC-KC 30', 'E8_kc_kc_inhibition', 'breadth3_tau0.3_kckc30'),
    ('the no-gain two-ρ fit (40 Hz, 6.8 %)', 'E63_rho_fit', 'thr40_rate0.068_rho4_k2_cap3'),
    # pass 2's archived counts were renamed *.invalid on the share (2026-11-09 19:07), so pass 1
    ('the searched weak-odor input, pass 1 (oracle)', 'E64_input_oracle', 'I_searched_pass1'),
]
V = 'spike_counts_cosine'


def recording_matrices(genotype: str) -> Dict[tuple, pd.DataFrame]:
    """{(date, fly_num, hemibrain): KCs x odors trial-mean matrix} of one genotype (the 16 odors)."""
    tm = H.yang_trialmeans()
    tm = tm[(tm.genotype == genotype) & tm.odor.isin(H.APL16_ODORS)]
    out = {}
    for rec, g in tm.groupby(['date', 'fly_num', 'hemibrain']):
        out[rec] = g.pivot_table(index='roi', columns='odor', values='value')[H.APL16_ODORS]
    return out


def responder_only(mat: pd.DataFrame, thresh: float) -> pd.DataFrame:
    return mat.where(mat > thresh, 0.0)


def odors_per_unit(resp: pd.DataFrame) -> Dict[str, float]:
    """Fractions of units responding to 0, 1, 2-3, 4-7 and 8 or more odors (`resp` boolean)."""
    n = resp.sum(axis=1)
    bins = {'odors_0': n == 0, 'odors_1': n == 1, 'odors_2_3': (n >= 2) & (n <= 3), 'odors_4_7': (n >= 4) & (n <= 7), 'odors_8plus': n >= 8}
    return {k: float(v.mean()) for k, v in bins.items()}


def responder_amplitudes(values: np.ndarray) -> Dict[str, float]:
    v = values[np.isfinite(values)]
    if not len(v):
        return dict(resp_amp_mean=np.nan, resp_amp_median=np.nan, resp_amp_cv=np.nan, resp_amp_p90_over_p50=np.nan)
    return dict(resp_amp_mean=float(v.mean()), resp_amp_median=float(np.median(v)), resp_amp_cv=float(v.std() / v.mean()),
        resp_amp_p90_over_p50=float(np.percentile(v, 90) / np.percentile(v, 50)))


def part_a() -> tuple:
    """Per-recording amplitude statistics of the KC data, their per-genotype means, and the control
    flies' pooled responder amplitudes (the rank-matching reference)."""
    rows, ref = [], None
    for genotype in ('TNT3c', 'TNT'):
        pooled = []
        for rec, m in recording_matrices(genotype).items():
            a_all = H.amplitude_stats(m).mean(numeric_only=True)
            a_resp = H.amplitude_stats(responder_only(m, 1.0)).mean(numeric_only=True)
            x = m.to_numpy(dtype=float)
            sq = x ** 2
            tot = np.nansum(sq)
            resp_vals = x[x > 1.0]
            pooled.append(resp_vals)
            rows.append(dict(genotype=genotype, date=rec[0], fly_num=rec[1], hemibrain=rec[2], n_kcs=m.shape[0],
                top1_share_all=a_all['top1_share'], top5_share_all=a_all['top5_share'], participation_ratio_all=a_all['participation_ratio'],
                top1_share_resp=a_resp['top1_share'], top5_share_resp=a_resp['top5_share'], participation_ratio_resp=a_resp['participation_ratio'],
                share_subthreshold=float(np.nansum(sq[x <= 1.0]) / tot), share_negative=float(np.nansum(sq[x < 0]) / tot),
                response_rate=float((x > 1.0).mean()), **responder_amplitudes(resp_vals), **odors_per_unit(m > 1.0)))
        if genotype == 'TNT3c':
            ref = np.concatenate(pooled)
    df = pd.DataFrame(rows)
    summ = df.drop(columns=['date', 'fly_num', 'hemibrain']).groupby('genotype').mean(numeric_only=True)
    return df, summ, ref


_CACHE: dict = {}


def _data_refs():
    if not _CACHE:
        for v in ('graded', 'thresholded', 'noise_corrected'):
            _CACHE[v] = H.yang_apl_effect(values=v)
        _CACHE['binary'] = H.yang_apl_effect(binarize=1.0)
        _CACHE['ctrl'] = H.yang_mean_rdm('TNT3c')
        _CACHE['tnt'] = H.yang_mean_rdm('TNT')
        _CACHE['ref66'] = H.reference_pairs()
    return _CACHE


def score_readout(on: pd.DataFrame, off: pd.DataFrame, kind: str, param, data: str) -> dict:
    refs = _data_refs()
    d_on, d_off = H.readout_pair_distances(on, kind, param), H.readout_pair_distances(off, kind, param)
    st = H.apl_effect_stats(d_on, d_off, refs[data])
    for v in DATA_VERSIONS:
        if v == 'binary' and data != 'binary':
            continue
        sv = H.apl_effect_stats(d_on, d_off, refs[v])
        st.update({f'vs_{v}__rmse': sv['apl_effect_rmse'], f'vs_{v}__rmse_centered': sv['apl_effect_rmse_centered'],
            f'vs_{v}__spearman': sv['apl_effect_spearman']})
    st['mean_dist_on'], st['mean_dist_off'] = float(d_on.mean()), float(d_off.mean())
    st['rho_on'] = H.compare_model_to_data(d_on, refs['ctrl'])['spearman']
    st['rho_off'] = H.compare_model_to_data(d_off, refs['tnt'])['spearman']
    oo, _ = H.compare_on_off(d_on, d_off)
    st['block'] = oo['block_fruity_small_iso']
    st['vcf_on'] = H.compare_to_references(d_on, refs['ref66'])['spearman_resid_vs_vcf_hyp']
    st['vcf_off'] = H.compare_to_references(d_off, refs['ref66'])['spearman_resid_vs_vcf_hyp']
    return st


def score_variant(job: tuple) -> List[dict]:
    """All readouts of one archived variant: [row per readout], plus a raw-counts concentration row
    under readout name '_concentration'."""
    folder, tag, targets, ref = job
    d = ARCHIVE / folder
    on = pd.read_parquet(d / f'{tag}_on_spike_counts.parquet')
    off = pd.read_parquet(d / f'{tag}_acute_off_spike_counts.parquet')
    rows = []
    base = dict(folder=folder, tag=tag)
    conc = dict(base, readout='_concentration')
    for cond, m in (('on', on), ('off', off)):
        a = H.amplitude_stats(m).mean(numeric_only=True)
        a_r = H.amplitude_stats(m.where(m >= 2, 0)).mean(numeric_only=True)
        conc.update({f'top1_share_{cond}': a['top1_share'], f'top5_share_{cond}': a['top5_share'],
            f'participation_ratio_{cond}': a['participation_ratio'], f'top1_share_resp_{cond}': a_r['top1_share'],
            f'participation_ratio_resp_{cond}': a_r['participation_ratio']})
        x = m.to_numpy(dtype=float)
        conc.update({f'{k}_{cond}': v for k, v in responder_amplitudes(x[x >= 2]).items()})
        conc.update({f'{k}_{cond}': v for k, v in odors_per_unit(m >= 2).items()})
    conc['max_count_on'], conc['max_count_off'] = float(on.to_numpy().max()), float(off.to_numpy().max())
    rows.append(conc)
    for name, kind, param, data in FIXED:
        rows.append(dict(base, readout=name, kind=kind, param=param, data=data, **score_readout(on, off, kind, param, data)))
    for name, kind, t in CALIBRATED:
        if kind == 'rank_match':
            param = H.rank_match_map(on, ref)
            pval = np.nan
        else:
            stat, resp_min = CAL_STATS[t]
            param = H.calibrate_readout(on, kind, targets[t], stat=stat, responders_min=resp_min)
            pval = param
        r = dict(base, readout=name, kind=kind, param=pval, data='graded', **score_readout(on, off, kind, param, 'graded'))
        if kind != 'rank_match':
            r['calibrated_stat'] = H.readout_stat(on, kind, param, stat, resp_min)
            r['calibration_target'] = targets[t]
            r['at_bound'] = bool(abs(r['calibrated_stat'] - targets[t]) > 0.02 * abs(targets[t]))
            r['calibrated_stat_off'] = H.readout_stat(off, kind, param, stat, resp_min)
        rows.append(r)
    return rows


def readout_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per readout, over the realistic variants: medians of the loss and its parts, the share of variants
    whose loss falls below their raw-count loss, the Spearman of the variants' order by loss against the
    raw order, and the best variant."""
    raw = df[df.readout == 'raw'].set_index(['folder', 'tag'])
    out = []
    for name, g in df[df.readout != '_concentration'].groupby('readout', sort=False):
        g = g.set_index(['folder', 'tag'])
        real = g[g['realism_flag'].fillna('') == '']
        r0 = raw.loc[real.index]
        best = real['apl_effect_rmse'].idxmin()
        out.append(dict(readout=name, n_variants=len(g), n_realistic=len(real),
            median_loss=float(real['apl_effect_rmse'].median()), median_mean_effect=float(real['apl_effect_mean'].median()),
            median_pattern_part=float(real['apl_effect_rmse_centered'].median()), median_rho_apl_effect=float(real['apl_effect_spearman'].median()),
            median_loss_vs_thresholded=float(real['vs_thresholded__rmse'].median()),
            median_loss_vs_noise_corrected=float(real['vs_noise_corrected__rmse'].median()),
            share_improved=float((real['apl_effect_rmse'] < r0['apl_effect_rmse']).mean()),
            order_rho_vs_raw=H.spearman(real['apl_effect_rmse'], r0['apl_effect_rmse']),
            best_folder=best[0], best_tag=best[1], best_loss=float(real.loc[best, 'apl_effect_rmse']),
            best_mean_effect=float(real.loc[best, 'apl_effect_mean']), best_rho=float(real.loc[best, 'apl_effect_spearman'])))
    return pd.DataFrame(out)


def order_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman between readouts of the realistic variants' APL-effect losses (each against the graded
    data, the binary readout against the binarized data): how much a readout reorders the models."""
    d = df[(df.readout != '_concentration') & (df['realism_flag'].fillna('') == '')]
    w = d.pivot_table(index=['folder', 'tag'], columns='readout', values='apl_effect_rmse')
    order = [r for r in d.readout.unique()]
    return w[order].corr(method='spearman')


NOISE_TABLES = ['E43_model_trial_noise/partB', 'E43_model_trial_noise/partB_lognormal', 'E48_block_target_noise/partB',
    'E51_noise_robust_block/partB', 'E53_kckc_held_block/partB', 'E58_gain_noise_block/partB', 'E63_rho_fit/partB']


def noise_apl_effect() -> pd.DataFrame:
    """The APL-effect loss through the trial-noise pipeline (the model's trial-to-trial variability
    experiment, E43, and the experiments that reused it), from the saved pairs_noise_long.csv tables:
    per model, noise level and transform ('graded', or 'thresholded': values <= 1 z set to 0), the
    model's per-pair effect from its noisy three-trial-mean distances ('raw', against the data's raw
    distances in the same transform), from its noise-corrected distances ('corrected', against the
    data's corrected ones), from its deterministic distances ('det', against the data's raw ones) and from
    the grand mean over trials ('grand', against the data's corrected ones)."""
    rows = []
    for rel in NOISE_TABLES:
        p = RESULTS / rel / 'pairs_noise_long.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p)
        for (model, level, transform), g in d.groupby(['model', 'level', 'transform']):
            on = g[g.condition == 'on'].set_index(['odor_a', 'odor_b'])
            off = g[g.condition == 'acute_off'].set_index(['odor_a', 'odor_b'])
            if on.empty or off.empty:
                continue
            for col, dcol in (('raw', 'data_raw'), ('corrected', 'data_corrected'), ('det', 'data_raw'), ('grand', 'data_corrected')):
                data_eff = H.apl_effect(on[dcol], off[dcol])
                st = H.apl_effect_stats(on[col], off[col], data_eff)
                rows.append(dict(source=rel, model=model, level=level, noise_sd=float(g['sigma' if 'sigma' in g else 'noise_sd'].iloc[0]), transform=transform, distances=col,
                    data_distances=dcol, data_effect_mean=float(data_eff.mean()), **st))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--jobs', type=int, default=12, help='worker processes for part B')
    ap.add_argument('--only-key', action='store_true', help='part B on the key models only (a quick check)')
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    dt = H.data_target
    # part A
    data_df, data_summ, ref = part_a()
    data_df.to_csv(out / 'data_amplitude.csv', index=False)
    data_summ.to_csv(out / 'data_amplitude_summary.csv')
    c = data_summ.loc['TNT3c']
    targets = {'pr': float(c['participation_ratio_resp']), 'top1': float(c['top1_share_resp']), 'cv': float(c['resp_amp_cv']),
               'pr_all': float(c['participation_ratio_all'])}
    print('calibration targets (control flies):', {k: round(v, 4) for k, v in targets.items()}, flush=True)
    # part B
    bv = pd.read_csv(RESULTS / 'apl_effect' / 'by_variant.csv')
    have = [(r.folder, r.tag) for r in bv.itertuples()
        if (ARCHIVE / r.folder / f'{r.tag}_on_spike_counts.parquet').exists() and (ARCHIVE / r.folder / f'{r.tag}_acute_off_spike_counts.parquet').exists()]
    if args.only_key:
        keys = {(f, t) for _, f, t in KEY_MODELS}
        have = [h for h in have if h in keys]
    jobs = [(f, t, targets, ref) for f, t in have]
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, res in enumerate(ex.map(score_variant, jobs, chunksize=4)):
            rows.extend(res)
            if (i + 1) % 100 == 0:
                print(f'{i + 1} / {len(jobs)} variants', flush=True)
    df = pd.DataFrame(rows)
    flags = bv.set_index(['folder', 'tag'])[['exp', 'rate_on', 'rate_off', 'spearman_rate_on_vs_data', 'realism_flag']]
    df = df.join(flags, on=['folder', 'tag'])
    df.to_csv(out / 'by_variant_readout.csv', index=False)
    model_conc = df[df.readout == '_concentration'].dropna(axis=1, how='all').copy()
    model_conc['realism_flag'] = df.loc[model_conc.index, 'realism_flag']
    model_conc.to_csv(out / 'model_amplitude.csv', index=False)
    summ = readout_summary(df)
    summ.to_csv(out / 'readout_summary.csv', index=False)
    order_matrix(df).to_csv(out / 'readout_order_rho.csv')
    noise = noise_apl_effect()
    noise.to_csv(out / 'noise_apl_effect.csv', index=False)
    # key models
    rows_k = []
    for label, folder, tag in KEY_MODELS:
        g = df[(df.folder == folder) & (df.tag == tag) & (df.readout != '_concentration')]
        for _, r in g.iterrows():
            rows_k.append(dict(model=label, folder=folder, tag=tag, readout=r['readout'], param=r.get('param'),
                **{k: r[k] for k in ('apl_effect_rmse', 'apl_effect_mean', 'apl_effect_rmse_centered', 'apl_effect_spearman',
                'mean_dist_on', 'mean_dist_off', 'rho_on', 'rho_off', 'block', 'vcf_on', 'vcf_off', 'realism_flag')},
                **{k: r[k] for k in r.index if k.startswith('vs_') or k in ('calibrated_stat', 'calibrated_stat_off', 'at_bound')}))
    key = pd.DataFrame(rows_k)
    key.to_csv(out / 'key_models.csv', index=False)
    write_table(key, df, data_summ, out, noise)
    make_figures(df, key, data_df, data_summ, model_conc, targets, out)
    inputs = {str(RESULTS / 'apl_effect' / 'by_variant.csv'): hashlib.sha256((RESULTS / 'apl_effect' / 'by_variant.csv').read_bytes()).hexdigest()}
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E65_spike_count_readout.py',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        inputs_sha256=inputs, n_variants=len(have), readouts_fixed=[f[0] for f in FIXED], readouts_calibrated=[c[0] for c in CALIBRATED],
        calibration_targets=targets, data_effect_mean=float(H.yang_apl_effect().mean()),
        data_effect_binary_mean=float(H.yang_apl_effect(binarize=1.0).mean()))
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n')
    print(summ.round(3).to_string())
    print('->', out)


def write_table(key: pd.DataFrame, df: pd.DataFrame, data_summ: pd.DataFrame, out: Path, noise: pd.DataFrame) -> None:
    dt = H.data_target
    shown = ['raw', 'binary', 'sqrt', 'power_cal_pr', 'saturating_cal_pr', 'saturating_cal_cv', 'saturating_cal_pr_all', 'rank_match']
    lines = ['| model | ' + ' | '.join(shown) + ' |', '|---|' + '---:|' * len(shown)]
    for label, folder, tag in KEY_MODELS:
        g = key[(key.folder == folder) & (key.tag == tag)].set_index('readout')
        if g.empty:
            continue
        cells = []
        for s in shown:
            if s in g.index:
                r = g.loc[s]
                cells.append(f"{r['apl_effect_rmse']:.3f} ({r['apl_effect_mean']:+.3f} / {r['apl_effect_spearman']:.2f})")
            else:
                cells.append('')
        flag = ' (realism flag)' if str(g['realism_flag'].iloc[0]) not in ('', 'nan') else ''
        lines.append(f'| {label}{flag} | ' + ' | '.join(cells) + ' |')
    lines.append(f"| KC data | floor {dt('apl_effect_rmse_floor'):.3f} ({H.yang_apl_effect().mean():+.3f} / ceiling {dt('apl_effect_rho_ceiling'):.2f}) | "
        f"({H.yang_apl_effect(binarize=1.0).mean():+.3f} binarized) |" + ' |' * (len(shown) - 2))
    lines += ['', 'Each cell: APL-effect loss (mean APL effect / ρ on the per-pair APL effect); lower loss is better. '
        '"binary" is scored against the binarized data\'s effect, every other readout against the graded data\'s.', '']
    # the loss of three readouts against each version of the data
    vers = [('graded', 'graded'), ('thresholded', 'sub-threshold zeroed'), ('noise_corrected', 'noise-corrected')]
    reads = ['raw', 'saturating_cal_pr', 'sqrt']
    lines += ['| model | ' + ' | '.join(f'{r} vs {lab}' for r in reads for _, lab in vers) + ' |', '|---|' + '---:|' * (len(reads) * len(vers))]
    for label, folder, tag in KEY_MODELS:
        g = key[(key.folder == folder) & (key.tag == tag)].set_index('readout')
        if g.empty:
            continue
        lines.append(f'| {label} | ' + ' | '.join(f"{g.loc[r, f'vs_{v}__rmse']:.3f} ({g.loc[r, f'vs_{v}__spearman']:.2f})" for r in reads for v, _ in vers) + ' |')
    means = ' / '.join(f"{lab} {H.yang_apl_effect(values=v).mean():+.3f}" for v, lab in vers)
    lines += ['', f'Loss (ρ on the per-pair APL effect) against the data\'s effect read three ways; the data\'s mean APL effect: {means}.', '']
    if len(noise):
        # one noise level per model: the data-matched level on APL on where the table has it
        g = noise[noise['transform'] == 'graded']
        lines += ['| noise table | model | noise level | loss, deterministic / noisy | mean APL effect, deterministic / noisy | '
                  'ρ on the per-pair APL effect, deterministic / noisy | data mean |', '|---|---|---|---:|---:|---:|---:|']
        for (src, model), gm in g.groupby(['source', 'model'], sort=False):
            lev = next((l for l in ('match_on', 'match_on+match_off', 'lo2', 'lo') if (gm['level'] == l).any()), None)
            if lev is None:
                continue
            gl = gm[gm['level'] == lev].set_index('distances')
            if not {'det', 'raw'} <= set(gl.index):
                continue
            d, r = gl.loc['det'], gl.loc['raw']
            lines.append(f"| {src} | {model} | {lev} | {d['apl_effect_rmse']:.3f} / {r['apl_effect_rmse']:.3f} | "
                f"{d['apl_effect_mean']:+.3f} / {r['apl_effect_mean']:+.3f} | {d['apl_effect_spearman']:.2f} / {r['apl_effect_spearman']:.2f} | "
                f"{r['data_effect_mean']:+.3f} |")
        lines += ['', 'Through the trial-noise pipeline (graded; the data-matched noise level on APL on where the table has it): "deterministic" '
            'is the model\'s effect without noise, "noisy" its effect from three-trial means with noise, both against the data\'s raw effect.']
    (out / 'comparison_table.md').write_text('\n'.join(lines) + '\n')


def make_figures(df, key, data_df, data_summ, model_conc, targets, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    graded_mean, floor = float(H.yang_apl_effect().mean()), dt('apl_effect_rmse_floor')
    # 1: loss and mean effect against the readout parameter, key models
    fams = [('power', [0.25, 0.33, 0.5, 0.75, 1.0], 'exponent p of n**p (1 = raw counts, 0.5 = square root)'),
            ('saturating', [1, 2, 5, 10, 20, 50], 'half-saturation k of n / (n + k), spikes')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), squeeze=False)
    cmap = plt.get_cmap('tab10')
    for j, (fam, grid, xlabel) in enumerate(fams):
        for i, (label, folder, tag) in enumerate(KEY_MODELS):
            g = key[(key.folder == folder) & (key.tag == tag)].set_index('readout')
            if g.empty:
                continue
            names = [('raw' if (fam == 'power' and p == 1.0) else 'sqrt' if (fam == 'power' and p == 0.5) else f'{fam}_{p:g}') for p in grid]
            pts = [(p, g.loc[n]) for p, n in zip(grid, names) if n in g.index]
            xs = [p for p, _ in pts]
            for row, col in ((0, 'apl_effect_rmse'), (1, 'apl_effect_mean')):
                axes[row, j].plot(xs, [r[col] for _, r in pts], 'o-', color=cmap(i % 10), lw=1, ms=4, label=label if (row == 0 and j == 0) else None)
                cal = f'{fam}_cal_pr'
                if cal in g.index and np.isfinite(g.loc[cal, 'param']):
                    axes[row, j].plot([g.loc[cal, 'param']], [g.loc[cal, col]], '*', color=cmap(i % 10), ms=11, mec='k', mew=0.5)
        axes[0, j].axhline(floor, color='k', ls='--', lw=1); axes[0, j].text(grid[-1], floor, 'KC data, floor ', fontsize=7, va='bottom', ha='right')
        axes[1, j].axhline(graded_mean, color='k', ls='--', lw=1); axes[1, j].text(grid[-1], graded_mean, 'KC data ', fontsize=7, va='bottom', ha='right')
        for row, ylabel in ((0, 'APL-effect loss (lower is better)'), (1, 'mean APL effect')):
            axes[row, j].set_xscale('log'); axes[row, j].set_ylabel(ylabel, fontsize=8.5); axes[row, j].set_xlabel(xlabel, fontsize=8.5)
            axes[row, j].set_xticks(grid); axes[row, j].set_xticklabels([f'{g:g}' for g in grid]); axes[row, j].minorticks_off()
            axes[row, j].spines[['top', 'right']].set_visible(False); axes[row, j].grid(True, color='0.92', lw=0.6)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle('The per-pair APL effect under compressive readouts of the model\'s spike counts (stars: calibrated to the control flies\' participation ratio)', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E65_readout_params.png', dpi=130); plt.close(fig)
    # 2: every variant, raw against calibrated, coloured by the APL-off / APL-on response-rate ratio
    raw = df[df.readout == 'raw'].set_index(['folder', 'tag'])
    ratio_all = raw['rate_off'] / raw['rate_on']
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), squeeze=False)
    for ax, name in zip(axes[0], ('power_cal_pr', 'saturating_cal_pr', 'rank_match')):
        g = df[df.readout == name].set_index(['folder', 'tag'])
        r0 = raw.loc[g.index]
        real = (g['realism_flag'].fillna('') == '').to_numpy()
        sc = ax.scatter(r0.loc[real, 'apl_effect_rmse'], g.loc[real, 'apl_effect_rmse'], s=9, c=ratio_all.loc[g.index][real], cmap='viridis',
            vmin=1.6, vmax=2.9, label='variants without a realism flag')
        ax.scatter(r0.loc[~real, 'apl_effect_rmse'], g.loc[~real, 'apl_effect_rmse'], s=7, c='0.75', marker='x', lw=0.6, label='realism flag')
        lim = [0, max(0.25, float(np.nanmax(r0['apl_effect_rmse'])))]
        ax.plot(lim, lim, ':', color='0.5'); ax.axhline(floor, color='k', ls='--', lw=1)
        ax.set_xlabel('APL-effect loss, raw counts', fontsize=8.5); ax.set_ylabel(f'APL-effect loss, {name}', fontsize=8.5)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6); ax.legend(fontsize=7, frameon=False, loc='upper left')
    cb = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.85, pad=0.01)
    cb.set_label('APL-off / APL-on response rate (KC data 2.2)', fontsize=8)
    fig.suptitle('Every archived variant: the APL-effect loss under a calibrated readout against raw counts (dashed: the KC data\'s floor)', fontsize=9.5)
    fig.savefig(out / 'E65_loss_raw_vs_calibrated.png', dpi=130, bbox_inches='tight'); plt.close(fig)
    # 2b: the mean APL effect against the response-rate ratio, raw counts and a calibrated readout
    cal = df[df.readout == 'saturating_cal_pr'].set_index(['folder', 'tag'])
    real_idx = raw.index[raw['realism_flag'].fillna('') == '']
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(ratio_all.loc[real_idx], raw.loc[real_idx, 'apl_effect_mean'], s=8, c='0.6', label='raw counts')
    ax.scatter(ratio_all.loc[real_idx], cal.loc[real_idx, 'apl_effect_mean'], s=8, c='#2b6cb0', label='saturating readout calibrated to the control flies\' participation ratio')
    ratio_d = dt('rate_ratio_fraction')
    for v, lab, ls in (('graded', 'KC data', '-'), ('thresholded', 'KC data, sub-threshold zeroed', '--'), ('noise_corrected', 'KC data, noise-corrected', ':')):
        ax.axhline(H.yang_apl_effect(values=v).mean(), color='k', ls=ls, lw=1, label=lab)
    ax.axvline(ratio_d, color='#d9480f', lw=1, label='KC data\'s response-rate ratio')
    ax.set_xlabel('APL-off / APL-on response rate', fontsize=9); ax.set_ylabel('mean APL effect', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6); ax.legend(fontsize=7, frameon=False)
    ax.set_title('Variants without a realism flag: the mean APL effect follows the response-rate ratio once the counts are compressed', fontsize=8.5)
    fig.tight_layout(); fig.savefig(out / 'E65_mean_effect_vs_rate_ratio.png', dpi=130); plt.close(fig)
    # 3: concentration, data against models
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    for ax, (col_m, col_d, xlabel) in zip(axes[0], (('participation_ratio_resp_on', 'participation_ratio', 'participation ratio (share of KCs carrying the squared response vector)'),
                                                    ('top1_share_resp_on', 'top1_share', 'share of the squared response vector in the top 1 % of KCs'))):
        ax.hist(model_conc[col_m].dropna(), bins=40, color='#2b6cb0', alpha=0.8, label='model variants, APL on (raw counts, responders)')
        for g, c, ls in (('TNT3c', 'k', '--'), ('TNT', '0.4', ':')):
            ax.axvline(data_summ.loc[g, f'{col_d}_resp'], color=c, ls=ls, lw=1.2, label=f"KC data ({'control flies' if g == 'TNT3c' else 'APL output blocked (TNT)'}), responders only")
            ax.axvline(data_summ.loc[g, f'{col_d}_all'], color=c, ls=ls, lw=0.6, alpha=0.6)
        ax.set_xlabel(xlabel, fontsize=8.5); ax.set_ylabel('variants', fontsize=8.5); ax.legend(fontsize=6.5, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('How concentrated each odor\'s response vector is: the models\' raw spike counts against the KC data (thin lines: the data with all values)', fontsize=9.5)
    fig.tight_layout(); fig.savefig(out / 'E65_concentration.png', dpi=130); plt.close(fig)
    # 4: odors per KC
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    bins = ['odors_0', 'odors_1', 'odors_2_3', 'odors_4_7', 'odors_8plus']
    labels = ['0', '1', '2-3', '4-7', '8+']
    x = np.arange(len(bins))
    ref_rows = model_conc[model_conc.folder.eq('E44_operating_point') & model_conc.tag.eq('thr40_gain1.5_rate0.068_k2_cap3')]
    real = model_conc['realism_flag'].fillna('') == ''
    for ax, (cond, g, glab) in zip(axes, (('on', 'TNT3c', 'control flies'), ('off', 'TNT', 'APL output blocked (TNT)'))):
        ax.bar(x - 0.27, [data_summ.loc[g, b] for b in bins], 0.27, color='k', label=f'KC data, {glab} (z > 1)')
        ax.bar(x, [model_conc.loc[real, f'{b}_{cond}'].median() for b in bins], 0.27, color='#2b6cb0',
            label=f"model variants without a realism flag, APL {cond} (median)")
        if len(ref_rows):
            ax.bar(x + 0.27, [float(ref_rows[f'{b}_{cond}'].iloc[0]) for b in bins], 0.27, color='#d9480f', label=f'the reference model, APL {cond}')
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel('number of the 16 odors a KC responds to')
        ax.set_title(f'APL {cond}' + (' / control flies' if cond == 'on' else ' / APL output blocked (TNT)'), fontsize=9)
        ax.legend(fontsize=7, frameon=False); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('fraction of KCs')
    fig.tight_layout(); fig.savefig(out / 'E65_odors_per_kc.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
