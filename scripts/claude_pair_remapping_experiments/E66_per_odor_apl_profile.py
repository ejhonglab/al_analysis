#!/usr/bin/env python3
"""E66: the per-odor profile of the APL effect. The pattern part of the APL-effect loss split into a
per-odor and a pair-specific part, and a per-odor APL oracle.

The main target is the per-pair APL effect on odor-odor distances (each pair's APL-on minus APL-off
cosine distance) against the KC data's (control flies minus APL output blocked (TNT)), scored by the
APL-effect loss (`kc_remapping_harness.apl_effect_stats`). Its pattern part (the loss with both means
removed) splits exactly into a per-odor part (the error's projection onto the additive space, a pair's
value the sum of its two odors' shares) and a pair-specific part (the residual)
(`kc_remapping_harness.apl_effect_decomposition`, `apl_effect_pair_parts`).

Part A (no simulations): every archived variant with APL-on and acute APL-off spike counts in
data/model_outputs/, scored under four readouts of its counts (`kc_remapping_harness.readout`): raw
counts (the target as defined), the square root, the compressive readout experiment's (E65) saturating
readout n / (n + k) with k calibrated per variant on its APL-on counts to the control flies' responder
participation ratio (as saved in results/E65_spike_count_readout/by_variant_readout.csv), and binarized
counts (>= 2 spikes) against the binarized data's effect. Per variant and readout: the loss and its
mean, per-odor and pair-specific parts, the per-odor shares against the data's and against each odor's
response fractions in both conditions. The data's shares get recording-bootstrap intervals, and the
floor of each part (a resampled data set against the measured one). For the key models: the pairs that
carry the pair-specific part and the KCs behind them (per-KC contributions to each pair's APL effect, by
whether the KC responds to both odors with APL on, only with APL off, or neither, and the glomeruli of
their claws). The two archived oracles read the same way: the per-odor rate oracle's (E62) arms singly
and combined across conditions (APL-on counts from the arm set on the APL-on fractions, APL-off counts
from the arm set on the APL-off fractions), and the bounded-input oracle's (E64) searched input and its
controls; in the oracle table every arm is read with its base model's calibrated readout parameter.

Part B (the per-odor APL oracle), at the fitted scales of five bases (BASES: the three reference models,
the first joint fit, the 40 Hz reference model with E64's searched input): each odor simulated on its own
(odors are independent at fixed scales), its APL->KC and APL->bouton scales multiplied by a per-odor
factor bisected on its APL-on response fraction against the control flies'; arm 'apl' with the inputs
unscaled, arm 'both' with each odor's input first scaled to its APL-off fraction against APL output
blocked (TNT) (E62's off-arm scales for the reference models, bisected here for the others). Variants
O_<base>_{untuned, apl, scales_off, both}; per-odor multipliers against odor properties
(multipliers.csv).

Part C (the breadth exponent): on the three reference models, the breadth exponent at 0-3 with the breadth
as the KC's APL-off spike total or its number of odors, the APL->KC scale re-tuned to hold the fitted
APL-on response rate. Variants C_<base>_<definition>_k<k>.

Outputs, <shared>/results/E66_per_odor_apl_profile/: part A: data_shares.csv, data_floor_parts.csv,
data_share_correlates.csv, by_variant_parts.csv, shares_long.csv, parts_summary.csv, key_models.csv,
worst_pairs.csv, worst_pair_kcs.csv, oracles.csv, comparison_table.md, figures; parts B / C:
part<B|C>_variant_stats.csv, model_pair_distances_long_<B|C>.csv, parts_<B|C>.csv, shares_<B|C>.csv,
multipliers.csv, multiplier_history.csv; provenance.json. Counts in ./outputs/E66_per_odor_apl_profile/.
Usage: ./venv/bin/python E66_per_odor_apl_profile.py --part A [--jobs N] [--boot 1000]
       OMP_NUM_THREADS=1 nice -n 10 ./venv/bin/python E66_per_odor_apl_profile.py --part B --jobs 8 [--bases ...]
       OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E66_per_odor_apl_profile.py --part C [--c-bases ...]
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E65_spike_count_readout import KEY_MODELS

EXPERIMENT = 'E66_per_odor_apl_profile'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
RESULTS = H.shared_dir() / 'results'
E65_TABLE = RESULTS / 'E65_spike_count_readout' / 'by_variant_readout.csv'
BY_VARIANT = RESULTS / 'apl_effect' / 'by_variant.csv'
CAL_READOUT = 'saturating_cal_pr'
# (name, kind, param, data): param 'cal' is the variant's calibrated parameter from E65's table;
# data 'graded' (the KC data's effect) or 'binary' (the binarized data's)
READOUTS = [('raw', 'raw', None, 'graded'), ('sqrt', 'sqrt', None, 'graded'),
    (CAL_READOUT, 'saturating', 'cal', 'graded'), ('binary', 'binary', 2.0, 'binary')]
READOUT_LABEL = {'raw': 'raw counts', 'sqrt': 'square root', CAL_READOUT: 'calibrated saturating',
    'binary': 'binarized (against the binarized data)'}
ODORS = H.APL16_ODORS
RESP_MIN = 2.0
PART_KEYS = ('apl_effect_rmse', 'apl_effect_mean', 'apl_effect_mean_error', 'apl_effect_rmse_centered',
    'apl_effect_per_odor', 'apl_effect_pair_specific', 'apl_effect_spearman', 'apl_effect_share_pearson',
    'apl_effect_model_additive_r2')
# the two archived oracles: (label, folder, base tag, {arm: (APL-on tag, APL-off tag)})
E62_MODELS = {'thr40_gain1.5': 'the reference model (eight at 1.5, 40 Hz, 6.8 %)',
    'thr70_gain1.5': 'eight at 1.5, 70 Hz, 6.8 %', 'thr70_gain1.25': 'eight at 1.25, 70 Hz, 6.8 %'}
ORACLES = [(f'{lab}: rate oracle', 'E62_rate_oracle', f'R_{m}_untuned', {
        'untuned': (f'R_{m}_untuned', f'R_{m}_untuned'),
        'input scales set on APL-on fractions': (f'R_{m}_on', f'R_{m}_on'),
        'input scales set on APL-off fractions': (f'R_{m}_off', f'R_{m}_off'),
        'both arms combined': (f'R_{m}_on', f'R_{m}_off')}) for m, lab in E62_MODELS.items()] + [
    ('the reference model: bounded-input oracle', 'E64_input_oracle', 'I_untuned', {
        'untuned': ('I_untuned', 'I_untuned'),
        'searched weak-odor input (pass 1)': ('I_searched_pass1', 'I_searched_pass1'),
        'controls (free input for odors fitted already)': ('I_controls', 'I_controls')})]
WORST_N = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_counts(folder: str, tag: str, cond: str) -> pd.DataFrame:
    """An archived variant's KC x odor spike counts, `cond` 'on' or 'acute_off'."""
    return pd.read_parquet(ARCHIVE / folder / f'{tag}_{cond}_spike_counts.parquet')


_DATA: Dict[str, object] = {}


def data_refs() -> Dict[str, object]:
    """The data's per-pair effects ('graded', 'binary') and per-odor response fractions (recording
    means; 'frac_on' control flies, 'frac_off' APL output blocked (TNT)), cached per process."""
    if not _DATA:
        _DATA['graded'] = H.yang_apl_effect()
        _DATA['binary'] = H.yang_apl_effect(binarize=1.0)
        _DATA['frac_on'] = H.yang_response_fractions('TNT3c')[ODORS].mean()
        _DATA['frac_off'] = H.yang_response_fractions('TNT')[ODORS].mean()
    return _DATA


def corr(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, join='inner')
    ok = a.notna() & b.notna()
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan


def score(on: pd.DataFrame, off: pd.DataFrame, kind: str, param, data: str
    ) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """The decomposition of one model under one readout: (stats, per-odor table, per-pair parts).
    The per-odor table has the model's and the data's shares, the share error and both sides'
    response fractions in both conditions (the model's: counts >= 2 spikes)."""
    refs = data_refs()
    d_on, d_off = H.readout_pair_distances(on, kind, param), H.readout_pair_distances(off, kind, param)
    stats, shares = H.apl_effect_decomposition(d_on, d_off, refs[data])
    parts = H.apl_effect_pair_parts(d_on, d_off, refs[data])
    f_on = H.response_fractions(on >= RESP_MIN).reindex(ODORS)
    f_off = H.response_fractions(off >= RESP_MIN).reindex(ODORS)
    t = shares.reindex(ODORS).copy()
    t['frac_on'], t['frac_off'] = f_on, f_off
    t['data_frac_on'], t['data_frac_off'] = refs['frac_on'], refs['frac_off']
    gain, dgain = f_off - f_on, refs['frac_off'] - refs['frac_on']
    stats.update(share_vs_gain_r=corr(t.model_share, gain), share_err_vs_gain_err_r=corr(t.share_error, gain - dgain),
        share_vs_ratio_r=corr(t.model_share, f_off / f_on), gain_vs_data_gain_r=corr(gain, dgain),
        mean_dist_on=float(d_on.mean()), mean_dist_off=float(d_off.mean()))
    return stats, t, parts


def score_job(job: tuple) -> Tuple[List[dict], List[pd.DataFrame]]:
    """All readouts of one archived variant: (rows, per-odor tables with identifying columns)."""
    folder, tag, cal = job
    on, off = load_counts(folder, tag, 'on'), load_counts(folder, tag, 'acute_off')
    rows, tabs = [], []
    for name, kind, param, data in READOUTS:
        p = cal if param == 'cal' else param
        if param == 'cal' and not np.isfinite(p):
            continue
        st, t, _ = score(on, off, kind, p, data)
        rows.append(dict(folder=folder, tag=tag, readout=name, param=p, **st))
        tabs.append(t.reset_index().assign(folder=folder, tag=tag, readout=name))
    return rows, tabs


def data_bootstrap(n_boot: int, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The data's per-odor shares with recording-bootstrap intervals (recordings drawn with
    replacement within each genotype, as the registry's floor), for the graded and the binarized
    data, and the floor of each part of the loss: a resampled data set's effect scored against the
    measured one. Returns (shares table, floor table)."""
    refs = data_refs()
    rows, floors = [], []
    for name, binarize in (('graded', None), ('binary', 1.0)):
        recs = {g: H.yang_recording_rdms(g, 'cosine', binarize=binarize) for g in ('TNT3c', 'TNT')}
        measured = refs[name]
        s0, r2 = H.odor_shares(measured)
        rng = np.random.default_rng(seed)
        boot, fl = [], []
        for _ in range(n_boot):
            m = H.resample_recording_means(recs, rng)
            st, sh = H.apl_effect_decomposition(m['TNT3c'], m['TNT'], measured)
            boot.append(sh['model_share'])
            fl.append({k: st[k] for k in PART_KEYS if k != 'apl_effect_model_additive_r2'} | {'additive_r2': st['apl_effect_model_additive_r2']})
        b = pd.DataFrame(boot)
        for o in ODORS:
            rows.append(dict(data=name, odor=o, share=float(s0[o]), share_lo=float(b[o].quantile(0.05)),
                share_hi=float(b[o].quantile(0.95)), share_sd=float(b[o].std())))
        f = pd.DataFrame(fl)
        add_sd = float(np.std(measured.to_numpy() - H.pair_design(measured.index, list(s0.index))[0] @ s0.to_numpy()))
        for k in f.columns:
            floors.append(dict(data=name, stat=k, mean=float(f[k].mean()), lo=float(f[k].quantile(0.05)),
                hi=float(f[k].quantile(0.95))))
        floors.append(dict(data=name, stat='measured_additive_r2', mean=r2, lo=np.nan, hi=np.nan))
        floors.append(dict(data=name, stat='measured_remainder_sd', mean=add_sd, lo=np.nan, hi=np.nan))
    shares = pd.DataFrame(rows)
    shares['frac_on'] = shares.odor.map(refs['frac_on'])
    shares['frac_off'] = shares.odor.map(refs['frac_off'])
    return shares, pd.DataFrame(floors)


def share_correlates(shares: pd.DataFrame) -> pd.DataFrame:
    """Pearson r of the data's per-odor shares with each odor's response fractions (control flies,
    APL output blocked (TNT)), their difference and their ratio."""
    out = []
    for name, g in shares.groupby('data'):
        g = g.set_index('odor')
        for lab, x in (('frac_on', g.frac_on), ('frac_off', g.frac_off), ('gain', g.frac_off - g.frac_on),
                ('ratio', g.frac_off / g.frac_on)):
            out.append(dict(data=name, correlate=lab, pearson=corr(g.share, x), spearman=H.spearman(g.share, x)))
    return pd.DataFrame(out)


def claw_glomeruli() -> pd.Series:
    """kc_id -> list of the glomeruli of its claws (the bouton base's wiring)."""
    w = pd.read_parquet(ARCHIVE / 'inputs' / 'wPNKC_bouton_base.parquet')
    vals = w.to_numpy()
    glom = w.columns.get_level_values('glomerulus').to_numpy()[vals.argmax(axis=1)]
    df = pd.DataFrame(dict(kc_id=w.index.get_level_values('kc_id').to_numpy(), glomerulus=glom))
    return df.groupby('kc_id').glomerulus.agg(list)


def kc_contributions(on: pd.DataFrame, off: pd.DataFrame, kind: str, param, a: str, b: str) -> pd.DataFrame:
    """Per KC, its contribution to the pair (a, b)'s APL effect under a readout: the effect is
    cos_off - cos_on, and each cosine a sum over KCs of x_a x_b / (|x_a| |x_b|). Columns: contribution,
    class ('both with APL on': responds to both odors with APL on; 'both only with APL off'; 'one or
    neither'), and the counts."""
    def terms(m):
        r = H.abbrev_columns(H.readout(m, kind, param))
        xa, xb = r[a].to_numpy(dtype=float), r[b].to_numpy(dtype=float)
        n = np.linalg.norm(xa) * np.linalg.norm(xb)
        return xa * xb / n if n > 0 else np.zeros_like(xa)
    c = terms(off) - terms(on)
    ca, cb = H.abbrev_columns(on), H.abbrev_columns(off)
    both_on = (ca[a] >= RESP_MIN) & (ca[b] >= RESP_MIN)
    both_off = (cb[a] >= RESP_MIN) & (cb[b] >= RESP_MIN)
    cls = np.where(both_on, 'both with APL on', np.where(both_off, 'both only with APL off', 'one or neither'))
    return pd.DataFrame(dict(kc_id=on.index.get_level_values('kc_id'), kc_type=on.index.get_level_values('kc_type'),
        contribution=c, kc_class=cls, on_a=ca[a].to_numpy(), on_b=ca[b].to_numpy(), off_a=cb[a].to_numpy(),
        off_b=cb[b].to_numpy()))


def worst_pairs(key_parts: Dict[tuple, pd.DataFrame], cal: Dict[tuple, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The pairs carrying the pair-specific part for the key models (raw counts and the calibrated
    readout), and the KCs behind each: per pair, how much of the model's per-pair effect comes from
    KCs that respond to both odors with APL on, from KCs that respond to both only with APL off, and
    from the rest; how many KCs carry half of it; the most frequent claw glomeruli among the 10 KCs with
    the largest contributions."""
    claws = claw_glomeruli()
    rows, kc_rows = [], []
    for (label, folder, tag, readout), parts in key_parts.items():
        if readout not in ('raw', CAL_READOUT):
            continue
        kind, param = ('raw', None) if readout == 'raw' else ('saturating', cal[(folder, tag)])
        on, off = load_counts(folder, tag, 'on'), load_counts(folder, tag, 'acute_off')
        top = parts.reindex(parts.pair_specific.abs().sort_values(ascending=False).index[:WORST_N])
        for rank, ((a, b), r) in enumerate(top.iterrows(), 1):
            kc = kc_contributions(on, off, kind, param, a, b)
            tot = kc.contribution.sum()
            absc = kc.contribution.abs().sort_values(ascending=False)
            n_half = int(np.searchsorted(np.cumsum(absc.to_numpy()), 0.5 * absc.sum()) + 1)
            by_cls = kc.groupby('kc_class').contribution.sum()
            top10 = kc.loc[absc.index[:10]]
            gl = pd.Series([g for k in top10.kc_id for g in claws.get(k, [])]).value_counts()
            rows.append(dict(model=label, folder=folder, tag=tag, readout=readout, rank=rank, odor_a=a, odor_b=b,
                model_effect=r.model, data_effect=r.data, error=r.error, per_odor=r.per_odor, pair_specific=r.pair_specific,
                kc_effect_total=tot, **{f'from_{c.replace(" ", "_")}': float(by_cls.get(c, 0.0)) for c in
                    ('both with APL on', 'both only with APL off', 'one or neither')},
                n_kcs_half=n_half, top10_share=float(absc.iloc[:10].sum() / absc.sum()) if absc.sum() > 0 else np.nan,
                top10_types=', '.join(f'{t} {n}' for t, n in top10.kc_type.value_counts().items()),
                top10_glomeruli=', '.join(f'{g} {n}' for g, n in gl.iloc[:4].items())))
            kc_rows.append(top10.assign(model=label, readout=readout, odor_a=a, odor_b=b, rank=rank))
    return pd.DataFrame(rows), (pd.concat(kc_rows, ignore_index=True) if kc_rows else pd.DataFrame())


def oracle_table(cal_table: pd.DataFrame) -> pd.DataFrame:
    """The archived oracles' arms under each readout, each arm read with its base model's calibrated
    parameter (so the readout does not move with the oracle)."""
    rows = []
    for label, folder, base, arms in ORACLES:
        base_cal = float(cal_table.loc[(folder, base)])
        for arm, (t_on, t_off) in arms.items():
            on, off = load_counts(folder, t_on, 'on'), load_counts(folder, t_off, 'acute_off')
            for name, kind, param, data in READOUTS:
                p = base_cal if param == 'cal' else param
                st, _, _ = score(on, off, kind, p, data)
                f_on = float((on >= RESP_MIN).mean().mean())
                f_off = float((off >= RESP_MIN).mean().mean())
                rows.append(dict(oracle=label, folder=folder, arm=arm, on_tag=t_on, off_tag=t_off, readout=name, param=p,
                    rate_on=f_on, rate_off=f_off, rate_ratio=f_off / f_on, **st))
    return pd.DataFrame(rows)


def parts_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per readout, over the variants without a realism flag: medians of the loss and its parts, the
    median share of the pattern part's variance that is per-odor, and the best variant on each part."""
    out = []
    for name, g in df.groupby('readout', sort=False):
        real = g[g.realism_flag.fillna('') == ''].copy()
        real['per_odor_var_share'] = real.apl_effect_per_odor ** 2 / real.apl_effect_rmse_centered ** 2
        row = dict(readout=name, n_variants=len(g), n_realistic=len(real))
        for k in ('apl_effect_rmse', 'apl_effect_rmse_centered', 'apl_effect_per_odor', 'apl_effect_pair_specific',
                'apl_effect_share_pearson', 'apl_effect_spearman', 'per_odor_var_share'):
            row[f'median_{k}'] = float(real[k].median())
        row['spearman_per_odor_vs_pair_specific'] = H.spearman(real.apl_effect_per_odor, real.apl_effect_pair_specific)
        for k in ('apl_effect_rmse', 'apl_effect_rmse_centered', 'apl_effect_per_odor', 'apl_effect_pair_specific'):
            best = real.loc[real[k].idxmin()]
            row[f'best_{k}'] = f'{best.folder}/{best.tag}'
            row[f'best_{k}_value'] = float(best[k])
        out.append(row)
    return pd.DataFrame(out)


def part_a(args) -> None:
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    e65 = pd.read_csv(E65_TABLE, low_memory=False)
    cal_table = e65[e65.readout == CAL_READOUT].set_index(['folder', 'tag'])['param']
    bv = pd.read_csv(BY_VARIANT)
    have = [(r.folder, r.tag) for r in bv.itertuples()
        if (ARCHIVE / r.folder / f'{r.tag}_on_spike_counts.parquet').exists()
        and (ARCHIVE / r.folder / f'{r.tag}_acute_off_spike_counts.parquet').exists()]
    print(f'{len(have)} archived variants', flush=True)

    shares, floors = data_bootstrap(args.boot)
    shares.to_csv(out / 'data_shares.csv', index=False)
    floors.to_csv(out / 'data_floor_parts.csv', index=False)
    correl = share_correlates(shares)
    correl.to_csv(out / 'data_share_correlates.csv', index=False)
    print(floors.round(4).to_string(), flush=True)
    print(correl.round(3).to_string(), flush=True)

    jobs = [(f, t, float(cal_table.get((f, t), np.nan))) for f, t in have]
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    rows, tabs = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, (r, t) in enumerate(ex.map(score_job, jobs, chunksize=4)):
            rows.extend(r)
            tabs.extend(t)
            if (i + 1) % 100 == 0:
                print(f'{i + 1} / {len(jobs)} variants', flush=True)
    df = pd.DataFrame(rows)
    flags = bv.set_index(['folder', 'tag'])[['exp', 'rate_on', 'rate_off', 'spearman_rate_on_vs_data', 'realism_flag']]
    df = df.join(flags, on=['folder', 'tag'])
    df.to_csv(out / 'by_variant_parts.csv', index=False)
    pd.concat(tabs, ignore_index=True).to_csv(out / 'shares_long.csv', index=False)
    summ = parts_summary(df)
    summ.to_csv(out / 'parts_summary.csv', index=False)

    key_rows, key_parts, key_shares = [], {}, []
    for label, folder, tag in KEY_MODELS:
        on, off = load_counts(folder, tag, 'on'), load_counts(folder, tag, 'acute_off')
        for name, kind, param, data in READOUTS:
            p = float(cal_table.loc[(folder, tag)]) if param == 'cal' else param
            st, t, parts = score(on, off, kind, p, data)
            key_rows.append(dict(model=label, folder=folder, tag=tag, readout=name, param=p, **st))
            key_parts[(label, folder, tag, name)] = parts
            key_shares.append(t.reset_index().assign(model=label, readout=name))
    key = pd.DataFrame(key_rows).join(flags[['rate_on', 'rate_off', 'realism_flag']], on=['folder', 'tag'])
    key.to_csv(out / 'key_models.csv', index=False)
    pd.concat(key_shares, ignore_index=True).to_csv(out / 'key_model_shares.csv', index=False)
    wp, wk = worst_pairs(key_parts, cal_table.to_dict())
    wp.to_csv(out / 'worst_pairs.csv', index=False)
    wk.to_csv(out / 'worst_pair_kcs.csv', index=False)

    orc = oracle_table(cal_table)
    orc.to_csv(out / 'oracles.csv', index=False)

    write_table_a(out, key, orc, floors, shares, correl, summ, wp)
    figures_a(out, df, key, shares, floors, pd.concat(key_shares, ignore_index=True))
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E66_per_odor_apl_profile.py --part A',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        al_analysis_dirty=bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip()),
        inputs_sha256={str(p): sha256(p) for p in (BY_VARIANT, E65_TABLE)}, n_variants=len(have), readouts=[r[0] for r in READOUTS],
        calibrated_readout=CAL_READOUT, response_threshold_spikes=RESP_MIN, n_boot=args.boot, boot_seed=0,
        oracles={lab: dict(folder=f, base=b, arms=a) for lab, f, b, a in ORACLES})
    (out / 'provenance_partA.json').write_text(json.dumps(prov, indent=1) + '\n')
    print(summ.round(3).T.to_string())
    print('->', out)


def fmt(x, spec='.3f') -> str:
    return '-' if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def write_table_a(out: Path, key: pd.DataFrame, orc: pd.DataFrame, floors: pd.DataFrame, shares: pd.DataFrame,
        correl: pd.DataFrame, summ: pd.DataFrame, wp: pd.DataFrame) -> None:
    fl = floors.set_index(['data', 'stat'])
    L = ['# E66 part A: the APL-effect loss split into its mean, per-odor and pair-specific parts', '',
         'Loss parts: rmse^2 = mean_error^2 + per_odor^2 + pair_specific^2 (pattern part^2 = per_odor^2 + pair_specific^2). '
         'Lower is better for the loss and every part; ρ on the per-pair APL effect and r of the shares: higher is better.', '',
         '## The KC data', '',
         '| statistic | graded data | binarized data |', '|---|---:|---:|']
    for stat, lab in (('apl_effect_rmse', 'floor of the loss (a resampled data set against the measured one)'),
            ('apl_effect_rmse_centered', 'floor of the pattern part'), ('apl_effect_per_odor', 'floor of the per-odor part'),
            ('apl_effect_pair_specific', 'floor of the pair-specific part'),
            ('measured_additive_r2', 'share of the per-pair effect\'s variance that is additive over odors'),
            ('measured_remainder_sd', 'sd of the non-additive remainder')):
        L.append(f'| {lab} | {fmt(fl.loc[("graded", stat), "mean"])} | {fmt(fl.loc[("binary", stat), "mean"])} |')
    L += ['', 'Correlates of the data\'s per-odor shares (Pearson r over the 16 odors): ' + ', '.join(
        f'{r.correlate} {r.pearson:+.2f}' for r in correl[correl.data == 'graded'].itertuples()) + ' (graded); ' + ', '.join(
        f'{r.correlate} {r.pearson:+.2f}' for r in correl[correl.data == 'binary'].itertuples()) + ' (binarized).', '',
        '## Key models', '',
        '| model | readout | loss | mean APL effect | pattern part | per-odor part | pair-specific part | ρ on the per-pair APL effect | r of shares with the data\'s | r of shares with own response-fraction gain |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in key.itertuples():
        L.append(f'| {r.model} | {READOUT_LABEL[r.readout]} | {fmt(r.apl_effect_rmse)} | {fmt(r.apl_effect_mean, "+.3f")} | '
                 f'{fmt(r.apl_effect_rmse_centered)} | {fmt(r.apl_effect_per_odor)} | {fmt(r.apl_effect_pair_specific)} | '
                 f'{fmt(r.apl_effect_spearman, ".2f")} | {fmt(r.apl_effect_share_pearson, ".2f")} | {fmt(r.share_vs_gain_r, ".2f")} |')
    L += ['', '## The archived oracles (each arm read with its base model\'s calibrated parameter)', '',
          '| oracle | arm | readout | loss | mean APL effect | pattern part | per-odor part | pair-specific part | ρ on the per-pair APL effect | r of shares | APL-on response rate | APL-off / APL-on |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in orc.itertuples():
        L.append(f'| {r.oracle} | {r.arm} | {READOUT_LABEL[r.readout]} | {fmt(r.apl_effect_rmse)} | {fmt(r.apl_effect_mean, "+.3f")} | '
                 f'{fmt(r.apl_effect_rmse_centered)} | {fmt(r.apl_effect_per_odor)} | {fmt(r.apl_effect_pair_specific)} | '
                 f'{fmt(r.apl_effect_spearman, ".2f")} | {fmt(r.apl_effect_share_pearson, ".2f")} | {fmt(r.rate_on)} | {fmt(r.rate_ratio, ".2f")} |')
    L += ['', '## Every archived variant without a realism flag (medians; best variant per part)', '',
          '| readout | n | loss | pattern part | per-odor part | pair-specific part | per-odor share of the pattern variance | r of shares | ρ(per-odor part, pair-specific part) over variants |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summ.itertuples():
        L.append(f'| {READOUT_LABEL[r.readout]} | {r.n_realistic} | {fmt(r.median_apl_effect_rmse)} | {fmt(r.median_apl_effect_rmse_centered)} | '
                 f'{fmt(r.median_apl_effect_per_odor)} | {fmt(r.median_apl_effect_pair_specific)} | {fmt(r.median_per_odor_var_share, ".2f")} | '
                 f'{fmt(r.median_apl_effect_share_pearson, ".2f")} | {fmt(r.spearman_per_odor_vs_pair_specific, ".2f")} |')
    L += ['']
    for r in summ.itertuples():
        L.append(f'- {READOUT_LABEL[r.readout]}: lowest pattern part {r.best_apl_effect_rmse_centered} ({r.best_apl_effect_rmse_centered_value:.3f}), '
                 f'lowest per-odor part {r.best_apl_effect_per_odor} ({r.best_apl_effect_per_odor_value:.3f}), '
                 f'lowest pair-specific part {r.best_apl_effect_pair_specific} ({r.best_apl_effect_pair_specific_value:.3f})')
    L += ['', '## The pairs carrying the pair-specific part (key models)', '',
          '| model | readout | pair | model effect | data effect | pair-specific error | from KCs responding to both with APL on | from KCs responding to both only with APL off | from the rest | KCs carrying half | top-10 KC glomeruli |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in wp[wp['rank'] <= 5].itertuples():
        L.append(f'| {r.model} | {READOUT_LABEL[r.readout]} | {r.odor_a}, {r.odor_b} | {fmt(r.model_effect, "+.3f")} | {fmt(r.data_effect, "+.3f")} | '
                 f'{fmt(r.pair_specific, "+.3f")} | {fmt(r.from_both_with_APL_on, "+.3f")} | {fmt(r.from_both_only_with_APL_off, "+.3f")} | '
                 f'{fmt(r.from_one_or_neither, "+.3f")} | {r.n_kcs_half} | {r.top10_glomeruli} |')
    (out / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def figures_a(out: Path, df: pd.DataFrame, key: pd.DataFrame, shares: pd.DataFrame, floors: pd.DataFrame,
        key_shares: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fl = floors.set_index(['data', 'stat'])['mean']
    # 1: per-odor part against pair-specific part, every variant, per readout
    fig, axes = plt.subplots(1, len(READOUTS), figsize=(4.2 * len(READOUTS), 4.2), sharex=True, sharey=True)
    for ax, (name, *_rest) in zip(axes, READOUTS):
        data = 'binary' if name == 'binary' else 'graded'
        g = df[df.readout == name]
        flag = g.realism_flag.fillna('') != ''
        ax.scatter(g.loc[flag, 'apl_effect_per_odor'], g.loc[flag, 'apl_effect_pair_specific'], s=8, c='0.75', label='realism flag', lw=0)
        ax.scatter(g.loc[~flag, 'apl_effect_per_odor'], g.loc[~flag, 'apl_effect_pair_specific'], s=8, c='C0', label='no flag', lw=0)
        k = key[key.readout == name]
        ax.scatter(k.apl_effect_per_odor, k.apl_effect_pair_specific, s=40, facecolor='none', edgecolor='C3', lw=1.5,
            label='key models')
        for r in k.itertuples():
            if 'reference model (eight' in r.model or 'first joint fit' in r.model or 'searched' in r.model:
                ax.annotate({True: 'reference', False: 'first joint fit'}.get('reference model (eight' in r.model, '')
                    if 'searched' not in r.model else 'E64 searched input', (r.apl_effect_per_odor, r.apl_effect_pair_specific),
                    fontsize=7, xytext=(4, 4), textcoords='offset points')
        ax.axvline(fl.loc[(data, 'apl_effect_per_odor')], ls=':', c='k', lw=1)
        ax.axhline(fl.loc[(data, 'apl_effect_pair_specific')], ls=':', c='k', lw=1, label='data floor (resampled recordings)')
        ax.set_title(READOUT_LABEL[name].replace(' (against', '\n(against'), fontsize=10)
        ax.set_xlabel('per-odor part of the APL-effect loss')
    axes[0].set_ylabel('pair-specific part of the APL-effect loss')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(out / 'E66_parts_all_variants.png', dpi=150)
    plt.close(fig)
    # 2: per-odor shares, data with intervals, reference model and first joint fit under raw and calibrated
    sh = shares[shares.data == 'graded'].set_index('odor').reindex(ODORS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    x = np.arange(len(ODORS))
    for ax, name in zip(axes, ('raw', CAL_READOUT)):
        ax.errorbar(x, sh.share, yerr=[sh.share - sh.share_lo, sh.share_hi - sh.share], fmt='o', c='k', ms=5, capsize=2,
            label='KC data (90 % interval over resampled recordings)')
        for i, (lab, col) in enumerate((('the reference model (eight at 1.5, 40 Hz, 6.8 %)', 'C0'),
                ('the first joint fit (k 2, 5 %)', 'C1'), ('the searched weak-odor input, pass 1 (oracle)', 'C2'))):
            k = key_shares[(key_shares.model == lab) & (key_shares.readout == name)].set_index('odor').reindex(ODORS)
            ax.plot(x + 0.15 * (i - 1), k.model_share, 's', c=col, ms=4, label=lab.replace('the ', '', 1))
        ax.axhline(0, c='0.8', lw=0.8)
        ax.set_xticks(x, ODORS, rotation=60)
        ax.set_title(f'per-odor share of the per-pair APL effect, {READOUT_LABEL[name]}', fontsize=10)
    axes[0].set_ylabel('share (pair effect = share a + share b)')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'E66_shares_key_models.png', dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# parts B and C: simulations at fixed scales
# base key -> label, experiment and tag of the fitted parameters (fits.csv), per-claw threshold,
# breadth exponent, glomerulus gain, ORN input (None: the lab's; else a CSV under the shared
# directory, glomeruli x odors as E64 saved it), E62's model key (its off-arm input scales), and the
# archived variant whose E65 calibrated readout parameter the base's variants use
BASES = {
    'thr40_gain1.5': dict(label='the reference model (eight at 1.5, 40 Hz, 6.8 %)', exp='E44_operating_point',
        tag='thr40_gain1.5_rate0.068_k2_cap3', thr=40.0, k=2.0, gain=1.5, input=None, e62='thr40_gain1.5',
        cal=('E62_rate_oracle', 'R_thr40_gain1.5_untuned')),
    'thr70_gain1.5': dict(label='eight at 1.5, 70 Hz, 6.8 %', exp='E44_operating_point', tag='thr70_gain1.5_rate0.068_k2_cap3',
        thr=70.0, k=2.0, gain=1.5, input=None, e62='thr70_gain1.5', cal=('E62_rate_oracle', 'R_thr70_gain1.5_untuned')),
    'thr70_gain1.25': dict(label='eight at 1.25, 70 Hz, 6.8 %', exp='E45_gain_at_ctrl_frac', tag='thr70_gain1.25_rate0.068_k2_cap3',
        thr=70.0, k=2.0, gain=1.25, input=None, e62='thr70_gain1.25', cal=('E62_rate_oracle', 'R_thr70_gain1.25_untuned')),
    'e10_k2': dict(label='the first joint fit (k 2, 5 %)', exp='E10_joint_fit', tag='k2', thr=None, k=2.0, gain=None, input=None,
        e62=None, cal=('E10_joint_fit', 'k2')),
    'thr40_gain1.5_e64input': dict(label='the reference model with the searched weak-odor input (E64, pass 1; an oracle)',
        exp='E44_operating_point', tag='thr40_gain1.5_rate0.068_k2_cap3', thr=40.0, k=2.0, gain=1.5,
        input='results/E64_input_oracle/orn_input_searched_pass1.csv', e62=None, cal=('E64_input_oracle', 'I_searched_pass1')),
}
REF_BASES = ('thr40_gain1.5', 'thr70_gain1.5', 'thr70_gain1.25')
# part B: the variants assembled from each base's two arms ('apl': inputs unscaled; 'both': each odor's
# input scaled to its APL-off fraction first), the APL-on counts at multiplier 1 or at the odor's own
# multiplier
B_VARIANTS = {'untuned': ('apl', 'm1'), 'apl': ('apl', 'mo'), 'scales_off': ('both', 'm1'), 'both': ('both', 'mo')}
B_LABEL = {'untuned': 'the base model (multiplier 1, inputs unscaled)',
    'apl': 'per-odor APL multiplier set on the APL-on fractions (inputs unscaled)',
    'scales_off': 'input scales set on the APL-off fractions (multiplier 1; the rate oracle\'s off arm)',
    'both': 'input scales on the APL-off fractions, then the APL multiplier on the APL-on fractions'}
SCALE_LO, SCALE_HI, SCALE_ITERS = 0.125, 8.0, 8      # as the rate oracle (E62)
M_LO, M_HI, M_ITERS = 1 / 16, 16.0, 12
# part C: (breadth definition, exponent); spikes k 2 is each base as fitted (the re-tune checks it)
C_CONFIGS = [('spikes', 0.0), ('spikes', 1.0), ('spikes', 2.0), ('spikes', 3.0), ('n_odors', 1.0), ('n_odors', 2.0), ('n_odors', 3.0)]
C_ITERS = 14
GAINED_GLOMS = ('D', 'DC2', 'DL1', 'DM1', 'DM5', 'DP1m', 'VA7m', 'VC1')

_WORKER: Dict[str, object] = {}


def load_base(key: str, base_dir: Path) -> dict:
    """A base's fitted parameters, wiring, normalized APL weights, breadth (the bouton base's APL-off
    spike totals, `E10_joint_fit.bouton_base`, from the lab's input as in every experiment), the
    simulator kwargs and its ORN input."""
    from E10_joint_fit import TAU, base_kws, bouton_base, breadth_weights
    from E47_wiring_features import fitted_params
    b = dict(BASES[key])
    lab = H.apl16_orn_deltas()
    p_fit, fr = fitted_params(b['exp'], b['tag'])
    initial, wnorm, W, breadth = bouton_base(lab, base_dir, claw_thr=b['thr'], gain=b['gain'])
    orn = lab
    if b['input'] is not None:
        alt = pd.read_csv(H.shared_dir() / b['input'], header=[0, 1], index_col=0)
        if list(alt.index) != list(lab.index) or [c[1] for c in alt.columns] != list(lab.columns.get_level_values('odor')):
            raise ValueError(f"{b['input']}: glomeruli / odors differ from apl16_orn_deltas()")
        orn = lab.astype(float).copy()
        orn[:] = alt.to_numpy(dtype=float)
    sim_kws = dict(base_kws(b['thr']), _wPNKC=W, _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'],
        kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
    b.update(key=key, p_fit=p_fit, fit_row=fr, wnorm=wnorm, W=W, breadth=breadth, orn=orn, lab_orn=lab,
        wAPLKC=breadth_weights(wnorm['wAPLKC'], breadth, b['k']), sim_kws=sim_kws, pn_kws=base_kws(b['thr']))
    return b


def off_base_counts(b: dict, base_dir: Path) -> pd.DataFrame:
    """The APL-off spike counts of a gained base's bouton base (`bouton_base`'s cached run, whose
    spike totals are the breadth)."""
    if b['gain'] is None and b['thr'] is None:
        return pd.read_parquet(H.shared_dir() / 'data' / 'hal_outputs' / 'E3_drive_dependent_gain'
            / 'bouton_pnapl_x1_acute_off_spike_counts.parquet')
    return pd.read_parquet(base_dir / f"gain{b['gain']:g}_thr{b['thr']:g}_base_acute_off_spike_counts.parquet")


def _init_worker(shared: dict) -> None:
    _WORKER.update(shared)


def _frac(out) -> float:
    return float(out[0].astype(float).iloc[:, 0].mean())


def odor_job(job: dict) -> dict:
    """One odor of one arm of part B, simulated on its own (odors are independent at fixed scales; a
    one-odor simulation equals that odor's column of the 16-odor one, checked 2026-11-09): for arm
    'both' the odor's input scale (given, or bisected on its APL-off fraction), then the multiplier on
    the APL->KC and APL->bouton scales bisected (log scale) on its APL-on fraction. Returns the counts
    and responses at the multiplier, at multiplier 1 and with APL off, the history and the PN rates of
    the odor's input."""
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    from E32_pn_stage import glomerulus_pn_rates
    w = _WORKER
    p = dict(w['p_fit'])
    col = w['orn'].iloc[:, [job['j']]].astype(float)
    tol = 0.5 / w['n_kcs']
    hist = []

    def build(scale):
        sim, _ = make_fixed_param_kc_simulator(col * scale, initial_params=p, _wAPLKC=w['wAPLKC'], **w['sim_kws'])
        return sim
    off_p = dict(p, wAPLKC_scale=0.0, wAPLPN_scale=0.0)
    scale, sim, off = 1.0, None, None
    if job['arm'] == 'both':
        if job['scale'] is not None:
            scale = float(job['scale'])
        else:
            lo, hi, best = np.log(SCALE_LO), np.log(SCALE_HI), None
            for it in range(SCALE_ITERS):
                s = float(np.exp(0.5 * (lo + hi)))
                sim_s = build(s)
                o = sim_s(off_p)
                f = _frac(o)
                hist.append(dict(stage='scale', iteration=it, value=s, fraction=f, target=job['target_off']))
                if best is None or abs(f - job['target_off']) < best[0]:
                    best = (abs(f - job['target_off']), s, sim_s, o)
                if abs(f - job['target_off']) < tol:
                    break
                if f > job['target_off']:
                    hi = np.log(s)
                else:
                    lo = np.log(s)
            _, scale, sim, off = best
    if sim is None:
        sim = build(scale)
    if off is None:
        off = sim(off_p)
    on1 = sim(p)
    t = job['target_on']
    evals = [(1.0, on1, _frac(on1))]
    hist.append(dict(stage='multiplier', iteration=-1, value=1.0, fraction=evals[0][2], target=t))
    lo, hi = np.log(M_LO), np.log(M_HI)
    if abs(evals[0][2] - t) >= tol:
        if evals[0][2] > t:
            lo = 0.0
        else:
            hi = 0.0
        for it in range(M_ITERS):
            m = float(np.exp(0.5 * (lo + hi)))
            o = sim(dict(p, wAPLKC_scale=p['wAPLKC_scale'] * m, wAPLPN_scale=p['wAPLPN_scale'] * m))
            f = _frac(o)
            evals.append((m, o, f))
            hist.append(dict(stage='multiplier', iteration=it, value=m, fraction=f, target=t))
            if abs(f - t) < tol:
                break
            if f > t:
                lo = np.log(m)
            else:
                hi = np.log(m)
    m, on_m, f_m = min(evals, key=lambda e: abs(e[2] - t))
    at_bound = bool(abs(f_m - t) >= 2 * tol and (m <= M_LO * 1.05 or m >= M_HI / 1.05 or
        np.exp(lo) <= M_LO * 1.05 or np.exp(hi) >= M_HI / 1.05))
    pn = glomerulus_pn_rates(col * scale, dict(w['pn_kws']), p).iloc[:, 0]
    return dict(arm=job['arm'], odor=job['odor'], j=job['j'], scale=scale, multiplier=m, frac_on=f_m, frac_on_m1=evals[0][2],
        frac_off=_frac(off), target_on=t, target_off=job['target_off'], at_bound=at_bound, n_sims=len(evals) + 1,
        on_m=on_m, on_1=on1, off=off, history=hist, pn=pn)


def stats_context() -> dict:
    """What the stats tables' scoring needs (as the rate oracle's, E62)."""
    from E10_joint_fit import make_loss
    from E18_bouton_depletion import all_refs
    from E44_operating_point import CTRL_FRAC
    data_mats = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[ODORS].mean() for g in ('TNT3c', 'TNT')}
    orn_pairs, drive, ref = H.orn16_consensus_pairs(), H.consensus_orn_drive(), H.reference_pairs()
    return dict(data_mats=data_mats, data_rates=data_rates, orn_pairs=orn_pairs, drive=drive, ref=ref, refs=all_refs(),
        loss5=make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=0.05),
        loss_ctrl=make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=CTRL_FRAC),
        data_pairs=pd.read_csv(RESULTS / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b']))


def score_variant(ctx: dict, tag: str, meta: dict, on_out, off_out, W, orn, cal: float, suffix: str,
        res_dir: Path, out_dir: Path) -> dict:
    """The harness scoring of one variant (the stats-table columns of the rate oracle, E62), plus the
    loss decomposition under each readout (the base's calibrated parameter); appends to the part's
    tables (part<suffix>_variant_stats.csv, model_pair_distances_long_<suffix>.csv, parts_<suffix>.csv,
    shares_<suffix>.csv) and saves the counts and responses."""
    from E3_drive_dependent_gain import per_odor_extras
    from E4_substrate import score as e4_score
    from E12_breadth_apl import density_excess
    from E16_claw_saturation import spearman_with_refs
    from E26_claw_power import sharing_deficit
    from E35_divisive_apl import suppression_stats
    from E44_operating_point import fraction_stats, losses
    V = H.PRIMARY_MODEL_RDM
    row, on, off = e4_score(meta, on_out, off_out, W, orn, ctx['orn_pairs'], ctx['drive'], ctx['data_mats'], ctx['ref'])
    on_p, off_p = on[V], off[V]
    row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, ctx['drive'], ctx['data_rates']))
    for cond, s in (('on', on_out[1]), ('off', off_out[1])):
        row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
    row.update(spearman_with_refs(on_p, off_p, ctx['refs']))
    row.update(sharing_deficit(off_out[1], ctx['data_pairs']))
    row.update(suppression_stats(H.response_fractions(on_out[0]), H.response_fractions(off_out[0]), ctx['data_rates']))
    row.update(fraction_stats(on_out[0], off_out[0], ctx['data_rates']))
    row.update(losses(ctx['loss5'], ctx['loss_ctrl'], on_out, off_out))
    parts, shares = [], []
    for name, kind, param, data in READOUTS:
        p = cal if param == 'cal' else param
        st, t, _ = score(on_out[1], off_out[1], kind, p, data)
        parts.append(dict(tag=tag, readout=name, param=p, **st))
        shares.append(t.reset_index().assign(tag=tag, readout=name))
        if name == CAL_READOUT:
            row.update({f'cal__{k}': v for k, v in st.items() if k.startswith('apl_effect_')})
        if name == 'raw':
            row.update({f'raw__{k}': st[k] for k in ('apl_effect_per_odor', 'apl_effect_pair_specific', 'apl_effect_share_pearson')})
    for name, df in ((f'part{suffix}_variant_stats.csv', pd.DataFrame([row])), (f'parts_{suffix}.csv', pd.DataFrame(parts)),
            (f'shares_{suffix}.csv', pd.concat(shares, ignore_index=True))):
        p = res_dir / name
        df.to_csv(p, mode='a', header=not p.exists(), index=False)
    long_path = res_dir / f'model_pair_distances_long_{suffix}.csv'
    pd.concat({(tag, 'on'): on_p, (tag, 'acute_off'): off_p}, names=['tag', 'condition', 'odor_a', 'odor_b']).rename(
        'distance').reset_index().to_csv(long_path, index=False, mode='a', header=not long_path.exists())
    for cond, (r_, s_) in (('on', on_out), ('acute_off', off_out)):
        r_.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
        s_.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
    print(f"{tag}: rates on {row['rate_on']:.4f} off {row['rate_off']:.4f}; APL-effect loss raw {row[V + '__apl_effect_rmse']:.3f} "
        f"(per-odor {row['raw__apl_effect_per_odor']:.3f}, pair-specific {row['raw__apl_effect_pair_specific']:.3f}), calibrated "
        f"{row['cal__apl_effect_rmse']:.3f} (per-odor {row['cal__apl_effect_per_odor']:.3f}, pair-specific "
        f"{row['cal__apl_effect_pair_specific']:.3f}); rho {row[V + '__on_spearman']:.3f} / {row[V + '__off_spearman']:.3f}", flush=True)
    return row


def done_tags(res_dir: Path, suffix: str) -> set:
    p = res_dir / f'part{suffix}_variant_stats.csv'
    return set(pd.read_csv(p)['tag'].astype(str)) if p.exists() else set()


def odor_properties(b: dict, off_counts: pd.DataFrame, pn: pd.DataFrame) -> pd.DataFrame:
    """Per odor, what an APL rule could depend on: the PN drive of its input (mean PN rate over the
    glomeruli of the input), the number of glomeruli whose PN rate exceeds the per-claw threshold
    (40 Hz where the base has none), the share of its PN drive in the eight gained glomeruli, the share
    of its APL-off spikes in the KCs of the top 5 % by breadth (APL-off spike total over odors, the
    breadth weight's variable) and in its own top 1 % of KCs."""
    thr = b['thr'] if b['thr'] is not None else 40.0
    drv = [g for g in pn.index if g in set(b['lab_orn'].index)]
    pnd = pn.loc[drv]
    br = b['breadth']
    high = set(br[br >= br.quantile(0.95)].index)
    c = H.abbrev_columns(off_counts)
    kc = c.index.get_level_values('kc_id')
    in_high = np.asarray([k in high for k in kc])
    out = {}
    for o in ODORS:
        x = c[o].to_numpy(dtype=float)
        tot = x.sum()
        top = np.sort(x)[::-1][:max(1, int(np.ceil(0.01 * len(x))))].sum()
        out[o] = dict(pn_mean_rate=float(pnd[o].mean()), pn_n_above_claw_thr=int((pnd[o] > thr).sum()),
            pn_share_gained=float(pnd.loc[[g for g in GAINED_GLOMS if g in pnd.index], o].clip(lower=0).sum() / pnd[o].clip(lower=0).sum()),
            off_spike_share_high_breadth=float(x[in_high].sum() / tot) if tot > 0 else np.nan,
            off_spike_share_top1=float(top / tot) if tot > 0 else np.nan, off_spikes_total=float(tot))
    return pd.DataFrame(out).T.rename_axis('odor')


def part_b(args) -> None:
    import time
    from E3_drive_dependent_gain import rate
    from E4_substrate import kc_by_glomerulus
    res_dir = H.results_dir(EXPERIMENT)
    out_dir = H.HERE / 'outputs' / EXPERIMENT
    base_dir = out_dir / 'bases'
    base_dir.mkdir(parents=True, exist_ok=True)
    for p in (H.HERE / 'outputs' / 'E62_rate_oracle' / 'bases').glob('*.parquet'):
        if not (base_dir / p.name).exists():
            (base_dir / p.name).write_bytes(p.read_bytes())
    e65 = pd.read_csv(E65_TABLE, low_memory=False)
    cal_table = e65[e65.readout == CAL_READOUT].set_index(['folder', 'tag'])['param']
    e62_scales = pd.read_csv(RESULTS / 'E62_rate_oracle' / 'scales.csv')
    ctx = stats_context()
    data_rates = ctx['data_rates']
    done = done_tags(res_dir, 'B')
    for key in args.bases:
        tags = {v: f'O_{key}_{v}' for v in B_VARIANTS}
        if all(t in done for t in tags.values()):
            continue
        t0 = time.time()
        b = load_base(key, base_dir)
        cal = float(cal_table.loc[b['cal']])
        given = None
        if b['e62'] is not None:
            s = e62_scales[(e62_scales.model == b['e62']) & (e62_scales.arm == 'off')].set_index('odor')['scale']
            given = s.reindex(ODORS)
        odors = list(H.abbrev_columns(b['orn']).columns)
        jobs = [dict(arm=arm, odor=o, j=j, scale=(None if given is None else float(given[o])) if arm == 'both' else None,
            target_on=float(data_rates['TNT3c'][o]), target_off=float(data_rates['TNT'][o]))
            for arm in ('apl', 'both') for j, o in enumerate(odors) if o in (args.odors or odors)]
        shared = dict(orn=b['orn'], p_fit=b['p_fit'], wAPLKC=b['wAPLKC'], sim_kws=b['sim_kws'], pn_kws=b['pn_kws'],
            n_kcs=len(off_base_counts(b, base_dir)))
        cache = out_dir / f'partB_{key}_odor_results.pkl'
        if cache.exists() and not args.odors:
            results = pd.read_pickle(cache)
            print(f'{key}: odor results from {cache}', flush=True)
        else:
            print(f'{key}: {len(jobs)} odor jobs on {args.jobs} processes', flush=True)
            results = []
            with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker, initargs=(shared,)) as ex:
                for r in ex.map(odor_job, jobs):
                    print(f"  {key} {r['arm']} {r['odor']}: scale {r['scale']:.3f}, multiplier {r['multiplier']:.3f}, APL-on fraction "
                        f"{r['frac_on']:.4f} (target {r['target_on']:.4f}; at multiplier 1 {r['frac_on_m1']:.4f}), APL-off "
                        f"{r['frac_off']:.4f} (target {r['target_off']:.4f}){' AT BOUND' if r['at_bound'] else ''}", flush=True)
                    results.append(r)
            if args.odors:
                print('smoke run (--odors): nothing assembled')
                continue
            pd.to_pickle(results, cache)
        res = {(r['arm'], r['odor']): r for r in results}
        per_odor, hist_rows = [], []
        pn_by_arm = {arm: pd.concat([res[(arm, o)]['pn'].rename(o) for o in odors], axis=1) for arm in ('apl', 'both')}
        for r in results:
            hist_rows += [dict(base=key, arm=r['arm'], odor=r['odor'], **h) for h in r['history']]
        for v, (arm, which) in B_VARIANTS.items():
            tag = tags[v]
            cols = [res[(arm, o)] for o in odors]
            on_r = pd.concat([c['on_m' if which == 'mo' else 'on_1'][0] for c in cols], axis=1)
            on_s = pd.concat([c['on_m' if which == 'mo' else 'on_1'][1] for c in cols], axis=1)
            off_r = pd.concat([c['off'][0] for c in cols], axis=1)
            off_s = pd.concat([c['off'][1] for c in cols], axis=1)
            scales = pd.Series([c['scale'] for c in cols], index=odors)
            mult = pd.Series([c['multiplier'] if which == 'mo' else 1.0 for c in cols], index=odors)
            if tag not in done:
                meta = dict(variant=v, tag=tag, part='B', model=key, arm=arm, base='bouton', claw_thr=b['thr'], k=b['k'], gain=b['gain'],
                    kc_kc_inh=b['p_fit']['wKCKC_scale'], ref_fit=b['tag'], ref_experiment=b['exp'], ref_fit_loss=float(b['fit_row']['loss']),
                    thr_factor_vs_fit=1.0, orn_input=b['input'] or 'lab', target_rate=float(H.data_target('ctrl_frac')),
                    n_kcs=len(on_r), n_glomeruli=b['W'].shape[1], rate_on=rate(on_r), rate_off=rate(off_r),
                    mean_claws_per_kc=float(kc_by_glomerulus(b['W']).sum(axis='columns').mean()),
                    scale_min=float(scales.min()), scale_max=float(scales.max()), multiplier_min=float(mult.min()),
                    multiplier_max=float(mult.max()), multiplier_geomean=float(np.exp(np.log(mult).mean())),
                    cal_readout_param=cal, seconds=time.time() - t0, **b['p_fit'])
                meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
                score_variant(ctx, tag, meta, (on_r, on_s), (off_r, off_s), b['W'], b['lab_orn'], cal, 'B', res_dir, out_dir)
            if v in ('apl', 'both'):
                props = odor_properties(b, off_s, pn_by_arm[arm])
                for o in odors:
                    c = res[(arm, o)]
                    per_odor.append(dict(base=key, variant=v, arm=arm, odor=o, scale=c['scale'], multiplier=c['multiplier'],
                        frac_on=c['frac_on'], frac_on_m1=c['frac_on_m1'], frac_off=c['frac_off'], target_on=c['target_on'],
                        target_off=c['target_off'], at_bound=c['at_bound'], n_sims=c['n_sims'], **props.loc[o].to_dict()))
        for name, rows in (('multipliers.csv', per_odor), ('multiplier_history.csv', hist_rows)):
            p = res_dir / name
            pd.DataFrame(rows).to_csv(p, mode='a', header=not p.exists(), index=False)
        print(f'{key} done in {time.time() - t0:.0f} s', flush=True)
    write_provenance(res_dir, 'B', args, dict(bases={k: {kk: vv for kk, vv in BASES[k].items()} for k in args.bases},
        scale_bracket=[SCALE_LO, SCALE_HI], scale_iters=SCALE_ITERS, multiplier_bracket=[M_LO, M_HI], multiplier_iters=M_ITERS))


def part_c(args) -> None:
    """The breadth exponent at 0-3 under both definitions of a KC's breadth on the reference models,
    the APL->KC scale re-tuned (bisection on a factor, log scale) to hold the fitted APL-on response
    rate; APL off is the same for every exponent."""
    import time
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    from E3_drive_dependent_gain import rate
    from E4_substrate import kc_by_glomerulus
    from E10_joint_fit import breadth_weights
    res_dir = H.results_dir(EXPERIMENT)
    out_dir = H.HERE / 'outputs' / EXPERIMENT
    base_dir = out_dir / 'bases'
    base_dir.mkdir(parents=True, exist_ok=True)
    for p in (H.HERE / 'outputs' / 'E62_rate_oracle' / 'bases').glob('*.parquet'):
        if not (base_dir / p.name).exists():
            (base_dir / p.name).write_bytes(p.read_bytes())
    e65 = pd.read_csv(E65_TABLE, low_memory=False)
    cal_table = e65[e65.readout == CAL_READOUT].set_index(['folder', 'tag'])['param']
    ctx = stats_context()
    done = done_tags(res_dir, 'C')
    for key in args.c_bases:
        todo = [(d, k) for d, k in C_CONFIGS if f'C_{key}_{d}_k{k:g}' not in done]
        if not todo:
            continue
        b = load_base(key, base_dir)
        cal = float(cal_table.loc[b['cal']])
        p = dict(b['p_fit'])
        target = float(b['fit_row']['rate_on'])
        # the n_odors breadth: odors with >= 2 spikes in the same APL-off run as the spike breadth
        off_base = off_base_counts(b, base_dir)
        n_odors = pd.Series((off_base.astype(float) >= 2).sum(axis='columns').to_numpy(),
            index=off_base.index.get_level_values(H.KC_ID_LEVEL))
        assert n_odors.index.equals(b['breadth'].index)
        breadths = {'spikes': b['breadth'], 'n_odors': n_odors}
        off_out = None
        for bdef, k in todo:
            tag = f'C_{key}_{bdef}_k{k:g}'
            t0 = time.time()
            wAPLKC = breadth_weights(b['wnorm']['wAPLKC'], breadths[bdef], k)
            sim, _ = make_fixed_param_kc_simulator(b['orn'], initial_params=p, _wAPLKC=wAPLKC, **b['sim_kws'])
            if off_out is None:
                off_out = sim(dict(p, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
            lo, hi, evals = np.log(1 / 64), np.log(64.0), []
            tol = 0.5 / off_out[0].size
            for it in range(C_ITERS):
                g = 1.0 if it == 0 else float(np.exp(0.5 * (lo + hi)))
                out = sim(dict(p, wAPLKC_scale=p['wAPLKC_scale'] * g))
                r = rate(out[0])
                evals.append((g, out, r))
                print(f'  {tag} iteration {it}: APL->KC factor {g:.4f}, APL-on rate {r:.4f} (target {target:.4f})', flush=True)
                if abs(r - target) < tol:
                    break
                if r > target:
                    lo = np.log(g)
                else:
                    hi = np.log(g)
            g, on_out, r = min(evals, key=lambda e: abs(e[2] - target))
            meta = dict(variant=f'{bdef}_k{k:g}', tag=tag, part='C', model=key, breadth_def=bdef, k=k, base='bouton', claw_thr=b['thr'],
                gain=b['gain'], kc_kc_inh=p['wKCKC_scale'], ref_fit=b['tag'], ref_experiment=b['exp'], ref_fit_loss=float(b['fit_row']['loss']),
                thr_factor_vs_fit=1.0, target_rate=target, apl_kc_factor=g, n_sims=len(evals), n_kcs=len(on_out[0]), n_glomeruli=b['W'].shape[1],
                rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), mean_claws_per_kc=float(kc_by_glomerulus(b['W']).sum(axis='columns').mean()),
                cal_readout_param=cal, seconds=time.time() - t0, **dict(p, wAPLKC_scale=p['wAPLKC_scale'] * g))
            meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
            score_variant(ctx, tag, meta, on_out, off_out, b['W'], b['lab_orn'], cal, 'C', res_dir, out_dir)
            del sim
    write_provenance(res_dir, 'C', args, dict(bases=list(args.c_bases), configs=C_CONFIGS, iters=C_ITERS, factor_bracket=[1 / 64, 64]))


PROPERTIES = {'pn_mean_rate': 'mean PN rate of the input', 'pn_n_above_claw_thr': 'glomeruli above the per-claw threshold',
    'pn_share_gained': 'share of PN drive in the eight gained glomeruli',
    'off_spike_share_high_breadth': 'share of APL-off spikes in the top 5 % of KCs by breadth',
    'off_spike_share_top1': "share of APL-off spikes in the odor's top 1 % of KCs", 'frac_on_m1': 'APL-on fraction at multiplier 1',
    'frac_off': 'APL-off fraction'}


def realism_extras(tag: str) -> Dict[str, float]:
    """Readout-free realism statistics of a saved variant (as the compressive readout experiment, E65,
    reports for the data): per condition, the responders-only participation ratio and top-1 % share
    (mean over odors), the peak spike count, and the share of KCs responding to no odor and to 8 or
    more odors."""
    from E65_spike_count_readout import odors_per_unit
    out_dir = H.HERE / 'outputs' / EXPERIMENT
    out = {}
    for cond, short in (('on', 'on'), ('acute_off', 'off')):
        p = out_dir / f'{tag}_{cond}_spike_counts.parquet'
        if not p.exists():
            continue
        m = pd.read_parquet(p)
        a = H.amplitude_stats(m.where(m >= RESP_MIN, 0)).mean(numeric_only=True)
        o = odors_per_unit(m >= RESP_MIN)
        out.update({f'participation_ratio_resp_{short}': float(a['participation_ratio']), f'top1_share_resp_{short}': float(a['top1_share']),
            f'max_count_{short}': float(m.to_numpy().max()), f'odors_0_{short}': o['odors_0'], f'odors_8plus_{short}': o['odors_8plus']})
    return out


def part_s(args) -> None:
    """Merges parts B and C into the experiment's stats_variants.csv and model_pair_distances_long.csv
    (read by apl_effect_scores.py and the progress plot) and writes the parts' tables: the loss and its
    parts under raw counts and the calibrated readout per variant, beside its base; the per-odor
    multipliers; their Spearman correlations with the odor properties."""
    res_dir = H.results_dir(EXPERIMENT)
    st = pd.concat([pd.read_csv(res_dir / f'part{s}_variant_stats.csv') for s in ('B', 'C') if (res_dir / f'part{s}_variant_stats.csv').exists()],
        ignore_index=True)
    # merged under the name apl_effect_scores.py and archive_model_outputs.py read; the per-part tables are
    # named so that the archive's 'stats*.csv' glob does not list their rows twice
    st.to_csv(res_dir / 'stats_variants.csv', index=False)
    pd.concat([pd.read_csv(res_dir / f'model_pair_distances_long_{s}.csv') for s in ('B', 'C')
        if (res_dir / f'model_pair_distances_long_{s}.csv').exists()], ignore_index=True).to_csv(res_dir / 'model_pair_distances_long.csv', index=False)
    parts = pd.concat([pd.read_csv(res_dir / f'parts_{s}.csv') for s in ('B', 'C') if (res_dir / f'parts_{s}.csv').exists()], ignore_index=True)
    V = H.PRIMARY_MODEL_RDM
    keep = ['tag', 'part', 'model', 'variant', 'rate_on', 'rate_off', 'rate_ratio_off_over_on', 'spearman_rate_on_vs_data',
        f'{V}__on_spearman', f'{V}__off_spearman', f'{V}__block_fruity_small_iso', f'{V}__on_spearman_resid_vs_vcf_hyp',
        f'{V}__off_spearman_resid_vs_vcf_hyp', 'multiplier_min', 'multiplier_max', 'apl_kc_factor']
    wide = parts.pivot_table(index='tag', columns='readout', values=['apl_effect_rmse', 'apl_effect_mean', 'apl_effect_rmse_centered',
        'apl_effect_per_odor', 'apl_effect_pair_specific', 'apl_effect_spearman', 'apl_effect_share_pearson'])
    wide.columns = [f'{r}__{k}' for k, r in wide.columns]
    table = st[[c for c in keep if c in st.columns]].set_index('tag').join(wide)
    from apl_effect_scores import realism_flag
    table['realism_flag'] = [realism_flag(r.rate_on, r.rate_off, r.spearman_rate_on_vs_data) for r in table.itertuples()]
    if 'spearman_rate_off_vs_data' in st.columns:
        table['spearman_rate_off_vs_data'] = st.set_index('tag')['spearman_rate_off_vs_data']
    table = table.join(pd.DataFrame({t: realism_extras(t) for t in table.index}).T)
    table.to_csv(res_dir / 'parts_bc_table.csv')
    mp = res_dir / 'multipliers.csv'
    corr_rows = []
    if mp.exists():
        mu = pd.read_csv(mp)
        for (base, variant), g in mu.groupby(['base', 'variant']):
            for prop in PROPERTIES:
                corr_rows.append(dict(base=base, variant=variant, property=prop,
                    spearman=H.spearman(np.log(g.multiplier), g[prop]), pearson=corr(np.log(g.multiplier), g[prop])))
        pd.DataFrame(corr_rows).to_csv(res_dir / 'multiplier_correlates.csv', index=False)
    write_table_bc(res_dir, table, pd.read_csv(mp) if mp.exists() else None, pd.DataFrame(corr_rows))
    if mp.exists():
        figure_multipliers(res_dir, pd.read_csv(mp))
    print(table.round(3).to_string())


def figure_multipliers(res_dir: Path, mu: pd.DataFrame) -> None:
    """The per-odor APL multipliers of each base, inputs unscaled (open) and scaled to the APL-off
    fractions (filled), log scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bases = list(dict.fromkeys(mu.base))
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(ODORS))
    for i, base in enumerate(bases):
        for v, filled in (('apl', False), ('both', True)):
            g = mu[(mu.base == base) & (mu.variant == v)].set_index('odor').reindex(ODORS)
            ax.plot(x + 0.12 * (i - (len(bases) - 1) / 2), g.multiplier, 'o', ms=5, c=f'C{i}',
                mfc=f'C{i}' if filled else 'none', label=f"{BASES[base]['label']}{', inputs scaled' if filled else ''}")
    ax.axhline(1, c='0.6', lw=0.8)
    ax.set_yscale('log')
    ax.set_xticks(x, ODORS, rotation=60)
    ax.set_ylabel('APL multiplier that gives the control flies\'\nAPL-on response fraction (× fitted scale)')
    ax.legend(fontsize=7, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(res_dir / 'E66_multipliers.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_table_bc(res_dir: Path, table: pd.DataFrame, mu: Optional[pd.DataFrame], corr_df: pd.DataFrame) -> None:
    V = H.PRIMARY_MODEL_RDM
    dt = H.data_target
    L = ['# E66 parts B and C', '', 'Loss and parts: raw counts / calibrated saturating readout (the base\'s parameter). Lower is better '
         'for the loss and its parts; data floors (resampled recordings): loss 0.039, per-odor part 0.020, pair-specific part 0.021.', '',
         '| variant | APL-on response rate | APL-off / APL-on | response rate ρ, APL on | loss | mean APL effect | per-odor part | pair-specific part | ρ on the per-pair APL effect | r of shares | ρ on odor-odor distances, APL on / off | block | realism flag |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for tag, r in table.iterrows():
        two = lambda k, spec='.3f': f"{fmt(r.get(f'raw__{k}'), spec)} / {fmt(r.get(f'{CAL_READOUT}__{k}'), spec)}"
        L.append(f"| {tag} | {fmt(r.rate_on)} | {fmt(r.rate_ratio_off_over_on, '.2f')} | {fmt(r.spearman_rate_on_vs_data, '.2f')} | "
                 f"{two('apl_effect_rmse')} | {two('apl_effect_mean', '+.3f')} | {two('apl_effect_per_odor')} | {two('apl_effect_pair_specific')} | "
                 f"{two('apl_effect_spearman', '.2f')} | {two('apl_effect_share_pearson', '.2f')} | {fmt(r[f'{V}__on_spearman'], '.2f')} / "
                 f"{fmt(r[f'{V}__off_spearman'], '.2f')} | {fmt(r[f'{V}__block_fruity_small_iso'], '+.3f')} | {r.realism_flag or '-'} |")
    L.append(f"| KC data | {fmt(dt('ctrl_frac'))} | 2.2 | ceiling {fmt(dt('ctrl_rate_rho_ceiling'), '.2f')} | floor 0.039 | {fmt(dt('apl_effect_mean'), '+.3f')} | "
             f"floor 0.020 | floor 0.021 | ceiling {fmt(dt('apl_effect_rho_ceiling'), '.2f')} | - | ceilings {fmt(dt('ctrl_ceiling'), '.2f')} / "
             f"{fmt(dt('tnt_ceiling'), '.2f')} | {fmt(dt('block'), '+.3f')} | - |")
    if mu is not None and len(mu):
        L += ['', '## Per-odor APL multipliers (×, on the APL->KC and APL->bouton scales)', '',
              '| base | variant | ' + ' | '.join(ODORS) + ' |', '|---|---|' + '---:|' * len(ODORS)]
        for (base, variant), g in mu.groupby(['base', 'variant'], sort=False):
            g = g.set_index('odor').reindex(ODORS)
            L.append(f'| {base} | {variant} | ' + ' | '.join(f"{m:.2f}{'*' if b else ''}" for m, b in zip(g.multiplier, g.at_bound)) + ' |')
        L += ['', '(* at the bracket, the fraction not reached)', '', '## Spearman of log multiplier with odor properties', '',
              '| base | variant | ' + ' | '.join(PROPERTIES.values()) + ' |', '|---|---|' + '---:|' * len(PROPERTIES)]
        for (base, variant), g in corr_df.groupby(['base', 'variant'], sort=False):
            g = g.set_index('property').reindex(list(PROPERTIES))
            L.append(f'| {base} | {variant} | ' + ' | '.join(fmt(x, '+.2f') for x in g.spearman) + ' |')
    (res_dir / 'comparison_table_bc.md').write_text('\n'.join(L) + '\n')


def write_provenance(res_dir: Path, part: str, args, extra: dict) -> None:
    p = res_dir / 'provenance.json'
    prov = json.loads(p.read_text()) if p.exists() else {}
    prov[f'part{part}'] = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script=f'E66_per_odor_apl_profile.py --part {part}', argv=vars(args),
        model_provenance=H.model_provenance(orn_deltas=H.apl16_orn_deltas()), **extra)
    p.write_text(json.dumps(prov, indent=1, default=str) + '\n')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=['A', 'B', 'C', 'S'], required=True, help='S: merge parts B and C and write their tables')
    ap.add_argument('--jobs', type=int, default=8, help='worker processes for parts A and B')
    ap.add_argument('--boot', type=int, default=1000, help='recording resamples for the data\'s shares and floors')
    ap.add_argument('--bases', nargs='+', default=list(BASES), choices=list(BASES), help='part B bases')
    ap.add_argument('--c-bases', nargs='+', default=list(REF_BASES), choices=list(BASES), help='part C bases')
    ap.add_argument('--odors', nargs='+', default=None, help='part B smoke run: only these odors, nothing assembled')
    args = ap.parse_args()
    {'A': part_a, 'B': part_b, 'C': part_c, 'S': part_s}[args.part](args)


if __name__ == '__main__':
    main()
