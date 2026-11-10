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

Part B (the oracle) and part C (the breadth exponent): see `part_b` and `part_c`.

Outputs, <shared>/results/E66_per_odor_apl_profile/: data_shares.csv, data_floor_parts.csv,
data_share_correlates.csv, by_variant_parts.csv, shares_long.csv, parts_summary.csv, key_models.csv,
worst_pairs.csv, worst_pair_kcs.csv, oracles.csv, comparison_table.md, figures, provenance.json.
Usage: ./venv/bin/python E66_per_odor_apl_profile.py --part A [--jobs N] [--boot 1000]
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


def part_b(args) -> None:
    raise NotImplementedError('part B (the per-odor APL oracle) is next')


def part_c(args) -> None:
    raise NotImplementedError('part C (the breadth exponent) is next')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=['A', 'B', 'C'], required=True)
    ap.add_argument('--jobs', type=int, default=8, help='worker processes for part A')
    ap.add_argument('--boot', type=int, default=1000, help='recording resamples for the data\'s shares and floors')
    args = ap.parse_args()
    {'A': part_a, 'B': part_b, 'C': part_c}[args.part](args)


if __name__ == '__main__':
    main()
