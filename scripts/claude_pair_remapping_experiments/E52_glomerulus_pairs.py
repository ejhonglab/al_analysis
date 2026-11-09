#!/usr/bin/env python3
"""E52: the glomerulus pairs behind the natural-source alignment.
Re-fitted on any of the four wiring shuffles, the reference model recovers the data's block
on some draws but never the natural-source alignment with APL off (0.20-0.32 on every
re-fitted draw of E47 and E50, against 0.37 on the real wiring and 0.36 in the KC data). All
four shuffles keep the wiring's margins (claw counts per KC and per glomerulus, or the binary
KC x glomerulus degrees, per KC type or not) and destroy which glomeruli share a KC beyond
what the margins imply, so the alignment rides on glomerulus co-input structure. This asks
which pairs carry it.
Part A (no simulation): for each pair of the 54 glomeruli of the bouton base's claw wiring
(1732 KCs, 9867 claws), the number of KCs with claws from both, against `--draws` draws of
each shuffle class (`al_analysis.wpnkc_shuffles.shuffle_wPNKC`, the draws the fits used):
excess over the null mean, z, empirical p; "over-shared" = z >= 2 against every class,
"under-shared" = z <= -2 against every class; the same per KC type against the within-type
classes.
Part B (no simulation): a wiring-sharing score per odor pair, the sum over glomerulus pairs
of the co-input matrix weighted by the two odors' ORN input to the two glomeruli (deltas
clipped at 0, each odor normalized to unit sum; `raw` keeps the Hz), from the observed
counts, each class's null mean and the excess (observed minus null mean) and z; on the 66
megamat pairs its Spearman with the hyperbolic natural-source distance (vcf_hyp), the ECFP
chemical distance, the ORN distance, the KC data's residuals around the linear ORN
prediction (control flies, APL output blocked (TNT), Remy's wild-type somata) and the fitted
real-wiring models' APL-off / APL-on residual minus the mean of E47's shuffle draws' (what
the real wiring contributes in the model), with odor-permutation p-values; the alignment
(rho of the data's residual with vcf_hyp) before and after partialling out the score; the
score on the fruity x small-ester pairs against the other between-group pairs.
Part C (simulation): `al_analysis.wpnkc_shuffles.lesion_coinput` removes only the
over-shared pairs' excess (claw swaps within KC type that keep every margin; each pair
brought to its null mean under the within-type Curveball) and, as the control, the same
excess from count-matched pairs that are neither over- nor under-shared, with the
over-shared pairs protected; the three reference models of E47's part A (the eight glomeruli
at 1.5 at 40 and 70 Hz, the eight at 1.25 at 70 Hz) at their fitted scales with the somatic
threshold re-tuned to 6.8 % on `--seeds` draws of each lesion, scored as E47 scores its
variants, beside E47's real-wiring and shuffle-class rows.
Outputs: <shared>/results/E52_glomerulus_pairs/ (coinput_pairs.csv, coinput_by_type.csv,
coinput_null_draws.npz, odor_pair_scores.csv, score_correlations.csv, alignment_partial.csv,
block_scores.csv, lesion_pairs.csv, lesion_outcome.csv, stats_variants.csv, comparison.csv,
comparison_table.md, provenance.json, the figures) and ./outputs/E52_glomerulus_pairs/
(lesioned wirings, swaps, per-variant responses / spike counts, partC/ caches).
Usage: ./venv/bin/python E52_glomerulus_pairs.py --part A [--draws 200]
       ./venv/bin/python E52_glomerulus_pairs.py --part B
       OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E52_glomerulus_pairs.py --part C [--seeds 0 1 2] [--models ...]
       ./venv/bin/python E52_glomerulus_pairs.py --summary
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H
from E47_wiring_features import (MODELS, SHUFFLES, CLASS_LABEL, STATS, V, fitted_params, SCALE_KEYS,
    CTRL_FRAC, data_values)

EXPERIMENT = 'E52_glomerulus_pairs'
E47 = 'E47_wiring_features'
NULL_CLASS = 'curveball_within_type'   # the class keeping the most of the margins: the reference null
Z_OVER = 2.0
TYPE_CLASSES = ('claw_permutation_within_type', 'curveball_within_type')
PLAIN_CLASSES = ('claw_permutation', 'curveball')   # the type-ignoring classes; their within-type twins keep the type structure
WEIGHTINGS = ('norm', 'raw')
LESIONS = ('lesion_targeted', 'lesion_control')
LESION_LABEL = {'lesion_targeted': 'over-shared pairs lesioned', 'lesion_control': 'matched control pairs lesioned'}


def base_wiring() -> pd.DataFrame:
    """The bouton base's claw x bouton wiring (the shared inputs copy; 0/1, columns
    (glomerulus, pn_id, bouton_id))."""
    return pd.read_parquet(H.shared_dir() / 'data' / 'model_outputs' / 'inputs' / 'wPNKC_bouton_base.parquet')


def triu_pairs(gloms: pd.Index) -> List[Tuple[str, str]]:
    return [(gloms[i], gloms[j]) for i in range(len(gloms)) for j in range(i + 1, len(gloms))]


def upper(mat: pd.DataFrame) -> np.ndarray:
    """The strict upper triangle of a square DataFrame, in `triu_pairs` order."""
    v = mat.to_numpy()
    return v[np.triu_indices(len(v), k=1)]


def square(values: np.ndarray, gloms: pd.Index, diag: float = 0.0) -> pd.DataFrame:
    """A symmetric DataFrame from strict-upper-triangle values (zero diagonal unless given)."""
    n = len(gloms)
    m = np.zeros((n, n), dtype=float)
    m[np.triu_indices(n, k=1)] = values
    m = m + m.T
    np.fill_diagonal(m, diag)
    return pd.DataFrame(m, index=gloms, columns=gloms)


# ---------------------------------------------------------------------------
# part A
def part_a(res_dir: Path, n_draws: int) -> pd.DataFrame:
    from al_analysis.wpnkc_shuffles import NAMED_SHUFFLES, shuffle_wPNKC, coinput_counts
    W = base_wiring()
    types = W.index.get_level_values('kc_type')
    type_names = [t for t in pd.unique(types)]
    obs = coinput_counts(W)
    gloms = obs.index
    pairs = triu_pairs(gloms)
    n_kcs = pd.Series(np.diag(obs.to_numpy()), index=gloms)
    obs_u = upper(obs).astype(int)
    obs_type = {t: upper(coinput_counts(W[types == t])).astype(int) for t in type_names}
    print(f'{len(gloms)} glomeruli, {len(pairs)} pairs, {int(n_kcs.max())} KCs on the largest glomerulus; '
        f'{n_draws} draws per class', flush=True)
    nulls: Dict[str, np.ndarray] = {}
    nulls_type: Dict[Tuple[str, str], np.ndarray] = {}
    deg_type_obs = {t: np.diag(coinput_counts(W[types == t]).to_numpy()).astype(int) for t in type_names}
    deg_type_null: Dict[Tuple[str, str], np.ndarray] = {}
    for cls in SHUFFLES:
        method, within = NAMED_SHUFFLES[cls]
        t0 = time.time()
        draws = np.zeros((n_draws, len(pairs)), dtype=np.int32)
        draws_t = {t: np.zeros((n_draws, len(pairs)), dtype=np.int32) for t in type_names} if cls in TYPE_CLASSES else {}
        deg_t = {t: np.zeros((n_draws, len(gloms)), dtype=np.int32) for t in type_names} if cls in PLAIN_CLASSES else {}
        for d in range(n_draws):
            S = shuffle_wPNKC(W, method, within_kc_type=within, seed=1000 + d)
            draws[d] = upper(coinput_counts(S))
            for t in draws_t:
                draws_t[t][d] = upper(coinput_counts(S[types == t]))
            for t in deg_t:
                deg_t[t][d] = np.diag(coinput_counts(S[types == t]).to_numpy())
        nulls[cls] = draws
        for t, a in draws_t.items():
            nulls_type[(cls, t)] = a
        for t, a in deg_t.items():
            deg_type_null[(cls, t)] = a
        print(f'  {cls}: {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(res_dir / 'coinput_null_draws.npz', observed=obs_u, glomeruli=np.array(gloms, dtype=object),
        **{f'null_{c}': a for c, a in nulls.items()},
        **{f'observed_{t}': a for t, a in obs_type.items()},
        **{f'null_{c}_{t}': a for (c, t), a in nulls_type.items()})
    orn_gloms = set(H.abbrev_columns(H.apl16_orn_deltas()).index)
    # the type structure: each glomerulus' number of KCs of each type against the type-ignoring
    # classes (the within-type classes keep these numbers exactly)
    n_type = {t: int((W.index.droplevel([l for l in W.index.names if l not in ('kc_id', 'kc_type')]).drop_duplicates()
        .get_level_values('kc_type') == t).sum()) for t in type_names}
    trows = []
    for t in type_names:
        for i, g in enumerate(gloms):
            r = dict(glomerulus=g, kc_type=t, n_kcs_of_type=n_type[t], observed=int(deg_type_obs[t][i]),
                frac_of_type=deg_type_obs[t][i] / n_type[t], in_orn_input=g in orn_gloms)
            for cls in PLAIN_CLASSES:
                a = deg_type_null[(cls, t)][:, i]
                mean, sd = a.mean(), a.std(ddof=1)
                r[f'{cls}_null_mean'] = mean; r[f'{cls}_null_sd'] = sd; r[f'{cls}_excess'] = r['observed'] - mean
                r[f'{cls}_z'] = (r['observed'] - mean) / sd if sd > 0 else (0.0 if r['observed'] == mean else np.sign(r['observed'] - mean) * np.inf)
            r['min_abs_z'] = min(abs(r[f'{cls}_z']) for cls in PLAIN_CLASSES)
            r['sign'] = int(np.sign(r[f'{PLAIN_CLASSES[0]}_excess']))
            trows.append(r)
    type_bias = pd.DataFrame(trows)
    type_bias.to_csv(res_dir / 'glomerulus_type_bias.csv', index=False)
    tb = type_bias[type_bias.min_abs_z >= Z_OVER]
    print(f'glomerulus x KC-type counts beyond the type-ignoring nulls (|z| >= {Z_OVER:g} against both): {len(tb)} of {len(type_bias)}', flush=True)
    df = pd.DataFrame({'glom_a': [p[0] for p in pairs], 'glom_b': [p[1] for p in pairs],
        'n_kcs_a': [int(n_kcs[p[0]]) for p in pairs], 'n_kcs_b': [int(n_kcs[p[1]]) for p in pairs],
        'in_orn_input': [(p[0] in orn_gloms) and (p[1] in orn_gloms) for p in pairs], 'observed': obs_u})
    zs = []
    for cls, draws in nulls.items():
        mean = draws.mean(axis=0); sd = draws.std(axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = np.where(sd > 0, (obs_u - mean) / sd, np.where(obs_u == mean, 0.0, np.sign(obs_u - mean) * np.inf))
        df[f'{cls}_null_mean'] = mean; df[f'{cls}_null_sd'] = sd; df[f'{cls}_excess'] = obs_u - mean
        df[f'{cls}_z'] = z
        df[f'{cls}_p_upper'] = (1 + (draws >= obs_u[None, :]).sum(axis=0)) / (n_draws + 1)
        df[f'{cls}_p_lower'] = (1 + (draws <= obs_u[None, :]).sum(axis=0)) / (n_draws + 1)
        zs.append(z)
    zs = np.vstack(zs)
    df['min_z'] = zs.min(axis=0); df['max_z'] = zs.max(axis=0)
    df['over_shared'] = (zs >= Z_OVER).all(axis=0)
    df['under_shared'] = (zs <= -Z_OVER).all(axis=0)
    df = df.sort_values('min_z', ascending=False).reset_index(drop=True)
    df.to_csv(res_dir / 'coinput_pairs.csv', index=False)
    rows = []
    for t in type_names:
        for i, p in enumerate(pairs):
            r = dict(kc_type=t, glom_a=p[0], glom_b=p[1], observed=int(obs_type[t][i]))
            for cls in TYPE_CLASSES:
                a = nulls_type[(cls, t)][:, i]
                mean, sd = a.mean(), a.std(ddof=1)
                r[f'{cls}_null_mean'] = mean; r[f'{cls}_null_sd'] = sd; r[f'{cls}_excess'] = r['observed'] - mean
                r[f'{cls}_z'] = (r['observed'] - mean) / sd if sd > 0 else (0.0 if r['observed'] == mean else np.sign(r['observed'] - mean) * np.inf)
            r['min_z'] = min(r[f'{cls}_z'] for cls in TYPE_CLASSES)
            rows.append(r)
    by_type = pd.DataFrame(rows).sort_values(['kc_type', 'min_z'], ascending=[True, False])
    by_type.to_csv(res_dir / 'coinput_by_type.csv', index=False)
    over = df[df.over_shared]; under = df[df.under_shared]
    print(f'over-shared (z >= {Z_OVER:g} against every class): {len(over)} of {len(df)} pairs '
        f'({int(over.in_orn_input.sum())} with both glomeruli in the ORN input); under-shared: {len(under)}', flush=True)
    print('glomeruli in over-shared pairs:', pd.concat([over.glom_a, over.glom_b]).value_counts().head(12).to_dict(), flush=True)
    pd.set_option('display.width', 250)
    print(over[['glom_a', 'glom_b', 'observed', f'{NULL_CLASS}_null_mean', 'min_z', 'in_orn_input']].head(25).round(2).to_string(index=False))
    figure_a(df, by_type, res_dir)
    return df


def figure_a(df: pd.DataFrame, by_type: pd.DataFrame, res_dir: Path) -> None:
    plt = H._plt()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    for ax, cls in zip(axes.flat[:4], SHUFFLES):
        x = df[f'{cls}_null_mean']; y = df.observed
        ax.scatter(x, y, s=8, color='0.6', alpha=0.6, label='other pairs')
        ax.scatter(x[df.over_shared], y[df.over_shared], s=14, color='#c0392b', label='over-shared (every class)')
        ax.scatter(x[df.under_shared], y[df.under_shared], s=14, color='#2874a6', label='under-shared (every class)')
        lim = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lim, lim, color='k', lw=0.8, ls='--')
        ax.set_xlabel(f'KCs sharing the pair, null mean ({CLASS_LABEL[cls]})'); ax.set_ylabel('KCs sharing the pair, real wiring')
        ax.set_title(CLASS_LABEL[cls], fontsize=10)
        if cls == SHUFFLES[0]:
            ax.legend(fontsize=8, frameon=False, loc='upper left')
    # z heatmap against the reference null, glomeruli ordered by their number of KCs
    gl = pd.concat([df[['glom_a', 'n_kcs_a']].rename(columns={'glom_a': 'g', 'n_kcs_a': 'n'}),
        df[['glom_b', 'n_kcs_b']].rename(columns={'glom_b': 'g', 'n_kcs_b': 'n'})]).drop_duplicates('g').set_index('g').n
    order = gl.sort_values(ascending=False).index
    zmat = pd.DataFrame(np.nan, index=order, columns=order)
    for r in df.itertuples():
        z = getattr(r, f'{NULL_CLASS}_z')
        zmat.loc[r.glom_a, r.glom_b] = zmat.loc[r.glom_b, r.glom_a] = z
    ax = axes.flat[4]
    im = ax.imshow(np.clip(zmat.to_numpy(dtype=float), -4, 4), cmap='RdBu_r', vmin=-4, vmax=4)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5)
    ax.set_title(f'z of the shared-KC count against the {CLASS_LABEL[NULL_CLASS]} null\n(glomeruli ordered by number of KCs)', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='z (clipped at ±4)')
    ax = axes.flat[5]
    top = df[df.over_shared].head(20)
    if len(top):
        labels = [f'{a} / {b}' for a, b in zip(top.glom_a, top.glom_b)]
        ax.barh(range(len(top))[::-1], top[f'{NULL_CLASS}_excess'], color='#c0392b', alpha=0.8)
        ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(labels, fontsize=7)
        for i, (e, z) in enumerate(zip(top[f'{NULL_CLASS}_excess'], top.min_z)):
            ax.text(e, len(top) - 1 - i, f' z≥{z:.1f}', va='center', fontsize=6)
        ax.set_xlabel(f'excess KCs sharing the pair over the {CLASS_LABEL[NULL_CLASS]} null')
        ax.set_title('over-shared pairs (largest smallest-z first)', fontsize=9)
    fig.suptitle('Which glomerulus pairs the hemibrain claw wiring puts on common KCs beyond each shuffle\'s margins', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(res_dir / f'{EXPERIMENT}_coinput_excess.png', dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# part B
def matrices_from_pairs(df: pd.DataFrame) -> Tuple[pd.Index, Dict[str, pd.DataFrame]]:
    gloms = pd.Index(sorted(set(df.glom_a) | set(df.glom_b)))
    idx = {g: i for i, g in enumerate(gloms)}
    out: Dict[str, pd.DataFrame] = {}
    cols = {'obs': 'observed'}
    for cls in SHUFFLES:
        cols[f'null_{cls}'] = f'{cls}_null_mean'; cols[f'excess_{cls}'] = f'{cls}_excess'; cols[f'z_{cls}'] = f'{cls}_z'
    for key, col in cols.items():
        m = np.zeros((len(gloms), len(gloms)))
        v = df[col].to_numpy(dtype=float)
        v = np.where(np.isfinite(v), v, 0.0)
        ia = df.glom_a.map(idx).to_numpy(); ib = df.glom_b.map(idx).to_numpy()
        m[ia, ib] = v; m[ib, ia] = v
        out[key] = pd.DataFrame(m, index=gloms, columns=gloms)
    # what the KC-type structure adds to the margins: the within-type null's mean minus the
    # type-ignoring null's (the part of the excess over a type-ignoring class that its
    # within-type twin explains)
    for cls in PLAIN_CLASSES:
        out[f'typestruct_{cls}'] = out[f'null_{cls}_within_type'] - out[f'null_{cls}']
    return gloms, out


def odor_weights(gloms: pd.Index) -> Dict[str, pd.DataFrame]:
    """ORN input per glomerulus (rows, the wiring's glomeruli; 0 where no ORN data) x odor:
    clipped at 0 and normalized per odor ('norm'), or clipped only ('raw')."""
    orn = H.abbrev_columns(H.apl16_orn_deltas()).astype(float)
    a = orn.clip(lower=0.0).reindex(gloms).fillna(0.0)
    missing = [g for g in orn.index if g not in gloms]
    if missing:
        print(f'ORN glomeruli not in the wiring (ignored): {missing}', flush=True)
    return {'norm': a / a.sum(axis=0), 'raw': a}


def pair_series(sq: pd.DataFrame) -> pd.Series:
    """Odor x odor square -> pair Series over PAIRS16 (canonical order)."""
    return pd.Series({p: float(sq.loc[p[0], p[1]]) for p in H.PAIRS16},
        index=pd.MultiIndex.from_tuples(H.PAIRS16, names=['odor_a', 'odor_b']))


def perm_p(score: pd.Series, target: pd.Series, n_perm: int = 5000, seed: int = 0) -> Tuple[float, float, int]:
    """Spearman of two pair Series and a two-sided p from permuting the odor labels of one."""
    s, t = H.align_pairs(score, target)
    odors = H.odor_order(set(k for p in s.index for k in p))
    n = len(odors); io = {o: i for i, o in enumerate(odors)}
    S = np.zeros((n, n)); T = np.zeros((n, n))
    for (a, b), v in s.items():
        S[io[a], io[b]] = S[io[b], io[a]] = v
    for (a, b), v in t.items():
        T[io[a], io[b]] = T[io[b], io[a]] = v
    iu = np.triu_indices(n, k=1)
    tr = stats.rankdata(T[iu])
    def rho(M):
        return float(stats.pearsonr(stats.rankdata(M[iu]), tr)[0])
    obs = rho(S)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        if abs(rho(S[perm][:, perm])) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1), len(s)


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    """Spearman of x and y with z partialled out (on the three pairwise Spearman rhos)."""
    rxy, rxz, ryz = H.spearman(x, y), H.spearman(x, z), H.spearman(y, z)
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def model_contributions(ref: pd.DataFrame) -> Dict[str, pd.Series]:
    """Per model and condition, the real wiring's residual around the linear ORN prediction
    minus the mean of E47's twenty shuffle draws' (the retuned part A variants), on the 66
    pairs; and the real wiring's residual itself."""
    path = H.shared_dir() / 'results' / E47 / 'model_pair_distances_long.csv'
    if not path.exists():
        print(f'no {path}: model contributions skipped', flush=True)
        return {}
    long = pd.read_csv(path).drop_duplicates(['tag', 'condition', 'odor_a', 'odor_b'], keep='last')
    out: Dict[str, pd.Series] = {}
    for mkey in MODELS:
        for cond, cname in (('on', 'on'), ('acute_off', 'off')):
            def resid_of(tag):
                sub = long[(long.tag == tag) & (long.condition == cond)]
                if sub.empty:
                    return None
                d = H.canonical_pairs(pd.Series(sub.distance.to_numpy(), index=list(zip(sub.odor_a, sub.odor_b))))
                d = d.loc[d.index.intersection(ref.index)]
                r, _ = H.linear_resid(d, ref['orn'].loc[d.index])
                return r
            real = resid_of(f'A_{mkey}_real_retuned')
            if real is None:
                continue
            shuf = [resid_of(f'A_{mkey}_{s}-seed{seed}_retuned') for s in SHUFFLES for seed in range(1, 6)]
            shuf = [r for r in shuf if r is not None]
            out[f'model_resid_{cname}_{mkey}'] = real
            if shuf:
                out[f'model_contrib_{cname}_{mkey}'] = real - pd.concat(shuf, axis=1).mean(axis=1).reindex(real.index)
    return out


def part_b(res_dir: Path) -> None:
    df = pd.read_csv(res_dir / 'coinput_pairs.csv')
    gloms, mats = matrices_from_pairs(df)
    weights = odor_weights(gloms)
    scores: Dict[str, pd.Series] = {}
    for wname, a in weights.items():
        A = a.to_numpy()
        for key, M in mats.items():
            sq = pd.DataFrame(A.T @ M.to_numpy() @ A, index=a.columns, columns=a.columns)
            scores[f'{wname}_{key}'] = pair_series(sq)
    ref = H.reference_pairs()
    targets: Dict[str, pd.Series] = {'vcf_hyp': ref['vcf_hyp'], 'chem': ref['chem'], 'orn': ref['orn']}
    for col, name in (('kc_ctrl', 'resid_ctrl'), ('kc_tnt', 'resid_tnt'), ('kc_soma', 'resid_wt')):
        targets[name], _ = H.linear_resid(ref[col], ref['orn'])
    targets.update(model_contributions(ref))
    table = pd.DataFrame(scores)
    table['in_megamat66'] = [p in set(H.MEGAMAT_PAIRS66) for p in table.index]
    for name, t in targets.items():
        table[name] = t.reindex(table.index)
    table.reset_index().to_csv(res_dir / 'odor_pair_scores.csv', index=False)
    rows = []
    for skey, s in scores.items():
        wname, kind = skey.split('_', 1)
        cls = next((c for c in SHUFFLES if kind.endswith(c)), '')
        kind0 = kind[:-(len(cls) + 1)] if cls else kind
        if kind0 == 'null' and cls in TYPE_CLASSES:
            pass
        for tname, t in targets.items():
            rho, p, n = perm_p(s, t)
            rows.append(dict(weighting=wname, kind=kind0, null_class=cls, target=tname, rho=rho, perm_p=p, n_pairs=n))
    corr = pd.DataFrame(rows)
    corr.to_csv(res_dir / 'score_correlations.csv', index=False)
    # the alignment with the score partialled out
    prow = []
    for geno, rname in (('control flies', 'resid_ctrl'), ('APL output blocked (TNT)', 'resid_tnt'), ("Remy's wild-type somata", 'resid_wt')):
        before = H.spearman(targets[rname], ref['vcf_hyp'])
        for wname in WEIGHTINGS:
            for key in ['obs'] + [f'excess_{c}' for c in SHUFFLES] + [f'null_{c}' for c in SHUFFLES]:
                s = scores[f'{wname}_{key}']
                prow.append(dict(genotype=geno, weighting=wname, score=key, alignment=before,
                    alignment_partial=partial_spearman(targets[rname], ref['vcf_hyp'], s),
                    rho_score_resid=H.spearman(s, targets[rname]), rho_score_vcf=H.spearman(s, ref['vcf_hyp'])))
    partial = pd.DataFrame(prow)
    partial.to_csv(res_dir / 'alignment_partial.csv', index=False)
    # the block pairs
    brow = []
    groups = H.ODOR_GROUPS
    for skey, s in scores.items():
        s66 = s.loc[[p for p in s.index if p in set(H.MEGAMAT_PAIRS66)]]
        kinds = {}
        for (a, b), v in s66.items():
            ga, gb = groups[a], groups[b]
            if {ga, gb} == {'fruity', 'small_ester'}:
                k = 'fruity x small-ester'
            elif ga == gb:
                k = 'within group'
            else:
                k = 'other between-group'
            kinds.setdefault(k, []).append(v)
        wname, kind = skey.split('_', 1)
        brow.append(dict(score=skey, weighting=wname, kind=kind, **{f'{k} mean': float(np.mean(v)) for k, v in kinds.items()},
            **{f'{k} n': len(v) for k, v in kinds.items()}))
    block = pd.DataFrame(brow)
    block.to_csv(res_dir / 'block_scores.csv', index=False)
    pd.set_option('display.width', 250)
    main = corr[(corr.weighting == 'norm') & (corr.kind.isin(['obs', 'excess', 'null', 'z'])) &
        (corr.null_class.isin(['', NULL_CLASS])) & (~corr.target.str.startswith('model_resid'))]
    print(main.pivot(index='target', columns=['kind', 'null_class'], values='rho').round(2).to_string())
    print('permutation p:')
    print(main.pivot(index='target', columns=['kind', 'null_class'], values='perm_p').round(3).to_string())
    print(partial[(partial.weighting == 'norm') & (partial.score == f'excess_{NULL_CLASS}')].round(3).to_string(index=False))
    print(block[(block.weighting == 'norm') & block.kind.isin(['obs', f'excess_{NULL_CLASS}', f'null_{NULL_CLASS}'])].round(4).to_string(index=False))
    figure_b(table, corr, res_dir)
    (res_dir / 'tables_ab.md').write_text(tables_ab(res_dir))


def figure_b(table: pd.DataFrame, corr: pd.DataFrame, res_dir: Path) -> None:
    plt = H._plt()
    t66 = table[table.in_megamat66]
    s = t66[f'norm_excess_{NULL_CLASS}']
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    groups = H.ODOR_GROUPS
    def is_block(p):
        return {groups[p[0]], groups[p[1]]} == {'fruity', 'small_ester'}
    blk = np.array([is_block(p) for p in t66.index])
    panels = [('vcf_hyp', 'natural-source distance (hyperbolic embedding)'), ('resid_tnt', 'KC distance residual around the ORN prediction, APL output blocked (TNT)'),
        ('resid_ctrl', 'KC distance residual around the ORN prediction, control flies')]
    panels += [(c, f'model APL-off residual, real wiring minus shuffle mean ({m})') for c, m in
        ((f'model_contrib_off_thr40_gain1.5', 'eight at 1.5, 40 Hz'), (f'model_contrib_off_thr70_gain1.25', 'eight at 1.25, 70 Hz')) if c in t66.columns]
    for ax, (col, label) in zip(axes.flat, panels):
        y = t66[col]
        ax.scatter(s[~blk], y[~blk], s=16, color='0.45')
        ax.scatter(s[blk], y[blk], s=22, color='#c0392b', label='fruity x small-ester pairs')
        for p, xv, yv in zip(t66.index, s, y):
            if abs(stats.zscore(s)[list(t66.index).index(p)]) > 1.3 or is_block(p):
                ax.annotate(f'{p[0]}/{p[1]}', (xv, yv), fontsize=6, xytext=(2, 2), textcoords='offset points')
        r = corr[(corr.weighting == 'norm') & (corr.kind == 'excess') & (corr.null_class == NULL_CLASS) & (corr.target == col)]
        if len(r):
            ax.set_title(f'ρ = {float(r.rho.iloc[0]):.2f} (odor-permutation p = {float(r.perm_p.iloc[0]):.3f})', fontsize=9)
        ax.set_xlabel(f'wiring-sharing score: excess shared KCs over the {CLASS_LABEL[NULL_CLASS]} null,\nweighted by the odors\' ORN input', fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
        if col == 'vcf_hyp':
            ax.legend(fontsize=8, frameon=False)
    ax = axes.flat[-1]
    sub = corr[(corr.weighting == 'norm') & (corr.target.isin(['vcf_hyp', 'chem', 'orn', 'resid_ctrl', 'resid_tnt', 'resid_wt']))]
    kinds = [('obs', ''), ('null', NULL_CLASS)] + [('excess', c) for c in SHUFFLES] + [('typestruct', c) for c in PLAIN_CLASSES]
    width = 0.8 / len(kinds)
    tnames = ['vcf_hyp', 'chem', 'orn', 'resid_ctrl', 'resid_tnt', 'resid_wt']
    for i, (kind, cls) in enumerate(kinds):
        r = sub[(sub.kind == kind) & (sub.null_class == cls)].set_index('target').reindex(tnames)
        if kind == 'obs':
            lab = 'observed counts'
        elif kind == 'null':
            lab = f'null mean ({CLASS_LABEL[cls]})'
        elif kind == 'typestruct':
            lab = f'KC-type structure (within-type minus plain null, {CLASS_LABEL[cls]})'
        else:
            lab = f'excess over {CLASS_LABEL[cls]}'
        ax.bar(np.arange(len(tnames)) + (i - len(kinds) / 2 + 0.5) * width, r.rho, width=width, label=lab)
    ax.set_xticks(range(len(tnames)))
    ax.set_xticklabels(['natural-source\ndistance', 'chemical\ndistance', 'ORN\ndistance', 'residual,\ncontrol flies', 'residual,\nAPL output\nblocked (TNT)', 'residual,\nwild-type\nsomata'], fontsize=7)
    ax.axhline(0, color='k', lw=0.6); ax.set_ylabel('Spearman ρ with the wiring-sharing score (66 pairs)', fontsize=8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))   # room for the legend above the bars
    ax.legend(fontsize=6, frameon=False, ncol=2, loc='upper left')
    fig.suptitle('From glomerulus pairs to odor pairs: how much the two odors\' glomeruli are over-wired onto common KCs, against what the pair does in the KC data', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(res_dir / f'{EXPERIMENT}_odor_pair_scores.png', dpi=130)
    plt.close(fig)


TARGET_LABEL = {'vcf_hyp': 'natural-source distance (hyperbolic embedding)', 'chem': 'chemical distance (ECFP)',
    'orn': 'ORN distance', 'resid_ctrl': 'KC residual around the ORN prediction, control flies',
    'resid_tnt': 'KC residual around the ORN prediction, APL output blocked (TNT)',
    'resid_wt': "KC residual around the ORN prediction, Remy's wild-type somata"}


def tables_ab(res_dir: Path) -> str:
    """Markdown tables of parts A and B for the write-up (from the saved CSVs)."""
    df = pd.read_csv(res_dir / 'coinput_pairs.csv')
    corr = pd.read_csv(res_dir / 'score_correlations.csv', keep_default_na=False)
    partial = pd.read_csv(res_dir / 'alignment_partial.csv')
    block = pd.read_csv(res_dir / 'block_scores.csv')
    out = []
    rows = ['**Over- and under-shared glomerulus pairs (part A): pairs of the 54 glomeruli whose number of KCs with claws from both '
        f'is z >= {Z_OVER:g} above (below) every shuffle class\'s null of 200 draws**', '',
        '| | all pairs | both glomeruli in the ORN input |', '|---|---:|---:|',
        f'| pairs | {len(df)} | {int(df.in_orn_input.sum())} |',
        f'| over-shared | {int(df.over_shared.sum())} | {int((df.over_shared & df.in_orn_input).sum())} |',
        f'| under-shared | {int(df.under_shared.sum())} | {int((df.under_shared & df.in_orn_input).sum())} |']
    for cls in SHUFFLES:
        rows.append(f'| z >= {Z_OVER:g} against the {CLASS_LABEL[cls]} alone | {int((df[f"{cls}_z"] >= Z_OVER).sum())} | '
            f'{int(((df[f"{cls}_z"] >= Z_OVER) & df.in_orn_input).sum())} |')
    out.append('\n'.join(rows))
    top = df[df.over_shared].sort_values('min_z', ascending=False)
    rows = ['**The over-shared pairs, strongest first (smallest z over the four classes); expected = the within-type Curveball null mean**', '',
        '| glomerulus pair | in the ORN input | KCs with claws from both | expected | excess | smallest z | z, claw permutation | z, within type | z, Curveball | z, within type |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in top.itertuples():
        rows.append(f'| {r.glom_a} / {r.glom_b} | {"yes" if r.in_orn_input else "no"} | {int(r.observed)} | '
            f'{getattr(r, f"{NULL_CLASS}_null_mean"):.1f} | {getattr(r, f"{NULL_CLASS}_excess"):+.1f} | {r.min_z:.1f} | '
            + ' | '.join(f'{getattr(r, f"{c}_z"):.1f}' for c in SHUFFLES) + ' |')
    out.append('\n'.join(rows))
    under = df[df.under_shared].sort_values('max_z').head(15)
    rows = ['**The fifteen most under-shared pairs (largest z over the four classes)**', '',
        '| glomerulus pair | in the ORN input | KCs with claws from both | expected | excess | largest z |', '|---|---|---:|---:|---:|---:|']
    for r in under.itertuples():
        rows.append(f'| {r.glom_a} / {r.glom_b} | {"yes" if r.in_orn_input else "no"} | {int(r.observed)} | '
            f'{getattr(r, f"{NULL_CLASS}_null_mean"):.1f} | {getattr(r, f"{NULL_CLASS}_excess"):+.1f} | {r.max_z:.1f} |')
    out.append('\n'.join(rows))
    tb_path = res_dir / 'glomerulus_type_bias.csv'
    if tb_path.exists():
        tb = pd.read_csv(tb_path)
        sel = tb[(tb.min_abs_z >= Z_OVER) & tb.in_orn_input].copy()
        sel['order'] = -sel.min_abs_z
        rows = ['**The KC-type structure (part A): glomeruli whose number of KCs of a type is |z| >= 2 away from both type-ignoring nulls '
            '(the within-type shuffles keep these numbers); glomeruli with ORN data, strongest first**', '',
            '| glomerulus | KC type | KCs of the type with a claw from it | fraction of the type | expected, claw permutation | z | expected, Curveball | z |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in sel.sort_values('order').itertuples():
            rows.append(f'| {r.glomerulus} | {r.kc_type} | {int(r.observed)} | {r.frac_of_type:.2f} | {r.claw_permutation_null_mean:.1f} | '
                f'{r.claw_permutation_z:+.1f} | {r.curveball_null_mean:.1f} | {r.curveball_z:+.1f} |')
        out.append('\n'.join(rows))
    cols = [('norm', 'obs', ''), ('norm', 'null', NULL_CLASS)] + [('norm', 'excess', c) for c in SHUFFLES] + \
        [('norm', 'typestruct', c) for c in PLAIN_CLASSES] + [('raw', 'excess', NULL_CLASS)]
    head = ['observed counts', f'margins only (null mean, {CLASS_LABEL[NULL_CLASS]})'] + [f'excess over the {CLASS_LABEL[c]}' for c in SHUFFLES] + \
        [f'KC-type structure ({CLASS_LABEL[c]}: within-type null minus plain null)' for c in PLAIN_CLASSES] + \
        [f'excess over the {CLASS_LABEL[NULL_CLASS]}, ORN input in Hz']
    rows = ['**The odor-pair wiring-sharing score against what the pair does elsewhere (part B): Spearman ρ over the 66 megamat pairs, '
        'with the two-sided p from 5000 permutations of the odor labels. The score sums the glomerulus pairs\' shared-KC counts '
        'weighted by the two odors\' ORN input (clipped at 0, normalized per odor unless "in Hz")**', '',
        '| the score is correlated with | ' + ' | '.join(head) + ' |', '|---|' + '---:|' * len(head)]
    targets = ['vcf_hyp', 'chem', 'orn', 'resid_ctrl', 'resid_tnt', 'resid_wt'] + sorted(c for c in corr.target.unique() if c.startswith('model_contrib_off')) + \
        sorted(c for c in corr.target.unique() if c.startswith('model_contrib_on'))
    mlabel = {'thr40_gain1.5': 'eight at 1.5, 40 Hz', 'thr70_gain1.5': 'eight at 1.5, 70 Hz', 'thr70_gain1.25': 'eight at 1.25, 70 Hz'}
    for tgt in targets:
        if tgt.startswith('model_contrib'):
            cond, m = tgt.split('_', 3)[2], tgt.split('_', 3)[3]
            label = f"the fitted model's wiring contribution, APL {cond} ({mlabel.get(m, m)})"
        else:
            label = TARGET_LABEL[tgt]
        cells = []
        for w, kind, cls in cols:
            r = corr[(corr.weighting == w) & (corr.kind == kind) & (corr.null_class == cls) & (corr.target == tgt)]
            cells.append(f'{float(r.rho.iloc[0]):+.2f} (p {float(r.perm_p.iloc[0]):.3f})' if len(r) == 1 else '-')
        rows.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    out.append('\n'.join(rows))
    rows = ['**The natural-source alignment (ρ of the KC residual around the ORN prediction with the natural-source distance) with the '
        f'excess score (over the {CLASS_LABEL[NULL_CLASS]}, normalized ORN input) partialled out**', '',
        '| KCs | alignment | with the score partialled out | ρ of the score with the residual | ρ of the score with the natural-source distance |',
        '|---|---:|---:|---:|---:|']
    for r in partial[(partial.weighting == 'norm') & (partial.score == f'excess_{NULL_CLASS}')].itertuples():
        rows.append(f'| {r.genotype} | {r.alignment:.3f} | {r.alignment_partial:.3f} | {r.rho_score_resid:+.2f} | {r.rho_score_vcf:+.2f} |')
    out.append('\n'.join(rows))
    rows = ['**The score on the fruity x small-ester pairs (normalized ORN input)**', '',
        '| score | fruity x small-ester (9 pairs) | other between-group (47) | within group (10) |', '|---|---:|---:|---:|']
    for kind, label in (('obs', 'observed shared-KC counts'), (f'null_{NULL_CLASS}', f'margins only (null mean, {CLASS_LABEL[NULL_CLASS]})'),
            (f'excess_{NULL_CLASS}', f'excess over the {CLASS_LABEL[NULL_CLASS]}')):
        r = block[(block.weighting == 'norm') & (block.kind == kind)]
        if len(r) == 1:
            r = r.iloc[0]
            rows.append(f'| {label} | {r["fruity x small-ester mean"]:.2f} | {r["other between-group mean"]:.2f} | {r["within group mean"]:.2f} |')
    out.append('\n'.join(rows))
    return '\n\n'.join(out) + '\n'


# ---------------------------------------------------------------------------
# part C
def lesion_plan(df: pd.DataFrame, res_dir: Path) -> pd.DataFrame:
    """The over-shared pairs with their targets (the reference null's mean, rounded) and, for
    each, a count-matched control pair that is neither over- nor under-shared (|z| < 1 against
    every class, same ORN-input status), lesioned by the same number of KCs."""
    over = df[df.over_shared].sort_values('min_z', ascending=False)
    cand = df[(df.min_z > -1) & (df.max_z < 1)].copy()
    used = set(); rows = []
    for r in over.itertuples():
        target = int(round(getattr(r, f'{NULL_CLASS}_null_mean')))
        excess = int(r.observed) - target
        rows.append(dict(kind='lesion_targeted', glom_a=r.glom_a, glom_b=r.glom_b, observed=int(r.observed), target=target,
            n_removed=excess, null_mean=getattr(r, f'{NULL_CLASS}_null_mean'), min_z=r.min_z, in_orn_input=bool(r.in_orn_input), matched_to=''))
        pool = cand[(cand.in_orn_input == r.in_orn_input) & ~cand.index.isin(used)]
        if pool.empty:
            pool = cand[~cand.index.isin(used)]
        j = (pool.observed - r.observed).abs().idxmin()
        used.add(j)
        c = cand.loc[j]
        rows.append(dict(kind='lesion_control', glom_a=c.glom_a, glom_b=c.glom_b, observed=int(c.observed),
            target=max(int(c.observed) - excess, 0), n_removed=min(excess, int(c.observed)), null_mean=c[f'{NULL_CLASS}_null_mean'],
            min_z=c.min_z, in_orn_input=bool(c.in_orn_input), matched_to=f'{r.glom_a}/{r.glom_b}'))
    plan = pd.DataFrame(rows)
    plan.to_csv(res_dir / 'lesion_pairs.csv', index=False)
    return plan


def make_lesions(W: pd.DataFrame, plan: pd.DataFrame, seeds: List[int], out_dir: Path, res_dir: Path) -> Dict[Tuple[str, int], pd.DataFrame]:
    from al_analysis.wpnkc_shuffles import lesion_coinput, coinput_counts, wPNKC_shuffle_summary
    obs = coinput_counts(W)
    out: Dict[Tuple[str, int], pd.DataFrame] = {}
    outcome_path = res_dir / 'lesion_outcome.csv'
    outcome = list(pd.read_csv(outcome_path).to_dict('records')) if outcome_path.exists() else []
    outcome = [r for r in outcome if (r['kind'], int(r['seed'])) not in {(k, s) for k in LESIONS for s in seeds}]
    over_pairs = [(a, b) for a, b in zip(plan[plan.kind == 'lesion_targeted'].glom_a, plan[plan.kind == 'lesion_targeted'].glom_b)]
    for kind in LESIONS:
        sub = plan[plan.kind == kind]
        targets = {(a, b): int(t) for a, b, t in zip(sub.glom_a, sub.glom_b, sub.target)}
        for seed in seeds:
            path = out_dir / f'wPNKC_{kind}-seed{seed}.parquet'
            if path.exists():
                L = pd.read_parquet(path)
                swaps = pd.read_csv(out_dir / f'swaps_{kind}-seed{seed}.csv')
            else:
                t0 = time.time()
                # the control protects the over-shared pairs, which can lock a few of its
                # KCs; it then stops short on that pair (lesion_outcome.csv has the counts)
                L, swaps = lesion_coinput(W, targets, seed=seed, within_kc_type=True,
                    protect=(over_pairs if kind == 'lesion_control' else None),
                    strict=(kind != 'lesion_control'))
                L.to_parquet(path); swaps.to_csv(out_dir / f'swaps_{kind}-seed{seed}.csv', index=False)
                print(f'{kind} seed {seed}: {len(swaps)} claw swaps in {time.time() - t0:.0f}s', flush=True)
            out[(kind, seed)] = L
            co = coinput_counts(L)
            s = wPNKC_shuffle_summary(W, L)
            reached = {f'{a}/{b}': int(co.loc[a, b]) for a, b in targets}
            over_after = {f'{a}/{b}': int(co.loc[a, b]) for a, b in over_pairs}
            outcome.append(dict(kind=kind, seed=seed, n_pairs=len(targets), n_swaps=len(swaps),
                frac_claws_changed=float(s['frac_claws_changed']), coinput_corr=float(s['coinput_corr']),
                coinput_resid_corr=float(s['coinput_resid_corr']), kc_n_claws_preserved=bool(s['kc_n_claws_preserved']),
                max_abs_glom_n_claws_change=float(s['max_abs_glom_n_claws_change']),
                sum_removed_listed=int(sum(int(obs.loc[a, b]) - int(co.loc[a, b]) for a, b in targets)),
                over_shared_total_before=int(sum(int(obs.loc[a, b]) for a, b in over_pairs)),
                over_shared_total_after=int(sum(over_after.values())),
                reached=json.dumps(reached)))
            pd.DataFrame(outcome).to_csv(outcome_path, index=False)
    return out


def part_c(args, res_dir: Path, out_dir: Path) -> None:
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    from E1_apl_knockout import bisect_threshold
    from E3_drive_dependent_gain import per_odor_extras, rate
    from E4_substrate import score, kc_by_glomerulus
    from E10_joint_fit import TAU, base_kws, bouton_base, breadth_weights, make_loss
    from E12_breadth_apl import density_excess
    from E16_claw_saturation import spearman_with_refs
    from E18_bouton_depletion import all_refs
    from E26_claw_power import sharing_deficit
    from E35_divisive_apl import suppression_stats
    from E44_operating_point import fraction_stats, losses
    df = pd.read_csv(res_dir / 'coinput_pairs.csv')
    plan = lesion_plan(df, res_dir)
    W0 = base_wiring()
    lesions = make_lesions(W0, plan, args.seeds, out_dir, res_dir)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs()
    data_mats = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    loss5 = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=0.05)
    loss_ctrl = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=CTRL_FRAC)
    model_prov = H.model_provenance(orn_deltas=orn)
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}
    refs = all_refs()
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame(
        {'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned_path = out_dir / 'tuned_params.json'
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    base_dir = out_dir / 'partC'; base_dir.mkdir(exist_ok=True)
    long_path = res_dir / 'model_pair_distances_long.csv'

    def finish(tag, meta, on_out, off_out, wPNKC):
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data_mats, ref)
        on_p, off_p = on[V], off[V]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        row.update(suppression_stats(H.response_fractions(on_out[0]), H.response_fractions(off_out[0]), data_rates))
        row.update(fraction_stats(on_out[0], off_out[0], data_rates))
        row.update(losses(loss5, loss_ctrl, on_out, off_out))
        rows.append(row)
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index)
        fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        pd.concat({(tag, 'on'): on_p, (tag, 'acute_off'): off_p}, names=['tag', 'condition', 'odor_a', 'odor_b']).rename(
            'distance').reset_index().to_csv(long_path, index=False, mode='a', header=not long_path.exists())
        print(f"{tag}: rates on {row['rate_on']:.4f} off {row['rate_off']:.4f}; rho ctrl {row[V + '__on_spearman']:.3f} "
            f"tnt {row[V + '__off_spearman']:.3f}; block {row[V + '__block_fruity_small_iso']:+.3f}; "
            f"struct wt {row[V + '__on_spearman_resid_vs_remap_wt']:+.2f}; vcf on {row[V + '__on_spearman_resid_vs_vcf_hyp']:+.2f} "
            f"off {row[V + '__off_spearman_resid_vs_vcf_hyp']:+.2f}; rate rho {row['spearman_rate_on_vs_data']:.2f}/"
            f"{row['spearman_rate_off_vs_data']:.2f}; loss @ctrl {row['loss_rate_ctrl']:.3f}; thr x{meta['thr_factor_vs_fit']:.3f}; "
            f"{meta['seconds']:.0f}s", flush=True)
        done.add(tag)

    for mkey in args.models:
        exp, ftag, thr, k, gain = MODELS[mkey]
        p_fit, fr = fitted_params(exp, ftag)
        for (kind, seed), L in lesions.items():
            wname = f'{kind}-seed{seed}'
            tag = f'C_{mkey}_{wname}_retuned'
            if tag in done:
                continue
            t0 = time.time()
            try:
                initial, wnorm, W, breadth = bouton_base(orn, base_dir, claw_thr=thr, gain=gain, wiring_matrix=L, wiring_tag=wname)
                sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p_fit, **base_kws(thr), _wPNKC=W,
                    _wAPLKC=breadth_weights(wnorm['wAPLKC'], breadth, k), _wKCAPL=wnorm['wKCAPL'],
                    _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
                t1 = time.time()
                extra = {kk: p_fit[kk] for kk in SCALE_KEYS}
                new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_fit['fixed_thr'], CTRL_FRAC, extra=extra)
                p_on = dict(p_fit, fixed_thr=float(new_thr))
                r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                for cond, (r_, s_) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
                    r_.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
                    s_.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
                tuned[tag] = dict(p_on, claw_thr=thr, k=k, gain=gain, ref_fit=ftag, ref_experiment=exp,
                    wiring=kind, wiring_seed=seed, thr_factor_vs_fit=float(thr_f))
                tuned_path.write_text(json.dumps(tuned, indent=1))
                meta = dict(variant='retuned', tag=tag, part='C', model=mkey, shuffle=kind, seed=seed, base='bouton',
                    target_rate=CTRL_FRAC, claw_thr=thr, k=k, kc_kc_inh=p_fit['wKCKC_scale'], gain=gain, ref_fit=ftag,
                    ref_experiment=exp, ref_fit_loss=float(fr['loss']), thr_factor_vs_fit=float(thr_f),
                    n_kcs=len(r_on), n_glomeruli=W.shape[1], rate_on=rate(r_on), rate_off=rate(r_off),
                    seconds=time.time() - t1, mean_claws_per_kc=float(kc_by_glomerulus(W).sum(axis='columns').mean()), **p_on)
                meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
                finish(tag, meta, (r_on, s_on), (r_off, s_off), W)
                del sim
            except Exception as e:  # keep going; record the failure
                print(f'{tag} FAILED after {time.time() - t0:.0f}s: {e!r}', flush=True)
                rows.append(dict(variant='retuned', tag=tag, part='C', model=mkey, shuffle=kind, seed=seed, error=repr(e)))
                pd.DataFrame(rows).to_csv(stats_path, index=False)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E52_glomerulus_pairs.py', part='C',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), models=args.models, seeds=args.seeds,
        null_class=NULL_CLASS, z_over=Z_OVER, lesions=list(LESIONS), model_sources={m: MODELS[m][:2] for m in args.models},
        ctrl_frac=CTRL_FRAC, kc_kc_tau=TAU, approx_conc_note=H.APPROX_CONC_NOTE, model=model_prov)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    summary(res_dir)


# ---------------------------------------------------------------------------
# summary (tables and figure for part C, beside E47's rows)
def summary(res_dir: Path) -> None:
    stats_path = res_dir / 'stats_variants.csv'
    if not stats_path.exists():
        print('no part C rows yet', flush=True)
        return
    mine = pd.read_csv(stats_path)
    if 'error' in mine.columns:
        mine = mine[~(mine.error.astype(str).str.len() > 3)]
    e47 = pd.read_csv(H.shared_dir() / 'results' / E47 / 'stats_variants.csv')
    if 'error' in e47.columns:
        e47 = e47[~(e47.error.astype(str).str.len() > 3)]
    e47 = e47[e47.part == 'A']
    comp = pd.concat([e47, mine], ignore_index=True)
    comp['vcf_on_minus_off'] = comp[f'{V}__on_spearman_resid_vs_vcf_hyp'] - comp[f'{V}__off_spearman_resid_vs_vcf_hyp']
    comp['weak_share_off'] = comp.weak_frac_off / comp.rate_off
    comp.to_csv(res_dir / 'comparison.csv', index=False)
    # per model and wiring group (the lesions and E47's shuffle classes): mean, sd and count over draws, for the summary
    cols = [c for c, *_ in STATS if c in comp.columns]
    grp = comp[comp.shuffle != 'real'].groupby(['model', 'shuffle'])[cols].agg(['mean', 'std', 'count'])
    grp.columns = [f'{c}__{s}' for c, s in grp.columns]
    grp.reset_index().to_csv(res_dir / 'per_class.csv', index=False)
    data = data_values()
    out = []
    groups = list(LESIONS) + list(SHUFFLES)
    glabel = {**LESION_LABEL, **CLASS_LABEL}
    for mkey, (exp, tag, thr, k, gain) in MODELS.items():
        sub = comp[comp.model == mkey]
        if sub[sub.shuffle.isin(LESIONS)].empty:
            continue
        real = sub[(sub.shuffle == 'real') & (sub.variant == 'retuned')]
        present = [g for g in groups if (sub.shuffle == g).any()]
        head1 = '| | | real wiring | ' + ' | '.join(f'{glabel[g]} (n = {int((sub.shuffle == g).sum())})' for g in present) + ' |'
        head2 = '| statistic | KC data | threshold re-tuned | ' + ' | '.join('mean ± sd' for _ in present) + ' |'
        sep = '|---|---|---|' + '---|' * len(present)
        rows = [f'**The eight glomeruli at {gain:g}, {thr:g} Hz per-claw threshold, fitted at 6.8 % on the real wiring; APL and KC-KC '
            f'scales fixed, somatic threshold re-tuned to 6.8 % per wiring**', '', head1, head2, sep]
        for col, label, fmt, dkey in STATS:
            if col not in sub.columns:
                continue
            cells = [format(float(real[col].iloc[0]), fmt) if len(real) == 1 and pd.notna(real[col].iloc[0]) else '-']
            for g in present:
                v = sub[sub.shuffle == g][col].astype(float)
                cells.append(f'{format(v.mean(), fmt)} ± {format(v.std(ddof=1), fmt.lstrip("+"))}' if v.notna().sum() > 1
                    else (format(float(v.iloc[0]), fmt) if v.notna().sum() == 1 else '-'))
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
    oc_path = res_dir / 'lesion_outcome.csv'
    if oc_path.exists():
        oc = pd.read_csv(oc_path)
        rows = ['**The lesions (per seed)**', '',
            '| lesion | seed | pairs | KCs removed from the listed pairs | claw swaps | claws moved (fraction) | co-input correlation with the real wiring | co-input residual correlation | over-shared pairs\' shared KCs, before -> after |',
            '|---|---|---|---|---|---|---|---|---|']
        for r in oc.itertuples():
            rows.append(f'| {LESION_LABEL[r.kind]} | {r.seed} | {r.n_pairs} | {r.sum_removed_listed} | {r.n_swaps} | {r.frac_claws_changed:.3f} | '
                f'{r.coinput_corr:.3f} | {r.coinput_resid_corr:.3f} | {r.over_shared_total_before} -> {r.over_shared_total_after} |')
        out.append('\n'.join(rows))
    (res_dir / 'comparison_table.md').write_text('\n\n'.join(out) + '\n')
    figure_c(comp, data, res_dir)
    print('\n\n'.join(out))


def figure_c(comp: pd.DataFrame, data: Dict[str, float], res_dir: Path) -> None:
    plt = H._plt()
    from matplotlib.lines import Line2D
    classes = ['real'] + [g for g in list(LESIONS) + list(SHUFFLES) if (comp.shuffle == g).any()]
    glabel = {'real': 'real wiring', **LESION_LABEL, **CLASS_LABEL}
    colors = {'thr70_gain1.5': '#1f5fbf', 'thr40_gain1.5': '#d1662a', 'thr70_gain1.25': '#2a9d5c'}
    offs = {'thr70_gain1.5': -0.22, 'thr40_gain1.5': 0.0, 'thr70_gain1.25': 0.22}
    panels = [s for s in STATS if s[0] in comp.columns][:10]
    fig, axes = plt.subplots(2, 5, figsize=(23, 9.5))
    for ax, (col, label, fmt, dkey) in zip(axes.flat, panels):
        for mkey in MODELS:
            sub = comp[comp.model == mkey]
            if sub.empty:
                continue
            c = colors[mkey]
            for i, s in enumerate(classes):
                g = sub[sub.shuffle == s]
                if s == 'real':
                    g = g[g.variant == 'retuned']
                v = g[col].astype(float).dropna()
                if v.empty:
                    continue
                x = i + offs[mkey]
                ax.scatter(np.full(len(v), x) + np.linspace(-0.04, 0.04, len(v)), v, s=14, color=c, alpha=0.6, zorder=3)
                if len(v) > 1:
                    ax.errorbar([x], [v.mean()], yerr=[v.std(ddof=1)], fmt='_', color=c, ms=14, mew=2, capsize=3, zorder=4)
                else:
                    ax.scatter([x], v, marker='_', s=180, color=c, linewidths=2, zorder=4)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='k', ls='--', lw=1)
        if len(classes) > 3:
            ax.axvline(2.5, color='0.7', lw=0.8)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([glabel[s] for s in classes], rotation=30, ha='right', fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
    handles = [Line2D([], [], marker='o', ls='', color=colors[m], label={'thr70_gain1.5': 'eight at 1.5, 70 Hz',
        'thr40_gain1.5': 'eight at 1.5, 40 Hz', 'thr70_gain1.25': 'eight at 1.25, 70 Hz'}[m]) for m in MODELS]
    handles.append(Line2D([], [], ls='--', color='k', label='KC data (target or ceiling)'))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle('The reference models at their fitted scales (threshold re-tuned to 6.8 %) on the real wiring, with the over-shared '
        'glomerulus pairs\' excess removed, with the same excess removed from matched control pairs, and on the four shuffles '
        '(points: seeds; bar: mean ± sd)', fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(res_dir / f'{EXPERIMENT}_lesions.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=['A', 'B', 'C'], default=None)
    ap.add_argument('--draws', type=int, default=200, help='shuffle draws per class (part A)')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2], help='lesion draws (part C)')
    ap.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    ap.add_argument('--summary', action='store_true', help='only (re)write part C\'s comparison tables and figure')
    ap.add_argument('--figures', action='store_true', help='with --part B: only redraw its figure from the saved tables')
    ap.add_argument('--out', type=Path, default=None, help='results folder (default: the shared results folder)')
    args = ap.parse_args()
    res_dir = args.out if args.out else H.results_dir(EXPERIMENT)
    res_dir.mkdir(parents=True, exist_ok=True)
    out_dir = H.outputs_dir(EXPERIMENT)
    if args.summary:
        if (res_dir / 'score_correlations.csv').exists():
            (res_dir / 'tables_ab.md').write_text(tables_ab(res_dir))
        summary(res_dir)
        return
    if args.part is None:
        ap.error('--part A, B or C (or --summary)')
    t0 = time.time()
    if args.part == 'A':
        part_a(res_dir, args.draws)
        prov_key = 'part_a'
    elif args.part == 'B':
        if args.figures:
            table = pd.read_csv(res_dir / 'odor_pair_scores.csv').set_index(['odor_a', 'odor_b'])
            figure_b(table, pd.read_csv(res_dir / 'score_correlations.csv', keep_default_na=False), res_dir)
            print('part B figure redrawn ->', res_dir)
            return
        part_b(res_dir)
        prov_key = 'part_b'
    else:
        part_c(args, res_dir, out_dir)
        return
    prov_path = res_dir / 'provenance.json'
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov[prov_key] = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E52_glomerulus_pairs.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), draws=args.draws, null_class=NULL_CLASS,
        z_over=Z_OVER, seconds=time.time() - t0, wiring='data/model_outputs/inputs/wPNKC_bouton_base.parquet',
        model=H.model_provenance(orn_deltas=H.apl16_orn_deltas()))
    prov_path.write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print(f'part {args.part} done in {time.time() - t0:.0f}s ->', res_dir)


if __name__ == '__main__':
    main()
