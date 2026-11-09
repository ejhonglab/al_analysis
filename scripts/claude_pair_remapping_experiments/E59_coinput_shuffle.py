#!/usr/bin/env python3
"""E59: the co-input-preserving shuffle, whether the real wiring's full pairwise matrix of shared-KC
counts suffices for the APL-off natural-source alignment; re-fits on convergence-bearing wirings.
The convergence transplant (E57) found the 32 over-shared pairs' shared-KC counts, put into within-type
Curveball draws, to restore the APL-on side of the alignment and none of the APL-off side, although
their removal from the real wiring (E54) loses part of it. What carries the rest is either the remainder
of the pairwise matrix (the under-shared pairs, the weakly over-shared tail) or the higher-order
arrangement (which KCs hold the pairs, what else they sample). Here a draw's whole 54 x 54 matrix of
shared-KC counts is matched to the real wiring's by annealed claw swaps
(`al_analysis.wpnkc_shuffles.match_coinput`), re-assigning which KCs hold the pairs:
(A) E47's within-type Curveball draws (seeds 1-3), matched within KC type with the type layer (per-type
    KCs per glomerulus) held: the real wiring's pairwise structure on re-assigned KCs;
(B) E47's plain Curveball draws (seeds 1-3), matched across types with the type layer left as drawn:
    the pairwise structure without the type layer.
The three reference models of E47's part A (the eight glomeruli at 1.5 at 40 and 70 Hz, the eight at
1.25 at 70 Hz) at their fitted scales with the somatic threshold re-tuned to 6.8 % on each wiring, scored
as E47 / E57; --summary puts them beside E47's untreated draws, E57's transplanted draws and the real
wiring, with paired (same-seed) differences, the matchings' table (residuals, co-input and three-way
co-input correlations with the real wiring, claws changed) and the re-fits' table.
(C) Re-fits, run separately as E54's part D (structure loss, k 2, APL->bouton capped at 3, 150
    evaluations, exact start from E44's optimum):
    `E10_joint_fit.py --experiment E59_coinput_shuffle --loss structure -k 2 --aplpn-max 3
     --rate-target ctrl_frac --maxfev 150 --claw-thr 40 --glom-gain 1.5 --init-from
     thr40_gain1.5_rate0.068_k2_cap3 --init-from-experiment E44_operating_point --init-from-exact
     --tag _cap3 --wiring-file <PARQUET> --wiring-tag <TAG>` with
    outputs/E57_convergence_transplant/wPNKC_curveball_within_type-seed1_transplant.parquet
    (curveball_within_type_transplant-seed1) and
    outputs/E59_coinput_shuffle/wPNKC_curveball_within_type-seed1_matched.parquet
    (curveball_within_type_matched-seed1); their fits.csv rows go into the re-fit table.
Outputs: <shared>/results/E59_coinput_shuffle/ (matching_outcome.csv, stats_variants.csv,
per_odor_response_fractions.csv, model_pair_distances_long.csv, comparison.csv, per_class.csv,
paired_differences.csv, comparison_table.md, E59_coinput_shuffle.png, provenance.json; fits.csv from the
re-fits) and ./outputs/E59_coinput_shuffle/ (the matched wirings, per-variant responses / spike counts).
Usage: OMP_NUM_THREADS=2 ./venv/bin/python E59_coinput_shuffle.py --plan      (the matchings only, 6 processes)
       OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E59_coinput_shuffle.py [--seeds 1 2 3]
       ./venv/bin/python E59_coinput_shuffle.py --summary
"""
from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E47_wiring_features import MODELS, STATS, V, CLASS_LABEL, data_values
from E52_glomerulus_pairs import base_wiring
from E54_gain_and_lesion import run_variants, MODEL_LABEL, EXPERIMENT as E54
from E57_convergence_transplant import PAIRED_COLS, E47, EXPERIMENT as E57

EXPERIMENT = 'E59_coinput_shuffle'
CLASSES = ('curveball_within_type', 'curveball')
KINDS = ('transplant', 'matched')  # transplant rows come from E57
GROUP_LABEL = {'real': 'real wiring',
    'curveball_within_type': 'Curveball within type, untreated (E47)', 'curveball': 'Curveball, untreated (E47)',
    'curveball_within_type_transplant': 'Curveball within type + the 32 pairs\' convergence (E57)',
    'curveball_transplant': 'Curveball + the 32 pairs\' convergence (E57)',
    'curveball_within_type_matched': 'Curveball within type, all 1431 pairs matched',
    'curveball_matched': 'Curveball, all 1431 pairs matched'}
KIND_LABEL = {'transplant': '+ the 32 pairs\' convergence (E57), minus untreated', 'matched': 'all pairs matched, minus untreated'}
# the annealing: ~6 s per 150k steps on hal; 3e6 steps take a draw from an objective of ~5e4 to a few tens
MATCH_KWS = dict(n_steps=3_000_000, t_start=5.0, t_end=0.05, type_layer_weight=4.0)
REFIT_TAG = 'thr40_gain1.5_rate0.068_k2_cap3'


def triplet_counts(wPNKC: pd.DataFrame) -> np.ndarray:
    """(glomeruli x glomeruli x glomeruli) number of KCs with claws from all three glomeruli."""
    from al_analysis.wpnkc_shuffles import _kc_counts, _column_groups
    counts = _kc_counts(wPNKC)
    codes, groups = _column_groups(wPNKC.columns, 'glomerulus')
    onehot = np.zeros((len(codes), len(groups)))
    onehot[np.arange(len(codes)), codes] = 1
    B = ((counts @ onehot) > 0).astype(np.float64)
    return np.einsum('ki,kj,kl->ijl', B, B, B)


def triplet_corr(T_real: np.ndarray, T_other: np.ndarray) -> float:
    """Pearson correlation over distinct glomerulus triples of the three-way shared-KC counts."""
    n = T_real.shape[0]
    idx = [(i, j, l) for i in range(n) for j in range(i + 1, n) for l in range(j + 1, n)]
    a = np.array([T_real[i, j, l] for i, j, l in idx]); b = np.array([T_other[i, j, l] for i, j, l in idx])
    return float(np.corrcoef(a, b)[0, 1])


def type_layer(wPNKC: pd.DataFrame) -> pd.DataFrame:
    """(KC type x glomerulus) number of KCs of the type with any claw from the glomerulus."""
    from al_analysis.mb_model import KC_ID, KC_TYPE
    from al_analysis.wpnkc_shuffles import coinput_counts
    kc = wPNKC.T.groupby(level='glomerulus').sum().T  # claw x glomerulus
    kc = kc.groupby(level=[KC_ID, KC_TYPE]).sum() > 0
    return kc.groupby(level=KC_TYPE).sum()


def _match_one(args: Tuple[str, int, str]) -> dict:
    """One draw's matching (a worker): shuffles, matches, saves the wiring and returns its outcome row."""
    cls, seed, out_dir = args
    from al_analysis.wpnkc_shuffles import NAMED_SHUFFLES, shuffle_wPNKC, match_coinput, wPNKC_shuffle_summary
    W = base_wiring()
    method, within = NAMED_SHUFFLES[cls]
    S = shuffle_wPNKC(W, method, within_kc_type=within, seed=seed)
    within_type = cls == 'curveball_within_type'
    t0 = time.time()
    L, stats = match_coinput(S, W, seed=100 + seed, within_kc_type=within_type, match_type_layer=within_type, **MATCH_KWS)
    L.to_parquet(Path(out_dir) / f'wPNKC_{cls}-seed{seed}_matched.parquet')
    s_draw = wPNKC_shuffle_summary(W, S); s_m = wPNKC_shuffle_summary(W, L); s_vs_draw = wPNKC_shuffle_summary(S, L)
    T_real = triplet_counts(W)
    tl_real = type_layer(W); tl_draw = type_layer(S).reindex_like(tl_real).fillna(0); tl_m = type_layer(L).reindex_like(tl_real).fillna(0)
    row = dict(shuffle=cls, seed=seed, within_kc_type=within_type, match_type_layer=within_type, **MATCH_KWS,
        **{k: (stats[k].item() if hasattr(stats[k], 'item') else stats[k]) for k in stats.index},
        draw_coinput_corr_vs_real=float(s_draw['coinput_corr']), matched_coinput_corr_vs_real=float(s_m['coinput_corr']),
        draw_coinput_resid_corr_vs_real=float(s_draw['coinput_resid_corr']), matched_coinput_resid_corr_vs_real=float(s_m['coinput_resid_corr']),
        draw_triplet_corr_vs_real=triplet_corr(T_real, triplet_counts(S)), matched_triplet_corr_vs_real=triplet_corr(T_real, triplet_counts(L)),
        draw_type_layer_abs_diff_max=int(np.abs(tl_draw.to_numpy() - tl_real.to_numpy()).max()),
        matched_type_layer_abs_diff_max=int(np.abs(tl_m.to_numpy() - tl_real.to_numpy()).max()),
        draw_frac_claws_changed_vs_real=float(s_draw['frac_claws_changed']), matched_frac_claws_changed_vs_real=float(s_m['frac_claws_changed']),
        frac_claws_changed_vs_draw=float(s_vs_draw['frac_claws_changed']), kc_n_claws_preserved=bool(s_m['kc_n_claws_preserved']),
        max_abs_glom_n_claws_change=int(s_m['max_abs_glom_n_claws_change']), max_abs_glom_n_kcs_change=int(s_m['max_abs_glom_n_kcs_change']),
        wall_seconds=float(time.time() - t0))
    return row


def make_wirings(seeds: List[int], out_dir: Path, res_dir: Path, workers: int = 6) -> Dict[Tuple[str, int], pd.DataFrame]:
    """The matched wirings (cached as parquet) and matching_outcome.csv (one row per draw)."""
    oc_path = res_dir / 'matching_outcome.csv'
    outcome = list(pd.read_csv(oc_path).to_dict('records')) if oc_path.exists() else []
    have = {(r['shuffle'], int(r['seed'])) for r in outcome}
    todo = [(cls, seed, str(out_dir)) for cls in CLASSES for seed in seeds
        if not ((cls, seed) in have and (out_dir / f'wPNKC_{cls}-seed{seed}_matched.parquet').exists())]
    if todo:
        print(f'matching {len(todo)} draws in {min(workers, len(todo))} processes', flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            for row in pool.map(_match_one, todo):
                outcome = [r for r in outcome if not (r['shuffle'] == row['shuffle'] and int(r['seed']) == row['seed'])] + [row]
                pd.DataFrame(outcome).to_csv(oc_path, index=False)
                print(f"{row['shuffle']} seed {row['seed']}: objective {row['objective_start']:.0f} -> {row['objective_end']:.0f}, "
                    f"pair |diff| max {row['pair_abs_diff_max']}, co-input corr {row['draw_coinput_corr_vs_real']:.3f} -> "
                    f"{row['matched_coinput_corr_vs_real']:.3f}, triplet corr {row['draw_triplet_corr_vs_real']:.3f} -> "
                    f"{row['matched_triplet_corr_vs_real']:.3f}, claws changed vs real {row['matched_frac_claws_changed_vs_real']:.3f}, "
                    f"{row['wall_seconds']:.0f}s", flush=True)
    return {(cls, seed): pd.read_parquet(out_dir / f'wPNKC_{cls}-seed{seed}_matched.parquet') for cls in CLASSES for seed in seeds}


def jobs_for(wirings: Dict[Tuple[str, int], pd.DataFrame]) -> List[dict]:
    jobs = []
    for mkey, spec in MODELS.items():
        for (cls, seed), L in wirings.items():
            group = f'{cls}_matched'
            jobs.append(dict(tag=f'M_{mkey}_{group}-seed{seed}_retuned', part='M', model=mkey, spec=spec, gain=spec[4], wiring=L,
                wiring_tag=f'{group}-seed{seed}', shuffle=group, seed=seed))
    return jobs


def _fit_row(r: dict, model: str, shuffle: str, part: str) -> dict:
    on = r.get('spike_counts_cosine__on_spearman_resid_vs_vcf_hyp', np.nan)
    return {'model': model, 'shuffle': shuffle, 'variant': 'fit', 'part': part, 'seed': r.get('wiring_seed', np.nan),
        f'{V}__block_fruity_small_iso': r['fit_block'], 'loss_rate_ctrl': r['loss'],
        f'{V}__on_spearman': r['fit_on_spearman'], f'{V}__off_spearman': r['fit_off_spearman'],
        'spearman_rate_on_vs_data': r['fit_po_on_spearman'], 'spearman_rate_off_vs_data': r['fit_po_off_spearman'],
        f'{V}__on_spearman_resid_vs_remap_wt': r['fit_wt_dev_on'], f'{V}__on_spearman_resid_vs_vcf_hyp': on,
        f'{V}__off_spearman_resid_vs_vcf_hyp': r['fit_vcf_off'], 'vcf_on_minus_off': on - r['fit_vcf_off'],
        'fixed_thr': r.get('fixed_thr'), 'kc_kc_inh': r.get('kc_kc_inh'), 'wAPLKC_scale': r.get('wAPLKC_scale'),
        'wAPLPN_scale': r.get('wAPLPN_scale'), 'n_evals': r.get('n_evals')}


def paired_differences(comp: pd.DataFrame) -> pd.DataFrame:
    """Each treated draw (E57's transplant, this experiment's match) against E47's untreated draw of the same
    seed, per statistic in PAIRED_COLS: the effect per draw and its mean and sd over draws (long format)."""
    rows = []
    for mkey in MODELS:
        sub = comp[comp.model == mkey]
        for cls in CLASSES:
            base = sub[sub.shuffle == cls].set_index('seed')
            for kind in KINDS:
                t = sub[sub.shuffle == f'{cls}_{kind}'].set_index('seed')
                seeds = sorted(set(t.index) & set(base.index))
                if not seeds:
                    continue
                for col in PAIRED_COLS:
                    if col not in sub.columns:
                        continue
                    tv = t.loc[seeds, col].astype(float).values; bv = base.loc[seeds, col].astype(float).values
                    d = tv - bv
                    rows.append(dict(model=mkey, shuffle=cls, kind=kind, stat=col, n=len(seeds), mean_diff=float(d.mean()),
                        sd_diff=float(d.std(ddof=1)) if len(seeds) > 1 else float('nan'), untreated_mean=float(bv.mean()),
                        treated_mean=float(tv.mean()), per_seed=json.dumps({int(sd): round(float(x), 4) for sd, x in zip(seeds, d)})))
    return pd.DataFrame(rows)


def summary(res_dir: Path) -> None:
    stats_path = res_dir / 'stats_variants.csv'
    mine = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    if 'error' in mine.columns:
        mine = mine[~(mine.error.astype(str).str.len() > 3)]
    e47 = pd.read_csv(H.shared_dir() / 'results' / E47 / 'stats_variants.csv')
    if 'error' in e47.columns:
        e47 = e47[~(e47.error.astype(str).str.len() > 3)]
    e47 = e47[(e47.part == 'A') & (e47.shuffle.isin(['real'] + list(CLASSES)))]
    e47 = e47[(e47.shuffle != 'real') | (e47.variant == 'retuned')]
    e57 = pd.read_csv(H.shared_dir() / 'results' / E57 / 'stats_variants.csv')
    if 'error' in e57.columns:
        e57 = e57[~(e57.error.astype(str).str.len() > 3)]
    e57 = e57[e57.shuffle.isin([f'{c}_transplant' for c in CLASSES])]
    comp = pd.concat([e47, e57, mine], ignore_index=True)
    comp['vcf_on_minus_off'] = comp[f'{V}__on_spearman_resid_vs_vcf_hyp'] - comp[f'{V}__off_spearman_resid_vs_vcf_hyp']
    comp['weak_share_off'] = comp.weak_frac_off / comp.rate_off
    # the fits: the real wiring's (E44), E54's re-fit on the lesioned draw, and this experiment's re-fits
    fitrows = []
    f44 = pd.read_csv(H.shared_dir() / 'results' / 'E44_operating_point' / 'fits.csv')
    r = f44[f44.tag == REFIT_TAG]
    if len(r) == 1:
        fitrows.append(_fit_row(r.iloc[0].to_dict(), 'thr40_gain1.5', 'fit_real', 'fit'))
    f54 = H.shared_dir() / 'results' / E54 / 'fits.csv'
    if f54.exists():
        for rr in pd.read_csv(f54).to_dict('records'):
            fitrows.append(_fit_row(rr, f"thr{rr['claw_thr']:g}_gain{rr['gain']:g}", 'refit_' + str(rr.get('wiring_tag')), 'E54'))
    fits_path = res_dir / 'fits.csv'
    if fits_path.exists():
        for rr in pd.read_csv(fits_path).to_dict('records'):
            fitrows.append(_fit_row(rr, f"thr{rr['claw_thr']:g}_gain{rr['gain']:g}", 'refit_' + str(rr.get('wiring_tag')), 'C'))
    if fitrows:
        comp = pd.concat([comp, pd.DataFrame(fitrows)], ignore_index=True)
    comp.to_csv(res_dir / 'comparison.csv', index=False)
    cols = [c for c, *_ in STATS if c in comp.columns]
    draws = comp[(comp.shuffle != 'real') & (comp.variant != 'fit')]
    grp = draws.groupby(['model', 'shuffle'])[cols].agg(['mean', 'std', 'count'])
    grp.columns = [f'{c}__{s}' for c, s in grp.columns]
    grp.reset_index().to_csv(res_dir / 'per_class.csv', index=False)
    paired = paired_differences(comp[comp.variant != 'fit'])
    paired.to_csv(res_dir / 'paired_differences.csv', index=False)
    data = data_values()
    out = []
    groups = ['curveball_within_type', 'curveball_within_type_transplant', 'curveball_within_type_matched',
        'curveball', 'curveball_transplant', 'curveball_matched']
    for mkey in MODELS:
        sub = comp[(comp.model == mkey) & (comp.variant != 'fit')]
        if sub[sub.shuffle.str.endswith('matched')].empty:
            continue
        present = [g for g in groups if (sub.shuffle == g).any()]
        real = sub[sub.shuffle == 'real']
        rows = [f'**The {MODEL_LABEL[mkey]} at its fitted scales (threshold re-tuned to 6.8 %) on the real wiring, on untreated Curveball '
            'draws (E47, five draws), on draws with the 32 over-shared pairs\' convergence transplanted (E57, three draws) and on draws with '
            'all 1431 pairs\' shared-KC counts matched to the real wiring\'s (three draws)**', '',
            '| | | real wiring | ' + ' | '.join(f'{GROUP_LABEL[g]} (n = {int((sub.shuffle == g).sum())})' for g in present) + ' |',
            '| statistic | KC data | threshold re-tuned | ' + ' | '.join('mean ± sd' for _ in present) + ' |',
            '|---|---|---|' + '---|' * len(present)]
        for col, label, fmt, dkey in STATS:
            if col not in sub.columns:
                continue
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            cells = [format(float(real[col].iloc[0]), fmt) if len(real) == 1 and pd.notna(real[col].iloc[0]) else '-']
            for g in present:
                v = sub[sub.shuffle == g][col].astype(float).dropna()
                cells.append(f'{format(v.mean(), fmt)} ± {format(v.std(ddof=1), fmt.lstrip("+"))}' if len(v) > 1
                    else (format(float(v.iloc[0]), fmt) if len(v) == 1 else '-'))
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
    # paired tables
    for mkey in MODELS:
        p = paired[paired.model == mkey]
        if p.empty or p[p.kind == 'matched'].empty:
            continue
        real = comp[(comp.model == mkey) & (comp.shuffle == 'real') & (comp.variant != 'fit')]
        rows = [f'**{MODEL_LABEL[mkey]} at its fitted scales: each treated draw against the same seed\'s untreated draw (the treatment\'s '
            'effect, mean ± sd over the three draws)**', '',
            '| | | | Curveball within type {span=3} | Curveball {span=3} |',
            '| statistic | KC data | real wiring | untreated (seeds 1 to 3) | ' + KIND_LABEL['transplant'] + ' | ' + KIND_LABEL['matched'] +
            ' | untreated (seeds 1 to 3) | ' + KIND_LABEL['transplant'] + ' | ' + KIND_LABEL['matched'] + ' |',
            '|---|---|---|---|---|---|---|---|---|']
        for col, label, fmt, dkey in STATS:
            if col not in PAIRED_COLS or col not in comp.columns:
                continue
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            cells = [format(float(real[col].iloc[0]), fmt) if len(real) == 1 and pd.notna(real[col].iloc[0]) else '-']
            dfmt = '+' + fmt.lstrip('+'); sfmt = fmt.lstrip('+')
            for cls in CLASSES:
                q = p[(p.shuffle == cls) & (p.stat == col)].set_index('kind')
                if q.empty:
                    cells += ['-', '-', '-']; continue
                cells.append(format(float(q.untreated_mean.iloc[0]), fmt))
                for kind in KINDS:
                    if kind in q.index:
                        rr = q.loc[kind]
                        cells.append(f'{format(rr.mean_diff, dfmt)} ± {format(rr.sd_diff, sfmt)}' if rr.n > 1 else format(rr.mean_diff, dfmt))
                    else:
                        cells.append('-')
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
    # the matchings
    oc_path = res_dir / 'matching_outcome.csv'
    if oc_path.exists():
        oc = pd.read_csv(oc_path)
        rows = ['**The matchings (per draw): annealed claw swaps toward the real wiring\'s 54 x 54 shared-KC counts**', '',
            '| draw | swaps within KC type | type layer held | objective: start -> end | pair |diff| max / mean | co-input correlation with '
            'the real wiring: draw -> matched | three-way co-input correlation with the real wiring: draw -> matched | type layer |diff| max: '
            'draw -> matched | claws changed vs the draw | claws differing from the real wiring: draw -> matched | seconds |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in oc.itertuples():
            rows.append(f'| {CLASS_LABEL[r.shuffle]} seed {r.seed} | {"yes" if r.within_kc_type else "no"} | {"yes" if r.match_type_layer else "no"} '
                f'| {r.objective_start:.0f} -> {r.objective_end:.0f} | {r.pair_abs_diff_max} / {r.pair_abs_diff_mean:.2f} | '
                f'{r.draw_coinput_corr_vs_real:.3f} -> {r.matched_coinput_corr_vs_real:.3f} | {r.draw_triplet_corr_vs_real:.3f} -> '
                f'{r.matched_triplet_corr_vs_real:.3f} | {r.draw_type_layer_abs_diff_max} -> {r.matched_type_layer_abs_diff_max} | '
                f'{r.frac_claws_changed_vs_draw:.3f} | {r.draw_frac_claws_changed_vs_real:.3f} -> {r.matched_frac_claws_changed_vs_real:.3f} | '
                f'{r.wall_seconds:.0f} |')
        out.append('\n'.join(rows))
    # the re-fits
    fits = comp[comp.variant == 'fit']
    if not fits[fits.part == 'C'].empty:
        def fixed(shuffle, seed, source=comp):
            s = source[(source.model == 'thr40_gain1.5') & (source.shuffle == shuffle) & (source.variant == 'retuned')]
            if seed is not None:
                s = s[s.seed == seed]
            return s
        e54 = pd.read_csv(H.shared_dir() / 'results' / E54 / 'comparison.csv')
        cols_fit = [('fit_real', fixed('real', None), 'real wiring'),
            ('refit_lesion_targeted-seed0', fixed('lesion_targeted', 0, e54), 'the 55 over-shared pairs lesioned (E52 / E54, draw 0)'),
            ('refit_curveball_within_type_transplant-seed1', fixed('curveball_within_type_transplant', 1),
                'Curveball within type + the 32 pairs\' convergence (E57, seed 1)'),
            ('refit_curveball_within_type_matched-seed1', fixed('curveball_within_type_matched', 1),
                'Curveball within type, all pairs matched (seed 1)')]
        cols_fit = [(tag, fx, lab) for tag, fx, lab in cols_fit if (fits.shuffle == tag).any()]
        rows = ['**The eight at 1.5, 40 Hz re-fitted on convergence-bearing wirings (structure loss, 150 evaluations, exact start from the '
            'real-wiring optimum), beside the fixed-scale run on the same wiring (threshold re-tuned)**', '',
            '| | | ' + ' | '.join(f'{lab} {{span=2}}' for _, _, lab in cols_fit) + ' |',
            '| statistic | KC data | ' + ' | '.join('fixed scales | re-fitted' for _ in cols_fit) + ' |',
            '|---|---|' + '---|---|' * len(cols_fit)]
        for col, label, fmt, dkey in STATS:
            if col not in comp.columns:
                continue
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            cells = []
            for tag, fx, _ in cols_fit:
                cells.append(format(float(fx[col].iloc[0]), fmt) if len(fx) >= 1 and col in fx.columns and pd.notna(fx[col].iloc[0]) else '-')
                v = fits[fits.shuffle == tag][col]
                cells.append(format(float(v.iloc[0]), fmt) if len(v) >= 1 and pd.notna(v.iloc[0]) else '-')
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))
    (res_dir / 'comparison_table.md').write_text('\n\n'.join(out) + '\n')
    figure(comp[comp.variant != 'fit'], data, res_dir, groups)
    print('\n\n'.join(out))


def figure(comp: pd.DataFrame, data: Dict[str, float], res_dir: Path, groups: List[str]) -> None:
    plt = H._plt()
    from matplotlib.lines import Line2D
    panels = [(f'{V}__off_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL off', 'tnt_vcf'),
        (f'{V}__on_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL on', 'ctrl_vcf'),
        (f'{V}__block_fruity_small_iso', 'fruity × small-ester block', 'block'),
        ('spearman_rate_on_vs_data', 'response rate ρ, APL on', 'ctrl_rate_rho_ceiling'),
        ('loss_rate_ctrl', 'extended loss at 6.8 % (lower is better)', None)]
    xs = ['real'] + [g for g in groups if (comp.shuffle == g).any()]
    colors = {'thr70_gain1.5': '#1f5fbf', 'thr40_gain1.5': '#d1662a', 'thr70_gain1.25': '#2a9d5c'}
    offs = {'thr70_gain1.5': -0.2, 'thr40_gain1.5': 0.0, 'thr70_gain1.25': 0.2}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 6.8))
    for ax, (col, label, dkey) in zip(axes, panels):
        for m in MODELS:
            sub = comp[comp.model == m]
            for i, g in enumerate(xs):
                v = sub[sub.shuffle == g][col].astype(float).dropna()
                if v.empty:
                    continue
                x = i + offs[m]
                ax.scatter(np.full(len(v), x) + np.linspace(-0.03, 0.03, len(v)), v, s=14, color=colors[m], alpha=0.6, zorder=3)
                if len(v) > 1:
                    ax.errorbar([x], [v.mean()], yerr=[v.std(ddof=1)], fmt='_', color=colors[m], ms=12, mew=2, capsize=3, zorder=4)
                else:
                    ax.scatter([x], v, marker='_', s=150, color=colors[m], linewidths=2, zorder=4)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='k', ls='--', lw=1)
        n_within = 1 + sum(1 for g in xs if g.startswith('curveball_within_type'))
        ax.axvline(n_within - 0.5, color='0.7', lw=0.8)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([GROUP_LABEL[g] for g in xs], rotation=35, ha='right', fontsize=7)
        ax.set_title(label, fontsize=9); ax.grid(axis='y', alpha=0.25)
    handles = [Line2D([], [], marker='o', ls='', color=colors[m], label=MODEL_LABEL[m]) for m in MODELS]
    handles.append(Line2D([], [], ls='--', color='k', label='KC data (target or ceiling)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle('The co-input-preserving shuffle: the reference models at fixed scales on Curveball draws (within type / plain), untreated, '
        'with the 32 pairs\' convergence transplanted, or with all 1431 pairs\' shared-KC counts matched to the real wiring\'s '
        '(points: draws; bar: mean ± sd)', fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(res_dir / f'{EXPERIMENT}.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3], help='the Curveball draws (E47\'s seeds)')
    ap.add_argument('--workers', type=int, default=6, help='processes for the matchings')
    ap.add_argument('--plan', action='store_true', help='only build the matched wirings and their table')
    ap.add_argument('--summary', action='store_true', help='only (re)write the comparison tables and the figure')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    if args.summary:
        summary(res_dir); return
    wirings = make_wirings(args.seeds, out_dir, res_dir, workers=args.workers)
    pd.set_option('display.width', 250)
    oc = pd.read_csv(res_dir / 'matching_outcome.csv')
    print(oc[['shuffle', 'seed', 'objective_start', 'objective_end', 'pair_abs_diff_max', 'pair_abs_diff_mean', 'draw_coinput_corr_vs_real',
        'matched_coinput_corr_vs_real', 'draw_triplet_corr_vs_real', 'matched_triplet_corr_vs_real', 'matched_type_layer_abs_diff_max',
        'frac_claws_changed_vs_draw', 'matched_frac_claws_changed_vs_real', 'wall_seconds']].to_string(index=False))
    if args.plan:
        return
    jobs = jobs_for(wirings)
    print(f'{len(jobs)} variants', flush=True)
    run_variants(jobs, res_dir, out_dir)  # writes provenance.json beside the stats
    summary(res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
