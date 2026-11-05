#!/usr/bin/env python3
"""E5: KC heterogeneity. Does per-KC / per-type heterogeneity add pair-specific deviations
(and the data's APL block / drive dependence) at the same mean response rate?

For each base model (chapter-3 claw model with and without claw dynamics; optionally the
hemibrain wd20 model) tuned with APL on to a target mean response rate, the variants:
- base: one threshold for all KCs (as E1)
- homeostatic: olfsysm homeostatic thresholds (one per KC; `homeostatic_thrs=True`)
- type_equal: one threshold per KC type so pre-APL rates are equal across types
  (`equalize_kc_type_sparsity=True`)
- type_abprime_x2: as type_equal, but a'b' KCs at twice the target (Inada et al. 2017:
  a'b' most excitable)
- coincidence_2 / coincidence_3: a KC spikes only if >= n claws are active
  (`n_claws_active_to_spike`; claw bases only)
- thr_nclaws_0.5 / thr_nclaws_1: per-KC threshold proportional to n_claws^k (k = 0.5, 1),
  rescaled (APL weights as in base) to the base's APL-on response rate
- apl_type_scaled: connectome APL<->KC weights multiplied per KC type (a'b' x1.6, ab x1,
  g x0.6, unknown x1; claw bases only), then tuned as usual

Each variant gives an APL-on model and an acute APL-off model (same thresholds,
`wAPLKC = 0`), scored with `kc_remapping_harness` as in E1: model on vs TNT3c, off vs TNT
(Spearman, Spearman of deviations from the ORN prediction), on vs off (Spearman, isotonic
R2, fruity x small-ester block, per-odor APL effect vs ORN drive), 66-pair references
(deviations vs Remy's WT somata, residual vs VCF, claws), response rates overall, per
type and per odor.

Each variant is simulated in its own subprocess (`--run-variant`), so an olfsysm crash in
one (hal saw a hard crash for homeostatic thresholds on the claw-dynamics model) does not
stop the rest; failures are recorded in `run_status.csv`. Re-running skips variants whose
outputs exist. Large outputs (responses / spike counts parquet) go to
./outputs/E5_kc_heterogeneity/, compact results to <shared>/results/E5_kc_heterogeneity/.

Usage: OMP_NUM_THREADS=6 nice -n 10 ./venv/bin/python E5_kc_heterogeneity.py
    [--bases claw-dynamics claw] [--variants ...] [--target 0.05] [--score-only]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H


EXP = 'E5_kc_heterogeneity'
GENOS = ('TNT3c', 'TNT')
CTRL, TNT = GENOS

BASES: Dict[str, dict] = {
    'claw-dynamics': dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True, pn_claw_to_apl=True, claw_dynamics=True,
        n_spikes_for_response=2,
    ),
    'claw': dict(one_row_per_claw=True, prat_claws=True, use_connectome_APL_weights=True,
        n_spikes_for_response=2,
    ),
    'hemibrain-wd20': dict(pn2kc_connections='hemibrain', weight_divisor=20,
        _drop_glom_with_plus=False,
    ),
}
CLAW_BASES = ('claw-dynamics', 'claw')
# variant -> extra fit_mb_model kwargs ('special' variants are built at run time)
VARIANTS: Dict[str, Optional[dict]] = {
    'base': {},
    'homeostatic': dict(homeostatic_thrs=True),
    'type_equal': dict(equalize_kc_type_sparsity=True),
    'type_abprime_x2': dict(equalize_kc_type_sparsity=True),  # ab' target set at run time
    'coincidence_2': dict(n_claws_active_to_spike=2),
    'coincidence_3': dict(n_claws_active_to_spike=3),
    'thr_nclaws_0.5': None,
    'thr_nclaws_1': None,
    'apl_type_scaled': None,
}
CLAW_ONLY_VARIANTS = ('coincidence_2', 'coincidence_3', 'apl_type_scaled')
# fit_mb_model kwargs that are only meaningful while tuning (dropped for the fixed-param
# APL-off re-fit)
TUNING_VARIANT_KWS = ('homeostatic_thrs', 'equalize_kc_type_sparsity',
    'ab_prime_response_rate_target')
APL_TYPE_FACTORS: Dict[str, float] = {"a'b'": 1.6, 'ab': 1.0, 'g': 0.6, 'unknown': 1.0}
KC_TYPE = 'kc_type'
KC_ID = 'kc_id'


def tag_of(base: str, variant: str, target: float) -> str:
    return f'{base}_sp{target:g}_{variant}'


def rate(responses: pd.DataFrame) -> float:
    return float(responses.astype(float).mean().mean())


def n_claws_per_kc(kws: dict, kc_ids: pd.Index) -> pd.Series:
    """Number of claws (input PN-KC connections) of each KC id, from the connectome."""
    from al_analysis.mb_model import connectome_wPNKC, get_connectome_wPNKC_params

    wPNKC = connectome_wPNKC(**get_connectome_wPNKC_params(kws))
    if 'claw_id' in wPNKC.index.names:
        n = wPNKC.groupby(level=KC_ID).size()
    else:
        n = wPNKC.sum(axis='columns').groupby(level=KC_ID).sum()
    out = n.reindex(kc_ids)
    assert not out.isna().any(), 'KC ids in model outputs missing from connectome matrix'
    return out.astype(float)


def type_scaled_apl_weights(kws: dict) -> Tuple[pd.Series, pd.Series]:
    """Connectome per-claw APL<->KC weights multiplied by `APL_TYPE_FACTORS` per KC type."""
    from al_analysis.mb_model import (connectome_wPNKC, connectome_APL_weights,
        get_connectome_wPNKC_params)

    params = get_connectome_wPNKC_params(kws)
    wPNKC = connectome_wPNKC(**params)
    wAPLKC, wKCAPL, _, _ = connectome_APL_weights(wPNKC=wPNKC, **params)
    types = wAPLKC.index.get_level_values(KC_TYPE)
    factors = pd.Series(types.map(APL_TYPE_FACTORS).to_numpy(dtype=float), index=wAPLKC.index)
    assert not factors.isna().any(), set(types) - set(APL_TYPE_FACTORS)
    return wAPLKC * factors, wKCAPL * factors


def bisect_scale(sim: Callable, thr_vec: np.ndarray, target: float, *, extra: dict,
    tol: float = 0.02, max_iter: int = 18) -> Tuple[float, float, pd.DataFrame, pd.DataFrame]:
    """Scale `thr_vec` by k (rate decreasing in k) until the mean response rate is within
    `tol` (relative) of `target`. Returns (k, rate, responses, spike_counts)."""
    def run(k: float):
        r, s = sim({'fixed_thr': thr_vec * k, **extra})
        return rate(r), r, s

    lo, hi = 0.2, 5.0
    r_lo, *_ = run(lo)
    r_hi, *_ = run(hi)
    assert r_lo > target > r_hi, (r_lo, target, r_hi)
    best = None
    for _ in range(max_iter):
        k = float(np.sqrt(lo * hi))
        rk, r, s = run(k)
        if best is None or abs(rk - target) < abs(best[1] - target):
            best = (k, rk, r, s)
        if abs(rk - target) / target < tol:
            break
        if rk > target:
            lo = k
        else:
            hi = k
    return best


def run_variant(base: str, variant: str, target: float, out_dir: Path) -> dict:
    """Simulates one variant (APL on, then acute APL off) and writes its outputs."""
    from al_analysis.mb_model import (fit_mb_model, make_fixed_param_kc_simulator,
        get_thr_and_APL_weights)

    t0 = time.time()
    orn = H.apl16_orn_deltas()
    kws = dict(BASES[base])
    common = dict(make_plots=False, silent=True)
    tag = tag_of(base, variant, target)
    info: dict = dict(base=base, variant=variant, target=target, tag=tag)
    extra = VARIANTS[variant]

    if variant in ('base', 'thr_nclaws_0.5', 'thr_nclaws_1', 'apl_type_scaled'):
        if variant == 'apl_type_scaled':
            wAPLKC, wKCAPL = type_scaled_apl_weights(kws)
            kws.update(_wAPLKC=wAPLKC, _wKCAPL=wKCAPL)
            info['apl_type_factors'] = APL_TYPE_FACTORS
        sim, params = make_fixed_param_kc_simulator(orn, **kws, target_sparsity=target,
            **common)
        thr0 = params['fixed_thr']
        info['params'] = {k: (float(v) if np.ndim(v) == 0 else float(np.mean(v)))
            for k, v in params.items()}
        if variant.startswith('thr_nclaws'):
            k_exp = float(variant.split('_')[-1])
            resp0, _ = sim({})
            r_base = rate(resp0)
            kc_ids = resp0.index.get_level_values(KC_ID)
            ncl = n_claws_per_kc(kws, kc_ids).to_numpy()
            vec = float(np.mean(thr0)) * (ncl / ncl.mean()) ** k_exp
            k, r_on, resp_on, spk_on = bisect_scale(sim, vec, r_base, extra={})
            thr = vec * k
            info.update(thr_exponent=k_exp, thr_scale=k, base_rate_on=r_base,
                n_claws_mean=float(ncl.mean()))
            resp_off, spk_off = sim({'fixed_thr': thr, 'wAPLKC_scale': 0.0})
        else:
            thr = thr0
            resp_on, spk_on = sim({})
            resp_off, spk_off = sim({'wAPLKC_scale': 0.0})
        del sim
    else:
        kws_tune = dict(kws, **extra)
        if variant == 'type_abprime_x2':
            kws_tune['ab_prime_response_rate_target'] = 2 * target
        resp_on, spk_on, _, pdict = fit_mb_model(orn, **kws_tune, target_sparsity=target,
            **common)
        fixed = get_thr_and_APL_weights(pdict, kws_tune)
        thr = fixed['fixed_thr']
        info['params'] = {k: (float(v) if np.ndim(v) == 0 else float(np.mean(v)))
            for k, v in fixed.items() if v is not None}
        if 'type2thr' in pdict:
            info['type2thr'] = {str(k): float(v) for k, v in pdict['type2thr'].items()}
        kws_off = {k: v for k, v in kws_tune.items() if k not in TUNING_VARIANT_KWS}
        fixed_off = dict(fixed, wAPLKC=0.0)
        resp_off, spk_off, _, _ = fit_mb_model(orn, **kws_off, **fixed_off, **common)

    thr_arr = np.asarray(thr, dtype=float)
    info.update(n_kcs=int(len(resp_on)), thr_mean=float(thr_arr.mean()),
        thr_sd=float(thr_arr.std()) if thr_arr.ndim else 0.0,
        thr_is_vector=bool(thr_arr.ndim), rate_on=rate(resp_on), rate_off=rate(resp_off),
        seconds=time.time() - t0,
    )
    if thr_arr.ndim and KC_TYPE in resp_on.index.names:
        types = resp_on.index.get_level_values(KC_TYPE)
        info['thr_mean_by_type'] = {str(t): float(thr_arr[types == t].mean())
            for t in sorted(set(types))}
    for cond, (r, s) in (('on', (resp_on, spk_on)), ('acute_off', (resp_off, spk_off))):
        r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
        s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
    (out_dir / f'{tag}_info.json').write_text(json.dumps(info, indent=1, default=str))
    return info


# ---------------------------------------------------------------------------
# scoring
def kc_type_rates(responses: pd.DataFrame) -> Dict[str, float]:
    if KC_TYPE not in responses.index.names:
        return {}
    r = responses.astype(float)
    return {f'rate_{t}': float(g.mean().mean()) for t, g in r.groupby(level=KC_TYPE)}


def score(out_dir: Path, res_dir: Path, tags: List[Tuple[str, str, float]]) -> pd.DataFrame:
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in GENOS},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in GENOS}}
    data_rates = {g: H.yang_response_fractions(g).mean() for g in GENOS}
    rows, pair_rows, odor_rows = [], [], []
    saved = {}
    for base, variant, target in tags:
        tag = tag_of(base, variant, target)
        info_p = out_dir / f'{tag}_info.json'
        if not info_p.exists():
            continue
        info = json.loads(info_p.read_text())
        conds = {}
        for cond in ('on', 'acute_off'):
            resp = pd.read_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
            spk = pd.read_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
            conds[cond] = (resp, spk, H.model_rdms(resp, spk))
            for variant_name, pairs in conds[cond][2].items():
                saved[(tag, cond, variant_name)] = pairs
            fr = H.response_fractions(resp)
            for odor, v in fr.items():
                odor_rows.append(dict(base=base, variant=variant, condition=cond, odor=odor,
                    response_fraction=v))
        on_r, on_s, on_rdm = conds['on']
        off_r, off_s, off_rdm = conds['acute_off']
        row = dict(base=base, variant=variant, target=target, n_kcs=info['n_kcs'],
            thr_mean=info['thr_mean'], thr_cv=info['thr_sd'] / info['thr_mean'],
            wAPLKC_scale=info['params'].get('wAPLKC_scale', info['params'].get('wAPLKC')),
            rate_on=info['rate_on'], rate_off=info['rate_off'], seconds=info['seconds'])
        row.update({f'{k}_on': v for k, v in kc_type_rates(on_r).items()})
        st, tabs = H.compare_on_off(on_rdm[H.PRIMARY_MODEL_RDM], off_rdm[H.PRIMARY_MODEL_RDM],
            orn=orn_pairs, drive=drive)
        for k in ('mean_on', 'mean_off', 'spearman_on_off', 'iso_r2', 'block_fruity_small_iso',
                'block_other_between_iso', 'spearman_per_odor_iso_vs_drive',
                'spearman_orn_on', 'spearman_orn_off', 'gap_closed'):
            row[k] = st[k]
        tabs['blocks'].to_csv(res_dir / f'blocks_{tag}.csv')
        tabs['per_odor'].to_csv(res_dir / f'per_odor_{tag}.csv')
        st_b, _ = H.compare_on_off(on_rdm['responses_cosine'], off_rdm['responses_cosine'],
            orn=orn_pairs, drive=drive)
        row['block_fruity_small_iso_binary'] = st_b['block_fruity_small_iso']
        row['iso_r2_binary'] = st_b['iso_r2']
        for cond, rdm, geno in (('on', on_rdm, CTRL), ('off', off_rdm, TNT)):
            c = H.compare_model_to_data(rdm[H.PRIMARY_MODEL_RDM], data['raw'][geno],
                orn=orn_pairs)
            row[f'spearman_vs_{geno}_{cond}'] = c['spearman']
            row[f'spearman_dev_vs_{geno}_{cond}'] = c['spearman_dev']
            row[f'dev_sd_model_{cond}'] = c['dev_sd_model']
            cb = H.compare_model_to_data(rdm['responses_cosine'], data['binarized'][geno],
                orn=orn_pairs)
            row[f'spearman_vs_{geno}_{cond}_binary'] = cb['spearman']
            row[f'spearman_dev_vs_{geno}_{cond}_binary'] = cb['spearman_dev']
            rr = H.compare_to_references(rdm[H.PRIMARY_MODEL_RDM], ref)
            for k in ('spearman_resid_vs_remap_wt', 'spearman_resid_vs_vcf_hyp',
                    'spearman_kc_claw', 'spearman_kc_soma', 'spearman_vcf_hyp'):
                row[f'{k}_{cond}'] = rr[k]
            rrn = H.compare_to_references(rdm['responses_pearson_nosilent'], ref)
            row[f'dev66_vs_wt_resp_pearson_nosilent_{cond}'] = rrn['spearman_resid_vs_remap_wt']
            fr = H.response_fractions(on_r if cond == 'on' else off_r)
            row[f'per_odor_rate_spearman_vs_{geno}_{cond}'] = H.spearman(fr,
                data_rates[geno].reindex(fr.index))
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(res_dir / 'E5_summary.csv', index=False)
    pd.DataFrame(odor_rows).to_csv(res_dir / 'response_fractions_per_odor.csv', index=False)
    if saved:
        long = pd.concat(saved, names=['tag', 'condition', 'variant', 'odor_a', 'odor_b']
            ).rename('distance').reset_index()
        long.to_csv(res_dir / 'model_pair_distances_long.csv', index=False)
    # data rows for reference
    st, _ = H.compare_on_off(data['raw'][CTRL], data['raw'][TNT], orn=orn_pairs, drive=drive)
    rr = {g: H.compare_to_references(data['raw'][g], ref) for g in GENOS}
    data_row = dict(base='DATA', variant='TNT3c vs TNT', mean_on=st['mean_on'],
        mean_off=st['mean_off'], spearman_on_off=st['spearman_on_off'], iso_r2=st['iso_r2'],
        block_fruity_small_iso=st['block_fruity_small_iso'],
        spearman_per_odor_iso_vs_drive=st['spearman_per_odor_iso_vs_drive'],
        spearman_orn_on=st['spearman_orn_on'], spearman_orn_off=st['spearman_orn_off'],
        gap_closed=st['gap_closed'],
        spearman_resid_vs_remap_wt_on=rr[CTRL]['spearman_resid_vs_remap_wt'],
        spearman_resid_vs_vcf_hyp_on=rr[CTRL]['spearman_resid_vs_vcf_hyp'],
        spearman_kc_claw_on=rr[CTRL]['spearman_kc_claw'],
        spearman_resid_vs_vcf_hyp_off=rr[TNT]['spearman_resid_vs_vcf_hyp'],
        spearman_kc_claw_off=rr[TNT]['spearman_kc_claw'],
        rate_on=float(data_rates[CTRL].mean()), rate_off=float(data_rates[TNT].mean()))
    pd.concat([tab, pd.DataFrame([data_row])]).to_csv(res_dir / 'E5_summary_with_data.csv',
        index=False)
    return tab


def make_figures(res_dir: Path, tab: pd.DataFrame, out_dir: Path) -> None:
    plt = H._plt()
    plt.rcParams.update({'font.size': 7})
    cols = [('rate_on', 'rate on'), ('thr_cv', 'threshold CV'),
        ('spearman_on_off', 'Spearman on/off'), ('iso_r2', 'iso R2 on/off'),
        ('block_fruity_small_iso', 'block residual'),
        ('spearman_per_odor_iso_vs_drive', 'per-odor effect vs drive'),
        (f'spearman_vs_{CTRL}_on', 'Spearman vs ctrl (on)'),
        (f'spearman_dev_vs_{CTRL}_on', 'dev vs ctrl (on)'),
        (f'spearman_vs_{TNT}_off', 'Spearman vs TNT (off)'),
        ('spearman_resid_vs_remap_wt_on', 'dev66 vs WT (on)'),
        ('spearman_resid_vs_vcf_hyp_on', 'resid vs VCF (on)'),
        ('spearman_kc_claw_on', 'Spearman claws (on)')]
    data_ref = {'spearman_on_off': 0.869, 'iso_r2': 0.939, 'block_fruity_small_iso': 0.055,
        'spearman_per_odor_iso_vs_drive': 0.50, 'spearman_resid_vs_remap_wt_on': 0.83,
        'spearman_resid_vs_vcf_hyp_on': 0.44, 'spearman_kc_claw_on': 0.68, 'rate_on': 0.068}
    t = tab.copy()
    t['label'] = t.base + ' / ' + t.variant
    n = len(cols)
    fig, axs = plt.subplots(1, n, figsize=(1.55 * n + 2.6, 0.3 * len(t) + 1.5), sharey=True)
    y = np.arange(len(t))[::-1]
    for ax, (col, lab) in zip(axs, cols):
        if col not in t:
            ax.set_visible(False)
            continue
        ax.scatter(t[col], y, s=16, color='#444444', zorder=3)
        base_mask = (t.variant == 'base').to_numpy()
        ax.scatter(t[col][base_mask], y[base_mask], s=30, facecolors='none',
            edgecolors=H.CONDITION_COLORS['control'], zorder=4)
        if col in data_ref:
            ax.axvline(data_ref[col], color=H.CONDITION_COLORS['TNT'], lw=1, ls='--')
        ax.set_title(lab, fontsize=7)
        ax.grid(axis='x', color='#eeeeee', lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(t.label, fontsize=7)
    fig.suptitle('E5 KC heterogeneity: variants (rows; circled = base) vs the data (red '
        'dashed, where applicable). Primary recipe: 1 - cosine over spike counts.', y=1.02)
    fig.savefig(res_dir / 'E5_summary.png', bbox_inches='tight', dpi=150)
    fig.savefig(res_dir / 'E5_summary.pdf', bbox_inches='tight')
    plt.close(fig)

    # on-vs-off scatters with data overlay, per base
    data_on = H.yang_mean_rdm(CTRL, 'cosine')
    data_off = H.yang_mean_rdm(TNT, 'cosine')
    for base, g in tab.groupby('base'):
        vs = list(g.variant)
        ncol = min(5, len(vs))
        nrow = int(np.ceil(len(vs) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.5 * nrow), squeeze=False)
        for ax, (_, r) in zip(axs.flat, g.iterrows()):
            tag = tag_of(base, r.variant, r.target)
            on = H.pair_distances(pd.read_parquet(out_dir / f'{tag}_on_spike_counts.parquet'))
            off = H.pair_distances(pd.read_parquet(
                out_dir / f'{tag}_acute_off_spike_counts.parquet'))
            H.plot_on_vs_off(ax, on, off, label=r.variant, data_on=data_on, data_off=data_off)
            ax.set_title(f'{base} / {r.variant}: Spearman {r.spearman_on_off:.2f}, iso R2 '
                f'{r.iso_r2:.2f}, block {r.block_fruity_small_iso:+.3f}', fontsize=7)
        for ax in axs.flat[len(vs):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(res_dir / f'E5_on_vs_off_{base}.png', dpi=120)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bases', nargs='+', default=list(CLAW_BASES), choices=list(BASES))
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--target', type=float, default=0.05)
    ap.add_argument('--timeout-min', type=float, default=45)
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--run-variant', nargs=2, metavar=('BASE', 'VARIANT'),
        help='(internal) simulate one variant in this process')
    args = ap.parse_args()
    out_dir = H.outputs_dir(EXP)
    res_dir = H.results_dir(EXP)

    if args.run_variant:
        base, variant = args.run_variant
        info = run_variant(base, variant, args.target, out_dir)
        print(json.dumps({k: v for k, v in info.items() if k != 'params'}, default=str))
        return

    plan = [(b, v, args.target) for b in args.bases for v in args.variants
        if not (v in CLAW_ONLY_VARIANTS and b not in CLAW_BASES)]
    status_p = res_dir / 'run_status.csv'
    status = pd.read_csv(status_p).to_dict('records') if status_p.exists() else []
    done = {(s['base'], s['variant'], s['target']) for s in status if s['status'] == 'ok'}
    if not args.score_only:
        for base, variant, target in plan:
            tag = tag_of(base, variant, target)
            if (base, variant, target) in done and (out_dir / f'{tag}_info.json').exists():
                continue
            t0 = time.time()
            cmd = [sys.executable, str(Path(__file__).resolve()), '--run-variant', base,
                variant, '--target', str(target)]
            print(f'[{datetime.now():%H:%M:%S}] running {tag} ...', flush=True)
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                    timeout=args.timeout_min * 60, cwd=H.HERE)
                ok = p.returncode == 0 and (out_dir / f'{tag}_info.json').exists()
                st = 'ok' if ok else f'failed (rc {p.returncode})'
                err_tail = '\n'.join((p.stderr or '').strip().splitlines()[-15:])
            except subprocess.TimeoutExpired:
                st, err_tail = 'timeout', ''
            status = [s for s in status if not (s['base'] == base and s['variant'] == variant
                and s['target'] == target)]
            status.append(dict(base=base, variant=variant, target=target, status=st,
                seconds=time.time() - t0, when=datetime.now(timezone.utc).isoformat(
                timespec='seconds'), stderr_tail=err_tail))
            pd.DataFrame(status).to_csv(status_p, index=False)
            print(f'  -> {st} ({time.time() - t0:.0f} s)' + ('' if st == 'ok' else
                f'\n{err_tail}'), flush=True)

    tab = score(out_dir, res_dir, plan)
    pd.set_option('display.width', 250)
    show = ['base', 'variant', 'rate_on', 'rate_off', 'thr_cv', 'spearman_on_off', 'iso_r2',
        'block_fruity_small_iso', 'spearman_per_odor_iso_vs_drive', f'spearman_vs_{CTRL}_on',
        f'spearman_dev_vs_{CTRL}_on', f'spearman_vs_{TNT}_off', 'spearman_resid_vs_remap_wt_on',
        'spearman_resid_vs_vcf_hyp_on', 'spearman_kc_claw_on']
    print(tab[[c for c in show if c in tab]].round(3).to_string())
    if len(tab):
        make_figures(res_dir, tab, out_dir)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E5_kc_heterogeneity.py', host=os.uname().nodename,
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        omp_threads=os.environ.get('OMP_NUM_THREADS'), plan=plan,
        apl_type_factors=APL_TYPE_FACTORS, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str))
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
