#!/usr/bin/env python3
"""E15: breadth x shared-input APL weighting, the mechanistic version of E14's oracle.

E14 showed that inhibiting the APL-off fruity x small-ester co-responders about 4x harder
than other KCs (coincidence-3 claw-dynamics base) reproduces the data's block and improves
nearly every other statistic, and that those cells are broad responders that all pool claws
from the eight glomeruli whose ORN drive is shared by the two ester groups. Here the per-KC
APL->KC weight is set by a rule that does not know who the co-responders are:

    w_i = (1 + breadth_i)^k * (1 + shared_claws_i)^m

with breadth_i = number of odors KC i responds to in the APL-off base model, shared_claws_i
= its number of claws from the shared-drive glomeruli (E11's set), k in {0, 1, 2, 3}, m in
{0, 1, 2} (k = m = 0 is the base); plus an activity-defined self-inhibition rule (Amin et al.
2020 in spirit): w_i = (1 + total APL-off spikes of KC i)^q, q in {1, 2}. The factors
multiply the connectome per-claw APL->KC weights of each KC's claws; the APL scale is
re-tuned to 5 %, so only the pattern of inhibition changes. APL on and acute off for each.

Scoring as E14 (harness statistics, E7's excess profile, the E14 co-responder set's APL-on
rate and surviving fraction) plus how well each rule's weights rank the oracle set (AUC of
w_i for co-responders vs other KCs).

Outputs: compact results to <shared>/results/E15_breadth_shared_apl/, per-KC outputs to
./outputs/E15_breadth_shared_apl/. Each variant runs in its own subprocess; `base` first.

Usage: OMP_NUM_THREADS=6 nice -n 10 ./venv/bin/python E15_breadth_shared_apl.py [--score-only]
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E5_kc_heterogeneity import BASES, rate, kc_type_rates
from E11_coincidence_claw_inhibition import excess_profile, shared_drive_glomeruli
from E14_oracle_coresponders import fit_on_off, coresponders, set_stats, FRUITY, SMALL, KC_ID


EXP = 'E15_breadth_shared_apl'
TARGET = 0.05
GENOS = ('TNT3c', 'TNT')
CTRL, TNT = GENOS
BASE_KWS = dict(BASES['claw-dynamics'], n_claws_active_to_spike=3)
BASE = 'claw-dynamics_coinc3'

KS = (0, 1, 2, 3)
MS = (0, 1, 2)
VARIANTS: Dict[str, Tuple[str, float, float]] = {'base': ('base', 0, 0)}
for k, m in itertools.product(KS, MS):
    if (k, m) != (0, 0):
        VARIANTS[f'k{k}_m{m}'] = ('breadth_shared', k, m)
VARIANTS['self1'] = ('self', 1, 0)
VARIANTS['self2'] = ('self', 2, 0)


def tag_of(variant: str) -> str:
    return f'{BASE}_sp{TARGET:g}_{variant}'


def kc_features(resp_off: pd.DataFrame, spk_off: pd.DataFrame, kws: dict) -> pd.DataFrame:
    """Per-KC APL-off breadth, total spikes, claws from shared-drive glomeruli."""
    from al_analysis.mb_model import connectome_wPNKC, get_connectome_wPNKC_params

    orn = H.apl16_orn_deltas()
    gl = list(shared_drive_glomeruli(orn).index)
    wPNKC = connectome_wPNKC(**get_connectome_wPNKC_params(kws))
    claw_glom = wPNKC.idxmax(axis='columns')
    claw_glom = pd.Series([g if isinstance(g, str) else g[0] for g in claw_glom], index=wPNKC.index)
    kc_ids = wPNKC.index.get_level_values(KC_ID)
    shared = pd.Series(claw_glom.isin(gl).to_numpy().astype(int), index=kc_ids).groupby(level=0).sum()
    r = H.abbrev_columns(resp_off.astype(float))
    ids = r.index.get_level_values(KC_ID)
    df = pd.DataFrame({'breadth_off': r.sum(axis='columns').to_numpy(),
        'spikes_off': H.abbrev_columns(spk_off.astype(float)).sum(axis='columns').to_numpy(),
        'shared_claws': shared.reindex(ids).fillna(0).to_numpy()}, index=pd.Index(ids, name=KC_ID))
    df['coresponder'] = df.index.isin(coresponders(resp_off, FRUITY, SMALL))
    return df


def rule_weights(feat: pd.DataFrame, kind: str, a: float, b: float) -> pd.Series:
    if kind == 'breadth_shared':
        w = (1 + feat.breadth_off) ** a * (1 + feat.shared_claws) ** b
    elif kind == 'self':
        w = (1 + feat.spikes_off) ** a
    else:
        raise ValueError(kind)
    return w / w.mean()


def per_claw_weights(kws: dict, w_kc: pd.Series) -> Tuple[pd.Series, pd.Series]:
    from al_analysis.mb_model import (connectome_wPNKC, connectome_APL_weights,
        get_connectome_wPNKC_params)

    params = get_connectome_wPNKC_params(kws)
    wPNKC = connectome_wPNKC(**params)
    wAPLKC, wKCAPL, _, _ = connectome_APL_weights(wPNKC=wPNKC, **params)
    f = w_kc.reindex(wAPLKC.index.get_level_values(KC_ID)).fillna(1.0).to_numpy()
    return wAPLKC * f, wKCAPL


def auc(score: pd.Series, positive: pd.Series) -> float:
    from scipy.stats import rankdata
    r = rankdata(score.to_numpy())
    pos = positive.to_numpy().astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_variant(variant: str, out_dir: Path) -> dict:
    t0 = time.time()
    orn = H.apl16_orn_deltas()
    kws = dict(BASE_KWS)
    common = dict(make_plots=False, silent=True)
    kind, a, b = VARIANTS[variant]
    tag = tag_of(variant)
    info: dict = dict(base=BASE, variant=variant, kind=kind, k=a, m=b, tag=tag)
    feat_p = out_dir / f'{BASE}_kc_features.csv'
    if variant == 'base':
        resp_on, spk_on, resp_off, spk_off, fixed = fit_on_off(orn, kws, common)
        feat = kc_features(resp_off, spk_off, kws)
        feat.to_csv(feat_p)
        info['n_coresponders'] = int(feat.coresponder.sum())
    else:
        feat = pd.read_csv(feat_p, index_col=0)
        w = rule_weights(feat, kind, a, b)
        info['weight_auc_for_coresponders'] = auc(w, feat.coresponder)
        info['weight_mean_coresponders'] = float(w[feat.coresponder].mean())
        info['weight_mean_others'] = float(w[~feat.coresponder].mean())
        info['weight_max'] = float(w.max())
        wAPLKC, wKCAPL = per_claw_weights(kws, w)
        kws.update(_wAPLKC=wAPLKC, _wKCAPL=wKCAPL, tune_apl_weights=True)
        resp_on, spk_on, resp_off, spk_off, fixed = fit_on_off(orn, kws, common)
    info['params'] = {k: (float(v) if np.ndim(v) == 0 else float(np.mean(v)))
        for k, v in fixed.items() if v is not None and not isinstance(v, pd.Series)}
    info.update(n_kcs=int(len(resp_on)), rate_on=rate(resp_on), rate_off=rate(resp_off),
        seconds=time.time() - t0)
    for cond, (r, s) in (('on', (resp_on, spk_on)), ('acute_off', (resp_off, spk_off))):
        r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
        s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
    (out_dir / f'{tag}_info.json').write_text(json.dumps(info, indent=1, default=str))
    return info


def score(out_dir: Path, res_dir: Path, variants: List[str]) -> pd.DataFrame:
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {g: H.yang_mean_rdm(g, 'cosine') for g in GENOS}
    data_bin = {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in GENOS}
    data_rates = {g: H.yang_response_fractions(g).mean() for g in GENOS}
    feat = pd.read_csv(out_dir / f'{BASE}_kc_features.csv', index_col=0)
    C = [int(x) for x in feat.index[feat.coresponder]]
    rows, saved = [], {}
    for variant in variants:
        tag = tag_of(variant)
        info_p = out_dir / f'{tag}_info.json'
        if not info_p.exists():
            continue
        info = json.loads(info_p.read_text())
        conds = {}
        for cond in ('on', 'acute_off'):
            resp = pd.read_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
            spk = pd.read_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
            conds[cond] = (resp, spk, H.model_rdms(resp, spk))
            for vn, pairs in conds[cond][2].items():
                saved[(tag, cond, vn)] = pairs
        on_r, _, on_rdm = conds['on']
        off_r, _, off_rdm = conds['acute_off']
        row = dict(variant=variant, kind=info['kind'], k=info['k'], m=info['m'],
            rate_on=info['rate_on'], rate_off=info['rate_off'],
            wAPLKC_scale=info['params'].get('wAPLKC', np.nan),
            weight_auc=info.get('weight_auc_for_coresponders', np.nan),
            weight_ratio_C_vs_others=(info.get('weight_mean_coresponders', np.nan) /
                info.get('weight_mean_others', np.nan)) if 'weight_mean_others' in info else np.nan,
            weight_max=info.get('weight_max', np.nan), seconds=info['seconds'])
        row.update({f'{k}_on': v for k, v in kc_type_rates(on_r).items()})
        row.update(set_stats(on_r, C, 'C_on'))
        st, tabs = H.compare_on_off(on_rdm[H.PRIMARY_MODEL_RDM], off_rdm[H.PRIMARY_MODEL_RDM],
            orn=orn_pairs, drive=drive)
        for kk in ('mean_on', 'mean_off', 'spearman_on_off', 'iso_r2', 'block_fruity_small_iso',
                'block_other_between_iso', 'spearman_per_odor_iso_vs_drive', 'spearman_orn_on',
                'spearman_orn_off', 'gap_closed'):
            row[kk] = st[kk]
        tabs['blocks'].to_csv(res_dir / f'blocks_{tag}.csv')
        st_b, _ = H.compare_on_off(on_rdm['responses_cosine'], off_rdm['responses_cosine'])
        row['block_fruity_small_iso_binary'] = st_b['block_fruity_small_iso']
        for cond, rdm, geno, r in (('on', on_rdm, CTRL, on_r), ('off', off_rdm, TNT, off_r)):
            c = H.compare_model_to_data(rdm[H.PRIMARY_MODEL_RDM], data[geno], orn=orn_pairs)
            row[f'spearman_vs_{geno}_{cond}'] = c['spearman']
            row[f'spearman_dev_vs_{geno}_{cond}'] = c['spearman_dev']
            cb = H.compare_model_to_data(rdm['responses_cosine'], data_bin[geno], orn=orn_pairs)
            row[f'spearman_vs_{geno}_{cond}_binary'] = cb['spearman']
            rr = H.compare_to_references(rdm[H.PRIMARY_MODEL_RDM], ref)
            for kk in ('spearman_resid_vs_remap_wt', 'spearman_resid_vs_vcf_hyp',
                    'spearman_kc_claw', 'spearman_kc_soma'):
                row[f'{kk}_{cond}'] = rr[kk]
            row.update({f'{kk}_{cond}': v for kk, v in excess_profile(r).items()})
            fr = H.response_fractions(r)
            row[f'per_odor_rate_spearman_vs_{geno}_{cond}'] = H.spearman(fr,
                data_rates[geno].reindex(fr.index))
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(res_dir / 'E15_summary.csv', index=False)
    if saved:
        pd.concat(saved, names=['tag', 'condition', 'variant', 'odor_a', 'odor_b']
            ).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False)
    feat.to_csv(res_dir / 'kc_features_base.csv')
    return tab


def make_figure(res_dir: Path, tab: pd.DataFrame) -> None:
    plt = H._plt()
    plt.rcParams.update({'font.size': 7})
    dx = pd.read_csv(H.results_dir('E11_coincidence_claw_inhibition') / 'data_excess_profile.csv',
        index_col=0)
    cols = [('weight_auc', 'rule weight AUC for oracle set', None),
        ('C_on_frac_still_coresponding', 'oracle set still co-responding (on)', None),
        ('block_fruity_small_iso', 'block (iso resid)', 0.055),
        ('excess_block_on', 'excess: fruity x small (on)', dx.loc[CTRL, 'excess_block']),
        ('excess_other_between_on', 'excess: other between (on)', dx.loc[CTRL, 'excess_other_between']),
        ('iso_r2', 'iso R2 on/off', 0.939), ('spearman_on_off', 'Spearman on/off', 0.869),
        (f'spearman_vs_{CTRL}_on', 'Spearman vs ctrl (on)', None),
        (f'spearman_dev_vs_{CTRL}_on', 'dev vs ctrl (on)', None),
        ('spearman_resid_vs_remap_wt_on', 'dev66 vs WT (on)', 0.83),
        ('spearman_resid_vs_vcf_hyp_on', 'resid vs VCF (on)', 0.44),
        (f'per_odor_rate_spearman_vs_{CTRL}_on', 'per-odor rates vs ctrl (on)', None)]
    n = len(cols)
    fig, axs = plt.subplots(1, n, figsize=(1.45 * n + 2.4, 0.32 * len(tab) + 1.4), sharey=True)
    y = np.arange(len(tab))[::-1]
    for ax, (col, lab, refv) in zip(axs, cols):
        if col not in tab:
            ax.set_visible(False)
            continue
        ax.scatter(tab[col], y, s=16, color='#444444', zorder=3)
        base_mask = (tab.variant == 'base').to_numpy()
        ax.scatter(tab[col][base_mask], y[base_mask], s=30, facecolors='none',
            edgecolors=H.CONDITION_COLORS['control'], zorder=4)
        if refv is not None:
            ax.axvline(refv, color=H.CONDITION_COLORS['TNT'], lw=1, ls='--')
        if col.startswith('excess_') and col.endswith('_on'):
            ax.axvline(dx.loc[TNT, col[:-3]], color=H.CONDITION_COLORS['TNT'], lw=1, ls=':')
        ax.set_title(lab, fontsize=7)
        ax.grid(axis='x', color='#eeeeee', lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(tab.variant, fontsize=7)
    fig.suptitle('E15: per-KC APL weight = (1 + breadth)^k (1 + shared-glomerulus claws)^m, or '
        '(1 + APL-off spikes)^q (self1 / self2); coincidence-3 claw-dynamics base (circled); red '
        'dashed = control data, dotted = TNT (excess)', y=1.03)
    fig.savefig(res_dir / 'E15_summary.png', bbox_inches='tight', dpi=150)
    fig.savefig(res_dir / 'E15_summary.pdf', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--timeout-min', type=float, default=45)
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--run-variant', metavar='VARIANT')
    args = ap.parse_args()
    out_dir = H.outputs_dir(EXP)
    res_dir = H.results_dir(EXP)
    if args.run_variant:
        info = run_variant(args.run_variant, out_dir)
        print(json.dumps({k: v for k, v in info.items() if k != 'params'}, default=str))
        return
    variants = ['base'] + [v for v in args.variants if v != 'base']
    status_p = res_dir / 'run_status.csv'
    status = pd.read_csv(status_p).to_dict('records') if status_p.exists() else []
    if not args.score_only:
        for variant in variants:
            tag = tag_of(variant)
            if (out_dir / f'{tag}_info.json').exists():
                continue
            t0 = time.time()
            cmd = [sys.executable, str(Path(__file__).resolve()), '--run-variant', variant]
            print(f'[{datetime.now():%H:%M:%S}] running {tag} ...', flush=True)
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                    timeout=args.timeout_min * 60, cwd=H.HERE)
                ok = p.returncode == 0 and (out_dir / f'{tag}_info.json').exists()
                st = 'ok' if ok else f'failed (rc {p.returncode})'
                err_tail = '\n'.join((p.stderr or '').strip().splitlines()[-12:])
            except subprocess.TimeoutExpired:
                st, err_tail = 'timeout', ''
            status = [s for s in status if s['variant'] != variant]
            status.append(dict(variant=variant, status=st, seconds=time.time() - t0,
                when=datetime.now(timezone.utc).isoformat(timespec='seconds'), stderr_tail=err_tail))
            pd.DataFrame(status).to_csv(status_p, index=False)
            print(f'  -> {st} ({time.time() - t0:.0f} s)' + ('' if st == 'ok' else f'\n{err_tail}'),
                flush=True)
            if variant == 'base' and st != 'ok':
                print('base failed; stopping', flush=True)
                break
    tab = score(out_dir, res_dir, variants)
    pd.set_option('display.width', 250)
    show = ['variant', 'rate_on', 'wAPLKC_scale', 'weight_auc', 'weight_ratio_C_vs_others',
        'C_on_frac_still_coresponding', 'spearman_on_off', 'iso_r2', 'block_fruity_small_iso',
        'excess_block_on', 'excess_other_between_on', 'excess_within_on', f'spearman_vs_{CTRL}_on',
        f'spearman_dev_vs_{CTRL}_on', 'spearman_resid_vs_remap_wt_on', 'spearman_resid_vs_vcf_hyp_on',
        f'per_odor_rate_spearman_vs_{CTRL}_on']
    print(tab[[c for c in show if c in tab]].round(3).to_string())
    if len(tab):
        make_figure(res_dir, tab)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E15_breadth_shared_apl.py', host=os.uname().nodename,
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=Path('~/src/olfsysm').expanduser(), capture_output=True, text=True).stdout.strip(),
        variants=VARIANTS, base_kws={k: v for k, v in BASE_KWS.items()},
        approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str))
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
