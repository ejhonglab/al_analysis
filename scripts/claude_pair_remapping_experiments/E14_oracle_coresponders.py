#!/usr/bin/env python3
"""E14: oracle co-responder suppression, a ceiling test for the fruity x small-ester block.

E11 (claw-targeted APL), E12 (breadth-weighted APL, k <= 1) and E13 (bouton base +
coincidence) all leave the model's fruity-ester x small-ester co-responders in place (APL-on
excess shared-responder distance -0.18 to -0.42 vs -0.05 / -0.08 in the control data). Before
building more mechanisms: is removing exactly those KCs even sufficient to produce the block
without breaking the other pairs and the per-odor rates?

Per base model: tune with APL on (5 %), run the acute APL-off model, and define the
co-responders C as the KCs that respond (APL off) to at least one fruity ester {2h, IaA, pa}
AND at least one small ester {2-but, eb, ep}. Variants (APL weights re-tuned to 5 % in each):
- `f2`, `f4`, `f8`, `f1000`: connectome APL->KC weights of C's claws multiplied by f (after
  olfsysm's mean normalization and re-tuning, f = 1000 means the APL acts on C only)
- `random_f4`, `random_f1000`: the same for an equal number of random non-C KCs (control)
- `othergroup_f4`: co-responders defined on alcohols {1-5ol, 1-6ol, 1-8ol} x aldehydes
  {6al, t2h} instead (control for "any co-responder set")
Each variant: APL on and acute off (same thresholds, wAPLKC = 0). Bases: claw-dynamics 5 %
with and without the coincidence-3 rule (E11's best).

Scoring: as E11 (harness statistics, E7's density-null excess profile on binarized
responses), plus the APL-on response rate of the C KCs and the fraction of them still
co-responding (did the manipulation remove them?).

Each variant runs in its own subprocess; `base` runs first and writes the KC sets. Outputs:
compact results to <shared>/results/E14_oracle_coresponders/, per-KC outputs to
./outputs/E14_oracle_coresponders/.

Usage: OMP_NUM_THREADS=6 nice -n 10 ./venv/bin/python E14_oracle_coresponders.py
    [--bases ...] [--variants ...] [--score-only]
"""
import argparse
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
from E5_kc_heterogeneity import BASES, rate, kc_type_rates, TUNING_VARIANT_KWS
from E11_coincidence_claw_inhibition import excess_profile


EXP = 'E14_oracle_coresponders'
TARGET = 0.05
GENOS = ('TNT3c', 'TNT')
CTRL, TNT = GENOS
FRUITY = ('2h', 'IaA', 'pa')
SMALL = ('2-but', 'eb', 'ep')
ALC = ('1-5ol', '1-6ol', '1-8ol')
ALD = ('6al', 't2h')
KC_ID = 'kc_id'

BASES14: Dict[str, dict] = {
    'claw-dynamics': dict(BASES['claw-dynamics']),
    'claw-dynamics_coinc3': dict(BASES['claw-dynamics'], n_claws_active_to_spike=3),
}
# variant -> (kc set name, factor)
VARIANTS: Dict[str, Tuple[Optional[str], Optional[float]]] = {
    'base': (None, None),
    'f2': ('coresponders', 2.0),
    'f4': ('coresponders', 4.0),
    'f8': ('coresponders', 8.0),
    'f1000': ('coresponders', 1000.0),
    'random_f4': ('random', 4.0),
    'random_f1000': ('random', 1000.0),
    'othergroup_f4': ('othergroup', 4.0),
}


def tag_of(base: str, variant: str) -> str:
    return f'{base}_sp{TARGET:g}_{variant}'


def coresponders(responses: pd.DataFrame, group_a, group_b) -> List[int]:
    r = H.abbrev_columns(responses.astype(float))
    ids = r.index.get_level_values(KC_ID)
    mask = (r[list(group_a)].sum(axis='columns') > 0) & (r[list(group_b)].sum(axis='columns') > 0)
    return sorted(int(x) for x in ids[mask.to_numpy()])


def per_kc_apl_weights(kws: dict, kc_ids: List[int], factor: float
    ) -> Tuple[pd.Series, pd.Series]:
    from al_analysis.mb_model import (connectome_wPNKC, connectome_APL_weights,
        get_connectome_wPNKC_params)

    params = get_connectome_wPNKC_params(kws)
    wPNKC = connectome_wPNKC(**params)
    wAPLKC, wKCAPL, _, _ = connectome_APL_weights(wPNKC=wPNKC, **params)
    claw_kc = wAPLKC.index.get_level_values(KC_ID)
    f = pd.Series(np.where(np.isin(claw_kc, kc_ids), factor, 1.0), index=wAPLKC.index)
    return wAPLKC * f, wKCAPL


def fit_on_off(orn, kws: dict, common: dict):
    from al_analysis.mb_model import fit_mb_model, get_thr_and_APL_weights

    resp_on, spk_on, _, pdict = fit_mb_model(orn, **kws, target_sparsity=TARGET, **common)
    fixed = get_thr_and_APL_weights(pdict, kws)
    kws_off = {k: v for k, v in kws.items() if k not in TUNING_VARIANT_KWS + ('tune_apl_weights',)}
    resp_off, spk_off, _, _ = fit_mb_model(orn, **kws_off, **dict(fixed, wAPLKC=0.0), **common)
    return resp_on, spk_on, resp_off, spk_off, fixed


def run_variant(base: str, variant: str, out_dir: Path) -> dict:
    t0 = time.time()
    orn = H.apl16_orn_deltas()
    kws = dict(BASES14[base])
    common = dict(make_plots=False, silent=True)
    tag = tag_of(base, variant)
    set_name, factor = VARIANTS[variant]
    info: dict = dict(base=base, variant=variant, target=TARGET, tag=tag, kc_set=set_name,
        factor=factor)
    sets_p = out_dir / f'{base}_kc_sets.json'
    if variant == 'base':
        resp_on, spk_on, resp_off, spk_off, fixed = fit_on_off(orn, kws, common)
        C = coresponders(resp_off, FRUITY, SMALL)
        other = coresponders(resp_off, ALC, ALD)
        all_ids = sorted(int(x) for x in resp_off.index.get_level_values(KC_ID))
        rng = np.random.default_rng(0)
        non_c = [k for k in all_ids if k not in set(C)]
        rand = sorted(int(x) for x in rng.choice(non_c, size=len(C), replace=False))
        sets = dict(coresponders=C, random=rand, othergroup=other, n_kcs=len(all_ids))
        sets_p.write_text(json.dumps(sets))
        info.update(n_coresponders=len(C), n_othergroup=len(other),
            frac_coresponders=len(C) / len(all_ids))
    else:
        sets = json.loads(sets_p.read_text())
        kc_ids = sets[set_name]
        wAPLKC, wKCAPL = per_kc_apl_weights(kws, kc_ids, factor)
        kws.update(_wAPLKC=wAPLKC, _wKCAPL=wKCAPL, tune_apl_weights=True)
        resp_on, spk_on, resp_off, spk_off, fixed = fit_on_off(orn, kws, common)
        info.update(n_kcs_weighted=len(kc_ids),
            frac_claws_weighted=float(np.isin(wAPLKC.index.get_level_values(KC_ID), kc_ids).mean()))
    info['params'] = {k: (float(v) if np.ndim(v) == 0 else float(np.mean(v)))
        for k, v in fixed.items() if v is not None and not isinstance(v, pd.Series)}
    info.update(n_kcs=int(len(resp_on)), rate_on=rate(resp_on), rate_off=rate(resp_off),
        seconds=time.time() - t0)
    for cond, (r, s) in (('on', (resp_on, spk_on)), ('acute_off', (resp_off, spk_off))):
        r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
        s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
    (out_dir / f'{tag}_info.json').write_text(json.dumps(info, indent=1, default=str))
    return info


# ---------------------------------------------------------------------------
def set_stats(resp: pd.DataFrame, kc_ids: List[int], prefix: str) -> Dict[str, float]:
    r = H.abbrev_columns(resp.astype(float))
    ids = r.index.get_level_values(KC_ID)
    sub = r[np.isin(ids, kc_ids)]
    both = (sub[list(FRUITY)].sum(axis='columns') > 0) & (sub[list(SMALL)].sum(axis='columns') > 0)
    rest = r[~np.isin(ids, kc_ids)]
    return {f'{prefix}_rate': float(sub.mean().mean()), f'{prefix}_frac_still_coresponding':
        float(both.mean()), f'{prefix}_rest_rate': float(rest.mean().mean()),
        f'{prefix}_n_coresponders_total': int(((r[list(FRUITY)].sum(axis='columns') > 0) &
            (r[list(SMALL)].sum(axis='columns') > 0)).sum())}


def score(out_dir: Path, res_dir: Path, bases: List[str], variants: List[str]) -> pd.DataFrame:
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {g: H.yang_mean_rdm(g, 'cosine') for g in GENOS}
    data_bin = {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in GENOS}
    data_rates = {g: H.yang_response_fractions(g).mean() for g in GENOS}
    rows, saved = [], {}
    for base in bases:
        sets_p = out_dir / f'{base}_kc_sets.json'
        if not sets_p.exists():
            continue
        sets = json.loads(sets_p.read_text())
        C = sets['coresponders']
        for variant in variants:
            tag = tag_of(base, variant)
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
            row = dict(base=base, variant=variant, kc_set=info['kc_set'], factor=info['factor'],
                n_coresponders=len(C), frac_coresponders=len(C) / sets['n_kcs'],
                rate_on=info['rate_on'], rate_off=info['rate_off'],
                wAPLKC_scale=info['params'].get('wAPLKC', np.nan), seconds=info['seconds'])
            row.update({f'{k}_on': v for k, v in kc_type_rates(on_r).items()})
            row.update(set_stats(on_r, C, 'C_on'))
            row.update(set_stats(off_r, C, 'C_off'))
            st, tabs = H.compare_on_off(on_rdm[H.PRIMARY_MODEL_RDM], off_rdm[H.PRIMARY_MODEL_RDM],
                orn=orn_pairs, drive=drive)
            for k in ('mean_on', 'mean_off', 'spearman_on_off', 'iso_r2', 'block_fruity_small_iso',
                    'block_other_between_iso', 'spearman_per_odor_iso_vs_drive', 'spearman_orn_on',
                    'spearman_orn_off', 'gap_closed'):
                row[k] = st[k]
            tabs['blocks'].to_csv(res_dir / f'blocks_{tag}.csv')
            st_b, _ = H.compare_on_off(on_rdm['responses_cosine'], off_rdm['responses_cosine'])
            row['block_fruity_small_iso_binary'] = st_b['block_fruity_small_iso']
            row['spearman_on_off_binary'] = st_b['spearman_on_off']
            for cond, rdm, geno, r in (('on', on_rdm, CTRL, on_r), ('off', off_rdm, TNT, off_r)):
                c = H.compare_model_to_data(rdm[H.PRIMARY_MODEL_RDM], data[geno], orn=orn_pairs)
                row[f'spearman_vs_{geno}_{cond}'] = c['spearman']
                row[f'spearman_dev_vs_{geno}_{cond}'] = c['spearman_dev']
                cb = H.compare_model_to_data(rdm['responses_cosine'], data_bin[geno], orn=orn_pairs)
                row[f'spearman_vs_{geno}_{cond}_binary'] = cb['spearman']
                rr = H.compare_to_references(rdm[H.PRIMARY_MODEL_RDM], ref)
                for k in ('spearman_resid_vs_remap_wt', 'spearman_resid_vs_vcf_hyp',
                        'spearman_kc_claw', 'spearman_kc_soma'):
                    row[f'{k}_{cond}'] = rr[k]
                row.update({f'{k}_{cond}': v for k, v in excess_profile(r).items()})
                fr = H.response_fractions(r)
                row[f'per_odor_rate_spearman_vs_{geno}_{cond}'] = H.spearman(fr,
                    data_rates[geno].reindex(fr.index))
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(res_dir / 'E14_summary.csv', index=False)
    if saved:
        pd.concat(saved, names=['tag', 'condition', 'variant', 'odor_a', 'odor_b']
            ).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False)
    return tab


def make_figure(res_dir: Path, tab: pd.DataFrame) -> None:
    plt = H._plt()
    plt.rcParams.update({'font.size': 7})
    dx = pd.read_csv(H.results_dir('E11_coincidence_claw_inhibition') / 'data_excess_profile.csv',
        index_col=0)
    cols = [('rate_on', 'rate on', None), ('C_on_rate', 'co-responders: rate on', None),
        ('C_on_frac_still_coresponding', 'co-responders still co-responding (on)', None),
        ('spearman_on_off', 'Spearman on/off', 0.869), ('iso_r2', 'iso R2 on/off', 0.939),
        ('block_fruity_small_iso', 'block (iso resid)', 0.055),
        ('excess_block_on', 'excess: fruity x small (on)', dx.loc[CTRL, 'excess_block']),
        ('excess_other_between_on', 'excess: other between (on)', dx.loc[CTRL, 'excess_other_between']),
        ('excess_within_on', 'excess: within (on)', dx.loc[CTRL, 'excess_within']),
        (f'spearman_vs_{CTRL}_on', 'Spearman vs ctrl (on)', None),
        (f'spearman_dev_vs_{CTRL}_on', 'dev vs ctrl (on)', None),
        ('spearman_resid_vs_remap_wt_on', 'dev66 vs WT (on)', 0.83),
        (f'per_odor_rate_spearman_vs_{CTRL}_on', 'per-odor rates vs ctrl (on)', None)]
    t = tab.copy()
    t['label'] = t.base + ' / ' + t.variant
    n = len(cols)
    fig, axs = plt.subplots(1, n, figsize=(1.45 * n + 2.6, 0.32 * len(t) + 1.4), sharey=True)
    y = np.arange(len(t))[::-1]
    for ax, (col, lab, refv) in zip(axs, cols):
        if col not in t:
            ax.set_visible(False)
            continue
        ax.scatter(t[col], y, s=16, color='#444444', zorder=3)
        base_mask = (t.variant == 'base').to_numpy()
        ax.scatter(t[col][base_mask], y[base_mask], s=30, facecolors='none',
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
    axs[0].set_yticklabels(t.label, fontsize=7)
    fig.suptitle('E14: oracle suppression of the APL-off co-responders (circled = base; red dashed = '
        'control data, dotted = TNT data for the excess profile)', y=1.03)
    fig.savefig(res_dir / 'E14_summary.png', bbox_inches='tight', dpi=150)
    fig.savefig(res_dir / 'E14_summary.pdf', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bases', nargs='+', default=list(BASES14), choices=list(BASES14))
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--timeout-min', type=float, default=45)
    ap.add_argument('--score-only', action='store_true')
    ap.add_argument('--run-variant', nargs=2, metavar=('BASE', 'VARIANT'))
    args = ap.parse_args()
    out_dir = H.outputs_dir(EXP)
    res_dir = H.results_dir(EXP)
    if args.run_variant:
        info = run_variant(args.run_variant[0], args.run_variant[1], out_dir)
        print(json.dumps({k: v for k, v in info.items() if k != 'params'}, default=str))
        return

    variants = ['base'] + [v for v in args.variants if v != 'base']
    status_p = res_dir / 'run_status.csv'
    status = pd.read_csv(status_p).to_dict('records') if status_p.exists() else []
    if not args.score_only:
        for base in args.bases:
            for variant in variants:
                tag = tag_of(base, variant)
                if (out_dir / f'{tag}_info.json').exists():
                    continue
                t0 = time.time()
                cmd = [sys.executable, str(Path(__file__).resolve()), '--run-variant', base, variant]
                print(f'[{datetime.now():%H:%M:%S}] running {tag} ...', flush=True)
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True,
                        timeout=args.timeout_min * 60, cwd=H.HERE)
                    ok = p.returncode == 0 and (out_dir / f'{tag}_info.json').exists()
                    st = 'ok' if ok else f'failed (rc {p.returncode})'
                    err_tail = '\n'.join((p.stderr or '').strip().splitlines()[-12:])
                except subprocess.TimeoutExpired:
                    st, err_tail = 'timeout', ''
                status = [s for s in status if not (s['base'] == base and s['variant'] == variant)]
                status.append(dict(base=base, variant=variant, status=st, seconds=time.time() - t0,
                    when=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    stderr_tail=err_tail))
                pd.DataFrame(status).to_csv(status_p, index=False)
                print(f'  -> {st} ({time.time() - t0:.0f} s)' + ('' if st == 'ok' else
                    f'\n{err_tail}'), flush=True)
                if variant == 'base' and st != 'ok':
                    print(f'  base failed for {base}; skipping its variants', flush=True)
                    break
    tab = score(out_dir, res_dir, args.bases, variants)
    pd.set_option('display.width', 250)
    show = ['base', 'variant', 'rate_on', 'wAPLKC_scale', 'C_on_rate', 'C_on_frac_still_coresponding',
        'C_on_n_coresponders_total', 'spearman_on_off', 'iso_r2', 'block_fruity_small_iso',
        'excess_block_on', 'excess_other_between_on', 'excess_within_on', f'spearman_vs_{CTRL}_on',
        f'spearman_dev_vs_{CTRL}_on', 'spearman_resid_vs_remap_wt_on',
        f'per_odor_rate_spearman_vs_{CTRL}_on']
    print(tab[[c for c in show if c in tab]].round(3).to_string())
    if len(tab):
        make_figure(res_dir, tab)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E14_oracle_coresponders.py', host=os.uname().nodename,
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=Path('~/src/olfsysm').expanduser(), capture_output=True, text=True).stdout.strip(),
        variants=VARIANTS, bases=list(BASES14), approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str))
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
