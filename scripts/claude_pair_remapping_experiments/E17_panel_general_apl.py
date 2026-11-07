#!/usr/bin/env python3
"""E17: panel-general shared-input APL weighting, and the anatomical check (non-circularity
test of E15).

E15's rule reproduces the block, but its shared-claw term used eight glomeruli chosen from the
two ester groups themselves. Here the per-glomerulus score is defined without reference to the
block, and the same per-KC rule as E15 is applied:

    S_i = sum over KC i's claws of score(glomerulus of the claw)
    w_i = (1 + APL-off breadth_i)^k * (1 + S_i)^m,   k in {0, 2}, m in {1, 2}

with scores (each rescaled so its mean over claws equals the mean of E15's binary
shared-drive membership, i.e. the fraction of claws from the 8 glomeruli):
- `codrive`: the glomerulus's summed pairwise minimum of max-normalized rectified ORN drive
  over ALL 120 odor pairs of the panel (how often it is co-driven by two different odors)
- `gbreadth`: the glomerulus's tuning breadth (panel odors with normalized drive > 0.5)
- `permuted`: `codrive` randomly permuted across glomeruli (control; seed 0)
- `blockset`: E15's binary membership (reference; k = 0 rows reproduce E15's k0_m1 / k0_m2)
Glomeruli without ORN input in the model get score 0.

Anatomy (no simulation): Spearman across claws of the hemibrain APL->claw and claw->APL
synapse counts with each glomerulus score, and the per-glomerulus mean synapse counts.

The base model (coincidence-3 claw-dynamics at 5 %) and the oracle set are E15's; its base
outputs are copied so E15's scoring applies unchanged. Outputs: compact results to
<shared>/results/E17_panel_general_apl/, per-KC outputs to ./outputs/E17_panel_general_apl/.

Usage: OMP_NUM_THREADS=6 nice -n 10 ./venv/bin/python E17_panel_general_apl.py [--score-only]
"""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H
import E15_breadth_shared_apl as E15
from E11_coincidence_claw_inhibition import shared_drive_glomeruli, FRUITY, SMALL
from E14_oracle_coresponders import fit_on_off, KC_ID


EXP = 'E17_panel_general_apl'
BASE = E15.BASE
TARGET = E15.TARGET
BASE_KWS = E15.BASE_KWS
SCORES = ('codrive', 'gbreadth', 'permuted', 'blockset')
KS = (0, 2)
MS = (1, 2)
VARIANTS: Dict[str, Tuple[str, float, float]] = {'base': ('base', 0, 0)}
for sc, k, m in itertools.product(SCORES, KS, MS):
    VARIANTS[f'{sc}_k{k}_m{m}'] = (sc, k, m)


def tag_of(variant: str) -> str:
    return f'{BASE}_sp{TARGET:g}_{variant}'


def glomerulus_scores(orn: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per-glomerulus scores (rows = input glomeruli): codrive, gbreadth, permuted, blockset."""
    x = H.abbrev_columns(orn).clip(lower=0)
    x = x / x.max(axis='index').replace(0, np.nan)
    x = x.fillna(0)
    odors = list(x.columns)
    co = pd.Series(0.0, index=x.index)
    for i, a in enumerate(odors):
        for b in odors[i + 1:]:
            co = co + np.minimum(x[a], x[b])
    gb = (x > 0.5).sum(axis='columns').astype(float)
    rng = np.random.default_rng(seed)
    perm = pd.Series(rng.permutation(co.to_numpy()), index=co.index)
    block = pd.Series(0.0, index=x.index)
    block[shared_drive_glomeruli(orn).index] = 1.0
    return pd.DataFrame({'codrive': co, 'gbreadth': gb, 'permuted': perm, 'blockset': block})


def claw_table(kws: dict) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Per-claw glomerulus plus hemibrain APL<->claw synapse counts (index = wPNKC rows)."""
    from al_analysis.mb_model import (connectome_wPNKC, connectome_APL_weights,
        get_connectome_wPNKC_params)

    params = get_connectome_wPNKC_params(kws)
    wPNKC = connectome_wPNKC(**params)
    wAPLKC, wKCAPL, _, _ = connectome_APL_weights(wPNKC=wPNKC, **params)
    claw_glom = wPNKC.idxmax(axis='columns')
    claw_glom = pd.Series([g if isinstance(g, str) else g[0] for g in claw_glom], index=wPNKC.index)
    return wPNKC, claw_glom, wAPLKC.reindex(wPNKC.index), wKCAPL.reindex(wPNKC.index)


def per_kc_scores(claw_glom: pd.Series, scores: pd.DataFrame) -> pd.DataFrame:
    """Per-KC summed claw scores (each score rescaled to the blockset's mean over claws)."""
    kc_ids = claw_glom.index.get_level_values(KC_ID)
    out = {}
    target_mean = float(claw_glom.isin(scores.index[scores.blockset > 0]).mean())
    for name in scores.columns:
        per_claw = claw_glom.map(scores[name]).fillna(0.0).astype(float)
        if name != 'blockset':
            per_claw = per_claw * (target_mean / per_claw.mean())
        out[name] = pd.Series(per_claw.to_numpy(), index=kc_ids).groupby(level=0).sum()
    return pd.DataFrame(out)


def anatomy_check(kws: dict, scores: pd.DataFrame, res_dir: Path) -> pd.DataFrame:
    wPNKC, claw_glom, wAPLKC, wKCAPL = claw_table(kws)
    df = pd.DataFrame({'glomerulus': claw_glom.to_numpy(), 'aplkc_syn': wAPLKC.to_numpy(),
        'kcapl_syn': wKCAPL.to_numpy()})
    rows = []
    for name in scores.columns:
        sc = df.glomerulus.map(scores[name])
        ok = sc.notna()
        for syn in ('aplkc_syn', 'kcapl_syn'):
            r, pv = stats.spearmanr(sc[ok], df.loc[ok, syn])
            rows.append(dict(score=name, synapses=syn, level='claw', n=int(ok.sum()), spearman=r, p=pv))
        g = df[ok].groupby('glomerulus')[['aplkc_syn', 'kcapl_syn']].mean()
        g['score'] = scores.loc[g.index, name]
        for syn in ('aplkc_syn', 'kcapl_syn'):
            r, pv = stats.spearmanr(g['score'], g[syn])
            rows.append(dict(score=name, synapses=syn, level='glomerulus', n=len(g), spearman=r, p=pv))
    tab = pd.DataFrame(rows)
    tab.to_csv(res_dir / 'anatomy_apl_synapses_vs_glomerulus_scores.csv', index=False)
    per_g = df.groupby('glomerulus')[['aplkc_syn', 'kcapl_syn']].agg(['mean', 'size'])
    per_g.columns = ['aplkc_syn_mean', 'n_claws', 'kcapl_syn_mean', 'n_claws2']
    per_g = per_g.drop(columns='n_claws2').join(scores, how='left')
    per_g.to_csv(res_dir / 'anatomy_per_glomerulus.csv')
    return tab


def run_variant(variant: str, out_dir: Path) -> dict:
    t0 = time.time()
    orn = H.apl16_orn_deltas()
    kws = dict(BASE_KWS)
    common = dict(make_plots=False, silent=True)
    kind, k, m = VARIANTS[variant]
    tag = tag_of(variant)
    info: dict = dict(base=BASE, variant=variant, kind=kind, k=k, m=m, tag=tag)
    feat_p = out_dir / f'{BASE}_kc_features.csv'
    if variant == 'base':
        # reuse E15's deterministic base (same model, same seed) when available
        src = H.outputs_dir('E15_breadth_shared_apl')
        src_tag = E15.tag_of('base')
        if (src / f'{src_tag}_info.json').exists():
            for cond in ('on', 'acute_off'):
                for what in ('responses', 'spike_counts'):
                    shutil.copy(src / f'{src_tag}_{cond}_{what}.parquet',
                        out_dir / f'{tag}_{cond}_{what}.parquet')
            shutil.copy(src / f'{BASE}_kc_features.csv', feat_p)
            info = json.loads((src / f'{src_tag}_info.json').read_text())
            info.update(variant='base', tag=tag, copied_from='E15')
            (out_dir / f'{tag}_info.json').write_text(json.dumps(info, indent=1, default=str))
            return info
        resp_on, spk_on, resp_off, spk_off, fixed = fit_on_off(orn, kws, common)
        feat = E15.kc_features(resp_off, spk_off, kws)
        feat.to_csv(feat_p)
    else:
        feat = pd.read_csv(feat_p, index_col=0)
        scores = glomerulus_scores(orn)
        _, claw_glom, _, _ = claw_table(kws)
        S = per_kc_scores(claw_glom, scores)[kind].reindex(feat.index).fillna(0.0)
        w = (1 + feat.breadth_off) ** k * (1 + S) ** m
        w = w / w.mean()
        info['weight_auc_for_coresponders'] = E15.auc(w, feat.coresponder)
        info['weight_mean_coresponders'] = float(w[feat.coresponder].mean())
        info['weight_mean_others'] = float(w[~feat.coresponder].mean())
        info['weight_max'] = float(w.max())
        info['score_auc_for_coresponders'] = E15.auc(S, feat.coresponder)
        wAPLKC, wKCAPL = E15.per_claw_weights(kws, w)
        kws.update(_wAPLKC=wAPLKC, _wKCAPL=wKCAPL, tune_apl_weights=True)
        resp_on, spk_on, resp_off, spk_off, fixed = fit_on_off(orn, kws, common)
    info['params'] = {kk: (float(v) if np.ndim(v) == 0 else float(np.mean(v)))
        for kk, v in fixed.items() if v is not None and not isinstance(v, pd.Series)}
    info.update(n_kcs=int(len(resp_on)), rate_on=E15.rate(resp_on), rate_off=E15.rate(resp_off),
        seconds=time.time() - t0)
    for cond, (r, s) in (('on', (resp_on, spk_on)), ('acute_off', (resp_off, spk_off))):
        r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
        s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
    (out_dir / f'{tag}_info.json').write_text(json.dumps(info, indent=1, default=str))
    return info


def make_figure(res_dir: Path, tab: pd.DataFrame) -> None:
    """E15's dot plot with E17's title and file names."""
    import matplotlib.pyplot as plt
    orig = plt.Figure.suptitle

    def retitle(self, t, *a, **k):
        return orig(self, 'E17: per-KC APL weight = (1 + breadth)^k (1 + summed glomerulus '
            'score of claws)^m; scores: all-pairs co-drive, glomerulus breadth, permuted '
            'control, block-specific set (= E15); coincidence-3 base (circled); red dashed = '
            'control data, dotted = TNT (excess)', *a, **k)

    plt.Figure.suptitle = retitle
    try:
        E15.make_figure(res_dir, tab)
    finally:
        plt.Figure.suptitle = orig
    for ext in ('png', 'pdf'):
        src = res_dir / f'E15_summary.{ext}'
        if src.exists():
            src.rename(res_dir / f'E17_summary.{ext}')


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
        print(json.dumps({kk: v for kk, v in info.items() if kk != 'params'}, default=str))
        return
    orn = H.apl16_orn_deltas()
    scores = glomerulus_scores(orn)
    scores.to_csv(res_dir / 'glomerulus_scores.csv')
    anat = anatomy_check(BASE_KWS, scores, res_dir)
    pd.set_option('display.width', 250)
    print('anatomy: Spearman of hemibrain APL<->claw synapse counts with glomerulus scores')
    print(anat.round(3).to_string())
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
    # E15's scorer works on this directory layout (same base tag scheme and feature file)
    E15.BASE = BASE
    tab = E15.score(out_dir, res_dir, variants)
    tab = tab.rename(columns={})
    tab.to_csv(res_dir / 'E17_summary.csv', index=False)
    if (res_dir / 'E15_summary.csv').exists():
        (res_dir / 'E15_summary.csv').unlink()
    # score AUCs (per-KC summed glomerulus score alone)
    extra = {}
    for v in variants:
        ip = out_dir / f'{tag_of(v)}_info.json'
        if ip.exists():
            extra[v] = json.loads(ip.read_text()).get('score_auc_for_coresponders', np.nan)
    tab['score_auc'] = tab.variant.map(extra)
    tab.to_csv(res_dir / 'E17_summary.csv', index=False)
    show = ['variant', 'rate_on', 'score_auc', 'weight_auc', 'C_on_frac_still_coresponding',
        'spearman_on_off', 'iso_r2', 'block_fruity_small_iso', 'excess_block_on',
        'excess_other_between_on', 'excess_within_on', f'spearman_vs_{E15.CTRL}_on',
        f'spearman_dev_vs_{E15.CTRL}_on', 'spearman_resid_vs_remap_wt_on',
        'spearman_resid_vs_vcf_hyp_on', f'per_odor_rate_spearman_vs_{E15.CTRL}_on']
    print(tab[[c for c in show if c in tab]].round(3).to_string())
    if len(tab):
        make_figure(res_dir, tab)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E17_panel_general_apl.py', host=os.uname().nodename,
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=H.olfsysm_dir(), capture_output=True, text=True).stdout.strip(),
        variants=VARIANTS, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str))
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
