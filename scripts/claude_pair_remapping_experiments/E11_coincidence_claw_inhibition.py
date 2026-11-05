#!/usr/bin/env python3
"""E11: claw coincidence combined with claw-level inhibition.

E5 found that requiring >= 2-3 active claws for a KC to spike (`n_claws_active_to_spike`)
is the one KC-level change that moves the claw models toward the data, by removing the
model's non-specific excess shared responders (E7's density null); what remains is the
fruity-ester x small-ester shared-responder block that, in the data, APL removes. E2's APL
units acted on claws but before a coincidence test existed. E11 asks whether claw-level
inhibition applied before the coincidence test removes that block specifically:

- `coinc{2,3}`: coincidence alone (reference, as E5)
- `coinc{2,3}_units{0,1}`: coincidence + APL split into connectome spatial units
  (`apl_unit_coupling` 0 = uncoupled, 1 = coupled), each inhibiting its own claws
- `coinc{2,3}_coinput_x{2,4}`: coincidence + per-claw connectome APL<->KC weights multiplied
  by f for claws from "shared-drive" glomeruli (the glomeruli with the most ORN drive common
  to the fruity esters {2h, IaA, pa} and the small esters {2-but, eb, ep}; top 8 of 36 by
  summed pairwise min drive), i.e. inhibition targeted at the claws that carry the block
- `coinput_x{2,4}`: the targeted weights without coincidence (control)

Base: chapter-3 claw-dynamics model at 5 % (2-spike criterion). Each variant: tuned with APL
on, then acute APL off (`wAPLKC = 0`) with the same thresholds. Scored with
`kc_remapping_harness` as E5 plus E7's density-null excess-similarity profile (binarized
responses): block / other between-group / within-group excess distance vs the data
(control: -0.05 / -0.05 / -0.33; TNT: -0.19 / -0.12 / -0.38).

Each variant runs in its own subprocess (as E5). Outputs: compact results to
<shared>/results/E11_coincidence_claw_inhibition/, per-KC outputs to
./outputs/E11_coincidence_claw_inhibition/.

Usage: OMP_NUM_THREADS=6 nice -n 10 ./venv/bin/python E11_coincidence_claw_inhibition.py
    [--variants ...] [--score-only]
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


EXP = 'E11_coincidence_claw_inhibition'
BASE = 'claw-dynamics'
TARGET = 0.05
GENOS = ('TNT3c', 'TNT')
CTRL, TNT = GENOS
FRUITY = ('2h', 'IaA', 'pa')
SMALL = ('2-but', 'eb', 'ep')
N_SHARED_GLOMS = 8

# variant -> (n_claws_active_to_spike or None, apl_unit_coupling or None, coinput factor or None)
VARIANTS: Dict[str, Tuple[Optional[int], Optional[float], Optional[float]]] = {
    'coinc2': (2, None, None),
    'coinc3': (3, None, None),
    'coinc2_units0': (2, 0.0, None),
    'coinc3_units0': (3, 0.0, None),
    'coinc2_units1': (2, 1.0, None),
    'coinc3_units1': (3, 1.0, None),
    'coinc2_coinput_x2': (2, None, 2.0),
    'coinc2_coinput_x4': (2, None, 4.0),
    'coinc3_coinput_x2': (3, None, 2.0),
    'coinc3_coinput_x4': (3, None, 4.0),
    'coinput_x2': (None, None, 2.0),
    'coinput_x4': (None, None, 4.0),
}


def tag_of(variant: str) -> str:
    return f'{BASE}_sp{TARGET:g}_{variant}'


def shared_drive_glomeruli(orn: pd.DataFrame, n: int = N_SHARED_GLOMS) -> pd.Series:
    """Summed pairwise min of (per-odor max-normalized, rectified) ORN drive over fruity x
    small-ester pairs, per glomerulus; returns the top-n glomeruli (sorted, descending)."""
    x = H.abbrev_columns(orn).clip(lower=0)
    x = x / x.max(axis='index').replace(0, np.nan)
    s = pd.Series(0.0, index=x.index)
    for a in FRUITY:
        for b in SMALL:
            s = s + np.minimum(x[a], x[b])
    return s.sort_values(ascending=False).head(n)


def coinput_apl_weights(kws: dict, factor: float, gloms: List[str]
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Connectome per-claw APL<->KC weights x `factor` for claws from `gloms`."""
    from al_analysis.mb_model import (connectome_wPNKC, connectome_APL_weights,
        get_connectome_wPNKC_params)

    params = get_connectome_wPNKC_params(kws)
    wPNKC = connectome_wPNKC(**params)
    wAPLKC, wKCAPL, _, _ = connectome_APL_weights(wPNKC=wPNKC, **params)
    claw_glom = wPNKC.idxmax(axis='columns')
    claw_glom = pd.Series([g if isinstance(g, str) else g[0] for g in claw_glom],
        index=wPNKC.index)
    f = pd.Series(np.where(claw_glom.isin(gloms), factor, 1.0), index=wPNKC.index)
    f = f.reindex(wAPLKC.index).fillna(1.0)
    return wAPLKC * f, wKCAPL * f, claw_glom


def run_variant(variant: str, out_dir: Path) -> dict:
    from al_analysis.mb_model import fit_mb_model, get_thr_and_APL_weights

    t0 = time.time()
    n_coinc, coupling, factor = VARIANTS[variant]
    orn = H.apl16_orn_deltas()
    kws = dict(BASES[BASE])
    common = dict(make_plots=False, silent=True)
    info: dict = dict(base=BASE, variant=variant, target=TARGET, tag=tag_of(variant),
        n_claws_active_to_spike=n_coinc, apl_unit_coupling=coupling, coinput_factor=factor)
    if n_coinc is not None:
        kws['n_claws_active_to_spike'] = n_coinc
    if coupling is not None:
        kws['apl_unit_coupling'] = coupling
    if factor is not None:
        gloms = shared_drive_glomeruli(orn)
        wAPLKC, wKCAPL, claw_glom = coinput_apl_weights(kws, factor, list(gloms.index))
        kws.update(_wAPLKC=wAPLKC, _wKCAPL=wKCAPL, tune_apl_weights=True)
        info['shared_drive_glomeruli'] = {g: float(v) for g, v in gloms.items()}
        info['frac_claws_weighted'] = float(claw_glom.isin(gloms.index).mean())

    resp_on, spk_on, _, pdict = fit_mb_model(orn, **kws, target_sparsity=TARGET, **common)
    fixed = get_thr_and_APL_weights(pdict, kws)
    info['params'] = {k: (float(v) if np.ndim(v) == 0 else float(np.mean(v)))
        for k, v in fixed.items() if v is not None and not isinstance(v, pd.Series)}
    kws_off = {k: v for k, v in kws.items() if k not in TUNING_VARIANT_KWS + ('tune_apl_weights',)}
    fixed_off = dict(fixed, wAPLKC=0.0)
    resp_off, spk_off, _, _ = fit_mb_model(orn, **kws_off, **fixed_off, **common)

    thr_arr = np.asarray(fixed['fixed_thr'], dtype=float)
    info.update(n_kcs=int(len(resp_on)), thr_mean=float(thr_arr.mean()),
        rate_on=rate(resp_on), rate_off=rate(resp_off), seconds=time.time() - t0)
    tag = tag_of(variant)
    for cond, (r, s) in (('on', (resp_on, spk_on)), ('acute_off', (resp_off, spk_off))):
        r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
        s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
    (out_dir / f'{tag}_info.json').write_text(json.dumps(info, indent=1, default=str))
    return info


# ---------------------------------------------------------------------------
def excess_profile(resp: pd.DataFrame) -> Dict[str, float]:
    """E7 density-null excess-distance profile of binarized responses (cosine)."""
    r = H.abbrev_columns(resp.astype(float))
    obs = H.pair_distances(r, 'cosine')
    null = 1 - H.density_null_similarity(r)
    exc = obs - null
    bm = H.block_means(exc)
    return dict(excess_mean=float(exc.mean()),
        excess_block=float(bm.get(H.FRUITY_SMALL_BLOCK, np.nan)),
        excess_other_between=float(bm.drop(['within', H.FRUITY_SMALL_BLOCK],
            errors='ignore').mean()),
        excess_within=float(bm.get('within', np.nan)))


def score(out_dir: Path, res_dir: Path, variants: List[str]) -> pd.DataFrame:
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {g: H.yang_mean_rdm(g, 'cosine') for g in GENOS}
    data_bin = {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in GENOS}
    data_rates = {g: H.yang_response_fractions(g).mean() for g in GENOS}
    mats = {g: H.yang_recording_matrices(g) for g in GENOS}
    data_excess = {}
    for g in GENOS:
        obs = H.recording_rdms_from_matrices(mats[g], values='binary').mean()
        null = H.recording_density_null_rdms(mats[g], values='binary').mean()
        bm = H.block_means(obs - null)
        data_excess[g] = dict(excess_mean=float((obs - null).mean()),
            excess_block=float(bm[H.FRUITY_SMALL_BLOCK]),
            excess_other_between=float(bm.drop(['within', H.FRUITY_SMALL_BLOCK]).mean()),
            excess_within=float(bm['within']))
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
        row = dict(variant=variant, n_claws_active_to_spike=info['n_claws_active_to_spike'],
            apl_unit_coupling=info['apl_unit_coupling'], coinput_factor=info['coinput_factor'],
            rate_on=info['rate_on'], rate_off=info['rate_off'],
            wAPLKC_scale=info['params'].get('wAPLKC', np.nan), seconds=info['seconds'])
        row.update({f'{k}_on': v for k, v in kc_type_rates(on_r).items()})
        st, tabs = H.compare_on_off(on_rdm[H.PRIMARY_MODEL_RDM], off_rdm[H.PRIMARY_MODEL_RDM],
            orn=orn_pairs, drive=drive)
        for k in ('mean_on', 'mean_off', 'spearman_on_off', 'iso_r2', 'block_fruity_small_iso',
                'block_other_between_iso', 'spearman_per_odor_iso_vs_drive', 'spearman_orn_on',
                'spearman_orn_off', 'gap_closed'):
            row[k] = st[k]
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
            for k in ('spearman_resid_vs_remap_wt', 'spearman_resid_vs_vcf_hyp',
                    'spearman_kc_claw', 'spearman_kc_soma'):
                row[f'{k}_{cond}'] = rr[k]
            row.update({f'{k}_{cond}': v for k, v in excess_profile(r).items()})
            fr = H.response_fractions(r)
            row[f'per_odor_rate_spearman_vs_{geno}_{cond}'] = H.spearman(fr,
                data_rates[geno].reindex(fr.index))
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(res_dir / 'E11_summary.csv', index=False)
    pd.DataFrame(data_excess).T.to_csv(res_dir / 'data_excess_profile.csv')
    if saved:
        pd.concat(saved, names=['tag', 'condition', 'variant', 'odor_a', 'odor_b']
            ).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False)
    return tab


def make_figure(res_dir: Path, tab: pd.DataFrame) -> None:
    plt = H._plt()
    plt.rcParams.update({'font.size': 7})
    dx = pd.read_csv(res_dir / 'data_excess_profile.csv', index_col=0)
    cols = [('rate_on', 'rate on', None), ('spearman_on_off', 'Spearman on/off', 0.869),
        ('iso_r2', 'iso R2 on/off', 0.939), ('block_fruity_small_iso', 'block (iso resid)', 0.055),
        ('excess_block_on', 'excess: fruity x small (on)', dx.loc[CTRL, 'excess_block']),
        ('excess_other_between_on', 'excess: other between (on)', dx.loc[CTRL, 'excess_other_between']),
        ('excess_within_on', 'excess: within (on)', dx.loc[CTRL, 'excess_within']),
        (f'spearman_vs_{CTRL}_on', 'Spearman vs ctrl (on)', None),
        (f'spearman_dev_vs_{CTRL}_on', 'dev vs ctrl (on)', None),
        ('spearman_resid_vs_remap_wt_on', 'dev66 vs WT (on)', 0.83),
        ('spearman_resid_vs_vcf_hyp_on', 'resid vs VCF (on)', 0.44),
        ('spearman_kc_claw_on', 'Spearman claws (on)', 0.68)]
    n = len(cols)
    fig, axs = plt.subplots(1, n, figsize=(1.5 * n + 2.4, 0.32 * len(tab) + 1.4), sharey=True)
    y = np.arange(len(tab))[::-1]
    for ax, (col, lab, ref) in zip(axs, cols):
        if col not in tab:
            ax.set_visible(False)
            continue
        ax.scatter(tab[col], y, s=16, color='#444444', zorder=3)
        if ref is not None:
            ax.axvline(ref, color=H.CONDITION_COLORS['TNT'], lw=1, ls='--')
        if col.startswith('excess_') and col.endswith('_on'):
            ax.axvline(dx.loc[TNT, col[:-3]], color=H.CONDITION_COLORS['TNT'], lw=1, ls=':')
        ax.set_title(lab, fontsize=7)
        ax.grid(axis='x', color='#eeeeee', lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(tab.variant, fontsize=7)
    fig.suptitle('E11: coincidence x claw-level inhibition (claw-dynamics 5 %); red dashed = '
        'control data, dotted = TNT data (excess profile)', y=1.03)
    fig.savefig(res_dir / 'E11_summary.png', bbox_inches='tight', dpi=150)
    fig.savefig(res_dir / 'E11_summary.pdf', bbox_inches='tight')
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

    status_p = res_dir / 'run_status.csv'
    status = pd.read_csv(status_p).to_dict('records') if status_p.exists() else []
    if not args.score_only:
        for variant in args.variants:
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
                when=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                stderr_tail=err_tail))
            pd.DataFrame(status).to_csv(status_p, index=False)
            print(f'  -> {st} ({time.time() - t0:.0f} s)' + ('' if st == 'ok' else
                f'\n{err_tail}'), flush=True)
    tab = score(out_dir, res_dir, args.variants)
    pd.set_option('display.width', 250)
    show = ['variant', 'rate_on', 'rate_off', 'spearman_on_off', 'iso_r2', 'block_fruity_small_iso',
        'excess_block_on', 'excess_other_between_on', 'excess_within_on', 'excess_block_off',
        f'spearman_vs_{CTRL}_on', f'spearman_dev_vs_{CTRL}_on', 'spearman_resid_vs_remap_wt_on',
        'spearman_resid_vs_vcf_hyp_on', 'spearman_kc_claw_on']
    print(tab[[c for c in show if c in tab]].round(3).to_string())
    if len(tab):
        make_figure(res_dir, tab)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E11_coincidence_claw_inhibition.py', host=os.uname().nodename,
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=Path('~/src/olfsysm').expanduser(), capture_output=True, text=True).stdout.strip(),
        variants=VARIANTS, n_shared_gloms=N_SHARED_GLOMS, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str))
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
