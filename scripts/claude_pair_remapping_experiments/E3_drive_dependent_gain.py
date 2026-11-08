#!/usr/bin/env python3
"""E3: drive-dependent APL gain. How much odor-level drive contrast reaches APL and KCs?

Data target (report, E7): APL's extra decorrelation per odor tracks total ORN drive
(Spearman 0.50) and, about equally, the per-odor change in response density. In the model
the antennal-lobe normalization flattens PN drive almost completely (atlas, 6c60b64: mean
PN odor rate max/min 1.11 across odors vs 2.1 for ORNs), so APL in the model sees nearly
the same total drive for every odor.

Parts:
- A. `pn_inhadd` sweep (strength of divisive normalization in the PN layer; `pn_inhsc`
  scaled with it to keep the gain at a typical inhibition level roughly constant),
  thresholds and APL re-tuned at each setting, for the claw-dynamics and hemibrain wd20
  bases at 5 %; APL on and acute off. Also records the PN odor-rate contrast (max/min of
  mean PN rate during the odor, and its Spearman with ORN drive) from the run variables.
- B. claw-dynamics model without the direct PN->APL drive (`pn_claw_to_apl=False`).
- C. bouton base (PN boutons with APL<->PN weights) with the PN<->APL weight scales
  multiplied by m in {0, 0.5, 1, 2, 4}, KC<->APL scales rescaled so the number of
  responses matches the unmodified model's; APL off = both APL output weights zero.

Scored as E4 (`E4_substrate.score`), plus: Spearman of the per-odor APL effect (isotonic
residual) with the per-odor density change (rate off - rate on) and with ORN drive, and
Spearman of per-odor model response rates with the data's (on vs TNT3c, off vs TNT).

Outputs: <shared>/results/E3_drive_dependent_gain/ and ./outputs/E3_drive_dependent_gain/.

Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E3_drive_dependent_gain.py
    [--parts A B C] [--inhadd 31.4088 100 314 1000 3141 10000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H
from E4_substrate import CLAWS, score, kc_by_glomerulus

EXPERIMENT = 'E3_drive_dependent_gain'
TARGET = 0.05
PN_INHSC_DEFAULT, PN_INHADD_DEFAULT = 368.6631, 31.4088
TYPICAL_INH = 2000.0   # 0.25 inhA + 0.75 inhB during odors is ~1300-5600 (atlas)
BASES = {
    'claw-dynamics': dict(CLAWS, pn_claw_to_apl=True, claw_dynamics=True),
    'hemibrain-wd20': dict(pn2kc_connections='hemibrain', weight_divisor=20,
        _drop_glom_with_plus=False),
}
BOUTON = dict(CLAWS, prat_boutons=True, pn_claw_to_apl=True)
DEFAULT_INHADD = [PN_INHADD_DEFAULT, 100.0, 314.0, 1000.0, 3141.0, 10000.0]
PN_APL_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0, 4.0]


def rate(responses: pd.DataFrame) -> float:
    return float(responses.astype(float).mean().mean())


def pn_contrast(mp, rv, orn: pd.DataFrame) -> Dict[str, float]:
    """Mean PN rate during the odor per odor: max/min ratio and Spearman with ORN drive."""
    from al_analysis.mb_model import get_time_index, get_sim_var
    t = np.asarray(get_time_index(mp, rv))
    win = (t >= 0.0) & (t <= 0.5)
    odors = [H.odor_abbrev(c) for c in orn.columns]
    pn_sims = get_sim_var(rv.pn, 'pn_sims')   # (odors, PNs, time)
    pn_mean = np.array([np.asarray(pn_sims[i])[:, win].mean() for i in range(len(odors))])
    drive = H.orn_input_drive(orn).reindex(odors).to_numpy()
    return dict(pn_rate_max_over_min=float(pn_mean.max() / pn_mean.min()),
        pn_rate_mean=float(pn_mean.mean()),
        spearman_pn_rate_vs_orn_drive=float(stats.spearmanr(pn_mean, drive)[0]),
        pearson_pn_rate_vs_orn_drive=float(stats.pearsonr(pn_mean, drive)[0]),
    )


def per_odor_extras(on_resp: pd.DataFrame, off_resp: pd.DataFrame, on_pairs: pd.Series,
    off_pairs: pd.Series, drive: pd.Series, data_rates: Dict[str, pd.Series]
    ) -> Dict[str, float]:
    """Per-odor APL effect vs density change / drive; per-odor rates vs data."""
    r_on = H.response_fractions(on_resp)
    r_off = H.response_fractions(off_resp)
    _, tabs = H.compare_on_off(on_pairs, off_pairs, drive=drive)
    po = tabs['per_odor']
    dfrac = (r_off - r_on).reindex(po.index)
    out = dict(
        spearman_per_odor_iso_vs_dfrac=float(stats.spearmanr(po.iso_resid, dfrac)[0]),
        spearman_per_odor_iso_vs_drive=float(stats.spearmanr(po.iso_resid, po.drive)[0]),
        spearman_dfrac_vs_drive=float(stats.spearmanr(dfrac, po.drive)[0]),
        spearman_rate_on_vs_data=float(stats.spearmanr(r_on.reindex(H.APL16_ODORS),
            data_rates['TNT3c'].reindex(H.APL16_ODORS))[0]),
        spearman_rate_off_vs_data=float(stats.spearmanr(r_off.reindex(H.APL16_ODORS),
            data_rates['TNT'].reindex(H.APL16_ODORS))[0]),
        rate_ratio_max_over_min_on=float(r_on.max() / max(r_on.min(), 1e-9)),
    )
    return out


def tune_and_knockout(orn, kws, out_dir, tag, *, with_pn_contrast: bool = True,
    target: float = TARGET):
    """fit_mb_model tuned with APL on to `target` (+ PN contrast from run vars), then APL off."""
    from al_analysis.mb_model import fit_mb_model
    common = dict(kws, make_plots=False, silent=True)
    ret = fit_mb_model(orn, target_sparsity=target, return_olfsysm_vars=with_pn_contrast,
        **common
    )
    resp_on, spk_on, wPNKC, params = ret
    extra = {}
    if with_pn_contrast:
        mp = params.pop('mp'); rv = params.pop('rv')
        extra.update(pn_contrast(mp, rv, orn))
        del mp, rv
    thr = params['fixed_thr']
    off_kws = dict(common, fixed_thr=thr, wAPLKC=0.0)
    if kws.get('prat_boutons'):
        off_kws['wAPLPN'] = 0.0
        off_kws['wPNAPL'] = float(params.get('wPNAPL_scale', params.get('wPNAPL', 0.0)))
    resp_off, spk_off, _, _ = fit_mb_model(orn, **off_kws)
    for cond, (r, s) in (('on', (resp_on, spk_on)), ('acute_off', (resp_off, spk_off))):
        r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
        s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
    extra.update(fixed_thr_mean=float(np.mean(thr)),
        wAPLKC=float(np.mean(params.get('wAPLKC_scale', params.get('wAPLKC', np.nan)))),
    )
    return (resp_on, spk_on), (resp_off, spk_off), wPNKC, params, extra


def match_kc_apl_scale(sim: Callable, params: dict, n_target: int, *, tol: int = 2,
    max_evals: int = 25) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Scales wAPLKC_scale and wKCAPL_scale together so the response count matches."""
    def run(f):
        p = dict(params, wAPLKC_scale=params['wAPLKC_scale'] * f,
            wKCAPL_scale=params['wKCAPL_scale'] * f)
        r, s = sim(p)
        return int((r.astype(float) != 0).to_numpy().sum()), r, s
    lo, hi = 0.002, 20.0
    n_lo, *_ = run(lo)
    n_hi, *_ = run(hi)
    assert n_lo > n_target > n_hi, (n_lo, n_target, n_hi)
    best = None
    for _ in range(max_evals):
        f = float(np.sqrt(lo * hi))
        n, r, s = run(f)
        if best is None or abs(n - n_target) < abs(best[1] - n_target):
            best = (f, n, r, s)
        if abs(n - n_target) <= tol:
            break
        if n > n_target:
            lo = f
        else:
            hi = f
    f, n, r, s = best
    return f, r, s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--parts', nargs='+', choices=['A', 'B', 'C'], default=['A', 'B', 'C'])
    ap.add_argument('--inhadd', nargs='+', type=float, default=DEFAULT_INHADD)
    args = ap.parse_args()
    from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator

    res_dir = H.results_dir(EXPERIMENT)
    out_dir = H.outputs_dir(EXPERIMENT)
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {
        'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')},
    }
    data_rates = {g: H.yang_response_fractions(g).mean() for g in ('TNT3c', 'TNT')}
    stats_path = res_dir / 'stats_variants.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}
    pairs = {}

    def finish(meta, on_out, off_out, wPNKC):
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
        row.update(per_odor_extras(on_out[0], off_out[0], on[H.PRIMARY_MODEL_RDM],
            off[H.PRIMARY_MODEL_RDM], drive, data_rates
        ))
        rows.append(row)
        pairs[(meta['tag'], 'on')] = on[H.PRIMARY_MODEL_RDM]
        pairs[(meta['tag'], 'acute_off')] = off[H.PRIMARY_MODEL_RDM]
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        print(f"{meta['tag']}: rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f}; "
            f"PN max/min {meta.get('pn_rate_max_over_min', float('nan')):.2f}; "
            f"iso vs drive {row['spearman_per_odor_iso_vs_drive']:+.2f}, vs dfrac "
            f"{row['spearman_per_odor_iso_vs_dfrac']:+.2f}; block "
            f"{row[H.PRIMARY_MODEL_RDM + '__block_fruity_small_iso']:+.3f}; dev vs ctrl "
            f"{row[H.PRIMARY_MODEL_RDM + '__on_spearman_dev']:+.2f}; {meta['seconds']:.0f}s",
            flush=True
        )

    if 'A' in args.parts or 'B' in args.parts:
        jobs = []
        if 'A' in args.parts:
            for base, kws in BASES.items():
                for inhadd in args.inhadd:
                    inhsc = PN_INHSC_DEFAULT * (inhadd + TYPICAL_INH) / (PN_INHADD_DEFAULT + TYPICAL_INH)
                    jobs.append((f'{base}_inhadd{inhadd:g}', base,
                        dict(kws, pn_inhadd=inhadd, pn_inhsc=inhsc),
                        dict(part='A', pn_inhadd=inhadd, pn_inhsc=inhsc)))
        if 'B' in args.parts:
            jobs.append(('claw-dynamics_noPNAPL', 'claw-dynamics',
                dict(BASES['claw-dynamics'], pn_claw_to_apl=False),
                dict(part='B', pn_inhadd=PN_INHADD_DEFAULT, pn_inhsc=PN_INHSC_DEFAULT)))
        for tag, base, kws, info in jobs:
            if tag in done:
                continue
            t0 = time.time()
            try:
                on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag)
            except Exception as e:
                print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                rows.append(dict(variant=tag, base=base, tag=tag, target_rate=TARGET,
                    error=repr(e)[:300], **info))
                pd.DataFrame(rows).to_csv(stats_path, index=False)
                continue
            meta = dict(variant=tag, base=base, seed=None, target_rate=TARGET, tag=tag,
                n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]),
                rate_off=rate(off_out[0]), seconds=time.time() - t0,
                mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
                **info, **extra,
            )
            finish(meta, on_out, off_out, wPNKC)

    if 'C' in args.parts:
        base = 'bouton'
        tag0 = f'{base}_pnapl_x1'
        t0 = time.time()
        try:
            on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, BOUTON, out_dir, tag0)
        except Exception as e:
            print(f'{tag0} FAILED: {e!r}'[:300], flush=True)
            rows.append(dict(variant=tag0, base=base, tag=tag0, target_rate=TARGET,
                error=repr(e)[:300], part='C', pn_apl_multiplier=1.0))
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            on_out = None
        if on_out is None:
            initial = None
        else:
            initial = {k: float(params[k]) for k in ('fixed_thr', 'wAPLKC_scale',
                'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')}
        n_target = int((on_out[0].astype(float) != 0).to_numpy().sum()) if on_out else 0
        if on_out is not None and tag0 not in done:
            meta = dict(variant=tag0, base=base, seed=None, target_rate=TARGET, tag=tag0,
                n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]),
                rate_off=rate(off_out[0]), seconds=time.time() - t0, part='C',
                pn_apl_multiplier=1.0, kc_apl_factor=1.0, **initial,
                mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
                **{k: v for k, v in extra.items() if k.startswith('pn_rate') or k.startswith('spearman_pn') or k.startswith('pearson_pn')},
            )
            finish(meta, on_out, off_out, wPNKC)
        sim = None
        if initial is not None:
            sim, _ = make_fixed_param_kc_simulator(orn, initial_params=initial, **BOUTON,
                make_plots=False, silent=True
            )
        for m in (PN_APL_MULTIPLIERS if sim is not None else []):
            tag = f'{base}_pnapl_x{m:g}'
            if tag in done or m == 1.0:
                continue
            t0 = time.time()
            try:
                p = dict(initial, wAPLPN_scale=initial['wAPLPN_scale'] * m,
                    wPNAPL_scale=initial['wPNAPL_scale'] * m)
                f, r_on, s_on = match_kc_apl_scale(sim, p, n_target)
                p_on = dict(p, wAPLKC_scale=p['wAPLKC_scale'] * f, wKCAPL_scale=p['wKCAPL_scale'] * f)
                r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
            except Exception as e:
                print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                rows.append(dict(variant=tag, base=base, tag=tag, target_rate=TARGET,
                    error=repr(e)[:300], part='C', pn_apl_multiplier=m))
                pd.DataFrame(rows).to_csv(stats_path, index=False)
                continue
            for cond, (r, s) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
                r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
                s.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
            meta = dict(variant=tag, base=base, seed=None, target_rate=TARGET, tag=tag,
                n_kcs=len(r_on), n_glomeruli=wPNKC.shape[1], rate_on=rate(r_on),
                rate_off=rate(r_off), seconds=time.time() - t0, part='C',
                pn_apl_multiplier=m, kc_apl_factor=f, **p_on,
                mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
            )
            finish(meta, (r_on, s_on), (r_off, s_off), wPNKC)
        del sim

    if pairs:
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance'
            ).reset_index().to_csv(res_dir / 'model_pair_distances_long.csv', index=False,
            mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E3_drive_dependent_gain.py', host=subprocess.run(['hostname'],
            capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=H.olfsysm_dir(), capture_output=True, text=True).stdout.strip(),
        bases=BASES, bouton=BOUTON, inhadd=args.inhadd, typical_inh=TYPICAL_INH,
        pn_apl_multipliers=PN_APL_MULTIPLIERS, approx_conc_note=H.APPROX_CONC_NOTE,
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), res_dir)
    print('done ->', res_dir)


def make_figure(stats: pd.DataFrame, res_dir: Path) -> None:
    plt = H._plt()
    v = H.PRIMARY_MODEL_RDM
    s = stats[stats['error'].isna()] if 'error' in stats else stats
    panels = [
        ('pn_rate_max_over_min', 'PN odor-rate contrast (max/min; ORN 2.1)'),
        ('spearman_per_odor_iso_vs_drive', 'per-odor APL effect vs ORN drive (data 0.50)'),
        ('spearman_per_odor_iso_vs_dfrac', 'per-odor APL effect vs density change (data 0.53)'),
        ('spearman_rate_on_vs_data', 'per-odor rates, model on vs data TNT3c'),
        (f'{v}__block_fruity_small_iso', 'fruity x small-ester block (data +0.055)'),
        (f'{v}__on_spearman_dev', 'deviations vs control data'),
        (f'{v}__on_spearman', 'Spearman with control data'),
        (f'{v}__on_mean_model', 'mean 1 - cosine, APL on (data 0.84)'),
    ]
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    colors = {'claw-dynamics': '#2a78d6', 'hemibrain-wd20': '#c8553d', 'bouton': '#4f9d69'}
    for ax, (col, title) in zip(axs.flat, panels):
        if col not in s:
            ax.set_title(title + ' (n/a)', fontsize=8)
            continue
        a = s[s.part == 'A'] if 'part' in s else s.iloc[0:0]
        for base, sub in a.groupby('base'):
            sub = sub.sort_values('pn_inhadd')
            ax.plot(sub.pn_inhadd, sub[col], 'o-', color=colors.get(base, '#555'), ms=4,
                lw=1.4, label=f'{base} (pn_inhadd sweep)'
            )
        c = s[s.part == 'C'] if 'part' in s else s.iloc[0:0]
        if len(c):
            ax2 = ax.twiny()
            cc = c.sort_values('pn_apl_multiplier')
            ax2.plot(cc.pn_apl_multiplier.replace(0, 0.1), cc[col], 's--',
                color=colors['bouton'], ms=4, lw=1.2, label='bouton (PN<->APL x m)'
            )
            ax2.set_xscale('log')
            ax2.set_xlabel('PN<->APL weight multiplier (0 drawn at 0.1)', fontsize=7)
            ax2.legend(fontsize=6, frameon=False, loc='lower right')
        b = s[s.part == 'B'] if 'part' in s else s.iloc[0:0]
        for _, r in b.iterrows():
            ax.axhline(r[col], color='#8c6bb1', lw=1, ls=':', label='claw-dynamics, no PN->APL')
        ax.set_xscale('log')
        ax.set_xlabel('pn_inhadd')
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, frameon=False)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.suptitle('E3: drive-dependent APL gain. ' + H.APPROX_CONC_NOTE[:100] + '...', fontsize=8)
    fig.tight_layout()
    fig.savefig(res_dir / 'E3_drive_dependent_gain.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
