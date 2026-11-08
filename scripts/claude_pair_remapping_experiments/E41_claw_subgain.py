#!/usr/bin/env python3
"""E41: the sub-threshold gain of the per-claw transfer (does a claw's weak input have to count for nothing?).

The per-claw threshold enters each claw's drive x as max(x - T, 0), so PN input below T (40-70 Hz)
counts for nothing. olfsysm's `claw_thr_subgain` g adds g * min(max(x, 0), T): slope g below the
threshold and 1 above it, continuous at T; g = 0 is the threshold model, g = 1 the model without a
dead zone (max(x, 0)).

Part A: the bouton base with the per-claw threshold at 70 and 40 Hz and g in {0, 0.25, 0.5, 1}, tuned at
5 % with APL on, APL removed acutely (E24 / E26 / E27's pipeline, `tune_and_knockout`).
Part B: at the reference model's fitted parameters (E34's capped gain-1.5 structure fits, 40 Hz and
70 Hz; every fitted scale held), g in {0, 0.25, 0.5}: the breadth weights from an APL-off run of the
model with that g at the base's tuned parameters (as the joint fit builds its model, and as E39 did per
variant), the somatic threshold re-tuned to the fit's APL-on response rate, APL off acutely; g = 0
reproduces the fit (checked against E34's saved outputs). Part B is scored with the joint fit's extended
loss as well.

Outputs: <shared>/results/E41_claw_subgain/ and ./outputs/E41_claw_subgain/.
Usage: OMP_NUM_THREADS=12 nice -n 10 ./venv/bin/python E41_claw_subgain.py [--parts A B] [--gains 0 0.25 0.5 1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E1_apl_knockout import bisect_threshold
from E3_drive_dependent_gain import BOUTON, TARGET, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E10_joint_fit import base_kws, bouton_base, breadth_weights, make_loss
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E26_claw_power import sharing_deficit, deficit_table, WEAK
from E27_glomerulus_gain import ORACLE, gained
from E35_divisive_apl import suppression_stats

EXPERIMENT = 'E41_claw_subgain'
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
TAU = 0.05
FITS = {'thr40': 'thr40_gain1.5_k2_cap3', 'thr70': 'thr70_gain1.5_k2_cap3'}
V = H.PRIMARY_MODEL_RDM + '__'


def subgain_kws(g: float) -> dict:
    """The model kwarg for sub-threshold gain g (none for g = 0, the plain threshold model)."""
    return {} if g == 0 else {'claw_thr_subgain': float(g)}


def make_figure(stats: pd.DataFrame, res_dir) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    a = stats[stats.part == 'A']
    panels = [('mean distance, APL on / off\n(data 0.84 / 0.66; noise-corrected 0.79 / 0.58)', [(V + 'on_mean_model', 'APL on'), (V + 'off_mean_model', 'APL off')], [0.84, 0.66]),
        ('natural-source alignment\n(data 0.44 APL on / 0.36 APL off)', [(V + 'on_spearman_resid_vs_vcf_hyp', 'APL on'), (V + 'off_spearman_resid_vs_vcf_hyp', 'APL off')], [0.44, 0.36]),
        ('structure vs wild-type somata (data 0.82)', [(V + 'on_spearman_resid_vs_remap_wt', 'APL on')], [0.82]),
        ('APL-on excess, block / other pairs\n(data −0.08 / −0.10)', [('on_excess_block', 'block pairs'), ('on_excess_other_between', 'other pairs')], [-0.08, -0.10]),
        ('ρ on odor-odor distances', [(V + 'on_spearman', 'APL on vs control flies'), (V + 'off_spearman', 'APL off vs APL output blocked (TNT)')], []),
        ('response rate ρ', [('spearman_rate_on_vs_data', 'APL on'), ('spearman_rate_off_vs_data', 'APL off')], []),
        ('weak odors\' mean APL-off response fraction', [('weak_off_mean', 'model')], []),
        ('APL-off sharing deficit (0 = data)', [('deficit_terp_x_fruity', 'terpene / 1-8ol × fruity ester'), ('deficit_ms_paa_pairs', 'ms / paa pairs'), ('deficit_other_pairs', 'other pairs')], [0.0])]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (title, cols, refs) in zip(axes.ravel(), panels):
        for thr, mk in ((70.0, 'o'), (40.0, 's')):
            s = a[a.claw_thr == thr].sort_values('claw_thr_subgain')
            for i, (col, lab) in enumerate(cols):
                if col in s:
                    ax.plot(s.claw_thr_subgain, s[col], ['-', '--', ':'][i] + mk, color=['#2b6cb0', '#d9480f', '#2f855a'][i], ms=5, lw=1.2,
                        label=f'{lab}, {thr:g} Hz')
        for r in refs:
            ax.axhline(r, color='0.45', lw=1, ls=':')
        ax.set_title(title, fontsize=9); ax.set_xlabel('sub-threshold gain g'); ax.legend(fontsize=6.5, frameon=False)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    fig.suptitle('E41: sub-threshold gain of the per-claw transfer (bouton base, tuned at 5 % with APL on, APL off acutely; '
        'g = 0 is the threshold model, g = 1 no dead zone)', fontsize=10.5)
    fig.tight_layout(); fig.savefig(res_dir / 'E41_claw_subgain.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--parts', nargs='+', default=['A', 'B'], choices=['A', 'B'])
    ap.add_argument('--gains', nargs='+', type=float, default=[0.0, 0.25, 0.5, 1.0], help='part A gains')
    ap.add_argument('--b-gains', nargs='+', type=float, default=[0.0, 0.25, 0.5], help='part B gains')
    ap.add_argument('--thrs', nargs='+', type=float, default=[70.0, 40.0], help='part A per-claw thresholds')
    ap.add_argument('--fits', nargs='+', default=list(FITS), choices=list(FITS), help='part B bases')
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        make_figure(pd.DataFrame(rows), res_dir); return
    done = {r['tag'] for r in rows}
    orn = H.apl16_orn_deltas()
    # code provenance at the start: the checkout can move (a pull) while the run goes on
    model_prov = H.model_provenance(orn_deltas=orn)
    orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs(); refs = all_refs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned = json.loads((out_dir / 'tuned_params.json').read_text()) if (out_dir / 'tuned_params.json').exists() else {}
    loss_fn = make_loss(data, data_rates, orn_pairs, drive, ref, structure=True)
    pairs = {}; deficits = []

    def finish(tag, meta, on_out, off_out, wPNKC, extra_cols=None, fixthr=None):
        meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row['on_excess_diff'] = row['on_excess_block'] - row['on_excess_other_between']
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        fon, foff = H.response_fractions(on_out[0]), H.response_fractions(off_out[0])
        row.update(suppression_stats(fon, foff, data_rates))
        row.update({f'fon_{o}': float(fon[o]) for o in fon.index}); row.update({f'foff_{o}': float(foff[o]) for o in foff.index})
        row['weak_off_mean'] = float(foff.reindex(WEAK).mean())
        row.update({f'xl_{kk}': vv for kk, vv in loss_fn({'on': on_out, 'off': off_out}).items()})
        row.update(extra_cols or {})
        d = deficit_table(off_out[1], data_pairs); d.insert(0, 'tag', tag); deficits.append(d)
        rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = fon.reindex(fracs.index); fracs[f'{tag}_acute_off'] = foff.reindex(fracs.index)
        if fixthr is not None:
            fracs[f'{tag}_on_fitthr'] = H.response_fractions(fixthr).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        wk = ' '.join(f'{o}:{foff[o]:.3f}' for o in WEAK)
        print(f"{tag}: rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f} (x{meta['rate_ratio_off_over_on']:.2f}); weak off [{wk}]; "
            f"mean dist on/off {row[V + 'on_mean_model']:.3f}/{row[V + 'off_mean_model']:.3f}; vcf on/off {row.get(V + 'on_spearman_resid_vs_vcf_hyp', np.nan):.2f}/"
            f"{row.get(V + 'off_spearman_resid_vs_vcf_hyp', np.nan):.2f}; wt {row.get(V + 'on_spearman_resid_vs_remap_wt', np.nan):.2f}; "
            f"excess on block/other {row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f}; block {row[V + 'block_fruity_small_iso']:+.3f}; "
            f"rho ctrl/TNT {row[V + 'on_spearman']:.3f}/{row[V + 'off_spearman']:.3f}; rate rho on/off {row['spearman_rate_on_vs_data']:.2f}/"
            f"{row['spearman_rate_off_vs_data']:.2f}; loss {row['xl_loss']:.3f}; {meta['seconds']:.0f}s", flush=True)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
        done.add(tag)

    if 'A' in args.parts:
        for thr in args.thrs:
            for g in args.gains:
                tag = f'A_thr{thr:g}_g{g:g}'
                if tag in done:
                    continue
                kws = dict(BOUTON, claw_thr=thr, **subgain_kws(g))
                t0 = time.time()
                try:
                    on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag, with_pn_contrast=False)
                except Exception as e:
                    print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                    rows.append(dict(variant=tag, tag=tag, part='A', claw_thr=thr, claw_thr_subgain=g, error=repr(e)[:300]))
                    pd.DataFrame(rows).to_csv(stats_path, index=False); continue
                tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
                tuned[tag].update(claw_thr=thr, claw_thr_subgain=g)
                (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1))
                meta = dict(variant=tag, tag=tag, part='A', base='bouton', seed=None, target_rate=TARGET, claw_thr=thr, claw_thr_subgain=g,
                    n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
                    mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()), **extra,
                    **{k: float(np.mean(np.asarray(params[k], dtype=float))) for k in PARAM_KEYS if k in params})
                finish(tag, meta, on_out, off_out, wPNKC)

    if 'B' in args.parts:
        from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator
        fits = pd.read_csv(H.results_dir('E34_gain_bounded') / 'fits.csv').set_index('tag')
        e34_out = H.outputs_dir('E34_gain_bounded')
        for b in args.fits:
            fit = fits.loc[FITS[b]]
            claw_thr = float(fit['claw_thr'])
            p_fit = {kk: float(fit[kk]) for kk in PARAM_KEYS}
            p_fit['wKCKC_scale'] = float(fit['kc_kc_inh'])
            initial, wnorm, wPNKC0, _ = bouton_base(orn, out_dir, claw_thr=claw_thr)
            W = gained(wPNKC0, ORACLE, float(fit['gain']))
            rate_target = None
            for g in sorted(args.b_gains):  # g = 0 first: it sets the rate target
                tag = f'B_{b}_g{g:g}'
                if tag in done and g != 0:
                    continue
                kws = dict(base_kws(claw_thr), **subgain_kws(g))
                t0 = time.time()
                try:
                    r_b, s_b, _, _ = fit_mb_model(orn, fixed_thr=initial['fixed_thr'], wAPLKC=0.0, wKCAPL=initial['wKCAPL_scale'],
                        wAPLPN=0.0, wPNAPL=initial['wPNAPL_scale'], **kws, _wPNKC=W)
                    breadth = pd.Series(s_b.astype(float).sum(axis='columns').to_numpy(), index=s_b.index.get_level_values(H.KC_ID_LEVEL))
                    sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p_fit, **kws, _wPNKC=W,
                        _wAPLKC=breadth_weights(wnorm['wAPLKC'], breadth, float(fit['k'])), _wKCAPL=wnorm['wKCAPL'],
                        _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
                    r_fix, s_fix = sim(p_fit)
                    check = {}
                    if g == 0:
                        saved = e34_out / f'{FITS[b]}_on_responses.parquet'
                        if saved.exists():
                            check['fit_responses_mismatch_vs_saved'] = int((pd.read_parquet(saved).astype(float).to_numpy() != r_fix.astype(float).to_numpy()).sum())
                        rate_target = rate(r_fix)
                        print(f'{tag}: fit reproduced at rate {rate_target:.5f} (fits.csv {float(fit["rate_on"]):.5f}); {check}', flush=True)
                        if tag in done:
                            continue
                    extra = {kk: p_fit[kk] for kk in p_fit if kk != 'fixed_thr'}
                    thr, thr_f, r_on, s_on = bisect_threshold(sim, p_fit['fixed_thr'], rate_target, extra=extra)
                    p_on = dict(p_fit, fixed_thr=float(np.mean(thr)))
                    r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                    del sim
                except Exception as e:
                    print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                    rows.append(dict(variant=tag, tag=tag, part='B', base=b, claw_thr_subgain=g, error=repr(e)[:300]))
                    pd.DataFrame(rows).to_csv(stats_path, index=False); continue
                for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off)), ('on_fitthr', (r_fix, s_fix)), ('base_acute_off', (r_b, s_b))):
                    r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet'); sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
                tuned[tag] = dict(p_on, claw_thr=claw_thr, claw_thr_subgain=g, gain=float(fit['gain']), k=float(fit['k']), fit=FITS[b],
                    thr_factor=float(thr_f), base_params=initial, kc_kc_tau=TAU)
                (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1, default=float))
                meta = dict(variant=tag, tag=tag, part='B', base=b, fit=FITS[b], seed=None, target_rate=rate_target, claw_thr=claw_thr,
                    claw_thr_subgain=g, gain=float(fit['gain']), k=float(fit['k']), thr_factor=float(thr_f), n_kcs=len(r_on),
                    n_glomeruli=W.shape[1], rate_on=rate(r_on), rate_off=rate(r_off), rate_on_fitthr=rate(r_fix), seconds=time.time() - t0,
                    kc_kc_inh=p_on['wKCKC_scale'], **{kk: float(np.mean(vv)) for kk, vv in p_on.items()}, **check)
                finish(tag, meta, (r_on, s_on), (r_off, s_off), W, fixthr=r_fix)

    if deficits:
        pd.concat(deficits).to_csv(res_dir / 'pair_deficits_long.csv', index=False, mode='a', header=not (res_dir / 'pair_deficits_long.csv').exists())
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E41_claw_subgain.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), parts=args.parts, base=BOUTON,
        thrs=args.thrs, gains=args.gains, b_gains=args.b_gains, fits={b: FITS[b] for b in args.fits},
        fits_source='results/E34_gain_bounded/fits.csv', tau=TAU, approx_conc_note=H.APPROX_CONC_NOTE, model=model_prov)
    (res_dir / f'provenance_{"".join(args.parts)}.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
