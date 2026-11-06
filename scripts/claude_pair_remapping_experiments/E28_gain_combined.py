#!/usr/bin/env python3
"""E28: the oracle glomerulus gain in the combined model (per-claw threshold x breadth-weighted
APL x KC-KC inhibition), E19's points with the weak odors' glomeruli strengthened.

E27 found that scaling the PN->claw weights of the eight glomeruli that carry the weak odors
(VA7m, DL1, D, DM1, DP1m, VC1, DC2, DM5) by 1.5 on the 70 Hz threshold base recruits the weak
odors with the fruity-ester overlap and brings the structure beyond the ORN input to the data,
without a block; E19's three-mechanism points have the block and the control excess profile
but little structure beyond the ORN input. Here E19's pipeline is re-run with the gained wiring:
for each (claw_thr, gain): the bouton base with `_wPNKC` = the gained matrix, APL weights
normalized as E19, threshold re-tuned at the gain to E16's 5 % response count (k 0, KC-KC 0),
breadth = this model's own APL-off spike counts at that point, KC<->APL scales re-matched for
the breadth exponent (as E12 / E19), then the KC-KC scale with the threshold re-tuned (as E8 /
E19); acute APL off = APL output weights zero. Points: claw_thr 70 x breadth 2 x KC-KC 100
(E19's block-first point) at gains 1, 1.25, 1.5, and claw_thr 40 x breadth 2 x KC-KC 30
(E19's balanced point) at gains 1, 1.5. Scored as E27 plus E19's statistics.

Outputs: <shared>/results/E28_gain_combined/ and ./outputs/E28_gain_combined/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E28_gain_combined.py [--points thr70_k2_kckc100 thr40_k2_kckc30] [--gains 1 1.25 1.5]
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
from E3_drive_dependent_gain import BOUTON, match_kc_apl_scale, per_odor_extras, rate
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E26_claw_power import sharing_deficit, deficit_table, WEAK
from E27_glomerulus_gain import ORACLE, gained

EXPERIMENT = 'E28_gain_combined'
E16_OUT = H.HERE / 'outputs' / 'E16_claw_saturation'
TAU = 0.05
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
POINTS = {'thr70_k2_kckc100': (70.0, 2.0, 100.0), 'thr40_k2_kckc30': (40.0, 2.0, 30.0)}


def make_figure(stats: pd.DataFrame, fracs: pd.DataFrame, res_dir) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    tags = list(stats.tag); x = np.arange(len(tags))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]; wdt = 0.13
    for k, (o, c) in enumerate(zip(WEAK, ['#1f1f1f', '#8a6d00', '#2b6cb0', '#d9480f', '#2f855a', '#6b46c1'])):
        ax.bar(x + (k - 2.5) * wdt, [fracs.loc[o, f'{t}_acute_off'] for t in tags], wdt, color=c, label=o)
        ax.axhline(fracs.loc[o, 'data_TNT'], color=c, lw=0.8, ls=':')
    ax.set_title('weak odors: APL-off response fraction (dotted: data)', fontsize=9); ax.legend(fontsize=7, frameon=False, ncol=3)
    panels = [('block (data +0.055) and APL-on excess, block / other (data -0.08 / -0.10)', [(p + 'block_fruity_small_iso', 'block'), ('on_excess_block', 'APL-on excess, block pairs'), ('on_excess_other_between', 'APL-on excess, other pairs')], [0.055, -0.08]),
        ('structure beyond the ORN input (data 0.44 on / 0.36 off)', [(p + 'on_spearman_resid_vs_vcf_hyp', 'APL on'), (p + 'off_spearman_resid_vs_vcf_hyp', 'APL off')], [0.44, 0.36]),
        ('rank agreement with control (on) / TNT (off)', [(p + 'on_spearman', 'APL on vs control'), (p + 'off_spearman', 'APL off vs TNT')], []),
        ('rate ratio off / on (data 1.9-2.2) and rate rank agreement on / off', [('rate_ratio_off_over_on', 'rate ratio'), ('spearman_rate_on_vs_data', 'rate rank, APL on'), ('spearman_rate_off_vs_data', 'rate rank, APL off')], [1.9]),
        ('APL-off sharing deficit (0 = data) and mean distance on (data 0.84)', [('deficit_terp_x_fruity', 'terpene / 1-8ol x fruity ester'), ('deficit_ms_paa_pairs', 'ms / paa pairs'), (p + 'on_mean_model', 'mean distance, APL on')], [0.0, 0.84])]
    for ax, (title, cols, refs) in zip(list(axes.ravel())[1:], panels):
        for i, (col, lab) in enumerate(cols):
            if col in stats:
                ax.plot(x, stats[col].to_numpy(), ['-o', '--s', ':^'][i], color=['#2b6cb0', '#d9480f', '#2f855a'][i], ms=5, lw=1.2, label=lab)
        for r in refs:
            ax.axhline(r, color='0.45', lw=1, ls=':')
        ax.set_title(title, fontsize=8.5); ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=30, ha='right', fontsize=8); ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    fig.suptitle("E28: the oracle glomerulus gain at the three-mechanism points (bouton base, 5 %)", fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / 'E28_gain_combined.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--points', nargs='+', default=list(POINTS), choices=list(POINTS))
    ap.add_argument('--gains', nargs='+', type=float, default=None, help='default: 1, 1.25, 1.5 for thr70; 1, 1.5 for thr40')
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        make_figure(pd.DataFrame(rows), pd.read_csv(fracs_path, index_col=0), res_dir); return
    done = {r['tag'] for r in rows}
    orn = H.apl16_orn_deltas(); orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs(); refs = all_refs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned16 = json.loads((E16_OUT / 'tuned_params.json').read_text())
    tuned = json.loads((out_dir / 'tuned_params.json').read_text()) if (out_dir / 'tuned_params.json').exists() else {}
    from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator
    pairs = {}; deficits = []
    for point in args.points:
        thr, k, s = POINTS[point]
        gains = args.gains if args.gains else ([1.0, 1.25, 1.5] if thr == 70.0 else [1.0, 1.5])
        src = f'bouton_thr{thr:g}'
        initial = {kk: float(tuned16[src][kk]) for kk in PARAM_KEYS}
        thr_kws = dict(BOUTON, claw_thr=thr, make_plots=False, silent=True)
        _, _, wPNKC0, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'], wAPLKC=initial['wAPLKC_scale'], wKCAPL=initial['wKCAPL_scale'],
            wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **thr_kws)
        wnorm = {kk: pd0[kk] / pd0[f'{kk}_scale'] for kk in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
        claw_kc = wnorm['wAPLKC'].index.get_level_values(H.KC_ID_LEVEL)
        on_resp0 = pd.read_parquet(E16_OUT / f'{src}_on_responses.parquet')
        n_target = int((on_resp0.astype(float) != 0).to_numpy().sum()); rate_target = rate(on_resp0)
        mean_claws = float(kc_by_glomerulus(wPNKC0).sum(axis='columns').mean())
        for g in gains:
            tag = f'{point}_g{g:g}'
            if tag in done:
                continue
            t0 = time.time()
            try:
                W = gained(wPNKC0, ORACLE, g) if g != 1.0 else wPNKC0
                # stage 0: plain threshold model with the gained wiring, threshold re-tuned at k 0 / KC-KC 0
                sim0, _ = make_fixed_param_kc_simulator(orn, initial_params=initial, **thr_kws, _wPNKC=W,
                    _wAPLKC=wnorm['wAPLKC'], _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=0.0, kc_kc_tau=TAU)
                thr0, thr0_f, r0, s0 = bisect_threshold(sim0, initial['fixed_thr'], rate_target, extra={kk: initial[kk] for kk in ('wAPLKC_scale', 'wKCAPL_scale')})
                p0 = dict(initial, fixed_thr=thr0, wKCKC_scale=0.0)
                r0_off, s0_off = sim0(dict(p0, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                b = pd.Series(s0_off.astype(float).sum(axis='columns').to_numpy(), index=s0_off.index.get_level_values(H.KC_ID_LEVEL)) + 1.0
                f_claw = ((b / b.mean()) ** k).reindex(claw_kc).fillna(1.0).to_numpy()
                del sim0
                # stage 1: breadth-weighted APL, KC<->APL scales re-matched to the k 0 response count
                sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p0, **thr_kws, _wPNKC=W,
                    _wAPLKC=wnorm['wAPLKC'] * f_claw, _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=0.0, kc_kc_tau=TAU)
                n0 = int((r0.astype(float) != 0).to_numpy().sum())
                f, _, _ = match_kc_apl_scale(sim, dict(p0, wKCKC_scale=0.0), n0)
                p_k = dict(p0, wAPLKC_scale=p0['wAPLKC_scale'] * f, wKCAPL_scale=p0['wKCAPL_scale'] * f, wKCKC_scale=0.0)
                # stage 2: KC-KC inhibition with the threshold re-tuned to the rate target
                extra = {kk: p_k[kk] for kk in ('wAPLKC_scale', 'wKCAPL_scale')}; extra['wKCKC_scale'] = s
                new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_k['fixed_thr'], rate_target, extra=extra)
                p_on = dict(p_k, fixed_thr=new_thr, wKCKC_scale=s)
                r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                del sim
            except Exception as e:
                print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                rows.append(dict(variant=tag, tag=tag, point=point, claw_thr=thr, k=k, kc_kc_inh=s, gain=g, error=repr(e)[:300])); pd.DataFrame(rows).to_csv(stats_path, index=False); continue
            for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off)), ('stage0_on', (r0, s0)), ('stage0_acute_off', (r0_off, s0_off))):
                r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet'); sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
            tuned[tag] = {kk: float(np.mean(np.asarray(v, dtype=float))) for kk, v in p_on.items()}
            tuned[tag].update(claw_thr=thr, k=k, kc_kc_inh=s, gain=g, glomeruli=ORACLE if g != 1.0 else [], stage0_fixed_thr=float(np.mean(thr0)), kc_apl_factor=f, thr_factor=thr_f)
            (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1))
            meta = dict(variant=tag, base='bouton', seed=None, target_rate=0.05, tag=tag, point=point, claw_thr=thr, k=k, kc_kc_inh=s, kc_kc_tau=TAU, gain=g,
                glomeruli=','.join(ORACLE) if g != 1.0 else '', stage0_fixed_thr=float(np.mean(thr0)), stage0_thr_factor=thr0_f, kc_apl_factor=f, thr_factor=thr_f,
                n_kcs=len(r_on), n_glomeruli=W.shape[1], rate_on=rate(r_on), rate_off=rate(r_off), seconds=time.time() - t0, mean_claws_per_kc=mean_claws,
                **{kk: float(np.mean(v)) for kk, v in p_on.items()})
            meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
            row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC0, orn, orn_pairs, drive, data, ref)
            on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
            row.update(per_odor_extras(r_on, r_off, on_p, off_p, drive, data_rates))
            for cond, sp in (('on', s_on), ('off', s_off)):
                row.update({f'{cond}_{kk}': v for kk, v in density_excess(sp).items()})
            row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(s_off, data_pairs))
            d = deficit_table(s_off, data_pairs); d.insert(0, 'tag', tag); deficits.append(d)
            rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            fracs[f'{tag}_on'] = H.response_fractions(r_on).reindex(fracs.index); fracs[f'{tag}_acute_off'] = H.response_fractions(r_off).reindex(fracs.index)
            fracs.to_csv(fracs_path)
            v = H.PRIMARY_MODEL_RDM
            wk = ' '.join(f"{o}:{fracs.loc[o, f'{tag}_acute_off']:.3f}" for o in WEAK)
            print(f"{tag}: thr0 {float(np.mean(thr0)):.0f} apl x{f:.3f} thr x{thr_f:.3f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f} (x{meta['rate_ratio_off_over_on']:.2f}); "
                f"block {row[v + '__block_fruity_small_iso']:+.3f}; excess on {row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f} off {row['off_excess_block']:+.3f}/{row['off_excess_other_between']:+.3f}; "
                f"vcf resid on/off {row.get(v + '__on_spearman_resid_vs_vcf_hyp', float('nan')):.2f}/{row.get(v + '__off_spearman_resid_vs_vcf_hyp', float('nan')):.2f}; vs ctrl {row[v + '__on_spearman']:.3f}; off vs TNT {row[v + '__off_spearman']:.3f}; "
                f"iso R2 {row[v + '__iso_r2']:.3f}; mean dist on/off {row[v + '__on_mean_model']:.3f}/{row[v + '__off_mean_model']:.3f}; rate rank on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; "
                f"weak odors off [{wk}]; deficit terp-x-fruity/ms-paa/other {row['deficit_terp_x_fruity']:+.3f}/{row['deficit_ms_paa_pairs']:+.3f}/{row['deficit_other_pairs']:+.3f}; {meta['seconds']:.0f}s", flush=True)
            pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
                res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
            done.add(tag)
    if deficits:
        pd.concat(deficits).to_csv(res_dir / 'pair_deficits_long.csv', index=False, mode='a', header=not (res_dir / 'pair_deficits_long.csv').exists())
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E28_gain_combined.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=(H.HERE / '../../../olfsysm').resolve(), capture_output=True, text=True).stdout.strip(),
        base=BOUTON, points={p: POINTS[p] for p in args.points}, gains=args.gains, oracle_glomeruli=ORACLE, tau=TAU, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), fracs, res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
