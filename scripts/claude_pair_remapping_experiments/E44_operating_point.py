#!/usr/bin/env python3
"""E44: the operating point: the bouton bases and the reference model at the control flies'
response fraction (6.8 %) instead of the program's 5 % convention.

Part A: the bouton base with the per-claw threshold (70 and 40 Hz; E16 / E24's pipeline) tuned
with APL on to half the control flies' fraction, 5 %, the control flies' fraction and 10 %
(fit_mb_model's target_sparsity), then acute APL off. olfsysm sets the somatic thresholds so
that twice the target responds before APL acts, so the acute APL-off rate is about twice the
target at every point (2 x 6.8 % = 13.7 %, against the 15.0 % of the flies with APL output
blocked).
Part C: E34's capped eight-glomerulus fits (40 and 70 Hz, gain 1.5, breadth exponent 2; the
reference model) at their fitted parameters (which must reproduce E34) and with only the
somatic threshold re-tuned (bisection, as E38's part B) so that the APL-on response rate is the
control flies' fraction: separates the threshold's effect from part B's refit.
Part B (the same two fits re-fitted with the loss's rate target at the control flies' fraction)
runs through `E10_joint_fit.py --experiment E44_operating_point --rate-target ctrl_frac`; its
rows are in this experiment's fits.csv, and --summary puts them in the comparison table.

Every variant is scored as E38 (E4's statistics, E7's excess profile, per-odor response
fractions, E21's APL-off sharing deficit, E35's suppression uniformity), plus the joint fit's
extended loss (E25's structure loss) with the APL-on rate target at 5 % and at the control
flies' fraction, and the mean absolute difference between the model's and the data's per-odor
response fractions (APL on vs control flies, acute APL off vs APL output blocked (TNT)).

Outputs: <shared>/results/E44_operating_point/ (stats_variants.csv,
per_odor_response_fractions.csv, comparison.csv, provenance.json, E44_operating_point.png) and
./outputs/E44_operating_point/ (responses / spike counts per variant, tuned_params.json).
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E44_operating_point.py [--parts A C]
       ./venv/bin/python E44_operating_point.py --summary   (comparison table and figure only)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E1_apl_knockout import bisect_threshold
from E3_drive_dependent_gain import BOUTON, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E10_joint_fit import TAU, base_kws, bouton_base, breadth_weights, make_loss
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E26_claw_power import WEAK, sharing_deficit
from E35_divisive_apl import suppression_stats

EXPERIMENT = 'E44_operating_point'
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
CTRL_FRAC = H.data_target('ctrl_frac')
# part A's operating points: half the control flies' fraction, the 5 % convention, the control
# flies' fraction, 10 %
TARGETS_A = (CTRL_FRAC / 2, 0.05, CTRL_FRAC, 0.10)
THRS_A = (70.0, 40.0)
# part C: tag in E34's fits.csv -> (per-claw threshold, breadth exponent, gain)
REF_EXPERIMENT = 'E34_gain_bounded'
REF_FITS: Dict[str, Tuple[float, float, float]] = {
    'thr40_gain1.5_k2_cap3': (40.0, 2.0, 1.5),
    'thr70_gain1.5_k2_cap3': (70.0, 2.0, 1.5),
}
V = H.PRIMARY_MODEL_RDM
SUMMARY_COLS = {
    'rate_on': 'rate_on', 'rate_off': 'rate_off', 'rate_ratio_off_over_on': 'rate_ratio_off_over_on',
    'rho_ctrl': f'{V}__on_spearman', 'rho_tnt': f'{V}__off_spearman',
    'block': f'{V}__block_fruity_small_iso', 'iso_r2': f'{V}__iso_r2',
    'mean_dist_on': f'{V}__on_mean_model', 'mean_dist_off': f'{V}__off_mean_model',
    'struct_wt_on': f'{V}__on_spearman_resid_vs_remap_wt',
    'vcf_on': f'{V}__on_spearman_resid_vs_vcf_hyp', 'vcf_off': f'{V}__off_spearman_resid_vs_vcf_hyp',
    'rate_rho_on': 'spearman_rate_on_vs_data', 'rate_rho_off': 'spearman_rate_off_vs_data',
    'frac_mae_on': 'frac_mae_on', 'frac_mae_off': 'frac_mae_off',
    'weak_frac_off': 'weak_frac_off', 'weak_frac_on': 'weak_frac_on',
    'excess_on_block': 'on_excess_block', 'excess_on_other': 'on_excess_other_between',
    'suppression_sd_log': 'suppression_sd_log', 'loss_rate5': 'loss_rate5',
    'loss_rate_ctrl': 'loss_rate_ctrl',
}


def fraction_stats(r_on: pd.DataFrame, r_off: pd.DataFrame, data_rates: Dict[str, pd.Series]) -> dict:
    """Mean absolute difference between the model's and the data's per-odor response fractions
    (APL on vs control flies, APL off vs APL output blocked (TNT)), and the weak odors' mean
    fractions."""
    on = H.response_fractions(r_on).reindex(H.APL16_ODORS)
    off = H.response_fractions(r_off).reindex(H.APL16_ODORS)
    d_on = data_rates['TNT3c'].reindex(H.APL16_ODORS)
    d_off = data_rates['TNT'].reindex(H.APL16_ODORS)
    return dict(frac_mae_on=float((on - d_on).abs().mean()), frac_mae_off=float((off - d_off).abs().mean()),
        weak_frac_on=float(on[WEAK].mean()), weak_frac_off=float(off[WEAK].mean()),
        data_weak_frac_on=float(d_on[WEAK].mean()), data_weak_frac_off=float(d_off[WEAK].mean()))


def losses(loss5, loss_ctrl, on_out, off_out) -> dict:
    """The structure loss of a variant with the APL-on rate target at 5 % and at the control
    flies' fraction (every other term is the same in both)."""
    outs = {'on': on_out, 'off': off_out}
    l5, lc = loss5(outs), loss_ctrl(outs)
    out = dict(loss_rate5=l5['loss'], loss_rate_ctrl=lc['loss'])
    out.update({f'loss_{k}': v for k, v in l5.items() if k.startswith('term_') and k != 'term_rate_on'})
    out.update(loss_term_rate_on_rate5=l5['term_rate_on'], loss_term_rate_on_rate_ctrl=lc['term_rate_on'])
    return out


def summarize(res_dir, out_dir, loss5, loss_ctrl) -> pd.DataFrame:
    """comparison.csv: one row per variant (parts A and C from stats_variants.csv, part B from
    fits.csv, scored here from its saved outputs where a column is missing)."""
    rows = []
    stats_path = res_dir / 'stats_variants.csv'
    if stats_path.exists():
        for r in pd.read_csv(stats_path).to_dict('records'):
            rows.append(dict(part=r['part'], tag=r['tag'], claw_thr=r['claw_thr'], target=r.get('target_rate'),
                **{k: r.get(c, np.nan) for k, c in SUMMARY_COLS.items()}))
    fits_path = res_dir / 'fits.csv'
    if fits_path.exists():
        data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
        for r in pd.read_csv(fits_path).to_dict('records'):
            tag = r['tag']
            on_out = tuple(pd.read_parquet(out_dir / f'{tag}_on_{x}.parquet') for x in ('responses', 'spike_counts'))
            off_out = tuple(pd.read_parquet(out_dir / f'{tag}_acute_off_{x}.parquet') for x in ('responses', 'spike_counts'))
            extra = dict(fraction_stats(on_out[0], off_out[0], data_rates), **losses(loss5, loss_ctrl, on_out, off_out),
                **suppression_stats(H.response_fractions(on_out[0]), H.response_fractions(off_out[0]), data_rates))
            rows.append(dict(part='B', tag=tag, claw_thr=r['claw_thr'], target=r.get('rate_target'),
                **{k: (extra[c] if c in extra else r.get(c, np.nan)) for k, c in SUMMARY_COLS.items()}))
    t = pd.DataFrame(rows)
    t.to_csv(res_dir / 'comparison.csv', index=False)
    return t


def make_figure(comp: pd.DataFrame, res_dir) -> None:
    """Part A's statistics against the operating point (one line per per-claw threshold), with
    parts B and C as points and the KC data as dotted lines."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [
        ('rho_ctrl', 'ρ on odor-odor distances,\nAPL on vs control flies', 'ctrl_ceiling'),
        ('rho_tnt', 'ρ on odor-odor distances,\nAPL off vs APL output blocked (TNT)', 'tnt_ceiling'),
        ('block', 'fruity × small-ester block', 'block'),
        ('struct_wt_on', 'structure beyond the ORN input\nvs wild-type somata, APL on', 'ctrl_struct_wt'),
        ('vcf_on', 'natural-source alignment, APL on', 'ctrl_vcf'),
        ('vcf_off', 'natural-source alignment, APL off', 'tnt_vcf'),
        ('mean_dist_on', 'mean cosine distance, APL on', 'ctrl_mean_dist'),
        ('mean_dist_off', 'mean cosine distance, APL off', 'tnt_mean_dist'),
        ('rate_rho_on', 'response rate ρ, APL on', 'ctrl_rate_rho_ceiling'),
        ('rate_rho_off', 'response rate ρ, APL off', 'tnt_rate_rho_ceiling'),
        ('weak_frac_off', 'weak odors\' response rate, APL off', None),
        ('frac_mae_off', 'per-odor response rate,\nmean |model - data|, APL off', None),
    ]
    colors = {70.0: '#2a6fdb', 40.0: '#d1495b'}
    fig, axs = plt.subplots(3, 4, figsize=(15, 10.5), constrained_layout=True)
    a = comp[comp.part == 'A'].sort_values('target')
    weak_data = float(H.yang_response_fractions('TNT')[H.APL16_ODORS].mean()[WEAK].mean())
    for ax, (col, title, key) in zip(axs.flat, panels):
        for thr in (70.0, 40.0):
            s = a[a.claw_thr == thr]
            ax.plot(100 * s.rate_on, s[col], '-o', color=colors[thr], lw=1.5, ms=5, label=f'bouton base, {thr:g} Hz (part A)')
            for part, marker, label in (('C', 's', 'reference model, threshold re-tuned (C)'), ('B', 'D', 'reference model re-fitted (B)')):
                p = comp[(comp.part == part) & (comp.claw_thr == thr)]
                if len(p):
                    ax.plot(100 * p.rate_on, p[col], marker, color=colors[thr], mfc='white', mew=1.5, ms=7, ls='none',
                        label=f'{label}, {thr:g} Hz')
        ref = H.data_target(key) if key else (weak_data if col == 'weak_frac_off' else None)
        if ref is not None:
            ax.axhline(ref, color='0.3', ls=':', lw=1.2, label='KC data')
        ax.axvline(100 * CTRL_FRAC, color='0.75', lw=1, zorder=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('APL-on response rate (%)')
        ax.grid(alpha=0.25, lw=0.5)
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("The operating point: statistics against the APL-on response rate (grey line: the control flies' "
        f"{100 * CTRL_FRAC:.1f} %)", fontsize=12)
    fig.savefig(res_dir / 'E44_operating_point.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--parts', nargs='+', choices=['A', 'C'], default=['A', 'C'])
    ap.add_argument('--summary', action='store_true', help='only (re)write comparison.csv and the figure')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    loss5 = make_loss(data, data_rates, orn_pairs, drive, ref, structure=True, rate_target=0.05)
    loss_ctrl = make_loss(data, data_rates, orn_pairs, drive, ref, structure=True, rate_target=CTRL_FRAC)
    if args.summary:
        comp = summarize(res_dir, out_dir, loss5, loss_ctrl)
        make_figure(comp, res_dir)
        print(comp.drop(columns=['tag']).round(3).to_string(index=False))
        return
    # recorded at the start, so a pull during the run cannot change the commit it names
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
    pairs = {}

    def finish(tag, meta, on_out, off_out, wPNKC):
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
        on_p, off_p = on[V], off[V]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        row.update(suppression_stats(H.response_fractions(on_out[0]), H.response_fractions(off_out[0]), data_rates))
        row.update(fraction_stats(on_out[0], off_out[0], data_rates))
        row.update(losses(loss5, loss_ctrl, on_out, off_out))
        rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index)
        fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long_parts_AC.csv', index=False, mode='a',
            header=not (res_dir / 'model_pair_distances_long_parts_AC.csv').exists())
        pairs.clear()
        print(f"{tag}: rates on {row['rate_on']:.4f} off {row['rate_off']:.4f} (x{row['rate_ratio_off_over_on']:.2f}); "
            f"rho ctrl {row[V + '__on_spearman']:.3f} tnt {row[V + '__off_spearman']:.3f}; block {row[V + '__block_fruity_small_iso']:+.3f}; "
            f"struct wt {row[V + '__on_spearman_resid_vs_remap_wt']:+.2f}; vcf on/off {row[V + '__on_spearman_resid_vs_vcf_hyp']:+.2f}/"
            f"{row[V + '__off_spearman_resid_vs_vcf_hyp']:+.2f}; mean dist {row[V + '__on_mean_model']:.3f}/{row[V + '__off_mean_model']:.3f}; "
            f"rate rho {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; weak off {row['weak_frac_off']:.3f} "
            f"(data {row['data_weak_frac_off']:.3f}); loss @5% {row['loss_rate5']:.3f} @ctrl {row['loss_rate_ctrl']:.3f}; "
            f"{meta['seconds']:.0f}s", flush=True)
        done.add(tag)

    if 'A' in args.parts:
        for thr in THRS_A:
            for target in TARGETS_A:
                tag = f'A_thr{thr:g}_t{target:.3f}'
                if tag in done:
                    continue
                t0 = time.time()
                kws = dict(BOUTON, claw_thr=thr)
                on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag,
                    with_pn_contrast=False, target=target)
                tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
                tuned[tag].update(claw_thr=thr, target_sparsity=target)
                tuned_path.write_text(json.dumps(tuned, indent=1))
                meta = dict(variant=tag, tag=tag, part='A', base='bouton', seed=None, target_rate=target, claw_thr=thr,
                    k=0.0, kc_kc_inh=0.0, gain=1.0, n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1],
                    rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
                    mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()), **extra,
                    **{k: float(np.mean(np.asarray(params[k], dtype=float))) for k in PARAM_KEYS if k in params})
                meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
                finish(tag, meta, on_out, off_out, wPNKC)

    if 'C' in args.parts:
        from al_analysis.mb_model import make_fixed_param_kc_simulator
        fits = pd.read_csv(H.shared_dir() / 'results' / REF_EXPERIMENT / 'fits.csv')
        for ftag, (thr, k, gain) in REF_FITS.items():
            fr = fits[fits.tag == ftag]
            if len(fr) != 1:
                raise ValueError(f'{ftag} not found once in {REF_EXPERIMENT}/fits.csv')
            fr = fr.iloc[0]
            p_fit = {kk: float(fr[kk]) for kk in PARAM_KEYS}
            p_fit['wKCKC_scale'] = float(fr['kc_kc_inh'])
            todo = [t for t in (f'C_{ftag}_fit', f'C_{ftag}_thr_ctrl_frac') if t not in done]
            if not todo:
                continue
            # a base folder of its own, so part B's fits (same gain and thresholds, in out_dir) do
            # not share its cache files
            vdir = out_dir / f'C_{ftag}'; vdir.mkdir(exist_ok=True)
            initial, wnorm, W, breadth = bouton_base(orn, vdir, claw_thr=thr, gain=gain)
            sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p_fit, **base_kws(thr), _wPNKC=W,
                _wAPLKC=breadth_weights(wnorm['wAPLKC'], breadth, k), _wKCAPL=wnorm['wKCAPL'],
                _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
            for tag in todo:
                t0 = time.time()
                if tag.endswith('_fit'):
                    p_on, thr_f = dict(p_fit), 1.0
                    r_on, s_on = sim(p_on)
                else:
                    extra = {kk: p_fit[kk] for kk in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')}
                    new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_fit['fixed_thr'], CTRL_FRAC, extra=extra)
                    p_on = dict(p_fit, fixed_thr=float(new_thr))
                r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                for cond, (r_, s_) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
                    r_.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
                    s_.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
                tuned[tag] = dict(p_on, claw_thr=thr, k=k, gain=gain, ref_fit=ftag, ref_experiment=REF_EXPERIMENT,
                    thr_factor_vs_fit=float(thr_f))
                tuned_path.write_text(json.dumps(tuned, indent=1))
                meta = dict(variant=tag, tag=tag, part='C', base='bouton', seed=None,
                    target_rate=(CTRL_FRAC if tag.endswith('ctrl_frac') else float(fr['rate_on'])), claw_thr=thr,
                    k=k, kc_kc_inh=p_fit['wKCKC_scale'], gain=gain, ref_fit=ftag, ref_experiment=REF_EXPERIMENT,
                    ref_fit_loss=float(fr['loss']), thr_factor_vs_fit=float(thr_f),
                    n_kcs=len(r_on), n_glomeruli=W.shape[1], rate_on=rate(r_on), rate_off=rate(r_off),
                    seconds=time.time() - t0, mean_claws_per_kc=float(kc_by_glomerulus(W).sum(axis='columns').mean()),
                    **p_on)
                meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
                finish(tag, meta, (r_on, s_on), (r_off, s_off), W)
            del sim

    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E44_operating_point.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), parts=args.parts,
        base=BOUTON, part_a_thresholds=THRS_A, part_a_targets=TARGETS_A, ctrl_frac=CTRL_FRAC,
        ref_experiment=REF_EXPERIMENT, ref_fits=REF_FITS, kc_kc_tau=TAU, loss_rate_targets=[0.05, CTRL_FRAC],
        approx_conc_note=H.APPROX_CONC_NOTE, model=model_prov)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    comp = summarize(res_dir, out_dir, loss5, loss_ctrl)
    make_figure(comp, res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
