#!/usr/bin/env python3
"""E39: decomposing the oracle glomerulus gain at the combined model's fitted parameters.

The reference model (E34's capped structure fit thr40_gain1.5_k2_cap3: bouton base, 40 Hz
per-claw threshold, breadth-weighted APL with exponent 2, KC-KC inhibition, APL->bouton scale
capped at 3x, PN->claw gain 1.5 on the eight oracle glomeruli VA7m, DL1, D, DM1, DP1m, VC1, DC2,
DM5) and its 70 Hz counterpart (thr70_gain1.5_k2_cap3) carry the gain as one set. Here the gain
1.5 is applied to subsets of the set, every fitted scale held (APL->KC, KC->APL, APL->bouton,
bouton->APL, KC-KC): none, the full set, each glomerulus alone, all but one, and the per-odor
sets of the glomerulus-gain experiment (E27: ms VA7m / DL1 / D; paa and B-myr DM1, which is the
DM1-alone variant; euc and -aPine DP1m / VC1; 1-8ol DC2 / DM5). For each variant, as the joint
fit built its model (E10_joint_fit.bouton_base): the breadth weights come from an APL-off run of
that variant's gained wiring at the base's tuned parameters; then the somatic threshold is
re-tuned (`bisect_threshold`) to the full-set fit's APL-on response rate, and APL is removed
acutely (APL->KC and APL->bouton weights zero). The full-set variant reproduces the fit (checked
against E34's saved outputs). The APL-on run at the fit's own threshold, before re-tuning, is
kept too (its rate and per-odor fractions).

Scored as the combined-model experiment (E28) plus the joint fit's extended loss (registry
targets since a103914) and a glomerulus x statistic table of marginal contributions (gain on that
glomerulus alone minus no gain; full set minus all-but-that-glomerulus).

Outputs: <shared>/results/E39_gain_decomposition/ and ./outputs/E39_gain_decomposition/.
Usage: OMP_NUM_THREADS=8 nice -n 10 ./venv/bin/python E39_gain_decomposition.py [--bases thr40 thr70]
       [--variants full none ...] [--figure-only]
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
from E3_drive_dependent_gain import per_odor_extras, rate
from E4_substrate import score
from E10_joint_fit import base_kws, bouton_base, breadth_weights, make_loss
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E26_claw_power import sharing_deficit, deficit_table, WEAK
from E27_glomerulus_gain import ORACLE, gained
from E35_divisive_apl import suppression_stats

EXPERIMENT = 'E39_gain_decomposition'
GAIN = 1.5
K = 2.0
TAU = 0.05
BASES = {'thr40': 'thr40_gain1.5_k2_cap3', 'thr70': 'thr70_gain1.5_k2_cap3'}
FIT_PARAMS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
SETS = {'ms': ['VA7m', 'DL1', 'D'], 'terp': ['DP1m', 'VC1'], '1-8ol': ['DC2', 'DM5']}
ESTERS = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep']
V = H.PRIMARY_MODEL_RDM + '__'
# statistics for the marginal-contribution table: (column, label)
MARGINAL_STATS = ([(f'foff_{o}', f'{o} APL-off fraction') for o in WEAK]
    + [(f'fon_{o}', f'{o} APL-on fraction') for o in ESTERS]
    + [('spearman_rate_on_vs_data', 'response rate ρ, APL on'), ('spearman_rate_off_vs_data', 'response rate ρ, APL off'),
       (V + 'on_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL on'),
       (V + 'off_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL off'),
       (V + 'on_spearman_resid_vs_remap_wt', 'structure vs wild type, APL on'),
       (V + 'block_fruity_small_iso', 'block'), ('on_excess_diff', 'APL-on excess, block - other'),
       (V + 'on_spearman', 'ρ on odor-odor distances, APL on'), (V + 'off_spearman', 'ρ on odor-odor distances, APL off'),
       (V + 'iso_r2', 'isotonic R²'), ('xl_loss', 'extended loss (lower is better)')])


def variants() -> dict:
    """Variant name -> glomeruli given the gain (the full set first: it sets the rate target)."""
    v = {'full': list(ORACLE), 'none': []}
    v.update({f'only_{g}': [g] for g in ORACLE})
    v.update({f'set_{n}': gs for n, gs in SETS.items()})
    v.update({f'loo_{g}': [x for x in ORACLE if x != g] for g in ORACLE})
    return v


def marginals(stats: pd.DataFrame, base: str) -> pd.DataFrame:
    """Glomerulus x (statistic, kind): kind 'alone' = only_<g> - none, 'needed' = full - loo_<g>."""
    s = stats[stats.base == base].set_index('variant')
    out = {}
    for col, lab in MARGINAL_STATS:
        if col not in s:
            continue
        for g in ORACLE:
            out.setdefault((lab, 'alone'), {})[g] = s.at[f'only_{g}', col] - s.at['none', col] if f'only_{g}' in s.index and 'none' in s.index else np.nan
            out.setdefault((lab, 'needed'), {})[g] = s.at['full', col] - s.at[f'loo_{g}', col] if f'loo_{g}' in s.index and 'full' in s.index else np.nan
    m = pd.DataFrame(out)
    m.columns.names = ['statistic', 'kind']
    m.index.name = 'glomerulus'
    return m


def make_figure(stats: pd.DataFrame, res_dir) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cols = [c for c, _ in MARGINAL_STATS if c in stats]
    labs = [l for c, l in MARGINAL_STATS if c in stats]
    bases = [b for b in BASES if b in set(stats.base)]
    order = list(variants())
    fig, axes = plt.subplots(1, len(bases), figsize=(11 * len(bases), 10), squeeze=False)
    for ax, b in zip(axes[0], bases):
        s = stats[stats.base == b].set_index('variant')
        rows = [v for v in ['none'] + [f'only_{g}' for g in ORACLE] + [f'set_{n}' for n in SETS]
            + [f'loo_{g}' for g in ORACLE] + ['full'] if v in s.index]
        vals = s.loc[rows, cols].astype(float)
        # color: each column scaled from its value without the gain (0) to its value with the full set (1)
        ref0 = vals.loc['none'] if 'none' in rows else vals.min()
        ref1 = vals.loc['full'] if 'full' in rows else vals.max()
        span = (ref1 - ref0).replace(0, np.nan)
        z = ((vals - ref0) / span).clip(-0.5, 1.5)
        ax.imshow(z.to_numpy(), cmap='PuOr_r', vmin=-0.5, vmax=1.5, aspect='auto')
        for i in range(len(rows)):
            for j in range(len(cols)):
                x = vals.iat[i, j]
                ax.text(j, i, '' if np.isnan(x) else f'{x:.2f}', ha='center', va='center', fontsize=6.5)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(labs, rotation=60, ha='right', fontsize=7.5)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
        ax.set_title(f'{BASES[b]}: gain {GAIN:g} on subsets of the oracle set\n'
            '(color: each column from its no-gain value, white-ish 0, to its full-set value, 1)', fontsize=9)
    fig.suptitle('E39: decomposing the oracle glomerulus gain at the fitted combined models '
        '(threshold re-tuned to the fit\'s APL-on rate; APL off acutely)', fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / 'E39_gain_decomposition.png', dpi=130); plt.close(fig)


# the statistics shown in the marginal-contribution figure (labels as in MARGINAL_STATS)
FIG_STATS = ['ms APL-off fraction', 'paa APL-off fraction', 'B-myr APL-off fraction', 'euc APL-off fraction',
    '1-8ol APL-off fraction', '2h APL-on fraction', 'IaA APL-on fraction', '2-but APL-on fraction',
    'response rate ρ, APL on', 'response rate ρ, APL off', 'natural-source alignment, APL off',
    'structure vs wild type, APL on', 'block', 'ρ on odor-odor distances, APL on', 'ρ on odor-odor distances, APL off',
    'extended loss (lower is better)']


def make_marginal_figure(stats: pd.DataFrame, res_dir) -> None:
    """Per base, glomerulus x statistic heatmaps of the two marginal contributions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bases = [b for b in BASES if b in set(stats.base)]
    fig, axes = plt.subplots(len(bases), 2, figsize=(17, 5.2 * len(bases)), squeeze=False)
    for i, b in enumerate(bases):
        m = marginals(stats, b)
        labs = [l for l in FIG_STATS if (l, 'alone') in m.columns]
        for j, (kind, title) in enumerate((('alone', 'gain on this glomerulus alone, minus no gain'),
                ('needed', 'full set, minus the full set without this glomerulus'))):
            ax = axes[i, j]
            d = m.xs(kind, axis=1, level='kind')[labs].astype(float)
            # each column scaled by its largest absolute value over both kinds, so colors compare within a column
            scale = m.loc[:, [(l, k) for l in labs for k in ('alone', 'needed')]].abs().T.groupby(level=0).max().max(axis=1)
            z = d / scale.reindex(labs).replace(0, np.nan).to_numpy()
            ax.imshow(z.to_numpy(), cmap='PuOr_r', vmin=-1, vmax=1, aspect='auto')  # orange = increase
            for r in range(d.shape[0]):
                for c in range(d.shape[1]):
                    x = d.iat[r, c]
                    ax.text(c, r, '' if np.isnan(x) else f'{x:+.2f}', ha='center', va='center', fontsize=7)
            ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=55, ha='right', fontsize=8)
            ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index, fontsize=9)
            ax.set_title(f'{BASES[b]}: {title}', fontsize=9.5)
    fig.suptitle('E39: what each oracle glomerulus contributes at the fitted combined models (gain 1.5; '
        'threshold re-tuned to the fit\'s APL-on response rate; APL off acutely)\n'
        'color: sign and size of the change within each column (orange: increase, purple: decrease)', fontsize=10.5)
    fig.tight_layout(); fig.savefig(res_dir / 'E39_marginals.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bases', nargs='+', default=list(BASES), choices=list(BASES))
    ap.add_argument('--variants', nargs='+', default=None, help='subset of the variants (default: all)')
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        st = pd.DataFrame(rows)
        make_figure(st, res_dir); make_marginal_figure(st, res_dir)
        for b in sorted(set(st.base)):
            marginals(st, b).to_csv(res_dir / f'marginals_{b}.csv')
        return
    done = {r['tag'] for r in rows}
    allv = variants()
    todo = {v: allv[v] for v in (args.variants or allv)}
    orn = H.apl16_orn_deltas(); orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs(); refs = all_refs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    loss_fn = make_loss(data, data_rates, orn_pairs, drive, ref, structure=True)
    fits = pd.read_csv(H.results_dir('E34_gain_bounded') / 'fits.csv').set_index('tag')
    e34_out = H.outputs_dir('E34_gain_bounded')
    tuned = json.loads((out_dir / 'tuned_params.json').read_text()) if (out_dir / 'tuned_params.json').exists() else {}
    from al_analysis.mb_model import fit_mb_model, make_fixed_param_kc_simulator
    # code provenance at the start: the checkout can move (a pull) while the run goes on
    try:
        model_prov = H.model_provenance(orn_deltas=orn)
    except Exception as e:  # keep the commit hash at least
        model_prov = dict(al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True,
            text=True).stdout.strip(), model_provenance_error=repr(e)[:200])
    pairs = {}; deficits = []
    for b in args.bases:
        fit_tag = BASES[b]
        fit = fits.loc[fit_tag]
        claw_thr = float(fit['claw_thr'])
        assert float(fit['gain']) == GAIN and float(fit['k']) == K, fit_tag
        p_fit = {kk: float(fit[kk]) for kk in FIT_PARAMS}
        p_fit['wKCKC_scale'] = float(fit['kc_kc_inh'])
        kws = base_kws(claw_thr)
        # the base's tuned parameters, normalized APL weights and un-gained wiring (as the fit's)
        initial, wnorm, wPNKC0, _ = bouton_base(orn, out_dir, claw_thr=claw_thr)
        rate_target = tuned.get(f'{b}__rate_target')
        for v, gloms in todo.items():
            tag = f'{b}__{v}'
            if tag in done:
                continue
            if rate_target is None and v != 'full':
                raise RuntimeError('run the full-set variant first: it sets the rate target')
            t0 = time.time()
            try:
                W = gained(wPNKC0, gloms, GAIN) if gloms else wPNKC0
                # the variant's own APL-off breadth at the base's tuned parameters (bouton_base)
                r_b, s_b, _, _ = fit_mb_model(orn, fixed_thr=initial['fixed_thr'], wAPLKC=0.0, wKCAPL=initial['wKCAPL_scale'],
                    wAPLPN=0.0, wPNAPL=initial['wPNAPL_scale'], **kws, _wPNKC=W)
                breadth = pd.Series(s_b.astype(float).sum(axis='columns').to_numpy(), index=s_b.index.get_level_values(H.KC_ID_LEVEL))
                check = {}
                if v in ('full', 'none'):
                    cached = (e34_out / f'gain{GAIN:g}_thr{claw_thr:g}_base_acute_off_spike_counts.parquet' if v == 'full' else
                        H.shared_dir() / 'data' / 'hal_outputs' / 'E16_claw_saturation' / f'bouton_thr{claw_thr:g}_acute_off_spike_counts.parquet')
                    if cached.exists():
                        c = pd.read_parquet(cached)
                        check['breadth_max_abs_diff_vs_saved'] = float(np.abs(c.astype(float).to_numpy() - s_b.astype(float).to_numpy()).max())
                sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p_fit, **kws, _wPNKC=W,
                    _wAPLKC=breadth_weights(wnorm['wAPLKC'], breadth, K), _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'],
                    _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
                r_fix, s_fix = sim(p_fit)
                if v == 'full':
                    saved = e34_out / f'{fit_tag}_on_responses.parquet'
                    if saved.exists():
                        c = pd.read_parquet(saved)
                        check['fit_responses_mismatch_vs_saved'] = int((c.astype(float).to_numpy() != r_fix.astype(float).to_numpy()).sum())
                    rate_target = rate(r_fix)
                    tuned[f'{b}__rate_target'] = rate_target
                    print(f'{tag}: fit reproduced at rate {rate_target:.5f} (fits.csv {float(fit["rate_on"]):.5f}); {check}', flush=True)
                extra = {kk: p_fit[kk] for kk in p_fit if kk != 'fixed_thr'}
                thr, thr_f, r_on, s_on = bisect_threshold(sim, p_fit['fixed_thr'], rate_target, extra=extra)
                p_on = dict(p_fit, fixed_thr=float(np.mean(thr)))
                r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                del sim
            except Exception as e:
                print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                rows.append(dict(variant=v, tag=tag, base=b, error=repr(e)[:300])); pd.DataFrame(rows).to_csv(stats_path, index=False); continue
            for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off)), ('on_fitthr', (r_fix, s_fix)), ('base_acute_off', (r_b, s_b))):
                r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet'); sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
            tuned[tag] = dict(p_on, claw_thr=claw_thr, k=K, gain=GAIN, glomeruli=gloms, thr_factor=float(thr_f), fit=fit_tag,
                base_params=initial, kc_kc_tau=TAU)
            (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1, default=float))
            meta = dict(variant=v, tag=tag, base=b, fit=fit_tag, seed=None, target_rate=rate_target, claw_thr=claw_thr, k=K, gain=GAIN,
                glomeruli=','.join(gloms), n_gained=len(gloms), thr_factor=float(thr_f), n_kcs=len(r_on), n_glomeruli=W.shape[1],
                rate_on=rate(r_on), rate_off=rate(r_off), rate_on_fitthr=rate(r_fix), base_rate_off=rate(r_b),
                seconds=time.time() - t0, kc_kc_tau=TAU, **{kk: float(np.mean(vv)) for kk, vv in p_on.items()}, **check)
            meta['kc_kc_inh'] = p_on['wKCKC_scale']
            meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
            row, on, off = score(meta, (r_on, s_on), (r_off, s_off), W, orn, orn_pairs, drive, data, ref)
            on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
            row.update(per_odor_extras(r_on, r_off, on_p, off_p, drive, data_rates))
            for cond, sp in (('on', s_on), ('off', s_off)):
                row.update({f'{cond}_{kk}': vv for kk, vv in density_excess(sp).items()})
            row['on_excess_diff'] = row['on_excess_block'] - row['on_excess_other_between']
            row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(s_off, data_pairs))
            fon, foff, ffix = H.response_fractions(r_on), H.response_fractions(r_off), H.response_fractions(r_fix)
            row.update(suppression_stats(fon, foff, data_rates))
            row.update({f'fon_{o}': float(fon[o]) for o in fon.index}); row.update({f'foff_{o}': float(foff[o]) for o in foff.index})
            row.update({f'xl_{kk}': vv for kk, vv in loss_fn({'on': (r_on, s_on), 'off': (r_off, s_off)}).items()})
            d = deficit_table(s_off, data_pairs); d.insert(0, 'tag', tag); deficits.append(d)
            rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            for name, f in (('on', fon), ('acute_off', foff), ('on_fitthr', ffix)):
                fracs[f'{tag}_{name}'] = f.reindex(fracs.index)
            fracs.to_csv(fracs_path)
            wk = ' '.join(f'{o}:{foff[o]:.3f}' for o in WEAK)
            es = ' '.join(f'{o}:{fon[o]:.3f}' for o in ['2h', 'IaA', '2-but'])
            print(f"{tag} ({len(gloms)} gained): thr x{float(thr_f):.3f}, rate on {meta['rate_on']:.4f} (at the fit's threshold {meta['rate_on_fitthr']:.4f}) "
                f"off {meta['rate_off']:.4f} (x{meta['rate_ratio_off_over_on']:.2f}); weak off [{wk}]; esters on [{es}]; "
                f"rate rho on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; "
                f"vcf on/off {row.get(V + 'on_spearman_resid_vs_vcf_hyp', np.nan):.2f}/{row.get(V + 'off_spearman_resid_vs_vcf_hyp', np.nan):.2f}; "
                f"wt {row.get(V + 'on_spearman_resid_vs_remap_wt', np.nan):.2f}; block {row[V + 'block_fruity_small_iso']:+.3f}; "
                f"rho ctrl/TNT {row[V + 'on_spearman']:.3f}/{row[V + 'off_spearman']:.3f}; iso {row[V + 'iso_r2']:.3f}; loss {row['xl_loss']:.3f}; {meta['seconds']:.0f}s", flush=True)
            pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
                res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
            done.add(tag)
        st = pd.DataFrame(rows)
        if len(st) and b in set(st.base):
            marginals(st, b).to_csv(res_dir / f'marginals_{b}.csv')
    if deficits:
        pd.concat(deficits).to_csv(res_dir / 'pair_deficits_long.csv', index=False, mode='a', header=not (res_dir / 'pair_deficits_long.csv').exists())
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E39_gain_decomposition.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), bases={b: BASES[b] for b in args.bases},
        fits_source='results/E34_gain_bounded/fits.csv', gain=GAIN, k=K, tau=TAU, oracle_glomeruli=ORACLE, per_odor_sets=SETS,
        variants={v: todo[v] for v in todo}, base_kws=base_kws(None), approx_conc_note=H.APPROX_CONC_NOTE)
    prov.update(model_prov)
    (res_dir / f'provenance_{"_".join(args.bases)}.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), res_dir); make_marginal_figure(pd.DataFrame(rows), res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
