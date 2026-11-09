#!/usr/bin/env python3
"""E62: the per-odor rate oracle: how much of the gap in rho on odor-odor distances is the per-odor
recruitment pattern.
The reference model ranks the odors' response rates at 0.47 with APL on (ceiling 0.98) and 0.80 with APL
off, and an odor's response rate shapes every distance it is in (a sparsely recruited odor sits far from
everything), so a wrong recruitment pattern caps the agreement of the odor-odor distances with the KC
matrices however right the pairwise structure is. Here the rate error is removed by construction: each
odor's ORN input vector is multiplied by a scale s_o (its pattern across glomeruli unchanged), and the 16
scales are bisected together (odors are simulated independently, so one simulation per iteration updates
all 16 brackets) until each odor's response fraction equals the KC data's per-odor fraction, at the fitted
scales and thresholds of the three reference models (E47's part A) on the real wiring; the agreement of the
odor-odor distances is the free readout.
Arms: 'on' (the scales set on the APL-on fractions against the control flies, APL off read out at the same
scales), 'off' (the scales set on the APL-off fractions against APL output blocked (TNT), APL on read out),
and 'untuned' (all scales 1: the fitted model as it is, the reference row).
Outputs: <shared>/results/E62_rate_oracle/ (scales.csv: the per-odor scales and fractions of every arm and
model with the data's fractions and the untuned model's; bisection_history.csv; stats_variants.csv with the
harness scoring columns, per_odor_response_fractions.csv, model_pair_distances_long.csv, comparison_table.md,
E62_rate_oracle.png, provenance.json) and ./outputs/E62_rate_oracle/ (per-variant responses / spike counts,
the bouton bases).
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E62_rate_oracle.py [--models thr40_gain1.5 ...] [--iters 8]
       ./venv/bin/python E62_rate_oracle.py --summary
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E47_wiring_features import MODELS, STATS, V, SCALE_KEYS, data_values, fitted_params
from E54_gain_and_lesion import MODEL_LABEL

EXPERIMENT = 'E62_rate_oracle'
ARMS = {'untuned': None, 'on': ('TNT3c', 'on'), 'off': ('TNT', 'off')}
ARM_LABEL = {'untuned': 'the fitted model (scales 1)', 'on': 'scales set on the APL-on fractions (control flies)',
    'off': 'scales set on the APL-off fractions (APL output blocked (TNT))'}
LO, HI = 0.125, 8.0  # the scales' bracket (log-space bisection)


def odor_abbrevs(orn: pd.DataFrame) -> pd.Index:
    """The odor abbreviation of each ORN column (the 'odor' level reads e.g. '2h @ -3')."""
    odors = orn.columns.get_level_values('odor') if isinstance(orn.columns, pd.MultiIndex) else orn.columns
    return pd.Index([str(x).split(' @ ')[0] for x in odors])


def scaled_orn(orn: pd.DataFrame, scales: pd.Series) -> pd.DataFrame:
    """The ORN input with each odor's column(s) multiplied by its scale (indexed by abbreviation)."""
    ab = odor_abbrevs(orn)
    missing = set(scales.index) - set(ab)
    if missing:
        raise ValueError(f'odors not in the ORN input: {sorted(missing)}')
    f = ab.map(scales).fillna(1.0).to_numpy(dtype=float)
    return orn.mul(f, axis=1)


def per_odor_fractions(responses: pd.DataFrame) -> pd.Series:
    return H.response_fractions(responses).reindex(H.APL16_ODORS)


def run_all(models: List[str], iters: int, res_dir: Path, out_dir: Path) -> None:
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    from E3_drive_dependent_gain import per_odor_extras, rate
    from E4_substrate import score, kc_by_glomerulus
    from E10_joint_fit import TAU, base_kws, bouton_base, breadth_weights, make_loss
    from E12_breadth_apl import density_excess
    from E16_claw_saturation import spearman_with_refs
    from E18_bouton_depletion import all_refs
    from E26_claw_power import sharing_deficit
    from E35_divisive_apl import suppression_stats
    from E44_operating_point import fraction_stats, losses, CTRL_FRAC
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    scales_path = res_dir / 'scales.csv'; hist_path = res_dir / 'bisection_history.csv'
    long_path = res_dir / 'model_pair_distances_long.csv'
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs()
    data_mats = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    loss5 = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=0.05)
    loss_ctrl = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=CTRL_FRAC)
    model_prov = H.model_provenance(orn_deltas=orn)
    refs = all_refs()
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame(
        {'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    scale_rows = list(pd.read_csv(scales_path).to_dict('records')) if scales_path.exists() else []
    hist_rows = list(pd.read_csv(hist_path).to_dict('records')) if hist_path.exists() else []
    base_dir = out_dir / 'bases'; base_dir.mkdir(exist_ok=True)

    def finish(tag, meta, on_out, off_out, W):
        row, on, off = score(meta, on_out, off_out, W, orn, orn_pairs, drive, data_mats, ref)
        on_p, off_p = on[V], off[V]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        row.update(suppression_stats(H.response_fractions(on_out[0]), H.response_fractions(off_out[0]), data_rates))
        row.update(fraction_stats(on_out[0], off_out[0], data_rates))
        row.update(losses(loss5, loss_ctrl, on_out, off_out))
        rows.append(row); pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index)
        fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        pd.concat({(tag, 'on'): on_p, (tag, 'acute_off'): off_p}, names=['tag', 'condition', 'odor_a', 'odor_b']).rename(
            'distance').reset_index().to_csv(long_path, index=False, mode='a', header=not long_path.exists())
        for cond, (r_, s_) in (('on', on_out), ('acute_off', off_out)):
            r_.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet'); s_.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        print(f"{tag}: rates on {row['rate_on']:.4f} off {row['rate_off']:.4f}; rho ctrl {row[V + '__on_spearman']:.3f} "
            f"tnt {row[V + '__off_spearman']:.3f}; block {row[V + '__block_fruity_small_iso']:+.3f}; "
            f"struct wt {row[V + '__on_spearman_resid_vs_remap_wt']:+.2f}; vcf on {row[V + '__on_spearman_resid_vs_vcf_hyp']:+.2f} "
            f"off {row[V + '__off_spearman_resid_vs_vcf_hyp']:+.2f}; rate rho {row['spearman_rate_on_vs_data']:.2f}/"
            f"{row['spearman_rate_off_vs_data']:.2f}; loss @ctrl {row['loss_rate_ctrl']:.3f}", flush=True)
        done.add(tag)

    for mkey in models:
        exp, ftag, thr, k, gain = MODELS[mkey]
        p_fit, fr = fitted_params(exp, ftag)
        initial, wnorm, W, breadth = bouton_base(orn, base_dir, claw_thr=thr, gain=gain)
        wAPLKC = breadth_weights(wnorm['wAPLKC'], breadth, k)

        def simulate(scales: pd.Series):
            """Both conditions at the fitted parameters with the ORN input scaled per odor."""
            sim, _ = make_fixed_param_kc_simulator(scaled_orn(orn, scales), initial_params=p_fit, **base_kws(thr), _wPNKC=W,
                _wAPLKC=wAPLKC, _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'],
                kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
            r_on, s_on = sim(dict(p_fit))
            r_off, s_off = sim(dict(p_fit, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
            del sim
            return (r_on, s_on), (r_off, s_off)

        ones = pd.Series(1.0, index=H.APL16_ODORS)
        for arm, spec in ARMS.items():
            tag = f'R_{mkey}_{arm}'
            if tag in done:
                continue
            t0 = time.time()
            if spec is None:
                scales = ones
                on_out, off_out = simulate(scales)
                n_iter = 0
            else:
                geno, cond = spec
                target = data_rates[geno].reindex(H.APL16_ODORS)
                lo = pd.Series(LO, index=H.APL16_ODORS); hi = pd.Series(HI, index=H.APL16_ODORS)
                best = None
                for it in range(iters):
                    scales = np.sqrt(lo * hi)
                    on_out, off_out = simulate(scales)
                    f = per_odor_fractions(on_out[0] if cond == 'on' else off_out[0])
                    err = (f - target).abs().mean()
                    if best is None or err < best[0]:
                        best = (err, scales.copy(), on_out, off_out)
                    for o in H.APL16_ODORS:
                        if f[o] > target[o]:
                            hi[o] = scales[o]
                        else:
                            lo[o] = scales[o]
                        hist_rows.append(dict(model=mkey, arm=arm, iteration=it, odor=o, scale=float(scales[o]), fraction=float(f[o]),
                            target=float(target[o])))
                    pd.DataFrame(hist_rows).to_csv(hist_path, index=False)
                    print(f'{tag} iteration {it}: mean |fraction - target| {err:.4f}; scales {scales.min():.2f} to {scales.max():.2f}',
                        flush=True)
                _, scales, on_out, off_out = best
                n_iter = iters
            f_on, f_off = per_odor_fractions(on_out[0]), per_odor_fractions(off_out[0])
            for o in H.APL16_ODORS:
                scale_rows.append(dict(model=mkey, arm=arm, odor=o, scale=float(scales[o]), at_bracket=bool(scales[o] <= LO * 1.01 or scales[o] >= HI / 1.01),
                    fraction_on=float(f_on[o]), fraction_off=float(f_off[o]), data_on=float(data_rates['TNT3c'][o]),
                    data_off=float(data_rates['TNT'][o]), orn_total=float(orn.loc[:, np.asarray(odor_abbrevs(orn) == o)].clip(lower=0).to_numpy().sum())))
            pd.DataFrame(scale_rows).to_csv(scales_path, index=False)
            meta = dict(variant=arm, tag=tag, part='R', model=mkey, shuffle='real', arm=arm, seed=np.nan, base='bouton', claw_thr=thr, k=k,
                kc_kc_inh=p_fit['wKCKC_scale'], gain=gain, ref_fit=ftag, ref_experiment=exp, ref_fit_loss=float(fr['loss']),
                thr_factor_vs_fit=1.0, n_iter=n_iter, n_kcs=len(on_out[0]), n_glomeruli=W.shape[1], rate_on=rate(on_out[0]),
                rate_off=rate(off_out[0]), seconds=time.time() - t0, mean_claws_per_kc=float(kc_by_glomerulus(W).sum(axis='columns').mean()),
                scale_min=float(scales.min()), scale_max=float(scales.max()), scale_geomean=float(np.exp(np.log(scales).mean())), **p_fit)
            meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
            finish(tag, meta, on_out, off_out, W)
    prov_path = res_dir / 'provenance.json'
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov['runs'] = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E62_rate_oracle.py', models=models,
        arms=list(ARMS), iters=iters, bracket=[LO, HI], model_provenance=model_prov)
    prov_path.write_text(json.dumps(prov, indent=1, default=str))


def summary(res_dir: Path) -> None:
    st = pd.read_csv(res_dir / 'stats_variants.csv')
    if 'error' in st.columns:
        st = st[~(st.error.astype(str).str.len() > 3)]
    sc = pd.read_csv(res_dir / 'scales.csv')
    data = data_values()
    out = []
    arms = [a for a in ARMS if (st.arm == a).any()]
    rows = ['**The reference models at their fitted scales and thresholds on the real wiring, with each odor\'s ORN input scaled so that the '
        'model\'s per-odor response fractions equal the KC data\'s (the oracle), against the fitted model as it is**', '',
        '| | | ' + ' | '.join(f'{MODEL_LABEL[m]} {{span={len(arms)}}}' for m in MODELS if (st.model == m).any()) + ' |',
        '| statistic | KC data | ' + ' | '.join(' | '.join(ARM_LABEL[a] for a in arms) for m in MODELS if (st.model == m).any()) + ' |',
        '|---|---|' + '---|' * (len(arms) * sum((st.model == m).any() for m in MODELS))]
    for col, label, fmt, dkey in STATS:
        if col not in st.columns:
            continue
        d = '-'
        if dkey and dkey in data:
            d = format(data[dkey], fmt)
            if 'ceiling' in dkey:
                d = f'ceiling {d}'
        cells = []
        for m in MODELS:
            if not (st.model == m).any():
                continue
            for a in arms:
                v = st[(st.model == m) & (st.arm == a)][col]
                cells.append(format(float(v.iloc[0]), fmt) if len(v) == 1 and pd.notna(v.iloc[0]) else '-')
        rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
    out.append('\n'.join(rows))
    # the scales
    tuned = [a for a in arms if a != 'untuned']
    models = [m for m in MODELS if (sc.model == m).any()]
    rows = ['**The per-odor input scales the oracle needs (a scale above 1 means the odor\'s ORN input is too weak for the KC data\'s recruitment '
        'at the fitted parameters), with the fitted model\'s response fractions and the data\'s**', '',
        '| | KC data {span=2} | ' + ' | '.join(f'{MODEL_LABEL[m]} {{span={2 + len(tuned)}}}' for m in models) + ' |',
        '| odor | control flies | APL output blocked (TNT) | ' + ' | '.join('fitted model, APL on | fitted model, APL off | ' + ' | '.join(
            f'scale, {a} arm' for a in tuned) for m in models) + ' |',
        '|---|---|---|' + '---|' * ((2 + len(tuned)) * len(models))]
    for o in H.APL16_ODORS:
        d_on = sc[(sc.odor == o)].data_on.iloc[0]; d_off = sc[(sc.odor == o)].data_off.iloc[0]
        cells = []
        for m in models:
            u = sc[(sc.model == m) & (sc.arm == 'untuned') & (sc.odor == o)]
            cells += [f'{u.fraction_on.iloc[0]:.3f}' if len(u) else '-', f'{u.fraction_off.iloc[0]:.3f}' if len(u) else '-']
            for a in tuned:
                t = sc[(sc.model == m) & (sc.arm == a) & (sc.odor == o)]
                cells.append((f'{t.scale.iloc[0]:.2f}' + (' (at the bracket)' if bool(t.at_bracket.iloc[0]) else '')) if len(t) else '-')
        rows.append(f'| {o} | {d_on:.3f} | {d_off:.3f} | ' + ' | '.join(cells) + ' |')
    out.append('\n'.join(rows))
    (res_dir / 'comparison_table.md').write_text('\n\n'.join(out) + '\n')
    figure(st, sc, data, res_dir)
    print('\n\n'.join(out))


def figure(st: pd.DataFrame, sc: pd.DataFrame, data: Dict[str, float], res_dir: Path) -> None:
    plt = H._plt()
    from matplotlib.lines import Line2D
    colors = {'thr70_gain1.5': '#1f5fbf', 'thr40_gain1.5': '#d1662a', 'thr70_gain1.25': '#2a9d5c'}
    models = [m for m in MODELS if (st.model == m).any()]
    arms = [a for a in ARMS if (st.arm == a).any()]
    panels = [(f'{V}__on_spearman', 'ρ on odor-odor distances, APL on', 'ctrl_ceiling'),
        (f'{V}__off_spearman', 'ρ on odor-odor distances, APL off', 'tnt_ceiling'),
        ('spearman_rate_on_vs_data', 'response rate ρ, APL on', 'ctrl_rate_rho_ceiling'),
        ('spearman_rate_off_vs_data', 'response rate ρ, APL off', 'tnt_rate_rho_ceiling'),
        (f'{V}__block_fruity_small_iso', 'fruity × small-ester block', 'block'),
        (f'{V}__off_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL off', 'tnt_vcf')]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.ravel()
    offs = {m: (i - (len(models) - 1) / 2) * 0.2 for i, m in enumerate(models)}
    for ax, (col, label, dkey) in zip(axes[:6], panels):
        for m in models:
            for i, a in enumerate(arms):
                v = st[(st.model == m) & (st.arm == a)][col].astype(float)
                if v.empty or pd.isna(v.iloc[0]):
                    continue
                ax.scatter([i + offs[m]], [v.iloc[0]], color=colors.get(m, 'k'), s=40, zorder=3)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='k', ls='--', lw=1)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=20, ha='right', fontsize=7)
        ax.set_title(label, fontsize=9); ax.grid(axis='y', alpha=0.25)
    for ax, arm in zip(axes[6:], [a for a in arms if a != 'untuned']):
        for m in models:
            t = sc[(sc.model == m) & (sc.arm == arm)].set_index('odor').reindex(H.APL16_ODORS)
            ax.plot(range(len(H.APL16_ODORS)), np.log2(t.scale), 'o-', color=colors.get(m, 'k'), ms=4, lw=1)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(range(len(H.APL16_ODORS))); ax.set_xticklabels(H.APL16_ODORS, rotation=60, ha='right', fontsize=7)
        ax.set_title(f'log2 of the per-odor input scale, {ARM_LABEL[arm]}', fontsize=9); ax.grid(axis='y', alpha=0.25)
    handles = [Line2D([], [], marker='o', ls='', color=colors.get(m, 'k'), label=MODEL_LABEL[m]) for m in models]
    handles.append(Line2D([], [], ls='--', color='k', label='KC data (target or ceiling)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle('The per-odor rate oracle: the reference models with each odor\'s ORN input scaled so that the per-odor response fractions '
        'equal the KC data\'s, against the fitted models as they are', fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(res_dir / f'{EXPERIMENT}.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--models', nargs='+', default=list(MODELS), help='keys of E47\'s MODELS')
    ap.add_argument('--iters', type=int, default=8, help='bisection iterations (each one simulation of both conditions)')
    ap.add_argument('--summary', action='store_true', help='only (re)write the tables and the figure')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    if args.summary:
        summary(res_dir); return
    run_all(args.models, args.iters, res_dir, out_dir)
    summary(res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
