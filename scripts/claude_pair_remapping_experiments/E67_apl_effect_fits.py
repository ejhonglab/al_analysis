#!/usr/bin/env python3
"""E67: the APL-effect fits. Tables and figures for the joint fits whose loss leads with the APL-effect loss
(E10_joint_fit.py --loss apl), each beside its start, under both definitions of the model's side.

The fits run with `E10_joint_fit.py --experiment E67_apl_effect_fits --loss apl --apl-readout raw|cal|both ...`
(outputs/E67_apl_effect_fits/run_viki_e67.sh for stage 1). For every fit in results/E67_apl_effect_fits/fits.csv,
this script re-reads its saved spike counts and responses (outputs/E67_apl_effect_fits/, or the model-output
archive once archived) and computes, under raw spike counts (cosine) and under E65's calibrated saturating readout
(K set on the fit's APL-on counts to the control flies' responders-only participation ratio, applied unchanged to
APL off; `kc_remapping_harness.calibrate_readout`): the APL-effect loss with its mean, per-odor and pair-specific
parts (`apl_effect_decomposition`) and ρ on the per-pair APL effect; the loss against the data read with
sub-threshold values zeroed and noise-corrected; the square-root readout and the binarized readout against the
binarized data (checks); the block under raw counts, the readout and binarized; the mean distances; the
participation ratio among responders with APL off under the readout (a prediction; data 8.3 %); the response rates,
their ratio and the response rate ρ in both conditions against the realism bands; the number of odors per KC in both
conditions; the two ρ's on odor-odor distances; the natural-source alignment (APL on / off / on minus off); the
fitted parameters. Each fit's start is its history's first evaluation (the loss's own statistics) and, where the
start is an archived model, the same full row from the archive.

Outputs (results/E67_apl_effect_fits/): comparison.csv (one row per fit and start), comparison_table.md,
E67_fit_histories.png, E67_mean_vs_pattern.png, provenance_comparison.json.
Usage: ./venv/bin/python E67_apl_effect_fits.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E67_apl_effect_fits'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
OPK_BINS = ('0', '1', '2_3', '4_7', '8plus')
# archived starts: (label, folder, tag); a fit's start is recognised from its init_from / init_params
STARTS = {
    'A': ('the reference model (eight at 1.5, 40 Hz, 6.8 %)', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'),
    'C': ('the first joint fit, k 3 (5 %)', 'E10_joint_fit', 'k3'),
    'C2': ('the first joint fit, k 2 (5 %)', 'E10_joint_fit', 'k2'),
    'E': ('the no-gain two-ρ fit (40 Hz, 6.8 %)', 'E63_rho_fit', 'thr40_rate0.068_rho4_k2_cap3'),
}


def start_key(row: pd.Series) -> str:
    """Which start a fits.csv row came from (STARTS key, 'D' for the DM1-alone start, 'C2*' for C2's fitted
    parameters given as --init-params in stage 3, 'from <tag>' for an exact start at another E67 fit)."""
    ip = row.get('init_params')
    if isinstance(ip, str) and 'E44_thr70' in ip:
        return 'D'
    if isinstance(ip, str) and 'E67_C2_cal' in ip:
        return 'C2*' + ip.split('E67_C2_cal', 1)[1]
    src = (row.get('init_from_experiment'), row.get('init_from'))
    if src[0] == EXPERIMENT:
        return f'from {src[1]}'
    for key, (_, folder, tag) in STARTS.items():
        if src == (folder, tag):
            return key
    return '?'


def load_outputs(folder_or_dir, tag: str) -> Optional[dict]:
    """{'on': (responses, spike counts), 'off': ...} from the archive folder name or a directory."""
    d = Path(folder_or_dir)
    if not d.is_absolute():
        d = ARCHIVE / str(folder_or_dir)
    out = {}
    for c, f in (('on', 'on'), ('off', 'acute_off')):
        ps = [d / f'{tag}_{f}_{kind}.parquet' for kind in ('responses', 'spike_counts')]
        if not all(p.exists() for p in ps):
            return None
        out[c] = tuple(pd.read_parquet(p) for p in ps)
    return out


def odors_per_kc(resp: pd.DataFrame) -> np.ndarray:
    n = (resp.to_numpy(dtype=float) > 0.5).sum(axis=1)
    return np.array([np.mean(n == 0), np.mean(n == 1), np.mean((n >= 2) & (n <= 3)), np.mean((n >= 4) & (n <= 7)), np.mean(n >= 8)])


_REFS: dict = {}


def refs() -> dict:
    if not _REFS:
        _REFS.update(ctrl=H.yang_mean_rdm('TNT3c'), tnt=H.yang_mean_rdm('TNT'), ref66=H.reference_pairs(),
            graded=H.yang_apl_effect(), thresholded=H.yang_apl_effect(values='thresholded'),
            noise_corrected=H.yang_apl_effect(values='noise_corrected'), binary=H.yang_apl_effect(binarize=1.0),
            rates={g: H.yang_response_fractions(g).mean().reindex(H.APL16_ODORS) for g in ('TNT3c', 'TNT')},
            opk={g: np.array([H.data_target(f'{n}_odors_per_kc_{b}') for b in OPK_BINS]) for g, n in (('TNT3c', 'ctrl'), ('TNT', 'tnt'))},
            cal_target=H.data_target('ctrl_participation_ratio_resp'))
    return _REFS


def full_row(o: dict) -> Dict[str, float]:
    """Every statistic of one model from its APL-on and APL-off responses and spike counts."""
    R = refs()
    (r_on, s_on), (r_off, s_off) = o['on'], o['off']
    row = {}
    K = H.calibrate_readout(s_on, 'saturating', R['cal_target'], stat='participation_ratio', responders_min=2.0)
    row['K'] = float(K)
    row['K_at_bound'] = bool(abs(H.readout_stat(s_on, 'saturating', K, 'participation_ratio', 2.0) - R['cal_target']) > 0.02 * R['cal_target'])
    dist = {'raw': (H.readout_pair_distances(s_on, 'raw'), H.readout_pair_distances(s_off, 'raw')),
            'cal': (H.readout_pair_distances(s_on, 'saturating', K), H.readout_pair_distances(s_off, 'saturating', K)),
            'sqrt': (H.readout_pair_distances(s_on, 'sqrt'), H.readout_pair_distances(s_off, 'sqrt')),
            'binary': (H.readout_pair_distances(s_on, 'binary', 2.0), H.readout_pair_distances(s_off, 'binary', 2.0))}
    for name in ('raw', 'cal'):
        on, off = dist[name]
        dec, _ = H.apl_effect_decomposition(on, off, R['graded'])
        row.update({f'{name}_loss': dec['apl_effect_rmse'], f'{name}_mean': dec['apl_effect_mean'], f'{name}_pattern': dec['apl_effect_rmse_centered'],
            f'{name}_per_odor': dec['apl_effect_per_odor'], f'{name}_pair_specific': dec['apl_effect_pair_specific'], f'{name}_rho': dec['apl_effect_spearman']})
        for v in ('thresholded', 'noise_corrected'):
            row[f'{name}_loss_vs_{v}'] = H.apl_effect_stats(on, off, R[v])['apl_effect_rmse']
        row[f'{name}_mean_dist_on'], row[f'{name}_mean_dist_off'] = float(on.mean()), float(off.mean())
        row[f'{name}_rho_on'] = H.spearman(on, R['ctrl'])
        row[f'{name}_rho_off'] = H.spearman(off, R['tnt'])
        row[f'{name}_vcf_on'] = H.compare_to_references(on, R['ref66'])['spearman_resid_vs_vcf_hyp']
        row[f'{name}_vcf_off'] = H.compare_to_references(off, R['ref66'])['spearman_resid_vs_vcf_hyp']
    for name in ('raw', 'cal', 'binary'):
        try:
            st, _ = H.compare_on_off(*dist[name])
            row[f'{name}_block'] = float(st['block_fruity_small_iso'])
        except Exception:
            row[f'{name}_block'] = np.nan
    row['sqrt_loss'] = H.apl_effect_stats(*dist['sqrt'], R['graded'])['apl_effect_rmse']
    row['sqrt_mean'] = H.apl_effect_stats(*dist['sqrt'], R['graded'])['apl_effect_mean']
    b = H.apl_effect_stats(*dist['binary'], R['binary'])
    row['binary_loss_vs_binary'], row['binary_mean'] = b['apl_effect_rmse'], b['apl_effect_mean']
    row['cal_pr_resp_off'] = H.readout_stat(s_off, 'saturating', K, 'participation_ratio', 2.0)
    row['rate_on'] = float(r_on.to_numpy(dtype=float).mean())
    row['rate_off'] = float(r_off.to_numpy(dtype=float).mean())
    row['ratio'] = row['rate_off'] / max(row['rate_on'], 1e-9)
    for c, r, g in (('on', r_on, 'TNT3c'), ('off', r_off, 'TNT')):
        fr = H.response_fractions(r).reindex(H.APL16_ODORS)
        row[f'rate_rho_{c}'] = H.spearman(fr, R['rates'][g])
        opk = odors_per_kc(r)
        row.update({f'opk_{c}_{b}': float(v) for b, v in zip(OPK_BINS, opk)})
        row[f'opk_{c}_dist'] = float(0.5 * np.abs(opk - R['opk'][g]).sum())
    flags = []
    if not 0.045 <= row['rate_on'] <= 0.085:
        flags.append(f"APL-on response rate {row['rate_on']:.3f}")
    if not 1.6 <= row['ratio'] <= 2.9:
        flags.append(f"APL-off / APL-on response rate {row['ratio']:.2f}")
    if row['rate_rho_on'] < 0.37:
        flags.append(f"response rate ρ, APL on, {row['rate_rho_on']:.2f}")
    # the odors-per-KC band (Tom, 2026-11-09 23:40): total variation distance from the flies' distribution
    for c, band in (('on', 'ctrl_odors_per_kc_tv_max'), ('off', 'tnt_odors_per_kc_tv_max')):
        if row[f'opk_{c}_dist'] > H.data_target(band):
            flags.append(f"odors per KC, APL {c}, {row[f'opk_{c}_dist']:.2f}")
    row['realism_flag'] = '; '.join(flags)
    return row


def history_start(res_dir: Path, tag: str) -> Dict[str, float]:
    """The loss's own statistics at a fit's first evaluation (its start), renamed to full_row's columns."""
    h = pd.read_csv(res_dir / f'history_{tag}.csv')
    r = h.iloc[0]
    m = {'raw_loss': 'apl_raw_rmse', 'raw_mean': 'apl_raw_mean', 'raw_pattern': 'apl_raw_pattern', 'raw_per_odor': 'apl_raw_per_odor',
        'raw_pair_specific': 'apl_raw_pair_specific', 'raw_rho': 'apl_raw_rho', 'cal_loss': 'apl_cal_rmse', 'cal_mean': 'apl_cal_mean',
        'cal_pattern': 'apl_cal_pattern', 'cal_per_odor': 'apl_cal_per_odor', 'cal_pair_specific': 'apl_cal_pair_specific',
        'cal_rho': 'apl_cal_rho', 'K': 'apl_cal_K', 'cal_block': 'block_cal', 'raw_block': 'block', 'rate_on': 'rate_on', 'rate_off': 'rate_off',
        'ratio': 'ratio', 'rate_rho_on': 'po_on_spearman', 'rate_rho_off': 'po_off_spearman', 'raw_rho_on': 'on_spearman',
        'raw_rho_off': 'off_spearman', 'raw_vcf_off': 'vcf_off', 'opk_off_dist': 'opk_off_dist', 'loss': 'loss'}
    return {k: float(r[v]) for k, v in m.items() if v in r.index}


def main() -> None:
    res_dir = H.results_dir(EXPERIMENT)
    out_dir = H.outputs_dir(EXPERIMENT)
    fits = pd.read_csv(res_dir / 'fits.csv')
    rows = []
    done_starts = set()
    for _, f in fits.iterrows():
        key = start_key(f)
        o = load_outputs(out_dir, f['tag']) or load_outputs(EXPERIMENT, f['tag'])
        if o is None:
            print('no saved outputs for', f['tag'])
            continue
        params = {c: f[c] for c in ('fixed_thr', 'wAPLKC_scale', 'wAPLPN_scale', 'kc_kc_inh', 'thr_factor', 'apl_factor_vs_e3', 'aplpn_factor_vs_e3') if c in f.index}
        # the unfitted bouton->APL / KC->APL input balance each start brings (unresolved_issues.md, --init-from)
        params['input_balance'] = float(f['wPNAPL_scale'] / f['wKCAPL_scale'])
        rows.append(dict(kind='fit', start=key, tag=f['tag'], apl_readout=f.get('apl_readout'), pattern_only=f.get('apl_pattern_only'),
            block_term=f.get('apl_block_term'), opk_term=f.get('apl_opk_term'), loss=f['loss'], initial_loss=f['initial_loss'],
            n_evals=f['n_evals'], **params, **full_row(o)))
        hs = history_start(res_dir, f['tag'])
        rows.append(dict(kind='start (first evaluation)', start=key, tag=f['tag'], apl_readout=f.get('apl_readout'), **hs))
        if key in STARTS and key not in done_starts:
            label, folder, tag = STARTS[key]
            oa = load_outputs(folder, tag)
            if oa is not None:
                rows.append(dict(kind='start (archived)', start=key, tag=f'{folder}:{tag}', **full_row(oa)))
                done_starts.add(key)
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / 'comparison.csv', index=False)
    write_table(df, res_dir)
    make_figures(df, res_dir)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E67_apl_effect_fits.py',
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        n_fits=int((df.kind == 'fit').sum()) if len(df) else 0, calibration_target=refs()['cal_target'],
        data_effect_means={v: float(refs()[v].mean()) for v in ('graded', 'thresholded', 'noise_corrected', 'binary')})
    (res_dir / 'provenance_comparison.json').write_text(json.dumps(prov, indent=1) + '\n')
    print(df[[c for c in ('kind', 'start', 'tag', 'raw_loss', 'raw_mean', 'raw_pattern', 'cal_loss', 'cal_mean', 'cal_pattern', 'cal_block',
        'rate_on', 'ratio', 'rate_rho_on', 'realism_flag') if c in df]].round(3).to_string())


def _f(x, spec):
    try:
        return '' if x is None or (isinstance(x, float) and np.isnan(x)) else format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def write_table(df: pd.DataFrame, res_dir: Path) -> None:
    dt = H.data_target
    L = ['| model | APL-effect loss<br>raw / calibrated | mean APL effect<br>raw / calibrated | pattern part<br>raw / calibrated | '
         'per-odor / pair-specific<br>raw | per-odor / pair-specific<br>calibrated | ρ on the per-pair APL effect<br>raw / calibrated | '
         'block<br>raw / calibrated / binarized | response rate APL on;<br>APL-off / APL-on ratio | response rate ρ<br>APL on / off | '
         'odors per KC, APL off<br>distance from the data | realism |', '|---|' + '---:|' * 10 + '---|']
    for _, r in df.iterrows():
        name = f"{r['kind']}: {r['tag']}"
        L.append(f"| {name} | {_f(r.get('raw_loss'), '.3f')} / {_f(r.get('cal_loss'), '.3f')} | {_f(r.get('raw_mean'), '+.3f')} / {_f(r.get('cal_mean'), '+.3f')} | "
            f"{_f(r.get('raw_pattern'), '.3f')} / {_f(r.get('cal_pattern'), '.3f')} | {_f(r.get('raw_per_odor'), '.3f')} / {_f(r.get('raw_pair_specific'), '.3f')} | "
            f"{_f(r.get('cal_per_odor'), '.3f')} / {_f(r.get('cal_pair_specific'), '.3f')} | {_f(r.get('raw_rho'), '.2f')} / {_f(r.get('cal_rho'), '.2f')} | "
            f"{_f(r.get('raw_block'), '.3f')} / {_f(r.get('cal_block'), '.3f')} / {_f(r.get('binary_block'), '.3f')} | "
            f"{_f(r.get('rate_on'), '.3f')}; {_f(r.get('ratio'), '.2f')} | {_f(r.get('rate_rho_on'), '.2f')} / {_f(r.get('rate_rho_off'), '.2f')} | "
            f"{_f(r.get('opk_off_dist'), '.2f')} | {r.get('realism_flag', '') if isinstance(r.get('realism_flag'), str) else ''} |")
    L.append(f"| KC data | floor {dt('apl_effect_rmse_floor'):.3f} | {dt('apl_effect_mean'):+.3f} | | | | ceiling {dt('apl_effect_rho_ceiling'):.2f} | "
        f"{dt('block'):.3f} | {dt('ctrl_frac'):.3f}; {dt('rate_ratio_fraction'):.2f} | ceiling {dt('ctrl_rate_rho_ceiling'):.2f} / {dt('tnt_rate_rho_ceiling'):.2f} | 0 | |")
    L += ['', '"raw": cosine distance on raw spike counts; "calibrated": E65\'s saturating readout with K set on the model\'s APL-on counts '
        'to the control flies\' responders-only participation ratio. "start (first evaluation)" rows are the loss\'s own statistics at the '
        'fit\'s first evaluation; "start (archived)" rows the full statistics of the archived start model.', '']
    fit = df[df.kind == 'fit']
    if len(fit):
        L += ['| fit | mean distance raw<br>APL on / off | mean distance calibrated<br>APL on / off | ρ on odor-odor distances raw<br>APL on / off | '
              'natural-source alignment raw<br>APL on / off / on minus off | participation ratio among responders,<br>APL off, calibrated | K | '
              'bouton→APL / KC→APL<br>input balance | odors per KC, distance<br>APL on / off | '
              'loss vs data sub-threshold zeroed / noise-corrected<br>raw | the same<br>calibrated | square root<br>loss (mean) | binarized vs binarized data<br>loss (mean) | '
              'somatic threshold factor / APL→KC / APL→bouton / KC–KC |', '|---|' + '---:|' * 12]
        for _, r in fit.iterrows():
            L.append(f"| {r['tag']} | {_f(r['raw_mean_dist_on'], '.3f')} / {_f(r['raw_mean_dist_off'], '.3f')} | {_f(r['cal_mean_dist_on'], '.3f')} / "
                f"{_f(r['cal_mean_dist_off'], '.3f')} | {_f(r['raw_rho_on'], '.2f')} / {_f(r['raw_rho_off'], '.2f')} | {_f(r['raw_vcf_on'], '.2f')} / "
                f"{_f(r['raw_vcf_off'], '.2f')} / {_f(r['raw_vcf_on'] - r['raw_vcf_off'], '+.2f')} | {_f(100 * r['cal_pr_resp_off'], '.1f')} % | {_f(r['K'], '.1f')} | "
                f"{_f(r.get('input_balance'), '.1f')} | {_f(r['opk_on_dist'], '.2f')} / {_f(r['opk_off_dist'], '.2f')} | "
                f"{_f(r['raw_loss_vs_thresholded'], '.3f')} / {_f(r['raw_loss_vs_noise_corrected'], '.3f')} | {_f(r['cal_loss_vs_thresholded'], '.3f')} / "
                f"{_f(r['cal_loss_vs_noise_corrected'], '.3f')} | {_f(r['sqrt_loss'], '.3f')} ({_f(r['sqrt_mean'], '+.3f')}) | "
                f"{_f(r['binary_loss_vs_binary'], '.3f')} ({_f(r['binary_mean'], '+.3f')}) | {_f(r.get('thr_factor'), '.3f')} / {_f(r.get('apl_factor_vs_e3'), '.3f')} / "
                f"{_f(r.get('aplpn_factor_vs_e3'), '.2f')} / {_f(r.get('kc_kc_inh'), '.1f')} |")
        L.append(f"| KC data | {dt('ctrl_mean_dist'):.3f} / {dt('tnt_mean_dist'):.3f} | {dt('ctrl_mean_dist_noise_corrected'):.3f} / {dt('tnt_mean_dist_noise_corrected'):.3f} "
            f"(noise-corrected) | ceiling {dt('ctrl_ceiling'):.2f} / {dt('tnt_ceiling'):.2f} | {dt('ctrl_vcf'):.2f} / {dt('tnt_vcf'):.2f} / {dt('vcf_apl_effect'):+.2f} | "
            f"{100 * dt('tnt_participation_ratio_resp'):.1f} % | | | bands {dt('ctrl_odors_per_kc_tv_max'):.2f} / {dt('tnt_odors_per_kc_tv_max'):.2f} | floor {dt('apl_effect_thresholded_rmse_floor'):.3f} / - | | | floor {dt('apl_effect_binary_rmse_floor'):.3f} "
            f"({dt('apl_effect_binary_mean'):+.3f}) | |")
    (res_dir / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def make_figures(df: pd.DataFrame, res_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    fit = df[df.kind == 'fit']
    if not len(fit):
        return
    # histories: the loss and the APL-effect loss under both readouts per evaluation
    n = len(fit)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.4 * n), squeeze=False)
    for i, (_, r) in enumerate(fit.iterrows()):
        h = pd.read_csv(res_dir / f"history_{r['tag']}.csv")
        axes[i, 0].plot(h['loss'].to_numpy(), lw=0.8, color='0.3')
        axes[i, 0].set_yscale('log'); axes[i, 0].set_ylabel('loss', fontsize=8); axes[i, 0].set_title(r['tag'], fontsize=8)
        for col, c, lab in (('apl_raw_rmse', '0.5', 'raw counts'), ('apl_cal_rmse', '#2b6cb0', 'calibrated readout')):
            if col in h:
                axes[i, 1].plot(h[col].to_numpy(), lw=0.8, color=c, label=lab)
        axes[i, 1].axhline(dt('apl_effect_rmse_floor'), color='k', ls='--', lw=0.8)
        axes[i, 1].set_ylim(0, 0.25); axes[i, 1].set_ylabel('APL-effect loss', fontsize=8)
        if i == 0:
            axes[i, 1].legend(fontsize=7, frameon=False)
        for ax in axes[i]:
            ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('evaluation', fontsize=8)
    fig.tight_layout(); fig.savefig(res_dir / 'E67_fit_histories.png', dpi=120); plt.close(fig)
    # the mean APL effect against the pattern part, start -> fit, under each definition: colour by start family,
    # marker by the definition in the loss (circle: the readout, square: raw counts, triangle: both), a short label
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    starts = df[df.kind == 'start (first evaluation)'].set_index('tag')
    fam = {'A': '#d9480f', 'C': '#7048e8', 'C2': '#2b6cb0', 'D': '#2f9e44', 'E': '#868e96'}
    marker = {'cal': 'o', 'raw': 's', 'both': '^'}
    for ax, name, lab in zip(axes, ('raw', 'cal'), ('raw spike counts', 'calibrated readout')):
        xs, ys = [dt('apl_effect_mean')], []
        for _, r in fit.iterrows():
            lab_short = fit_label(r)
            family = lab_short.split()[0]
            c = fam.get(family, 'k')
            s0 = starts.loc[r['tag']] if r['tag'] in starts.index else None
            if s0 is not None and np.isfinite(s0.get(f'{name}_mean', np.nan)):
                ax.annotate('', xy=(r[f'{name}_mean'], r[f'{name}_pattern']), xytext=(s0[f'{name}_mean'], s0[f'{name}_pattern']),
                    arrowprops=dict(arrowstyle='->', color=c, lw=0.8, alpha=0.6))
                xs.append(s0[f'{name}_mean']); ys.append(s0[f'{name}_pattern'])
            ax.plot(r[f'{name}_mean'], r[f'{name}_pattern'], marker.get(r.get('apl_readout'), 'o'), color=c, ms=7, mec='k', mew=0.4)
            ax.annotate(lab_short, (r[f'{name}_mean'], r[f'{name}_pattern']), fontsize=6, xytext=(4, 2), textcoords='offset points')
            xs.append(r[f'{name}_mean']); ys.append(r[f'{name}_pattern'])
        ax.set_xlim(min(xs) - 0.01, max(xs) + 0.01); ax.set_ylim(min(ys) - 0.004, max(ys) + 0.004)
        ax.axvline(dt('apl_effect_mean'), color='k', ls='--', lw=0.8)
        ax.set_xlabel(f'mean APL effect, {lab} (dashed: KC data)', fontsize=9); ax.set_title(lab, fontsize=9.5)
        ax.set_ylabel('pattern part of the APL-effect loss (lower is better)', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.92', lw=0.6)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker='o', ls='', label=k) for k, c in fam.items()] + [
        Line2D([], [], color='k', marker=m, ls='', mfc='w', label=f'fitted {"under the readout" if d == "cal" else "on raw counts" if d == "raw" else "under both"}')
        for d, m in marker.items()]
    axes[1].legend(handles=handles, fontsize=7, frameon=False, loc='upper right')
    fig.suptitle('Each fit from its start (arrow), scored under both definitions; A, C, C2, D, E: the start families', fontsize=9.5)
    fig.tight_layout(); fig.savefig(res_dir / 'E67_mean_vs_pattern.png', dpi=130); plt.close(fig)


def fit_label(r) -> str:
    """A short label: the start family (A, C, C2 for C2's start and every fit from C2's parameters, D, E), the
    definition in the loss and the arm."""
    start = str(r.get('start', ''))
    tag = str(r['tag'])
    fam = 'C2' if (start.startswith('C2') or start.startswith('from rate0.068_aplcal_k2') or tag.startswith('rate0.068_aplraw_k2')) else (
        'D' if ('DM1' in tag) else start)
    arm = ''
    for key, text in (('aplcalblk', ' +block'), ('aplcalopk', ' +odors/KC'), ('aplboth', ' both'), ('aplrawpat', ' pattern only'),
                      ('c2bal5', ' balance 5.1'), ('_k1.5_', ' exponent 1.5'), ('_k1_', ' exponent 1')):
        if key in tag:
            arm += text
    return f"{fam} {r.get('apl_readout')}{arm}"

if __name__ == '__main__':
    main()
