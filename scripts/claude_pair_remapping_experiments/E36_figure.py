#!/usr/bin/env python3
"""E36 figure and tables: KC-spike-driven vs claw-input-driven APL in the combined model.

Stage 1: the three-mechanism points (E28's pipeline) with spike-driven APL (results/E36_spike_apl/
stats_variants.csv) against E28's claw-driven points. Stage 2: the structure-loss joint fits with
spike-driven APL restarted from the claw-driven optima (hal: results/E36_spike_apl/fits.csv;
atlas: results/E36_spike_apl_jointfit/fits.csv) against E31's / E34's claw-driven capped fits,
with E35's suppression uniformity computed from the archived APL-on / APL-off responses.
Writes results/E36_spike_apl/{E36_spike_apl.png, stage1_points.csv, stage2_fits.csv}.
Usage: ./venv/bin/python E36_figure.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E35_divisive_apl import suppression_stats

R = H.shared_dir() / 'results'
A = H.shared_dir() / 'data' / 'model_outputs'
OUT = R / 'E36_spike_apl'
V = H.PRIMARY_MODEL_RDM + '__'
POINTS = ['thr70_k2_kckc100_g1', 'thr70_k2_kckc100_g1.5', 'thr40_k2_kckc30_g1', 'thr40_k2_kckc30_g1.5']
POINT_LABEL = {'thr70_k2_kckc100_g1': '70 Hz, no gain', 'thr70_k2_kckc100_g1.5': '70 Hz, gain 1.5',
    'thr40_k2_kckc30_g1': '40 Hz, no gain', 'thr40_k2_kckc30_g1.5': '40 Hz, gain 1.5'}
# stage 2: (label, claw-driven (experiment, tag, archive stem), spike-driven (experiment, tag, outputs stem))
FITS = [
    ('70 Hz, no gain', ('E31_aplpn_bound', 'thr70_k2_aplpnmax3', A / 'E31_aplpn_bound' / 'thr70_k2_aplpnmax3'),
        ('E36_spike_apl', 'spk_thr70_k2_s2', H.HERE / 'outputs' / 'E36_spike_apl' / 'spk_thr70_k2_s2')),
    ('70 Hz, gain 1.5', ('E34_gain_bounded', 'thr70_gain1.5_k2_cap3', A / 'E34_gain_bounded' / 'thr70_gain1.5_k2_cap3'),
        ('E36_spike_apl', 'spk_thr70_gain1.5_k2_s2', H.HERE / 'outputs' / 'E36_spike_apl' / 'spk_thr70_gain1.5_k2_s2')),
    ('40 Hz, gain 1.5', ('E34_gain_bounded', 'thr40_gain1.5_k2_cap3', A / 'E34_gain_bounded' / 'thr40_gain1.5_k2_cap3'),
        ('E36_spike_apl_jointfit', 'spk_thr40_gain1.5_k2_cap3_s2', A / 'E36_spike_apl_jointfit' / 'spk_thr40_gain1.5_k2_cap3_s2')),
]
DATA = dict(loss=0.0, block=0.055, ratio=2.2, ctrl=0.98, tnt=0.97, iso=0.94, rr_on=0.97, rr_off=0.97, wt=0.83, vcf=0.36,
    spread=0.148, rho_sup=-0.12)


def stage1() -> pd.DataFrame:
    dr = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    rows = []
    for drive, exp in (('claw input', 'E28_gain_combined'), ('KC spikes', 'E36_spike_apl')):
        st = pd.read_csv(R / exp / 'stats_variants.csv').set_index('tag')
        fr = pd.read_csv(R / exp / 'per_odor_response_fractions.csv', index_col=0)
        for p in POINTS:
            r = st.loc[p]
            sup = suppression_stats(fr[f'{p}_on'], fr[f'{p}_acute_off'], dr)
            rows.append(dict(point=POINT_LABEL[p], drive=drive, block=r[V + 'block_fruity_small_iso'], ratio=r['rate_ratio_off_over_on'],
                ctrl=r[V + 'on_spearman'], tnt=r[V + 'off_spearman'], iso=r[V + 'iso_r2'], rr_on=r['spearman_rate_on_vs_data'],
                rr_off=r['spearman_rate_off_vs_data'], spread=sup['suppression_sd_log'], rho_sup=sup['spearman_suppression_vs_on_frac']))
    return pd.DataFrame(rows)


def stage2() -> pd.DataFrame:
    dr = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    rows = []
    for label, claw, spike in FITS:
        for drive, (exp, tag, stem) in (('claw input', claw), ('KC spikes', spike)):
            r = pd.read_csv(R / exp / 'fits.csv').set_index('tag').loc[tag]
            on = H.response_fractions(pd.read_parquet(f'{stem}_on_responses.parquet'))
            off = H.response_fractions(pd.read_parquet(f'{stem}_acute_off_responses.parquet'))
            sup = suppression_stats(on, off, dr)
            rows.append(dict(fit=label, drive=drive, experiment=exp, tag=tag, loss=r['loss'], block=r['fit_block'],
                ratio=r['rate_ratio_off_over_on'], ctrl=r['fit_on_spearman'], tnt=r['fit_off_spearman'], iso=r['fit_iso_r2'],
                rr_on=r['fit_po_on_spearman'], rr_off=r['fit_po_off_spearman'], wt=r['fit_wt_dev_on'], vcf=r['fit_vcf_off'],
                spread=sup['suppression_sd_log'], rho_sup=sup['spearman_suppression_vs_on_frac'],
                thr_factor=r['thr_factor'], apl_kc=r['apl_factor_vs_e3'], apl_bouton=r['aplpn_factor_vs_e3'], kc_kc=r['kc_kc_inh']))
    return pd.DataFrame(rows)


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    s1 = stage1(); s2 = stage2()
    s1.to_csv(OUT / 'stage1_points.csv', index=False); s2.to_csv(OUT / 'stage2_fits.csv', index=False)
    colors = {'claw input': '#6b7280', 'KC spikes': '#15803d'}
    panels = [('loss', 'extended loss (lower is better; KC data 0)'), ('spread', 'spread of suppression across odors\n(sd of log APL-off / APL-on responders; data 0.15)'),
        ('rr_on', 'response rate ρ, APL on vs control flies\n(ceiling about 0.97)'), ('block', 'fruity × small-ester block (data +0.055)'),
        ('ratio', 'response rate, APL off relative to on (data 2.2)'), ('ctrl', 'ρ with the control matrix, APL on\n(ceiling 0.98)')]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fits = list(dict.fromkeys(s2.fit))
    x = np.arange(len(fits)); w = 0.36
    for ax, (col, title) in zip(axes.ravel(), panels):
        for i, drive in enumerate(colors):
            v = s2[s2.drive == drive].set_index('fit').loc[fits, col]
            ax.bar(x + (i - 0.5) * w, v, w, color=colors[drive], edgecolor='white')
        if col in DATA and col != 'loss':
            ax.axhline(DATA[col], color='0.2', lw=1, ls=':')
        ax.set_xticks(x); ax.set_xticklabels(fits, fontsize=9); ax.set_title(title, fontsize=9.5)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, axis='y', color='0.9', lw=0.6)
    axes[0, 0].legend(handles=[Patch(color=c, label=f'APL driven by {d}') for d, c in colors.items()], fontsize=8.5, frameon=False)
    fig.suptitle('Structure-loss joint fits with the APL->bouton scale capped at 3x: APL driven by the summed claw input vs by KC spikes '
        '(dotted: KC data)', fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / 'E36_spike_apl.png', dpi=130); plt.close(fig)
    pd.set_option('display.width', 250)
    print(s1.round(3).to_string(index=False)); print(); print(s2.drop(columns=['experiment', 'tag']).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
