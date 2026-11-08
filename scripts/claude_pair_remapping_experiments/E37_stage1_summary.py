#!/usr/bin/env python3
"""E37 stage 1 against its starting points: each spike-only joint fit next to the E36 fit it
started from (re-simulated in stage 0, scored with the current loss targets) and that fit with
the bouton->APL input removed but not refitted (stage 0's `nopnapl`).

Writes results/E37_spike_only_apl/stage1_comparison.csv in the shared directory: the extended
loss, the fitted statistics, E35's suppression uniformity (computed here for the stage-1 fits
from their saved APL-on / acute APL-off responses) and the absolute fitted scales (at olfsysm's
0.5 ms step; spike-driven KC->APL drive per spike scales with the step, see unresolved_issues.md).

Usage: ./venv/bin/python E37_stage1_summary.py
"""
import pandas as pd

import kc_remapping_harness as H
from E35_divisive_apl import suppression_stats

EXPERIMENT = 'E37_spike_only_apl'
# (label, E36 optimum's stage-0 tag stem, stage-1 fit tag)
FITS = [
    ('70 Hz, no gain', 'spk_thr70_k2_s2', 'spko_thr70_k2'),
    ('70 Hz, gain 1.5', 'spk_thr70_gain1.5_k2_s2', 'spko_thr70_gain1.5_k2'),
    ('40 Hz, gain 1.5', 'spk_thr40_gain1.5_k2_cap3_s2', 'spko_thr40_gain1.5_k2'),
]
MODELS = [
    ('E36 fit (APL driven by KC spikes and boutons)', '_ref', 'stage0'),
    ('bouton->APL input removed, not refitted', '_nopnapl', 'stage0'),
    ('spike-only refit', '', 'stage1'),
]


def main() -> None:
    res = H.results_dir(EXPERIMENT)
    out = H.outputs_dir(EXPERIMENT)
    dr = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    s0 = pd.read_csv(res / 'stats_variants.csv').set_index('tag')
    s1 = pd.read_csv(res / 'fits.csv').set_index('tag')
    rows = []
    for label, e36, e37 in FITS:
        for model, suffix, stage in MODELS:
            if stage == 'stage0':
                tag = e36 + suffix
                r = s0.loc[tag]
                spread = r['suppression_sd_log']
                rho = r['spearman_suppression_vs_on_frac']
            else:
                tag = e37
                r = s1.loc[tag]
                on = H.response_fractions(pd.read_parquet(out / f'{tag}_on_responses.parquet'))
                off = H.response_fractions(pd.read_parquet(
                    out / f'{tag}_acute_off_responses.parquet'))
                sup = suppression_stats(on, off, dr)
                spread = sup['suppression_sd_log']
                rho = sup['spearman_suppression_vs_on_frac']
            rows.append(dict(fit=label, model=model, stage=stage, tag=tag, loss=r['loss'],
                block=r['fit_block'], rate_ratio=r['rate_ratio_off_over_on'],
                rho_ctrl=r['fit_on_spearman'], rho_tnt=r['fit_off_spearman'],
                iso_r2=r['fit_iso_r2'], rate_rho_on=r['fit_po_on_spearman'],
                rate_rho_off=r['fit_po_off_spearman'], wt_dev=r['fit_wt_dev_on'],
                vcf_off=r['fit_vcf_off'], suppression_spread=spread, suppression_rho_on=rho,
                fixed_thr=r['fixed_thr'], wAPLKC_scale=r['wAPLKC_scale'],
                wKCAPL_scale=r['wKCAPL_scale'], wAPLPN_scale=r['wAPLPN_scale'],
                wPNAPL_scale=r['wPNAPL_scale'], kc_kc_inh=r['kc_kc_inh']))
    t = pd.DataFrame(rows)
    t.to_csv(res / 'stage1_comparison.csv', index=False)
    print(t.drop(columns=['tag']).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
