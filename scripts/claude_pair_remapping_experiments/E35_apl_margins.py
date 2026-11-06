#!/usr/bin/env python3
"""E35 diagnostic: the KCs' margins above threshold without APL, and the inhibition each
odor would need for the data's uniform APL suppression.

For one per-claw threshold (or none), runs the subtractive bouton base's threshold tuning
once with the olfsysm run variables returned, which gives each KC's peak somatic drive per
odor before APL (`rv.kc.pks`; the drive the threshold is chosen from) and so each KC's margin
above threshold. Then, per odor:
- the effective inhibition of each E35 form: the margin c such that the fraction of KCs with
  margin above c equals that form's APL-on response fraction (APL acting as a shift of the
  threshold, which is what any inhibition amounts to for a KC with a hard threshold);
- the inhibition the data's uniform suppression would need: the c leaving the odor's APL-off
  fraction divided by 2.2 (and, as a check, by the odor's own ratio in the data);
- the APL activity per form (mean and peak of the APL trace over the odor window, from a
  run at the tuned parameters with the dynamics returned);
- the margin distribution's scale: the median margin among the APL-off responders and the
  odor's mean drive.
Writes results/E35_divisive_apl/margins_<thrtag>.csv and prints the rank correlations of
each per-odor quantity with the odor's APL-off response fraction.

Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E35_apl_margins.py --thr 70 [--forms sub kcdiv boutondiv bothdiv]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats as sstats

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON, TARGET
from E35_divisive_apl import EXPERIMENT, FORMS, thr_tag

DATA_RATIO = 2.2


def shift_for_fraction(m: np.ndarray, frac: float) -> float:
    """Smallest threshold shift c leaving a fraction `frac` of KCs with margin above c."""
    if frac <= 0:
        return float(np.max(m))
    return float(np.quantile(m, 1.0 - frac))


def apl_trace_stats(inh) -> pd.DataFrame:
    """Mean and peak APL activity per odor over the odor window (time >= 0)."""
    da = inh
    tdim = [d for d in da.dims if d.startswith('time')][0]
    odim = [d for d in da.dims if d != tdim][0]
    post = da.where(da[tdim] >= 0, drop=True)
    df = pd.DataFrame({'apl_mean': post.mean(tdim).to_series(), 'apl_peak': post.max(tdim).to_series()})
    df.index = [H.odor_abbrev(o) for o in df.index]
    return df


def main() -> None:
    from al_analysis.mb_model import fit_mb_model
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--thr', default='70')
    ap.add_argument('--forms', nargs='+', default=list(FORMS), choices=list(FORMS))
    args = ap.parse_args()
    thr = None if args.thr == 'none' else float(args.thr)
    tt = thr_tag(thr)
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    tuned = json.loads((out_dir / f'tuned_params_{tt}.json').read_text())
    fracs = pd.read_csv(res_dir / f'per_odor_response_fractions_{tt}.csv', index_col=0)
    orn = H.apl16_orn_deltas()
    kws = dict(BOUTON, make_plots=False, silent=True)
    if thr is not None:
        kws['claw_thr'] = thr

    # 1. the threshold tuning once, subtractive (the margins do not depend on the APL form)
    ret = fit_mb_model(orn, target_sparsity=TARGET, return_olfsysm_vars=True, **kws)
    resp, spk, wPNKC, params = ret
    mp = params.pop('mp'); rv = params.pop('rv')
    pks = np.asarray(rv.kc.pks, dtype=float)
    thr_kc = np.asarray(rv.kc.thr, dtype=float).reshape(-1)
    spont = np.asarray(rv.kc.spont_in, dtype=float).reshape(-1)
    assert pks.shape[0] == thr_kc.shape[0], (pks.shape, thr_kc.shape)
    margins = pks - (thr_kc - 2.0 * spont)[:, None]
    odors = [H.odor_abbrev(o) for o in resp.columns]
    assert pks.shape[1] == len(odors), (pks.shape, len(odors))
    margins = pd.DataFrame(margins, columns=odors)
    del mp, rv
    tuned_thr = float(np.mean(params['fixed_thr']))
    print(f'{tt}: tuned threshold {tuned_thr:.1f} (sweep {tuned[f"sub_{tt}"]["fixed_thr"]:.1f}); '
        f'fraction with margin > 0 {float((margins > 0).to_numpy().mean()):.4f}', flush=True)

    data_on = fracs['data_TNT3c']; data_off = fracs['data_TNT']
    rows = []
    for o in H.APL16_ODORS:
        m = margins[o].to_numpy()
        off_model = float(fracs.loc[o, f'sub_{tt}_acute_off'])
        resp_m = m[m > 0]
        row = dict(odor=o, off_frac_margin=float((m > 0).mean()), off_frac_model=off_model,
            data_ratio=float(data_off[o] / data_on[o]), mean_drive=float(np.mean(m)),
            top5_drive=float(np.quantile(m, 0.95)), median_margin_responders=float(np.median(resp_m)) if len(resp_m) else np.nan,
            c_req_uniform=shift_for_fraction(m, off_model / DATA_RATIO),
            c_req_data_ratio=shift_for_fraction(m, off_model / float(data_off[o] / data_on[o])))
        for f in args.forms:
            col = f'{f}_{tt}_on'
            if col in fracs:
                row[f'on_frac_{f}'] = float(fracs.loc[o, col])
                row[f'c_model_{f}'] = shift_for_fraction(m, float(fracs.loc[o, col]))
        rows.append(row)
    tab = pd.DataFrame(rows).set_index('odor')

    # 2. APL activity per odor for each form, at its tuned parameters
    for f in args.forms:
        key = f'{f}_{tt}'
        if key not in tuned:
            continue
        t = tuned[key]
        fkws = dict(kws)
        if FORMS[f][0]:
            fkws['apl_inh_divisive'] = True
        if FORMS[f][1]:
            fkws['bouton_inh_divisive'] = True
        _, _, _, p2 = fit_mb_model(orn, fixed_thr=t['fixed_thr'], wAPLKC=t['wAPLKC_scale'], wKCAPL=t['wKCAPL_scale'],
            wAPLPN=t['wAPLPN_scale'], wPNAPL=t['wPNAPL_scale'], return_dynamics=True, **fkws)
        st = apl_trace_stats(p2['inh_sims'])
        tab[f'apl_mean_{f}'] = st['apl_mean'].reindex(tab.index)
        tab[f'apl_peak_{f}'] = st['apl_peak'].reindex(tab.index)
        del p2
        print(f'{key}: APL trace done', flush=True)

    tab.to_csv(res_dir / f'margins_{tt}.csv')
    pd.set_option('display.width', 250)
    print(tab.round(3).to_string())
    print('\nrank correlation with the odor\'s APL-off response fraction (model):')
    for c in tab.columns:
        if c in ('off_frac_model',):
            continue
        ok = tab[c].notna() & np.isfinite(tab[c])
        if ok.sum() > 3:
            print(f'  {c}: {sstats.spearmanr(tab.loc[ok, c], tab.loc[ok, "off_frac_model"])[0]:+.2f}')


if __name__ == '__main__':
    main()
