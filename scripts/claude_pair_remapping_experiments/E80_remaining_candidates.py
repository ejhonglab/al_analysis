#!/usr/bin/env python3
"""E80: the flies' pulse in the model and the remaining candidates for the common error.

The entry is in experiment_priorities.md (planned by a Fable subagent, claude-fable-5-2, spawned from viki 2026-11-10
23:41). Part A (the pulse) is in E80_pulse.py, parts B and C (the screen, the gate) in E80_screen.py; this script has part D (no simulations: the
response amplitude against a KC's breadth in the flies, in baseline-noise units, and in the archived fits of the best
families; the breadth-weighted reading of the fits against the effect in baseline-noise units and in Yang's z; each KC
type alone with recording-bootstrap intervals) and the tables. Only shared data are read.

    python E80_remaining_candidates.py --part {D,A,B,C,tables}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E80_remaining_candidates'


CLASSES = (('1', 1, 1), ('2-3', 2, 3), ('4-7', 4, 7), ('8+', 8, 16))
# the best families' fits read in part D: (label, folder, tag)
FITS = [('C2, the breadth-2 APL-effect fit', 'E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3'),
        ('the firing-ceiling fit at strength 2 (E69 fit 2)', 'E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.1114t1000_k2'),
        ('E76 fit 0 (rate terms in baseline-noise units)', 'E76_data_side_target', 'nu_rate0.068_aplboth_refr3_adapt0.1114t1000_k2'),
        ('E76 fit 1 (APL-effect term in baseline-noise units too)', 'E76_data_side_target', 'nu_rate0.068_aplbothdnu_refr3_adapt0.1114t1000_k2')]
NOISE = H.shared_dir() / 'data' / 'yang_apl16_kc_noise_units'


def noise_threshold() -> float:
    return float(json.loads((NOISE / 'provenance.json').read_text())['noise_threshold'])


def fly_matrices(genotype: str) -> Dict[tuple, pd.DataFrame]:
    """Each recording's KC x odor trial means in baseline-noise units (data/yang_apl16_kc_noise_units/), odors in the
    APL16 order; genotype 'TNT3c' (control flies) or 'TNT' (APL output blocked)."""
    t = pd.read_parquet(NOISE / 'trialmean_long.parquet')
    t = t[t.genotype == genotype]
    out = {}
    for rec, g in t.groupby(['date', 'fly_num', 'hemibrain']):
        out[rec] = g.pivot_table(index='roi', columns='odor', values='value')[list(H.APL16_ODORS)]
    return out


def class_amplitudes(values: np.ndarray, resp: np.ndarray) -> Dict[str, Tuple[float, float, int]]:
    """Per odors-per-KC class, (mean value over responder entries, share of the summed squared value over responder
    entries, number of KCs); KCs x odors arrays; classes by each KC's number of responses."""
    n = resp.sum(axis=1)
    tot = float((values[resp] ** 2).sum())
    res = {}
    for name, lo, hi in CLASSES:
        k = (n >= lo) & (n <= hi)
        m = resp & k[:, None]
        res[name] = (float(values[m].mean()) if m.any() else np.nan, float((values[m] ** 2).sum()) / tot if tot > 0 else np.nan, int(k.sum()))
    return res


def amplitude_by_breadth(out: Path) -> pd.DataFrame:
    """Part D: per odors-per-KC class (1, 2-3, 4-7, 8 or more), the responders' mean amplitude and share of the summed
    squared response, in the flies (baseline-noise units) and in every archived fit of the best families (spike counts),
    with APL on and off. Writes amplitude_by_breadth.csv (one row per source, condition and class; 'pooled' over the
    recordings and 'recording_mean', the mean of each recording's value)."""
    import E72_wiring_population as E72
    thr = noise_threshold()
    rows = []
    for geno, label in (('TNT3c', 'control flies'), ('TNT', 'APL output blocked (TNT)')):
        mats = fly_matrices(geno)
        per = []
        allv, allr = [], []
        for rec, m in mats.items():
            v = m.to_numpy(dtype=float)
            r = np.nan_to_num(v, nan=-np.inf) > thr
            v = np.nan_to_num(v)
            per.append(class_amplitudes(v, r))
            allv.append(v)
            allr.append(r)
        pooled = class_amplitudes(np.vstack(allv), np.vstack(allr))
        for name, _, _ in CLASSES:
            rows.append(dict(source=label, kind='flies', condition='data', cls=name, mean_amplitude=pooled[name][0],
                             share_sq=pooled[name][1], n_kcs=pooled[name][2],
                             mean_amplitude_recording_mean=float(np.nanmean([p[name][0] for p in per])),
                             share_sq_recording_mean=float(np.nanmean([p[name][1] for p in per])), units='baseline-noise units'))
    for label, folder, tag in FITS:
        o = E72.load_fit(folder, tag)
        if o is None:
            print(f'{label}: no saved outputs', flush=True)
            continue
        for c in ('on', 'off'):
            r = o[c][0].to_numpy(dtype=float) > 0.5
            s = o[c][1].to_numpy(dtype=float)
            ca = class_amplitudes(s, r)
            for name, _, _ in CLASSES:
                rows.append(dict(source=label, kind='model', condition=f'APL {c}', cls=name, mean_amplitude=ca[name][0],
                                 share_sq=ca[name][1], n_kcs=ca[name][2], folder=folder, tag=tag, units='spike counts'))
    res = pd.DataFrame(rows)
    res.to_csv(out / 'amplitude_by_breadth.csv', index=False)
    print(res.pivot_table(index=['source', 'condition'], columns='cls', values=['mean_amplitude', 'share_sq'], sort=False).round(3).to_string(), flush=True)
    return res


def breadth_weighted(out: Path) -> pd.DataFrame:
    """Part D: the fits' counts weighted by (odors per KC) ** p, the weight set on the control flies' curve, scored
    against the APL effect in baseline-noise units and in Yang's z. Writes breadth_weighted.csv. Each condition's KCs
    are weighted by their own number of responses (two or more spikes); p 0 is the fit as it is; p_flies is the log-log
    slope of the control flies' mean amplitude against the class's middle number of odors (amplitude_by_breadth.csv)."""
    import E72_wiring_population as E72
    amp = pd.read_csv(out / 'amplitude_by_breadth.csv')
    ctl = amp[(amp.source == 'control flies')].set_index('cls').reindex([c[0] for c in CLASSES])
    mids = np.array([1.0, 2.5, 5.5, 10.0])
    p_flies = float(np.polyfit(np.log(mids), np.log(ctl.mean_amplitude.to_numpy()), 1)[0])
    eff = pd.read_csv(H.results_dir('E76_data_side_target') / 'effect_by_unit_pairs.csv').set_index(['odor_a', 'odor_b'])
    targets = {'noise': eff['effect__noise__graded'], 'z': eff['effect__z__graded']}
    targets = {k: v.rename(None) for k, v in targets.items()}
    rows = []
    for label, folder, tag in FITS:
        o = E72.load_fit(folder, tag)
        if o is None:
            continue
        for p in (0.0, 0.1, 0.2, 0.3, 0.5, p_flies):
            d = {}
            for c in ('on', 'off'):
                r, s = o[c]
                n = (r.to_numpy(dtype=float) > 0.5).sum(axis=1)
                w = np.where(n > 0, n.astype(float), 1.0) ** p
                d[c] = H.readout_pair_distances(s * w[:, None], 'raw')
            for tn, te in targets.items():
                st = H.apl_effect_stats(d['on'], d['off'], H.canonical_pairs(te))
                rows.append(dict(fit=label, folder=folder, tag=tag, p=p, p_is_flies=bool(p == p_flies), target=tn,
                                 loss=st['apl_effect_rmse'], mean=st['apl_effect_mean'], pattern=st['apl_effect_rmse_centered'],
                                 rho=st['apl_effect_spearman']))
    res = pd.DataFrame(rows)
    res.to_csv(out / 'breadth_weighted.csv', index=False)
    print(f'p from the control flies\' curve: {p_flies:.3f}', flush=True)
    print(res.round(4).to_string(index=False), flush=True)
    return res


def type_split(out: Path) -> pd.DataFrame:
    """Part D: each KC type alone (alpha'beta', alphabeta, gamma) for the eight wirings' mean, under the readout, with
    recording-bootstrap intervals. Writes type_split.csv: per type (and all KCs), the APL-on / APL-off response rates of
    C2 on the real wiring, and for the eight wirings' mean (K set on that type's own APL-on counts, as E72's `distances`
    sets it for all KCs) the pattern part and the ester / alcohol group errors against all eight recordings, with 90 %
    intervals over 500 resamples of the recordings."""
    import E72_wiring_population as E72
    import E77_input_side_common_error as E77
    from E67_apl_effect_fits import refs
    mem = E77.members()
    fits = [E72.load_fit(f, tg) for _, f, tg in mem]
    ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
    tnt = H.yang_recording_rdms_values('TNT', 'graded')
    rng = np.random.default_rng(0)
    boots = [(rng.integers(0, len(ctrl), len(ctrl)), rng.integers(0, len(tnt), len(tnt))) for _ in range(500)]
    rows = []
    for typ in (None, "a'b'", 'ab', 'g'):
        ds = []
        for o in fits:
            sub = {}
            for c in ('on', 'off'):
                r, s = o[c]
                keep = np.ones(len(s), bool) if typ is None else np.asarray(s.index.get_level_values('kc_type') == typ)
                sub[c] = (r[keep], s[keep])
            K = H.calibrate_readout(sub['on'][1], 'saturating', refs()['cal_target'], stat='participation_ratio', responders_min=2.0)
            ds.append({c: H.readout_pair_distances(sub[c][1], 'saturating', K) for c in ('on', 'off')})
        md = {c: E77._cs(v) for c, v in E72.mean_distances(ds).items()}
        me = E77._cs(H.apl_effect(md['on'], md['off']))

        def err_for(c_df, t_df):
            de = E77._cs(H.apl_effect(H.canonical_pairs(c_df.mean()), H.canonical_pairs(t_df.mean()))).reindex(me.index)
            return (me - me.mean()) - (de - de.mean())
        e0 = err_for(ctrl, tnt)
        masks = E77.kind_masks(e0.index)
        bt = np.array([[np.sqrt(np.mean(e.to_numpy() ** 2)), e[masks['ester']].mean(), e[masks['alcohol']].mean()]
                       for e in (err_for(ctrl.iloc[a], tnt.iloc[b]) for a, b in boots)])
        o0 = fits[0]
        keep0 = np.ones(len(o0['on'][1]), bool) if typ is None else np.asarray(o0['on'][1].index.get_level_values('kc_type') == typ)
        rows.append(dict(kc_type='all' if typ is None else typ, n_kcs=int(keep0.sum()),
                         rate_on=float((o0['on'][0].to_numpy(dtype=float)[keep0] > 0.5).mean()),
                         rate_off=float((o0['off'][0].to_numpy(dtype=float)[keep0] > 0.5).mean()),
                         pattern=float(np.sqrt(np.mean(e0.to_numpy() ** 2))), pattern_lo=float(np.quantile(bt[:, 0], 0.05)),
                         pattern_hi=float(np.quantile(bt[:, 0], 0.95)), err_ester=float(e0[masks['ester']].mean()),
                         err_ester_lo=float(np.quantile(bt[:, 1], 0.05)), err_ester_hi=float(np.quantile(bt[:, 1], 0.95)),
                         err_alcohol=float(e0[masks['alcohol']].mean()), err_alcohol_lo=float(np.quantile(bt[:, 2], 0.05)),
                         err_alcohol_hi=float(np.quantile(bt[:, 2], 0.95))))
    res = pd.DataFrame(rows)
    res.to_csv(out / 'type_split.csv', index=False)
    print(res.round(3).to_string(index=False), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['D', 'A', 'B', 'C', 'tables'])
    args = ap.parse_args()
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'D':
        for f in (amplitude_by_breadth, breadth_weighted, type_split):
            f(out)
    elif args.part in ('A', 'B', 'C'):
        if args.part == 'A':
            import E80_pulse as Pp
            Pp.run_pulse(out)
        else:
            import E80_screen as S
            {'B': S.screen, 'C': S.gate}[args.part](out)
    else:
        raise NotImplementedError(args.part)


if __name__ == '__main__':
    main()
