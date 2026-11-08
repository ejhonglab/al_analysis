#!/usr/bin/env python3
"""E37 stage 0: spike-only APL at the three E36 spike-driven optima, every fitted scale held.

The structure-loss joint fits with APL driven by the KCs' spikes (E36; APL->bouton scale capped
at 3x the spike-driven base) left APL's suppression odor-selective: its rank correlation with
the APL-on response fraction is -0.67 to -0.78 (data -0.12). Two candidate sources: the
bouton->APL input (wPNAPL), still on in the spike-driven base, and the KC-KC term. At each
optimum, with every fitted scale held:
  ref             the fit itself, re-simulated (checked against its fits.csv row and, where
                  they exist, its saved outputs)
  nopnapl         (i) bouton->APL input off (wPNAPL_scale 0; the APL->bouton output kept)
  nokckc          (ii) KC-KC scale 0
  nopnapl_nokckc  (iii) both
(i)-(iii) with the somatic threshold re-tuned to the reference's APL-on response rate (E1's
bisect_threshold); the APL-on rate of (i)-(iii) at the fit's own threshold is recorded too.
Every variant also with acute APL off (APL output weights zero; KC-KC as in the variant).

Optima (E36 tags): 70 Hz without the gain (E36_spike_apl spk_thr70_k2_s2), 70 Hz with the 1.5x
gain on the eight oracle glomeruli (E36_spike_apl spk_thr70_gain1.5_k2_s2) and 40 Hz with the
gain (E36_spike_apl_jointfit spk_thr40_gain1.5_k2_cap3_s2). Each is rebuilt as E10_joint_fit.py
built it (its bouton_base, breadth weights and simulator kwargs).

Readouts, one row per variant in stats_variants.csv (harness column names): E35's suppression
uniformity (`suppression_stats`: spread, rank correlation with the APL-on fraction, ratio by odor
group), response rate ρ with APL on and off, the block, ρ on odor-odor distances with both KC
matrices, isotonic R², the APL-off / APL-on response-rate ratio, the natural-source alignment,
the structure against wild-type somata, the extended loss of the structure-loss fits (E10's
`make_loss(structure=True)`, with the current data-target registry: `loss` and `fit_*` columns),
and APL's activity per odor over the odor window (peak and mean, as E35_apl_margins.py) with its
spread across odors (apl_peak_* columns; per odor in apl_activity.csv).

Each variant writes its own part (results/E37_spike_only_apl/parts/<tag>.json), so runs with
different --fits / --variants can go in parallel. Every run, and --figure-only, merges the parts
into stats_variants.csv, per_odor_response_fractions.csv, apl_activity.csv,
stage0_pair_distances_long.csv, tuned_params.json (absolute parameters per variant, usable as
`E10_joint_fit.py --init-params results/E37_spike_only_apl/tuned_params.json:<tag>`) and
provenance.json, and draws E37_spike_only_apl.png and E37_per_odor_suppression.png. Model
outputs (KC x odor responses and spike counts per variant and condition) go to
./outputs/E37_spike_only_apl/.

Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E37_spike_only_apl.py
           [--fits spk_thr70_k2_s2 spk_thr70_gain1.5_k2_s2 spk_thr40_gain1.5_k2_cap3_s2]
           [--variants ref nopnapl nokckc nopnapl_nokckc] [--redo]
       ./venv/bin/python E37_spike_only_apl.py --figure-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sstats

import kc_remapping_harness as H
import E10_joint_fit as J
from E1_apl_knockout import bisect_threshold
from E3_drive_dependent_gain import per_odor_extras, rate
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess
from E35_divisive_apl import suppression_stats

EXPERIMENT = 'E37_spike_only_apl'
# E36 fit tag -> (experiment holding its fits.csv, label)
FITS = {
    'spk_thr70_k2_s2': ('E36_spike_apl', '70 Hz, no gain'),
    'spk_thr70_gain1.5_k2_s2': ('E36_spike_apl', '70 Hz, gain 1.5'),
    'spk_thr40_gain1.5_k2_cap3_s2': ('E36_spike_apl_jointfit', '40 Hz, gain 1.5'),
}
# variant -> (bouton->APL input off, KC-KC off)
VARIANTS = {
    'ref': (False, False),
    'nopnapl': (True, False),
    'nokckc': (False, True),
    'nopnapl_nokckc': (True, True),
}
VARIANT_LABEL = {'ref': 'the fit', 'nopnapl': 'bouton->APL input off', 'nokckc': 'KC-KC off',
    'nopnapl_nokckc': 'both off'}
VARIANT_COLOR = {'ref': '#6b7280', 'nopnapl': '#1d4ed8', 'nokckc': '#c2410c', 'nopnapl_nokckc': '#7c3aed'}
PARAMS = J.PARAM_KEYS + ('wKCKC_scale',)
OFF = {'wAPLKC_scale': 0.0, 'wAPLPN_scale': 0.0}


def fit_row(fit: str) -> pd.Series:
    """The E36 fit's row of its fits.csv (parameters parsed exactly as written)."""
    exp, _ = FITS[fit]
    f = pd.read_csv(H.shared_dir() / 'results' / exp / 'fits.csv', float_precision='round_trip').set_index('tag')
    return f.loc[fit]


def setup_fit(orn: pd.DataFrame, out_dir: Path, r: pd.Series) -> dict:
    """The fit's model as E10_joint_fit.py built it: the spike-driven bouton base (its
    `base_params`), the gained wiring if the fit had the gain, breadth-weighted APL->KC
    weights, and the simulator kwargs. Returns the fitted parameters (absolute), the kwargs and
    the wiring."""
    if r['apl_drive'] != 'spikes' or pd.notna(r.get('wiring')):
        raise ValueError(f'{r.name}: expected a spike-driven fit on the base wiring')
    claw_thr = None if pd.isna(r['claw_thr']) else float(r['claw_thr'])
    gain = None if (pd.isna(r['gain']) or float(r['gain']) == 1.0) else float(r['gain'])
    gloms = str(r['gain_glomeruli']).split(',') if gain is not None else None
    k = float(r['k'])
    _, wnorm, wPNKC, breadth = J.bouton_base(orn, out_dir, claw_thr, gain=gain, gain_gloms=gloms,
        apl_drive='spikes', base_params=r['base_params'])
    params = {p: float(r[p]) for p in J.PARAM_KEYS}
    params['wKCKC_scale'] = float(r['kc_kc_inh'])
    kws = dict(J.base_kws(claw_thr, 'spikes'), _wAPLKC=J.breadth_weights(wnorm['wAPLKC'], breadth, k),
        _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'],
        kc_kc_inh=params['wKCKC_scale'], kc_kc_tau=float(r['kc_kc_tau'])
    )
    if gain is not None:
        kws['_wPNKC'] = wPNKC
    return dict(params=params, kws=kws, wPNKC=wPNKC, claw_thr=claw_thr, gain=gain, gloms=gloms, k=k)


def apl_activity(orn: pd.DataFrame, params: Dict[str, float], kws: dict,
    responses: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[bool]]:
    """APL's activity per odor over the odor window (time >= 0): peak and mean, as
    E35_apl_margins.py measured them.

    One fit_mb_model call at the fixed parameters, then the KC layer re-run with only APL's
    trace recorded (olfsysm `save_inh_sims`; `return_dynamics` would also keep every KC and
    claw trace). Returns the table (odor abbreviations as index) and, if `responses` (the same
    variant's APL-on responses from the simulator) is passed, whether the re-run's per-odor
    response counts equal them.
    """
    import olfsysm as osm
    from al_analysis.mb_model import fit_mb_model, _fixed_param_model_kws, get_sim_var, get_time_index
    _, _, _, pdict = fit_mb_model(orn, return_olfsysm_vars=True, **_fixed_param_model_kws(kws, params))
    mp = pdict.pop('mp')
    rv = pdict.pop('rv')
    del pdict
    mp.kc.save_inh_sims = True
    rv.log.disable()
    rv.verbose = False
    osm.run_KC_sims(mp, rv, False)
    inh = np.asarray(get_sim_var(rv.kc, 'inh_sims'), dtype=float)
    inh = inh.reshape(inh.shape[0], -1)
    ts = get_time_index(mp, rv).to_numpy()
    odors = [H.odor_abbrev(o) for o in orn.columns]
    if inh.shape != (len(odors), len(ts)):
        raise RuntimeError(f'APL trace shape {inh.shape}, expected ({len(odors)}, {len(ts)})')
    post = inh[:, ts >= 0]
    tab = pd.DataFrame({'apl_peak': post.max(axis=1), 'apl_mean': post.mean(axis=1)}, index=odors)
    match = None
    if responses is not None:
        # olfsysm's columns are in orn's column order
        n_rerun = pd.Series((np.asarray(rv.kc.responses) != 0).sum(axis=0), index=odors)
        n_sim = (H.abbrev_columns(responses.astype(float)) != 0).sum(axis='index')
        match = bool((n_rerun.reindex(n_sim.index) == n_sim).all())
    del mp, rv
    return tab, match


def apl_spread(tab: pd.DataFrame, on_frac: pd.Series, off_frac: pd.Series) -> dict:
    """APL's activity across odors: range and spread of its peak (and the spread of its mean),
    and the rank correlation of the peak with the odor's APL-off / APL-on response fraction."""
    pk = tab['apl_peak'].reindex(H.APL16_ODORS)
    mn = tab['apl_mean'].reindex(H.APL16_ODORS)
    return dict(apl_peak_mean=float(pk.mean()), apl_peak_min=float(pk.min()), apl_peak_max=float(pk.max()),
        apl_peak_max_over_min=float(pk.max() / pk.min()) if pk.min() > 0 else np.nan,
        apl_peak_sd_log=float(np.log(pk[pk > 0]).std(ddof=1)), apl_peak_cv=float(pk.std(ddof=1) / pk.mean()),
        apl_mean_sd_log=float(np.log(mn[mn > 0]).std(ddof=1)),
        spearman_apl_peak_vs_off_frac=float(sstats.spearmanr(pk, off_frac.reindex(H.APL16_ODORS))[0]),
        spearman_apl_peak_vs_on_frac=float(sstats.spearmanr(pk, on_frac.reindex(H.APL16_ODORS))[0]),
    )


def saved_spike_counts(exp: str, tag: str, cond: str) -> Optional[pd.DataFrame]:
    """The fit's own saved spike counts (this host's outputs, else the shared archive)."""
    for d in (H.HERE / 'outputs' / exp, H.shared_dir() / 'data' / 'model_outputs' / exp):
        p = d / f'{tag}_{cond}_spike_counts.parquet'
        if p.exists():
            return pd.read_parquet(p)
    return None


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (pd.Timestamp, Path)):
        return str(x)
    raise TypeError(f'not JSON serializable: {type(x)}')


def run_fit(fit: str, variants, *, orn, orn_pairs, drive, ref, data, data_rates, loss_fn,
    res_dir: Path, out_dir: Path, redo: bool) -> None:
    """Runs the requested variants at one E36 optimum, writing one part per variant."""
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    parts = res_dir / 'parts'
    parts.mkdir(exist_ok=True)
    todo = [v for v in variants if redo or not (parts / f'{fit}_{v}.json').exists()]
    if not todo:
        print(f'{fit}: all requested variants done', flush=True)
        return
    exp, label = FITS[fit]
    r = fit_row(fit)
    t0 = time.time()
    s = setup_fit(orn, out_dir, r)
    fit_params = s['params']
    sim, _ = make_fixed_param_kc_simulator(orn, initial_params=fit_params, **s['kws'])
    print(f'{fit}: model rebuilt ({time.time() - t0:.0f}s); fitted parameters {fit_params}', flush=True)
    r_ref_on, s_ref_on = sim(fit_params)
    target = rate(r_ref_on)
    v = H.PRIMARY_MODEL_RDM
    for var in todo:
        t1 = time.time()
        pnapl_off, kckc_off = VARIANTS[var]
        over = {}
        if pnapl_off:
            over['wPNAPL_scale'] = 0.0
        if kckc_off:
            over['wKCKC_scale'] = 0.0
        tag = f'{fit}_{var}'
        if var == 'ref':
            p_v, thr_f, rate_fit_thr = dict(fit_params), 1.0, target
            r_on, s_on = r_ref_on, s_ref_on
        else:
            r_ft, _ = sim(dict(fit_params, **over))
            rate_fit_thr = rate(r_ft)
            thr, thr_f, r_on, s_on = bisect_threshold(sim, fit_params['fixed_thr'], target, extra=over)
            p_v = dict(fit_params, **over, fixed_thr=float(thr))
        r_off, s_off = sim(dict(p_v, **OFF))
        for cond, (rr, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
            rr.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
            sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        apl, apl_match = apl_activity(orn, p_v, s['kws'], responses=r_on)
        on_frac, off_frac = H.response_fractions(r_on), H.response_fractions(r_off)
        meta = dict(variant=tag, tag=tag, base='bouton', seed=None, point=fit, source_experiment=exp,
            source_label=label, manipulation=var, bouton_apl_input=not pnapl_off, kc_kc_on=not kckc_off,
            claw_thr=s['claw_thr'], k=s['k'], gain=(s['gain'] if s['gain'] is not None else np.nan),
            gain_glomeruli=(','.join(s['gloms']) if s['gloms'] else None),
            apl_drive=('spikes_only' if pnapl_off else 'spikes'), pn_claw_to_apl=False,
            base_params=r['base_params'], fixed_thr=p_v['fixed_thr'], fit_fixed_thr=fit_params['fixed_thr'],
            thr_factor=p_v['fixed_thr'] / fit_params['fixed_thr'], thr_retuned=(var != 'ref'),
            wAPLKC_scale=p_v['wAPLKC_scale'], wKCAPL_scale=p_v['wKCAPL_scale'],
            wAPLPN_scale=p_v['wAPLPN_scale'], wPNAPL_scale=p_v['wPNAPL_scale'],
            wKCKC_scale=p_v['wKCKC_scale'], kc_kc_inh=p_v['wKCKC_scale'], kc_kc_tau=s['kws']['kc_kc_tau'],
            target_rate=target, rate_on_at_fit_thr=rate_fit_thr, rate_on=rate(r_on), rate_off=rate(r_off),
            rate_ratio_off_over_on=rate(r_off) / max(rate(r_on), 1e-9), n_kcs=len(r_on),
            n_glomeruli=s['wPNKC'].shape[1],
            mean_claws_per_kc=float(kc_by_glomerulus(s['wPNKC']).sum(axis='columns').mean()),
            apl_rerun_matches=apl_match,
        )
        row, on, off = score(meta, (r_on, s_on), (r_off, s_off), s['wPNKC'], orn, orn_pairs, drive, data, ref)
        row.update(per_odor_extras(r_on, r_off, on[v], off[v], drive, data_rates))
        for cond, sp in (('on', s_on), ('off', s_off)):
            row.update({f'{cond}_{kk}': vv for kk, vv in density_excess(sp).items()})
        row.update(suppression_stats(on_frac, off_frac, data_rates))
        row.update(apl_spread(apl, on_frac, off_frac))
        losses = loss_fn({'on': (r_on, s_on), 'off': (r_off, s_off)})
        row['loss'] = losses['loss']
        row.update({f'fit_{kk}': vv for kk, vv in losses.items() if kk != 'loss'})
        if var == 'ref':
            # the fit's own record (its loss used the targets of its day)
            row['fitscsv_loss'] = float(r['loss'])
            for c in ('rate_on', 'rate_off', 'fit_on_spearman', 'fit_off_spearman', 'fit_block', 'fit_iso_r2',
                    'fit_po_on_spearman', 'fit_po_off_spearman', 'fit_wt_dev_on', 'fit_vcf_off'):
                row[f'fitscsv_{c}'] = float(r[c])
            for cond, sp in (('on', s_on), ('acute_off', s_off)):
                saved = saved_spike_counts(exp, fit, cond)
                if saved is not None:
                    same = saved.shape == sp.shape and saved.index.equals(sp.index) and saved.columns.equals(sp.columns)
                    row[f'saved_{cond}_n_differ'] = (int((saved.to_numpy() != sp.to_numpy()).sum())
                        if same else -1)
        row['seconds'] = time.time() - t1
        part = dict(tag=tag, fit=fit, variant=var, row=row,
            fracs={'on': on_frac.to_dict(), 'acute_off': off_frac.to_dict()},
            apl={c: apl[c].to_dict() for c in apl.columns},
            pairs={cond: [[a, b, float(d)] for (a, b), d in p[v].items()] for cond, p in (('on', on), ('acute_off', off))},
            params=dict(p_v, claw_thr=s['claw_thr'], k=s['k'], gain=s['gain'], gain_glomeruli=s['gloms'],
                kc_kc_inh=p_v['wKCKC_scale'], kc_kc_tau=s['kws']['kc_kc_tau'],
                apl_drive=meta['apl_drive'], source_fit=f'{exp}:{fit}', fit_fixed_thr=fit_params['fixed_thr'],
                thr_factor=meta['thr_factor'], target_rate=target, base_params=r['base_params']),
            run=dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), host=socket.gethostname(),
                seconds=row['seconds'], model=start_provenance(orn)),
        )
        (parts / f'{tag}.json').write_text(json.dumps(part, indent=1, default=_jsonable) + '\n')
        print(f"{tag}: thr x{thr_f:.3f} (APL-on rate at the fit's threshold {rate_fit_thr:.4f}); rates on "
            f"{meta['rate_on']:.4f} off {meta['rate_off']:.4f} (x{meta['rate_ratio_off_over_on']:.2f}); "
            f"suppression spread {row['suppression_sd_log']:.2f}, rho vs on-frac {row['spearman_suppression_vs_on_frac']:+.2f}; "
            f"response rate rho on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; "
            f"block {row[v + '__block_fruity_small_iso']:+.3f}; rho ctrl/TNT {row[v + '__on_spearman']:.3f}/"
            f"{row[v + '__off_spearman']:.3f}; iso R2 {row[v + '__iso_r2']:.3f}; wt {row[v + '__on_spearman_resid_vs_remap_wt']:.2f}; "
            f"vcf off {row[v + '__off_spearman_resid_vs_vcf_hyp']:.2f}; loss {row['loss']:.3f}; APL peak "
            f"{row['apl_peak_min']:.1f}-{row['apl_peak_max']:.1f} (sd log {row['apl_peak_sd_log']:.2f}, rerun match "
            f"{apl_match}); {row['seconds']:.0f}s", flush=True)
        if var == 'ref':
            print(f"{tag} vs its fits.csv row: " + ', '.join(f"{c} {row[c]:.4f}/{row['fitscsv_' + c]:.4f}"
                for c in ('rate_on', 'rate_off', 'fit_on_spearman', 'fit_off_spearman', 'fit_block', 'fit_iso_r2',
                    'fit_po_on_spearman', 'fit_po_off_spearman', 'fit_wt_dev_on', 'fit_vcf_off'))
                + f"; loss {row['loss']:.4f} (fits.csv {row['fitscsv_loss']:.4f}, older targets); saved outputs "
                f"differ in {row.get('saved_on_n_differ', 'n/a')} / {row.get('saved_acute_off_n_differ', 'n/a')} "
                "KC x odor spike counts (on / off)", flush=True)
    del sim


def merge(res_dir: Path, data_rates: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collects the parts into the experiment's tables, tuned_params.json and provenance.json."""
    order = {f'{f}_{v}': (i, j) for i, f in enumerate(FITS) for j, v in enumerate(VARIANTS)}
    parts = [json.loads(p.read_text()) for p in sorted((res_dir / 'parts').glob('*.json'))]
    parts = sorted(parts, key=lambda p: order.get(p['tag'], (99, 99)))
    if not parts:
        raise FileNotFoundError(f'no parts in {res_dir / "parts"}')
    stats = pd.DataFrame([p['row'] for p in parts])
    stats.to_csv(res_dir / 'stats_variants.csv', index=False)
    fracs = pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']}).reindex(H.APL16_ODORS)
    apl = pd.DataFrame(index=H.APL16_ODORS)
    long = []
    for p in parts:
        for cond, d in p['fracs'].items():
            fracs[f"{p['tag']}_{cond}"] = pd.Series(d).reindex(fracs.index)
        for c, d in p['apl'].items():
            apl[f"{p['tag']}_{c}"] = pd.Series(d).reindex(apl.index)
        for cond, rows in p['pairs'].items():
            long += [dict(tag=p['tag'], condition=cond, odor_a=a, odor_b=b, distance=d) for a, b, d in rows]
    fracs.index.name = apl.index.name = 'odor'
    fracs.to_csv(res_dir / 'per_odor_response_fractions.csv')
    apl.to_csv(res_dir / 'apl_activity.csv')
    pd.DataFrame(long).to_csv(res_dir / 'stage0_pair_distances_long.csv', index=False)
    (res_dir / 'tuned_params.json').write_text(json.dumps({p['tag']: p['params'] for p in parts}, indent=1,
        default=_jsonable) + '\n')
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E37_spike_only_apl.py',
        merged_on=socket.gethostname(), merged_at_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        fits={f: dict(experiment=e, label=l) for f, (e, l) in FITS.items()},
        manipulations={v: dict(bouton_apl_input_off=a, kc_kc_off=b, label=VARIANT_LABEL[v])
            for v, (a, b) in VARIANTS.items()},
        design='each E36 optimum rebuilt as E10_joint_fit.py built it, every fitted scale held; variants other than '
            "ref with the somatic threshold re-tuned (E1 bisect_threshold, 2 % relative) to ref's APL-on response "
            'rate; acute APL off = wAPLKC_scale and wAPLPN_scale 0',
        # the bouton base's kwargs (each row's per-claw threshold is in its claw_thr column)
        base=J.base_kws(None, 'spikes'),
        base_kws_by_fit={f: J.base_kws(None if pd.isna(fit_row(f)['claw_thr']) else float(fit_row(f)['claw_thr']),
            'spikes') for f in FITS},
        tau=J.TAU,
        loss_weights=dict(J.LOSS_WEIGHTS, **J.STRUCTURE_WEIGHTS),
        data_targets_sha256=hashlib.sha256(H.data_targets_path().read_bytes()).hexdigest(),
        approx_conc_note=H.APPROX_CONC_NOTE,
        # the first run's model provenance; every run's is under runs
        model=parts[0]['run']['model'],
        model_same_for_all_runs=len({json.dumps([p['run']['model']['al_analysis'].get('git_hash'),
            p['run']['model']['al_analysis'].get('diff_sha256'), p['run']['model']['olfsysm'].get('build')],
            default=str, sort_keys=True) for p in parts}) == 1,
        runs={p['tag']: dict(params=p['params'], **p['run']) for p in parts},
    )
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=_jsonable) + '\n')
    return stats, fracs, apl


def make_figures(stats: pd.DataFrame, fracs: pd.DataFrame, res_dir: Path) -> None:
    """Bars per optimum and variant (spread and rank correlation of the suppression, response
    rate ρ, block, extended loss, APL peak spread), and per-odor suppression against the APL-on
    fraction at each optimum."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import ScalarFormatter
    d = H.data_target
    p = H.PRIMARY_MODEL_RDM + '__'
    st = stats.set_index('tag')
    fits = [f for f in FITS if any(t.startswith(f + '_') for t in st.index)]
    variants = [v for v in VARIANTS if any(f'{f}_{v}' in st.index for f in fits)]
    x = np.arange(len(fits))
    w = 0.8 / max(len(variants), 1)

    def val(col, f, var):
        t = f'{f}_{var}'
        return float(st.loc[t, col]) if (t in st.index and col in st.columns) else np.nan

    panels = [
        ('suppression_sd_log', None, 'spread of suppression across odors\n(sd of log APL-off / APL-on responders)',
            [(d('supp_spread'), ':', 'KCs', '.2f'), (d('supp_noise'), '--', 'bootstrap noise of the KC data', '.2f')]),
        ('spearman_suppression_vs_on_frac', None, 'rank correlation of the suppression\nwith the APL-on response fraction',
            [(d('supp_rho_on'), ':', 'KCs', '+.2f')]),
        ('spearman_rate_on_vs_data', 'spearman_rate_off_vs_data',
            'response rate ρ: APL on vs control flies (bars),\nAPL off vs APL output blocked (TNT) (circles)',
            [(d('ctrl_rate_rho_ceiling'), ':', 'ceiling, control flies', '.2f'),
                (d('tnt_rate_rho_ceiling'), '--', 'ceiling, APL output blocked (TNT)', '.2f')]),
        (p + 'block_fruity_small_iso', None, 'fruity × small-ester block (on vs off)', [(d('block'), ':', 'KCs', '+.3f')]),
        ('loss', None, 'extended loss of the structure-loss fits\n(lower is better; KC data 0)', []),
        ('apl_peak_sd_log', None, "spread of APL's peak across odors\n(sd of log peak during the odor)", []),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (col, col2, title, refs) in zip(axes.ravel(), panels):
        for j, var in enumerate(variants):
            xs = x + (j - (len(variants) - 1) / 2) * w
            ax.bar(xs, [val(col, f, var) for f in fits], w, color=VARIANT_COLOR[var], edgecolor='white')
            if col2 is not None:
                ax.plot(xs, [val(col2, f, var) for f in fits], 'o', mfc='white', mec=VARIANT_COLOR[var], ms=6)
        for y, ls, lab, fmt in refs:
            ax.axhline(y, color='0.25' if ls == ':' else '0.6', lw=1, ls=ls, label=f'{lab} ({y:{fmt}})')
        if col == 'spearman_rate_on_vs_data':
            # room above the ceilings for their legend
            ax.set_ylim(top=1.3)
        if refs:
            ax.legend(fontsize=7, frameon=False)
        ax.axhline(0, color='0.8', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([FITS[f][1] for f in fits], fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, axis='y', color='0.92', lw=0.6)
    fig.legend(handles=[Patch(color=VARIANT_COLOR[v], label=VARIANT_LABEL[v]) for v in variants], loc='lower center',
        ncol=len(variants), fontsize=9, frameon=False)
    fig.suptitle('Spike-only APL at the spike-driven structure-loss optima, fitted scales held '
        '(threshold re-tuned to the fit\'s APL-on response rate)', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(res_dir / 'E37_spike_only_apl.png', dpi=130)
    plt.close(fig)

    floor = 0.5 / 1732
    fig, axes = plt.subplots(1, len(fits), figsize=(5.2 * len(fits), 4.6), squeeze=False)
    d_on = fracs['data_TNT3c'].reindex(H.APL16_ODORS)
    d_ratio = (fracs['data_TNT'] / fracs['data_TNT3c']).reindex(H.APL16_ODORS)
    for ax, f in zip(axes[0], fits):
        ax.scatter(d_on, d_ratio, s=30, color='black', zorder=5,
            label='KCs: APL output blocked (TNT) / control flies')
        for var in variants:
            t = f'{f}_{var}'
            if f'{t}_on' not in fracs:
                continue
            on = fracs[f'{t}_on'].reindex(H.APL16_ODORS)
            off = fracs[f'{t}_acute_off'].reindex(H.APL16_ODORS)
            ax.scatter(on, off / on.clip(lower=floor), s=20, color=VARIANT_COLOR[var], alpha=0.9,
                label=f'model, {VARIANT_LABEL[var]}')
            if var in ('ref', 'nopnapl', 'nopnapl_nokckc'):
                for o in ('ms', 'paa', 'B-myr'):
                    ax.annotate(o, (on[o], off[o] / max(on[o], floor)), xytext=(3, 2), textcoords='offset points',
                        fontsize=7, color=VARIANT_COLOR[var])
        ax.set_xscale('symlog', linthresh=0.005)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.tick_params(axis='y', which='minor', labelsize=7)
        ax.set_xlabel('fraction of KCs responding with APL on (model) or in control flies (KCs)', fontsize=8)
        ax.set_ylabel('responders with APL off / with APL on', fontsize=8)
        ax.set_title(FITS[f][1], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle('Per-odor APL suppression against the APL-on response fraction\n(odors without APL-on '
        'responders counted as half a KC)', fontsize=10)
    fig.tight_layout()
    fig.savefig(res_dir / 'E37_per_odor_suppression.png', dpi=130)
    plt.close(fig)


_START_PROVENANCE = {}


def start_provenance(orn):
    """The model provenance (code commits, build, inputs), taken once at the start of the run, so
    that a pull in this checkout during the run does not change the commit recorded for it."""
    if 'p' not in _START_PROVENANCE:
        _START_PROVENANCE['p'] = H.model_provenance(orn_deltas=orn)
    return _START_PROVENANCE['p']


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--fits', nargs='+', default=list(FITS), choices=list(FITS), help='E36 optima to run')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--redo', action='store_true', help='re-run variants whose parts already exist')
    ap.add_argument('--figure-only', action='store_true', help='only merge the parts and redraw the figures')
    ap.add_argument('--experiment', default=EXPERIMENT, help='results / outputs folder name (e.g. for a test run)')
    args = ap.parse_args()
    res_dir = H.results_dir(args.experiment)
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    if not args.figure_only:
        out_dir = H.outputs_dir(args.experiment)
        orn = H.apl16_orn_deltas()
        start_provenance(orn)
        orn_pairs = H.orn16_consensus_pairs()
        drive = H.consensus_orn_drive()
        ref = H.reference_pairs()
        data = {
            'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
            'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
            'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')},
        }
        loss_fn = J.make_loss(data, data_rates, orn_pairs, drive, ref=ref, structure=True)
        for fit in args.fits:
            run_fit(fit, args.variants, orn=orn, orn_pairs=orn_pairs, drive=drive, ref=ref, data=data,
                data_rates=data_rates, loss_fn=loss_fn, res_dir=res_dir, out_dir=out_dir, redo=args.redo)
    stats, fracs, _ = merge(res_dir, data_rates)
    make_figures(stats, fracs, res_dir)
    print('done ->', res_dir, flush=True)


if __name__ == '__main__':
    main()
