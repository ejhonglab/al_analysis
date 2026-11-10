#!/usr/bin/env python3
"""E49: Yang's low-concentration KC data as constraints: do the fitted models recruit KCs at
10^-7, and does APL act on them as in the flies?

Fixed-parameter predictions (nothing re-tuned) of models fitted or tuned on the APL16 panel, run
on the APL16 panel plus the low-concentration odors of Yang's diagnostic-odor mixture panel
(diag_mix: 2h and ma at 10^-7 and 10^-8, farn at 10^-3) and dilute natural-mixture panel
(natmix_dilute: 2h at 10^-5, ea at 10^-5 and 10^-7) for which ORN spike deltas exist within Tom's
one-log-unit rule (data/yang_weak_odor_kc/orn_input_availability.csv; the substitutions are
recorded per odor), and on the binary mixtures whose components are covered, with the components'
ORN deltas combined as their elementwise maximum per glomerulus (Tom's rule, 2026-11-08: the sum overestimates
most glomeruli, the max fits the lab's measured mixtures better; the measured air mixtures of these odors are at
other concentrations).
Odors are simulated independently in olfsysm at fixed thresholds and APL weights, so adding
columns changes nothing for the APL16 odors; the APL16 columns are checked against each fit's
saved outputs (checks.json).

Models (`--models`): base70 (the 70 Hz claw-threshold bouton base tuned at 5 %, E16 / E19),
ref40 (E34's thr40_gain1.5_k2_cap3, the reference model at 5 %), the reference model re-fitted at
the control flies' 6.8 % (E44, 40 and 70 Hz) and the eight glomeruli at 1.25 at 70 Hz (E45),
built from their fits.csv rows as E43_model_trial_noise.py builds them (`fit_model`).

Part A: deterministic APL-on and acute APL-off runs: per-odor response fractions against the
data's (control flies and APL output blocked; trial-mean > 1), the per-odor APL-off / APL-on ratio
(the data's 2.3-3.1 for these odors against 2.2 for the APL16 mean), the responders' breadth over
the APL16 odors. Part B (unless --no-noise): the same through E43's matched per-trial KC
threshold jitter (Gaussian, each model's APL-on-matched sigma and re-tuned threshold from
results/E43_model_trial_noise/partB or results/E48_block_target_noise/partB), n pseudo-recordings
x trials, response fractions under the data's rule (trial-mean spike count >= 1, 'count1') and
per trial. Part C: the mixtures against the union of the components' responders (model) and
Yang's union prediction (data).

Outputs: <shared>/results/E49_low_conc_kc/ (fractions.csv, mixtures.csv, checks.json,
provenance.json, E49_low_conc_kc.png, comparison_table.md) and ./outputs/E49_low_conc_kc/
(deterministic responses / spike counts per model, trials as npz, the bases' cache files).
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E49_low_conc_kc.py [--models ...] [--no-noise]
       [--n-recordings 10] [--n-trials 3] [--local-test DIR]
       ./venv/bin/python E49_low_conc_kc.py --summary   (tables and figure only)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E10_joint_fit import base_kws, e19_row
import E43_model_trial_noise as E43

EXPERIMENT = 'E49_low_conc_kc'
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
CONDS = ('on', 'acute_off')
PANEL = 'yang_lowconc'
# model name -> fits.csv source (None for the base) and a label
MODEL_SOURCES: Dict[str, Optional[Tuple[str, str]]] = {
    'base70': None,
    'ref40': ('E34_gain_bounded', 'thr40_gain1.5_k2_cap3'),
    'ref40_68': ('E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'),
    'ref70_68': ('E44_operating_point', 'thr70_gain1.5_rate0.068_k2_cap3'),
    'g125_70_68': ('E45_gain_at_ctrl_frac', 'thr70_gain1.25_rate0.068_k2_cap3'),
    # E72 part F: the breadth-2 APL-effect fit (C2) and the no-gain 40 Hz fit of the APL-effect fits (E67)
    'c2_68': ('E67_apl_effect_fits', 'rate0.068_aplcal_k2_cap3'),
    'nogain40_68': ('E67_apl_effect_fits', 'thr40_rate0.068_aplcal_k2_cap3'),
}
# the models E49 itself ran (the default of --models)
E49_MODELS = ('base70', 'ref40', 'ref40_68', 'ref70_68', 'g125_70_68')
MODEL_LABELS = {
    'base70': '70 Hz threshold base, tuned at 5 %',
    'ref40': 'the reference model (eight at 1.5, 40 Hz), fitted at 5 %',
    'ref40_68': 'the reference model (eight at 1.5, 40 Hz), fitted at 6.8 %',
    'ref70_68': 'eight at 1.5, 70 Hz, fitted at 6.8 %',
    'g125_70_68': 'eight at 1.25, 70 Hz, fitted at 6.8 %',
    'c2_68': "the first joint fit's family at breadth 2 (no gain, no per-claw threshold), APL-effect fit at 6.8 %",
    'nogain40_68': 'no gain, 40 Hz per-claw threshold, APL-effect fit at 6.8 %',
}
# the E43 / E48 part B model names whose matched sigma each model uses
NOISE_SOURCE = {
    'base70': ('E43_model_trial_noise', 'base70'),
    'ref40': ('E43_model_trial_noise', 'ref40'),
    'ref40_68': ('E48_block_target_noise', 'E44-thr40_gain1.5_rate0.068_k2_cap3'),
    'ref70_68': ('E48_block_target_noise', 'E44-thr70_gain1.5_rate0.068_k2_cap3'),
    'g125_70_68': ('E48_block_target_noise', 'E45-thr70_gain1.25_rate0.068_k2_cap3'),
    'c2_68': ('E67_apl_effect_fits', 'E67-rate0.068_aplcal_k2_cap3'),
}
# Yang's odor (olfdb spelling) -> the ORN spike-delta column that stands in for it
# (data/yang_weak_odor_kc/orn_input_availability.csv; exact or within one log unit, the stronger
# concentration where the data's is not measured)
ORN_SOURCE: Dict[str, Tuple[str, str]] = {
    '2h @ -7': ('glomeruli_diagnostics', '2h @ -6'),
    'ma @ -7': ('glomeruli_diagnostics', 'ma @ -7'),
    'ma @ -8': ('glomeruli_diagnostics', 'ma @ -7'),
    'farn @ -3': ('glomeruli_diagnostics', 'farn @ -2'),
    '2h @ -5': ('control', '2h @ -5'),
    'ea @ -5': ('kiwi', 'ea @ -4.2'),
    'ea @ -7': ('glomeruli_diagnostics', 'ea @ -8'),
}
# mixtures with both components covered: Yang's label -> components (spike deltas combined as the elementwise max
# per glomerulus, never the sum: Tom, 2026-11-08, CLAUDE.md "Data from Remy and Yang")
MIXTURES: Dict[str, Tuple[str, str]] = {
    '2h @ -7 + ma @ -7': ('2h @ -7', 'ma @ -7'),
    '2h @ -7 + farn @ -3': ('2h @ -7', 'farn @ -3'),
    'ma @ -7 + farn @ -3': ('ma @ -7', 'farn @ -3'),
}
# which panel's data each odor is compared with (both where both have it)
DATA_PANELS = {'2h @ -7': ('diag_mix', 'natmix_dilute'), 'ma @ -7': ('diag_mix', 'natmix_dilute'), 'ma @ -8': ('diag_mix',),
    'farn @ -3': ('diag_mix',), '2h @ -5': ('natmix_dilute',), 'ea @ -5': ('natmix_dilute',), 'ea @ -7': ('natmix_dilute',),
    '2h @ -7 + ma @ -7': ('diag_mix', 'natmix_dilute'), '2h @ -7 + farn @ -3': ('diag_mix',), 'ma @ -7 + farn @ -3': ('diag_mix',)}
# fit_mb_model parses odor names with hong2p's parse_odor_name (one ' @ '), so a mixture column gets a
# single-concentration name (as the natural-mixture panel's 'ea+eb @ 0'); the tables use Yang's label
MIX_COLNAME = {'2h @ -7 + ma @ -7': '2h+ma @ -7', '2h @ -7 + farn @ -3': '2h-7+farn-3 @ 0', 'ma @ -7 + farn @ -3': 'ma-7+farn-3 @ 0'}
COL_TO_LABEL = {v: k for k, v in MIX_COLNAME.items()}
EXTRA_ODORS = list(ORN_SOURCE) + list(MIXTURES)
# Yang's odors whose ORN input is the same column as another simulated odor's (farn @ -2.5 and farn @ -3 both map to
# farn @ -2, the only farn concentration in the lab's ORN data): the tables show them with the aliased odor's model
# values and their own KC data (Tom, 2026-11-08, via hal), flagged
ALIAS = {'farn @ -2.5': 'farn @ -3', '2h @ -7 + farn @ -2.5': '2h @ -7 + farn @ -3', 'ma @ -7 + farn @ -2.5': 'ma @ -7 + farn @ -3'}


def yang_dir() -> Path:
    return H.shared_dir() / 'data' / 'yang_weak_odor_kc'


def data_fractions() -> pd.DataFrame:
    """The data's per-odor response fractions (trial-mean > 1), per panel and genotype group, with
    the union-of-components prediction for mixtures: columns panel, group, odor, n_recordings,
    frac, sem, union."""
    rows = []
    for panel in ('diag_mix', 'natmix_dilute'):
        d = pd.read_csv(yang_dir() / panel / 'computed' / 'response_fraction_recording_mean.csv')
        d = d[d.grouping == 'genotype_group']
        for r in d.to_dict('records'):
            rows.append(dict(panel=panel, group=r['group'], odor=r['odor'], n_recordings=r['n_recordings'],
                frac=r['frac_trialmean_gt1_mean'], sem=r['frac_trialmean_gt1_sem'],
                union=r.get('frac_union_components_trialmean_mean', np.nan)))
    return pd.DataFrame(rows)


def substitutions() -> Dict[str, str]:
    """Each extra odor's ORN-input status from the data folder's availability table ('exact' or
    'substitute (<source>)')."""
    av = pd.read_csv(yang_dir() / 'orn_input_availability.csv').set_index('odor')
    out = {}
    for odor, (panel, col) in ORN_SOURCE.items():
        st = av.loc[odor, 'spike_deltas_status'] if odor in av.index else 'unlisted'
        out[odor] = 'exact' if st == 'exact' else f'{st}: {col} ({panel})'
    for mix, comps in MIXTURES.items():
        out[mix] = 'max of ' + ' and '.join(f'{c} [{out[c]}]' for c in comps)
    return out


def extended_orn(orn16: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """The APL16 input with the extra odors' columns appended (panel PANEL), on the APL16 input's
    36 glomeruli: the natural-mixture / diagnostics spike deltas restricted to those glomeruli
    (the ones they lack are filled with 0, recorded in the returned info)."""
    from al_analysis.mb_model import natmix_orn_deltas
    nm = natmix_orn_deltas(drop_diags=False)
    gl = list(orn16.index)
    missing = sorted(set(gl) - set(nm.index)); dropped = sorted(set(nm.index) - set(gl))
    nm = nm.reindex(gl).fillna(0.0)
    cols = {}
    for odor, (panel, col) in ORN_SOURCE.items():
        cols[odor] = nm[(panel, col)].astype(float)
    for mix, comps in MIXTURES.items():
        cols[MIX_COLNAME[mix]] = np.maximum(cols[comps[0]], cols[comps[1]])
    extra = pd.DataFrame(cols, index=orn16.index)   # keeps the index name ('glomerulus'), which fit_mb_model checks
    extra.columns = pd.MultiIndex.from_tuples([(PANEL, o) for o in extra.columns], names=orn16.columns.names)
    orn = pd.concat([orn16, extra], axis='columns')
    orn.index.name = orn16.index.name
    info = dict(glomeruli=gl, natmix_glomeruli_missing_filled_0=missing, natmix_glomeruli_dropped=dropped,
        extra_odors=EXTRA_ODORS, orn_source={k: list(v) for k, v in ORN_SOURCE.items()},
        mixtures={k: list(v) for k, v in MIXTURES.items()}, mixture_rule='elementwise max of the components per glomerulus')
    return orn, info


def build_model(name: str, orn16: pd.DataFrame, orn_ext: pd.DataFrame, out_dir: Path) -> dict:
    """The model at its fitted / tuned parameters (kws built on the APL16 input, as the fit was),
    simulated on the extended input; returns sim, params, mp / rv, saved-output paths, info."""
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    if name == 'base70':
        r = e19_row(70.0, 0.0, 0.0)
        p_on = {k: float(r[k]) for k in PARAM_KEYS}
        kws = dict(base_kws(70.0))
        hal = H.shared_dir() / 'data' / 'hal_outputs' / 'E16_claw_saturation'
        saved = {c: [hal / f'bouton_thr70_{c}_{kind}.parquet' for kind in ('responses', 'spike_counts')] for c in CONDS}
        info = dict(source='E19 stats row thr70_k0_kckc0 (E16 bouton_thr70, tuned at 5 % with APL on)', claw_thr=70.0, k=0.0, gain=1.0)
    else:
        experiment, tag = MODEL_SOURCES[name]
        e43name = E43.model_from(f'{experiment}:{tag}') if name != 'ref40' else 'ref40'
        fm = E43.fit_model(e43name, orn16, out_dir)
        if not fm['orn'].equals(orn16):
            raise NotImplementedError(f'{name}: the fit used another ORN input; not supported here')
        p_on, kws, saved, info = fm['p_on'], fm['kws'], fm['saved'], fm['info']
    store: dict = {}
    with E43.captured_olfsysm_vars(store):
        sim, _ = make_fixed_param_kc_simulator(orn_ext, initial_params=p_on, **kws)
    params = {'on': dict(p_on), 'acute_off': dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0)}
    return dict(sim=sim, params=params, mp=store['mp'], rv=store['rv'], saved=saved, info=info)


def odor_labels(df: pd.DataFrame) -> List[str]:
    """Column labels: the APL16 odors by abbreviation, the extra odors by Yang's label (the model's
    output columns may or may not keep the panel level, so the extra odors are recognised by name)."""
    extra_cols = set(ORN_SOURCE) | set(MIX_COLNAME.values())
    out = []
    for c in df.columns:
        odor = c[-1] if isinstance(c, tuple) else c
        out.append(COL_TO_LABEL.get(odor, odor) if odor in extra_cols else H.odor_abbrev(c))
    assert len(set(out)) == len(out), f'duplicate odor labels: {out}'
    return out


def frac_series(responses: pd.DataFrame) -> pd.Series:
    f = responses.astype(float).mean(axis=0)
    f.index = odor_labels(responses)
    return f


def matched_noise(name: str) -> dict:
    """Each model's APL-on-matched threshold-jitter sigma and fixed_thr ratio from the E43 / E48
    part B tables (Gaussian jitter, match_on level)."""
    exp, model = NOISE_SOURCE[name]
    st = pd.read_csv(H.shared_dir() / 'results' / exp / 'partB' / 'stats_noise.csv')
    r = st[(st.model == model) & (st.level.astype(str).str.startswith('match_on'))]
    if r.empty:
        raise ValueError(f'no match_on level for {model} in {exp}/partB/stats_noise.csv')
    r = r.iloc[0]
    return dict(sigma=float(r['sigma']), fixed_thr_ratio=float(r['fixed_thr_ratio']), source=f'results/{exp}/partB/stats_noise.csv ({model}, match_on)')


def noisy_trials(m: dict, sigma: float, thr_ratio: float, n_rec: int, n_tr: int, seed: int, n_odors: int) -> dict:
    """Per-trial KC threshold jitter as E43 part B: thr = max(thr0 (1 + sigma xi), 0.01 thr0) with
    thr0 = fixed_thr * ratio + 2 spont_in, one xi per KC and trial, the same thresholds for the
    APL-on and APL-off run. Returns {cond: recordings x trials x KCs x odors spike counts} and the
    per-trial responses."""
    spont = np.asarray(m['rv'].kc.spont_in, dtype=float).ravel()
    n_kcs = spont.size
    f0 = float(m['params']['on']['fixed_thr']) * thr_ratio
    thr0 = f0 + 2.0 * spont
    rng = np.random.default_rng(seed)
    out = {f'{c}_spk': np.zeros((n_rec, n_tr, n_kcs, n_odors), dtype=np.uint16) for c in CONDS}
    out.update({f'{c}_resp': np.zeros((n_rec, n_tr, n_kcs, n_odors), dtype=bool) for c in CONDS})
    for rep in range(n_rec):
        for tr in range(n_tr):
            xi = rng.standard_normal(n_kcs)
            thr = np.maximum(thr0 * (1.0 + sigma * xi), 0.01 * thr0)
            with E43.kc_threshold_override(thr):
                for c in CONDS:
                    p = dict(m['params'][c], fixed_thr=f0)
                    r, s = m['sim'](p)
                    out[f'{c}_resp'][rep, tr] = r.astype(float).to_numpy() > 0
                    out[f'{c}_spk'][rep, tr] = s.astype(float).to_numpy().astype(np.uint16)
    return out


def noisy_fractions(trials: dict, labels: List[str]) -> Dict[str, pd.DataFrame]:
    """Per-odor response fractions of the trials, per condition: 'count1' (trial-mean spike count
    >= 1, the data's rule) and 'trial' (per-trial responses averaged), each a mean over
    pseudo-recordings, with the sd over recordings."""
    out = {}
    for c in CONDS:
        spk = trials[f'{c}_spk'].astype(float); resp = trials[f'{c}_resp'].astype(float)
        count1 = (spk.mean(axis=1) >= 1).mean(axis=1)   # rec x odors
        trial = resp.mean(axis=(1, 2))                   # rec x odors
        out[c] = pd.DataFrame({'count1': count1.mean(axis=0), 'count1_sd': count1.std(axis=0, ddof=1),
            'trial': trial.mean(axis=0), 'trial_sd': trial.std(axis=0, ddof=1)}, index=labels)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--models', nargs='+', choices=list(MODEL_SOURCES), default=list(E49_MODELS),
        help='models to run (default: E49\'s five); a model without a matched-noise source runs deterministically only')
    ap.add_argument('--into', metavar='EXPERIMENT', default=None,
        help="write results and outputs under <EXPERIMENT>/low_conc/ instead of E49's folders (E72 part F)")
    ap.add_argument('--no-noise', action='store_true', help='part A and C only')
    ap.add_argument('--n-recordings', type=int, default=10)
    ap.add_argument('--n-trials', type=int, default=3)
    ap.add_argument('--seed', type=int, default=49)
    ap.add_argument('--summary', action='store_true', help='only (re)write the tables and the figure')
    ap.add_argument('--local-test', metavar='DIR', default=None, help='write results and outputs under DIR (a smoke test)')
    args = ap.parse_args()
    if args.local_test:
        res_dir = Path(args.local_test) / 'results'; out_dir = Path(args.local_test) / 'outputs'
        res_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    elif args.into:
        res_dir = H.results_dir(args.into) / 'low_conc'; out_dir = H.outputs_dir(args.into) / 'low_conc'
        res_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    else:
        res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    if args.summary:
        summarize(res_dir); return
    orn16 = H.apl16_orn_deltas()
    orn, orn_info = extended_orn(orn16)
    labels = odor_labels(orn)
    data = data_fractions()
    subs = substitutions()
    model_prov = H.model_provenance(orn_deltas=orn16)
    t0 = time.time()
    frac_rows, mix_rows, checks = [], [], {}
    (res_dir / 'orn_input_extra.csv').write_text(orn.xs(PANEL, axis='columns', level=0).to_csv())
    for name in args.models:
        t1 = time.time()
        m = build_model(name, orn16, orn, out_dir)
        det = {c: m['sim'](p) for c, p in m['params'].items()}
        for c, (r, s) in det.items():
            r.to_parquet(out_dir / f'{name}_{c}_responses.parquet'); s.to_parquet(out_dir / f'{name}_{c}_spike_counts.parquet')
        # the APL16 columns against the fit's saved outputs
        chk = {}
        for c in CONDS:
            paths = m['saved'][c]
            if all(p.exists() for p in paths):
                saved_r = pd.read_parquet(paths[0]).astype(float); saved_s = pd.read_parquet(paths[1]).astype(float)
                r16 = det[c][0].astype(float).iloc[:, :16]; s16 = det[c][1].astype(float).iloc[:, :16]
                chk[c] = dict(responses_mismatch=int((r16.to_numpy() != saved_r.to_numpy()).sum()),
                    spike_counts_mismatch=int((s16.to_numpy() != saved_s.to_numpy()).sum()), saved=str(paths[0]))
            else:
                chk[c] = dict(saved_missing=str(paths[0]))
        checks[name] = dict(chk, seconds_build=time.time() - t1, **{k: v for k, v in m['info'].items() if k != 'gain_glomeruli'})
        fr = {c: frac_series(det[c][0]) for c in CONDS}
        resp16 = {c: det[c][0].astype(float).iloc[:, :16] for c in CONDS}
        noise = None; nf = None
        if not args.no_noise and name in NOISE_SOURCE:
            noise = matched_noise(name)
            trials = noisy_trials(m, noise['sigma'], noise['fixed_thr_ratio'], args.n_recordings, args.n_trials, args.seed + len(frac_rows), len(labels))
            np.savez_compressed(out_dir / f'{name}_trials.npz', **trials, sigma=noise['sigma'], fixed_thr_ratio=noise['fixed_thr_ratio'])
            nf = noisy_fractions(trials, labels)
            checks[name].update(noise=noise, apl_on_rate_noisy_count1=float(nf['on']['count1'].iloc[:16].mean()),
                apl_on_rate_det=float(fr['on'].iloc[:16].mean()))
        for c in CONDS:
            for lab in labels:
                row = dict(model=name, label=MODEL_LABELS[name], odor=lab, kind=('mixture' if lab in MIXTURES else ('low_conc' if lab in ORN_SOURCE else 'apl16')),
                    cond=c, frac_det=float(fr[c][lab]), substitution=subs.get(lab, 'APL16 input'))
                if lab in ORN_SOURCE or lab in MIXTURES:
                    # breadth over the APL16 odors of the responders to this odor (deterministic)
                    resp = det[c][0].astype(float)
                    col = resp.iloc[:, labels.index(lab)].to_numpy() > 0
                    row['responder_breadth_apl16'] = float(resp16[c].to_numpy()[col].sum(axis=1).mean()) if col.any() else np.nan
                    row['n_responders_det'] = int(col.sum())
                if nf is not None:
                    row.update(frac_count1=float(nf[c].loc[lab, 'count1']), frac_count1_sd=float(nf[c].loc[lab, 'count1_sd']),
                        frac_trial=float(nf[c].loc[lab, 'trial']), sigma=noise['sigma'], fixed_thr_ratio=noise['fixed_thr_ratio'])
                frac_rows.append(row)
        # mixtures: the model's mixture response against the union of its components' responders
        for mix, comps in MIXTURES.items():
            for c in CONDS:
                resp = det[c][0].astype(float).to_numpy() > 0
                i_mix = labels.index(mix); i_a, i_b = labels.index(comps[0]), labels.index(comps[1])
                row = dict(model=name, label=MODEL_LABELS[name], mixture=mix, cond=c, frac_mix_det=float(resp[:, i_mix].mean()),
                    frac_union_det=float((resp[:, i_a] | resp[:, i_b]).mean()), frac_a_det=float(resp[:, i_a].mean()), frac_b_det=float(resp[:, i_b].mean()))
                if nf is not None:
                    spk = trials[f'{c}_spk'].astype(float).mean(axis=1) >= 1   # rec x KCs x odors
                    row.update(frac_mix_count1=float(spk[:, :, i_mix].mean()), frac_union_count1=float((spk[:, :, i_a] | spk[:, :, i_b]).mean()))
                mix_rows.append(row)
        pd.DataFrame(frac_rows).to_csv(res_dir / 'fractions.csv', index=False)
        pd.DataFrame(mix_rows).to_csv(res_dir / 'mixtures.csv', index=False)
        (res_dir / 'checks.json').write_text(json.dumps(checks, indent=1, default=str))
        print(f"{name}: APL-on rate {fr['on'].iloc[:16].mean():.4f} off {fr['acute_off'].iloc[:16].mean():.4f}; low-conc fractions on/off: " +
            ', '.join(f"{o} {fr['on'][o]:.3f}/{fr['acute_off'][o]:.3f}" for o in ORN_SOURCE) +
            (f"; noisy (count1) on: " + ', '.join(f"{o} {nf['on'].loc[o, 'count1']:.3f}" for o in ORN_SOURCE) if nf is not None else '') +
            f"; checks {chk}; {time.time() - t1:.0f}s", flush=True)
        del m
    data.to_csv(res_dir / 'data_fractions.csv', index=False)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E49_low_conc_kc.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), models=args.models,
        model_sources=MODEL_SOURCES, noise_sources=NOISE_SOURCE, n_recordings=args.n_recordings, n_trials=args.n_trials, seed=args.seed,
        no_noise=args.no_noise, orn_input=orn_info, substitutions=subs, data_dir=str(yang_dir()),
        approx_conc_note=H.APPROX_CONC_NOTE, seconds=time.time() - t0, model=model_prov)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    summarize(res_dir)
    print('done ->', res_dir)


def summarize(res_dir: Path) -> None:
    """comparison_table.md and the figure from fractions.csv, mixtures.csv and data_fractions.csv."""
    fr = pd.read_csv(res_dir / 'fractions.csv'); mx = pd.read_csv(res_dir / 'mixtures.csv')
    data = pd.read_csv(res_dir / 'data_fractions.csv') if (res_dir / 'data_fractions.csv').exists() else data_fractions()
    models = [m for m in MODEL_SOURCES if (fr.model == m).any()]
    noisy = 'frac_count1' in fr.columns and fr.frac_count1.notna().any()

    def dcell(odor, group, panel=None):
        d = data[(data.odor == odor) & (data.group == group)]
        if panel is not None:
            d = d[d.panel == panel]
        if d.empty:
            return '-'
        return ' / '.join(f'{r.frac:.3f} ({r.panel[:4]}, n {int(r.n_recordings)})' for r in d.itertuples())

    lines = ["**Response fractions at the low concentrations: the data (trial-mean > 1; control flies / APL output blocked (TNT)) "
        "against each model's deterministic fraction (APL on / acute APL off)" + (" and its fraction through the matched threshold "
        "jitter (trial-mean spike count >= 1; APL on / off)" if noisy else '') + ". ORN input: exact or substituted as noted.**", '',
        '| odor | ORN input | KC data, control flies | KC data, APL output blocked | ' + ' | '.join(
            (f'{MODEL_LABELS[m]}<br>det on / off' + ('<br>noisy on / off' if noisy else '')) for m in models) + ' |',
        '|---|---|---|---|' + '---|' * len(models)]
    odors = []
    for o in list(ORN_SOURCE) + list(MIXTURES):
        odors.append(o); odors += [a for a, base in ALIAS.items() if base == o]
    for odor in odors:
        base = ALIAS.get(odor, odor)
        sub = fr[fr.odor == base]
        cells = []
        for m in models:
            s = sub[sub.model == m].set_index('cond')
            c = f"{s.loc['on', 'frac_det']:.3f} / {s.loc['acute_off', 'frac_det']:.3f}"
            if noisy:
                c += f"<br>{s.loc['on', 'frac_count1']:.3f} / {s.loc['acute_off', 'frac_count1']:.3f}"
            cells.append(c)
        subst = sub.substitution.iloc[0] if not sub.empty else '-'
        if odor in ALIAS:
            subst = f'the same input as {base} (model values repeated)'
        lines.append(f'| {odor} | {subst} | {dcell(odor, "control")} | {dcell(odor, "apl_output_blocked")} | ' + ' | '.join(cells) + ' |')
    # APL-off / APL-on ratios
    lines += ['', '**APL-off / APL-on response ratio per odor (data: APL output blocked / control flies, diag_mix; model: acute APL off / APL on, deterministic' + (' and noisy' if noisy else '') + ')**', '',
        '| odor | KC data | ' + ' | '.join(MODEL_LABELS[m] for m in models) + ' |', '|---|---|' + '---|' * len(models)]
    for odor in list(ORN_SOURCE):
        dc = data[(data.odor == odor) & (data.group == 'control') & (data.panel == 'diag_mix')]
        dt = data[(data.odor == odor) & (data.group == 'apl_output_blocked') & (data.panel == 'diag_mix')]
        d = f'{float(dt.frac.iloc[0]) / float(dc.frac.iloc[0]):.2f}' if len(dc) == 1 and len(dt) == 1 and float(dc.frac.iloc[0]) > 0 else '-'
        cells = []
        for m in models:
            s = fr[(fr.odor == odor) & (fr.model == m)].set_index('cond')
            on, off = s.loc['on', 'frac_det'], s.loc['acute_off', 'frac_det']
            c = f'{off / on:.2f}' if on > 0 else ('inf' if off > 0 else '-')
            if noisy:
                on, off = s.loc['on', 'frac_count1'], s.loc['acute_off', 'frac_count1']
                c += f" / {off / on:.2f}" if on > 0 else ' / -'
            cells.append(c)
        lines.append(f'| {odor} | {d} | ' + ' | '.join(cells) + ' |')
    # mixtures
    lines += ['', '**Mixtures: the measured mixture against the union of the components (data: Yang\'s union prediction; model: KCs responding to either component), control flies / APL on and APL output blocked / APL off**', '',
        '| mixture | KC data: mixture / union, control | the same, APL output blocked | ' + ' | '.join(f'{MODEL_LABELS[m]}<br>mixture / union, APL on<br>mixture / union, APL off' for m in models) + ' |',
        '|---|---|---|' + '---|' * len(models)]
    mixes = []
    for mi in MIXTURES:
        mixes.append(mi); mixes += [a for a, base in ALIAS.items() if base == mi]
    for mix in mixes:
        def dm(group):
            d = data[(data.odor == mix) & (data.group == group) & (data.panel == 'diag_mix')]
            return f'{float(d.frac.iloc[0]):.3f} / {float(d.union.iloc[0]):.3f}' if len(d) == 1 else '-'
        cells = []
        for m in models:
            s = mx[(mx.mixture == ALIAS.get(mix, mix)) & (mx.model == m)].set_index('cond')
            cells.append(f"{s.loc['on', 'frac_mix_det']:.3f} / {s.loc['on', 'frac_union_det']:.3f}<br>{s.loc['acute_off', 'frac_mix_det']:.3f} / {s.loc['acute_off', 'frac_union_det']:.3f}")
        label = mix + (' (model: the same input as ' + ALIAS[mix] + ')' if mix in ALIAS else '')
        lines.append(f'| {label} | {dm("control")} | {dm("apl_output_blocked")} | ' + ' | '.join(cells) + ' |')
    (res_dir / 'comparison_table.md').write_text('\n'.join(lines) + '\n')
    make_figure(fr, mx, data, models, noisy, res_dir)
    print('\n'.join(lines))


def make_figure(fr: pd.DataFrame, mx: pd.DataFrame, data: pd.DataFrame, models: List[str], noisy: bool, res_dir: Path) -> None:
    plt = H._plt()
    odors = list(ORN_SOURCE)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    x = np.arange(len(odors)); w = 0.8 / (len(models) + 1)
    for ax, cond, group, title in ((axes[0], 'on', 'control', 'APL on vs control flies'), (axes[1], 'acute_off', 'apl_output_blocked', 'APL off vs APL output blocked (TNT)')):
        d = [data[(data.odor == o) & (data.group == group)].sort_values('n_recordings', ascending=False) for o in odors]
        dv = [float(di.frac.iloc[0]) if len(di) else np.nan for di in d]
        de = [float(di['sem'].iloc[0]) if len(di) else 0.0 for di in d]   # ['sem']: DataFrame.sem is a method
        ax.bar(x - 0.4 + w / 2, dv, w, yerr=de, color='k', alpha=0.75, label='KC data')
        for i, m in enumerate(models):
            s = fr[(fr.model == m) & (fr.cond == cond)].set_index('odor').reindex(odors)
            ax.bar(x - 0.4 + w * (i + 1.5), s.frac_det, w, label=MODEL_LABELS[m] + ' (deterministic)')
            if noisy:
                ax.scatter(x - 0.4 + w * (i + 1.5), s.frac_count1, marker='_', s=120, color='k', zorder=3, label='through the matched noise' if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(odors, rotation=35, ha='right', fontsize=8); ax.set_title(title, fontsize=10)
        ax.set_ylabel('fraction of KCs responding'); ax.grid(axis='y', alpha=0.25)
    ax = axes[2]
    mixes = list(MIXTURES); xm = np.arange(len(mixes))
    dm = [data[(data.odor == mi) & (data.group == 'control') & (data.panel == 'diag_mix')] for mi in mixes]
    ax.bar(xm - 0.4 + w / 2, [float(d.frac.iloc[0]) if len(d) else np.nan for d in dm], w, color='k', alpha=0.75, label='KC data, mixture')
    ax.scatter(xm - 0.4 + w / 2, [float(d.union.iloc[0]) if len(d) else np.nan for d in dm], marker='v', color='k', zorder=3, label='union of components')
    for i, m in enumerate(models):
        s = mx[(mx.model == m) & (mx.cond == 'on')].set_index('mixture').reindex(mixes)
        ax.bar(xm - 0.4 + w * (i + 1.5), s.frac_mix_det, w)
        ax.scatter(xm - 0.4 + w * (i + 1.5), s.frac_union_det, marker='v', color='k', zorder=3)
    ax.set_xticks(xm); ax.set_xticklabels(mixes, rotation=35, ha='right', fontsize=8)
    ax.set_title('mixtures (control flies / APL on): bar = mixture, triangle = union of components', fontsize=10); ax.grid(axis='y', alpha=0.25)
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle("Yang's low-concentration odors: response fractions in the data and in the APL16-fitted models at fixed parameters", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(res_dir / f'{EXPERIMENT}.png', dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
