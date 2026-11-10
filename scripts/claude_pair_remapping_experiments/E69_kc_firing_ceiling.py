#!/usr/bin/env python3
"""E69: a firing ceiling for the KCs. A refractory period and spike-frequency adaptation in the
model, read on the APL effect under both of its definitions and on the APL-off recruitment.

olfsysm's KCs had no refractory period and no adaptation: after a spike the potential is reset to 0
and integration resumes at the next step, so a strongly driven KC fires up to 100+ spikes to a 0.5 s
odor (the reference model's largest APL-off count is 117). Since olfsysm commit "Add a KC refractory
period and spike-frequency adaptation" three parameters cap that (`fit_mb_model` kwargs
`kc_refractory_period`, `kc_adapt_inc`, `kc_adapt_tau`; olfsysm `mp.kc.refractory_period`,
`adapt_inc`, `adapt_tau`, all off by default): after a spike the KC's potential stays at 0 and no
spike can occur for the refractory period; each spike raises a per-KC adaptation variable a by
`adapt_inc`, which decays with time constant `adapt_tau`, and the KC spikes when its potential
exceeds thr * (1 + a) (thr its threshold), so `adapt_inc` is in units of the KC's own threshold.

Part A (this script), at fixed scales. Bases (BASES): (1) the reference model (eight glomeruli at
1.5, 40 Hz per-claw threshold, 6.8 %; E44's optimum); (2) the APL-effect fits' (E67) best fit under
the calibrated readout and (3) on raw counts, among the fits inside the realism bands when part A
started (`resolve_bases`, recorded in bases.json). Each base is first re-run with the new olfsysm build
and must reproduce its saved spike counts exactly (`--stage check`). A pilot of APL-off runs of base 1
with the 3 ms refractory period at the 1 s time constant sets three adaptation strengths so that the
largest APL-off spike count is near 40, 20 and 10 (`--stage pilot`). Settings (`settings`): base 1,
the 3 ms refractory period alone, then with it the three strengths x the time constants 50 ms, 200 ms
and 1 s; bases 2 and 3, the 3 ms refractory period with the three strengths at 1 s and the middle one
at 200 ms. At every setting (`--stage setting`, one process each, since a C++ exception inside an
OpenMP region kills the process without a traceback) both response rates are held at the base's by
two bisections (log scale): the somatic threshold factor on the APL-off response rate, then one
common factor on the APL->KC and APL->bouton scales on the APL-on response rate; the KC-KC scale and
everything else stay as fitted, and the re-tuned absolute values are recorded.

Readout per variant (the base and each setting), beside the KC data: the harness stats columns (as
the per-odor APL profile experiment's, E66 `score_variant`: the two ρ's on odor-odor distances, the
block, the natural-source alignment APL on / off, the mean distances, response rates and response
rate ρ in both conditions, the APL-effect loss); the APL-effect loss with its mean, per-odor and
pair-specific parts and ρ on the per-pair APL effect on raw counts, under the calibrated readout (K
re-calibrated on the variant's APL-on counts, recorded) and binarized against the binarized data; the
block under the readout; the spike-count distribution in both conditions (largest count, the
responders' 99th percentile and median, the share of responding entries at 20 spikes or more and of
the squared response vectors they carry, the KCs firing 20 or more to some odor, the responders-only
participation ratio, top-1 % share and coefficient of variation, raw and under the readout); E66's
five worst pairs of the reference model with the number of KCs carrying half of each pair's model
effect; the amplitude statistics through the low end of the readout experiment's (E68) conversion to
z with the data's measurement noise (one-spike floor with noise, 'floor_noise', and the plain
readout with noise, 'cal_noise': entries above 4 z, the responders' mean amplitude, the 0.5-1.5 z
band, both conditions); the odors-per-KC distribution in both conditions with its distance from the
flies'; the realism flag (apl_effect_scores.realism_flag, all four bands).

Outputs: results/E69_kc_firing_ceiling/ (bases.json, base_checks.json, pilot.json, settings_log.csv,
stats_variants.csv, model_pair_distances_long.csv, ceiling_readouts.csv, worst_pairs.csv,
partA_variant_stats.csv, parts_A.csv, shares_A.csv, comparison_table.md, figures, provenance.json);
counts and responses in outputs/E69_kc_firing_ceiling/<tag>_<on|acute_off>_<spike_counts|responses>.parquet.

Usage (from this folder; one process at a time):
    OMP_NUM_THREADS=16 nice -n 10 ../../venv/bin/python E69_kc_firing_ceiling.py --stage run
    (runs every missing check, the pilot and every missing setting, each in its own process, then
    --stage summarize)
    ../../venv/bin/python E69_kc_firing_ceiling.py --stage summarize
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E69_kc_firing_ceiling'
RESULTS = H.shared_dir() / 'results'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
E67_FOLDER = 'E67_apl_effect_fits'
ODORS = list(H.APL16_ODORS)
V = H.PRIMARY_MODEL_RDM
RESP_MIN = 2.0
CONDS = ('on', 'acute_off')

REFRACTORY = 3e-3
TAUS = (0.05, 0.2, 1.0)
PILOT_TAU = 1.0
PILOT_TARGETS = (40, 20, 10)
PILOT_BRACKET = (1e-4, 10.0)
PILOT_ITERS = 24
THR_BRACKET, THR_STEP = (0.2, 1.6), 1.1
APL_BRACKET, APL_STEP = (1 / 32, 32.0), 1.4
BISECT_ITERS = 20
# the APL-effect fits' cap on the APL->bouton scale, as a multiple of the bouton base's (E3)
APLPN_CAP = 3.0
STRONG_COUNT = 20

BASES = {
    'ref40': dict(label='the reference model (eight at 1.5, 40 Hz, 6.8 %)', exp='E44_operating_point',
        tag='thr40_gain1.5_rate0.068_k2_cap3'),
    'e67cal': dict(label='the best APL-effect fit under the calibrated readout', exp=E67_FOLDER, definition='cal'),
    'e67raw': dict(label='the best APL-effect fit on raw counts', exp=E67_FOLDER, definition='raw'),
}
# E68's part B re-simulated the two E67 fits (spike counts checked against E67's saved rates and
# distances); its saved counts are the exact reference for this build
E68_E67_NPZ = {'cal': 'e67_cal_vm.npz', 'raw': 'e67_raw_vm.npz'}
GENO_LABEL = {'on': 'control flies', 'acute_off': 'APL output blocked (TNT)'}


def res_dir() -> Path:
    return H.results_dir(EXPERIMENT)


def out_dir() -> Path:
    return H.outputs_dir(EXPERIMENT)


def base_dir() -> Path:
    d = out_dir() / 'bases'
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str) + '\n')


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def code_state() -> dict:
    """The al_analysis commit and whether its tracked files are modified, and the olfsysm build's
    commit and clean flag, for each process's record."""
    import olfsysm
    git = lambda *a: subprocess.run(['git', *a], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    bi = olfsysm.build_info() if hasattr(olfsysm, 'build_info') else {}
    return dict(al_analysis_commit=git('rev-parse', 'HEAD'),
        al_analysis_dirty=bool(git('status', '--porcelain', '--untracked-files=no')),
        olfsysm_commit=bi.get('git_hash'), olfsysm_dirty=bi.get('git_dirty'))


# ---------------------------------------------------------------------------
# bases
def resolve_bases() -> Dict[str, dict]:
    """BASES with the E67 tags filled in: per definition, the fit with the lowest APL-effect loss
    under it (fits.csv `fit_apl_cal_rmse` / `fit_apl_raw_rmse`) among the fits inside the realism
    bands on the response rates, their ratio, the APL-on response rate ρ and the odors per KC with
    APL off (`fit_opk_off_dist`; the APL-on distance is not in fits.csv and is checked at
    `--stage check`). Recorded in bases.json the first time and read from there afterwards, so that
    every setting uses the same bases."""
    p = res_dir() / 'bases.json'
    saved = read_json(p)
    if saved is not None:
        return saved
    # (the APL-effect fits run on viki, whose E67 archive and fits.csv grow during part A: the bases are those of
    # the first call)
    from apl_effect_scores import realism_flag
    f = pd.read_csv(RESULTS / E67_FOLDER / 'fits.csv')
    f['flag'] = [realism_flag(r.rate_on, r.rate_off, r.spearman_rate_on_vs_data, np.nan, r.fit_opk_off_dist)
        for r in f.itertuples()]
    ok = f[f.flag == '']
    out = {}
    for key, b in BASES.items():
        b = dict(b)
        if b['exp'] == E67_FOLDER:
            col = f"fit_apl_{b['definition']}_rmse"
            r = ok.loc[ok[col].idxmin()]
            b.update(tag=str(r.tag), fit_loss=float(r[col]), fit_loss_other=float(r[
                'fit_apl_raw_rmse' if b['definition'] == 'cal' else 'fit_apl_cal_rmse']),
                candidates={str(t): float(v) for t, v in zip(ok.tag, ok[col])},
                excluded={str(t): fl for t, fl in zip(f.tag, f.flag) if fl})
        out[key] = b
    write_json(p, dict(out, _resolved=dict(time=now(), host=socket.gethostname())))
    return read_json(p)


def bases() -> Dict[str, dict]:
    return {k: v for k, v in resolve_bases().items() if not k.startswith('_')}


def copy_base_caches() -> None:
    """The APL-off runs behind the gained bases' breadth, as E68's part B cached them (bouton_base
    re-simulates any that are missing, at the base's tuned parameters)."""
    for src in (H.outputs_dir('E68_readout_low_end') / 'partB', H.outputs_dir('E66_per_odor_apl_profile') / 'bases'):
        for p in src.glob('*_base_acute_off_*.parquet'):
            dst = base_dir() / p.name
            if not dst.exists():
                dst.write_bytes(p.read_bytes())


class Model:
    """One base at its fitted parameters, as a fixed-scale simulator (`make_fixed_param_kc_simulator`
    with the KC refractory / adaptation kwargs `kc`, set in its initial call, so every simulation
    has them); `sim(params)` returns (responses, spike counts) with columns in ODORS order."""

    def __init__(self, key: str, kc: Optional[dict] = None):
        import E43_model_trial_noise as E43
        from al_analysis.mb_model import make_fixed_param_kc_simulator
        from E10_joint_fit import bouton_base
        b = bases()[key]
        self.key, self.base, self.kc = key, b, dict(kc or {})
        copy_base_caches()
        name = E43.model_from(f"{b['exp']}:{b['tag']}")
        self.lab_orn = H.apl16_orn_deltas()
        fm = E43.fit_model(name, self.lab_orn, base_dir())
        if fm['info'].get('wiring') or fm['info'].get('orn_input'):
            raise NotImplementedError(f"{key}: re-drawn wirings / other ORN inputs are not used as bases here")
        self.orn, self.p, self.kws, self.info = fm['orn'], dict(fm['p_on']), fm['kws'], fm['info']
        self.fit_rates, self.saved = fm['rates'], fm['saved']
        gain = self.info.get('gain', 1.0)
        _, _, self.W, _ = bouton_base(self.orn, base_dir(), claw_thr=self.info.get('claw_thr'),
            gain=None if gain == 1.0 else gain, gain_gloms=self.info.get('gain_glomeruli') or None)
        self.fit = E43.fit_row(b['exp'], b['tag'])
        t0 = time.time()
        self._sim, _ = make_fixed_param_kc_simulator(self.orn, initial_params=self.p, **self.kws, **self.kc)
        self.build_seconds = time.time() - t0
        self.n_sims = 0

    def sim(self, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """(responses, spike counts), columns (the simulator's labels, e.g. '2h @ -3', as every
        archived variant has them) in ODORS order."""
        self.n_sims += 1
        r, s = self._sim(params)
        order = sorted(r.columns, key=lambda c: ODORS.index(H.odor_abbrev(c)))
        return r[order], s[order]

    def off_params(self, p: Optional[dict] = None) -> dict:
        return dict(p or self.p, wAPLKC_scale=0.0, wAPLPN_scale=0.0)


def rate(responses: pd.DataFrame) -> float:
    return float(responses.to_numpy(dtype=float).mean())


def n_responses(responses: pd.DataFrame) -> int:
    return int(responses.to_numpy(dtype=float).sum())


# ---------------------------------------------------------------------------
# scoring
_CTX: Dict[str, object] = {}


def ctx() -> dict:
    if not _CTX:
        import E66_per_odor_apl_profile as E66
        _CTX.update(E66.stats_context())
        _CTX['pr_target'] = H.data_target('ctrl_participation_ratio_resp')
        _CTX['effect'] = H.yang_apl_effect()
        _CTX['effect_binary'] = H.yang_apl_effect(binarize=1.0)
    return _CTX


def worst_pairs_e66() -> List[Tuple[str, str]]:
    """The five pairs with the largest pair-specific error of the reference model on raw counts (the
    per-odor APL profile experiment, E66)."""
    w = pd.read_csv(RESULTS / 'E66_per_odor_apl_profile' / 'worst_pairs.csv')
    w = w[(w.tag == BASES['ref40']['tag']) & (w.readout == 'raw')].sort_values('rank')
    return [(str(a), str(b)) for a, b in zip(w.odor_a, w.odor_b)][:5]


def count_stats(m: pd.DataFrame, K: float, short: str) -> dict:
    """The spike-count distribution of one condition (KCs x odors counts)."""
    x = m.to_numpy(dtype=float)
    resp = x >= RESP_MIN
    rv = x[resp]
    sq = x ** 2
    r = m.where(m >= RESP_MIN, 0.0)
    a_raw = H.amplitude_stats(r).mean(numeric_only=True)
    a_cal = H.amplitude_stats(H.readout(r, 'saturating', K)).mean(numeric_only=True)
    return {f'max_count_{short}': float(x.max()), f'p99_resp_count_{short}': float(np.percentile(rv, 99)) if len(rv) else np.nan,
        f'median_resp_count_{short}': float(np.median(rv)) if len(rv) else np.nan,
        f'share_resp_ge{STRONG_COUNT}_{short}': float((rv >= STRONG_COUNT).mean()) if len(rv) else np.nan,
        f'sq_share_ge{STRONG_COUNT}_{short}': float(sq[x >= STRONG_COUNT].sum() / sq.sum()) if sq.sum() > 0 else np.nan,
        f'n_kcs_ge{STRONG_COUNT}_{short}': int((x >= STRONG_COUNT).any(axis=1).sum()), f'n_kcs_{short}': int(x.shape[0]),
        f'pr_resp_raw_{short}': float(a_raw['participation_ratio']), f'top1_resp_raw_{short}': float(a_raw['top1_share']),
        f'cv_resp_raw_{short}': H.responder_cv(rv),
        f'pr_resp_cal_{short}': float(a_cal['participation_ratio']), f'top1_resp_cal_{short}': float(a_cal['top1_share']),
        f'cv_resp_cal_{short}': H.responder_cv(H.readout(m, 'saturating', K).to_numpy()[resp])}


def amplitude_z_stats(on: pd.DataFrame, off: pd.DataFrame, K: float) -> dict:
    """The amplitude statistics through E68's conversion to z with the data's measurement noise (its
    pseudo-recordings, seeds and calibration targets): 'floor_noise' (K, the one-spike floor b and the
    conversion c calibrated jointly on the APL-on counts) and 'cal_noise' (K as given, c on the APL-on
    response rate)."""
    import E68_readout_low_end as E68
    s, floor_z = E68.refs()['noise_sd'], E68.refs()['floor_z']
    arr = {c: x.to_numpy(dtype=float) for c, x in (('on', on), ('acute_off', off))}
    out = {}
    for name, how in (('floor_noise', 'joint_floor'), ('cal_noise', 'e65')):
        cal = E68.calibrate(arr['on'], how, K)
        Kc, b, c = cal['K'], cal['floor'], cal['c']
        out.update({f'z_{name}_K': Kc, f'z_{name}_floor': b, f'z_{name}_c': c,
            f'z_{name}_at_bound': bool(cal.get('c_at_bound') or cal.get('K_at_bound') or cal.get('floor_at_bound'))})
        for cond, x in arr.items():
            z = c * E68.saturating(x, Kc, b)
            trials = H.sample_pseudo_recordings(z, s, n_recordings=E68.N_RECORDINGS, n_trials=E68.N_TRIALS,
                seed=E68.SEED_ON if cond == 'on' else E68.SEED_OFF, return_trials=True)
            pooled = np.concatenate([t.ravel() for t in trials.mean(axis=1)])
            st = E68.amplitude_summary(pooled, floor_z)
            out.update({f'z_{name}_{k}_{cond}': st[k] for k in ('share_above_4', 'resp_amp_mean', 'share_05_15', 'rate_z',
                'share_025_1', 'share_below_m025')})
    return out


def opk_stats(r_on: pd.DataFrame, r_off: pd.DataFrame) -> dict:
    from apl_effect_scores import OPK_BINS, odors_per_kc, opk_data
    out = {}
    for cond, r in (('on', r_on), ('off', r_off)):
        d = odors_per_kc(r)
        out.update({f'opk_{cond}_{b}': float(v) for b, v in zip(OPK_BINS, d)})
        out[f'opk_{cond}_tv'] = float(0.5 * np.abs(d - opk_data(cond)).sum())
    return out


def worst_pair_rows(tag: str, on: pd.DataFrame, off: pd.DataFrame, K: float) -> List[dict]:
    """E66's five worst pairs under raw counts and the calibrated readout: the model's and the
    data's per-pair effect, the pair-specific part, and how many KCs carry half of the model's
    per-pair effect (E66 `kc_contributions`)."""
    import E66_per_odor_apl_profile as E66
    rows = []
    for readout, kind, param in (('raw', 'raw', None), ('cal', 'saturating', K)):
        d_on, d_off = H.readout_pair_distances(on, kind, param), H.readout_pair_distances(off, kind, param)
        parts = H.apl_effect_pair_parts(d_on, d_off, ctx()['effect'])
        for rank, (a, b) in enumerate(worst_pairs_e66(), 1):
            key = (a, b) if (a, b) in parts.index else (b, a)
            r = parts.loc[key]
            kc = E66.kc_contributions(on, off, kind, param, a, b)
            absc = kc.contribution.abs().sort_values(ascending=False)
            n_half = int(np.searchsorted(np.cumsum(absc.to_numpy()), 0.5 * absc.sum()) + 1) if absc.sum() > 0 else 0
            rows.append(dict(tag=tag, readout=readout, rank=rank, odor_a=a, odor_b=b, model_effect=float(r.model),
                data_effect=float(r.data), error=float(r.error), per_odor=float(r.per_odor), pair_specific=float(r.pair_specific),
                n_kcs_half=n_half))
    return rows


def score(tag: str, meta: dict, model: Model, on_out, off_out) -> dict:
    """Every readout of one variant: E66's `score_variant` (the harness stats columns, the loss
    decomposition under each readout, the counts saved) plus this experiment's readouts, appended to
    ceiling_readouts.csv and worst_pairs.csv."""
    import E66_per_odor_apl_profile as E66
    from E10_joint_fit import apl_effect_terms
    from apl_effect_scores import realism_flag
    c = ctx()
    on_s, off_s = on_out[1].astype(float), off_out[1].astype(float)
    K = H.calibrate_readout(on_s, 'saturating', c['pr_target'], stat='participation_ratio', responders_min=RESP_MIN)
    row = E66.score_variant(c, tag, meta, on_out, off_out, model.W, model.lab_orn, K, 'A', res_dir(), out_dir())
    ex = dict(tag=tag, cal_K=K)
    ae = apl_effect_terms(H.pair_distances(on_s, 'cosine'), H.pair_distances(off_s, 'cosine'), on_s, off_s, c['effect'],
        c['pr_target'])
    ex.update(ae)
    dec_b, _ = H.apl_effect_decomposition(H.readout_pair_distances(on_s, 'binary', RESP_MIN),
        H.readout_pair_distances(off_s, 'binary', RESP_MIN), c['effect_binary'])
    ex.update({f'apl_bin_{k[len("apl_effect_"):]}': v for k, v in dec_b.items() if k.startswith('apl_effect_')})
    for cond, m in (('on', on_s), ('off', off_s)):
        ex.update(count_stats(m, K, cond))
    ex.update(amplitude_z_stats(on_s, off_s, K))
    ex.update(opk_stats(on_out[0], off_out[0]))
    ex['realism_flag'] = realism_flag(row['rate_on'], row['rate_off'], row.get('spearman_rate_on_vs_data'),
        ex['opk_on_tv'], ex['opk_off_tv'])
    for name, df in (('ceiling_readouts.csv', pd.DataFrame([ex])), ('worst_pairs.csv', pd.DataFrame(worst_pair_rows(tag, on_s, off_s, K)))):
        p = res_dir() / name
        df.to_csv(p, mode='a', header=not p.exists(), index=False)
    print(f"{tag}: largest count on {ex['max_count_on']:.0f} / off {ex['max_count_off']:.0f}; loss raw {ae['apl_raw_rmse']:.3f} "
        f"(mean {ae['apl_raw_mean']:+.3f}), calibrated {ae['apl_cal_rmse']:.3f} (mean {ae['apl_cal_mean']:+.3f}, K {K:.2f}); "
        f"APL off above 4 z {ex['z_floor_noise_share_above_4_acute_off']:.4f}, responders' mean "
        f"{ex['z_floor_noise_resp_amp_mean_acute_off']:.2f} z; odors per KC {ex['opk_on_tv']:.3f} / {ex['opk_off_tv']:.3f}; "
        f"flag: {ex['realism_flag'] or '-'}", flush=True)
    return {**row, **ex}


def done_tags() -> set:
    p = res_dir() / 'ceiling_readouts.csv'
    return set(pd.read_csv(p)['tag'].astype(str)) if p.exists() else set()


def base_meta(model: Model) -> dict:
    """Identifying and parameter columns shared by a base's variants."""
    b, info, fit = model.base, model.info, model.fit
    return dict(base=model.key, base_label=b['label'], ref_experiment=b['exp'], ref_fit=b['tag'],
        claw_thr=info.get('claw_thr'), gain=info.get('gain'), gain_glomeruli=','.join(info.get('gain_glomeruli') or []),
        k=info.get('k'), kc_kc_tau=info.get('kc_kc_tau'), aplpn_factor_vs_e3_base=float(fit.get('aplpn_factor_vs_e3', np.nan)),
        apl_factor_vs_e3_base=float(fit.get('apl_factor_vs_e3', np.nan)), n_kcs=None, n_glomeruli=int(model.W.shape[1]))


def params_meta(p: dict) -> dict:
    return dict(fixed_thr=p['fixed_thr'], wAPLKC_scale=p['wAPLKC_scale'], wKCAPL_scale=p['wKCAPL_scale'],
        wAPLPN_scale=p['wAPLPN_scale'], wPNAPL_scale=p['wPNAPL_scale'], wKCKC_scale=p['wKCKC_scale'],
        kc_kc_inh=p['wKCKC_scale'])


# ---------------------------------------------------------------------------
# stages
def stage_check(key: str) -> None:
    """Re-runs base `key` with the new olfsysm build (no refractory period, no adaptation) and checks
    it reproduces the saved spike counts exactly; scores it as variant <key>_base."""
    t0 = time.time()
    m = Model(key)
    on, off = m.sim(m.p), m.sim(m.off_params())
    checks = {}
    b = m.base
    arch = ARCHIVE / b['exp']
    for cond, out in (('on', on), ('acute_off', off)):
        p = arch / f"{b['tag']}_{cond}_spike_counts.parquet"
        if p.exists():
            a = H.abbrev_columns(pd.read_parquet(p).astype(float))[ODORS]
            checks[f'{cond}_archive_index_equal'] = bool(a.index.equals(out[1].index))
            checks[f'{cond}_archive_mismatch'] = int((a.to_numpy() != out[1].to_numpy(dtype=float)).sum())
    # (the rates of the fit's row in its fits.csv)
    checks['rate_on_vs_fit'] = rate(on[0]) - float(m.fit_rates[0])
    checks['rate_off_vs_fit'] = rate(off[0]) - float(m.fit_rates[1])
    if b['exp'] == E67_FOLDER:
        npz = ARCHIVE / 'E68_readout_low_end' / E68_E67_NPZ[b['definition']]
        z = np.load(npz)
        info = json.loads(str(z['info']))
        if info['tag'] == b['tag']:
            for cond, out in (('on', on), ('acute_off', off)):
                checks[f'{cond}_e68_mismatch'] = int((z[f'{cond}_spk'] != out[1].to_numpy(dtype=float)).sum())
        else:
            # (E68's part B simulated another fit: only the archive's counts, the rates and the distances are compared)
            checks['e68_npz_tag'] = info['tag']
        d = pd.read_csv(RESULTS / E67_FOLDER / 'model_pair_distances_long.csv')
        d = d[(d.tag == b['tag']) & (d['variant'] == V)] if 'variant' in d.columns else d[d.tag == b['tag']]
        for cond, out in (('on', on), ('acute_off', off)):
            dd = d[d.condition == cond]
            if len(dd):
                mine = H.pair_distances(out[1].astype(float), 'cosine')
                theirs = H.canonical_pairs(dd.set_index(['odor_a', 'odor_b'])['distance'])
                x, y = H.align_pairs(mine, theirs)
                checks[f'{cond}_distance_max_abs_diff_vs_fit'] = float(np.abs(x - y).max())
                checks[f'{cond}_distance_n_pairs'] = int(len(x))
    n_compared = sum(1 for k in checks if k.endswith('mismatch'))
    bad = [k for k, v in checks.items() if (k.endswith('mismatch') and v != 0) or (k.endswith('index_equal') and not v)
        or (k.startswith('rate_') and abs(v) > 1e-9) or (k.endswith('_vs_fit') and 'distance' in k and v > 1e-9)]
    if n_compared == 0 or bad:
        raise RuntimeError(f'{key}: the re-simulated base does not reproduce its saved outputs: {checks}')
    meta = dict(variant='base', tag=f'{key}_base', part='A', setting='base', **base_meta(m), kc_refractory_period=0.0,
        kc_adapt_inc=0.0, kc_adapt_tau=np.nan, adapt_strength='', thr_factor=1.0, apl_factor=1.0,
        target_rate_on=rate(on[0]), target_rate_off=rate(off[0]), rate_on=rate(on[0]), rate_off=rate(off[0]),
        thr_at_bound=False, apl_at_bound=False, aplpn_factor_vs_e3=float(m.fit.get('aplpn_factor_vs_e3', np.nan)),
        aplpn_over_cap=False, n_sims=m.n_sims, seconds=time.time() - t0, **params_meta(m.p))
    meta['n_kcs'] = int(on[0].shape[0])
    meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
    if f'{key}_base' not in done_tags():
        score(f'{key}_base', meta, m, on, off)
    p = res_dir() / 'base_checks.json'
    allc = read_json(p, {})
    allc[key] = dict(checks=checks, base=b, model_info=m.info, params=m.p, rate_on=rate(on[0]), rate_off=rate(off[0]),
        kws_scalar={k: v for k, v in m.kws.items() if not k.startswith('_')},
        n_responses_on=n_responses(on[0]), n_responses_off=n_responses(off[0]), size=int(on[0].size),
        max_count_on=float(on[1].to_numpy().max()), max_count_off=float(off[1].to_numpy().max()),
        build_seconds=m.build_seconds, time=now(), host=socket.gethostname(), **code_state())
    write_json(p, allc)
    print(f'{key}: checks passed {checks}', flush=True)


def stage_pilot() -> None:
    """APL-off runs of base 1 at its fitted parameters with the 3 ms refractory period and the 1 s
    time constant: the adaptation strength at which the largest APL-off spike count is each of
    PILOT_TARGETS (bisection on log strength; the count does not rise with the strength)."""
    m = Model('ref40')
    off_p = m.off_params()
    hist = []
    cache: Dict[float, float] = {}

    def max_off(inc: float) -> float:
        if inc not in cache:
            kc = dict(kc_refractory_period=REFRACTORY)
            if inc > 0:
                kc.update(kc_adapt_inc=inc, kc_adapt_tau=PILOT_TAU)
            m.n_sims += 1
            _, s = m._sim(off_p, kc_spiking=kc)
            cache[inc] = float(s.to_numpy().max())
            hist.append(dict(adapt_inc=inc, max_count_off=cache[inc]))
            print(f'  pilot: adapt_inc {inc:.5g} -> largest APL-off count {cache[inc]:.0f}', flush=True)
        return cache[inc]

    refr_only = max_off(0.0)
    strengths = {}
    for target in PILOT_TARGETS:
        lo, hi = np.log(PILOT_BRACKET[0]), np.log(PILOT_BRACKET[1])
        best = None
        for _ in range(PILOT_ITERS):
            mid = 0.5 * (lo + hi)
            x = float(np.exp(mid))
            c = max_off(x)
            if best is None or abs(c - target) < abs(best[1] - target) or (abs(c - target) == abs(best[1] - target) and x < best[0]):
                best = (x, c)
            if c == target:
                break
            if c > target:
                lo = mid
            else:
                hi = mid
        strengths[str(target)] = dict(adapt_inc=best[0], max_count_off=best[1])
    write_json(res_dir() / 'pilot.json', dict(base='ref40', tag=BASES['ref40']['tag'], refractory_period=REFRACTORY,
        adapt_tau=PILOT_TAU, targets=list(PILOT_TARGETS), max_count_off_base=float(m.sim(off_p)[1].to_numpy().max()),
        max_count_off_refractory_only=refr_only, strengths=strengths, history=hist, n_sims=m.n_sims, time=now(),
        host=socket.gethostname(), **code_state()))
    print(f'pilot: {strengths}', flush=True)


def strengths() -> List[float]:
    p = read_json(res_dir() / 'pilot.json')
    if p is None:
        raise RuntimeError('run --stage pilot first')
    return [float(p['strengths'][str(t)]['adapt_inc']) for t in PILOT_TARGETS]


def settings() -> List[dict]:
    """Every setting of part A: base, refractory period, strength index (1: the pilot's largest count
    near 40, 2: near 20, 3: near 10; 0: none), adaptation strength and time constant."""
    s = strengths()
    out = [dict(base='ref40', refr=REFRACTORY, a=0, inc=0.0, tau=np.nan)]
    out += [dict(base='ref40', refr=REFRACTORY, a=i + 1, inc=s[i], tau=tau) for i in range(3) for tau in TAUS]
    for key in ('e67cal', 'e67raw'):
        out += [dict(base=key, refr=REFRACTORY, a=i + 1, inc=s[i], tau=1.0) for i in range(3)]
        out += [dict(base=key, refr=REFRACTORY, a=2, inc=s[1], tau=0.2)]
    for x in out:
        x['tag'] = setting_tag(x)
    return out


def setting_tag(x: dict) -> str:
    t = f"{x['base']}_r{x['refr'] * 1e3:g}"
    if x['a'] > 0:
        t += f"_a{x['a']}_t{x['tau'] * 1e3:g}"
    return t


def bisect_factor(run: Callable[[float], Tuple[pd.DataFrame, pd.DataFrame]], target_n: int, size: int,
        bracket: Tuple[float, float], step0: float, label: str) -> Tuple[float, tuple, List[dict], bool]:
    """The factor at which run(factor)'s number of responses is `target_n` (the response rate falls
    with the factor), searched on a log scale: from 1, steps of `step0` (doubling) in the direction
    of the target until it is bracketed (within `bracket`), then regula falsi on the number of
    responses with the Illinois modification (each new point kept within the middle 90 % of the
    bracket), until the count is hit or the bracket is narrower than 1e-9 (log units) or
    BISECT_ITERS evaluations. Returns (factor, outputs, history, at_bound), the evaluated point
    nearest the target if it cannot be hit exactly; at_bound when `bracket` does not contain it."""
    hist, best = [], None

    def ev(x: float) -> int:
        nonlocal best
        f = float(np.exp(x))
        out = run(f)
        n = n_responses(out[0])
        hist.append(dict(stage=label, factor=f, n_responses=n, rate=n / size, target_n=target_n))
        print(f'  {label}: factor {f:.6g} -> {n} responses (rate {n / size:.5f}; target {target_n})', flush=True)
        if best is None or abs(n - target_n) < abs(best[1] - target_n):
            best = (f, n, out)
        return n - target_n

    lo_end, hi_end = np.log(bracket[0]), np.log(bracket[1])
    y0 = ev(0.0)
    if y0 != 0:
        # too many responses (y > 0): a larger factor
        direction = 1.0 if y0 > 0 else -1.0
        x_prev, y_prev, step = 0.0, y0, np.log(step0)
        while True:
            x = float(np.clip(x_prev + direction * step, lo_end, hi_end))
            y = ev(x)
            if y == 0 or np.sign(y) != np.sign(y0):
                break
            if x in (lo_end, hi_end):
                return best[0], best[2], hist, True
            x_prev, y_prev, step = x, y, 2 * step
        if y != 0:
            # a: y > 0 (smaller factor), b: y < 0
            (xa, ya), (xb, yb) = sorted([(x_prev, y_prev), (x, y)])
            last = 0
            for _ in range(BISECT_ITERS):
                if xb - xa < 1e-9:
                    break
                xm = xa - ya * (xb - xa) / (yb - ya)
                xm = float(np.clip(xm, xa + 0.05 * (xb - xa), xb - 0.05 * (xb - xa)))
                ym = ev(xm)
                if ym == 0:
                    break
                if ym > 0:
                    xa, ya = xm, ym
                    if last == 1:
                        yb /= 2
                    last = 1
                else:
                    xb, yb = xm, ym
                    if last == -1:
                        ya /= 2
                    last = -1
    f, n, out = best
    return f, out, hist, False


def stage_setting(tag: str) -> None:
    """One setting: the base with the refractory period / adaptation of the setting, its threshold
    factor bisected on the base's APL-off response rate, then one factor on the APL->KC and
    APL->bouton scales on the base's APL-on response rate; scored."""
    t0 = time.time()
    x = next(s for s in settings() if s['tag'] == tag)
    if tag in done_tags():
        print(f'{tag}: done', flush=True)
        return
    checks = read_json(res_dir() / 'base_checks.json', {})
    if x['base'] not in checks:
        raise RuntimeError(f"run --stage check --base {x['base']} first")
    bc = checks[x['base']]
    kc = dict(kc_refractory_period=float(x['refr']))
    if x['inc'] > 0:
        kc.update(kc_adapt_inc=float(x['inc']), kc_adapt_tau=float(x['tau']))
    m = Model(x['base'], kc)
    p0, size = dict(m.p), int(bc['size'])
    thr0 = p0['fixed_thr']
    f, off_out, h1, thr_bound = bisect_factor(lambda f: m.sim(m.off_params(dict(p0, fixed_thr=thr0 * f))),
        int(bc['n_responses_off']), size, THR_BRACKET, THR_STEP, f'{tag} threshold factor (APL off)')
    p1 = dict(p0, fixed_thr=thr0 * f)
    g, on_out, h2, apl_bound = bisect_factor(lambda g: m.sim(dict(p1, wAPLKC_scale=p0['wAPLKC_scale'] * g,
        wAPLPN_scale=p0['wAPLPN_scale'] * g)), int(bc['n_responses_on']), size, APL_BRACKET, APL_STEP,
        f'{tag} APL factor (APL on)')
    p = dict(p1, wAPLKC_scale=p0['wAPLKC_scale'] * g, wAPLPN_scale=p0['wAPLPN_scale'] * g)
    hist = pd.DataFrame(h1 + h2).assign(tag=tag)
    hp = res_dir() / 'bisection_history.csv'
    hist.to_csv(hp, mode='a', header=not hp.exists(), index=False)
    aplpn_e3 = float(m.fit.get('aplpn_factor_vs_e3', np.nan)) * g
    meta = dict(variant=f"r{x['refr'] * 1e3:g}" + (f"_a{x['a']}_t{x['tau'] * 1e3:g}" if x['a'] else ''), tag=tag, part='A',
        setting=tag, **base_meta(m), kc_refractory_period=float(x['refr']), kc_adapt_inc=float(x['inc']),
        kc_adapt_tau=float(x['tau']) if x['a'] else np.nan, adapt_strength=x['a'], thr_factor=f, apl_factor=g,
        target_rate_on=bc['rate_on'], target_rate_off=bc['rate_off'], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]),
        thr_at_bound=thr_bound, apl_at_bound=apl_bound, aplpn_factor_vs_e3=aplpn_e3, aplpn_over_cap=bool(aplpn_e3 > APLPN_CAP),
        n_sims=m.n_sims, seconds=np.nan, **params_meta(p))
    meta['n_kcs'] = int(on_out[0].shape[0])
    meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
    meta['seconds'] = time.time() - t0
    score(tag, meta, m, on_out, off_out)
    lp = res_dir() / 'settings_log.csv'
    pd.DataFrame([dict(tag=tag, base=x['base'], refractory_period=x['refr'], adapt_inc=x['inc'], adapt_tau=x['tau'],
        thr_factor=f, apl_factor=g, n_responses_on=n_responses(on_out[0]), n_responses_off=n_responses(off_out[0]),
        target_n_on=bc['n_responses_on'], target_n_off=bc['n_responses_off'], n_sims=m.n_sims,
        build_seconds=m.build_seconds, seconds=time.time() - t0, time=now(), host=socket.gethostname(), **code_state())]).to_csv(
        lp, mode='a', header=not lp.exists(), index=False)


def child(args: List[str], log: Path) -> int:
    """Runs this script with `args` in its own process (a C++ exception inside the parallel region
    kills the process without a traceback), output to `log`."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'a') as fh:
        fh.write(f'\n# {now()} {" ".join(args)}\n')
        fh.flush()
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args], stdout=fh, stderr=subprocess.STDOUT,
            cwd=H.HERE)
    print(f"{' '.join(args)}: exit {r.returncode} (log {log.name})", flush=True)
    return r.returncode


def stage_run() -> None:
    logs = out_dir() / 'logs'
    resolve_bases()
    failed = []
    for key in BASES:
        if key not in read_json(res_dir() / 'base_checks.json', {}):
            if child(['--stage', 'check', '--base', key], logs / f'check_{key}.log'):
                failed.append(f'check {key}')
    if not (res_dir() / 'pilot.json').exists():
        if child(['--stage', 'pilot'], logs / 'pilot.log'):
            raise RuntimeError('the pilot failed; see its log')
    checked = read_json(res_dir() / 'base_checks.json', {})
    for s in settings():
        if s['tag'] in done_tags() or s['base'] not in checked:
            continue
        if child(['--stage', 'setting', '--tag', s['tag']], logs / f"{s['tag']}.log"):
            failed.append(s['tag'])
    (res_dir() / 'run_failures.json').write_text(json.dumps(dict(time=now(), failed=failed), indent=1) + '\n')
    stage_summarize()
    if failed:
        print(f'FAILED: {failed}', flush=True)


# ---------------------------------------------------------------------------
# summary: merged tables, comparison table, figures, provenance
def merged() -> pd.DataFrame:
    st = pd.read_csv(res_dir() / 'partA_variant_stats.csv')
    ex = pd.read_csv(res_dir() / 'ceiling_readouts.csv')
    st = st.drop_duplicates('tag', keep='last').set_index('tag')
    ex = ex.drop_duplicates('tag', keep='last').set_index('tag')
    df = st.join(ex[[c for c in ex.columns if c not in st.columns]], how='inner')
    order = {s['tag']: i for i, s in enumerate([dict(tag=f'{k}_base') for k in BASES] + settings())}
    df['_o'] = [order.get(t, 999) for t in df.index]
    return df.sort_values('_o').drop(columns='_o').reset_index()


def fmt(x, spec='.3f') -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return '-'
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def pct(x, spec='.1f') -> str:
    return '-' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{100 * x:{spec}} %'


def setting_label(r) -> str:
    if r.setting == 'base':
        return 'base (no ceiling)'
    s = '3 ms refractory'
    if r.kc_adapt_inc > 0:
        s += f'<br>adaptation {r.kc_adapt_inc:.3g} (strength {int(r.adapt_strength)}), τ {r.kc_adapt_tau * 1e3:g} ms'
    return s


def write_table(df: pd.DataFrame) -> None:
    from apl_effect_scores import opk_band
    dt = H.data_target
    pilot = read_json(res_dir() / 'pilot.json', {})
    bs = bases()
    L = ['# E69: a firing ceiling for the KCs, part A (fixed scales, both response rates held)', '',
        f'Generated {datetime.now():%Y-%m-%d %H:%M} by E69_kc_firing_ceiling.py. Bases: ' +
        '; '.join(f"{k}: {b['label']} ({b['exp']}, {b['tag']})" for k, b in bs.items()) + '.', '',
        f"Pilot (base ref40, APL off, 3 ms refractory period, τ {PILOT_TAU:g} s): largest APL-off count {fmt(pilot.get('max_count_off_base'), '.0f')} "
        f"without a ceiling, {fmt(pilot.get('max_count_off_refractory_only'), '.0f')} with the refractory period alone; strengths " +
        ', '.join(f"{v['adapt_inc']:.4g} (largest count {v['max_count_off']:.0f}, target {t})" for t, v in pilot.get('strengths', {}).items()) +
        '. Adaptation is in units of the KC\'s own threshold (a KC spikes above thr x (1 + a); each spike raises a by the strength).', '',
        'Each setting re-tuned: threshold factor on the base\'s APL-off response rate, then one factor on the APL->KC and APL->bouton '
        'scales on its APL-on response rate. The APL-effect loss: root mean squared difference of the per-pair APL effect from the '
        'data\'s over the 120 pairs, lower is better; "cal" under the calibrated saturating readout (K re-calibrated on each variant\'s '
        'APL-on counts). Amplitudes in z: E68\'s one-spike floor with measurement noise.', '']
    for key, g in df.groupby('base', sort=False):
        L += [f"## {bs[key]['label']} ({key})", '',
            '| setting | largest count<br>APL on / off | APL-effect loss, raw<br>(mean; per-odor; pair-specific) | ρ per-pair<br>raw | '
            'APL-effect loss, cal<br>(mean; per-odor; pair-specific) | ρ per-pair<br>cal | K | APL off: above 4 z /<br>responders\' mean z | '
            'odors per KC<br>APL on / off | block<br>raw / cal | response rate ρ<br>APL on / off | rates on / off<br>(ratio) | '
            'threshold /<br>APL factor | realism flag |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in g.itertuples():
            L.append(f"| {setting_label(r)} | {r.max_count_on:.0f} / {r.max_count_off:.0f} | {fmt(r.apl_raw_rmse)} ({r.apl_raw_mean:+.3f}; "
                f"{fmt(r.apl_raw_per_odor)}; {fmt(r.apl_raw_pair_specific)}) | {fmt(r.apl_raw_rho, '.2f')} | {fmt(r.apl_cal_rmse)} "
                f"({r.apl_cal_mean:+.3f}; {fmt(r.apl_cal_per_odor)}; {fmt(r.apl_cal_pair_specific)}) | {fmt(r.apl_cal_rho, '.2f')} | "
                f"{fmt(r.cal_K, '.2f')} | {pct(r.z_floor_noise_share_above_4_acute_off)} / {fmt(r.z_floor_noise_resp_amp_mean_acute_off, '.2f')} | "
                f"{fmt(r.opk_on_tv, '.2f')} / {fmt(r.opk_off_tv, '.2f')} | {fmt(r._asdict()[f'{V}__block_fruity_small_iso'])} / {fmt(r.block_cal)} | "
                f"{fmt(r.spearman_rate_on_vs_data, '.2f')} / {fmt(r.spearman_rate_off_vs_data, '.2f')} | {r.rate_on:.4f} / {r.rate_off:.4f} "
                f"({r.rate_off / r.rate_on:.2f}) | {fmt(r.thr_factor, '.4f')} / {fmt(r.apl_factor, '.3f')} | {r.realism_flag if isinstance(r.realism_flag, str) and r.realism_flag else '-'} |")
        L.append(f"| KC data | - | floor {fmt(dt('apl_effect_rmse_floor'))} (+{dt('apl_effect_mean'):.3f}) | ceiling "
            f"{fmt(dt('apl_effect_rho_ceiling'), '.2f')} | floor {fmt(dt('apl_effect_rmse_floor'))} (+{dt('apl_effect_mean'):.3f}) | - | - | "
            f"{pct(dt('tnt_share_z_above_4'))} / {fmt(dt('tnt_resp_amp_mean_z'), '.2f')} | bands {opk_band('on'):.2f} / {opk_band('off'):.2f} | "
            f"{fmt(dt('block'))} / - | - | {dt('ctrl_frac'):.4f} / {dt('tnt_frac'):.4f} ({dt('rate_ratio_fraction'):.2f}) | - | - |")
        L += ['', '| setting | responders\' median / 99th percentile count, APL on | APL off | share of responding entries at '
            f'{STRONG_COUNT}+ spikes, on / off | their share of the squared responses, on / off | KCs at {STRONG_COUNT}+ to some odor, on / off | '
            'responders-only participation ratio, raw (cal), on / off | top-1 % share, raw, on / off | responders\' CV, raw, on / off | '
            'APL on: above 4 z / responders\' mean z | 0.5-1.5 z band, on / off | binarized loss (mean) | ρ on odor-odor distances, '
            'APL on / off | natural-source alignment, APL on / off / on minus off | mean distance, APL on / off |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in g.itertuples():
            d = r._asdict()
            von, voff = d[f'{V}__on_spearman_resid_vs_vcf_hyp'], d[f'{V}__off_spearman_resid_vs_vcf_hyp']
            L.append(f"| {setting_label(r)} | {r.median_resp_count_on:.0f} / {r.p99_resp_count_on:.0f} | {r.median_resp_count_off:.0f} / "
                f"{r.p99_resp_count_off:.0f} | {pct(r.share_resp_ge20_on)} / {pct(r.share_resp_ge20_off)} | {pct(r.sq_share_ge20_on, '.0f')} / "
                f"{pct(r.sq_share_ge20_off, '.0f')} | {r.n_kcs_ge20_on} / {r.n_kcs_ge20_off} of {r.n_kcs_on} | {pct(r.pr_resp_raw_on)} "
                f"({pct(r.pr_resp_cal_on)}) / {pct(r.pr_resp_raw_off)} ({pct(r.pr_resp_cal_off)}) | {fmt(r.top1_resp_raw_on, '.2f')} / "
                f"{fmt(r.top1_resp_raw_off, '.2f')} | {fmt(r.cv_resp_raw_on, '.2f')} / {fmt(r.cv_resp_raw_off, '.2f')} | "
                f"{pct(r.z_floor_noise_share_above_4_on)} / {fmt(r.z_floor_noise_resp_amp_mean_on, '.2f')} | "
                f"{pct(r.z_floor_noise_share_05_15_on)} / {pct(r.z_floor_noise_share_05_15_acute_off)} | {fmt(r.apl_bin_rmse)} "
                f"({r.apl_bin_mean:+.3f}) | {fmt(d[f'{V}__on_spearman'])} / {fmt(d[f'{V}__off_spearman'])} | {fmt(von, '.2f')} / "
                f"{fmt(voff, '.2f')} / {fmt(von - voff, '+.2f')} | {fmt(d[f'{V}__on_mean_model'])} / {fmt(d[f'{V}__off_mean_model'])} |")
        L.append(f"| KC data | - | - | - | - | - | {pct(dt('ctrl_participation_ratio_resp'))} / {pct(dt('tnt_participation_ratio_resp'))} | "
            f"{dt('ctrl_top1_share_resp'):.2f} / {dt('tnt_top1_share_resp'):.2f} | {dt('ctrl_resp_amp_cv'):.2f} / {dt('tnt_resp_amp_cv'):.2f} | "
            f"{pct(dt('ctrl_share_z_above_4'))} / {dt('ctrl_resp_amp_mean_z'):.2f} | {pct(dt('ctrl_share_z_05_15'))} / {pct(dt('tnt_share_z_05_15'))} | "
            f"floor {fmt(dt('apl_effect_binary_rmse_floor'))} (+{dt('apl_effect_binary_mean'):.3f}) | ceilings 0.98 / 0.97 | "
            f"{dt('ctrl_vcf'):.2f} / {dt('tnt_vcf'):.2f} / {dt('vcf_apl_effect'):+.2f} | {dt('ctrl_mean_dist'):.3f} / {dt('tnt_mean_dist'):.3f} |")
        L.append('')
    wp = pd.read_csv(res_dir() / 'worst_pairs.csv').drop_duplicates(['tag', 'readout', 'rank'], keep='last')
    L += ['## The reference model\'s five worst pairs (E66): pair-specific error, and the KCs carrying half of the model\'s per-pair effect', '',
        '| variant | readout | ' + ' | '.join(f'{a}-{b}' for a, b in worst_pairs_e66()) + ' |', '|---|---|' + '---:|' * 5]
    for (tag, readout), g in wp.groupby(['tag', 'readout'], sort=False):
        g = g.sort_values('rank')
        L.append(f'| {tag} | {readout} | ' + ' | '.join(f'{r.pair_specific:+.3f} ({r.n_kcs_half} KCs)' for r in g.itertuples()) + ' |')
    (res_dir() / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def figures(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = H.data_target
    colors = {'ref40': '#2a78d6', 'e67cal': '#1baf7a', 'e67raw': '#eb6834'}
    bs = bases()
    tau_marker = {0.05: 'v', 0.2: 's', 1.0: 'o'}

    def marker(r):
        return 'D' if r.setting == 'base' else ('^' if not r.kc_adapt_inc > 0 else tau_marker.get(round(r.kc_adapt_tau, 3), 'o'))

    panels = [('apl_raw_rmse', 'APL-effect loss, raw counts', dt('apl_effect_rmse_floor'), 'data floor'),
        ('apl_cal_rmse', 'APL-effect loss, calibrated readout', dt('apl_effect_rmse_floor'), 'data floor'),
        ('apl_raw_mean', 'mean APL effect, raw counts', dt('apl_effect_mean'), 'KC data'),
        ('apl_cal_mean', 'mean APL effect, calibrated readout', dt('apl_effect_mean'), 'KC data'),
        ('apl_raw_pair_specific', 'pair-specific part, raw counts', None, None),
        ('apl_cal_pair_specific', 'pair-specific part, calibrated readout', None, None)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (col, title, ref, ref_lab) in zip(axes.T.ravel(), panels):
        for key, g in df.groupby('base', sort=False):
            g = g.sort_values('max_count_off')
            ax.plot(g.max_count_off, g[col], '-', color=colors[key], alpha=0.35, lw=1)
            for r in g.itertuples():
                ax.plot(r.max_count_off, getattr(r, col), marker(r), color=colors[key], ms=7, mfc='none' if r.setting == 'base' else colors[key])
        if ref is not None:
            ax.axhline(ref, color='k', ls=':', lw=1.2, label=ref_lab)
        ax.set_xscale('log')
        ax.set_xlabel('largest APL-off spike count (log)')
        ax.set_title(title, fontsize=10)
    handles = [plt.Line2D([], [], color=c, marker='o', ls='', label=bs[k]['label']) for k, c in colors.items()]
    handles += [plt.Line2D([], [], color='gray', marker=m, ls='', mfc='gray' if m != 'D' else 'none', label=l) for m, l in
        (('D', 'base (no ceiling)'), ('^', '3 ms refractory alone'), ('v', 'adaptation, τ 50 ms'), ('s', 'τ 200 ms'), ('o', 'τ 1 s'))]
    handles.append(plt.Line2D([], [], color='k', ls=':', label='KC data (floor of the loss / mean APL effect)'))
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8, frameon=False)
    fig.suptitle('The firing ceiling: the APL-effect loss and its parts against the largest APL-off spike count (both response rates held)', fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(res_dir() / 'E69_loss_vs_ceiling.png', dpi=130)
    plt.close(fig)

    panels = [('z_floor_noise_share_above_4_acute_off', 'APL off: entries above 4 z', dt('tnt_share_z_above_4'), 'APL output blocked (TNT)', True),
        ('z_floor_noise_resp_amp_mean_acute_off', 'APL off: responders\' mean amplitude (z)', dt('tnt_resp_amp_mean_z'), 'APL output blocked (TNT)', False),
        ('z_floor_noise_share_05_15_acute_off', 'APL off: entries 0.5-1.5 z', dt('tnt_share_z_05_15'), 'APL output blocked (TNT)', True),
        ('opk_off_tv', 'odors per KC, APL off: distance from the flies', None, None, False),
        ('opk_on_tv', 'odors per KC, APL on: distance from the flies', None, None, False),
        ('pr_resp_raw_off', 'APL off: responders-only participation ratio, raw', dt('tnt_participation_ratio_resp'), 'APL output blocked (TNT)', True)]
    from apl_effect_scores import opk_band
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (col, title, ref, ref_lab, is_pct) in zip(axes.ravel(), panels):
        for key, g in df.groupby('base', sort=False):
            g = g.sort_values('max_count_off')
            ax.plot(g.max_count_off, g[col] * (100 if is_pct else 1), '-', color=colors[key], alpha=0.35, lw=1)
            for r in g.itertuples():
                ax.plot(r.max_count_off, getattr(r, col) * (100 if is_pct else 1), marker(r), color=colors[key], ms=7,
                    mfc='none' if r.setting == 'base' else colors[key])
        if ref is not None:
            ax.axhline(ref * (100 if is_pct else 1), color='k', ls=':', lw=1.2, label=ref_lab)
        if col.startswith('opk_'):
            ax.axhline(opk_band('off' if 'off' in col else 'on'), color='crimson', ls='--', lw=1, label='realism band')
        ax.set_xscale('log')
        ax.set_xlabel('largest APL-off spike count (log)')
        ax.set_ylabel('%' if is_pct else '')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    fig.legend(handles=handles[:-1], loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle('The firing ceiling: the APL-off amplitudes (z, E68\'s conversion with the data\'s noise) and the odors per KC', fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(res_dir() / 'E69_apl_off_amplitudes.png', dpi=130)
    plt.close(fig)

    # the reference base's spike-count distributions with APL off (responding entries), at τ 1 s
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    g = df[(df.base == 'ref40') & ((df.setting == 'base') | (df.kc_adapt_inc.fillna(0) == 0) | (df.kc_adapt_tau.round(3) == 1.0))]
    cmap = plt.get_cmap('viridis')
    for i, r in enumerate(g.itertuples()):
        for ax, cond in zip(axes, CONDS):
            m = pd.read_parquet(out_dir() / f'{r.tag}_{cond}_spike_counts.parquet').to_numpy(dtype=float).ravel()
            v = np.sort(m[m >= RESP_MIN])
            ccdf = 1 - np.arange(len(v)) / len(v)
            ax.step(v, ccdf, where='post', color=cmap(i / max(1, len(g) - 1)), label=setting_label(r).replace('<br>', ', '))
    for ax, cond in zip(axes, CONDS):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.axvline(STRONG_COUNT, color='gray', ls=':', lw=1)
        ax.set_xlabel('spike count of a responding entry (2 or more spikes)')
        ax.set_ylabel('share of responding entries with at least this count')
        ax.set_title(f"{bs['ref40']['label']}: APL {'on' if cond == 'on' else 'off'}", fontsize=10)
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(res_dir() / 'E69_spike_count_distributions.png', dpi=130)
    plt.close(fig)


def stage_summarize() -> None:
    df = merged()
    df.to_csv(res_dir() / 'stats_variants.csv', index=False)
    long = pd.read_csv(res_dir() / 'model_pair_distances_long_A.csv')
    long = long.drop_duplicates(['tag', 'condition', 'odor_a', 'odor_b'], keep='last')
    long.to_csv(res_dir() / 'model_pair_distances_long.csv', index=False)
    write_table(df)
    figures(df)
    write_provenance(df)
    cols = ['tag', 'max_count_off', 'apl_raw_rmse', 'apl_raw_mean', 'apl_cal_rmse', 'apl_cal_mean', 'cal_K',
        'z_floor_noise_share_above_4_acute_off', 'z_floor_noise_resp_amp_mean_acute_off', 'opk_on_tv', 'opk_off_tv',
        f'{V}__block_fruity_small_iso', 'block_cal', 'spearman_rate_on_vs_data', 'thr_factor', 'apl_factor', 'realism_flag']
    print(df[[c for c in cols if c in df.columns]].round(4).to_string(index=False), flush=True)


def write_provenance(df: pd.DataFrame) -> None:
    import olfsysm
    checks = read_json(res_dir() / 'base_checks.json', {})
    # (bases: each base's fit_mb_model kwargs, as the archive's params migration reads them; the selection in
    # base_selection)
    prov = dict(generated=now(), host=socket.gethostname(), script='E69_kc_firing_ceiling.py', experiment=EXPERIMENT,
        bases={k: v.get('kws_scalar', {}) for k, v in checks.items()}, base_selection=bases(), base_checks=checks,
        pilot=read_json(res_dir() / 'pilot.json', {}),
        settings=settings(), refractory_period=REFRACTORY, taus=TAUS, pilot_targets=PILOT_TARGETS, thr_bracket=THR_BRACKET,
        apl_bracket=APL_BRACKET, bisect_iters=BISECT_ITERS, aplpn_cap=APLPN_CAP, worst_pairs=worst_pairs_e66(),
        adaptation_form='olfsysm sim_KC_layer: a(t) = exp(-dt / adapt_tau) a(t-1) + adapt_inc spike(t-1); spike when Vm > thr + a |thr| '
            '(thr * (1 + a) for the positive thresholds here); refractory: Vm held at 0 and no spike for ceil(refractory_period / dt) - 1 '
            'steps after a spike',
        olfsysm_build=olfsysm.build_info() if hasattr(olfsysm, 'build_info') else None,
        model_provenance=H.model_provenance(orn_deltas=H.apl16_orn_deltas()),
        n_variants=int(len(df)), run_failures=read_json(res_dir() / 'run_failures.json', {}))
    if isinstance(prov['olfsysm_build'], dict):
        prov['olfsysm_build'].pop('git_diff', None)
    write_json(res_dir() / 'provenance.json', prov)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--stage', required=True, choices=['check', 'pilot', 'setting', 'run', 'summarize', 'bases', 'list'])
    ap.add_argument('--base', choices=list(BASES))
    ap.add_argument('--tag')
    args = ap.parse_args()
    if args.stage == 'bases':
        print(json.dumps(resolve_bases(), indent=1, default=str))
    elif args.stage == 'list':
        for s in settings():
            print(s)
    elif args.stage == 'check':
        stage_check(args.base)
    elif args.stage == 'pilot':
        stage_pilot()
    elif args.stage == 'setting':
        stage_setting(args.tag)
    elif args.stage == 'run':
        stage_run()
    elif args.stage == 'summarize':
        stage_summarize()


if __name__ == '__main__':
    main()
