#!/usr/bin/env python3
"""E71: the input-scaled KC threshold. Which KCs are released when APL is blocked, and whether the APL
effect survives a release spread over the KCs as in the flies.

Every model KC had the same somatic threshold apart from a term proportional to its spontaneous input
(threshold = fixed_thr + 2 x spont_in), so a KC with ten claws needs the same depolarization as one with
three. Here each KC's threshold is multiplied by (W / mean W)^a, W the KC's summed PN->claw weight in the
wiring the model is given (the glomerulus gain included where the base has one; without a gain W is the
KC's number of claws), through the simulator's per-KC threshold factor (`kc_thr_factor`;
E10_joint_fit.kc_input_thr_factor), identically in the APL-on and the acute APL-off runs. a = 0 is the
present model.

Part A (`--stage run`): the sweep at fixed scales. Bases (`resolve_bases`, recorded in bases.json on first
use): 'c2', the APL-effect fits' (E67) `rate0.068_aplboth_k2_cap3` as the firing ceiling experiment (E69)
loaded it (its bases.json 'e67cal'); 'c2ceil', the same with the firing ceiling at hal's E69 part-B setting
(its fit 1, at strength 1 and tau 1 s, when fits.csv and the archive have it, else part A's re-tuned
`e67cal_r3_a1_t1000`); 'ref40', the reference model (E44_operating_point `thr40_gain1.5_rate0.068_k2_cap3`).
Exponents: c2 -0.5, 0.25, 0.5, 0.75, 1, and 0.5, 1 with the base's breadth kept (the control that separates
the threshold from the change in the APL weights); ref40 0.25, 0.5, 0.75, 1; c2ceil 0.5, 1 and the
neighbours of c2's chosen exponent on the 0.25 grid (its choice itself when it is neither 0.5 nor 1; see
`c2ceil_exponents`). Each base is first re-run and must reproduce its archived spike counts exactly
(`--stage check`). At every setting (`--stage setting`, one process each, since a C++ exception inside an
OpenMP region kills the process without a traceback): the breadth (each KC's APL-off spike total in the
family's tuned base, E10_joint_fit.bouton_base's pre-pass, which the APL->KC weights are built from) is
recomputed with the same factor, the pre-pass's common threshold re-tuned to its own APL-off response
rate (`E10_joint_fit.input_scaled_prepass`); both response rates are held at the base's by two bisections
(E69's `bisect_factor`: the somatic threshold factor on the APL-off response rate, then one common factor
on the APL->KC and APL->bouton scales on the APL-on response rate; KC-KC and everything else as fitted);
the re-tuned absolute values are recorded.

Readout per variant, beside the base and the KC data (`score`): the firing ceiling experiment's (E69's
`score` minus its worst pairs: the harness stats columns of E66's `score_variant`, the APL-effect loss with
its mean, per-odor and pair-specific parts and ρ on the per-pair APL effect on raw counts and under the
calibrated readout with K recorded, binarized against the binarized data, the spike-count distribution,
the amplitudes through E68's conversion, the odors per KC against the bands, the realism flag), plus: the
mean pair distances with APL on and off under each reading; the share of silent KCs and of KCs responding
to 8 or more odors and the squared coefficient of variation of the number of odors per KC against the
recordings' range; who is released (responding entries with APL on and off by claw-count group, each
group's share of the responses that blocking APL adds, ρ of W and of the claw count with the number of
odors per KC in each condition; released_by_group.csv); E70's high-firing set and survivors (20 or more
spikes to some odor with APL off; and with APL on) and how many of the base's survivors
(results/E70_high_firing_kcs/survivors.csv) remain; the KC thresholds (mean, spread, the share of the
spontaneous-input term, the log-log slope and ρ of the threshold on W: how much of the threshold already
scales with the input).

The exponent of a base (`choose_exponent`): among its settings with the breadth recomputed and the base
itself (a = 0) whose bisections held both response rates, those whose APL-on odors-per-KC distance is
inside its band and whose APL-on response rate ρ is at or above 0.37; of these the smallest APL-off
odors-per-KC distance; within 0.01 of it, the exponent nearest 0. No odor-odor distance and no APL effect
enters the choice.

Part B (`--stage partB-commands`): the E10_joint_fit.py commands for the re-fits under the APL-effect loss
with both definitions (`--loss apl --apl-readout both`, k 2, `--aplpn-max 3`, `--rate-target ctrl_frac`,
150 evaluations, `--kc-thr-input-exp A` fixed, from part A's tuned_params.json, results in this experiment's
folder), printed only for the fits whose conditions hold; `--stage partB-check` compares each fit's first
evaluation with part A's row of its start. Part C (`--stage partC-commands`): E43_model_trial_noise.py commands (the block-target
experiment's settings, E48: 10 pseudo-recordings of 3 trials at the matched noise, seed 43, each
pseudo-recording's pair distances saved) and the single-pair scoring of the saved pseudo-recordings
(`--stage partC`).

Outputs: results/E71_input_scaled_threshold/ (bases.json, base_checks.json, settings_log.csv,
bisection_history.csv, partA_variant_stats.csv, input_readouts.csv, released_by_group.csv, parts_A.csv,
shares_A.csv, stats_variants.csv, model_pair_distances_long.csv, tuned_params.json, exponents.json,
comparison_table.md, figures, provenance.json); per-variant counts and responses in
outputs/E71_input_scaled_threshold/<tag>_<on|acute_off>_<spike_counts|responses>.parquet, the bases'
breadth runs and the pre-passes in outputs/E71_input_scaled_threshold/bases/.

Usage (from this folder; one process at a time, about 9 GB each):
    ../../venv/bin/python E71_input_scaled_threshold.py --plan
    OMP_NUM_THREADS=12 OMP_THREAD_LIMIT=12 nice -n 10 ../../venv/bin/python -u E71_input_scaled_threshold.py --stage run
    ../../venv/bin/python E71_input_scaled_threshold.py --stage summarize
    ../../venv/bin/python E71_input_scaled_threshold.py --stage partB-commands
    --local-test DIR: results and outputs under DIR (a smoke test; with --stage setting --tag <tag> one setting)
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sstats

import kc_remapping_harness as H
import E69_kc_firing_ceiling as E69

EXPERIMENT = 'E71_input_scaled_threshold'
RESULTS = H.shared_dir() / 'results'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
E67_EXP = 'E67_apl_effect_fits'
E69_EXP = 'E69_kc_firing_ceiling'
ODORS = list(H.APL16_ODORS)
V = H.PRIMARY_MODEL_RDM
RESP_MIN = 2.0
CONDS = ('on', 'acute_off')

# the bisections (E69's: log scale, regula falsi with the Illinois modification)
THR_BRACKET, THR_STEP = (0.1, 10.0), 1.1
APL_BRACKET, APL_STEP = (1 / 32, 32.0), 1.4
# the APL-effect fits' cap on the APL->bouton scale, as a multiple of the bouton base's (E3)
APLPN_CAP = 3.0
# E70's definitions: the high-firing set (>= 20 spikes to some odor with APL off) and the survivors (also with APL on)
HIGH_SPIKES = 20.0
# claw-count groups for who is released (1-4: the third of KCs with the fewest claws; 9+: the 14 % with the most)
CLAW_GROUPS = (('1-4', 1, 4), ('5-6', 5, 6), ('7-8', 7, 8), ('9+', 9, 10 ** 6))
EXPONENTS = {'c2': (-0.5, 0.25, 0.5, 0.75, 1.0), 'ref40': (0.25, 0.5, 0.75, 1.0)}
KEPT_BREADTH = {'c2': (0.5, 1.0)}
C2CEIL_FIXED = (0.5, 1.0)
GRID_STEP = 0.25
# the exponent rule: the APL-on response rate ρ floor (the realism band) and the tie margin on the APL-off distance
TIE_MARGIN = 0.01
BASE_ORDER = ('c2', 'ref40', 'c2ceil')
GENO_LABEL = {'on': 'control flies', 'acute_off': 'APL output blocked (TNT)'}

# part B / C (the entry's conditions)
PARTB_OPK_GAIN = 0.03
PARTB_MAXFEV = 150
PARTC_FIT_MARGIN = 0.005
C2_LOSSES = dict(raw=0.0932860975999109, cal=0.0560594911547915)

_DIRS: Dict[str, Path] = {}


def res_dir() -> Path:
    d = _DIRS.get('res') or H.results_dir(EXPERIMENT)
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir() -> Path:
    d = _DIRS.get('out') or H.outputs_dir(EXPERIMENT)
    d.mkdir(parents=True, exist_ok=True)
    return d


def base_dir() -> Path:
    d = out_dir() / 'bases'
    d.mkdir(parents=True, exist_ok=True)
    return d


read_json, write_json, now, code_state = E69.read_json, E69.write_json, E69.now, E69.code_state
rate, n_responses = E69.rate, E69.n_responses


# ---------------------------------------------------------------------------
# bases
def base2_from_e69() -> dict:
    """The firing-ceiling base: hal's E69 part-B fit 1 (C2's family at the literature setting, strength 1,
    tau 1 s) when results/E69_kc_firing_ceiling/fits.csv has it and its counts are archived; else part A's
    re-tuned setting `e67cal_r3_a1_t1000` (C2's family at E69's re-tuned absolute parameters)."""
    e69_bases = read_json(RESULTS / E69_EXP / 'bases.json')
    c2 = e69_bases['e67cal']
    tp = read_json(RESULTS / E69_EXP / 'tuned_params.json')['e67cal_r3_a1_t1000']
    fits = RESULTS / E69_EXP / 'fits.csv'
    if fits.exists():
        f = pd.read_csv(fits)
        need = ('kc_refractory_period', 'kc_adapt_inc', 'kc_adapt_tau', 'k', 'claw_thr', 'gain', 'init_params')
        if all(c in f.columns for c in need):
            m = f[np.isclose(f.kc_refractory_period, tp['kc_refractory_period']) & np.isclose(f.kc_adapt_inc, tp['kc_adapt_inc'])
                & np.isclose(f.kc_adapt_tau, tp['kc_adapt_tau']) & np.isclose(f.k, 2.0) & f.claw_thr.isna()
                & (f.gain.isna() | np.isclose(f.gain.fillna(1.0), 1.0))
                & f.init_params.astype(str).str.contains('e67cal_r3_a1_t1000')]
            m = m[[(ARCHIVE / E69_EXP / f'{t}_on_spike_counts.parquet').exists() for t in m.tag]]
            if len(m):
                r = m.iloc[-1]
                return dict(label='C2 with the firing ceiling (hal\'s E69 part-B fit at the literature setting: 3 ms refractory, '
                    f"adaptation {tp['kc_adapt_inc']:.4g} with tau 1 s)", exp=E69_EXP, tag=str(r.tag), params_from=None,
                    archive=f'{E69_EXP}/{r.tag}', survivors_model=f"e67__{c2['tag']}", source='E69 part B fit (fits.csv)')
    return dict(label='C2 with the firing ceiling (E69 part A\'s e67cal_r3_a1_t1000: 3 ms refractory, adaptation '
        f"{tp['kc_adapt_inc']:.4g} with tau 1 s, both response rates re-tuned)", exp=c2['exp'], tag=c2['tag'],
        params_from=f'{E69_EXP}/tuned_params.json:e67cal_r3_a1_t1000', archive=f'{E69_EXP}/e67cal_r3_a1_t1000',
        survivors_model=f"e67__{c2['tag']}", source='E69 part A setting (hal\'s part-B fit not archived at resolution)')


def resolve_bases(write: bool = True, base2: Optional[str] = None) -> Dict[str, dict]:
    """The three bases, from bases.json once written (so every setting uses the same bases). `base2`
    ('partA' or an E69 fits.csv tag) overrides the firing-ceiling base's resolution before that."""
    p = res_dir() / 'bases.json'
    saved = read_json(p)
    if saved is not None:
        return saved
    e69_bases = read_json(RESULTS / E69_EXP / 'bases.json')
    c2, ref = e69_bases['e67cal'], e69_bases['ref40']
    out = dict(
        c2=dict(label='C2: the first joint fit\'s family at breadth exponent 2, fitted under the APL-effect loss with both '
            'definitions at 6.8 % (E67)', exp=c2['exp'], tag=c2['tag'], params_from=None, archive=f"{c2['exp']}/{c2['tag']}",
            survivors_model=f"e67__{c2['tag']}", source='E69 bases.json e67cal'),
        ref40=dict(label='the reference model (eight at 1.5, 40 Hz, 6.8 %)', exp=ref['exp'], tag=ref['tag'], params_from=None,
            archive=f"{ref['exp']}/{ref['tag']}", survivors_model='ref40', source='E69 bases.json ref40'),
        c2ceil=base2_from_e69())
    if base2 == 'partA':
        out['c2ceil'] = dict(base2_from_e69(), exp=c2['exp'], tag=c2['tag'],
            params_from=f'{E69_EXP}/tuned_params.json:e67cal_r3_a1_t1000', archive=f'{E69_EXP}/e67cal_r3_a1_t1000',
            source='E69 part A setting (chosen with --base2 partA)')
    elif base2:
        out['c2ceil'] = dict(out['c2ceil'], exp=E69_EXP, tag=base2, params_from=None, archive=f'{E69_EXP}/{base2}',
            source='E69 part B fit (chosen with --base2)')
    if write:
        write_json(p, dict(out, _resolved=dict(time=now(), host=socket.gethostname())))
        return read_json(p)
    return out


def bases() -> Dict[str, dict]:
    return {k: v for k, v in resolve_bases().items() if not k.startswith('_')}


def copy_base_caches() -> None:
    """The bases' APL-off breadth runs as the firing ceiling experiment archived them (E69 checked its bases
    reproduce their archived counts with them); bouton_base regenerates any that are missing."""
    for src in (ARCHIVE / E69_EXP / 'bases', H.outputs_dir(E69_EXP) / 'bases' if (H.HERE / 'outputs' / E69_EXP).exists() else None):
        if src is None or not src.exists():
            continue
        for p in src.glob('*_base_acute_off_*.parquet'):
            dst = base_dir() / p.name
            if not dst.exists():
                dst.write_bytes(p.read_bytes())


def n_claws_per_kc(wPNKC: pd.DataFrame) -> pd.Series:
    """Claws per KC (claw rows with any input), indexed by KC id; for a KC x glomerulus matrix, its row sums."""
    if H.KC_ID_LEVEL in (wPNKC.index.names or []) and len(wPNKC.index.names) > 1:
        return (wPNKC != 0).any(axis='columns').groupby(level=H.KC_ID_LEVEL).sum().astype(float)
    return wPNKC.astype(float).sum(axis='columns')


class Model:
    """One base with the input-scaled threshold at exponent `a` (0: the base itself), as a fixed-scale
    simulator: the base's fitted (or re-tuned) parameters, its KC refractory period / adaptation, the
    APL->KC weights from the breadth recomputed with the factor (`breadth='recomputed'`) or the base's
    (`'kept'`); `sim(params)` passes the factor to every simulation and returns (responses, spike counts)
    with columns in ODORS order."""

    def __init__(self, key: str, a: float = 0.0, breadth: str = 'recomputed'):
        import E43_model_trial_noise as E43
        from al_analysis.mb_model import make_fixed_param_kc_simulator
        from E10_joint_fit import bouton_base, breadth_weights, kc_input_thr_factor, kc_input_weight
        if breadth not in ('recomputed', 'kept'):
            raise ValueError(f'{breadth=}')
        b = bases()[key]
        self.key, self.base, self.a, self.breadth_mode = key, b, float(a), breadth
        copy_base_caches()
        name = E43.model_from(f"{b['exp']}:{b['tag']}")
        self.lab_orn = H.apl16_orn_deltas()
        fm = E43.fit_model(name, self.lab_orn, base_dir())
        if fm['info'].get('wiring') or fm['info'].get('orn_input'):
            raise NotImplementedError(f'{key}: re-drawn wirings / other ORN inputs are not used as bases here')
        self.orn, self.p, self.kws, self.info = fm['orn'], dict(fm['p_on']), dict(fm['kws']), fm['info']
        self.fit = E43.fit_row(b['exp'], b['tag'])
        # the KC refractory period / adaptation: from the fit row (already in the kwargs), or with the parameters of
        # a re-tuned setting (`params_from`, passed to the simulator here)
        kc_keys = ('kc_refractory_period', 'kc_adapt_inc', 'kc_adapt_tau')
        self.kc_extra: Dict[str, float] = {}
        if b.get('params_from'):
            path, k = b['params_from'].rsplit(':', 1)
            tp = read_json(H.shared_dir() / 'results' / path)[k]
            self.p = {kk: float(tp[kk]) for kk in self.p}
            self.kc_extra = {kk: float(tp[kk]) for kk in kc_keys if tp.get(kk) is not None}
            if any(kk in self.kws for kk in self.kc_extra):
                raise RuntimeError(f'{key}: the fit row already sets a firing ceiling')
        self.kc = {**{kk: float(self.kws[kk]) for kk in kc_keys if kk in self.kws}, **self.kc_extra}
        gain = self.info.get('gain', 1.0)
        recompute = self.a != 0.0 and breadth == 'recomputed'
        pre: dict = {}
        self.initial, wnorm, self.W, breadth_used = bouton_base(self.orn, base_dir(), claw_thr=self.info.get('claw_thr'),
            gain=None if gain == 1.0 else gain, gain_gloms=self.info.get('gain_glomeruli') or None,
            kc_thr_input_exp=self.a if recompute else None, info=pre)
        self.prepass = pre.get('kc_thr_input_prepass', {})
        if recompute:
            self.kws['_wAPLKC'] = breadth_weights(wnorm['wAPLKC'], breadth_used, float(self.info['k']))
        self.kc_ids = pd.Index(breadth_used.index)
        self.W_in = kc_input_weight(self.W).reindex(self.kc_ids)
        self.n_claws = n_claws_per_kc(self.W).reindex(self.kc_ids)
        self.factor = kc_input_thr_factor(self.W, self.kc_ids, self.a) if self.a != 0.0 else None
        t0 = time.time()
        store: dict = {}
        with E43.captured_olfsysm_vars(store):
            self._sim, _ = make_fixed_param_kc_simulator(self.orn, initial_params=self.p, **self.kws, **self.kc_extra)
        self.spont_in = np.array(store['rv'].kc.spont_in, dtype=float).ravel()
        self.build_seconds = time.time() - t0
        self.n_sims = 0
        self._checked = False

    def sim(self, params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.n_sims += 1
        r, s = self._sim(params) if self.factor is None else self._sim(params, kc_thr_factor=self.factor)
        if not self._checked:
            # the factor is in the simulator's KC order (that of the breadth run)
            if not pd.Index(r.index.get_level_values(H.KC_ID_LEVEL)).equals(self.kc_ids):
                raise RuntimeError(f'{self.key}: the model\'s KC order differs from the breadth run\'s')
            self._checked = True
        order = sorted(r.columns, key=lambda c: ODORS.index(H.odor_abbrev(c)))
        return r[order], s[order]

    def off_params(self, p: Optional[dict] = None) -> dict:
        return dict(p or self.p, wAPLKC_scale=0.0, wAPLPN_scale=0.0)

    def thresholds(self, fixed_thr: float) -> np.ndarray:
        """Each KC's threshold as the simulator sets it: (fixed_thr + 2 spont_in) x the factor."""
        t = fixed_thr + self.spont_in * 2.0
        return t if self.factor is None else t * self.factor


# ---------------------------------------------------------------------------
# scoring
_DATA: Dict[str, object] = {}


def data_refs() -> dict:
    """The KC data's values this experiment reports beside the models': the squared CV of the number of odors
    per KC per recording (responders at trial mean > 1, as the odors-per-KC band), the binarized mean distances."""
    if not _DATA:
        for g, cond in (('TNT3c', 'on'), ('TNT', 'off')):
            v = []
            for _, m in H.yang_recording_matrices(g).items():
                n = (H.abbrev_columns(m)[ODORS].to_numpy(dtype=float) > 1).sum(axis=1)
                v.append(float(n.var() / n.mean() ** 2))
            _DATA[f'opk_cv2_{cond}'] = dict(mean=float(np.mean(v)), min=float(min(v)), max=float(max(v)), n=len(v))
            _DATA[f'mean_dist_bin_{cond}'] = float(H.yang_mean_rdm(g, 'cosine', binarize=1.0).mean())
            _DATA[f'mean_dist_raw_{cond}'] = float(H.yang_mean_rdm(g, 'cosine').mean())
    return _DATA


def survivors_of(model_key: str) -> pd.Index:
    """The base's survivors (E70, the real wiring), by KC id."""
    s = pd.read_csv(RESULTS / 'E70_high_firing_kcs' / 'survivors.csv')
    s = s[(s.model == model_key) & (s.wiring == 'real')]
    return pd.Index(s.kc_id.astype(np.int64).unique())


def release_stats(tag: str, model: Model, r_on: pd.DataFrame, r_off: pd.DataFrame) -> Tuple[dict, List[dict]]:
    """Who is released: responding entries with APL on and off by claw-count group, each group's share of the
    responses blocking APL adds, the mean odors per KC; ρ of W and of the claw count with the odors per KC."""
    on = r_on.to_numpy(dtype=float) > 0.5
    off = r_off.to_numpy(dtype=float) > 0.5
    n_on, n_off = on.sum(axis=1), off.sum(axis=1)
    added = float(n_off.sum() - n_on.sum())
    claws = model.n_claws.to_numpy()
    W = model.W_in.to_numpy()
    out, rows = {}, []
    for lab, lo, hi in CLAW_GROUPS:
        m = (claws >= lo) & (claws <= hi)
        g = dict(tag=tag, group=lab, n_kcs=int(m.sum()), mean_claws=float(claws[m].mean()), mean_W=float(W[m].mean()),
            n_resp_on=int(n_on[m].sum()), n_resp_off=int(n_off[m].sum()),
            share_added=float((n_off[m].sum() - n_on[m].sum()) / added) if added else np.nan,
            opk_on=float(n_on[m].mean()), opk_off=float(n_off[m].mean()),
            silent_on=float((n_on[m] == 0).mean()), silent_off=float((n_off[m] == 0).mean()))
        rows.append(g)
        s = 'g' + lab.replace('-', '_').replace('+', 'plus')
        out.update({f'rel_{s}_{k}': g[k] for k in ('n_kcs', 'n_resp_on', 'n_resp_off', 'share_added', 'opk_on', 'opk_off')})
    for cond, n in (('on', n_on), ('off', n_off)):
        out[f'rho_W_opk_{cond}'] = float(sstats.spearmanr(W, n)[0])
        out[f'rho_claws_opk_{cond}'] = float(sstats.spearmanr(claws, n)[0])
        out[f'opk_cv2_{cond}'] = float(n.var() / n.mean() ** 2) if n.mean() > 0 else np.nan
        out[f'opk_mean_{cond}'] = float(n.mean())
    out['n_added_responses'] = added
    return out, rows


def survivor_stats(model: Model, on: pd.DataFrame, off: pd.DataFrame) -> dict:
    """E70's high-firing set and survivors of this variant, and how many of the base's survivors remain."""
    import E70_high_firing_kcs as E70
    cls = E70.classify_kcs(on, off, HIGH_SPIKES)
    ids = pd.Index(on.index.get_level_values(H.KC_ID_LEVEL))
    base_surv = survivors_of(model.base['survivors_model'])
    c = pd.Series(cls.to_numpy(), index=ids)
    bs = c.reindex(base_surv)
    return dict(n_high_firing=int(cls.isin(['survivor', 'high_firing_other']).sum()), n_survivors=int((cls == 'survivor').sum()),
        base_survivors_model=model.base['survivors_model'], n_base_survivors=int(len(base_surv)),
        base_survivors_remaining=int((bs == 'survivor').sum()), base_survivors_high_firing=int(bs.isin(['survivor', 'high_firing_other']).sum()),
        base_survivors_max_on=float(on.max(axis=1).set_axis(ids).reindex(base_surv).mean()) if len(base_surv) else np.nan,
        base_survivors_max_off=float(off.max(axis=1).set_axis(ids).reindex(base_surv).mean()) if len(base_surv) else np.nan)


def threshold_stats(model: Model, fixed_thr: float) -> dict:
    """The KC thresholds as simulated: mean, CV, the share of the spontaneous-input term in the factor-free
    threshold, and how the threshold scales with W (log-log slope, ρ)."""
    t = model.thresholds(fixed_thr)
    t0 = fixed_thr + model.spont_in * 2.0
    W = model.W_in.to_numpy()
    lw = np.log(W / W.mean())
    return dict(thr_mean=float(t.mean()), thr_cv=float(t.std() / t.mean()), thr_min=float(t.min()), thr_max=float(t.max()),
        spont_share_mean=float(np.mean(2.0 * model.spont_in / t0)), spont_in_mean=float(model.spont_in.mean()),
        thr_loglog_slope_W=float(np.polyfit(lw, np.log(t), 1)[0]), thr_rho_W=float(sstats.spearmanr(W, t)[0]),
        W_mean=float(W.mean()), W_cv=float(W.std() / W.mean()))


def score(tag: str, meta: dict, model: Model, on_out, off_out) -> dict:
    """Every readout of one variant: E66's `score_variant` (the harness stats columns, the loss decomposition
    under each readout, the counts saved), E69's readouts and this experiment's; appended to
    input_readouts.csv and released_by_group.csv."""
    import E66_per_odor_apl_profile as E66
    from E10_joint_fit import apl_effect_terms
    from apl_effect_scores import realism_flag
    c = E69.ctx()
    on_s, off_s = on_out[1].astype(float), off_out[1].astype(float)
    K = H.calibrate_readout(on_s, 'saturating', c['pr_target'], stat='participation_ratio', responders_min=RESP_MIN)
    row = E66.score_variant(c, tag, meta, on_out, off_out, model.W, model.lab_orn, K, 'A', res_dir(), out_dir())
    ex = dict(tag=tag, cal_K=K)
    ae = apl_effect_terms(H.pair_distances(on_s, 'cosine'), H.pair_distances(off_s, 'cosine'), on_s, off_s, c['effect'],
        c['pr_target'])
    ex.update(ae)
    b_on, b_off = H.readout_pair_distances(on_s, 'binary', RESP_MIN), H.readout_pair_distances(off_s, 'binary', RESP_MIN)
    dec_b, _ = H.apl_effect_decomposition(b_on, b_off, c['effect_binary'])
    ex.update({f'apl_bin_{k[len("apl_effect_"):]}': v for k, v in dec_b.items() if k.startswith('apl_effect_')})
    ex.update(mean_dist_raw_on=float(row[f'{V}__on_mean_model']), mean_dist_raw_off=float(row[f'{V}__off_mean_model']),
        mean_dist_bin_on=float(b_on.mean()), mean_dist_bin_off=float(b_off.mean()))
    for cond, m in (('on', on_s), ('off', off_s)):
        ex.update(E69.count_stats(m, K, cond))
    ex.update(E69.amplitude_z_stats(on_s, off_s, K))
    ex.update(E69.opk_stats(on_out[0], off_out[0]))
    rel, rel_rows = release_stats(tag, model, on_out[0], off_out[0])
    ex.update(rel)
    ex.update(survivor_stats(model, on_s, off_s))
    ex.update(threshold_stats(model, float(meta['fixed_thr'])))
    ex['realism_flag'] = realism_flag(row['rate_on'], row['rate_off'], row.get('spearman_rate_on_vs_data'),
        ex['opk_on_tv'], ex['opk_off_tv'])
    for name, df in (('input_readouts.csv', pd.DataFrame([ex])), ('released_by_group.csv', pd.DataFrame(rel_rows))):
        p = res_dir() / name
        df.to_csv(p, mode='a', header=not p.exists(), index=False)
    print(f"{tag}: odors per KC {ex['opk_on_tv']:.3f} / {ex['opk_off_tv']:.3f} (silent off {ex['opk_off_0']:.3f}, 8+ off "
        f"{ex['opk_off_8plus']:.3f}, CV2 {ex['opk_cv2_on']:.2f} / {ex['opk_cv2_off']:.2f}); 9+ claws' share of the added "
        f"responses {ex['rel_g9plus_share_added']:.2f}; binarized mean distance off {ex['mean_dist_bin_off']:.3f}, effect "
        f"{ex['apl_bin_mean']:+.3f} (loss {ex['apl_bin_rmse']:.3f}); loss raw {ae['apl_raw_rmse']:.3f} (mean {ae['apl_raw_mean']:+.3f}), "
        f"calibrated {ae['apl_cal_rmse']:.3f} (mean {ae['apl_cal_mean']:+.3f}, K {K:.2f}); survivors {ex['n_survivors']} (base's remaining "
        f"{ex['base_survivors_remaining']} of {ex['n_base_survivors']}); flag: {ex['realism_flag'] or '-'}", flush=True)
    return {**row, **ex}


def done_tags() -> set:
    p = res_dir() / 'input_readouts.csv'
    return set(pd.read_csv(p)['tag'].astype(str)) if p.exists() else set()


def base_meta(model: Model) -> dict:
    b, info, fit = model.base, model.info, model.fit
    return dict(base=model.key, base_label=b['label'], ref_experiment=b['exp'], ref_fit=b['tag'], base_params_from=b.get('params_from'),
        claw_thr=info.get('claw_thr'), gain=info.get('gain'), gain_glomeruli=','.join(info.get('gain_glomeruli') or []),
        k=info.get('k'), kc_kc_tau=info.get('kc_kc_tau'), aplpn_factor_vs_e3_base=float(fit.get('aplpn_factor_vs_e3', np.nan)),
        apl_factor_vs_e3_base=float(fit.get('apl_factor_vs_e3', np.nan)), n_glomeruli=int(model.W.shape[1]),
        kc_refractory_period=model.kc.get('kc_refractory_period', 0.0), kc_adapt_inc=model.kc.get('kc_adapt_inc', 0.0),
        kc_adapt_tau=model.kc.get('kc_adapt_tau', np.nan), kc_thr_input_exp=model.a, breadth_kept=model.breadth_mode == 'kept',
        prepass_thr_factor=model.prepass.get('thr_factor', np.nan), prepass_fixed_thr=model.prepass.get('fixed_thr', np.nan),
        prepass_n_responses=model.prepass.get('n_responses', np.nan), prepass_target_n_responses=model.prepass.get('target_n_responses', np.nan),
        factor_cv=float(model.factor.std() / model.factor.mean()) if model.factor is not None else 0.0,
        factor_min=float(model.factor.min()) if model.factor is not None else 1.0,
        factor_max=float(model.factor.max()) if model.factor is not None else 1.0)


def params_meta(p: dict) -> dict:
    return dict(fixed_thr=p['fixed_thr'], wAPLKC_scale=p['wAPLKC_scale'], wKCAPL_scale=p['wKCAPL_scale'],
        wAPLPN_scale=p['wAPLPN_scale'], wPNAPL_scale=p['wPNAPL_scale'], wKCKC_scale=p['wKCKC_scale'], kc_kc_inh=p['wKCKC_scale'])


# ---------------------------------------------------------------------------
# settings
def setting_tag(base: str, a: float, breadth: str = 'recomputed') -> str:
    return f'{base}_base' if a == 0 else f'{base}_a{a:g}' + ('_kb' if breadth == 'kept' else '')


def c2ceil_exponents() -> Tuple[List[float], Optional[float], str]:
    """The firing-ceiling base's exponents: 0.5 and 1, with the neighbours of c2's chosen exponent on the
    0.25 grid (above 0, at most 1.5) and that exponent itself (when it is neither 0 nor already in);
    (exponents, c2's choice, status). Before c2's settings are all scored, 0.5 and 1 only."""
    fixed = list(C2CEIL_FIXED)
    st = exponent_choice('c2')
    if st['status'] != 'chosen':
        return fixed, None, st['status']
    best = float(st['exponent'])
    extra = [best - GRID_STEP, best, best + GRID_STEP]
    out = sorted({round(x, 6) for x in fixed + [x for x in extra if 0 < x <= 1.5]})
    return out, best, 'chosen'


def settings() -> List[dict]:
    """Every part-A setting: base, exponent, breadth mode, tag."""
    out = [dict(base='c2', a=a, breadth='recomputed') for a in EXPONENTS['c2']]
    out += [dict(base='c2', a=a, breadth='kept') for a in KEPT_BREADTH['c2']]
    out += [dict(base='ref40', a=a, breadth='recomputed') for a in EXPONENTS['ref40']]
    out += [dict(base='c2ceil', a=a, breadth='recomputed') for a in c2ceil_exponents()[0]]
    for x in out:
        x['tag'] = setting_tag(x['base'], x['a'], x['breadth'])
    return out


# ---------------------------------------------------------------------------
# stages
def stage_check(key: str) -> None:
    """Re-runs base `key` (no factor) and checks it reproduces its archived spike counts exactly, and its
    fits.csv rates where it is a fit at its fitted parameters; scores it as variant <key>_base."""
    t0 = time.time()
    m = Model(key, 0.0)
    on, off = m.sim(m.p), m.sim(m.off_params())
    checks = {}
    arch_folder, arch_tag = m.base['archive'].split('/', 1)
    for cond, out in (('on', on), ('acute_off', off)):
        p = ARCHIVE / arch_folder / f'{arch_tag}_{cond}_spike_counts.parquet'
        a = H.abbrev_columns(pd.read_parquet(p).astype(float))[ODORS]
        checks[f'{cond}_archive_index_equal'] = bool(a.index.equals(out[1].index))
        checks[f'{cond}_archive_mismatch'] = int((a.to_numpy() != out[1].to_numpy(dtype=float)).sum())
    if not m.base.get('params_from'):
        checks['rate_on_vs_fit'] = rate(on[0]) - float(m.fit['rate_on'])
        checks['rate_off_vs_fit'] = rate(off[0]) - float(m.fit['rate_off'])
    bad = [k for k, v in checks.items() if (k.endswith('mismatch') and v != 0) or (k.endswith('index_equal') and not v)
        or (k.startswith('rate_') and abs(v) > 1e-12)]
    if bad:
        raise RuntimeError(f'{key}: the re-simulated base does not reproduce its archived outputs: {checks}')
    meta = dict(variant='base', tag=f'{key}_base', part='A', setting='base', **base_meta(m), thr_factor=1.0, apl_factor=1.0,
        target_rate_on=rate(on[0]), target_rate_off=rate(off[0]), rate_on=rate(on[0]), rate_off=rate(off[0]),
        thr_at_bound=False, apl_at_bound=False, aplpn_factor_vs_e3=m.p['wAPLPN_scale'] / m.initial['wAPLPN_scale'],
        aplpn_over_cap=bool(m.p['wAPLPN_scale'] / m.initial['wAPLPN_scale'] > APLPN_CAP + 1e-9), n_sims=m.n_sims, seconds=np.nan, **params_meta(m.p))
    meta['n_kcs'] = int(on[0].shape[0])
    meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
    meta['seconds'] = time.time() - t0
    if f'{key}_base' not in done_tags():
        score(f'{key}_base', meta, m, on, off)
    p = res_dir() / 'base_checks.json'
    allc = read_json(p, {})
    allc[key] = dict(checks=checks, base=m.base, model_info=m.info, params=m.p, kc_spiking=m.kc, rate_on=rate(on[0]),
        rate_off=rate(off[0]), kws_scalar={k: v for k, v in m.kws.items() if not k.startswith('_')},
        n_responses_on=n_responses(on[0]), n_responses_off=n_responses(off[0]), size=int(on[0].size),
        max_count_on=float(on[1].to_numpy().max()), max_count_off=float(off[1].to_numpy().max()),
        thresholds=threshold_stats(m, float(m.p['fixed_thr'])), build_seconds=m.build_seconds, seconds=time.time() - t0,
        time=now(), host=socket.gethostname(), **code_state())
    write_json(p, allc)
    print(f'{key}: checks passed {checks}', flush=True)


def stage_setting(tag: str) -> None:
    """One setting: the factor (and the breadth recomputed with it), the threshold factor bisected on the
    base's APL-off response count, then one factor on the APL->KC and APL->bouton scales on its APL-on
    count; scored."""
    t0 = time.time()
    x = next((s for s in settings() if s['tag'] == tag), None)
    if x is None:
        raise ValueError(f'{tag}: not a setting ({[s["tag"] for s in settings()]})')
    if tag in done_tags():
        print(f'{tag}: done', flush=True)
        return
    checks = read_json(res_dir() / 'base_checks.json', {})
    if x['base'] not in checks:
        raise RuntimeError(f"run --stage check --base {x['base']} first")
    bc = checks[x['base']]
    m = Model(x['base'], x['a'], x['breadth'])
    p0, size = dict(m.p), int(bc['size'])
    thr0 = p0['fixed_thr']
    f, off_out, h1, thr_bound = E69.bisect_factor(lambda f: m.sim(m.off_params(dict(p0, fixed_thr=thr0 * f))),
        int(bc['n_responses_off']), size, THR_BRACKET, THR_STEP, f'{tag} threshold factor (APL off)')
    p1 = dict(p0, fixed_thr=thr0 * f)
    g, on_out, h2, apl_bound = E69.bisect_factor(lambda g: m.sim(dict(p1, wAPLKC_scale=p0['wAPLKC_scale'] * g,
        wAPLPN_scale=p0['wAPLPN_scale'] * g)), int(bc['n_responses_on']), size, APL_BRACKET, APL_STEP,
        f'{tag} APL factor (APL on)')
    p = dict(p1, wAPLKC_scale=p0['wAPLKC_scale'] * g, wAPLPN_scale=p0['wAPLPN_scale'] * g)
    hist = pd.DataFrame(h1 + h2).assign(tag=tag)
    hp = res_dir() / 'bisection_history.csv'
    hist.to_csv(hp, mode='a', header=not hp.exists(), index=False)
    # (as E10_joint_fit's aplpn_factor_vs_e3: relative to the bouton base's tuned APL->bouton scale)
    aplpn_e3 = p['wAPLPN_scale'] / m.initial['wAPLPN_scale']
    meta = dict(variant=tag[len(x['base']) + 1:], tag=tag, part='A', setting=tag, **base_meta(m), thr_factor=f, apl_factor=g,
        target_rate_on=bc['rate_on'], target_rate_off=bc['rate_off'], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]),
        thr_at_bound=thr_bound, apl_at_bound=apl_bound, aplpn_factor_vs_e3=aplpn_e3, aplpn_over_cap=bool(aplpn_e3 > APLPN_CAP),
        n_sims=m.n_sims, seconds=np.nan, **params_meta(p))
    meta['n_kcs'] = int(on_out[0].shape[0])
    meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
    meta['seconds'] = time.time() - t0
    score(tag, meta, m, on_out, off_out)
    lp = res_dir() / 'settings_log.csv'
    pd.DataFrame([dict(tag=tag, base=x['base'], kc_thr_input_exp=x['a'], breadth=x['breadth'], thr_factor=f, apl_factor=g,
        n_responses_on=n_responses(on_out[0]), n_responses_off=n_responses(off_out[0]), target_n_on=bc['n_responses_on'],
        target_n_off=bc['n_responses_off'], prepass_thr_factor=m.prepass.get('thr_factor', np.nan),
        prepass_seconds=m.prepass.get('seconds', np.nan), n_sims=m.n_sims, build_seconds=m.build_seconds,
        seconds=time.time() - t0, max_rss_gb=max_rss_gb(), time=now(), host=socket.gethostname(), **code_state())]).to_csv(
        lp, mode='a', header=not lp.exists(), index=False)


def max_rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def child(args: List[str], log: Path) -> int:
    """Runs this script with `args` in its own process, output to `log` (the --local-test folder passed on)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    extra = ['--local-test', str(_DIRS['root'])] if 'root' in _DIRS else []
    with open(log, 'a') as fh:
        fh.write(f'\n# {now()} {" ".join(args + extra)}\n')
        fh.flush()
        # (-u: Python 3.14 holds a redirected log's output in 128 KB blocks otherwise)
        r = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *args, *extra], stdout=fh, stderr=subprocess.STDOUT,
            cwd=H.HERE)
    print(f"{' '.join(args)}: exit {r.returncode} (log {log.name})", flush=True)
    return r.returncode


def stage_run(only_bases: Optional[List[str]] = None) -> None:
    """Every missing base check, then every missing setting (c2's first, so the firing-ceiling base's
    exponents can follow its choice), each in its own process, then the summary."""
    logs = out_dir() / 'logs'
    resolve_bases()
    keys = [k for k in BASE_ORDER if only_bases is None or k in only_bases]
    failed = []
    for key in keys:
        if key not in read_json(res_dir() / 'base_checks.json', {}):
            if child(['--stage', 'check', '--base', key], logs / f'check_{key}.log'):
                failed.append(f'check {key}')
    for key in keys:
        checked = read_json(res_dir() / 'base_checks.json', {})
        # (settings() again per base: c2ceil's exponents depend on c2's scored settings)
        for s in [s for s in settings() if s['base'] == key]:
            if s['tag'] in done_tags() or s['base'] not in checked:
                continue
            if child(['--stage', 'setting', '--tag', s['tag']], logs / f"{s['tag']}.log"):
                failed.append(s['tag'])
    write_json(res_dir() / 'run_failures.json', dict(time=now(), failed=failed))
    stage_summarize()
    if failed:
        print(f'FAILED: {failed}', flush=True)


# ---------------------------------------------------------------------------
# the exponent rule
def merged() -> pd.DataFrame:
    st = pd.read_csv(res_dir() / 'partA_variant_stats.csv').drop_duplicates('tag', keep='last').set_index('tag')
    ex = pd.read_csv(res_dir() / 'input_readouts.csv').drop_duplicates('tag', keep='last').set_index('tag')
    df = st.join(ex[[c for c in ex.columns if c not in st.columns]], how='inner')
    order = {t: i for i, t in enumerate([f'{k}_base' for k in BASE_ORDER] + [s['tag'] for s in settings_static()])}
    df['_o'] = [order.get(t, 999) for t in df.index]
    return df.sort_values(['_o']).drop(columns='_o').reset_index()


def settings_static() -> List[dict]:
    """settings() without resolving c2ceil's neighbours (no recursion through the exponent rule)."""
    out = [dict(base='c2', a=a, breadth='recomputed') for a in EXPONENTS['c2']]
    out += [dict(base='c2', a=a, breadth='kept') for a in KEPT_BREADTH['c2']]
    out += [dict(base='ref40', a=a, breadth='recomputed') for a in EXPONENTS['ref40']]
    out += [dict(base='c2ceil', a=a, breadth='recomputed') for a in sorted(set(C2CEIL_FIXED) | {0.25, 0.75, 1.25, 1.5})]
    for x in out:
        x['tag'] = setting_tag(x['base'], x['a'], x['breadth'])
    return out


def admissible(r) -> Tuple[bool, str]:
    """Whether a row may be chosen: both response rates held, the APL-on odors-per-KC distance inside its band,
    the APL-on response rate ρ at or above the band's floor; with the reason when not."""
    from apl_effect_scores import opk_band, RATE_RHO_MIN
    why = []
    if bool(r.get('thr_at_bound')) or bool(r.get('apl_at_bound')):
        why.append('a bisection at its bound')
    if not float(r['opk_on_tv']) <= opk_band('on'):
        why.append(f"APL-on odors per KC {float(r['opk_on_tv']):.3f} outside the band {opk_band('on'):.3f}")
    if not float(r['spearman_rate_on_vs_data']) >= RATE_RHO_MIN:
        why.append(f"APL-on response rate ρ {float(r['spearman_rate_on_vs_data']):.2f} below {RATE_RHO_MIN:.2f}")
    return not why, '; '.join(why)


def exponent_choice(base: str, df: Optional[pd.DataFrame] = None) -> dict:
    """The exponent rule on `base`'s scored rows (the base and its settings with the breadth recomputed):
    {'status': 'chosen' / 'pending' / 'none admissible', 'exponent', 'tag', 'candidates': [...]}; 'pending'
    until the base and every one of its fixed-list settings are scored."""
    if df is None:
        if not ((res_dir() / 'partA_variant_stats.csv').exists() and (res_dir() / 'input_readouts.csv').exists()):
            return dict(status='pending', exponent=None, tag=None, candidates=[])
        df = merged()
    need = [f'{base}_base'] + [setting_tag(base, a) for a in EXPONENTS.get(base, C2CEIL_FIXED)]
    have = set(df.tag)
    if not set(need) <= have:
        return dict(status='pending', exponent=None, tag=None, candidates=[], missing=sorted(set(need) - have))
    g = df[(df.base == base) & (~df.breadth_kept.astype(bool))]
    cands = []
    for _, r in g.iterrows():
        ok, why = admissible(r)
        cands.append(dict(tag=r.tag, a=float(r.kc_thr_input_exp), opk_off_tv=float(r.opk_off_tv), opk_on_tv=float(r.opk_on_tv),
            rho_on=float(r.spearman_rate_on_vs_data), admissible=ok, why=why))
    ok = [c for c in cands if c['admissible']]
    if not ok:
        return dict(status='none admissible', exponent=None, tag=None, candidates=cands)
    best = min(c['opk_off_tv'] for c in ok)
    near = [c for c in ok if c['opk_off_tv'] <= best + TIE_MARGIN]
    pick = min(near, key=lambda c: (abs(c['a']), c['a']))
    return dict(status='chosen', exponent=pick['a'], tag=pick['tag'], best_opk_off_tv=best, candidates=cands)


# ---------------------------------------------------------------------------
# summary: merged tables, comparison table, figures, provenance
def fmt(x, spec='.3f') -> str:
    return E69.fmt(x, spec)


def pct(x, spec='.1f') -> str:
    return E69.pct(x, spec)


def setting_label(r) -> str:
    if float(r.kc_thr_input_exp) == 0:
        return 'base (a = 0)'
    return f'a = {float(r.kc_thr_input_exp):g}' + (' (breadth kept)' if bool(r.breadth_kept) else '')


def write_table(df: pd.DataFrame, choices: Dict[str, dict]) -> None:
    from apl_effect_scores import opk_band
    dt, dr = H.data_target, data_refs()
    bs = bases()
    L = ['# E71: the input-scaled KC threshold, part A (fixed scales, both response rates held)', '',
        f'Generated {datetime.now():%Y-%m-%d %H:%M} by E71_input_scaled_threshold.py. Each KC\'s threshold (fixed_thr + 2 x its '
        'spontaneous input) x (W / mean W)^a, W its summed PN->claw weight (gain included). Each setting re-tuned: the threshold factor '
        'on the base\'s APL-off response count, then one factor on the APL->KC and APL->bouton scales on its APL-on count; the breadth '
        'behind the APL->KC weights recomputed with the same factor (pre-pass threshold re-tuned to its own APL-off count) unless '
        '"breadth kept". The APL-effect loss: root mean squared difference of the per-pair APL effect from the data\'s over the 120 pairs, '
        'lower is better; "cal" under the calibrated saturating readout (K re-calibrated on each variant\'s APL-on counts); "bin" binarized '
        '(2 or more spikes) against the binarized data. Bases: ' + '; '.join(f"{k}: {b['label']} ({b['exp']}, {b['tag']}"
        + (f", parameters {b['params_from']}" if b.get('params_from') else '') + ')' for k, b in bs.items()) + '.', '',
        'Exponent rule: among the base and its settings with the breadth recomputed whose bisections held both rates, those with the '
        f'APL-on odors-per-KC distance inside its band ({opk_band("on"):.3f}) and the APL-on response rate ρ at or above 0.37; the '
        f'smallest APL-off distance, and within {TIE_MARGIN} of it the exponent nearest 0. Choices: ' + '; '.join(
        f"{k}: {c['status']}" + (f" a = {c['exponent']:g}" if c.get('exponent') is not None else '') for k, c in choices.items()) + '.', '']
    for key, g in df.groupby('base', sort=False):
        ch = choices.get(key, {})
        mark = lambda r: ' (chosen)' if ch.get('tag') == r.tag else ''
        L += [f"## {bs[key]['label']} ({key})", '', '### Recruitment and realism', '',
            '| setting | odors per KC, distance<br>APL on / off | silent KCs<br>APL on / off | 8+ odors<br>APL on / off | '
            'CV² of odors per KC<br>APL on / off | 9+ claws: share of<br>added responses | 1-4 claws: share of<br>added responses | '
            'ρ(W, odors per KC)<br>APL on / off | high-firing / survivors<br>(base\'s remaining) | largest count<br>APL on / off | '
            'rates on / off<br>(ratio) | response rate ρ<br>APL on / off | threshold /<br>APL factor | pre-pass<br>threshold factor | '
            'APL->bouton<br>over cap | realism flag |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
        for r in g.itertuples():
            L.append(f"| {setting_label(r)}{mark(r)} | {fmt(r.opk_on_tv, '.3f')} / {fmt(r.opk_off_tv, '.3f')} | {pct(r.opk_on_0)} / "
                f"{pct(r.opk_off_0)} | {pct(r.opk_on_8plus)} / {pct(r.opk_off_8plus)} | {fmt(r.opk_cv2_on, '.2f')} / {fmt(r.opk_cv2_off, '.2f')} | "
                f"{pct(r.rel_g9plus_share_added, '.0f')} | {pct(r.rel_g1_4_share_added, '.0f')} | {fmt(r.rho_W_opk_on, '.2f')} / "
                f"{fmt(r.rho_W_opk_off, '.2f')} | {r.n_high_firing} / {r.n_survivors} ({r.base_survivors_remaining} of {r.n_base_survivors}) | "
                f"{r.max_count_on:.0f} / {r.max_count_off:.0f} | {r.rate_on:.4f} / {r.rate_off:.4f} ({r.rate_off / r.rate_on:.2f}) | "
                f"{fmt(r.spearman_rate_on_vs_data, '.2f')} / {fmt(r.spearman_rate_off_vs_data, '.2f')} | {fmt(r.thr_factor, '.4f')} / "
                f"{fmt(r.apl_factor, '.3f')} | {fmt(r.prepass_thr_factor, '.4f')} | {'yes' if bool(r.aplpn_over_cap) else '-'} | "
                f"{r.realism_flag if isinstance(r.realism_flag, str) and r.realism_flag else '-'} |")
        L.append(f"| KC data | bands {opk_band('on'):.3f} / {opk_band('off'):.3f} | {pct(dt('ctrl_odors_per_kc_0'))} / "
            f"{pct(dt('tnt_odors_per_kc_0'))} | {pct(dt('ctrl_odors_per_kc_8plus'))} / {pct(dt('tnt_odors_per_kc_8plus'))} | "
            f"{dr['opk_cv2_on']['mean']:.2f} ({dr['opk_cv2_on']['min']:.2f}-{dr['opk_cv2_on']['max']:.2f}) / {dr['opk_cv2_off']['mean']:.2f} "
            f"({dr['opk_cv2_off']['min']:.2f}-{dr['opk_cv2_off']['max']:.2f}) | - | - | - | - | - | {dt('ctrl_frac'):.4f} / {dt('tnt_frac'):.4f} "
            f"({dt('rate_ratio_fraction'):.2f}) | - | - | - | - | - |")
        L += ['', '### The APL effect', '',
            '| setting | APL-effect loss, raw<br>(mean; per-odor; pair-specific) | ρ per-pair<br>raw | APL-effect loss, cal<br>'
            '(mean; per-odor; pair-specific) | ρ per-pair<br>cal | K | APL-effect loss, bin<br>(mean; per-odor; pair-specific) | '
            'mean distance, raw<br>APL on / off | mean distance, cal<br>APL on / off | mean distance, bin<br>APL on / off | block<br>raw / cal |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in g.itertuples():
            L.append(f"| {setting_label(r)}{mark(r)} | {fmt(r.apl_raw_rmse)} ({r.apl_raw_mean:+.3f}; {fmt(r.apl_raw_per_odor)}; "
                f"{fmt(r.apl_raw_pair_specific)}) | {fmt(r.apl_raw_rho, '.2f')} | {fmt(r.apl_cal_rmse)} ({r.apl_cal_mean:+.3f}; "
                f"{fmt(r.apl_cal_per_odor)}; {fmt(r.apl_cal_pair_specific)}) | {fmt(r.apl_cal_rho, '.2f')} | {fmt(r.cal_K, '.2f')} | "
                f"{fmt(r.apl_bin_rmse)} ({r.apl_bin_mean:+.3f}; {fmt(r.apl_bin_per_odor)}; {fmt(r.apl_bin_pair_specific)}) | "
                f"{fmt(r.mean_dist_raw_on)} / {fmt(r.mean_dist_raw_off)} | {fmt(r.mean_dist_cal_on)} / {fmt(r.mean_dist_cal_off)} | "
                f"{fmt(r.mean_dist_bin_on)} / {fmt(r.mean_dist_bin_off)} | {fmt(r._asdict()[f'{V}__block_fruity_small_iso'])} / "
                f"{fmt(r.block_cal)} |")
        L.append(f"| KC data | floor {fmt(dt('apl_effect_rmse_floor'))} (+{dt('apl_effect_mean'):.3f}) | ceiling "
            f"{fmt(dt('apl_effect_rho_ceiling'), '.2f')} | floor {fmt(dt('apl_effect_rmse_floor'))} (+{dt('apl_effect_mean'):.3f}) | - | - | "
            f"floor {fmt(dt('apl_effect_binary_rmse_floor'))} (+{dt('apl_effect_binary_mean'):.3f}) | {dr['mean_dist_raw_on']:.3f} / "
            f"{dr['mean_dist_raw_off']:.3f} | {dr['mean_dist_raw_on']:.3f} / {dr['mean_dist_raw_off']:.3f} | {dr['mean_dist_bin_on']:.3f} / "
            f"{dr['mean_dist_bin_off']:.3f} | {fmt(dt('block'))} / - |")
        L += ['', '### Odor-odor distances, amplitudes, thresholds', '',
            '| setting | ρ on odor-odor distances<br>APL on / off | natural-source alignment<br>APL on / off / on minus off | '
            'APL off: above 4 z /<br>responders\' mean z | APL on: above 4 z /<br>responders\' mean z | 0.5-1.5 z band<br>APL on / off | '
            'threshold CV | share of the<br>spontaneous term | threshold vs W<br>log-log slope / ρ |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in g.itertuples():
            d = r._asdict()
            von, voff = d[f'{V}__on_spearman_resid_vs_vcf_hyp'], d[f'{V}__off_spearman_resid_vs_vcf_hyp']
            L.append(f"| {setting_label(r)}{mark(r)} | {fmt(d[f'{V}__on_spearman'])} / {fmt(d[f'{V}__off_spearman'])} | {fmt(von, '.2f')} / "
                f"{fmt(voff, '.2f')} / {fmt(von - voff, '+.2f')} | {pct(r.z_floor_noise_share_above_4_acute_off)} / "
                f"{fmt(r.z_floor_noise_resp_amp_mean_acute_off, '.2f')} | {pct(r.z_floor_noise_share_above_4_on)} / "
                f"{fmt(r.z_floor_noise_resp_amp_mean_on, '.2f')} | {pct(r.z_floor_noise_share_05_15_on)} / "
                f"{pct(r.z_floor_noise_share_05_15_acute_off)} | {fmt(r.thr_cv, '.2f')} | {pct(r.spont_share_mean)} | "
                f"{fmt(r.thr_loglog_slope_W, '.2f')} / {fmt(r.thr_rho_W, '.2f')} |")
        L.append(f"| KC data | ceilings 0.98 / 0.97 | {dt('ctrl_vcf'):.2f} / {dt('tnt_vcf'):.2f} / {dt('vcf_apl_effect'):+.2f} | "
            f"{pct(dt('tnt_share_z_above_4'))} / {fmt(dt('tnt_resp_amp_mean_z'), '.2f')} | {pct(dt('ctrl_share_z_above_4'))} / "
            f"{dt('ctrl_resp_amp_mean_z'):.2f} | {pct(dt('ctrl_share_z_05_15'))} / {pct(dt('tnt_share_z_05_15'))} | - | - | - |")
        L.append('')
    rel = pd.read_csv(res_dir() / 'released_by_group.csv').drop_duplicates(['tag', 'group'], keep='last')
    L += ['## Who is released: responding KC x odor entries by claw-count group (APL on -> APL off; share of the added responses)', '',
        '| variant | ' + ' | '.join(f'{lab} claws' for lab, _, _ in CLAW_GROUPS) + ' |', '|---|' + '---:|' * len(CLAW_GROUPS)]
    for tag in df.tag:
        gg = rel[rel.tag == tag].set_index('group')
        if len(gg) != len(CLAW_GROUPS):
            continue
        L.append(f'| {tag} | ' + ' | '.join(f"{int(gg.loc[lab, 'n_resp_on'])} -> {int(gg.loc[lab, 'n_resp_off'])} "
            f"({100 * gg.loc[lab, 'share_added']:.0f} %)" for lab, _, _ in CLAW_GROUPS) + ' |')
    (res_dir() / 'comparison_table.md').write_text('\n'.join(L) + '\n')


def figures(df: pd.DataFrame, choices: Dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from apl_effect_scores import opk_band
    dt, dr = H.data_target, data_refs()
    colors = {'c2': '#1baf7a', 'c2ceil': '#6250d6', 'ref40': '#2a78d6'}
    bs = bases()
    panels = [('opk_off_tv', 'odors per KC, APL off: distance from the flies', opk_band('off'), 'realism band'),
        ('opk_on_tv', 'odors per KC, APL on: distance from the flies', opk_band('on'), 'realism band'),
        ('opk_cv2_off', 'APL off: CV² of the number of odors per KC', dr['opk_cv2_off']['mean'], 'APL output blocked (TNT), mean'),
        ('mean_dist_bin_off', 'APL off: binarized mean pair distance', dr['mean_dist_bin_off'], 'APL output blocked (TNT)'),
        ('apl_bin_mean', 'mean APL effect, binarized', dt('apl_effect_binary_mean'), 'KC data'),
        ('apl_bin_rmse', 'APL-effect loss, binarized', dt('apl_effect_binary_rmse_floor'), 'data floor'),
        ('apl_raw_rmse', 'APL-effect loss, raw counts', dt('apl_effect_rmse_floor'), 'data floor'),
        ('apl_cal_rmse', 'APL-effect loss, calibrated readout', dt('apl_effect_rmse_floor'), 'data floor'),
        ('apl_cal_mean', 'mean APL effect, calibrated readout', dt('apl_effect_mean'), 'KC data'),
        ('apl_raw_mean', 'mean APL effect, raw counts', dt('apl_effect_mean'), 'KC data'),
        ('rel_g9plus_share_added', 'KCs with 9+ claws: share of the responses blocking APL adds', None, None),
        ('spearman_rate_on_vs_data', 'response rate ρ, APL on', 0.37, 'realism floor')]
    fig, axes = plt.subplots(3, 4, figsize=(19, 12))
    for ax, (col, title, ref, ref_lab) in zip(axes.ravel(), panels):
        for key, g in df.groupby('base', sort=False):
            for kept in (False, True):
                gg = g[g.breadth_kept.astype(bool) == kept].sort_values('kc_thr_input_exp')
                if kept:
                    gg = pd.concat([g[g.kc_thr_input_exp == 0], gg])
                if not len(gg):
                    continue
                ax.plot(gg.kc_thr_input_exp, gg[col], '--' if kept else '-', marker='s' if kept else 'o', color=colors.get(key, 'gray'),
                    mfc='none' if kept else colors.get(key, 'gray'), ms=6, lw=1.2, alpha=0.85)
            ch = choices.get(key, {})
            if ch.get('tag') in set(g.tag):
                r = g[g.tag == ch['tag']].iloc[0]
                ax.plot(r.kc_thr_input_exp, r[col], '*', color='k', ms=13, zorder=5)
        if ref is not None:
            realism = ref_lab in ('realism band', 'realism floor')
            ax.axhline(ref, color='crimson' if realism else 'k', ls='--' if realism else ':', lw=1.2)
        ax.axvline(0, color='gray', lw=0.6)
        ax.set_xlabel('exponent a of (W / mean W)^a')
        ax.set_title(title, fontsize=10)
    handles = [plt.Line2D([], [], color=c, marker='o', ls='-', label=bs[k]['label'][:70]) for k, c in colors.items() if k in bs]
    handles += [plt.Line2D([], [], color='gray', marker='s', mfc='none', ls='--', label='breadth kept (control)'),
        plt.Line2D([], [], color='k', marker='*', ls='', ms=11, label='the exponent rule\'s choice'),
        plt.Line2D([], [], color='k', ls=':', label='KC data / floor of the loss'),
        plt.Line2D([], [], color='crimson', ls='--', label='realism band')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('The input-scaled KC threshold: recruitment with APL off and the APL effect against the exponent '
        '(both response rates held)', fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(res_dir() / 'E71_sweep.png', dpi=120)
    plt.close(fig)

    rel = pd.read_csv(res_dir() / 'released_by_group.csv').drop_duplicates(['tag', 'group'], keep='last')
    keys = [k for k in BASE_ORDER if k in set(df.base)]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.4), squeeze=False)
    labs = [lab for lab, _, _ in CLAW_GROUPS]
    for ax, key in zip(axes[0], keys):
        g = df[(df.base == key) & (~df.breadth_kept.astype(bool))].sort_values('kc_thr_input_exp')
        cmap = plt.get_cmap('viridis')
        n = len(g)
        for i, r in enumerate(g.itertuples()):
            gg = rel[rel.tag == r.tag].set_index('group').reindex(labs)
            ax.plot(range(len(labs)), gg.opk_off, '-o', color=cmap(i / max(1, n - 1)), label=f'a = {r.kc_thr_input_exp:g}')
            ax.plot(range(len(labs)), gg.opk_on, ':', color=cmap(i / max(1, n - 1)), lw=1)
        ax.set_xticks(range(len(labs)), [f'{lab} claws' for lab in labs])
        ax.set_ylabel('mean odors per KC (solid: APL off; dotted: APL on)')
        ax.set_title(bs[key]['label'][:60], fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle('Who is released: the number of odors a KC responds to, by its number of claws', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(res_dir() / 'E71_release_by_claws.png', dpi=120)
    plt.close(fig)


def write_tuned_params(df: pd.DataFrame) -> None:
    """tuned_params.json (results and outputs): every variant's absolute parameters, {tag: {param: value}}, the
    form E10_joint_fit.py's --init-params FILE:KEY reads (part B starts from them), with its KC refractory period /
    adaptation, the input-scaled threshold's exponent and whether the breadth was kept."""
    keys = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale', 'kc_refractory_period',
        'kc_adapt_inc', 'kc_adapt_tau', 'kc_thr_input_exp', 'prepass_fixed_thr', 'prepass_thr_factor')
    out = {}
    for _, r in df.iterrows():
        out[r['tag']] = {k: (None if pd.isna(r[k]) else float(r[k])) for k in keys}
        out[r['tag']]['breadth_kept'] = bool(r['breadth_kept'])
    for d in (res_dir(), out_dir()):
        write_json(d / 'tuned_params.json', out)


def write_provenance(df: pd.DataFrame, choices: Dict[str, dict]) -> None:
    import olfsysm
    checks = read_json(res_dir() / 'base_checks.json', {})
    prov = dict(generated=now(), host=socket.gethostname(), script='E71_input_scaled_threshold.py', experiment=EXPERIMENT,
        bases={k: v.get('kws_scalar', {}) for k, v in checks.items()}, base_selection=bases(), base_checks=checks,
        settings=settings(), exponents=EXPONENTS, kept_breadth=KEPT_BREADTH, c2ceil_exponents=c2ceil_exponents()[0],
        exponent_choices=choices, thr_bracket=THR_BRACKET, apl_bracket=APL_BRACKET, bisect_iters=E69.BISECT_ITERS,
        aplpn_cap=APLPN_CAP, high_spikes=HIGH_SPIKES, claw_groups=CLAW_GROUPS, data_refs=data_refs(),
        factor_form=('each KC\'s threshold (fixed_thr + 2 spont_in) x (W / mean W)^a, W = E10_joint_fit.kc_input_weight of the '
            'wiring the model is given (the glomerulus gain included), mean over the model\'s KCs; passed as the simulator\'s '
            'kc_thr_factor to the APL-on and APL-off runs alike; the breadth pre-pass (E10_joint_fit.input_scaled_prepass) with the '
            'same factor, its fixed_thr re-tuned to the factor-free pre-pass\'s APL-off response count'),
        olfsysm_build=olfsysm.build_info() if hasattr(olfsysm, 'build_info') else None,
        model_provenance=H.model_provenance(orn_deltas=H.apl16_orn_deltas()), approx_conc_note=H.APPROX_CONC_NOTE,
        n_variants=int(len(df)), run_failures=read_json(res_dir() / 'run_failures.json', {}), **code_state())
    if isinstance(prov['olfsysm_build'], dict):
        prov['olfsysm_build'].pop('git_diff', None)
    write_json(res_dir() / 'provenance.json', prov)


def stage_summarize() -> None:
    df = merged()
    df.to_csv(res_dir() / 'stats_variants.csv', index=False)
    choices = {k: exponent_choice(k, df) for k in BASE_ORDER if k in set(df.base)}
    write_json(res_dir() / 'exponents.json', dict(choices, _c2ceil_exponents=c2ceil_exponents()[0], _time=now()))
    write_tuned_params(df)
    long = pd.read_csv(res_dir() / 'model_pair_distances_long_A.csv')
    long = long.drop_duplicates(['tag', 'condition', 'odor_a', 'odor_b'], keep='last')
    lp = res_dir() / 'model_pair_distances_long.csv'
    if lp.exists():
        # (part B's fits, E10_joint_fit.py --experiment E71_..., append their rows to the same file: kept)
        old = pd.read_csv(lp)
        long = pd.concat([old[~old.tag.isin(set(long.tag))], long], ignore_index=True)
    long.to_csv(lp, index=False)
    write_table(df, choices)
    figures(df, choices)
    write_provenance(df, choices)
    cols = ['tag', 'opk_on_tv', 'opk_off_tv', 'opk_cv2_off', 'rel_g9plus_share_added', 'mean_dist_bin_off', 'apl_bin_mean',
        'apl_bin_rmse', 'apl_raw_rmse', 'apl_raw_mean', 'apl_cal_rmse', 'apl_cal_mean', 'spearman_rate_on_vs_data', 'thr_factor',
        'apl_factor', 'aplpn_over_cap', 'realism_flag']
    pd.set_option('display.width', 250)
    print(df[[c for c in cols if c in df.columns]].round(4).to_string(index=False), flush=True)
    print('exponent choices:', {k: (c['status'], c.get('exponent')) for k, c in choices.items()}, flush=True)


# ---------------------------------------------------------------------------
# parts B and C (command lines; the runs are the session's)
def partB_commands() -> List[str]:
    """E10_joint_fit.py commands for part B, per the entry: only if part A lowers c2's APL-off odors-per-KC
    distance by PARTB_OPK_GAIN or more at an admissible setting (fit 1, c2's family at its exponent); fit 2
    (the firing-ceiling base at its exponent) only if hal's E69 part B leaves a ceiling fit within 0.005 of 0.058
    under both definitions (checked here on results/E69_kc_firing_ceiling/fits.csv); fit 3 (the reference set at
    its exponent) only if part A brings it inside the APL-off odors-per-KC band."""
    from apl_effect_scores import opk_band
    df = merged()
    choices = {k: exponent_choice(k, df) for k in BASE_ORDER}
    cmds = []
    py = '../../venv/bin/python -u'
    env = 'OMP_NUM_THREADS=12 OMP_THREAD_LIMIT=12 nice -n 10'
    common = (f'-k 2 --experiment {EXPERIMENT} --aplpn-max 3 --rate-target ctrl_frac --loss apl --apl-readout both '
        f'--maxfev {PARTB_MAXFEV}')

    def row(tag):
        return df[df.tag == tag].iloc[0]

    c = choices['c2']
    if c['status'] == 'chosen' and float(row('c2_base').opk_off_tv) - c['best_opk_off_tv'] >= PARTB_OPK_GAIN:
        a = c['exponent']
        cmds.append(f"# fit 1: C2's family at a = {a:g} (APL-off odors-per-KC distance {row('c2_base').opk_off_tv:.3f} -> "
            f"{row(c['tag']).opk_off_tv:.3f})")
        cmds.append(f"{env} {py} E10_joint_fit.py {common} --kc-thr-input-exp {a:g} --init-params "
            f"results/{EXPERIMENT}/tuned_params.json:{c['tag']}")
    else:
        cmds.append(f"# fit 1 not run: c2 {c['status']}" + (f", best APL-off distance {c.get('best_opk_off_tv', np.nan):.3f} against "
            f"the base's {row('c2_base').opk_off_tv:.3f} (needs a fall of {PARTB_OPK_GAIN})" if c['status'] == 'chosen' else ''))
        return cmds
    c = choices['c2ceil']
    e69 = RESULTS / E69_EXP / 'fits.csv'
    ok2 = False
    if e69.exists():
        f = pd.read_csv(e69)
        if {'fit_apl_raw_rmse', 'fit_apl_cal_rmse', 'claw_thr'} <= set(f.columns):
            # (C2's family with the ceiling: no per-claw threshold)
            ok2 = bool(((f.fit_apl_raw_rmse <= 0.058 + 0.005) & (f.fit_apl_cal_rmse <= 0.058 + 0.005) & f.claw_thr.isna()).any())
    b2 = bases()['c2ceil']
    if b2.get('params_from'):
        path, k = b2['params_from'].rsplit(':', 1)
        tp = read_json(H.shared_dir() / 'results' / path)[k]
    else:
        tp = pd.read_csv(RESULTS / E69_EXP / 'fits.csv').set_index('tag').loc[b2['tag']].to_dict()
    if ok2 and c['status'] == 'chosen':
        kc = (f"--kc-refractory-period {tp['kc_refractory_period']!r} --kc-adapt-inc {tp['kc_adapt_inc']!r} --kc-adapt-tau "
            f"{tp['kc_adapt_tau']!r}")
        cmds.append(f"# fit 2: the firing-ceiling base at a = {c['exponent']:g}")
        cmds.append(f"{env} {py} E10_joint_fit.py {common} {kc} --kc-thr-input-exp {c['exponent']:g} --init-params "
            f"results/{EXPERIMENT}/tuned_params.json:{c['tag']}")
    else:
        cmds.append(f"# fit 2 not run: E69 ceiling fit within 0.005 of 0.058 under both definitions: {ok2}; c2ceil {c['status']}")
    c = choices['ref40']
    if c['status'] == 'chosen' and c['exponent'] != 0 and float(row(c['tag']).opk_off_tv) <= opk_band('off'):
        cmds.append(f"# fit 3: the reference set at a = {c['exponent']:g}")
        cmds.append(f"{env} {py} E10_joint_fit.py {common} --claw-thr 40 --glom-gain 1.5 --kc-thr-input-exp {c['exponent']:g} "
            f"--init-params results/{EXPERIMENT}/tuned_params.json:{c['tag']}")
    else:
        cmds.append(f"# fit 3 not run: ref40 {c['status']}" + (f", APL-off distance {row(c['tag']).opk_off_tv:.3f} against the band "
            f"{opk_band('off'):.3f}" if c.get('tag') else ''))
    return cmds


def partB_start_check() -> pd.DataFrame:
    """Part B's requirement that each fit's first evaluation reproduce part A's row of its start: for every part-B
    fit in results/E71_input_scaled_threshold/fits.csv (tags with thrin<a>_), its history's first evaluation (the
    --init-params start) against the part-A row named in its init_params, on the response rates, the APL-effect
    losses and the APL-off odors-per-KC distance; partB_start_check.csv."""
    fits = res_dir() / 'fits.csv'
    if not fits.exists():
        raise FileNotFoundError(f'{fits}: no part-B fits yet')
    df = merged().set_index('tag')
    rows = []
    for r in pd.read_csv(fits).itertuples():
        start = str(r.init_params).rsplit(':', 1)[-1] if isinstance(r.init_params, str) else None
        h = res_dir() / f'history_{r.tag}.csv'
        if start not in df.index or not h.exists():
            continue
        h0 = pd.read_csv(h).iloc[0]
        a = df.loc[start]
        for hc, ac in (('rate_on', 'rate_on'), ('rate_off', 'rate_off'), ('apl_raw_rmse', 'apl_raw_rmse'),
                ('apl_cal_rmse', 'apl_cal_rmse'), ('opk_off_dist', 'opk_off_tv')):
            if hc in h0.index and ac in a.index:
                rows.append(dict(fit=r.tag, start=start, stat=hc, fit_first_eval=float(h0[hc]), part_a=float(a[ac]),
                    abs_diff=abs(float(h0[hc]) - float(a[ac]))))
    out = pd.DataFrame(rows)
    out.to_csv(res_dir() / 'partB_start_check.csv', index=False)
    print(out.to_string(index=False), flush=True)
    return out


def partC_commands() -> List[str]:
    """E43_model_trial_noise.py commands for part C (the block-target experiment's settings, E48: per-trial KC
    threshold jitter calibrated to the data's single-trial floors, 10 pseudo-recordings of 3 trials at the matched
    levels, seed 43; each pseudo-recording's pair distances saved): on C2 itself, and on part B's fit 1 if it is
    within PARTC_FIT_MARGIN of C2's loss under either definition (its input-scaled threshold read from fits.csv,
    the jitter multiplying it)."""
    py = '../../venv/bin/python -u'
    env = 'OMP_NUM_THREADS=12 OMP_THREAD_LIMIT=12 nice -n 10'
    common = (f'--part B --tune-trials 3 --seed 43 --n-recordings 10 --n-trials 3 --experiment {EXPERIMENT} '
        '--levels match_on match_off --save-recording-distances')
    c2 = bases()['c2']
    cmds = ['# C2 itself (calibrated anew: E67 did not run C2 through the pipeline)',
        f"{env} {py} E43_model_trial_noise.py {common} --model-from {c2['exp']}:{c2['tag']}"]
    fits = res_dir() / 'fits.csv'
    if not fits.exists():
        cmds.append('# fit 1: no part-B fits.csv yet')
        return cmds
    f = pd.read_csv(fits)
    # (fit 1: C2's family, no per-claw threshold, no firing ceiling)
    no_ceiling = f.kc_adapt_inc.isna() if 'kc_adapt_inc' in f.columns else pd.Series(True, index=f.index)
    f = f[f.tag.astype(str).str.contains('thrin') & f.claw_thr.isna() & no_ceiling]
    for r in f.itertuples():
        ok = (r.fit_apl_raw_rmse <= C2_LOSSES['raw'] + PARTC_FIT_MARGIN) or (r.fit_apl_cal_rmse <= C2_LOSSES['cal'] + PARTC_FIT_MARGIN)
        if ok:
            cmds.append(f'# fit 1 ({r.tag}): loss raw {r.fit_apl_raw_rmse:.4f}, calibrated {r.fit_apl_cal_rmse:.4f} (C2 '
                f"{C2_LOSSES['raw']:.4f} / {C2_LOSSES['cal']:.4f})")
            cmds.append(f'{env} {py} E43_model_trial_noise.py {common} --model-from {EXPERIMENT}:{r.tag}')
        else:
            cmds.append(f'# fit 1 ({r.tag}) not run: loss raw {r.fit_apl_raw_rmse:.4f}, calibrated {r.fit_apl_cal_rmse:.4f}, more '
                f'than {PARTC_FIT_MARGIN} above C2 under both')
    return cmds


def stage_partC() -> None:
    """Single pseudo-recording pairs scored as E70's single-fly yardstick: each APL-on pseudo-recording i (at the
    APL-on-matched level) with each APL-off pseudo-recording j != i (at the APL-off-matched level; and, as E48
    reads the pipeline, both at the APL-on-matched level), the APL-effect loss and its parts of d_on(i) - d_off(j)
    (raw cosine distances of the three-trial means of spike counts) against the KC data's mean per-pair effect;
    partC_single_pairs.csv and partC_summary.csv (median, quartiles, range per model), beside the data's yardstick
    (results/E70_high_firing_kcs/yardstick.csv)."""
    p = res_dir() / 'partB' / 'pairs_recordings_long.csv'
    if not p.exists():
        raise FileNotFoundError(f'{p}: run the part-C pipeline commands first (--stage partC-commands)')
    d = pd.read_csv(p)
    d = d[(d.kind == 'raw') & (d.transform == 'graded')]
    data = H.yang_apl_effect()
    rows = []
    for model, g in d.groupby('model'):
        levels = set(g.level)
        lab_on = next((x for x in ('match_on', 'match_on+match_off') if x in levels), None)
        lab_off = next((x for x in ('match_off', 'match_on+match_off') if x in levels), None)
        for reading, (lo, lf) in (('matched_each', (lab_on, lab_off)), ('both_at_match_on', (lab_on, lab_on))):
            if lo is None or lf is None:
                continue
            on = {rec: H.canonical_pairs(x.set_index(['odor_a', 'odor_b'])['distance'])
                for rec, x in g[(g.level == lo) & (g.condition == 'on')].groupby('recording')}
            off = {rec: H.canonical_pairs(x.set_index(['odor_a', 'odor_b'])['distance'])
                for rec, x in g[(g.level == lf) & (g.condition == 'acute_off')].groupby('recording')}
            for i, a in on.items():
                for j, b in off.items():
                    if i == j:
                        continue
                    st, _ = H.apl_effect_decomposition(a, b, data)
                    rows.append(dict(model=model, reading=reading, level_on=lo, level_off=lf, recording_on=i, recording_off=j,
                        n_pairs=int(min(len(a), len(b))), **{k.replace('apl_effect_', ''): v for k, v in st.items() if k in (
                        'apl_effect_rmse', 'apl_effect_rmse_centered', 'apl_effect_mean', 'apl_effect_per_odor',
                        'apl_effect_pair_specific', 'apl_effect_spearman')}))
    df = pd.DataFrame(rows)
    df.to_csv(res_dir() / 'partC_single_pairs.csv', index=False)
    yard = pd.read_csv(RESULTS / 'E70_high_firing_kcs' / 'yardstick.csv')
    summ = []
    for (model, reading), g in list(df.groupby(['model', 'reading'])) + [(('KC data (E70 yardstick)', 'one control with one '
            'APL-blocked recording vs the others\' mean'), yard)]:
        summ.append(dict(model=model, reading=reading, n=len(g), **{f'{c}_{q}': float(g[c].quantile(v)) for c in ('rmse', 'mean')
            for q, v in (('median', 0.5), ('q25', 0.25), ('q75', 0.75), ('min', 0.0), ('max', 1.0))}))
    out = pd.DataFrame(summ)
    out.to_csv(res_dir() / 'partC_summary.csv', index=False)
    print(out.round(4).to_string(index=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--stage', choices=['check', 'setting', 'run', 'summarize', 'bases', 'list', 'partB-commands',
        'partB-check', 'partC-commands', 'partC'])
    ap.add_argument('--base', choices=list(BASE_ORDER))
    ap.add_argument('--bases', nargs='+', choices=list(BASE_ORDER), default=None, help='--stage run: only these bases')
    ap.add_argument('--tag')
    ap.add_argument('--base2', default=None, help="the firing-ceiling base before bases.json exists: 'partA' or an E69 fits.csv tag")
    ap.add_argument('--plan', action='store_true', help='print the bases and settings only (nothing written)')
    ap.add_argument('--local-test', metavar='DIR', default=None,
        help='write results and outputs under DIR instead of the shared / outputs folders (a smoke test)')
    args = ap.parse_args()
    if args.local_test:
        root = Path(args.local_test).resolve()
        _DIRS.update(root=root, res=root / 'results', out=root / 'outputs')
    if args.plan:
        b = resolve_bases(write=False, base2=args.base2)
        print(json.dumps({k: v for k, v in b.items() if not k.startswith('_')}, indent=1, default=str))
        ex, best, status = c2ceil_exponents()
        print(f'c2ceil exponents now: {ex} (c2\'s choice: {status}{"" if best is None else f", a = {best:g}"})')
        for s in settings():
            print(f"  {s['tag']}: base {s['base']}, a {s['a']:g}, breadth {s['breadth']}")
        print(f'{len(settings())} settings + {len(BASE_ORDER)} base checks; results -> {res_dir()}, outputs -> {out_dir()}')
        return
    if args.base2 is not None:
        resolve_bases(base2=args.base2)
    if args.stage == 'bases':
        print(json.dumps(resolve_bases(), indent=1, default=str))
    elif args.stage == 'list':
        for s in settings():
            print(s)
    elif args.stage == 'check':
        stage_check(args.base)
    elif args.stage == 'setting':
        stage_setting(args.tag)
    elif args.stage == 'run':
        stage_run(args.bases)
    elif args.stage == 'summarize':
        stage_summarize()
    elif args.stage == 'partB-commands':
        print('\n'.join(partB_commands()))
    elif args.stage == 'partB-check':
        partB_start_check()
    elif args.stage == 'partC-commands':
        print('\n'.join(partC_commands()))
    elif args.stage == 'partC':
        stage_partC()
    else:
        ap.error('--stage or --plan required')


if __name__ == '__main__':
    main()
