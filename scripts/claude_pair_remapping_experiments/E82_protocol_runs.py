#!/usr/bin/env python3
"""E82 part B: runs at the flies' pulse (2 s, the simulation ending 0.25 s after it) with both response rates re-held
(E77_input_screen.hold_rates, as E80_pulse held them: the breadth from the bouton base's pre-pass at the same pulse and
options), spike times saved. The design is the E82 entry's part B in experiment_priorities.md.

The models (`MODELS`; each at its own fitted parameters, as saved):
- C2, the breadth-2 APL-effect fit (E67) on the real wiring; P1_b2, P2_b2, M1_b2, its family's fits on E72's re-drawn
  wirings (E72 part B, `E77_input_side_common_error.members`);
- fit0, the firing-ceiling fit (E76's fit 0: a 3 ms refractory period and an adaptation of 0.111 of the threshold per
  spike decaying over 1 s; `E80_pulse.fit0_parts`);
- fit1, E69's fit 1, the firing ceiling at the literature setting (0.041 per spike, 1 s) on the real wiring; P1_fc,
  P2_fc, M1_fc, E72 part G's fits of it on the re-drawn wirings.
They are built as E77_input_screen.build_c2_simulator builds C2 (`build_simulator`, the same kwargs, with KC-KC
inhibition's time constant as an argument, which build_c2_simulator fixes at E10's 0.05 s), on the wiring's claw x
bouton matrix as E77_input_side_common_error.wiring_simulator rebuilds it (`wiring_parts`); the firing-ceiling fits
add their refractory period and adaptation; the breadth pre-pass has none (as E10_joint_fit / E43.fit_model build it).

The points (`point_specs`), each family starting with identity points at the 0.5 s pulse:
- the identities (I_<family>_<model>): the model at the 0.5 s pulse with the timing kwargs passed at olfsysm's defaults,
  the family's option at the fit's own value (B2 kc_adapt_tau 1 s, B3 bouton_ves_p 0 with bouton_tau_r 1 s, B4 kc_kc_tau
  0.05 s, B5 apl_taum 0.05 s) and its breadth from a pre-pass with the same kwargs, which must equal the model's own
  breadth; fixed parameters, spike times saved. Its spike counts and responses must equal the model's saved ones exactly.
- B1, the populations: every model at 2 s, re-held. B1_C2 and B1_fit0 must reproduce E80's C2_pulse2_held and
  fit0_pulse2_held exactly (counts, responses, spike times, parameters, breadth) before any other 2 s point runs.
- B2, the adaptation's time constant (kc_adapt_tau): fit0 at 0.3, 3 and 10 s, fit1 at 3 and 10 s (for their 1 s).
- B3, bouton depletion (bouton_ves_p 0.01 and 0.03, bouton_tau_r 1 s, in the pre-pass too, as E77's S6), on C2 and fit0.
- B4, the KC-KC time constant (kc_kc_tau 0.2 and 1 s for 0.05 s; the scale stays the fit's), on C2 and fit0.
- B5, APL's time constant (apl_taum 0.01, 0.25 and 1 s on fit0, 0.01 and 0.25 s on C2, for 0.05 s); needs mb_model's
  `apl_taum` kwarg (`apl_taum_supported`), skipped with a message until it is there. APL is off in the breadth pre-pass,
  so neither apl_taum nor kc_kc_tau enters it (KC-KC inhibition is not in the bouton base either).
Every 2 s point holds the APL-off response count, then the APL-on one, at its model's own saved counts (0.5 s), from
its model's fitted parameters. B2 to B5 name their model's B1 point as `reference` (the gates' re-held 2 s start).

Per point (E80_pulse's readings): E77_input_screen.Scorer against the model's own saved outputs at 0.5 s (the
APL-effect loss with its parts on raw counts, under the calibrated readout and binarized, ρ on the per-pair APL effect,
the common-error projection and transfer, the ester / alcohol group errors, the four realism bands under both rules,
the per-odor response fractions), E80_pulse.count_stats (the largest spike counts) and E80_pulse.window_rows (spike
counts and the APL effect on raw counts in windows of the response). The protocol reading (indicator kernels over the
flies' window) and part A's late-spike numbers are E82_flies_protocol.py's, from the saved spike times.

Outputs, results/E82_flies_protocol/: runs.csv (one row per point; resumable), runs_windows.csv, runs_pairs.csv,
runs_params.json (each point's spec, absolute parameters before and after, factors, bisection histories, kwargs,
breadth hashes), runs_errors.csv (points that failed or did not reproduce), runs_populations.csv (the two populations of
four, the mean over wirings of the pair distances per condition, at 0.5 and 2 s, on counts; E72.population_row),
runs_provenance.json. Responses, spike counts and spike times of every point under outputs/E82_flies_protocol/runs/
(<point_id>_{on,acute_off}_{responses,spike_counts,spike_times}.parquet, E80_pulse's naming). The breadth pre-passes
are cached by E77_input_screen.handle_breadth under outputs/E77_input_side_common_error/breadth/.

Usage (from this folder; one process at a time; a C++ error inside olfsysm's parallel region ends the process without
a traceback, so run the parts in separate processes, `--part B1`, then B2 to B5: done points are skipped):
    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 nice -n 10 ../../venv/bin/python -u E82_protocol_runs.py [--part B1 ...]
        [--only ID ...] [--list] [--results-dir DIR --outputs-dir DIR]
    ../../venv/bin/python E82_protocol_runs.py --populations
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
import E80_pulse as E80P

EXPERIMENT = 'E82_flies_protocol'
PULSE = 2.0
IDENTITY_PULSE = E80P.IDENTITY_PULSE
# E69's fit 1 (the firing ceiling at the literature setting, real wiring); E72's G_REAL
FIT1 = ('E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.04142t1000_k2')
DRAW_LABELS = ('P1', 'P2', 'M1')
MODELS = ('C2', 'fit0') + tuple(f'{w}_b2' for w in DRAW_LABELS) + ('fit1',) + tuple(f'{w}_fc' for w in DRAW_LABELS)
POPULATIONS = {'b2': ('C2',) + tuple(f'{w}_b2' for w in DRAW_LABELS),
               'fc': ('fit1',) + tuple(f'{w}_fc' for w in DRAW_LABELS)}
# E80's re-held 2 s points that B1 must reproduce exactly before any other 2 s point
E80_HELD = {'C2': 'C2_pulse2_held', 'fit0': 'fit0_pulse2_held'}
REPRODUCTIONS = tuple(f'B1_{m}' for m in E80_HELD)
# the fits' own values of the options (checked against the fits and olfsysm at run time)
FIT_ADAPT_TAU = 1.0
KC_KC_TAU = 0.05
APL_TAUM_DEFAULT = 0.05
B2_TAUS = {'fit0': (0.3, 3.0, 10.0), 'fit1': (3.0, 10.0)}
B3_VES_P = (0.01, 0.03)
B3_TAU_R = 1.0
B3_MODELS = ('C2', 'fit0')
B4_TAUS = (0.2, 1.0)
B4_MODELS = ('C2', 'fit0')
B5_TAUMS = {'fit0': (0.01, 0.25, 1.0), 'C2': (0.01, 0.25)}
KC_SPIKING = E80P.KC_SPIKING
SCALES = ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')
OFF = E80P.OFF


def _n(responses: pd.DataFrame) -> int:
    return int((responses.to_numpy(dtype=float) > 0.5).sum())


def _git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=H.HERE, capture_output=True, text=True).stdout.strip()


def apl_taum_supported() -> bool:
    """Whether mb_model's fit_mb_model takes `apl_taum` (B5 needs it)."""
    from al_analysis.mb_model import fit_mb_model
    return 'apl_taum' in inspect.signature(fit_mb_model).parameters


# ---------------------------------------------------------------------------
# the points

def _spec(pid: str, part: str, model: str, *, identity: bool, option: str = '', value: float = np.nan,
          opts: Optional[dict] = None, kc_kc_tau: float = KC_KC_TAU, prepass: Optional[dict] = None,
          after: Sequence[str] = (), reference: str = '', reproduces: str = '', requires: str = '',
          population: str = '') -> dict:
    pulse = IDENTITY_PULSE if identity else PULSE
    pop = next((k for k, v in POPULATIONS.items() if model in v), '') if part == 'B1' else population
    return dict(point_id=pid, part=part, model=model, identity=bool(identity), option=option, value=float(value),
                pulse=float(pulse), **E80P.timing_kws(pulse), held=not identity, opts=dict(opts or {}),
                kc_kc_tau=float(kc_kc_tau), prepass=dict(prepass or {}), after=list(after), reference=reference,
                reproduces=reproduces, population=pop, requires=requires)


def point_specs() -> List[dict]:
    """Part B's points (B1 to B5), each family's identity points (0.5 s, fixed) before its 2 s points (re-held); the
    identities of C2 and fit0 and E80's two re-held 2 s points first.

    A spec has point_id, part (B1 to B5), model (`MODELS`), identity, option (the kwarg the family varies, '' for B1),
    value (its value; the fit's own at an identity), pulse, time_stim_end, time_end, held, opts (model kwargs added to
    the model's own), kc_kc_tau, prepass (kwargs added to the breadth pre-pass, besides the timing ones), after (points
    that must have passed first), reference (the model's B1 point, for B2 to B5), reproduces (E80's point, for B1_C2 /
    B1_fit0), population ('b2' / 'fc' for B1 points of the two populations) and requires ('apl_taum' for B5)."""
    pts: List[dict] = []
    for m in E80_HELD:
        pts.append(_spec(f'I_B1_{m}', 'B1', m, identity=True))
    for m, e80 in E80_HELD.items():
        pts.append(_spec(f'B1_{m}', 'B1', m, identity=False, after=[f'I_B1_{m}'], reproduces=e80))
    gate = list(REPRODUCTIONS)
    for m in MODELS:
        if m in E80_HELD:
            continue
        pts.append(_spec(f'I_B1_{m}', 'B1', m, identity=True))
        pts.append(_spec(f'B1_{m}', 'B1', m, identity=False, after=[f'I_B1_{m}'] + gate))

    for m in B2_TAUS:
        pts.append(_spec(f'I_B2_{m}', 'B2', m, identity=True, option='kc_adapt_tau', value=FIT_ADAPT_TAU,
                         opts=dict(kc_adapt_tau=FIT_ADAPT_TAU)))
    for m, taus in B2_TAUS.items():
        for t in taus:
            pts.append(_spec(f'B2_{m}_tau{t:g}', 'B2', m, identity=False, option='kc_adapt_tau', value=t,
                             opts=dict(kc_adapt_tau=t), after=[f'I_B2_{m}'] + gate, reference=f'B1_{m}'))

    for m in B3_MODELS:
        dep = dict(bouton_ves_p=0.0, bouton_tau_r=B3_TAU_R)
        pts.append(_spec(f'I_B3_{m}', 'B3', m, identity=True, option='bouton_ves_p', value=0.0, opts=dep, prepass=dep))
    for m in B3_MODELS:
        for p in B3_VES_P:
            dep = dict(bouton_ves_p=p, bouton_tau_r=B3_TAU_R)
            pts.append(_spec(f'B3_{m}_ves{p:g}', 'B3', m, identity=False, option='bouton_ves_p', value=p, opts=dep,
                             prepass=dep, after=[f'I_B3_{m}'] + gate, reference=f'B1_{m}'))

    for m in B4_MODELS:
        pts.append(_spec(f'I_B4_{m}', 'B4', m, identity=True, option='kc_kc_tau', value=KC_KC_TAU, kc_kc_tau=KC_KC_TAU))
    for m in B4_MODELS:
        for t in B4_TAUS:
            pts.append(_spec(f'B4_{m}_kckctau{t:g}', 'B4', m, identity=False, option='kc_kc_tau', value=t,
                             kc_kc_tau=t, after=[f'I_B4_{m}'] + gate, reference=f'B1_{m}'))

    for m in B5_TAUMS:
        pts.append(_spec(f'I_B5_{m}', 'B5', m, identity=True, option='apl_taum', value=APL_TAUM_DEFAULT,
                         opts=dict(apl_taum=APL_TAUM_DEFAULT), requires='apl_taum'))
    for m, taums in B5_TAUMS.items():
        for t in taums:
            pts.append(_spec(f'B5_{m}_taum{t:g}', 'B5', m, identity=False, option='apl_taum', value=t,
                             opts=dict(apl_taum=t), after=[f'I_B5_{m}'] + gate, reference=f'B1_{m}',
                             requires='apl_taum'))
    pts += b6_specs()
    return pts


# part B6 (the entry's part D, "if a setting passed the protocol gate"): the literature setting's four wirings re-held
# at the passing setting with the smallest change, the ceiling's adaptation at 3 s (B2_fit0_tau3 passed, 2026-11-11 06:44)
B6_TAU = 3.0
B6_MODELS = ('fit1',) + tuple(f'{w}_fc' for w in DRAW_LABELS)


def b6_specs() -> List[dict]:
    """Part B6's points: each of the literature setting's four wirings (E69's fit 1 and E72 part G's P1, P2, M1 fits) at
    the flies' pulse, re-held, with the adaptation's time constant at B6_TAU; population 'fc_tau3'."""
    gate = list(REPRODUCTIONS)
    pts = [_spec(f'I_B6_{m}', 'B6', m, identity=True, option='kc_adapt_tau', value=FIT_ADAPT_TAU,
                 opts=dict(kc_adapt_tau=FIT_ADAPT_TAU)) for m in B6_MODELS]
    pts += [_spec(f'B6_{m}_tau{B6_TAU:g}', 'B6', m, identity=False, option='kc_adapt_tau', value=B6_TAU,
                  opts=dict(kc_adapt_tau=B6_TAU), after=[f'I_B6_{m}'] + gate, reference=f'B1_{m}', population='fc_tau3')
             for m in B6_MODELS]
    return pts


def with_prerequisites(specs: Sequence[dict], want: Set[str]) -> Set[str]:
    """`want` with every point its points wait for (`after`, transitively)."""
    by = {s['point_id']: s for s in specs}
    out, stack = set(want), list(want)
    while stack:
        for a in by[stack.pop()]['after']:
            if a not in out:
                out.add(a)
                stack.append(a)
    return out


# ---------------------------------------------------------------------------
# the models

_MODELS: Dict[str, dict] = {}
_WIRINGS: Dict[str, dict] = {}


def _empty(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v) == ''


def _draw_seed(label: str) -> int:
    import E72_wiring_population as E72
    prefix = dict(E72.DRAWS)[label]
    return int(prefix.split('seed')[1].rstrip('_'))


def model_source(model: str) -> Tuple[str, str, Optional[str]]:
    """(results folder, tag, wiring label or None) of a model's saved fit."""
    import E72_wiring_population as E72
    import E77_input_screen as S
    import E77_input_side_common_error as E77
    if model == 'C2':
        return S.C2[0], S.C2[1], None
    if model == 'fit0':
        return E80P.FIT0_FOLDER, E80P.fit0_parts()['tag'], None
    if model == 'fit1':
        if tuple(E72.G_REAL) != FIT1:
            raise RuntimeError(f"E72's part G real-wiring fit is {E72.G_REAL}, not {FIT1}")
        return FIT1[0], FIT1[1], None
    label, fam = model.split('_')
    if label not in DRAW_LABELS:
        raise ValueError(model)
    if fam == 'b2':
        mem = {lab: (f, tg) for lab, f, tg in E77.members()}
        return mem[label][0], mem[label][1], label
    if fam == 'fc':
        return E72.EXPERIMENT, dict(E72.G_DRAWS)[label] + E72.G_SUFFIX, label
    raise ValueError(model)


def fit_row(folder: str, tag: str, wiring: Optional[str]) -> pd.Series:
    """The fits.csv row of a fit, checked to be what `build_simulator` builds: the bouton base with breadth exponent 2,
    KC-KC inhibition at E10's tau, APL driven by the claws, no per-claw threshold, glomerulus gain, other ORN input or
    input-scaled threshold, and the wiring `wiring` (None: the real one)."""
    import E10_joint_fit as J
    import E77_input_screen as S
    f = pd.read_csv(H.results_dir(folder) / 'fits.csv')
    m = f[f.tag == tag]
    if len(m) != 1:
        raise RuntimeError(f'{folder}/fits.csv has {len(m)} rows for {tag}')
    r = m.iloc[0]
    for c in ('claw_thr', 'gain', 'gain_glomeruli', 'orn_input', 'kc_thr_input_exp', 'claw_thr_subgain', 'claw_thr_width',
              'base_params'):
        if c in r.index and not _empty(r[c]):
            raise RuntimeError(f'{tag} has {c}={r[c]!r}; build_simulator would not build it')
    if str(r.get('apl_drive', 'claws')) != 'claws' or float(r['k']) != S.BREADTH_K or float(r['kc_kc_tau']) != J.TAU:
        raise RuntimeError(f"{tag}: apl_drive {r.get('apl_drive')}, k {r['k']}, kc_kc_tau {r['kc_kc_tau']}")
    w, wf, wt = (r.get(c) for c in ('wiring', 'wiring_file', 'wiring_tag'))
    if wiring is None:
        bad = [c for c, v in (('wiring', w), ('wiring_file', wf), ('wiring_tag', wt)) if not _empty(v)]
        if bad:
            raise RuntimeError(f'{tag}: {bad} set for a real-wiring fit')
    else:
        seed = _draw_seed(wiring)
        if wiring.startswith('P'):
            ok = str(w) == 'claw_permutation_within_type' and int(float(r['wiring_seed'])) == seed and _empty(wf)
        else:
            ok = (_empty(w) and str(wt) == f'curveball_within_type_matched-seed{seed}'
                  and Path(str(wf)).name == f'wPNKC_curveball_within_type-seed{seed}_matched.parquet')
        if not ok:
            raise RuntimeError(f'{tag}: wiring {w!r} / seed {r.get("wiring_seed")} / file {wf!r} / tag {wt!r} is not {wiring}')
    return r


def model_parts(model: str) -> dict:
    """A model's pieces (once per process): 'p' (its fitted absolute parameters), 'kc' (its refractory period and
    adaptation kwargs, {} for the breadth-2 family), 'saved' ({'on', 'off'}: (responses, spike counts) as saved),
    'target_on' / 'target_off' (its response counts, two or more spikes), 'source' (folder, tag), 'wiring' (None or
    the draw label)."""
    if model in _MODELS:
        return _MODELS[model]
    import E72_wiring_population as E72
    import E77_input_screen as S
    if model == 'C2':
        P = S.c2_parts()
        m = dict(p=dict(P['p']), kc={}, saved=P['saved'], target_on=P['target_on'], target_off=P['target_off'],
                 source=tuple(S.C2), wiring=None)
    elif model == 'fit0':
        F = E80P.fit0_parts()
        m = dict(p=dict(F['p']), kc=dict(F['kc']), saved=F['saved'], target_on=F['target_on'],
                 target_off=F['target_off'], source=(E80P.FIT0_FOLDER, F['tag']), wiring=None)
    else:
        folder, tag, wiring = model_source(model)
        r = fit_row(folder, tag, wiring)
        has_kc = not _empty(r.get('kc_refractory_period'))
        if has_kc != (model == 'fit1' or model.endswith('_fc')):
            raise RuntimeError(f'{model} ({tag}): kc_refractory_period {r.get("kc_refractory_period")!r}')
        kc = {c: float(r[c]) for c in KC_SPIKING} if has_kc else {}
        saved = E72.load_fit(folder, tag)
        if saved is None:
            raise RuntimeError(f"{model}'s saved outputs ({folder}, {tag}) not found")
        m = dict(p=E72.fitted_params(folder, tag), kc=kc, saved=saved, target_on=_n(saved['on'][0]),
                 target_off=_n(saved['off'][0]), source=(folder, tag), wiring=wiring)
    _MODELS[model] = m
    return m


def wiring_parts(label: str) -> dict:
    """A re-drawn wiring as E77_input_side_common_error.wiring_simulator rebuilds it (E10_joint_fit.bouton_base with
    E72's draw, its APL-off run cached under outputs/E77_input_side_common_error/): 'W' (claw x bouton), 'breadth' (the
    base's at olfsysm's 0.5 s pulse), 'kws' (the draw's bouton_base kwargs), 'wPNKC_sha256'. Checks that the normalized
    APL weights are C2's (build_simulator uses those)."""
    if label in _WIRINGS:
        return _WIRINGS[label]
    import E10_joint_fit as J
    import E77_input_screen as S
    import E77_input_side_common_error as E77
    P = S.c2_parts()
    seed = _draw_seed(label)
    if label.startswith('P'):
        wk = dict(wiring='claw_permutation_within_type', wiring_seed=seed)
        rec = dict(wk)
    else:
        f = H.shared_dir() / 'data' / 'model_outputs' / 'E59_coinput_shuffle' / f'wPNKC_curveball_within_type-seed{seed}_matched.parquet'
        wk = dict(wiring_matrix=pd.read_parquet(f), wiring_tag=f'curveball_within_type_matched-seed{seed}')
        rec = dict(wiring_file=str(f), wiring_tag=wk['wiring_tag'])
    _, wnorm, W, breadth = J.bouton_base(P['orn'], H.outputs_dir(E77.EXPERIMENT), None, **wk)
    for k, v in wnorm.items():
        if not (v.index.equals(P['wnorm'][k].index) and np.array_equal(v.to_numpy(), P['wnorm'][k].to_numpy())):
            raise RuntimeError(f"{label}: the normalized {k} weights differ from C2's")
    _WIRINGS[label] = dict(W=W, breadth=breadth, kws=rec, wPNKC_sha256=S.frame_sha256(W))
    return _WIRINGS[label]


def own_breadth(m: dict) -> pd.Series:
    """The breadth behind a model's saved APL->KC weights (the bouton base's pre-pass at olfsysm's 0.5 s pulse on its
    wiring)."""
    import E77_input_screen as S
    return S.c2_parts()['breadth'] if m['wiring'] is None else wiring_parts(m['wiring'])['breadth']


def point_breadth(spec: dict, m: dict) -> pd.Series:
    """The breadth pre-pass at the point's pulse with its pre-pass kwargs on its model's wiring
    (E77_input_screen.handle_breadth: the bouton base at E3's parameters, APL off, no KC-KC inhibition, no refractory
    period or adaptation)."""
    import E77_input_screen as S
    W = None if m['wiring'] is None else wiring_parts(m['wiring'])['W']
    return S.handle_breadth(dict(E80P.timing_kws(spec['pulse']), **spec['prepass']), wPNKC=W)


def build_simulator(extra_kws: Optional[dict], *, wPNKC: Optional[pd.DataFrame], breadth: pd.Series,
                    params: Dict[str, float], kc_kc_tau: float = KC_KC_TAU) -> Tuple[Callable, dict]:
    """E77_input_screen.build_c2_simulator with KC-KC inhibition's time constant as an argument (build_c2_simulator
    sets E10's 0.05 s after the extra kwargs): the bouton base's kwargs plus `extra_kws`, C2's normalized connectome APL
    weights, breadth exponent 2 on the APL->KC weights with `breadth`, KC-KC inhibition at the parameters' scale with
    time constant `kc_kc_tau`, on the wiring `wPNKC` (None: the hemibrain's). Returns (simulate, info) as
    build_c2_simulator does; info['kws'] has kc_kc_tau."""
    import E10_joint_fit as J
    import E77_input_screen as S
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    if {'kc_kc_tau', 'kc_kc_inh'} & set(extra_kws or {}):
        raise ValueError("pass kc_kc_tau as its own argument (the scale is the parameters' wKCKC_scale)")
    P = S.c2_parts()
    p = dict(params)
    wAPLKC = J.breadth_weights(P['wnorm']['wAPLKC'], breadth, S.BREADTH_K)
    kws = dict(J.base_kws(None), **(extra_kws or {}))
    scalar = dict(S._scalar_kws(kws), kc_kc_tau=float(kc_kc_tau))
    if wPNKC is not None:
        kws['_wPNKC'] = wPNKC
    kws.update(_wAPLKC=wAPLKC, _wKCAPL=P['wnorm']['wKCAPL'], _wAPLPN=P['wnorm']['wAPLPN'], _wPNAPL=P['wnorm']['wPNAPL'],
               kc_kc_inh=p['wKCKC_scale'], kc_kc_tau=float(kc_kc_tau))
    sim, _ = make_fixed_param_kc_simulator(P['orn'], initial_params=p, **kws)
    cells = {k: c for k, c in zip(sim.__code__.co_freevars, sim.__closure__)}
    if not np.array_equal(cells['wAPLKC_unscaled'].cell_contents.ravel(), wAPLKC.to_numpy()):
        raise RuntimeError("the simulator's unscaled APL->KC weights are not the ones it was built with")
    return sim, dict(params=p, kws=scalar, breadth=breadth, wAPLKC=wAPLKC, orn=P['orn'],
                     wPNKC=P['W'] if wPNKC is None else wPNKC, cells=cells)


def model_kws(spec: dict, m: dict) -> dict:
    """The point's model kwargs beside the bouton base's: the model's refractory period and adaptation, the family's
    option, the timing kwargs."""
    return dict(m['kc'], **spec['opts'], **E80P.timing_kws(spec['pulse']))


def check_option(spec: dict, m: dict) -> None:
    """At an identity point the option must be at the fit's own value (RuntimeError otherwise)."""
    import E10_joint_fit as J
    if float(J.TAU) != KC_KC_TAU:
        raise RuntimeError(f"E10's KC-KC tau is {J.TAU}, not {KC_KC_TAU}")
    if not spec['identity']:
        return
    if spec['option'] == 'kc_adapt_tau' and m['kc'].get('kc_adapt_tau') != spec['value']:
        raise RuntimeError(f"{spec['point_id']}: the fit's own kc_adapt_tau is {m['kc'].get('kc_adapt_tau')}")
    if spec['option'] == 'apl_taum':
        import olfsysm as osm
        d = float(osm.ModelParams().kc.apl_taum)
        if d != spec['value']:
            raise RuntimeError(f"{spec['point_id']}: olfsysm's default apl_taum is {d}, not {spec['value']}")


def run_spec(spec: dict) -> Tuple[Dict[str, tuple], dict, Dict[str, pd.DataFrame], dict]:
    """One point: its outputs ({'on', 'off'}: (responses, spike counts)), its run record (E80_pulse.run_point's: the
    fixed parameters, or hold_rates' with the spike-time runs checked against it), its spikes and a record of its build
    ('breadth', 'model_kws', 'sim_kws', 'build_seconds'). An identity point's pre-pass must give its model's own
    breadth exactly (RuntimeError otherwise)."""
    m = model_parts(spec['model'])
    check_option(spec, m)
    t0 = time.time()
    b = point_breadth(spec, m)
    if spec['identity']:
        own = own_breadth(m)
        if not (b.index.equals(own.index) and np.array_equal(b.to_numpy(), own.to_numpy())):
            raise RuntimeError(f"{spec['point_id']}: the breadth pre-pass at {spec['pulse']} s with {spec['prepass']} "
                               f"differs from {spec['model']}'s own breadth")
    W = None if m['wiring'] is None else wiring_parts(m['wiring'])['W']
    kws = model_kws(spec, m)
    sim, info = build_simulator(kws, wPNKC=W, breadth=b, params=m['p'], kc_kc_tau=spec['kc_kc_tau'])
    built = time.time() - t0
    print(f'  built in {built:.0f} s', flush=True)
    o, res, spikes = E80P.run_point(dict(point_id=spec['point_id'], held=spec['held']), sim, info, m)
    rec = dict(breadth=b, model_kws=kws, sim_kws=info['kws'], build_seconds=built)
    del sim, info
    gc.collect()
    return o, res, spikes, rec


# ---------------------------------------------------------------------------
# the checks

def spikes_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Whether two spike-time frames (`simulate(..., spike_times=True)`'s, sorted by odor, KC and step) hold the same
    spikes."""
    return len(a) == len(b) and all(np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in ('kc', 'odor', 'step'))


def e80_point(pid: str) -> dict:
    """E80's saved point: 'out' ({'on', 'off'}: (responses, spike counts)), 'spikes', 'params' (pulse_params.json's
    record), from outputs/E80_remaining_candidates/pulse/ or the model-output archive."""
    rec = json.loads((H.results_dir(E80P.EXPERIMENT) / 'pulse_params.json').read_text())[pid]
    for d in (H.HERE / 'outputs' / E80P.EXPERIMENT / 'pulse',
              H.shared_dir() / 'data' / 'model_outputs' / E80P.EXPERIMENT / 'pulse'):
        fs = {(c, k): d / f'{pid}_{cc}_{k}.parquet' for c, cc in (('on', 'on'), ('off', 'acute_off'))
              for k in ('responses', 'spike_counts', 'spike_times')}
        if all(f.exists() for f in fs.values()):
            r = {k: pd.read_parquet(f) for k, f in fs.items()}
            return dict(out={c: (r[(c, 'responses')], r[(c, 'spike_counts')]) for c in ('on', 'off')},
                        spikes={c: r[(c, 'spike_times')] for c in ('on', 'off')}, params=rec, folder=str(d))
    raise FileNotFoundError(f"E80's saved outputs of {pid} not found")


def check_point(spec: dict, m: dict, o: Dict[str, tuple], res: dict, spikes: Dict[str, pd.DataFrame],
                breadth: pd.Series) -> Tuple[bool, dict]:
    """(passed, record): an identity point must reproduce its model's saved spike counts and responses exactly at
    factors 1; B1_C2 / B1_fit0 must reproduce E80's point exactly (counts, responses, spike times, parameters on and
    off, breadth hash). Other points pass."""
    if spec['identity']:
        d = E80P.count_diffs(o, m['saved'])
        d['factors'] = [res['thr_factor'], res['apl_factor']]
        ok = d['exact'] and res['thr_factor'] == 1.0 and res['apl_factor'] == 1.0
        return ok, dict(kind='identity', against=':'.join(m['source']), **d)
    if spec['reproduces']:
        e = e80_point(spec['reproduces'])
        d = E80P.count_diffs(o, e['out'])
        d['spikes_equal'] = all(spikes_equal(spikes[c], e['spikes'][c]) for c in ('on', 'off'))
        d['params_equal'] = (res['params'] == {k: float(v) for k, v in e['params']['params'].items()}
                             and res['params_off'] == {k: float(v) for k, v in e['params']['params_off'].items()})
        d['breadth_equal'] = E80P._sha(breadth.to_numpy()) == e['params']['breadth_sha256']
        ok = d['exact'] and d['spikes_equal'] and d['params_equal'] and d['breadth_equal']
        return ok, dict(kind='reproduction', against=f"{E80P.EXPERIMENT}:{spec['reproduces']}", folder=e['folder'], **d)
    return True, dict(kind='')


# ---------------------------------------------------------------------------
# the run

def _save_outputs(d: Path, pid: str, o: Dict[str, tuple], spikes: Dict[str, pd.DataFrame]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for c, cc in (('on', 'on'), ('off', 'acute_off')):
        o[c][0].to_parquet(d / f'{pid}_{cc}_responses.parquet')
        o[c][1].to_parquet(d / f'{pid}_{cc}_spike_counts.parquet')
        spikes[c].to_parquet(d / f'{pid}_{cc}_spike_times.parquet')


def load_run(d: Path, pid: str) -> Dict[str, tuple]:
    """A saved point's {'on', 'off'}: (responses, spike counts)."""
    return {c: tuple(pd.read_parquet(d / f'{pid}_{cc}_{k}.parquet') for k in ('responses', 'spike_counts'))
            for c, cc in (('on', 'on'), ('off', 'acute_off'))}


def default_outputs() -> Path:
    return H.outputs_dir(EXPERIMENT) / 'runs'


def write_provenance(out: Path, specs: Sequence[dict]) -> None:
    """runs_provenance.json: the model's provenance (`H.model_provenance`, on the lab's input and C2's wiring), the
    points, the brackets of hold_rates, the constants; each point's source and parameters are in runs_params.json."""
    import E77_input_screen as S
    P = S.c2_parts()
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E82_protocol_runs.py',
                al_analysis_commit=_git('rev-parse', 'HEAD'),
                al_analysis_dirty=bool(_git('status', '--porcelain', '--untracked-files=no')),
                model=H.model_provenance(orn_deltas=P['orn'], _wPNKC=P['W']), apl_taum_supported=apl_taum_supported(),
                c2=dict(experiment=S.C2[0], tag=S.C2[1]), fit0=dict(experiment=E80P.FIT0_FOLDER),
                fit1=dict(experiment=FIT1[0], tag=FIT1[1]), draws=list(DRAW_LABELS), populations=POPULATIONS,
                e80_reproduced=E80_HELD, bouton_base_params=P['init'], breadth_k=S.BREADTH_K, tied_on=S.TIED_ON,
                thr_bracket=S.THR_BRACKET, thr_step=S.THR_STEP, apl_bracket=S.APL_BRACKET, apl_step=S.APL_STEP,
                tail=E80P.TAIL, windows=[list(w) for w in E80P.WINDOWS], points=list(specs),
                approx_conc_note=H.APPROX_CONC_NOTE,
                breadth=('the bouton base (E3 parameters, APL off, no KC-KC inhibition, no refractory period or adaptation) '
                         "on the model's wiring at the point's pulse with its pre-pass kwargs (depletion for B3); "
                         'E77_input_screen.handle_breadth'),
                scoring=("E77_input_screen.Scorer against the model's own saved outputs at 0.5 s; E80_pulse.count_stats; "
                         "E80_pulse.window_rows (raw-count cosine distances on the spikes in each window)"))
    (out / 'runs_provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')


def _log_error(path: Path, pid: str, kind: str, detail) -> None:
    pd.DataFrame([dict(point_id=pid, time=datetime.now().isoformat(timespec='seconds'), kind=kind,
                       detail=str(detail)[:2000])]).to_csv(path, index=False, mode='a', header=not path.exists())


def _nan(d: dict, k: str) -> float:
    return float(d[k]) if k in d else np.nan


def run_points(out: Path, only: Optional[Sequence[str]] = None, outputs: Optional[Path] = None) -> pd.DataFrame:
    """Runs part B (resumable); writes runs.csv, runs_windows.csv, runs_pairs.csv, runs_params.json, runs_errors.csv,
    runs_populations.csv and runs_provenance.json.

    Args:
        out: the results folder (results/E82_flies_protocol/).
        only: point ids to run (default all); the points they wait for (their identity points and E80's two
            reproductions) run first if not done.
        outputs: where the points' parquet files go (default outputs/E82_flies_protocol/runs/).

    Returns runs.csv's rows. A point that fails, or an identity or reproduction that differs, is listed in
    runs_errors.csv and not marked done, and the points waiting for it are skipped; the others go on. RuntimeError at
    the end if anything failed.
    """
    import E77_input_screen as S
    out.mkdir(parents=True, exist_ok=True)
    outputs = default_outputs() if outputs is None else Path(outputs)
    specs = point_specs()
    ids = [s['point_id'] for s in specs]
    want = set(ids) if only is None else set(only)
    unknown = want - set(ids)
    if unknown:
        raise ValueError(f'unknown points {sorted(unknown)}')
    want = with_prerequisites(specs, want)
    write_provenance(out, specs)
    have_taum = apl_taum_supported()

    rp, wp, pp, jp, ep = (out / f for f in ('runs.csv', 'runs_windows.csv', 'runs_pairs.csv', 'runs_params.json',
                                            'runs_errors.csv'))
    rows = pd.read_csv(rp) if rp.exists() else pd.DataFrame()
    done = set(rows.point_id) if len(rows) else set()
    for f in (wp, pp):
        E80P._filtered_csv(f, done)
    params = json.loads(jp.read_text()) if jp.exists() else {}
    scorers: Dict[str, object] = {}
    failed: List[str] = []
    for spec in specs:
        pid = spec['point_id']
        if pid not in want or pid in done:
            continue
        if spec['requires'] == 'apl_taum' and not have_taum:
            print(f"{pid}: skipped; B5 needs mb_model's apl_taum option (fit_mb_model has no apl_taum kwarg in this "
                  'al_analysis checkout)', flush=True)
            continue
        missing = [a for a in spec['after'] if a not in done]
        if missing:
            print(f'{pid}: waits for {missing}; skipped', flush=True)
            continue
        print(f"--- {pid} ({spec['model']}, pulse {spec['pulse']:g} s, simulation to {spec['time_end']:g} s, "
              f"{'both response rates re-held' if spec['held'] else 'fixed parameters'}"
              + (f", {spec['option']} {spec['value']:g}" if spec['option'] else '') + ')', flush=True)
        try:
            m = model_parts(spec['model'])
            o, res, spikes, rec = run_spec(spec)
            ok, chk = check_point(spec, m, o, res, spikes, rec['breadth'])
        except Exception as e:
            print(f'{pid} FAILED: {e!r}'[:1000], flush=True)
            _log_error(ep, pid, 'error', repr(e))
            failed.append(pid)
            gc.collect()
            continue
        if not ok:
            print(f"{pid}: the {chk['kind']} check FAILED against {chk['against']}: {chk}", flush=True)
            _log_error(ep, pid, chk['kind'], json.dumps(chk, default=str))
            failed.append(pid)
            del o, res, spikes, rec
            gc.collect()
            continue
        if chk['kind']:
            print(f"{pid}: reproduces {chk['against']}'s saved outputs exactly ({chk['kind']})", flush=True)

        t1 = time.time()
        if spec['model'] not in scorers:
            scorers[spec['model']] = S.Scorer(m['saved'])
        row, pairs = scorers[spec['model']].score(o)
        wrows = E80P.window_rows(pid, spec, o, spikes)
        rp_ = res['params']
        kws = rec['model_kws']
        meta = dict(point_id=pid, part=spec['part'], model=spec['model'], source=':'.join(m['source']),
                    wiring=m['wiring'] or 'real', population=spec['population'], option=spec['option'],
                    value=spec['value'], identity=spec['identity'], reproduces=spec['reproduces'],
                    reference=spec['reference'], pulse=spec['pulse'], time_stim_end=spec['time_stim_end'],
                    time_end=spec['time_end'], held=spec['held'], breadth_sha256=E80P._sha(rec['breadth'].to_numpy()),
                    prepass=json.dumps(spec['prepass'], sort_keys=True), fixed_thr=rp_['fixed_thr'],
                    fixed_thr_off=res['params_off']['fixed_thr'], **{k: rp_[k] for k in SCALES},
                    fixed_thr_start=m['p']['fixed_thr'], wAPLKC_scale_start=m['p']['wAPLKC_scale'],
                    **{k: _nan(kws, k) for k in KC_SPIKING + ('bouton_ves_p', 'bouton_tau_r', 'apl_taum')},
                    kc_kc_tau=spec['kc_kc_tau'], thr_factor=res['thr_factor'], apl_factor=res['apl_factor'],
                    n_on=_n(o['on'][0]), n_off=_n(o['off'][0]), target_on=m['target_on'], target_off=m['target_off'],
                    at_bound_thr=res['at_bound_thr'], at_bound_apl=res['at_bound_apl'], n_sims=res['n_sims'],
                    build_seconds=rec['build_seconds'], seconds=rec['build_seconds'] + res['seconds'] + (time.time() - t1),
                    check=chk['kind'] or '', **E80P.count_stats(o))
        row = {k: v for k, v in row.items() if k not in meta}
        pd.DataFrame(wrows).to_csv(wp, index=False, mode='a', header=not wp.exists())
        pairs.insert(0, 'point_id', pid)
        pairs.to_csv(pp, index=False, mode='a', header=not pp.exists())
        params[pid] = dict(spec, source=list(m['source']), wiring=m['wiring'], start_params=m['p'], params=rp_,
                           params_off=res['params_off'], thr_factor=res['thr_factor'], apl_factor=res['apl_factor'],
                           n_on=meta['n_on'], n_off=meta['n_off'], target_on=m['target_on'],
                           target_off=m['target_off'], history=res.get('history', []),
                           breadth_sha256=meta['breadth_sha256'], model_kws=kws, sim_kws=rec['sim_kws'],
                           wiring_record=None if m['wiring'] is None else
                           dict(wiring_parts(m['wiring'])['kws'], wPNKC_sha256=wiring_parts(m['wiring'])['wPNKC_sha256']),
                           check=chk)
        jp.write_text(json.dumps(params, indent=1, default=str) + '\n')
        _save_outputs(outputs, pid, o, spikes)
        new = dict(meta, **row)
        rows = pd.concat([rows, pd.DataFrame([new])], ignore_index=True) if len(rows) else pd.DataFrame([new])
        rows.to_csv(rp, index=False)
        done.add(pid)
        r = rows.iloc[-1]
        win = {w['window']: w for w in wrows}
        print(f"{pid}: thr x{r.thr_factor:.4f}, APL x{r.apl_factor:.4f}; response rates {r.rate_on:.4f} / "
              f"{r.rate_off:.4f} (ratio {r.ratio:.3f}; {r.n_on} / {r.n_off} responses); largest counts "
              f"{r.max_count_on} / {r.max_count_off}; readout loss {r.cal_loss:.4f} (mean {r.cal_mean:+.3f}, pattern "
              f"{r.cal_pattern:.4f}), raw {r.raw_loss:.4f} (mean {r.raw_mean:+.3f}, pattern {r.raw_pattern:.4f}); "
              'windows (raw mean effect): ' + ', '.join(f"{k} {v.get('raw_mean', np.nan):+.3f}" for k, v in win.items())
              + f"; realism: {r.realism_flag if isinstance(r.realism_flag, str) and r.realism_flag else 'inside the bands'}"
              + f' ({meta["seconds"]:.0f} s)', flush=True)
        del o, res, spikes, rec
        gc.collect()
    population_rows(out, outputs)
    if failed:
        raise RuntimeError(f'points failed or did not reproduce (runs_errors.csv): {failed}')
    return rows


def population_rows(out: Path, outputs: Optional[Path] = None) -> pd.DataFrame:
    """runs_populations.csv: each population of four (`POPULATIONS`) at the 0.5 s pulse (its identity points, which
    are the saved fits) and at 2 s re-held (its B1 points), where all four are done: E72.population_row (the mean over
    the wirings of the pair distances per condition, on raw counts, under the calibrated readout with K per wiring and
    binarized; the wiring-mean response rates and fractions; odors per KC)."""
    import E72_wiring_population as E72
    outputs = default_outputs() if outputs is None else Path(outputs)
    rp = out / 'runs.csv'
    done = set(pd.read_csv(rp).point_id) if rp.exists() else set()
    res = []
    for fam, members in POPULATIONS.items():
        for pulse, prefix in ((IDENTITY_PULSE, 'I_B1_'), (PULSE, 'B1_')):
            pids = [f'{prefix}{m}' for m in members]
            if not all(p in done for p in pids):
                continue
            fits = [load_run(outputs, p) for p in pids]
            dist = {ro: [E72.distances(o, ro) for o in fits] for ro in ('raw', 'cal', 'binary')}
            res.append(dict(population=fam, pulse=pulse, members=';'.join(pids), **E72.population_row(fits, dist)))
    df = pd.DataFrame(res)
    if len(df):
        df.to_csv(out / 'runs_populations.csv', index=False)
        print(df[['population', 'pulse', 'n_wirings', 'raw_loss', 'raw_mean', 'raw_pattern', 'cal_loss', 'cal_mean',
                  'cal_pattern', 'rate_on', 'ratio']].round(4).to_string(), flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', nargs='+', help='point ids to run (the points they wait for run first)')
    ap.add_argument('--part', nargs='+', choices=('B1', 'B2', 'B3', 'B4', 'B5', 'B6'),
                    help="the parts' points to run (with --only: both; the points they wait for run first)")
    ap.add_argument('--list', action='store_true', help='list the points and exit')
    ap.add_argument('--populations', action='store_true', help='only rewrite runs_populations.csv from saved points')
    ap.add_argument('--results-dir', type=Path, help='results folder (default <shared>/results/E82_flies_protocol)')
    ap.add_argument('--outputs-dir', type=Path, help='parquet folder (default outputs/E82_flies_protocol/runs)')
    args = ap.parse_args()
    if args.list:
        for s in point_specs():
            print(s['point_id'], s['part'], s['model'], f"pulse {s['pulse']:g}", 'held' if s['held'] else 'fixed',
                  s['option'] or '-', '' if np.isnan(s['value']) else f"{s['value']:g}", 'after', ','.join(s['after']) or '-')
        return
    out = args.results_dir or H.results_dir(EXPERIMENT)
    if args.populations:
        population_rows(out, args.outputs_dir)
        return
    only = None
    if args.only or args.part:
        only = list(args.only or []) + [s['point_id'] for s in point_specs() if s['part'] in (args.part or ())]
    run_points(out, only=only, outputs=args.outputs_dir)


if __name__ == '__main__':
    main()
