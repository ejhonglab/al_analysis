#!/usr/bin/env python3
"""E10: joint fit of one parameter set to the control (APL on) and TNT (APL off) matrices.

Family (the general one, per E8 / E12 / E17, not E15's block-specific rule): bouton base
(E3's wiring; connectome per-claw APL weights, PN-bouton <-> APL) with each KC's APL->KC
weight multiplied by ((breadth + 1) / mean(breadth + 1))^k (E12's breadth weighting, breadth
= total APL-off spike count of the KC in E3's tuned bouton model, regenerated locally), k on a
grid, plus olfsysm's connectome-weighted KC-KC lateral inhibition (E8; `kc_kc_inh`, tau
0.05 s). Per k, Nelder-Mead (`optimize_mb_model_params`, two conditions: 'on' = the fitted
parameters, 'off' = the same with wAPLKC_scale = wAPLPN_scale = 0) over fixed_thr,
wAPLKC_scale (wKCAPL / wPNAPL tied), wAPLPN_scale and wKCKC_scale (= kc_kc_inh).

Loss (each term ~1 when bad, 0 when matched; weights in LOSS_WEIGHTS):
  on_mat    1 - Spearman(model APL on, control; 120 pairs, 1 - cosine over spike counts)
  off_mat   1 - Spearman(model APL off, TNT)
  block     ((block residual - 0.055) / 0.05)^2; E7's on-vs-off fruity x small-ester block
            (isotonic residual), data +0.055 [0.035, 0.075]: the discriminating target
  rate_on   ((mean APL-on response rate - 0.05) / 0.01)^2 (the usual 5 % tuning target)
  ratio     ((log(off / on rate ratio) - log(data ratio)) / log 1.5)^2; data = ratio of the
            mean response fractions of TNT and TNT3c over the 16 odors (2.2; E8 quoted 1.9)
  iso       max(0, 0.94 - isotonic R2 of off vs on) / 0.05 (data 0.94; monotone on/off relation)
  po_on     1 - Spearman(per-odor APL-on rates, TNT3c per-odor fractions)
  po_off    1 - Spearman(per-odor APL-off rates, TNT per-odor fractions)
Reported but not fitted: E7's density-null excess profile and E4's full score of each best fit.

Round 2 (--claw-thr T): the same on E16 / E19's bouton base with the rectified per-claw
offset T (olfsysm `claw_thr`, hal's replacement for the coincidence rule), starting from
E19's grid point (thr<T>_k<k>_kckc<kckc0>) when it exists; breadth then comes from hal's E16
APL-off run of that base (data/hal_outputs/E16_claw_saturation/, else regenerated here).

Outputs: <shared>/results/E10_joint_fit/ (fits.csv, history_k<k>.csv, best-fit score rows,
figure) and ./outputs/E10_joint_fit/ (responses / spike counts of each best fit).
Usage: OMP_NUM_THREADS=6 nice -n 10 ./venv/bin/python E10_joint_fit.py -k 3 1 [--maxfev 150]
       [--kckc0 60] [--claw-thr 40] [--method Nelder-Mead|direct|differential_evolution] [--smoke]
Glomerulus gain (E28's joint-fit arm): `--glom-gain 1.5 [--gain-glomeruli ...]` multiplies the
PN->claw weights of the E27 oracle glomeruli (default set) in the wiring passed to the fit;
tags get a `gain<G>_` prefix after the threshold prefix, and fits.csv / provenance record the
gain and glomeruli.
Second wiring draw (E30): `--wiring claw_permutation --wiring-seed N` replaces the base's claw x bouton
matrix by that shuffle (al_analysis.wpnkc_shuffles) before any gain; tag prefix wiring-<name>-seed<N>_.
Bound controls (E31): `--aplpn-max F`, `--aplpn-fixed F`, `--aplkc-min F` bound or fix the APL->bouton /
APL->KC scales relative to the base's tuned values (use --tag to name the variant).
What drives APL: `--apl-drive spikes` (E36) drives it by the KCs' spikes plus the bouton rates
(pn_claw_to_apl=False), from the spike-driven base given by --base-params; `--apl-drive spikes_only`
(E37) also removes the bouton->APL input (wPNAPL = 0): the same --base-params base, its APL scales
re-tuned to 5 % with wPNAPL at 0 (`tune_spike_only_base`), and the fit keeps wPNAPL at 0 (the
APL->bouton scale is still fitted); tag prefix spko_.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON, per_odor_extras, rate
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import e3_bouton_params, density_excess

EXPERIMENT = 'E10_joint_fit'
TAU = 0.05
# data targets come from the data-target registry (data_targets.py); 2026-11-07, before which
# the fits used hand-typed values (block 0.055, iso R2 0.94, structure 0.83, natural-source
# alignment 0.36, excess difference 0.02), each within 0.006 of the registry's
BLOCK_TARGET, BLOCK_SCALE = H.data_target('block'), 0.05
RATE_TARGET, RATE_SCALE = 0.05, 0.01
RATIO_LOG_SCALE = np.log(1.5)
ISO_TARGET, ISO_SCALE = H.data_target('iso_r2'), 0.05
LOSS_WEIGHTS = dict(on_mat=1.0, off_mat=1.0, block=1.0, rate_on=1.0, ratio=0.5, po_on=0.5,
    po_off=0.5, iso=0.25)
# E12's KC<->APL factor matching the breadth-weighted model's response count to k = 0's
# (spikes_k<k> rows of its stats): used only as the initial wAPLKC_scale per k
E12_FACTORS = {0.0: 1.0, 0.5: 0.82, 1.0: 0.65, 2.0: 0.36, 3.0: 0.18, 4.0: 0.085}
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')


def e19_stats() -> pd.DataFrame:
    return pd.read_csv(H.shared_dir() / 'results' / 'E19_thr_breadth_kckc' / 'stats_variants.csv')


def e19_row(claw_thr: float, k: float, kckc: float):
    """E19's grid point (claw_thr, k, KC-KC scale), or None."""
    s = e19_stats()
    m = s[s.tag == f'thr{claw_thr:g}_k{k:g}_kckc{kckc:g}']
    return None if len(m) == 0 else m.iloc[0]


def base_kws(claw_thr, apl_drive: str = 'claws') -> dict:
    """The bouton base's fit_mb_model kwargs; `apl_drive='spikes'` drives APL by the KCs'
    spikes (plus the bouton rates) instead of the summed claw input (pn_claw_to_apl=False;
    E35 / E36). 'spikes_only' (E37) has the same kwargs: the bouton->APL input is removed by
    its scale (wPNAPL_scale = 0), which is a parameter, not a kwarg."""
    kws = dict(BOUTON, make_plots=False, silent=True)
    if claw_thr is not None:
        kws['claw_thr'] = float(claw_thr)
    if apl_drive in ('spikes', 'spikes_only'):
        kws['pn_claw_to_apl'] = False
    elif apl_drive != 'claws':
        raise ValueError(f'{apl_drive=}')
    return kws


def read_params(spec: str) -> Dict[str, float]:
    """Absolute parameters from 'FILE:KEY' (a JSON file of {key: {param: value}}, e.g. a
    tuned_params.json; FILE relative to the shared directory unless absolute)."""
    path, key = spec.rsplit(':', 1)
    path = Path(path)
    if not path.is_absolute():
        path = H.shared_dir() / path
    d = json.loads(path.read_text())[key]
    return {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in d.items()
        if k in PARAM_KEYS + ('wKCKC_scale',)}


SPIKE_ONLY_TIED = ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale')


def tune_spike_only_base(orn: pd.DataFrame, claw_thr, spike_base: Dict[str, float], wnorm: dict,
    *, target: float = RATE_TARGET, tol: float = 0.02, max_iter: int = 30
    ) -> Tuple[Dict[str, float], dict]:
    """The spike-only base (E37): a spike-driven base with the bouton->APL input removed.

    wPNAPL_scale is set to 0 and the other three APL scales (APL->KC, KC->APL, APL->bouton) are
    multiplied by one factor, as olfsysm's tuning ties them, found by geometric bisection so
    that the APL-on response rate is within `tol` (relative) of `target`. The somatic
    threshold stays the spike-driven base's: olfsysm picks it with APL silent, so it does not
    depend on wPNAPL. Simulated on the base's own wiring (no gain or re-drawn wiring) with the
    normalized APL weights `wnorm` (unweighted by breadth), as the base was tuned.

    Args:
        orn: ORN input (`H.apl16_orn_deltas()`).
        claw_thr: per-claw threshold of the base (None for none).
        spike_base: absolute parameters of the spike-driven base (wPNAPL_scale > 0).
        wnorm: normalized APL weights of the base ({'wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL'}).

    Returns:
        The spike-only base's absolute parameters (PARAM_KEYS, wPNAPL_scale 0), and a dict with
        the factor applied to the tied scales, the APL-on response rate reached, and the
        number of simulations.
    """
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    if not spike_base['wPNAPL_scale'] > 0:
        raise ValueError('spike_base must be a spike-driven base with wPNAPL_scale > 0 (the '
            'spike-only base is re-tuned from it)')
    p0 = {k: float(spike_base[k]) for k in PARAM_KEYS}
    p0['wPNAPL_scale'] = 0.0
    sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p0, **base_kws(claw_thr, 'spikes_only'),
        _wAPLKC=wnorm['wAPLKC'], _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'],
        _wPNAPL=wnorm['wPNAPL']
    )
    rates: Dict[float, float] = {}

    def run(f: float) -> float:
        if f not in rates:
            r, _ = sim({k: p0[k] * f for k in SPIKE_ONLY_TIED})
            rates[f] = rate(r)
            print(f'  spike-only base: APL scales x{f:.4g} -> APL-on rate {rates[f]:.4f}', flush=True)
        return rates[f]

    # the rate falls as the factor rises; bracket the target, then bisect in log space
    lo = hi = 1.0
    while run(lo) < target:
        lo /= 2.0
        if lo < 1e-3:
            raise RuntimeError(f'APL-on rate below {target} even at APL scales x{lo:g}')
    while run(hi) > target:
        hi *= 2.0
        if hi > 1e3:
            raise RuntimeError(f'APL-on rate above {target} even at APL scales x{hi:g}')
    best = min(rates, key=lambda f: abs(rates[f] - target))
    for _ in range(max_iter):
        if abs(rates[best] - target) / target < tol:
            break
        f = float(np.sqrt(lo * hi))
        rk = run(f)
        if abs(rk - target) < abs(rates[best] - target):
            best = f
        if rk > target:
            lo = f
        else:
            hi = f
    del sim
    params = dict(p0, **{k: p0[k] * best for k in SPIKE_ONLY_TIED})
    info = dict(factor=best, rate_on=rates[best], target=target, tol=tol, n_sims=len(rates),
        spike_base={k: float(spike_base[k]) for k in PARAM_KEYS})
    return params, info


def bouton_base(orn: pd.DataFrame, out_dir: Path, claw_thr=None, gain: Optional[float] = None,
    gain_gloms=None, wiring: Optional[str] = None, wiring_seed: int = 0, apl_drive: str = 'claws',
    base_params: Optional[str] = None, info: Optional[dict] = None):
    """The bouton base at its tuned parameters (E3's; with `claw_thr`, E16's as recorded in
    E19's thr<T>_k0_kckc0 row): normalized APL weights, wPNKC, and the per-KC APL-off
    breadth (total spike count; hal's saved run if copied to the shared directory, else
    regenerated here).

    With `gain`, the PN->claw weights of the glomeruli in `gain_gloms` (default: the oracle
    set of the glomerulus-gain experiment, E27) are multiplied by `gain` (E27's `gained`);
    the returned wPNKC is the gained matrix (pass it to the fit as `_wPNKC`) and the breadth
    is recomputed from an APL-off run of the gained wiring at the base's parameters. The
    tuned base parameters themselves are not re-tuned at the gain: the joint fit fits the
    threshold and the APL / KC-KC scales, and `--init-from` sets the start.

    With `wiring` (a name from `al_analysis.wpnkc_shuffles.NAMED_SHUFFLES`, e.g.
    'claw_permutation' or 'curveball_within_type'), the base's claw x bouton matrix is
    replaced by that shuffle of it at `wiring_seed` (E4's substrate nulls: claw counts per
    KC and per bouton kept, the assignment re-drawn), before any gain; the breadth is
    recomputed from the re-drawn wiring. For replication of a fit on a second draw (E30).

    With `apl_drive='spikes'` the base's tuned parameters must come from `base_params`
    ('FILE:KEY', e.g. E36's spike-driven base tuned at 5 %); the APL-off breadth is the same
    as the claw-driven base's at the same threshold (APL is off), so the saved runs apply.
    With `apl_drive='spikes_only'` (E37) `base_params` is likewise the spike-driven base, and
    the returned parameters are the spike-only base re-tuned from it (`tune_spike_only_base`;
    the breadth is unchanged, since the threshold is); the tuning details go into `info`, if
    passed."""
    from al_analysis.mb_model import fit_mb_model
    hal_out = H.shared_dir() / 'data' / 'hal_outputs'
    if claw_thr is None:
        initial = e3_bouton_params()
        # hal's E3 APL-off run (the breadth E12 / E8 used)
        hal_path = hal_out / 'E3_drive_dependent_gain' / 'bouton_pnapl_x1_acute_off_spike_counts.parquet'
        off_path = out_dir / 'base_acute_off_spike_counts.parquet'
    else:
        r = e19_row(claw_thr, 0.0, 0.0)
        if r is None:
            raise ValueError(f'no E19 base row for claw_thr {claw_thr}')
        initial = {k: float(r[k]) for k in PARAM_KEYS}
        # hal's E16 APL-off run of this base (the breadth E19 used)
        hal_path = hal_out / 'E16_claw_saturation' / f'bouton_thr{claw_thr:g}_acute_off_spike_counts.parquet'
        off_path = out_dir / f'thr{claw_thr:g}_base_acute_off_spike_counts.parquet'
    if apl_drive in ('spikes', 'spikes_only'):
        if base_params is None:
            raise ValueError(f'apl_drive={apl_drive!r} needs base_params (FILE:KEY of a spike-driven base)')
        claw_initial = initial
        initial = read_params(base_params)
        if abs(initial['fixed_thr'] - claw_initial['fixed_thr']) > 1e-6 * claw_initial['fixed_thr']:
            # the saved APL-off breadth is at the claw-driven base's threshold
            hal_path = out_dir / 'no_such_file'
            off_path = out_dir / ('spk_' + off_path.name)
        print(f'spike-driven APL; base parameters from {base_params}: {initial}', flush=True)
    elif base_params is not None:
        initial = read_params(base_params)
    kws = base_kws(claw_thr, apl_drive)
    _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=initial['fixed_thr'],
        wAPLKC=initial['wAPLKC_scale'], wKCAPL=initial['wKCAPL_scale'],
        wAPLPN=initial['wAPLPN_scale'], wPNAPL=initial['wPNAPL_scale'], **kws
    )
    wnorm = {k: pd0[k] / pd0[f'{k}_scale'] for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
    if wiring is not None:
        from al_analysis.wpnkc_shuffles import NAMED_SHUFFLES, shuffle_wPNKC
        if wiring not in NAMED_SHUFFLES:
            raise ValueError(f'{wiring=} not in {list(NAMED_SHUFFLES)}')
        method, within = NAMED_SHUFFLES[wiring]
        wPNKC = shuffle_wPNKC(wPNKC, method, within_kc_type=within, seed=int(wiring_seed))
        kws = dict(kws, _wPNKC=wPNKC)
        hal_path = out_dir / 'no_such_file'
        off_path = out_dir / (f'wiring-{wiring}-seed{int(wiring_seed)}_' + off_path.name)
        print(f'wiring re-drawn: {wiring}, seed {wiring_seed}', flush=True)
    if gain is not None and gain != 1.0:
        from E27_glomerulus_gain import ORACLE, gained
        gloms = list(gain_gloms) if gain_gloms else list(ORACLE)
        missing = set(gloms) - set(wPNKC.columns.get_level_values('glomerulus'))
        if missing:
            raise ValueError(f'gain glomeruli not in the wiring: {sorted(missing)}')
        wPNKC = gained(wPNKC, gloms, gain)
        kws = dict(kws, _wPNKC=wPNKC)
        # the gained wiring's own APL-off breadth (hal's saved base runs do not apply)
        hal_path = out_dir / 'no_such_file'
        # the cache names the glomerulus set unless it is the oracle set (so two sets in one
        # outputs folder, e.g. E40's, do not share an APL-off run)
        gset = '' if sorted(gloms) == sorted(ORACLE) else '-'.join(sorted(gloms)) + '_'
        off_path = out_dir / (f'gain{gain:g}_{gset}' + off_path.name)
        print(f'gain {gain:g} on {gloms}', flush=True)
    if hal_path.exists():
        off_spk = pd.read_parquet(hal_path)
        print(f'breadth from {hal_path}', flush=True)
    elif off_path.exists():
        off_spk = pd.read_parquet(off_path)
    else:
        r_off, off_spk, _, _ = fit_mb_model(orn, fixed_thr=initial['fixed_thr'], wAPLKC=0.0,
            wKCAPL=initial['wKCAPL_scale'], wAPLPN=0.0, wPNAPL=initial['wPNAPL_scale'], **kws
        )
        off_spk.to_parquet(off_path)
        r_off.to_parquet(off_path.with_name(off_path.name.replace('spike_counts', 'responses')))
        print(f'base APL-off rate {rate(r_off):.4f} (E3 / E19 bases: 0.098)', flush=True)
    breadth = pd.Series(off_spk.astype(float).sum(axis='columns').to_numpy(),
        index=off_spk.index.get_level_values(H.KC_ID_LEVEL)
    )
    if apl_drive == 'spikes_only':
        t0 = time.time()
        initial, tune_info = tune_spike_only_base(orn, claw_thr, initial, wnorm)
        print(f'spike-only base (wPNAPL 0): APL scales x{tune_info["factor"]:.4f} of the spike-driven '
            f'base, APL-on rate {tune_info["rate_on"]:.4f} ({tune_info["n_sims"]} simulations, '
            f'{time.time() - t0:.0f}s): {initial}', flush=True)
        if info is not None:
            info.update(tune_info)
    return initial, wnorm, wPNKC, breadth


def breadth_weights(wAPLKC0: pd.Series, breadth: pd.Series, k: float) -> pd.Series:
    bp1 = breadth + 1.0
    f = ((bp1 / bp1.mean()) ** k).reindex(wAPLKC0.index.get_level_values(H.KC_ID_LEVEL))
    return wAPLKC0 * f.fillna(1.0).to_numpy()


STRUCTURE_WEIGHTS = dict(wt_dev=1.0, vcf_off=1.0, excess_on=0.5)
# data: structure beyond the ORN input vs wild-type somata (APL on), natural-source alignment
# of the APL-off residual, and block excess - other excess (APL on)
WT_DEV_TARGET, VCF_OFF_TARGET = H.data_target('ctrl_struct_wt'), H.data_target('tnt_vcf')
EXCESS_DIFF_TARGET, EXCESS_DIFF_SCALE = H.data_target('ctrl_excess_block_minus_other'), 0.05


def make_loss(data: dict, data_rates: Dict[str, pd.Series], orn_pairs: pd.Series,
    drive: pd.Series, ref: Optional[pd.DataFrame] = None, structure: bool = False):
    """Joint loss over {'on': (responses, spike_counts), 'off': ...} (see module doc).

    With `structure=True` (E25) three terms are added: the structure beyond the ORN input
    with APL on (rank agreement of the KC residual from a linear ORN prediction with the
    same residual in Remy's wild-type somata; data 0.82), the natural-source alignment of
    the APL-off residual (data 0.36), both as (target - value) / target clipped at 0, and
    the APL-on excess profile ((excess of the block pairs - excess of the other between-group
    pairs) vs the data's +0.024, scale 0.05); targets from the data-target registry.
    `ref` = `H.reference_pairs()`.
    """
    ratio_target = float(data_rates['TNT'].reindex(H.APL16_ODORS).mean() /
        data_rates['TNT3c'].reindex(H.APL16_ODORS).mean())
    weights = dict(LOSS_WEIGHTS, **(STRUCTURE_WEIGHTS if structure else {}))
    ctrl = data['raw']['TNT3c']
    tnt = data['raw']['TNT']
    v = H.PRIMARY_MODEL_RDM

    def loss_fn(outputs) -> Dict[str, float]:
        r_on, s_on = outputs['on']
        r_off, s_off = outputs['off']
        on = H.model_rdms(r_on, s_on)[v]
        off = H.model_rdms(r_off, s_off)[v]
        out = {}
        out['on_spearman'] = H.spearman(on, ctrl)
        out['off_spearman'] = H.spearman(off, tnt)
        try:
            st, _ = H.compare_on_off(on, off, orn=orn_pairs, drive=drive)
            out['block'] = float(st['block_fruity_small_iso'])
            out['iso_r2'] = float(st['iso_r2'])
        except Exception:
            out['block'], out['iso_r2'] = np.nan, np.nan
        out['rate_on'] = rate(r_on)
        out['rate_off'] = rate(r_off)
        out['ratio'] = out['rate_off'] / max(out['rate_on'], 1e-9)
        po_on = H.response_fractions(r_on).reindex(H.APL16_ODORS)
        po_off = H.response_fractions(r_off).reindex(H.APL16_ODORS)
        out['po_on_spearman'] = float(stats.spearmanr(po_on,
            data_rates['TNT3c'].reindex(H.APL16_ODORS))[0])
        out['po_off_spearman'] = float(stats.spearmanr(po_off,
            data_rates['TNT'].reindex(H.APL16_ODORS))[0])
        terms = dict(
            on_mat=1.0 - out['on_spearman'],
            off_mat=1.0 - out['off_spearman'],
            block=((np.nan_to_num(out['block'], nan=-0.2) - BLOCK_TARGET) / BLOCK_SCALE) ** 2,
            rate_on=((out['rate_on'] - RATE_TARGET) / RATE_SCALE) ** 2,
            ratio=((np.log(max(out['ratio'], 1e-3)) - np.log(ratio_target)) / RATIO_LOG_SCALE) ** 2,
            po_on=1.0 - np.nan_to_num(out['po_on_spearman'], nan=0.0),
            po_off=1.0 - np.nan_to_num(out['po_off_spearman'], nan=0.0),
            iso=max(0.0, ISO_TARGET - np.nan_to_num(out['iso_r2'], nan=0.0)) / ISO_SCALE,
        )
        if structure:
            rr_on = H.compare_to_references(on, ref)
            rr_off = H.compare_to_references(off, ref)
            ex = density_excess(s_on)
            out['wt_dev_on'] = float(rr_on['spearman_resid_vs_remap_wt'])
            out['vcf_off'] = float(rr_off['spearman_resid_vs_vcf_hyp'])
            out['excess_on_block'] = float(ex['excess_block'])
            out['excess_on_other'] = float(ex['excess_other_between'])
            terms['wt_dev'] = max(0.0, WT_DEV_TARGET - np.nan_to_num(out['wt_dev_on'], nan=0.0)) / WT_DEV_TARGET
            terms['vcf_off'] = max(0.0, VCF_OFF_TARGET - np.nan_to_num(out['vcf_off'], nan=0.0)) / VCF_OFF_TARGET
            terms['excess_on'] = ((out['excess_on_block'] - out['excess_on_other'] - EXCESS_DIFF_TARGET)
                / EXCESS_DIFF_SCALE) ** 2
        for name, t in terms.items():
            out[f'term_{name}'] = float(weights[name] * t)
        out['loss'] = float(sum(out[f'term_{n}'] for n in terms))
        return {kk: float(vv) for kk, vv in out.items()}

    loss_fn.ratio_target = ratio_target
    return loss_fn


def init_from_fit(tag: str, base: Dict[str, float], experiment: str = 'E10_joint_fit'
    ) -> Dict[str, float]:
    """Starting point taken from an earlier fit's row in fits.csv: its threshold, APL and
    APL->bouton factors relative to its own base are applied to `base` (so a solution found
    without a per-claw threshold can seed a fit with one), with its KC-KC scale."""
    f = pd.read_csv(H.results_dir(experiment) / 'fits.csv')
    r = f[f.tag == tag]
    if len(r) == 0:
        raise ValueError(f'no fit {tag!r} in fits.csv')
    r = r.iloc[0]
    return dict(fixed_thr=base['fixed_thr'] * float(r['thr_factor']),
        wAPLKC_scale=base['wAPLKC_scale'] * float(r['apl_factor_vs_e3']),
        wKCAPL_scale=base['wKCAPL_scale'] * float(r['apl_factor_vs_e3']),
        wAPLPN_scale=base['wAPLPN_scale'] * float(r['aplpn_factor_vs_e3']),
        wPNAPL_scale=base['wPNAPL_scale'], wKCKC_scale=float(r['kc_kc_inh']),
    )


def fit_one(k: float, orn, initial, wnorm, breadth, loss_fn, *, kckc0: float, maxfev: int,
    method: str, thr0_factor: float, claw_thr=None, init_from: Optional[str] = None,
    init_from_experiment: str = 'E10_joint_fit', verbose: bool = True,
    wPNKC_override: Optional[pd.DataFrame] = None, init_thr: Optional[float] = None,
    aplpn_max: Optional[float] = None, aplpn_fixed: Optional[float] = None,
    aplkc_min: Optional[float] = None, apl_drive: str = 'claws',
    init_params: Optional[str] = None):
    from al_analysis.mb_model import optimize_mb_model_params
    wAPLKC = breadth_weights(wnorm['wAPLKC'], breadth, k)
    r19 = None if (claw_thr is None or apl_drive != 'claws') else e19_row(claw_thr, k, kckc0)
    if init_params is not None:
        init = read_params(init_params)
        init.setdefault('wKCKC_scale', kckc0)
        kckc0 = init['wKCKC_scale']
        print(f'starting from {init_params}: {init}', flush=True)
    elif init_from is not None:
        init = init_from_fit(init_from, initial, init_from_experiment)
        kckc0 = init['wKCKC_scale']
        print(f'starting from fit {init_from}: {init}', flush=True)
    elif r19 is not None:
        # start at E19's grid point (APL scales matched to the k = 0 count, threshold
        # re-tuned to 5 % with this KC-KC scale)
        init = {kk: float(r19[kk]) for kk in PARAM_KEYS}
        init['wKCKC_scale'] = kckc0
        print(f'starting from E19 row {r19.tag}', flush=True)
    else:
        f0 = E12_FACTORS.get(k, float(np.exp(np.interp(k, sorted(E12_FACTORS),
            np.log([E12_FACTORS[x] for x in sorted(E12_FACTORS)])))))
        init = dict(fixed_thr=initial['fixed_thr'] * thr0_factor,
            wAPLKC_scale=initial['wAPLKC_scale'] * f0, wKCAPL_scale=initial['wKCAPL_scale'] * f0,
            wAPLPN_scale=initial['wAPLPN_scale'], wPNAPL_scale=initial['wPNAPL_scale'],
            wKCKC_scale=kckc0,
        )
    if init_thr is not None:
        # a warm start at a known threshold (e.g. the one tuned for a gained wiring)
        init['fixed_thr'] = float(init_thr)
        print(f'start threshold overridden: {init_thr} Hz', flush=True)
    bounds = {
        'fixed_thr': (initial['fixed_thr'] / 4, initial['fixed_thr'] * 4),
        'wAPLKC_scale': (init['wAPLKC_scale'] / 30, init['wAPLKC_scale'] * 30),
        'wAPLPN_scale': (init['wAPLPN_scale'] / 30, init['wAPLPN_scale'] * 30),
        'wKCKC_scale': (0.0, 300.0),
    }
    # bound controls (E31): the APL->bouton and APL->KC scales relative to the base's tuned
    # values (`initial`, E3's for the plain bouton base, E16's for a threshold base)
    params = ['fixed_thr', 'wAPLKC_scale', 'wAPLPN_scale', 'wKCKC_scale']
    tie = True
    if aplpn_fixed is not None:
        init['wAPLPN_scale'] = initial['wAPLPN_scale'] * aplpn_fixed
        init['wPNAPL_scale'] = initial['wPNAPL_scale'] * aplpn_fixed
        params.remove('wAPLPN_scale'); bounds.pop('wAPLPN_scale')
        # tie only the KC->APL scale to the APL->KC scale: with the default (True) the
        # non-optimized bouton scales would be tied to it too and move with it
        tie = ('wKCAPL_scale',)
        print(f'APL->bouton scale fixed at {aplpn_fixed:g}x the base ({init["wAPLPN_scale"]:.4g})', flush=True)
    elif aplpn_max is not None:
        hi = initial['wAPLPN_scale'] * aplpn_max
        init['wAPLPN_scale'] = min(init['wAPLPN_scale'], hi)
        bounds['wAPLPN_scale'] = (min(bounds['wAPLPN_scale'][0], hi / 30), hi)
        print(f'APL->bouton scale bounded at <= {aplpn_max:g}x the base ({hi:.4g}); start {init["wAPLPN_scale"]:.4g}', flush=True)
    if aplkc_min is not None:
        lo = initial['wAPLKC_scale'] * aplkc_min
        init['wAPLKC_scale'] = max(init['wAPLKC_scale'], lo)
        init['wKCAPL_scale'] = init['wAPLKC_scale'] * (initial['wKCAPL_scale'] / initial['wAPLKC_scale'])
        bounds['wAPLKC_scale'] = (lo, max(bounds['wAPLKC_scale'][1], lo * 30))
        print(f'APL->KC scale bounded at >= {aplkc_min:g}x the base ({lo:.4g}); start {init["wAPLKC_scale"]:.4g}', flush=True)
    if apl_drive == 'spikes_only':
        # no bouton->APL input (E37): wPNAPL stays at 0, and only the KC->APL scale is tied
        # to the APL->KC scale (the starting point may come from a fit that had the input)
        init['wPNAPL_scale'] = 0.0
        tie = ('wKCAPL_scale',)
    opt_kws = {}
    if method == 'differential_evolution':
        opt_kws = dict(popsize=6, seed=0, polish=False, init='sobol')
    # (only with a glomerulus gain, so the default path is unchanged)
    wiring_kws = {} if wPNKC_override is None else {'_wPNKC': wPNKC_override}
    res = optimize_mb_model_params(orn, loss_fn=loss_fn,
        condition_param_overrides={'on': {}, 'off': {'wAPLKC_scale': 0.0, 'wAPLPN_scale': 0.0}},
        params_to_optimize=params,
        initial_params=init, bounds=bounds, tie_apl_weights=tie, method=method,
        maxfev=maxfev, optimizer_kws=opt_kws, verify=False, verbose=verbose,
        **base_kws(claw_thr, apl_drive), **wiring_kws, _wAPLKC=wAPLKC, _wKCAPL=wnorm['wKCAPL'],
        _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=kckc0, kc_kc_tau=TAU
    )
    return res, init, bounds


def make_figure(res_dir: Path, data: dict) -> None:
    """Loss histories per fit, best-fit statistics vs k, and the best fit's matrices."""
    plt = H._plt()
    fits = pd.read_csv(res_dir / 'fits.csv').sort_values(['k', 'tag'])
    hists = {f.stem.replace('history_', ''): pd.read_csv(f) for f in sorted(res_dir.glob('history_*.csv'))}
    v = H.PRIMARY_MODEL_RDM
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax = axes[0, 0]
    for tag, h in hists.items():
        ax.plot(h.index, h['loss'].cummin(), label=tag)
    ax.set(xlabel='evaluation', ylabel='best loss so far', title='Nelder-Mead progress', yscale='log')
    ax.legend(fontsize=7)
    panels = [
        (f'{v}__block_fruity_small_iso', 'fruity x small-ester block (on vs off)', 0.055, (0.035, 0.075)),
        (f'{v}__on_spearman', 'rank agreement with the control matrix (APL on)', None, None),
        (f'{v}__off_spearman', 'rank agreement with the APL-blocked matrix (APL off)', None, None),
        ('rate_ratio_off_over_on', 'response rate, APL off relative to on', 2.2, None),
        ('kc_kc_inh', 'fitted KC-KC scale (kc_kc_inh)', None, None),
    ]
    for ax, (col, title, target, band) in zip(axes.flat[1:], panels):
        grp = (fits.get('claw_thr', pd.Series(np.nan, index=fits.index)).fillna(0).map(
            lambda t: 'no claw_thr' if t == 0 else f'claw_thr {t:g}') + ', ' +
            fits.tag.str.contains('start2').map({False: 'start 1', True: 'start 2'}))
        for name, sub in fits.groupby(grp):
            ax.plot(sub.k, sub[col], 'o-', label=name)
        if target is not None:
            ax.axhline(target, color='k', ls='--', lw=1, label='data')
        if band is not None:
            ax.axhspan(*band, color='k', alpha=0.08)
        ax.set(xlabel='breadth exponent k', title=title)
        ax.legend(fontsize=7)
    fig.suptitle('Joint fit of the bouton model (breadth-weighted APL, KC-KC inhibition, optional per-claw threshold) to the control and APL-blocked matrices', fontsize=11)
    fig.tight_layout()
    fig.savefig(res_dir / 'E10_joint_fit.png', dpi=130)
    plt.close(fig)
    # best fit overall: on / off model vs data and on-vs-off
    best = fits.sort_values('loss').iloc[0]
    long = pd.read_csv(res_dir / 'model_pair_distances_long.csv')
    sub = long[long.tag == best.tag]
    pairs = {c: g.set_index(['odor_a', 'odor_b'])['distance'] for c, g in sub.groupby('condition')}
    if len(pairs) == 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        on, off = pairs['on'], pairs['acute_off']
        H.plot_model_vs_data(axes[0], on, data['raw']['TNT3c'], label=f'{best.tag} APL on')
        axes[0].set_title(f'APL on vs control (rank agreement {best[v + "__on_spearman"]:.2f})')
        H.plot_model_vs_data(axes[1], off, data['raw']['TNT'], label=f'{best.tag} APL off')
        axes[1].set_title(f'APL off vs APL-blocked data (rank agreement {best[v + "__off_spearman"]:.2f})')
        H.plot_on_vs_off(axes[2], on, off, label=best.tag)
        axes[2].set_title(f'on vs off (block {best[v + "__block_fruity_small_iso"]:+.3f}, iso R2 {best[v + "__iso_r2"]:.2f})')
        fig.suptitle(f'Best joint fit: {best.tag} (loss {best.loss:.2f})', fontsize=11)
        fig.tight_layout()
        fig.savefig(res_dir / 'E10_best_fit.png', dpi=130)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-k', '--exponents', nargs='+', type=float, default=[3.0, 2.0, 1.0, 0.0])
    ap.add_argument('--maxfev', type=int, default=150)
    ap.add_argument('--kckc0', type=float, default=60.0, help='initial KC-KC scale')
    ap.add_argument('--thr0-factor', type=float, default=0.7,
        help='initial threshold as a factor of E3 bouton threshold (E8: 0.6-0.77 at KC-KC 30-100)')
    ap.add_argument('--claw-thr', type=float, default=None,
        help='round 2: rectified per-claw offset (olfsysm claw_thr; E16 / E19), e.g. 40 or 70')
    ap.add_argument('--method', default='Nelder-Mead')
    ap.add_argument('--tag', default='', help='suffix for this run (e.g. a second start)')
    ap.add_argument('--experiment', default=EXPERIMENT,
        help='results / outputs folder name (E25 uses E25_joint_fit_structure)')
    ap.add_argument('--loss', choices=['base', 'structure'], default='base',
        help="'structure' adds the structure-beyond-ORN and APL-on excess-profile terms (E25)")
    ap.add_argument('--init-from-experiment', default='E10_joint_fit',
        help='experiment whose fits.csv --init-from reads')
    ap.add_argument('--init-from', default=None, metavar='TAG',
        help='start from an earlier fit (row TAG of fits.csv): its relative threshold / APL factors on this base, its KC-KC scale')
    ap.add_argument('--glom-gain', type=float, default=None, metavar='G',
        help="multiply the PN->claw weights of the gain glomeruli by G (the oracle gain of "
        "E27 / E28; tag prefix gain<G>_)")
    ap.add_argument('--gain-glomeruli', nargs='+', default=None, metavar='GLOM',
        help='glomeruli the gain applies to (default: the E27 oracle set: DC2 DL1 DM1 DM5 D DP1m VA7m VC1)')
    ap.add_argument('--wiring', default=None, metavar='SHUFFLE',
        help="re-draw the base's claw x bouton wiring with this al_analysis.wpnkc_shuffles "
        "shuffle (claw_permutation, claw_permutation_within_type, curveball, "
        "curveball_within_type; E30's second draw); tag prefix wiring-<SHUFFLE>-seed<N>_")
    ap.add_argument('--wiring-seed', type=int, default=0, help='seed of the --wiring draw')
    ap.add_argument('--init-thr', type=float, default=None, metavar='HZ',
        help='start the somatic threshold here (overrides the seed / --thr0-factor; e.g. the '
        'threshold tuned for a gained wiring)')
    ap.add_argument('--aplpn-max', type=float, default=None, metavar='F',
        help='bound the APL->bouton scale at <= F x the base\'s tuned value (E31)')
    ap.add_argument('--aplpn-fixed', type=float, default=None, metavar='F',
        help='fix the APL->bouton scale at F x the base\'s tuned value and fit the other three (E31)')
    ap.add_argument('--aplkc-min', type=float, default=None, metavar='F',
        help='bound the APL->KC scale at >= F x the base\'s tuned value (E31)')
    ap.add_argument('--apl-drive', default='claws', choices=['claws', 'spikes', 'spikes_only'],
        help="what drives APL: 'claws' (the bouton base's summed claw input), 'spikes' (KC "
        "spikes plus the bouton rates, pn_claw_to_apl=False; E36; needs --base-params; tag prefix spk_) "
        "or 'spikes_only' (KC spikes only, the bouton->APL scale wPNAPL held at 0, from the "
        "--base-params spike-driven base re-tuned to 5 %% with wPNAPL 0; E37; tag prefix spko_)")
    ap.add_argument('--base-params', default=None, metavar='FILE:KEY',
        help='tuned base parameters (absolute) from a JSON file (relative to the shared directory), '
        'e.g. results/E36_spike_apl/tuned_params.json:spk_base_thr70; the reference for '
        '--aplpn-max / --aplpn-fixed / --aplkc-min and the *_factor_vs_e3 columns (with '
        '--apl-drive spikes_only, the spike-only base re-tuned from it is the reference)')
    ap.add_argument('--init-params', default=None, metavar='FILE:KEY',
        help='start from these absolute parameters (fixed_thr, the APL scales, wKCKC_scale), e.g. '
        'results/E36_spike_apl/tuned_params.json:thr70_k2_kckc100_g1.5')
    ap.add_argument('--smoke', action='store_true', help='time a few evaluations and exit')
    ap.add_argument('--figure-only', action='store_true', help='only redraw the figures from saved results')
    args = ap.parse_args()

    res_dir = H.results_dir(args.experiment)
    out_dir = H.outputs_dir(args.experiment)
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {
        'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')},
    }
    data_rates = {g: H.yang_response_fractions(g).mean() for g in ('TNT3c', 'TNT')}
    if args.figure_only:
        make_figure(res_dir, data)
        return
    # code provenance at the start: the checkout can move (a pull) while the fits run
    start_prov = dict(al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.olfsysm_dir(),
            capture_output=True, text=True).stdout.strip(),
        model=H.model_provenance(orn_deltas=orn))
    loss_fn = make_loss(data, data_rates, orn_pairs, drive, ref=ref, structure=(args.loss == 'structure'))
    print(f'data off/on rate ratio target {loss_fn.ratio_target:.3f}; loss variant {args.loss}', flush=True)

    t0 = time.time()
    gain = None if args.glom_gain in (None, 1.0) else float(args.glom_gain)
    base_info = {}
    initial, wnorm, wPNKC, breadth = bouton_base(orn, out_dir, args.claw_thr, gain=gain,
        gain_gloms=args.gain_glomeruli, wiring=args.wiring, wiring_seed=args.wiring_seed,
        apl_drive=args.apl_drive, base_params=args.base_params, info=base_info
    )
    wiring_tag = '' if args.wiring is None else f'wiring-{args.wiring}-seed{args.wiring_seed}_'
    gain_gloms = None
    if gain is not None:
        from E27_glomerulus_gain import ORACLE
        gain_gloms = list(args.gain_glomeruli) if args.gain_glomeruli else list(ORACLE)
    print(f'bouton base ready ({time.time() - t0:.0f}s; claw_thr {args.claw_thr}; gain {gain}; '
        f'wiring {args.wiring}); tuned params {initial}', flush=True)
    prefix = {'spikes': 'spk_', 'spikes_only': 'spko_'}.get(args.apl_drive, '') + ('' if args.claw_thr is None else f'thr{args.claw_thr:g}_') + wiring_tag + (
        '' if gain is None else f'gain{gain:g}_')

    fits_path = res_dir / 'fits.csv'
    for k in args.exponents:
        tag = f'{prefix}k{k:g}{args.tag}'
        t0 = time.time()
        res, init, bounds = fit_one(k, orn, initial, wnorm, breadth, loss_fn, kckc0=args.kckc0,
            maxfev=5 if args.smoke else args.maxfev, method=args.method,
            thr0_factor=args.thr0_factor, claw_thr=args.claw_thr, init_from=args.init_from,
            init_from_experiment=args.init_from_experiment,
            wPNKC_override=(wPNKC if (gain is not None or args.wiring is not None) else None),
            init_thr=args.init_thr, aplpn_max=args.aplpn_max, aplpn_fixed=args.aplpn_fixed,
            aplkc_min=args.aplkc_min, apl_drive=args.apl_drive, init_params=args.init_params
        )
        hist = res['history']
        hist.to_csv(res_dir / f'history_{tag}.csv')
        if args.smoke:
            print(hist[['loss', 'time_s']].describe().T, flush=True)
            print(hist.iloc[-1].to_dict(), flush=True)
            return
        r_on, s_on = res['condition_outputs']['on']
        r_off, s_off = res['condition_outputs']['off']
        for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
            r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
            sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        p = res['params']
        meta = dict(variant=tag, tag=tag, base='bouton', seed=None, k=k, claw_thr=args.claw_thr,
            gain=gain, gain_glomeruli=(','.join(gain_gloms) if gain_gloms else None),
            wiring=args.wiring, wiring_seed=(args.wiring_seed if args.wiring else None),
            method=args.method,
            maxfev=args.maxfev, n_evals=len(hist), kckc0=args.kckc0, thr0_factor=args.thr0_factor,
            init_from=args.init_from, init_thr=args.init_thr, loss_variant=args.loss,
            apl_drive=args.apl_drive, base_params=args.base_params, init_params=args.init_params,
            aplpn_max=args.aplpn_max, aplpn_fixed=args.aplpn_fixed, aplkc_min=args.aplkc_min,
            fixed_thr=p['fixed_thr'], thr_factor=p['fixed_thr'] / initial['fixed_thr'],
            wAPLKC_scale=p['wAPLKC_scale'], wKCAPL_scale=p['wKCAPL_scale'],
            wAPLPN_scale=p['wAPLPN_scale'], wPNAPL_scale=p['wPNAPL_scale'],
            kc_kc_inh=p['wKCKC_scale'], kc_kc_tau=TAU,
            apl_factor_vs_e3=p['wAPLKC_scale'] / initial['wAPLKC_scale'],
            aplpn_factor_vs_e3=p['wAPLPN_scale'] / initial['wAPLPN_scale'],
            # (E37) the re-tuned spike-only base the *_factor columns are relative to
            **({f'base_{kk}': vv for kk, vv in initial.items()} if args.apl_drive == 'spikes_only' else {}),
            **(dict(spike_only_base_factor=base_info['factor']) if base_info else {}),
            loss=res['loss'], initial_loss=res['initial_losses']['loss'],
            n_kcs=len(r_on), n_glomeruli=wPNKC.shape[1], rate_on=rate(r_on),
            rate_off=rate(r_off), rate_ratio_off_over_on=rate(r_off) / max(rate(r_on), 1e-9),
            seconds=time.time() - t0,
            mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()),
        )
        meta.update({f'fit_{kk}': vv for kk, vv in res['losses'].items() if kk != 'loss'})
        row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs, drive,
            data, ref
        )
        row.update(per_odor_extras(r_on, r_off, on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM],
            drive, data_rates
        ))
        for cond, sp in (('on', s_on), ('off', s_off)):
            row.update({f'{cond}_{kk}': vv for kk, vv in density_excess(sp).items()})
        # re-read right before writing: two processes (different k) share this file
        rows = list(pd.read_csv(fits_path).to_dict('records')) if fits_path.exists() else []
        rows = [r for r in rows if r.get('tag') != tag] + [row]
        pd.DataFrame(rows).to_csv(fits_path, index=False)
        pd.concat({(tag, 'on'): on[H.PRIMARY_MODEL_RDM], (tag, 'acute_off'): off[H.PRIMARY_MODEL_RDM]},
            names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index(
            ).to_csv(res_dir / 'model_pair_distances_long.csv', index=False, mode='a',
            header=not (res_dir / 'model_pair_distances_long.csv').exists()
        )
        v = H.PRIMARY_MODEL_RDM
        print(f"{tag} done: loss {res['loss']:.3f} (initial {meta['initial_loss']:.3f}, "
            f"{len(hist)} evals, {meta['seconds']/60:.0f} min); thr x{meta['thr_factor']:.3f}, "
            f"APL x{meta['apl_factor_vs_e3']:.3f}, APL>PN x{meta['aplpn_factor_vs_e3']:.3f}, "
            f"KC-KC {meta['kc_kc_inh']:.1f}; rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f} "
            f"(x{meta['rate_ratio_off_over_on']:.2f}); block {row[v + '__block_fruity_small_iso']:+.3f}; "
            f"vs ctrl {row[v + '__on_spearman']:.3f} dev {row[v + '__on_spearman_dev']:+.2f}; off vs TNT "
            f"{row[v + '__off_spearman']:.3f}; iso R2 {row[v + '__iso_r2']:.3f}; excess on "
            f"{row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f}; per-odor rates "
            f"{row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}", flush=True
        )
        prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            script='E10_joint_fit.py', host=subprocess.run(['hostname'], capture_output=True,
                text=True).stdout.strip(),
            **start_prov,
            base=base_kws(args.claw_thr, args.apl_drive), apl_drive=args.apl_drive,
            init_params=args.init_params, claw_thr=args.claw_thr, gain=gain, gain_glomeruli=gain_gloms,
            wiring=args.wiring, wiring_seed=(args.wiring_seed if args.wiring else None),
            aplpn_max=args.aplpn_max, aplpn_fixed=args.aplpn_fixed, aplkc_min=args.aplkc_min,
            base_params=initial, loss_variant=args.loss,
            loss_weights=dict(LOSS_WEIGHTS, **(STRUCTURE_WEIGHTS if args.loss == 'structure' else {})),
            block_target=BLOCK_TARGET,
            block_scale=BLOCK_SCALE, rate_target=RATE_TARGET, ratio_target=loss_fn.ratio_target,
            ratio_log_scale=RATIO_LOG_SCALE, iso_target=ISO_TARGET, iso_scale=ISO_SCALE,
            **(dict(wt_dev_target=WT_DEV_TARGET, vcf_off_target=VCF_OFF_TARGET,
                excess_diff_target=EXCESS_DIFF_TARGET, excess_diff_scale=EXCESS_DIFF_SCALE)
                if args.loss == 'structure' else {}),
            data_targets_sha256=hashlib.sha256(H.data_targets_path().read_bytes()).hexdigest(),
            tau=TAU, bounds={kk: list(vv) for kk, vv in bounds.items()}, initial=init,
            approx_conc_note=H.APPROX_CONC_NOTE,
            # (E37: how the spike-only base in base_params was re-tuned from --base-params)
            **(dict(spike_only_base_tuning=base_info) if base_info else {}),
        )
        (res_dir / f'provenance_{tag}.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    try:
        make_figure(res_dir, data)
    except Exception as e:
        print('figure failed:', repr(e)[:200])
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
