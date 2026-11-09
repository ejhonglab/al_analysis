#!/usr/bin/env python3
"""E51 part A: where the fitted fruity x small-ester block lives (which KCs carry it, how close
to threshold they sit, and which pairs' excess survives the trial-to-trial noise).

Models: the reference model at 6.8 % (E44_operating_point, thr40_gain1.5_rate0.068_k2_cap3) and
the eight glomeruli at 1.25 at 70 Hz (E45_gain_at_ctrl_frac, thr70_gain1.25_rate0.068_k2_cap3),
rebuilt at their fitted parameters by E43_model_trial_noise.py's `fit_model` / `NoisyModel` (as
E48's pipeline did); the deterministic APL-on and acute APL-off runs are checked against the fits'
saved outputs (0 mismatching entries required) and against the fits.csv response rates.

1. Pair decomposition. For each of the nine block pairs ({2h, IaA, pa} x {2-but, eb, ep}) and each
   condition, the cosine distance of the deterministic spike-count vectors x, y (over KCs) is split
   exactly by KC: with u = x / |x| and v = y / |y|, 1 - cos = sum_i (u_i - v_i)^2 / 2, so each
   KC's term c_i >= 0 is its contribution to the distance. KCs are classed by the model's response
   rule (>= 2 spikes): responding to both odors, to exactly one (the pair's "carriers": they add to
   one norm and nothing to the dot product), or to neither. Reported per pair: the counts, each
   class's share of the distance, of the dot product and of the norms, and the change from APL off
   to APL on (sum_i (c_on_i - c_off_i), exact) split into the KCs that separate the pair with APL
   on (of which "APL-made": responding to both with APL off), those responding to both with APL on,
   and the rest. The pair's excess is its APL-on distance's residual from the isotonic fit on the
   APL-off distance over the 120 pairs (as E48 / `kc_remapping_harness.compare_on_off`).
2. Threshold margins. The deterministic runs are repeated with olfsysm's `save_vm_sims` (spike
   counts checked to reproduce exactly), once with the thresholds out of reach (no KC spikes) and
   once as they are. olfsysm integrates Vm(t) = Vm(t-1) + (D(t) - Vm(t-1)) dt / taum and resets Vm
   to exactly 0 at a spike, where D(t) is the KC's summed claw input (after the per-claw knee and
   the APL inhibition at claws and boutons) minus its KC-KC inhibition; so D(t) = Vm(t-1) + (Vm(t)
   - Vm(t-1)) taum / dt wherever Vm(t) was not reset. APL is driven by the claws here, not by KC
   spikes, so the no-spike run gives the drive without KC-KC inhibition at every step; the KC-KC
   inhibition is rebuilt from the real run's spikes (read off the Vm resets, checked against the
   spike counts) and rv.kc.wKCKC as olfsysm computes it, and the full drive is checked against the
   one recovered from the real run's Vm wherever that is defined. Each KC's drive is then replayed
   through the same integration at a threshold f x thr, for f by bisection, giving its critical
   factor f*: it responds (>= 2 spikes) below f*, not above, the other KCs' spikes (their KC-KC
   inhibition) held as in the deterministic run and the KC's own self-inhibition replayed with its
   own spikes (the replay at f = 1 is checked against the spike counts). The margin is f* - 1, in
   the units of E48's jitter, which multiplies each threshold by ratio x (1 + sigma xi) (sigma 0.207
   and 0.23, the APL-on-matched values; ratio the re-tuned mean threshold, 1.066 and 1.041): the
   entry's probability of changing in a jittered trial with the rest of the network fixed is
   Phi((f* / ratio - 1) / sigma) for a non-response and 1 minus that for a response (p_change). The
   peak drive relative to threshold is reported too, but it is a poor margin: the drive peaks
   briefly at odor onset, faster than Vm (taum 10 ms) follows, so many non-responders peak above
   their threshold. Margin distributions per class (all responders, the carriers' responded and
   other odor, the both-responders, the APL-made carriers), with the percentile of each carrier's
   margin among all responders to the same odor (0.5 = typical) and the spike counts.
3. The pair excess per pseudo-recording: E48's part-B trials (outputs/E48_block_target_noise/
   partB/trials, the APL-on-matched sigma; 10 pseudo-recordings x 3 trials, APL on and off on the
   same threshold draws), each pseudo-recording processed as the KC data (three-trial means, cosine
   over KCs; noise_floor_trial_distances.analyse), the excess per pair from that recording's APL-on
   against its own APL-off distances ('paired') and against the mean APL-off distances over the
   recordings ('vs mean APL off', the construction the KC data allow: control and APL-blocked
   recordings are different flies). The KC data per control-fly recording against the mean
   APL-output-blocked (TNT) distances for reference.

Outputs: <shared>/results/E51_noise_robust_block/partA/ (block_carriers.csv, carrier_margins.csv,
pair_excess_per_recording.csv, pair_excess_summary.csv, E51_pair_excess_per_recording.png,
E51_carrier_margins.png, provenance.json with the checks, README.md); ./outputs/
E51_noise_robust_block/partA/ (per model: <model>_kc_odor.parquet, the per-KC x odor x condition
spike counts, responses, thresholds, critical factors, peak and odor-window mean drive; the checks;
the breadth runs' bouton_base caches). With the per-KC tables present the models are not rebuilt
(--resim rebuilds them).
Usage: nice -n 10 ./venv/bin/python E51_block_carriers.py [--resim] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scipy import stats as sstats

import kc_remapping_harness as H
import noise_floor_trial_distances as NF
import E43_model_trial_noise as E43

EXPERIMENT = 'E51_noise_robust_block'
SUB = 'partA'
# (label, experiment, fits.csv tag)
MODELS = [
    ('the reference model at 6.8 %', 'E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3'),
    ('the eight glomeruli at 1.25 at 70 Hz', 'E45_gain_at_ctrl_frac', 'thr70_gain1.25_rate0.068_k2_cap3'),
]
ODORS = list(H.APL16_ODORS)
FRUITY, SMALL = ('2h', 'IaA', 'pa'), ('2-but', 'eb', 'ep')
BLOCK_PAIRS = [(a, b) for a in FRUITY for b in SMALL]
BLOCK_ODORS = list(FRUITY + SMALL)
CONDS = ('on', 'acute_off')
COND_LABEL = {'on': 'APL on', 'acute_off': 'APL off'}
E48 = 'E48_block_target_noise'
E48_OUT = H.HERE / 'outputs' / E48 / 'partB'
E48_RES = H.shared_dir() / 'results' / E48 / 'partB'
LEVEL = 'match_on'
GRADED = 'graded'
MARGIN_NOTE = ('margin = f* - 1, f* the factor on a KC\'s threshold at which its response to an odor (>= 2 spikes) appears or '
    'disappears, found by replaying the KC\'s exact drive D(t) (summed claw input after the per-claw knee and APL inhibition, '
    'minus the KC-KC inhibition from the other KCs\' spikes in the deterministic run; its own self-inhibition replayed with its '
    'own spikes) through olfsysm\'s integration at thresholds f x thr (bisection); the other KCs held as in the deterministic '
    'run. p_change: the probability that the entry changes in one of E48\'s jittered trials (threshold thr x ratio x (1 + sigma '
    'xi)), the rest of the network held fixed: Phi((f* / ratio - 1) / sigma) for a non-response, 1 minus that for a response.')


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair_name(a: str, b: str) -> str:
    return f'{a} / {b}'


# ---------------------------------------------------------------------------
# 1 + 2: the deterministic runs with the effective drive
def drive_from_vm(V: np.ndarray, taum: float, dt: float) -> np.ndarray:
    """D(t) = Vm(t-1) + (Vm(t) - Vm(t-1)) taum / dt for the steps 1.. of a KCs x steps Vm array
    (exact where no spike reset Vm(t))."""
    prev, cur = V[:, :-1], V[:, 1:]
    return prev + (cur - prev) * (taum / dt)


def replay_counts(D: np.ndarray, w_self: np.ndarray, th: np.ndarray, k: float, decay: float, tau: float) -> np.ndarray:
    """Spike counts of independent KCs (rows of D, the drive without their own KC-KC
    self-inhibition) integrated as olfsysm does (Vm += (D - w_self a - Vm) dt / taum, spike and
    reset to 0 above `th`; a the KC's own low-pass filtered spikes, as olfsysm's kc_kc_activity)."""
    n_kc, n_t = D.shape
    v, a = np.zeros(n_kc), np.zeros(n_kc)
    sp = np.zeros(n_kc, dtype=bool)
    n = np.zeros(n_kc, dtype=np.int64)
    self_inh = bool(np.any(w_self))
    for t in range(n_t):
        if self_inh:
            a = decay * a + sp / tau
            v = v + (D[:, t] - w_self * a - v) * k
        else:
            v = v + (D[:, t] - v) * k
        sp = v > th
        n += sp
        v[sp] = 0.0
    return n


def critical_factor(D: np.ndarray, w_self: np.ndarray, thr: np.ndarray, n_resp: int, k: float, decay: float, tau: float,
        n_iter: int = 32) -> np.ndarray:
    """Per KC, the threshold factor f* at which it stops (or starts) responding (>= n_resp
    spikes): responds for f < f*, not above (by bisection; 0 if it does not respond even at a
    threshold of 1e-6 x its own, i.e. no positive drive). The drive D is held fixed: the other KCs'
    spikes (their KC-KC inhibition) as in the deterministic run."""
    hi = np.maximum(D.max(axis=1) / thr, 1.0) * (1 + 1e-9) + 1e-12
    lo = np.full(len(thr), 1e-6)
    ok_lo = replay_counts(D, w_self, lo * thr, k, decay, tau) >= n_resp
    if (replay_counts(D, w_self, hi * thr, k, decay, tau) >= n_resp).any():
        raise RuntimeError('a KC responds with its threshold above its peak drive')
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        r = replay_counts(D, w_self, mid * thr, k, decay, tau) >= n_resp
        lo, hi = np.where(r, mid, lo), np.where(r, hi, mid)
    return np.where(ok_lo, 0.5 * (lo + hi), 0.0)


def simulate_model(label: str, experiment: str, tag: str, orn: pd.DataFrame, out_dir: Path) -> tuple:
    """Builds one model at its fitted parameters, checks its deterministic run, reconstructs each
    KC's drive per odor and condition, and finds its critical threshold factor. Returns (per-KC x
    odor x condition table, checks dict)."""
    from scipy.signal import lfilter
    from al_analysis.mb_model import get_time_index
    name = E43.model_from(f'{experiment}:{tag}')
    t0 = time.time()
    m = E43.NoisyModel(name, orn, out_dir)
    checks = dict(model=name, label=label, experiment=experiment, tag=tag, build_seconds=time.time() - t0, **m.info)
    # the deterministic run against the fit's saved outputs (as E43's Session.get)
    saved = {}
    for cond, (rp, sp) in m.saved.items():
        if not (rp.exists() and sp.exists()):
            raise FileNotFoundError(f'{name}: no saved outputs to check against ({rp})')
        pr, ps = E43.to_arrays(pd.read_parquet(rp), pd.read_parquet(sp))
        saved[cond] = dict(file=str(rp), responses_mismatch=int((pr != m.det[cond][0]).sum()),
            spike_counts_mismatch=int((ps != m.det[cond][1]).sum()), responses_sha256=sha256(rp), spike_counts_sha256=sha256(sp))
    checks['det_vs_saved'] = saved
    rate_det = {c: float(m.det[c][0].mean()) for c in CONDS}
    checks['rate_det'] = rate_det
    checks['rate_det_minus_fits_csv'] = {c: rate_det[c] - r for c, r in zip(CONDS, m.fit_rates)}
    print(f'{name}: built in {checks["build_seconds"]:.0f}s; deterministic vs saved {saved}', flush=True)
    bad = [c for c, v in saved.items() if v['responses_mismatch'] or v['spike_counts_mismatch']]
    assert not bad, f'{name}: the deterministic run differs from the saved outputs ({saved})'
    assert all(abs(d) < 1e-9 for d in checks['rate_det_minus_fits_csv'].values()), checks['rate_det_minus_fits_csv']
    mp, rv = m.mp, m.rv
    if int(mp.kc.n_claws_active_to_spike) >= 0:
        raise NotImplementedError('per-claw spiking thresholds (n_claws_active_to_spike >= 0)')
    if float(mp.kc.ves_p) != 0.0:
        raise NotImplementedError('KC vesicle depletion (mp.kc.ves_p != 0): the KC-KC drive would need nves')
    if not bool(mp.kc.pn_claw_to_apl):
        raise NotImplementedError('APL driven by KC spikes (pn_claw_to_apl False): the no-spike run would change APL')
    taum, dt = float(mp.kc.taum), float(mp.time_dt)
    k = dt / taum
    kckc = bool(mp.kc.use_kc_kc_inh)
    tau = float(mp.kc.kc_kc_tau) if kckc else 1.0
    decay = 1.0 - dt / tau
    W = rv.kc.wKCKC.tocsr() if kckc else None
    w_self = W.diagonal().copy() if kckc else np.zeros(len(m.det['on'][0]))
    t = np.asarray(get_time_index(mp, rv), dtype=float)
    win = (t >= mp.time_stim_start) & (t < mp.time_stim_end)
    # (olfsysm's KC loop starts at the step of time_start + 1; Vm is 0 before)
    i0 = int(np.searchsorted(t, mp.time_start - dt / 2)) - 2
    checks.update(taum=taum, dt=dt, n_steps=int(len(t)), first_step_kept=i0, odor_window=[float(mp.time_stim_start),
        float(mp.time_stim_end)], kc_kc_inh=kckc, kc_kc_tau=tau, wKCKC_nnz=int(W.nnz) if kckc else 0,
        wKCKC_diag_nonzero=int((w_self != 0).sum()), n_spikes_for_response=int(mp.kc.n_spikes_for_response))
    n_resp = int(mp.kc.n_spikes_for_response)
    w = win[i0 + 1:]
    rows = []
    mp.kc.save_vm_sims = True
    try:
        for cond, p in m.params.items():
            t1 = time.time()
            # 1. no KC spikes (threshold out of reach): the drive without KC-KC inhibition, exactly
            r0, s0 = E43.to_arrays(*m.sim(dict(p, fixed_thr=1e12)))
            assert s0.sum() == 0
            vms = rv.kc.vm_sims
            if len(vms) != len(m.pn_idx) or vms[0].shape[1] != len(t):
                raise RuntimeError(f'unexpected vm_sims layout: {len(vms)} x {vms[0].shape}')
            d_free = {}
            for j, o in enumerate(ODORS):
                i = m.pn_idx[j]
                V = np.asarray(vms[i])
                vms[i] = None
                assert not V[:, :i0].any(), 'Vm nonzero before the KC loop'
                d_free[o] = drive_from_vm(V[:, i0:], taum, dt)
            del vms
            # 2. the real run: spikes from the resets, the KC-KC inhibition from them, the full drive
            r, s = E43.to_arrays(*m.sim(p))
            mism = int((r != m.det[cond][0]).sum() + (s != m.det[cond][1]).sum())
            checks[f'vm_run_mismatch_vs_det_{cond}'] = mism
            assert mism == 0, f'{name} {cond}: the run with the voltages saved differs from the deterministic run ({mism})'
            thr = np.asarray(rv.kc.thr, dtype=float).ravel()
            vms = rv.kc.vm_sims
            n_bad_spikes, max_dev, n_bad_replay = 0, 0.0, 0
            for j, o in enumerate(ODORS):
                i = m.pn_idx[j]
                V = np.asarray(vms[i])[:, i0:]
                vms[i] = None
                prev, cur = V[:, :-1], V[:, 1:]
                spike = (cur == 0) & (prev != 0)
                n_bad_spikes += int((spike.sum(axis=1) != s[:, j]).sum())
                D = d_free.pop(o)
                if kckc:
                    a = lfilter([0.0, 1.0 / tau], [1.0, -decay], spike.astype(float), axis=1)
                    D = D - W @ a
                # the reconstruction against the drive recovered from this run's Vm (steps without a reset)
                obs = ~spike & ~((cur == 0) & (prev == 0))
                dev = np.abs(D - drive_from_vm(V, taum, dt))[obs]
                max_dev = max(max_dev, float(dev.max() / thr.min()) if dev.size else 0.0)
                # the KC's own self-inhibition is replayed with its own spikes
                D_ext = D + w_self[:, None] * a if kckc else D
                n1 = replay_counts(D_ext, w_self, thr, k, decay, tau)
                n_bad_replay += int((n1 != s[:, j]).sum())
                f_crit = critical_factor(D_ext, w_self, thr, n_resp, k, decay, tau)
                mean_w = D[:, w].mean(axis=1)
                rows.append(pd.DataFrame(dict(model=name, condition=cond, kc=np.arange(len(thr)), odor=o,
                    spikes=s[:, j].astype(int), response=r[:, j], thr=thr, crit_factor=f_crit, replay_spikes=n1,
                    peak_drive_rel=D.max(axis=1) / thr - 1, mean_drive_odor_rel=mean_w / thr - 1)))
            del vms
            checks[f'n_kc_odor_spike_count_from_vm_resets_mismatch_{cond}'] = n_bad_spikes
            checks[f'drive_reconstruction_max_abs_dev_over_thr_{cond}'] = max_dev
            checks[f'n_kc_odor_replay_spike_count_mismatch_{cond}'] = n_bad_replay
            assert n_bad_spikes == 0, f'{name} {cond}: spike counts from the Vm resets differ from the model\'s in {n_bad_spikes} entries'
            assert max_dev < 1e-6, f'{name} {cond}: the reconstructed drive deviates from the Vm-derived one by {max_dev} x thr'
            assert n_bad_replay <= 5, f'{name} {cond}: the replay at the model\'s thresholds misses {n_bad_replay} spike counts'
            print(f'{name}: {COND_LABEL[cond]} drive reconstructed and replayed in {time.time() - t1:.0f}s (spike counts from '
                f'resets: {n_bad_spikes} mismatches; drive max deviation {max_dev:.2g} x thr; replay at f = 1: {n_bad_replay} '
                'mismatches)', flush=True)
    finally:
        mp.kc.save_vm_sims = False
    # the flag off again: the deterministic run is unchanged
    r, s = E43.to_arrays(*m.sim(m.params['on']))
    checks['after_mismatch_on'] = int((r != m.det['on'][0]).sum() + (s != m.det['on'][1]).sum())
    assert checks['after_mismatch_on'] == 0
    m.close()
    kc = pd.concat(rows, ignore_index=True)
    kc['margin'] = kc['crit_factor'] - 1
    inconsistent = (kc['response'] != (kc['margin'] > 0))
    checks['n_kc_odor_margin_sign_vs_response_mismatch'] = int(inconsistent.sum())
    return kc, checks


# ---------------------------------------------------------------------------
# 1: pair decomposition
def kc_terms(spk: np.ndarray, ja: int, jb: int):
    """(per-KC distance terms c_i, distance, dot-product terms, squared norms) of odors ja, jb."""
    x, y = spk[:, ja].astype(float), spk[:, jb].astype(float)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return None
    u, v = x / nx, y / ny
    c = 0.5 * (u - v) ** 2
    d = 1 - float(u @ v)
    assert abs(c.sum() - d) < 1e-9, (c.sum(), d)
    return c, d, x * y, x ** 2, y ** 2


def classes(resp: np.ndarray, ja: int, jb: int) -> np.ndarray:
    ra, rb = resp[:, ja], resp[:, jb]
    return np.where(ra & rb, 'both', np.where(ra, 'only_a', np.where(rb, 'only_b', 'neither')))


def pct_rank_within_odor(kc: pd.DataFrame) -> pd.Series:
    """Per responding KC x odor entry, the percentile of its margin among all responders to that
    odor in the same model and condition (mid-rank; 0.5 = typical)."""
    out = pd.Series(np.nan, index=kc.index)
    r = kc[kc['response']]
    for _, g in r.groupby(['model', 'condition', 'odor']):
        out.loc[g.index] = g['margin'].rank(method='average', pct=True) - 0.5 / len(g)
    return out


def decompose(name: str, kc: pd.DataFrame, sigma: float, excess_det: pd.Series) -> tuple:
    """block_carriers rows for one model, and the (KC, odor) index sets of each margin class."""
    spk = {c: kc[kc.condition == c].pivot(index='kc', columns='odor', values='spikes')[ODORS].to_numpy() for c in CONDS}
    resp = {c: kc[kc.condition == c].pivot(index='kc', columns='odor', values='response')[ODORS].to_numpy().astype(bool) for c in CONDS}
    marg = {c: kc[kc.condition == c].pivot(index='kc', columns='odor', values='margin')[ODORS].to_numpy() for c in CONDS}
    pct = {c: kc[kc.condition == c].pivot(index='kc', columns='odor', values='pct_rank')[ODORS].to_numpy() for c in CONDS}
    pch = {c: kc[kc.condition == c].pivot(index='kc', columns='odor', values='p_change')[ODORS].to_numpy() for c in CONDS}
    sets = {c: {k: set() for k in ('carrier_resp', 'carrier_other', 'both', 'apl_made_resp', 'apl_made_other')} for c in CONDS}
    rows = []
    terms = {}
    for a, b in BLOCK_PAIRS:
        ja, jb = ODORS.index(a), ODORS.index(b)
        for cond in CONDS:
            tt = kc_terms(spk[cond], ja, jb)
            if tt is None:
                raise RuntimeError(f'{name} {cond}: no spikes to {a} or {b}')
            c, d, dot, xx, yy = tt
            cl = classes(resp[cond], ja, jb)
            terms[(a, b, cond)] = (c, d, cl)
            sep = (cl == 'only_a') | (cl == 'only_b')
            both = cl == 'both'
            # the carriers' margins: on the odor responded to, and on the other odor
            m_resp = np.where(cl == 'only_a', marg[cond][:, ja], marg[cond][:, jb])[sep]
            m_other = np.where(cl == 'only_a', marg[cond][:, jb], marg[cond][:, ja])[sep]
            s_resp = np.where(cl == 'only_a', spk[cond][:, ja], spk[cond][:, jb])[sep]
            p_resp = np.where(cl == 'only_a', pct[cond][:, ja], pct[cond][:, jb])[sep]
            q_resp = np.where(cl == 'only_a', pch[cond][:, ja], pch[cond][:, jb])[sep]
            q_other = np.where(cl == 'only_a', pch[cond][:, jb], pch[cond][:, ja])[sep]
            for k in np.flatnonzero(sep):
                jr, jo = (ja, jb) if cl[k] == 'only_a' else (jb, ja)
                sets[cond]['carrier_resp'].add((k, ODORS[jr]))
                sets[cond]['carrier_other'].add((k, ODORS[jo]))
            for k in np.flatnonzero(both):
                sets[cond]['both'].update({(k, a), (k, b)})
            row = dict(model=name, condition=cond, condition_label=COND_LABEL[cond], pair=pair_name(a, b), odor_a=a, odor_b=b,
                distance=d, excess=float(excess_det[(a, b)]) if cond == 'on' else np.nan,
                n_resp_a=int(resp[cond][:, ja].sum()), n_resp_b=int(resp[cond][:, jb].sum()), n_both=int(both.sum()),
                n_only_a=int((cl == 'only_a').sum()), n_only_b=int((cl == 'only_b').sum()), n_carriers=int(sep.sum()),
                carrier_frac_of_union=float(sep.sum() / max(sep.sum() + both.sum(), 1)),
                carrier_share_of_distance=float(c[sep].sum() / d), both_share_of_distance=float(c[both].sum() / d),
                rest_share_of_distance=float(c[~sep & ~both].sum() / d),
                both_share_of_dot=float(dot[both].sum() / dot.sum()) if dot.sum() > 0 else np.nan,
                only_a_share_of_norm_a=float(xx[cl == 'only_a'].sum() / xx.sum()),
                only_b_share_of_norm_b=float(yy[cl == 'only_b'].sum() / yy.sum()),
                carrier_margin_resp_median=float(np.median(m_resp)) if len(m_resp) else np.nan,
                carrier_margin_other_median=float(np.median(m_other)) if len(m_other) else np.nan,
                carrier_frac_resp_margin_lt_sigma=float((m_resp < sigma).mean()) if len(m_resp) else np.nan,
                carrier_frac_other_margin_gt_minus_sigma=float((m_other > -sigma).mean()) if len(m_other) else np.nan,
                carrier_spikes_median=float(np.median(s_resp)) if len(s_resp) else np.nan,
                carrier_frac_2_spikes=float((s_resp == 2).mean()) if len(s_resp) else np.nan,
                carrier_pct_rank_within_odor_median=float(np.nanmedian(p_resp)) if len(p_resp) else np.nan,
                carrier_p_drop_mean=float(q_resp.mean()) if len(q_resp) else np.nan,
                carrier_p_other_joins_mean=float(q_other.mean()) if len(q_other) else np.nan,
                carrier_p_separation_kept_mean=float(((1 - q_resp) * (1 - q_other)).mean()) if len(q_resp) else np.nan,
                both_p_drop_mean=float(np.r_[pch[cond][both, ja], pch[cond][both, jb]].mean()) if both.any() else np.nan,
                both_margin_median=float(np.median(np.r_[marg[cond][both, ja], marg[cond][both, jb]])) if both.any() else np.nan,
                sigma=sigma)
            rows.append(row)
        # the change from APL off to APL on, split by KC class
        c_on, d_on, cl_on = terms[(a, b, 'on')]
        c_off, d_off, cl_off = terms[(a, b, 'acute_off')]
        dc = c_on - c_off
        sep_on = (cl_on == 'only_a') | (cl_on == 'only_b')
        sep_off = (cl_off == 'only_a') | (cl_off == 'only_b')
        made = sep_on & (cl_off == 'both')
        both_on = cl_on == 'both'
        for k in np.flatnonzero(made):
            jr, jo = (ja, jb) if cl_on[k] == 'only_a' else (jb, ja)
            sets['on']['apl_made_resp'].add((k, ODORS[jr]))
            sets['on']['apl_made_other'].add((k, ODORS[jo]))
        m_made = np.where(cl_on == 'only_a', marg['on'][:, ja], marg['on'][:, jb])[made]
        m_made_other = np.where(cl_on == 'only_a', marg['on'][:, jb], marg['on'][:, ja])[made]
        delta = d_on - d_off
        assert abs(dc.sum() - delta) < 1e-9
        rows[-2].update(delta_distance=delta, delta_from_carriers=float(dc[sep_on].sum()),
            delta_from_apl_made_carriers=float(dc[made].sum()), delta_from_both=float(dc[both_on].sum()),
            delta_from_rest=float(dc[~sep_on & ~both_on].sum()),
            carrier_share_of_delta=float(dc[sep_on].sum() / delta) if delta != 0 else np.nan,
            n_carriers_off=int(sep_off.sum()), n_carriers_both_conditions=int((sep_on & sep_off).sum()),
            n_apl_made_carriers=int(made.sum()), n_carriers_lost=int((sep_off & ~sep_on).sum()),
            apl_made_margin_resp_median=float(np.median(m_made)) if made.any() else np.nan,
            apl_made_margin_other_median=float(np.median(m_made_other)) if made.any() else np.nan,
            apl_made_frac_other_margin_gt_minus_sigma=float((m_made_other > -sigma).mean()) if made.any() else np.nan)
    return rows, sets


MARGIN_CLASSES = [
    ('all responders (16 odors)', None),
    ('responders to the six block odors', None),
    ('non-responders to the six block odors', None),
    ('carriers: the odor responded to', 'carrier_resp'),
    ('carriers: the other odor', 'carrier_other'),
    ('both-responders of a block pair', 'both'),
    ('APL-made carriers: the odor responded to', 'apl_made_resp'),
    ('APL-made carriers: the other odor', 'apl_made_other'),
]


def summarize_class(g: pd.DataFrame, sigma: float) -> dict:
    m, s = g['margin'].to_numpy(), g['spikes'].to_numpy()
    if not len(g):
        return dict(n=0)
    q = np.quantile(m, [0.1, 0.25, 0.5, 0.75, 0.9])
    return dict(n=int(len(g)), n_kcs=int(g['kc'].nunique()), margin_mean=float(m.mean()), margin_q10=q[0], margin_q25=q[1],
        margin_median=q[2], margin_q75=q[3], margin_q90=q[4], frac_abs_margin_lt_sigma=float((np.abs(m) < sigma).mean()),
        frac_margin_0_to_sigma=float(((m >= 0) & (m < sigma)).mean()), frac_margin_minus_sigma_to_0=float(((m < 0) & (m > -sigma)).mean()),
        frac_margin_gt_0=float((m > 0).mean()), spikes_mean=float(s.mean()), spikes_median=float(np.median(s)),
        frac_2_spikes=float((s == 2).mean()), frac_le_3_spikes=float((s <= 3).mean()),
        frac_responding=float(g['response'].mean()), p_change_jitter_mean=float(g['p_change'].mean()),
        pct_rank_within_odor_median=float(g['pct_rank'].median()) if g['pct_rank'].notna().any() else np.nan,
        peak_drive_rel_median=float(g['peak_drive_rel'].median()), mean_drive_odor_rel_median=float(g['mean_drive_odor_rel'].median()))


def margin_rows(name: str, label: str, kc: pd.DataFrame, sets: dict, sigma: float) -> list:
    rows = []
    for cond in CONDS:
        k = kc[kc.condition == cond].set_index(['kc', 'odor'], drop=False)
        for cls, key in MARGIN_CLASSES:
            if key is None:
                if cls.startswith('all responders'):
                    g = k[k['response']]
                elif cls.startswith('responders'):
                    g = k[k['response'] & k['odor'].isin(BLOCK_ODORS)]
                else:
                    g = k[~k['response'] & k['odor'].isin(BLOCK_ODORS)]
            else:
                idx = sorted(sets[cond][key])
                if not idx:
                    continue
                g = k.loc[idx]
            rows.append(dict(model=name, label=label, condition=cond, condition_label=COND_LABEL[cond], kc_class=cls, sigma=sigma,
                **summarize_class(g, sigma)))
    return rows


# ---------------------------------------------------------------------------
# 3: the pair excess per pseudo-recording
def recording_pairs(arrs: dict, cond: str) -> pd.Series:
    """Raw (three-trial mean) cosine pair distances of one pseudo-recording, as the KC data's."""
    recs, _ = E43.recordings([arrs], cond, GRADED)
    _, pr = NF.analyse(recs[0], GRADED)
    return pr['raw']


def block_excess(on: pd.Series, off: pd.Series) -> pd.Series:
    resid, _ = H.isotonic_resid(on, off)
    return resid.reindex(BLOCK_PAIRS)


def per_recording_excess(name: str, label: str, sigma: float, n_rec: int, n_tr: int, inputs: dict) -> tuple:
    on, off = [], []
    for rep in range(n_rec):
        p = E43.trials_path(E48_OUT, name, sigma, n_tr, rep, E43.B_KEY)
        a = E43.load_npz(p)
        inputs[str(p)] = sha256(p)
        on.append(recording_pairs(a, 'on'))
        off.append(recording_pairs(a, 'acute_off'))
    mean_on, mean_off = pd.concat(on, axis=1).mean(axis=1), pd.concat(off, axis=1).mean(axis=1)
    rows = []
    for rep in range(n_rec):
        e_p = block_excess(on[rep], off[rep])
        e_m = block_excess(on[rep], mean_off)
        for (a, b) in BLOCK_PAIRS:
            rows.append(dict(model=name, label=label, recording=f'rep{rep:02d}', pair=pair_name(a, b), odor_a=a, odor_b=b, sigma=sigma,
                excess_paired=float(e_p[(a, b)]), excess_vs_mean_off=float(e_m[(a, b)]),
                distance_on=float(on[rep][(a, b)]), distance_off=float(off[rep][(a, b)])))
        rows.append(dict(model=name, label=label, recording=f'rep{rep:02d}', pair='block (mean of the nine)', sigma=sigma,
            excess_paired=float(e_p.mean()), excess_vs_mean_off=float(e_m.mean())))
    three_trial_means = block_excess(mean_on, mean_off)
    return pd.DataFrame(rows), three_trial_means, (mean_on, mean_off)


def data_per_recording(inputs: dict) -> tuple:
    """The KC data: per control-fly recording, the excess against the mean APL-output-blocked (TNT)
    distances; and the measured excess (mean against mean)."""
    raws = {}
    for g in ('TNT3c', 'TNT'):
        p = H.yang_data_dir() / g / f'dataset_{NF.DATASETS[g]}_long.csv'
        inputs[str(p)] = sha256(p)
        recs = NF.recording_arrays(NF.load_trials(g))
        raws[g] = {rec: NF.analyse(by_odor, GRADED)[1]['raw'] for rec, by_odor in recs.items()}
    mean_off = pd.concat(raws['TNT'].values(), axis=1).mean(axis=1)
    mean_on = pd.concat(raws['TNT3c'].values(), axis=1).mean(axis=1)
    rows = []
    for rec, on in raws['TNT3c'].items():
        e = block_excess(on, mean_off)
        rid = '_'.join(map(str, rec))
        for (a, b) in BLOCK_PAIRS:
            rows.append(dict(model='KC data', label='KC data (control flies against APL output blocked (TNT), mean)', recording=rid,
                pair=pair_name(a, b), odor_a=a, odor_b=b, excess_paired=np.nan, excess_vs_mean_off=float(e[(a, b)]),
                distance_on=float(on.get((a, b), np.nan)), distance_off=float(mean_off[(a, b)])))
        rows.append(dict(model='KC data', label='KC data (control flies against APL output blocked (TNT), mean)', recording=rid,
            pair='block (mean of the nine)', excess_vs_mean_off=float(e.mean())))
    return pd.DataFrame(rows), block_excess(mean_on, mean_off)


def excess_summary(per_rec: pd.DataFrame, det: dict, three: dict, data_measured: pd.Series) -> pd.DataFrame:
    rows = []
    for (model, pair), g in per_rec.groupby(['model', 'pair'], sort=False):
        r = dict(model=model, label=g['label'].iloc[0], pair=pair, n_recordings=int(len(g)))
        for col in ('excess_paired', 'excess_vs_mean_off'):
            v = g[col].dropna().to_numpy()
            if len(v):
                r.update({f'{col}_mean': float(v.mean()), f'{col}_sd': float(v.std(ddof=1)),
                    f'{col}_frac_positive': float((v > 0).mean()), f'{col}_min': float(v.min()), f'{col}_max': float(v.max())})
        if model in det:
            if pair.startswith('block'):
                r.update(excess_deterministic=float(det[model].mean()), excess_three_trial_means=float(three[model].mean()))
            else:
                key = tuple(pair.split(' / '))
                r.update(excess_deterministic=float(det[model][key]), excess_three_trial_means=float(three[model][key]))
        else:
            r['excess_measured'] = float(data_measured.mean()) if pair.startswith('block') else float(data_measured[tuple(pair.split(' / '))])
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
def make_figures(out: Path, per_rec: pd.DataFrame, summ: pd.DataFrame, kc_all: dict, sets_all: dict, sigmas: dict,
        labels: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs = [pair_name(a, b) for a, b in BLOCK_PAIRS]
    x = np.arange(len(pairs))
    names = [n for n in labels]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4.6 * (len(names) + 1), 4.6), sharey=True)
    rng = np.random.default_rng(0)
    ylo, yhi = -0.25, 0.17
    for ax, model in zip(axes, names + ['KC data']):
        g = per_rec[(per_rec.model == model) & per_rec.pair.isin(pairs)]
        s = summ[summ.model == model].set_index('pair').reindex(pairs)
        ax.axhline(0, color='0.6', lw=0.8)
        cols = [('excess_paired', '#2a78d6', -0.17, 'per pseudo-recording, against its own APL off'),
            ('excess_vs_mean_off', '#1baf7a', 0.17, 'per recording, against the mean APL off')]
        if model == 'KC data':
            cols = [('excess_vs_mean_off', '0.45', 0.0, 'per control-fly recording, against the mean\nAPL output blocked (TNT)')]
        clip_labelled = False
        for col, color, dx, lab in cols:
            for i, p in enumerate(pairs):
                v = g.loc[g.pair == p, col].dropna().to_numpy()
                xs = i + dx + rng.uniform(-0.05, 0.05, len(v))
                inside = v >= ylo
                ax.plot(xs[inside], v[inside], 'o', ms=3.2, color=color, alpha=0.55, mec='none', label=lab if i == 0 else None)
                # (values below the axis drawn at its edge)
                lab_clip = None
                if (~inside).any() and not clip_labelled:
                    lab_clip, clip_labelled = 'below the axis (drawn at its edge)', True
                ax.plot(xs[~inside], np.full((~inside).sum(), ylo + 0.006), 'v', ms=4, color=color, mfc='none', label=lab_clip)
            ax.plot(x + dx, s[f'{col}_mean'], '_', color=color, ms=13, mew=2)
        if model == 'KC data':
            ax.plot(x, s['excess_measured'], 'D', color='k', ms=5, mfc='none', label='measured (mean against mean)')
            ax.set_title('KC data', fontsize=9)
        else:
            ax.plot(x, s['excess_deterministic'], 'o', color='k', ms=6, label='deterministic run')
            ax.plot(x, s['excess_three_trial_means'], 's', color='#eb6834', ms=5, mfc='none',
                label="from the recordings' mean distances (E48)")
            ax.set_title(f'{labels[model]}\nmatched threshold jitter σ = {sigmas[model]:g}', fontsize=9)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=60, fontsize=7.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, axis='y', color='0.92', lw=0.6)
        ax.legend(fontsize=6.5, frameon=False, loc='lower right')
    axes[0].set_ylabel('excess of the APL-on distance over the\nisotonic trend from the APL-off distance')
    fig.suptitle('The nine fruity × small-ester pairs, recording by recording (10 pseudo-recordings of 3 trials; '
        'bars: means)', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / 'E51_pair_excess_per_recording.png', dpi=130)
    plt.close(fig)

    # margins: the carriers against all responders (APL on)
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 4.0 * len(names)))
    axes = np.atleast_2d(axes)
    import matplotlib.ticker as mticker
    spec = [('all responders (16 odors)', '0.35', '-'), ('carriers: the odor responded to', '#2a78d6', '-'),
        ('carriers: the other odor', '#eb6834', '-'), ('both-responders of a block pair', '#1baf7a', '-'),
        ('APL-made carriers: the odor responded to', '#6250d6', '--'), ('APL-made carriers: the other odor', '#c8553d', '--')]
    short = {'all responders (16 odors)': 'all responders', 'both-responders of a block pair': 'both-responders'}
    for row, model in zip(axes, names):
        k = kc_all[model]
        k = k[k.condition == 'on'].set_index(['kc', 'odor'], drop=False)
        sg = sigmas[model]
        for cls, color, ls in spec:
            if cls.startswith('all responders'):
                g = k[k['response']]
            else:
                key = dict(MARGIN_CLASSES)[cls]
                idx = sorted(sets_all[model]['on'][key])
                if not idx:
                    continue
                g = k.loc[idx]
            m = np.sort(g['margin'].to_numpy())
            lab = short.get(cls, cls.replace(': the ', ', '))
            row[0].step(m, np.arange(1, len(m) + 1) / len(m), where='post', color=color, ls=ls, lw=1.6, label=f'{lab} (n = {len(m)})')
            if not cls.endswith('other odor'):
                sp = np.sort(g['spikes'].to_numpy())
                row[1].step(sp, np.arange(1, len(sp) + 1) / len(sp), where='post', color=color, ls=ls, lw=1.6,
                    label=f'{lab} (median {np.median(sp):g})')
        for v in (-sg, sg):
            row[0].axvline(v, color='0.5', lw=0.8, ls=':')
        row[0].axvline(0, color='0.3', lw=0.8)
        row[0].text(sg, 0.02, ' +σ', fontsize=7, color='0.4')
        row[0].text(-sg, 0.02, ' −σ', fontsize=7, color='0.4')
        row[0].set_xlim(-1.0, 1.5)
        row[0].set_xlabel('margin: critical threshold factor − 1 (APL on)')
        row[0].set_ylabel('cumulative fraction')
        row[0].set_title(f'{labels[model]}: threshold margins (σ = {sg:g}, the matched jitter)', fontsize=9)
        row[1].set_xscale('log')
        row[1].xaxis.set_major_locator(mticker.FixedLocator([2, 3, 5, 10, 20, 50]))
        row[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
        row[1].xaxis.set_minor_formatter(mticker.NullFormatter())
        row[1].set_xlabel('spike count (APL on; responders: >= 2 spikes)')
        row[1].set_title(f'{labels[model]}: spike counts of the responses', fontsize=9)
        for ax in row:
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(True, color='0.92', lw=0.6)
        row[0].legend(fontsize=6.5, frameon=False, loc='center right')
        row[1].legend(fontsize=6.5, frameon=False, loc='lower right')
    fig.suptitle('Threshold margins with APL on: the factor on a KC\'s threshold at which its response appears or disappears, '
        'minus 1 (other KCs held fixed)', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / 'E51_carrier_margins.png', dpi=130)
    plt.close(fig)


README = """# E51 part A: where the fitted fruity × small-ester block lives

Made by `E51_block_carriers.py` (al_analysis {commit}), {generated}. Models: {models}, rebuilt at their fitted
parameters (E43_model_trial_noise.py's `fit_model`), the deterministic runs checked against the fits' saved outputs
(0 mismatching entries; provenance.json, "checks"). The nine pairs: {{2h, IaA, pa}} × {{2-but, eb, ep}}. A pair's
excess: its APL-on distance's residual from the isotonic fit on the APL-off distance over the 120 pairs (the block is
the mean of the nine). Response: >= 2 spikes. paa, euc and -aPine use ORN input at substituted concentrations (not
among the block odors, but in the 120 pairs the isotonic fit uses).

- `block_carriers.csv`: per model, condition (APL on / APL off) and pair: the deterministic cosine distance and excess;
  KCs responding to each odor, to both (`n_both`), to exactly one (`n_only_a`, `n_only_b`, `n_carriers`: the KCs that
  separate the two odors); each class's share of the distance (exact split: 1 − cos = Σ_i (u_i − v_i)² / 2 with u, v
  the unit-normalized spike-count vectors), of the dot product and of the norms; on the APL-on rows, the change of
  the distance from APL off to APL on (`delta_distance`) split into the KCs separating the pair with APL on
  (`delta_from_carriers`, of which `delta_from_apl_made_carriers`: KCs responding to both with APL off),
  both-responders and the rest, the carriers' share of that change, the counts of carriers with APL off, in both
  conditions, made by APL and lost; and the carriers' margins (median margin on the odor responded to and on the
  other odor, the fraction within σ of threshold, spike counts, percentile among all responders to the same odor)
  and their mean probabilities, in a jittered trial with the rest of the network fixed, of dropping out
  (`carrier_p_drop_mean`), of the other odor joining (`carrier_p_other_joins_mean`) and of the separation surviving
  (`carrier_p_separation_kept_mean`); `both_p_drop_mean` for the both-responders.
- `carrier_margins.csv`: per model, condition and KC class (all responders; responders and non-responders to the six
  block odors; the carriers' responded and other odor; both-responders; APL-made carriers), over unique KC × odor
  entries: margin quantiles, the fraction within ±σ of threshold, spike-count summaries, the mean probability of
  changing in a jittered trial (`p_change_jitter_mean`), the median percentile among all responders to the same odor,
  and the median peak and odor-window mean drive relative to threshold.
  Margin = f* − 1, f* the critical threshold factor: the factor on the KC's threshold at which its response to the
  odor (>= 2 spikes) disappears (responders, f* > 1) or appears (non-responders, f* < 1). Found by replaying the
  KC's exact drive (summed claw input after the per-claw knee and APL inhibition, minus KC-KC inhibition: rebuilt
  from a no-spike run and the real run's spikes, checked against the drive recovered from olfsysm's saved Vm)
  through olfsysm's integration at thresholds f × thr, the other KCs held as in the deterministic run. σ is the
  threshold jitter E48 matched to the data's single-trial same-odor floor (the reference model at 6.8 %: 0.207; the
  eight at 1.25: 0.23); a margin m changes in a jittered trial with probability about Φ(−|m| / σ) (with the
  re-tuned mean threshold, ×1.066 and ×1.041, taken into account in `p_change`). The peak drive is not a margin:
  it peaks briefly at odor onset, faster than Vm follows, so many non-responders peak above threshold.
- `pair_excess_per_recording.csv`: per model, pseudo-recording (E48's part-B trials at the APL-on-matched σ, 3
  trials each, processed as the KC data) and pair: the excess against the same recording's APL-off distances
  (`excess_paired`) and against the mean APL-off distances over the recordings (`excess_vs_mean_off`); rows
  "block (mean of the nine)"; and the KC data per control-fly recording against the mean APL output blocked (TNT)
  distances.
- `pair_excess_summary.csv`: per model and pair: the deterministic excess, the excess of the recordings' mean
  distances (E48's three-trial-means value), and the per-recording mean, SD, min, max and fraction positive.
- `E51_pair_excess_per_recording.png`: the nine pairs' per-recording excess per model, with the deterministic and
  E48's values; the KC data per control-fly recording.
- `E51_carrier_margins.png`: per model, the cumulative distributions of the margin (APL on) and of the spike counts
  for all responders, the carriers (responded and other odor), the both-responders and the APL-made carriers.
- `provenance.json`: code provenance (`model`, taken at the start), the checks, the input files' sha256.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--resim', action='store_true', help='rebuild the models even if the per-KC tables exist')
    ap.add_argument('--out', type=Path, default=None, help='results folder (default <shared>/results/E51_noise_robust_block/partA)')
    args = ap.parse_args()
    out = args.out or (H.results_dir(EXPERIMENT) / SUB)
    out.mkdir(parents=True, exist_ok=True)
    loc = H.outputs_dir(EXPERIMENT) / SUB
    loc.mkdir(parents=True, exist_ok=True)
    orn = H.apl16_orn_deltas()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E51_block_carriers.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), al_analysis_commit=commit,
        model=H.model_provenance(orn_deltas=orn), approx_conc_note=H.APPROX_CONC_NOTE)
    inputs = {}
    levels = json.loads((E48_RES / 'noise_levels.json').read_text())
    inputs[str(E48_RES / 'noise_levels.json')] = sha256(E48_RES / 'noise_levels.json')
    kc_all, sets_all, sigmas, labels, checks, det_excess, carriers, margins = {}, {}, {}, {}, {}, {}, [], []
    per_rec, three, thr_ratios = [], {}, {}
    for label, experiment, tag in MODELS:
        name = f'{E43.exp_id(experiment)}-{tag}'
        labels[name] = label
        f = H.shared_dir() / 'results' / experiment / 'fits.csv'
        inputs[str(f)] = sha256(f)
        lv = levels[name]
        sigma = float(lv['levels'][LEVEL])
        sigmas[name] = sigma
        cache, ck = loc / f'{name}_kc_odor.parquet', loc / f'{name}_checks.json'
        if cache.exists() and ck.exists() and not args.resim:
            kc = pd.read_parquet(cache)
            checks[name] = json.loads(ck.read_text())
            checks[name]['from_cache'] = str(cache)
            print(f'{name}: per-KC table from {cache}', flush=True)
        else:
            kc, checks[name] = simulate_model(label, experiment, tag, orn, loc)
            kc.to_parquet(cache)
            E43.write_json(ck, checks[name])
        inputs[str(cache)] = sha256(cache)
        kc['pct_rank'] = pct_rank_within_odor(kc)
        # the probability that a KC x odor entry changes (a response drops out, a non-response
        # appears) in one of E48's jittered trials, the rest of the network held fixed: thresholds
        # thr_det * ratio * (1 + sigma xi), ratio the re-tuned mean threshold
        ratio = float(lv['tuned_fixed_thr'][LEVEL]) / float(lv['fixed_thr_det'])
        thr_ratios[name] = ratio
        p_resp = sstats.norm.cdf((kc['crit_factor'] / ratio - 1) / sigma)
        kc['p_change'] = np.where(kc['response'], 1 - p_resp, p_resp)
        kc_all[name] = kc
        # the deterministic excess (cosine over the spike counts, 120 pairs)
        dp = {c: E43.pairs_of(kc[kc.condition == c].pivot(index='kc', columns='odor', values='spikes')[ODORS].to_numpy())
            for c in CONDS}
        st, tabs = H.compare_on_off(dp['on'], dp['acute_off'])
        det_excess[name] = tabs['pairs']['iso_resid'].reindex(BLOCK_PAIRS)
        checks[name]['block_det'] = st['block_fruity_small_iso']
        fit_block = float(E43.fit_row(experiment, tag)['fit_block'])
        checks[name]['block_det_minus_fits_csv'] = st['block_fruity_small_iso'] - fit_block
        assert abs(checks[name]['block_det_minus_fits_csv']) < 1e-9, (st['block_fruity_small_iso'], fit_block)
        rows, sets = decompose(name, kc, sigma, det_excess[name])
        for r in rows:
            r['label'] = label
        carriers += rows
        sets_all[name] = sets
        margins += margin_rows(name, label, kc, sets, sigma)
        pr, three[name], (mean_on, mean_off) = per_recording_excess(name, label, sigma, int(lv['n_recordings']), int(lv['n_trials']),
            inputs)
        per_rec.append(pr)
        # against E48's tables: the same processing of the same trials
        pl = pd.read_csv(E48_RES / 'pairs_noise_long.csv')
        inputs[str(E48_RES / 'pairs_noise_long.csv')] = sha256(E48_RES / 'pairs_noise_long.csv')
        pl = pl[(pl['model'] == name) & (pl['level'] == LEVEL) & (pl['transform'] == GRADED)]
        diffs = {}
        for cond, mine in (('on', mean_on), ('acute_off', mean_off)):
            theirs = pl[pl['condition'] == cond].set_index(['odor_a', 'odor_b'])
            assert len(theirs) == len(H.PAIRS16), f'{name} {cond}: {len(theirs)} pairs in E48 pairs_noise_long.csv'
            diffs[cond] = dict(raw=float((theirs['raw'] - mine.reindex(theirs.index)).abs().max()),
                det=float((theirs['det'] - dp[cond].reindex(theirs.index)).abs().max()))
        checks[name]['vs_e48_pairs_noise_long_max_abs_diff'] = diffs
        assert all(v < 1e-9 for d in diffs.values() for v in d.values()), diffs
    pd.DataFrame(carriers).to_csv(out / 'block_carriers.csv', index=False)
    pd.DataFrame(margins).to_csv(out / 'carrier_margins.csv', index=False)
    data_rec, data_measured = data_per_recording(inputs)
    per_rec = pd.concat(per_rec + [data_rec], ignore_index=True)
    per_rec.to_csv(out / 'pair_excess_per_recording.csv', index=False)
    summ = excess_summary(per_rec, det_excess, three, data_measured)
    summ.to_csv(out / 'pair_excess_summary.csv', index=False)
    make_figures(out, per_rec, summ, kc_all, sets_all, sigmas, labels)
    prov.update(models=[dict(name=f'{E43.exp_id(e)}-{t}', label=lab, experiment=e, tag=t, sigma_match_on=sigmas[f'{E43.exp_id(e)}-{t}'],
        retuned_thr_ratio=thr_ratios[f'{E43.exp_id(e)}-{t}']) for lab, e, t in MODELS], checks=checks, inputs_sha256=inputs,
        trials_source=str(E48_OUT / 'trials'), level=LEVEL, response_rule='>= 2 spikes (mp.kc.n_spikes_for_response)',
        margin_definition=MARGIN_NOTE)
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    (out / 'README.md').write_text(README.format(commit=commit[:7], generated=prov['generated'],
        models='; '.join(f'{lab} ({e}, {t})' for lab, e, t in MODELS)))
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    c = pd.DataFrame(carriers)
    print(c[c.condition == 'on'][['model', 'pair', 'distance', 'excess', 'n_both', 'n_carriers', 'n_carriers_off', 'n_apl_made_carriers',
        'carrier_share_of_distance', 'delta_distance', 'delta_from_carriers', 'delta_from_apl_made_carriers',
        'carrier_margin_resp_median', 'carrier_margin_other_median', 'carrier_pct_rank_within_odor_median', 'carrier_p_drop_mean',
        'carrier_p_other_joins_mean', 'carrier_p_separation_kept_mean', 'both_p_drop_mean']].round(3).to_string())
    print(pd.DataFrame(margins)[['model', 'condition', 'kc_class', 'n', 'margin_q25', 'margin_median', 'margin_q75',
        'frac_abs_margin_lt_sigma', 'p_change_jitter_mean', 'spikes_median', 'frac_2_spikes', 'pct_rank_within_odor_median',
        'peak_drive_rel_median']].round(3).to_string())
    print(summ.round(3).to_string())
    print('->', out)


if __name__ == '__main__':
    main()
