#!/usr/bin/env python3
"""Time-step check of the olfsysm integrator on tuned models (numerics, not a mechanism).

olfsysm integrates every state variable with forward Euler at dt = 0.5 ms and generates
KC spikes by threshold crossing with reset to 0 and no refractory period. This script
runs tuned models with every parameter fixed at dt = 0.5, 0.25 and 0.125 ms (the whole
ORN -> LN -> PN -> KC pipeline, APL on and acute APL off) and reports what changes: the
response rate, the fraction of (KC, odor) responses that flip, the spike counts, the
harness statistics against the KC data, and the APL (inh) peak, for the claw-driven bouton
base and the spike-driven base, both with the 70 Hz per-claw threshold at their tuned 5 %
parameters.

Two dt dependences are expected from the code: (1) the inter-spike interval of a strongly
driven KC is quantized to whole steps (no refractory period), so its spike count depends on
dt; (2) a KC spike adds wKCAPL * 1e4 to dIs/dt for one step, so the KC->APL drive per spike
scales with dt in the spike-driven base (pn_claw_to_apl=False), whereas the claw-driven
drive and the KC-KC term are rates and dt-invariant.

olfsysm sizes its run variables from the parameters at construction, so the step is set by
giving `fit_mb_model` a ModelParams constructor that sets `time_dt` (the only way to change
it without a new kwarg); the dt = 0.5 ms run reproduces the stored model exactly.

Outputs: <shared>/results/numerics_dt_check/{per_dt_stats.csv, flip_fractions.csv,
provenance.json}.
Usage: OMP_NUM_THREADS=16 ./venv/bin/python numerics_dt_check.py
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as sstats

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON

DTS = [0.5e-3, 0.25e-3, 0.125e-3]
BASES = {
    # tuned at 5 % with APL on (E16 / E38's A_lab re-run; E36's spike-driven base)
    'claw_driven_thr70': (dict(BOUTON, claw_thr=70.0), dict(fixed_thr=130.69146143570123, wAPLKC=0.19377886836027758,
        wKCAPL=0.00011188156371840549, wAPLPN=0.19377886836027758, wPNAPL=0.000498146191157526)),
    'spike_driven_thr70': (dict(BOUTON, claw_thr=70.0, pn_claw_to_apl=False), dict(fixed_thr=130.69146143570123,
        wAPLKC=0.2610902617397995, wKCAPL=0.00015074495481512672, wAPLPN=0.2610902617397995, wPNAPL=0.0006711831921331603)),
}


class ParamsWithDt:
    """Stands in for olfsysm.ModelParams inside mb_model: the default parameters with
    the time step set (and APL time courses saved)."""

    def __init__(self, orig, dt: float):
        self.orig, self.dt = orig, dt

    def __call__(self):
        m = self.orig()
        m.time_dt = self.dt
        m.kc.save_inh_sims = True
        return m


def run(orn, kws, params, dt, cond, osm, mb_model):
    from al_analysis.mb_model import fit_mb_model, get_sim_var
    orig = osm.ModelParams
    osm.ModelParams = ParamsWithDt(orig, dt)
    try:
        p = dict(params)
        if cond == 'off':
            p['wAPLKC'] = 0.0; p['wAPLPN'] = 0.0
        resp, spk, wPNKC, pdict = fit_mb_model(orn, **p, **kws, make_plots=False, silent=True, return_olfsysm_vars=True)
    finally:
        osm.ModelParams = orig
    mp, rv = pdict['mp'], pdict['rv']
    assert abs(float(mp.time_dt) - dt) < 1e-12, (mp.time_dt, dt)
    try:
        inh = np.asarray(get_sim_var(rv.kc, 'inh_sims'), dtype=float); inh_peak = float(np.nanmax(inh))
    except Exception:
        inh_peak = np.nan
    return H.abbrev_columns(resp.astype(float))[H.APL16_ODORS], H.abbrev_columns(spk.astype(float))[H.APL16_ODORS], inh_peak


def main() -> None:
    import olfsysm as osm
    import al_analysis.mb_model as mb_model
    res_dir = H.shared_dir() / 'results' / 'numerics_dt_check'; res_dir.mkdir(exist_ok=True)
    orn = H.apl16_orn_deltas()
    data = {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    rows, flips = [], []
    for base, (kws, params) in BASES.items():
        ref = {}
        for dt in DTS:
            t0 = time.time(); out = {}
            for cond in ('on', 'off'):
                resp, spk, inh_peak = run(orn, kws, params, dt, cond, osm, mb_model)
                out[cond] = dict(resp=resp, spk=spk, inh_peak=inh_peak)
            on, off = out['on'], out['off']
            rd_on = H.pair_distances(on['spk'], 'cosine'); rd_off = H.pair_distances(off['spk'], 'cosine')
            st, _ = H.compare_on_off(rd_on, rd_off)
            row = dict(base=base, dt_ms=dt * 1e3, rate_on=float(on['resp'].to_numpy().mean()), rate_off=float(off['resp'].to_numpy().mean()),
                total_spikes_on=float(on['spk'].to_numpy().sum()), total_spikes_off=float(off['spk'].to_numpy().sum()),
                max_spikes_per_kc_odor_on=float(on['spk'].to_numpy().max()), inh_peak_on=on['inh_peak'],
                rho_ctrl_on=H.spearman(rd_on, data['TNT3c']), rho_tnt_off=H.spearman(rd_off, data['TNT']),
                mean_dist_on=float(rd_on.mean()), mean_dist_off=float(rd_off.mean()), block=st['block_fruity_small_iso'], iso_r2=st['iso_r2'],
                rate_rho_on=float(sstats.spearmanr(H.response_fractions(on['resp']).reindex(H.APL16_ODORS), data_rates['TNT3c'].reindex(H.APL16_ODORS))[0]),
                rate_rho_off=float(sstats.spearmanr(H.response_fractions(off['resp']).reindex(H.APL16_ODORS), data_rates['TNT'].reindex(H.APL16_ODORS))[0]),
                seconds=time.time() - t0)
            if dt == DTS[0]:
                ref = dict(on=on, off=off, rd_on=rd_on, rd_off=rd_off)
            else:
                for cond, cur in (('on', on), ('off', off)):
                    r0, r1 = ref[cond]['resp'].to_numpy(), cur['resp'].to_numpy(); s0, s1 = ref[cond]['spk'].to_numpy(), cur['spk'].to_numpy()
                    either = (r0 + r1) > 0
                    flips.append(dict(base=base, dt_ms=dt * 1e3, condition=cond,
                        frac_entries_flipped=float((r0 != r1).mean()), frac_responses_flipped=float((r0 != r1)[either].mean()) if either.any() else np.nan,
                        n_responses_ref=int(r0.sum()), n_responses_new=int(r1.sum()),
                        spearman_counts_responding=float(sstats.spearmanr(s0[either], s1[either])[0]) if either.sum() > 2 else np.nan,
                        mean_abs_count_change_responding=float(np.abs(s0 - s1)[either].mean()) if either.any() else np.nan,
                        count_ratio_new_over_ref=float(s1.sum() / max(s0.sum(), 1.0))))
                row['rho_on_vs_ref_dt'] = H.spearman(rd_on, ref['rd_on']); row['rho_off_vs_ref_dt'] = H.spearman(rd_off, ref['rd_off'])
            rows.append(row)
            print(f"{base} dt {dt*1e3:g} ms: rate on {row['rate_on']:.4f} off {row['rate_off']:.4f}; spikes on {row['total_spikes_on']:.0f} off {row['total_spikes_off']:.0f} (max/KC {row['max_spikes_per_kc_odor_on']:.0f}); "
                f"inh peak {row['inh_peak_on']:.3g}; rho ctrl {row['rho_ctrl_on']:.3f} tnt {row['rho_tnt_off']:.3f}; mean dist {row['mean_dist_on']:.3f}/{row['mean_dist_off']:.3f}; "
                f"block {row['block']:+.3f}; iso {row['iso_r2']:.3f}; rate rho {row['rate_rho_on']:.2f}/{row['rate_rho_off']:.2f}; {row['seconds']:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(res_dir / 'per_dt_stats.csv', index=False)
            if flips:
                pd.DataFrame(flips).to_csv(res_dir / 'flip_fractions.csv', index=False)
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print(pd.DataFrame(flips).round(4).to_string(index=False))
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='numerics_dt_check.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), dts_ms=[d * 1e3 for d in DTS],
        bases={k: dict(kws=v[0], params=v[1]) for k, v in BASES.items()}, model=H.model_provenance(orn_deltas=orn))
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('->', res_dir)


if __name__ == '__main__':
    main()
