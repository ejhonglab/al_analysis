#!/usr/bin/env python3
"""E6 (re-scoped): readout window and dynamics of the model's odor x odor geometry.

Remy's per-frame KC-claw and PN-bouton matrices (data/remy_rdm_timeseries/) show that the
pair structure is stable over the response (frame-vs-window Spearman 0.94 at 0.75 s, 0.98 to
0.99 from 1 to 3 s) while only the overall between-odor similarity level falls (claws 0.38 ->
0.27 -> 0.19 at 0.5-0.75 / 1-1.5 / 2-3 s; boutons 0.30 -> 0.25 -> 0.22). This script asks
the same of the model:

(a) Does the model's pair structure depend on the readout window at all (spikes in early
    [0, 0.25) s vs late [0.25, 0.75) s after odor onset; the 0.5 s pulse; a GCaMP-like
    kernel), with and without APL? Readout: Spearman / isotonic R2 / block between window
    matrices, and each window's agreement with the control / TNT data and the 66-pair
    references (claws, somata, VCF).
(b) How do the between-odor similarity level and the pair-structure stability evolve over
    the pulse (25 ms bins and cumulative counts), with and without APL, and how does that
    depend on `claw_tau` and `apl_taum` (fixed thresholds and APL scale; rates reported)?

Spike trains come from re-running olfsysm's KC layer (`run_KC_sims`) on the tuned model
with `save_spike_recordings` on, for APL on and acute APL off (wAPLKC = 0). Model time:
odor 0-0.5 s, simulation to 0.75 s, dt 0.5 ms.

Outputs to <shared>/results/E6_readout_window/ (CSV tables, figures, provenance.json);
per-base spike-count-by-window parquet under ./outputs/E6_readout_window/.

Usage: OMP_NUM_THREADS=6 nice -n 10 ./venv/bin/python E6_readout_window.py [--bases ...]
    [--no-tau-sweep]
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H


EXP = 'E6_readout_window'
GENOS = ('TNT3c', 'TNT')
CTRL, TNT = GENOS
BASES: Dict[str, dict] = {
    'claw-dynamics': dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True, pn_claw_to_apl=True, claw_dynamics=True,
        n_spikes_for_response=2,
    ),
    'claw': dict(one_row_per_claw=True, prat_claws=True, use_connectome_APL_weights=True,
        n_spikes_for_response=2,
    ),
    'hemibrain-wd20': dict(pn2kc_connections='hemibrain', weight_divisor=20,
        _drop_glom_with_plus=False,
    ),
}
WINDOWS: Dict[str, Tuple[float, float]] = {
    'early': (0.0, 0.25), 'late': (0.25, 0.75), 'pulse': (0.0, 0.5), 'post': (0.5, 0.75),
    'whole': (0.0, 0.75),
}
BIN_S = 0.025
GCAMP_RISE_S, GCAMP_DECAY_S = 0.05, 1.0
CLAW_TAUS = (0.025, 0.05, 0.1)
APL_TAUMS = (0.025, 0.05, 0.1, 0.2)
REMY_DIR = 'data/remy_rdm_timeseries'


# ---------------------------------------------------------------------------
# model spike trains
class SpikeSimulator:
    """Re-runs the KC layer of a tuned model and returns spike trains (odor x KC x time)."""

    def __init__(self, base: str, target: float):
        from al_analysis.mb_model import fit_mb_model
        import olfsysm as osm

        self.osm = osm
        self.base = base
        self.kws = dict(BASES[base])
        self.orn = H.apl16_orn_deltas()
        t0 = time.time()
        responses, spike_counts, _, pdict = fit_mb_model(self.orn, **self.kws,
            target_sparsity=target, make_plots=False, silent=True, return_olfsysm_vars=True)
        self.fit_seconds = time.time() - t0
        self.responses = H.abbrev_columns(responses.astype(float))
        self.spike_counts = H.abbrev_columns(spike_counts.astype(float))
        self.mp, self.rv = pdict['mp'], pdict['rv']
        self.rv.log.disable()
        self.rv.verbose = False
        self.mp.kc.save_spike_recordings = True
        self.wAPLKC0 = np.array(self.rv.kc.wAPLKC).copy()
        self.scale0 = float(self.rv.kc.wAPLKC_scale) if self.mp.kc.preset_wAPLKC else None
        self.n_spikes = self.kws.get('n_spikes_for_response', 1)
        self.defaults = dict(claw_tau=float(self.mp.kc.claw_tau),
            apl_taum=float(self.mp.kc.apl_taum))
        self.params = {k: (float(v) if np.ndim(v) == 0 else float(np.mean(v)))
            for k, v in pdict.items() if k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale',
            'wAPLKC', 'wKCAPL') and v is not None}

    def run(self, apl_on: bool, **mp_kc) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (spikes[odor, kc, time] as int8, times_s) for APL on/off and optional
        `mp.kc` overrides (e.g. claw_tau=0.1); overrides persist until reset."""
        from al_analysis.mb_model import get_sim_var, get_time_index

        for k, v in mp_kc.items():
            setattr(self.mp.kc, k, v)
        if apl_on:
            self.rv.kc.wAPLKC = self.wAPLKC0
            if self.scale0 is not None:
                self.rv.kc.wAPLKC_scale = self.scale0
        else:
            self.rv.kc.wAPLKC = 0.0 * self.wAPLKC0
            if self.scale0 is not None:
                self.rv.kc.wAPLKC_scale = 0.0
        self.osm.run_KC_sims(self.mp, self.rv, False)
        arr = get_sim_var(self.rv.kc, 'spike_recordings')
        arr = np.nan_to_num(arr, nan=0.0).astype(np.int8)
        ts = get_time_index(self.mp, self.rv).to_numpy()
        assert arr.shape[-1] == len(ts), (arr.shape, len(ts))
        return arr, ts

    def reset(self) -> None:
        for k, v in self.defaults.items():
            setattr(self.mp.kc, k, v)


def window_counts(spk: np.ndarray, ts: np.ndarray, t0: float, t1: float) -> np.ndarray:
    sel = (ts >= t0) & (ts < t1)
    return spk[:, :, sel].sum(axis=-1)


def gcamp_peak(spk: np.ndarray, ts: np.ndarray, *, rise: float = GCAMP_RISE_S,
    decay: float = GCAMP_DECAY_S, step: float = 0.005) -> np.ndarray:
    """Peak (over t >= 0) of spike trains convolved with a (1 - e^{-t/rise}) e^{-t/decay}
    kernel, after binning spikes to `step` seconds. Returns odor x KC."""
    dt = float(np.median(np.diff(ts)))
    n = max(1, int(round(step / dt)))
    sel = ts >= -0.05
    s = spk[:, :, sel].astype(np.float32)
    T = s.shape[-1] // n * n
    s = s[:, :, :T].reshape(s.shape[0], s.shape[1], T // n, n).sum(-1)
    tk = np.arange(0, 3 * decay, step)
    k = (1 - np.exp(-tk / rise)) * np.exp(-tk / decay)
    k /= k.max()
    out = np.empty(s.shape[:2], dtype=np.float32)
    for i in range(s.shape[0]):
        conv = np.apply_along_axis(lambda x: np.convolve(x, k)[:len(x)], 1, s[i])
        out[i] = conv.max(axis=1)
    return out


def rdm_from_counts(counts: np.ndarray, odors: List[str], kc_index: pd.Index,
    measure: str = 'cosine') -> pd.Series:
    df = pd.DataFrame(counts.T, index=kc_index, columns=odors)
    return H.pair_distances(df, measure=measure)


def between_level_and_stability(spk: np.ndarray, ts: np.ndarray, odors: List[str],
    kc_index: pd.Index, ref: pd.Series, *, bin_s: float = BIN_S, t_start: float = -0.1,
    t_end: float = 0.75) -> pd.DataFrame:
    """Per time bin (and cumulative from onset): mean between-odor cosine similarity and
    Spearman of the bin RDM with the whole-window RDM `ref`."""
    rows = []
    edges = np.arange(t_start, t_end + 1e-9, bin_s)
    for a, b in zip(edges[:-1], edges[1:]):
        for kind, (t0, t1) in (('bin', (a, b)), ('cumulative', (0.0, b))):
            if kind == 'cumulative' and b <= 0:
                continue
            c = window_counts(spk, ts, t0, t1)
            d = rdm_from_counts(c, odors, kc_index)
            ok = d.notna()
            rows.append(dict(kind=kind, t_mid=(a + b) / 2, t_end=b,
                n_spikes=int(c.sum()), frac_kcs_active=float((c.sum(0) > 0).mean()),
                mean_similarity=float(1 - d[ok].mean()) if ok.any() else np.nan,
                n_pairs_defined=int(ok.sum()),
                spearman_vs_whole=H.spearman(d[ok], ref[ok]) if ok.sum() > 10 else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# data: Remy's per-frame matrices
def remy_curves() -> pd.DataFrame:
    d = H.shared_dir() / REMY_DIR
    rows = []
    for stage, arr_name, t_name in (('claws', 'kc_stim_Fc_zscore.npy', 'times_kc.npy'),
            ('boutons', 'pn_stim_F_zscore.npy', 'times_pn.npy')):
        arr = np.load(d / arr_name)
        ts = np.load(d / t_name)
        iu = np.triu_indices(arr.shape[-1], 1)
        m = np.nanmean(arr, axis=0)
        ref = np.nanmean(m[(ts >= 0.5) & (ts < 3.0)], axis=0)[iu]
        for i, t in enumerate(ts):
            v = m[i][iu]
            rows.append(dict(stage=stage, t=float(t), mean_similarity=float(1 - np.nanmean(v)),
                spearman_vs_window=float(stats.spearmanr(v, ref, nan_policy='omit')[0])))
    return pd.DataFrame(rows)


def remy_window_stats() -> Dict[str, float]:
    """Spearman between Remy's fig7 early and late claw / bouton windows (fly means)."""
    d = H.shared_dir() / REMY_DIR
    out = {}
    for stage, arr_name, t_name in (('claws', 'kc_stim_Fc_zscore.npy', 'times_kc.npy'),
            ('boutons', 'pn_stim_F_zscore.npy', 'times_pn.npy')):
        arr = np.load(d / arr_name)
        ts = np.load(d / t_name)
        iu = np.triu_indices(arr.shape[-1], 1)
        m = np.nanmean(arr, axis=0)
        w = {k: np.nanmean(m[(ts >= a) & (ts < b)], axis=0)[iu] for k, (a, b) in
            (('early', (0.5, 0.75)), ('late', (1.5, 2.0)), ('odor', (0.5, 3.0)))}
        out[f'{stage}_spearman_early_late'] = float(stats.spearmanr(w['early'], w['late'])[0])
        out[f'{stage}_mean_sim_early'] = float(1 - np.nanmean(w['early']))
        out[f'{stage}_mean_sim_late'] = float(1 - np.nanmean(w['late']))
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--bases', nargs='+', default=['claw-dynamics', 'claw'], choices=list(BASES))
    ap.add_argument('--target', type=float, default=0.05)
    ap.add_argument('--no-tau-sweep', action='store_true')
    args = ap.parse_args()
    pd.set_option('display.width', 220)
    res = H.results_dir(EXP)
    out = H.outputs_dir(EXP)
    orn_pairs = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {g: H.yang_mean_rdm(g, 'cosine') for g in GENOS}
    data_bin = {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in GENOS}

    curves_rows, win_rows, pairwin_rows, sweep_rows, meta = [], [], [], [], {}
    for base in args.bases:
        sim = SpikeSimulator(base, args.target)
        odors = list(sim.responses.columns)
        kc_index = sim.responses.index
        meta[base] = dict(n_kcs=len(kc_index), fit_seconds=sim.fit_seconds,
            params=sim.params, defaults=sim.defaults)
        print(f'{base}: tuned in {sim.fit_seconds:.0f} s; {len(kc_index)} KCs', flush=True)
        rdms: Dict[Tuple[str, str], pd.Series] = {}
        for cond, apl_on in (('on', True), ('acute_off', False)):
            spk, ts = sim.run(apl_on)
            total = spk.sum(-1).T  # kc x odor
            if cond == 'on':
                diff = np.abs(total - sim.spike_counts.to_numpy()).max()
                meta[base]['max_abs_diff_total_vs_spike_counts'] = float(diff)
                print(f'  spike totals vs fit_mb_model spike_counts: max |diff| {diff}')
            # window RDMs
            wins = {}
            for w, (t0, t1) in WINDOWS.items():
                c = window_counts(spk, ts, t0, t1)
                wins[w] = c
                pd.DataFrame(c.T, index=kc_index, columns=odors).to_parquet(
                    out / f'{base}_{cond}_{w}_counts.parquet')
            g = gcamp_peak(spk, ts)
            wins['gcamp_peak'] = g
            for w, c in wins.items():
                for recipe in ('cosine_counts', 'cosine_binary'):
                    if recipe == 'cosine_binary':
                        c_use = (c >= sim.n_spikes).astype(float)
                        if w == 'gcamp_peak':
                            continue
                    else:
                        c_use = c
                    d = rdm_from_counts(c_use, odors, kc_index)
                    rdms[(cond, w, recipe)] = d
                    geno = CTRL if cond == 'on' else TNT
                    tgt = data[geno] if recipe == 'cosine_counts' else data_bin[geno]
                    mvd = H.compare_model_to_data(d, tgt, orn=orn_pairs)
                    rr = H.compare_to_references(d, ref)
                    win_rows.append(dict(base=base, condition=cond, window=w, recipe=recipe,
                        t0=WINDOWS.get(w, (np.nan, np.nan))[0], t1=WINDOWS.get(w, (np.nan, np.nan))[1],
                        rate=float((c >= sim.n_spikes).mean()), mean_dist=float(d.mean()),
                        n_nan_pairs=int(d.isna().sum()),
                        spearman_vs_data=mvd['spearman'], spearman_dev_vs_data=mvd['spearman_dev'],
                        spearman_orn=H.spearman(d, orn_pairs),
                        **{k: rr[k] for k in ('spearman_kc_claw', 'spearman_kc_soma',
                            'spearman_vcf_hyp', 'spearman_resid_vs_vcf_hyp',
                            'spearman_resid_vs_remap_wt')}))
            # time curves
            ref_whole = rdms[(cond, 'whole', 'cosine_counts')]
            cur = between_level_and_stability(spk, ts, odors, kc_index, ref_whole)
            cur.insert(0, 'condition', cond)
            cur.insert(0, 'base', base)
            curves_rows.append(cur)
            del spk
        # window-vs-window and on-vs-off comparisons
        for cond in ('on', 'acute_off'):
            for recipe in ('cosine_counts', 'cosine_binary'):
                for wa, wb in (('early', 'late'), ('early', 'whole'), ('late', 'whole'),
                        ('pulse', 'post'), ('whole', 'gcamp_peak')):
                    if (cond, wa, recipe) not in rdms or (cond, wb, recipe) not in rdms:
                        continue
                    st, _ = H.compare_on_off(rdms[(cond, wb, recipe)], rdms[(cond, wa, recipe)])
                    pairwin_rows.append(dict(base=base, condition=cond, recipe=recipe,
                        comparison=f'{wa}_vs_{wb}', spearman=st['spearman_on_off'],
                        iso_r2=st['iso_r2'], lin_r2=st['lin_r2'],
                        block_iso=st['block_fruity_small_iso'],
                        mean_a=st['mean_off'], mean_b=st['mean_on']))
        for w in list(WINDOWS) + ['gcamp_peak']:
            for recipe in ('cosine_counts', 'cosine_binary'):
                if ('on', w, recipe) not in rdms:
                    continue
                st, _ = H.compare_on_off(rdms[('on', w, recipe)], rdms[('acute_off', w, recipe)],
                    orn=orn_pairs, drive=drive)
                pairwin_rows.append(dict(base=base, condition='on_vs_off', recipe=recipe,
                    comparison=f'window_{w}', spearman=st['spearman_on_off'],
                    iso_r2=st['iso_r2'], lin_r2=st['lin_r2'],
                    block_iso=st['block_fruity_small_iso'], mean_a=st['mean_off'],
                    mean_b=st['mean_on'],
                    per_odor_vs_drive=st['spearman_per_odor_iso_vs_drive']))
        long = pd.concat({k: v for k, v in rdms.items()},
            names=['condition', 'window', 'recipe', 'odor_a', 'odor_b']).rename('distance')
        long.reset_index().assign(base=base).to_csv(res / f'model_pair_distances_{base}.csv',
            index=False)

        # tau sweep (fixed thresholds / APL scale)
        if not args.no_tau_sweep and base == 'claw-dynamics':
            for claw_tau in CLAW_TAUS:
                for apl_taum in APL_TAUMS:
                    for cond, apl_on in (('on', True), ('acute_off', False)):
                        spk, ts = sim.run(apl_on, claw_tau=claw_tau, apl_taum=apl_taum)
                        whole = rdm_from_counts(window_counts(spk, ts, 0, 0.75), odors, kc_index)
                        early = rdm_from_counts(window_counts(spk, ts, 0, 0.25), odors, kc_index)
                        late = rdm_from_counts(window_counts(spk, ts, 0.25, 0.75), odors, kc_index)
                        cur = between_level_and_stability(spk, ts, odors, kc_index, whole,
                            bin_s=0.05, t_start=0.0, t_end=0.75)
                        b = cur[cur.kind == 'bin']
                        st, _ = H.compare_on_off(late, early)
                        geno = CTRL if cond == 'on' else TNT
                        mvd = H.compare_model_to_data(whole, data[geno], orn=orn_pairs)
                        sweep_rows.append(dict(base=base, claw_tau=claw_tau, apl_taum=apl_taum,
                            condition=cond,
                            rate=float((window_counts(spk, ts, 0, 0.75) >= sim.n_spikes).mean()),
                            mean_dist_whole=float(whole.mean()),
                            spearman_early_late=st['spearman_on_off'], iso_r2_early_late=st['iso_r2'],
                            sim_first_100ms=float(b[b.t_end <= 0.1].mean_similarity.mean()),
                            sim_last_250ms_pulse=float(b[(b.t_mid > 0.25) & (b.t_end <= 0.5)]
                                .mean_similarity.mean()),
                            stability_min_after_100ms=float(b[b.t_mid > 0.1].spearman_vs_whole.min()),
                            spearman_vs_data=mvd['spearman'], spearman_dev_vs_data=mvd['spearman_dev']))
                        del spk
                        print(f'  tau sweep claw_tau={claw_tau} apl_taum={apl_taum} {cond}: '
                            f'rate {sweep_rows[-1]["rate"]:.3f}, early-late Spearman '
                            f'{st["spearman_on_off"]:.3f}', flush=True)
            sim.reset()
        del sim

    curves = pd.concat(curves_rows, ignore_index=True)
    curves.to_csv(res / 'model_time_curves.csv', index=False)
    wins = pd.DataFrame(win_rows)
    wins.to_csv(res / 'window_stats.csv', index=False)
    pw = pd.DataFrame(pairwin_rows)
    pw.to_csv(res / 'window_comparisons.csv', index=False)
    if sweep_rows:
        sw = pd.DataFrame(sweep_rows)
        sw.to_csv(res / 'tau_sweep.csv', index=False)
    rc = remy_curves()
    rc.to_csv(res / 'remy_time_curves.csv', index=False)
    rws = remy_window_stats()

    print('\nwindow stats (cosine over spike counts):')
    print(wins[wins.recipe == 'cosine_counts'][['base', 'condition', 'window', 'rate',
        'mean_dist', 'spearman_vs_data', 'spearman_dev_vs_data', 'spearman_orn',
        'spearman_kc_claw', 'spearman_kc_soma', 'spearman_resid_vs_vcf_hyp']].round(3).to_string())
    print('\nwindow comparisons:')
    print(pw.round(3).to_string())
    print('\nRemy windows:', {k: round(v, 3) for k, v in rws.items()})
    if sweep_rows:
        print('\ntau sweep:')
        print(sw.round(3).to_string())

    make_figures(res, curves, rc, pw, wins, sw if sweep_rows else None)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        script='E6_readout_window.py', host=os.uname().nodename,
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE,
            capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'],
            cwd=Path('~/src/olfsysm').expanduser(), capture_output=True, text=True).stdout.strip(),
        windows=WINDOWS, gcamp_kernel=dict(rise_s=GCAMP_RISE_S, decay_s=GCAMP_DECAY_S),
        claw_taus=CLAW_TAUS, apl_taums=APL_TAUMS, meta=meta, remy_window_stats=rws,
        approx_conc_note=H.APPROX_CONC_NOTE)
    (res / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str))
    print('done ->', res)


def make_figures(res: Path, curves: pd.DataFrame, rc: pd.DataFrame, pw: pd.DataFrame,
    wins: pd.DataFrame, sw: Optional[pd.DataFrame]) -> None:
    plt = H._plt()
    plt.rcParams.update({'font.size': 8})
    colors = {'on': H.CONDITION_COLORS['control'], 'acute_off': H.CONDITION_COLORS['TNT']}
    bases = list(curves.base.unique())
    fig, axs = plt.subplots(2, len(bases) + 1, figsize=(4.2 * (len(bases) + 1), 6.4),
        squeeze=False)
    for j, base in enumerate(bases):
        for kind, ls in (('bin', '-'), ('cumulative', ':')):
            for cond in ('on', 'acute_off'):
                c = curves[(curves.base == base) & (curves.kind == kind) & (curves.condition == cond)]
                axs[0, j].plot(c.t_mid, c.mean_similarity, ls, color=colors[cond], lw=1.4,
                    label=f'{cond}, {kind}')
                axs[1, j].plot(c.t_mid, c.spearman_vs_whole, ls, color=colors[cond], lw=1.4,
                    label=f'{cond}, {kind}')
        for ax in axs[:, j]:
            ax.axvspan(0, 0.5, color='#eeeeee', zorder=0)
            ax.set_xlabel('time from odor onset (s)')
        axs[0, j].set_title(f'model {base}: between-odor cosine similarity')
        axs[1, j].set_title(f'model {base}: Spearman of bin RDM vs whole-window RDM')
        axs[1, j].set_ylim(0, 1.02)
        axs[0, j].legend(frameon=False, fontsize=7)
    j = len(bases)
    for stage, col in (('claws', '#555555'), ('boutons', '#999999')):
        r = rc[(rc.stage == stage) & (rc.t >= -1) & (rc.t <= 6)]
        axs[0, j].plot(r.t, r.mean_similarity, color=col, lw=1.4, label=f'Remy {stage}')
        axs[1, j].plot(r.t, r.spearman_vs_window, color=col, lw=1.4, label=f'Remy {stage}')
    for ax in axs[:, j]:
        ax.axvspan(0, 3, color='#eeeeee', zorder=0)
        ax.set_xlabel('time from odor onset (s)')
    axs[0, j].set_title('data (megamat17, per-frame 1 - r): between-odor similarity')
    axs[1, j].set_title('data: Spearman of frame RDM vs 0.5-3 s window RDM')
    axs[1, j].set_ylim(0, 1.02)
    axs[0, j].legend(frameon=False, fontsize=7)
    fig.suptitle('E6. Similarity level and pair-structure stability over the response: model '
        '(0.5 s pulse, grey) vs Remy\'s claws / boutons (3 s pulse)', y=1.01)
    fig.tight_layout()
    fig.savefig(res / 'E6_time_curves.png', dpi=140, bbox_inches='tight')
    fig.savefig(res / 'E6_time_curves.pdf', bbox_inches='tight')
    plt.close(fig)

    if sw is not None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, col, title in zip(axs, ['spearman_early_late', 'sim_last_250ms_pulse', 'rate'],
                ['Spearman early vs late window', 'mean similarity, last 250 ms of pulse',
                 'response rate (whole window)']):
            for cond, marker in (('on', 'o'), ('acute_off', 's')):
                d = sw[sw.condition == cond]
                for ct, ls in zip(CLAW_TAUS, ('-', '--', ':')):
                    dd = d[d.claw_tau == ct].sort_values('apl_taum')
                    ax.plot(dd.apl_taum, dd[col], ls, marker=marker, ms=4, color=colors[cond],
                        lw=1.2, label=f'{cond}, claw_tau {ct}')
            ax.set_xscale('log')
            ax.set_xlabel('apl_taum (s)')
            ax.set_title(title, fontsize=8)
        axs[0].legend(frameon=False, fontsize=6, ncol=2)
        fig.suptitle('E6. claw_tau x apl_taum sweep (fixed thresholds and APL scale; claw-dynamics 5 %)',
            y=1.03)
        fig.tight_layout()
        fig.savefig(res / 'E6_tau_sweep.png', dpi=140, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
