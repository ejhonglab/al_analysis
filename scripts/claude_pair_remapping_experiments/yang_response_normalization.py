#!/usr/bin/env python3
"""How Yang's APL16 KC responses are made from suite2p's traces, checked on the recordings themselves.

Yang (2026-11-10, in viki's session): each KC's trace is `Fc`, z-scored over the whole recording; the response
of a trial is the mean z within 2-8 s after odor onset minus the mean within the 5 s before. Her pipeline
(KC_analytical_scripts/Step07_allcell_Fc_Correlation_fulltracezscore_minuseach_20240909.ipynb on her data
drive) computes `Fc = F - 0.7 * Fneu` from suite2p's `F.npy` / `Fneu.npy` (suite2p/combined of the
`<recording>_els_moco_s2p0` run), keeps the ROIs with iscell == 1, z-scores each over all volumes
(scipy.stats.zscore, ddof 0), and takes, per trial, the mean over volumes round(2 v) .. round(8 v) - 1
after the onset volume minus the mean over the round(5 v) volumes before it (v = frame_rate /
frame_per_volume, the volume rate, from the recording's YAML).

This script, for each of the 19 recordings in olfdb datasets 19 / 20 (8 APL output blocked (TNT), 11 control
flies), reads only from Yang's data drive and writes only to the shared directory:
1. recomputes Fc, its z-score and every trial's response, and compares them with Yang's saved
   `_Fc_zscored_full_cell.npy`, `_resp_df.csv` and the olfdb values (the reproduction check);
2. records the suite2p settings that shape F and Fneu (version, neuropil coefficient and mask, ROI
   detection, registration) and the timing (frames, volumes, window lengths);
3. measures what the whole-trace z-score does: each KC's scale (the SD of Fc over the whole recording,
   which is what a z unit is) against its noise in the pre-odor baselines, how that ratio grows with the
   number of odors the KC responds to, and the ceiling the whole-trace z-score puts on a KC's mean squared
   response (the z-scored trace has mean square 1 over its T volumes, so the 51 response windows of w
   volumes can hold at most T / (51 w) of it on average);
4. re-expresses the responses against each KC's baseline noise (dFc / SD of Fc in the pre-odor windows,
   each window's mean removed), with one threshold chosen so that the control flies' mean response rate
   equals Yang's (trial-mean > 1 z), and compares the response rates, their APL-off / APL-on ratio and the
   number of odors per KC between the two normalizations.

Outputs (`--out`, default <shared>/notes/yang_response_normalization/): recordings.csv (one row per
recording), kc_scales.csv (per KC: whole-trace SD, baseline-noise SD, their ratio, the number of odors
responded to, mean squared response), normalization_rates.csv, summary.json, two figures.

The reader (`read_suite2p`, which accepts only the 19 recordings of `RECORDINGS`) is shared with
E84_trace_time_course.py.

Usage: ./venv/bin/python yang_response_normalization.py [--drive-root /media/yang] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import zscore

import kc_remapping_harness as H

RECORDINGS = (
    [('TNT', r) for r in ('20240717_2FLfun1', '20240717_3FLfun1', '20240725_1FLfun1', '20240801_4FRfun1', '20240802_5FLfun1',
                          '20240807_3FLfun1', '20240814_3FLfun1', '20240814_3FRfun2')]
    + [('TNT3c', r) for r in ('20240710_2FLfun1', '20240710_2FRfun1', '20240712_2FLfun1', '20240726_4FRfun1', '20240801_3FLfun1',
                              '20240801_3FRfun1', '20240802_5FRfun2', '20240809_3FLfun2', '20240809_3FRfun1', '20240814_1FLfun1',
                              '20240830_3FRfun2')])
# Yang's odor names (YAML odor_pin) -> olfdb's
ODOR_NAMES = {'MS(-3)': 'ms @ -3', 'Phe(-3)': 'paa @ -3', '2-hep(-3)': '2h @ -3', '1-6ol(-3)': '1-6ol @ -3', 'E2H(-3)': 't2h @ -3',
              '2-but(-3)': '2-but @ -3', 'b-Myr(-2)': 'B-myr @ -2', 'Euc(-2)': 'euc @ -2', 'a-Pine(-2)': '-aPine @ -2',
              '6al(-3)': '6al @ -3', 'IAA(-3)': 'IaA @ -3', 'PA(-3)': 'pa @ -3', '1-5ol(-3)': '1-5ol @ -3', '1-8ol(-3)': '1-8ol @ -3',
              'EB(-3)': 'eb @ -3', 'EP(-3)': 'ep @ -3', 'PFO': 'pfo @ 0'}
OPS_KEYS = ('suite2p_version', 'nplanes', 'fs', 'nframes', 'neucoeff', 'inner_neuropil_radius', 'min_neuropil_pixels', 'allow_overlap',
            'circular_neuropil', 'lam_percentile', 'anatomical_only', 'diameter', 'do_registration', 'nonrigid', 'baseline',
            'win_baseline', 'sig_baseline', 'high_pass', 'spikedetect', 'tau')


def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if not hits:
        raise FileNotFoundError(f'{root}/{pattern}')
    return hits[0]


def olfdb_matrix(genotype: str, rec: str) -> pd.DataFrame:
    """olfdb's per-trial values of one recording: KCs (roi) x (odor, repeat)."""
    ds = {'TNT': 19, 'TNT3c': 20}[genotype]
    d = pd.read_csv(H.shared_dir() / 'data' / 'yang_apl16_kc_olfdb' / genotype / f'dataset_{ds}_long.csv')
    date = f'{rec[:4]}-{rec[4:6]}-{rec[6:8]}'
    fly, hemi = int(rec.split('_')[1][0]), rec.split('_')[1][2]
    d = d[(d.date == date) & (d.fly_num == fly) & (d.hemibrain == hemi)]
    return d.pivot_table(index='roi', columns=['odor', 'repeat'], values='value')


def read_suite2p(genotype: str, rec: str, drive_root: Path) -> dict:
    """One of the 19 recordings' suite2p output and stimulus records, read-only, by its exact name.

    Only the recordings in `RECORDINGS` are accepted (ValueError otherwise): Yang agreed to the reading of these
    (notes/yang_drive_data_agreement.md, item 2); nothing else on her drives is read or listed. Reads the recording's
    YAML file (`YANGZHENG_0*/0_2P/<date>/<recording>.yaml`), suite2p's ops.npy, F.npy, Fneu.npy and iscell.npy (the
    `<recording>_els_moco_s2p0` run's suite2p/combined in `<date>_fromHal`) and the onset table
    (`<recording>_els_moco_thor_df_summary_n.csv` in the `_PCA_result` folder).

    Returns dict: genotype, recording, cfg (the YAML), ops, yaml_path, s2p_dir, pca_dir, n_rois, F and Fneu (float64,
    the ROIs with iscell == 1, in suite2p's order: olfdb's roi), onsets (the onset volume of each presentation, in
    time order), pins (the stimulus of each presentation, Yang's odor names).
    """
    if (genotype, rec) not in RECORDINGS:
        raise ValueError(f'{genotype} {rec} is not one of the 19 recordings in olfdb (RECORDINGS)')
    date = rec.split('_')[0]
    y = find_one(drive_root, f'YANGZHENG_0*/0_2P/{date}/{rec}.yaml')
    cfg = yaml.safe_load(y.read_text())
    day = y.parent.parent / f'{date}_fromHal'
    s2p = day / f'{rec}_els_moco_s2p0' / 'suite2p' / 'combined'
    pca = day / f'{rec}_els_moco_s2p0_PCA_result'
    ops = np.load(s2p / 'ops.npy', allow_pickle=True).item()
    F = np.load(s2p / 'F.npy').astype(np.float64)
    Fneu = np.load(s2p / 'Fneu.npy').astype(np.float64)
    iscell = np.load(s2p / 'iscell.npy')
    keep = iscell[:, 0] == 1
    onsets = pd.read_csv(pca / f'{rec}_els_moco_thor_df_summary_n.csv', index_col=0)['VC'].to_numpy(dtype=int)
    pins = np.array(cfg['odor_pin'])[np.array(cfg['odor_stim_seq']) - 1]
    return dict(genotype=genotype, recording=rec, cfg=cfg, ops=ops, yaml_path=y, s2p_dir=s2p, pca_dir=pca,
                n_rois=int(iscell.shape[0]), F=F[keep], Fneu=Fneu[keep], onsets=onsets, pins=pins)


def recording(genotype: str, rec: str, drive_root: Path) -> tuple:
    r = read_suite2p(genotype, rec, drive_root)
    cfg, ops, pca = r['cfg'], r['ops'], r['pca_dir']
    Fc = r['F'] - 0.7 * r['Fneu']
    Fmean = r['F'].mean(axis=1)
    Fneumean = r['Fneu'].mean(axis=1)
    del r['F'], r['Fneu']
    z = zscore(Fc, axis=1)
    row = dict(genotype=genotype, recording=rec, n_rois=r['n_rois'], n_kcs=int(Fc.shape[0]), n_volumes=int(Fc.shape[1]),
               frame_rate=float(cfg['frame_rate']), frame_per_volume=int(cfg['frame_per_volume']), base_s=float(cfg['basePeriodLen']),
               peak_start_s=float(cfg['peak_start']), peak_end_s=float(cfg['peak_end']), odor_pulse_s=float(cfg['odorPulseLen']),
               inter_odor_s=float(cfg['interOdorInterval']), induction_hours=cfg.get('Thirtytwo_hours'),
               **{f'ops_{k}': (ops[k] if not isinstance(ops.get(k), (list, np.ndarray)) else str(ops[k])) for k in OPS_KEYS if k in ops})
    saved = pca / f'{rec}_els_moco_s2p0_Fc_zscored_full_cell.npy'
    if saved.exists():
        zs = np.load(saved)
        row['zscore_max_abs_diff_vs_saved'] = float(np.abs(zs - z).max()) if zs.shape == z.shape else np.nan
        del zs
    vr = cfg['frame_rate'] / cfg['frame_per_volume']
    B, S, P = round(cfg['basePeriodLen'] * vr), round(cfg['peak_start'] * vr), round(cfg['peak_end'] * vr)
    row.update(volume_rate=vr, baseline_volumes=B, window_first=S, window_last=P - 1, window_volumes=P - S)
    onsets, pins = r['onsets'], r['pins']
    K = len(onsets)
    row['n_trials'] = K
    win = np.stack([z[:, n + S:n + P].mean(axis=1) - z[:, n - B:n].mean(axis=1) for n in onsets], axis=1)
    # the same in Fc units (un-normalized), and each KC's noise in the pre-odor windows (each window's mean removed)
    dFc = np.stack([Fc[:, n + S:n + P].mean(axis=1) - Fc[:, n - B:n].mean(axis=1) for n in onsets], axis=1)
    base = np.concatenate([Fc[:, n - B:n] - Fc[:, n - B:n].mean(axis=1, keepdims=True) for n in onsets], axis=1)
    sd_base = np.sqrt((base ** 2).sum(axis=1) / (base.shape[1] - K))
    sd_full = Fc.std(axis=1)
    # slow drift: the SD over trials of the pre-odor means, in units of the whole-trace SD
    drift = np.stack([Fc[:, n - B:n].mean(axis=1) for n in onsets], axis=1).std(axis=1) / sd_full
    resp_csv = pca / f'{rec}_els_moco_s2p0_Fc_zscored_full_cell_resp_df.csv'
    if resp_csv.exists():
        rd = pd.read_csv(resp_csv, index_col=0).to_numpy(dtype=float)
        row['resp_max_abs_diff_vs_saved'] = float(np.abs(rd - win).max()) if rd.shape == win.shape else np.nan
    # olfdb: rows in the same KC order (roi = row), trials matched by odor and the repeat's order of presentation
    om = olfdb_matrix(genotype, rec)
    names = np.array([ODOR_NAMES[p] for p in pins])
    rep = np.zeros(K, dtype=int)
    for i in range(K):
        rep[i] = int((names[:i] == names[i]).sum())
    cols = pd.MultiIndex.from_arrays([names, rep])
    ours = pd.DataFrame(win, columns=cols)
    common = [c for c in ours.columns if c in om.columns]
    if len(om) == len(ours) and common:
        row['olfdb_max_abs_diff'] = float(np.abs(om[common].to_numpy() - ours[common].to_numpy()).max())
        row['olfdb_trials_matched'] = len(common)
    else:
        row['olfdb_max_abs_diff'] = np.nan
        row['olfdb_n_kcs'] = len(om)
    odor_mask = names != 'pfo @ 0'
    odors = [o for o in dict.fromkeys(names[odor_mask])]
    tm_z = np.stack([win[:, names == o].mean(axis=1) for o in odors], axis=1)
    tm_d = np.stack([dFc[:, names == o].mean(axis=1) for o in odors], axis=1)
    ms = (win ** 2).mean(axis=1)
    row.update(ceiling_mean_sq=row['n_volumes'] / (K * (P - S)), max_mean_sq=float(ms.max()), p99_mean_sq=float(np.percentile(ms, 99)),
               max_trial_mean=float(tm_z.max()), frac_window_volumes=K * (P - S) / row['n_volumes'],
               median_scale_ratio=float(np.median(sd_full / sd_base)), median_drift=float(np.median(drift)),
               median_F=float(np.median(Fmean)), median_Fneu=float(np.median(Fneumean)),
               frac_Fc_mean_negative=float((Fc.mean(axis=1) <= 0).mean()))
    kc = pd.DataFrame(dict(genotype=genotype, recording=rec, kc=np.arange(len(Fc)), sd_full=sd_full, sd_base=sd_base,
                           scale_ratio=sd_full / sd_base, drift=drift, mean_sq_resp=ms, F_mean=Fmean, Fneu_mean=Fneumean,
                           n_odors_z1=(tm_z > 1).sum(axis=1)))
    return row, kc, tm_z, tm_d / sd_base[:, None], odors


def rates(tm: np.ndarray, thr: float) -> tuple:
    r = tm > thr
    return float(r.mean()), r.sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--drive-root', type=Path, default=Path('/media/yang'))
    ap.add_argument('--out', type=Path, default=H.shared_dir() / 'notes' / 'yang_response_normalization')
    ap.add_argument('--figures-only', action='store_true', help='only redraw the figure from the saved tables in --out')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        figures(pd.read_csv(args.out / 'recordings.csv'), pd.read_csv(args.out / 'kc_scales.csv'),
                pd.read_csv(args.out / 'amplitude_by_breadth.csv'), args.out)
        return
    rows, kcs, tz, tb = [], [], {}, {}
    for g, rec in RECORDINGS:
        row, kc, tm_z, tm_b, odors = recording(g, rec, args.drive_root)
        rows.append(row)
        kcs.append(kc)
        tz[(g, rec)], tb[(g, rec)] = tm_z, tm_b
        print(f"{g} {rec}: {row['n_kcs']} KCs, {row['n_volumes']} volumes, window {row['window_first']}-{row['window_last']}, "
              f"z vs saved {row.get('zscore_max_abs_diff_vs_saved', np.nan):.2g}, resp vs saved {row.get('resp_max_abs_diff_vs_saved', np.nan):.2g}, "
              f"vs olfdb {row['olfdb_max_abs_diff']:.2g}; mean-square ceiling {row['ceiling_mean_sq']:.2f}, max {row['max_mean_sq']:.2f}", flush=True)
    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(args.out / 'recordings.csv', index=False)
    kc_df = pd.concat(kcs, ignore_index=True)
    kc_df.to_csv(args.out / 'kc_scales.csv', index=False)
    # one threshold on the baseline-noise responses that gives the control flies Yang's mean response rate
    ctrl_rate_z = np.mean([rates(tz[k], 1.0)[0] for k in tz if k[0] == 'TNT3c'])
    lo, hi = 0.1, 100.0
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        r = np.mean([rates(tb[k], mid)[0] for k in tb if k[0] == 'TNT3c'])
        lo, hi = (mid, hi) if r > ctrl_rate_z else (lo, mid)
    thr_b = float(np.sqrt(lo * hi))
    bins = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 99)]
    out = []
    for name, tm, thr in (('whole-trace z (Yang), > 1', tz, 1.0), (f'baseline-noise units, > {thr_b:.2f}', tb, thr_b)):
        for g in ('TNT3c', 'TNT'):
            rs, opk = [], []
            for k in tm:
                if k[0] != g:
                    continue
                r, n = rates(tm[k], thr)
                rs.append(r)
                opk.append([float(((n >= a) & (n <= b)).mean()) for a, b in bins])
            out.append(dict(normalization=name, genotype=g, threshold=thr, response_rate=float(np.mean(rs)), n_recordings=len(rs),
                            **{f'odors_per_kc_{a}_{b}': v for (a, b), v in zip(bins, np.mean(opk, axis=0))}))
    nr = pd.DataFrame(out)
    nr.to_csv(args.out / 'normalization_rates.csv', index=False)
    # amplitude against breadth: responders' mean trial-mean response, by the number of odors responded to (Yang's criterion)
    amp = []
    for k in tz:
        n = (tz[k] > 1).sum(axis=1)
        for nn in range(1, 17):
            sel = n == nn
            if sel.sum() < 5:
                continue
            m = tz[k][sel] > 1
            amp.append(dict(genotype=k[0], recording=k[1], n_odors=nn, n_kcs=int(sel.sum()), amp_z=float(tz[k][sel][m].mean()),
                            amp_base=float(tb[k][sel][m].mean())))
    amp = pd.DataFrame(amp)
    amp.to_csv(args.out / 'amplitude_by_breadth.csv', index=False)
    summ = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='yang_response_normalization.py',
                al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
                baseline_threshold=thr_b, control_rate_z=float(ctrl_rate_z),
                rate_ratio={name: float(nr[(nr.normalization == name) & (nr.genotype == 'TNT')].response_rate.iloc[0]
                                        / nr[(nr.normalization == name) & (nr.genotype == 'TNT3c')].response_rate.iloc[0])
                            for name in nr.normalization.unique()},
                scale_ratio_by_breadth={int(n): float(kc_df[kc_df.n_odors_z1 == n].scale_ratio.median()) for n in range(0, 12)
                                        if (kc_df.n_odors_z1 == n).sum() >= 20})
    (args.out / 'summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    figures(rec_df, kc_df, amp, args.out)
    print(nr.round(4).to_string())
    print(json.dumps(summ, indent=1))


def figures(rec_df: pd.DataFrame, kc_df: pd.DataFrame, amp: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, ink2, grid = '#0b0b0b', '#52514e', '#e6e5e1'
    col = {'TNT3c': '#2a78d6', 'TNT': '#eb6834'}
    lab = {'TNT3c': 'control flies', 'TNT': 'APL output blocked (TNT)'}
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': ink2,
                         'xtick.color': ink2, 'ytick.color': ink2})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    for g in ('TNT3c', 'TNT'):
        d = kc_df[kc_df.genotype == g].groupby('n_odors_z1').scale_ratio.median()
        d = d[kc_df[kc_df.genotype == g].groupby('n_odors_z1').size() >= 20]
        ax.plot(d.index, d.values, '-o', color=col[g], lw=2, ms=5, label=lab[g])
    ax.set_xlabel('odors the KC responds to (trial mean > 1 z)')
    ax.set_ylabel('whole-trace SD / baseline-noise SD (median)')
    ax.set_title('the z unit grows with the KC\'s responses', color=ink)
    ax.legend(frameon=False)
    ax.grid(axis='y', color=grid)
    ax = axes[1]
    for g in ('TNT3c', 'TNT'):
        d = amp[amp.genotype == g].groupby('n_odors')[['amp_z', 'amp_base']].median()
        ax.plot(d.index, d.amp_z, '-o', color=col[g], lw=2, ms=5, label=f'{lab[g]}, whole-trace z')
        ax.plot(d.index, d.amp_base / d.amp_base.iloc[0] * d.amp_z.iloc[0], '--', color=col[g], lw=1.5,
                label=f'{lab[g]}, baseline-noise units (scaled to match at 1)')
    ax.set_xlabel('odors the KC responds to')
    ax.set_ylabel('responders\' mean response (median over recordings)')
    ax.set_title('response amplitude against breadth\n(solid: whole-trace z; dashed: baseline-noise units,\nscaled to the solid line at 1 odor)', color=ink)
    ax.grid(axis='y', color=grid)
    ax = axes[2]
    ceiling = float(rec_df.ceiling_mean_sq.median())
    bins_ = np.linspace(0, max(ceiling, float(kc_df.mean_sq_resp.max())) * 1.05, 60)
    for g in ('TNT3c', 'TNT'):
        ax.hist(kc_df[kc_df.genotype == g].mean_sq_resp, bins=bins_, color=col[g], alpha=0.55, label=lab[g], log=True)
    ax.axvline(ceiling, color=ink2, ls=':', lw=1.5)
    ax.text(ceiling, 0.55, 'ceiling set by the\nwhole-trace z-score ', color=ink2, fontsize=8, va='center', ha='right',
            transform=ax.get_xaxis_transform())
    ax.set_xlabel('KC\'s mean squared response over its 51 trials (z^2)')
    ax.set_ylabel('KCs (all 19 recordings)')
    ax.set_title('each KC\'s mean squared response\nand the ceiling of the whole-trace z-score', color=ink)
    ax.legend(frameon=False)
    ax.grid(axis='y', color=grid)
    fig.tight_layout()
    fig.savefig(out / 'yang_response_normalization.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
