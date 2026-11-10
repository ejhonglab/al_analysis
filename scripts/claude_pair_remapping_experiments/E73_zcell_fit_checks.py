#!/usr/bin/env python3
"""E73 part C: checks of the per-KC session z-score options of E10_joint_fit.py and E43_model_trial_noise.py.

1. Options off: the first evaluation of E10_joint_fit.py started exactly at a saved fit (`--init-from-exact
   --maxfev 1`) reproduces that fit's best evaluation in its saved history, every column (C2, the first joint
   fit's family fitted under the APL-effect loss on both readings, E67 rate0.068_aplboth_k2_cap3, run on viki;
   E69's fit 1, the same with the firing ceiling at strength 1, run on hal).
2. Options on (`--apl-readout zcell --rate-criterion zcell`): the first evaluation from each of part C's starts
   (C2; E69's fit 1; the reference model, E44 thr40_gain1.5_rate0.068_k2_cap3; E69's fit 3) reproduces part B's
   row of that variant (results/E73_per_kc_normalization/by_variant.csv, reading_def zcell_saturating, graded):
   the APL-effect loss with its parts, the response rates and their ratio, the response rate ρ, the odors-per-KC
   distributions, the amplitudes and the calibration. Also fit 4's form (the APL-effect term under the reading,
   the rate terms under the two-spike criterion) from C2.
3. The seconds per evaluation with the options on and off (the history's time_s).
4. E43_model_trial_noise.py --part B --analyse-only --zcell on E69's saved trials reproduces part B's reading of
   those trials (reading_def zcell_saturating_trials, all three readings) and of the deterministic runs.

E10_joint_fit.py runs in its own process with the results folder of the scratch experiment E73_zcell_check
redirected to outputs/E73_per_kc_normalization/partC_checks/ (nothing is written to the shared results); the
E43 check writes its tables there too. Writes partC_checks/checks.json and prints a summary.
Usage (from this folder; two E10 processes at a time, 16 threads each):
    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 nice -n 10 ../../venv/bin/python -u E73_zcell_fit_checks.py [--only NAME ...] [--jobs 2]
    ../../venv/bin/python E73_zcell_fit_checks.py --compare-only   (the comparisons again, from the saved histories)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H

SCRATCH = 'E73_zcell_check'
ROOT = H.HERE / 'outputs' / 'E73_per_kc_normalization' / 'partC_checks'
COMMON = ['-k', '2', '--loss', 'apl', '--rate-target', 'ctrl_frac', '--aplpn-max', '3', '--maxfev', '1', '--init-from-exact',
    '--experiment', SCRATCH]
CEIL1 = ['--kc-refractory-period', '0.003', '--kc-adapt-inc', '0.041417845143644', '--kc-adapt-tau', '1.0']
CEIL2 = ['--kc-refractory-period', '0.003', '--kc-adapt-inc', '0.1113973859994802', '--kc-adapt-tau', '1.0']
REF = ['--claw-thr', '40', '--glom-gain', '1.5']
STARTS = {
    'C2': ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3', []),
    'fit1': ('E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.04142t1000_k2', CEIL1),
    'ref': ('E44_operating_point', 'thr40_gain1.5_rate0.068_k2_cap3', REF),
    'fit3': ('E69_kc_firing_ceiling', 'thr40_gain1.5_rate0.068_aplboth_refr3_adapt0.1114t1000_k2', REF + CEIL2),
}
# name: (start, options); 'off' runs are compared with the start's history, the others with part B's row
RUNS = {
    'off_C2': ('C2', ['--apl-readout', 'both']),
    'off_fit1': ('fit1', ['--apl-readout', 'both']),
    'on_C2': ('C2', ['--apl-readout', 'zcell', '--rate-criterion', 'zcell']),
    'on_fit1': ('fit1', ['--apl-readout', 'zcell', '--rate-criterion', 'zcell']),
    'on_ref': ('ref', ['--apl-readout', 'zcell', '--rate-criterion', 'zcell']),
    'on_fit3': ('fit3', ['--apl-readout', 'zcell', '--rate-criterion', 'zcell']),
    'aplonly_C2': ('C2', ['--apl-readout', 'zcell']),
}
# part B's by_variant columns and the history columns holding them
APL_COLS = {'apl_zcell_rmse': 'rmse', 'apl_zcell_mean': 'mean', 'apl_zcell_pattern': 'rmse_centered',
    'apl_zcell_per_odor': 'per_odor', 'apl_zcell_pair_specific': 'pair_specific', 'apl_zcell_rho': 'spearman'}
PARAM_COLS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')
E69_TRIAL_MODELS = ('rate0.068_aplboth_refr3_adapt0.04142t1000_k2', 'rate0.068_aplboth_refr3_adapt0.1114t1000_k2')


def run_e10(root: Path, e10_args: list) -> None:
    """E10_joint_fit.main() with the scratch experiment's results and outputs folders under `root`."""
    orig_r, orig_o = H.results_dir, H.outputs_dir

    def results_dir(e):
        if e != SCRATCH:
            return orig_r(e)
        d = root / 'results'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def outputs_dir(e):
        if e != SCRATCH:
            return orig_o(e)
        d = root / 'outputs'
        d.mkdir(parents=True, exist_ok=True)
        return d

    H.results_dir, H.outputs_dir = results_dir, outputs_dir
    import E10_joint_fit
    sys.argv = ['E10_joint_fit.py', *e10_args]
    E10_joint_fit.main()


def launch(name: str) -> dict:
    start, opts = RUNS[name]
    exp, tag, extra = STARTS[start]
    root = ROOT / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    args = COMMON + ['--init-from', tag, '--init-from-experiment', exp] + extra + opts + (['--tag', '_cap3'] if 'cap3' in tag else [])
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-e10', str(root), '--', *args]
    t0 = time.time()
    with open(root / 'log.txt', 'w') as log:
        p = subprocess.run(cmd, cwd=H.HERE, stdout=log, stderr=subprocess.STDOUT)
    hist = sorted((root / 'results').glob('history_*.csv'))
    print(f'{name}: exit {p.returncode}, {time.time() - t0:.0f} s, {[h.name for h in hist]}', flush=True)
    return dict(name=name, returncode=p.returncode, seconds=time.time() - t0, args=args,
        history=str(hist[0]) if hist else None)


def read_csv(p) -> pd.DataFrame:
    return pd.read_csv(p, float_precision='round_trip')


def compare_off(r: dict) -> dict:
    """The first evaluation against the saved history's evaluation at the same parameters (its best)."""
    exp, tag, _ = STARTS[RUNS[r['name']][0]]
    new = read_csv(r['history']).iloc[0]
    old = read_csv(H.results_dir(exp) / f'history_{tag}.csv')
    row = old.loc[old['loss'].idxmin()]
    cols = [c for c in old.columns if c not in ('eval', 'time_s')]
    missing = [c for c in cols if c not in new.index]
    extra = [c for c in new.index if c not in old.columns and c not in ('eval', 'time_s')]
    diff = {}
    for c in cols:
        if c in new.index and not (new[c] == row[c] or (pd.isna(new[c]) and pd.isna(row[c]))):
            diff[c] = dict(new=float(new[c]), saved=float(row[c]), abs=float(abs(new[c] - row[c])),
                ulps=float(abs(new[c] - row[c]) / np.spacing(abs(row[c]))) if row[c] != 0 else np.inf)
    # (the parameters of an exact start pass through fits.csv and the optimizer's unit-cube transform, so they can
    # differ in the last digits; the statistics show whether the simulation and the loss are the same)
    params = set(PARAM_COLS)
    stat_diff = {c: v for c, v in diff.items() if c not in params}
    return dict(saved_eval=int(row['eval']), n_columns=len(cols), n_statistics=len([c for c in cols if c not in params]),
        statistics_identical=not stat_diff and not missing and not extra, differing_statistics=stat_diff,
        differing_parameters={c: v for c, v in diff.items() if c in params}, missing=missing, extra=extra,
        time_s_new=float(new['time_s']), time_s_saved=float(row['time_s']))


def compare_on(r: dict, byv: pd.DataFrame) -> dict:
    """The first evaluation's reading against part B's row of the start (zcell_saturating, graded)."""
    exp, tag, _ = STARTS[RUNS[r['name']][0]]
    new = read_csv(r['history']).iloc[0]
    b = byv[(byv.folder == exp) & (byv.tag == tag) & (byv.reading_def == 'zcell_saturating') & (byv.reading == 'graded')]
    if len(b) != 1:
        return dict(error=f'{len(b)} part B rows')
    b = b.iloc[0]
    pairs = dict(APL_COLS)
    pairs.update({c: c[len('zcell_'):] for c in new.index if c.startswith('zcell_') and c[len('zcell_'):] in b.index})
    worst, rel, n = 0.0, 0.0, 0
    vals = {}
    for hc, bc in pairs.items():
        x, y = float(new[hc]), float(b[bc])
        if np.isnan(x) and np.isnan(y):
            continue
        n += 1
        d = abs(x - y)
        worst = max(worst, d)
        rel = max(rel, d / max(abs(y), 1e-300))
        vals[hc] = (x, y)
    key = {k: vals[k] for k in ('apl_zcell_rmse', 'apl_zcell_mean', 'apl_zcell_pattern', 'apl_zcell_rho', 'zcell_rate_on',
        'zcell_rate_off', 'zcell_rate_ratio', 'zcell_rate_rho_on', 'zcell_opk_off_tv', 'zcell_K', 'zcell_noise') if k in vals}
    terms = {c: float(new[c]) for c in new.index if c.startswith('term_')}
    return dict(n_compared=n, max_abs_diff=worst, max_rel_diff=rel, key_values=key, terms=terms, loss=float(new['loss']),
        time_s=float(new['time_s']), two_spike=dict(rate_on=float(new['rate_on']), rate_off=float(new['rate_off']),
        on_spearman=float(new['on_spearman']), apl_cal_rmse=float(new['apl_cal_rmse'])))


def check_e43() -> dict:
    """E43 --part B --analyse-only --zcell on E69's saved trials, against part B's reading of the same trials."""
    root = ROOT / 'e43'
    if root.exists():
        shutil.rmtree(root)
    (root / 'results').mkdir(parents=True)
    src = H.results_dir('E69_kc_firing_ceiling') / 'trial_noise'
    shutil.copy(src / 'noise_levels.json', root / 'results' / 'noise_levels.json')
    out = H.HERE / 'outputs' / 'E69_kc_firing_ceiling' / 'trial_noise'
    if not out.exists():
        out = H.shared_dir() / 'data' / 'model_outputs' / 'E69_kc_firing_ceiling' / 'trial_noise'
    cmd = [sys.executable, '-u', 'E43_model_trial_noise.py', '--part', 'B', '--analyse-only', '--zcell',
        '--model-from', f'E69_kc_firing_ceiling:{E69_TRIAL_MODELS[0]}', '--results-dir', str(root / 'results'),
        '--outputs-dir', str(out)]
    t0 = time.time()
    with open(root / 'log.txt', 'w') as log:
        p = subprocess.run(cmd, cwd=H.HERE, stdout=log, stderr=subprocess.STDOUT)
    res = dict(returncode=p.returncode, seconds=time.time() - t0, outputs_dir=str(out))
    path = root / 'results' / 'zcell_noise.csv'
    if p.returncode or not path.exists():
        return res
    z = read_csv(path)
    byv = read_csv(H.results_dir('E73_per_kc_normalization') / 'by_variant.csv')
    import E73_per_kc_normalization as E73
    cmp = []
    for tag in E69_TRIAL_MODELS:
        model = f'E69-{tag}'
        pat = E73.TRIAL_SETS.get(('E69_kc_firing_ceiling', tag), '')
        for rd_def, level in (('zcell_saturating_trials', None), ('zcell_saturating', 'deterministic')):
            for rd in H.ZCELL_READINGS:
                b = byv[(byv.folder == 'E69_kc_firing_ceiling') & (byv.tag == tag) & (byv.reading_def == rd_def) & (byv.reading == rd)]
                m = z[(z.model == model) & (z.reading_def == rd_def) & (z.reading == rd)]
                if level is not None:
                    m = m[m.level == level]
                else:
                    # the level part B read: the sigma in TRIAL_SETS' file pattern
                    sig = float(pat.split('_sigma')[1].split('_')[0]) if '_sigma' in pat else np.nan
                    m = m[np.isclose(m.sigma, sig)]
                if len(b) != 1 or len(m) != 1:
                    cmp.append(dict(model=model, reading_def=rd_def, reading=rd, error=f'{len(b)} part B rows, {len(m)} E43 rows'))
                    continue
                b, m = b.iloc[0], m.iloc[0]
                cols = [c for c in b.index if c in m.index and pd.api.types.is_number(b[c]) and c not in ('q',)
                    and not (pd.isna(b[c]) and pd.isna(m[c]))]
                d = np.array([abs(float(m[c]) - float(b[c])) for c in cols])
                cmp.append(dict(model=model, reading_def=rd_def, reading=rd, n_compared=len(cols), max_abs_diff=float(np.nanmax(d)),
                    rmse=(float(m['rmse']), float(b['rmse'])), mean=(float(m['mean']), float(b['mean']))))
    res['comparisons'] = cmp
    return res


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == '--run-e10':
        root = Path(sys.argv[2])
        run_e10(root, sys.argv[4:])
        return
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', nargs='+', default=None, help=f'runs to do (of {list(RUNS)} and e43)')
    ap.add_argument('--jobs', type=int, default=2, help='E10 processes at a time')
    ap.add_argument('--compare-only', action='store_true', help='no runs: compare the histories already in partC_checks/')
    args = ap.parse_args()
    names = [n for n in RUNS if args.only is None or n in args.only]
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.compare_only:
        runs = []
        for n in names:
            h = sorted((ROOT / n / 'results').glob('history_*.csv'))
            if h:
                runs.append(dict(name=n, returncode=0, history=str(h[0])))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            runs = list(ex.map(launch, names))
    out_path = ROOT / 'checks.json'
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    byv = read_csv(H.results_dir('E73_per_kc_normalization') / 'by_variant.csv')
    for r in runs:
        if r['returncode'] or r['history'] is None:
            out[r['name']] = dict(r, error='run failed; see log.txt')
            continue
        out[r['name']] = dict(out.get(r['name'], {}), **r, **(compare_off(r) if r['name'].startswith('off_') else compare_on(r, byv)))
    if (args.only is None or 'e43' in args.only) and not args.compare_only:
        out['e43'] = check_e43()
    out['host'] = os.uname().nodename
    out['commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip()
    out['dirty'] = bool(subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=H.HERE, capture_output=True,
        text=True).stdout.strip())
    out_path.write_text(json.dumps(out, indent=1, default=str) + '\n')
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
