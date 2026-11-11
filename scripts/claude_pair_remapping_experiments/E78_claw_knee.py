#!/usr/bin/env python3
"""E78: the claw's transfer under the main target. A high per-claw knee without a dead zone on the best family (C2, the
breadth-2 APL-effect fit, E67 rate0.068_aplboth_k2_cap3), for the odor pairs with near-identical input that the flies'
KCs hold apart between claw and soma.

The entry (experiment_priorities.md, item 3; planned by a Fable subagent, claude-fable-5-2, spawned from hal 2026-11-10
17:47) is the spec. Parts:

- A: the hard knee at 100 and 130 Hz in the best family, `E10_joint_fit.py` as it is (run by hal's session; fits.csv and
  histories in results/E78_claw_knee/).
- B (`--part B`, no simulations but one run per model for the time courses): the archive's knee variants on E77's kinds
  of pairs and on the six near pairs, each against its own base (archive_knee_variants.csv); the six pairs through the
  saved trial-noise rows (noise_six_pairs.csv); who selects the responders (who_selects.csv).
- C (`--part C`): fixed scales on C2 (and, for the five best points, on E69's fit 1 and E72's P1 fit), both response
  rates held as E77's screen holds them: the somatic threshold bisected to the start's APL-off response count, the
  breadth weights recomputed for the transfer, the APL->KC scale (KC->APL and bouton->APL tied to it) bisected to the
  start's APL-on response count; KC-KC and APL->bouton scales the start's. C1: knees 100 / 115 / 130 Hz with a
  sub-threshold gain g of 0 / 0.1 / 0.25 / 0.5, 70 Hz with g 0 / 0.25, a softplus width of 15 Hz at the two best;
  C2: the knee relative to the odor (the per-odor external inhibition at the boutons, E9's `ext_inh_bouton`, at 0.5 /
  0.7 / 0.85 times the odor's largest bouton rate, in both conditions). Writes sweep.csv, sweep_pairs.csv,
  sweep_params.json; `--part C-low` runs the five best at Yang's low concentrations (E49's script, --into).
- gate (`--part gate`): the entry's four conditions on part C's points (gate.json).
- D (`--part D`): the launch commands of the fits (E10_joint_fit.py with --claw-thr / --claw-thr-subgain /
  --claw-thr-width from the gate point's parameters, on the real wiring and P1, P2, M1), the held-out run on P3, P4, M2,
  M3, the paired and population tables.
- E (`--part E`): every fit and gate point under the per-KC reading (E73's scorer), the release by claw count.
- summarize (`--part summarize`): comparison_table.md and the figures.

Part C's model (`KneeModel`): the start on E77's fixed-scale runner (E77_input_screen's `build_c2_simulator`, the start's
parameters, wiring and firing ceiling; checked to reproduce the start's saved spike counts), the transfer as model kwargs
(claw_thr, claw_thr_subgain, claw_thr_width; the odor-relative knee as ext_inh_bouton), both response rates held. Two runs:
- 'prepass' (`--breadth prepass`, the default; planned at hal 21:39 after the first run): each point's breadth weights from
  E10_joint_fit.transfer_prepass on the plain bouton base of the start's wiring with the whole transfer (its threshold
  re-tuned to the transfer-free run's APL-off response count; at the identity C2's breadth exactly, `--part check-C`), and
  with `--kckc tied` (default) the KC-KC scale multiplied by the somatic threshold's factor when the APL-off count is
  held (KC-KC inhibition is driven by the other KCs' spikes, so with the scale fixed a 100-130 Hz hard knee cannot reach
  the APL-off count); the two best knee points also with the KC-KC scale fixed, as controls (not gated). Tags
  <start>_pp_<point>[_kf]; points_prepass/<start>/; sweep.csv, gate.json.
- 'held' (the first run, bd78d6e): the breadth from the threshold-held APL-off run (breadth_from='held'), the KC-KC scale
  at the start's; 'identity_held' is the start with only that breadth. Kept for the record: tags <start>_<point>;
  points/<start>/; sweep_held.csv, gate_identity_held.json.
The 'identity' point is the start itself. The gate compares with C2 itself; points whose bisection ended at a bound
(rates_held false) fail it and are left out of the best points.

Outputs: <shared>/results/E78_claw_knee/ (points_prepass/<start>/ and points/<start>/: <point>.json, _pairs.csv,
_record.json and provenance_*.json; sweep[_held].csv, sweep_pairs[_held].csv, sweep_params[_held].json, and
stats_variants.csv and model_pair_distances_long.csv of both, built from them; check_prepass_identity.json;
archive_knee_variants.csv, noise_six_pairs.csv, who_selects.csv, pn_rates.csv; low_conc/; gate.json, gate.csv;
init_params.json, partD_commands.json, paired.csv, population.csv; release.csv; comparison_table[_held].md, E78_*.png); spike
counts and responses under ./outputs/E78_claw_knee/C/<start>_<point>_<on|acute_off>_<spike_counts|responses>.parquet.

Usage (from this folder; at most two 16-thread processes on hal, under nice, as systemd-run --user units for long runs):
    ../../venv/bin/python E78_claw_knee.py --part check-C   (the pre-pass run's identity check on C2; before launching)
    ../../venv/bin/python E78_claw_knee.py --part C-launch  (the pre-pass run as four chained units; `c_launch`), or step by
        step:
    OMP_NUM_THREADS=16 OMP_THREAD_LIMIT=16 nice -n 10 ../../venv/bin/python -u E78_claw_knee.py --part C [--only ID ...]
        (C2; split the points over two processes with --only; run once more without --only for the softplus width and the
        KC-KC-fixed controls at the two best knee points; --breadth held / --kckc fixed for the other settings)
    ... --part C-low --best             (the five best and C2 at the low concentrations)
    ... --part C --start fit1 --best    and    ... --part C --start P1 --best    (the five best repeated)
    ../../venv/bin/python E78_claw_knee.py --part gate [--breadth held --gate-ref identity_held] | --part summarize
        [--breadth held] | --part B | --part E
    ../../venv/bin/python E78_claw_knee.py --part check-e10 [--baseline COMMIT] [--jobs N]   (E10's options checked)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H

EXPERIMENT = 'E78_claw_knee'
C2 = ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3')
# the starts of part C (experiment, fit tag, wiring options as E10_joint_fit.py takes them)
STARTS = {
    'C2': (C2[0], C2[1], {}),
    'fit1': ('E69_kc_firing_ceiling', 'rate0.068_aplboth_refr3_adapt0.04142t1000_k2', {}),
}
# E72's re-drawn wirings (part B fits; P: within-type claw permutations, M: the co-input-preserving shuffle's matched
# wirings, E59), for the repeats of part C and part D's starts and held-out runs
E72_FOLDER = 'E72_wiring_population'
M_FILE = 'data/model_outputs/E59_coinput_shuffle/wPNKC_curveball_within_type-seed{n}_matched.parquet'
for _n in (1, 2, 3, 4):
    STARTS[f'P{_n}'] = (E72_FOLDER, f'wiring-claw_permutation_within_type-seed{_n}_rate0.068_aplboth_k2_cap3',
        dict(wiring='claw_permutation_within_type', wiring_seed=_n))
for _n in (1, 2, 3):
    STARTS[f'M{_n}'] = (E72_FOLDER, f'wiring-curveball_within_type_matched-seed{_n}_rate0.068_aplboth_k2_cap3',
        dict(wiring_file=M_FILE.format(n=_n), wiring_tag=f'curveball_within_type_matched-seed{_n}'))
FIT_WIRINGS = ('C2', 'P1', 'P2', 'M1')
HELD_OUT = ('P3', 'P4', 'M2', 'M3')
ESTER = ('2h', 'IaA', 'pa', 'eb', 'ep', 'ms')
ALCOHOL = ('1-8ol', '1-5ol', '1-6ol', '6al')
# the six pairs with an ORN distance below 0.25 (the entry's table) and E77's kinds of ester pairs
SIX = (('eb', 'ep'), ('6al', 't2h'), ('IaA', 'pa'), ('2h', 'pa'), ('1-5ol', '1-6ol'), ('2h', 'IaA'))
NEAREST = (('2h', 'IaA'), ('2h', 'pa'), ('IaA', 'pa'), ('eb', 'ep'))
BLOCK = tuple((a, b) for a in ('2h', 'IaA', 'pa') for b in ('eb', 'ep'))
MS_PAIRS = tuple(('ms', o) for o in ('2h', 'IaA', 'pa', 'eb', 'ep'))
KNEES = (100.0, 115.0, 130.0)
SUBGAINS = (0.0, 0.1, 0.25, 0.5)
LINK_KNEE, LINK_SUBGAINS = 70.0, (0.0, 0.25)
SOFT_WIDTH = 15.0
REL_LEVELS = (0.5, 0.7, 0.85)
N_BEST = 5
# the gate (the entry's part D, against C2 itself; hal's planning, 21:39): (i) ester group's error reduced by a third or
# more and the six pairs' raw APL effect raised by 0.03 or more; (ii) readout loss 0.060 or lower on C2 (the firing
# ceiling's points within 0.006 of their own start, the re-drawn wirings' within 0.004); (iii) every odor keeps half of C2's
# response fraction in both conditions, response rate ρ with APL on >= 0.50; (iv) 2h at 10^-5 keeps half the fraction C2
# recruits
GATE = dict(ester_reduction=1 / 3, six_raise=0.03, loss_max_c2=0.060, loss_tol=0.004, loss_tol_ceiling=0.006, odor_frac_keep=0.5,
    rate_rho_min=0.50, low_conc_keep=0.5)


# ---------------------------------------------------------------------------
# checks of E10_joint_fit.py's --claw-thr-subgain / --claw-thr-width (before they are pushed)
CHECK_SCRATCH = 'E78_e10_check'
CHECK_COMMON = ['-k', '2', '--loss', 'apl', '--rate-target', 'ctrl_frac', '--aplpn-max', '3', '--maxfev', '1',
    '--init-from-exact', '--experiment', CHECK_SCRATCH, '--tag', '_cap3']
# name: (experiment, fit tag, E10 options); each run starts exactly at the fit and its first evaluation is compared with
# the fit's best evaluation in its saved history, every statistic. 'off_*': the options off; 'zero_*': the options on at
# values that change nothing (a sub-threshold gain of 0, a width of 0: the hard knee)
CHECK_RUNS = {
    'off_C2': (C2[0], C2[1], ['--apl-readout', 'both']),
    'off_P1': ('E72_wiring_population', 'wiring-claw_permutation_within_type-seed1_rate0.068_aplboth_k2_cap3',
        ['--apl-readout', 'both', '--wiring', 'claw_permutation_within_type', '--wiring-seed', '1']),
    'off_thr40': ('E67_apl_effect_fits', 'thr40_rate0.068_aplcal_k2_cap3', ['--apl-readout', 'cal', '--claw-thr', '40']),
    'zero_thr40_sg0': ('E67_apl_effect_fits', 'thr40_rate0.068_aplcal_k2_cap3',
        ['--apl-readout', 'cal', '--claw-thr', '40', '--claw-thr-subgain', '0']),
    'zero_thr40_w0': ('E67_apl_effect_fits', 'thr40_rate0.068_aplcal_k2_cap3',
        ['--apl-readout', 'cal', '--claw-thr', '40', '--claw-thr-width', '0']),
}
CHECK_PARAM_COLS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')


def check_root() -> Path:
    return H.HERE / 'outputs' / EXPERIMENT / 'e10_checks'


def _run_e10_redirected(root: Path, e10_args: Sequence[str]) -> None:
    """E10_joint_fit.main() with the scratch experiment's results and outputs folders under `root` (nothing is written to
    the shared results). With the environment variable E78_E10_DIR, E10_joint_fit.py is imported from that folder (a
    baseline of an earlier commit)."""
    import os
    import sys
    orig_r, orig_o = H.results_dir, H.outputs_dir

    def redirect(orig, sub):
        def f(e):
            if e != CHECK_SCRATCH:
                return orig(e)
            d = root / sub
            d.mkdir(parents=True, exist_ok=True)
            return d
        return f

    H.results_dir, H.outputs_dir = redirect(orig_r, 'results'), redirect(orig_o, 'outputs')
    old = os.environ.get('E78_E10_DIR')
    if old:
        # a baseline run: E10_joint_fit.py as of an earlier commit, from that folder
        sys.path.insert(0, old)
    import E10_joint_fit
    print(f'E10_joint_fit from {E10_joint_fit.__file__}', flush=True)
    sys.argv = ['E10_joint_fit.py', *e10_args]
    E10_joint_fit.main()


def compare_history(new_path: Path, experiment: str, tag: str) -> dict:
    """The first evaluation of a check run against the saved history's best evaluation: every column but the evaluation
    number and its time; statistics must be identical (the parameters of an exact start pass through fits.csv and the
    optimizer's unit-cube transform, so they may differ in the last digits)."""
    new = pd.read_csv(new_path, float_precision='round_trip').iloc[0]
    old = pd.read_csv(H.results_dir(experiment) / f'history_{tag}.csv', float_precision='round_trip')
    row = old.loc[old['loss'].idxmin()]
    cols = [c for c in old.columns if c not in ('eval', 'time_s')]
    diff = {}
    for c in cols:
        if c in new.index and not (new[c] == row[c] or (pd.isna(new[c]) and pd.isna(row[c]))):
            diff[c] = dict(new=float(new[c]), saved=float(row[c]), abs=float(abs(new[c] - row[c])))
    missing = [c for c in cols if c not in new.index]
    extra = [c for c in new.index if c not in old.columns and c not in ('eval', 'time_s')]
    stat_diff = {c: v for c, v in diff.items() if c not in CHECK_PARAM_COLS}
    return dict(saved_eval=int(row['eval']), n_columns=len(cols), statistics_identical=not stat_diff and not missing and not extra,
        differing_statistics=stat_diff, differing_parameters={c: v for c, v in diff.items() if c in CHECK_PARAM_COLS},
        missing=missing, extra=extra, loss_new=float(new['loss']), loss_saved=float(row['loss']))


def compare_runs(a: Path, b: Path) -> dict:
    """Two check runs' first evaluations, every column but the evaluation number and its time."""
    x = pd.read_csv(a, float_precision='round_trip').iloc[0]
    y = pd.read_csv(b, float_precision='round_trip').iloc[0]
    cols = sorted((set(x.index) | set(y.index)) - {'eval', 'time_s'})
    d = [c for c in cols if not (c in x.index and c in y.index and (x[c] == y[c] or (pd.isna(x[c]) and pd.isna(y[c]))))]
    return dict(n_columns=len(cols), identical=not d, differing=d)


def check_e10(only: Optional[Sequence[str]] = None, jobs: int = 1, baseline: Optional[str] = None) -> dict:
    """Runs CHECK_RUNS (one process each, `jobs` at a time) and compares each with its fit's history; with `baseline` (a
    commit), the 'off_*' runs again with E10_joint_fit.py as of that commit (names base_*), each compared with its run on
    the current code; the 'zero_*' runs are compared with off_thr40. Writes outputs/E78_claw_knee/e10_checks/checks.json
    and returns it."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor
    jobs_list = [(n, *v, None) for n, v in CHECK_RUNS.items() if only is None or n in only]
    if baseline:
        old_dir = check_root() / f'_E10_{baseline[:12]}'
        old_dir.mkdir(parents=True, exist_ok=True)
        src = subprocess.run(['git', 'show', f'{baseline}:./E10_joint_fit.py'], cwd=H.HERE, capture_output=True, text=True, check=True).stdout
        (old_dir / 'E10_joint_fit.py').write_text(src)
        jobs_list += [('base' + n[3:], *CHECK_RUNS[n], str(old_dir)) for n, *_ in list(jobs_list) if n.startswith('off_')]

    def one(job):
        name, exp, tag, opts, old_dir = job
        root = check_root() / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        args = CHECK_COMMON + ['--init-from', tag, '--init-from-experiment', exp] + list(opts)
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-e10', str(root), '--', *args]
        env = dict(os.environ, **({'E78_E10_DIR': old_dir} if old_dir else {}))
        t0 = time.time()
        with open(root / 'log.txt', 'w') as log:
            p = subprocess.run(cmd, cwd=H.HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
        hist = sorted((root / 'results').glob('history_*.csv'))
        r = dict(returncode=p.returncode, seconds=time.time() - t0, args=args, e10_dir=old_dir, history=str(hist[0]) if hist else None)
        if p.returncode == 0 and hist:
            r.update(compare_history(hist[0], exp, tag))
        print(f"{name}: exit {p.returncode}, {r['seconds']:.0f} s; statistics identical to the saved history: "
            f"{r.get('statistics_identical')}; loss {r.get('loss_new')} vs {r.get('loss_saved')}", flush=True)
        return name, r

    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as ex:
        res = dict(ex.map(one, jobs_list))
    path = check_root() / 'checks.json'
    allr = json.loads(path.read_text()) if path.exists() else {}
    allr.update(res)
    for name, r in list(allr.items()):
        if not isinstance(r, dict) or not r.get('history'):
            continue
        ref = None
        if name.startswith('base_') and allr.get('off_' + name[5:], {}).get('history'):
            ref = 'off_' + name[5:]
        elif name.startswith('zero_') and allr.get('off_thr40', {}).get('history'):
            ref = 'off_thr40'
        if ref:
            r[f'vs_{ref}'] = compare_runs(Path(r['history']), Path(allr[ref]['history']))
            print(f"{name} vs {ref}: identical {r[f'vs_{ref}']['identical']} {r[f'vs_{ref}']['differing']}", flush=True)
    res = allr
    old = res
    old.update(host=os.uname().nodename, commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True,
        text=True).stdout.strip(), dirty_files=subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no', '--', '.'],
        cwd=H.HERE, capture_output=True, text=True).stdout.split('\n'))
    path.write_text(json.dumps(old, indent=1, default=str) + '\n')
    return old


# ---------------------------------------------------------------------------
# part B
# the archive's knee variants: folder -> rule giving (base folder, base tag, knee in Hz) for a tag, or None to skip
DEFAULT_TOP_PN = None   # filled from E32's default PN rates


def _knee_of(tag: str) -> float:
    import re
    m = re.search(r'thr(\d+(?:\.\d+)?)', tag)
    return float(m.group(1)) if m else np.nan


def archive_variant_list() -> List[dict]:
    """The archive's knee variants with their own bases: E16's per-claw threshold variants (bouton and claw-dynamics
    bases, with and without a saturation) against the base without a knee; E22's soft knees against the hard 70 Hz
    knee; E26's power transfers against the same knee with power 1; E41's sub-threshold gains (part A on the base, part
    B on E34's fits) against the same knee with gain 0; E32's antennal-lobe variants on the 70 Hz knee against its
    default (the PN gain halved is inhsc0.5x); the knee fits of the APL-effect fits (E67) against the same fit without
    the knee where one exists; E72's 40 Hz fits (part C) against the same wiring's part B fit; this experiment's part A
    fits against C2. Each with the knee and the base's largest PN rate (E32's saved rates)."""
    res = H.shared_dir() / 'results'
    tops = {p.stem.replace('glomerulus_pn_rates_', ''): float(pd.read_csv(p, index_col=0).max().max())
        for p in (res / 'E32_pn_stage').glob('glomerulus_pn_rates_*.csv')}
    top0 = tops['default']
    out = []

    def tags(folder):
        p = res / folder / 'model_pair_distances_long.csv'
        return list(dict.fromkeys(pd.read_csv(p, usecols=['tag'])['tag'].astype(str))) if p.exists() else []

    for t in tags('E16_claw_saturation'):
        if 'thr' in t:
            out.append(dict(folder='E16_claw_saturation', tag=t, base_folder='E16_claw_saturation',
                base_tag=t.split('_')[0] + '_nocap', knee=_knee_of(t), top_pn=top0))
    for t in tags('E22_soft_knee'):
        if t != 'thr70_w0':
            out.append(dict(folder='E22_soft_knee', tag=t, base_folder='E22_soft_knee', base_tag='thr70_w0', knee=70.0, top_pn=top0))
    for t in tags('E26_claw_power'):
        if not t.endswith('_pow1'):
            out.append(dict(folder='E26_claw_power', tag=t, base_folder='E26_claw_power', base_tag=t.split('_')[0] + '_pow1',
                knee=_knee_of(t), top_pn=top0))
    for t in tags('E41_claw_subgain'):
        if not t.endswith('_g0'):
            out.append(dict(folder='E41_claw_subgain', tag=t, base_folder='E41_claw_subgain', base_tag=t.rsplit('_g', 1)[0] + '_g0',
                knee=_knee_of(t), top_pn=top0))
    for t in tags('E32_pn_stage'):
        if t != 'default':
            out.append(dict(folder='E32_pn_stage', tag=t, base_folder='E32_pn_stage', base_tag='default', knee=70.0,
                top_pn=tops.get(t, np.nan)))
    e67 = set(tags('E67_apl_effect_fits'))
    for t in e67:
        if t.startswith('thr'):
            b = t.split('_', 1)[1]
            out.append(dict(folder='E67_apl_effect_fits', tag=t, base_folder='E67_apl_effect_fits', base_tag=b if b in e67 else None,
                knee=_knee_of(t), top_pn=top0))
    e72 = set(tags('E72_wiring_population'))
    for t in e72:
        if t.startswith('thr'):
            b = t.split('_', 1)[1].replace('aplcal', 'aplboth')
            out.append(dict(folder='E72_wiring_population', tag=t, base_folder='E72_wiring_population',
                base_tag=b if b in e72 else None, knee=_knee_of(t), top_pn=top0))
    for t in tags(EXPERIMENT):
        if t.startswith('thr') and not any(t.startswith(s + '_') for s in STARTS):
            out.append(dict(folder=EXPERIMENT, tag=t, base_folder=C2[0], base_tag=C2[1], knee=_knee_of(t), top_pn=top0))
    for r in out:
        r['knee_over_top'] = r['knee'] / r['top_pn'] if np.isfinite(r['knee']) and np.isfinite(r['top_pn']) else np.nan
    return out


def pair_stats(on: pd.Series, off: pd.Series, ro: str) -> Dict[str, float]:
    """One model's per-pair APL effect statistics under readout `ro` ('raw' or 'cal' distances given): the loss with
    its parts and ρ (E72), the group errors and the projection on the common error (E75), and per kind of pair
    (KINDS) the mean distances with APL on and off, the APL effect and the pattern error."""
    import E72_wiring_population as E72
    import E75_glomerulus_apl_profile as E75
    d = {'on': on, 'off': off}
    row = {f'{ro}_{k}': v for k, v in E72.effect_stats(d, ro).items()}
    row[f'{ro}_n_pairs'] = int(H.apl_effect(on, off).notna().sum())
    err = H.apl_effect_pair_parts(on, off, E72.data_for(ro))['error']
    try:
        row.update({f'{ro}_{k}': v for k, v in E75.error_vs_common(err, ro).items()})
    except Exception as e:
        print(f'error_vs_common ({ro}): {e!r}'[:200], flush=True)
    for kind, prs in KINDS.items():
        row[f'{ro}_{kind}_on'], row[f'{ro}_{kind}_off'] = kind_mean(on, prs), kind_mean(off, prs)
        row[f'{ro}_{kind}_effect'] = row[f'{ro}_{kind}_on'] - row[f'{ro}_{kind}_off']
        row[f'{ro}_{kind}_error'] = kind_mean(err, prs)
    return row


def _variant_stats(folder: str, tag: str) -> Optional[dict]:
    import E72_wiring_population as E72
    p = H.shared_dir() / 'results' / folder / 'model_pair_distances_long.csv'
    d = pd.read_csv(p)
    d = d[d['tag'].astype(str) == tag].drop_duplicates(['condition', 'odor_a', 'odor_b'], keep='last')
    if not ({'on', 'acute_off'} <= set(d.condition)):
        return None
    on = d[d.condition == 'on'].set_index(['odor_a', 'odor_b'])['distance']
    off = d[d.condition == 'acute_off'].set_index(['odor_a', 'odor_b'])['distance']
    row = pair_stats(on, off, 'raw')
    o = E72.load_fit(folder, tag)
    row['has_counts'] = o is not None
    if o is not None:
        dc = E72.distances(o, 'cal')
        row.update(pair_stats(dc['on'], dc['off'], 'cal'))
        for c in ('on', 'off'):
            fr = H.response_fractions(o[c][0]).reindex(H.APL16_ODORS)
            row.update({f'frac_{c}_{od}': float(v) for od, v in fr.items()})
            row[f'rate_{c}'] = float(o[c][0].to_numpy(dtype=float).mean())
            row[f'silent_odors_{c}'] = ','.join(od for od, v in fr.items() if v == 0)
    return row


ARCHIVE_KEYS = ('raw_loss', 'raw_pattern', 'raw_mean', 'raw_rho', 'raw_err_ester', 'raw_err_alcohol', 'raw_proj_common',
    'raw_six_effect', 'raw_six_on', 'raw_six_off', 'raw_nearest_effect', 'cal_loss', 'cal_pattern', 'cal_rho', 'cal_err_ester',
    'cal_err_alcohol', 'cal_proj_common', 'cal_six_effect')


def archive_knee_variants(out: Path) -> pd.DataFrame:
    """The archive's knee variants (`archive_variant_list`) on E77's kinds of pairs (ms's, the four nearest, the block's
    six, the alcohol group) and on the six pairs, raw and under the calibrated readout where the spike counts are
    archived, each with its own base's values and the change from it (d_<stat>), the knee and the knee over the base's
    largest PN rate. Writes archive_knee_variants.csv."""
    cache: Dict[Tuple[str, str], Optional[dict]] = {}
    rows = []
    for v in archive_variant_list():
        key = (v['folder'], v['tag'])
        if key not in cache:
            cache[key] = _variant_stats(*key)
        st = cache[key]
        if st is None:
            continue
        row = dict(v, **st)
        if v['base_tag']:
            bk = (v['base_folder'], v['base_tag'])
            if bk not in cache:
                cache[bk] = _variant_stats(*bk)
            b = cache[bk]
            if b is not None:
                for k in ARCHIVE_KEYS:
                    if k in st and k in b:
                        row[f'base_{k}'] = b[k]
                        row[f'd_{k}'] = st[k] - b[k]
                row['base_silent_odors_on'] = b.get('silent_odors_on')
        rows.append(row)
        print(f"{v['folder']}:{v['tag']} (knee {v['knee']:g} Hz, {v['knee_over_top']:.2f} of the largest PN rate): six pairs "
            f"{st['raw_six_on']:.3f} / {st['raw_six_off']:.3f} ({st['raw_six_effect']:+.3f}); ester / alcohol error "
            f"{st.get('raw_err_ester', np.nan):+.3f} / {st.get('raw_err_alcohol', np.nan):+.3f}; pattern {st['raw_pattern']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'archive_knee_variants.csv', index=False)
    return df


def noise_six_pairs(out: Path) -> pd.DataFrame:
    """The six pairs (and E77's other kinds) through every saved trial-noise table (results/*/partB and E69's
    trial_noise pairs_noise_long.csv: E72 part E's wirings, E75 part A's, E73's, ...): per model, level and transform,
    the kind's mean distance with APL on and off and its APL effect, through the noise ('raw', the pipeline's
    three-trial-mean distances, like for like with the flies' raw), with the model's own trial noise removed
    ('corrected') and deterministic ('det'), beside the flies' raw and corrected values; plus the mean over the models of
    each table ('population'). Writes noise_six_pairs.csv."""
    res = H.shared_dir() / 'results'
    paths = sorted(res.glob('E*/partB/pairs_noise_long.csv')) + sorted(res.glob('E*/trial_noise/pairs_noise_long.csv'))
    rows = []
    for p in paths:
        d = pd.read_csv(p)
        if not {'model', 'level', 'condition', 'odor_a', 'odor_b', 'raw', 'corrected'} <= set(d.columns):
            continue
        tr = d['transform'] if 'transform' in d.columns else pd.Series('graded', index=d.index)
        d = d.assign(transform=tr)
        cols = [c for c in ('raw', 'corrected', 'det', 'data_raw', 'data_corrected') if c in d.columns]
        groups = list(d.groupby(['model', 'level', 'transform'])) + [((f'population of {d.model.nunique()}', lv, t), g)
            for (lv, t), g in d.groupby(['level', 'transform']) if d.model.nunique() > 1]
        for (model, level, transform), g in groups:
            g = g.groupby(['condition', 'odor_a', 'odor_b'])[cols].mean()
            row = dict(folder=str(p.parent.parent.name), table=str(p.relative_to(res)), model=model, level=level, transform=transform)
            for kind, prs in KINDS.items():
                for col in cols:
                    vals = {}
                    for cond in ('on', 'acute_off'):
                        if cond in g.index.get_level_values(0):
                            vals[cond] = kind_mean(g.loc[cond][col], prs)
                    if len(vals) == 2:
                        row[f'{kind}_{col}_on'], row[f'{kind}_{col}_off'] = vals['on'], vals['acute_off']
                        row[f'{kind}_{col}_effect'] = vals['on'] - vals['acute_off']
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'noise_six_pairs.csv', index=False)
    print(f'noise_six_pairs.csv: {len(df)} rows from {len(paths)} tables', flush=True)
    return df


def who_selects(out: Path, models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Who selects the responders, from one run of each model with the bouton rates and APL's activity saved (the odor
    window, 0-0.5 s): C2 and the part-A fits (fits.csv rows with a knee) that exist. Each claw's input is its PN->claw
    weight times its bouton's rate (after APL's bouton inhibition) at every time step. Per odor and condition: the share
    of claws whose input exceeds the knee at some time in the window and the share of claw-time above it (100 and 130 Hz
    for C2, the fit's own knee otherwise); for the responders, the number of claws carrying half of the KC's transferred
    input (the knee's transfer applied at every time step, then the window mean; median over responders) and the KC's
    threshold over that summed input (median); APL's inhibition at the claw (wAPLKC_scale x the claw's unscaled weight x
    APL's window-mean activity, as olfsysm subtracts it from the claw's drive), over all claws and over the responders'
    claws, and over the odor's largest bouton rate. This follows the per-claw transfer, not the KC's integration, so the
    threshold ratio is approximate. Writes who_selects.csv."""
    import gc
    import E43_model_trial_noise as E43
    import E10_joint_fit as J
    from al_analysis.mb_model import make_fixed_param_kc_simulator, get_time_index, get_sim_var
    res = H.results_dir(EXPERIMENT)
    specs = [('C2', C2[0], C2[1])]
    fp = res / 'fits.csv'
    if fp.exists():
        f = pd.read_csv(fp)
        specs += [(str(t), EXPERIMENT, str(t)) for t in f.tag if str(t).startswith('thr')]
    if models is not None:
        specs = [s for s in specs if s[0] in models]
    path = out / 'who_selects.csv'
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['model'])
    rows = [r for r in old.to_dict('records')]
    done = set(old.model.astype(str)) if len(old) else set()
    for name, exp, tag in specs:
        if name in done:
            continue
        orn = H.apl16_orn_deltas()
        fm = E43.fit_model(E43.model_from(f'{exp}:{tag}'), orn, H.outputs_dir(EXPERIMENT) / 'C' / 'bases')
        kws = dict(fm['kws'])
        claw_thr = fm['info'].get('claw_thr')
        knees = [claw_thr] if claw_thr else [100.0, 130.0]
        g = float(kws.get('claw_thr_subgain', 0.0) or 0.0)
        w = float(kws.get('claw_thr_width', 0.0) or 0.0)
        _, _, W, _ = J.bouton_base(orn, H.outputs_dir(EXPERIMENT) / 'C' / 'bases', claw_thr, **({} if not fm['info'].get('wiring') else
            dict(wiring=fm['info']['wiring'], wiring_seed=int(fm['info']['wiring_seed']))))
        store: dict = {}
        with E43.captured_olfsysm_vars(store):
            sim, _ = make_fixed_param_kc_simulator(orn, initial_params=fm['p_on'], **kws)
        mp, rv = store['mp'], store['rv']
        # (APL's activity time course; the flag is read at every simulation)
        mp.kc.save_inh_sims = True
        t = np.asarray(get_time_index(mp, rv))
        win = (t >= 0.0) & (t <= 0.5)
        Wm = W.to_numpy(dtype=float)
        kc_of_claw = np.asarray(W.index.get_level_values(H.KC_ID_LEVEL))
        claws_of = pd.Series(np.arange(len(kc_of_claw))).groupby(kc_of_claw).apply(np.asarray).to_dict()
        top = largest_bouton_rates('C2')
        for cond, p in (('on', fm['p_on']), ('off', dict(fm['p_on'], wAPLKC_scale=0.0, wAPLPN_scale=0.0))):
            r, s = sim(p)
            kc_ids = np.asarray(r.index.get_level_values(H.KC_ID_LEVEL))
            btn = np.asarray(get_sim_var(rv.pn, 'bouton_sims'), dtype=float)        # odors x boutons x time
            if btn.shape[1] != Wm.shape[1]:
                raise RuntimeError(f'{name}: {btn.shape[1]} boutons simulated, {Wm.shape[1]} in the wiring')
            inh = np.asarray(get_sim_var(rv.kc, 'inh_sims'), dtype=float) if cond == 'on' else None
            thr = np.asarray(rv.kc.thr, dtype=float).ravel()
            resp = r.to_numpy(dtype=float)
            for i, col in enumerate(r.columns):
                od = H.odor_abbrev(col)
                bt = btn[i][:, win]                                 # boutons x time (the odor window)
                xt = Wm @ bt                                        # claws x time: each claw's input
                b = bt.mean(axis=1)
                row = dict(model=name, experiment=exp, tag=tag, cond=cond, odor=od, claw_thr=claw_thr, subgain=g, width=w,
                    top_bouton_rate=float(b.max()), top_bouton_rate_peak=float(bt.max()), n_claws=int(xt.shape[0]))
                for T in knees:
                    row[f'share_claws_above_{T:g}'] = float((xt.max(axis=1) > T).mean())
                    row[f'time_share_above_{T:g}'] = float((xt > T).mean())
                T0 = knees[0]
                if claw_thr and w > 0:
                    ft = w * np.logaddexp(0.0, (xt - T0) / w) + g * np.clip(xt, 0, T0)
                elif claw_thr:
                    ft = np.maximum(xt - T0, 0) + g * np.clip(xt, 0, T0)
                else:
                    ft = np.maximum(xt, 0)
                fx = ft.mean(axis=1)                                # each claw's transferred input, window mean
                del xt, ft
                rsp = resp[:, i] > 0.5
                resp_claws = np.zeros(len(fx), dtype=bool)
                if rsp.any() and len(thr) == len(kc_ids):
                    n_half, ratio = [], []
                    for k in np.flatnonzero(rsp):
                        idx = claws_of[kc_ids[k]]
                        resp_claws[idx] = True
                        v = np.sort(fx[idx])[::-1]
                        tot = v.sum()
                        n_half.append(int(np.searchsorted(np.cumsum(v), 0.5 * tot) + 1) if tot > 0 else 0)
                        ratio.append(thr[k] / tot if tot > 0 else np.inf)
                    row.update(n_claws_half_drive_median=float(np.median(n_half)), thr_over_drive_median=float(np.median(ratio)),
                        n_responders=int(rsp.sum()))
                if inh is not None and inh.size:
                    a = float(np.asarray(inh[i])[..., win].mean())
                    row['apl_activity'] = a
                    # the claw's APL->KC inhibition (olfsysm: wAPLKC_scale x the unscaled per-claw weight x APL's activity),
                    # window mean, over all claws and over the responders' claws, and against the odor's largest bouton rate
                    wk = np.asarray(fm['kws']['_wAPLKC'], dtype=float).ravel() * float(p['wAPLKC_scale'])
                    if len(wk) == len(fx):
                        row['apl_at_claw_mean'] = float(np.mean(wk * a))
                        row['apl_at_responder_claw_mean'] = float(np.mean(wk[resp_claws] * a)) if resp_claws.any() else np.nan
                        row['apl_at_claw_over_top'] = row['apl_at_claw_mean'] / max(float(b.max()), 1e-9)
                        row['apl_at_responder_claw_over_top'] = row['apl_at_responder_claw_mean'] / max(float(b.max()), 1e-9)
                rows.append(row)
        del sim, store
        gc.collect()
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f'who_selects: {name} done', flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# part C
# the brackets of E77's runner, widened (a 130 Hz knee takes the somatic threshold to about a tenth)
THR_BRACKET, APL_BRACKET = (0.01, 4.0), (1 / 32, 32.0)
COND_FILE = {'on': 'on', 'off': 'acute_off'}
_CACHE: Dict[object, object] = {}


# the two part C runs: 'held' (the first run, bd78d6e: the breadth from the threshold-held APL-off run, KC-KC held at the
# start's scale; points/<start>/, tags <start>_<point>), kept for the record, and 'prepass' (the rerun planned 21:39: the
# breadth from E10's transfer pre-pass, the KC-KC scale tied to the somatic threshold by default; points_prepass/<start>/,
# tags <start>_pp_<point>[_kf])
BREADTHS = ('prepass', 'held')
KCKC_MODES = ('tied', 'fixed')
# the pre-pass's threshold-factor bracket, widened from E10's (0.02, 50) for the odor-relative knee
PREPASS_BRACKET = (1e-3, 1e3)


def points_dir(start: str, breadth: str = 'prepass') -> Path:
    d = H.results_dir(EXPERIMENT) / ('points' if breadth == 'held' else f'points_{breadth}') / start
    d.mkdir(parents=True, exist_ok=True)
    return d


def point_key(pt: dict, breadth: str = 'prepass') -> str:
    """A point's file name and tag part: its id, with _kf for a KC-KC-fixed point of the pre-pass run (and _kt for a
    KC-KC-tied point of the held run)."""
    k = pt.get('kckc')
    if pt['point_id'] in ('identity', 'identity_held') or k is None:
        return pt['point_id']
    if breadth == 'prepass':
        return pt['point_id'] + ('_kf' if k == 'fixed' else '')
    return pt['point_id'] + ('_kt' if k == 'tied' else '')


def point_tag(start: str, pt: dict, breadth: str = 'prepass') -> str:
    """A point's tag: <start>_pp_<point key> in the pre-pass run, <start>_<point key> in the held run; the start itself
    ('identity') is <start>_identity in both."""
    if pt['point_id'] == 'identity':
        return f'{start}_identity'
    return f'{start}_' + ('pp_' if breadth == 'prepass' else '') + point_key(pt, breadth)


def c_outputs_dir() -> Path:
    d = H.outputs_dir(EXPERIMENT) / 'C'
    d.mkdir(parents=True, exist_ok=True)
    return d


def largest_bouton_rates(start: str = 'C2') -> pd.Series:
    """Each odor's largest bouton rate in the start's model (the PN rate of the most driven glomerulus in the wiring, mean
    over the odor window, Hz), by odor abbreviation."""
    if 'pn_rates' not in _CACHE:
        import E10_joint_fit as J
        import E32_pn_stage as E32
        from E12_breadth_apl import e3_bouton_params
        p = H.results_dir(EXPERIMENT) / 'pn_rates.csv'
        if p.exists():
            r = pd.read_csv(p, index_col=0, float_precision='round_trip')
        else:
            # (the PN rates depend on the ORN input and the antennal lobe only, not on the KCs' parameters, the wiring's
            # claw assignment or APL; every start has the same glomeruli)
            r = E32.glomerulus_pn_rates(H.apl16_orn_deltas(), J.base_kws(None), e3_bouton_params())
            r.index.name = 'glomerulus'
            r.to_csv(p)
        _CACHE['pn_rates'] = r
    return _CACHE['pn_rates'].max(axis=0)


def largest_bouton_rates_for(orn: pd.DataFrame) -> pd.Series:
    """`largest_bouton_rates` for another ORN input (E49's extended input with the low-concentration odors), by E49's
    odor labels (abbreviations for the APL16 odors, Yang's labels for the extra ones): the PN rates as E32's
    `glomerulus_pn_rates` computes them (0-0.5 s mean), the largest over the wiring's glomeruli."""
    import E10_joint_fit as J
    import E49_low_conc_kc as E49
    from al_analysis.mb_model import fit_mb_model, get_time_index, get_sim_var
    from E12_breadth_apl import e3_bouton_params
    key = ('pn_rates', tuple(map(str, orn.columns)))
    if key not in _CACHE:
        p0 = e3_bouton_params()
        _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=p0['fixed_thr'], wAPLKC=p0['wAPLKC_scale'], wKCAPL=p0['wKCAPL_scale'],
            wAPLPN=p0['wAPLPN_scale'], wPNAPL=p0['wPNAPL_scale'], **J.base_kws(None), return_olfsysm_vars=True)
        mp, rv = pd0['mp'], pd0['rv']
        t = np.asarray(get_time_index(mp, rv))
        win = (t >= 0.0) & (t <= 0.5)
        pn = get_sim_var(rv.pn, 'pn_sims')
        gloms = list(dict.fromkeys(wPNKC.columns.get_level_values('glomerulus')))
        labels = E49.odor_labels(orn)
        _CACHE[key] = pd.DataFrame({lab: np.asarray(pn[i])[:, win].mean(axis=1) for i, lab in enumerate(labels)}, index=gloms)
    return _CACHE[key].max(axis=0)


def odor_relative_inhibition(W: pd.DataFrame, orn: pd.DataFrame, top: pd.Series, level: float) -> pd.DataFrame:
    """The odor-relative knee as olfsysm's per-odor external inhibition at the boutons: a bouton x odor matrix (the
    wiring's bouton columns, the ORN input's odor columns) whose every row is `level` times the odor's largest bouton
    rate `top` (`largest_bouton_rates`)."""
    try:
        import E49_low_conc_kc as E49
        labels = E49.odor_labels(orn)
    except Exception:
        labels = [H.odor_abbrev(c) for c in orn.columns]
    vals = np.array([float(level) * float(top[lab] if lab in top.index else top[H.odor_abbrev(c)]) for c, lab in zip(orn.columns, labels)])
    return pd.DataFrame(np.tile(vals, (W.shape[1], 1)), index=W.columns, columns=orn.columns)


def knee_point(T: float, g: float, w: Optional[float] = None) -> dict:
    t = dict(claw_thr=float(T))
    if g:
        t['claw_thr_subgain'] = float(g)
    if w:
        t['claw_thr_width'] = float(w)
    pid = f'knee{T:g}_g{g:g}' + (f'_w{w:g}' if w else '')
    return dict(point_id=pid, family='knee' if not w else 'knee_soft', claw_thr=float(T), claw_thr_subgain=float(g),
        claw_thr_width=float(w) if w else 0.0, rel=np.nan, transfer=t)


def _usable(d: pd.DataFrame) -> pd.DataFrame:
    """Rows of a sweep table whose two response rates were held (neither bisection at a bound) and that are not controls."""
    if not len(d):
        return d
    held = d['rates_held'].astype(str).str.lower().isin(['true', '1', '1.0']) if 'rates_held' in d else pd.Series(True, index=d.index)
    ctrl = d['control'].astype(str).str.lower().isin(['true', '1', '1.0']) if 'control' in d else pd.Series(False, index=d.index)
    return d[held & ~ctrl]


def best_knee_points(sweep_df: Optional[pd.DataFrame], n: int = 2, kckc: str = 'tied') -> List[dict]:
    """The n best knee points on C2 (hard knee, with or without a sub-threshold gain) by the readout loss, among those
    whose rates were held, inside the bands where n of them are, with the run's KC-KC mode."""
    if sweep_df is None or not len(sweep_df):
        return []
    k = _usable(sweep_df[(sweep_df.family == 'knee') & (sweep_df.start == 'C2')])
    if 'kckc' in k:
        k = k[k.kckc.fillna(kckc) == kckc]
    inside = k[k.realism_flag_noise.fillna('') == '']
    k = inside if len(inside) >= n else k
    return [knee_point(float(r.claw_thr), float(r.claw_thr_subgain)) for _, r in k.sort_values('cal_loss').head(n).iterrows()]


def point_specs(sweep_df: Optional[pd.DataFrame] = None, breadth: str = 'prepass', kckc: str = 'tied') -> List[dict]:
    """Part C's points: the start itself ('identity'; for the held run also 'identity_held', the start with only the held
    breadth), C1 (knee x sub-threshold gain, the 70 Hz link points; the softplus width at the two best of them once
    `sweep_df` has them) and C2 (the odor-relative knee at REL_LEVELS), each with the run's KC-KC mode `kckc`; for the
    pre-pass run with the KC-KC scale tied, also the two best knee points with it fixed ('control', not gated). Each a
    dict with point_id, family ('identity', 'identity_held', 'knee', 'knee_soft', 'rel'), the settings, 'kckc', 'control'
    and 'transfer' (the claw kwargs; the odor-relative knee's inhibition matrix is built by KneeModel)."""
    ids = ('identity', 'identity_held') if breadth == 'held' else ('identity',)
    pts = [dict(point_id=i, family=i, claw_thr=np.nan, claw_thr_subgain=np.nan, claw_thr_width=np.nan, rel=np.nan, transfer={},
        kckc=None, control=False) for i in ids]
    rest = [knee_point(T, g) for T in KNEES for g in SUBGAINS] + [knee_point(LINK_KNEE, g) for g in LINK_SUBGAINS]
    rest += [dict(point_id=f'rel{lv:g}', family='rel', claw_thr=np.nan, claw_thr_subgain=np.nan, claw_thr_width=np.nan,
        rel=float(lv), transfer={}) for lv in REL_LEVELS]
    best = best_knee_points(sweep_df, 2, kckc)
    rest += [knee_point(b['claw_thr'], b['claw_thr_subgain'], SOFT_WIDTH) for b in best]
    for p in rest:
        p.update(kckc=kckc, control=False)
    pts += rest
    if breadth == 'prepass' and kckc == 'tied':
        pts += [dict(b, kckc='fixed', control=True) for b in best]
    return pts


class KneeModel:
    """A start (STARTS) at its fitted parameters on E77's fixed-scale runner (E77_input_screen: `build_c2_simulator`, C2's
    family with the start's parameters, wiring and firing ceiling; `hold_rates`), checked against the start as
    E10_joint_fit.py built it (E43_model_trial_noise.fit_model) and against its saved spike counts, with a transfer set
    on top (`hold`): the somatic threshold factor bisected to the start's APL-off response count, the breadth weights
    recomputed from that threshold-held APL-off run (`breadth_from='held'`, as the entry specifies; KC-KC inhibition
    included), then the APL->KC scale (KC->APL and bouton->APL tied to it, as the fits tie them; APL->bouton unchanged)
    bisected to the start's APL-on response count (E69's `bisect_factor`). The start itself is the 'identity' point; the
    'identity_held' point is the start with only the held breadth (no transfer), the control for that choice."""

    def __init__(self, start: str, out_dir: Path):
        import E10_joint_fit as J
        import E43_model_trial_noise as E43
        import E77_input_screen as S
        exp, tag, wiring = STARTS[start]
        self.start, self.exp, self.tag, self.wiring = start, exp, tag, dict(wiring)
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.orn = H.apl16_orn_deltas()
        fm = E43.fit_model(E43.model_from(f'{exp}:{tag}'), self.orn, out_dir)
        if fm['info'].get('claw_thr') is not None or fm['kc_thr_factor'] is not None or fm['info'].get('orn_input'):
            raise NotImplementedError(f'{start}: starts with a per-claw threshold, an input-scaled threshold or another '
                'ORN input are not used here')
        self.fm, self.info = fm, fm['info']
        self.p, self.kws, self.k = dict(fm['p_on']), dict(fm['kws']), float(fm['info']['k'])
        if self.k != S.BREADTH_K or abs(float(self.kws.get('kc_kc_tau', J.TAU)) - J.TAU) > 0:
            raise NotImplementedError(f"{start}: breadth exponent {self.k} / KC-KC tau {self.kws.get('kc_kc_tau')} differ from C2's")
        # (the firing ceiling of E69's fits: fixed model kwargs, passed to the runner's simulator with the transfer)
        self.spiking = {k: self.kws[k] for k in ('kc_refractory_period', 'kc_adapt_inc', 'kc_adapt_tau') if k in self.kws}
        wk = dict(self.wiring)
        if 'wiring_file' in wk:
            wk = dict(wiring_matrix=pd.read_parquet(H.shared_dir() / wk['wiring_file']), wiring_tag=wk['wiring_tag'])
        self.initial, self.wnorm, self.W, self.breadth0 = J.bouton_base(self.orn, out_dir, None, **wk)
        w0 = J.breadth_weights(self.wnorm['wAPLKC'], self.breadth0, self.k)
        if not np.array_equal(np.asarray(w0, dtype=float), np.asarray(self.kws['_wAPLKC'], dtype=float)):
            raise RuntimeError(f"{start}: the breadth weights rebuilt here differ from fit_model's")
        self.wPNKC = self.W if self.wiring else None
        # (the plain base's kwargs and wiring, for the pre-pass; its wiring names the pre-pass caches)
        self.plain_kws = dict(J.base_kws(None), **({'_wPNKC': self.W} if self.wiring else {}))
        self.wiring_name = ('real' if not wiring else (f"{wiring['wiring']}-seed{int(wiring['wiring_seed'])}" if 'wiring' in wiring
            else wiring['wiring_tag']))
        self._base_off = None
        sim, info = S.build_c2_simulator(self.spiking or None, wPNKC=self.wPNKC, breadth=self.breadth0, params=self.p)
        on, off = sim(self.p), sim(self.off(self.p))
        del sim, info
        self.check = {}
        for c, (r, s) in (('on', on), ('acute_off', off)):
            paths = fm['saved'][c]
            if not all(Path(x).exists() for x in paths):
                raise FileNotFoundError(f'{start}: saved outputs {paths} missing')
            ss = H.abbrev_columns(pd.read_parquet(paths[1]).astype(float))[list(H.APL16_ODORS)]
            sa = H.abbrev_columns(s.astype(float))[list(H.APL16_ODORS)]
            n = int((ss.to_numpy() != sa.to_numpy()).sum()) if ss.shape == sa.shape else -1
            self.check[f'{c}_spike_count_mismatch'] = n
            if n != 0 or not ss.index.equals(s.index):
                raise RuntimeError(f"{start}: E77's runner at the start's parameters differs from its saved {c} spike counts in "
                    f'{n} entries')
        self.ref = {'on': on, 'off': off}
        self.n_on = int((on[0].to_numpy(dtype=float) > 0.5).sum())
        self.n_off = int((off[0].to_numpy(dtype=float) > 0.5).sum())
        self.size = int(on[0].size)
        print(f'{start}: {exp}:{tag} on E77\'s runner reproduces its saved spike counts; {self.n_on} APL-on and {self.n_off} APL-off '
            f'responses of {self.size}', flush=True)

    @staticmethod
    def off(p: dict) -> dict:
        return dict(p, wAPLKC_scale=0.0, wAPLPN_scale=0.0)

    def transfer_kws(self, pt: dict, orn: Optional[pd.DataFrame] = None) -> dict:
        """The point's model kwargs on the ORN input `orn` (default the APL16 input): the claw kwargs, or the
        odor-relative knee's per-odor inhibition at the boutons (each odor's own largest bouton rate)."""
        orn = self.orn if orn is None else orn
        if pt['family'] == 'rel':
            top = largest_bouton_rates(self.start) if orn is self.orn else largest_bouton_rates_for(orn)
            return dict(ext_inh_bouton=odor_relative_inhibition(self.W, orn, top, pt['rel']))
        return dict(pt['transfer'])

    def breadth_path(self, tag: str) -> Path:
        return c_outputs_dir() / f'{tag}_breadth.parquet'

    def base_off_spk(self) -> pd.DataFrame:
        """The plain base's APL-off run at its tuned parameters (the run the start's breadth comes from), simulated and
        checked against the breadth."""
        if getattr(self, '_base_off', None) is None:
            import E10_joint_fit as J
            from al_analysis.mb_model import make_fixed_param_kc_simulator
            p0 = {k: float(self.initial[k]) for k in J.PARAM_KEYS}
            sim, _ = make_fixed_param_kc_simulator(self.orn, initial_params=p0, **self.plain_kws)
            _, s0 = sim(self.off(p0))
            del sim
            if not np.array_equal(s0.astype(float).sum(axis='columns').to_numpy(), self.breadth0.to_numpy(dtype=float)):
                raise RuntimeError(f"{self.start}: the plain base's APL-off run does not reproduce the breadth")
            self._base_off = s0
        return self._base_off

    def prepass_breadth(self, pt: dict, force: bool = False) -> Tuple[pd.Series, Optional[dict]]:
        """(breadth, pre-pass record) of point `pt` in the pre-pass run: E10_joint_fit.transfer_prepass on the plain bouton
        base of the start's wiring (E3's tuned parameters, APL off, no KC-KC inhibition, no firing ceiling) with the whole
        transfer (claw_thr, claw_thr_subgain, claw_thr_width, or the odor-relative ext_inh_bouton), its threshold re-tuned to
        the transfer-free run's APL-off response count (bracket PREPASS_BRACKET); cached by point id. Without a transfer
        the start's own breadth, unless `force` (the identity check: the pre-pass itself with an empty transfer)."""
        import E10_joint_fit as J
        transfer = self.transfer_kws(pt)
        if not transfer and not force:
            return self.breadth0, None
        info: dict = {}
        old = J.TRANSFER_PREPASS_BRACKET
        J.TRANSFER_PREPASS_BRACKET = PREPASS_BRACKET
        try:
            spk = J.transfer_prepass(self.orn, self.plain_kws, self.initial, self.base_off_spk(), transfer,
                self.out_dir / f'{self.wiring_name}_plain_base_acute_off_spike_counts.parquet', info, label=pt['point_id'])
        finally:
            J.TRANSFER_PREPASS_BRACKET = old
        b = pd.Series(spk.astype(float).sum(axis='columns').to_numpy(), index=spk.index.get_level_values(H.KC_ID_LEVEL))
        return b, info['claw_transfer_prepass']

    def point_kws(self, pt: dict, tag: str, orn: Optional[pd.DataFrame] = None) -> dict:
        """The simulator kwargs of a held point (`hold` must have run; for E49's extended input in `low_conc`): the start's
        kwargs with the point's breadth (saved by `hold` under `tag`; the start's own for 'identity') and its transfer on
        `orn`."""
        import E10_joint_fit as J
        if pt['point_id'] == 'identity':
            breadth = self.breadth0
        else:
            b = pd.read_parquet(self.breadth_path(tag))
            breadth = pd.Series(b['breadth'].to_numpy(dtype=float), index=b.index)
        return dict(self.kws, _wAPLKC=J.breadth_weights(self.wnorm['wAPLKC'], breadth, self.k), **self.transfer_kws(pt, orn))

    def hold(self, pt: dict, breadth: str = 'prepass', force_prepass: bool = False) -> Tuple[dict, Dict[str, tuple], dict]:
        """(absolute parameters, {'on': (responses, spike counts), 'off': ...}, record) of the start with point `pt`'s
        transfer, both response rates held; the point's breadth is saved under its tag (`breadth_path`).

        breadth 'prepass': the breadth from `prepass_breadth`, fixed before the bisections; on E77's `build_c2_simulator`
        (the start's parameters, wiring and firing ceiling) the somatic threshold is bisected with APL off to the start's
        APL-off response count, with the KC-KC scale multiplied by the same factor when pt['kckc'] is 'tied' (it is driven
        by the other KCs' spikes, so it does not shrink when a knee cuts the drive), then the APL->KC scale (KC->APL and
        bouton->APL tied to it; APL->bouton the start's) to the start's APL-on count at those values (E69's
        `bisect_factor`). breadth 'held': E77's `hold_rates` with breadth_from='held' and the KC-KC scale fixed (the first
        run)."""
        import E69_kc_firing_ceiling as E69
        import E77_input_screen as S
        import time
        tag = point_tag(self.start, pt, breadth)
        label = tag
        extra = dict(self.transfer_kws(pt), **self.spiking)
        kckc = pt.get('kckc') or 'fixed'
        if breadth == 'held':
            if kckc != 'fixed':
                raise NotImplementedError("breadth 'held' with the KC-KC scale tied")
            res = S.hold_rates(lambda: S.build_c2_simulator(extra or None, wPNKC=self.wPNKC, breadth=self.breadth0, params=self.p),
                target_off=self.n_off, target_on=self.n_on, params=self.p, breadth_from='held', label=label,
                thr_bracket=THR_BRACKET, apl_bracket=APL_BRACKET)
            b = res['breadth']
            b.rename('breadth').to_frame().to_parquet(self.breadth_path(tag))
            rec = dict(point_id=pt['point_id'], start=self.start, breadth_from='held', kckc='fixed', kckc_factor=1.0,
                thr_factor=res['thr_factor'], apl_factor=res['apl_factor'], thr_at_bound=res['at_bound_thr'],
                apl_at_bound=res['at_bound_apl'], n_responses_on=res['n_on'], n_responses_off=res['n_off'],
                target_n_on=res['target_on'], target_n_off=res['target_off'], n_sims=res['n_sims'], seconds=res['seconds'],
                bisections=res['history'], breadth_sha256=res['breadth_sha256'], runner_kws=res['kws'])
            p_on = dict(res['params'])
            outs = {'on': res['on'], 'off': res['off']}
        else:
            t0 = time.time()
            b, prepass = self.prepass_breadth(pt, force=force_prepass)
            b.rename('breadth').to_frame().to_parquet(self.breadth_path(tag))
            sim, info = S.build_c2_simulator(extra or None, wPNKC=self.wPNKC, breadth=b, params=self.p)
            p0 = dict(self.p)
            scaled = ('fixed_thr', 'wKCKC_scale') if kckc == 'tied' else ('fixed_thr',)
            f, off_out, h1, b1 = E69.bisect_factor(lambda x: sim(self.off(dict(p0, **{k: p0[k] * x for k in scaled}))), self.n_off,
                self.size, THR_BRACKET, S.THR_STEP, f'{label} threshold' + (' and KC-KC' if kckc == 'tied' else '') + ' (APL off)')
            p1 = dict(p0, **{k: p0[k] * f for k in scaled})
            x, on_out, h2, b2 = E69.bisect_factor(lambda y: sim(dict(p1, **{k: p0[k] * y for k in S.TIED_ON})), self.n_on, self.size,
                APL_BRACKET, S.APL_STEP, f'{label} APL scale (APL on)')
            p_on = dict(p1, **{k: p0[k] * x for k in S.TIED_ON})
            del sim, info
            outs = {'on': on_out, 'off': off_out}
            rec = dict(point_id=pt['point_id'], start=self.start, breadth_from='prepass', kckc=kckc,
                kckc_factor=float(f) if kckc == 'tied' else 1.0, thr_factor=float(f), apl_factor=float(x), thr_at_bound=bool(b1),
                apl_at_bound=bool(b2), n_responses_on=int((on_out[0].to_numpy(dtype=float) > 0.5).sum()),
                n_responses_off=int((off_out[0].to_numpy(dtype=float) > 0.5).sum()), target_n_on=self.n_on, target_n_off=self.n_off,
                n_sims=len(h1) + len(h2), seconds=time.time() - t0, bisections=h1 + h2, breadth_sha256=_sha(b.to_numpy(dtype=float)))
            if prepass is not None:
                rec['prepass'] = prepass
        rec['rates_held'] = not (rec['thr_at_bound'] or rec['apl_at_bound'])
        rec['breadth_corr_vs_start'] = float(np.corrcoef(b.reindex(self.breadth0.index).to_numpy(dtype=float),
            self.breadth0.to_numpy(dtype=float))[0, 1])
        if pt['family'] == 'rel':
            top = largest_bouton_rates(self.start)
            rec['rel_level_hz'] = {o: float(pt['rel'] * top[o]) for o in H.APL16_ODORS}
        return p_on, outs, rec

    def check_prepass_identity(self) -> dict:
        """The pre-pass run's identity check: with an empty transfer the pre-pass breadth (`transfer_prepass` itself) is the
        start's breadth, and the point held with it (KC-KC tied) equals the start's saved spike counts in both
        conditions. Returns the counts' mismatches and raises if any is not 0."""
        b, rec = self.prepass_breadth(dict(point_id='identity_check', family='identity', transfer={}), force=True)
        res = dict(start=self.start, breadth_equal=bool(np.array_equal(b.to_numpy(dtype=float), self.breadth0.to_numpy(dtype=float))
            and b.index.equals(self.breadth0.index)), prepass_thr_factor=rec['thr_factor'])
        pt = dict(point_id='identity_check', family='identity', claw_thr=np.nan, claw_thr_subgain=np.nan, claw_thr_width=np.nan,
            rel=np.nan, transfer={}, kckc='tied', control=False)
        p, o, r = self.hold(pt, 'prepass', force_prepass=True)
        for c, cond in (('on', 'on'), ('off', 'acute_off')):
            res[f'{cond}_mismatch'] = int((o[c][1].to_numpy(dtype=float) != self.ref[c][1].to_numpy(dtype=float)).sum())
        res.update(thr_factor=r['thr_factor'], apl_factor=r['apl_factor'], params_equal=all(p[k] == self.p[k] for k in self.p))
        res['ok'] = bool(res['breadth_equal'] and res['on_mismatch'] == 0 and res['acute_off_mismatch'] == 0)
        if not res['ok']:
            raise RuntimeError(f'{self.start}: the pre-pass identity check failed: {res}')
        return res


def _sha(x: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, dtype='<f8')).tobytes()).hexdigest()


def _canon(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.MultiIndex.from_tuples([tuple(sorted(i)) for i in s.index], names=['odor_a', 'odor_b'])
    return s.sort_index()


def six_typical_effect() -> pd.Series:
    """The KC data's per-pair APL effect against the six typical recordings with APL output blocked (TNT) (the eight
    without the two strongest, E75's STRONG_TNT) and all eleven control recordings (graded, cosine)."""
    if 'six' not in _CACHE:
        import E75_glomerulus_apl_profile as E75
        ctrl = H.yang_recording_rdms_values('TNT3c', 'graded')
        tnt = H.yang_recording_rdms_values('TNT', 'graded')
        keep = [r for r in tnt.index if (str(r[0]), int(r[1]), str(r[2])) not in E75.STRONG_TNT]
        if len(keep) != len(tnt) - 2:
            raise RuntimeError(f'expected to drop 2 of {len(tnt)} TNT recordings, kept {len(keep)}')
        _CACHE['six'] = ctrl.mean() - tnt.loc[keep].mean()
    return _CACHE['six']


KINDS = {'six': SIX, 'nearest': NEAREST, 'block': BLOCK, 'ms': MS_PAIRS,
    'alcohol': tuple((a, b) for i, a in enumerate(ALCOHOL) for b in ALCOHOL[i + 1:])}


def kind_mean(s: pd.Series, pairs: Sequence[Tuple[str, str]]) -> float:
    c = _canon(s)
    idx = [tuple(sorted(p)) for p in pairs]
    return float(c.reindex(pd.MultiIndex.from_tuples(idx)).mean())


def score_point(o: Dict[str, tuple], ref: Dict[str, tuple]) -> Tuple[dict, List[dict]]:
    """A point's statistics: the per-pair APL effect on raw counts, under the calibrated readout (K re-calibrated on its
    APL-on counts) and binarized; the loss with its parts and ρ on the per-pair APL effect against all eight recordings
    with APL output blocked (TNT) (E75's `score_point`: also the ester and alcohol group errors, the projection on the
    common error, the change from the start, the class split), the six typical ones and the effect in baseline-noise
    units; E77's kinds of pairs and the six pairs; the 16 odors' response fractions in both conditions with the response
    rate ρ against the KC data in Yang's z and in baseline-noise units; odors per KC; the largest spike counts; the
    realism bands under both rules. Returns (row, per-pair rows)."""
    import E72_wiring_population as E72
    import E75_glomerulus_apl_profile as E75
    import E76_data_side_target as E76
    import apl_effect_scores as S
    base_eff = {ro: H.apl_effect(*[E72.distances(ref, ro)[c] for c in ('on', 'off')]) for ro in ('raw', 'cal')}
    row, eff_rows = E75.score_point(o, base_eff)
    # (E75's per-pair APL effects as condition 'effect'; the model's distances follow as 'on' / 'acute_off')
    pairs = [dict(readout=r['readout'], condition='effect', odor_a=r['odor_a'], odor_b=r['odor_b'], distance=r['effect'])
        for r in eff_rows]
    s_on, s_off = o['on'][1].astype(float), o['off'][1].astype(float)
    dist = {ro: E72.distances(o, ro) for ro in ('raw', 'cal')}
    graded = E72.data_for('raw')
    six = six_typical_effect()
    nu = E76.data_effect('noise')
    K_nu = H.calibrate_readout(s_on, 'saturating', H.data_target('noise_apl_effect_cal_target'), stat='participation_ratio',
        responders_min=2.0)
    dist_nu = {'on': H.readout_pair_distances(s_on, 'saturating', K_nu), 'off': H.readout_pair_distances(s_off, 'saturating', K_nu)}
    for ro, d, data, name in (('raw', dist['raw'], six, 'six'), ('cal', dist['cal'], six, 'six'), ('raw', dist['raw'], nu, 'nu'),
            ('cal', dist_nu, nu, 'nu')):
        dec, _ = H.apl_effect_decomposition(d['on'], d['off'], data)
        row.update({f'{ro}_{name}_loss': dec['apl_effect_rmse'], f'{ro}_{name}_mean': dec['apl_effect_mean'],
            f'{ro}_{name}_pattern': dec['apl_effect_rmse_centered'], f'{ro}_{name}_rho': dec['apl_effect_spearman']})
        try:
            row.update({f'{ro}_{name}_{k}': v for k, v in E75.error_vs_common(H.apl_effect_pair_parts(d['on'], d['off'], data)['error'],
                ro).items() if k in ('err_ester', 'err_alcohol', 'proj_common')})
        except Exception as e:
            print(f'error_vs_common ({ro}, {name}): {e!r}'[:200], flush=True)
    row['cal_nu_K'] = float(K_nu)
    data_on, data_off = H.yang_mean_rdm('TNT3c'), H.yang_mean_rdm('TNT')
    for kind, prs in KINDS.items():
        for ro in ('raw', 'cal'):
            on_d, off_d = dist[ro]['on'], dist[ro]['off']
            row[f'{ro}_{kind}_on'] = kind_mean(on_d, prs)
            row[f'{ro}_{kind}_off'] = kind_mean(off_d, prs)
            row[f'{ro}_{kind}_effect'] = row[f'{ro}_{kind}_on'] - row[f'{ro}_{kind}_off']
            err = H.apl_effect_pair_parts(on_d, off_d, graded)['error']
            row[f'{ro}_{kind}_error'] = kind_mean(err, prs)
        row[f'data_{kind}_on'], row[f'data_{kind}_off'] = kind_mean(data_on, prs), kind_mean(data_off, prs)
        row[f'data_{kind}_effect'] = kind_mean(graded, prs)
    fr = {c: H.response_fractions(o[c][0]).reindex(H.APL16_ODORS) for c in ('on', 'off')}
    for c in ('on', 'off'):
        row.update({f'frac_{c}_{od}': float(v) for od, v in fr[c].items()})
        row[f'noise_rate_rho_{c}'] = H.spearman(fr[c], S.frac_data(c, 'noise').reindex(H.APL16_ODORS))
        opk = S.odors_per_kc(o[c][0])
        row[f'noise_opk_{c}_tv'] = float(0.5 * np.abs(opk - S.opk_data(c, 'noise')).sum())
        row[f'opk_{c}_tv'] = float(0.5 * np.abs(opk - S.opk_data(c, 'z')).sum())
        x = o[c][1].to_numpy(dtype=float)
        row[f'max_count_{c}'] = float(x.max())
        rv = x[x >= 2.0]
        row[f'p99_resp_count_{c}'] = float(np.percentile(rv, 99)) if len(rv) else np.nan
    row['realism_flag_noise'] = S.realism_flag(row['rate_on'], row['rate_off'], row['noise_rate_rho_on'], row['noise_opk_on_tv'],
        row['noise_opk_off_tv'], 'noise')
    row['realism_flag_z'] = S.realism_flag(row['rate_on'], row['rate_off'], row['rate_rho_on'], row['opk_on_tv'], row['opk_off_tv'], 'z')
    st = H.apl_effect_stats(dist['raw']['on'], dist['raw']['off'], H.yang_apl_effect())
    row.update({f'{H.PRIMARY_MODEL_RDM}__{k}': v for k, v in st.items()})
    row['silent_odors_on'] = ','.join(od for od, v in fr['on'].items() if v == 0)
    row['silent_odors_off'] = ','.join(od for od, v in fr['off'].items() if v == 0)
    for ro in ('raw', 'cal'):
        for c, cond in (('on', 'on'), ('off', 'acute_off')):
            pairs += [dict(readout=ro, condition=cond, odor_a=a, odor_b=b, distance=float(v)) for (a, b), v in dist[ro][c].items()]
    return row, pairs


def sweep(out: Path, start: str = 'C2', only: Optional[Sequence[str]] = None, breadth: str = 'prepass',
    kckc: str = 'tied') -> pd.DataFrame:
    """Runs (resumably) part C's points on `start` in run `breadth` with KC-KC mode `kckc` and scores each
    (`score_point`): one JSON row, a per-pair CSV and a record per point (points_dir(start, breadth)/<point key>.*), the
    spike counts / responses and the breadth under outputs/E78_claw_knee/C/<tag>_*.parquet, with the point's absolute
    parameters, transfer, bisections, breadth_from, kckc, kckc_factor, rates_held and the code / model provenance; then
    `aggregate`. `only`: point ids (the identity points always run); the points added once others are done (the softplus
    width and, in the pre-pass run, the KC-KC-fixed controls at the two best knee points) run when `only` is None."""
    import json
    import time
    if breadth not in BREADTHS or kckc not in KCKC_MODES:
        raise ValueError(f'{breadth=} {kckc=}')
    pdir = points_dir(start, breadth)

    def specs() -> List[dict]:
        return point_specs(aggregate(out, breadth), breadth, kckc)

    def todo_points() -> List[dict]:
        pts = specs()
        if only is not None:
            pts = [p for p in pts if (p['point_id'] in only and not p['control']) or p['family'] in ('identity', 'identity_held')]
        return [p for p in pts if not (pdir / f'{point_key(p, breadth)}.json').exists()
            and not (pdir / f'{point_key(p, breadth)}.failed.txt').exists()]

    if only is not None:
        known = {p['point_id'] for p in specs()} | {knee_point(T, g, SOFT_WIDTH)['point_id'] for T in KNEES + (LINK_KNEE,)
            for g in SUBGAINS}
        missing = set(only) - known
        if missing:
            raise ValueError(f'unknown points {sorted(missing)}')
    todo = todo_points()
    if not todo:
        print(f'{start} ({breadth}, KC-KC {kckc}): every point done', flush=True)
        return aggregate(out, breadth)
    model = KneeModel(start, H.outputs_dir(EXPERIMENT) / 'C' / 'bases')
    prov = H.model_provenance(orn_deltas=model.orn)
    import E69_kc_firing_ceiling as E69
    code = E69.code_state()
    ppath = pdir / f"provenance_{time.strftime('%Y%m%d-%H%M%S')}.json"
    ppath.write_text(json.dumps(dict(model=prov, start=dict(experiment=model.exp, tag=model.tag, wiring=model.wiring,
        info=model.info, check=model.check), breadth=breadth, kckc=kckc, prepass_bracket=PREPASS_BRACKET,
        script='E78_claw_knee.py'), indent=1, default=str) + '\n')
    while todo:
        for pt in todo:
            t0 = time.time()
            key, tag = point_key(pt, breadth), point_tag(start, pt, breadth)
            if pt['point_id'] == 'identity':
                p, o, rec = dict(model.p), model.ref, dict(point_id='identity', start=start, thr_factor=1.0, apl_factor=1.0,
                    thr_at_bound=False, apl_at_bound=False, n_responses_on=model.n_on, n_responses_off=model.n_off,
                    target_n_on=model.n_on, target_n_off=model.n_off, n_sims=0, bisections=[], breadth_from='start', kckc='',
                    kckc_factor=1.0, rates_held=True)
            else:
                try:
                    p, o, rec = model.hold(pt, breadth)
                except Exception as e:
                    print(f'{tag} FAILED: {e!r}'[:400], flush=True)
                    (pdir / f'{key}.failed.txt').write_text(repr(e) + '\n')
                    continue
            for c, (r, s) in o.items():
                r.to_parquet(c_outputs_dir() / f'{tag}_{COND_FILE[c]}_responses.parquet')
                s.to_parquet(c_outputs_dir() / f'{tag}_{COND_FILE[c]}_spike_counts.parquet')
            row, pairs = score_point(o, model.ref)
            meta = dict(tag=tag, start=start, start_experiment=model.exp, start_tag=model.tag, point_id=pt['point_id'], point_key=key,
                run=breadth, control=bool(pt.get('control')), family=pt['family'], claw_thr=pt['claw_thr'],
                claw_thr_subgain=pt['claw_thr_subgain'], claw_thr_width=pt['claw_thr_width'], rel=pt['rel'], k=model.k,
                kc_kc_tau=float(model.kws.get('kc_kc_tau', np.nan)), **{kk: float(v) for kk, v in p.items()},
                kc_kc_inh=float(p['wKCKC_scale']),
                **{kk: model.kws[kk] for kk in ('kc_refractory_period', 'kc_adapt_inc', 'kc_adapt_tau') if kk in model.kws},
                **{kk: v for kk, v in rec.items() if kk not in ('bisections', 'prepass', 'rel_level_hz', 'point_id', 'start',
                    'runner_kws')},
                **({f'prepass_{kk}': rec['prepass'][kk] for kk in ('thr_factor', 'fixed_thr', 'n_responses', 'target_n_responses')}
                    if 'prepass' in rec else {}),
                top_bouton_rate_max=float(largest_bouton_rates(start).max()) if pt['family'] in ('knee', 'knee_soft') else np.nan,
                knee_over_top=float(pt['claw_thr'] / largest_bouton_rates(start).max()) if pt['family'] in ('knee', 'knee_soft') else np.nan,
                **code, provenance_file=_rel(ppath), seconds_total=time.time() - t0)
            full = dict(meta, **row)
            (pdir / f'{key}_pairs.csv').write_text(pd.DataFrame(pairs).to_csv(index=False))
            (pdir / f'{key}_record.json').write_text(json.dumps(dict(rec, params=p), indent=1, default=str) + '\n')
            (pdir / f'{key}.json').write_text(json.dumps(full, indent=1, default=_json_default) + '\n')
            print(f"{tag}: thr x{rec['thr_factor']:.4f}" + (' (KC-KC tied)' if rec.get('kckc') == 'tied' else '') +
                f", APL x{rec['apl_factor']:.4f}{'' if rec['rates_held'] else ' (RATES NOT HELD: a bisection at its bound)'}; "
                f"readout loss {row['cal_loss']:.4f} (pattern {row['cal_pattern']:.4f}, mean {row['cal_mean']:+.3f}), raw "
                f"{row['raw_loss']:.4f} (mean {row['raw_mean']:+.3f}); rho on the per-pair effect {row['cal_rho']:.3f}; six pairs' raw "
                f"effect {row['raw_six_effect']:+.3f}; ester / alcohol error {row['cal_err_ester']:+.3f} / {row['cal_err_alcohol']:+.3f}; "
                f"response rate rho {row['rate_rho_on']:.2f}; silent on [{row['silent_odors_on']}] off [{row['silent_odors_off']}]; "
                f"flag: {row['realism_flag_noise'] or '-'}; {time.time() - t0:.0f} s", flush=True)
        aggregate(out, breadth)
        # (the softplus width and the controls at the two best knee points, once those are done)
        todo = todo_points()
    return aggregate(out, breadth)


def check_c(out: Path, start: str = 'C2') -> dict:
    """The pre-pass run's identity check on `start` (`KneeModel.check_prepass_identity`), written to
    check_prepass_identity.json (one entry per start)."""
    import json
    import E69_kc_firing_ceiling as E69
    model = KneeModel(start, H.outputs_dir(EXPERIMENT) / 'C' / 'bases')
    res = dict(model.check_prepass_identity(), start_check=model.check, **E69.code_state())
    path = out / 'check_prepass_identity.json'
    allr = json.loads(path.read_text()) if path.exists() else {}
    allr[start] = res
    path.write_text(json.dumps(allr, indent=1, default=_json_default) + '\n')
    print(f'pre-pass identity check on {start}: {res}', flush=True)
    return res


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(H.shared_dir()))
    except ValueError:
        return str(p)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


def _family(out: Path, breadth: str) -> Tuple[List[dict], List[pd.DataFrame], dict]:
    import json
    rows, pairs, params = [], [], {}
    root = out / ('points' if breadth == 'held' else f'points_{breadth}')
    for p in sorted(root.glob('*/*.json')):
        if p.name.endswith('_record.json') or p.name.startswith(('provenance_', 'check_')):
            continue
        r = json.loads(p.read_text())
        # (the held run's first points have no run / kckc / rates_held columns)
        r.setdefault('run', breadth)
        r.setdefault('point_key', r['point_id'])
        r.setdefault('control', False)
        if breadth == 'held':
            r.setdefault('kckc', '' if r['point_id'] == 'identity' else 'fixed')
            r.setdefault('kckc_factor', 1.0)
        r.setdefault('rates_held', not (bool(r.get('thr_at_bound')) or bool(r.get('apl_at_bound'))))
        rows.append(r)
        pp = p.with_name(p.stem + '_pairs.csv')
        if pp.exists():
            pairs.append(pd.read_csv(pp).assign(tag=r['tag'], start=r['start'], point_id=r['point_id'], run=breadth))
        rec = p.with_name(p.stem + '_record.json')
        if rec.exists():
            rr = json.loads(rec.read_text())
            params[r['tag']] = dict(rr['params'], claw_thr=r['claw_thr'], claw_thr_subgain=r['claw_thr_subgain'],
                claw_thr_width=r['claw_thr_width'], rel=r['rel'], start=r['start'], start_experiment=r['start_experiment'],
                start_tag=r['start_tag'], k=r['k'], run=breadth, breadth_from=r.get('breadth_from'), kckc=r.get('kckc'),
                kckc_factor=r.get('kckc_factor'), rates_held=r['rates_held'],
                **({'rel_level_hz': rr['rel_level_hz']} if 'rel_level_hz' in rr else {}),
                **({'prepass': rr['prepass']} if 'prepass' in rr else {}))
    return rows, pairs, params


def aggregate(out: Path, breadth: str = 'prepass') -> pd.DataFrame:
    """From the point files of both runs: the pre-pass run's sweep.csv, sweep_pairs.csv and sweep_params.json, the held
    run's sweep_held.csv, sweep_pairs_held.csv and sweep_params_held.json (one row per start and point; per-pair model
    distances raw and calibrated; absolute parameters and transfer per tag), and the harness-readable stats_variants.csv
    and model_pair_distances_long.csv of both (raw cosine distances; the identity points, which are other experiments'
    fits, left out; every other tag of model_pair_distances_long.csv, part A's and part D's fits, kept). Returns run
    `breadth`'s table."""
    import json
    tables, all_pairs = {}, []
    for b in BREADTHS:
        rows, pairs, params = _family(out, b)
        suffix = '' if b == 'prepass' else f'_{b}'
        df = pd.DataFrame(rows)
        tables[b] = df
        if len(df):
            df.to_csv(out / f'sweep{suffix}.csv', index=False)
            (out / f'sweep_params{suffix}.json').write_text(json.dumps(params, indent=1, default=_json_default) + '\n')
        if pairs:
            pl = pd.concat(pairs, ignore_index=True)
            pl.to_csv(out / f'sweep_pairs{suffix}.csv', index=False)
            all_pairs.append(pl)
    both = pd.concat([t for t in tables.values() if len(t)], ignore_index=True) if any(len(t) for t in tables.values()) else pd.DataFrame()
    if all_pairs:
        pl = pd.concat(all_pairs, ignore_index=True)
        raw = pl[(pl.readout == 'raw') & (pl.point_id != 'identity') & pl.condition.isin(['on', 'acute_off'])][
            ['tag', 'condition', 'odor_a', 'odor_b', 'distance']]
        mp = out / 'model_pair_distances_long.csv'
        mine = set(both.tag)
        other = pd.read_csv(mp) if mp.exists() else pd.DataFrame(columns=raw.columns)
        other = other[~other['tag'].astype(str).isin(mine)]
        pd.concat([other, raw], ignore_index=True).to_csv(mp, index=False)
    if len(both):
        sv = both[both.point_id != 'identity']
        sv = sv.assign(variant=sv.tag, spearman_rate_on_vs_data=sv.get('rate_rho_on'), spearman_rate_off_vs_data=sv.get('rate_rho_off'))
        sv.to_csv(out / 'stats_variants.csv', index=False)
    return tables[breadth]


def best_points(sweep_df: pd.DataFrame, n: int = N_BEST, start: str = 'C2') -> List[str]:
    """The n best points of the sweep on `start` by the readout loss among those whose rates were held and that are not
    controls, inside the bands (16:10 rule) where n of them are, else the best inside first and then the rest."""
    d = _usable(sweep_df[(sweep_df.start == start) & ~sweep_df.point_id.isin(['identity', 'identity_held'])])
    ok = d[d.realism_flag_noise.fillna('') == ''].sort_values('cal_loss')
    rest = d[~d.index.isin(ok.index)].sort_values('cal_loss')
    return list(pd.concat([ok, rest]).point_id.head(n))


LANES = (
    ('knee100_g0', 'knee100_g0.1', 'knee100_g0.25', 'knee100_g0.5', 'knee115_g0', 'knee115_g0.1', 'knee115_g0.25', 'knee115_g0.5',
     'knee70_g0'),
    ('knee130_g0', 'knee130_g0.1', 'knee130_g0.25', 'knee130_g0.5', 'knee70_g0.25', 'rel0.5', 'rel0.7', 'rel0.85'),
)


def c_launch(wait_units: Sequence[str] = ()) -> List[str]:
    """Part C's pre-pass run end to end as systemd-run --user units (nice 10, 16 threads each, python -u, two at a time;
    logs under outputs/E78_claw_knee/logs/partC_*_<time>.log), each waiting first for `wait_units` to be inactive:
    units a and b run the two lanes of points on C2 (LANES); unit c, once both are done, runs the softplus width and the
    KC-KC-fixed controls at the two best knee points, the five best and C2 at the low concentrations, the gate's first
    reading, then the five best on the firing ceiling's fit 1 while unit d runs them on P1, then (once d is done) the
    gate again, the held run's gate against its identity_held point, and the summaries of both runs. Returns the unit
    names."""
    import subprocess
    import time
    logs = H.outputs_dir(EXPERIMENT) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    py = str(Path(__file__).resolve().parents[2] / 'venv' / 'bin' / 'python')
    me = f'{py} -u E78_claw_knee.py'
    stamp = time.strftime('%H%M%S')
    names = {k: f'e78c2-{k}-{stamp}' for k in 'abcd'}
    log = lambda name: f'{logs}/partC_{name}_{stamp}.log'
    wait_a = ''.join(f'while systemctl --user is-active --quiet {u}; do sleep 60; done; ' for u in wait_units)
    gate1_done = logs / f'partC_gate1_{stamp}.done'

    def wait_for(*units):
        return ''.join(f'sleep 5; while systemctl --user is-active --quiet {u}; do sleep 30; done; ' for u in units)

    script = {
        'a': f"{wait_a}{me} --part C --only {' '.join(LANES[0])} > {log('a')} 2>&1",
        'b': f"{wait_a}{me} --part C --only {' '.join(LANES[1])} > {log('b')} 2>&1",
        'c': (f"{wait_a}{wait_for(names['a'], names['b'])}{me} --part C > {log('width_controls')} 2>&1; "
              f"{me} --part C-low --best > {log('low')} 2>&1; {me} --part gate > {log('gate1')} 2>&1; touch {gate1_done}; "
              f"{me} --part C --start fit1 --best > {log('fit1')} 2>&1; {wait_for(names['d'])}"
              f"{me} --part gate > {log('gate2')} 2>&1; {me} --part gate --breadth held --gate-ref identity_held > {log('gate_held')} 2>&1; "
              f"{me} --part summarize > {log('summarize')} 2>&1; {me} --part summarize --breadth held > {log('summarize_held')} 2>&1"),
        'd': f"{wait_a}until [ -f {gate1_done} ]; do sleep 30; done; {me} --part C --start P1 --best > {log('P1')} 2>&1",
    }
    for k in 'abcd':
        subprocess.run(['systemd-run', '--user', f'--unit={names[k]}', '--collect', '-p', 'Nice=10', f'--working-directory={H.HERE}',
            '--setenv=OMP_NUM_THREADS=16', '--setenv=OMP_THREAD_LIMIT=16', '--setenv=PYTHONUNBUFFERED=1', '/bin/bash', '-c',
            script[k]], check=True)
        print(f"{names[k]}: {script[k]}", flush=True)
    return list(names.values())


def low_conc(out: Path, point_ids: Sequence[str], start: str = 'C2', breadth: str = 'prepass') -> pd.DataFrame:
    """The points at Yang's low concentrations, deterministic (E49's extended input: ma at 10^-7 and 10^-8, 2h at 10^-7
    and 10^-5, ea, farn and the binary mixtures; odors are simulated independently, so the APL16 columns must equal the
    point's saved outputs), at each point's absolute parameters with its transfer (the odor-relative knee at each odor's
    own largest bouton rate) and its APL16 breadth, beside the start's own fractions and the flies' (E49's
    data_fractions). Writes low_conc/fractions.csv (tag x odor x condition), low_conc/points_low_conc.csv (one row per
    tag: low_<odor key>_frac_<on|off>) and low_conc/data_fractions.csv."""
    import json
    import E49_low_conc_kc as E49
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    ldir = out / 'low_conc'
    ldir.mkdir(parents=True, exist_ok=True)
    sw = aggregate(out, breadth)
    params = json.loads((out / ('sweep_params.json' if breadth == 'prepass' else f'sweep_params_{breadth}.json')).read_text())
    nc = sw[~sw['control'].astype(str).str.lower().isin(['true', '1', '1.0'])] if 'control' in sw else sw
    kck = nc.set_index('point_id')['kckc'].to_dict() if 'kckc' in nc else {}
    specs = {p['point_id']: p for p in point_specs(sw, breadth) if not p['control']}
    orn16 = H.apl16_orn_deltas()
    orn, info = E49.extended_orn(orn16)
    labels = E49.odor_labels(orn)
    model = KneeModel(start, H.outputs_dir(EXPERIMENT) / 'C' / 'bases')
    fpath = ldir / 'fractions.csv'
    old = pd.read_csv(fpath) if fpath.exists() else pd.DataFrame(columns=['tag'])
    rows = [r for r in old.to_dict('records')]
    done = set(old.tag) if len(old) else set()
    checks = {}
    for pid in ['identity'] + [p for p in point_ids if p != 'identity']:
        pt = dict(specs[pid], kckc=(kck.get(pid) or specs[pid].get('kckc')) if pid != 'identity' else None)
        tag = point_tag(start, pt, breadth)
        if tag in done:
            continue
        p = {k: float(v) for k, v in params[tag].items() if k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale',
            'wPNAPL_scale', 'wKCKC_scale')}
        kws = model.point_kws(pt, tag, orn)
        sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p, **kws)
        det = {'on': sim(p), 'acute_off': sim(model.off(p))}
        del sim
        chk = {}
        for c, (r, s) in det.items():
            saved = pd.read_parquet(c_outputs_dir() / f'{tag}_{c}_spike_counts.parquet').astype(float)
            s16 = H.abbrev_columns(s.astype(float).iloc[:, :16])[list(H.APL16_ODORS)]
            sv = H.abbrev_columns(saved)[list(H.APL16_ODORS)]
            chk[f'{c}_apl16_mismatch'] = int((s16.to_numpy() != sv.to_numpy()).sum())
        checks[tag] = chk
        if any(chk.values()):
            print(f'{tag}: the APL16 columns differ from the saved outputs: {chk}', flush=True)
        for c, (r, s) in det.items():
            fr = r.astype(float).mean(axis=0)
            fr.index = labels
            for lab in labels:
                rows.append(dict(tag=tag, start=start, point_id=pid, odor=lab, cond=c, frac_det=float(fr[lab]),
                    kind='apl16' if lab in H.APL16_ODORS else ('mixture' if lab in E49.MIXTURES else 'low_conc'),
                    substitution=E49.substitutions().get(lab, 'APL16 input')))
        pd.DataFrame(rows).to_csv(fpath, index=False)
        lo = {lab: rr for lab, rr in ((x['odor'], x) for x in rows if x['tag'] == tag and x['cond'] == 'on')}
        print(f"{tag}: low concentrations, APL on: " + ', '.join(f'{o} {lo[o]["frac_det"]:.4f}' for o in E49.ORN_SOURCE) + f'; {chk}',
            flush=True)
    fr = pd.DataFrame(rows)
    wide = []
    for tag, g in fr.groupby('tag'):
        w = dict(tag=tag)
        for _, x in g[g.kind != 'apl16'].iterrows():
            key = x.odor.replace(' @ ', '_m').replace(' + ', '_plus_').replace('-', '').replace(' ', '')
            w[f"low_{key}_frac_{'on' if x.cond == 'on' else 'off'}"] = float(x.frac_det)
        wide.append(w)
    pw = pd.DataFrame(wide)
    pw.to_csv(ldir / 'points_low_conc.csv', index=False)
    E49.data_fractions().to_csv(ldir / 'data_fractions.csv', index=False)
    (ldir / 'checks.json').write_text(json.dumps(dict(json.loads((ldir / 'checks.json').read_text()) if (ldir / 'checks.json').exists()
        else {}, **checks), indent=1) + '\n')
    return pw


# ---------------------------------------------------------------------------
# the gate
def gate_point(row: dict, ref: dict, ceiling: bool = False, loss_limit: Optional[float] = None) -> Dict[str, object]:
    """The entry's four conditions for one point (`row`, a sweep.csv row) against `ref` (C2's identity row, the start
    itself), inside the four bands (16:10 rule, `realism_flag_noise`), with both response rates held (`rates_held`; a
    point whose bisection ended at a bound fails every condition):
    (i) the ester group's error under the calibrated readout (`cal_err_ester`) smaller in magnitude by a third or more and
        the six pairs' APL effect on raw counts (`raw_six_effect`) raised by 0.03 or more;
    (ii) the readout loss (`cal_loss`) at most `loss_limit` (default `ref`'s plus 0.004, or 0.006 with `ceiling`; the gate
        passes 0.060 for C2's points, the start's own plus 0.006 for the firing ceiling's and plus 0.004 for the re-drawn
        wiring's);
    (iii) each of the 16 odors keeps half or more of `ref`'s response fraction in both conditions (`frac_<on|off>_<odor>`,
        odors `ref` does not recruit skipped) and the response rate ρ with APL on (`rate_rho_on`, against the KC data in
        Yang's z, C2's 0.60) is 0.50 or more;
    (iv) 2h at 10^-5 keeps half or more of `ref`'s APL-on response fraction (`low_2h_m5_frac_on`); None (not passed)
        while the point has not been run at the low concentrations.
    Returns each condition's values and flag, 'rates_held', 'bands', 'pass' (all four and the bands) and
    'pass_without_iv'."""
    g = GATE
    f = lambda k, d=row: float(d[k]) if k in d and d[k] is not None and not (isinstance(d[k], float) and np.isnan(d[k])) else np.nan
    held = row.get('rates_held', True)
    held = not (held is False or str(held).lower() in ('false', '0', '0.0'))
    out: Dict[str, object] = dict(point_id=row.get('point_id'), start=row.get('start'), rates_held=held)
    e, e0 = f('cal_err_ester'), f('cal_err_ester', ref)
    red = 1.0 - abs(e) / abs(e0) if np.isfinite(e) and np.isfinite(e0) and e0 != 0 else np.nan
    raise6 = f('raw_six_effect') - f('raw_six_effect', ref)
    out.update(ester_error=e, ester_error_start=e0, ester_reduction=red, six_effect=f('raw_six_effect'),
        six_effect_start=f('raw_six_effect', ref), six_raise=raise6)
    out['i'] = bool(held and np.isfinite(red) and red >= g['ester_reduction'] - 1e-12 and np.isfinite(raise6)
        and raise6 >= g['six_raise'] - 1e-12)
    if loss_limit is None:
        loss_limit = f('cal_loss', ref) + (g['loss_tol_ceiling'] if ceiling else g['loss_tol'])
    out.update(cal_loss=f('cal_loss'), cal_loss_start=f('cal_loss', ref), loss_limit=float(loss_limit))
    out['ii'] = bool(held and np.isfinite(out['cal_loss']) and out['cal_loss'] <= loss_limit + 1e-12)
    worst, worst_odor = np.inf, None
    for c in ('on', 'off'):
        for od in H.APL16_ODORS:
            a, b = f(f'frac_{c}_{od}'), f(f'frac_{c}_{od}', ref)
            if np.isfinite(b) and b > 0:
                q = a / b if np.isfinite(a) else 0.0
                if q < worst:
                    worst, worst_odor = q, f'{od} ({c})'
    out.update(worst_frac_ratio=float(worst), worst_frac_odor=worst_odor, rate_rho_on=f('rate_rho_on'))
    out['iii'] = bool(held and worst >= g['odor_frac_keep'] - 1e-12 and np.isfinite(out['rate_rho_on'])
        and out['rate_rho_on'] >= g['rate_rho_min'] - 1e-12)
    lo, lo0 = f('low_2h_m5_frac_on'), f('low_2h_m5_frac_on', ref)
    out.update(low_2h_m5_frac_on=lo, low_2h_m5_frac_on_start=lo0,
        low_ratio=lo / lo0 if np.isfinite(lo) and np.isfinite(lo0) and lo0 > 0 else np.nan)
    out['iv'] = None if not (np.isfinite(lo) and np.isfinite(lo0)) else bool(held and (lo0 <= 0 or lo >= g['low_conc_keep'] * lo0 - 1e-12))
    flag = row.get('realism_flag_noise')
    flag = '' if flag is None or (isinstance(flag, float) and np.isnan(flag)) else str(flag)
    out.update(realism_flag_noise=flag, bands=flag == '')
    out['pass_without_iv'] = bool(held and out['bands'] and out['i'] and out['ii'] and out['iii'])
    out['pass'] = bool(out['pass_without_iv'] and out['iv'] is True)
    return out


def gate_suffix(breadth: str, ref_point: str) -> str:
    if (breadth, ref_point) == ('prepass', 'identity'):
        return ''
    if (breadth, ref_point) == ('held', 'identity_held'):
        return '_identity_held'
    return f'_{breadth}_{ref_point}'


def gate(out: Path, ref_point: str = 'identity', breadth: str = 'prepass') -> dict:
    """The gate on run `breadth`'s points (every start; not the identity points and not the KC-KC-fixed controls), with
    the low-concentration fractions merged from low_conc/ where run: (i), (iii) and (iv) against C2's `ref_point` row
    ('identity', C2 itself, as the entry specifies; 'identity_held', C2 with only the held breadth, for the held run),
    (ii) the readout loss at most GATE['loss_max_c2'] (0.060) for C2's points against C2 itself (C2's identity_held
    plus 0.004 against it), the start's own `ref_point` row plus 0.006 for the firing ceiling's (fit1) and plus 0.004 for
    the re-drawn wirings'. In the held run each row also carries the conditions against the other reference (held_* /
    ident_*). The transfers that go on to part D: at most two, the best passing knee point on C2 (by the readout loss)
    and the best passing odor-relative point if it passes too. Writes gate.json / gate.csv for the pre-pass run against
    C2 itself, gate_identity_held.* for the held run against identity_held, gate_<run>_<ref>.* otherwise."""
    import json
    if ref_point not in ('identity', 'identity_held') or breadth not in BREADTHS:
        raise ValueError(f'{ref_point=} {breadth=}')
    df = aggregate(out, breadth)
    lc = out / 'low_conc' / 'points_low_conc.csv'
    if lc.exists() and len(df):
        low = pd.read_csv(lc)
        df = df.drop(columns=[c for c in low.columns if c.startswith('low_') and c in df.columns]).merge(low, on='tag', how='left')
    rows, g = [], pd.DataFrame()
    c2 = df[df.start == 'C2'] if len(df) else df
    c2_refs = {i: c2[c2.point_id == i].iloc[0].to_dict() for i in ('identity', 'identity_held') if (c2.point_id == i).any()}
    if ref_point in c2_refs:
        other = 'identity_held' if ref_point == 'identity' else 'identity'

        def limit(start: str, refs: Dict[str, dict], which: str) -> float:
            if start == 'C2' and which == 'identity':
                return GATE['loss_max_c2']
            own = refs[which]['cal_loss']
            return float(own) + (GATE['loss_tol_ceiling'] if start == 'fit1' else GATE['loss_tol'])

        for start, d in df.groupby('start'):
            refs = {i: d[d.point_id == i].iloc[0].to_dict() for i in ('identity', 'identity_held') if (d.point_id == i).any()}
            if ref_point not in refs:
                continue
            ctrl = d['control'].astype(str).str.lower().isin(['true', '1', '1.0']) if 'control' in d else False
            for _, r in d[~d.point_id.isin(['identity', 'identity_held']) & ~ctrl].iterrows():
                row = dict(gate_point(r.to_dict(), c2_refs[ref_point], loss_limit=limit(start, refs, ref_point)), tag=r['tag'],
                    family=r['family'], ref_point=ref_point, run=breadth, kckc=r.get('kckc'))
                if other in refs and other in c2_refs:
                    alt = gate_point(r.to_dict(), c2_refs[other], loss_limit=limit(start, refs, other))
                    pre = 'held_' if other == 'identity_held' else 'ident_'
                    row.update({pre + k: alt[k] for k in ('i', 'ii', 'iii', 'iv', 'pass', 'pass_without_iv', 'loss_limit',
                        'ester_error_start', 'six_effect_start')})
                rows.append(row)
        g = pd.DataFrame(rows)
    suffix = gate_suffix(breadth, ref_point)
    if len(g):
        g.to_csv(out / f'gate{suffix}.csv', index=False)
    sel = []
    passed = g[(g.start == 'C2') & g['pass']] if len(g) else g
    if len(passed):
        cal = df.set_index('tag')['cal_loss']
        for fam in (('knee', 'knee_soft'), ('rel',)):
            x = passed[passed.family.isin(fam)]
            if len(x):
                best = x.loc[x.tag.map(cal).idxmin()]
                sel.append(dict(tag=best.tag, point_id=best.point_id, family=best.family, cal_loss=float(cal[best.tag])))
    res = dict(run=breadth, ref_point=ref_point, n_points=int(len(g)), n_pass=int(g['pass'].sum()) if len(g) else 0,
        n_pass_without_iv=int(g['pass_without_iv'].sum()) if len(g) else 0,
        n_rates_not_held=int((~g['rates_held']).sum()) if len(g) else 0, selected=sel, thresholds=GATE,
        points=g.to_dict('records') if len(g) else [])
    (out / f'gate{suffix}.json').write_text(json.dumps(res, indent=1, default=_json_default) + '\n')
    print(f"gate ({breadth} run, against {ref_point}): {res['n_pass']} of {res['n_points']} points pass ({res['n_pass_without_iv']} "
        f"without the low-concentration condition; {res['n_rates_not_held']} with a rate not held); selected "
        f"{[s['tag'] for s in sel]}", flush=True)
    return res


# ---------------------------------------------------------------------------
# part D
D_COMMON = ['-k', '2', '--aplpn-max', '3', '--rate-target', 'ctrl_frac', '--loss', 'apl', '--apl-readout', 'both', '--maxfev', '150',
    '--experiment', EXPERIMENT]


def wiring_args(start: str) -> List[str]:
    """E10_joint_fit.py's wiring options of a start (none for the real wiring)."""
    w = STARTS[start][2]
    if 'wiring' in w:
        return ['--wiring', w['wiring'], '--wiring-seed', str(int(w['wiring_seed']))]
    if 'wiring_file' in w:
        return ['--wiring-file', str(H.shared_dir() / w['wiring_file']), '--wiring-tag', w['wiring_tag']]
    return []


def d_commands(g: dict) -> List[List[str]]:
    """E10_joint_fit.py argument lists of part D's fits: for each knee transfer the gate selected (the odor-relative knee
    is not fitted), on the real wiring and on P1, P2 and M1, each from that wiring's held point (part C's point with the
    transfer on the wiring's part-B optimum, both response rates held; `sweep(out, start, [point])` makes it when
    missing; the pre-pass run, KC-KC tied), whose absolute parameters go to results/E78_claw_knee/init_params.json under
    the point's tag."""
    import json
    out = H.results_dir(EXPERIMENT)
    cmds = []
    ipath = out / 'init_params.json'
    init = json.loads(ipath.read_text()) if ipath.exists() else {}
    for sel in g.get('selected', []):
        if sel['family'] not in ('knee', 'knee_soft'):
            continue
        pid = sel['point_id']
        for start in FIT_WIRINGS:
            tag = point_tag(start, dict(point_id=pid, kckc='tied'), 'prepass')
            df = aggregate(out)
            if not ((df.tag == tag).any() if len(df) else False):
                sweep(out, start, [pid])
                df = aggregate(out)
            r = df.set_index('tag').loc[tag]
            init[tag] = {k: float(r[k]) for k in ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale',
                'wKCKC_scale')}
            ipath.write_text(json.dumps(init, indent=1) + '\n')
            c = D_COMMON + ['--claw-thr', f"{float(r['claw_thr']):g}"]
            if float(r['claw_thr_subgain']) > 0:
                c += ['--claw-thr-subgain', f"{float(r['claw_thr_subgain']):g}"]
            if float(r['claw_thr_width']) > 0:
                c += ['--claw-thr-width', f"{float(r['claw_thr_width']):g}"]
            c += wiring_args(start) + ['--init-params', f'results/{EXPERIMENT}/init_params.json:{tag}', '--tag', f'_from_{pid}']
            cmds.append(c)
    (out / 'partD_commands.json').write_text(json.dumps(cmds, indent=1) + '\n')
    return cmds


def d_launch(cmds: Sequence[Sequence[str]], max_jobs: int = 2) -> List[str]:
    """systemd-run --user units for the fits (nice 10, 16 threads, python -u, logs under outputs/E78_claw_knee/logs/),
    `max_jobs` at a time: each unit waits for the one `max_jobs` before it (After= / Requires= are not used, a wrapper
    loop waits on its pid file instead); returns the unit names."""
    import subprocess
    import time
    logs = H.outputs_dir(EXPERIMENT) / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    units = []
    py = str(Path(__file__).resolve().parents[2] / 'venv' / 'bin' / 'python')
    for i, c in enumerate(cmds):
        unit = f"e78d-{i}-{time.strftime('%H%M%S')}"
        wait = ''
        if i >= max_jobs:
            prev = units[i - max_jobs]
            wait = f'while systemctl --user is-active --quiet {prev}; do sleep 30; done; '
        cmd = ' '.join(['exec', py, '-u', 'E10_joint_fit.py', *map(str, c)])
        subprocess.run(['systemd-run', '--user', f'--unit={unit}', '--collect', '-p', 'Nice=10', f'--working-directory={H.HERE}',
            '--setenv=OMP_NUM_THREADS=16', '--setenv=OMP_THREAD_LIMIT=16', '--setenv=PYTHONUNBUFFERED=1',
            '/bin/bash', '-c', f'{wait}{cmd} > {logs}/{unit}.log 2>&1'], check=True)
        units.append(unit)
        print(f'{unit}: {cmd}', flush=True)
    return units


def wiring_label(r) -> Optional[str]:
    """The STARTS name of a fits.csv row's wiring: 'C2' for the real wiring, P<n> / M<n> for E72's re-drawn ones."""
    w, wt = r.get('wiring'), r.get('wiring_tag')
    if (w is None or (isinstance(w, float) and np.isnan(w))) and (wt is None or (isinstance(wt, float) and np.isnan(wt))):
        return 'C2'
    if w == 'claw_permutation_within_type':
        return f"P{int(float(r['wiring_seed']))}"
    if isinstance(wt, str) and wt.startswith('curveball_within_type_matched-seed'):
        return 'M' + wt.rsplit('seed', 1)[1]
    return None


def d_tables(out: Path) -> pd.DataFrame:
    """The part D fits (fits.csv rows with a --tag _from_<point>) paired with C2 and E72's part B fits per wiring, and the
    population of four (the mean over the wirings of the pair distances per condition, E72's population_row). Writes
    paired.csv and population.csv."""
    import E72_wiring_population as E72
    f = pd.read_csv(out / 'fits.csv')
    f = f[f.tag.astype(str).str.contains('_from_')]
    rows, pops = [], {}
    for _, r in f.iterrows():
        tag = str(r.tag)
        o = E72.load_fit(EXPERIMENT, tag)
        if o is None:
            continue
        wl = wiring_label(r)
        ref = E72.load_fit(*STARTS[wl][:2]) if wl else None
        row = dict(tag=tag, wiring=wl, point=tag.split('_from_')[1])
        for ro in ('raw', 'cal'):
            row.update({f'{ro}_{k}': v for k, v in E72.effect_stats(E72.distances(o, ro), ro).items()})
            if ref is not None:
                row.update({f'ref_{ro}_{k}': v for k, v in E72.effect_stats(E72.distances(ref, ro), ro).items()})
        rows.append(row)
        pops.setdefault(row['point'], []).append(o)
    pd.DataFrame(rows).to_csv(out / 'paired.csv', index=False)
    prow = []
    for pt, fits in pops.items():
        if len(fits) < 2:
            continue
        dist = {ro: [E72.distances(o, ro) for o in fits] for ro in ('raw', 'cal', 'binary')}
        prow.append(dict(point=pt, **E72.population_row(fits, dist)))
    pd.DataFrame(prow).to_csv(out / 'population.csv', index=False)
    return pd.DataFrame(rows)


def d_heldout(out: Path, point_id: str) -> pd.DataFrame:
    """Held out: the mean absolute parameters of part D's four fits of `point_id`'s transfer (real wiring, P1, P2, M1;
    fits.csv rows tagged _from_<point_id>) run once on P3, P4, M2 and M3 with the transfer (each wiring's own pre-pass
    breadth), nothing re-fitted or re-bisected; scored as part C's points (`score_point`, against the wiring's own part
    B fit) and as their population (E72's population_row). Writes heldout_<point_id>.csv and appends the population to
    population.csv as point 'heldout_<point_id>'."""
    import E72_wiring_population as E72
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    f = pd.read_csv(out / 'fits.csv')
    f = f[f.tag.astype(str).str.endswith(f'_from_{point_id}')]
    if len(f) < 4:
        raise RuntimeError(f'{len(f)} part D fits of {point_id}; need the four')
    keys = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
    pm = {k: float(f[k].mean()) for k in keys}
    pm['wKCKC_scale'] = float(f['kc_kc_inh'].mean())
    spec = {p['point_id']: p for p in point_specs(aggregate(out))}[point_id]
    import E10_joint_fit as J
    T = float(spec['claw_thr'])
    transfer = J.claw_transfer_kws(spec['claw_thr_subgain'] or None, spec['claw_thr_width'] or None)
    rows, fits = [], []
    orn = H.apl16_orn_deltas()
    for start in HELD_OUT:
        # the model as E10_joint_fit.py would build it for this wiring with the transfer (its base and pre-pass)
        w = dict(STARTS[start][2])
        if 'wiring_file' in w:
            w = dict(wiring_matrix=pd.read_parquet(H.shared_dir() / w['wiring_file']), wiring_tag=w['wiring_tag'])
        _, wnorm, W, breadth = J.bouton_base(orn, H.outputs_dir(EXPERIMENT) / 'C' / 'bases', T, claw_transfer=transfer or None, **w)
        kws = dict(J.base_kws(T, transfer=transfer or None), _wPNKC=W, _wAPLKC=J.breadth_weights(wnorm['wAPLKC'], breadth, 2.0),
            _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=pm['wKCKC_scale'], kc_kc_tau=J.TAU)
        sim, _ = make_fixed_param_kc_simulator(orn, initial_params=pm, **kws)
        o = {'on': sim(pm), 'off': sim(KneeModel.off(pm))}
        del sim
        ref = E72.load_fit(*STARTS[start][:2])
        row, _ = score_point(o, ref)
        rows.append(dict(start=start, point_id=point_id, **pm, **row))
        fits.append(o)
        print(f"held out {start}: readout loss {row['cal_loss']:.4f}, raw {row['raw_loss']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f'heldout_{point_id}.csv', index=False)
    dist = {ro: [E72.distances(o, ro) for o in fits] for ro in ('raw', 'cal', 'binary')}
    pp = out / 'population.csv'
    pop = pd.read_csv(pp) if pp.exists() else pd.DataFrame(columns=['point'])
    pop = pop[pop.point.astype(str) != f'heldout_{point_id}']
    pd.concat([pop, pd.DataFrame([dict(point=f'heldout_{point_id}', **E72.population_row(fits, dist))])]).to_csv(pp, index=False)
    return df


# ---------------------------------------------------------------------------
# part E and the summary
def release_rows(out: Path) -> pd.DataFrame:
    """Who is released when APL is blocked, by claw count (E72's release.csv form, E71's claw groups): for C2, every part
    A / D fit of this experiment and every part C point, the share of the responses blocking APL adds that comes from
    each claw-count group and the squared CV of the number of odors per KC with APL on and off, beside the KC data's.
    Writes release.csv."""
    import E10_joint_fit as J
    import E71_input_scaled_threshold as E71
    import E72_wiring_population as E72
    _, _, W, _ = J.bouton_base(H.apl16_orn_deltas(), H.outputs_dir(EXPERIMENT) / 'C' / 'bases', None)
    claws = E71.n_claws_per_kc(W)
    items = [('C2', C2[0], C2[1], None)]
    fp = out / 'fits.csv'
    if fp.exists():
        items += [(str(t), EXPERIMENT, str(t), None) for t in pd.read_csv(fp).tag]
    for p in sorted(c_outputs_dir().glob('*_on_responses.parquet')):
        tag = p.name[:-len('_on_responses.parquet')]
        items.append((tag, None, tag, c_outputs_dir()))
    rows = []
    for name, folder, tag, d in items:
        if d is None:
            o = E72.load_fit(folder, tag)
        else:
            o = {c: (pd.read_parquet(d / f'{tag}_{COND_FILE[c]}_responses.parquet'), None) for c in ('on', 'off')}
        if o is None:
            continue
        on, off = (o[c][0].to_numpy(dtype=float) > 0.5 for c in ('on', 'off'))
        n_on, n_off = on.sum(axis=1), off.sum(axis=1)
        c = claws.reindex(o['on'][0].index.get_level_values(H.KC_ID_LEVEL)).to_numpy()
        added = float(n_off.sum() - n_on.sum())
        row = dict(name=name, folder=folder or f'{EXPERIMENT} (part C)', tag=tag, n_added=added,
            opk_cv2_on=float(n_on.var() / n_on.mean() ** 2), opk_cv2_off=float(n_off.var() / n_off.mean() ** 2))
        for g, lo, hi in E71.CLAW_GROUPS:
            m = (c >= lo) & (c <= hi)
            row[f'share_added_{g}'] = float((n_off[m].sum() - n_on[m].sum()) / added) if added else np.nan
            row[f'frac_kcs_{g}'] = float(m.mean())
        rows.append(row)
    dref = E71.data_refs()
    rows.append(dict(name='KC data', folder='', tag='', opk_cv2_on=dref['opk_cv2_on']['mean'], opk_cv2_off=dref['opk_cv2_off']['mean']))
    res = pd.DataFrame(rows)
    res.to_csv(out / 'release.csv', index=False)
    return res


def e_score(out: Path) -> pd.DataFrame:
    """Part E: who is released when APL is blocked (`release_rows`); then every fit and gate point under the per-KC
    reading with E73's committed scorer, once archived (`E73_per_kc_normalization.py --part B --folders E78_claw_knee`, in
    its own process; it reads the model-output archive), and E77's inherited / curve / pair-specific split when its
    harness function exists (`H.apl_effect_inherited_split`)."""
    import subprocess
    import sys
    rel = release_rows(out)
    arch = H.shared_dir() / 'data' / 'model_outputs' / EXPERIMENT
    if arch.exists():
        subprocess.run([sys.executable, '-u', 'E73_per_kc_normalization.py', '--part', 'B', '--folders', EXPERIMENT], cwd=H.HERE,
            check=False)
    else:
        print(f'{arch} does not exist yet: archive the outputs first (archive_model_outputs.py {EXPERIMENT}), then run --part E '
            'again for the per-KC reading', flush=True)
    if hasattr(H, 'apl_effect_inherited_split'):
        import E75_glomerulus_apl_profile as E75
        df = aggregate(out)
        d_on, d_off = H.yang_mean_rdm('TNT3c'), H.yang_mean_rdm('TNT')
        rows = []
        for _, r in df.iterrows():
            pp = points_dir(r['start'], r.get('run') or 'prepass') / f"{r.get('point_key') or r['point_id']}_pairs.csv"
            if not pp.exists():
                continue
            d = pd.read_csv(pp)
            for ro in ('raw', 'cal'):
                x = d[d.readout == ro]
                on = x[x.condition == 'on'].set_index(['odor_a', 'odor_b'])['distance']
                off = x[x.condition == 'acute_off'].set_index(['odor_a', 'odor_b'])['distance']
                try:
                    sp = H.apl_effect_inherited_split(on, off, d_on, d_off)
                except Exception as e:
                    print(f"{r['tag']} ({ro}): inherited split failed: {e!r}"[:200], flush=True)
                    continue
                row = dict(tag=r['tag'], start=r['start'], point_id=r['point_id'], readout=ro)
                tot = float((sp['error_c'] ** 2).sum()) if 'error_c' in sp else np.nan
                for term in ('inherited', 'curve', 'pair_specific'):
                    c = sp[f'{term}_c'].to_numpy(dtype=float)
                    row[f'{term}_rms'] = float(np.sqrt(np.mean(c ** 2)))
                    row[f'{term}_share'] = float((c * sp['error_c'].to_numpy(dtype=float)).sum() / tot) if tot else np.nan
                    for grp, odors in (('ester', ESTER), ('alcohol', ALCOHOL)):
                        row[f'{term}_{grp}'] = float(np.nanmean(c[E75.pair_mask(sp.index, odors)]))
                rows.append(row)
        pd.DataFrame(rows).to_csv(out / 'inherited_split.csv', index=False)
    return rel


def _f(x, spec='.3f') -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '-'
    return '-' if not np.isfinite(x) else format(x, spec)


def summarize(out: Path, breadth: str = 'prepass') -> None:
    """comparison_table.md (the sweep's points against their start and the KC data: the readout loss with its pattern
    part, the raw loss and mean, ρ on the per-pair APL effect, the six pairs, the ester and alcohol group errors, the
    response rates, the response rate ρ, the weak odors' fractions, the bands; the gate; the archive's knee variants)
    and the figures (E78_sweep.png: the loss, the pattern part, the six pairs' effect and the ester error against the knee
    and the gain; E78_weak_odors.png: the weak odors' response fractions), for run `breadth` (the held run's files carry
    a _held suffix)."""
    import json
    sfx = '' if breadth == 'prepass' else f'_{breadth}'
    df = aggregate(out, breadth)
    if not len(df):
        print('no points yet', flush=True)
        return
    data_six = kind_mean(H.yang_apl_effect(), SIX)
    import apl_effect_scores as S
    fr_on, fr_off = S.frac_data('on', 'noise'), S.frac_data('off', 'noise')
    lines = [f'# E78: the high per-claw knee without a dead zone, part C (fixed scales, both response rates held)', '',
        'Rows: each start (C2, the breadth-2 APL-effect fit; fit1, the same with the firing ceiling at strength 1; P1, ..., '
        "E72's re-drawn wirings) and each point on it. Lower is better for the losses; the floor of the APL-effect loss is "
        f"{H.data_target('apl_effect_rmse_floor'):.3f}. Six pairs: eb-ep, 6al-t2h, IaA-pa, 2h-pa, 1-5ol-1-6ol, 2h-IaA (raw counts; "
        f'the KC data\'s APL effect +{data_six:.3f}). Bands: the 16:10 rule (two spikes against the KC data in baseline-noise units).', '',
        (("Breadth (the pre-pass run): each point's breadth weights from E10's transfer pre-pass, the plain bouton base of the "
        "start's wiring at E3's tuned parameters with APL off and the whole transfer, its threshold re-tuned to the transfer-free "
        "run's APL-off response count (at the identity C2's breadth exactly); KC-KC column 'tied': the KC-KC scale multiplied by "
        "the somatic threshold's factor when the APL-off count is held, 'fixed': held at the start's (the controls, not gated). "
        "Part A's E10 fits at 100 / 130 Hz (and part D's) take their breadth from E16's knee base (its APL-off run at its own "
        "tuned threshold) and fit the KC-KC scale.") if breadth == 'prepass' else
        ("Breadth (the held run, the first one): every point but 'identity' takes the breadth weights from its own "
        "threshold-held APL-off run (E77's runner, breadth_from='held', KC-KC inhibition included) with the KC-KC scale held at "
        "the start's; 'identity_held' is the start with only the held breadth. Kept for the record; the pre-pass run replaces "
        "it.")) + " Rows with 'no' under rates held had a bisection end at its bound (the response counts not reached).", '',
        '| start | point | knee (Hz) | g | width | rel | readout loss | pattern part | raw loss | raw mean | ρ on the per-pair APL effect '
        '(readout) | six pairs, APL on / off (raw) | six pairs\' APL effect (raw) | ester group error | alcohol group error | '
        'response rate, APL on / off | response rate ρ, APL on | silent odors (on / off) | bands | KC-KC | rates held |',
        '|---|---|' + '---:|' * 15 + '---|---|---|---|']
    order = {s: i for i, s in enumerate(STARTS)}
    df = df.assign(_o=df.start.map(order)).sort_values(['_o', 'family', 'claw_thr', 'claw_thr_subgain', 'rel'])
    for _, r in df.iterrows():
        lines.append(f"| {r.start} | {r.point_id} | {_f(r.claw_thr, 'g')} | {_f(r.claw_thr_subgain, 'g')} | {_f(r.claw_thr_width, 'g')} | "
            f"{_f(r.rel, 'g')} | {_f(r.cal_loss, '.4f')} | {_f(r.cal_pattern, '.4f')} | {_f(r.raw_loss, '.4f')} | {_f(r.raw_mean, '+.3f')} | "
            f"{_f(r.cal_rho)} | {_f(r.raw_six_on)} / {_f(r.raw_six_off)} | {_f(r.raw_six_effect, '+.3f')} | {_f(r.cal_err_ester, '+.3f')} | "
            f"{_f(r.cal_err_alcohol, '+.3f')} | {_f(r.rate_on, '.4f')} / {_f(r.rate_off, '.4f')} | {_f(r.rate_rho_on, '.2f')} | "
            f"{r.silent_odors_on if isinstance(r.silent_odors_on, str) and r.silent_odors_on else '-'} / "
            f"{r.silent_odors_off if isinstance(r.silent_odors_off, str) and r.silent_odors_off else '-'} | "
            f"{'inside' if not (isinstance(r.realism_flag_noise, str) and r.realism_flag_noise) else r.realism_flag_noise} | "
            f"{r.get('kckc') if isinstance(r.get('kckc'), str) and r.get('kckc') else '-'} | "
            f"{'yes' if str(r.get('rates_held')).lower() in ('true', '1', '1.0') else 'no'} |")
    lines += [f"| KC data | | | | | | floor {H.data_target('apl_effect_rmse_floor'):.3f} | | | +{H.yang_apl_effect().mean():.3f} | "
        f"ceiling {H.data_target('apl_effect_rho_ceiling'):.2f} | {kind_mean(H.yang_mean_rdm('TNT3c'), SIX):.3f} / "
        f"{kind_mean(H.yang_mean_rdm('TNT'), SIX):.3f} | +{data_six:.3f} | 0 | 0 | {fr_on.mean():.4f} / {fr_off.mean():.4f} | | | | | |"]
    gp = out / ('gate.json' if breadth == 'prepass' else 'gate_identity_held.json')
    if gp.exists():
        g = json.loads(gp.read_text())
        lines += ['', '## The gate', '', f"{g['n_pass']} of {g['n_points']} points pass ({g['n_pass_without_iv']} without the "
            f"low-concentration condition); selected: {', '.join(s['tag'] for s in g['selected']) or 'none'}.", '',
            f"Against C2's {g.get('ref_point', 'identity')} row; (ii) against the loss limit given.", '',
            '| tag | (i) ester error reduced / six pairs raised | (ii) readout loss / limit | (iii) worst odor kept / response rate ρ | '
            '(iv) 2h at 10^-5 kept | bands | pass |', '|---|---|---|---|---|---|---|']
        for p in g['points']:
            lines.append(f"| {p['tag']} | {_f(p['ester_reduction'], '.2f')} / {_f(p['six_raise'], '+.3f')} ({'yes' if p['i'] else 'no'}) | "
                f"{_f(p['cal_loss'], '.4f')} / {_f(p.get('loss_limit', p.get('cal_loss_start')), '.4f')} ({'yes' if p['ii'] else 'no'}) | "
                f"{_f(p['worst_frac_ratio'], '.2f')} ({p['worst_frac_odor']}) / {_f(p['rate_rho_on'], '.2f')} ({'yes' if p['iii'] else 'no'}) | "
                f"{_f(p['low_ratio'], '.2f')} ({'-' if p['iv'] is None else ('yes' if p['iv'] else 'no')}) | "
                f"{'yes' if p['bands'] else 'no'} | {'yes' if p['pass'] else 'no'} |")
    ap = out / 'archive_knee_variants.csv'
    if ap.exists():
        a = pd.read_csv(ap)
        lines += ['', "## The archive's knee variants (part B; raw counts, each against its own base)", '',
            '| folder | tag | base | knee (Hz) | knee / largest PN rate | six pairs, APL on / off (effect) | change of the six pairs\' effect | '
            'ester / alcohol group error | pattern part (raw) | ρ on the per-pair APL effect (raw) | silent odors, APL on |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for _, r in a.iterrows():
            lines.append(f"| {r.folder} | {r.tag} | {r.base_tag if isinstance(r.base_tag, str) else '-'} | {_f(r.knee, 'g')} | "
                f"{_f(r.knee_over_top, '.2f')} | {_f(r.raw_six_on)} / {_f(r.raw_six_off)} ({_f(r.raw_six_effect, '+.3f')}) | "
                f"{_f(r.get('d_raw_six_effect'), '+.3f')} | {_f(r.get('raw_err_ester'), '+.3f')} / {_f(r.get('raw_err_alcohol'), '+.3f')} | "
                f"{_f(r.raw_pattern)} | {_f(r.raw_rho, '.2f')} | {r.silent_odors_on if isinstance(r.get('silent_odors_on'), str) and r.silent_odors_on else '-'} |")
    (out / f'comparison_table{sfx}.md').write_text('\n'.join(lines) + '\n')
    _figures(df[df.get('control', False).astype(str).str.lower() != 'true'] if 'control' in df else df, out, sfx)
    print(f"{out / f'comparison_table{sfx}.md'} written ({len(df)} points)", flush=True)


def _figures(df: pd.DataFrame, out: Path, sfx: str = '') -> None:
    plt = H._plt()
    d = df[(df.start == 'C2') & df.family.isin(['knee', 'knee_soft'])]
    ref = df[(df.start == 'C2') & (df.point_id == 'identity')]
    if len(d):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
        panels = [('cal_loss', 'APL-effect loss, calibrated readout'), ('cal_pattern', 'its pattern part'),
            ('raw_six_effect', "six near pairs' APL effect (raw counts)"), ('cal_err_ester', 'ester group error (readout)')]
        data_vals = {'raw_six_effect': kind_mean(H.yang_apl_effect(), SIX), 'cal_err_ester': 0.0,
            'cal_loss': H.data_target('apl_effect_rmse_floor')}
        for ax, (col, title) in zip(axes, panels):
            for (T, w), g in d.groupby(['claw_thr', 'claw_thr_width']):
                g = g.sort_values('claw_thr_subgain')
                ax.plot(g.claw_thr_subgain, g[col], '-o' if not w else ':s', ms=5, label=f'knee {T:g} Hz' + (f', width {w:g}' if w else ''))
            if len(ref):
                ax.axhline(float(ref[col].iloc[0]), color='0.3', ls='--', lw=1, label='C2 (no knee)')
            if col in data_vals:
                ax.axhline(data_vals[col], color='k', ls=':', lw=1, label='KC data' if col != 'cal_loss' else 'floor')
            ax.set(xlabel='sub-threshold gain g', title=title)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle('The per-claw knee on the breadth-2 APL-effect fit, both response rates held', fontsize=10.5)
        fig.tight_layout()
        fig.savefig(out / f'E78_sweep{sfx}.png', dpi=130)
        plt.close(fig)
    weak = ['ms', 'B-myr', 'euc', '-aPine', 'paa', '1-8ol']
    d = df[df.start == 'C2']
    if len(d):
        import apl_effect_scores as S
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=False)
        x = np.arange(len(d))
        for ax, c in zip(axes, ('on', 'off')):
            for i, od in enumerate(weak):
                ax.plot(x, d[f'frac_{c}_{od}'], '-o', ms=4, label=od)
            ax.set_xticks(x)
            ax.set_xticklabels(d.point_id, rotation=60, ha='right', fontsize=7)
            ax.set(title=f"weak odors' response fractions, APL {c}", ylabel='response fraction')
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f'E78_weak_odors{sfx}.png', dpi=130)
        plt.close(fig)


def main() -> None:
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == '--run-e10':
        _run_e10_redirected(Path(sys.argv[2]), sys.argv[4:])
        return
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', required=True, choices=['check-e10', 'check-C', 'B', 'C', 'C-launch', 'C-low', 'gate', 'D', 'E',
        'summarize'])
    ap.add_argument('--breadth', default='prepass', choices=list(BREADTHS),
        help="parts C, C-low, gate, summarize: the run: 'prepass' (default; each point's breadth from E10's transfer pre-pass) or "
            "'held' (the first run: the breadth from the threshold-held APL-off run)")
    ap.add_argument('--kckc', default=None, choices=list(KCKC_MODES),
        help="part C: 'tied' (the default of the pre-pass run; the KC-KC scale multiplied by the somatic threshold's factor when "
            "the APL-off count is held) or 'fixed' (held at the start's, the entry's letter; the held run's only mode)")
    ap.add_argument('--start', default='C2', choices=list(STARTS), help='part C: the fit the points are set on')
    ap.add_argument('--only', nargs='+', default=None, help='part C: these point ids only (check-e10: these runs)')
    ap.add_argument('--best', action='store_true', help=f"part C / C-low: the {N_BEST} best points of the sweep on C2 (by the "
        'readout loss, inside the bands), e.g. with --start fit1 or P1 for the repeats')
    ap.add_argument('--jobs', type=int, default=1, help='check-e10: runs at a time')
    ap.add_argument('--gate-ref', default='identity', choices=['identity', 'identity_held'],
        help="gate: compare each point with its start ('identity', the entry's reference) or with the start carrying only "
            "the held breadth ('identity_held', like for like with the points' breadth); both are in the table")
    ap.add_argument('--baseline', default=None, metavar='COMMIT', help="check-e10: also run the 'off_*' runs on E10_joint_fit.py as of COMMIT")
    ap.add_argument('--local-test', metavar='DIR', default=None,
        help="write this experiment's results and outputs under DIR/results and DIR/outputs (a smoke test)")
    args = ap.parse_args()
    if args.kckc is None:
        args.kckc = 'tied' if args.breadth == 'prepass' else 'fixed'
    if args.local_test:
        root = Path(args.local_test).resolve()
        orig_r, orig_o = H.results_dir, H.outputs_dir
        H.results_dir = lambda e, _o=orig_r: (root / 'results') if e == EXPERIMENT else _o(e)
        H.outputs_dir = lambda e, _o=orig_o: (root / 'outputs') if e == EXPERIMENT else _o(e)
        for sub in ('results', 'outputs'):
            (root / sub).mkdir(parents=True, exist_ok=True)
    out = H.results_dir(EXPERIMENT)
    out.mkdir(parents=True, exist_ok=True)
    if args.part == 'check-e10':
        check_e10(args.only, args.jobs, args.baseline)
    elif args.part == 'B':
        for f in (archive_knee_variants, noise_six_pairs, who_selects):
            f(out)
    elif args.part == 'C':
        only = args.only
        if args.best:
            only = best_points(aggregate(out, args.breadth))
            print(f'the {N_BEST} best points on C2: {only}', flush=True)
        sweep(out, args.start, only, args.breadth, args.kckc)
    elif args.part == 'check-C':
        check_c(out, args.start)
    elif args.part == 'C-launch':
        c_launch()
    elif args.part == 'C-low':
        low_conc(out, args.only if args.only and not args.best else best_points(aggregate(out, args.breadth)), args.start, args.breadth)
    elif args.part == 'gate':
        gate(out, args.gate_ref, args.breadth)
    elif args.part == 'D':
        d_launch(d_commands(gate(out)))
    elif args.part == 'E':
        e_score(out)
    else:
        summarize(out, args.breadth)


if __name__ == '__main__':
    main()
