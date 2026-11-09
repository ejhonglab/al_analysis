#!/usr/bin/env python3
"""E54: the gain and the wiring behind the natural-source alignment.
The glomerulus-pairs experiment (E52) found two wiring carriers of the fitted models' natural-source
alignment: the KC-type organization of the glomerular input, and the within-type over-sharing of the
food glomeruli (55 pairs), shown by a lesion of those pairs' excess that took the fixed-scale alignment
with APL off from 0.36 to 0.31 while a count-matched control lesion did not. Four of the eight gained
glomeruli (DM1, DL1, DC2, D) are in those pairs, so the lesion also redistributes gained claws, and the
no-gain fits at 6.8 % have an alignment of only 0.25 / 0.27 against 0.36 / 0.37 with the gain. This asks
whether the pairwise carrier is the convergence itself or the gain riding on it, and whether a re-fit on
the lesioned wiring recovers the alignment.
Parts (all at the control flies' 6.8 %, scored as E47 / E52 score their variants):
(A) the three reference models at their fitted scales with the gain set to 1 on the real wiring
    (threshold re-tuned), beside E45's no-gain fits;
(B) E52's targeted and control lesions (its three draws, outputs/E52_glomerulus_pairs/wPNKC_*.parquet
    on the host that ran E52) under the two no-gain fits at their fitted scales;
(C) under the gained models, the targeted lesion split into the over-shared pairs with ORN data that
    involve a gained glomerulus and those that do not (the pairs among glomeruli without ORN data left
    out), each with its own count-matched control lesion and the other subset's pairs protected, two
    draws each (`lesion_coinput`, within KC type, every margin kept);
(D) the re-fit, run separately: `E10_joint_fit.py --experiment E54_gain_and_lesion --loss structure -k 2
    --aplpn-max 3 --rate-target ctrl_frac --maxfev 150 --claw-thr 40 --glom-gain 1.5 --wiring-file
    outputs/E52_glomerulus_pairs/wPNKC_lesion_targeted-seed0.parquet --wiring-tag lesion_targeted-seed0
    --init-from thr40_gain1.5_rate0.068_k2_cap3 --init-from-experiment E44_operating_point
    --init-from-exact --tag _cap3`; --summary puts its fits.csv row beside the fixed-scale runs.
Outputs: <shared>/results/E54_gain_and_lesion/ (lesion_pairs.csv, lesion_outcome.csv, stats_variants.csv,
per_odor_response_fractions.csv, model_pair_distances_long.csv, comparison.csv, comparison_table.md,
E54_gain_and_lesion.png, provenance.json; fits.csv from the re-fit) and ./outputs/E54_gain_and_lesion/
(lesioned wirings, swaps, per-variant responses / spike counts, partC/ caches).
Usage: ./venv/bin/python E54_gain_and_lesion.py --plan            (the lesions only)
       OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E54_gain_and_lesion.py [--parts A B C] [--seeds 0 1]
       ./venv/bin/python E54_gain_and_lesion.py --summary
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E47_wiring_features import MODELS, STATS, V, fitted_params, SCALE_KEYS, CTRL_FRAC, data_values
from E52_glomerulus_pairs import base_wiring, NULL_CLASS, EXPERIMENT as E52

EXPERIMENT = 'E54_gain_and_lesion'
# the no-gain fits at 6.8 % (E45): key -> (experiment, tag, claw_thr, k, gain)
MODELS_NOGAIN: Dict[str, Tuple[str, str, float, float, Optional[float]]] = {
    'thr40_nogain': ('E45_gain_at_ctrl_frac', 'thr40_rate0.068_k2_cap3', 40.0, 2.0, None),
    'thr70_nogain': ('E45_gain_at_ctrl_frac', 'thr70_rate0.068_k2_cap3', 70.0, 2.0, None),
}
MODEL_LABEL = {'thr70_gain1.5': 'eight at 1.5, 70 Hz', 'thr40_gain1.5': 'eight at 1.5, 40 Hz',
    'thr70_gain1.25': 'eight at 1.25, 70 Hz', 'thr40_nogain': 'no gain, 40 Hz', 'thr70_nogain': 'no gain, 70 Hz'}
SPLIT_LESIONS = ('lesion_gained', 'control_gained', 'lesion_ungained', 'control_ungained')
# part C2: the two halves together (the 32 over-shared pairs with ORN data) and the 23 pairs among glomeruli
# without ORN data on their own, each with its control
PART_C2 = ('lesion_orn_all', 'control_orn_all', 'lesion_nonorn', 'control_nonorn')
ALL_LESIONS = SPLIT_LESIONS + PART_C2
GROUP_LABEL = {'real': 'real wiring', 'nogain': 'gain set to 1', 'lesion_targeted': 'all 55 over-shared pairs lesioned (E52)',
    'lesion_control': 'matched control pairs lesioned (E52)', 'lesion_gained': 'over-shared pairs with a gained glomerulus lesioned',
    'control_gained': 'their matched control pairs lesioned', 'lesion_ungained': 'over-shared pairs without gained glomeruli lesioned',
    'control_ungained': 'their matched control pairs lesioned', 'lesion_orn_all': 'all 32 over-shared pairs with ORN data lesioned',
    'control_orn_all': 'their 32 matched control pairs lesioned', 'lesion_nonorn': 'the 23 over-shared pairs without ORN data lesioned',
    'control_nonorn': 'their matched control pairs lesioned (E52\'s)'}


def e52_results() -> Path:
    return H.shared_dir() / 'results' / E52


def split_plan(res_dir: Path) -> pd.DataFrame:
    """E52's targeted pairs with ORN data split by whether a gained glomerulus is in the pair, each
    subset with its own count-matched control pairs (|z| < 1 against every class, same ORN-input
    status, not used by E52's control or the other subset's)."""
    from E27_glomerulus_gain import ORACLE
    df = pd.read_csv(e52_results() / 'coinput_pairs.csv')
    plan52 = pd.read_csv(e52_results() / 'lesion_pairs.csv')
    targeted = plan52[(plan52.kind == 'lesion_targeted') & plan52.in_orn_input]
    gained = set(ORACLE)
    used = {(a, b) for a, b in zip(plan52.glom_a, plan52.glom_b)}
    cand = df[(df.min_z > -1) & (df.max_z < 1) & df.in_orn_input].copy()
    cand = cand[[(a, b) not in used for a, b in zip(cand.glom_a, cand.glom_b)]]
    rows = []
    taken = set()
    for subset in ('gained', 'ungained'):
        sub = targeted[[((a in gained) or (b in gained)) == (subset == 'gained') for a, b in zip(targeted.glom_a, targeted.glom_b)]]
        for r in sub.itertuples():
            excess = int(r.observed) - int(r.target)
            rows.append(dict(kind=f'lesion_{subset}', glom_a=r.glom_a, glom_b=r.glom_b, observed=int(r.observed), target=int(r.target),
                n_removed=excess, null_mean=r.null_mean, min_z=r.min_z, matched_to=''))
            pool = cand[~cand.index.isin(taken)]
            j = (pool.observed - r.observed).abs().idxmin()
            taken.add(j)
            c = cand.loc[j]
            rows.append(dict(kind=f'control_{subset}', glom_a=c.glom_a, glom_b=c.glom_b, observed=int(c.observed),
                target=max(int(c.observed) - excess, 0), n_removed=min(excess, int(c.observed)), null_mean=c[f'{NULL_CLASS}_null_mean'],
                min_z=c.min_z, matched_to=f'{r.glom_a}/{r.glom_b}'))
    plan = pd.DataFrame(rows)
    # part C2: the two halves together, and the pairs without ORN data (with E52's controls matched to them)
    both = plan[plan.kind.isin(['lesion_gained', 'lesion_ungained'])].assign(kind='lesion_orn_all')
    both_c = plan[plan.kind.isin(['control_gained', 'control_ungained'])].assign(kind='control_orn_all')
    nonorn = plan52[(plan52.kind == 'lesion_targeted') & ~plan52.in_orn_input].assign(kind='lesion_nonorn', matched_to='')
    nonorn_keys = {f'{a}/{b}' for a, b in zip(nonorn.glom_a, nonorn.glom_b)}
    nonorn_c = plan52[(plan52.kind == 'lesion_control') & plan52.matched_to.isin(nonorn_keys)].assign(kind='control_nonorn')
    cols = ['kind', 'glom_a', 'glom_b', 'observed', 'target', 'n_removed', 'null_mean', 'min_z', 'matched_to']
    plan = pd.concat([plan, both, both_c, nonorn[cols], nonorn_c[cols]], ignore_index=True)
    plan.to_csv(res_dir / 'lesion_pairs.csv', index=False)
    return plan


def make_split_lesions(W: pd.DataFrame, plan: pd.DataFrame, seeds: List[int], out_dir: Path, res_dir: Path,
    kinds: Tuple[str, ...] = SPLIT_LESIONS) -> Dict[Tuple[str, int], pd.DataFrame]:
    """The four split lesions per seed: each subset's pairs brought to their targets with the other
    over-shared pairs (and, for a control, every over-shared pair) protected."""
    from al_analysis.wpnkc_shuffles import lesion_coinput, coinput_counts, wPNKC_shuffle_summary
    plan52 = pd.read_csv(e52_results() / 'lesion_pairs.csv')
    over_all = [(a, b) for a, b in zip(plan52[plan52.kind == 'lesion_targeted'].glom_a, plan52[plan52.kind == 'lesion_targeted'].glom_b)]
    obs = coinput_counts(W)
    out: Dict[Tuple[str, int], pd.DataFrame] = {}
    oc_path = res_dir / 'lesion_outcome.csv'
    outcome = list(pd.read_csv(oc_path).to_dict('records')) if oc_path.exists() else []
    outcome = [r for r in outcome if not (r['kind'] in kinds and int(r['seed']) in seeds)]
    for kind in kinds:
        sub = plan[plan.kind == kind]
        targets = {(a, b): int(t) for a, b, t in zip(sub.glom_a, sub.glom_b, sub.target)}
        own = set(targets)
        protect = [p for p in over_all if p not in own]
        for seed in seeds:
            path = out_dir / f'wPNKC_{kind}-seed{seed}.parquet'
            if path.exists():
                L = pd.read_parquet(path); swaps = pd.read_csv(out_dir / f'swaps_{kind}-seed{seed}.csv')
            else:
                t0 = time.time()
                L, swaps = lesion_coinput(W, targets, seed=seed, within_kc_type=True, protect=protect, strict=False)
                L.to_parquet(path); swaps.to_csv(out_dir / f'swaps_{kind}-seed{seed}.csv', index=False)
                print(f'{kind} seed {seed}: {len(swaps)} claw swaps in {time.time() - t0:.0f}s', flush=True)
            out[(kind, seed)] = L
            co = coinput_counts(L); s = wPNKC_shuffle_summary(W, L)
            outcome.append(dict(kind=kind, seed=seed, n_pairs=len(targets), n_swaps=len(swaps),
                frac_claws_changed=float(s['frac_claws_changed']), coinput_corr=float(s['coinput_corr']),
                coinput_resid_corr=float(s['coinput_resid_corr']), kc_n_claws_preserved=bool(s['kc_n_claws_preserved']),
                max_abs_glom_n_claws_change=float(s['max_abs_glom_n_claws_change']),
                sum_removed_listed=int(sum(int(obs.loc[a, b]) - int(co.loc[a, b]) for a, b in targets)),
                sum_planned=int(sub.n_removed.sum()),
                over_shared_total_before=int(sum(int(obs.loc[a, b]) for a, b in over_all)),
                over_shared_total_after=int(sum(int(co.loc[a, b]) for a, b in over_all)),
                protected_changed=int(sum(abs(int(obs.loc[a, b]) - int(co.loc[a, b])) for a, b in protect)),
                reached=json.dumps({f'{a}/{b}': int(co.loc[a, b]) for a, b in targets})))
            pd.DataFrame(outcome).to_csv(oc_path, index=False)
    return out


def e52_lesions(seeds: List[int]) -> Dict[Tuple[str, int], pd.DataFrame]:
    d = H.outputs_dir(E52)
    out = {}
    for kind in ('lesion_targeted', 'lesion_control'):
        for seed in seeds:
            p = d / f'wPNKC_{kind}-seed{seed}.parquet'
            if p.exists():
                out[(kind, seed)] = pd.read_parquet(p)
            else:
                print(f'missing {p} (E52 ran on another host?)', flush=True)
    return out


def run_variants(jobs: List[dict], res_dir: Path, out_dir: Path) -> None:
    """Each job: tag, part, model (key), spec (experiment, fit tag, claw_thr, k, gain), gain (the gain to
    build the base with; None for 1), wiring (matrix or None), wiring_tag, shuffle (group label), seed."""
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    from E1_apl_knockout import bisect_threshold
    from E3_drive_dependent_gain import per_odor_extras, rate
    from E4_substrate import score, kc_by_glomerulus
    from E10_joint_fit import TAU, base_kws, bouton_base, breadth_weights, make_loss
    from E12_breadth_apl import density_excess
    from E16_claw_saturation import spearman_with_refs
    from E18_bouton_depletion import all_refs
    from E26_claw_power import sharing_deficit
    from E35_divisive_apl import suppression_stats
    from E44_operating_point import fraction_stats, losses
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs()
    data_mats = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    loss5 = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=0.05)
    loss_ctrl = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=CTRL_FRAC)
    model_prov = H.model_provenance(orn_deltas=orn)
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    done = {r['tag'] for r in rows}
    refs = all_refs()
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame(
        {'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned_path = out_dir / 'tuned_params.json'
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    base_dir = out_dir / 'partC'; base_dir.mkdir(exist_ok=True)
    long_path = res_dir / 'model_pair_distances_long.csv'

    def finish(tag, meta, on_out, off_out, wPNKC):
        row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data_mats, ref)
        on_p, off_p = on[V], off[V]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        row.update(suppression_stats(H.response_fractions(on_out[0]), H.response_fractions(off_out[0]), data_rates))
        row.update(fraction_stats(on_out[0], off_out[0], data_rates))
        row.update(losses(loss5, loss_ctrl, on_out, off_out))
        rows.append(row)
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index)
        fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        pd.concat({(tag, 'on'): on_p, (tag, 'acute_off'): off_p}, names=['tag', 'condition', 'odor_a', 'odor_b']).rename(
            'distance').reset_index().to_csv(long_path, index=False, mode='a', header=not long_path.exists())
        print(f"{tag}: rates on {row['rate_on']:.4f} off {row['rate_off']:.4f}; rho ctrl {row[V + '__on_spearman']:.3f} "
            f"tnt {row[V + '__off_spearman']:.3f}; block {row[V + '__block_fruity_small_iso']:+.3f}; "
            f"struct wt {row[V + '__on_spearman_resid_vs_remap_wt']:+.2f}; vcf on {row[V + '__on_spearman_resid_vs_vcf_hyp']:+.2f} "
            f"off {row[V + '__off_spearman_resid_vs_vcf_hyp']:+.2f}; rate rho {row['spearman_rate_on_vs_data']:.2f}/"
            f"{row['spearman_rate_off_vs_data']:.2f}; loss @ctrl {row['loss_rate_ctrl']:.3f}; thr x{meta['thr_factor_vs_fit']:.3f}; "
            f"{meta['seconds']:.0f}s", flush=True)
        done.add(tag)

    for job in jobs:
        tag = job['tag']
        if tag in done:
            continue
        exp, ftag, thr, k, fit_gain = job['spec']
        p_fit, fr = fitted_params(exp, ftag)
        t0 = time.time()
        try:
            initial, wnorm, W, breadth = bouton_base(orn, base_dir, claw_thr=thr, gain=job['gain'],
                wiring_matrix=job['wiring'], wiring_tag=job['wiring_tag'])
            sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p_fit, **base_kws(thr), _wPNKC=W,
                _wAPLKC=breadth_weights(wnorm['wAPLKC'], breadth, k), _wKCAPL=wnorm['wKCAPL'],
                _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
            t1 = time.time()
            extra = {kk: p_fit[kk] for kk in SCALE_KEYS}
            new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_fit['fixed_thr'], CTRL_FRAC, extra=extra)
            p_on = dict(p_fit, fixed_thr=float(new_thr))
            r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
            for cond, (r_, s_) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
                r_.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet')
                s_.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
            tuned[tag] = dict(p_on, claw_thr=thr, k=k, gain=job['gain'], fit_gain=fit_gain, ref_fit=ftag, ref_experiment=exp,
                wiring=job['shuffle'], wiring_seed=job['seed'], thr_factor_vs_fit=float(thr_f))
            tuned_path.write_text(json.dumps(tuned, indent=1))
            meta = dict(variant='retuned', tag=tag, part=job['part'], model=job['model'], shuffle=job['shuffle'], seed=job['seed'],
                base='bouton', target_rate=CTRL_FRAC, claw_thr=thr, k=k, kc_kc_inh=p_fit['wKCKC_scale'], gain=job['gain'],
                fit_gain=fit_gain, ref_fit=ftag, ref_experiment=exp, ref_fit_loss=float(fr['loss']), thr_factor_vs_fit=float(thr_f),
                n_kcs=len(r_on), n_glomeruli=W.shape[1], rate_on=rate(r_on), rate_off=rate(r_off), seconds=time.time() - t1,
                mean_claws_per_kc=float(kc_by_glomerulus(W).sum(axis='columns').mean()), **p_on)
            meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
            finish(tag, meta, (r_on, s_on), (r_off, s_off), W)
            del sim
        except Exception as e:  # keep going; record the failure
            print(f'{tag} FAILED after {time.time() - t0:.0f}s: {e!r}', flush=True)
            rows.append(dict(variant='retuned', tag=tag, part=job['part'], model=job['model'], shuffle=job['shuffle'], seed=job['seed'], error=repr(e)))
            pd.DataFrame(rows).to_csv(stats_path, index=False)
    prov_path = res_dir / 'provenance.json'
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov['runs'] = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E54_gain_and_lesion.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(), n_jobs=len(jobs), tags=[j['tag'] for j in jobs],
        model_sources={**{m: MODELS[m][:2] for m in MODELS}, **{m: MODELS_NOGAIN[m][:2] for m in MODELS_NOGAIN}},
        ctrl_frac=CTRL_FRAC, kc_kc_tau=TAU, approx_conc_note=H.APPROX_CONC_NOTE, model=model_prov)
    prov_path.write_text(json.dumps(prov, indent=1, default=str) + '\n')


def jobs_for(parts: List[str], seeds: List[int], res_dir: Path, out_dir: Path) -> List[dict]:
    jobs = []
    if 'A' in parts:
        for mkey, spec in MODELS.items():
            jobs.append(dict(tag=f'A_{mkey}_nogain_retuned', part='A', model=mkey, spec=spec, gain=None, wiring=None,
                wiring_tag=None, shuffle='nogain', seed=None))
    if 'B' in parts:
        les = e52_lesions([0, 1, 2])
        for mkey, spec in MODELS_NOGAIN.items():
            for (kind, seed), L in les.items():
                jobs.append(dict(tag=f'B_{mkey}_{kind}-seed{seed}_retuned', part='B', model=mkey, spec=spec, gain=None, wiring=L,
                    wiring_tag=f'{kind}-seed{seed}', shuffle=kind, seed=seed))
    for part, kinds in (('C', SPLIT_LESIONS), ('C2', PART_C2)):
        if part not in parts:
            continue
        plan = split_plan(res_dir)
        les = make_split_lesions(base_wiring(), plan, seeds, out_dir, res_dir, kinds=kinds)
        for mkey, spec in MODELS.items():
            for (kind, seed), L in les.items():
                jobs.append(dict(tag=f'{part}_{mkey}_{kind}-seed{seed}_retuned', part=part, model=mkey, spec=spec, gain=spec[4], wiring=L,
                    wiring_tag=f'{kind}-seed{seed}', shuffle=kind, seed=seed))
    return jobs


def summary(res_dir: Path) -> None:
    stats_path = res_dir / 'stats_variants.csv'
    if not stats_path.exists():
        print('no rows yet'); return
    mine = pd.read_csv(stats_path)
    if 'error' in mine.columns:
        mine = mine[~(mine.error.astype(str).str.len() > 3)]
    e52 = pd.read_csv(e52_results() / 'comparison.csv')
    e52 = e52[e52.shuffle.isin(['real', 'lesion_targeted', 'lesion_control'])]
    e52 = e52[(e52.shuffle != 'real') | (e52.variant == 'retuned')]
    comp = pd.concat([e52, mine], ignore_index=True)
    comp['vcf_on_minus_off'] = comp[f'{V}__on_spearman_resid_vs_vcf_hyp'] - comp[f'{V}__off_spearman_resid_vs_vcf_hyp']
    comp['weak_share_off'] = comp.weak_frac_off / comp.rate_off
    # the no-gain fits themselves (E45's rows), as fitted
    fits45 = pd.read_csv(H.shared_dir() / 'results' / 'E45_gain_at_ctrl_frac' / 'fits.csv')
    fitrows = []
    for mkey, (exp, ftag, thr, k, g) in MODELS_NOGAIN.items():
        r = fits45[fits45.tag == ftag]
        if len(r) == 1:
            r = r.iloc[0]
            fitrows.append({'model': mkey, 'shuffle': 'real', 'variant': 'fit', 'part': 'fit', f'{V}__block_fruity_small_iso': r['fit_block'],
                'loss_rate_ctrl': r['loss'], f'{V}__on_spearman': r['fit_on_spearman'], f'{V}__off_spearman': r['fit_off_spearman'],
                'spearman_rate_on_vs_data': r['fit_po_on_spearman'], 'spearman_rate_off_vs_data': r['fit_po_off_spearman'],
                f'{V}__on_spearman_resid_vs_remap_wt': r['fit_wt_dev_on'],
                f'{V}__on_spearman_resid_vs_vcf_hyp': r.get('spike_counts_cosine__on_spearman_resid_vs_vcf_hyp', np.nan),
                f'{V}__off_spearman_resid_vs_vcf_hyp': r['fit_vcf_off'],
                'vcf_on_minus_off': r.get('spike_counts_cosine__on_spearman_resid_vs_vcf_hyp', np.nan) - r['fit_vcf_off']})
    # the re-fit on the lesioned wiring (part D), if in
    fits_path = res_dir / 'fits.csv'
    if fits_path.exists():
        for r in pd.read_csv(fits_path).to_dict('records'):
            fitrows.append({'model': f"thr{r['claw_thr']:g}_gain{r['gain']:g}", 'shuffle': 'refit_' + str(r.get('wiring_tag', 'lesion')),
                'variant': 'fit', 'part': 'D', f'{V}__block_fruity_small_iso': r['fit_block'], 'loss_rate_ctrl': r['loss'],
                f'{V}__on_spearman': r['fit_on_spearman'], f'{V}__off_spearman': r['fit_off_spearman'],
                'spearman_rate_on_vs_data': r['fit_po_on_spearman'], 'spearman_rate_off_vs_data': r['fit_po_off_spearman'],
                f'{V}__on_spearman_resid_vs_remap_wt': r['fit_wt_dev_on'],
                f'{V}__on_spearman_resid_vs_vcf_hyp': r.get('spike_counts_cosine__on_spearman_resid_vs_vcf_hyp', np.nan),
                f'{V}__off_spearman_resid_vs_vcf_hyp': r['fit_vcf_off'],
                'vcf_on_minus_off': r.get('spike_counts_cosine__on_spearman_resid_vs_vcf_hyp', np.nan) - r['fit_vcf_off'],
                'fixed_thr': r.get('fixed_thr'), 'kc_kc_inh': r.get('kc_kc_inh'), 'wAPLKC_scale': r.get('wAPLKC_scale')})
    comp = pd.concat([comp, pd.DataFrame(fitrows)], ignore_index=True)
    comp.to_csv(res_dir / 'comparison.csv', index=False)
    cols = [c for c, *_ in STATS if c in comp.columns]
    grp = comp[comp.shuffle != 'real'].groupby(['model', 'shuffle'])[cols].agg(['mean', 'std', 'count'])
    grp.columns = [f'{c}__{s}' for c, s in grp.columns]
    grp.reset_index().to_csv(res_dir / 'per_class.csv', index=False)
    data = data_values()
    out = []

    def cell(sub, col, fmt):
        v = sub[col].astype(float).dropna()
        if len(v) > 1:
            return f'{format(v.mean(), fmt)} ± {format(v.std(ddof=1), fmt.lstrip("+"))}'
        return format(float(v.iloc[0]), fmt) if len(v) == 1 else '-'

    def table(title, head1, head2, columns, model_rows):
        rows = [title, '', head1, head2, '|---|---|' + '---|' * len(columns)]
        for col, label, fmt, dkey in STATS:
            if col not in comp.columns:
                continue
            d = '-'
            if dkey and dkey in data:
                d = format(data[dkey], fmt)
                if 'ceiling' in dkey:
                    d = f'ceiling {d}'
            cells = [cell(sub, col, fmt) for sub in columns]
            rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(rows))

    # A: the gain's own share
    cols_a = []; heads = []
    for mkey in MODELS:
        sub = comp[comp.model == mkey]
        cols_a += [sub[(sub.shuffle == 'real')], sub[sub.shuffle == 'nogain']]
        heads += [f'{MODEL_LABEL[mkey]}, as fitted', f'{MODEL_LABEL[mkey]}, gain set to 1']
    for mkey in MODELS_NOGAIN:
        cols_a.append(comp[(comp.model == mkey) & (comp.shuffle == 'real')]); heads.append(f'{MODEL_LABEL[mkey]}, fitted without gain')
    if any(len(c) for c in cols_a):
        table('**A. The gain\'s own share: the reference models at their fitted scales with the eight glomeruli\'s gain set to 1 '
            '(threshold re-tuned to 6.8 %), beside the fits made without gain**',
            '| | | ' + ' | '.join(heads) + ' |', '| statistic | KC data | ' + ' | '.join('' for _ in heads) + ' |', cols_a, None)
    # B: E52's lesions under the no-gain fits
    for mkey in MODELS_NOGAIN:
        sub = comp[comp.model == mkey]
        if sub[sub.shuffle.isin(['lesion_targeted', 'lesion_control'])].empty:
            continue
        groups = ['lesion_targeted', 'lesion_control']
        table(f'**B. The {MODEL_LABEL[mkey]} fit at its fitted scales on E52\'s lesions (threshold re-tuned)**',
            '| | | real wiring (the fit) | ' + ' | '.join(f'{GROUP_LABEL[g]} (n = {int((sub.shuffle == g).sum())})' for g in groups) + ' |',
            '| statistic | KC data | | ' + ' | '.join('mean ± sd' for _ in groups) + ' |',
            [sub[sub.shuffle == 'real']] + [sub[sub.shuffle == g] for g in groups], None)
    # C: the split lesions under the gained models
    for mkey in MODELS:
        sub = comp[comp.model == mkey]
        if sub[sub.shuffle.isin(ALL_LESIONS)].empty:
            continue
        groups = [g for g in list(ALL_LESIONS) + ['lesion_targeted', 'lesion_control'] if (sub.shuffle == g).any()]
        table(f'**C. The {MODEL_LABEL[mkey]} at its fitted scales on the split lesions (threshold re-tuned)**',
            '| | | real wiring | ' + ' | '.join(f'{GROUP_LABEL[g]} (n = {int((sub.shuffle == g).sum())})' for g in groups) + ' |',
            '| statistic | KC data | threshold re-tuned | ' + ' | '.join('mean ± sd' for _ in groups) + ' |',
            [sub[(sub.shuffle == 'real')]] + [sub[sub.shuffle == g] for g in groups], None)
    # D: the re-fit
    refit = comp[comp.shuffle.astype(str).str.startswith('refit_')]
    if not refit.empty:
        sub = comp[comp.model == 'thr40_gain1.5']
        table('**D. The eight at 1.5, 40 Hz re-fitted on the first targeted-lesion draw, beside the fixed-scale run on the same wiring '
            'and the real-wiring fit**',
            '| | | real wiring, fitted | the lesioned wiring, fixed scales (3 draws) | the lesioned wiring, re-fitted |',
            '| statistic | KC data | | mean ± sd | |',
            [sub[sub.shuffle == 'real'], sub[sub.shuffle == 'lesion_targeted'], refit], None)
    oc_path = res_dir / 'lesion_outcome.csv'
    if oc_path.exists():
        oc = pd.read_csv(oc_path)
        rows = ['**The split lesions (per seed)**', '',
            '| lesion | seed | pairs | KCs removed from the listed pairs (planned) | claw swaps | claws moved (fraction) | co-input correlation with the real wiring | over-shared pairs\' shared KCs, before -> after | protected pairs changed by |',
            '|---|---|---|---|---|---|---|---|---|']
        for r in oc.itertuples():
            rows.append(f'| {GROUP_LABEL[r.kind]} | {r.seed} | {r.n_pairs} | {r.sum_removed_listed} ({r.sum_planned}) | {r.n_swaps} | '
                f'{r.frac_claws_changed:.3f} | {r.coinput_corr:.3f} | {r.over_shared_total_before} -> {r.over_shared_total_after} | {r.protected_changed} |')
        out.append('\n'.join(rows))
    (res_dir / 'comparison_table.md').write_text('\n\n'.join(out) + '\n')
    figure(comp, data, res_dir)
    print('\n\n'.join(out))


def figure(comp: pd.DataFrame, data: Dict[str, float], res_dir: Path) -> None:
    plt = H._plt()
    from matplotlib.lines import Line2D
    panels = [(f'{V}__off_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL off', 'tnt_vcf'),
        (f'{V}__on_spearman_resid_vs_vcf_hyp', 'natural-source alignment, APL on', 'ctrl_vcf'),
        (f'{V}__block_fruity_small_iso', 'fruity × small-ester block', 'block'),
        ('spearman_rate_on_vs_data', 'response rate ρ, APL on', 'ctrl_rate_rho_ceiling'),
        ('loss_rate_ctrl', 'extended loss at 6.8 % (lower is better)', None)]
    groups = ['real', 'nogain', 'lesion_targeted', 'lesion_control'] + list(ALL_LESIONS)
    groups = [g for g in groups if (comp.shuffle == g).any()]
    models = [m for m in list(MODELS) + list(MODELS_NOGAIN) if (comp.model == m).any()]
    colors = {'thr70_gain1.5': '#1f5fbf', 'thr40_gain1.5': '#d1662a', 'thr70_gain1.25': '#2a9d5c', 'thr40_nogain': '#8e44ad', 'thr70_nogain': '#7f8c8d'}
    offs = {m: (i - (len(models) - 1) / 2) * 0.15 for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 7))
    for ax, (col, label, dkey) in zip(axes, panels):
        for m in models:
            sub = comp[comp.model == m]
            for i, g in enumerate(groups):
                v = sub[sub.shuffle == g][col].astype(float).dropna()
                if v.empty:
                    continue
                x = i + offs[m]
                ax.scatter(np.full(len(v), x) + np.linspace(-0.03, 0.03, len(v)), v, s=14, color=colors[m], alpha=0.6, zorder=3)
                if len(v) > 1:
                    ax.errorbar([x], [v.mean()], yerr=[v.std(ddof=1)], fmt='_', color=colors[m], ms=12, mew=2, capsize=3, zorder=4)
                else:
                    ax.scatter([x], v, marker='_', s=150, color=colors[m], linewidths=2, zorder=4)
        if dkey and dkey in data:
            ax.axhline(data[dkey], color='k', ls='--', lw=1)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([GROUP_LABEL[g] for g in groups], rotation=35, ha='right', fontsize=7)
        ax.set_title(label, fontsize=9); ax.grid(axis='y', alpha=0.25)
    handles = [Line2D([], [], marker='o', ls='', color=colors[m], label=MODEL_LABEL[m]) for m in models]
    handles.append(Line2D([], [], ls='--', color='k', label='KC data (target or ceiling)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle('The gain and the wiring behind the natural-source alignment: the models at fixed scales (threshold re-tuned to 6.8 %) '
        'with the gain set to 1, on E52\'s lesions (the no-gain fits) and on the split lesions (points: draws; bar: mean ± sd)', fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(res_dir / f'{EXPERIMENT}.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--parts', nargs='+', choices=['A', 'B', 'C', 'C2'], default=['A', 'B', 'C', 'C2'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1], help='draws of the split lesions (part C)')
    ap.add_argument('--plan', action='store_true', help='only build the split lesions and their tables')
    ap.add_argument('--summary', action='store_true', help='only (re)write the comparison tables and the figure')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    if args.summary:
        summary(res_dir); return
    if args.plan:
        plan = split_plan(res_dir)
        pd.set_option('display.width', 250)
        print(plan.groupby('kind')[['n_removed', 'observed']].agg(['sum', 'count']))
        make_split_lesions(base_wiring(), plan, args.seeds, out_dir, res_dir, kinds=ALL_LESIONS)
        print(pd.read_csv(res_dir / 'lesion_outcome.csv').drop(columns=['reached']).to_string(index=False))
        return
    jobs = jobs_for(args.parts, args.seeds, res_dir, out_dir)
    print(f'{len(jobs)} variants', flush=True)
    run_variants(jobs, res_dir, out_dir)
    summary(res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
