#!/usr/bin/env python3
"""E64: the weak-input odors' ORN input as a bounded oracle: how much of the gap in rho on odor-odor
distances is the weak odors' pattern across glomeruli.
The gap decomposition (E61) put 84 to 87 % of the reference model's disagreement with the two KC matrices
on the weak-input odors' pairs (paa, B-myr and 2-but above all, which the model collapses onto the same KCs
while the flies hold them apart; the ORN input mis-orders the same pairs by about half as much), and the
per-odor rate oracle (E62) found that the odors' recruitment level moves the agreement by at most a few
hundredths. So the question is each weak odor's pattern across glomeruli, not its magnitude: here, on the
reference model (the eight glomeruli at 1.5 at 40 Hz, E44's refit at 6.8 %) at its fitted scales and
threshold, each searched odor's per-glomerulus ORN response is multiplied by a factor vector bounded in
[0.5, 2] (the lab's value halved to doubled; glomeruli without a response stay without), and the vector is
searched by a (1 + POP) evolution strategy in log-factor space to maximize the sum of the two rho's on
odor-odor distances (the model's APL-on distances against the control flies, APL-off against APL output
blocked (TNT)), with a penalty when the odor's response fraction leaves +-15 % of the KC data's in either
condition. Odors are simulated independently at fixed parameters, so POP candidate vectors are stacked as
POP pseudo-odors in one simulation of both conditions, and each candidate's KC response replaces the odor's
column in the incumbent's KC matrices for the scoring. The weak odors are searched in turn, in two passes
(each odor sees the others' updated input); two strong odors (2h, 6al) are searched once from the untuned
model as controls for what a free input buys an odor the model already gets right.
Outputs: <shared>/results/E64_input_oracle/ (search_history.csv: every generation; factors.csv: the searched
factor and the ORN input before and after, per odor and glomerulus, with Hallem's receptor deltas where
Hallem has the odor; per_odor_gains.csv: the two rho's and the odor's fractions before and after each odor's
search; stats_variants.csv with the harness scoring columns for the untuned model, each pass's result and
the controls; per_odor_response_fractions.csv; model_pair_distances_long.csv; comparison_table.md;
E64_input_oracle.png; provenance.json) and ./outputs/E64_input_oracle/ (responses / spike counts of the
scored variants, the bouton base).
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E64_input_oracle.py [--pop 32] [--generations 10 6]
           [--control-generations 8] [--odors ...] [--pass2-odors paa B-myr 2-but] [--controls ...] [--smoke]
       (a generation is one simulation of both conditions with --pop pseudo-odors: about 1.5 to 2 min at 16 threads,
       most of it the simulator's rebuild for the new input; the first pass searches every weak odor, later passes
       only --pass2-odors)
       ./venv/bin/python E64_input_oracle.py --summary
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E47_wiring_features import MODELS, STATS, V, data_values, fitted_params
from E54_gain_and_lesion import MODEL_LABEL

EXPERIMENT = 'E64_input_oracle'
MODEL = 'thr40_gain1.5'
SEARCH_ODORS = ['paa', 'B-myr', '2-but', 'euc', '-aPine', 'ms', '1-8ol']
CONTROL_ODORS = ['2h', '6al']
BOUNDS = (0.5, 2.0)
RATE_TOL, PENALTY = 0.15, 10.0
SIGMA0, SIGMA_MIN, SIGMA_MAX = 0.3, 0.03, 1.0
RHO_ON, RHO_OFF = f'{V}__on_spearman', f'{V}__off_spearman'


def odor_abbrevs(orn: pd.DataFrame) -> List[str]:
    return [H.odor_abbrev(c) for c in orn.columns]


def candidate_orn(orn: pd.DataFrame, odor: str, factors: np.ndarray) -> pd.DataFrame:
    """The ORN input with `factors.shape[0]` pseudo-odors, each the searched odor's column times one factor
    vector (glomeruli in `orn`'s order); the first keeps the odor's own column label, the others get
    unique labels with the same concentration (`<abbrev>_c<j> @ <conc>`)."""
    ab = odor_abbrevs(orn)
    j = ab.index(odor)
    panel, label = orn.columns[j]
    name, conc = label.split(' @ ')
    base = orn.iloc[:, j].to_numpy(dtype=float)
    cols = pd.MultiIndex.from_tuples([(panel, label)] + [(panel, f'{name}_c{i} @ {conc}') for i in range(1, len(factors))],
        names=orn.columns.names)
    return pd.DataFrame((base[:, None] * factors.T), index=orn.index, columns=cols)


def out_col(df: pd.DataFrame, odor: str):
    """The column of a model output (responses / spike counts) that is this odor (by abbreviation)."""
    cols = [c for c in df.columns if H.odor_abbrev(c) == odor]
    assert len(cols) == 1, (odor, cols)
    return cols[0]


def pair_rhos(s_on: pd.DataFrame, s_off: pd.DataFrame, ctrl: pd.Series, tnt: pd.Series) -> Tuple[float, float]:
    return (H.spearman(H.pair_distances(s_on), ctrl), H.spearman(H.pair_distances(s_off), tnt))


def rate_penalty(frac_on: float, frac_off: float, d_on: float, d_off: float) -> float:
    pen = 0.0
    for f, d in ((frac_on, d_on), (frac_off, d_off)):
        pen += PENALTY * max(0.0, abs(f / d - 1.0) - RATE_TOL)
    return pen


def run(args, res_dir: Path, out_dir: Path) -> None:
    from al_analysis.mb_model import make_fixed_param_kc_simulator
    from E3_drive_dependent_gain import per_odor_extras, rate
    from E4_substrate import score, kc_by_glomerulus
    from E10_joint_fit import TAU, base_kws, bouton_base, breadth_weights, make_loss
    from E12_breadth_apl import density_excess
    from E16_claw_saturation import spearman_with_refs
    from E18_bouton_depletion import all_refs
    from E26_claw_power import sharing_deficit
    from E35_divisive_apl import suppression_stats
    from E44_operating_point import fraction_stats, losses, CTRL_FRAC
    rng = np.random.default_rng(args.seed)
    orn = H.apl16_orn_deltas()
    ab = odor_abbrevs(orn)
    orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs()
    data_mats = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    ctrl, tnt = data_mats['raw']['TNT3c'], data_mats['raw']['TNT']
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    loss5 = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=0.05)
    loss_ctrl = make_loss(data_mats, data_rates, orn_pairs, drive, ref, structure=True, rate_target=CTRL_FRAC)
    model_prov = H.model_provenance(orn_deltas=orn)
    refs = all_refs()
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    exp, ftag, thr, k, gain = MODELS[MODEL]
    p_fit, fr = fitted_params(exp, ftag)
    base_dir = out_dir / 'bases'; base_dir.mkdir(exist_ok=True)
    initial, wnorm, W, breadth = bouton_base(orn, base_dir, claw_thr=thr, gain=gain)
    wAPLKC = breadth_weights(wnorm['wAPLKC'], breadth, k)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    long_path = res_dir / 'model_pair_distances_long.csv'
    rows: List[dict] = []
    fracs = pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    hist: List[dict] = []; gains: List[dict] = []
    n_sims = [0]

    def simulate(orn_in: pd.DataFrame):
        sim, _ = make_fixed_param_kc_simulator(orn_in, initial_params=p_fit, **base_kws(thr), _wPNKC=W, _wAPLKC=wAPLKC,
            _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'], kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
        r_on, s_on = sim(dict(p_fit))
        r_off, s_off = sim(dict(p_fit, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
        del sim
        n_sims[0] += 1
        return (r_on, s_on), (r_off, s_off)

    def finish(tag: str, variant: str, on_out, off_out, factors: Dict[str, np.ndarray], note: str) -> dict:
        meta = dict(variant=variant, tag=tag, part='I', model=MODEL, shuffle='real', seed=args.seed, base='bouton', claw_thr=thr, k=k,
            kc_kc_inh=p_fit['wKCKC_scale'], gain=gain, ref_fit=ftag, ref_experiment=exp, ref_fit_loss=float(fr['loss']), thr_factor_vs_fit=1.0,
            n_kcs=len(on_out[0]), n_glomeruli=W.shape[1], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), n_sims=n_sims[0],
            searched_odors=' '.join(sorted(factors)), note=note, mean_claws_per_kc=float(kc_by_glomerulus(W).sum(axis='columns').mean()), **p_fit)
        meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
        row, on, off = score(meta, on_out, off_out, W, orn, orn_pairs, drive, data_mats, ref)
        on_p, off_p = on[V], off[V]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        row.update(suppression_stats(H.response_fractions(on_out[0]), H.response_fractions(off_out[0]), data_rates))
        row.update(fraction_stats(on_out[0], off_out[0], data_rates))
        row.update(losses(loss5, loss_ctrl, on_out, off_out))
        rows.append(row); pd.DataFrame(rows).to_csv(stats_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index)
        fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        pd.concat({(tag, 'on'): on_p, (tag, 'acute_off'): off_p}, names=['tag', 'condition', 'odor_a', 'odor_b']).rename(
            'distance').reset_index().to_csv(long_path, index=False, mode='a', header=not long_path.exists())
        for cond, (r_, s_) in (('on', on_out), ('acute_off', off_out)):
            r_.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet'); s_.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        print(f"{tag}: rates on {row['rate_on']:.4f} off {row['rate_off']:.4f}; rho ctrl {row[RHO_ON]:.3f} tnt {row[RHO_OFF]:.3f}; "
            f"block {row[V + '__block_fruity_small_iso']:+.3f}; vcf on {row[V + '__on_spearman_resid_vs_vcf_hyp']:+.2f} "
            f"off {row[V + '__off_spearman_resid_vs_vcf_hyp']:+.2f}; rate rho {row['spearman_rate_on_vs_data']:.2f}/"
            f"{row['spearman_rate_off_vs_data']:.2f}; loss @ctrl {row['loss_rate_ctrl']:.3f}", flush=True)
        return row

    if long_path.exists():
        long_path.unlink()
    # the untuned model: the incumbent to start from
    on0, off0 = simulate(orn)
    finish('I_untuned', 'untuned', on0, off0, {}, 'the fitted model, the lab\'s ORN input')
    glomeruli = list(orn.index)

    def search(odor: str, inc_on, inc_off, orn_now: pd.DataFrame, factors0: np.ndarray, generations: int, label: str, pass_no: int):
        """Searches one odor's factor vector from `factors0` on the incumbent (inc_on, inc_off at orn_now);
        returns (factors, orn with the odor's column updated, incumbent outputs with the odor's column replaced)."""
        j = ab.index(odor)
        col = orn.columns[j]
        d_on, d_off = float(data_rates['TNT3c'][odor]), float(data_rates['TNT'][odor])
        r_on, s_on = inc_on; r_off, s_off = inc_off
        oc = out_col(s_on, odor)  # the outputs' column label for this odor (the same in all four outputs)
        rho_on0, rho_off0 = pair_rhos(s_on, s_off, ctrl, tnt)
        f_on0, f_off0 = float(r_on[oc].astype(float).mean()), float(r_off[oc].astype(float).mean())
        parent = factors0.copy(); parent_obj = rho_on0 + rho_off0 - rate_penalty(f_on0, f_off0, d_on, d_off)
        best_state = (rho_on0, rho_off0, f_on0, f_off0)
        sigma = SIGMA0
        lo, hi = np.log(BOUNDS[0]), np.log(BOUNDS[1])
        for g in range(generations):
            t0 = time.time()
            logf = np.log(parent)[None, :] + sigma * rng.standard_normal((args.pop, len(glomeruli)))
            offspring = np.exp(np.clip(logf, lo, hi))
            c_on, c_off = simulate(candidate_orn(orn_now, odor, offspring))
            objs = np.empty(args.pop); parts = []
            for i in range(args.pop):
                s_on_i = s_on.copy(); s_on_i[oc] = c_on[1].iloc[:, i].to_numpy()
                s_off_i = s_off.copy(); s_off_i[oc] = c_off[1].iloc[:, i].to_numpy()
                ro, rf = pair_rhos(s_on_i, s_off_i, ctrl, tnt)
                fo, ff = float(c_on[0].iloc[:, i].astype(float).mean()), float(c_off[0].iloc[:, i].astype(float).mean())
                objs[i] = ro + rf - rate_penalty(fo, ff, d_on, d_off); parts.append((ro, rf, fo, ff))
            b = int(np.argmax(objs)); success = objs[b] > parent_obj
            if success:
                parent = offspring[b]; parent_obj = float(objs[b]); best_state = parts[b]
                # the incumbent takes the winner's KC response for this odor (odors are independent at fixed parameters)
                for df_, c_ in ((r_on, c_on[0]), (s_on, c_on[1]), (r_off, c_off[0]), (s_off, c_off[1])):
                    df_[oc] = c_.iloc[:, b].to_numpy()
            sigma = float(np.clip(sigma * (1.5 if success else 0.8), SIGMA_MIN, SIGMA_MAX))
            hist.append(dict(label=label, pass_no=pass_no, odor=odor, generation=g, best_offspring_obj=float(objs[b]), parent_obj=parent_obj,
                success=bool(success), sigma=sigma, rho_on=best_state[0], rho_off=best_state[1], frac_on=best_state[2], frac_off=best_state[3],
                data_frac_on=d_on, data_frac_off=d_off, seconds=time.time() - t0))
            pd.DataFrame(hist).to_csv(res_dir / 'search_history.csv', index=False)
            print(f'{label} pass {pass_no} {odor} gen {g}: best {objs[b]:+.4f} (parent {parent_obj:+.4f}), rho on {best_state[0]:.3f} off '
                f'{best_state[1]:.3f}, frac on {best_state[2]:.3f}/{d_on:.3f} off {best_state[3]:.3f}/{d_off:.3f}, sigma {sigma:.2f}, '
                f'{time.time() - t0:.0f}s', flush=True)
        gains.append(dict(label=label, pass_no=pass_no, odor=odor, rho_on_before=rho_on0, rho_off_before=rho_off0, rho_on_after=best_state[0],
            rho_off_after=best_state[1], frac_on_before=f_on0, frac_off_before=f_off0, frac_on_after=best_state[2], frac_off_after=best_state[3],
            data_frac_on=d_on, data_frac_off=d_off, generations=generations, log2_factor_rms=float(np.sqrt(np.mean(np.log2(parent) ** 2))),
            n_glomeruli_moved_2x=int((np.abs(np.log2(parent)) > 0.5).sum())))
        pd.DataFrame(gains).to_csv(res_dir / 'per_odor_gains.csv', index=False)
        orn_new = orn_now.copy(); orn_new[col] = orn.iloc[:, j].to_numpy(dtype=float) * parent
        return parent, orn_new, (r_on, s_on), (r_off, s_off)

    def record_factors(factors: Dict[str, np.ndarray], label: str):
        try:
            from E38_hallem_input import hallem_deltas
            hal = hallem_deltas(glomeruli)
        except Exception:
            hal = None
        out = []
        for odor, f in factors.items():
            j = ab.index(odor)
            base = orn.iloc[:, j].to_numpy(dtype=float)
            for gi, glom in enumerate(glomeruli):
                hv = np.nan
                if hal is not None:
                    hc = [c for c in hal.columns if H.odor_abbrev(c) == odor]
                    if hc and glom in hal.index:
                        hv = float(hal.loc[glom, hc[0]])
                out.append(dict(label=label, odor=odor, glomerulus=glom, orn_delta=float(base[gi]), factor=float(f[gi]),
                    log2_factor=float(np.log2(f[gi])), searched_delta=float(base[gi] * f[gi]), hallem_delta=hv))
        pd.DataFrame(out).to_csv(res_dir / f'factors_{label}.csv', index=False)

    def copy_outputs(on_out, off_out):
        return (on_out[0].copy(), on_out[1].copy()), (off_out[0].copy(), off_out[1].copy())

    odors = args.odors; controls = args.controls
    ones = np.ones(len(glomeruli))
    # the weak odors, two passes
    factors: Dict[str, np.ndarray] = {o: ones.copy() for o in odors}
    orn_now = orn.copy(); inc_on, inc_off = copy_outputs(on0, off0)
    for pass_no, gens in enumerate(args.generations, 1):
        for odor in (odors if pass_no == 1 else [o for o in args.pass2_odors if o in odors]):
            factors[odor], orn_now, inc_on, inc_off = search(odor, inc_on, inc_off, orn_now, factors[odor], gens, 'weak', pass_no)
        on_p, off_p = simulate(orn_now)  # the incumbent re-simulated whole, for the record
        finish(f'I_searched_pass{pass_no}', f'searched_pass{pass_no}', on_p, off_p, dict(factors), f'the weak odors\' input searched, pass {pass_no}')
        inc_on, inc_off = copy_outputs(on_p, off_p)
        record_factors(factors, f'weak_pass{pass_no}')
        pd.DataFrame(orn_now.to_numpy(), index=orn_now.index, columns=orn_now.columns).to_csv(res_dir / f'orn_input_searched_pass{pass_no}.csv')
    # the controls, one pass each from the untuned model
    cf: Dict[str, np.ndarray] = {}
    orn_c = orn.copy(); c_on, c_off = copy_outputs(on0, off0)
    for odor in controls:
        cf[odor], orn_c, c_on, c_off = search(odor, c_on, c_off, orn_c, ones.copy(), args.control_generations, 'control', 1)
    if controls:
        on_c, off_c = simulate(orn_c)
        finish('I_controls', 'controls', on_c, off_c, cf, 'the control odors\' input searched from the untuned model')
        record_factors(cf, 'controls')
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E64_input_oracle.py', model=MODEL,
        search_odors=odors, control_odors=controls, pop=args.pop, generations=list(args.generations), control_generations=args.control_generations,
        bounds=list(BOUNDS), rate_tol=RATE_TOL, penalty=PENALTY, sigma0=SIGMA0, seed=args.seed, n_simulations=n_sims[0], model_provenance=model_prov)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str))


def summary(res_dir: Path) -> None:
    st = pd.read_csv(res_dir / 'stats_variants.csv')
    gains = pd.read_csv(res_dir / 'per_odor_gains.csv')
    data = data_values()
    out = []
    variants = [v for v in ('untuned', 'searched_pass1', 'searched_pass2', 'controls') if (st.variant == v).any()]
    vlabel = {'untuned': 'the fitted model, the lab\'s ORN input', 'searched_pass1': 'the weak odors\' input searched, pass 1',
        'searched_pass2': 'the weak odors\' input searched, pass 2', 'controls': 'the control odors (2h, 6al) searched instead'}
    rows = [f'**{MODEL_LABEL[MODEL]} at its fitted scales on the real wiring: the lab\'s ORN input, the weak odors\' input searched within a '
        'factor of two of it (the per-glomerulus pattern, the response fraction held), and the two control odors searched instead**', '',
        '| statistic | KC data | ' + ' | '.join(vlabel[v] for v in variants) + ' |', '|---|---|' + '---|' * len(variants)]
    for col, label, fmt, dkey in STATS:
        if col not in st.columns:
            continue
        d = '-'
        if dkey and dkey in data:
            d = format(data[dkey], fmt)
            if 'ceiling' in dkey:
                d = f'ceiling {d}'
        cells = []
        for v in variants:
            x = st[st.variant == v][col]
            cells.append(format(float(x.iloc[0]), fmt) if len(x) == 1 and pd.notna(x.iloc[0]) else '-')
        rows.append(f'| {label} | {d} | ' + ' | '.join(cells) + ' |')
    out.append('\n'.join(rows))
    rows = ['**Each odor\'s search: ρ on odor-odor distances (APL on / APL off) before and after, the odor\'s response fractions after '
        '(data in brackets), and how far its input moved (rms of log2 factor over glomeruli; glomeruli moved more than twofold)**', '',
        '| | | ρ on odor-odor distances, APL on {span=2} | ρ on odor-odor distances, APL off {span=2} | response fraction after (KC data) {span=2} | | |',
        '| odor | pass | before | after | before | after | APL on | APL off | rms log2 factor | glomeruli > 2x |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in gains.itertuples():
        lab = f'{r.odor}' + (' (control)' if r.label == 'control' else '')
        rows.append(f'| {lab} | {r.pass_no} | {r.rho_on_before:.3f} | {r.rho_on_after:.3f} | {r.rho_off_before:.3f} | {r.rho_off_after:.3f} | '
            f'{r.frac_on_after:.3f} ({r.data_frac_on:.3f}) | {r.frac_off_after:.3f} ({r.data_frac_off:.3f}) | {r.log2_factor_rms:.2f} | '
            f'{r.n_glomeruli_moved_2x} |')
    out.append('\n'.join(rows))
    fpath = res_dir / 'factors_weak_pass2.csv'
    if not fpath.exists():
        fpath = res_dir / 'factors_weak_pass1.csv'
    if fpath.exists():
        fac = pd.read_csv(fpath)
        rows = [f'**The searched input ({fpath.stem.replace("factors_", "")}): per odor, the glomeruli moved most (log2 of the factor on the '
            'lab\'s ORN response; the response before and after, in spike-rate deltas; Hallem\'s receptor delta where Hallem has the odor)**', '',
            '| odor | glomerulus | ORN delta (lab) | factor | searched delta | Hallem delta |', '|---|---|---|---|---|---|']
        for odor, g in fac.groupby('odor', sort=False):
            g = g.assign(a=g.log2_factor.abs()).sort_values('a', ascending=False)
            g = g[(g.orn_delta.abs() > 1) & (g.a > 0.3)].head(8)
            for r in g.itertuples():
                hv = '-' if pd.isna(r.hallem_delta) else f'{r.hallem_delta:.0f}'
                rows.append(f'| {odor} | {r.glomerulus} | {r.orn_delta:.0f} | {r.factor:.2f} | {r.searched_delta:.0f} | {hv} |')
        out.append('\n'.join(rows))
    (res_dir / 'comparison_table.md').write_text('\n\n'.join(out) + '\n')
    figure(st, gains, res_dir)
    print('\n\n'.join(out))


def figure(st: pd.DataFrame, gains: pd.DataFrame, res_dir: Path) -> None:
    plt = H._plt()
    data = data_values()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax = axes[0]
    variants = [v for v in ('untuned', 'searched_pass1', 'searched_pass2', 'controls') if (st.variant == v).any()]
    x = np.arange(len(variants))
    for i, (col, lab, dkey) in enumerate([(RHO_ON, 'ρ on odor-odor distances, APL on', 'ctrl_ceiling'), (RHO_OFF, 'ρ on odor-odor distances, APL off', 'tnt_ceiling')]):
        vals = [float(st[st.variant == v][col].iloc[0]) for v in variants]
        ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=20, ha='right', fontsize=8); ax.set_ylim(0.5, 1.0)
    ax.axhline(0.65, color='0.5', ls=':', lw=1); ax.axhline(0.74, color='0.5', ls='--', lw=1)
    ax.set_title('the two agreements (dotted: the ORN input\'s own with the control flies, 0.65; dashed: with APL output blocked (TNT), 0.74)', fontsize=8)
    ax.legend(fontsize=8, frameon=False); ax.grid(axis='y', alpha=0.25)
    ax = axes[1]
    g = gains.copy(); g['lab'] = g.odor + np.where(g.label == 'control', ' (ctrl)', '') + ' p' + g.pass_no.astype(str)
    xs = np.arange(len(g))
    ax.bar(xs - 0.2, g.rho_on_after - g.rho_on_before, width=0.38, label='APL on')
    ax.bar(xs + 0.2, g.rho_off_after - g.rho_off_before, width=0.38, label='APL off')
    ax.set_xticks(xs); ax.set_xticklabels(g.lab, rotation=60, ha='right', fontsize=7); ax.axhline(0, color='k', lw=0.8)
    ax.set_title('gain in ρ on odor-odor distances from each odor\'s search', fontsize=9); ax.legend(fontsize=8, frameon=False); ax.grid(axis='y', alpha=0.25)
    ax = axes[2]
    fpath = res_dir / 'factors_weak_pass2.csv'
    if not fpath.exists():
        fpath = res_dir / 'factors_weak_pass1.csv'
    if fpath.exists():
        fac = pd.read_csv(fpath)
        mat = fac.pivot(index='odor', columns='glomerulus', values='log2_factor')
        base = fac.pivot(index='odor', columns='glomerulus', values='orn_delta')
        mat = mat.where(base.abs() > 1)
        im = ax.imshow(mat.to_numpy(dtype=float), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=8)
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.03, label='log2 factor on the lab\'s ORN response (glomeruli the odor drives)')
        ax.set_title('the searched input, per glomerulus', fontsize=9)
    fig.suptitle('The weak-input odors\' ORN input as a bounded oracle: the reference model at its fitted scales', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(res_dir / f'{EXPERIMENT}.png', dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--pop', type=int, default=48, help='candidates per generation (pseudo-odors per simulation)')
    ap.add_argument('--generations', nargs='+', type=int, default=[12, 8], help='generations per odor in each pass')
    ap.add_argument('--control-generations', type=int, default=10)
    ap.add_argument('--odors', nargs='+', default=SEARCH_ODORS); ap.add_argument('--controls', nargs='*', default=CONTROL_ODORS)
    ap.add_argument('--pass2-odors', nargs='*', default=['paa', 'B-myr', '2-but'],
        help='the odors searched again in the second and later passes (default: the three that carry most of the gap, E61)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='one odor, one generation, four candidates, no controls, into a scratch folder')
    ap.add_argument('--summary', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.pop, args.generations, args.controls, args.odors = 4, [1], [], ['paa']
    exp = EXPERIMENT + ('_smoke' if args.smoke else '')
    res_dir = H.results_dir(exp); out_dir = H.outputs_dir(exp)
    if args.summary:
        summary(res_dir); return
    run(args, res_dir, out_dir)
    summary(res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
