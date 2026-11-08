#!/usr/bin/env python3
"""E38: Hallem's receptor data as the ORN input for the weak-input odors (ms, paa, B-myr, 1-8ol).

The APL-blocked sharing diagnostic (E21) traced the models' too-low sharing among the non-ester
pairs with APL output blocked, the residual disagreement with the TNT matrix and most of the
per-odor rate deficit to five weak-input odors, and its addendum found that the lab's ORN input
and Hallem & Carlson's receptor data agree for the esters and aldehydes and disagree for exactly
these odors (ms is DL1 at 258 Hz in Hallem against 70 Hz in the lab's data; paa and 1-8ol peak in
different glomeruli; B-myr is weak in both; euc is not in Hallem's set). This runs the model on the
alternative input that exists, as a bound on the input uncertainty and as a direct test of
single-glomerulus recruitment (Hallem's ms is one glomerulus at 258 Hz).

Input variants, built from `H.apl16_orn_deltas()` (the lab's input, 36 glomeruli) for the four
Hallem odors, on the 16 glomeruli Hallem's 24 receptors and the model share (drosolf
`orns.orns(add_sfr=False)`, receptor -> glomerulus as `mb_model` maps them, DM3+DM5 dropped):
  lab                 the lab's input (reference; the E16 / E27 threshold model re-run here)
  hallem_shared       the 16 shared glomeruli take Hallem's spike deltas, the other 20 keep the lab's
  hallem_only         Hallem's on the 16, zero on the other 20
  max                 the elementwise maximum of the lab's and Hallem's on the 16, the lab's elsewhere
  hallem_shared_<o>   hallem_shared for one odor at a time
  lab_scaled          the lab's row of each odor scaled so that its maximum over the 16 shared
                      glomeruli equals Hallem's there (the lab's profile at Hallem's level: the
                      control that separates profile from level; the factor is < 1 for B-myr)
  lab_scaled_ms       lab_scaled for ms only
Hallem's odors are at 10^-2 where Yang's ms, paa and 1-8ol are at 10^-3 (B-myr at 10^-2): within
Tom's one-log-unit rule but at its edge and on the high side, so these inputs are upper bounds.

Part A: each variant on the bouton base with the 70 Hz per-claw threshold, tuned at 5 % with APL
on, acute APL off (E24 / E27's pipeline, `tune_and_knockout`). Part B: the reference model (E34's
thr40_gain1.5_k2_cap3: 40 Hz, breadth 2, gain 1.5 on the oracle glomeruli, the fitted APL->KC,
APL->bouton and KC-KC scales held; also the 70 Hz capped fit) on a subset of the variants, the
breadth weights recomputed from the variant's own APL-off run as the fit's pipeline does
(`E10_joint_fit.bouton_base`) and the somatic threshold re-tuned to the fit's APL-on response rate.
Scored as E27 (E13 statistics, E7 excess by block, per-odor response fractions, E16 reference
ranks, E21 / E26's APL-off sharing deficit by pair type) plus E35's suppression uniformity and a
responder analysis for the four odors (breadth and exclusivity of their responders, claws from
the candidate glomeruli among them, and the rank agreement of their co-responder profile with
the fly's from E23). All statistics against the data use the lab's ORN reference matrix (the
data's input is what it is; only the model's drive changed).

Outputs: <shared>/results/E38_hallem_input/ and ./outputs/E38_hallem_input/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E38_hallem_input.py [--parts A B] [--variants lab hallem_shared ...]
       ./venv/bin/python E38_hallem_input.py --figure-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sstats

import kc_remapping_harness as H
from E1_apl_knockout import bisect_threshold
from E3_drive_dependent_gain import BOUTON, TARGET, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E10_joint_fit import TAU, base_kws, bouton_base, breadth_weights
from E12_breadth_apl import density_excess
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E26_claw_power import WEAK, deficit_table, sharing_deficit
from E35_divisive_apl import suppression_stats

EXPERIMENT = 'E38_hallem_input'
HALLEM_ODORS: Dict[str, str] = {'ms': 'methyl salicylate', 'paa': 'phenylacetaldehyde',
    'B-myr': 'b-myrcene', '1-8ol': '1-octanol'}
HALLEM_CONC = -2.0
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')
# glomeruli whose claws are counted among each odor's responders: the lab's top glomeruli (E27's
# oracle sets) and Hallem's top shared glomeruli for the same odor
CAND_GLOMS: Dict[str, list] = {'ms': ['DL1', 'VA7m', 'D', 'VA5'], 'paa': ['DM1', 'DM6', 'VA5', 'DL1'],
    'B-myr': ['DM1', 'DM6', 'VM5v', 'DP1m'], '1-8ol': ['DC2', 'DM5', 'VC3', 'DC1']}
# reference fits for part B: tag -> (experiment, per-claw threshold, breadth exponent, gain)
REF_FITS: Dict[str, Tuple[str, float, float, float]] = {
    'thr40_gain1.5_k2_cap3': ('E34_gain_bounded', 40.0, 2.0, 1.5),
    'thr70_gain1.5_k2_cap3': ('E34_gain_bounded', 70.0, 2.0, 1.5),
}
PART_B_DEFAULT = ['lab', 'hallem_shared', 'hallem_only', 'max', 'hallem_shared_ms']


def hallem_deltas(model_glomeruli) -> pd.DataFrame:
    """Hallem & Carlson's spike deltas (10^-2) for the four odors, glomerulus x odor abbreviation,
    restricted to the glomeruli the model's input has (receptor -> glomerulus as drosolf names
    them; the multiglomerular DM3+DM5 receptor dropped, as `mb_model` does)."""
    from drosolf import orns
    d = orns.orns(add_sfr=False, drop_sfr=False, columns='glomerulus')
    d = d.loc[:, [g for g in d.columns if '+' not in g]]
    shared = sorted(set(d.columns) & set(model_glomeruli))
    out = pd.DataFrame({a: d.loc[name, shared].astype(float) for a, name in HALLEM_ODORS.items()})
    out.index.name = 'glomerulus'
    return out


def input_variants(orn0: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, dict]]:
    """{tag: (ORN input, description)} for every variant in the module doc."""
    hal = hallem_deltas(orn0.index)
    shared = list(hal.index)
    col = {H.odor_abbrev(c): c for c in orn0.columns}
    scale = {o: float(hal[o].max() / orn0.loc[shared, col[o]].max()) for o in HALLEM_ODORS}

    def swap(odors, how: str) -> pd.DataFrame:
        orn = orn0.astype(float).copy()
        for o in odors:
            c = col[o]
            lab = orn.loc[shared, c]
            if how == 'hallem_shared':
                orn.loc[shared, c] = hal[o].to_numpy()
            elif how == 'hallem_only':
                orn[c] = 0.0
                orn.loc[shared, c] = hal[o].to_numpy()
            elif how == 'max':
                orn.loc[shared, c] = np.maximum(lab.to_numpy(), hal[o].to_numpy())
            elif how == 'lab_scaled':
                orn[c] = orn[c] * scale[o]
            else:
                raise ValueError(how)
        return orn

    allo = list(HALLEM_ODORS)
    out = {'lab': (orn0.astype(float).copy(), dict(how='lab', odors=[]))}
    for how in ('hallem_shared', 'hallem_only', 'max'):
        out[how] = (swap(allo, how), dict(how=how, odors=allo))
    for o in allo:
        out[f'hallem_shared_{o}'] = (swap([o], 'hallem_shared'), dict(how='hallem_shared', odors=[o]))
    out['lab_scaled'] = (swap(allo, 'lab_scaled'), dict(how='lab_scaled', odors=allo, factors=scale))
    out['lab_scaled_ms'] = (swap(['ms'], 'lab_scaled'), dict(how='lab_scaled', odors=['ms'], factors={'ms': scale['ms']}))
    for tag, (orn, meta) in out.items():
        meta['shared_glomeruli'] = shared
        meta['n_changed_glomeruli'] = int((~np.isclose(orn.to_numpy(), orn0.to_numpy().astype(float))).any(axis=1).sum())
    return out


def _kc_index(df: pd.DataFrame) -> pd.Index:
    return pd.Index(df.index.get_level_values(H.KC_ID_LEVEL) if isinstance(df.index, pd.MultiIndex) else df.index)


def responder_analysis(tag: str, cond: str, resp: pd.DataFrame, wPNKC: pd.DataFrame,
    data_overlap: pd.DataFrame, genotype: str) -> Tuple[list, list]:
    """Per odor (the four Hallem odors plus the other weak odors): fraction responding, mean
    breadth and exclusivity of its responders, the rank agreement of its co-responder profile
    (observed / expected co-responders with each other odor) with the fly's (E23,
    `genotype`), and, per candidate glomerulus, the mean number of claws from it among the
    responders, the fraction of responders with at least one, and the fraction of all KCs with
    at least one (the population reference). Returns (summary rows, claw rows)."""
    R = H.abbrev_columns(resp.astype(float)) > 0
    R.index = _kc_index(R)
    odors = [o for o in H.APL16_ODORS if o in R.columns]
    R = R[odors]
    N = len(R); n = R.sum(); breadth = R.sum(axis=1)
    claws = kc_by_glomerulus(wPNKC).astype(float)
    claws.index = _kc_index(claws)
    claws = claws.T.groupby(level='glomerulus').sum().T if isinstance(claws.columns, pd.MultiIndex) else claws
    claws = claws.reindex(R.index).fillna(0.0)
    d_ov = data_overlap[data_overlap.genotype == genotype]
    rows, claw_rows = [], []
    for o in sorted(set(WEAK) | set(HALLEM_ODORS)):
        if o not in R.columns:
            continue
        r = R[o]
        row = dict(tag=tag, condition=cond, odor=o, frac=float(r.mean()), n_responders=int(r.sum()))
        if r.sum() > 0:
            b = breadth[r]
            row.update(mean_breadth=float(b.mean()), frac_exclusive=float((b == 1).mean()))
            prof = pd.Series({o2: ((r & R[o2]).sum() / (n[o] * n[o2] / N) if n[o2] > 0 else np.nan)
                for o2 in odors if o2 != o})
            dp = d_ov[d_ov.odor == o].set_index('other')['ratio'].reindex(prof.index)
            ok = prof.notna() & dp.notna()
            row['spearman_overlap_vs_data'] = float(sstats.spearmanr(prof[ok], dp[ok])[0]) if ok.sum() >= 4 else np.nan
            row['top_overlap_odors'] = ','.join(prof.sort_values(ascending=False).head(4).index)
        else:
            row.update(mean_breadth=np.nan, frac_exclusive=np.nan, spearman_overlap_vs_data=np.nan, top_overlap_odors='')
        cands = [g for g in CAND_GLOMS.get(o, []) if g in claws.columns]
        if cands:
            tot = claws[cands].sum(axis=1)
            row['frac_responders_single_cand_claw'] = float((tot[r] == 1).mean()) if r.sum() else np.nan
            row['frac_responders_no_cand_claw'] = float((tot[r] == 0).mean()) if r.sum() else np.nan
            row['frac_responders_ge2_cand_claws'] = float((tot[r] >= 2).mean()) if r.sum() else np.nan
            for g in cands:
                claw_rows.append(dict(tag=tag, condition=cond, odor=o, glomerulus=g,
                    mean_claws_in_responders=float(claws.loc[r, g].mean()) if r.sum() else np.nan,
                    frac_responders_with_claw=float((claws.loc[r, g] > 0).mean()) if r.sum() else np.nan,
                    frac_all_kcs_with_claw=float((claws[g] > 0).mean())))
        rows.append(row)
    return rows, claw_rows


def data_responder_rows(genotype: str) -> list:
    st = pd.read_csv(H.shared_dir() / 'results' / 'E23_weak_odor_responders' / 'responder_stats.csv')
    st = st[st.genotype == genotype]
    return [dict(tag='data', condition={'TNT': 'acute_off', 'TNT3c': 'on'}[genotype], odor=r.odor, frac=r.frac,
        mean_breadth=r.mean_breadth, frac_exclusive=r.frac_exclusive) for r in st.itertuples() if r.odor in set(WEAK) | set(HALLEM_ODORS)]


def make_figure(stats: pd.DataFrame, fracs: pd.DataFrame, resp: pd.DataFrame, res_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    stats = stats.copy()
    stats['order'] = [(0 if t.startswith('A_') else 1, i) for i, t in enumerate(stats.tag)]
    stats = stats.sort_values('order')
    tags = list(stats.tag); x = np.arange(len(tags))
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    ax = axes[0, 0]; wdt = 0.16
    odors = ['ms', 'paa', 'B-myr', '1-8ol', 'euc']
    for k, (o, c) in enumerate(zip(odors, ['#1f1f1f', '#8a6d00', '#d9480f', '#6b46c1', '#2b6cb0'])):
        ax.bar(x + (k - 2) * wdt, [fracs.loc[o, f'{t}_acute_off'] for t in tags], wdt, color=c, label=o)
        ax.axhline(fracs.loc[o, 'data_TNT'], color=c, lw=0.8, ls=':')
    ax.set_title('APL-off response fraction of the weak odors (dotted: APL output blocked (TNT) flies)', fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    panels = [('APL-off sharing deficit (0 = data)', [('deficit_terp_x_fruity', 'terpene / 1-8ol x fruity ester'), ('deficit_ms_paa_pairs', 'ms / paa pairs'), ('deficit_other_pairs', 'other pairs')], [0.0]),
        ('response rate ρ with the KC data (ceiling 0.98 / 0.97)', [('spearman_rate_on_vs_data', 'APL on'), ('spearman_rate_off_vs_data', 'APL off')], []),
        ('ρ on odor-odor distances with control flies (on) / APL output blocked (TNT) (off)', [(p + 'on_spearman', 'APL on'), (p + 'off_spearman', 'APL off')], []),
        ('structure beyond the ORN input (natural-source; data 0.44 on / 0.36 off)', [(p + 'on_spearman_resid_vs_vcf_hyp', 'APL on'), (p + 'off_spearman_resid_vs_vcf_hyp', 'APL off')], [0.44, 0.36]),
        ('block (data +0.055) and APL-on excess, block / other pairs (data -0.08 / -0.10)', [(p + 'block_fruity_small_iso', 'block'), ('on_excess_block', 'APL-on excess, block pairs'), ('on_excess_other_between', 'APL-on excess, other pairs')], [0.055, -0.08])]
    for ax, (title, cols, refs) in zip(list(axes.ravel())[1:], panels):
        for i, (col, lab) in enumerate(cols):
            ax.plot(x, stats[col].to_numpy(), ['-o', '--s', ':^'][i], color=['#2b6cb0', '#d9480f', '#2f855a'][i], ms=5, lw=1.2, label=lab)
        for r in refs:
            ax.axhline(r, color='0.45', lw=1, ls=':')
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7, frameon=False)
    for ax in axes.ravel():
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=40, ha='right', fontsize=7.5)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
        nA = sum(t.startswith('A_') for t in tags)
        if 0 < nA < len(tags):
            ax.axvline(nA - 0.5, color='0.6', lw=0.8)
    fig.suptitle("E38: Hallem's receptor data as the ORN input for ms, paa, B-myr and 1-8ol. A_: bouton base, 70 Hz per-claw threshold, tuned at 5 %; "
        "B_: the reference model's fitted scales, threshold re-tuned", fontsize=10)
    fig.tight_layout(); fig.savefig(res_dir / 'E38_hallem_input.png', dpi=130); plt.close(fig)
    # responders: ms and paa breadth / exclusivity / overlap agreement across variants
    sub = resp[(resp.condition == 'acute_off')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, title) in zip(axes, [('frac_exclusive', 'fraction of responders exclusive to the odor (APL off)'),
            ('mean_breadth', 'mean breadth of the responders (odors of 16; APL off)'),
            ('spearman_overlap_vs_data', "rank agreement of the co-responder profile with the flies' (APL off)")]):
        for o, c in zip(['ms', 'paa', 'B-myr', '1-8ol'], ['#1f1f1f', '#8a6d00', '#d9480f', '#6b46c1']):
            s = sub[sub.odor == o].set_index('tag')[col].reindex(tags)
            ax.plot(x, s.to_numpy(), '-o', color=c, ms=4, lw=1.1, label=o)
            dv = sub[(sub.odor == o) & (sub.tag == 'data')][col]
            if len(dv) and col != 'spearman_overlap_vs_data':
                ax.axhline(float(dv.iloc[0]), color=c, lw=0.8, ls=':')
        ax.set_xticks(x); ax.set_xticklabels(tags, rotation=40, ha='right', fontsize=7.5); ax.set_title(title, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6); ax.legend(fontsize=7, frameon=False)
    fig.suptitle('E38: the weak odors\' responders (dotted: APL output blocked (TNT) flies)', fontsize=10)
    fig.tight_layout(); fig.savefig(res_dir / 'E38_responders.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--parts', nargs='+', choices=['A', 'B'], default=['A', 'B'])
    ap.add_argument('--variants', nargs='+', default=None, help='restrict part A to these variants')
    ap.add_argument('--part-b-variants', nargs='+', default=PART_B_DEFAULT)
    ap.add_argument('--ref-fits', nargs='+', default=list(REF_FITS), choices=list(REF_FITS))
    ap.add_argument('--thr', type=float, default=70.0, help='per-claw threshold of part A')
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    resp_path = res_dir / 'responders.csv'; claw_path = res_dir / 'responder_claws.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        make_figure(pd.DataFrame(rows), pd.read_csv(fracs_path, index_col=0), pd.read_csv(resp_path), res_dir); return
    done = {r['tag'] for r in rows}
    orn0 = H.apl16_orn_deltas()
    variants = input_variants(orn0)
    (res_dir / 'inputs').mkdir(exist_ok=True)
    for tag, (orn, meta) in variants.items():
        H.abbrev_columns(orn).to_csv(res_dir / 'inputs' / f'{tag}.csv')
    hal = hallem_deltas(orn0.index); hal.to_csv(res_dir / 'inputs' / 'hallem_shared_glomeruli.csv')
    (res_dir / 'inputs' / 'variants.json').write_text(json.dumps({t: m for t, (_, m) in variants.items()}, indent=1) + '\n')
    orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive(); ref = H.reference_pairs(); refs = all_refs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    data_overlap = pd.read_csv(H.shared_dir() / 'results' / 'E23_weak_odor_responders' / 'overlap_profiles.csv')
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    resp_rows = list(pd.read_csv(resp_path).to_dict('records')) if resp_path.exists() else data_responder_rows('TNT') + data_responder_rows('TNT3c')
    claw_rows = list(pd.read_csv(claw_path).to_dict('records')) if claw_path.exists() else []
    tuned = json.loads((out_dir / 'tuned_params.json').read_text()) if (out_dir / 'tuned_params.json').exists() else {}
    pairs = {}; deficits = []

    def finish(tag, meta, on_out, off_out, wPNKC, extra_row: dict):
        row, on, off = score(meta, on_out, off_out, wPNKC, orn0, orn_pairs, drive, data, ref)
        on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
        row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
        for cond, s in (('on', on_out[1]), ('off', off_out[1])):
            row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
        row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
        row.update(suppression_stats(H.response_fractions(on_out[0]), H.response_fractions(off_out[0]), data_rates))
        row.update(extra_row)
        d = deficit_table(off_out[1], data_pairs); d.insert(0, 'tag', tag); deficits.append(d)
        for cond, r_, geno in (('on', on_out[0], 'TNT3c'), ('acute_off', off_out[0], 'TNT')):
            rr, cr = responder_analysis(tag, cond, r_, wPNKC, data_overlap, geno)
            resp_rows.extend(rr); claw_rows.extend(cr)
        rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
        pd.DataFrame(rows).to_csv(stats_path, index=False)
        pd.DataFrame(resp_rows).to_csv(resp_path, index=False); pd.DataFrame(claw_rows).to_csv(claw_path, index=False)
        fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index); fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
        fracs.to_csv(fracs_path)
        v = H.PRIMARY_MODEL_RDM
        wk = ' '.join(f"{o}:{fracs.loc[o, f'{tag}_acute_off']:.3f}" for o in ['ms', 'paa', 'B-myr', '1-8ol', 'euc'])
        ms = [r for r in resp_rows if r['tag'] == tag and r['condition'] == 'acute_off' and r['odor'] == 'ms']
        ms_s = (f"ms responders off: breadth {ms[0].get('mean_breadth', float('nan')):.1f} excl {ms[0].get('frac_exclusive', float('nan')):.2f} "
            f"single-cand-claw {ms[0].get('frac_responders_single_cand_claw', float('nan')):.2f} overlap-vs-data {ms[0].get('spearman_overlap_vs_data', float('nan')):.2f}") if ms else ''
        print(f"{tag}: thr {meta.get('fixed_thr_mean', float('nan')):.0f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f} (x{meta['rate_ratio_off_over_on']:.2f}); weak odors off [{wk}] (data "
            f"{' '.join(f'{o}:{fracs.loc[o, 'data_TNT']:.3f}' for o in ['ms', 'paa', 'B-myr', '1-8ol', 'euc'])}); deficit terp-x-fruity/ms-paa/other {row['deficit_terp_x_fruity']:+.3f}/{row['deficit_ms_paa_pairs']:+.3f}/"
            f"{row['deficit_other_pairs']:+.3f}; rate rho on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; mean dist on/off "
            f"{row[v + '__on_mean_model']:.3f}/{row[v + '__off_mean_model']:.3f}; block {row[v + '__block_fruity_small_iso']:+.3f}; vs ctrl {row[v + '__on_spearman']:.3f}; off vs TNT {row[v + '__off_spearman']:.3f}; "
            f"vcf resid on/off {row.get(v + '__on_spearman_resid_vs_vcf_hyp', float('nan')):.2f}/{row.get(v + '__off_spearman_resid_vs_vcf_hyp', float('nan')):.2f}; {ms_s}; {meta['seconds']:.0f}s", flush=True)
        pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
            res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
        done.add(tag)

    if 'A' in args.parts:
        for vtag, (orn, vmeta) in variants.items():
            if args.variants and vtag not in args.variants:
                continue
            tag = f'A_{vtag}'
            if tag in done:
                continue
            kws = dict(BOUTON, claw_thr=args.thr)
            t0 = time.time()
            try:
                on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag, with_pn_contrast=False)
            except Exception as e:
                print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                rows.append(dict(variant=vtag, tag=tag, part='A', claw_thr=args.thr, error=repr(e)[:300])); pd.DataFrame(rows).to_csv(stats_path, index=False); continue
            tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
            tuned[tag].update(claw_thr=args.thr, input=vmeta)
            (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1))
            meta = dict(variant=vtag, tag=tag, part='A', base='bouton', seed=None, target_rate=TARGET, claw_thr=args.thr, input_how=vmeta['how'],
                input_odors=','.join(vmeta['odors']), k=0.0, kc_kc_inh=0.0, gain=1.0,
                n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
                mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()), **extra,
                **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale') if k in params})
            meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
            finish(tag, meta, on_out, off_out, wPNKC, {})

    if 'B' in args.parts:
        from al_analysis.mb_model import make_fixed_param_kc_simulator
        for ftag in args.ref_fits:
            exp, thr, k, gain = REF_FITS[ftag]
            fits = pd.read_csv(H.shared_dir() / 'results' / exp / 'fits.csv')
            fr = fits[fits.tag == ftag]
            if len(fr) != 1:
                raise ValueError(f'{ftag} not found once in {exp}/fits.csv')
            fr = fr.iloc[0]
            p_fit = {kk: float(fr[kk]) for kk in PARAM_KEYS}
            p_fit['wKCKC_scale'] = float(fr['kc_kc_inh'])
            rate_target = float(fr['rate_on'])
            for vtag in args.part_b_variants:
                tag = f'B_{ftag}_{vtag}'
                if tag in done:
                    continue
                orn, vmeta = variants[vtag]
                vdir = out_dir / f'B_{ftag}_{vtag}'; vdir.mkdir(exist_ok=True)
                t0 = time.time()
                try:
                    initial, wnorm, W, breadth = bouton_base(orn, vdir, claw_thr=thr, gain=gain)
                    wAPLKC = breadth_weights(wnorm['wAPLKC'], breadth, k)
                    sim, _ = make_fixed_param_kc_simulator(orn, initial_params=p_fit, **base_kws(thr), _wPNKC=W,
                        _wAPLKC=wAPLKC, _wKCAPL=wnorm['wKCAPL'], _wAPLPN=wnorm['wAPLPN'], _wPNAPL=wnorm['wPNAPL'],
                        kc_kc_inh=p_fit['wKCKC_scale'], kc_kc_tau=TAU)
                    extra = {kk: p_fit[kk] for kk in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'wKCKC_scale')}
                    new_thr, thr_f, r_on, s_on = bisect_threshold(sim, p_fit['fixed_thr'], rate_target, extra=extra)
                    p_on = dict(p_fit, fixed_thr=float(new_thr))
                    r_off, s_off = sim(dict(p_on, wAPLKC_scale=0.0, wAPLPN_scale=0.0))
                    del sim
                except Exception as e:
                    print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                    rows.append(dict(variant=vtag, tag=tag, part='B', ref_fit=ftag, claw_thr=thr, error=repr(e)[:300])); pd.DataFrame(rows).to_csv(stats_path, index=False); continue
                for cond, (r_, s_) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
                    r_.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet'); s_.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
                tuned[tag] = dict(p_on, claw_thr=thr, k=k, gain=gain, ref_fit=ftag, ref_experiment=exp, thr_factor_vs_fit=float(thr_f), input=vmeta)
                (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1))
                meta = dict(variant=vtag, tag=tag, part='B', base='bouton', seed=None, target_rate=rate_target, claw_thr=thr, input_how=vmeta['how'],
                    input_odors=','.join(vmeta['odors']), k=k, kc_kc_inh=p_fit['wKCKC_scale'], gain=gain, ref_fit=ftag, ref_experiment=exp,
                    thr_factor_vs_fit=float(thr_f), fixed_thr_mean=float(new_thr), fit_fixed_thr=p_fit['fixed_thr'],
                    n_kcs=len(r_on), n_glomeruli=W.shape[1], rate_on=rate(r_on), rate_off=rate(r_off), seconds=time.time() - t0,
                    mean_claws_per_kc=float(kc_by_glomerulus(W).sum(axis='columns').mean()),
                    **{kk: p_fit[kk] for kk in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale')})
                meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
                finish(tag, meta, (r_on, s_on), (r_off, s_off), W, {})

    if deficits:
        pd.concat(deficits).to_csv(res_dir / 'pair_deficits_long.csv', index=False, mode='a', header=not (res_dir / 'pair_deficits_long.csv').exists())
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E38_hallem_input.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        base=BOUTON, part_a_claw_thr=args.thr, parts=args.parts, ref_fits={t: REF_FITS[t] for t in args.ref_fits}, part_b_variants=args.part_b_variants,
        hallem_odors=HALLEM_ODORS, hallem_conc_log10=HALLEM_CONC, yang_conc={o: H.YANG_CONC[o] for o in HALLEM_ODORS},
        hallem_conc_note='Hallem & Carlson 2006 odors at 10^-2; Yang\'s ms, paa and 1-8ol at 10^-3 and B-myr at 10^-2, so the Hallem-based inputs are within the one-log-unit rule but on its high side (upper bounds)',
        shared_glomeruli=list(hal.index), candidate_glomeruli=CAND_GLOMS, variants={t: m for t, (_, m) in variants.items()},
        approx_conc_note=H.APPROX_CONC_NOTE, model=H.model_provenance(orn_deltas=orn0, hallem_deltas=hal))
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    make_figure(pd.DataFrame(rows), pd.read_csv(fracs_path, index_col=0), pd.read_csv(resp_path), res_dir)
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
