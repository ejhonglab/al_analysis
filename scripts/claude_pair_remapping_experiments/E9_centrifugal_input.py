#!/usr/bin/env python3
"""E9: structured centrifugal input to the calyx (LHCENT / MB-C1 inhibition of KCs and boutons).

Builds, from the hemibrain v1.2.1 tables in al_analysis/data/hemibrain_neuprint/
(pull_neuprint_connectivity.py), an odor-dependent inhibitory drive onto the model's KCs and
PN boutons and hands it to olfsysm (fit_mb_model ext_inh_kc / ext_inh_bouton / ext_inh_scale;
rv.kc.ext_inh, rv.pn.bouton_ext_inh, a step over the odor window):
  1. glomerulus PN rates per odor: the bouton base model's PN rates (E3's tuned parameters),
     evoked (0-0.5 s mean) and spontaneous (pre-odor mean) per glomerulus;
  2. a lateral-horn-neuron (LHN) layer: LHN rate per odor = rectified PN-synapse-weighted mean
     of the odor-evoked PN rate change over its input glomeruli (uniglomerular PN -> LHN
     synapse counts in LH(R), >= 3 synapses; LHCENT cells excluded from the layer);
  3. centrifugal rates: for each LHCENT cell with calyx output and for MB-C1, the rectified
     synapse-weighted mean of its LHN inputs' rates and of its direct uniglomerular PN inputs'
     evoked changes (conn_to_centrifugal_all.csv, >= 3 synapses; MB-C1's KC input, 59 % of its
     synapses, is not modeled: its rate here is its PN / LHN drive);
  4. their projection onto the model: ext_inh_kc (KC x odor) = sum over cells of synapse count
     onto the KC (CA(R), >= 3) times the cell's rate; ext_inh_bouton (bouton x odor) likewise
     from the counts onto each PN, split evenly over the PN's boutons. Each matrix is
     normalized to mean 1 over its targeted rows x odors, so ext_inh_scale is the mean
     inhibition (Hz-equivalents) a targeted KC (or bouton) receives.
Variants: bouton base and the bouton base with the 70 Hz per-claw threshold (--bases), each
with no centrifugal input (control) and with KC targets only (scale 10 / 30 / 100), bouton
targets only (5 / 15 / 40) or both (KC scale with the bouton matrix at 0.4 x), thresholds
tuned at 5 % with APL on, acute APL off. Scored as E24 / E26 (E13 statistics incl. the
natural-source alignment of the residual, E7 excess by block, per-odor response fractions,
E16 reference ranks, E21's APL-off sharing deficit by pair type) plus the APL-independent
pair effects the report named: the residual of each pair's distance around the linear ORN
prediction for ep-6al, eb-t2h, ep-t2h and paa-B-myr (divergent in the data) and B-myr-euc,
B-myr-(-aPine) (convergent), model vs data, APL on and off.

Outputs: <shared>/results/E9_centrifugal_input/ and ./outputs/E9_centrifugal_input/.
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E9_centrifugal_input.py [--build-only]
    [--bases bouton bouton_thr70] [--targets kc bouton both] [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON, TARGET, per_odor_extras, rate, tune_and_knockout
from E4_substrate import score, kc_by_glomerulus
from E12_breadth_apl import density_excess, e3_bouton_params
from E16_claw_saturation import spearman_with_refs
from E18_bouton_depletion import all_refs
from E26_claw_power import sharing_deficit, deficit_table, WEAK
from pull_neuprint_connectivity import glomerulus_of

EXPERIMENT = 'E9_centrifugal_input'
HEMIBRAIN = (H.HERE.parents[1] / 'al_analysis' / 'data' / 'hemibrain_neuprint').resolve()
MIN_SYN = 3
BOUTON_REL = 0.4   # bouton matrix weight relative to the KC matrix in the 'both' variants
SCALES = {'kc': [10.0, 30.0, 100.0], 'bouton': [5.0, 15.0, 40.0], 'both': [10.0, 30.0, 100.0]}
BASES = {'bouton': dict(BOUTON), 'bouton_thr70': dict(BOUTON, claw_thr=70.0)}
DIVERGE = [('ep', '6al'), ('eb', 't2h'), ('ep', 't2h'), ('paa', 'B-myr')]
CONVERGE = [('B-myr', 'euc'), ('B-myr', '-aPine')]
NAMED_PAIRS = DIVERGE + CONVERGE
PARAM_KEYS = ('fixed_thr', 'wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale', 'ext_inh_scale')


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(HEMIBRAIN / name)


def glomerulus_pn_rates(orn: pd.DataFrame, res_dir, rebuild: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """(evoked: glomerulus x odor PN rate, 0-0.5 s mean; spont: glomerulus pre-odor PN rate;
    wPNKC) from the bouton base model at E3's tuned parameters (cached in res_dir)."""
    from al_analysis.mb_model import fit_mb_model, get_time_index, get_sim_var
    ev_path, sp_path = res_dir / 'glomerulus_rates_evoked.csv', res_dir / 'glomerulus_rates_spont.csv'
    init = e3_bouton_params()
    _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=init['fixed_thr'], wAPLKC=init['wAPLKC_scale'],
        wKCAPL=init['wKCAPL_scale'], wAPLPN=init['wAPLPN_scale'], wPNAPL=init['wPNAPL_scale'], **BOUTON,
        make_plots=False, silent=True, return_olfsysm_vars=True)
    gloms = list(dict.fromkeys(wPNKC.columns.get_level_values('glomerulus')))
    if ev_path.exists() and sp_path.exists() and not rebuild:
        return pd.read_csv(ev_path, index_col=0), pd.read_csv(sp_path, index_col=0).iloc[:, 0], wPNKC
    mp, rv = pd0['mp'], pd0['rv']
    t = np.asarray(get_time_index(mp, rv)); win = (t >= 0.0) & (t <= 0.5); pre = t < 0.0
    pn = get_sim_var(rv.pn, 'pn_sims')
    assert np.asarray(pn[0]).shape[0] == len(gloms), (np.asarray(pn[0]).shape, len(gloms))
    evoked = pd.DataFrame({H.odor_abbrev(c): np.asarray(pn[i])[:, win].mean(axis=1) for i, c in enumerate(orn.columns)}, index=gloms)
    spont = pd.Series(np.mean([np.asarray(pn[i])[:, pre].mean(axis=1) for i in range(len(orn.columns))], axis=0), index=gloms, name='spont')
    evoked.rename_axis('glomerulus').to_csv(ev_path); spont.rename_axis('glomerulus').to_csv(sp_path)
    return evoked, spont, wPNKC


def pn_to_lhn_weights(min_weight: int = MIN_SYN) -> pd.DataFrame:
    """Glomerulus x LHN matrix of PN -> LHN synapse counts in LH(R) (uniglomerular PNs summed
    per glomerulus; connections below `min_weight` synapses dropped; LHCENT cells excluded)."""
    c = read('conn_pn_to_lhn_LH.csv')
    c['glomerulus'] = c['type_pre'].map(glomerulus_of)
    c = c[c['glomerulus'].notna() & (c['weight'] >= min_weight) & ~c['type_post'].str.startswith('LHCENT', na=False)]
    return c.pivot_table(index='glomerulus', columns='bodyId_post', values='weight', aggfunc='sum', fill_value=0)


def lhn_rates(evoked: pd.DataFrame, spont: pd.Series, w: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """LHN x odor rates: rectified synapse-weighted mean of the odor-evoked PN rate change over
    the LHN's input glomeruli that the model has (Hz-equivalents)."""
    d = evoked.sub(spont.reindex(evoked.index), axis=0)
    common = w.index.intersection(d.index)
    wc, dc = w.loc[common], d.loc[common]
    tot = wc.sum(axis=0)
    keep = tot > 0
    r = (wc.loc[:, keep].T @ dc).div(tot[keep], axis=0).clip(lower=0)
    info = dict(n_lhn=int(keep.sum()), n_lhn_total=int(w.shape[1]), glomeruli_in_model=sorted(common),
        glomeruli_not_in_model=sorted(set(w.index) - set(common)),
        pn_synapses_in_model=float(wc.to_numpy().sum() / w.to_numpy().sum()))
    return r, info


def centrifugal_rates(lhn: pd.DataFrame, evoked: pd.DataFrame, spont: pd.Series, min_weight: int = MIN_SYN) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(rates: cell x odor for the calyx-projecting LHCENT cells and MB-C1, rectified
    synapse-weighted mean of their LHN inputs' rates and direct uniglomerular PN inputs'
    evoked changes; info: per cell, type and the fraction of its input synapses used)."""
    calyx = pd.concat([read('conn_lhcent_to_calyx_CA.csv'), read('conn_mbc1_to_calyx_CA.csv')])
    cells = calyx.groupby('bodyId_pre')['type_pre'].first()
    c = read('conn_to_centrifugal_all.csv')
    c = c[c['bodyId_post'].isin(cells.index)]
    total = c.groupby('bodyId_post')['weight'].sum()
    c = c[c['weight'] >= min_weight]
    is_lhn = c['type_pre'].str.startswith('LH', na=False) & ~c['type_pre'].str.startswith('LHCENT', na=False) & c['bodyId_pre'].isin(lhn.index)
    V = c[is_lhn].pivot_table(index='bodyId_post', columns='bodyId_pre', values='weight', aggfunc='sum', fill_value=0)
    cp = c[c['type_pre'].map(glomerulus_of).notna()].copy(); cp['glomerulus'] = cp['type_pre'].map(glomerulus_of)
    d = evoked.sub(spont.reindex(evoked.index), axis=0)
    cp = cp[cp['glomerulus'].isin(d.index)]
    U = cp.pivot_table(index='bodyId_post', columns='glomerulus', values='weight', aggfunc='sum', fill_value=0)
    rows, info = {}, []
    for cell in cells.index:
        v = V.loc[cell] if cell in V.index else pd.Series(dtype=float)
        u = U.loc[cell] if cell in U.index else pd.Series(dtype=float)
        num = pd.Series(0.0, index=evoked.columns)
        if len(v):
            num = num + (lhn.loc[v.index].mul(v, axis=0)).sum(axis=0)
        if len(u):
            num = num + (d.loc[u.index].mul(u, axis=0)).sum(axis=0)
        den = float(v.sum() + u.sum())
        rows[cell] = (num / den).clip(lower=0) if den > 0 else num
        info.append(dict(bodyId=cell, type=cells[cell], lhn_synapses=float(v.sum()), pn_synapses=float(u.sum()),
            total_input_synapses=float(total.get(cell, np.nan)), fraction_used=(den / total[cell]) if cell in total.index and total[cell] > 0 else np.nan,
            n_lhn_inputs=int((v > 0).sum()), n_glomeruli=int((u > 0).sum())))
    rates = pd.DataFrame(rows).T; rates.index.name = 'bodyId'
    rates.insert(0, 'type', cells.reindex(rates.index))
    return rates, pd.DataFrame(info).set_index('bodyId')


def calyx_projection(cent: pd.DataFrame, wPNKC: pd.DataFrame, min_weight: int = MIN_SYN) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(ext_inh_kc: KC x odor, ext_inh_bouton: bouton x odor, info) from the cells' synapse
    counts onto the model's KCs and PNs (per-PN counts split over the PN's boutons), each
    normalized to mean 1 over its targeted rows x odors."""
    odors = [c for c in cent.columns if c != 'type']
    s = pd.concat([read('conn_lhcent_to_calyx_CA.csv'), read('conn_mbc1_to_calyx_CA.csv')])
    s = s[(s['weight'] >= min_weight) & s['bodyId_pre'].isin(cent.index)]
    r = cent[odors].astype(float)
    kc_order = list(dict.fromkeys(wPNKC.index.get_level_values('kc_id')))
    kc = s[s['type_post'].str.startswith('KC', na=False)]
    S_kc = kc.pivot_table(index='bodyId_pre', columns='bodyId_post', values='weight', aggfunc='sum', fill_value=0)
    S_kc = S_kc.reindex(index=r.index, fill_value=0).reindex(columns=kc_order, fill_value=0)
    ext_kc = pd.DataFrame(S_kc.T.to_numpy() @ r.to_numpy(), index=pd.Index(kc_order, name='kc_id'), columns=odors)
    pn = s[s['type_post'].str.contains('PN', na=False)]
    S_pn = pn.pivot_table(index='bodyId_pre', columns='bodyId_post', values='weight', aggfunc='sum', fill_value=0).reindex(index=r.index, fill_value=0)
    boutons = wPNKC.columns
    pn_ids = boutons.get_level_values('pn_id')
    n_b = pd.Series(pn_ids).value_counts()
    S_b = pd.DataFrame(0.0, index=r.index, columns=range(len(boutons)))
    for j, pid in enumerate(pn_ids):
        if pid in S_pn.columns:
            S_b[j] = S_pn[pid] / n_b[pid]
    ext_b = pd.DataFrame(S_b.T.to_numpy() @ r.to_numpy(), index=boutons, columns=odors)
    info = dict(n_kcs_targeted=int((ext_kc.sum(axis=1) > 0).sum()), n_kcs=len(kc_order),
        n_boutons_targeted=int((ext_b.sum(axis=1) > 0).sum()), n_boutons=len(boutons),
        kc_synapses_in_model=float(S_kc.to_numpy().sum()), kc_synapses_total=float(kc['weight'].sum()),
        pn_synapses_in_model=float(S_b.to_numpy().sum()), pn_synapses_total=float(pn['weight'].sum()))
    for name, m in (('kc', ext_kc), ('bouton', ext_b)):
        targeted = m[m.sum(axis=1) > 0]
        norm = float(targeted.to_numpy().mean()) if len(targeted) else 1.0
        info[f'{name}_norm'] = norm
        if norm > 0:
            m /= norm
    return ext_kc, ext_b, info


def named_pair_resid(pairs_on: pd.Series, pairs_off: pd.Series, orn_pairs: pd.Series) -> dict:
    out = {}
    for cond, pr in (('on', pairs_on), ('off', pairs_off)):
        resid, _ = H.linear_resid(pr, orn_pairs)
        vals = {}
        for a, b in NAMED_PAIRS:
            vals[(a, b)] = float(resid.get((a, b), resid.get((b, a), np.nan)))
            out[f'{cond}_resid_{a}_{b}'] = vals[(a, b)]
        out[f'{cond}_resid_diverge_mean'] = float(np.nanmean([vals[p] for p in DIVERGE]))
        out[f'{cond}_resid_converge_mean'] = float(np.nanmean([vals[p] for p in CONVERGE]))
    return out


def build(orn, res_dir, rebuild: bool = False):
    """Builds and saves the LHN layer, centrifugal rates and the two projection matrices."""
    evoked, spont, wPNKC = glomerulus_pn_rates(orn, res_dir, rebuild)
    w = pn_to_lhn_weights()
    lhn, lhn_info = lhn_rates(evoked, spont, w)
    cent, cent_info = centrifugal_rates(lhn, evoked, spont)
    ext_kc, ext_b, proj_info = calyx_projection(cent, wPNKC)
    lhn.rename_axis('bodyId').to_csv(res_dir / 'lhn_rates.csv')
    cent.to_csv(res_dir / 'centrifugal_rates.csv'); cent_info.to_csv(res_dir / 'centrifugal_inputs.csv')
    ext_kc.to_csv(res_dir / 'ext_inh_kc.csv'); ext_b.to_csv(res_dir / 'ext_inh_bouton.csv')
    kc_type = pd.Series(wPNKC.index.get_level_values('kc_type'), index=wPNKC.index.get_level_values('kc_id')).groupby(level=0).first()
    by_type = ext_kc.groupby(kc_type.reindex(ext_kc.index).to_numpy()).mean()
    by_type.rename_axis('kc_type').to_csv(res_dir / 'ext_inh_kc_by_type.csv')
    (res_dir / 'build_info.json').write_text(json.dumps(dict(lhn=lhn_info, projection=proj_info, min_synapses=MIN_SYN,
        bouton_rel=BOUTON_REL), indent=1, default=str) + '\n')
    make_build_figure(cent, lhn, by_type, ext_b, wPNKC, res_dir)
    print(f"LHN layer: {lhn_info['n_lhn']} of {lhn_info['n_lhn_total']} LHNs have input from modeled glomeruli "
        f"({lhn_info['pn_synapses_in_model']:.2f} of PN->LHN synapses); glomeruli not in the model: {lhn_info['glomeruli_not_in_model']}")
    print(f"centrifugal cells: {len(cent)}; projection: {proj_info}")
    print(cent.round(1).to_string())
    return ext_kc, ext_b, wPNKC, cent


def make_build_figure(cent, lhn, by_type, ext_b, wPNKC, res_dir) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    odors = [c for c in cent.columns if c != 'type']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    ax = axes[0]
    m = cent[odors].astype(float); labels = [f'{t} {i}' for t, i in zip(cent['type'], cent.index)]
    im = ax.imshow(m.to_numpy(), aspect='auto', cmap='magma'); ax.set_yticks(range(len(m))); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(range(len(odors))); ax.set_xticklabels(odors, rotation=90, fontsize=7); ax.set_title('centrifugal cell rate per odor (Hz-equiv.)', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[1]
    frac = (lhn > 5).mean(axis=0).reindex(odors)
    ax.bar(range(len(odors)), frac.to_numpy(), color='#2b6cb0'); ax.set_xticks(range(len(odors))); ax.set_xticklabels(odors, rotation=90, fontsize=7)
    ax.set_title('fraction of LHNs with evoked rate > 5 Hz-equiv.', fontsize=9)
    ax = axes[2]
    bt = by_type.reindex(columns=odors)
    im = ax.imshow(bt.to_numpy(), aspect='auto', cmap='viridis'); ax.set_yticks(range(len(bt))); ax.set_yticklabels(bt.index, fontsize=8)
    ax.set_xticks(range(len(odors))); ax.set_xticklabels(odors, rotation=90, fontsize=7); ax.set_title('KC-level inhibition by KC type (mean over KCs; matrix mean 1 on targets)', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    for ax in axes[1:2]:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('E9: lateral-horn centrifugal / MB-C1 drive built from hemibrain v1.2.1 and the base model PN rates', fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / 'E9_centrifugal_build.png', dpi=130); plt.close(fig)


def make_figure(stats: pd.DataFrame, data_named: dict, res_dir, base: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = H.PRIMARY_MODEL_RDM + '__'
    if 'error' in stats:
        stats = stats[stats['error'].isna()]
    st = stats[stats.base == base]
    if not len(st):
        return
    ctrl = st[st.target == 'none']
    panels = [('fruity x small-ester block (data +0.055)', [(p + 'block_fruity_small_iso', 'block')], [0.055]),
        ('divergent pairs: residual vs ORN (ep-6al, eb-t2h, ep-t2h, paa-B-myr)', [('on_resid_diverge_mean', 'APL on'), ('off_resid_diverge_mean', 'APL off')],
            [data_named['TNT3c_diverge'], data_named['TNT_diverge']]),
        ('convergent pairs: residual vs ORN (B-myr-euc, B-myr-(-aPine))', [('on_resid_converge_mean', 'APL on'), ('off_resid_converge_mean', 'APL off')],
            [data_named['TNT3c_converge'], data_named['TNT_converge']]),
        ('natural-source alignment of the residual (data 0.44 ctrl / 0.36 TNT)', [(p + 'on_spearman_resid_vs_vcf_hyp', 'APL on'), (p + 'off_spearman_resid_vs_vcf_hyp', 'APL off')], [0.44, 0.36]),
        ('rank agreement with control (on) / TNT (off)', [(p + 'on_spearman', 'APL on vs control'), (p + 'off_spearman', 'APL off vs TNT')], []),
        ('mean pair distance (data 0.84 on / 0.66 off)', [(p + 'on_mean_model', 'APL on'), (p + 'off_mean_model', 'APL off')], [0.840, 0.661]),
        ('rate rank agreement with the data (ceiling 0.98 / 0.97)', [('spearman_rate_on_vs_data', 'APL on'), ('spearman_rate_off_vs_data', 'APL off')], []),
        ('APL-on excess, block pairs (data -0.08) / other (data -0.10)', [('on_excess_block', 'block pairs'), ('on_excess_other_between', 'other pairs')], [-0.08]),
        ('APL-off sharing deficit (0 = data)', [('deficit_terp_pairs', 'terpene / 1-8ol pairs'), ('deficit_ms_paa_pairs', 'ms / paa pairs'), ('deficit_other_pairs', 'other pairs')], [0.0])]
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11))
    colors = {'kc': '#2b6cb0', 'bouton': '#d9480f', 'both': '#2f855a'}
    for ax, (title, cols, refs) in zip(axes.ravel(), panels):
        for ci, (col, lab) in enumerate(cols):
            if col not in st:
                continue
            for tgt, c in colors.items():
                sub = st[st.target == tgt].sort_values('kc_scale')
                if not len(sub):
                    continue
                x = [0.0] + sub.kc_scale.tolist(); y = ([float(ctrl[col].iloc[0])] if len(ctrl) else [np.nan]) + sub[col].tolist()
                ax.plot(x, y, ['-o', '--s', ':^'][ci], color=c, ms=4, lw=1.2, label=f'{tgt} targets, {lab}')
        for rv_ in refs:
            if rv_ is not None and np.isfinite(rv_):
                ax.axhline(rv_, color='0.45', lw=1, ls=':')
        ax.set_title(title, fontsize=8.5); ax.set_xlabel('centrifugal scale (mean Hz-equiv. on targeted KCs; bouton scale = 0.4 x)', fontsize=7)
        ax.legend(fontsize=6, frameon=False); ax.spines[['top', 'right']].set_visible(False); ax.grid(True, color='0.9', lw=0.6)
    fig.suptitle(f'E9: centrifugal inhibition on the {base} base, thresholds tuned at 5 %', fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / f'E9_centrifugal_{base}.png', dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--rebuild', action='store_true', help='recompute the PN rates and the matrices')
    ap.add_argument('--bases', nargs='+', default=list(BASES), choices=list(BASES))
    ap.add_argument('--targets', nargs='+', default=['kc', 'bouton', 'both'], choices=['kc', 'bouton', 'both'])
    ap.add_argument('--figure-only', action='store_true')
    args = ap.parse_args()
    res_dir = H.results_dir(EXPERIMENT); out_dir = H.outputs_dir(EXPERIMENT)
    orn = H.apl16_orn_deltas()
    orn_pairs = H.orn16_consensus_pairs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    data_named = {}
    for g in ('TNT3c', 'TNT'):
        resid, _ = H.linear_resid(data['raw'][g], orn_pairs)
        for a, b in NAMED_PAIRS:
            data_named[f'{g}_{a}_{b}'] = float(resid.get((a, b), resid.get((b, a), np.nan)))
        data_named[f'{g}_diverge'] = float(np.nanmean([data_named[f'{g}_{a}_{b}'] for a, b in DIVERGE]))
        data_named[f'{g}_converge'] = float(np.nanmean([data_named[f'{g}_{a}_{b}'] for a, b in CONVERGE]))
    pd.Series(data_named).rename_axis('key').to_csv(res_dir / 'named_pairs_data.csv', header=['value'])
    stats_path = res_dir / 'stats_variants.csv'; fracs_path = res_dir / 'per_odor_response_fractions.csv'
    rows = list(pd.read_csv(stats_path).to_dict('records')) if stats_path.exists() else []
    rows = [r for r in rows if not (isinstance(r.get('error'), str) and r['error'])]
    if args.figure_only:
        for b in args.bases:
            make_figure(pd.DataFrame(rows), data_named, res_dir, b)
        return
    ext_kc, ext_b, wPNKC0, cent = build(orn, res_dir, args.rebuild)
    if args.build_only:
        return
    done = {r['tag'] for r in rows}
    drive = H.consensus_orn_drive(); ref = H.reference_pairs(); refs = all_refs()
    data_rates = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean() for g in ('TNT3c', 'TNT')}
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv').set_index(['odor_a', 'odor_b'])
    fracs = pd.read_csv(fracs_path, index_col=0) if fracs_path.exists() else pd.DataFrame({'data_TNT3c': data_rates['TNT3c'], 'data_TNT': data_rates['TNT']})
    tuned = json.loads((out_dir / 'tuned_params.json').read_text()) if (out_dir / 'tuned_params.json').exists() else {}
    pairs = {}; deficits = []
    for base in args.bases:
        variants = [('none', 0.0)] + [(t, s) for t in args.targets for s in SCALES[t]]
        for target, scale in variants:
            tag = f'{base}_none' if target == 'none' else f'{base}_{target}_s{scale:g}'
            if tag in done:
                continue
            kws = dict(BASES[base])
            if target in ('kc', 'both'):
                kws['ext_inh_kc'] = ext_kc
            if target in ('bouton', 'both'):
                kws['ext_inh_bouton'] = ext_b * (BOUTON_REL if target == 'both' else 1.0)
            if target != 'none':
                kws['ext_inh_scale'] = scale
            t0 = time.time()
            try:
                on_out, off_out, wPNKC, params, extra = tune_and_knockout(orn, kws, out_dir, tag, with_pn_contrast=False)
            except Exception as e:
                print(f'{tag} FAILED: {e!r}'[:300], flush=True)
                rows.append(dict(variant=tag, tag=tag, base=base, target=target, kc_scale=scale, error=repr(e)[:300])); pd.DataFrame(rows).to_csv(stats_path, index=False); continue
            tuned[tag] = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in params.items() if k in PARAM_KEYS}
            tuned[tag].update(base=base, target=target, kc_scale=scale, bouton_scale=(scale * (BOUTON_REL if target == 'both' else 1.0)) if target in ('bouton', 'both') else 0.0,
                claw_thr=BASES[base].get('claw_thr', 0.0))
            (out_dir / 'tuned_params.json').write_text(json.dumps(tuned, indent=1))
            meta = dict(variant=tag, base=base, seed=None, target_rate=TARGET, tag=tag, target=target, kc_scale=scale,
                bouton_scale=tuned[tag]['bouton_scale'], claw_thr=BASES[base].get('claw_thr', 0.0),
                n_kcs=len(on_out[0]), n_glomeruli=wPNKC.shape[1], rate_on=rate(on_out[0]), rate_off=rate(off_out[0]), seconds=time.time() - t0,
                mean_claws_per_kc=float(kc_by_glomerulus(wPNKC).sum(axis='columns').mean()), **extra,
                **{k: float(params[k]) for k in ('wAPLKC_scale', 'wKCAPL_scale', 'wAPLPN_scale', 'wPNAPL_scale') if k in params})
            meta['rate_ratio_off_over_on'] = meta['rate_off'] / meta['rate_on']
            row, on, off = score(meta, on_out, off_out, wPNKC, orn, orn_pairs, drive, data, ref)
            on_p, off_p = on[H.PRIMARY_MODEL_RDM], off[H.PRIMARY_MODEL_RDM]
            row.update(per_odor_extras(on_out[0], off_out[0], on_p, off_p, drive, data_rates))
            for cond, s in (('on', on_out[1]), ('off', off_out[1])):
                row.update({f'{cond}_{kk}': v for kk, v in density_excess(s).items()})
            row.update(spearman_with_refs(on_p, off_p, refs)); row.update(sharing_deficit(off_out[1], data_pairs))
            row.update(named_pair_resid(on_p, off_p, orn_pairs))
            d = deficit_table(off_out[1], data_pairs); d.insert(0, 'tag', tag); deficits.append(d)
            rows.append(row); pairs[(tag, 'on')] = on_p; pairs[(tag, 'acute_off')] = off_p
            pd.DataFrame(rows).to_csv(stats_path, index=False)
            fracs[f'{tag}_on'] = H.response_fractions(on_out[0]).reindex(fracs.index); fracs[f'{tag}_acute_off'] = H.response_fractions(off_out[0]).reindex(fracs.index)
            fracs.to_csv(fracs_path)
            v = H.PRIMARY_MODEL_RDM
            print(f"{tag}: thr {meta.get('fixed_thr_mean', float('nan')):.0f}, rates on {meta['rate_on']:.4f} off {meta['rate_off']:.4f}; block {row[v + '__block_fruity_small_iso']:+.3f}; "
                f"diverge on/off {row['on_resid_diverge_mean']:+.3f}/{row['off_resid_diverge_mean']:+.3f} (data {data_named['TNT3c_diverge']:+.3f}/{data_named['TNT_diverge']:+.3f}); "
                f"converge on/off {row['on_resid_converge_mean']:+.3f}/{row['off_resid_converge_mean']:+.3f} (data {data_named['TNT3c_converge']:+.3f}/{data_named['TNT_converge']:+.3f}); "
                f"vcf resid on/off {row.get(v + '__on_spearman_resid_vs_vcf_hyp', float('nan')):.2f}/{row.get(v + '__off_spearman_resid_vs_vcf_hyp', float('nan')):.2f}; "
                f"mean dist on/off {row[v + '__on_mean_model']:.3f}/{row[v + '__off_mean_model']:.3f}; vs ctrl {row[v + '__on_spearman']:.3f}; off vs TNT {row[v + '__off_spearman']:.3f}; "
                f"rate rank on/off {row['spearman_rate_on_vs_data']:.2f}/{row['spearman_rate_off_vs_data']:.2f}; on excess block/other {row['on_excess_block']:+.3f}/{row['on_excess_other_between']:+.3f}; {meta['seconds']:.0f}s", flush=True)
            pd.concat(pairs, names=['tag', 'condition', 'odor_a', 'odor_b']).rename('distance').reset_index().to_csv(
                res_dir / 'model_pair_distances_long.csv', index=False, mode='a', header=not (res_dir / 'model_pair_distances_long.csv').exists()); pairs.clear()
            done.add(tag)
        make_figure(pd.DataFrame(rows), data_named, res_dir, base)
    if deficits:
        pd.concat(deficits).to_csv(res_dir / 'pair_deficits_long.csv', index=False, mode='a', header=not (res_dir / 'pair_deficits_long.csv').exists())
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E9_centrifugal_input.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        olfsysm_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=(H.HERE / '../../../olfsysm').resolve(), capture_output=True, text=True).stdout.strip(),
        bases={b: BASES[b] for b in args.bases}, scales=SCALES, bouton_rel=BOUTON_REL, min_synapses=MIN_SYN, named_pairs=dict(diverge=DIVERGE, converge=CONVERGE),
        hemibrain=json.loads((HEMIBRAIN / 'provenance.json').read_text()), approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
