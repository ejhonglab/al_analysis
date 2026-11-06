#!/usr/bin/env python3
"""Bouton->claw synapse counts in Pratyush's hemibrain segmentation: consistency with the
loader, their spread, and the effect of weighting bouton>claw connections by them.

Part 1 (data, no simulation): rebuilds `connectome_wPNKC`'s collapsed claw table from the
per-synapse file it reads (`PNbouton-to-KCclaw_Connectivity_2025-12-09-19-47-36.parquet`),
with the loader's filters (null claw IDs, '+' and multiglomerular PN types, infinite dist_to_root,
null KC types and unassigned boutons dropped; -1 claws kept; >= 5 synapses per (KC, claw,
PN); claws with two PNs dropped), keeping the per-bouton synapse counts the loader
discards, and compares it with the loader's actual output (claw set; the bouton each claw
was assigned to under the current "first bouton by ID" rule). Then: the distribution of
synapses per bouton>claw connection and per claw; how many connections / claws are below
the 5-synapse threshold per bouton vs after merging a claw's boutons; how the counts vary
across glomeruli and across KC types.

Part 2 (--model): bouton base tuned at 5 % with claw weights = presence (current) vs
synapse counts, and multi-bouton claws assigned to the first / dominant bouton or split by
synapses (mb_model `claw_weight`, `multibouton_claws`), scored as the experiments.

Outputs: <shared>/results/bouton_claw_synapses/ (tables, summary.md) and a printed summary.
Usage: ./venv/bin/python check_bouton_claw_synapses.py [--loader-table loader_claw2bouton.parquet] [--model]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import kc_remapping_harness as H

SYN_PATH = (H.REPO / 'al_analysis' / 'data' / 'from_pratyush' / '2025-12-05' /
    'PNbouton-to-KCclaw_Connectivity_2025-12-09-19-47-36.parquet')
THRESH = 5   # synapses per (KC, claw, PN), as in connectome_wPNKC
OUT = H.shared_dir() / 'results' / 'bouton_claw_synapses'


def glomerulus_of(pn_type: str):
    return pn_type.split('_')[0] if isinstance(pn_type, str) and '_' in pn_type and not pn_type.startswith('M_') else None


def kc_type_of(type_post: str):
    t = str(type_post)
    return "a'b'" if "a'b'" in t else ('ab' if 'ab' in t else ('g' if t.startswith('KCg') else t))


def rebuild_claws():
    """The loader's collapsed table, plus per-bouton synapse counts."""
    syn = pd.read_parquet(SYN_PATH)
    n0 = len(syn)
    syn = syn.dropna(subset=['anatomical_claw_corrected'])
    n_null_claw = n0 - len(syn)
    # the loader's filters, in its order (connectome_wPNKC, prat_claws + prat_boutons):
    # PN types with a '+' (PNs assigned to two glomeruli; _drop_glom_with_plus=True),
    # multiglomerular PNs (glomerulus 'M', not a task glomerulus), synapses with an
    # infinite dist_to_root, synapses of KCs with a null type, synapses with bouton -1
    # (unassigned) or null
    filt = {}
    m = syn.type_pre.str.contains(r'\+', regex=True); filt['plus_pn_types'] = int(m.sum()); syn = syn[~m]
    m = syn.type_pre.str.startswith('M_'); filt['multiglomerular_pns'] = int(m.sum()); syn = syn[~m]
    m = ~np.isfinite(syn.dist_to_root); filt['inf_dist_to_root'] = int(m.sum()); syn = syn[~m]
    m = syn.type_post.isna(); filt['null_kc_type'] = int(m.sum()); syn = syn[~m]
    m = syn.anatomical_bouton.isna() | (syn.anatomical_bouton == -1); filt['bouton_unassigned'] = int(m.sum()); syn = syn[~m]
    syn = syn.rename(columns={'bodyId_post': 'kc_id', 'anatomical_claw_corrected': 'claw_id',
        'bodyId_pre': 'pn_id', 'anatomical_bouton': 'bouton_id'})
    syn['claw_id'] = syn.claw_id.astype(int); syn['bouton_id'] = syn.bouton_id.astype(int)
    syn['glomerulus'] = syn.type_pre.map(glomerulus_of)
    syn['kc_type'] = syn.type_post.map(kc_type_of)
    keys = ['kc_id', 'claw_id', 'pn_id']
    per_bouton = syn.groupby(keys + ['bouton_id']).agg(n_syn=('bouton_id', 'size'),
        glomerulus=('glomerulus', 'first'), kc_type=('kc_type', 'first')).reset_index()
    per_claw = per_bouton.groupby(keys).agg(n_syn=('n_syn', 'sum'), n_boutons=('bouton_id', 'nunique'),
        glomerulus=('glomerulus', 'first'), kc_type=('kc_type', 'first')).reset_index()
    # loader filters
    above = per_claw[per_claw.n_syn >= THRESH]
    n_sub = len(per_claw) - len(above)
    multi_pn = above.groupby(['kc_id', 'claw_id']).filter(lambda g: len(g) > 1)
    kept = above.drop(index=multi_pn.index)
    per_bouton_kept = per_bouton.merge(kept[keys], on=keys)
    info = dict(synapses=n0, synapses_null_claw=n_null_claw, synapses_dropped=filt, claw_pn_groups=len(per_claw),
        claw_pn_below_threshold=n_sub, claws_with_two_pns_dropped=multi_pn.groupby(['kc_id', 'claw_id']).ngroups,
        claws_kept=len(kept), kcs_kept=int(kept.kc_id.nunique()))
    return kept, per_bouton_kept, per_claw, per_bouton, info


def compare_with_loader(kept, per_bouton_kept, loader_path: Path) -> dict:
    lo = pd.read_parquet(loader_path)
    cols = {c.lower(): c for c in lo.columns}
    kc, claw, pn, bt = cols['kc_id'], cols['claw_id'], cols['pn_id'], cols['bouton_id']
    lo = lo[[kc, claw, pn, bt]].rename(columns={kc: 'kc_id', claw: 'claw_id', pn: 'pn_id', bt: 'bouton_id'})
    lo['pn_id'] = lo.pn_id.astype(int); lo['bouton_id'] = lo.bouton_id.astype(int)
    a = set(map(tuple, kept[['kc_id', 'claw_id', 'pn_id']].to_numpy()))
    b = set(map(tuple, lo[['kc_id', 'claw_id', 'pn_id']].to_numpy()))
    first = per_bouton_kept.sort_values('bouton_id').groupby(['kc_id', 'claw_id', 'pn_id']).bouton_id.first()
    dominant = per_bouton_kept.sort_values(['n_syn', 'bouton_id'], ascending=[False, True]).groupby(['kc_id', 'claw_id', 'pn_id']).bouton_id.first()
    lo_assign = lo.set_index(['kc_id', 'claw_id', 'pn_id']).bouton_id
    common = lo_assign.index.intersection(first.index)
    out = dict(claws_rebuilt=len(a), claws_loader=len(b), only_rebuilt=len(a - b), only_loader=len(b - a),
        loader_assignment_matches_first_bouton=int((lo_assign.loc[common] == first.loc[common]).sum()),
        loader_assignment_matches_dominant_bouton=int((lo_assign.loc[common] == dominant.loc[common]).sum()),
        common=len(common))
    return out


def stats(kept, per_bouton_kept) -> dict:
    nb = kept.n_boutons.value_counts().sort_index()
    pb = per_bouton_kept.n_syn
    multi = per_bouton_kept.merge(kept[kept.n_boutons > 1][['kc_id', 'claw_id', 'pn_id']])
    g = multi.groupby(['kc_id', 'claw_id', 'pn_id']).n_syn
    share = g.max() / g.sum()
    res = dict(
        boutons_per_claw=nb.to_dict(),
        syn_per_connection_quantiles=pb.quantile([0, .1, .25, .5, .75, .9, 1]).round(1).to_dict(),
        syn_per_connection_cv=float(pb.std() / pb.mean()),
        syn_per_claw_quantiles=kept.n_syn.quantile([0, .1, .25, .5, .75, .9, 1]).round(1).to_dict(),
        syn_per_claw_cv=float(kept.n_syn.std() / kept.n_syn.mean()),
        connections_below_threshold=int((pb < THRESH).sum()), connections_total=len(pb),
        connections_below_threshold_frac=float((pb < THRESH).mean()),
        multibouton_claws=int((kept.n_boutons > 1).sum()),
        multibouton_claws_dominant_share_quantiles=share.quantile([0, .1, .25, .5, .75, .9, 1]).round(2).to_dict() if len(share) else {},
        multibouton_claws_no_bouton_above_threshold=int((g.max() < THRESH).sum()),
        multibouton_claws_secondary_boutons_below_threshold=int(((g.sum() - g.max()) < THRESH).sum()),
    )
    return res


def by_group(kept, per_bouton_kept) -> dict:
    tabs = {}
    for col in ('glomerulus', 'kc_type'):
        t = kept.groupby(col).n_syn.agg(['count', 'mean', 'median', 'std']).round(2)
        t['cv'] = (t['std'] / t['mean']).round(2)
        t['frac_multibouton'] = kept.groupby(col).n_boutons.apply(lambda x: (x > 1).mean()).round(3)
        tabs[col] = t.sort_values('mean')
    return tabs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--loader-table', type=Path, default=None, help="parquet of the loader's claw->bouton assignment (stacked connectome_wPNKC)")
    ap.add_argument('--model', action='store_true', help='also run the model comparison (part 2)')
    ap.add_argument('--model-only', action='store_true', help='only run the model comparison')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.model_only:
        run_model_comparison(); return
    kept, per_bouton_kept, per_claw, per_bouton, info = rebuild_claws()
    print('loader-equivalent table:', json.dumps(info))
    res = dict(rebuild=info)
    if args.loader_table and args.loader_table.exists():
        res['consistency'] = compare_with_loader(kept, per_bouton_kept, args.loader_table)
        print('consistency with the loader:', json.dumps(res['consistency']))
    res['stats'] = stats(kept, per_bouton_kept)
    print('stats:', json.dumps(res['stats'], indent=1, default=str))
    tabs = by_group(kept, per_bouton_kept)
    for col, t in tabs.items():
        t.to_csv(OUT / f'synapses_per_claw_by_{col}.csv')
        print(f'--- synapses per claw by {col} (lowest / highest means):'); print(pd.concat([t.head(5), t.tail(5)]).to_string())
    kept.to_csv(OUT / 'claws_rebuilt.csv', index=False)
    per_bouton_kept.to_csv(OUT / 'bouton_claw_connections.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    if args.model:
        run_model_comparison()
    print('->', OUT)


MODEL_VARIANTS = [
    ('first', 'presence'), ('argmax', 'presence'), ('split', 'presence'),
    ('first', 'synapses'), ('split', 'synapses'),
]


def _raw_synapse_weight_stats(mb: str):
    """Raw per-claw (or per (claw, bouton) connection, for mb='split') synapse counts that
    claw_weight='synapses' normalizes to mean 1, from part 1's connection table (None if
    that table has not been written)."""
    path = OUT / 'bouton_claw_connections.csv'
    if not path.exists():
        return None
    c = pd.read_csv(path)
    n = c.n_syn if mb == 'split' else c.groupby(['kc_id', 'claw_id']).n_syn.sum()
    return dict(n_nonzero=int(len(n)), mean=float(n.mean()), median=float(n.median()),
        cv=float(n.std() / n.mean()), min=float(n.min()), max=float(n.max()))


def run_model_comparison(variants=None) -> None:
    """Bouton base (E3 kwargs) tuned at 5 % with APL on, acute APL off, for each
    (multibouton_claws, claw_weight) variant; scored as the experiments plus the rank
    agreement of each variant's pair distances with the default's and the fraction of
    (KC, odor) responses that differ from the default's."""
    from E3_drive_dependent_gain import BOUTON, TARGET, rate
    from E4_substrate import score, kc_by_glomerulus
    from al_analysis.mb_model import fit_mb_model
    orn = H.apl16_orn_deltas(); orn_pairs = H.orn16_consensus_pairs(); drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')},
        'binarized': {g: H.yang_mean_rdm(g, 'cosine', binarize=1.0) for g in ('TNT3c', 'TNT')},
        'pearson': {g: H.yang_mean_rdm(g, 'pearson') for g in ('TNT3c', 'TNT')}}
    rows, outs = [], {}
    v = H.PRIMARY_MODEL_RDM
    syn_info = {}
    for mb, cw in (variants or MODEL_VARIANTS):
        tag = f'{mb}_{cw}'
        # claw_weight='synapses' is normalized to mean 1 over the non-zero weights inside
        # connectome_wPNKC (raw counts, mean ~15 per claw, scale every claw's input ~15x and
        # the threshold search does not converge), so thresholds and claw_thr keep their
        # meaning and the manipulation is the spread of claw strength (CV ~0.43)
        kws = dict(BOUTON, multibouton_claws=mb, claw_weight=cw, make_plots=False, silent=True)
        if cw == 'synapses':
            syn_info[tag] = _raw_synapse_weight_stats(mb)
        r_on, s_on, wPNKC, params = fit_mb_model(orn, target_sparsity=TARGET, **kws)
        r_off, s_off, _, _ = fit_mb_model(orn, fixed_thr=params['fixed_thr'], wAPLKC=0.0,
            wKCAPL=params['wKCAPL_scale'], wAPLPN=0.0, wPNAPL=params['wPNAPL_scale'], **kws)
        meta = dict(variant=tag, tag=tag, multibouton_claws=mb, claw_weight=cw, base='bouton',
            seed=None, target_rate=TARGET, fixed_thr=float(np.mean(params['fixed_thr'])),
            wAPLKC_scale=params.get('wAPLKC_scale'), wKCAPL_scale=params.get('wKCAPL_scale'),
            wAPLPN_scale=params.get('wAPLPN_scale'), wPNAPL_scale=params.get('wPNAPL_scale'),
            n_kcs=len(r_on), n_glomeruli=wPNKC.shape[1], rate_on=rate(r_on), rate_off=rate(r_off),
            rate_ratio_off_over_on=rate(r_off) / max(rate(r_on), 1e-9),
            synapse_weights_normalized_to_mean_1=(cw == 'synapses'),
            raw_synapse_weight_mean=(syn_info.get(tag) or {}).get('mean'),
            wpnkc_nonzero_per_claw_mean=float((wPNKC > 0).sum(axis='columns').mean()),
            wpnkc_row_sum_cv=float(wPNKC.sum(axis='columns').std() / wPNKC.sum(axis='columns').mean()))
        row, on, off = score(meta, (r_on, s_on), (r_off, s_off), wPNKC, orn, orn_pairs, drive, data, ref)
        outs[tag] = dict(on=on[v], off=off[v], r_on=r_on, r_off=r_off)
        rows.append(row)
        for cond, (r, sp) in (('on', (r_on, s_on)), ('acute_off', (r_off, s_off))):
            out_dir = H.outputs_dir('bouton_claw_synapses'); out_dir.mkdir(parents=True, exist_ok=True)
            r.to_parquet(out_dir / f'{tag}_{cond}_responses.parquet'); sp.to_parquet(out_dir / f'{tag}_{cond}_spike_counts.parquet')
        print(f'{tag}: thr {meta["fixed_thr"]:.1f}, rates on {meta["rate_on"]:.4f} off {meta["rate_off"]:.4f}; block {row[v + "__block_fruity_small_iso"]:+.3f}; vs ctrl {row[v + "__on_spearman"]:.3f}; vs TNT {row[v + "__off_spearman"]:.3f}; iso {row[v + "__iso_r2"]:.3f}', flush=True)
    base = outs['first_presence']
    for row in rows:
        o = outs[row['tag']]
        row['spearman_on_vs_default'] = H.spearman(o['on'], base['on'])
        row['spearman_off_vs_default'] = H.spearman(o['off'], base['off'])
        a, b = o['r_on'].astype(bool).align(base['r_on'].astype(bool), join='inner')
        row['frac_responses_changed_on'] = float((a != b).to_numpy().mean())
        row['frac_responders_changed_on'] = float((a != b).to_numpy().sum() / max(b.to_numpy().sum(), 1))
        a, b = o['r_off'].astype(bool).align(base['r_off'].astype(bool), join='inner')
        row['frac_responses_changed_off'] = float((a != b).to_numpy().mean())
    pd.DataFrame(rows).to_csv(OUT / 'model_comparison.csv', index=False)
    if any(syn_info.values()):
        (OUT / 'synapse_weight_normalization.json').write_text(json.dumps(syn_info, indent=1) + '\n')
    cols = ['tag', 'fixed_thr', 'rate_on', 'rate_off', v + '__block_fruity_small_iso', v + '__on_spearman', v + '__off_spearman', v + '__iso_r2', v + '__on_mean_model', 'spearman_on_vs_default', 'spearman_off_vs_default', 'frac_responses_changed_on', 'frac_responders_changed_on']
    print(pd.DataFrame(rows)[cols].round(3).to_string())


if __name__ == '__main__':
    main()
