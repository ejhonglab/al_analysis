#!/usr/bin/env python3
"""E21: why the model's APL-blocked code shares too little among non-ester pairs (diagnostic).

With the per-claw threshold on, the model's APL-blocked matrix matches the data's excess
sharing for the fruity x small-ester pairs but shares too little among the other
between-group pairs, and its rank agreement with the APL-blocked KC data stays at 0.58-0.68
in every model (ceiling 0.97). No new models: for the untouched bouton base, the per-claw
threshold models (40 / 70 Hz; E16), the three-mechanism grid's two operating points (E19) and
any archived joint-fit model (E10), the per-pair excess sharing (graded cosine distance
observed minus the density null) of the APL-off (and APL-on) code is compared pair by pair
with the TNT (control) data (E7's B_pairs_graded.csv), and the deficit is broken down by odor
group pair, by involvement of the three odors at substituted ORN concentrations (paa, euc,
-aPine), per odor, and against the glomeruli the two odors co-drive in the ORN input and
above the threshold in the model (per-glomerulus PN rates per odor from the base model).

Outputs: <shared>/results/E21_apl_off_sharing/ (pairs_excess.csv, group_deficit.csv,
per_odor.csv, pair_codrive.csv, E21_apl_off_sharing.png, provenance.json).
Usage: OMP_NUM_THREADS=16 nice -n 10 ./venv/bin/python E21_apl_off_sharing.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H
from E3_drive_dependent_gain import BOUTON
from E12_breadth_apl import e3_bouton_params

EXPERIMENT = 'E21_apl_off_sharing'
ARCHIVE = H.shared_dir() / 'data' / 'model_outputs'
SUBST = {'paa', 'euc', '-aPine'}
MODELS = {   # label -> (archive folder, variant)
    'base': ('E16_claw_saturation', 'bouton_nocap'),
    'thr40': ('E16_claw_saturation', 'bouton_thr40'),
    'thr70': ('E16_claw_saturation', 'bouton_thr70'),
    'thr40_k2_kckc30': ('E19_thr_breadth_kckc', 'thr40_k2_kckc30'),
    'thr70_k2_kckc100': ('E19_thr_breadth_kckc', 'thr70_k2_kckc100'),
}


def load_counts(folder: str, variant: str, cond: str) -> pd.DataFrame:
    return pd.read_parquet(ARCHIVE / folder / f'{variant}_{cond}_spike_counts.parquet')


def pair_excess(spk: pd.DataFrame) -> pd.Series:
    obs = H.pair_distances(spk, 'cosine')
    null = 1 - H.density_null_similarity(spk)
    return (obs - null.reindex(obs.index)).rename('excess')


def group_of(o: str) -> str:
    return H.ODOR_GROUPS.get(o, o)


def main() -> None:
    res_dir = H.results_dir(EXPERIMENT)
    orn = H.apl16_orn_deltas()
    odors = list(H.APL16_ODORS)
    data_pairs = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_pairs_graded.csv')
    data_pairs = data_pairs.set_index(['odor_a', 'odor_b'])
    data_po = pd.read_csv(H.shared_dir() / 'results' / 'E7_confound_controls' / 'B_per_odor.csv', index_col=0)
    # joint-fit models archived by atlas, if any
    e10 = ARCHIVE / 'E10_joint_fit'
    models = dict(MODELS)
    if e10.exists():
        for p in sorted(e10.glob('*_acute_off_spike_counts.parquet')):
            tag = p.name.replace('_acute_off_spike_counts.parquet', '')
            if tag in ('k1', 'k2', 'k3'):
                models[f'fit_{tag}'] = ('E10_joint_fit', tag)

    # 1. per-pair excess, model vs data, both conditions
    rows = []
    for label, (folder, variant) in models.items():
        for cond, dcol in (('acute_off', 'TNT_excess'), ('on', 'TNT3c_excess')):
            try:
                spk = load_counts(folder, variant, cond)
            except FileNotFoundError:
                print(f'{label} {cond}: no file', flush=True); continue
            ex = pair_excess(spk)
            for (a, b), v in ex.items():
                d = data_pairs[dcol].get((a, b), data_pairs[dcol].get((b, a), np.nan))
                rows.append(dict(model=label, condition=cond, odor_a=a, odor_b=b, group_a=group_of(a), group_b=group_of(b),
                    block=H.block_label(a, b),
                    involves_subst=int(a in SUBST or b in SUBST), model_excess=float(v), data_excess=float(d), deficit=float(v - d)))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(res_dir / 'pairs_excess.csv', index=False)

    # 2. aggregates: by group pair, by substituted-odor involvement, per odor (APL off vs TNT)
    off = pairs[pairs.condition == 'acute_off']
    grp = off.groupby(['model', 'block']).agg(n=('deficit', 'size'), model_excess=('model_excess', 'mean'),
        data_excess=('data_excess', 'mean'), deficit=('deficit', 'mean')).reset_index()
    sub = off.groupby(['model', 'involves_subst']).agg(n=('deficit', 'size'), deficit=('deficit', 'mean'),
        model_excess=('model_excess', 'mean'), data_excess=('data_excess', 'mean')).reset_index()
    grp.to_csv(res_dir / 'group_deficit.csv', index=False)
    sub.to_csv(res_dir / 'subst_deficit.csv', index=False)
    per_odor = []
    for label in off.model.unique():
        m = off[off.model == label]
        for o in odors:
            mm = m[(m.odor_a == o) | (m.odor_b == o)]
            per_odor.append(dict(model=label, odor=o, group=group_of(o), deficit=float(mm.deficit.mean()),
                model_excess=float(mm.model_excess.mean()), data_excess=float(mm.data_excess.mean())))
    per_odor = pd.DataFrame(per_odor)
    # per-odor response fractions, APL off, model vs TNT
    fr = {}
    for label, (folder, variant) in models.items():
        try:
            fr[label] = H.response_fractions(pd.read_parquet(ARCHIVE / folder / f'{variant}_acute_off_responses.parquet')).reindex(odors)
        except FileNotFoundError:
            pass
    fr = pd.DataFrame(fr); fr['data_TNT'] = data_po['frac_tnt'].reindex(odors)
    per_odor = per_odor.merge(fr.rename_axis('odor').reset_index().melt(id_vars='odor', var_name='model', value_name='frac_off'), how='left')
    per_odor.to_csv(res_dir / 'per_odor.csv', index=False)

    # 3. glomerulus co-drive: per-glomerulus PN rate per odor (base model), above-threshold counts per pair
    from al_analysis.mb_model import fit_mb_model, get_time_index, get_sim_var
    init = e3_bouton_params()
    _, _, wPNKC, pd0 = fit_mb_model(orn, fixed_thr=init['fixed_thr'], wAPLKC=init['wAPLKC_scale'],
        wKCAPL=init['wKCAPL_scale'], wAPLPN=init['wAPLPN_scale'], wPNAPL=init['wPNAPL_scale'], **BOUTON,
        make_plots=False, silent=True, return_olfsysm_vars=True)
    mp, rv = pd0['mp'], pd0['rv']
    t = np.asarray(get_time_index(mp, rv)); win = (t >= 0.0) & (t <= 0.5)
    pn = get_sim_var(rv.pn, 'pn_sims')
    gloms = list(dict.fromkeys(wPNKC.columns.get_level_values('glomerulus')))
    rate = pd.DataFrame({H.odor_abbrev(c): np.asarray(pn[i])[:, win].mean(axis=1) for i, c in enumerate(orn.columns)}, index=gloms)
    rate = rate[odors]
    driven = [g for g in gloms if g in set(orn.index)]
    rate_d = rate.loc[driven]
    rate.to_csv(res_dir / 'glomerulus_rates_per_odor.csv')
    orn_n = orn.reindex(driven)[orn.columns].clip(lower=0)
    orn_n.columns = [H.odor_abbrev(c) for c in orn_n.columns]
    orn_n = orn_n / orn_n.max(axis=1).replace(0, np.nan).to_numpy()[:, None]
    co = []
    thr_ex = {lab: off[off.model == lab].set_index(['odor_a', 'odor_b']).deficit for lab in ('thr40', 'thr70', 'thr70_k2_kckc100') if lab in set(off.model)}
    for a, b in combinations(odors, 2):
        key = (a, b) if (a, b) in thr_ex.get('thr70', pd.Series(dtype=float)).index else (b, a)
        r = dict(odor_a=a, odor_b=b, block=H.block_label(a, b),
            involves_subst=int(a in SUBST or b in SUBST),
            orn_codrive=float(np.minimum(orn_n[a], orn_n[b]).sum()),
            n_glom_both_above40=int(((rate_d[a] > 40) & (rate_d[b] > 40)).sum()),
            n_glom_both_above70=int(((rate_d[a] > 70) & (rate_d[b] > 70)).sum()),
            n_glom_one_above70=int(((rate_d[a] > 70) ^ (rate_d[b] > 70)).sum()),
            top_codriven_gloms=','.join(np.minimum(rate_d[a], rate_d[b]).sort_values(ascending=False).head(3).index))
        for lab, s in thr_ex.items():
            r[f'deficit_{lab}'] = float(s.get(key, np.nan))
        co.append(r)
    co = pd.DataFrame(co)
    co.to_csv(res_dir / 'pair_codrive.csv', index=False)
    corr = {}
    for lab in thr_ex:
        d = co.dropna(subset=[f'deficit_{lab}'])
        corr[lab] = dict(
            vs_n_both_above70=float(stats.spearmanr(d[f'deficit_{lab}'], d.n_glom_both_above70)[0]),
            vs_n_both_above40=float(stats.spearmanr(d[f'deficit_{lab}'], d.n_glom_both_above40)[0]),
            vs_orn_codrive=float(stats.spearmanr(d[f'deficit_{lab}'], d.orn_codrive)[0]),
            subst_vs_not=float(d[d.involves_subst == 1][f'deficit_{lab}'].mean() - d[d.involves_subst == 0][f'deficit_{lab}'].mean()),
        )
    (res_dir / 'deficit_correlations.json').write_text(json.dumps(corr, indent=1))

    # summary print
    pd.set_option('display.width', 220)
    print('mean deficit (model APL-off excess - TNT excess; + = model shares too little) by group pair:')
    print(grp.pivot(index='block', columns='model', values='deficit').round(3).to_string())
    print('\nby substituted-odor involvement:'); print(sub.pivot(index='involves_subst', columns='model', values='deficit').round(3).to_string())
    print('\nper-odor deficit, thr70 model:'); print(per_odor[per_odor.model == 'thr70'].set_index('odor')[['group', 'deficit', 'model_excess', 'data_excess', 'frac_off']].round(3).to_string())
    print('\ndeficit correlations:', json.dumps(corr, indent=0))
    make_figure(pairs, grp, per_odor, co, res_dir)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E21_apl_off_sharing.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        models=models, data_table='results/E7_confound_controls/B_pairs_graded.csv (graded recipe)', approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('done ->', res_dir)


def make_figure(pairs, grp, per_odor, co, res_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    off = pairs[pairs.condition == 'acute_off']
    labels = [m for m in ('base', 'thr70', 'thr70_k2_kckc100', 'fit_k1', 'fit_k2') if m in set(off.model)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    colors = {'fruity-small_ester': '#d9480f', 'within': '#2f855a'}
    for ax, lab in zip(axes[0], labels[:3]):
        m = off[off.model == lab]
        c = [colors.get(b, '#2b6cb0') for b in m.block]
        ax.scatter(m.data_excess, m.model_excess, s=14, c=c, alpha=0.8)
        lo, hi = min(m.data_excess.min(), m.model_excess.min()) - 0.02, max(m.data_excess.max(), m.model_excess.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], ':', color='0.5', lw=1)
        ax.set_xlabel('TNT data: excess sharing per pair'); ax.set_ylabel('model APL off: excess sharing')
        ax.set_title(f'{lab}: per-pair excess, model vs data (orange = block, green = within-group)', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    ax = axes[1, 0]
    piv = grp[grp.model.isin(labels)].pivot(index='block', columns='model', values='deficit')[labels]
    im = ax.imshow(piv.to_numpy(), cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_title('mean deficit by group pair (+ = model shares too little)', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[1, 1]
    po = per_odor[per_odor.model == 'thr70'].set_index('odor')
    x = np.arange(len(po)); ax.bar(x - 0.2, po.data_excess, 0.4, color='0.6', label='TNT data'); ax.bar(x + 0.2, po.model_excess, 0.4, color='#2b6cb0', label='thr70 model, APL off')
    ax.set_xticks(x); ax.set_xticklabels(po.index, rotation=90, fontsize=7); ax.set_title("per-odor mean excess over the odor's 15 pairs", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax = axes[1, 2]
    if 'deficit_thr70' in co:
        d = co.dropna(subset=['deficit_thr70'])
        ax.scatter(d.n_glom_both_above70 + np.random.default_rng(0).uniform(-0.15, 0.15, len(d)), d.deficit_thr70, s=14,
            c=['#d9480f' if s else '#2b6cb0' for s in d.involves_subst], alpha=0.8)
        ax.set_xlabel('glomeruli driven above 70 Hz by both odors (base model)'); ax.set_ylabel('deficit, thr70 model (APL off)')
        ax.set_title('deficit vs co-driven glomeruli (orange = pair involves paa / euc / -aPine)', fontsize=9)
        ax.axhline(0, color='0.5', lw=1, ls=':'); ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('E21: why the APL-blocked model code shares too little among non-ester pairs', fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / 'E21_apl_off_sharing.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
