#!/usr/bin/env python3
"""E29: the hemibrain substrate check for the oracle glomeruli: are the eight glomeruli whose
claws the oracle gain (E27 / E28) strengthens (VA7m, DL1, D, DM1, DP1m, VC1, DC2, DM5)
under-represented in the model's calyx wiring?

Per glomerulus: uniglomerular PNs in the hemibrain (neurons_pn.csv) and their synapses inside
the calyx (ca_pre, per-ROI counts from neuPrint) vs the PNs, boutons, claws and PN->KC
synapses the model's bouton wiring carries (wPNKC_bouton_base and the rebuilt claw table of
check_bouton_claw_synapses.py); the fraction of each glomerulus's PNs and of its calyx output
synapses the model represents. The oracle set is compared with the other driven glomeruli by
rank and by a permutation test over random sets of eight. Writes the per-glomerulus table, the
comparison, a candidate per-glomerulus correction (hemibrain calyx output per glomerulus over
the model's PN->KC synapses, median-normalized) and a figure.

Outputs: <shared>/results/E29_oracle_substrate/. Usage: ./venv/bin/python E29_oracle_substrate.py
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import kc_remapping_harness as H
from E27_glomerulus_gain import ORACLE, TOP_GLOMS

EXPERIMENT = 'E29_oracle_substrate'
HEMIBRAIN = (H.HERE.parents[1] / 'al_analysis' / 'data' / 'hemibrain_neuprint').resolve()


def main() -> None:
    res_dir = H.results_dir(EXPERIMENT)
    orn = H.apl16_orn_deltas(); driven = set(orn.index)
    pn = pd.read_csv(HEMIBRAIN / 'neurons_pn.csv'); pn = pn[pn['uniglomerular'] == True]
    hb = pn.groupby('glomerulus').agg(hb_n_pn=('bodyId', 'nunique'), hb_ca_pre=('ca_pre', 'sum'), hb_ca_post=('ca_post', 'sum'), hb_lh_pre=('lh_pre', 'sum'), hb_pre_total=('pre', 'sum'))
    w = pd.read_parquet(H.shared_dir() / 'data' / 'model_outputs' / 'inputs' / 'wPNKC_bouton_base.parquet')
    cols = w.columns.to_frame(index=False)
    model = cols.groupby('glomerulus').agg(model_n_pn=('pn_id', 'nunique'), model_n_boutons=('bouton_id', 'count'))
    gl = w.columns.get_level_values('glomerulus')
    model['model_n_claws'] = pd.Series((w.to_numpy() > 0).sum(axis=0), index=gl).groupby(level=0).sum()
    conn = pd.read_csv(H.shared_dir() / 'results' / 'bouton_claw_synapses' / 'bouton_claw_connections.csv')
    syn = conn.groupby('glomerulus').agg(model_claw_synapses=('n_syn', 'sum'), model_syn_per_claw=('n_syn', 'mean'))
    t = hb.join(model, how='outer').join(syn, how='left')
    t['driven'] = t.index.isin(driven); t['oracle'] = t.index.isin(ORACLE)
    t['pn_fraction_in_model'] = t['model_n_pn'] / t['hb_n_pn']
    t['calyx_synapses_represented'] = t['model_claw_synapses'] / t['hb_ca_pre']
    t['boutons_per_hb_pn'] = t['model_n_boutons'] / t['hb_n_pn']
    t['claws_per_hb_ca_pre'] = t['model_n_claws'] / t['hb_ca_pre']
    t = t.sort_values('calyx_synapses_represented')
    t.to_csv(res_dir / 'glomerulus_substrate.csv')
    d = t[t['driven'] & t['hb_n_pn'].notna() & t['model_n_pn'].notna()]
    metrics = ['hb_n_pn', 'model_n_pn', 'pn_fraction_in_model', 'hb_ca_pre', 'model_claw_synapses', 'calyx_synapses_represented', 'model_n_boutons', 'boutons_per_hb_pn', 'model_syn_per_claw', 'claws_per_hb_ca_pre']
    rng = np.random.default_rng(0)
    comp = []
    others = d[~d['oracle']]; orc = d[d['oracle']]
    n_or = len(orc); idx = np.arange(len(d))
    for m in metrics:
        vals = d[m].to_numpy(dtype=float)
        obs = float(np.nanmean(vals[d['oracle'].to_numpy()]))
        null = np.array([np.nanmean(vals[rng.choice(idx, size=n_or, replace=False)]) for _ in range(20000)])
        comp.append(dict(metric=m, oracle_mean=obs, others_mean=float(np.nanmean(vals[~d['oracle'].to_numpy()])), oracle_median=float(np.nanmedian(vals[d['oracle'].to_numpy()])),
            others_median=float(np.nanmedian(vals[~d['oracle'].to_numpy()])), p_lower=float((null <= obs).mean()), p_upper=float((null >= obs).mean()),
            oracle_rank_mean=float(d[m].rank().loc[orc.index].mean()), n_driven=int(len(d))))
    comp = pd.DataFrame(comp); comp.to_csv(res_dir / 'oracle_vs_others.csv', index=False)
    # candidate correction: calyx output per glomerulus in the hemibrain over what the model carries, median-normalized over driven glomeruli
    corr = (d['hb_ca_pre'] / d['model_claw_synapses']); corr = corr / corr.median()
    corr.rename('gain_calyx_output').to_csv(res_dir / 'candidate_gain_by_glomerulus.csv')
    per_odor = {o: {g: dict(pn_fraction_in_model=float(t.loc[g, 'pn_fraction_in_model']), calyx_synapses_represented=float(t.loc[g, 'calyx_synapses_represented']), candidate_gain=float(corr.get(g, np.nan))) for g in gs if g in t.index} for o, gs in TOP_GLOMS.items()}
    (res_dir / 'per_weak_odor_top_glomeruli.json').write_text(json.dumps(per_odor, indent=1) + '\n')
    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, m, title in zip(axes, ['pn_fraction_in_model', 'calyx_synapses_represented', 'model_syn_per_claw'],
            ['PNs in the model wiring / PNs in the hemibrain', 'model PN->KC synapses / hemibrain calyx output synapses', 'synapses per claw (model)']):
        dd = d.sort_values(m); x = np.arange(len(dd))
        ax.bar(x, dd[m].to_numpy(), color=['#d9480f' if o else '#9aa5b1' for o in dd['oracle']])
        ax.set_xticks(x); ax.set_xticklabels(dd.index, rotation=90, fontsize=7); ax.set_title(title, fontsize=9); ax.spines[['top', 'right']].set_visible(False)
    axes[0].text(0.02, 0.95, 'orange: the oracle glomeruli', transform=axes[0].transAxes, fontsize=8, va='top')
    fig.suptitle('E29: the oracle glomeruli in the hemibrain vs the model wiring (36 driven glomeruli)', fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / 'E29_oracle_substrate.png', dpi=130); plt.close(fig)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E29_oracle_substrate.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        hemibrain=json.loads((HEMIBRAIN / 'provenance.json').read_text()), oracle=ORACLE, top_glomeruli=TOP_GLOMS, permutations=20000)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    pd.set_option('display.width', 220)
    print(t.loc[[g for g in ORACLE if g in t.index], ['hb_n_pn', 'model_n_pn', 'pn_fraction_in_model', 'hb_ca_pre', 'model_claw_synapses', 'calyx_synapses_represented', 'model_n_boutons', 'model_syn_per_claw']].round(2).to_string())
    print('driven-glomeruli medians:', d[['pn_fraction_in_model', 'calyx_synapses_represented', 'model_syn_per_claw']].median().round(2).to_dict())
    print(comp[['metric', 'oracle_mean', 'others_mean', 'p_lower', 'p_upper', 'oracle_rank_mean']].round(3).to_string())
    print('candidate gain (oracle):', corr.reindex([g for g in ORACLE if g in corr.index]).round(2).to_dict())
    print('done ->', res_dir)


if __name__ == '__main__':
    main()
