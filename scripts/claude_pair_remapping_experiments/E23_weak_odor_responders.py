#!/usr/bin/env python3
"""E23: which KCs respond to the weak-input odors (ms, B-myr, euc, paa, -aPine, 1-8ol) in
Yang's recordings, and what could wire them (data analysis, no simulation).

Per recording and genotype (TNT = APL output blocked, TNT3c = control), a KC responds to an
odor if its trial-mean z-scored response exceeds 1. For every odor: its responders' breadth
(number of the 16 odors they respond to), the fraction that respond to that odor only, and
the overlap of its responders with each other odor's responders relative to the density
expectation (n_a n_b / N). For the weak odors, the overlap profile across the other 15 odors
is compared with the ORN response profile of the glomeruli that carry them in the model's
input (DL1 / VA7m / D for ms; DM1 / VM5v / DP1m for B-myr; etc.): if their responders are
KCs driven through one glomerulus, their co-response with odor X should track that
glomerulus's response to X. From the hemibrain wiring, the number of KCs with >= 1 / >= 2
claws from each odor's top glomeruli (the largest population a single-glomerulus input could
recruit) is compared with the data's response fraction.

Outputs: <shared>/results/E23_weak_odor_responders/ (responder_stats.csv, overlap_profiles.csv,
glomerulus_populations.csv, E23_weak_odor_responders.png, provenance.json).
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H

EXPERIMENT = 'E23_weak_odor_responders'
WEAK = ['ms', 'B-myr', 'euc', 'paa', '-aPine', '1-8ol']
TOP_GLOMS = {'ms': ['DL1', 'VA7m', 'D'], 'B-myr': ['DM1', 'VM5v', 'DP1m'], 'euc': ['DP1m', 'VC1', 'DC1'],
    'paa': ['DM1', 'VL2a', 'DM6'], '-aPine': ['VC1', 'DC1', 'VM7d'], '1-8ol': ['DC2', 'DM5', 'VM7d'],
    '2h': ['DM3', 'DM2', 'VM5v'], 'eb': ['DM2', 'VM7d', 'DM5']}


def main() -> None:
    res_dir = H.results_dir(EXPERIMENT)
    odors = list(H.APL16_ODORS)
    orn = H.apl16_orn_deltas(); orn.columns = [H.odor_abbrev(c) for c in orn.columns]
    stat_rows, prof_rows = [], []
    for g in ('TNT', 'TNT3c'):
        mats = H.yang_recording_matrices(g)
        for key, m in mats.items():
            R = (m[odors] > 1.0)
            N = len(R); n = R.sum()
            breadth = R.sum(axis=1)
            for o in odors:
                r = R[o]
                if r.sum() == 0:
                    continue
                b = breadth[r]
                stat_rows.append(dict(genotype=g, recording='_'.join(map(str, key)), odor=o, n_kcs=N, frac=float(r.mean()),
                    mean_breadth=float(b.mean()), frac_exclusive=float((b == 1).mean()), median_breadth=float(b.median())))
                for o2 in odors:
                    if o2 == o:
                        continue
                    obs = int((r & R[o2]).sum()); exp = float(n[o] * n[o2] / N) if N else np.nan
                    prof_rows.append(dict(genotype=g, recording='_'.join(map(str, key)), odor=o, other=o2, observed=obs,
                        expected=exp, ratio=(obs / exp if exp > 0 else np.nan)))
    st = pd.DataFrame(stat_rows); pr = pd.DataFrame(prof_rows)
    st.to_csv(res_dir / 'responder_stats_per_recording.csv', index=False)
    agg = st.groupby(['genotype', 'odor']).agg(frac=('frac', 'mean'), mean_breadth=('mean_breadth', 'mean'),
        frac_exclusive=('frac_exclusive', 'mean'), n_rec=('recording', 'nunique')).reset_index()
    agg['weak'] = agg.odor.isin(WEAK).astype(int)
    agg.to_csv(res_dir / 'responder_stats.csv', index=False)
    prof = pr.groupby(['genotype', 'odor', 'other']).agg(ratio=('ratio', 'mean'), observed=('observed', 'mean'),
        expected=('expected', 'mean')).reset_index()
    # ORN-side predictor: for each weak odor, the response of its top glomerulus to each other odor
    rows = []
    for g in ('TNT', 'TNT3c'):
        for o, gls in TOP_GLOMS.items():
            p = prof[(prof.genotype == g) & (prof.odor == o)].set_index('other')['ratio']
            for gl in gls:
                if gl not in orn.index:
                    continue
                drive = orn.loc[gl].reindex(p.index)
                rows.append(dict(genotype=g, odor=o, glomerulus=gl,
                    spearman_overlap_vs_glom_response=float(stats.spearmanr(p.to_numpy(), drive.to_numpy())[0]),
                    top_overlap_odors=','.join(p.sort_values(ascending=False).head(4).index),
                    top_glom_odors=','.join(drive.sort_values(ascending=False).head(4).index)))
            # also vs the odor's own whole ORN profile similarity (ORN pair distance)
            orn_sim = orn.corr(method='pearson')[o].reindex(p.index)
            rows.append(dict(genotype=g, odor=o, glomerulus='(ORN profile correlation)',
                spearman_overlap_vs_glom_response=float(stats.spearmanr(p.to_numpy(), orn_sim.to_numpy())[0]),
                top_overlap_odors=','.join(p.sort_values(ascending=False).head(4).index),
                top_glom_odors=','.join(orn_sim.sort_values(ascending=False).head(4).index)))
    prof.to_csv(res_dir / 'overlap_profiles.csv', index=False)
    pred = pd.DataFrame(rows); pred.to_csv(res_dir / 'overlap_vs_glomerulus_drive.csv', index=False)
    # connectome: KCs with claws from each odor's top glomeruli
    w = pd.read_parquet(H.shared_dir() / 'data' / 'model_outputs' / 'inputs' / 'wPNKC_bouton_base.parquet')
    gl = w.columns.get_level_values('glomerulus'); kc = w.index.get_level_values(H.KC_ID_LEVEL)
    n_kc = w.index.get_level_values(H.KC_ID_LEVEL).nunique()
    pop = []
    for o, gls in TOP_GLOMS.items():
        anyc = pd.Series(False, index=pd.Index(sorted(set(kc))))
        for gl_ in gls:
            cols = gl == gl_
            per_kc = pd.Series((w.loc[:, cols].to_numpy() > 0).any(axis=1), index=kc).groupby(level=0).sum()
            pop.append(dict(odor=o, glomerulus=gl_, boutons=int(cols.sum()), kcs_ge1_claw=int((per_kc >= 1).sum()),
                frac_ge1=float((per_kc >= 1).sum() / n_kc), kcs_ge2_claws=int((per_kc >= 2).sum()), frac_ge2=float((per_kc >= 2).sum() / n_kc)))
            anyc = anyc | (per_kc.reindex(anyc.index).fillna(0) >= 1)
        pop.append(dict(odor=o, glomerulus='any of the three', boutons=np.nan, kcs_ge1_claw=int(anyc.sum()), frac_ge1=float(anyc.mean()),
            kcs_ge2_claws=np.nan, frac_ge2=np.nan))
    pop = pd.DataFrame(pop); pop.to_csv(res_dir / 'glomerulus_populations.csv', index=False)
    pd.set_option('display.width', 220)
    print('responders (mean over recordings):'); print(agg.pivot(index='odor', columns='genotype', values=['frac', 'mean_breadth', 'frac_exclusive']).round(3).loc[odors].to_string())
    print('\noverlap profile vs glomerulus drive (Spearman over the other 15 odors):'); print(pred.round(2).to_string())
    print('\nconnectome populations:'); print(pop.round(3).to_string())
    make_figure(agg, pred, pop, res_dir)
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='E23_weak_odor_responders.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        threshold='trial-mean z > 1', top_gloms=TOP_GLOMS, approx_conc_note=H.APPROX_CONC_NOTE)
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    print('done ->', res_dir)


def make_figure(agg, pred, pop, res_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    odors = list(H.APL16_ODORS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (col, title) in zip(axes[:2], [('mean_breadth', "responders' mean breadth (odors responded to, of 16)"), ('frac_exclusive', 'fraction of responders responding to this odor only')]):
        for g, c, off in (('TNT3c', '#2b6cb0', -0.2), ('TNT', '#d9480f', 0.2)):
            a = agg[agg.genotype == g].set_index('odor').reindex(odors)
            ax.bar(np.arange(len(odors)) + off, a[col], 0.4, color=c, label=('control' if g == 'TNT3c' else 'APL blocked'))
        ax.set_xticks(range(len(odors))); ax.set_xticklabels(odors, rotation=90, fontsize=8); ax.set_title(title, fontsize=9)
        for i, o in enumerate(odors):
            if o in WEAK:
                ax.get_xticklabels()[i].set_color('#d9480f')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7, frameon=False)
    ax = axes[2]
    p = pred[(pred.genotype == 'TNT') & (pred.glomerulus != '(ORN profile correlation)')]
    x = np.arange(len(p)); ax.bar(x, p.spearman_overlap_vs_glom_response, color='#2b6cb0')
    ax.set_xticks(x); ax.set_xticklabels([f'{r.odor}: {r.glomerulus}' for r in p.itertuples()], rotation=90, fontsize=7)
    ax.axhline(0, color='0.5', lw=1); ax.set_title("APL blocked: rank agreement of an odor's responder overlap profile with its top glomerulus's ORN response profile", fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('E23: who responds to the weak-input odors in Yang\'s recordings (weak odors in orange)', fontsize=11)
    fig.tight_layout(); fig.savefig(res_dir / 'E23_weak_odor_responders.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    main()
