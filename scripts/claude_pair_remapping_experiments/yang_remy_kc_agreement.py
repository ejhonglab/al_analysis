#!/usr/bin/env python3
"""How Yang's KC odor-odor distances agree with Remy's megamat KC data on the 12 shared odors.

Yang's APL16 KC data (control flies and flies with APL output blocked (TNT); cosine distance
between trial-mean KC population responses, per recording, averaged over recordings) and
Remy's wild-type megamat KC data (correlation distance, 1 - Pearson, between trial-mean
responses across KC somata: `kc_remy_combo`, 18 flies, and the newer 4-fly soma set; and
across KC claws, 4 flies) share 12 odors, 66 odor pairs (`H.reference_pairs()`, the report's
shared12_pairs.csv). This script measures how well the two labs' KC matrices agree on those
pairs, so the summary can say what "structure vs wild-type somata" is measured against.

Statistics (all on the 66 pairs): rho on odor-odor distances and Pearson r between each Yang
matrix and each Remy matrix (and the ORN input, 1 - Pearson across the 36 glomeruli imaged in
both ORN panels); the split-half measurement ceiling of each Yang matrix on these pairs; each
Yang recording's rho with Remy's somata against its rho with the mean of the other Yang
recordings of its genotype (leave-one-out); 90 % bootstrap intervals over recordings; the
mean distance of the fruity x small-ester block pairs and of the within-group pairs in each
matrix; the residual-structure agreement (`compare_to_references`); the pairs whose rank
differs most between Yang's control matrix and Remy's somata.

Outputs: <shared>/results/yang_remy_agreement/ (agreement_stats.csv, per_recording.csv,
pair_table.csv, disagreements.csv, yang_remy_agreement.png, provenance.json).
Usage: ./venv/bin/python yang_remy_kc_agreement.py
`agreement()` is imported by data_targets.py for the registry.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H

OUT_NAME = 'yang_remy_agreement'
# column in shared12_pairs.csv -> label used in tables and the figure
REMY = {'kc_soma': 'KC somata, wild type (Remy, 18 flies)', 'kc_soma_new': 'KC somata, wild type (Remy, 2022 flies)',
    'kc_claw': 'KC claws, wild type (Remy, 4 flies)'}
YANG = {'kc_ctrl': ('TNT3c', 'KCs, control flies (Yang)'), 'kc_tnt': ('TNT', 'KCs, APL output blocked (TNT) (Yang)')}


def pair_type(a: str, b: str) -> str:
    lab = H.block_label(a, b)
    return 'within group' if lab == 'within' else ('fruity x small-ester block' if lab == H.FRUITY_SMALL_BLOCK
        else 'other between-group')


def recordings_66(ref: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per-recording cosine distances of both genotypes, restricted to the 66 shared pairs."""
    out = {}
    for g in ('TNT3c', 'TNT'):
        rec = H.yang_recording_rdms(g, 'cosine')
        cols = [p for p in rec.columns if p in ref.index]
        out[g] = rec[cols]
    return out


def agreement(ref: pd.DataFrame | None = None, recs: Dict[str, pd.DataFrame] | None = None,
    n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Every scalar statistic of the comparison (the registry reads these).

    Keys (x in ctrl / tnt; r in soma / soma_2022 / claw / orn):
    rho_<x>_remy_<r>, pearson_<x>_remy_<r> (orn without the 'remy_'); rho_remy_soma_orn,
    rho_remy_soma_claw, rho_remy_soma_soma_2022; <x>_ceiling_66, <x>_split_half_66;
    <x>_rec_rho_remy_soma (mean over recordings of one recording's rho with Remy's somata),
    <x>_rec_rho_loo (mean of one recording's rho with the mean of the other recordings);
    <x>_mean_66, remy_<r>_mean_66; <x>_block_mean_66, remy_soma_block_mean_66,
    <x>_within_mean_66, remy_soma_within_mean_66; rho_<x>_remy_soma_lo / _hi (90 % bootstrap
    over recordings).
    """
    ref = H.reference_pairs() if ref is None else ref
    recs = recordings_66(ref) if recs is None else recs
    out: Dict[str, float] = {}
    refs = {'soma': 'kc_soma', 'soma_2022': 'kc_soma_new', 'claw': 'kc_claw'}
    for ycol, (g, _) in YANG.items():
        x = 'ctrl' if ycol == 'kc_ctrl' else 'tnt'
        y = ref[ycol]
        for r, col in refs.items():
            out[f'rho_{x}_remy_{r}'] = H.spearman(y, ref[col])
            out[f'pearson_{x}_remy_{r}'] = H.pearson(y, ref[col])
        out[f'rho_{x}_orn_66'] = H.spearman(y, ref['orn'])
        out[f'{x}_mean_66'] = float(y.mean())
        types = pd.Series([pair_type(a, b) for a, b in ref.index], index=ref.index)
        out[f'{x}_block_mean_66'] = float(y[types == 'fruity x small-ester block'].mean())
        out[f'{x}_within_mean_66'] = float(y[types == 'within group'].mean())
        rec = recs[g]
        sh = H.split_half_reliability(rec)
        out[f'{x}_split_half_66'] = sh['split_half']
        out[f'{x}_ceiling_66'] = sh['ceiling']
        soma = ref['kc_soma']
        rr, loo = [], []
        for i in range(len(rec)):
            one = rec.iloc[i].reindex(ref.index)
            others = rec.drop(rec.index[i]).mean().reindex(ref.index)
            rr.append(H.spearman(one, soma))
            loo.append(H.spearman(one, others))
        out[f'{x}_rec_rho_remy_soma'] = float(np.mean(rr))
        out[f'{x}_rec_rho_loo'] = float(np.mean(loo))
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(rec), len(rec))
            boot.append(H.spearman(rec.iloc[idx].mean().reindex(ref.index), soma))
        out[f'rho_{x}_remy_soma_lo'], out[f'rho_{x}_remy_soma_hi'] = [float(v) for v in np.percentile(boot, [5, 95])]
    types = pd.Series([pair_type(a, b) for a, b in ref.index], index=ref.index)
    for r, col in refs.items():
        out[f'remy_{r}_mean_66'] = float(ref[col].mean())
    out['remy_soma_block_mean_66'] = float(ref['kc_soma'][types == 'fruity x small-ester block'].mean())
    out['remy_soma_within_mean_66'] = float(ref['kc_soma'][types == 'within group'].mean())
    out['orn_mean_66'] = float(ref['orn'].mean())
    out['rho_remy_soma_orn'] = H.spearman(ref['kc_soma'], ref['orn'])
    out['rho_remy_soma_claw'] = H.spearman(ref['kc_soma'], ref['kc_claw'])
    out['rho_remy_soma_soma_2022'] = H.spearman(ref['kc_soma'], ref['kc_soma_new'])
    out['rho_ctrl_tnt_66'] = H.spearman(ref['kc_ctrl'], ref['kc_tnt'])
    out['n_pairs_66'] = float(len(ref))
    return out


def per_recording_table(ref: pd.DataFrame, recs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for g, rec in recs.items():
        for i in range(len(rec)):
            one = rec.iloc[i].reindex(ref.index)
            others = rec.drop(rec.index[i]).mean().reindex(ref.index)
            rows.append(dict(genotype=g, recording='_'.join(str(v) for v in rec.index[i]),
                rho_remy_soma=H.spearman(one, ref['kc_soma']), rho_remy_soma_2022=H.spearman(one, ref['kc_soma_new']),
                rho_remy_claw=H.spearman(one, ref['kc_claw']), rho_other_recordings=H.spearman(one, others),
                rho_orn=H.spearman(one, ref['orn']), mean_66=float(one.mean())))
    return pd.DataFrame(rows)


def square(pairs: pd.Series, odors) -> pd.DataFrame:
    m = pd.DataFrame(0.0, index=odors, columns=odors)
    for (a, b), v in pairs.items():
        m.loc[a, b] = m.loc[b, a] = v
    return m


def make_figure(ref: pd.DataFrame, per_rec: pd.DataFrame, stats_: Dict[str, float], out_png) -> None:
    plt = H._plt()
    odors = [o for o in H.MEGAMAT12 if o in {x for p in ref.index for x in p}]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6))
    mats = [('kc_ctrl', 'KCs, control flies', 'cosine distance across KC responses'),
        ('kc_soma', 'KC somata, wild type (Remy)', 'correlation distance across KC responses'),
        ('kc_tnt', 'KCs, APL output blocked (TNT)', 'cosine distance across KC responses')]
    vmax = float(np.ceil(10 * max(ref[c].max() for c, *_ in mats)) / 10)
    for ax, (col, title, cbar) in zip(axes[0], mats):
        m = square(ref[col], odors)
        im = ax.imshow(m.to_numpy(), cmap='Blues', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(odors))); ax.set_yticks(range(len(odors)))
        ax.set_xticklabels(odors, rotation=90, fontsize=8); ax.set_yticklabels(odors, fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(cbar, fontsize=8)
    colors = {'within group': '#2a6fdb', 'fruity x small-ester block': '#d1495b', 'other between-group': '0.55'}
    types = pd.Series([pair_type(a, b) for a, b in ref.index], index=ref.index)
    for ax, ycol, xlab, key in ((axes[1][0], 'kc_ctrl', 'KCs, control flies: cosine distance', 'ctrl'),
            (axes[1][1], 'kc_tnt', 'KCs, APL output blocked (TNT): cosine distance', 'tnt')):
        for t, c in colors.items():
            sel = types == t
            ax.scatter(ref.loc[sel, ycol], ref.loc[sel, 'kc_soma'], s=22, color=c, label=t, alpha=0.85, edgecolor='none')
        lo, hi = 0.1, max(ref[ycol].max(), ref['kc_soma'].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], color='0.8', lw=1, zorder=0)
        ax.set_xlabel(xlab, fontsize=9); ax.set_ylabel('KC somata, wild type (Remy):\ncorrelation distance', fontsize=9)
        ax.set_title(f"66 shared pairs\n(ρ on odor-odor distances {stats_[f'rho_{key}_remy_soma']:.2f}, "
            f"Pearson {stats_[f'pearson_{key}_remy_soma']:.2f})", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axes[1][0].legend(fontsize=7, loc='upper left', frameon=False)
    ax = axes[1][2]
    xs = {'TNT3c': 0, 'TNT': 1}
    for g, x0 in xs.items():
        sub = per_rec[per_rec.genotype == g]
        for off, col, c, lab in ((-0.15, 'rho_remy_soma', '#d1495b', "with Remy's somata"),
                (0.15, 'rho_other_recordings', '#2a6fdb', 'with the mean of the other recordings')):
            ax.scatter(np.full(len(sub), x0 + off) + np.linspace(-0.05, 0.05, len(sub)), sub[col], s=18, color=c,
                alpha=0.8, label=lab if g == 'TNT3c' else None)
            ax.plot([x0 + off - 0.08, x0 + off + 0.08], [sub[col].mean()] * 2, color=c, lw=2)
    for g, x0 in xs.items():
        key = 'ctrl' if g == 'TNT3c' else 'tnt'
        ax.scatter([x0], [stats_[f'rho_{key}_remy_soma']], marker='_', s=300, color='k', zorder=5,
            label="recording mean with Remy's somata" if g == 'TNT3c' else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['control flies', 'APL output blocked (TNT)'], fontsize=9)
    ax.set_ylabel('ρ on odor-odor distances (66 pairs)', fontsize=9)
    ax.set_title("one Yang recording's agreement\nwith Remy's somata vs with its own genotype's other recordings", fontsize=9)
    ax.legend(fontsize=7, loc='lower left', frameon=False)
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    fig.suptitle("Yang's KC distances against Remy's wild-type megamat KC data on the 12 shared odors", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    res_dir = H.shared_dir() / 'results' / OUT_NAME
    res_dir.mkdir(exist_ok=True)
    ref = H.reference_pairs()
    recs = recordings_66(ref)
    st = agreement(ref, recs)
    pd.Series(st, name='value').rename_axis('statistic').to_csv(res_dir / 'agreement_stats.csv')
    per_rec = per_recording_table(ref, recs)
    per_rec.to_csv(res_dir / 'per_recording.csv', index=False)
    tab = ref[['kc_ctrl', 'kc_tnt', 'kc_soma', 'kc_soma_new', 'kc_claw', 'orn', 'vcf_hyp']].copy()
    tab.insert(0, 'pair_type', [pair_type(a, b) for a, b in tab.index])
    for c in ('kc_ctrl', 'kc_tnt', 'kc_soma'):
        tab[f'{c}_rank'] = tab[c].rank()
    tab['rank_diff_ctrl_minus_soma'] = tab['kc_ctrl_rank'] - tab['kc_soma_rank']
    tab.to_csv(res_dir / 'pair_table.csv')
    dis = tab.reindex(tab['rank_diff_ctrl_minus_soma'].abs().sort_values(ascending=False).index).head(10)
    dis.to_csv(res_dir / 'disagreements.csv')
    make_figure(ref, per_rec, st, res_dir / f'{OUT_NAME}.png')
    prov = dict(generated=datetime.now(timezone.utc).isoformat(timespec='seconds'), script='yang_remy_kc_agreement.py',
        host=subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
        al_analysis_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=H.HERE, capture_output=True, text=True).stdout.strip(),
        inputs=dict(shared12_pairs=str(H.APL_PAIR_DIR / 'shared12_pairs.csv')),
        remy_columns=REMY, n_recordings={g: int(len(r)) for g, r in recs.items()},
        measures=dict(yang='1 - cosine similarity between trial-mean KC population responses, per recording, mean over recordings',
            remy='1 - Pearson correlation between trial-mean responses across KC somata (kc_remy_combo) or claws'))
    (res_dir / 'provenance.json').write_text(json.dumps(prov, indent=1, default=str) + '\n')
    pd.set_option('display.width', 200)
    print(pd.Series(st).round(3).to_string())
    print(per_rec.round(3).to_string(index=False))
    print(dis[['pair_type', 'kc_ctrl', 'kc_soma', 'kc_tnt', 'orn', 'rank_diff_ctrl_minus_soma']].round(3).to_string())
    print('->', res_dir)


if __name__ == '__main__':
    main()
