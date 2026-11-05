#!/usr/bin/env python3
"""The summary report's background figure: the three 16 x 16 odor-odor distance matrices (ORN
input, KCs in control flies, KCs with APL output blocked (TNT)), each KC matrix against the
ORN input pair by pair, and control against APL output blocked (TNT) with the monotone
(isotonic) fit whose residual defines the fruity x small-ester block.

Wording rules (CLAUDE.md, "Summary reports"): genotype labels are "control flies" and "APL
output blocked (TNT)" everywhere; matrix panel titles name only what is shown and the
distance function appears once, in the colorbar label, by name; long titles are split over
two lines. Writes results/summary_background/summary_background.png and merges the numbers it
quotes into background_numbers.json there.
Usage: ./venv/bin/python summary_background_figure.py
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

import kc_remapping_harness as H

OUT = H.shared_dir() / 'results' / 'summary_background'


def square(pairs: pd.Series, odors) -> pd.DataFrame:
    m = pd.DataFrame(0.0, index=odors, columns=odors)
    for a, b in combinations(odors, 2):
        v = pairs.get((a, b), pairs.get((b, a), np.nan))
        m.loc[a, b] = m.loc[b, a] = v
    return m


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    odors = list(H.APL16_ODORS)
    orn = H.orn16_consensus_pairs()
    ctrl = H.yang_mean_rdm('TNT3c', 'cosine')
    tnt = H.yang_mean_rdm('TNT', 'cosine')
    orn, ctrl = H.align_pairs(orn, ctrl); _, tnt = H.align_pairs(ctrl, tnt)
    idx = ctrl.index
    within = pd.Series([H.ODOR_GROUPS.get(a, a) == H.ODOR_GROUPS.get(b, b) for a, b in idx], index=idx)
    block = pd.Series([H.block_label(a, b) == H.FRUITY_SMALL_BLOCK for a, b in idx], index=idx)
    iso_resid, iso = H.isotonic_resid(ctrl, tnt)
    lin_resid, lin = H.linear_resid(ctrl, tnt)
    nums = dict(spearman_ctrl_orn=H.spearman(ctrl, orn), spearman_tnt_orn=H.spearman(tnt, orn), spearman_ctrl_tnt=H.spearman(ctrl, tnt),
        iso_r2=iso['r2'], block=float(iso_resid[block].mean()), lin_r2=lin['r2'], block_linear=float(lin_resid[block].mean()),
        mean_orn=float(orn.mean()), mean_ctrl=float(ctrl.mean()), mean_tnt=float(tnt.mean()))
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / 'background_numbers.json'
    old = json.loads(jp.read_text()) if jp.exists() else {}
    old.update(nums); jp.write_text(json.dumps(old, indent=1) + '\n')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.2))
    mats = [(square(orn, odors), 'ORN input', 'cosine distance (1 - cosine similarity) across glomerular responses'),
        (square(ctrl, odors), 'KCs, control flies', 'cosine distance (1 - cosine similarity) across KC responses'),
        (square(tnt, odors), 'KCs, APL output blocked (TNT)', 'cosine distance (1 - cosine similarity) across KC responses')]
    for ax, (m, title, cbar) in zip(axes[0], mats):
        im = ax.imshow(m.to_numpy(), cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(16)); ax.set_xticklabels(odors, rotation=90, fontsize=8); ax.set_yticks(range(16)); ax.set_yticklabels(odors, fontsize=8)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(cbar, fontsize=8)
    def scatter(ax, x, y, xlabel, ylabel, title):
        ax.plot([0, 1], [0, 1], ls='--', color='0.8', lw=1, zorder=0)
        ax.scatter(x[~within], y[~within], s=16, color='#5c6370', label='between-group pairs')
        ax.scatter(x[within], y[within], s=16, color='#1f77ff', label='within-group pairs')
        ax.scatter(x[block], y[block], s=42, facecolors='none', edgecolors='#d9480f', lw=1.4, label='fruity x small-ester pairs')
        ax.set_xlim(0, 1.03); ax.set_ylim(0, 1.03); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11); ax.legend(fontsize=8, loc='lower right')
        ax.spines[['top', 'right']].set_visible(False)
    scatter(axes[1, 0], orn, ctrl, 'ORN input: cosine distance', 'KCs, control flies: cosine distance',
        f"KCs, control flies vs ORN input\n(rank agreement {nums['spearman_ctrl_orn']:.2f})")
    scatter(axes[1, 1], orn, tnt, 'ORN input: cosine distance', 'KCs, APL output blocked (TNT): cosine distance',
        f"KCs, APL output blocked (TNT) vs ORN input\n(rank agreement {nums['spearman_tnt_orn']:.2f})")
    ax = axes[1, 2]
    ax.plot([0, 1], [0, 1], ls='--', color='0.8', lw=1, zorder=0)
    ax.scatter(tnt[~block], ctrl[~block], s=12, color='#5c6370', label='pairs')
    ax.scatter(tnt[block], ctrl[block], s=42, facecolors='none', edgecolors='#d9480f', lw=1.4, label='fruity x small-ester pairs')
    order = np.argsort(tnt.to_numpy()); fit = (ctrl - iso_resid).to_numpy()
    ax.plot(tnt.to_numpy()[order], fit[order], color='#333333', lw=1.4, label='monotone (isotonic) fit')
    ax.set_xlim(0, 1.03); ax.set_ylim(0, 1.03); ax.set_xlabel('KCs, APL output blocked (TNT): cosine distance'); ax.set_ylabel('KCs, control flies: cosine distance')
    ax.set_title(f"KCs, control flies vs APL output blocked (TNT)\n(monotone fit: isotonic R2 {nums['iso_r2']:.2f}; fruity x small-ester block {nums['block']:+.3f})", fontsize=11)
    ax.legend(fontsize=8, loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle("Yang's APL16 KC data: the ORN->KC remapping in control flies and what blocking APL output (TNT) does to it", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / 'summary_background.png', dpi=130); plt.close(fig)
    print({k: round(v, 3) for k, v in nums.items()})


if __name__ == '__main__':
    main()
