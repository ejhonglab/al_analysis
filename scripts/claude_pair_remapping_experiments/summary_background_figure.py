#!/usr/bin/env python3
"""The summary report's background figure: the three 16 x 16 odor-odor matrices (ORN input,
KCs in control flies, KCs with APL output blocked (TNT)), each KC matrix against the ORN input
pair by pair, and control against APL output blocked (TNT) with the monotone (isotonic) fit
whose residual defines the fruity x small-ester block. Drawn twice:

- summary_background.png: the KC matrices as cosine distance (the program's primary measure).
  The ORN matrix is the report's, one minus the Pearson correlation across the 40 glomeruli
  that respond to at least one of the 16 odors (apl_pair_analysis/apl_analysis.py), and is
  labelled as such.
- summary_background_pearson.png: everything as Pearson correlation (r, not distance), the
  matrices in olfdb's blue-white-red Oklab colormap (white at r = 0, symmetric -1..1, diagonal
  masked); shown in a collapsible block under the first figure in summary reports.

Wording rules (CLAUDE.md, "Summary reports"): genotype labels are "control flies" and "APL
output blocked (TNT)" everywhere; matrix panel titles name only what is shown and the measure
appears once, in the colorbar label, by name; long titles are split over two lines. Writes
results/summary_background/{summary_background,summary_background_pearson}.png and merges the
numbers it quotes into background_numbers.json there.
Usage: ./venv/bin/python summary_background_figure.py
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

import kc_remapping_harness as H

OUT = H.shared_dir() / 'results' / 'summary_background'
BLOCK_COLOR = '#d9480f'


def square(pairs: pd.Series, odors, diag: float = 0.0) -> pd.DataFrame:
    m = pd.DataFrame(diag, index=odors, columns=odors)
    for a, b in combinations(odors, 2):
        v = pairs.get((a, b), pairs.get((b, a), np.nan))
        m.loc[a, b] = m.loc[b, a] = v
    return m


def stats_for(orn: pd.Series, ctrl: pd.Series, tnt: pd.Series, block: pd.Series) -> dict:
    """Rank agreements, the monotone fit and the block, computed on distances (1 - similarity)."""
    iso_resid, iso = H.isotonic_resid(ctrl, tnt)
    lin_resid, lin = H.linear_resid(ctrl, tnt)
    return dict(spearman_ctrl_orn=H.spearman(ctrl, orn), spearman_tnt_orn=H.spearman(tnt, orn),
        spearman_ctrl_tnt=H.spearman(ctrl, tnt), iso_r2=iso['r2'], block=float(iso_resid[block].mean()),
        lin_r2=lin['r2'], block_linear=float(lin_resid[block].mean()), mean_orn=float(orn.mean()),
        mean_ctrl=float(ctrl.mean()), mean_tnt=float(tnt.mean())), iso_resid


def draw(orn, ctrl, tnt, within, block, iso_resid, nums, *, measure: str, out_name: str) -> None:
    """measure 'cosine': matrices and axes in distance (KCs cosine, ORNs correlation distance);
    'pearson': matrices and axes in Pearson correlation r = 1 - distance."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    odors = list(H.APL16_ODORS)
    pearson = measure == 'pearson'
    tf = (lambda s: 1.0 - s) if pearson else (lambda s: s)
    o, c, t = tf(orn), tf(ctrl), tf(tnt)
    fit = tf(ctrl - iso_resid)
    if pearson:
        orn_lab, kc_lab = 'Pearson correlation across glomerular responses', 'Pearson correlation across KC responses'
        ax_lab = 'Pearson correlation'
        cmap = H.oklab_diverging_cmap(); vmin, vmax = -1.0, 1.0
        lo = min(o.min(), c.min(), t.min()); lo = min(0.0, np.floor(lo * 10) / 10)
        lim = (lo - 0.03, 1.03)
    else:
        orn_lab, kc_lab = 'Pearson correlation distance across glomerular responses', 'cosine distance across KC responses'
        ax_lab = None
        cmap = 'Blues'; vmin, vmax = 0.0, 1.0
        lim = (0.0, 1.03)
        # ORN correlation distances exceed 1 for anti-correlated pairs (up to ~1.13)
        orn_vmax = max(1.0, float(np.ceil(o.max() * 10) / 10))
        orn_lim = (0.0, max(1.03, float(o.max()) + 0.03))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.2))
    mats = [(square(o, odors, np.nan if pearson else 0.0), 'ORNs', orn_lab),
        (square(c, odors, np.nan if pearson else 0.0), 'KCs, control flies', kc_lab),
        (square(t, odors, np.nan if pearson else 0.0), 'KCs, APL output blocked (TNT)', kc_lab)]
    for k, (ax, (m, title, cbar)) in enumerate(zip(axes[0], mats)):
        if pearson:
            ax.set_facecolor('#e5e7eb')
        im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=vmin, vmax=(orn_vmax if (k == 0 and not pearson) else vmax))
        ax.set_xticks(range(16)); ax.set_xticklabels(odors, rotation=90, fontsize=8); ax.set_yticks(range(16)); ax.set_yticklabels(odors, fontsize=8)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(cbar, fontsize=8)
    orn_ax = 'ORNs: Pearson correlation' if pearson else 'ORNs: Pearson correlation distance'
    kc_ax = (lambda g: f'KCs, {g}: ' + ('Pearson correlation' if pearson else 'cosine distance'))
    corner = 'upper left' if pearson else 'lower right'

    def scatter(ax, x, y, xlabel, ylabel, title):
        xlim = lim if pearson else orn_lim
        ax.plot([0, max(xlim[1], lim[1])], [0, max(xlim[1], lim[1])], ls='--', color='0.8', lw=1, zorder=0)
        ax.scatter(x[~within], y[~within], s=16, color='#5c6370', label='between-group pairs')
        ax.scatter(x[within], y[within], s=16, color='#1f77ff', label='within-group pairs')
        ax.scatter(x[block], y[block], s=42, facecolors='none', edgecolors=BLOCK_COLOR, lw=1.4, label='fruity x small-ester pairs')
        ax.set_xlim(xlim); ax.set_ylim(lim); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11); ax.legend(fontsize=8, loc=corner)
        ax.spines[['top', 'right']].set_visible(False)
    scatter(axes[1, 0], o, c, orn_ax, kc_ax('control flies'),
        f"KCs, control flies vs ORNs\n(ρ={nums['spearman_ctrl_orn']:.2f})")
    scatter(axes[1, 1], o, t, orn_ax, kc_ax('APL output blocked (TNT)'),
        f"KCs, APL output blocked (TNT) vs ORNs\n(ρ={nums['spearman_tnt_orn']:.2f})")
    ax = axes[1, 2]
    ax.plot(lim, lim, ls='--', color='0.8', lw=1, zorder=0)
    ax.scatter(t[~block], c[~block], s=12, color='#5c6370', label='pairs')
    ax.scatter(t[block], c[block], s=42, facecolors='none', edgecolors=BLOCK_COLOR, lw=1.4, label='fruity x small-ester pairs')
    order = np.argsort(t.to_numpy())
    ax.plot(t.to_numpy()[order], fit.to_numpy()[order], color='#333333', lw=1.4, label='monotone (isotonic) fit')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel(kc_ax('APL output blocked (TNT)')); ax.set_ylabel(kc_ax('control flies'))
    where = 'below' if pearson else 'above'
    ax.set_title(f"KCs, control flies vs APL output blocked (TNT)\n(isotonic R2 {nums['iso_r2']:.2f}; fruity x small-ester pairs "
        f"{abs(nums['block']):.3f} {where} the fit)", fontsize=11)
    ax.legend(fontsize=8, loc=corner); ax.spines[['top', 'right']].set_visible(False)
    what = 'Pearson correlation' if pearson else 'KCs: cosine distance'
    fig.suptitle(f"Yang's APL16 KC data: the ORN->KC remapping in control flies and what blocking APL output (TNT) does to it ({what})", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / out_name, dpi=130); plt.close(fig)


def main() -> None:
    odors = list(H.APL16_ODORS)
    orn = H.orn16_consensus_pairs()
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / 'background_numbers.json'
    old = json.loads(jp.read_text()) if jp.exists() else {}
    for measure, out_name, prefix in (('cosine', 'summary_background.png', ''), ('pearson', 'summary_background_pearson.png', 'pearson_')):
        ctrl = H.yang_mean_rdm('TNT3c', measure)
        tnt = H.yang_mean_rdm('TNT', measure)
        o, ctrl = H.align_pairs(orn, ctrl); _, tnt = H.align_pairs(ctrl, tnt)
        idx = ctrl.index
        within = pd.Series([H.ODOR_GROUPS.get(a, a) == H.ODOR_GROUPS.get(b, b) for a, b in idx], index=idx)
        block = pd.Series([H.block_label(a, b) == H.FRUITY_SMALL_BLOCK for a, b in idx], index=idx)
        nums, iso_resid = stats_for(o, ctrl, tnt, block)
        draw(o, ctrl, tnt, within, block, iso_resid, nums, measure=measure, out_name=out_name)
        old.update({prefix + k: v for k, v in nums.items()})
        print(measure, {k: round(v, 3) for k, v in nums.items()})
    old.update(rate_ratios())
    jp.write_text(json.dumps(old, indent=1) + '\n')


def rate_ratios(n_boot: int = 5000) -> dict:
    """The data's APL-off (TNT) / APL-on (control) response-rate ratio, two ways, over the 16
    odors: the fraction of KCs responding (trial-mean > 1; the definition model rate ratios use)
    and the mean response size (trial-mean responses with negative values set to 0, over all
    KCs and odors), each with a 90 % bootstrap interval over recordings."""
    rng = np.random.default_rng(0)
    frac = {g: H.yang_response_fractions(g)[H.APL16_ODORS].mean(axis=1) for g in ('TNT3c', 'TNT')}
    tm = H.yang_trialmeans()
    tm = tm[tm.odor.isin(H.APL16_ODORS)].assign(v=lambda d: d.value.clip(lower=0))
    size = {g: tm[tm.genotype == g].groupby(['date', 'fly_num', 'hemibrain']).v.mean() for g in ('TNT3c', 'TNT')}
    out = {}
    for name, per_rec, pooled in (('fraction', frac, None),
            ('mean_response', size, {g: float(tm[tm.genotype == g].v.mean()) for g in ('TNT3c', 'TNT')})):
        point = (pooled['TNT'] / pooled['TNT3c']) if pooled else float(per_rec['TNT'].mean() / per_rec['TNT3c'].mean())
        boots = [per_rec['TNT'].iloc[rng.integers(0, len(per_rec['TNT']), len(per_rec['TNT']))].mean()
            / per_rec['TNT3c'].iloc[rng.integers(0, len(per_rec['TNT3c']), len(per_rec['TNT3c']))].mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [5, 95])
        out[f'rate_ratio_{name}'] = float(point)
        out[f'rate_ratio_{name}_ci90'] = [float(lo), float(hi)]
    return out


if __name__ == '__main__':
    main()
