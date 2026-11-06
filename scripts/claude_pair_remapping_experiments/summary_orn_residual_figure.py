#!/usr/bin/env python3
"""Example figure for the summary report's footnote on "structure beyond the ORN input": the
least-squares line of the control flies' KC distances on the mean ORN distances over the 66
megamat pairs with the residuals drawn, then the two comparisons the statistic makes (the control
residual against the same residual of the wild-type KC somata; the residual with APL output
blocked (TNT) against the natural-source distances). Same computation as
`kc_remapping_harness.compare_to_references`. Writes results/summary_background/
orn_residual_example.png.
Usage: ./venv/bin/python summary_orn_residual_figure.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import kc_remapping_harness as H

OUT = H.shared_dir() / 'results' / 'summary_background' / 'orn_residual_example.png'


def main() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref = H.reference_pairs()
    res_c, fit = H.linear_resid(ref.kc_ctrl, ref.orn)
    res_t, _ = H.linear_resid(ref.kc_tnt, ref.orn)
    res_wt, _ = H.linear_resid(ref.kc_soma, ref.orn)
    filled = ref.vcf_raw.isna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    ax = axes[0]
    x, y = ref.orn, ref.kc_ctrl
    pred = fit['intercept'] + fit['slope'] * x
    for p in ref.index:
        ax.plot([x[p], x[p]], [pred[p], y[p]], color='#d9480f' if res_c[p] > 0 else '#2563eb', lw=0.8, alpha=0.6, zorder=1)
    ax.scatter(x, y, s=18, color='#374151', zorder=3)
    xs = np.linspace(x.min() - 0.02, x.max() + 0.02, 50)
    ax.plot(xs, fit['intercept'] + fit['slope'] * xs, color='black', lw=1.5, zorder=2, label='least-squares line')
    for p in res_c.abs().sort_values(ascending=False).index[:4]:
        ax.annotate(f'{p[0]}–{p[1]}', (x[p], y[p]), xytext=(4, 4 if res_c[p] > 0 else -10), textcoords='offset points', fontsize=7.5)
    ax.set_xlabel('ORNs: Pearson correlation distance (mean over flies)')
    ax.set_ylabel('KCs, control flies: cosine distance (mean over recordings)')
    ax.set_title(f"KCs, control flies vs ORNs, {len(ref)} megamat pairs\n(b={fit['slope']:.2f}, a={fit['intercept']:.2f}, R² {fit['r2']:.2f}; "
        'residuals: red above, blue below)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right', frameon=False)

    ax = axes[1]
    ax.axhline(0, color='0.85', lw=0.8); ax.axvline(0, color='0.85', lw=0.8)
    ax.scatter(res_c, res_wt, s=18, color='#374151')
    ax.set_xlabel('KCs, control flies: residual')
    ax.set_ylabel("KCs, wild-type somata (Remy's megamat data): residual")
    ax.set_title(f'structure vs wild-type somata\n(ρ={H.spearman(res_c, res_wt):.2f})', fontsize=10)

    ax = axes[2]
    ax.axvline(0, color='0.85', lw=0.8)
    ax.scatter(res_t[~filled], ref.vcf_hyp[~filled], s=18, color='#374151', label='pairs')
    ax.scatter(res_t[filled], ref.vcf_hyp[filled], s=30, facecolors='white', edgecolors='#374151', lw=1.2,
        label='raw natural-source value missing (embedding-filled)')
    ax.set_xlabel('KCs, APL output blocked (TNT): residual')
    ax.set_ylabel('natural-source distance (hyperbolic embedding)')
    ax.set_title(f'natural-source alignment\n(ρ={H.spearman(res_t, ref.vcf_hyp):.2f})', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower right', frameon=False)
    for a in axes:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
