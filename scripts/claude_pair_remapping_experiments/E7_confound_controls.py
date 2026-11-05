#!/usr/bin/env python3
"""E7: confound controls and noise floor for the control (TNT3c) vs APL>TNT KC matrices.

Data-only (Yang's APL16 per-recording KC data, via `kc_remapping_harness`). Four parts:

A. Noise floor. Bootstrap over recordings and over flies of every pair-level statistic
   the report used (and the harness computes); per-pair bootstrap SD of the isotonic
   residual (which pairs deviate from the monotone control-vs-TNT relation beyond
   noise); split-half comparisons within each genotype (what the between-genotype
   statistics look like when both halves are the same condition).
B. Density (response-rate) null. For each recording, the cosine similarity expected if
   KC identities were shuffled independently within each odor (per-odor response
   distributions kept): is the control-vs-TNT structure (rank order, monotone relation,
   fruity x small-ester block, per-odor APL effect) still there in the similarity in
   excess of that null, and does the per-odor APL effect just track per-odor density
   change?
C. Density-matched thresholds. Binarize / threshold TNT more strictly (or control more
   loosely) so both genotypes have the same mean response fraction, and re-score.
D. Pipeline variants. Graded / binary / thresholded values x all / responsive cells x
   cosine / Pearson, for both genotypes: how much do the statistics depend on the RDM
   recipe (so model outputs can be scored with a recipe whose sensitivity is known)?

Compact outputs (CSV tables, PDF/PNG figures, summary.json) go to the shared
results/E7_confound_controls/; nothing large is produced.

Usage: ./venv/bin/python E7_confound_controls.py [--n-boot 2000] [--n-split 1000] [--quick]
"""
import argparse
from functools import partial
import json
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import kc_remapping_harness as H


EXP = 'E7_confound_controls'
GENOS = ('TNT3c', 'TNT')
CTRL, TNT = GENOS
COLORS = {CTRL: H.CONDITION_COLORS['control'], TNT: H.CONDITION_COLORS['TNT'],
    'between': '#444444', 'null': '#999999', 'block': '#111111',
}
LABELS = {CTRL: 'control (TNT3c)', TNT: 'APL>TNT'}

# statistics tracked everywhere (from compare_on_off and compare_to_references)
KEY_ON_OFF: List[str] = ['mean_on', 'mean_off', 'spearman_on_off', 'lin_r2', 'iso_r2',
    'lin_resid_sd', 'iso_resid_sd', 'block_fruity_small_iso', 'block_other_between_iso',
    'block_fruity_small_lin', 'spearman_orn_on', 'spearman_orn_off', 'gap_closed',
    'apl_share_resid_var', 'spearman_resid_orn_on_off', 'spearman_apl_eff_vs_resid_orn_on',
    'spearman_per_odor_iso_vs_drive', 'spearman_per_odor_delta_vs_drive',
]
KEY_REF: List[str] = ['spearman_orn', 'spearman_kc_claw', 'spearman_kc_soma',
    'spearman_vcf_hyp', 'spearman_resid_vs_vcf_hyp', 'spearman_resid_vs_remap_wt',
    'spearman_model_hemibrain_wd20',
]
# the subset shown in the figures / headline tables
HEADLINE: List[str] = ['mean_on', 'mean_off', 'spearman_on_off', 'iso_r2',
    'block_fruity_small_iso', 'spearman_orn_on', 'spearman_orn_off', 'gap_closed',
    'spearman_per_odor_iso_vs_drive', 'TNT3c_spearman_resid_vs_vcf_hyp',
    'TNT_spearman_resid_vs_vcf_hyp', 'TNT3c_spearman_kc_claw', 'TNT_spearman_kc_claw',
]


def stats_fn(means: Dict[str, pd.Series], *, orn: pd.Series, drive: pd.Series,
    ref: Optional[pd.DataFrame], with_ref: bool = True) -> Dict[str, float]:
    """All tracked statistics for one {genotype: mean pair Series}."""
    s, _ = H.compare_on_off(means[CTRL], means[TNT], orn=orn, drive=drive)
    out = {k: s[k] for k in KEY_ON_OFF if k in s}
    if with_ref and ref is not None:
        for g in GENOS:
            r = H.compare_to_references(means[g], ref)
            for k in KEY_REF:
                out[f'{g}_{k}'] = r[k]
        for k in ('spearman_vcf_hyp', 'spearman_resid_vs_vcf_hyp', 'spearman_kc_claw'):
            out[f'diff_{k}_tnt_minus_ctrl'] = out[f'{TNT}_{k}'] - out[f'{CTRL}_{k}']
    return out


def boot_table(fn: Callable, recs: Dict[str, pd.DataFrame], *, n_boot: int, seed: int,
    unit: str = 'recording') -> pd.DataFrame:
    return H.bootstrap_recordings(fn, recs, n_boot=n_boot, seed=seed, unit=unit)


def flat_row(bt: pd.DataFrame, keys: List[str]) -> Dict[str, float]:
    """One flat dict (stat, stat_lo, stat_hi) from a bootstrap table."""
    row = {}
    for k in keys:
        if k in bt.index:
            row[k] = bt.loc[k, 'observed']
            row[f'{k}_lo'] = bt.loc[k, 'lo']
            row[f'{k}_hi'] = bt.loc[k, 'hi']
    return row


# ---------------------------------------------------------------------------
# A. noise floor
def per_pair_bootstrap(recs: Dict[str, pd.DataFrame], *, n_boot: int, seed: int,
    unit: str) -> pd.DataFrame:
    """Per-pair observed values and bootstrap SDs of the mean distances and of the
    isotonic / linear residuals of control on TNT."""
    rng = np.random.default_rng(seed)
    on0, off0 = H.align_pairs(recs[CTRL].mean(), recs[TNT].mean())
    iso0, _ = H.isotonic_resid(on0, off0)
    lin0, _ = H.linear_resid(on0, off0)
    on_d, off_d, iso_d, lin_d = [], [], [], []
    for _ in range(n_boot):
        m = H.resample_recording_means(recs, rng, unit=unit)
        on, off = H.align_pairs(m[CTRL], m[TNT])
        on = on.reindex(on0.index)
        off = off.reindex(on0.index)
        iso, _ = H.isotonic_resid(on, off)
        lin, _ = H.linear_resid(on, off)
        on_d.append(on)
        off_d.append(off)
        iso_d.append(iso.reindex(on0.index))
        lin_d.append(lin.reindex(on0.index))
    on_d, off_d = pd.DataFrame(on_d), pd.DataFrame(off_d)
    iso_d, lin_d = pd.DataFrame(iso_d), pd.DataFrame(lin_d)
    df = pd.DataFrame({'on': on0, 'off': off0, 'on_sd': on_d.std(), 'off_sd': off_d.std(),
        'iso_resid': iso0, 'iso_resid_sd': iso_d.std(),
        'iso_resid_lo': iso_d.quantile(0.05), 'iso_resid_hi': iso_d.quantile(0.95),
        'frac_boot_iso_resid_opposite_sign': (np.sign(iso_d) != np.sign(iso0)).mean(),
        'lin_resid': lin0, 'lin_resid_sd': lin_d.std(),
    })
    df['iso_z'] = df.iso_resid / df.iso_resid_sd
    df['block'] = [H.block_label(a, b) for a, b in df.index]
    return df


def split_half(recs: Dict[str, pd.DataFrame], *, n_split: int, seed: int, unit: str
    ) -> pd.DataFrame:
    """Statistics between two disjoint halves of the same genotype (within_*), and
    between a control half and a TNT half (between_halves), over random splits."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_split):
        halves = {}
        for g in GENOS:
            df = recs[g]
            if unit == 'fly':
                flies = df.index.droplevel('hemibrain').unique()
                perm = flies[rng.permutation(len(flies))]
                h = len(perm) // 2
                a = pd.concat([df.xs(f, level=('date', 'fly_num'), drop_level=False)
                    for f in perm[:h]])
                b = pd.concat([df.xs(f, level=('date', 'fly_num'), drop_level=False)
                    for f in perm[h:]])
            else:
                perm = rng.permutation(len(df))
                h = len(perm) // 2
                a, b = df.iloc[perm[:h]], df.iloc[perm[h:]]
            halves[g] = (a.mean(), b.mean())
        comps = {f'within_{CTRL}': halves[CTRL], f'within_{TNT}': halves[TNT],
            'between_halves': (halves[CTRL][0], halves[TNT][0]),
        }
        for name, (x, y) in comps.items():
            s, _ = H.compare_on_off(x, y)
            rows.append(dict(split=i, comparison=name, spearman=s['spearman_on_off'],
                pearson=s['pearson_on_off'], lin_r2=s['lin_r2'], iso_r2=s['iso_r2'],
                iso_resid_sd=s['iso_resid_sd'],
                block_fruity_small_iso=s['block_fruity_small_iso'],
                block_other_between_iso=s['block_other_between_iso'],
                mean_x=s['mean_on'], mean_y=s['mean_off'],
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B. density null
def excess_distance_recs(mats: Dict[str, Dict[tuple, pd.DataFrame]], **transform_kws
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Per-recording observed, null-expected and excess (observed - null) cosine
    distances; excess < 0 means the odors are more similar than density alone predicts."""
    obs, null, excess = {}, {}, {}
    for g, m in mats.items():
        obs[g] = H.recording_rdms_from_matrices(m, measure='cosine', **transform_kws)
        null[g] = H.recording_density_null_rdms(m, **transform_kws)
        excess[g] = obs[g] - null[g]
    return obs, null, excess


def density_regressed_recs(obs: Dict[str, pd.DataFrame], null: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
    """Per-recording residual of observed similarity on null similarity (one linear fit
    per recording across its pairs), returned as a distance-like quantity (negated)."""
    out = {}
    for g in obs:
        rows = {}
        for rec in obs[g].index:
            sim = 1 - obs[g].loc[rec]
            nul = 1 - null[g].loc[rec]
            ok = sim.notna() & nul.notna()
            slope, intercept = np.polyfit(nul[ok], sim[ok], 1)
            rows[rec] = -(sim - (intercept + slope * nul))
        out[g] = pd.DataFrame(rows).T
        out[g].index = obs[g].index
    return out


def per_odor_density(report: Dict[str, pd.Series], fracs: Dict[str, pd.DataFrame],
    orn: pd.Series, drive: pd.Series, rec_rdms: Dict[str, pd.DataFrame], *,
    n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-odor APL effect (isotonic residual) against per-odor response-fraction change
    and ORN drive, with recording-bootstrap CIs on the correlations."""
    def one(means: Dict[str, pd.Series], frac_means: Dict[str, pd.Series]
        ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        _, tabs = H.compare_on_off(means[CTRL], means[TNT], orn=orn, drive=drive)
        po = tabs['per_odor'].copy()
        po['frac_ctrl'] = frac_means[CTRL].reindex(po.index)
        po['frac_tnt'] = frac_means[TNT].reindex(po.index)
        po['dfrac'] = po.frac_tnt - po.frac_ctrl
        po['frac_ratio'] = po.frac_tnt / po.frac_ctrl
        st = {
            'spearman_iso_vs_drive': H.spearman(po.iso_resid, po.drive),
            'spearman_iso_vs_dfrac': H.spearman(po.iso_resid, po.dfrac),
            'spearman_iso_vs_frac_ratio': H.spearman(po.iso_resid, po.frac_ratio),
            'spearman_iso_vs_frac_ctrl': H.spearman(po.iso_resid, po.frac_ctrl),
            'spearman_iso_vs_frac_tnt': H.spearman(po.iso_resid, po.frac_tnt),
            'spearman_drive_vs_dfrac': H.spearman(po.drive, po.dfrac),
            'spearman_drive_vs_frac_ctrl': H.spearman(po.drive, po.frac_ctrl),
            'spearman_delta_vs_dfrac': H.spearman(po.delta, po.dfrac),
        }
        # partial rank correlations: rank-residualize on the covariate
        def partial(y: str, x: str, z: str) -> float:
            ry, rx, rz = (stats.rankdata(po[c]) for c in (y, x, z))
            ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
            ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
            return float(stats.pearsonr(ey, ex)[0])

        st['partial_iso_vs_drive_given_dfrac'] = partial('iso_resid', 'drive', 'dfrac')
        st['partial_iso_vs_dfrac_given_drive'] = partial('iso_resid', 'dfrac', 'drive')
        return po, st

    frac_means = {g: fracs[g].mean() for g in GENOS}
    po, observed = one({g: report[g] for g in GENOS}, frac_means)

    rng = np.random.default_rng(seed)
    draws = {k: [] for k in observed}
    for _ in range(n_boot):
        means, fm = {}, {}
        for g in GENOS:
            idx = rng.integers(0, len(rec_rdms[g]), len(rec_rdms[g]))
            means[g] = rec_rdms[g].iloc[idx].mean()
            fm[g] = fracs[g].iloc[idx].mean()
        _, st = one(means, fm)
        for k, v in st.items():
            draws[k].append(v)
    rows = {k: dict(observed=v, boot_sd=np.nanstd(draws[k]),
        lo=np.nanpercentile(draws[k], 5), hi=np.nanpercentile(draws[k], 95))
        for k, v in observed.items()
    }
    return po, pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# C. density-matched thresholds
def per_recording_matched_rdms(mats: Dict[tuple, pd.DataFrame], target_frac: float, *,
    values: str, measure: str = 'cosine') -> Tuple[pd.DataFrame, pd.Series]:
    """Each recording thresholded at its own value giving mean response fraction
    `target_frac`; returns per-recording RDMs and the thresholds."""
    rows, ths = {}, {}
    for rec, m in mats.items():
        th = H.match_threshold({rec: m}, target_frac)
        ths[rec] = th
        rows[rec] = H.pair_distances(H.transform_matrix(m, values=values, thresh=th),
            measure=measure
        )
    df = pd.DataFrame(rows).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=H.RECORDING_KEYS)
    return df, pd.Series(ths)


# ---------------------------------------------------------------------------
# plots
def _setup_plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110,
    })
    return plt


def savefig(fig, out_dir, name: str) -> None:
    fig.savefig(out_dir / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=160)


def fig_scatter_with_noise(plt, pairs: pd.DataFrame, bt: pd.DataFrame, out_dir) -> None:
    """Control vs TNT pair distances with per-pair bootstrap SD bars; block pairs marked."""
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    blk = pairs.block == H.FRUITY_SMALL_BLOCK
    ax.errorbar(pairs.off[~blk], pairs.on[~blk], xerr=pairs.off_sd[~blk],
        yerr=pairs.on_sd[~blk], fmt='o', ms=3.5, color=COLORS['between'], ecolor='#bbbbbb',
        elinewidth=0.7, capsize=0, label='other pairs (n = %d)' % (~blk).sum(), mew=0
    )
    ax.errorbar(pairs.off[blk], pairs.on[blk], xerr=pairs.off_sd[blk],
        yerr=pairs.on_sd[blk], fmt='o', ms=5, mfc='none', mec=COLORS[TNT], mew=1.3,
        ecolor=COLORS[TNT], elinewidth=0.7, capsize=0, label='fruity x small-ester pairs (9)'
    )
    on, off = pairs.on, pairs.off
    from sklearn.isotonic import IsotonicRegression
    xs = np.linspace(off.min(), off.max(), 200)
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(off, on)
    ax.plot(xs, iso.predict(xs), color=COLORS[CTRL], lw=1.5, label='isotonic fit')
    ax.plot([0, 1], [0, 1], color='#cccccc', lw=0.8, ls='--')
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel('APL>TNT: 1 - cosine (mean of 8 recordings)')
    ax.set_ylabel('control: 1 - cosine (mean of 11 recordings)')
    txt = (f"Spearman {bt.loc['spearman_on_off', 'observed']:.2f} "
        f"[{bt.loc['spearman_on_off', 'lo']:.2f}, {bt.loc['spearman_on_off', 'hi']:.2f}]\n"
        f"isotonic R$^2$ {bt.loc['iso_r2', 'observed']:.2f} "
        f"[{bt.loc['iso_r2', 'lo']:.2f}, {bt.loc['iso_r2', 'hi']:.2f}]\n"
        f"block residual {bt.loc['block_fruity_small_iso', 'observed']:+.3f} "
        f"[{bt.loc['block_fruity_small_iso', 'lo']:+.3f}, "
        f"{bt.loc['block_fruity_small_iso', 'hi']:+.3f}]\n(90% CI, recording bootstrap)"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', fontsize=7)
    ax.legend(loc='lower right', frameon=False)
    ax.set_title('A. control vs APL>TNT, 120 pairs, with per-pair noise')
    savefig(fig, out_dir, 'A1_ctrl_vs_tnt_with_noise')
    plt.close(fig)


def fig_pair_z(plt, pairs: pd.DataFrame, out_dir) -> None:
    """Isotonic residual of each pair with its bootstrap 90% interval, sorted."""
    d = pairs.sort_values('iso_resid')
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    x = np.arange(len(d))
    blk = (d.block == H.FRUITY_SMALL_BLOCK).to_numpy()
    cols = np.where(blk, COLORS[TNT], COLORS['between'])
    ax.vlines(x, d.iso_resid_lo, d.iso_resid_hi, color=cols, lw=0.9, alpha=0.8)
    ax.scatter(x, d.iso_resid, s=9, c=cols, zorder=3)
    ax.axhline(0, color='#cccccc', lw=0.8)
    sig = (d.iso_resid_lo > 0) | (d.iso_resid_hi < 0)
    ax.set_xticks(x[sig])
    ax.set_xticklabels([f'{a}-{b}' for a, b in d.index[sig]], rotation=90, fontsize=5.5)
    ax.set_ylabel('isotonic residual\n(control - monotone(TNT))')
    ax.set_title(f'A. per-pair deviation from a monotone control-vs-TNT relation; '
        f'{int(sig.sum())}/{len(d)} pairs exclude 0 at 90% (labeled); '
        'fruity x small-ester pairs in red'
    )
    savefig(fig, out_dir, 'A2_pair_isotonic_residuals')
    plt.close(fig)


def fig_split_half(plt, sh: pd.DataFrame, observed: Dict[str, float], out_dir,
    unit: str) -> None:
    cols = [('spearman', 'Spearman between matrices', observed['spearman_on_off']),
        ('iso_r2', 'isotonic R$^2$', observed['iso_r2']),
        ('iso_resid_sd', 'isotonic residual SD', observed['iso_resid_sd']),
        ('block_fruity_small_iso', 'fruity x small-ester block residual',
            observed['block_fruity_small_iso']),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8))
    comp_cols = {f'within_{CTRL}': COLORS[CTRL], f'within_{TNT}': COLORS[TNT],
        'between_halves': COLORS['between']}
    comp_labels = {f'within_{CTRL}': 'control half vs control half',
        f'within_{TNT}': 'TNT half vs TNT half',
        'between_halves': 'control half vs TNT half'}
    for ax, (col, title, obs) in zip(axs, cols):
        lo = min(sh[col].min(), obs)
        hi = max(sh[col].max(), obs)
        bins = np.linspace(lo, hi, 35)
        for comp, g in sh.groupby('comparison'):
            ax.hist(g[col], bins=bins, histtype='step', lw=1.4, color=comp_cols[comp],
                label=comp_labels[comp], density=True
            )
        ax.axvline(obs, color='black', lw=1.2, ls='--', label='control vs TNT, all recordings')
        ax.set_title(title)
        ax.set_yticks([])
    axs[0].legend(frameon=False, loc='upper left', fontsize=6)
    fig.suptitle(f'A. split-half ({unit}s) noise floor vs the observed between-genotype '
        'statistics', y=1.04
    )
    savefig(fig, out_dir, f'A3_split_half_{unit}')
    plt.close(fig)


def fig_density_null(plt, obs_mean: Dict[str, pd.Series], null_mean: Dict[str, pd.Series],
    excess_mean: Dict[str, pd.Series], bt: pd.DataFrame, out_dir, tag: str) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
    ax = axs[0]
    for g in GENOS:
        ax.scatter(1 - null_mean[g], 1 - obs_mean[g], s=10, color=COLORS[g],
            label=LABELS[g], alpha=0.8, edgecolors='none'
        )
    lim = [0, max(1 - pd.concat(list(obs_mean.values())).min(), 0.5)]
    ax.plot(lim, lim, color='#cccccc', lw=0.8, ls='--')
    ax.set_xlabel('cosine similarity expected from density alone')
    ax.set_ylabel('observed cosine similarity')
    ax.set_title(f'observed vs density-null similarity ({tag})')
    ax.legend(frameon=False)

    ax = axs[1]
    labels = pd.Series([H.block_label(a, b) for a, b in excess_mean[CTRL].index],
        index=excess_mean[CTRL].index
    )
    blk = labels == H.FRUITY_SMALL_BLOCK
    x, y = excess_mean[TNT], excess_mean[CTRL]
    ax.scatter(x[~blk], y[~blk], s=10, color=COLORS['between'], alpha=0.75,
        edgecolors='none', label='other pairs'
    )
    ax.scatter(x[blk], y[blk], s=24, facecolors='none', edgecolors=COLORS[TNT], lw=1.2,
        label='fruity x small-ester'
    )
    from sklearn.isotonic import IsotonicRegression
    xs = np.linspace(x.min(), x.max(), 200)
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(x, y)
    ax.plot(xs, iso.predict(xs), color=COLORS[CTRL], lw=1.4)
    ax.axhline(0, color='#dddddd', lw=0.8)
    ax.axvline(0, color='#dddddd', lw=0.8)
    ax.set_xlabel('APL>TNT: excess distance (observed - null)')
    ax.set_ylabel('control: excess distance (observed - null)')
    txt = (f"Spearman {bt.loc['spearman_on_off', 'observed']:.2f} "
        f"[{bt.loc['spearman_on_off', 'lo']:.2f}, {bt.loc['spearman_on_off', 'hi']:.2f}]\n"
        f"iso R$^2$ {bt.loc['iso_r2', 'observed']:.2f}; block "
        f"{bt.loc['block_fruity_small_iso', 'observed']:+.3f} "
        f"[{bt.loc['block_fruity_small_iso', 'lo']:+.3f}, "
        f"{bt.loc['block_fruity_small_iso', 'hi']:+.3f}]"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', fontsize=7)
    ax.set_title('density-corrected control vs TNT')
    ax.legend(frameon=False, loc='lower right')

    ax = axs[2]
    bm = pd.DataFrame({g: H.block_means(excess_mean[g]) for g in GENOS})
    bm = bm.loc[bm.index != 'within'].sort_values(CTRL)
    yy = np.arange(len(bm))
    for i, g in enumerate(GENOS):
        ax.barh(yy + (i - 0.5) * 0.38, bm[g], height=0.36, color=COLORS[g], label=LABELS[g])
    ax.set_yticks(yy)
    ax.set_yticklabels(bm.index, fontsize=6)
    ax.axvline(0, color='#888888', lw=0.8)
    ax.set_xlabel('mean excess distance of block')
    ax.set_title('excess distance by between-group block')
    ax.legend(frameon=False, loc='upper left')
    fig.suptitle(f'B. within-odor KC-identity shuffle (density) null, {tag}', y=1.03)
    savefig(fig, out_dir, f'B1_density_null_{tag}')
    plt.close(fig)


def fig_per_odor(plt, po: pd.DataFrame, pst: pd.DataFrame, out_dir) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, (xcol, xlabel, key) in zip(axs, [
            ('drive', 'ORN drive (mean consensus dF/F over glomeruli)', 'spearman_iso_vs_drive'),
            ('dfrac', 'response fraction change (TNT - control)', 'spearman_iso_vs_dfrac'),
            ('frac_ratio', 'response fraction ratio (TNT / control)', 'spearman_iso_vs_frac_ratio')]):
        ax.scatter(po[xcol], po.iso_resid, s=16, color=COLORS['between'], edgecolors='none')
        for o, r in po.iterrows():
            ax.annotate(o, (r[xcol], r.iso_resid), fontsize=6, xytext=(2, 2),
                textcoords='offset points',
                color=COLORS[TNT] if o in H.APPROX_CONC_ODORS else '#333333'
            )
        ax.axhline(0, color='#cccccc', lw=0.8, ls='--')
        ax.set_xlabel(xlabel)
        ax.set_title(f"Spearman {pst.loc[key, 'observed']:.2f} "
            f"[{pst.loc[key, 'lo']:.2f}, {pst.loc[key, 'hi']:.2f}]")
    axs[0].set_ylabel('per-odor APL effect\n(mean isotonic residual of its pairs)')
    fig.suptitle('B. per-odor extra decorrelation vs ORN drive and vs density change '
        '(red labels: ORN concentration approximated)', y=1.04
    )
    savefig(fig, out_dir, 'B2_per_odor_effect_vs_drive_and_density')
    plt.close(fig)


def fig_variants(plt, table: pd.DataFrame, stat_cols: List[Tuple[str, str]], out_dir,
    name: str, title: str, label_col: str = 'variant') -> None:
    """Dot-and-interval plot of key statistics across variants (rows)."""
    n = len(stat_cols)
    fig, axs = plt.subplots(1, n, figsize=(2.1 * n + 2.5, 0.32 * len(table) + 1.4),
        sharey=True
    )
    y = np.arange(len(table))[::-1]
    for ax, (col, lab) in zip(np.atleast_1d(axs), stat_cols):
        lo, hi = table.get(f'{col}_lo'), table.get(f'{col}_hi')
        if lo is not None:
            ax.hlines(y, lo, hi, color='#aaaaaa', lw=1.2)
        ax.scatter(table[col], y, s=18, color=COLORS['between'], zorder=3)
        ax.axvline(table[col].iloc[0], color=COLORS[CTRL], lw=0.8, ls=':')
        ax.set_title(lab, fontsize=8)
        ax.grid(axis='x', color='#eeeeee', lw=0.6)
    np.atleast_1d(axs)[0].set_yticks(y)
    np.atleast_1d(axs)[0].set_yticklabels(table[label_col], fontsize=7)
    fig.suptitle(title, y=1.02)
    savefig(fig, out_dir, name)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--n-split', type=int, default=1000)
    ap.add_argument('--quick', action='store_true', help='few resamples, for testing')
    ap.add_argument('--parts', default='ABCD', help='which parts to run (subset of ABCD); '
        'summary.json is merged into an existing one when not all parts run')
    args = ap.parse_args()
    parts = set(args.parts.upper())
    n_boot, n_split = (100, 50) if args.quick else (args.n_boot, args.n_split)
    n_boot_var = max(100, n_boot // 4)

    t0 = time.time()
    out = H.results_dir(EXP)
    pd.set_option('display.width', 200)
    plt = _setup_plt()
    summary: Dict[str, object] = {'n_boot': n_boot, 'n_split': n_split}
    if parts != set('ABCD') and (out / 'summary.json').exists():
        with open(out / 'summary.json') as f:
            summary = json.load(f)
        summary.update(n_boot=n_boot, n_split=n_split)

    orn = H.orn16_consensus_pairs()
    drive = H.consensus_orn_drive()
    ref = H.reference_pairs()
    report = H.report_matrices()
    fn = partial(stats_fn, orn=orn, drive=drive, ref=ref)
    fn_noref = partial(stats_fn, orn=orn, drive=drive, ref=None, with_ref=False)

    mats = {g: H.yang_recording_matrices(g) for g in GENOS}
    recs = {g: H.yang_recording_rdms(g) for g in GENOS}
    for g in GENOS:
        assert (recs[g].mean() - report[g]).abs().max() < 1e-6
        assert list(recs[g].index) == list(mats[g].keys()), g
    fracs = {g: H.recording_response_fractions(mats[g], 1.0) for g in GENOS}
    summary['n_recordings'] = {g: int(len(recs[g])) for g in GENOS}
    summary['n_flies'] = {g: int(len(recs[g].index.droplevel('hemibrain').unique()))
        for g in GENOS
    }
    summary['mean_response_fraction_thresh1'] = {g: float(fracs[g].mean().mean())
        for g in GENOS
    }

    # ---------------- A. noise floor
    if 'A' in parts:
        run_a(out, plt, fn, recs, report, orn, drive, n_boot, n_split, summary)
    if 'B' in parts:
        run_b(out, plt, fn_noref, mats, recs, report, fracs, orn, drive, n_boot, n_boot_var,
            summary)
    if 'C' in parts:
        run_c(out, plt, fn, mats, fracs, n_boot_var, summary)
    if 'D' in parts:
        run_d(out, plt, fn, mats, n_boot_var, summary)

    summary['runtime_s'] = time.time() - t0
    summary['approx_conc_note'] = H.APPROX_CONC_NOTE
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    print(f'done in {time.time() - t0:.0f} s; outputs in {out}')


def run_a(out, plt, fn, recs, report, orn, drive, n_boot, n_split, summary) -> None:
    print('A. bootstrap of all statistics (recording and fly units)')
    bootA = {u: boot_table(fn, recs, n_boot=n_boot, seed=i, unit=u)
        for i, u in enumerate(('recording', 'fly'))
    }
    tabA = pd.concat(bootA, axis='columns')
    tabA.columns = [f'{u}_{c}' if c != 'observed' else c for u, c in tabA.columns]
    tabA = tabA.loc[:, ~tabA.columns.duplicated()]
    tabA.to_csv(out / 'A_bootstrap_stats.csv')
    print(tabA[['observed', 'recording_lo', 'recording_hi', 'fly_lo', 'fly_hi']]
        .loc[HEADLINE].round(3).to_string())
    summary['A_headline'] = {k: dict(observed=float(tabA.loc[k, 'observed']),
        rec_ci=[float(tabA.loc[k, 'recording_lo']), float(tabA.loc[k, 'recording_hi'])],
        fly_ci=[float(tabA.loc[k, 'fly_lo']), float(tabA.loc[k, 'fly_hi'])])
        for k in HEADLINE
    }

    print('A. per-pair bootstrap of the isotonic residual')
    pairsA = per_pair_bootstrap(recs, n_boot=n_boot, seed=10, unit='recording')
    pairsA.sort_values('iso_z', key=np.abs, ascending=False).to_csv(
        out / 'A_pair_residuals.csv'
    )
    sig = (pairsA.iso_resid_lo > 0) | (pairsA.iso_resid_hi < 0)
    noise_sd = float(np.sqrt((pairsA.iso_resid_sd ** 2).mean()))
    obs_sd = float(pairsA.iso_resid.std())
    summary['A_pairs'] = {
        'n_pairs_excluding_0_at_90pct': int(sig.sum()),
        'pairs_excluding_0': [f'{a}-{b}' for a, b in pairsA.index[sig]],
        'n_block_pairs_excluding_0': int((sig & (pairsA.block == H.FRUITY_SMALL_BLOCK)).sum()),
        'rms_boot_sd_of_iso_resid': noise_sd,
        'observed_iso_resid_sd': obs_sd,
        'frac_iso_resid_var_above_noise': float(1 - noise_sd ** 2 / obs_sd ** 2),
        'median_pair_mean_sd_ctrl': float(pairsA.on_sd.median()),
        'median_pair_mean_sd_tnt': float(pairsA.off_sd.median()),
    }
    print(f'  {int(sig.sum())}/120 pairs with isotonic residual excluding 0 at 90%; '
        f'rms per-pair noise SD {noise_sd:.3f} vs observed residual SD {obs_sd:.3f}')
    print('  top |z|:')
    print(pairsA.sort_values('iso_z', key=np.abs, ascending=False)
        [['on', 'off', 'iso_resid', 'iso_resid_sd', 'iso_z', 'block']].head(15).round(3)
        .to_string())
    fig_scatter_with_noise(plt, pairsA, bootA['recording'], out)
    fig_pair_z(plt, pairsA, out)

    print('A. split-half')
    obs_on_off, _ = H.compare_on_off(report[CTRL], report[TNT], orn=orn, drive=drive)
    sh_rows = []
    for i, unit in enumerate(('recording', 'fly')):
        sh = split_half(recs, n_split=n_split, seed=20 + i, unit=unit)
        sh.to_csv(out / f'A_split_half_{unit}_draws.csv', index=False)
        summ = sh.groupby('comparison').agg(['mean', 'std',
            lambda s: s.quantile(0.05), lambda s: s.quantile(0.95)])
        summ.columns = [f'{c}_{n}' for c, n in zip(summ.columns.get_level_values(0),
            ['mean', 'sd', 'lo', 'hi'] * (len(summ.columns) // 4))]
        summ.insert(0, 'unit', unit)
        sh_rows.append(summ)
        fig_split_half(plt, sh, obs_on_off, out, unit)
    shs = pd.concat(sh_rows)
    shs.to_csv(out / 'A_split_half_summary.csv')
    print(shs[['unit', 'spearman_mean', 'spearman_lo', 'spearman_hi', 'iso_r2_mean',
        'iso_r2_lo', 'iso_r2_hi', 'iso_resid_sd_mean', 'block_fruity_small_iso_mean',
        'block_fruity_small_iso_lo', 'block_fruity_small_iso_hi']].round(3).to_string())
    summary['A_split_half'] = json.loads(shs.reset_index().to_json(orient='records'))



def run_b(out, plt, fn_noref, mats, recs, report, fracs, orn, drive, n_boot, n_boot_var,
    summary) -> None:
    print('B. density null')
    # check the analytic expectation against shuffles for one recording
    rec0 = next(iter(mats[TNT]))
    rng = np.random.default_rng(0)
    for values in ('binary', 'graded'):
        shuf = pd.concat([H.shuffled_pair_distances(mats[TNT][rec0], rng, values=values)
            for _ in range(200)], axis='columns').mean(axis='columns')
        ana = 1 - H.density_null_similarity(H.transform_matrix(mats[TNT][rec0],
            values=values))
        print(f'  analytic vs 200-shuffle mean null distance ({values}, one TNT '
            f'recording): max |diff| {(shuf - ana).abs().max():.4f}, '
            )
    summary['B'] = {}
    bootB = {}
    for values in ('binary', 'graded'):
        obs, null, excess = excess_distance_recs(mats, values=values, thresh=1.0)
        regressed = density_regressed_recs(obs, null)
        obs_m = {g: obs[g].mean() for g in GENOS}
        null_m = {g: null[g].mean() for g in GENOS}
        exc_m = {g: excess[g].mean() for g in GENOS}
        bt_obs = boot_table(fn_noref, obs, n_boot=n_boot_var, seed=30)
        bt_exc = boot_table(fn_noref, excess, n_boot=n_boot_var, seed=31)
        bt_reg = boot_table(fn_noref, regressed, n_boot=n_boot_var, seed=32)
        bootB[values] = {'observed': bt_obs, 'excess': bt_exc, 'regressed': bt_reg}
        # block-level excess similarity per genotype, with CIs
        def block_fn(means: Dict[str, pd.Series]) -> Dict[str, float]:
            o = {}
            for g in GENOS:
                bm = H.block_means(means[g])
                o[f'{g}_block_fruity_small'] = bm[H.FRUITY_SMALL_BLOCK]
                o[f'{g}_other_between'] = bm.drop(['within', H.FRUITY_SMALL_BLOCK]).mean()
                o[f'{g}_within'] = bm['within']
                o[f'{g}_block_minus_other'] = (o[f'{g}_block_fruity_small'] -
                    o[f'{g}_other_between'])
                o[f'{g}_mean'] = float(means[g].mean())
            o['block_minus_other_ctrl_minus_tnt'] = (o[f'{CTRL}_block_minus_other'] -
                o[f'{TNT}_block_minus_other'])
            o['mean_ctrl_minus_tnt'] = o[f'{CTRL}_mean'] - o[f'{TNT}_mean']
            return o
        bt_blocks = pd.concat({'observed_dist': boot_table(block_fn, obs, n_boot=n_boot_var,
            seed=33), 'null_dist': boot_table(block_fn, null, n_boot=n_boot_var, seed=34),
            'excess_dist': boot_table(block_fn, excess, n_boot=n_boot_var, seed=35)},
            axis='columns'
        )
        bt_blocks.to_csv(out / f'B_blocks_{values}.csv')
        pd.DataFrame({f'{g}_{k}': v[g] for k, v in (('observed', obs_m), ('null', null_m),
            ('excess', exc_m)) for g in GENOS}).to_csv(out / f'B_pairs_{values}.csv')
        rows = pd.DataFrame({'observed': flat_row(bt_obs, KEY_ON_OFF),
            'excess (observed - null)': flat_row(bt_exc, KEY_ON_OFF),
            'density-regressed': flat_row(bt_reg, KEY_ON_OFF)}).T
        rows.index.name = 'matrix'
        rows.to_csv(out / f'B_on_off_stats_{values}.csv')
        print(f'  [{values}] mean distance observed ctrl {obs_m[CTRL].mean():.3f} / TNT '
            f'{obs_m[TNT].mean():.3f}; null {null_m[CTRL].mean():.3f} / {null_m[TNT].mean():.3f}; '
            f'excess {exc_m[CTRL].mean():.3f} / {exc_m[TNT].mean():.3f}')
        print(rows[[c for c in rows.columns if not c.endswith(('_lo', '_hi'))]]
            [['mean_on', 'mean_off', 'spearman_on_off', 'iso_r2', 'block_fruity_small_iso',
            'block_other_between_iso', 'spearman_orn_on', 'spearman_orn_off', 'gap_closed',
            'spearman_per_odor_iso_vs_drive']].round(3).to_string())
        print(bt_blocks.xs('observed', axis='columns', level=1).round(3).to_string())
        summary['B'][values] = {
            'mean_observed': {g: float(obs_m[g].mean()) for g in GENOS},
            'mean_null': {g: float(null_m[g].mean()) for g in GENOS},
            'mean_excess': {g: float(exc_m[g].mean()) for g in GENOS},
            'on_off': json.loads(rows.to_json(orient='index')),
            'blocks': json.loads(bt_blocks.xs('observed', axis='columns', level=1)
                .to_json(orient='index')),
            'blocks_ci': {m: json.loads(bt_blocks[m][['lo', 'hi']].to_json(orient='index'))
                for m in ('observed_dist', 'null_dist', 'excess_dist')},
        }
        fig_density_null(plt, obs_m, null_m, exc_m, bt_exc, out, values)

    print('B. per-odor effect vs density change')
    po, pst = per_odor_density(report, fracs, orn, drive, recs, n_boot=n_boot, seed=40)
    po.round(4).to_csv(out / 'B_per_odor.csv')
    pst.round(4).to_csv(out / 'B_per_odor_correlations.csv')
    print(po.sort_values('iso_resid').round(3).to_string())
    print(pst.round(3).to_string())
    summary['B']['per_odor'] = json.loads(pst.to_json(orient='index'))
    fig_per_odor(plt, po, pst, out)



def run_c(out, plt, fn, mats, fracs, n_boot_var, summary) -> None:
    print('C. density-matched thresholds')
    f_ctrl = float(fracs[CTRL].mean().mean())
    f_tnt = float(fracs[TNT].mean().mean())
    th_tnt = H.match_threshold(mats[TNT], f_ctrl)
    th_ctrl = H.match_threshold(mats[CTRL], f_tnt)
    print(f'  control frac at 1.0: {f_ctrl:.4f}; TNT frac at 1.0: {f_tnt:.4f}; TNT matched '
        f'threshold {th_tnt:.3f}; control matched threshold {th_ctrl:.3f}')
    summary['C'] = {'frac_ctrl_thresh1': f_ctrl, 'frac_tnt_thresh1': f_tnt,
        'tnt_threshold_matching_ctrl_density': th_tnt,
        'ctrl_threshold_matching_tnt_density': th_ctrl,
    }
    variants = [
        ('graded, all cells (report)', dict(values='graded'), dict(values='graded')),
        ('binary > 1, both', dict(values='binary', thresh=1.0), dict(values='binary', thresh=1.0)),
        (f'binary, TNT > {th_tnt:.2f} (density matched to control)',
            dict(values='binary', thresh=1.0), dict(values='binary', thresh=th_tnt)),
        (f'binary, control > {th_ctrl:.2f} (density matched to TNT)',
            dict(values='binary', thresh=th_ctrl), dict(values='binary', thresh=1.0)),
        ('thresholded > 1, both', dict(values='thresholded', thresh=1.0),
            dict(values='thresholded', thresh=1.0)),
        (f'thresholded, TNT > {th_tnt:.2f} (density matched)',
            dict(values='thresholded', thresh=1.0), dict(values='thresholded', thresh=th_tnt)),
        ('binary > 0.75, both', dict(values='binary', thresh=0.75),
            dict(values='binary', thresh=0.75)),
    ]
    rowsC = []
    for name, kc, kt in variants:
        rv = {CTRL: H.recording_rdms_from_matrices(mats[CTRL], **kc),
            TNT: H.recording_rdms_from_matrices(mats[TNT], **kt)}
        bt = boot_table(fn, rv, n_boot=n_boot_var, seed=50)
        row = dict(variant=name, frac_ctrl=float(H.recording_response_fractions(mats[CTRL],
            kc.get('thresh', 1.0)).mean().mean()) if kc['values'] != 'graded' else np.nan,
            frac_tnt=float(H.recording_response_fractions(mats[TNT],
            kt.get('thresh', 1.0)).mean().mean()) if kt['values'] != 'graded' else np.nan)
        row.update(flat_row(bt, KEY_ON_OFF + [f'{g}_{k}' for g in GENOS for k in KEY_REF]))
        rowsC.append(row)
    # per-recording matched TNT thresholds
    rv_pr, ths_pr = per_recording_matched_rdms(mats[TNT], f_ctrl, values='binary')
    rv = {CTRL: H.recording_rdms_from_matrices(mats[CTRL], values='binary', thresh=1.0),
        TNT: rv_pr}
    bt = boot_table(fn, rv, n_boot=n_boot_var, seed=51)
    row = dict(variant='binary, TNT threshold matched per recording', frac_ctrl=f_ctrl,
        frac_tnt=f_ctrl)
    row.update(flat_row(bt, KEY_ON_OFF + [f'{g}_{k}' for g in GENOS for k in KEY_REF]))
    rowsC.append(row)
    summary['C']['per_recording_tnt_thresholds'] = {str(k): float(v)
        for k, v in ths_pr.items()}
    tabC = pd.DataFrame(rowsC)
    tabC.to_csv(out / 'C_density_matched.csv', index=False)
    showC = ['variant', 'frac_ctrl', 'frac_tnt', 'mean_on', 'mean_off', 'spearman_on_off',
        'iso_r2', 'block_fruity_small_iso', 'block_fruity_small_iso_lo',
        'block_fruity_small_iso_hi', 'spearman_orn_on', 'spearman_orn_off', 'gap_closed',
        'spearman_per_odor_iso_vs_drive', f'{CTRL}_spearman_resid_vs_vcf_hyp',
        f'{TNT}_spearman_resid_vs_vcf_hyp']
    print(tabC[showC].round(3).to_string())
    summary['C']['table'] = json.loads(tabC[showC].to_json(orient='records'))
    fig_variants(plt, tabC, [('mean_on', 'mean dist, control'), ('mean_off', 'mean dist, TNT'),
        ('spearman_on_off', 'Spearman ctrl vs TNT'), ('iso_r2', 'isotonic R$^2$'),
        ('block_fruity_small_iso', 'fruity x small block'), ('gap_closed', 'ORN gap closed'),
        ('spearman_per_odor_iso_vs_drive', 'per-odor effect vs drive')], out,
        'C1_density_matched', 'C. density-matched thresholds (90% CI, recording bootstrap)'
    )



def run_d(out, plt, fn, mats, n_boot_var, summary) -> None:
    print('D. pipeline variants')
    rowsD = []
    for values in ('graded', 'binary', 'thresholded'):
        for cells in ('all', 'responsive'):
            for measure in ('cosine', 'pearson'):
                kws = dict(values=values, cells=cells, thresh=1.0)
                rv = {g: H.recording_rdms_from_matrices(mats[g], measure=measure, **kws)
                    for g in GENOS}
                bt = boot_table(fn, rv, n_boot=n_boot_var, seed=60)
                row = dict(variant=f'{values}, {cells} cells, {measure}', values=values,
                    cells=cells, measure=measure,
                    n_nan_pairs=int(sum(rv[g].mean().isna().sum() for g in GENOS)))
                row.update(flat_row(bt, KEY_ON_OFF + [f'{g}_{k}' for g in GENOS
                    for k in KEY_REF]))
                rowsD.append(row)
    tabD = pd.DataFrame(rowsD)
    tabD.to_csv(out / 'D_pipeline_variants.csv', index=False)
    showD = ['variant', 'n_nan_pairs', 'mean_on', 'mean_off', 'spearman_on_off', 'iso_r2',
        'block_fruity_small_iso', 'block_fruity_small_iso_lo', 'block_fruity_small_iso_hi',
        'spearman_orn_on', 'spearman_orn_off', 'gap_closed', 'spearman_per_odor_iso_vs_drive',
        f'{CTRL}_spearman_resid_vs_vcf_hyp', f'{TNT}_spearman_resid_vs_vcf_hyp',
        f'{CTRL}_spearman_kc_claw', f'{TNT}_spearman_kc_claw']
    print(tabD[showD].round(3).to_string())
    summary['D'] = json.loads(tabD[showD].to_json(orient='records'))
    fig_variants(plt, tabD, [('mean_on', 'mean dist, control'), ('mean_off', 'mean dist, TNT'),
        ('spearman_on_off', 'Spearman ctrl vs TNT'), ('iso_r2', 'isotonic R$^2$'),
        ('block_fruity_small_iso', 'fruity x small block'),
        ('spearman_orn_on', 'Spearman ORN, ctrl'), ('spearman_orn_off', 'Spearman ORN, TNT'),
        (f'{CTRL}_spearman_resid_vs_vcf_hyp', 'resid vs VCF, ctrl'),
        (f'{TNT}_spearman_resid_vs_vcf_hyp', 'resid vs VCF, TNT')], out,
        'D1_pipeline_variants', 'D. RDM pipeline variants (90% CI, recording bootstrap)'
    )


if __name__ == '__main__':
    main()
