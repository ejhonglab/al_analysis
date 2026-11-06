#!/usr/bin/env python3
"""Which odors lose rank against the control flies' per-odor response rates once the
oracle glomerulus gain is in the joint fit.

Reads the archived KC x odor response matrices of the structure-loss joint fits without
the gain (E25 on the real wiring, E30 on re-drawn wiring 2) and with the 1.5x gain on the
eight oracle glomeruli (E28's joint-fit arm, E33), plus the fixed-point variants of E28
(gain added without re-fitting), and compares each variant's per-odor response fractions
with the KC data (control flies for APL on, APL output blocked (TNT) for APL off): the
rate rank agreement, each odor's rank, and how the fractions move between the no-gain and
gain fits, by odor group (fruity esters / weak-input odors / the rest).

Writes results/gain_rate_cost/ in the shared directory: per_odor_fractions.csv (every
variant, both conditions, with the data), rank_table.csv (odor x variant ranks and
fraction ratios), group_means.csv and gain_rate_cost.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kc_remapping_harness as H

SHARED = Path('/local/tensor/Tom/claude_kc_remapping')
ARCH = SHARED / 'data' / 'model_outputs'
OUT = SHARED / 'results' / 'gain_rate_cost'

# label -> (archive folder, variant stem). Pairs share everything but the gain.
FITS = {
    ('real wiring', 70, 'no gain'): ('E25_joint_fit_structure', 'thr70_k2_ext'),
    ('real wiring', 70, 'gain 1.5'): ('E28_gain_combined_jointfit', 'thr70_gain1.5_k2'),
    ('real wiring', 40, 'no gain'): ('E25_joint_fit_structure', 'thr40_k2_ext'),
    ('real wiring', 40, 'gain 1.5'): ('E28_gain_combined_jointfit', 'thr40_gain1.5_k2'),
    ('re-draw 2', 70, 'no gain'): ('E30_wiring_replication',
        'thr70_wiring-claw_permutation_within_type-seed2_k2'),
    ('re-draw 2', 70, 'gain 1.5'): ('E33_gain_redraw',
        'thr70_wiring-claw_permutation_within_type-seed2_gain1.5_k2'),
    ('re-draw 2', 40, 'no gain'): ('E30_wiring_replication',
        'thr40_wiring-claw_permutation_within_type-seed2_k2'),
    ('re-draw 2', 40, 'gain 1.5'): ('E33_gain_redraw',
        'thr40_wiring-claw_permutation_within_type-seed2_gain1.5_k2'),
}
ESTERS = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep']
WEAK = ['ms', 'paa', 'B-myr', 'euc', '1-8ol']


def group(odor: str) -> str:
    if odor in ESTERS:
        return 'fruity esters'
    if odor in WEAK:
        return 'weak-input odors'
    return 'other'


def data_fractions() -> pd.DataFrame:
    ctrl = H.yang_response_fractions('TNT3c').mean()
    tnt = H.yang_response_fractions('TNT').mean()
    df = pd.DataFrame({'data_on': ctrl, 'data_off': tnt}).drop(index='pfo', errors='ignore')
    return df


def model_fractions(folder: str, stem: str) -> pd.DataFrame:
    on = pd.read_parquet(ARCH / folder / f'{stem}_on_responses.parquet')
    off = pd.read_parquet(ARCH / folder / f'{stem}_acute_off_responses.parquet')
    return pd.DataFrame({'on': H.response_fractions(on), 'off': H.response_fractions(off)})


def ranks(s: pd.Series) -> pd.Series:
    """1 = the odor with the largest response fraction."""
    return s.rank(ascending=False, method='average')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = data_fractions()
    odors = [o for o in H.odor_order(data.index) if o in data.index]
    data = data.loc[odors]

    frac = {}
    for key, (folder, stem) in FITS.items():
        frac[key] = model_fractions(folder, stem).reindex(odors)

    # E28's fixed points: the gain added without re-fitting (per-odor fractions already
    # tabulated by E28_gain_combined.py).
    fixed = pd.read_csv(SHARED / 'results' / 'E28_gain_combined' /
        'per_odor_response_fractions.csv', index_col='odor').reindex(odors)
    for point in ['thr70_k2_kckc100', 'thr40_k2_kckc30']:
        thr = int(point[3:5])
        for g in ['1', '1.25', '1.5']:
            if f'{point}_g{g}_on' not in fixed.columns:
                continue
            key = ('fixed point', thr, 'no gain' if g == '1' else f'gain {g} (no re-fit)')
            frac[key] = pd.DataFrame({'on': fixed[f'{point}_g{g}_on'],
                'off': fixed[f'{point}_g{g}_acute_off']})

    long = []
    for (wiring, thr, gain), df in frac.items():
        for cond in ['on', 'off']:
            for odor in odors:
                long.append(dict(wiring=wiring, claw_thr=thr, gain=gain, condition=cond,
                    odor=odor, group=group(odor), model_frac=df.loc[odor, cond],
                    data_frac=data.loc[odor, f'data_{cond}'],
                ))
    long = pd.DataFrame(long)
    long.to_csv(OUT / 'per_odor_fractions.csv', index=False)

    # Rank agreement per variant and condition.
    agree = []
    for (wiring, thr, gain), df in frac.items():
        row = dict(wiring=wiring, claw_thr=thr, gain=gain)
        for cond in ['on', 'off']:
            row[f'rate_rank_agreement_{cond}'] = H.spearman(df[cond], data[f'data_{cond}'])
            row[f'mean_rate_{cond}'] = float(df[cond].mean())
        agree.append(row)
    agree = pd.DataFrame(agree)
    agree.to_csv(OUT / 'rate_rank_agreement.csv', index=False)

    # Rank table (APL on vs the control flies) and the no-gain -> gain movement per odor.
    rank_tab = pd.DataFrame({'group': [group(o) for o in odors],
        'data_frac': data['data_on'], 'data_rank': ranks(data['data_on'])}, index=odors)
    for (wiring, thr, gain), df in frac.items():
        if wiring == 'fixed point':
            continue
        tag = f'{wiring}, {thr} Hz, {gain}'
        rank_tab[f'{tag}: frac'] = df['on']
        rank_tab[f'{tag}: rank'] = ranks(df['on'])
    for wiring in ['real wiring', 're-draw 2']:
        for thr in [70, 40]:
            a = frac[(wiring, thr, 'no gain')]['on']
            b = frac[(wiring, thr, 'gain 1.5')]['on']
            rank_tab[f'{wiring}, {thr} Hz: gain / no gain'] = b / a
            rank_tab[f'{wiring}, {thr} Hz: rank change'] = ranks(b) - ranks(a)
    rank_tab.index.name = 'odor'
    rank_tab.to_csv(OUT / 'rank_table.csv')

    # Group means: mean fraction by odor group, data and each variant (APL on and off).
    gm = (long.groupby(['wiring', 'claw_thr', 'gain', 'condition', 'group'])
        [['model_frac', 'data_frac']].mean().reset_index())
    gm.to_csv(OUT / 'group_means.csv', index=False)

    # Figure: data vs model per-odor fraction (APL on), no-gain (hollow) -> gain (filled).
    colors = {'fruity esters': '#c2410c', 'weak-input odors': '#1d4ed8', 'other': '#6b7280'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharex=True, sharey=True)
    for ax, (wiring, thr) in zip(axes.flat, [('real wiring', 70), ('real wiring', 40),
            ('re-draw 2', 70), ('re-draw 2', 40)]):
        a = frac[(wiring, thr, 'no gain')]['on']
        b = frac[(wiring, thr, 'gain 1.5')]['on']
        x = data['data_on']
        lim = (0, max(x.max(), a.max(), b.max()) * 1.12)
        ax.plot(lim, lim, color='#d4d4d8', lw=1, zorder=0)
        for odor in odors:
            c = colors[group(odor)]
            ax.plot([x[odor], x[odor]], [a[odor], b[odor]], color=c, lw=1, alpha=0.7)
            ax.scatter(x[odor], a[odor], s=36, facecolors='white', edgecolors=c, lw=1.4,
                zorder=3)
            ax.scatter(x[odor], b[odor], s=36, color=c, zorder=4)
            ax.annotate(odor, (x[odor], b[odor]), xytext=(4, 3), textcoords='offset points',
                fontsize=7.5, color=c)
        ra = agree[(agree.wiring == wiring) & (agree.claw_thr == thr)].set_index('gain')
        ax.set_title(f'{wiring}, {thr} Hz per-claw threshold\n(rate rank agreement '
            f'{ra.loc["no gain", "rate_rank_agreement_on"]:.2f} without the gain, '
            f'{ra.loc["gain 1.5", "rate_rank_agreement_on"]:.2f} with it)', fontsize=10)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
    for ax in axes[1]:
        ax.set_xlabel('KCs, control flies: fraction of KCs responding')
    for ax in axes[:, 0]:
        ax.set_ylabel('model, APL on: fraction of KCs responding')
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=g)
        for g, c in colors.items()]
    handles += [plt.Line2D([], [], marker='o', ls='', markerfacecolor='white',
        markeredgecolor='black', label='joint fit without the gain'),
        plt.Line2D([], [], marker='o', ls='', color='black', label='joint fit with the 1.5x gain')]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8.5, frameon=False)
    fig.suptitle('Per-odor response rates of the structure-loss joint fits, without and '
        'with the oracle glomerulus gain', fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / 'gain_rate_cost.png', dpi=150)

    pd.set_option('display.width', 200)
    print(agree.round(3).to_string(index=False))
    print()
    show = gm[gm.condition == 'on'].pivot_table(index=['wiring', 'claw_thr', 'gain'],
        columns='group', values='model_frac')
    show['data'] = ''
    print('APL on, mean fraction by odor group (data: esters %.3f, weak %.3f, other %.3f)' % tuple(
        data.groupby([group(o) for o in odors])['data_on'].mean()
        [['fruity esters', 'weak-input odors', 'other']]))
    print(show.round(3).to_string())
    print()
    cols = ['group', 'data_frac', 'data_rank'] + [c for c in rank_tab.columns
        if c.startswith('real wiring, 70 Hz')]
    print(rank_tab[cols].round(3).to_string())


if __name__ == '__main__':
    main()
