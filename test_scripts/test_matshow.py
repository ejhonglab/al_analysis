#!/usr/bin/env python3

import warnings

import pandas as pd
import matplotlib.pyplot as plt

from hong2p import viz


#warnings.filterwarnings('error', message='constrained_layout not applied.*')

diverging_cmap = plt.get_cmap('RdBu_r')

def main():
    s2 = pd.read_pickle('s2.p')
    print(s2)

    level_fn = lambda d: d['fixed_thr']
    group_text = True
    format_fixed_thr = lambda x: f'{x:.0f}'
    # trying to just use this to format last row/col index level (wAPLKC).
    # fixed_thr should be handled by group_text stuff, which i might want to change
    # handling of inside hong2p
    # TODO what is input to fn gonna be tho? prob need x[1]?
    format_wAPLKC = lambda x: f'{x[1]:.1f}'
    xticklabels = format_wAPLKC
    yticklabels = format_wAPLKC

    # ./test_matshow.py:47: UserWarning: This figure includes Axes that are not
    # compatible with tight_layout, so results might be incorrect.
    # ./test_matshow.py:47: UserWarning: Tight layout not applied. The bottom and top
    # margins cannot be made large enough to accommodate all axes decorations.
    # fig, ax = plt.subplots(layout='tight')

    fig, _ = viz.matshow(s2,
        cmap=diverging_cmap,
        vmin=-1.0, vmax=1.0, levels_from_labels=False,
        hline_level_fn=level_fn, vline_level_fn=level_fn,
        hline_group_text=group_text, vline_group_text=group_text,
        group_fontsize=10, xtickrotation='horizontal',
        # TODO change hong2p.viz to have any levels not used to group formatted into
        # label?
        xticklabels=xticklabels, yticklabels=yticklabels,
        vgroup_formatter=format_fixed_thr, hgroup_formatter=format_fixed_thr
    )

    #fig.tight_layout()

    fig.savefig('test.png', bbox_inches='tight')

    plt.show()
    #import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

