#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hong2p import viz

from al_analysis import diverging_cmap, plot_corr


def main():
    ann_output_dir = Path('data/ann_model_outputs')
    # was originally operating directly out of here
    #ann_output_dir = Path.home() / 'Documents/MATLAB'

    plot_dir = Path('ann_outputs')
    plot_dir.mkdir(exist_ok=True)

    #orn_deltas = pd.read_csv(ann_output_dir / 'orn_deltas.csv', index_col='Row')
    # TODO would want to index to first 110 odors and glomeruli (of 51), where name does
    # not start with 'non-hallem'
    # (i named glomeruli not in ORN.HCList with 'non-hallem<i>')
    #print(f'{orn_deltas.shape=}')

    kcs = pd.read_csv(ann_output_dir / 'kcs.csv')
    print(f'{kcs.shape=}')
    #import ipdb; ipdb.set_trace()

    # TODO TODO TODO compare spike count mean(s) (along diff axes / all) between matt's
    # outputs and these

    kcs_binarized = kcs.copy()
    kcs_binarized[kcs_binarized >= 1] = 1
    # TODO precompute corr to not show x/yticklabels (to be consistent w/ other ones)
    plot_corr(kcs_binarized, plot_dir, 'kc_binarized_corr',
        title="Ann's model KCs (binarized)"
    )
    # TODO TODO try computing correlation w/ only non-silent cells? not sure there's any
    # reason to think ann might be doing that... might be more fair comparison in my
    # case though? since she doesn't actually know other ORNs don't have responses, but
    # i sorta should?

    orn_corr = pd.read_csv(ann_output_dir / 'orn_corr.csv', header=None)
    assert orn_corr.index.equals(orn_corr.columns)
    # TODO compare to mine (should be same)
    plot_corr(orn_corr, plot_dir, 'orn_corr', title='Hallem ORN deltas')

    kc_corr = pd.read_csv(ann_output_dir / 'kc_corr.csv', header=None)
    assert kc_corr.index.equals(kc_corr.columns)
    plot_corr(kc_corr, plot_dir, 'kc_corr', title="Ann's model KCs")

    # TODO TODO TODO actually compare response rates across ann / matt models
    # TODO TODO TODO compare ann / matt behavior on specific megamat 17 odors

    # TODO TODO TODO do the 'caron' connectivity outputs of matt's model look more
    # similar to these (maybe after binarizing?)

    # TODO TODO TODO how to get matt's outputs to be more like ann's before binarizing?
    # she's counting spikes? over what window?

    # TODO TODO to what extent is this the difference between 2k and 1.8k cells?

    #import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

