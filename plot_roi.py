#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from hong2p import viz

from al_analysis import (ij_roi_responses_cache, get_fly_roi_ids, dropna,
    plot_all_roi_mean_responses
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roi_strs', nargs='+', help='ROI names (e.g. DM5, 3-30/1/0)')
    args = parser.parse_args()
    roi_strs = args.roi_strs

    df = pd.read_pickle(ij_roi_responses_cache)
    fly_roi_ids = get_fly_roi_ids(df)
    df.columns = fly_roi_ids
    df.columns.name = 'roi'

    df = dropna(df.loc[df.index.get_level_values('is_pair') == False, :])
    df = df[df.index.get_level_values('odor1') != 'pfo @ 0']

    # TODO maybe move this earlier and explicitly group on all vars in the column index
    # other than the thorimage_id level (which should be the only one we are really
    # aggregating across) (so that i can use it here and also in al_analysis)
    def merge_dupe_cols(gdf):
        # As long as this doesn't trip, we don't have to worry about choosing which
        # column to take data from: there will only ever be at most one not NaN.
        assert not (gdf.notna().sum(axis='columns') > 1).any()
        ser = gdf.bfill(axis='columns').iloc[:, 0]
        return ser

    df = df.groupby('roi', axis='columns', sort=False).apply(merge_dupe_cols)

    matching = np.any([df.columns.str.endswith(x) for x in roi_strs], axis=0)
    subset_df = dropna(df.loc[:, matching])

    # TODO want to use roi_sortkeys (could try to put in same order as arguments passed?
    # or always named ones first?)
    fig, _ = plot_all_roi_mean_responses(subset_df, odor_sort=False)

    plt.show()


if __name__ == '__main__':
    main()

