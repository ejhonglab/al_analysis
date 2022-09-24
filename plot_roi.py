#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hong2p import util

from al_analysis import (ij_roi_responses_cache, get_fly_roi_ids, dropna,
    plot_all_roi_mean_responses
)
import al_analysis as al


def extract_ij_responses(analysis_dir):
    analysis_dir = Path(analysis_dir)

    # TODO how to get thorsync dir for just one thorimage_dir?
    thorimage_dir = al.analysis2thorimage_dir(analysis_dir)

    date_str = analysis_dir.parts[-3]
    keys_and_paired_dirs = list(
        al.paired_thor_dirs(matching_substrs=[str(thorimage_dir)],
            start_date=date_str, end_date=date_str
        )
    )
    assert len(keys_and_paired_dirs) == 1
    thorsync_dir = keys_and_paired_dirs[0][1][1]

    _, _, odor_lists = util.thorimage2yaml_info_and_odor_lists(thorimage_dir)

    bounding_frames = al.assign_frames_to_odor_presentations(thorsync_dir,
        thorimage_dir, analysis_dir
    )

    recording_keys = analysis_dir.parts[-3:]
    # TODO maybe dont use this, as it might make it easier to use the mocorr_concat.tif
    # as input...
    movie = al.load_movie(*recording_keys, min_input='mocorr')
    # TODO TODO TODO expose options to not collapse across planes, b/c we won't want
    # that when calling with a particular imagej ROI we are playing around with
    # TODO TODO TODO will need some way of specifing the ROI plane too then... or maybe
    # i should just save one .roi file for the selected ROI, and only load that rather
    # than letting ij_traces load all of the RoiSet.zip
    traces, rois, z_indices = al.ij_traces(analysis_dir, movie)
    trial_df = al.compute_trial_stats(traces, bounding_frames, odor_lists)

    return trial_df


# TODO profile, especially the analysis_dir path
def main():
    parser = argparse.ArgumentParser(description='Reads and plots ROI stats from '
        f'{ij_roi_responses_cache}'
    )
    parser.add_argument('roi_strs', nargs='+', help='ROI names (e.g. DM5, 3-30/1/0)')
    parser.add_argument('-a', '--analysis-dir', help='If passed, analyze data from this'
        f' directory, rather than loading data from {ij_roi_responses_cache}.'
    )
    args = parser.parse_args()
    roi_strs = args.roi_strs
    analysis_dir = args.analysis_dir

    if analysis_dir is None:
        df = pd.read_pickle(ij_roi_responses_cache)
        fly_roi_ids = get_fly_roi_ids(df)
        df.columns = fly_roi_ids
        df.columns.name = 'roi'

        df = dropna(df.loc[df.index.get_level_values('is_pair') == False, :])
        df = df[df.index.get_level_values('odor1') != 'pfo @ 0']

        # TODO maybe move this earlier and explicitly group on all vars in the column
        # index other than the thorimage_id level (which should be the only one we are
        # really aggregating across) (so that i can use it here and also in al_analysis)
        def merge_dupe_cols(gdf):
            # As long as this doesn't trip, we don't have to worry about choosing which
            # column to take data from: there will only ever be at most one not NaN.
            assert not (gdf.notna().sum(axis='columns') > 1).any()
            ser = gdf.bfill(axis='columns').iloc[:, 0]
            return ser

        df = df.groupby('roi', axis='columns', sort=False).apply(merge_dupe_cols)

    # TODO TODO TODO options to make this work w/ all recordings for a given fly
    # TODO TODO TODO option to show all cached stuff w/ same substring above data from
    # analysis_dir
    else:
        df = extract_ij_responses(analysis_dir)

    matching = np.any([df.columns.str.endswith(x) for x in roi_strs], axis=0)
    if matching.sum() == 0:
        raise ValueError(f'no ROIs matching any of {roi_strs}')

    subset_df = dropna(df.loc[:, matching])

    # TODO want to use roi_sortkeys (could try to put in same order as arguments passed?
    # or always named ones first?)
    fig, _ = plot_all_roi_mean_responses(subset_df, odor_sort=False)

    plt.show()


if __name__ == '__main__':
    main()

