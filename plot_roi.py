#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hong2p import util

from al_analysis import (ij_roi_responses_cache, get_fly_roi_ids, dropna,
    plot_all_roi_mean_responses, mocorr_concat_tiff_basename
)
import al_analysis as al


# TODO maybe add an option to run as some kind of singleton, and like add stuff to a
# plot rather than making a new one (if called with some arguments the same at least,
# i.e. same fly)? otherwise not sure easiest way to have imagej plot multiple...
# TODO would probably be easier to have imagej maintain list of indices on successive
# 'p' presses, and have it call same script with that list of indices, but approach
# suggested above would probably be faster... (movie and stuff already loaded)
def extract_ij_responses(analysis_dir, roi_index, roiset_path=None, sort=True):
    analysis_dir = Path(analysis_dir)
    fly_dir = analysis_dir.parent

    # This should generally be a symlink to a TIFF from a particular suite2p run.
    mocorr_concat_tiff = fly_dir / mocorr_concat_tiff_basename

    if mocorr_concat_tiff.exists():
        # Not actually getting a particular ThorImage dir w/ this fn here, but rather
        # the raw data fly directory that should contain all of the relevant ThorImage
        # dirs.
        raw_fly_dir = al.analysis2thorimage_dir(fly_dir)
        print(raw_fly_dir)
        match_str = str(raw_fly_dir)
    else:
        thorimage_dir = al.analysis2thorimage_dir(analysis_dir)
        match_str = str(thorimage_dir)

    date_str = analysis_dir.parts[-3]
    keys_and_paired_dirs = list(
        al.paired_thor_dirs(matching_substrs=[match_str],
            start_date=date_str, end_date=date_str
        )
    )
    del analysis_dir
    # TODO make conditional on not loading all fly data / no mocorr concat (/ delete)
    #assert len(keys_and_paired_dirs) == 1

    # TODO maybe just use trial_frames_and_odors.json when available...
    # (in both concat / single recording case) (though if i would need to back convert
    # odor data from str, maybe not)

    subset_dfs = []
    for (_, _), (thorimage_dir, thorsync_dir) in keys_and_paired_dirs:
        recording_keys = thorimage_dir.parts[-3:]
        analysis_dir = al.get_analysis_dir(*recording_keys)

        _, _, odor_lists = util.thorimage2yaml_info_and_odor_lists(thorimage_dir)

        bounding_frames = al.assign_frames_to_odor_presentations(thorsync_dir,
            thorimage_dir, analysis_dir
        )
        movie = al.load_movie(*recording_keys, min_input='mocorr')

        if roiset_path is None:
            roiset_path = analysis_dir

        masks = util.ijroi_masks(roiset_path, thorimage_dir)

        # TODO refactor this + hong2p.util.ij_traces, to share a bit more code (maybe
        # break out some of ij_traces into another helper fn?)
        # TODO silence this here / in general
        traces = pd.DataFrame(util.extract_traces_bool_masks(movie, masks))
        traces.index.name = 'frame'
        traces.columns.name = 'roi'
        traces.columns = masks.roi_name.values

        trial_df = al.compute_trial_stats(traces, bounding_frames, odor_lists)

        subset_df = trial_df.iloc[:, [roi_index]]

        panel = al.get_panel(thorimage_dir)
        is_pair = al.is_pairgrid(odor_lists)

        new_level_names = ['panel', 'is_pair']
        new_level_vals = [panel, is_pair]
        subset_df = util.addlevel(subset_df, new_level_names, new_level_vals)

        subset_dfs.append(subset_df)

    # TODO TODO TODO was i dropping is_pair == True stuff before building up ij trace
    # cache?  should i do that here too? or should i not do that when forming cache?
    # probably latter...
    # TODO maybe provide option to drop it?

    df = pd.concat(subset_dfs, verify_integrity=True)

    if sort:
        df = al.sort_odors(df)

    return df


# TODO profile, especially the analysis_dir path
def main():
    parser = argparse.ArgumentParser(description='Reads and plots ROI stats from '
        f'{ij_roi_responses_cache}'
    )
    # TODO TODO still check it is specified in case where -a not passed
    parser.add_argument('roi_strs', nargs='*', help='ROI names (e.g. DM5, 3-30/1/0)')
    parser.add_argument('-a', '--analysis-dir', help='If passed, analyze data from this'
        f' directory, rather than loading data from {ij_roi_responses_cache}.'
    )
    # TODO store as int(s)
    # TODO change to roi_indices (tho would i then need to select all from ROI manager
    # rather than overlay? might be tricky)
    parser.add_argument('-i', '--roi-index', help='The index of the ROI to analyze. '
        'Only relevant when also passing -a/--analysis-dir.'
    )
    parser.add_argument('-r', '--roiset-path', help='Path to the RoiSet.zip to load '
        'ImageJ ROIs from. Only relevant in -a/--analysis-dir case.'
    )
    args = parser.parse_args()
    roi_strs = args.roi_strs
    analysis_dir = args.analysis_dir
    roi_index = int(args.roi_index)
    roiset_path = args.roiset_path

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

        matching = np.any([df.columns.str.endswith(x) for x in roi_strs], axis=0)
        if matching.sum() == 0:
            raise ValueError(f'no ROIs matching any of {roi_strs}')

        subset_df = dropna(df.loc[:, matching])

    # TODO TODO TODO option to show all cached stuff w/ same substring above data from
    # analysis_dir
    # TODO TODO maybe another option to show everything that did NOT match substring as
    # well (matching cached -> specific indexed ROI not in cache -> NON-matching cached)
    else:
        subset_df = extract_ij_responses(analysis_dir, roi_index,
            roiset_path=roiset_path
        )

    # TODO TODO if i ever allow plotting multiple (from raw data where i don't yet want
    # to merge across planes, cause i'm dealing w/ single planes), specify Z in name in
    # plot (at least when there are two w/ same name, from diff planes)
    # (should i support same name in same plane?)

    # TODO want to use roi_sortkeys (could try to put in same order as arguments passed?
    # or always named ones first?)
    fig, _ = plot_all_roi_mean_responses(subset_df, odor_sort=False)

    plt.show()


if __name__ == '__main__':
    main()

