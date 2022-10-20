
from multiprocessing import Process
from pathlib import Path
from functools import lru_cache
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hong2p import util, olf

from al_analysis import (ij_roi_responses_cache, get_fly_roi_ids, dropna,
    plot_all_roi_mean_responses, mocorr_concat_tiff_basename, init_logger
)
import al_analysis as al


def _get_odor_name_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index()[['odor1','odor2']].applymap(olf.parse_odor_name)


@lru_cache(maxsize=6)
def cached_load_movie(*recording_keys):
    print('loading movie (cache miss)')
    return al.load_movie(*recording_keys, min_input='mocorr')


def extract_ij_responses(analysis_dir, roi_index, roiset_path=None):

    analysis_dir = Path(analysis_dir).resolve()
    fly_dir = analysis_dir.parent

    try:
        int(fly_dir.name)

    # If we were given input directory that was already a fly directory, rather than a
    # recording directory (which is a child of a fly directory). Happens if plotting
    # triggered from the mocorr_concat.tif in ImageJ
    except ValueError:
        try:
            int(analysis_dir.name)
        except ValueError:
            raise FileNotFoundError('input did not seem to be either fly / recording '
                'dir'
            )

        fly_dir = analysis_dir

    # This should generally be a symlink to a TIFF from a particular suite2p run.
    mocorr_concat_tiff = fly_dir / mocorr_concat_tiff_basename

    # TODO maybe assert mocorr_concat_tiff exists if input was fly dir and not analysis
    # dir? otherwise which recording are we supposed to use?

    if mocorr_concat_tiff.exists():
        # Not actually getting a particular ThorImage dir w/ this fn here, but rather
        # the raw data fly directory that should contain all of the relevant ThorImage
        # dirs.
        raw_fly_dir = al.analysis2thorimage_dir(fly_dir)
        match_str = str(raw_fly_dir)
    else:
        thorimage_dir = al.analysis2thorimage_dir(analysis_dir)
        match_str = str(thorimage_dir)

    del analysis_dir

    date_str = fly_dir.parts[-2]
    keys_and_paired_dirs = list(
        al.paired_thor_dirs(matching_substrs=[match_str],
            start_date=date_str, end_date=date_str
        )
    )
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
        movie = cached_load_movie(*recording_keys)

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

    return df


def plot(subset_df):
    # TODO also/only try slightly changing odor ticklabel text color between changes in
    # odor name?
    vline_level_fn = lambda odor_str: olf.parse_odor_name(odor_str)

    # TODO TODO if i ever allow plotting multiple (from raw data where i don't yet want
    # to merge across planes, cause i'm dealing w/ single planes), specify Z in name in
    # plot (at least when there are two w/ same name, from diff planes)
    # (should i support same name in same plane?)

    # The cached ROIs will have '/' in ROI name (e.g. '3-30/1/DM4'), and only the last
    # row from the newly extracted data will not. We want a white line between these two
    # groups of data.
    hline_level_fn = lambda roi_name_str: '/' in roi_name_str

    # TODO TODO normalize within each fly (if comparing)? or at least option to?
    # (+ maybe label colorbars differently in each case if so)

    # TODO make colorbar generally/always the height of the main Axes (maybe a bit
    # larger if just ~one row?)
    # TODO want to use roi_sortkeys (could try to put in same order as arguments passed?
    # or always named ones first?)?
    fig, _ = plot_all_roi_mean_responses(subset_df, odor_sort=False,
        hline_level_fn=hline_level_fn, vline_level_fn=vline_level_fn
    )

    # didn't work in isolation
    #plt.show(block=False)

    # works but OS soon thinks the plot is unresponsive, and can't use controls.
    # adding plt.pause(0.001) in `while True` loop in main didn't fix it.
    #plt.draw()
    #plt.pause(0.001)

    # TODO TODO some way to make this non-blocking
    # (AND either replace current plot w/ new data or add new data to current plot,
    # depending on arguments)
    plt.show()


# TODO somehow profile just client calls to this, to see why even w/ movie and
# cached_responses_df loaded, it still takes ~0.34s (of ~0.45s loading movie but not
# cached df)
def load_and_plot(args):
    global cached_responses_df

    load_start_s = time.time()

    roi_strs = args.roi_strs

    analysis_dir = args.analysis_dir
    roi_index = args.roi_index
    roiset_path = args.roiset_path
    compare = not args.no_compare

    plot_pair_data = args.pairs
    plot_other_odors = args.other_odors

    subset_dfs = []
    if analysis_dir is not None:
        new_df = extract_ij_responses(analysis_dir, roi_index, roiset_path=roiset_path)

        new_roi_names = new_df.columns
        # would need to relax + modify code if i wanted to ever return multiple ROIs
        # from one extract_ij_responses call
        assert len(new_roi_names) == 1
        new_roi_name = new_roi_names[0]

        # So that if it's just a numbered ROI like '5', it doesn't pull up all the
        # 'DM5',etc data for comparison.
        if not al.is_ijroi_certain(new_roi_name):
            compare = False

        roi_strs.append(new_roi_name)

        subset_dfs.append(new_df)

    if compare or analysis_dir is None:
        assert len(roi_strs) > 0

        # TODO would need to either cache all of the df (might only save on load cost,
        # which is probably small) or have keys be the same thing we use to subset the
        # df, which would be more complicated and might not make senes
        df = pd.read_pickle(ij_roi_responses_cache)

        if analysis_dir is not None and not plot_other_odors:
            # TODO or (more work, but...) consider equal if within ~1-2 orders of
            # magnitude? or option to match exactly only?
            fly_odor_set = {tuple(x[1:]) for x in
                _get_odor_name_df(new_df).itertuples()
            }

            shared_odors = _get_odor_name_df(df).apply(
                tuple, axis='columns').isin(fly_odor_set).values

            # TODO maybe grey out names of odors not having matching concentration?
            # TODO TODO change odor sorting so odors w/ same name are next to each
            # other, regardless of panel? how to implement? maybe just define one
            # order for everything besides diagnostic panel? or prefer panel(s) from
            # current fly for name order somehow?

            df = df[shared_odors].copy()

        fly_roi_ids = get_fly_roi_ids(df)
        df.columns = fly_roi_ids
        df.columns.name = 'roi'

        # TODO maybe move this earlier and explicitly group on all vars in the
        # column index other than the thorimage_id level (which should be the only
        # one we are really aggregating across) (so that i can use it here and also
        # in al_analysis)
        # TODO TODO add comment on what exactly this is doing (where are duplicates
        # coming from? looking back at what df columns are before changing them
        # above might remind me.
        def merge_dupe_cols(gdf):
            # As long as this doesn't trip, we don't have to worry about choosing
            # which column to take data from: there will only ever be at most one
            # not NaN.
            assert not (gdf.notna().sum(axis='columns') > 1).any()
            ser = gdf.bfill(axis='columns').iloc[:, 0]
            return ser

        df = df.groupby('roi', axis='columns', sort=False).apply(merge_dupe_cols)

        matching = np.any([df.columns.str.endswith(x) for x in roi_strs], axis=0)
        if matching.sum() == 0:
            raise ValueError(f'no ROIs matching any of {roi_strs}')

        cached_responses_df = dropna(df.loc[:, matching])

        # Putting at start so it should be plotted above
        subset_dfs.insert(0, cached_responses_df)

    subset_df = pd.concat(subset_dfs, axis='columns', verify_integrity=True)

    if not plot_pair_data:
        subset_df = dropna(
            subset_df.loc[subset_df.index.get_level_values('is_pair') == False, :]
        )

    subset_df = subset_df[subset_df.index.get_level_values('odor1') != 'pfo @ 0']

    # TODO refactor (move sorting into plot_all_roi_mean_responses, after mean? need to
    # drop too though...) so that order from current panel is used, but data from any
    # ROIs from older panels that shared some odors is still shown. as is, it will
    # either only show for the first panel or only show separately
    #
    # I don't think order would necessarily be preserved wrt data if sort=False either,
    # and would want to ensure that before providing an option to not sort.
    sort = True
    if sort:
        # TODO option to switch between these? modify olf.sort_odors to allow keeping
        # certain panels in a fixed position + name_order, but to allow treating a
        # subset of panels as a single panel?
        # TODO vline_level_fn using panel data? might need to pass in some precomputed
        # list or something, which i think might require changes to hong2p.viz.matshow
        # handling
        #subset_df = al.sort_odors(subset_df)

        # TODO or determine order by sorting by how much old data we have for each odor
        # name?

        subset_df = olf.sort_odors(subset_df)

    analysis_time_s = time.time() - load_start_s
    print(f'loading/analyzing took {analysis_time_s:.2f}s')

    plot(subset_df)
