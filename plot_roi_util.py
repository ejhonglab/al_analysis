
from multiprocessing import Process
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

from hong2p import util, olf
from hong2p.types import Pathlike

from al_analysis import (ij_roi_responses_cache, get_fly_roi_ids, dropna,
    plot_all_roi_mean_responses, mocorr_concat_tiff_basename
)
import al_analysis as al


def _get_odor_name_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index()[['odor1','odor2']].applymap(olf.parse_odor_name)

# Rough hack to get cache to not crash the process on my laptop, but to be more useful
# as a cache on my lab computer.
_meminfo = psutil.virtual_memory()
total_mem_GB = _meminfo.total / 1e9
if total_mem_GB < 10:
    CACHE_MAX_SIZE = 1
else:
    # 6 = 2 flies w/ 3 movies each
    CACHE_MAX_SIZE = 6

@lru_cache(maxsize=CACHE_MAX_SIZE)
def cached_load_movie(*recording_keys):
    return al.load_movie(*recording_keys, min_input='mocorr')


def extract_ij_responses(input_dir: Pathlike, roi_index: int,
    roiset_path: Optional[Pathlike] = None) -> pd.DataFrame:
    """
    Args:
        input_dir: directory with either a single recording's TIFFs/ROIs, or a directory
            containing all such directories for a given fly

        roi_index: index of ImageJ ROI to analyze (same as in ROI manager, 0-indexed)

        roiset_path: if passed, load ROIs from this, rather than from <input_dir>

    Currently returns a dataframe with only a single column level ('roi', just the
    string ROI name). No levels for ['date','fly_num','thorimage_id'] that we start with
    from the loaded al_analysis.py output.
    """
    # NOTE: this will also resolve symlinks (which I didn't want and complicated
    # match_str calculation, but can't figure out a pathlib way to resolve just relative
    # paths INCLUDING stuff like '/..' at the end of a path, without also resolving
    # symlinks.
    input_dir = Path(input_dir).resolve()
    fly_dir = input_dir.parent

    # doesn't deal w/ trailing '/..' (which it would need to)
    #input_dir = Path(input_dir).absolute()

    fly_dir = input_dir.parent
    try:
        int(fly_dir.name)

    # If we were given input directory that was already a fly directory, rather than a
    # recording directory (which is a child of a fly directory). Happens if plotting
    # triggered from the mocorr_concat.tif in ImageJ
    except ValueError:
        try:
            int(input_dir.name)
        except ValueError:
            raise FileNotFoundError('input did not seem to be either fly / recording '
                'dir'
            )

        fly_dir = input_dir

    # This should generally be a symlink to a TIFF from a particular suite2p run.
    mocorr_concat_tiff = fly_dir / mocorr_concat_tiff_basename

    # TODO maybe assert mocorr_concat_tiff exists if input was fly dir and not input
    # dir? otherwise which recording are we supposed to use?

    if mocorr_concat_tiff.exists():
        # Not actually getting a particular ThorImage dir w/ this fn here, but rather
        # the raw data fly directory that should contain all of the relevant ThorImage
        # dirs.
        raw_fly_dir = al.analysis2thorimage_dir(fly_dir)
        match_parts = raw_fly_dir.parts[-2:]
    else:
        thorimage_dir = al.analysis2thorimage_dir(input_dir)
        match_parts = thorimage_dir.parts[-3:]

    match_str = str(Path(*match_parts))

    date_str = fly_dir.parts[-2]
    keys_and_paired_dirs = list(
        al.paired_thor_dirs(matching_substrs=[match_str],
            start_date=date_str, end_date=date_str,
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
        del movie
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

    # TODO TODO was i dropping is_pair == True stuff before building up ij trace
    # cache?  should i do that here too? or should i not do that when forming cache?
    # probably latter...
    # TODO maybe provide option to drop it?

    df = pd.concat(subset_dfs, verify_integrity=True)

    return df


def plot(subset_df):

    fly_roi_sortkeys = []
    for roi_str in subset_df.columns:

        from_cache = '/' in roi_str
        if from_cache:
            date_str, fly_str, roi_name = roi_str.split('/')
            is_newly_analyzed = False
        else:
            is_newly_analyzed, date_str, fly_str, roi_name = (True, '', '', roi_str)

        if len(newly_analyzed_roi_names) == 0:
            roi_key = roi_name
        else:
            if from_cache:
                # These should all be clean "certain" ROI names, which is also what our
                # cache input should be (if comparing to newly analyzed data).
                roi_key = newly_analyzed_roi_strs.index(roi_name)
            else:
                roi_key = newly_analyzed_roi_names.index(roi_name)

        # First component of tuple will put newly analyzed stuff at end.
        fly_roi_sortkeys.append((is_newly_analyzed, roi_key, '', ''))

    # TODO TODO maybe replace [h|v]lines w/ black lines, or change the missing-data
    # color to something like grey (from white), so that it's easier to see the
    # divisions against a border of a lot of missing data

    # TODO also/only try slightly changing odor ticklabel text color between changes in
    # odor name?
    vline_level_fn = lambda odor_str: olf.parse_odor_name(odor_str)

    # The cached ROIs will have '/' in ROI name (e.g. '3-30/1/DM4'), and only the last
    # row from the newly extracted data will not. We want a white line between these two
    # groups of data.
    hline_level_fn = lambda roi_str: ('/' in roi_str, roi_str.split('/')[-1])

    # TODO TODO normalize within each fly (if comparing)? or at least option to?
    # (+ maybe label colorbars differently in each case if so)

    # TODO make colorbar generally/always the height of the main Axes (maybe a bit
    # larger if just ~one row?)
    # TODO want to use roi_sortkeys (could try to put in same order as arguments passed?
    # or always named ones first?)?
    fig, _ = plot_all_roi_mean_responses(subset_df, odor_sort=False,
        roi_sortkey=fly_roi_sortkeys,
        hline_level_fn=hline_level_fn, vline_level_fn=vline_level_fn
    )
    plt.show()


newly_analyzed_dfs = []
newly_analyzed_roi_names = []
newly_analyzed_roi_strs = []

most_recent_plot_proc = None
keep_comparison_to_cache = False
plotting_processes = []

def load_and_plot(args):
    global newly_analyzed_dfs
    global newly_analyzed_roi_names
    global newly_analyzed_roi_strs
    global most_recent_plot_proc
    global keep_comparison_to_cache
    global plotting_processes

    # TODO TODO TODO add option to load hallem data + compare as well
    # (not sure if i want in the same figure or not...)
    # (take int and plot top-correlating-n glomeruli)

    roi_strs = args.roi_strs

    analysis_dir = args.analysis_dir
    roi_index = args.roi_index
    roiset_path = args.roiset_path
    compare = not args.no_compare

    plot_pair_data = args.pairs
    plot_other_odors = args.other_odors

    add_to_plot = args.add

    # TODO might want to instead change most_recent_plot_proc to point to most recent
    # still-alive process (b/c we might have manually closed the otherwise most recent
    # one) (then don't set add_to_plot = False here)
    if most_recent_plot_proc is not None and not most_recent_plot_proc.is_alive():
        most_recent_plot_proc = None
        add_to_plot = False

    plotting_processes = [p for p in plotting_processes if p.is_alive()]
    if len(plotting_processes) == 0:
        add_to_plot = False

    if add_to_plot:
        subset_dfs = list(newly_analyzed_dfs)
    else:
        subset_dfs = []

        newly_analyzed_dfs = []
        newly_analyzed_roi_names = []
        newly_analyzed_roi_strs = []
        keep_comparison_to_cache = False

    if analysis_dir is not None:
        new_df = extract_ij_responses(analysis_dir, roi_index, roiset_path=roiset_path)

        new_roi_names = new_df.columns
        # would need to relax + modify code if i wanted to ever return multiple ROIs
        # from one extract_ij_responses call
        assert len(new_roi_names) == 1
        new_roi_name = new_roi_names[0]

        # So that if it's just a numbered ROI like '5', it doesn't pull up all the
        # 'DM5',etc data for comparison.
        if not al.is_ijroi_named(new_roi_name):
            compare = False
        else:
            keep_comparison_to_cache = True

            # NOTE: just going to support the syntax '<roi name 1>|<roi name 2>|...',
            # rather than anything more complicated involving square brackets, as it
            # would complicate parsing. Also support having a single '?' at end.
            new_roi_str = new_roi_name.strip('?')
            for roi_name in new_roi_str.split('|'):
                if roi_name not in newly_analyzed_roi_strs:
                    newly_analyzed_roi_strs.append(roi_name)

        if new_roi_name in newly_analyzed_roi_names:
            for ndf in newly_analyzed_dfs:
                assert len(ndf.columns) == 1
                ndf_roi_name = ndf.columns[0]

                if ndf_roi_name == new_roi_name:
                    # If we've gotten this far, we must already have some plots open
                    # (with this ROI too), so no need to update the plot if the data is
                    # also equal.
                    if ndf.equals(new_df):
                        return

                    # Script must have been called with an ROI of the same name, but
                    # drawn differently (perhaps in a different plane).
                    i = 1
                    while True:
                        # Not using '-' as separator, to differentiate from what
                        # ImageJ uses in ROI manager.
                        suffixed_name = f'{new_roi_name}_{i}'
                        if suffixed_name not in newly_analyzed_roi_names:
                            assert len(new_df.columns) == 1
                            assert new_df.columns[0] == new_roi_name
                            new_roi_name = suffixed_name
                            new_df.columns = [suffixed_name]
                            break

                        i += 1

        subset_dfs.append(new_df)

        newly_analyzed_roi_names.append(new_roi_name)
        newly_analyzed_dfs.append(new_df)

    if compare or analysis_dir is None or keep_comparison_to_cache:
        roi_strs.extend(newly_analyzed_roi_strs)
        assert len(roi_strs) > 0

        df = pd.read_pickle(ij_roi_responses_cache)

        if analysis_dir is not None and not plot_other_odors:
            # TODO or (more work, but...) consider equal if within ~1-2 orders of
            # magnitude? or option to match exactly only?
            # TODO TODO a coarse-concentration-matching for odors might be useful in
            # some other places too

            # TODO maybe grey out names of odors not having matching concentration?
            fly_odor_set = {tuple(x[1:]) for x in
                _get_odor_name_df(new_df).itertuples()
            }

            shared_odors = _get_odor_name_df(df).apply(
                tuple, axis='columns').isin(fly_odor_set).values

            df = df[shared_odors].copy()

        # TODO just factor this into plotting fn in al_analysis/hong2p, and leave all
        # the metadata on the column multiindex?
        fly_roi_ids = get_fly_roi_ids(df)
        df.columns = fly_roi_ids
        df.columns.name = 'roi'

        # TODO maybe it should match the last '/' separated part exactly?
        # any reason i didn't want that?
        # (then i could get get index of matching name in newly analyzed ROI names to
        # sort these...)
        matching = np.any([df.columns.str.endswith(x) for x in roi_strs],
            axis=0
        )
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

    if add_to_plot and most_recent_plot_proc is not None:
        most_recent_plot_proc.terminate()

    # currently just counting on the process eventually terminating, and thus the
    # corresponding Process object being cleaned up
    proc = Process(target=plot, args=(subset_df,))
    most_recent_plot_proc = proc
    proc.start()

    plotting_processes.append(proc)

