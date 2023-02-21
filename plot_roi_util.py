
from multiprocessing import Process
from pathlib import Path
from functools import lru_cache
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

from hong2p import util, olf, viz
from hong2p.roi import extract_traces_bool_masks, ijroi_masks
from hong2p.types import Pathlike
from drosolf import orns

from al_analysis import (ij_roi_responses_cache, get_fly_roi_ids, dropna,
    plot_all_roi_mean_responses, mocorr_concat_tiff_basename
)
import al_analysis as al


def _get_odor_name_df(df: pd.DataFrame) -> pd.DataFrame:
    # TODO return columns should always just be ['name1','name2'] or something
    df = df.reset_index()[['odor1','odor2']].applymap(olf.parse_odor_name)
    df.columns = ['name1', 'name2']
    return df

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
    for (date, fly_num), (thorimage_dir, thorsync_dir) in keys_and_paired_dirs:
        recording_keys = thorimage_dir.parts[-3:]

        analysis_dir = al.get_analysis_dir(*recording_keys)

        _, _, odor_lists = util.thorimage2yaml_info_and_odor_lists(thorimage_dir)

        bounding_frames = al.assign_frames_to_odor_presentations(thorsync_dir,
            thorimage_dir, analysis_dir
        )
        movie = cached_load_movie(*recording_keys)

        if roiset_path is None:
            roiset_path = analysis_dir

        masks = ijroi_masks(roiset_path, thorimage_dir)

        # TODO TODO refactor this + hong2p.util.ij_traces, to share a bit more code
        # (maybe break out some of ij_traces into another helper fn?)
        # TODO silence this here / in general
        traces = pd.DataFrame(extract_traces_bool_masks(movie, masks))
        del movie
        traces.index.name = 'frame'
        # TODO have this name preserved at output? i'm assuming it's not now
        traces.columns.name = 'roi'
        traces.columns = masks.roi_name.values

        trial_df = al.compute_trial_stats(traces, bounding_frames, odor_lists)

        subset_df = trial_df.iloc[:, [roi_index]]

        panel = al.get_panel(thorimage_dir)
        is_pair = al.is_pairgrid(odor_lists)

        new_level_names = ['panel', 'is_pair']
        new_level_vals = [panel, is_pair]
        subset_df = util.addlevel(subset_df, new_level_names, new_level_vals)

        subset_df.columns.name = 'roi'
        subset_df = util.addlevel(subset_df,
            ['from_hallem','newly_analyzed','date','fly_num'],
            [False, True, None, None], axis='columns'
        )

        subset_dfs.append(subset_df)

    df = pd.concat(subset_dfs, verify_integrity=True)

    return df


def plot(df, sort_rois=True, show=True, **kwargs):
    # For sorting based on order ROIs added to new analysis, so if plot is updated to
    # include new data, the new data is always towards the bottom of each section.
    def roi_sort_index(col_index_vals):
        index_vals = dict(zip(df.columns.names, col_index_vals))
        roi = index_vals['roi']
        if index_vals['newly_analyzed']:
            return newly_analyzed_roi_names.index(roi)

        # TODO should i also just disable --add if --hallem?
        elif index_vals['from_hallem']:
            return None

        else:
            return newly_analyzed_roi_strs.index(roi)

    sort_first_on = None
    if len(newly_analyzed_roi_names) > 0:
        assert all(x in df.columns.names for x in ('newly_analyzed', 'roi'))
        sort_first_on = list(zip(
            df.columns.get_level_values('newly_analyzed'),
            df.columns.map(roi_sort_index)
        ))
        if df.columns.get_level_values('from_hallem').any():
            sort_first_on = [(h,) + x for h, x in zip(
                df.columns.get_level_values('from_hallem'), sort_first_on
            )]

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
    fig, _ = plot_all_roi_mean_responses(df, odor_sort=False,
        roi_sort=sort_rois, sort_rois_first_on=sort_first_on,
        hline_level_fn=hline_level_fn, vline_level_fn=vline_level_fn, **kwargs
    )
    if show:
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

    roi_strs = args.roi_strs

    analysis_dir = args.analysis_dir
    roi_index = args.roi_index
    roiset_path = args.roiset_path

    compare = not args.no_compare
    hallem = args.hallem

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
        assert len(new_df.columns) == 1

        new_roi_names = new_df.columns.get_level_values('roi').unique()
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
                ndf_roi_name = ndf.columns.get_level_values('roi')[0]

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
                            new_roi_name = suffixed_name

                            index_df = new_df.columns.to_frame(index=False)
                            index_df.loc[:, 'roi'] = new_roi_name
                            new_df.columns = pd.MultiIndex.from_frame(index_df)
                            break

                        i += 1

        subset_dfs.append(new_df)

        fly_odor_set = {tuple(x[1:]) for x in _get_odor_name_df(new_df).itertuples()}


        # TODO want to actually limit to top n? leaning no, b/c ruling stuff out also
        # useful...
        if hallem:
            # Absolute ORN firing rates (reported deltas, w/ reported SFR added back in)
            orn_df = orns.orns(columns='glomerulus')

            # Appending asterisks for 'DM3' and 'DM3.1' to avoid confusion w/ just 'DM3'
            orn_df.columns = orn_df.columns.map(
                lambda x: f'{x}*' if x.startswith('DM3') else x
            )
            # Since I'm not sure whether to compute DM3 signal as a weighted average of
            # 47a ("DM3.1") and 33b ("DM3") inputs, or which weights to use if so.
            '''
            orn_df = orn_df.drop(
                columns=[c for c in orn_df.columns if c.startswith('DM3')]
            )
            assert not orn_df.columns.duplicated().any()
            '''

            orn_df = orn_df.rename(index={
                'b-citronellol': 'B-citronellol',
                'isopentyl acetate': 'isoamyl acetate',
                'E2-hexenal': 'trans-2-hexenal',
            })

            orn_df = orn_df.rename(index=olf.odor2abbrev)

            # TODO TODO replace w/ some kind of coarse matching on concentration
            orn_df.index = orn_df.index.str.cat([' @ -3'] * len(orn_df))

            # TODO TODO also use lower concentration data (not loaded by current version
            # of drosolf)

            odor1_name_set = {x[0] for x in fly_odor_set if x[1] == None}

            # TODO i already have a fn i can use for this?
            new_df_nomix = new_df[
                new_df.index.get_level_values('odor2') == olf.solvent_str
            ]

            new_df_mean = new_df_nomix.groupby('odor1', sort=False).mean()
            assert new_df_mean.shape[1] == 1
            new_df_mean = new_df_mean.iloc[:, 0]

            hallem_overlap_df = orn_df[orn_df.index.isin(new_df_mean.index)]

            # TODO TODO consider normalizing (within all hallem, and all my data / fly)
            # to [0, 1], then do euclidean or something? don't want corr w/ glomeruli
            # that are essentially nonresponsive to count for as much as it does...
            # ig i can just ignore as long as i'm showing responses...
            hallem_corrs = hallem_overlap_df.corrwith(new_df_mean).sort_values(
                ascending=False
            )
            hallem_overlap_df = hallem_overlap_df.loc[:, hallem_corrs.index]

            # TODO subpanel w/ hallem_corrs in small part on left and full responses on
            # right? maybe just two plots?

            hallem_corrs.index.name = 'roi'
            hallem_corrs = hallem_corrs.to_frame()

            hallem_overlap_df.columns.name = 'roi'
            hallem_overlap_df.index.name = 'odor1'

            # TODO try to put in a subplot in the hallem plot below (just don't divide
            # horizontal space evenly)
            #'''
            def plot_corrs(df):
                fig, _ = viz.matshow(df, cmap='RdBu_r')
                ax = plt.gca()
                ax.set_title(new_roi_name)
                ax.get_xaxis().set_visible(False)
                plt.show()

            proc = Process(target=plot_corrs, args=(hallem_corrs,))
            proc.start()
            #'''


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
            # some other places too (see Index.get_loc?)

            # TODO maybe grey out names of odors not having matching concentration?

            shared_odors = _get_odor_name_df(df).apply(
                tuple, axis='columns').isin(fly_odor_set).values

            df = df[shared_odors].copy()

        matching = np.any([df.columns.get_level_values('roi') == x for x in roi_strs],
            axis=0
        )
        if matching.sum() == 0:
            # TODO just print? not sure if i'll be able to see either, as i'm currently
            # calling from imagej
            warnings.warn(f'no ROIs matching any of {roi_strs} in cached responses')
        else:
            df = util.addlevel(df, ['from_hallem','newly_analyzed'], [False, False],
                axis='columns'
            )
            cached_responses_df = dropna(df.loc[:, matching])

            # Putting at start so it should be plotted above
            subset_dfs.insert(0, cached_responses_df)

    if hallem:
        new_level_names = ['panel', 'is_pair', 'odor2', 'repeat']
        new_level_vals = ['hallem', False, olf.solvent_str, 0]
        hallem_overlap_df = util.addlevel(hallem_overlap_df, new_level_names,
            new_level_vals
        )
        hallem_overlap_df = hallem_overlap_df.reorder_levels(new_df.index.names)

        hallem_overlap_df = util.addlevel(hallem_overlap_df,
            ['from_hallem','newly_analyzed','date','fly_num'],
            [True, False, None, None], axis='columns'
        )

        subset_dfs.append(hallem_overlap_df)

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

    # TODO TODO maybe if the name is shared (prefix? cause DM3[.1] thing) w/ current roi
    # name (prefix again, cause '?'?), then call out that one, maybe by bolding name in
    # yticklabels of both hallem plots or something? or sorting that one out?
    if hallem:
        n_overlapping = len(hallem_overlap_df.columns)

        hallem_overlap_df = subset_df.iloc[:, -n_overlapping:]
        subset_df = subset_df.iloc[:, :-n_overlapping]

        # Constrained layout causes width_ratios thing to fail.
        # TODO nice way to do something similar within the context of constrained
        # layout?
        @viz.no_constrained_layout
        def hallem_comparison_plot():
            # Passing width_ratios directly here only supported as of matplotlib>=3.6.0
            fig, (corr_ax, resp_ax) = plt.subplots(ncols=2, width_ratios=[1, 3])

            viz.matshow(hallem_corrs, ax=corr_ax, cmap='RdBu_r')
            #corr_ax.set_title(new_roi_name)
            corr_ax.get_xaxis().set_visible(False)

            plot(hallem_overlap_df, ax=resp_ax, sort_rois=False, show=False,
                title=f'Hallem data (sorted by corr with {new_roi_name})'
            )
            plt.show()

        # TODO delete after debugging
        #hallem_comparison_plot()
        #

        #proc = Process(target=hallem_comparison_plot)
        #proc.start()

        # TODO TODO have this Axes be same physical size, so it's easier to compare
        # the two plots
        # TODO move up "title" (xlabel) on this one / tight_layout / something
        #'''
        proc = Process(target=plot, args=(hallem_overlap_df,),
            kwargs={'sort_rois': False,
                'title': f'Hallem data (sorted by corr with {new_roi_name})',
        })
        proc.start()
        #'''

    if add_to_plot and most_recent_plot_proc is not None:
        most_recent_plot_proc.terminate()

    _debug_plot = False
    if not _debug_plot:
        # currently just counting on the process eventually terminating, and thus the
        # corresponding Process object being cleaned up
        proc = Process(target=plot, args=(subset_df,))
        most_recent_plot_proc = proc
        proc.start()
        plotting_processes.append(proc)
    else:
        # So I can use a debugger
        plot(subset_df)

