
from multiprocessing import Process
from pathlib import Path
from pprint import pformat
from functools import lru_cache
from typing import Optional
import warnings
import socket
from struct import pack
import traceback
# TODO check this is going to same place as plot_roi.py process that is calling stuff
# from here (init_logging should be configuring root logger now, so logging.<x> should
# work)
import logging
from os import getenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from matplotlib.backend_bases import MouseButton

from hong2p import util, olf, viz
from hong2p.roi import extract_traces_bool_masks, ijroi_masks
from hong2p.types import Pathlike
from drosolf import orns

from al_analysis import (ij_roi_responses_cache, dropna, plot_all_roi_mean_responses,
    mocorr_concat_tiff_basename, warn
)
import al_analysis as al


# TODO TODO fix plotting so it's always a fixed horizontal size (the actual Axes area),
# so that plots are easy to line up and compare to each other
# TODO in add_to_plot=True case, try to keep window in roughly the same position as
# closed one

# LINE_PROFILE is the same env var line_profiler (pip installed as line-profiler) uses
# to decide whether to run (when script is called normally)
do_profiling = getenv('LINE_PROFILE') == '1'
if do_profiling:
    # TODO possible to configure as if `-u 1`? does that matter for saving .lprof, or
    # just for viewing it (it the latter, doesn't matter here)?
    from line_profiler import profile
    # TODO also print about how to view results of profiling
    # (via `python -m line_profiler <.lprof file>` ?, which i think requires source
    # lines to not have changed to make sense)
else:
    # TODO will this screw with understanding any of the code (tracebacks / etc)?
    # some other way to have an optional wrapper (that when disabled does literally
    # nothing)?
    profile = lambda x: x

# hardcoded to same name as in plot_roi.py, which calls init_logger to set it up.
#
# TODO could try as a sub-logger (terminology?), like 'plot_roi.util' or something, but
# not sure that'd be useful
log = logging.getLogger('plot_roi')

# TODO try disabling constrained layout, to see how much that is affecting
# plotting speed

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
    warn('because system has <10GB total RAM, setting CACHE_MAX_SIZE=1. will need to '
        'load movies more often (will be slower)'
    )
else:
    # e.g. 2 flies w/ 5 recordings each, or one fly with 10 recordings.
    CACHE_MAX_SIZE = 10

@lru_cache(maxsize=CACHE_MAX_SIZE)
def cached_load_movie(*recording_keys):
    return al.load_movie(*recording_keys, min_input='mocorr')


@profile
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

    # TODO switch to calling one per fly (on concatenated stuff, when available), rather
    # than once per recording (would want to switch al_analysis.py behavior at the same
    # time, which would take a fair bit of work)
    # TODO TODO since i'm not actually computing best plane though, maybe i can still
    # just compute ROI mask once per fly (assuming ROIs never actually differ across
    # recordings, which they don't actually. always symlinked to diagnostics1)
    subset_dfs = []
    for (date, fly_num), (thorimage_dir, thorsync_dir) in keys_and_paired_dirs:
        recording_keys = thorimage_dir.parts[-3:]

        analysis_dir = al.get_analysis_dir(*recording_keys)

        # NOTE: still takes a reasonable fraction of a second
        # TODO cache? just use lru_cache decorator again? shouldn't change while doing
        # analysis really...
        _, _, odor_lists = util.thorimage2yaml_info_and_odor_lists(thorimage_dir)

        bounding_frames = al.assign_frames_to_odor_presentations(thorsync_dir,
            thorimage_dir, analysis_dir
        )
        movie = cached_load_movie(*recording_keys)

        if roiset_path is None:
            roiset_path = analysis_dir

        # NOTE: still takes a reasonable fraction of a second w/ 1 ROI
        #
        # TODO at least unless roiset_path points to another real file (after
        # resolving symlinks), cache mask computation as well!
        # (may not be very important now that i'm only computing mask for roi_index i
        # want?)
        masks = ijroi_masks(roiset_path, thorimage_dir, drop_maximal_extent_rois=False,
            indices=[roi_index]
        )

        # TODO TODO can i change to only compute w/ single mask i'm actually planning on
        # analyzing? is that why it's slow?
        #
        # TODO TODO refactor this + hong2p.util.ij_traces, to share a bit more code
        # (maybe break out some of ij_traces into another helper fn?)
        # TODO silence this here / in general
        traces = pd.DataFrame(extract_traces_bool_masks(movie, masks))
        del movie
        traces.index.name = 'frame'
        # TODO have this name preserved at output? i'm assuming it's not now
        traces.columns.name = 'roi'
        traces.columns = masks.roi_name.values

        # NOTE: still takes a reasonable fraction of a second w/ 1 ROI
        #
        # odor abbreviating happens inside here (in odor_lists_to_multiindex call)
        trial_df = al.compute_trial_stats(traces, bounding_frames, odor_lists)

        # now we should be subsetting via indices= kwarg to ijroi_masks above
        subset_df = trial_df

        panel = al.get_panel(thorimage_dir)
        is_pair = al.is_pairgrid(odor_lists)

        new_level_names = ['panel', 'is_pair']
        new_level_vals = [panel, is_pair]
        subset_df = util.addlevel(subset_df, new_level_names, new_level_vals)

        subset_df.columns.name = 'roi'
        subset_df = util.addlevel(subset_df,
            ['from_hallem', 'newly_analyzed', 'date', 'fly_num'],
            [False, True, None, None], axis='columns'
        )

        subset_dfs.append(subset_df)

    try:
        df = pd.concat(subset_dfs, verify_integrity=True)

    # 2023-10-16: currently only triggered in 2023-10-15/1 data where I repeated 2-mib
    # in it's own recording (validation2_4, because it was disconnected in
    # validation2_2).
    except ValueError as err:
        warn(traceback.format_exc())
        df = pd.concat(subset_dfs)

    return df


# TODO modify to also send ROI index (row clicked) (for ROIs that came from the current
# data)
def send_odor_index_to_imagej_script(odor_index):
    # TODO these the options i want (mostly unsure of socket.SOCK_STREAM)?
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # TODO this work for address? should i use 127.0.0.1? can i omit?
    #
    # same port as in overlay_rois_in_curr_z.py
    port = 49007

    _debug = False

    # TODO something more elegant than looping until connection is not refused? want to
    # continue showing plot too... make another thread?
    while True:
        try:
            log.debug(f'trying to connect socket to {port=}')
            if _debug:
                print(f'connecting on {port=}')

            client.connect(('0.0.0.0', 49007))
            break

        # TODO which error is this actually? what to catch (to be specific)?
        except:
            log.debug('could not connect! returning!')
            if _debug:
                print('could not connect! returning!')

            return

    log.debug('connected')
    if _debug:
        print('connected')

    # '!' = "network encoding" (big Endian)
    client.send(pack('!i', odor_index))

    log.debug(f'sent {odor_index}')
    if _debug:
        print(f'sent {odor_index}')

    # TODO replace w/ context manager for real deal
    client.close()


def _get_fig_df(event):
    # NOTE: we also have access to the Axes through event.inaxes, but it was easier
    # to get the figure elsewhere, where the main Axes (along with the 2nd Axes in
    # the figure, for the colorbar) were created inside plot_all_...
    fig = event.canvas.figure

    df = None
    if hasattr(fig, 'df'):
        df = fig.df
    else:
        # TODO log instead (though this shouldn't happen...)
        # TODO delete
        print('fig does not have DataFrame asssociated!')
        #

    return fig, df


def on_click(event):
    # TODO if i only associate df with fig (what i'm currently doing) and not the
    # specific axes (i.e. not the colorbar axes), check behavior in case we click on the
    # colorbar (+ probably need to find a way to exclude actions there)
    # TODO did i get `is` (rather than `==`) checking from an example? is that the right
    # way here? add comment if so.
    if event.button is MouseButton.LEFT:
        xdata = event.xdata

        # TODO when does this actually happen? need it?
        if xdata is None:
            return
        #

        _, df = _get_fig_df(event)
        if df is None:
            return

        # could check this against df shape, but eh...
        '''
        ax = event.inaxes
        xmin, xmax = ax.get_xlim()
        x_extent = xmax - xmin
        '''

        # event.xdata ~ [0, # odors - 1]
        sorted_odor_index = int(round(xdata))

        # df should be of length equal to number of total presentations, whereas
        # axes should be averaged across number of repeats. factorize() is just to
        # number the odors in sorted order (as they appear in plot)
        # https://stackoverflow.com/questions/47492685
        _, uniques = df.index.droplevel(['repeat', 'odor_index']).factorize()

        # TODO possible to factor this warning silencing out nicely? copied from
        # al_analysis.py
        with warnings.catch_warnings():
            # To ignore the "indexing past lexsort depth" warning.
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

            matching_rows = df.loc[
                tuple(uniques.to_frame(index=False).iloc[sorted_odor_index])
            ]

        unique_indices = matching_rows.index.get_level_values('odor_index').unique()
        assert len(unique_indices) == 1
        presentation_odor_index = unique_indices[0]

        send_odor_index_to_imagej_script(presentation_odor_index)


# TODO does this interfere w/ default 's'->save? (don't want to, but nbd if it does)
# NOTE: not working! no new plot is made (+ stuff freezes when i try to make one?)
def on_key_press(event):
    global most_recent_plot_proc
    global plotting_processes
    global _no_multiprocessing

    if event.key == 'r':
        fig, df = _get_fig_df(event)
        if df is None:
            return

        # TODO delete
        # (working)
        #log.debug('in on_key_press (event.key == "r")')

        # TODO get title of existing fig (or otherwise check for signs it was made w/
        # roi_min_max_scale=True, setting my own in plot if needed) ->
        # don't make new plot if already made w/ roi_min_max_scale=True

        log.debug('BEFORE making new roi_min_max_scale=True plot process')

        # TODO TODO TODO why is this freezing?
        proc = plot_process(plot, df, roi_min_max_scale=True,
            title='[0, 1] scaled per ROI'
        )

        log.debug('AFTER making new roi_min_max_scale=True plot process')

        if not _no_multiprocessing:
            most_recent_plot_proc = proc
            plotting_processes.append(proc)


def plot(df, sort_rois=True, **kwargs):
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
    # TODO TODO try replacing this w/ something similar to vline_level_fn in across-fly
    # ijroi part of al_analysis.py code (format_panel, w/ levels_from_labels=False)?
    vline_level_fn = lambda odor_str: olf.parse_odor_name(odor_str)

    # The cached ROIs will have '/' in ROI name (e.g. '3-30/1/DM4'), and only the last
    # row from the newly extracted data will not. We want a white line between these two
    # groups of data.
    hline_level_fn = lambda roi_str: ('/' in roi_str, roi_str.split('/')[-1])

    # TODO TODO normalize within each fly (if comparing)? or at least option to?
    # (+ maybe label colorbars differently in each case if so)

    # TODO TODO fix how there are no longer lines between panels always
    # (what change broke that?) (or sometimes there is a line between *each* odor,
    # regardless of whether they are in the same panel. don't want that either.)
    #
    # TODO make colorbar generally/always the height of the main Axes (maybe a bit
    # larger if just ~one row?)
    # TODO maybe just use odor_sort=True? could make passing odor_index to imagej script
    # harder / impossible tho...
    fig, _ = plot_all_roi_mean_responses(df, odor_sort=False,
        # TODO TODO make figs fixed width
        # (really it's the AXES that i want to be a fixed width... it may have been
        # other parts of the figure changing size of axes, rather than number of columns
        # changing anyway...)
        #
        # TODO (still?) refactor to share w/ roi_plot_kws in al_analysis (move into
        # plot_all...  defaults?), if same values work (they don't,
        # extra_figsize[1]==0.0 led to:
        # "constrained_layout not applied because axes sizes collapsed to zero. ...")
        # could try 0.8 for this value in al_analysis.py:roi_plot_kws tho
        #
        # TODO now that these options often produce a cbar w/ only a couple ticks,
        # change cbar_kws to make sure a reasonable number of ticks shown?
        inches_per_cell=0.08,
        # long odor names might get cut off w/ only 1.0. <=0.9 cuts off some of my
        # abbreviated odor names.
        extra_figsize=(2.0, 1.0),

        # TODO TODO use cmap w/ TwoSlopeNorm, perceptually linear diverging
        # colormap (once i get that figured out in al_analysis.py diverging_cmap_kwargs,
        # maybe can just use that?)

        roi_sort=sort_rois, sort_rois_first_on=sort_first_on,
        hline_level_fn=hline_level_fn, vline_level_fn=vline_level_fn,

        # Since we want vlines between odors, we can't easily (w/ current viz.matshow
        # behavior) also have vlines between panels, and thus the (group label, odor)
        # combinations will have duplicates... (because of the few odors both in
        # diagnostic and Remy panel, e.g. 'ms @ -3')
        # TODO try to have it warn about dupes or modify labels to include panel?
        allow_duplicate_labels=True,

        # More readable than default 100 for stuff with a lot of odors (e.g. validation2
        # + diagnostsics, which has 46 odors total)
        # TODO probably just change font params tho? or idk
        dpi=120,

        **kwargs
    )
    # TODO delete (for checking figs are now fixed width)
    # width of 6 should be fine.
    #print(f'figsize={tuple(fig.get_size_inches())}')
    #

    # so that on_click can access the associated data
    fig.df = df

    # TODO also say fig number? or fig id(...)?
    log.debug('connecting on_click via matplotlib button_press_event')

    # TODO log debug this?
    # NOTE: the on_click callback requires to the fig.df attribute set above
    plt.connect('button_press_event', on_click)

    # TODO can i also use fig.canvas.mpl_connect (rather than plt.connect) for above?
    # try!

    # TODO TODO TODO fix + restore!
    #log.debug('connecting on_key_press via matplotlib key_press_event')
    #fig.canvas.mpl_connect('key_press_event', on_key_press)

    log.debug('calling matplotlib plt.show()')

    # TODO delete
    # draw() was 0.4s per hit (that's probably the worst that constrained layout is
    # impacting each call). other 0.4s per call were in savefig (and plt.show() would
    # probably be similar or greater?)
    #fig.canvas.draw()
    #fig.set_layout_engine('none')
    #

    # plot_all_roi_mean_responses was most of (~15%) the rest of the time, w/ around
    # 0.1s per hit (not too much)
    #
    # TODO some way to exclude this from profiling (or at least idle time?)?
    # this was ~85% of time (1.2s per hit) (when i was manually closing each plot after
    # it popped up, but maybe that's not the best test)
    plt.show()

    log.debug('done showing plot')


_no_multiprocessing = False
# TODO could add a start=True kwarg if that one case that currently assigns proc before
# starting needs that order
def plot_process(plot_fn, *args, **kwargs):
    if not _no_multiprocessing:
        proc = Process(target=plot_fn, args=args, kwargs=kwargs)
        # TODO possible to also log when it dies? multiprocessing-logging work with more
        # than one level of multliprocess? prob not?
        # TODO possible to at least get PID of plotting process here if not?
        log.debug(f'starting Process for plot_fn={plot_fn.__name__}, with args:\n'
            f'{pformat(args)}\nkwargs:\n{pformat(kwargs)}'
        )
        proc.start()
        return proc
    else:
        plot_fn(*args, **kwargs)


newly_analyzed_dfs = []
newly_analyzed_roi_names = []
newly_analyzed_roi_strs = []

keep_comparison_to_cache = False
most_recent_plot_proc = None
plotting_processes = []

@profile
def load_and_plot(args):
    global newly_analyzed_dfs
    global newly_analyzed_roi_names
    global newly_analyzed_roi_strs
    global keep_comparison_to_cache
    global most_recent_plot_proc
    global plotting_processes
    global _no_multiprocessing

    roi_strs = args.roi_strs

    analysis_dir = args.analysis_dir
    roi_index = args.roi_index
    roiset_path = args.roiset_path

    compare = not args.no_compare
    hallem = args.hallem

    plot_pair_data = args.pairs
    plot_other_odors = args.other_odors

    add_to_plot = args.add

    _no_multiprocessing = args.debug

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
            # TODO TODO delete? or fix so this branch actually only worked when i wanted
            # to keep these. commenting because this is the only thing keep comparisons
            # still showing up (when --no-compare CLI arg is passed), and actually i
            # never want these comparisons at present (cache location is out of date
            # anyway, so not showing any of new data)
            #
            # TODO do i really want this True just in the else here? additional check?
            # (no, seems broken as-is)
            #keep_comparison_to_cache = True

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

            # TODO test
            # TODO move to option in orns.orns(...)?
            orn_df = orn_df[[g for g in orn_df.columns if g != 'DM3+DM5']].copy()

            assert not orn_df.columns.duplicated().any()

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

            plot_process(plot_corrs, hallem_corrs)

        newly_analyzed_roi_names.append(new_roi_name)
        newly_analyzed_dfs.append(new_df)

    if compare or analysis_dir is None or keep_comparison_to_cache:
        roi_strs.extend(newly_analyzed_roi_strs)
        assert len(roi_strs) > 0

        # TODO TODO TODO handle this not existing (never saving to this path anymore
        # anyway). delete all this code? update saving to a new place i can rely on?
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
            warn(f'no ROIs matching any of {roi_strs} in cached responses')
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

    # TODO this one also need error handling in case where verify_integrity would fail?
    subset_df = pd.concat(subset_dfs, axis='columns', verify_integrity=True)

    if not plot_pair_data:
        subset_df = dropna(
            subset_df.loc[subset_df.index.get_level_values('is_pair') == False, :]
        )

    subset_df = subset_df[subset_df.index.get_level_values('odor1') != 'pfo @ 0']

    # TODO care to have same order as manually specified in panel2name_order in
    # al_analysis.py? try to unify handling if so... maybe just import it from there?
    # or cache it, as w/ abbrevs?
    #
    # TODO refactor (move sorting into plot_all_roi_mean_responses, after mean? need to
    # drop too though...) so that order from current panel is used, but data from any
    # ROIs from older panels that shared some odors is still shown. as is, it will
    # either only show for the first panel or only show separately
    #
    # I don't think order would necessarily be preserved wrt data if sort=False either,
    # and would want to ensure that before providing an option to not sort.
    sort = True
    if sort:
        # TODO or determine order by sorting by how much old data we have for each odor
        # name?

        index = subset_df.index.to_frame().reset_index(drop=True)

        # we could relax this requirement, but then we'd really need to check that #
        # repeats is consistent (as checked in next assertion...)
        assert 'panel' in subset_df.index.names

        # https://stackoverflow.com/questions/47492685
        odor_index, _ = subset_df.index.droplevel('repeat').factorize()

        # should also currently only be triggered by 2023-10-15/1 data w/ duplicate
        # 2-mib
        # TODO maybe delete here if i'm going to keep warning in except branch of
        # pd.concat(subset_dfs, verify_integrity=True)? or provide more info here and
        # delete that other one?
        if len(set(pd.Series(odor_index).value_counts())) != 1:
            # TODO TODO at least say how the excess are being handled. and how are they
            # (all averaged together? everything but last n_repeats dropped?)?
            # TODO TODO if these ever actually cause a problem for either olf.sort_odors
            # or al.sort_odors, just check that those fns warn themselves in these cases
            # (at least by default)
            warn('variable number of repeats')

        index['odor_index'] = odor_index

        index = pd.MultiIndex.from_frame(index)
        # TODO TODO maybe it'd be easier for imagej code to work in more cases if i do
        # either presentation / frame indices here (rather than odors)?
        assert subset_df.index.identical(index.droplevel('odor_index'))
        subset_df.index = index

        # TODO delete. for debugging.
        #subset_df.to_pickle('subset_df.p')
        #

        #subset_df = olf.sort_odors(subset_df)

        # TODO option to switch between these? modify olf.sort_odors to allow keeping
        # certain panels in a fixed position + name_order, but to allow treating a
        # subset of panels as a single panel?
        # TODO TODO vline_level_fn using panel data (as in many of the across-fly ijroi
        # plots in al_analysis.py)? might need to pass in some precomputed list or
        # something, which i think might require changes to hong2p.viz.matshow handling
        subset_df = al.sort_odors(subset_df)


    # TODO TODO maybe if the name is shared w/ current roi name (prefix again, cause
    # '?'?), then call out that one, maybe by bolding name in yticklabels of both hallem
    # plots or something? or sorting that one out?
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

            plot(hallem_overlap_df, ax=resp_ax, sort_rois=False,
                title=f'Hallem data (sorted by corr with {new_roi_name})'
            )

        # TODO delete after debugging
        #hallem_comparison_plot()
        #

        #plot_process(hallem_comparison_plot)

        # TODO add option to not run this through Process? some reason i can't?
        # (to make use of debugger within possible)
        #
        # TODO TODO have this Axes be same physical size, so it's easier to compare
        # the two plots
        # TODO move up "title" (xlabel) on this one / tight_layout / something
        plot_process(plot, hallem_overlap_df, sort_rois=False,
            title=f'Hallem data (sorted by corr with {new_roi_name})',
        )

    if add_to_plot and most_recent_plot_proc is not None:
        most_recent_plot_proc.terminate()

    # currently just counting on the process eventually terminating, and thus the
    # corresponding Process object being cleaned up

    # TODO matter that this is before proc.start()? can also refactor this part if
    # not...
    # TODO delete if order did not seem to matter
    #proc = Process(target=plot, args=(subset_df,))
    #most_recent_plot_proc = proc
    #proc.start()

    proc = plot_process(plot, subset_df)

    if not _no_multiprocessing:
        most_recent_plot_proc = proc
        plotting_processes.append(proc)

    # TODO maybe as a hack to get easy profiling to work, just sys.exit here if we are
    # profiling? (working OK to manually call `pkill -SIGINT -f '/plot_roi'` after)
    #
    # (not sure how i could do this and also profile w/o initial load time, from a prior
    # call, tho...)


# TODO delete
'''
if __name__ == '__main__':
    df = pd.read_pickle('subset_df.p')
    # TODO also check it works w/ al.sort_odors
    df = olf.sort_odors(df)
    plot(df)
    plt.show()
'''
#
