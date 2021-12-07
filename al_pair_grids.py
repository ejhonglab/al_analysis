#!/usr/bin/env python3

import argparse
import atexit
import os
from os.path import join, split, exists, expanduser
from pprint import pprint, pformat
from collections import defaultdict
import warnings
import time
import shutil
import traceback
import subprocess
import pickle
from pathlib import Path
import glob
from itertools import starmap
import multiprocessing as mp
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import tifffile
import yaml
from suite2p import run_s2p
import ijroi
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
# TODO find a nicer perceptually linear colormap from here?
#import colorcet as cc
from drosolf import orns

from hong2p import util, thor, viz, olf
from hong2p import suite2p as s2p
from hong2p.suite2p import LabelsNotModifiedError, LabelsNotSelectiveError
from hong2p.util import shorten_path, format_date
from hong2p.olf import format_odor, format_mix_from_strs, format_odor_list
from hong2p.viz import dff_latex


plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.constrained_layout.w_pad'] = 1/72
plt.rcParams['figure.constrained_layout.h_pad'] = 0.5/72
plt.rcParams['figure.constrained_layout.wspace'] = 0
plt.rcParams['figure.constrained_layout.hspace'] = 0

###################################################################################
# Constants that affect behavior of `process_experiment`
###################################################################################
analysis_intermediates_root = util.analysis_intermediates_root()

# TODO try to make this more granular, so the suite2p / non-suite2p analysis can be
# skipped separately (latter should be generated on first run, but former will at least
# require some manual steps [and the initial automated suite2p run may also fail])

# TODO also make more granular in the sense that we don't necessarily want to run
# suite2p on glomerli diagnostics stuff, etc
# TODO probably make trial (+ avg across trial) dff image plots in their own directory,
# so it's easier to just check if that directory exists (or maybe check if any svg are
# in exp dir root? maybe assume plots are made if we have a suite2p directory?)
skip_if_experiment_plot_dir_exists = False

# TODO TODO probably make another category or two for data marked as failed (in the
# breakdown of data by pairs * concs at the end)
retry_previously_failed = False

# Whether to only analyze experiments sampling 2 odors at all pairwise concentrations
# (the main type of experiment for this project)
analyze_pairgrids_only = False

# If there are multiple experiments with the same odors, only the data from the most
# recent concentrations for those odors will be analyzed.
final_concentrations_only = True

analyze_reverse_order = True

# Will be set False if analyze_pairgrids_only=True
analyze_glomeruli_diagnostics = True

# Whether to analyze any single plane data that is found under the enumerated
# directories.
analyze_2d_tests = False

non_suite2p_analysis = True
# TODO maybe put this one behind a CLI flag?
ignore_existing_nonsuite2p_outputs = True

# Whether to run the suite2p pipeline (generates outputs among raw data, in 'suite2p'
# subdirectories)
do_suite2p = False

# Will just skip if already exists if False
overwrite_suite2p = False

analyze_suite2p_outputs = False

analyze_ijrois = True

# Since some of the pilot experiments had 6 planes (but top 5 should be the same as in
# experiments w/ only 5 total), and that last plane largely doesn't have anything
# measurably happening. All steps should have been 12um, so picking top n will yield a
# consistent total height of the volume.
n_top_z_to_analyze = 5

ignore_bounding_frame_cache = False

want_dff_tiff = True

links_to_input_dirs = True

# TODO shorten any remaining absolute paths if this is True, so we can diff outputs
# across installs w/ data in diff paths
print_full_paths = False

save_figs = True
plot_fmt = os.environ.get('plot_fmt', 'svg')

cmap = 'plasma'
diverging_cmap = 'RdBu_r'
# TODO could try TwoSlopeNorm, but would probably want to define bounds per fly (or else
# compute in another pass / plot these after aggregating?)
diverging_cmap_kwargs = dict(cmap=diverging_cmap, norm=colors.CenteredNorm())

dff_cbar_title = f'{dff_latex}'

# TODO better name
trial_stat_cbar_title = f'Mean peak {dff_latex}'

diff_cbar_title = f'$\Delta$ mean peak {dff_latex}'

single_dff_image_row_figsize = (6.4, 2.0)

dff_vmin = 0
dff_vmax = 3.0

ax_fontsize = 7

bad_suite2p_analysis_dirs = (
    # skipping for now cause suite2p output looks weird (for both recordings)
    '2021-03-07/2',

    # Just a few glomeruli visible in (at least the last odor) trials.
    # Also imaged too much of other side, so just under half of potentially-good ROIs
    # from other side.
    '2021-05-24/1/1oct3ol_and_2h',

    # Either suite2p ROIs bad or experiment bad too.
    '2021-05-24/2/ea_and_etb',

    # eb_and_ea recording looks bad (mainly just one glomerulus? lots of stuff
    # seems to come in 4s at times rather than 3s [well really just one group of
    # 4], and generally not much signal)
    '2021-05-25/1/eb_and_ea',

    # Was having too much of a hard time manually correcting this one. Was trying to
    # merge, but shelving it for now.
    '2021-05-18/1/1o3ol_and_2h',

    # Too much important stuff seems overmerged right out of suite2p. May want to go
    # down for these odors anyway. Still some nice signal in some places.
    '2021-05-18/1/eb_and_ea',

    # Mainly a suite2p issue hopefully. ROIs not all of very good shapes + hard to find
    # good merges across planes, but perhaps mainly b/c suite2p garbage out?
    '2021-05-24/2/1o3ol_and_2h',

    # Small number of glomeruli w/ similar responses. Small # of glom also apparent in
    # dF/F images.
    '2021-05-24/2/1o3ol_and_2h',
)


# TODO set bool_fillna_false=False (kwarg to gsheet_to_frame) and manually fix any
# unintentional NaN in these columns if I need to use the missing data for early
# diagnostic panels (w/o some of the odors only in newest set) for anything

# This file is intentionally not tracked in git, so you will need to create it and
# paste in the link to this Google Sheet as the sole contents of that file. The
# sheet is located on our drive at:
# 'Hong Lab documents/Tom - odor mixture experiments/pair_grid_data'
gdf = util.gsheet_to_frame('pair_grid_data_gsheet_link.txt', normalize_col_names=True)
gdf.set_index(['date', 'fly'], verify_integrity=True, inplace=True)

# This is the name as converted by what `normalize_col_names=True` triggers.
last_gsheet_col_before_glomeruli_diag_statuses = 'notes'
last_gsheet_col_glomeruli_diag_statuses = 'all_labelled'

first_glomeruli_diag_col_idx = list(gdf.columns
    ).index(last_gsheet_col_before_glomeruli_diag_statuses) + 1

last_glomeruli_diag_col_idx = list(gdf.columns
    ).index(last_gsheet_col_glomeruli_diag_statuses)

glomeruli_diag_status_df = gdf.iloc[
    :, first_glomeruli_diag_col_idx:(last_glomeruli_diag_col_idx + 1)
]
# Column names should be lowercased names of target glomeruli after this.
glomeruli_diag_status_df.rename(columns=lambda x: x.split('_')[0], inplace=True)

# Since I called butanone's target glomerulus VM7 in my olfactometer config, but updated
# the name of the column in the gsheet to VM7d, as that is the specific part it should
# activate.
# TODO TODO TODO check all / all_bad still there after i renamed / moved some stuff in
# sheet
glomeruli_diag_status_df.rename(columns={'vm7d': 'vm7', 'all': 'all_bad'}, inplace=True)

# For cases where there were multiple glomeruli diagnostic experiments (e.g. both sides
# imaged, and only one used for subsequent experiments w/in fly). Paths (relative to
# data root) to the recordings not followed up on / representative should go here.
unused_glomeruli_diagnostics = (
    # glomeruli_diagnostics_otherside is the one used here
    '2021-05-25/2/glomeruli_diagnostics',
)

# For aggregating good[/bad] examples of the activation of each of these glomeruli by
# the odors targetting them.
across_fly_glomeruli_diags_dir = join(plot_fmt, 'glomeruli_diagnostics')


odor2abbrev = {
    'methyl salicylate': 'MS',
    'hexyl hexanoate': 'HH',
    'furfural': 'FUR',
    '1-hexanol': 'HEX',
    '1-octen-3-ol': 'OCT',
    '2-heptanone': '2H',
    'acetone': 'ACE',
    'butanal': 'BUT',
    'ethyl acetate': 'EA',
    'ethyl butyrate': 'EB',
    'ethyl hexanoate': 'EH',
    'hexyl acetate': 'HA',
}

if analyze_pairgrids_only:
    analyze_glomeruli_diagnostics = False

###################################################################################
# Modified inside `process_experiment`
###################################################################################

# TODO maybe convert to dict -> None (+ conver to set after
# process_experiment loop) (since mp.Manager doesn't seem to have a set)
#odors_without_abbrev = set()
odors_without_abbrev = []

# TODO refactor so this is not necessary if possible
# This will get populated w/ full paths matching path fragments in variable above,
# to filter out these paths in printing out status of suite2p analyses at the end.
full_bad_suite2p_analysis_dirs = []

exp_processing_time_data = []

failed_assigning_frames_to_odors = []

s2p_roi_dfs = []
# TODO TODO just agg (+ cache) this in process_experiment, and calculate all others from
# this after the main loop over experiments
# TODO also agg + cache full traces (maybe just use CSV? also for above)
# TODO rename to ij_trialstat_dfs or something
ij_roi_dfs = []

ij_mean_dfs = []
ij_sum_diff_dfs = []
ij_max_diff_dfs = []

# Using dict rather than defaultdict(list) so handling is more consistent in case when
# multiprocessing DictProxy overrides this.
names_and_concs2analysis_dirs = dict()

###################################################################################
# Modified inside `run_suite2p`
###################################################################################
failed_suite2p_dirs = []

###################################################################################
# Modified inside `suite2p_trace_plots`
###################################################################################
s2p_not_run = []
iscell_not_modified = []
iscell_not_selective = []
no_merges = []

###################################################################################


# TODO replace similar fn (if still exists?) already in hong2p? or use the hong2p one?
# (just want to prefer the "fast" data root)
def get_analysis_dir(date, fly_num, thorimage_basedir):
    return join(analysis_intermediates_root,
        util.get_fly_dir(date, fly_num), thorimage_basedir
    )


# Especially running process_experiment in parallel, the many-figures-open memory
# warning will get tripped at the default setting, hence `close=True`.
def savefig(fig, experiment_fig_dir, desc, close=True, **kwargs):
    # If True, the directory name containing (date, fly, thorimage_dir) information will
    # also be in the prefix for each of the plots saved within that directory (harder to
    # lose track in image viewers / after copying, but more verbose).
    prefix_plot_fnames = False
    basename = util.to_filename(desc) + plot_fmt

    if prefix_plot_fnames:
        experiment_basedir = split(experiment_fig_dir)[0]
        fname_prefix = experiment_basedir + '_'
        basename = fname_prefix + basename

    fig_path = join(experiment_fig_dir, basename)
    if save_figs:
        fig.savefig(fig_path, **kwargs)

    if close:
        plt.close(fig)

    return fig_path


dirs_to_delete_if_empty = []
def makedirs(d):
    """Make directory if it does not exist, and register for deletion if empty.
    """
    os.makedirs(d, exist_ok=True)
    dirs_to_delete_if_empty.append(d)


def delete_if_empty(d):
    """Delete directory if empty, do nothing otherwise.
    """
    if not any(os.scandir(d)):
        os.rmdir(d)


def delete_empty_dirs():
    """Deletes empty directories in `dirs_to_delete_if_empty`
    """
    for d in dirs_to_delete_if_empty:
        delete_if_empty(d)


def symlink(target, link):
    """Create symlink link pointing to target, doing nothing if link exists.
    """
    try:
        os.symlink(os.path.abspath(target), link)
    except FileExistsError:
        return


FAIL_INDICATOR_PREFIX = 'FAILING_'

# TODO maybe take error object / traceback and save stack trace actually?
def make_fail_indicator_file(analysis_dir, suffix, err=None):
    """Makes empty file (e.g. FAILING_suite2p) in analysis_dir, to mark step as failing
    """
    path = Path(join(analysis_dir, f'{FAIL_INDICATOR_PREFIX}{suffix}'))
    # TODO TODO i thought i was passing this everywhere? why are some of these files
    # empty (see 2021-07-21/2/FAILING_assign_frames)
    # TODO TODO TODO and regarding same file above, why is it under ./2, not 2/<exp
    # subdir>??? something left over from an old bug? current one?
    if err is None:
        path.touch()
    else:
        err_str = ''.join(traceback.format_exception(type(err), err, err.__traceback__))
        path.write_text(err_str)


def _list_fail_indicators(analysis_dir):
    return glob.glob(join(analysis_dir, FAIL_INDICATOR_PREFIX + '*'))


def last_fail_suffixes(analysis_dir):
    """Returns if an analysis_dir has any fail indicators and str with their suffixes
    """
    suffixes = [
        split(x)[1][len(FAIL_INDICATOR_PREFIX):]
        for x in _list_fail_indicators(analysis_dir)
    ]
    if len(suffixes) == 0:
        return False, None
    else:
        return True, ' AND '.join(suffixes)


def clear_fail_indicators(analysis_dir):
    """Deletes any fail indicator files in analysis_dir
    """
    for f in _list_fail_indicators(analysis_dir):
        os.remove(f)


# TODO did i already implement this logic somewhere in this file? use this code if so
def n_odors_per_trial(odor_lists):
    """
    Assumes same number of odors on each trial
    """
    len_set = {len(x) for x in odor_lists}
    assert len(len_set) == 1
    return len_set.pop()


def odor_lists2names_and_conc_ranges(odor_lists):
    """
    Gets a hashable representation of the odors and each of their concentration ranges.
    Ex: ( ('butanal', (-5, -4, -3)), ('acetone', (-5, -4, -3)) )

    Tuples of concentrations measured will be sorted in ascending order.

    What is returned doesn't have any information about order of presentation, and
    experiments are only equivalent if that didn't change, nor other decisions about the
    trial structure (typically these things have stayed pretty constant though)
    """
    def name_i(i):
        name_set = {x[i]['name'] for x in odor_lists}
        assert len(name_set) == 1, ('assuming odor_lists w/ same odor (names, not concs'
            f') at each trial ({name_set=})'
        )
        return name_set.pop()

    def conc_range_i(i):
        concs_i_including_solvent = {x[i]['log10_conc'] for x in odor_lists}
        concs_i = sorted([x for x in concs_i_including_solvent if x is not None])
        return tuple(concs_i)

    n = n_odors_per_trial(odor_lists)
    names_and_conc_ranges = tuple((name_i(i), conc_range_i(i)) for i in range(n))
    return names_and_conc_ranges


def is_pairgrid(odor_lists):
    # TODO TODO reimplement in a way that actually checks there are all pairwise
    # concentrations, rather than just assuming so if there are 2 odors w/ 3 non-zero
    # concs each
    try:
        names_and_conc_ranges = odor_lists2names_and_conc_ranges(odor_lists)
    # TODO delete hack
    except AssertionError:
        return False

    return len(names_and_conc_ranges) == 2 and (
        all([len(cs) == 3 for _, cs in names_and_conc_ranges])
    )


def is_reverse_order(odor_lists):
    o1_list = [o1 for o1, _ in odor_lists if o1['log10_conc'] is not None]

    def get_conc(o):
        return o['log10_conc']

    return get_conc(o1_list[0]) > get_conc(o1_list[-1])


def odor_strs2single_odor_name(index, name_conc_delim='@'):
    odors = {x.split(name_conc_delim)[0].strip() for x in index}
    odors = {x for x in odors if x != 'solvent'}
    assert len(odors) == 1
    return odors.pop()


def odor_lists_to_multiindex(odor_lists, **format_odor_kwargs):

    unique_lens = {len(x) for x in odor_lists}
    if len(unique_lens) != 1:
        raise NotImplementedError

    # This one would be more straight forward to relax than the above one
    if unique_lens != {2}:
        raise NotImplementedError

    odor1_str_list = []
    odor2_str_list = []

    odor_mix_counts = defaultdict(int)
    odor_mix_repeats = []

    for odor_list in odor_lists:
        odor1, odor2 = odor_list

        # TODO refactor
        if odor1['name'] in odor2abbrev:
            odor1['name'] = odor2abbrev[odor1['name']]

        if odor2['name'] in odor2abbrev:
            odor2['name'] = odor2abbrev[odor2['name']]

        odor1_str = format_odor(odor1, **format_odor_kwargs)
        odor1_str_list.append(odor1_str)

        odor2_str = format_odor(odor2, **format_odor_kwargs)
        odor2_str_list.append(odor2_str)
        #

        odor_mix = (odor1_str, odor2_str)
        odor_mix_repeats.append(odor_mix_counts[odor_mix])
        odor_mix_counts[odor_mix] += 1

    # NOTE: relying on sorting odor_list(s) at load time seems to produce consistent
    # ordering, though that alphabetical ordering (based on full odor names) is
    # different from what would be produced sorting on abbreviated odor names (at least
    # in some cases)

    index = pd.MultiIndex.from_arrays([odor1_str_list, odor2_str_list,
        odor_mix_repeats
    ])
    index.names = ['odor1', 'odor2', 'repeat']

    return index


def separate_names_and_concs_tuples(names_and_concs_tuple):
    """
    Takes input like ( ('butanal', (-5, -4, -3)), ('acetone', (-5, -4, -3)) ) to
    output like ('acetone', 'butanal'), ((-5, -4, -3), (-5, -4, -3))
    """
    names = tuple(n for n, _ in names_and_concs_tuple)
    concs = tuple(cs for _, cs in names_and_concs_tuple)
    return names, concs


def odor_names2final_concs(**paired_thor_dirs_kwargs):
    """Returns dict of odor names tuple -> concentrations tuples + ...

    Loops over same directories as main analysis
    """
    keys_and_paired_dirs = util.paired_thor_dirs(verbose=False,
        **paired_thor_dirs_kwargs
    )

    seen_stimulus_yamls2thorimage_dirs = defaultdict(list)
    names2final_concs = dict()
    names_and_concs_tuples = []
    for (_, _), (thorimage_dir, _) in keys_and_paired_dirs:

        xml = thor.get_thorimage_xmlroot(thorimage_dir)
        ti_time = thor.get_thorimage_time_xml(xml)

        yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(xml)

        seen_stimulus_yamls2thorimage_dirs[yaml_path].append(thorimage_dir)

        try:
            names_and_concs_tuple = odor_lists2names_and_conc_ranges(odor_lists)

        # TODO change error from assertionerror if gonna catch it...
        except AssertionError:
            continue

        if not is_pairgrid(odor_lists):
            continue

        names_and_concs_tuples.append(names_and_concs_tuple)

        names, curr_concs = separate_names_and_concs_tuples(names_and_concs_tuple)
        names2final_concs[names] = curr_concs

    return names2final_concs, seen_stimulus_yamls2thorimage_dirs, names_and_concs_tuples


# TODO write tests for this function for the case when n_trial_length_args is empty and
# also has at least one appropriate element
# NOTE: it is important that keyword argument are after *n_trial_length_args, so they
# are interpreted as keyword ONLY arguments.
# TODO TODO add kwargs to get only the n frames / <=x seconds after each odor onset,
# and explore effect on some of things computed with this
def split_into_trials(movie_length_array, bounding_frames, *n_trial_length_args,
    after_onset_only=True, squeeze=True, subtract_and_divide_baseline=True,
    exclude_last_baseline_frame=True):

    # TODO TODO if i actually might want to default to exclude_last_baseline_frame=
    # True, b/c of problems w/ my frame assignment / whatever, should i not also
    # consider that last frame for computing stats for a given trial (treating it as if
    # it were first_odor_frame)?

    # TODO probably instead raise a ValueError here w/ message indicating
    # movie_length_array probably needs to be transposed
    assert all([all([i < len(movie_length_array) for i in trial_bounds])
        for trial_bounds in bounding_frames
    ])

    # TODO this should also probably be a ValueError
    assert all([len(x) == len(bounding_frames) for x in n_trial_length_args])

    for (start_frame, odor_onset_frame, end_frame), *trial_args in zip(bounding_frames,
        *n_trial_length_args):

        # Not doing + 1 to odor_onset_frame since we only want to include up to the
        # previous frame in the baseline. This slicing seems to work the same for both
        # numpy arrays and pandas DataFrames of the same shape (at least for stuff where
        # the rows are the default RangeIndex, which is all I tested)
        before_onset = movie_length_array[start_frame:odor_onset_frame]
        after_onset = movie_length_array[odor_onset_frame:(end_frame + 1)]

        if subtract_and_divide_baseline:
            if not exclude_last_baseline_frame:
                baseline = movie_length_array[start_frame:odor_onset_frame]
            else:
                baseline = movie_length_array[start_frame:(odor_onset_frame - 1)]

            # TODO maybe also accept a function to compute baseline / accept appropriate
            # dimensional input [same as mean would be] to subtract directly?
            # TODO test this baselining approach works w/ other dimensional inputs too

            mean_baseline = baseline.mean(axis=0)

            after_onset = (after_onset - mean_baseline) / mean_baseline

        if after_onset_only:
            if squeeze and len(trial_args) == 0:
                yield after_onset
            else:
                yield (after_onset,) + tuple(trial_args)

        # TODO maybe delete this branch? not sure when i'd actually want to use it...
        else:
            if subtract_and_divide_baseline:
                before_onset = (before_onset - mean_baseline) / mean_baseline

            # NOTE: no need to squeeze here because always at least length 2
            yield (before_onset, after_onset) + tuple(trial_args)


# TODO also use this to compute dF/F images in loop below?
# TODO TODO try to make a test case for the above
# TODO maybe default to mean in a fixed window as below? or not matter as much since i'm
# actually using this on data already meaned within an ROI (though i could use on other
# data later)?
def compute_trial_stats(traces, bounding_frames, odor_order_with_repeats=None,
    stat=lambda x: np.max(x, axis=0)):
    # TODO unify documentation of this list-of-list-of-dicts format, including
    # expectations on the dicts (perhaps also use dataclasses/similar in place of the
    # dicts and include type hints)
    """
    Args:
        odor_order_with_repeats: if passed, will be passed to odor_lists_to_multiindex.
            Should be a list of lists, with each internal list containing dicts that
            each represent a single odor. Each internal list represents all odors
            presented together on one trial.
    """

    if odor_order_with_repeats is not None:
        assert len(bounding_frames) == len(odor_order_with_repeats)

    # TODO return as pandas series if odor_order_with_repeats is passed, with odor
    # index containing that data? test this would also be somewhat natural in 2d/3d case

    trial_stats = []

    for trial_traces in split_into_trials(traces, bounding_frames):
        curr_trial_stats = stat(trial_traces)

        # TODO TODO adapt to also work in case input is a movie
        # TODO TODO also work in 1d input case (i.e. if just data from single ROI was
        # passed)
        # traces.shape[1] == # of ROIs
        assert curr_trial_stats.shape == (traces.shape[1],)

        trial_stats.append(curr_trial_stats)

    trial_stats = np.stack(trial_stats)

    if odor_order_with_repeats is None:
        index = None
    else:
        index = odor_lists_to_multiindex(odor_order_with_repeats)

    trial_stats_df = pd.DataFrame(index=index, data=trial_stats)
    trial_stats_df.index.name = 'trial'

    # TODO maybe implement somthing that also works w/ xarrays? maybe make my own
    # function that dispatches to the appropriate concatenation function accordingly?

    # Since np.stack (probably as well as other similar numpy functions) converts pandas
    # stuff to numpy, and pd.concat doesn't work with numpy arrays.
    if hasattr(traces, 'columns'):
        trial_stats_df.columns = traces.columns
    else:
        # TODO maybe only do this if a certain dimension if 2d input (time x ROIs) is
        # passed in (but is it possible to name these correctly based on dimension in
        # general, or even for any particular dimension? if not, don't name ever)
        trial_stats_df.columns.name = 'roi'

    return trial_stats_df


# TODO share (parts?) w/ similar key fn now in hong2p.olf?
def odor_index_sort_key(level):
    # The assignment below failed for some int dtype levels, even though the boolean
    # mask dictating where assignment should happen must have been all False...
    if level.dtype != np.dtype('O'):
        return level

    sort_key = level.values.copy()

    solvent_elements = sort_key == 'solvent'
    conc_delimiter = '@'
    assert all([conc_delimiter in x for x in sort_key[~ solvent_elements]])

    # TODO share this + fn wrapping this that returns sort key w/ other place that
    # is using sorted(..., key=...)
    def parse_log10_conc(odor_str):
        assert conc_delimiter in odor_str
        parts = odor_str.split(conc_delimiter)
        assert len(parts) == 2
        return float(parts[1].strip())

    conc_keys = [parse_log10_conc(x) for x in sort_key[~ solvent_elements]]
    sort_key[~ solvent_elements] = conc_keys

    # Setting solvent to an unused log10 concentration just under lowest we have,
    # such that it gets sorted into overall order as I want (first, followed by
    # lowest concentration).
    # TODO TODO maybe just use float('-inf') or numpy equivalent here?
    # should give me the sorting i want.
    sort_key[solvent_elements] = min(conc_keys) - 1

    # Converting back to an index so that `level=<previous level name>` arg to
    # `DataFrame.sort_index` doesn't get broken. This key function is used to generate
    # an intermediate Index pandas uses to sort, and that intermediate needs to have the
    # same level names to be able to refer to them as if it was the input object.
    return pd.Index(sort_key, name=level.name)


odor_cols = ['odor1', 'odor2']
def sort_odor_indices(df):

    def levels_to_sort(index):
        return [k for k in index.names if k in odor_cols]

    levels = levels_to_sort(df.index)
    for axis_name in ('index', 'columns'):
        levels = levels_to_sort(getattr(df, axis_name))
        if len(levels) > 0:
            # TODO check my level sort key fn works in both case of 1 level passed in as
            # well as 2
            df = df.sort_index(key=odor_index_sort_key, axis=axis_name, level=levels,
                sort_remaining=False
            )

    return df


def dff_imshow(ax, dff_img, **imshow_kwargs):

    vmin = imshow_kwargs.pop('vmin', dff_vmin)
    vmax = imshow_kwargs.pop('vmax', dff_vmax)

    im = ax.imshow(dff_img, vmin=vmin, vmax=vmax, **imshow_kwargs)

    # TODO TODO figure out how do what this does EXCEPT i want to leave the
    # xlabel / ylabel (just each single str)
    ax.set_axis_off()

    # part but not all of what i want above
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])

    return im


def plot_roi_stats_odorpair_grid(single_roi_series, show_repeats=False, ax=None,
    **kwargs):

    assert single_roi_series.index.names == ['odor1', 'odor2', 'repeat']

    roi_df = single_roi_series.unstack(level=0)

    roi_df = sort_odor_indices(roi_df)

    title = f'ROI {single_roi_series.name}'

    # TODO now that order of odors within each trial should be sorted at odor_list(s)
    # creation, do we still need this transpose_sort_key? if not, can probably delete
    # odor_strs2single_odor_name
    common_matshow_kwargs = dict(
        ax=ax, title=title, transpose_sort_key=odor_strs2single_odor_name, cmap=cmap,
        # not working
        #vmin=-0.2, vmax=2.0,
        **kwargs
    )

    if show_repeats:
        # TODO should i just let this make the axes and handle the colorbar? is the
        # colorbar placement any better / worse if done that way? i guess i might
        # occasionally want to have this plot in half of a figure (one Axes in an array
        # of two, w/ the other being a view of the ROI footprint or something)?
        fig, _ = viz.matshow(roi_df.droplevel('repeat'), group_ticklabels=True,
            **common_matshow_kwargs
        )
    else:
        # 'odor2' is the one on the row axis, as one level alongside 'repeat'
        # TODO not sure why sort=False seems to be ignored... bug?
        mean_df = sort_odor_indices(roi_df.groupby('odor2', sort=False).mean())

        fig, _ = viz.matshow(mean_df, **common_matshow_kwargs)

    return fig


def plot_all_roi_mean_responses(trial_stats, title=None, roi_sortkeys=None, **kwargs):

    assert trial_stats.index.names == ['odor1', 'odor2', 'repeat']

    mean_df = trial_stats.groupby(['odor1', 'odor2'], sort=False).mean()
    mean_df = sort_odor_indices(mean_df)

    if roi_sortkeys is not None:
        assert len(roi_sortkeys) == len(trial_stats.columns)

        roi_sortkey_dict = dict(zip(trial_stats.columns, roi_sortkeys))

        def roi_sortkey_fn(index):
            return [roi_sortkey_dict[x] for x in index]

        mean_df.sort_index(key=roi_sortkey_fn, axis='columns', inplace=True)

    # TODO check whether passing `xticklabels=str` works, and fix if not
    fig, _ = viz.matshow(mean_df, title=title, xticklabels=str,
        yticklabels=format_mix_from_strs, cmap=cmap, **kwargs
    )

    return fig, mean_df


# TODO kwarg to allow passing trial stat fn in that includes frame rate info as closure,
# for picking frames in a certain time window after onset and computing mean?
# TODO TODO TODO to the extent that i need to convert suite2p rois to my own and do my
# own trace extraction, maybe just modify my fork of suite2p to save sufficient
# information in combined view stat.npy to invert the tiling? unless i really can find a
# reliable way to invert that step...
def suite2p_trace_plots(analysis_dir, bounding_frames, odor_lists, plot_dir=None):

    try:
        traces, roi_stats, ops, merges = s2p.load_s2p_combined_outputs(analysis_dir,
            merge_inputs=True, err=True
        )

    except (IOError, LabelsNotModifiedError, LabelsNotSelectiveError) as e:

        if isinstance(e, IOError):
            s2p_not_run.append(analysis_dir)

        elif isinstance(e, LabelsNotModifiedError):
            iscell_not_modified.append(analysis_dir)

        elif isinstance(e, LabelsNotSelectiveError):
            iscell_not_selective.append(analysis_dir)

        print(f'NOT generating suite2p plots because {e}')
        return

    if len(merges) == 0:
        no_merges.append(analysis_dir)

    # TODO TODO TODO also try passing input of (another call to?) compute_trial_stats to
    # remerge_suite2p_merged, so it's actually using local dF/F rather than max-min of
    # the entire raw trace

    verbose = True
    # TODO TODO TODO modify so that that merging w/in plane works (before finding best
    # plane) + test
    traces, rois = s2p.remerge_suite2p_merged(traces, roi_stats, ops, merges,
        verbose=False
    )

    trial_stats = compute_trial_stats(traces, bounding_frames, odor_lists)

    # TODO TODO refactor so that this is outside of the fn named indicating it is just
    # making plots from suite2p outputs. could do simultaneous w/ refactoring to unify
    # suite2p / manual-imagej-roi trace handling
    print('writing ROI traces and trial statistics to CSVs')

    traces.to_csv(join(analysis_dir, 'traces.csv'))
    # W/ default CSV output format, all columns after 'repeat' (last level of row
    # MultiIndex) are suite2p ROI numbers (not necessarily consecutive b/c of preceding
    # selection) which should be the same numbers in the columns of traces.csv.
    trial_stats.to_csv(join(analysis_dir, 'trial_stats.csv'))

    # TODO check numbering is consistent w/ suite2p numbering in case where there is
    # some merging (currently handling will treat it incorrectly)
    # TODO also plot roi / outline of roi on corresponding [mean?] plane / maybe with
    # other planes for context?
    if save_figs and plot_dir is not None:
        roi_dir = join(plot_dir, 'roi')
        makedirs(roi_dir)

    z_indices = [roi_stats[x]['iplane'] for x in rois.roi.values]

    title = 'Suite2p ROIs\nOrdered by Z plane'

    # TODO TODO maybe also order by strength of response / clustering of response/mixing
    # types?
    # TODO TODO probably put lines between levels of sortkey if int (e.g. 'iplane')
    # (and also show on plot as second label above/below roi labels?)
    fig, mean_df = plot_all_roi_mean_responses(trial_stats, title=title,
        roi_sortkeys=z_indices, cbar_label=trial_stat_cbar_title, cbar_shrink=0.4
    )

    if plot_dir is not None:
        savefig(fig, roi_dir, 'all_rois_by_z')

    #include_rois = True
    include_rois = False

    for roi in trial_stats.columns:
        if include_rois:
            fig, axs = plt.subplots(nrows=2, ncols=1)
            ax = axs[0]
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # TODO more globally appropriate title? (w/ fly and other metadata. maybe number
        # flies within each odor pair / sequentially across days, and just use that as a
        # short ID?)

        #fig.suptitle(f'ROI {roi}')

        roi1_series = trial_stats.loc[:, roi]
        plot_roi_stats_odorpair_grid(roi1_series, ax=ax,
            cbar_label=trial_stat_cbar_title, cbar_shrink=0.4
        )

        # TODO TODO (assuming colors are saved / randomization is seeded and easily
        # reproducible in suite2p) copy suite2p coloring for plotting ROIs (or at least
        # make my own color scheme consistent across all plots w/in experiment)
        # TODO TODO separate unified ROI plot (like combined view, but maybe all in
        # one row for consistency with my other plots) with same color scheme
        # TODO TODO have scale in single ROI plots and ~"combined" view be the same
        # (each plot pixel same physical size)
        if include_rois:
            roi_stat = roi_stats[roi]
            s2p.plot_roi(roi_stat, ops, ax=axs[1])

        if plot_dir is not None:
            savefig(fig, roi_dir, str(roi))

    # TODO TODO [option to] use non-weighted footprints (as well as footprints that have
    # had the binary closing operation applied before uniform weighting)

    # TODO [option to] exclude stuff that doesn't have any 0's in iscell / warn [/ open
    # gui for labelling?]

    print('generated plots based on suite2p traces')

    # TODO maybe refactor
    date_str, fly_str, thorimage_id = analysis_dir.split(os.path.sep)[-3:]

    roi_df = trial_stats
    new_level_names = ['date', 'fly_num', 'thorimage_id']
    new_level_values = [pd.Timestamp(date_str), int(fly_str), thorimage_id]

    roi_df = util.addlevel(roi_df, new_level_names, new_level_values)

    s2p_roi_dfs.append(roi_df)

    return True


# TODO TODO TODO refactor this + above suite2p version to share most of the code
def ij_trace_plots(analysis_dir, bounding_frames, odor_lists, movie, plot_dir=None):

    # TODO clean this up
    thorimage_dir = analysis_dir.replace('analysis_intermediates', 'raw_data')
    #

    try:
        masks = util.ijroi_masks(analysis_dir, thorimage_dir)

    except IOError:
        # TODO probably print something here / in try above

        # This experiment didn't have any ImageJ ROIs (in canonical place)
        return

    # TODO TODO check this is working correctly with this new type of input +
    # change fn to preserve input type (w/ the metadata xarrays / dfs have)
    traces = pd.DataFrame(util.extract_traces_bool_masks(movie, masks))
    traces.index.name = 'frame'
    traces.columns.name = 'roi'

    # TODO also try merging via correlation/overlap thresholds?

    # TODO TODO TODO update this to use df/f (same update needed on suite2p branch)
    roi_quality = traces.max() - traces.min()

    roi_nums, rois = util.rois2best_planes_only(masks, roi_quality)

    traces = traces[roi_nums].copy()

    # This is currently the 'name' (rois2best_planes_only drops other levels from ROI
    # index)
    traces.columns = rois.roi.values

    trial_stats = compute_trial_stats(traces, bounding_frames, odor_lists)

    # TODO TODO refactor so that this is outside of the fn named indicating it is just
    # making plots from suite2p outputs. could do simultaneous w/ refactoring to unify
    # suite2p / manual-imagej-roi trace handling
    '''
    print('writing ROI traces and trial statistics to CSVs')

    traces.to_csv(join(analysis_dir, 'traces.csv'))
    # W/ default CSV output format, all columns after 'repeat' (last level of row
    # MultiIndex) are suite2p ROI numbers (not necessarily consecutive b/c of preceding
    # selection) which should be the same numbers in the columns of traces.csv.
    trial_stats.to_csv(join(analysis_dir, 'trial_stats.csv'))
    '''

    if save_figs and plot_dir is not None:
        roi_dir = join(plot_dir, 'ijroi')
        makedirs(roi_dir)

    z_indices = masks.roi_z[masks.roi_num.isin(roi_nums)].values

    title = 'ImageJ ROIs\nOrdered by Z plane\n*possibly [over/under]merged'

    fig, mean_df = plot_all_roi_mean_responses(trial_stats, roi_sortkeys=z_indices,
        title=title, cbar_label=trial_stat_cbar_title, cbar_shrink=0.4
    )

    if plot_dir is not None:
        savefig(fig, roi_dir, 'all_rois_by_z')

    component_df = mean_df[
        (mean_df.index.to_frame() == 'solvent').sum(axis='columns') == 1
    ]

    odor1_df = component_df[component_df.index.get_level_values('odor2') == 'solvent'
        ].droplevel('odor2')

    odor2_df = component_df[component_df.index.get_level_values('odor1') == 'solvent'
        ].droplevel('odor1')

    # TODO TODO TODO recheck this math
    # This order of DataFrame operations is required to get indexing to work (or at
    # least some other ways of ordering it doesn't work).
    sum_diff_df = -(mean_df - odor1_df - odor2_df).dropna()

    max_df = mean_df * np.nan
    for odor1 in sum_diff_df.index.get_level_values('odor1').unique():
        for odor2 in sum_diff_df.index.get_level_values('odor2').unique():
            o1_series = odor1_df.loc[odor1]
            o2_series = odor2_df.loc[odor2]
            max_series = np.maximum(o1_series, o2_series)

            max_idx = np.argmax(
                (max_df.index.get_level_values('odor1') == odor1) &
                (max_df.index.get_level_values('odor2') == odor2)
            )
            max_df.iloc[max_idx] = max_series
            # TODO why didn't this work?
            '''
            max_df.loc[
                (max_df.index.get_level_values('odor1') == odor1) &
                (max_df.index.get_level_values('odor2') == odor2)
            ] = max_series
            '''

    # dropna() is getting rid of the rows where >=1 odors are solvent
    max_diff_df = (max_df - mean_df).dropna()

    for diff_df, desc in ((max_diff_df, 'max'), (sum_diff_df, 'sum')):
        # TODO less gradations on these color bars? kinda packed.
        # TODO cbar_label? or just ok to leave it in title?
        viz.matshow(diff_df, title=f'Component {desc} minus observed', cbar_shrink=0.4,
            # TODO check that `xticklabels=str` works
            xticklabels=str, yticklabels=format_mix_from_strs, **diverging_cmap_kwargs
        )

        if plot_dir is not None:
            savefig(fig, roi_dir, f'diff_{desc}')

    for roi in trial_stats.columns:
        roi1_series = trial_stats.loc[:, roi]
        fig = plot_roi_stats_odorpair_grid(roi1_series, label=dff_cbar_title,
            cbar_shrink=0.4
        )

        if plot_dir is not None:
            savefig(fig, roi_dir, str(roi))

    print('generated plots based on traces from ImageJ ROIs')

    # TODO maybe refactor
    date_str, fly_str, thorimage_id = analysis_dir.split(os.path.sep)[-3:]

    # Not including concentrations in metadata to add, b/c I generally run this script
    # skipping all but final concentrations (process_experiment returns None for all
    # non-final concentrations)
    (name1, _), (name2, _) = odor_lists2names_and_conc_ranges(odor_lists)

    # TODO maybe factor metadata handling to process_experiment (though would need to
    # return dataframes from here rather than append to a global variable...)
    # (so we don't need to recompute names_and_conc_ranges, etc)

    new_level_names = ['date', 'fly_num', 'thorimage_id', 'name1', 'name2']
    new_level_values = [pd.Timestamp(date_str), int(fly_str), thorimage_id,
        name1, name2
    ]

    roi_df = trial_stats

    roi_df = util.addlevel(roi_df, new_level_names, new_level_values)

    mean_df = util.addlevel(mean_df, new_level_names, new_level_values)
    sum_diff_df = util.addlevel(sum_diff_df, new_level_names, new_level_values)
    max_diff_df = util.addlevel(max_diff_df, new_level_names, new_level_values)

    ij_roi_dfs.append(roi_df)

    ij_mean_dfs.append(mean_df)
    ij_sum_diff_dfs.append(sum_diff_df)
    ij_max_diff_dfs.append(max_diff_df)


# TODO maybe refactor (part of?) this to hong2p.suite2p
def run_suite2p(thorimage_dir, analysis_dir, overwrite=False):

    # TODO expose if_exists kwarg as run_suite2p  kwarg?
    # This will create a TIFF <analysis_dir>/raw.tif, if it doesn't already exist
    util.thor2tiff(thorimage_dir, output_dir=analysis_dir, if_exists='ignore',
        verbose=False
    )

    suite2p_dir = s2p.get_suite2p_dir(analysis_dir)
    if exists(suite2p_dir):
        # TODO TODO but maybe actually just return here? because if it failed with the
        # user-level options the first time, won't it just fail this run too?  (or are
        # there another random, small, fixeable errors?)
        '''
        # Since we currently depend on the contents of this directory existing for
        # analysis, and it's generated as one of the last steps in what suite2p does, so
        # many errors will cause this directory to not get generated.
        if not exists(s2p.get_suite2p_combined_dir(analysis_dir)):
            print('suite2p directory existed, but combined subdirectory did not. '
                're-running suite2p!'
            )
            overwrite = True
        '''

        if overwrite:
            shutil.rmtree(suite2p_dir)
            os.mkdir(suite2p_dir)
        else:
            return

    ops_dir = expanduser('~/.suite2p/ops')
    # NOTE: {ops_dir}/ops_user.npy is what gets written w/ the gui button to save new
    # defaults
    ops_file = join(ops_dir, 'ops_user.npy')
    # TODO print the path of the ops*.npy file we ultimately load ops from
    ops = s2p.load_s2p_ops(ops_file)

    # TODO TODO perhaps try having threshold_scaling depend on plane, and use run_plane
    # instead? (decrease on lower planes, where it's harder to find stuff generally)

    # TODO maybe use suite2p's options for ignoring flyback frames to ignore depths
    # beyond those i've now settled on?

    data_specific_ops = s2p.suite2p_params(thorimage_dir)
    for k, v in data_specific_ops.items():
        assert k in ops
        ops[k] = v

    # Only available in my fork of suite2p
    ops['exclude_filenames'] = (
        'ChanA_Preview.tif',
        'ChanB_Preview.tif',
        'lastpair_avg_mean_dff.tif',
    )

    db = {
        #'data_path': [thorimage_dir],
        # TODO TODO TODO update suite2p to take substrs / globs to ignore input files on
        # (at least for TIFFs specifically, to ignore other TIFFs in input dir)
        'data_path': [analysis_dir],
    }

    # TODO probably build up a list of directories for which suite2p was run for the
    # first time, and print at end, so the ROIs can be manually
    # inspected/categorized/merged

    success = False
    try:
        # TODO actually care about ops_end?
        ops_end = run_s2p(ops=ops, db=db)
        success = True

    except Exception as err:
        traceback.print_exc()
        failed_suite2p_dirs.append(analysis_dir)

        make_fail_indicator_file(analysis_dir, 'suite2p', err)

        # TODO should i do this or should i just use the other plane data?
        # or have some necessary steps not run by the point the likely error is
        # encountered (usually a ValueError about ROIs not being found, and usually in
        # one of the deeper planes), even for the earlier planes?
        #print(f'Removing suite2p created {suite2p_dir} because run_s2p failed')
        #shutil.rmtree(suite2p_dir)

    if success:
        # NOTE: this is not changing the iscell.npy files in the plane<x> folders,
        # as I'm trying to avoid dealing with those folders at all as much as possible
        combined_dir = s2p.get_suite2p_combined_dir(analysis_dir)
        s2p.mark_all_suite2p_rois_good(combined_dir)


def multiprocessing_namespace_from_globals(manager):
    types2manager_types = {
        list: manager.list,
    }
    namespace = manager.dict()
    for name, value in globals().items():
        val_type = type(value)
        if val_type in types2manager_types:
            namespace[name] = types2manager_types[val_type]()

    return namespace


def update_globals_from_shared_state(shared_state):
    if shared_state is None:
        return

    for k, v in shared_state.items():
        globals()[k] = v


def proxy2orig_type(proxy):

    if type(proxy) is mp.managers.DictProxy:
        return {k: proxy2orig_type(v) for k, v in proxy.items()}

    elif type(proxy) is mp.managers.ListProxy:
        return [proxy2orig_type(x) for x in proxy]

    else:
        # This should only be reached in cases where `proxy` is not actually a proxy.
        return proxy


def multiprocessing_namespace_to_globals(shared_state):
    globals().update({
        k: proxy2orig_type(v) for k, v in shared_state.items()
    })


# NOTE: can't use with multiprocessing since pickle (what it uses to serialize) can't
# handle closures / things like this
# TODO try joblib though. they use something other than pickle by default i think, for
# what seems like related reasons
def capture_stdout_and_stderr(fn):
    """Calls `fn`, returns (<args>, <captured output>, <fn return value>)
    """
    def fn_captured(*args, **kwargs):
        f = StringIO()
        print('args:', args)
        print('kwargs:', kwargs)
        # TODO using same StringIO object for both work? if not, how else to have
        # outputs interleaved as they would be normally?
        with redirect_stdout(f), redirect_stderr(f):
            ret = fn(*args, **kwargs)

        print('stdout/stderr:')
        print(f)
        # TODO convert f to regular str here?
        return args, f, ret

    return fn_captured


# TODO TODO probably refactor so that this is essentially just populating
# lists[/listproxies] of dataframes from s2p/ijroi stuff (extracting in ij case, merging
# in both cases, also converting to trial stats in both), and then move most plotting to
# after this (and cache output of the loop over calls to this) maybe leave plotting of
# dF/F images and stuff closer to raw data in here. (or at least factor out time
# intensive df/f image plots and cache/skip those separately)
def process_experiment(date_and_fly_num, thor_image_and_sync_dir, shared_state=None):
    """
    Args:
        ...
        shared_state (multiprocessing.managers.DictProxy): str global variable names ->
        other proxy objects
    """

    # TODO since this is updating **globals** after all, try moving this to a wrapper fn
    # (combine w/ output capture?) and remove shared_state kwarg here if it works the
    # same (not possible as long as i'm using multiprocessing and not something like
    # pathos, for same reason i couldn't use a wrapped process_experiment to capture
    # stdout: can't serialize closures w/ pickle, which is what multiprocessing uses)
    #
    # Only relevant if called via multiprocessing, where this is how we get access to
    # the proxies that will feed back to the corresponding global variables that get
    # modified under this function.
    update_globals_from_shared_state(shared_state)

    date, fly_num = date_and_fly_num
    thorimage_dir, thorsync_dir = thor_image_and_sync_dir

    if 'diag' in thorimage_dir:
        if not analyze_glomeruli_diagnostics:
            print('skipping because experiment is just glomeruli diagnostics\n')
            return

        is_glomeruli_diagnostics = True
    else:
        is_glomeruli_diagnostics = False

    analysis_dir = get_analysis_dir(date, fly_num, split(thorimage_dir)[1])
    makedirs(analysis_dir)

    if retry_previously_failed:
        clear_fail_indicators(analysis_dir)
    else:
        has_failed, suffixes_str = last_fail_suffixes(analysis_dir)
        if has_failed:
            print(f'skipping because previously failed {suffixes_str}\n')
            return

    exp_start = time.time()

    single_plane_fps, xy, z, c, n_flyback, _, xml = thor.load_thorimage_metadata(
        thorimage_dir, return_xml=True
    )

    if not analyze_2d_tests and z == 1:
        print('skipping because experiment is single plane\n')
        return

    experiment_id = shorten_path(thorimage_dir)
    experiment_basedir = util.to_filename(experiment_id, period=False)

    # Created below after we decide whether to skip a given experiment based on the
    # experiment type, etc.
    # TODO rename to experiment_plot_dir or something
    plot_dir = join(plot_fmt, experiment_basedir)

    # TODO refactor scandir thing to not_empty or something
    if (skip_if_experiment_plot_dir_exists and exists(plot_dir)
        and any(os.scandir(plot_dir))):

        print(f'skipping because {plot_dir} exists\n')
        return

    def suptitle(title, fig=None):
        if title is None:
            return

        if fig is None:
            fig = plt.gcf()

        fig.suptitle(f'{experiment_id}\n{title}')

    def exp_savefig(fig, desc, **kwargs):
        return savefig(fig, plot_dir, desc, **kwargs)

    makedirs(plot_dir)

    if links_to_input_dirs:
        symlink(thorimage_dir, join(plot_dir, 'thorimage'))
        symlink(analysis_dir, join(plot_dir, 'analysis'))

    yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(xml)
    print('yaml_path:', shorten_path(yaml_path, n_parts=2))

    # TODO also exclude stuff where stimuli were not pairs. maybe just try/except
    # the code extracting stimulus info in here? or separate fn, run first, to
    # detect *if* we are dealing w/ pair-grid data?
    if not is_glomeruli_diagnostics:
        # So that we can count how many flies we have for each odor pair (and
        # concentration range, in case we varied that at one point)
        names_and_concs_tuple = odor_lists2names_and_conc_ranges(odor_lists)

        if final_concentrations_only:
            names, curr_concs = separate_names_and_concs_tuples(
                names_and_concs_tuple
            )

            if (names in names2final_concs and
                names2final_concs[names] != curr_concs):

                print('skipping because not using final concentrations for '
                    f'{names}\n'
                )
                return

    pulse_s = float(int(yaml_data['settings']['timing']['pulse_us']) / 1e6)
    if pulse_s < 3:
        print(f'skipping because odor pulses were {pulse_s} (<3s) long (old)\n')
        return

    if is_pairgrid(odor_lists):
        if not analyze_reverse_order and is_reverse_order(odor_lists):
            print('skipping because a reverse order experiment\n')
            return
    else:
        if analyze_pairgrids_only:
            print('skipping because not a pair grid experiment\n')
            return

    # Checking here even though `seen_stimulus_yamls2thorimage_dirs` was pre-computed
    # elsewhere because we don't necessarily want to err if this would only get
    # triggered for stuff that would get skipped in this function.
    if (yaml_path in seen_stimulus_yamls2thorimage_dirs and
        seen_stimulus_yamls2thorimage_dirs[yaml_path] != [thorimage_dir]):

        short_yaml_path = shorten_path(yaml_path, n_parts=2)

        err_msg = (f'stimulus yaml {short_yaml_path} seen in:\n'
            f'{pformat(seen_stimulus_yamls2thorimage_dirs[yaml_path])}'
        )
        warnings.warn(err_msg)
        #raise ValueError(err_msg)

    if not is_glomeruli_diagnostics:
        # In case where this is a DictProxy, these empty lists (ListProxy in that case)
        # should all have been added before parallel starmap (outside this fn).
        if names_and_concs_tuple not in names_and_concs2analysis_dirs:
            names_and_concs2analysis_dirs[names_and_concs_tuple] = []

        names_and_concs2analysis_dirs[names_and_concs_tuple].append(analysis_dir)

        #names_concs_and_analysis_dir_tuples.append(
        #    (names_and_concs_tuple, analysis_dir)
        #)

    # NOTE: converting to list-of-str FIRST, so that each element will be
    # hashable, and thus can be counted inside `olf.remove_consecutive_repeats`
    odor_order_with_repeats = [format_odor_list(x) for x in odor_lists]
    odor_order, n_repeats = olf.remove_consecutive_repeats(odor_order_with_repeats)

    # TODO delete if i manage to refactor code below to only do the formatting of odors
    # right before plotting, rather than in the few lines before this
    #
    # This is just intended to target glomeruli information for the glomeruli
    # diagnostic experiments.
    odor_str2target_glomeruli = {
        s: o[0]['glomerulus'] for s, o in zip(odor_order_with_repeats, odor_lists)
        if len(o) == 1 and 'glomerulus' in o[0]
    }

    # TODO use that list comprehension way of one lining this? equiv for sets?
    name_lists = [[o['name'] for o in os] for os in odor_lists]
    for ns in name_lists:
        for n in ns:
            if n not in odor2abbrev:
                #odors_without_abbrev.add(n)
                if n not in odors_without_abbrev:
                    odors_without_abbrev.append(n)
    del name_lists


    before = time.time()

    bounding_frame_yaml_cache = join(analysis_dir, 'trial_bounding_frames.yaml')

    if ignore_bounding_frame_cache or not exists(bounding_frame_yaml_cache):
        # TODO TODO don't bother doing this if we only have imagej / suite2p analysis
        # left to do, and the required output directory doesn't exist / ROIs haven't
        # been manually drawn/filtered / etc
        try:
            bounding_frames = thor.assign_frames_to_odor_presentations(thorsync_dir,
                thorimage_dir
            )
            assert len(bounding_frames) == len(odor_order_with_repeats)

            # TODO TODO move inside assign_frames_to_odor_presentations
            # Converting numpy int types to python int types, and tuples to lists,
            # for (much) nicer YAML output.
            bounding_frames = [ [int(x) for x in xs] for xs in bounding_frames]

            with open(bounding_frame_yaml_cache, 'w') as f:
                yaml.dump(bounding_frames, f)

        # Currently seems to reliably happen iff we somehow accidentally also image
        # with the red channel (which was allowed despite those channels having gain
        # 0 in the few cases so far)
        except AssertionError as err:
            traceback.print_exc()
            print()

            failed_assigning_frames_to_odors.append(thorimage_dir)

            make_fail_indicator_file(analysis_dir, 'assign_frames', err)

            return

    else:
        with open(bounding_frame_yaml_cache, 'r') as f:
            bounding_frames = yaml.safe_load(f)

    # (loading the HDF5 should be the main time cost in the above fn)
    load_hdf5_s = time.time() - before


    if do_suite2p:
        run_suite2p(thorimage_dir, analysis_dir, overwrite=overwrite_suite2p)

    if analyze_suite2p_outputs:
        if not any([b in thorimage_dir for b in bad_suite2p_analysis_dirs]):
            suite2p_trace_plots(analysis_dir, bounding_frames, odor_lists,
                plot_dir=plot_dir
            )
        else:
            full_bad_suite2p_analysis_dirs.append(analysis_dir)
            print('not making suite2p plots because outputs marked bad\n')

    if not non_suite2p_analysis:
        print()

        # TODO how to have this work from inside multiprocessing invokation too? just
        # refactor? return True/False for whether we should keep going?
        #if quick_test_only:
        #    break

        return

    if not ignore_existing_nonsuite2p_outputs:
        # Assuming that if analysis_dir has *any* plots directly inside of it, it
        # has all of what we want from non_suite2p_analysis (including any
        # intermediates that would go in analysis_dir).
        # Set ignore_existing_nonsuite2p_outputs=False to do this analysis
        # regardless, regenerating any overlapping plots.
        if len(glob.glob(join(plot_dir, f'*.{plot_fmt}'))) > 0:
            print('skipping non-suite2p analysis because plot dir contains '
                f'{plot_fmt}\n'
            )

            #if quick_test_only:
            #    break

            return

    before = time.time()

    movie = thor.read_movie(thorimage_dir)

    read_movie_s = time.time() - before

    # TODO maybe move ijroi stuff to after loop and compute df/f movie if this is
    # requested, IF IT MAKES ANY MORE SENSE TO START W/ DF/F MOVIE RATHER THAN
    # EXTRACTING TRACES FIRST

    # TODO refactor fn to separate plotting from suite2p/ijroi data source
    if analyze_ijrois:
        # TODO TODO make sure none of the stuff w/ suite2p outputs marked bad should
        # also just generally be marked bad, s.t. not run here
        ij_trace_plots(analysis_dir, bounding_frames, odor_lists, movie,
            plot_dir=plot_dir
        )

    # TODO only save this computed from motion corrected movie, in any future cases
    # where we are actually motion correcting as part of this pipeline
    # TODO TODO maybe make a plot like this, but use the actual frame times
    # (via thor.get_frame_times) + actual odor onset / offset times, and see if
    # that lines up any better?
    '''
    avg = util.full_frame_avg_trace(movie)
    ffavg_fig, ffavg_ax = plt.subplots()
    ffavg_ax.plot(avg)
    for _, first_odor_frame, _ in bounding_frames:
        # TODO need to specify ymin/ymax to existing ymin/ymax?
        ffavg_ax.axvline(first_odor_frame)

    ffavg_ax.set_xlabel('Frame Number')
    exp_savefig(ffavg_fig, 'ffavg')
    #plt.show()
    '''

    # TODO actually possible for it to be non-int in Experiment.xml?
    zstep_um = int(round(float(xml.find('ZStage').attrib['stepSizeUM'])))

    def micrometer_depth_title(ax, z_index) -> None:
        curr_z = -zstep_um * z_index
        ax.set_title(f'{curr_z} $\\mu$m', fontsize=ax_fontsize)


    def plot_and_save_dff_depth_grid(dff_depth_grid, fname_prefix, title=None,
        cbar_label=None, **imshow_kwargs):

        # Will be of shape (1, z), since squeeze=False
        fig, axs = plt.subplots(ncols=z, squeeze=False,
            figsize=single_dff_image_row_figsize
        )

        for d in range(z):
            ax = axs[0, d]

            if z > 1:
                micrometer_depth_title(ax, d)

            im = dff_imshow(ax, dff_depth_grid[d], **imshow_kwargs)

        viz.add_colorbar(fig, im, label=cbar_label, shrink=0.68)

        suptitle(title, fig)
        fig_path = exp_savefig(fig, fname_prefix)

        return fig_path


    if z > n_top_z_to_analyze:
        warnings.warn(f'{thorimage_dir}: only analyzing top {n_top_z_to_analyze} '
            'slices'
        )
        movie = movie[:, :n_top_z_to_analyze, :, :]
        assert movie.shape[1] == n_top_z_to_analyze
        z = n_top_z_to_analyze

    anat_baseline = movie.mean(axis=0)
    plot_and_save_dff_depth_grid(anat_baseline, 'avg', 'average of whole movie',
        vmin=anat_baseline.min(), vmax=anat_baseline.max(), cmap='gray'
    )

    save_dff_tiff = want_dff_tiff
    if save_dff_tiff:
        dff_tiff_fname = join(analysis_dir, 'dff.tif')
        if exists(dff_tiff_fname):
            # To not write large file unnecessarily. This should never really change,
            # especially not if computed from non-motion-corrected movie.
            save_dff_tiff = False

    if save_dff_tiff:
        trial_dff_movies = []

    trial_max_dff_list = []

    for i, odor_str in enumerate(odor_order):

        if odor_str in odor_str2target_glomeruli:
            target_glomerulus = odor_str2target_glomeruli[odor_str]
            odor_str = f'{odor_str} ({target_glomerulus})'
        else:
            target_glomerulus = None

        # TODO either:
        # - always use 2 digits (leading 0)
        # - pick # of digits from len(odor_order)
        plot_desc = f'{i + 1}_{odor_str}'

        trial_heatmap_fig, trial_heatmap_axs = plt.subplots(nrows=n_repeats,
            ncols=z, squeeze=False, figsize=(6.4, 3.9)
        )

        trial_mean_dffs = []

        for n in range(n_repeats):
            # This works because the repeats of any particular odor were all
            # consecutive in all of these experiments.
            presentation_index = (i * n_repeats) + n

            start_frame, first_odor_frame, end_frame = bounding_frames[
                presentation_index
            ]

            # NOTE: was previously skipping the frame right before the odor frame
            # too, but I think this was mainly out of fear the assignment of the
            # first odor frame might have been off by one, but I haven't really
            # seen examples of this, after looking through a lot of examples.
            # Possible examples where the frame before first_odor_frame also has
            # some response (looking at first repeat of last odor pair in
            # volumetric stuff only):
            # - 2021-03-08/1/acetone_and_butanal
            # - 2021-03-08/1/acetone_and_butanal_redo
            # - 2021-03-08/1/1hexanol_and_ethyl_hexanoate
            # - 2021-04-28/1/butanal_and_acetone
            # (and didn't check stuff past that for the moment)
            # conclusion: need to keep ignoring the last frame until i can fix this
            # issue
            baseline = movie[start_frame:(first_odor_frame - 1)].mean(axis=0)
            #baseline = movie[start_frame:first_odor_frame].mean(axis=0)

            '''
            print('baseline.min():', baseline.min())
            print('baseline.mean():', baseline.mean())
            print('baseline.max():', baseline.max())
            '''
            # hack to make df/f values more reasonable
            # TODO still an issue?
            # TODO TODO maybe just add like 1e4 or something?
            #baseline = baseline + 10.

            # TODO TODO why is baseline.max() always the same???
            # (is it still?)

            dff = (movie[first_odor_frame:(end_frame + 1)] - baseline) / baseline

            trial_max_dff = dff.max(axis=0)
            trial_max_dff_list.append(trial_max_dff)

            if save_dff_tiff:
                trial_dff_movie = (
                    (movie[start_frame:(end_frame + 1)] - baseline) / baseline
                )
                trial_dff_movies.append(trial_dff_movie)

            # TODO TODO change to using seconds and rounding to nearest[higher/lower
            # multiple?] from there
            # TODO TODO TODO if i'm not going to quantitatively check (at least once on
            # a reasonable sample of my data) that more won't make better plots, then at
            # least try with like ~5 and compare that to current
            #response_volumes = 1
            response_volumes = 2

            # TODO off by one at start? (still relevant?)
            mean_dff = dff[:response_volumes].mean(axis=0)

            trial_mean_dffs.append(mean_dff)

            # was checking range of pixels values made sense. some are reported as max i
            # believe, and probably still are. could maybe just be real noise though.
            '''
            max_dff = dff.max(axis=0)
            print(max_dff.min())
            print(max_dff.mean())
            print(max_dff.max())
            print()
            '''

            for d in range(z):
                ax = trial_heatmap_axs[n, d]

                # TODO offset to left so it doesn't overlap and re-enable
                if d == 0:
                    # won't work until i fix set_axis_off thing in dff_imshow above
                    ax.set_ylabel(f'Trial {n + 1}', fontsize=ax_fontsize,
                        rotation='horizontal'
                    )

                if n == 0 and z > 1:
                    micrometer_depth_title(ax, d)

                im = dff_imshow(ax, mean_dff[d])

        viz.add_colorbar(trial_heatmap_fig, im, label=dff_cbar_title, shrink=0.32)

        suptitle(odor_str, trial_heatmap_fig)
        exp_savefig(trial_heatmap_fig, plot_desc + '_trials')

        avg_mean_dff = np.mean(trial_mean_dffs, axis=0)
        fig_path = plot_and_save_dff_depth_grid(avg_mean_dff, plot_desc, title=odor_str,
            cbar_label=f'Mean {dff_latex}'
        )

        # TODO TODO TODO (at least until i have stuff in a standardized l/r orientation
        # (flipping one), add average image / projection on left / bottom of glomeruli
        # diag dF/F plots, to tell from looking at one plot whether glomerulus is on
        # right side)
        # TODO TODO maybe also include some quick reference to
        # previously-presented-stimulus, to check for constamination components of
        # noise?
        # TODO TODO TODO also include a subdir of glomeruli_diagnostics that just links
        # to the corresponding directories of individual flies, to not have to sift
        # through them at the top-level that also has real experiments
        if target_glomerulus is not None:
            # gsheet only has labels on a per-fly basis, and those should apply to the
            # glomeruli diagnostic experiment corresponding to the same FOV as the other
            # experiments. Don't want to link any other experiments anywhere under here.
            rel_exp_dir = '/'.join(analysis_dir.split(os.sep)[-3:])
            if rel_exp_dir in unused_glomeruli_diagnostics:
                continue

            glomerulus_dir = join(across_fly_glomeruli_diags_dir, target_glomerulus)
            makedirs(glomerulus_dir)

            label_subdir = None
            try:
                fly_diag_statuses = glomeruli_diag_status_df.loc[(date, fly_num)]

                fly_diags_labelled = fly_diag_statuses.any()
                if not fly_diags_labelled:
                    label_subdir = 'unlabelled'

            except KeyError:
                label_subdir = 'unlabelled'

            date_str = f'{date:%Y-%m-%d}'

            if label_subdir is None:
                try:
                    curr_diag_good = fly_diag_statuses.loc[target_glomerulus.lower()]

                except KeyError:
                    warnings.warn(f'target glomerulus {target_glomerulus} not in Google'
                        ' sheet! add column and label data. currently not linking these'
                        ' plots!'
                    )
                    continue

                if curr_diag_good:
                    label_subdir = 'good'
                else:
                    label_subdir = 'bad'
            else:
                warnings.warn('please label quality glomeruli diagnostics for fly '
                    f'{(date_str, fly_num)} in google sheet.'
                )

            label_dir = join(glomerulus_dir, label_subdir)
            makedirs(label_dir)

            link_prefix = '_'.join(experiment_id.split(os.sep)[:-1])
            link_path = join(label_dir, f'{link_prefix}.{plot_fmt}')

            if exists(link_path):
                # Just warning so that all the average images, etc, will still be
                # created, so those can be used to quickly tell which experiment
                # corresponded to the same side as the real experiments in the same fly.
                warnings.warn(f'{(date_str, fly_num)} has multiple glomeruli diagnostic'
                    ' experiments. add all but one to unused_glomeruli_diagnostics. '
                    'FIRST IS CURRENTLY LINKED BUT MAY NOT BE THE RELEVANT EXPERIMENT!'
                )
                continue

            # TODO make symlinks relative (more complicated) to make it easier to move
            # the directory manually
            symlink(fig_path, link_path)

    if save_dff_tiff:
        delta_f_over_f = np.concatenate(trial_dff_movies)

        assert delta_f_over_f.shape == movie.shape

        print(f'writing dF/F TIFF to {dff_tiff_fname}...', flush=True, end='')

        util.write_tiff(dff_tiff_fname, delta_f_over_f, strict_dtype=False)

        print(' done', flush=True)

        del delta_f_over_f, trial_dff_movies

    max_trial_dff = np.max(trial_max_dff_list, axis=0)
    max_trial_dff_tiff_fname = join(analysis_dir, 'max_trial_dff.tif')
    util.write_tiff(max_trial_dff_tiff_fname, max_trial_dff, strict_dtype=False)
    print(f'wrote max dF/F to {max_trial_dff_tiff_fname}')

    plot_and_save_dff_depth_grid(max_trial_dff, 'max_dff',
        cbar_label=f'Max {dff_latex}', vmin=None, vmax=None
    )

    plt.close('all')

    exp_total_s = time.time() - exp_start
    exp_processing_time_data.append((load_hdf5_s, read_movie_s, exp_total_s))

    print()


def n_largest_signal_rois(sdf, n=5):

    def print_pd(x):
        print(x.to_string(float_format=lambda x: '{:.1f}'.format(x)))

    max_df = sdf.groupby(['date', 'fly_num']).max()
    print('max signal across all ROIs:')
    print_pd(max_df.max(axis=1, skipna=True))
    print()

    for index, row in max_df.iterrows():
        # TODO want to dropna? how does nlargest handle if n goes into NaN?
        nlargest = row.nlargest(n=n)

        print(f'{format_date(index[0])}/{index[1]}')

        print_pd(nlargest)
        print()


def format_fly_and_roi(fly_and_roi):
    fly, roi = fly_and_roi
    return f'{fly}: {roi}'


# TODO refactor. probably already have something for this.
def odor_str2conc(odor_str):
    return 0 if odor_str == 'solvent' else np.float_power(10,
        float(odor_str.split('@')[-1].strip())
    )


# TODO TODO TODO if they are fitting using a function like this, that only has as
# product of fmax*e_i, then how do they get a single fmax for computing the model
# response to a mixture? presumably the fmax fit on each component will differ?
# or do they do some subsequent algebra to get one fmax and then a new e_i from each of
# the outputs of the component fits?
# TODO i shouldn't deal w/ concs in log units or something should i?
def model_orn_singleodor_responses(concs, fmax_times_efficacy, ec50):
    # TODO check this vectorized formula works correctly
    return (concs * fmax_times_efficacy / ec50) / (1 + concs / ec50)


# TODO better name for this fn
def analyze_onepair(trial_df, mean_df, sum_diff_df, max_diff_df):

    # TODO refactor to share code for odor1 and odor2 stuff
    odor1_df = mean_df.loc[:, (slice(None), 'solvent')].droplevel('odor2',
        axis='columns'
    )
    concs = odor1_df.columns.map(odor_str2conc).values

    fit_params = []
    # TODO do i need to call this per row or nah? and if not, will is still behave same
    # as this way (w/ diff params for each row) if i do call it w/ 2D input?
    for _, roi_mean_responses in odor1_df.iterrows():
        # TODO add reasonable bounds to parameters?
        popt, pcov = curve_fit(model_orn_singleodor_responses, concs,
            roi_mean_responses
        )
        fit_params.append(popt)

    odor1_param_df = pd.DataFrame(data=fit_params, index=odor1_df.index,
        columns=['fmax_times_efficacy', 'ec50']
    )


    odor2_df = mean_df.loc[:, ('solvent', slice(None))].droplevel('odor1',
        axis='columns'
    )
    concs = odor2_df.columns.map(odor_str2conc).values

    fit_params = []
    # TODO do i need to call this per row or nah? and if not, will is still behave same
    # as this way (w/ diff params for each row) if i do call it w/ 2D input?
    for _, roi_mean_responses in odor2_df.iterrows():
        popt, pcov = curve_fit(model_orn_singleodor_responses, concs,
            roi_mean_responses
        )
        fit_params.append(popt)

    odor2_param_df = pd.DataFrame(data=fit_params, index=odor2_df.index,
        columns=['fmax_times_efficacy', 'ec50']
    )

    param_df = pd.concat([odor1_param_df, odor2_param_df], names=['odor'],
        keys=['odor1','odor2'], axis='columns'
    )

    # TODO rename/refactor. it's just mean_df w/ a diff col index
    #conc_df = mean_df.copy()
    #conc_df.columns = pd.MultiIndex.from_frame(
    #    conc_df.columns.to_frame().applymap(odor_str2conc)
    #)

    concs_df = mean_df.columns.to_frame().applymap(odor_str2conc)
    concs_df.columns.name = 'odor'
    odor_col_names = concs_df.columns
    ests = []
    #for concs in concs_df.itertuples(index=False):

    odor_nums = []
    odor_dens = []
    for odor, conc_df in concs_df.groupby('odor', axis='columns', sort=False):

        odor_param_df = param_df.loc[:, odor]
        fmax_times_efficacy = odor_param_df['fmax_times_efficacy']
        ec50 = odor_param_df['ec50']

        # TODO calculate this in a less stupid way
        odor_num = (np.outer(fmax_times_efficacy, conc_df) /
            np.outer(ec50, np.ones_like(conc_df))
        )
        odor_den = np.outer(1/ec50, conc_df)

        odor_nums.append(odor_num)
        odor_dens.append(odor_den)

    num = np.sum(odor_nums, axis=0)
    den = np.sum(odor_dens, axis=0) + 1

    est = num / den
    assert concs_df.index.equals(mean_df.columns)
    est_df = pd.DataFrame(est, columns=concs_df.index, index=param_df.index)

    # TODO TODO TODO maybe test my fitting method on the singh et al 2019 data, to make
    # sure i'm getting it right, esp w/ the fmax handling, to the extent it doesn't end
    # up being trivial

    # TODO TODO TODO how to do fitting like above, but also incorporating lateral
    # inhibition?


    # TODO TODO TODO plot a few example curves (for components and mix), capturing
    # diversity of difference / fit quality (+ to see what kinds of behavior model is
    # actually capturing, as fit on my data). to what extent is lack of saturation a
    # problem? ways to deal with it?

    shared_kwargs = dict(
        figsize=(3.0, 7.0),
        xticklabels=format_mix_from_strs,
        yticklabels=format_fly_and_roi,
        ylabel='ROI',
        # TODO is this cbar_label appropriate for all plots this is used in?
        cbar_shrink=0.3,
    )

    matshow_kwargs = dict(cbar_label=trial_stat_cbar_title, cmap=cmap, **shared_kwargs)

    matshow_diverging_kwargs = dict(cbar_label=diff_cbar_title, **diverging_cmap_kwargs,
        **shared_kwargs
    )

    # TODO TODO do i have enough of a baseline to compute stddev there and use that to
    # get responders significantly above[/below] baseline?

    # TODO TODO TODO maybe a normalized version of this (+ clustering on that?) to also
    # see general interaction types of weak responders?
    # or maybe some kind of z-scored version where noiser stuff gets weighted a bit less
    # too (need all trials for that)?
    fig, _ = viz.matshow(mean_df, **matshow_kwargs)

    fig, _ = viz.matshow(sum_diff_df, title='Sum - obs', **matshow_diverging_kwargs)

    fig, _ = viz.matshow(max_diff_df, title='Max - obs', **matshow_diverging_kwargs)

    fig, _ = viz.matshow(est_df, title='Competetive binding model', **matshow_kwargs)

    cb_diff_df = est_df - mean_df

    fig, _ = viz.matshow(cb_diff_df, title='Competitive binding model - obs',
        **matshow_diverging_kwargs
    )

    # TODO TODO could try using row_colors derived from non-hierarchichal clustering
    # methods to plots those? maybe even disabling the dendrogram?
    # TODO or from fly / glomeruli (for those w/ names)?
    # TODO TODO for glomeruli, maybe do one version w/ and w/o non-named glomeruli

    # 'average' is the default for sns.clustermap
    # 'single' is the default for scipy.cluster.hierarchy.linkage
    # (and there are ofc several others)
    #cluster_methods = ['average', 'single']
    # 'euclidean' is the default for both. just wanted to try 'correlation' too
    #cluster_metrics = ['euclidean', 'correlation']

    cluster_methods = ['average']
    cluster_metrics = ['correlation']

    diverging_clustermap_kwargs = {k: v for k, v in matshow_diverging_kwargs.items()
        if k not in ('figsize', 'cbar_shrink')
    }

    for method in cluster_methods:
        for metric in cluster_metrics:

            # TODO refactor to also loop over dfs (and their kwargs...)?

            # TODO some way to get cbar on right that plays nice w/ constrained layout?
            # maybe just disable and add after (other things seaborn is doing preclude
            # constrained layout anyway)?

            clustergrid = viz.clustermap(mean_df, col_cluster=False, cmap=cmap,
                xticklabels=format_mix_from_strs, yticklabels=format_fly_and_roi,
                cbar_label=trial_stat_cbar_title, ylabel='ROI', method=method
            )
            clustergrid.ax_heatmap.set_title(
                f'clustering params: method={method}, metric={metric}'
            )

            clustergrid = viz.clustermap(cb_diff_df, col_cluster=False,
                method=method, **diverging_clustermap_kwargs
            )
            clustergrid.ax_heatmap.set_title(
                'Competetive binding model - obs\n'
                f'clustering params: method={method}, metric={metric}'
            )


# TODO better name for this fn (+ probably call at end of main, not just behind -c flag)
def analyze_cache():
    # TODO TODO TODO plot hallem activations for odors in each pair, to see which
    # glomeruli we'd expect to find (at least at their concentrations)
    # (though probably factor it into its own fn and maybe call in main rather than
    # here?)

    warnings.simplefilter('error', pd.errors.PerformanceWarning)

    all_trial_df = pd.read_pickle('trial_df.p').T
    all_mean_df = pd.read_pickle('mean_df.p').T
    all_sum_diff_df = pd.read_pickle('sum_diff_df.p').T
    all_max_diff_df = pd.read_pickle('max_diff_df.p').T


    # TODO TODO TODO add fly / recording IDs by numbering groups

    # TODO print fly -> (date, fly_num, thorimage_id) legend

    import ipdb; ipdb.set_trace()

    def onepair_dfs(name1, name2, *dfs):
        assert len(dfs) > 0

        def onepair_df(name1, name2, df):
            # TODO will probably need to fix this part after moving ['date', 'fly_num',
            # 'thorimage_id'] keys to other index before `fly`
            import ipdb; ipdb.set_trace()
            df = df.droplevel(['date', 'fly_num', 'thorimage_id'], axis='columns')

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

                return sort_odor_indices(df.loc[:, (name1, name2)].dropna(axis='index'))

        return tuple([onepair_df(name1, name2, df) for df in dfs])


    for name1, name2 in all_trial_df.columns.to_frame(index=False)[['name1','name2']
        ].drop_duplicates().itertuples(index=False):

        trial_df, mean_df, sum_diff_df, max_diff_df = onepair_dfs(name1, name2,
            all_trial_df, all_mean_df, all_sum_diff_df, all_max_diff_df
        )
        # TODO delete
        print(name1, name2)
        import ipdb; ipdb.set_trace()
        #

        analyze_onepair(trial_df, mean_df, sum_diff_df, max_diff_df)

    # TODO TODO TODO try to cluster odor mixture behavior types across odor pairs, but
    # either sorting so ~most activating is always A or maybe by adding data duplicated
    # such that it appears once for A=x,B=y and once for A=y,B=x
    # TODO other ways to achieve some kind of symmetry here?

    plt.show()
    import ipdb; ipdb.set_trace()
    #"""


# TODO function for making "abbreviated" tiffs, each with the first ~2-3s of baseline,
# 4-7s of response (~7 should allow some baseline after) per trial, for more quickly
# drawing ROIs (without using some more processed version) (leaning towards just trying
# to use the max-trial-dff or something instead though... may still be useful for
# checking that?)


def main():
    global names2final_concs
    global seen_stimulus_yamls2thorimage_dirs
    global names_and_concs2analysis_dirs

    atexit.register(delete_empty_dirs)

    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--parallel', action='store_true',
        help='Enables parallel calls to process_experiment. '
        'Disabled by default because it can complicate debugging.'
    )

    parser.add_argument('-c', '--analyze-cache-only', action='store_true',
        help='only analyze cached aggregate ROI statistics across flies'
    )

    #parser.add_argument('-t', '--test', action='store_true',
    #    help='only *complete* main loop 1 time, for faster testing'
    #)
    args = parser.parse_args()

    parallel = args.parallel
    analyze_cache_only = args.analyze_cache_only
    #quick_test_only = args.test

    del parser, args

    if analyze_cache_only:
        analyze_cache()
        return

    if parallel:
        import matplotlib
        # Won't get warnings that some of the interactive backends give in the
        # multiprocessing case, but can't make interactive plots.
        matplotlib.use('agg')

    if analyze_glomeruli_diagnostics:
        # Always want to delete and remake this in case labels in gsheet have changed.
        if exists(across_fly_glomeruli_diags_dir):
            shutil.rmtree(across_fly_glomeruli_diags_dir)

        makedirs(across_fly_glomeruli_diags_dir)

    # TODO note that 2021-07-(21,27,28) contain reverse-order experiments. indicate
    # this fact in the plots for these experiments!!! (just skipping them for now
    # anyway)

    # TODO revisit "not cell"s for 2021-04-29/1/butanal_and_acetone

    # Using this in addition to ignore_prepairing in call below, because that changes
    # the directories considered for Thor[Image/Sync] pairing, and would cause that
    # process to fail in some of these cases.
    bad_thorimage_dirs = [
        # Previously skipped because responses in df/f images seem weak.
        # Also using old pulse lengths, so would be skipped anyway.
        '2021-03-08/2',

        # Just has glomeruli diagnostics. No real data.
        '2021-05-11/1',

        # Both flies only have glomeruli diagnostics.
        '2021-06-07',

        # dF/F images are basically empty. suite2p only pulled out a ~4-5 ROIs that
        # seemed to have any signal, and all weak.
        '2021-06-24/1/msl_and_hh',

        # suite2p is failing on this b/c it can't find any ROIs once it gets to plane 1
        # (planes are numbered from 0). dF/F images much weaker than other flies w/ this
        # odor pair.
        '2021-05-05/1/butanal_and_acetone',

        # Looking at the matrices, it's pretty clear that two concentrations were
        # swapped here (two of butanal, I believe). Could try comparing after
        # re-ordering data appropriately (though possible order effects could make data
        # non-comparable)
        # TODO potentially just delete this data to avoid accidentally including it...
        '2021-05-10/1/butanal_and_acetone',

        # Frame <-> time assignment is currently failing for all the real data from this
        # day.
        #'2021-06-24/2',

        # All planes same (probably piezo not set up properly at acquisition), in all
        # three recordings.
        '2021-07-21/1',

    ]
    common_paired_thor_dirs_kwargs = dict(
        start_date='2021-03-07', ignore=bad_thorimage_dirs, ignore_prepairing=('anat',)
    )

    names2final_concs, seen_stimulus_yamls2thorimage_dirs, names_and_concs_tuples = \
        odor_names2final_concs(**common_paired_thor_dirs_kwargs)

    keys_and_paired_dirs = util.paired_thor_dirs(verbose=True, print_skips=False,
        print_fast=False, print_full_paths=print_full_paths,
        **common_paired_thor_dirs_kwargs
    )
    del common_paired_thor_dirs_kwargs

    # TODO maybe factor this to / use existing stuff in hong2p in place of this
    makedirs(analysis_intermediates_root)

    main_start_s = time.time()

    if not parallel:
        # `list` call is just so `starmap` actually evaluates the fn on its input.
        # `starmap` just returns a generator otherwise.
        list(starmap(process_experiment, keys_and_paired_dirs))

    else:
        with mp.Manager() as manager:
            print(f'Processing experiments with {os.cpu_count()} workers')

            # TODO maybe define as context manager where __enter__ returns (yields,
            # actually, right?) this + __exit__ casts back to orig types and puts those
            # values in globals()
            # TODO possible to have a manager.<type> object under __globals__, or too
            # risky? to avoid need to pass extra
            shared_state = multiprocessing_namespace_from_globals(manager)

            # TODO no way to instantiate new empty manager.lists() as values inside a
            # worker, is there? could remove some of the complexity i added if that were
            # the case. didn't initially seem so...
            _names_and_concs2analysis_dirs = manager.dict()
            for ns_and_cs in names_and_concs_tuples:
                _names_and_concs2analysis_dirs[ns_and_cs] = manager.list()

            shared_state['names_and_concs2analysis_dirs'] = \
                _names_and_concs2analysis_dirs

            # NOTE: multiprocessing can't pickle this function, but if i switched to
            # using something like pathos, which uses dill instead of pickle to
            # serialize (and can serialize closures, etc), it might work
            # https://github.com/uqfoundation/pathos
            # https://stackoverflow.com/questions/19984152
            #worker_fn = capture_stdout_and_stderr(process_experiment)

            with manager.Pool() as pool:
                ret = pool.starmap(
                    #worker_fn,
                    process_experiment,
                    [x + (shared_state,) for x in keys_and_paired_dirs]
                )

            multiprocessing_namespace_to_globals(shared_state)

            names_and_concs2analysis_dirs = {
                k: ds for k, ds in names_and_concs2analysis_dirs.items() if len(ds) > 0
            }

    total_s = time.time() - main_start_s

    def earliest_analysis_dir_date(analysis_dirs):
        return min(d.split(os.sep)[-3] for d in analysis_dirs)

    # TODO exclude non-pairgrid stuff from here (might already be by virtue of what is
    # added to `names_and_concs2analysis_dirs` ...)
    print('\nOdor pair counts for all data considered (including stuff where suite2p '
        'plots not generated for various reasons):'
    )
    for names_and_concs, analysis_dirs in sorted(names_and_concs2analysis_dirs.items(),
        key=lambda x: earliest_analysis_dir_date(x[1])):

        names_and_concs_strs = []
        for name, concs in names_and_concs:
            conc_range_str = ','.join([str(c) for c in concs])
            #print(f'{name} @ {conc_range_str}')
            names_and_concs_strs.append(f'{name} @ {conc_range_str}')

        print(f'({len(analysis_dirs)}) ' + ' mixed with '.join(names_and_concs_strs))

        # TODO maybe come up with a local-file-format (part of a YAML in the raw data
        # dir?) type indicator that (current?) suite2p output just looks bad and doesn't
        # just need labelling, to avoid spending more effort on bad data?
        # (and include that as a status here probably)

        s2p_statuses = (
            'not run',
            'ROIs need manual labelling (or output looks bad)',
            'ROIs may need merging (done if not)',
            'marked bad',
            'done',
        )
        not_done_dirs = set()
        for s2p_status in s2p_statuses:

            if s2p_status == s2p_statuses[0]:
                status_dirs = [x for x in analysis_dirs if x in s2p_not_run]
                not_done_dirs.update(status_dirs)

            elif s2p_status == s2p_statuses[1]:
                status_dirs = [
                    x for x in analysis_dirs
                    if (x in iscell_not_modified) or (x in iscell_not_selective)
                ]
                not_done_dirs.update(status_dirs)

            elif s2p_status == s2p_statuses[2]:
                status_dirs = [
                    x for x in analysis_dirs if x in no_merges
                ]
                not_done_dirs.update(status_dirs)

            elif s2p_status == s2p_statuses[3]:
                # maybe don't show these ones?
                status_dirs = [
                    x for x in analysis_dirs if x in full_bad_suite2p_analysis_dirs
                ]
                not_done_dirs.update(status_dirs)

            elif s2p_status == s2p_statuses[4]:
                status_dirs = [x for x in analysis_dirs if x not in not_done_dirs]

            else:
                assert False

            show_empty_statuses = False
            if show_empty_statuses or len(status_dirs) > 0:
                print(f' - {s2p_status} ({len(status_dirs)})')
                for analysis_dir in sorted(status_dirs):
                    short_id = shorten_path(analysis_dir)
                    print(f'   - {short_id}')

        print()
    print()

    # TODO also check that all loaded data is using same stimulus program
    # (already skipping stuff with odor pulse < 3s tho)

    print(f'Took {total_s:.0f}s\n')

    # Only actually shortens if print_full_paths=False
    def shorten_and_pprint(paths):
        if not print_full_paths:
            paths = [shorten_path(p) for p in paths]

        pprint(sorted(paths))


    def print_nonempty_path_list(name, paths, alt_msg=None):
        if len(paths) == 0:
            if alt_msg is not None:
                print(alt_msg)

            return

        print(f'{name} ({len(paths)}):')
        shorten_and_pprint(paths)
        print()


    print_nonempty_path_list(
        'failed_assigning_frames_to_odors:', failed_assigning_frames_to_odors
    )

    if do_suite2p:
        print_nonempty_path_list(
            'suite2p failed:', failed_suite2p_dirs
        )

    # TODO TODO factor so that still print summary of output even in case where we have
    # all suite2p analysis turned off
    if analyze_suite2p_outputs:
        print_nonempty_path_list(
            'suite2p needs to be run on the following data:', s2p_not_run,
            alt_msg='suite2p has been run on all currently included data'
        )

        # NOTE: only possible if suite2p for these was run outside of this pipeline, as
        # `run_suite2p` in this file currently marks all ROIs as "good" (as cells, as
        # far as suite2p is concerned) immediately after a successful suite2p run.
        print_nonempty_path_list(
            'suite2p outputs with ROI labels not modified:', iscell_not_modified
        )

        print_nonempty_path_list(
            'suite2p outputs where no ROIs were marked bad:', iscell_not_selective
        )
        print_nonempty_path_list(
            'suite2p outputs where no ROIs were merged:', no_merges
        )

    if len(odors_without_abbrev) > 0:
        print('odors_without_abbrev:')
        pprint(odors_without_abbrev)

    # TODO TODO probably print stuff in gsheet but not local and vice versa


    keys = ['date', 'fly_num', 'thorimage_id']
    def fix_metadata(df):
        """Moves recording metadata from rows to columns, for more natural handling
        """
        vals = df.index.to_frame(index=False)[keys].drop_duplicates()
        assert len(vals) == 1
        vals = vals.iloc[0]

        df = df.droplevel(keys)
        df.index.name = 'roi'

        return util.addlevel(df, keys, vals, axis='columns')


    def agg_dfs(dfs):
        df = pd.concat([fix_metadata(x) for x in dfs], axis='columns')
        return df

    trial_df = agg_dfs(ij_roi_dfs)

    mean_df = agg_dfs(ij_mean_dfs)
    sum_diff_df = agg_dfs(ij_sum_diff_dfs)
    max_diff_df = agg_dfs(ij_max_diff_dfs)

    # TODO TODO TODO cache these on a per-experiment basis so that we don't need to
    # recompute all of them to add the values for a new experiment (or at least load and
    # determine what we can skip?). at bare minimum, don't overwrite unless we are
    # processing all the data.
    trial_df.to_pickle('trial_df.p')
    mean_df.to_pickle('mean_df.p')
    sum_diff_df.to_pickle('sum_diff_df.p')
    max_diff_df.to_pickle('max_diff_df.p')

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

