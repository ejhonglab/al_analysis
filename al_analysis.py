#!/usr/bin/env python3

import argparse
import atexit
from datetime import date
import os
from os.path import join, split, exists, expanduser, islink
from pprint import pprint, pformat
from collections import defaultdict, Counter
import warnings
import time
import shutil
import traceback
import subprocess
import sys
import pickle
from pathlib import Path
import glob
from itertools import starmap
import multiprocessing as mp
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import tifffile
import yaml
import ijroi
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import colorama
from termcolor import cprint, colored
from drosolf import orns
# `from suite2p import run_s2p` now done at start of function that would run suite2p

from hong2p import util, thor, viz, olf
from hong2p import suite2p as s2p
from hong2p.suite2p import LabelsNotModifiedError, LabelsNotSelectiveError
from hong2p.util import shorten_path, shorten_stimfile_path, format_date
from hong2p.olf import format_odor, format_mix_from_strs, format_odor_list, solvent_str
from hong2p.viz import dff_latex
from hong2p.types import ExperimentOdors


colorama.init()

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.constrained_layout.w_pad'] = 1/72
plt.rcParams['figure.constrained_layout.h_pad'] = 0.5/72
plt.rcParams['figure.constrained_layout.wspace'] = 0
plt.rcParams['figure.constrained_layout.hspace'] = 0

###################################################################################
# Constants that affect behavior of `process_experiment`
###################################################################################
analysis_intermediates_root = util.analysis_intermediates_root(create=True)

# Whether to only analyze experiments sampling 2 odors at all pairwise concentrations
# (the main type of experiment for this project)
analyze_pairgrids_only = False

# If there are multiple experiments with the same odors, only the data from the most
# recent concentrations for those odors will be analyzed.
final_concentrations_only = True

analyze_reverse_order = True

# NOTE: not actually a constant now. set True in main if CLI flag set to *only* analyze
# glomeruli diagnostics
# Will be set False if analyze_pairgrids_only=True
analyze_glomeruli_diagnostics = True

# Whether to analyze any single plane data that is found under the enumerated
# directories.
analyze_2d_tests = False

# Whether to run the suite2p pipeline (generates outputs among raw data, in 'suite2p'
# subdirectories)
do_suite2p = False

# Will just skip if already exists if False
overwrite_suite2p = False

# TODO change back to False. just for testing while refactoring to share code w/ ijroi
# stuff
#analyze_suite2p_outputs = True
analyze_suite2p_outputs = False

# TODO TODO TODO return to True
analyze_ijrois = True

# TODO TODO change to using seconds and rounding to nearest[higher/lower
# multiple?] from there
# TODO some quantitative check this is ~optimal?
# Note this is w/ volumetric sampling of ~1Hz.
# 4 seems to produce similar outputs to 2, though slightly dimmer in most cases.
# 3 is not noticeably dimmer than 2, so since it's averaging a little more data, I'll
# use that one.
n_response_volumes_for_trial_mean = 3

n_response_volumes_in_fname = False

# Since some of the pilot experiments had 6 planes (but top 5 should be the same as in
# experiments w/ only 5 total), and that last plane largely doesn't have anything
# measurably happening. All steps should have been 12um, so picking top n will yield a
# consistent total height of the volume.
n_top_z_to_analyze = 5

ignore_bounding_frame_cache = False

# Also required if do_suite2p is True, but that is always going to be False the only
# time this currently is (when is_acquisition_host is True).
convert_raw_to_tiff = True

# If False, will not write any TIFFs (other than raw.tif, which will always only get
# written if it doesn't already exist), including dF/F TIFF.
write_processed_tiffs = True
want_dff_tiff = True

want_dff_tiff = want_dff_tiff and write_processed_tiffs

links_to_input_dirs = True

# TODO shorten any remaining absolute paths if this is True, so we can diff outputs
# across installs w/ data in diff paths
print_full_paths = False

save_figs = True
# TODO TODO TODO fix png in case it doesn't exist before running w/ -c flag
plot_fmt = os.environ.get('plot_fmt', 'png')

cmap = 'plasma'
diverging_cmap = 'RdBu_r'
# TODO could try TwoSlopeNorm, but would probably want to define bounds per fly (or else
# compute in another pass / plot these after aggregating?)
diverging_cmap_kwargs = dict(cmap=diverging_cmap,
    # TODO delete kwargs / determine from data in each case (and why does it
    # seem fixed to [-1, 1] without this?)
    # TODO TODO TODO am i understanding this correctly? (and was default range really
    # [-1, 1], and not the data range, with no kwargs to this (or was it transforming
    # data and was that the range of the transformed data???)?
    norm=colors.CenteredNorm(halfrange=2.0),
)

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

pair_directories_root = join(plot_fmt, 'pairs')

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
    'ethanol': 'EtOH',
    'isoamyl alcohol': 'IAol',
    'isoamyl acetate': 'IAA',
    'valeric acid': 'VA',
    'kiwi approx.': '~kiwi',
}

if analyze_pairgrids_only:
    analyze_glomeruli_diagnostics = False

frame_assign_fail_prefix = 'assign_frames'
suite2p_fail_prefix = 'suite2p'

spatial_dims = ['z', 'y', 'x']

checks = True

# Changed as a globals in main (exposed as command line arguments)
ignore_existing = False
# TODO probably make another category or two for data marked as failed (in the breakdown
# of data by pairs * concs at the end) (if i don't refactor completely...)
retry_previously_failed = False
analyze_glomeruli_diagnostics_only = False
print_skipped = False

is_acquisition_host = util.is_acquisition_host()
if is_acquisition_host:
    analyze_ijrois = False
    final_concentrations_only = False
    do_suite2p = False
    analyze_suite2p_outputs = False
    write_processed_tiffs = False
    want_dff_tiff = False
    # Non-admin users don't have permission to make (the equivlant of?) symbolic links
    # in Windows, and I'm not sure what all the ramifications of enabling that would
    # be.
    links_to_input_dirs = False
    convert_raw_to_tiff = False

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

response_volumes_list = []

s2p_trial_dfs = []
# TODO TODO just agg (+ cache) this in process_experiment, and calculate all others from
# this after the main loop over experiments
# TODO also agg + cache full traces (maybe just use CSV? also for above)
# TODO rename to ij_trialstat_dfs or something
ij_trial_dfs = []

# Using dict rather than defaultdict(list) so handling is more consistent in case when
# multiprocessing DictProxy overrides this.
names_and_concs2analysis_dirs = dict()

###################################################################################
# Modified inside `run_suite2p`
###################################################################################
failed_suite2p_dirs = []

###################################################################################
# Modified inside `suite2p_traces`
###################################################################################
s2p_not_run = []
iscell_not_modified = []
iscell_not_selective = []
no_merges = []

###################################################################################

dirs_with_ijrois = []
# TODO check if any have original ROI name format (and what is it for a 3D TIFF?)
#dirs_with_ijrois_needing_merge = []


def formatwarning_msg_only(msg, category, *args, **kwargs):
    """Format warning without line/lineno (which are often not the relevant line)
    """
    return colored(f'{category.__name__}: {msg}\n', 'yellow')

warnings.formatwarning = formatwarning_msg_only


def warn(msg):
    warnings.warn(msg)


# TODO replace similar fn (if still exists?) already in hong2p? or use the hong2p one?
# (just want to prefer the "fast" data root)
def get_analysis_dir(date, fly_num, thorimage_basedir):
    return join(analysis_intermediates_root,
        util.get_fly_dir(date, fly_num), thorimage_basedir
    )


def get_plot_dir(experiment_id, relative=False):
    plot_dir = util.to_filename(experiment_id, period=False)
    if not relative:
        plot_dir = join(plot_fmt, plot_dir)

    return plot_dir


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
    # TODO shortcircuit to returning if we already made it this run, to avoid the checks
    # on subsequent calls? they probably aren't a big deal though...
    os.makedirs(d, exist_ok=True)
    dirs_to_delete_if_empty.append(d)


def delete_if_empty(d):
    """Delete directory if empty, do nothing otherwise.
    """
    # TODO don't we still want to delete any broken links / links to empty dirs?
    if not exists(d) or islink(d):
        return

    if not any(os.scandir(d)):
        os.rmdir(d)


def delete_empty_dirs():
    """Deletes empty directories in `dirs_to_delete_if_empty`
    """
    for d in dirs_to_delete_if_empty:
        delete_if_empty(d)


# TODO probably need a recursive solution combining deletion of empty symlinks and
# directories to cleanup all hierarchies that could be created w/ symlink and makedirs
links_created = []
def symlink(target, link, relative=True):
    """Create symlink link pointing to target, doing nothing if link exists.

    Also registers `link` for deletion at end if what it points to no
    """
    # TODO TODO err if link exists and was created *in this same run* (indicating trying
    # to point to multiple different outputs from the same link; a bug)

    # Will slightly simplify cleanup logic by mostly ensuring broken links only come
    # from deleted directories.
    if not exists(target):
        raise FileNotFoundError

    try:
        if relative:
            link_dir = link if os.path.isdir(link) else os.path.dirname(link)
            os.symlink(os.path.relpath(target, link_dir), link)
        else:
            os.symlink(os.path.abspath(target), link)

    except FileExistsError:
        return

    links_created.append(link)


def delete_link_if_target_missing(link):
    if not islink(link):
        # TODO what caused this? some race condition between two runs?
        #raise IOError(f'input {link} was not a link')
        warn(f'input {link} was not a link. doing nothing with it.')
        return

    if not exists(link):
        # Will fail if link is a true directory, but work if it's a link to a directory
        os.remove(link)


def delete_broken_links():
    for x in links_created:
        delete_link_if_target_missing(x)


def cleanup_created_dirs_and_links():
    """Removes created directories/links that are empty/broken.
    """
    delete_empty_dirs()
    delete_broken_links()

    if links_to_input_dirs:
        # Also want to delete directories that now *just* contain a single link
        # ('thorimage') to raw data directory.
        for d in dirs_to_delete_if_empty:
            # May have already been deleted
            if not exists(d):
                continue

            if os.listdir(d) == ['thorimage']:
                ti_link_path = join(d, 'thorimage')
                if islink(ti_link_path):
                    os.remove(ti_link_path)
                    # Directory should now be empty (not atomic, but no multiprocessing
                    # when this is called)
                    os.rmdir(d)


FAIL_INDICATOR_PREFIX = 'FAILING_'

# TODO maybe take error object / traceback and save stack trace actually?
def make_fail_indicator_file(analysis_dir, suffix, err=None):
    """Makes empty file (e.g. FAILING_suite2p) in analysis_dir, to mark step as failing
    """
    path = Path(join(analysis_dir, f'{FAIL_INDICATOR_PREFIX}{suffix}'))
    if err is None:
        path.touch()
    else:
        err_str = ''.join(traceback.format_exception(type(err), err, err.__traceback__))
        path.write_text(err_str)


def _list_fail_indicators(analysis_dir):
    return glob.glob(join(analysis_dir, FAIL_INDICATOR_PREFIX + '*'))


def last_fail_suffixes(analysis_dir):
    """Returns if an analysis_dir has any fail indicators and list of their suffixes
    """
    suffixes = [
        split(x)[1][len(FAIL_INDICATOR_PREFIX):]
        for x in _list_fail_indicators(analysis_dir)
    ]
    if len(suffixes) == 0:
        return False, None
    else:
        return True, suffixes


def clear_fail_indicators(analysis_dir):
    """Deletes any fail indicator files in analysis_dir
    """
    for f in _list_fail_indicators(analysis_dir):
        os.remove(f)


# default = 'netcdf4'
# need to `pip install h5netcdf` for this
#netcdf_engine = 'h5netcdf'
def load_dataarray(fname):
    """Loads xarray object and restores odor MultiIndex (['odor1','odor2','repeat'])
    """
    # NOTE: no longer using netcdf as it's had an obtuse error when trying to use it tot
    # serialize across-fly data:
    # "ValueError: could not broadcast input array from shape (2709,21) into shape
    # (2709,31)"
    # (related to odor index, it seems, as 2709 is the length of that)

    # TODO see if other drivers for loading are faster
    #arr = xr.load_dataarray(fname, engine=netcdf_engine)
    #return arr.set_index({'odor': ['odor1', 'odor2', 'repeat']})

    with open(fname, 'rb') as f:
        return pickle.load(f)


def write_dataarray(arr, fname) -> None:
    """Writes xarray object with odor MultiIndex (w/ levels ['odor1','odor2','repeat'])
    """
    # Only doing reset_index so we can serialize the DataArray via netCDF (recommended
    # format in xarray docs). If odor MultiIndex is left in, it causes an error which
    # says to reset_index() and references: https://github.com/pydata/xarray/issues/1077
    #arr.reset_index('odor').to_netcdf(fname, engine=netcdf_engine)
    # TODO delete eventually
    #assert arr.identical(load_dataarray(fname, engine=netcdf_engine))
    #

    with open(fname, 'wb') as f:
        # "use the highest protocol (-1) because it is way faster than the default text
        # based pickle format"
        pickle.dump(arr, f, protocol=-1)


# TODO did i already implement this logic somewhere in this file? use this code if so
def n_odors_per_trial(odor_lists: ExperimentOdors):
    """
    Assumes same number of odors on each trial
    """
    len_set = {len(x) for x in odor_lists}
    assert len(len_set) == 1
    return len_set.pop()


def odor_lists2names_and_conc_ranges(odor_lists: ExperimentOdors):
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


def is_pairgrid(odor_lists: ExperimentOdors):
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


def is_reverse_order(odor_lists: ExperimentOdors):
    o1_list = [o1 for o1, _ in odor_lists if o1['log10_conc'] is not None]

    def get_conc(o):
        return o['log10_conc']

    return get_conc(o1_list[0]) > get_conc(o1_list[-1])


def odor_strs2single_odor_name(index, name_conc_delim='@'):
    # TODO factor LHS into hong2p.olf.parse_odor_name fn or something
    odors = {x.split(name_conc_delim)[0].strip() for x in index}
    odors = {x for x in odors if x != solvent_str}
    assert len(odors) == 1
    return odors.pop()


# TODO indicate subclass of pd.Index (Type[pd.Index]?) as return type
def odor_lists_to_multiindex(odor_lists: ExperimentOdors, **format_odor_kwargs):

    unique_lens = {len(x) for x in odor_lists}
    if len(unique_lens) != 1:
        raise NotImplementedError

    # This one would be more straight forward to relax than the above one
    if unique_lens == {2}:
        pairs = True

    elif unique_lens == {1}:
        pairs = False
    else:
        raise NotImplementedError

    odor1_str_list = []
    odor2_str_list = []

    odor_mix_counts = defaultdict(int)
    odor_mix_repeats = []

    for odor_list in odor_lists:

        if pairs:
            odor1, odor2 = odor_list
        else:
            odor1 = odor_list[0]
            assert len(odor_list) == 1
            # format_odor -> format_mix_from_strs should treat this as:
            # 'solvent' (hong2p.olf.solvent_str) -> not being shown as part of mix
            # TODO is this actually reached? i'm not seeing it in some of the
            # ij_trial_dfs at end of main
            odor2 = {'name': 'no_second_odor', 'log10_conc': None}

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

        # An odor name wasn't in a consistent position across the odor lists somewhere
        # (probably would only happen for a non-pair experiment).
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
def compute_trial_stats(traces, bounding_frames,
    odor_order_with_repeats: Optional[ExperimentOdors] = None,
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

    solvent_elements = sort_key == solvent_str
    conc_delimiter = '@'
    assert all([conc_delimiter in x for x in sort_key[~ solvent_elements]])

    # TODO share this + fn wrapping this that returns sort key w/ other place that
    # is using sorted(..., key=...)
    def parse_log10_conc(odor_str):
        assert conc_delimiter in odor_str
        parts = odor_str.split(conc_delimiter)
        assert len(parts) == 2
        return float(parts[1].strip())

    if not all(solvent_elements):
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


# TODO TODO add some kind of lookup for odor panels (might just need to get the set of
# all (odor name, odor concentrations) used in experiment and compare that.  -> force
# consistent order for things like kiwi.yaml/control1.yaml experiments (anything not
# pair that we actually wanna see plots for actually. probably just don't wanna sort
# glomeruli diagnostics) (only really relevant if i actually start randomizing order in
# those experiments... for now, could just not sort)
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


def get_panel(date: pd.Timestamp, thorimage_id: str, odor_list: ExperimentOdors
    ) -> Optional[str]:
    """Return None or str describing the odors used in the experiment.

    Some panels are split across multiple types of experiments. For example, 'kiwi' is
    the panel for both experiments collecting mainly the components alone as well as
    those collecting just mixtures of the most intense 2 components (run via
    `olf kiwi.yaml` and `olf kiwi_ea_eb_only.yaml`, respectively).
    """
    if 'diag' in thorimage_id:
        return 'glomeruli_diagnostics'

    elif 'kiwi' in thorimage_id:
        return 'kiwi'

    elif 'control' in thorimage_id:
        return 'control'

    # TODO TODO handle old pair stuff too (panel='<name1>+<name2>' or something) + maybe
    # use get_panel to replace the old name1 + name2 means grouping by effectively panel

    else:
        return None


def ijroi_plot_dir(plot_dir):
    return join(plot_dir, 'ijroi')


def suite2p_plot_dir(plot_dir):
    return join(plot_dir, 'suite2p_roi')


# TODO TODO maybe i should check for all of a minimum set of files, or just the mtime on
# the df caches, in case a partial run erroneously prevents future runs
def ij_last_analysis_time(plot_dir):
    roi_plot_dir = ijroi_plot_dir(plot_dir)
    return util.most_recent_contained_file_mtime(roi_plot_dir)


def suite2p_outputs_mtime(analysis_dir):
    combined_dir = s2p.get_suite2p_combined_dir(analysis_dir)
    return util.most_recent_contained_file_mtime(combined_dir)


def suite2p_last_analysis_time(plot_dir):
    roi_plot_dir = suite2p_plot_dir(plot_dir)
    return util.most_recent_contained_file_mtime(roi_plot_dir)


def names2fname_prefix(name1, name2):
    return util.to_filename(f'{name1}_{name2}'.lower(), period=False)


def get_pair_dir(name1, name2):
    return join(pair_directories_root, names2fname_prefix(name1, name2))


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

    trial_df = single_roi_series.unstack(level=0)

    trial_df = sort_odor_indices(trial_df)

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
        fig, _ = viz.matshow(trial_df.droplevel('repeat'), group_ticklabels=True,
            **common_matshow_kwargs
        )
    else:
        # 'odor2' is the one on the row axis, as one level alongside 'repeat'
        # TODO not sure why sort=False seems to be ignored... bug?
        mean_df = sort_odor_indices(trial_df.groupby('odor2', sort=False).mean())

        fig, _ = viz.matshow(mean_df, **common_matshow_kwargs)

    return fig


def plot_all_roi_mean_responses(trial_df, title=None, roi_sortkeys=None, roi_rows=True,
    sort_odors=True, **kwargs):
    """Plots odor x ROI data displayed with odors as columns and ROI means as rows.

    Args:
        trial_df: DataFrame with ['odor1', 'odor2', 'repeat'] index names and a column
            for each ROI

        roi_sortkeys: sequence of same length as trial_df, used to order ROIs

        roi_rows: (default=True) if True, matrix will be transposed relative to input,
            with ROIs as rows and odors as columns

    """
    for c in ['odor1', 'odor2', 'repeat']:
        assert c in trial_df.index.names

    # This will throw away any metadta in multiindex levels other than these two
    # (so can't just add metadata once at beginning and have it propate through here,
    # without extra work at least)
    mean_df = trial_df.groupby(['odor1', 'odor2'], sort=False).mean()

    if sort_odors:
        mean_df = sort_odor_indices(mean_df)

    if roi_sortkeys is not None:
        assert len(roi_sortkeys) == len(trial_df.columns)

        roi_sortkey_dict = dict(zip(trial_df.columns, roi_sortkeys))

        def roi_sortkey_fn(index):
            return [roi_sortkey_dict[x] for x in index]

        mean_df.sort_index(key=roi_sortkey_fn, axis='columns', inplace=True)

    if roi_rows:
        xticklabels = format_mix_from_strs
        yticklabels = True
        mean_df = mean_df.T
    else:
        xticklabels = True
        yticklabels = format_mix_from_strs

    # TODO maybe put lines between levels of sortkey if int (e.g. 'iplane')
    # (and also show on plot as second label above/below roi labels?)

    fig, _ = viz.matshow(mean_df, title=title, xticklabels=xticklabels,
        yticklabels=yticklabels, cmap=cmap, **kwargs
    )

    return fig, mean_df


def suite2p_traces(analysis_dir):
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

        print(f'NOT generating suite2p traces because {e}')
        return None

    if len(merges) == 0:
        no_merges.append(analysis_dir)

    # TODO TODO TODO also try passing input of (another call to?) compute_trial_stats to
    # remerge_suite2p_merged, so it's actually using local dF/F rather than max-min of
    # the entire raw trace (or maybe just the dF?)

    # TODO TODO [option to] use non-weighted footprints (as well as footprints that have
    # had the binary closing operation applied before uniform weighting)

    # TODO TODO modify so that that merging w/in plane works (before finding best
    # plane) + test
    traces, rois = s2p.remerge_suite2p_merged(traces, roi_stats, ops, merges,
        verbose=False
    )
    z_indices = np.array([roi_stats[x]['iplane'] for x in rois.roi.values])

    return traces, rois, z_indices, roi_stats


# TODO change to just take thorimage_dir after refactoring probably
#def ij_traces(thorimage_dir, analysis_dir, movie):
def ij_traces(analysis_dir, movie):
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
    # do need to add this again it seems (and i think one above *might* ahve been used
    # inside `rois2best_planes_only`
    traces.columns.name = 'roi'

    # TODO can i just use rois.roi_z.values?
    z_indices = masks.roi_z[masks.roi_num.isin(roi_nums)].values

    # TODO maybe just return rois and have z index information there in a way consistent
    # w/ output from corresponding suite2p fn?
    return traces, rois, z_indices


def trace_plots(roi_plot_dir, trial_df, z_indices, main_plot_title, odor_lists, *,
    roi_stats=None, show_suite2p_rois=False):

    if show_suite2p_rois and roi_stats is None:
        raise ValueError('must pass roi_stats if show_suite2p_rois')

    is_pair = is_pairgrid(odor_lists)

    makedirs(roi_plot_dir)

    fig, mean_df = plot_all_roi_mean_responses(trial_df, roi_sortkeys=z_indices,
        sort_odors=is_pair, title=main_plot_title, cbar_label=trial_stat_cbar_title,
        cbar_shrink=0.4
    )
    savefig(fig, roi_plot_dir, 'all_rois_by_z')

    if not is_pair:
        return

    for roi in trial_df.columns:
        if show_suite2p_rois:
            fig, axs = plt.subplots(nrows=2, ncols=1)
            ax = axs[0]
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        roi1_series = trial_df.loc[:, roi]
        plot_roi_stats_odorpair_grid(roi1_series, cbar_label=trial_stat_cbar_title,
            cbar_shrink=0.4, ax=ax
        )

        # TODO (assuming colors are saved / randomization is seeded and easily
        # reproducible in suite2p) copy suite2p coloring for plotting ROIs (or at least
        # make my own color scheme consistent across all plots w/in experiment)
        # TODO TODO separate unified ROI plot (like combined view, but maybe all in
        # one row for consistency with my other plots) with same color scheme
        # TODO TODO have scale in single ROI plots and ~"combined" view be the same
        # (each plot pixel same physical size)
        if show_suite2p_rois:
            roi_stat = roi_stats[roi]
            s2p.plot_roi(roi_stat, ops, ax=axs[1])

        savefig(fig, roi_plot_dir, str(roi))


# TODO kwarg to allow passing trial stat fn in that includes frame rate info as closure,
# for picking frames in a certain time window after onset and computing mean?
# TODO TODO TODO to the extent that i need to convert suite2p rois to my own and do my
# own trace extraction, maybe just modify my fork of suite2p to save sufficient
# information in combined view stat.npy to invert the tiling? unless i really can find a
# reliable way to invert that step...
def suite2p_trace_plots(analysis_dir, bounding_frames, odor_lists, plot_dir):

    outputs = suite2p_traces(analysis_dir)
    if outputs is None:
        return
    traces, rois, z_indices, roi_stats = outputs

    trial_df = compute_trial_stats(traces, bounding_frames, odor_lists)

    # TODO also plot roi / outline of roi on corresponding [mean?] plane / maybe with
    # other planes for context?

    roi_plot_dir = suite2p_plot_dir(plot_dir)
    title = 'Suite2p ROIs\nOrdered by Z plane'

    trace_plots(roi_plot_dir, trial_df, z_indices, title, odor_lists,
        roi_stats=roi_stats, show_suite2p_rois=False
    )
    print('generated plots based on suite2p traces')

    return trial_df


def ij_trace_plots(analysis_dir, bounding_frames, odor_lists, movie, plot_dir):

    traces, rois, z_indices = ij_traces(analysis_dir, movie)

    trial_df = compute_trial_stats(traces, bounding_frames, odor_lists)

    roi_plot_dir = ijroi_plot_dir(plot_dir)
    title = 'ImageJ ROIs\nOrdered by Z plane\n*possibly [over/under]merged'

    trace_plots(roi_plot_dir, trial_df, z_indices, title, odor_lists)

    print('generated plots based on traces from ImageJ ROIs')

    return trial_df


# TODO maybe refactor (part of?) this to hong2p.suite2p
def run_suite2p(thorimage_dir, analysis_dir, overwrite=False):
    """
    Depends on util.thor2tiff already being run to create TIFF in analysis_dir
    """
    from suite2p import run_s2p

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
        # TODO refactor write_tiff calls into local version that warns if basename isn't
        # in a module-level tuple that gets appended to thor stuff above
        # NOTE: you MUST include any additional TIFFs you create in the analysis
        # directories here so that suite2p doesn't try to run on them (just the name,
        # not the directories containing them)
        'dff.tif',
        'max_trialmean_dff.tif',
    )

    db = {
        #'data_path': [thorimage_dir],
        # TODO update suite2p to take substrs / globs to ignore input files on
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
        cprint(traceback.format_exc(), 'red', file=sys.stderr)
        failed_suite2p_dirs.append(analysis_dir)

        make_fail_indicator_file(analysis_dir, suite2p_fail_prefix, err)

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
# TODO TODO figure out why this seems to be using up a lot of memory now. is something
# not being cleaned up properly? or maybe i just had very low memory available and it
# wasn't abnormal?
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

    # NOTE: This should be called at least once before process_experiment returns, and
    # the first call must be *before* any other calls that make output (prints,
    # warnings, errors).
    # If function returns after yaml_path is defined, it should be called at least once
    # with yaml_path as an argument.
    _printed_thor_dirs = False
    _printed_yaml_path = False
    def print_inputs_once(yaml_path=None):
        nonlocal _printed_thor_dirs
        nonlocal _printed_yaml_path

        if not _printed_thor_dirs:
            util.print_thor_paths(thorimage_dir, thorsync_dir,
                print_full_paths=print_full_paths
            )
            _printed_thor_dirs = True

        if yaml_path is not None and not _printed_yaml_path:
            print('yaml_path:', shorten_stimfile_path(yaml_path))
            _printed_yaml_path = True

    def print_skip(msg, yaml_path=None, *, color=None, file=None):
        if not print_skipped:
            return

        print_inputs_once(yaml_path)

        if color is not None:
            msg = colored(msg, color)

        print(msg, file=file)
        print()

    _did_nonskipped_prints = False
    _to_print_if_not_skipped = []
    def print_if_not_skipped(x):
        assert not _did_nonskipped_prints, ('after '
            'do_nonskipped_experiment_prints_and_warns, use regular print/warn'
        )
        if print_skipped:
            print(x)
        else:
            _to_print_if_not_skipped.append(x)

    def warn_if_not_skipped(x):
        assert not _did_nonskipped_prints, ('after '
            'do_nonskipped_experiment_prints_and_warns, use regular print/warn'
        )
        if print_skipped:
            warn(x)
        else:
            _to_print_if_not_skipped.append(UserWarning(x))

    def do_nonskipped_experiment_prints_and_warns(yaml_path):
        nonlocal _did_nonskipped_prints
        # We should only be doing this once
        assert not _did_nonskipped_prints

        # Because in this case we should be printing / warning everything ASAP.
        if print_skipped:
            return

        print_inputs_once(yaml_path)
        for x in _to_print_if_not_skipped:
            if type(x) is UserWarning:
                warn(x)
            else:
                print(x)

        _did_nonskipped_prints = True

    # If we are printing skipped, we might as well print each of the input paths as soon
    # as possible, to make debugging easier (so relevant paths will be more likely to
    # appear before traceback -> termination).
    if print_skipped:
        print_inputs_once()

    if 'diag' in thorimage_dir:
        if not analyze_glomeruli_diagnostics:
            print_skip('skipping because experiment is just glomeruli diagnostics')
            return

        is_glomeruli_diagnostics = True
    else:
        if analyze_glomeruli_diagnostics_only:
            print('skipping because experiment is NOT glomeruli diagnostics\n')
            return

        is_glomeruli_diagnostics = False

    exp_start = time.time()

    single_plane_fps, xy, z, c, n_flyback, _, xml = thor.load_thorimage_metadata(
        thorimage_dir, return_xml=True
    )

    if not analyze_2d_tests and z == 1:
        print_skip('skipping because experiment is single plane')
        return

    experiment_id = shorten_path(thorimage_dir)
    experiment_basedir = get_plot_dir(experiment_id, relative=True)

    # Created below after we decide whether to skip a given experiment based on the
    # experiment type, etc.
    # TODO rename to experiment_plot_dir or something
    plot_dir = join(plot_fmt, experiment_basedir)

    def suptitle(title, fig=None):
        if title is None:
            return

        if fig is None:
            fig = plt.gcf()

        fig.suptitle(f'{experiment_id}\n{title}')

    def exp_savefig(fig, desc, **kwargs):
        return savefig(fig, plot_dir, desc, **kwargs)

    thorimage_basename = split(thorimage_dir)[1]
    analysis_dir = get_analysis_dir(date, fly_num, thorimage_basename)
    makedirs(analysis_dir)
    makedirs(plot_dir)

    if links_to_input_dirs:
        symlink(thorimage_dir, join(plot_dir, 'thorimage'), relative=False)
        symlink(analysis_dir, join(plot_dir, 'analysis'), relative=False)

    # TODO try to speed up? or just move my stimfiles to local storage?
    # currently takeing ~0.04s per call, -> 3-4 seconds @ ~80ish calls
    # (same issue w/ yaml.safe_load on bounding frames though, so maybe it's not
    # storage? or maybe it's partially a matter of seek time? should be ~5-10ms tho...)
    yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(xml)

    # Again, in this case, we might as well print all input paths ASAP.
    if print_skipped:
        print_inputs_once(yaml_path)

    pulse_s = float(int(yaml_data['settings']['timing']['pulse_us']) / 1e6)
    if pulse_s < 3:
        print_skip(f'skipping because odor pulses were {pulse_s} (<3s) long (old)',
            yaml_path
        )
        return

    is_pair = is_pairgrid(odor_lists)
    if is_pair:
        # So that we can count how many flies we have for each odor pair (and
        # concentration range, in case we varied that at one point)
        names_and_concs_tuple = odor_lists2names_and_conc_ranges(odor_lists)

        if final_concentrations_only:
            names, curr_concs = separate_names_and_concs_tuples(
                names_and_concs_tuple
            )

            if (names in names2final_concs and
                names2final_concs[names] != curr_concs):

                print_skip('skipping because not using final concentrations for '
                    f'{names}', yaml_path
                )
                return

        if not analyze_reverse_order and is_reverse_order(odor_lists):
            print_skip('skipping because a reverse order experiment', yaml_path)
            return

        # In case where this is a DictProxy, these empty lists (ListProxy in that case)
        # should all have been added before parallel starmap (outside this fn).
        if names_and_concs_tuple not in names_and_concs2analysis_dirs:
            names_and_concs2analysis_dirs[names_and_concs_tuple] = []

        names_and_concs2analysis_dirs[names_and_concs_tuple].append(analysis_dir)

        (name1, _), (name2, _) = names_and_concs_tuple
        name1 = odor2abbrev.get(name1, name1)
        name2 = odor2abbrev.get(name2, name2)

        if not is_acquisition_host:
            # TODO TODO do this for all panels/experiment types (or (panel, is_pair)
            # combinations...)
            pair_dir = get_pair_dir(name1, name2)
            makedirs(pair_dir)
            symlink(plot_dir, join(pair_dir, experiment_basedir), relative=True)
    else:
        if analyze_pairgrids_only:
            print_skip('skipping because not a pair grid experiment', yaml_path)
            return

    # Checking here even though `seen_stimulus_yamls2thorimage_dirs` was pre-computed
    # elsewhere because we don't necessarily want to err if this would only get
    # triggered for stuff that would get skipped in this function.
    if not is_glomeruli_diagnostics and (yaml_path in seen_stimulus_yamls2thorimage_dirs
        and seen_stimulus_yamls2thorimage_dirs[yaml_path] != [thorimage_dir]):

        short_yaml_path = shorten_stimfile_path(yaml_path)

        err_msg = (f'stimulus yaml {short_yaml_path} seen in:\n'
            f'{pformat(seen_stimulus_yamls2thorimage_dirs[yaml_path])}'
        )
        warn_if_not_skipped(err_msg)

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

    # TODO TODO refactor (probably to one skip-check gaurding each corresponding
    # step-that-could-fail, so that we don't e.g. skip dF/F calculation from movie
    # because suite2p had failed, perhaps even if we aren't requesting suite2p analysis)
    # TODO also don't want to clear fail indicators for things that we aren't requesting
    # (e.g. so we can still see why suite2p failed without having to re-run everything,
    # requesting suite2p again)
    if retry_previously_failed:
        clear_fail_indicators(analysis_dir)
    else:
        has_failed, suffixes = last_fail_suffixes(analysis_dir)
        if has_failed:
            if frame_assign_fail_prefix in suffixes:
                failed_assigning_frames_to_odors.append(thorimage_dir)

            if suite2p_fail_prefix in suffixes:
                failed_suite2p_dirs.append(analysis_dir)

            suffixes_str = ' AND '.join(suffixes)

            print_skip(f'skipping because previously failed {suffixes_str}', yaml_path,
                color='red'
            )
            return

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
            print_skip(traceback.format_exc(), yaml_path, color='red', file=sys.stderr)

            failed_assigning_frames_to_odors.append(thorimage_dir)

            make_fail_indicator_file(analysis_dir, frame_assign_fail_prefix, err)
            return
    else:
        with open(bounding_frame_yaml_cache, 'r') as f:
            bounding_frames = yaml.safe_load(f)

    # (loading the HDF5 should be the main time cost in the above fn)
    load_hdf5_s = time.time() - before

    if convert_raw_to_tiff:
        # TODO TODO TODO share loading (passing in movie) when possible (or returning
        # from here and using if needed. i forget the order)
        #
        # TODO TODO TODO integrate w/ as-yet-unmerged-hong2p flipping code (via metadata
        # `flip_lr: <bool>` flag in to this (+ elsewhere too potentially).
        #
        # This will create a TIFF <analysis_dir>/raw.tif, if it doesn't already exist
        util.thor2tiff(thorimage_dir, output_dir=analysis_dir, if_exists='ignore',
            verbose=False
        )

    if do_suite2p:
        run_suite2p(thorimage_dir, analysis_dir, overwrite=overwrite_suite2p)

    # Not including concentrations in metadata to add, b/c I generally run this script
    # skipping all but final concentrations (process_experiment returns None for all
    # non-final concentrations)
    # (already defined in is_pair case)
    if not is_pair:
        name1 = np.nan
        name2 = np.nan

    new_col_level_names = ['date', 'fly_num', 'thorimage_id']
    new_col_level_values = [date, fly_num, thorimage_basename]

    # TODO is this still how i want to handle things for the return to natural mix
    # experiments (where some but not all data we want to analyze are pairs)?
    new_row_level_names = ['name1', 'name2']
    new_row_level_values = [name1, name2]

    def add_metadata(df):
        df = util.addlevel(df, new_row_level_names, new_row_level_values)
        return util.addlevel(df, new_col_level_names, new_col_level_values,
            axis='columns'
        )

    if analyze_suite2p_outputs:
        if not any([b in thorimage_dir for b in bad_suite2p_analysis_dirs]):
            s2p_trial_df_cache_fname = join(analysis_dir, 'suite2p_trial_df_cache.p')

            s2p_analysis_current = (
                suite2p_outputs_mtime(analysis_dir) <
                suite2p_last_analysis_time(plot_dir)
            )

            if not exists(s2p_trial_df_cache_fname):
                s2p_analysis_current = False

            if not ignore_existing and s2p_analysis_current:
                print_if_not_skipped('suite2p outputs unchanged since last analysis')
                s2p_trial_df = pd.read_pickle(s2p_trial_df_cache_fname)
            else:
                s2p_trial_df = suite2p_trace_plots(analysis_dir, bounding_frames,
                    odor_lists, plot_dir
                )
                s2p_trial_df = add_metadata(s2p_trial_df)
                s2p_trial_df.to_pickle(s2p_trial_df_cache_fname)

            s2p_trial_dfs.append(s2p_trial_df)
        else:
            full_bad_suite2p_analysis_dirs.append(analysis_dir)
            print_if_not_skipped('not making suite2p plots because outputs marked bad')

    # Assuming that if analysis_dir has *any* plots directly inside of it, it has all of
    # what we want (including any outputs that would go in analysis_dir).
    do_nonroi_analysis = (
        ignore_existing or len(glob.glob(join(plot_dir, f'*.{plot_fmt}'))) == 0
    )

    do_ij_analysis = False
    if analyze_ijrois:

        if util.has_ijrois(analysis_dir):
            dirs_with_ijrois.append(analysis_dir)

            ij_trial_df_cache_fname = join(analysis_dir, 'ij_trial_df_cache.p')

            # TODO TODO TODO fix (or maybe just return appropriate inf value from
            # ijroi_mtime / other util mtime fn used by the RHS fn?)
            # (what was wrong with this?)
            try:
                ij_analysis_current = (
                    util.ijroi_mtime(analysis_dir) < ij_last_analysis_time(plot_dir)
                )
            # (comparing None to float)
            except TypeError:
                ij_analysis_current = False

           # TODO delete after generating them all? slightly more robust to interrupted
            # runs if i leave it
            if not exists(ij_trial_df_cache_fname):
                ij_analysis_current = False

            if not ignore_existing:
                if ij_analysis_current:
                    print_if_not_skipped(
                        'ImageJ ROIs unchanged since last analysis. reading cache.'
                    )
                    ij_trial_df = pd.read_pickle(ij_trial_df_cache_fname)
                    ij_trial_dfs.append(ij_trial_df)
                else:
                    print_inputs_once(yaml_path)
                    print('ImageJ ROIs were modified. re-analyzing.')

            do_ij_analysis = ignore_existing or not ij_analysis_current
        else:
            # TODO also print for other stuff we wanna analyze that doesn't have ImageJ
            # ROIs
            if is_pair:
                print_if_not_skipped('no ImageJ ROIs')

    response_volume_cache_fname = join(analysis_dir, 'trial_response_volumes.p')
    response_volume_cache_exists = exists(response_volume_cache_fname)
    if not response_volume_cache_exists:
        do_nonroi_analysis = True

    if response_volume_cache_exists and not do_nonroi_analysis:
        response_volumes_list.append(load_dataarray(response_volume_cache_fname))

    if not (do_nonroi_analysis or do_ij_analysis):
        print_skip('not loading movie because neither non-ROI nor ImageJ ROI analysis '
            'were requested', yaml_path
        )
        return

    do_nonskipped_experiment_prints_and_warns(yaml_path)

    before = time.time()

    movie = thor.read_movie(thorimage_dir)

    read_movie_s = time.time() - before

    if do_ij_analysis:
        # TODO make sure none of the stuff w/ suite2p outputs marked bad should also
        # just generally be marked bad, s.t. not run here
        ij_trial_df = ij_trace_plots(analysis_dir, bounding_frames, odor_lists, movie,
            plot_dir
        )

        # TODO populate dirs_with_ijrois_needing_merge
        #import ipdb; ipdb.set_trace()

        ij_trial_df = add_metadata(ij_trial_df)
        ij_trial_df.to_pickle(ij_trial_df_cache_fname)
        ij_trial_dfs.append(ij_trial_df)

    if not do_nonroi_analysis:
        print(f'skipping non-ROI analysis because plot dir contains {plot_fmt}\n')
        return

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
        warn(f'{thorimage_dir}: only analyzing top {n_top_z_to_analyze} '
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

    # List of length equal to the total number of trials, each element an array of shape
    # (z, y, x).
    all_trial_mean_dffs = []

    # Each element mean of the (mean) volumetric responses across the trials of a
    # particular odor, of shape (z, y, x)
    odor_mean_dff_list = []

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

        # Each element is mean-within-a-window response for one trial, of shape
        # (z, y, x)
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

            if save_dff_tiff:
                trial_dff_movie = (
                    (movie[start_frame:(end_frame + 1)] - baseline) / baseline
                )
                trial_dff_movies.append(trial_dff_movie)

            # TODO off by one at start? (still relevant?)
            mean_dff = dff[:n_response_volumes_for_trial_mean].mean(axis=0)

            # This one is for constructing an xarray of the response volume after the
            # loop over odors. Below is just for calculating mean across trials of an
            # odor.
            all_trial_mean_dffs.append(mean_dff)

            trial_mean_dffs.append(mean_dff)

            # was checking range of pixels values made sense. some are reported as max i
            # believe, and probably still are. could maybe just be real noise though.
            #print(dff.max())

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

        fname_prefix = plot_desc + '_trials'

        if n_response_volumes_in_fname:
            fname_prefix += f'_n{n_response_volumes_for_trial_mean}'

        exp_savefig(trial_heatmap_fig, fname_prefix)

        fname_prefix = plot_desc
        if n_response_volumes_in_fname:
            fname_prefix += f'_n{n_response_volumes_for_trial_mean}'

        avg_mean_dff = np.mean(trial_mean_dffs, axis=0)
        fig_path = plot_and_save_dff_depth_grid(avg_mean_dff, fname_prefix,
            title=odor_str, cbar_label=f'Mean {dff_latex}'
        )

        odor_mean_dff_list.append(avg_mean_dff)

        # TODO TODO TODO (at least until i have stuff in a standardized l/r orientation
        # (flipping one), add average image / projection on left / bottom of glomeruli
        # diag dF/F plots, to tell from looking at one plot whether glomerulus is on
        # right side)
        # TODO TODO maybe also include some quick reference to
        # previously-presented-stimulus, to check for constamination components of
        # noise?
        # TODO TODO also include a subdir of glomeruli_diagnostics that just links
        # to the corresponding directories of individual flies, to not have to sift
        # through them at the top-level that also has real experiments
        if not is_acquisition_host and target_glomerulus is not None:
            # gsheet only has labels on a per-fly basis, and those should apply to the
            # glomeruli diagnostic experiment corresponding to the same FOV as the other
            # experiments. Don't want to link any other experiments anywhere under here.
            rel_exp_dir = '/'.join(analysis_dir.split(os.sep)[-3:])
            if rel_exp_dir in unused_glomeruli_diagnostics:
                continue

            glomerulus_dir = join(across_fly_glomeruli_diags_dir,
                util.to_filename(target_glomerulus, period=False).strip('_')
            )
            makedirs(glomerulus_dir)

            label_subdir = None
            try:
                fly_diag_statuses = glomeruli_diag_status_df.loc[(date, fly_num)]

                fly_diags_labelled = fly_diag_statuses.any()
                if not fly_diags_labelled:
                    label_subdir = 'unlabelled'

            except KeyError:
                label_subdir = 'unlabelled'

            date_str = format_date(date)

            if label_subdir is None:
                try:
                    curr_diag_good = fly_diag_statuses.loc[target_glomerulus.lower()]

                except KeyError:
                    warn(f'target glomerulus {target_glomerulus} not in Google'
                        ' sheet! add column and label data. currently not linking these'
                        ' plots!'
                    )
                    continue

                if curr_diag_good:
                    label_subdir = 'good'
                else:
                    label_subdir = 'bad'
            else:
                warn('please label quality glomeruli diagnostics for fly '
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
                warn(f'{(date_str, fly_num)} has multiple glomeruli diagnostic'
                    ' experiments. add all but one to unused_glomeruli_diagnostics. '
                    'FIRST IS CURRENTLY LINKED BUT MAY NOT BE THE RELEVANT EXPERIMENT!'
                )
                continue

            symlink(fig_path, link_path, relative=True)

    # TODO maybe refactor so it doesn't need to be computed both here and in both the
    # (ij/suite2p) trace handling fns (though they currently also use odor_lists to
    # compute is_pairgrid, so might want to refactor that too)
    odor_index = odor_lists_to_multiindex(odor_lists)
    assert len(odor_index) == len(all_trial_mean_dffs)

    panel = get_panel(date, thorimage_basename, odor_lists)

    # TODO maybe factor out the add_metadata fn above to hong2p.util + also handle
    # xarray inputs there?
    # TODO TODO any reason to use attrs for these rather than additional coords?
    # either make concatenating more natural?
    metadata = {
        'panel': panel,
        'is_pair': is_pair,
        'date': date,
        'fly_num': fly_num,
        'thorimage_id': thorimage_basename,
    }
    coords = metadata.copy()
    coords['odor'] = odor_index

    # TODO use long_name attr for fly info str?
    # TODO populate units (how to specify though? pint compat?)?
    arr = xr.DataArray(all_trial_mean_dffs, dims=['odor', 'z', 'y', 'x'], coords=coords)

    # TODO probably move in constructor above if it ends up being useful to do here
    arr = arr.assign_coords({n: np.arange(dict(zip(arr.dims, arr.shape))[n])
        for n in spatial_dims
    })

    response_volumes_list.append(arr)
    write_dataarray(arr, response_volume_cache_fname)

    if save_dff_tiff:
        delta_f_over_f = np.concatenate(trial_dff_movies)

        assert delta_f_over_f.shape == movie.shape

        print(f'writing dF/F TIFF to {dff_tiff_fname}...', flush=True, end='')

        util.write_tiff(dff_tiff_fname, delta_f_over_f, strict_dtype=False)

        print(' done', flush=True)

        del delta_f_over_f, trial_dff_movies

    max_trialmean_dff = np.max(odor_mean_dff_list, axis=0)
    if write_processed_tiffs:
        max_trialmean_dff_tiff_fname = join(analysis_dir, 'max_trialmean_dff.tif')
        util.write_tiff(max_trialmean_dff_tiff_fname, max_trialmean_dff,
            strict_dtype=False, dims='ZYX'
        )
        print(f'wrote TIFF with max across mean odor dF/F volumes')

    fname_prefix = 'max_trialmean_dff'
    if n_response_volumes_in_fname:
        fname_prefix += f'_n{n_response_volumes_for_trial_mean}'

    plot_and_save_dff_depth_grid(max_trialmean_dff, fname_prefix,
        title=f'Max of trial-mean {dff_latex}', cbar_label=f'{dff_latex}',
    )

    plt.close('all')

    exp_total_s = time.time() - exp_start
    exp_processing_time_data.append((load_hdf5_s, read_movie_s, exp_total_s))

    print()

    # Just to indicate that something was analyzed. Could replace w/ some data
    # potentially, still just checking the # of None/falsey return values to get the #
    # of non-analyzed things.
    return True


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
    return 0 if odor_str == solvent_str else np.float_power(10,
        float(odor_str.split('@')[-1].strip())
    )


# TODO maybe refactor so this happens in analyze_cache, if at all (splitting fly-by-fly
# first or nah?)
def plot_diffs_from_max_and_sum(mean_df, roi_plot_dir=None, **matshow_kwargs):
    mean_df = mean_df.T

    component_df = mean_df[
        (mean_df.index.to_frame() == solvent_str).sum(axis='columns') == 1
    ]

    odor1_df = component_df[component_df.index.get_level_values('odor2') == solvent_str
        ].droplevel('odor2')

    odor2_df = component_df[component_df.index.get_level_values('odor1') == solvent_str
        ].droplevel('odor1')

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
        diff_df = diff_df.T

        # TODO less gradations on these color bars? kinda packed.
        # TODO cbar_label? or just ok to leave it in title?
        fig, _ = viz.matshow(diff_df, title=f'Component {desc} minus observed',
            #xticklabels=True, yticklabels=format_mix_from_strs, #**diverging_cmap_kwargs,
            **matshow_kwargs
        )

        if roi_plot_dir is not None:
            savefig(fig, roi_plot_dir, f'diff_{desc}')

        # TODO refactor (unroll loop?)
        if desc == 'max':
            max_diff_fig = fig
        elif desc == 'sum':
            sum_diff_fig = fig

    #return sum_diff_df, max_diff_df
    return sum_diff_fig, max_diff_fig


# TODO TODO TODO if they are fitting using a function like this, that only has as
# product of fmax*e_i, then how do they get a single fmax for computing the model
# response to a mixture? presumably the fmax fit on each component will differ?
# or do they do some subsequent algebra to get one fmax and then a new e_i from each of
# the outputs of the component fits?
# TODO i shouldn't deal w/ concs in log units or something should i?
def model_orn_singleodor_responses(concs, fmax_times_efficacy, ec50):
    # TODO check this vectorized formula works correctly
    return (concs * fmax_times_efficacy / ec50) / (1 + concs / ec50)


# TODO TODO modify to use trial_df not mean_df, and weight by standard deviation(?) like
# in the paper
def competitive_binding_model(mean_df):

    # TODO refactor to share code for odor1 and odor2 stuff
    odor1_df = mean_df.loc[:, (slice(None), solvent_str)].droplevel('odor2',
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


    odor2_df = mean_df.loc[:, (solvent_str, slice(None))].droplevel('odor1',
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

    return est_df, param_df


def pair_savefig(fig_or_sns_obj, fname_prefix, names):
    name1, name2 = names
    pair_dir = get_pair_dir(name1, name2)
    if save_figs:
        fig_or_sns_obj.savefig(join(pair_dir, f'{fname_prefix}.{plot_fmt}'))


# TODO better name for this fn
def analyze_onepair(trial_df):

    name1 = odor_strs2single_odor_name(trial_df.columns.get_level_values('odor1'))
    name2 = odor_strs2single_odor_name(trial_df.columns.get_level_values('odor2'))

    def savefig(fig_or_sns_obj, fname_prefix):
        pair_savefig(fig_or_sns_obj, fname_prefix, (name1, name2))

    fly_ids = trial_df.index.get_level_values('fly').unique().sort_values()
    n_flies = len(fly_ids)

    fly_str = ','.join([str(x) for x in
        trial_df.index.get_level_values('fly').unique().sort_values()
    ])
    fly_str = f'n={n_flies} {{{fly_str}}}'

    mean_df = sort_odor_indices(
        trial_df.groupby(level=['odor1','odor2'], axis='columns').mean()
    )

    # TODO TODO TODO how to do fitting like in this fn, but also incorporating lateral
    # inhibition?
    est_df, param_df = competitive_binding_model(mean_df)
    cb_diff_df = est_df - mean_df

    # TODO TODO TODO maybe test my fitting method on the singh et al 2019 data, to make
    # sure i'm getting it right, esp w/ the fmax handling, to the extent it doesn't end
    # up being trivial

    # TODO TODO TODO plot a few example curves (and also a reference set of plots for
    # all?) (for components and mix), capturing diversity of difference / fit quality (+
    # to see what kinds of behavior model is actually capturing, as fit on my data). to
    # what extent is lack of saturation a problem? ways to deal with it?

    # TODO TODO show / output parameters for competitive binding model

    shared_kwargs = dict(
        figsize=(3.0, 7.0),
        xticklabels=format_mix_from_strs,
        yticklabels=format_fly_and_roi,
        ylabel=f'ROI',
        # TODO is this cbar_label appropriate for all plots this is used in?
        ylabel_rotation='horizontal',
        # Default labelpad should be 4.0
        ylabel_kws=dict(labelpad=10.0),
        cbar_shrink=0.3,
    )

    matshow_kwargs = dict(cbar_label=trial_stat_cbar_title, cmap=cmap, **shared_kwargs)

    matshow_diverging_kwargs = dict(cbar_label=diff_cbar_title, **diverging_cmap_kwargs,
        **shared_kwargs
    )

    # TODO TODO do i have enough of a baseline to compute stddev there and use that to
    # get responders significantly above[/below] baseline?

    # TODO TODO z-score (+ subsequent clustering?) using either trial or baseline noise,
    # rather than just relying on what seaborn can do from mean?

    # TODO maybe compute in here and make these plots again (+ just show clustered
    # versions if i do? strictly more info and nicer layout)
    #fig, _ = viz.matshow(sum_diff_df, title='Sum - obs', **matshow_diverging_kwargs)
    #fig, _ = viz.matshow(max_diff_df, title='Max - obs', **matshow_diverging_kwargs)

    max_diff_fig, sum_diff_fig = plot_diffs_from_max_and_sum(mean_df,
        dpi=300, **matshow_diverging_kwargs
    )
    savefig(max_diff_fig, 'max_diff')
    savefig(sum_diff_fig, 'sum_diff')

    #'''
    fig, _ = viz.matshow(est_df, title=f'Competitive binding model\n{fly_str}',
        dpi=300, vmin=0, vmax=3.0, **matshow_kwargs
    )
    savefig(fig, 'cb')

    fig, _ = viz.matshow(cb_diff_df, dpi=300,
        title=f'Competitive binding model - obs\n{fly_str}', **matshow_diverging_kwargs
    )
    savefig(fig, 'cb_diff')

    cb_diff_df_mixes_only = cb_diff_df.loc[:,
        (cb_diff_df.columns.to_frame() == solvent_str).sum(axis='columns') == 0
    ]
    fig, _ = viz.matshow(cb_diff_df_mixes_only, dpi=300,
        title=f'Competitive binding model - obs\n{fly_str}', **matshow_diverging_kwargs
    )
    savefig(fig, 'cb_diff_mixes_only')

    #'''

    # TODO TODO could try using row_colors derived from non-hierarchichal clustering
    # methods to plots those? maybe even disabling the dendrogram?
    # TODO or from fly / glomeruli (for those w/ names)?
    # TODO TODO for glomeruli, maybe do one version w/ and w/o non-named glomeruli

    # 'average' is the default for sns.clustermap
    # 'single' is the default for scipy.cluster.hierarchy.linkage
    # (and there are ofc several others)
    #cluster_methods = ['average', 'single', 'ward']
    # 'euclidean' is the default for both. just wanted to try 'correlation' too
    #cluster_metrics = ['euclidean', 'correlation']

    cluster_methods = ['average']
    cluster_metrics = ['correlation']

    non_clustermap_kwargs = ('figsize', 'cbar_shrink')
    clustermap_kwargs = {
        k: v for k, v in matshow_kwargs.items() if k not in non_clustermap_kwargs
    }
    diverging_clustermap_kwargs = {k: v for k, v in matshow_diverging_kwargs.items()
        if k not in ('figsize', 'cbar_shrink')
    }

    # TODO may want to compare using z_score (or standard_scale) to passing in my own
    # linkage without either operation and then just plotting the [0, 1] data
    for method in cluster_methods:
        for metric in cluster_metrics:

            if method == 'ward' and metric != 'euclidean':
                continue

            shared_clustermap_kwargs = dict(
                col_cluster=False, method=method, metric=metric
            )
            cluster_param_desc = f'clustering: method={method}, metric={metric}'

            # 0=z-score rows, 1=z-score columns (1 since we are clustering rows only)
            #for z_score in (None, 0):
            for z_score in (None,):

                preprocessing_desc = 'pre: z-score, ' if z_score is not None else ''
                param_desc = preprocessing_desc + cluster_param_desc

                desc = f'{fly_str}\n{param_desc}'

                # TODO should i change cmap to diverging one if i'm gonna z-score?

                # TODO refactor to also loop over dfs (and their kwargs...)?

                # TODO some way to get cbar on right that plays nice w/ constrained
                # layout?  maybe just disable and add after (other things seaborn is
                # doing preclude constrained layout anyway)?

                clustergrid = viz.clustermap(mean_df, z_score=z_score, title=desc,
                    **clustermap_kwargs, **shared_clustermap_kwargs
                )

                fname_prefix = f'clust_{method}_{metric}'
                if z_score is not None:
                    fname_prefix += '_zscore'

                savefig(clustergrid, fname_prefix)

            # TODO it doesn't make sense to z-score the difference from model does
            # it? seems too hard to read...
            #'''
            desc = f'{fly_str}\n{cluster_param_desc}'
            clustergrid = viz.clustermap(cb_diff_df,
                title=f'Competitive binding model - obs\n{desc}',
                **diverging_clustermap_kwargs, **shared_clustermap_kwargs
            )

            fname_prefix = f'cb_diff_clust_{method}_{metric}'
            savefig(clustergrid, fname_prefix)
            #'''

    #plt.show()
    #import ipdb; ipdb.set_trace()


# TODO write a csv in addition (and maybe use as main cache too, but would need to
# handle indices)? parquet?
roi_response_stats_cache_fname = 'roi_mean_response_df.p'
response_volume_cache_fname = 'allfly_trial_response_volumes.p'
# TODO better name for this fn (+ probably call at end of main, not just behind -c flag)
def analyze_cache():
    fly_keys = ['date', 'fly_num']
    recording_keys = fly_keys + ['thorimage_id']

    # TODO TODO TODO plot hallem activations for odors in each pair, to see which
    # glomeruli we'd expect to find (at least at their concentrations)
    # (though probably factor it into its own fn and maybe call in main rather than
    # here?)

    warnings.simplefilter('error', pd.errors.PerformanceWarning)

    df = pd.read_pickle(roi_response_stats_cache_fname).T

    # This will just be additional presentations of solvent, interspersed throughout the
    # experiment. Not interesting.
    df = df.loc[:, df.columns.get_level_values('repeat')  <= 2].copy()

    fly_id = df.groupby(level=fly_keys).ngroup() + 1
    fly_id.name = 'fly'

    # TODO maybe write these to a text file as well?
    print('"fly" = ordered across days, "fly_num" = order within a day')
    print(fly_id.to_frame().reset_index()[['fly'] + fly_keys].drop_duplicates(
        ).sort_values('fly').to_string(index=False)
    )
    print()

    '''
    recording_id = df.groupby(level=recording_keys).ngroup() + 1
    recording_id.name = 'recording'

    print(recording_id.to_frame().reset_index()[['recording'] + recording_keys
        ].drop_duplicates().sort_values('recording').to_string(index=False)
    )
    print()
    '''

    df.set_index(fly_id, append=True, inplace=True)
    #df.set_index(recording_id, append=True, inplace=True)

    df = df.droplevel(recording_keys).reorder_levels(['fly', 'roi'])
    #df = df.droplevel(recording_keys).reorder_levels(['fly', 'recording', 'roi'])

    def onepair_dfs(name1, name2, *dfs):
        assert len(dfs) > 0

        def onepair_df(name1, name2, df):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

                return sort_odor_indices(df.loc[:, (name1, name2)].dropna(axis='index'))

        return tuple([onepair_df(name1, name2, df) for df in dfs])

    odf = orns.orns(add_sfr=False)
    odf.columns = pd.Index(data=[orns.receptor2glomerulus[x] for x in odf.columns],
        name='glomerulus'
    )
    abbrev2odor = {v: k for k, v in odor2abbrev.items()}

    def print_hallem(df):
        print(df.T.rename_axis(columns='').to_string(max_cols=None,
            float_format=lambda x: '{:.0f}'.format(x)
        ))

    for name1, name2 in df.columns.to_frame(index=False)[['name1','name2']
        ].drop_duplicates().sort_values(['name1','name2']).itertuples(index=False):

        orig_name1 = abbrev2odor[name1] if name1 in abbrev2odor else name1
        orig_name2 = abbrev2odor[name2] if name2 in abbrev2odor else name2

        # TODO TODO TODO also get values for lower concentrations, when available
        # (not in main data used in orns.orns(), but should be in other csv, maybe also
        # in drosolf somewhere)

        o1df = odf.loc[orig_name1].sort_values(ascending=False).to_frame()
        print_hallem(o1df)
        fig, _ = viz.matshow(o1df)
        pair_savefig(fig, name1, (name1, name2))

        print()

        o2df = odf.loc[orig_name2].sort_values(ascending=False).to_frame()
        print_hallem(o2df)
        fig, _ = viz.matshow(o2df)
        pair_savefig(fig, name2, (name1, name2))

        print('\n')

        pair_df = onepair_dfs(name1, name2, df)[0]
        analyze_onepair(pair_df)

    # TODO TODO TODO try to cluster odor mixture behavior types across odor pairs, but
    # either sorting so ~most activating is always A or maybe by adding data duplicated
    # such that it appears once for A=x,B=y and once for A=y,B=x
    # TODO other ways to achieve some kind of symmetry here?

    #plt.show()
    #import ipdb; ipdb.set_trace()

    # TODO add back some optional csv exporting
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
    global ignore_existing
    global retry_previously_failed
    global analyze_glomeruli_diagnostics_only
    global analyze_glomeruli_diagnostics
    global print_skipped

    atexit.register(cleanup_created_dirs_and_links)

    parser = argparse.ArgumentParser()

    parser.add_argument('matching_substrs', nargs='*', help='If passed, only data whose'
        ' ThorImage path contains one of these substrings will be analyzed.'
    )

    # TODO TODO TODO what is currently causing this to hang on ~ when it is done with
    # iterating over the inputs? some big data it's trying to [de]serialize?
    parser.add_argument('-j', '--parallel', action='store_true',
        help='Enables parallel calls to process_experiment. '
        'Disabled by default because it can complicate debugging.'
    )
    parser.add_argument('-i', '--ignore-existing', action='store_true',
        help='Re-calculate non-ROI analysis and analysis downstream of ImageJ/suite2p '
        'ROIs.'
    )
    parser.add_argument('-r', '--retry-failed', action='store_true',
        help='Retry steps that previously failed (frame-to-odor assignment or suite2p).'
    )
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Also prints paths to data that has some analysis skipped, with reasons '
        'for why things were skipped.'
    )
    # TODO rename to across fly or something
    parser.add_argument('-c', '--analyze-cache-only', action='store_true',
        help='Only analyze cached aggregate ROI statistics across flies.'
    )
    # TODO try to still link everything already generated (same w/ pairs)
    parser.add_argument('-g', '--glomeruli-diags-only', action='store_true',
        help='Only analyze glomeruli diagnostics (mainly for use on acquisition '
        'computer).'
    )

    args = parser.parse_args()

    matching_substrs = args.matching_substrs

    parallel = args.parallel
    analyze_cache_only = args.analyze_cache_only
    ignore_existing = args.ignore_existing
    retry_previously_failed = args.retry_failed
    analyze_glomeruli_diagnostics_only = args.glomeruli_diags_only
    print_skipped = args.verbose

    del parser, args

    if analyze_cache_only:
        analyze_cache()
        return

    if parallel:
        import matplotlib
        # Won't get warnings that some of the interactive backends give in the
        # multiprocessing case, but can't make interactive plots.
        matplotlib.use('agg')

    if not is_acquisition_host:
        # TODO switch to doing just first time we would add something there?  (and do
        # same for other dirs, especially those we might not want generated on the
        # acquisition computer)
        # TODO rename to 'panels'/'experiment_types' or somethings
        makedirs(pair_directories_root)

    if analyze_glomeruli_diagnostics_only:
        analyze_glomeruli_diagnostics = True

    if analyze_glomeruli_diagnostics:
        # TODO if i am gonna keep this, need a way to just re-link stuff without also
        # having to compute the heatmaps in the same run (current behavior)
        #
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
        # TODO TODO add back / handle bad elsewhere if i still decide too weak. for
        # suite2p alone it shouldn't be in this list.
        #'2021-05-05/1/butanal_and_acetone',

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

    # NOTE: this start_date should exclude all pair-only experiments and only select
    # experiments as part of the return to kiwi-approximation experiments (that now
    # include ramps of eb/ea/kiwi mixture). For 2021 pair experiments, was using
    # start_date of '2021-03-07'
    #start_date = '2021-03-07'
    start_date = '2022-02-04'

    common_paired_thor_dirs_kwargs = dict(
        start_date=start_date, ignore=bad_thorimage_dirs, ignore_prepairing=('anat',)
    )

    names2final_concs, seen_stimulus_yamls2thorimage_dirs, names_and_concs_tuples = \
        odor_names2final_concs(**common_paired_thor_dirs_kwargs)

    keys_and_paired_dirs = util.paired_thor_dirs(matching_substrs=matching_substrs,
        **common_paired_thor_dirs_kwargs
    )
    del common_paired_thor_dirs_kwargs

    main_start_s = time.time()

    if not parallel:
        # `list` call is just so `starmap` actually evaluates the fn on its input.
        # `starmap` just returns a generator otherwise.
        was_analyzed = list(starmap(process_experiment, keys_and_paired_dirs))

    else:
        with mp.Manager() as manager:
            # "If processes is None then the number returned by os.cpu_count() is used
            # [for the number of processes in the Pool]"
            # https://docs.python.org/3/library/multiprocessing.html
            n_workers = os.cpu_count()
            print(f'Processing experiments with {n_workers} workers')

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

            with manager.Pool(n_workers) as pool:
                was_analyzed = pool.starmap(
                    #worker_fn,
                    process_experiment,
                    [x + (shared_state,) for x in keys_and_paired_dirs]
                )

            multiprocessing_namespace_to_globals(shared_state)

            names_and_concs2analysis_dirs = {
                k: ds for k, ds in names_and_concs2analysis_dirs.items() if len(ds) > 0
            }

    print(f'Checked {len(was_analyzed)} experiment(s)')

    n_analyzed = sum([x is not None for x in was_analyzed])
    print(f'Analyzed {n_analyzed} experiment(s)')

    total_s = time.time() - main_start_s
    print(f'Took {total_s:.0f}s\n')

    if not is_acquisition_host and len(response_volumes_list) > 0:

        # TODO factor all this into a function so i can more neatly call it multiple
        # times w/ diff downsampling factors

        # downsampling factor
        # 192 / 4 = 48
        # TODO TODO better name
        # TODO switch back to 0 / a sequence including this + 0, and change code to try
        # both?
        ds = 4
        pixel_corr_basename = 'pixel_correlations'
        if ds > 0:
            pixel_corr_basename = f'{pixel_corr_basename}_ds{ds}'

        pixel_corr_plots_root = join(plot_fmt, pixel_corr_basename)
        makedirs(pixel_corr_plots_root)

        #spatial_dims = ['z', 'y', 'x']
        spatial_shapes = [tuple(dict(zip(x.dims, x.shape))[n] for n in spatial_dims)
            for x in response_volumes_list
        ]
        # doesn't handle ties, but that should be fine
        most_common_shape = Counter(spatial_shapes).most_common(1)[0][0]

        # TODO switch to not throwing out any shapes. just need to test the NaNs are
        # handled appropriately in any downsteam correlation / plotting code
        consistent_spatial_shape_response_volumes = []
        for shape, recording_response_volumes in zip(spatial_shapes,
            response_volumes_list):

            if shape != most_common_shape:
                # Each recording_response_volumes should only have one value for each of
                # these.
                date = pd.to_datetime(recording_response_volumes.date.item(0)).date()
                fly_num = recording_response_volumes.fly_num.item(0)
                thorimage_id = recording_response_volumes.thorimage_id.item(0)

                warn(f'not including {date}/{fly_num}/{thorimage_id} in '
                    f'response_volumes because it had shape of {shape} (!= most common '
                    f'{most_common_shape})'
                )
                continue

            consistent_spatial_shape_response_volumes.append(recording_response_volumes)

        # NOTE: memory usage is just under 4GB loading 85 experiments from 2022-02-04 to
        # 2022-04-03.
        response_volumes = xr.concat(consistent_spatial_shape_response_volumes, 'odor')

        assert response_volumes.notnull().sum().item(0) == sum([
            x.notnull().sum().item(0) for x in consistent_spatial_shape_response_volumes
        ])

        # NOTE: will need to remove this assertion if i stop throwing out stuff not the
        # most common shape.
        #
        # concatenating along odor axis should give us no NaN, as long as movie shapes
        # are first restricted to all be the same (as otherwise the odor dimension
        # should be the only one that varies in length)
        assert response_volumes.isnull().sum().item(0) == 0

        # TODO fix in case only one experiment is here:
        # ValueError: If using all scalar values, you must pass an index
        response_volumes = util.add_group_id(response_volumes,
            ['date', 'fly_num', 'thorimage_id'], name='recording_id', dim='odor'
        )

        # TODO TODO TODO also try calculating correlations only for pixels where the max
        # dF/F (across all trials) is above some threshold (or maybe threshold a pixel
        # z-score?)

        if ds > 0:
            print('downsampling response volumes...', end='', flush=True)
            response_volumes = response_volumes.coarsen(y=ds, x=ds).mean()
            print('done', flush=True)

        # TODO TODO TODO motion correct within each fly (all recordings to one template
        # too), to facilitate across-recording, within-fly pixel correlation, among
        # other things

        # TODO TODO TODO try to concatenate recordings from within fly together so that
        # i can correlate pixels across them.
        # TODO memory profile w/ just first group, to be more clear on usage of just the
        # grouping itself and (more importantly) all of the steps in the loop
        # TODO tqdm/time loop
        # TODO TODO profile
        corr_list = []
        _checked = False
        for labels, garr in response_volumes.groupby('recording_id'):

            panel = garr.panel.item(0)

            if panel == 'glomeruli_diagnostics':
                continue

            # TODO maybe incoporate this / some of this data into a version of these
            # plots that also has all the non-pair-experiment data for the same panel?
            # (probably want across movie registration first, as mentioned above)
            # TODO TODO TODO or at least make a separate version that just compares
            # 2H/1o3ol at the various concentrations. not necessarily any mixtures.
            if garr.is_pair.item(0):
                continue

            # TODO make a (hong2p) fn for getting all metadata for an experiment from
            # either xarray / pandas input? maybe config recording / fly keys at start
            # via some setters? (and/or for going directly to a str like this?)
            experiment_id = (f'{format_date(garr.date.item(0))}/{garr.fly_num.item(0)}/'
                f'{garr.thorimage_id.item(0)}'
            )
            plot_dir = get_plot_dir(experiment_id)

            # TODO TODO TODO try smoothing/downsampling before calculating correlation
            # and see whether / how much that improves things (include description of
            # process + params in fnames + plot titles too)

            #corr = xr.corr(garr, garr.rename(odor='odor_b'), dim=spatial_dims)
            #
            # Need to do this ugly thing rather than commented line above because one we
            # have two different dimensions (odor and odor_b) that have MultiIndices
            # with levels that have the same name (e.g. odor and odor_b both have an
            # 'odor1' level), it seems impossible to get out of that situation with
            # standard xarray calls, and it makes some other calls (including xr.concat)
            # impossible.
            # TODO make a PR to fix this xarray behavior (probably could just make
            # rename also work with multiindex levels, as if they were any regular
            # coordinate name)? currently get:
            # "ValueError: cannot rename 'odor1' because it is not a variable or
            # dimension in this dataset" when trying to rename MultiIndex levels
            #
            # TODO TODO TODO at least factor into a function for renaming multiindex
            # levels
            garr2 = garr.reset_index('odor').rename(
                odor='odor_b', odor1='odor1_b', odor2='odor2_b', repeat='repeat_b'
                ).set_index(odor_b=['odor1_b', 'odor2_b', 'repeat_b'])

            corr = xr.corr(garr, garr2, dim=spatial_dims)

            if checks and not _checked:
                stacked = garr.stack(pixel=spatial_dims)
                # TODO TODO TODO update to rename odor_b levels as in garr2 above +
                # uncomment
                #corr2 = xr.corr(stacked, stacked.rename(odor='odor_b'), dim='pixel')
                # TODO better way to check coordinates equal?
                #assert corr.coords.to_dataset().identical(corr2.coords.to_dataset())

                # Both of this are True with or without .values after each argument.
                # In both cases, neither seems to have np.array_equals True for the same
                # inputs, so it seems there are some pervasive numerical differences
                # (which shouldn't matter, and I'm not sure which is the more correct of
                # the inputs).
                #assert np.allclose(corr2, corr)
                assert np.allclose(corr, stacked.to_pandas().T.corr())
                _checked = True

            corr_list.append(corr)

            xticklabels = format_mix_from_strs
            yticklabels = format_mix_from_strs

            with warnings.catch_warnings():
                # For the warning from format_mix_from_strs since we aren't dropping
                # 'repeat' level
                warnings.simplefilter('ignore', UserWarning)

                # TODO shared vmin/vmax?
                # TODO colorbar label
                fig, _  = viz.matshow(corr, title=experiment_id,
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    # NOTE: this would currently cause failure on the pair experiment
                    # data (because the multiple solvent entries i assume)
                    group_ticklabels=True,
                )

            # TODO tight_layout / whatever appropriate to not have cbar label cut off
            # and not have so much space on the left

            # TODO test that on a fresh run (or if another run finishes first and
            # deletes stuff...) that this still works / fix if not (always makedirs in
            # savefig?)
            fig_path = savefig(fig, plot_dir, 'pixel_corr')

            assert type(panel) is str, 'ensure panel is defined / update code to skip'
            panel_dir = join(pixel_corr_plots_root, panel)
            makedirs(panel_dir)

            link_prefix = '_'.join(experiment_id.split('/')[:-1])
            link_path = join(panel_dir, f'{link_prefix}.{plot_fmt}')

            symlink(fig_path, link_path, relative=True)

        # TODO TODO TODO aggregate these to average across flies (+ make plots of that,
        # including n in figure)
        # TODO worth having a version w/o trials?

        # TODO TODO try to swap dims around / concat in a way that we dont get a bunch
        # of nans. possible? could just concat separately for each panel?
        corrs = xr.concat(corr_list, 'fly')
        print(f'{corrs.shape=}\n')

        # TODO TODO TODO get average correlations across flies to work (what loop below
        # was trying)
        #
        # reset_index() because otherwise we get: "NotImplementedError: isna is not
        # defined for MultiIndex" when trying to group on panel.
        #
        # TODO TODO might need to also groupby experiment or something else if i make
        # corr pltos for pair stuff in loop above + also include that data in corrs
        for panel, garr in corrs.reset_index(['odor', 'odor_b']).groupby('panel',
            squeeze=False, restore_coord_dims=True):

            print(f'{panel=}')
            print(f'{garr.shape=}')

            # TODO TODO TODO why is it getting reshaped to (7290,) w/
            # 'stacked_fly_odor_odor_b'

            # "dropping along multiple dimensions simultaneously is not yet supported"
            import ipdb; ipdb.set_trace()
            garr = garr.dropna('odor', how='all').dropna('odor_b', how='all')
            import ipdb; ipdb.set_trace()

            #panel = garr.panel.item(0)

        import ipdb; ipdb.set_trace()
        #

        # TODO TODO TODO don't overwrite if we are only running on a subset of data? did
        # i handle this w/ trial_df.p writing (if not, do, and hopefully in same way)?
        # TODO TODO TODO uncomment after figuring out commment above + implementing
        #write_dataarray(response_volumes, response_volume_cache_fname)


    #if len(odors_without_abbrev) > 0:
    #    print('Odors without abbreviations:')
    #    pprint(odors_without_abbrev)

    def earliest_analysis_dir_date(analysis_dirs):
        return min(d.split(os.sep)[-3] for d in analysis_dirs)

    failed_assigning_frames_analysis_dirs = [
        x.replace('raw_data', 'analysis_intermediates')
        for x in failed_assigning_frames_to_odors
    ]

    # TODO TODO TODO extend to other stuff we want to analyze (e.g. at least the
    # kiwi/control1 panel data)
    show_empty_statuses = False
    print('odor pair counts (of data considered) at various analysis stages:')
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

        if analyze_suite2p_outputs:
            print('suite2p:')
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

                if show_empty_statuses or len(status_dirs) > 0:
                    print(f' - {s2p_status} ({len(status_dirs)})')
                    for analysis_dir in sorted(status_dirs):
                        short_id = shorten_path(analysis_dir)
                        print(f'   - {short_id}')

            print()

        if analyze_ijrois:
            print('ImageJ:')

            ij_status2dirs = {
                'have ROIs': [x for x in analysis_dirs if x in dirs_with_ijrois
                    and x not in failed_assigning_frames_analysis_dirs
                ],
                'need ROIs': [x for x in analysis_dirs if x not in dirs_with_ijrois
                    and x not in failed_assigning_frames_analysis_dirs
                ],
                # TODO implement + add needing merge category (i.e. still has some
                # default names)
                'failed assigning frames': [x for x in analysis_dirs if x in
                    failed_assigning_frames_analysis_dirs
                ],
            }

            for ij_status, status_dirs in ij_status2dirs.items():
                if show_empty_statuses or len(status_dirs) > 0:
                    print(f' - {ij_status} ({len(status_dirs)})')
                    for analysis_dir in sorted(status_dirs):
                        short_id = shorten_path(analysis_dir)
                        print(f'   - {short_id}')

            print()
    print()

    # TODO also check that all loaded data is using same stimulus program
    # (already skipping stuff with odor pulse < 3s tho)

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
        'Failed assigning frames to odors', failed_assigning_frames_to_odors
    )

    if do_suite2p:
        print_nonempty_path_list(
            'suite2p failed:', failed_suite2p_dirs
        )

    if analyze_suite2p_outputs:
        print_nonempty_path_list(
            'suite2p needs to be run on the following data', s2p_not_run,
            alt_msg='suite2p has been run on all currently included data'
        )

        # NOTE: only possible if suite2p for these was run outside of this pipeline, as
        # `run_suite2p` in this file currently marks all ROIs as "good" (as cells, as
        # far as suite2p is concerned) immediately after a successful suite2p run.
        print_nonempty_path_list(
            'suite2p outputs with ROI labels not modified', iscell_not_modified
        )

        print_nonempty_path_list(
            'suite2p outputs where no ROIs were marked bad', iscell_not_selective
        )
        print_nonempty_path_list(
            'suite2p outputs where no ROIs were merged', no_merges
        )

    # TODO TODO probably print stuff in gsheet but not local and vice versa

    if len(ij_trial_dfs) == 0:
        cprint('No ImageJ ROIs defined for current experiments!', 'yellow')
        return

    trial_df = pd.concat(ij_trial_dfs, axis='columns')

    # TODO define globally as a tuple / fix whatever fucky thing my multiprocessing
    # wrapper code is doing to some/all global lists, or at least clearly document it
    fly_keys = ['date', 'fly_num']
    recording_keys = fly_keys + ['thorimage_id']

    trial_df.sort_index(level=recording_keys, sort_remaining=False, axis='columns')

    trial_df.to_pickle(roi_response_stats_cache_fname)

    '''
    # TODO factor into analysis that loops over all relevant data + delete
    pdf = ij_trial_dfs[-2]
    cdf = ij_trial_dfs[-1]
    # TODO TODO try to concat in a way that also works for what i was doing for pair
    # stuff (or split into pair / not [strictly] pair stuff and concat each separately)
    df = pd.concat([d.droplevel('thorimage_id', axis='columns') for d in [cdf, pdf]])

    # TODO delete if this isn't required for exising plotting fns to work [/ modify them
    # to not care + delete]
    df = df.droplevel(['date', 'fly_num'], axis='columns')

    # will probably be useful for doing stuff to specifically pair recording actually
    # (given overlap in top two components between the two recordings)
    #df = df.droplevel(['name1', 'name2'])

    def reorder_odors(df):
        nonpair_rows = []
        pair_rows = []
        for x in df.index:
            if x[:2] == (np.nan, np.nan):
                nonpair_rows.append(x)
            else:
                pair_rows.append(x)

        vialmix_rows = [x for x in nonpair_rows
            # TODO change to 'kiwi approx' or whatever the prefix actually should be
            if (x[2].startswith('control mix') or x[2].startswith('kiwi'))
        ]
        component_rows = [x for x in nonpair_rows if x not in vialmix_rows]

        # TODO probably plot pair A (at highest conc, alone) next to nonpair A
        # TODO TODO but use a different color font for pair data legend or something
        # else to indicate

        # TODO TODO TODO can we rely on odors A, B already being in a constent order?
        # fix if not. document why we can rely on this if so.
        a_rows = []
        b_rows = []
        # TODO TODO TODO will i need to sort the diag rows into a particular order?
        diag_rows = []
        # TODO TODO will i need to do some sorting here? should i replot the concs
        # alone?
        br_edge_rows = []

        for x in pair_rows:
            _, _, o1, o2, _ = x

            if o2 == solvent_str and o1 != solvent_str:
                a_rows.append(x)
            elif o1 == solvent_str and o2 != solvent_str:
                b_rows.append(x)

            # TODO TODO TODO how to just get the diagonal?
            #elif 

        import ipdb; ipdb.set_trace()

    # TODO TODO TODO define odor order appropriately (by re-ordering before plot all),
    # if that is all it takes to get plot betty wants
    # TODO TODO + probably drop other data (from pair grid) she didnt want in summary
    df = reorder_odors(df)

    # TODO define roi_sortkeys kwarg to get an interesting order
    # (or at least by mean activation across experiments within panel / alphabetical)
    fig, mean_df = plot_all_roi_mean_responses(df, sort_odors=False,
        title='03-10/1 test summary', cbar_label=trial_stat_cbar_title, cbar_shrink=0.4
    )
    plt.show()
    #savefig(fig, roi_plot_dir, 'all_rois_by_z')
    #
    import ipdb; ipdb.set_trace()
    '''


if __name__ == '__main__':
    main()

