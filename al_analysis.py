#!/usr/bin/env python3

import argparse
import atexit
import os
from os.path import join, split, exists, expanduser, islink, getmtime
from pprint import pprint, pformat
from collections import defaultdict, Counter
from copy import deepcopy
from functools import wraps
import warnings
import time
import shutil
import traceback
import subprocess
import sys
import logging
import pickle
from pathlib import Path
import glob
from itertools import starmap
import multiprocessing
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Optional, Tuple, List, Type, Union, Dict, Any
import json
import re

import numpy as np
import pandas as pd
import xarray as xr
import tifffile
import yaml
import ijroi
from matplotlib import colors
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from scipy.optimize import curve_fit
import statsmodels.api as sm
import colorama
from termcolor import cprint, colored
import olfsysm as osm
import drosolf
from drosolf import orns, pns
from latex.exc import LatexBuildError
from tqdm import tqdm
# suite2p imports are currently done at the top of functions that use them

from hong2p import util, thor, viz, olf
from hong2p import suite2p as s2p
from hong2p.suite2p import LabelsNotModifiedError, LabelsNotSelectiveError
from hong2p.roi import (rois2best_planes_only, ijroi_filename, has_ijrois, ijroi_mtime,
    ijroi_masks, extract_traces_bool_masks, ijroiset_default_basename, is_ijroi_named,
    is_ijroi_certain, ijroi_name_as_if_certain, ijroi_comparable_via_name,
    certain_roi_indices, select_certain_rois
)
from hong2p.util import (shorten_path, shorten_stimfile_path, format_date, date_fmt_str,
    # TODO refactor current stuff to use these (num_[not]null)
    num_null, num_notnull, add_fly_id
)
from hong2p.olf import (format_odor, format_mix_from_strs, format_odor_list,
    solvent_str, odor2abbrev, odor_lists_to_multiindex
)
from hong2p.viz import dff_latex, no_constrained_layout
from hong2p.err import NoStimulusFile
from hong2p.thor import OnsetOffsetNumMismatch
from hong2p.types import ExperimentOdors, Pathlike
from hong2p.xarray import (move_all_coords_to_index, unique_coord_value, scalar_coords,
    drop_scalar_coords, assign_scalar_coords_to_dim, odor_corr_frame_to_dataarray
)
from hong2p.latex import make_pdf
import natmix
# TODO rename these [load|write]_corr_dataarray fns to remove reference to "corr"
# (since correlations are not what i'm using with these fns here)
# (but maybe add something else descriptive?)
from natmix import load_corr_dataarray, drop_nonlone_pair_expt_odors, dropna_odors
from natmix import write_corr_dataarray as _write_corr_dataarray

# TODO move this to hong2p probably
from hong_logging import init_logger


# not sure i'll be able to fix this one...
# can't seem to convert this warning to error this way.
warnings.filterwarnings('error', 'invalid value encountered in mulitply')
# TODO TODO wasn't there some SO post saying errors from certain c extensions needed
# separate handling? figure that out!

colorama.init()

# TODO move some/all of these into rcParam context manager surrounding calls these were
# important for (presumably image_grid/plot_rois and related plots? move into
# hong2p.viz?) (or delete if i end up switching only stuff that needed this to
# non-constrained layout...)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.constrained_layout.w_pad'] = 1/72
plt.rcParams['figure.constrained_layout.h_pad'] = 0.5/72
plt.rcParams['figure.constrained_layout.wspace'] = 0
plt.rcParams['figure.constrained_layout.hspace'] = 0

# TODO also add to a matplotlibrc file (under ~/.matplotlib?)?
# TODO 42 same as TrueType?
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# TODO replace this + use of warnings.warn w/ logging.warning (w/ logger probably
# currently just configured to output to stdout/stderr)
def formatwarning_msg_only(msg, category, *args, **kwargs):
    """Format warning without line/lineno (which are often not the relevant line)
    """
    warn_type = category.__name__ if category.__name__ != 'UserWarning' else 'Warning'
    return colored(f'{warn_type}: {msg}\n', 'yellow')

# TODO do just in main?
# TODO TODO maybe also toggle this w/ -v CLI flag (still have it colored orange tho...)
# (or a dedicated flag for this?)
warnings.formatwarning = formatwarning_msg_only


# TODO maybe log all warnings?
# TODO TODO replace w/ logging.warning (have init_logger just hook into warnings.warn?
# some standard mechanism for that?)
def warn(msg):
    warnings.warn(str(msg))


orn_drivers = {'pebbled'}

###################################################################################
# Constants that affect behavior of `process_recording`
###################################################################################
analysis_intermediates_root = util.analysis_intermediates_root(create=True)

# TODO TODO TODO cli flag to quickly disable motion correction
# TODO test that w/ do_regis...=True and min_input<'mocorr', still do mocorr if we have
# side labelled (and then just stick w/ that min_input?)

# TODO TODO TODO actually rerun mocorr if it is using less than the current subset of
# movies (or at least have a flag to do this, and prominently warn if False)

# 'mocorr' | 'flipped' | 'raw'
# load_movie will raise an IOError for any experiments w/o at least a TIFF of this level
# of processing (motion corrected, flipped-L/R-if-needed, and raw, respectively)
#min_input = 'mocorr'
# TODO modify so i can specify left|right (or _l|r?) in filename, to override what
# gsheet says for a given fly (or to not use the gsheet in particular cases?)
# and/or in Experiment.xml?
##min_input = 'flipped'
min_input = 'raw'

# TODO don't do this if we don't have (some notion of) a minimum set of recordings?
# or just add a CLI flag to not do this?
#
# Whether to motion correct all recordings done, in a given fly, to each other.
#do_register_all_fly_recordings_together = False
do_register_all_fly_recordings_together = True

# TODO add --ignore-existing option to set this True?
#
# If True, will run suite2p registration for any flies without a registration run whose
# parameters match the current suite2p default parameters (in the GUI).
# If False, will use any existing registration, for flies where they exist.
rerun_old_param_registrations = False

# Whether to only analyze experiments sampling 2 odors at all pairwise concentrations
# (the main type of experiment for this project)
analyze_pairgrids_only = False

# TODO delete?
# If there are multiple experiments with the same odors, only the data from the most
# recent concentrations for those odors will be analyzed.
final_pair_concentrations_only = True

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
analyze_suite2p_outputs = False

analyze_ijrois = True

do_analyze_response_volumes = False

# TODO TODO TODO change to using seconds and rounding to nearest[higher/lower
# multiple?] from there
# TODO some quantitative check this is ~optimal?
# Note this is w/ volumetric sampling of ~1Hz.
# 4 seems to produce similar outputs to 2, though slightly dimmer in most cases.
# 3 is not noticeably dimmer than 2, so since it's averaging a little more data, I'll
# use that one.
#n_volumes_for_response = 3
# TODO TODO restore to 3 / set based on data each time
# (using 2 to accomodate earlier recordings for Remy's project, where pulse was still
# 2s)
n_volumes_for_response = 2

# If this is None, it will use all the everything from the start of the trial to the
# frame before the first odor frames. Otherwise, will use this many frames before first
# odor frame.
n_volumes_for_baseline = None

# Whether to exclude last frame before odor onset in calculating the baseline for each
# trial. These baselines are used to define the dF/F for each trial. This can help hedge
# against off-by-one bugs in the frame<-> assignment code, but might otherwise weaken
# signals.
exclude_last_pre_odor_frame = False

# I seemed to end up imaging this side more anyway...
# Movies taken of any fly's right AL's will be flipped along left/right axis to be more
# easily comparable to the data from left ALs.
standard_side_orientation = 'left'
assert standard_side_orientation in ('left', 'right')

ignore_bounding_frame_cache = False

# Also required if do_suite2p is True, but that is always going to be False the only
# time this currently is (when is_acquisition_host is True).
do_convert_raw_to_tiff = True

# If False, will not write any TIFFs (other than raw.tif, which will always only get
# written if it doesn't already exist), including dF/F TIFF.
write_processed_tiffs = True
want_dff_tiff = False

want_dff_tiff = want_dff_tiff and write_processed_tiffs

# When concatenating multiple recordings (from one fly, registered together) and their
# metadata, fail if any frame<->odor metadata is missing if True, else warn.
allow_missing_trial_and_frame_json = True

links_to_input_dirs = True

# TODO shorten any remaining absolute paths if this is True, so we can diff outputs
# across installs w/ data in diff paths
print_full_paths = False

save_figs = True

# TODO support multiple (don't know if i want this to be default though, cause extra
# time)
# TODO add CLI arg to override this?
#plot_fmt = os.environ.get('plot_fmt', 'pdf')
plot_fmt = os.environ.get('plot_fmt', 'png')

# Overall folder structure should be: <driver>_<indicator>/<plot_fmt>/...
across_fly_ijroi_dirname = 'ijroi'
across_fly_pair_dirname = 'pairs'
across_fly_diags_dirname = 'glomeruli_diagnostics'

trial_and_frame_json_basename = 'trial_frames_and_odors.json'

mocorr_concat_tiff_basename = 'mocorr_concat.tif'

# NOTE: trial_dff* is still a mean within a trial, but trialmean_dff* is a mean OF THOSE
# MEANS (across trials)
trial_dff_tiff_basename = 'trial_dff.tif'
trialmean_dff_tiff_basename = 'trialmean_dff.tif'
max_trialmean_dff_tiff_basename = 'max_trialmean_dff.tif'
min_trialmean_dff_tiff_basename = 'min_trialmean_dff.tif'

cmap = 'plasma'
# TODO change cmap to 'vlag', along with remy (and in all places w/ diverging)
# (why did i want to change to 'vlag'? what is it?)
diverging_cmap = 'RdBu_r'

# TODO is there a 'gray_r'? try that?
# NOTE: 'Greys' is reversed wrt 'gray' (maybe not exactly, but it is white->black),
# and I wanted to try it to be more printer friendly, but at least without additional
# tweaking, it seemed a bit harder to use to see faint ROIs.
#anatomical_cmap = 'Greys'
anatomical_cmap = 'gray'

# This background will often either be anatomical (e.g. an average of the movie),
# but may also be computed from functional data (e.g. max across trial-mean dF/Fs),
# but we still want to use the same colormap for consistency (+ ease of seeing ROI
# boundaries/labels against it)
roi_bg_cmap = anatomical_cmap

# TODO could try TwoSlopeNorm, but would probably want to define bounds per fly (or else
# compute in another pass / plot these after aggregating?)
diverging_cmap_kwargs = dict(cmap=diverging_cmap,
    # TODO delete kwargs / determine from data in each case (and why does it
    # seem fixed to [-1, 1] without this?)
    # TODO TODO am i understanding this correctly? (and was default range really
    # [-1, 1], and not the data range, with no kwargs to this (or was it transforming
    # data and was that the range of the transformed data???)?
    norm=colors.CenteredNorm(halfrange=2.0),
)

# TODO change cmap to 'vlag', along with remy (and in all places w/ diverging)
# (why did i want to change to 'vlag'? what is it?)
remy_matshow_kwargs = dict(cmap=diverging_cmap, vmin=-1, vmax=1, fontsize=10.0)

dff_cbar_title = f'{dff_latex}'

# TODO better name
trial_stat_cbar_title = f'Mean peak {dff_latex}'

diff_cbar_title = f'$\Delta$ mean peak {dff_latex}'

single_dff_image_row_figsize = (6.4, 1.6)

# TODO TODO should i switch to a diverging colormap now that i'm using min < 0
dff_vmin = -0.5
# TODO TODO restore / set on a per-fly basis based on some percetile of dF/F
##dff_vmax = 3.0
dff_vmax = 2.0

diag_example_plot_roi_kws = dict(
    #vmin=0.0, vmax=0.75,
    vmin=0.0, vmax=0.8,
    cbar_label=f'mean {dff_cbar_title}',
)
roi_background_minmax_clip_frac = 0.025

ax_fontsize = 7

diag_panel_str = 'glomeruli_diagnostics'

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

# TODO TODO refactor google sheet metadata handling so it doesn't download until it's
# needed (or at least not outside of __main__?)?

# TODO set bool_fillna_false=False (kwarg to gsheet_to_frame) and manually fix any
# unintentional NaN in these columns if I need to use the missing data for early
# diagnostic panels (w/o some of the odors only in newest set) for anything

# This file is intentionally not tracked in git, so you will need to create it and
# paste in the link to this Google Sheet as the sole contents of that file. The
# sheet is located on our drive at:
# 'Hong Lab documents/Tom - odor mixture experiments/pair_grid_data'
# TODO TODO rename prefix to al_analysis or something more general
gdf = util.gsheet_to_frame('pair_grid_data_gsheet_link.txt', normalize_col_names=True)
gdf.set_index(['date', 'fly'], verify_integrity=True, inplace=True)

# Currently has some explicitly labelled 'pebbled' (for new megamat experiments where I
# also have some some 'GH146' data), but all other data should come from pebbled flies.
gdf.driver = gdf.driver.fillna('pebbled')

# TODO TODO edit sheet for 2022-07-02 flies to remove '?' from '6f?', or handle question
# marks
# TODO TODO if i don't switch off 8m for the PN experiments, first fillna w/ '8m' for
# GH146 flies
gdf.indicator = gdf.indicator.fillna('6f')

# This is the name as converted by what `normalize_col_names=True` triggers.
last_gsheet_col_before_glomeruli_diag_statuses = 'side'
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
# TODO check all / all_bad still there after i renamed / moved some stuff in sheet
glomeruli_diag_status_df.rename(columns={'vm7d': 'vm7', 'all': 'all_bad'}, inplace=True)


# For cases where there were multiple glomeruli diagnostic experiments (e.g. both sides
# imaged, and only one used for subsequent experiments w/in fly). Paths (relative to
# data root) to the recordings not followed up on / representative should go here.
unused_glomeruli_diagnostics = (
    # glomeruli_diagnostics_otherside is the one used here
    '2021-05-25/2/glomeruli_diagnostics',
)

# TODO TODO TODO clarify how these behave if something is missing (in comment)
panel2name_order = deepcopy(natmix.panel2name_order)
panel_order = list(natmix.panel_order)

# TODO TODO any reason for this order? just use order loaded from config /
# glomeruli_diagnostics.yaml?
# TODO actually load from generator config (union of all loaded w/ this panel?)
# -> use associated glomeruli keys of odors to sort
#
# Sorted manually to roughly alphabetically sort by names of glomeruli we are trying to
# target with these diagnostics.
panel2name_order[diag_panel_str] = [
    '3mtp',
    'va',
    # DL5
    't2h',
    'ms',
    'a-terp',
    'geos',
    # DM5
    'e3hb',
    'mhex',
    # DM4
    'ma',
    # DM1
    'ea',
    # VM7d
    '2-but',
    '2h',
    'ga',
    'HCl',
    'carene',
    'farn',
    '4h-ol',
    '2but',
    # VA2
    '2,3-b',
    'p-cre',
    'aphe',

    '1-6ol',
    'paa',
    'fench',
    'CO2',

    # VM2/VA2/?
    'ecrot',

    # VC4 (+VA2)
    'elac',
    # ~VM3
    'acetoin',
]

panel_order = [diag_panel_str] + panel_order

# TODO TODO get order from yaml files if not specified? per-panel flag to do this?
# TODO factor out to something like natmix (or natmix itself?)?
# TODO (as w/ other stuff), load from either generated or generator-input YAMLs
# (in this case, tom_olfactometer_configs/megamat0.yaml)
# TODO try deleting this entry (just don't want it to add warnings...), b/c this is
# supposed to be alphabetical anyway...
panel2name_order['megamat'] = [
    # TODO TODO TODO make versions with this order AS WELL AS alphabetical order
    # TODO is this the current (non-alphabetical / cluster) order that remy uses too
    # (pretty sure)?
    '2h', 'IaA', 'pa', '2-but', 'eb', 'ep', 'aa', 'va', 'B-cit', 'Lin', '6al',
    't2h', '1-8ol', '1-5ol', '1-6ol', 'benz', 'ms'
]
# TODO replace w/ just excluding an entry for this from panel2name_order (might wanna
# handle panel_order explicitly then...)
'''
    '1-5ol',
    # TODO TODO make sure it works w/ abbrev='6ol' too
    '1-6ol',
    '1-8ol',
    '2-but',
    '2h',
    '6al',
    'B-cit',
    'IaA',
    'Lin',
    'aa',
    'benz',
    'eb',
    'ep',
    'ms',
    'pa',
    't2h',
    'va',
]
'''
# Putting this before my 'control' panel, so that shared odors are plotted in correct
# order for this data
panel_order.insert(1, 'megamat')

if analyze_pairgrids_only:
    analyze_glomeruli_diagnostics = False

frame_assign_fail_prefix = 'assign_frames'
suite2p_fail_prefix = 'suite2p'

spatial_dims = ['z', 'y', 'x']
fly_keys = ['date', 'fly_num']


checks = True

# TODO distinguish between these two in CLI doc (or delete blanket -i)
# TODO should i even have a blanket '-i' option?
ignore_if_explicitly_requested = ('json',)
also_ignore_if_ignore_existing_true = ('nonroi', 'ijroi', 'suite2p')
ignore_existing_options = (
    ignore_if_explicitly_requested + also_ignore_if_ignore_existing_true
)

# Changed as a globals in main (exposed as command line arguments)
ignore_existing = False
# TODO probably make another category or two for data marked as failed (in the breakdown
# of data by pairs * concs at the end) (if i don't refactor completely...)
retry_previously_failed = False

# TODO TODO probably delete this + pairgrid only stuff. leave diagnostic only stuff for
# use on acquisition computer.
analyze_glomeruli_diagnostics_only = False
print_skipped = False

is_acquisition_host = util.is_acquisition_host()
if is_acquisition_host:
    analyze_ijrois = False
    final_pair_concentrations_only = False
    do_suite2p = False
    analyze_suite2p_outputs = False
    write_processed_tiffs = False
    want_dff_tiff = False
    # Non-admin users don't have permission to make (the equivlant of?) symbolic links
    # in Windows, and I'm not sure what all the ramifications of enabling that would
    # be.
    links_to_input_dirs = False
    do_convert_raw_to_tiff = False

    min_input = 'raw'

    do_register_all_fly_recordings_together = False

###################################################################################
# Modified inside `process_recording`
###################################################################################

# TODO maybe convert to dict -> None (+ conver to set after
# process_recording loop) (since multiprocessing.Manager doesn't seem to have a set)
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
# TODO also agg + cache full traces (maybe just use CSV? also for above)
# TODO rename to ij_trialstat_dfs or something
ij_trial_dfs = []

# Using dict rather than defaultdict(list) so handling is more consistent in case when
# multiprocessing DictProxy overrides this.
names_and_concs2analysis_dirs = dict()

flies_with_new_processed_tiffs = []

# TODO refactor all the paired dir handling to a Recording object?
#
# (date, fly, panel, is_pair) -> list of directory pairs that are the recordings for
# that experiment
#
# NOTE: currently only populating for stuff not skipping (via early return) in
# process_recording.
experiment2recording_dirs = dict()

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


def get_fly_analysis_dir(date, fly_num) -> Path:
    """Returns path for storing fly-level (across-recording) analysis artifacts

    Creates the directory if it does not exist.
    """
    fly_analysis_dir = analysis_intermediates_root / util.get_fly_dir(date, fly_num)
    fly_analysis_dir.mkdir(exist_ok=True, parents=True)
    return fly_analysis_dir


# TODO replace similar fn (if still exists?) already in hong2p? or use the hong2p one?
# (just want to prefer the "fast" data root)
def get_analysis_dir(date, fly_num, thorimage_dir) -> Path:
    """Returns path for storing recording-level analysis artifacts

    Args:
        thorimage_dir: either a path ending in a ThorImage directory, or just the last
            part of the path to one

    Creates the directory if it does not exist.
    """
    # TODO switch to pathlib
    thorimage_basedir = split(thorimage_dir)[1]
    analysis_dir = get_fly_analysis_dir(date, fly_num) / thorimage_basedir
    # TODO probably replace w/ my makedirs to be consistent w/ empty dir cleanup
    analysis_dir.mkdir(exist_ok=True, parents=True)
    return analysis_dir


# TODO change rest of code to only compute within each (driver, indicator) combination
# (but to also not require separate runs for each combination)
_last_driver_indicator_combo = None
def driver_indicator_output_dir(driver: str, indicator: str) -> Path:
    """Returns path containing driver+indicator for outputs/plots, under current dir.

    Directory returned is for all flies sharing the same driver+indicator combination,
    not just the specific fly passed in.

    Raises NotImplementedError if fly's driver or indicator (from Google sheet) differ
    from other flies seen.
    """
    global _last_driver_indicator_combo

    driver_indicator_combo = (driver, indicator)
    if _last_driver_indicator_combo is not None:
        if _last_driver_indicator_combo != driver_indicator_combo:
            raise NotImplementedError('analyzing multiple driver+indicator combinations'
                ' in a single run is not supported. pass matching_substrs CLI arg to '
                'restrict analysis to a particular combo.'
            )
    else:
        _last_driver_indicator_combo = driver_indicator_combo

    # To not make directories with weird characters if I forget to edit any of these
    # out.
    assert '?' not in driver, driver
    assert '?' not in indicator, indicator

    # Makes in current directory
    output_dir = Path(f'{driver}_{indicator}')
    makedirs(output_dir)
    return output_dir


def output_dir2driver(path: Path) -> str:
    parts = path.name.split('_')
    if len(parts) != 2:
        raise ValueError(f'path {path} did not have 2 components. expected '
            "'<driver>_<indicator>' directory name, with neither driver nor indicator "
            'containing an underscore'
        )

    driver, indicator = parts
    return driver


# TODO TODO let root of this be overridden by env var, so i can put it on external hard
# drive if i'm running on my space limited laptop
def get_plot_root(driver, indicator) -> Path:
    return driver_indicator_output_dir(driver, indicator) / plot_fmt


def fly2driver_indicator(date, fly_num) -> Tuple[str, str]:
    """Returns tuple with driver and indicator of fly, via Google sheet metadata lookup
    """
    fly_row = gdf.loc[(pd.Timestamp(date), int(fly_num))]
    return fly_row.driver, fly_row.indicator


def fly2driver_indicator_output_dir(date, fly_num) -> Path:
    """Looks up driver+indicator of fly and returns path for outputs/plots.

    Directory returned is for all flies sharing the same driver+indicator combination,
    not just the specific fly passed in.

    Raises NotImplementedError if fly's driver or indicator (from Google sheet) differ
    from other flies seen.
    """
    driver, indicator = fly2driver_indicator(date, fly_num)
    return driver_indicator_output_dir(driver, indicator)


def fly2plot_root(date, fly_num) -> Path:
    """Returns <driver>_<indicator>/<plot_fmt> directory to save plots under.

    Makes directory if it does not exist.
    """
    plot_root = fly2driver_indicator_output_dir(date, fly_num) / plot_fmt
    makedirs(plot_root)
    return plot_root


# TODO factor to hong2p (renaming to be more general)
# TODO maybe this shouldn't be strict on input types tho?
def keys2rel_plot_dir(date: pd.Timestamp, fly_num: int, thorimage_id: str) -> Path:
    """Returns relative path for plot dir given fixed type (date, fly_num, thorimage_id)

    Args:
        thorimage_id: must just be the final part of the ThorImage path (contain no '/')
    """
    assert len(Path(thorimage_id).parts) == 1, \
        f'{thorimage_id} contained path separator'

    return f'{format_date(date)}_{fly_num}_{thorimage_id}'


# TODO modify to only accept date, fly, thorimage_id like other similar fns in hong2p?
def get_plot_dir(date, fly_num, thorimage_id: str) -> Path:
    """
    Does NOT currently work with thorimage_id containing directory seperator. Must just
    contain the name of the terminal directory.
    """
    rel_plot_dir = keys2rel_plot_dir(date, fly_num, thorimage_id)
    plot_dir = fly2plot_root(date, fly_num) / rel_plot_dir

    makedirs(plot_dir)

    return plot_dir


def ijroi_plot_dir(plot_dir: Path) -> Path:
    return plot_dir / 'ijroi'


def suite2p_plot_dir(plot_dir: Path) -> Path:
    # TODO test doesn't break stuff
    return plot_dir / 'suite2p_roi'


# TODO sort name1, name2 first?
# TODO rename this or the ...rel_plot_dir to be consistent (both just last part of dir)
def get_pair_dirname(name1, name2) -> str:
    # TODO did i not have this under like a 'pair' subdir?
    return names2fname_prefix(name1, name2)


# TODO delete? sort_odors below not do what i wanted in some pair data stuff?
def sort_concs(df: pd.DataFrame) -> pd.DataFrame:
    return olf.sort_odors(df, sort_names=False)


# TODO also let this take a list of odors? or somehow use output + input to do something
# that is effectively like an argsort, and use that to index some other type of object
# (where we convert it to a DataFrame just for sorting)
def sort_odors(df: pd.DataFrame, add_panel: Optional[str] = None) -> pd.DataFrame:

    # TODO add to whichever axis has odor info automatically? or too complicated.
    # currently doesn't work if odors are in columns.
    if add_panel is not None:
        # TODO assert 'panel' not already in (relevant axis) index level names
        df = util.addlevel(df, 'panel', add_panel)

    return olf.sort_odors(df, panel_order=panel_order,
        panel2name_order=panel2name_order, if_panel_missing=None
    )


# TODO flag to select whether ROI or (date, fly) take priority?
# TODO move to hong2p + test
def sort_fly_roi_cols(df: pd.DataFrame, flies_first: bool = False, sort_first_on=None
    ) -> pd.DataFrame:
    # TODO delete key if i can do w/o it (by always just sorting a second time when i
    # want some outer level)
    """Sorts column MultiIndex with ['date','fly_num','roi'] levels.

    Args:
        df: data to sort by fly/ROI column values

        flies_first: if True, sorts on ['date', 'fly_num'] columns primarily, followed
            by 'roi' ROI names.

        sort_first_on: sequence of same length as df.columns, used to order ROIs.
            Within each level of this key, will sort on the default date/fly_num ->
            with higher priority than roi, but will then group all "named" ROIs before
            all numbered/autonamed ROIs.
    """
    index_names = df.columns.names
    assert 'roi' in index_names or 'roi' == df.columns.name

    levels = ['not_named', 'roi']
    if 'date' in index_names and 'fly_num' in index_names:
        if not flies_first:
            levels = levels + ['date', 'fly_num']
        else:
            levels = ['date', 'fly_num'] + levels

    levels_to_drop = []
    to_concat = [df.columns.to_frame(index=False)]

    assert 'not_named' not in df.columns.names
    not_named = df.columns.get_level_values('roi').map(
        lambda x: not is_ijroi_named(x)).to_frame(index=False, name='not_named')

    levels_to_drop.append('not_named')
    to_concat.append(not_named)

    if sort_first_on is not None:
        # NOTE: for now, just gonna support this being of-same-length as df.columns
        assert len(sort_first_on) == len(df.columns)

        # Seems to also work when input is a list of tuples (so you can list(zip(...))
        # multiple iterables of keys, in the order you want them to take priority).
        sort_first_on = pd.Series(list(sort_first_on), name='_sort_first_on').to_frame()

        levels = ['_sort_first_on'] + levels
        levels_to_drop.append('_sort_first_on')
        to_concat.append(sort_first_on)

    df.columns = pd.MultiIndex.from_frame(pd.concat(to_concat, axis='columns'))

    # TODO get numbers to actually sort like numbers, for any numbered ROIs
    # (or maybe just set name to NaN there, just for the sort, and rely on them already
    # being in order?) (would probably have to sort the numbered section separately from
    # the named one, casting the numbered ROI names to ints)

    # The order of level here determines the sort-priority of each level.
    return df.sort_index(level=levels, sort_remaining=False, kind='stable',
        axis='columns').droplevel(levels_to_drop, axis='columns')


recording_col = 'thorimage_id'

def drop_redone_odors(df: pd.DataFrame) -> pd.DataFrame:
    """Drops all but last recording for each (panel, odor) combo.
    """
    assert recording_col in df.columns.names

    # TODO TODO warn / err if any panel values are NaN? likely to cause issues...

    def drop_single_fly_redone_odors(fly_df):
        fly_df = fly_df.copy()

        recording_has_odor = fly_df.groupby('thorimage_id', axis='columns',
            sort=False).apply(lambda x: x.notna().any(axis='columns'))

        redone_odors = fly_df[recording_has_odor.sum(axis='columns') > 1].index

        # should only be 1 element each each of these unique(...) outputs
        date = fly_df.columns.unique('date')[0]
        fly_num = fly_df.columns.unique('fly_num')[0]

        # TODO make less shitty? don't want to have to hard code indices like this...
        # looping over index elements below just gives us tuples (of row index values)
        # tho.
        assert fly_df.index.names[0] == 'panel'
        assert fly_df.index.names[2:4] == ['odor1', 'odor2']

        recordings = fly_df.columns.get_level_values('thorimage_id').unique()

        for panel_odor in redone_odors:
            recording_has_curr_odor = recording_has_odor.loc[panel_odor]
            final_recording = recording_has_curr_odor[::-1].idxmax()

            # TODO TODO TODO how to warn about which ones we are tossing this way tho???
            nonfinal_recordings= (
                fly_df.columns.get_level_values('thorimage_id') != final_recording
            )

            panel = panel_odor[0]
            # NOTE: ignoring 'repeat' (which might be different across recordings in
            # someone elses use, but isn't in any of my data)
            odors = panel_odor[2:4]

            nonfinal_recording_set = (
                set(recording_has_curr_odor[recording_has_curr_odor].index)
                - {final_recording}
            )

            # TODO consistent formatting of nonfinal_recording_set (wrt final_recording,
            # at least)
            warn(f'{format_date(date)}/{fly_num} (panel={panel}): dropping '
                f'{format_mix_from_strs(odors)} from {nonfinal_recording_set} '
                f'(redone in {final_recording})'
            )

            fly_df.loc[panel_odor, nonfinal_recordings] = np.nan

        return fly_df

    # TODO warn if any column levels other than these (+ 'roi')
    # panel?
    #
    # This is to merge across across multiple values for recording_col
    # (for a each combination of group keys), which is what we have when a fly has
    # multiple recordings
    df = df.groupby(['date','fly_num'], axis='columns', sort=False
        ).apply(drop_single_fly_redone_odors)

    # TODO TODO does this check both columns (.columns) and rows (.index)?
    util.check_index_vals_unique(df)

    return df


# TODO prob move to hong2p
def merge_rois_across_recordings(df: pd.DataFrame) -> pd.DataFrame:
    """Merges ROI columns across recordings (thorimage_id level)

    Merges within each unique combination of ['date','fly_num','roi']
    """
    assert recording_col in df.columns.names

    def merge_single_flyroi_across_recordings(gdf):
        # recording_col is the only thing that should really be varying here,
        # except maybe 'panel' (which should vary <= as much as recording_col).
        #
        # This at least ensures nothing else is varying within a particular value
        # for recording_col.
        assert gdf.shape[1] == len(
            gdf.columns.get_level_values(recording_col).unique()
        )

        # As long as this doesn't trip, we don't have to worry about choosing which
        # column to take data from: there will only ever be at most one not NaN.
        assert not (gdf.notna().sum(axis='columns') > 1).any()

        # TODO rename if ser isn't actually a series
        ser = gdf.bfill(axis='columns').iloc[:, 0]
        return ser

    # TODO factor this kind of #-notnull-preserving check into a decorator?
    n_before = df.notnull().sum().sum()

    # TODO warn if any column levels other than these and recording_col?
    #
    # This is to merge across across multiple values for recording_col
    # (for a each combination of group keys), which is what we have when a fly has
    # multiple recordings (but the ROIs definitions for each recording are the same
    # / overlapping).
    df = df.groupby(['date','fly_num','roi'], axis='columns', sort=False
        ).apply(merge_single_flyroi_across_recordings)

    assert df.notnull().sum().sum() == n_before
    assert recording_col not in df.columns.names

    util.check_index_vals_unique(df)

    return df


def drop_superfluous_uncertain_rois(df: pd.DataFrame) -> pd.DataFrame:
    # TODO doc!

    # TODO warn once at end, after building up a more nicely formatted error message?

    def single_fly_drop_uncertain_if_we_have_certain(fly_df):

        def _single_flyroi_drop(roi_df):
            if roi_df.shape[1] == 1:
                return roi_df

            # .map gives us an Index w/ boolean vals, but can't negate it w/ ~
            certain = np.array(
                roi_df.columns.get_level_values('roi').map(is_ijroi_certain),
                dtype=bool
            )
            if not certain.any():
                return roi_df

            # TODO delete this or outer
            assert roi_df.columns.names[:2] == ['date', 'fly_num']
            date, fly_num = roi_df.columns[0][:2]
            date_str = format_date(date)
            fly_str = f'{date_str}/{fly_num}'

            dropping = roi_df.columns.get_level_values('roi')[~certain]

            # TODO how annoying will this get?
            # (a bit. depends how easy they are to resolve...)
            # TODO TODO make summary CSV?
            warn(f'{fly_str}: dropping ROIs {"".join(dropping)} because matching '
                'certain ROI existed'
            )

            # NOTE: this was my original implementation attempt, but it led to some
            # confusing errors (inside concat step of one of [outermost?] the enclosing
            # groupbys). some examples online return indexed versions of group data so
            # I'm not sure the cause is straightforward...
            # (AssertionError: Cannot concat indices that do not have the same number of
            # levels)
            #return roi_df.loc[:, certain]

            roi_df = roi_df.copy()
            roi_df.loc[:, ~certain] = np.nan
            return roi_df

        # TODO less hacky way? (don't want to have to modify index to add a temporary
        # variable to group on)
        roi_index = fly_df.columns.names.index('roi')
        group_fn = lambda index_tuple: ijroi_name_as_if_certain(index_tuple[roi_index])

        # TODO need to handle dropna=False case (in fn above?)? just set to True?
        # if i did, couldn't plot stuff like 'VM2|VM3' later... not that i actually
        # have any of that in data i'm analyzing now...
        return fly_df.groupby(group_fn, axis='columns', sort=False, dropna=False).apply(
            _single_flyroi_drop
        )

    df = df.groupby(['date','fly_num'], axis='columns', sort=False).apply(
        single_fly_drop_uncertain_if_we_have_certain
    )
    # This is what will actually reduce the size along the column axis
    # (if any will be dropped)
    df = dropna(df)

    assert df.columns.names == ['date', 'fly_num', 'roi']
    fly_rois = df.columns.to_frame(index=False)
    fly_rois['name_as_if_certain'] = df.columns.get_level_values('roi').map(
        ijroi_name_as_if_certain
    )
    # TODO TODO TODO fix issue probably added (2023-10-29) by editing 2023-05-09/1 (or
    # fly edited before that?) (probably by 'x?' and 'x??' or something...)
    # Traceback (most recent call last):
    #   File "./al_analysis.py", line 10534, in <module>
    #     main()
    #   File "./al_analysis.py", line 10119, in main
    #     trial_df = drop_superfluous_uncertain_rois(trial_df)
    #   File "./al_analysis.py", line 1095, in drop_superfluous_uncertain_rois
    #     assert (
    # AssertionError
    try:
        assert (
            len(fly_rois[['date','fly_num','roi']].drop_duplicates()) ==
            len(fly_rois[['date','fly_num','name_as_if_certain']].drop_duplicates())
        )
    except AssertionError as err:
        warn(err)
    # TODO restore after dealing w/ only exception:
    # when name_as_if_certain=None (b/c there were multiple parts)
    # (although if there were ever >=2 of these in one fly, above assertion would fail
    # first anyway)
    #assert not fly_rois.isna().any().any()

    util.check_index_vals_unique(df)
    return df


def paired_thor_dirs(*args, **kwargs):
    return util.paired_thor_dirs(*args, ignore_prepairing=('anat',), **kwargs)


# TODO factor into hong2p.thor
# TODO rename to load_*, if that would be more consistent w/ what i already have
def read_thor_tiff_sequence(thorimage_dir: Path) -> xr.DataArray:
    # Needs to have at least two '_' characters, to exclude stuff like:
    # 'ChanA_Preview.tif'
    tiffs = sorted(thorimage_dir.glob('Chan*_*_*.tif'),
        key=lambda x: x.name.split('_')
    )

    # TODO assert len tiffs expected from thorimage xml metadata
    # how to get whether second channel is enabled again?
    # (or probably check both channels, cause maybe just red is enabled?)
    xml = thor.get_thorimage_xmlroot(thorimage_dir)

    n_zstream_frames = thor.get_thorimage_z_stream_frames(xml)

    (nx, ny), nz, n_channels = thor.get_thorimage_dims(xml)

    # TODO delete
    '''
    nz = thor.get_thorimage_z_xml(xml)

    # TODO could also check Wavelength elements under Wavelengths (and maybe
    # ChannelEnable at same level) to see which were actually selected to record,
    # but for now assuming if a PMT was enabled and had gain > 0, it was recorded
    pmt = xml.find('PMT').attrib
    # TODO factor into hong2p.thor (some similar stuff currently inside
    # thor.get_thorimage_n_channels_xml)
    def is_channel_enabled(channel):
        enabled = bool(int(pmt[f'enable{channel}']))
        # TODO warn if enabled but gain == 0?
        gain = int(pmt[f'gain{channel}'])
        return enabled and gain > 0

    a_enabled = is_channel_enabled('A')
    b_enabled = is_channel_enabled('B')
    n_channels = sum([a_enabled, b_enabled])
    '''

    frame_metadata = defaultdict(list)
    frames = []

    # TODO maybe use part of thorimage_dir in desc instead?
    for tiff in tqdm(tiffs, total=len(tiffs), unit='frame',
        desc='loading TIFF sequence', leave=False):

        # TODO factor metadata parsing (from tiff fname) into its own fn
        parts = tiff.stem.split('_')

        channel_part = parts[0]
        assert channel_part.startswith('Chan') and len(channel_part) == 5
        # Should be 'A' or 'B' from e.g. 'ChanA_...'
        thor_chan = channel_part[-1]
        if thor_chan == 'A':
            channel = 'green'

        elif thor_chan == 'B':
            channel = 'red'
        else:
            raise ValueError('unexpected ThorImage channel: {thor_chan}')

        z = int(parts[-2]) - 1
        z_stream_index = (int(parts[-1]) - 1) % n_zstream_frames
        assert 0 <= z_stream_index < n_zstream_frames, z_stream_index

        frame_metadata['channel'].append(channel)
        frame_metadata['z'].append(z)
        frame_metadata['z_stream_index'].append(z_stream_index)
        # TODO delete (unless this is more useful in getting a DataArray constructed
        # in a way useful for reshaping)
        '''
        frame_metadata.append({
            'channel': channel,
            'z': z,
            'z_stream_index': z_stream_index,
        })
        '''

        # is_ome=False to disable processing of OME metadata in the first TIFF in
        # each channel x Z-stream index combination.
        frame = tifffile.imread(tiff, is_ome=False)
        frames.append(frame)

    # TODO TODO how to do while reshaping in xarray, using frame_metadata?
    movie = np.stack(frames).reshape((n_channels, nz, n_zstream_frames, ny, nx))

    movie = xr.DataArray(movie, dims=['c','z','t','y','x'])

    # TODO factor into hong2p.xarray (already used in at least one other place here)
    movie = movie.assign_coords({n: np.arange(movie.sizes[n]) for n in movie.dims})

    return movie



# TODO some of the registration tiff handling code benefit from this?
def find_movie(*keys, tiff_priority=('mocorr', 'flipped', 'raw'), min_input=min_input,
    verbose=False) -> Path:

    assert min_input is None or min_input in tiff_priority

    analysis_dir = get_analysis_dir(*keys)

    # TODO modify to load binary from suite2p if available, and if mocorr tiff is not
    # available
    for tiff_prefix in tiff_priority:

        tiff_path = analysis_dir / f'{tiff_prefix}.tif'
        if verbose:
            print(f'checking for {shorten_path(tiff_path, n_parts=4)}...', end='')

        if tiff_path.exists():
            if verbose:
                # See https://unicode-table.com / unicodedata stdlib module for more fun
                print(u'\N{WHITE HEAVY CHECK MARK}')

            return tiff_path

        if verbose:
            # TODO replace w/ something red?
            print(u'\N{HEAVY MULTIPLICATION X}')

        if tiff_prefix == min_input:

            # Want to just try thor.read_movie in this case, b/c if we would accept the
            # raw TIFF, we should also accept the (what should be) equivalent ThorImage
            # .raw file
            # TODO is this really how i want to handle it?
            if tiff_prefix == 'raw':
                break

            raise IOError(f"did not have a TIFF of at least {min_input=} status")

    thorimage_dir = util.thorimage_dir(*keys)

    # would add complication to get path to raw file here, and don't currently actually
    # care about path in that case, so just returning containing directory
    return thorimage_dir


# TODO factor to hong2p.util (along w/ get_analysis_dir, removing old analysis dir
# stuff for it [which would ideally involve updating old code that used it...])
def load_movie(*args, **kwargs):

    tiff_path_or_thorimage_dir = find_movie(*args, **kwargs)

    if tiff_path_or_thorimage_dir.name.endswith('.tif'):
        return tifffile.imread(tiff_path_or_thorimage_dir)
    else:
        assert tiff_path_or_thorimage_dir.is_dir()
        return thor.read_movie(thorimage_dir)


# TODO TODO move to hong2p.util
# TODO unit test?
#
# TODO default verbose=None and try to use default of wrapped fn then
# (or True otherwise?)
# (still need to test behavior when wrapped fn has existing verbose kwarg)
# TODO make this an attribute of this/one of inner fns (rather than module level)?
_fn2seen_inputs = dict()
def produces_output(_fn=None, *, verbose=True):
    # for how to make a decorator with optional  arg:
    # https://realpython.com/primer-on-python-decorators

    # TODO what would be a good name for this?
    def wrapper_helper(fn):

        assert fn.__name__ not in _fn2seen_inputs, (
            'seen set would have been overwritten'
        )
        # TODO some reason to use lists like i was in savefig? was that just for easier
        # use in multiprocessing access (no set equiv of IPC data type?)?
        # that matter anymore?
        _fn2seen_inputs[fn.__name__] = set()

        @wraps(fn)
        def wrapped_fn(data, path: Pathlike, *args, verbose=verbose, **kwargs):
            # TODO easy to check type of matching positional arg is Path/Pathlike
            # (if specified)?
            # see: https://stackoverflow.com/questions/71082545 for one way

            # TODO add option (for use during debugging) that checks outputs
            # have not changed since last run (to the extent the format allows it...)

            assert fn.__name__ in _fn2seen_inputs

            seen_inputs = _fn2seen_inputs[fn.__name__]
            # TODO want to have wrapper add a kwarg to disable this assertion?
            # TODO test w/ both Path and str (and mix)
            assert path not in seen_inputs, (f'would have overwritten output {path} ('
                'previously written elsewhere in this run)!'
            )
            seen_inputs.add(path)

            # TODO test! (and test arg kwarg actually useable on wrapped fn, whether or
            # not already wrapped fn has this kwarg. can start by assuming it doesn't
            # have this kwarg tho...)!
            #
            # (have already manually tested cases where wrapped fns do not have existin
            # verbose= kwarg. just need to test case where wrapped fn DOES have existing
            # verbose= kwarg now.)
            if verbose:
                # TODO test w/ Path and str name input (that output looks good)
                print(f'writing {path}')

            return fn(data, path, *args, **kwargs)

        return wrapped_fn

    if _fn is None:
        return wrapper_helper
    else:
        return wrapper_helper(_fn)


@produces_output
# input could be at least Series|DataFrame
def to_csv(data, path: Pathlike, **kwargs) -> None:
    data.to_csv(path, **kwargs)

@produces_output(verbose=False)
# input could be at least Series|DataFrame
def to_pickle(data, path: Pathlike) -> None:
    data.to_pickle(path)

# TODO check this behaves as verbose=True
# (esp if that fn already has verbose kwarg in natmix. want to test that case)
write_corr_dataarray = produces_output(_write_corr_dataarray)

# TODO TODO also use wrapper for TIFFs (w/ [what should be default] verbose=True)
# (wrap util.write_tiff, esp if that fn already has verbose kwarg. want to test that
# case)

# TODO CLI flag to (or just always?) warn if there are old figs in any/some of the dirs
# we saved figs in (would only want if very verbose...)?
# TODO maybe refactor to automatically prefix path with '{plot_fmt}/' in here?
#
# Especially running process_recording in parallel, the many-figures-open memory
# warning will get tripped at the default setting, hence `close=True`.
#
# sns.FacetGrid and sns.ClusterGrid should both be subclasses of sns.axisgrid.Grid
# (can't seem to import sns.ClusterGrid anyway... maybe it's somewhere else?)
# TODO try to replace logic w/ this decorator
#@produces_output(verbose=False)
_savefig_seen_paths = set()
def savefig(fig_or_seaborngrid: Union[Figure, Type[sns.axisgrid.Grid]],
    fig_dir: Pathlike, desc: str, *, close: bool = True, **kwargs) -> Path:

    # TODO actually modify to_filename to not throw out '.', and manually remove that in
    # any remaining cases where i didn't want it? for concentrations like '-3.5', this
    # makes them more confusing to read... (-> '-35')
    basename = util.to_filename(desc) + plot_fmt
    # TODO delete / fix
    makedirs(fig_dir)
    #
    fig_path = Path(fig_dir) / basename

    # TODO share logic w/ to_csv above
    abs_fig_path = fig_path.resolve()

    # TODO delete try/except (can i repro failure?)
    # 2023-12-04:
    # $ ./al_analysis.py -t 2022-02-03 -e 2022-04-03 -v -s model
    #thorimage_dir: 2022-02-22/1/kiwi_ea_eb_only
    #thorsync_dir: 2022-02-22/1/SyncData003
    #yaml_path: 20220222_184517_stimuli/20220222_184517_stimuli_0.yaml
    #TIFF (/ motion correction) changed. updating non-ROI outputs.
    #ImageJ ROIs were modified. re-analyzing.
    #...
    #merging ROI VM7d?
    #selecting input ROI 10 as best plane
    #dropping other input ROIs [9]
    #           roi_quality
    #roi_index
    #9             0.049183
    #10            0.054798
    #
    #Uncaught exception
    #Traceback (most recent call last):
    #  File "./al_analysis.py", line 10766, in <module>
    #    main()
    #  File "./al_analysis.py", line 9580, in main
    #    was_processed = list(starmap(process_recording, keys_and_paired_dirs))
    #  File "./al_analysis.py", line 3949, in process_recording
    #    ij_trial_df, best_plane_rois, full_rois = ij_trace_plots(analysis_dir,
    #  File "./al_analysis.py", line 3096, in ij_trace_plots
    #    trial_df = trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
    #  File "./al_analysis.py", line 3044, in trace_plots
    #    savefig(fig, roi_plot_dir, str(roi))
    #  File "./al_analysis.py", line 1406, in savefig
    #    assert abs_fig_path not in _savefig_seen_paths
    #AssertionError
    try:
        assert abs_fig_path not in _savefig_seen_paths
    except AssertionError:
        print(f'{abs_fig_path=}')
        print(f'{desc=}')
        import ipdb; ipdb.set_trace()
    #
    # TODO delete. should be what a duplicate gets saved of above.
    if abs_fig_path.name == 'DP1l.png':
        print(f'{abs_fig_path=}')
        print(f'{desc=}')
        import ipdb; ipdb.set_trace()
    #

    _savefig_seen_paths.add(abs_fig_path)

    if save_figs:
        fig_or_seaborngrid.savefig(fig_path, **kwargs)

    fig = None
    if isinstance(fig_or_seaborngrid, Figure):
        fig = fig_or_seaborngrid

    elif isinstance(fig_or_seaborngrid, sns.axisgrid.Grid):
        fig = fig_or_seaborngrid.fig

    # TODO cli flag for this?
    debug = False
    if debug:
        print(fig_path)
        assert fig is not None
        print(f'figsize={tuple(fig.get_size_inches())}, dpi={fig.dpi}')
        print()

    if close:
        if fig is None:
            raise ValueError(
                f'cannot close unknown plot object of type={type(fig_or_seaborngrid)}'
            )
        plt.close(fig)

    return fig_path


_dirs_to_delete_if_empty = []
# TODO work w/ pathlib input?
def makedirs(d):
    """Make directory if it does not exist, and register for deletion if empty.
    """
    # TODO make sure if we make a directory as well as some of its parent directories,
    # that if (all) the leaf dirs are empty, the whole empty tree gets deleted
    # TODO shortcircuit to returning if we already made it this run, to avoid the checks
    # on subsequent calls? they probably aren't a big deal though...
    os.makedirs(d, exist_ok=True)
    # TODO only do this if we actually made the directory in the above call?
    # not sure we really care for empty dirs in any circumstances tho...
    _dirs_to_delete_if_empty.append(d)


# TODO work w/ pathlib input?
def delete_if_empty(d):
    """Delete directory if empty, do nothing otherwise.
    """
    # TODO don't we still want to delete any broken links / links to empty dirs?
    if not exists(d) or islink(d):
        return

    if not any(os.scandir(d)):
        os.rmdir(d)


def delete_empty_dirs():
    """Deletes empty directories in `_dirs_to_delete_if_empty`
    """
    for d in set(_dirs_to_delete_if_empty):
        delete_if_empty(d)


# TODO probably need a recursive solution combining deletion of empty symlinks and
# directories to cleanup all hierarchies that could be created w/ symlink and makedirs

# TODO maybe just pick relative/abs based on whether target is under plot_dir (also
# probably err if link is not under plot_dir), because the whole reason we wanted some
# links relative is so i could move the whole plot directory and have the (internal,
# relative) links still be valid, rather than potentially pointing to plots generated in
# the original path after (relative=None, w/ True/False set based on this)
# (also want stuff linking from&to analysis root to be relative too, also for copying)
links_created = []
# TODO maybe support link being a dir (that exists), target being a file, and then
# use basename of file in new dir by default?
# TODO complain early on if filesystem HONG2P_DATA / HONG2P_FAST_DATA point to do not
# support symlinks (probably at init, not in this fn)
def symlink(target, link, relative=True, checks=True, replace=False):
    """Create symlink link pointing to target, doing nothing if link exists.

    Also registers `link` for deletion at end if what it points to no
    """
    # TODO err if link exists and was created *in this same run* (indicating trying to
    # point to multiple different outputs from the same link; a bug)
    target = Path(target)
    link = Path(link)

    # Will slightly simplify cleanup logic by mostly ensuring broken links only come
    # from deleted directories.
    if not target.exists():
        raise FileNotFoundError

    # TODO delete
    verbose = False
    if verbose:
        print('input:')
        print(f'target={target}')
        print(f'link={link}')
        print(f'{target.is_dir()=}')
        print(f'{link.is_dir()=}')
    #
    if relative:
        # TODO delete if the old link_dir code was actually useful...
        # (if i run all parts of my analysis that make symlinks fresh and this doesn't
        # trigger, can delete)
        assert not (link.is_dir() and not link.is_symlink())

        # seemed to work for some(/all? unclear...) uses, but would cause my
        # link-already-exists checks to fail...
        link_dir = link.parent

        # TODO delete
        if verbose:
            print(f'{link_dir=}')
            print(f'{os.path.relpath(target, link_dir)=}')
        #

        # From pathlib docs: "PurePath.relative_to() requires self to be the subpath of
        # the argument, but os.path.relpath() does not."
        # ...so probably can't use it as a direct replacement here.
        # TODO test this behaves correctly. depend on whether target is a dir/not?
        target = Path(os.path.relpath(target, link_dir))
    else:
        # Because relative paths are resolved wrt current working directory, not wrt
        # directory of target (or link) (same w/ os.path.abspath)
        assert target.is_absolute()

        # TODO do i even want to do this? isn't it just modifying relative components
        # inside of an absolute path OR paths containing symlinks at this point? i don't
        # think i have the former and i don't know if i would want symlinks resolved...
        #
        # From pathlib docs: "os.path.abspath() does not resolve symbolic links while
        # Path.resolve() does."
        # Not sure if relevant to any of my use cases.
        target = target.resolve()

    # TODO delete
    if verbose:
        print('final:')
        print(f'target={target}')
        print(f'link={link}')
    #

    def check_written_link():
        resolved_link = link.resolve()

        if target.is_absolute():
            resolved_target = target.resolve()
        else:
            resolved_target = (link.parent / target).resolve()

        # TODO delete try/except
        try:
            assert resolved_link == resolved_target, (f'link: {link}\n'
                f'target: {target}\n{resolved_link} != {resolved_target}'
            )
        except AssertionError as err:
            print()
            print(str(err))
            import ipdb; ipdb.set_trace()

    # TODO delete?
    # TODO maybe this should just be the default? why not? maybe warn if replacing?
    #replace = True

    # NOTE: link.exists() will return False for broken symlinks, but link.is_symlink()
    # will return True.
    if link.is_symlink():
        if replace or not link.exists():
            link.unlink()
        else:
            if checks:
                check_written_link()

            return

    link.symlink_to(target)
    if checks:
        # This should fail if the link is broken right after creation
        assert link.is_symlink()
        assert link.resolve().exists(), f'link broken! ({link} -> {target})'

    # TODO delete?
    if checks:
        check_written_link()
    #

    links_created.append(link)

    # TODO delete
    if verbose:
        print()
    #


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
        for d in _dirs_to_delete_if_empty:
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
def make_fail_indicator_file(analysis_dir: Path, suffix: str, err=None):
    """Makes empty file (e.g. FAILING_suite2p) in analysis_dir, to mark step as failing
    """
    path = analysis_dir / f'{FAIL_INDICATOR_PREFIX}{suffix}'
    if err is None:
        path.touch()
    elif type(err) is str:
        err_str = err
    else:
        err_str = ''.join(traceback.format_exception(type(err), err, err.__traceback__))

    path.write_text(err_str)


def _list_fail_indicators(analysis_dir: Path):
    return glob.glob(str(analysis_dir / (FAIL_INDICATOR_PREFIX + '*')))


def last_fail_suffixes(analysis_dir: Path) -> Tuple[bool, Optional[List[str]]]:
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


def clear_fail_indicators(analysis_dir: Path) -> None:
    """Deletes any fail indicator files in analysis_dir
    """
    for f in _list_fail_indicators(analysis_dir):
        os.remove(f)


def should_ignore_existing(name: str, explicit_only: bool = False) -> bool:
    """
    Args:
        name: step to check whether it should be recomputed.
            see `ignore_existing_options`.

        explicit_only: will only ignore `name` if it is in `ignore_existing`, not if
            `ignore_existing == True`
    """
    if type(ignore_existing) is bool:

        # Some steps are unlikely enough to need recomputation, we will only do so if we
        # explicitly request that step be recomputed.
        if name in ignore_if_explicitly_requested:
            return False

        return ignore_existing
    else:
        assert name in ignore_existing_options
        return name in ignore_existing


# TODO factor into hong2p maybe (+ add support for xarray?)
def dropna(df: pd.DataFrame, how: str = 'all', cols_first: bool = True, _checks=True
    ) -> pd.DataFrame:
    """Drops rows/columns where all values are NaN.
    """
    assert how in ('all', 'any')

    if how == 'all' and _checks:
        notna_before = df.notna().sum().sum()

    # TODO need to alternate (i.e. does order ever matter? ever not idempotent?)?
    if cols_first:
        # Not sure whether axis='rows' has always been supported, but it seems to behave
        # same as axis='index'... Was originally a mistake, but more clear I think.
        df = df.dropna(how=how, axis='columns').dropna(how=how, axis='rows')
    else:
        df = df.dropna(how=how, axis='rows').dropna(how=how, axis='columns')

    if how == 'all' and _checks:
        assert df.notna().sum().sum() == notna_before

    return df


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


# TODO TODO TODO test this works even if there are e.g. extra solvent presentations
# (and anything else that was in the kiwi ea/eb only + similar experiments)
def is_pairgrid(odor_lists: ExperimentOdors):
    # TODO reimplement in a way that actually checks there are all pairwise
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


def odor_strs2single_odor_name(index):
    odors = {olf.parse_odor_name(x) for x in index}
    # None corresponds to input that was equal to hong2p.olf.solvent_str
    odors = {x for x in odors if x is not None}
    assert len(odors) == 1
    return odors.pop()


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
    keys_and_paired_dirs = paired_thor_dirs(verbose=False, **paired_thor_dirs_kwargs)

    seen_stimulus_yamls2thorimage_dirs = defaultdict(list)
    names2final_concs = dict()
    names_and_concs_tuples = []
    for (_, _), (thorimage_dir, _) in keys_and_paired_dirs:

        xml = thor.get_thorimage_xmlroot(thorimage_dir)
        ti_time = thor.get_thorimage_time_xml(xml)

        try:
            yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(
                xml
            )
        except NoStimulusFile as err:
            # TODO still do this if verbose
            #warn(f'{err}. skipping.')
            continue

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


# TODO also factor to hong2p
bounding_frame_yaml_cache_basename = 'trial_bounding_frames.yaml'
def has_cached_frame_odor_assignments(analysis_dir: Pathlike) -> bool:
    return (Path(analysis_dir) / bounding_frame_yaml_cache_basename).exists()


# TODO factor to hong2p / just supply a analysis_dir arg to the hong2p.thor fn?
# TODO if i keep a wrapper of assign_frames_to_odor_presentations here, it should
# probably clear any previous fail indicators if called w/ ignore_cache=True
# (or just always if not using cached output?)
def assign_frames_to_odor_presentations(thorsync_dir, thorimage_dir, analysis_dir,
    ignore_cache: bool = ignore_bounding_frame_cache):

    bounding_frame_yaml_cache = analysis_dir / bounding_frame_yaml_cache_basename

    if ignore_bounding_frame_cache or not bounding_frame_yaml_cache.exists():
        print('assigning frames to odor presentations for:\n'
            f'ThorImage: {shorten_path(thorimage_dir)}\n'
            f'ThorSync:  {shorten_path(thorsync_dir)}', flush=True
        )
        # This may raise an error, and calling code should handle if so.
        bounding_frames = thor.assign_frames_to_odor_presentations(thorsync_dir,
            thorimage_dir
        )
        # Converting numpy int types to python int types, and tuples to lists, for
        # (much) nicer YAML output.
        bounding_frames = [ [int(x) for x in xs] for xs in bounding_frames]

        # TODO regenerate all data (+ recopy stuff that has been copied)
        #
        # Writing thor[image|sync]_dir so that if I need to copy a YAML file from a
        # similar experiment to one failing frame<->odor assignment, I can keep track of
        # where it came from.
        yaml_data = {
            'thorimage_dir': shorten_path(thorimage_dir),
            'thorsync_dir': shorten_path(thorsync_dir),
            'bounding_frames': bounding_frames,
        }
        # TODO TODO also write thorsync and thorimage dirs as keys to the yaml, but
        # don't load them under most circumstances. so that i can tell easier if i
        # copied the yaml file from one recording to a similar recording where the frame
        # assignment is failing.
        with open(bounding_frame_yaml_cache, 'w') as f:
            yaml.dump(yaml_data, f)
    else:
        with open(bounding_frame_yaml_cache, 'r') as f:
            yaml_data = yaml.safe_load(f)

        # TODO delete after all yaml files have been regenerated / recopied s.t. they
        # all include the same set of keys
        if 'bounding_frames' not in yaml_data:
            bounding_frames = yaml_data
        #
        else:
            bounding_frames = yaml_data['bounding_frames']

    return bounding_frames


# TODO add kwargs to get only the n frames / <=x seconds after each odor onset,
# and explore effect on some of things computed with this
# TODO add `strict` kwarg to also check bounding_frames refers to same # of frames as
# length of movie_length_array (though i do often have small errors in frame assignment,
# even if typically/always just at end of individual recordings...)?
# TODO maybe default to keep_pre_odor=True? might want to implement DataArray TODO below
# first
# TODO TODO (option to) return DataArrays w/ a time index where 0 indicates onset for
# the trial
# TODO TODO TODO also accept a function to compute baseline / accept appropriate
# dimensional input [same as mean would be] to subtract directly?
# TODO test this baselining approach works w/ other dimensional inputs too
# TODO cache within a run?
def delta_f_over_f(movie_length_array, bounding_frames, *,
    n_volumes_for_baseline: Optional[int] = n_volumes_for_baseline,
    exclude_last_pre_odor_frame: bool = exclude_last_pre_odor_frame,
    keep_pre_odor: bool = False):
    """
    Args:
        movie_length_array: signal to compute dF/F for, e.g. traces for ROIs or a
            (possibly volumetric) movie. first dimension size should be equal to the
            number of timepoints in the corresponding movie.

        bounding_frames: list of (start_frame, first_odor_frame, end_frame) indices
            delimiting each trial in the corresponding movie

        n_volumes_for_baseline: how many volumes (or frames, in a single-plane movie)
            to use for calculation of baseline. these frames will come from just before
            the odor onset for each trial. if this is None, all volumes from the start
            of the trial will be used for the baseline.

        exclude_last_pre_odor_frame: whether to exclude the last frame before odor onset
            when calculating each trial's baseline. can help hedge against off-by-one
            errors in frame<->trial assignment, but might otherwise dilute the signal.

        keep_pre_odor: if True, yields data for all timepoints, so that output can be
            concatenated to something with an equal number of timepoints as input. if
            False, only yield timepoints after odor onset for each trial (to simplify
            some response statistic calculations)
    """

    # TODO probably instead raise a ValueError here w/ message indicating
    # movie_length_array probably needs to be transposed
    assert all([all([i < len(movie_length_array) for i in trial_bounds])
        for trial_bounds in bounding_frames
    ])

    for start_frame, first_odor_frame, end_frame in bounding_frames:
        # NOTE: this is the frame *AFTER* the last frame included in the baseline
        baseline_afterend_frame = first_odor_frame

        baseline_start_frame = start_frame

        if n_volumes_for_baseline is not None:
            baseline_start_frame = baseline_afterend_frame - n_volumes_for_baseline
            assert baseline_start_frame < first_odor_frame

        if exclude_last_pre_odor_frame:
            baseline_afterend_frame -= 1

        for_baseline = movie_length_array[baseline_start_frame:baseline_afterend_frame]

        # TODO explicitly mean over time dimension if input is xarray
        # (or specify all other dimensions, if that's how to make it work)
        baseline = for_baseline.mean(axis=0)

        # TODO delete + add quantitative check baseline isn't too close to zero or
        # anything (better threshold?)
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

        if not keep_pre_odor:
            after_onset = movie_length_array[first_odor_frame:(end_frame + 1)]
            dff = (after_onset - baseline) / baseline
        else:
            trial = movie_length_array[start_frame:(end_frame + 1)]
            dff = (trial - baseline) / baseline

        # TODO also replace w/ check that no values are too crazy high / warning?
        #
        # was checking range of pixels values made sense. some are reported as max i
        # believe, and probably still are. could maybe just be real noise though (could
        # it? must be a baselining issue, no?).
        #print(dff.max())

        yield dff


# TODO maybe default to mean in a fixed window as below? or not matter as much since i'm
# actually using this on data already meaned within an ROI (though i could use on other
# data later)?
# TODO TODO TODO homogenize stat (max here, mean elsewhere) behavior here vs in response
# volume calculation in process_recording (still the case?)
# TODO TODO TODO compare results w/ old stat=max (using all volumes from onset to end of
# trial) on old data, and if nothing really gets worse (and it improves results on new
# PN data, as I expect), then stick with mean for everything
# (otherwise make driver -> settings dict or something, and only use for PNs)
def compute_trial_stats(traces, bounding_frames,
    odor_order_with_repeats: Optional[ExperimentOdors] = None,
    # TODO TODO TODO special case so it's mean by default for pebbled (to better capture
    # inhibition), and max by default for GH146 (b/c PN spontaneous activity. this make
    # sense? was it max and not mean that worked for me for GH146? maybe it was the
    # other way around?)
    # TODO TODO TODO TODO check GH146 correlations again to see which looked better: max
    # or mean (and maybe doesn't matter on new data?)
    # TODO might need to change dF/F scale now that i'm going back to mean? check
    #stat=lambda x: np.max(x, axis=0),
    stat=lambda x: np.mean(x, axis=0),
    n_volumes_for_response: Optional[int] = n_volumes_for_response
    ):
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

    for trial_traces in delta_f_over_f(traces, bounding_frames):
        if n_volumes_for_response is None:
            for_response = trial_traces
        else:
            for_response = trial_traces[:n_volumes_for_response]

        curr_trial_stats = stat(for_response)

        # TODO TODO adapt to also work in case input is a movie (done?)
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


def get_panel(thorimage_id: Pathlike) -> Optional[str]:
    """Return None or str describing the odors used in the experiment.

    Some panels are split across multiple types of experiments. For example, 'kiwi' is
    the panel for both experiments collecting mainly the components alone as well as
    those collecting just mixtures of the most intense 2 components (run via
    `olf kiwi.yaml` and `olf kiwi_ea_eb_only.yaml`, respectively).
    """
    thorimage_id = Path(thorimage_id).name

    # TODO TODO TODO default to first part of thorimage path (after splitting on '_')?
    # warn if there are >2 parts (maybe indicating can't reliably get panel this way?)?
    # or just warn if we get panel this way in general (saying to enter here?)?
    # (want minimal changes to code to be *required* to get output on new experiments)

    if 'diag' in thorimage_id:
        return diag_panel_str

    elif 'kiwi' in thorimage_id:
        return 'kiwi'

    elif 'control' in thorimage_id:
        return 'control'

    # First set of Remy's experiments that I also did measurements for (initially in
    # ORNs)
    elif 'megamat' in thorimage_id:
        return 'megamat'

    elif 'validation2' in thorimage_id:
        return 'validation2'

    # TODO TODO handle old pair stuff too (panel='<name1>+<name2>' or something) + maybe
    # use get_panel to replace the old name1 + name2 means grouping by effectively panel

    else:
        return None


# TODO maybe i should check for all of a minimum set of files, or just the mtime on
# the df caches, in case a partial run erroneously prevents future runs
def ij_last_analysis_time(plot_dir: Path):
    roi_plot_dir = ijroi_plot_dir(plot_dir)
    return util.most_recent_contained_file_mtime(roi_plot_dir)


def suite2p_outputs_mtime(analysis_dir):
    combined_dir = s2p.get_suite2p_combined_dir(analysis_dir)
    return util.most_recent_contained_file_mtime(combined_dir)


def suite2p_last_analysis_time(plot_dir):
    roi_plot_dir = suite2p_plot_dir(plot_dir)
    return util.most_recent_contained_file_mtime(roi_plot_dir)


def nonroi_last_analysis_time(plot_dir):
    # If we recursed, it would (among other things) visit the ImageJ/suite2p analysis
    # subdirectories, which may be updated more frequently.
    # TODO only check dF/F image / processed TIFF files / response volume cache files?
    # TODO TODO test that this is still accurate now that we are saving a lot of things
    # are root of what used to be plot dir / at same level (+ roi analyses might change
    # mtime of *something* in `plot_dir`, as it's defined here)
    # TODO was usage of nonroi_last_analysis_time ever not correct? delete if no reason
    # to believe...
    #return util.most_recent_contained_file_mtime(plot_dir, recurse=False, verbose=True)

    # TODO TODO maybe what i really want here is the oldest time of a set of files
    # that should change whenever nonroi inputs (movie, whether raw/flipped/mocorr)
    # changes. fix! (and maybe also in other cases)
    # TODO should also be considering the TIFFs spit out under analysis dir
    return util.most_recent_contained_file_mtime(plot_dir, recurse=False)


def format_mtime(mtime: float) -> str:
    """Formats mtime like default `ls -l` output (e.g. 'Oct 11 18:24').
    """
    return time.strftime('%b %d %H:%M', time.localtime(mtime))


def names2fname_prefix(name1, name2):
    return util.to_filename(f'{name1}_{name2}'.lower(), period=False)


def dff_imshow(ax, dff_img, **imshow_kwargs):

    vmin = imshow_kwargs.pop('vmin', dff_vmin)
    vmax = imshow_kwargs.pop('vmax', dff_vmax)

    # TODO TODO make one histogram per fly w/ these dF/F values, for picking global
    # vmin/vmax values (assuming i dont just change to calc vmin/vmax for each fly)
    # (make these in process_recordings)
    #
    # TODO maybe move (prob w/ vmin/vmax default getting above) to process_recording,
    # to have one warning per fly (rather than potentially one per plane)
    #
    # TODO have -v CLI flag also make theshold_frac=0
    threshold_frac = 0.002

    frac_less_than_vmin = (dff_img < vmin).sum() / dff_img.size

    # TODO TODO TODO move these warnings into hong2p.viz.image_grid / plot_rois /
    # similar too (unless kwarg is passed to silence them)
    if frac_less_than_vmin > threshold_frac:
        warn(f'{frac_less_than_vmin:.3f} of dF/F pixels < {vmin=}')

    frac_greater_than_vmax = (dff_img > vmax).sum() / dff_img.size
    if frac_greater_than_vmax > threshold_frac:
        warn(f'{frac_greater_than_vmax:.3f} of dF/F pixels > {vmax=}')
    #

    im = ax.imshow(dff_img, vmin=vmin, vmax=vmax, **imshow_kwargs)

    # TODO TODO figure out how do what this does EXCEPT i want to leave the
    # xlabel / ylabel (just each single str)
    ax.set_axis_off()

    # part but not all of what i want above
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])

    return im


# TODO rename now that i'm also allowing input w/o date/fly_num attributes?
def fly_roi_id(row, *, fly_only: bool = False) -> str:
    """
    Args:
        fly_only: if False, will include date, fly, and ROI information.
            If True, will exclude ROI information, so long as the ROI was given a name
            (and not autonamed/numbered).
    """
    # NOTE: assuming no need to use row.thorimage_id (or a panel / something like
    # that), as assumed this will only be used within a context where that is
    # context (e.g. a plot w/ kiwi data, but no control data)
    try:
        parts = []
        if pd.notnull(row.date):
            date_str = f'{row.date:%-m-%d}'
            parts.append(date_str)

        if pd.notnull(row.fly_num):
            fly_num_str = f'{row.fly_num:0.0f}'
            parts.append(fly_num_str)

        # TODO also support a 'fly'/'fly_id' key in place of (date, fly[_num])?
        # (for lettered/sequential simplified IDs, for nicer plots)

        roi = row.roi
        if not is_ijroi_named(roi):
            fly_only = False

        if not fly_only:
            parts.append(str(roi))

        # fly_only=True for when [h|v]line_[level_fn+group_text] code is drawing ROI
        # labels
        return '/'.join(parts)

    except AttributeError:
        return f'{row.roi}'


def plot_roi_stats_odorpair_grid(single_roi_series, show_repeats=False, ax=None,
    cmap=cmap, **kwargs):

    assert single_roi_series.index.names == ['odor1', 'odor2', 'repeat']

    trial_df = single_roi_series.unstack(level=0)

    trial_df = sort_concs(trial_df)

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
        mean_df = sort_concs(trial_df.groupby('odor2', sort=False).mean())

        fig, _ = viz.matshow(mean_df, **common_matshow_kwargs)

    return fig


# TODO TODO rename this, and similar w/ 'roi_' (any others?), to exclude that?
# what else would i be plotting responses of? this is the main type of response i'd want
# to plot...
# TODO TODO always(/option to) break each of these into a number of plots such that we
# can always see the xticklabels (at the top), without having to scroll up?
def plot_all_roi_mean_responses(trial_df: pd.DataFrame, title=None, roi_sort=True,
    sort_rois_first_on=None, odor_sort=True, keep_panels_separate=True,
    roi_min_max_scale=False, cmap=cmap, **kwargs):
    # TODO rename odor_sort -> conc_sort (or delete altogether)
    # TODO TODO add fly_min_max_scale flag (like roi_min_max_scale)?
    """Plots odor x ROI data displayed with odors as columns and ROI means as rows.

    Args:
        trial_df: ['odor1', 'odor2', 'repeat'] index names and a column for each ROI.

        roi_sort: whether to sort columns

        sort_rois_first_on: passed to sort_fly_roi_cols's sort_first_on kwarg

        keep_panels_separate: if 'panel' is among trial_df index level names, and there
            are any odors shared by multiple panels, this will prevent data from
            different panels from being averaged together

        roi_min_max_scale: if True, scales data within each ROI to [0, 1].
            if `cbar_label` is in `kwargs`, will append '[0,1] scaled per ROI'.

        **kwargs: passed thru to hong2p.viz.matshow

    """
    # TODO factor out this odor-index checking to hong2p.olf
    # TODO maybe also assert these are only index levels
    # (tho 'panel' should also be allowed)
    assert (all(c in trial_df.index.names for c in ['odor1', 'odor2', 'repeat']) or
        trial_df.index.name == 'odor1'
    )

    # TODO also check ROI index (and also factor that to hong2p)
    # TODO maybe also support just 'fly' on the column index (where plot title might be
    # the glomerulus name, and we are showing all fly data for a particular glomerulus)

    avg_levels = ['odor1']
    if 'odor2' in trial_df.index.names:
        avg_levels.append('odor2')

    # TODO unsupport keep_panels_separate=False?
    if keep_panels_separate and 'panel' in trial_df.index.names:
        # TODO TODO TODO warn/err if any null panel values. will silently be dropped as
        # is.
        # TODO or change fn to handle them gracefully (sorting alphabetically w/in?)
        avg_levels = ['panel'] + avg_levels

    # This will throw away any metadata in multiindex levels other than these two
    # (so can't just add metadata once at beginning and have it propate through here,
    # without extra work at least)
    mean_df = trial_df.groupby(avg_levels, sort=False).mean()
    # TODO assertion of the set of remaining levels (besides avg_levels) there are?
    # it should just be 'repeat' and stuff that shouldn't vary / matter, right?
    # (at most. some input probably doesn't have even that?)

    # TODO might wanna drop 'panel' level after mean in keep_panels_separate case, so
    # that we don't get the format_mix_from_strs warning about other levels (or just
    # delete that warning...) (still relevant?)

    if roi_min_max_scale:
        # TODO may need to check vmin/vmax aren't in kwargs and change if so

        # The .min()/.max() functions should return Series where index elements are ROI
        # labels (or at least it won't be the odor axis based on above assertions...).
        mean_df -= mean_df.min()
        mean_df /= mean_df.max()
        assert np.isclose(mean_df.min().min(), 0)
        assert np.isclose(mean_df.max().max(), 1)

        # TODO set this as full title if not in kwargs?
        if 'cbar_label' in kwargs:
            # (won't modify input)
            kwargs['cbar_label'] += '\n[0,1] scaled per ROI'

    if odor_sort:
        mean_df = sort_concs(mean_df)

    if roi_sort:
        mean_df = sort_fly_roi_cols(mean_df, sort_first_on=sort_rois_first_on)

    # TODO deal w/ warning this is currently producing (not totally sure it's this call
    # tho)
    xticklabels = format_mix_from_strs

    # TODO TODO numbered ROIs should be shown as before, and not have number shown
    # as an ROI group label (via hline_* stuff) (ideally in same plot w/ some named ROIs
    # grouped, but maybe just disable if not all certain/named)
    # (which plots currently affected by this? still relevant?)
    # (did *uncertain.<plot_fmt> roi matrix plots used to group non-numbered ROIs
    # together? don't think it's doing that now...)

    # TODO try to move some of this logic into viz.matshow?
    # (the automatic enabling of hline_group_text if we have levels_from_labels?)
    # (also, just to not have to redefine the default value of levels_from_labels...)
    hline_group_text = False
    # (assuming it's a valid callable if so)
    if 'hline_level_fn' in kwargs and not kwargs.get('levels_from_labels', True):
        if all([x in trial_df.columns.names for x in fly_keys]):
            # TODO maybe still check if there is >1 fly too (esp if this path produces
            # bad looking figures in that case)

            # will show the ROI label only once for each group of rows where the
            # ROI name is the same (e.g. but coming from different flies)
            hline_group_text = True

    yticklabels = lambda x: fly_roi_id(x, fly_only=hline_group_text)

    vline_group_text = kwargs.pop('vline_group_text', 'panel' in trial_df.index.names)

    mean_df = mean_df.T

    # TODO maybe put lines between levels of sortkey if int (e.g. 'iplane')
    # (and also show on plot as second label above/below roi labels?)

    fig, _ = viz.matshow(mean_df, title=title, cmap=cmap,
        xticklabels=xticklabels, yticklabels=yticklabels,
        hline_group_text=hline_group_text, vline_group_text=vline_group_text, **kwargs
    )

    # TODO just mean across trials right? do i actually use this anywhere?
    # would probably make more sense to just recompute, considering how often i find
    # myself writing `fig, _ = plot...`
    return fig, mean_df


def plot_responses_and_scaled_versions(df: pd.DataFrame, plot_dir: Path,
    fname_prefix: str, *, bbox_inches=None, **kwargs) -> None:
    """Saves response matrix plots to <plot_dir>/<fname_prefix>[_normed].<plot_fmt>

    Args:
        bbox_inches: None is also matplotlib savefig default
    """
    fig, _ = plot_all_roi_mean_responses(df, **kwargs)
    savefig(fig, plot_dir, f'{fname_prefix}', bbox_inches=bbox_inches)

    fig, _ = plot_all_roi_mean_responses(df, roi_min_max_scale=True, **kwargs)
    savefig(fig, plot_dir, f'{fname_prefix}_normed', bbox_inches=bbox_inches)

    # TODO TODO and a scaled-within-each-fly version?


def suite2p_traces(analysis_dir):
    try:
        traces, roi_stats, ops, merges = s2p.load_s2p_combined_outputs(analysis_dir,
            # TODO what does "merge_inputs" mean here? if i recall, it means to keep the
            # ROIs that were merged within suite2p to create other ROIs, but i might
            # wanna rename this kwarg to be more clear if so
            # TODO when would err trigger an error? rename to be more clear?
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


# TODO change in hong2p to allow having non-named ROIs drawn with slightly duller red /
# thinner lines?
# TODO change how LUT is defined to at least allow ~single pixels NOT washing out whole
# range (e.g. from motion correction artifact at edge)
def plot_rois(*args, nrows: Optional[int] = 1, cmap=roi_bg_cmap, **kwargs):
    # TODO test display of cmap default in generated docs. try to get it to look nice
    # (without having to manually specify the default in the docstring)
    """
    Args:
        *args: passed thru to `hong2p.viz.plot_rois`

        cmap: passed to `hong2p.viz.plot_rois` (which has it's own default of
            `hong2p.viz.DEFAULT_ANATOMICAL_CMAP='gray'`)

        **kwargs: passed thru to `hong2p.viz.plot_rois`
    """
    # TODO version of this plot post merging, to show which planes are selected?
    # (maybe w/ non-used planes still shown w/ lower color saturation / something?)

    # TODO maybe define global glomerulus name -> color palette here?

    if not any(x in kwargs for x in ('vmin', 'vmax')):
        # which fraction of min/max to clip, when picking colorscale range for
        # background image intensities. unless scale_per_plane=True, colorscale is
        # shared across all planes.
        minmax_clip_frac = roi_background_minmax_clip_frac
    else:
        # (this is also the default inside hong2p.viz.image_grid, and shouldn't conflict
        # with v[min|max] being passed)
        # TODO only add to kwargs in above branch, and delete this branch then?
        # adding to kwargs is fine, right?
        minmax_clip_frac = 0.0

    if 'palette' not in kwargs:
        # TODO may want to delete if i implement simpler color= instead
        if 'colors' in kwargs:
            palette = None
        else:
        #
            palette = sns.color_palette('hls', n_colors=10)
    else:
        palette = kwargs['palette']

    # TODO TODO TODO why is color seed apparently not working (should be constant
    # default of 0 inside plot_rois)? (still true?)
    # TODO did i actually need _pad=False? comment why (hong2p.plot_rois default is
    # False, at least now....)
    return viz.plot_rois(*args, nrows=nrows, cmap=cmap, _pad=False,
        # TODO maybe switch to something where it warns at least? does it now?
        # (plot_closed_contours default for this is 'err')
        if_multiple='ignore',

        # TODO TODO inside hong2p.viz.plot_rois, warn if this clipping happens
        # (just like other code in here that does regarding dff_vp[min|max])?
        minmax_clip_frac=minmax_clip_frac,

        # Without n_colors=10, there will be too many colors to meaninigfully tell them
        # all apart, so might as well focus on cycling a smaller set of more distinct
        # colors (using randomization now, but maybe graph coloring if I really want to
        # avoid having neighbors w/ similar colors).
        # TODO just move this default into hong2p.viz.plot_rois...?
        # for most of these?
        palette=palette,
        # TODO delete. didn't like as much as hls10
        # 'bright' will only produce <=10 colors no matter n_colors
        # This will get rid of the grey color, where (r == g == b), giving us 9 colors.
        #palette=[x for x in sns.color_palette('bright') if len(set(x)) > 1],

        **kwargs
    )


_date_dir_parent_index = -4
_analysis_dir_part = 'analysis_intermediates'
_thorimage_dir_part = 'raw_data'

def analysis2thorimage_dir(analysis_dir: Path) -> Path:
    # TODO maybe also check only 3 (/maybe 2 sometimes) parts after, and that they are
    # parseable as expected (can't use this exactly cause sometimes fly dir is passed
    # in, e.g. in plot_roi.py)
    #assert _analysis_dir_part == analysis_dir.parts[_date_dir_parent_index]
    assert _analysis_dir_part in analysis_dir.parts
    return Path(str(analysis_dir).replace(_analysis_dir_part, _thorimage_dir_part))


def thorimage2analysis_dir(thorimage_dir : Path) -> Path:
    assert _thorimage_dir_part in thorimage_dir.parts
    return Path(str(thorimage_dir).replace(_thorimage_dir_part, _analysis_dir_part))


# TODO rename to indicate it also [can] plot rois (+ it also returns full_rois now?)
# TODO add plot_roi_kws?
# TODO TODO TODO refactor so that i can most code both called in single recordings (as
# now), and in (not yet implemented) code doing the same on all data w/in a fly
# (concatenated)
def ij_traces(analysis_dir: Path, movie, bounding_frames, roi_plots=False):

    thorimage_dir = analysis2thorimage_dir(analysis_dir)
    try:
        full_rois = ijroi_masks(analysis_dir, thorimage_dir)

    except IOError:
        raise

    # TODO TODO check this is working correctly with this new type of input +
    # change fn to preserve input type (w/ the metadata xarrays / dfs have)
    # TODO TODO TODO modify extract_traces_bool_masks to return output of same type as
    # input (w/ any additional metadata DataFrame / DataArray might have kept intact)
    traces = pd.DataFrame(extract_traces_bool_masks(movie, full_rois))
    traces.index.name = 'frame'
    # TODO rename 'roi_index' for now? (or delete if not used, as i suspect)
    traces.columns.name = 'roi'

    # TODO either assert that index of traces now matches some index in full_rois, or
    # change so that is true (by fixing extract_traces_bool_masks to preserve metadata?)
    # (assertion would prob fail now that ijroi_masks can do some subsetting. they must
    # at least be the same shape though, since it is literally computed w/ full_rois...)

    # TODO TODO TODO refactor to compute these across all data for each fly
    # (concatenated across movies, done after all the process_recording calls)

    # TODO also try merging via correlation/overlap thresholds?

    # TODO equivalent suite2p ROI handling code also need to be updated to use
    # compute_trial_stats for roi quality?
    # TODO maybe also ~z-score before picking best plane (dividing by variability in
    # baseline period first)
    trial_dff = compute_trial_stats(traces, bounding_frames)
    roi_quality = trial_dff.max()

    # TODO TODO TODO refactor so all this calculation is done across all recordings
    # within each fly, rather than just within each recording
    # (could add in parallel for now, and then phase out old way)
    roi_indices, best_plane_rois = rois2best_planes_only(full_rois, roi_quality)

    # TODO make full_rois have roi name represented same way as in best_plane_rois
    # (former currently has roi_name, latter currently has just .roi)
    # (would make uniform handling in plot_rois(...) easier. not sure i care about that
    # anymore tho...) (or vice versa?)

    # (if one roi is defined across N planes, it counts N times here)
    n_roi_planes = full_rois.sizes['roi']

    is_best_plane = np.zeros(n_roi_planes, dtype=bool)
    is_best_plane[roi_indices] = True

    # NOTE: is_best_plane currently only part of 'roi' coordinates that is not in that
    # one big MultiIndex (unlikely to matter).
    full_rois = full_rois.assign_coords({'is_best_plane': ('roi', is_best_plane)})

    # TODO delete?
    #
    # same type of assertion already made in rois2best_planes_only, so just checking i'm
    # doing basic xarray stuff right... (and no need to check best_plane_rois.roi
    # (where the names are stored there))
    input_roi_names = set(full_rois.roi_name.values)
    output_roi_names = set(full_rois.sel(roi=full_rois.is_best_plane).roi_name.values)
    assert input_roi_names == output_roi_names
    del input_roi_names, output_roi_names

    assert n_roi_planes == traces.shape[1]
    # weak check that these are actually indices, and not ROI nums. wouldn't catch many
    # cases tho...
    assert ((0 <= roi_indices) & (roi_indices < n_roi_planes)).all()

    # TODO change to .iloc[:, roi_indices] to be more clear? it's not roi nums we are
    # dealing with anymore. rois2best_plane_only numbers continuous indices in there
    # (not skipping any, like roi_num might, b/c of stuff like the '+' suffix which
    # causes ROIs to be dropped circa loading)
    traces = traces[roi_indices].copy()

    # TODO don't i want to ideally return roi_name and stuff like that (might want to
    # keep most metadata in full_rois coords? why not? something currently break if so?)
    # ig i'd had to rename roi_name -> roi at some point, but is that the only
    # fundamental thing?
    #
    # needed this (b/c rois2best_planes_only returned more metadata than intended) when
    # i had xarray 2022.6.0 somehow, but much of the rest of the code was broken.
    # getting back to xarray==0.19.0 in that conda env seemed to fix it (though I might
    # have had the code working with something more like 0.23 at some point?)
    #traces.columns = best_plane_rois.roi_name.values
    traces.columns = best_plane_rois.roi.values

    # do need to add this again it seems (and i think one above *might* have been used
    # inside `rois2best_planes_only`)
    traces.columns.name = 'roi'

    # TODO can i just use best_plane_rois.roi_z.values?
    z_indices = full_rois.isel(roi=roi_indices).roi_z.values

    # TODO delete
    #
    # TODO is identical every useful? had similar issue where identical was false but
    # these two were True...
    # r1 = full_rois.sel(roi=full_rois.is_best_plane)
    # r2 = full_rois.sel(roi=full_rois.is_best_plane.values)
    # > r1.equals(r2)
    # True
    # > str(r1) == str(r2)
    # True
    # > r1.identical(r2)
    # False
    #
    # z1 = full_rois.isel(roi=roi_indinces).roi_z
    # z2  = masks2.roi_z[masks2.roi_index.isin(roi_indices)]
    # not sure why (for z1, z2 as defined above):
    # z1.equals(z2) but NOT z1.identical(z2). don't think the difference matters, and
    # asserting .values are equal to check.
    full_rois2 = full_rois.assign_coords(
        {'roi_index': ('roi', range(full_rois.sizes['roi']))}
    )
    z_indices2 = full_rois2.roi_z[full_rois2.roi_index.isin(roi_indices)].values
    assert np.array_equal(z_indices, z_indices2)
    del full_rois2, z_indices2
    #

    ret = (traces, best_plane_rois, z_indices, full_rois)
    # TODO delete this kwarg (have always be true?)?
    if not roi_plots:
        return ret

    panel = get_panel(thorimage_dir)

    # NOTE: now i often have multiple recordings with this panel (b/c so many odors)
    # TODO maybe change to only do on first?
    #
    # Only plotting ROI spatial extents (+ making symlinks to those plots) for
    # diagnostic experiment for now, because for most fles I had symlinked all other
    # RoiSet.zip files to the one for the diagnostic experiment.
    if panel != diag_panel_str:
        return ret
    # TODO TODO am i unnecessarily calling this ROI plotting stuff for stuff w/ multiple
    # diagnostic recordings? check against that dict pointing to single diagnostic
    # recordings, rather than just checking panel (above)

    # TODO refactor to do plotting elsewhere / explicitly pass in plot_dir / something?

    date, fly_num, thorimage_id = util.dir2keys(analysis_dir)

    plot_dir = get_plot_dir(date, fly_num, thorimage_id)
    ij_plot_dir = ijroi_plot_dir(plot_dir)

    across_fly_ijroi_dir = fly2plot_root(date, fly_num) / across_fly_ijroi_dirname

    ijroi_spatial_extents_plot_dir = across_fly_ijroi_dir / 'spatial_extents'
    makedirs(ijroi_spatial_extents_plot_dir)

    experiment_id = shorten_path(thorimage_dir)
    experiment_link_prefix = experiment_id.replace('/', '_')

    # TODO save these ROI images in a place where we can do it once for each fly,
    # and have the background be computed across all the input movies
    # (do in loop that makes concat tiff, after process_recording?)

    movie_mean = movie.mean(axis=0)

    # TODO TODO try minmax_clip_frac > 0 (maybe 0.025 or something?)
    # (to set colorscale limits to inner 95% percentile, at 0.025)
    #description_background_kwarg_tuples = [('avg', movie_mean, dict())]
    # TODO TODO need to explicitly handle <=0 values w/ norm='log'? or will be clipped
    # fine anyway (and i'm cool with that?)
    description_background_kwarg_tuples = [
        ('avg', movie_mean, dict(cbar_label='mean F (a.u.)'))
        # TODO TODO TODO do i actually like 'log' better than default tho?
        # (how to adjust label to reflect norm anyway? need to?)
        #('avg', movie_mean, dict(norm='log', cbar_label='mean F (a.u.)'))

        # TODO this should still work, right? rn it seems label must be passed via
        # cbar_label. fix?
        #('avg', movie_mean, dict(norm='log', cbar_kws=dict(label='mean F (a.u.)')))
    ]

    # TODO maybe do all this plotting in a separate step at the end, so i can get one
    # colormap that maps all certain ROI names across all flies to particular colors, to
    # make it easier to eyeball similarities across flies?

    # TODO maybe pick diag_example_dff_v[min|max] on a per fly basis (from some
    # percentile?)?

    # NOTE: as currently implemented, this will need to be generated on an earlier run
    # of this script, as these TIFFs are saved after where the ROI analysis is done.
    max_trialmean_dff_tiff_fname = analysis_dir / max_trialmean_dff_tiff_basename
    if max_trialmean_dff_tiff_fname.exists():
        max_trialmean_dff = tifffile.imread(max_trialmean_dff_tiff_fname)
        # TODO TODO move vmin/vmax warning into image_grid, so it is handled
        # homogenously? (where is it currently?)
        # TODO TODO also try dict() / dict(minmax_clip_frac=<something greater than 0>)?
        description_background_kwarg_tuples.append(
            # TODO [just?] try diff norm?
            # TODO make sure warning still/also getting generated if too much data is
            # below/above vmin/vmax here (and all other places stuff like vmin/max used)
            # TODO replace cbar label here to be accurate (it's max-across-odors of
            # current label (mean dF/F), at least w/in recording...)
            ('max_trialmean_dff', max_trialmean_dff, diag_example_plot_roi_kws)
        )
    else:
        warn(f'{max_trialmean_dff_tiff_fname} did not exist. re-run script to use as '
            'ROI plot background.'
        )

    zstep_um = thor.get_thorimage_zstep_um(thorimage_dir)

    # TODO maybe also plot on stddev image or something?

    # TODO TODO TODO revert all constant scale modifications?
    # (both on average and dF/F bgs) contrast within each plane tends to get
    # worse... and scale doesn't actually matter in any of those cases...
    # was just to appease betty, and i think she might agree it's not making the figure
    # better (at least, so far, towards the diagnostic example supplemental fig)

    for bg_desc, background, imagegrid_kws in description_background_kwarg_tuples:

        # TODO TODO provide <date>/<fly_num> suptitle for all of these
        # TODO probably black out behind text, like the imagej option, to make ROI names
        # easier to read

        # TODO color option to desaturate plane-ROIs that are NOT the "best" plane
        # (for a given volumetric-ROI) (or change line properties?)

        # TODO TODO TODO also (/only?) do a version that only shows the plane actually
        # used (or at least those planes emphasized? change line / sat?)

        # TODO delete this comment (+ try/except)?
        # what issue was happening when not plotted "correctly"? and can i still
        # reproduce at all?
        #
        # NOTE: RuntimeError should no longer be triggered now that I'm adding
        # if_multiple='ignore' in plot_rois wrapper.
        #
        # In 5/5 examples I checked across 2023-04-22 - 2023-05-09 data, the contours
        # were being plotted correctly despite matplotlib seemingly producing multiple
        # disconnected paths. The label was sometimes offset more than I'd like, but
        # that's only a minor issue. warned about
        try:
            # TODO rename from imagegrid_kws (here and elsewhere). some are not just
            # passed to image_grid
            fig = plot_rois(full_rois, background, zstep_um=zstep_um, **imagegrid_kws)

        # TODO make custom error message for this in hong2p (or handle, removing need
        # for error)
        #
        # For "RuntimeError: multiple disconnected paths in one footprint" raised by
        # hong2p.viz.plot_closed_contours
        except RuntimeError as err:
            assert False, 'should be unreachable (2)'
            warn(f'{shorten_path(analysis_dir)}: {err}')
            continue

        fig_path = savefig(fig, ij_plot_dir, f'all_rois_on_{bg_desc}')

        all_roi_dir = ijroi_spatial_extents_plot_dir / f'all_rois_on_{bg_desc}'
        makedirs(all_roi_dir)
        symlink(fig_path, all_roi_dir / f'{experiment_link_prefix}.{plot_fmt}')

        fig = plot_rois(full_rois, background, certain_only=True, zstep_um=zstep_um,
            **imagegrid_kws
        )
        fig_path = savefig(fig, ij_plot_dir, f'certain_rois_on_{bg_desc}')

        certain_roi_dir = ijroi_spatial_extents_plot_dir / f'certain_rois_on_{bg_desc}'
        makedirs(certain_roi_dir)
        symlink(fig_path, certain_roi_dir / f'{experiment_link_prefix}.{plot_fmt}')

    return ret


def trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
    main_plot_title, *, roi_stats=None, show_suite2p_rois=False):

    # TODO TODO remake directory (or at least make sure plots from ROIs w/ names no
    # longer in set of ROI names are deleted)

    if show_suite2p_rois and roi_stats is None:
        raise ValueError('must pass roi_stats if show_suite2p_rois')

    # TODO update to pathlib
    makedirs(roi_plot_dir)

    # TODO TODO TODO plot raw responses (including pre-odor period), w/ axvline for odor
    # onset
    # TODO TODO plot *MEAN* (across trials) timeseries responses (including pre-odor
    # period), w/ axvline for odor onset
    # TODO concatenate these DataFrames into a DataArray somehow -> serialize
    # (for other people to analyze)?
    dff_traces = list(delta_f_over_f(traces, bounding_frames, keep_pre_odor=True))

    odor_index = odor_lists_to_multiindex(odor_lists)

    # TODO try one of those diverging colormaps w/ diff scales for the two sides
    # (since the range of inhibition is smaller)
    matshow_kwargs = dict(cmap=diverging_cmap, vmin=-dff_vmax, vmax=dff_vmax)

    timeseries_plot_dir = roi_plot_dir / 'timeseries'
    # TODO update to pathlib
    makedirs(timeseries_plot_dir)

    # TODO TODO factor out this timeseries plotting to a fn

    # TODO don't hardcode number of trials? do i elsewhere?
    n_trials = 3

    # TODO delete this debug path after adding a quantitative check that there isn't an
    # offset in any odor onset frames
    debug = False
    #debug = True

    curr_odor_i = 0

    # TODO TODO probably cluster to order rows by default

    axs = None
    # TODO if i'm not gonna concat dff_traces into some xarray thing, maybe just use
    # generator in zip rather than converting to list first above
    # TODO TODO maybe condense into one plot instead somehow? (odors left to right,
    # probably just showing a mean rather than all trials)
    for trial_dff_traces, trial_bounds, trial_odors in zip(dff_traces, bounding_frames,
        odor_index):

        start_frame, first_odor_frame, _ = trial_bounds

        # TODO TODO replace w/ general odor mixture handling that supports odor2 !=
        # 'solvent'. hack for lab meeting. index 0 = odor1
        trial_odor, _, repeat = trial_odors

        if repeat > 2:
            # TODO warn?
            continue

        if repeat == 0:
            # NOTE: layout kwarg not available in older (< ~3.6) versions of matplotlib
            fig, axs = plt.subplots(nrows=1, ncols=n_trials, layout='constrained',
                # TODO lower dpi when not debugging (setting high so i can have a chance
                # at reading xticklabel frame numbers)
                dpi=600 if debug else 300, figsize=(6, 3)
            )
            first_odor_frames = []

        first_odor_frames.append(first_odor_frame)

        assert axs is not None

        # TODO change  [v|h]line_level_fns so that they are computed on index values,
        # not [x|y]ticklabels (at least for pandas input? ig there would be a
        # consistency issue then...)
        # TODO update matshow so ints can be passed in for vline_level, not just
        # fns defining levels
        # TODO fix so xticklabels w/ ints works?
        vline_level_fn = lambda frame: int(frame) >= first_odor_frame

        # TODO change viz.matshow to have better default [x|y]ticklabels
        # (should work here, shouldn't need to be a *str* one level index to use it)
        # TODO try w/o the list(...) call
        xticklabels = [str(x) for x in trial_dff_traces.index]

        ax = axs[repeat]

        _, im = viz.matshow(trial_dff_traces.T, ax=ax,

            vline_level_fn=vline_level_fn, xticklabels=xticklabels, linecolor='k',
            fontsize=2.0 if debug else None, xtickrotation='vertical',

            # TODO fix
            # viz.matshow colorbar behavior broken for ax=<Axes from a subplot array>,
            # now that viz.matshow forces constrained layout
            colorbar=False,

            **matshow_kwargs
        )
        # TODO probably delete
        #vline = (first_odor_frame - start_frame - 1) + 0.5
        #ax.axvline(vline, linewidth=0.5, color='k')

        if not debug:
            ax.xaxis.set_ticks([])
            # TODO TODO convert frames->seconds ideally (might need to pass some other
            # info in...) (when not debugging frame assignment)

            # TODO figure out how to do this but also still get axes to line up
            # (this slightly shrinks the axes with the xlabel added)
            # (or was it the yticks on the first one shrinking it actually?)
            #if repeat == 0:
            #    ax.set_xlabel('Frame')

        # TODO figure out alignment + restore
        #if repeat != 0:
        #    ax.yaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        # TODO TODO where is alignment issue coming from? colorbar really changing size
        # of first axes more (and vertically, too!)?
        # something in viz.matshow?

        # TODO fix hack
        if repeat == 2:
            fig.colorbar(im, ax=axs, shrink=0.6)

            title = trial_odor
            for_filename = f'{curr_odor_i}_{trial_odor}_trials'
            if debug:
                title = f'{title}\nfirst_odor_frames={pformat(first_odor_frames)}'
                for_filename = f'debug_{for_filename}'

            fig.suptitle(title)

            savefig(fig, timeseries_plot_dir, for_filename)

            curr_odor_i += 1

    # TODO TODO TODO compare dF/F traces (or at least response means from these traces)
    # (as currently calculated), to those calculated from dF/F'd movie (in
    # response_volumes, calculated in process_recording)
    #
    # Mean dF/F for each ROI x trial
    trial_df = compute_trial_stats(traces, bounding_frames, odor_lists)

    is_pair = is_pairgrid(odor_lists)

    # TODO maybe replace odor_sort kwarg with something like odor_sort_fn, and pass
    # sort_concs in this case, and maybe something else for the non-pair experiments i'm
    # mainly dealing with now

    # TODO make axhlines bewteen changes in z_indices
    fig, mean_df = plot_all_roi_mean_responses(trial_df, sort_rois_first_on=z_indices,
        odor_sort=is_pair, title=main_plot_title, cbar_label=trial_stat_cbar_title,
        cbar_shrink=0.4
    )
    # TODO rename to 'traces' or something (to more clearly disambiguate wrt spatial
    # extent plots)
    savefig(fig, roi_plot_dir, 'all_rois_by_z')

    if not is_pair:
        return trial_df

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

    return trial_df


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

    # TODO TODO TODO are these traces also starting at odor onset, like ones calculated
    # via my delta_f_over_f fn (within ij_traces)
    import ipdb; ipdb.set_trace()

    # TODO also plot roi / outline of roi on corresponding [mean?] plane / maybe with
    # other planes for context?

    roi_plot_dir = suite2p_plot_dir(plot_dir)
    title = 'Suite2p ROIs\nOrdered by Z plane'

    trial_df = trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
        title, roi_stats=roi_stats, show_suite2p_rois=False
    )
    print('generated plots based on suite2p traces')

    return trial_df


def ij_trace_plots(analysis_dir, bounding_frames, odor_lists, movie, plot_dir):

    # TODO TODO TODO maybe each fn that returns traces (this and suite2p one), should
    # already add the metadata compute_trial_stats adds, or more? maybe adding a
    # seconds time index w/ 0 on each trial's first_odor_frame?
    #
    # could probably pass less variables if some of them were already in the coordinates
    # of a DataArray traces
    traces, best_plane_rois, z_indices, full_rois = ij_traces(analysis_dir, movie,
        bounding_frames, roi_plots=True
    )

    roi_plot_dir = ijroi_plot_dir(plot_dir)
    title = 'ImageJ ROIs\nOrdered by Z plane\n*possibly [over/under]merged'

    trial_df = trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
        title
    )

    print('generated plots based on traces from ImageJ ROIs')

    return trial_df, best_plane_rois, full_rois


# TODO factor to hong2p.suite2p
_default_ops = None
def load_default_ops(_cache=True):
    global _default_ops
    if _cache and _default_ops is not None:
        return _default_ops.copy()

    ops_dir = Path.home() / '.suite2p/ops'
    # This file is what gets written w/ the GUI button to save new defaults
    ops_file = ops_dir / 'ops_user.npy'
    ops = s2p.load_s2p_ops(ops_file)
    _default_ops = ops
    return ops.copy()


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

    ops = load_default_ops()

    # TODO TODO perhaps try having threshold_scaling depend on plane, and use run_plane
    # instead? (decrease on lower planes, where it's harder to find stuff generally)

    # TODO maybe use suite2p's options for ignoring flyback frames to ignore depths
    # beyond those i've now settled on?

    data_specific_ops = s2p.suite2p_params(thorimage_dir)
    for k, v in data_specific_ops.items():
        assert k in ops
        ops[k] = v

    # TODO TODO probably add a whitelist parameter inside suite2p at this point, so i
    # can just specify excatly the tiff(s) i want
    # TODO TODO replace w/ usage of (new?) ops['tiff_list'] parameters?

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
        trial_dff_tiff_basename,
        trialmean_dff_tiff_basename,
        max_trialmean_dff_tiff_basename,
        min_trialmean_dff_tiff_basename,
        # TODO may need to configure this to also ignore either raw.tif or flipped.tif,
        # depending on current value of min_input (never want to include *both* raw.tif
        # and flipped.tif)
    )

    db = {
        #'data_path': [thorimage_dir],
        # TODO update suite2p to take substrs / globs to ignore input files on
        # (at least for TIFFs specifically, to ignore other TIFFs in input dir)
        'data_path': [str(analysis_dir)],
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

    if type(proxy) is multiprocessing.managers.DictProxy:
        return {k: proxy2orig_type(v) for k, v in proxy.items()}

    elif type(proxy) is multiprocessing.managers.ListProxy:
        return [proxy2orig_type(x) for x in proxy]

    else:
        # This should only be reached in cases where `proxy` is not actually a proxy.
        return proxy


def multiprocessing_namespace_to_globals(shared_state):
    globals().update({
        k: proxy2orig_type(v) for k, v in shared_state.items()
    })


# TODO would it be possible to factor so that:
# - core loading functions make data available in a consistent manner
#   (maybe a Recording/Experiment object can be populated w/ relevant data,
#   using attributes to only load/compute stuff as necessary)
#
# - fns deciding whether to skip stuff at various steps can be passed in
#
# - fns w/ certain analysis steps to do, again potentially divided into different steps
#   (like maybe pre/post registration/extraction, in a way that different types of
#   experiments (maybe again w/ those matching some indicator fn?) can have different
#   steps done?)
#
# - consideration for stuff done on a fly / experiment level, with maybe the option to
#   alternate between the two in a specific serialized order (e.g. so we can guarantee
#   we have the outputs needed for subsequent steps, such as to register stuff together
#   and then analyze the experiments separately)
#
# - the multiprocessing option is factored into the fn and doesn't need to wrap it in
#   each analysis
#
# maybe have analysis types define what experiment types are valid input for them (and,
# at least by default, just run all of those?)

# TODO probably refactor so that this is essentially just populating lists[/listproxies]
# of dataframes from s2p/ijroi stuff (extracting in ij case, merging in both cases, also
# converting to trial stats in both), and then move most plotting to after this (and
# cache output of the loop over calls to this) maybe leave plotting of dF/F images and
# stuff closer to raw data in here. (or at least factor out time intensive df/f image
# plots and cache/skip those separately)
# TODO TODO figure out why this seems to be using up a lot of memory now. is something
# not being cleaned up properly? or maybe i just had very low memory available and it
# wasn't abnormal?
def process_recording(date_and_fly_num, thor_image_and_sync_dir, shared_state=None):
    """Analyzes a single recording, in isolation.

    Args:
        ...
        shared_state (multiprocessing.managers.DictProxy): str global variable names ->
        other proxy objects
    """
    # Only relevant if called via multiprocessing, where this is how we get access to
    # the proxies that will feed back to the corresponding global variables that get
    # modified under this function.
    update_globals_from_shared_state(shared_state)

    date, fly_num = date_and_fly_num
    thorimage_dir, thorsync_dir = thor_image_and_sync_dir

    date_str = format_date(date)

    # These functions for defering printing/warning are so we can control what gets
    # printed in skipped data in one place. We want the option to not print any thing
    # for data that is ultimately skipped, but we also want to see all the output for
    # non-skipped data.

    # NOTE: This should be called at least once before process_recording returns, and
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

    # May want to handle calls to this after _called_do_nonskipped is True
    # differently... (always print? don't add to global record of what is skipped, for
    # multiple pass stuff?).
    def print_skip(msg, yaml_path=None, *, color=None, file=None):
        """
        This should be called preceding any premature return from process_recording.
        """
        # TODO TODO TODO also add the fly, date, thorimage[, thorsync] as keys in a
        # global dict marking stuff as having-been-skipped-in-a-previous-run, to not
        # need to recompute stuff
        # TODO TODO TODO and for non-skipped stuff, maybe add all metadata we would have
        # up to the point of the last skip (or at least some core stuff actually used
        # later, like odor metadata), to not have to reload / compute that stuff
        if not print_skipped:
            return

        print_inputs_once(yaml_path)

        if color is not None:
            msg = colored(msg, color)

        print(msg, file=file)
        print()

    _called_do_nonskipped = False
    _to_print_if_not_skipped = []
    def print_if_not_skipped(x):
        assert not _called_do_nonskipped, ('after '
            'do_nonskipped_experiment_prints_and_warns, use regular print/warn'
        )
        if print_skipped:
            print(x)
        else:
            _to_print_if_not_skipped.append(x)

    def warn_if_not_skipped(x):
        assert not _called_do_nonskipped, ('after '
            'do_nonskipped_experiment_prints_and_warns, use regular print/warn'
        )
        if print_skipped:
            warn(x)
        else:
            _to_print_if_not_skipped.append(UserWarning(x))

    def do_nonskipped_experiment_prints_and_warns(yaml_path):
        nonlocal _called_do_nonskipped
        # We should only be doing this once
        assert not _called_do_nonskipped
        _called_do_nonskipped = True

        # Because in this case we should be printing / warning everything ASAP.
        if print_skipped:
            return

        print_inputs_once(yaml_path)
        for x in _to_print_if_not_skipped:
            if type(x) is UserWarning:
                warn(x)
            else:
                print(x)

    # If we are printing skipped, we might as well print each of the input paths as soon
    # as possible, to make debugging easier (so relevant paths will be more likely to
    # appear before traceback -> termination).
    if print_skipped:
        print_inputs_once()

    thorimage_basename = thorimage_dir.name
    panel = get_panel(thorimage_basename)

    # TODO delete all the analyze_glomeruli_diagnostics stuff (always behave as if True)
    if panel == diag_panel_str:
        if not analyze_glomeruli_diagnostics:
            print_skip('skipping because experiment is just glomeruli diagnostics')
            return

        is_glomeruli_diagnostics = True
    else:
        if analyze_glomeruli_diagnostics_only:
            print_skip('skipping because experiment is NOT glomeruli diagnostics\n')
            return

        is_glomeruli_diagnostics = False

    exp_start = time.time()

    single_plane_fps, xy, z, c, n_flyback, _, xml = thor.load_thorimage_metadata(
        thorimage_dir, return_xml=True
    )

    if not analyze_2d_tests and z == 1:
        print_skip('skipping because experiment is single plane')
        return

    plot_dir = get_plot_dir(date, fly_num, thorimage_basename)
    experiment_id = shorten_path(thorimage_dir)

    def suptitle(title, fig=None, *, experiment_id_in_title: bool = True):
        if title is None:
            return

        if fig is None:
            fig = plt.gcf()

        if experiment_id_in_title:
            title = f'{experiment_id}\n{title}'

        fig.suptitle(title)

    # TODO rename to 'recording'/'rec'/similar (to be consistent w/ new meanings for
    # 'recording' and 'experiment', where the latter can have multiple of the former)
    def exp_savefig(fig, desc, **kwargs) -> Path:
        return savefig(fig, plot_dir, desc, **kwargs)

    analysis_dir = get_analysis_dir(date, fly_num, thorimage_dir)
    fly_analysis_dir = analysis_dir.parent

    if links_to_input_dirs:
        symlink(thorimage_dir, plot_dir / 'thorimage', relative=False)
        symlink(analysis_dir, plot_dir / 'analysis', relative=False)

    # TODO try to speed up? or just move my stimfiles to local storage?
    # currently takeing ~0.04s per call, -> 3-4 seconds @ ~80ish calls
    # (same issue w/ yaml.safe_load on bounding frames though, so maybe it's not
    # storage? or maybe it's partially a matter of seek time? should be ~5-10ms tho...)
    try:
        yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(xml)

    except NoStimulusFile:
        # currently redundant w/ warning in case of same error in preprocess_recordings
        print_skip(f'skipping because no stimulus YAML file referenced in ThorImage '
            'Experiment.xml'
        )
        return

    # Again, in this case, we might as well print all input paths ASAP.
    if print_skipped:
        print_inputs_once(yaml_path)

    pulse_s = float(int(yaml_data['settings']['timing']['pulse_us']) / 1e6)
    assert pulse_s <= 3
    if pulse_s < 3:
        warn_if_not_skipped(f'odor {pulse_s=} not the standard 3 seconds')

        # Remy started the config she gave me as 2s pulses, but we are now on 3s.
        # I might still want to average the 2s pulse data...
        if panel != 'megamat':
            print_skip(f'skipping because odor pulses were {pulse_s} (<3s) long (old)',
                yaml_path
            )
            return

    is_pair = is_pairgrid(odor_lists)
    if is_pair:
        # So that we can count how many flies we have for each odor pair (and
        # concentration range, in case we varied that at one point)
        names_and_concs_tuple = odor_lists2names_and_conc_ranges(odor_lists)

        if final_pair_concentrations_only:
            names, curr_concs = separate_names_and_concs_tuples(names_and_concs_tuple)

            if names in names2final_concs and names2final_concs[names] != curr_concs:

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

        name1 = olf.abbrev(name1)
        name2 = olf.abbrev(name2)

        if not is_acquisition_host:
            experiment_basedir = keys2rel_plot_dir(date, fly_num, thorimage_basename)

            # TODO delete (unless i need output_root anyway...)
            output_root = fly2driver_indicator_output_dir(date, fly_num)

            # TODO delete (actually, may have since broken this again. fix!)
            experiment_basedir2 = plot_dir.relative_to(output_root)
            #assert experiment_basedir2 == experiment_basedir, \
            #    f'{experiment_basedir2} != {experiment_basedir}'
            #

            # TODO TODO do this for all panels/experiment types (or (panel, is_pair)
            # combinations...)

            plot_root = fly2plot_root(date, fly_num)
            pair_dir = plot_root / 'pairs' / get_pair_dirname(name1, name2)

            makedirs(pair_dir)
            symlink(plot_dir, pair_dir / experiment_basedir)
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

        seen_yamls = seen_stimulus_yamls2thorimage_dirs[yaml_path]
        # TODO elaborate to say it's only bad if intended to have a unique random order
        # for each?
        err_msg = (f'stimulus yaml {short_yaml_path} seen in:\n'
            f'{pformat([str(x) for x in seen_yamls])}'
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
        has_failed, fail_suffixes = last_fail_suffixes(analysis_dir)
        if has_failed:
            if suite2p_fail_prefix in fail_suffixes:
                failed_suite2p_dirs.append(analysis_dir)

            # Need to not skip stuff if *only* frame assigning fail indicator is there,
            # AND we also have a trial<->frames YAML (e.g. that was copied in from a
            # similar experiment)
            if (set(fail_suffixes) == {frame_assign_fail_prefix} and
                has_cached_frame_odor_assignments(analysis_dir)):

                # TODO share message w/ duped code in write_trial_and_frame_json?
                # need to format tho... fn? or just share non-formatted part of message?
                # TODO TODO TODO trigger this warning if YAML has thorimage name not
                # matching directory containing it (and for both means of triggering,
                # print which experiment it was copied from)
                warn(f'{shorten_path(thorimage_dir)} previously failed frame<->odor '
                    'assignment, but using assignment present in YAML in analysis '
                    'directory'
                )

            else:
                suffixes_str = ' AND '.join(fail_suffixes)

                print_skip(f'skipping because previously failed {suffixes_str}',
                    yaml_path, color='red'
                )
                return

    # NOTE: trying to move towards using 'experiment' to mean one (or more) recordings,
    # in a particular fly, with a common set of odors (whose presentations might be
    # split across recordings). 'recording' should now mean the output of a single Thor
    # acquisition run.
    experiment_key = (date, fly_num, panel, is_pair)

    if experiment_key not in experiment2recording_dirs:
        experiment2recording_dirs[experiment_key] = [(thorimage_dir, thorsync_dir)]
    else:
        experiment2recording_dirs[experiment_key].append((thorimage_dir, thorsync_dir))

    before = time.time()

    # TODO don't bother doing this if we only have imagej / suite2p analysis left to do,
    # and the required output directory doesn't exist / ROIs haven't been manually
    # drawn/filtered / etc
    # TODO thread thru kwargs
    bounding_frames = assign_frames_to_odor_presentations(thorsync_dir, thorimage_dir,
        analysis_dir
    )

    # TODO TODO why was this getting triggered on 2023-09-14/1/george01?
    # was there a pulse from previous recording? stopped early or something?
    # could be nothing interesting (and that data almost certainly not useful)
    #
    # TODO delete unless this gets triggered (and fix if it does).
    # should be getting detected in write_trial_frames_and_json now, and skipped just
    # above b/c of fail indicator now)
    if len(bounding_frames) != len(odor_order_with_repeats):
        # TODO TODO try to indicate if it seems like the yaml path could have been
        # entered incorrectly in the Experiment.xml note field (via manual entry at time
        # of experiment collection, in ThorImage note field. should at least actually be
        # detecting dupes here... s.t. if the same yaml is entered twice for one fly,
        # it triggers a warning at least)
        # TODO TODO TODO better error message. sam has this happen somewhat often.
        assert False, 'should be unreachable'
    #

    # (loading the HDF5 should be the main time cost in the above fn)
    load_hdf5_s = time.time() - before

    if do_suite2p:
        run_suite2p(thorimage_dir, analysis_dir, overwrite=overwrite_suite2p)

    # Not including concentrations in metadata to add, b/c I generally run this script
    # skipping all but final concentrations (process_recording returns None for all
    # non-final concentrations)
    # (already defined in is_pair case)
    if not is_pair:
        name1 = np.nan
        name2 = np.nan

    new_col_level_names = ['date', 'fly_num', 'thorimage_id']
    new_col_level_values = [date, fly_num, thorimage_basename]

    # TODO TODO still somehow support the arbitrary pair experiment data (w/o panel
    # defined via get_panel, currently) (just add 'name1','name2' to 'panel','is_pair'?)
    #new_row_level_names = ['name1', 'name2']
    #new_row_level_values = [name1, name2]
    new_row_level_names = ['panel', 'is_pair']
    new_row_level_values = [panel, is_pair]

    # TODO delete xarray support here (or move all scalar coords to their own index?
    # able to make non-scalar then? and would that alone fix the indexing issues i've
    # been having? or the 'odor' dim (so it could for sure be non-scalar)?)
    def add_metadata(data):
        if isinstance(data, pd.DataFrame):
            df = util.addlevel(data, new_row_level_names, new_row_level_values)
            return util.addlevel(df, new_col_level_names, new_col_level_values,
                axis='columns'
            )
        # Assuming DataArray here
        # TODO TODO TODO do i ever actually use this? did i mean to?
        # (seems yes, but only in ijroi case)
        # TODO TODO TODO probably factor something like this into top level /
        # hong2p.xarray/util
        else:
            new_coords = dict(zip(
                new_col_level_names + new_row_level_names,
                new_col_level_values + new_row_level_values
            ))

            # TODO make this more graceful. do for all of a list of odor-specific params
            # (given how pixelwise corr code works, panel should not be included in such
            # a list)
            # TODO refactor to something to append coordinates (w/ scalar values, at
            # lesat) to existing dimensions
            # TODO may want to change to set_index
            new_coords['is_pair'] = ('odor', [is_pair] * data.sizes['odor'])
            new_coords['is_pair_b'] = ('odor_b', [is_pair] * data.sizes['odor_b'])

            data = data.assign_coords(new_coords)

            # NOTE: this currently results in different order of the index levels wrt
            # the older pixelwise corr DataArray code. Shouldn't matter, hopefully
            # doesn't...
            # TODO also factor out w/ above (if i do)
            return data.set_index(odor=['is_pair'], append=True
                ).set_index(odor_b=['is_pair_b'], append=True)

    if analyze_suite2p_outputs:
        if not any([b in thorimage_dir for b in bad_suite2p_analysis_dirs]):
            s2p_trial_df_cache_fname = analysis_dir / 'suite2p_trial_df_cache.p'

            s2p_last_run = suite2p_outputs_mtime(analysis_dir)
            s2p_last_analysis = suite2p_last_analysis_time(plot_dir)

            # TODO handle more appropriately (checking for dir/contents first, etc)
            assert s2p_last_run is not None, f'{analysis_dir} suite2p dir empty'

            if s2p_last_analysis is None or s2p_last_analysis < s2p_last_run:
                s2p_analysis_current = False
            else:
                s2p_analysis_current = True

            if not exists(s2p_trial_df_cache_fname):
                s2p_analysis_current = False

            if not should_ignore_existing('suite2p') and s2p_analysis_current:
                print_if_not_skipped('suite2p outputs unchanged since last analysis')
                s2p_trial_df = pd.read_pickle(s2p_trial_df_cache_fname)
            else:
                s2p_trial_df = suite2p_trace_plots(analysis_dir, bounding_frames,
                    odor_lists, plot_dir
                )
                s2p_trial_df = add_metadata(s2p_trial_df)
                to_pickle(s2p_trial_df, s2p_trial_df_cache_fname)
                # TODO why am i calling print_inputs_once(yaml_path) in ijroi stuff
                # below but not here? what if i just wanna analyze the suite2p stuff?

            s2p_trial_dfs.append(s2p_trial_df)
        else:
            full_bad_suite2p_analysis_dirs.append(analysis_dir)
            print_if_not_skipped('not making suite2p plots because outputs marked bad')

    nonroi_last_analysis = nonroi_last_analysis_time(plot_dir)

    try:
        tiff_path = find_movie(date, fly_num, thorimage_basename)

        # TODO refactor to share checking w/ find_movie?
        if tiff_path.name.startswith('mocorr'):
            tiff_path_link = tiff_path
            assert tiff_path_link.is_symlink(), f'{tiff_path_link=} not a symlink'

            # .resolve() should find the file actually pointed to, by any series of
            # file/directory symlinks
            tiff_path = tiff_path_link.resolve()

        assert tiff_path.is_file() and not tiff_path.is_symlink()

        # TODO this should also be invalidating any cached roi-based analysis on this
        # data (but if the mocorr(/tiff) is more recent than the roi definitions, then
        # they would need to be redrawn/edited first... so just warn?)
        #
        # NOTE: this could be of a motion corrected TIFF or a raw/flipped TIFF,
        # whichever is the best (i.e. mocorr > flipped > raw) available.
        tiff_mtime = getmtime(tiff_path)

        if nonroi_last_analysis is None:
            nonroi_analysis_current = False
        else:
            nonroi_analysis_current = tiff_mtime < nonroi_last_analysis

        if not nonroi_analysis_current:
            print_if_not_skipped(
                'TIFF (/ motion correction) changed. updating non-ROI outputs.'
            )

    except IOError as err:
        warn(f'{err}\n')
        return

    # TODO move to module level?
    desired_processed_tiffs = (
        trial_dff_tiff_basename,
        trialmean_dff_tiff_basename,
        max_trialmean_dff_tiff_basename,
        min_trialmean_dff_tiff_basename,
    )
    for name in desired_processed_tiffs:
        desired_output = analysis_dir / name
        if not desired_output.exists() or tiff_mtime >= getmtime(desired_output):
            nonroi_analysis_current = False
            break

    # Assuming that if analysis_dir has *any* plots directly inside of it, it has all of
    # what we want (including any outputs that would go in analysis_dir).
    do_nonroi_analysis = should_ignore_existing('nonroi') or not nonroi_analysis_current

    # TODO factor into loop checking desired_processed_tiffs above?
    response_volume_cache_fname = analysis_dir / 'trial_response_volumes.p'
    response_volume_cache_exists = response_volume_cache_fname.exists()
    if not response_volume_cache_exists:
        do_nonroi_analysis = True

    if response_volume_cache_exists and not do_nonroi_analysis:
        # TODO rename from load_corr_dataarray. confusing here (since a correlation is
        # not what it's loading)
        response_volumes_list.append(load_corr_dataarray(response_volume_cache_fname))

    # TODO am i currently refusing to do any imagej ROI analysis on non-mocorred
    # stuff? maybe i should?
    do_ij_analysis = False
    if analyze_ijrois:
        have_ijrois = has_ijrois(analysis_dir)

        fly_key = (date, fly_num)
        if (not have_ijrois and fly_key in fly2diag_thorimage_dir and
            (fly_analysis_dir / mocorr_concat_tiff_basename).exists()):

            diag_analysis_dir = thorimage2analysis_dir(fly2diag_thorimage_dir[fly_key])
            diag_ijroi_fname = ijroi_filename(diag_analysis_dir, must_exist=False)

            if diag_ijroi_fname.is_file():
                diag_ijroi_link = analysis_dir / ijroiset_default_basename

                # TODO maybe i should switch to saving RoiSet.zip's in fly analysis dirs
                # (at root, rather than under each recording's subdir) -> delete all
                # RoiSet.zip symlinking code?
                print_if_not_skipped(f'no {ijroiset_default_basename}. linking to ROIs'
                    ' defined on diagnostic recording '
                    f'{shorten_path(diag_analysis_dir)}.'
                )
                # NOTE: changing the target of the link should also trigger
                # recomputation of ImageJ ROI outputs for directories w/ links to the
                # changed RoiSet.zip. tested.
                symlink(diag_ijroi_fname, diag_ijroi_link)
                have_ijrois = True

        if have_ijrois:
            dirs_with_ijrois.append(analysis_dir)

            # TODO also save as a CSV for easy inspection / transfer? or just leave that
            # to the top-level one containing all / most data?
            ij_trial_df_cache_fname = analysis_dir / 'ij_trial_df_cache.p'

            ijroi_last_analysis = ij_last_analysis_time(plot_dir)

            if ijroi_last_analysis is None:
                ij_analysis_current = False
            else:
                # TODO TODO make sure that if we could would make any of the roi plots
                # in ij_traces, we also consider ij_analysis_current=False if we don't
                # have those

                # We don't need to check LHS for None b/c have_ijrois check earlier.
                ij_analysis_current = (
                    ijroi_mtime(analysis_dir) < ijroi_last_analysis
                )

            # TODO delete after generating them all? slightly more robust to interrupted
            # runs if i leave it
            if not ij_trial_df_cache_fname.exists():
                ij_analysis_current = False

            ignore_existing_ijroi = should_ignore_existing('ijroi')

            do_ij_analysis = True
            if not ignore_existing_ijroi:
                if ij_analysis_current:
                    do_ij_analysis = False

                    print_if_not_skipped(
                        'ImageJ ROIs unchanged since last analysis. reading cache.'
                    )
                    ij_trial_df = pd.read_pickle(ij_trial_df_cache_fname)
                    ij_trial_dfs.append(ij_trial_df)
                    # TODO TODO TODO also load full_rois [+ best_plane_rois], for use in
                    # plot_rois call below
                else:
                    print_inputs_once(yaml_path)
                    print('ImageJ ROIs were modified. re-analyzing.')
            else:
                print_inputs_once(yaml_path)
                print('ignoring existing ImageJ ROI analysis. re-analyzing.')
        else:
            print_if_not_skipped('no ImageJ ROIs')

    # TODO maybe we should do_nonskipped... here, to use it more for stuff skipped for
    # what the raw data+metadata IS, rather than what analysis outputs we already have?
    # distinction could make factoring out skip logic easier

    if not (do_nonroi_analysis or do_ij_analysis):
        # (technically we could have already loaded movie if we converted raw to TIFF in
        # this function call)
        print_skip('not loading movie because neither non-ROI nor ImageJ ROI analysis '
            'were requested', yaml_path
        )
        return

    before = time.time()

    # TODO find a way to keep track of whether the tiff was flipped, and invalidate +
    # complain (or just flip) any ROIs drawn on original non-flipped TIFFs? similar
    # thing but w/ any motion correction i add... any of this possible (does ROI format
    # have reference to tiff it was drawn on anywhere?)?

    try:
        movie = load_movie(date, fly_num, thorimage_basename)
    except IOError as err:
        warn(f'{err}\n')
        return

    do_nonskipped_experiment_prints_and_warns(yaml_path)

    read_movie_s = time.time() - before

    # TODO fix handling of None for these + serialization
    best_plane_rois = None
    full_rois = None
    #

    # TODO delete
    # TODO TODO make sure everything below that requires full_rois/best_plane_rois
    # to be not None below only happens if do_ij_analysis == True
    # (or actually, just actually cache and load these in have_ijrois [but not
    # !do_ij_analysis] case...)

    if do_ij_analysis:
        # TODO just return additional stuff from this to use best_plane_rois in
        # plot_rois as like full_rois (some metadata dropped when converting latter to
        # former)?
        ij_trial_df, best_plane_rois, full_rois = ij_trace_plots(analysis_dir,
            bounding_frames, odor_lists, movie, plot_dir
        )

        # TODO TODO check responses have similar distribution of where the pixels with
        # which intensity ranks are (as response coming from a different place w/in ROI
        # is likely contamination from something nearby?). check center of intensity
        # mass? warn if deviations are above some threshold?
        # do in a script to be called from an imagej macro?

        # TODO TODO TODO compare best_plane_rois across experiments (+ probably refactor
        # their computation to always be across all recordings for a fly anyway, rather
        # than computing within each recording...)

        # TODO also pickle best_plane_rois & full_rois (so we can do the nonroi plots w/
        # ijrois no matter whether we are currently re-running ijroi analysis or not)
        # (for now, maybe just `-i nonroi,ijroi`, as a hack)
        # (probably wanna move all this after process_recording anyway though, computing
        # everything on concatenated data, so may not be worth it)

        ij_trial_df = add_metadata(ij_trial_df)
        to_pickle(ij_trial_df, ij_trial_df_cache_fname)
        ij_trial_dfs.append(ij_trial_df)

    # TODO TODO TODO compute lags between odor onset (times) and peaks fluoresence times
    # -> warn if they differ by a certain amount, ideally an amount indicating
    # off-by-one frame misassignment, if that is even reliably possible

    if not do_nonroi_analysis:
        print_skip('skipping non-ROI analysis\n')
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

    # TODO TODO TODO figures like these produce, but also overlaying ROIs as plot_rois
    # does (put somewhere in the ijroi dependent part). prob use b/w colormap plot_rois
    # does by default.
    # (need to move any dF/F image calculation earlier?)
    # TODO maybe label plot_rois w/ zstep in general anyway?
    zstep_um = thor.get_thorimage_zstep_um(xml)

    def micrometer_depth_title(ax, z_index) -> None:
        viz.micrometer_depth_title(ax, zstep_um, z_index, fontsize=ax_fontsize)

    def plot_and_save_dff_depth_grid(dff_depth_grid, fname_prefix, title=None,
        cbar_label=None, experiment_id_in_title=False, **imshow_kwargs) -> Path:

        # Will be of shape (1, z), since squeeze=False
        fig, axs = plt.subplots(ncols=z, squeeze=False,
            figsize=single_dff_image_row_figsize
        )

        for d in range(z):
            ax = axs[0, d]

            # TODO this isn't what i do in other cases tho, right? can't i just fix
            # float specified inside micrometer_depth_title, if "-0 uM" or something
            # like that is why i'm special casing this? (it does do -0, but not actually
            # sure why...)
            if z > 1:
                micrometer_depth_title(ax, d)

            im = dff_imshow(ax, dff_depth_grid[d], **imshow_kwargs)

        viz.add_colorbar(fig, im, label=cbar_label, shrink=0.68)

        suptitle(title, fig, experiment_id_in_title=experiment_id_in_title)
        fig_path = exp_savefig(fig, fname_prefix)
        return fig_path

    anat_baseline = movie.mean(axis=0)

    # TODO rename fn (since input here is not dF/F)
    # TODO remove[/don't plot] colorbar on this one?
    plot_and_save_dff_depth_grid(anat_baseline, 'avg', 'average of whole movie',
        vmin=anat_baseline.min(), vmax=anat_baseline.max(), cmap=anatomical_cmap
    )

    save_dff_tiff = want_dff_tiff
    if save_dff_tiff:
        dff_tiff_fname = analysis_dir / 'dff.tif'
        if dff_tiff_fname.exists():
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

    # TODO refactor loop below to zip over these, rather than needing to explicitly call
    # next() as is
    dff_after_onset_iterator = delta_f_over_f(movie, bounding_frames)
    dff_full_trial_iterator = delta_f_over_f(movie, bounding_frames, keep_pre_odor=True)

    # TODO do i still need these 2-3 nested loops?
    for i, odor_str in enumerate(odor_order):

        if odor_str in odor_str2target_glomeruli:
            target_glomerulus = odor_str2target_glomeruli[odor_str]
            title = f'{odor_str} ({target_glomerulus})'
        else:
            target_glomerulus = None
            title = odor_str

        # TODO either:
        # - always use 2 digits (leading 0)
        # - pick # of digits from len(odor_order)
        plot_desc = f'{i + 1}_{title}'

        trial_heatmap_fig, trial_heatmap_axs = plt.subplots(nrows=n_repeats,
            ncols=z, squeeze=False, figsize=(6.4, 3.9)
        )

        # Each element is mean-within-a-window response for one trial, of shape
        # (z, y, x)
        trial_mean_dffs = []

        for n in range(n_repeats):
            dff = next(dff_after_onset_iterator)

            if save_dff_tiff:
                trial_dff_movie = next(dff_full_trial_iterator)
                trial_dff_movies.append(trial_dff_movie)

            # TODO TODO TODO refactor so all response computation goes through
            # compute_trial_stats (to ensure no divergence)?
            # (test on PN data in particular, where greater spontaneous activity had
            # caused some problems with mean based statistics [at least the mean in the
            # baseline...])
            # TODO off by one at start? (still relevant?)
            mean_dff = dff[:n_volumes_for_response].mean(axis=0)

            # This one is for constructing an xarray of the response volume after the
            # loop over odors. Below is just for calculating mean across trials of an
            # odor.
            all_trial_mean_dffs.append(mean_dff)

            trial_mean_dffs.append(mean_dff)

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

                # NOTE: last value this is set to will be used to add colorbar
                # TODO so does this assume that color scale for each image is the same?
                # assert? where are vmin/vmax getting in now? inside dff_imshow?
                im = dff_imshow(ax, mean_dff[d])

        viz.add_colorbar(trial_heatmap_fig, im, label=dff_cbar_title, shrink=0.32)

        suptitle(title, trial_heatmap_fig)
        exp_savefig(trial_heatmap_fig, plot_desc + '_trials')

        avg_mean_dff = np.mean(trial_mean_dffs, axis=0)

        trialmean_dff_fig_path = plot_and_save_dff_depth_grid(avg_mean_dff, plot_desc,
            title=title, cbar_label=f'mean {dff_latex}'
        )

        # TODO TODO warn if target glomerulus is None? (<- delete)
        # TODO TODO TODO fix elsewhere to not need this hack (/better name at least)
        # TODO or should i highlight all in these cases?
        focus_roi_fixes = {
            # ethyl 3-hydroxybutyrate @ -6
            'DM5/VM5d': 'DM5',
        }
        focus_roi = focus_roi_fixes.get(target_glomerulus, target_glomerulus)

        # TODO TODO TODO similar plot, but using max dF/F (also try stddev [of raw
        # movie? dF/F?]?), showing all ROIs. -> maybe switch individual odor plots to
        # only spelling out single ROIs (/ highlighting and listing odor and/or target
        # glom in text to the side / above odor row?)
        # TODO TODO TODO colorbars! (+ maybe try depth specific and/or odor specific
        # color scales?)

        # TODO TODO TODO is color seed not working on plot_rois? (still relevant?)

        # TODO TODO add colorbar too
        #
        # NOTE: unlike dff_imshow (or stuff calling it), this won't warn about large
        # fractions of pixels above vmax / below vmin.
        # TODO off-by-one on z?
        # TODO try w/ best_plane_rois instead of full_rois (or color desat thing)?
        # TODO change colormap so there is never any ~gray. completely invisible (?)
        # TODO TODO experiment w/ desaturating other colors and/or maybe making
        # line around current ROI solid / thicker (or other lines thinner).
        # want to call attention to ROI targetted by current diagnostic odor.
        # TODO desat the text/lines equally? maybe lines only?
        # TODO push everything to black a bit (via vmin/vmax?), so roi labels are more
        # legible?
        # TODO maybe try have highlighted ROI always be that default red color?
        # (probably try desat first...)
        #
        # TODO TODO try one colorscale per roi
        # TODO check have_ijrois is what i want here (/ works)
        if have_ijrois:
            # TODO delete (when is this happening? probably when cache actually isn't
            # loaded, cause it's not implemented...?)
            # TODO delete try/except.
            # 2023-12-05 (got this when trying to repro savefig error from previous day...):
            # $ ./al_analysis.py -t 2022-02-03 -e 2022-04-03 -v -s model
            # ...
            #thorimage_dir: 2022-02-22/1/kiwi_ea_eb_only
            #thorsync_dir: 2022-02-22/1/SyncData003
            #yaml_path: 20220222_184517_stimuli/20220222_184517_stimuli_0.yaml
            #TIFF (/ motion correction) changed. updating non-ROI outputs.
            #ImageJ ROIs unchanged since last analysis. reading cache.
            #Warning: 0.004 of dF/F pixels < vmin=-0.5
            #Warning: 0.003 of dF/F pixels < vmin=-0.5
            #Warning: 0.003 of dF/F pixels > vmax=2.0
            #Uncaught exception
            #Traceback (most recent call last):
            #  File "./al_analysis.py", line 10774, in <module>
            #    main()
            #  File "./al_analysis.py", line 9588, in main
            #    was_processed = list(starmap(process_recording, keys_and_paired_dirs))
            #  File "./al_analysis.py", line 4188, in process_recording
            #    assert full_rois is not None
            #AssertionError
            # (above also happened for 3-10/2[/control1_ramps] on another run)
            try:
                assert full_rois is not None
            except AssertionError:
                import ipdb; ipdb.set_trace()
            #

            # TODO don't really care for a non-certain-only version of these, do i?
            # still have those versions for ones over average / recording-max-dF/F

            # TODO TODO TODO also probably use same (SMALLER) dF/F range as in the max
            # dF/F plot_rois call (or even smaller range here!)

            # TODO TODO TODO text to left (/above?) that has odor name (probably
            # horizontal) (and target glomerulus?)
            # TODO TODO abbreviation only (instead of full name, as in titles in some
            # other similar plots [the ones saved w/ exp_savefig])

            # TODO TODO TODO are colorbar labels getting cut off or are they just not
            # passed?

            # TODO do something drastic, like only showing name for focus_roi?
            # TODO still wanna try the red arrow thing? did i already?

            # TODO TODO TODO try having the red/blue version below each row w/ black
            # background (change ROI lines to gray + don't show labels, assuming each is
            # paired with a version where you can see labels)

            # TODO TODO try a version like this, but w/ colors=['black'] and background
            # colormap some kind of blue<->red (to also show inhibition)?
            #
            # TODO delete?
            # indexing this way rather than just roi_name=focus_roi, so that the
            # roi_name metadata variable isn't lost (used by plot_rois)
            # TODO is there a simpler way to achieve that?
            '''
            only_focus = full_rois.sel(
                roi=[x == focus_roi for x in full_rois.roi_name.values]
            )
            # TODO TODO or should plot_rois work w/ empty input?
            if only_focus.sizes['roi'] > 0:
                # current behavior could also be achieved by omitting focus_roi=,
                # passing colors=['red'], and omitting palette (which I assume is in the
                # *kws)
                trialmean_dff_w_focusroi_fig = plot_rois(only_focus, avg_mean_dff,
                    certain_only=True, focus_roi=focus_roi, zstep_um=zstep_um,
                    title=title, **diag_example_plot_roi_kws
                )
                # TODO put all variants of these roi plots under ijroi/spatial_extents/?
                # TODO rename all to '*/odoravg_*' (other plots like
                # 'certain_rois_on_avg.png' say what they are on)?
                savefig(trialmean_dff_w_focusroi_fig, plot_dir / 'ijroi/with_focusroi',
                    plot_desc
                )
            '''

            # TODO move certain_only=True, best_planes_only=True into
            # diag_example_plot_kws (if i like them)? (probably do for
            # certain_only=True, but probably not for best_planes_only=True)
            #
            # TODO shorten names (trialmean->odor? what else?)
            trialmean_dff_w_bestplane_rois_fig = plot_rois(full_rois, avg_mean_dff,
                focus_roi=focus_roi, certain_only=True, best_planes_only=True,
                zstep_um=zstep_um, title=title, **diag_example_plot_roi_kws
            )
            savefig(trialmean_dff_w_bestplane_rois_fig,
                plot_dir / 'ijroi/with_bestplane_rois', plot_desc
            )

            # TODO prat request: try to ensure top of cbar labelled?

            # TODO vertical instead (on col per odor), to maximize resolution available
            # for ROI boundaries/labels in each image? idk...

            trialmean_dff_w_rois_fig = plot_rois(full_rois, avg_mean_dff,
                focus_roi=focus_roi, certain_only=True, zstep_um=zstep_um,
                title=title, **diag_example_plot_roi_kws
            )
            # TODO make sure these are sorted correctly by PDF aggregation code
            # (in fixed odor order, as other diagnostic dF/F images should be)
            # (can't figure how this is happening, if it is [in commented code, no
            # less]... delete comment?)
            savefig(trialmean_dff_w_rois_fig, plot_dir / 'ijroi/with_rois', plot_desc)

        odor_mean_dff_list.append(avg_mean_dff)

        # TODO TODO TODO is this stuff not getting run? fix if so
        # TODO maybe also include some quick reference to previously-presented-stimulus,
        # to check for contamination components of noise?
        if not is_acquisition_host and target_glomerulus is not None:
            # gsheet only has labels on a per-fly basis, and those should apply to the
            # glomeruli diagnostic experiment corresponding to the same FOV as the other
            # experiments. Don't want to link any other experiments anywhere under here.
            rel_exp_dir = '/'.join(analysis_dir.parts[-3:])
            if rel_exp_dir in unused_glomeruli_diagnostics:
                continue

            plot_root = fly2plot_root(date, fly_num)
            across_fly_diags_dir = plot_root / across_fly_diags_dirname

            # TODO also include odor name in dir?
            glomerulus_dir = (across_fly_diags_dir /
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

            if label_subdir is None:
                try:
                    curr_diag_good = fly_diag_statuses.loc[target_glomerulus.lower()]

                except KeyError:
                    warn(f'target glomerulus {target_glomerulus} not in Google'
                        ' sheet! add column and label data. currently not linking these'
                        ' plots!'
                    )
                    continue

                # NOTE: if there are ever two columns in gsheet with same glomerulus
                # name, this could truth test could raise a pandas ValueError
                if curr_diag_good:
                    label_subdir = 'good'
                else:
                    label_subdir = 'bad'
            else:
                warn('please label quality glomeruli diagnostics for fly '
                    f'{date_str}/{fly_num} in Google Sheet'
                )

            label_dir = join(glomerulus_dir, label_subdir)
            makedirs(label_dir)

            # TODO refactor? i do this at least one other place (and maybe switch to
            # using metadata directly rather than relying on a str while i'm at it?)
            link_prefix = '_'.join(experiment_id.split(os.sep)[:-1])
            # TODO pathlib
            link_path = join(label_dir, f'{link_prefix}.{plot_fmt}')

            #  TODO TODO does this work w/ the 'glomeruli_diagnostics_part2' recordings?
            # (also note that one of those is still misspelled, at
            # 2022-11-30/glomerli_diagnostics_part2)
            # (also want 'diagnotics<n>' to work now split across however many)
            if exists(link_path):
                # TODO update warning to indicate it only really applies in this
                # particular limited context (right?) [or handle better, to avoid need
                # for this warning]
                #
                # Just warning so that all the average images, etc, will still be
                # created, so those can be used to quickly tell which experiment
                # corresponded to the same side as the real experiments in the same fly.
                warn(f'{date_str}/{fly_num} has multiple glomeruli diagnostic'
                    ' experiments. add all but one to unused_glomeruli_diagnostics. '
                    'FIRST IS CURRENTLY LINKED BUT MAY NOT BE THE RELEVANT EXPERIMENT!'
                )
                continue

            symlink(trialmean_dff_fig_path, link_path)

    # (odor2abbrev used inside odor_lists_to_multiindex)
    # TODO TODO move *use of* odor2abbrev to shortly before / in plotting, to not have
    # to recompute cached stuff just to change abbreviations (this clearly isn't where
    # main abbreviation is happening, as is_pair path not used in current analysis of
    # remy's data...)
    # TODO should i just accept odor2abbrev or as kwarg to viz.matshow (requiring an
    # odor level in one of the indices)? where else do i need to use abbrevs?  [sub]plot
    # titles? elsewhere?
    #
    # TODO maybe refactor so it doesn't need to be computed both here and in both the
    # (ij/suite2p) trace handling fns (though they currently also use odor_lists to
    # compute is_pairgrid, so might want to refactor that too)
    odor_index = odor_lists_to_multiindex(odor_lists)
    assert len(odor_index) == len(all_trial_mean_dffs)

    # TODO maybe factor out the add_metadata fn above to hong2p.util + also handle
    # xarray inputs there?
    # TODO any reason to use attrs for these rather than additional coords?
    # either make concatenating more natural (no great way to concat with attrs as of
    # 2022-09, it seems)?
    metadata = {
        'panel': panel,
        'is_pair': is_pair,
        'date': date,
        'fly_num': fly_num,
        'thorimage_id': thorimage_basename,
    }
    coords = metadata.copy()
    coords['odor'] = odor_index

    # TODO some reason i'm not using add_metadata on this DataArray? delete DataArray
    # path in that code / move it to hong2p if i am not going to use it?
    #
    # TODO use long_name attr for fly info str?
    # TODO populate units (how to specify though? pint compat?)?
    arr = xr.DataArray(all_trial_mean_dffs, dims=['odor', 'z', 'y', 'x'], coords=coords)

    # TODO probably move in constructor above if it ends up being useful to do here (?)
    # TODO factor to hong2p.xarray
    arr = arr.assign_coords({n: np.arange(arr.sizes[n]) for n in spatial_dims})

    response_volumes_list.append(arr)
    # TODO rename to remove 'corr'. not what we are writing here...
    write_corr_dataarray(arr, response_volume_cache_fname)

    # TODO try to save all tiffs with timing / xy resolution / step size information, as
    # much as possible (in a format useable by imagej, ideally same as if entered
    # manually)

    # TODO any tiff metadata fields where i could save certain details about the
    # parameters used to compute these processed TIFFs?

    if save_dff_tiff:
        dff_movie = np.concatenate(trial_dff_movies)
        assert dff_movie.shape == movie.shape

        print(f'writing dF/F TIFF to {dff_tiff_fname}...', flush=True, end='')

        util.write_tiff(dff_tiff_fname, dff_movie, strict_dtype=False)

        print(' done', flush=True)

        del dff_movie, trial_dff_movies

    max_trialmean_dff = np.max(odor_mean_dff_list, axis=0)
    min_trialmean_dff = np.min(odor_mean_dff_list, axis=0)

    if write_processed_tiffs:
        # Of length (.shape[0]) equal to number of odor presentations
        # (including each repeat separately).
        trial_dff_tiff = analysis_dir / trial_dff_tiff_basename
        trial_dff = np.array(all_trial_mean_dffs)
        util.write_tiff(trial_dff_tiff, trial_dff, strict_dtype=False)

        # Of length equal to number of odor presentations, AVERAGED ACROSS
        # (consecutive?) TRIALS.
        trialmean_dff_tiff = analysis_dir / trialmean_dff_tiff_basename
        trialmean_dff = np.array(odor_mean_dff_list)
        util.write_tiff(trialmean_dff_tiff, trialmean_dff, strict_dtype=False)

        max_trialmean_dff_tiff = analysis_dir / max_trialmean_dff_tiff_basename
        util.write_tiff(max_trialmean_dff_tiff, max_trialmean_dff,
            strict_dtype=False, dims='ZYX'
        )

        min_trialmean_dff_tiff = analysis_dir / min_trialmean_dff_tiff_basename
        util.write_tiff(min_trialmean_dff_tiff, min_trialmean_dff,
            strict_dtype=False, dims='ZYX'
        )

        flies_with_new_processed_tiffs.append(fly_analysis_dir)

        print('wrote processed TIFFs')

    # TODO TODO generate these from the min-of-mins and max-of-maxes TIFFs now
    # (or at least load across the individual TIFFs within each panel and compute within
    # there?)
    path = plot_and_save_dff_depth_grid(max_trialmean_dff,
        'max_trialmean_dff', title=f'max of trial-mean {dff_latex}',
        cbar_label=f'{dff_latex}'
    )
    # To see if strong inhibition ever helps quickly identify glomeruli
    # TODO TODO use ~only-negative colorscale for min_trialmean_dff figure
    plot_and_save_dff_depth_grid(min_trialmean_dff,
        'min_trialmean_dff', title=f'min of trial-mean {dff_latex}',
        cbar_label=f'{dff_latex}'
    )

    plt.close('all')

    exp_total_s = time.time() - exp_start
    exp_processing_time_data.append((load_hdf5_s, read_movie_s, exp_total_s))

    print()

    # Just to indicate that something was analyzed. Could replace w/ some data
    # potentially, still just checking the # of None/falsey return values to get the #
    # of non-analyzed things.
    return True


# TODO rename to not be exlusive to registration (may want to check slightly different
# set of things then tho?)
def was_suite2p_registration_successful(suite2p_dir: Path) -> bool:

    # TODO check for combined dir too (/only?) ?

    def plane_successful(plane_dir):
        reg_bin = plane_dir / 'data.bin'
        if not reg_bin.exists():
            return False
        # TODO might or might not gain something from also checking its size / whether
        # ops exists
        return True

    if not suite2p_dir.exists():
        return False

    plane_dirs = list(suite2p_dir.glob('plane*/'))
    if len(plane_dirs) == 0:
        return False

    return all(plane_successful(p) for p in plane_dirs)


# TODO TODO refactor / merge into run_suite2p. copied from that fn above.
# TODO probably refactor(+rename) to be able to handle either single/multiple movies as
# input, and make a new function that handles multiple tiffs specifically (to make split
# binaries into multiple TIFFs as necessary) the multiple-input-movie case is currently
# mostly handled surrounding the call to the function in
# register_all_fly_recordings_together
_rrt_printed_how_to_rerun = False
def register_recordings_together(thorimage_dirs, tiffs, fly_analysis_dir: Path,
    overwrite: bool = False, verbose: bool = False) -> bool:
    # TODO TODO doc w/ clarification on how this is different from
    # register_all_fly_recordings_together
    """

    Args:
        overwrite: whether to overwrite any existant suite2p runs that match the
            currently requested input + parameters. May be useful if suite2p code gets
            updated and behavior changes without parameters or input changing.

    Returns whether registration was successful
    """
    from suite2p import run_s2p

    global _rrt_printed_how_to_rerun

    # TODO TODO option to indicate we only care about suite2p motion correction, and
    # then only re-run if one of the motion correction related parameters differs from
    # current setting (for use with imagej ROI code)

    # TODO refactor so this whole logic (of having multiple runs in parallel and
    # updating a symlink to the one with the params we want) can be used inside
    # recording directories too (not just fly directories), where in those cases the
    # input should be only one recordings data. how to be clear as to whether to use the
    # across run stuff vs single run stuff? just in the gsheet i suppose is fine.
    # (eh...)

    ops = load_default_ops()

    # TODO TODO assert we get the same value for all of these w/ stuff we are trying to
    # register together? (for now just assuming first applies to the rest)
    # TODO TODO or at least check all the shapes are compatible? maybe suite2p has a
    # good enough error message already there tho...
    data_specific_ops = s2p.suite2p_params(thorimage_dirs[0])
    for k, v in data_specific_ops.items():
        assert k in ops
        ops[k] = v
    # TODO TODO TODO try to cleanup suite2p dir (esp if we created it) if we ctrl-c
    # before it finishes (as part of this, delete links if they would be broken by end
    # of cleanup)

    # Path objects don't work in suite2p for this variable.
    ops['tiff_list'] = [str(x) for x in tiffs]

    # TODO if i manage to fix suite2p ops['save_folder'] (so all stuff gets written
    # there rather than everything but the registered binaries, which go to
    # .../'suite2p'/... regardless), could just use that rather than having to make my
    # own directory hierarchy (naming them 'suite2p<n>')
    suite2p_runs_dir = fly_analysis_dir / 'suite2p_runs'
    # TODO include under something like my makedirs empty-dir deletion handling
    # (assuming a crash causes no children to be successfully created)
    suite2p_runs_dir.mkdir(exist_ok=True)

    # Should generally be numbered directories, each of which should contain a single
    # subdirectory named 'suite2p', created by suite2p.
    suite2p_run_dirs = [d for d in suite2p_runs_dir.iterdir() if d.is_dir()]

    # TODO rename fly_analysis_dir if that's all it takes for this fn to basically
    # support multi-tiff and single-tiff input cases
    suite2p_dir_link = s2p.get_suite2p_dir(fly_analysis_dir)

    existing_suite2p_dir_link_target = None
    existing_run_dir = None
    if suite2p_dir_link.exists():
        assert suite2p_dir_link.is_symlink()

        existing_suite2p_dir_link_target = suite2p_dir_link.resolve()

        # .parent, because elements of suite2p_run_dirs do not have final '/suite2p'
        # component of path
        #
        # each numbered run directory contains only a 'suite2p' directory)
        existing_run_dir = existing_suite2p_dir_link_target.parent
        assert existing_run_dir in suite2p_run_dirs

        # Moving current target to start of list, so it will be checked first and
        # nothing will be printed if we don't need to change the mocorr.
        suite2p_run_dirs.remove(existing_run_dir)
        suite2p_run_dirs.insert(0, existing_run_dir)

    # TODO TODO make sure we are deleting directories that are being written too if
    # suite2p is interrupted (or at least on the next run, which could be trickier)

    suite2p_dir = None
    max_seen_s2p_run_num = -1
    for d in suite2p_run_dirs:

        curr_suite2p_dir = d / 'suite2p'

        # TODO at least assert tiff_list is still equal / rerun if not?
        # prob doesn't happen too often that they wouldn't be equal tho...
        # TODO do i have some mechanism preventing register_all_recordings_together from
        # being run on a subset of data for a fly? maybe add one if not
        # TODO TODO move to before the loop (inside code checking if existing
        # suite2p_dir_link exists), because we might have suite2p runs but none of them
        # linked to
        # TODO TODO also refactor so that check run was successful (-> deletion if not)
        # happens before this
        if not rerun_old_param_registrations:
            # TODO TODO fix. got triggered. i must have Ctrl-C'ed before the link was
            # generated? termination of registration should ideally lead to all
            # intermediates it would output being deleted
            # (still an issue? can i replicate?)
            assert existing_suite2p_dir_link_target is not None

            if verbose:
                if not _rrt_printed_how_to_rerun:
                    extra_info = ('. set rerun_old_param_registrations=False to re-run '
                        'if params change'
                    )
                    _rrt_printed_how_to_rerun = True
                else:
                    extra_info = ''

                print('using existing suite2p registration (not checking parameters'
                    f'{extra_info})'
                )

            suite2p_dir = curr_suite2p_dir
            break

        fly_str = '/'.join(fly_analysis_dir.parts[-2:])
        cprint(f'registering recordings for fly {fly_str} to each other...', 'blue',
            flush=True
        )

        # This won't fail if you decide you want to rename some of the run directories.
        # It does have the downside where if we only have some directory <n>, it will
        # lead to directory <n+1> being made, whether or not we have directories in
        # [0, n-1]
        try:
            curr_s2p_run_num = int(d.name)
            max_seen_s2p_run_num = max(max_seen_s2p_run_num, curr_s2p_run_num)
        except ValueError:
            pass

        # TODO should i load ops from 'plane0' dir? 'combined'? does it matter? how do
        # the contents of their 'ops.npy' files differ?
        curr_ops = s2p.load_s2p_ops(curr_suite2p_dir / 'plane0')

        # For the formatting+printing of differences in parameters/inputs.
        # label1 and 2 correspond to curr_ops and ops, respectively.
        diff_kwargs = dict(
            # TODO change 2nd arg to shorten_path to 5 if we are working in a recording
            # directory rather than a fly directory (or change to start from a
            # particular part, like the date, rather than counting parts from the end)
            label1=f'{shorten_path(d, 4)}:', label2='new:',
            header='\nsuite2p inputs differed:'
        )
        if not s2p.inputs_equal(curr_ops, ops, **diff_kwargs):
            continue

        message_prefix = 'Past'
        if (existing_suite2p_dir_link_target is not None and
            existing_suite2p_dir_link_target == curr_suite2p_dir):
            message_prefix = 'Current'

        # TODO again, make shorten_path 2nd arg depend on whether fly/recording level
        print(f'{message_prefix} suite2p run matched requested input & parameters:',
            shorten_path(curr_suite2p_dir, 5)
        )

        if not was_suite2p_registration_successful(curr_suite2p_dir):
            warn('This suite2p run had failed. Deleting.')
            shutil.rmtree(curr_suite2p_dir)
            continue

        suite2p_dir = curr_suite2p_dir
        break

    def make_suite2p_dir_symlink(suite2p_dir: Path) -> None:

        if (existing_suite2p_dir_link_target is not None and
            existing_suite2p_dir_link_target == suite2p_dir):
            return

        # TODO again, mind # of path parts we want, and maybe reimplement to just count
        # from date part to end
        print(f'Linking {shorten_path(suite2p_dir_link)} -> '
            f'{shorten_path(suite2p_dir, 5)}\n'
        )

        # My symlink(...) currently doesn't support existing links, though it will fail
        # (by default) if they exist and point to the wrong target.
        suite2p_dir_link.unlink(missing_ok=True)

        symlink(suite2p_dir, suite2p_dir_link)
        assert suite2p_dir_link.resolve() == suite2p_dir

    if suite2p_dir is None:
        suite2p_run_num = max_seen_s2p_run_num + 1
        suite2p_run_dir = suite2p_runs_dir / str(suite2p_run_num)
        suite2p_dir =  suite2p_run_dir / 'suite2p'
        print('Found no matching suite2p run!\nMaking new suite2p directory under:')
        print(suite2p_run_dir)

        # TODO do i need exist_ok for the parents too, or will it only fail if the final
        # directory already exists? (test w/ and w/o .../sutie2p_runs existing)

        # This is to create directories like:
        # analysis_intermediates/2022-03-10/1/suite2p_runs/0
        # which are the parent of the 'suite2p' directory created within (by run_s2p)
        suite2p_dir.parent.mkdir(parents=True)
    else:
        if overwrite:
            assert False, 'need to define suite2p_run_dir here'
            # TODO TODO TODO need to still have suite2p_run_dir defined in this case, as
            # it is used in db argument to run_s2p (would be more urgent if i was using
            # this overwrite path...)
            print('Deleting past suite2p run because overwrite=True')
            shutil.rmtree(suite2p_dir)
            # TODO do we even need to do this, or will suite2p run_s2p make this anyway?
            # don't make unless we need to
            # (delete if code works correctly with this commented)
            #suite2p_dir.mkdir()
        else:
            make_suite2p_dir_symlink(suite2p_dir)
            return True

    print('Input TIFFs:')
    pprint([str(shorten_path(x, 4)) for x in tiffs])

    # May need to implement some kind of fallback to smaller batch sizes, if I end up
    # experiencing OOM issues with large batches.
    batch_size = sum(thor.get_thorimage_n_frames(d) for d in thorimage_dirs)
    # TODO change 'suite2p' back to 'registration', if batch_size only affects
    # registration
    print(f'Setting {batch_size=} to attempt suite2p in one batch')
    ops['batch_size'] = batch_size

    db = {
        'data_path': [str(fly_analysis_dir)],
        # TODO test all outputs same as if these were not specified (just in a diff
        # folder)
        'save_path0': str(suite2p_run_dir),
    }

    # TODO update suite2p message:
    # "NOTE: not registered / registration forced with ops['do_registration']>1
    #  (no previous offsets to delete)"
    # to say '=1' unless there is actually a time to do '>1'. at least '>=1', no?
    # TODO make suite2p work to view registeration w/o needing to extract cells too

    try:
        # TODO actually care about ops_end?
        # TODO TODO save ops['refImg'] into some plots and see how they look?
        # (if useful, probably also load ops_end from data above, if we can get
        # something equivalent to it from the saved data...)
        # TODO TODO also do the same w/ ops['meanImg']
        ops_end = run_s2p(ops=ops, db=db)

        # TODO also is save path in here? if so, does it include 'suite2p' part?
        # and if i explicitly pass save path, can i avoid need for that part?

        # TODO TODO inspect ops['yoff'], xoff and corrXY to see how close we are to
        # maxregshift?
        # TODO how does corrXY differ from xoff/yoff?
        # TODO TODO are ops_end same as if loading from the ops file in the
        # corresponding plane directory (so ig ops_end must be a list of ops dicts, if
        # that were true?)?
        #import ipdb; ipdb.set_trace()
        print()

    # TODO TODO why did i want to catch arbitrary exceptions here again? document.
    # TODO TODO does ctrl-c also get caught here? would want to cleanup (delete any
    # directories made) if ctrl-c'd, but probably not if there was a regular error
    # (so i can investigate the errors)
    except Exception as err:
        cprint(traceback.format_exc(), 'red', file=sys.stderr)
        # TODO might want to add this back, but need to manage clearing it and stuff
        # then...
        ##make_fail_indicator_file(fly_analysis_dir, suite2p_fail_prefix, err)
        return False

    make_suite2p_dir_symlink(suite2p_dir)

    # TODO TODO uncomment (handle case when file doesn't exist tho, if roi detection not
    # requested)
    '''
    # NOTE: this is not changing the iscell.npy files in the plane<x> folders,
    # as I'm trying to avoid dealing with those folders at all as much as possible
    combined_dir = s2p.get_suite2p_combined_dir(fly_analysis_dir)
    # TODO test this doesn't cause failure even if just running registration steps
    # (then delete try/except)
    try:
        s2p.mark_all_suite2p_rois_good(combined_dir)
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()
    '''

    return True


def suite2p_ordered_tiffs_to_num_frames(suite2p_dir: Path, thorimage_dir: Path
    ) -> Dict[Path, int]:

    # TODO TODO is dtype='int16' a mistake on the part of the suite2p people (they at
    # least seem to be consistent about it w/in io.binary)? shouldn't it be uint16? or
    # can it really be negative? precision loss when converting from uint16 (and that is
    # the type of data i get from thorimage, right? should i rescale?)
    # TODO might need to handle for tests of equavilancy between this and my other raw
    # data formats
    """
    Args:
        suite2p_dir: path to suite2p directory, containing 'plane<x>' folders

        thorimage_dir: path containing ThorImage output (for the Experiment.xml that
            contains movie shape information)

    Returns ordered dict of input TIFF filename -> number of timepoints in the movie.
    Dict order same as concatenation order for suite2p motion correction.
    """
    plane_dirs = sorted(suite2p_dir.glob('plane*/'))

    # TODO maybe factor using 'plane0' into some fn (maybe even load_s2p_ops), to ensure
    # we are being consistent, especially if there actually are any differences between
    # (for example) 'combined' and 'plane<n>' ops.
    #
    # the metadata we care about should be the same regardless of which we plane we load
    # the ops from
    ops = s2p.load_s2p_ops(plane_dirs[0])

    # list of (always?)-absolute paths to input files, presumably in concatenation order
    filelist = [Path(x) for x in ops['filelist']]

    # of same length as filelist. e.g. array([ 756, 1796,  945])
    # converting from np.int64 (not serializable in json)
    frames_per_file = [int(x) for x in ops['frames_per_file']]

    return dict(zip(filelist, frames_per_file))


# TODO factor to hong2p.suite2p
def load_suite2p_binaries(suite2p_dir: Path, thorimage_dir: Path,
    registered: bool = True, to_uint16: bool = True, verbose: bool = False
    ) -> Dict[Path, np.ndarray]:

    # TODO TODO is dtype='int16' a mistake on the part of the suite2p people (they at
    # least seem to be consistent about it w/in io.binary)? shouldn't it be uint16? or
    # can it really be negative? precision loss when converting from uint16 (and that is
    # the type of data i get from thorimage, right? should i rescale?)
    # TODO might need to handle for tests of equavilancy between this and my other raw
    # data formats
    """
    Args:
        suite2p_dir: path to suite2p directory, containing 'plane<x>' folders

        thorimage_dir: path containing ThorImage output (for the Experiment.xml that
            contains movie shape information)

        registered: if True, load registered data in 'data.bin' files, otherwise
            load raw data from 'data_raw.bin' files

        to_uint16: if True, will convert int16 suite2p data to the uint16 type our input
            data is, to make the output more directly comparable with input TIFFs

        verbose: if True, print frame ranges for each input TIFF

    Returns dict of input TIFF filename -> array of shape (t, z, y, x).
    """
    from suite2p.io import BinaryFile

    # without this sort key, plane10 gets put between plane1 and plane2
    plane_dirs = sorted(suite2p_dir.glob('plane*/'),
        key=lambda x: int(x.name[len('plane'):])
    )

    # Depending on ops['keep_movie_raw'], 'data_raw.bin' may or may not exist.
    # Just using to test binary reading (for TIFF creation + direct use).
    name = 'data.bin' if registered else 'data_raw.bin'
    binaries = [d / name for d in plane_dirs]

    # TODO can i just replace this w/ usage of some of the other entries in ops?
    # (currently only use ops in suite2p_ordered_tiffs_to_num_frames)
    (x, y), z, c = thor.get_thorimage_dims(thorimage_dir)
    # TODO TODO actually test on some test data where x != y
    frame_width = x
    frame_height = y

    plane_data_list = []
    for binary in binaries:
        with BinaryFile(frame_height, frame_width, binary) as bf:
            # "nImg x Ly x Lx", meaning T x frame height x frame width
            plane_data = bf.data
            plane_data_list.append(plane_data)

    data = np.stack(plane_data_list, axis=1)

    input_tiff2num_frames = suite2p_ordered_tiffs_to_num_frames(suite2p_dir,
        thorimage_dir
    )

    # TODO figure out how to use this if i want to support loading data from multiple
    # folders (as it would probably be if you managed to use the GUI to concatenate in a
    # particular order, with only one TIFF per each input folder)
    #'frames_per_folder': array([3497], dtype=int32),

    start_idx = 0
    input_tiff2movie_range = dict()
    for input_tiff, n_input_frames in input_tiff2num_frames.items():

        end_idx = start_idx + n_input_frames
        movie_range = data[start_idx:end_idx]
        assert movie_range.shape[0] == n_input_frames

        if verbose:
            # TODO TODO maybe i should write this to a file instead (so i don't need to
            # run right before inspecting...)? factor out if so

            # Converting to 1-based indices, as these prints are intended for inspecting
            # boundaries in ImageJ
            print(f'{shorten_path(input_tiff, n_parts=4)}: {start_idx + 1} - {end_idx}')

        # TODO convert this to a test (pulling in the commented code below that calls
        # this fn w/ registered=False)
        if checks and not registered:
            # TODO double check on thorimage .raw dtype
            #
            # Note: our TIFF inputs are uint16 (as is the ThorImage data, I believe),
            # and suite2p does floor division by 2 before conversion to dtype np.int16
            # (for inputs of dtype np.uint16).
            tiff = tifffile.imread(input_tiff)
            assert np.array_equal(movie_range, (tiff // 2).astype(np.int16))

        if to_uint16:
            assert movie_range.dtype == np.int16

            # TODO could also try just setting to zero since i think it is a small
            # number of pixels (and typically much less negative than max range, e.g.
            # min=-657, max=4387 for 2022-02-04/1/kiwi)
            #
            # This was to deal w/ small negative values (eyeballing it seemed like they
            # were just registration artifacts at the edges, often ~1x3 pixels or so),
            # that got converted to huge positive values when changing to uint16.
            min_pixel = movie_range.min()

            if min_pixel < 0:
                # TODO warn about what fraction of pixels is < 0
                movie_range[movie_range < 0] = 0

                # This seemed to introduce some very noticeable baseline change across
                # experiment boundaries for some recordings (2022-02-07, for instance).
                # Not sure if it would have actually affected correlations much, but
                # still not a great sign.
                # This will make the new minimum 0, shifting everything slightly more
                # positive.
                #movie_range = movie_range - min_pixel

            movie_range = (movie_range * 2).astype(np.uint16)

        input_tiff2movie_range[input_tiff] = movie_range
        start_idx += n_input_frames

    if verbose:
        print()

    return input_tiff2movie_range


def should_flip_lr(date, fly_num) -> Optional[bool]:
    # TODO unify w/ half-implemented hong2p flip_lr metadata key?
    # TODO why .at not working all of a sudden?
    try:
        # np.nan / 'left' / 'right'
        #side_imaged = gdf.at[(date, fly_num), 'side']
        side_imaged = gdf.loc[(date, fly_num), 'side']
        assert not hasattr(side_imaged, 'shape')
    except KeyError:
        side_imaged = None
    # TODO delete (was for troubleshooting .at issue)
    #except ValueError:
    #    import ipdb; ipdb.set_trace()
    #

    # TODO only warn if fly has at least one real experiment (that is also
    # has frame<->odor assignment working and everything)

    if pd.isnull(side_imaged):
        # TODO maybe err / warn w/ higher severity (red?), especially if i require
        # downstream analysis to use the flipped version
        # TODO don't warn if there are no non-glomeruli diagnostic recordings
        # for a given fly? might want to do this even if i do make skip handling
        # consistent w/ process_recording.
        # TODO TODO or maybe just warn separately if not in spreadsheet (but only if
        # some real data seems to be there. maybe add a column in the sheet to mark
        # stuff that should be marked bad and not warned about too)
        warn(f'fly {format_date(date)}/{fly_num} needs side labelled left/right'
            ' in Google Sheet'
        )
        # This will produce a tiff called 'raw.tif', rather than 'flipped.tif'
        flip_lr = None
    else:
        assert side_imaged in ('left', 'right')
        flip_lr = (side_imaged != standard_side_orientation)

    return flip_lr


def convert_raw_to_tiff(thorimage_dir, date, fly_num) -> None:
    """Writes a TIFF for .raw file in referenced ThorImage directory
    """
    flip_lr = should_flip_lr(date, fly_num)

    analysis_dir = get_analysis_dir(date, fly_num, thorimage_dir)

    # TODO just (sym)link flipped.tif->raw.tif if we don't need to flip?

    # TODO maybe delete any existing raw.tif when we save a flipped.tif
    # (so that we can still see the data for stuff where we haven't labelled a
    # side yet, but to otherwise save space / avoid confusion)?

    # Creates a TIFF <analysis_dir>/flipped.tif, if it doesn't already exist.
    util.thor2tiff(thorimage_dir, output_dir=analysis_dir, if_exists='ignore',
        # TODO TODO TODO fix so discard_channel_b not necessary
        flip_lr=flip_lr, discard_channel_b=True, check_round_trip=checks, verbose=True
    )


# TODO factor type hint for odor_data to hong2p + expand type of odor_lists (-> make
# hong2p type alias for that odor_lists type)
def write_trial_and_frame_json(thorimage_dir, thorsync_dir, err=False
    ) -> Optional[Tuple[Path, Dict[str, Any], list]]:
    """
    Will recompute if global ignore_existing explicitly included 'json'.

    Returns (yaml_path, yaml_data, odor_lists), as returned by
    `util.thorimage2yaml_info_and_odor_lists`, or None if no YAML file.
    """
    analysis_dir = thorimage2analysis_dir(thorimage_dir)
    json_fname = analysis_dir / trial_and_frame_json_basename

    try:
        yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(
            thorimage_dir
        )
        odor_data = yaml_path, yaml_data, odor_lists

    except NoStimulusFile as e:
        # TODO if it's just like an extra 1-2 frames (presumably at end, but not sure if
        # there's a way to know), maybe i should just ignore that discrepancy?  (could
        # compound if i'm concatenating recordings together tho...)
        # TODO try to determine if the extra 1-2 frames are always at end (and if so,
        # just fix frame<->trial assignment code, with this in mind)
        warn(f'{e}. could not write {json_fname}!')
        return None

    # NOTE: NOT recomputing these if -i/--ignore-existing passed with no string argument
    # (which will recompute most of the other things this CLI arg applies to)
    if not should_ignore_existing('json') and json_fname.exists():
        return odor_data

    # TODO option for CLI --ignore-existing flag to recompute these frame<->odor
    # assignments?
    has_failed, fail_suffixes = last_fail_suffixes(analysis_dir)
    if has_failed and frame_assign_fail_prefix in fail_suffixes:
        failed_assigning_frames_to_odors.append(thorimage_dir)

        # If we manually copied the cached frame<->trial assignment YAML from another
        # similar experiment, we want the analysis to be able to proceed as if those
        # assignments were correct.
        if has_cached_frame_odor_assignments(analysis_dir):
            # TODO say which experiment it was copied from (assuming it is specified in
            # YAML; if it is not specified probably warn that it is unclear where YAML
            # came from)
            # TODO elsewhere we are using the cached assignments, warn if dirs in YAML
            # don't match current experiment (WHETHER OR NOT there is a fail indicator
            # file in the analysis dir)
            warn(f'{shorten_path(thorimage_dir)} previously failed frame<->odor '
                'assignment, but using assignment present in YAML in analysis '
                'directory'
            )
        else:
            warn(f'skipping {shorten_path(thorimage_dir)} with previously failed '
                'frame<->odor assignment'
            )
            return odor_data

    try:
        bounding_frames = assign_frames_to_odor_presentations(thorsync_dir,
            thorimage_dir, analysis_dir
        )

    # TODO convert any remaining expected-to-sometimes-trigger AssertionErrors to their
    # own suitable error type -> catch those
    except (OnsetOffsetNumMismatch, AssertionError) as e:
        if err:
            raise
        else:
            failed_assigning_frames_to_odors.append(thorimage_dir)
            make_fail_indicator_file(analysis_dir, frame_assign_fail_prefix, e)

            # TODO print full traceback / add message to warning about this fn
            warn(f'{e}. could not write {json_fname}!')
            return odor_data

    # Currently seems to reliably happen iff we somehow accidentally also image with the
    # red channel (which was allowed despite those channels having gain 0 in the few
    # cases so far)
    if len(bounding_frames) != len(odor_lists):
        err_msg = (f'len(bounding_frames) ({len(bounding_frames)}) != len(odor_lists) '
            f'({len(odor_lists)}). not able to make {json_fname}!'
        )
        warn(err_msg)

        failed_assigning_frames_to_odors.append(thorimage_dir)
        make_fail_indicator_file(analysis_dir, frame_assign_fail_prefix, err_msg)
        return odor_data

    odor_order_with_repeats = [format_odor_list(x) for x in odor_lists]

    # For trying to load in ImageJ plugin (assuming stdlib json module works there)
    json_dicts = []
    for trial_frames, trial_odors in zip(bounding_frames, odor_lists):

        start_frame, first_odor_frame, end_frame = trial_frames

        trial_json = {
            'start_frame': start_frame,
            'first_odor_frame': first_odor_frame,
            'end_frame': end_frame,
            # TODO use abbrevs / at least include another field with them
            'odors': format_odor_list(trial_odors),
        }

        # Mainly want to get the 'glomerulus' value, when specified (for diagnostic
        # odors), but might as well include any other odor data.
        #
        # NOTE: not currently supporting odor metadata passthru for trials with multiple
        # odors
        if len(trial_odors) == 1:
            trial_odor_dict = trial_odors[0]

            for k, v in trial_odor_dict.items():
                assert k not in trial_json
                trial_json[k] = v

        json_dicts.append(trial_json)

    json_fname.write_text(json.dumps(json_dicts))
    return odor_data


fly2anat_thorimage_dir = dict()
# TODO probably just transition to always saving my ROIs at the root by default
# (maybe / maybe not supporting experiment specific ROIs that can override the top-level
# ones?)
#
# Only used to check diagnostic experiment analysis directory for ImageJ ROIs
# (RoiSet.zip), to use those ROIs when other experiments don't have their own.
fly2diag_thorimage_dir = dict()
def preprocess_recordings(keys_and_paired_dirs, verbose=False) -> None:
    """
    - Writes a TIFF for .raw file in referenced ThorImage directory
    - Writes a JSON with trial/frame info
    - Populates fly2diag_thorimage_dir with particular diagnostic recordings for each
      fly that has them, so other code can use ROIs defined there if other recordings
      don't have their own.
    - Adds any odors with abbreviations defined in olfactometer YAMLs to odor2abbrev.
    """
    for (date, fly_num), (thorimage_dir, thorsync_dir) in keys_and_paired_dirs:

        # TODO also print directories here if CLI verbose flag is true?

        fly_key = (date, fly_num)

        # Since the ThorImage <-> ThorSync pairing is currently set up to exclude 'anat'
        # recordings (mainly because I often don't save ThorSync recordings for them,
        # but also because they would need their own analysis).
        # TODO TODO TODO uncomment
        '''
        if fly_key not in fly2anat_thorimage_dir:
            # TODO TODO deal with (or just fix data layout for)
            # 2021-05-*/[other|without_thorsync]/anat* stuff? any examples in older data
            # currently just on dropbox?
            fly_dir = thorimage_dir.parent

            # rglob rather than glob, to also handle cases in older data (e.g. 2021-05*)
            # where I had copied anat recording to a subfolder, to side step issues with
            # ThorImage <-> ThorSync pairing at the time.
            anat_dirs = list(fly_dir.rglob('anat*'))
            # TODO TODO warn if 0 / >1, rather than asserting
            assert len(anat_dirs) == 1
            anat_thorimage_dir = anat_dirs[0]

            fly2anat_thorimage_dir[fly_key] = anat_thorimage_dir
        '''

        panel = get_panel(thorimage_dir)
        if panel == diag_panel_str:
            # Assuming we just want to use the chronologically first one.
            # Otherwise maybe natsort on thorimage_dir.names?
            if fly_key not in fly2diag_thorimage_dir:
                fly2diag_thorimage_dir[fly_key] = thorimage_dir

        if do_convert_raw_to_tiff:
            # TODO if we wrote a flipped.tif for a fly, that should trigger updating any
            # outputs that depend on it (nonroi stuff esp, but probably just everything)
            convert_raw_to_tiff(thorimage_dir, date, fly_num)

        # TODO maybe also don't do this in is_acquisition_host case?
        # just don't even call preprocess_recordings? then delete
        # do_convert_raw_to_tiff flag?
        odor_data = write_trial_and_frame_json(thorimage_dir, thorsync_dir)
        if odor_data is None:
            continue

        yaml_path, _, odor_lists = odor_data
        olf.add_abbrevs_from_odor_lists(odor_lists, yaml_path=yaml_path)

        # TODO delete
        # (can't use solvent info to exclude va/aa in water, at least not w/o fixing
        # generated yamls for megamat experiments, which still don't specify
        # solvent='water' for those despite experiments after 2023-04-26 always using
        # water for them (no matter for which panel))
        '''
        _printed = False
        for trial_odors in odor_lists:
            for odor in trial_odors:
                if odor['abbrev'] not in ('va', 'aa'):
                    continue

                if not _printed:
                    print()
                    print(shorten_path(thorimage_dir))
                    print(shorten_path(yaml_path))
                    _printed = True

                solvent = odor.get('solvent', 'unspecified')
                print(f'{odor["abbrev"]}: {solvent=}')

        if _printed:
            print()
        '''
        #

    # also happens automatically atexit, but saving explicitly here incase there is an
    # unclean exit
    olf.save_odor2abbrev_cache()


# TODO TODO flag to not change re-link mocorr directories (so that this script can be
# run on a subset of data to use diff mocorr params for them, but then overall analysis
# can be run without re-running mocorr (potentially producing worse results) for the
# other experiments)
# TODO doc
def register_all_fly_recordings_together(keys_and_paired_dirs, verbose: bool = False):
    # TODO TODO doc w/ clarification on how this is different from
    # register_recordings_together

    # TODO try to skip the same stuff we would skip in the process_recording loop
    # below (and at that point, maybe also treat converstion to tiffs above
    # consistently?)

    date_and_fly2thorimage_dirs = defaultdict(list)
    for (date, fly_num), (thorimage_dir, _) in keys_and_paired_dirs:
        # Since keys_and_paired_dirs should have everything in chronological order,
        # these ThorImage directories should also be in the order they were acquired
        # in, which might make registering them all into one big movie more
        # easily interpretable. Might do this for some intermediates.
        date_and_fly2thorimage_dirs[(date, fly_num)].append(thorimage_dir)

    # TODO option to suppress suite2p output (most of it is logged to
    # ./suite2p/run.log anyway), so i can tqdm this loop (nevermind, i can't seem to
    # find run.log files when i invoke via this code, rather than via the gui...
    # bug?)
    for (date, fly_num), all_thorimage_dirs in date_and_fly2thorimage_dirs.items():

        date_str = format_date(date)
        fly_str = f'{date_str}/{fly_num}'
        fly_analysis_dir = get_fly_analysis_dir(date, fly_num)

        thorimage_dirs = []
        tiffs = []
        for thorimage_dir in all_thorimage_dirs:
            panel = get_panel(thorimage_dir)
            # TODO may also wanna try excluding 'glomeruli_diagnostic' panel
            # recordings
            # TODO warn though?
            if panel is None:
                continue

            # TODO may also need to filter on shape (hasn't been an issue so far)

            analysis_dir = get_analysis_dir(date, fly_num, thorimage_dir)
            tiff_fname = analysis_dir / 'flipped.tif'
            del analysis_dir
            if not tiff_fname.exists():
                warn(f'TIFF {tiff_fname} did not exist! will not include in across '
                    'recording registration (if there are any other valid TIFFs for'
                    ' this fly)!'
                )
                continue

            thorimage_dirs.append(thorimage_dir)
            tiffs.append(tiff_fname)

        if len(thorimage_dirs) == 0:
            # TODO TODO better error message if we are only in this situation because we
            # didn't have side (left/right) labelled in gsheet (and thus didni't have
            # the flipped.tif files) (maybe just need a message somewhere else in that
            # case?)
            warn(f'no panels we want to register for fly {fly_str}. modify get_panel to'
                ' return a panel str for matching ThorImage output directories! '
                # TODO test!
                f'ThorImage output dirs for this fly:\n{pformat(all_thorimage_dirs)}\n'
            )
            continue

        # TODO write text/yaml not in suite2p directory explaining what all data
        # went into it (and which frames boundaries correspond to what, for manual
        # inspection)
        # TODO TODO or maybe make as part of concat trial/frame JSON?

        success = register_recordings_together(thorimage_dirs, tiffs, fly_analysis_dir,
            verbose=verbose
        )
        if not success:
            continue

        # "raw" in the sense that they aren't motion corrected. They may still be
        # flipped left/right according to labelling of which side I imaged (from the
        # labels in the Google Sheet).
        raw_concat_tiff = fly_analysis_dir / 'raw_concat.tif'
        if not raw_concat_tiff.is_file():
            raw_tiffs = [tifffile.imread(t) for t in tiffs]
            raw_concat = np.concatenate(raw_tiffs, axis=0)
            print(f'writing {raw_concat_tiff}', flush=True)
            util.write_tiff(raw_concat_tiff, raw_concat)
            del raw_tiffs

        # TODO TODO make temporary code to read existing TIFFs (or at least the
        # mocorr_concat.tif ones, probably) that are derived from suite2p binaries, and
        # prints the min/max/dtype of the movies, to make sure we shouldn't be getting
        # divide by zero-type errors anymore. re-run stuff as necessary (or at least
        # change code to generate from binaries, and re-run that).
        # (probably just do this for directly the suite2p binaries themselves)

        # This Path is a symlink to a particular suite2p run, created and updated by
        # register_recordings_together (above)
        suite2p_dir = s2p.get_suite2p_dir(fly_analysis_dir)
        assert suite2p_dir.is_symlink()


        # TODO TODO also check if concat json has length equal to sum of all input
        # json lengths (because maybe another json was added after suite2p run, e.g. if
        # a frame<->trial YAML was copied into an experiment dir to allow analysis to
        # proceed but *AFTER* suite2p was run)

        # TODO probably add '_concat' in name to be consistent w/ other stuff in fly
        # analysis directories
        trial_and_frame_concat_json = suite2p_dir / trial_and_frame_json_basename
        trial_and_frame_concat_json_link = (
            fly_analysis_dir / trial_and_frame_json_basename
        )
        # This shouldn't need to change enough to be worth checking mtimes.
        if should_ignore_existing('json') or not (
            trial_and_frame_concat_json.exists() and
            trial_and_frame_concat_json_link.exists()):

            input_tiff2num_frames = suite2p_ordered_tiffs_to_num_frames(suite2p_dir,
                thorimage_dirs[0]
            )

            json_dicts = []
            curr_frame_offset = 0
            for input_tiff, n_frames in input_tiff2num_frames.items():
                json_fname = input_tiff.parent / trial_and_frame_json_basename
                if not json_fname.exists():
                    err_msg = (f'{shorten_path(json_fname, n_parts=5)} did not exist! '
                        'can not create (complete) concatenated trial/frame JSON!'
                    )
                    if allow_missing_trial_and_frame_json:
                        warn(err_msg)
                        continue
                    else:
                        raise FileNotFoundError(err_msg)

                # TODO maybe also save input recording thorimage folder name?
                curr_json_dicts = json.loads(json_fname.read_text())
                for d in curr_json_dicts:
                    d['start_frame'] += curr_frame_offset
                    d['first_odor_frame'] += curr_frame_offset
                    d['end_frame'] += curr_frame_offset

                json_dicts.extend(curr_json_dicts)
                curr_frame_offset += n_frames

            trial_and_frame_concat_json.write_text(json.dumps(json_dicts))

            symlink(trial_and_frame_concat_json, trial_and_frame_concat_json_link)


        mocorr_concat_tiff = suite2p_dir / mocorr_concat_tiff_basename

        def input_tiff2mocorr_tiff(input_tiff):
            return suite2p_dir / f'{input_tiff.parent.name}.tif'

        expected_tiffs = [input_tiff2mocorr_tiff(t) for t in tiffs]
        expected_tiffs.append(mocorr_concat_tiff)

        have_all_tiffs = True
        for t in expected_tiffs:
            if not t.is_file():
                have_all_tiffs = False
                break
            else:
                assert not t.is_symlink(), ('all elements of expected_tiffs should be '
                    'real files, not symlinks'
                )

        # TODO may also want to test we have all the symlinks we expect
        # (AND may need some temporary code to either delete all existing links that
        # should be relative but currently aren't, or may need to do that manually)
        if have_all_tiffs:
            continue

        # TODO TODO convert to test + factor load_suite2p_binaries into hong2p
        #if checks:
        #    # This has an assertion inside that the raw matches the input TIFF
        #    load_suite2p_binaries(suite2p_dir, thorimage_dirs[0], registered=False)

        # TODO TODO also write ops['refImg'] and some of the more useful
        # registration quality metrics from ops (there are some in there, yea?)

        # TODO refactor how this is currently printing frame ranges for each input
        # movie so i can also write a a file, for reference when inspecting TIFF
        #
        # As long as the shape for each movie is the same (which it should be if we
        # got this far), it doesn't matter which thorimage_dir we pass, so I'm just
        # taking the first.
        input_tiff2registered = load_suite2p_binaries(suite2p_dir, thorimage_dirs[0],
            verbose=True
        )
        # TODO double check that this iterates through in acquisition order
        # (+clarify in doc of load_suite2p_binaries)
        # (part generating concat array, after this loop, already seems to assume that)
        for input_tiff, registered in input_tiff2registered.items():

            motion_corrected_tiff = input_tiff2mocorr_tiff(input_tiff)

            print(f'writing {motion_corrected_tiff}', flush=True)
            util.write_tiff(motion_corrected_tiff, registered)

            motion_corrected_tiff_link = input_tiff.with_name('mocorr.tif')

            # For example:
            # (link) analysis_intermediates/2022-02-04/1/kiwi/mocorr.tif ->
            # (file) analysis_intermediates/2022-02-04/1/suite2p/kiwi.tif
            #
            # Since 'suite2p' in the target of the link is itself a symlink,
            # these links should not need to be updated, and the files they refer to
            # will change when the directory the 'suite2p' link is pointing to does.
            symlink(motion_corrected_tiff, motion_corrected_tiff_link)

        # Essentially the same one I'm pulling apart in the above function, but we
        # are just putting it back together to be able to make it a TIFF to inspect
        # the boundaries.
        mocorr_concat = np.concatenate(
            [x for x in input_tiff2registered.values()], axis=0
        )
        print(f'writing {mocorr_concat_tiff}', flush=True)
        util.write_tiff(mocorr_concat_tiff, mocorr_concat)
        del mocorr_concat

        mocorr_concat_tiff_link = fly_analysis_dir / mocorr_concat_tiff_basename
        symlink(mocorr_concat_tiff, mocorr_concat_tiff_link)

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


def format_fly_and_roi(fly_and_roi) -> str:
    fly, roi = fly_and_roi
    return f'{fly}: {roi}'


# TODO type hint w/ appropriate type from hong2p.types
def format_fly_key_list(fly_key_list) -> str:
    return pformat([f'{format_date(d)}/{n}' for d, n in sorted(fly_key_list)])


# TODO refactor stuff to use this
# TODO datelike type? have one already?
def format_datefly(date, fly_num: int) -> str:
    return f'{format_date(date)}/{fly_num}'


def odor_str2conc(odor_str: str) -> float:
    if odor_str == solvent_str:
        return 0.0

    log10_conc = olf.parse_log10_conc(odor_str, require=True)
    return np.float_power(10, log10_conc)


# TODO maybe factor to natmix/hong2p?
# TODO TODO also support DataFrame corr_list input
# TODO switch to taking plot dir and compute (what is currently passed as) output_root
# from .parent (most things just take plot_dirs)?
def plot_corrs(corr_list: List[xr.DataArray], output_root: Path, plot_relpath: Pathlike,
    *, per_fly_figs: bool = True) -> None:
    # TODO is it true that fly_panel_id must be there in per_fly_figs=True case?
    # if so, add to doc OR factor computation of that into here, so it isn't required on
    # input
    """
    Args:
        corr_list: list of correlation DataArrays, each from a different fly.

        output_root: path to save outputs. If CSVs are written, will be written directly
            in this path.

        plot_relpath: relative path from `<output_root>/<plot_fmt>` to directory to
            save plots under
    """

    # TODO should i assert this is in the input + each list element has a different
    # one? or not matter? delete if not.
    #corr_group_var = 'fly_panel_id'

    corrshapes2counts = dict(Counter(x.shape for x in corr_list))
    if len(corrshapes2counts) > 1:
        # TODO TODO TODO print which fly/recording IDs that correspond to the counts
        # (probably in place of the counts?)
        # (could just compute here if there is an issues with the counts...)
        warn('correlation shapes unequal (in plot_corrs input)! shapes->counts: '
            f'{pformat(corrshapes2counts)}\n'
            'some mean correlations will have more N than others!'
        )
        # TODO TODO TODO some output to vizually represent how many N i have for
        # each correlation entry? or throw out all shapes but the max shape / most
        # common shape (and warn)?
        # TODO delete
        #import ipdb; ipdb.set_trace()

    corr_plot_root = output_root / plot_fmt / plot_relpath
    makedirs(corr_plot_root)

    # TODO try to swap dims around / concat in a way that we dont get a bunch of
    # nans. possible?
    corr_avg_dim = 'fly_panel'
    corrs = xr.concat(corr_list, corr_avg_dim)

    # TODO TODO refactor this concat -> assign_scalar_coords_to_dim to hong2p.xarray fn
    corrs = assign_scalar_coords_to_dim(corrs, corr_avg_dim)

    # TODO .copy() *ever* useful in these .sel calls?

    # .sel / .loc didn't work here cause 'panel' isn't (always) part of an index,
    # despite always being associated w/ the 'fly_panel' dimension
    #corrs = corrs.sel(panel=diag_panel_str).copy()
    corrs = corrs.where(corrs.panel != diag_panel_str, drop=True)

    # TODO don't drop all of this if i can get subsetting via drop_nonlone... to work
    corrs = corrs.sel(is_pair=False, is_pair_b=False).copy()

    corrs = corrs.dropna(corr_avg_dim, how='all')

    # TODO TODO serialize corrs for scatterplot directly comparing KC and ORN
    # correlation consistency in the same figure/axes?

    # TODO make a fn for grouping -> mean (+ embedding number of flies [maybe also w/ a
    # separate list of metadata for those flies] in the DataArray attrs or something)
    # -> maybe require those type of attrs for corresponding natmix plotting fn?
    #
    # TODO better error message if than:
    # ValueError: panel must not be empty
    # 2023-11-08: can repro above via:
    # ./al_analysis.py -d pebbled -n 6f -s intensity,model -t 2023-07-29 \
    # -i ijroi,nonroi 2023-10-19/1/diag
    #
    for panel, garr in corrs.reset_index(['odor', 'odor_b']).groupby('panel',
        squeeze=False, restore_coord_dims=True):

        panel_dir = corr_plot_root / panel
        # only really need to make this explicitly if i wanna write CSVs before first
        # savefig call in here (which would make it behind-the-scenes)
        makedirs(panel_dir)

        garr = dropna_odors(garr)

        # garr.mean(...) will otherwise throw out some / all of this information.
        meta_dict = {
            'date': garr.date.values,
            'fly_num': garr.fly_num.values,
        }
        if hasattr(garr, 'thorimage_id'):
            meta_dict['thorimage_id'] = garr.thorimage_id.values

        # garr seems to be of shape:
        # (# flies[/recordings sometimes maybe], # odors, # odors)
        # TODO may need to fix. check against dedeuping below
        # TODO TODO TODO update computation to work for megamat stuff
        # (still relevant?  what all does this effect? just N in filenames?)
        # (where len(garr) seems to be # of unique (date, fly_num, **thorimage_id**),
        # when now we don't want to count multiple thorimage_id as diff n (prob didn't
        # before either...)
        # may need to change groupby/something above
        n_flies = len(garr)

        if panel != 'megamat':
            kwargs = dict()
        else:
            kwargs = remy_matshow_kwargs

        panel_mean = garr.mean(corr_avg_dim)

        # TODO TODO probably convert to a kwarg to save CSVs / always save CSV?
        # (would need to take additional arg/something to make name unique, if i want to
        # save this for more than just imagej inputs)
        # TODO what does this condtional mean? rename vars / comment clarifying?
        # (when is across_fly_ijroi_dirname not in plot_relpath parts?)
        if across_fly_ijroi_dirname in Path(plot_relpath).parts:
            df = move_all_coords_to_index(panel_mean).to_pandas()

            df = df.droplevel('odor2').droplevel('odor2_b', axis='columns')

            # TODO TODO drop identity correlations (will have 1 avgd in, as implemented
            # now)
            mdf = df.groupby('odor1').mean().groupby('odor1_b', axis='columns').mean()

            # TODO rename 'odor1' to just 'odor' in csv
            # TODO TODO TODO save a version w/ correlations for each fly separately
            # (to compute the error different ways / decide to drop particular
            # odors/trials)
            # TODO TODO TODO also save at least one error metric too (B: standardize on
            # SD now, as she thinks that's what Remy is using now? check that.)

            # TODO TODO TODO still verify we aren't saving to the same file twice in one
            # run, which would indicate a bug! factor out all fns that write serious
            # outputs to something that checks that (e.g. to_csv, to_pickle,
            # write_dataarray, etc)
            # TODO TODO TODO maybe for this and most things in top level of output_root,
            # prefix <driver> (or <driver>_<indicator>?) to filename?
            # TODO delete
            print('PREFIX DRIVER/INDICATOR IF I CAN (WORTH PASSING? COMPUTE FROM DIR)')
            #import ipdb; ipdb.set_trace()
            #
            csv_name = f'{panel}_ijroi_corr_n{n_flies}.csv'
            print('RESTORE CSV SAVING (AFTER DE-DUPING...)')
            #to_csv(mdf, output_root / csv_name)

        # TODO TODO change so i don't need to manually pass in name_order?
        # TODO will this have broken any usage outside of megamat case?
        # (previously i wasn't defining this nor passing it in to natmix.plot_corrs)
        name_order = panel2name_order.get(panel)

        # TODO pass warn=True if script run with -v flag?
        fig = natmix.plot_corr(panel_mean, title=f'{panel} (n={n_flies})',
            name_order=name_order, **kwargs
        )
        savefig(fig, corr_plot_root, f'{panel}_mean')

        # TODO TODO update error calculation to match whatever remy is using
        # (probably for everything, but at least for megamat stuff, as well as my
        # modeling in that context)
        panel_sem = garr.std(corr_avg_dim, ddof=1) / np.sqrt(n_flies)
        # TODO try (a version) w/o forcing same scale (as it currently does)
        fig = natmix.plot_corr(panel_sem,
            title=f'{panel} (n={n_flies})\nSEM for mean of correlations',
            name_order=name_order, **kwargs
        )
        savefig(fig, corr_plot_root, f'{panel}_sem')

        # TODO factor out the correlation consistency plotting code to its own fn (maybe
        # in natmix?) and call here

        fly_panel_sers = []
        for arr in garr:
            coord_dict = scalar_coords(arr)
            arr = move_all_coords_to_index(drop_scalar_coords(arr))

            # TODO add flag to this to exclude the diagonal itself?
            # TODO why do i have keep_duplicate_values=True? is it that otherwise only a
            # weird subset (half, but maybe in a weird order) of facets are populated?
            # add comment explaining
            ser = util.melt_symmetric(arr.to_pandas(), suffixes=('', '_b'),
                keep_duplicate_values=True
            )

            # TODO add an option to not do this, esp when factoring out corr consistency
            # plotting. probably still drop the diagonal itself, leaving just the values
            # from different trials to average over.
            #
            # Dropping all correlations between an odor and itself, before averaging
            # over trials, so we don't have some averages that include the perfect
            # correlations on the diagonal (or even just 3/6 values to average vs 9 for
            # other correlations).
            ser = ser[
                ser.index.get_level_values('odor1') !=
                ser.index.get_level_values('odor1_b')
            ]

            mean_ser = ser.groupby(['odor1', 'odor1_b']).mean()

            mean_ser = util.addlevel(mean_ser, coord_dict.keys(), coord_dict.values())
            fly_panel_sers.append(mean_ser)

        panel_tidy_corrs = pd.concat(fly_panel_sers).reset_index(name='correlation')
        assert sum(len(x) for x in fly_panel_sers) == len(panel_tidy_corrs)

        panel_tidy_corrs = sort_odors(panel_tidy_corrs)

        # TODO TODO TODO factor some general pfo dropping into hong2p.olf (or natmix) (+
        # try to support odor vars being in diff places (index/columns/levels of those)
        # & DataArrays)
        panel_tidy_corrs = panel_tidy_corrs[ ~(
            panel_tidy_corrs.odor1.str.startswith('pfo') |
            panel_tidy_corrs.odor1_b.str.startswith('pfo')
        )]

        # TODO TODO TODO maybe plot on same axis as KC data, w/ diff markers?
        # (probably w/o hue='fly_id' in these plots)
        # (need to get that data from remy)

        # TODO TODO drop correlations i don't care about (or do in a plotting fn, maybe
        # a new one in natmix.viz)
        # TODO TODO TODO maybe only have three Axes: one with highest mix conc (vs
        # components), one with just mixes compared together (maybe just highest vs
        # other two?), and one w/ top component (vs components and mix?)
        # (or have those in addition to one plot showing ~all correlations)

        # TODO if i were to reimplement this to be more general (detecting which odors
        # have multiple concs, rather than hardcoding), then factor it out
        def format_xtick_odor(ostr):
            # These are the only odors for which, within a panel, the concentration
            # varies, so we need that information for these odors.
            # TODO update this code to not just assume these are the only w/
            # concentration varying within a panel. actual compute which that is true
            # for.
            # TODO TODO update to kmix/cmix, in time
            if any(ostr.startswith(p) for p in ('~kiwi', 'control mix')):
                return ostr
            # For the other odors, we can drop the concentration information to tidy up
            # the xticklabels.
            else:
                assert ostr != solvent_str
                return olf.parse_odor_name(ostr)

        # TODO TODO also thread thru if_panel_missing here (maybe just set to None?)
        # (after implementing in hong2p.olf, to match behavior as in olf.sort_odors)
        #
        # TODO prevent this from formatting odors from columns ['odor1','odor1_b'] as a
        # mixture (or at least provide kwarg to control).  only stuff like ['odor1<x>',
        # 'odor2<x>'] should be formatted as such. it should probably get order from
        # union of all odor columns when not to be formatted a mixture.
        # NOTE: do not need to pass these through the formatter fn supplied to catplot
        odor_order = olf.panel_odor_orders(panel_tidy_corrs[['panel', 'odor1_b']],
            panel2name_order
        )[panel]

        # This will add a 'fly_id' column from unique ['date','fly_num'] combinations.
        # Internal sorting should produce same palette as in other places, given same
        # input.
        # TODO maybe call this earlier, so it also gets saved w/ meta_df?
        # doesn't currently work w/ dataarray input tho, if that matters...
        fly_id_palette = natmix.get_fly_id_palette(panel_tidy_corrs)

        with mpl.rc_context({'figure.constrained_layout.use': False}):
            # TODO TODO show errorbars (SEM?) (to also show points, would need to use a
            # FacetGrid, and map in two calls / use custom plotting fn)
            # TODO TODO maybe use same vmin/vmax as correlation heatmaps
            # (at least so it's consistent across kiwi/control panels)
            g = sns.catplot(data=panel_tidy_corrs, col='odor1', col_wrap=5, x='odor1_b',
                order=odor_order, y='correlation', hue='fly_id', palette=fly_id_palette,
                kind='swarm', legend=False,
                # Note this required upgrading from seaborn 0.11.2 to 0.12.0
                formatter=format_xtick_odor,
                # Default marker size of 5 produced warning about not being able to
                # place points.
                size=4
            )
            g.set_xticklabels(rotation=90)
            g.set_titles(col_template='{col_name}')
            g.set_xlabels('odor')
            g.fig.suptitle(f'{panel}\ncorrelation consistency')
            # TODO if i still intend to share y, is it necessary to disable sharey just
            # to make this call work (doesn't seem so. compare to plot w/o changing
            # limits tho)?
            # TODO especially if seaborn doesn't warn/fail, warn/fail if some data
            # is out of this hardcoded range
            # TODO may want to use a diff scale in the pixel corr case (currently data
            # seems to occupy ~[0.2, 1.0] on average
            g.set(ylim=(-0.4, 1.0))
            viz.fix_facetgrid_axis_labels(g)
            g.tight_layout()

            # TODO TODO actually show other odors on (probably all of) the x-axes
            # (as xticklabels)

            # TODO TODO fix "<x>% of points cannot be placed" error here (take mean
            # across something first? just change marker size?) (or disable this...
            # i don't use these plots)
            #
            # NOTE: assuming unique w/in corr_plot_root
            # TODO TODO change to 'consistency' under panel dir
            savefig(g, corr_plot_root, f'{panel}_consistency')

        # TODO more idiomatic way? to_dataframe seemed to be too much and
        # to_[series|index|pandas] didn't seem to readily do what i wanted
        # also pd.DataFrame(garr.date.to_series(), <other series>) was behaving strange
        meta_df = pd.DataFrame(meta_dict)

        # TODO probably just delete this whole thing. try to reproduce on old data first
        # tho... require diff corr_group_var (only supporting 'fly_panel_id' now anyway)
        # or something like that?
        if panel != 'megamat':
            try:
                assert len(meta_df) == len(
                    meta_df[['date', 'fly_num']].drop_duplicates()
                )
            except:
                print(f'{meta_df=}')
                import ipdb; ipdb.set_trace()
        #
        assert len(meta_df) == n_flies

        # TODO TODO change to flies.csv under panel dir? (and similar w/ other stuff not
        # under dirs!)
        #
        # Since this is mainly context for the plots, rather than something to be
        # further analyzed, I think it makes more sense for this to stay under one of
        # the <plot_fmt> directories, as it currently is.
        to_csv(meta_df, corr_plot_root / f'{panel}_flies.csv', index=False)

        # TODO delete?
        # (below should only be relevant if i am not dropping is_pair data, as i
        # currently am before loop)
        # TODO TODO serialize + use to check that panel_mean values are the same whether
        # or not we drop the is_pair == True stuff
        # TODO TODO especially if values are NOT the same, need to add some mechanism to
        # make sure we aren't averaging part of data from other types of experiments
        # into the experiment we are actually trying to plot
        '''
        if not per_fly_figs:
            #write_corr_dataarray(panel_mean, f'{panel}_w_pairs.p')
            import ipdb; ipdb.set_trace()
        '''

    if not per_fly_figs:
        return

    # TODO maybe just use squeeze=True to not need squeeze inside? or will groupby
    # squeeze not drop like i want?
    for fly_panel_id, garr in corrs.reset_index(['odor', 'odor_b']).groupby(
        'fly_panel_id', squeeze=False, restore_coord_dims=True):

        panel = unique_coord_value(garr.panel)

        date_str = format_date(unique_coord_value(garr.date))
        fly_num = unique_coord_value(garr.fly_num)
        fly_str = f'{date_str}/{fly_num}'

        # TODO might need to deal w/ NaNs some other way than just dropna (to avoid
        # dropping specific trials w/ baseline NaN issues. or fix the baseline cause
        # of the issue.)
        corr = dropna_odors(garr.squeeze(drop=True))

        if panel != 'megamat':
            kwargs = dict()
        else:
            kwargs = remy_matshow_kwargs

        fig = natmix.plot_corr(corr, title=fly_str, **kwargs)

        fly_plot_prefix = fly_str.replace('/', '_')

        # TODO tight_layout / whatever appropriate to not have cbar label cut off
        # and not have so much space on the left

        fig_path = savefig(fig, panel_dir, fly_plot_prefix)


# TODO rename to remove 'natmix' prefix?
def natmix_activation_strength_plots(df: pd.DataFrame, intensities_plot_dir: Path
    ) -> None:

    # TODO factor to some general test for any natmix data?
    if not df.panel.isin(natmix.panel2name_order).any():
        # Not warning here because it just adds to the noise when I'm not analyzing any
        # natmix experiments.
        return

    makedirs(intensities_plot_dir)

    # TODO before deleting all _checks code, turn into a test of some kind at least
    _checks = False
    _debug = False

    # TODO TODO TODO version of these plots using ROI responses as inputs (refactor)

    # TODO TODO TODO reorganize directories so all (downsampled) pixelbased stuff is in
    # a ds<x> or pixelwise[_ds<x] folder, with folders underneath as necessary, same as
    # ijroi/

    g1 = natmix.plot_activation_strength(df, color_flies=True, _checks=_checks,
        _debug=_debug
    )
    savefig(g1, intensities_plot_dir, 'mean_activations_per_fly', bbox_inches=None)

    g2 = natmix.plot_activation_strength(df, _checks=_checks, _debug=_debug)
    savefig(g2, intensities_plot_dir, 'mean_activations', bbox_inches=None)


response_volume_cache_fname = 'trial_response_volumes.p'
def analyze_response_volumes(response_volumes_list, output_root, write_cache=True):

    # TODO factor all this into a function so i can more neatly call it multiple
    # times w/ diff downsampling factors

    # downsampling factor
    # 192 / 4 = 48
    # TODO TODO better name
    # TODO switch back to 0 / a sequence including this + 0, and change code to try
    # both?
    ds = 4

    #spatial_dims = ['z', 'y', 'x']
    spatial_shapes = [tuple(dict(zip(x.dims, x.shape))[n] for n in spatial_dims)
        for x in response_volumes_list
    ]
    # doesn't handle ties, but that should be fine
    most_common_shape = Counter(spatial_shapes).most_common(1)[0][0]

    # TODO TODO TODO switch to not throwing out any shapes. just need to test the NaNs
    # are handled appropriately in any downsteam correlation / plotting code
    consistent_spatial_shape_response_volumes = []
    for shape, recording_response_volumes in zip(spatial_shapes,
        response_volumes_list):

        # TODO delete
        date = pd.to_datetime(recording_response_volumes.date.item(0)).date()
        fly_num = recording_response_volumes.fly_num.item(0)
        thorimage_id = recording_response_volumes.thorimage_id.item(0)

        # TODO TODO TODO TODO change conversion to uint16 handling to set min to at
        # least 1 so there shouldn't be any divide-by-zero errors, then regenerate
        # mocorr TIFFs and regenerate response volume pickles
        # TODO then delete this code. just a hack.
        n_null = recording_response_volumes.isnull().sum().item(0)
        if n_null > 0:
            warn(f'{date}/{fly_num}/{thorimage_id}: {n_null} null (probably from '
                'divide-by-zero)'
            )
            # TODO TODO TODO update to not allow inf either (which value to use for a
            # max then? should i even limit?)
            recording_max_pixel = recording_response_volumes.max().item(0)
            recording_response_volumes = recording_response_volumes.fillna(
                recording_max_pixel
            )
        #
        # TODO TODO TODO maybe in general i should be capping dF/F at some sane
        # value, like say 10? not like anything higher is plausibly anything other
        # than noise, no?

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

    response_volumes = assign_scalar_coords_to_dim(response_volumes, 'odor')

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

    # TODO TODO maybe rename 'odor' to 'trial' in all xarray things (may need to
    # recompute), or something else more clear (indicating it's also the fly #,
    # repeat, etc, that vary)

    # TODO fix in case only one experiment is here:
    # ValueError: If using all scalar values, you must pass an index
    #
    # TODO fix, now i'm getting (maybe diff circumstances?):
    # (happens when some of the coordinatest are scalars, but not all. e.g.
    # date/fly_num, but not thorimage_id)
    # TypeError: len() of unsized object
    # circumstances for TypeError:
    # ...
    # Coordinates:
    #     panel         (odor) <U21 'glomeruli_diagnostics' ... 'kiwi'
    #     is_pair       (odor) bool False False False False ... True True True True
    #     date          datetime64[ns] 2022-03-30
    #     fly_num       int64 1
    #     thorimage_id  (odor) <U21 'glomeruli_diagnostics' ... 'kiwi_ramps'
    #   * odor          (odor) MultiIndex
    #   - odor1         (odor) object 'ethyl lactate @ -7' ... 'EA @ -4.2'
    #   - odor2         (odor) object 'solvent' 'solvent' ... 'EB @ -3.5' 'EB @ -3.5'
    #   - repeat        (odor) int64 0 1 2 0 1 2 0 1 2 0 1 2 ... 1 2 0 1 2 0 1 2 0 1 2
    #   * z             (z) int64 0 1 2 3 4
    #   * y             (y) int64 0 1 2 3 4 5 6 7 ... 184 185 186 187 188 189 190 191
    #   * x             (x) int64 0 1 2 3 4 5 6 7 ... 184 185 186 187 188 189 190 191
    # ipdb> len(response_volumes)
    # 198
    # ipdb> response_volumes.shape
    # (198, 5, 192, 192)
    # ipdb> response_volumes.dims
    # ('odor', 'z', 'y', 'x')
    response_volumes = util.add_group_id(response_volumes,
        ['date', 'fly_num', 'thorimage_id'], name='recording_id', dim='odor'
    )

    # For calculating correlations of, e.g. kiwi + kiwi_ramps (that had been
    # registered together) in one fly.
    response_volumes = util.add_group_id(response_volumes,
        ['date', 'fly_num', 'panel'], name='fly_panel_id', dim='odor'
    )

    # TODO handle trial_df.p or whatever the same way?
    if write_cache:
        # TODO actually load and use this cache under some circumstances (-c)...
        # otherwise, delete it
        write_corr_dataarray(response_volumes, response_volume_cache_fname)
    else:
        warn('not saving response_volumes to cache because script was run with '
            'positional arguments to restrict the data analyzed'
        )

    # TODO TODO TODO also try calculating correlations only for pixels where the max
    # dF/F (across all trials) is above some threshold (or maybe threshold a pixel
    # z-score?)

    if ds > 0:
        print(f'downsampling response volumes by {ds} in XY...', end='', flush=True)
        response_volumes = response_volumes.coarsen(y=ds, x=ds).mean()
        print('done', flush=True)

    # TODO nicer way to change index?
    # NOTE: set_index `TypeError: unhashable type: 'numpy.ndarray'` seems to come from
    # is_pair.
    ser = response_volumes.mean(spatial_dims).reset_index('odor').set_index(odor=[
        'panel','is_pair','date','fly_num','odor1','odor2','repeat'
        ]).to_pandas()

    mean = ser.groupby([x for x in ser.index.names if x != 'repeat']).mean()
    df = mean.to_frame('mean_dff').reset_index()

    plot_root_dir = output_root / plot_fmt
    intensities_plot_dir = plot_root_dir / 'activation_strengths'
    natmix_activation_strength_plots(df, intensities_plot_dir)

    # TODO TODO TODO maybe rename basename here + factor into
    # natmix_activation_strength_plots (to remove reference to 'pixel') (change any
    # model_mixes_mb code that currently hardcodes this filename)
    # TODO TODO TODO also save the equivalent from the ijroi analysis elsewhere
    # (again, for use w/ model_mixes_mb)
    # TODO need to update name to be more clear what this is, now that it's not in an
    # 'activation_strengths' directory?
    to_csv(df, output_root / 'mean_pixel_dff.csv', index=False)

    # TODO TODO TODO try both filling in pfo for stuff that doesn't have it w/ NaN,
    # as well as just not showing pfo on any of the correlation plots
    # (don't want presence/absence of pfo to prevent some data from otherwise being
    # included in an average correlation)

    # was it only a problem if an experiment type was done twice for one fly?)
    # TODO maybe add an assertion plots don't already exist (assuming we are starting in
    # a fresh directory. otherwise need to maintain a list of plots written in here, and
    # assert we don't see repeats there, within a run of this script.)
    corr_group_var = 'fly_panel_id'

    # TODO memory profile w/ just first group, to be more clear on usage of just the
    # grouping itself and (more importantly) all of the steps in the loop
    # TODO tqdm/time loop
    # TODO profile
    corr_list = []
    _checked = False

    for _, garr in response_volumes.groupby(corr_group_var):

        panel = unique_coord_value(garr.panel)
        if panel == diag_panel_str:
            continue

        date_str = format_date(unique_coord_value(garr.date))
        fly_num = unique_coord_value(garr.fly_num)
        fly_str = f'{date_str}/{fly_num}'

        # TODO try to avoid need for dropping/re-adding metadata?
        # (more idiomatic xarray calls?)

        # TODO TODO factor all this variable re-shuffling stuff to hong2p.xarray?
        # (maybe via a fn that records+drops -> calls callable passed in (corr in
        # one case) -> adds metadata back)
        garr = garr.drop_vars(['thorimage_id', 'recording_id'])

        # TODO TODO possible to generalize to work w/ non-unique stuff too (like
        # thorimage_id, when corr_group_var == 'fly_panel_id')? i feel like it might
        # not be...
        # otherwise, any way to preserve this metadata for later stuff?
        coords_to_drop = ['panel', 'fly_panel_id', 'date', 'fly_num']
        meta = {k: unique_coord_value(garr, k) for k in coords_to_drop}
        garr = garr.drop_vars(coords_to_drop)

        # TODO TODO factor correlation handling to hong2p.xarray

        # TODO factor this out + use in place of code in process_recording that
        # currently deals w/ adding odor index metadata?
        garr = garr.reset_index('odor').set_index(
            odor=['is_pair', 'odor1', 'odor2', 'repeat']
        )

        # TODO TODO compare to what happens if we drop after calculating
        # correlation, so that (if we add this) we can test support for dropping
        # along odor_b dim simultaneously w/ odor dim
        garr = drop_nonlone_pair_expt_odors(garr)

        # TODO clean up...
        garr2 = garr.reset_index('odor').rename(odor='odor_b',
            is_pair='is_pair_b', odor1='odor1_b', odor2='odor2_b',
            repeat='repeat_b').set_index(
            odor_b=['is_pair_b', 'odor1_b', 'odor2_b', 'repeat_b']
        )

        corr = xr.corr(garr, garr2, dim=spatial_dims)

        # TODO also factor this into the fn to remove metadata -> call fn -> reapply
        # it (mentioned in related code above)
        corr = corr.assign_coords(meta)

        if checks and not _checked:
            stacked = garr.stack(pixel=spatial_dims)

            # TODO TODO TODO update to rename odor_b levels as in garr2 above +
            # uncomment
            #corr2 = xr.corr(stacked, stacked.rename(odor='odor_b'), dim='pixel')
            # TODO better way to check coordinates equal?
            #assert corr.coords.to_dataset().identical(corr2.coords.to_dataset())

            # Both of this are True with or without .values after each argument.
            # In both cases, neither seems to have np.array_equal True for the same
            # inputs, so it seems there are some pervasive numerical differences
            # (which shouldn't matter, and I'm not sure which is the more correct of
            # the inputs).
            #assert np.allclose(corr2, corr)
            assert np.allclose(corr, stacked.to_pandas().T.corr())
            _checked = True

        # TODO TODO guaranteed to be false now that i'm only using
        # corr_group_var='fly_panel_id'? delete if so
        if 'thorimage_id' in corr.coords:
            corr = corr.drop_vars(['thorimage_id', 'recording_id'])

        corr_list.append(corr)

    # TODO TODO TODO where is this one w/ duplicate coming from?
    # (index 5)
    # TODO keep metadata so i can better identify stuff like this. at least in
    # attrs...

    # (odor_b was duplicated IFF odor was (just index 5))
    # did i really somehow include two control1 expts for one fly? improper redo
    # handling?
    # TODO delete
    if any([x.odor.to_pandas().index.duplicated().any() for x in corr_list]):
        print('FIX HACK')
        import ipdb; ipdb.set_trace()

    # TODO TODO TODO delete hack (turn into assertion there is nothing duplicated
    # like this -> fix if so)
    orig_corr_list = corr_list
    corr_list = [x for x in corr_list if not
        (x.odor.to_pandas().index.duplicated().any())
    ]

    # Before this, we also have shapes of (42, 42) and (27, 27),  from experiments
    # w/o pfo in kiwi/control panel and w/o pair experiment, respectively.
    # NOTE: have NOT checked whether any other shapes were tossed in filtering above
    #corr_list = [x for x in corr_list if x.shape == (45, 45)]
    #

    pixel_corr_basename = 'pixel_corr'
    if ds > 0:
        pixel_corr_basename = f'{pixel_corr_basename}_ds{ds}'

    # TODO why is this empty when calling on old pair data? is it just b/c i was
    # excluding all but diagnostic recording? are diags dropped by here?
    plot_corrs(corr_list, output_root, pixel_corr_basename)


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
    pair_dir = get_pair_dirname(name1, name2)
    # TODO check consistent w/ other place that generates pair dirs now
    # (in process_recording) (or delete all this old code...)
    import ipdb; ipdb.set_trace()
    if save_figs:
        savefig(fig_or_sns_obj, pair_dir, fname_prefix)


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

    mean_df = sort_concs(
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
ij_roi_responses_cache = 'ij_roi_stats.p'
# TODO TODO delete / factor into current analysis (mainly would want to borrow some of
# the competitive binding stuff, for my mixture experiments)
def analyze_cache():
    recording_keys = fly_keys + ['thorimage_id']

    # TODO TODO TODO plot hallem activations for odors in each pair, to see which
    # glomeruli we'd expect to find (at least at their concentrations)
    # (though probably factor it into its own fn and maybe call in main rather than
    # here?)

    warnings.simplefilter('error', pd.errors.PerformanceWarning)

    df = pd.read_pickle(ij_roi_responses_cache).T

    # This will just be additional presentations of solvent, interspersed throughout the
    # experiment. Not interesting.
    df = df.loc[:, df.columns.get_level_values('repeat')  <= 2].copy()

    import ipdb; ipdb.set_trace()
    # TODO TODO figure out if this is only broken for data i don't actually want to push
    # through here, but maybe also just fix anyway
    # TODO replace w/ hong2p add_group_id fn
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

                return sort_concs(df.loc[:, (name1, name2)].dropna(axis='index'))

        return tuple([onepair_df(name1, name2, df) for df in dfs])

    odf = orns.orns(add_sfr=False, columns='glomerulus')
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


# TODO why does this decorator not seem required anymore? mpl/sns version thing?
#
# decorator to fix "There are no gridspecs with layoutgrids" warning that would
# otherwise happen in any following savefig calls
#@no_constrained_layout
# TODO find where sns.ClusterGrid is actually defined and use that as return type?
# shouldn't need any more generality (Grid was used above to include FacetGrid)
def cluster_rois(df: pd.DataFrame, title=None, odor_sort: bool = True, **kwargs
    ) -> sns.axisgrid.Grid:

    if odor_sort:
        # After transpose: columns=odors
        df = olf.sort_odors(df.T)

    if 'odor1' in df.columns.names:
        # TODO why? comment explaining? just use an appropriate index-value-dict ->
        # formatted str fn? isn't that what i do w/ hong2p.viz.matshow calls?
        df.columns = df.columns.get_level_values('odor1')

    cg = viz.clustermap(df, col_cluster=False, **kwargs)
    ax = cg.ax_heatmap
    ax.set_yticks([])
    ax.set_xlabel('odor')

    if title is not None:
        ax.set_title(title)
        #cg.fig.suptitle(title)

    return cg


# TODO refactor (duped in pn_convergence_vs_hallem_corr.py)
# (and may want to use elsewhere in here for picking odors to use for constructing dF/F
# -> spike rate fn)
# TODO TODO hardcode abbrevs for all hallem odors (or at least all we've ever used?)
# in olf.odor2abbrev, so they can all be found here (input names could be diff from what
# we routinely use) (in the meantime, check all odors i have decent data for are NOT in
# hallem)
def abbrev_hallem_odor_index(df: pd.DataFrame, axis='index') -> pd.DataFrame:
    """Abbreviates Hallem odor names in single-level row index.
    """
    # TODO assert some got replaced (as a check axis was correct)

    # - a-terp (a-terpineol)
    # - 3mtp (3-methylthio-1-propanol)
    # - carene ((1S)-(+)-3-carene)
    #
    # - (don't currently have data, but may soon) o-cre (2-methylphenol)
    df = df.rename({
            # TODO just move all these into olf.odor2abbrev
            'b-citronellol': 'B-citronellol',
            'isopentyl acetate': 'isoamyl acetate',
            'E2-hexenal': 'trans-2-hexenal',
            'a-terpineol': 'alpha-terpineol',
            '3-methylthio-1-propanol': '3-methylthiopropanol',
            '(1S)-(+)-3-carene': '(1S)-(+)-carene',
            '2-methylphenol': 'o-cresol',
        },
        axis=axis
    )
    df = df.rename(olf.odor2abbrev, axis=axis)
    return df


def setnull_old_wrong_solvent_aa_and_va_data(trial_df: pd.DataFrame) -> pd.DataFrame:
    # I don't think any old 2-component air-mixture data (where odor2 is defined) is
    # affected, and I'm unlikely to reanalyze that data at this point.
    odor1 = trial_df.index.get_level_values('odor1')
    wrong_solvent_odors = odor1.str.startswith('va @') | odor1.isin(('aa @ -3',))

    # TODO TODO TODO do i really just have 17 glomeruli per fly (on average) in the 6
    # flies >= 2023-04-26? why do i have ~39 glomeruli per fly in the 2 flies on 4-22?
    last_wrong_solvent_date = pd.Timestamp('2023-04-22')
    assert trial_df.columns.names[0] == 'date', 'date slicing below needs this'

    # Works w/o pd.Timestamp (using str for date) if date is in range of data, but
    # throws an error like `TypeError: Level type mismatch: 2023-04-22` if not in
    # input range (w/ pandas 1.3.1, at least)
    to_null = trial_df.loc[wrong_solvent_odors, :last_wrong_solvent_date]
    n_new_null = num_notnull(to_null)
    if n_new_null > 0:
        msg = 'nulling VA/AA data for flies (from before switch to water solvent):\n'
        msg += '\n'.join([f'- {format_datefly(*x)}'
            for x in to_null.columns.droplevel('roi').unique()
        ])
        warn(msg)

    n_before = num_notnull(trial_df)

    trial_df = trial_df.copy()
    trial_df.loc[wrong_solvent_odors, :last_wrong_solvent_date] = float('nan')

    # TODO should i just dropna trial_df here to get rid of these?
    assert num_notnull(trial_df) == n_before - n_new_null

    return trial_df


# TODO rename all remy->megamat
def plot_remy_drosolf_corr(df, for_filename, for_title, plot_root,
    plot_responses=False):

    df = abbrev_hallem_odor_index(df)

    remy_df = df.loc[panel2name_order['megamat']]

    # TODO TODO it's really -2 though right? say that? (though maybe w/ diff style
    # olfactometer it is comparable?)
    #remy_df.index = remy_df.index.str.cat([' @ -3'] * len(remy_df))
    remy_df.index = remy_df.index.str.cat([' @ -2'] * len(remy_df))

    if plot_responses:
        resp_df = sort_odors(remy_df, add_panel='megamat')

        # b/c plot_all_roi_mean_responses is picky about index level names...
        resp_df = resp_df.droplevel('panel')
        resp_df.index.name = 'odor1'
        resp_df.columns.name = 'roi'

        fig, _ = plot_all_roi_mean_responses(resp_df, odor_sort=False, dpi=1000)

        savefig(fig, plot_root, f'remy_{for_filename}')

    remy_df.index = pd.MultiIndex.from_arrays(
        [[x for x in remy_df.index], ['solvent'] * len(remy_df)],
        names=['odor1', 'odor2']
    )

    remy_corr = remy_df.T.corr()
    remy_corr_da = odor_corr_frame_to_dataarray(remy_corr)

    # TODO share kwargs w/ other places i'm defining them for plotting correlations
    # for remy's experiment
    # TODO replace all kwargs here w/ remy_matshow_kwargs?
    fig = natmix.plot_corr(remy_corr_da, cmap=diverging_cmap, vmin=-1, vmax=1,
        fontsize=10.0
    )
    savefig(fig, plot_root, f'remy_{for_filename}_corr')

    cg = cluster_rois(remy_df, f'{for_title} (megamat odors only)')
    savefig(cg, plot_root, f'remy_{for_filename}_clust')


# TODO refactor to share drop_multiglomerular_receptors default w/ fit_mb_model
# TODO TODO check there aren't some cases where drop_maximal_extent_rois=True (in ROI ->
# mask generation code) is leaving '+' in some ROI names (maybe if they aren't at end in
# input, e.g. if there's something like 'DM5+?' (and handle / change that code if so)
def handle_multiglomerular_receptors(df: pd.DataFrame,
    drop_multiglomerular_receptors: bool = True) -> pd.DataFrame:

    if not drop_multiglomerular_receptors:
        raise NotImplementedError

    assert df.index.name == 'glomerulus'
    # glomeruli should only contain '+' delimiter (e.g. 'DM3+DM5') if the
    # measurement was originally from a receptor that is expressed in each
    # of the glomeruli in the string.
    multiglomerular_receptors = df.index.str.contains('+', regex=False)
    if multiglomerular_receptors.any():
        # TODO warn? at least is some verbose flag is True?
        df = df.loc[~multiglomerular_receptors, :].copy()

    return df


matt_data_dir = Path('../matt/matt-modeling/data')

def hemibrain_wPNKC(_use_matt_wPNKC=False) -> pd.DataFrame:
    # TODO doc
    """
    """

    # TODO refactor to share w/ calling code (also defined there...)?
    glomerulus2receptors = orns.task_glomerulus2receptors()

    # TODO TODO TODO which was that other CSV (that maybe derived these?) that was
    # full PN->KC connectome matrix?
    #
    # NOTE: gkc-halfmat[-wide].csv have 22 columns for glomeruli (/receptors).
    # This should be the number excluding 2a and 33b.
    gkc_wide = pd.read_csv(matt_data_dir / 'hemibrain/halfmat/gkc-halfmat-wide.csv')

    # TODO TODO TODO see if above include more than hallem glomeruli (and find
    # scripts that generated these -> figure out how to regen w/ more than hallem
    # glomeruli)
    # TODO TODO TODO process gkc_wide to have consistent glomerulus/receptor labels
    # where possible (consistent w/ what i hope to also return in random
    # connectivity cases, etc) (presumbly if it's already a subset, should be
    # possible for all of that subset?)
    #import ipdb; ipdb.set_trace()

    # All other columns are glomerulus names.
    assert gkc_wide.columns[0] == 'bodyid'

    wPNKC = gkc_wide.set_index('bodyid', verify_integrity=True)
    wPNKC.columns.name = 'glomerulus'

    # TODO TODO TODO why are these max values in the 1-4 range? synapse counts (/
    # weights) in what prat gave me are almost all >10 (he sets cutoff of >= 5 to even
    # count connection...). does matt's code actually do anything with any of these >1?
    # wPNKC.max()

    # TODO TODO TODO where are values >1 coming from in here:
    # ipdb> mdf.w.value_counts()
    # 1    9218
    # 2     359
    # 3      15
    # 4       1
    mdf = pd.read_csv(matt_data_dir / 'hemibrain/glom-kc-cxns.csv')
    mdf.glom = mdf.glom.replace({'VC3l': 'VC3', 'VC3m': 'VC3'})

    # NOTE: if we do this, mdf_wide.max() is only >1 for VC3 (and it's 2 there, from
    # merging VC3l and VC3m)
    #mdf.loc[mdf.w > 1, 'w'] = 1

    # TODO are all >1 weights below coming from 'w' values that are already >1 before
    # this sum? set all to 1, recompute, and see?
    # (seems so?)
    # TODO replace groupby->pivot w/ pivot_table (aggfunc='count'/'sum')?
    # seemed possible w/ pratyush input (but 'weight' input there was max 1...)
    mcounts = mdf.groupby(['glom', 'bodyid']).sum('w').reset_index()

    mdf_wide = mcounts.pivot(columns='glom', index='bodyid', values='w').fillna(0
        ).astype(int)

    # TODO try to remove need for orns.orns + handle_multiglomerular_receptors in here
    # TODO refactor to share w/ code calling hemibrain_wPNKC? or get from a module level
    # const in drosolf (maybe add one)?
    hallem_orn_deltas = orns.orns(add_sfr=False, drop_sfr=False, columns='glomerulus').T
    hallem_glomeruli = handle_multiglomerular_receptors(hallem_orn_deltas,
        drop_multiglomerular_receptors=True
    ).index
    del hallem_orn_deltas

    mdf_wide = mdf_wide[[x for x in hallem_glomeruli if x != 'DA4m']].copy()

    mdf_wide = mdf_wide[mdf_wide.sum(axis='columns') > 0].copy()

    # TODO move creation of mdf_wide + checking against wPNKC to model_test.py / similar
    assert wPNKC.columns.equals(mdf_wide.columns)
    assert set(mdf_wide.index) == set(wPNKC.index)
    mdf_wide = mdf_wide.loc[wPNKC.index].copy()
    assert mdf_wide.equals(wPNKC)
    del mdf_wide

    # from matt-hemibrain/docs/data-loading.html
    # pn_gloms <- read_csv("data/misc/pn-major-gloms.csv")
    # pn_kc_cxns <- read_csv("data/cxns/pn-kc-cxns.csv")
    # glom_kc_cxns <- pn_kc_cxns %>%
    #   filter(weight >= 3) %>%
    #   inner_join(pn_gloms, by=c("bodyid_pre" = "bodyid")) %>%
    #   group_by(major_glom, bodyid_post) %>%
    #   summarize(w = n(), .groups = "drop") %>%
    #   rename(bodyid = bodyid_post, glom = major_glom)
    # write_csv(glom_kc_cxns, "data/cxns/glom-kc-cxns.csv")

    # inspecting some of the files from above:
    # tom@atlas:~/src/matt/matt-hemibrain/data/misc$ head pn-major-gloms.csv
    # bodyid,major_glom
    # 294792184,DC1
    # 480927537,DC1
    # 541632990,DC1
    # 542311358,DC2
    # 542634818,DM1
    # ...
    # tom@atlas:~/src/matt/matt-hemibrain/data$ head cxns/pn-kc-cxns.csv
    # bodyid_pre,bodyid_post,weight,weight_hp
    # 542634818,487489028,17,9
    # 542634818,548885313,1,0
    # 542634818,549222167,1,1
    # 542634818,5813021736,6,4
    # ...
    # NOTE: pn-kc-cxns.csv above should also be what matt uses to generate distribution
    # of # claws per KC (in matt-hemibrain/docs/mb-claws.html)

    # TODO TODO TODO does distribution of synapse counts in wPNKC (from matt's gkc-...)
    # match what we expect? does matt already make this somewhere?
    # TODO TODO TODO what about double draw (KC drawing from same glomerulus)
    # frequencies? those match the literature? matt make this somewhere?

    # (looks like it was c.weight > 3 actually)
    #
    # This should be from Pratyush, generated on v1.2.1, via something like:
    # MATCH (a:Neuron)-[c.ConnectsTo]->(b:Neuron)
    # WHERE a.Instance CONTAINS "PN"
    # AND b.Instance CONTAINS "KC"
    # AND c.weight > 5
    # RETURN a.bodyId, a.Instance, a.type, b.bodyId, b.Instance, b.type, c.weight
    df = pd.read_excel('data/PNtoKC_connections_raw.xlsx')

    # a.type should all be roughly of form: <glomerulus-str>_<PN-group>, where
    # PN-group are distributed as follows:
    # adPN       6927
    # lPN        2367
    # ilPN        296
    # lvPN        262
    # l2PN1       250
    # l2PN        137
    # adPNm4       85
    # ivPN         76
    # il2PN        49
    # lPNm11D      42
    # vPN          23
    # lvPN2        10
    # l2PNm16       7
    # adPNm5        5
    # lPNm13        2
    # adPNm7        1
    # lvPN1         1
    assert (df['a.type'].str.count('_') == 1).all()

    glom_strs = df['a.type'].str.split('_').apply(lambda x: x[0])
    # glom_strs.value_counts()
    # DP1m         481
    # DM1          377
    # DC1          342
    # DL1          327
    # DM2          312
    # VM5d         310
    # DP1l         309
    # VC3l         301
    # DM6          297
    # DA1          290
    # DM4          283
    # VA2          280
    # VL2p         270
    # VC3m         255
    # VP1d+VP4     250
    # VA4          237
    # VA7m         231
    # DC2          231
    # VA3          228
    # DC3          226
    # VA6          217
    # VC4          207
    # DM3          207
    # DM5          205
    # VL2a         203
    # DL2d         197
    # V            195
    # D            179
    # VA5          168
    # VC2          159
    # VM3          154
    # VM2          150
    # DA2          147
    # VC5          142
    # M            142
    # VP1m         137
    # DL2v         137
    # DL5          136
    # VM5v         133
    # VA1d         131
    # VA7l         129
    # VC1          125
    # VM7d         109
    # VM7v          99
    # VM4           98
    # VA1v          93
    # DA4l          83
    # VM1           78
    # VP3+VP1l      76
    # VP2           71
    # VP1m+VP5      64
    # DC4           55
    # VP1d          49
    # DL4           38
    # DL3           38
    # VL1           37
    # DA3           35
    # DA4m          29
    # VP3+          17
    # VP5+Z         17
    # VP1m+VP2      11
    # VP4            6

    df['glomerulus'] = glom_strs

    hemibrain_gloms = set(df.glomerulus.unique())

    task_gloms = set(glomerulus2receptors.keys())
    # TODO matt only had 67 VM2 connections, but it seems even in 1.1 and 1.2 had 150
    # VM2 connections (and 1.0 right?) (nor did earlier version seem to have diff VA1v
    # connections, right?)
    # TODO TODO TODO investigate wPNKC differences further! (VM2 & VA1v diffs first)

    # TODO TODO TODO try to recreate matt's wPNKC matrix by subsetting prat's stuff to
    # hallem glomeruli and body IDs of KCs that have some connections from them

    # TODO TODO what to do about these (more the latter probably)?
    # > task_gloms - hemibrain_gloms
    # {'VC3', 'VP3', 'VM6v', 'VM6m', 'VM6l', 'VP5', 'VP1l'}
    # > hemibrain_gloms - task_gloms
    # {'VC3m', 'VP3+', 'VP5+Z', 'VC3l', 'VP1m+VP2', 'M', 'VP3+VP1l', 'VP1m+VP5', 'VP1d+VP4'}

    # TODO maybe combine these after pivoting? will probably need pivot_table instead of
    # pivot otherwise?
    df.glomerulus = df.glomerulus.replace({'VC3l': 'VC3', 'VC3m': 'VC3'})

    # TODO TODO TODO fix how w/ aggfunc='count', max is 1 (inconsistent wrt matt's), but
    # w/ aggfunc='sum' currently uses weights that i don't want to use
    # TODO TODO TODO want to count each 'a.bodyId' (PN) separately, right?
    # (seems this is doing that?)
    wPNKC2 = pd.pivot_table(df, values='c.weight', index='b.bodyId',
        columns='glomerulus', aggfunc='count').fillna(0).astype(int)

    wPNKC2.index.name = 'bodyid'

    # TODO make distribution of this, like in matt-hemibrain/docs/mb-claws.html
    # (seems to match up pretty well by inspecting .value_counts())
    #wPNKC2.sum(axis='columns')

    # TODO TODO TODO does prat's excel sheet somehow already exclude multiglomerular
    # PNs? these things are both (length) 135, if it means anything:
    # df['a.bodyId'].nunique()
    # df[['a.bodyId', 'glomerulus']].drop_duplicates()

    # TODO delete
    w2 = wPNKC2[wPNKC.columns].copy()
    # ipdb> (w2.sum(axis='columns') > 0).sum()
    # 1652
    w2 = w2[(w2.sum(axis='columns') > 0)].copy()

    # ipdb> w2.index.isin(wPNKC.index).all()
    # False
    # ipdb> wPNKC.index.isin(w2.index).all()
    # False
    #
    # ipdb> len(set(wPNKC.index) - set(w2.index))
    # 43
    # ipdb> len(set(w2.index) - set(wPNKC.index))
    # 65
    #
    # ipdb> w2.shape
    # (1652, 22)
    # ipdb> wPNKC.shape
    # (1630, 22)

    # TODO maybe compare values between shared bodyids across w2 and wPNKC?

    # TODO TODO TODO seems like wPNKC tends to have some connections w2 doesn't...
    # what's up with that?
    #
    # glomerulus
    # DL5     0
    # VM3     0
    # DL1     0
    # DC1     0
    # DM2     0
    # DA3     0
    # VC3     0
    # DA4l    0
    # VM2     1
    # DM3     0
    # VA1v    0
    # VA5     0
    # DM4     0
    # DL3     0
    # DM6     0
    # VC4     0
    # VA6     0
    # DM5     0
    # VM5d    0
    # DL4     0
    # VA1d    0
    # VM5v    0
    # dtype: int64
    # ipdb> (w2[w2.index.isin(wPNKC.index)] - wPNKC[wPNKC.index.isin(w2.index)]).max().max()
    # 1
    # ipdb> (w2[w2.index.isin(wPNKC.index)] - wPNKC[wPNKC.index.isin(w2.index)]).min().min()
    # -2
    # ipdb> (w2[w2.index.isin(wPNKC.index)] - wPNKC[wPNKC.index.isin(w2.index)]).min()
    # glomerulus
    # DL5    -1
    # VM3    -1
    # DL1    -1
    # DC1    -1
    # DM2    -1
    # DA3    -1
    # VC3    -2
    # DA4l   -1
    # VM2    -1
    # DM3    -1
    # VA1v   -1
    # VA5    -1
    # DM4    -1
    # DL3     0
    # DM6    -2
    # VC4    -1
    # VA6    -1
    # DM5    -1
    # VM5d   -1
    # DL4    -1
    # VA1d   -1
    # VM5v   -1
    #
    # looks like ~70% of KCs have same connections:
    # ipdb> (w2[w2.index.isin(wPNKC.index)].sort_index() == wPNKC[wPNKC.index.isin(w2.index)].sort_index())
    # .all(axis='columns').sum()
    # 1143
    # ipdb> w2.shape
    # (1652, 22)
    # ipdb> 1143/1652
    # 0.6918886198547215
    #
    # ipdb> w2[w2.index.isin(wPNKC.index)].sort_index()[(w2[w2.index.isin(wPNKC.index)].sort_index() != wPN
    # KC[wPNKC.index.isin(w2.index)].sort_index())].sum()
    # glomerulus
    # DL5      0.0
    # VM3      0.0
    # DL1      0.0
    # DC1      3.0
    # DM2      0.0
    # DA3      0.0
    # VC3     23.0
    # DA4l     0.0
    # VM2     82.0
    # DM3      0.0
    # VA1v    16.0
    # VA5      0.0
    # DM4      0.0
    # DL3      0.0
    # DM6      1.0
    # VC4      0.0
    # VA6      0.0
    # DM5      2.0
    # VM5d     3.0
    # DL4      0.0
    # VA1d     0.0
    # VM5v     0.0

    # TODO are they close enough at this point?
    # maybe the differences don't matter?

    #

    # TODO print columns being discarded
    wPNKC2 = wPNKC2[task_gloms & set(wPNKC2.columns)].copy()

    # TODO sort wPNKC2 so that all hallem stuff is first?

    # TODO TODO delete? not sure this will work. might also want conditional on
    # hallem_input?
    if not _use_matt_wPNKC:
        #import ipdb; ipdb.set_trace()
        wPNKC = wPNKC2
        print('USING PRAT WPNKC (W/ SOME INCONSISTENCIES WRT MATT VERSION)')
    #

    return wPNKC


# TODO delete Optional in RHS of return Tuple after implementing in other cases
# TODO if orn_deltas is passed, should we assume we should tune on hallem? or assume we
# should tune on that input?
# TODO rename drop_receptors_not_in_hallem -> glomeruli
# TODO some kind of enum instead of str for pn2kc_connections?
# TODO accept sparsities argument (or scalar avg probably?), for target when tuning
# TODO delete _use_matt_wPNKC after resolving differences wrt Prat's? maybe won't be
# possible though, and still want to be able to reproduce matt's stuff...
# TODO doc that sim_odors is Optional[Set[str]]
def fit_mb_model(orn_deltas=None, sim_odors=None, *, tune_on_hallem: bool = True,
    pn2kc_connections: str = 'hemibrain', n_claws: Optional[int] = None,
    drop_multiglomerular_receptors: bool = True,
    drop_receptors_not_in_hallem: bool = False, seed: int = 12345,
    target_sparsity: Optional[float] = None,
    _use_matt_wPNKC=False, _add_back_methanoic_acid_mistake=False
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Returns responses, wPNKC
    """
    # TODO maybe make it so sim_odors is ignored if orn_deltas is passed in?
    # or err [/ assert same odors as orn_deltas]? would then need to conditionally pass
    # in calls in here...

    pn2kc_connections_options = {'uniform', 'caron', 'hemidraw', 'hemibrain'}
    if pn2kc_connections not in pn2kc_connections_options:
        raise ValueError(f'{pn2kc_connections=} not in {pn2kc_connections_options}')

    if pn2kc_connections == 'caron':
        raise NotImplementedError

    variable_n_claw_options = {'uniform', 'caron', 'hemidraw'}
    variable_n_claws = False
    if pn2kc_connections not in variable_n_claw_options:
        if n_claws is not None:
            raise ValueError(f'n_claws only supported for {variable_n_claw_options}')
    else:
        # TODO TODO TODO also default to averaging over at least a few seeds in all
        # these cases? how much do things actually tend to vary, seed to seed?
        variable_n_claws = True
        if n_claws is None:
            # NOTE: it seems to default to 6 in olfsysm.cpp
            raise ValueError('n_claws must be passed an int if pn2kc_connections in '
                f'{variable_n_claw_options}'
            )

    hallem_input = False
    if orn_deltas is None:
        hallem_input = True
        # TODO just load orn_deltas here?
    else:
        # TODO delete
        orn_deltas = orn_deltas.copy()
        #

        # TODO switch to requiring 'glomerulus' (and in the one test that passes hallem
        # as input explicitly, process to convert to glomeruli before calling this fn)?
        valid_orn_index_names = ('receptor', 'glomerulus')
        if orn_deltas.index.name not in valid_orn_index_names:
            raise ValueError(f"{orn_deltas.index.name=} not in {valid_orn_index_names}")

        if orn_deltas.index.name == 'receptor':
            # TODO delete? (/ use to explain what is happening in case where
            # verbose=True and we are dropping stuff below)
            receptors = orn_deltas.index.copy()
            #

            glomeruli = [
                orns.find_glomeruli(r, verbose=False) for r in orn_deltas.index
            ]
            assert not any('+' in g for gs in glomeruli for g in gs)
            glomeruli = ['+'.join(gs) for gs in glomeruli]

            orn_deltas.index = glomeruli
            orn_deltas.index.name = 'glomerulus'

            # Should drop any input glomeruli w/ '+' in name (e.g. 'DM3+DM5')
            orn_deltas = handle_multiglomerular_receptors(orn_deltas,
                drop_multiglomerular_receptors=drop_multiglomerular_receptors
            )

        # TODO if orn_deltas.index.name == 'glomerulus', assert all input are in
        # task/connectome glomerulus names
        # TODO same check on hallem glomeruli names too (below)?

    mp = osm.ModelParams()

    # TODO TODO what was matt using this for in narrow-odors-jupyter/modeling.ipynb
    #
    # Betty seemed to think this should always be True?
    # TODO was this actualy always True for matt's other stuff (including what's in
    # preprint? does it matter?)
    # Doesn't seem to affect any of the comparisons to Matt's outputs, whether this is
    # True or not (though I'm not clear on why it wouldn't do something, looking at the
    # code...)
    mp.kc.ignore_ffapl = True

    mp.kc.thr_type = 'uniform'

    if target_sparsity is not None:
        mp.kc.sp_target = target_sparsity

    # TODO assert that this csv is equiv to orn_deltas / orns.orns data?
    hc_data_csv = str(Path('~/src/olfsysm/hc_data.csv').expanduser())
    # TODO TODO add comment explaining what this is doing
    osm.load_hc_data(mp, hc_data_csv)
    del hc_data_csv

    hallem_orn_deltas = orns.orns(add_sfr=False, drop_sfr=False, columns='glomerulus').T
    # Should drop 'DM3+DM5'
    hallem_orn_deltas = handle_multiglomerular_receptors(hallem_orn_deltas,
        drop_multiglomerular_receptors=drop_multiglomerular_receptors
    )

    sfr_col = 'spontaneous firing rate'
    sfr = hallem_orn_deltas[sfr_col]
    assert hallem_orn_deltas.columns[-1] == sfr_col
    hallem_orn_deltas = hallem_orn_deltas.iloc[:, :-1].copy()
    n_hallem_odors = hallem_orn_deltas.shape[1]
    assert n_hallem_odors == 110

    hallem_orn_deltas = abbrev_hallem_odor_index(hallem_orn_deltas, axis='columns')

    if hallem_input and sim_odors is not None:
        sim_odors_names2concs = dict()
        for odor_str in sim_odors:
            name = olf.parse_odor_name(odor_str)
            log10_conc = olf.parse_log10_conc(odor_str)

            # If input has any odor at multiple concentrations, this will fail...
            assert name not in sim_odors_names2concs
            sim_odors_names2concs[name] = log10_conc

        assert len(sim_odors_names2concs) == len(sim_odors)

        # These should have any abbreviations applied, but should currently all be the
        # main data (excluding lower concentration ramps + fruits), and not include
        # concentration (via suffixes like '@ -3')
        hallem_odors = hallem_orn_deltas.columns

        # TODO would need to relax this if i ever add lower conc data to hallem input
        assert all(olf.parse_log10_conc(x) is None for x in hallem_odors)

        # TODO TODO replace w/ hope_hallem_minus2_is_our_minus3 code used elsewhere
        # (refactoring to share), rather than overcomplicating here?
        #
        # TODO TODO warn about any fuzzy conc matching (maybe later, only if
        # hallem_input=True?)
        # (easier if i split this into ~2 steps?)
        hallem_sim_odors = [n for n in hallem_odors
            if n in sim_odors_names2concs and -3 <= sim_odors_names2concs[n] < -1
        ]
        # this may not be all i want to check
        assert len(hallem_sim_odors) == len(sim_odors)

        # Since we are appending ' @ -2' to hallem_orn_deltas.columns below
        hallem_sim_odors = [f'{n} @ -2' for n in hallem_sim_odors]

    # TODO factor to drosolf.orns?
    assert hallem_orn_deltas.columns.name == 'odor'
    hallem_orn_deltas.columns += ' @ -2'

    if hallem_input:
        orn_deltas = hallem_orn_deltas.copy()

        if _add_back_methanoic_acid_mistake:
            warn('intentionally mangling Hallem methanoic acid responses, to recreate '
                'old bug in Ann/Matt modeling analysis! do not use for any new '
                'results!'
            )
            orn_deltas['methanoic acid @ -2'] = [
                -2,-14,31,0,33,-8,-6,-9,8,-1,-20,3,25,2,5,12,-8,-9,14,7,0,4,14
            ]

    # TODO TODO TODO implement means of getting threshold from hallem input + hallem
    # glomeruli only -> somehow applying that threshold [+APL inh?] globally
    # (and running subsequent stuff w/ all glomeruli, including non-hallem ones)
    # (even possible?)

    wPNKC = hemibrain_wPNKC(_use_matt_wPNKC=_use_matt_wPNKC)

    if not hallem_input:
        zero_filling = (~ wPNKC.columns.isin(orn_deltas.index))
        if zero_filling.any():
            msg = 'zero filling spike deltas for glomeruli not in data:'
            # TODO TODO sort by glomerulus (and elsewhere)
            # TODO refactor to share printing w/ other similar code?
            msg += '\n- '.join([''] + [f'{g}' for g in wPNKC.columns[zero_filling]])
            msg += '\n'
            warn(msg)

        # TODO TODO if i add 'uniform' draw path, make sure zero filling is keeping
        # glomeruli that would implicitly be dropped in connectome (/ hemidraw / caron)
        # cases (as we don't need wPNKC info in 'uniform' case, as all glomeruli are
        # sampled equally, without using any explicit connectivity / distribution)
        # (don't warn there either)
        #
        # Any stuff w/ '+' in name (e.g. 'DM3+DM5' in Hallem) should already have been
        # dropped.
        input_glomeruli = set(orn_deltas.index)
        glomeruli_missing_in_wPNKC = input_glomeruli - set(wPNKC.columns)
        if len(glomeruli_missing_in_wPNKC) > 0:
            warn('dropping glomeruli not in wPNKC (while zero-filling): '
                f'{glomeruli_missing_in_wPNKC}'
            )

        if tune_on_hallem:
            hallem_not_in_wPNKC = set(hallem_orn_deltas.index) - set(wPNKC.columns)
            assert len(hallem_not_in_wPNKC) == 0 or hallem_not_in_wPNKC == {'DA4m'}, \
                f'unexpected {hallem_not_in_wPNKC=}'

            if len(hallem_not_in_wPNKC) > 0:
                warn(f'dropping glomeruli not in wPNKC {hallem_not_in_wPNKC} from '
                    'Hallem data to be used for tuning'
                )

            # this will be concatenated with orn_deltas below, and we don't want to add
            # back the glomeruli not in wPNKC
            hallem_orn_deltas = hallem_orn_deltas.loc[
                [c for c in hallem_orn_deltas.index if c in wPNKC.columns]
            ].copy()

        # TODO simplify this. not a pandas call for it? reindex_like seemed to not
        # behave as expected, but maybe it's for something else / i was using it
        # incorrectly
        # TODO just do w/ pd.concat? or did i want shape to match hallem exactly in that
        # case? matter?
        orn_deltas = pd.DataFrame([
                orn_deltas.loc[x].values if x in orn_deltas.index
                # TODO correct? after concat across odors in tune_on_hallem=True case?
                else np.zeros(len(orn_deltas.columns))
                for x in wPNKC.columns
            ], index=wPNKC.columns, columns=orn_deltas.columns
        )

        # TODO need to be int (doesn't seem so)?
        mean_sfr = sfr.mean()
        # TODO TODO also warn about this happening (also try imputing w/ 0?)

        sfr = pd.Series(index=wPNKC.columns,
            data=[(sfr.loc[g] if g in sfr else mean_sfr) for g in wPNKC.columns]
        )
        assert sfr.index.equals(orn_deltas.index)
    #

    input_odors = orn_deltas.columns
    # TODO just pick one...
    #n_input_odors = orn_deltas.shape[1]
    n_input_odors = len(input_odors)

    # TODO maybe set tune_on_hallem=False (early on) if orn_deltas is None?
    if tune_on_hallem and not hallem_input:
        # TODO maybe make a list (largely just so i can access it more than once)?
        # TODO where is default defined for this? not seeing it... behave same as if not
        # passed (e.g. if only have hallem odors)
        mp.kc.tune_from = range(n_hallem_odors)

        # Will need to change this after initial (threshold / inhibition setting) sims.
        # TODO interactions between this and tune_from? must sim_only contain tune_from?
        mp.sim_only = range(n_hallem_odors)

        # TODO worth setting a seed here (as model_mix_responses.py did, but maybe not
        # for good reason)?

        # at this point, if i pass in orn_deltas=orns.orns(add_sfr=False).T, only
        # columns differ (b/c odor renaming)

        # TODO delete
        od = orn_deltas.copy()
        #

        # TODO TODO test on my actual data (just tried hallem duped so far).
        # (inspect here to check for weirdness?)
        # TODO TODO need to align (if mismatching sets of glomeruli)?
        # TODO add metadata to more easily separate?
        # TODO TODO maybe add verify_integrity=True (or at least test that everything
        # works in case where columns are verbatim duplicated across the two, which
        # could probably happen if an odor was at minus 2, or if i add support for other
        # concentrations?)
        orn_deltas = pd.concat([hallem_orn_deltas, orn_deltas], axis='columns')

        # since other checks will compare these two indices later
        assert set(sfr.index) == set(orn_deltas.index)

        orn_deltas = orn_deltas.loc[sfr.index].copy()

        assert sim_odors is None or sim_odors == set(input_odors), 'why'

    if not hallem_input:
        # TODO maybe i should still have an option here to tune on more data than what i
        # ultimately return (perhaps including diagnostic data? though they prob don't
        # have representative KC sparsities either...)

        # TODO TODO TODO try to implement other strategies where we don't need to throw
        # away input glomeruli/receptors
        # (might need to make my own gkc_wide csv equivalent, assuming it only contains
        # the connections involving the hallem glomeruli)
        # (also, could probably not work in the tune_on_hallem case...)

        # TODO delete here (already moved into conditional below)
        #hallem_glomeruli = hallem_orn_deltas.index

        # TODO TODO TODO TODO raise NotImplementedError/similar if tune_on_hallem=True,
        # not hallem_input, and not drop_receptors_not_in_hallem?

        # TODO TODO TODO maybe this needs to be True if tune_on_hallem=True? at least as
        # implemented now?
        # TODO rename to drop_glomeruli_not_in_hallem?
        if drop_receptors_not_in_hallem:
            # NOTE: this should already have had 'DM3+DM5' (Or33b) removed above
            hallem_glomeruli = hallem_orn_deltas.index

            glomerulus2receptors = orns.task_glomerulus2receptors()
            receptors = np.array(
                ['+'.join(glomerulus2receptors[g]) for g in orn_deltas.index]
            )
            # technically this would also throw away 33b, but that is currently getting
            # thrown out above w/ the drop_multiglomerular_receptors path
            receptors_not_in_hallem = ~orn_deltas.index.isin(hallem_glomeruli)
            if receptors_not_in_hallem.sum() > 0:
                msg = 'dropping glomeruli not in Hallem:'
                # TODO sort on glomeruli names (seems it already is. just from hallem
                # order? still may want to sort here to ensure)
                msg += '\n- '.join([''] + [f'{g} ({r})' for g, r in
                    zip(orn_deltas.index[receptors_not_in_hallem],
                        receptors[receptors_not_in_hallem])
                ])
                msg += '\n'
                warn(msg)

            # TODO delete
            #if receptors_not_in_hallem.any():
            #    import ipdb; ipdb.set_trace()
            #
            orn_deltas = orn_deltas[~receptors_not_in_hallem].copy()
            sfr = sfr[~receptors_not_in_hallem].copy()

        # TODO TODO TODO another option to use this input for fitting thresholds (+ APL
        # inhibition), w/o using hallem at all

        # TODO TODO am i not seeing inhibition to the extent that i might expect by
        # comparing the deltas to hallem? is it something i can improve by changing my
        # dF/F -> spike delta estimation process, or is it a limitation of my
        # measurements / the differences between the hallem data and ours

    # TODO move earlier?
    assert orn_deltas.columns.name == 'odor'

    # If using Matt's wPNKC, we may have removed this above:
    if 'DA4m' in hallem_orn_deltas.index:
        assert np.array_equal(hallem_orn_deltas, mp.orn.data.delta)

        if hallem_input:
            # TODO just do this before we would modify sfr (in that one branch above)?
            assert np.array_equal(sfr, mp.orn.data.spont[:, 0])

    # TODO TODO merge da4m/l hallem data (pretty sure they are both in my own wPNKC?)?
    # TODO TODO do same w/ 33b (adding it into 47a and 85a Hallem data, for DM3 and DM5,
    # respectively)?

    # TODO TODO TODO only drop DA4m if it's not in wPNKC (which should only be if
    # _use_matt_wPNKC=False?)?
    #
    # We may have already implicitly dropped this in the zero-filling code
    # (if that code ran, and if wPNKC doesn't have DA4m in its columns)
    have_DA4m = 'DA4m' in sfr.index or 'DA4m' in orn_deltas.index

    # TODO also only do if _use_matt_wPNKC=True (prat's seems to have DA4m...)?
    #if (hallem_input or tune_on_hallem) and have_DA4m:
    # TODO this aligned with what i want?
    if 'DA4m' not in wPNKC.columns and have_DA4m:
        # TODO why was he dropping it tho? was it really just b/c it wasn't in (his
        # version of) hemibrain?
        # DA4m should be the glomerulus associated with receptor Or2a that Matt was
        # dropping.
        # TODO TODO TODO why was i doing this? delete? put behind descriptive flag at
        # least? if i didn't need to keep receptors in line w/ what's already in osm,
        # then why do the skipping above? if i did, then is this not gonna cause a
        # problem? is 2a (DA4m) actually something i wanted to remove? why?
        # (was it just b/c it [for some unclear reason] wasn't in matt's wPNKC?)

        # TODO maybe replace by just having wPNKC all 0 for DA4m in _use_matt_wPNKC
        # case, where i would need to fill in those zeros in wPNKC (which doesn't
        # already have DA4m (Or2a), i believe)? could be slightly less special-casey...?
        sfr = sfr[sfr.index != 'DA4m'].copy()
        orn_deltas = orn_deltas.loc[orn_deltas.index != 'DA4m'].copy()

    # TODO don't do if 'uniform' draw path (/ cxn_distrib, but check that there)?
    assert sfr.index.equals(orn_deltas.index)
    # TODO TODO warn here? this always OK?
    wPNKC = wPNKC[sfr.index].copy()

    # TODO try removing .copy()?
    mp.orn.data.spont = sfr.copy()
    mp.orn.data.delta = orn_deltas.copy()

    # TODO in narrow-odors-jupyter/modeling.ipynb, why does matt set
    # mp.kc.tune_from = np.arange(110, step=2)
    # (only tuning on every other odor from hallem, it seems)

    # TODO need to remove DA4m (2a) from wPNKC first too (already out, it seems)?
    # don't see matt doing it in hemimat-modeling... (i don't think i need to.
    # rv.pn.pn_sims below had receptor-dim length of 22)

    # TODO also take an optional parameter to control this number?
    # (for variable_n_claws cases mainly)
    mp.kc.N = len(wPNKC)

    if variable_n_claws:
        # TODO is seed actually only used in variable_n_claws=True cases?
        # (seems so, and doesn't seem to matter it is set right before KC sims)
        # TODO should seed be Optional?
        mp.kc.seed = seed
        mp.kc.nclaws = n_claws

    if pn2kc_connections == 'hemibrain':
        mp.kc.preset_wPNKC = True

    elif pn2kc_connections == 'hemidraw':
        # TODO check index (glomeruli) is same as sfr/etc (all other things w/ glomeruli
        # that model uses)
        # TODO just set directly into mp.kc.cxn_distrib?
        # (and in other places that set this)
        cxn_distrib = wPNKC.sum()

        # TODO compute this from something?
        n_hallem_glomeruli = 23
        assert mp.kc.cxn_distrib.shape == (1, n_hallem_glomeruli)

        # TODO can we modify olfsysm to break if input shape is wrong? why does it work
        # for mp.orn.data.spont but not this? (shape of mp.orn.data.spont is (n, 1)
        # before, not (1, n) as this is)
        # (maybe it was fixed in commit that added allowdd option, and maybe that's why
        # i hadn't noticed it? or i just hadn't actually tested this path before?)
        #
        # NOTE: this reshaping (from (n_glomeruli,) to (1, n_glomeruli)) was critical
        # for correct output (at least w/ olfsysm.cpp from 0d23530f, before allowdd)
        mp.kc.cxn_distrib = cxn_distrib.to_frame().T
        assert mp.kc.cxn_distrib.shape == (1, len(cxn_distrib))

        # TODO still return wPNKC in this case (maybe below) (is it stored under diff
        # variable when not preset?)
        wPNKC = None

    # NOTE: if i implement this, need to make sure cxn_distrib is getting reshaped as in
    # 'hemidraw' case above. was critical for correct behavior there.
    #elif pn2kc_connections == 'caron':
    #    # TODO could modify this (drop same index for 2a) if i wanted to use caron
    #    # distrib Of shape (1, 23), where 23 is from 24 Hallem ORs minus 33b probably?
    #    cxn_distrib = mp.kc.cxn_distrib[0, :].copy()
    #    assert len(cxn_distrib) == 23

    elif pn2kc_connections == 'uniform':
        mp.kc.uniform_pns = True

        # TODO fix! still return!
        wPNKC = None

    rv = osm.RunVars(mp)
    if pn2kc_connections == 'hemibrain':
        rv.kc.wPNKC = wPNKC

    osm.run_ORN_LN_sims(mp, rv)
    osm.run_PN_sims(mp, rv)

    # This is the only place where build_wPNKC and fit_sparseness are called, and they
    # are only called if the 3rd parameter (regen=) is True.
    osm.run_KC_sims(mp, rv, True)

    # TODO how to see the output of the logging calls?
    # TODO should i save configure it to save to a file / stdout before calling?
    # possible from python?
    '''
    print('dir(rv):')
    pprint(dir(rv))

    print('rv.log:')
    print(rv.log)

    print('dir(rv.log):')
    pprint(dir(rv.log))

    import ipdb; ipdb.set_trace()
    '''
    #

    # TODO is it all zeros after the n_hallem odors?
    # TODO do responses to first n_hallem odors stay same after changing sim_only and
    # re-running below?
    # Of shape (n_kcs, n_odors). odors as columns, as elsewhere.
    responses = rv.kc.responses.copy()
    responses_after_tuning = responses.copy()

    # TODO need to update to work again? (in all cases, too)
    #
    # huh so glycerol really has no cells responding to it haha? guess that's reassuring
    #print('odors with no KCs responding:')
    #print(input_odors[:n_hallem_odors][
    #    responses_after_tuning[:, :n_hallem_odors].sum(axis=0) == 0
    #])

    # TODO delete / comment / verbose flag?
    # TODO is 44% of cells really the percentage of silent KCs matt was getting w/
    # hemibrain (and uniform threshold)?
    # TODO TODO only do if first n_hallem_odors actually are hallem!
    # TODO TODO TODO do similar later, on odors we will actually simulate!
    if tune_on_hallem:
        frac_silent = (responses[:, :n_hallem_odors].sum(axis=1) == 0).sum() / mp.kc.N
        print(f'{frac_silent=} (on hallem), after initial tuning')

    if tune_on_hallem and not hallem_input:
        assert (responses[:, n_hallem_odors:] == 0).all()

        # TODO TODO also assert in here that sim_odors is None or sim_odors ==
        # input_odors? (or move that assertion, which should be somewhere above, outside
        # other conditionals)

        mp.sim_only = range(n_hallem_odors, n_hallem_odors + n_input_odors)

        osm.run_ORN_LN_sims(mp, rv)
        osm.run_PN_sims(mp, rv)
        # Don't want to do either build_wPNKC or fit_sparseness here (after tuning)
        osm.run_KC_sims(mp, rv, False)

        responses = rv.kc.responses

        assert np.array_equal(
            responses_after_tuning[:, :n_hallem_odors], responses[:, :n_hallem_odors]
        )

        # TODO also test where appended stuff has slightly diff number of odors than
        # hallem (maybe missing [one random/first/last] row?)
        responses = responses[:, n_hallem_odors:]

        # TODO assert responses are right shape

        # Seems to be ~10% (and may necessarily be close b/c tuning?) in (some) practice
        # Mainly just to rule out it's all still zero.
        # TODO should probably just be a warning if it's small, and maybe only an error
        # if it's all 0
        assert ((responses > 0).sum() / responses.size) > 0.02

    assert responses.shape[1] == n_input_odors
    responses = pd.DataFrame(responses, columns=input_odors)
    responses.index.name = 'model_kc'

    # TODO TODO do this earlier, and regardless of hallem_input, so we can actually only
    # run simulations for odors we want output for (plus any prior tuning)
    # (though only should need to do fuzzy conc matching in hallem_input case)
    # (also to be able to compare hallem and my data more easily, interactively within
    # this fn)
    # NOTE: currently doing after simulation, because i haven't yet implemented support
    # for tuning running on the full set of (hallem) odors, with subsequent simulation
    # running on a different set of stuff
    if hallem_input and sim_odors is not None:
        assert all(x in responses.columns for x in hallem_sim_odors)
        # TODO delete (replace w/ setting up sim_only s.t. only hallem_sim_odors are
        # simulated)
        responses = responses[hallem_sim_odors].copy()

        # TODO also print fraction of silent KCs here
        # (refactor that printing to an internal fn here)

        # TODO print out threshold(s) / inhibition? possible to summarize each? both
        # scalar? (may want to use these values from one run / tuning to parameterize
        # for more glomeruli / diff runs?)

    # TODO maybe in wPNKC index name clarify which connectome they came from (or
    # something similarly appropriate for each type of random draws)
    # TODO try to shuffle things around so i don't need this second return value
    # (pass in gkc_wide and only use for checking if passed?)
    return responses, wPNKC


_fit_and_plot_saved_plot_prefixes = set()
def fit_and_plot_mb_model(plot_dir, sim_odors=None, **model_kws):

    # TODO also support sim_odors=None (returning all)? or just make positional arg?

    my_data = f'pebbled {dff_latex}'

    if 'orn_deltas' in model_kws:
        responses_to = my_data
    else:
        responses_to = 'hallem'

    # TODO TODO TODO also try tuning on remy's subset of hallem odors?

    # TODO share default w/ fit_mb_model somehow?
    tune_on_hallem = model_kws.get('tune_on_hallem', True)
    if tune_on_hallem:
        tune_from = 'hallem'
    else:
        # TODO TODO TODO also clarify it's the hallem subset here (and in responses_to
        # in this case, for now!)
        tune_from = my_data

    # TODO fix (give actual default? make positional?)
    pn2kc_connections = model_kws['pn2kc_connections']

    # TODO TODO also use param_str for title? maybe just replace in the other direction,
    # to add dff_latex in pebbled case as necessary?). or just use filename only for
    # many parameters?

    # TODO will this title get cut off (about a 1/3rd of last (of 3) lines, yes)?
    # TODO maybe return mp from fit_mb_model, and also include mp.kc.sp_target (target
    # sparsity) in title (or include in title anyway, computing it via other means?)?
    # sp_target default seems to be 0.1 actually, not 0.2 as i thought once.
    # TODO TODO either way, try to include target sparsity (and other parameters) in
    # title somehow
    title = (
        # TODO TODO TODO be clear in the drop_nonhallem=True
        # (drop_receptors_not_in_hallem=True) case about the fact that each of these is
        # a subset?
        f'KC thresh [/APL inh] from: {tune_from}\n'
        f'responses to: {responses_to}\n'
        f'wPNKC: {pn2kc_connections}'
    )

    param_str = ', '.join([''] + [
        f'{k}={v}' for k, v in model_kws.items() if k != 'orn_deltas'
    ])

    # TODO clean up / refactor. hack to make filename not atrocious when these are
    # 'pebbled_\$\\Delta_F_F\$'
    if responses_to.startswith('pebbled'):
        responses_to = 'pebbled'

    if tune_from.startswith('pebbled'):
        tune_from = 'pebbled'
    #

    print(f'fitting model ({responses_to=}{param_str})...',
        flush=True
    )

    responses, wPNKC = fit_mb_model(sim_odors=sim_odors, **model_kws)

    print('done', flush=True)

    # TODO always subset odors to plot (or compute corrs to plot) to just remy's odors
    # (so may need to assert input always has it?). so i can sort in same order and so
    # plots will be comparable. (should currently be handled by sim_odors, which is
    # effectively a constant = remy's odors, as currently used...)

    # TODO fix how sort_odors can only add_panel on rows as-is. transposing just to
    # sidestep that.
    # NOTE: this will fail if ever fit_mb_model ever returns odor names without the
    # '@ <log10_conc>' concentration information at the end (e.g. raw Hallem names).
    # Currently it modifiers Hallem names to append ' @ -2', so this isn't an issue.
    responses = sort_odors(responses.T, add_panel='megamat').T

    # TODO TODO TODO also simply plot responses

    # The panel was really just so sort_odors knew which panel order to use, we didn't
    # actually want to add it.
    responses = responses.droplevel('panel', axis='columns')

    # TODO just fix natmix.plot_corr to also work w/ level named 'odor'?
    # (or maybe odor_corr_frame_to_dataarray?)
    responses.columns.name = 'odor1'

    # TODO delete
    matt_order = True
    #matt_order = False

    # TODO TODO does this mean i didn't have some of the odor names right then?
    #if responses_to != 'pebbled':
    #    matt_order = False

    if matt_order:
        # TODO delete hack (maybe fix by adding name_orer to natmix.plot_corr, and
        # passing this there?
        # TODO TODO TODO also show my ORN/PN correlation matrices w/ this order?
        # is this the current (non-alphabetical / cluster) order that remy uses too?
        name_order = [
            '2h', 'IaA', 'pa', '2-but', 'eb', 'ep', 'aa', 'va', 'B-cit', 'Lin', '6al',
            't2h', '1-8ol', '1-5ol', '1-6ol', 'benz', 'ms'
        ]
        '''
        odor_order = [f'{n} @ -3' for n in name_order]
        try:
            responses = responses[odor_order]
        except KeyError:
            import ipdb; ipdb.set_trace()
        '''
    #

    model_corr_df = responses.corr()
    model_corr = odor_corr_frame_to_dataarray(model_corr_df)

    # TODO TODO TODO also try using odor order from fig 3 in preprint, to actually be
    # able to compare to those plots!
    # 2-hep, iaa, pa, 2-but, eb, ep, aac, vac, b-cit, lin, 6al, e2-h, 1-8ol, 1-5ol,
    # 1-6ol, bnz, msl

    # TODO factor (most of) natmix.plot_corr into a similar fn in hong2p.viz?

    # TODO delete. just for comparing to matt's plots. do both ways permanently?
    if matt_order:
        fig = natmix.plot_corr(model_corr, title=title, name_order=name_order,
            **remy_matshow_kwargs
        )
    else:
        fig = natmix.plot_corr(model_corr, title=title, **remy_matshow_kwargs)
    #

    #fig = natmix.plot_corr(model_corr, title=title, **remy_matshow_kwargs)

    # (this goes thru util.to_filename inside savefig)
    for_filename = f'responses-to_{responses_to}'
    if len(param_str) > 0:
        for_filename += '__'
        for_filename += (
            param_str.strip(', ').replace('_','-').replace(', ','__').replace('=','_')
        )

    if matt_order:
        for_filename = f'matt-order_{for_filename}'

    # TODO TODO tweak plot (constrained layout? subplots_adjust? [title] fontsize?)
    # so odor info isn't cut off at top, and paramater info (in "title", below x-axis)
    # isn't either
    # TODO param text larger than i'd like too, so prob decrease first anyway

    # TODO delete
    #print(f'{for_filename=}')
    #

    # to make sure we are accounting for all parameters we might vary in filename
    assert for_filename not in _fit_and_plot_saved_plot_prefixes
    _fit_and_plot_saved_plot_prefixes.add(for_filename)

    savefig(fig, plot_dir, for_filename)


def model_mb_responses(certain_df, parent_plot_dir):
    # TODO make and use a subdir in plot_dir (for everything in here, including
    # fit_and_plot... calls)

    plot_dir = parent_plot_dir / 'mb_modeling'

    # TODO w/ a verbose flag to say which odors / glomeruli overlapped

    # I think deltas make more sense to fit than absolute rates, as both can go negative
    # and then we could better filter out points from non-responsive (odor, glomerulus)
    # combinations, if we wanted to.
    hallem_delta = orns.orns(columns='glomerulus', add_sfr=False)
    hallem_delta = abbrev_hallem_odor_index(hallem_delta)

    our_odors = {olf.parse_odor_name(x) for x in certain_df.index.unique('odor1')}

    hallem_odors = set(hallem_delta.index)

    # as of odors in experiments in the months before 2023-06-30, checked all these
    # are actually not in hallem.
    #
    # this could be a mix of stuff actually not in Hallem and stuff we dont have an
    # abbreviation mapping from full Hallem name. want to rule out the latter.
    # TODO TODO still always print these?
    unmatched_odors = our_odors - hallem_odors

    # TODO TODO TODO match glomeruli up to hallem names
    # (may need to make some small decisions)
    # (basically, are there any that are currently unmatched that can be salveaged?)

    # TODO TODO TODO also check which our our_odors are in hallem lower conc data
    # TODO TODO may want to first fix drosolf so it gives us that too
    # (or just read a csv here myself?)
    #
    # odors w/ conc series in Hallem '06 (whether or not we have data):
    # - ea
    # - pa
    # - eb
    # - ms
    # - 1-6ol
    # - 1o3ol
    # - E2-hexenal (t2h)
    # - 2,3-b (i use -5 for this? same question as w/ ga below)
    # - 2h
    # - ga (treat -5 as -6? -4? interpolate?)
    #
    # (each has -2, -4, -6, -8)

    our_glomeruli = set(certain_df.columns.unique('roi'))

    assert hallem_delta.columns.name == 'glomerulus'
    hallem_glomeruli = set(hallem_delta.columns)

    # TODO check no naming issues
    # {'DM3+DM5', 'DA4m' (2a), 'VA1d' (88a), 'DA4l' (43a), 'DA3' (23a), 'VA1v' (47b),
    # 'DL3' (65a, 65b, 65c), 'DL4' (49a, 85f)}
    print(f'{(hallem_glomeruli - our_glomeruli)=}')

    # TODO TODO check pdf receptor names matches what i get from drosolf w/o passing
    # columns='glomerulus', then use drosolf receptors to check these
    # TODO TODO check receptors of all these are not in hallem
    # TODO TODO TODO print this out and check again. not clear on why DM3 was ever
    # here...
    # - VA2 (92a)
    # - DP1m (Ir64a)
    # - VA4 (85d)
    # - DC2 (13a)
    # - VA7m (UNK)
    # - DA2 (56a, 33a)
    # - VL2a (Ir84a)
    # - DM1 (42b)
    # - VC1 (33c, 85e)
    # - VC2 (71a)
    # - VA7l (46a)
    # - VL1 (Ir75d)
    # - VM7d (42a)
    # - DC4 (Ir64a)
    # - DL2v (Ir75c)
    # - VL2p (Ir31a)
    # - V (Gr21a, Gr63a)
    # - D (69aA, 69aB)
    # - VM7v ("1") (59c)
    # - VC5 (Ir41a)
    # - DL2d (Ir75b)
    # - DP1l (Ir75a)
    # - VA3 (67b)
    # - DC3 (83c)
    print(f'{(our_glomeruli - hallem_glomeruli)=}')

    glomerulus2receptors = orns.task_glomerulus2receptors()

    hallem_glomeruli = np.array(sorted(hallem_glomeruli))
    hallem_glomeruli_in_task = np.array([
        x in glomerulus2receptors.keys() for x in hallem_glomeruli
    ])
    assert set(hallem_glomeruli[~ hallem_glomeruli_in_task]) == {'DM3+DM5'}

    our_glomeruli = np.array(sorted(our_glomeruli))
    our_glomeruli_in_task = np.array([
        x in glomerulus2receptors.keys() for x in our_glomeruli
    ])
    # True for now, but may not always be?
    assert our_glomeruli_in_task.all()

    # TODO start by dropping all concentrations other than -3, and just comparing to
    # main text data (then include conc series stuff later)?
    #
    # TODO maybe ['panel', 'odor1']? or just drop diagnostic panel 'ms @ -3'?
    # TODO TODO convert (date, fly_num) levels to fly_id (1,2,...)
    fly_mean_df = certain_df.groupby('odor1').mean()
    fly_mean_df.index.name = 'odor'

    fly_mean_df = util.add_group_id(fly_mean_df.T.reset_index(),
        ['date', 'fly_num'], name='fly_id').set_index(['fly_id', 'roi']
        ).drop(columns=['date', 'fly_num']).T

    # TODO replace w/ call just renaming 'roi'->'glomerulus'
    assert 'fly_id' == fly_mean_df.columns.names[0]
    fly_mean_df.columns.names = ['fly_id', 'glomerulus']

    mean_df = fly_mean_df.groupby('glomerulus', axis='columns').mean()

    # TODO factor out?
    def melt_odor_by_glom_responses(df, value_name):
        n_before = num_notnull(df)
        df = df.melt(value_name=value_name, ignore_index=False)
        assert num_notnull(df[value_name]) == n_before
        return df

    # TODO factor into fn alongside current abbrev handling
    #
    # TODO TODO TODO actually check this? reason to think this? why did remy originally
    # choose to do -3 for everything? PID?
    # TODO make adjustments for everything else then?
    # TODO TODO guess-and-check scalar adjustment factor to decrease all hallem spike
    # deltas to make more like our -3? or not matter / scalar not helpful?
    hope_hallem_minus2_is_our_minus3 = True
    if hope_hallem_minus2_is_our_minus3:
        warn('treating all Hallem data as if -2 on their olfactometer is comparable to'
            ' -3 on ours (for estimating dF/F -> spike rate fn)'
        )
        # TODO TODO pass abbreved + conc added hallem to model_mb... fn? (to not
        # recompute there...)
        # TODO maybe it'd be more natural to pass in our data, and round all concs to:
        # -2,-4,-6,-8? might simplify consideration across this case + hallem conc
        # series case?
        hallem_delta.index += ' @ -3'
    else:
        raise NotImplementedError('no alternative at the moment...')

    dff_col = 'delta_f_over_f'

    # TODO TODO delete all mean_df stuff if i get fly_mean_df version working?
    # (or just scale w/in each fly before reducing fly_mean_df -> mean_df)
    # (or make choice to take mean right before plotting (to switch easier?)?
    # plus then it would work post-scaling, which is what i would want)
    #mean_df = melt_odor_by_glom_responses(mean_df, dff_col)
    fly_mean_df = melt_odor_by_glom_responses(fly_mean_df, dff_col)

    hallem_delta = melt_odor_by_glom_responses(hallem_delta, 'delta_spike_rate')

    # TODO TODO TODO should i be scaling before subsetting to hallem only stuff?
    # try it?
    #
    # TODO move this before merging w/ hallem_delta?
    # TODO factor out?
    def scale_one_fly(gdf, method='minmax'):
        assert not gdf.fly_id.isna().any() and gdf.fly_id.nunique() == 1
        col_to_scale = dff_col

        new_dff_col = f'{method}_scaled_{dff_col}'

        if method == 'minmax':
            gdf[new_dff_col] = gdf[col_to_scale].copy()
            gdf[new_dff_col] -= gdf[new_dff_col].min()
            gdf[new_dff_col] /= gdf[new_dff_col].max()
            assert np.isclose(gdf[new_dff_col].min(), 0)
            assert np.isclose(gdf[new_dff_col].max(), 1)

        elif method == 'zscore':
            to_scale = gdf[col_to_scale]
            gdf[new_dff_col] = (to_scale - to_scale.mean()) / to_scale.std()
        else:
            raise NotImplementedError(f'scaling {method=} not supported')

        return gdf

    # TODO and maybe w/ scaling per depth? or just have that in model (if at all)?
    fly_mean_df = fly_mean_df.groupby('fly_id', sort=False).apply(
        lambda x: scale_one_fly(x, method='minmax')
    )
    fly_mean_df = fly_mean_df.groupby('fly_id', sort=False).apply(
        lambda x: scale_one_fly(x, method='zscore')
    )

    # TODO TODO where are these NaN coming from (just the "wrong solvent" (va/aa)
    # stuff?)?
    # TODO maybe plot unique combinations of (fly keys X odor X glomerulus) that are NaN
    # (so it's invariant to averaging later, etc)
    print(f'{fly_mean_df.delta_f_over_f.isna().sum() / len(fly_mean_df)} frac of null '
        'dF/F'
    )
    #

    # TODO better name for df... (or factor to fn so it doesn't matter)
    #
    # doesn't seem to matter that odor is index and glomerulus is column. equiv to:
    # pd.merge(mean_df.reset_index(), hallem_delta.reset_index(),
    #     on=['odor', 'glomerulus']
    # )
    #df = mean_df.merge(hallem_delta, on=['odor', 'glomerulus']).reset_index()
    # TODO better name for this
    fdf = fly_mean_df.merge(hallem_delta, on=['odor', 'glomerulus']).reset_index()

    # TODO TODO filter out low intensity stuff? (more points there + maybe more noise in
    # dF/F)
    # TODO TODO fit from full matrix input rather than just each glomerulus as attempt
    # at ephaptic stuff?
    # TODO TODO TODO also color plots by fly in separate plot [/ style in same plot?]
    # (to get a sense of whether we need scaling w/in each fly)
    # TODO TODO try separate plot for each glomerulus, to make it easier to see if there
    # is a large fraction that could be fit well alone? and to see whether it would make
    # sense to have diff parameters for diff glomeruli

    # TODO for each fly, make a histogram of dF/F values (within each (glomerulus x
    # odor) combo)? to decide whether it's worth sampling to decrease representation of
    # stuff around (0,0) (pre-scaling). probably not worth.

    # TODO also print / save fly_id -> (date, fly_num) legend
    assert not fdf.fly_id.isna().any(), 'nunique does not count NaN'
    fly_palette = dict(zip(
        sorted(fdf.fly_id.unique()),
        sns.color_palette(cc.glasbey, fdf.fly_id.nunique())
    ))

    # TODO delete?
    #assert not fdf.glomerulus.isna().any(), 'nunique does not count NaN'
    ## This one seems to produce more obviously distinct hues that hsl
    #glom_palette = dict(zip(
    #    sorted(fdf.glomerulus.unique()),
    #    sns.color_palette(cc.glasbey, fdf.glomerulus.nunique())
    #))

    # TODO get glomerulus depth metadata somewhere earlier (or global per glom? mean
    # plane/depth across flies?) (-> either include in plots / model / both)

    # TODO add receptors in parens after glom, for easy ref to hallem paper

    # TODO try to get an interactive version w/ showing odor on point hover?
    #fig, ax = plt.subplots()
    #sns.scatterplot(df, x='delta_spike_rate', y=dff_col, hue='glomerulus',
    #    legend='full', palette=glom_palette, ax=ax
    #)
    ## TODO TODO also reflect any scaling in ylabel
    #ax.set_ylabel(f'mean glomerulus {dff_latex}\none point per (glomerulus X odor)')
    # TODO savefig

    # TODO TODO how to scale in such a way that we keep negative and positive values
    # (leave ~0 as such), but can also do so comprably across flies. just handle
    # positive and negative separately, or any one fn that could do it?
    # (negative dF/F doesn't actually seem common now, *particularly* when restricted to
    # hallem glomeruli)

    dff_col2desc = {
        dff_col: f'mean glomerulus {dff_latex}',

        f'minmax_scaled_{dff_col}':
            f'mean glomerulus {dff_latex}\n[0,1] scaled within fly',

        f'zscore_scaled_{dff_col}':
            f'mean glomerulus {dff_latex}\nZ-scored within fly',
    }

    _found = False
    # None of these look too different from each other honestly...
    for curr_dff_col, col_desc in dff_col2desc.items():
        fig, ax = plt.subplots()
        common_kws = dict(data=fdf, y='delta_spike_rate', x=curr_dff_col, ax=ax)

        # TODO maybe compare this to an lmplot call w/o the hue variable?
        sns.scatterplot(hue='fly_id', palette=fly_palette, legend='full', **common_kws)

        # TODO TODO TODO make my own regplot-like fn in hong2p.viz, but also:
        # 1) returning model
        # 2) w/ options to plot equation / r**2 / etc
        #
        # TODO TODO maybe just replace w / statsmodels / scipy / own. want the fit at
        # some point anyway (though maybe just for z-scored, or best looking of these?)
        #
        # Letting above plot call do the point plotting. Just using this to plot
        # fit + CI.
        sns.regplot(scatter=False, **common_kws)

        ax.set_xlabel(f'{col_desc}\none point per (fly X glomerulus X odor)')

        # TODO TODO restore close=True after debugging plots below (resolved?)
        close = True
        if curr_dff_col == f'zscore_scaled_{dff_col}':
            _found = True
            close = False

        savefig(fig, plot_dir, f'hallem_vs_{curr_dff_col}', close=close)

        # not what i want, because it also plots one line per hue
        #sns.lmplot(fdf, y='delta_spike_rate', x=curr_dff_col, hue='fly_id',
        #    palette=fly_palette #, legend='full', ax=ax
        #)

    assert _found, 'delete me after debugging linear est plot below'

    # TODO TODO TODO factor statsmodels fiting (+ plotting as matches seaborn)
    # into hong2p.viz (-> share w/ use in
    # natural_odors/scripts/kristina/lit_total_conc_est.py)

    # TODO factor all model fitting + plotting (w/ CIs) into some hong2p fns.
    # TODO specify subset as just the dF/F col which we expect might have NaN here
    # (though should already be behaving that way)
    col_to_fit = 'zscore_scaled_delta_f_over_f'

    # TODO where are the NaN in fdf[dff_col] coming from? is the merge not 'inner'?
    to_fit = fdf.dropna()
    y_train = to_fit.delta_spike_rate
    X_train = sm.add_constant(to_fit[col_to_fit])
    # TODO TODO why does this model produce a different result from the seaborn call
    # above (can tell by sooming in on upper right region of plot)???
    model = sm.OLS(y_train, X_train).fit()

    # TODO still show if verbose=True or something?
    #print('dF/F -> spike delta model summary:')
    #print(model.summary())

    # TODO delete / factor all into a larger fn. just for testing fill_between params
    # (above comment in right place?)
    # TODO rename to clarify type of data input/output
    def predict(df):
        # TODO TODO TODO add_constant to input if needed

        # TODO should i? maybe i should return a new df w/ column added?
        df = df.copy()

        # TODO do this on a particular subset=[column]
        df = df.dropna()

        X = sm.add_constant(df[col_to_fit])

        # TODO should model be computed in this plotting fn, w/ maybe a flag to turn off
        # either the recomputation (maybe by passing a model in?)
        # and/or a flag to turn the plotting off?
        color = 'blue'

        # TODO does it also work w/ just the series col_to_fit input, or do i always
        # need the add_constant col first (to add a 'const'=1 column)? (seems you do
        # always need add_const. maybe make a wrapper that handles that?)
        #
        # TODO how to just draw full line of prediction? just start and stop range +
        # linestyle='-'? or np.arange input on some interval domain?
        y_pred = model.get_prediction(X)

        # alpha=0.05 by default
        #pred_df = y_pred.summary_frame(alpha=0.05)
        pred_df = y_pred.summary_frame(alpha=0.025)

        predicted = y_pred.predicted

        # NOTE: .get_prediction(...) seems to return an object where more information is
        # available about the fit (e.g. confidence intervals, etc). .predict(...) will
        # just return simple output of model (same as <PredictionResult>.predicted).
        # (also seems to be same as pred_df['mean'])
        assert np.array_equal(predicted, pred_df['mean'])
        assert np.array_equal(predicted, model.predict(X))

        # TODO TODO how are these CI's actually computed? how does that differ from how
        # seaborn computes them? why are they different?
        # TODO what are obs_ci_[lower|upper]? seems newer versions of statsmodels might
        # not have them anyway? or at least they aren't documented...
        xs = X[col_to_fit]
        # TODO should i maybe just define xs from df[col_to_fit]?
        assert xs.equals(df[col_to_fit])

        # TODO TODO TODO use prediction instead of delta_spike_rate, if input doesn't
        # have that column (and maybe change plot then too. color markers as in
        # scatterplot? maybe w/ dotted line up to actual data point?)

        # TODO delete sort= path + revent to sort=False behavior, if sorting doesn't
        # actually matter here (even if linestyle != '')
        sort = True
        if sort:
            sorted_indices = np.argsort(xs).values
            xs = xs.iloc[sorted_indices]
            pred_df = pred_df.iloc[sorted_indices]

        fig, ax = plt.subplots()

        # TODO TODO comment explaining why only some of input has this (and/or refactor
        # to be more general and not tied to specific hardcoded col names...)
        if 'delta_spike_rate' in df.columns:
            # TODO try to at least share y def w/ above
            sns.scatterplot(data=df, hue='fly_id', palette=fly_palette, legend='full',
                y='delta_spike_rate', x=col_to_fit, ax=ax
            )

            ax.plot(xs, pred_df['mean'], color=color, linestyle='', marker='x',
                alpha=0.1
            )

        else:
            df['est_delta_spike_rate'] = predicted
            sns.scatterplot(data=df, hue='fly_id', palette=fly_palette, legend='full',
                y='est_delta_spike_rate', x=col_to_fit, ax=ax, marker='x'
            )

        # TODO check whether xs and pred_df['mean'] sorting matters
        # (sort flag above)
        ax.plot(xs, pred_df['mean'], color=color)

        ax.fill_between(xs,
            pred_df['mean_ci_lower'],
            pred_df['mean_ci_upper'],
            color=color, alpha=0.2,
            # TODO each of these needed? try to recreate seaborn (set color_palette the
            # same / use that seaborn blue?)
            linestyle='', linewidth=0, edgecolor='white'
        )

        return df, fig

    # TODO TODO add comment explaining different between fdf and fly_mean_df (and
    # probably also rename to better variable names) + explain why only one has
    # 'delta_spike_rate' in columns
    #
    # TODO or don't return fig, and savefig inside (prob would need to add a filename /
    # desc arg then)
    # TODO rename fdf to be clear it contains overlap between hallem and my data
    # (and not odors only in my data)
    fdf, f1 = predict(fdf)
    # TODO are only NaNs in the dff_col in here coming from setting-wrong-odors-NaN
    # above?
    fly_mean_df, f2 = predict(fly_mean_df)

    import ipdb; ipdb.set_trace()
    # TODO TODO savefig (both?) all (just do in predict(...), and force an input to make
    # save names unique?)

    # TODO TODO TODO try a depth specific model too
    # (quite clear overall scale changes when i need to avoid strongest responding plane
    # b/c contamination. e.g. often VA4 has contamination p-cre response in highest
    # (strongest) plane, from one of the nearby/above glomeruli that responds to that)

    # TODO plot histogram of fly_mean_df.est_delta_spike_rate (maybe resting on the x
    # axis in the same kind of scatter plot of (x=dF/F, y=delta spike rate,
    # hue=fly_id)?)

    mean_est_df = fly_mean_df.reset_index().groupby(['odor', 'glomerulus'])[
        'est_delta_spike_rate'].mean()

    # Then odors will be columns and glomeruli will be rows, which is same as
    # orns.orns().T
    mean_est_df = mean_est_df.unstack('odor')

    # TODO more direct list of these odors available earlier? maybe in sort order (where
    # i think they are spelled out)? maybe it makes sense to compute here tho...
    remy_odors = set(certain_df.loc['megamat'].index.get_level_values('odor1').unique())

    remy_odor_cols = [x for x in mean_est_df.columns if x in remy_odors]
    assert len(set(remy_odor_cols)) == len(remy_odor_cols)

    mean_est_df = mean_est_df[remy_odor_cols].copy()


    # TODO TODO TODO possible to get spike rates for any of the other door data sources?
    # available in their R package?

    # TODO TODO TODO how to do an ephaptic model? possible to optimize one using my
    # data as input (where we never have the channels separated)? if using hallem, for
    # what fraction of sensilla do we have all / most contained ORN types?
    # TODO which data to use for ephaptic effects / how?
    # TODO plot ephaptic model adjusted dF/F (subtracting from other ORNs in sensilla)
    # vs spike rate?


    # TODO TODO plot mean_est_df vs same subset of hallem, just for sanity checking
    # (as a matrix in each case)
    # TODO TODO TODO and do w/ my ORN input, also my ORN input subset to hallem stuff
    # (computing correlation in each case, with nothing going thru MB model first)

    my_data = f'pebbled {dff_latex}'

    # TODO TODO drop all non-megamat odors prior to running through fit_mb_model?
    # (diagnostics prob gonna produce much lower KC sparsity)S
    # (are any of the non-HALLEM odors (which is probably most non-megamat odors?)
    # actually influencing model in fit_mb_model tho?)

    # TODO TODO TODO actually fit to average sparsities remy observes
    # (actually, might make more senes to just sweep sparsity a bit, like B suggested)

    # TODO TODO TODO TODO try fitting on hallem, and then running on my data passed thru
    # dF/F model (fitting on all hallem might produce very different thresholds from
    # fitting on the subset of odors remy uses!!!)

    # TODO TODO TODO try sweeping target sparsities (maybe over range [5, 20]%)
    # (maybe a separate output folder for each?)
    for model_kws in [
            # TODO TODO prob just always explicitly list out kwargs (independent of
            # defaults) (also in plot filenames?)

            # TODO TODO TODO also explore combinations w/ _use_matt_wPNKC=True?
            # maybe see how choices for this affect tuning on + responses to hallem
            # first?

            # TODO check this one makes less sense than tuning on hallem
            # (b/c presumably these are largely strongly activating odors?)
            # TODO TODO at least, until using remy's sparsities (ideally... not sure
            # it'll work out that way...)
            dict(
                orn_deltas=mean_est_df, tune_on_hallem=False,
                pn2kc_connections='hemibrain'
            ),

            # TODO TODO TODO this actually make sense to try (w/ tune_on_hallem=True and
            # drop_receptors_not_in_hallem=False?) does my code even support currently?
            #dict(orn_deltas=mean_est_df, tune_on_hallem=True,
            #    pn2kc_connections='hemibrain'
            #),

            dict(orn_deltas=mean_est_df, tune_on_hallem=True,
                drop_receptors_not_in_hallem=True,
                pn2kc_connections='hemibrain'
            ),

            dict(orn_deltas=mean_est_df, tune_on_hallem=False,
                drop_receptors_not_in_hallem=True,
                pn2kc_connections='hemibrain'
            ),

            # TODO what should i actually use for n_claws?

            dict(orn_deltas=mean_est_df, tune_on_hallem=False,
                pn2kc_connections='uniform', n_claws=7
            ),

            # TODO TODO TODO this actually make sense to try (w/ tune_on_hallem=True and
            # drop_receptors_not_in_hallem=False?) does my code even support currently?
            #dict(orn_deltas=mean_est_df, tune_on_hallem=True,
            #    pn2kc_connections='uniform', n_claws=7
            #),

            # TODO TODO TODO is this output consistent w/ what matt had?
            # add _use_matt_wPNKC and see if that fixes it?
            dict(pn2kc_connections='hemibrain'),

            dict(pn2kc_connections='hemibrain', _use_matt_wPNKC=True),

            dict(pn2kc_connections='uniform', n_claws=7),
        ]:

        for target_sparsity in (0.03, 0.05, 0.1):
            # TODO clean up...
            #
            # 0.1 should be the default target sparsity (mp.kc.sp_target)
            if target_sparsity != 0.1:
                #model_kws = dict(model_kws)
                _model_kws = deepcopy(model_kws)
                _model_kws['target_sparsity'] = target_sparsity
            else:
                _model_kws = deepcopy(model_kws)

            fit_and_plot_mb_model(plot_dir, sim_odors=remy_odors, **_model_kws)

    # (maybe do some/all of this outside of model_mb_responses though...)
    # TODO TODO TODO plot correlation (w/ error [when possible], calculated same as remy
    # does) for (probably use plot_corrs?) (all on megamat odors only):
    # - my full ORN input
    # - my ORN input subset to the hallem glomeruli
    # - model output computed w/ (both? at least computed w/ my full input)
    # - hallem (already have under .../<plot_fmt>/remy_hallem_orns_corr.png)


# TODO also return indicator (to print in all_corr..._plots) (or compute there?)
def most_recent_GH146_output_dir():
    # cwd is where output dirs should be created (see driver_indicator_output_dir)
    #
    # TODO refactor to use something like output_dir2driver (but for indicator), and if
    # it fails exclude those (instead of manually checking # parts here too)
    dirs = [x for x in Path.cwd().glob('GH146_*/') if x.name.count('_') == 1]
    return sorted(dirs, key=util.most_recent_contained_file_mtime)[-1]


# TODO type hint return of optional set[str]
def get_gh146_glomeruli():
    gh146_output_dir = most_recent_GH146_output_dir()

    # TODO factor out? hong2p.util even?
    def _fmt_rel_path(p):
        return f'./{p.relative_to(Path.cwd())}'

    # TODO TODO share w/ saving code
    plot_str = 'corr/gh146_rois_only'

    gh146_output = gh146_output_dir / ij_roi_responses_cache
    # TODO don't hardcode plot path here (also switch '_certain_' substr on certain
    # flag)
    print(f'loading GH146 glomeruli from {_fmt_rel_path(gh146_output)}, to restrict'
        ' current '
        f'ORN glomeruli (for {plot_str} plots)'
    )

    if not gh146_output.exists():
        warn(f'no GH146 data found at {_fmt_rel_path(gh146_output_dir)}. can not '
            'generate correlation plots restricted to GH146 glomeruli!'
        )
        return None

    gh146_df = pd.read_pickle(gh146_output)
    certain_gh146_df = select_certain_rois(gh146_df)

    gh146_glom_counts = certain_gh146_df.groupby(level='roi', axis='columns'
        ).size()
    gh146_glom_counts.index.name = 'glomerulus'
    gh146_glom_counts.name = 'n_flies'

    n_flies = certain_gh146_df.groupby(level=['date', 'fly_num'], axis='columns'
        ).ngroups

    # TODO TODO TODO refactor to share dropping with subsetting in gh146 plot making
    reliable_gh146_gloms = gh146_glom_counts > (n_flies / 2)
    if (~ reliable_gh146_gloms).any():
        # TODO sanity check this set
        warn(f'excluding GH146 glomeruli seen in <1/2 of {n_flies} GH146 '
            f'flies:\n{gh146_glom_counts[~reliable_gh146_gloms].to_string()}'
        )

    # TODO warn instead?
    assert reliable_gh146_gloms.sum() > 0, ('no glomeruli confidently '
        f'identified in >=1/2 of total {n_flies} GH146 flies!'
    )
    reliable_gh146_gloms.name = 'count_as_GH146'

    for_csv = pd.DataFrame([gh146_glom_counts, reliable_gh146_gloms]).T

    # TODO TODO write which GH146 glomeruli we use to a CSV under pebbled corr
    # plot dir (alongside plot), to have a record of which glomeruli i
    # considered to be a part of GH146 (include counts? include all and add a
    # column saying which we included? include n_flies in file/column name or
    # something?)

    # TODO warn if any of the GH146 ones not seen in pebbled data

    gh146_glomeruli = set(reliable_gh146_gloms[reliable_gh146_gloms].index)
    return gh146_glomeruli


# TODO rename to across_fly...?
def all_correlation_plots(output_root: Path, trial_df: pd.DataFrame, *,
    certain_only=True) -> None:

    # TODO TODO TODO save CSVs of [hallem|gh146]-restricted / bouton-frequency-repeated
    # version as well, so remy can compute and plot the corrs? (maybe outside of this
    # fn, around existing CSV saving? might just wanna call this fn w/ diff inputs,
    # in that case, rather than separately restricting/etc here)
    # TODO and/or get code remy uses to compute plot (ideally w/ some example data)? she
    # still have code she used on my CSVs last time (and it was my CSV versions she
    # used?)?

    # TODO TODO any cases where i need to preprocess this / my glomeruli names?
    hallem_glomeruli = set(orns.orns(columns='glomerulus').columns)

    # TODO move this correlation calculation after loop once i'm done debugging it
    # (possible?)
    ij_corr_list = []
    hallem_roi_only_ij_corr_list = []

    # TODO TODO TODO also restrict pebbled glomeruli to those (i can identify in?) GH146

    # TODO try to make GH146 data loading agnostic to driver (pick most recent directory
    driver = output_dir2driver(output_root)
    gh146_roi_only_version = False
    if driver in orn_drivers:
        gh146_roi_only_ij_corr_list = []

        gh146_glomeruli = get_gh146_glomeruli()
        if gh146_glomeruli is not None:
            gh146_roi_only_version = True

    # TODO TODO where to decide on GH146 glomeruli? pretty sure there's not just some i
    # could pull out if analysis was more final / if i had other landmarks?
    # also get 
    # TODO only restrict to GH146 glomeruli in pebbled analysis, but maybe err if GH146
    # analysis has any glomeruli not in set we determine. at least if we determine that
    # set with some other data than just loading my GH146 outputs, which would be
    # circular...

    # TODO also don't fail if pebbled ROI outputs don't exist (but warn)


    # TODO TODO TODO also duplicate pebbled glomeruli by hemibrain bouton frequencies
    # (probably don't care for GH146?) -> save to separate output folder

    fly_panel_id = 0
    for (date, fly_num), fly_df in trial_df.groupby(fly_keys, axis='columns',
        sort=False):

        # TODO TODO maybe modify plot_corrs so it accepts a list of length == # of flies
        # (i.e. doesn't need one entry for each (fly, panel) combination)).
        # only matters for ease + if i want to actually plot correlations between odors
        # from two diff panels
        # (...but if it's complaining about diff size corr inputs, and w/in panel size
        # doesn't change, is plot_corrs current handling even correct?)
        for panel, fly_panel_df in fly_df.groupby('panel', sort=False):

            # There will only be one of these, and we already have access to it.
            # It'll just pollute the correlation matrix indices if we leave it.
            fly_panel_df = fly_panel_df.droplevel('panel')

            if certain_only:
                for_corr = select_certain_rois(fly_panel_df)
            else:
                for_corr = fly_panel_df

            # Since in len(2)==2 case, everything will either be +/- 1, and we don't
            # want a degenerate correlation like that averaged with the others.
            # len(1)==1 produces all NaN. Neither meaningful.
            if len(for_corr.columns) <= 2:
                continue
            # TODO TODO TODO save single fly corr CSVs (+ across fly error CSV) here?
            # (delete similar note in place currently saving the mean CSV
            # [in plot_corrs] + move that here, if so)

            # TODO TODO may need to handle (skip) data w/ no such ROIs? test!
            hallem_for_corr_df = for_corr.loc[:,
                for_corr.columns.get_level_values('roi').isin(hallem_glomeruli)
            ]

            corr_df = for_corr.T.corr()
            hallem_corr_df = hallem_for_corr_df.T.corr()

            # TODO delete
            # TODO can i define hallem_for_corr_df after `corr_df = for_corr.T.corr()`,
            # to save a line? try!
            #hallem_for_corr_df2 = 

            meta = {
                'date': date,
                'fly_num': fly_num,
                'panel': panel,
                'fly_panel_id': fly_panel_id,
            }
            corr = odor_corr_frame_to_dataarray(corr_df, meta)
            # TODO TODO refactor to filter on ROIs at the end??? (and also handle the
            # GH146 ROIs only that way too)
            hallem_rois_only_corr = odor_corr_frame_to_dataarray(hallem_corr_df, meta)

            if gh146_roi_only_version:
                gh146_for_corr_df = for_corr.loc[:,
                    for_corr.columns.get_level_values('roi').isin(gh146_glomeruli)
                ]
                gh146_corr_df = gh146_for_corr_df.T.corr()
                gh146_rois_only_corr = odor_corr_frame_to_dataarray(gh146_corr_df, meta)
                gh146_roi_only_ij_corr_list.append(gh146_rois_only_corr)

            ij_corr_list.append(corr)
            hallem_roi_only_ij_corr_list.append(hallem_rois_only_corr)
            fly_panel_id += 1

    # TODO TODO probably also put some text in figure (title?)
    # (add kwarg to append to titles in plot_corrs)
    if certain_only:
        # TODO remove '_only' suffix on these first 2
        corr_dirname = 'corr_certain_only'
        hallem_corr_dirname = 'corr_certain_hallem_only'
        #
        gh146_corr_dirname = 'corr_certain_gh146'
    else:
        corr_dirname = 'corr_all_rois'
        hallem_corr_dirname = 'corr_all_hallem'
        gh146_corr_dirname = 'corr_all_gh146'

    # TODO TODO make a 'correlations' dir and change all dirs made in here to subdirs of
    # that (minus the 'corr_' prefix)? ijroi/ dir getting a bit crowded...

    # TODO TODO try to figure out if (what seems like) a baselining issue showing up in
    # diff correlations on some odors first trials can be improved (more obvious in
    # hallem only stuff) (or maybe it's a real "first trial effect" to some extent?)
    # TODO TODO TODO update all these plots to:
    # - use whichever error metric remy used (SD?)
    # - trial averaged versions (ideally computing average response, and then computing
    #   correlation, tho that would involve more change here...)
    ijroi_corr_plot_reldir = Path(across_fly_ijroi_dirname) / corr_dirname
    plot_corrs(ij_corr_list, output_root, ijroi_corr_plot_reldir)

    hallem_rois_only_reldir = Path(across_fly_ijroi_dirname) / hallem_corr_dirname
    plot_corrs(hallem_roi_only_ij_corr_list, output_root, hallem_rois_only_reldir)

    if gh146_roi_only_version:
        # TODO time to refactor?
        gh146_rois_only_reldir = Path(across_fly_ijroi_dirname) / gh146_corr_dirname
        plot_corrs(gh146_roi_only_ij_corr_list, output_root, gh146_rois_only_reldir)


# TODO adapt -> share w/ (at least) drop_redone_odors?
def format_panel(x):
    panel = x.get('panel')
    # TODO maybe return None if it'd make more consistent vline_level_fn usage
    # possible (across single/multi panel cases)? would prob need to handle in viz
    # regardless
    if not panel:
        return ''

    if panel == diag_panel_str:
        # Less jarring than 'glomeruli_diagnostics'
        return 'diagnostics'

    assert type(panel) is str
    return panel


# TODO fix how grouped ROI labels currently a bit low (still?) (revert va to default?
# offset?)
# TODO TODO put 'n=<n>' text below ROI labels? esp for stuff w/ less than full
def roi_label(index_dict):
    roi = index_dict['roi']
    if is_ijroi_named(roi):
        return roi
    # Don't want numbered ROIs to be grouped together in plots, as the numbers
    # aren't meanginful across flies.
    return ''


# TODO TODO refactor inches_per_cell (+extra_figsize) to share w/ plot_roi_util?
# just move into plot_all_... default? (would need extra_figsize[1] == 1.0 to work here)
#
# TODO rename to clarify these aren't for plot_rois calls...
roi_plot_kws = dict(
    inches_per_cell=0.08,
    # TODO adjust based on whether we have extra text labels / title / etc?
    # 1.7 better for the single panel plots, 2.0 needed for ijrois_certain.png, etc
    extra_figsize=(2.0, 0.0),

    fontsize=4.5,

    linewidth=0.5,
    dpi=300,

    hgroup_label_offset=5.5,

    vgroup_label_offset=7,

    # TODO define separate ones for colorbar + title/ylabel (+ check colorbar one is
    # working)
    bigtext_fontsize_scaler=1.5,

    # TODO TODO try a single colorscale for all? better/worse than current plots?
    # pick from data or use fixed range (would need to warn/err if data is outside
    # fixed range)?

    cbar_label=trial_stat_cbar_title,

    odor_sort=False,
    # TODO need to prevent hline_level_fn from triggering in case where we already
    # only have one row per roi?
    #  'roi' should always be a level
    hline_level_fn=roi_label,
    vline_level_fn=format_panel,
    # TODO try to delete levels_from_labels (switching to only == False case),
    # (inside viz)
    levels_from_labels=False,
)

# TODO replace deepcopy w/ just dict(...)?
roimean_plot_kws = deepcopy(roi_plot_kws)
roimean_plot_kws['inches_per_cell'] = 0.15
roimean_plot_kws['extra_figsize'] = (1.0, 0.0)
roimean_plot_kws['vgroup_label_offset'] = 5

# TODO can replace title back w/ usual title here, now that we have it in cbar
# TODO replace deepcopy w/ just dict(...)?
n_roi_plot_kws = deepcopy(roimean_plot_kws)
n_roi_plot_kws['cbar_label'] = 'number of flies (n)'

def response_matrix_plots(plot_dir: Path, df: pd.DataFrame,
    fname_prefix: Optional[str] = None) -> None:
    # TODO doc
    """
    """
    if fname_prefix is not None:
        assert not fname_prefix.endswith('_')
        fname_prefix = f'{fname_prefix}_'
    else:
        fname_prefix = ''

    # TODO TODO TODO versions of these plots that only focus on pebbled ROIs that
    # are in GH146 (that i have also been able to find in it / are known to be in it)
    # (so that i can put them side-by-side)
    # TODO interactive plot (/format) where scrolling past view of (top) xlabel text
    # keeps xlabel text in view, but just scrolls through data? to be able to actually
    # read odors for stuff plotted further down (or break into multiple plots, according
    # to a reasonable vertical size?)

    # TODO TODO separate option for handling ROIs that only ever have '?' suffix
    # (e.g. 'DL3?' when i haven't corroborated this ROI from other data)
    # (want to still be able to see these things in some versions of the average plots,
    # to compare new flies to this mean, but don't want to drop the suffix cause the
    # high uncertainty)

    # TODO TODO per-roi-mean plot (both normalized within ROI and not) where ROIs
    # are ordered according to clustering rather than just alphabetically

    # TODO TODO change '+' ROI handling from just suffix to anywhere in (but maybe
    # include debug list of all ROIs matching, to make sure there aren't some that i
    # didn't actually want discarded? (anything containing '+' should be dropped)

    # TODO TODO versions of all-roi-mean plots where if only uncertain ( "<x>?") ROIs
    # exist (within a recording/fly) for a given glomerulus name, then those are also
    # included in average (but still exclude uncertain when they exist alongside certain
    # ROIs of same name)
    # TODO and maybe (also?) break these into their own plot?

    # TODO version of mean plot w/ zeros added for non-responsive glomeruli? (maybe w/
    # some indicator for those we are particularly unsure of / in general? how tho?)

    # TODO version of all-roi-mean plots where any (odor, ROI) combos w/ less than all
    # flies having that odor are shown as missing (white)? (actually want this? maybe
    # some threshold such as at least 2 or half?)

    certain_df = select_certain_rois(df)

    if len(certain_df.columns) == 0:
        # I.e. no "certain" ROIs (those named and without '?' suffix, or other
        # characters marking uncertainty)
        warn(f'{plot_dir.name}: no ImageJ ROIs were confidently identified!')
        # TODO TODO do i want to do any uncertain ROI plots in here tho? maybe?
        return

    makedirs(plot_dir)

    # TODO define savefig wrapper in here to cut down on boilerplate?

    # TODO change date/fly_num part to to name A,B,... w/in an ROI name
    # (or sequential numbers via ~add_group_id), for at least one version of the
    # *certain* plots (+ write a CSV key mapping date+fly_num to these IDs)
    # (implement in plot_all_mean... ? def not here)
    # TODO separate representation showing n for each particular glomerulus name
    # (in yticklabels)?
    # TODO shouldn't this be using certain_df (for certain plots, if i ever do any
    # uncertain plots in here) (actually matter?)?
    # just do away w/ this altogether anyway? can i replace w/ some variant of
    # n_per_odor_and_glom (as in loop above)?
    # TODO may want to replace w/ something derived from n_per_odor_and_glom if i can
    n_flies = len(certain_df.columns.to_frame(index=False)[['date','fly_num']
        ].drop_duplicates())

    # TODO assert plot_dir is where a <driver>_<indicator> dir should be?
    # other checks?
    driver, indicator = plot_dir.parts[0].split('_')

    # TODO maybe (here and elsewhere), map driver to something simple like ('ORN'/'PN'),
    # for use in titles/etc
    # TODO remove the '<=' part eventually (or special case when whole plot has same n)
    # or just show min? range?
    # TODO TODO only show '<=' if some odors have less n than odors (as w/ aa/va in
    # megamat panel) (otherwise, just use '=')
    # TODO or show n range instead?
    title = f'{driver}>{indicator} (n<={n_flies})'

    n_per_odor_and_glom = certain_df.notna().groupby('roi', sort=False,
        axis='columns').sum()
    # TODO at least for panels below, show min N for each glomerulus?
    # (maybe as a separate single-column matshow w/ it's own colorbar?)
    # (only relevant in plots that take mean across flies)

    max_n = n_per_odor_and_glom.max().max()
    # discrete colormap: https://stackoverflow.com/questions/14777066
    cmap = plt.get_cmap('cividis', max_n)
    # want to display 0 as distinct (white rather than dark blue)
    cmap.set_under('white')

    # TODO TODO refactor so i can call this separately in main, where i'm currently
    # making some extra (similar) plots
    #
    # TODO version of this with ROIs sorted by (min? mean?) N?
    # TODO show n in each matrix element?
    fig, _ = plot_all_roi_mean_responses(n_per_odor_and_glom, cmap=cmap,
        title=f'{title}\nsample size (n) per (glomerulus X odor)',
        vmin=0.5, vmax=(max_n + 0.5),
        cbar_kws=dict(ticks=np.arange(1, max_n + 1)), **n_roi_plot_kws
    )

    # unuseable as is. font too big and may not be transposed and/or aligned correctly.
    #
    # TODO was it constrained layout that was causing (most of?) the issues?
    # can i do without it?
    # TODO would probably have to move this into plot_all_roi_mean...
    # (or otherwise ensure plotted order matches order of n_per_odor_and_glom
    # (as averaged / sorted here) for purposes of drawing N on each cell)
    # n_per_odor_and_glom.groupby([x for x in df.index.names if x != 'repeat'],
    #     sort=False).mean()
    # .max(axis='rows') above, and save to csv (for now)?
    '''
    # TODO implement in such a way that we don't just assume the first axes is the
    # non-colorbar one? it probably always will be tho...
    # not sure i could trust plt.gca() any more either...
    assert len(fig.axes) == 2
    ax = fig.axes[0]

    # TODO TODO TODO need to transpose n_per_odor_and_glom?
    #
    # https://stackoverflow.com/questions/20998083
    for (i, j), n in np.ndenumerate(n_per_odor_and_glom):
        # TODO color visible enough? way to put white behind?
        # or just use some color distinguishable from whole colormap?
        ax.text(j, i, n, ha='center', va='center')
    '''

    savefig(fig, plot_dir, f'{fname_prefix}certain_n')

    # TODO TODO normalized **w/in fly** versions too (instead of just per ROI)?

    # TODO bbox_inches='tight' even work? was using constrained layout, but it doesn't
    # seem to work w/ text... (seems to work OK, but haven't checked the extent to which
    # relative sizes of other plot elements have changed from what i had just before
    # this. not sure it matters anyway.)
    # TODO just do bbox_inches='tight' for all?
    # TODO just check # of columns to decide if we want to add bbox_inches='tight'
    # (between diags + megamat and that number + validation2)? or always pass it?
    plot_responses_and_scaled_versions(certain_df, plot_dir, f'{fname_prefix}certain',
        bbox_inches='tight', title=title, **roi_plot_kws
    )

    # TODO TODO TODO should i be normalizing within fly or something before taking mean?
    #
    # TODO TODO factor into option of plot_all_..., so that i can share code to
    # show N for each ROI w/ case calling from before loop over panels
    mean_certain_df = certain_df.groupby('roi', sort=False, axis='columns').mean()

    # TODO check and remove uncertainty from this comment...
    # I think this (plot_all...?) (now wrapped behond plot_responses_and_scaled...) is
    # sorting on output of the grouping fn (on ROI name), as I want.
    plot_responses_and_scaled_versions(mean_certain_df, plot_dir,
        f'{fname_prefix}certain_mean', title=title, **roimean_plot_kws
    )


# TODO function for making "abbreviated" tiffs, each with the first ~2-3s of baseline,
# 4-7s of response (~7 should allow some baseline after) per trial, for more quickly
# drawing ROIs (without using some more processed version) (leaning towards just trying
# to use the max-trial-dff or something instead though... may still be useful for
# checking that?)
def main():
    # TODO make it so it doesn't matter where i run script from. install it via
    # setup.py?
    global names2final_concs
    global seen_stimulus_yamls2thorimage_dirs
    global names_and_concs2analysis_dirs
    global ignore_existing
    global retry_previously_failed
    global analyze_glomeruli_diagnostics_only
    global analyze_glomeruli_diagnostics
    global print_skipped

    # TODO actually use log/delete / go back to global?
    # (might just get a lot of matplotlib, etc logging there (maybe regardless)
    # (or still init at top, but don't do the any logging / making dirs in init_logger
    # unless __name__ is __main__? don't want that to happen if we just import some
    # stuff from al_analysis.py...)
    log = init_logger(__name__, __file__)

    # TODO TODO TODO would this screw up any existing plots? for some i was already
    # specifying this explicitly... (look at at least plot_rois outputs)
    #
    # TODO maybe default to 300 if plot_fmt is pdf? ever worth time/space savings to do
    # 100 instead of 300, or just always do 300?
    #
    # Up from default of 100. Putting in main so it doesn't affect plot_roi[_util].py
    # interactive plotting.
    plt.rcParams['figure.dpi'] = 300

    # TODO any issues if one process is started, is stuck at a debugger (or otherwise
    # finishes after 2nd proecss), and then a 2nd process starts and finishes after?
    # worst case?
    atexit.register(cleanup_created_dirs_and_links)

    parser = argparse.ArgumentParser()

    # TODO support ending path substrings with '/' to indicate, for instance, that
    # '2-22/1/kiwi/' should not run on 2-22/1/kiwi_ea_eb_only data

    parser.add_argument('matching_substrs', nargs='*', help='If passed, only data whose'
        ' ThorImage path contains one of these substrings will be analyzed.'
    )

    # TODO TODO argument / script to delete all mocorr related outputs for a given fly?
    # (making sure to leave any things like e.g. RoiSet.zip in place)

    # TODO script / CLI argument to delete all TIFFs generated from suite2p binary that
    # do NOT correspond to currently linked mocorr dirs
    # (or to delete all dirs under suite2p_runs/ dirs that are not linked to)
    # TODO or to just delete all other runs wholly (probably also renumbering remaining
    # suite2p_runs/ subdir to '0/')

    # TODO TODO what is currently causing this to hang on ~ when it is done with
    # iterating over the inputs? some big data it's trying to [de]serialize?
    parser.add_argument('-j', '--parallel', action='store_true',
        help='Enables parallel calls to process_recording. '
        'Disabled by default because it can complicate debugging.'
    )
    # TODO add 'bounding-frames' or something similar as another --ignore-existing
    # option, to recompute all the trial_bounding_frames.yaml files
    # TODO use choices= kwarg w/ ignore_existing_options?
    # TODO update first part of help?
    parser.add_argument('-i', '--ignore-existing', nargs='?', const=True, default=False,
        help='Re-calculate non-ROI analysis and analysis downstream of ImageJ/suite2p '
        'ROIs. If an argument is supplied, must be a comma separated list of strings, '
        f'whose elements must be in {ignore_existing_options}.'
    )
    # TODO or maybe -s (with no string following) should skip default, and no steps
    # skipped if -s isn't passed (probably)?
    # TODO add across fly pdfs to this (probably also in defaults?)?
    skippable_steps = ('intensity', 'corr', 'model')
    parser.add_argument('-s', '--skip', nargs='?', const='',
        default='intensity,corr,model',
        help='Comma separated list of steps to skip (default: %(default)s). '
        f'Elements must be in {skippable_steps}. '
        '-s with no following string skips NO steps.'
    )
    parser.add_argument('-r', '--retry-failed', action='store_true',
        help='Retry steps that previously failed (frame-to-odor assignment or suite2p).'
    )
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Also prints paths to data that has some analysis skipped, with reasons '
        'for why things were skipped.'
    )

    # TODO TODO make something like these, but for panel (but always do diagnostics
    # anyway) (maybe -p/--panel or -e/--experiment). would have to handle diff than
    # driver/indicator, because need to get this information from data (not from
    # gsheet), and may also want to handle on something like a per-recording basis
    #
    # TODO replace these w/ changing script to always analyze all (calling w/ subsets of
    # data where needed)
    parser.add_argument('-d', '--driver', help='Only analyze data from this driver')
    parser.add_argument('-n', '--indicator',
        help='Only analyze data from this indicator'
    )

    # TODO validate
    #
    # NOTE: default start_date should exclude all pair-only experiments and only select
    # experiments as part of the return to kiwi-approximation experiments (that now
    # include ramps of eb/ea/kiwi mixture). For 2021 pair experiments, was using
    # start_date of '2021-03-07'
    #
    # 2022-10-06 should be the date beyond which the 'Exclude' (+ 'Exclusion reason')
    # column in the Google sheet is used whenever needed.
    parser.add_argument('-t', '--start-date', default='2022-10-06',
        help='Only analyze data from dates >= this. Should be in YYYY-MM-DD format '
        '(default: %(default)s)'
    )
    parser.add_argument('-e', '--end-date',
        help='Only analyze data from dates <= this. Should be in YYYY-MM-DD format'
    )
    # TODO try to still link everything already generated (same w/ pairs)
    # TODO delete? still used?
    parser.add_argument('-g', '--glomeruli-diags-only', action='store_true',
        help='Only analyze glomeruli diagnostics (mainly for use on acquisition '
        'computer).'
    )

    args = parser.parse_args()

    matching_substrs = args.matching_substrs

    parallel = args.parallel
    ignore_existing = args.ignore_existing
    steps_to_skip = args.skip
    retry_previously_failed = args.retry_failed
    analyze_glomeruli_diagnostics_only = args.glomeruli_diags_only

    driver = args.driver
    indicator = args.indicator

    start_date = args.start_date
    end_date = args.end_date

    # TODO maybe have this also apply to warnings about stuff skipped in
    # PREprocess_recording (now that i moved frame<->odor assignment fail handling code
    # there)
    verbose = args.verbose
    print_skipped = verbose

    # TODO share --ignore-existing and --skip parsing (prob refactoring into parser arg
    # to add_argument calls?) (make sure to handle no-string-passed --skip and bool
    # --ignore-existing)
    if type(ignore_existing) is not bool:
        ignore_existing = {x for x in ignore_existing.split(',') if len(x) > 0}

        for x in ignore_existing:
            if x not in ignore_existing_options:
                raise ValueError('-i/--ignore-existing must either be given no argument'
                    ', or a comma separated list of elements from '
                    f"{ignore_existing_options}. got '{ignore_existing}'."
                )

    assert type(steps_to_skip) is str
    steps_to_skip = {x for x in steps_to_skip.split(',') if len(x) > 0}
    for x in steps_to_skip:
        if x not in skippable_steps:
            # TODO share part w/ above
            raise ValueError('-s/--skip must either be given no argument, or a '
                f' comma separated list of elements from {skippable_steps}. '
                f"got '{steps_to_skip}'."
            )

    del parser, args

    main_start_s = time.time()

    if parallel:
        import matplotlib
        # Won't get warnings that some of the interactive backends give in the
        # multiprocessing case, but can't make interactive plots.
        matplotlib.use('agg')

    if analyze_glomeruli_diagnostics_only:
        analyze_glomeruli_diagnostics = True

    # TODO if i am gonna keep this, need a way to just re-link stuff without also
    # having to compute the heatmaps in the same run (current behavior) (?)
    #
    # Always want to delete and remake this in case labels in gsheet have changed.
    '''
    if exists(across_fly_diags_dir):
        shutil.rmtree(across_fly_diags_dir)

    makedirs(across_fly_diags_dir)
    '''

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

    common_paired_thor_dirs_kwargs = dict(
        start_date=start_date, end_date=end_date, ignore=bad_thorimage_dirs,
    )
    # TODO TODO TODO where are warnings like these coming from??? at least some (e.g.
    # '1-3ol' are NOT in the hardcoded order. at least, not as defined in hong2p.olf.
    # is it just a modification in here? is some code setting them from the yaml files
    # or something?)
    # TODO TODO from odor2abbrev_cache (loaded by magic inside hong2p.olf init...)?

    # TODO TODO delete this?
    # TODO TODO check names2final_concs stuff works with anything other than pairs
    # (code in process_recording is currently only run in pair case)
    # (and maybe delete code if not, or at least rename to indicate it's just for pair
    # stuff)
    #
    # NOTE: names2final_concs and seen_stimulus_yamls2thorimage_dirs are only used as
    # globals after this, not directly within main
    names2final_concs, seen_stimulus_yamls2thorimage_dirs, names_and_concs_tuples = \
        odor_names2final_concs(**common_paired_thor_dirs_kwargs)

    # This list will contain elements like:
    # ( (<recording-date>, <fly-num>), (<thorimage-dir>, <thorsync-dir>) )
    #
    # Within each (date, fly-num), the directory pairs are guaranteed to be in
    # acquisition order.
    #
    # list(...) because otherwise we get a generator back, which we will only be able to
    # iterate over once (and we are planning on using this multiple times)
    #
    # TODO TODO rename to something like recordings_to_analyze
    keys_and_paired_dirs = list(paired_thor_dirs(matching_substrs=matching_substrs,
        **common_paired_thor_dirs_kwargs
    ))
    del common_paired_thor_dirs_kwargs

    # TODO try to get autocomplete to work for panels / indicators (+ dates?)

    not_in_gsheet = [(k, d) for k, d in keys_and_paired_dirs if k not in gdf.index]
    flies_not_in_gsheet = set(k for k, d in not_in_gsheet)
    if len(flies_not_in_gsheet) > 0:
        warn('flies not in gsheet will not be analyzed if not added:\n'
            f'{format_fly_key_list(flies_not_in_gsheet)}'
        )
        keys_and_paired_dirs = [(k, d) for k, d in keys_and_paired_dirs
            if k not in flies_not_in_gsheet
        ]

    if driver is not None:
        keys_and_paired_dirs = [(k, d) for k, d in keys_and_paired_dirs
            if k in gdf.index and gdf.loc[k, 'driver'] == driver
        ]
        if len(keys_and_paired_dirs) == 0:
            raise ValueError(f'no flies with {driver=} to analyze')

    if indicator is not None:
        keys_and_paired_dirs = [(k, d) for k, d in keys_and_paired_dirs
            if k in gdf.index and gdf.loc[k, 'indicator'] == indicator
        ]
        if len(keys_and_paired_dirs) == 0:
            if driver is None:
                raise ValueError(f'no flies with {indicator=} to analyze')
            else:
                raise ValueError(f'no flies with {driver=}, {indicator=} to analyze')

    fly_key_set = set((d, f) for (d, f), _ in keys_and_paired_dirs)
    fly_key_set &= set(gdf.index.to_list())

    # Subset of the Google Sheet pertaining to the flies being analyzed.
    gsheet_subset = gdf.loc[list(fly_key_set)]
    del fly_key_set

    # Flies where 'Exclude' column in Google Sheet is True, indicating the whole fly
    # should be excluded. Each fly excluded this way should also have a reason listed in
    # the 'Exclusion Reason' column.
    n_excluded = gsheet_subset.exclude.sum()
    if n_excluded > 0:
        # TODO TODO show more info if verbose? including reason, etc
        # TODO mention specifically that it is flies matching this driver/indicator at
        # least (it is, right?)?
        warn(f'excluding {n_excluded} flies marked for exclusion in Google Sheet')

        exclude_subset = gsheet_subset[gsheet_subset.exclude]
        missing_reason = exclude_subset[exclude_subset.exclusion_reason.isna()]
        msg = "please fill in missing 'Exclusion Reason' in Google Sheet for:"
        if len(missing_reason) > 0:
            for date, fly_num in missing_reason.index:
                msg += f'\n- {format_date(date)}/{fly_num}'

            warn(msg)

        keys_and_paired_dirs = [(k, d) for k, d in keys_and_paired_dirs
            if not gdf.loc[k, 'exclude']
        ]
        gsheet_subset = gsheet_subset[~gsheet_subset.exclude]

    # TODO TODO warn if any (date, fly_num) are NaN for rows from gsheet (and say which
    # rows, so it can be fixed easily). will cause errors later (may be more obvious if
    # multiple such rows, but will probably cause errors regardless. sam ran into this
    # and it wasn't obvious to him what the issue was)

    # TODO TODO update code to loop over drivers/indicators in all relevant places
    # (or add to groupbys, etc)
    # TODO delete if ends up being redundant w/ checks in
    # fly2driver_indicator_output_dir
    unique_drivers = set(gsheet_subset.driver.unique())
    assert len(unique_drivers) == 1, (f'multiple drivers in input {unique_drivers} not '
        'supported! pass matching_substrs CLI args to limit data to one driver.'
    )
    unique_indicators = set(gsheet_subset.indicator.unique())
    assert len(unique_indicators) == 1, ('multiple indicators in input '
        f'({unique_indicators}) not supported! pass matching_substrs CLI args to limit'
        'data to one indicator.'
    )
    del gsheet_subset

    driver = unique_drivers.pop()
    indicator = unique_indicators.pop()

    # TODO TODO print output_root, so we know where to look for output of this run
    # TODO log to output_root too? and maybe prefer starting from scratch each time?
    # or at least append to the log?
    output_root = driver_indicator_output_dir(driver, indicator)
    plot_root = get_plot_root(driver, indicator)


    # TODO -i option for these?
    #
    # TODO TODO maybe this stuff should always be in a <plot_fmt> dir at root,
    # independent of driver? or maybe just at same level as this script, as before?
    # TODO TODO or maybe only regen if doesn't exist / a new -i option is passed
    # TODO separate script?
    #
    # add_sfr=True is the default, just including here for clarity.
    hallem_abs = orns.orns(columns='glomerulus', add_sfr=True)

    # TODO TODO version of this plot using deltas as input too?
    plot_remy_drosolf_corr(hallem_abs, 'hallem_orns', 'Hallem ORNs', plot_root,
        plot_responses=True
    )

    # TODO do w/ kennedy PNs too!
    # TODO TODO does it even make sense to propagate up thru olsen model as deltas?  is
    # that actually what i'm doing now? maybe pns() should force add_sfr=True
    # internally?
    pn_df = pns.pns(columns='glomerulus', add_sfr=True)
    plot_remy_drosolf_corr(pn_df, 'olsen_pns', 'Olsen PNs', plot_root)


    # TODO refactor to skip things here consistent w/ how i would in process_recording?
    # (have all skipping decisions made in a prior loop that returns a filtered
    # keys_and_paired_dirs? only exceptions might be in cases like registration across
    # recordings, where we might still want to include stuff that would otherwise be
    # filtered?)
    preprocess_recordings(keys_and_paired_dirs, verbose=verbose)

    # TODO TODO uncomment. was just to analyze some new data quickly
    '''
    for (date, fly_num), anat_dir in fly2anat_thorimage_dir.items():

        flip_lr = should_flip_lr(date, fly_num)
        if flip_lr in (True, False):
            anat_tif_name = 'flipped_anat.tif'
        else:
            assert flip_lr is None
            anat_tif_name = 'anat.tif'

        fly_analysis_dir = get_fly_analysis_dir(date, fly_num)
        anat_tif_path = fly_analysis_dir / anat_tif_name

        # TODO make -i option for these?
        if anat_tif_path.exists():
            # TODO delete
            print('would have skipped existant anat tiff (delete me)')
            # TODO uncomment
            #continue

        # TODO TODO modify to include 'green'/'red' in channel coords
        # (rather than 0/1)
        anat = read_thor_tiff_sequence(anat_dir)

        # Median over the "z-stream" / time dimension, to get a single frame per
        # depth/channel
        # TODO try mean / other?
        anat = anat.median('t')

        if flip_lr:
            # TODO TODO TODO test! (x or y?)
            # https://stackoverflow.com/questions/54677161
            anat = anat.reindex(x=anat.x[::-1])

        # should be getting the green channel
        anat_for_reg = anat.sel(c=0)

        # should go from XY shape of (384, 384) to (192, 192), to match functional
        anat_for_reg = anat_for_reg.coarsen(y=2, x=2).mean().data

        # TODO TODO TODO load correct z-spacing from xml to check


        # TODO TODO TODO (option to) handle each recording separately?
        # (to check my plane matching, for one)
        # TODO try to use suite2p binaries, rather than mocorr_concat.tif
        mocorr_concat = tifffile.imread(fly_analysis_dir / 'mocorr_concat.tif')
        # Before mean, should be of shape: (time, z, y, x)
        avg_mocorr_concat = mocorr_concat.mean(axis=0)
        del mocorr_concat

        # TODO TODO TODO get functional zstep_um
        # TODO TODO TODO get anatomical zstep_um
        # TODO use combination to get # of anatomical slices that should correspond to a
        # functional slice

        # TODO delete

        # TODO remove
        from suite2p.registration.zalign import compute_zpos

        print('compute_zpos method:')
        zcorr_list = []
        # TODO check we don't need to natsort to get 'plane0' and 'plane10' (/ similar)
        # in correct order
        for ops_path in sorted((fly_analysis_dir / 'suite2p').glob('plane*')):
            print(ops_path.name)
            ops = s2p.load_s2p_ops(ops_path)

            # TODO which path was this supposed to check? both?
            assert not bool(ops['smooth_sigma_time'])

            # TODO TODO TODO diy solution that doesn't register each frame separately
            # (just wanna register to average of a certain recording/concat, rather than
            # each time point)
            before = time.time()
            _, zcorr = compute_zpos(anat_for_reg, ops)
            dur = time.time() - before
            print(f'(at one depth) took {dur:.3f}')
            print(f'{zcorr.shape=}')
            plt.figure()
            plt.plot(zcorr.max(axis=1))
            plt.title(f'compute_pos zcorr.max() ({zcorr.shape=})')

            zcorr_list.append(zcorr)

        print()

        # TODO which thorimage_dir to use here? try to remove need for that?
        #load_suite2p_binaries(ops_path.parent)

        # TODO also want shift_frames?
        from suite2p.registration.register import register_frames

        # TODO delete this data when i'm done
        # TODO TODO TODO which data did i make this for (2022-10-07/1)? any reason i
        # can't just use my usual suite2p outputs? was i just worried i might be
        # modifying something?
        # TODO check nothing gets modified?
        #ops_path = fly_analysis_dir / 'test' / 'plane0'

        ops_path = fly_analysis_dir / 'suite2p' / 'plane0'
        ops = s2p.load_s2p_ops(ops_path)

        # TODO which path was this supposed to check? both?
        assert not bool(ops['smooth_sigma_time'])

        print('register frames method:')

        rigid_max_corr_list = []
        nonrigid_max_corr_list = []

        for avg_mocorr_plane in avg_mocorr_concat:
            # TODO matter which side i use as a template?

            # ig to use the same ROIs, it would be easier to use avg mocorr concat as a
            # template...
            template = avg_mocorr_plane.astype(np.int16)

            # copying just b/c it seems it may be modified in place in suite2p fn.
            # not sure it is. could check
            frames_to_register = anat_for_reg.astype(np.int16).copy()

            before = time.time()

            ret = register_frames(template, frames_to_register, ops=ops.copy())

            dur = time.time() - before
            print(f'(at one depth) took {dur:.3f}')

            # (and cmax = rigig max corr, if i want that)
            #frames, ymax, xmax, cmax, ymax1, xmax1, cmax1 = ret
            registered_anat = ret[0]
            # this seems to be of shape (z,)
            rigid_max_corr = ret[3]
            rigid_max_corr_list.append(rigid_max_corr)

            plt.figure()
            plt.plot(rigid_max_corr)

            # TODO why is this of shape (z, 36)? in each block? average them?
            nonrigid_max_corr = ret[-1]
            nonrigid_max_corr_list.append(nonrigid_max_corr)

            plt.figure()
            plt.plot(nonrigid_max_corr.mean(axis=-1))

            #break

        print()

        # TODO assert new shape matches functional here, if this all works

        plt.show()

        import ipdb; ipdb.set_trace()
        #

        # TODO TODO TODO register average movies frames to volume (assuming 12um
        # spacing, or letting each go free?)
        # TODO just for average mocorr movie? for average of each recording?

        # TODO TODO TODO register each plane of average fn recording to each plane in
        # median-anat, and look for peaks in some registration metric (after
        # smoothing?)?
        # TODO try any fns suite2p might have that could already do some of this?

        # TODO check this works w/ DataArray input (+ support, if not)
        #
        # If we weren't eliminating the "z-stream"/time dimension, we would want
        # dims='CZTYX'.
        # TODO move writing of this just after check it exists
        # (just down here to prevent skipping code i want to test)
        util.write_tiff(anat_tif_path, anat.data, dims='CZYX', strict_dtype=False)

        import ipdb; ipdb.set_trace()
    '''

    # TODO TODO TODO also check set of odors [+ panel, at least] would never be
    # concatenated together (as diagnostics1 / diagnostics1_redo somehow were for
    # 2023-06-22/1...)
    # TODO TODO TODO how did diagnostics1 not get flagged w/ redo logic in that
    # 2023-06-22/1 case? something like already having been partially run?

    # TODO --ignore-existing option for mocorr?
    if do_register_all_fly_recordings_together:
        # TODO rename to something like just "register_recordings" and implement
        # switching between no registration / whole registration / movie-by-movie
        # registration within?
        register_all_fly_recordings_together(keys_and_paired_dirs, verbose=verbose)
        print()

    if not parallel:
        # `list` call is just so `starmap` actually evaluates the fn on its input.
        # `starmap` just returns a generator otherwise.
        was_processed = list(starmap(process_recording, keys_and_paired_dirs))
    else:
        with multiprocessing.Manager() as manager:
            # "If processes is None then the number returned by os.cpu_count() is used
            # [for the number of processes in the Pool]"
            # https://docs.python.org/3/library/multiprocessing.html
            n_workers = os.cpu_count()
            print(f'Processing recordings with {n_workers} workers')

            # TODO maybe define as context manager where __enter__ returns (yields,
            # actually, right?) this + __exit__ casts back to orig types and puts those
            # values in globals()
            # TODO possible to have a manager.<type> object under __globals__, or too
            # risky? to avoid need to pass extra
            # TODO print everything processed here w/ a verbosity CLI option, to debug
            # multiprocessing path
            shared_state = multiprocessing_namespace_from_globals(manager)

            # I'm handling names_and_concs2analysis_dirs specially (rather than letting
            # multiprocessing_namespace_from_globals handle it), because I want
            # something like a defaultdict, but there is no multiprocesssing proxy for
            # that, and it seemed to make the most sense to just pre-populate all the
            # values as empty list proxies (I don't think the workers could do it
            # themselves).

            # TODO no way to instantiate new empty manager.lists() as values inside a
            # worker, is there? could remove some of the complexity i added if that were
            # the case. didn't initially seem so...
            _names_and_concs2analysis_dirs = manager.dict()
            for ns_and_cs in names_and_concs_tuples:
                _names_and_concs2analysis_dirs[ns_and_cs] = manager.list()

            shared_state['names_and_concs2analysis_dirs'] = \
                _names_and_concs2analysis_dirs

            with manager.Pool(n_workers) as pool:
                was_processed = pool.starmap(
                    #worker_fn,
                    process_recording,
                    [x + (shared_state,) for x in keys_and_paired_dirs]
                )

            multiprocessing_namespace_to_globals(shared_state)

            names_and_concs2analysis_dirs = {
                k: ds for k, ds in names_and_concs2analysis_dirs.items() if len(ds) > 0
            }

    if verbose:
        print(f'Checked {len(was_processed)} recordings(s)')

        n_processed = sum([x is not None for x in was_processed])
        print(f'Processed {n_processed} recordings(s)')

        # TODO also report time for next step(s)? if verbose maybe report time for most
        # steps?
        # TODO TODO make a new decorator to wrap all major-analysis-step fns, and use
        # that to report times of most things in verbose case?
        # TODO and print name of those fns by default too (as they are called)?
        total_s = time.time() - main_start_s
        print(f'Took {total_s:.0f}s\n')

    # TODO also have it regenerate if for some reason a concat tiff doesn't already
    # exist for a processed tiff type we want
    for fly_analysis_dir in sorted(set(flies_with_new_processed_tiffs)):
        # TODO refactor? currently duplicating in code that decides whether to link to
        # diagnostic RoiSet.zip in process_recording
        # TODO TODO TODO but may need to check that input tiffs didn't change
        # (e.g. might have since added frame<->odor YAML for one fly that was already in
        # suite2p run, but didn't previously have any of the concat tiffs / json files)
        # TODO warn if not all recordings included in suite2p run (need to read ops for
        # ops['tiff_list'], or another way?)

        # TODO TODO compare mtime of output of concatenation w/ concatenation
        # inputs (and don't continue on just this...)

        # TODO change this (want concatenating to work when min_input < 'mocorr')
        if not (fly_analysis_dir / mocorr_concat_tiff_basename).exists():
            continue

        for tiff_basename in (trial_dff_tiff_basename, trialmean_dff_tiff_basename):

            tiffs_to_concat = sorted(fly_analysis_dir.glob(f'*/{tiff_basename}'),
                key=lambda d: thor.get_thorimage_time(analysis2thorimage_dir(d.parent))
            )

            concat_tiff_path = (
                fly_analysis_dir / f'{Path(tiff_basename).stem}_concat.tif'
            )

            data_to_concat = [tifffile.imread(x) for x in tiffs_to_concat]

            # This special handling was needed to concatenate data from experiments with
            # multiple odors (typical) with those with just a single odor (CO2 in the
            # case I needed it for).
            shape_lens = {len(x.shape) for x in data_to_concat}
            if len(shape_lens) > 1:
                max_len = max(shape_lens)
                min_len = min(shape_lens)
                assert min_len == max_len - 1

                max_len_shapes_without_time_dim = {
                    x.shape[1:] for x in data_to_concat if len(x.shape) == max_len
                }
                assert len(max_len_shapes_without_time_dim) == 1

                # Assuming these are all simply missing the time/odor dimension,
                # but otherwise share the spatial[/channel] part of the shape
                min_len_shapes = {
                    x.shape for x in data_to_concat if len(x.shape) == min_len
                }
                assert min_len_shapes == max_len_shapes_without_time_dim

                data_to_concat = [x if len(x.shape) == max_len else np.expand_dims(x, 0)
                    for x in data_to_concat
                ]

            processed_concat = np.concatenate(data_to_concat)

            # TODO print that we are doing this / what we are writing
            util.write_tiff(concat_tiff_path, processed_concat, strict_dtype=False)

        # TODO also run on flies that may not have concatenated processed tiffs, but
        # already have the inputs (from older analyses). might wanna just run this
        # script on any affected data w/ `-i nonroi` to regen the old processed tiffs
        # tho...

        # TODO refactor so min/max sections can share code
        #
        # For these maximum-across-odors dF/F TIFFs, we want to compute max of inputs,
        # rather than concatenating. No use sorting, since we will collapse across
        # "time" (trial/odor) dimension.
        input_max_dff_tiffs = fly_analysis_dir.glob(
            f'*/{max_trialmean_dff_tiff_basename}'
        )
        max_of_maxes = np.max([tifffile.imread(x) for x in input_max_dff_tiffs], axis=0)
        max_of_maxes_tiff_path = fly_analysis_dir / max_trialmean_dff_tiff_basename
        util.write_tiff(max_of_maxes_tiff_path, max_of_maxes, strict_dtype=False)

        input_min_dff_tiffs = fly_analysis_dir.glob(
            f'*/{min_trialmean_dff_tiff_basename}'
        )
        min_of_mins = np.min([tifffile.imread(x) for x in input_min_dff_tiffs], axis=0)
        min_of_mins_tiff_path = fly_analysis_dir / min_trialmean_dff_tiff_basename
        util.write_tiff(min_of_mins_tiff_path, min_of_mins, strict_dtype=False)

        # TODO also save image grid plots of each of the above aggregates?


    # TODO uncomment
    # TODO move after ijroi stuff? or at least don't fail here if odors duplicated
    # TODO TODO TODO put in fn outside of main -> skip this step by default (via -s)
    '''
    # TODO should i be making one pdf per fly instead (picking one ROI plot for each,
    # etc)? would require lots of changes
    #
    # Should contain one PDF per (fly, panel) combo, mainly to be used for printing
    # out and comparing response/summary images side-by-side, with odors and figures
    # in the same order.
    # TODO TODO is odor sorting across recordings actually working? is that what i want?
    # or just presentation order? probably do want for randomized stuff... to compare.
    # test!
    all_fly_panel_pdfs_dir = plot_root / 'fly_panel_pdfs'
    makedirs(all_fly_panel_pdfs_dir)

    section_order = [
        'ROIs',
        'Summary images',
        'Trial-mean response volumes',
    ]
    # TODO replace w/ just *2fig_names (as none of these ever actually use the
    # wildcard. ...so far)
    #
    # ('.<plot_fmt>' added to end of each)
    section_names2fig_globs = {
        # NOTE: these are currently only saved for glomeruli diagnostic recordings
        # (in ij_traces)
        'ROIs': ['ijroi/all_rois_on_avg'],

        # lumping these together into one "section" for now, b/c as is template.tex
        # puts a pagebreak after each section
        # TODO maybe just add an option to not do that though?
        # TODO TODO need to deal w/ multiple recordings each having these?
        # (things other than plots 1:1 w/ odors in experiment, counting across all
        # recordings)
        'Summary images': [
            'avg', 'max_trialmean_dff', 'min_trialmean_dff'
        ],
    }
    # Since glob doesn't actually use regex, and what it can do is very limited.
    # Only one pattern per section here.
    section_names2fig_regex = {

        # E.g. '1_pfo_0.pdf'
        # Excluding stuff like: '1_pfo_0_trials.pdf'
        #
        # To exclude '_trials': https://stackoverflow.com/questions/24311832
        #
        # TODO how to indicate that these should be sorted according to odors?
        'Trial-mean response volumes': r'\d+_(?!.*_trials).*',
    }
    required_sections = {
        'Trial-mean response volumes',
    }
    assert len(
        set(section_names2fig_globs.keys()) & set(section_names2fig_regex.keys())
    ) == 0
    assert (
        set(section_order) ==
        set(section_names2fig_globs.keys()) | set(section_names2fig_regex.keys())
    )

    # NOTE: requires name of each corresponding figure to start with an integer,
    # followed by an underscore. These ints must be the presentation order of the
    # corresponding odors. This can then be combined with the known odor order, to
    # match up odors and figures, so that we can sort figures by odors.
    #
    # currently assumed that these sections will be in section_names2fig_regexes
    # (not checked in glob path below)
    sections_with_one_fig_per_experiment_odor = {'Trial-mean response volumes'}

    # TODO TODO TODO test new pdf generation code on stuff that actually splits
    # experiments across recordings (e.g. 17 odor megamat stuff, or any new diagnostics)

    print()
    print(f'saving summary PDFs under {all_fly_panel_pdfs_dir}:')

    for experiment_key, thor_image_and_sync_dirs in experiment2recording_dirs.items():

        date, fly_num, panel, is_pair = experiment_key

        # if this triggers, drop / skip these
        # TODO TODO TODO fix (want analysis to work on most things. just continue here?)
        #assert panel is not None
        if panel is None:
            continue

        experiment_type_str = str(panel)
        # initially added this to differentiate the kiwi/control recordings from the
        # kiwi/control recordings just ramping the top two components of each
        if is_pair:
            # TODO check i like how this looks in the output / output filenames
            experiment_type_str += ' pairs'

        fly_panel_id = f'{experiment_type_str}/{format_date(date)}/{fly_num}'
        fly_panel_pdf_path = (all_fly_panel_pdfs_dir /
            f"{fly_panel_id.replace('/', '_').replace(' ', '_')}.pdf"
        )

        section_names2fig_paths = defaultdict(list)

        # TODO try to de-nest this mess...
        for section_name, fig_globs in section_names2fig_globs.items():

            for fig_glob in fig_globs:
                fig_glob = f'{fig_glob}.{plot_fmt}'

                for thorimage_dir, _ in thor_image_and_sync_dirs:
                    plot_dir = get_plot_dir(date, fly_num, thorimage_dir.name)
                    assert plot_dir.is_dir()

                    # TODO if verbose, print section name and these
                    fig_paths = list(plot_dir.glob(fig_glob))

                    if len(fig_paths) == 0:
                        # Not warning about missing ROI plots for panels OTHER than the
                        # diagnostic, because this script currently only generates those
                        # plots for the diagnostic recordings.
                        if not (section_name == 'ROIs' and panel != diag_panel_str):
                            # TODO put behind verbose flag?
                            warn(f'{shorten_path(thorimage_dir)}: no figures matching '
                                f'{fig_glob}'
                            )
                        continue

                    section_names2fig_paths[section_name].extend(fig_paths)

        _thorimage2sort_index = dict()

        # TODO why was 2022-03-01/1/glomeruli_diagnosics (with no plots, just a few
        # files in analysis_dir and only thorimage/analysis links in plot_dir) among
        # experiment2recording_dirs anyway? shouldn't there have been plots if it wasn't
        # somehow getting skipped?

        for section_name, fig_regex in section_names2fig_regex.items():

            fig_regex = f'^{fig_regex}\.{plot_fmt}'

            # TODO better name (and `sort_df` inside)
            if section_name in sections_with_one_fig_per_experiment_odor:
                sort_dfs = []
            else:
                sort_dfs = None

            for thorimage_dir, _ in thor_image_and_sync_dirs:
                plot_dir = get_plot_dir(date, fly_num, thorimage_dir.name)
                assert plot_dir.is_dir()

                # Since we want to match the relative paths, as they start from
                # within plot_dir
                fig_paths = [
                    x.relative_to(plot_dir) for x in plot_dir.rglob(f'*.{plot_fmt}')
                ]
                fig_paths = [
                    plot_dir / x for x in fig_paths if re.match(fig_regex, str(x))
                ]

                if len(fig_paths) == 0:
                    warn(f'{shorten_path(thorimage_dir)}: no figures matching '
                        f'{fig_regex}'
                    )
                    continue

                if sort_dfs is None:
                    section_names2fig_paths[section_name].extend(fig_paths)
                else:
                    # TODO only do if needed (i.e. we have some of the relevant figs)
                    # TODO refactor to not need this cache?
                    if thorimage_dir not in _thorimage2sort_index:
                        # TODO need to catch something here?
                        _, _, odor_lists = util.thorimage2yaml_info_and_odor_lists(
                            thorimage_dir
                        )
                        odor_index = odor_lists_to_multiindex(odor_lists)

                        # TODO or could make a new, pandas specific, fn w/ something
                        # like this: data = data.loc[data.shift() != data]
                        # http://blog.adeel.io/2016/10/30/removing-neighboring-consecutive-only-duplicates-in-a-pandas-dataframe/
                        #
                        # TODO modify olf.remove_consecutive_repeats to work w/
                        # multiindex(/df?) input (so i don't need the list() call)?
                        # or to work w/ non-hashable input, so i could call it on
                        # odor_lists and then pass that to multiindex creation fn?
                        for_sort_index, _ = olf.remove_consecutive_repeats(
                            list(odor_index.droplevel('repeat'))
                        )
                        sort_index = pd.MultiIndex.from_tuples(for_sort_index,
                            names=[n for n in odor_index.names if n != 'repeat']
                        )

                        _thorimage2sort_index[thorimage_dir] = sort_index
                    else:
                        sort_index = _thorimage2sort_index[thorimage_dir]

                    fig_presentation_order = [
                        int(x.name.split('_')[0]) for x in fig_paths
                    ]

                    assert len(fig_paths) == len(sort_index)
                    assert (
                        len(set(fig_presentation_order)) == len(fig_presentation_order)
                    )
                    assert (
                        set(fig_presentation_order) ==
                        set(range(1, len(sort_index) + 1))
                    )
                    # could also pass odor strs thru to_filename and check that's in
                    # plot name, but that's a bit extra

                    fig_path2presentation_order = dict(
                        zip(fig_paths, fig_presentation_order)
                    )
                    fig_paths = sorted(fig_paths, key=fig_path2presentation_order.get)

                    rec_sort_df = pd.DataFrame(data=fig_paths, index=sort_index,
                        columns=['fig_path']
                    )

                    duplicated_index_rows = sort_index.duplicated(keep='first')
                    if duplicated_index_rows.any():
                        # (from the ramp experiments that interspersed them, to check
                        # for contamination)
                        assert (
                            {x for x in sort_index[duplicated_index_rows]} ==
                            {(solvent_str, solvent_str)}
                        )

                        # TODO modify handling (mainly sort_odors usage) so that we can
                        # keep the duplicates (though with the plots no longer in
                        # presentation order, the extra solvent stuff won't really be
                        # useful...)

                        # TODO warn we are dropping these (if verbose?)
                        rec_sort_df = rec_sort_df[~ duplicated_index_rows].copy()

                    sort_dfs.append(rec_sort_df)

            if sort_dfs is not None:
                if len(sort_dfs) == 0:
                    # no need to warn because currently totally redundant w/ warning
                    # about no figs matching fig_regex
                    continue

                # TODO TODO modify sort_odors args to also sort on a new variable to
                # encode the number of times each odor pair was seen before (note that
                # we already have one figure for each 3 trials here, so it's about
                # repeats of the 3 trials groups)
                # (remove the code de-duplicating the (solvent, solvent) duplicates
                # above if so)
                # TODO also check that the only pairs duplicated are (solvent, solvent)
                # TODO TODO TODO get this to not fail if i do experiments e.g. 2 sides
                # in one fly (how though? just flag to disable this whole analysis or
                # something?)
                sort_df = pd.concat(sort_dfs, verify_integrity=True)

                sort_df = sort_odors(sort_df, add_panel=panel)
                sorted_fig_paths = list(sort_df.fig_path)
                section_names2fig_paths[section_name].extend(sorted_fig_paths)

        section_names2fig_paths = {
            n: section_names2fig_paths[n] for n in section_order
        }

        for section_name in required_sections:
            if len(section_names2fig_paths[section_name]) == 0:
                warn(f"missing figures for required section '{section_name}'. "
                    'not generating PDF.'
                )
                continue

        if fly_panel_pdf_path.exists():
            # TODO add ignore existing option for this
            pdf_creation_time = getmtime(fly_panel_pdf_path)
            most_recent_fig_change = max(
                getmtime(x) for fig_paths in section_names2fig_paths.values()
                for x in fig_paths
            )
            if pdf_creation_time > most_recent_fig_change:
                if verbose:
                    print(f'{fly_panel_pdf_path.relative_to(all_fly_panel_pdfs_dir)}'
                        ' already up-to-date'
                    )

                continue

        # TODO or maybe split driver/indicator and fly_panel_id across header and
        # footer?
        header = f'{driver} ({indicator}): {fly_panel_id}'

        print(f'making {fly_panel_pdf_path.relative_to(all_fly_panel_pdfs_dir)}')

        try:
            # TODO why does first page of trialmean response volumes seem to have the 4
            # figures justified vertically differently than on the second page? fix!
            #
            # I initially tried matplotlib's PdfPages and img2pdf for more simply making
            # a PDF with a bunch of images, but neither supported multiple figures per
            # page, which is what I wanted.
            make_pdf(fly_panel_pdf_path, '.', section_names2fig_paths, header=header,
                # TODO set to true if i wanna debug
                print_tex_on_err=False
            )

        # TODO troubleshoot
        # TODO TODO try to also not have the tex printed on err
        # (the stuff still printed is something i added, right?)
        except LatexBuildError:
            # TODO TODO at least include more relevant info explaining why this happened
            # (duplicate odors? how?)
            warn('making PDF failed!')
            #cprint(traceback.format_exc(), 'yellow', file=sys.stderr)
            #print()

    print()
    '''

    if do_analyze_response_volumes and (
        not is_acquisition_host and len(response_volumes_list) > 0):

        # TODO also drop va/aa <= 2023-04-22 (when solvent was pfo) here, if i ever
        # want to re-enable do_analyze_response_volumes
        print('DROP VA/AA <= 2023-04-22 FROM EACH ELEMENT OF RESPONSE_VOLUMES_LIST')
        import ipdb; ipdb.set_trace()
        raise NotImplementedError

        # Crude way to ensure we don't overwrite if we only run on a subset of the data
        write_cache = len(matching_substrs) == 0

        # TODO TODO TODO get this to not err in quick-test case (2023-04-03 data,
        # before panel stuff). should also work w/ diagnostic only input (or at least
        # not err)
        analyze_response_volumes(response_volumes_list, output_root,
            write_cache=write_cache
        )

    # TODO delete?
    #if len(odors_without_abbrev) > 0:
    #    print('Odors without abbreviations:')
    #    pprint(odors_without_abbrev)

    def earliest_analysis_dir_date(analysis_dirs):
        return min(d.parts[-3] for d in analysis_dirs)

    failed_assigning_frames_analysis_dirs = [
        Path(str(x).replace('raw_data', 'analysis_intermediates'))
        for x in failed_assigning_frames_to_odors
    ]

    # TODO delete / reorganize into a fn that only runs if we did suite2p analysis?
    # (or at least to clean up organization of main)
    #
    # TODO TODO extend to other stuff we want to analyze (e.g. at least the
    # kiwi/control1 panel data)
    # TODO TODO also do this on a per-fly basis, now that we typically are only
    # analyzing stuff where recordings have been registered together (and only one set
    # of ROIs defined, on the diagnostic recording)
    '''
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
    # (wouldn't want to do for any diagnostic panel stuff)

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
    '''

    if len(ij_trial_dfs) == 0:
        cprint('No ImageJ ROIs defined for current experiments!', 'yellow')
        return

    n_before = sum([num_notnull(x) for x in ij_trial_dfs])

    trial_df = pd.concat(ij_trial_dfs, axis='columns')

    util.check_index_vals_unique(trial_df)
    assert num_notnull(trial_df) == n_before

    trial_df = drop_redone_odors(trial_df)

    # this does the same checks as above, internally
    # (that number of notna values does not change and index values are unique)
    trial_df = merge_rois_across_recordings(trial_df)

    trial_df_isna = trial_df.isna()
    assert not trial_df_isna.all(axis='columns').any()
    assert not trial_df_isna.all(axis='rows').any()
    del trial_df_isna

    # TODO maybe warn about / don't drop anything where it's not in a '?+' or '+?'
    # suffix (e.g. so 'VM2+VM3' will still show up in some plots?)
    #
    # TODO TODO just change behavior of hong2p.roi fn that is already dropping ROIs w/
    # '+' suffix, to drop if they have '+' anywhere (-> delete this)?
    # (would need to recompute cached ijroi stuff)
    #
    # TODO although check that if we are dropping 'x+?', 'x'/'x?' also exists? or at
    # least separate warning about that?
    contained_plus = np.array(
        [('+' in x) for x in trial_df.columns.get_level_values('roi')]
    )
    # no need to copy, because indexing with a bool mask always does
    trial_df = trial_df.loc[:, ~contained_plus]

    trial_df = drop_superfluous_uncertain_rois(trial_df)

    # TODO test that moving this before merge_rois_across_recordings doesn't change
    # anything (because merge_... currently fails w/ duplicate odors [where one had been
    # repeated later, to overwrite former], and i'd like to use a similar strategy to
    # deal w/ those cases)
    trial_df = setnull_old_wrong_solvent_aa_and_va_data(trial_df)

    trial_df = sort_fly_roi_cols(trial_df, flies_first=True)
    trial_df = sort_odors(trial_df)

    # TODO TODO TODO hardcode groups of panels to save to split out into particular
    # output CSVs? ideally would only do it when run with a canonical set of good
    # inputs, but i might need to handle that separately for now. should probably move
    # to using checkboxes/similar in google sheet for that?
    # TODO TODO TODO also print N for each combination of (panel, is_pair) in CSV

    # TODO TODO TODO save w/ driver+indicator in name
    #
    # TODO TODO TODO TODO delete. hack to remove (small number. ~5/169) ROIs that still
    # had ambiguous names, while serializing for anoop)
    #
    # just removing the 'glomeruli_diagnostics' panel
    '''
    print('FIX THIS HACK')
    natmix_rows = trial_df.index.get_level_values('panel').isin(('kiwi', 'control'))
    to_csv(
        trial_df.loc[natmix_rows, trial_df.columns.get_level_values('roi').map(
            lambda x: is_ijroi_certain(x) or not is_ijroi_named(x)
        )],
        output_root / '2023-02-21_orn_glomeruli_responses.csv',
        date_format=date_fmt_str
    )
    import ipdb; ipdb.set_trace()
    '''
    #

    # TODO TODO TODO delete (/ only do for one certain only version)
    certain_df = select_certain_rois(trial_df)

    # TODO TODO TODO make this >= half of flies (and say that's what we are doing, and
    # what the threshold is) (so plots/outputs still useful when looking at data that
    # doesn't have much yet)
    #
    # Betty picked this number when I asked if we could drop glomeruli based on this
    # criteria
    n_for_consensus = 3
    certain_glom_counts = certain_df.columns.get_level_values('roi').value_counts()
    consensus_gloms = set(
        certain_glom_counts[certain_glom_counts >= n_for_consensus].index
    )
    consensus_df = certain_df.loc[
        :, certain_df.columns.get_level_values('roi').isin(consensus_gloms)
    ]
    if len(consensus_df) < len(certain_df):
        dropped_rois = certain_df.columns.difference(consensus_df.columns).to_frame(
            index=False
        )
        warn(f'dropping the following data from glomeruli measured <{n_for_consensus} '
            f'times:\n{dropped_rois.to_string(index=False)}'
        )

    # NOTE: seems to be no need to drop GH146-IDed-only stuff in pebbled case
    # (at least as glomeruli are currently IDed, as there currently aren't any (after
    # the consensus / certain criteria, though DA1/etc came close)
    #
    # these glomeruli were only seen in pebbled (though VC5/V might also be in GH146):
    # {'DL2d', 'DC4', 'DP1l', 'V', 'VC5', 'DL2v', 'VL1', 'VM5d', 'VC4', 'VM5v', 'VC3'}

    to_csv(consensus_df, output_root / 'ij_certain-roi_stats.csv',
        date_format=date_fmt_str
    )
    to_pickle(consensus_df, output_root / 'ij_certain-roi_stats.p')
    #

    # TODO TODO TODO save another version dropping all non-certain ROIs?
    # TODO TODO and split out by panel (or just tell remy to only use 'megamat' panel?)?

    # TODO TODO any way to get date_format to apply to stuff in MultiIndex?
    # or is the issue that it's a pd.Timestamp and not datetime?
    # TODO link the github issue that might or might not still be open about this
    # date_format + index level issue (may need to update pandas if it has been
    # resolved)
    to_csv(trial_df, output_root / 'ij_roi_stats.csv', date_format=date_fmt_str)

    # TODO TODO TODO still have one global cache with all data (for use in plot_roi.py /
    # related) (or just use current ij_roi_stats.p files, where they are?)
    # TODO TODO -> use in new realtime analysis feature like hallem-correlating one, but
    # using my data, specifically certain ROIs for other flies from same driver?
    # (or maybe even pebbled by default? warn if using diff driver?)
    #
    # TODO at least don't overwrite these if we have subset str defined (same w/ csv
    # prob) (maybe also not start / end date?)
    # (would want to warn if i wasn't gonna over write this for either of these
    # reasons...)
    to_pickle(trial_df, output_root / ij_roi_responses_cache)

    # TODO factor index reshuffling into natmix.drop_mix_dilutions?
    # would it screw with usage internal to natmix (for some DataFrame inputs)?
    #
    # TODO still want to keep this?
    # This is dropping '~kiwi' and 'control mix' at concentrations other than undiluted
    # ('@ 0') concentration.
    # TODO TODO test still works on new (without natmix odors) and old (with natmix
    # odors) data, now that i moved index reshuffling into natmix.drop_mix_dilutions
    trial_df = natmix.drop_mix_dilutions(trial_df)


    across_fly_ijroi_dir = plot_root / across_fly_ijroi_dirname
    makedirs(across_fly_ijroi_dir)

    # TODO TODO also add skip option for these (but don't skip by default)
    # TODO relabel <date>/<fly_num> to one letter for both. write a text key to the same
    # directory
    print('saving across fly ImageJ ROI response matrices...', flush=True)

    uncertain_roi_plot_kws = {
        k: v for k, v in roi_plot_kws.items() if k != 'hline_level_fn'
    }

    # TODO TODO re-establish a real-time-analysis script that can compare current ROI
    # to:
    # - certain ROIs of same name, in cached (e.g. ij_roi_stats.p) driver/indicator
    #   (/pebbled) data
    #   (to see if this call would be ~consistent with other data)
    #
    # - N most correlated ROIs in the same fly, to the mean certain response profile
    #   of the cached data (at least among the uncertain ones in the current fly)
    #   (to see if any other ROIs are a better match for this glomerulus name)
    #
    # - most correlated other mean certain ROIs?

    response_matrix_plots(across_fly_ijroi_dir, trial_df)

    # TODO cluster all uncertain ROIs? to see there are any things that could be added?

    # TODO and what would it take to include ROI positions in clustering again?
    # doesn't require picking a single global best plane, right? but might require
    # caching / concating / returning masks from process_recording?

    # TODO select in a way that doesn't rely on 'panel' level being in this position?
    assert trial_df.index.names[0] == 'panel'
    # the extra square brackets prevent 'panel' level from being lost
    # (for concatenating with other subsets w/ different panels)
    # https://stackoverflow.com/questions/47886401
    diag_df = trial_df.loc[[diag_panel_str]]
    diag_df = dropna(diag_df)
    # TODO warn if diag_df is empty (after dropping some NaNs?)?

    for panel, panel_df in trial_df.groupby('panel', sort=False):
        _printed_any_flies = False

        # TODO TODO switch all code to work w/ initial 2 index levels:
        # ['panel', 'is_pair'] (-> replace all pdf w/ panel_df)
        # (was previously dropping for everything, but now that i want some plots with
        # diag panel as well, and odors might be same as some in panel, need that level
        # for now. could maybe still get rid of is_pair, but why?)
        pdf = panel_df.copy()

        # TODO switch to indexing by name (via a mask), to make more robust to changes?
        assert pdf.index.names[0] == 'panel' and pdf.index.names[1] == 'is_pair'

        # TODO warn if these drop anything? what all should they be dropping?
        # TODO these should just be dropping ROIs, right?
        # (assuming all panel odors presented in each fly w/ that panel?
        # which ig might not always be true...)
        panel_df = dropna(panel_df)
        with warnings.catch_warnings():
            # To ignore the "indexing past lexsort depth" warning.
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

            # Selecting just the is_pair=False rows, w/ the False here.
            pdf = dropna(pdf.loc[(panel, False), :])

        assert not pdf.columns.duplicated().any()

        # TODO maybe append the f' (n<={n_flies})' part later as appropriate
        # (could be diff for diff panels...)
        title = f'{driver}>{indicator}'

        # response_matrix_plots will make these if needed
        panel_ijroi_dir = across_fly_ijroi_dir / 'by_panel' / panel

        # TODO return title from this, so i can use below? not sure i care enough for a
        # title there...
        response_matrix_plots(panel_ijroi_dir, pdf)

        if panel != diag_panel_str:
            # TODO TODO still want stuff in panel_df but w/o diags tho. modify
            # (though in data i'm currently analyzing, this shouldn't ever be the
            # case...)
            # (join='outer' -> separate call dropping any stuff null for all of current
            # panel?)
            #
            # join='inner' to drop ROIs (columns) that are only in one of the inputs
            diag_and_panel_df = pd.concat([diag_df, panel_df], join='inner',
                verify_integrity=True
            )
            # TODO warn + skip if current panel doesn't have any associated diag data
            # (shouldn't be the case in any data i'm analyzing rn, but people might want
            # it)

            # TODO TODO TODO still make a version not dropping these?
            if driver == 'GH146':
                # TODO refactor
                gh146_glomeruli = get_gh146_glomeruli()
                # (so plot ACTUALLY lines up with pebbled-subset-to-GH146 plot)_
                diag_and_panel_df = diag_and_panel_df.loc[:,
                    diag_and_panel_df.columns.get_level_values('roi'
                        ).isin(gh146_glomeruli)
                ]
                # TODO TODO also plot the pebbled-only stuff in a separate plot
                # (or make some kind of matshow comparison plot, w/ 2 df inputs, and use
                # that to make something in here? where pebbled only stuff would be
                # sorted in a separate section at bottom)

            # TODO replace above response_matrix_call w/ this one (when we have diags,
            # at least?)
            response_matrix_plots(panel_ijroi_dir, diag_and_panel_df, 'with-diags')

            # TODO TODO TODO need to at least drop stuff from GH146 plots that get
            # dropped in get_gh146_glomeruli (the stuff in <1/2 of flies)
            # TODO also only do for panels we are doing both gh146 and pebbled on
            # (e.g. megamat). same restriction for this variant of correlation analysis
            if driver in orn_drivers:
                # TODO TODO or maybe try just NaNing in gh146 case (to add rows for
                # stuff in pb but not gh146)
                gh146_glomeruli = get_gh146_glomeruli()
                gh146_only_diag_and_panel_df = diag_and_panel_df.loc[:,
                    diag_and_panel_df.columns.get_level_values('roi'
                        ).isin(gh146_glomeruli)
                ]
                response_matrix_plots(panel_ijroi_dir, gh146_only_diag_and_panel_df,
                    'gh146-only_with-diags'
                )

            # TODO some reason i'm not using .map(...)?
            mask = np.array([ijroi_comparable_via_name(x)
                for x in diag_and_panel_df.columns.get_level_values('roi')
            ])
            # no need to copy, because indexing with a bool mask always does
            comparable_via_name = diag_and_panel_df.loc[:, mask]
            del mask

            # TODO show a mean for the certain things for each group? (eh...)
            # (rather than each of the lines individually?)

            df = comparable_via_name
            del comparable_via_name

            # TODO TODO TODO why is this shape (129, 0) and not original (129, 294)?
            # (for panel='megamat') (matter? delete?)
            #df.dropna(how='any', axis='columns').shape

            fly_rois = df.columns.to_frame(index=False)
            fly_rois['name_as_if_certain'] = fly_rois.roi.map(ijroi_name_as_if_certain)

            # TODO might need to handle if triggered
            assert not fly_rois.name_as_if_certain.isna().any()

            # TODO warn if number of ROIs w/ same ijroi_name_as_if_certain are
            # < n_flies for any (including if all certain / not)
            # (still want that separately? currently warning on a fly-by-fly basis,
            # which could include all the same ROIs by the end...)
            # or build + write summary CSV?
            #
            # TODO factor out n_flies calc?
            # TODO nunique?
            n_flies = len(fly_rois[['date','fly_num']].drop_duplicates())

            glom_counts = fly_rois.name_as_if_certain.value_counts()
            # TODO TODO how could this possibly fail? fix!
            # ipdb> fly_rois[fly_rois.name_as_if_certain == 'DP1l']
            #           date fly_num     roi name_as_if_certain
            # 16  2023-04-22       2    DP1l               DP1l
            # 56  2023-04-22       3    DP1l               DP1l
            # 95  2023-04-26       2    DP1l               DP1l
            # 133 2023-04-26       3    DP1l               DP1l
            # 167 2023-05-08       1    DP1l               DP1l
            # 206 2023-05-08       3   DP1l?               DP1l
            # 207 2023-05-08       3  DP1l??               DP1l
            # 244 2023-05-09       1    DP1l               DP1l
            # 285 2023-05-10       1    DP1l               DP1l
            # 325 2023-06-22       1   DP1l?               DP1l
            # TODO better error message (in case i accidentally do this again)
            assert (glom_counts <= n_flies).all()

            fly_rois['certain'] = fly_rois.roi.map(is_ijroi_certain)

            allfly_certain_roi_set = set(fly_rois[fly_rois.certain].roi)
            # TODO delete?
            #allfly_roi_set = set(fly_rois.name_as_if_certain)
            #print('ROIs only ever uncertain: '
            #    f'{pformat(allfly_roi_set - allfly_certain_roi_set)}'
            #)

            # TODO replace ['date','fly_num'] w/ fly_keys
            # (and <same> + ['roi'] w/ roi_keys) ...here, and elsewhere
            # TODO groupby_fly_keys fn?
            # TODO some way to groupby df/df.columns directly (WHILE simplifying this
            # section, in the process) (so as to not need to make fly_rois)?
            for (date, fly_num), onefly_rois in fly_rois.groupby(['date','fly_num']):

                date_str = format_date(date)
                fly_str = f'{date_str}/{fly_num} ({panel}):'

                _printed_flystr = False
                def _print_flystr_if_havent():
                    nonlocal _printed_flystr
                    nonlocal _printed_any_flies
                    if not _printed_flystr:
                        print(fly_str)
                        _printed_flystr = True
                        _printed_any_flies = True

                def _print_rois(rois):
                    if isinstance(rois, pd.Series) and rois.dtype == bool:
                        rois = rois[rois].index

                    # TODO sort s.t. certain counts prioritized?
                    # (if i'm just gonna sort on one number tho, what i'm sorting on now
                    # [i.e. name_as_if_certain] is probably the way to go)
                    # TODO also include glom_counts value in parens (rather than just
                    # sorting on them?)
                    pprint(set(sorted(rois, key=glom_counts.get)))

                missing_completely = (
                    allfly_certain_roi_set - set(onefly_rois.name_as_if_certain)
                )
                if len(missing_completely) > 0:
                    _print_flystr_if_havent()

                    # TODO TODO TODO also just drop stuff in <=~2 (/9) flies?
                    # (earlier? maybe even before saving CSV?)
                    print('missing completely (marked certain in >=1 fly):')
                    _print_rois(missing_completely)

                # TODO sort=False actually doing anything?
                uncertain_only = ~ onefly_rois.groupby('name_as_if_certain', sort=False
                    ).certain.any()

                if uncertain_only.any():
                    _print_flystr_if_havent()
                    print('uncertain:')
                    # NOTE: uncertain_only only has name_as_if_certain, not raw name
                    # (w/ '?' or whatever suffix)
                    _print_rois(uncertain_only)

                # TODO TODO TODO also warn about stuff where the name entered doesn't
                # refer to the specific glomerulus name (e.g. 'VA7' for 'VA7m' vs 'VA7l'
                # (or 'DL2' vs 'DL2v'/'DL2d', etc) )
                # (could just do based on whether any ROIs have another ROI name as a
                # prefix? any cases where this wouldn't work?)

                if _printed_flystr:
                    print()

            # TODO count use certain_counts == 0 if i want another summary of
            # uncertain-only ROIs (besides fly-by-fly prints in loop below)
            #certain_counts = fly_rois.groupby('name_as_if_certain').certain.sum()
            # TODO don't actually need/want n_flies, right? cause if all certain and
            # have less than n flies, still want to drop from plot, no?
            #certain_only = (certain_counts == n_flies)

            certain_only = fly_rois.groupby('name_as_if_certain').certain.all()
            certain_only = set(certain_only[certain_only].index)

            df = df.loc[:, ~df.columns.get_level_values('roi').isin(certain_only)]

            # TODO version of this plot, also comparing to most correlated other stuff
            # in same fly each comes from? (would be too busy?)

            # TODO diff hline thicknesses between uncertain/certain vs ROI-name-changed
            # cases? possible? or change color/font of ticklabels for subgroups?
            #
            # default ROI sorting should put uncertain after certain
            # (just b/c default of 'x' < 'x?')
            plot_responses_and_scaled_versions(df, panel_ijroi_dir,
                'with-diags_certain_vs_uncertain', title=title, **roi_plot_kws
            )
            del df

        # TODO change so it only happens if there are more panels to come (print at
        # start of loop?)
        if _printed_any_flies:
            print()

        # TODO TODO move all the below plots into response_matrix_plots (/delete)?
        # (maybe behind a new flag like plot_uncertain?)

        # TODO want to keep this plot? what purpose does it serve?
        # is something focusing on just uncertain ROIs not enough?
        fig, _ = plot_all_roi_mean_responses(pdf, title=title, **roi_plot_kws)
        savefig(fig, panel_ijroi_dir, 'with-uncertain')

        uncertain_rois = ~certain_roi_indices(pdf)

        # TODO cluster all ROIs within each fly (or at least the non-certain ones?)
        # (for answering the question: is there anything with similar tuning to this ROI
        # i'm currently trying to identify?)

        # TODO these plots still useful?
        # TODO TODO also cluster + plot w/ normalized (max->1, min->0) rows
        # (/ "z-scored" ok?)
        # TODO maybe turn this into a fn looking up max using index?
        glom_maxes = pdf.max(axis='rows')
        fig, _ = plot_all_roi_mean_responses(pdf.loc[:, uncertain_rois],
            # negative glom_maxes, so sort is as if ascending=False
            sort_rois_first_on=-glom_maxes[uncertain_rois], title=title,
            **uncertain_roi_plot_kws
        )
        savefig(fig, panel_ijroi_dir, 'uncertain_by_max_resp')

        # TODO TODO (also) cluster only uncertain stuff?
        # or maybe only unnamed stuff? maybe uncertain stuff (specifically the subset w/
        # '?' suffix), should be handled mainly in another plot, comparing to each
        # other / similar certain ROIs in other flies (when available)?

        # TODO same type of clustering, but with responses to *all* odors
        # (or maybe panel + diags? couldn't  really do all since everything would
        # probably get dropped, since flies tend to only have one panel beyond diags...)
        # (for trying to identify set of glomeruli we are dealing with)
        #
        # Mean across repeats (done by plot_all_roi_mean_responses in other cases).
        # Should be one row per odor after.
        mean_df = pdf.groupby('odor1').mean()

        # TODO replace this w/ sequential drops along both axes (maybe in order that
        # keeps most data, if there is one?)? currently getting empty matrices in some
        # cases where i shouldn't need too (index=odors, for reference)
        # (on which data?  still an issue?)
        #
        # Clustering will fail if there are any NaN in the input.
        clust_df = mean_df.dropna(how='any', axis='index')

        # TODO TODO and why was dropna(df, how='any') not working even when ROIs
        # were dropped first? shouldn't be empty, no?
        # (still an issue?)

        # Clustering will also fail if the dropna above drops *everything*
        # TODO TODO fail early / skip here if df empty
        # (still an issue?)

        # TODO TODO fix cause of (it's b/c after dropna above, clust_df can be
        # empty):
        # ValueError: zero-size array to reduction operation fmin which has no
        # identity
        # (is it simply an issue of only have one recording as input?)
        try:
            cg = cluster_rois(clust_df, odor_sort=False, title=title)
            savefig(cg, panel_ijroi_dir, 'with-uncertain_clust')
        except ValueError:
            traceback.print_exc()
            import ipdb; ipdb.set_trace()

    print('done')


    # TODO if --verbose, print when we skip this
    if 'corr' not in steps_to_skip:
        # TODO TODO move this enumeration of certain_only values inside
        # all_correlation_plots?)
        # TODO rename across_fly...? (and do same for similar fns to organize across fly
        # analysis steps?)
        all_correlation_plots(output_root, trial_df, certain_only=True)
        all_correlation_plots(output_root, trial_df, certain_only=False)


    # TODO if --verbose, print when we skip this
    if 'intensity' not in steps_to_skip:
        trial_ser = trial_df.stack(trial_df.columns.names)
        assert trial_df.notnull().sum().sum() == trial_ser.notnull().sum()
        tidy_trial_df = trial_ser.reset_index(name='mean_dff')

        # TODO TODO how come the pixelwise analysis has a few more rows than this?
        #
        # Taking mean across ROIs and across trials, so there should be one number
        # describing activation strength (still in mean_dff column), for each fly X odor
        mean_df = tidy_trial_df.groupby([
            x for x in tidy_trial_df.columns if x not in ('repeat', 'roi', 'mean_dff')
        ], sort=False).mean_dff.mean().reset_index()

        intensities_plot_dir = across_fly_ijroi_dir / 'activation_strengths'
        natmix_activation_strength_plots(mean_df, intensities_plot_dir)


    # TODO at least if --verbose, print we are skipping step (and in all cases we skip
    # steps)
    if 'model' not in steps_to_skip:
        # TODO worth warning that model won't be run otherwise?
        if driver in orn_drivers:
            model_mb_responses(certain_df, across_fly_ijroi_dir)
        else:
            print(f'not running MB model(s), as driver not in {orn_drivers=}')


if __name__ == '__main__':
    main()

