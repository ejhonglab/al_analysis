#!/usr/bin/env python3

import argparse
import atexit
import os
from os.path import join, split, exists, expanduser, islink, getmtime
from pprint import pprint, pformat
from collections import defaultdict, Counter
from copy import deepcopy
from functools import wraps
import filecmp
from datetime import datetime
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
import itertools
import multiprocessing
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Optional, Tuple, List, Type, Union, Dict, Set, Any
import json
import re
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import xarray as xr
import tifffile
import yaml
import ijroi
from matplotlib import colors, patches
from matplotlib.colors import to_rgba
import matplotlib.patheffects as PathEffects
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import matplotlib.pyplot as plt
# TODO need to install something else for this, in a new env?
# (might have manually installed before in current one...)
from matplotlib.testing import compare as mpl_compare
#
from matplotlib_scalebar.scalebar import ScaleBar
import seaborn as sns
import colorcet as cc
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr, f_oneway
from sklearn.preprocessing import maxabs_scale as sk_maxabs_scale
from sklearn.preprocessing import minmax_scale as sk_minmax_scale
from sklearn import metrics
import statsmodels.api as sm
import colorama
from termcolor import cprint, colored
import olfsysm as osm
from drosolf import orns, pns
from latex.exc import LatexBuildError
from tqdm import tqdm
# suite2p imports are currently done at the top of functions that use them
#
# for type hinting
from statsmodels.regression.linear_model import RegressionResultsWrapper

from hong2p import util, thor, viz, olf
from hong2p import suite2p as s2p
from hong2p.suite2p import LabelsNotModifiedError, LabelsNotSelectiveError
from hong2p.roi import (rois2best_planes_only, ijroi_filename, has_ijrois, ijroi_mtime,
    ijroi_masks, extract_traces_bool_masks, ijroiset_default_basename, is_ijroi_named,
    is_ijroi_certain, ijroi_name_as_if_certain, ijroi_comparable_via_name,
    certain_roi_indices, select_certain_rois, is_ijroi_plane_outline
)
from hong2p.util import (shorten_path, shorten_stimfile_path, format_date, date_fmt_str,
    # TODO refactor current stuff to use these (num_[not]null)
    num_null, num_notnull, add_fly_id, frame_pdist, pd_allclose
)
from hong2p.olf import (format_mix_from_strs, format_odor_list, solvent_str,
    odor2abbrev, odor_lists_to_multiindex
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


# TODO TODO TODO restore (triggered in dF/F calc for test no-fly dry-run data)
# RuntimeWarning: invalid value encountered in scalar multiply
#warnings.filterwarnings('error', 'invalid value encountered in')

# TODO delete or add note about what was previously causing this (+ shorten/remove
# traceback)
#
# TODO fix! (can i repro this now? maybe fixed?)
# (venv) tom@atlas:~/src/al_analysis$ ./al_analysis.py -d pebbled -n 6f -t 2023-11-19
# ...
# thorimage_dir: 2024-01-05/1/diagnostics1
# thorsync_dir: 2024-01-05/1/SyncData001
# yaml_path: 20240105_164541_stimuli/20240105_164541_stimuli_0.yaml
# ImageJ ROIs were modified. re-analyzing.
# Warning: dropping 17 ROIs with '+' suffix
# picking best plane for each ROI
# scale_per_plane=False
# minmax_clip_frac=0.025
# kwargs.get("norm")='log'
# from minmax_clip_frac: vmin=114.7911111111111 vmax=3190.0786666666663
# Uncaught exception
# Traceback (most recent call last):
#   File "./al_analysis.py", line 11600, in <module>
#     main()
#   File "./al_analysis.py", line 10705, in main
#     was_processed = list(starmap(process_recording, keys_and_paired_dirs))
#   File "./al_analysis.py", line 4380, in process_recording
#     ij_trial_df, best_plane_rois, full_rois = ij_trace_plots(analysis_dir,
#   File "./al_analysis.py", line 3486, in ij_trace_plots
#     traces, best_plane_rois, z_indices, full_rois = ij_traces(analysis_dir, movie,
#   File "./al_analysis.py", line 3220, in ij_traces
#     fig_path = savefig(fig, ij_plot_dir, f'all_rois_on_{bg_desc}')
#   File "./al_analysis.py", line 1578, in savefig
#     fig_or_seaborngrid.savefig(fig_path, **kwargs)
#   ...
# UserWarning: There are no gridspecs with layoutgrids. Possibly did not call parent GridSpec with the "figure" keyword
#warnings.filterwarnings('error', message='There are no gridspecs with layoutgrids.*')

# should be resolved now. was from a sns.lineplot call with kwargs indicating we wanted
# error, but without anything to compute error over in input data.
warnings.filterwarnings('error', message='All-NaN axis encountered')

# to filter on substring of message: https://stackoverflow.com/questions/65761137
#
# may need to restrict this, as i think it might still be getting triggered some places.
# i think this might be the only way to make sure we don't have plots with text outside
# visible plot, as for ax.text added for viz.matshow group labels (if offsets, etc,
# don't work out such that text is in visible region). can't seem to find a good
# solution to getting text reliably in plot w/ constrained layout. see:
# https://github.com/matplotlib/matplotlib/issues/19358
# https://stackoverflow.com/questions/63505647/add-external-margins-with-constrained-layout
# TODO can i have that part of viz.matshow add (a restricted version of?) this filter
# automatically during call?
warnings.filterwarnings('error', message='constrained_layout not applied.*')

# initially encountered when making FacetGrids w/o disabling constrained layout in
# context manager
#warnings.filterwarnings('error', message='The figure layout has changed to tight')

# most commonly:
# pandas.errors.PerformanceWarning: indexing past lexsort depth may impact performance.
# but some other things can trigger this warning with different messages.
#warnings.simplefilter('error', pd.errors.PerformanceWarning)

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

# TODO set (w/ context manager?) so it only applies to the diag example plot_rois
# outputs. (and i assume it was impossible to set directly for some reason? why tho?)
#
# https://stackoverflow.com/questions/12434426
# KeyError: 'lines.dashed_style is not a valid rc parameter ...
plt.rcParams['lines.dashed_pattern'] = [2.0, 1.0]

# TODO also add to a matplotlibrc file (under ~/.matplotlib?)?
# TODO 42 same as TrueType?
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# TODO delete. to enable some extra debugging prints, trying to not have them show up if
# sam uses that version of hong2p
# TODO restore True and see if there's anything unresolved about matshow
# add_norm_options / twoslopenorm [+ cbar ticks?] handling ?
viz._debug = False


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


# TODO TODO also use for wPNKC(s)? anything else?
data_outputs_root = Path('data')
hallem_csv_root = data_outputs_root / 'preprocessed_hallem'

hallem_delta_csv = hallem_csv_root / 'hallem_orn_deltas.csv'
hallem_sfr_csv = hallem_csv_root / 'hallem_sfr.csv'

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

# TODO warn once if skip_singlefly_trace_plots=True? cli option to re-enable them (prob
# not)?
# TODO TODO restore True (after debugging motion issue in 2023-11-21/2 b-Myr)
skip_singlefly_trace_plots = False
#skip_singlefly_trace_plots = True

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

# NOTE: never really used this, nor seriously compared outputs generated using it.
#
# if False, one baseline per trial. if True, baseline to first trial of each odor.
one_baseline_per_odor = False

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
plot_fmt = os.environ.get('plot_fmt', 'pdf')
#plot_fmt = os.environ.get('plot_fmt', 'png')

# Overall folder structure should be: <driver>_<indicator>/<plot_fmt>/...
across_fly_ijroi_dirname = 'ijroi'
across_fly_pair_dirname = 'pairs'
across_fly_diags_dirname = 'glomeruli_diagnostics'

trial_and_frame_json_basename = 'trial_frames_and_odors.json'
ij_trial_df_cache_basename = 'ij_trial_df_cache.p'

mocorr_concat_tiff_basename = 'mocorr_concat.tif'

all_frame_avg_tiff_basename = 'mocorr_concat_avg.tif'

# NOTE: trial_dff* is still a mean within a trial, but trialmean_dff* is a mean OF THOSE
# MEANS (across trials)
trial_dff_tiff_basename = 'trial_dff.tif'
trialmean_dff_tiff_basename = 'trialmean_dff.tif'
max_trialmean_dff_tiff_basename = 'max_trialmean_dff.tif'
min_trialmean_dff_tiff_basename = 'min_trialmean_dff.tif'

# TODO replace some use of this w/ diverging_cmap_kwargs?
# (e.g. in response matrix plots)
#cmap = 'plasma'
# to match remy
cmap = 'magma'

# TODO is there a 'gray_r'? try that?
# NOTE: 'Greys' is reversed wrt 'gray' (maybe not exactly, but it is white->black),
# and I wanted to try it to be more printer friendly, but at least without additional
# tweaking, it seemed a bit harder to use to see faint ROIs.
#anatomical_cmap = 'Greys'
anatomical_cmap = 'gray'

# started w/ 'RdBu_r', but Remy had wanted to change to 'vlag', which she said is
# supposed to be similar, but perceptually uniform
#
# TODO maybe still use 'vlag' for diagnostic ROI vs dF/F image plots (version i
# generated was that...)
#
# actually, not switching to 'vlag', to not give Remy more work regenerating old figs
diverging_cmap = plt.get_cmap('RdBu_r')

# since default set_bad seems to be ~white, which was fine for 'plasma' (which doesn't
# contain white), but now is not distinct from cmap midpoint (0.)
#
# lighter gray than 'gray', and according to Sam, B takes less issue with this color
diverging_cmap.set_bad((0.8, 0.8, 0.8))

# TODO actually set colors for stuff outside diverging_cmap range?
# (just need cmap.set_[over|under] and extend='both' (or 'min'|'max', if only want
# one of the two))

# TODO could try TwoSlopeNorm ([-0.5, 0] and [0, 2.0]?), but would probably want to
# define bounds per fly (or else compute in another pass / plot these after
# aggregating?)
# TODO rename to diverging_cmap_kws
diverging_cmap_kwargs = dict(
    cmap=diverging_cmap,

    # TODO TODO test default clip behavior of this is OK. (and contrast w/ e.g.
    # CenteredNorm, which actually has a clip=True/False kwarg, like most)
    # TODO TODO test what happens if diverging_cmap_kwargs used w/o vmin/vmax
    # specified (as diag_example_kws currently adds below) (want to get from data)
    # NOTE: specifying norm classes this way only works because some of my hong2p.viz
    # wrappers
    norm='two-slope',

    # TODO TODO want clip=True for diag_example_kws?
    # TODO TODO with clip=False (the default), is this colormap even clearly
    # showing over/under cmap range values distinctly (not sure this was ever that
    # related to clip, but pretty sure we need extend='both' set on colorbar creation
    # for this)?
)

# TODO still using this? delete?
# TODO replace some/all uses of these w/ my own diverging_cmap_kwargs?
# (since we aren't using my generated plots for any of these anyway...)
#
# NOTE: not actually sure she will switch to 'vlag' (from 'RdBu_r'), though I have
# in diverging_cmap.
remy_corr_matshow_kwargs = dict(cmap='RdBu_r', vmin=-1, vmax=1, fontsize=10.0)

dff_cbar_title = f'{dff_latex}'

# TODO better name
trial_stat_cbar_title = f'mean peak {dff_latex}'

diff_cbar_title = f'$\Delta$ mean peak {dff_latex}'

single_dff_image_row_figsize = (6.4, 1.6)

# TODO set on a per-fly basis based on some percetile of dF/F (could use plot_rois /
# image_grid)? maybe not for all... constant scale has been useful...
dff_vmin = -0.5
dff_vmax = 2.0

# for ORN example dF/F image in paper (not initial preprint):
# (2023-05-08/3, plane -10)
#dff_vmin = -0.5
#dff_vmax = 1.5

# for GH146 example dF/F images in paper (not initial preprint):
# (final pick was 2023-06-23/1, plane -20)
#dff_vmin = -0.5
#dff_vmax = 1.5

diag_example_plot_roi_kws = dict(
    # TODO is 2023-05-10/1 the example for paper? add comment saying so, if so.
    # (it's not for the dF/F images in the main figure [that has 1-5ol, 1-6ol, ep], but
    # it may be for the diagnostic examples?)
    #
    # in 2023-05-10/1/diagnostics*, no inhibition is more negative than ~ -.283 on
    # average (paa -5 in DC1). most is > -.20 (> -.15 even) the smallest positive dF/F
    # are ~0.3, with many/most > 0.8 and 4 > 1.5 (some ~2.0)
    vmin=-0.3, vmax=1.0,

    # for some of the the ORN/GH146 example dF/F images in paper (not initial preprint):
    #vmin=dff_vmin, vmax=dff_vmax,

    cbar_label=f'mean {dff_cbar_title}',

    # TODO just move fontsize=7 to plot_rois default now? want for other ones too...
    # TODO rename now that these aren't actually "titles"
    # TODO also specify x, y kwargs here (just to not rely on default inside hong2p.viz)
    depth_text_kws=dict(fontsize=7, as_overlay=True),
    # 6 is pretty good for these
    scalebar_kws = dict(font_properties=dict(size=6)),

    # TODO set scalebar / over-image-depth-info (/ other?) default colors (either white
    # or black, probably), based on whether ~0 is more white/black in colormap plot_rois
    # is using (try to do automatically inside plot_rois)

    # for suptitle. default is 'large', but not sure what that is in points...
    title_kws=dict(fontsize=8),

    # TODO delete / implement working version
    #smooth=True,

    # TODO TODO TODO + add outline around red ROI boundary there (for contrast vs red
    # cmap upper end) (or use diff means to highlight that ROI outline, other than
    # color)

    # TODO translate to singular in viz.plot_rois?
    # (+ be consistent w/ outline handling. maybe just centralize in plot_closed*)
    #
    # 'dotted' should be plot_closed_contours/plot_rois default
    linestyles='dashed',

    # https://stackoverflow.com/questions/12434426
    # both fail with: ValueError: Do not know how to convert <x> to dashes
    #linestyles=(0, (2., 20.)),
    #linestyles=(2., 20.),

    # TODO TODO check this is actually working
    # https://stackoverflow.com/questions/35099130
    # TODO TODO TODO is this not working?
    #dashes=(4, 20),
    #linestyles='dotted',

    # (1.2 was/is default in plot_closed_contours)
    # pretty good
    #linewidth=1.0,
    #linewidth=0.8,

    # to try to minimize jagedness seeming of non-smooth contour dashes (per B's
    # request, though just a hack to try to achieve similar)
    linewidth=0.6,

    # TODO revert to default (None work for that?)
    # 1.0 might have been too low
    #focus_roi_linewidth=1.1,
    focus_roi_linewidth=0.8,

    # trying everything black (now that cmap is blue<->red)
    color='k',
    # TODO may want black outline on this line too (like black outline on red text of
    # focus roi name)?
    #focus_roi_color='k',

    # TODO what's a good (colorblind friendly) color to use, for contrast wrt
    # blood-ish red cmap upper end? current bright red OK? just add black outline?
    # think i might leave it at red. i find it easy enough to see against other red, and
    # i'm assuming that isn't colorvision dependent.
    #focus_roi_color='green',
    # harder to see against often dim-red dF/F (/ white regions nearby)
    #focus_roi_color='orange',
    # TODO TODO TODO just draw small white outline around this / others? how?
    # TODO TODO TODO purple? which?
    # https://matplotlib.org/stable/gallery/color/named_colors.html
    #focus_roi_color='magenta',
    #focus_roi_color='deeppink',
    focus_roi_color='darkviolet',

    # B didn't want the name overlay.
    show_names=False,

    # TODO rename all this stuff to not refer to label specifically (or split into two
    # sets of kwargs, and use the other one). it also operates on ROI outline, and
    # that's what i'm using it for now.
    #
    # (should also be default now)
    label_outline=True,
    # TODO TODO why these defaults on first call? intended i think...
    # (having trouble reproducing... might be a bug in plot_closed_... kwarg handling,
    # or handling in callers... shouldn't certain_rois_on_avg.pdf (/similar) still use
    # those?)
    #
    # rois labelled:
    # default 0.75 seemed a bit much
    #
    # with_rois:
    # - too small: 0.6
    # - too much: 1.5
    # - pretty good: 1.2, 1.3 (verging on too much)
    # TODO do on focus ROI only?
    # TODO TODO possible to get this to go around each dash, and not just on sides?
    # NOTE: these two currently ignored if linestyle='dotted'
    label_outline_linewidth=1.2,
    # TODO switch between color based on whether using diverging cmap, like other stuff
    # (have fn for that in hong2p now, but currently requires imshow(...)/similar
    # already to have been called on Axes. may have something for that that takes cmap
    # too?)
    # default: 'black'
    label_outline_color='w',

    # TODO delete? was just for label right (not using anymore)?
    # TODO make fontweight not 'bold'? see defaults in plot_closed_contours
    # (no longer relevant, w/ show_names=False)
    text_kws=dict(fontsize=5),

    # if ROIs named 'AL' are in any planes, the area outside those ROIs (only in those
    # planes!) will be zeroed (to hide distracting background noise / motion artifacts,
    # when they are not informative)
    zero_outside_plane_outlines=True,

    # trying columns now. not enough space in rows (for >=8 planes at least, w/ all in
    # one row).
    nrows=None, ncols=1,

    # after ImageGrid, do still need some extra width (w/o bbox_inches, haven't tried
    # with).
    # TODO can bbox_inches[='tight'? something else?] also make fig larger to have
    # stuff not cut off (ylabels of axes/cax + suptitles, etc)? may not need
    # extra_figsize if so?
    # at extra width < ~0.9, cbar info is cut off (tho there is still space on left...)
    # TODO some way to get constrained layout to respect all colorbar contents?
    image_kws=dict(height=10.5, extra_figsize=(1.0, 0),
        # top, bottom, width, height of ImageGrid area of figure
        imagegrid_rect=(0.09, 0.01, 0.775, 0.955)
    ),

    **diverging_cmap_kwargs
)
# vmin/vmax (as in above) override this. otherwise, this fraction of data is excluded,
# and new extremes are used for vmin/vmax.
roi_background_minmax_clip_frac = 0.025

# for both ORN and GH146 example dF/F images in paper (not initial preprint):
# (no log scale on colormap)
#roi_background_minmax_clip_frac = 0.01

ax_fontsize = 7

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


def roi_label(index_dict):
    roi = index_dict['roi']
    if is_ijroi_named(roi):
        return roi
    # Don't want numbered ROIs to be grouped together in plots, as the numbers
    # aren't meanginful across flies.
    return ''


# TODO refactor inches_per_cell (+extra_figsize) to share w/ plot_roi_util?
# just move into plot_all_... default? (would need extra_figsize[1] == 1.0 to work here)
# TODO rename to clarify these aren't for plot_rois calls...
roi_plot_kws = dict(
    inches_per_cell=0.08,
    # TODO adjust based on whether we have extra text labels / title / etc?
    # 1.7 better for the single panel plots, 2.0 needed for ijrois_certain.png, etc
    extra_figsize=(2.0, 0.0),

    fontsize=4.5,

    linewidth=0.5,
    # TODO just set this in rcParams (or patching them at module level up top?)?
    dpi=300,

    # controls spacing from glomerulus names and <date>/<fly_num> IDs in yticklabels
    # (only relevant for plots w/ data from multiple flies).
    # default 0.12 was also OK here, just a bit much.
    #
    # was using this before trying to find a value for kiwi/control matrices
    #hgroup_label_offset=0.095,
    #
    # TODO TODO can one value work for both megamat/validation/etc (w/ many odors), and
    # just kiwi/control (when shown w/o diags)? or define dynamically based on number of
    # odors? layout method that is agnostic?
    # TODO this even doing anything? .15 was not clearly diff from .095
    hgroup_label_offset=0.5,

    # controls spacing between odor xticklabels and panel strings above them (for matrix
    # plots where we have data from multiple panels)
    #
    # though 0.08 works for roimean_plot_kws below (where flies are averaged over, and
    # thus less rows), since figure is so tall when there is a row per fly X glomerulus,
    # i think both axes and data coordinates have the problem of scaling with the height
    # of the figure, so this spacing would lead to a silly gap
    #
    # good for ijroi/certain.pdf, but didn't check much else
    # TODO also check ijroi/by_panel/megamat/with-diags_certain.pdf (may need to use
    # something like figure, rather than axes, coords; or special case the value for
    # diff plots w/ diff heights)
    vgroup_label_offset=0.0145,

    # TODO define separate ones for colorbar + title/ylabel (+ check colorbar one is
    # working)
    bigtext_fontsize_scaler=1.5,

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

roimean_plot_kws = dict(roi_plot_kws)
roimean_plot_kws['inches_per_cell'] = 0.15
roimean_plot_kws['extra_figsize'] = (1.0, 0.0)
roimean_plot_kws['vgroup_label_offset'] = 0.1

# starting from roimean_plot_kws b/c total amount of data will be more like that
single_fly_roi_plot_kws = dict(roimean_plot_kws)
# TODO remove levels_from_labels? don't think i should need to?
single_fly_roi_plot_kws = {k: v for k, v in single_fly_roi_plot_kws.items()
    if k not in ('hline_level_fn', 'hgroup_label_offset')
}


diag_panel_str = 'glomeruli_diagnostics'

# TODO delete. (these in my master gsheet? can i just mark for exclusion there?)
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

script_dir = Path(__file__).resolve().parent
# TODO refactor google sheet metadata handling so it doesn't download until it's needed
# (or at least not outside of __main__?)?
#
# TODO set bool_fillna_false=False (kwarg to gsheet_to_frame) and manually fix any
# unintentional NaN in these columns if I need to use the missing data for early
# diagnostic panels (w/o some of the odors only in newest set) for anything
#
# This file is intentionally not tracked in git, so you will need to create it and
# paste in the link to this Google Sheet as the sole contents of that file. The
# sheet is located on our drive at:
# 'Hong Lab documents/Tom - odor mixture experiments/tom_antennal_lobe_data'
#
# Sam has his own sheet following a similar format, as should any extra user of this
# pipeline.
gdf = util.gsheet_to_frame('metadata_gsheet_link.txt', normalize_col_names=True,
    # so that the .txt file can be found no matter where we run this code from
    # (hong2p defaults to checking current working dir and a hong2p root)
    extra_search_dirs=[script_dir]
)
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

# TODO clarify how these behave if something is missing (in comment)
panel2name_order = deepcopy(natmix.panel2name_order)
panel_order = list(natmix.panel_order)

# TODO any reason for this order (i think it might be same as order in yaml [which more
# or less goes from odors activating glomeruli in higher planes to lower planes], so
# maybe could load from there now?)? just use order loaded from config /
# glomeruli_diagnostics.yaml?
#
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

    # TODO was this working before? didn't i have my own abbrev as 6ol? or did something
    # else overwrite that?
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

# TODO get order from yaml files if not specified? per-panel flag to do this?
# TODO (as w/ other stuff), load from either generated or generator-input YAMLs
# (in this case, tom_olfactometer_configs/megamat0.yaml) (should have code for initial
# part of this now)
panel2name_order['megamat'] = [
    '2h', 'IaA', 'pa', '2-but', 'eb', 'ep', 'aa', 'va', 'B-cit', 'Lin', '6al',
    't2h', '1-8ol', '1-5ol', '1-6ol', 'benz', 'ms'
]
# Putting this before my 'control' panel, so that shared odors are plotted in correct
# order for this data
panel_order.insert(1, 'megamat')

panel_order.append('validation2')
# TODO TODO which of these did i actually want? check all odors match up in my data
# (maybe fix some of my abbrevs to match remy's?)
panel2name_order['validation2'] = [
    '+pul',
    'menth',
    'long',
    'sab',
    '-bCar',
    '-aPine',
    'euc',
    '2-mib',
    'geos',
    'guai',
    'mchav',
    'PEA',

    # NOTE: diff from remy's 'PAA'
    'paa',
    'PAA',

    'bbenz',
    'B-myr',
    'ger',

    # NOTE: diff from remy's '1-prop'
    '1-3ol',
    '1-prop',

    '1p3one',
    '1o3one',
    'EtOct',
    '2-mba',
    'Z4-7al',
]

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
also_ignore_if_ignore_existing_true = ('nonroi', 'ijroi', 'suite2p', 'dff2spiking',
    'model'
)
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
verbose = False

bootstrap_seed = 1337
check_outputs_unchanged = False

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

roi2best_plane_depth_list = []

# Using dict rather than defaultdict(list) so handling is more consistent in case when
# multiprocessing DictProxy overrides this.
names_and_concs2analysis_dirs = dict()

all_odor_str2target_glomeruli = dict()

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


megamat_odor_names = set(panel2name_order['megamat'])

def odor_is_megamat(odor: str) -> bool:
    """Takes odor str (with conc) like 'va @ -3' to whether it's among 17 megamat odors.
    """
    return olf.parse_odor_name(odor) in megamat_odor_names


# TODO also want to (and does it?) return DataFrame when input has single level?
def index_uniq(index: pd.Index, levels: Optional[List[str]] = None) -> pd.DataFrame:
    df = index.to_frame(index=False)

    if levels is not None:
        df = df[levels]

    return df.drop_duplicates()


def format_index_uniq(index: pd.Index, levels: Optional[List[str]] = None) -> str:
    return index_uniq(index, levels).to_string(index=False)


# TODO use everwhere this logic currently duped (including in load_antennal_csv)
# TODO factor to hong2p.util
def print_index_uniq(index: pd.Index, levels: Optional[List[str]] = None) -> None:
    print(format_index_uniq(index, levels))


def format_uniq(df: Union[pd.DataFrame, pd.Series], levels: Optional[List[str]] = None
    ) -> None:

    df = df.reset_index()

    if levels is not None:
        df = df[levels]

    return df.drop_duplicates().to_string(index=False)


# TODO use everwhere this logic currently duped (including in load_antennal_csv)
# TODO factor to hong2p.util
# TODO already have a type i can use instead of this union?
def print_uniq(df: Union[pd.DataFrame, pd.Series], levels: Optional[List[str]] = None
    ) -> None:
    print(format_uniq(df, levels))


# TODO factor to hong2p.viz
# TODO simpler way?
def rotate_xticklabels(ax, rotation=90):
    for x in ax.get_xticklabels():
        x.set_rotation(rotation)


# TODO move to hong2p.util or something?
def n_choose_2(n: int) -> int:
    ret = (n - 1) * n / 2
    assert np.isclose(ret, int(ret))
    return int(ret)


# TODO refactor to hong2p (replace util.melt_symmetric [used once in here])?
# TODO TODO test that a given pair will always be returned in a fixed order, so that no
# matter the set of odors we pass in here, we can average across outputs from diff calls
# to this fn. relevant for now loading old megamat data. not sure current implementation
# has this property... (as long as i'm always using ordered_pairs where needed, should
# be ok... kind of a hack though)
# TODO TODO add option to pass sequence of odors, to generate something like ordered
# pairs? (for convenience when averaging over groups w/ diff sets of odors/pairs)
def corr_triangular(corr_df, *, ordered_pairs=None):
    assert corr_df.index.equals(corr_df.columns)

    # TODO this causing difficulties later? alternatives?
    # (w/ needing to sort again)
    #
    # sorting index so that diff calls to this will produce mergeable outputs.
    # otherwise, though the set of pairs will be the same, some may be represented
    # in the other order in this index (and thus merging won't find a matching
    # pair).
    # TODO .loc still work w/ list(...)?
    pairs = list(itertools.combinations(corr_df.index.sort_values(), 2))

    corr_ser = corr_df.stack(dropna=False)

    if ordered_pairs is not None:
        # does fail in call from end of load_remy_2e...
        # (only happened to be true for my first use case. don't think it actually
        # matters)
        #assert set(ordered_pairs) - set(corr_ser.index) == set()

        pairs = [(b,a) if (b,a) in ordered_pairs else (a,b) for a,b in pairs]

    # itertools.combinations will not give us any combinations of an element with
    # itself. in other words, we won't be keeping the identity correlations.
    assert not any(a == b for a, b in pairs)

    # itertools.combinations essentially selects one triangular, excluding diagonal
    corr_ser = corr_ser.loc[pairs]

    # TODO switch to assertion(s) on input index/column names?
    # (just to fail sooner / be more clear)
    #
    # TODO make more general than assuming 'odor' prefix?
    assert len(corr_ser.index.names) == 2
    assert all(x.startswith('odor') for x in corr_ser.index.names)

    # TODO do 'a','b' instead? other suffix ('_row','_col')? (to not confused w/
    # 'odor1'/'odor2' used in many other MultiIndex levels in here, where 'odor2' is a
    # almost-never-used-anymore optional 2nd odor, where 2 delivered at same time
    # (most recently in kiwi/control 2-component ramp experiments).
    corr_ser.index.names = ['odor1', 'odor2']

    # TODO sort output so odors appear in same order as in input (within each component
    # of pair, at least)?

    return corr_ser


def invert_corr_triangular(corr_ser, diag_value=1., _index=None, name='odor'):
    if _index is None:
        for_odor_index = corr_ser.index
    else:
        for_odor_index = _index

    # TODO make more general than assuming 'odor' prefix?
    # TODO + factor to share w/ what corr_triangual sets by default (at least), in case
    # i change suffix added there
    #assert for_odor_index.names == ['odor1', 'odor2']
    # TODO rename all "odor" stuff to be more general (now that i'm not requiring
    # 'odor1'/'odor2')
    assert len(for_odor_index.names) == 2

    # unique values for odor1 and odor2 will not be same (each should have one value not
    # in other). could just sort, for purposes of generating one combined order.
    # for now, assuming (correctly, it seems) that first value in odor1 and last value
    # in odor2 are the only non shared, and that otherwise we want to keep the order
    #
    # pandas <Series>.unique() keeps order of input (assuming all are adjacent, at
    # least)
    odor1 = for_odor_index.get_level_values(0).unique()
    odor2 = for_odor_index.get_level_values(1).unique()

    # TODO TODO try to make work without these assertions (would need to change how
    # odor_indx is defined below). these seem to work if index is sorted, but not for
    # (at least some) indices before sort_index call.
    assert all(odor2[:-1] == odor1[1:])
    assert odor1[0] not in set(odor2)
    assert odor2[-1] not in set(odor1)

    # TODO maybe columns and index should have diff names? keep odor1/odor2?
    # TODO get shared prefix of cols for name=? accept as kwarg?
    odor_index = pd.Index(list(odor1) + [odor2[-1]], name=name)

    square_corr = pd.DataFrame(index=odor_index, columns=odor_index, data=float('nan'))
    for a in odor_index:
        for b in odor_index:
            if a == b:
                assert (a, b) not in corr_ser
                square_corr.at[a, b] = diag_value
                continue

            # TODO clean up
            try:
                if (a, b) in corr_ser:
                    assert (b, a) not in corr_ser
                    c = corr_ser.at[a, b]
                else:
                    assert (b, a) in corr_ser
                    c = corr_ser.at[b, a]
            except AssertionError:
                #print(f'{a=}')
                #print(f'{b=}')
                #import ipdb; ipdb.set_trace()
                c = float('nan')
            #

            square_corr.at[a, b] = c

    return square_corr


# TODO TODO use inside plot_corr, for its calc when input has multiple flies (/seeds)
# TODO add option to drop nonresponders (whether glomeruli or KCs), before computing?
# could only do stuff that is actually 0 (either via filling or in model KC responses)
# though, not just close to 0, or would have to decide how to threshold...
# TODO refactor part of internals out into a new fn that keeps id_cols metadata, but has
# corr_triangular pairs (as the 2 level MultiIndex it returns) on opposite (column) axis
# index (-> use that to calc mean -> invert_corr_triangular in here)
def mean_of_fly_corrs(df: pd.DataFrame, *, id_cols: Optional[List[str]] = None,
    square: bool = True) -> Union[pd.Series, pd.DataFrame]:
    """
    Args:
        df: DataFrame with odor level on row index, and levels from id_cols
            in column index.

        id_cols: column index levels, unique combinations of which identify individual
            flies (/ experimental units). if None, defaults to ['date', 'fly_num'].

        square: if True, returns square correlation matrix as a DataFrame. otherwise,
            returns one triangular (excluding diagonal) as a Series.
    """

    # TODO also allow selecting 'fly_id' as default, if there?
    if id_cols is None:
        id_cols = ['date', 'fly_num']

    # TODO TODO also work w/ 'odor' level (have in loaded model responses)
    # (or just expose as kwarg?)
    # TODO only do if 'repeat' level present? assert it's there?
    #
    # assumes 'odor2' level, if present, doesn't vary.
    # TODO assert assumption about possible 'odor2' level?
    trialmean_df = df.groupby(level='odor1', sort=False).mean()
    n_odors = len(trialmean_df)

    # TODO also want to  keep track of and append metadata?
    fly_corrs = []
    for fly, fly_df in trialmean_df.groupby(level=id_cols, axis='columns', sort=False):
        # TODO TODO pass in ordered_pairs? just expose as kwarg to this fn? (or fix
        # corr_triangular to always have a pair in a fixed order? even possible?)
        corr = corr_triangular(fly_df.T.corr())
        fly_corrs.append(corr)

    # TODO delete (just notes re-assuring myself this fn is working properly, when run
    # on megamat input)
    #
    # in case of Hallem subset of ORN data (when called from top-level modelling fn):
    # ipdb> trialmean_df.shape
    # (17, 140)
    #
    # ipdb> [num_null(x) for _, x in trialmean_df.groupby(level=id_cols,
    #     axis='columns', sort=False)]
    # [32, 30, 0, 0, 0, 0, 0, 0, 0]
    #
    # ipdb> [x.shape for _, x in trialmean_df.groupby(level=id_cols, axis='columns',
    #     sort=False)]
    # [(17, 16), (17, 15), (17, 16), (17, 15), (17, 15), (17, 16), (17, 16), (17, 16), (17, 15)]
    #
    # ipdb> [x.columns for _, x in trialmean_df.groupby(level=id_cols, axis='columns',
    #     sort=False)][0]
    # MultiIndex([('2023-04-22', 2,  'DC1'),
    #             ('2023-04-22', 2,  'DL1'),
    #             ('2023-04-22', 2,  'DL5'),
    #             ('2023-04-22', 2,  'DM2'),
    #             ('2023-04-22', 2,  'DM3'),
    #             ('2023-04-22', 2,  'DM4'),
    #             ('2023-04-22', 2,  'DM5'),
    #             ('2023-04-22', 2,  'DM6'),
    #             ('2023-04-22', 2,  'VA5'),
    #             ('2023-04-22', 2,  'VA6'),
    #             ('2023-04-22', 2,  'VC3'),
    #             ('2023-04-22', 2,  'VC4'),
    #             ('2023-04-22', 2,  'VM2'),
    #             ('2023-04-22', 2,  'VM3'),
    #             ('2023-04-22', 2, 'VM5d'),
    #             ('2023-04-22', 2, 'VM5v')],
    #            names=['date', 'fly_num', 'roi'])

    n_flies = len(fly_corrs)

    corrs = pd.concat(fly_corrs, axis='columns', verify_integrity=True)
    assert corrs.shape[1] == n_flies

    n_odors_choose_2 = n_choose_2(n_odors)
    # n choose 2. 136 for all non-identity combinations of 17 odors.
    assert len(corrs) == n_odors_choose_2

    # excludes NaN (e.g. va/aa in first 2 megamat flies)
    mean_corr_triangular = corrs.mean(axis='columns')
    assert mean_corr_triangular.shape == (n_odors_choose_2,)

    odor_order = trialmean_df.index
    assert not odor_order.duplicated().any()
    # TODO similar assertion to below, but against mean_corr_triangular?
    # (don't think i can...)

    if not square:
        return mean_corr_triangular

    mean_corr = invert_corr_triangular(mean_corr_triangular)
    assert mean_corr.shape == (n_odors, n_odors)

    assert set(odor_order) == set(mean_corr.index) == set(mean_corr.columns)

    # re-ordering odors to keep same order as input
    mean_corr = mean_corr.loc[odor_order, odor_order].copy()
    return mean_corr


# TODO more idiomatic way? i feel like there has to be...
def _index_set_where_true(ser):
    # TODO doc
    assert ser.dtype == bool

    # TODO what was `x[-1] if type(x) is tuple` special case for? relax to be more
    # general (for use in print_dataframe_diff)
    # x[-1] was to select 'odor1' values (assuming panel level first, which should
    # all be diag_panel_str if stuff actually differing anyway)
    # (and presumably other axis had a single level index where this was called? really
    # tho?)
    #ret = [ (x[-1] if type(x) is tuple else x) for x in ser.index[ser] ]
    # TODO still want to keep output as list? simplify?
    ret = [x for x in ser.index[ser]]

    # TODO why this base case and not an empty list? just so i can dropna easily to
    # remove all that stuff? add comment explaining
    if len(ret) == 0:
        return np.nan

    return ret


def _print_diff_series(ser: pd.Series, *, max_width: int = 125,  min_spaces: int = 2):
    # TODO doc

    index_strs = []
    for index in ser.index:

        # TODO delete
        '''
        # TODO at least check .names too (if keeping)
        if ser.index.name == 'roi':
            index_str = index
        else:
            # TODO TODO TODO fix (/remove)
            try:
                # just getting odor, assuming panel is always same
                assert index[0] == diag_panel_str
            except:
                print(f'{index=}')
                import ipdb; ipdb.set_trace()
            #
            # TODO TODO what was this to exclude? relax to support more general use in
            # print_dataframe_diff
            index_str = index[-1]
        '''
        # TODO maybe accept formatting fn as input? or levels to use?
        index_str = index

        index_strs.append(index_str)

    # TODO right base case?
    # need this (for now) so max(...) below won't fail for empty input
    if len(index_strs) == 0:
        return

    max_index_str_len = max(len(x) for x in index_strs)

    # TODO actually wrapping at width? seems to cut sooner... tmux doing something
    # weird?
    width = max_width - (max_index_str_len + min_spaces)
    # TODO some way to show only what doesn't differ when almost everything does
    # (would need access to extra info in here...)?
    # (e.g. when only 'aphe @ -5' matches for mdf/mcdf, in ['VA4', 'VL1'])
    ser = ser.map(lambda x: pformat(x, width=width, compact=True))

    # since series values are lists, they would have flanking '[' / ']' chars i don't
    # want
    ser = ser.str.strip('[]')

    # TODO delete? or at least also gate behind `if` below (only for odors tho?)?
    # removing "'" from str components (e.g. odor strs) of values
    ser = ser.str.replace("'", '')

    # TODO make more general (only apply to that one level, maybe pass in fns for
    # special cases like this)
    # TODO delete
    #if ser.index.name == 'roi':
    if 'roi' in ser.index.names:
        ser = ser.str.replace('@ ', '')

    indent_level = max_index_str_len + min_spaces

    # TODO provide arg for specifying levels we want to ignore (e.g. 'repeat', when we
    # expect that if any things differ for one repeat, they will for all, and we can add
    # ( an assertion to that effect)
    #
    # (or maybe do this where i calculate the diff series passed in to this fn?)
    #
    # (for now, assuming user will preprocess input to reduce across these, e.g. by
    # taking mean across repeats)

    for index_str, diff_list_str in zip(index_strs, ser):

        lines = diff_list_str.splitlines()
        lines = [(' ' * indent_level) + x.strip() for x in lines]
        # TODO delete
        #lines = [lines[0]] + [(' ' * indent_level) + x.strip() for x in lines[1:]]
        diff_list_str = '\n'.join(lines)

        print(f'{index_str}:\n{diff_list_str}')
        # TODO delete
        #n_spaces = indent_level - len(index_str)
        #spaces = ' ' * n_spaces
        #print(f'{index_str}{spaces}{diff_list_str}')
        #


def print_dataframe_diff(a: pd.DataFrame, b: pd.DataFrame) -> None:
    # TODO doc

    # TODO also handle special casing of odor formatting at top level like this, rather
    # than inside _print_diff_series
    #
    # so that i don't need to special case handling of these values inside fns i call
    # from here
    def _convert_date_levels_to_str(df: pd.DataFrame) -> pd.DataFrame:
        date_col = 'date'
        assert date_col not in df.index.names
        df = df.copy()

        names = df.columns.names
        if date_col in names:
            date_level_idx = names.index(date_col)
            df.columns = df.columns.set_levels(
                df.columns.levels[date_level_idx].map(format_date), level=date_level_idx
            )

        return df

    a = _convert_date_levels_to_str(a)
    b = _convert_date_levels_to_str(b)

    def _index_diff(x: pd.DataFrame, y: pd.DataFrame, *, axis='index') -> pd.Index:
        assert axis in ('index', 'columns')
        x_index = getattr(x, axis)
        y_index = getattr(y, axis)
        return x_index.difference(y_index)

    def _format_index(index) -> str:
        return index.to_frame(index=False).to_string(index=False)

    def _print_index(index) -> None:
        print(_format_index(index))


    def _print_index_diff(a: pd.DataFrame, b: pd.DataFrame, axis: str) -> None:

        a_only = _index_diff(a, b, axis=axis)
        if len(a_only) > 0:
            # TODO print like `diff` output (w/ '< ' prefix for left only stuff, and '>
            # ' prefix for right only stuff)?
            print(f'only in left {axis}:')
            _print_index(a_only)
            print()

        b_only = _index_diff(b, a, axis=axis)
        if len(b_only) > 0:
            print(f'only in right {axis}:')
            _print_index(b_only)
            print()


    # TODO want to assert row/col index level names/dtypes/etc same first?
    for axis in ('index', 'columns'):
        _print_index_diff(a, b, axis)

    fill_val = 0
    # TODO TODO modify to not need to fillna(0) (esp if either of these assertions
    # fails)
    assert not (a == fill_val).any().any()
    assert not (b == fill_val).any().any()
    # just to make easier to compare against (what used to be) NaNs
    a = a.fillna(fill_val)
    b = b.fillna(fill_val)

    # TODO TODO TODO intersection rows and cols, and reindex both dataframe to that
    # before continuing? (or otherwise implement to work w/o exactly equal row/col
    # indices)
    #import ipdb; ipdb.set_trace()
    diff = (a != b)

    diff_rows = diff.T.sum() > 0
    diff_cols = diff.sum() > 0

    n_diff_rows = diff_rows.sum()
    n_diff_cols = diff_cols.sum()

    # TODO also print how many total elements mismatch (and/or what fraction of total
    # elements)

    print_col_diff = n_diff_cols > 0
    print_row_diff = n_diff_rows > 0

    # b/c each conditional should be printing same differences, just from two diff
    # perspectives
    shorter_only = True
    if shorter_only:

        if n_diff_cols > n_diff_rows:
            print_col_diff = False

        elif n_diff_rows > n_diff_cols:
            print_row_diff = False

    if n_diff_cols > 0:
        rows_diff_per_col = diff.apply(_index_set_where_true).dropna()
        assert n_diff_cols == len(rows_diff_per_col)

        assert diff.sum()[diff_cols].equals(rows_diff_per_col.str.len())

        # TODO want a check like this? see comments where i copied this from. couldn't
        # always be asserted as-is
        #
        #    assert diff.loc[:, diff_cols].apply(_index_set_where_true).equals(
        #        diff.apply(_index_set_where_true).loc[diff_cols]
        #    )

        print()
        print(f'{n_diff_cols} non-matching columns')

        if print_col_diff:
            print('rows that differ for each non-matching column:')
            # TODO add option for this to show what the two values are (currently just
            # shows index values that mismatch)?
            _print_diff_series(rows_diff_per_col)

    if n_diff_rows > 0:
        cols_diff_per_row = diff.apply(_index_set_where_true, axis='columns').dropna()
        assert n_diff_rows == len(cols_diff_per_row)

        # TODO restore (some variant of this?). similar check on cols couldn't be used
        # in all cases in code this was copied from (so probably couldn't generally rely
        # on being true for rows either)
        #
        #assert diff.loc[diff_rows].apply(_index_set_where_true, axis='columns').equals(
        #    diff.apply(_index_set_where_true, axis='columns').loc[diff_rows]
        #)

        assert diff.T.sum()[diff_rows].equals(cols_diff_per_row.str.len())

        print()
        print(f'{n_diff_rows} non-matching rows')

        if print_row_diff:
            print('columns that differ for each non-matching row:')
            _print_diff_series(cols_diff_per_row)
        import ipdb; ipdb.set_trace()


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
# TODO olf.sort_odors allow specifying axis?
# TODO TODO TODO how to get this to not sort control panel '2h @ -5 + oct @ -3' (air
# mix, using odor2 level) right after '2h @ -5'? (and same for kiwi)
def sort_odors(df: pd.DataFrame, add_panel: Optional[str] = None, **kwargs
    ) -> pd.DataFrame:

    # TODO add to whichever axis has odor info automatically? or too complicated.
    # currently doesn't work if odors are in columns.
    if add_panel is not None:
        assert type(add_panel) is str
        # TODO assert 'panel' not already in (relevant axis) index level names
        # (or just ignore if it is, at least if already there)
        df = util.addlevel(df, 'panel', add_panel)

    return olf.sort_odors(df, panel_order=panel_order,
        # TODO what does if_panel_missing=None do? comment explaining (/ change code to
        # not require an explanation...)
        panel2name_order=panel2name_order, if_panel_missing=None, **kwargs
    )


# TODO flag to select whether ROI or (date, fly) take priority?
# TODO move to hong2p + test
def sort_fly_roi_cols(df: pd.DataFrame, flies_first: bool = False, sort_first_on=None
    ) -> pd.DataFrame:
    # TODO delete key if i can do w/o it (by always just sorting a second time when i
    # want some outer level)
    # TODO is doc for sort_first_on right (or is it maybe just describing one use case
    # for it?)?
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
    # TODO option to do certain instead of named?
    not_named = df.columns.get_level_values('roi').map(
        lambda x: not is_ijroi_named(x)).to_frame(index=False, name='not_named')

    levels_to_drop.append('not_named')
    to_concat.append(not_named)

    if sort_first_on is not None:
        # NOTE: for now, just gonna support this being of-same-length as df.columns

        # TODO delete try/except
        # triggered when trying to adapt each_fly diag resp matrix code to across fly
        # case
        try:
            assert len(sort_first_on) == len(df.columns)
        except AssertionError:
            print(f'{sort_first_on=}')
            print(f'{df.columns=}')
            print(f'{len(sort_first_on)=}')
            print(f'{len(df.columns)=}')
            import ipdb; ipdb.set_trace()

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

    # the order of level here determines the sort-priority of each level.
    sorted_df = df.sort_index(level=levels, sort_remaining=False, kind='stable',
        axis='columns').droplevel(levels_to_drop, axis='columns')

    # index_names were column names at input
    assert sorted_df.columns.names == index_names

    return sorted_df


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


def drop_nonconsensus_odors(df: pd.DataFrame, n_for_consensus: Optional[int] = None, *,
    verbose=True) -> pd.DataFrame:

    # TODO TODO finish (or make n_for_consensus non-optional)
    '''
    if n_for_consensus is None:
        n_flies = len(
            panel_and_diag_df.columns.to_frame(index=False)[['date','fly_num']
                ].drop_duplicates()
        )
        # >= half of flies (the flies w/ any certain ROIs, at least)
        n_for_consensus = int(np.ceil(n_flies / 2))
    '''
    # TODO refactor def of fly_cols (['date','fly_num'])
    #
    # if a fly has any glomeruli w/ non-NaN data an odor, that fly will
    # contribute one count in the sum.
    odor_counts = df.groupby(['date','fly_num'], axis='columns',
        sort=False).apply(lambda x: x.notna().any(axis='columns')
        ).sum(axis='columns')

    # TODO TODO move warning back out? (to retain `panel` part of msg)
    if verbose and (odor_counts < n_for_consensus).any():
        odor_counts.name = 'n_flies'

        msg = f'dropping odors seen <{n_for_consensus} (n_for_consensus) times:'
        #msg = f'dropping odors seen <{n_for_consensus} (n_for_consensus) times'
        #msg += f' in {panel=} flies:\n'
        msg += format_uniq(odor_counts[
                (odor_counts < n_for_consensus) & (odor_counts > 0)
            ], ['panel','odor1','n_flies']
        )
        msg += '\n'

        warn(msg)

    return df[odor_counts >= n_for_consensus].copy()


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
        # TODO fix for merging roi_best_plane_depths (handling that without this fn for
        # now)
        # values in that case:
        # ipdb> gdf
        # date                            2023-04-22
        # fly_num                                  2
        # thorimage_id                  diagnostics1 diagnostics2 diagnostics3 megamat1 megamat2
        # roi                                    DC1          DC1          DC1      DC1      DC1
        # panel                 is_pair
        # glomeruli_diagnostics False           10.0         10.0         20.0      NaN      NaN
        # megamat               False            NaN          NaN          NaN      0.0      0.0
        # validation2           False            NaN          NaN          NaN      NaN      NaN
        #
        # ipdb> gdf.notna().sum(axis='columns')
        # panel                  is_pair
        # glomeruli_diagnostics  False      3
        # megamat                False      2
        # validation2            False      0
        try:
            assert not (gdf.notna().sum(axis='columns') > 1).any()
        except AssertionError:
            import ipdb; ipdb.set_trace()

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

    # TODO TODO delete? this even an issue? maybe just print cases where there are
    # multiple 'roi' for a given 'name_as_if_certain'?
    #
    # TODO fix issue probably added (2023-10-29) by editing 2023-05-09/1 (or fly edited
    # before that?) (probably by 'x?' and 'x??' or something...)
    # Traceback (most recent call last):
    #   File "./al_analysis.py", line 10534, in <module>
    #     main()
    #   File "./al_analysis.py", line 10119, in main
    #     trial_df = drop_superfluous_uncertain_rois(trial_df)
    #   File "./al_analysis.py", line 1095, in drop_superfluous_uncertain_rois
    #     assert (
    # AssertionError
    # TODO TODO don't err in cases like this (where 2 ROIs map to None name_as_if_certain)
    #urois = fly_rois.groupby(['date','fly_num','name_as_if_certain'], dropna=False).roi.unique()
    #ipdb> urois[urois.str.len() > 1]
    #date        fly_num  name_as_if_certain
    #2023-10-19  2        NaN                   [D|DC3?, VC5|VM3?]
    try:
        # TODO TODO TODO actually print the specific ROI(s) that map to the same
        # name_as_if_certain
        assert (
            len(fly_rois[['date','fly_num','roi']].drop_duplicates()) ==
            len(fly_rois[['date','fly_num','name_as_if_certain']].drop_duplicates())
        )
    except AssertionError:
        # TODO in case like that w/ 'D|DC3?' and 'VC5|VM3?' in comment above (where
        # both map to None), are those dropped from output? say '(not dropped)' in this
        # warning if they should never be (pretty sure they aren't)
        # TODO is it only when the multiple map to None that they aren't dropped?
        # (shouldn't be)
        warn('drop_superfluous_uncertain_roi: some ROIs map to same name_as_if_certain'
            ' (not dropped)'
        )

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


# TODO factor to hong2p.olf?
# TODO rename to odor_metadata_from_input_yaml or something?
def load_olf_input_yaml(yaml_name: str, olf_config_dir: Optional[Path] = None,
    default_solvent: Optional[str] = 'pfo') -> pd.DataFrame:
    """
    Args:
        default_solvent: if None, NaN in 'solvent' column will not be filled in. else,
            will be filled in with this value (adding this column if necessary).
    """
    if olf_config_dir is None:
        olf_config_dir = Path.home() / 'src/tom_olfactometer_configs'

    input_yaml = olf_config_dir / yaml_name
    assert input_yaml.exists()

    yaml_data = yaml.safe_load(input_yaml.read_text())['odors']

    df = pd.DataFrame.from_records(yaml_data)

    if default_solvent is not None:
        solvent_col = 'solvent'

        if solvent_col not in df.columns:
            df[solvent_col] = np.nan

        df[solvent_col] = df[solvent_col].fillna(default_solvent)

    return df


# TODO TODO unit test all combinations of changed/unchanged w/ CSV / pickle / some plot
# formats
# TODO option to use np.isclose or something instead of exact file comparison?
# (mainly thinking for CSVs, where the specific way the computation is done might change
# and just lead to a non-important numerical change. hasn't been an issue so far though)
# TODO change so save_fn is the optional one (rather than data) and automatically use
# .savefig if that positional argument has that attribute (or is a Figure/FacetGrid?
# checking for savefig attr probably better...)
# TODO also support objects w/ .save(path) method? (e.g. statsmodels models)
# TODO option to touch files-that-would-be-unchanged to have mtime as if they were just
# written?
def _check_output_would_not_change(path: Path, save_fn, data=None, **kwargs) -> None:
    """Raises RuntimeError if output would change.

    Args:
        path: must already exist (raises IOError if not)
        *args, **kwargs: passed to `save_fn`
    """
    if not path.exists():
        raise IOError(f'{path} did not exist!')

    # TODO derive name in deterministic way from path, so that same input will always
    # overwrite any pre-existing temp outputs? not a huge priority, just some temp files
    # could get left around as-is (which is probably fine. generally get deleted on
    # reboot)
    #
    # TODO test w/ input where output filename doesn't already have suffix?
    # support that?
    # (if input doesn't have '.' in name, suffix is ''. if input name starts with '.',
    # and there isn't another '.', suffix is also '')
    #
    # need to use existing suffix when savefig is matplotlib Figure.savefig, as it will
    # save to `temp_file + '.png'` instead of `temp_file`.
    # path.suffix for e.g. Path('x/y/z.pdf') is '.pdf'.
    temp_file = NamedTemporaryFile(delete=False, suffix=path.suffix)
    # also includes directory
    temp_file_path = Path(temp_file.name)

    # TODO move some/all of use_mpl_comparison/is_pickle def to just before conditional
    # (as part of factoring out a file comparison fn from within this)
    use_mpl_comparison = False

    # for save_fn input like:
    # <bound method Figure.savefig of <Figure size 1920x1440 with 1 Axes>>
    # I wasn't actually seeing __name__ in dir(save_fn) for that, but accessing
    # .__name__ still worked (providing 'savefig'). __func__ / __self__ may be the
    # function and bound instance, but not sure, and this seems like it might work ok.
    #
    # TODO test w/ seaborn input
    if save_fn.__name__ == 'savefig':
        use_mpl_comparison = True

    is_pickle = False
    if path.suffix == '.p':
        # TODO could also check if save_fn is to_pickle (or has 'pickle' in name, or
        # name split on '_'?). i always name pickles w/ .p, but if anyone else uses this
        # fn (for pickles), might matter.
        # TODO more generally, check if there is a read_<x> fn defined in same scope as
        # input to_<x> fn name (-> use read fn to compare if filecmp fails). or accept
        # read fn as optional input?
        assert not use_mpl_comparison
        is_pickle = True

    # TODO set verbose=False if save_fn already had that?
    # assuming we don't need to for now. any other issues w/ calling wrapped
    # fn twice?
    # TODO possible to refactor to not need to check data (maybe using *args, and
    # changing all fns using produces_output to swap order of data and path args?)?
    if data is not None:
        save_fn(data, temp_file_path, **kwargs)
    else:
        # TODO assert use_mpl_comparison here (and only here?)? (should be only case i'm
        # currently not passing in data, and no real plans for that to change)
        save_fn(temp_file_path, **kwargs)

    temp_file.close()
    # TODO assert some bytes have been written to file?
    # (would have caught save_fn appending suffix issue before i was using existing
    # suffix)

    to_delete = []
    # TODO TODO also include mtime of existing file in message
    err_msg = f'{path} would have changed! (run without -c to ignore)'

    # TODO TODO TODO factor out file comparison fn from this
    # (-> use to check certain key outputs same as what is committed to repo, e.g.
    # CSVs w/ model responses in data/sent_to_anoop vs current ones)

    if not use_mpl_comparison:
        # https://stackoverflow.com/questions/1072569
        unchanged = filecmp.cmp(path, temp_file_path, shallow=False)

        # could also *always* do this comparison, instead of filecmp above, but I'm
        # assuming actually loading the pickles is more expensive than the above.
        # if the filecmp approach *usually* works for pickles (as it seems to), then
        # it's probably better to only load+compare the pickles if that check fails.
        if is_pickle and not unchanged:
            # TODO these read_pickle fns always return output consistent w/
            # pd.read_pickle (in only case checked so far, yes)?
            old = read_pickle(path)
            new = read_pickle(temp_file_path)

            if hasattr(old, 'equals'):
                unchanged = old.equals(new)
            else:
                unchanged = old == new
                try:
                    unchanged = bool(unchanged)

                # will trigger if old/new are like numpy arrays, like:
                # ValueError: The truth value of an array with more than one element is
                # ambiguous. Use a.any() or a.all()
                #
                # even though the `old.equals(new)` check above should be used for
                # pandas objects (and probably anything else that would have this type
                # of error...), a similarly worded error would be emitted if trying to
                # coerce pandas elementwise comparisons to a single bool. e.g.
                # ValueError: The truth value of a DataFrame is ambiguous. Use a.empty,
                # a.bool(), a.item(), a.any() or a.all().
                except ValueError as err:
                    # TODO also filter on err msg, only trying to coerce check w/ .all()
                    # if msg matches [some parts of] expected err msg?
                    unchanged = np.all(unchanged)
    else:
        # TODO TODO when factoring out file comparison fn, def use_mpl_comparison from
        # whether extension is in `mpl_compare.comparable_formats()`?
        # (currently ['png', 'pdf', 'eps', 'svg'])
        #
        # TODO can i remove this? what happens if compare_images gets an input w/ a
        # non-comparable format?
        assert path.suffix[1:].lower() in mpl_compare.comparable_formats()

        # TODO want to actually use tolerance > 0 ever? (reading mpl's code, if tol is
        # 0, np.array_equal is used, rather than mpl_compare.calculate_rms)
        tolerance = 0

        # compare_images seems to save a png alongside input image, and i don't think
        # there are options to not do that, so i'm just deleting those files after
        #
        # https://matplotlib.org/devdocs/api/testing_api.html#module-matplotlib.testing.compare
        #
        # from docs: "Return None if the images are equal within the given tolerance."
        diff_dict = mpl_compare.compare_images(path, str(temp_file_path), tolerance,
            # TODO remove this, since i couldn't really use it to find files to delete
            # anyway (since it's None if no diff)?
            in_decorator=True
        )

        # paths in diff_dict (for temp_file_path=/tmp/tmpdcsvhhzb.pdf) should look like:
        # 'actual': '/tmp/tmpdcsvhhzb_pdf.png',
        # 'diff': '/tmp/tmpdcsvhhzb_pdf-failed-diff.png',
        # 'expected': 'pebbled_6f/pdf/ijroi/mb_modeling/hist_hallem_pdf.png',

        # TODO try to move the to_delete handling to after this conditional
        # (so i can factor this whole conditional, w/ some of earlier stuff, into fn for
        # just comparing files, not doing any of the temp file creation / cleanup)

        # bit more flexible than Path.with_suffix
        def with_suffix(filepath, suffix):
            return filepath.parent / f'{filepath.stem}{suffix}'

        def temp_with_suffix(suffix):
            return with_suffix(temp_file_path, suffix)

        if plot_fmt != 'png':
            # don't think i need to worry about these two if plot_fmt == 'png'
            to_delete.extend([
                temp_with_suffix(f'_{plot_fmt}.png'),
                with_suffix(path, f'_{plot_fmt}.png')
            ])

        if diff_dict is None:
            # TODO delete
            # NOTE: i think this one is only created if comparison fails
            # (and we probably want to keep it then anyway)
            '''
            if plot_fmt != 'png':
                # TODO TODO is this one just <x>_failed-diff.png for path=<x>.png ?
                to_delete.append(temp_with_suffix(f'_{plot_fmt}-failed-diff.png'))
            '''
            #

            unchanged = True
        else:
            unchanged = False
            err_msg += ('\n\nmatplotlib.testing.compare.compare_images output:\n'
                f'{pformat(diff_dict)}\n\ndiff image kept for inspection'
            )
            # TODO assert diff_dict['actual'] is same as temp_file_path?

            # TODO also open up (xdg-open / whatever) temp png matplotlib wrote (the
            # diff image)?
            # (and yes, it does convert everything to PNG for this, named like
            # /tmp/tmpa3hi5nxy_pdf-failed-diff.png (for diff), /tmp/tmpa3hi5nxy_pdf.png)


    # want to leave this file around if changed, so we can compare to existing output
    if unchanged:
        to_delete.append(temp_file_path)
    else:
        err_msg += ('\n\ncompare with what would have been new output, saved at: '
            f'{temp_file_path}'
        )

    for temp_path in to_delete:
        assert temp_path.exists(), f'{temp_path=} did not exist!'
        temp_path.unlink()

    if unchanged:
        if verbose:
            print(f'{path} would be unchanged')

        return

    # TODO move temp file into place, if DOESN'T match (after warning) (would need new
    # value for -c, to warn instead of err. not sure i want)?
    # (also, would have to handle in callers)

    raise RuntimeError(err_msg)


# if one output would get written two twice in one run of this script.
# for most outputs, we only intend to write them once, and this indicates an error.
class MultipleSavesPerRunException(IOError):
    pass


# TODO TODO move to hong2p.util
# TODO unit test?
#
# TODO default verbose=None and try to use default of wrapped fn then
# (or True otherwise?)
# (still need to test behavior when wrapped fn has existing verbose kwarg)
# TODO make this an attribute of this/one of inner fns (rather than module level)?
_fn2seen_inputs = dict()
# TODO what is _fn for again? keep?
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
        # TODO delete *args (if assertion it's unused passes for a wihle)
        def wrapped_fn(data, path: Pathlike, *args, verbose=verbose, **kwargs):
            # TODO delete (probably delete *args in sig above if so)
            assert len(args) == 0
            #

            # TODO easy to check type of matching positional arg is Path/Pathlike
            # (if specified)?
            # see: https://stackoverflow.com/questions/71082545 for one way

            # TODO add option (for use during debugging) that checks outputs
            # have not changed since last run (to the extent the format allows it...)

            assert fn.__name__ in _fn2seen_inputs

            seen_inputs = _fn2seen_inputs[fn.__name__]

            # TODO want to have wrapper add a kwarg to disable this assertion?
            # TODO test w/ both Path and str (and mix)
            if path in seen_inputs:
                raise MultipleSavesPerRunException('would have overwritten output '
                    f' {path} (previously written elsewhere in this run)!'
                )

            seen_inputs.add(path)

            path = Path(path)

            if check_outputs_unchanged and path.exists():
                try:
                    # TODO wrapper kwarg (that would be added to some/all decorations)
                    # for disabling this for some calls? e.g. for formats where they can
                    # change for uninteresting reasons, like creation time
                    _check_output_would_not_change(path, fn, data, **kwargs)
                    return

                except RuntimeError:
                    raise

            # TODO test! (and test arg kwarg actually useable on wrapped fn, whether or
            # not already wrapped fn has this kwarg. can start by assuming it doesn't
            # have this kwarg tho...)!
            #
            # (have already manually tested cases where wrapped fns do not have existin
            # verbose= kwarg. just need to test case where wrapped fn DOES have existing
            # verbose= kwarg now.)
            if verbose:
                print(f'writing {path}')

            # TODO delete
            #fn(data, path, *args, **kwargs)
            fn(data, path, **kwargs)

        return wrapped_fn

    # TODO what is this for again?
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
# TODO add flag (maybe via changes to wrapper?) that allow overwriting same thing
# written already in current run
def to_pickle(data, path: Pathlike) -> None:
    path = Path(path)

    if isinstance(data, xr.DataArray):
        path = Path(path)
        # read via: pickle.loads(path.read_bytes())
        # (note lack of need to specify protocol)
        # just specifying protocol b/c docs say it is (sometimes?) much faster
        path.write_bytes(pickle.dumps(data, protocol=-1))
        return

    if hasattr(data, 'to_pickle'):
        # TODO maybe do this if instance DataFrame/Series, but otherwise fall back to
        # something like generic read_pickle?
        data.to_pickle(path)

    path.write_bytes(pickle.dumps(data))


# TODO move to hong2p?
def read_pickle(path: Pathlike):
    path = Path(path)
    return pickle.loads(path.read_bytes())


# TODO also use for saving modeling choices series i'm already writing (not the
# dF/F->spiking model. the downstream stuff)
def read_series_csv(csv: Pathlike, **kwargs) -> pd.Series:
    df = pd.read_csv(csv, header=None, index_col=0, **kwargs)
    assert df.shape[1] == 1
    ser = df.iloc[:, 0].copy()

    # these should both be autogenerated, and not actually in CSV
    assert ser.index.name == 0
    assert ser.name == 1
    ser.index.name = None
    ser.name = None

    return ser


# TODO check this behaves as verbose=True
# (esp if that fn already has verbose kwarg in natmix. want to test that case)
write_corr_dataarray = produces_output(_write_corr_dataarray)

# TODO TODO also use wrapper for TIFFs (w/ [what should be default] verbose=True)
# (wrap util.write_tiff, esp if that fn already has verbose kwarg. want to test that
# case)

exit_after_saving_fig_containing = None
# TODO CLI flag to (or just always?) warn if there are old figs in any/some of the dirs
# we saved figs in (would only want if very verbose...)?
# TODO maybe refactor to automatically prefix path with '{plot_fmt}/' in here?
#
# Especially running process_recording in parallel, the many-figures-open memory
# warning will get tripped at the default setting, hence `close=True`.
#
# TODO why Type[sns.axisgrid.Grid] instead of just sns.axisgrid.Grid?
# (is below comment why? i assume so...)
# sns.FacetGrid and sns.ClusterGrid should both be subclasses of sns.axisgrid.Grid
# (can't seem to import sns.ClusterGrid anyway... maybe it's somewhere else?)
# TODO TODO try to replace logic w/ this decorator
#@produces_output(verbose=False)
_savefig_seen_paths = set()
# TODO possible to use verbose kwarg alongside global verbose?
# (for using this fn in other scripts, would rather have kwarg, but also don't want to
# have to pass verbose=verbose everywhere in here, nor use global as default kwarg,
# which wouldn't be update by CLI arg correctly...)
def savefig(fig_or_seaborngrid: Union[Figure, Type[sns.axisgrid.Grid]],
    fig_dir: Pathlike, desc: str, *, close: bool = True, normalize_fname: bool = True,
    debug: bool = False, **kwargs) -> Path:

    global exit_after_saving_fig_containing

    if normalize_fname:
        prefix = util.to_filename(desc)
    else:
        # util.to_filename in branch above adds a '.' suffix by default
        prefix = f'{desc}.'

    # TODO also allow input to have extension (use that if passed)?
    # (meh, already exposed plot_fmt kwarg)
    # TODO actually modify to_filename to not throw out '.', and manually remove that in
    # any remaining cases where i didn't want it? for concentrations like '-3.5', this
    # makes them more confusing to read... (-> '-35')
    basename = prefix + plot_fmt

    # TODO delete / fix
    makedirs(fig_dir)
    #
    fig_path = Path(fig_dir) / basename

    # TODO share logic w/ to_csv above (meaning also want to resolve() in
    # produces_output? or no? already doing that?)
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
    # TODO delete
    # TODO TODO why is it trying to save this twice?
    if '/'.join(abs_fig_path.parts[-3:]) in (
        '2022-02-22_1_kiwi_ea_eb_only/ijroi/DL1.png',
        # TODO TODO fix this too. can repro by same command as above
        # (w/o need for `-i ijroi`)
        # other stuff also affected here, also (prob not exclusively):
        # corr_certain_only/kiwi/2022-03-31_1.png
        'corr_certain_only/kiwi/2022-03-30_1.png'
        ):

        print('SAVING ONE OF WHAT WILL BE A DUPLICATE FIGURE NAME')
        traceback.print_stack(file=sys.stdout)
        #import ipdb; ipdb.set_trace()

    #
    try:
        assert abs_fig_path not in _savefig_seen_paths
    except AssertionError:
        print(f'{abs_fig_path=}')
        print(f'{desc=}')
        # TODO TODO TODO fix:
        # no uncertain ROIs. not generating uncertain_by_max_resp fig
        # done
        # Warning: correlation shapes unequal (in plot_corrs input)! shapes->counts: {(33, 33): 10, (72, 72): 5}
        # some mean correlations will have more N than others!
        # PREFIX DRIVER/INDICATOR IF I CAN (WORTH PASSING? COMPUTE FROM DIR)
        # RESTORE CSV SAVING (AFTER DE-DUPING...)
        # writing pebbled_6f/pdf/ijroi/corr_certain_only/control_flies.csv
        # PREFIX DRIVER/INDICATOR IF I CAN (WORTH PASSING? COMPUTE FROM DIR)
        # RESTORE CSV SAVING (AFTER DE-DUPING...)
        # writing pebbled_6f/pdf/ijroi/corr_certain_only/kiwi_flies.csv
        # abs_fig_path=PosixPath('/home/tom/src/al_analysis/pebbled_6f/pdf/ijroi/corr_certain_only/kiwi/2024-09-03_1.pdf')
        # desc='2024-09-03_1'
        # > /home/tom/src/al_analysis/al_analysis.py(2581)savefig()
        # -> 2581     _savefig_seen_paths.add(abs_fig_path)
        # ipdb> u
        # > /home/tom/src/al_analysis/al_analysis.py(8313)plot_corrs()
        # -> 8313         fig_path = savefig(fig, panel_dir, fly_plot_prefix)
        import ipdb; ipdb.set_trace()
    #

    _savefig_seen_paths.add(abs_fig_path)

    _skip_saving = False
    # TODO delete if i manage to restore use of produces_output wrapper around savefig
    if check_outputs_unchanged and fig_path.exists():
        save_fn = fig_or_seaborngrid.savefig
        try:
            # TODO wrapper kwarg (that would be added to some/all decorations)
            # for disabling this for some calls? e.g. for formats where they can
            # change for uninteresting reasons, like creation time
            _check_output_would_not_change(fig_path, save_fn, **kwargs)
            _skip_saving = True

        except RuntimeError:
            raise
    #

    if save_figs and not _skip_saving:
        fig_or_seaborngrid.savefig(fig_path, **kwargs)

    fig = None
    if isinstance(fig_or_seaborngrid, Figure):
        fig = fig_or_seaborngrid

    elif isinstance(fig_or_seaborngrid, sns.axisgrid.Grid):
        fig = fig_or_seaborngrid.fig

    assert fig is not None

    # using global verbose flag, set True/False by CLI in main.
    if (verbose and not _skip_saving) or debug:
        # TODO may shorten to remove first two components of path by default
        # (<driver>_<indicator>/<plot_fmt> in most/all cases)?
        color = 'light_blue'
        cprint(fig_path, color)

    if (exit_after_saving_fig_containing and
        exit_after_saving_fig_containing in str(fig_path)):

        warn('exiting after saving fig matching '
            f"'{exit_after_saving_fig_containing}':\n{fig_path}"
        )
        sys.exit()

    # TODO warn if any figs are wider than would fit in a sheet of paper?
    # at least for a subset of figs marked as for publications?
    # (e.g. diagnostic example plot_rois fig i'm working on now)

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


# TODO rename after done (to match what it actually ends up doing)?
# TODO TODO exclude stuff not in gsheet / marked exclude in gsheet
# (or otherwise don't include stuff w/ less than all recordings for panel, e.g.
# 2023-12-24 w/ only one diagnostic recording)
def final_panel_concs(**paired_thor_dirs_kwargs):
    """Returns panel2final_conc_dict, panel2final_conc_start_time
    """
    # TODO TODO update to not have returned dict have None as a key ever
    # (seems to happen w/ some diag stuff, when run on sam's test data, but this stuff
    # seems to maybe be run on more than just sam's data there?
    # `./al_analysis.py -d sam -n 6f -v`)

    verbose = False

    # TODO move these type annotations to return annotation of this fn
    #
    # NOTE: only intended to work w/ panels that only have odors a single at one
    # concentration per recording (e.g. megamat, validation2, and [until the recent
    # exception w/ 2h, where I also used it at a higher conc for VA4]
    # glomeruli_diagnostics)
    panel2final_conc_dict: Dict[str, Dict[str, float]] = dict()
    # TODO use to limit when we use above to drop data (/delete)
    # (e.g. to not drop latest glomeruli_diagnostic data, where we won't have entries in
    # above?)
    panel2final_conc_start_time: Dict[str, datetime] = dict()

    last_recording_time = None
    last_date_and_fly = None
    # using None to indicate a certain fly should be excluded from calculating final
    # concs, for that panel
    curr_fly_panel2conc_dict: Dict[str, Optional[Dict[str, float]]] = dict()
    curr_fly_panel2start_time: Dict[str, datetime] = dict()
    seen_date_and_fly = set()

    def update_panel2final_conc_dict(panel2conc_dict, panel2start_time):
        for panel, conc_dict in panel2conc_dict.items():
            if conc_dict is None:
                # TODO warn?
                continue

            if panel not in panel2final_conc_dict:
                panel2final_conc_dict[panel] = conc_dict
                panel2final_conc_start_time[panel] = panel2start_time[panel]

            elif panel2final_conc_dict[panel] != conc_dict:
                # TODO TODO fix. hack to exclude stuff w/o all recordings, but
                # should probably use lack of presence in gsheet / exclusion mark /
                # comparison to YAML or something for that
                if len(conc_dict) < len(panel2final_conc_dict[panel]):
                    # TODO at least say from which fly / experiment (refactor?)?
                    warn('final_panel_concs: assuming shorter panel is incomplete. '
                        'ignoring.'
                    )
                    return
                #
                panel2final_conc_dict[panel] = conc_dict
                panel2final_conc_start_time[panel] = panel2start_time[panel]


    keys_and_paired_dirs = paired_thor_dirs(verbose=False, **paired_thor_dirs_kwargs)

    for (date, fly_num), (thorimage_dir, _) in keys_and_paired_dirs:

        try:
            yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(
                thorimage_dir
            )
        except NoStimulusFile as err:
            # TODO still do this if verbose
            #warn(f'{err}. skipping.')
            continue

        # TODO TODO TODO update to work w/ panel + in case where panel is split
        # across recordings (probably can't also support case where one experiment
        # has 2 concentrations for one odor..., e.g. in some late 2023 diagnostic
        # stuff, w/ i think 2h at 2 concs?)
        # TODO or maybe also group on target glomerulus (/ other odor metadata,
        # except abbrev), when available? or only consider concs to be old if that
        # that conc didn't appear in older experiments?
        panel = get_panel(thorimage_dir)

        recording_time = thor.get_thorimage_time(thorimage_dir)
        if last_recording_time is not None:
            assert recording_time > last_recording_time
        last_recording_time = recording_time

        if (date, fly_num) != last_date_and_fly:
            if last_date_and_fly is not None:
                # TODO change to warning?
                assert len(curr_fly_panel2conc_dict) > 0
                #
                update_panel2final_conc_dict(curr_fly_panel2conc_dict,
                    curr_fly_panel2start_time
                )

            # (since things should be iterated over in chronological order, and we don't
            # return to flies after starting recordings on another fly)
            assert (date, fly_num) not in seen_date_and_fly
            seen_date_and_fly.add((date, fly_num))

            curr_fly_panel2conc_dict = dict()
            curr_fly_panel2start_time = dict()

            last_date_and_fly = (date, fly_num)

        # NOTE: only currently planning to support experiments were only one odor is
        # presented at a time, for this.
        #
        # otherwise, would need a tuple -> conc for each (and would want to force an
        # order for each tuple), and i don't have a need to support that lately.
        if not all(len(x) == 1 for x in odor_lists):
            # TODO warn!
            continue

        if panel not in curr_fly_panel2conc_dict:
            curr_fly_panel2conc_dict[panel] = dict()
            curr_fly_panel2start_time[panel] = recording_time

        curr_panel_dict = curr_fly_panel2conc_dict[panel]

        if curr_panel_dict is None:
            # TODO warn (or just leave to first one, that sets this None?)?
            continue

        for odor_list in odor_lists:
            for odor in odor_list:
                name = odor['name']
                # TODO want/need to handle stuff w/o conc specified?
                # (don't think i need to for now)
                log10_conc = odor['log10_conc']

                # stuff that is redone could appear twice at the same conc, and
                # that's fine
                if name in curr_panel_dict and curr_panel_dict[name] != log10_conc:
                    warn(f'{shorten_path(thorimage_dir)}: {name} at >1 conc '
                        f'(at least {log10_conc} and {curr_panel_dict[name]})'
                    )
                    # TODO handle differently (store all concs?)
                    curr_fly_panel2conc_dict[panel] = None

                curr_panel_dict[name] = log10_conc

        # TODO how to handle case where an odor isn't seen after a certain point?
        #
        # might mean that i should skip a whole panel for a fly (that might be split
        # across recordings), rather than just skip one recording, so that it
        # doesn't seem like some odors fall into this category.

    update_panel2final_conc_dict(curr_fly_panel2conc_dict, curr_fly_panel2start_time)

    if verbose:
        for panel, conc_dict in panel2final_conc_dict.items():
            start_time = panel2final_conc_start_time[panel]
            print(f'{panel=}')
            print(format_time(start_time))
            print(f'{len(conc_dict)=}')
            pprint(conc_dict)

    # TODO delete? actually need start time dict?
    #return panel2final_conc_dict, panel2final_conc_start_time
    return panel2final_conc_dict


def odor_names2final_concs(**paired_thor_dirs_kwargs):
    """Returns dict of odor names tuple -> concentrations tuples + ...

    Loops over same directories as main analysis (so should be chronological)
    """
    keys_and_paired_dirs = paired_thor_dirs(verbose=False, **paired_thor_dirs_kwargs)

    seen_stimulus_yamls2thorimage_dirs = defaultdict(list)
    names2final_concs = dict()
    names_and_concs_tuples = []

    for (_, _), (thorimage_dir, _) in keys_and_paired_dirs:

        try:
            yaml_path, yaml_data, odor_lists = util.thorimage2yaml_info_and_odor_lists(
                thorimage_dir
            )
        except NoStimulusFile as err:
            # TODO still do this if verbose
            #warn(f'{err}. skipping.')
            continue

        seen_stimulus_yamls2thorimage_dirs[yaml_path].append(thorimage_dir)

        try:
            # NOTE: this fn does not work w/ pretty much any non-pair input, and will
            # likely raise AssertionError (may also raise that in some other cases, if
            # data doesn't match how i originally layed it out in pairgrid case...)
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
# TODO also accept a function to compute baseline / accept appropriate dimensional input
# [same as mean would be] to subtract directly?
# TODO test this baselining approach works w/ other dimensional inputs too
# TODO cache within a run?
# TODO type hint odor_index as Optional[Index]
#
def delta_f_over_f(movie_length_array, bounding_frames, *,
    n_volumes_for_baseline: Optional[int] = n_volumes_for_baseline,
    exclude_last_pre_odor_frame: bool = exclude_last_pre_odor_frame,
    one_baseline_per_odor: bool = one_baseline_per_odor, odor_index=None,
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

        one_baseline_per_odor: if False, baselines each trial to pre-odor period for
            that trial. if True (requires `odor_index != None`), baselines to pre-odor
            period of first trial of that odor (which may be a previous trial,
            potentially much before).

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

    if one_baseline_per_odor:
        # TODO warn if repeats for any odor are not all presented back-to-back?
        assert odor_index is not None
        assert len(odor_index) == len(bounding_frames)

        odor2frames = pd.DataFrame(index=odor_index, data=bounding_frames,
            columns=['start_frame', 'first_odor_frame', 'end_frame']
        )

        assert odor_index.names == ['odor1', 'odor2', 'repeat']
        odor2first_trial_frames = odor2frames.reset_index().drop_duplicates(
            subset=['odor1','odor2'], keep='first'
        )
        assert (odor2first_trial_frames.repeat == 0).all()
        odor2first_trial_frames = odor2first_trial_frames[
            [c for c in odor2first_trial_frames.columns if c != 'repeat']
        ]

        odor2first_trial_frames = odor2first_trial_frames.set_index(['odor1', 'odor2'],
            verify_integrity=True
        )

    for i, (start_frame, first_odor_frame, end_frame) in enumerate(bounding_frames):

        if not one_baseline_per_odor:
            # NOTE: this is the frame *AFTER* the last frame included in the baseline
            baseline_afterend_frame = first_odor_frame

            baseline_start_frame = start_frame
        else:
            odor1, odor2 = odor_index[i][:2]
            baseline_start_frame, baseline_afterend_frame = odor2first_trial_frames.loc[
                (odor1, odor2), ['start_frame', 'first_odor_frame']
            ]

        if n_volumes_for_baseline is not None:
            baseline_start_frame = baseline_afterend_frame - n_volumes_for_baseline
            assert baseline_start_frame < first_odor_frame

        if exclude_last_pre_odor_frame:
            baseline_afterend_frame -= 1

        for_baseline = movie_length_array[baseline_start_frame:baseline_afterend_frame]

        # TODO TODO median / similar maybe?
        # TODO explicitly mean over time dimension if input is xarray
        # (or specify all other dimensions, if that's how to make it work)
        baseline = for_baseline.mean(axis=0)

        # TODO support region defined off to side of movie (that should not have
        # signal), to use to subtract before any other calculations?

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
# TODO TODO homogenize stat (max here, mean elsewhere) behavior here vs in response
# volume calculation in process_recording (still an issue?)
# TODO TODO TODO compare results w/ old stat=max (using all volumes from onset to end of
# trial) on old data, and if nothing really gets worse (and it improves results on new
# PN data, as I expect), then stick with mean for everything
# (otherwise make driver -> settings dict or something, and only use for PNs)
def compute_trial_stats(traces, bounding_frames,
    odor_order_with_repeats: Optional[ExperimentOdors] = None, *,
    # TODO TODO TODO special case so it's mean by default for pebbled (to better capture
    # inhibition), and max by default for GH146 (b/c PN spontaneous activity. this make
    # sense? was it max and not mean that worked for me for GH146? maybe it was the
    # other way around?)
    # TODO TODO TODO check GH146 correlations again to see which looked better: max
    # or mean (and maybe doesn't matter on new data?)
    # TODO might need to change dF/F scale now that i'm going back to mean? check
    #stat=lambda x: np.max(x, axis=0),
    # TODO TODO double check all response matrices i've been sending to people were
    # using this (and document alongside those outputs)
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

    if odor_order_with_repeats is None:
        # TODO TODO TODO fail here if one_baseline_per_odor=True
        # (will fail below regardless)
        index = None
    else:
        index = odor_lists_to_multiindex(odor_order_with_repeats)

    # TODO return as pandas series if odor_order_with_repeats is passed, with odor
    # index containing that data? test this would also be somewhat natural in 2d/3d case

    trial_stats = []

    for trial_traces in delta_f_over_f(traces, bounding_frames, odor_index=index):
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

    # Adding Sam's panels, for testing analysis on some of his data.
    elif 'ban_2but' in thorimage_id.lower():
        return 'ban_2but'
    elif 'ban_solvent' in thorimage_id.lower():
        return 'ban_solvent'
    elif '2but_solvent' in thorimage_id.lower():
        return '2but_solvent'
    elif 'isoamylacetate_solvent' in thorimage_id.lower():
        return 'isoamylacetate_solvent'
    elif 'ban_purestrain_solvent' in thorimage_id.lower():
        return 'ban_purestrain_solvent'
    elif 'mono_screen_solvent' in thorimage_id.lower():
        return 'mono_screen_solvent'
    elif 'nat_screen_solvent' in thorimage_id.lower():
        return 'nat_screen_solvent'

    # TODO TODO handle old pair stuff too (panel='<name1>+<name2>' or something) + maybe
    # use get_panel to replace the old name1 + name2 means grouping by effectively panel

    else:
        return None


def ij_last_analysis_time(analysis_dir: Path):
    ij_trial_df_cache = analysis_dir / ij_trial_df_cache_basename
    if not ij_trial_df_cache.exists():
        return None

    return getmtime(ij_trial_df_cache)


# TODO maybe i should check for all of a minimum set of files, or just the mtime on
# the df caches, in case a partial run erroneously prevents future runs
# TODO refactor these so they (also?) return all files older than a certain mtime?  so
# that it's easier to decide warn in here, if files triggering a step-rerun are not
# generated in current run
def suite2p_outputs_mtime(analysis_dir, **kwargs):
    combined_dir = s2p.get_suite2p_combined_dir(analysis_dir)
    return util.most_recent_contained_file_mtime(combined_dir, **kwargs)


def suite2p_last_analysis_time(plot_dir, **kwargs):
    roi_plot_dir = suite2p_plot_dir(plot_dir)
    return util.most_recent_contained_file_mtime(roi_plot_dir, **kwargs)


def nonroi_last_analysis_time(plot_dir, **kwargs):
    # If we recursed, it would (among other things) visit the ImageJ/suite2p analysis
    # subdirectories, which may be updated more frequently.
    # TODO only check dF/F image / processed TIFF files / response volume cache files?
    # TODO TODO test that this is still accurate now that we are saving a lot of things
    # at root of what used to be plot dir / at same level (+ roi analyses might change
    # mtime of *something* in `plot_dir`, as it's defined here)

    # TODO TODO maybe what i really want here is the oldest time of a set of files
    # that should change whenever nonroi inputs (movie, whether raw/flipped/mocorr)
    # changes. fix! (and maybe also in other cases)
    # TODO should also be considering the TIFFs spit out under analysis dir
    return util.most_recent_contained_file_mtime(plot_dir, recurse=False, **kwargs)


# TODO include in a format_time, which also accepts datetime / Timestamp input?
# TODO move to hong2p.util
def format_mtime(mtime: float, year: bool = False) -> str:
    """Formats mtime like default `ls -l` output (e.g. 'Oct 11 18:24').
    """
    fstr = '%b %d %H:%M'
    if year:
        fstr += ' %Y'

    return time.strftime(fstr, time.localtime(mtime))


# TODO factor to a format_time fn (hong2p.util?)?
# TODO probably switch to just using one format str...
# TODO include seconds too?
def format_time(t):
    return f'{format_date(t)} {t.strftime("%H:%M")}'


def names2fname_prefix(name1, name2):
    return util.to_filename(f'{name1}_{name2}'.lower(), period=False)


def plot_corr(df: pd.DataFrame, plot_dir: Path, prefix: str, *, title: str = '',
    as_corr_dist: bool = False, verbose: bool = False, _save_kws=None, **kwargs
    ) -> pd.DataFrame:

    # otherwise, we assume input is already a correlation (/ difference of correlations)
    if not df.columns.equals(df.index):
        # TODO delete?
        if len(df.columns) == len(df.index):
            print('double check input is not already a correlation [diff]!')
            import ipdb; ipdb.set_trace()
        #

        # TODO TODO use new mean_of_fly_corrs instead (when appropriate, e.g. when input
        # has multiple flies [/ model seeds])?
        corr = df.corr()
    else:
        corr = df.copy()
        # to check not a corr dist input
        # TODO also check that range is consistent w/ corr and not corr-dist?
        #
        # TODO delete. won't work w/ corr_diff input i'm using in one place.
        # TODO TODO TODO fix (prob w/ new mean_of_... input)
        '''
        try:
            assert (corr.max() == 1).all()
        except AssertionError:
            import ipdb; ipdb.set_trace()
        '''

    if not as_corr_dist:
        vmin = -1.0
        vmax = 1.0
        vcenter = 0.0
        cmap = diverging_cmap
        to_plot = corr
        # TODO have cbar ticks be [-1, -0.5, 0, 0.5, 1.0] in this case
        # (in both cases i just want multiples of 0.5)
    else:
        vmin = 0.0
        vcenter = 1.0
        vmax = 2.0
        cmap = diverging_cmap.reversed()
        corr_dist = 1 - corr
        # should be this on the diagonals
        assert (corr_dist.min() == 0).all()
        to_plot = corr_dist
        # TODO have cbar ticks be [0, 0.5, 1, 1.5, 2] in this case
        # (in both cases i just want multiples of 0.5)

    # TODO TODO check that results w/ norm/vcenter not passed equiv to new results w/
    # norm='two-slope'+vcenter=0
    fig, _ = viz.matshow(to_plot, cmap=cmap, vmin=vmin, vcenter=vcenter, vmax=vmax,
        # just using 'two-slope', since 'centered' norm code would require
        # modification to get it to work w/ vcenter != 0.
        norm='two-slope', **kwargs
    )

    if len(title) > 0:
        fig.suptitle(title)

    if _save_kws is None:
        _save_kws = dict()

    # TODO any downside to hardcoding bbox_inches='tight'? was unspecified before
    savefig(fig, plot_dir, prefix, bbox_inches='tight', debug=verbose, **_save_kws)

    return corr


# TODO use/delete (and maybe refactor to include much/all of the kwargs used in
# plot_all...? see also mean_df/etc plotting in main, that recreates much of those
# kwargs for use w/ viz.matshow)
def plot_responses(df: pd.DataFrame, plot_dir: Path, prefix: str, *,
    vmin=None, vmax=None, title: str = '', _save_kws=None, **kwargs) -> None:

    fig, _ = viz.matshow(df, vmin=vmin, vmax=vmax, **diverging_cmap_kwargs, **kwargs)

    if len(title) > 0:
        fig.suptitle(title)

    if _save_kws is None:
        _save_kws = dict()

    # TODO any downside to hardcoding bbox_inches='tight'? was unspecified before
    savefig(fig, plot_dir, prefix, bbox_inches='tight', **_save_kws)


# TODO also break out a plot_responses fn from first part (-> use here)?
def plot_responses_and_corr(df: pd.DataFrame, plot_dir: Path, prefix: str, *,
    vmin=None, vmax=None, title: str = '', **kwargs) -> pd.DataFrame:

    # TODO expose bbox_inches (or remove kwarg in plot_[responses|corr], and just accept
    # the hardcode to bbox_inches='tight' in both of those, if no downside)? ever need
    # diff between the two calls?

    plot_responses(df, plot_dir, prefix, vmin=vmin, vmax=vmax, title=title, **kwargs)

    # TODO thread thru bbox_inches kwarg here (for savefig call)?
    return plot_corr(df, plot_dir, f'{prefix}_corr', title=title, **kwargs)


def count_n_per_odor_and_glom(df: pd.DataFrame, *, count_zero: bool = True
    ) -> pd.DataFrame:

    if not count_zero:
        # TODO need to do anything special to keep rows for things that would then be
        # fully NaN?
        #
        # since one call of this happens downstream of some 0-filling betty wanted, that
        # i don't think should count for this
        # TODO assert there are actually some 0.0 vals? or warn if not?
        df = df.replace(0.0, np.nan)
    else:
        if (df == 0.0).any().any():
            warn('count_n_per_odor_and_glom: have exact 0.0 data values. also counting '
                'them, though they may just be fill values. pass count_zero=False to '
                'exclude'
            )

    # TODO might need level= here, for sort=False to work as expected?
    # (didn't have it where i copied it from)
    n_per_odor_and_glom = df.notna().groupby(level='roi', sort=False,
        axis='columns').sum()

    return n_per_odor_and_glom


# TODO rename (either this or others) to be consistent about "plot_*" fns either saving
# outputs or not? use decorator to add save (option?) to plot fns that dont save
# (having all either just return fig, or at least having fig as first var returned?)?
def plot_n_per_odor_and_glom(df: pd.DataFrame, *, input_already_counts: bool = False,
    count_zero: bool = True, cmap: str = 'cividis', zero_color='white',
    title: bool = True, title_prefix='', **kwargs) -> Tuple[Figure, pd.DataFrame]:

    if not input_already_counts:
        n_per_odor_and_glom = count_n_per_odor_and_glom(df, count_zero=count_zero)
    else:
        n_per_odor_and_glom = df

    # TODO at least for panels below, show min N for each glomerulus?
    # (maybe as a separate single-column matshow w/ it's own colorbar?)
    # (only relevant in plots that take mean across flies)

    # TODO hong2p.viz tricks to set cmap max dynamically according to data?
    # possible? clean enough to be worth? would also want to intercept vmin/vmax, and
    # set the ticks in cbar_kws as i'm doing below...
    max_n = n_per_odor_and_glom.max().max()
    # discrete colormap: https://stackoverflow.com/questions/14777066
    cmap = plt.get_cmap(cmap, max_n)
    # want to display 0 as distinct (white rather than dark blue)
    cmap.set_under(zero_color)

    n_roi_plot_kws = dict(roimean_plot_kws)
    n_roi_plot_kws['cbar_label'] = 'number of flies (n)'

    if title:
        if len(title_prefix) > 0:
            title_prefix = f'{title_prefix}\n'

        # TODO de-dupe w/ cbar_label? just title_prefix?
        n_roi_plot_kws['title'] = \
            f'{title_prefix}sample size (n) per (glomerulus X odor)'
    else:
        assert len(title_prefix) == 0

    fig, _ = plot_all_roi_mean_responses(n_per_odor_and_glom, cmap=cmap,

        # TODO more elegant solution -> delete (detect whether cmap diverging inside
        # plot_all*?)
        use_diverging_cmap=False,

        # TODO why isn't 0 in the bar tho? if the data had 0, would there be?
        #
        # vmin has to be > 0, so that zero_color set correctly via cmap's set_under
        vmin=0.5, vmax=(max_n + 0.5),
        cbar_kws=dict(ticks=np.arange(1, max_n + 1)), **n_roi_plot_kws, **kwargs
    )

    # TODO delete
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

    # TODO need to transpose n_per_odor_and_glom?
    #
    # https://stackoverflow.com/questions/20998083
    for (i, j), n in np.ndenumerate(n_per_odor_and_glom):
        # TODO color visible enough? way to put white behind?
        # or just use some color distinguishable from whole colormap?
        ax.text(j, i, n, ha='center', va='center')
    '''

    return fig, n_per_odor_and_glom


# TODO how to specify defaults for vmin/vmax that override what add_norm_options
# does though (prob still want dff_imshow to use dff_v[min|max])?
# (currently just using this + wrapper below w/o '_' prefix for this)
@viz.add_norm_options
def _dff_imshow(dff_img, ax, **imshow_kwargs):
    im = ax.imshow(dff_img, **imshow_kwargs)

    # TODO figure out how do what this does EXCEPT i want to leave the xlabel / ylabel
    # (just each single str)
    ax.set_axis_off()

    # part but not all of what i want above
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])

    return im


def dff_imshow(dff_img, ax, *, vmin=dff_vmin, vmax=dff_vmax, **kwargs):
    return _dff_imshow(dff_img, ax, vmin=vmin, vmax=vmax, **kwargs)


# TODO rename now that i'm also allowing input w/o date/fly_num attributes?
def fly_roi_id(row: pd.Series, *, fly_only: bool = False) -> str:
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

        if hasattr(row, 'fly_id') and pd.notnull(row.fly_id):
            fly_id = row.fly_id
            assert type(fly_id) is str
            parts.append(fly_id)

        # TODO option to have these (or fly_id) parenthetical, and still show all 3
        # vars?
        else:
            if hasattr(row, 'date') and pd.notnull(row.date):
                date_str = f'{row.date:%-m-%d}'
                parts.append(date_str)

            if hasattr(row, 'fly_num') and pd.notnull(row.fly_num):
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
        assert not fly_only, "no fly ID vars ('fly_id' or ['date', 'fly_num'])"
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


# TODO delete odor_min_max_scale if i don't end up using
# TODO TODO rename this, and similar w/ 'roi_' (any others?), to exclude that?
# what else would i be plotting responses of? this is the main type of response i'd want
# to plot...
# TODO TODO always(/option to) break each of these into a number of plots such that we
# can always see the xticklabels (at the top), without having to scroll up?
# TODO add option to chop off odor concentrations in odor matshow xticklabels IF it's
# all the same on the input data? maybe default to that?
def plot_all_roi_mean_responses(trial_df: pd.DataFrame, title=None, roi_sort=True,
    sort_rois_first_on=None, odor_sort=True, keep_panels_separate=True,
    roi_min_max_scale=False, odor_min_max_scale=False,

    use_diverging_cmap: bool = True,

    # TODO delete hack!
    yticklabels=None,

    # TODO keep?
    avg_repeats: bool = True,

    single_fly: bool = False,
    odor_glomerulus_combos_to_highlight: Optional[List[Dict]] = None, **kwargs):
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

        odor_min_max_scale: if True, scales data within each odor to [0, 1].
            if `cbar_label` is in `kwargs`, will append '[0,1] scaled per odor'.

        odor_glomerulus_combos_to_highlight: list of dicts with 'odor' and 'glomerulus'
            keys. cells where `odor1` matches 'odor' (with no odor in `odor2`) and `roi`
            matches 'glomerulus' will have a red box drawn around them.

        **kwargs: passed thru to hong2p.viz.matshow

    """
    # TODO factor out this odor-index checking to hong2p.olf?
    # may also have 'panel', 'repeat', 'odor2', and arbitrary other metadata levels.
    assert 'odor1' in trial_df.index.names or 'odor1' == trial_df.index.name, \
        f'{trial_df.index=}'

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

    # TODO delete
    if trial_df.index.name == 'odor1':
        print("also support trial_df.index.name = 'odor1' (.names = None, right?)")
        print(f'{trial_df.index.names=}')
        #import ipdb; ipdb.set_trace()
        # TODO fix hack?
        print('ASSUMING INPUT IS MEAN ALREADY')
        mean_df = trial_df.copy()
    #
    else:
        avg_levels = [x for x in avg_levels if x in trial_df.index.names]

        if not avg_repeats:
            # TODO want this first assertion?
            assert 'repeat' in trial_df.index.names

            assert 'repeat' not in avg_levels
            avg_levels.append('repeat')

        # This will throw away any metadata in multiindex levels other than these two
        # (so can't just add metadata once at beginning and have it propate through
        # here, without extra work at least)
        mean_df = trial_df.groupby(avg_levels, sort=False).mean()

        # TODO assertion of the set of remaining levels (besides avg_levels) there are?
        # it should just be 'repeat' and stuff that shouldn't vary / matter, right?
        # (at most. some input probably doesn't have even that?)

        if not avg_repeats:
            # TODO want this assertion?
            assert 'repeat' in mean_df.index.names

    # TODO might wanna drop 'panel' level after mean in keep_panels_separate case, so
    # that we don't get the format_mix_from_strs warning about other levels (or just
    # delete that warning...) (still relevant?)

    if roi_min_max_scale:
        assert not odor_min_max_scale

        # TODO may need to check vmin/vmax aren't in kwargs and change if so

        # The .min()/.max() functions should return Series where index elements are ROI
        # labels (or at least it won't be the odor axis based on above assertions...).
        # equivalent to mean_df.[min|max](axis='rows')
        mean_df -= mean_df.min()
        mean_df /= mean_df.max()

        assert np.isclose(mean_df.min().min(), 0)
        assert np.isclose(mean_df.max().max(), 1)

        # TODO set this as full title if not in kwargs?
        if 'cbar_label' in kwargs:
            # (won't modify input)
            kwargs['cbar_label'] += '\n[0,1] scaled per ROI'

    if odor_min_max_scale:
        mean_df = mean_df.T.copy()
        # I tried passing axis='columns' (without transposing first), but then the
        # subtracting didn't seem able to align (nor would other ops, probably)
        mean_df -= mean_df.min()
        mean_df /= mean_df.max()

        mean_df = mean_df.T.copy()

        assert np.isclose(mean_df.min().min(), 0)
        assert np.isclose(mean_df.max().max(), 1)

        if 'cbar_label' in kwargs:
            kwargs['cbar_label'] += '\n[0,1] scaled per odor'

    if odor_sort:
        mean_df = sort_concs(mean_df)

    # TODO TODO also add option to fillna, adding rows until the rows match (or at least
    # include) all the hemibrain glomeruli? maybe sort those not in ANY of my data down
    # below (though would be more complicated...)?
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

    # TODO delete hack?
    if yticklabels is None:
        if not single_fly:
            # (assuming it's a valid callable if so)
            if 'hline_level_fn' in kwargs and not kwargs.get('levels_from_labels', True):
                if all([x in trial_df.columns.names for x in fly_keys]):
                    # TODO maybe still check if there is >1 fly too (esp if this path
                    # produces bad looking figures in that case)

                    # will show the ROI label only once for each group of rows where the
                    hline_group_text = True

            # TODO allow overriding w/ kwarg for case where i wanna call this w/ single
            # fly data? (to make diag examples, but calling as part of
            # acrossfly_response_matrix_plots)
            yticklabels = lambda x: fly_roi_id(x, fly_only=hline_group_text)
        else:
            # TODO factor out to a is_single_fly check or something?
            fly_cols = ['date', 'fly_num']
            if all(x in trial_df.columns for x in fly_cols):
                n_flies = len(
                    trial_df.columns.to_frame(index=False)[fly_cols].drop_duplicates()
                )
                assert n_flies == 1
            else:
                assert not any(x in trial_df.columns for x in fly_cols)
            #
            yticklabels = lambda x: x.roi

    vline_group_text = kwargs.pop('vline_group_text', 'panel' in trial_df.index.names)

    mean_df = mean_df.T

    # TODO maybe put lines between levels of sortkey if int (e.g. 'iplane')
    # (and also show on plot as second label above/below roi labels?)

    if roi_min_max_scale or odor_min_max_scale:
        # TODO TODO TODO change [h|vline]s to black in this case
        use_diverging_cmap = False
        # TODO assert no norm / diverging cmap in kwargs?
        if cmap not in kwargs:
            kwargs['cmap'] = cmap

    # TODO detect (using viz.is_diverging_cmap?)?
    if use_diverging_cmap:
        kwargs = {**diverging_cmap_kwargs, **kwargs}
        # center of diverging cmap should be white, so we'll use black lines here
        kwargs['linecolor'] = 'k'

    fig, _ = viz.matshow(mean_df, title=title, xticklabels=xticklabels,
        yticklabels=yticklabels, hline_group_text=hline_group_text,
        vline_group_text=vline_group_text, **kwargs
    )

    if odor_glomerulus_combos_to_highlight is not None:
        # colorbar should be fig.axes[1] if it's there at all
        ax = fig.axes[0]

        # this seems to be default colorbar label. for other ax (one i want), default
        # label seems to be '' here.
        assert '<colorbar>' != ax.get_label()

        for combo in odor_glomerulus_combos_to_highlight:
            odor1 = combo['odor']
            roi = combo['glomerulus']

            matching_roi = mean_df.index.get_level_values('roi') == roi
            matching_odor = (
                (mean_df.columns.get_level_values('odor1') == odor1) &
                (mean_df.columns.get_level_values('odor2') == solvent_str)
            )
            if matching_roi.sum() == 0 or matching_odor.sum() == 0:
                continue

            # TODO TODO if there are a few adjacent, find outer edge and just draw one
            # rect?
            # (for highlighting same on plots that have multiple fly data)
            #
            # other cases not currently supported (would have to think about handling)

            # should be fine to ignore / delete, or significantly weaken
            # TODO TODO not true actually, as i'm currently only drawing box around
            # FIRST matching index pair
            #'''
            assert matching_odor.sum() == 1
            # TODO delete try/except
            try:
                assert matching_roi.sum() == 1
            except AssertionError:
                # TODO be more descriptive (say which plot(s) affected) in warning
                warn(f'{matching_roi.sum()=} > 1. disabling box drawing!')
                continue
                #import ipdb; ipdb.set_trace()
            #'''

            # these will get index of first True value
            odor_index = np.argmax(matching_odor)
            roi_index = np.argmax(matching_roi)

            # TODO possible to compute good value for this tho (for flush w/ edge of
            # cell)?
            # since the rect (+ path effect) extend a bit beyond each cell, and it looks
            # kinda bad. 0.05 seems to produce good results (w/ linewidth=1, or
            # linewith=0.5 + patheffects w/ lw=1.0).
            # TODO decrease shrink slightly? some (but--for some reason--not all) boxes
            # seem to show a tiny bit of underlying color on right edge. seemed to
            # happen more on the bright yellow ones. not sure it's consistent...
            # (may only be an issue w/ png too?)
            shrink = 0.05

            # https://stackoverflow.com/questions/37435369
            # -0.5 seems needed for both in order to center box on each matshow cell
            anchor = (odor_index - 0.5 + shrink, roi_index - 0.5 + shrink)
            box_size = 1 - 2 * shrink
            # linewidth: 0.75 bit too much
            rect = patches.Rectangle(anchor, box_size, box_size, facecolor='none',

                # TODO try something other than white/red (that doesn't need PathEffect
                # black outline maybe?) (red pretty bad on magma cmap i'm using).
                # dotted (first try was pretty bad)?
                #
                # don't like w/ edgecolor='r', lw=0.4, PathEffect lw=1.0
                # OK (w/ edgecolor='w', lw=0.5, PathEffect lw=1.0). lw=0.3 prob too low.
                #edgecolor='w', linewidth=0.4,
                #path_effects=[
                #    # w/ linewidth=1 above: 0.75 too little, 2.0 a bit too much
                #    # w/ linewidth=0.75 above: 1.5 OK, but maybe highlights that either
                #    # rect or path effect is offset very slightly (~1px)?
                #    PathEffects.withStroke(linewidth=1.0, foreground='black')
                #],

                # OK (try a brighter green? might need path effect then...)
                #edgecolor='g', linewidth=1.0,

                # OK. maybe my fav so far?
                #edgecolor='k', linewidth=1.0,

                # OK. bit too close to yellow (too light) maybe?.
                # gray ('1.0' = white, '0.0' = black)
                #edgecolor='0.6', linewidth=1.0,

                # among my favorites.
                #edgecolor='0.4', linewidth=1.0,

                # TODO restore
                # think i'll stick with this one for now
                edgecolor='0.5', linewidth=1.0,

                # bad. at least with the linewidth=1.0 (too much) and not densely dotted
                #edgecolor='k', linewidth=1.0, linestyle='dotted',
                # cyan?
            )
            ax.add_patch(rect)

    # TODO just mean across trials right? do i actually use this anywhere?
    # would probably make more sense to just recompute, considering how often i find
    # myself writing `fig, _ = plot...`
    return fig, mean_df


# TODO delete odor_scaled_version kwarg if i don't end up using
def plot_responses_and_scaled_versions(df: pd.DataFrame, plot_dir: Path,
    fname_prefix: str, *, odor_scaled_version=False, bbox_inches=None,
    vmin=None, vmax=None, cbar_kws=None,

    # TODO TODO fix sort_rois_first_on handling in stddev case ->
    # delete this option (which was added just to special case fix that)
    # (or have this option replace that code?)
    sort_glomeruli_with_diags_first=False,

    **kwargs) -> None:
    """Saves response matrix plots to <plot_dir>/<fname_prefix>[_normed].<plot_fmt>

    Args:
        vmin: only used for non-[roi|odor]-scaled versions of plots

        vmax: only used for non-[roi|odor]-scaled versions of plots

        bbox_inches: None is also matplotlib savefig default

        **kwargs: passed to `plot_all_roi_mean_responses`
    """
    if any(x == 0 for x in df.shape):
        warn('plot_responses_and_scaled_versions: input empty for '
            f'{plot_dir=}, {fname_prefix=}. can not generate plots!'
        )
        return

    fig, _ = plot_all_roi_mean_responses(df, vmin=vmin, vmax=vmax, cbar_kws=cbar_kws,
        **kwargs
    )
    savefig(fig, plot_dir, f'{fname_prefix}', bbox_inches=bbox_inches)

    fig, _ = plot_all_roi_mean_responses(df, roi_min_max_scale=True, **kwargs)
    savefig(fig, plot_dir, f'{fname_prefix}_normed', bbox_inches=bbox_inches)

    # TODO TODO and a scaled-within-each-fly version?

    # TODO delete? not sure i want this
    # TODO or maybe just move out to the one place i might want this (diagnostics
    # response matrix example for now)
    if odor_scaled_version:
        fig, _ = plot_all_roi_mean_responses(df, odor_min_max_scale=True, **kwargs)
        savefig(fig, plot_dir, f'{fname_prefix}_odor-normed', bbox_inches=bbox_inches)

    # TODO delete?
    # don't want this for the stddev plots
    kwargs.pop('xticks_also_on_bottom', None)

    # TODO TODO TODO delete hack
    if 'repeat' not in df.index.names:
        print('PLOT_RESPONSES_AND_SCALED_VERSIONS RETURNING EARLY!')
        return
    #

    assert 'repeat' in df.index.names
    names_before = set(df.index.names)
    # taking mean across trials first
    fly_means = df.groupby([x for x in df.index.names if x != 'repeat'],
        sort=False
    ).mean()

    assert names_before - set(fly_means.index.names) == {'repeat'}
    del names_before

    stddev = fly_means.groupby('roi', axis='columns', sort=False).std()

    fly_cols = ['date', 'fly_num']
    if not all(x in df.columns.names for x in fly_cols):
        # or either way, can't compute stddev across flies
        n_flies = 1
    else:
        # TODO refactor to share w/ response_matrix plots (adapted from there)
        n_flies = len(df.columns.to_frame(index=False)[fly_cols].drop_duplicates())

    # TODO only compute+save if >2? >3?
    if n_flies <= 1:
        return

    # TODO fix / refactor sort_rois_first_on / diag-glomeruli-first sorting -> delete
    # this code (copied from calling place that calculates sort_rois_first_on)
    # (only broken in stddev case, cause shape changes when it averages across flies)
    glomeruli_with_diags = set(all_odor_str2target_glomeruli.values())
    def glom_has_diag(index_dict):
        glom = index_dict['roi']
        return glom in glomeruli_with_diags
    #

    # TODO check this isn't screwing up some other plots (diag, uncertain, etc)?
    # NOTE: updating kwargs w/ roimean_plot_kws is a hack to try to get this behaving
    # like mean plots (e.g. ijroi/certain_mean.pdf) which also use roimean_plot_kws
    #
    # intended to overwrite any keys in kwargs with value from roimean_plot_kws
    kwargs = {**kwargs, **roimean_plot_kws}

    if sort_glomeruli_with_diags_first:
        assert 'sort_rois_first_on' in  kwargs
        # replacing with recalulated version that should have right shape
        kwargs['sort_rois_first_on'] = [x not in glomeruli_with_diags
            for x in stddev.columns.get_level_values('roi')
        ]
        # TODO warn if overwriting? should prob just refactor anyway
        kwargs['hline_level_fn'] = glom_has_diag

        # TODO why (here and in normal sort_rois_first_on stuff above)
        # is VM5d sorted right before VM7d (it's b/c 2h appearing twice in that
        # diagnostic data, at 2 concs for 2 diff targets), but otherwise all diagnostics
        # are ordered correctly?  (related to 2h appearing twice [for va4 at -3 in new
        # data])

    # TODO maybe handle case where it's not in kwargs (just set to 'std' or something?)
    assert 'cbar_label' in kwargs
    kwargs['cbar_label'] = f'standard deviation of {kwargs["cbar_label"]}'

    # (currently just passing in consensus_df instead of trial_df in only
    # acrossfly_response_matrix_plots call)
    #
    # TODO drop ROIs w/ less than enough flies to compute (e.g. VM1)
    # (or at least display as NaN rather than 0)
    # or just only pass / pass a version of) input w/ that stuff dropped?
    # want to line up mean vs stddev anyway...

    fig, _ = plot_all_roi_mean_responses(stddev, vmin=0, vmax=vmax, **kwargs)
    savefig(fig, plot_dir, f'{fname_prefix}_stddev', bbox_inches=bbox_inches)


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

    # TODO also try passing input of (another call to?) compute_trial_stats to
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
def plot_rois(*args, nrows=1, cmap=anatomical_cmap, **kwargs):
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

    # TODO select between two defaults for minmax_clip_frac depending on whether cmap is
    # anatomical cmap? or define in groups of kws outside? prob want ~0.025 for dF/F
    # backgrounds and 0.001 or so for anatomical ones?
    # (maybe it was just that specific GH146 example fly, 2023-07-28/1, that i wanted
    # 0.001 for tho?)
    if not any(x in kwargs for x in ('vmin', 'vmax')):
        # TODO is this used for both avg and dF/F based backgrounds? i would assume so?
        #
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
        if 'color' in kwargs:
            palette = None
        else:
            # TODO TODO just let viz.plot_rois handle this? should be same default
            # (unless it matters that i pass in a colormap rather than a str, so there
            # are 2 nested color_palette calls)
            palette = sns.color_palette('hls', n_colors=10)
    else:
        palette = kwargs['palette']

    # TODO TODO why is color seed apparently not working (should be constant default of
    # 0 inside plot_rois)? (2023-12-18: still seems to not be working)
    # TODO did i actually need _pad=False? comment why (hong2p.plot_rois default is
    # False, at least now....)
    return viz.plot_rois(*args, nrows=nrows, cmap=cmap, _pad=False,
        # TODO maybe switch to something where it warns at least? does it now?
        # (plot_closed_contours default for this is 'err')
        if_multiple='ignore',

        # TODO TODO inside hong2p.viz.plot_rois, warn if this clipping happens
        # (just like other code in here that does regarding dff_vp[min|max])?
        minmax_clip_frac=minmax_clip_frac,

        # TODO experiment w/ only showing on one? which is more annoying, not being able
        # to pick which odor shows it, or having to break apart groups and manually
        # delete for most?
        # TODO set these to 10uM by default (to be consistent w/ examples in fig 5)?
        # (currently 25uM). see new add_colorbar code added in this file.
        scalebar=True,

        # Without n_colors=10, there will be too many colors to meaninigfully tell them
        # all apart, so might as well focus on cycling a smaller set of more distinct
        # colors (using randomization now, but maybe graph coloring if I really want to
        # avoid having neighbors w/ similar colors).
        # TODO just move this default into hong2p.viz.plot_rois...?
        # for most of these?
        palette=palette,
        # TODO delete. didn't like as much as hls10
        # 'bright' will only produce <=10 colors no matter n_colors
        # This will get rid of the gray color, where (r == g == b), giving us 9 colors.
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
def ij_traces(analysis_dir: Path, movie, bounding_frames, roi_plots=False,
    # TODO make this positional now? or at least properly handle if missing
    odor_lists=None):

    thorimage_dir = analysis2thorimage_dir(analysis_dir)
    try:
        full_rois = ijroi_masks(analysis_dir, thorimage_dir)

    except IOError:
        raise

    if not verbose:
        # (if verbose=True passed to rois2best_planes_only, it has a lot of output)
        #
        # TODO even want this? reword to say calculating traces?
        print('picking best plane for each ROI')

    # TODO delete?
    #
    # without full-plane outlines (drawn around entire AL, minus nerve/commissure, in
    # each plane)
    # TODO TODO try to replace other subsetting re: plane outline (below) with this
    # TODO TODO or still extract all -> subset traces / best_plane_rois similarly right
    # after?
    #
    # TODO still extract traces for plane outlines, for separate analyses (like what
    # betty wanted sam to do)?
    '''
    full_rois_no_outlines = full_rois.sel(
        roi=[not is_ijroi_plane_outline(x) for x in best_plane_rois.roi]
    )
    traces_no_outlines = pd.DataFrame(
        extract_traces_no_outline_bool_masks(movie, full_rois_no_outlines)
    )
    traces_no_outlines.index.name = 'frame'
    traces_no_outlines.columns.name = 'roi'
    trial_dff = compute_trial_stats(traces_no_outlines, bounding_frames, odor_lists)
    roi_quality = trial_dff.max()
    roi_indices, best_plane_rois_no_outlines = rois2best_planes_only(
        full_rois_no_outlines, roi_quality, verbose=verbose
    )
    n_roi_planes = full_rois_no_outlines.sizes['roi']
    is_best_plane = np.zeros(n_roi_planes, dtype=bool)
    is_best_plane[roi_indices] = True
    full_rois_no_outlines = full_rois_no_outlines.assign_coords(
        {'is_best_plane': ('roi', is_best_plane)}
    )
    '''
    #

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
    trial_dff = compute_trial_stats(traces, bounding_frames, odor_lists)
    roi_quality = trial_dff.max()

    # TODO TODO TODO refactor so all this calculation is done across all recordings
    # within each fly, rather than just within each recording
    # (could add in parallel for now, and then phase out old way)
    roi_indices, best_plane_rois = rois2best_planes_only(full_rois, roi_quality,
        verbose=verbose
    )

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


    # TODO TODO TODO delete / check equiv to below / above subsetting
    #
    # TODO try to replace similar subsetting below w/ this (or use version up top)
    '''
    best_plane_rois_no_outlines2 = best_plane_rois.sel(
        roi=[not is_ijroi_plane_outline(x) for x in best_plane_rois.roi]
    )
    traces2 = traces.loc[:, [not is_ijroi_plane_outline(x) for x in traces.columns]]
    # TODO delete full_* if i don't end up using.
    full_traces2 = traces2.copy()
    # TODO TODO TODO need to subset roi_indices to also remove plane outlines first tho,
    # right?
    traces2 = traces2[roi_indices].copy()
    traces2.columns = best_plane_rois_no_outlines2.roi.values
    traces2.columns.name = 'roi'
    '''
    #

    # TODO delete if i don't end up using
    full_traces = traces.copy()
    assert full_traces.shape[1] == len(full_rois.roi_name.values)
    full_traces.columns = full_rois.roi_name.values
    full_traces.columns.name = 'roi'
    #

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

    # TODO move earlier (so it can be shared w/ full_traces) (or not make sense?)
    # (.name reset when assigning to .columns, as in previous line tho, i assume?)
    #
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

    non_outline_mask = np.array([not is_ijroi_plane_outline(x) for x in traces.columns])
    # TODO delete (should never get triggered)
    # (-> use non_outline_mask to subset best_plane_rois below)
    assert np.array_equal(best_plane_rois.roi.values, traces.columns)

    assert len(z_indices) == traces.shape[1]

    # TODO try to replace w/ selecting subset of full_rois before calculating these?
    # (i think it was mainly so i could make plots w/o plane outlines, but including all
    # traces?)
    #
    # only leaving the ROIs named 'AL' in full_rois (to indicate outer extent of AL, in
    # each plane, for plot_rois to draw and zero outside of)
    best_plane_rois = best_plane_rois.sel(
        roi=[not is_ijroi_plane_outline(x) for x in best_plane_rois.roi]
    )
    assert traces.columns.name == 'roi'
    traces = traces.loc[:, non_outline_mask]
    z_indices = z_indices[non_outline_mask]

    # TODO delete if i don't end up using
    #
    # TODO check this works (only fly w/ plane outlines currently 2023-05-10/1)!
    full_traces = full_traces.loc[:,
        [not is_ijroi_plane_outline(x) for x in full_traces.columns]
    ]
    assert set(traces.columns) == set(full_traces.columns)
    #

    ret = (traces, best_plane_rois, z_indices, full_rois)
    # TODO delete this kwarg (have always be true?)?
    if not roi_plots:
        return ret

    # TODO delete?
    #del traces, best_plane_rois, z_indices

    # TODO make plots comparing responses across planes (one per ROI)
    # TODO TODO or just warn if any particular ROIs have planes that deviate too much?

    date, fly_num, thorimage_id = util.dir2keys(analysis_dir)

    plot_dir = get_plot_dir(date, fly_num, thorimage_id)
    ij_plot_dir = ijroi_plot_dir(plot_dir)

    experiment_id = shorten_path(thorimage_dir)
    experiment_link_prefix = experiment_id.replace('/', '_')

    across_fly_ijroi_dir = fly2plot_root(date, fly_num) / across_fly_ijroi_dirname

    perplane_trial_df = compute_trial_stats(full_traces, bounding_frames, odor_lists)

    date_str = format_date(date)
    fly_str = f'{date_str}/{fly_num}'

    # TODO provide means of disabling this at least? (same flag as what is currently
    # controlling timeseries plot generation?)
    #
    # TODO TODO separate version of these w/ mean across trials? ONLY need that version?
    #
    # TODO save w/ bbox_inches='tight' (or otherwise so yticklabels not cut off)
    # TODO TODO include plane information for each row (and use group text to show ROI
    # name)
    # TODO hlines separating ROIs
    # TODO vlines separating odors (group text for each odor, regular tick to show
    # repeat num)
    # TODO TODO ensure trials are left in presentation order
    # TODO TODO figure out how to call out best plane (bold text? add support!)
    # TODO TODO actually do need something to group ROIs w/ same name together tho.
    # roi_sort=True (just omit roi_sort=False) might get us most of where we want
    plot_responses_and_scaled_versions(perplane_trial_df, ij_plot_dir, 'allplanes',
        title=fly_str, single_fly=True, avg_repeats=False, #roi_sort=False,

        # TODO fix need for this?
        allow_duplicate_labels=True,

        **single_fly_roi_plot_kws
    )

    panel = get_panel(thorimage_dir)

    # may generally want this True. I set to False in process of making ORN/PN example
    # dF/F images for paper (mostly to have these for my reference. may not end up using
    # the versions of the
    spatial_roi_plots_for_diag_only = False
    # NOTE: now i often have multiple recordings with this panel (b/c so many odors)
    # TODO maybe change to only do on first?
    #
    # Only plotting ROI spatial extents (+ making symlinks to those plots) for
    # diagnostic experiment for now, because for most fles I had symlinked all other
    # RoiSet.zip files to the one for the diagnostic experiment.
    if spatial_roi_plots_for_diag_only and panel != diag_panel_str:
        return ret

    # TODO TODO am i unnecessarily calling this ROI plotting stuff for stuff w/ multiple
    # diagnostic recordings? check against that dict pointing to single diagnostic
    # recordings, rather than just checking panel (above)

    # TODO refactor to do plotting elsewhere / explicitly pass in plot_dir / something?

    ijroi_spatial_extents_plot_dir = across_fly_ijroi_dir / 'spatial_extents'
    makedirs(ijroi_spatial_extents_plot_dir)

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
        #('avg', movie_mean, dict(cbar_label='mean F (a.u.)'))

        # TODO do i actually like 'log' better than default tho? (probably, see below)
        # (cbar label is uglier tho, and that might be reason to just show default
        # version... might wanna change vmin/vmax tho)
        #
        # in 2023-05-10/1 example data, norm='log' boosts weaker signals (so i do think
        # i like it better for just seeing some glomeruli i might not otherwise), but
        # not really any clear examples of constrast being improved (and might be made
        # worse for stuff already near saturation)
        # TODO this cbar_label getting cut off?
        # TODO try savefig w/ bbox_inches='tight' or somethingi?
        ('avg', movie_mean, dict(
            # TODO restore (was tweaking for GH146 example figs)?
            #norm='log',
            cbar_label='mean F (a.u.)'
        ))
    ]

    # TODO maybe do all this plotting in a separate step at the end, so i can get one
    # colormap that maps all certain ROI names across all flies to particular colors, to
    # make it easier to eyeball similarities across flies?

    # TODO maybe pick diag_example_dff_v[min|max] on a per fly basis (from some
    # percentile?)?

    # TODO delete this version? ever want to use it?
    # could occasionally be useful to show we aren't missing signals (or at least, not
    # excitatory ones).
    #
    # NOTE: as currently implemented, this will need to be generated on an earlier run
    # of this script, as these TIFFs are saved after where the ROI analysis is done.
    max_trialmean_dff_tiff_fname = analysis_dir / max_trialmean_dff_tiff_basename
    if max_trialmean_dff_tiff_fname.exists():
        max_trialmean_dff = tifffile.imread(max_trialmean_dff_tiff_fname)
        # TODO TODO move vmin/vmax warning into image_grid, so it is handled
        # homogenously? (where is it currently?)
        # (i now have something like it in add_norm_options [which gets called via
        # wrapped viz.imshow viz.image_grid uses, but may want to tweak to consolidate
        # warnings + have same thresholds as warnings in here?)
        #
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
    #

    zstep_um = thor.get_thorimage_zstep_um(thorimage_dir)
    xy_pixelsize_um = thor.get_thorimage_pixelsize_um(thorimage_dir)

    # TODO TODO fix color seed! can't line up colors between these plots and diag
    # example ones (not coloring in diag example plots anymore, so nbd now) otherwise
    # (was it always just b/c different numbers of ROIs plotted across the calls?)! (if
    # i still want colors in diag example plot... betty had wanted a version without...)
    # (seems to be OK across diagnostic recordings within a fly, and within one analysis
    # run... are these vs diagnostic odor-specific ones also good w/in a run?  or the
    # fact that some things are consistent feels like there should be a solution...)
    # TODO at least provide instructions to repro the issue in comment above...
    shared_kws = dict(zstep_um=zstep_um, pixelsize_um=xy_pixelsize_um,

        # TODO check this is working
        label_outline=True,

        # TODO TODO why does spacing between image axes seem more than in first
        # ImageGrid usage i was testing? horz and vert spacing not same?
        #
        # adjusted w/ nrows=1, ncols=None
        # TODO TODO shouldn't nrows, ncols be set here then??? (still true?)
        # (am i getting from diag_example* or something?)
        image_kws=dict(imagegrid_rect=(0.005, 0.0, 0.95, 0.955)),

        # TODO TODO if i'm gonna rely on diag example kws for one of the cases, need to
        # duplicate some of the values here for the avg bg case! (try to still share
        # somewhat w/ diag example kws)
        # TODO try to delete these after
        #depth_text_kws=dict(fontsize=7),
        # for ROI names only
        # fontsize:
        # - too small: 5
        # - OK: 6 (~1 case where names overlap, but not too bad)
        #text_kws=dict(fontsize=6),
    )

    # TODO revert all constant scale modifications?
    # (both on average and dF/F bgs) contrast within each plane tends to get
    # worse... and scale doesn't actually matter in any of those cases...
    # was just to appease betty, and i think she might agree it's not making the figure
    # better (at least, so far, towards the diagnostic example supplemental fig)

    for bg_desc, background, specific_kws in description_background_kwarg_tuples:

        plot_rois_kws = {**shared_kws, **specific_kws}

        key_overlap = set(shared_kws.keys()) & set(specific_kws.keys())
        for k in key_overlap:
            v1 = shared_kws[k]
            v2 = specific_kws[k]
            # TODO refactor to avoid instances where this is triggered
            #
            # may generally want to try to eliminate such cases, by re-organizing the
            # kws dicts.
            warn(f"{bg_desc}: shared key '{k}':\n{v1} (overwritten)\n{v2}")

        # TODO provide <date>/<fly_num> suptitle for all of these?

        # TODO color option to desaturate plane-ROIs that are NOT the "best" plane
        # (for a given volumetric-ROI) (or change line properties?)

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
            fig = plot_rois(full_rois, background, **plot_rois_kws)

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

        # TODO want to keep these symlinks / dirs?

        all_roi_dir = ijroi_spatial_extents_plot_dir / f'all_rois_on_{bg_desc}'
        makedirs(all_roi_dir)
        symlink(fig_path, all_roi_dir / f'{experiment_link_prefix}.{plot_fmt}')


        fig = plot_rois(full_rois, background, certain_only=True, **plot_rois_kws)
        fig_path = savefig(fig, ij_plot_dir, f'certain_rois_on_{bg_desc}')

        certain_roi_dir = ijroi_spatial_extents_plot_dir / f'certain_rois_on_{bg_desc}'
        makedirs(certain_roi_dir)
        symlink(fig_path, certain_roi_dir / f'{experiment_link_prefix}.{plot_fmt}')

    return ret


def trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
    main_plot_title, *, skip_all_plots=skip_singlefly_trace_plots, roi_stats=None,
    show_suite2p_rois=False):
    # TODO check it is always (unconditionally!) mean trial dF/F, or update
    # skip_all_plots doc to reference which variables trial stat depends on
    """
    Args:
        skip_all_plots: if True, only compute and return ROI x mean trial dF/F
            dataframe. makes no plots.
    """
    # TODO delete (related to one of savefig duplicate fig name issues on latest kiwi
    # data. i think the first example)
    print(f'trace_plots: {roi_plot_dir=}')
    #

    # TODO TODO remake directory (or at least make sure plots from ROIs w/ names no
    # longer in set of ROI names are deleted)

    if show_suite2p_rois and roi_stats is None:
        raise ValueError('must pass roi_stats if show_suite2p_rois')

    # TODO TODO add option to compute + return this w/o making other plots in here
    # (don't care about them much of the time...). or move computation of this out?
    #
    # TODO compare dF/F traces (or at least response means from these traces) (as
    # currently calculated), to those calculated from dF/F'd movie (in response_volumes,
    # calculated in process_recording) (still care?)
    #
    # Mean dF/F for each ROI x trial
    trial_df = compute_trial_stats(traces, bounding_frames, odor_lists)

    if skip_all_plots:
        return trial_df

    # TODO update to pathlib
    makedirs(roi_plot_dir)

    odor_index = odor_lists_to_multiindex(odor_lists)

    # TODO TODO TODO plot raw responses (including pre-odor period), w/ axvline for odor
    # onset
    # TODO TODO plot *MEAN* (across trials) timeseries responses (including pre-odor
    # period), w/ axvline for odor onset
    # TODO concatenate these DataFrames into a DataArray somehow -> serialize
    # (for other people to analyze)?
    dff_traces = list(delta_f_over_f(traces, bounding_frames, keep_pre_odor=True,
        odor_index=odor_index
    ))

    # TODO try one of those diverging colormaps w/ diff scales for the two sides
    # (since the range of inhibition is smaller)
    # TODO TODO modify diverging_cmap_kwargs / plotting fn handling so that norms
    # can use the custom vmin/vmax, as here (then replace w/ diverging_cmap_kwargs)
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

    # TODO probably cluster to order rows (=rois?) by default (just use cluster_rois if
    # so? already calling that in here below...)

    axs = None
    # TODO if i'm not gonna concat dff_traces into some xarray thing, maybe just use
    # generator in zip rather than converting to list first above
    # TODO TODO maybe condense into one plot instead somehow? (odors left to right,
    # probably just showing a mean rather than all trials)
    for trial_dff_traces, trial_bounds, trial_odors in zip(dff_traces, bounding_frames,
        odor_index):

        start_frame, first_odor_frame, _ = trial_bounds

        trial_odors = dict(zip(odor_index.names, trial_odors))
        repeat = trial_odors['repeat']

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
        # TODO TODO also include offset, like i think same does now too
        vline_level_fn = lambda frame: int(frame) >= first_odor_frame

        # TODO change viz.matshow to have better default [x|y]ticklabels
        # (should work here, shouldn't need to be a *str* one level index to use it)
        # TODO try w/o the list(...) call
        # TODO add comment explaining what trial_dff_traces.index IS here
        # (or what this is doing)
        xticklabels = [str(x) for x in trial_dff_traces.index]

        ax = axs[repeat]

        # TODO TODO TODO use similar colorscale as for other new figs (want inhibition
        # to use a much smaller range) -> see what 2023-11-21/2 B-myr inh looks like in
        # timeseries
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

        # TODO where is alignment issue coming from? colorbar really changing size
        # of first axes more (and vertically, too!)?
        # something in viz.matshow?

        # TODO fix hack (what was hack? just assuming 2 is max?)
        if repeat == 2:
            fig.colorbar(im, ax=axs, shrink=0.6)

            mix_str = format_mix_from_strs(trial_odors)
            title = mix_str
            for_filename = f'{curr_odor_i}_{mix_str}_trials'
            if debug:
                title = f'{title}\nfirst_odor_frames={pformat(first_odor_frames)}'
                for_filename = f'debug_{for_filename}'

            fig.suptitle(title)

            savefig(fig, timeseries_plot_dir, for_filename)

            curr_odor_i += 1

    is_pair = is_pairgrid(odor_lists)

    # TODO maybe replace odor_sort kwarg with something like odor_sort_fn, and pass
    # sort_concs in this case, and maybe something else for the non-pair experiments i'm
    # mainly dealing with now

    # TODO make axhlines between changes in z_indices
    fig, mean_df = plot_all_roi_mean_responses(trial_df, sort_rois_first_on=z_indices,
        odor_sort=is_pair, title=main_plot_title, cbar_label=trial_stat_cbar_title,
        cbar_shrink=0.4
    )
    # TODO rename to 'traces' or something (to more clearly disambiguate wrt spatial
    # extent plots)
    savefig(fig, roi_plot_dir, 'all_rois_by_z')

    # (copied from across fly ijroi analysis)
    # TODO odors sorted already (probably just b/c they were presented in that
    # order?)? do that here if not (w/ al_analysis.sort_odors) (that's the only reason
    # we have odor_sort=False below, but setting that True using different sorting than
    # what i might want)
    # TODO factor cmap handling into cluster_rois defaults?
    # TODO group xticklabels on this one
    #
    # TODO why is this not having same recursion error issue as newer attempts to use
    # this for plot model KC responses? (i assume it was just a data size thing...)
    # trial_df.columns.name = 'roi'
    # trial_df.indiex.names = ['odor1', 'odor2', 'repeat']
    # trial_df.shape = (30, 40)
    #
    cg = cluster_rois(trial_df, odor_sort=False,
        cmap=cmap
        # also not doing what i want. would need to fix viz.add_norm_options /
        # viz.clustermap cbar handling.
        #cmap=diverging_cmap
        #
        # TODO TODO fix clustermap wrapper to make cbar small in case of
        # two-slope norm (for side w/ smaller magnitude from vcenter)
        # (to make consistent w/ matshow behavior, which i think is handled by
        # add_colorbar there) (-> then restore these kwargs, instead of cmap)
        #**diverging_cmap_kwargs
    )
    savefig(cg, roi_plot_dir, 'all_rois_clust_trials')

    mean_df = trial_df.groupby(level='odor1', sort=False).mean()
    cg = cluster_rois(mean_df, odor_sort=False, cmap=cmap)
    savefig(cg, roi_plot_dir, 'all_rois_clust')

    if not is_pair:
        return trial_df

    # TODO delete
    print('SKIPPING BROKEN PAIR PATH IN TRACE_PLOTS')
    return trial_df
    #
    # TODO TODO fix! somewhat broken for at least one old kiwi pair fly
    # (2 ROIs w/ same DL1 name?)
    # ./al_analysis.py -d pebbled -n 6f -v -t 2022-02-22 -e 2022-04-03 -s model -i ijroi
    #
    # ...
    # SAVING ONE OF WHAT WILL BE A DUPLICATE FIGURE NAME
    #   File "./al_analysis.py", line 13431, in <module>
    #     main()
    #   File "./al_analysis.py", line 12480, in main
    #     was_processed = list(starmap(process_recording, keys_and_paired_dirs))
    #   File "./al_analysis.py", line 5149, in process_recording
    #     ij_trial_df, best_plane_rois, full_rois = ij_trace_plots(analysis_dir,
    #   File "./al_analysis.py", line 4257, in ij_trace_plots
    #     trial_df = trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
    #   File "./al_analysis.py", line 4206, in trace_plots
    #     savefig(fig, roi_plot_dir, str(roi))
    #   File "./al_analysis.py", line 1795, in savefig
    #     traceback.print_stack(file=sys.stdout)
    # pebbled_6f/png/2022-02-22_1_kiwi_ea_eb_only/ijroi/DL1.png
    #
    # SAVING ONE OF WHAT WILL BE A DUPLICATE FIGURE NAME
    #   File "./al_analysis.py", line 13431, in <module>
    #     main()
    #   File "./al_analysis.py", line 12480, in main
    #     was_processed = list(starmap(process_recording, keys_and_paired_dirs))
    #   File "./al_analysis.py", line 5149, in process_recording
    #     ij_trial_df, best_plane_rois, full_rois = ij_trace_plots(analysis_dir,
    #   File "./al_analysis.py", line 4257, in ij_trace_plots
    #     trial_df = trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
    #   File "./al_analysis.py", line 4206, in trace_plots
    #     savefig(fig, roi_plot_dir, str(roi))
    #   File "./al_analysis.py", line 1795, in savefig
    #     traceback.print_stack(file=sys.stdout)
    # abs_fig_path=PosixPath('/home/tom/src/al_analysis/pebbled_6f/png/2022-02-22_1_kiwi_ea_eb_only/ijroi/DL1.png')
    # desc='DL1?'
    # > /home/tom/src/al_analysis/al_analysis.py(1806)savefig()
    #    1805
    # -> 1806     _savefig_seen_paths.add(abs_fig_path)
    #    1807
    #
    # ipdb>
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

    # TODO TODO are these traces also starting at odor onset, like ones calculated via
    # my delta_f_over_f fn (within ij_traces)
    import ipdb; ipdb.set_trace()

    # TODO also plot roi / outline of roi on corresponding [mean?] plane / maybe with
    # other planes for context?

    roi_plot_dir = suite2p_plot_dir(plot_dir)
    title = 'Suite2p ROIs\nOrdered by Z plane'

    trial_df = trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
        title, roi_stats=roi_stats, show_suite2p_rois=False
    )

    return trial_df


def ij_trace_plots(analysis_dir, bounding_frames, odor_lists, movie, plot_dir):

    # TODO TODO TODO maybe each fn that returns traces (this and suite2p one), should
    # already add the metadata compute_trial_stats adds, or more? maybe adding a
    # seconds time index w/ 0 on each trial's first_odor_frame?
    #
    # could probably pass less variables if some of them were already in the coordinates
    # of a DataArray traces
    traces, best_plane_rois, z_indices, full_rois = ij_traces(analysis_dir, movie,
        bounding_frames, roi_plots=True, odor_lists=odor_lists
    )

    roi_plot_dir = ijroi_plot_dir(plot_dir)
    title = 'ImageJ ROIs\nOrdered by Z plane\n*possibly [over/under]merged'

    trial_df = trace_plots(traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
        title
    )

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

        mocorr_concat_tiff_basename,
        all_frame_avg_tiff_basename,

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

    # TODO TODO build this up across flies (for use in diag example response matrix
    # plotting, which is currently part of acrossfly_response_matrix_plots)
    # TODO or maybe i should just move that in here?
    # TODO TODO also apply target_glomerulus_renames to that (prob make that global too)
    #
    # TODO delete if i manage to refactor code below to only do the formatting of odors
    # right before plotting, rather than in the few lines before this
    #
    # This is just intended to target glomeruli information for the glomeruli
    # diagnostic experiments.
    odor_str2target_glomeruli = {
        s: o[0]['glomerulus'] for s, o in zip(odor_order_with_repeats, odor_lists)
        if len(o) == 1 and 'glomerulus' in o[0]
    }

    # TODO fix elsewhere to not need this hack (/better name at least)?
    target_glomerulus_renames = {
        # ethyl 3-hydroxybutyrate @ -6
        'DM5/VM5d': 'DM5',
    }

    # TODO build this up in preprocessing instead? and move target_glomerulus_renames to
    # module level?
    for o, g in odor_str2target_glomeruli.items():

        g = target_glomerulus_renames.get(g, g)

        # data that will be used with all_odor_str2target_glomeruli will have odors
        # already abbreviated. olf.abbrev(...) now also works for input with
        # concentration info (e.g. 'ethyl acetate @ -4').
        o = olf.abbrev(o)

        # TODO TODO fix (erring on some old kiwi data):
        # ./al_analysis.py -d pebbled -n 6f -v -t 2022-02-07 -e 2022-04-03 -s model -i ijroi
        # ...
        # thorimage_dir: 2022-02-28/1/glomeruli_diagnostics
        # thorsync_dir: 2022-02-28/1/SyncData001
        # yaml_path: 20220228_165028_stimuli.yaml
        # Uncaught exception
        # Traceback (most recent call last):
        #   File "./al_analysis.py", line 13476, in <module>
        #     main()
        #   File "./al_analysis.py", line 12525, in main
        #     was_processed = list(starmap(process_recording, keys_and_paired_dirs))
        #   File "./al_analysis.py", line 4811, in process_recording
        #     assert g == old_g, f'{o}, target glom mismatch: {g} != (previous) {old_g}'
        # AssertionError: 2-but @ -6, target glom mismatch: VM7d != (previous) VM7
        if o in all_odor_str2target_glomeruli:
            old_g = all_odor_str2target_glomeruli[o]
            # TODO maybe switch to warning if this is actually getting triggered
            #assert g == old_g, f'{o}, target glom mismatch: {g} != (previous) {old_g}'
            # TODO TODO restore assertion after fixing
            if g != old_g:
                warn(f'{o}, target glom mismatch: {g} != (previous) {old_g}')
                continue

        all_odor_str2target_glomeruli[o] = g


    # TODO use that list comprehension way of one lining this? equiv for sets?
    name_lists = [[o['name'] for o in os] for os in odor_lists]
    for ns in name_lists:
        for n in ns:
            if n not in odor2abbrev:
                # didn't use a set for this because back when i used multiprocessing
                # option to call process_recording, wasn't a great type to support
                # across-process shared sets
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
        # TODO do i ever actually use this? did i mean to?
        # (seems yes, but only in ijroi case) (still?)
        # TODO probably factor something like this into top level / hong2p.xarray/util
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

    # TODO print which file this is hitting on (to debug)?
    # make a verbose flag for that?
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
            # TODO TODO this accurate? did rsync update a time on something it shouldn't
            # have? or does this just get printed in ijroi changing case too (/similar)?
            # TODO print tiff_path to debug? and is it a link? what determines mtime on
            # a symlink? can you touch it? is rsync effectively doing that?
            # TODO or was it just that i switched plot_fmt?
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

    zstep_um = thor.get_thorimage_zstep_um(xml)

    # only defined inside here b/c add_metadata call currently also is (and needs to
    # be, to avoid passing the metadata it uses)
    def _compute_roi2best_plane_depth(full_rois, best_plane_rois) -> pd.Series:
        # full_rois has more metadata on roi index, including roi_z that we want

        full_rois_bestplane = full_rois[dict(roi=best_plane_rois.roi_index.values)]

        assert np.array_equal(full_rois_bestplane, best_plane_rois)
        assert np.array_equal(
            best_plane_rois.roi.values, full_rois_bestplane.roi_name.values
        )
        assert (
            len(set(full_rois_bestplane.roi_name.values)) ==
            len(full_rois_bestplane.roi_name)
        )
        roi2best_plane = dict(zip(
            full_rois_bestplane.roi_name.values,
            full_rois_bestplane.roi_z.values
        ))
        # assuming we start at same point. would add complexity to register
        # to anat (-> get better depth), and may not help muc
        roi2best_plane_depth = {k: v * zstep_um for k, v in roi2best_plane.items()}
        roi2best_plane_depth = pd.Series(roi2best_plane_depth,
            name='best_plane_depth_um'
        )
        roi2best_plane_depth.index.name = 'roi'
        roi2best_plane_depth = add_metadata(roi2best_plane_depth.to_frame().T)
        junk_level = -1
        assert (
            set(roi2best_plane_depth.index.get_level_values(junk_level)) ==
            {'best_plane_depth_um'}
        )
        assert roi2best_plane_depth.index.names[junk_level] is None
        roi2best_plane_depth = roi2best_plane_depth.droplevel(junk_level, axis='index')
        return roi2best_plane_depth


    full_rois = None

    # TODO am i currently refusing to do any imagej ROI analysis on non-mocorred
    # stuff? maybe i should?
    do_ij_analysis = False
    if analyze_ijrois:
        have_ijrois = has_ijrois(analysis_dir)
        fly_key = (date, fly_num)

        if (fly_key in fly2diag_thorimage_dir and
            (fly_analysis_dir / mocorr_concat_tiff_basename).exists()):

            diag_analysis_dir = thorimage2analysis_dir(fly2diag_thorimage_dir[fly_key])
            diag_ijroi_fname = ijroi_filename(diag_analysis_dir, must_exist=False)

            if not have_ijrois:
                if diag_ijroi_fname.is_file():
                    diag_ijroi_link = analysis_dir / ijroiset_default_basename

                    # TODO maybe i should switch to saving RoiSet.zip's in fly analysis
                    # dirs (at root, rather than under each recording's subdir) ->
                    # delete all RoiSet.zip symlinking code?
                    print_if_not_skipped(f'no {ijroiset_default_basename}. linking to '
                        'ROIs defined on diagnostic recording '
                        f'{shorten_path(diag_analysis_dir)}'
                    )
                    # NOTE: changing the target of the link should also trigger
                    # recomputation of ImageJ ROI outputs for directories w/ links to
                    # the changed RoiSet.zip. tested.
                    symlink(diag_ijroi_fname, diag_ijroi_link)
                    have_ijrois = True
            else:
                # checking whether we saved the RoiSet.zip in the right place
                # (currently needs to be in the chronologically first recording)
                if not diag_ijroi_fname.exists():
                    ijroi_fname = ijroi_filename(analysis_dir)
                    # delete link if this gets triggered (but it shouldn't)
                    assert not ijroi_fname.is_symlink()
                    # this will return True for symlinks too, so we check that above
                    assert ijroi_fname.is_file()

                    # TODO just support this path too? might be more effort than i want
                    raise IOError('expected RoiSet.zip at chronologically first '
                        f'recording:\n{shorten_path(diag_analysis_dir)}\n\n'
                        f'found non-symlink ROIs at:\n{shorten_path(analysis_dir)}'
                        '\n\nmove this RoiSet.zip to expected location!'
                    )

        # TODO err if any one fly has >1 non-symlink RoiSet.zip and any symlink ones
        # (b/c probably not set up right) (or if any other dirs missing files/links)

        if have_ijrois:
            dirs_with_ijrois.append(analysis_dir)

            ij_trial_df_cache = analysis_dir / ij_trial_df_cache_basename

            full_rois_cache = analysis_dir / 'full_rois.p'
            best_plane_rois_cache = analysis_dir / 'best_plane_rois.p'

            # no longer always making plots in here (flags to disable some, and only
            # generating others for diagnostic experiments), so just checking mtime of
            # analysis_dir / ij_trial_df_cache_basename, rather than plots under ijroi
            # plot subdir
            ijroi_last_analysis = ij_last_analysis_time(analysis_dir)

            if ijroi_last_analysis is None:
                ij_analysis_current = False
            else:
                # We don't need to check LHS for None b/c have_ijrois check earlier.
                ij_analysis_current = (
                    ijroi_mtime(analysis_dir) < ijroi_last_analysis
                )

            ignore_existing_ijroi = should_ignore_existing('ijroi')

            do_ij_analysis = True
            if not ignore_existing_ijroi:
                if ij_analysis_current:
                    do_ij_analysis = False

                    print_if_not_skipped(
                        'ImageJ ROIs unchanged since last analysis. reading cache.'
                    )
                    ij_trial_df = pd.read_pickle(ij_trial_df_cache)
                    ij_trial_dfs.append(ij_trial_df)

                    full_rois = read_pickle(full_rois_cache)

                    # TODO delete separate check this exists after regenerating all
                    if best_plane_rois_cache.exists():
                        best_plane_rois = read_pickle(best_plane_rois_cache)
                    else:
                        # TODO should this be an error instead? feels like it...
                        warn(f'{best_plane_rois_cache} did not exist, but '
                            f'{full_rois_cache} did. can add `-i ijroi` to CLI args to '
                            'regenerate.'
                        )
                    #

                    roi2best_plane_depth = _compute_roi2best_plane_depth(full_rois,
                        best_plane_rois
                    )
                    roi2best_plane_depth_list.append(roi2best_plane_depth)
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

    # TODO TODO make sure everything below that requires full_rois/best_plane_rois
    # to be not None below only happens if do_ij_analysis == True
    # (or actually, just actually cache and load these in have_ijrois [but not
    # !do_ij_analysis] case...)
    # (nothing below should currently use best_plane_rois at all)

    if do_ij_analysis:
        # TODO just return additional stuff from this to use best_plane_rois in
        # plot_rois as like full_rois (some metadata dropped when converting latter to
        # former)?
        # TODO refactor to do plot_rois stuff ij_trace_plots does (after early
        # return option) after here instead. -> use cached input if we request
        # recomputation of those plots (worth it? actually save time?)
        ij_trial_df, best_plane_rois, full_rois = ij_trace_plots(analysis_dir,
            bounding_frames, odor_lists, movie, plot_dir
        )

        # TODO TODO check responses have similar distribution of where the pixels with
        # which intensity ranks are (as response coming from a different place w/in ROI
        # is likely contamination from something nearby?). check center of intensity
        # mass? warn if deviations are above some threshold?
        # do in a script to be called from an imagej macro?

        # TODO TODO compare best_plane_rois across recordings (+ probably refactor
        # their computation to always be across all recordings for a fly anyway, rather
        # than computing within each recording...)

        ij_trial_df = add_metadata(ij_trial_df)
        to_pickle(ij_trial_df, ij_trial_df_cache)
        ij_trial_dfs.append(ij_trial_df)

        assert full_rois is not None and best_plane_rois is not None

        roi2best_plane_depth = _compute_roi2best_plane_depth(full_rois,
            best_plane_rois
        )
        roi2best_plane_depth_list.append(roi2best_plane_depth)

        # probably wanna move all this after process_recording anyway though, computing
        # everything on concatenated data...
        #
        # was previously using `-i nonroi,ijroi`, as a workaround to accomplish similar
        # TODO delete comment in line above
        #
        # (not currently using best_plane_rois outside of ij_traces!)
        #
        # we don't currently use metadata on this (don't concat across flies).
        # could add most from path details if needed.
        to_pickle(full_rois, full_rois_cache)
        to_pickle(best_plane_rois, best_plane_rois_cache)

    # TODO TODO compute lags between odor onset (times) and peaks fluoresence times
    # -> warn if they differ by a certain amount, ideally an amount indicating
    # off-by-one frame misassignment, if that is even reliably possible

    if not do_nonroi_analysis:
        print_skip('skipping non-ROI analysis\n')
        return

    def micrometer_depth_title(ax, z_index) -> None:
        viz.micrometer_depth_title(ax, zstep_um, z_index, fontsize=ax_fontsize)

    xy_pixelsize_um = thor.get_thorimage_pixelsize_um(xml)

    scalebar_kws = dict(font_properties=dict(size=3.5))

    def plot_and_save_dff_depth_grid(dff_depth_grid, fname_prefix, title=None,
        cbar_label=None, experiment_id_in_title=False, normalize_fname=True,
        anatomical=False, cmap=None, **imshow_kwargs) -> Path:

        if cmap is None:
            if not anatomical:
                imshow_kwargs = {**diverging_cmap_kwargs, **imshow_kwargs}
            else:
                anatomical_kws = dict(cmap=anatomical_cmap)
                imshow_kwargs = {**anatomical_kws, **imshow_kwargs}
        else:
            imshow_kwargs['cmap'] = cmap

        # Will be of shape (1, z), since squeeze=False
        fig, axs = plt.subplots(ncols=z, squeeze=False,
            figsize=single_dff_image_row_figsize
        )

        vmin0 = None
        vmax0 = None

        for d in range(z):
            # TODO could also do axs.flat[d]?
            ax = axs[0, d]

            # TODO this isn't what i do in other cases tho, right? can't i just fix
            # float specified inside micrometer_depth_title, if "-0 uM" or something
            # like that is why i'm special casing this? (it does do -0, but not actually
            # sure why...)
            if z > 1:
                micrometer_depth_title(ax, d)

            im = dff_imshow(dff_depth_grid[d], ax, **imshow_kwargs)

            # checking range is same on all (so it's OK to make cbar from last)
            # (or replace plot_and_save... w/ viz.[image_grid|plot_rois], which already
            # does)
            if vmin0 is None:
                vmin0 = im.norm.vmin
                vmax0 = im.norm.vmax
            else:
                assert im.norm.vmin == vmin0
                assert im.norm.vmax == vmax0

        # TODO delete extend='both'?
        viz.add_colorbar(fig, im, label=cbar_label, shrink=0.68, extend='both')

        viz.add_scalebar(axs.flat[0], xy_pixelsize_um, **scalebar_kws)

        suptitle(title, fig, experiment_id_in_title=experiment_id_in_title)
        fig_path = exp_savefig(fig, fname_prefix, normalize_fname=normalize_fname)
        return fig_path

    anat_baseline = movie.mean(axis=0)

    # TODO just replace this plot_and_save... call w/ either plot_rois or
    # viz.image_grid (both of which have minmax_clip_frac, which this logic is
    # duplicated from, as well as the norm='log' option [if i want that...])
    vmin = np.quantile(anat_baseline, roi_background_minmax_clip_frac)
    vmax = np.quantile(anat_baseline, 1 - roi_background_minmax_clip_frac)
    # log does honestly look a bit better, including at the 0.001 clip frac for both.
    # preprint doesn't have cbar for background fluoresence images, but might still need
    # to say something in legend if i use log for this one?
    norm = None
    #norm = 'log'
    kws = dict()
    if norm == 'log':
        kws['norm'] = 'log'
        norm_suffix = '_log'
    else:
        norm_suffix = '_nolog'

    # TODO rename fn (since input here is not dF/F)?
    plot_and_save_dff_depth_grid(anat_baseline,
        f'avg_{roi_background_minmax_clip_frac:.2g}{norm_suffix}',
        'average of whole movie', normalize_fname=False,
        #
        anatomical=True, vmin=vmin, vmax=vmax, **kws
    )
    del vmin, vmax, kws, norm

    save_dff_tiff = want_dff_tiff
    if save_dff_tiff:
        dff_tiff_fname = analysis_dir / 'dff.tif'
        if dff_tiff_fname.exists():
            # To not write large file unnecessarily. This should never really change,
            # especially not if computed from non-motion-corrected movie.
            save_dff_tiff = False

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

    # TODO refactor loop below to zip over these, rather than needing to explicitly call
    # next() as is
    dff_after_onset_iterator = delta_f_over_f(movie, bounding_frames,
        odor_index=odor_index
    )
    dff_full_trial_iterator = delta_f_over_f(movie, bounding_frames, keep_pre_odor=True,
        odor_index=odor_index
    )

    if save_dff_tiff:
        trial_dff_movies = []

    # List of length equal to the total number of trials, each element an array of shape
    # (z, y, x).
    all_trial_mean_dffs = []

    # Each element mean of the (mean) volumetric responses across the trials of a
    # particular odor, of shape (z, y, x)
    odor_mean_dff_list = []

    for i, odor_str in enumerate(odor_order):

        abbrev_odor_str = olf.abbrev(odor_str)

        # TODO more clear if i do .replace(' @ ', ' ')?
        abbrev_odor_str = abbrev_odor_str.replace(' @', '')

        if odor_str in odor_str2target_glomeruli:
            target_glomerulus = odor_str2target_glomeruli[odor_str]
            title = f'{odor_str} ({target_glomerulus})'
            # TODO maybe use just abbrev_odor_str for plot_rois version w/ focus ROI
            # name highlighted in label over image (by ROI)?
            abbrev_title = f'{abbrev_odor_str} ({target_glomerulus})'
        else:
            target_glomerulus = None
            title = odor_str
            abbrev_title = abbrev_odor_str

        # TODO either:
        # - always use 2 digits (leading 0)
        # - pick # of digits from len(odor_order)
        plot_desc = f'{i + 1}_{title}'

        trial_heatmap_fig, trial_heatmap_axs = plt.subplots(nrows=n_repeats,
            ncols=z, squeeze=False, figsize=(6.4, 3.9)
        )

        vmin0 = None
        vmax0 = None

        # Each element is mean-within-a-window response for one trial, of shape
        # (z, y, x)
        trial_mean_dffs = []

        # TODO do mean_dff/etc calculation in this loop, but move plotting below (or
        # just replace plotting w/ existing/new plot_rois/similar calls?
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
                im = dff_imshow(mean_dff[d], ax)

                if n == 0 and d == 0:
                    viz.add_scalebar(ax, xy_pixelsize_um, **scalebar_kws)

                # checking range is same on all (so it's OK to make cbar from last)
                # (or replace plot_and_save... w/ viz.[image_grid|plot_rois], which
                # already does)
                if vmin0 is None:
                    vmin0 = im.norm.vmin
                    vmax0 = im.norm.vmax
                else:
                    assert im.norm.vmin == vmin0
                    assert im.norm.vmax == vmax0

        avg_mean_dff = np.mean(trial_mean_dffs, axis=0)

        # TODO delete extend='both'?
        viz.add_colorbar(trial_heatmap_fig, im, label=dff_cbar_title, shrink=0.32,
            extend='both'
        )

        suptitle(title, trial_heatmap_fig)
        exp_savefig(trial_heatmap_fig, plot_desc + '_trials')

        trialmean_dff_fig_path = plot_and_save_dff_depth_grid(avg_mean_dff, plot_desc,
            title=title, cbar_label=f'mean {dff_latex}'
        )

        focus_roi = target_glomerulus_renames.get(target_glomerulus, target_glomerulus)

        # TODO check have_ijrois is what i want here (/ works)
        if have_ijrois:
            # TODO TODO try depth specific colorbars?

            # TODO replace main dF/F plots with this (as long as these also can
            # work w/o ROIs, or can be modified to do so)?
            # (currently hardcoding reduced scale in this version, so prob not?)

            # TODO symlink to these in across fly ijroi dir? (still want?)

            # TODO delete eventually (may need to rerun ijroi analysis on everything i
            # care about)
            assert full_rois is not None

            trialmean_dff_w_rois_fig = plot_rois(full_rois, avg_mean_dff,
                focus_roi=focus_roi, zstep_um=zstep_um, pixelsize_um=xy_pixelsize_um,
                title=abbrev_title,

                # TODO move certain_only=True diag_example_plot_kws (if i like it)?
                # (why not? can't see why i'd want uncertain ones for these plots...)
                certain_only=True,

                **diag_example_plot_roi_kws
            )

            # TODO TODO restore? other colors / ways of showing clipping?
            #
            # TODO clean up!
            # TODO TODO work (actually, yea, seems to)? like the colors?
            # TwoSlopeNorm may not draw over/under colors correctly:
            # https://stackoverflow.com/questions/69351535
            '''
            [
                #x.get_images()[0].cmap.set_under('green') for x in
                x.get_images()[0].cmap.set_under('gray') for x in
                trialmean_dff_w_rois_fig.get_axes() if len(x.get_images()) > 0
            ]
            [
                x.get_images()[0].cmap.set_over('gray') for x in
                trialmean_dff_w_rois_fig.get_axes() if len(x.get_images()) > 0
            ]
            '''
            #

            # TODO make sure these are sorted correctly by PDF aggregation code
            # (in fixed odor order, as other diagnostic dF/F images should be)
            # (can't figure how this is happening, if it is [in commented code, no
            # less]... delete comment?)
            savefig(trialmean_dff_w_rois_fig, plot_dir / 'ijroi/with_rois', plot_desc,

                # TODO find a png viewer that displays this as white instead of
                # checkerboard? or a pdf viewer that is as quick to use as default png
                # viewer? or only set True sometimes?
                #transparent=True,

                # TODO why this causing failure (seems to be because of inset_axes)?
                # just wanted this to have colorbar ticks/label not cut off, but could
                # probably handle that another way)
                # (may be fixed in mpl 3.8. i'm using 3.7.2 now and can't upgrade w/o at
                # least python 3.9. i'm still on python 3.8.12 for now.)
                #
                # https://github.com/jupyter/notebook/issues/7052
                # after upgrading to 3.7.3, it no longer errs, but cbar position gets
                # screwed up.
                #
                # cbars work with this now that their axes are created via ImageGrid
                #
                # (both still 2.8125 x 10.5 figsize, at least when savefig is called, if
                # not in output)
                # w/ bbox_inches='tight':
                # Page size:      161.998 x 677.995 pts
                # w/o:
                # Page size:      202.5 x 756 pts
                #bbox_inches='tight'
            )

        odor_mean_dff_list.append(avg_mean_dff)

        # TODO TODO is this stuff not getting run? fix if so (/delete...)
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

    assert len(odor_index) == len(all_trial_mean_dffs)
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

    # TODO delete (esp of actually running in parallel... seems like it could cause
    # problems)
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


def should_flip_lr(date, fly_num, _warn=True) -> Optional[bool]:
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
        if _warn:
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
        have_some_known_panels = False

        # TODO TODO consider min_input here? what if min_input='raw'? wouldn't we want
        # to concat all of those? (or otherwise, maybe delete min_input + related?)
        tiff_name_to_register = 'flipped.tif'

        assert len(all_thorimage_dirs) > 0

        for thorimage_dir in all_thorimage_dirs:
            panel = get_panel(thorimage_dir)
            # TODO warn though (maybe just after loop, if we haven't already warned
            # about NO recognized panels being found?)?
            if panel is None:
                continue

            have_some_known_panels = True

            # TODO may also need to filter on shape (hasn't been an issue so far)

            analysis_dir = get_analysis_dir(date, fly_num, thorimage_dir)
            tiff_fname = analysis_dir / tiff_name_to_register
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

            if have_some_known_panels:
                warn(f'{fly_str}: no {tiff_name_to_register} TIFFs to register! other '
                    'TIFFs not included in across-recording registration. may need to '
                    "populate 'Side' column in Google Sheet."
                )
            else:
                warn(f'{fly_str}: no recognized panels! other recordings not included '
                    'in across-recording registration! modify get_panel to return a '
                    'panel str for matching ThorImage output directories.\n\nThorImage '
                    f'output dirs for this fly:\n{pformat(all_thorimage_dirs)}\n'
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


# TODO delete? replace w/ olf.parse_log10_conc?
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

    # TODO serialize corrs for scatterplot directly comparing KC and ORN correlation
    # consistency in the same figure/axes?

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

        kwargs = remy_corr_matshow_kwargs

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
            # TODO replace w/ corr_triangular (+ move [invert_]corr_triangular to
            # hong2p.util?)
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
            if any(ostr.startswith(p) for p in ('kmix','cmix', '~kiwi', 'control mix')):
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

    # TODO TODO TODO show NaNs (at least those that appear for specific flies but not
    # the panel at large) (e.g. in single flies in validation2 where some odors were
    # nulled, as well as those va/aa flies in megamat)
    #
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
        # TODO TODO drop to only panel odors instead (?)
        corr = dropna_odors(garr.squeeze(drop=True))

        kwargs = remy_corr_matshow_kwargs

        name_order = panel2name_order.get(panel)

        fig = natmix.plot_corr(corr, title=fly_str, name_order=name_order, **kwargs)

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

    # TODO version of these plots using ROI responses as inputs (refactor)
    # (is it not currently? delete comment?)

    # TODO (delete?) reorganize directories so all (downsampled) pixelbased stuff is in
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

    # TODO TODO maybe rename basename here + factor into
    # natmix_activation_strength_plots (to remove reference to 'pixel') (change any
    # model_mixes_mb code that currently hardcodes this filename)
    # TODO TODO also save the equivalent from the ijroi analysis elsewhere
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
        fig, _ = viz.matshow(diff_df, title=f'component {desc} minus observed',
            #xticklabels=True, yticklabels=format_mix_from_strs,
            #**diverging_cmap_kwargs,
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


# TODO rewrite stuff in plot_roi[_util].py that uses this, so i can load from all files
# matching <driver>_<indicator>_ij_roi_stats.p (or rewrite here to save all data to one
# global file, but liking the sound of that less...)
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

    # NOTE: broken. these should also be under <driver>_<indicator> "output" dirs now
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

    # TODO TODO try to cluster odor mixture behavior types across odor pairs, but
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
def cluster_rois(df: pd.DataFrame, title=None, odor_sort: bool = True, cmap=cmap,
    **kwargs) -> sns.axisgrid.Grid:
    # TODO doc expectations on what rows / columns of input are

    # TODO why transposing? stop doing that (-> change all inputs). will just cause
    # confusion if i use [row/col]_colors...
    if odor_sort:
        # TODO why olf.sort_odors not one defined in here?
        # After transpose: columns=odors
        df = olf.sort_odors(df.T)
    else:
        # just in case we didn't sort above. we still don't want to modify input...
        df = df.T.copy()

    # TODO plus, this will screw up [h|v]line_level_fn stuff...
    # TODO warn/fail if this is not the case? accidentally didn't hit this when there
    # was a bug in above
    if 'odor1' in df.columns.names:
        # TODO why? comment explaining? just use an appropriate index-value-dict ->
        # formatted str fn? isn't that what i do w/ hong2p.viz.matshow calls?
        df.columns = df.columns.get_level_values('odor1')

    # TODO TODO add option to color rows by fly (-> generate row_colors Series in here)
    # (values of series should be colors)

    cg = viz.clustermap(df, col_cluster=False, cmap=cmap, **kwargs)
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
# TODO fn to combine this w/ loading of hallem data (-> only use that wrapper)?
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

            # these should be the only 7 odors in validation2 panel that are in Hallem,
            # courtesy of what Remy sent me on slack 2024-02-12.
            # already abbreviated (concentration in validation2 panel in parens):
            # 1-propanol (-3), ethyl octanoate (-2), geraniol (-2),
            # phenylacetaldehyde (-4)
            #
            # (-2)
            'b-myrcene': 'beta-myrcene',
            # (-1.5)
            'a-pinene': '(-)-alpha-pinene',
            # (-3)
            '(-)-trans-caryophyllene': '(-)-beta-caryophyllene',

            # TODO restore -> re-run hallem modelling -> check nothing broke
            # (-> maybe remove some/all other hacks to odor abbrev handling in creation
            # of my versions of 2E plot)
            #
            ## abbreviations Remy uses in some of 2E data she sent me.
            ## (for the 24 - 17 odors she is had some data for, beyond megamat17).
            ## all should be in hallem.
            ## I already had an abbreviation for the 7th ('1-penten-3-ol' -> '1p3ol').
            #'eugenol': 'eug',
            #'ethyl cinnamate': 'ECin',
            #'propyl acetate': 'PropAc',
            ## these two had full names w/ 'gamma'/'delta' prefix in some of Remy's
            ## things, but these LHS values are what I need to convert from in my Hallem
            ## representation.
            #'g-hexalactone': 'g-6lac',
            #'d-decalactone': 'd-dlac',
            ## TODO TODO might need 'moct' -> 'MethOct' (assuming my abbrev has
            ## already been applied [it prob wasn't, considering it's called below...].
            ## how have i been handling any other existing abbrev conflicts? are there
            ## any others?)
            ## TODO TODO how do i want to handle this? leaning towards just renaming
            ## to 'moct' when loading remy's 2e data...
            ##'methyl octanoate': 'MethOct',
        },
        axis=axis
    )
    df = df.rename(olf.odor2abbrev, axis=axis)
    return df


# TODO TODO refactor to also do this for certain odors in older validation2 data?
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


def setnull_old_wrong_solvent_1p3one_data(trial_df: pd.DataFrame) -> pd.DataFrame:
    # I don't think any old 2-component air-mixture data (where odor2 is defined) is
    # affected, and I'm unlikely to reanalyze that data at this point.
    odor1 = trial_df.index.get_level_values('odor1')
    wrong_solvent_odors = odor1.str.startswith('1p3one @')

    last_wrong_solvent_date = pd.Timestamp('2023-10-19')
    assert trial_df.columns.names[0] == 'date', 'date slicing below needs this'

    to_null = trial_df.loc[wrong_solvent_odors, :last_wrong_solvent_date]
    n_new_null = num_notnull(to_null)
    if n_new_null > 0:
        # NOTE: remy was always using water for this, i just accidentally (probably)
        # used pfo instead of water for my 2023-10-15 and 2023-10-19 validation2 flies
        msg = 'nulling 1p3one data for flies (from before switch to water solvent):\n'
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


# TODO replace this whole thing (+ final_panel_concs), w/ just backfilling any NaN in
# trial_df (eh...)?
#
# TODO share some type defs (in sig annot) w/ final_panel_concs
# TODO move near final_panel_concs
def setnull_nonfinal_panel_odors(panel2final_conc_dict: Dict[str, Dict[str, float]],
    # TODO delete?
    #panel2final_conc_start_time: Dict[str, datetime],
    df: pd.DataFrame) -> pd.DataFrame:

    # TODO do i even need panel2final_conc_start_time?
    # TODO do i need a map of (date, fly_num, panel) -> [max?] recording time, in
    # order to make use of panel2final_conc_start_time?

    df = df.copy()

    assert df.index.names[0] == 'panel'

    # TODO fix this fn (or at least more permanently check and special case) for pair
    # data
    try:
        assert (df.index.get_level_values('is_pair') == False).all()
    except AssertionError:
        print('SKIPPING SETNULL_NONFINAL_PANEL_ODORS b/c have some pair data')
        return df

    for panel, conc_dict in panel2final_conc_dict.items():
        panel_index = (panel, False)

        try:
            panel_df = df.loc[panel_index]

        # panel2final_conc_dict can currently have panels not in the data to be analyzed
        # (e.g. if a driver/indicator used to subset data, rather than date range, as in
        # `./al_analysis.py -d sam -n 6f`)
        except KeyError:
            continue

        odor1 = panel_df.index.get_level_values('odor1')

        # TODO restore? delete?
        if panel == diag_panel_str:
            continue

        # TODO need inexact match on conc (e.g. np.isclose) (don't think so)?
        final_odors = {(olf.abbrev(n), c) for n, c in conc_dict.items()}

        odor1_dicts = [olf.parse_odor(x) for x in odor1]

        at_final_conc = np.array([
            (x['name'], x['log10_conc']) in final_odors for x in odor1_dicts
        ])

        n_before = num_notnull(df)

        # TODO refactor to share w/ line below? possible?
        to_null = df.loc[panel_index][~at_final_conc]

        n_new_null = num_notnull(to_null)
        if n_new_null > 0:
            # TODO TODO ideally say which final conc is for each
            msg = f'non-final {panel} concentrations that will be nulled:\n'
            msg += pformat(set(to_null.index.get_level_values('odor1')))

            msg += '\n...for flies:\n'
            msg += '\n'.join([f'- {format_datefly(*x)}' for x in
                to_null.dropna(axis='columns', how='all'
                    ).columns.droplevel('roi').unique()
            ])
            warn(msg)

        # TODO why does this work? still change to something perhaps less fragile?
        # TODO try to break this
        df.loc[panel_index][~at_final_conc] = np.nan

        assert num_notnull(df) == n_before - n_new_null

    return df


# TODO rename all remy->megamat
def plot_remy_drosolf_corr(df, for_filename, for_title, plot_root,
    plot_responses=False) -> None:

    df = abbrev_hallem_odor_index(df)

    # TODO factor into subsetting to megamat odors (-> share w/ model_test.py
    # sensitivity analysis stuff, etc) (maybe combine w/ above abbreviating)
    remy_df = df.loc[panel2name_order['megamat']]

    # TODO TODO it's really -2 though right? say that? (though maybe w/ diff style
    # olfactometer it is comparable?)
    #remy_df.index = remy_df.index.str.cat([' @ -3'] * len(remy_df))
    remy_df.index = remy_df.index.str.cat([' @ -2'] * len(remy_df))

    if plot_responses:
        resp_df = sort_odors(remy_df, panel='megamat')

        # b/c plot_all_roi_mean_responses is picky about .name(s)...
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

    # TODO replace w/ plot_corr from in here [/ in hong2p.viz, if i move there]?
    # TODO share kwargs w/ other places i'm defining them for plotting correlations
    # for remy's experiment
    fig = natmix.plot_corr(remy_corr_da, **remy_corr_matshow_kwargs)
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

    # TODO TODO TODO print columns being discarded
    # ipdb> task_gloms - set(wPNKC2.columns)
    # {'VM6v', 'VP5', 'VM6l', 'VP3', 'VP1l', 'VM6m'}
    # ipdb> set(wPNKC2.columns) - task_gloms
    # {'VP5+Z', 'VP3+VP1l', 'VP1d+VP4', 'VP1m+VP2', 'VP3+', 'VP1m+VP5', 'M'}
    #import ipdb; ipdb.set_trace()
    wPNKC2 = wPNKC2[task_gloms & set(wPNKC2.columns)].copy()

    # TODO sort wPNKC2 so that all hallem stuff is first?

    # TODO TODO delete? not sure this will work. might also want conditional on
    # hallem_input?
    if not _use_matt_wPNKC:
        #import ipdb; ipdb.set_trace()
        wPNKC = wPNKC2
        print('USING PRAT WPNKC (W/ SOME INCONSISTENCIES WRT MATT VERSION)')
    #

    # TODO or do i want it so that all the hallem ones are first, in that order? matter
    # elsewhere?
    # TODO restore. see if i can also sort output in uniform case (b/c if not, might
    # suggest there is other order-of-glomeruli dependence in fit_mb_model THAT THERE
    # SHOULD NOT BE)
    # TODO i can not seem to recreate uniform output (by sorting wPNKC post-hoc), but
    # i'm not sure that's actually a problem. maybe input *should* always just be in a
    # particular order, and shouldn't necessarily matter that it's this one...
    wPNKC = wPNKC.sort_index(axis='columns')

    # TODO delete
    #
    # TODO TODO does this fn (for matt=False case, at least) have deterministic column
    # order in output? if so, what else is re-ordering cols in hemibrain case
    # (see comparison of wPNKC to cache_wPNKC)?
    '''
    hemi_wPNKC_cache = Path(f'DELETEME_hemibrain_wPNKC_matt{_use_matt_wPNKC}.p')
    if hemi_wPNKC_cache.exists():
        # TODO TODO TODO why this False? sort columns before subsetting to task_gloms?
        # TODO TODO TODO order wrt other things it's used against matter? or
        # appropriately indexed (b/c both DFs? enough in my usage context?) after that?
        print('HEMIBRAIN wPNKC equal to cache contents?')
        cache_wPNKC = pd.read_pickle(hemi_wPNKC_cache)
        print(wPNKC.equals(cache_wPNKC))
        #import ipdb; ipdb.set_trace()

    try:
        to_pickle(wPNKC, hemi_wPNKC_cache)
    except AssertionError:
        print('hemibrain wPNKC cache already written this run')
    '''
    #

    return wPNKC


# TODO try to use in filling i do in main (copied/adapted from there)
# TODO also try to use in modelling stuff?
def fill_to_hemibrain(df: pd.DataFrame, value=np.nan, *, verbose=False) -> pd.DataFrame:

    # TODO replace w/ call to hemibrain_wPNKC (w/ _use_matt_wPNKC=False)?
    # TODO + assert data/ subdir CSV matches
    #
    # NOTE: the md5 of this file (3710390cdcfd4217e1fe38e0782961f6) matches what I
    # uploaded to initial Dropbox folder (Tom/hong_depasquale_collab) for Grant from
    # the DePasquale lab.
    #
    # Also matches ALL wPNKC.csv outputs I currently have under modeling output
    # subdirs, except those created w/ _use_matt_wPNKC=True (those wPNKC.csv files
    # have md5 2bc8b74c5cfd30f782ae5c2048126562). Though, none of my current outputs
    # had drop_receptors_not_in_hallem=True, which would lead to a different CSV.
    #
    # Also equal to wPNKC right after call to hemibrain_wPNKC(_use_matt_wPNKC=False)
    prat_hemibrain_wPNKC_csv = \
        'data/sent_to_grant/2024-04-05/connectivity/wPNKC.csv'

    # TODO cache, at least
    wPNKC_for_filling = pd.read_csv(prat_hemibrain_wPNKC_csv).set_index('bodyid')
    wPNKC_for_filling.columns.name = 'glomerulus'

    hemibrain_glomeruli = set(wPNKC_for_filling.columns)
    glomeruli = set(df.columns.get_level_values('roi'))

    hemibrain_glomeruli_not_in_data = hemibrain_glomeruli - glomeruli
    assert len(hemibrain_glomeruli_not_in_data) > 0

    hemibrain_glomeruli_not_in_data = sorted(hemibrain_glomeruli_not_in_data)

    # TODO also print value counts of glomeruli present across my data?
    # TODO delete?
    if verbose:
        print('hemibrain glomeruli not in data:')
        print(hemibrain_glomeruli_not_in_data)
        print()
    #

    df = df.copy()

    # TODO replace w/ just MultiIndex path (assuming code can be adapted to work for
    # both)?
    if not isinstance(df.columns, pd.MultiIndex):
        assert df.columns.name == 'roi'
        # TODO replace w/ reindexing or something like that (just forcing columns to be
        # hemibrain glomeruli, essentially, automatically filling NaN for values not in
        # index previously)? would probably simplify.
        #
        # this adds new columns
        df[hemibrain_glomeruli_not_in_data] = value
    else:
        assert df.columns.names == ['date', 'fly_num', 'roi']
        # TODO easier/cleaner way?
        fly_df_list = []
        for fly, fly_df in df.groupby(level=['date', 'fly_num'], axis='columns'):
            fly_df = fly_df.loc[:, fly].copy()

            # TODO delete?
            fly_glomeruli = set(fly_df.columns.get_level_values('roi'))
            hemibrain_glomeruli_not_in_fly = hemibrain_glomeruli - fly_glomeruli

            if verbose:
                print(fly)
                print('hemibrain glomeruli not in fly:')
                print(hemibrain_glomeruli_not_in_fly)
                print()

            # TODO delete
            # (yup, it fails)
            '''
            try:
                # TODO TODO may need to use hemibrain not in fly instead tho?
                # similar to where i copied this code from?
                # (and then may also need to use some of the more complicated filling logic
                # from where i copied this from...)
                assert hemibrain_glomeruli_not_in_fly == set(
                    hemibrain_glomeruli_not_in_data
                ), 'need to fix if this assertion fails (fill on fly below, +extra logic)'
            except AssertionError:
                import ipdb; ipdb.set_trace()
            '''
            #

            # TODO TODO do i actually want that more complicated filling logic
            # (i seem to be moving closer and closer to that...)?
            # or can i just fill NaN everywhere?

            # NOTE: works w/ `df[...] = x` but NOT `df.loc[...] = x`.
            # TODO why?
            fly_df[sorted(hemibrain_glomeruli_not_in_fly)] = value

            # TODO delete
            # already sorted
            #fly_df[hemibrain_glomeruli_not_in_data] = value

            # NOTE: ROI names no longer sorted at this point

            fly_df = util.addlevel(fly_df, ['date', 'fly_num'], fly, axis='columns')
            fly_df_list.append(fly_df)

        # does not sort the column axis, and no options for that (only for non-concat
        # axis)
        df = pd.concat(fly_df_list, axis='columns', verify_integrity=True)

    roi_names = df.columns.get_level_values('roi')
    certain_mask = [is_ijroi_certain(x) for x in roi_names]

    certain_roi_set = {x for x, c in zip(roi_names, certain_mask) if c}

    certain_rois_not_in_hemibrain = certain_roi_set - hemibrain_glomeruli
    assert len(certain_rois_not_in_hemibrain) == 0, ('certain ROIs '
        f'{certain_rois_not_in_hemibrain} in data, but missing from hemibrain wPNKC'
    )

    # TODO TODO kwarg flag option to drop this non-certain stuff instead?
    if len(set(roi_names) - certain_roi_set) > 0:
        warn('fill_to_hemibrain: have some non-certain (thus non-hemibrain) ROIs. '
            'will not currently be dropped (unimplemented)!'
        )

    # TODO issue that sorting now groups all ROIs of same name together, rather than
    # keeping all data from one fly together (as df.sort_index(axis='columns') did)?
    # probably not...
    #
    # negating mask so non-certain are all at end (rather than all at start)
    df = sort_fly_roi_cols(df, sort_first_on=[not x for x in certain_mask])

    # TODO delete
    # TODO TODO can i only repro this if the second call is in ipdb?
    # (yea, it seems like it... wtf! i'm copy-pasting this line verbatim into ipdb too)
    # TODO TODO this case broken (some levels seem left over from first call), when
    # called twice in a row?
    #print()
    #print('trying to sort again...')
    #df = sort_fly_roi_cols(df, sort_first_on=[
    #    not is_ijroi_certain(x) for x in df.columns.get_level_values('roi')
    #])
    #print()
    #print('THIRD sort...')
    #sort_fly_roi_cols(df, sort_first_on=[
    #    not is_ijroi_certain(x) for x in df.columns.get_level_values('roi')
    #])
    #

    return df


def drop_silent_model_cells(responses: pd.DataFrame) -> pd.DataFrame:
    # .any() checks for any non-zero, so should also work for spike counts
    # (or 1.0/0.0, instead of True/False)
    nonsilent_cells = responses.T.any()

    # TODO maybe restore as warning, or behind a checks flag or something.
    # (also works if index.name == 'model_kc')
    # (commented cause was failing b/c index name was diff in natmix_data/analysis.py
    # script, though i could have changed that...)
    #assert 'model_kc' in nonsilent_cells.index.names

    return responses.loc[nonsilent_cells].copy()


# TODO delete Optional in RHS of return Tuple after implementing in other cases
# TODO if orn_deltas is passed, should we assume we should tune on hallem? or assume we
# should tune on that input?
# TODO rename drop_receptors_not_in_hallem -> glomeruli
# TODO some kind of enum instead of str for pn2kc_connections?
# TODO accept sparsities argument (or scalar avg probably?), for target when tuning
# TODO delete _use_matt_wPNKC after resolving differences wrt Prat's? maybe won't be
# possible though, and still want to be able to reproduce matt's stuff...
# TODO doc that sim_odors is Optional[Set[str]]
# TODO actually, probably can delete sim_odors now? why even have it? to tune on a diff
# set than to return responses for?
# TODO default tune_on_hallem to False?
def fit_mb_model(orn_deltas=None, sim_odors=None, *, tune_on_hallem: bool = True,
    pn2kc_connections: str = 'hemibrain', n_claws: Optional[int] = None,
    drop_multiglomerular_receptors: bool = True,
    drop_receptors_not_in_hallem: bool = False, seed: int = 12345,
    target_sparsity: Optional[float] = None,
    _use_matt_wPNKC=False, _add_back_methanoic_acid_mistake=False,
    fixed_thr: Optional[float] = None,
    wAPLKC: Optional[float] = None, wKCAPL: Optional[float] = None,
    print_olfsysm_log: Optional[bool] = None, plot_dir: Optional[Path] = None,
    make_plots: bool = True, title: str = '',
    drop_silent_cells_before_analyses: bool = True, repro_preprint_s1d: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    # TODO TODO doc point of sim_odors. do we need to pass them in?
    # (even when neither tuning nor running on any hallem data?)
    """
    Args:
        title: only used if plot_dir passed. used as prefix for titles of plots.
        drop_silent_cells_before_analyses: only relevant if `make_plots=True`

        repro_preprint_s1d: whether to add fake odors + return data to allow
            reproduction of preprint figure S1D (showing model response rates to fake
            CO2, fake ms, and real eb)

    Returns responses, wPNKC
    """
    # TODO maybe make it so sim_odors is ignored if orn_deltas is passed in?
    # or err [/ assert same odors as orn_deltas]? would then need to conditionally pass
    # in calls in here...

    pn2kc_connections_options = {'uniform', 'caron', 'hemidraw', 'hemibrain'}
    if pn2kc_connections not in pn2kc_connections_options:
        raise ValueError(f'{pn2kc_connections=} not in {pn2kc_connections_options}')

    if pn2kc_connections == 'caron':
        # TODO TODO support? may need for comparisons to ann's model?
        raise NotImplementedError

    variable_n_claw_options = {'uniform', 'caron', 'hemidraw'}
    variable_n_claws = False
    if pn2kc_connections not in variable_n_claw_options:
        if n_claws is not None:
            raise ValueError(f'n_claws only supported for {variable_n_claw_options}')
    else:
        # TODO also default to averaging over at least a few seeds in all these cases?
        # how much do things actually tend to vary, seed to seed?
        variable_n_claws = True
        if n_claws is None:
            # NOTE: it seems to default to 6 in olfsysm.cpp
            raise ValueError('n_claws must be passed an int if pn2kc_connections in '
                f'{variable_n_claw_options}'
            )

    # TODO TODO rename hallem_input to only_run_on_hallem (or something better)?
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

    if fixed_thr is not None:
        assert target_sparsity is None
        assert wAPLKC is not None, 'for now, assuming both passed if either is'

        # TODO move these values / notes to model_test.py, or wherever sensitivity
        # analysis gets moved to.
        #fixed_thr = 145.973
        # these both break exact similarity to matt's hemimat outputs, as expected:
        #fixed_thr = 140
        #fixed_thr = 147
        mp.kc.fixed_thr = fixed_thr

        mp.kc.add_fixed_thr_to_spont = True
        # actually do need this. may or may not need thr_type='fixed' too
        mp.kc.use_fixed_thr = True
        mp.kc.thr_type = 'fixed'
    else:
        mp.kc.thr_type = 'uniform'

    # TODO after getting model to accept hardcoded wAPLKC and wKCAPL, only do this if
    # not hardcoding those (+ fixed thr)
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

    # TODO TODO TODO how to handle this for stuff not in hallem? matter if it's 0?
    # (currently imputing mean sfr, but maybe try 0 and see if output differs?)

    sfr_col = 'spontaneous firing rate'
    sfr = hallem_orn_deltas[sfr_col]
    assert hallem_orn_deltas.columns[-1] == sfr_col
    hallem_orn_deltas = hallem_orn_deltas.iloc[:, :-1].copy()
    n_hallem_odors = hallem_orn_deltas.shape[1]
    assert n_hallem_odors == 110

    # TODO refactor
    hallem_orn_deltas = abbrev_hallem_odor_index(hallem_orn_deltas, axis='columns')

    # TODO delete
    # TODO any code i'm still using break if hallem_input=True and sim_odors is not
    # passed in? not currently passing in sim_odors anymore...
    # (only 1 call in model_test.py explicitly passes in, and only to check against
    # calls that don't)
    '''
    if hallem_input:
        assert sim_odors is None
        #print('see comment above')
        #import ipdb; ipdb.set_trace()
    '''
    #

    # TODO delete? still want to support?
    # TODO add comment explaining purpose of this block
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

        # since we are appending ' @ -2' to hallem_orn_deltas.columns below
        hallem_sim_odors = [f'{n} @ -2' for n in hallem_sim_odors]

    # TODO factor to drosolf.orns?
    assert hallem_orn_deltas.columns.name == 'odor'
    hallem_orn_deltas.columns += ' @ -2'

    # so that glomerulus order in Hallem CSVs will match eventual wPNKC output (which
    # has glomeruli sorted)
    #
    # making a copy to sort by glomeruli, since that would break an assertion later
    # (comparing against mp.orn internal data), if I sorted source variables.
    hallem_orn_deltas_for_csv = hallem_orn_deltas.sort_index(axis='index')
    sfr_for_csv = sfr.sort_index()

    if hallem_delta_csv.exists():
        assert hallem_sfr_csv.exists()
        # TODO or just save to root, but only do so if not already there? and load and
        # check against that otherwise? maybe save to ./data

        # TODO could just load first time we reach this (per run of script)...
        deltas_from_csv = pd.read_csv(hallem_delta_csv, index_col='glomerulus')
        sfr_from_csv = pd.read_csv(hallem_sfr_csv, index_col='glomerulus')

        deltas_from_csv.columns.name = 'odor'

        assert sfr_from_csv.shape[1] == 1
        sfr_from_csv = sfr_from_csv.iloc[:, 0].copy()

        assert sfr_for_csv.equals(sfr_from_csv)
        # changing abbreviations of some odors broke this previously
        # (hence why i replaced it w/ the two assertions below. now ignoring odor
        # columns)
        #assert hallem_orn_deltas_for_csv.equals(deltas_from_csv)
        assert np.array_equal(hallem_orn_deltas_for_csv, deltas_from_csv)
        assert hallem_orn_deltas_for_csv.index.equals(deltas_from_csv.index)
    else:
        # TODO assert columns of the two match here (so i don't need to check from
        # loaded versions, and so i can only check one against wPNKC, not both)
        to_csv(hallem_orn_deltas_for_csv, hallem_delta_csv)
        to_csv(sfr_for_csv, hallem_sfr_csv)

        # TODO delete? unused
        #deltas_from_csv = hallem_orn_deltas_for_csv.copy()
        #sfr_from_csv = sfr_for_csv.copy()
        #

    del hallem_orn_deltas_for_csv, sfr_for_csv

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

    # TODO TODO (delete?) implement means of getting threshold from hallem input +
    # hallem glomeruli only -> somehow applying that threshold [+APL inh?] globally (and
    # running subsequent stuff w/ all glomeruli, including non-hallem ones) (even
    # possible?)
    # TODO TODO now that i can just hardcode the 2 params, can i make plots where i
    # "tune" on hallem and then apply those params to the model using my data as input,
    # w/ all glomeruli (or does it still not make sense to use the same global params,
    # w/ new PNs w/ presumably new spontaneous input now there? think it might not make
    # sense...)
    # TODO delete
    '''
    if not hallem_input and tune_on_hallem:
        # (think i always have this True when tune_on_hallem=True, at the moment, but if
        # i can do what i'm asking in comment above, could try letting this be False
        # while tune_on_hallem=True, for input that has more glomeruli than in Hallem)
        print(f'{drop_receptors_not_in_hallem=}')
        import ipdb; ipdb.set_trace()
    '''
    #

    # TODO TODO check that nothing else depends on order of columns (glomeruli) in these
    wPNKC = hemibrain_wPNKC(_use_matt_wPNKC=_use_matt_wPNKC)
    glomerulus_index = wPNKC.columns

    if not hallem_input:
        zero_filling = (~ glomerulus_index.isin(orn_deltas.index))
        if zero_filling.any():
            msg = 'zero filling spike deltas for glomeruli not in data:'
            # TODO TODO sort by glomerulus (and elsewhere)
            # TODO refactor to share printing w/ other similar code?
            # TODO TODO condense this warning to one line when reasonably possible
            msg += '\n- '.join([''] + [f'{g}' for g in glomerulus_index[zero_filling]])
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
        glomeruli_missing_in_wPNKC = input_glomeruli - set(glomerulus_index)
        if len(glomeruli_missing_in_wPNKC) > 0:
            # TODO assert False? seems we could do that at least for megamat data...
            warn('dropping glomeruli not in wPNKC (while zero-filling): '
                f'{glomeruli_missing_in_wPNKC}'
            )

        if tune_on_hallem:
            # TODO make sure we aren't writing wPNKC in this case (and maybe not other
            # hallem CSVs? they are probably fine either way...)
            hallem_not_in_wPNKC = set(hallem_orn_deltas.index) - set(glomerulus_index)
            assert len(hallem_not_in_wPNKC) == 0 or hallem_not_in_wPNKC == {'DA4m'}, \
                f'unexpected {hallem_not_in_wPNKC=}'

            if len(hallem_not_in_wPNKC) > 0:
                warn(f'dropping glomeruli not in wPNKC {hallem_not_in_wPNKC} from '
                    'Hallem data to be used for tuning'
                )

            # this will be concatenated with orn_deltas below, and we don't want to add
            # back the glomeruli not in wPNKC
            hallem_orn_deltas = hallem_orn_deltas.loc[
                [c for c in hallem_orn_deltas.index if c in glomerulus_index]
            ].copy()

        orn_deltas_pre_filling = orn_deltas.copy()

        # TODO simplify this. not a pandas call for it? reindex_like seemed to not
        # behave as expected, but maybe it's for something else / i was using it
        # incorrectly
        # TODO just do w/ pd.concat? or did i want shape to match hallem exactly in that
        # case? matter?
        # TODO reindex -> fillna?
        orn_deltas = pd.DataFrame([
                orn_deltas.loc[x].values if x in orn_deltas.index
                # TODO correct? after concat across odors in tune_on_hallem=True case?
                else np.zeros(len(orn_deltas.columns))
                for x in wPNKC.columns
            ], index=glomerulus_index, columns=orn_deltas.columns
        )

        # TODO need to be int (doesn't seem so)?
        mean_sfr = sfr.mean()
        # TODO TODO also warn about this happening (also try imputing w/ 0?)

        sfr = pd.Series(index=glomerulus_index,
            data=[(sfr.loc[g] if g in sfr else mean_sfr) for g in glomerulus_index]
        )
        assert sfr.index.equals(orn_deltas.index)
    #

    odor_index = orn_deltas.columns
    n_input_odors = orn_deltas.shape[1]


    extra_orn_deltas = None
    # TODO delete/comment after i'm done?
    #'''
    eb_mask = orn_deltas.columns.get_level_values('odor').str.startswith('eb @')
    # should be true for megamat and hallem
    if repro_preprint_s1d and eb_mask.sum() == 1:
        # TODO TODO finish support for "extra" odors (to be simmed, but not tuned on)
        # (expose as kwarg eventually prob)
        # (would not be conditional on eb if so... just a hack to skip validation, and
        # only want S1D if we do have eb, as thats what preprint one used)

        # TODO try to move up above any modifications to orn_deltas (mainly the
        # glomeruli filling above) after getting it to work down here? (why? just to
        # make easier to convert to kwarg?)

        fake_odors = ['fake ms @ 0']
        if 'V' in orn_deltas.index:
            fake_odors.append('fake CO2 @ 0')

        # should only be in 'hallem' cases
        else:
            # TODO TODO how did matt handle this? (not in wPNKC I'm currently using in
            # Hallem case) (pretty sure most of his modelling is done w/o 'V' (or any
            # non-Hallem glomeruli) in wPNKC. so what is he doing for wPNKC here? and
            # what data is he using for the non-Hallem glomeruli for tuning?)
            #
            # (not doing fake-CO2 in 'hallem' context for now)
            warn("glomerulus 'V' not in wPNKC, so not adding fake CO2!")

        # TODO convert to kwarg -> def in model_mb... and pass in thru there?
        extra_orn_deltas = pd.DataFrame(index=orn_deltas.index, columns=fake_odors,
            data=0
        )
        assert 'odor' in orn_deltas.columns.names
        extra_orn_deltas.columns.name = 'odor'
        # TODO handle appending ' @ 0' automatically if needed (only if i actually
        # expose extra_orn_deltas as a kwarg)?

        extra_orn_deltas.loc['DL1', 'fake ms @ 0'] = 300
        if 'V' in orn_deltas.index:
            extra_orn_deltas.loc['V', 'fake CO2 @ 0'] = 300

        eb_deltas = orn_deltas.loc[:, eb_mask]

        if 'panel' in eb_deltas.columns.names:
            eb_deltas = eb_deltas.droplevel('panel', axis='columns')

        assert eb_deltas.shape[1] == 1
        eb_deltas = eb_deltas.iloc[:, 0]
        assert len(eb_deltas) == len(extra_orn_deltas)

        # eb_deltas.name will be like 'eb @ -3' (previous odor columns level value)
        extra_orn_deltas[eb_deltas.name] = eb_deltas

        if sim_odors is not None:
            assert hallem_input
            # sim_odors contents not used for anything other than internal plots below,
            # so sufficient to only grow hallem_sim_odors here (which is used to subset
            # responses right before returning, and for nothing else [past this point at
            # least])
            #
            # not also growing by extra `eb_deltas.name`, because that gets removed
            # before hallem_sim_odors used (extra eb is not returned in hallem or any
            # other case. only checked against responses to existing eb col)
            hallem_sim_odors.extend(fake_odors)
    #'''

    n_extra_odors = 0
    if extra_orn_deltas is not None:
        n_extra_odors = extra_orn_deltas.shape[1]

        if 'panel' in orn_deltas.columns.names:
            assert 'extra' not in set(orn_deltas.columns.get_level_values('panel'))
            extra_orn_deltas = util.addlevel(extra_orn_deltas, 'panel', 'extra',
                axis='columns'
            )

        # TODO assert row index unchanged and column index up-to-old-length too?
        #
        # removed verify_integrity=True since there is currently duplicate 'eb' in
        # hallem case (w/o 'panel' level to disambiguate) (only when adding extra odors)
        orn_deltas = pd.concat([orn_deltas, extra_orn_deltas], axis='columns')

        odor_index = orn_deltas.columns

        # TODO make sure we aren't overwriting either of these below before running!
        mp.kc.tune_from = range(n_input_odors)
        mp.sim_only = range(n_input_odors)


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

        # TODO TODO adapt to work w/ panel col level? what to use for hallem? 'hallem'?
        # None?
        # TODO assert this concat doesn't change odor (col) index? inspect what it's
        # doing to sanity check?
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

        # TODO delete? not sure if it's triggered outside of case where i accidentally
        # passed input where all va/aa stuff was dropped (by calling script w/
        # 2023-04-22 as end of date range, rather than start)
        try:
            # TODO support input w/ panel level on odor index / delete
            #
            # TODO if i wanna keep this, move earlier (or at least have another version
            # of this earlier? maybe in one of first lines in fit_mb_model, or in
            # whatever is processing orn_deltas before it's passed to fit_mb_model?)
            # (the issue seems to be created before we get into fit_mb_model)
            assert sim_odors is None or sim_odors == set(
                odor_index.get_level_values('odor')
            ), 'why'
        except AssertionError:
            import ipdb; ipdb.set_trace()

    if not hallem_input:
        # TODO maybe i should still have an option here to tune on more data than what i
        # ultimately return (perhaps including diagnostic data? though they prob don't
        # have representative KC sparsities either...)

        # TODO TODO try to implement other strategies where we don't need to throw
        # away input glomeruli/receptors
        # (might need to make my own gkc_wide csv equivalent, assuming it only contains
        # the connections involving the hallem glomeruli)
        # (also, could probably not work in the tune_on_hallem case...)

        # TODO delete here (already moved into conditional below)
        #hallem_glomeruli = hallem_orn_deltas.index

        # TODO TODO raise NotImplementedError/similar if tune_on_hallem=True,
        # not hallem_input, and not drop_receptors_not_in_hallem?

        # TODO TODO maybe this needs to be True if tune_on_hallem=True? at least as
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
                # TODO warn differently (/only?) for stuff that was actually in our
                # input data, and not just zero filled above?
                msg = 'dropping glomeruli not in Hallem:'
                # TODO sort on glomeruli names (seems it already is. just from hallem
                # order? still may want to sort here to ensure)
                msg += '\n- '.join([''] + [f'{g} ({r})' for g, r in
                    zip(orn_deltas.index[receptors_not_in_hallem],
                        receptors[receptors_not_in_hallem])
                ])
                msg += '\n'
                warn(msg)

            orn_deltas = orn_deltas[~receptors_not_in_hallem].copy()
            sfr = sfr[~receptors_not_in_hallem].copy()

            # TODO refactor to not use glomerulus index, and just always use
            # wPNKC.columns, to not have to deal w/ the two separately? (here and
            # elsewhere...)
            assert glomerulus_index.equals(wPNKC.columns)
            glomerulus_index = glomerulus_index[~receptors_not_in_hallem].copy()
            wPNKC = wPNKC.loc[:, ~receptors_not_in_hallem].copy()

        # TODO TODO another option to use this input for fitting thresholds (+ APL
        # inhibition), w/o using hallem at all?

        # TODO TODO am i not seeing inhibition to the extent that i might expect by
        # comparing the deltas to hallem? is it something i can improve by changing my
        # dF/F -> spike delta estimation process, or is it a limitation of my
        # measurements / the differences between the hallem data and ours

    # TODO TODO probably still support just one .name == 'odor' tho...
    # (esp for calls w/ just hallem input, either old ones here or model_test.py?)
    # TODO move earlier?
    assert orn_deltas.columns.name == 'odor' or (
        orn_deltas.columns.names == ['panel', 'odor']
    )

    # TODO would we or would we not have removed it in that case? and what about
    # pratyush wPNKC case?
    # If using Matt's wPNKC, we may have removed this above:
    if 'DA4m' in hallem_orn_deltas.index:
        assert np.array_equal(hallem_orn_deltas, mp.orn.data.delta)

        if hallem_input:
            # TODO just do this before we would modify sfr (in that one branch above)?
            assert np.array_equal(sfr, mp.orn.data.spont[:, 0])

    # TODO TODO merge da4m/l hallem data (pretty sure they are both in my own wPNKC?)?
    # TODO TODO do same w/ 33b (adding it into 47a and 85a Hallem data, for DM3 and DM5,
    # respectively)?

    # TODO TODO add comment explaining circumstances when we wouldn't have this.  it
    # seems to be zero filled (presumably just b/c in wPNKC earlier, and i think that's
    # the case whether _use_matt_wPNKC is True or False). maybe just in non-hemibrain
    # stuff? can i assert it's always true and delete some of this code?
    # TODO TODO TODO only drop DA4m if it's not in wPNKC (which should only be if
    # _use_matt_wPNKC=False?)?
    #
    # We may have already implicitly dropped this in the zero-filling code
    # (if that code ran, and if wPNKC doesn't have DA4m in its columns)
    have_DA4m = 'DA4m' in sfr.index or 'DA4m' in orn_deltas.index

    # TODO replace by just checking one for have_DA4m def above, w/ an assertion the
    # indices are (still) equal here?
    if have_DA4m:
        assert 'DA4m' in sfr.index and 'DA4m' in orn_deltas.index

    # TODO delete
    if not have_DA4m:
        print()
        print('did not have DA4m in sfr.index. add comment explaining current input')
        import ipdb; ipdb.set_trace()
    #

    # TODO also only do if _use_matt_wPNKC=True (prat's seems to have DA4m...)?
    #if (hallem_input or tune_on_hallem) and have_DA4m:
    # TODO this aligned with what i want?
    # TODO revert to using wPNKC.columns instead of glomerulus_index, for clarity?
    if 'DA4m' not in glomerulus_index and have_DA4m:
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

        # TODO TODO also remove DA4m from orn_deltas_pre_filling?
        # (maybe just subset to what's in sfr/orn_deltas but not orn_deltas_pre_filling,
        # but down by usage of orn_deltas_pre_filling?)

    # TODO don't do if 'uniform' draw path (/ cxn_distrib, but check that there)?
    assert sfr.index.equals(orn_deltas.index)
    # TODO TODO warn here? this always OK?
    # TODO what is purpose here? comment explaining

    # TODO delete
    _wPNKC_shape_changed = False
    if wPNKC.shape != wPNKC[sfr.index].shape:
        print()
        print(f'wPNKC shape BEFORE subsetting to sfr.index: {wPNKC.shape}')
        _wPNKC_shape_changed = True
    #

    # TODO TODO also need to subset glomerulus_index here now? just always use
    # wPNKC.columns and remove glomerulus_index?
    wPNKC = wPNKC[sfr.index].copy()

    # TODO delete
    if _wPNKC_shape_changed:
        # TODO TODO is this only triggered IFF have_DA4m? move all this wPNKC stuff into
        # that conditional above?
        print(f'wPNKC shape AFTER subsetting to sfr.index: {wPNKC.shape}')
        print()
        print('NEED TO SUBSET GLOMERULUS_INDEX HERE (/ refactor to just use wPNKC)?')
        import ipdb; ipdb.set_trace()
        print()
    del _wPNKC_shape_changed
    #

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

        # TODO delete?
        if hallem_input:
            # TODO compute this from something?
            n_hallem_glomeruli = 23
            assert mp.kc.cxn_distrib.shape == (1, n_hallem_glomeruli)
        #

        # TODO TODO TODO what currently happens if using # glomeruli other > hallem?
        # seems like it may already be broken? (and also in uniform case. not sure if
        # this is why tho)
        #
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

        wPNKC = None

    rv = osm.RunVars(mp)

    # TODO need delete=False?
    temp_log_file = NamedTemporaryFile(suffix='.olfsysm.log', delete=False)

    # also includes directory
    temp_log_path = temp_log_file.name

    if print_olfsysm_log is None:
        print_olfsysm_log = verbose

    if print_olfsysm_log:
        print(f'writing olfsysm log to {temp_log_path}')

    try:
        # TODO should i only be doing this right before running? it causing issues?
        #
        # it seems to just append to this file, if it already exists (should no longer
        # an issue now that I'm making temp files)
        rv.log.redirect(temp_log_path)

    # just so i can experiment w/ reverting to old olfsysm, before i added this
    except AttributeError:
        # TODO is this currently the path being taken?
        pass

    # may or may not care to relax this later
    # (so that we can let either be defined from one, or to try varying separately)
    if wAPLKC is None and wKCAPL is not None:
        raise NotImplementedError('wKCAPL can only be specified if wAPLKC is too')

    def _single_unique_val(arr: np.ndarray) -> float:
        """Returns single unique value from array.

        Raises AssertionError if array has more than one unique value (including NaNs).
        """
        unique_vals = set(np.unique(arr))
        assert len(unique_vals) == 1
        return unique_vals.pop()

    # TODO TODO double check olfsysm flag that used to be called enable_apl
    # actually only affected tuning, and that my hardcoded weights are still being used
    # TODO TODO double check olfsysm tuning is only changing the 3 params i'm
    # hardcoding (and especially in step that goes from 0.2 to 0.1 response rate)
    if wAPLKC is not None:
        assert target_sparsity is None
        assert fixed_thr is not None, 'for now, assuming both passed if either is'

        mp.kc.tune_apl_weights = False

        # TODO delete
        #print(f'(hardcoded via kwarg) {wAPLKC=}')

        # NOTE: min/max for these should all be the same. they are essentially scalars,
        # at least as tuned before
        # rv.kc.wKCAPL.shape=(1, 1630)
        # rv.kc.wKCAPL.max()=0.002386503067484662
        # rv.kc.wAPLKC.shape=(1630, 1)
        # rv.kc.wAPLKC.max()=3.8899999999999992
        #
        # TODO still need allclose/similar (at output)?
        # TODO TODO specify one by dividing other by / double(p.kc.N)
        #wAPLKC = 3.8899999999999992
        rv.kc.wAPLKC = np.ones((mp.kc.N, 1)) * wAPLKC

        # TODO TODO (delete) what is required to go from 0.2 to 0.1 response rate again
        # (and why am i having such a hard time finding other values that will do
        # that?)?

        if wKCAPL is not None:
            # TODO delete
            #print(f'(hardcoded via kwarg) {wKCAPL=}')

            rv.kc.wKCAPL = np.ones((1, mp.kc.N)) * wKCAPL

            # TODO delete
            '''
            calc_wKCAPL = wAPLKC / mp.kc.N
            print(f'{calc_wKCAPL=}')
            print(f'{wKCAPL - calc_wKCAPL=}')
            '''
            #
        else:
            wKCAPL = wAPLKC / mp.kc.N
            # this shape should be correct (wAPLKC's shape is transposed, so defining
            # this from that matrix could have caused issues?). passing transpose of
            # this in does NOT work correctly (how to get model to recognize that?).
            rv.kc.wKCAPL = np.ones((1, mp.kc.N)) * wKCAPL

        # TODO TODO try setting wAPLKC = 1 (or another reasonable constant), and only
        # vary wKCAPL?
        # (or probably vice versa, where wKCAPL = 1 / mp.kc.N, and wAPLKC varies)

        # TODO save/print APL activity (timecourse?) to check it's reasonable?
        # (but it's non-spiking... what is reasonable?)

    if pn2kc_connections == 'hemibrain':
        rv.kc.wPNKC = wPNKC

    osm.run_ORN_LN_sims(mp, rv)
    osm.run_PN_sims(mp, rv)

    before_any_tuning = time.time()

    # This is the only place where build_wPNKC and fit_sparseness are called, and they
    # are only called if the 3rd parameter (regen=) is True.
    osm.run_KC_sims(mp, rv, True)

    tuning_time_s = time.time() - before_any_tuning

    # TODO is it all zeros after the n_hallem odors?
    # TODO do responses to first n_hallem odors stay same after changing sim_only and
    # re-running below?
    # Of shape (n_kcs, n_odors). odors as columns, as elsewhere.
    responses = rv.kc.responses.copy()
    responses_after_tuning = responses.copy()

    spike_counts = rv.kc.spike_counts.copy()

    if extra_orn_deltas is not None:
        # TODO TODO maybe just sim the last bit and concat to existing responses,
        # instead of re-running all
        #mp.sim_only = range(n_input_odors, n_input_odors + n_extra_odors)
        mp.sim_only = range(n_input_odors + n_extra_odors)

        osm.run_ORN_LN_sims(mp, rv)
        osm.run_PN_sims(mp, rv)
        # Don't want to do either build_wPNKC or fit_sparseness here (after tuning)
        osm.run_KC_sims(mp, rv, False)

        # TODO also .copy() for both here (or don't above), for consistency
        responses = rv.kc.responses
        spike_counts = rv.kc.spike_counts

        assert np.array_equal(
            responses_after_tuning[:, :n_input_odors], responses[:, :n_input_odors]
        )


    if tune_on_hallem and not hallem_input:
        # TODO TODO fix!
        # ./al_analysis.py -d pebbled -n 6f -t 2023-04-22 -e 2023-06-22 -s
        #   ijroi,intensity,corr
        # ...
        # fitting model (responses_to='pebbled', tune_on_hallem=True,
        #   drop_receptors_not_in_hallem=True, pn2kc_connections=hemibrain,
        #   target_sparsity=0.03)...
        # ...
        # AssertionError
        try:
            assert (responses[:, n_hallem_odors:] == 0).all()
        except AssertionError:
            print('TRIGGERED ABOVE ASSERTIONERROR')
            import ipdb; ipdb.set_trace()

        # TODO also assert in here that sim_odors is None or sim_odors == odor_index?
        # (or move that assertion, which should be somewhere above, outside other
        # conditionals)

        mp.sim_only = range(n_hallem_odors,
            n_hallem_odors + n_input_odors + n_extra_odors
        )

        osm.run_ORN_LN_sims(mp, rv)
        osm.run_PN_sims(mp, rv)

        # Don't want to do either build_wPNKC or fit_sparseness here (after tuning)
        osm.run_KC_sims(mp, rv, False)

        # TODO also .copy() for both here (or don't above), for consistency
        responses = rv.kc.responses
        spike_counts = rv.kc.spike_counts

        # TODO TODO in hallem/hemibrain (or hemidraw???) case, add fake MS / CO2 odor
        # responses (300Hz in each cognate glomerulus) -> check i can recreate S1D? i
        # assume it's all actually from hemibrain (despite what legend says)? ms input
        # actually is just 300Hz, or did they use hallem data there?

        assert np.array_equal(
            responses_after_tuning[:, :n_hallem_odors], responses[:, :n_hallem_odors]
        )

        # TODO also test where appended stuff has slightly diff number of odors than
        # hallem (maybe missing [one random/first/last] row?)
        responses = responses[:, n_hallem_odors:]
        spike_counts = spike_counts[:, n_hallem_odors:]


    temp_log_file.close()

    if print_olfsysm_log:
        print('olfsysm log:')
        log_txt = Path(temp_log_path).read_text()
        cprint(log_txt, 'light_yellow')

    Path(temp_log_path).unlink()

    if fixed_thr is not None:
        # just checking what we set above hasn't changed
        assert mp.kc.fixed_thr == fixed_thr
        assert mp.kc.add_fixed_thr_to_spont == True
        # actually do need this. may or may not need thr_type='fixed' too
        assert mp.kc.use_fixed_thr == True
        assert mp.kc.thr_type == 'fixed'
    else:
        thr = rv.kc.thr
        try:
            # line that would trigger the AttributeError
            spont_in = rv.kc.spont_in

            # this should correspond to the thr_const variable inside
            # olfsysm.choose_KC_thresh_uniform (and can be set by passing as the
            # fixed_thr kwarg to this function, which will also set
            # mp.kc.add_fixed_thr_to_spont=True)
            unique_fixed_thrs = np.unique(thr - 2*spont_in)
            # could take median instead? shouldn't really matter tho...
            fixed_thr = unique_fixed_thrs[0]
            assert np.allclose(fixed_thr, unique_fixed_thrs)

            # TODO return / put in input dict instead (/ too)?
            print(f'fixed_thr: {fixed_thr}')

        # to allow trying older versions of olfsysm, that didn't have rv.kc.spont_in
        except AttributeError:
            pass

    # these should either be the same as any hardcoded wAPLKC [+ wKCAPL] inputs, or the
    # values chosen by the tuning process. _single_unique_val will raise AssertionError
    # if the input arrays contain more than one unique value.
    rv_scalar_wAPLKC = _single_unique_val(rv.kc.wAPLKC)
    rv_scalar_wKCAPL = _single_unique_val(rv.kc.wKCAPL)

    if wAPLKC is not None:
        # TODO delete? just checking what we set above hasn't changed
        assert mp.kc.tune_apl_weights == False

        assert rv_scalar_wAPLKC == wAPLKC

        # this should now be defined whenever wAPLKC is, whether passed in or not...
        assert wKCAPL is not None
        assert rv_scalar_wKCAPL == wKCAPL
    else:
        # TODO delete prints?
        print(f'wAPLKC: {rv_scalar_wAPLKC}')
        print(f'wKCAPL: {rv_scalar_wKCAPL}')
        wAPLKC = rv_scalar_wAPLKC
        wKCAPL = rv_scalar_wKCAPL

    param_dict = {
        'fixed_thr': fixed_thr,
        'wAPLKC': wAPLKC,
        'wKCAPL': wKCAPL,
    }

    tuning_dict = {
        # parameters relevant to model threshold + APL tuning process
        'sp_acc': mp.kc.sp_acc,
        'max_iters': mp.kc.max_iters,
        'sp_lr_coeff': mp.kc.sp_lr_coeff,
        'apltune_subsample': mp.kc.apltune_subsample,

        # should be how many iterations it took to tune,
        'tuning_iters': rv.kc.tuning_iters,

        # removed tuning_time_s from this, because it would cause -c checks to fail
    }
    if fixed_thr is None:
        print('tuning parameters:')
        pprint(tuning_dict)
        print('tuning time: {tuning_time_s:.1f}s')
        print()

    param_dict = {**param_dict, **tuning_dict}

    assert responses.shape[1] == (n_input_odors + n_extra_odors)
    responses = pd.DataFrame(responses, columns=odor_index)
    responses.index.name = 'model_kc'

    assert spike_counts.shape[1] == (n_input_odors + n_extra_odors)
    assert len(responses) == len(spike_counts)
    spike_counts = pd.DataFrame(spike_counts, columns=odor_index)
    spike_counts.index.name = 'model_kc'

    if extra_orn_deltas is not None:
        extra_responses = responses.iloc[:, -n_extra_odors:]

        if 'panel' in extra_responses.columns.names:
            extra_responses = extra_responses.droplevel('panel', axis='columns')

        old_eb = responses.iloc[:, :-n_extra_odors].loc[:, eb_mask]
        if 'panel' in responses.columns.names:
            old_eb = old_eb.droplevel('panel', axis='columns')

        assert old_eb.shape[1] == 1
        old_eb = old_eb.iloc[:, 0]

        eb_idx = -1

        new_eb = extra_responses.iloc[:, eb_idx]
        assert new_eb.name.startswith('eb @')

        assert new_eb.equals(old_eb)

        # just removing eb, so there won't be that duplicate, which could cause some
        # problems later (did cause some of the plotting code in here to fail i think).
        # doesn't matter now that we know new and old are equal.
        responses = responses.iloc[:, :eb_idx].copy()
        spike_counts = spike_counts.iloc[:, :eb_idx].copy()

        # TODO delete? am i not removing 'eb' now anyway?
        if make_plots:
            # causes errors re: duplicate ticklabels in some of the orn_deltas plots
            # currently (would need to remove 'eb' from all of those plots, but also
            # prob want to remove all extra_orn_deltas odors for them. still need to
            # keep in returned responses tho)
            warn('fit_mb_model: setting make_plots=False since not currently supported '
                'in extra_orn_deltas case'
            )
        make_plots = False
        #

    # TODO delete
    # TODO TODO was spontaneous activity of PNs not in hallem handled reasonably
    # (when using my data as input, where some glomeruli not in hallem are there, and
    # where presumably we are also using a wPNKC/rest-of-model that preserves these
    # additional channels)
    # TODO currently mean sfr imputed, which seems ok. try setting to 0 tho, and
    # see if it helps w/ low end corr issue?
    # TODO TODO what do ann/matt do in cases like this? presumably they sometimes also
    # use model ORN/PNs not in hallem (even if what they do is pretty simple)?

    if variable_n_claws:
        wPNKC = rv.kc.wPNKC

        # TODO should these not also be the case in variable_n_claws == False case?
        # move these two assertions out?
        assert wPNKC.shape[1] == len(glomerulus_index)
        assert len(wPNKC) == len(responses)

        wPNKC = pd.DataFrame(data=wPNKC, index=responses.index,
            columns=glomerulus_index
        )

        # TODO delete
        assert mp.kc.seed == seed

    # TODO why is this seemingly a list of arrays, while the equiv kc variable seems to
    # be an array immediately? binding code seems similar...
    orn_sims = np.array(rv.orn.sims)
    # orn_sims.shape=(110, 22, 5500)

    n_samples = orn_sims.shape[-1]
    # from default parameters:
    # p.time.pre_start  = -2.0;
    # p.time.start      = -0.5;
    # p.time.end        = 0.75;
    # p.time.stim.start = 0.0;
    # p.time.stim.end   = 0.5;
    # p.time.dt         = 0.5e-3;
    assert (n_samples * mp.time_dt) == (mp.time_end - mp.time_pre_start)
    assert (mp.time_pre_start < mp.time_start < mp.time_stim_start < mp.time_stim_end <
        mp.time_end
    )

    # also a list out of the box
    # pn_sims.shape=(110, 22, 5500)
    pn_sims = np.array(rv.pn.pn_sims)
    assert pn_sims.shape[-1] == n_samples

    ts = np.linspace(mp.time_pre_start, mp.time_end, num=n_samples)

    # TODO delete
    # units seems to be firing rates (absolute i think. actually, there are some
    # negative values, even in hallem_input=True [w/ matt config] case. that's not a
    # mistake though, is it?)
    '''
    dl5 = glomerulus_index.get_loc('DL5')
    t2h = odor_index.get_loc('t2h @ -2')

    fig, ax = plt.subplots()

    ax.plot(ts, orn_sims[t2h, dl5, :], label='ORN')
    ax.plot(ts, pn_sims[t2h, dl5, :], label='PN')

    ax.axvline(mp.time_stim_start, label='stim start')
    ax.axvline(mp.time_stim_end, label='stim end')

    ax.set_title('DL5 response to t2h')
    ax.legend()
    '''
    # TODO TODO so should i average starting a bit after stim start? seems it still take
    # a bit to peak. what did matt do?
    # in (DL5, t2h) case, PN peak is ~0.047, and ORN plateaus after ~0.077

    stim_start_idx = np.searchsorted(ts, mp.time_stim_start)
    stim_end_idx = np.searchsorted(ts, mp.time_stim_end)

    # TODO stim_end_idx + 1?
    orn_df = pd.DataFrame(index=odor_index, columns=glomerulus_index,
        data=orn_sims[:, :, stim_start_idx:stim_end_idx].mean(axis=-1)
    )
    pn_df = pd.DataFrame(index=odor_index, columns=glomerulus_index,
        data=pn_sims[:, :, stim_start_idx:stim_end_idx].mean(axis=-1)
    )

    if sim_odors is not None:
        input_odor_names = {olf.parse_odor_name(x) for x in sim_odors}
    else:
        input_odor_names = {
            olf.parse_odor_name(x) for x in odor_index.get_level_values('odor')
        }

    # TODO may want to discard some for some plots (e.g. in cases when input is not
    # hallem, but also has diagnostics / fake odors added in addition to megamat odors)?
    # (kinda like plots as is there actually, but may want to add lines to separate
    # megamat from rest?)
    #
    # OK if we have more odors (for Hallem input case, right?)
    megamat = len(megamat_odor_names - input_odor_names) == 0
    del input_odor_names
    if megamat:
        # TODO assert 'panel' only megamat if we do have it?
        panel = None if 'panel' in orn_deltas.columns.names else 'megamat'

        if not hallem_input:
            # at least as configured now, this isn't doing anything.
            # TODO just assert that all of orn_deltas_pre_filling.index are still in
            # orn_deltas.index for now (commentin this)?
            orn_deltas_pre_filling = orn_deltas_pre_filling.loc[
                [g for g in orn_deltas_pre_filling.index if g in orn_deltas.index]
            ].copy()

            # TODO can i rely on panel already being there now? just remove all this
            # sorting? (or sort orn_deltas before, unconditionally)
            # (i assume it's not there when input is hallem?)
            # (but still want to sort when input is hallem, adding megamat, so all
            # megamat odors are first)

            orn_deltas_pre_filling = sort_odors(orn_deltas_pre_filling, panel=panel,
                warn=False
            )

        orn_deltas = sort_odors(orn_deltas, panel=panel, warn=False)

        # TODO maybe only do this one on a copy we don't return? probably don't really
        # care if it's already sorted tho...
        # TODO even care to do this? if just for a corr diff thing, even need?
        responses = sort_odors(responses, panel=panel, warn=False)
        spike_counts = sort_odors(spike_counts, panel=panel, warn=False)

        orn_df = sort_odors(orn_df, panel=panel, warn=False)
        pn_df = sort_odors(pn_df, panel=panel, warn=False)
    else:
        # TODO delete if i modify below to also make plots for other panels
        # (replacing w/ sorting -> dropping levels for all these, as above)
        #
        # the only place these should all be used is in the plotting code below, which
        # currently only runs in `megamat == True` case
        del orn_deltas, orn_df, pn_df

    # TODO still want?
    if 'panel' in responses.columns.names:
        assert 'panel' in spike_counts.columns.names
        responses = responses.droplevel('panel', axis='columns')
        spike_counts = spike_counts.droplevel('panel', axis='columns')

    # TODO probably also do for other panels?
    #if plot_dir is not None and make_plots:
    if (plot_dir is not None and make_plots) and megamat:
        orn_df = orn_df.T
        pn_df = pn_df.T

        # TODO probably also a seperate plot including any hallem deltas tuned on (but
        # not in responses that would be returned). not that i actualy use that path
        # now...

        plot_dir = plot_dir / 'model_internals'
        plot_dir.mkdir(exist_ok=True)

        fig, _ = viz.matshow(sfr.to_frame(), xtickrotation='horizontal',
            **diverging_cmap_kwargs
        )
        savefig(fig, plot_dir, 'sfr')

        # NOTE: assuming input title already reflects whether we are dropping silent
        # cells (which it should) (or that we don't care that title shows it).
        # this could be confusing for ORN/PN plots, which currently are UNAFFECTED by
        # this flag. only silent model KCs are ever dropped b/c of this flag!

        def _plot_internal_responses_and_corrs(subset_fn=lambda x: x, suffix='',
            **plot_kws):

            # TODO TODO maybe subset these down to things that are still in
            # orn_deltas/sfr/wPNKC though?
            #
            # orn_deltas_pre_filling only defined in this case. otherwise, there
            # shouldn't really *be* any filling.
            if not hallem_input:
                orn_delta_prefill_corr = plot_responses_and_corr(
                    subset_fn(orn_deltas_pre_filling), plot_dir,
                    f'orn-deltas-prefill{suffix}', title=title, **plot_kws
                )
            else:
                orn_delta_prefill_corr = None

            # TODO also do orn_deltas + sfr?
            # (to see how that changes correlation)
            # TODO with and without filling? or just post filling?

            # TODO TODO why doesn't drop_nonhallem...=True
            # [and/or tune_on_hallem=True] not look better? try again? shouldn't input
            # not get correlated so much?
            # TODO TODO compare ORN correlations in that case (after dropping and
            # delta estimate) vs hallem correlations: to what extent are they different?
            # TODO TODO is it related to wPNKC handling? not returning to ~halfmat or
            # whatever if starting from hallem/hemibrain case?

            orn_delta_corr = plot_responses_and_corr(subset_fn(orn_deltas), plot_dir,
                f'orn-deltas{suffix}', title=title, **plot_kws
            )
            orn_corr = plot_responses_and_corr(subset_fn(orn_df), plot_dir,
                f'orns{suffix}', title=title, **plot_kws
            )
            # not going to subtract pn corrs from kc corrs, so don't need return value
            plot_responses_and_corr(subset_fn(pn_df), plot_dir, f'pns{suffix}',
                title=title, **plot_kws
            )

            # model KC corr + responses should be plotted externally. responses plotted
            # differently there too, clustering after dropping silent cells.
            return orn_delta_prefill_corr, orn_delta_corr, orn_corr

        # TODO TODO why in hallem_input cases do orn-deltas* outputs seem to be
        # pre-filling (and we have no separate orn-deltas_prefill* outputs). fix for
        # consistency?

        if hallem_input:
            # (not necessarily true in general? but probably for only case i actually
            # want to plot. might need to change some of these conditionals/assertions)
            assert orn_df.shape[1] == 110
            assert pn_df.shape[1] == 110
            # TODO delete?
            #assert sim_odors is not None

            def subset_sim_odors(df):
                if megamat:
                    df = sort_odors(df, panel='megamat', warn=False)

                if sim_odors is None:
                    return df
                else:
                    return df.loc[:, [x in sim_odors for x in df.columns]]

            # assuming we won't be passing sim_odors for other cases (other than what?
            # which case is this? elaborate in comment), for now
            if sim_odors is not None:
                # TODO move this assertion into _plot_internal... ? why here?
                # (after sorting above, all these things should have matching columns)
                assert orn_deltas.columns.equals(orn_df.columns)
                _plot_internal_responses_and_corrs(subset_fn=subset_sim_odors)

            # to compare to figures in ann's paper, where 110 odors are in hallem order
            def resort_into_hallem_order(df):
                # don't need to worry about panel level being present on odor_index, as
                # it never is in hallem_input case (only place this is used)
                return df.loc[:, odor_index]

            # TODO TODO are these not being generated in latest hallem_input call
            # (restoring hemibrain path)?
            orn_delta_prefill_corr, orn_delta_corr, orn_corr = \
                _plot_internal_responses_and_corrs(suffix='_all-hallem',
                    subset_fn=resort_into_hallem_order
                )

            if megamat:
                responses = resort_into_hallem_order(responses).copy()
                spike_counts = resort_into_hallem_order(spike_counts).copy()
        else:
            orn_delta_prefill_corr, orn_delta_corr, orn_corr = \
                _plot_internal_responses_and_corrs()

        # TODO plot distribution of spike counts -> compare to w/ ann's outputs

        if hallem_input:
            suffix = '_all-hallem'
        else:
            suffix = ''

        # NOTE: this should be the only time the model KC responses are used inside the
        # plotting this fn does (and thus, the only time this flag is relevant here).
        # still returning silent cells regardless, so stuff can make this decision
        # downstream of response caching.
        if drop_silent_cells_before_analyses:
            model_kc_corr = drop_silent_model_cells(responses).corr()
            # drop_silent_model_cells should also work w/ spike count input
            spike_count_corr = drop_silent_model_cells(spike_counts).corr()
        else:
            model_kc_corr = responses.corr()
            spike_count_corr = spike_counts.corr()

        # for sanity checking some of the diffs. should also be saving this outside.
        plot_corr(model_kc_corr, plot_dir, f'kcs_corr{suffix}', title=title)
        #
        # TODO TODO also sanity check by extra plot_corr calls w/ orn_[delta_]corr
        # inputs (to check i'm using the right ones, etc)? should be exactly same as
        # orn-deltas_corr.pdf / orns_corr.pdf generated above.

        plot_corr(spike_count_corr, plot_dir, f'kcs_spike-count_corr{suffix}',
            title=title
        )

        if orn_delta_prefill_corr is not None:
            corr_diff_from_prefill_deltas = model_kc_corr - orn_delta_prefill_corr
            plot_corr(corr_diff_from_prefill_deltas, plot_dir,
                f'model_vs_orn-deltas-prefill_corr_diff{suffix}',
                title=title, xlabel=f'model KC - model ORN (deltas, pre-filling) corr'
            )

        # TODO keep the seperate versions comparing against orn-deltas vs average
        # of dynamic internal orns? actually ever diff?
        corr_diff_from_deltas = model_kc_corr - orn_delta_corr
        plot_corr(corr_diff_from_deltas, plot_dir,
            f'model_vs_orn-deltas_corr_diff{suffix}',
            title=title, xlabel=f'model KC - model ORN (deltas) corr'
        )

        corr_diff = model_kc_corr - orn_corr
        # the 'dyn' prefix is to differentiate from a plot saved in parent of plot_dir,
        # by other code.
        plot_corr(corr_diff, plot_dir, f'model_vs_dyn-orn_corr_diff{suffix}',
            title=title, xlabel=f'model KC - model ORN (avg of dynamics) corr'
        )

        if hallem_input:
            # TODO delete?
            #
            # for sanity checking some of the diffs. should also be saving this outside.
            model_kc_corr_only_megamat = subset_sim_odors(
                subset_sim_odors(model_kc_corr).T
            )
            plot_corr(model_kc_corr_only_megamat, plot_dir, 'kcs_corr', title=title)
            #

            spike_count_corr_only_megamat = subset_sim_odors(
                subset_sim_odors(spike_count_corr).T
            )
            plot_corr(spike_count_corr_only_megamat, plot_dir, 'kcs_spike-count_corr',
                title=title
            )

            # TODO also support subset_fn for plot_corr, rather than this kind of
            # subsetting?
            corr_diff_from_deltas_only_megamat = subset_sim_odors(
                subset_sim_odors(corr_diff_from_deltas).T
            )
            plot_corr(corr_diff_from_deltas_only_megamat, plot_dir,
                'model_vs_orn-deltas_corr_diff', title=title,
                xlabel=f'model KC - model ORN (deltas) corr'
            )

            corr_diff_only_megamat = subset_sim_odors(subset_sim_odors(corr_diff).T)
            plot_corr(corr_diff_only_megamat, plot_dir, 'model_vs_dyn-orn_corr_diff',
                title=title, xlabel=f'model KC - model ORN (avg of dynamics) corr'
            )

    # NOTE: currently doing after simulation, because i haven't yet implemented support
    # for tuning running on the full set of (hallem) odors, with subsequent simulation
    # running on a different set of stuff
    if hallem_input and sim_odors is not None:
        assert all(x in responses.columns for x in hallem_sim_odors)
        # TODO delete (replace w/ setting up sim_only s.t. only hallem_sim_odors are
        # simulated)
        responses = responses[hallem_sim_odors].copy()
        spike_counts = spike_counts[hallem_sim_odors].copy()

        # TODO also print fraction of silent KCs here
        # (refactor that printing to an internal fn here)

        # TODO print out threshold(s) / inhibition? possible to summarize each? both
        # scalar? (may want to use these values from one run / tuning to parameterize
        # for more glomeruli / diff runs?)

    # TODO also return model if i can make it pickle-able (+ verify that. it's possible,
    # but not likely, that it can already be [de]serialized)
    #
    # TODO maybe in wPNKC index name clarify which connectome they came from (or
    # something similarly appropriate for each type of random draws)
    return responses, spike_counts, wPNKC, param_dict


# e.g. before calculating correlations across model KC populations.
#
# Remy generally DOES drop "bad" cells, which are largely silent cells, but that isn't
# controlled by this flag. my analysis of her data generally also drops the same cells
# she does.
drop_silent_model_kcs = True

n_seeds = 100
# TODO TODO and re-run whole script once implemented for those 2 (to compare
# sd / ('ci',95) / ('pi',95|50) for each)
#
# relevant for # odors vs fraction of KCs (response breadth) plot, as well
# as ORN vs KC correlation scatterplot.
#
# currently also used to show error across flies in plot_n_odors_per_cell
# (those plots will also have model seed errors shown in separate lines)
#
# +/- 1 SD (~68% of data, if normal. ('sd', 2) should be ~95% if normal).
# this should be same as ('sd', 1), if i understand docs correctly.
#seed_errorbar = 'sd'
#
# IQR (i.e. 25th - 75th percentile)
#seed_errorbar = ('pi', 50)
# TODO TODO also try iqr ('pi', 50), or other percentile based methods?
# default errorbar is ('ci', 95) (and this is what the preprint says it is
# plotting)
#
seed_errorbar = ('ci', 95)

# was at B'c request to make new 2E versions using ('ci', 95), taking the first 20
# (/100) seeds.
#
# TODO clarify. not sure yet if she wants me to handle other seed_errorbar plots
# this way too... (don't think we do)
# TODO revert to 20 (but maybe ignore for 3B scatterplots [and S1C?])
#n_first_seeds_for_errorbar = 20
n_first_seeds_for_errorbar = None

def _get_seed_err_text_and_fname_suffix(*, errorbar=seed_errorbar,
    n_first_seeds=n_first_seeds_for_errorbar):

    if errorbar is None:
        fname_suffix = ''
    elif type(errorbar) is not str:
        fname_suffix = f'_{"-".join([str(x) for x in errorbar])}'
    else:
        fname_suffix = f'_{errorbar}'

    # for use in plot titles / similar
    err_text = f'errorbar={errorbar}'

    if n_first_seeds is not None:
        fname_suffix += f'_{n_first_seeds}first-seeds-only'
        err_text += (f'\nonly analyzing first {n_first_seeds}/{n_seeds} '
            'seeds'
        )

    return err_text, fname_suffix

seed_err_text, seed_err_fname_suffix = _get_seed_err_text_and_fname_suffix()


# TODO use in other places that do something similar?
# TODO factor to hong2p.util?
# TODO use monotonic / similar dynamic attributes if input is an appropriate pandas
# type (e.g. Index. is Series?)?
def is_sequential(data) -> bool:
    # works with np.ndarray input (and probably also pandas Series)
    #
    # NOTE: will not currently work w/ some other things I might want to use it on
    # (e.g. things that don't have  .min()/.max() methods)
    return set(range(data.min(), data.max() + 1)) == set(data)


def select_first_n_seeds(df: pd.DataFrame, *,
    n_first_seeds: Optional[int] = n_first_seeds_for_errorbar) -> pd.DataFrame:

    # assuming this function simply won't be called otherwise
    assert n_first_seeds is not None

    # assuming this fn only called on data w/ seed information (either as a column or
    # row index level)
    if 'seed' in df.columns:
        seed_vals = df.seed
    else:
        assert 'seed' in df.index.names
        seed_vals = df.index.get_level_values('seed')

    warn(f'subsetting model data to first {n_first_seeds} seeds!')

    first_n_seeds = seed_vals.sort_values().unique()[:n_first_seeds]
    assert seed_vals.min() == first_n_seeds.min() and is_sequential(first_n_seeds)

    # NOTE: not copy-ing. assuming caller won't try to mutate output w/o manually
    # .copy()-ing it first.
    subset = df[seed_vals.isin(first_n_seeds)]

    # wouldn't play nice if there were ever e.g. a diff number of cells per seed, but
    # that's not how it is now. this assertion isn't super important though, just a
    # sanity check.
    assert np.isclose(len(subset) / len(df), n_first_seeds / n_seeds)

    return subset


def plot_n_odors_per_cell(responses, ax, *, ax_for_ylabel=None, title=None,
    label='# odors per cell', label_suffix='', color='blue', linestyle='-',
    log_yscale=False) -> None:

    # TODO say how many total cells (looks like 1630 in halfmat model now?)

    # 'stim' is what Remy binary responses currently has
    # 'odor' seems to be what I get from my saved responses pickles
    assert responses.columns.name in ('odor1', 'stim', 'odor')

    n_odors = responses.shape[1]

    n_odors_col = 'n_odors'
    frac_responding_col = 'frac_responding_to_n_odors'

    # need the +1 on stop to be inclusive of n_odors
    # (so we can have a bin for cells that respond to 0 odors, as well as bin for
    # cells that respond to all 110 odors)
    n_odor_index = pd.RangeIndex(0, (n_odors + 1), name=n_odors_col)

    lineplot_kws = dict(
        # TODO refactor to share (subset of) these w/ other plots using seed_errorbar?
        #
        # like 'white' more than 'None' for markerfacecolor here.
        marker='o', markerfacecolor='white', linestyle=linestyle, legend=False,
        ax=ax
    )

    label = f'{label}{label_suffix}'

    def _n_odors2frac_per_cell(n_odors_per_cell):
        # TODO delete sort_index? prob redundant since i'm reindexing below...
        #
        # this will be ordered w/ silent cells first,
        # cells responding to 1 odor 2nd, ...
        n_odors_per_cell_counts = n_odors_per_cell.value_counts().sort_index()
        # TODO delete? made irrelevant by reindex below (prob)?
        n_odors_per_cell_counts.name = n_odors_col

        # .at[0] raising a KeyError should have the same interpretation
        assert n_odors_per_cell_counts.at[0] > 0, ('plot would be wrong if input '
            'already had silent cells dropped'
        )

        assert n_odors_per_cell.sum() == (
            (n_odors_per_cell_counts.index * n_odors_per_cell_counts).sum()
        )

        # shouldn't really need .fillna(0), b/c either 0/NaN shouldn't show up in
        # (currently log scaled) plots.
        # TODO may want to keep anyway, in case i want to try switching off log scale?
        n_odors_per_cell_counts = n_odors_per_cell_counts.reindex(n_odor_index
            ).fillna(0)

        assert n_odors_per_cell_counts.sum() == len(n_odors_per_cell)

        frac_responding_to_n_odors = n_odors_per_cell_counts / len(n_odors_per_cell)
        frac_responding_to_n_odors.name = frac_responding_col

        # (was 0.9999999999999999 in some cases)
        assert np.isclose(frac_responding_to_n_odors.sum(), 1)

        return frac_responding_to_n_odors


    # NOTE: works whether responses contains {0.0, 1.0} or {False, True}
    assert set(np.unique(responses.values)) == {0, 1}
    # how many odors each cell responds to
    n_odors_per_cell = responses.sum(axis='columns')

    experimental_unit_opts = {remy_fly_id, 'seed'}

    experimental_unit_levels = set(responses.index.names) & experimental_unit_opts
    assert len(experimental_unit_levels) <= 1

    if len(experimental_unit_levels) > 0:
        experimental_unit = experimental_unit_levels.pop()

        if n_first_seeds_for_errorbar is not None and experimental_unit == 'seed':
            responses = select_first_n_seeds(responses)

        errorbar = seed_errorbar
        lineplot_kws['errorbar'] = errorbar
        lineplot_kws['seed'] = bootstrap_seed
        lineplot_kws['err_style'] = 'bars'

        # assuming each use of this fn will have at least SOME model data (true as of
        # 2024-08-06), otherwise may not want to always use `seed_err_text` which
        # sometimes has an extra line about only using first N seeds (when relevant
        # variable is set)
        if title is None:
            title = seed_err_text
        else:
            title += f'\n{seed_err_text}'

        lineplot_kws['x'] = n_odors_col
        lineplot_kws['y'] = frac_responding_col

        frac_responding_to_n_odors = n_odors_per_cell.groupby(level=experimental_unit
            ).apply(_n_odors2frac_per_cell)

        # TODO reset_index necessary?
        frac_responding_to_n_odors = frac_responding_to_n_odors.reset_index()

    # should only happen for modelling inputs w/ hemibrain wPNKC (as the other wPNKC
    # options should have a 'seed' level)
    else:
        frac_responding_to_n_odors = _n_odors2frac_per_cell(n_odors_per_cell)

    sns.lineplot(frac_responding_to_n_odors, label=label, color=color,
        markeredgecolor=color, **lineplot_kws
    )

    if log_yscale:
        n_cells_for_ylim = 2000

        # TODO this working w/ twinx() (seems to be, but why? why only need to use
        # ax_for_ylabel for ylabel, and not yscale / etc)?
        # TODO add comment explaining what nonpositive='mask' does (and are there any
        # alternatives? what, and why did i pick this?)
        # (i think nonpositive='clip' is the default, and the only alternative)
        ax.set_yscale('log', nonpositive='mask')

        # TODO try just using len(responses)? (would cause problems if that ever
        # differed across calls made on same Axes...)
        ax.set_ylim([1 / n_cells_for_ylim, 1])

    ylabel = 'cell fraction responding to N odors'
    if ax_for_ylabel is None:
        ax.set_ylabel(ylabel)
    else:
        ax_for_ylabel.set_ylabel(ylabel)

    # https://stackoverflow.com/questions/30914462
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    xlabel = '# odors'
    ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title)


# TODO TODO still needed? don't i have some corr calc that resorts to input order?
# plot_corr?
#
# TODO try to fix corr calc to not re-order stuff (was that the issue?) -> delete
# this?
# NOTE: need to re-sort since corr_triangular necessarily (why? can't i have it sort
# to original order or something?) sorts internally
def _resort_corr(corr, add_panel):
    # TODO delete (check that below is equiv first)
    #
    # can't use add_panel kwarg to sort_odors because that only adds to index
    # levels, but we also need to add to column levels here.
    corr2 = sort_odors(util.addlevel(util.addlevel(corr, 'panel', add_panel).T,
        'panel', add_panel), warn=False).droplevel('panel',
        axis='columns').droplevel('panel', axis='index')
    #

    corr = sort_odors(corr, panel=add_panel)

    # TODO delete
    assert corr.equals(corr2)
    #

    return corr


# TODO factor to hong2p.viz?
def add_unity_line(ax: Axes, *, linestyle='--', color='r', **kwargs) -> None:
    ax.axline((0, 0), slope=1, linestyle=linestyle, color=color, **kwargs)


# TODO delete? (for debugging)
_spear_inputs2dfs = dict()
#
def bootstrapped_corr(df: pd.DataFrame, x: str, y: str, *, n_resamples=1000,
    # TODO default to 95% ci?
    # TODO delete debug _plot_dir kwarg?
    ci=90, method='spearman', _plot_dir=None) -> str:
    # TODO update doc to include new values also returned:
    # corr_text, corr, ci_lower, ci_upper, pval
    """Returns str summary of Spearman's R between columns x and y.

    Summary contains Spearman's R, the associated p-value, and a bootstrapped 95% CI.
    """
    assert 0 < ci < 100, 'ci must be between 0 and 100'

    # TODO delete
    # (after replacing model_mb...  _spear_inputs2dfs usage w/ equiv corrs from loaded
    # responses)
    #
    # rhs check just to exclude hallem stuff i don't care about that is causing resort
    # to fail
    if (_plot_dir is not None and not _plot_dir.name.startswith('data_hallem__') and
        # should already have an equivalent 'orn_corr' version here (w/ corresponding
        # non-dist y too)
        y != 'orn_corr_dist'):

        assert method == 'spearman'

        key = (_plot_dir, x, y)
        assert key not in _spear_inputs2dfs, f'{key=} already seen!'
        _spear_inputs2dfs[key] = df.copy()

        pdf = df.copy()
        if pdf.index.names != ['odor1','odor2']:
            assert all(x in pdf.columns for x in ['odor1','odor2'])
            pdf = df.set_index(['odor1','odor2'])

        if x.endswith('_dist'):
            assert y.endswith('_dist')
            # converting back from correlation distance to correlation
            pdf[x] = 1 - pdf[x]
            pdf[y] = 1 - pdf[y]
        else:
            assert not y.endswith('_dist')

        # TODO delete (along with containing debug code,
        '''
        panel = 'megamat'

        try:
            # verbose=True should cause savefig to print where these are being written
            plot_corr(
                _resort_corr(invert_corr_triangular(pdf[x]), panel),
                _plot_dir, f'_debug-corr_{x}{_suffix}', verbose=True
            )
            plot_corr(
                _resort_corr(invert_corr_triangular(pdf[y]), panel),
                _plot_dir, f'_debug-corr_{y}{_suffix}', verbose=True
            )
        except:
            print()
            traceback.print_exc()
            print()
            import ipdb; ipdb.set_trace()
        '''
    #

    to_check = df.copy()
    if x.endswith('_dist'):
        assert y.endswith('_dist')
        # converting back from correlation distance to correlation
        to_check[x] = 1 - to_check[x]
        to_check[y] = 1 - to_check[y]
    else:
        assert not y.endswith('_dist')

    assert to_check[x].max() <= 1, f'{x=} probably mislabelled correlation DISTANCE'
    assert to_check[y].max() <= 1, f'{y=} probably mislabelled correlation DISTANCE'
    del to_check

    if df[[x,y]].isna().any().any():
        # TODO fix NaN handling in method='pearson' case
        # (just dropna in all cases, and remove nan_policy arg to spearmanr)
        assert method == 'spearman', ('would need to restore NaN dropping. pearsonr '
            'does not have the same nan_policy arg spearmanr does.'
        )

        # TODO delete
        # only the Hallem cases (which dont' pass _plot_dir) should have any null model
        # corrs
        assert _plot_dir is None
        #
        assert x == 'model_corr' or x == 'model_corr_dist'
        assert not df[y].isna().any()
        # so that spearmanr doesn't return NaN here (dropping seems consistent w/ what
        # pandas calc does by default)
        #df = df.dropna(subset=[x])

    if method == 'spearman':
        # nan_policy='omit' consistent w/ pandas behavior. should only be relevant for a
        # small subset of the Hallem model outputs (default spearmanr behavior would be
        # to return NaN here)
        results = spearmanr(df[x], df[y], nan_policy='omit')

    elif method == 'pearson':
        # NOTE: no nan_policy arg here. would need to manually drop, as I had before.
        results = pearsonr(df[x], df[y])

    else:
        raise ValueError(f"{method=} unrecognized. should be either "
            "'spearman'/'pearson'"
        )

    corr = results.correlation
    pval = results.pvalue

    # the .at[x,y] is to get a scalar from matrix like
    #                mean_kc_corr  mean_orn_corr
    # mean_kc_corr       1.000000       0.657822
    # mean_orn_corr      0.657822       1.000000
    assert np.isclose(df[[x,y]].corr(method=method).at[x,y], corr)

    # TODO try this kind of CI as well?
    # https://stats.stackexchange.com/questions/18887
    # TODO try "jacknife" version mentioned in wikipedia? is my basic bootstrapping
    # approach even reasonable?

    # TODO tqdm? slow (when doing 1000) (yes! but tolerable)?
    result_list = []
    for i in range(n_resamples):
        resampled_df = df[[x, y]].sample(
                n=len(df), replace=True, random_state=(bootstrap_seed + i)
            ).reset_index(drop=True)

        if method == 'spearman':
            # TODO TODO also need nan_policy='omit' here? (or just drop in advance, to
            # also work in pearson case...)
            curr_results = spearmanr(resampled_df[x], resampled_df[y])
        elif method == 'pearson':
            curr_results = pearsonr(resampled_df[x], resampled_df[y])

        result_list.append({
            'sample': i,
            method: curr_results.correlation,
            'pval': curr_results.pvalue,
        })

    bootstrap_corrs = pd.DataFrame(result_list)

    alpha = (1 - ci / 100) / 2

    corr_ci = bootstrap_corrs[method].quantile(q=[alpha, 1 - alpha])
    corr_ci_lower = corr_ci.iloc[0]
    corr_ci_upper = corr_ci.iloc[1]
    corr_ci_text = f'{ci:.0f}% CI = [{corr_ci_lower:.2f}, {corr_ci_upper:.2f}]'

    # TODO put n_resamples in text too?

    # .2E will show 2 places after decimal then exponent (scientific notation),
    # e.g. 1.89E-180
    corr_text = f'{method}={corr:.2f}, p={pval:.2E}, {corr_ci_text}'
    return corr_text, corr, corr_ci_lower, corr_ci_upper, pval


model_responses_cache_name = 'responses.p'
model_spikecounts_cache_name = 'spike_counts.p'

_fit_and_plot_seen_param_dirs = set()
# TODO why is sim_odors an explicit kwarg? just to not have included in strs describing
# model params? i already special case orn_deltas to exclude it. why not do something
# like that (if i keep the param at all)?
# TODO TODO try to get [h|v]lines between components, 2-component mixes, and 5-component
# mix for new kiwi/control data (at least for responses and correlation matrices)
# (could just check for '+' character, to handle all cases)
def fit_and_plot_mb_model(plot_dir, sensitivity_analysis: bool = False,
    # TODO rename comparison_responses to indicate it's only used for sensitivity
    # analysis stuff? (and to be more clear how it differs from comparison_[kcs|orns])
    comparison_responses: Optional[pd.DataFrame] = None,
    n_seeds: int = 1, restrict_sparsity: bool = False,
    min_sparsity: float = 0.03, max_sparsity: float = 0.25,
    _in_sens_analysis: bool = False,
    # TODO just use model_kws for fixed_thr/wAPLKC?
    # (may now make sense, if i'm gonna add a flag to indicate whether we are in a
    # sensitivity analysis subcall)
    fixed_thr: Optional[float] = None, wAPLKC: Optional[float] = None,
    drop_silent_cells_before_analyses: bool = drop_silent_model_kcs,
    _add_combined_plot_legend=False, sim_odors=None, comparison_orns=None,
    comparison_kc_corrs=None, responses_to_suffix='',
    _strip_concs_comparison_kc_corrs=False, param_dir_prefix: str = '',
    title_prefix: str= '', extra_params: Optional[dict] = None,
    _only_return_params: bool = False, **model_kws):
    # TODO doc which extra plots made by each of comparison* inputs (or which plots are
    # changed, if no new ones)
    """
    Args:
        min_sparsity: only used for models parameterized with fixed `fixed_thr` and
            `wAPLKC` (typically in context of sensitivity analysis). return before
            generating plots if output sparsity is outside these bounds.

        max_sparsity: see min_sparsity.

        extra_params: saved alongside internal params in cache pickle/CSV
            (for keeping tracking of important external parameters, for reproducibility)
    """
    # TODO doc

    # TODO also save input data to csv for each?
    # (for more easy reproducibility w/o needing to re-load + run dF/F->spike delta fn)

    assert n_seeds >= 1

    if not restrict_sparsity:
        min_sparsity = 0
        max_sparsity = 1

    # TODO delete. isn't responses_to just overwritten w/ 'pebbled' below
    # (before being used, right?)
    my_data = f'pebbled {dff_latex}'

    # TODO fix how this might misrepresent stuff if i pass hallem data in manually?
    # currently using responses_to_suffix to try to clarify it is in fact (modified)
    # hallem input in those cases

    if 'orn_deltas' in model_kws:
        responses_to = my_data
    else:
        responses_to = 'hallem'
    #

    # TODO also try tuning on remy's subset of hallem odors?
    # (did i already? is it clear that what's in preprint was not done this way?)

    # TODO share default w/ fit_mb_model somehow?
    tune_on_hallem = model_kws.get('tune_on_hallem', True)
    if tune_on_hallem:
        tune_from = 'hallem'
    else:
        tune_from = my_data
    del my_data

    # TODO fix (give actual default? make positional?) (is this ever not available?
    # when?)
    pn2kc_connections = model_kws['pn2kc_connections']

    # TODO also use param_str for title? maybe just replace in the other direction, to
    # add dff_latex in pebbled case as necessary?). or just use filename only for many
    # parameters?

    # responses_to handled below, circa def of param_dir
    param_abbrevs = {
        'tune_on_hallem': 'hallem-tune',
        'pn2kc_connections': 'pn2kc',
        'target_sparsity': 'target_sp',
    }
    exclude_params = ('orn_deltas', 'title', 'repro_preprint_s1d')
    # TODO sort params first? (so changing order in code doesn't cause
    # cache miss...)
    # TODO why adding [''] again? why not just prepend ', ' to output if i want?
    param_str = ', '.join([''] + [
        f'{param_abbrevs[k] if k in param_abbrevs else k}={v}'
        for k, v in model_kws.items() if k not in exclude_params
    ])
    if fixed_thr is not None or wAPLKC is not None:
        assert fixed_thr is not None and wAPLKC is not None

        # TODO TODO maybe only do if _in_sens_analysis. don't think i actually want in
        # hemibrain case (other than within sens analysis subcalls)
        #
        # in n_seeds > 1 case, fixed_thr/wAPLKC will be lists of floats, and will be too
        # cumbersome to format into this
        if n_seeds == 1:
            param_str += f', fixed_thr={fixed_thr:.0f}, wAPLKC={wAPLKC:.2f}'

    if n_seeds > 1:
        param_str += f', n_seeds={n_seeds}'

    # TODO clean up / refactor. hack to make filename not atrocious when these are
    # 'pebbled_\$\\Delta_F_F\$'
    if responses_to.startswith('pebbled'):
        responses_to = 'pebbled'

    if tune_from.startswith('pebbled'):
        tune_from = 'pebbled'

    responses_to = f'{responses_to}{responses_to_suffix}'
    # TODO rename this kwarg (responses_to_suffix) to indicate it applies to tune_from
    # as well?
    if not tune_on_hallem:
        tune_from = f'{tune_from}{responses_to_suffix}'

    # this way it will also be included in params_for_csv, and we won't need to manually
    # pass to all fit_mb_model calls
    model_kws['drop_silent_cells_before_analyses'] = drop_silent_cells_before_analyses

    # TODO refactor so param_str defined from this, and then f-str below (+ for_dirname
    # def) doesn't separately specify {responses_to=}?
    params_for_csv = {
        'responses_to': responses_to,
        'tune_from': tune_from,
    }
    params_for_csv.update(
        {k: v for k, v in model_kws.items() if k not in exclude_params}
    )
    # prefix defaults to empty str
    title = title_prefix
    if _in_sens_analysis:
        assert fixed_thr is not None and wAPLKC is not None

        # assumed to be passed in (but not created by) sensitivity analysis calls
        # (recursive calls below)
        #
        # the parent directory of this should have plot_dir_prefix in it, and don't feel
        # the need to also include here.
        param_dir = plot_dir

        # TODO TODO if i allow vector fixed_thr/wAPLKC, will need to special case here
        # TODO delete? should always be redefed below...
        # (if so, then why is this code even here?)
        #title += f'thr={fixed_thr:.2f}, wAPLKC={wAPLKC:.2f} (sparsity={sparsity:.2f})'
        title += f'thr={fixed_thr:.2f}, wAPLKC={wAPLKC:.2f}'
        title_including_silent_cells = title
    else:
        # TODO refactor for_dirname handling to not specialcase responses_to/others?
        # possible to have simple code not split by fixed_thr/wAPLKC None or not?
        # TODO need to pass thru util.to_filename / simliar normalization myself now
        # (since i'm putting this in a dirname now, not the final filename of the plot)
        for_dirname = f'data_{responses_to}'
        if len(param_str) > 0:
            for_dirname += '__'
            for_dirname += param_str.strip(', ').replace('_','-').replace(', ','__'
                ).replace('=','_')

        # TODO rename plot_dir + this to be more clear?
        # plot_dir contains all modelling (mb_modeling)
        # param_dir contains outputs from model run w/ specific choice of params
        # (and only contains stuff downstream of dF/F -> spiking model
        # creation/application)
        param_dir = plot_dir / f'{param_dir_prefix}{for_dirname}'
        del for_dirname

        # TODO will this title get cut off (about a 1/3rd of last (of 3) lines, yes)?
        # fix! (still?)
        title += (
            # TODO TODO be clear in the drop_nonhallem=True
            # (drop_receptors_not_in_hallem=True) case about the fact that each of these
            # is a subset?
            #
            # TODO TODO remove / condense these first two parts of title?
            # pretty much always the same these days... (may also want tune_from to be
            # able to indicate we tuned on kiwi+control data (can it currently?)
            # TODO TODO and do the same w/ param_str def, so dirnames aren't as
            # cluttered
            f'KC thresh [/APL inh] from: {tune_from}\n'
            # TODO even need this one? were hallem / pebbled (i.e. megamat) plots ever
            # possible to confuse?
            f'responses to: {responses_to}\n'

            f'wPNKC: {pn2kc_connections}\n'
        )

        if 'target_sparsity' in model_kws:
            assert model_kws['target_sparsity'] is not None
            assert fixed_thr is None and wAPLKC is None
            # .3g will show up to 3 sig figs (regardless of their position wrt decimal
            # point), but also strip any trailing 0s (0.0915 -> '0.0915', 0.1 -> '0.1')
            title += f'target_sparsity: {model_kws["target_sparsity"]:.3g}\n'

        elif fixed_thr is not None or wAPLKC is not None:
            assert fixed_thr is not None and wAPLKC is not None

            # in n_seeds > 1 case, fixed_thr/wAPLKC will be lists of floats, and will be
            # too cumbersome to format into this
            if n_seeds == 1:
                title += f'fixed_thr={fixed_thr:.0f}, wAPLKC={wAPLKC:.2f}\n'
        else:
            # should be unreachable
            assert False

        # NOTE: this is for analyses that either always include or always drop silent
        # cells, regardless of value of `drop_silent_cells_before_analyses`
        # (e.g. should be used for analyses using `responses_including_silent_cells`)
        title_including_silent_cells = title

        # TODO say how many silent cells dropped (but maybe only in case where we aren't
        # averaging over seeds, because then would need to report average number of
        # cells dropped) (and couldn't compute that up here, esp before passing in to
        # fit_mb_model anwyay, which currently relies on the silent cell part of title
        # being added here, despite dropping silent cells for its internal plots)
        if drop_silent_cells_before_analyses:
            title += '(silent cells dropped)'
        else:
            title += '(silent cells INCLUDED)'

        # to save plots of internal ORN / PN matrices (and their correlations, etc),
        # exactly as used to run model
        model_kws['plot_dir'] = param_dir
        model_kws['title'] = title
    #

    params_for_csv['output_dir'] = param_dir.name

    model_responses_cache = param_dir / model_responses_cache_name
    model_spikecounts_cache = param_dir / model_spikecounts_cache_name
    # TODO rename this to have "fit"/"tuned" in name or something (and change
    # model_mb_responses `tuned` var/outputs to not), since this is all tuned, and
    # latter is a mix (stuff from this, but also stuff hardcoded from above / output
    # statistics)
    param_cache_name = 'params_for_csv.p'
    param_dict_cache = param_dir / param_cache_name
    wPNKC_cache_name = 'wPNKC.p'
    wPNKC_cache = param_dir / wPNKC_cache_name
    made_param_dir = False

    extra_responses_cache_name = 'extra_responses.p'
    extra_responses_cache = param_dir / extra_responses_cache_name
    extra_responses = None

    extra_spikecounts_cache_name = 'extra_spikecounts.p'
    extra_spikecounts_cache = param_dir / extra_spikecounts_cache_name
    extra_spikecounts = None

    use_cache = not should_ignore_existing('model') and (
        # checking both since i had previously only been returning+saving the 1st
        model_responses_cache.exists() and model_spikecounts_cache.exists()
    )

    tuning_output_dir = None
    # TODO delete? or implement somewhere else? (maybe just add flag to force ignore on
    # certain calls, and handle in model_mb...?)
    # TODO refactor def of 'tuning_output_dir' str
    if (extra_params is not None and 'tuning_output_dir' in extra_params and
        # NOTE: currently code in this conditional not working on _in_sens_analysis=True
        # subcalls, and we don't need anything defined in here in any of those cases
        # anyway
        not _in_sens_analysis):

        assert 'tuning_panels' in extra_params
        tuning_panels_str = extra_params['tuning_panels']

        # e.g. plot_dir=PosixPath('pebbled_6f/pdf/ijroi/mb_modeling/kiwi') ->
        # tuning_panel_dir=PosixPath('pebbled_6f/pdf/ijroi/mb_modeling/control-kiwi')
        tuning_panel_dir = plot_dir.parent / tuning_panels_str
        # NOTE: before i added `not _in_sens_analysis` condition, this was tripped in
        # those subcalls
        assert tuning_panel_dir.is_dir()

        tuning_output_dir = tuning_panel_dir / extra_params['tuning_output_dir']
        assert tuning_output_dir.is_dir()

        tuning_responses_cache = tuning_output_dir / model_responses_cache_name
        assert tuning_responses_cache.exists()

        # TODO delete? doesn't really matter unless fixed_thr/wAPLKC actually changed,
        # right? isn't that what i should be testing?
        if model_responses_cache.exists():
            curr_cache_mtime = getmtime(model_responses_cache)
            tuning_cache_mtime = getmtime(tuning_responses_cache)

            if tuning_cache_mtime >= curr_cache_mtime:
                warn(f'{tuning_responses_cache} was newer than {model_responses_cache}'
                    '! setting use_cache=False!'
                )
                use_cache = False

        if param_dict_cache.exists():
            param_dict = read_pickle(param_dict_cache)

            # np.array_equal works with both float and list-of-float inputs
            if (not np.array_equal(fixed_thr, param_dict['fixed_thr']) or
                not np.array_equal(wAPLKC, param_dict['wAPLKC'])
                ):

                warn(f'{param_dict_cache} fixed_thr/wAPLKC did not match current '
                    'inputs! setting use_cache=False!'
                )
                use_cache = False
        else:
            assert not use_cache

        # TODO also check that cached params references same tuning_output_dir (and set
        # use_cache = False if not)? or just assert it's same if already in cache?
        # NOTE: would have to load the param CSV instead of the pickle. the pickle
        # doesn't have those extra params
    #

    # to make sure we are accounting for all parameters we might vary in filename
    if param_dir in _fit_and_plot_seen_param_dirs:
        # otherwise, param_dir being in seen set would indicate an error
        assert _only_return_params, f'{param_dir=} already seen!'
        use_cache = True

    _fit_and_plot_seen_param_dirs.add(param_dir)

    print()
    # TODO TODO default to also skipping any plots made before returning? maybe add
    # another ignore-existing option ('model-plots'?) if i really want to be able to
    # remake plots w/o changing model outputs? takes a lot of time to make plots on all
    # the model outputs...
    if use_cache:
        print(f'loading model responses (+params) from cache {model_responses_cache}')
        responses = pd.read_pickle(model_responses_cache)
        spike_counts = pd.read_pickle(model_spikecounts_cache)
        param_dict = read_pickle(param_dict_cache)

        if extra_responses_cache.exists():
            extra_responses = pd.read_pickle(extra_responses_cache)

        if extra_spikecounts_cache.exists():
            extra_spikecounts = pd.read_pickle(extra_spikecounts_cache)
    else:
        # doensn't necessarily matter if it already existed. will be deleted if sparsity
        # outside bounds (and inside a sensitivity analysis call)
        made_param_dir = True

        # TODO use makedirs instead? (so if empty at end, will be deleted?)
        param_dir.mkdir(exist_ok=True, parents=True)

        print(f'fitting model ({responses_to=}{param_str})...', flush=True)

        # TODO check i can replace model_test.py portion like this w/ this
        # implementation?
        if n_seeds > 1:
            assert fixed_thr is None or type(fixed_thr) is list
            assert wAPLKC is None or type(wAPLKC) is list

            # only to regenerate model internal plots (which only ever are saved on the
            # first seed, in cases where there would be multiple runs w/ diff seeds)
            # without waiting for rest of seed runs to finish. will NOT write to
            # responses cache or make any plots based on output responses in this case!
            #first_seed_only = True
            first_seed_only = False
            if first_seed_only:
                # first_seed_only=True only intended for regenerating these internal
                # plots. probably a mistake if it's True any other time.
                assert 'plot_dir' in model_kws

            # TODO make kwarg
            # same seed Matt starts at in
            # matt-modeling/docs/independent-draw-reference.html
            initial_seed = 94894 + 1

            # TODO get good desc for tqdm
            #desc=f'{draw_type} ({n_claws=})'
            seeds = []
            responses_list = []
            spikecounts_list = []
            param_dict_list = []
            first_param_dict = None
            wPNKC_list = []

            _fixed_thr = None
            _wAPLKC = None

            if fixed_thr is not None:
                # this branch should not run in any sensitivity analyis subcalls, as
                # currently only doing that for n_seeds=1 (i.e. hemibrain) case
                # (otherwise, we would need to test if tuning_output_dir is None / etc)
                assert not _in_sens_analysis

                assert len(fixed_thr) == len(wAPLKC) == n_seeds

                assert tuning_output_dir is not None

                tuning_wPNKC_cache = tuning_output_dir / wPNKC_cache_name
                assert tuning_wPNKC_cache.exists()

                tuning_wPNKC = read_pickle(tuning_wPNKC_cache)
                # assuming all entries of a given seed are at adjacent indices in the
                # seed level values (should never be False given how i'm implementing
                # things)
                tuning_seeds = tuning_wPNKC.index.get_level_values('seed').unique()

            # TODO TODO include at least pn2kc_... in progress bar
            # TODO some way to have a nested progress bar, so that outer on (in
            # model_mb_... i'm imagining) increments for each model type, and this inner
            # one increments for each seed? or do something else to indicate outer
            # progress?
            for i in tqdm(range(n_seeds), unit='seed'):
                seed = initial_seed + i
                seeds.append(seed)
                assert 'seed' not in model_kws

                if fixed_thr is not None:
                    _fixed_thr = fixed_thr[i]
                    _wAPLKC = wAPLKC[i]
                    # would need to use same seed sequence if this ever failed
                    assert seed == tuning_seeds[i]

                responses, spike_counts, wPNKC, param_dict = fit_mb_model(
                    # TODO or can i handle fixed_thr/wAPLKC thru model_kws (prob not)?
                    # (maybe i will soon be able to, if i'm gonna replace some of their
                    # usage with a flag to indicate whether we are in a sensitivity
                    # analysis subcall...)
                    sim_odors=sim_odors, fixed_thr=_fixed_thr, wAPLKC=_wAPLKC,
                    seed=seed,
                    # ORN/PN plots would be redundant, and overwrite each other.
                    # currently those are the only plots I'm making in here.
                    make_plots=(i == 0), **model_kws
                )

                if fixed_thr is not None:
                    # could prob delete. should be sufficienet to check the seeds equal,
                    # as we are doing above
                    assert tuning_wPNKC.loc[seed].equals(wPNKC)

                if first_seed_only:
                    warn('stopping after model run with first seed '
                        '(first_seed_only=True)! model response caches / downstream '
                        'plots not updated!'
                    )
                    return None

                responses = util.addlevel(responses, 'seed', seed)
                spike_counts = util.addlevel(spike_counts, 'seed', seed)

                # TODO assert order of wPNKC columns same in each?
                wPNKC = util.addlevel(wPNKC, 'seed', seed)

                if first_param_dict is None:
                    first_param_dict = param_dict
                else:
                    assert param_dict.keys() == first_param_dict.keys()

                responses_list.append(responses)
                spikecounts_list.append(spike_counts)
                param_dict_list.append(param_dict)

                wPNKC_list.append(wPNKC)

            responses = pd.concat(responses_list, verify_integrity=True)
            spike_counts = pd.concat(spikecounts_list, verify_integrity=True)
            wPNKC = pd.concat(wPNKC_list, verify_integrity=True)

            param_dict = {
                k: [x[k] for x in param_dict_list] for k in first_param_dict.keys()
            }
        else:
            # isinstance works w/ both float and np.float64 (but not int)
            assert fixed_thr is None or isinstance(fixed_thr, float)
            assert wAPLKC is None or isinstance(wAPLKC, float)

            # TODO rename param_dict everywhere -> tuned_params?
            responses, spike_counts, wPNKC, param_dict = fit_mb_model(
                # TODO or can i handle fixed_thr/wAPLKC thru model_kws (prob not)?
                # (maybe i will soon be able to, if i'm gonna replace some of their
                # usage with a flag to indicate whether we are in a sensitivity analysis
                # subcall...)
                sim_odors=sim_odors, fixed_thr=fixed_thr, wAPLKC=wAPLKC, **model_kws
            )

        print('done', flush=True)

        orn_deltas = None
        if responses_to != 'hallem':
            orn_deltas = model_kws['orn_deltas']
            input_odors = orn_deltas.columns
        else:
            # NOTE: not saving model input (the Hallem ORN deltas) here, b/c it's added
            # by fit_mb_model internally, and it should be safe to assume this will not
            # change across runs. If it does change, hopefully the history of that is
            # accurately reflected in commit history of my drosolf repo.
            # TODO maybe refactor so i can define orn_deltas for this case too (and thus
            # so it's also saved below in that case)?
            n_hallem_odors = 110
            assert responses.shape[1] >= n_hallem_odors
            input_odors = responses.columns[:n_hallem_odors]

        # remove any odors added by `extra_orn_deltas` code (internal to fit_mb_model)
        if len(input_odors) < responses.shape[1]:
            if 'panel' in input_odors.names:
                input_odors = input_odors.droplevel('panel')

            assert responses.columns[:len(input_odors)].equals(input_odors)
            # (defined as None above)
            extra_responses = responses.iloc[:, len(input_odors):].copy()
            extra_spikecounts = spike_counts.iloc[:, len(input_odors):].copy()

            responses = responses.iloc[:, :len(input_odors)].copy()
            spike_counts = spike_counts.iloc[:, :len(input_odors)].copy()

        del input_odors

        # TODO also copy over mb_modeling/dff2spiking_fit.p (so we can actually
        # repro that) (how to do in a way i can easily support w/ -c?)
        # TODO load and re-save mb_modeling/dff2spiking_model_input.csv? could figure
        # out model from that at least?

        if orn_deltas is not None:
            # just saving these for manual reference, or for use in -c check.
            # not loaded elsewhere in the code.
            to_pickle(orn_deltas, param_dir / 'orn_deltas.p')

            # TODO also save a hemibrain-filled version of this?
            #
            # current format like:
            # panel	megamat	         megamat          ...
            # odor	2h @ -3	         IaA @ -3         ...
            # glomerulus
            # D	40.845711426286  37.2453183810278 ...
            # DA2	15.325702916103	 11.4666387062239 ...
            # ...
            to_csv(orn_deltas, param_dir / 'orn_deltas.csv')

        # NOTE: saving raw (unsorted, etc) responses to cache for now, so i can modify
        # that bit. CSV saving is currently after all sorting / post-processing.
        to_pickle(responses, model_responses_cache)
        to_pickle(spike_counts, model_spikecounts_cache)
        to_pickle(wPNKC, wPNKC_cache)

        # TODO update this comment? i think param_dict might have a lot more stuff
        # now...
        #
        # in n_seeds=1 case, param_dict keys are:
        # 'fixed_thr', 'wAPLKC', and 'wKCAPL' (all w/ scalar values)
        #
        # in n_seeds > 1 case, should be same keys, but list values (of length equal to
        # n_seeds)
        to_pickle(param_dict, param_dict_cache)

        # TODO check these look ok (+ roughly match what i already uploaded for depasq),
        # in both seed and no seed cases
        # TODO don't save in sensitivity analysis subcalls, as this should not change
        # across those
        to_csv(wPNKC, param_dir / 'wPNKC.csv', verbose=(not _in_sens_analysis))

        # saving after all the other things, so that (if script run w/ -c) checks
        # against old/new outputs have an opportunity to trip and fail before this is
        # written
        if extra_responses is not None:
            assert extra_spikecounts is not None
            to_pickle(extra_responses, extra_responses_cache)
            to_pickle(extra_spikecounts, extra_spikecounts_cache)
        else:
            assert extra_spikecounts is None

            # delete any existing extra_responses pickles
            # (don't want stale versions of these being loaded alongside newer
            # responses.p data)
            if extra_responses_cache.exists():
                extra_responses_cache.unlink()

            if extra_spikecounts_cache.exists():
                extra_spikecounts_cache.unlink()

    # TODO change order so sparsity comes after target_sparsity?
    # (potentially just moving target_sparsity to end of input params?
    # not sure. fixed_thr, wAPLKC, wKCAPL currently between them)
    #
    # param_dict should include 'fixed_thr', 'wAPLKC' and 'wKCAPL' parameters, as
    # they are at the end of the model run (either tuned or
    # hardcoded-from-the-beginning)
    # TODO just assert the things mentioned in comment above are there in
    # param_dict?
    assert not any(k in params_for_csv for k in param_dict.keys())
    params_for_csv.update(param_dict)

    # NOTE: if there were ever different number of cells for the different seeds (in the
    # cases where the row index has a 'seed' level, in addition to the 'cell' level,
    # e.g. the pn2kc_connections='uniform' case), then we'd want to compute sparsities
    # within seeds and then average those (to not weight different MB instantiations
    # differently, which is consistent w/ how Remy mean sparsity computed on real fly
    # data).
    sparsity = (responses > 0).mean().mean()
    params_for_csv['sparsity'] = sparsity

    # TODO factor out this subsetting to (internal?) fn? or just use megamat_responses
    # directly below?
    # TODO .get_level_values if i restore panel level preservation thru fit_mb_model
    megamat_mask = responses.columns.map(odor_is_megamat)

    # should be true in both hallem (which has ~110 odors, including all the 17
    # megamat) and pebbled-megamat input cases
    have_megamat = megamat_mask.values.sum() >= 17

    if have_megamat:
        megamat_responses = responses.loc[:, megamat_mask]
        megamat_sparsity = (megamat_responses > 0).mean().mean()
        del megamat_responses
        params_for_csv['megamat_sparsity'] = megamat_sparsity

    if extra_params is not None:
        assert not any(k in params_for_csv for k in extra_params.keys())
        params_for_csv.update(extra_params)

    # TODO does param_series have anything useful for repro that we dont have in
    # params_for_csv.p (param_dict_cache, saved earlier)?
    param_series = pd.Series(params_for_csv)
    try:
        # just to manually inspect all relevant parameters for outputs in a given
        # param_dir
        to_csv(param_series, param_dir / 'params.csv', header=False,
            verbose=(not _in_sens_analysis)
        )

    # TODO change code to avoid this happening in the first place?
    # (should only happen on second call used for getting inh params on a panel set, to
    # then run a model with a single panel and those inh params later)
    except MultipleSavesPerRunException:
        if _only_return_params:
            return params_for_csv
        else:
            raise

    del param_series

    # TODO even need to rename at this point? anything downstream actually not work with
    # 'odor' instead of 'odor1'?
    # TODO just fix natmix.plot_corr to also work w/ level named 'odor'?
    # (or maybe odor_corr_frame_to_dataarray?)
    #
    # even if input to fit_mb_model has a 'panel' level on odor index, the output will
    # not
    assert len(responses.columns.shape) == 1 and responses.columns.name == 'odor'
    responses.columns.name = 'odor1'

    assert len(spike_counts.columns.shape) == 1 and spike_counts.columns.name == 'odor'
    spike_counts.columns.name = 'odor1'

    panel = None
    if responses_to == 'hallem':
        assert have_megamat
        # the non-megamat odors will just be sorted to end
        panel = 'megamat'
    else:
        orn_deltas = model_kws['orn_deltas']
        assert 'panel' in orn_deltas.columns.names

        panels = set(orn_deltas.columns.get_level_values('panel'))
        del orn_deltas

        if len(panels) == 1:
            panel = panels.pop()
            assert type(panel) is str
        else:
            # should currently only be true in the calls w/ multiple panel inputs (e.g.
            # for pre-tuning on kiwi+control, to then run this fn w/ just kiwi input).
            # just gonna return early w/ params, skipping this stuff, fow now.
            panel = None

    if panel is not None:
        responses = sort_odors(responses, panel=panel, warn=False)
        spike_counts = sort_odors(spike_counts, panel=panel, warn=False)

    # TODO update these wrappers to also make dir if not exist (if they don't already)
    to_csv(responses, param_dir / 'responses.csv', verbose=(not _in_sens_analysis))
    to_csv(spike_counts, param_dir / 'spike_counts.csv',
        verbose=(not _in_sens_analysis)
    )

    if _only_return_params:
        return params_for_csv

    # TODO delete? should be handled by _only_return_params (cases they are triggered
    # should be the same)
    if panel is None:
        # TODO is there any code below that actually doesn't work w/ multiple panels?
        # care to get plots (prob not)?
        warn('returning from fit_and_plot_model before making plots, because input had'
            ' multiple panels (currently unsupported)'
        )
        return params_for_csv
    #

    # TODO use one/both of these col defs outside of just for s1d?
    odor_col = 'odor1'
    sparsity_col = 'response rate'

    def _per_odor_tidy_model_response_rates(responses: pd.DataFrame) -> pd.DataFrame:
        """Returns dataframe with [odor_col, sparsity_col [, 'seed']] columns.

        Returned dataframe also has a 'seed' column, if input index has a 'seed' level,
        with response rates computed within each 'seed' value in input.
        """
        # TODO warn / err if there are no silent cells in responses (would almost
        # certainly indicate mistake in calling code)?

        if 'seed' in responses.index.names:
            response_rates = responses.groupby('seed', sort=False).mean()
            assert response_rates.columns.name == odor_col
            assert response_rates.index.name == 'seed'

            response_rates = response_rates.melt(value_name=sparsity_col,
                # ignore_index=False to keep seed
                ignore_index=False
            ).reset_index()

            assert 'seed' in response_rates.columns
            assert odor_col in response_rates.columns
        else:
            response_rates = responses.mean()
            assert response_rates.index.name == odor_col
            response_rates = response_rates.reset_index(name=sparsity_col)

        return response_rates


    # TODO rename to plot_and_save... / something? consistent way to indicate which of
    # my plotting fns (also) save, and which do not?
    # TODO refactor to use this for s1d (maybe w/ boxplot=True option or something?
    # requiring box plot if there are multiple seeds on input?)?
    # TODO move def outside of fit_and_plot... (near plot_n_odors_per_cell def?)?
    def plot_sparsity_per_odor(sparsity_per_odor, comparison_sparsity_per_odor, suffix
        ) -> Tuple[Figure, Axes]:

        fig, ax = plt.subplots()
        # TODO rename sparsity -> response_fraction in all variables / col names too
        # (or 'response rate'/response_rate, now in col def for s1d?)
        ylabel = 'response fraction'

        title = title_including_silent_cells

        err_kws = dict()
        if 'seed' in sparsity_per_odor.columns:
            assert n_first_seeds_for_errorbar is None, 'implement here if using'

            # TODO factor (subset of?) these kws into a seed_errorbar_style_kws or
            # something? to share these w/ plot_n_odors_per_cell (+ other places that
            # should use same errorbar style)
            #
            # TODO have markerfacecolor='None', whether or not we want to show
            # errorbars (maybe after a -c check that hemibrain stuff unchanged w/o)?
            err_kws = dict(markerfacecolor='white', errorbar=seed_errorbar,
                seed=bootstrap_seed, err_style='bars'
            )
            # TODO refactor to share w/ place copied from (plot_n_odors_per_cell)?
            if title is None:
                title = seed_err_text
            else:
                title += f'\n{seed_err_text}'

        color = 'blue'
        sns.lineplot(sparsity_per_odor, x=odor_col, y=sparsity_col, color=color,
            marker='o', markeredgecolor=color, legend=False, label=ylabel, ax=ax,
            **err_kws
        )
        if comparison_sparsity_per_odor is not None:
            # TODO how to label this? label='tuned'?
            color = 'gray'
            sns.lineplot(comparison_sparsity_per_odor, x=odor_col, y=sparsity_col,
                color=color, marker='o', markeredgecolor=color, legend=False,
                label=f'{ylabel} (tuned)', ax=ax, **err_kws
            )

        # renaming from column name odor_col
        ax.set_xlabel('odor'
            f'\nmean response rate: {sparsity_per_odor[sparsity_col].mean():.3g}'
        )

        # TODO add dotted line for target sparsity, when applicable?

        rotate_xticklabels(ax, 90)

        ax.set_title(title)
        ax.set_ylabel(ylabel)

        savefig(fig, param_dir, f'sparsity_per_odor{suffix}')
        return fig, ax


    repro_preprint_s1d = model_kws.get('repro_preprint_s1d', False)

    eb_mask = responses.columns.get_level_values(odor_col).str.startswith('eb @')
    assert eb_mask.sum() <= 1
    # should only be true if panel is validation2 (pebbled/megamat or hallem should
    # both have it)
    if repro_preprint_s1d and eb_mask.sum() == 0:
        repro_preprint_s1d = False

    if repro_preprint_s1d:
        assert extra_responses is not None

        s1d_responses = pd.concat([extra_responses, responses.loc[:, eb_mask]],
            axis='columns', verify_integrity=True
        )
        # (extra_responses still had 'odor' and responses had 'odor1' at time of concat)
        s1d_responses.columns.name = odor_col

        s1d_sparsities = _per_odor_tidy_model_response_rates(s1d_responses)

        # TODO delete 1 of these 2 plots below?

        fig, ax = plt.subplots()
        # TODO TODO adjust formatting so outlier points don't overlap (reduce alpha /
        # jitter?) (see pebbled/hemidraw one, which may be the one we want to use)
        sns.boxplot(data=s1d_sparsities, x=odor_col, y=sparsity_col, ax=ax, color='k',
            fill=False, flierprops=dict(alpha=0.175)
        )
        ax.set_title(title_including_silent_cells)
        savefig(fig, param_dir, 's1d_private_odor_sparsity')

        # TODO rewrite plot_sparsity... to make these 2nd arg optional?
        plot_sparsity_per_odor(s1d_sparsities, None, '_s1d')


    responses_including_silent = responses.copy()
    # TODO TODO what did Ann do for this?
    # (Matt did not drop silent cells. not sure about what Ann did.)
    if drop_silent_cells_before_analyses:
        # NOTE: important this happens after def of sparsity above
        responses = drop_silent_model_cells(responses)

    # TODO delete (/ move up before early return, right after sparsity calc)
    # TODO is there a big mismatch betweeen target_sparsity and sparsity (yes, see
    # below)?
    # TODO err / warn if differs much at all
    # TODO inspect cases where it differs (including olfsysm log)
    # (seems like nearly every case differs somewhat seriously)
    # ...
    # fitting model (responses_to='pebbled', tune_on_hallem=True,
    #   drop_receptors_not_in_hallem=True, pn2kc_connections=hemibrain,
    #   target_sparsity=0.1)...
    # ...
    # model_kws.get("target_sparsity")=0.1
    # sparsity=0.1804732780428448
    # ...
    # fitting model (responses_to='pebbled', tune_on_hallem=True,
    #   drop_receptors_not_in_hallem=True, pn2kc_connections=hemibrain,
    #   target_sparsity=0.05)...
    # ...
    # fixed_thr: 221.8443262928323
    # wAPLKC: 5.523460576698389
    # wKCAPL: 0.0030067831119751703
    # done
    # responses.shape=(1837, 17)
    # model_kws.get("target_sparsity")=0.05
    # sparsity=0.08511319606775754
    #
    # TODO TODO anything i can change to make tuning converge better?
    # relevant parameters:
    # rv.kc.tuning_iters (not a cap tho, just to keep track of it?)
    # mp.kc.max_iters (a cap for above) (default=10)
    # mp.kc.apltune_subsample (default=1)
    # mp.kc.sp_lr_coeff (initial learning rate, from which subsequent iteration learning
    #     rates decrease w/ sqrt num iters i think) (default=10.0)
    #
    #     ...
    #     double lr = p.kc.sp_lr_coeff / sqrt(double(rv.kc.tuning_iters));
    #     double delta = (sp - p.kc.sp_target) * lr/p.kc.sp_target;
    #     rv.kc.wAPLKC.array() += delta;
    #     ...
    #
    # mp.kc.sp_acc (the tolerance acceptable) (default=0.1)
    #     "the fraction +/- of the given target that is considered an acceptable
    #     sparsity"
    #
    # tuning proceeds while: ( (abs(sp-p.kc.sp_target)>(p.kc.sp_acc*p.kc.sp_target)) &&
    # (rv.kc.tuning_iters <= p.kc.max_iters) )
    if 'target_sparsity' in model_kws:
        target_sparsity = model_kws['target_sparsity']
        print()
        print(f'target_sparsity={target_sparsity:.3g}')
        print(f'sparsity={sparsity:.3g}')

        adiff = sparsity - target_sparsity
        rdiff = adiff / target_sparsity
        # TODO TODO use to inspect improvement when increasing tuning time (+ also
        # time model running, to see how much extra time extra tuning adds)
        print(f'{(rdiff * 100):.1f}% rel sparsity diff')
        print(f'{adiff:.2g} abs sparsity diff')

        # TODO TODO assert one/both below some threshold? warn above some thresh?

        # TODO delete?
        # TODO what fraction is passing atol=0.005? should i make it higher? .01?
        # (at least for target_sparsity=.1, w/ other params in the single choice i
        # actually do sens analysis on, it's only off by ~.009...)
        #if not np.isclose(sparsity, model_kws['target_sparsity'], atol=0.005):
        #    import ipdb; ipdb.set_trace()

        if have_megamat:
            if not np.isclose(megamat_sparsity, sparsity):
                print(f'megamat_sparsity={megamat_sparsity:.3g}')

        print()
    #

    # TODO drop panel here (before computing) if need be (after switching to pass input
    # that has that as a level on odor axis)
    if responses.index.name is None and 'seed' in responses.index.names:
        # TODO TODO refactor to use new mean_of_fly_corrs (passing in 'seed' for id
        # level)?
        corr_list = []
        seeds = []
        # level=<x> vs <x> didn't seem to matter here (at least, checking seed_corrs
        # after concat)
        for seed, seed_df in responses.groupby(level='seed', sort=False):
            seeds.append(seed)

            # each element of list is a Series now, w/ a 2-level multiindex for odor
            # combinations
            # NOTE: odor levels currently ('odor1', 'odor1') (SAME NAME, which might
            # cause problems...)
            corr_list.append(corr_triangular(seed_df.corr()))

        seed_corrs = pd.concat(corr_list, axis=1, keys=seeds, names='seed',
            verify_integrity=True
        )
        assert list(seed_corrs.columns) == seeds

        # converts from (row=['odor1','odor2'] X col='seed') to
        # row=['odor1','odor2','seed'] series
        # TODO TODO TODO has adding dropna=False broken anything? it was to fix odor
        # pairs not matching up in merge in comparison_orns code below
        # (only was triggered in hallem/uniform case)
        seed_corr_ser = seed_corrs.stack(dropna=False)

        # TODO can i convert below comment to an assertion / delete then (seems from
        # parenthetical below i felt i had figured it out)
        # TODO why is len(seed_corr_ser) (or len(model_corr_df)) == 56290, while
        # seed_corrs.size == 59950 (= n_seeds (10) * 5995 (= [110**2 - 110]/2) )
        # ipdb> seed_corrs.size - seed_corrs.isna().sum().sum()
        # 56290
        # (so it's just NaN elements that are the issue)
        seed_corr_ser.name = 'model_corr'
        model_corr_df = seed_corr_ser.reset_index()

        # TODO delete
        '''
        # TODO TODO fix how glycerol corrs get lost (leave NaN?)
        # (seems i'd have to fillna w/ a placeholder before stack() above, then replace
        # w/ NaN after)
        if responses_to == 'hallem' and pn2kc_connections == 'uniform':
            import ipdb; ipdb.set_trace()
        '''

        # TODO rename to 'odor_a', 'odor_b'? (here and in *corr_triangular?)? assuming
        # 'odor2' here isn't the for-mixtures 'odor2' i often have in odor
        # multiindices...
        odor_levels = ['odor1', 'odor2']
        mean_pearson_ser = seed_corr_ser.groupby(level=odor_levels, sort=False).mean()

        # TODO below 2 comments still an issue?
        # TODO TODO TODO fix! (only in hallem/uniform, after no longer only passing
        # megamat odors as sim_odors)
        # TODO TODO TODO check at input of this what is reducing length of
        # triangular series below expected shape. missing at least (in either order):
        # a='g-decalactone @ -2'
        # b='glycerol @ -2'
        try:
            # TODO rename _index kwarg?
            # TODO TODO +fix so i don't need to pass it (what was the purpose of
            # passing it again? doc in comment) (doesn't seem to still be triggering?)
            pearson = invert_corr_triangular(mean_pearson_ser, _index=seed_corrs.index)
        except AssertionError:
            print()
            traceback.print_exc()
            print()
            import ipdb; ipdb.set_trace()
    else:
        pearson = responses.corr()

        corr_ser = corr_triangular(pearson)
        corr_ser.name = 'model_corr'
        model_corr_df = corr_ser.reset_index()

        # TODO just start w/ 'odor_a', 'odor_b' here, to avoid issues later?
        # or even 'odor_row', 'odor_col'?
        #
        # just to match invert_corr_triangular output above (from 'odor1' for both here)
        pearson.index.name = 'odor'
        pearson.columns.name = 'odor'

    pearson = _resort_corr(pearson, panel)

    # TODO TODO try deleting this and checking i can remake all the same
    # megamat/validation plots? feel like i might not need this anymore (or maybe i want
    # to stop needing it anyway... could then support mix dilutions for kiwi/control)
    # TODO refactor to share w/ other places?
    def _strip_index_and_col_concs(df):
        assert df.index.name.startswith('odor')
        assert df.columns.name.startswith('odor')

        # assuming no duplicate odors in input
        assert len(set(df.index)) == len(df.index)
        assert len(set(df.columns)) == len(df.columns)

        # TODO just use hong2p.olf.parse_odor_name instead of all this?
        delim = ' @ '
        assert df.index.str.contains(delim).all()
        assert df.columns.str.contains(delim).all()
        df = df.copy()
        df.index = df.index.map(lambda x: x.split(delim)[0])
        df.columns = df.columns.map(lambda x: x.split(delim)[0])
        #

        # TODO delete try/except (did i not rename diag 'ms @ -3' appropriately?)
        try:
            # assuming dropping concentration info hasn't created duplicates
            # (which would happen if input has any 1 odor presented at >1 conc...)
            assert len(set(df.index)) == len(df.index)
        except AssertionError:
            # TODO deal with other odors duplicated
            # (either by also mangling before, like 'ms @ -3' -> 'diag ms @ -3', or by
            # subsetting after?)
            # ipdb> df.index.value_counts()
            # 2h         2
            # t2h        2
            # aphe       2
            # 2-but      2
            # va         2
            # 1-6ol      2
            print(f'{len(set(df.index))=}')
            print(f'{len(df.index)=}')

            # TODO also, why are 'pfo' row / col NaN here (including identity...)? data?
            # mishandling? want to drop pfo anyway?

            # TODO TODO has air mix been handled appropriately up until here?
            # seeems like we may have just dropped odor2 and lumped them in w/ ea/oct,
            # which would be bad. prob want to keep air mix? could also drop and just
            # use in-vial 2-component mix

            # (not currently an issue since i added the hack to move conc info to name
            # part for those odors)
            # TODO fix for new kiwi vs control data
            # (seems to be caused by dilutions of mixture. just drop those first? could
            # call the natmix fn for that)
            import ipdb; ipdb.set_trace()

        assert len(set(df.columns)) == len(df.columns)
        return df

    pearson = _strip_index_and_col_concs(pearson)

    if _in_sens_analysis:
        assert fixed_thr is not None and wAPLKC is not None

        if ((min_sparsity is not None and sparsity < min_sparsity) or
            (max_sparsity is not None and sparsity > max_sparsity)):

            warn(f'sparsity out of [{min_sparsity}, {max_sparsity}] bounds! returning '
                'without making plots!'
            )

            # TODO register atexit instead (use some kind of wrapped dir creation fn
            # that handles that for me automatically? factor out of savefig/whatever i
            # have that currently does something like that?)?
            if made_param_dir:
                # TODO err here if -c CLI arg passed?
                print('deleting {param_dir}!')
                shutil.rmtree(param_dir)

            return params_for_csv

        # don't think i wanted to return this for stuff outside sparsity bounds
        # (return above)
        params_for_csv['pearson'] = pearson

        if n_seeds == 1:
            title = (
                f'thr={fixed_thr:.2f}, wAPLKC={wAPLKC:.2f} (sparsity={sparsity:.3g})'
            )
        else:
            title = f'sparsity={sparsity:.3g}'

    plot_corr(pearson, param_dir, 'corr', xlabel=title)

    if responses_to == 'hallem':
        # TODO factor to use n_megamat_odors / something instead of 17...
        #
        # all megamat odors should have been sorted before other hallem odors, so we
        # should be able to get the megamat17 subset by indexing this way
        plot_corr(pearson.iloc[:17, :17], param_dir, 'corr_megamat', xlabel=title)
    #

    def _compare_model_kc_to_orn_data(comparison_orns, desc=None):
        # TODO assert input odors match comparison_orns odors exactly?
        # (currently stripping conc in at least corr diff case?)
        # (or assert around merge below, that we have all same odor pairs in both
        # dataframes being merged)

        if desc is None:
            orn_fname_part = 'orn'
            # might cause some confusion if comparison_orns are hallem data...
            orn_label_part = 'ORN'
        else:
            # assuming we don't need to normalize desc for filename
            orn_fname_part = f'orn-{desc}'
            orn_label_part = f'ORN ({desc})'

        # TODO switch to checking if ['date', 'fly_num'] (or 'fly_id') in column levels,
        # maybe adding an assertion columns.name == 'glomerulus' if not? might make it
        # nicer to refactor into plot_corr (for deciding whether to call
        # mean_of_fly_corrs)
        if comparison_orns.columns.name == 'glomerulus':
            mean_orn_corrs = corr_triangular(comparison_orns.T.corr())
        else:
            assert comparison_orns.columns.names == ['date', 'fly_num', 'roi']
            # will exclude NaN (e.g. va/aa in first 2 megamat flies)
            mean_orn_corrs = mean_of_fly_corrs(comparison_orns, square=False)

        mean_orn_corrs.name = 'orn_corr'

        model_corr_df_odor_pairs = set(
            model_corr_df.set_index(['odor1', 'odor2']).index
        )
        orn_odor_pairs = set(mean_orn_corrs.index)

        # NOTE: changing abbrev_hallem_odor_index will likely cause this to fail if
        # model outputs are not also regenerated (via CLI arg `-i model`)
        assert model_corr_df_odor_pairs == orn_odor_pairs

        # TODO delete? seems like it would fail if any NaN...
        assert len(mean_orn_corrs.values) == len(np.unique(mean_orn_corrs.values))

        df = model_corr_df.merge(mean_orn_corrs, on=['odor1', 'odor2'])

        # TODO any reason to think this is actually an issue? couldn't we just
        # have bona fide duplicate corrs (yea, we prob do)?
        #
        # 2024-05-17: still an issue (seemingly only in hallem/uniform case, not
        # pebbled/uniform or hallem/hemibrain)
        #
        # TODO fix to work w/ some NaN corr values?
        # TODO was this actually caused by duplicate correlat
        # TODO why just failing in hallem/uniform, and not hallem/hemibrain,
        # case? both have some NaN kc corrs... (i think it was probably more a matter of
        # one having duplicate corrs...)
        # TODO TODO was this actually duplicate corrs though? that would make more sense
        # for KC outputs w/ small number of inputs (maybe?), but in ORN inputs?
        try:
            assert len(mean_orn_corrs.values) == len(np.unique(df['orn_corr']))
        except AssertionError:
            # TODO actually summarize these if i want to keep warn here at all?
            # or just delete?
            warn('some duplicate corrs! (may not actually be an issue...)')
            # ipdb> len(mean_orn_corrs.values)
            # 5995
            # 2024-05-20: this is now 5886 (one NaN? no)
            # ipdb> len(np.unique(df['orn_corr']))
            # 5885
            #import ipdb; ipdb.set_trace()

        # converting to correlation distance, like in matt's
        df['model_corr_dist'] = 1 - df['model_corr']
        df['orn_corr_dist'] = 1 - df['orn_corr']

        # TODO only do in megamat case
        df['odor1_is_megamat'] = df.odor1.map(odor_is_megamat)
        df['odor2_is_megamat'] = df.odor2.map(odor_is_megamat)
        df['pair_is_megamat'] = df[['odor1_is_megamat','odor2_is_megamat']
            ].all(axis='columns')

        if n_first_seeds_for_errorbar is not None and 'seed' in df.columns:
            df = select_first_n_seeds(df)

        def _save_kc_vs_orn_corr_scatterplot(metric_name):
            # to recreate preprint fig 3B

            if metric_name == 'correlation distance':
                col_suffix = '_corr_dist'

                # TODO rename dists -> dist (to share w/ col_suffix -> deleting this
                # after)?
                fname_suffix = '_corr_dists'

                # TODO double check language
                help_str = 'top-left: decorrelated, bottom-right: correlated'

                # TODO just derive bounds for either corr or corr-dist version from the
                # other -> consolidate to share assertion?
                # (/ refactor some other way...)
                plot_max = 1.5
                plot_min = 0.0

            elif metric_name == 'correlation':
                col_suffix = '_corr'
                fname_suffix = '_corr'

                # confident in language on this one
                help_str = 'top-left: correlated, bottom-right: decorrelated'

                plot_max = 1
                plot_min = -.5
            else:
                assert False, 'only above 2 metric_name values supported'

            if 'seed' in df.columns:
                errorbar = seed_errorbar
            else:
                # no seeds to compute CI over here. sns.lineplot would generate a
                # RuntimeWarning (about an all-NaN axis), if I tried to generate error
                # bars same way.
                errorbar = None

            if not df.pair_is_megamat.all():
                # just removing errorbar b/c was taking a long time and don't really
                # care about this plot anymore... (shouldn't take long if not using
                # bootstrapped CI, if i change errorbar)
                errorbar = None

                color_kws = dict(
                    hue='pair_is_megamat', hue_order=[False, True],
                    # TODO also try to have diff err/marker alphas here? prob not worth
                    # it, considering i don't really use this version of the plots...
                    palette={True: to_rgba('red', 0.7), False: to_rgba('black', 0.1)}
                )
            else:
                color_kws = dict(color='black')

            fig, ax = plt.subplots()
            add_unity_line(ax)

            orn_col = f'orn{col_suffix}'
            model_col = f'model{col_suffix}'

            lineplot_kws = dict(
                ax=ax, data=df, x=orn_col, y=model_col, linestyle='',
            )
            lineplot_kws = {**lineplot_kws, **color_kws}

            marker_only_kws = dict(
                markers=True, marker='o', errorbar=None,

                # to remove white edge of markers (were not respecting alpha)
                # (seem to work file w/ alpha, at least when set in non-'palette' case
                # below... was probably an issue when using hue/palette?)
                markeredgecolor='none',
            )
            # TODO should point / error display not be consistent between this and S1C /
            # 2E?
            err_only_kws = dict(
                markers=False, errorbar=errorbar, err_style='bars', seed=bootstrap_seed,
                # TODO make these thinner (to not need such fine tuning on alpha?)?
            )
            # more trouble than worth w/ palette (where values are 4 tuples w/ alpha)
            if 'palette' not in color_kws:
                # seems to default to white otherwise
                marker_only_kws['markeredgecolor'] = color_kws['color']
                # TODO if i like, refactor to share w/ other seed_errorbar plots?
                # TODO TODO like 'None' more than 'white' here? for some other pltos
                # (mainly those w/ lines thru them too), i liked 'white' more.
                marker_only_kws['markerfacecolor'] = 'None'

                # TODO still want some alpha < 1, when just showing edge (not face) of
                # markers?
                #
                # .3 too high, .2 pretty good, .15 maybe too low
                marker_only_kws['alpha'] = 0.175

                # 0.5 maybe verging on too low by itself, but still bit too crowded when
                # overlapping. .4 pretty good
                err_only_kws['alpha'] = 0.35

            # no other way I could find to get separate alpha for markers and errorbars,
            # other than to make 2 calls. setting alpha in kws led to a duplicate kwarg
            # error (rather than overwriting one from general kwargs).

            # plot points
            sns.lineplot(**lineplot_kws, **marker_only_kws)

            if errorbar is not None:
                # plot errorbars
                sns.lineplot(**lineplot_kws, **err_only_kws)

            ax.set_xlabel(f'{metric_name} of {orn_label_part} tuning (observed)'
                f'\n{help_str}'
            )
            if 'pn2kc_connections' in model_kws:
                ax.set_ylabel(
                    f'{metric_name} of {model_kws["pn2kc_connections"]} model KCs'
                )
            else:
                ax.set_ylabel(f'{metric_name} of model KCs')

            metric_max = max(df[model_col].max(), df[orn_col].max())
            metric_min = min(df[model_col].min(), df[orn_col].min())

            assert metric_max <= plot_max, \
                f'{param_dir}\n{desc=}: {metric_max=} > {plot_max=}'
            assert metric_min >= plot_min, \
                f'{param_dir}\n{desc=}: {metric_min=} < {plot_min=}'

            ax.set_xlim([plot_min, plot_max])
            ax.set_ylim([plot_min, plot_max])

            # should give us an Axes that is of square size in figure coordinates
            ax.set_box_aspect(1)

            if 'seed' in df.columns:
                # averaging correlations over seed, before calculating bootstrapped
                # spearman (so that CI is correct. otherwise showed no error)
                for_spearman = df.groupby(['odor1','odor2'])[[model_col,orn_col]].mean()
            else:
                for_spearman = df.copy()

            spear_text, _, _, _, _ = bootstrapped_corr(for_spearman, model_col, orn_col,
                # TODO delete (for debugging)
                # don't want to do for 'orn-est-spike-delta' case, as would need code
                # changes and don't care about that
                _plot_dir=param_dir if orn_fname_part == 'orn-raw-dff' else None,
                #
            )

            if errorbar is None:
                ax.set_title(f'{title}\n\n{spear_text}')
            else:
                ax.set_title(f'{title}\n\n{seed_err_text}\n{spear_text}')

            savefig(fig, param_dir, f'model_vs_{orn_fname_part}{fname_suffix}')


        _save_kc_vs_orn_corr_scatterplot('correlation distance')
        _save_kc_vs_orn_corr_scatterplot('correlation')


        # TODO will probably need to pass _index here to have invert_corr_triangular
        # work.... (doesn't seem like we've been failing w/ same AssertionError i had
        # needed to catch in other place...)
        square_mean_orn_corrs = _resort_corr(invert_corr_triangular(mean_orn_corrs),
            panel
        )

        # stripping conc to match processing of `pearson` above
        square_mean_orn_corrs = _strip_index_and_col_concs(square_mean_orn_corrs)
        try:
            assert pearson.index.equals(square_mean_orn_corrs.index)
            assert pearson.columns.equals(square_mean_orn_corrs.columns)
        # TODO still reachable? delete?
        except AssertionError:
            # TODO care enough to find intersection and just take diff there?
            # (or dropna and resort?)
            print(f'not plotting corr diff wrt {orn_label_part} (index mismatch)')
            return

        corr_diff = pearson - square_mean_orn_corrs
        plot_corr(corr_diff, param_dir, f'model_vs_{orn_fname_part}_corr_diff',
            title=title, xlabel=f'model KC corr - {orn_label_part} corr'
        )
        # square_mean_corrs should be plotted elsewhere (potentially in diff places
        # depending on whether input is Hallem vs pebbled data?)
        # TODO check + comment where each should be saved?


    if comparison_orns is not None:
        if type(comparison_orns) is dict:
            for desc, comparison_data in comparison_orns.items():
                _compare_model_kc_to_orn_data(comparison_data, desc)
        else:
            _compare_model_kc_to_orn_data(comparison_orns)


    if comparison_kc_corrs is not None:
        # TODO assert input odors match comparison_kc_corrs odors exactly?

        # TODO just do this unconditionally like in comparison_orns code? or make that
        # part explicit too?
        if _strip_concs_comparison_kc_corrs:
            comparison_kc_corrs = _strip_index_and_col_concs(comparison_kc_corrs)

            n_combos_before = len(model_corr_df[['odor1', 'odor2']].drop_duplicates())
            # TODO use parse_odor_name in other stripping fn here too?
            model_corr_df['odor1'] = model_corr_df.odor1.apply(olf.parse_odor_name)
            model_corr_df['odor2'] = model_corr_df.odor2.apply(olf.parse_odor_name)

            n_combos_after = len(model_corr_df[['odor1', 'odor2']].drop_duplicates())
            assert n_combos_before == n_combos_after

        kc_corrs = corr_triangular(comparison_kc_corrs)
        kc_corrs.name = 'observed_kc_corr'

        df = model_corr_df.merge(kc_corrs, on=['odor1', 'odor2'])

        # converting to correlation distance, like in matt's
        df['model_corr_dist'] = 1 - df['model_corr']
        df['observed_kc_corr_dist'] = 1 - df['observed_kc_corr']

        fig, ax = plt.subplots()

        # doing this first so everything else gets plotted over it
        # TODO why do points from call below seem to be plotted under this? way to force
        # a certain Z order?
        add_unity_line(ax)

        sns.regplot(data=df, x='observed_kc_corr_dist', y='model_corr_dist',
            x_estimator=np.mean, x_ci=None, color='black', scatter_kws=dict(alpha=0.3),
            fit_reg=False
        )

        # averaging over 'seed' level to get mean correlation for each pair, because we
        # don't show error for each point in this plot (i.e. error across seeds). we
        # only show a CI for the regression line shown (handled in regplot call below)
        if 'seed' in df.columns:
            df = df.groupby(['odor1','odor2']).mean().reset_index()

        # TODO assert len(df) always n_choose_2(n_odors) at this point?
        # (seems true in pebbled/hemibrain at least. check uniform)

        corr_dist_max = max(df.model_corr_dist.max(), df.observed_kc_corr_dist.max())
        corr_dist_min = min(df.model_corr_dist.min(), df.observed_kc_corr_dist.min())
        plot_max = 1.3
        plot_min = 0.0
        assert corr_dist_max <= plot_max, f'{param_dir}\n{corr_dist_max=} > {plot_max=}'
        assert corr_dist_min >= plot_min, f'{param_dir}\n{corr_dist_min=} < {plot_min=}'

        # need to set these before regplot call below (which makes regression line +
        # CI), so that the line actually goes to these limits.
        ax.set_xlim([plot_min, plot_max])
        ax.set_ylim([plot_min, plot_max])

        # NOTE: none of the KC vs model KC scatter plots in preprint have seed-error
        # shown, so not including errorbar=seed_errorbar here. just want error on
        # regression line in this plot, which we have (and we are happy w/ default 95%
        # CI on mean for that)
        sns.regplot(data=df, x='observed_kc_corr_dist', y='model_corr_dist',
            color='black', scatter=False, truncate=False, seed=bootstrap_seed
        )

        spear_text, _, _, _, _ = bootstrapped_corr(df, 'model_corr_dist',
            'observed_kc_corr_dist', method='spearman',
            # TODO delete (for debugging)
            _plot_dir=param_dir,
            #
        )
        ax.set_title(f'{title}\n\n{spear_text}')

        ax.set_xlabel('KC correlation distance (observed)')
        ax.set_ylabel('model KC correlation distance')

        # should give us an Axes that is of square size in figure coordionates
        ax.set_box_aspect(1)

        # TODO rename to indicate they are corr-dists, not just corrs (no other version
        # of the plot tho...)?
        #
        # to reproduce preprint figures 3 Di/Dii
        savefig(fig, param_dir, 'model_vs_kc_corrs')


    # TODO why am i getting the following error w/ my current viz.clustermap usage?
    # 3625, in _dendrogram_calculate_info
    #     _dendrogram_calculate_info(
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/scipy/cluster/hierarchy.py", line 3658, in _dendrogram_calculate_info
    #     _dendrogram_calculate_info(
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/scipy/cluster/hierarchy.py", line 3625, in _dendrogram_calculate_info
    #     _dendrogram_calculate_info(
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/scipy/cluster/hierarchy.py", line 3625, in _dendrogram_calculate_info
    #     _dendrogram_calculate_info(
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/scipy/cluster/hierarchy.py", line 3555, in _dendrogram_calculate_info
    #     _append_singleton_leaf_node(Z, p, n, level, lvs, ivl,
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/scipy/cluster/hierarchy.py", line 3433, in _append_singleton_leaf_node
    #     ivl.append(str(int(i)))
    # RecursionError: maximum recursion depth exceeded while getting the str of an object

    # https://github.com/scipy/scipy/issues/7271
    # https://github.com/MaayanLab/clustergrammer/issues/34
    sys.setrecursionlimit(100000)
    # TODO maybe don't need to change sys setrecursionlimit now that i'm dropping silent
    # cells?

    # TODO try a version of this using first N (< n_seeds) seeds? try all (i assume it'd
    # be much too slow, and also unreadable [assuming we try to show all cells, and not
    # cluster means / similar reduction])?
    #
    # only including silent here so we can count them in line below
    to_cluster = responses_including_silent
    clust_suffix = ''

    if n_seeds > 1:
        first_seed = to_cluster.index.get_level_values('seed')[0]
        to_cluster = to_cluster.loc[first_seed]
        clust_suffix = '_first-seed-only'

    silent_cells = (to_cluster == 0).all(axis='columns')

    if silent_cells.all():
        # TODO also return before generating the corr plots (+ any others) above?
        # TODO err instead?
        #
        # TODO why was* (can not currently repro) this the case for ALL attempts in
        # 'kiwi' case now?  really need steps so different from 'megamat' case? if so,
        # why? or am i just not calling it right at all for some reason (related to
        # that failing check to repro output w/ fixed wAPLKC/fixed_thr?)
        warn('all model cells were silent! returning before generating further plots!')
        return params_for_csv

    # ~30" height worked for ~1837 cells, but don't need all that when not plotting
    # silent cells
    cg = cluster_rois(to_cluster[~ silent_cells].T, odor_sort=False, figsize=(7, 12),
        # seeing if this resolves recursion error...
        # does not fix it.
        #optimal_ordering=False
    )

    cg.fig.suptitle(f'{title_including_silent_cells}\n\n'
        # TODO just define n_silent_cells alongside responses_including_silent/responses
        # def, then remove separate use of `silent_cells` here (+ use responses instead
        # of responses_including_silent)?
        # (doing it earlier would be complicated/impossible in n_seeds > 1 case
        # though...)
        f'{silent_cells.sum()} silent cells / {len(to_cluster)} total'
    )
    savefig(cg, param_dir, f'responses_nosilent{clust_suffix}')

    # TODO TODO also plot wPNKC (clustered?) for matts + my stuff?
    # TODO same for other model vars? thresholds?

    # TODO corr diff plot too (even if B doesn't want to plot it for now)?

    # TODO and (maybe later) correlation diffs wrt model w/ tuned params

    # TODO assert no sparsity (/ value) goes outside cbar/scale limits
    # (do in sparsity plotting fn?)

    sparsity_per_odor = _per_odor_tidy_model_response_rates(responses_including_silent
        ).set_index(odor_col)

    if comparison_responses is not None:
        comparison_sparsity_per_odor = _per_odor_tidy_model_response_rates(
            comparison_responses
        )
        comparison_sparsity_per_odor = comparison_sparsity_per_odor.sort_values(
            sparsity_col
        )
        # TODO also sort correlation odors by same order?
        # TODO assert set of odors are the same first
        sparsity_per_odor = sparsity_per_odor.loc[comparison_sparsity_per_odor.odor1]
    else:
        comparison_sparsity_per_odor = None

    sparsity_per_odor = sparsity_per_odor.reset_index()

    sparsity_per_odor.odor1 = sparsity_per_odor.odor1.map(lambda x: x.split(' @ ')[0])

    if comparison_responses is not None:
        # TODO need (just to remove diff numbering in index, wrt sparsity_per_odor, in
        # case that changes behavior of some plotting...)?
        # TODO why drop=True here, but not in sparsity_per_odor.reset_index() above?
        comparison_sparsity_per_odor = comparison_sparsity_per_odor.reset_index(
            drop=True
        )
        # TODO refactor (duped above)?
        comparison_sparsity_per_odor.odor1 = comparison_sparsity_per_odor.odor1.map(
            lambda x: x.split(' @ ')[0]
        )
        assert comparison_sparsity_per_odor.odor1.equals(sparsity_per_odor.odor1)

    if responses_to == 'hallem':
        # assuming megamat for now (otherwise this would be empty)
        megamat_sparsity_per_odor = sparsity_per_odor.loc[
            sparsity_per_odor.odor1.isin(panel2name_order['megamat'])
        ]
        # this is only used in sensitivity analysis now anyway. would need to also
        # subset this if not.
        assert comparison_sparsity_per_odor is None
        plot_sparsity_per_odor(megamat_sparsity_per_odor, comparison_sparsity_per_odor,
            '_megamat'
        )

    combined_fig, sparsity_ax = plot_sparsity_per_odor(sparsity_per_odor,
        comparison_sparsity_per_odor, ''
    )

    #sparsity_ylim_max = 0.5
    # to exceed .706 in (fixed_thr=120.85, wAPLKC=0.0) param case
    sparsity_ylim_max = 0.71
    sparsity_ax.set_ylim([0, sparsity_ylim_max])
    # comparison_sparsity_per_odor isn't being pushed to the same extreme, and should be
    # well within this limit.
    assert not (sparsity_per_odor[sparsity_col] > sparsity_ylim_max).any()

    # https://stackoverflow.com/questions/33264624
    # NOTE: without other fiddling, need to keep references to both of these axes, as
    # the Axes created by `ax.twinx()` is what we need to control the ylabel
    # https://stackoverflow.com/questions/54718818
    n_odor_ax_for_ylabel = sparsity_ax.twinx()
    n_odor_ax = n_odor_ax_for_ylabel.twiny()

    fig, ax = plt.subplots()
    plot_n_odors_per_cell(responses_including_silent, ax,
        title=title_including_silent_cells
    )
    if comparison_responses is not None:
        plot_n_odors_per_cell(comparison_responses, ax, label_suffix=' (tuned)',
            color='gray', title=title_including_silent_cells
        )

    savefig(fig, param_dir, 'n_odors_per_cell')

    if responses_to == 'hallem':
        fig, ax = plt.subplots()
        # TODO assert this is getting just megamat odors (/reimplement so it only could)
        # (b/c prior sorting, that should have put all them before rest of hallem odors,
        # they should be)
        # TODO factor out + use megamat subsetting fn here
        plot_n_odors_per_cell(responses_including_silent.iloc[:, :17], ax,
            title=title_including_silent_cells
        )

        assert comparison_responses is None
        # if assertion fails, will also need to subset comparison_responses to megamat
        # odors (in commented code below)
        '''
        if comparison_responses is not None:
            plot_n_odors_per_cell(comparison_responses, ax, label_suffix=' (tuned)',
                color='gray', title=title_including_silent_cells
            )
        '''
        savefig(fig, param_dir, 'n_odors_per_cell_megamat')

    # only currently running sensitivity analysis in pebbled/hemibrain case.
    # some of code below (all of which should deal with sensitivity analysis in some
    # way, from here on) may not work w/ multiple seeds. could delete this early return
    # and try though.
    if n_seeds > 1:
        assert not sensitivity_analysis
        return params_for_csv

    # only want to save this combined plot in case of senstivity analysis
    # (where we need as much space as we can save)
    if _in_sens_analysis:
        assert fixed_thr is not None and wAPLKC is not None

        plot_n_odors_per_cell(responses_including_silent, n_odor_ax,
            ax_for_ylabel=n_odor_ax_for_ylabel, linestyle='dashed', log_yscale=True
        )

        if comparison_responses is not None:
            plot_n_odors_per_cell(comparison_responses, n_odor_ax,
                ax_for_ylabel=n_odor_ax_for_ylabel, label_suffix=' (tuned)',
                color='gray', linestyle='dashed', log_yscale=True
            )

        # TODO figure out how to build one legend from the two axes?
        if _add_combined_plot_legend:
            # planning to manually adjust when assembling figure
            sparsity_ax.legend(loc='upper right')
            n_odor_ax.legend(loc='center right')

        savefig(combined_fig, param_dir, 'combined_odors-per-cell_and_sparsity')

    if sensitivity_analysis:
        # TODO probably remove these assertions (to do sensitivity analysis around each
        # of kiwi/control runs, where they had inh params set from one run w/
        # kiwi+control data)
        #
        # TODO delete
        print('want to restore some of these assertions?')
        #assert fixed_thr is None and wAPLKC is None
        #assert ('target_sparsity' in model_kws and
        #    model_kws['target_sparsity'] is not None
        #)

        shared_model_kws = {k: v for k, v in model_kws.items() if k not in (
            # plot_dir would conflict with first positional arg of
            # fit_and_plot_mb_model, and we don't want plots for sensitivity analysis
            # subcalls anyway (could fix if we did want those plots).
            'plot_dir',
            # excluding target_sparsity b/c that is mutually exclusive w/ fixing
            # threshold and KC<->APL inhibition, as all these calls will.
            'target_sparsity',
            # will default to False (via fit_mb_model default) once I remove this, which
            # is what I want
            'repro_preprint_s1d',
        )}
        # TODO try to not need to specially treat wAPLKC / fixed_thr (instead trying to
        # continue handling just via model_kws) tho? make things too difficult?

        tuned_fixed_thr = param_dict['fixed_thr']
        tuned_wAPLKC = param_dict['wAPLKC']
        assert tuned_fixed_thr is not None and tuned_wAPLKC is not None

        checks = True
        # TODO move to some kind of unit test. in olfsysm, maybe?
        if checks:
            print('checking we can recreate responses by hardcoding tuned '
                'fixed_thr/wAPLKC...', end=''
            )
            # TODO TODO figure out why this wasn't working on some runs of
            # kiwi/control data (re-running w/ `-i model` seemed to fix it, but unclear
            # on why that would be. using cache after break it again?) (doesn't seem
            # like it...)
            # TODO why *was* responses2.sum().sum() == 0 in kiwi/control case???
            # (prob same reason sens analysis failing there)
            # (not sure i can repro, same as w/ failing assertion below)
            #
            # TODO silence output here? makes surrounding prints hard to follow
            # TODO also check spike_counts (2nd / 4 returned values)?
            responses2, _, _, _ = fit_mb_model(sim_odors=sim_odors,
                fixed_thr=tuned_fixed_thr, wAPLKC=tuned_wAPLKC, **shared_model_kws
            )
            assert responses_including_silent.equals(responses2)
            print(' we can!\n')

        parent_output_dir = param_dir / 'sensitivity_analysis'

        # deleting this directory (and all contents) before run, to clear plot dirs from
        # param choices only in previous sweeps (and tried.csv with same)
        if parent_output_dir.exists():
            # TODO TODO warn / err if -c (can't check new plots against existing if we
            # are deleting whole dir...)?
            # TODO make a rmtree wrapper (->use here and elsewhere) and always warn/err
            # if -c?
            warn(f'deleting {parent_output_dir} and all contents!')
            shutil.rmtree(parent_output_dir)

        # savefig will generally do this for us below
        parent_output_dir.mkdir(exist_ok=True)

        tried_param_cache = parent_output_dir / 'tried.csv'
        # should never exist as we're currently deleting parent dir above
        # (may change to not delete above though, so keeping this)
        if tried_param_cache.exists():
            tried = pd.read_csv(tried_param_cache)
            # TODO assert columns are same as below (refactor col def above conditional)
        else:
            tried = pd.DataFrame(columns=['fixed_thr', 'wAPLKC', 'sparsity'])

        # needs to be odd so grid has tuned values (that produce outputs used elsewhere
        # in paper) as center.
        n_steps = 3

        # TODO might need to be <1 to have lower end be reasonable?  but that prob won't
        # be enough (meaning? delete comment or elaborate) for upper ends...
        #
        # had previously tried up to at least 3, as well (not sure how it compared now
        # though...)
        #
        # for getting upper left 3x3 from original 4x4 (which was generated w/ this
        # param doubled, and n_steps=5 rather than 3)
        fixed_thr_param_lim_factor = 0.5

        # TODO TODO try seeing if we can push this high enough to start getting missing
        # correlations. 1000? why only getting those for high fixed_thr? and where
        # exactly do they come from?
        #
        # NOTE: was not seemingly able to get odors w/ no cells responding to them by
        # increasing this param (up to 100.0)...
        # 10.0 was used for most of early versions of this
        wAPLKC_param_lim_factor = 5.0

        # TODO TODO expose as kwarg , so i can set a wider one for kiwi/control case
        # than i used for megamat paper supp figures (-> maybe use these wider commented
        # vals)
        # (need special handling so they don't end up in param_strs / CSVs in places i
        # don't want?)
        #
        # TODO delete (maybe first use to regen sparsities wide CSV -> pick steps?)
        '''
        n_steps = 9
        fixed_thr_param_lim_factor = 1.0
        wAPLKC_param_lim_factor = 10.0
        '''
        #

        drop_nonpositive_fixed_thr = True
        drop_negative_wAPLKC = True

        def steps_around_tuned(tuned_param, param_lim_factor, param_name, *,
            drop_negative=True, drop_zero=False):

            step_size = tuned_param * param_lim_factor
            param_steps = np.linspace(tuned_param - step_size, tuned_param + step_size,
                num=n_steps
            )

            if drop_zero:
                assert drop_negative

            if drop_negative:
                param_steps[param_steps < 0] = 0

                if (param_steps == 0).any():
                    last_zero_idx = np.argwhere(param_steps == 0)[-1, 0]

                    first_idx_to_use = last_zero_idx
                    if drop_zero:
                        first_idx_to_use += 1

                    if first_idx_to_use > 0:
                        warn(f'{param_name=}: setting lowest {first_idx_to_use} steps '
                            '(negative) to 0'
                        )

                    param_steps = param_steps[first_idx_to_use:]

                    if not drop_zero:
                        bottom_step_size = np.diff(param_steps[:2])[0]
                        warn(f'{param_name=}: step sizes uneven after setting negative '
                            f'values to 0. step size previously all {step_size:.4f} '
                            f'(from {param_lim_factor=}). bottom step now '
                            f'{bottom_step_size:.4f}'
                        )

            assert len(param_steps) >= 3, ('{param_name=}: must have at least 1 step on'
                ' either side of tuned param'
            )

            assert np.isclose(tuned_param, param_steps).sum() == 1, \
                f'{param_name=}: tuned param not in steps'

            # TODO or (given how it's actually implemented) are negative values
            # meaningful for threshold? (surely not for wAPLKC)
            assert (param_steps >= 0).all()

            return param_steps

        # TODO try 0 for min of fixed_thr_steps (remove drop_zero=True, and tweak step
        # size to get at least 1 <=0) (current steps not clipped by it)?
        # is 0 the lowest value that maximizes response rate? or are negative vals
        # meaningful given how this is implemented?
        # (would allow me to simplify code slightly if this could be handled as wAPLKC)
        fixed_thr_steps = steps_around_tuned(tuned_fixed_thr,
            fixed_thr_param_lim_factor, 'fixed_thr', drop_negative=True,
            drop_zero=drop_nonpositive_fixed_thr
        )
        wAPLKC_steps = steps_around_tuned(tuned_wAPLKC, wAPLKC_param_lim_factor,
            'wAPLKC', drop_negative=drop_negative_wAPLKC
        )

        print(f'{tuned_fixed_thr=}')
        print(f'{tuned_wAPLKC=}')
        print()
        print('parameter steps around sparsity-tuned values (above):')
        print(f'{fixed_thr_steps=}')
        print(f'{wAPLKC_steps=}')
        print()

        step_choice_params = pd.Series({
            'tuned_fixed_thr': tuned_fixed_thr,
            'tuned_wAPLKC': tuned_wAPLKC,
            'n_steps': n_steps,

            'fixed_thr_param_lim_factor': fixed_thr_param_lim_factor,
            'wAPLKC_param_lim_factor': wAPLKC_param_lim_factor,

            'drop_negative_wAPLKC': drop_negative_wAPLKC,
            'drop_nonpositive_fixed_thr': drop_nonpositive_fixed_thr,

            # should be fully determined by above, but just including for easier
            # inspection
            'fixed_thr_steps': fixed_thr_steps,
            'wAPLKC_steps': wAPLKC_steps,
        })
        to_csv(step_choice_params, parent_output_dir / 'step_choices.csv',
            # so column name '0' doesn't get added (also added if doing ser.to_frame())
            header=False
        )

        # TODO try to just derive from tried, but this seemed easier for now.
        tried_wide = pd.DataFrame(data=float('nan'), columns=fixed_thr_steps,
            index=wAPLKC_steps
        )
        tried_wide.columns.name = 'fixed_thr'
        tried_wide.index.name = 'wAPLKC'

        # should be something that won't appear in actual computed values. NaN may
        # appear in computed values. after loop, we check that we no longer have any of
        # these.
        corr_placeholder = 10

        row_index = pd.MultiIndex.from_product([fixed_thr_steps, wAPLKC_steps],
            names=['fixed_thr', 'wAPLKC']
        )
        col_index = corr_triangular(pearson).index
        pearson_at_each_param_combo = pd.DataFrame(index=row_index, columns=col_index,
            data=corr_placeholder
        )

        # TODO any point in having this if we are deleting root of all these above?
        # delete?
        ignore_existing = True

        _add_combined_plot_legend = True

        for fixed_thr, wAPLKC in tqdm(itertools.product(fixed_thr_steps, wAPLKC_steps),
            total=len(fixed_thr_steps) * len(wAPLKC_steps),
            unit='fixed_thr+wAPLKC combos'):

            print(f'{fixed_thr=}')
            print(f'{wAPLKC=}')

            # NOTE: created by inner fit_and_plot... call below
            output_dir = parent_output_dir / f'thr{fixed_thr:.2f}_wAPLKC{wAPLKC:.2f}'

            if output_dir.exists() and not ignore_existing:
                print(f'{output_dir} already existed. skipping!')
                continue

            if ((tried.fixed_thr <= fixed_thr) & (tried.wAPLKC <= wAPLKC) &
                (tried.sparsity < min_sparsity)).any():

                print(f'sparsity would be < {min_sparsity=}')
                continue

            elif ((tried.fixed_thr >= fixed_thr) & (tried.wAPLKC >= wAPLKC) &
                    (tried.sparsity > max_sparsity)).any():

                print(f'sparsity would be > {max_sparsity=}')
                continue

            curr_params = fit_and_plot_mb_model(output_dir,
                comparison_responses=responses_including_silent,
                sim_odors=sim_odors, sensitivity_analysis=False, _in_sens_analysis=True,
                fixed_thr=fixed_thr, wAPLKC=wAPLKC,
                _add_combined_plot_legend=_add_combined_plot_legend,
                # not passing param_dir_prefix here, b/c that should be in parent
                # directory, and should be easy enough to keep track of from that
                extra_params=extra_params,
                **shared_model_kws
            )
            # should only be None in first_seed_only=True case, but not doing any
            # multi-seed runs w/ sensitivity analysis. only doing sensitivity analysis
            # for hemibrain + pebbled case currently.
            assert curr_params is not None

            # (added only if wAPLKC/fixed_thr passed)
            pearson = curr_params['pearson']

            pearson_ser = corr_triangular(pearson)
            assert pearson_ser.index.equals(pearson_at_each_param_combo.columns)
            pearson_at_each_param_combo.loc[fixed_thr, wAPLKC] = pearson_ser

            sparsity = curr_params['sparsity']
            print(f'sparsity={sparsity:.3g}')

            if output_dir.exists():
                tried_wide.loc[wAPLKC, fixed_thr] = sparsity

                # only want for first plot
                if _add_combined_plot_legend:
                    _add_combined_plot_legend = False

            tried = tried.append(
                {
                    'fixed_thr': fixed_thr, 'wAPLKC': wAPLKC, 'sparsity':  sparsity,
                },
                ignore_index=True
            )
            tried = tried.sort_values(['fixed_thr','wAPLKC'])

            # can't use my to_csv as it currently errs if same CSV would get written >1
            # time in a given run
            tried.to_csv(tried_param_cache, index=False)

            print()

        # to make rows=fixed_thr, cols=wAPLKC (consistent w/ how i had been laying out
        # the figure grids)
        tried_wide = tried_wide.T

        # (this, but not columns.name, makes it into CSV. it will be top left element.)
        tried_wide.index.name = 'rows=fixed_thr, cols=wAPLKC'
        # TODO rename var to match csv name (~similar)
        # TODO also add row / col index levels for what param_lim_factor we'd need to
        # get each of those steps?
        to_csv(tried_wide, parent_output_dir / 'sparsities_by_params_wide.csv')

        # NOTE: `not in set(...)` check probably doesn't work as intended w/ NaN,
        # but assuming corr_placeholder is not NaN, should be fine
        assert corr_placeholder not in set(
            np.unique(pearson_at_each_param_combo.values)
        )

        # TODO TODO how to deal w/ NaNs prior to spearman calc? do spearman calc in
        # a way that ignores NaN (pretty sure that's default behavior)?
        # TODO save one version w/ dropna first (to keep # of non-NaN input correlations
        # same across corrs that might or might not have NaN)?

        # after transposing, output corr will be of shape:
        # (# param combos, # param combos)
        spearman_of_pearsons = pearson_at_each_param_combo.T.corr(method='spearman')

        # TODO how to get text over (/to side of) ticklabels (to label full name of 2
        # params)? add support to viz.matshow for that?

        level_fn = lambda d: d['fixed_thr']
        group_text = True
        format_fixed_thr = lambda x: f'{x:.0f}'
        # trying to just use this to format last row/col index level (wAPLKC).
        # fixed_thr should be handled by group_text stuff, which i might want to change
        # handling of inside hong2p
        format_wAPLKC = lambda x: f'{x[1]:.1f}'
        xticklabels = format_wAPLKC
        yticklabels = format_wAPLKC

        fig, _ = viz.matshow(spearman_of_pearsons,
            cmap=diverging_cmap,
            vmin=-1.0, vmax=1.0, levels_from_labels=False,
            hline_level_fn=level_fn, vline_level_fn=level_fn,
            hline_group_text=group_text, vline_group_text=group_text,
            group_fontsize=10, xtickrotation='horizontal',
            # TODO change hong2p.viz to have any levels not used to group formatted into
            # label?
            xticklabels=xticklabels, yticklabels=yticklabels,
            vgroup_formatter=format_fixed_thr, hgroup_formatter=format_fixed_thr
        )
        fig.suptitle("Spearman of odor X odor Pearson correlations")
        savefig(fig, parent_output_dir, 'spearman_of_pearsons')

    return params_for_csv


# TODO also anchor path to script dir? would only be to support running from elsewhere,
# which i prob don't care about
remy_data_dir = Path('data/from_remy')
n_final_megamat_kc_flies = 4
n_megamat_odors = 17

# TODO refactor to share?
remy_date_col = 'date_imaged'
# she uses 'fly_num' same as I do, to number flies within each day (i.e. the numbers not
# unique across days). these two can generally be used to compute/lookup values for
# remy_fly_id.
remy_fly_cols = [remy_date_col, 'fly_num']

# 0, 1, ..., 3 (or all 0 in one of the CSVs, by accident, but that CSV redundant
# anyway)
remy_fly_id = 'acq'

# e.g. '1-5ol @ -3.0'
remy_odor_col = 'stim'

# TODO put in docstring which files we are loading from
def _load_remy_megamat_kc_responses(drop_nonmegamat: bool = True) -> pd.DataFrame:
    n_odors = len(megamat_odor_names)
    assert n_odors == n_megamat_odors

    fly_response_root = remy_data_dir / 'megamat17' / 'per_fly'
    response_file_to_use = 'xrds_suite2p_respvec_mean_peak.nc'
    # Remy confirmed it's this one
    response_calc_to_use = 'Fc_zscore'

    olddata_fly_response_root = remy_data_dir / '2024-11-12'
    olddata_response_file_to_use = 'xrds_responses.nc'

    # TODO is it a problem that we are using peak_amp here and something zscored above?
    # is this actually zscored too? matter?
    #
    # other variables in these Datasets:
    # Data variables:
    #     peak_amp          (trials, cells) float64 ...
    #     peak_response     (trials, cells) float64 ...
    #     bin_response      (trials, cells) int64 ...
    #     baseline_std      (trials, cells) float64 ...
    #     baseline_med      (trials, cells) float64 ...
    #     peak_idx          (trials, cells) int64 ...
    olddata_response_calc_to_use = 'peak_amp'

    if verbose:
        print()
        print('loading Remy megamat KC responses to compute (odor X odor) corrs:')

    _seen_date_fly_combos = set()
    mean_response_list = []

    for fly_dir in fly_response_root.glob('*/'):

        if not fly_dir.is_dir():
            continue

        # corresponding correlation .nc file in fly_dir / 'RDM_trials' should also be
        # equiv to one element of above `corrs`
        fly_response_dir = fly_dir / 'respvec'
        fly_response_file = fly_response_dir / response_file_to_use

        if verbose:
            print(fly_response_file)

        responses = xr.open_dataset(fly_response_file)

        date = pd.Timestamp(responses.attrs[remy_date_col])
        assert len(remy_fly_cols) == 2 and 'fly_num' == remy_fly_cols[1]
        # should already be an int, just weird numpy.int64 type, and not sure that
        # behaves same in sets (prob does).
        fly_num = int(responses.attrs['fly_num'])
        thorimage = responses.thorimage
        if verbose:
            # NOTE: responses.attrs[x] seems to be equiv to responses.x
            print('/'.join(
                [str(responses.attrs[x]) for x in remy_fly_cols] + [thorimage]
            ))

        # excluding thorimage, b/c also don't want 2 recordings from one fly making it
        # in, like happened w/ her precomputed corrs
        assert (date, fly_num) not in _seen_date_fly_combos
        _seen_date_fly_combos.add( (date, fly_num) )

        n_cells = responses.sizes['cells']
        if verbose:
            print(f'{n_cells=}')

        assert (responses.iscell == 1).all().item()
        assert len(responses.attrs['bad_trials']) == 0

        # TODO move to checks=True?
        all_xid_set = set(responses.xid0.values)
        # TODO factor out this assertion to hong2p.util (probably do something like this
        # in a lot of places. use in those places too.)
        assert all_xid_set == set(range(max(all_xid_set) + 1))

        # NOTE: isin(...) does not work here if input is a Python set()
        # (so keeping good_xid as a DataArray, or whatever type it is)
        good_xid = responses.attrs['good_xid']
        good_cells_mask = responses.xid0.isin(good_xid)

        good_xid_set = set(good_xid)
        # we have some xid0 values other than those in attrs['good_xid']
        # (so Remy did not pre-subset the data, and we should have all the cells)
        assert len(all_xid_set - good_xid_set) > 0
        #

        n_good_cells = good_cells_mask.sum().item()
        assert n_good_cells < n_cells

        n_bad_cells = (~ good_cells_mask).sum().item()
        if verbose:
            print(f'{n_bad_cells=}')

        assert (n_good_cells + n_bad_cells) == responses.sizes['cells']

        checks = True
        if checks:
            single_fly_nc_files = list(remy_binary_response_dir.glob((
                f'{format_date(date)}__fly{fly_num:>02}__*/'
                f'{remy_fly_binary_response_fname}'
            )))
            assert len(single_fly_nc_files) == 1
            fly_binary_response_file = single_fly_nc_files[0]

            binary_responses = load_remy_fly_binary_responses(fly_binary_response_file,
                reset_index=False
            )

            binary_response_xids = set(binary_responses.index.get_level_values('xid0'))
            # binary responses don't have a subset of the XID, they have all of them
            # (i.e. they haven't been subset to just the good_xid cells by Remy)
            assert binary_response_xids == all_xid_set
            # TODO use factored out version of this when i make it
            assert binary_response_xids == set(range(max(binary_response_xids) + 1))

            assert np.array_equal(
                binary_responses.index.to_frame(index=False)[['cells_level_0','xid0']],
                np.array([responses.cells, responses.xid0]).T
            )

            # seems pretty good:
            # responders? False
            # 305 good-XID-cells / 1634 cells (0.187)
            # responders? True
            # 2166 good-XID-cells / 2547 cells (0.850)
            def _print_frac_good_xid(gdf):
                responder_val_set = set(gdf.responder)
                assert len(responder_val_set) == 1
                responder_val = responder_val_set.pop()
                assert responder_val in (False, True)

                if responder_val:
                    print('among responders:')
                else:
                    print('among silent cells:')
                # TODO delete
                #print(f'responders? {responder_val}')

                # this isin DOES (pandas index LHS, not DataArray) work w/ python set
                # arg
                good_xid_mask = gdf.index.get_level_values('xid0').isin(good_xid_set)

                n_good_cells = good_xid_mask.sum()
                n_cells = len(good_xid_mask)
                good_cell_frac = n_good_cells / n_cells
                # TODO actually inspect these outputs -> decide i'm happy with them ->
                # only run this code if verbose (via settings checks flag above in this
                # fn)
                print(f'{n_good_cells} good-XID-cells / {n_cells} cells'
                    f' ({good_cell_frac:.3f})'
                )

            if verbose:
                binary_responses['responder'] = binary_responses.any(axis='columns')
                binary_responses.groupby('responder').apply(_print_frac_good_xid)
                print()

        # another way to do the same thing:
        # responses = responses.where(good_cells_mask, drop=True)
        responses = responses.sel(cells=good_cells_mask)
        assert responses.sizes['cells'] == n_good_cells

        # doing this doesn't seem to preserve attrs (some way to? or are they only for
        # Dataset not DataArray?). seems DataArray support .attrs, but may just need to
        # manually assign from DataSet?
        #
        # (odors X trials) X cells
        responses = responses[response_calc_to_use]

        # odors X cells
        mean_responses = responses.groupby('stim').mean(dim='trials')

        # the reset_index(drop=True) is to remove cell numbers (which currently have
        # missing cells, because of xid-based dropping above, which might be confusing)
        mean_response_df = mean_responses.to_pandas().T.reset_index(drop=True)
        mean_response_df.index.name = 'cell'

        # referring to flies this way should make it simpler to compare to corrs in CSVs
        # Remy gave me for making fig 2E (which have a 'datefly' column, that should be
        # formatted like this)
        datefly = f'{format_date(date)}/{fly_num}'
        mean_response_df = util.addlevel(mean_response_df, 'datefly', datefly)

        # just to conform to format in loop below. only one recording for each of these
        # flies.
        mean_response_df = util.addlevel(mean_response_df, 'thorimage', thorimage)

        mean_response_list.append(mean_response_df)

        if verbose:
            print()

    # TODO delete. just to try to get new concat of new + old data to be in a similar
    # format.
    new_mean_responses = pd.concat(mean_response_list, verify_integrity=True)
    #

    # TODO TODO refactor to share as much of body of loop w/ above (convert to one loop,
    # and just special case a few things based on parent dir?). currently copied from
    # loop above.
    #
    # for this old data, don't have the same set of cells across any of the multiple
    # recordings for any one fly. always gonna be a diff set of cells.
    for fly_dir in olddata_fly_response_root.glob('*/'):

        if not fly_dir.is_dir():
            continue

        # corresponding correlation .nc file in fly_dir / 'RDM_trials' should also be
        # equiv to one element of above `corrs`
        fly_response_dir = fly_dir / 'respvec'
        fly_response_file = fly_response_dir / olddata_response_file_to_use

        if verbose:
            print(fly_response_file)

        responses = xr.open_dataset(fly_response_file)

        date = pd.Timestamp(responses.attrs[remy_date_col])
        assert len(remy_fly_cols) == 2 and 'fly_num' == remy_fly_cols[1]
        # should already be an int, just weird numpy.int64 type, and not sure that
        # behaves same in sets (prob does).
        fly_num = int(responses.attrs['fly_num'])
        thorimage = responses.thorimage
        if verbose:
            # NOTE: responses.attrs[x] seems to be equiv to responses.x
            print('/'.join(
                [str(responses.attrs[x]) for x in remy_fly_cols] + [thorimage]
            ))

        n_cells = responses.sizes['cells']
        if verbose:
            print(f'{n_cells=}')

        # old data doesn't have the attributes iscell or bad_trials

        # from Remy's code snippet she sent on 2024-11-12 via slack
        good_cells_mask = responses['iscell_responder'] == 1

        n_good_cells = good_cells_mask.sum().item()
        assert n_good_cells < n_cells

        n_bad_cells = (~ good_cells_mask).sum().item()
        if verbose:
            print(f'{n_bad_cells=}')

        assert (n_good_cells + n_bad_cells) == responses.sizes['cells']

        # another way to do the same thing:
        # responses = responses.where(good_cells_mask, drop=True)
        responses = responses.sel(cells=good_cells_mask)
        assert responses.sizes['cells'] == n_good_cells

        responses = responses[olddata_response_calc_to_use]

        # odors X cells
        mean_responses = responses.groupby('stim').mean(dim='trials')

        # the reset_index(drop=True) is to remove cell numbers (which currently have
        # missing cells, because of dropping above, which might be confusing)
        mean_response_df = mean_responses.to_pandas().T.reset_index(drop=True)
        mean_response_df.index.name = 'cell'

        # referring to flies this way should make it simpler to compare to corrs in CSVs
        # Remy gave me for making fig 2E (which have a 'datefly' column, that should be
        # formatted like this)
        datefly = f'{format_date(date)}/{fly_num}'
        mean_response_df = util.addlevel(mean_response_df, 'datefly', datefly)

        # multiple recordings for most/all flies, presumably each w/ diff odors
        mean_response_df = util.addlevel(mean_response_df, 'thorimage', thorimage)

        mean_response_list.append(mean_response_df)
        if verbose:
            print()

    # TODO compare format to new_mean_responses (temp debug var)
    mean_responses = pd.concat(mean_response_list, verify_integrity=True)

    odors = [olf.parse_odor(x) for x in mean_responses.columns]

    names = [x['name'] for x in odors]
    assert megamat_odor_names - set(names) == set(), 'missing some megamat odors'

    # cast_int_concs=True to convert '-3.0' to '-3', to be consistent w/ mine
    odor_strs = [olf.format_odor(x, cast_int_concs=True) for x in odors]
    mean_responses.columns = odor_strs

    # so mean_of_fly_corrs works
    mean_responses.columns.name = 'odor1'

    # when also loading old (prior to final 4 flies) megamat data:
    # (set(names) - megamat_odor_names)={'PropAc', '1p3ol', 'pfo', 'g-6lac', 'eug',
    # 'd-dlac', 'MethOct', 'ECin'}
    if drop_nonmegamat:
        megamat_mean_responses = mean_responses.loc[:,
            [x in megamat_odor_names for x in names]
        ]
        if verbose:
            nonmegamat_odors = mean_responses.columns.difference(
                megamat_mean_responses.columns
            )
            warn('dropping the following non-megamat odors:\n'
                f'{pformat(list(nonmegamat_odors))}'
            )

        mean_responses = megamat_mean_responses

    return mean_responses


# TODO rename all of these fns to remove '_megamat' (unless i actually drop down to just
# megamat, but i don't think i want that?)? or just do it anyway to shorten these names?
def _remy_megamat_flymean_kc_corrs(ordered_pairs=None, **kwargs) -> pd.DataFrame:
    mean_responses = _load_remy_megamat_kc_responses(**kwargs)

    # TODO move some functionality like this into mean_of_fly_corrs (to average within
    # fly across recordings first)?
    recording_corrs = mean_responses.groupby(level=['datefly', 'thorimage'], sort=False
        ).apply(lambda x: corr_triangular(x.corr(), ordered_pairs=ordered_pairs))

    fly_corrs = recording_corrs.groupby(level='datefly', sort=False).mean()

    # TODO delete
    # TODO check above equiv to this, at least if we no longer load old data?
    # TODO check this works if there are multiple thorimage level values (i.e.
    # recordings) for one pair (e.g. 1-6ol, 2-but) for any fly. should average the corrs
    # first, then compute average across flies.
    #mean_corr = mean_of_fly_corrs(mean_responses.T, id_cols=['datefly'])

    return fly_corrs


# don't need ordered_pairs here b/c output of this fn should be square, so it no longer
# matters.
def load_remy_megamat_mean_kc_corrs(**kwargs) -> pd.DataFrame:
    """Returns mean of fly correlations, for Remy's 4 final megamat KC flies.

    Drops cells from bad clusters (as Remy does, using xarray attrs['good_xid'] that she
    sets to good clusters, excluding clusters of bad cells, which should mostly be
    silent cells) before computing correlations. The 3 trials for each odor are
    averaged together into a single odor X cell response matrix before computing each
    fly's correlation. Correlation is computed within each fly, and then the average is
    computed across these correlations. This should all be consistent with how Remy
    computes correlations.
    """
    fly_corrs = _remy_megamat_flymean_kc_corrs(**kwargs)
    mean_corr_ser = fly_corrs.mean()
    mean_corr = invert_corr_triangular(mean_corr_ser)
    return mean_corr


remy_2e_metric = 'correlation_distance'

# TODO TODO try a version of this w/ either hollow points or no points (to show small
# errorbars that would otherwise get subsumed into point)
_fig2e_shared_plot_kws = dict(
    x='odor_pair_str',
    y=remy_2e_metric,

    errorbar=seed_errorbar,
    seed=bootstrap_seed,
    err_kws=dict(linewidth=1.5),

    markersize=7,
    #markeredgewidth=0,
)

def _check_2e_metric_range(df) -> None:
    # TODO cases where i'd want to warn instead?
    """Raises AssertionError if data range seems inconsistent w/ `remy_2e_metric`.
    """
    # TODO TODO assert things seem consistent w/ being correlation distance (or at
    # least, not correlation)
    if remy_2e_metric == 'correlation_distance':
        metric = df[remy_2e_metric]
        # if it were < 0, would suggest it's a correlation, not a correlation DISTANCE
        assert metric.min() >= 0
        # do we actually have values over 1 always tho? can just remove this if need be
        assert metric.max() > 1
    else:
        # could also do similar for 'correlation', but only ever using this one
        raise NotImplementedError("checking range only supported for remy_2e_metric="
            "'correlation_distance'"
        )

# TODO add some kind of module level dict of fig ID -> pair_order, and use to check each
# fig is getting the same pair_order across these two calls? or use so that only first
# call even takes pair_order, but then assert model pairs are a subset in the
# subsequence call(s)?
#
# need @no_constrained_layout since otherwise FacetGrid creation would warn with
# Warning: The figure layout has changed to tight
# (since my MPL config has constrained layout as default)
@no_constrained_layout
def _create_2e_plot_with_obs_kc_corrs(df_obs: pd.DataFrame, pair_order: np.array, *,
    fill_markers=True) -> sns.FacetGrid:

    _check_2e_metric_range(df_obs)

    odor_pair_set = set(pair_order)
    assert odor_pair_set == set(df_obs.odor_pair_str.unique())
    assert len(odor_pair_set) == len(pair_order)

    # don't have any identity correlations (odors correlated with themselves)
    assert not (df_obs.abbrev_row == df_obs.abbrev_col).any()

    color = 'k'

    if fill_markers:
        marker_kws = dict(markeredgewidth=0)
    else:
        # TODO TODO why are lines on these points thinner than in model corr plot call
        # (below)? (was because linewidth)
        marker_kws = dict(markerfacecolor='white', markeredgecolor=color)

    # other types besides array might work for pair_order, but I've only been using
    # arrays (as in Remy's code I adapted from)
    g = sns.catplot(
        data=df_obs,

        # TODO work to omit if input has x='odor_pair_str' values in sorted order i
        # want (and would it matter if subsequent calls had same order, or would it be
        # aligned?)
        # TODO what about if x column is a pd.Categorical(..., ordered=True)
        # (and if there are sometimes cases where data isn't aligned correctly across
        # calls, does this change the situation?)
        order=pair_order,

        kind='point',

        # TODO so it's jittering? can i seed that? not that it really matters, except
        # for running w/ -c flag...  i'm assuming seed= doesn't also seed jitter?
        # (haven't had -c flag trip, so i'm assuming it's not actually jittering [maybe
        # not enough data that there is a need?] or same seed controls that)
        #
        # jitter=False,

        color=color,

        aspect=2.5,
        height=7,
        #linewidth=1,

        **_fig2e_shared_plot_kws,
        **marker_kws
    )

    # test output same whether input is 'correlation' or 'correlation_distance', as
    # expected.
    pair_metrics = []
    for _, gdf in df_obs.groupby('odor_pair_str'):
        pair_metrics.append(gdf[remy_2e_metric].to_numpy())

    # one way ANOVA (null is that groups have same population mean. groups can be diff
    # sizes)
    result = f_oneway(*pair_metrics)
    # from scipy docs:
    # result.statistic: "The computed F statistic of the test."
    # result.pvalue: "The associated p-value from the F distribution."

    g.ax.set_title(
        f'{len(odor_pair_set)} non-identity odor pairs\n'
        # .2E will show 2 places after decimal w/ exponent (scientific notation)
        f'(one way ANOVA) F-statistic: {result.statistic:.2f}, p={result.pvalue:.2E}'
    )

    return g


@no_constrained_layout
def _2e_plot_model_corrs(g: sns.FacetGrid, df: pd.DataFrame, pair_order: np.ndarray,
    n_first_seeds: Optional[int] = n_first_seeds_for_errorbar, **kwargs) -> None:

    _check_2e_metric_range(df)

    if n_first_seeds is not None and 'seed' in df.columns:
        df = select_first_n_seeds(df, n_first_seeds=n_first_seeds)

    # TODO some way to get hue/palette to work w/ markeredgecolor? i assume not
    if 'hue' not in kwargs:
        assert 'color' in kwargs
        # TODO like? factor to share w/ other seed_errorbar plots?
        marker_kws = dict(markerfacecolor='None', markeredgecolor=kwargs['color'])
    else:
        # TODO keep? remy had before, but obviously prevents markeredgecolor working in
        # above case. not sure i care about this in hue/palette case.
        marker_kws = dict(markeredgewidth=0)

    sns.pointplot(data=df,
        order=pair_order,

        linestyle='none',
        ax=g.ax,

        **_fig2e_shared_plot_kws, **kwargs, **marker_kws
    )


@no_constrained_layout
def _finish_remy_2e_plot(g: sns.FacetGrid, *, n_first_seeds=n_first_seeds_for_errorbar
    ) -> None:

    g.set_axis_labels('odor pairs', remy_2e_metric)
    # 0.9 wasn't enough to have axes title and suptitle not overlap
    g.fig.subplots_adjust(bottom=0.2, top=0.85)

    # TODO use paper's 1.2 instead? or just leave unset? just set min to 0?
    # TODO TODO why does it seem to be showing as 1.2 w/ this at 1.4 anyway?
    g.ax.set_ylim(0, 1.4)

    seed_err_text, _ = _get_seed_err_text_and_fname_suffix(n_first_seeds=n_first_seeds)

    g.fig.suptitle(f'odor-odor {remy_2e_metric}\n{seed_err_text}')

    sns.despine(fig=g.fig, trim=True, offset=2)
    g.ax.xaxis.set_tick_params(rotation=90, labelsize=8)

    # TODO move legend to bottom left (in top right now)?


# TODO rename to ...corr_dists or something?
def load_remy_2e_corrs(plot_dir=None, *, use_preprint_data=False) -> pd.DataFrame:

    # just for some debug outputs (currently 1 CSV w/ flies listed for each odor pair,
    # and recreation of Remy's old 2E plot). nothing hugely important.
    if plot_dir is not None:
        output_root = plot_dir
    else:
        output_root = Path('.')

    # TODO move relevant data to my own path in this repo (to pin version, independent
    # of what remy pushes to this repo) (-> use those files below)
    repo_root = Path.home() / 'src/OdorSpaceShare'
    assert repo_root.is_dir()

    preprint_data_folder = repo_root / 'preprint/data/figure-02/02e'

    # TODO roughly compare old vs new data? or just make plots w/ both (after settling
    # on error repr...)
    if use_preprint_data:
        warn('using pre-print data for 2E (set use_preprint_data=False to use newer '
            'data)!'
        )
        data_folder = preprint_data_folder
        csv_name = 'df_obs_plot_trialavg.csv'
    else:
        # TODO TODO TODO which flies are in this but not in old megamat data i'm now
        # loading for a lot of things? any?
        data_folder = repo_root / 'manuscript/data/figure-02/02e'
        # df_obs.csv in the same folder was one of her earlier attempts to get me a
        # newer version of this data, but was not completely consistent w/ format of old
        # CSV (and also had 'correlation' col that was actually correlation distance).
        # df_obs.csv should not be used.
        csv_name = 'df_obs_for_tom.csv'

    assert data_folder.is_dir()

    csv_path = data_folder.joinpath(csv_name)
    if verbose:
        print(f'loading Remy correlations for 2E from {csv_path}')

    df_obs = pd.read_csv(csv_path)

    assert not df_obs.isna().any().any()

    df_obs[['abbrev_row','abbrev_col']] = df_obs['odor_pair_str'].str.split(pat=', ',
        expand=True
    )
    assert (
        len(df_obs[['abbrev_row','abbrev_col']].drop_duplicates()) ==
        len(df_obs.odor_pair_str.drop_duplicates())
    )

    # (currently renaming my model output odors to match remy's, during creation of my
    # 2E plots, so no need for now)
    # TODO rename 'MethOct' -> 'moct', to be consistent w/ mine

    # TODO refactor to share def of these 2 odor cols w/ elsewhere?
    #
    # within each fly, expect each pair to only be reported once
    assert not df_obs.duplicated(subset=['datefly','abbrev_row','abbrev_col']).any()

    identity = df_obs.abbrev_col == df_obs.abbrev_row
    assert (df_obs[identity].correlation == 1).all()

    # we don't want to include these on plots, and the repeated 1.0 values also
    # interfere with some of the checks on my sorting.
    df_obs = df_obs[~identity].reset_index(drop=True)
    assert not (df_obs.correlation == 1).any()

    # plot ordering of odor pairs (ascending observed correlations)
    mean_pair_corrs = df_obs.groupby('odor_pair_str').correlation.mean()
    df_obs['mean_pair_corr'] = df_obs.odor_pair_str.map(mean_pair_corrs)

    # will start with low correlations, and end w/ the high ones (currently identity 1.0
    # corrs), as in preprint order.
    #
    # NOTE: if we ever really care to *exactly* recreate preprint 2E, we may need to use
    # Remy's order from:
    # np.load(data_folder.joinpath('odor_pair_ord_trialavg.npy'), allow_pickle=True)
    #
    # (previously committed code to use this order, but deleted now that I have my own
    # replacement for all new versions [which also either almost/exactly matches
    # preprint fig too])
    df_obs = df_obs.sort_values('mean_pair_corr', kind='stable').reset_index(drop=True)

    # TODO factor this into some check fn? haven't i done something similar elsewhere?
    #
    # checking all rows with a given pairs are adjacent after above sorting.
    # should be True since we are now dropping identity rows before sorting.
    # allows us to more easily derive order from output, for plotting against my own
    # model runs.
    last_seen_index = None
    for _, gdf in df_obs.groupby('odor_pair_str', sort=False):
        if last_seen_index is not None:
            assert last_seen_index + 1 == gdf.index.min()
        else:
            assert gdf.index.min() == 0

        last_seen_index = gdf.index.max()
        assert set(gdf.index) == set(range(gdf.index.min(), gdf.index.max() + 1))

    # .unique() has output in order first-seen, which (given check above), should be
    # same as sorting all pairs by mean correlation
    pair_order = df_obs.odor_pair_str.unique()
    assert np.array_equal(pair_order, mean_pair_corrs.sort_values().index)

    # TODO does it make sense that there is such a diversity of N counts for specific
    # pairs?  inspect pairs / flies + talk to remy.
    #
    # ipdb> df_obs.odor_pair_str.value_counts()
    # 1-6ol, 1-6ol    22
    # 2-but, 2-but    21
    # 1-6ol, 2-but    21
    # benz, benz      20
    # 1-6ol, benz     20
    #                 ..
    # aa, benz         4
    # ep, eug          3
    # PropAc, va       3
    # eug, ECin        3
    # 2h, MethOct      3
    # Name: odor_pair_str, Length: 205, dtype: int64
    # ipdb> set(df_obs.odor_pair_str.value_counts())
    # {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22}
    s1 = df_obs.groupby('odor_pair_str').size()
    assert s1.equals(df_obs.groupby('odor_pair_str').nunique().datefly)

    # TODO delete (replacing w/ just 'datefly')?
    assert (df_obs.datefly.map(lambda x: x[:2]) == '20').all()
    df_obs['datefly_abbrev'] = df_obs.datefly.map(lambda x: x[2:])
    #

    unique_datefly_per_pair = df_obs.groupby('odor_pair_str', sort=False
        ).datefly_abbrev.unique()

    assert s1.equals(unique_datefly_per_pair.str.len().sort_index())
    del s1

    # TODO delete?
    n_summary = pd.concat([
            df_obs.groupby('odor_pair_str', sort=False).size(),
            unique_datefly_per_pair
        ], axis='columns'
    )
    n_summary.columns = ['n', 'datefly']

    assert np.array_equal(n_summary.index, pair_order)

    n_summary.datefly = n_summary.datefly.map(lambda x: ', '.join(x))

    if use_preprint_data:
        suffix = '_OLD-PREPRINT-DATA'
    else:
        suffix = ''

    # TODO TODO inspect with remy
    to_csv(n_summary, output_root / f'remy_2e_n_per_pair{suffix}.csv')
    #

    df_obs['correlation_distance'] = 1 - df_obs.correlation
    assert df_obs['correlation_distance'].max() <= 2.0
    # excluding equality b/c we already dropped identity
    assert df_obs['correlation_distance'].min() > 0

    # TODO check whether order in plots is same w/ and w/o passing order explicitly to
    # sns calls below (now that we are sorting df_obs to put pairs in same order).
    # (unclear from docs how categorical order is inferred...)

    # TODO want to subset before returning? any actual need to?
    # affect plots (no, right?)?
    #
    # RY: "reorder columns"
    df_obs = df_obs.loc[:, [
        # not sure any of these 3 needed for (/affect) plots, but could be useful at
        # output for comparing to other correlations i load / compute from remy's data.
        'datefly',
        # TODO refactor to share def of these 2 cols?
        'abbrev_row',
        'abbrev_col',

        'odor_pair_str',

        # TODO delete
        #'correlation',

        'correlation_distance',
        # TODO how does this differ from odor_pair_str? del?
        # not referenced anywhere else in this file...
        #'odor_pair',
    ]]

    # only want to make this plot (to show we can recreate preprint figure), when data
    # we are loading is same as in preprint. currently i'm only ever using that data to
    # show we can recreate this plot.
    plot = use_preprint_data

    if plot:
        # TODO fix so i can pass new errorbar into plotting fns, so that i can force
        # that seed_errorbar value for reproducing this plot?
        if seed_errorbar != ('ci', 95):
            warn("set seed_errorbar=('ci', 95) if you want to reproduce preprint 2E. "
                "returning!"
            )
            return

        g = _create_2e_plot_with_obs_kc_corrs(df_obs, pair_order)

        # seems to already have abbrev_[row|col]
        df_mdl = pd.read_csv(preprint_data_folder.joinpath('df_mdl_plot_trialavg.csv'))

        # df_mdl also contains uniform_4 and hemidraw_4
        model_types_to_plot = ['uniform_7', 'hemidraw_7', 'hemimatrix']

        pal = sns.color_palette()
        palette = {
            'hemidraw_7': pal[0],
            'uniform_7': pal[1],
            'hemimatrix': pal[2],
        }

        # TODO refactor? (to also check observed KC corrs in _create_..., and to move
        # this into _2e_plot...?)
        identity = df_mdl.abbrev_col == df_mdl.abbrev_row
        assert (df_mdl[identity].correlation == 1).all()

        df_mdl = df_mdl[~identity].copy()
        assert not (df_mdl.correlation == 1).any()

        df_mdl['correlation_distance'] = 1 - df_mdl.correlation
        assert df_mdl['correlation_distance'].max() <= 2.0
        assert df_mdl['correlation_distance'].min() > 0
        #

        _2e_plot_model_corrs(g, df_mdl.query('model in @model_types_to_plot'),
            pair_order, hue='model', palette=palette
        )

        _finish_remy_2e_plot(g)

        # NOTE: no seed_errorbar part in filename here, as only saving this if it's same
        # as preprints ('ci', 95)
        savefig(g, output_root, '2e_preprint-repro_old_data')


    checks = True
    if checks and not use_preprint_data:
        # TODO delete
        print('ARE CHANGES TO 2E CHECKS CORRECT?')
        # TODO TODO TODO and how does data i'm loading now (old megamat + same new
        # megamat i had been loading) compare to the corrs from this csv remy gave me?
        # am i matching all the data now, or am i dropping one or the other to not be
        # comparing all the data?
        #

        # TODO refactor w/ place copied from in model_mb...?
        remy_pairs = set(list(zip(df_obs.abbrev_row, df_obs.abbrev_col)))

        # TODO does it actually matter? does df_obs have all the corrs i would compute
        # for old flies anyway? maybe just expand checks below to also check those
        # flies?
        #
        # TODO TODO TODO can i switch things to using corrs from
        # _load_remy_megamat_kc_responses? cause otherwise would prob need to have Remy
        # regen this file, including older megamat data betty now wants us to include...
        #
        # TODO TODO use -c check to verify i 2e outputs not changed by switching this
        # fn? add option to -c to pass substrs of outputs to check (ignoring rest)?
        #
        # TODO update comment. no longer just final 4.
        # data from best 4 "final" flies, which are the only megamat odor correlations
        # used anywhere in the paper except for figure 2E.
        #
        # TODO delete
        #mean_responses = _load_remy_megamat_kc_responses(drop_nonmegamat=False)
        #
        flymean_corrs = _remy_megamat_flymean_kc_corrs(ordered_pairs=remy_pairs,
            drop_nonmegamat=False
        )

        final_megamat_datefly = set(flymean_corrs.index.get_level_values('datefly'))
        # TODO delete (or update to include final 4 + however many old megamat flies i'm
        # now supposed to include)
        assert n_final_megamat_kc_flies <= len(final_megamat_datefly)

        flymean_corrs.columns = pd.MultiIndex.from_frame(
            flymean_corrs.columns.to_frame(index=False).applymap(olf.parse_odor_name)
        )
        assert not flymean_corrs.columns.duplicated().any()
        # TODO delete
        #mean_responses.columns = mean_responses.columns.map(olf.parse_odor_name)
        #assert not mean_responses.columns.duplicated().any()

        # TODO relax to include other pairs? or just drop? i assume we still won't have
        # all the data in df_obs if we just don't drop from latest set of (the old)
        # flies i'm loading?
        #assert set(mean_responses.columns) == megamat_odor_names

        # TODO delete? (replace w/ flymean_corrs)
        #corrs = mean_responses.groupby(level='datefly').apply(
        #    lambda x: corr_triangular(x.corr(), ordered_pairs=remy_pairs)
        #)
        #assert not corrs.isna().any().any()
        #

        # TODO move this dropna into above fn?
        flymean_corrs = flymean_corrs.dropna(how='all', axis='columns')

        # TODO also move this into fn above?
        flymean_corrs = flymean_corrs.loc[:,
            (flymean_corrs.columns.to_frame() != 'pfo').T.all()
        ]

        corrs = flymean_corrs

        n_megamat_only_pairs = n_choose_2(n_megamat_odors)
        # TODO delete? already relaxed from == to >=
        assert len(corrs.columns) >= n_megamat_only_pairs

        # TODO TODO fix (/delete?)
        s1 = set(
                corrs.columns.get_level_values('odor1') + ', ' +
                corrs.columns.get_level_values('odor2')
        )
        s2 = set(df_obs.odor_pair_str)

        # (this was before i was .dropna-ing flymean_corrs above)
        # TODO need to do anything about s1 - s2? e.g.
        # {'pfo, va', '6al, MethOct', 'MethOct, pa', '1p3ol, 6al', 'IaA, d-dlac',
        # '1-8ol, pfo', 'aa, d-dlac', '2-but, pfo', '1-8ol, ECin', 'B-cit, PropAc',
        # '1p3ol, pfo', 'benz, pfo', '1-8ol, g-6lac', '1p3ol, aa', 'pfo, t2h', '6al,
        # g-6lac', 'eb, pfo', '1-5ol, d-dlac', 'g-6lac, pfo', 'g-6lac, pa', 'B-cit,
        # ECin', 'ms, pfo', 'd-dlac, pfo', '6al, PropAc', 'ECin, aa', 'd-dlac, eb', 'aa,
        # pfo', 'B-cit, pfo', '2h, pfo', 'ECin, t2h', '2h, PropAc', '1p3ol, eb',
        # 'PropAc, t2h', '1-5ol, g-6lac', 'ECin, pa', 'B-cit, d-dlac', 'Lin, pfo',
        # 'd-dlac, va', 'ep, pfo', 'ECin, IaA', '1-5ol, 1p3ol', 'pa, pfo', '6al,
        # d-dlac', '1-8ol, 1p3ol', 'ECin, pfo', '1p3ol, B-cit', 'ECin, ep', 'MethOct,
        # pfo', 'aa, eug', '1-5ol, pfo', 'PropAc, ms', 'd-dlac, pa', 'g-6lac, t2h',
        # 'MethOct, aa', '1-5ol, MethOct', 'MethOct, t2h', 'aa, g-6lac', 'B-cit,
        # g-6lac', 'eug, pfo', 'eb, g-6lac', 'g-6lac, va', 'd-dlac, t2h', '1p3ol, pa',
        # 'Lin, PropAc', '6al, pfo', 'IaA, pfo', '1-5ol, ECin', 'Lin, d-dlac', 'PropAc,
        # eb', '1-8ol, d-dlac', '2h, ECin', 'PropAc, pfo', '1-6ol, pfo', '1p3ol, ms',
        # 'eug, pa', 'ECin, Lin', 'B-cit, MethOct', '1p3ol, t2h'}

        # TODO TODO fix. (is it just supposed to be s1 - s2? think so)
        try:
            #assert s2 - s1 == set()
            assert s1 - s2 == set()
        except:
            import ipdb; ipdb.set_trace()
        '''
        try:
            # all the (ordered) pairs we have in corrs are in df_obs.odor_pair_str
            assert set(
                    corrs.columns.get_level_values('odor1') + ', ' +
                    corrs.columns.get_level_values('odor2')
                ) - set(df_obs.odor_pair_str) == set()
        except:
            s1 = set(
                    corrs.columns.get_level_values('odor1') + ', ' +
                    corrs.columns.get_level_values('odor2')
            )
            s2 = set(df_obs.odor_pair_str)
            print()
            print(f'{s1=}')
            print(f'{s2=}')
            import ipdb; ipdb.set_trace()
        '''

        for datefly in corrs.index:
            fly_df = df_obs[df_obs.datefly == datefly]

            # TODO delete
            '''
            try:
                # none of the final 4 flies had other odor pairs measured
                # (not just using those 4 flies now)
                assert len(fly_df) >= n_megamat_only_pairs
            except:
                print()
                print(f'{len(fly_df)=}')
                print(f'{n_megamat_only_pairs=}')
                import ipdb; ipdb.set_trace()
            '''

            fly_ser = fly_df[['abbrev_row', 'abbrev_col', 'correlation_distance']
                ].set_index(['abbrev_row', 'abbrev_col'])

            # just to convert from shape (n, 1) to (n,)
            fly_ser = fly_ser.iloc[:, 0]

            fly_ser.index.names = ['odor1', 'odor2']

            # convert from correlation distance to correlation (to match what we have in
            # corrs)
            fly_ser = 1 - fly_ser
            fly_ser.name = 'correlation'

            # TODO TODO need some dropna or something now?
            # TODO just delete the assertion?
            fly_ser2 = corrs.loc[datefly]
            # (this didn't fail yet, i just think it might)
            '''
            try:
                assert set(fly_ser.index) == set(fly_ser2.index)
            except:
                import ipdb; ipdb.set_trace()
            '''

            # TODO delete?
            #assert np.allclose(fly_ser.loc[fly_ser2.index], fly_ser2)
            #
            assert np.allclose(fly_ser2.loc[fly_ser.index], fly_ser)

        df_megamat = df_obs[
            df_obs.abbrev_row.isin(megamat_odor_names) &
            df_obs.abbrev_col.isin(megamat_odor_names)
        ]

        # TODO TODO TODO are all of the old megamat flies that i'm now supposed to use a
        # subset of these? do the correlations match what i would compute?
        #
        # ipdb> len(set(df_megamat.datefly) - final_megamat_datefly)
        # 18
        # ipdb> pp (set(df_megamat.datefly) - final_megamat_datefly)
        # {'2018-10-21/1',
        #  '2019-03-06/3',
        #  '2019-03-06/4',
        #  '2019-03-07/2',
        #  '2019-04-26/4',
        #  '2019-05-09/4',
        #  '2019-05-09/5',
        #  '2019-05-23/2',
        #  '2019-05-24/1',
        #  '2019-05-24/3',
        #  '2019-05-24/4',
        #  '2019-07-19/2',
        #  '2019-09-12/1',
        #  '2019-09-12/2',
        #  '2022-09-21/1',
        #  '2022-09-22/2',o
        #  '2022-09-26/1',
        #  '2022-09-26/3'}
        # TODO TODO TODO just compare db_obs.datefly set to set i have from flymean corr
        # fn
        assert final_megamat_datefly - set(df_megamat.datefly) == set()
        df_megamat_nonfinal = df_megamat[
            ~df_megamat.datefly.isin(final_megamat_datefly)
        ]

        # TODO delete (/update) (no longer just using final 4 flies)
        #
        # only the 4 "final" flies have all 17 odors measured (-> all 136 non-identity
        # pairs)
        #
        # ipdb> [len(x) for _, x in df_megamat_nonfinal.groupby('datefly')]
        # [47, 10, 28, 30, 21, 38, 38, 36, 36, 36, 57, 79, 71, 71, 3, 3, 3, 3]
        #assert all(len(x) < n_megamat_only_pairs
        #    for _, x in df_megamat_nonfinal.groupby('datefly')
        #)

        assert remy_2e_metric == 'correlation_distance'
        mean_nonfinal_corrdist = df_megamat_nonfinal.groupby(['abbrev_row','abbrev_col']
            )[remy_2e_metric].mean()

        mean_nonfinal_corrdist.index.names = ['odor1', 'odor2']

        # TODO better check than this try/except
        try:
            square_nonfinal_corrdist = invert_corr_triangular(mean_nonfinal_corrdist,
                diag_value=0, _index=corrs.columns
            )

            square_nonfinal_corrs = 1 - square_nonfinal_corrdist

            # since sorting expects concentrations apparently...
            square_nonfinal_corrs.columns = square_nonfinal_corrs.columns + ' @ -3'
            square_nonfinal_corrs.index = square_nonfinal_corrs.index + ' @ -3'

            square_nonfinal_corrs = sort_odors(util.addlevel(
                    util.addlevel(square_nonfinal_corrs, 'panel', 'megamat').T,
                'panel', 'megamat'
                ), warn=False
            )

            square_nonfinal_corrs = square_nonfinal_corrs.droplevel('panel',
                axis='columns'
            ).droplevel('panel', axis='index')

            plot_corr(square_nonfinal_corrs, output_root,
                '2e_remy_nonfinal-flies-only_corr', xlabel='non-final flies only'
            )

            # TODO actually plot this / delete
            '''
            nonfinal_pair_n = df_megamat_nonfinal.groupby(['abbrev_row','abbrev_col']
                ).size()
            # TODO just rename these cols in dataframe before (so we don't have to do
            # this here and for mean)
            nonfinal_pair_n.index.names = ['odor1', 'odor2']
            nonfinal_pair_n = invert_corr_triangular(nonfinal_pair_n, diag_value=np.nan,
                _index=corrs.columns
            )
            '''
        # ...
        #   File "./al_analysis.py", line 1208, in invert_corr_triangular
        #     assert all(odor2[:-1] == odor1[1:])
        except AssertionError:
            warn('could not plot 2e square matrix corr plots')
    #

    return df_obs


# NOTE: contains sparsities in top level CSVs, as well as individual fly binarized
# responses in subdirectories
#
# downloaded from Dropbox folder:
# Remy/odor_space_collab/analysis_outputs/multistage/multiregion_data/\
#     response_breadth/by_trialavg_ref_stim/megamat
# (Remy sent me a link on Slack 2024-04-04)
remy_sparsity_dir = (remy_data_dir /
    'response_rates/refstim__ep_at_-3.0__median__0.120-0.130'
)

# contains CSVs remy made from the pickles she sent earlier under 'by_acq'
# (which i couldn't load)
remy_binary_response_dir = remy_sparsity_dir / 'by_acq_csvs'

# looks like the top-level sparsity CSVs are computed from peak amplitude alone
# (rather than options involving "std"), so only going to load these pickles
# (out of the 3 options in each fly_dir)
remy_fly_binary_response_fname = 'df_stim_responders_from_peak.csv'


remy_conc_str = ' @ -3.0'
# TODO refactor to share w/ some similar fns?
def _strip_remy_concs(x):
    assert x.str.endswith(remy_conc_str).all()
    return x.str.replace(remy_conc_str, '', regex=False)


# TODO add verbose kwarg and set False when calling for debug purposes from kc response
# loading code
def load_remy_fly_binary_responses(fly_sparsity_path: Path,
    acq_ledger: Optional[pd.DataFrame]=None, *, reset_index: bool = True,
    _seen_date_fly_combos=None) -> pd.DataFrame:
    # TODO check row/col is correct
    """Loads single fly NetCDF file to boolean (cell row X odor column) DataFrame.

    Args:
        acq_ledger: if passed in, has `index.names == remy_fly_cols`, and has
            `remy_fly_id in df.columns`

            maps ['date', 'fly_num'] to int 'acq' (`remy_fly_id`) (0-indexed).

        reset_index: if True, only return int 'cell' index. otherwise, retains all
            metadata columns in source CVS (i.e. row index names in the DataFrame)
    """
    fly_dirname = fly_sparsity_path.parent.name

    if verbose:
        print(fly_sparsity_path)

    # sep/index_col params from snippet Remy sent me
    #
    # there do only seem to be 5 index columns in these CSVs. all columns after are
    # odor names (e.g. '1-5ol @ -3.0')
    fly_df = pd.read_csv(fly_sparsity_path, sep='\t', index_col=list(range(5)))

    # 17 columns, 1 for each megamat odor.
    # index levels: cells_level_0 iscell iscell_xid0 xid0 embedding0
    #
    # iscells seems to all be 1, so no need to subset based on this/keep
    assert fly_df.index.get_level_values('iscell').astype(bool).all()

    # TODO delete (probably unimportant anyway, as no flies have only some elements
    # True, so this never has any info...)
    # TODO why is this true for some but not all?
    '''
    try:
        # not sure how this is different (from above), but at least it's also all True
        assert fly_df.index.get_level_values('iscell_xid0').all()
        print('iscell_xid0 assertion pass')
    except AssertionError:
        # NOTE: all False for this one fly (2022-11-10/1 [/megamat0_dsub3])
        # (all True for the other 3 final megamat flies)
        print('iscell_xid0 assertion fail!')
    '''

    # so i guess if cells were subset, they were renumbered (or more likely cells
    # were not subset at this point)
    cells_level_0 = fly_df.index.get_level_values('cells_level_0')
    assert pd.RangeIndex(cells_level_0.max() + 1).equals(cells_level_0)

    # don't think there's anything worth checking about 'embedding0'

    # TODO TODO do something w/ 'xid0' index level? as part of comparison wrt
    # response data (the stuff loaded in KC corr loading fn)?
    # TODO see if which xid0 cells have are consistent w/ those in responses i'm
    # loading elsewhere?
    #import ipdb; ipdb.set_trace()

    assert set(np.unique(fly_df.values)) == {0, 1}
    fly_df = fly_df.astype(bool)

    # TODO can i recreate these from the data Remy sent to anoop (what should
    # still be the final 4 megamat KC flies), which is in our Dropbox at:
    # Remy/odor_space_collab/for_mendy/data/megamat17
    # (using thresholds from CSV above)?

    # NOTE: there must be double counted cells. don't think Remy makes any attempt
    # to de-duplicate them.
    n_cells = len(fly_df)

    silent_cells = ~ (fly_df.any(axis='columns'))
    assert len(silent_cells) == n_cells
    n_silent = silent_cells.sum()

    parts = fly_dirname.split('__')
    assert len(parts) >= 3
    date_str, fly_str = parts[:2]

    date = pd.Timestamp(date_str)

    # e.g. 'fly01' -> 1
    fly_num = int(fly_str.replace('fly', ''))

    if _seen_date_fly_combos is not None:
        # we already know the ledger doesn't have duplicate flies (or 'acq' values),
        # from checks in main caller of this code.
        #
        # now we are just checking the individual files we are loading also don't ever
        # refer to the same fly in >1 of the files (could accidentally be 2 recordings
        # of same fly).
        assert (date, fly_num) not in _seen_date_fly_combos
        _seen_date_fly_combos.add( (date, fly_num) )

    if reset_index:
        fly_df = fly_df.reset_index(drop=True)
        # TODO what was it before? (maybe still 'cell', the reset_index() call seems to
        # clear this)
        # TODO still do this in else case?
        fly_df.index.name = 'cell'

    if acq_ledger is not None:
        curr_acq = acq_ledger.loc[(date, fly_num), remy_fly_id]
        fly_df = util.addlevel(fly_df, remy_fly_id, curr_acq)

        msg = f'{fly_dirname} ({remy_fly_id}={curr_acq})'
    else:
        msg = f'{fly_dirname}'

    if verbose:
        print(msg)

        # silent = responds to no odors
        #
        # NOTE: not sure if silent/not cells are equally represented (i.e. equally
        # double counted), so maybe fraction of silent cells is off?
        print(f'{n_silent} silent / {n_cells} cells ({(n_silent / n_cells):.3f})')
        print()

    return fly_df


_remy_megamat_kc_binary_responses = None
def load_remy_megamat_kc_binary_responses() -> pd.DataFrame:
    # TODO doc

    global _remy_megamat_kc_binary_responses
    if _remy_megamat_kc_binary_responses is not None:
        return _remy_megamat_kc_binary_responses

    # should be able to map acq to fly CSV dir using first 4 cols:
    # ['acq' (=remy_fly_id), 'date_imaged' (=remy_date_col), 'fly_num',
    # 'thorimage_name'] (last 2 cols not important)
    acq_ledger = pd.read_csv(remy_sparsity_dir / 'df_acqs.csv')
    acq_ledger[remy_date_col] = pd.to_datetime(acq_ledger[remy_date_col])

    acq_ledger = acq_ledger.set_index(remy_fly_cols, verify_integrity=True)
    assert not acq_ledger[remy_fly_id].duplicated().any()

    if verbose:
        print()
        print('loading Remy megamat KC binarized responses (for sparsity + S1C):')

    _seen_date_fly_combos = set()
    fly_binary_responses_list = []

    for fly_sparsity_path in sorted(
            remy_binary_response_dir.glob(f'*/{remy_fly_binary_response_fname}')
        ):

        fly_df = load_remy_fly_binary_responses(fly_sparsity_path, acq_ledger,
            _seen_date_fly_combos=_seen_date_fly_combos
        )
        fly_binary_responses_list.append(fly_df)

    binary_responses = pd.concat(fly_binary_responses_list, verify_integrity=True)
    binary_responses.columns.name = remy_odor_col

    binary_responses.columns = _strip_remy_concs(binary_responses.columns)
    assert binary_responses.shape[1] == n_megamat_odors

    binary_responses = binary_responses.sort_index()

    assert _remy_megamat_kc_binary_responses is None
    _remy_megamat_kc_binary_responses = binary_responses

    return binary_responses


def remy_megamat_sparsity() -> float:
    """Returns mean response rate in Remy's final megamat KC data.

    Weights each fly (of 4) equally, regardless of number of cells per fly.
    """
    # md5 of this and
    # response_rates/old_megamat_sparsities/refstim__ep_at_-3.0__median__0.120-0.130
    # match, so they are the same.
    megamat_csv = remy_sparsity_dir / 'tidy_sparsities_ascending.csv'

    # columns: ['acq', 'stim', 'peak_amp_thresh', 'max_sparsity', 'min_sparsity',
    # 'kc_soma']
    #
    # 'kc_soma' should be between or equal to '[max|min]_sparsity' limits (which should
    # correspond to range of thresholds referenced in path name). 'kc_soma' is what Remy
    # said I should use.
    df = pd.read_csv(megamat_csv)
    sparsity_col = 'kc_soma'

    # TODO what are units/meaning of "0.120-0.130" in path? how does it relate to
    # 'peak_amp_thresh' col (min=0.95, max=1.75)?

    # TODO delete? only important column in this should match the corresponding (diff
    # named) column in other csv
    megamat_csv2 = remy_sparsity_dir / 'df_sparsity_recomputed.csv'
    df2 = pd.read_csv(megamat_csv2)
    sparsity_col2 = 'sparsity_from_peak_thr'

    assert df2[remy_odor_col].equals(df[remy_odor_col])
    assert df2[sparsity_col2].equals(df[sparsity_col])
    # df2[remy_fly_id] won't match. df2[remy_fly_id].unique() == [0] (through a
    # mistake on her end, Remy created this CSV w/ ID 0 for all the data in this CSV,
    # even though it actually has data from 4 flies. other CSV includes the correct IDs,
    # and rest of data matches)
    del df2

    assert not df.isna().any().any()
    assert df[remy_fly_id].nunique() == n_final_megamat_kc_flies
    assert set(x.stim.nunique() for _, x in df.groupby(remy_fly_id)) == {
        n_megamat_odors
    }

    df[remy_odor_col] = _strip_remy_concs(df[remy_odor_col])

    # low->high response rate
    odors_in_sparsity_order = df.groupby(remy_odor_col, sort=False)[sparsity_col].mean(
        ).sort_values().index

    df = df.sort_values(remy_odor_col, kind='stable',
        key=lambda x: x.map(odors_in_sparsity_order.get_loc)
    )
    # TODO just force ep to be last? sort order actually diff (-> imply data is diff?)?
    # maybe just slightly though... (?)

    plot = False
    if plot:
        fig, ax = plt.subplots()
        sns.pointplot(ax=ax, data=df, x=remy_odor_col, y=sparsity_col, hue=remy_fly_id)
        ax.set_xlabel('odor')
        ax.set_ylabel('response rate')
        ax.set_ylim([0, 0.2])
        rotate_xticklabels(ax, 90)
        # https://stackoverflow.com/questions/44620013
        ax.get_legend().set_title('fly')

        # should match preprint S1B
        fig, ax = plt.subplots()
        # should have the same 95% CI as reported in paper
        sns.pointplot(ax=ax, data=df, x=remy_odor_col, y=sparsity_col, color='black')
        ax.set_xlabel('odor')
        ax.set_ylabel('response rate')
        ax.set_ylim([0, 0.2])
        rotate_xticklabels(ax, 90)

        # TODO put mean_sparsity in xlabel, like in sparsity_per_odor plots
        # (silent cells too?) (delete comment? still care?)

        # TODO save plots? return figs?

    # TODO TODO recreate model KC response rates plot, on model tuned on hallem, but
    # subset to 17 megamat odors (using this code too, in case there was an issue w/
    # other code?) (done? still want?)

    mean_of_odor_means = df.set_index([remy_odor_col, remy_fly_id]).groupby(
        level=remy_odor_col, sort=False).mean()[sparsity_col].mean()

    mean_sparsity = df[sparsity_col].mean()
    # as expected from the math. numerically slightly (but not consequentially) diff
    assert np.isclose(mean_sparsity, mean_of_odor_means)

    binary_responses = load_remy_megamat_kc_binary_responses()
    assert set(binary_responses.columns) == set(df[remy_odor_col])

    recomputed_df = binary_responses.groupby(remy_fly_id).mean().melt(
        value_name=sparsity_col, ignore_index=False).reset_index()

    recomputed_df = recomputed_df.set_index([remy_fly_id, remy_odor_col],
        verify_integrity=True).sort_index()

    df = df.set_index([remy_fly_id, remy_odor_col], verify_integrity=True).sort_index()
    # changing index affects outcome, but only in irrelevant numerical way
    assert np.isclose(df[sparsity_col].mean(), mean_sparsity)

    for acq, recomputed_fly_df in recomputed_df.groupby(remy_fly_id):
        recomputed_ser = recomputed_fly_df.droplevel(remy_fly_id).iloc[:, 0]
        remy_fly_ser = df.loc[acq, sparsity_col]
        assert np.allclose(recomputed_ser, remy_fly_ser)

    # NOTE: can't just do binarized_responses.mean().mean(), as there are different
    # numbers of cells for different flies, so we need to average within each fly first
    # (to not weight some flies more than others)
    assert np.isclose(recomputed_df[sparsity_col].mean(), mean_sparsity)

    return mean_sparsity


# TODO factor out? replace w/ [light wrapper around] sklearn's minmax_scale fn?
def minmax_scale(data: pd.Series) -> pd.Series:
    scaled = data.copy()
    scaled -= scaled.min()
    scaled /= scaled.max()

    assert np.isclose(scaled.min(), 0)
    assert np.isclose(scaled.max(), 1)

    # TODO delete
    s2 = pd.Series(index=data.index, name=data.name, data=sk_minmax_scale(data))
    # not .equals, but this assertion is true
    assert np.allclose(s2, scaled)
    #
    return scaled


def maxabs_scale(data: pd.Series) -> pd.Series:
    # sklearn.preprocessing.maxabs_scale does not preserve Series input
    # (returns a numpy array)
    return pd.Series(index=data.index, name=data.name, data=sk_maxabs_scale(data))


def model_mb_responses(certain_df, parent_plot_dir, roi_depths=None,
    skip_sensitivity_analysis=False, skip_models_with_seeds=False):
    # TODO delete. for debugging.
    global _spear_inputs2dfs
    #

    # TODO make and use a subdir in plot_dir (for everything in here, including
    # fit_and_plot... calls)

    plot_dir = parent_plot_dir / 'mb_modeling'

    # TODO w/ a verbose flag to say which odors / glomeruli overlapped

    # I think deltas make more sense to fit than absolute rates, as both can go negative
    # and then we could better filter out points from non-responsive (odor, glomerulus)
    # combinations, if we wanted to.
    hallem_delta = orns.orns(columns='glomerulus', add_sfr=False)

    hallem_delta = abbrev_hallem_odor_index(hallem_delta)

    #our_odors = {olf.parse_odor_name(x) for x in certain_df.index.unique('odor1')}

    # TODO delete?
    # TODO TODO TODO or print intersection of these w/ stuff here (or at least stuff
    # that also matches some nearness criteria on the concentration? where is that
    # currently handled, if it is at all?)
    #hallem_odors = set(hallem_delta.index)

    # as of odors in experiments in the months before 2023-06-30, checked all these
    # are actually not in hallem.
    #
    # this could be a mix of stuff actually not in Hallem and stuff we dont have an
    # abbreviation mapping from full Hallem name. want to rule out the latter.
    # TODO still always print these?
    # TODO delete?
    #unmatched_odors = our_odors - hallem_odors

    # TODO delete
    #validation_odors = {olf.parse_odor_name(x) for x in
    #    certain_df.loc['validation2'].index.unique('odor1')
    #}
    #print(f'{len(validation_odors)=}')
    #print(f'{validation_odors & hallem_odors=}')
    ## (len 7 as expected)
    #print(f'{len(validation_odors & hallem_odors)=}')
    #

    # TODO TODO TODO match glomeruli up to hallem names
    # (may need to make some small decisions)
    # (basically, are there any that are currently unmatched that can be salveaged?)

    # TODO TODO TODO also check which of our_odors are in hallem lower conc data
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

    # TODO delete. (after actually checking...)
    # TODO check no naming issues
    # {'DM3+DM5', 'DA4m' (2a), 'VA1d' (88a), 'DA4l' (43a), 'DA3' (23a), 'VA1v' (47b),
    # 'DL3' (65a, 65b, 65c), 'DL4' (49a, 85f)}
    print(f'{(hallem_glomeruli - our_glomeruli)=}')
    #

    # TODO delete. (after actually checking...)
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
    #

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

    # TODO TODO assert that 'is_pair' is all False? maybe drop it in a separate step
    # before this? (want to also drop is_pair for roi_depths in a consistent way.
    assert (certain_df.index.get_level_values('is_pair') == False).all()
    certain_df = certain_df.droplevel('is_pair', axis='index')
    if roi_depths is not None:
        assert (roi_depths.index.get_level_values('is_pair') == False).all()
        roi_depths = roi_depths.droplevel('is_pair', axis='index')

    drop_pfo = True
    # shouldn't be in megamat/validation/diagnostic data. added for new kiwi/control
    # data.
    if drop_pfo:
        odor_names = certain_df.index.get_level_values('odor1').map(
            olf.parse_odor_name
        )
        odor_concs = certain_df.index.get_level_values('odor1').map(
            olf.parse_log10_conc
        )

        pfo_mask = odor_names == 'pfo'
        if pfo_mask.any():
            pfo_conc_set = set(odor_concs[pfo_mask])

            # TODO if this gets triggered by None/NaN, adapt to also include None/NaN
            # (if there are any negative float concs, that would indicate a bug)
            #
            # NOTE: {0.0} == {0} is True
            assert pfo_conc_set == {0}

            # TODO warn that we are dropping (if any actually dropped)
            certain_df = certain_df.loc[~pfo_mask].copy()


    index_df = certain_df.index.to_frame(index=False)

    odor1_names = index_df.odor1.apply(olf.parse_odor_name)
    mix_strs = (
        odor1_names + '+' +
        index_df.odor2.apply(olf.parse_odor_name) + ' (air mix) @ 0'
    )
    # TODO work as-is (seems to...)? need to subset RHS to be same shape?
    # (could add assertions that other part, i.e. `index_df.odor2 == solvent_str`
    # doesn't change)
    index_df.loc[index_df.odor2 != solvent_str, 'odor1'] = mix_strs

    # NOTE: see panel2name_order modifications in natmix.olf.panel2name_order, which
    # define order of these hacky new "odors" ('kmix0','kmix-1',...'cmix0','cmix-1',...)
    #
    # e.g. 'kmix @ 0' -> 'kmix0' (so concentration not recognized as such, and thus it
    # should work in modelling code that currently strips that)
    hack_strs_to_fix_mix_dilutions = (
        odor1_names + index_df.odor1.apply(olf.parse_log10_conc).map(
            lambda x: f'{x:.0f}'
        )
    )
    # expecting all these to get stripped off in modelling code
    hack_strs_to_fix_mix_dilutions = hack_strs_to_fix_mix_dilutions + ' @ 0'

    index_df.loc[odor1_names.str.endswith('mix'), 'odor1'] = \
        hack_strs_to_fix_mix_dilutions

    certain_df.index = pd.MultiIndex.from_frame(index_df)
    del index_df

    # after above two hacks, for kiwi/control data, sort_odors should order odors as:
    # pfo, components, 2-component mix, 2-component air mix, 5-component mix, dilutions
    # of 5-component mix (with more dilute mixtures further towards the end)
    certain_df = sort_odors(certain_df)

    # TODO TODO may want to preserve panel just so i can fit dF/F -> spike delta fn
    # on all, then subset to specific panels for certain plots
    #
    # TODO maybe ['panel', 'odor1']? or just drop diagnostic panel 'ms @ -3'?
    # TODO sort=False? (since i didn't have that pre-panel support, may need to sort to
    # compare to that output, regardless...)
    fly_mean_df = certain_df.groupby(['panel', 'odor1'], sort=False).mean()
    # TODO delete? restore and change code to expect 'odor' instead of 'odor1'?
    # this is just to rename 'odor1' -> 'odor'
    fly_mean_df.index.names = ['panel', 'odor']

    n_before = num_notnull(fly_mean_df)
    shape_before = fly_mean_df.shape

    # TODO actually helpful to drop ['date', 'fly_num'] cols? keeping could make
    # summarizing model input easier later... (storing alongside in fly_ids for now)
    fly_mean_df = util.add_group_id(fly_mean_df.T.reset_index(), ['date', 'fly_num'],
        name='fly_id'
    )

    fly_ids = fly_mean_df[['fly_id','date','fly_num']].drop_duplicates()
    # column level names kinda non-sensical at this intermediate ['panel', 'odor'], but
    # I think it's all fine again by end of reshaping (shape, #-not-null, and set of
    # dtypes don't change)
    fly_ids = fly_ids.droplevel('odor', axis='columns')
    # nulling out nonsensical 'panel' name
    fly_ids.columns.name = None

    fly_ids = fly_ids.set_index('fly_id')

    fly_mean_df = fly_mean_df.set_index(['fly_id', 'roi']).drop(
        columns=['date', 'fly_num'], level=0).T

    # TODO replace w/ call just renaming 'roi'->'glomerulus'
    assert 'fly_id' == fly_mean_df.columns.names[0]
    # TODO also assert len of names and/or names[1] is 'roi'?
    fly_mean_df.columns.names = ['fly_id', 'glomerulus']

    assert num_notnull(fly_mean_df) == n_before
    assert fly_mean_df.shape == shape_before
    assert set(fly_mean_df.dtypes) == {np.dtype('float64')}

    # TODO delete? here and elsewhere? (was before fly_mean_df code)
    mean_df = fly_mean_df.groupby('glomerulus', axis='columns').mean()

    # TODO factor out?
    def melt_odor_by_glom_responses(df, value_name):
        n_before = num_notnull(df)
        df = df.melt(value_name=value_name, ignore_index=False)
        assert num_notnull(df[value_name]) == n_before
        return df

    # TODO factor into fn alongside current abbrev handling
    #
    # TODO actually check this? reason to think this? why did remy originally choose to
    # do -3 for everything? PID?
    # (don't think it was b/c they had reason to think that was the best intensity-match
    # of the Hallem olfactometer... think it might have just been fear of
    # contamination...)?
    #
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

    # TODO TODO allow slop of +/- 1 order of magnitude in general for merging w/
    # hallem (for validation stuff in particular)?

    dff_col = 'delta_f_over_f'
    spike_delta_col = 'delta_spike_rate'
    est_spike_delta_col = f'est_{spike_delta_col}'

    # TODO TODO delete all mean_df stuff if i get fly_mean_df version working
    # (which i think i have?)?
    # (or just scale w/in each fly before reducing fly_mean_df -> mean_df)
    # (or make choice to take mean right before plotting (to switch easier?)?
    # plus then it would work post-scaling, which is what i would want)
    mean_df = melt_odor_by_glom_responses(mean_df, dff_col)
    #

    n_notnull_before = num_notnull(fly_mean_df)
    n_null_before = num_null(fly_mean_df)

    if roi_depths is not None:
        assert fly_mean_df.columns.get_level_values('glomerulus').equals(
            roi_depths.columns.get_level_values('roi')
        )
        # to also replace the ['date','fly_num'] levels w/ 'fly_id, as was done to
        # fly_mean_df above
        roi_depths.columns = fly_mean_df.columns.copy()

    fly_mean_df = melt_odor_by_glom_responses(fly_mean_df, dff_col)

    roi_depth_col = 'roi_depth_um'

    if roi_depths is not None:
        # should be ['panel', 'odor']
        index_levels_before = fly_mean_df.index.names
        shape_before = fly_mean_df.shape

        roi_depths = melt_odor_by_glom_responses(roi_depths, roi_depth_col
            ).reset_index()

        fly_mean_df = fly_mean_df.reset_index().merge(roi_depths,
            on=['panel', 'fly_id', 'glomerulus']
        )

        fly_mean_df = fly_mean_df.set_index(index_levels_before)

        assert fly_mean_df.shape[0] == shape_before[0]
        assert fly_mean_df.shape[-1] == (shape_before[-1] + 1)

    assert num_notnull(fly_mean_df[dff_col]) == n_notnull_before
    assert num_null(fly_mean_df[dff_col]) == n_null_before

    fly_mean_df = fly_mean_df.dropna(subset=[dff_col])
    if roi_depths is not None:
        # TODO and should i also check we aren't dropping stuff that's non-NaN in depth
        # col (by dropna on dff_col) (no, some odors are nulled before)?
        #
        # this should be defined whenever dff_col is
        assert not fly_mean_df[roi_depth_col].isna().any()

    assert num_notnull(fly_mean_df[dff_col]) == n_notnull_before
    assert num_null(fly_mean_df) == 0

    # TODO delete if ends up being easier (in terms of less postprocessing) to subset
    # out + reshape stuff from merged tidy df
    hallem_delta_wide = hallem_delta.copy()
    #
    hallem_delta = melt_odor_by_glom_responses(hallem_delta, spike_delta_col)


    def scaling_method_to_col(method: Optional[str]) -> str:
        if method is None:
            return dff_col
        else:
            return f'{method}_scaled_{dff_col}'


    # quantile 0 = min, 1 = max. after unstacking, columns will be quantiles.
    fly_quantiles = fly_mean_df.groupby('fly_id')[dff_col].quantile(
        [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]).unstack()

    avg_flymin = fly_quantiles[0].mean()
    avg_flymax = fly_quantiles[1].mean()

    # TODO TODO compare to quantiles after applying new transform i come up with
    #
    # NOTE: seems to be more variation in upper end than in inhibitory values
    # TODO maybe i should be scaling the two sides diff then?
    #
    #             0.00      0.01      0.05      0.50      0.95      0.99      1.00
    # fly_id
    # 1      -0.393723 -0.185341 -0.066075  0.088996  0.771770  1.408845  2.061222
    # 2      -0.265394 -0.113677 -0.023615  0.075681  0.546017  0.915291  1.209619
    # 3      -0.290391 -0.151183 -0.032448  0.082085  0.541723  0.831624  1.747735
    # 4      -0.284210 -0.142273 -0.026323  0.103612  0.690146  1.038761  2.208872
    # 5      -0.450574 -0.142619 -0.033605  0.135551  1.006864  1.508309  2.735914
    # 6      -0.469455 -0.135344 -0.033547  0.091594  0.722606  1.246020  1.969335
    # 7      -0.301425 -0.172252 -0.039677  0.121347  0.739866  1.235905  3.101158
    # 8      -0.377237 -0.180162 -0.048570  0.114664  0.958002  1.807228  2.692789
    # 9      -0.316237 -0.066018 -0.021773  0.058443  0.525360  0.952334  1.579931
    # 10     -0.449154 -0.238127 -0.077457  0.091946  0.841725  1.547193  2.411627
    # 11     -0.444483 -0.228012 -0.073913  0.121594  0.808650  1.315351  2.476759
    # 12     -0.524139 -0.212304 -0.065836  0.079312  0.683135  1.259982  2.623360
    # 13     -0.373534 -0.193339 -0.057145  0.141604  1.019016  1.895920  3.960306
    # 14     -0.351293 -0.218182 -0.056444  0.093895  0.878827  1.745915  2.475087

    # TODO share w/ plots from model fitting below?
    dff_desc = f'mean glomerulus {dff_latex}'
    # TODO refactor to preprend dff_desc inside loop (rather than manually for each
    # of these)
    scaling_method2desc = {
        None: dff_desc,

        #'minmax': f'{dff_desc}\n[0,1] scaled within fly',

        'zscore': f'{dff_desc}\nZ-scored within fly',

        'maxabs': dff_desc + '\n$fly_{max} \\rightarrow 1$, 0-preserved',

        'to-avg-max':
            dff_desc + '\n$fly_{max} \\rightarrow \overline{fly_{max}}$, 0-preserved',

        'split-minmax':
            # TODO latex working yet?
            f'{dff_desc}\n$+ \\rightarrow [0, 1]$\n$- \\rightarrow [-1, 0]$',

        'split-minmax-to-avg': (dff_desc +
            # TODO latex working yet?
            '\n$+ \\rightarrow [0, \overline{fly_{max}}]$\n'
            '$- \\rightarrow [\overline{fly_{min}}, 0]$'
        ),
    }

    # TODO factor out?
    # TODO rename to "add_scaled_dff_col" or something?
    def scale_one_fly(gdf, method='zscore'):
        """Adds <method>_scaled_<dff_col> column with scaled <dff_col> values.

        Does not change any existing columns of input.
        """
        assert not gdf.fly_id.isna().any() and gdf.fly_id.nunique() == 1
        col_to_scale = dff_col
        to_scale = gdf[col_to_scale]
        n_nan_before = to_scale.isna().sum()

        new_dff_col = scaling_method_to_col(method)
        assert new_dff_col not in gdf.columns

        if method == 'minmax':
            scaled = minmax_scale(to_scale)

        elif method == 'zscore':
            scaled = (to_scale - to_scale.mean()) / to_scale.std()

        # TODO maybe try a variant of 'zscore' where we dont subtract mean first? (b/c
        # want to preserve 0) (std() doesn't seem that related to fly maxes... not
        # encouraging for this strategy)

        # also preserves 0, like split-minmax* methods below, but just one scalar
        # applied to all data
        # (new min will be > -1 (and < 0), assuming abs(min) < abs(max) (and neg min))
        elif method in ('maxabs', 'to-avg-max'):
            scaled = maxabs_scale(to_scale)

            if method == 'to-avg-max':
                # in theory, max(abs) could come from negative values, but the data
                # should have larger positive dF/F, so that shouldn't happen
                assert np.isclose(scaled.max(), 1)
                scaled *= avg_flymax
                assert np.isclose(scaled.max(), avg_flymax)

        elif method in ('split-minmax', 'split-minmax-to-avg'):
            # TODO warn if no negative values in input (tho there should always be as
            # i'm currently using it) (prob fine to keep as assertion for now)
            assert (to_scale < 0).any()

            # NOTE: to_scale.index.duplicated().any() == True, so probably can't use
            # index as-is to split/re-combine data
            # TODO delete if not needed
            index = to_scale.index
            to_scale = to_scale.reset_index(drop=True)
            #

            neg = to_scale < 0
            nonneg = to_scale >= 0
            n_neg = neg.sum()
            assert len(to_scale) == (n_neg + nonneg.sum() + to_scale.isna().sum())

            scaled = to_scale.copy()
            scaled[nonneg] = minmax_scale(scaled[nonneg])

            # after minmax_scale, just scaled * (max - min) + min, to go to new range
            scaled[neg] = minmax_scale(scaled[neg]) - 1
            assert np.isclose(scaled.min(), -1)
            assert np.isclose(scaled[neg].max(), 0)

            if method == 'split-minmax-to-avg':
                scaled[nonneg] *= avg_flymax
                scaled[neg] *= abs(avg_flymin)

            # not true b/c some max of to_scale[neg] gets mapped to 0, presumably
            #assert n_neg == (scaled < 0).sum()
            # rhs here can also include 0 from min of to_scale[nonneg]
            assert n_neg <= (scaled <= 0).sum()
            assert (scaled < 0).any()

            # so it's not really re-ordering anything. that's good.
            assert np.array_equal(
                np.argsort(scaled[scaled != 0]),
                np.argsort(gdf.reset_index()[scaled != 0][col_to_scale])
            )

            # TODO delete if i remove related code changing index above
            scaled.index = index

        # TODO try pinning particular odor(s)? how?
        # TODO maybe use diags for the pinning, to share w/ validation panel flies more
        # easily?

        else:
            raise NotImplementedError(f'scaling {method=} not supported')

        assert scaled.isna().sum() == n_nan_before, 'scaling changed number of NaN'
        gdf[new_dff_col] = scaled
        return gdf


    columns_before = fly_mean_df.columns

    methods = [
        # TODO delete
        #'minmax',

        'zscore',
        'maxabs',
        'to-avg-max',
        'split-minmax',
        'split-minmax-to-avg',
    ]
    for method in methods:
        # each of these calls adds a new column, with a scaled version of dff_col.
        fly_mean_df = fly_mean_df.groupby('fly_id', sort=False).apply(
            lambda x: scale_one_fly(x, method=method)
        )

    # TODO recompute and compare quartiles?
    # TODO replace w/ refactoring loop over scaling_method2desc.items() to loop over
    # scaling methods from added columns? (would need to track scaling methods for the
    # added cols, probably in scale_one_fly?)
    # (or just `continue` if column not in df...)
    # (OR now could prob use `methods` list above)
    scaled_cols = [c for c in fly_mean_df.columns if c not in columns_before]
    # to ensure we are making plots for each scaled column added
    assert (
        {scaling_method_to_col(x) for x in scaling_method2desc.keys()} ==
        {dff_col} | set(scaled_cols)
    )
    #

    # TODO delete
    # TODO better name for df... (or factor to fn so it doesn't matter)
    # (renamed fdf->merged_dff_and_hallem. need to also rename this, or probably just
    # delete it)
    #
    # doesn't seem to matter that odor is index and glomerulus is column. equiv to:
    # pd.merge(mean_df.reset_index(), hallem_delta.reset_index(),
    #     on=['odor', 'glomerulus']
    # )
    #df = mean_df.merge(hallem_delta, on=['odor', 'glomerulus']).reset_index()

    # TODO TODO decide how to handle panel when merging w/ hallem
    # (mean first for fitting dF/F -> spike delta fn, but then separately merge w/in
    # each panel for running model?)
    # (what is currently happening?)A

    # TODO TODO delete? or move to before odor2 level effectively dropped?
    # TODO gate behind there being and odor2 level?
    #
    # TODO to make this merging easier, might actually want to format mixtures down to
    # one str column, so that if [hypothetically, not in current data] odor1=solvent and
    # odor2 is in hallem, we can still match it up
    # TODO just rename hallem 'odor' -> 'odor1', and reset_index() on both
    # TODO add solvent odor2 to hallem and merge on=(odor_cols + ['glomerulus'])?
    #
    # if we only have odor2 odors for mixtures where odor1 is also an odor, we would
    # never want to merge those with hallem anyway, so we can just drop those rows
    # before merging
    '''
    odor1 = fly_mean_df.index.get_level_values('odor1')
    odor2 = fly_mean_df.index.get_level_values('odor2')
    # TODO TODO warn / err if any of this odor2 stuff is actually != solvent_str?
    # (since not currently supporting that, nor thinking that's the way i'll try to do
    # it...)
    assert not (odor1[odor2 != solvent_str] == solvent_str).any()
    # .reset_index() b/c left_on didn't seem to work w/ a mix of cols and index levels
    for_merging = fly_mean_df[odor2 == solvent_str].reset_index()

    merged_dff_and_hallem = for_merging.merge(hallem_delta,
    '''
    merged_dff_and_hallem = fly_mean_df.reset_index().merge(hallem_delta,
        left_on=['odor', 'glomerulus'], right_on=['odor', 'glomerulus']
    ).reset_index()

    assert not merged_dff_and_hallem[spike_delta_col].isna().any()

    # TODO delete
    # (note this was before 'odor'->odor_cols change)
    # TODO what's going on here? don't i have geraniol dF/F data?
    # (2024-05-09: can't repro, at least not passing all data as input. maybe passing
    # just validation2? not seeing geraniol at all now tho... that an issue?
    # can't repro w/ that input either. maybe if i don't use consensus df for input?
    # that'd probably still be dropped above tho...)
    #
    # must have been one that changed conc? after also excluding flies 2023-10-15/1,2
    # and 2023-10-19/2 (and 2024-01-05/4, not that I think this one mattered here), now
    # I'm getting the empty set for this (and geraniol not here anymore, as it's been -2
    # since 2023-11-19)
    #ipdb> set(merged_dff_and_hallem.odor.unique()) -
    # set(merged_dff_and_hallem.dropna().odor.unique())
    #{'ger @ -3'}

    # TODO print odors left after merging. something like
    # sorted(merged_dff_and_hallem.odor.unique())
    # TODO print # of (fly X glomeruli) combos (at least those that overlap w/
    # hallem) too, for each odor

    # TODO filter out low intensity stuff? (more points there + maybe more noise in
    # dF/F)
    # TODO fit from full matrix input rather than just each glomerulus as attempt at
    # ephaptic stuff?

    # TODO also print / save fly_id -> (date, fly_num) legend
    assert not merged_dff_and_hallem.fly_id.isna().any(), 'nunique does not count NaN'
    fly_palette = dict(zip(
        sorted(merged_dff_and_hallem.fly_id.unique()),
        sns.color_palette(cc.glasbey, merged_dff_and_hallem.fly_id.nunique())
    ))

    # still too hard to see density when many overlap, but 0.2 also has that issue, and
    # too hard to make out single fly colors at that point (when points arent
    # overlapping)
    scatterplot_alpha = 0.3
    # existing values in fly_palette are 3-tuples (color w/o alpha)
    fly_palette = {f: c + (scatterplot_alpha,) for f, c in fly_palette.items()}

    _cprint_color = 'blue'
    # hack to tell whether we should fit model (if input is megamat panel [which
    # overlaps well enough w/ hallem], and has at least 7 flies there, we should).
    # otherwise, we should try to load a saved model, and use that.
    try:
        # TODO delete
        '''
        if len(certain_df.loc['megamat'].dropna(axis='columns', how='all'
            ).columns.to_frame(index=False)[['date','fly_num']].drop_duplicates()
            ) >= 7:
        '''

        # TODO TODO TODO should i be recomputing now that i collected the new
        # kiwi/control data after the imaging system had a lot of time to drift?
        # check hists of dF/F / something (to compare old megamat/etc to new
        # kiwi/control data)?
        # TODO TODO maybe now i should switch to always z-scoring/similar the dF/F input
        # (before passing thru a fn to get spike deltas out), so that i can tune on one
        # panel and run on another more easily? (current attempt to tune on megamat and
        # run on control/kiwi had all silent cells in first call)

        # TODO TODO cleaner solution for this hack (probably involving preserving
        # panel throughout, and splitting each panel out before passing thru model, then
        # just always recompute model and do all in one run? now doing a prior run just
        # to save model, then later runs to pass each particular panel thru model)
        #
        # hack to only fit model if we are passing all panel (including validation)
        # data, on all flies (shape is 198x517 there)
        if (certain_df.shape[1] > 500 and len(certain_df) > 150 and

                set(certain_df.index.get_level_values('panel')) == {
                    'megamat', 'validation2', 'glomeruli_diagnostics'
                } and len(
                    certain_df.columns.to_frame(index=False)[['date','fly_num']
                        ].drop_duplicates()
                # NOTE: 9 final megamat flies and 5 final validation2 flies (after
                # dropping the 1 Betty wanted). see reproducing.md or
                # CSVs under data/sent_to_anoop/v1 for the specific flies.
                ) == (9 + 5)
            ):

            use_saved_dff_to_spiking_model = False
        else:
            use_saved_dff_to_spiking_model = True

    except KeyError:
        use_saved_dff_to_spiking_model = True

    # this option currently can't actually trigger recomputation that wouldn't happen
    # anyway... (always recomputed if input data is large enough, never otherwise)
    # TODO delete this option then?
    if use_saved_dff_to_spiking_model and should_ignore_existing('dff2spiking'):
        warn('would NOT have saved dff->spiking model, but requested regeneration of '
            'it!\n\nchange args to run on all data (so that model would get saved. see '
            'reproducing.md), OR remove `-i dff2spiking` option.'
            '\n\nexiting!'
        )
        sys.exit()

    # for histograms of dF/F, transformed versions, or estimated spike deltas derived
    # from one of the former
    n_bins = 50

    if not use_saved_dff_to_spiking_model:
        # just for histogram in loop below
        tidy_merged = merged_dff_and_hallem.reset_index()
        tidy_pebbled = fly_mean_df.reset_index()

        # TODO TODO may want to also plot on just panel subsets (here? or below, but
        # easier to do on diff scaling choices here, if i want that)
        # (only panel subsets for hist plots, not linear fit plots?)

        for scaling_method, col_desc in scaling_method2desc.items():
            curr_dff_col = scaling_method_to_col(scaling_method)
            assert curr_dff_col in merged_dff_and_hallem, f'missing {curr_dff_col}'

            # TODO put all these hists into a subdir? cluttering folder...

            fig, ax = plt.subplots()
            sns.histplot(data=tidy_merged, x=curr_dff_col, bins=n_bins, ax=ax)
            ax.set_title('pebbled (only odors & receptors also in Hallem)')
            # should be same subset of data used to fit dF/F->spiking model
            # (and same values, when scaling method matches scaling_method_to_use)
            savefig(fig, plot_dir, f'hist_pebbled_hallem-overlap_{curr_dff_col}')

            fig, ax = plt.subplots()
            sns.histplot(data=tidy_pebbled, x=curr_dff_col, bins=n_bins, ax=ax)
            ax.set_title('all pebbled')
            savefig(fig, plot_dir, f'hist_pebbled_{curr_dff_col}')

            # TODO also hist megamat subset of each of these? or at least of the pebbled
            # itself?
            # TODO or just loop over panels? easier below?


    # TODO iterate over options and verify that what i'm using is actually the best (or
    # not far off)?
    #scaling_method_to_use = None

    # 'to-avg-max'/'split-minmax-to-avg'/None all produce extremely visually similar
    # megamat/est_orn_spike_deltas*.pdf plots (including the correlation plots)
    # (as expected, since they keep 0)
    #
    # fit quality between these two identical, as expected
    # (one is the other multiplied by scalar)
    scaling_method_to_use = 'to-avg-max'
    #
    # TODO delete
    #scaling_method_to_use = 'maxabs'

    # TODO delete
    # fit quality also identical between these two.
    #
    # pretty junk w/ just a single line, b/c so much less negative data.
    # also pretty junk w/ 2 no-constant lines. negative one actually has opposite slope
    # from what i'd expect...
    #scaling_method_to_use = 'split-minmax'
    #scaling_method_to_use = 'split-minmax-to-avg'

    # TODO delete. not good.
    #scaling_method_to_use = 'zscore'

    add_constant = False

    # tested w/ None, 'split-minmax', and 'split-minmax-to-avg'. in all cases, the fit
    # on the negative dF/F data (and aligned subset of Hallem data) looked very bad (fit
    # was equivalent across the 2 cases, as expected). slope was negative, so more
    # negative dF/F meant less inhibition, which is nonsense.
    #
    # scaling methods were verified to not be re-ordering negative component of data.
    #
    # TODO maybe also plot just negative dF/F data (w/ aligned hallem data), to sanity
    # check the fact i was getting negative slopes?
    separate_inh_model = False

    col_to_fit = scaling_method_to_col(scaling_method_to_use)

    # TODO factor all model fitting + plotting (w/ CIs) into some hong2p fns?
    # TODO factor statsmodels fitting (+ plotting as matches seaborn)
    # into hong2p.viz (-> share w/ use in
    # natural_odors/scripts/kristina/lit_total_conc_est.py)

    # TODO refactor to move type of model to one place above?
    # NOTE: RegressionResults does not seem to be a subclass of
    # RegressionResultsWrapper. sad.
    def fit_dff2spiking_model(to_fit: pd.DataFrame) -> Tuple[RegressionResultsWrapper,
        Optional[RegressionResultsWrapper]]:

        # would need to dropna otherwise
        assert not to_fit.isna().any().any()
        to_fit = to_fit.copy()
        y_train = to_fit[spike_delta_col]

        # TODO try adding (0, 0) as as point, even if still using Ax+b as a model? is
        # that actually a valid practice? probably not, right?

        if add_constant:
            X_train = sm.add_constant(to_fit[col_to_fit])
        else:
            X_train = to_fit[col_to_fit].to_frame()

        if not separate_inh_model:
            # TODO why does this model produce a different result from the seaborn call
            # above (can tell by zooming in on upper right region of plot)??
            # TODO rename to "results"? technically the .fit() returns a results wrapper
            # or something (and do i only want to serialize the model part? can that
            # even store the parameters separately) (online info seems to say it should
            # return RegressionResults, so not sure why i'm getting
            # RegressionResultsWrapper...)
            model = sm.OLS(y_train, X_train).fit()
            inh_model = None
        else:
            nonneg = X_train[col_to_fit] >= 0
            neg = X_train[col_to_fit] < 0

            model = sm.OLS(y_train[nonneg], X_train[nonneg]).fit()
            inh_model = sm.OLS(y_train[neg], X_train[neg]).fit()

        return model, inh_model


    def predict_spiking_from_dff(df: pd.DataFrame, model: RegressionResultsWrapper,
        inh_model: Optional[RegressionResultsWrapper] = None, *, alpha=0.05,
        ) -> pd.DataFrame:
        """
        Returns dataframe with 3 additional columns: [est_spike_delta_col,
        <est_spike_delta_col>_ci_[lower|upper] ]
        """
        # TODO doc input requirements

        # TODO delete unless add_constant line below w/ series input might mutate df
        # (unlikely)
        df = df.copy()

        # would otherwise need to dropna
        assert not df.isna().any().any()

        # TODO assert saved model only has const term if add_constant?
        # do above where we load model + choices?

        # TODO see if this can be replaced w/ below (and do in other place if so)
        if add_constant:
            # returns a DataFrame w/ an extra 'const' col (=1.0 everywhere)
            X = sm.add_constant(df[col_to_fit])
        else:
            X = df[col_to_fit].to_frame()

        if not separate_inh_model:
            y_pred = model.get_prediction(X)

            # TODO what are obs_ci_[lower|upper] cols? i assume i'm right to use
            # mean_ci_[upper|lower] instead (seems so)?
            # https://stackoverflow.com/questions/60963178
            #
            # alpha=0.05 by default (in statsmodels, if not passed)
            pred_df = y_pred.summary_frame(alpha=alpha)

            predicted = y_pred.predicted
        else:
            # fly_mean_df input (call where important estimates get added) currently has
            # an index that would fail the verify_integrity=True checks below, so saving
            # this index to restore later.
            # TODO do i actually need to restore index tho?
            # TODO delete?
            #index = X.index
            X = X.reset_index(drop=True)

            nonneg = X[col_to_fit] >= 0
            neg = X[col_to_fit] < 0

            y_pred_nonneg = model.get_prediction(X[nonneg])
            y_pred_neg = inh_model.get_prediction(X[neg])

            pred_df_nonneg = y_pred_nonneg.summary_frame(alpha=alpha)
            pred_df_neg = y_pred_neg.summary_frame(alpha=alpha)

            predicted_nonneg = pd.Series(
                data=y_pred_nonneg.predicted, index=X[nonneg].index
            )
            predicted_neg = pd.Series(data=y_pred_neg.predicted, index=X[neg].index)

            pred_df = pd.concat([pred_df_nonneg, pred_df_neg], verify_integrity=True)

            # just on the RangeIndex of input (should have all consecutive indices from
            # start to end after concatenating)
            pred_df = pred_df.sort_index()
            assert pred_df.index.equals(X.index)

            predicted = pd.concat([predicted_nonneg, predicted_neg],
                verify_integrity=True
            )
            predicted = predicted.sort_index()
            assert predicted.index.equals(X.index)

            # TODO TODO restore (w/ above)? will this make predict_spiking_from_dff fn
            # return val make more sense? what was issue again?
            # TODO delete?
            #X.index = index

        # NOTE: .get_prediction(...) seems to return an object where more information is
        # available about the fit (e.g. confidence intervals, etc). .predict(...) will
        # just return simple output of model (same as <PredictionResult>.predicted).
        # (also seems to be same as pred_df['mean'])
        assert np.array_equal(predicted, pred_df['mean'])
        if not separate_inh_model:
            assert np.array_equal(predicted, model.predict(X))

        # TODO was this broken in separate inh case? (still think that case was
        # probably a dead end, so not necessarily worth fixing...)
        # (but megamat/est_orn_spike_deltas[_corr].pdf plots were all NaN it seems?)
        # (not sure i can repro)
        df[est_spike_delta_col] = predicted

        # TODO how are these CI's actually computed? how does that differ from how
        # seaborn computes them? why are they different?
        # TODO what are obs_ci_[lower|upper]? seems newer versions of statsmodels might
        # not have them anyway? or at least they aren't documented...
        for c in ('mean_ci_lower', 'mean_ci_upper'):
            df[f'{est_spike_delta_col}{c.replace("mean", "")}'] = pred_df[c]

        # TODO sort df by est_spike_delta_col before returning? would that make
        # plotting fn avoid need to do that? or prob just do in plotting fn...
        return df


    # TODO (reword to make accurate again) delete _model kwarg.
    # just using to test serialization of OLS model, since i can't figure out why this
    # equality check fails (no .equals avail):
    # > model.save('test_model.p')
    # > deserialized_model = sm.load('test_model.p')
    # > deserialized_model == model
    # False
    # > deserialized_model.remove_data()
    # > model.remove_data()
    # > deserialized_model == model
    # False
    def plot_dff2spiking_fit(df: pd.DataFrame, model: RegressionResultsWrapper,
        inh_model: Optional[RegressionResultsWrapper] = None, *, scatter=True,
        title_prefix=''):
        """
        Args:
            scatter: if True, scatterplot merged data w/ a hue for each fly. otherwise,
                plot a 2d histogram of data.
        """

        assert col_to_fit in df.columns
        assert spike_delta_col in df.columns

        ci_lower_col = f'{est_spike_delta_col}_ci_lower'
        ci_upper_col = f'{est_spike_delta_col}_ci_upper'
        est_cols = (est_spike_delta_col, ci_lower_col, ci_upper_col)
        assert all(x not in df.columns for x in est_cols)

        if separate_inh_model:
            assert inh_model is not None
        else:
            assert inh_model is None

        # functions passed to FacetGrid.map[_dataframe] must plot to current Axes
        ax = plt.gca()

        # TODO was seaborn results suggesting i wanted alpha 0.025 for 95% CI here?
        # (honestly, seaborn CI [which is supposedly 95%, tho bootstrapped] looks wider
        # in all cases)
        #
        # from looking at statsmodels code, I'm pretty sure their "95% CI" is centered
        # on estimate, w/ alpha/2 on either side (so 0.05 correct for 95%, not 0.025)
        alpha_for_ci = 0.05

        # TODO include what alpha is in name of cols returned from predict (-> delete
        # explicit pass-in here)?
        df = predict_spiking_from_dff(df, model, inh_model, alpha=alpha_for_ci)

        assert all(x in df.columns for x in est_cols)

        plot_kws = dict(ax=ax, data=df, y=spike_delta_col, x=col_to_fit)
        if scatter:
            sns.scatterplot(hue='fly_id', palette=fly_palette, legend='full',
                edgecolors='none', **plot_kws
            )
        else:
            # TODO set bins= (seems OK w/o)?
            #
            # default blue 2d hist color would probably not work well w/ current blue
            # fit line
            sns.histplot(color='red', cbar=True, **plot_kws)

        # TODO can i replace all est_df below w/ just df?

        xs = df[col_to_fit]
        if not separate_inh_model:
            est_df = df
        else:
            neg = df[col_to_fit] < 0
            nonneg = df[col_to_fit] >= 0

            est_df = df[nonneg]
            xs = xs[nonneg]

        # sorting was necessary for fill_between below to work correctly
        sorted_indices = np.argsort(xs).values
        xs = xs.iloc[sorted_indices]
        est_df = est_df.iloc[sorted_indices]

        color = 'blue'
        fill_between_kws = dict(alpha=0.2,
            # TODO each of these needed? try to recreate seaborn (set color_palette the
            # same / use that seaborn blue?)
            linestyle='', linewidth=0, edgecolor='white'
        )
        ax.plot(xs, est_df[est_spike_delta_col], color=color)
        ax.fill_between(xs, est_df[ci_lower_col], est_df[ci_upper_col], color=color,
            **fill_between_kws
        )
        if separate_inh_model:
            inh_color = 'red'
            xs = df[col_to_fit][neg]
            est_df = df[neg]

            # TODO refactor to share w/ above?
            sorted_indices = np.argsort(xs).values
            xs = xs.iloc[sorted_indices]
            est_df = est_df.iloc[sorted_indices]

            ax.plot(xs, est_df[est_spike_delta_col], color=inh_color)
            ax.fill_between(xs, est_df[ci_lower_col], est_df[ci_upper_col],
                color=inh_color, **fill_between_kws
            )

        if add_constant:
            # TODO refactor
            model_eq = (f'$\\Delta$ $spike$ $rate = {model.params[col_to_fit]:.1f} x + '
                f'{model.params["const"]:.1f}$'
            )
            # TODO assert no other parameters besides col_to_fit and const?
        else:
            model_eq = f'$\\Delta$ $spike$ $rate = {model.params[col_to_fit]:.1f} x$'
            # TODO assert no other parameters besides col_to_fit?

        if separate_inh_model:
            assert not add_constant, 'not yet implemented'
            model_eq = (f'{model_eq}\n$\\Delta$ '
                f'$spike$ $rate_{{inh}} = {inh_model.params[col_to_fit]:.1f} x$'
            )

        y_train = df[spike_delta_col]

        # https://en.wikipedia.org/wiki/Coefficient_of_determination
        ss_res = ((y_train - df[est_spike_delta_col])**2).sum()

        ss_tot = ((y_train - y_train.mean())**2).sum()

        # TODO make sense that this can be negative??? (for some of the glomerulus
        # specific fits. see by-glom_dff_vs_hallem__dff_scale-to-avg-max.pdf)
        # think so: https://stats.stackexchange.com/questions/12900
        # just means fit is worse than a horizontal line?
        #
        # using this R**2 just temporarily to be more comparable to values
        # reported for add_constant=True cases
        r_squared = 1 - ss_res / ss_tot

        if add_constant:
            assert np.isclose(r_squared, model.rsquared)

        # for why R**2 (as reported by model.rsquared) higher w/o intercept:
        # https://stats.stackexchange.com/questions/267325
        # https://stats.stackexchange.com/questions/26176

        # ...and some discussion about whether it makes sense to fit w/o intercept:
        # https://stats.stackexchange.com/questions/7948
        # https://stats.stackexchange.com/questions/102709

        # now only including this in title if we are able to recalculate R**2
        # (may be possible from model alone, but not sure how to access y_train from
        # model, if possible)
        #
        # TODO rsquared_adj useful in comparing these two models w/ diff # of
        # parameters?
        # TODO want anything else in here? don't really think p-val would be useful

        goodness_of_fit_str = (f'$R^2 = {r_squared:.4f}$'
            # TODO delete (or recalc for add_constant=False case, as w/ R**2
            # above)
            # TODO only include this if we have more than 1 param (i.e. if
            # add_constant=True). otherwise, R**2 should be equal to R**2_adj
            #f', $R^2_{{adj}} = {model.rsquared_adj:.4f}$'
        )
        ci_str = f'{((1 - alpha_for_ci) * 100):.3g}% CI on fit'
        title = f'{title_prefix}{model_eq}\n{goodness_of_fit_str}\n{ci_str}'
        ax.set_title(title)

        assert ax.get_xlabel() == col_to_fit
        desc = scaling_method2desc[scaling_method_to_use]
        ax.set_xlabel(f'{col_to_fit}\n{desc}')


    dff_to_spiking_model_path = plot_dir / 'dff2spiking_fit.p'
    if separate_inh_model:
        dff_to_spiking_inh_model_path = plot_dir / 'dff2spiking_inh_fit.p'

    dff_to_spiking_data_csv = plot_dir / 'dff2spiking_model_input.csv'

    # TODO move all fitting + plotting up above (~where current seaborn plotting
    # is), so i can do for all scaling choices, like w/ current seaborn plots (still
    # just doing modelling w/ one scaling choice)

    # so we have a record of which scaling choice we made (modelling plots already show
    # many parameters and this one isn't going to vary across modelling outputs from a
    # given run)
    # TODO TODO save this and other non-plot outputs we need to load saved dF/F->spiking
    # fit outside of plot dirs, so i can swap between png and pdf w/o issue...
    dff_to_spiking_model_choices_csv = plot_dir / 'dff2spiking_model_choices.csv'

    # TODO anything else i need to include in this?
    dff_to_spiking_model_choices = pd.Series({
        # TODO None survive round trip here? use 'none' / NaN instead?
        'scaling_method_to_use': scaling_method_to_use,
        'add_constant': add_constant,
        'separate_inh_model': separate_inh_model,
    })

    def read_dff_to_spiking_model_choices():
        bool_params = ('add_constant', 'separate_inh_model')

        ser = read_series_csv(dff_to_spiking_model_choices_csv,
            # TODO some way to infer add_constant dtype correctly (as bool)
            # (this didn't work)
            #dtype={'add_constant': bool}
        )

        for x in bool_params:
            if not type(ser[x]) is bool:
                x_lower = ser[x].lower()
                assert x_lower in ('true', 'false')
                ser[x] = x_lower == 'true'

        return ser

    if not use_saved_dff_to_spiking_model:
        to_csv(dff_to_spiking_model_choices, dff_to_spiking_model_choices_csv,
            header=False
        )
        # to check no more dtype issues (and just that we saved correctly)
        saved = read_dff_to_spiking_model_choices()
        assert saved.equals(dff_to_spiking_model_choices)
        del saved

        # TODO also add depth col (when available) here?
        cols_to_save = ['fly_id', 'odor', 'glomerulus', dff_col]

        if col_to_fit != dff_col:
            cols_to_save.append(col_to_fit)

        cols_to_save.append(spike_delta_col)

        assert all(c in merged_dff_and_hallem.columns for c in cols_to_save)
        dff_to_spiking_data = merged_dff_and_hallem[cols_to_save].copy()

        dff_to_spiking_data = dff_to_spiking_data.rename({
                col_to_fit: f'{col_to_fit} (X_train)',
                spike_delta_col: f'hallem_{spike_delta_col} (y_train)',
            }, axis='columns'
        )
        shape_before = dff_to_spiking_data.shape

        dff_to_spiking_data = dff_to_spiking_data.merge(fly_ids, left_on='fly_id',
            right_index=True
        )
        assert dff_to_spiking_data.shape == (shape_before[0], shape_before[1] + 2)
        assert num_null(dff_to_spiking_data) == 0

        # NOTE: ['odor', 'fly_id', 'glomerulus'] would be unique if not for those few
        # odors that are in two panels in one fly (e.g. 'ms @ -3' in diag and megamat).
        # we don't currently have 'panel' info in this df, so may need to keep in mind
        # if using saved data.
        # TODO include panel? (would need to merge in a way that preserves that. don't
        # think i currently do... shouldn't really matter)

        # TODO also save hallem receptor in a col here?
        to_csv(dff_to_spiking_data, dff_to_spiking_data_csv, index=False,
            date_format=date_fmt_str
        )

        model, inh_model = fit_dff2spiking_model(merged_dff_and_hallem)

        # TODO also save model.summary() to text file?
        cprint(f'saving dF/F -> spike delta model to {dff_to_spiking_model_path}',
            _cprint_color
        )
        # TODO delete
        # TODO TODO is this deterministic (seems so? diffing output file against
        # an older one returned no change)? can i seed it to make it so (if not)?  test
        # w/ -c! (reason for pebbled/uniform repro issue?)
        #print(f'is {dff_to_spiking_model_path} saving deterministic? can it be?')
        #import ipdb; ipdb.set_trace()
        #
        model.save(dff_to_spiking_model_path)

        if separate_inh_model:
            cprint(
                f'saving separate inhibition model to {dff_to_spiking_inh_model_path}',
                _cprint_color
            )
            inh_model.save(dff_to_spiking_inh_model_path)

        # TODO delete / put behind checks flag
        #deserialized_model = sm.load(dff_to_spiking_model_path)
        # TODO other comparison that would work after model.remove_data()?
        # care to remove_data? probably not
        #
        # ok, this is true at least...
        # TODO why is this failing now (seems to only be a line containing Time, and
        # only in the time part of that line. doesn't matter.)
        # from diffing these:
        # Path('model_summary.txt').write_text(str(model.summary()))
        # Path('deser_model_summary.txt').write_text(str(deserialized_model.summary()))
        # tom@atlas:~/src/al_analysis$ diff model_summary.txt deser_model_summary.txt
        # 7c7
        # < Time:                        15:46:35   Log-Likelihood:                         -17078.
        # ---
        # > Time:                        15:46:46   Log-Likelihood:                         -17078.
        #
        # TODO find a replacement check?
        #assert str(deserialized_model.summary()) == str(model.summary())
        #

        # TODO (reword to update / delete) use _model kwarg in predict to check
        # serialized->deser model is behaving same (below, where predict is called)
    else:
        # TODO TODO print some short summary of this data (panels, numbers of flies,
        # etc)
        cprint(f'using saved dF/F -> spiking model {dff_to_spiking_model_path}',
            _cprint_color
        )
        if separate_inh_model:
            cprint(
                f'using separate inhibition model from {dff_to_spiking_inh_model_path}',
                _cprint_color
            )

        cached_model_choices = read_dff_to_spiking_model_choices()
        if not cached_model_choices.equals(dff_to_spiking_model_choices):
            # TODO reword. -i dff2spiking w/ input data < all of it will not actually do
            # anything (cause model only ever saved if all data passed in)
            warn('current hardcoded model choices did not match those from saved model!'
                '\nre-run, adding `-i dff2spiking` to overwrite cached model (or just '
                'w/ all data as input... see reproducing.md) exiting!'
            )
            sys.exit()

        # TODO TODO save this and other non-plot outputs we need to load saved
        # dF/F->spiking fit outside of plot dirs, so i can swap between png and pdf w/o
        # issue...
        #
        # possible this alone has the input data, but save/loading that in parallel,
        # since i couldn't figure out how to access from here
        model = sm.load(dff_to_spiking_model_path)
        if separate_inh_model:
            inh_model = sm.load(dff_to_spiking_inh_model_path)
        else:
            inh_model = None

        # TODO load + summarize model input data
        # TODO +also load+summarize model choices

    # TODO still show if verbose=True or something?
    #print('dF/F -> spike delta model summary:')
    #print(model.summary())

    X0_test = pd.DataFrame({col_to_fit: 0.0}, index=[0])
    if add_constant:
        X0_test = sm.add_constant(X0_test)

    y0 = model.get_prediction(X0_test).predicted
    if add_constant:
        # may be close, but unlikely to equal 0 exactly
        assert y0 != 0.0
    else:
        assert y0 == 0.0

    # don't want to clutter str w/ the most typical values of these
    exclude_param_for_vals = {
        # always want to show scaling_method_to_use value
        #
        # TODO when these are true, maybe just include the str (w/o the '-True' suffix)?
        'add_constant': False,
        'separate_inh_model': False,
    }
    assert all(k in dff_to_spiking_model_choices.keys() for k in exclude_param_for_vals)
    params_for_suffix = {k: v for k, v in dff_to_spiking_model_choices.items()
        if not (k in exclude_param_for_vals and v == exclude_param_for_vals[k])
    }

    param_abbrevs = {
        'scaling_method_to_use': 'dff_scale',
        'add_constant': 'add_const',
        'separate_inh_model': 'separate_inh',
    }
    # TODO also thread (something like) this thru to be included in titles?
    dff2spiking_choices_str = '__'.join([
        f'{param_abbrevs[k] if k in param_abbrevs else k}-{v}'
        for k, v in params_for_suffix.items()
    ])

    pebbled_param_dir_prefix = f'{dff2spiking_choices_str}__'

    if not use_saved_dff_to_spiking_model:
        plot_fname = f'dff_vs_hallem__{dff2spiking_choices_str}'

        fig, _ = plt.subplots()
        # TODO factor into same fn that fits model?
        plot_dff2spiking_fit(merged_dff_and_hallem, model)
        # normalize_fname=False to prevent '__' from getting replaced w/ '_'
        savefig(fig, plot_dir, plot_fname, normalize_fname=False)

        fig, _ = plt.subplots()
        # this one should plot fit over a 2d hist of data
        plot_dff2spiking_fit(merged_dff_and_hallem, model, scatter=False)
        savefig(fig, plot_dir, f'{plot_fname}_hist2d', normalize_fname=False)

        _seen_group_vals = set()
        def fit_and_plot_dff2spiking_model(*args, group_col=None, **kwargs):
            assert len(args) == 0
            assert 'label' not in kwargs
            # TODO OK to throw away color kwarg like this? my plotting fn uses
            # fly_palette internally...
            assert set(kwargs.keys()) == {'data', 'color'}

            df = kwargs['data']
            model, inh_model = fit_dff2spiking_model(df)

            group_vals = set(df[group_col].unique())
            assert len(group_vals) == 1
            group_val = group_vals.pop()
            group_tuple = (group_col, group_val)
            assert group_tuple not in _seen_group_vals, f'{group_tuple=} already seen'
            _seen_group_vals.add(group_tuple)

            assert group_col in ('glomerulus', 'depth_bin')
            if group_col == 'glomerulus':
                # TODO add receptors in parens after glom, for easy ref to hallem paper?
                if roi_depths is not None:
                    avg_depth_col = f'avg_{roi_depth_col}'
                    assert df[avg_depth_col].nunique() == 1
                    avg_roi_depth_um = df[avg_depth_col].unique()[0]

                    title_prefix = \
                        f'{group_val} (avg depth: {avg_roi_depth_um:.1f} $\\mu$m)\n'
                else:
                    title_prefix = f'{group_val}\n'

            elif group_col == 'depth_bin':
                title_prefix = f'{group_val} $\\mu$m\n'
                # TODO delete (or get wrap to work)
                '''
                title_prefix = (f'{group_val} $\\mu$m\n'
                    # TODO like this second line?
                    # doesn't indicate counts either...
                    # TODO make this wrap at a certain limit? overlaps across facets
                    # now, and that section unreadable
                    f'{",".join(sorted(df.glomerulus.unique()))}\n'
                )
                '''

            plot_dff2spiking_fit(df, model, inh_model, title_prefix=title_prefix)


        if roi_depths is not None:
            avg_depth_per_glomerulus = merged_dff_and_hallem.groupby('glomerulus')[
                roi_depth_col].mean()
            avg_depth_per_glomerulus.name = f'avg_{roi_depth_col}'
            merged_dff_and_hallem = merged_dff_and_hallem.merge(
                avg_depth_per_glomerulus, left_on='glomerulus', right_index=True
            )

            merged_dff_and_hallem = merged_dff_and_hallem.sort_values(
                f'avg_{roi_depth_col}').reset_index(drop=True)

        else:
            merged_dff_and_hallem = merged_dff_and_hallem.sort_values('glomerulus'
                ).reset_index(drop=True)

        col = 'glomerulus'
        # TODO default behavior do this anyway? easier way?
        grid_len = int(np.ceil(np.sqrt(merged_dff_and_hallem[col].nunique())))

        # to remove warning 'The figure layout has changed to tight' otherwise generated
        # when each FacetGrid is contructed (b/c my mpl rcParams use constrained layout
        # by default).
        #
        # setting layout='tight' via FacetGrid subplot_kws didn't work to fix (produced
        # an error), nor could gridspec_kws, as those are ignored if col_wrap passed.
        with mpl.rc_context({'figure.constrained_layout.use': False}):
            g = sns.FacetGrid(data=merged_dff_and_hallem, col=col, col_wrap=grid_len)
            g.map_dataframe(fit_and_plot_dff2spiking_model, group_col=col)

            viz.fix_facetgrid_axis_labels(g)
            savefig(g, plot_dir, f'by-glom_{plot_fname}', normalize_fname=False)

        if roi_depths is not None:
            # TODO maybe also show list of glomeruli in each bin for plot below?
            # (would need to get wrapping to work in commented code above. too many to
            # list nicely in one line.)
            # TODO or least print these glomeruli (+ value_counts?)

            n_depth_bins_options = (2, 3, 4, 5)
            for n_depth_bins in n_depth_bins_options:
                # should be a series of length equal to merged_dff_and_hallem
                # (w/ CategoricalDtype)
                depth_bins = pd.cut(merged_dff_and_hallem[roi_depth_col], n_depth_bins)

                df = merged_dff_and_hallem.copy()

                df['depth_bin'] = depth_bins

                col = 'depth_bin'
                grid_len = int(np.ceil(np.sqrt(df[col].nunique())))

                with mpl.rc_context({'figure.constrained_layout.use': False}):
                    g = sns.FacetGrid(data=df, col=col, col_wrap=grid_len)
                    g.map_dataframe(fit_and_plot_dff2spiking_model, group_col=col)

                    viz.fix_facetgrid_axis_labels(g)
                    savefig(g, plot_dir, f'by-depth-bin-{n_depth_bins}_{plot_fname}',
                        normalize_fname=False
                    )

        # TODO try a depth specific model too (seems not worth, from depth binned plots)
        # (quite clear overall scale changes when i need to avoid strongest responding
        # plane b/c contamination. e.g. often VA4 has contamination p-cre response in
        # highest (strongest) plane, from one of the nearby/above glomeruli that
        # responds to that)
        #
        # TODO try directly estimating fn like:
        # A(B*depth*x)? how would be best?
        # since it's linear, couldn't i just do:
        # A*depth*x?
        # i guess i might like to try nonlinear fns of depth, or at least to make depth
        # scaling more interpretable, but idk...
        # TODO or maybe i want A(B*depth + x)?
        # either way, i want to make sure that 0 dff is always 0 est spike delta
        # (regardless of depth), so making me thing i dont want this additive model...
        # TODO try scipy optimization stuff?
        #
        # TODO also compare to just adding a param for depth in linear model
        # (but otherwise still using all data for fit)
        # TODO possible to have automated detection of outlier glomeruli? i.e. those
        # benefitting from different fits


    # TODO TODO histogram of est spike deltas this spits out
    # (to what extent is that already done in loop over panels below?)
    # TODO are only NaNs in the dff_col in here coming from setting-wrong-odors-NaN
    # above?
    # TODO can delete branches in predict only for plotting w/ this input (don't
    # actually care about the plot here)
    #
    # this predict(...) call is the one actually adding the estimated spike deltas,
    # computed from data to be modelled (which can be different from data originally
    # used to compute dF/F -> spike delta est fn).
    fly_mean_df = predict_spiking_from_dff(fly_mean_df, model, inh_model)

    # TODO also save fly_mean_df similar to how we save dff2spiking_model_input.csv
    # above (for other people to analyze arbitrary subsets of est spike deltas /
    # whatever) (maybe refactor above to share + use panel_prefix from below?)
    # TODO + do same under each panel dir, for each panels data?

    # TODO TODO test whether downstream code works fine w/o stopping here (at least
    # check equiv in megamat case. may want to hardcode a skip of the validation2 panel
    # by default anyway [w/ a flag])
    # TODO would checking megamat subset of fly_mean_df is same between two runs (w/ all
    # data vs just megamat flies) get us most/all of the way there?
    #
    # TODO delete hack (see corresponding hack above where this flag is defined)
    # (plus now rest of modeling code loops over panels anyway, no?)
    if not use_saved_dff_to_spiking_model:
        print('EXITING EARLY AFTER HAVING SAVED MODEL ON ALL DATA (analyze specific '
            'panels with additional al_analysis runs)!'
        )
        sys.exit()
    #

    # TODO plot histogram of fly_mean_df[est_spike_delta_col] (maybe resting on the x
    # axis in the same kind of scatter plot of (x=dF/F, y=delta spike rate,
    # hue=fly_id)?)

    # TODO rename? it's a series here, not a df (tho that should change in next
    # re-assignment, the one where RHS is unstacked...)
    mean_est_df = fly_mean_df.reset_index().groupby(['panel', 'odor', 'glomerulus'],
        sort=False)[est_spike_delta_col].mean()

    # then odors will be columns and glomeruli will be rows, which is same as
    # orns.orns().T
    mean_est_df = mean_est_df.unstack(['panel', 'odor'])

    # TODO delete?
    # TODO why was this getting triggered when run w/ megamat data, but not w/
    # kiwi/control data? not sure it matters...
    # was gonna sort again here, but seems true already
    #assert mean_est_df.equals(sort_odors(mean_est_df))
    #
    mean_est_df = sort_odors(mean_est_df)

    # TODO factor all dF/F -> spike delta fitting (above, ending ~here) into one fn in
    # here at least, to make model_mb_responses more readable

    # TODO TODO possible to get spike rates for any of the other door data sources?
    # available in their R package?

    # TODO TODO how to do an ephaptic model? possible to optimize one using my
    # data as input (where we never have the channels separated)? if using hallem, for
    # what fraction of sensilla do we have all / most contained ORN types?
    # TODO which data to use for ephaptic effects / how?
    # TODO plot ephaptic model adjusted dF/F (subtracting from other ORNs in sensilla)
    # vs spike rate?

    # TODO TODO plot mean_est_df vs same subset of hallem, just for sanity checking
    # (as a matrix in each case)
    # TODO TODO and do w/ my ORN input (untransformed by dF/F -> spike delta model?),
    # also my ORN input subset to hallem stuff
    # (computing correlation in each case, with nothing going thru MB model first)

    # TODO drop all non-megamat odors prior to running through fit_mb_model?
    # (no, want to model other stuff now too...) (am i not currently doing this tho?)
    # (diagnostics prob gonna produce much lower KC sparsity)
    # (are any of the non-HALLEM odors (which is probably most non-megamat odors?)
    # actually influencing model in fit_mb_model tho?)

    # TODO are relative sparsities recapitulating what remy seems? even broadly?

    # TODO TODO try fitting on hallem, and then running on my data passed thru
    # dF/F model (fitting on all hallem might produce very different thresholds from
    # fitting on the subset of odors remy uses!!!)

    hallem_for_comparison = hallem_delta_wide.copy()
    assert hallem_for_comparison.index.str.contains(' @ -3').all()
    # so things line up in comparison_orns path (fit_mb_model hallem data has '@ -2'
    # for each conc)
    hallem_for_comparison.index = hallem_for_comparison.index.str.replace(
        ' @ -3', ' @ -2'
    )
    # TODO delete? actually needed by anything?
    hallem_for_comparison.index.name = 'odor1'

    # TODO move all hallem version of this above loop over panels (done?)
    tidy_hallem = hallem_for_comparison.T.stack()
    # just to rename the second level from 'odor1'->'odor', to be consistent w/
    # above
    tidy_hallem.index.names = ['glomerulus', 'odor']
    tidy_hallem.name = spike_delta_col
    tidy_hallem = tidy_hallem.reset_index()
    fig, ax = plt.subplots()
    sns.histplot(data=tidy_hallem, x=spike_delta_col, bins=n_bins, ax=ax)
    ax.set_title('all Hallem')
    savefig(fig, plot_dir, 'hist_hallem')

    # TODO move inside loop (doing for every panel, not just megamat)?
    tidy_hallem_megamat = tidy_hallem.loc[
        tidy_hallem.odor.apply(odor_is_megamat)
    ].copy()

    fig, ax = plt.subplots()
    sns.histplot(data=tidy_hallem_megamat, x=spike_delta_col, bins=n_bins, ax=ax)
    ax.set_title('Hallem megamat')
    savefig(fig, plot_dir, 'hist_hallem_megamat')

    # will be saved alongside later params, inside each model output dir
    # (for reproducibility)
    extra_params = {
        f'dff2spiking_{k}': v for k, v in dff_to_spiking_model_choices.to_dict().items()
    }

    # slightly nicer number (less sig figs) that is almost exactly the same as the
    # sparsity computed on remy's data (from binarized outputs she gave me on
    # 2024-04-03)
    remy_sparsity = 0.0915

    checks = True
    if checks:
        # TODO compute + use separate remy sparsity from validation data (not sure i'm
        # actually going to continue doing any modelling for the validation data. don't
        # think any of it is making it in to paper)? have what i need for that already,
        # or need something new from her?
        #
        # 0.091491682899149
        remy_sparsity_exact = remy_megamat_sparsity()

        # remy_sparsity_exact - remy_sparsity: -8.317100850752102e-06
        assert abs(remy_sparsity_exact - remy_sparsity) <= 1e-5

    target_sparsities = (remy_sparsity,)

    # TODO delete (/ expand to collection + include kiwi/control)
    # (now that we also want sensitivity analysis there)
    # (replace w/ list of panels to NOT run sens analysis on? e.g. validation)
    #
    # if model_kws (in loop below) contains sensitivity_analysis=True, will only do that
    # analysis for this specific panel and target sparsity
    #panel_for_sensitivity_analysis = 'megamat'
    #target_sparsity_for_sensitivity_analysis = remy_sparsity
    #assert target_sparsity_for_sensitivity_analysis in target_sparsities

    # TODO delete eventually
    assert mean_est_df.equals(sort_odors(mean_est_df))

    # TODO support list values (-> iterate over)? (as long as directories would have
    # diff names)
    #
    # which panel(s) to use to "tune" the model (i.e. set the two inhibitory
    # parameters), to achieve the target sparsity. if a panel is not included in keys
    # here, it will just be tuned on it's own data.
    panel2tuning_panels = {
        'kiwi': ('kiwi', 'control'),
        'control': ('kiwi', 'control'),

        # TODO any way to salvage this idea? check dF/F distributions between the two
        # first? maybe z-score first or something? currently getting all silent cells in
        # first model (control + hemibrain) run this way.
        # TODO TODO and how do the parameters compare across the panels again?
        # i thought they were in a similar range? different enough i guess?
        #
        # running w/ `./al_analysis.py -d pebbled -n 6f -t 2023-04-22` for this.
        # (certain_df only has the flies i expected, which is the 9 megamat +
        # 5 validation + 9 kiwi/control flies)
        #'kiwi': ('megamat',),
        #'control': ('megamat',),

        # TODO also try using 'megamat' tuning for 'validation', and see how that
        # affects things?

        # TODO delete? was to test pre-tuning code working as expected.
        # (used new script al_analysis/check_pretuned_vs_not.py to compare responses +
        # spike counts from each)
        #
        # TODO TODO how to keep this in as an automated check? or move to a separate
        # test script (model_test.py, or something simliar?)? currently i need to
        # manually compare the outputs across the old/new dirs
        # (add panel2tuning_panels as kwarg of model_mb_responses -> make 2 calls?)
        #'megamat': ('megamat',)
        #
    }
    assert all(type(x) is tuple for x in panel2tuning_panels.values())
    # sorting so dir (which will include tuning panels) will always be the same
    panel2tuning_panels = {k: tuple(sorted(v)) for k, v in panel2tuning_panels.items()}

    tuning_panel_delim = '-'

    new_panels = {tuning_panel_delim.join(x) for x in panel2tuning_panels.values()}
    existing_panels = set(mean_est_df.columns.get_level_values('panel'))
    if any(x in existing_panels for x in new_panels):
        warn(f'some of {new_panels=} are already in {existing_panels=}! should only '
            'see this warning if testing that we can reproduce model output by '
            'pre-tuning with the same panel we later use to run the model!'
        )
    del new_panels, existing_panels

    # TODO want to drop the panel column level? or want to use it inside calls to
    # fit_and_plot...? groupby kwarg for dropping, if i want former?
    for panel, panel_est_df in mean_est_df.groupby('panel', axis='columns', sort=False):

        if panel == diag_panel_str:
            continue

        # TODO delete eventually
        assert panel_est_df.equals(sort_odors(panel_est_df))
        #
        # TODO delete (assertion above passing. seems we can rely on mean_est_df being
        # sorted)
        #panel_est_df = sort_odors(panel_est_df)

        panel_plot_dir = plot_dir / panel
        makedirs(panel_plot_dir)

        # these will have one row per model run, with all relevant parameters (as well
        # as a few other variables/statistics computed within model runs, e.g. sparsity)
        model_param_csv = panel_plot_dir / 'tuned_params.csv'
        model_params = None

        raw_dff_panel_df = sort_odors(certain_df.loc[panel], panel=panel)

        mean_fly_dff_corr = mean_of_fly_corrs(raw_dff_panel_df)

        # just checking that mean_of_fly_corrs isn't screwing up odor order, since
        # raw_dff_panel_df odors are sorted (and easier to check against panel_est_df,
        # as that doesn't have the repeats in it like raw_dff_panel_does, but the order
        # of the odors in the two should be the same)
        assert mean_fly_dff_corr.columns.equals(mean_fly_dff_corr.index)
        # this doesn't check .name, which is good, b/c mean_fly_dff_corr has 'odor1',
        # not 'odor'
        assert mean_fly_dff_corr.columns.equals(
            panel_est_df.columns.get_level_values('odor')
        )

        # TODO (just for ticklabels in plots) for kiwi/control at least (but maybe for
        # everything?) hide the '@ 0' part of conc strs [maybe unless there is another
        # odor w/ a diff conc, but may not matter]

        # TODO restore response matrix plot versions of these (i.e. plot responses in
        # addition to just corrs) (would technically be duped w/ ijroi versions, for
        # convenient comparison to 'est_orn_spike_deltas*' versions? or symlink to the
        # ijroi one?
        plot_corr(mean_fly_dff_corr, panel_plot_dir, 'orn_dff_corr',
            xlabel=f'ORN {dff_latex}'
        )

        fly_dff_hallem_subset = raw_dff_panel_df.loc[:,
            raw_dff_panel_df.columns.get_level_values('roi').isin(
                hallem_delta_wide.columns
            )
        ]
        mean_fly_dff_hallem_corr = mean_of_fly_corrs(fly_dff_hallem_subset)
        plot_corr(mean_fly_dff_hallem_corr, panel_plot_dir,
            'orn_dff_hallem-subset_corr',
            xlabel=f'ORN {dff_latex}\nHallem glomeruli only'
        )
        plot_corr(mean_fly_dff_hallem_corr, panel_plot_dir,
            'orn_dff_hallem-subset_corr-dist',
            xlabel=f'ORN {dff_latex}\nHallem glomeruli only', as_corr_dist=True
        )

        gh146_glomeruli = get_gh146_glomeruli()
        # NOTE: true for megamat at least, may not be true for validation2
        if panel == 'validation2':
            assert {'VA4'} == (
                gh146_glomeruli - set(raw_dff_panel_df.columns.get_level_values('roi'))
            )
        else:
            # TODO TODO TODO relax to not err on new data? presumably this is what was
            # failing?
            # TODO TODO just warn instead? only do anything if we seem to be running on
            # the final megamat data?
            print('update / delete gh146 glomeruli checking code')
            #assert 0 == len(
            #    gh146_glomeruli - set(raw_dff_panel_df.columns.get_level_values('roi'))
            #)

        fly_dff_gh146_subset = raw_dff_panel_df.loc[:,
            raw_dff_panel_df.columns.get_level_values('roi').isin(gh146_glomeruli)
        ]
        mean_fly_dff_gh146_corr = mean_of_fly_corrs(fly_dff_gh146_subset)
        plot_corr(mean_fly_dff_gh146_corr, panel_plot_dir,
            'orn_dff_gh146-subset_corr',
            xlabel=f'ORN {dff_latex}\nGH146 glomeruli only'
        )
        plot_corr(mean_fly_dff_gh146_corr, panel_plot_dir,
            'orn_dff_gh146-subset_corr-dist',
            xlabel=f'ORN {dff_latex}\nGH146 glomeruli only', as_corr_dist=True
        )

        # should i also be passing each *individual fly* data thru dF/F -> est spike
        # delta fn (-> recomputing)? should i be doing that w/ all of modeling?
        # (no, Betty and i agreed it wasn't worth it for now)

        # TODO no need for copy, right?
        # TODO maybe i don't need to drop panel here?
        #
        # also, why the double transpose here? est_df used apart from for this plot?
        # (b/c usage as comparison_orns below)
        #
        # NOTE: this should currently be saved as a pickle+CSV under each model output
        # directory, at orn_deltas.[csv|p] (done by fit_and_plot...)
        est_df = panel_est_df.droplevel('panel', axis='columns').T.copy()

        # TODO TODO also plot hemibrain filled version(s) of this
        est_corr = plot_responses_and_corr(est_df.T, panel_plot_dir,
            f'est_orn_spike_deltas_{dff2spiking_choices_str}',
            # TODO maybe borrow final part from scaling_method2desc (but current strs
            # there have more info than i want)
            xlabel=('est. ORN spike deltas\n'
                f'{dff_latex} scaling: {scaling_method_to_use}'
            ),
        )
        del est_corr

        # TODO TODO plot responses + corrs for (est_orn_spike_deltas + sfr) and
        # (hallem_spike_deltas + sfr) too. compare to values from dynamic ORNs and
        # deltas alone. (probably do in fit_mb_model internals plotting?)
        #
        # (actually care about adding sfr? does it actually change corrs? if so, to a
        # meaningful degree?)

        # TODO or just move before loop over panels?
        if panel == 'megamat':
            hallem_megamat = hallem_delta_wide.loc[
                # get_level_values('odor') should work whether panel_est_df has 'odor'
                # as one level of a MultiIndex, or as single level of a regular Index
                panel_est_df.columns.get_level_values('odor')
            ].sort_index(axis='columns')

            # TODO label cbar w/ spike delta units
            plot_responses_and_corr(hallem_megamat.T, panel_plot_dir,
                'hallem_spike_deltas', xlabel='Hallem OR spike deltas'
            )

            # TODO TODO also NaN-fill Hallem to hemibrain, and plot those responses (if
            # i haven't already somewhere else). no need to plot corrs there, as they
            # should be same as raw hallem.

            # TODO TODO only zero fill just as in fitting tho (how is it diff? at least
            # add comment about how it's diff...)? current method also drops stuff like
            # DA3, which is in hallem but not in my data...
            # TODO TODO leave that to one of the model_internals plots in that case?
            # maybe just delete this then?
            #
            # TODO does this change correlation (yes, moderately increased)?
            # TODO plot delta corr wrt above?
            # TODO print about what the reindex is dropping (if verbose?)?
            zerofilled_hallem = hallem_megamat.reindex(panel_est_df.index,
                axis='columns').fillna(0)
            plot_responses_and_corr(zerofilled_hallem.T, panel_plot_dir,
                'hallem_spike_deltas_filled', xlabel='Hallem OR spike deltas\n'
                '(zero-filled to my consensus glomeruli)'
            )

        # TODO TODO and same thing with my raw data honestly. not sure i have that.
        # here might not be the place though (top-level ijroi stuff?)
        # TODO TODO matrix plot actually making my est spike deltas as comparable
        # as possible to the relevant subset of the hallem data (+ relevant subset of my
        # data)
        # (not here, but at least once for master version of hallem and pebbled data,
        # maybe just in megamat context)

        # TODO just use one of the previous things that was already tidy? and already
        # had hallem data?
        tidy_est = panel_est_df.droplevel('panel', axis='columns').stack()
        tidy_est.name = est_spike_delta_col
        tidy_est = tidy_est.reset_index()

        fig, ax = plt.subplots()
        sns.histplot(data=tidy_est, x=est_spike_delta_col, bins=n_bins, ax=ax)
        ax.set_title(f'pebbled {panel}')
        # TODO or save in panel dir? this consistent w/ saving of hallem megamat stuff
        # above tho...
        savefig(fig, plot_dir,
            f'{pebbled_param_dir_prefix}hist_est-spike-delta_{panel}'
        )
        del tidy_est

        pebbled_input_df = panel_est_df
        responses_to_suffix = ''

        # TODO check outputs against those run previous way (without
        # explicitly passing inputs, when using + tuning on hallem)
        #hallem_input_df = hallem_for_comparison.T.copy()

        comparison_orns = None
        comparison_kc_corrs = None

        # TODO delete (still want this? or maybe for other panels, e.g. kiwi?)
        if panel != 'megamat':
            print('GET COMPARISON_ORNS (+ COMPARISON_KCS) WORKING ON VALIDATION2 DATA')
        #
        if panel == 'megamat':
            # TODO TODO don't i still want comparison_orns in validation case?
            # TODO TODO what about comparison_kcs in validation case? what betty
            # has said so far (re: validation modelling figures) is that we only want
            # sparsity + correlation, but not sure that'll remain true...
            comparison_orns = {
                'raw-dff': raw_dff_panel_df,

                # NOTE: this one does not have single fly data like raw_dff_panel_df
                # (it's just mean responses), so correlation computed not exactly
                # apples-to-apples with most others (but similar to how model output
                # corr computed, given model is run on mean data)
                'est-spike-delta': est_df,

                # TODO also a version zero-filling like fit_mb_model does internally
                # (happy enough w/ corr_diff plots i added in fit_mb_model?)
            }

            # TODO rename to comparison_kc_corrs or something? observed_mean_kc_corrs?
            #
            # this is a mean-of-fly-corrs (WAS for Remy's 4 final KC flies, but now
            # adapting to also load the older data too)
            comparison_kc_corrs = load_remy_megamat_mean_kc_corrs()

            # TODO replace these two lines w/ just sorting, if that works (would have to
            # add panel to both column and index, at least one manually...)
            # (name order already cluster order, in panel2name_order?)
            # (current strategy will probably no longer work w/ panel_est_df/est_df
            # having panel level...)
            assert set(est_df.index) == set(comparison_kc_corrs.index)
            comparison_kc_corrs = comparison_kc_corrs.loc[est_df.index, est_df.index
                ].copy()
            #

            # TODO also an as_corr_dist=True version of my mean ORN corrs (to finish off
            # fig 3 C)
            # TODO same for model corrs (corr.pdf under each model param dir)

            # TODO delete similar code in comparison_kc_corrs branch inside
            # fit_and_plot...
            #
            # TODO would probably need to redo diverging_cmap + vmin/vmax to work w/
            # correlation distance (this was from before i converted correlations to
            # correlation distances. could also move this before converting above...)
            # TODO or convert back to correlations, if currently as distances
            plot_corr(comparison_kc_corrs, panel_plot_dir, 'remy_kc_corr',
                xlabel='observed KCs'
            )
            plot_corr(comparison_kc_corrs, panel_plot_dir, 'remy_kc_corr-dist',
                xlabel='observed KCs', as_corr_dist=True
            )
            # TODO make corr diff plots wrt orn inputs
            # (probably just the raw dF/F, or maybe also the transformed stuff before
            # fitting?)
            # TODO compare uniform - hemibrain corr to experimental correlation change
            # from ORNs->KCs?
            # TODO load responses.[p|csv] from each dir -> compute corrs -> diff from
            # there (might just make a separate script for that...)?

            assert set(comparison_kc_corrs.index) == set(
                raw_dff_panel_df.index.get_level_values('odor1')
            )
            mean_orn_corrs = mean_of_fly_corrs(raw_dff_panel_df, square=False)
            mean_kc_corrs = corr_triangular(comparison_kc_corrs)

            assert mean_kc_corrs.index.equals(mean_orn_corrs.index)

            orn_col = 'mean_orn_corr'
            kc_col = 'mean_kc_corr'
            mean_orn_corrs.name = orn_col
            mean_kc_corrs.name = kc_col

            merged_corrs = pd.concat([mean_orn_corrs, mean_kc_corrs], axis='columns')

            # TODO TODO refactor to share w/ where i copied from
            fig, ax = plt.subplots()
            add_unity_line(ax)
            lineplot_kws = dict(
                ax=ax, data=merged_corrs, x=orn_col, y=kc_col, linestyle='',
                color='black'
            )
            marker_only_kws = dict(
                markers=True, marker='o', errorbar=None,

                # seems to default to white otherwise
                markeredgecolor='black',

                markerfacecolor='None',
                alpha=0.175,
            )
            # plot points
            sns.lineplot(**lineplot_kws, **marker_only_kws)

            metric_name = 'correlation'
            ax.set_xlabel(f'{metric_name} of raw ORN {dff_latex} (observed)')
            ax.set_ylabel(f'{metric_name} of KCs (observed)')

            metric_max = max(merged_corrs[kc_col].max(), merged_corrs[orn_col].max())
            metric_min = min(merged_corrs[kc_col].min(), merged_corrs[orn_col].min())

            plot_max = 1
            plot_min = -.5
            assert metric_max <= plot_max, \
                f'{param_dir}\n{desc=}: {metric_max=} > {plot_max=}'
            assert metric_min >= plot_min, \
                f'{param_dir}\n{desc=}: {metric_min=} < {plot_min=}'

            ax.set_xlim([plot_min, plot_max])
            ax.set_ylim([plot_min, plot_max])

            # should give us an Axes that is of square size in figure coordinates
            ax.set_box_aspect(1)

            spear_text, _, _, _, _ = bootstrapped_corr(merged_corrs, kc_col, orn_col,
                method='spearman',
                # TODO delete (for debugging)
                _plot_dir=panel_plot_dir
            )
            ax.set_title(spear_text)

            # TODO also include errorbars along both x and y here? (across flies whose
            # correlations went into mean corr)

            savefig(fig, panel_plot_dir, 'remy-kc_vs_orn-raw-dff_corrs')
            # (end part to refactor to share w/ copied code)


        model_kw_list = [
            # TODO TODO why is DA4m in in these but not in regular hallem input call in
            # separate list below? conform preprocessing to match (+ sort glomeruli in
            # all those internal plots?)
            dict(
                orn_deltas=pebbled_input_df,
                responses_to_suffix=responses_to_suffix,

                tune_on_hallem=False,
                pn2kc_connections='hemibrain',

                # NOTE: this will be removed for all target_sparsity values (and all
                # panels) except ones specified above to be used for sensitivity
                # analysis
                sensitivity_analysis=True,

                comparison_orns=comparison_orns,
                comparison_kc_corrs=comparison_kc_corrs,
            ),

            # TODO restore?
            ## TODO TODO this actually make sense to try (w/ tune_on_hallem=True
            ## and drop_receptors_not_in_hallem=False?) does my code even support
            ## currently?
            ##dict(orn_deltas=pebbled_input_df, tune_on_hallem=True,
            ##    pn2kc_connections='hemibrain'
            ##),

            #dict(orn_deltas=pebbled_input_df, tune_on_hallem=True,
            #    drop_receptors_not_in_hallem=True,
            #    pn2kc_connections='hemibrain'
            #),

            #dict(orn_deltas=pebbled_input_df, tune_on_hallem=False,
            #    drop_receptors_not_in_hallem=True,
            #    pn2kc_connections='hemibrain'
            #),

            ## TODO what should i actually use for n_claws?

            # TODO TODO + correlation plotting including that error, plotting error
            # in a separate stddev plot, as for rest of data elsewhere
            dict(
                orn_deltas=pebbled_input_df,
                responses_to_suffix=responses_to_suffix,
                tune_on_hallem=False,
                pn2kc_connections='uniform', n_claws=7, n_seeds=n_seeds,
                comparison_orns=comparison_orns,
                comparison_kc_corrs=comparison_kc_corrs,
            ),

            dict(
                orn_deltas=pebbled_input_df,
                responses_to_suffix=responses_to_suffix,
                tune_on_hallem=False,
                pn2kc_connections='hemidraw', n_claws=7, n_seeds=n_seeds,
                comparison_orns=comparison_orns,
                comparison_kc_corrs=comparison_kc_corrs,
            ),

            ## TODO TODO this actually make sense to try (w/ tune_on_hallem=True and
            ## drop_receptors_not_in_hallem=False?) does my code even support
            ## currently?
            ##dict(orn_deltas=pebbled_input_df, tune_on_hallem=True,
            ##    pn2kc_connections='uniform', n_claws=7, n_seeds=n_seeds
            ##),
        ]
        if panel == 'megamat':
            # TODO TODO make sure that my derived wPNKC from pratyush's data
            # actually has KCs getting at least some input from all the PN types i have
            # in wPNKC PN labels (and that i'm not subsetting down to "halfmat"
            # unnecessarily, from earlier comparisons to matt's stuff, or whatever)
            # TODO TODO could probably just compare outputs of hallem model w/ and
            # w/o _use_matt_wPNKC...
            # TODO TODO can pass hallem data in explicitly if that makes it
            # easier to get things handled exactly the same
            #
            # TODO delete (/ move inside fit_and_plot..., behind conditional checking
            # use_matt_wPNKC or whatever it's called)
            #print('switch matt calls to prats wPNKC!')

            # NOTE: model responses (including cache) should only include these odors.
            # could `-i model` if wanted to change and regen cache.
            sim_odors = sorted(hallem_for_comparison.index)
            # TODO make work again. need to make corr diff plot for all odors, w/
            # megamat ones pulled out. this seemed the easiest way...
            #sim_odors = None

            # parameter combinations to recreate preprint figures, using same Hallem
            # data as input (that Matt did when making those figures, before we had our
            # own ORN outputs)
            preprint_repro_model_kw_list = [
                # TODO TODO also try fitting on just 17-odor megamat subset of
                # hallem, to see if that really is part of the issue

                dict(pn2kc_connections='hemibrain',

                    # TODO TODO TODO delete/comment
                    # need to fix breakpoint hit in fit_mb_model
                    _use_matt_wPNKC=True,

                    sim_odors=sim_odors,

                    comparison_orns=hallem_for_comparison,
                    comparison_kc_corrs=comparison_kc_corrs,

                    # TODO TODO don't require this passed in! (do unconditionally)
                    _strip_concs_comparison_kc_corrs=True,
                ),

                dict(pn2kc_connections='uniform', n_claws=7,

                    # TODO TODO TODO delete/comment
                    # need to fix breakpoint hit in fit_mb_model
                    _use_matt_wPNKC=True,

                    # TODO probably also _add_back_methanoic_acid_mistake=True?
                    # shouldn't matter...
                    #_add_back_methanoic_acid_mistake=True,

                    n_seeds=n_seeds,

                    sim_odors=sim_odors,

                    comparison_orns=hallem_for_comparison,
                    comparison_kc_corrs=comparison_kc_corrs,

                    # TODO TODO don't require this passed in! (do unconditionally)
                    #
                    # since outputs of model will have ' @ -2' when using Hallem input,
                    # but KC comparison data has ' @ -3'. this will strip conc from all
                    # odor strings, when comparing data from these variables.
                    # NOTE: comparison_orns path currently strips unconditionally...
                    _strip_concs_comparison_kc_corrs=True,
                ),
                dict(pn2kc_connections='hemidraw', n_claws=7,
                    _use_matt_wPNKC=True,
                    n_seeds=n_seeds,
                    sim_odors=sim_odors,
                    comparison_orns=hallem_for_comparison,
                    comparison_kc_corrs=comparison_kc_corrs,
                    _strip_concs_comparison_kc_corrs=True,
                ),
            ]
            model_kw_list = model_kw_list + preprint_repro_model_kw_list

        # hack to skip long running models, if I want to test something on pebbled and
        # hallem cases w/o re-running many seeds before getting an answer on the test.
        if skip_models_with_seeds:
            old_len = len(model_kw_list)
            model_kw_list = [x for x in model_kw_list if 'n_seeds' not in x]

            n_skipped = old_len - len(model_kw_list)
            warn(f'currently skipping {n_skipped} models with seeds! (because '
                '`-s model-seeds` CLI option)'
            )

        for model_kws in model_kw_list:

            for target_sparsity in target_sparsities:
                _model_kws = dict(model_kws)
                _model_kws['target_sparsity'] = target_sparsity

                if panel == 'megamat':
                    _model_kws['repro_preprint_s1d'] = True

                do_sensitivity_analysis = False

                if model_kws.get('sensitivity_analysis', False):
                    # TODO delete (replace w/ checking list of panels to NOT run sens
                    # analysis on? e.g. validation)
                    #if (panel == panel_for_sensitivity_analysis and
                    #    target_sparsity == target_sparsity_for_sensitivity_analysis):

                    do_sensitivity_analysis = True

                if skip_sensitivity_analysis or not do_sensitivity_analysis:
                    try:
                        # assumes the default is False. could also set False explicitly,
                        # but not passing that in explicitly for other model_kws
                        # iterated over.
                        del _model_kws['sensitivity_analysis']
                    except KeyError:
                        pass

                if ('orn_deltas' in model_kws and
                    model_kws['orn_deltas'] is pebbled_input_df):

                    param_dir_prefix = pebbled_param_dir_prefix
                else:
                    # these cases should be all and only the hallem input data cases,
                    # where the only parameter in this prefix (the dF/F scaling before
                    # fitting) is not relevant.
                    param_dir_prefix = ''

                # TODO TODO is this loop working as expected? in run on kiwi+control
                # data, i feel like i've seen more progress bars than i expected...
                # (should be 2 * 3, no? i.e. {hemidraw, uniform} X {control-kiwi,
                # control, kiwi}?) none of the duplicate-save-within-run detection
                # seemed to trip tho...

                fixed_thr = None
                wAPLKC = None
                _extra_params = dict(extra_params)
                # checking for orn_deltas because we don't want to ever do this
                # pre-tuning for hallem data (where the ORN data isn't passed here, but
                # loaded inside fit_mb_model)
                if 'orn_deltas' in model_kws and panel in panel2tuning_panels:
                    tuning_panels = panel2tuning_panels[panel]
                    tuning_panels_str = tuning_panel_delim.join(tuning_panels)

                    tuning_panels_plot_dir = plot_dir / tuning_panels_str
                    makedirs(tuning_panels_plot_dir)

                    panel_mask = mean_est_df.columns.get_level_values('panel'
                        ).isin(tuning_panels)

                    if panel_mask.sum() == 0:
                        raise RuntimeError(f'no data from {tuning_panels=}!\n\nedit '
                            'panel2tuning_panels if you do not intended to tune '
                            f'{panel=} data on these panels.\n\nyou may also just need '
                            'to change script CLI args to include this data.'
                        )

                    tuning_df = mean_est_df.loc[:, panel_mask]

                    tuning_model_kws = {k: v for k, v in _model_kws.items()
                        if k not in (
                            'orn_deltas', 'comparison_kc_corrs', 'comparison_orns'
                        )
                    }

                    # NOTE: if i wanted to do this pre-tuning on hallem data (which is
                    # loaded in fit_mb_model if orn_deltas not passed here), i'd need to
                    # not pass this. no real need to use this on hallem data tho.
                    #
                    # TODO (delete) need to drop panel level on tuning_df first
                    # (doesn't seem so...)? (if so, prob also want to prefix panel
                    # to odor names, or otherwise ensure no dupes?)
                    tuning_model_kws['orn_deltas'] = tuning_df

                    param_dict = fit_and_plot_mb_model(tuning_panels_plot_dir,
                        param_dir_prefix=param_dir_prefix,
                        extra_params=extra_params,
                        # TODO maybe set False for 'megamat' (if also tuning on
                        # 'megamat'), so that we can compare those two directories more
                        # easily, and maybe leave that test code in?
                        _only_return_params=True,
                        **tuning_model_kws
                    )

                    fixed_thr = param_dict['fixed_thr']
                    wAPLKC = param_dict['wAPLKC']
                    assert fixed_thr is not None
                    assert wAPLKC is not None

                    try:
                        # should be a list if it has a len
                        len(fixed_thr)
                        # will only get here if model has multiple seeds
                        # (i.e. if fixed_thr is a list)
                        assert len(fixed_thr) == len(wAPLKC)

                        # NOTE: currently relying on the pre-tuning + actual modelling
                        # calls using the same sequences of seeds (which they do, b/c
                        # initial seed currently hardcoded, and i always increment
                        # following seeds by one from there), so that applying the
                        # sequence of inh params across the two makes sense

                    # to catch:
                    # TypeError: object of type 'int' has no len()
                    # TypeError: object of type 'float' has no len()
                    except TypeError:
                        pass

                    del _model_kws['target_sparsity']
                    _model_kws['title_prefix'] = f'tuning panels: {tuning_panels_str}\n'

                    _extra_params['tuning_panels'] = tuning_panels_str
                    _extra_params['tuning_output_dir'] = param_dict['output_dir']

                    param_dir_prefix = \
                        f'tuned-on_{tuning_panels_str}__{param_dir_prefix}'

                params_for_csv = fit_and_plot_mb_model(panel_plot_dir,
                    param_dir_prefix=param_dir_prefix, extra_params=_extra_params,
                    fixed_thr=fixed_thr, wAPLKC=wAPLKC, **_model_kws
                )

                # should only be the case if first_seed_only=True inside fit_and_plot...
                # (just used to regen model internal plots, which are made on first seed
                # for multi-seed runs. no downstream plots/caches are updated by
                # fit_and_plot... in that case).
                if params_for_csv is None:
                    continue

                if skip_models_with_seeds:
                    warn(f'not writing to {model_param_csv} (b/c '
                        'skip_models_with_seeds=True)!'
                    )
                    continue

                # should only be added if wAPLKC/fixed_thr passed, which should not
                # be the case in any of these calls
                assert 'pearson' not in params_for_csv

                if model_params is None:
                    model_params = pd.Series(params_for_csv).to_frame().T
                else:
                    # works (adding NaN) in both cases where appended row has
                    # more/less columns than existing data.
                    model_params = model_params.append(params_for_csv,
                        ignore_index=True
                    )

                # just doing in loop so if i interrupt early i still get this. don't
                # think i mind always overwritting this from past runs.
                #
                # NOTE: can't use wrapper here b/c it asserts output wasn't already
                # saved this run.
                model_params.to_csv(model_param_csv, index=False)

        if skip_models_with_seeds:
            warn('not making across-model plots (S1C/2E) (b/c '
                'skip_models_with_seeds=True)!'
            )
            continue

        if panel != 'megamat':
            # TODO (at least if verbose) warn we warn we are skipping rest?
            continue

        # NOTE: special casing handling of this plot. other plots dealing with errorbars
        # across seeds will NOT subset seeds to first 20 (using global
        # `n_first_seeds_for_errorbar = None` instead)
        fig2e_n_first_seeds = 20
        _, fig2e_seed_err_fname_suffix = _get_seed_err_text_and_fname_suffix(
            n_first_seeds=fig2e_n_first_seeds
        )

        remy_2e_corrs = load_remy_2e_corrs(panel_plot_dir)

        # don't actually care about output data here, but it will save extra a plot
        # showing we can recreate the preprint fig 2E when use_preprint_data=True
        load_remy_2e_corrs(panel_plot_dir, use_preprint_data=True)

        # should already be sorted by mean-pair-correlation in load_remy_2e_corrs,
        # with all entries of each pair grouped together
        remy_2e_pair_order = remy_2e_corrs.odor_pair_str.unique()

        remy_2e_odors = (
            set(remy_2e_corrs.abbrev_row) | set(remy_2e_corrs.abbrev_col)
        )

        remy_pairs = set(list(zip(
            remy_2e_corrs.abbrev_row, remy_2e_corrs.abbrev_col
        )))


        # TODO refactor to share w/ load fn? delete in one or the other?
        pal = sns.color_palette()
        # green: hemibrain, orange: uniform, blue: hemidraw, black: observed
        label2color = {
            # green (but not 'green' exactly)
            'hemibrain': pal[2],
            # orange (but not 'orange' exactly)
            'uniform': pal[0],
            # blue (but not 'blue' exactly)
            'hemidraw': pal[1],
        }
        #

        # TODO try to make error bars only shown outside hollow circles?

        assert not model_params.output_dir.duplicated().any()

        # (fails if first_seed_only=True in fit_and_plot..., but that's only for
        # manual regeneration of fit_mb_model's model_internals/ plots, and should
        # never stay True)
        assert len(model_kw_list) == len(model_params)
        pebbled_mask = np.array(
            [x.get('orn_deltas') is pebbled_input_df for x in model_kw_list]
        )

        pn2kc_order = [
            'hemidraw',
            'uniform',
            'hemibrain',
        ]
        def _sort_pn2kc(x):
            if x in pn2kc_order:
                return pn2kc_order.index(x)
            else:
                return float('inf')

        # TODO i assume these are all in hallem?
        # NOTE: none of these are in Remy's validation2 panel (so I don't have them
        # in any of my pebbled data, as they also aren't in megamat, which is the
        # only other panel of hers I collected)
        #
        # ones not in megamat 17:
        # - 1-penten-3-ol
        # - delta-decalactone
        # - ethyl cinnamate
        # - eugenol
        # - gamma-hexalactone
        # - methyl octanoate
        # - propyl acetate

        # intentionally not dropping any silent/bad cells here. always want all
        # cells included for these type of plots.
        remy_binary_responses = load_remy_megamat_kc_binary_responses()

        for desc, mask in (('pebbled', pebbled_mask), ('hallem', ~ pebbled_mask)):

            if mask.sum() == 0:
                warn(f'no {desc} data in current model runs. skipping 2E/S1C!')
                continue

            # one row per model run
            curr_model_params = model_params.loc[mask]

            curr_model_params = curr_model_params.sort_values('pn2kc_connections',
                kind='stable', key=lambda x: x.map(_sort_pn2kc)
            )

            # since we'll use this for line labels (e.g. 'hemibrain', 'uniform')
            assert not curr_model_params.pn2kc_connections.duplicated().any()

            # e.g. 'hemibrain' -> DataFrame (Series?) with hemibrain model correlations
            pn_kc_cxn2model_corrs = dict()

            # inside the loop, we also make another version that only shows the KC data
            # that also has model data
            remy_2e_facetgrid = _create_2e_plot_with_obs_kc_corrs(remy_2e_corrs,
                remy_2e_pair_order, fill_markers=False
            )

            s1c_fig, s1c_ax = plt.subplots()

            first_model_pairs = None
            remy_2e_modelsubset_facetgrid = None

            for i, row in enumerate(curr_model_params.itertuples()):
                output_dirname = row.output_dir
                output_dir = panel_plot_dir / output_dirname
                responses_cache = output_dir / model_responses_cache_name
                responses = pd.read_pickle(responses_cache)

                label = row.pn2kc_connections
                assert type(label) is str and label != ''

                color = label2color[label]

                responses.columns = responses.columns.map(olf.parse_odor_name)
                assert not responses.columns.isna().any()
                assert not responses.columns.duplicated().any()

                # at least for now, doing this here so that i don't need to re-run
                # model after abbrev_hallem_odor_index change (currently commented).
                # would also need to figure out how to deal w/ 'moct' if i wanted to
                # remove this (was thinking of changing 'MethOct' -> 'moct' in
                # load_remy_2e...)
                # thought I needed to do before corr_triangular, in order to get same
                # order as remy has for all the pairs, but moving this here didn't fix
                # all of that issue.
                my2remy_odor_names = {
                    'eugenol': 'eug',
                    'ethyl cinnamate': 'ECin',
                    'propyl acetate': 'PropAc',
                    'g-hexalactone': 'g-6lac',
                    'd-decalactone': 'd-dlac',
                    'moct': 'MethOct',
                    # I already had an abbreviation for the 7th
                    # ('1-penten-3-ol' -> '1p3ol'), which is consistent w/ hers.
                }
                ordered_pairs = None
                # thought I needed to do before corr_triangular, in order to get same
                # order as remy has for all the pairs, but moving this here didn't fix
                # all of that issue. still doing before corr_triangular, so my odor
                # names will line up with Remy's when I now pass in new ordered_pairs
                # kwarg to corr_triangular, which I added to manually fix this issue.
                if desc == 'hallem':
                    odor_strs = responses.columns

                    for old, new in my2remy_odor_names.items():
                        assert (odor_strs == new).sum() == 0
                        assert (odor_strs == old).sum() == 1
                        # TODO delete
                        #assert odor_strs.str.contains(f'{new} @').sum() == 0
                        #assert odor_strs.str.contains(f'{old} @').sum() == 1

                        odor_strs = odor_strs.str.replace(old, new)

                        # TODO delete
                        #assert odor_strs.str.contains(f'{new} @').sum() == 1
                        assert (odor_strs == new).sum() == 1

                    responses.columns = odor_strs

                    # any pairs (a, b) seen here will be used over any (b, a)
                    # corr_triangular would otherwise use. OK if not all pairs
                    # represented here (e.g. like how Remy's pairs are not all of the
                    # possible Hallem pairs, but this will at least make sure the
                    # overlap is consistent)
                    ordered_pairs = remy_pairs

                responses_including_silent = responses.copy()

                # TODO or factor corr calc + dropping into one fn, and call that in the
                # 3 places that currently use this?
                if drop_silent_model_kcs:
                    responses = drop_silent_model_cells(responses)

                # TODO refactor to combine dropping -> correlation [->mean across seeds]
                if 'seed' in responses.index.names:
                    # TODO refactor to share w/ internals of mean_of_fly_corrs?
                    # (use new square=False kwarg?)
                    corrs = responses.groupby(level='seed').apply(
                        lambda x: corr_triangular(x.corr(), ordered_pairs=ordered_pairs)
                    )
                    assert len(corrs) == n_seeds
                else:
                    corrs = corr_triangular(responses.corr(),
                        ordered_pairs=ordered_pairs
                    )
                    # so shape/type is same as in seed case above.
                    # name shouldn't be important.
                    corrs = corrs.to_frame(name='correlation').T

                del ordered_pairs

                # TODO where are NaN coming from in here?
                # ipdb> corrs.isna().sum().sum()
                # 34773
                # ipdb> corrs.size
                # 599500
                # ipdb> corrs.isna().sum().sum() / corrs.size
                # 0.05800333611342786

                # TODO is this weird? just some seeds have odors w/o cells
                # responding to them?
                #
                # ipdb> responses.shape
                # (163000, 110)
                # ipdb> corrs.shape[1]
                # 5995
                # ipdb> (corrs == 1).sum()
                # odor1              odor2
                # -aPine @ -2        -bCar @ -2          0
                # ...
                # t2h @ -2           terpinolene @ -2    0
                #                    va @ -2             0
                # terpinolene @ -2   va @ -2             0
                # Length: 5995, dtype: int64
                # ipdb> (corrs == 1).sum().sum()
                # 63
                # ipdb> np.isclose(corrs, 1).sum().sum()
                # 63

                pairs = corrs.columns.to_frame(index=False)
                # choosing 2 means none of the pairs are from the diagonal of the
                # correlation matrix (no identity elements. no correlations of odors
                # with themselves.)
                assert not (pairs.odor1 == pairs.odor2).any()

                assert not pairs.duplicated().any()

                # TODO delete. now doing this on responses.columns above
                '''
                # removing the concentration part of each odor str, e.g.
                # 'a @ -3' -> 'a' (since Remy and I format that part slightly diff)
                pairs = pairs.applymap(olf.parse_odor_name)
                # if any odor is presented at >1 conc, this 1st assertion would trip
                assert not pairs.duplicated().any()
                assert not pairs.isna().any().any()
                '''

                # TODO delete. doing before corr_triangular now.
                #pairs = pairs.replace(my2remy_odor_names)

                corrs.columns = pd.MultiIndex.from_frame(pairs)

                model_odors = set(pairs.odor1) | set(pairs.odor2)

                model_pairs = set(list(zip(pairs.odor1, pairs.odor2)))

                # we only ever have one representation of a given pair, and it's
                # always the same one across remy_pairs and model_pairs
                assert not any(
                    (b, a) in model_pairs or (b, a) in remy_pairs
                    for a, b in remy_pairs | model_pairs
                )

                n_odors = responses.shape[1]
                assert corrs.shape[1] == n_choose_2(n_odors)

                if desc == 'hallem':
                    assert n_odors == 110
                    # NOTE: unlike in pebbled cases below, we do sometimes have some
                    # odors without cells responding to them in here, and thus some
                    # NaN correlations

                    assert len(remy_2e_odors - model_odors) == 0

                    # TODO any other assertions in here? maybe something to complement
                    # currently-failing one above? (re: (a,b) vs (b,a))
                    # or will renaming those few odors fix that?
                    #import ipdb; ipdb.set_trace()

                # true as long as we don't also want to use this to plot
                # megamat+validation2 data (or validation2 alone)
                # (currently this code all only runs for megamat panel)
                elif desc == 'pebbled':
                    assert n_odors == n_megamat_odors == len(model_odors)

                    # might also not be true in cases other than megamat
                    assert not corrs.isna().any().any()

                    assert len(model_odors - remy_2e_odors) == 0

                    remy_2e_odors_not_in_model = remy_2e_odors - model_odors
                    if i == 0 and len(remy_2e_odors_not_in_model) > 0:
                        # we are already checking model_odors doesn't change across
                        # iterations of this inner loop, so it's OK to only warn on
                        # first iteration.
                        warn(f'Remy 2e odors not in current ({desc}) model outputs: '
                            f'{remy_2e_odors_not_in_model}'
                        )
                    #

                    # TODO also want something like this in desc='hallem' case?
                    #
                    # seems Remy and I are constructing our pairs in the same way
                    # (so that I don't need to re-construct one or the other to make
                    # sure we never have (a, b) in one and (b, a) in the other)
                    assert len(model_pairs - remy_pairs) == 0

                    # all the other pairs Remy has include at least one non-megamat
                    # odor
                    assert not any([
                        (a in megamat_odor_names) and (b in megamat_odor_names)
                        for a, b in remy_pairs - model_pairs
                    ])


                if i == 0:
                    assert first_model_pairs is None
                    first_model_pairs = model_pairs

                    if desc != 'hallem':
                        remy_2e_corrs_in_model_mask = remy_2e_corrs.apply(lambda x:
                            (x.abbrev_row, x.abbrev_col) in model_pairs, axis=1
                        )
                        # TODO reset_index(drop=True)? prob no real effect on plots...
                        remy_2e_corrs_in_model = remy_2e_corrs[
                            remy_2e_corrs_in_model_mask
                        ]

                        assert 0 == len(
                            # in pebbled+megamat case, the two sets should also be
                            # equal.  in hallem case, model_pairs will have many pairs
                            # not in what Remy gave me (but all of Remy's pairs should
                            # have both odors in Hallem).
                            set(list(zip(
                                remy_2e_corrs_in_model.abbrev_row,
                                remy_2e_corrs_in_model.abbrev_col
                            ))) - model_pairs
                        )

                        # unlike pair sets, elements here are str (e.g. 'a, b')
                        remy_2e_pair_order_in_model = np.array([
                            x for x in remy_2e_pair_order
                            if tuple(x.split(', ')) in model_pairs
                        ])
                        assert (
                            set(remy_2e_corrs_in_model.odor_pair_str) ==
                            set(remy_2e_pair_order_in_model)
                        )

                        # we also make a version of this where we show all KC pairs (and
                        # only model data when we can) before this loop.
                        remy_2e_modelsubset_facetgrid = \
                            _create_2e_plot_with_obs_kc_corrs(
                                remy_2e_corrs_in_model, remy_2e_pair_order_in_model,
                                fill_markers=False
                        )
                else:
                    assert first_model_pairs is not None
                    # checking each iteration of this loop would be plotting the same
                    # subset of data
                    assert first_model_pairs == model_pairs

                corrs.columns.names = ['abbrev_row', 'abbrev_col']

                corr_dists = 1 - corrs

                # ignore_index=False so index (one 'seed' level only) is preserved,
                # so error can be computed across seeds for plot
                corr_dists = corr_dists.melt(ignore_index=False,
                    value_name='correlation_distance').reset_index()

                assert label not in pn_kc_cxn2model_corrs
                # label is str describing pn2kc connections (e.g. 'hemibrain')
                pn_kc_cxn2model_corrs[label] = corrs

                corr_dists['odor_pair_str'] = (
                    corr_dists.abbrev_row + ', ' + corr_dists.abbrev_col
                )

                _2e_plot_model_corrs(remy_2e_facetgrid, corr_dists,
                    remy_2e_pair_order, color=color, label=label,
                    n_first_seeds=fig2e_n_first_seeds
                )

                if desc != 'hallem':
                    assert remy_2e_modelsubset_facetgrid is not None
                    _2e_plot_model_corrs(remy_2e_modelsubset_facetgrid, corr_dists,
                        remy_2e_pair_order_in_model, color=color, label=label,
                        n_first_seeds=fig2e_n_first_seeds
                    )

                # TODO why does the hemibrain line on this seem more like ~0.6 than
                # the ~0.5 in preprint? matter (remy wasn't concerned enough to
                # track down which outputs she originally made plot from)?
                # TODO also, why does tail seem different in pebbled plot? meaningful?
                if desc == 'hallem':
                    responses_including_silent = responses_including_silent.loc[:,
                        # TODO delete (or revert, if plot_n_odors_per_cell doesn't work
                        # w/ concs stripped from responses...)
                        #responses_including_silent.columns.map(odor_is_megamat)
                        #
                        # responses.columns now have concentrations stripped, so
                        # checking this way rather than .map(odor_is_megamat)
                        responses_including_silent.columns.isin(megamat_odor_names)
                    ]

                assert len(responses_including_silent.columns) == n_megamat_odors
                plot_n_odors_per_cell(responses_including_silent, s1c_ax, label=label,
                    color=color
                )


            _finish_remy_2e_plot(remy_2e_facetgrid, n_first_seeds=fig2e_n_first_seeds)

            if desc != 'hallem':
                # TODO delete
                assert all(
                    '__data_pebbled__' in x.name or x.name == 'megamat'
                    for x, _, _ in _spear_inputs2dfs.keys()
                )

                mc_key = (
                    Path('pebbled_6f/pdf/ijroi/mb_modeling/megamat'),
                    'mean_kc_corr',
                    'mean_orn_corr'
                )
                assert _spear_inputs2dfs[mc_key].equals(merged_corrs)
                del _spear_inputs2dfs[mc_key]

                assert all(
                    (x.endswith('_dist') and y.endswith('_dist')) or
                    not (x.endswith('_dist') or y.endswith('_dist'))
                    for _, x, y in _spear_inputs2dfs.keys()
                )

                len_before = len(_spear_inputs2dfs)
                # would raise error if search didn't work on one (hence del above).
                # replacing Path objects with the relevant str part of their name,
                # for easier accessing.
                _spear_inputs2dfs = {
                    (re.search('pn2kc_([^_]*)__', p.name).group(1), x, y): df
                    # TODO delete. why wasn't this working for uniform/hemidraw
                    # (included extra past '__')?
                    #(re.search('pn2kc_(.*)__', p.name).group(1), x, y): df
                    for (p, x, y), df in _spear_inputs2dfs.items()
                }
                # checking we didn't map any 2 keys before to 1 key now
                assert len(_spear_inputs2dfs) == len_before

                assert orn_col == 'mean_orn_corr'

                model_corrs = []
                prev_model_corr = None
                for (pn2kc, x, y), odf in _spear_inputs2dfs.items():

                    if odf.index.names != ['odor1','odor2']:
                        odf = odf.set_index(['odor1','odor2'], verify_integrity=True)

                    if y == 'orn_corr':
                        s1 = merged_corrs[orn_col]
                        model_corr = odf['model_corr'].copy()
                        model_corr.name = f'{pn2kc}_corr'
                        model_corrs.append(model_corr)
                    else:
                        assert y == 'observed_kc_corr_dist'
                        # converting to correlation DISTANCE, to match `y` here
                        s1 = 1 - merged_corrs[kc_col]

                        model_corr = 1 - odf['model_corr_dist']

                        # these y == 'observed_kc_corr_dist' entries should always
                        # follow a y == 'orn_corr' entry with the same pn2kc value
                        assert prev_model_corr is not None
                        assert pd_allclose(model_corr, prev_model_corr)

                    prev_model_corr = model_corr

                    s2 = odf[y]
                    assert pd_allclose(s1, s2)

                merged_corrs = pd.concat([merged_corrs] + model_corrs, axis='columns',
                    verify_integrity=True
                )

                index_no_concs = merged_corrs.index.map(
                    # takes 2-tuples of ['odor1','odor2'] strs and strips concs
                    lambda x: (x[0].split(' @ ')[0], x[1].split(' @ ')[0])
                )
                assert all(
                    x.columns.equals(index_no_concs)
                    for x in pn_kc_cxn2model_corrs.values()
                )

                model_corrs2 = []
                for pn2kc, corrs in pn_kc_cxn2model_corrs.items():
                    # each `corrs` should be of shape (1|n_seeds, n_odors_choose_2)
                    if len(corrs) > 1:
                        assert corrs.index.name == 'seed'

                    mean_corrs = corrs.mean()
                    mean_corrs.name = f'{pn2kc}_corr'
                    model_corrs2.append(mean_corrs)

                model_corrs2 = pd.concat(model_corrs2, axis='columns',
                    verify_integrity=True
                )
                assert model_corrs2.index.equals(index_no_concs)
                model_corrs2.index = merged_corrs.index

                model_corrs1 = merged_corrs.iloc[:, 2:]
                assert set(model_corrs2.columns) == set(model_corrs1.columns)
                model_corrs2 = model_corrs2.loc[:, model_corrs1.columns]

                # TODO replace all model_corrs1 code w/ model_corrs2? (-> delete _spear*
                # global / etc)? (assertion below passing, so they are equiv now)
                #
                # no NaN in either, else we would want equal_nan=True
                assert pd_allclose(model_corrs1, model_corrs2)

                # checking nothing looks like a correlation DISTANCE (range [0, 2])
                #
                # the .[min|max]() calls returns series w/ index the 2 real (ORN, KC)
                # corrs, and the 3 model corrs, w/ the min|max for each, so we are
                # checking that each corr column has expected range.
                assert (merged_corrs.min() < 0).all()
                assert (merged_corrs.max() < 1).all()

                col_pairs = list(itertools.combinations(merged_corrs.columns, 2))
                # TODO names matter (for invert*)? omit?
                index = pd.MultiIndex.from_tuples(col_pairs, names=['c1', 'c2'])

                # TODO 95% instead?
                ci = 90
                ci_str = f'{ci:.0f}% CI'
                # TODO TODO do something other than average over 100 seeds before then
                # ignoring the 

                # TODO keep pearson?
                # TODO also try to get euclidean (below) to work here? prob just wanna
                # delete it...
                for method in ('spearman', 'pearson'):
                    corr_of_pearsons = merged_corrs.corr(method=method)
                    xlabel = f"{method.title()}s-of-Pearsons"
                    plot_corr(corr_of_pearsons, panel_plot_dir, f'{method}_of_pearsons',
                        overlay_values=True, xlabel=xlabel
                    )

                    import ipdb; ipdb.set_trace()
                    corrs = []
                    lower_cis = []
                    upper_cis = []
                    for x, y in col_pairs:
                        _, corr, ci_lower, ci_upper, _, = bootstrapped_corr(
                            merged_corrs, x, y, method=method, ci=ci
                        )
                        corrs.append(corr)
                        lower_cis.append(ci_lower)
                        upper_cis.append(ci_upper)

                    corr_df = pd.DataFrame(
                        {'corr': corrs, 'lower': lower_cis, 'upper': upper_cis},
                        index=index
                    )

                    square_corr = invert_corr_triangular(corr_df['corr'], name=None)
                    assert pd_allclose(square_corr, corr_of_pearsons)

                    # TODO are CI's symmetric (no) (i.e. can i get one measure of error
                    # for each matrix element, or will it make more sense to have 2
                    # extra matrix plots, one for lower CI and one for upper CI?)
                    square_lower = invert_corr_triangular(corr_df['lower'], name=None)
                    square_upper = invert_corr_triangular(corr_df['upper'], name=None)

                    plot_corr(square_lower, panel_plot_dir,
                        f'{method}_of_pearsons_lower', overlay_values=True,
                        xlabel=f'{xlabel}\nlower side of {ci_str}'
                    )
                    plot_corr(square_upper, panel_plot_dir,
                        f'{method}_of_pearsons_upper', overlay_values=True,
                        xlabel=f'{xlabel}\nupper side of {ci_str}'
                    )

                # TODO want to try plotting any other distances/metrics/norms?
                #
                # metric can be anything scipy pdist takes
                euclidean_of_pearsons = frame_pdist(merged_corrs, metric='euclidean')
                # TODO delete. don't want to support non-corr vmin/vmax, which euclidean
                # would need.
                #plot_corr(euclidean_of_pearsons, panel_plot_dir,
                #    'euclidean_of_pearsons', overlay_values=True
                #)
                fig, _ = viz.matshow(euclidean_of_pearsons, overlay_values=True,
                    # TODO or just use red half of the diverging cmap, to try to keep
                    # things more comparable?
                    xlabel='Euclidean-of-Pearsons', cmap=cmap
                )
                savefig(fig, panel_plot_dir, 'euclidean_of_pearsons')

                # TODO TODO save CSV of pearson ranks (sorted by observed KC ranks?)
                # (for everything in merged_corrs, maybe making one new rank col for
                # each existing, to have side by side? or two CSVs, one like this and
                # one w/ just ranks?)

                assert remy_2e_modelsubset_facetgrid is not None
                _finish_remy_2e_plot(remy_2e_modelsubset_facetgrid,
                    n_first_seeds=fig2e_n_first_seeds
                )

            # seed_errorbar is used internally by plot_n_odors_per_cell
            savefig(remy_2e_facetgrid, panel_plot_dir,
                f'2e_{desc}{fig2e_seed_err_fname_suffix}'
            )

            # model subset same in this case
            if desc != 'hallem':
                savefig(remy_2e_modelsubset_facetgrid, panel_plot_dir,
                    f'2e_{desc}_model-subset{fig2e_seed_err_fname_suffix}'
                )

            # TODO double check error bars are 95% ci. some reason matt's are so much
            # larger? previous remy data really much more noisy here?
            # (pretty sure current errorbars are right. not sure if old ones were, or
            # what spread was like there)
            plot_n_odors_per_cell(remy_binary_responses, s1c_ax, label='observed',
                color='black'
            )

            s1c_ax.legend()
            # errorbars are really small for model here, and can barely see CI's get
            # bigger increasing from 95 to 99
            s1c_ax.set_title(f'model run on {desc}\n{seed_err_text}')

            savefig(s1c_fig, panel_plot_dir,
                f's1c_n_odors_vs_cell_frac_comparison_{desc}'
            )


# TODO also return indicator (to print in all_corr..._plots) (or compute there?)
def most_recent_GH146_output_dir():
    # cwd is where output dirs should be created (see driver_indicator_output_dir)
    #
    # TODO refactor to use something like output_dir2driver (but for indicator), and if
    # it fails exclude those (instead of manually checking # parts here too)
    dirs = [x for x in Path.cwd().glob('GH146_*/') if x.name.count('_') == 1]
    # TODO warn if it's not 'GH146_6f'?
    return sorted(dirs, key=util.most_recent_contained_file_mtime)[-1]


_gh146_glomeruli = None
def get_gh146_glomeruli() -> Optional[Set[str]]:
    """Returns set of glomerulus names that are in >= 1/2 of GH146 flies.

    The GH146 flies considered should be the 7 final megamat flies for the paper with
    Remy.
    """
    global _gh146_glomeruli

    if _gh146_glomeruli is not None:
        return _gh146_glomeruli

    gh146_output_dir = most_recent_GH146_output_dir()

    # TODO factor out? hong2p.util even?
    def _fmt_rel_path(p):
        return f'./{p.relative_to(Path.cwd())}'

    gh146_output = gh146_output_dir / ij_roi_responses_cache

    # TODO don't hardcode plot path here (also switch '_certain_' substr on certain
    # flag) (?)
    print(f'loading GH146 glomeruli from {_fmt_rel_path(gh146_output)}, to subset'
        ' current glomeruli'
    )

    if not gh146_output.exists():
        # TODO TODO err instead (and update type hint return)
        warn(f'no GH146 data found at {_fmt_rel_path(gh146_output_dir)}. can not '
            'generate correlation plots restricted to GH146 glomeruli!'
        )
        return None

    gh146_df = pd.read_pickle(gh146_output)

    # no validation2 panel flies for GH146. just the 7 flies each w/ diagnostic+megamat
    assert gh146_df.groupby(['date', 'fly_num'], axis='columns').ngroups == 7, \
        'GH146 data did not have expected 7 flies (# in final megamat data)'

    assert set(gh146_df.index.get_level_values('panel')) == {
        'megamat', diag_panel_str
    }

    certain_gh146_df = select_certain_rois(gh146_df)

    gh146_glom_counts = certain_gh146_df.groupby(level='roi', axis='columns'
        ).size()
    gh146_glom_counts.index.name = 'glomerulus'
    gh146_glom_counts.name = 'n_flies'

    n_flies = certain_gh146_df.groupby(level=['date', 'fly_num'], axis='columns'
        ).ngroups

    # TODO refactor to share dropping with subsetting in gh146 plot making?
    reliable_gh146_gloms = gh146_glom_counts >= (n_flies / 2)
    if (~ reliable_gh146_gloms).any():
        # TODO sanity check this set
        warn(f'excluding GH146 glomeruli seen in <1/2 of {n_flies} GH146 '
            f'flies:\n{gh146_glom_counts[~reliable_gh146_gloms].to_string()}'
        )

    # TODO warn instead?
    assert reliable_gh146_gloms.sum() > 0, ('no glomeruli confidently '
        f'identified in >=1/2 of total {n_flies} GH146 flies!'
    )
    considered_gh146_col = 'count_as_GH146'
    reliable_gh146_gloms.name = considered_gh146_col

    for_csv = pd.DataFrame([gh146_glom_counts, reliable_gh146_gloms]).T

    # DataFrame constructor seems to only accept a single dtype, so even though
    # reliable_gh146_gloms was already bool in constructor, it gets cast to int
    for_csv[considered_gh146_col] = for_csv[considered_gh146_col].astype(bool)

    for_csv = for_csv.sort_values('n_flies', kind='stable', ascending=False)

    to_csv(for_csv, gh146_output_dir / 'GH146_consensus_glomeruli.csv')

    # TODO warn if any of the GH146 ones not seen in pebbled data
    # (would need to find + load that here, or do outside of this fn)

    gh146_glomeruli = set(reliable_gh146_gloms[reliable_gh146_gloms].index)

    # future calls will just return the value in this variable
    _gh146_glomeruli = gh146_glomeruli

    return gh146_glomeruli


def acrossfly_correlation_plots(output_root: Path, trial_df: pd.DataFrame, *,
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
    # TODO TODO update all these plots to:
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


# TODO TODO (at least when plot format is svg) add metadata to matshow plots to allow
# showing glomerulus / odor on hover, as in:
# https://matplotlib.org/stable/gallery/user_interfaces/svg_tooltip_sgskip.html
# (requires loading svg in a browser, or at least something that actually runs the
# scripts, so not default ubuntu image viewer)
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

    # NOTE: n_per_odor_and_glom currently unused in rest of this fn. may want to
    # redefine n_flies (above) from it tho?
    fig, n_per_odor_and_glom = plot_n_per_odor_and_glom(certain_df, title_prefix=title)
    savefig(fig, plot_dir, f'{fname_prefix}certain_n')

    # TODO TODO should i be normalizing within fly or something before taking mean?
    # (maybe scaling to response to diagnostic panel? then remaking them in a timely
    # manner might be more important...)
    #
    # TODO TODO factor into option of plot_all_..., so that i can share code to
    # show N for each ROI w/ case calling from before loop over panels
    mean_certain_df = certain_df.groupby('roi', sort=False, axis='columns').mean()

    # TODO delete (what was this even for?)
    '''
    if plot_dir.name != 'ijroi' and fname_prefix == '':
        print(plot_dir.name)
        # taking mean across repeats before getting min/max (should corresond to entries
        # in mean plots)
        mdf = mean_certain_df.groupby('odor1').mean()
        print(f'{mdf.shape=}')
        # also averaging acrosss repeats
        print(f'{mdf.max().max()=}')
        print(f'{mdf.min().min()=}')
    '''
    # glomeruli_diagnostics (when run on validation2 data only)
    # mdf.shape=(25, 40)
    # mdf.max().max()=2.1187355870376936
    # mdf.min().min()=-0.3089174054102511
    #
    # validation2
    # mdf.shape=(22, 40)
    # mdf.max().max()=2.4357667244018546
    # mdf.min().min()=-0.2681457206864409
    #
    # glomeruli_diagnostics (when run on megamat data only)
    # mdf.shape=(26, 38)
    # mdf.max().max()=1.3093118527266558
    # mdf.min().min()=-0.1576990730421217
    #
    # megamat
    # mdf.shape=(17, 38)
    # mdf.max().max()=1.628468070216096
    # mdf.min().min()=-0.18102526894048887
    #

    # taking mean across repeats before getting min/max (should corresond to entries
    # in mean plots)
    mdf = mean_certain_df.groupby('odor1').mean()
    min_mean = mdf.min().min()
    max_mean = mdf.max().max()

    # want one range for all of these plots in paper, to more easily compare across
    # panels (though might still not make 100% sense w/o any kind of scaling per fly)
    vmin = -0.35
    vmax = 2.5
    if vmin <= min_mean and max_mean <= vmax:
        # TODO or calculate rather than hardcoding (prob hard...)?
        # TODO err if vmin/vmax very different from cbar limits / the former doesn't
        # contain latter?
        #cbar_ticks = [-0.3, 0, 2.5]
        # everything after -0.3 here is what would automatically be shown if cbar_kws
        # not passed below
        #cbar_ticks = [-0.3, 0, 0.5, 1.0, 1.5, 2.0, 2.5]
        # TODO TODO this work (or can i make it if i set float formatting right?)?
        # (yes, it does show same number of sig figs for all)
        # TODO TODO try to change formatter to fix
        cbar_ticks = [vmin, 0, 0.5, 1.0, 1.5, 2.0, vmax]
        cbar_kws = dict(ticks=cbar_ticks)
        # TODO delete (after stopping hardcode)
        print(f'response_matrix_plots: HARDCODING {vmin=} {vmax=}')
        print(f'response_matrix_plots: HARDCODING CBAR TICKS TO {cbar_ticks=}')
        #
    else:
        # TODO does it matter to have one consistent scale in this context? if so, may
        # still need to hardcode / similar...
        # TODO round up/down to nearest multiple of 0.5/0.25 or something, when using
        # dmin/dmax?
        # TODO just leave None?
        #vmin = min_mean
        #vmax = max_mean
        # (defaults)
        vmin = None
        vmax = None
        # the default
        cbar_kws = None

    # TODO refactor to not duplicate defs of so many kwargs across the two calls below
    # (just make another var here)

    # TODO just check # of columns to decide if we want to add bbox_inches='tight'
    # (between diags + megamat and that number + validation2)? or always pass it?
    # what is the downside to always passing it?
    plot_responses_and_scaled_versions(certain_df, plot_dir, f'{fname_prefix}certain',
        # NOTE: wasn't planning on passing vmin/vmax, but need to so stddev plot can
        # share limit w/ mean plots generated in next call
        #
        # TODO refactor stddev plotting to just another plot_all_... call here (rather
        # than lumping it into '*certain' call below), to make it easier to share
        # vmin/vmax w/ '*certain_mean' call w/o forcing individual fly data in
        # '*certain' onto same scale?
        vmin=vmin, vmax=vmax, title=title, cbar_kws=cbar_kws, bbox_inches='tight',

        # TODO only actually pass thru xticks_also_on_bottom for the plots that have
        # separate rows for ROIs from diff flies (e.g. NOT the stddev plot)
        # (currently have a hack inside plot_responses... that should exclude it in
        # most/all of those cases)
        xticks_also_on_bottom=True,

        # NOTE: currently ignoring these for stddev plot made within, and using
        # roimean_plot_kws there instead. hoping that hack is good enough for now.
        **roi_plot_kws
    )

    # TODO check and remove uncertainty from this comment...
    # I think this (plot_all...?) (now wrapped behind plot_responses_and_scaled...) is
    # sorting on output of the grouping fn (on ROI name), as I want.
    plot_responses_and_scaled_versions(mean_certain_df, plot_dir,
        f'{fname_prefix}certain_mean', vmin=vmin, vmax=vmax, title=title,
        cbar_kws=cbar_kws, bbox_inches='tight', **roimean_plot_kws
    )
    # TODO refactor so stddev is plotted here, at least to make text / other plot params
    # more consistent (or otherwise fix that font size consistency issue)?
    # TODO TODO or should i make + save mean in plot_responses_and_scaled_version (would
    # it be easier?)

    # TODO TODO normalized **w/in fly** versions too (instead of just per ROI)?
    # (copy scaling from my MB modeling code?) (+ use same glomeruli, fillna-ing where
    # appropriate, across all.)


def acrossfly_response_matrix_plots(trial_df, across_fly_ijroi_dir, driver, indicator
    ) -> None:

    # TODO relabel <date>/<fly_num> to one letter for both. write a text key to the same
    # directory
    print('saving across fly ImageJ ROI response matrices...', flush=True)

    # TODO add comment saying why we are doing this (/ delete)
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

    filled_trial_df = fill_to_hemibrain(trial_df)
    response_matrix_plots(across_fly_ijroi_dir, filled_trial_df, 'filled')
    del filled_trial_df

    # TODO cluster all uncertain ROIs? to see there are any things that could be added?

    # TODO and what would it take to include ROI positions in clustering again?
    # doesn't require picking a single global best plane, right? but might require
    # caching / concating / returning masks from process_recording?

    if diag_panel_str in trial_df.index.get_level_values('panel'):
        # TODO select in a way that doesn't rely on 'panel' level being in this
        # position?
        assert trial_df.index.names[0] == 'panel'

        # the extra square brackets prevent 'panel' level from being lost
        # (for concatenating with other subsets w/ different panels)
        # https://stackoverflow.com/questions/47886401
        diag_df = trial_df.loc[[diag_panel_str]]
        # TODO warn if diag_df is empty (after dropping some NaNs?)?
        diag_df = dropna(diag_df)
    else:
        diag_df = None


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

        # TODO move all after stuff we also wanna do w/ diag panel -> dedent + continue
        # if diag panel (before stuff we don't wanna do for diag panel alone)
        if diag_df is not None and panel != diag_panel_str:
            # NOTE: currently assuming all flies in current data have some diagnostic
            # panel data (in addition to current panel, presumably)
            #
            # join='inner' to drop ROIs (columns) that are only in one of the inputs
            # (e.g. when flies in diagnostics aren't in current panel. diag_df should
            # contain data from all panels)
            assert all(x in diag_df.columns for x in panel_df.columns)
            diag_and_panel_df = pd.concat([diag_df, panel_df], join='inner',
                verify_integrity=True
            )

            # TODO TODO still make a version not dropping these?
            if driver == 'GH146':
                # TODO refactor
                gh146_glomeruli = get_gh146_glomeruli()
                # (so plot ACTUALLY lines up with pebbled-subset-to-GH146 plot)_
                # TODO (why would it not, if we are doing this when driver is already
                # gh146? just b/c we drop down to "consensus" glomeruli in
                # get_gh146_glomeruli?)
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

            # NOTE: want diag_and_panel_df.loc[diag_panel_str] instead of diag_df
            # because diag_df also includes diagnostic data from flies that don't have
            # any data from this panel (presumably flies that have other panel(s) +
            # diags)
            filled_panel_diag_df = fill_to_hemibrain(
                diag_and_panel_df.loc[diag_panel_str]
            )
            response_matrix_plots(panel_ijroi_dir, filled_panel_diag_df, 'diags')
            del filled_panel_diag_df

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
                # TODO TODO TODO should i be making my own correlation plots here (w/o
                # diags)? or also hand off to remy?

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

                # TODO TODO TODO also warn (when merging ROIs) if strongest signal comes
                # from a plane marked uncertain (also w/ +?)

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

        # TODO only save this (and similar) if there are *some* uncertain, right?
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

        # TODO TODO just continue early if no uncertain stuff?
        if uncertain_rois.sum() > 0:
            glom_maxes = pdf.max(axis='rows')
            fig, _ = plot_all_roi_mean_responses(pdf.loc[:, uncertain_rois],
                # negative glom_maxes, so sort is as if ascending=False
                sort_rois_first_on=-glom_maxes[uncertain_rois], title=title,
                **uncertain_roi_plot_kws
            )
            savefig(fig, panel_ijroi_dir, 'uncertain_by_max_resp')
        else:
            # TODO warn instead?
            print('no uncertain ROIs. not generating uncertain_by_max_resp fig')

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
        # NOTE: I specifically need to use level= rather than by= or using positional
        # arg (to specifiy 'odor1' in all cases), or else it will be sorted even with
        # sort=False. https://github.com/pandas-dev/pandas/issues/17537
        mean_df = pdf.groupby(level='odor1', sort=False).mean()

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
            # not sorting so we can preserve sorted order (which uses a different
            # sorting function than cluster_rois does)
            cg = cluster_rois(clust_df, odor_sort=False, title=title,
                cmap=cmap
                # also not doing what i want. would need to fix viz.add_norm_options /
                # viz.clustermap cbar handling.
                #cmap=diverging_cmap
                #
                # TODO TODO fix clustermap wrapper to make cbar small in case of
                # two-slope norm (for side w/ smaller magnitude from vcenter)
                # (to make consistent w/ matshow behavior, which i think is handled by
                # add_colorbar there) (-> then restore these kwargs, instead of cmap)
                #**diverging_cmap_kwargs
            )
            savefig(cg, panel_ijroi_dir, 'with-uncertain_clust')
        except ValueError:
            traceback.print_exc()
            import ipdb; ipdb.set_trace()

        # TODO TODO TODO also cluster ROIs same way in ROI plots made within calls made
        # in process_recording (some that would be regen if `-i ijroi` passed)
        # TODO or otherwise, may want a fly-specific version her?

        # TODO TODO response matrix plots showing each trial for BEST PLANE ROI
        # (mainly to identify motion issues?)

        # <driver_indicator>/<plot_fmt>/ijroi/by_panel/<diag_panel_str>/each_fly
        each_fly_diag_response_dir = panel_ijroi_dir / 'each_fly'

        certain_df = select_certain_rois(panel_df)

        # TODO re-organize this + all other !diag stuff above, to simplify
        # TODO or just put this stuff in a loop over diag_df above panel_df loop?
        # (not doing for any other panels...)
        if panel == diag_panel_str:
            # only did CO2 in one fly, and not planning on routinely presenting it.
            diags_to_drop = ['CO2 @ 0']
            certain_df = certain_df.loc[
                ~ certain_df.index.get_level_values('odor1').isin(diags_to_drop), :
            ]

            # TODO calculate these once up top of this fn?
            glomeruli_with_diags = set(all_odor_str2target_glomeruli.values())
            odor_glom_combos_to_highlight = [
                dict(odor=o, glomerulus=g)
                for o, g in all_odor_str2target_glomeruli.items()
            ]

            # TODO define up top?
            def glom_has_diag(index_dict):
                glom = index_dict['roi']
                return glom in glomeruli_with_diags

            # TODO delete?
            '''
            # `x not in glomeruli_with_diags` puts those with diags first, as I want
            sort_rois_first_on = [x not in glomeruli_with_diags
                for x in certain_df.columns.get_level_values('roi')
            ]
            '''

            # TODO version that doesn't have the glomeruli w/o diags?

            # TODO TODO TODO update to handle one odor at two concs (or at least assert
            # / warn if an odor appears at two concs in diag panel). currently causing
            # problems w/ 2h appearing twice in new data (validation2 flies only, and
            # only last 2).
            #
            # TODO sort to group useful-to-compare odors/glomeruli next to each
            # other? not sure how...
            # TODO or order odors by how reliable of diagnostics they are?
            # (maybe just using strength of response in target?)
            # TODO or leave sorted same as i typically have odors ordered
            # (alphabetical)?
            #
            # should produce a nice, easy-to-follow, diagonal
            odors_sorted_by_target_glom = sorted(
                certain_df.index.get_level_values('odor1'),
                key=all_odor_str2target_glomeruli.get
            )
            # checking that .get always found key
            assert all(x is not None for x in odors_sorted_by_target_glom)

            names = [olf.parse_odor_name(o) for o in odors_sorted_by_target_glom]
            # TODO delete
            '''
            print(f'{len(set(odors_sorted_by_target_glom))=}')
            print(f'{len(set(names))=}')
            import ipdb; ipdb.set_trace()
            '''
            #
            # TODO delete
            # failing because aphe at -4 and -5. prob doesn't matter?
            #assert len(set(odors_sorted_by_target_glom)) == len(set(names))

            # list(set(names)) does NOT preserve order of names, so doing it this
            # way instead.
            name_order = []
            for n in names:
                if n not in name_order:
                    name_order.append(n)

            # TODO refactor to share w/ fly-specific (this copied from there)?

            certain_df = olf.sort_odors(certain_df, name_order=name_order)

            # TODO maybe use same dF/F range as in plot_rois invocations that
            # produce diagnostic examples (clipped)?

            # TODO TODO TODO do a certain_mean version of diag-highlight plot!!!
            # (still want?)
            # TODO (with a stddev version too)
            print('ADD CERTAIN_MEAN (+stddev) VERSION OF DIAG-HIGHLIGHT PLOT')

            # TODO TODO can i pass this as a fn, rather than a list (to make work w/
            # stddev inside plot_responses_and_scaled_versions more easily?)
            #
            # `x not in glomeruli_with_diags` puts those with diags first, as I want
            sort_rois_first_on = [x not in glomeruli_with_diags
                for x in certain_df.columns.get_level_values('roi')
            ]

            # TODO TODO save not in each_fly (unless i rename it to something fitting
            # both) (just save up one level) (maybe save both in one subdir though?)
            # TODO TODO restore roi label hline group text
            plot_responses_and_scaled_versions(certain_df, each_fly_diag_response_dir,
                f'diag-highlight_certain',
                # TODO TODO add a thicker hline between diag / non-diag stuff?
                # (prob still want hlines between ROIs in non-single-fly case...)
                # TODO could use hline_level_fn as below in stddev case internal to
                # plot_responses_and_scaled_versions tho...)
                sort_rois_first_on=sort_rois_first_on,
                odor_glomerulus_combos_to_highlight=odor_glom_combos_to_highlight,

                # TODO TODO fix sort_rois_first_on handling in stddev case ->
                # delete this option (which was added just to special case fix that)
                sort_glomeruli_with_diags_first=True,

                # TODO TODO TODO prob just fix... now getting:
                #  File "./al_analysis.py", line 2594, in plot_all_roi_mean_responses
                #    assert matching_roi.sum() == 1
                # TODO was there something that this didn't work with, that i care
                # about?
                allow_duplicate_labels=True,

                # TODO like (eh... i still think i find the roi scaled more useful)?
                # delete?
                #odor_scaled_version=True,

                # NOTE: this will no longer have only one white hline between diag'd and
                # non-diag'd glomeruli. there will be usual hline between each
                # glomerulus name
                **roi_plot_kws
            )

        for (date, fly_num), fly_df in certain_df.groupby(['date', 'fly_num'],
            axis='columns'):

            # TODO refactor to share w/ other places
            date_str = format_date(date)
            fly_str = f'{date_str}/{fly_num}'

            # drop any odors that are fully NaN (mainly to handle cases where
            # concentration changed, such as aphe -5 -> -4)
            fly_df = fly_df.dropna(how='all', axis='rows')

            # TODO clean up. just have two loops (one for diags, one for not, and
            # w/o this if/else?)
            if panel == diag_panel_str:
                fly_df = olf.sort_odors(fly_df, name_order=name_order)

                # `x not in glomeruli_with_diags` puts those with diags first, as I want
                sort_rois_first_on = [x not in glomeruli_with_diags
                    for x in fly_df.columns.get_level_values('roi')
                ]
                hline_level_fn = glom_has_diag
                odor_glom_combos_to_highlight = odor_glom_combos_to_highlight
                # TODO like (eh... i still think i find the roi scaled more useful)?
                # delete?
                odor_scaled_version = True

                _extra_kws = dict()
                # TODO diff fname_prefix (instead of f'{fly_str}_certain') here?
            else:
                sort_rois_first_on = None

                # TODO TODO or is there a default one i can leave it as (from
                # single_fly_roi_plot_kws?)?
                hline_level_fn = None

                odor_glom_combos_to_highlight = None
                odor_scaled_version = False

                # TODO TODO add vlines between odors (now that there are 3 trials)
                # TODO TODO vline group text for odor name + trial for xticklabel (or
                # don't show)
                def panel_odor_tuple(x):
                    return (format_panel(x), x.get('odor1'))

                # TODO fix need for allow_duplicate_labels=True?
                _extra_kws = dict(
                    avg_repeats=False,

                    # TODO remove this for the non-'_trials' version?
                    allow_duplicate_labels=True,
                    vline_level_fn=panel_odor_tuple,
                    vline_group_text=False,
                    # TODO test if there actually are multiple adjacent (or even
                    # non-adjacent?) odors w/ same panel. probably wouldn't work...
                    # (in which case, either just set False or maybe try to rework how
                    # level_fn/group_text stuff works?)
                    # TODO TODO maybe change vline_level_fn to accept an iterable of
                    # fns (using first or something for vline_group_text?)?
                    # and/or allow taking fn for vline_group_text, mapping level values
                    # to group text (and/or to tick labels too)?
                    group_ticklabels=True,
                )
                # TODO TODO TODO also do a version averaged across trials

                # TODO TODO fix how for this plot (and presumably others), '2h @ -5' and
                # '2h @ -5' + 'oct @ -3' do NOT have a vline between them (b/c 'odor1'
                # same, despite 'odor2' being diff). see how i changed plot_roi_util.py
                # vline_level_fn to new hong2p.olf.strip_concs_... fn.
                plot_responses_and_scaled_versions(fly_df, each_fly_diag_response_dir,
                    f'{fly_str}_certain_trials', title=f'{fly_str}\n({title})',
                    single_fly=True,

                    sort_rois_first_on=sort_rois_first_on,
                    hline_level_fn=hline_level_fn,
                    odor_glomerulus_combos_to_highlight=odor_glom_combos_to_highlight,

                    odor_scaled_version=odor_scaled_version,

                    # using this syntax with the goal of overwriting vline_level_fn in
                    # single_fly_roi_plot_kws
                    **{**single_fly_roi_plot_kws, **_extra_kws}
                )

            # TODO try to make sure that (for the non-diag plots at least, but probably
            # all?) all the glomeruli shown is the same on each (NaNing rows where
            # appropriate, when not found in certain flies) (to compare plots across
            # flies)
            # (can use fill_to_hemibrain now, which i'm already doing in some places i
            # want it)

            # TODO maybe use same dF/F range as in plot_rois invocations that
            # produce diagnostic examples (clipped)?
            plot_responses_and_scaled_versions(fly_df, each_fly_diag_response_dir,
                f'{fly_str}_certain', title=f'{fly_str}\n({title})',
                single_fly=True,

                sort_rois_first_on=sort_rois_first_on,
                hline_level_fn=hline_level_fn,
                odor_glomerulus_combos_to_highlight=odor_glom_combos_to_highlight,

                odor_scaled_version=odor_scaled_version,

                **single_fly_roi_plot_kws
            )

    print('done')


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
    global exit_after_saving_fig_containing
    global verbose
    global check_outputs_unchanged
    # TODO add other things modified like this
    global ij_trial_dfs

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
    # TODO extend skip CLI arg to work on some of the steps referenced by -i flag?
    # (was nonroi one of the more time consuming ones?)
    skippable_steps = ('intensity', 'corr', 'model', 'model-sensitivity',
        'model-seeds', 'ijroi'
    )
    # NOTE: model-sensitivity can only run as part of model step
    default_steps_to_skip = ('intensity', 'corr', 'model')
    # TODO add options for sensitivity_analysis + any MB model fitting w/ n_seeds>1?
    # + hallem fitting? or cache some of these (possibly adding to -i options)?
    parser.add_argument('-s', '--skip', nargs='?', const='',
        default=','.join(default_steps_to_skip),
        help='Comma separated list of steps to skip (default: %(default)s). '
        f'Elements must be in {skippable_steps}. '
        '-s with no following string skips NO steps.'
    )
    parser.add_argument('-r', '--retry-failed', action='store_true',
        help='Retry steps that previously failed (frame-to-odor assignment or suite2p).'
    )
    parser.add_argument('-F', '--exit-after-fig',
        dest='exit_after_saving_fig_containing',
        help='Exits after saving a figure whose path contains this string. For faster '
        'testing.'
    )
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Also prints paths to data that has some analysis skipped, with reasons '
        'for why things were skipped.'
    )

    # TODO TODO make something like these, but for panel (but always do diagnostics
    # anyway) (maybe -p/--panel or -e/--experiment). would have to handle diff than
    # driver/indicator, because need to get this information from data (not from
    # gsheet), and may also want to handle on something like a per-recording basis
    # TODO TODO and support including all flies w/ that panel, so as to not need to
    # manually specify date range for each panel? also allow comma sep, to analyze
    # mulitple together (diags should be implied, as long as flies contain one of other
    # panels)
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

    parser.add_argument('-A', '--force-across-fly', action='store_true',
        # TODO share 'matching_substrs' w/ above
        help='Forces across-fly analysis to be run, even if matching_substrs positional'
        'arg(s) restricts analysis to a subset of the data. No effect if '
        'matching_substrs is not passed.'
    )

    # TODO try to still link everything already generated (same w/ pairs)
    # TODO delete? still used?
    parser.add_argument('-g', '--glomeruli-diags-only', action='store_true',
        help='Only analyze glomeruli diagnostics (mainly for use on acquisition '
        'computer).'
    )

    # TODO option to warn but not err as well?
    # TODO warn in cases like sensitivity_analysis's deletion of it's root output folder
    # before starting (invalidating these checks...) (err if any folder would be
    # deleted when we have this flag?)
    # TODO TODO maybe this should prompt for pickles/csvs by default (w/ option to
    # approve single or all?)? maybe backup ones that would be replaced too?
    parser.add_argument('-c', '--check-outputs-unchanged', action='store_true',
        # TODO TODO specifically call out which formats this will/won't work for (png?)
        # work for PDF? implement some kind of image based diffing to support those?
        # TODO or maybe just err if this is passed with an unsupported plot format being
        # requested
        help='For CSVs/pickles/plots (saved with my to_[csv|pickle]/savefig wrappers), '
        'check new outputs against any existing outputs they would overwrite. If there '
        'is a descrepancy, exit with an error. Currently do not support certain plot '
        'formats (where metadata includes things like file creation time, so same '
        'strategy can not be used to check files for equality).'
    )

    args = parser.parse_args()

    matching_substrs = args.matching_substrs
    force_across_fly = args.force_across_fly

    parallel = args.parallel
    ignore_existing = args.ignore_existing
    steps_to_skip = args.skip
    retry_previously_failed = args.retry_failed
    analyze_glomeruli_diagnostics_only = args.glomeruli_diags_only

    driver = args.driver
    indicator = args.indicator

    start_date = args.start_date
    end_date = args.end_date

    exit_after_saving_fig_containing  = args.exit_after_saving_fig_containing

    # TODO maybe have this also apply to warnings about stuff skipped in
    # PREprocess_recording (now that i moved frame<->odor assignment fail handling code
    # there)
    verbose = args.verbose
    print_skipped = verbose

    check_outputs_unchanged = args.check_outputs_unchanged

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

    # TODO now it always warns, even if we are just using default! don't do that, or
    # delete this! (may need to change so no default in argparse itself? and probably
    # don't want that, as i like how that makes it easy to show default in -h/--help
    # output)
    '''
    if steps_to_skip == set(default_steps_to_skip):
        warn(f'manually specified skipping same as defaults{default_steps_to_skip}. '
            'you may omit -s/--skip <steps> option, to same effect.'
        )
    '''

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

    # TODO TODO update final concs stuff to work w/in each panel -> restore?
    # (use to NaN stuff from before changes in concentrations)
    # (or delete final concs stuff if i don't want to / can't easily update...)
    #
    # TODO delete this?
    # TODO TODO check names2final_concs stuff works with anything other than pairs
    # (don't think it does)
    # (code in process_recording is currently only run in pair case)
    # (and maybe delete code if not, or at least rename to indicate it's just for pair
    # stuff)
    #
    # NOTE: names2final_concs and seen_stimulus_yamls2thorimage_dirs are only used as
    # globals after this, not directly within main
    names2final_concs, seen_stimulus_yamls2thorimage_dirs, names_and_concs_tuples = \
        odor_names2final_concs(**common_paired_thor_dirs_kwargs)

    # NOTE: only intended to work for stuff where each odor presented once (at one
    # concentration) per recording, and each odor is presented alone.
    # TODO delete? actually need final conc start time?
    #panel2final_conc_dict, panel2final_conc_start_time = final_panel_concs(
    panel2final_conc_dict = final_panel_concs(**common_paired_thor_dirs_kwargs)

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

    if len(keys_and_paired_dirs) == 0:
        # TODO mention which run params relevant here (just start/end date and
        # matching_substr?)?
        raise IOError('no flies found with current run parameters')

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
            # TODO maybe this (and all below) should be some kind of IOError instead?
            # (to be consistent w/ above)
            # TODO TODO maybe this (and 2 errors in indicator branch below) should also
            # show which flies we did find [and we should have some, b/c
            # len(keys_and_paired_dirs) check above] (that presumably just aren't
            # labelled in sheet yet)? better error message
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


    # TODO -i option for these (could just check if remy_hallem_orns.png exists, and
    # assume others do if so)?
    # TODO or just separate script?
    # TODO delete? even want anymore?
    '''
    #
    # TODO TODO maybe this stuff should always be in a <plot_fmt> dir at root,
    # independent of driver? or maybe just at same level as this script, as before?
    #
    # add_sfr=True is the default, just including here for clarity.
    hallem_abs = orns.orns(columns='glomerulus', add_sfr=True)

    # TODO TODO version of this plot using deltas as input too?
    plot_remy_drosolf_corr(hallem_abs, 'hallem_orns', 'Hallem ORNs', plot_root,
        plot_responses=True
    )
    del hallem_abs

    # TODO do w/ kennedy PNs too!
    # TODO TODO does it even make sense to propagate up thru olsen model as deltas?  is
    # that actually what i'm doing now? maybe pns() should force add_sfr=True
    # internally?
    pn_df = pns.pns(columns='glomerulus', add_sfr=True)
    plot_remy_drosolf_corr(pn_df, 'olsen_pns', 'Olsen PNs', plot_root)
    del pn_df
    '''


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

    # TODO TODO TODO also save + concat an average baseline fluorescence tiff, for more
    # quickly making roi judgements on that basis
    #
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

        # TODO TODO do this in a loop separate from flies_with_new_processed_tiffs
        # (or in motion correction step, right after saving mocorr_concat_tiff)
        # (doesn't actually depend on any processed TIFFs! at least, none computed in
        # process_recording. just mocorr)
        #
        # TODO TODO TODO draw ROI example (version that has big images and shows names)
        # on this instead of average from just whichever recording
        # (would need to deal w/ ROIs though... as best plane still depends on
        # recording. doesn't matter if i don't show best planes only/differently tho...)
        #
        # NOTE: this one just depends on mocorr_concat.tif
        # (and only depends on mocorr, not choice of any response windows. all frames
        # are averaged, weighted equally.)
        mocorr_concat_tiff = fly_analysis_dir / mocorr_concat_tiff_basename
        all_frame_avg_tiff = fly_analysis_dir / all_frame_avg_tiff_basename
        if (not all_frame_avg_tiff.exists() or
            getmtime(mocorr_concat_tiff) > getmtime(all_frame_avg_tiff)):

            # TODO print we are reading this (could be slow)?
            mocorr_concat = tifffile.imread(mocorr_concat_tiff)

            # (t, z, y, x) -> (z, y, x)
            all_frame_avg = np.mean(mocorr_concat, axis=0)

            # TODO TODO update so these are saved in a manner considered "properly
            # volumetric" by my imagej_macros stuff (some of it currently complaining w/
            # above quote in error message). may also just need to fix some of the
            # image_macros stuff?
            util.write_tiff(all_frame_avg_tiff, all_frame_avg, strict_dtype=False)
            del mocorr_concat
        #

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

    if not force_across_fly and len(matching_substrs) > 0:
        # TODO just cprint yellow?
        # TODO share '-A' w/ CLI def via var
        warn('only processed a subset of recordings. exiting before across-fly '
            'analysis! (pass -A to run across-fly analysis anyway)'
        )
        sys.exit()

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

    ij_trial_dfs = olf.pad_odor_indices_to_max_components(ij_trial_dfs)
    trial_df = pd.concat(ij_trial_dfs, axis='columns', verify_integrity=True)

    assert len(ij_trial_dfs) == len(roi2best_plane_depth_list)

    roi_best_plane_depths = pd.concat(roi2best_plane_depth_list, axis='columns',
        verify_integrity=True
    )
    util.check_index_vals_unique(roi_best_plane_depths)

    # failing assertion (b/c depth defined for each recording, and not differentiated by
    # odor metadata along row axis, as w/ trial_df)
    #roi_best_plane_depths = merge_rois_across_recordings(roi_best_plane_depths)
    #
    # TODO TODO this what i want for merging depths across recordings? can i change to
    # not take average, and use depth for each specific recording? good enough?
    roi_best_plane_depths = roi_best_plane_depths.groupby(
        level=[x for x in roi_best_plane_depths.columns.names if x != 'thorimage_id'],
        sort=False, axis='columns'
    ).mean()
    util.check_index_vals_unique(roi_best_plane_depths)

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

    # to justify indexing it the same below
    # TODO maybe i should check more than just the roi level tho?
    assert trial_df.columns.get_level_values('roi').equals(
        roi_best_plane_depths.columns.get_level_values('roi')
    )

    # no need to copy, because indexing with a bool mask always does
    trial_df = trial_df.loc[:, ~contained_plus]
    roi_best_plane_depths = roi_best_plane_depths.loc[:, ~contained_plus]

    # TODO add comment explaining what this drops (+ doc in doc str for this fn)
    trial_df = drop_superfluous_uncertain_rois(trial_df)
    roi_best_plane_depths = drop_superfluous_uncertain_rois(roi_best_plane_depths)

    # TODO TODO should i always merge DL2d/v data? should be same exact receptors, etc?
    # and don't always have a good number of planes labelled for each, b/c often unclear
    # where boundary is

    # TODO test that moving this before merge_rois_across_recordings doesn't change
    # anything (because merge_... currently fails w/ duplicate odors [where one had been
    # repeated later, to overwrite former], and i'd like to use a similar strategy to
    # deal w/ those cases)
    # TODO err/warn if this nulls all stuff for those odors (would have indicated i ran
    # things wrong the one time i mixed up the -t/-e (start vs end) date options, and
    # ended up only using data from the affected flies)
    trial_df = setnull_old_wrong_solvent_aa_and_va_data(trial_df)

    trial_df = setnull_old_wrong_solvent_1p3one_data(trial_df)

    # TODO TODO update final_panel_concs + setnull_nonfinal* to also check solvent (+
    # probably return final solvent for each)
    # TODO and maybe replace some of above (would NEED to edit the YAMLs to have solvent
    # tag always accurately reflect which solvent i was using, as sometimes i was slow
    # to update that)
    trial_df = setnull_nonfinal_panel_odors(panel2final_conc_dict, trial_df)

    # TODO TODO NaN all (fly, odor) combos w/ self correlation not > some
    # threshold?

    # TODO NaN all (fly, odor, ROI) combos w/ self correlation not > some threshold?

    trial_df = sort_fly_roi_cols(trial_df, flies_first=True)
    trial_df = sort_odors(trial_df)

    roi_best_plane_depths = sort_fly_roi_cols(roi_best_plane_depths, flies_first=True)

    # TODO in a global cache that is saved (for use in real time analysis when drawing
    # ROIs. not currently implemented anymore), probably only update it to REMOVE flies
    # if they become marked exclude in google sheet (and otherwise just filter data to
    # compare against during that realtime analysis)

    certain_df = select_certain_rois(trial_df)

    certain_df = add_fly_id(certain_df.T, letter=True).set_index('fly_id', append=True
        ).reorder_levels(['fly_id'] + certain_df.columns.names).T

    all_fly_id_cols = ['fly_id', 'date', 'fly_num']
    fly_id_legend = index_uniq(certain_df.columns, all_fly_id_cols)

    if verbose:
        n_flies = len(fly_id_legend)
        print()
        print(f'{n_flies} flies:')
        print_uniq(fly_id_legend, all_fly_id_cols)

    # (note: same issue in comments here also affects consensus_df CSV/pickle saved
    # below)
    #
    # TODO put panel in this so switching between megamat/validation2 al_analysis runs
    # (e.g. `-t 2023-04-23 -e 2023-06-22` vs `-t 2023-11-19 -e 2024-01-05`)
    # doesn't trigger -c on this
    # TODO TODO or don't save if not run on all flies (as should also be true for some
    # of the other main outputs, e.g. CSVs)
    # TODO TODO or maybe just include start/end date in it? defeat the point?
    # am i giving this CSV to anyone now? what do i actually want to use it for?
    # TODO TODO TODO are these fly_ids not deleted before any real use? is it just used
    # for mean_df/etc? move this saving there then?
    to_csv(fly_id_legend, output_root / 'fly_ids.csv', date_format=date_fmt_str,
        index=False
    )
    id2datenum = fly_id_legend.set_index('fly_id', drop=True)


    # if False, only non-consensus *glomeruli* will be dropped, and all odors will
    # remain, in both certain_df/consensus_df
    do_drop_nonconsensus_odors = False

    mean_df_list = []
    stddev_df_list = []
    n_per_odor_and_glom_list = []

    dropped_fly_rois = set()

    consensus_dfs = []
    for panel, panel_df in certain_df.groupby(level='panel', sort=False):

        if verbose:
            print()
            print(f'{panel=}')
            print()

        panel_df = panel_df.dropna(axis='columns', how='all')

        # TODO delete if this doesn't end up being more useful than panel_df? or maybe
        # delete panel_df? (or delete panel_df now? pretty sure i wanna keep this)
        panel_and_diag_df = certain_df.loc[
            certain_df.index.get_level_values('panel').isin((diag_panel_str, panel))
        ].copy()

        panel_and_diag_df = panel_and_diag_df.dropna(axis='columns', how='all')

        # TODO worth comparing to panel_and_diag_df after dropping stuff where panel is
        # all NaN? probably not.
        #
        # panel_df.columns has levels ['fly_id', 'date', 'fly_num', 'roi'] here.
        # this is selecting the flies (+ their ROIs) that have some data in the current
        # panel, but still have the rows with panel data as well as (for these
        # particular flies) any diagnostic data.
        panel_and_diag_df = panel_and_diag_df[panel_df.columns].copy()
        del panel_df

        # TODO at least if verbose, print n_flies for each panel
        n_flies = len(
            panel_and_diag_df.columns.to_frame(index=False)[['date','fly_num']
                ].drop_duplicates()
        )
        # >= half of flies (the flies w/ any certain ROIs, at least)
        n_for_consensus = int(np.ceil(n_flies / 2))

        if panel != diag_panel_str:
            certain_glom_counts = panel_and_diag_df.columns.get_level_values('roi'
                ).value_counts()

            consensus_gloms = set(
                certain_glom_counts[certain_glom_counts >= n_for_consensus].index
            )
            consensus_glom_mask = panel_and_diag_df.columns.get_level_values('roi'
                ).isin(consensus_gloms)

            will_drop = panel_and_diag_df.loc[:, ~consensus_glom_mask].columns

            warn(f'dropping glomeruli seen <{n_for_consensus} (n_for_consensus) times '
                f'in {panel=} flies:\n{format_index_uniq(will_drop)}\n'
            )
            dropped_fly_rois.update(will_drop)

            # NOTE: this is what gets concatenated to form consensus_df used for
            # downstream analyses. remainder of this loop is to make additional outputs
            # (with filling Betty requested), which generated megamat consensus CSV I
            # initially sent Grant from the DePasquale lab.
            panel_consensus_df = panel_and_diag_df.loc[:, consensus_glom_mask]
            del consensus_gloms

        # in this diagnostic panel case, we'll subset data after loop based on consensus
        # glomeruli for OTHER panels (so that we don't show means responses for one
        # panel but not the diagnostics). should only be relevant to mean/stddev/N
        # variables built up in here.
        else:
            if verbose:
                print('not defining/dropping glomeruli for diagnostic panel! '
                    'should be dropped after loop instead, based on consensus glomeruli'
                    ' from other panels.'
                )

            panel_consensus_df = panel_and_diag_df

        del panel_and_diag_df

        if do_drop_nonconsensus_odors:
            # TODO (delete? handled? still relevant?) do i want to drop e.g. 'aphe @ -4'
            # in the megamat flies, even though it's consensus in the validation2 flies
            # (and we still have 'aphe @ -4' for 2/9 megamat flies, and no 'aphe @ -5'
            # for those 2)?  how to do things differently if not?
            #
            # Warning: dropping odors seen <7 (n_for_consensus) times in
            #  panel='glomeruli_diagnostics' flies:
            #                 panel   odor1  n_flies
            # glomeruli_diagnostics 2h @ -3        1
            # glomeruli_diagnostics CO2 @ 0        1
            #
            # Warning: dropping odors seen <5 (n_for_consensus) times in panel='megamat'
            #  flies:
            #                 panel     odor1  n_flies
            # glomeruli_diagnostics   2h @ -3        0
            # glomeruli_diagnostics aphe @ -4        2
            # glomeruli_diagnostics   CO2 @ 0        1
            #
            # Warning: dropping odors seen <3 (n_for_consensus) times in
            #  panel='validation2' flies:
            #                 panel     odor1  n_flies
            # glomeruli_diagnostics   2h @ -3        1
            # glomeruli_diagnostics aphe @ -5        0
            # glomeruli_diagnostics   CO2 @ 0        0
            #
            # NOTE: doing this non-consensus odor dropping only in non-diag panels
            # didn't seem to change anything in a direction I wanted (consensus_df after
            # loop same shape in either case, and still didn't have 'aphe @ -4' for
            # megamat panel flies)
            panel_consensus_df = drop_nonconsensus_odors(panel_consensus_df,
                n_for_consensus
            )

        # each element of consensus_dfs should contain both its own data, as well as all
        # the diagnostic panel data for the same flies, so we don't need to append when
        # the panel is the diagnostic panel.
        if panel != diag_panel_str:
            assert diag_panel_str in panel_consensus_df.index.get_level_values('panel')
            consensus_dfs.append(panel_consensus_df)

        # NOTE: happening *AFTER* appending to consensus_dfs. whether it happens before
        # is conditional on flag.
        #
        # if we dropped these above, no need to drop again, but if we DIDN'T drop above,
        # we still want to drop here (to keep outputs I made for Vlad consistent, but
        # also since there are currently some odors that appear 1 or 2 times only, and
        # those are currently making assertion about mean_df / stddev_df fail below
        # (since stddev will be NaN for those)).
        if not do_drop_nonconsensus_odors:
            panel_consensus_df = drop_nonconsensus_odors(panel_consensus_df,
                n_for_consensus, verbose=False
            )

        # panel_consensus_df still has diagnostic panel here (when `panel` variable
        # refers to any of the other panels)
        panel_df = panel_consensus_df.loc[panel].copy()

        if verbose:
            print('panel flies:')
            # TODO also use in load_antennal_csv.py
            print_index_uniq(panel_df.columns, all_fly_id_cols)
            print()

        panel_df = panel_df.droplevel(['is_pair', 'odor2'])
        panel_df = panel_df.droplevel(['date','fly_num'], axis='columns')

        # TODO TODO probably only do this style of filling if driver is pebbled. we'd
        # expected a certain set of other glomeruli to be missing in e.g. GH146 case

        # TODO replace w/ call to hemibrain_wPNKC (w/ _use_matt_wPNKC=False)?
        # TODO + assert data/ subdir CSV matches
        #
        # NOTE: the md5 of this file (3710390cdcfd4217e1fe38e0782961f6) matches what I
        # uploaded to initial Dropbox folder (Tom/hong_depasquale_collab) for Grant from
        # the DePasquale lab.
        #
        # Also matches ALL wPNKC.csv outputs I currently have under modeling output
        # subdirs, except those created w/ _use_matt_wPNKC=True (those wPNKC.csv files
        # have md5 2bc8b74c5cfd30f782ae5c2048126562). Though, none of my current outputs
        # had drop_receptors_not_in_hallem=True, which would lead to a different CSV.
        #
        # Also equal to wPNKC right after call to hemibrain_wPNKC(_use_matt_wPNKC=False)
        prat_hemibrain_wPNKC_csv = \
            'data/sent_to_grant/2024-04-05/connectivity/wPNKC.csv'

        wPNKC_for_filling = pd.read_csv(prat_hemibrain_wPNKC_csv).set_index('bodyid')
        wPNKC_for_filling.columns.name = 'glomerulus'

        hemibrain_glomeruli = set(wPNKC_for_filling.columns)
        # TODO delete (so I can keep change from panel_consensus_df -> panel_df below)
        assert panel_df.columns.get_level_values('roi').equals(
            panel_consensus_df.columns.get_level_values('roi')
        )
        #
        glomeruli_in_panel_flies = set(
            # TODO can i use panel_df instead of panel_consensus_df here (should be able
            # to)? could del panel_consensus_df where panel_df is defined if so
            #panel_consensus_df.columns.get_level_values('roi')
            panel_df.columns.get_level_values('roi')
        )

        glomeruli_not_in_hemibrain = glomeruli_in_panel_flies - hemibrain_glomeruli
        # seems to be true on all of my data (validation2 panel included), at least
        # since this loop is using certain ROIs
        assert len(glomeruli_not_in_hemibrain) == 0, ('glomeruli '
            f'{glomeruli_not_in_hemibrain} in panel={panel} data, but missing from '
            'hemibrain wPNKC'
        )

        hemibrain_glomeruli_not_in_panel_flies = (
            hemibrain_glomeruli - glomeruli_in_panel_flies
        )
        assert len(hemibrain_glomeruli_not_in_panel_flies) > 0

        # TODO delete?
        if verbose:
            # TODO also print value counts of glomeruli present across any of the panel
            # flies?
            print('hemibrain glomeruli not panel flies:')
            print(sorted(hemibrain_glomeruli_not_in_panel_flies))
            print()
        #

        _first_fly_cols = None

        filled_fly_dfs = []
        for fly_id, fly_df in panel_df.groupby(level='fly_id', axis='columns'):
            if verbose:
                date, fly_num = id2datenum.loc[fly_id]
                date_str = format_date(date)
                fly_str = f'{date_str}/{fly_num}'

                print(f'{fly_id=} ({fly_str})')

            assert type(fly_id) is str
            assert set(fly_df.columns.get_level_values('fly_id')) == {fly_id}
            # TODO nice way to add columns w/o dropping and re-adding this?
            fly_df = fly_df.droplevel('fly_id', axis='columns')
            assert fly_df.columns.name == 'roi'

            # important that this is computed before we potentially add some NaN columns
            # in conditional below
            fly_odors_all_nan = fly_df.isna().all(axis='columns')

            # might need extra considerations if this were true
            assert fly_odors_all_nan.equals(fly_df.isna().any(axis='columns')), \
                'some odors measured in this fly for only some glomeruli. unexpected.'

            fly_glomeruli = set(fly_df.columns.get_level_values('roi'))

            glomeruli_only_in_other_panel_flies = \
                glomeruli_in_panel_flies - fly_glomeruli

            if len(glomeruli_only_in_other_panel_flies) > 0:
                if verbose:
                    print('glomeruli only in other panel flies: '
                        f'{sorted(glomeruli_only_in_other_panel_flies)}'
                    )

                # TODO warn about what we are doing here?
                fly_df[sorted(glomeruli_only_in_other_panel_flies)] = float('nan')

            else:
                if verbose:
                    print('no glomeruli only in other panel flies')

            # might make some calculations on filled data easier if this were true
            # (and we'd never expected the dF/F to be *exactly* 0 for any real data)
            # TODO: n_per_odor_and_glom calculation might be modified to rely on this
            assert not (fly_df == 0.0).any().any(), ('data already had values at '
                'exactly 0.0 before 0-filling'
            )

            # TODO TODO include that this is happening in a message (w/ which glomeruli)
            #
            # this adds new columns
            fly_df[sorted(hemibrain_glomeruli_not_in_panel_flies)] = 0.0
            # this replaces any 0 (added above) for non-measured odors w/ NaN.
            # assertion above should ensure this is behaving correctly.
            fly_df.loc[fly_odors_all_nan] = float('nan')

            fly_df = fly_df.sort_index(axis='columns')

            if _first_fly_cols is None:
                _first_fly_cols = fly_df.columns
            else:
                assert fly_df.columns.equals(_first_fly_cols)

            # TODO warning message w/ fly_id and specific glomeruli in each fill
            # category

            fly_df = util.addlevel(fly_df, 'fly_id', fly_id, axis='columns')

            filled_fly_dfs.append(fly_df)

            if verbose:
                print()

        # TODO now that i'm only dropping non-consensus glomeruli AFTER the loop for the
        # diagnostic panel, check diag csv reasonable, or maybe don't save it
        # (probably wasn't gonna use/send it anyway... so nbd)
        filled_df = pd.concat(filled_fly_dfs, axis='columns', verify_integrity=True)
        # TODO assert filled_df has columns.is_monotonic_increasing?
        # (since we are sorting pre-concat, unlike newer fill_to_hemibrain fn...)
        # prob doesn't matter.

        # TODO save in a way where -c won't get triggered if run on diff set of flies?
        # (may mainly be an issue w/ diag panel, which will be present for many other
        # diff sets of flies)
        # TODO also say filled/similar in name?
        to_csv(filled_df, output_root / f'{panel}_consensus.csv',
            # TODO date_format even doing anything?
            date_format=date_fmt_str
        )
        del filled_fly_dfs

        # TODO delete?
        # TODO move before odor consensus dropping, so 'aphe @ -4' also dropped in this
        # case? (would make mean_df.isna() equal stddev_df.isna() below, fixing old
        # assertion)
        only_diags_from_megamat_flies = False
        if only_diags_from_megamat_flies and panel == diag_panel_str:
            dates = certain_df.columns.get_level_values('date')

            megamat_subset = certain_df.loc[:,
                (dates >= '2023-04-22') & (dates <= '2023-06-22')
            ]
            assert set(megamat_subset.dropna(how='all').index.get_level_values('panel')
                ) == {'glomeruli_diagnostics', 'megamat'}

            n_megamat_flies = 9
            megamat_flies = index_uniq(megamat_subset.columns, all_fly_id_cols)
            assert len(megamat_flies) == n_megamat_flies
            megamat_fly_ids = set(megamat_flies['fly_id'])
            assert len(megamat_fly_ids) == n_megamat_flies

            warn('for diagnostic panel, dropping non-megamat flies (because '
                f'{only_diags_from_megamat_flies=})!'
            )

            filled_df = filled_df.loc[:,
                filled_df.columns.get_level_values('fly_id').isin(megamat_fly_ids)
            ].copy()
            assert set(filled_df.columns.get_level_values('fly_id')) == megamat_fly_ids

        # averaging across trials ('repeat' level in row index).
        # still have separate fly info in 'fly_id' column level info after this.
        trialmean_df = filled_df.groupby(level='odor1', sort=False).mean()
        del filled_df

        # TODO delete (if deleting corresponding plot after loop)
        if panel == diag_panel_str:
            mean_df_diag_input = trialmean_df.copy()
        #

        trialmean_df = util.addlevel(trialmean_df, 'panel', panel)

        mean_df = trialmean_df.groupby(level='roi', sort=False, axis='columns'
            ).mean()
        # NOTE: as configured now, this seems to be NaN if only 1 fly (for a given roi),
        # but defined for even 2 (or more) flies for a given roi.
        stddev_df = trialmean_df.groupby(level='roi', sort=False,
            axis='columns').std()

        n_per_odor_and_glom = count_n_per_odor_and_glom(trialmean_df,
            # NOTE: count_zero=False working correctly relies on assertion above that no
            # exactly 0.0 values already existed in data
            count_zero=False
        )
        assert mean_df.index.equals(n_per_odor_and_glom.index)
        assert mean_df.columns.equals(n_per_odor_and_glom.columns)
        assert (
            n_per_odor_and_glom.sum().sum() ==
            trialmean_df.replace(0.0, np.nan).notna().sum().sum()
        )

        zeromean_gloms = (mean_df == 0).any()
        assert zeromean_gloms.equals((mean_df == 0).all())

        # otherwise, our filling could be skewing the mean of some glomeruli we do
        # actually detect sometimes.
        assert zeromean_gloms.equals(
            trialmean_df.groupby(level='roi', sort=False, axis='columns'
                ).apply(lambda x: (x == 0).any().any())
        )
        del zeromean_gloms

        panels_in_mean = set(mean_df.index.get_level_values('panel'))
        assert panels_in_mean == {panel}
        del panels_in_mean

        mean_df_list.append(mean_df)
        stddev_df_list.append(stddev_df)
        n_per_odor_and_glom_list.append(n_per_odor_and_glom)
    #


    # NOTE: this will sort the row index (don't think it's avoidable, esp since the
    # input df row indices aren't guaranteed to all be the same)
    consensus_df = pd.concat(consensus_dfs, axis='columns')

    # TODO factor out?
    def _nondiag_subset(df):
        return df[df.index.get_level_values('panel') != diag_panel_str]

    def merge_across_nondiag_panels(df):
        n_nondiag_before = num_notnull(_nondiag_subset(df))

        def merge_one_flyroi_across_nondiag_panels(fdf):
            nondiag = _nondiag_subset(fdf)
            # need <= 1 for (at least) data where va/aa NaN'd in megamat flies
            assert (nondiag.notna().sum(axis='columns') <= 1).all()

            diag = fdf.loc[fdf.index.difference(nondiag.index)]
            if diag.shape[-1] > 1:
                # TODO any reasons this would have issues? test if different consensus
                # judgements made based on the different non-diag panels? (won't be the
                # case for the current 2024 kiwi / control data)
                assert diag.T.duplicated(keep=False).all()

            filled_ser = fdf.bfill(axis='columns').iloc[:, 0]
            return filled_ser

        df = df.groupby(df.columns.names, axis='columns', sort=False).apply(
            merge_one_flyroi_across_nondiag_panels
        )
        n_nondiag_after = num_notnull(_nondiag_subset(df))
        assert n_nondiag_after == n_nondiag_before
        return df

    # consensus_dfs currently has one entry for each non-diag panel, and if a fly has 2
    # such panels, the diag component will be duplicated. this is to reduce across such
    # duplicate columns, filling data across non-diag panels (shouldn't change number of
    # non-NaN in non-diag portion)
    consensus_df = merge_across_nondiag_panels(consensus_df)

    # TODO fix panel sorting so that in new kiwi/control flies, diag panel still comes
    # before other two (currently control coming first)

    # TODO TODO TODO assert consensus_df output (megamat+validation) still matches what
    # i gave anoop (since adding merge_across_nondiag_panels step, rather than just
    # verify_integrity=True in concat)
    # (i'm not sure it does...)
    print('CHECK MERGE_ACROSS_NONDIAG_PANELS WOULD NOT CHANGED OLD CSVS SENT TO ANOOP')

    # these should just be the odors dropped (for each panel) in the loop above,
    # for not being presented at least the consensus (e.g. half flies each panel)
    # number of times
    odors_to_drop = certain_df.index.difference(consensus_df.index)

    if not do_drop_nonconsensus_odors:
        assert len(odors_to_drop) == 0

    if len(odors_to_drop) > 0:
        # NOTE: this is irrelevant if I don't return to actually doing anything with
        # certain_df below (should restore writing of it, so should be at least a little
        # relevant, but may still use consensus_df instead for much of the other
        # downstream analyses)
        #
        # TODO update comment language? is it actually true that it's non-diag?
        # (across all sets of non-diag panel flies. could still be odors only
        # non-consensus within the diagnostic data for all sets of panel flies)
        warn('certain_df: dropping odors seen < # for consensus times in *each* '
            f'panel:\n{format_index_uniq(odors_to_drop, ["panel", "odor1"])}\n'
        )
        certain_df = certain_df.drop(odors_to_drop)

    # TODO TODO make separate fn to drop odors consistent w/ how i had been recently
    # doing that here / above (to do right before plotting when needed, rather than
    # doing that for the output here)?

    consensus_df = sort_odors(consensus_df)
    # odor metadata should be same (unless I implement odor-consensus-dropping for
    # consensus_df and NOT certain_df. currently, same flag controls odor dropping for
    # both)
    assert consensus_df.index.equals(certain_df.index)

    # since the diag component duplicated across consensus_dfs for flies w/ multiple
    # non-diag panels, need to only check the non-diag part
    assert (
        sum([num_notnull(_nondiag_subset(x)) for x in consensus_dfs]) ==
        num_notnull(_nondiag_subset(consensus_df))
    )

    if consensus_df.shape[1] < certain_df.shape[1]:
        dropped_rois = certain_df.columns.difference(consensus_df.columns)

        # TODO add comment explaining what this means (/ at least give better var
        # names)
        assert set(dropped_rois) == dropped_fly_rois

        warn('dropped fly-glomeruli seen < # for consensus times in each non-diag '
            f'panel:\n{format_index_uniq(dropped_rois)}\n'
        )
        del dropped_rois, dropped_fly_rois

    # NOTE: seems to be no need to drop GH146-IDed-only stuff in pebbled case
    # (at least as glomeruli are currently IDed, as there currently aren't any (after
    # the consensus / certain criteria, though DA1/etc came close)
    #
    # these glomeruli were only seen in pebbled (though VC5/V might also be in GH146):
    # {'DL2d', 'DC4', 'DP1l', 'V', 'VC5', 'DL2v', 'VL1', 'VM5d', 'VC4', 'VM5v', 'VC3'}

    # TODO check all downstream stuff works same w/ or w/o this 'fly_id' level that
    # these column indices didn't use to have (-> delete this level dropping)
    # (also note that trial_df doesn't currently have this level. maybe i should
    # eventually move adding it to trial_df, and just let certain_df inherit from
    # there?)
    # (next line actually does currently break w/o dropping these)
    certain_df = certain_df.droplevel('fly_id', axis='columns')
    consensus_df = consensus_df.droplevel('fly_id', axis='columns')
    #

    # TODO want to also allow a version of this using certain_df instead of
    # consensus_df?
    # modelling currently hardcoding consensus_df as input (below), and this is only
    # place i'm planning to use that, so shouldn't matter.
    roi_best_plane_depths = roi_best_plane_depths.loc[:, consensus_df.columns]

    # TODO TODO only save these two if being run on all data?
    # (or just move saving to per panel directory [/name w/ panel] if not?)
    # TODO TODO also save certain_df still
    # TODO TODO rename these to "consensus", either way
    # (but be careful to not cause confusion w/ other people who already have some of
    # these files...)
    to_csv(consensus_df, output_root / 'ij_certain-roi_stats.csv',
        date_format=date_fmt_str
    )
    to_pickle(consensus_df, output_root / 'ij_certain-roi_stats.p')

    # since we already filled, these should all be hemibrain glomeruli
    assert all(
        x.columns.equals(mean_df_list[0].columns)
        for x in mean_df_list
    )

    mean_df = pd.concat(mean_df_list, verify_integrity=True)
    stddev_df = pd.concat(stddev_df_list, verify_integrity=True)
    n_per_odor_and_glom = pd.concat(n_per_odor_and_glom_list, verify_integrity=True)

    assert len(mean_df) == sum(x.shape[0] for x in mean_df_list)

    assert stddev_df.index.equals(mean_df.index)
    assert stddev_df.columns.equals(mean_df.columns)

    assert n_per_odor_and_glom.index.equals(mean_df.index)
    assert n_per_odor_and_glom.columns.equals(mean_df.columns)

    non_diag_mean = mean_df[
        mean_df.index.get_level_values('panel') != diag_panel_str
    ]
    assert not non_diag_mean.isna().any().any()

    gloms_never_consensus = (non_diag_mean == 0).all()
    if gloms_never_consensus.any():

        diag_subset = mean_df.loc[diag_panel_str, gloms_never_consensus]

        # below assertions are all just to know that we can only warn about the columns
        # other than the all 0 ones, and only those should have at least some real data
        # worth warning about zero-ing

        # VM1 all NaN if this is True, so assertion will fail
        # (below code modified to work in this case too)
        if not only_diags_from_megamat_flies:
            assert not diag_subset.isna().all().any()

        assert not ( diag_subset.isna().any() & (diag_subset == 0).any() ).any()
        assert (diag_subset == 0).any().equals( (diag_subset == 0).all() )

        have_diag_data = ~ ( (diag_subset == 0).all() | diag_subset.isna().all() )

        warn('zero-ing diagnostic panel glomeruli not consensus in *any* other panel:'
            f'\n{sorted(diag_subset.loc[:, have_diag_data].columns)}\n'
        )

        # TODO maybe do same 0-filling to a copy of mean consensus df, to compare?

        mean_df.loc[diag_panel_str, gloms_never_consensus] = 0.0
        stddev_df.loc[diag_panel_str, gloms_never_consensus] = 0.0
        n_per_odor_and_glom.loc[diag_panel_str, gloms_never_consensus] = 0

    del gloms_never_consensus

    assert (mean_df == 0).equals(stddev_df == 0)

    # TODO fix hack in this case (failing b/c of 'aphe @ -4' and glomeruli measured now
    # only 1 time, i think. that set seems like 'DP1l', 'VL1', 'VL2p', 'VM3')
    if not only_diags_from_megamat_flies:
        # (currently always dropping these odors for mean_df/etc, regardless of
        # do_drop_nonconsensus_odors value, so below comment irrelevant)
        # TODO TODO fix (after removing odor consensus dropping [via setting
        # do_drop_nonconsensus_odors=False], no longer true)
        # (why though, aren't both of these computed differently? still rely on same
        # odor consensus dropping? restore for these, indep of new flag?)
        assert mean_df.isna().equals(stddev_df.isna())

    assert not n_per_odor_and_glom.isna().any().any()

    being_run_on_all_final_pebbled_data = (start_date == '2023-04-22' and
        end_date == '2024-01-05' and driver == 'pebbled' and indicator == '6f'
    )
    if being_run_on_all_final_pebbled_data:
        # it does make sense that these NaN remain, as each is for a glomerulus only
        # found in (or at least, only consensus in) the validation2 panel flies (DL4 and
        # VM4), and I had switched aphe from -5 to -4 by the time I did these flies.
        #
        # i switched before last 2 [/9] megamat flies, so all validation2 flies had
        # aphe at -4. aphe -4 did not reach consensus (odor not presented in >= half of
        # flies) in megamat flies, so null in
        # ijroi/by_panel/glomeruli_diagnostics/certain.pdf.
        #
        # compare pdf/all-panel_mean_zero-mask.pdf with pdf/ijroi/certain.pdf to see
        # where the NaN are coming from.
        panelodor_with_some_nan = (diag_panel_str, 'aphe @ -5')

        assert set(mean_df.columns[
            mean_df.loc[panelodor_with_some_nan].isna()
        ]) == {'DL4', 'VM4'}

        # TODO TODO fix/adapt assertion in only_diags_from_megamat_flies=True case
        # ('DL4' & 'VM4' also null for ALL diag data in that case)
        if not only_diags_from_megamat_flies:
            assert not mean_df[
                mean_df.index != panelodor_with_some_nan
            ].isna().any().any()

    # TODO delete
    # TODO compare mean diagnostic panel responses from just megamat flies vs just
    # validation2 flies (e.g. does the impression change much if just take mean w/in
    # megamat flies vs if we use all)?
    #
    # there's a few instances where they differ, but they aren't THAT different. i think
    # i can just proceed with averaging all the data.
    # compare ijroi/by_panel/[megamat|validation2]/diags_certain_mean.pdf to each other
    # (and also compare to left hand side of all-panel_mean_zero-mask.pdf)
    #
    # TODO should i drop HCl because i think there might have just been
    # a problem with the odor vial (at least, for almost all of the validation2 flies)
    # (eh, can still see DC4 signal in all-panel_mean_zero-mask.pdf fine)

    trialmean_consensus_df = consensus_df.groupby(level=['panel','odor1'], sort=False
        ).mean()
    mean_consensus_df = trialmean_consensus_df.groupby(level='roi', axis='columns'
        ).mean()
    # ipdb> mean_df.shape
    # (64, 54)
    # ipdb> mcdf.shape
    # (64, 40)
    # ipdb> set(mcdf.columns) - set(mean_df.loc[:, (mean_df != 0).all()].columns)
    # {'VL1', 'VA4', 'DL4', 'VM4'}
    #
    # everything == or isclose other than cols in set above? (no)
    tcdf = trialmean_consensus_df.copy()
    mcdf = mean_consensus_df

    mdf = mean_df.replace(0, np.nan).dropna(how='all', axis='columns')

    # TODO at least add a comment about what these two are checking...
    assert mdf.columns.equals(mcdf.columns)

    # it's just the diagnostic panel portion that has anything differing.
    # seems true whether or not do_drop_nonconsensus_odors is True.
    assert mdf.loc[mdf.index.get_level_values('panel') != diag_panel_str
        ].equals(mcdf.loc[mcdf.index.get_level_values('panel') != diag_panel_str])

    # they will diverge if this is False, b/c mean_df/etc are *always* constructed
    # dropping non-consensus odors. this flag only applies to certain_df|consensus_df
    if do_drop_nonconsensus_odors:
        assert mdf.index.equals(mcdf.index)

        # TODO fix/adapt assertion for only_diags_from_megamat_flies=True case
        if being_run_on_all_final_pebbled_data and not only_diags_from_megamat_flies:
            assert mdf.drop('aphe @ -4', level='odor1').drop(columns=['VA4', 'VL1']
                ).equals(
                    mcdf.drop('aphe @ -4', level='odor1').drop(columns=['VA4', 'VL1'])
                )

    # seems mdf has 1 extra NaN in VA4 and VL1 (which odor(s)?)

    # TODO delete (/ refactor + replace code w/ this)
    #print_dataframe_diff(mdf, mcdf)

    # just to make easier to compare against (what used to be) NaNs
    mdf0 = mdf.fillna(0)
    mcdf0 = mcdf.fillna(0)
    del mdf, mcdf

    if not do_drop_nonconsensus_odors:
        # these should be odors dropped in mean_df def (b/c non-consensus), but not
        # dropped in consensus_df construction b/c do_drop_nonconsensus_odors=False
        mdf_only_odors = mcdf0.index.difference(mdf0.index)
        warn('odors in mean_df but not consensus_df, b/c do_drop_nonconsensus_odors='
            f'False:\n{mdf_only_odors.to_frame(index=False).to_string(index=False)}'
        )
        # this will drop the values not in mdf0.index
        mcdf0 = mcdf0.reindex(mdf0.index)

    diff = (mdf0 != mcdf0)
    # ipdb> diff.sum().sum()
    # 84

    diff_rows = diff.T.sum() > 0
    diff_cols = diff.sum() > 0

    # (all 1 in the ... parts)
    # ipdb> diff.sum()[diff_cols]
    # roi
    # D        1
    # ...
    # VA3      1
    # VA4     24
    # VA5      1
    # ...
    # VC4      1
    # VL1     24
    # VL2a     1
    # ...
    # VM7v     1
    if diff_cols.any():
        odors_diff_per_glom = diff.apply(_index_set_where_true).dropna()
        # TODO delete
        #d1 = diff.loc[:, diff_cols].apply(_index_set_where_true)
        #d2 = diff.apply(_index_set_where_true).loc[diff_cols]
        #print()
        #print(f'{diff.shape=}')
        #print(f'{diff.loc[:, diff_cols].shape=}')
        #print(f'{d1.equals(d2)=}')
        # when (failing) `do_drop_nonconsensus_odors=False`:
        # ipdb> d1
        # roi          VA4          VL1
        # 0      3mtp @ -5    3mtp @ -5
        # 1        va @ -4      va @ -4
        # 2       t2h @ -6     t2h @ -6
        # 3        ms @ -3      ms @ -3
        # 4    a-terp @ -3  a-terp @ -3
        # 5      geos @ -5    geos @ -5
        # 6      e3hb @ -6    e3hb @ -6
        # 7      mhex @ -7    mhex @ -7
        # 8        ma @ -7      ma @ -7
        # 9        ea @ -8      ea @ -8
        # 10    2-but @ -6   2-but @ -6
        # 11       2h @ -6      2h @ -6
        # 12       ga @ -4      ga @ -4
        # 13      HCl @ -1     HCl @ -1
        # 14   carene @ -3  carene @ -3
        # 15     farn @ -2    farn @ -2
        # 16    4h-ol @ -6   4h-ol @ -6
        # 17    2,3-b @ -6   2,3-b @ -6
        # 18    p-cre @ -3   p-cre @ -3
        # 19     aphe @ -4    aphe @ -4
        # 20    1-6ol @ -6   1-6ol @ -6
        # 21      paa @ -5     paa @ -5
        # 22    fench @ -5   fench @ -5
        # 23     elac @ -7    elac @ -7
        # ipdb> d2
        # roi
        # VA4    [3mtp @ -5, va @ -4, t2h @ -6, ms @ -3, a-terp...
        # VL1    [3mtp @ -5, va @ -4, t2h @ -6, ms @ -3, a-terp...
        # dtype: object
        # ipdb> d2.str.len()
        # roi
        # VA4    24
        # VL1    24
        # dtype: int64
        #
        # when `do_drop_nonconsensus_odors=True`:
        # ipdb> diff.shape
        # (64, 40)
        # ipdb> diff.loc[:, diff_cols].shape
        # (64, 38)
        # ipdb> d1
        # roi
        # D                                             [aphe @ -4]
        # ...
        # VA3                                           [aphe @ -4]
        # VA4     [3mtp @ -5, va @ -4, t2h @ -6, ms @ -3, a-terp...
        # VA5                                           [aphe @ -4]
        # ...
        # VC4                                           [aphe @ -4]
        # VL1     [3mtp @ -5, va @ -4, t2h @ -6, ms @ -3, a-terp...
        # VL2a                                          [aphe @ -4]
        # ...
        # VM7v                                          [aphe @ -4]
        # dtype: object
        # ipdb> d2
        # roi
        # D                                             [aphe @ -4]
        # ...
        # VA3                                           [aphe @ -4]
        # VA4     [3mtp @ -5, va @ -4, t2h @ -6, ms @ -3, a-terp...
        # VA5                                           [aphe @ -4]
        # ...
        # VC3                                           [aphe @ -4]
        # VC4                                           [aphe @ -4]
        # VL1     [3mtp @ -5, va @ -4, t2h @ -6, ms @ -3, a-terp...
        # VL2a                                          [aphe @ -4]
        # ...
        # VM7v                                          [aphe @ -4]
        # dtype: object
        #import ipdb; ipdb.set_trace()
        #

        # and was this ever True, or were we just not hitting `diff_cols.any()`? try w/
        # do_drop_nonconsensus_odors=True again? (yes, it does get hit, and assertion
        # works)
        # TODO what is this checking (to do unconditional on
        # do_drop_nonconsensus_odors)? worth fixing?
        # TODO and why do we not want to do this check if
        # do_drop_nonconsensus_odors=False? modify?
        if do_drop_nonconsensus_odors:
            assert diff.loc[:, diff_cols].apply(_index_set_where_true).equals(
                diff.apply(_index_set_where_true).loc[diff_cols]
            )

        # TODO TODO TODO fix
        # AttributeError: 'DataFrame' object has no attribute 'str'
        try:
            assert diff.sum()[diff_cols].equals(odors_diff_per_glom.str.len())
        # TODO delete
        except AttributeError:
            print()
            # ipdb> diff.sum()[diff_cols]
            # roi
            # D       10
            # DA2     10
            # DC1     10
            # DC3     10
            # DC4     10
            # DL1     10
            # DL5     10
            # DM1     10
            # DM2     10
            # DM3     10
            # DM4     10
            # DM5     10
            # DM6     10
            # DP1m    10
            # VA2     10
            # VA6     10
            # VC1     10
            # VC2     10
            # VC3     10
            # VC4     10
            # VM2     10
            # VM5d    10
            # dtype: int64
            # ipdb> odors_diff_per_glom
            # Empty DataFrame
            # Columns: [D, DA2, DC1, DC3, DC4, DL1, DL5, DM1, DM2, DM3, DM4, DM5, DM6, DP1m, VA2, VA6, VC1, VC2, VC3, VC4, VL2a, VM2, VM5d, VM7d]
            # Index: []
            print(f'{odors_diff_per_glom=}')
            print(f'{type(odors_diff_per_glom)=}')
            #import ipdb; ipdb.set_trace()
        #

        print()
        print('#' * 90)
        print('differences between mean_df and mean of consensus_df:')
        print('odors that differ for each glomerulus:')
        _print_diff_series(odors_diff_per_glom)

        # TODO TODO fix / gate so doesn't err on new kiwi/control data
        #
        try:
            # these are only 2 that differ by 24 entries (and they are same odors)
            assert odors_diff_per_glom['VA4'] == odors_diff_per_glom['VL1']
            # (all diagnostic odors, except 'aphe @ -5'. currently 25 diag odors in `diff`,
            # including 'aphe @ -4')
            # ipdb> len(odors_diff_per_glom['VA4'])
            # 24
        except KeyError as err:
            print('FIX ME')
            print(err)
            #import ipdb; ipdb.set_trace()


    if diff_rows.any():
        gloms_diff_per_odor = diff.apply(_index_set_where_true, axis='columns').dropna()

        assert diff.loc[diff_rows].apply(_index_set_where_true, axis='columns').equals(
            diff.apply(_index_set_where_true, axis='columns').loc[diff_rows]
        )
        assert diff.T.sum()[diff_rows].equals(gloms_diff_per_odor.str.len())

        print()
        print('glomeruli that differ for each odor:')
        _print_diff_series(gloms_diff_per_odor)
        # (all 2 where ...)
        # ipdb> diff.T.sum()[diff_rows]
        # panel                  odor1
        # glomeruli_diagnostics  3mtp @ -5       2
        # ...
        #                        p-cre @ -3      2
        #                        aphe @ -4      38
        #                        1-6ol @ -6      2
        # ...
        #                        elac @ -7       2
        print('#' * 90)
        print()

    # TODO how do these things actually differ tho? and where is diff in
    # non-'aphe @ -4' case coming from???
    #
    # from comparing pdf/DELETEME_mean_df_diag_input.pdf and pdf/ijroi/certain.pdf:
    # VA4 has data from 11-21/2 and 1-05/1 (validation2 flies, which wouldn't reach
    # consensus in just diag panel, as it's 2/5 flies) in mean_df input but not
    # consensus_df.
    #
    # VL1 is similar, w/ 11-21/3 and 1-05/1 having data, but no other validation2 flies.
    #
    #
    # TODO TODO also look at magnitude of diffs
    # TODO check NaNing those last 2 megamat flies (in input to mean_df calc) in aphe @
    # -4 can recapitulate diff for that odor?
    # TODO how would glomeruli/odors in one index but not the other currently be
    # represented above? need to add support for that (esp if i want to use these fns
    # more broadly, like in as-yet-unimplemented csvdiff CLI)

    # TODO TODO probably update my consensus_df calc to behave more like for mean_df
    # (where 'aphe @ -4' not dropped for last 2 megamat flies, which don't have 'aphe @
    # -5')
    #
    # just want to change handling of diag panel, so shouldn't affect outputs i
    # gave to other people, or at least not the non-diag part of them
    # (could change modeling slightly, only insofar as it changes dF/F->spiking fn, so
    # maybe not worth? so i don't have to resend modeling outputs to anoop)

    # TODO replace w/ correction to global hong2p.olf.odor2abbrev -> recompute all
    # per-recording ijroi analysis to update
    my_abbrev2remy = {
        'paa': 'PAA',
        '1-3ol': '1-prop',
        # to homogenize my diagnostic abbreviations w/ hers
        # (these already seem to have been applied in mean_df/etc, but still need to
        # transform when loading glomeruli_diagnostics.yaml below)
        '2but': '2-but',
        '6ol': '1-6ol',
    }
    def convert_my_abbrevs_to_remy(df: pd.DataFrame) -> pd.DataFrame:
        assert df.index.names == ['panel', 'odor1']

        # TODO replace (part of) below w/ abbrev(x, abbrevs=my_abbrev2remy)?

        odor_dicts = df.index.get_level_values('odor1').map(olf.parse_odor)

        new_odor_dicts = []
        for x in odor_dicts:
            if x['name'] in my_abbrev2remy:
                x = dict(x)
                x['name'] = my_abbrev2remy[x['name']]

            new_odor_dicts.append(x)

        new_odor_strs = [olf.format_odor(x) for x in new_odor_dicts]
        for_index = df.index.to_frame(index=False)
        # TODO TODO warn about which ones change
        for_index['odor1'] = new_odor_strs
        new_index = pd.MultiIndex.from_frame(for_index)

        df = df.copy()
        df.index = new_index
        return df

    mean_df = convert_my_abbrevs_to_remy(mean_df)
    stddev_df = convert_my_abbrevs_to_remy(stddev_df)
    n_per_odor_and_glom = convert_my_abbrevs_to_remy(n_per_odor_and_glom)

    dmin = mean_df.min().min()
    dmax = mean_df.max().max()
    vmin = -0.35
    # value I had been using for megamat/validation data. new 2024 kiwi/control data
    # seems to go outside this somewhat...
    vmax = 2.5
    if vmin <= dmin and dmax <= vmax:
        # TODO TODO are these just changing the labels (breaking meaning of their
        # position on the cbar?) why else are these not at the limits?
        # (no, i think it was just vmin2/vmax2 had a larger range than dmin/dmax)
        # TODO check another plot actually using vmin/vmax tho
        #
        # was using these for old megamat/validation analysis (pre 2024 kiwi/control)
        # TODO generate these dynamically
        cbar_ticks = [vmin, 0, 0.5, 1.0, 1.5, 2.0, vmax]

        cbar_kws = dict(ticks=cbar_ticks)
    else:
        # TODO does it matter to have one consistent scale in this context? if so, may
        # still need to hardcode / similar...
        # TODO round up/down to nearest multiple of 0.5/0.25 or something, when using
        # dmin/dmax?
        # TODO just leave None?
        #vmin = dmin
        #vmax = dmax
        # (defaults)
        vmin = None
        vmax = None
        # the default
        cbar_kws = None

    # TODO align more w/ roimean_plot_kws, to have mean/stddev output font size /
    # spacing look more like n_per_odor_and_glom plot below (other things already seem
    # same)? or allow overriding w/ these values, and pass them in to plot_n_... below?
    # TODO or use some of the plotting fns i'm currently using roimean_plot_kws instead
    # of viz.matshow?
    # TODO why am i defining this sep from e.g. roimean_plot_kws? how do they differ?
    # just use that?
    shared_kws = dict(
        xticklabels=format_mix_from_strs, yticklabels=True, levels_from_labels=False,
        vline_level_fn=format_panel, vline_group_text=True, vgroup_label_offset=0.12,
        cbar_kws=cbar_kws, linecolor='k'
    )
    prefix = 'all-panel_'

    # TODO TODO replace all mean/stddev plotting w/ plot_all...? why not?

    # TODO make sure (maybe via changing hong2p.viz default behavior) that cbar for
    # mean/stddev plots both have max of cbar labelled (and maybe also have min
    # labelled?). share w/ other code that hardcodes cbar ticks?

    # TODO still drop 'ms @ -3' from diag panel before sending to vlad?
    # talk to betty about it?
    # TODO delete
    # was checking that 2 'ms @ -3' vectors not too diff.
    #
    # biggest difference is in D. almost everything else is small and pretty similar
    # across the 2 'ms @ -3' vials (including DL1, the target of this diagnostic).
    #
    # ipdb> mean_df[mean_df.index.get_level_values('odor1') ==
    #     'ms @ -3'].T
    # panel glomeruli_diagnostics   megamat
    # odor1               ms @ -3   ms @ -3
    # roi
    # D                  0.789951  0.295207
    # ...
    # DL1                0.465520  0.408943
    # ...
    #ms_mean = mean_df[
    #    mean_df.index.get_level_values('odor1') == 'ms @ -3'
    #]
    #fig, _ = viz.matshow(ms_mean.replace(0, vmin - 1e-5).T,
    #    vmin=vmin, vmax=vmax, norm='two-slope', cmap=cmap, **shared_kws
    #)
    #savefig(fig, plot_root, 'ms_comparison_mean_zero-mask', bbox_inches='tight')
    #
    # dropping this since I think it's likely the (small) differences between responses
    # in the diagnostic vs megamat panel is likely due to contamination in the
    # diagnostic vial, and likely to cause confusion having an odor in 2 panels. likely
    # more correct to just use megamat values here, rather than taking mean of both.
    # TODO restore True?
    #drop_diag_panel_ms = True
    drop_diag_panel_ms = False
    if drop_diag_panel_ms:
        warn("dropping diagnostic 'ms @ -3' data, to avoid confusion with megamat data"
            ' (because drop_diag_panel_ms=True)'
        )
        # TODO fix to relax when being run on just validation data
        # (or just use being_run_on... ?)
        #assert set(mean_df.index[mean_df.droplevel('panel').index.duplicated()]
        #    ) == {('megamat', 'ms @ -3')}

        shape_before = mean_df.shape

        to_drop = (diag_panel_str, 'ms @ -3')
        mean_df = mean_df.drop(index=to_drop)
        stddev_df = stddev_df.drop(index=to_drop)
        n_per_odor_and_glom = n_per_odor_and_glom.drop(index=to_drop)

        expected_shape_after = (shape_before[0] - 1, shape_before[1])
        assert mean_df.shape == expected_shape_after
        assert stddev_df.shape == expected_shape_after
        assert n_per_odor_and_glom.shape == expected_shape_after

        # checking that no odors (other than 'ms @ -3' we just dropped) appear in >1
        # panels.
        # assuming same holds true for other 2 DataFrames (checked indexes equal above).
        assert not mean_df.droplevel('panel').index.duplicated().any()

    if being_run_on_all_final_pebbled_data:
        fig, _ = viz.matshow(mean_df.T, vmin=vmin, vmax=vmax, **diverging_cmap_kwargs,
            **shared_kws
        )
        savefig(fig, plot_root, f'{prefix}mean', bbox_inches='tight')

        fig, _ = viz.matshow(stddev_df.T, vmin=0, vmax=vmax, **diverging_cmap_kwargs,
            **shared_kws
        )
        savefig(fig, plot_root, f'{prefix}stddev', bbox_inches='tight')

        cmap = diverging_cmap.copy()
        # TODO are there already under/over before i set them (yes)?
        #cmap.set_under('black')
        # TODO refactor to share w/ set_bad gray def above
        cmap.set_under((0.8, 0.8, 0.8))

        # TODO delete
        #'''
        # so ROIs are actually grouped together
        mean_df_diag_input = mean_df_diag_input.sort_index(level='roi', axis='columns')

        new_fly_cols = mean_df_diag_input.columns.get_level_values('fly_id').map(
            lambda x: tuple(id2datenum.loc[x])
        )
        new_fly_cols.names = ['date', 'fly_num']

        for_index = new_fly_cols.to_frame(index=False)
        assert mean_df_diag_input.columns.names == ['fly_id', 'roi']
        for_index['roi'] = mean_df_diag_input.columns.get_level_values('roi')
        new_index = pd.MultiIndex.from_frame(for_index)

        # comment this line if you want 'fly_id' instead of ['date','fly_num'] in plot
        mean_df_diag_input.columns = new_index

        vmin2 = mean_df_diag_input.min().min()
        vmax2 = mean_df_diag_input.max().max()

        kws = dict(shared_kws)
        kws['yticklabels'] = lambda x: fly_roi_id(x, fly_only=True)
        # wrong range for this plot
        del kws['cbar_kws']

        # TODO delete. was to test set_over (which works, as long as extend='both' set
        # in cbar_kws)
        #vmax2 = 2.5
        #cmap2 = cmap.copy()
        #
        #cmap2.set_over('magenta')
        # TODO 'red' (yea, ish) |'orangered' distinct enough?
        # https://matplotlib.org/stable/gallery/color/named_colors.html
        #cmap2.set_over('red')
        #
        # TODO can i automatically set colorbar(...) kwarg extend='both'|'min'|'max'
        # appropriately (maybe by detecting if set_over/under set? or are there always
        # indistinguishable defaults for cmaps i'm using anyway?).
        # would want to detect + set in viz.matshow or viz.add_colorbar
        #
        # to get over/under colors to show up as triangles above/below cbar
        #kws['cbar_kws'] = dict(extend='both')

        fig, _ = viz.matshow(mean_df_diag_input.replace(0, vmin2 - 1e-5).T, vmin=vmin2,
            vmax=vmax2, norm='two-slope', cmap=cmap, hline_level_fn=roi_label,
            hline_group_text=True, inches_per_cell=0.08, fontsize=4.5, linewidth=0.5,
            dpi=300, hgroup_label_offset=0.2, **kws
        )
        savefig(fig, plot_root, 'DELETEME_mean_df_diag_input', bbox_inches='tight')
        #'''
        #

        fig, _ = viz.matshow(mean_df.replace(0, vmin - 1e-5).T, vmin=vmin, vmax=vmax,
            norm='two-slope', cmap=cmap, **shared_kws
        )
        savefig(fig, plot_root, f'{prefix}mean_zero-mask', bbox_inches='tight')

        fig, _ = viz.matshow(stddev_df.replace(0, -1e-5).T, vmin=0, vmax=vmax,
            norm='two-slope', cmap=cmap, **shared_kws
        )
        savefig(fig, plot_root, f'{prefix}stddev_zero-mask', bbox_inches='tight')
        del vmin, vmax, shared_kws

        # nothing in shared_kws above should be meaningfully different from what this fn
        # will provide, though font size and spacing seem to be slightly diff from above
        fig, _ = plot_n_per_odor_and_glom(n_per_odor_and_glom,
            input_already_counts=True
        )
        savefig(fig, plot_root, f'{prefix}_n_per_odor_and_glom', bbox_inches='tight')
        del prefix


        output_prefix = 'consensus_'
        # TODO refactor below

        # no dates in any of these, so don't need date_format=date_fmt_str part

        to_csv(mean_df, output_root / f'{output_prefix}mean.csv')
        to_pickle(mean_df, output_root / f'{output_prefix}mean.p')

        to_csv(stddev_df, output_root / f'{output_prefix}stddev.csv')
        to_pickle(stddev_df, output_root / f'{output_prefix}stddev.p')

        to_csv(n_per_odor_and_glom,
            output_root / f'{output_prefix}n_per_odor_and_glom.csv'
        )
        to_pickle(n_per_odor_and_glom,
            output_root / f'{output_prefix}n_per_odor_and_glom.p'
        )

        # TODO TODO TODO which hemibrain glomeruli not in task? and vice versa?
        # also add warnings about this in same hemibrain modelling code
        glomerulus2receptors = orns.task_glomerulus2receptors()
        receptors = mean_df.columns.map(lambda x: ','.join(glomerulus2receptors[x]))
        glomeruli_to_receptors_df = pd.DataFrame({
            'glomerulus': mean_df.columns, 'receptors': receptors
        })
        # TODO warn about which have multiple too
        to_csv(glomeruli_to_receptors_df,
            output_root / f'{output_prefix}glomeruli_to_receptors.csv', index=False
        )


    # NOTE: copied from my chemutils package. rather not install that package now, cause
    # its cache handling is pretty slow at start/atexit (and don't want to screw up the
    # caches).
    def pubchem_url(cid):
        if pd.isnull(cid):
            return cid
        # TODO replace w/ f-string
        return 'https://pubchem.ncbi.nlm.nih.gov/compound/{}'.format(cid)

    def get_unique_col(df: pd.DataFrame, substr: str) -> Optional[str]:
        matching_cols = [c for c in df.columns if substr in c.lower()]

        if len(matching_cols) > 1:
            raise ValueError('multiple matching columns')

        if len(matching_cols) == 0:
            return None
        return matching_cols[0]

    def get_abbrev_col(df: pd.DataFrame) -> Optional[str]:
        return get_unique_col(df, 'abbrev')

    def get_cid_col(df: pd.DataFrame) -> Optional[str]:
        # TODO filter stuff w/ 'fuzzy' in name?
        return get_unique_col(df, 'cid')

    chem_id_cols = ['name', 'abbrev', 'cid']

    my_name2remy = {
        'isoamyl acetate': 'isopentyl acetate',
        'B-citronellol': 'b-citronellol',
        # Remy uses the name 'pentanoic acid' for this
        'valeric acid': 'pentanoic acid',

        # 'perfume' prefix was only ever for internal use really, to clarify
        # supplier we got it from. same O-28 that both I used (for diagnostics)
        # and Remy used (for our validation2 panel).
        'perfume geosmin': 'geosmin',
    }

    remy_odor_metadata_dir = Path('data/from_remy/2024-06-05')
    panel2master_sheet = {
        'megamat': 'Sheet1',
        'validation2': 'validation2',
    }
    panel2sheets_to_skip = {
        'megamat': (
            # has a different CID For B-cit (8842, instead of 101977)
            # TODO TODO which is correct tho?
            # TODO TODO TODO probably DO use 8842. remy unclear on why she used other.
            # 8842 is racemic/ambig stereochem, which matches what we ordered from sigma
            # better.
            'coconut_ids',

            # only has fuzzy CIDs
            'megamat17_fuzzy_cids',
            # assuming strict CIDs are what I want, don't need mapping to fuzzy CIDs
            'strict_2_fuzzy_cids',
        ),
        'validation2': (
            'validation2_fuzzy_cids',
            'validation2_fuzzy_cid_2_vcf_url',
        ),
    }
    if verbose:
        print()

    odor_metadata_dfs = []
    for panel, pdf in mean_df.groupby(level='panel', sort=False):
        if verbose:
            print(f'{panel=}')

        my_abbrevs = {
            olf.parse_odor_name(x) for x in pdf.index.get_level_values('odor1')
        }

        # these don't seem to have InChI (some do actually, maybe just megamat?), but do
        # have PubChem CID / SMILES / IsomericSMILES / etc.
        remy_panel_odor_metadata = remy_odor_metadata_dir / f'{panel}.xlsx'
        if remy_panel_odor_metadata.exists():
            sheet2df = pd.read_excel(remy_panel_odor_metadata, sheet_name=None)

            master_sheet_name = panel2master_sheet[panel]
            df = sheet2df[master_sheet_name]

            abbrev_col = get_abbrev_col(df)
            assert abbrev_col is not None

            sheet_abbrevs = set(df[abbrev_col])
            assert sheet_abbrevs == my_abbrevs

            cid_col = get_cid_col(df)
            assert cid_col is not None

            abbrev2cid = dict(zip(df[abbrev_col], df[cid_col]))

            name_col_opts = ('olfactometer_name', 'name')
            name_col = None
            for n in name_col_opts:
                if n in df.columns:
                    name_col = n
                    break

            assert name_col is not None
            # this one has both 'name' and 'olfactometer_name' columns
            if panel != 'validation2':
                assert not any(x in df.columns for x in name_col_opts if x != name_col)

            # TODO delete? convert to assertion?
            else:
                # TODO where did each of these come from?
                print('diff names:')
                print(df.loc[df['olfactometer_name'] != df['name'],
                    ['name', 'olfactometer_name']
                ])
                # TODO assert name and olfactometer_name cols equal
                #import ipdb; ipdb.set_trace()

            abbrev2name = dict(zip(df[abbrev_col], df[name_col]))

            # TODO rename odf->other_df when done
            for sheet_name, odf in sheet2df.items():
                if sheet_name == master_sheet_name:
                    continue

                if verbose:
                    print(f'{sheet_name=}')

                sheets_to_skip = panel2sheets_to_skip[panel]
                if sheet_name in sheets_to_skip:
                    continue

                other_abbrev_col = get_abbrev_col(odf)
                other_cid_col = get_cid_col(odf)

                if other_cid_col is None:
                    if verbose:
                        print('missing CID col! skipping!')
                        print()

                    continue

                # assuming abbrev -> name mappings don't need to be checked against
                # master sheet.

                other_abbrev2cid = dict(zip(odf[other_abbrev_col], odf[other_cid_col]))
                try:
                    assert abbrev2cid == other_abbrev2cid
                    if verbose:
                        print('abbrev->cid map matched master sheet!')
                except:
                    # NOTE: no longer reached after ignoring sheet that had 8842 for
                    # b-cit, but i'm also manually forcing that CID below

                    assert set(odf[other_abbrev_col]) == set(df[abbrev_col])

                    print('mismatching CIDs:')
                    for k, v in other_abbrev2cid.items():
                        if k not in abbrev2cid:
                            print(f'{k} not in abbrev2cid!')
                            continue

                        v0 = abbrev2cid[k]
                        if v0 != v:
                            print(f'{k}: master={v0}, other={v}')
                            print(pubchem_url(v0))
                            print(pubchem_url(v))

                    # TODO fix
                    import ipdb; ipdb.set_trace()

                if verbose:
                    print()

            # TODO assert name == olfactometer_name for all (otherwise pick one)
            # (they aren't, but not sure it matters. would need to check against
            # supplier part numbers probably.)
            # (checked all CIDs by going from product pages, so names from pubchem /
            # product pages might be more accurate, but they can always go from CIDs
            # themselves at least)

            df = df[[name_col, abbrev_col, cid_col]].rename(columns={
                name_col: 'name',
                abbrev_col: 'abbrev',
                cid_col: 'cid',
            })

            panel_odor_metadata = load_olf_input_yaml(f'{panel}.yaml')
            panel_odor_metadata['name'] = panel_odor_metadata.name.replace(my_name2remy)
            # NOTE: no 'cid' column in these (except my own
            # 'glomeruli_diagnostics.yaml', loaded below)
            panel_odor_metadata = panel_odor_metadata[['name', 'abbrev', 'solvent']]

            assert set(panel_odor_metadata.name) == set(df.name)
            assert set(panel_odor_metadata.abbrev) == set(df.abbrev)

            len_before = len(df)

            df = df.merge(panel_odor_metadata, on=['name', 'abbrev'])
            assert len(df) == len_before

            if verbose:
                print()

        else:
            # TODO reorganize code to do this earlier in loop, and try to reduce overall
            # nesting
            if panel != diag_panel_str:
                warn(f'not including {panel=} in odor metadata CSV')
                continue
            #

            # TODO also use this for defining target glomeruli (elsewhere)? or do i also
            # have those in output yamls? how am i handling now?
            df = load_olf_input_yaml('glomeruli_diagnostics.yaml')
            df = df[chem_id_cols + ['solvent']].copy()

            # also implies no missing (b/c then dtype would be float / object)
            assert df.cid.dtype == int

            df['abbrev'] = df.abbrev.map(lambda x: my_abbrev2remy.get(x, x))
            yaml_abbrevs = set(df.abbrev)

            data_abbrevs = {
                olf.parse_odor_name(x)
                for x in mean_df.loc[diag_panel_str].index.get_level_values('odor1')
            }
            if drop_diag_panel_ms:
                assert data_abbrevs - yaml_abbrevs == set()
                assert yaml_abbrevs - data_abbrevs == {'ms'}
            else:
                # TODO TODO fix
                try:
                    assert data_abbrevs == yaml_abbrevs
                # TODO delete
                except:
                    # TODO what caused this?
                    # ipdb> data_abbrevs - yaml_abbrevs
                    # {'HCl'}
                    # ipdb> yaml_abbrevs - data_abbrevs
                    # set()
                    print()
                    print(f'{data_abbrevs=}')
                    print(f'{yaml_abbrevs=}')
                    #import ipdb; ipdb.set_trace()
                #

            df['name'] = df.name.replace(my_name2remy)

        odor_metadata_dfs.append(df)

    odor_metadata = pd.concat(odor_metadata_dfs, ignore_index=True)

    cid_corrections = {
        # for both of these, I'm pretty confident in the CIDs.
        # for each, I did: supplier part no. -> CAS -> CID.
        # not sure how Remy got hers, and don't think she had a better reason.
        #
        # geosmin
        29746: 1213,
        # b-cit
        101977: 8842,

        # menthone (manual CID is a mixture of isomers, from Fischer part AAA1766518
        # -> CID they list directly on product page)
        26447: 6986,

        # (-)-alpha-pinene. Remy thinks she used 4-23 (Sigma 80599-1ML -> 7785-26-4 ->
        # 440968), not (racemic?) F-5 (Sigma 147524 -> 80-56-8 -> 6654).
        6654: 440968,
    }
    if being_run_on_all_final_pebbled_data:
        assert all(x in set(odor_metadata.cid) for x in cid_corrections.keys())

    odor_metadata['cid'] = odor_metadata.cid.replace(cid_corrections)

    assert set(odor_metadata.columns) == set(chem_id_cols + ['solvent'])
    n_unique_including_solvent = len(odor_metadata.drop_duplicates())
    # assuming 'solvent' same for unique combination of preceding 3 cols.
    odor_metadata = odor_metadata.drop_duplicates(subset=chem_id_cols,
        ignore_index=True
    )
    assert not any(odor_metadata[c].duplicated().any() for c in chem_id_cols)
    # to check that one odor doesn't have 2 diff solvents listed.
    assert len(odor_metadata) == n_unique_including_solvent

    odor_metadata['pubchem_url'] = odor_metadata.cid.map(pubchem_url)

    # TODO also only compute above odor_metadata if this is true? not used below...
    if being_run_on_all_final_pebbled_data:
        to_csv(odor_metadata, output_root / 'odor_metadata.csv', index=False)

    # only have 14/54 hemibrain glomeruli that we never find via consensus:
    # ipdb> ( ~(mean_df.isna() | (mean_df == 0) ).all() ).sum()
    # 40
    # ipdb> ( (mean_df.isna() | (mean_df == 0) ).all() ).sum()
    # 14

    # TODO TODO any way to get date_format to apply to stuff in MultiIndex?
    # or is the issue that it's a pd.Timestamp and not datetime?
    # TODO link the github issue that might or might not still be open about this
    # date_format + index level issue (may need to update pandas if it has been
    # resolved)
    to_csv(trial_df, output_root / 'ij_roi_stats.csv', date_format=date_fmt_str)

    # TODO TODO still have one global cache with all data (for use in plot_roi.py /
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

    # TODO delete? probably
    # TODO still want to keep this?
    # This is dropping '~kiwi' and 'control mix' at concentrations other than undiluted
    # ('@ 0') concentration.
    #trial_df = natmix.drop_mix_dilutions(trial_df)

    # TODO delete / fix.
    # some across fly ijroi plotting currently broken w/ pair data (or maybe just when
    # there is pair + non-pair w/ some of same odors?)
    if trial_df.index.get_level_values('is_pair').any():
        # TODO convert print to warning
        print('DROPPING ALL PAIR DATA SINCE SOME OF ACROSS FLY IJROI PLOTTING BROKEN')
        trial_df = trial_df[trial_df.index.get_level_values('is_pair') == False]

    # TODO CLI flag for this?
    # TODO TODO delete (/ change analyses to select consensus theirselves? or pass
    # both as input, so certain plots can still be made w/ uncertain or non-consensus
    # ROIs, where we might want that, e.g. ijroi response plots.)
    #
    # NOTE: WANT this True for paper figures, for now.
    # NOTE: modelling is currently only downstream analysis that will always use
    # TODO reason for that?
    # consensus_df. everything else uses trial_df.
    #use_consensus_for_all_acrossfly = True
    use_consensus_for_all_acrossfly = False
    # TODO delete + fix
    print(f'{use_consensus_for_all_acrossfly=} (should be True for paper figures)')
    #
    if use_consensus_for_all_acrossfly:
        warn('using consensus_df instead of trial_df for all across fly analyses!!! '
            'may want to change!'
        )

        # hack to avoid having to make a separate version for this or refactor consensus
        # creation to share w/ this
        trial_df = consensus_df

        # TODO delete? not currently used after here...
        #
        # only modelling below [WAS] currently using this instead of trial_df.
        # consensus_df should already only be certain ROIs.
        #certain_df = consensus_df
    #

    across_fly_ijroi_dir = plot_root / across_fly_ijroi_dirname
    makedirs(across_fly_ijroi_dir)

    # TODO if --verbose, print when we skip this
    if 'ijroi' not in steps_to_skip:
        # TODO TODO maybe internally recompute consensus where i want it?
        # pass both (w/ diff flags? probably in two calls?)
        #
        # was there a reason i was passing consensus_df here instead of one of two dfs
        # set to that in the use_consensus_for_all_acrossfly==True case? assuming it was
        # a mistake...
        acrossfly_response_matrix_plots(trial_df, across_fly_ijroi_dir, driver,
            indicator
        )

    # TODO if --verbose, print when we skip this
    if 'corr' not in steps_to_skip:
        # TODO TODO move this enumeration of certain_only values inside
        # acrossfly_correlation_plots?)
        # TODO rename across_fly...? (and do same for similar fns to organize across fly
        # analysis steps?)
        acrossfly_correlation_plots(output_root, trial_df, certain_only=True)
        acrossfly_correlation_plots(output_root, trial_df, certain_only=False)

    # TODO if --verbose, print when we skip this
    if 'intensity' not in steps_to_skip:
        # TODO drop mix dilutions here (now that i'm not doing to trial_df
        # unconditionally above, at least when it's not overwritten by consensus_df...)?

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
        # TODO TODO make sure these are also including the new 2-component mixtures
        # included. (or at least that some versions are)
        natmix_activation_strength_plots(mean_df, intensities_plot_dir)

    # would not be true for versions of repo from before I added data/ (affect Sam?)
    assert data_outputs_root.is_dir()
    # TODO what is using this? add comment explaining (+ why are we making here and not
    # there?)
    #
    # (subdirectory of data_outputs_root)
    hallem_csv_root.mkdir(exist_ok=True)

    # TODO at least if --verbose, print we are skipping step (and in all cases we skip
    # steps)
    # TODO TODO get this to work if script is run with megamat + validation input
    # (and ideally make sure megamat is run first, esp if we serialize model params
    # there to use in validation panel, for the dff->spikedelta transform)
    # (not happy w/ current behavior?)
    if 'model' not in steps_to_skip:
        assert 'model-sensitivity' in skippable_steps
        skip_sensitivity_analysis = 'model-sensitivity' in steps_to_skip

        assert 'model-seeds' in skippable_steps
        skip_models_with_seeds = 'model-seeds' in steps_to_skip

        # TODO worth warning that model won't be run otherwise?
        # TODO TODO TODO was this not consensus_df before? what do i want?
        # TODO TODO TODO compare cached input to dF/F -> spiking model (if i had one
        # yet, if not regen after making decision) to figure out whether i used
        # consensus_df or certain_df for that, and do i want to change that? maybe after
        # forming consensus glomeruli only within each panel?
        if driver in orn_drivers:
            # NOTE: use_consensus_for_all_acrossfly must be False if I want to try using
            # certain_df again here (if True, certain_df is redefined to consensus_df
            # above)
            model_mb_responses(consensus_df, across_fly_ijroi_dir,
                roi_depths=roi_best_plane_depths,
                skip_sensitivity_analysis=skip_sensitivity_analysis,
                skip_models_with_seeds=skip_models_with_seeds,
            )
        else:
            print(f'not running MB model(s), as driver not in {orn_drivers=}')


if __name__ == '__main__':
    main()

