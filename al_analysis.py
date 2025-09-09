#!/usr/bin/env python3

import argparse
import atexit
import os
from os.path import join, split, exists, islink, getmtime
from pprint import pprint, pformat
from collections import defaultdict, Counter
from datetime import datetime
import warnings
import time
import shutil
import traceback
import sys
from pathlib import Path
import glob
from itertools import starmap
import multiprocessing
from typing import Optional, Tuple, List, Union, Dict, Set, Any, Callable
import json

import numpy as np
import pandas as pd
import xarray as xr
import tifffile
import yaml
from matplotlib import patches
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import zscore as scipy_zscore
import colorama
from termcolor import cprint, colored
from tqdm import tqdm
# suite2p imports are currently done at the top of functions that use them

from drosolf import orns
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
    num_notnull, add_fly_id, pd_allclose
)
from hong2p.olf import (format_mix_from_strs, format_odor_list, solvent_str,
    odor2abbrev, odor_lists_to_multiindex
)
from hong2p.viz import dff_latex
from hong2p.err import NoStimulusFile
from hong2p.thor import OnsetOffsetNumMismatch
from hong2p.types import ExperimentOdors, Pathlike
from hong2p.xarray import (move_all_coords_to_index, unique_coord_value, scalar_coords,
    drop_scalar_coords, assign_scalar_coords_to_dim, odor_corr_frame_to_dataarray
)
import natmix
# TODO rename these [load|write]_corr_dataarray fns to remove reference to "corr"
# (since correlations are not what i'm using with these fns here)
# (but maybe add something else descriptive?)
from natmix import load_corr_dataarray, drop_nonlone_pair_expt_odors, dropna_odors
from natmix import write_corr_dataarray as _write_corr_dataarray

# TODO move this to hong2p probably
from hong_logging import init_logger
#
from al_util import (savefig, abbrev_hallem_odor_index, sort_odors, panel2name_order,
    diag_panel_str, warn, should_ignore_existing, ignore_existing_options, data_root,
    produces_output, to_csv, to_pickle, read_pickle, plot_fmt, makedirs, cmap,
    diverging_cmap, diverging_cmap_kwargs, bootstrap_seed, cluster_rois, plot_corr,
    plot_responses, mean_of_fly_corrs
)
# TODO check we can set global flags from main below (only reason this is imported)
# (and in a way s.t. al_util fns actually use new values...)
import al_util
#
from mb_model import model_mb_responses


# TODO TODO restore (triggered in dF/F calc for test no-fly dry-run data)
# RuntimeWarning: invalid value encountered in scalar multiply
#warnings.filterwarnings('error', 'invalid value encountered in')


# TODO TODO catch:
# FutureWarning: The default value of numeric_only in DataFrameGroupBy.std is
# deprecated. In a future version, numeric_only will default to False. Either specify
# numeric_only or select only columns which should be valid for the function.
#
# Warning: There are no gridspecs with layoutgrids. Possibly did not call parent
# GridSpec with the "figure" keyword


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

# TODO delete? didn't seem to fix issue (also w/ mpl.use('Agg'))
# TODO need these settings from set_font_settings_for_testing from here:
# https://github.com/matplotlib/matplotlib/blob/v3.10.0/lib/matplotlib/testing/__init__.py#L24-L25
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['text.hinting'] = 'none'
plt.rcParams['text.hinting_factor'] = 8

mpl.rcParams['svg.hashsalt'] = 'matplotlib'
# ...in order to get reproducible PDFs for compare_images calls?
# doesn't seem to have helped my issue.
#

# TODO delete?
# https://stackoverflow.com/questions/15896174
#mpl.rcParams['text.usetex'] = True
#

# TODO this screwing up reproducbility of pdfs? (and thus, ability to use -c/-C to check
# output plots aren't changing)?
# TODO also add to a matplotlibrc file (under ~/.matplotlib?)?
# TODO 42 same as TrueType?
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# TODO delete. to enable some extra debugging prints, trying to not have them show up if
# sam uses that version of hong2p
# TODO restore True and see if there's anything unresolved about matshow
# add_norm_options / twoslopenorm [+ cbar ticks?] handling ?
viz._debug = False


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

# TODO delete
# Whether to only analyze experiments sampling 2 odors at all pairwise concentrations
# (the main type of experiment for this project)
analyze_pairgrids_only = False

# TODO delete
# If there are multiple experiments with the same odors, only the data from the most
# recent concentrations for those odors will be analyzed.
final_pair_concentrations_only = True

# TODO delete
analyze_reverse_order = True

# NOTE: not actually a constant now. set True in main if CLI flag set to *only* analyze
# glomeruli diagnostics
# Will be set False if analyze_pairgrids_only=True
analyze_glomeruli_diagnostics = True

# Whether to analyze any single plane data that is found under the enumerated
# directories.
analyze_2d_tests = False

# TODO delete? i assume it's something else that controls currently used suite2p
# registration outputs?
#
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

# TODO TODO change to using seconds and rounding [to nearest? higher/lower?] from there
# TODO some quantitative check this is ~optimal?
# Note this is w/ volumetric sampling of ~1Hz (actually almost all my stuff is closer to
# ~0.6Hz, so may need to re-evaluate).
# NOTE: currently need this =2 to recreate paper outputs (for pebbled at least.  may
# change params i want to use for gh146, but had been using 2 as well [and all same
# params] for GH146 too, at least until 2025-07-07). had also experimented w/ =3.
# (using 2 to accomodate earlier recordings for Remy's project, where pulse was still
# 2s) (which were affected? am i still using those? don't think so? methods summary CSVs
# should help clarify that, assuming pulse timing part of those are accurate...)
n_volumes_for_response = 2

# If this is None, it will use all the everything from the start of the trial to the
# frame before the first odor frames (see also exclude_last_pre_odor_frame below).
# Otherwise, will use this many frames before first odor frame.
n_volumes_for_baseline = None

# TODO use a fn for this, like in compute_trial_stats stat= kwarg?
# NOTE: this did not exist for a long time (baseline was always a mean of however many
# volumes). use False to preserve that behavior.
median_for_baseline = False

# Whether to exclude last frame before odor onset in calculating the baseline for each
# trial. These baselines are used to define the dF/F for each trial. This can help hedge
# against off-by-one bugs in the frame<-> assignment code, but might otherwise weaken
# signals.
exclude_last_pre_odor_frame = False

# NOTE: never really used this, nor seriously compared outputs generated using it.
#
# if False, one baseline per trial. if True, baseline to first trial of each odor.
one_baseline_per_odor = False

response_stat_fn = np.mean
# NOTE: __name__ of this ends up being "amax" instead of "max"
# (but it's just "mean" if `response_stat_fn = np.mean`)
#response_stat_fn = np.max

# TODO TODO delete (output does not make that much sense. also already have per-panel
# trace zscoring imlemented now, from cached raw traces. output there also doesn't make
# much sense)
#
# NOTE: currently will warn + exit before running model, if this is True. assuming for
# now I probably don't want to run model with these inputs. (but can't currently tell if
# cached responses were computed that way...)
zscore_traces_per_recording = False
#

# TODO TODO pickle these to each dir i save response stats in, and recompute if
# there is a mismatch
# TODO also include one_baseline_per_odor and other less-used params in here, at least
# if they have values other than typical? (or include in dict always, but only include
# in str in that case)
response_calc_params = dict(
    n_volumes_for_response=n_volumes_for_response,
    response_stat_fn=response_stat_fn,
    median_for_baseline=median_for_baseline,
    zscore_traces_per_recording=zscore_traces_per_recording,
)
response_calc_str = '__'.join(
    f"{k.replace('_', '-')}={getattr(v, '__name__', v)}".replace('=', '_')
    for k, v in response_calc_params.items()
)

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

# TODO doc (/delete / move to hardcode where this is used)
links_to_input_dirs = True

# TODO shorten any remaining absolute paths if this is True, so we can diff outputs
# across installs w/ data in diff paths
print_full_paths = False

# Overall folder structure should be: <driver>_<indicator>/<plot_fmt>/...
across_fly_ijroi_dirname = 'ijroi'
across_fly_pair_dirname = 'pairs'
across_fly_diags_dirname = 'glomeruli_diagnostics'

trial_and_frame_json_basename = 'trial_frames_and_odors.json'
# TODO even if i'm not going to support having multiple versions of this (and all
# other affected per-recording pickles, which is most of them, including
# best_plane_rois.p & trial_response_volumes.p), store which params were used to compute
# response statistics per-recording too, so i can include in some summary outputs at
# end? currently want for comparing diff response stats in GH146 data case
ij_trial_df_cache_basename = 'ij_trial_df_cache.p'

mocorr_concat_tiff_basename = 'mocorr_concat.tif'

all_frame_avg_tiff_basename = 'mocorr_concat_avg.tif'

# NOTE: trial_dff* is still a mean within a trial, but trialmean_dff* is a mean OF THOSE
# MEANS (across trials)
trial_dff_tiff_basename = 'trial_dff.tif'
trialmean_dff_tiff_basename = 'trialmean_dff.tif'
max_trialmean_dff_tiff_basename = 'max_trialmean_dff.tif'
min_trialmean_dff_tiff_basename = 'min_trialmean_dff.tif'

# TODO is there a 'gray_r'? try that?
# NOTE: 'Greys' is reversed wrt 'gray' (maybe not exactly, but it is white->black),
# and I wanted to try it to be more printer friendly, but at least without additional
# tweaking, it seemed a bit harder to use to see faint ROIs.
#anatomical_cmap = 'Greys'
anatomical_cmap = 'gray'

# TODO replace some/all uses of these w/ my own diverging_cmap_kwargs?
# (since we aren't using my generated plots for any of these anyway...)
#
# NOTE: not actually sure she will switch to 'vlag' (from 'RdBu_r'), though I have
# in diverging_cmap.
remy_corr_matshow_kwargs = dict(cmap='RdBu_r', vmin=-1, vmax=1, fontsize=10.0)

if response_stat_fn.__name__ == 'mean':
    # TODO want to avoid just 'mean', to avoid confusion w/ 'mean ' prefixes that might
    # get prepended, once we start dealing with means over either trials or flies?
    # would get pretty verbose...
    # (currently, this just refers to mean over timepoints in a response window, used to
    # get one number for a given unit [e.g. glomerulus, pixel, etc] and trial)
    trial_stat_desc = 'mean'

# currently, np.max seems to have __name__ of amax, but checking both in case that isn't
# always true.
elif response_stat_fn.__name__.isin('max', 'amax'):
    trial_stat_desc = 'peak'
else:
    assert False, f'unrecognized {response_stat_fn.__name__=}'

if not zscore_traces_per_recording:
    response_desc = f'{trial_stat_desc} {dff_latex}'
else:
    # TODO TODO also mention the additional baseline subtraction step? add delta symbol?
    response_desc = f'{trial_stat_desc} Z-scored F'

# never saying "mean mean <x>", just specifying outer mean separately when
# trial_stat_desc == 'max'
if trial_stat_desc != 'mean':
    # going to make no effort to clarify whether it's a mean over trials or flies.
    # should be clear from plot, and will get too verbose + create too many variables
    mean_response_desc = f'mean {response_desc}'
else:
    mean_response_desc = response_desc

# TODO check there are no cases that want to use trial_stat_fn, rather than
# unconditionally using mean, but also while unconditionally using dF/F instead of
# Z-scored F. if there are, prob want another variable for them.
mean_dff_desc = f'mean {dff_latex}'

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

    cbar_label=mean_dff_desc,

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

    # TODO check this is actually working
    # https://stackoverflow.com/questions/35099130
    # TODO TODO is this not working?
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
    # TODO TODO just draw small white outline around this / others? how?
    # https://matplotlib.org/stable/gallery/color/named_colors.html
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

    cbar_label=mean_response_desc,

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

gsheet_df = None

def get_gsheet_metadata() -> pd.DataFrame:
    """Downloads and formats Google Sheet experiment/fly metadata.

    Loads 'metadata_gsheet_link.txt' from directory containing this script, which should
    contain the full URL to your metadata Google Sheet.

    Most important columns in this sheet are:
    - 'Date': YYYY-MM-DD format dates for when experiments were conducted

    - 'Fly': integers counting up from 1, numbering flies within each date

    - 'Driver': the driver being used to drive indicator expression in this fly
       (e.g. 'pebbled' for pebbled-Gal4, our standard all-ORN driver)

    - 'Indicator': abbrevation for indicator the fly is expressing
       (e.g. '6f' for UAS-GCaMP6f)

    - 'Exclude': a checkbox-column where a check indicates the analysis should not be
       run on this experiment

    - 'Side': values should all be either 'right'|'left'|empty (I use a dropdown to
       enforce this). My recordings have all been imaging only one hemisphere of the
       brain at a time (either the left or the right), but we want to flip them all into
       a standard orientation to make the spatial patterns more easily comparable across
       experiments. All recordings will be flipped to `standard_side_orientation`, if
       not already in that orientation.

    My sheet is called 'tom_antennal_lobe_data' in the Hong lab Google Drive. Sam has
    his own. New users of the pipeline should probably start by copying one of ours, to
    get the right column names, data validation, etc.
    """
    script_dir = Path(__file__).resolve().parent

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
    df = util.gsheet_to_frame('metadata_gsheet_link.txt', normalize_col_names=True,
        # so that the .txt file can be found no matter where we run this code from
        # (hong2p defaults to checking current working dir and a hong2p root)
        extra_search_dirs=[script_dir]
    )
    df.set_index(['date', 'fly'], verify_integrity=True, inplace=True)

    # Currently has some explicitly labelled 'pebbled' (for new megamat experiments
    # where I also have some some 'GH146' data), but all other data should come from
    # pebbled flies.
    df.driver = df.driver.fillna('pebbled')

    # TODO if i don't switch off 8m for the PN experiments, first fillna w/ '8m' for
    # GH146 flies
    df.indicator = df.indicator.fillna('6f')

    return df


def get_diag_status_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: as output by `get_gsheet_metadata`

    Returns subset of Google Sheet metadata indicating whether we saw responses for each
    (glomerulus, diagnostic-odor) pair (at least those with a corresponding
    checkbox-column in your metadata Google Sheet).

    Output columns should be all-lowercase glomeruli names, except the final
    'all_labelled' column (from "All labelled" in sheet), which is used to indicate the
    other values have actually been labelled for this fly (rather than not being
    entered). Values should be all boolean. False should be interpreted as a missing
    value if 'all_labelled' is False for a given fly.
    """
    # "Side" in my sheet (-> 'side' from `normalize_col_names=True`). All checkbox
    # diagnostic status columns should immediately *follow* this column.
    last_gsheet_col_before_glomeruli_diag_statuses = 'side'

    # "All labelled" in my sheet (-> 'all_labelled' after `normalize_col_names=True`
    # path in `gsheet_to_frame`). All checkbox diagnostic status columns should
    # immediately *preceed* this column.
    last_diag_status_col = 'all_labelled'

    first_glomeruli_diag_col_idx = list(df.columns
        ).index(last_gsheet_col_before_glomeruli_diag_statuses) + 1

    last_glomeruli_diag_col_idx = list(df.columns
        ).index(last_diag_status_col)

    # This subset of Google Sheet data should should all be empty or checkboxes. Each
    # column name in this range of the sheet (except the last) should be in the format:
    # "<glomerulus-name> (<cognate-diagnostic-odor-abbrev> <diag-log10-conc>)", e.g.
    # "DM4 (MA -7)", "DL5 (t2h -6)", etc.
    #
    # The last column, named "All labelled" in my sheet (-> 'all_labelled' by the time
    # we are done defining this dataframe), is for indicating that any unchecked boxes
    # in that row can be interpreted as the experiment actually missing that
    # glomerulus's response to its diagnostic, rather than that we just hadn't checked
    # the box yet.
    glomeruli_diag_status_df = df.iloc[
        :, first_glomeruli_diag_col_idx:(last_glomeruli_diag_col_idx + 1)
    ]
    # Column names should be lowercased names of target glomeruli after this.
    glomeruli_diag_status_df.rename(columns=lambda x: x.split('_')[0], inplace=True)

    # Since I called butanone's target glomerulus VM7 in my olfactometer config, but
    # updated the name of the column in the gsheet to VM7d, as that is the specific part
    # it should activate.
    glomeruli_diag_status_df.rename(columns={'vm7d': 'vm7', 'all': 'all_labelled'},
        inplace=True
    )

    assert (glomeruli_diag_status_df.dtypes == bool).all()

    return glomeruli_diag_status_df


# TODO delete? do i have another mechanism to deal with this now tho? not a very common
# issue...
#
# For cases where there were multiple glomeruli diagnostic experiments (e.g. both sides
# imaged, and only one used for subsequent experiments w/in fly). Paths (relative to
# data root) to the recordings not followed up on / representative should go here.
unused_glomeruli_diagnostics = (
    # glomeruli_diagnostics_otherside is the one used here
    '2021-05-25/2/glomeruli_diagnostics',
)
#

if analyze_pairgrids_only:
    analyze_glomeruli_diagnostics = False

frame_assign_fail_prefix = 'assign_frames'
suite2p_fail_prefix = 'suite2p'

spatial_dims = ['z', 'y', 'x']
fly_keys = ['date', 'fly_num']


checks = True

# TODO probably make another category or two for data marked as failed (in the breakdown
# of data by pairs * concs at the end) (if i don't refactor completely...)
retry_previously_failed = False

# TODO TODO probably delete this + pairgrid only stuff. leave diagnostic only stuff for
# use on acquisition computer.
analyze_glomeruli_diagnostics_only = False
print_skipped = False
verbose = False

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
experiment2method_data = dict()

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


# TODO let root of this be overridden by env var? so i can put it on external hard
# drive if i'm running on my space limited laptop
def get_plot_root(driver, indicator) -> Path:
    return driver_indicator_output_dir(driver, indicator) / plot_fmt


def fly2driver_indicator(date, fly_num) -> Tuple[str, str]:
    """Returns tuple with driver and indicator of fly, via Google sheet metadata lookup
    """
    # NOTE: gsheet_df currently actually defined in main
    assert gsheet_df is not None
    fly_row = gsheet_df.loc[(pd.Timestamp(date), int(fly_num))]
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
        # TODO TODO fix how this failed on gh146 data (when re-analyzed 2025):
        # ./al_analysis.py -d GH146 -n 6f -t 2023-06-22 -e 2023-07-28 -s intensity,corr -v -i ijroi
        # (no odors actually in redone_odors below tho, so inconsequential in that case)

        # TODO TODO would this also fail on pebbled data if i were to re-analyze it
        # (probably)?
        try:
            assert fly_df.index.names[2:4] == ['odor1', 'odor2']
        # TODO delete
        except AssertionError:
            print()
            print(f'{fly_df.index.names=}')
            print(f'{fly_df.index.names[2:4]=}')
            warn('ASSERTION FAILED. FIX.')
            #import ipdb; ipdb.set_trace()
        #

        recordings = fly_df.columns.get_level_values('thorimage_id').unique()

        for panel_odor in redone_odors:
            recording_has_curr_odor = recording_has_odor.loc[panel_odor]
            final_recording = recording_has_curr_odor[::-1].idxmax()

            # TODO TODO how to warn about which ones we are tossing this way tho???
            nonfinal_recordings= (
                fly_df.columns.get_level_values('thorimage_id') != final_recording
            )

            panel = panel_odor[0]
            # TODO TODO TODO update this line to not depend on specific indices (or at
            # least compute them earlier based on position of odor1[[/odor2]/etc] levels
            # NOTE: ignoring 'repeat' (which might be different across recordings in
            # someone elses use, but isn't in any of my data)
            odors = panel_odor[2:4]
            # TODO delete
            if 'odor2' not in fly_df.index.names:
                print()
                print(f'{panel_odor=}')
                print(f'{odors=}')
                import ipdb; ipdb.set_trace()
            #

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
    #
    # group_keys=False to preserve old behavior (and silence FutureWarning, after
    # upgrading pandas to 1.5.0 from 1.3.?)
    df = df.groupby(['date','fly_num'], axis='columns', sort=False, group_keys=False
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
        return fly_df.groupby(group_fn, axis='columns', sort=False, dropna=False,
            group_keys=False).apply(_single_flyroi_drop)

    # group_keys=False to preserve old behavior (and silence FutureWarning, after
    # upgrading pandas to 1.5.0 from 1.3.?)
    df = df.groupby(['date','fly_num'], axis='columns', sort=False, group_keys=False
        ).apply(single_fly_drop_uncertain_if_we_have_certain)

    # this is what will actually reduce the size along the column axis
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
            print(f'checking for {shorten_path(tiff_path, n=4)}...', end='')

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


# TODO delete?
# TODO factor to hong2p.util (along w/ get_analysis_dir, removing old analysis dir
# stuff for it [which would ideally involve updating old code that used it...])
def load_movie(*args, **kwargs):

    tiff_path_or_thorimage_dir = find_movie(*args, **kwargs)

    if tiff_path_or_thorimage_dir.name.endswith('.tif'):
        return tifffile.imread(tiff_path_or_thorimage_dir)
    else:
        assert tiff_path_or_thorimage_dir.is_dir()
        return thor.read_movie(tiff_path_or_thorimage_dir)


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


# TODO check this behaves as verbose=True
# (esp if that fn already has verbose kwarg in natmix. want to test that case)
write_corr_dataarray = produces_output(_write_corr_dataarray)

# TODO TODO also use wrapper for TIFFs (w/ [what should be default] verbose=True)
# (wrap util.write_tiff, esp if that fn already has verbose kwarg. want to test that
# case)

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
    """Deletes empty directories in `al_util._dirs_to_delete_if_empty`
    """
    for d in set(al_util._dirs_to_delete_if_empty):
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
        for d in al_util._dirs_to_delete_if_empty:
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
        # TODO delete this first branch? pretty sure i haven't been using it, as the
        # write_text call never failed for err_str not being defined... (before i added
        # the `return` here)
        path.touch()
        return

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
    # TODO doc

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
def delta_f_over_f(movie_length_array, bounding_frames, *,
    n_volumes_for_baseline: Optional[int] = n_volumes_for_baseline,
    median_for_baseline: bool = median_for_baseline,
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

        median_for_baseline: if True, use compute baseline with median (w/ default
            [presumably linear] interpolation) rather than mean

        exclude_last_pre_odor_frame: whether to exclude the last frame before odor onset
            when calculating each trial's baseline. can help hedge against off-by-one
            errors in frame<->trial assignment, but might otherwise dilute the signal.

        one_baseline_per_odor: if False, baselines each trial to pre-odor period for
            that trial. if True (requires `odor_index != None`), baselines to pre-odor
            period of first trial of that odor (which may be a previous trial,
            potentially much before).

        odor_index: currently just used for `one_baseline_per_odor=True` path, and not
            added to output metadata.

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

    # TODO if passed, use odor_index to add that metadata to outputs?

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

        # TODO explicitly mean over time dimension if input is xarray
        # (or specify all other dimensions, if that's how to make it work)
        if not median_for_baseline:
            baseline = for_baseline.mean(axis=0)
        else:
            if len(for_baseline) <= 2:
                warn(f'taking median of only {len(for_baseline)} timepoints for '
                    'baseline. equivalent to mean (median_for_baseline=True) without '
                    'more baseline timepoints!'
                )

            # TODO delete (can it even be anything other than a DataFrame?)
            if not isinstance(for_baseline, pd.DataFrame):
                # TODO test! if np.ndarray, may need np.quantile(<x>, 0.5) or something.
                # (test on xarray too?)
                import ipdb; ipdb.set_trace()
            #
            baseline = for_baseline.median(axis=0)

        # TODO support region defined off to side of movie (that should not have
        # signal), to use to subtract before any other calculations?

        # TODO delete + add quantitative check baseline isn't too close to zero (is
        # this currently an issue in practice for me at all? only on
        # [essentially non-important] pixelwise stuff?)
        #
        #print('baseline.min():', baseline.min())
        #print('baseline.mean():', baseline.mean())
        #print('baseline.max():', baseline.max())
        #
        # hack to make df/f values more reasonable
        # TODO still an issue?
        # TODO TODO maybe just add like 1e4 or something?
        #baseline = baseline + 10.

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


# TODO type hint
def trial_response_traces(raw_traces, bounding_frames, *, odor_index=None,
    # TODO delete zscore_traces_per_recording default (replace w/ `= False`)
    zscore: bool = zscore_traces_per_recording, keep_pre_odor: bool = False,
    n_volumes_for_baseline: Optional[int] = n_volumes_for_baseline, _checks=False,
    **dff_kws):
    # TODO doc
    """
    Args:
        zscore: if False, use `delta_f_over_f`. otherwise, Z-score input along time
            dimension (and then subtract a pre-odor baseline from each trial).

        n_volumes_for_baseline: see `delta_f_over_f` doc. same usage. in `zscore=True`
            case, this is used for baseline-subtracting after Z-scoring, as Remy does.

        keep_pre_odor: see `delta_f_over_f` doc. same usage.

        odor_index: see `delta_f_over_f`. only used in that case (when `zscore=False`).

        **dff_kws: passed to `delta_f_over_f`. incompatible with `zscore=True`.
    """
    # NOTE: _checks=True is set in test_al_analysis.test_trial_response_traces

    if not zscore:
        yield from delta_f_over_f(raw_traces, bounding_frames, odor_index=odor_index,
            keep_pre_odor=keep_pre_odor, n_volumes_for_baseline=n_volumes_for_baseline,
            **dff_kws
        )
        # do need this empty `return` after `yield from`, or else execution will
        # continue
        return

    # may not be able to rely on odor_index being useful here, as don't actually use in
    # other branch now (outside of unused one_baseline_per_odor=True)

    try:
        # checking these module-level vars b/c these are not passed thru calls, but only
        # set via delta_f_over_f defaults
        assert not median_for_baseline
        assert not exclude_last_pre_odor_frame
        assert not one_baseline_per_odor
        # TODO delete. think i want these ones
        #assert n_volumes_for_baseline is None

        # TODO or need to check any values? (would be same as assertions above, if so)
        assert len(dff_kws) == 0

    except AssertionError as err:
        # TODO add info saying incompatible args w/ zscore=True (+ raise as ValueError?)
        raise

    zscored = scipy_zscore(raw_traces, axis=0)

    if _checks:
        first_roi_timeseries = raw_traces.iloc[:, 0]
        assert len(first_roi_timeseries) == len(raw_traces)

        # TODO delete (or make more specific, so as not to fail if input is ndarray)
        # TODO ever any other index names? remove? or just check it's not 'roi', if
        # there is a .index.name?
        #assert first_roi_timeseries.index.name == 'frame'

        assert pd_allclose(zscored.iloc[:, 0], scipy_zscore(first_roi_timeseries))

    # TODO can i share any of this from delta_f_over_f (copied and adapted from there)
    for i, (start_frame, first_odor_frame, end_frame) in enumerate(bounding_frames):

        # TODO flag to disable this baseline subtracting?
        if not keep_pre_odor:
            trial_traces = zscored[first_odor_frame:(end_frame + 1)]
        else:
            trial_traces = zscored[start_frame:(end_frame + 1)]

        # NOTE: this is the frame *AFTER* the last frame included in the baseline
        baseline_afterend_frame = first_odor_frame
        baseline_start_frame = start_frame

        if n_volumes_for_baseline is not None:
            baseline_start_frame = baseline_afterend_frame - n_volumes_for_baseline
            assert baseline_start_frame < first_odor_frame

        for_baseline = zscored[baseline_start_frame:baseline_afterend_frame]
        baseline = for_baseline.mean(axis=0)

        baseline_subtracted = trial_traces - baseline

        if _checks and not keep_pre_odor:
            stat = response_stat_fn
            if n_volumes_for_response is None:
                for_r1 = trial_traces
                for_r2 = baseline_subtracted
            else:
                for_r1 = trial_traces[:n_volumes_for_response]
                for_r2 = baseline_subtracted[:n_volumes_for_response]

            r1 = stat(for_r1, axis=0) - baseline
            r2 = stat(for_r2, axis=0)

            # TODO pd_allclose work w/ non-pandas input? need to support any of that in
            # this fn?
            #
            # NOTE: this will likely NOT be true if this path ever uses a statistic
            # other than mean for computing baseline (e.g. median, which I had briefly
            # tried in zscore=False path)
            #
            # so order of baseline subtracting and mean-in-response-window don't matter,
            # and we don't need to do the baseline subtracting in compute_response_stats
            assert pd_allclose(r1, r2)

        yield baseline_subtracted


# TODO type hint
# TODO maybe also try z-scoring on a trial or odor basis?
def compute_trial_stats(raw_traces, bounding_frames,
    odor_order_with_repeats: Optional[ExperimentOdors] = None, *,
    # TODO special case so it's mean by default for pebbled (to better capture
    # inhibition), and max by default for GH146? (b/c PN spontaneous activity. this make
    # sense? was it max and not mean that worked for me for GH146? maybe it was the
    # other way around?)
    # TODO check GH146 correlations again to see which looked better: max or mean (and
    # maybe doesn't matter on new data?) (paper GH146 outputs were using same
    # `n_volumes_for_response=2, stat=mean` as everything else)
    # TODO delete zscore_traces_per_recording default (replace w/ `= False`)
    stat: Callable = response_stat_fn, zscore: bool = zscore_traces_per_recording,
    n_volumes_for_response: Optional[int] = n_volumes_for_response):
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
        # TODO fail here if one_baseline_per_odor=True
        # (will fail below regardless)
        index = None
    else:
        index = odor_lists_to_multiindex(odor_order_with_repeats)

    # TODO return as pandas series if odor_order_with_repeats is passed, with odor
    # index containing that data? test this would also be somewhat natural in 2d/3d case

    trial_stats_list = []
    for trial_traces in trial_response_traces(raw_traces, bounding_frames,
        odor_index=index, zscore=zscore):

        if n_volumes_for_response is None:
            for_response = trial_traces
        else:
            for_response = trial_traces[:n_volumes_for_response]

        try:
            curr_trial_stats = stat(for_response, axis=0)

        except TypeError as err:
            # TODO better way to get err msg?
            err_msg = str(err)

            # only trying to catch errors like:
            # TypeError: 'axis' is an invalid keyword argument for max()
            assert str(err).startswith("'axis' is an invalid keyword argument for ")

            # if stat doesn't accept axis=0 kwarg, we'll try without that
            curr_trial_stats = stat(for_response)

        # TODO adapt to also work in case input is a movie (done? test)
        # TODO also work in 1d input case (i.e. if just data from single ROI was passed)
        #
        # raw_traces.shape[1] == # of ROIs
        assert curr_trial_stats.shape == (raw_traces.shape[1],)

        trial_stats_list.append(curr_trial_stats)

    trial_stats = np.stack(trial_stats_list)

    # TODO why are we passing index here and also to delta_f_over_f above? maintain that
    # information when appending in list, rather than re-adding here? (odor_index=index
    # currently only used by delta_f_over_f for one_baseline_per_odor=True path, and not
    # to add that metadata to outputs)
    trial_stats_df = pd.DataFrame(index=index, data=trial_stats)
    trial_stats_df.index.name = 'trial'

    # TODO maybe implement somthing that also works w/ xarrays? maybe make my own
    # function that dispatches to the appropriate concatenation function accordingly?

    # Since np.stack (probably as well as other similar numpy functions) converts pandas
    # stuff to numpy, and pd.concat doesn't work with numpy arrays.
    if hasattr(raw_traces, 'columns'):
        trial_stats_df.columns = raw_traces.columns
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


# TODO factor to a format_time fn (hong2p.util?)?
# TODO probably switch to just using one format str...
# TODO include seconds too (kwarg for it?)?
def format_time(t: Union[datetime, pd.Timestamp]) -> str:
    # TODO doc w/ example
    return f'{format_date(t)} {t.strftime("%H:%M")}'


def names2fname_prefix(name1, name2):
    return util.to_filename(f'{name1}_{name2}'.lower(), period=False)


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
    """Plots odor x ROI data displayed with odors as columns and ROI means as rows.

    Args:
        trial_df: ['odor1', 'odor2', 'repeat'] index names and a column for each ROI.
            ['odor2', 'repeat'] are optional, and 'odor' may be used in place of
            'odor1'.

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
    if 'odor' in trial_df.index.names:
        assert 'odor1' not in trial_df.index.names
        odor_var = 'odor'
    else:
        assert 'odor1' in trial_df.index.names
        odor_var = 'odor1'

    # TODO also check ROI index (and also factor that to hong2p)
    # TODO maybe also support just 'fly' on the column index (where plot title might be
    # the glomerulus name, and we are showing all fly data for a particular glomerulus)

    avg_levels = [odor_var]
    # TODO handle in a way agnostic to # of components? e.g. supporting also 'odor3',
    # etc, if present
    if 'odor2' in trial_df.index.names:
        avg_levels.append('odor2')

    # TODO unsupport keep_panels_separate=False?
    if keep_panels_separate and 'panel' in trial_df.index.names:
        # TODO TODO TODO warn/err if any null panel values. will silently be dropped as
        # is.
        # TODO or change fn to handle them gracefully (sorting alphabetically w/in?)
        avg_levels = ['panel'] + avg_levels

    if trial_df.index.name == odor_var:
        # assuming input is mean already columns probably are still just 'roi', as I
        # assume is also true in most cases below (as we are only ever computing
        # groupby->mean across row groups in this fn)
        mean_df = trial_df.copy()
    else:
        avg_levels = [x for x in avg_levels if x in trial_df.index.names]

        if not avg_repeats:
            assert 'repeat' in trial_df.index.names
            assert 'repeat' not in avg_levels
            avg_levels.append('repeat')

        # This will throw away any metadata in multiindex levels other than these
        # (so can't just add metadata once at beginning and have it propate through
        # here, without extra work at least)
        mean_df = trial_df.groupby(avg_levels, sort=False).mean()

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

        # TODO factor this box drawing into some hong2p.viz fn?
        # (use for some plots of sensitivity analysis, to highlight the tuned param
        # combo stepped around? like the one in here, or the one in
        # natmix_data/analysis.py?)
        for combo in odor_glomerulus_combos_to_highlight:
            odor = combo[odor_var]
            roi = combo['glomerulus']

            matching_roi = mean_df.index.get_level_values('roi') == roi

            matching_odor = mean_df.columns.get_level_values(odor_var) == odor
            if 'odor2' in mean_df.columns.names:
                matching_odor &= (
                    mean_df.columns.get_level_values('odor2') == solvent_str
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
    #
    # error is plotted below (currently stddev), so returning early without multiple
    # flies
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


full_traces_cache_name = 'full_traces.p'
best_plane_traces_cache_name = 'best_plane_traces.p'

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

    # TODO change fn to preserve input type (w/ the metadata xarrays / dfs have)
    traces = pd.DataFrame(extract_traces_bool_masks(movie, full_rois))
    traces.index.name = 'frame'
    traces.columns.name = 'roi'

    # TODO also try merging via correlation/overlap thresholds?

    # TODO maybe also ~z-score before picking best plane (dividing by variability in
    # baseline period first)
    trial_dff = compute_trial_stats(traces, bounding_frames, odor_lists)
    roi_quality = trial_dff.max()

    # TODO TODO refactor so all this calculation is done across all recordings
    # within each fly, rather than just within each recording
    # (concatenated across movies, done after all the process_recording calls)
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

    fullrois_nonoutline_mask = np.array(
        [not is_ijroi_plane_outline(x) for x in full_traces.columns]
    )
    # TODO delete if i don't end up using
    #
    # TODO check this works (only fly w/ plane outlines currently 2023-05-10/1)!
    full_traces = full_traces.loc[:, fullrois_nonoutline_mask]
    assert set(traces.columns) == set(full_traces.columns)
    #

    assert np.array_equal(
        full_rois.roi_name.values[fullrois_nonoutline_mask], full_traces.columns
    )
    full_traces.columns = pd.MultiIndex.from_arrays([full_traces.columns,
        pd.Series(full_rois.roi_z.values[fullrois_nonoutline_mask], name='z')
    ])
    # TODO TODO try picking planes directly from concatenated full_traces.p contents?
    # don't actually need ROIs themselves to later pick the best plane, no?
    to_pickle(full_traces, analysis_dir / full_traces_cache_name)

    # TODO also return these, so i don't need to load them all from disk cache?
    to_pickle(traces, analysis_dir / best_plane_traces_cache_name)

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


def trace_plots(raw_traces, z_indices, bounding_frames, odor_lists, roi_plot_dir,
    main_plot_title, *, skip_all_plots=skip_singlefly_trace_plots, roi_stats=None,
    show_suite2p_rois=False):
    """
    Args:
        skip_all_plots: if True, only compute and return ROI x mean trial dF/F
            dataframe. makes no plots.
    """
    if show_suite2p_rois and roi_stats is None:
        raise ValueError('must pass roi_stats if show_suite2p_rois')

    # mean dF/F for each ROI x trial
    trial_df = compute_trial_stats(raw_traces, bounding_frames, odor_lists)

    if skip_all_plots:
        return trial_df

    # TODO update to pathlib
    makedirs(roi_plot_dir)

    # TODO TODO TODO plot raw responses (including pre-odor period), w/ axvline for odor
    # onset
    # TODO TODO plot *MEAN* (across trials) timeseries responses (including pre-odor
    # period), w/ axvline for odor onset
    # TODO concatenate these DataFrames into a DataArray somehow -> serialize
    # (for other people to analyze)?
    # TODO TODO raw version of these, for troubleshooting consequeces params that
    # go into response stat calculation (in particular regarding variable inhibition
    # seen frequently in what-should-be-final paper GH146 6f data)
    # TODO add [dotted?] line for end of stimulus (or at least end of volumes we
    # will use for response calc)
    timeseries_plot_dir = roi_plot_dir / 'timeseries'
    # TODO update to pathlib
    makedirs(timeseries_plot_dir)

    # TODO TODO factor out this timeseries plotting to a fn

    # TODO TODO don't hardcode number of trials? do i elsewhere?
    n_trials = 3

    # TODO delete this debug path after adding a quantitative check that there isn't an
    # offset in any odor onset frames
    debug = False
    #debug = True

    # TODO probably cluster to order rows (=rois?) by default (just use cluster_rois if
    # so? already calling that in here below...)

    odor_index = odor_lists_to_multiindex(odor_lists)

    # NOTE: trial_response_traces zscore= currently only set via default (which is set
    # to module-level zscore_traces_per_recording)
    traces = list(trial_response_traces(raw_traces, bounding_frames, keep_pre_odor=True,
        odor_index=odor_index
    ))
    # this loop (and everything below) is just for plotting timeseries traces, and does
    # not affect what will be returned (trial_df)
    curr_odor_i = 0
    first_odor_frames = None
    axs = None
    # TODO if i'm not gonna concat traces into some xarray thing, maybe just use
    # generator in zip rather than converting to list first above?
    for trial_traces, trial_bounds, trial_odors in zip(traces, bounding_frames,
        odor_index):

        start_frame, first_odor_frame, _ = trial_bounds

        trial_odors = dict(zip(odor_index.names, trial_odors))
        repeat = trial_odors['repeat']

        if repeat > 2:
            warn('trace_plots (timeseries plot): skipping repeat > 2!')
            continue

        if repeat == 0:
            # NOTE: layout kwarg not available in older (< ~3.6) versions of matplotlib
            fig, axs = plt.subplots(nrows=1, ncols=n_trials, layout='constrained',
                # TODO lower dpi when not debugging (setting high so i can have a chance
                # at reading xticklabel frame numbers)
                dpi=600 if debug else 300, figsize=(6, 3)
            )
            first_odor_frames = []

        assert first_odor_frames is not None
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

        assert trial_traces.index.name == 'frame'
        # ('roi' = glomerulus)
        assert trial_traces.columns.name == 'roi'

        # TODO TODO why are these only showing 3 frames before onset in megamat
        # data? shouldn't there be 7 / 1.6 = 4.375 -> floor -> 4?

        ax = axs[repeat]

        # TODO see what 2023-11-21/2 B-myr inh looks like  in timeseries, now that
        # diverging cmap kwargs (w/ two slow norm) being used
        _, im = viz.matshow(trial_traces.T, ax=ax,

            vline_level_fn=vline_level_fn, linecolor='k',

            # NOTE: these are int frame indices, not resetting across trials (so first
            # index on trial 0 might be 0, but on trial 1 it might then be 19 [but must
            # be >0, and 1 + index of last frame in trial 0])
            xticklabels=True,

            # TODO TODO TODO work? (no!!!) how else to show glomeruli labels?
            yticklabels=True,

            fontsize=2.0 if debug else None, xtickrotation='vertical',

            # TODO TODO easier if i just group all 3 trials into adjacent rows, then
            # only have one Axes?
            # TODO fix (still? try again w/o this)
            # viz.matshow colorbar behavior broken for ax=<Axes from a subplot array>,
            # now that viz.matshow forces constrained layout
            colorbar=False,
            # TODO define from percentile of data instead? on 2023 GH146 data, many
            # cases (e.g. 2023-06-22/2 megamat1 aa, and it's not just aa, but other
            # odors too) had overwhelming negative responses, so can't just use data
            # limits
            # TODO TODO (2025) still broken? maybe?
            #vmin=dff_vmin, vmax=dff_vmax,
            # TODO TODO TODO use other limits (or just don't specify) if uzing zscored
            # responses (diff scales)?
            vmin=-dff_vmax, vmax=dff_vmax,

            **diverging_cmap_kwargs
        )
        # TODO probably delete (use similar code for end of stimulus tho?)
        #vline = (first_odor_frame - start_frame - 1) + 0.5
        #ax.axvline(vline, linewidth=0.5, color='k')

        if not debug:
            ax.xaxis.set_ticks([])
            # TODO if i want to show, convert frames->seconds (might need to pass some
            # other info in...)? or subtract first frame index in each trial from other
            # indices?

        # TODO TODO group trials as row group, rather than having separate facet for
        # each trial?

        assert repeat <= 2, 'need to update conditional below if this trips'
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
        odor_sort=is_pair, title=main_plot_title, cbar_label=mean_response_desc,
        cbar_shrink=0.4
    )
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
        plot_roi_stats_odorpair_grid(roi1_series, cbar_label=mean_response_desc,
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

            # TODO define ops -> uncomment line below (currently undefined in commented
            # line below. not using this code now anyway)
            raise NotImplementedError
            #s2p.plot_roi(roi_stat, ops, ax=axs[1])

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


# TODO delete? not sure this is used in any of my current registration pipeline...
#
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

glomeruli_diag_status_df = None
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
    # TODO move after update_globals_from_shared_state? may want to just delete that
    # bit, as well as multiprocessing option it was for anyway... haven't used in a long
    # time
    global gsheet_df
    global glomeruli_diag_status_df
    if glomeruli_diag_status_df is None:
        assert gsheet_df is not None
        glomeruli_diag_status_df = get_diag_status_df(gsheet_df)
    #

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
    # TODO TODO use for computing appropriate n_volumes_for_response?
    # TODO refactor to share w/ hong2p.suite2p.suite2p_params?
    volumes_per_second = single_plane_fps / (z + n_flyback)
    method_data = {
        # NOTE: w/o splitting up 'xy' tuple, pd.concat later would have 'xy'=[198, 198],
        # with all other columns duplicated twice.
        'x': xy[0],
        'y': xy[1],

        'z': z,
        'c': c,
        'n_flyback': n_flyback,
        'single_plane_fps': single_plane_fps,
        'volumes_per_second': volumes_per_second,
    }

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

    def _timing_param_in_seconds(key):
        val_s = float(int(yaml_data['settings']['timing'][key]) / 1e6)
        assert val_s > 0
        return val_s

    pre_pulse_s = _timing_param_in_seconds('pre_pulse_us')
    # TODO TODO use to automatically pick a reasonable n_volumes_for_response?
    # (in combination w/ volumetric sampling rate)
    pulse_s = _timing_param_in_seconds('pulse_us')
    post_pulse_s = _timing_param_in_seconds('post_pulse_us')

    method_data.update({
        'pre_pulse_s': pre_pulse_s,
        'pulse_s': pulse_s,
        'post_pulse_s': post_pulse_s,
    })

    # TODO delete? which flies were affected
    if pulse_s != 3:
        # TODO delete (or delete containing block). just to check i can delete
        # containing block w/o affecting things.
        # TODO TODO TODO re-run on all paper inputs first
        assert False

        warn_if_not_skipped(f'odor {pulse_s=} not the standard 3 seconds')

        # Remy started the config she gave me as 2s pulses, but we are now on 3s.
        # I might still want to average the 2s pulse data...
        if panel != 'megamat':
            print_skip(f'skipping because odor pulses were {pulse_s} (!= 3s) long',
                yaml_path
            )
            return
    #

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

    n_averaged_frames = thor.get_thorimage_n_averaged_frames_xml(xml)
    zstep_um = thor.get_thorimage_zstep_um(xml)
    xy_pixelsize_um = thor.get_thorimage_pixelsize_um(xml)
    # TODO consolidate all imaging params method_data into a hong2p fn that returns all
    # of them in a dict, without having to separately parse and add things (-> use to
    # get which keys are imaging params, to use for warning if any of them change w/in
    # fly?)? maybe even have it call parse_thorimage_notes? or at least move these
    # imaging param entries up above curr def of method_data?
    method_data['n_averaged_frames'] = n_averaged_frames
    method_data['zstep_um'] = zstep_um
    method_data['xy_pixelsize_um'] = xy_pixelsize_um
    # TODO get thorimage power regulator setting (to use later to invalidate ffilling
    # power measurement past this, for systems where regulator has hysteresis, like
    # downstairs non-pockel one)
    # TODO also invalidate if PowerRegulator/offset changes?

    x_fov_um = method_data['x'] * xy_pixelsize_um
    y_fov_um = method_data['y'] * xy_pixelsize_um

    lsm = xml.find('LSM').attrib
    width_um = float(lsm['widthUM'])
    height_um = float(lsm['heightUM'])
    # TODO and is width = x and height = y? test w/ recording where x != y
    # TODO move checks bellow into new thor fn(s) to get fov sizes?
    # NOTE: there is actually a small difference. not sure which might be more accurate.
    # the difference is probably smaller than anything that might matter.
    # e.g. width_um=95.58 vs x_fov_um=95.616
    assert np.isclose(width_um, x_fov_um, rtol=.001)
    assert np.isclose(height_um, y_fov_um, rtol=.001)
    del width_um, height_um

    method_data['x_fov_um'] = x_fov_um
    method_data['y_fov_um'] = y_fov_um

    # e.g. '3.0.2016.10131'
    thorimage_version = thor.get_thorimage_version(xml)
    method_data['thorimage_version'] = thorimage_version

    # TODO possible to tell upstairs vs downstairs system easily? (and beyond just
    # software version). prob not.
    # <User name="User" /> (lowercase 'user' is only difference between User/Computer in
    # Yang's output she sent me. same in what george sent me.)
    # <Computer name="USER-PC" />

    scanner = thor.get_thorimage_scannertype(xml)
    method_data['scanner'] = scanner

    # TODO delete after checking whether i can rule out some recordings accidentally not
    # using piezo? (='ThorStage' for all, so unless it was misconfigured for all...)
    # see comments around this thor fn tho, for tests i would need to do downstairs.
    #
    # TODO TODO maybe this was an error in me setting up this aquisition tho (check
    # other experiments? at least for `-d pebbled -n 6f -t 2023-04-22 -e 2023-06-22`,
    # zstage_type is all 'ThorStage'? thinking the downstairs system + software might
    # just always call it this? probably doesn't mean something isn't set up wrong...
    #
    # TODO really no reference to piezo (or multiple ztages even...) in xml? searched in
    # 2023-04-22/2/diagnostics1 xml and didn't see...
    # (in yang's output, there is a ZStage tag w/ name="ThorZPiezo", and that's only
    # explicit reference)
    # TODO Streaming/zFastEnable="1" enough to imply we are using a piezo (same question
    # also in a comment in get_thorimage_zstage_type fn def)?
    # NO! (at least it's possible to misconfigure downstairs system so that non-piezo
    # stage is selected for both primary/secondary, and it will still let you collect a
    # recording with fast Z apparently enabled)
    #
    # zstage_type can be either 'ThorStage' or 'ThorZPiezo', but so far seems that it
    # might always say 'ThorStage' from downstairs system, even if piezo was actually
    # what was being used.
    zstage_type = thor.get_thorimage_zstage_type(xml)
    method_data['zstage'] = zstage_type
    method_data['fast_z'] = thor.is_fast_z_enabled_xml(xml)
    # TODO care about Streaming/useReferenceVoltageForFastZPockels?

    # regtype currently either 'pockel' or 'non-pockel' (i.e. some kind of rotating
    # [half?  quarter?] waveplate, which i think is downstream from an element to select
    # only a particular polarization of the laser light)
    regtype, power_level = thor.get_thorimage_power_regtype_and_level(xml)
    method_data['power_regulator'] = regtype
    method_data['power_level'] = power_level

    # TODO care about <LSM ... dwellTime="2.4" />?

    # TODO delete
    #print()
    #print()
    #thor.print_xml(xml)
    #print()
    #print(f'{thorimage_dir=}')
    #import ipdb; ipdb.set_trace()
    #

    # can set debug=True for more information on parsing. mainly useful for making
    # changes to parse_thorimage_notes. debug=False should be the default.
    note_data = thor.parse_thorimage_notes(xml, debug=False)

    # important that method_data is not updated after we add note_data, as then it's
    # possible for key collisions/overwriting
    assert not any(k in method_data for k in note_data.keys())
    method_data.update(note_data)

    # NOTE: trying to move towards using 'experiment' to mean one (or more) recordings,
    # in a particular fly, with a common set of odors (whose presentations might be
    # split across recordings). 'recording' should now mean the output of a single Thor
    # acquisition run.
    experiment_key = (date, fly_num, panel, is_pair)

    if experiment_key not in experiment2recording_dirs:
        experiment2recording_dirs[experiment_key] = [(thorimage_dir, thorsync_dir)]
        experiment2method_data[experiment_key] = {thorimage_dir: method_data}
    else:
        experiment2recording_dirs[experiment_key].append((thorimage_dir, thorsync_dir))

        assert thorimage_dir not in experiment2method_data[experiment_key]
        experiment2method_data[experiment_key][thorimage_dir] = method_data
    del method_data

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
                        # (currently raising NotImplementedError below, since
                        # best_plane_rois not defined below in this case, but don't
                        # think this branch is getting hit)
                        warn(f'{best_plane_rois_cache} did not exist, but '
                            f'{full_rois_cache} did. can add `-i ijroi` to CLI args to '
                            'regenerate.'
                        )
                        raise NotImplementedError
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

            # TODO TODO refactor so all response computation goes through
            # compute_trial_stats (to ensure no divergence)? (maybe i want mean for this
            # tho, even if i might want max for some roi-trace stuff?)
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
        viz.add_colorbar(trial_heatmap_fig, im, label=dff_latex, shrink=0.32,
            extend='both'
        )

        suptitle(title, trial_heatmap_fig)
        exp_savefig(trial_heatmap_fig, plot_desc + '_trials')

        trialmean_dff_fig_path = plot_and_save_dff_depth_grid(avg_mean_dff, plot_desc,
            title=title, cbar_label=mean_dff_desc
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

    # TODO generate these from the min-of-mins and max-of-maxes TIFFs now
    # (or at least load across the individual TIFFs within each panel and compute within
    # there?)
    path = plot_and_save_dff_depth_grid(max_trialmean_dff,
        # TODO use mean_dff_desc in title?
        'max_trialmean_dff', title=f'max of trial-mean {dff_latex}',
        cbar_label=f'{dff_latex}'
    )
    # To see if strong inhibition ever helps quickly identify glomeruli
    # TODO use ~only-negative colorscale for min_trialmean_dff figure
    plot_and_save_dff_depth_grid(min_trialmean_dff,
        # TODO use mean_dff_desc in title?
        'min_trialmean_dff', title=f'min of trial-mean {dff_latex}',
        cbar_label=dff_latex
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
            print(f'{shorten_path(input_tiff, n=4)}: {start_idx + 1} - {end_idx}')

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


def should_flip_lr(date, fly_num, *, _warn=True) -> Optional[bool]:
    # NOTE: gsheet_df currently actually defined in main
    assert gsheet_df is not None

    # TODO unify w/ half-implemented hong2p flip_lr metadata key?
    try:
        # np.nan / 'left' / 'right'
        side_imaged = gsheet_df.loc[(date, fly_num), 'side']
        # expecting .loc above to give us a single value, not a range of rows.
        # (i.e. each (date, fly_num) combo should be unique to a single row)
        assert not hasattr(side_imaged, 'shape')
    except KeyError:
        side_imaged = None

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
    Will recompute if al_util.ignore_existing explicitly included 'json'.

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

        # TODO TODO (?) is it even possible for trial_bounding_frames.yaml to overwrite
        # this? need that to work, right? (for sam+george new experiments, where frame
        # assignment not working)
        #
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
                    err_msg = (f'{shorten_path(json_fname, n=5)} did not exist! '
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


# TODO rename just recompute_responses_per_panel?
def recompute_responses_from_traces_per_panel(keys_and_paired_dirs, *,
    zscore: bool = False, verbose: bool = False) -> List[pd.DataFrame]:
    """Loads cached raw traces, concatenates within (fly, panel), and returns responses.

    Loads `full_traces_cache_name` from each fly analysis dir. This must have already
    been computed and saved within a `process_recording` call (via the `'ijroi'`
    analysis path in there).

    Also re-computes best plane for each ROI, based on current method of response
    calculation used in here (for each panel, across all fly-recordings, picks plane
    with highest response calculation).

    Returns list of response dataframes, with one response statistic for each (fly,
    panel, trial). Output is in similar format to `ij_trial_dfs` used elsewhere,
    although one element per (fly, panel) instead of one element per recording.
    """
    # TODO share some /all of this w/ register_all... it was adapted from?

    date_and_fly2thor_dirs = defaultdict(list)
    for (date, fly_num), (thorimage_dir, thorsync_dir) in keys_and_paired_dirs:
        # Since keys_and_paired_dirs should have everything in chronological order,
        # these ThorImage directories should be in the order they were acquired in
        date_and_fly2thor_dirs[(date, fly_num)].append((thorimage_dir, thorsync_dir))

    pick_best_plane_here = True

    fly_panel_dfs = []
    for (date, fly_num), all_thor_dirs in date_and_fly2thor_dirs.items():

        assert len(all_thor_dirs) > 0

        date_str = format_date(date)
        fly_str = f'{date_str}/{fly_num}'
        fly_analysis_dir = get_fly_analysis_dir(date, fly_num)

        panel2traces = defaultdict(list)
        panel2bounding_frames = defaultdict(list)
        panel2odors = defaultdict(list)
        # need this to get things to concat w/o duplicates in columns (otherwise one fly
        # will have same ROI once for each recording)
        panel2thorimage_ids = defaultdict(list)

        for thorimage_dir, thorsync_dir in all_thor_dirs:
            panel = get_panel(thorimage_dir)
            if panel is None:
                # TODO warn
                continue

            analysis_dir = get_analysis_dir(date, fly_num, thorimage_dir)

            if pick_best_plane_here:
                full_traces_cache = analysis_dir / full_traces_cache_name
                if not full_traces_cache.exists():
                    continue
                traces = pd.read_pickle(full_traces_cache)

            # TODO TODO try computing w/ previous plane selection method, to
            # check that's not the reason new (z-score based) stuff looks noisier
            # (is it actually noisier?)
            else:
                best_plane_traces_cache = analysis_dir / best_plane_traces_cache_name
                if not best_plane_traces_cache.exists():
                    continue
                traces = pd.read_pickle(best_plane_traces_cache)

            try:
                _, _, odors = util.thorimage2yaml_info_and_odor_lists(thorimage_dir)
            except NoStimulusFile as e:
                # TODO warn
                continue

            bounding_frames = assign_frames_to_odor_presentations(thorsync_dir,
                thorimage_dir, analysis_dir
            )
            assert len(bounding_frames) == len(odors)

            assert traces.index[0] == 0
            assert bounding_frames[0][0] == 0
            assert bounding_frames[-1][-1] == traces.index[-1] == (len(traces) - 1)

            panel2traces[panel].append(traces)
            panel2odors[panel].extend(odors)
            # appending rather than extending here, to make easier to increment indices
            # as needed, across recordings
            panel2bounding_frames[panel].append(bounding_frames)
            panel2thorimage_ids[panel].append(thorimage_dir.name)


        for panel, traces_list in panel2traces.items():
            # TODO assert all traces.columns are same? should be
            traces = pd.concat(traces_list, ignore_index=True)
            traces.index.name = 'frame'

            odors = panel2odors[panel]

            offset = 0
            bounding_frames = []
            for ri, recording_frames in enumerate(panel2bounding_frames[panel]):
                bounding_frames.extend([
                    [i + offset, j + offset, k + offset] for i, j, k in recording_frames
                ])
                offset = bounding_frames[-1][-1] + 1

            assert len(odors) == len(bounding_frames)

            # TODO assert those cases line up after a var to indicate which i'm
            # loading
            #
            # this should just be b/c we are loading full traces instead of traces
            # preselected for "best" plane within recordings
            if pick_best_plane_here:
                assert traces.columns.get_level_values('roi').duplicated().any()
                # TODO TODO also try averaging over all planes?

                df = compute_trial_stats(traces, bounding_frames, odors, zscore=zscore)
                roi_quality = df.max()

                for_index = roi_quality.index.to_frame()
                for_index['i'] = np.arange(len(for_index))
                roi_quality.index = pd.MultiIndex.from_frame(for_index)

                indices_maximizing_each_glom = roi_quality.groupby('roi').idxmax(
                    ).apply(lambda x: x [-1]).values

                assert np.array_equal(roi_quality.iloc[indices_maximizing_each_glom],
                    roi_quality.groupby('roi').max()
                )

                df = df.iloc[:, indices_maximizing_each_glom].droplevel('z',
                    axis='columns'
                )

                d2 = compute_trial_stats(traces.iloc[:, indices_maximizing_each_glom],
                    bounding_frames, odors, zscore=zscore
                ).droplevel('z', axis='columns')
                assert d2.equals(df)
            else:
                assert not traces.columns.get_level_values('roi').duplicated().any()
                df = compute_trial_stats(traces, bounding_frames, odors, zscore=zscore)

            # TODO delete? work?
            # TODO TODO after modifying merging code to work w/ panel on columns instead
            # of thorimage_id, change name of this level back
            #
            # NOTE: also adding panel level to columns, so no duplicate if trying to
            # merge across panels
            df = util.addlevel(df, 'thorimage_id', panel, axis='columns')
            #

            # TODO TODO how to define this properly? (and need in columns too?)
            df = util.addlevel(df, 'is_pair', False)

            df = util.addlevel(df, 'panel', panel)
            df = util.addlevel(df, ['date', 'fly_num'], (date, fly_num), axis='columns')

            fly_panel_dfs.append(df)

    return fly_panel_dfs


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

    g2 = natmix.plot_activation_strength(df, seed=bootstrap_seed, _checks=_checks,
        _debug=_debug
    )
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

        # TODO TODO TODO change conversion to uint16 handling to set min to at
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

    sum_diff_fig = None
    max_diff_fig = None
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

    assert sum_diff_fig is not None and max_diff_fig is not None

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
    if al_util.save_figs:
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

    matshow_kwargs = dict(cbar_label=mean_response_desc, cmap=cmap, **shared_kwargs)

    # TODO use some toplevel stf for all but '$\\Delta$ ' prefix of this
    diff_cbar_title = f'$\\Delta$ mean peak {dff_latex}'
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


# TODO TODO move to al_util?
# TODO try to use in filling i do in main (copied/adapted from there)
# TODO also try to use in modelling stuff?
def fill_to_hemibrain(df: pd.DataFrame, value=np.nan, *, verbose=False) -> pd.DataFrame:

    # TODO replace w/ call to connectome_wPNKC (w/ _use_matt_wPNKC=False)?
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
    # Also equal to wPNKC right after call to connectome_wPNKC(_use_matt_wPNKC=False)
    prat_hemibrain_wPNKC_csv = (
        data_root / 'sent_to_grant/2024-04-05/connectivity/wPNKC.csv'
    )

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
# TODO TODO modify to load from committed paper (megamat?) data (or at least warn if
# output would be different from that)
def get_gh146_glomeruli() -> Optional[Set[str]]:
    # TODO doc what is loaded, and how to recompute it
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
        # TODO err instead (and update type hint return)
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

    # may not need the sort_index() call. just keeping for consistency w/ old
    # calculation
    gh146_glom_counts = certain_gh146_df.columns.get_level_values('roi').value_counts(
        ).sort_index()

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

    # TODO also restrict pebbled glomeruli to those (i can identify in?) GH146
    # (have been doing in modeling code, but that's now commented [after refactoring],
    # and prob want to move here anyway)

    # TODO try to make GH146 data loading agnostic to driver (pick most recent directory
    driver = output_dir2driver(output_root)
    gh146_roi_only_version = False
    gh146_roi_only_ij_corr_list = []
    if driver in orn_drivers:
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
    # TODO which plots does this make?
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

    # TODO re-establish a real-time-analysis script that can compare current ROI
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


        pdf_certain = pdf.loc[:,
            pdf.columns.get_level_values('roi').map(is_ijroi_certain)
        ]
        mean_certain = pdf_certain.groupby('odor1', sort=False).mean(
            ).groupby('roi', sort=False, axis='columns').mean()

        filled_mean_certain = fill_to_hemibrain(mean_certain)
        filled_mean_certain = sort_odors(filled_mean_certain, add_panel=panel)

        # TODO TODO move into plot_responses? have something else for this?
        filled_mean_certain.index = filled_mean_certain.index.get_level_values('odor1'
            ).map(olf.parse_odor_name)
        #

        # TODO TODO TODO fix (on ORN data):
        # ...
        # Uncaught exception
        # Traceback (most recent call last):
        #   File "./al_analysis.py", line 13386, in <module>
        #     main()
        #   File "./al_analysis.py", line 13297, in main
        #     acrossfly_response_matrix_plots(trial_df, across_fly_ijroi_dir, driver,
        #   File "./al_analysis.py", line 9829, in acrossfly_response_matrix_plots
        #     plot_responses(filled_mean_certain.T, ... 'certain_mean_filled',
        #   File "/home/tom/src/al_analysis/al_util.py", line 1572, in plot_responses
        #     fig, _ = viz.matshow(df, ... vmax=vmax, **diverging_cmap_kwargs, **kwargs)
        #   File "/home/tom/src/hong2p/hong2p/viz.py", line 531, in wrapped_plot_fn
        #     return plot_fn(data, *args, norm=norm, **kwargs)
        #   File "/home/tom/src/hong2p/hong2p/viz.py", line 565, in wrapped_plot_fn
        #     return plot_fn(*args, **kwargs)
        #   File "/home/tom/src/hong2p/hong2p/viz.py", line 216, in wrapped_plot_fn
        #     return plot_fn(df, *args, **kwargs)
        #   File "/home/tom/src/hong2p/hong2p/viz.py", line 1313, in matshow
        #     set_ticklabels(ax, 'x', xticklabels,
        #   File "/home/tom/src/hong2p/hong2p/viz.py", line 1276, in set_ticklabels
        #     raise ValueError(err_msg)
        # ValueError: duplicate xticklabels duplicated entries, with counts:
        # 'aphe' (2)
        try:
            plot_responses(filled_mean_certain.T, panel_ijroi_dir,
                'certain_mean_filled', cbar_label=mean_response_desc, title=title
            )
        except ValueError:
            warn('plotting certain_mean_filled failed!')
            pass

        # TODO remove other code that just deals w/ plotting correlations?
        # (has long been unused)
        if panel != diag_panel_str:
            corr = mean_of_fly_corrs(pdf_certain)
            plot_corr(corr, panel_ijroi_dir, 'corr_certain', title=title)

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

            # TODO still check GH146 glomeruli against output of get_gh146_glomeruli, if
            # `driver == 'GH146'? (maybe in main or something tho?)??
            if driver in orn_drivers:
                gh146_glomeruli = get_gh146_glomeruli()
                gh146_only_diag_and_panel_df = diag_and_panel_df.loc[:,
                    diag_and_panel_df.columns.get_level_values('roi'
                        ).isin(gh146_glomeruli)
                ]
                response_matrix_plots(panel_ijroi_dir, gh146_only_diag_and_panel_df,
                    'gh146-only_with-diags'
                )
                # TODO should i be making my own correlation plots here (w/o
                # diags)? or also hand off to remy? (currently doing from outputs i sent
                # remy, in separate script under scripts/ dir)

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

                    print('missing completely (marked certain in >=1 panel fly):')
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

        # TODO TODO only save this (and similar) if there are *some* uncertain, right?
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
            # TODO this not still working? also include '2h @ -3' (what was that for /
            # which fly was it in anyway? seems to have been in some pebbled megamat or
            # validation2 fly)? or was there some other mechanism to drop these from the
            # across-panel plots? diff consensus_df computation, that also included
            # odors?
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
    global retry_previously_failed
    global analyze_glomeruli_diagnostics_only
    global analyze_glomeruli_diagnostics
    global print_skipped
    global verbose
    global ij_trial_dfs
    global roi2best_plane_depth_list
    global gsheet_df

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

    # TODO delete all multiprocessing stuff? haven't used in a long time
    # TODO what is currently causing this to hang on ~ when it is done with
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
        'model-seeds', 'model-hallem', 'ijroi'
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
    parser.add_argument('-M', '--first-model-only', action='store_true',
        help='When calling model_mb_responses, sets first_model_kws_only=True to skip '
        'all but the first set of model parameters (in model_kw_list internal to that '
        'function).'
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

    parser.add_argument('-G', '--only-report-missing-glomeruli', action='store_true',
        help='Report which glomeruli are missing from current ImageJ ROIs, then exit.'
    )

    group = parser.add_mutually_exclusive_group()
    # TODO option to warn but not err as well?
    # TODO warn in cases like sensitivity_analysis's deletion of it's root output folder
    # before starting (invalidating these checks...) (err if any folder would be
    # deleted when we have this flag?)
    # TODO TODO maybe this should prompt for pickles/csvs by default (w/ option to
    # approve single or all?)? maybe backup ones that would be replaced too?
    group.add_argument('-c', '--check-outputs-unchanged', action='store_true',
        # TODO TODO update doc? is it actually true there are any plot formats i don't
        # support? or at least, this isn't the reason anymore, right? now it should just
        # be anything that mpl fn i'm using (which converts things to png i think) works
        # w/?
        #
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
    group.add_argument('-C', '--check-nonmain-outputs-unchanged', action='store_true',
        help='Like -c, but excludes outputs saved in main() from checks, so that '
        'per-panel analysis outputs can be checked separately. Outputs that would '
        'trigger a warning with this flag will not be overwritten.'
    )
    parser.add_argument('-P', '--prompt-if-changed', action='store_true',
        help='If -c/-C would trigger an error because a file changed, will instead '
        'prompt about the would-be change, and pause execution until user indicates '
        'whether the file should be overwritten.'
    )

    args = parser.parse_args()

    matching_substrs = args.matching_substrs
    force_across_fly = args.force_across_fly

    parallel = args.parallel
    # TODO work?
    al_util.ignore_existing = args.ignore_existing
    #
    steps_to_skip = args.skip
    first_model_kws_only = args.first_model_only
    retry_previously_failed = args.retry_failed
    analyze_glomeruli_diagnostics_only = args.glomeruli_diags_only

    only_report_missing_glomeruli = args.only_report_missing_glomeruli

    driver = args.driver
    indicator = args.indicator

    start_date = args.start_date
    end_date = args.end_date

    # TODO work? trying to refactor stuff shared between mb_model and al_analysis to
    # new al_util.
    al_util.exit_after_saving_fig_containing = args.exit_after_saving_fig_containing

    # TODO maybe have this also apply to warnings about stuff skipped in
    # PREprocess_recording (now that i moved frame<->odor assignment fail handling code
    # there)
    verbose = args.verbose

    # used by some stuff defined in al_util, as well as by mb_model
    al_util.verbose = verbose

    print_skipped = verbose

    al_util.check_outputs_unchanged = args.check_outputs_unchanged
    check_nonmain_outputs_unchanged = args.check_nonmain_outputs_unchanged
    if check_nonmain_outputs_unchanged:
        assert not al_util.check_outputs_unchanged
        al_util.check_outputs_unchanged = 'nonmain'

    al_util.prompt_if_changed = args.prompt_if_changed
    if al_util.prompt_if_changed:
        assert al_util.check_outputs_unchanged != False

    # TODO share --ignore-existing and --skip parsing (prob refactoring into parser arg
    # to add_argument calls?) (make sure to handle no-string-passed --skip and bool
    # --ignore-existing)
    if type(al_util.ignore_existing) is not bool:
        al_util.ignore_existing = {
            x for x in al_util.ignore_existing.split(',') if len(x) > 0
        }
        for x in al_util.ignore_existing:
            if x not in ignore_existing_options:
                raise ValueError('-i/--ignore-existing must either be given no argument'
                    ', or a comma separated list of elements from '
                    f"{ignore_existing_options}. got '{al_util.ignore_existing}'."
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

    gsheet_df = get_gsheet_metadata()

    not_in_gsheet = [(k,d) for k, d in keys_and_paired_dirs if k not in gsheet_df.index]
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
            if k in gsheet_df.index and gsheet_df.loc[k, 'driver'] == driver
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
            if k in gsheet_df.index and gsheet_df.loc[k, 'indicator'] == indicator
        ]
        if len(keys_and_paired_dirs) == 0:
            if driver is None:
                raise ValueError(f'no flies with {indicator=} to analyze')
            else:
                raise ValueError(f'no flies with {driver=}, {indicator=} to analyze')

    fly_key_set = set((d, f) for (d, f), _ in keys_and_paired_dirs)
    fly_key_set &= set(gsheet_df.index.to_list())

    # Subset of the Google Sheet pertaining to the flies being analyzed.
    gsheet_subset = gsheet_df.loc[list(fly_key_set)]
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
            if not gsheet_df.loc[k, 'exclude']
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
        # TODO TODO also say how many flies/experiments?
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

    def earliest_analysis_dir_date(analysis_dirs):
        return min(d.parts[-3] for d in analysis_dirs)

    failed_assigning_frames_analysis_dirs = [
        Path(str(x).replace('raw_data', 'analysis_intermediates'))
        for x in failed_assigning_frames_to_odors
    ]

    methods_list = []
    # TODO assert certain things constant within each fly, across all values for
    # remaining (panel, is_pair) keys?
    index_dict = None
    for experiment, thorimage2method_data in experiment2method_data.items():
        date, fly_num, panel, is_pair = experiment

        for thorimage_dir, method_data in thorimage2method_data.items():
            index_dict = {
                'date': date,
                # NOTE: 'fly' is the string value for the 'fly:' key
                'fly_num': fly_num,
                'panel': panel,
                'is_pair': is_pair,

                'thorimage_dir': shorten_path(thorimage_dir),
            }
            assert not any(k in index_dict for k in method_data)
            # TODO change how index is specified (want concat to be able to detect
            # duplicates)? possible w/ Series? if switch back to trying DataFrames, need
            # to check each is only one row (could be more if iterables in method_data
            # values get interpreted wrong, as happened for 'xy' tuple)
            curr_methods = pd.Series({**index_dict, **method_data})
            methods_list.append(curr_methods)

    # TODO warn about any flies missing certain keys completely?
    # (or assert keys same in all, as they prob are, and that missing values are just
    # represented as None/NaN)
    method_df = pd.concat(methods_list, axis='columns').T.copy()
    assert len(methods_list) == len(method_df)

    # to homogenize how some note fields have both NaN and None (at least 'power_note')
    # (NaN coming from dataframe creation? None should all be from parsing)
    method_df = method_df.replace({None: np.nan})

    assert index_dict is not None
    method_df = method_df.set_index(list(index_dict.keys()), verify_integrity=True)

    if 'driver' in method_df.columns:
        # to match what i would use to specify this driver at CLI, e.g. `-d pebbled`.
        # indicator strs should already all match CLI counterpart.
        method_df.driver = method_df.driver.replace({'pb': 'pebbled'})
    # TODO warn if non-NaN driver/indicator don't match CLI specified ones (if
    # specified)

    # just to get better dtypes for numeric columns (to more easily summarize ranges of
    # data for each column later). doesn't change contents, or anything besides dtypes.
    method_df = method_df.infer_objects()

    # TODO remove single_plane_fps col (/ don't add it in first place?), at least if we
    # have volumetric fps (or z>1 anywhere)?
    move_to_end = ['prose', 'power_level', 'power_note', 'c', 'scanner', 'zstage',
        'fast_z', 'power_regulator', 'n_flyback', 'n_averaged_frames',
        'single_plane_fps', 'fly', 'odors', 'thorimage_version'
    ]
    method_df = method_df[
        [c for c in method_df.columns if c not in move_to_end] +
        [c for c in move_to_end if c in method_df.columns]
    ].copy()

    # TODO have that code automatically generate it from fns defined in main instead?
    # TODO TODO or make wrapper also take kwarg flag to include indivdual calls under
    # -C? could also use for saving mean_est_spike_deltas.csv in modelling code
    #
    # hack to get -C to to also ignore stuff saved by save_method_csvs
    al_util._consider_as_main.append('save_method_csvs')

    def save_method_csvs(df, suffix='', *, by_fly_version=False):
        if (df.index.get_level_values('is_pair') == False).all():
            df = df.copy()
            df.index = df.index.droplevel('is_pair')

        # TODO save all w/ start/stop date range and driver+indicator CLI args (if
        # passed?) and/or n_flies in csv fnames?
        # TODO or move/copy all to a dir named that way?

        to_csv(df, output_root / f'methods_by-recording{suffix}.csv')
        # TODO also only do this one if by_fly_version? (rename flag if so)
        to_csv(df.sort_values('panel', kind='stable'),
            output_root / f'methods_by-recording{suffix}_panel-sort.csv'
        )

        # TODO actually save by_fly_version in unfilled case too? prob not.
        # <groupby>.first() skips NaN (unless all NaN) by default anyway
        #
        # using this flag to only save in filled-within-fly case, b/c if there's a mix
        # of NaN/not (much more common in unfilled case), prob don't care about the
        # NaN...
        if by_fly_version:
            group_over = [x for x in df.index.names if x != 'thorimage_dir']
            by_fly = df.groupby(group_over).first()

            to_csv(by_fly, output_root / f'methods_by-fly{suffix}.csv')
            to_csv(by_fly.sort_values('panel', kind='stable'),
                output_root / f'methods_by-fly{suffix}_panel-sort.csv'
            )

        range_df = df.select_dtypes(include=[float, int])
        # new index will be ['min','max'], w/ same columns as range_df
        # (but w/o any of the old range_df index values:
        # ['date', 'fly_num', 'panel', 'thorimage_dir']
        lim_df = range_df.agg(['min','max'])

        #Index(['fast_z', 'fly', 'odors', 'power_note', 'power_regulator', 'prose',
        #       'scanner', 'thorimage_version', 'zstage'],
        object_or_bool_cols = df.columns.difference(range_df.columns)
        # TODO -> save into one summary csv
        rest_df = df[object_or_bool_cols]
        # could do this for all columns, but don't think these values give me more than
        # just extra noise for any columns in range_df.
        #
        # TODO TODO fix:
        # ./al_analysis.py -d pebbled -n 6f -t 2023-11-29 -e 2023-11-29 -s model,corr,intensity -v -i ijroi

        # ...
        # writing pebbled_6f/methods_by-recording_no-ffill_panel-sort.csv
        # Uncaught exception
        # Traceback (most recent call last):
        #   File "./al_analysis.py", line 13241, in <module>
        #     main()
        #   File "./al_analysis.py", line 11382, in main
        #     save_method_csvs(unfilled_method_df, '_no-ffill')
        #   File "./al_analysis.py", line 11371, in save_method_csvs
        #     unique_vals = rest_df.apply(lambda x: x.unique()).to_frame('unique_vals').T
        #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/pandas/core/generic.py", line 5478, in __getattr__
        #     return object.__getattribute__(self, name)
        # AttributeError: 'DataFrame' object has no attribute 'to_frame'
        unique_vals = rest_df.apply(lambda x: x.unique()).to_frame('unique_vals').T

        # all lim_df.columns will be before all unique_vals.columns
        summary_df = pd.concat([lim_df, unique_vals])
        # TODO also only save this one for by_fly_version (or another flag w/ maybe a
        # new name, but probably using same flag for all but one csv?)
        to_csv(summary_df, output_root / f'methods_summary{suffix}.csv')


    if not only_report_missing_glomeruli:
        unfilled_method_df = method_df.copy()
        # to inspect and sanity check ffilling isn't doing anything too crazy
        save_method_csvs(unfilled_method_df, '_no-ffill')

        # TODO update comment below, now that we should have NaN instead of empty string
        # for certain cols
        #
        # at least for final megamat pebbled data, only these cols have any missing
        # values:
        # ipdb> unfilled_method_df.isna().any().replace(False, np.nan).dropna()
        # power_mw      True
        # n_days_old    True
        # power_note    True
        # fly           True
        # odors         True

        # TODO use thorimage power_level (+offset/similar, which would need to add to
        # parseing) to invalidate ffilling power (or even the same power being listed?),
        # assuming that setting ever changed (at least on the non-pockels one, w/
        # hysteresis)?
        # TODO + have non-empty (/non-matching, e.g. excluding stuff w/ 'just measured'
        # or 'unchanged' maybe) power_note str also invalidate?

        method_df = method_df.groupby(['date', 'fly_num'], sort=False).ffill()
        # TODO assert imaging params don't change w/in any fly (across recordings)?
        # most? all, except maybe gain/power_level, but even that should maybe generate
        # a warning? at least x,y,z,c, xy_pixel_size, zstep_um. gain? anything else?

        # TODO update comment below, now that we should have NaN instead of empty string
        # for certain cols
        #
        # cols still missing some values after ffilling within fly (again, for final
        # megamat pebbled data):
        # ipdb> method_df.isna().any().replace(False, np.nan).dropna()
        # power_mw      True
        # power_note    True
        # odors         True
        # TODO do some ffilling within ['date'] alone? how much would that help
        # eliminate remaining missing values?

        assert unfilled_method_df.index.equals(method_df.index)

        save_method_csvs(method_df, by_fly_version=True)


    # TODO delete. didn't seem reasonable on my ORN / PN data.
    #zscore_flypanel_dfs = recompute_responses_from_traces_per_panel(
    #    keys_and_paired_dirs, zscore=True
    #)
    #ij_trial_dfs = zscore_flypanel_dfs
    #roi2best_plane_depth_list = None
    #

    # used some of same code to reanalyze my typical dF/F way (rather than
    # remy-style z-scoring this was for), just picking best-ROIs per panel. didn't have
    # a huge effect for either my ORN or PN dendrite megamat data (see some slides in
    # my Google Slides 2025-07-07_pn_forensics).
    #
    # TODO TODO ...and then make ROI example plots based on these chosen indices?
    # (only if i decide it's worth replacing my current outputs with those. might be for
    # PN dendrite data?)
    #
    # TODO add CLI option to do this?
    # TODO delete?
    #print('REPLACING IJ_TRIAL_DFS WITH RESPONSES RECOMPUTED PER-PANEL!!!')
    #ij_flypanel_dfs = recompute_responses_from_traces_per_panel(keys_and_paired_dirs)
    #ij_trial_dfs = ij_flypanel_dfs
    #roi2best_plane_depth_list = None
    #

    # TODO TODO try averaging over all planes instead of picking best?
    # (-> compare to current w/ best plane picked only)
    #
    # (would need to know at least # of pixels per ROI to weight properly... could load
    # them but recompute_responses... currently doesn't)

    # TODO TODO try some method of scaling by variability (defined from just
    # pre-odor period? sufficiently far towards end of trial too? def don't want to use
    # response portion for that)

    # TODO TODO TODO add script to compute the same trace files recompute_responses...
    # uses, but from a directory organized as remy does, and re-extracting her suite2p
    # data to form my own traces (or getting timeseries of F from her suite2p
    # extraction, if available, but i don't think it is), that also then pushes all of
    # them through recompute_responses... and downstream (e.g. to re-extract for her
    # kiwi/control data, to compare to my model outputs)

    if len(ij_trial_dfs) == 0:
        cprint('No ImageJ ROIs defined for current experiments!', 'yellow')
        return

    # TODO TODO TODO refactor to share all this post processing w/ zscore dfs
    n_before = sum([num_notnull(x) for x in ij_trial_dfs])

    ij_trial_dfs = olf.pad_odor_indices_to_max_components(ij_trial_dfs)
    trial_df = pd.concat(ij_trial_dfs, axis='columns', verify_integrity=True)

    # NOTE: roi2best_plane_depth_list must either be set None or recomputed, if using
    # responses computed across recordings
    # (e.v. via recompute_responses_from_traces_per_panel), rather than using my usual
    # per-recording responses from ij_trial_dfs.
    if roi2best_plane_depth_list is not None:
        assert len(ij_trial_dfs) == len(roi2best_plane_depth_list)

        roi_best_plane_depths = pd.concat(roi2best_plane_depth_list, axis='columns',
            verify_integrity=True
        )
        util.check_index_vals_unique(roi_best_plane_depths)

        # TODO delete? did i ever actually need this for roi_best_plane_depths?
        # failing assertion (b/c depth defined for each recording, and not
        # differentiated by odor metadata along row axis, as w/ trial_df)
        #roi_best_plane_depths = merge_rois_across_recordings(roi_best_plane_depths)
        #
        # TODO TODO this what i want for merging depths across recordings? can i change
        # to not take average, and use depth for each specific recording? good enough?
        roi_best_plane_depths = roi_best_plane_depths.groupby(level=[
                x for x in roi_best_plane_depths.columns.names if x != 'thorimage_id'
            ], sort=False, axis='columns'
        ).mean()
        util.check_index_vals_unique(roi_best_plane_depths)
    else:
        # TODO warning here saying we are throwing this away?
        roi_best_plane_depths = None

    util.check_index_vals_unique(trial_df)
    assert num_notnull(trial_df) == n_before
    trial_df = drop_redone_odors(trial_df)

    # TODO what is this doing exactly?
    #
    # this does the same checks as above, internally
    # (that number of notna values does not change and index values are unique)
    trial_df = merge_rois_across_recordings(trial_df)

    trial_df_isna = trial_df.isna()
    assert not trial_df_isna.all(axis='columns').any()
    assert not trial_df_isna.all(axis='rows').any()
    del trial_df_isna

    if roi_best_plane_depths is not None:
        # to justify indexing it the same
        # TODO maybe i should check more than just the roi level tho?
        assert trial_df.columns.get_level_values('roi').equals(
            roi_best_plane_depths.columns.get_level_values('roi')
        )

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
    # TODO add comment explaining what this drops (+ doc in doc str for this fn)
    trial_df = drop_superfluous_uncertain_rois(trial_df)

    if roi_best_plane_depths is not None:
        roi_best_plane_depths = roi_best_plane_depths.loc[:, ~contained_plus]
        roi_best_plane_depths = drop_superfluous_uncertain_rois(roi_best_plane_depths)

        roi_best_plane_depths = sort_fly_roi_cols(roi_best_plane_depths,
            flies_first=True
        )

        # TODO another assertion here (or pre-sort line above) that
        # roi_best_plane_depths and trial_df still have same column index (or at least
        # 'roi' level, but prob all), after processing both above?

    # TODO TODO should i always merge DL2d/v data? should be same exact receptors, etc?
    # and don't always have a good number of planes labelled for each, b/c often unclear
    # where boundary is

    # TODO delete? still using any of these flies?
    #
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

    # TODO in a global cache that is saved (for use in real time analysis when drawing
    # ROIs. not currently implemented anymore), probably only update it to REMOVE flies
    # if they become marked exclude in google sheet (and otherwise just filter data to
    # compare against during that realtime analysis)

    certain_df = select_certain_rois(trial_df)

    certain_df = add_fly_id(certain_df.T, letter=True).set_index('fly_id', append=True
        ).reorder_levels(['fly_id'] + certain_df.columns.names).T

    all_fly_id_cols = ['fly_id', 'date', 'fly_num']
    fly_id_legend = index_uniq(certain_df.columns, all_fly_id_cols)

    if verbose or only_report_missing_glomeruli:
        n_flies = len(fly_id_legend)
        print()
        print(f'{n_flies} flies:')
        print_uniq(fly_id_legend, all_fly_id_cols)

    being_run_on_all_final_pebbled_data = (start_date == '2023-04-22' and
        end_date == '2024-01-05' and driver == 'pebbled' and indicator == '6f'
    )
    final_pebbled_flies = {
        # megamat flies
        ('2023-04-22', 2),
        ('2023-04-22', 3),
        ('2023-04-26', 2),
        ('2023-04-26', 3),
        ('2023-05-08', 1),
        ('2023-05-08', 3),
        ('2023-05-09', 1),
        ('2023-05-10', 1),
        ('2023-06-22', 1),

        # validation2 flies
        ('2023-11-19', 1),
        ('2023-11-21', 1),
        ('2023-11-21', 2),
        ('2023-11-21', 3),
        ('2024-01-05', 1)
    }

    fly_id_csv = output_root / 'fly_ids.csv'
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
    # TODO TODO are these fly_ids not deleted (by what?) before any real use? is it just
    # used for mean_df/etc? move this saving there then?
    if being_run_on_all_final_pebbled_data:
        curr_fly_set = {
            (format_date(date), fly)
            for date, fly in fly_id_legend[fly_keys].itertuples(index=False)
        }
        assert curr_fly_set == final_pebbled_flies

        to_csv(fly_id_legend, fly_id_csv, date_format=date_fmt_str, index=False)

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

        if verbose or only_report_missing_glomeruli:
            print()
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
            if verbose or only_report_missing_glomeruli:
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

        if verbose or only_report_missing_glomeruli:
            print('panel flies:')
            # TODO also use in load_antennal_csv.py
            print_index_uniq(panel_df.columns, all_fly_id_cols)
            print()

        odor_levels_to_drop = ['is_pair', 'odor2']
        panel_df = panel_df.droplevel(
            # b/c we might not have 'odor2' level now
            [x for x in odor_levels_to_drop if x in panel_df.index.names]
        )
        panel_df = panel_df.droplevel(['date','fly_num'], axis='columns')

        # TODO TODO probably only do this style of filling if driver is pebbled. we'd
        # expected a certain set of other glomeruli to be missing in e.g. GH146 case

        # TODO replace w/ call to connectome_wPNKC (w/ _use_matt_wPNKC=False)?
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
        # Also equal to wPNKC right after call to
        # connectome_wPNKC(_use_matt_wPNKC=False)
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

        if verbose or only_report_missing_glomeruli:
            # TODO TODO some version of the glomeruli-missing report in a context where
            # we aren't already dropping non-consensus glomeruli (as we are, earlier in
            # this loop)?
            # TODO maintain a list of ones we should be able to see (maybe from loading
            # old data? hardcoded?), and (also?) report difference wrt that set?
            print('hemibrain glomeruli not in panel flies:')
            print(sorted(hemibrain_glomeruli_not_in_panel_flies))
            print()

        _first_fly_cols = None

        filled_fly_dfs = []
        for fly_id, fly_df in panel_df.groupby(level='fly_id', axis='columns'):
            if verbose or only_report_missing_glomeruli:
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
                if verbose or only_report_missing_glomeruli:
                    print('glomeruli only in other panel flies: '
                        f'{sorted(glomeruli_only_in_other_panel_flies)}'
                    )

                # TODO warn about what we are doing here?
                fly_df[sorted(glomeruli_only_in_other_panel_flies)] = float('nan')

            else:
                if verbose or only_report_missing_glomeruli:
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

            if verbose or only_report_missing_glomeruli:
                print()

        # TODO now that i'm only dropping non-consensus glomeruli AFTER the loop for the
        # diagnostic panel, check diag csv reasonable, or maybe don't save it
        # (probably wasn't gonna use/send it anyway... so nbd)
        filled_df = pd.concat(filled_fly_dfs, axis='columns', verify_integrity=True)
        # TODO assert filled_df has columns.is_monotonic_increasing?
        # (since we are sorting pre-concat, unlike newer fill_to_hemibrain fn...)
        # prob doesn't matter.

        # TODO TODO other panels also require gating behind being_run_on_all_final...?
        # (i.e. does computation of filled_df, e.g. for megamat panel, depend on whether
        # we are running on megamat vs megamat + validation data?)
        #
        # NOTE: i think the contents of filled_df don't depend on whether we are we are
        # running on e.g. validation2 data only vs validation2 + megamat data, BUT the
        # columns (letters from fly_id_legend) currently DO differ based on this, so we
        # can't compare these outputs if we aren't running on all data
        consensus_csv = output_root / f'{panel}_consensus.csv'

        if not only_report_missing_glomeruli:
            # TODO also say filled/similar in name?
            # TODO date_format even doing anything?
            to_csv(filled_df, consensus_csv, date_format=date_fmt_str)

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

        # numeric_only=True is only to silence a FutureWarning. all input columns are
        # float.
        #
        # NOTE: as configured now, this seems to be NaN if only 1 fly (for a given roi),
        # but defined for even 2 (or more) flies for a given roi.
        stddev_df = trialmean_df.groupby(level='roi', sort=False, axis='columns'
            ).std(numeric_only=True)

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

    if only_report_missing_glomeruli:
        sys.exit()

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

    if roi_best_plane_depths is not None:
        # TODO want to also allow a version of this using certain_df instead of
        # consensus_df?
        # modelling currently hardcoding consensus_df as input (below), and this is only
        # place i'm planning to use that, so shouldn't matter.
        roi_best_plane_depths = roi_best_plane_depths.loc[:, consensus_df.columns]

    # TODO only save these two if being run on all data?
    # (or just move saving to per panel directory [/name w/ panel] if not?)
    # (less of an issue now that -C is an option to not overwrite these when running on
    # diff subsets of data)
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

    # TODO fix (failing in gh146 case, re-running 2025) (still?)
    # (think i was actually running w/ 'pebbled' for driver in comomand below, and there
    # probably aren't any pebbled flies in this date range? prob still want a better
    # error message in that case tho)
    # ./al_analysis.py -d pebbled -n 6f -t 2023-06-22 -e 2023-07-28 -s intensity,corr -v
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

        if len(mdf_only_odors) > 0:
            warn('odors in mean_df but not consensus_df, b/c do_drop_nonconsensus_odors'
                '=False:\n'
                f'{mdf_only_odors.to_frame(index=False).to_string(index=False)}'
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

    # TODO say (print? warn?) that we aren't saving these plots b/c this is False, in
    # that case?
    # TODO TODO CLI option to save thes plots despite this flag being False?
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
        #"""
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
        #"""
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

            # not sure why this line was triggering this pylint error-level code.
            # "E1120: No value for argument 'new' in method call"
            # pandas docs (for version 1.3, since i'm using 1.3.1 at time of testing)
            # don't seem to have a 'new' argument pylint is talking about.
            # pylint: disable-next=no-value-for-parameter
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
    use_consensus_for_all_acrossfly = True
    #use_consensus_for_all_acrossfly = False
    # TODO delete + fix
    print(f'{use_consensus_for_all_acrossfly=} (should be True for paper figures, but '
        'model will use consensus either way)'
    )
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

    # TODO at least if --verbose, print we are skipping step (and in all cases we skip
    # steps)
    # TODO get this to work if script is run with megamat + validation input
    # (and ideally make sure megamat is run first, esp if we serialize model params
    # there to use in validation panel, for the dff->spikedelta transform)
    # (not happy w/ current behavior?)
    if 'model' not in steps_to_skip:
        assert 'model-sensitivity' in skippable_steps
        skip_sensitivity_analysis = 'model-sensitivity' in steps_to_skip

        assert 'model-seeds' in skippable_steps
        skip_models_with_seeds = 'model-seeds' in steps_to_skip

        assert 'model-hallem' in skippable_steps
        skip_hallem_models = 'model-hallem' in steps_to_skip

        # TODO worth warning that model won't be run otherwise?
        if driver in orn_drivers:
            # TODO TODO may want to skip in general if we have trial responses redefined
            # from post-processed per-panel_concat traces, and not just if
            # zscore_traces_per_recording=True (or just if zscore was True when
            # recomputing...) (via recompute_responses_from_traces_per_panel)
            if zscore_traces_per_recording:
                # TODO TODO add way of telling how cached outputs were computed, so we
                # can check here (or err if any inconsistent choices among files being
                # loaded)
                warn('probably do not want to run model with Z-scored F input, so '
                    'skipping modelling. set zscore_traces_per_recording=False and '
                    're-run (with `-i ijroi`, if any cached responses were computed '
                    'this way!)'
                )
                return

            # NOTE: use_consensus_for_all_acrossfly must be False if I want to try using
            # certain_df again here (if True, certain_df is redefined to consensus_df
            # above)
            model_mb_responses(consensus_df, across_fly_ijroi_dir,
                roi_depths=roi_best_plane_depths,
                skip_sensitivity_analysis=skip_sensitivity_analysis,
                skip_models_with_seeds=skip_models_with_seeds,
                skip_hallem_models=skip_hallem_models,
                first_model_kws_only=first_model_kws_only,
            )
        else:
            print(f'not running MB model(s), as driver not in {orn_drivers=}')

    # TODO TODO replace w/ copying the key outputs to a dir created w/ these params?
    # (at least, w/ some flag set?) (use response_calc_str for dir name)
    # TODO delete
    #print()
    #print('PARAMS RELEVANT FOR RESPONSE STATISTICS:')
    #print(response_calc_str)
    # TODO TODO make dir for response_calc_str, and copy relevant files over
    #import ipdb; ipdb.set_trace()
    #


if __name__ == '__main__':
    main()

