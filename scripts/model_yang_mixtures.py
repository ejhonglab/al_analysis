#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import Counter, defaultdict
import contextlib
from pathlib import Path
from pprint import pformat, pprint
from itertools import combinations, product
import shutil
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Sequence, Tuple, Union
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LogNorm, LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.collections import PolyCollection
from matplotlib.container import BarContainer
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm

from hong2p.util import (pd_allclose, pd_indices_equal, addlevel, add_group_id, reindex,
    subset_same_in_all_dicts, format_params
)
from hong2p import olf
from hong2p.olf import (parse_log10_conc, solvent_str, component_delim,
    format_odor, format_mix_from_strs
)
from hong2p.viz import (dff_latex, add_group_labels_and_lines, map_each_series_to_rgb,
    stripplot
)
from hong2p.util import symlink, is_scalar
from hong2p.types import ParamDict, Color, DataFrameOrSeries, Axis, Palette
from natmix import drop_mix_dilutions

from al_analysis import al_util
from al_analysis.al_util import (savefig, plot_responses, read_parquet, to_csv,
    to_parquet, data_root, fly_cols, flyroi_cols, warn, cluster_rois, _have_fly_cols,
    add_legends_and_colorbars, diverging_cmap, plot_cols_with_diff_colormaps,
    panel2name_order, load_natmix_dff, plot_corr, mean_of_fly_corrs, format_panel,
    mean_response_desc, plot_fmt, add_check_outputs_unchanged_CLI_flag_and_parse_args,
    to_pickle
)
from al_analysis.al_util import mean_response_desc as orn_response_desc
from al_analysis.mb_model import (megamat_orn_deltas, fit_and_plot_mb_model,
    megamat_orn_deltas, natmix_orn_deltas, get_thr_and_APL_weights, format_model_params,
    get_odor_fname_suffix, KC_ID, CLAW_ID, dict_seq_product, abbrev_model_id,
    read_params, exclude_params, calc_mix_suppression, get_diff_col, diff_col2desc,
    FULL_MODEL_KW_LIST, NoCachedModelOutputsError, logistic, summarize_response_classes,
    add_missing_cells_to_nonresponders, format_response_class, kc_type_hue_order,
    get_fly_color_series, KC_TYPE, count_flies_and_rois, get_fitmbmodel_default,
    TRY_ALL_MODELS_WITH, TRY_NONCLAW_MODELS_WITH, TRY_CLAW_MODELS_WITH,
    TRY_BOUTON_MODELS_WITH, drop_binaries_mixdilutions_and_pfo, drop_silent_model_cells,
    analyze_spatial_claws, model_pnkc_class, is_mix, EXPECTED_MODEL_PNKC_CLASSES,
    REMY_KC_RESPONSE_THRESHOLD, NATMIX_ORN_RESPONSE_THRESH, plot_means_and_counts,
    print_logistic_scaling_effect, CLASS_SIZE_FRAC_THRESH, CI, plot_avg_mixsupp,
    plot_response_class_summary
)
from al_analysis.al_analysis import fill_to_hemibrain


# TODO restore True and selectively silence (w/ context managers, manually specifying
# layout, whatever. or diff slicing/indexing method for PerformanceWarning) as much as
# possible
DEBUG: bool = False
warn_handling = 'error' if DEBUG else 'ignore'
warnings.filterwarnings(warn_handling, message='The figure layout has changed')
# PerformanceWarning: indexing past lexsort depth may impact performance.
warnings.simplefilter(warn_handling, pd.errors.PerformanceWarning)

# TODO double check these are set i want to use from looking at -m output
nonshared_model_kws: List[ParamDict] = [
    dict(pn2kc_connections='uniform', n_claws=7),
    dict(weight_divisor=20),
    # TODO keep pn_claw_to_apl=True? (it is what -m -M selected, i think)
    # TODO TODO TODO try reproing claw_dynamics=True improvement w/ change to KC tau in
    # cases above (and here)
    # TODO re-order params (even possible if i want to define w/ dict_seq_product as
    # now?) (or always sort somewhere, when making model_dirname?) so this one hits the
    # cached existing dir? or manually copy to fix for now...?
    dict(one_row_per_claw=True, prat_claws=True, pn_claw_to_apl=True, claw_dynamics=True),
]
# passing the CLI arg -f will use FULL_MODEL_KW_LIST (currently len 137) instead of this
SHORT_MODEL_TUNE_KWS: List[ParamDict] = dict_seq_product(nonshared_model_kws,
    # this doesn't changethe length of nonshared_model_kws, just adds these parameters
    # to each dict in existing list
    [dict(target_sparsity=0.05, n_spikes_for_response=2)]
)

# TODO assert the above are a subset of FULL_MODEL_KW_LIST? they are, right?
# (for both OLD_* and new SHORT_*)
test_with_connectome_vs_uniform_apl: List[ParamDict] = [
    dict(weight_divisor=20),
    dict(one_row_per_claw=True, prat_claws=True),
    dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True),
]
# passing the CLI arg -O will use this instead of SHORT_MODEL_TUNE_KWS
OLD_SHORT_MODEL_TUNE_KWS: List[ParamDict] = [
    # comparison for all other model cases, to see to what extent changes to PN>KC
    # weight matrix (and potentially other changes) matter
    dict(pn2kc_connections='uniform', n_claws=7),
] + dict_seq_product(test_with_connectome_vs_uniform_apl,
    [dict(), dict(use_connectome_APL_weights=True)]
)
del test_with_connectome_vs_uniform_apl

# to run natmix_data/analysis.py on current models analyzed by this script,
# use the folowing CLI args:
# ```
# ./analysis ~/src/al_analysis/scripts/yang_mix_outputs/last_chosen_modeldirs.csv \
#    -r ~/src/al_analysis/scripts/yang_mix_outputs
# ```
# helpful to check logistic scaling fit outputs, as that's not currently done in here.
# in this script, currently just use hardcoded values in list below.
# natmix_data/analysis.py should be runnable in this same environment capable of running
# this script.
#
# TODO try calling mb_model.scale_model_spike_counts in here, to fit vs KC data?
# TODO at least factor out code (from scale_model_spike_counts verbose=True) to print #
#
# spikes [0, 5] -> scaling output, and call that here?
# use `-l <index-from-0>` to select an element from this list to use for logistic
# scaling. If not the default (0), logistic params will be included in model output
# dirnames.
# NOTE: *important* if adding to this list, do not change the position of the first
# element, and only add to positions after that, as currently the logistic scaling
# parameters are only included in output dir & model-order parquet/plots names for
# indices *other than* the default 0.
LOGISTIC_SCALING_PARAM_LIST: List[ParamDict] = [
    # NOTE: do not change what element is at index 0 in this list!
    # default element selected
    # NOTE: k=growth rate, x0=midpoint, L="supermum" (upper bound)
    dict(k=2.0, x0=2.0, L=3.0),

    # TODO TODO would current models tend to always get scaling params more like below
    # than above? should that be new default?
    # were outputs above all from versions of model w/ parameters from before
    # optimizing (sorting)for mixture suppression?
    # TODO TODO check the params listed in natmix_data/analysis.py when running w/ these
    # models as input.
    dict(k=0.5, x0=5.0, L=2.5),
]

# in sorted order, if useful for checking other things against this
NATMIX_PANELS: Tuple[str] = ('control', 'kiwi')
assert NATMIX_PANELS == tuple(sorted(NATMIX_PANELS))

NATMIX_MIX_TYPES: List[str] = ['binary', '5comp']

# TODO use elsewhere too
# TODO rename kc_panel_cols or something? 'mix' mostly (only?) used for kc case, at
# least for 5comp + binary, right?
PANEL_COLS: List[str] = ['panel', 'mix']

PNKC_CLASS_COL: str = 'model_pnkc_class'
EXPECTED_NONMODEL_PNKC_VALS: Set[str] = {'KCs', 'ORNs', 'EAG'}

# may not have 'connectome_apl' depending on whether simplify_models=True/etc.
# these generally used as index cols, with 'source' generally starting as model output
# directory name, and typically having information redundant with any of the other
# columns stripped out after those columns are defined. some abbreviations may also be
# applied
MODEL_COLS: List[str] = [PNKC_CLASS_COL, 'connectome_apl', 'source']

# .N is the # of colors in list (256)
# making a new colormap that is just the top (red) half of the old one
_colors = diverging_cmap(np.linspace(0.5, 1.0, diverging_cmap.N // 2))
diverging_cmap_tophalf = LinearSegmentedColormap.from_list(
    f'top_half_{diverging_cmap.name}', _colors
)
midpoint_before = diverging_cmap(0.5)
top_before = diverging_cmap(1.0)
bottom_after = diverging_cmap_tophalf(0.0)
top_after = diverging_cmap_tophalf(1.0)
assert midpoint_before == bottom_after, f'{midpoint_before=}\n{bottom_after=}'
assert top_before == top_after, f'{top_before=}\n{top_after=}'

# TODO share w/ CLI input checking? (-> allow passing order_by_mean_mixsupp_from as CLI
# arg?)
MODEL_STATS: List[str] = ['logistic_scaled_num_spikes', 'num_spikes']

# this is only taken a param of pointplot (which catplot should pass it to),
# but pointplot only defines it per hue level, so would prob need to either
# duplicate colors (and have same len list of colors and markers) or two calls?
#
# TODO delete?
uniform_apl_marker = '.'
# can't get this to reproduce my open circles w/ stripplot (if only i could access
# fillstyle...)
#uniform_apl_marker = 'o'
connectome_apl_marker = '+'

uniform_apl_linestyle = '-'
connectome_apl_linestyle = '--'

# NOTE: ambiguous w/ above. assumed we will only use these for plots only analyzing
# non-connectome-APL model variants (and/or non-model stuff)
mix_linestyle = '-'
component_linestyle = '--'

# TODO TODO this working? kc_point_alpha is
kc_err_alpha = 0.4
#
kc_point_alpha = 0.3

# TODO should any of these be (duplicated? placed instead in?) main kwargs to
# pointplot calls, instead of specifically to the err_kws=<dict> this gets passed to?
kc_err_kws = dict(linewidth=1.5, alpha=kc_err_alpha)

# CI from mb_model should be 95
kc_pointplot_kws = dict(marker='o', markerfacecolor='none', linestyle='none', seed=1,
    errorbar=('ci', CI), capsize=0, err_kws=kc_err_kws
)

# TODO jitter/dodge more (than default)?
# TODO set jitter=False?
# TODO use model_markersize here? (+ rename [point|dot]_markersize?) (no. for
# whatever reason, seems we need way less here than there) 3.5 was too small
perfly_stripplot_kws = dict(alpha=kc_point_alpha, legend=False, size=5.0)

# TODO delete this flag if it wasn't the difference between plotting fns that was
# causing plot_response_strength_dists to get Terminated in `-f` case
# (ever make this plots w/ -f before? anything i can do differently?)
_USE_KDEPLOT: bool = False

#  will be set to model_alpha_for_legend=0.7 in add_fixed_legend (still?)
MODEL_ALPHA: float = 0.5

# 1.5 seems to be pretty much the same as the default being used for KC line in
# mixsupp distplot
MODEL_LINEWIDTH: float = 1.5
# 8.0 seems too small now for some reason. related all the "The figure layout has
# changed to tight" warnings i'm seeing?
# for some reason this needs to be much larger than the KC stripplot markersize
# values
# 15.0 was a bit too much
MODEL_MARKERSIZE: float = 13.0

# TODO caps name?
marker_kws = dict(
    # default markeredgewidth seems ~2
    # TODO TODO set even lower markeredgewidth/markersize if full_model_params?
    # (would want to copy this dict to model_marker_kws, since this is
    # currently also used for KC data)
    # TODO TODO as long as i'm using strip plot and not pointplot, i don't think i
    # need this
    linestyle='none',
    # TODO take out of here? use model_markersize (always the largest one?)
    markersize=MODEL_MARKERSIZE
)

# TODO -.35 instead? ig preprint stuff wasn't using signed abs max tho
ORN_VMIN: float = -0.5


def parse_odor_name(odor: str) -> str:
    return olf.parse_odor_name(odor, require_conc=False)


# TODO use elsewhere?
def get_model_stat_label(stat: str, logistic_scaling_title_str: str = '', *,
    latex: bool = True) -> str:
    assert stat in MODEL_STATS, f'{stat=} not in {MODEL_STATS=}'

    if not latex:
        desc = stat

    n_spikes_latex = 'N_{spikes}'
    if stat == 'num_spikes':
        if latex:
            # TODO like this?
            # TODO happy with this?
            desc = f'${n_spikes_latex}$'
        return desc

    assert stat == 'logistic_scaled_num_spikes'
    if latex:
        # TODO happy with this?
        desc = f'$logistic({n_spikes_latex})$'

    desc += logistic_scaling_title_str
    return desc


# set to an int >1 (but << #-ROIs-per-fly / 2), to group ROIs within fly, and then only
# hierarchichal cluster those means instead of all ROI data. can make rows easier to
# see, to help rule out artifacts / illusions
KC_ROW_COARSEN: int = None

# TODO compare to 'auto' again?
# TODO reduce for logistic_scaled_num_spikes stuff? how (w/o making distplot too
# context specific?). just make the plots w/ diff stats w/ separate calls?
N_BINS: int = 30

# TODO move to hong2p.viz?
# TODO TODO ever try KDE again, after getting hists i like?
# NOTE: kde= overrides histplot kde= arg, as I don't think I currently will want both
# plotted together. it can be used to override global _USE_KDEPLOT.
def distplot(data=None, *, kde: Optional[bool] = None, common_norm: bool = False,
    # TODO delete. prefer default cut=3.0 actually. cut=0 abruptly stops density line at
    # data limits, but does not otherwise change the rest of the line.
    #cut: float = 0.0,
    **kwargs):
    # TODO any reason to use kdeplot instead of sns.histplot(...  kde=True)? histplot
    # can show a hist and a KDE, but kdeplot can't show a hist. displot is a figure
    # rather than axes level plotting fn. binning algorithms might be different by
    # default [or always?]? care? https://stackoverflow.com/questions/63798214
    assert data is not None

    log_yscale = False
    if 'log_scale' in kwargs:
        log_scale = kwargs['log_scale']
        if type(log_scale) is bool:
            log_yscale = log_scale
        else:
            assert type(log_scale) is tuple and len(log_scale) == 2
            # TODO also disable fill if log xscale?
            log_yscale = log_scale[1]

    # TODO also explicitly accept log_yscale here? how was other code doing it?

    if 'x' in kwargs:
        assert isinstance(data, pd.DataFrame)
        x = kwargs['x']
        values = data[x]
    else:
        if isinstance(data, pd.DataFrame):
            raise ValueError('must pass x=<col-name> for DataFrame input')

        # NOTE: assuming 1D Series/list/array here
        assert isinstance(data, (pd.Series, np.ndarray, list)), f'{type(data)=}'
        if isinstance(data, np.ndarray):
            assert len(data.shape) == 1, f'{data.shape=}'
        values = data

    if kde is None:
        kde = _USE_KDEPLOT

    if not kde:
        discrete = np.allclose(values, values.astype(int))

        # TODO any way to get ends of step to start and end at 0? thats main thing i
        # don't like about it. ever an issue w/ fill=True, or just not seeing it now
        # that i set more bins?
        # TODO ok to set fill=False for all log_yscale=True, or are there any plots
        # where it makes sense?
        # TODO does alpha just change face alpha? (it does at least do that, right?)
        fill = kwargs.get('fill', not log_yscale)
        # TODO 0.3 a good default here?
        alpha = kwargs.get('alpha', 0.3 if fill else 1.0)
        element_kws = dict(element='step', fill=fill, alpha=alpha)

        # so i can override element_kws
        # TODO do i actually want that? maybe for alpha?
        kws = dict(element_kws)
        kws.update(kwargs)

        if not any(x in kwargs for x in ('bins', 'binwidth', 'common_bins')):
            bins = N_BINS
            # TODO also check bins is int (anything else compatible?) if binrange
            # passed? would seaborn fail with helpful message if i don't do anything
            # there?
            kws['bins'] = bins

        # NOTE: for stat='density' vs 'probability':
        # https://stackoverflow.com/questions/73872722
        # TL;DR: 'probability' ignores bar height, which might be what I want?
        # 'density' ensures integral of bar areas sums to 1, which includes bar width.
        # is that something i want?
        # TODO actually test appearance of 'probability' vs 'density', w/ other things
        # same?
        #stat = 'probability'
        stat = 'density'

        # TODO TODO try cumulative=True? (for ECDF)
        return sns.histplot(data=data, common_norm=common_norm, stat=stat,
            discrete=discrete, **kws
        )
    else:
        return sns.kdeplot(data=data, common_norm=common_norm, **kwargs)


# TODO move to hong2p.util?
# TODO TODO use everwhere else i duplicate this
# TODO add allow_nan=False flag? idk
def get_single_unique(ser: Union[pd.Series, pd.Index]) -> Any:

    # TODO maybe if all NaN, it's ok? flag for this?
    assert ser.notna().all(), (f'get_unique expects no NaN. {ser.name=}'
        f'\nser value counts:\n{ser.value_counts(dropna=False).to_string()}'
    )

    unique = ser.unique()
    assert len(unique) == 1, f'{unique=}'
    return unique[0]


def get_single_unique_kc_panel(data: pd.DataFrame) -> str:
    check_not_in_index_or_column_names(data, 'panel')
    # operating on unique values might be a bit faster, but whatever
    kc_panel = data.panel.map(panel2kc_panel)
    return get_single_unique(kc_panel)


# TODO ok to have norm_to_kc_panel=True as default?
def get_single_unique_panel_and_mix(data: pd.DataFrame, *, norm_to_kc_panel: bool = True
    ) -> Tuple[str, str]:
    # TODO support these being anything other than columns? or just (.T +) reset_index()
    # (or get frame from index) first if needed? (prob not. get_single_unique_kc_panel
    # would also need changing)
    if norm_to_kc_panel:
        # this normalizes panel names (combining a few model ones) *BEFORE* checking
        # unique (i.e. we can't just apply `panel2kc_panel` fn on output of
        # `get_single_unique(data.panel)` here, as there would be multiple unique value
        # (-> error)
        panel = get_single_unique_kc_panel(data)
    else:
        panel = get_single_unique(data.panel)

    mix = get_single_unique(data.mix)
    return panel, mix


# TODO move to hong2p.util?
def get_index(data: DataFrameOrSeries, *, axis: Axis = 'index') -> pd.Index:
    row_axis_opts = (0, 'index', 'rows')
    col_axis_opts = (1, 'columns')
    all_axis_opts = row_axis_opts + col_axis_opts

    if axis in row_axis_opts:
        index = data.index
    elif axis in col_axis_opts:
        if isinstance(data, pd.Series):
            raise ValueError(f'axis={axis} requested for Series data, which only have '
                f'row index, as specified by any of {row_axis_opts}'
            )
        index = data.columns
    else:
        raise ValueError(f'{axis=} was not in {all_axis_opts}')
    return index


# TODO move to hong2p.olf?
def get_odors(data: DataFrameOrSeries, *, axis: Axis = 'index') -> pd.Index:
    index = get_index(data, axis=axis)
    # TODO loop over axes and get odors for first that has first_odor_level not raise
    # ValueError (or assert all equal if multiple axes have it)? (if axis is not None,
    # and default to None?)
    # TODO ever need (default) allow_multiple=True? thread thru kwarg?
    level = olf.first_odor_level(index, allow_multiple=False)
    odors = index.get_level_values(level)
    assert odors.notna().all(), 'get_odors expects no NaN odor values'
    return odors


# TODO use this elsewhere this value is currently hardcoded
N_VARIANTS_DELIM: str = ' ('

# TODO use elsewhere
def normalize_pnkc_class(x: str) -> str:
    """Strips any '<N_VARIANTS_DELIM><#-variants> variant(s))' suffixes from input

    Returned value should always be in `EXPECTED_MODEL_PNKC_CLASSES`
    """
    parts = x.split(N_VARIANTS_DELIM)
    assert len(parts) <= 2, f'{parts=}'
    pnkc_class = parts[0]
    assert pnkc_class in EXPECTED_MODEL_PNKC_CLASSES, (
        f'{pnkc_class=} not in {EXPECTED_MODEL_PNKC_CLASSES=}'
    )
    return pnkc_class


# TODO use elsewhere
def pnkc_class_is_model(x: Union[float, str]) -> bool:
    # just float cause might have np.nan in input

    # assume any NaN in input is for non-model data
    if pd.isnull(x):
        return False

    assert isinstance(x, str)

    # TODO use a regex to strip the whole end exactly, expecting the closing paren
    # at end and everything? (meh)
    is_model = x in EXPECTED_MODEL_PNKC_CLASSES or (N_VARIANTS_DELIM in x and
        # TODO maybe don't err if normalize_pnkc_class would throw assertion?
        # or otherwise, don't even check `N_VARIANTS_DELIM in x` first?
        normalize_pnkc_class(x) in EXPECTED_MODEL_PNKC_CLASSES
    )
    if not is_model:
        assert x in EXPECTED_NONMODEL_PNKC_VALS, f'{x=}'

    return is_model


# TODO delete dropna= flag after fixing all cases where there are NaN in this col?
# or nah?
def has_model(df: pd.DataFrame, *, dropna: bool = False) -> bool:
    """Checks `PNKC_CLASS_COL` is in `df.columns`, and whether it contains model values.

    Values matching or starting with one of `EXPECTED_MODEL_PNKC_CLASSES` count as model
    entries.

    Raises ValueError if unexpected values in this column, but NaN currently allowed
    (assumed non-model). Besides model values and NaN, only values in
    `EXPECTED_MODEL_PNKC_CLASSES` are allowed.

    Args:
        dropna: if False, will assert `df[PNKC_CLASS_COL]` has no NaN
    """
    assert PNKC_CLASS_COL not in df.index.names, 'handle?'

    if PNKC_CLASS_COL not in df.columns:
        # TODO ever any cases where i'd want to check something else?
        return False

    # TODO restore? i think i've fixed this now. no calls currently set dropna=True
    #
    # TODO delete. there is NaN sometimes currently (e.g. from some plot_fn calls)
    # TODO assert no NaN in this column? i have it filled pretty much everywhere,
    # right? (could just drop if it's an issue)
    #assert not df[PNKC_CLASS_COL].isna().any(), ('expected NaN values to be filled '
    #    'with "KCs" or something, for non-model data. could dropna here instead'
    #)

    # doing the dropna() since I could not always currently assert no NaN
    # (assuming NaN would only ever be for non-model stuff tho)
    vals = df[PNKC_CLASS_COL]
    if dropna:
        vals = vals.dropna()
    else:
        assert vals.notna().all(), ('has_model: set dropna=True to ignore NaN here, '
            f'or fix why there are NaN in this {PNKC_CLASS_COL=} in the first place'
        )

    vals = vals.unique()
    for x in vals:
        if x in EXPECTED_NONMODEL_PNKC_VALS:
            continue

        if pnkc_class_is_model(x):
            return True

        raise ValueError(f'{x=} (nor prefix preceding {N_VARIANTS_DELIM=}) not in '
            f'either:\n{EXPECTED_MODEL_PNKC_CLASSES=}, nor...\n'
            f'{EXPECTED_MODEL_PNKC_CLASSES=}'
        )

    return False


def has_connectome_apl(df: pd.DataFrame, *, apl_col: str = 'connectome_apl') -> bool:
    # TODO TODO factor a fn to get whatever variable from either index or columns?
    if apl_col in df.index.names:
        assert apl_col not in df.columns, (
            f'{apl_col} in both index.names and columns! drop one'
        )
        apl_vals = df.index.get_level_values(apl_col)
    elif apl_col in df.columns:
        apl_vals = df[apl_col]
    #
    else:
        return False

    return apl_vals.any()


def check_not_in_index_or_column_names(df: pd.DataFrame, x: str) -> None:
    """Raise if `x` in `df.index.names` or `df.columns.names`

    Usually so we know we only have to check for the variable name in `.columns`, or as
    part of assertion that the variable in nowhere (including in `.columns`).
    """
    assert all(x not in names for names in (df.index.names, df.columns.names)), \
        f'expecting {x=} in at most .columns, not index.names or columns.names'


def add_source_and_class_cols(df: pd.DataFrame, val: str, *, copy: bool = True
    ) -> pd.DataFrame:
    """Adds ['source', PNKC_CLASS_COL, 'source_type'] columns to `df`, all `= val`

    For setting this metadata for KC & ORN inputs, where these columns should not
    already exist, and for which they should all have the same value. For model data,
    'source' & PNKC_CLASS_COL can take multiple str values, and only 'source_type' would
    be expected to be a single value (='model').
    """
    # TODO also handle model data in here (just checking 'source' / PNKC_CLASS_COL, and
    # adding 'source_type' -> 'model')?) (meh)
    if copy:
        df = df.copy()

    # TODO keep this? (not if i add support for model input, at least not
    # unconditionally)
    assert val in EXPECTED_NONMODEL_PNKC_VALS, (
        f'expected {val=} in {EXPECTED_NONMODEL_PNKC_VALS=}'
    )

    cols_to_add = ['source', PNKC_CLASS_COL, 'source_type']
    for x in cols_to_add:
        # TODO TODO allow them to exist, but check all matches `val` if so?
        # (might make easier to call this in more places i want to?)
        check_not_in_index_or_column_names(df, x)
        assert x not in df.columns, f'{x=} already in {df.columns=}!'
        df[x] = val

    return df


def get_unique_source_types(df: pd.DataFrame) -> Set[str]:
    assert len(df) > 0
    assert df.source_type.notna().all()
    source_types = set(df.source_type.unique())
    assert source_types - EXPECTED_NONMODEL_PNKC_VALS == {'model'}, (
        f"expected all {source_types=} to be either 'model' or in "
        f'{EXPECTED_NONMODEL_PNKC_VALS=}'
    )
    return source_types


DEFAULT_SOURCE_TYPES_TO_PLOT: Tuple[str] = ('model', 'KCs')
assert set(DEFAULT_SOURCE_TYPES_TO_PLOT) - set(EXPECTED_NONMODEL_PNKC_VALS) == {'model'}

class MissingRequestedSourcesError(ValueError):
    pass

def get_model_kc_orn_data(df: pd.DataFrame, *,
    # TODO change copy default to True?
    # TODO should i even have a default for source_types? make it positional?
    source_types: Sequence[str] = DEFAULT_SOURCE_TYPES_TO_PLOT, copy: bool = False
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # TODO doc
    """Returns (model, KC, ORN) data, subset from `df`, each either a DataFrame or None.

    Args:
        df: data to subset model/KC/ORN data from. must have a 'source_type' column,
            with values all str and one of 'model' / 'KCs' / 'ORNs'.

        source_types: 'source_type' values for which to return dataframe subsets

        copy: whether to call `.copy()` on each dataframe output before returning.
            NOTE: unlike my other functions that use this kwarg, this one defaults to
            False currently.

    Raises MissingRequestedSourcesError if `df` does not have all requested sources.
    """
    # TODO add EXPECTED_SOURCE_TYPES module-level def, for better error message if
    # requrested type is never going to exist?
    df_source_types = get_unique_source_types(df)
    requested = set(source_types)
    if not requested - df_source_types == set():
        raise MissingRequestedSourcesError('not all requested source_types='
            f'{requested} were in df source_types={df_source_types}\n...for panel/mix:'
            f"\n{df[['panel','mix']].drop_duplicates().to_string(index=False)}"
        )

    model_data = df[df.source_type == 'model'] if 'model' in requested else None
    kc_data = df[df.source_type == 'KCs'] if 'KCs' in requested else None
    orn_data = df[df.source_type == 'ORNs'] if 'ORNs' in requested else None

    if copy:
        if model_data is not None:
            model_data = model_data.copy()

        if kc_data is not None:
            kc_data = kc_data.copy()

        if orn_data is not None:
            orn_data = orn_data.copy()

    return model_data, kc_data, orn_data


# TODO pick only either this or my other fn (odor_sort_key, which is just used manually
# w/ sort_[values|index]()) defined in here for it? would prob need to adapt either to
# make work...
def sort_odors(data: pd.DataFrame, **kwargs):
    return olf.sort_odors(data, panel2name_order=panel2name_order,
        # this is to silence warning if panel is not in panel2name_order
        # (e.g. 'diag-binaries_max')
        if_panel_missing=None, **kwargs
    )


def has_multiple_mixes(df: pd.DataFrame, *, norm_to_kc_panel: bool = True) -> bool:
    # checking only one (/no) panel too, because that's the only context in which this
    # was defined before (in loop over panels), and that's the context within which we
    # want to check for and multiple multiple mixes
    if 'panel' in df.columns:
        check_not_in_index_or_column_names(df, 'panel')
        # this will also raise assertion error if not a single unique non-NaN value
        # (or after normalizing to kc panel, in that case)
        if norm_to_kc_panel:
            get_single_unique_kc_panel(df)
        else:
            get_single_unique(df.panel)

    check_not_in_index_or_column_names(df, 'mix')

    # TODO will this be true?
    assert df.mix.notna().all(), 'expected no NaN mix'

    return 'mix' in df.columns and df.mix.nunique() > 1


def assert_all_mix_after_component(df_or_odors: Union[pd.DataFrame, Sequence]) -> None:
    if isinstance(df_or_odors, pd.DataFrame):
        df = df_or_odors
        unique_odors = df.odor.unique()
        odors = df.odor
    else:
        unique_odors = sorted(set(df_or_odors))
        odors = df_or_odors

    mix = None
    comp = None
    for x in unique_odors:
        if is_mix(x):
            assert mix is None, 'multiple odors w/ is_mix(x)=True'
            mix = x
        else:
            comp = x
    assert mix is not None, 'no odor w/ is_mix(x)=True'
    assert comp is not None
    last_component_idx = (odors == comp)[::-1].idxmax()
    first_mix_idx = (odors == mix).idxmax()
    assert last_component_idx < first_mix_idx, 'need to sort!'


def calc_max_flymean(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('odor').value.mean().max()


NORM_PREFIX: str = 'normalized_'
# TODO put NORM_[TO_FLYMEAN_MAX|PER_FLY] in fnames?

# If False, the max individual fly (trial-averaged) response (across all KC data,
# and all odors), will be sent to 1. If True, the max fly-averaged (after trial
# averaging) odor response will be sent to 1, so some individual fly points could
# exceed 1.
NORM_TO_FLYMEAN_MAX: bool = True

# If False, normalize (max->1) within each of the two datasets (across all odors):
# - all models
# - all individual points within all KC data, no matter the fly or exp_type
#
# If True, models are handled the same, but now each KC fly will have one odor with
# each stat max=1.
# TODO (delete?) add assertion that recalculates mean of CI and asserts close to 1,
# right before plotting?
NORM_PER_FLY: bool = False
# TODO delete? make sense? enough flies have all / most odors?
#NORM_PER_FLY: bool = True

def normalize_one_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    stats = raw_df.stat.unique()
    assert len(stats) == 1
    stat = stats[0]
    assert not stat.startswith(NORM_PREFIX)

    # TODO need to handle negative values?
    vmin = raw_df.value.min()
    if vmin < 0:
        warn(f'{vmin=} < 0 for data. issue for normalization?')

    normed_df = raw_df.copy()
    # TODO also try subtracting min? meh
    # TODO try NORM_PER_FLY again?

    if NORM_TO_FLYMEAN_MAX:
        normed_df['value'] /= calc_max_flymean(normed_df)
        assert np.isclose(calc_max_flymean(normed_df), 1)
        assert pd_allclose(raw_df.value / calc_max_flymean(raw_df), normed_df.value)
        assert np.isclose(normed_df.groupby('odor').value.mean().max(), 1)

    elif not NORM_PER_FLY:
        normed_df['value'] = normed_df.value / normed_df.value.max()
        assert np.isclose(normed_df.value.max(), 1)
    else:
        # if i decide to include stuff defined across odors in these KC
        # dfs (i.e. mix suppression), may need to change handling here
        # (probably will handle that stuff separately though)
        #
        # oh, so all flies do have all odors actually? at least, before any
        # attempt to lump flies from the other dataset (with one of these 3
        # mixtures, and its components) in with this data
        assert (
            len(raw_df.groupby(fly_cols).odor.unique().map(tuple).unique()) == 1
        ), 'not all flies had all odors'

        cols_before = list(normed_df.columns)
        # NOTE: EAG has fly_id instead of fly_cols, if it matters (doesn't, but just b/c
        # this path is not being run...)
        normed_df = normed_df.set_index(fly_cols + ['odor'],
            verify_integrity=True
        )
        normed_df['value'] /= normed_df.groupby(fly_cols).value.max()
        normed_df = normed_df.reset_index()[cols_before].copy()
        assert np.allclose(normed_df.groupby(fly_cols).value.max(), 1)

    normed_df['stat'] = f'{NORM_PREFIX}{stat}'
    return normed_df


MEAN_PREFIX: str = 'mean_'
def get_single_unique_stat(df: pd.DataFrame, *, strip_mean_prefix: bool = True,
    strip_norm_prefix: bool = True) -> str:
    """Returns unique value in df.stat columns, removing prefixes optionally
    """
    stat = get_single_unique(df.stat)
    # TODO want to support this?
    assert not (NORM_PREFIX in stat and MEAN_PREFIX in stat)

    assert (MEAN_PREFIX not in stat or stat.startswith(MEAN_PREFIX))
    if strip_mean_prefix:
        stat = stat.replace(MEAN_PREFIX, '')

    assert (NORM_PREFIX not in stat or stat.startswith(NORM_PREFIX))
    if strip_norm_prefix:
        stat = stat.replace(NORM_PREFIX, '')

    return stat


def get_hierarch_clust_fname_prefix(fname_suffix: str, *,
    row_coarsen_factor: Optional[int] = None) -> str:

    if ' - ' in fname_suffix:
        # this is for when -L/--leave-concs is set, so we get names like this:
        # 'clust_hierarch_diag-binaries_2h-7_farn-3'
        # instead of like this:
        # 'clust_hierarch_diag-binaries_2h-7 - farn-3'
        #
        # shouldn't be relevant if -L not passed, where names should be like:
        # 'clust_hierarch_diag-binaries_2h-farn'
        fname_suffix = fname_suffix.replace(' - ', '_')
        assert ' ' not in fname_suffix, f'{fname_suffix=}'

    hierarch_clust_fname_prefix = f'clust_hierarch_{fname_suffix}'
    if row_coarsen_factor is not None:
        hierarch_clust_fname_prefix += f'_row-downsample{row_coarsen_factor}'

    return hierarch_clust_fname_prefix


def get_hierarch_clust_plot_path(plot_dir: Path, fname_suffix: str, **kwargs) -> Path:
    hierarch_clust_fname_prefix = get_hierarch_clust_fname_prefix(
        fname_suffix=fname_suffix, **kwargs
    )
    hierarch_clust_plot_path = plot_dir / f'{hierarch_clust_fname_prefix}.{plot_fmt}'
    return hierarch_clust_plot_path


def get_hierarch_clust_plot_exists_and_warn_if_so(hierarch_clust_plot_path: Path,
    *, warn_: bool = True, show_n_dirs: int = 1) -> bool:
    assert show_n_dirs >= 0
    exists = hierarch_clust_plot_path.exists()
    if warn_ and exists:
        plot_dir = hierarch_clust_plot_path.parent

        parent = plot_dir
        dir_str = ''
        for _ in range(show_n_dirs):
            dir_str = f'{parent.name}/{dir_str}'
            assert parent != parent.parent, \
                f'parent == parent.parent within {show_n_dirs=}'
            parent = parent.parent

        # NOTE: this fn should only be called if ignore_existing=False
        warn(f'not remaking {dir_str}{hierarch_clust_plot_path.name} b/c '
            'ignore_existing=False'
        )

    return exists


# TODO TODO refactor to share w/ natmix_data/analysis.py i copied this from (though
# there is a lot more model specific code currently in there, and right now i just want
# to use this for real KC data here) (and was buried in cluster_rois_and_plot there, not
# a separate fn)
# TODO type hint for linkage?
def plot_hierarch_clustered_rois(plot_dir: Path, df: pd.DataFrame, fname_suffix: str, *,
    ignore_existing: bool = False, title: str = '', cbar_label: Optional[str] = None,
    dendrogram: bool = False, row_colors: bool = False, optimal_ordering: bool = True,
    row_coarsen_factor: Optional[int] = None, wPNKC: Optional[pd.DataFrame] = None,
    kc_spont_in: Optional[pd.Series] = None, warn_: bool = True, **kwargs) -> Optional:

    # TODO also put row_coarsen factor in titles, not just fnames
    hierarch_clust_fname_prefix = get_hierarch_clust_fname_prefix(fname_suffix,
        row_coarsen_factor=row_coarsen_factor
    )
    hierarch_clust_plot_path = get_hierarch_clust_plot_path(plot_dir, fname_suffix,
        row_coarsen_factor=row_coarsen_factor
    )
    if not ignore_existing:
        exists = get_hierarch_clust_plot_exists_and_warn_if_so(
            hierarch_clust_plot_path, warn_=warn_
        )
        if exists:
            return

    vmin = df.min().min()
    vmax = df.max().max()
    # TODO delete? was this just before i was using two slope norm? any reason to keep?
    #if vmax > abs(vmin):
    #    vmin = -vmax

    if row_colors is False:
        row_colors = None

    have_fly_cols = _have_fly_cols(df)
    if have_fly_cols:
        # we should have this if model input, and only model input should have any of
        # the following variables in assertions
        assert all(x is None for x in [wPNKC, kc_spont_in]), \
            'these variables are for model inputs only'

        assert KC_TYPE not in df.columns.names
        if row_colors:
            fly_colors_ser = get_fly_color_series(df)
            row_colors = fly_colors_ser

        n_flies, n_rois = count_flies_and_rois(df)
        # TODO factor this fly counting to a fn (in al_util or hong2p.util)
        n_flies = len(df.columns.to_frame(index=False)[fly_cols].drop_duplicates())
        if title != '':
            title += '\n'
        title += f'n={n_flies}\n# ROIs: {n_rois}'

        if cbar_label is None:
            # TODO TODO update for accuracy
            cbar_label = f'mean Z-scored {dff_latex}'

    # TODO refactor to share w/ mb_model.py (copied from there)
    # (much has been added here since then)
    elif row_colors:
        # TODO TODO add a cbar_label in here (or pass from outside)? can i tell from
        # input whether it's raw or logistic scaled spike counts?

        # TODO move all row_values* up here (including df from concatenating all
        # of them), then start refactoring from there (pass to fn that should handle
        # creation of colors from palettes and handle legends/colorbars as requested)
        # (may need a 2nd fn to draw those legends/colorbars on figures later)

        # this is currently a Series where the index and values are both the same,
        # but could probably just have the values and get the same output when used
        # below. only currently used as <x>.loc[kc_ids], to subset any data that
        # contains more KCs, and to get in saame order.
        kc_ids = df.columns.get_level_values(KC_ID).to_series()
        assert not kc_ids.isna().any()
        assert kc_ids.dtype == int
        assert not kc_ids.index.duplicated().any()
        assert not kc_ids.duplicated().any()

        row_values_list = []
        if KC_TYPE in df.columns.names:
            # TODO rename to kc_types?
            row_values1 = df.columns.to_frame(index=False).set_index(KC_ID)[KC_TYPE]
            row_values_list.append(row_values1)

        # TODO TODO add colorbar(s) for plots that use row_colors (how? want segmented
        # one like place in al_analysis where i use cividis)
        # https://stackoverflow.com/questions/49439408 ?
        # (done?)

        # TODO version sorted by these row_colors?
        # (and for when row_colors are mean distance of claws from CA center, for
        # each KC?)

        # TODO also plot these (w/ same colorscale?) alongside
        # fixed-#-cluster-clustering means?
        # (or just report mean both in and out of multiresponders? barplot for
        # things like that?)

        # TODO TODO define claw_df from wPNKC here? unless claw_df passed in?
        if wPNKC is not None:
            claw_df = None
            if CLAW_ID in wPNKC.index.names:
                claw_df = wPNKC.index.to_frame(index=False).set_index([KC_ID, CLAW_ID])

            # TODO move defs of these up top (/in some other repo)?
            #
            # both lists from Fig 3B of "Structured sampling of olfactory input by the
            # fly mushroom body" by Zheng et al (2022).
            #
            # "core" from main dark green box (top left)
            core_community_gloms = [
                'DM2', 'DP1m', 'VM2', 'DL2v', 'DM3', 'DM4', 'DM1', 'VM3', 'VA2', 'VA4',
            ]
            # "weaker" from the light green group after the "core" PNs.
            # also overconvergent, but not to the same extent.
            weaker_community_gloms = [
                'DM6', 'DM5', 'DC1', 'DL2d', 'VC4', 'VC3l', 'DP1l', 'VA6', 'VM5d',
            ]
            # TODO assert all in wPNKC (/something else)?
            # NOTE: not true for only VC3l
            # TODO TODO address (just warn?)
            #assert all(x in wPNKC.columns
            #    for x in [core_community_gloms + weaker_community_gloms]
            #)

            glomeruli = wPNKC.columns.get_level_values('glomerulus')

            # TODO every need anything more than these?
            to_drop = [x for x in wPNKC.index.names if x not in (KC_ID, CLAW_ID)]

            # TODO check this calculation also works for models that don't split out
            # claws (wPNKC should then contain counts of claws from glom->KC)
            n_core_community_inputs = wPNKC.loc[:, glomeruli.isin(core_community_gloms)
                ].T.sum().droplevel(to_drop).rename('n_core_community_inputs')

            # TODO why need astype(int) now? (will i still if i use as_cmap=True for
            # cmap def? prob not)
            # TODO can i even use as_cmap=True, for things like this where i want a
            # discrete # of colors? sns docs say n_colors= (or its default of 6) is
            # ignored if as_cmap=True
            assert not n_core_community_inputs.isna().any()
            assert pd_allclose(
                n_core_community_inputs, n_core_community_inputs.astype(int)
            )
            n_core_community_inputs = n_core_community_inputs.astype(int)

            n_ext_community_inputs = wPNKC.loc[
                :, glomeruli.isin(core_community_gloms + weaker_community_gloms)
            ].T.sum().droplevel(to_drop).rename('n_ext_community_inputs')

            # TODO why need astype(int) now? (will i still if i use as_cmap=True for
            # cmap def? prob not)
            assert not n_ext_community_inputs.isna().any()
            assert pd_allclose(
                n_ext_community_inputs, n_ext_community_inputs.astype(int)
            )
            n_ext_community_inputs = n_ext_community_inputs.astype(int)

            # TODO does .loc even change anything? is wPNKC already subset to same set?
            # maybe order diff?
            n_core_community_inputs = n_core_community_inputs.loc[kc_ids]
            n_ext_community_inputs = n_ext_community_inputs.loc[kc_ids]

            # TODO why didn't i seem to have to do this in natmix_data/analysis.py?
            if claw_df is not None:
                assert CLAW_ID in n_core_community_inputs.index.names
                assert CLAW_ID in n_ext_community_inputs.index.names
                n_core_community_inputs = n_core_community_inputs.groupby(KC_ID).sum()
                n_ext_community_inputs = n_ext_community_inputs.groupby(KC_ID).sum()

            # TODO remove dupe var
            row_values4 = n_core_community_inputs
            # TODO keep? or pick one between this and core? (compute new quantity as
            # ratio or something?)
            # TODO remove dupe var
            row_values5 = n_ext_community_inputs
            row_values_list.extend([row_values4, row_values5])

        if kc_spont_in is not None:
            # TODO fix for case when kc_spont_in of len # claws (not sure i'll keep that
            # n_claws_active_to_spike code anyway tho...) don't care about spont_in for
            # any KCs not in df, either here or till end of fn
            kc_spont_in = kc_spont_in.loc[kc_ids]
            row_values6 = kc_spont_in
            # TODO say normalized here? unless we can change downstream colormap
            # handling to not require it to be pre-normalized
            row_values6.name = 'spont_in (~threshold)'

            # TODO instead of this hardcode, have fn to create row_values drop all
            # levels not shared by indices of all inputs (and then assert what remains
            # is not duplicated?) (all other ones currently just have KC_ID as their
            # index)
            if KC_TYPE in row_values6.index.names:
                row_values6 = row_values6.droplevel(KC_TYPE)

            row_values_list.append(row_values6)

        if claw_df is not None:
            n_claws_in_surround, n_claws_in_center = analyze_spatial_claws(claw_df)

            # subsetting to just those in df, so max of colormap is max in data
            n_claws_in_surround = n_claws_in_surround.loc[kc_ids]
            n_claws_in_center = n_claws_in_center.loc[kc_ids]

            row_values_list.extend([n_claws_in_surround, n_claws_in_center])

        # TODO or just handle? will this ever be encountered?
        assert len(row_values_list) > 0, ('had no KC_TYPE or any of extra wPNKC/claw_df'
            '/kc_spont_in metadta, at least some of which currently expected for model '
            'outputs'
        )

        # TODO TODO also include wAPLKC and wKCAPL (but may need to handle
        # separately, so maybe not here, esp for one-row-per-claw=True
        # connectome-APL=True cases)
        #
        # or as column colors, once claws are pivoted like that [no, would need separate
        # for each row AND column, since not same weight for same claw ID for two diff
        # KCs.  would need a matrix of same shape as claw responses to show, just except
        # one column group for each odor]? although for non connectome APL version, i
        # suppose this is fine?
        #
        # maybe even sum within each KC? ig then it would be a constant... so wouldn't
        # make sense to plot) (can still show per-claw in a sensible way if we adapt the
        # BY-CLAW versions)

        assert all(isinstance(x, pd.Series) for x in row_values_list)

        assert all(type(x.name) is str for x in row_values_list)
        # TODO TODO fix
        #assert set(x.name for x in row_values_list) == len(row_values_list)

        i0 = row_values_list[0].index
        # TODO TODO fix for case where we have claw inputs too (currently n_*_inputs
        # have KC_ID, CLAW_ID, and kc_type/spont_in just KC_ID
        # TODO TODO shouldn't n_*_inputs be summed over claws anyway? claw_df not in
        # expected form?
        assert all(x.index.equals(i0) for x in row_values_list)
        assert all(x.index.name == i0.name for x in row_values_list)
        assert not i0.duplicated().any()

        # doesn't really matter here, as my own asserts should cover it, but:
        # ...what exactly does verify_integrity=True do when axis='columns'? at least in
        # this call, it does not raise an error if input series have duplicate names
        # (which produces duplicate columns in output df) (is it equivalent to checking
        # if any of inputs have duplicates in row indices?)
        row_values = pd.concat(row_values_list, axis='columns')
        assert row_values.index.equals(i0)
        assert row_values.index.name == i0.name

        # TODO need to do anything to get palette consistent w/ histograms?
        # (seems so, if i care)
        # TODO move both kc_type_palette and type2color to module level? (/delete if i
        # can, after refactoring color handling)
        kc_type_palette = sns.color_palette(n_colors=len(kc_type_hue_order))
        type2color = dict(zip(kc_type_hue_order, kc_type_palette))
        assert all(type(t) is str for t in type2color.keys())
        del kc_type_palette

        # NOTE: currently going to assume that we always want to share colormap ranges
        # for inputs requesting same palette
        name2palette = {
            # TODO or eventually just use default 'tab10' (which should be the default
            # colors originally used to construct type2color, w/ no palette name
            # specified to sns.color_palette)? or still want separate palette for types,
            # shared with other places?
            'kc_type': type2color,

            'n_surround_claws': 'cividis',
            'n_center_claws': 'cividis',

            'n_core_community_inputs': 'magma',
            'n_ext_community_inputs': 'magma',

            # TODO modify hong2p.viz to allow more names in here than we have names
            # in input color dataframe (if were to keep the n_claws_active_to_spike
            # code, would not have spont in here [as currently it would be of len equal
            # to # of claws there, unless i process it back to # KCs somehow])
            # TODO use one of the colorcet ~linear black<->grey ones instead? (this is
            # not perceptually ~linear)
            'spont_in (~threshold)': 'Greys',

            # TODO TODO load claw maxes up here, and compute breadth metrics for
            # each KC (-> use as additional row colors values)?
        }
        row_colors, for_legends, for_cbars = map_each_series_to_rgb(row_values,
            name2palette=name2palette
        )
        # TODO TODO TODO actually use these for_legends below (and replace old that just
        # had kc_type, where applicable)

        assert df.columns.get_level_values(KC_ID).equals(row_values.index)
        assert df.columns.get_level_values(KC_ID).equals(row_colors.index)
        # TODO don't do this? keep as just KC_ID?
        row_values.index = df.columns
        row_colors.index = df.columns

    dilution_factors = None
    # TODO keep this special casing out when refactoring?
    if 'pair_dilution_factor' in df.index.names:
        # should already be sorted and grouped (highest dilution first)
        dilution_factors = df.index.get_level_values('pair_dilution_factor')
        df = df.copy()
        df.index = df.index.get_level_values('odor')

    cbar_orientation = 'horizontal'
    # TODO TODO what do i need to center it with data? (may need to disable and make
    # myself, to really do that nicely...)
    #
    # cbar_pos: (left, bottom, width, height)
    #
    # this puts it in bottom right
    #cbar_pos = (0.65, 0.0, 0.18, 0.015)
    cbar_width = 0.18
    # TODO TODO need to move to move this to the left if fly_colors (or other
    # row_colors), right?
    cbar_pos = (0.5 - cbar_width / 2, 0.0, cbar_width, 0.015)

    discrete = np.allclose(df, df.astype(int))
    nonnegative = (df >= 0).all().all()

    # TODO delete. just a sanity check in here for now.
    stat_part = fname_suffix.split('_')[0]
    if stat_part == 'num-spikes':
        assert discrete
    else:
        assert not discrete, f'{stat_part=}'

    if stat_part == 'logistic-scaled-num-spikes':
        # may not be true? but i think it should be
        assert nonnegative
    #
    if nonnegative:
        cmap = diverging_cmap_tophalf
        # set_bad is all that's needed w/ LogNorm at least (0 values counted as bad)
        cmap.set_bad('w')
        cmap.set_under('w')
    else:
        cmap = diverging_cmap

    if not discrete:
        if not nonnegative:
            # TODO delete
            # TODO TODO is vmin actually working correctly for this?
            # or at least, is cbar actually reflecting correct range, or should i set
            # colorbar=False and add my own via add_colorbar? i think the bluest blue is
            # probably vmin, which should be ~-.5 or something, if from the data, not
            # the -vmax, if size of colorbar is to be believed [negative half has no
            # ticklabels currently tho])
            print('fix negative side of clust hierarch cbars / cbar ticks')
            #
            norm_kws = dict(norm='two-slope', vmin=vmin, vmax=vmax)
        else:
            norm_kws = dict(vmin=vmin, vmax=vmax)
    else:
        assert nonnegative, ('assumed only non-negative spike count data '
            'would be discrete'
        )
        # TODO vmin=0 even work here? (NO!) just make really small? set to 1?
        # (i think i want it to show up as light gray? set_bad/under in cmap above?
        # already done?)
        norm_kws = dict(norm=LogNorm(vmin=0.5, vmax=vmax))

    print('running hierarchichal clustering to make '
        f'{hierarch_clust_plot_path.name}...', end='', flush=True
    )
    ret = cluster_rois(df, cmap=cmap, odor_sort=False, cbar_pos=cbar_pos,
        # TODO delete
        # can't pass str positions to cbar_pos. =(
        #cbar_pos='bottom',
        # setting this None just doesn't plot cbar at all (as docs say). maybe just do
        # that and manually make it myself?
        #cbar_pos=None,
        # didn't work, despite SO post saying it should for at least sns.heatmap:
        # https://stackoverflow.com/questions/47916205
        #cbar_kws=dict(orientation=cbar_orientation, use_gridspec=False, location='bottom'),
        #
        cbar_kws=dict(orientation=cbar_orientation), row_colors=row_colors,
        # TODO restore optimal_ordering=True for inputs under a certain size?
        # and warn if we are disabling it? (move that switch into cluster_rois?
        # still letting this kwarg override that default behavior)
        #
        # default optimal_ordering=True did seem to be the main slowdown
        # (at one point at least, maybe no longer a huge issue?)
        optimal_ordering=optimal_ordering, title=title, cbar_label=cbar_label,
        row_coarsen_factor=row_coarsen_factor, **norm_kws, **kwargs
    )
    row_linkage = None
    if type(ret) is tuple:
        assert kwargs['return_linkages']
        assert len(ret) == 3, 'expected cg and row/col linkage'
        cg, row_linkage, _ = ret
        assert row_linkage is not None
    else:
        cg = ret

    print('done', flush=True)

    # TODO this help w/ cbar positioning? (no, delete)
    #cg.fig.set_layout_engine('none')

    # TODO hide fly colors too? or add legend?
    if not dendrogram:
        cg.ax_row_dendrogram.set_visible(False)

    # TODO keep special casing out?
    if dilution_factors is not None:
        dilution_factor_strs = [
            # 2->'.01x', 1->'.1x', 0->'1x'
            f'{np.power(10.0, -x):.1g}x' if x != 0 else '1x' for x in dilution_factors
        ]
        # TODO i thought there was a 'dilution' label? did i delete that?
        add_group_labels_and_lines(cg.ax_heatmap, x=dilution_factor_strs,
            line_offset=1.0
        )
    #

    # TODO keep, except for the yang plots i wan't to put side-by-side? flag?
    #cg.ax_heatmap.set_ylabel('cell')
    # defaults to something otherwise. 'cell' was just to override default 'kc_id'
    cg.ax_heatmap.set_ylabel('')

    # TODO even need this call? seems for_legends and for_cbars are both empty lists
    # in have_fly_cols case in natmix_data/analysis.py
    for_legends = []
    for_cbars = []
    add_legends_and_colorbars(cg.fig, for_legends, for_cbars)

    assert (
        hierarch_clust_plot_path.name.split(f'.{plot_fmt}')[0] ==
        hierarch_clust_fname_prefix
    ), 'get_hierarch_clust_fname_prefix has diverged from something'
    assert plot_dir == hierarch_clust_plot_path.parent, ('get_hierarch_clust_plot_path '
        'has diverged from actual dir written below'
    )
    # TODO either add a save= kwarg to disable saving (returning plot), or
    # refactor to have one fn that does most and returns plot?
    # or add kwarg to pass in a fn to call before saving?
    # (to avoid baking in the special case stuff to this fn that could otherwise be
    # shared w/ natmix_data/analysis.py /etc)
    savefig(cg, plot_dir, hierarch_clust_fname_prefix, normalize_fname=False)

    return row_linkage


def yang2tom_odor_index(df: pd.DataFrame) -> pd.MultiIndex:
    """Returns new column index for `df`, with levels 'odor' and 'repeat'
    """
    # requiring ')' before '.' b/c don't want to match '.' instead parens like
    # 'farn(-2.5)'
    # TODO just strip last two chars if 2nd-from-last char is '.'? check equiv?
    parts = df.columns.str.split('\)\.')
    parts_len = parts.str.len()

    # so that PFO and other cases can be handled the same way below, to calculate repeat
    # number
    parts = parts.map(lambda x:
        # TODO delete. need more general handling, since we also have stuff like
        # 'Bmix28.1', 'banana.1', etc
        #x[0].split('.') if any(x[0].startswith(y) for y in ('PFO','air')) else x
        # TODO this work? or need to check for absense of ')' / '('
        # TODO assert '(' not in x[0] for all where len(x) == 1?
        x[0].split('.') if (len(x) == 1 and '(' not in x[0]) else x
    )

    # TODO TODO TODO some flies might only have 3-4 trials. check NaNs. maybe drop only
    # to only first # common trials (or average across all later)
    # (i think yang was saying it might only have been for the subdirectory data)
    assert (parts_len <= 2).all()
    # 1,2,...5 (repeat=0 ommitted. that's the case where parts is len 1)
    repeat_strs = parts.map(lambda x: '0' if len(x) == 1 else x[1])
    # cast to int should fail if not possible for all elements
    repeat = repeat_strs.astype(int)
    assert repeat.astype(str).equals(repeat_strs)
    odors = parts.map(lambda x: x[0]).str.strip(')')

    def yang2tom_odor_str(x: str) -> str:
        replace_dict = {
            '(': ' @ ',
            ')': '',
            '+': component_delim,
        }
        for k, v in replace_dict.items():
            x = x.replace(k, v)

        # TODO or return solvent_str?
        if x == 'PFO':
            return 'pfo'

        return x

    # TODO assert same # of duplicates as before? (and 1:1 w/ what we had before?)
    tom_odors = odors.map(yang2tom_odor_str)

    both = pd.concat([
            tom_odors.rename('tom').to_frame(index=False),
            odors.rename('yang').to_frame(index=False)
        ], axis='columns'
    )
    # ipdb> both.drop_duplicates(subset='tom')
    #                       tom              yang
    # 0                 2h @ -8             2h(-8
    # 6                 2h @ -7             2h(-7
    # 12              farn @ -4           farn(-4
    # 18              farn @ -3           farn(-3
    # 24                ma @ -8             ma(-8
    # 30                ma @ -7             ma(-7
    # 36    2h @ -7 + farn @ -3    2h(-7)+farn(-3
    # 42      2h @ -7 + ma @ -7      2h(-7)+ma(-7
    # 48    ma @ -7 + farn @ -3    ma(-7)+farn(-3
    # 54                    pfo               PFO
    # 57            farn @ -2.5         farn(-2.5
    # 63  2h @ -7 + farn @ -2.5  2h(-7)+farn(-2.5
    # 69  ma @ -7 + farn @ -2.5  ma(-7)+farn(-2.5
    #
    # so the string processing shouldn't have changed the information in the strings.
    assert both.drop_duplicates(subset='tom').index.equals(
        both.drop_duplicates(subset='yang').index
    )

    odors = tom_odors.rename('odor')
    odor_index = pd.MultiIndex.from_arrays([odors, repeat.rename('repeat')])
    assert odor_index.get_level_values('odor').equals(odors)

    return odor_index


# TODO TODO add flag to only return wildtype (or at least no TNT (i.e. excluding
# TNT_nolabel)) (+ to parent load_... fn)?
def preprocess_yang_data(df: pd.DataFrame, *, verbose: bool = True,
    drop_expt: bool = True, drop_exp_type: bool = True, _check_farn: bool = False,
    expected_nonmix_odors: Optional[Set[str]] = None) -> pd.DataFrame:
    # TODO doc

    have_exp_type = False
    if 'exp_type' in df.columns:
        index_cols = ['exp_type', 'brain', 'roi']
        have_exp_type = True
    else:
        index_cols = ['brain', 'roi']

    assert not df.loc[:, index_cols].isna().any().any()

    parts = df.brain.str.split('_')
    assert (parts.str.len() == 2).all()
    # TODO explicitly specify format? seems fine w/o...
    # (Yang uses YYYYMMDD format)
    df['date'] = pd.to_datetime(parts.map(lambda x: x[0]))

    rest = parts.map(lambda x: x[1])
    prefix_and_recording_num = rest.str.split('fun')
    assert (prefix_and_recording_num.str.len() == 2).all()
    recording_num = prefix_and_recording_num.map(lambda x: int(x[-1]))
    assert recording_num.min() == 1
    if recording_num.max() > 1:
        # TODO TODO make sure these are getting dropped too if needed. add separate col
        # w/ recording # if we have any that are >1?
        # (check handling in one_recording_per_fly code)
        df['recording_num'] = recording_num
    # TODO delete. subdir has some 'fun2' suffices
    #assert rest.str.endswith('fun1').all()

    # TODO also convert to int? assert all positive? assert no dupes (apart from side)
    # within day? assert all consecutive w/in day (don't really care...)?
    df['fly_num'] = rest.map(lambda x: int(x[0]))
    assert not df.fly_num.isna().any()

    df['sex'] = rest.map(lambda x: x[1])
    # TODO drop the one 'M' or did it only have TNT_label data anyway?
    assert {'F'} <= set(df.sex.unique()) <= {'F', 'M'}

    df['side'] = rest.map(lambda x: x[2])
    assert set(df.side.unique()) == {'L', 'R'}

    if have_exp_type:
        recording_cols = ['exp_type'] + fly_cols + ['side']
    else:
        recording_cols = fly_cols + ['side']

    if 'recording_num' in df.columns:
        recording_cols += ['recording_num']

    assert (df.groupby(fly_cols).sex.nunique() == 1).all()

    # TODO maintain sex? warn if not all the same?
    to_drop = ['sex']
    if drop_expt:
        # TODO rename this col expt then?
        to_drop += ['brain']
    else:
        recording_cols += ['brain']

    df = df.drop(columns=to_drop)

    # TODO do this (and reset_index() again?) before str processing currently above?
    # or just do (something like this) after str processing, using some of those cols in
    # place of some of these?
    df = df.set_index(recording_cols + ['roi'], verify_integrity=True)

    orig_n_odors = len(df.columns)

    # TODO should i lump data from all of exp_type=['WT', 'TNTin_nolabel',
    # 'TNTin_label'] cases (everything but 'TNT_label') together?
    # that's what i'm doing here now:
    # TODO add flag to just use WT. might be safer. compare (not sure spread looks any
    # better there, esp w/o tossing an outlier. from looking at first side of each at
    # least)
    if have_exp_type:
        exp_type_to_drop = 'TNT_label'
        exp_types = df.index.get_level_values('exp_type')
        exp_types_to_keep = exp_types != exp_type_to_drop
        if verbose and (~ exp_types_to_keep).any():
            # "...will lump all data from exp_types=['WT', 'TNTin_nolabel',
            # 'TNTin_label'] together"
            warn(f'dropping exp_type={exp_type_to_drop}. will lump all data from '
                f'exp_types={list(exp_types[exp_types_to_keep].unique())} together.'
            )

        df = df[exp_types_to_keep].copy()
        if drop_exp_type:
            df = df.droplevel('exp_type')

        recording_cols = [x for x in recording_cols if x != 'exp_type']

    # TODO (delete) so were flies only numbered within experimental conditions per day?
    # or was this a mistake? oh, ig one side was labelled and the other wasn't.
    # TODO may still want to assert we don't have certain combos of exp_type
    # within a fly, but we shouldn't anyway
    # MultiIndex([(  'TNTin_label', '2025-03-21', 3, 'L'),
    #             ('TNTin_nolabel', '2025-03-21', 3, 'R')],
    # TODO what is dropping these two flies (after comparison w/ one_recording_per_fly
    # above)? is it just that these two are only TNT_label flies? (probably, but assert
    # that?)
    # ipdb> one_recording_per_fly.droplevel('side').difference(df.index.droplevel('roi'
    #  ).drop_duplicates())
    # MultiIndex([('2025-02-14', 4),
    #             ('2025-02-19', 3)],
    #            names=['date', 'fly_num'])
    assert not df.index.duplicated().any()

    # TODO if refactoring, add flag to not drop any side (/recordings) (might want to
    # drop duplicate recordings, but keep multiple sides [at least, that's how yang
    # operates i believe])?
    # (and make sure rest of fn actually supports it. maybe add to some internal var
    # like fly_cols, but that optionally includes 'side' too?)
    #
    # one fly has two exp_type values (b/c one side labelled and one is not)
    # TODO should exp_type be in this or not (now that i've added
    # drop_exp_type[=False])?
    non_recording_cols = ['roi']
    recording_index = df.index.droplevel(non_recording_cols)
    # TODO use .last() instead? might select the later recordings yang seems to think
    # are often more stable (but later). flag to select between the two?
    one_recording_per_fly = pd.MultiIndex.from_frame(recording_index.drop_duplicates(
        ).to_frame(index=False).groupby(fly_cols).first().reset_index()
    )
    # TODO doesn't break anything in drop_exp_type=True (old) case, right?
    one_recording_per_fly = one_recording_per_fly.reorder_levels([
        x for x in df.index.names if x not in non_recording_cols
    ])
    #
    assert not one_recording_per_fly.droplevel('side').duplicated().any()
    recording_to_keep_mask = recording_index.isin(one_recording_per_fly)
    if verbose and (~ recording_to_keep_mask).any():
        expts_dropped = df[~recording_to_keep_mask].index.droplevel('roi'
            ).drop_duplicates()

        warn('dropping the following experiments, for which each fly has another '
            'recording (probably on the other side) not dropped:\n'
            f'{expts_dropped.to_frame().to_string(index=False)}'
        )
    # NOTE: probably important that this dropping happens after dropping based on
    # exp_type. In case we accidentally select only the TNT_label case for a fly that
    # also has TNT_nolabel.
    non_fly_cols = [x for x in df.index.names if x not in fly_cols]
    flies_before = df.index.droplevel(non_fly_cols).drop_duplicates().sort_values()

    df = df[recording_to_keep_mask].copy()
    assert df.index.droplevel(non_recording_cols).drop_duplicates().sort_values(
        ).equals(one_recording_per_fly.sort_values())

    flies_after = df.index.droplevel(non_fly_cols).drop_duplicates().sort_values()
    assert flies_before.equals(flies_after)

    # if this 1st assertion fails, code above is not working correctly
    assert (
        len(df.index.droplevel('roi').drop_duplicates()) == len(flies_after)
    )
    to_drop = set(non_fly_cols) - {'roi'}
    if not drop_expt:
        assert len(df.index.get_level_values('brain').unique()) == len(flies_after)
        to_drop -= {'brain'}

    if not drop_exp_type:
        assert len(df.index.droplevel([
            x for x in df.index.names if x not in fly_cols + ['exp_type']]
            ).drop_duplicates()
        ) == len(flies_after)
        to_drop -= {'exp_type'}

    assert {'side'} <= to_drop <= {'side', 'recording_num'}, f'{to_drop=}'

    df = df.droplevel(list(to_drop))
    assert not df.index.duplicated().any()
    recording_cols = [x for x in recording_cols if x not in to_drop]

    odor_index = yang2tom_odor_index(df)
    odors = odor_index.get_level_values('odor')
    # TODO replace some of above odor index construction w/ this? factor into one fn?
    #
    # TODO TODO this not working? why farn still after ma in ma+farn?
    # TODO TODO oh, maybe strs aren't even being derived from this? probably should be
    # tho...
    odor_lists = odors.map(lambda x: olf.parse_odor_list(x))

    odor_index = olf.odor_lists_to_multiindex(odor_lists)
    assert odor_index.names == ['odor1', 'odor2', 'repeat']

    mix_rows = (odor_index.droplevel('repeat').to_frame() != solvent_str).T.all()
    assert (mix_rows | (odor_index.get_level_values('odor2') == solvent_str)).all()
    mix_index = mix_rows[mix_rows].index
    mix_components = set(mix_index.get_level_values('odor1').unique())
    mix_components |= set(mix_index.get_level_values('odor2').unique())
    assert solvent_str not in mix_components
    # TODO assert all mixes are only preset in a single odor1/2 order?
    # (should be guaranteed by my olf fns?)

    # NOTE: mismatches between her main concs and diags:
    # - 2h @ -6 in diag, -8 here
    # - farn @ -2 in diag, -2.5/-3 here
    # - ma @ -7 in diag, and here (-8 in diag ever? why was i thinking one was at -8?)
    # TODO how similar is her component data from one vs the other conc? she have both
    # in any flies (if so, how many?)?
    # TODO do all flies only have one pair or the other? or some flies have both?
    #
    # ipdb> pp [x for x in odor_index.get_level_values('odor').unique()]
    # ['2h @ -8',
    #  '2h @ -7',
    #  'farn @ -4',
    #  'farn @ -3',
    #  'ma @ -8',
    #  'ma @ -7',
    #  '2h @ -7 + farn @ -3',
    #  '2h @ -7 + ma @ -7',
    #  'ma @ -7 + farn @ -3',
    #  'pfo',
    #  'farn @ -2.5',
    #  '2h @ -7 + farn @ -2.5',
    #  'ma @ -7 + farn @ -2.5']
    mix_or_comp_mask = (
        odors.isin(mix_components) | odors.str.contains(component_delim, regex=False)
    )
    nonmix_odors = set(odors[~mix_or_comp_mask].unique())
    if verbose and len(nonmix_odors) > 0:
        warn('dropping the following odors, which are not components of any mixture:\n'
            f'{nonmix_odors}'
        )

    if expected_nonmix_odors is not None:
        assert expected_nonmix_odors == nonmix_odors, (
            f'{expected_nonmix_odors=}\n{nonmix_odors=}'
        )
    else:
        if verbose:
            warn('set expected_nonmix_odors=<set-of-odor-strs> to enable assertion')

    odor_index = olf.add_mix_str_index_level(odor_index)
    assert odor_index.names == ['odor1', 'odor2', 'repeat', 'odor']
    df.columns = odor_index.droplevel(['odor1', 'odor2']).reorder_levels(
        ['odor', 'repeat']
    )
    assert len(odors) == len(df.columns) == orig_n_odors
    df = df.loc[:, mix_or_comp_mask].copy()
    odors = df.columns.get_level_values('odor')
    mix_mask = odors.str.contains(component_delim, regex=False)
    comps = odors[~mix_mask]

    metadata = df.groupby('odor', axis='columns').mean().stack(
        ).droplevel(['roi']).index.to_frame(index=False)
    recordings_per_odor = metadata.drop_duplicates(['odor'] + recording_cols)
    assert metadata.drop_duplicates(['odor'] + fly_cols)[fly_cols].equals(
        recordings_per_odor[fly_cols]
    )
    if verbose:
        print(f'components: {sorted(comps.unique())}')

        recordings_per_mix = recordings_per_odor[recordings_per_odor.odor.str.contains(
            component_delim, regex=False)].odor.value_counts().sort_index().rename(
            'count').rename_axis('mix').to_frame()

        print('mixes (-> # flies):\n' + recordings_per_mix.to_string())
        print('mixes (ignoring concs): ' + str(sorted(
            odors[mix_mask].unique().map(olf.strip_concs_from_odor_str).unique()
        )))

    # TODO use something like this in main to make summary plots (to merge this
    # data w/ model KC data using different concentrations) (doing that. just need to
    # warn now and can delete this). would need to separately warn there.
    without_concs = comps.map(olf.strip_concs_from_odor_str)

    # TODO diag farn conc is -2, so maybe just keep -2.5 actually? what fraction of
    # data is that?
    # TODO still decide between 'farn @ -3' and '@ -2.5'? or average both together
    # and rename to one (leaning towards that, but will handle outside of this fn)?
    # (diag is -2. see counts of each reported below, for her two concs)
    # TODO TODO refactor to share this code w/ get_yang... fn, to warn near end, where i
    # actually use this stuff (and where i actually want to strip concs)
    # (or just move this code there? any point to having it here?)
    odors_at_multiple_concs = pd.DataFrame({
        'name': without_concs, 'odor': comps
    }).drop_duplicates().groupby('name').filter(lambda x: len(x) > 1)
    if len(odors_at_multiple_concs) > 0:
        # TODO actually implement dropping/merging of these odors at mulitiple concs
        # (no, delete)? (or maybe only when actually making summary plots including
        # both, so i can warn about concentration differences there, when comparing
        # model [based on my diags] and KC stuff)
        if verbose:
            warn('the following odors are at multiple concentrations:\n'
                f'{odors_at_multiple_concs.odor.sort_values().to_string(index=False)}'
            )
        # oh, it looks like all the flies have 'farn @ -3' (at least those remaining
        # here), and some of them also have 'farn @ -2.5':
        # ipdb> df.loc[:, odors.isin(odors_at_multiple_concs.odor)].groupby('odor',
        #    axis='columns').mean().isna().sum()
        # odor
        # farn @ -2.5    11810
        # farn @ -3          0
        #
        # TODO just keep 'farn @ -3' then? to not end up averaging across flies?

        recordings_per_multipleconc_odor = recordings_per_odor[
            recordings_per_odor.odor.isin(odors_at_multiple_concs.odor)
        ].reset_index(drop=True)

        # TODO delete
        recordings_per_odor2 = df.loc[:, odors.isin(odors_at_multiple_concs.odor)
            ].groupby('odor', axis='columns').mean().stack().droplevel(['roi']
            ).index.to_frame(index=False).drop_duplicates(['odor'] + fly_cols
            ).reset_index(drop=True)
        assert recordings_per_odor2.equals(recordings_per_multipleconc_odor)
        #
        n_recordings_per_odor = recordings_per_odor.odor.value_counts().sort_index()
        if verbose:
            warn('# recordings per odor (for odors at multiple concs):\n' +
                n_recordings_per_odor.to_string(name=False)
            )

        # TODO move outside this fn? would be hard to refactor tho... put behind some
        # flag / extra kwarg indicating this is the check to make?
        if _check_farn:
            recordings_per_odor = recordings_per_odor.set_index(['odor'] + fly_cols,
                verify_integrity=True
            )
            farn3_flies = recordings_per_odor.loc['farn @ -3'].index.sort_values()
            farn2p5_flies = recordings_per_odor.loc['farn @ -2.5'].index.sort_values()
            assert len(farn2p5_flies.difference(farn3_flies)) == 0

            all_flies = recordings_per_odor.droplevel('odor').index.drop_duplicates(
                ).sort_values()
            assert all_flies.equals(farn3_flies)

    isna = df.isna()
    assert not isna.all().any()
    assert not isna.T.all().any()
    # TODO also check for cols all NaN w/in fly (e.g. to detect flies that only have 3-4
    # instead of 6 trials)

    fly2n_repeats = df.groupby(fly_cols, group_keys=False).apply(lambda x:
        x.dropna(how='all', axis='columns').columns.get_level_values('repeat').max()
    )
    unique_n_repeats = fly2n_repeats.unique()
    if verbose and len(unique_n_repeats) > 1:
        warn('not all flies had same # of repeats!\n# repeats -> # flies with that many'
            f':\n{fly2n_repeats.value_counts().to_string()}'
        )
    # TODO (delete) maybe dropping that single subdir fly w/ only 2 trials. don't think
    # i want to drop the other flies in subdir down from 3-4 to only 2 trials...
    # TODO and/or drop single parent dir fly w/ 3 instead of 5 repeats?
    # TODO could maybe do that as well as dropping all 4->3?
    # TODO did any other recording for any of these flies have more repeats? also
    # select based on that, when picking recording?

    # assuming this pre-sorting is necessary to get consistent fly_ids in next step
    df = df.sort_index(level=fly_cols, kind='stable')
    index_df = df.index.to_frame(index=False)

    # TODO also delete date/fly_num cols (nah)? or concat them into one str ID
    # rather than this integer one?
    # TODO also do this for natmix data i'm also comparing some stuff against
    index_df = add_group_id(index_df, fly_cols, 'fly_id')
    # TODO add recording_id too? esp if i'm including e.g. multiple sides for one
    # fly
    df.index = pd.MultiIndex.from_frame(index_df)

    put_at_end = ['roi']
    if have_exp_type and not drop_exp_type:
        put_at_end.append('exp_type')

    assert all(x in df.index.names for x in put_at_end)
    df = df.reorder_levels(
        [x for x in df.index.names if x not in put_at_end] + put_at_end
    )
    # TODO also sort_index() before returning?

    if verbose:
        print()

    return df


def load_yang_kc_data(*, drop_exp_type: bool = True, verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns concatenated response amplitude and binarized response data.
    """
    # TODO (delete) is all this data (both yang_dir and yang_subdir contents) stuff i
    # have run through the model for her (no, concs differ somewhat, at lesat for
    # yang_dir contents)? or can i not generate appropriate input for some of them
    # (maybe for subdir? do i or sam have right concs anywhere?)

    # should contain binary diagnostic odor mix data
    yang_dir = data_root / 'from_yang/20260415_YZtransfer'

    # only file in yang_dir with responses is concatenated data in:
    # 20250129_0326_KC_odormix_resp_good.csv
    mean_csv = yang_dir / '20250129_0326_KC_odormix_resp_good.csv'
    # col 0 is just ROI number (seemingly concatenated across recordings). don't see any
    # other fly/recording IDs in this output.
    df = pd.read_csv(mean_csv, index_col=0)
    # (delete) always pick L/R for all? average / compute across the two within each?
    # yang have a record of which came first, or which was better, for each fly?
    # (first yes, it's in filename / brain ID, but better not really. she things later
    # is generally more stable, but I might prefer earlier typically. shouldn't really
    # matter)
    #
    # (delete) do all flies have both hemispheres (not quite)?
    # ipdb> df.loc[:, df.dtypes == np.dtype('O')].drop_duplicates()
    #            brain       exp_type
    # 20250129_1FLfun1             WT
    # 20250129_1FRfun1             WT
    # 20250129_2FLfun1             WT
    # 20250129_2FRfun1             WT
    # 20250320_1FLfun1             WT
    # 20250320_1FRfun1             WT
    # 20250320_2FLfun1             WT
    # 20250214_2FRfun1  TNTin_nolabel
    # 20250321_3FRfun1  TNTin_nolabel
    # 20250221_1FRfun1    TNTin_label
    # 20250224_2FLfun1    TNTin_label
    # 20250224_3FLfun1    TNTin_label
    # 20250321_3FLfun1    TNTin_label
    # 20250324_1FRfun1    TNTin_label
    # 20250324_2FLfun1    TNTin_label
    # 20250214_3MLfun1    TNTin_label
    # 20250214_4FLfun1      TNT_label
    # 20250219_3FLfun1      TNT_label
    # 20250219_3FRfun1      TNT_label
    df.index.name = 'roi'
    df = df.reset_index()
    df = preprocess_yang_data(df, drop_exp_type=drop_exp_type, drop_expt=False,
        # TODO have pfo in there by default?
        expected_nonmix_odors={'pfo', '2h @ -8', 'farn @ -4', 'ma @ -8'},
        _check_farn=True, verbose=verbose
    )

    # should be able to calculate:
    # All_resprate_respthresh1_trialfr0p5_wmixpred.csv from per fly files like:
    # 20250129_1FLfun1_resp_binary_respthresh1_trialfr0p5_wmixpred.csv
    #
    # same for both yang_dir and subdir? no, for subdir it's:
    # _responsive_binlabel_df_respthresh1_trialfr0.5.csv
    binarized_suffix = '_resp_binary_respthresh1_trialfr0p5_wmixpred.csv'
    fly_bin_dfs = []
    for resp_csv in sorted(yang_dir.glob(f'*{binarized_suffix}')):
        # NOTE: these also have columns like:
        # - 'experiment' (i assume this should all match prefix of file)
        # - '2h(-7)+ma(-7)_predicted' (which can all be dropped?)
        fly_bin_df = pd.read_csv(resp_csv, index_col=0)
        fly_bin_df.index.name = 'roi'

        expts = fly_bin_df['experiment'].unique()
        assert len(expts) == 1
        expt = expts[0]
        recording_id = resp_csv.name[:-len(binarized_suffix)]
        assert expt == recording_id
        fly_bin_df = fly_bin_df.drop(columns='experiment')

        # yang says this is just union of whether the cell responded to the components.
        # don't need that.
        fly_bin_df = fly_bin_df.drop(columns=[
            x for x in fly_bin_df.columns if x.endswith('_predicted')
        ])
        fly_bin_df = addlevel(fly_bin_df, 'brain', recording_id)
        fly_bin_dfs.append(fly_bin_df.astype(float))

    bdf = pd.concat(fly_bin_dfs, verify_integrity=True)

    df_recordings = set(df.index.get_level_values('brain'))
    df = df.droplevel('brain')

    bdf = bdf[bdf.index.get_level_values('brain').isin(df_recordings)].copy()
    bdf = preprocess_yang_data(bdf.reset_index(), drop_exp_type=drop_exp_type,
        verbose=False, expected_nonmix_odors={'pfo', '2h @ -8', 'farn @ -4', 'ma @ -8'},
        _check_farn=True
    )
    # seems like maybe neither way already sorted? at least on rows?
    df = df.sort_index()
    bdf = bdf.sort_index()
    # do seem to need this too, at least on one
    df = df.sort_index(axis='columns')
    bdf = bdf.sort_index(axis='columns')
    if drop_exp_type:
        assert df.index.equals(bdf.index)
    else:
        assert df.index.droplevel('exp_type').equals(bdf.index)
        # to add exp_type level (the last level) to bdf.index
        bdf.index = df.index.copy()

    # TODO refactor to share w/ assertions below?
    assert df.columns.get_level_values('odor').unique().equals(
        bdf.columns.get_level_values('odor')
    )
    assert (bdf.columns.get_level_values('repeat') == 0).all()
    assert df.columns.get_level_values('repeat').max() > 0
    #

    # should contain ma+2h and ea+eb data. all wildtype (exp_type='WT' from above)
    yang_subdir = yang_dir / '20260116_20260210'

    # should be able to calculate all (e.g.):
    # 20260121_2FLfun1_resp_rate_respthresh1trialfr0p5_wmixpred.csv
    # (which just contains mean response rates per odor) from (e.g.):
    # 20260121_2FLfun1_responsive_binlabel_df_respthresh1_trialfr0.5.csv
    binarized_suffix = '_responsive_binlabel_df_respthresh1_trialfr0.5.csv'

    # the 3rd and final file for each fly should have responses:
    # 20260116_1FLfun1_els_moco_s2p_mean_curated_Fc_zscored_full_cell_resp_df_odor_sorted.csv
    #
    # all flies in this subfolder should be "good", and can thus be concatenated
    # together to create comparable input as loaded from the single "good" CSV in
    # yang_dir
    response_csv_suffix = (
        '_els_moco_s2p_mean_curated_Fc_zscored_full_cell_resp_df_odor_sorted.csv'
    )
    fly_resp_csvs = sorted(yang_subdir.glob(f'*{response_csv_suffix}'))
    # the '*.csv' glob is really just to exclude stuff like '.~lock.<filename>.csv#'
    # when opening CSVs in OpenOffice
    all_subdir_files = [x for x in yang_subdir.glob('*.csv') if x.is_file()]
    assert len(fly_resp_csvs) * 3 == len(all_subdir_files), \
        'expected 3 files for each fly, and nothing else'

    fly_dfs = []
    fly_bin_dfs = []
    for resp_csv in fly_resp_csvs:
        # TODO note some/all of these have odor='air'. make sure that's also being
        # dropped (should be)
        fly_df = pd.read_csv(resp_csv, index_col=0)
        fly_df.index.name = 'roi'
        # all columns should be odor
        # TODO assert all columns same across CSVs? how else to concat? just let there
        # be NaNs for some odor X repeat combos, and handle inside preproces fn
        # probably
        recording_id = resp_csv.name[:-len(response_csv_suffix)]

        # TODO TODO oh, so these are collapsed over trials? want that? have some
        # version where it's not?
        fly_bin_df = pd.read_csv(yang_subdir / f'{recording_id}{binarized_suffix}',
            index_col=0
        )
        fly_bin_df.index.name = 'roi'

        odor_index1 = yang2tom_odor_index(fly_df)
        odors1 = odor_index1.get_level_values('odor').unique()

        odor_index2 = yang2tom_odor_index(fly_bin_df)
        odors2 = odor_index2.get_level_values('odor').unique()
        assert len(set(odors2)) == len(fly_bin_df.columns)
        assert (odor_index2.get_level_values('repeat') == 0).all()

        assert odors1.equals(odors2)
        max_n_trials = odor_index1.get_level_values('repeat').max() + 1
        # NOTE: only one fly had only 2 trials. drop this fly? rest had 3 or 4.
        # recording_id='20260116_1FLfun1'
        # max_n_trials=2
        # recording_id='20260121_2FLfun1'
        # max_n_trials=4
        # recording_id='20260121_2FLfun2'
        # max_n_trials=3
        # recording_id='20260121_2FRfun1'
        # max_n_trials=3
        # recording_id='20260202_1FLfun1'
        # max_n_trials=3
        # recording_id='20260202_1FRfun1'
        # max_n_trials=3
        # recording_id='20260203_2FLfun1'
        # max_n_trials=3
        # recording_id='20260203_2FRfun1'
        # max_n_trials=3
        # recording_id='20260210_2FRfun1'
        # max_n_trials=4
        #
        # TODO delete
        #print(f'{recording_id=}')
        #print(f'{max_n_trials=}')
        #

        fly_df = addlevel(fly_df, 'brain', recording_id)
        fly_df = addlevel(fly_df, 'exp_type', 'WT')
        fly_dfs.append(fly_df)

        fly_bin_df = addlevel(fly_bin_df, 'brain', recording_id)
        fly_bin_df = addlevel(fly_bin_df, 'exp_type', 'WT')
        # TODO casting to float to cause a bit less of a headache when concat later
        # introduces NaN. currently get a FutureWarning on first attempt of using the
        # output of that concatenation:
        # FutureWarning: In a future version, object-dtype columns with all-bool values
        # will not be included in reductions with bool_only=True. Explicitly cast to
        # bool dtype instead.
        fly_bin_dfs.append(fly_bin_df.astype(float))

    # TODO assert set of columns (union across all input dfs) is same as output columns
    # of concat?
    df2 = pd.concat(fly_dfs, verify_integrity=True)
    bdf2 = pd.concat(fly_bin_dfs, verify_integrity=True)

    # TODO (delete) is it only 2 repeats for each of these? or varying number?
    # (2 for one fly, 3-4 for rest)
    df2 = preprocess_yang_data(df2.reset_index(), drop_exp_type=drop_exp_type,
        expected_nonmix_odors={'pfo', 'Iaa @ -2', 'banana', 'Bmix28', 'air'},
        verbose=verbose
    )
    bdf2 = preprocess_yang_data(bdf2.reset_index(), drop_exp_type=drop_exp_type,
        verbose=False,
        expected_nonmix_odors={'pfo', 'Iaa @ -2', 'banana', 'Bmix28', 'air'}
    )

    assert df2.index.equals(bdf2.index)
    assert df2.columns.get_level_values('odor').unique().equals(
        bdf2.columns.get_level_values('odor')
    )
    assert (bdf2.columns.get_level_values('repeat') == 0).all()
    assert df2.columns.get_level_values('repeat').max() > 0

    # so despite there being '2h @ -7 + ma @ -7' (+ component) data in both datasets,
    # none of the flies overlap between the datasets
    # TODO drop it from the subdir dataset? or also plot it there?
    # otherwise, subdir dataset has (at two concs each) ea+eb and 2h+1o3ol, though
    # probably don't have any ideal ORN data to compare it too, since all lower than
    # concs I've used other than in the (poorly done / analyzed) ramp experiments
    assert len(df.index.droplevel('roi').drop_duplicates().intersection(
        df2.index.droplevel('roi').drop_duplicates()
    )) == 0

    # TODO better names than df and df2?
    df = addlevel(df, 'panel', 'diag-binaries')
    bdf = addlevel(bdf, 'panel', 'diag-binaries')

    # TODO TODO also do any other analysis on this elsewhere?
    df2 = addlevel(df2, 'panel', 'natmix-top2-dilute')
    bdf2 = addlevel(bdf2, 'panel', 'natmix-top2-dilute')

    df = pd.concat([df, df2], verify_integrity=True)
    bdf = pd.concat([bdf, bdf2], verify_integrity=True)

    # averaging over repeats
    # NOTE: need level= (in current pandas) or else sort=False doesn't behave
    df = df.groupby(level='odor', axis='columns', sort=False).mean()

    assert (bdf.columns.get_level_values('repeat') == 0).all()
    bdf = bdf.droplevel('repeat', axis='columns')

    assert df.columns.equals(bdf.columns)
    assert df.index.equals(bdf.index)
    assert df.isna().equals(bdf.isna())

    return df, bdf


# TODO factor some/all of this to mb_model?
#
# separates all parameters in filenames
fname_param_delim: str = '__'
# this, and everything after it, should be the fixed threshold and APL weight
# scales set via tuning on tune_root (megamat), and then applied to a model
# otherwise sharing the same parameters, but run on whichever other panel (e.g.
# kiwi/control)
tune_params_delim: str = f'{fname_param_delim}fixed-thr_'
# TODO TODO add fn for getting all dirs under non-tuned panel dir that match current
# tuned model_str
def model_str2matching_pretuned_dir(model_str: str, pretuned_panel_dir: Path
    ) -> Optional[Path]:
    """Returns path to matching non-tuned directory, or None if there isn't one.

    Everything any matching directory name contains after `tune_params_delim` should
    all relate to parameters fixed by the pre-tuning process (e.g. the fixed threshold
    or APL weight scale(s)).

    Args:
        model_str: the exact name of the tuned directory, that set the parameters for
            the matching pre-tuned model run (ifthere were one matching). There should
            also be at most one directory in `pretuned_panel_dir` that starts with
            `f{model_str}{tune_params_delim}`.

        pretuned_panel_dir: directory under which model output directories exist, all of
            which should be pre-tuned on some other panel (and thus have
            `tune_params_delim` and following to indicate those parameters fixed by the
            pre-tuning on the other panel)
    """
    # TODO assert *all* contents of pretuned_panel_dir contain tune_params_delim,
    # rather than just checking those that start with model_str? shouldn't matter...
    prefix = f'{model_str}{tune_params_delim}'
    matching_dirs = [x for x in pretuned_panel_dir.glob(f'{prefix}*/') if x.is_dir()]

    assert len(matching_dirs) <= 1, (f'only <=1 directory should start with {prefix}! '
        f'got:\n{pformat(matching_dirs)}'
    )
    # TODO assert none are just model_str exactly? (or like in comment above, that
    # everything contains tune_params_delim?)
    if len(matching_dirs) == 0:
        return None

    return matching_dirs[0]
#

# TODO move to hong2p.viz?
# TODO type hint for artist?
def artist_var(artist, name: str) -> Any:
    return getattr(artist, f'get_{name}')()

# TODO type hint for artist?
# TODO move to hong2p.viz?
def artist_rgb_and_alpha(artist) -> Tuple[Tuple[float, float, float], float]:
    try:
        color = artist.get_color()
        # assume that this one will always be available if first works?
        alpha = artist.get_alpha()

    except AttributeError:
        if isinstance(artist, BarContainer):
            # should all be matplotlib.patches.Rectangle
            rects = artist.get_children()

            def get_single(name: str) -> Any:
                unique = set(artist_var(x, name) for x in rects)
                assert len(unique) == 1
                return unique.pop()

            # NOTE: this does not seem to correspond to fill= from
            # histplot call. it seems always True. whether edge or face
            # color should be used *does* seem to depend on outer fill=
            # tho, and can be inferred from values of [edge|face]color
            # in here
            # TODO delete?
            fill = get_single('fill')
            assert fill, 'seemed independent of histplot fill= value'
            #

            # seems to be RGBA, w/ alpha currently 0.5 in first call
            edgecolor = get_single('edgecolor')

            # seems to be RGBA, w/ alpha currently 0.5 in first call
            facecolor = get_single('facecolor')

            # this was the case, at least for element='bars' fill=True,
            # on first distplot call checked. could relax (or use this
            # color if facecolor not set appropriately for some reason)
            if edgecolor == (0, 0, 0, 1):
                color = facecolor
                set_color_fn = lambda x: (
                    r.set_facecolor(x) for r in rects
                )
            else:
                # seems to be what we get w/ histplot fill=False
                assert facecolor == (0, 0, 0, 0)

                color = edgecolor
                set_color_fn = lambda x: (
                    r.set_edgecolor(x) for r in rects
                )

            alpha = get_single('alpha')

        # below for matplotlib.collections.PolyCollection, created by
        # (at least) fill=True w/ element='step', in sns.histplot.
        # code might not work for other artist types (like
        # BarCollection), but many other artists may still have
        # edgecolor, so not only using this code in that case
        else:
            color = artist.get_edgecolor()
            # letting assertions below handle other color shapes
            if len(color) == 1:
                color = tuple(color[0])

            alpha = artist.get_alpha()
            # TODO get 4th element of color (edge or face tho? they
            # differ in first case i checked.  1.0 for edge and 0.5 for
            # face), and use instead of get_alpha() below? at least
            # get_alpha is defined here.
            # ipdb> artist.get_facecolor()
            # array([[0.12, 0.47, 0.71, 0.5 ]])
            # ipdb> artist.get_edgecolor()
            # array([[0.12, 0.47, 0.71, 1.  ]])

            # NOTE: set_color also defined (as is set_[edge|face]color)
            # presumably it sets them both to the same?

    if alpha is not None:
        assert len(color) == 3, assert_msg
        rgb = color
    else:
        assert len(color) == 4, assert_msg
        rgb = color[:-1]
        alpha = color[-1]

    assert is_scalar(alpha)

    # just in case it was a list/array
    rgb = tuple(rgb)
    return rgb, alpha


# TODO uppercase all / most of these?
sorted_mixsupp_fname_prefix: str = 'model-mix-supps_sorted-by-mean-natmix_'
order_by_mean_mixsupp_from: str = 'logistic_scaled_num_spikes'
mixsupp_order_fname_suffix: str = '_panelmean.parquet'
# TODO (done?) share w/ below? (bottom, near parquet saving)
mixsupp_prefix: str = 'mixsupp_'
mixsupp_col: str = f'{mixsupp_prefix}{order_by_mean_mixsupp_from}'

def stat2fname_part(stat: str) -> str:
    logistic_scaled_prefix = 'logistic_scaled'
    if stat.startswith(logistic_scaled_prefix):
        expected_suffix = '_num_spikes'
        assert stat[len(logistic_scaled_prefix):] == expected_suffix, ('if suffix not '
            f'always {expected_suffix}, can not set stat="logistic_scaled" for fname'
        )
        # just to shorten this part of fnames (spikes are the only thing currently
        # logistic scaled)
        stat = 'logistic_scaled'

    return stat.replace('_', '-')


def model_root2tune_root(model_root: Path) -> Path:
    tune_root = model_root / 'megamat-tuned'
    return tune_root


# TODO rename?
def mixsupp_parts2order_fname(stat: str, logistic_scaling_fname_part: str = '') -> str:
    # NOTE: expact stat to start with mixsupp_prefix. something like:
    # ['mixsupp_logistic_scaled_num_spikes', 'mixsupp_num_spikes']
    stat_for_fname = stat2fname_part(stat)
    assert stat.startswith(mixsupp_prefix), (
        f'expected {stat=} to start with {mixsupp_prefix=} (and end with one of stats)'
    )
    return (f'{sorted_mixsupp_fname_prefix}{stat_for_fname}'
        f'{logistic_scaling_fname_part}{mixsupp_order_fname_suffix}'
    )


def mixsupp_order_fname2shared_prefix(mixsupp_order_fname: str) -> str:
    shared_prefix, rest = mixsupp_order_fname.split(mixsupp_order_fname_suffix)
    assert rest == ''
    return shared_prefix


def sort_pnkc_classes(unique_model_pnkc_classes: Sequence[str]) -> np.ndarray:
    return np.array(sorted(unique_model_pnkc_classes,
        key=lambda x: EXPECTED_MODEL_PNKC_CLASSES.index(x.split(N_VARIANTS_DELIM)[0])
    ))


def pnkc_classes2source_palette(unique_model_pnkc_classes: Sequence[str]
    ) -> Dict[str, Any]:
    # TODO try to keep blue=uniform, orange=nonclaw, green=claw, red=bouton
    # (even if simplify_models also removes nonclaw)? (meh)
    source_palette = dict(zip(
        sort_pnkc_classes(unique_model_pnkc_classes),
        sns.color_palette(n_colors=len(unique_model_pnkc_classes))
    ))
    return source_palette


def plot_model_params_ordered_by_avg_mixsupp(model_order: pd.DataFrame,
    model_root: Path, mixsupp_order_parquet: Path, *, title_suffix: str = '',
    show_per_panel_mixsupp: bool = False, separate_binary_cbars: bool = True,
    model_dirname_yticks: bool = False) -> None:
    # TODO doc model_order index/cols
    """
    All boolean flags below will add their own suffices to filename, if non-default.

    Args:
        model_order: model parameters ordered by average mixture suppression (also
            contained in `mixsupp_col`)

        mixsupp_order_parquet: order already defined in model_order. This is only used
            to choose plot name (e.g. to share logistic scaling param suffix that might
            be in it) and when `show_per_panel_mixsupp=True`, to process the name of
            this and load similar CSVs with mixture suppression computed on different
            data subsets (or w/ diff scaling/stat).`

        show_per_panel_mixsupp: if False, just show one the one average mixture
            suppression value used to sort models (average across 5comp/binary and
            kiwi/control, computed on logistic_scaled_num_spikes)

            If True, also show separate average mix suppression values for all
            combinations of panel=kiwi/control, mix=binary/5comp,
            stat=num_spikes/logistic_scaled_num_spikes.

        separate_binary_cbars: whether 5comp and binary mixtures will have their own
            colorbars (each using the same palette though), each with a different range.
            Only relevant if `show_per_panel_mixsupp=True`.

        model_dirname_yticks: if True, will include full model output directory name for
            each model in Y-axis ticklabels (can click in plot and copy/paste to look
            into stuff). Otherwise, will just number them (nicer looking, for thesis).
    """
    assert model_order is not None
    mixsupp_order_fname = mixsupp_order_parquet.name

    # TODO try computing megamat corr, both adding noise (done, at least manually
    # checking models sorted by mixsupp) and also logistic scaling (not sure it would
    # matter)
    # TODO factor into a fn for calculating evaluation metrics of model,
    # and share w/ other stuff in step_pn_apl_weights?

    # (to what extent do these all line up? some models that are more consistently
    # better than others?)
    # TODO and w/ frac segmenting cells?

    # TODO refactor to be able to make this type of plot for arbitrary inputs?

    extra_suffix = ''
    if model_dirname_yticks:
        extra_suffix = '_with-dirnames'

    if show_per_panel_mixsupp:
        extra_suffix += '_sep-stats'
        if separate_binary_cbars:
            extra_suffix += '_sep-binary-cbars'
    else:
        assert not separate_binary_cbars, ('separate_binary_cbars=True only supported '
            'if show_per_panel_mixsupp=True'
        )

    # TODO refactor to share this index setting w/ other place that loads order
    # parquets?
    assert model_order.index.names == [None]
    assert all(x in model_order.columns for x in MODEL_COLS), (
        f'not all of {MODEL_COLS=} in {model_order.columns=}'
    )
    model_order = model_order.set_index(MODEL_COLS, verify_integrity=True)
    # TODO delete (assuming we always have connectome_apl)
    #model_cols = [c for c in MODEL_COLS if c in model_order.columns]
    #assert set(MODEL_COLS) - set(model_cols) <= {'connectome_apl'}

    model_ids = model_order.drop(columns=[
        c for c in model_order.columns if c != 'model_dirname'
    ]).squeeze()

    tune_root = model_root2tune_root(model_root)
    # TODO can i get model_ids from model_order? what's minimum i actually need to
    # pass in?
    model_params_list = []
    for model_id, model_dirname in model_ids.items():
        model_dir = tune_root / model_dirname
        assert model_dir.is_dir(), f'{model_dir=} did not exist'
        params = read_params(model_dir)
        # TODO any i don't actually wanna filter here? should be ok
        params = {k: v for k, v in params.items() if k not in exclude_params}
        model_params_list.append(params)

    model_params = pd.DataFrame(model_params_list, index=pd.Index(model_ids))
    # since we will drop several separate columns below, which (across them)
    # contain the same info as this
    model_params[PNKC_CLASS_COL] = model_ids.index.get_level_values(PNKC_CLASS_COL)

    # so we don't need to include both in plot
    assert model_params['one_row_per_claw'].equals(model_params['prat_claws'])

    # should just be 7 (for the single pn2kc_connections='uniform' case)
    assert model_params['n_claws'].dropna().nunique() == 1

    # only =20 for the 6 nonclaw models currently used
    assert model_params['weight_divisor'].dropna().nunique() == 1

    exclude = {'one_row_per_claw', 'fixed_thr', 'wAPLKC', 'n_claws',
        'weight_divisor'
    }

    # TODO delete
    # since these should all be contained in model_pnkc_class
    #if not simplify_models:
    #    included_in_pnkc_class = {'prat_claws', 'prat_boutons', 'pn2kc_connections'}
    #else:
    #    included_in_pnkc_class = {'prat_claws', 'pn2kc_connections'}
    included_in_pnkc_class = {'prat_claws', 'prat_boutons', 'pn2kc_connections'}
    exclude.update({x for x in included_in_pnkc_class if x in model_params.columns})

    # currently just: {'scale_pre_tuning': False}
    same_in_each = subset_same_in_all_dicts(model_params_list)
    exclude.update(same_in_each.keys())

    model_params = model_params.drop(columns=exclude)
    assert not model_params.duplicated().any()
    # tautology but whatever
    assert not model_params.index.duplicated().any()

    # fortunately all of these can currently be filled within all rows, regardless
    # of model_pnkc_class
    defaults_not_in_fitmbmodel_sig = {
        # it's =None in sig (can i change so it's 0.1 explicitly there? why do
        # i have it as-is? just ignore if wAPLKC/fixed_thr/whatever also passed?
        'target_sparsity': 0.1,

        'n_spikes_for_response': 1,
    }
    for k, d in defaults_not_in_fitmbmodel_sig.items():
        # NOTE: this would fail if we ever explicitly specified any of these
        # parameters to their default, when running model. could just comment
        # assertion then.
        assert d not in set(model_params[k].dropna().unique()), ('assumed this '
            'param would only have NaN when this default value used'
        )
        # TODO assert get_fitmbmodel_default is None for this value? i assume it
        # will be for all of them? (is for target_sparsity)
        model_params[k] = model_params[k].fillna(d)

    # since FULL_MODEL_KW_LIST is constructed from all pairwise combinations of
    # parameters (within each model_pnkc_class), there will always be one dict that
    # just contains the parameter of interest, so we can just inspect those to see
    # which parameters are unique to each model_pnkc_class value
    # (all more compilcated models should include all of these from less complicated
    # models. increasing complexity order is:
    # - "all" (non-connectome, e.g. 'uniform)
    # - "nonclaw" (aka connectome wPNKC)
    # - "claw" (one_row_per_claw=True, and almost always also prat_claws=True)
    # - "bouton" (prat_boutons=True)
    def get_paramset(dict_list: List[ParamDict]) -> Set[str]:
        return {list(x.keys())[0] for x in dict_list if len(x) == 1}

    # TODO TODO why aren't there actually a few variants of uniform model then? just
    # an issue with my construction of FULL_MODEL_KW_LIST, or is there currently a
    # big for some of those parameters and uniform (that would be kinda
    # surprising...)
    #
    # TODO worth comparing TRY_ALL_MODELS_WITH to TRY_NONCLAW_MODELS_WITH (only
    # thing in there should be use_connectome_APL_weights, and
    allmodel_params = get_paramset(TRY_ALL_MODELS_WITH)
    # TODO assert these are either all in default_not_in_fitmbmodel_sig above, or
    # use_connectome_APL_weights?
    nonclaw_params = get_paramset(TRY_NONCLAW_MODELS_WITH)

    # TODO equiv?
    have_connectome_apl = has_connectome_apl(model_params,
        apl_col='use_connectome_APL_weights'
    )
    have_connectome_apl2 = has_connectome_apl(model_order)
    assert have_connectome_apl == have_connectome_apl2, (
        f'{have_connectome_apl=} != {have_connectome_apl2}'
    )
    # TODO delete
    #if not simplify_models:
    # TODO TODO check this is equiv to passing simplify models, and checking
    # `not simplify_models` before? or at least, that this works
    if have_connectome_apl:
        model_params.use_connectome_APL_weights = \
            model_params.use_connectome_APL_weights.fillna(False)

    # NOTE: this does not depend on actually using use_connectome_APL_weights
    # anywhere, so we don't need to special case based on simplify_models
    #
    # use_connectome_APL_weights is only NaN for the single uniform model i
    # have. fine to fill that False the same as all >=nonclaw models.
    assert nonclaw_params - allmodel_params == {'use_connectome_APL_weights'}

    assert len(allmodel_params - nonclaw_params) == 0

    claw_params = get_paramset(TRY_CLAW_MODELS_WITH)
    assert len(nonclaw_params - claw_params) == 0
    # would also be in bouton case too (as it's further up the complexity hierarchy)
    clawonly_params = claw_params - nonclaw_params

    # since we try to fill all clawonly_params to default below, and we don't
    # currently support only filling in claw mask when filling with
    # defaults_not_in_fitmmodel_sig (could change latter if needed tho)
    assert len(clawonly_params & set(defaults_not_in_fitmbmodel_sig.keys())) == 0

    bouton_params = get_paramset(TRY_BOUTON_MODELS_WITH)
    assert len(claw_params - bouton_params) == 0
    # NOTE: there should currently not be any parameters unique to bouton case
    # (although there could be, e.g. pre-tuning weight scales on different weights
    # from the four PN<>APL and APL<>KC weights)
    assert bouton_params == claw_params, ('previously had no bouton-specific params'
        '. may want to do some default filling only in bouton model subset now?'
    )
    boutononly_params = bouton_params - claw_params
    assert len(boutononly_params & set(defaults_not_in_fitmbmodel_sig.keys())) == 0

    unique_model_pnkc_classes = sort_pnkc_classes(model_params[PNKC_CLASS_COL].unique())
    uc2 = sort_pnkc_classes(model_order.index.get_level_values(PNKC_CLASS_COL).unique())
    assert np.array_equal(unique_model_pnkc_classes, uc2), \
        f'{unique_model_pnkc_classes=} != {uc2=}'
    del uc2
    source_palette = pnkc_classes2source_palette(unique_model_pnkc_classes)

    # ipdb> unique_model_pnkc_classes
    # array(['uniform (1 variant)', 'nonclaw (6 variants)',
    #        'claw (54 variants)', 'bouton (28 variants)'], dtype=object)
    #
    # getting the class values that contain the ' (<n> variants)' suffixes, just
    # from knowing the prefixes i want.
    claw_and_bouton_classes = set()
    # TODO delete
    #if simplify_models:
    #    classes_to_check = ('claw',)
    #else:
    #    classes_to_check = ('claw', 'bouton')
    classes_to_check = ('claw', 'bouton')
    for prefix in classes_to_check:
        # exact equality when no ' (<n> variants)' suffixes (e.g. when -m, now)
        mask = [x == prefix or x.startswith(f'{prefix} ')
            for x in unique_model_pnkc_classes
        ]
        mask_sum = sum(mask)
        assert mask_sum <= 1, f'{prefix=}: {mask_sum=} > 1'
        if mask_sum == 0:
            assert prefix == 'bouton', ("expected mask_sum=0 only for prefix='bouton', "
                f'not {prefix=}'
            )
            continue

        claw_and_bouton_classes.add(unique_model_pnkc_classes[mask][0])

    claw_and_bouton_mask = model_params[PNKC_CLASS_COL].map(
        lambda x: any(x.startswith(c) for c in claw_and_bouton_classes)
    )
    assert claw_and_bouton_mask.sum() > 0, ('saved model_params should include '
        'some claw/bouton entries'
    )
    assert len(clawonly_params) > 0
    for x in clawonly_params:
        default = get_fitmbmodel_default(x)
        assert not pd.isnull(default), ('expected non-null default, or else should '
            'be handled by hardcode in defaults_not_in_fitmbmodel_sig above'
        )
        model_params.loc[claw_and_bouton_mask, x] = model_params.loc[
            claw_and_bouton_mask, x
        ].fillna(default)

    # TODO delete
    # to get a sense of the type of NaN->default filling i might want to do:
    # ipdb> model_params.loc[model_params[PNKC_CLASS_COL].str.startswith('nonclaw')
    #   ].reset_index(drop=True).drop(columns=PNKC_CLASS_COL).T
    #                                0     1    2     3     4    5
    # pn_claw_to_apl               NaN   NaN  NaN   NaN   NaN  NaN
    # claw_dynamics                NaN   NaN  NaN   NaN   NaN  NaN
    # use_connectome_APL_weights  True   NaN  NaN  True  True  NaN
    # target_sparsity             0.05  0.05  NaN   NaN   NaN  NaN
    # n_spikes_for_response        2.0   2.0  2.0   2.0   NaN  NaN
    # allow_net_inh_per_claw       NaN   NaN  NaN   NaN   NaN  NaN

    name2palette = dict()

    # TODO delete?
    # since source_palette doesn't currently have ' (<n> variants)' suffices here,
    # and want to use the suffices with the counts as they were in saved
    # model_params
    #key_update = dict(zip(
    #    model_params[PNKC_CLASS_COL].str.split().apply(lambda x: x[0]),
    #    model_params[PNKC_CLASS_COL]
    #))
    #new_source_palette = {key_update[k]: v for k, v in source_palette.items()}
    #name2palette[PNKC_CLASS_COL] = new_source_palette
    #
    assert model_params[PNKC_CLASS_COL].str.contains(N_VARIANTS_DELIM, regex=False
        ).all(), 'restore old code? adding #-variants suffices back?'
    name2palette[PNKC_CLASS_COL] = source_palette
    # only two things not already explicitly specified should be
    # n_spikes_for_response and target_sparsity now

    # TODO delete. wasn't easy to modify hong2p fn to treat float values as
    # categorical (for use w/ dict palettes)
    #
    # the 'magma' this was picking by default was making it hard to distinguish the
    # light yellow (one end of magma) from adjacent white values in bool columns
    #name2palette['target_sparsity'] = {0.1: 'red', 0.05: 'blue'}
    #

    # using ListedColormaps for these, to imply less of a continuum since i actually
    # only have two distinct values for each.
    # cbars show w/ discrete regions (one per color in list).
    # https://matplotlib.org/stable/users/explain/colors/colormap-manipulation.html
    def name2listed_cmap(name):
        cm = plt.get_cmap(name)
        # NOTE: currently assumes we only need two colors for each of these.
        # would need to call cmap on however many distinct values input has if i
        # want to extend this to cases where that's not true
        # NOTE: does need to be 0.0/1.0, instead of 0/1, or some/all cmaps will just
        # return the values for 0.0 (for both 0 and 1)
        colors = [cm(0.0), cm(1.0)]
        ret = ListedColormap(colors, name=f'{len(colors)}colors_from_{name}')
        return ret

    name2palette['target_sparsity'] = name2listed_cmap('coolwarm')
    name2palette['n_spikes_for_response'] = name2listed_cmap('viridis')

    ordered_mixsupp = model_order[mixsupp_col]
    assert ordered_mixsupp.is_monotonic_increasing
    # one was defined from the other, and shouldn't be been re-ordered by here, so
    # should be a tautology
    assert np.array_equal(model_ids.values, model_order.model_dirname)
    # without setting this index, assignment to create the new col seems to just
    # create all NaN
    ordered_mixsupp.index = model_params.index.copy()
    model_params[mixsupp_col] = ordered_mixsupp
    assert not model_params[mixsupp_col].isna().any()
    #

    # TODO should map_each_series_to_rgb (and thus new plot_cols_with* fn wrapping
    # it) default to sharing palette for all bool stuff (including bool+NaN stuff)?
    bool_cols = [c for c in model_params.columns
        if set(model_params[c].dropna()) == {True, False}
    ]
    for c in model_params.columns:
        if c in bool_cols:
            continue
        # checking we don't have any columns that are only all True or False
        # (excluding NaNs, which should have already been filled above, in a way
        # that should have eliminated any remaining failures we might see here)
        assert not set(model_params[c].dropna()) <= {True, False}, (
            f'model_params column {c} appeared to be a bool_col w/ only one '
            'unique non-NaN value'
        )

    # NaN is handled internally in the hong2p fn now (w/ color specified by
    # na_color, gray by default)
    bool_palette = {False: 'black', True: 'white'}

    name2palette.update({
        c: bool_palette for c in bool_cols
    })

    fillna_dict = {c: False for c in bool_cols}
    # behavior is more like True when this is not specified (i.e. for non-claw
    # versions of model). this flag was mainly introduced to be able to reproduce
    # the old behavior, while using a model with separate claws.
    fillna_dict['allow_net_inh_per_claw'] = True

    to_corr = model_params.fillna(fillna_dict).select_dtypes(include=[bool, float])
    assert set(model_params.columns) - set(to_corr.columns) == {PNKC_CLASS_COL}
    for_order = to_corr.corr().loc[mixsupp_col].drop(mixsupp_col).abs(
        ).sort_values()
    col_order = list(for_order.index) + [PNKC_CLASS_COL, mixsupp_col]
    model_params = model_params[col_order].copy()

    logistic_palette = 'Reds_r'
    name2palette[mixsupp_col] = logistic_palette

    pnkc_class_legend_loc = 'upper right'
    if not (model_dirname_yticks or show_per_panel_mixsupp):
        # would over lap too much w/ suptitle w/o other tweaking here
        pnkc_class_legend_loc = 'lower left'

    # these values work for 91 total models, with:
    # - 3 colorbars in show_per_panel_mixsupp=False case, and...
    # - 4 colorbars in =True case
    # and a max of 166 characters in model_dirname xticklabel strings
    if model_dirname_yticks:
        # larger >(6, 20) figsize did actually fix:
        # UserWarning: constrained_layout not applied because axes sizes
        # collapsed to zero. Try making figure larger or axes decorations
        # smaller
        # (that i was getting w/ figsize=(6, 20) when model_dirname_yticks=True)
        figsize = (12, 20)
        if show_per_panel_mixsupp:
            cbar_left = 0.91
        else:
            cbar_left = 0.82
        cbar_width = 0.008
        cbar_hmargin = 0.06
    else:
        figsize = (6, 18)
        if show_per_panel_mixsupp:
            cbar_left = 0.75
        else:
            cbar_left = 0.65
        cbar_width = 0.015
        cbar_hmargin = 0.15

    # TODO maybe also weight the binary mixsupp differently when taking mean? is there
    # even as strong an ordering? (at least kiwi panel seems to favor
    # target_sparsity=0.1 over =0.05, in contrast to 5comp behavior)
    if show_per_panel_mixsupp:
        # # of rows / columns / data only determined by whether '_panelmean' in
        # parquet fname suffix, and otherwise it's just the sort order that varies,
        # so don't need to load both of the non-panelmean versions to access all
        # per-panel/mix data (each contains mix supp calcs based on both stats)
        per_panel_mixsupp = read_parquet(
            model_root / mixsupp_order_fname.replace('_panelmean', '')
        )
        group_cols = MODEL_COLS + PANEL_COLS
        per_panel_mixsupp = per_panel_mixsupp.set_index(group_cols,
            verify_integrity=True).rename_axis(columns='stat').unstack(level=PANEL_COLS
            ).loc[model_order.index]

        stat_metadata = per_panel_mixsupp.columns.to_frame(index=False)
        stat_metadata.stat = stat_metadata.stat.replace({
            'mixsupp_logistic_scaled_num_spikes': 'logistic',
            'mixsupp_num_spikes': 'n_spikes',
        })

        # should just be excluding 'diag-binaries_max', which is probably not
        # particularly useful info (doesn't seem to vary much, and don't really care
        # that much if it does. should all be lower than mix suppression in
        # kiwi/control stuff anyway)
        to_drop = ~ stat_metadata.panel.isin(NATMIX_PANELS)
        stat_metadata.mix = stat_metadata.mix.replace({'5comp': '5', 'binary': '2'})

        per_panel_stat_strs = pd.Index(stat_metadata.T.apply(lambda x: '/'.join(x)),
            name='panel_stat'
        )
        per_panel_mixsupp.columns = per_panel_stat_strs
        to_drop.index = per_panel_stat_strs

        per_panel_mixsupp = per_panel_mixsupp.loc[:, ~to_drop]

        assert per_panel_mixsupp.index.map(model_ids).equals(model_params.index)
        per_panel_mixsupp.index = model_params.index

        def stat_sort_key(x):
            parts = x.str.split('/')
            assert (parts.str.len() == 3).all()

            def parts2key(ps):
                scaling, panel, n_comps = ps
                return (scaling, -int(n_comps), panel)

            return parts.map(parts2key)

        per_panel_mixsupp = per_panel_mixsupp.sort_index(key=stat_sort_key,
            axis='columns'
        )

        n_spikes_palette = 'Greys_r'
        for c in per_panel_mixsupp.columns:
            scaling, panel, n_comps = c.split('/')
            n_comps = int(n_comps)
            assert n_comps in (2, 5)
            assert panel in NATMIX_PANELS
            if scaling == 'logistic':
                palette = logistic_palette
            else:
                assert scaling == 'n_spikes'
                palette = n_spikes_palette

            if separate_binary_cbars and n_comps == 2:
                cm = plt.get_cmap(palette).copy()
                cm.name = f'{cm.name}_copy'
                palette = cm

            name2palette[c] = palette

        model_params = pd.concat([model_params, per_panel_mixsupp], axis='columns')

    fig, ax = plot_cols_with_diff_colormaps(model_params, name2palette=name2palette,
        legend_and_cbar_kws=dict(
            legend_locations=['lower right', pnkc_class_legend_loc],
            width=cbar_width, hmargin=cbar_hmargin, left=cbar_left
        ),
        # using ', ' instead of default var_delim='/', since some var names themselves
        # have '/' in them here
        fig_kws=dict(figsize=figsize), var_delim=', ',
    )
    # otherwise yticklabels below will not show
    yticks = np.arange(len(model_params.index))
    ax.set_yticks(yticks)

    if model_dirname_yticks:
        # 6 is small enough. could even go slightly bigger. 7 good currently
        ax.set_yticklabels([str(x) for x in model_ids], fontsize=7.0)
    else:
        ax.set_yticklabels([str(x) for x in yticks])

    ax.set_ylabel('model')

    # TODO (delete) plot (or just include in CSV) when each model was saved? to
    # try to identify any particularly stale outputs? ideally i'd have olfsysm
    # know whether it was compiled with a clean repo (at least to source files),
    # and it would include commit, and i could reference those too, but i have
    # not set that up yet, and probably won't now

    # TODO provide some mechanism for adding titles to legends, in my fn(s)
    # above, to prevent the need for this? (and also for specifying location by
    # name or something? or how else is the order of shared palettes
    # determined?)
    # TODO delete. wasn't easy to modify hong2p fn to treat float values as
    # categorical (for use w/ dict palettes)
    #other_legends = {PNKC_CLASS_COL, 'target_sparsity'}
    other_legends = {PNKC_CLASS_COL,}
    # there should only be two categorical sets of variables (and thus two
    # legends), and the one we want to change the (long ass) title of is the one
    # that is not the model_pnkc_class one
    bool_legends = [x for x in fig.get_children() if isinstance(x, Legend) and
        x.get_title().get_text() not in other_legends
    ]
    assert len(bool_legends) == 1, \
        f'{[x.get_title().get_text() for x in bool_legends]=}'

    # TODO could also check the labels of all the legend_handles contain the
    # keys of bool_palette (+ 'NaN', added by plotting fn)? (this works for now
    # tho)
    bool_legend = bool_legends[0]
    bool_legend.set_title('booleans')

    # only want the double xticks when making internal-use version of plot
    # (aka when model_dirname_yticks=True)
    ax.tick_params(axis='x', bottom=model_dirname_yticks, top=True,
        labelbottom=model_dirname_yticks, labeltop=True
    )
    fig.suptitle('model parameters\nsorted by avg kiwi/control (5comp+binary) '
        f'mix suppression\ncalculated using {order_by_mean_mixsupp_from}{title_suffix}',
        y=1.04
    )
    # TODO add lines separating stats from other things?

    plot_fname = mixsupp_order_fname2shared_prefix(mixsupp_order_fname)
    savefig(fig, model_root, f'{plot_fname}{extra_suffix}', bbox_inches='tight')

    # TODO also make a separate plot w/ params for models that could not run
    # successfully, or were skipped (w/o mix suppression or other things we'd
    # need outputs for, obviously), just to quickly see which model types not
    # successfully being run


# TODO move to hong2p.util?
# TODO how to type hint generator return? Sequence?
def all_bool_combos_of_length_n(n: int, *, false_first: bool = True) -> Sequence:
    # TODO doc, with examples
    bools = [False, True]
    if not false_first:
        bools = bools[::-1]

    return product(*([bools] * n))


# TODO move to hong2p.util? / olf?
def subset_second_to_odors_in_first(df1: pd.DataFrame, df2: pd.DataFrame, *,
    copy: bool = True) -> pd.DataFrame:
    # NOTE: currently assuming all panels have distinct odors, at least for the panels
    # i'll be using for df2 inputs
    # TODO maybe if mix/panel is in df2, we also subset either to mix/panel from df1
    # (which should have only one of each, if present)
    s1 = set(df1.odor.unique())

    df2 = df2[df2.odor.isin(s1)]
    s2 = set(df2.odor.unique())
    # also implies df2 non-empty, obviously
    assert s1 == s2, ('df2 did not have all odors in df1!\ndf1 odors: {s1}\ndf2 odors: '
        f'{s2}'
    )
    if copy:
        df2 = df2.copy()
    return df2


# TODO remove level= eventually? get_odors should handle now
def rename_natmix_odors(df: pd.DataFrame, level='odor') -> pd.DataFrame:
    odors = get_odors(df)
    new_level = odors.name
    # TODO TODO delete level= kwarg (and then remove separate call to this on orn
    # data, that passes level='odor1'?), if this keeps passing
    assert level == new_level, 'must continue passing level= kwarg'

    # TODO delete
    #new_level = olf.first_odor_level(df.index)
    #
    rename_dict = {
        'cmix @ 0': 'cmix',
        'kmix @ 0': 'kmix',
        'ea+eb @ 0': 'ea+eb',
        '1o3ol+2h @ 0': '1o3ol+2h',
    }
    # TODO delete. everything besides maybe orn_df gonna be missing at least half of
    # these
    #odor_set = set(odors.unique())
    #rename_keys = set(rename_dict.keys())
    ## TODO fail on any KC / ORN input? i guess no one KC dataset will have either? warn
    ## about this instead? delete?
    #assert rename_keys - odor_set == set(), ('df was missing the following odors from '
    #    f'rename_dict.keys():\n{rename_keys - odor_set}\ndf odors: {odor_set}'
    #)

    # TODO delete
    # TODO TODO need values here to depend on leave_concs_in_odors flag?
    # maybe for the binary?
    print('need rename_natmix_odors behavior to depend on leave_concs_in_odors?')
    #
    return df.rename(index=rename_dict, level=new_level)


# TODO should i not also be calling this for on `natmix_df`? seems like it's been
# working ok so far (and none of that stuff currently processed w/ -L anyway, right?)
def preprocess_natmix_df(df: pd.DataFrame) -> pd.DataFrame:
    """Move 'panel' level from row to column index, but does not change data.
    """
    # TODO delete
    #print()
    #print('before preprocess_natmix_df, df:')
    #print(df)
    #
    df = df.rename_axis(index={'odor1': 'odor'}).rename_axis(
        columns={'cells': 'roi'}
    )
    df = rename_natmix_odors(df)
    dfs = []
    for panel in df.index.get_level_values('panel').unique():
        pdf = addlevel(df.loc[panel].dropna(how='all', axis='columns'), 'panel',
            panel, axis='columns'
        )
        dfs.append(pdf)

    ret = pd.concat(dfs, verify_integrity=True)
    # TODO check whether input data has any NaN, instead of this flag? this work?
    # (yes)
    check_data_eq = df.isna().any().any()
    if check_data_eq:
        # so data .values doesn't change at all, it's just whether 'panel' is in
        # index or columns
        assert np.array_equal(ret, df, equal_nan=True)
    else:
        # for ORN input, it does change, but just b/c it adds NaN (since the ORN
        # data does have both panels for all flies, unlike the KC data)
        assert ret.notna().sum().sum() == df.notna().sum().sum()

    assert set(df.index.names) == {'panel', 'odor'}
    assert set(ret.index.names) == {'odor'}

    old_cols = set(df.columns.names)
    assert old_cols == set(flyroi_cols)
    assert set(ret.columns.names) == (old_cols | {'panel'})

    # TODO delete
    #print()
    #print('after preprocess_natmix_df, df:')
    #print(df)
    #breakpoint()
    #
    return ret


def panel2kc_panel(panel: str) -> str:
    """Just to normalize away multiple suffixes model data had for 'diag-binaries' panel

    The only one analyzed now is 'diag-binaries_max', but had 'diag-binaries_mean' and
    'diag-binaries_max-rest0d' before.
    """
    if panel.startswith('diag-binaries_'):
        return 'diag-binaries'
    return panel


# TODO use an IndexOrSeries type here? add to hong2p.types?
def is_natmix_full_mix(values: Union[pd.Index, pd.Series], *, check_one: bool = False
    ) -> Union[pd.Index, pd.Series]:
    # NOTE: this would also return True for mix dilution, as-is, if those were still in
    # input data.
    # TODO assert only 1? or no other substrs indicating it is a dilution of the full
    # mix?
    mask = values.str.contains('mix')
    if check_one:
        assert mask.sum() == 1, f'{mask.sum()=} != 1'
    return mask


def split_mixes(df: pd.DataFrame, responded: Optional[pd.DataFrame] = None, *,
    drop_nonresponders_per_mix: bool = False, panel: Optional[str] = None) -> Tuple[
        List[pd.DataFrame], Optional[pd.DataFrame]
    ]:
    """Returns binary mix dataframes and (if panel in kiwi/control) full mix dataframe.
    """
    # so there should be nothing besides a single 'odor' / 'odor1' level here.
    # no 'panel'/'mix'/anything
    check_unique_odor_only(df.index)

    columns = df.columns
    names = columns.names
    assert 'mix' not in names, f"columns {names=} already had 'mix' level"

    df_panel = None
    if 'panel' in names:
        df_panel = get_single_unique(columns.get_level_values('panel'))
        df_panel = panel2kc_panel(df_panel)

    elif panel is None:
        raise ValueError("'panel' level must either be in column names, or "
            'panel=<str> must be passed in'
        )

    if panel is not None and 'panel' in names:
        assert df_panel == panel, (
            f'panel in df ({df_panel}) did not match {panel=} passed in'
        )

    if panel is None:
        # this assertion shouldn't be hit
        assert df_panel is not None
        panel = df_panel

    supported_panels = ('diag-binaries',) + NATMIX_PANELS
    assert panel in supported_panels, f'{panel=} not in {supported_panels=}. add?'

    if drop_nonresponders_per_mix:
        assert responded is not None, ('must mass responded=<bool-dataframe>, with '
            f'same indices as df, if {drop_nonresponders_per_mix=}'
        )

    index = df.index
    if responded is not None:
        # the columns will NOT be equal when there are >1 stats in df
        # (in that case df will have twice as many columns)
        if 'stat' in df.columns.names:
            assert df.columns.droplevel('stat').drop_duplicates().equals(
                responded.columns
            )

        assert responded.index.equals(index), \
            'may not be able to use masks defined from one on the other'

    binary_mix_mask = index.str.contains('\+')
    assert binary_mix_mask.sum() > 0, ('expected some binary mixes for these '
        'panels'
    )
    binary_mix_list = []
    for binary_mix in index[binary_mix_mask]:
        ca, cb = [x.strip() for x in binary_mix.split('+')]
        assert all((index == x).sum() == 1 for x in (ca,cb)), (
            f'{index=}\n{ca=} {cb=}'
        )

        binary_and_comp_mask = index.isin((ca, cb, binary_mix))
        assert binary_and_comp_mask.sum() == 3, f'{binary_and_comp_mask.sum()=}'
        for_binary = df[binary_and_comp_mask]

        # TODO easy to do this after, and leave all cells for now? any analyses
        # downstream of this actually need them preserved? (if it's just mixsupp and
        # (per responding (KC, odor)) response strength dists, then no)
        # (should be fine to just keep doing in here w/ the flag)
        if drop_nonresponders_per_mix:
            responder_mask = responded[binary_and_comp_mask].any()
            # even tho for_binary sometimes has multiple stats (in final MultiIndex
            # level), indexing on responder_mask still seems to work, producing same set
            # of ['source','model_pnkc_class','roi'] unique values in first test case I
            # checked:
            # > c1 = for_binary.loc[:, responder_mask].columns.drop_duplicates()
            # > c2 = responded.columns[responder_mask]
            # > c1.droplevel('stat').drop_duplicates().equals(c2)
            # True
            for_binary = for_binary.loc[:, responder_mask]

            # TODO delete? put behind checks=True flag?
            # TODO duplicate for 5comp stuff below?
            c1 = for_binary.loc[:, responder_mask].columns.drop_duplicates()
            if 'stat' in c1.names:
                c1 = c1.droplevel('stat').drop_duplicates()
            c2 = responded.columns[responder_mask]
            assert c1.equals(c2)
            #

        if panel == 'diag-binaries':
            for_binary = addlevel(for_binary, 'mix', binary_mix, axis='columns')

        elif panel in NATMIX_PANELS:
            # TODO also use mix name here, for consistency? (would also have to
            # change KC handling) (meh)
            for_binary = addlevel(for_binary, 'mix', 'binary', axis='columns')

        binary_mix_list.append(for_binary)
        del binary_mix

    if panel in NATMIX_PANELS:
        assert len(binary_mix_list) == 1
    else:
        assert len(binary_mix_list) > 1

    full_mix_mask = is_natmix_full_mix(df.index, check_one=False)
    if panel == 'diag-binaries':
        assert full_mix_mask.sum() == 0
        full_mix_df = None

    elif panel in NATMIX_PANELS:
        assert full_mix_mask.sum() == 1, f'{full_mix_mask.sum()=}'
        # TODO delete here, after defining below from full_mix_df/whatever
        #full_mix = df.index[full_mix_mask][0]
        #
        full_mix_and_comp_mask = ~binary_mix_mask
        assert full_mix_and_comp_mask.sum() == 6
        full_mix_df = df[full_mix_and_comp_mask]

        if drop_nonresponders_per_mix:
            responder_mask = responded[full_mix_and_comp_mask].any()
            full_mix_df = full_mix_df.loc[:, responder_mask]
            # TODO delete? put behind checks=True flag?
            c1 = full_mix_df.loc[:, responder_mask].columns.drop_duplicates()
            if 'stat' in c1.names:
                c1 = c1.droplevel('stat').drop_duplicates()
            c2 = responded.columns[responder_mask]
            assert c1.equals(c2)
            #

        # TODO use name of mix instead? (meh)
        full_mix_df = addlevel(full_mix_df, 'mix', '5comp', axis='columns')

    # TODO should i also addlevel(x, 'panel', panel) for all output, if it
    # didn't already have it? (original usage of this code didn't seem to want that idk)

    return binary_mix_list, full_mix_df


def plot_kc_orn_eag_intensity_comparison(plot_dir: Path, palette: Dict[str, Color],
    kc_df: pd.DataFrame, orn_df: pd.DataFrame, eag_df: Optional[pd.DataFrame] = None
    ) -> None:
    # EAG data doesn't have binary mixture, but eag_df should be None when kc_df mix is
    # 'binary'
    # TODO TODO unnnormalized versions of these, where data are in different facets
    # w/ diff ylabels and sharey=False
    # TODO delete
    #print('UNNORMALIZED VERSION OF INTENSITY PLOT (or rather, add + change yscales)')
    # TODO TODO or actually, just overwrite existing left y-axis and add new right
    # y-axis, each with one absolute scale for the kc+orn only version of plot?
    # then just need one plot... (tried some code doing this below w/ secondary_yaxis,
    # but having trouble getting it to work right)
    # TODO tho may still want to compare to a raw plot with comprable limit set, just
    # for sanity checking
    #
    panel, mix = get_single_unique_panel_and_mix(kc_df)

    # TODO rename df1=kc_df, df2=orn_df, df3=eag_df eventually

    df1 = kc_df[~kc_df.stat.str.startswith(NORM_PREFIX)].copy()
    assert df1.stat.nunique() == 1, f'{df1.stat.unique()=}'
    df1 = add_group_id(df1, fly_cols, 'fly_id')

    df2 = orn_df.reset_index()
    df2 = add_group_id(df2, fly_cols, 'fly_id')
    df2 = df2.drop(columns=fly_cols)

    df1 = df1.drop(columns=list(set(df1.columns) - set(df2.columns)))
    assert set(df1.columns) == set(df2.columns)

    fname_parts = ['kc', 'orn']
    to_subset = [df2]
    if eag_df is not None:
        df3 = eag_df.reset_index()
        assert set(df2.columns) == set(df3.columns)
        to_subset.append(df3)
        # would want to do this after checking we could subset odors to same, if we
        # don't only pass eag_df when it has the necessary odors (i.e. mix != 'binary')
        # (but currently i am only passing it in the mix == '5comp' case, so this is
        # fine)
        fname_parts.append('eag')

    # no longer need to drop is_normalized above
    assert all('is_normalized' not in x.columns for x in to_subset)

    subset_dfs = []
    for df in to_subset:
        # this will assert df has all odors in df1
        # TODO assert only one panel/mix after this, if present? do that inside this
        # subset fn?
        df = subset_second_to_odors_in_first(df1, df)
        subset_dfs.append(df)

    dfs = [df1] + subset_dfs

    # need to calculate this on raw data, before normalizing, so i can get scales for
    # secondary y-axes
    max_flymeans = [calc_max_flymean(x) for x in dfs]

    # TODO delete after debugging below
    odfs = [x.copy() for x in dfs]
    #

    # TODO assert dfs all have same set of odors (they should now anyway)
    dfs = [normalize_one_panel(x) for x in dfs]

    dfs_with_n = []
    new_palette = dict()
    for curr_df in dfs:
        assert not curr_df.fly_id.isna().any()
        n_flies = curr_df.fly_id.nunique()
        n_suffix = f' (n={n_flies})'
        source = curr_df.source.unique()[0]
        with_n = source + n_suffix
        new_palette[with_n] = palette[source]

        curr_df['source'] = curr_df.source + n_suffix
        dfs_with_n.append(curr_df)

    intensity_comp_df = pd.concat(dfs_with_n, ignore_index=True)
    intensity_comp_df = sort_odors(intensity_comp_df)

    fig, ax = plt.subplots()
    shared_kws = dict(ax=ax, x='odor', y='value', hue='source', palette=new_palette)
    # TODO why is there stil a legend inside the axes?
    # i think we do need this to use a fig.legend, so would have to hide the pointplot
    # one, if i still wanted one outside axes
    sns.pointplot(intensity_comp_df, dodge=True, legend=True,
        **kc_pointplot_kws, **shared_kws
    )
    stripplot(intensity_comp_df, **shared_kws, **perfly_stripplot_kws)
    ax.set_ylabel('normalized intensity (max mean -> 1)'
        f'\n(with {CI:.0f}% CI on mean)'
    )

    # this happening before secondary_yax def causing problems? (doesnt seem to matter)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim([0.0, ymax])

    # TODO fix/delete
    '''
    # ipdb> odfs[1].value.max()
    # 1.027183224543563
    # ipdb> odfs[0].value.max()
    # 0.7569451967522122
    # ipdb> odfs[0].groupby('odor').value.mean().max()
    # 0.5911110921263032
    # ipdb> max_flymeans
    # [0.5911110921263032, 0.8402593444470375]
    # 1.027183224543563
    # ipdb> odfs[0].value.max()
    # TODO delete
    print()
    print(f'{odfs[0].value.max()=}')
    print(f'{odfs[1].value.max()=}')
    print(f'{max_flymeans=}')
    odf = odfs[1]
    print()
    #
    # TODO delete restore?
    #locations = ['left', 'right']
    loc_dicts = [dict(location='right'), dict(location=1.2, transform=ax.transAxes)]
    # also doesn't seem to matter if i define two vs 1 (wrong either way)
    #for df, max_flymean, loc in zip(dfs[1:], max_flymeans[1:], loc_dicts[:1]):
    for df, max_flymean, loc in zip(dfs, max_flymeans, locations):
        source = get_single_unique(df.source)

        stat = get_single_unique(df.stat)
        if stat.startswith(NORM_PREFIX):
            stat = stat[len(NORM_PREFIX):]

        # removing N suffix we now have in source
        label = f'{source.split()[0]} {stat}'
        color = new_palette[source]
        # TODO delete
        print(f'{source=}')
        print(f'{max_flymean=}')
        print()
        #
        # TODO TODO offset the loc = 'left' one with some transform, like example
        # (secondary_yaxis(3, transform=ax.transData), w/ 3 in data coords)
        # x * max_flymean should be correct to convert from normed to unnormed.
        # TODO TODO is this correct or do i have them backwards?
        yax = ax.secondary_yaxis(**loc,
            # TODO what is appropriate inverse? why do docs have an example passing the
            # same fn for both?
            # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.secondary_yaxis.html
            # TODO TODO did i have it backwards?
            # TODO TODO TODO why is this not working?
            functions=(lambda x: x * max_flymean, lambda x: x / max_flymean),
            # no, this is even worse
            #functions=(lambda x: x / max_flymean, lambda x: x * max_flymean),
            # seems similar to above
            #functions=(lambda x: x / max_flymean, lambda x: x / max_flymean),
        )
        # none of this code was really changing anything either
        # TODO refactor to share w/ twinx mb_model APL Vm plotting /etc?
        yax.set_ylabel(label, color=color)
        # TODO delete. did not accept color
        #yax.set_label(label, color=color)
        # TODO need to specify axis='y' or something? (probably, as set_label above
        # needed to be replaced w/ set_Ylabel, in order to accept color)
        yax.tick_params(axis='y', color=color)
        for text in yax.yaxis.get_ticklabels():
            # alpha also not supported here
            text.set_color(color)
        # TODO maybe hide original KC normalized ylabel?
    '''

    ax.set_title(f'intensity comparison\n{panel}/{mix}')

    # TODO like this loc? legend populated properly?
    ##fig.legend(loc='outside right upper')
    #fig.legend(loc='outside left upper')

    savefig(fig, plot_dir, f'intensity_{panel}_{mix}_{"-".join(fname_parts)}')
    # TODO delete
    #breakpoint()
    #


# TODO factor to hong2p.olf?
def check_unique_odor_only(index: pd.Index) -> None:
    """Checks index has only a single odor str level and odors unique, raising if not

    This is typically to check data comes from only one 'panel' [+'mix'], and can be
    passed to analyses that expect that (e.g. response class / mix suppression stuff).

    Also expects that the name of the single level will have `olf.is_odor_var(<name>)`
    return True for it (i.e. so this level could also be returned via
    `olf.first_odor_level`, if part of a MultiIndex somewhere else).
    """
    # TODO allow panel/mix if only contain a single unique value each?
    # prob not, as many fns this is checking for expect to be able to reference the
    # odors just via the index itself, not a level from it
    names = index.names
    assert len(names) == 1, (f'index {names=} was len > 1, so not just a single odor '
        'level'
    )
    name = names[0]
    # True for things like 'odor', 'odor1'
    assert olf.is_odor_var(name), f'olf.is_odor_var(name) was False for index {name=}'

    assert index.notna().all(), f'expected no NaN odors! {index=}'
    assert not index.duplicated().any(), f'duplicate odors in index! {index=}'


# NOTE: would have to use logic from odor_sort_key fn elsewhere in this script,
# if wanted to use for data where sort_odors won't be sufficient
def check_odor_index_sorted_within_panel(data: DataFrameOrSeries) -> None:
    index = data.index
    odor_level = olf.first_odor_level(index)
    assert index.names == ['panel', odor_level], (f'{index.names=} were not '
        "['panel', <some-odor-name, e.g. 'odor' | 'odor1'>]"
    )
    sorted_index = data.groupby(level='panel', sort=False, group_keys=False).apply(
        sort_odors).index
    assert sorted_index.equals(index), ('odors were not already sorted by panel! would '
        'have to apply the kind of sorting this function does (assuming appropriate for'
        ' this input data)'
    )


dilution_factor_delim: str = ' / '
# TODO also sort components in fixed order? matter (prob not?)?
def odor_sort_fn(x):
    panels_to_check = set(NATMIX_PANELS)
    # TODO matter? set global if so (or pass)
    #if skip_panels is not None:
    #    panels_to_check -= set(skip_panels)
    panels_to_check = sorted(panels_to_check)

    odors = x.unique()
    # NOTE: this approach will not work for input with multiple panels...
    # (as this fn is only ever called on odor level as a key, at least now)
    panel = None
    for curr_panel in panels_to_check:
        curr_name_order = panel2name_order[curr_panel]
        # TODO TODO still need to handle these for kiwi/control, or never gonna
        # analyse those w/ -L/--leave-concs?
        # TODO TODO just try parse_odor_name? anything it wouldn't work on?
        # TODO TODO actually, wasn't i just not calling process_odor_str_for_model
        # when i should have been?
        assert not any('@' in x for x in odors), ('would need parse_odor_name in '
            'panel loop below if we had concs sometimes'
        )
        if all(o in curr_name_order for o in odors):
            assert panel is None, (f'multiple panels in {NATMIX_PANELS=} matched '
                f'{odors=}'
            )
            panel = curr_panel

    if panel is not None:
        name_order = panel2name_order[panel]
        # TODO or need negative of index?
        key = x.map(name_order.index)
        return key

    # TODO TODO are components sorted alphabetically be default? sorted at all?
    # check!

    # TODO why are we also including '/'? are all mixtures not using '+'? provide
    # example of what uses '/' at least (or switch all to using '+'?)
    # TODO TODO must have been for the synthetic diags. test with this again before
    # deleting
    # TODO delete if i can, now that i'm using ' / <x>' to  indicate pair dilutions
    #v1 = 1 * (x.str.contains('+', regex=False)) | (x.str.contains('/', regex=False))
    v1 = 1 * x.str.contains('+', regex=False)

    # to put the cmix/kmix at end
    v2 = 2 * x.str.contains('mix', regex=False)

    key = v1 + v2

    # TODO TODO keep this code? try to change handling to using a diff index level,
    # and my add_group_labels_and_lines fn, and delete these (and delete code adding
    # this to odor strs)?
    if x.str.contains(dilution_factor_delim, regex=False).any():
        # TODO want to reverse order of this? just flip sign? flag for that?
        dilution_factor = x.str.split(dilution_factor_delim, regex=False).map(
            lambda x: 0.0 if len(x) == 1 else float(x[1])
        )
        # so lower concentrations (higher dilution factors) are placed first in
        # order
        dilution_factor = -1 * dilution_factor
        # TODO flag to swap dilution_factor and key here?
        key = pd.Series(index=x.index, data=zip(dilution_factor, key))
    #

    return key


KC_RESP_COL: str = 'mean_Fc_zscore'
# TODO refactor? how?
ORN_RESP_COL: str = 'mean_peak_dff'

def mix_supp_list2flystat_df(mix_supp_dfs: List[pd.DataFrame], *,
    stat: str = KC_RESP_COL) -> pd.DataFrame:
    # TODO doc

    # TODO assert stat is in something already? like a column name? and do i want to
    # remove that column name? (or is it not?)

    mix_supp = pd.concat(mix_supp_dfs, verify_integrity=True).reset_index()
    diff_col = get_diff_col(mix_supp)

    assert 'stat' not in mix_supp.columns
    # TODO problem that it isn't previxed w/ MEAN_PREFIX
    # (will probably have to change handling of model ones anyway, for same reason,
    # if theres an issue)
    # TODO and we are just assuming this is already computed mixsupp from the
    # named stat?
    mix_supp['stat'] = stat

    assert 'value' not in mix_supp.columns
    return mix_supp


def tidy2responses_and_response_mask(mix_df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame
    ]:
    """Returns wide dataframes (responses, responded) with same shape and indices.

    Args:
        mix_df: tidy (long form) dataframe with at least columns (no NaN in any):
            - `al_util.flyroi_cols` (`['date', 'fly_num', 'roi']`)
               (will be columns levels of output)

            - 'value','responded': will form values of `responses` and `responded`,
              respectively. should respectively be of types `float|int` and `bool`.

            - 'odor': will be single-level row index of output

            - 'panel','mix': currently just used to check that input only has a single
              unique value for each (and will also currently fail if missing or NaN),
              but not included in output (maybe they should be, alongside 'odor' in row
              index.  some code currently assumes single-level str index downstream of
              this tho...)
    """
    # this is just to check input only has one unique (panel, mix)
    get_single_unique_panel_and_mix(mix_df)

    # TODO should i also actually use the (panel, mix) (see note in docstring tho)

    # dtype of this is asserted bool in response_class_means_and_perfly_counts
    responded = mix_df.pivot(columns=flyroi_cols, index='odor', values='responded')
    responded = responded.sort_index(kind='stable', key=odor_sort_fn)

    is_responder = responded.any()
    assert _have_fly_cols(is_responder)
    assert not is_responder.all(), 'expected to still have non-responders here'
    del is_responder

    # TODO refactor to share pivot -> sorting w/ `responded` above
    responses = mix_df.pivot(columns=flyroi_cols, index='odor', values='value')
    responses = responses.sort_index(kind='stable', key=odor_sort_fn)
    assert not responses.isna().any().any()

    return responses, responded


def response_class_means_and_perfly_counts(responded: pd.DataFrame,
    responses: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Returns response class means and counts.

    Returns class means (all ROIs weighted equally) and per-(class,fly) counts of each.

    NOTE: this function returns (means, counts), which is opposite order (currently) of
    `summarize_response_classes`.
    """
    # TODO assert _have_fly_cols(input)? only really takes a single
    # summarize_response_classes call for model without need to track counts per fly, so
    # no need to call this on model input
    assert _have_fly_cols(responded)

    check_unique_odor_only(responded.index)
    assert pd_indices_equal(responded, responses)

    dtype = get_single_unique(responded.dtypes)
    assert np.issubdtype(dtype, bool)
    del dtype

    # TODO take verbose= kwarg to this fn, and use to control verbose= for this first
    # call?
    class_sizes_by_fly, perfly_class_means = summarize_response_classes(
        responded, responses=responses
    )
    # not using classes sizes here, because is not split apart by fly (with
    # fly_cols also in index) like we want. all ROIs should be weighted equally in class
    # means here, instead of calculating mean per fly and then weighting flies equally.
    _, class_means = summarize_response_classes(responded, responses=responses,
        verbose=False, sum_across_flies=True
    )
    assert class_sizes_by_fly.sum() == len(responded.columns)

    # class_means should be weighting all ROIs equally, ignoring fly
    class_means2 = perfly_class_means.groupby(level=class_means.columns.names,
        axis='columns', sort=False
    ).mean()
    assert pd_indices_equal(class_means, class_means2)
    assert all(x.notna().all().all() for x in [class_means, class_means2])
    assert not pd_allclose(class_means, class_means2)
    # so it's not just a relatively meaningless (almost numerical) issue
    assert (
        class_means2 - class_means
    ).abs().mean().mean() > 0.01, \
        'lower threshold if needed. calculations should be different tho'
    # this is out of (for KC data, not sure which panel):
    # ipdb> class_means.max().max()
    # 3.0088203744445132
    # ipdb> (class_means2 - class_means).abs().mean().mean()
    # 0.07600456432801901
    # seen 0.017494055096707442 in kiwi/binary (prob lowest?)

    return class_means, class_sizes_by_fly


# NOTE: currently just seems to be used in add_fixed_legend called from plot_stats
# (at least, outside of current main)
kc_color: Color = 'm'

source_col: str = 'source'
def add_fixed_legend(g: sns.axisgrid.FacetGrid, df: pd.DataFrame, palette: Palette, *,
    lines: bool = True, add_mix_vs_comp_linestyle: bool = False) -> None:
    # TODO doc lines in particular. say more?
    # TODO TODO part about replacing label doesn't seem always true. still says 'KCs'
    # for kc-only mixsupp dist plot i'm looking at. is that just for some other cases?
    # even still used?
    """
    Args:
        lines: if True, uses `line_kws` instead of `marker_kws` for APL artists added.
            If False, also replaces 'KCs' label with 'KCs (<ci>% CI on mean)'
    """
    # TODO TODO what to do if this doesn't already have the model values, but
    # have_model=True (like seems to be the case in response-strength_* plots now.
    # still? delete?)?
    legend_data = dict(g._legend_data)

    # how to get multiple "titles" with different section in legend:
    # https://stackoverflow.com/questions/24787041
    # other (more complicated answer) might even allow nice left alignment, but
    # going with simpler solution using title_proxy Rectangle for now.
    title_proxy = Rectangle((0,0), 0, 0, color='w')
    # TODO (delete. i think issue was just that there *are* no uniform-APL 'boutons'
    # cases) try to replace all existing (PN>KC) (so != 'KCs') legend entries w/
    # the circle artists of the same hue, so it doesn't matter which we plot first
    # (currently only the bouton case is getting + as marker. it does have some
    # non-connectome-APL cases right? or no?)
    # TODO add a suffix to 'boutons' clarifying it only has connectome-APL variant,
    # if leaving marker as-is there?

    line_kws = dict(linestyle='-', marker='none')
    # TODO add line_kws in here? or still want the ability to override (prob on
    # linestyle at least... marker='none' always? just overwrite line_kws w/ kws?
    # TODO also default to color='k', alpha=0.5?
    def line_artist(**kws):
        return Line2D([0], [0], **kws)

    have_model = has_model(df)
    if not have_model:
        label_order = []
    else:
        all_pnkc_classes = df[PNKC_CLASS_COL].unique()
        # TODO sort order or nah?
        model_only_classes = [x for x in all_pnkc_classes if pnkc_class_is_model(x)]
        label_order = model_only_classes
        del all_pnkc_classes, model_only_classes

        model_alpha_for_legend: float = 0.7

        for k in label_order:
            color = palette[k]
            assert type(color) is tuple and len(color) == 3, (f'{color=} was '
                f'not a RGB tuple for {k=} (in palette)'
            )
            artist = None
            curr_line_kws = dict(line_kws)
            if k in legend_data:
                existing = legend_data[k]
                rgb, _ = artist_rgb_and_alpha(existing)

                # TODO maybe need to check w/ np.allclose? (doesn't seem so. delete
                # comment)
                assert rgb == color, f'existing artist {rgb=} != desired {color=}'

                # TODO anything else? (so far, no)
                if isinstance(existing, (BarContainer, PolyCollection)):
                    # TODO (delete?) ever want line artist for PolyCollection? check
                    # get_facecolor() is not white / transparent or something?
                    # (and use line if so)
                    # TODO actually need label=k?
                    artist = Patch(facecolor=color, alpha=model_alpha_for_legend,
                        label=k
                    )
                else:
                    assert isinstance(existing, Line2D), (f'{type(existing)=} '
                        # TODO say what plot calls BarContainer and PolyCollection
                        # are from? i'm assuming former is from some bar[h] calls,
                        # but what about PolyCollection?
                        'currently unsupported. only Line2D/BarContainer/'
                        'PolyCollection currently are'
                    )
                    # TODO (delete? seems ok so far) linewidth too? anything else?
                    vars_to_copy = ['linestyle', 'marker', 'markeredgecolor',
                        'markeredgewidth', 'markerfacecolor', 'markersize'
                    ]
                    for x in vars_to_copy:
                        val = artist_var(existing, x)
                        curr_line_kws[x] = val

            if artist is None:
                artist = line_artist(alpha=model_alpha_for_legend, color=color,
                    **curr_line_kws
                )

            legend_data[k] = artist

        pnkc_title = 'model PN>KC connectivity:'
        legend_data[pnkc_title] = title_proxy
        label_order = [pnkc_title] + label_order

    # TODO is this only an issue because mean_response_rate doesn't have a
    # separate twin ax to plot real KC data on? or would it be an issue
    # regardless? either way, this is probably the easiest fix.
    if 'exp_type' in df.columns:
        # TODO assert not have_model? delete this conditional anyway?
        unique_exp_types = df.exp_type.dropna().unique()
        legend_data = {
            k: v for k, v in legend_data.items() if k not in unique_exp_types
        }
        # TODO TODO also redefine label order (to match set of keys)? actually test this
        # case again?

    # TODO only do this if i need to (i.e. if something is coming next?)?
    # (or is something always coming next?)
    empty_line = ''
    legend_data[empty_line] = title_proxy

    if len(label_order) > 0:
        label_order.append(empty_line)

    replace = {
        'bouton': 'bouton (no uniform APL)',
    }

    # since they are handled w/ a separate plot call that just plots KCs, and we
    # set legend=False on that
    # this even being hit? (yes)
    # (and commenting it did get rid of erroneous KCs line in model-only legends)
    # TODO why didn't it cause assertion below (about label_order matching
    # legend_data.keys()) to fail, when i commented it
    unique_sources = df[source_col].unique()
    put_at_top = []
    for x in ('KCs', 'ORNs'):
        if x in unique_sources and x not in legend_data:
            # TODO delete
            # TODO TODO does df[source_col] really have 'KCs' when i don't want it
            # in legend (seems so) (when? can i fix outside?)
            print(f'adding {x} line to legend')
            #
            # TODO pass in kc_alpha (but use this as default)?
            legend_data[x] = line_artist(color=kc_color, alpha=kc_err_alpha, **line_kws)
            # TODO also do one for KC points? (if any. add flag?)

        # TODO if we didn't add it to legend_data above, would these always be
        # there if in unique_sources? and vice versa?
        if x in legend_data:
            put_at_top.append(x)

            if not lines:
                replace.update({
                    x: f'{x} ({CI:.0f}% CI on mean)',
                })

    if len(put_at_top) > 0:
        label_order = put_at_top + [empty_line] + label_order

    legend_data = {
        (replace[k] if k in replace else k): v for k, v in legend_data.items()
    }
    label_order = [replace[k] if k in replace else k for k in label_order]

    have_connectome_apl = has_connectome_apl(df)
    assert not have_connectome_apl, 'must drop the connectome_apl data'

    odor_artist_kws = dict(color='k', alpha=0.5)
    odor_artist_kws.update(line_kws)

    mix_kws = dict(linestyle=mix_linestyle)
    component_kws = dict(linestyle=component_linestyle)

    if add_mix_vs_comp_linestyle:
        odor_title = 'odor (within each hue):'
        legend_data[odor_title] = title_proxy
        label_order.append(odor_title)

        mix_odor = 'mix'
        assert mix_odor not in legend_data, \
            'would overwrite previous legend entry'

        legend_data[mix_odor] = line_artist(
            **{**odor_artist_kws, **mix_kws}
        )
        component_odor = 'component'
        legend_data[component_odor] = line_artist(
            **{**odor_artist_kws, **component_kws}
        )
        label_order += [mix_odor, component_odor]

    if have_connectome_apl:
        apl_artist_kws = dict(color='k', alpha=0.5)
        if lines:
            apl_artist_kws.update(line_kws)
        else:
            apl_artist_kws.update(marker_kws)

        # assumed the markers we want are either lines (solid vs dashed) or point
        # circles/X's markers
        if lines:
            uniform_apl_kws = dict(linestyle=uniform_apl_linestyle)
            connectome_apl_kws = dict(linestyle=connectome_apl_linestyle)
        else:
            uniform_apl_kws = dict(marker=uniform_apl_marker, markeredgewidth=0.0)
            connectome_apl_kws = dict(
                marker=connectome_apl_marker, markeredgewidth=1.0
            )

        apl_title = 'model APL (within each hue):'
        legend_data[apl_title] = title_proxy
        label_order.append(apl_title)

        # TODO (delete) automatically switch between *_apl_marker and
        # *_apl_linestyle, based on type of other legend artists?
        #
        # adding an extra space at end here so that it doesn't conflict with
        # 'uniform' PN>KC entry from earlier
        uniform_apl = 'uniform '
        assert uniform_apl not in legend_data, \
            'would overwrite previous PN>KC uniform'

        legend_data[uniform_apl] = line_artist(
            **{**apl_artist_kws, **uniform_apl_kws}
        )
        connectome_apl = 'connectome'
        legend_data[connectome_apl] = line_artist(
            **{**apl_artist_kws, **connectome_apl_kws}
        )
        label_order += [uniform_apl, connectome_apl]

    assert set(label_order) == set(legend_data.keys()), \
        f'{label_order=} {legend_data.keys()=}'

    g.add_legend(legend_data=legend_data, label_order=label_order,
        # TODO just always have it be source_col?
        #title='model variant' if source_col == 'model' else 'source'
        title=''
    )


def plot_one_dist_per_fly(data: pd.DataFrame, *, flies_share_bins: bool = True,
    **kwargs) -> None:

    if flies_share_bins and _USE_KDEPLOT:
        warn('plot_one_dist_per_fly: can not use flies_share_bins=True'
            f' when {_USE_KDEPLOT=}. ignoring!'
        )
        flies_share_bins = False

    if flies_share_bins:
        assert 'x' in kwargs
        x = kwargs['x']
        values = data[x]
        binrange = (values.min(), values.max())
        # TODO delete
        #print(f'plot_one_dist_per_fly: {binrange=}')
        #

    assert all(x in data.columns for x in fly_cols), 'need level= kwarg for sort=False'
    # TODO or CI across flies? possible?
    for gn, gdf in data.groupby(fly_cols, sort=False):
        # TODO or try low alpha instead of fill=False?
        distplot(data=gdf, fill=False, **kwargs)


MODEL_STAT_ORDER: List[str] = ['logistic_scaled_num_spikes', 'num_spikes']

# TODO TODO should everything be z-scored before computing mix - max?
# or how to make more comparable (at least, in terms of the expected offset
# from 0)? i suppose even just making logistic ceiling higher would do that?
def plot_one_dist_per_model(data: pd.DataFrame, model_only: bool = False, **kwargs
    ) -> None:
    pnkc_class = get_single_unique(data[PNKC_CLASS_COL])

    assert 'alpha' not in kwargs, 'we set this manually below'

    # it can be NaN some places still, right? hopefully not here tho
    assert pd.notna(pnkc_class)

    model_input = pnkc_class_is_model(pnkc_class)
    if model_only:
        assert model_input

    # either 'binary'/'5comp' (for control/kiwi KC data), or
    # '2h+farn'/'farn+ma'/'2h+ma' (for Yang's diag-binaries data)
    # TODO still or NaN for model? must not be now? (no? delete?)
    mix = get_single_unique(data.mix)

    if not model_input:
        # TODO delete (don't want in new row_per_mix=False code)
        # TODO refactor to a kc_alpha const?
        #alpha = 1.0
        #distplot(data=data, fill=False, alpha=alpha, **kwargs)
        distplot(data=data, fill=False, **kwargs)
        if not model_input:
            return

    # TODO restore verbose=True
    verbose = False
    if verbose:
        stat = data.stat.unique()[0]
        print(f'{mix=} {stat=}')
        # TODO TODO check impression of these against hists/KDEs
        to_print = data[diff_col].round(decimals=1).value_counts(
            ).sort_index().to_frame().T
        to_print.columns.name = diff_col
        to_print.index = ['count']
        # was originally trying to fit them all in one row, but not
        # happening, so transposing again
        print(to_print.T.reset_index().to_string(index=False))
        print()

    group_cols = ['source']
    have_connectome_apl = has_connectome_apl(data)
    if have_connectome_apl:
        # so connectome_apl=False comes second, which is hopefully what
        # makes it into legend (so i don't have the dotted lines there, just
        # the hue)
        data = data.sort_values(by='connectome_apl', ascending=False,
            kind='stable'
        )
        group_cols.append('connectome_apl')

    # to avoid: FutureWarning: In a future version of pandas, a length 1
    # tuple will be returned when iterating over a groupby with a grouper
    # equal to a list of length 1. Don't supply a list with a single grouper
    # to avoid this warning.
    if len(group_cols) == 1:
        group_cols = group_cols[0]

    for gn, gdf in data.groupby(group_cols, sort=False):
        if have_connectome_apl:
            source, connectome_apl = gn
        else:
            source = gn
            connectome_apl = False

        # TODO (delete? presumably this was fixed?) fix legend so it shows
        # both linestyles (and uses that to show connectome vs uniform APL,
        # w/o color, like i do w/ markers for other legend. refactor?)
        linestyle = '--' if connectome_apl else '-'
        # I think I do like (default) fill=True here, right?
        distplot(data=gdf, linewidth=MODEL_LINEWIDTH, linestyle=linestyle,
            alpha=MODEL_ALPHA, **kwargs
        )


MIX_SUPP_IN_RESPONDERS_ONLY: bool = True

def plot_mixsupp_dists(df: pd.DataFrame, plot_dir: Path, palette: Palette,
    source_types: Sequence[str] = DEFAULT_SOURCE_TYPES_TO_PLOT, *,
    fname_suffix: str = '', **kwargs) -> None:
    """
    Args:
        **kwargs: passed to `map_dataframe` call, which calls `plot_one_dist_per_model`
    """
    panel = get_single_unique_kc_panel(df)
    multiple_mixes = has_multiple_mixes(df)

    # TODO use a const for this instead everywhere?
    diff_col = get_diff_col(df)

    model_df, kc_df, orn_df = get_model_kc_orn_data(df, source_types=source_types)

    # TODO rewrite below to not need these flags?
    model_only = model_df is not None and (kc_df is None and orn_df is None)
    kc_only = kc_df is not None and (model_df is None and orn_df is None)
    kc_and_orn_only = kc_df is not None and orn_df is not None and model_df is None
    model_vs_kcs = kc_df is not None and model_df is not None and orn_df is None
    assert sum([model_only, kc_only, kc_and_orn_only, model_vs_kcs]) == 1
    #

    facet_kws = dict(sharex=False, sharey=False, hue=PNKC_CLASS_COL, palette=palette)

    # TODO check no stats have NORM_PREFIX? (or at least not used? subset above if
    # present?)

    model_stat = None
    kc_stat = None
    plot_kws = dict(x=diff_col)
    if model_df is not None:
        # TODO TODO delete? what about KC vs ORN case? need that flag there too?
        plot_kws.update(dict(model_only=False))

        if kc_df is None:
            facet_kws.update(dict(col='stat', col_order=MODEL_STAT_ORDER))
            # TODO TODO delete?
            data = model_df
            #
        else:
            assert facet_kws.get('col') is None
            # TODO only print this if we actually have any stats w/ num_spikes?
            warn('plot_mixsupp_dists: dropping raw num_spikes for '
                'comparison between model and KC data! see model_only=True'
                ' version of plot for the distribution of those values'
            )
            # TODO check no other stats have 'num_spikes' in them (at least, w/o
            # 'logistic_scaled' also in them)?
            model_df = model_df[model_df.stat != 'num_spikes'].copy()

            # don't even need ignore_index=True now, since were just split from one df
            # w/ same index
            data = pd.concat([kc_df, model_df])

            # TODO compute from relevant subsections of data? or assert matches at
            # least? (prefer former)
            model_stat = 'logistic_scaled_num_spikes'
            kc_stat = 'mean_Fc_zscore'
            # TODO TODO restore just checking the specific elements we split from above?
            # if i define unique_stats from df, would not be subsetr, so have more than
            # wanted
            #unique_stats = set(data.stat.unique())
            #assert unique_stats == {kc_stat, model_stat}, f'{unique_stats=}'
    else:
        if orn_df is None:
            data = kc_df.copy()
            # TODO why col='stat' here? are there even multiple?
            facet_kws.update(dict(col='stat'))
        else:
            # TODO TODO TODO so if i keep this for ORN case, it will also have a
            # separate col for each. want to try on same axes instead?
            #
            # col='stat' would also split them apart, but want at least the 'KCs' vs
            # 'ORNs' in Axes titles
            facet_kws.update(dict(col='source'))
            # TODO TODO need other options here?
            data = pd.concat([kc_df, orn_df])

    # if False, will try using alpha per mix instead, by making a custom palette.
    # assumes only one combo of (source, mix) per axes in cases trying to use
    # `row_per_mix=False`.
    row_per_mix: bool = False
    if multiple_mixes:
        mix_order = sorted(df.mix.unique())[::-1]

        if set(df.mix.unique()) != set(NATMIX_MIX_TYPES):
            if not row_per_mix:
                warn(f'row_per_mix=False not currently supported for {panel=}')
            row_per_mix = True

        if not row_per_mix and model_df is None:
            data['source_and_mix'] = data['source'] + '/' + data['mix']
            # TODO want to support 'diag-binaries'? care?
            #assert set(df.mix.unique()) == set(NATMIX_MIX_TYPES), ('will have to make '
            #    f'below conditional if this trips. {df.mix.unique()=}'
            #)

            # TODO or use linestyle?
            # TODO TODO fix! alpha doesn't seem respected. dict palettes even allow
            # that?
            # TODO delete. not sure it will work w/ alpha
            #binary_alpha = 0.3
            #fullmix_alpha = 1.0
            #
            # TODO like this? 0.3 was too much desat for kc_color='m', and OK for cyan
            # 0.5 was still similar.
            #binary_desat = 0.65
            # 0.65 not enough
            binary_desat = 0.55
            fullmix_desat = 1.0

            # TODO actually do something with mix order here or no?
            kc_rgb = to_rgb(palette['KCs'])
            orn_rgb = None
            if orn_df is not None:
                orn_rgb = to_rgb(palette['ORNs'])

            palette = dict()
            # ['KCs/5comp', 'KCs/binary', 'ORNs/5comp', 'ORNs/binary']
            for source_and_mix in data['source_and_mix'].unique():
                source, mix = source_and_mix.split('/')
                if source == 'KCs':
                    rgb = kc_rgb
                else:
                    assert source == 'ORNs'
                    assert orn_rgb is not None
                    rgb = orn_rgb

                if mix == 'binary':
                    #alpha = binary_alpha
                    desat = binary_desat
                else:
                    assert mix == '5comp'
                    #alpha = fullmix_alpha
                    desat = fullmix_desat

                # TODO work? (no!) delete
                #palette[source_and_mix] = rgb + (alpha,)
                #
                palette[source_and_mix] = sns.desaturate(rgb, desat)

            facet_kws['palette'] = palette
            # replacing hue=PNKC_CLASS_COL
            facet_kws['hue'] = 'source_and_mix'
        else:
            facet_kws['row'] = 'mix'
            # will put 'binary' on top, and '5comp' on bottom. not sure i care
            # about order for other mixtures (e.g. yang's diag binaries), but
            # would have to do something else if i did
            facet_kws['row_order'] = mix_order

    # TODO TODO remove right column model+KC plot (num_spikes) (done),
    # unless i can find a way to get scales in line (still want to try?)
    # (prob just try that on a separate plot anyway, if at all)
    g = sns.FacetGrid(data=data, **facet_kws)
    g.map_dataframe(plot_one_dist_per_model, **plot_kws, **kwargs)

    # TODO TODO replace w/ different means of choosing when (and what) to fix xlim to?
    # TODO delete
    # yea, don't like this range for ORN data
    if orn_df is None and not (model_only or kc_only):
        # TODO update ranges to be wider for even the one plot i still want
        # to keep comparing logistic scaled model and KC data?
        if panel != 'diag-binaries':
            xlim = (-6, 4)
        else:
            xlim = (-6, 6)
        warn(f'plot_mixsupp_dists: hardcoding {xlim=} for model vs KC plot')
        g.set(xlim=xlim)
    #

    unit_str = 'KCs'
    if orn_df is not None:
        unit_str += '/glomeruli'

    suptitle = f'{panel}\ndistribution of "mixture suppression" across {unit_str}'
    suptitle_y = 1.10
    if MIX_SUPP_IN_RESPONDERS_ONLY:
        suptitle += '\nsilent KCs dropped'
        if orn_df is None and not (model_only or kc_only):
            suptitle += ' (both model and real KCs)'
    else:
        suptitle += '\nall KCs included'

    if g.col_names != []:
        # TODO also assert there are multiple columns? or not always true?
        assert facet_kws.get('col') is not None
        # this case is just when col= is in facet_kws
        assert model_stat is None, ('we should have two separat model stats when '
            'plotting model_only, so no single model_stat def'
        )
        # TODO restore?
        #assert model_only or kc_only
        #
        # currently have ORN and KC data on separate columns, but could change if data
        # range seems comparable
        assert model_only or kc_only or kc_and_orn_only
        # TODO change so col='source' instead, so it's labelled KCs / ORNs
        g.set_titles('{col_name}')
    else:
        # currently have ORN and KC data on separate columns, but could change if data
        # range seems comparable
        assert not (model_only or kc_only or kc_and_orn_only)
        # TODO restore?
        #assert not (model_only or kc_only)
        assert model_stat is not None
        g.set_titles('')
        stat_part = ('\n\nmix suppression computed on:\n'
            f'{model_stat} for model\n{kc_stat} for observed'
        )
        suptitle += stat_part
        suptitle_y += 0.1

    # TODO fontsize small enough? y high enough?
    g.fig.suptitle(suptitle, y=suptitle_y, fontsize=9)

    if model_df is not None or row_per_mix:
        # TODO what's lines=True doing for us?
        add_fixed_legend(g, data, palette, lines=True)
    # TODO like better than ax.legend() below?
    # trying this for both KC-only and KC-vs-ORN plot
    else:
        g.add_legend()

    # TODO put odor (components + mix? just mix name?) in this actually
    # (instead of ylabel), and show for both rows?
    g.set_xlabels(diff_col2desc(diff_col))

    ylabel_suffix = ''
    if 'row' in facet_kws:
        assert len(g.row_names) > 1, f'{g.row_names=}'
        assert g.axes.shape[0] == len(g.row_names)
    else:
        if 'col' in facet_kws:
            n_cols = data[facet_kws['col']].nunique()
            assert g.axes.shape == (1, n_cols)
        else:
            assert g.axes.shape == (1, 1)

        ylabel_suffix = f'density across {unit_str}'

    for (i, j, hue), gdf in g.facet_data():
        ax = g.axes[i, j]
        # TODO delete
        # TODO need to fix per-axes legends here? (maybe if i weren't using new
        # row_per_mix=False)
        #if kc_df is not None and orn_df is not None:
        #    breakpoint()
        #

        # TODO TODO still make sure that axes xlim are shared within a column, for the
        # kc-orn plot? (as i'll no longer be fixing xlim above)
        # (if i don't collapse down so each is just two lines, w/ diff alphas/whatever
        # [one for each of mix=binary/5comp])
        # TODO TODO if i make a plot like that, put legend inside each axes
        if hue != 0:
            continue

        ylabel = ''
        if j == 0:
            if 'row' in facet_kws:
                ylabel = f'{g.row_names[i]} mix\n'
            ylabel += ylabel_suffix

        if i != 0:
            ax.set_title('')

        ax.set_ylabel(ylabel)

        # TODO work? (yes, but would want smaller, at least for KC only. will try
        # g.add_legend())
        #ax.legend()

    # each plot should contain all mixes, so no mix in fname
    fname = f'mixsupp_dists_{panel}'
    if model_only:
        fname += '_model-only'

    if kc_only:
        fname += '_kc-only'

    if kc_and_orn_only:
        fname += '_kc-orn'

    savefig(g, plot_dir, f'{fname}{fname_suffix}')


def plot_response_strength_dist_per_model(data: pd.DataFrame,
    kc_df: Optional[pd.DataFrame] = None, **kwargs) -> None:

    model_only = False
    have_model = True
    have_orns = False
    unique_sources = set(data.source.unique())
    # TODO make args more explicit, rather than meaning of
    # data/kc_df being conditional?
    if kc_df is None:
        if has_model(data):
            model_only = True
        # assuming data is KC data only in this case
        else:
            have_model = False
            if 'ORNs' in unique_sources:
                assert unique_sources == {'ORNs'}, f'{unique_sources=}'
                have_orns=True
    else:
        group_cols = ['panel','mix','odor']
        model_odors = data[group_cols].drop_duplicates().reset_index(drop=True)
        kc_odors = kc_df[group_cols].drop_duplicates().reset_index(drop=True)
        # TODO may need to sort both? ok so far
        #
        # otherwise checking data for each ax._kc_df attribute I set, when making
        # kc_ax from ax.twiny(), would not make sense.
        assert model_odors.equals(kc_odors)
        del group_cols
    #

    if kc_df is not None:
        assert not have_orns
        # otherwise, kc_df should just be in data (and should not `kc_df = data`, as it
        # would screw up some checks below)
        assert have_model

    model_stat = None
    have_connectome_apl = has_connectome_apl(data)
    if have_model:
        model_stat = get_single_unique(data.stat)

    ax = plt.gca()
    if not model_only:
        if have_model:
            # TODO TODO sanity check we are only calling twiny once for the same ax? or
            # maybe define outside? (we were not, but now we should be. i assume it was
            # just plotting the same thing on top of itself tho)
            # TODO delete
            #print(f'{hasattr(ax, "_twiny_ax")=}')
            #print(f'{ax=} {id(ax)=}')
            if not hasattr(ax, '_twiny'):
                kc_ax = ax.twiny()
                # TODO delete
                # TODO TODO afterwards, assert all these have same ylim as ax do (and
                # set to share if not)
                print(f'called ax.twiny() to produce {kc_ax=} {id(kc_ax)=}')
                print('assert all twiny ax end with same ylim as ax? or do not hide '
                    'yticks?'
                )
                #
                ax._twiny_ax = kc_ax
                # NOTE: need to make sure only one unique kc_df is used per ax (and thus
                # per `kc_ax = ax.twiny()`, or checking this wouldn't make sense)
                ax._kc_df = kc_df.copy()
            else:
                assert hasattr(ax, '_kc_df')
                assert ax._kc_df.equals(kc_df)
            # TODO delete
            #print(f'{hasattr(ax, "_twiny_ax")=}')
            #

            # to check we no longer need subset_second_to... below
            # (because we should be subsetting outside appropriately now)
            assert set(kc_df.odor.unique()) == set(data.odor.unique())

        if kc_df is not None:
            kc_stat = get_single_unique_stat(kc_df)
            # TODO delete eventually? just to check we aren't also doing the
            # threshold subtracting outside now (does that even change name of stat?)
            assert kc_stat == 'Fc_zscore', f'{kc_stat=}'
        else:
            if have_orns:
                # NOTE: currently (at least when on separate cols, data *only* has ORN
                # data for calls with ORN data. and kc_df=None for all of those calls)
                orn_stat = get_single_unique_stat(data, strip_mean_prefix=False)
                assert orn_stat == ORN_RESP_COL, f'{orn_stat=}'
            # TODO not actually used tho right? delete?
            else:
                assert unique_sources == {'KCs'}
                kc_stat = get_single_unique_stat(data)
                assert kc_stat == 'Fc_zscore', f'{kc_stat=}'
            #

        # TODO TODO even make sense in this case? pretty sure it doesn't
        # when we are comparing against the logistic scaled data, right?
        # TODO still add flag to decide whether to do this, even for
        # num_spikes?
        if kc_df is not None and model_stat == 'num_spikes':
            # TODO add assertion kc_min is thresh (at least if using
            # remy's data, where it's one single thresh. couldn't do
            # same for yang's)?
            # TODO in yang's case, maybe i should be subtracting
            # something other than just min tho? can the threshold
            # really be reduced to one number in her case? matter?
            kc_min = kc_df.value.min()
            assert not np.isclose(kc_min, 0)
            kc_df = kc_df.copy()
            # TODO warn we are doing this? what was reason i thought it
            # made sense again?
            kc_df.value -= kc_min
            assert np.isclose(kc_df.value.min(), 0)
            kc_stat += ' - threshold'

    if not model_only:
        obs_df = kc_df
        if have_orns:
            obs_df = data
        # to handle KC-only case
        elif unique_sources == {'KCs'}:
            obs_df = data
        assert obs_df is not None

        obs_kws = dict(kwargs)
        obs_kws['fill'] =False

        assert_all_mix_after_component(obs_df)
        assert 'odor' in obs_df.columns, ('must pass level= for sort=False to work if '
            'in index.names and not .columns'
        )
        for odor, gdf in obs_df.groupby('odor', sort=False):
            # TODO refactor to share w/ below? (+ legend fixing)
            linestyle = mix_linestyle if is_mix(odor) else component_linestyle

            # TODO change label to include bit about mix? (+ component) (and to hide KCs
            # label if that's the only color being plotted? via removing label='KCs'
            # from obs_kws above) (or just put mix + comp info in title?)
            distplot(data=gdf, linestyle=linestyle, **obs_kws)

        # TODO fix kc color in outputs like response-strength_dists_control_5comp.pdf
        # (that plot model and KC data on same axes)
        # (currently gray, same as mix/component lines in legend, but add_fixed_legend
        # didn't seem to be responsible f or making them gray)
        # TODO + try to fix so area of KC and model curves actualy seem comparable.
        # show kc_ax yticklabels to sanity check? (otherwise, can't really use this
        # version of the plots anyway)

        # kc_df is always None in KC vs ORN case too, so wioll not end up screwing with
        # their shared axes limits / ticks / labels
        if kc_df is not None:
            kc_color = kwargs.get('color')
            assert kc_color is not None
            # TODO also want to change color of anything for ORN? prob not...
            kc_ax.set_xlabel(f'KC {kc_stat}', color=kc_color)
            # TODO refactor to share w/ twinx mb_model APL Vm plotting?
            # alpha not supported here
            kc_ax.tick_params(axis='x', color=kc_color)
            for text in kc_ax.xaxis.get_ticklabels():
                # alpha also not supported here
                text.set_color(kc_color)
            #

            # TODO set color of top spine / ticks, like in APL Vm plotting elsewhere?
            # TODO factor out fn for setting color of all the things i typically want,
            # to hong2p.viz?
            kc_ax.spines['right'].set_visible(False)
            kc_min = kc_df.value.min()
            kc_max = kc_df.value.max()
            # TODO any point in this?
            kc_ax.set_xlim([kc_min - 0.5, kc_max + 0.5])

    if not have_model:
        return

    group_cols = ['source']

    # previously had a code path that would use linestyles or at least some other
    # row/col/whatever for connectome-APL, but now i'm just always dropping
    # connectome-APL so i can use linestyle for mix vs top-comp, and don't care to
    # show connectome-APL cases currently. so now, i'm just unconditionally
    # asserting `not have_connectome_apl`
    #if simplify_models:
    #    assert not have_connectome_apl
    #
    assert not have_connectome_apl

    # TODO even need sort, or just the reset_index?
    data = data.sort_values(by='odor', kind='stable', key=odor_sort_fn
        ).reset_index(drop=True)

    # TODO and why does this matter again? might just not assign to linestyles
    # properly? am i still assuming a certain order for that?
    assert_all_mix_after_component(data)
    group_cols.append('odor')

    if have_connectome_apl:
        # so connectome_apl=False comes second, which is hopefully what
        # makes it into legend (so i don't have the dotted lines there,
        # just the hue)
        data = data.sort_values(by='connectome_apl', ascending=False, kind='stable')
        group_cols.append('connectome_apl')

    assert all(x in data.columns for x in group_cols), ('would need level=<by> '
        f'if {group_cols=} were not in data.columns (but data.index.names), for '
        'sort=False to actually work'
    )
    # to avoid a FutureWarning
    if len(group_cols) == 1:
        group_cols = group_cols[0]

    for gn, gdf in data.groupby(group_cols, sort=False):
        source, odor = gn
        # TODO refactor to share w/ above?
        linestyle = mix_linestyle if is_mix(odor) else component_linestyle

        # TODO sufficient to just check NaN? need the dirname -> class
        # -> pnkc_class_is_model checking below too (prob not)?
        assert gdf.model_dirname.notna().all()

        model_dirname = get_single_unique(gdf.model_dirname)
        pnkc_class = model_pnkc_class(model_dirname)
        assert pnkc_class_is_model(pnkc_class), (f'{source=} does not seem to '
            'refer to model data'
        )
        assert source not in EXPECTED_NONMODEL_PNKC_VALS, f'{source=}'

        # TODO fill=False? (for at least some [which?], yes. everything?)
        distplot(data=gdf, ax=ax, fill=False, linewidth=MODEL_LINEWIDTH,
            linestyle=linestyle, **kwargs
        )


FIXED_NUM_SPIKES_XMIN: float = 0.0
# TODO like this?
# much more (double digit percentages) of both binary/5comp kiwi panels are past 10,
# but barely any of control mixes are.
# TODO TODO TODO care? currently still (w/ =15.0):
# - 14% (66/469 responder) (KC, odor) counts above X-max in kiwi/5comp case
# - 33% (141/423) in kiwi/binary case
# - only 2.8% (17/605) (binary) and 1.5% (9/606) (5comp) control cases
#
# increased to =20.0, and now:
# - 24.1% in kiwi/binary
# - 9.4% in kiwi/5comp
# ...and not sure i want to extend xmax much beyond this. starting to be harder to focus
# on the part of the curve i care about.
# TODO TODO still try =25.0 or something, to compare?
#
# TODO TODO and is it claw model mainly to blame for most of clipped counts?
FIXED_NUM_SPIKES_XMAX: float = 20.0

# TODO delete log_yscale flag? don't think i ever liked =True
def plot_response_strength_dists(df: pd.DataFrame, plot_dir: Path, palette: Palette,
    source_types: Sequence[str] = DEFAULT_SOURCE_TYPES_TO_PLOT, *,
    orn_kc_on_diff_rows: bool = False, log_yscale: bool = False) -> None:

    # should only have input with one (normalized) panel and one mix
    panel, mix = get_single_unique_panel_and_mix(df)

    model_data, kc_data, orn_data = get_model_kc_orn_data(df, source_types=source_types)
    assert not (model_data is None and kc_data is None)
    # TODO or at least plot ORNs separately by themselves, in a way comparable
    # to KC data, if can't get a good plot with them on the same axes. maybe just plot
    # w/ separate row / col (and share[x|y]/common_norm=False, as appropriate)
    if orn_data is not None:
        assert kc_data is not None and model_data is None, ('currently expect to '
            'plot only either model-vs-KC or ORN-vs-KC'
        )
    else:
        assert not orn_kc_on_diff_rows

    facet_kws = dict(hue=PNKC_CLASS_COL, palette=palette)
    if model_data is not None:
        data = model_data
        assert has_model(data)
        if has_connectome_apl(data):
            # TODO make a separate plot with the connectome_apl=True
            # data in this case? prob don't really care to...
            warn('plot_response_strength_dist: subsetting data to '
                'connectome_apl=False. linestyle would be ambiguous between '
                'uniform-vs-connectome APL and mix-vs-component'
            )
            data = data[data.connectome_apl == False]

        # TODO delete if fixed (or disable these plots?)
        if kc_data is not None:
            warn('plot_response_strength_dists: KC vs model hist areas may not both be '
                'correct for this model + KC version of the plot (and KC color '
                'currently broken)'
            )
        #
        facet_kws.update(dict(col='stat', col_order=MODEL_STAT_ORDER))
    else:
        if orn_data is None:
            data = kc_data
            # TODO can i simplify called fn at all, now that using hue/palette here?
            # have i already? (i.e. removing separate explicit references to KC color,
            # etc) (or is that part of what broke the KC color in model vs KC case?
            # don't care too much about that case anyway for this plot type)
            facet_kws.update(dict(hue=PNKC_CLASS_COL, palette=palette))
        else:
            data = pd.concat([kc_data, orn_data])
            facet_kws.update(dict(hue=PNKC_CLASS_COL, palette=palette))
            # can i have it be both row and hue? yes
            if orn_kc_on_diff_rows:
                facet_kws['row'] = PNKC_CLASS_COL

    g = sns.FacetGrid(data=data, sharey=False, sharex=False, **facet_kws)

    # TODO delete
    #print()
    #print('plot_response_strength_dists: before map_dataframe:')
    #
    # TODO rename plot_response_strength_dist_per_source
    g.map_dataframe(plot_response_strength_dist_per_model,
        kc_df=(None if model_data is None else kc_data),
        x='value', alpha=MODEL_ALPHA, log_scale=(False, log_yscale)
    )
    # TODO delete
    #print()
    #print('plot_response_strength_dists: after map_dataframe')
    #print()
    #

    # TODO want any fixed x/y limits in any cases? maybe for some of other
    # other plots? maybe if a -F CLI arg set, and then fix all as desired for thesis.
    # (currently fixing num_spikes, but maybe only do want to do that if requested by
    # such a CLI arg...)

    g.fig.subplots_adjust(hspace=0.6)

    # TODO version of this plot including nonresponding values too
    # (mainly for sanity checking)?
    units_str = 'KCs'
    pair_unit_str = 'KC'
    if orn_data is not None:
        units_str += ' & ORNs'
        pair_unit_str += '|glomerulus'

    add_fixed_legend(g, data, palette, lines=True, add_mix_vs_comp_linestyle=True)

    # TODO TODO set something with N? copy code for that from natmix_data/analysis.py?
    g.set_titles('')

    const_stat_name = None

    if not orn_kc_on_diff_rows:
        # it should be this value when row= not specified in FacetGrid init
        assert g.row_names == []
        assert g.axes.shape[0] == 1

        if orn_data is not None and not orn_kc_on_diff_rows:
            # TODO also only do this if source/stat/etc not in row=/col= of facet_kws?
            assert kc_data is not None
            orn_stat = get_single_unique_stat(orn_data, strip_mean_prefix=False)
            kc_stat = get_single_unique_stat(kc_data, strip_mean_prefix=False)
            # TODO or two lines? this is ok. slightly clipped on left side, w/ current
            # fig size (prefer two lines. delete)
            #const_stat_name = f'ORNs: {orn_stat}, KCs: {kc_stat}'
            #
            const_stat_name = f'ORNs: {orn_stat}\nKCs: {kc_stat}'

    ij2stat_name = dict()
    num_spikes_dropped_count = 0
    num_spikes_ax = None
    for (i, j, hue), gdf in g.facet_data():
        # should only be true if (as in ORN vs KC case), hues exist only on one row/col,
        # or something like that
        if len(gdf) == 0:
            continue

        ax = g.axes[i, j]

        mmin = gdf.value.min()
        mmax = gdf.value.max()

        model_stat_name = None
        if model_data is not None and kc_data is None:
            # should still only be one stat per facet
            model_stat_name = get_single_unique_stat(gdf)

        if model_stat_name == 'num_spikes':
            # this should be the hue level (share def w/ above tho?), so also should
            # only have one per iteration of this loop
            pnkc_class = get_single_unique(gdf[PNKC_CLASS_COL])

            num_spikes_ax = ax

            n_clipped = (gdf.value > FIXED_NUM_SPIKES_XMAX).sum()
            # TODO delete
            # TODO ever fail? warn instead if so
            # this is not true w/ 15 (but probably do want 15 instead of 10 if i want a
            # chance of not throwing away so much kiwi data)
            #assert n_clipped > 0, ('expected all models to have some spike counts above'
            #    f' {FIXED_NUM_SPIKES_XMAX=}'
            #)
            if n_clipped == 0:
                warn('expected all models to have some spike counts above '
                    f'{FIXED_NUM_SPIKES_XMAX=}, but {panel}/{mix} {pnkc_class} did not'
                )

            if n_clipped > 0:
                warn(f'plot_response_strength_dists: setting {FIXED_NUM_SPIKES_XMAX=} '
                    f'for model-only num_spikes facet, which clips {n_clipped}/'
                    f'{len(gdf)} (KC, odor) counts above this for hue {pnkc_class=}'
                )
                num_spikes_dropped_count += n_clipped
        else:
            # in case there are multiple hue levels sharing same facet
            curr_xmin, curr_xmax = ax.get_xlim()

            # TODO (delete?) 0.05 instead of 0.25?
            margin = (mmax - mmin) * 0.05
            # TODO is there actually any need for this? was this just to counteract
            # fixed margins i had above? try deleting?
            ax.set_xlim([min(curr_xmin, mmin - margin), max(curr_xmax, mmax + margin)])

        stat_name = const_stat_name
        if const_stat_name is None:
            if g.col_names == []:
                assert model_data is None
                stat_name = get_single_unique_stat(gdf)
            else:
                assert model_data is not None
                stat_name = g.col_names[j]

        ij = (i, j)
        if ij in ij2stat_name:
            # to catch cases where two things w/ diff stats are plotted on same axes
            # (like ORN vs KCs, when facet_kws doesn't separate them on row=/col=
            assert stat_name == ij2stat_name[ij], f'{stat_name=} != {ij2stat_name[ij]=}'
            # don't want to skip hues that only exist in separate ij
            if hue != 0:
                continue

        ij2stat_name[ij] = stat_name

        gdf_is_orn_data = False
        if orn_kc_on_diff_rows:
            assert orn_data is not None
            unique_source = get_single_unique(gdf.source)
            gdf_is_orn_data = unique_source == 'ORNs'
            if not gdf_is_orn_data:
                assert unique_source == 'KCs'

            stat_name = f'{unique_source} ({stat_name})'

        ylabel = ''
        if j == 0:
            ylabel = 'density across KCs'
            if orn_data is not None:
                if not orn_kc_on_diff_rows:
                    ylabel += ' / glomeruli'

                # TODO assert all KC data otherwise here?
                elif gdf_is_orn_data:
                    assert i == 1, 'expected ORNs on second row'
                    ylabel = 'density across glomeruli'

        if i != 0:
            ax.set_title('')

        ax.set_ylabel(ylabel)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel(stat_name)

    suptitle_y = 1.10
    # this is only (currently) set for the model-only version of the plot
    if num_spikes_ax is not None:
        assert num_spikes_dropped_count > 0, ('expected some num_spikes values would be'
            f' above {FIXED_NUM_SPIKES_XMAX=}, but seemingly none'
        )

        # TODO TODO also change bins, to not clip stuff? meh

        n_total_num_spike_count = (data.stat == 'num_spikes').sum()
        frac_dropped = num_spikes_dropped_count / n_total_num_spike_count
        n_dropped_str = (
            f'{num_spikes_dropped_count}/{n_total_num_spike_count} ({frac_dropped:.1%})'
        )
        warn(f'plot_response_strength_dists: setting {FIXED_NUM_SPIKES_XMAX=} '
            f'(for model-only num_spikes facet) clipped a total of {n_dropped_str}'
        )

        # TODO log_yscale this one? prob not...
        num_spikes_ax.set_xlim([FIXED_NUM_SPIKES_XMIN, FIXED_NUM_SPIKES_XMAX])

        title = num_spikes_ax.get_title()
        if title != '':
            title += '\n'
        title += f'{n_dropped_str} (KC, odor) counts above X-max'
        num_spikes_ax.set_title(title, fontsize=7)

        suptitle_y += 0.05

    suptitle = (f'{panel}/{mix}\nactivation strengths across {units_str}\n'
        f'responder ({pair_unit_str}, odor) pairs only'
    )
    g.fig.suptitle(suptitle, y=suptitle_y)

    fname = f'response-strength_dists_{panel}_{mix}'
    if model_data is None:
        if orn_data is not None:
            fname += '_kc-orn'
            if orn_kc_on_diff_rows:
                fname += '_sep-rows'
        else:
            fname += '_kc-only'
    else:
        assert orn_data is None

    if kc_data is None:
        assert orn_data is None
        fname += '_model-only'

    if log_yscale:
        fname += '_logy'

    savefig(g, plot_dir, fname)


def main():
    high_capacity_root: Path = Path('/mnt/d0')

    # NOTE: redefined based on `-r <output-root-name>` CLI arg, if passed
    output_root_name: str = 'yang_mix_outputs'

    first_for_each: List[str] = [PNKC_CLASS_COL, 'connectome_apl']

    stat_fname_part = stat2fname_part(mixsupp_col)
    # NOTE: this is redefined below to include suffix with logistic scaling params,
    # if using any nondefault ones (via -l/--logistic-scaling-param-index)
    # (put before mixsupp_order_fname_suffix)
    #
    # the versions of the CSV/parquet *without* '_panelmean' suffix include the mix
    # suppression values for all panels, for each model, although still all models
    # are ordered by mean natmix mixsupp (so excluding e.g. diag-binaries)
    # TODO change formatter so that fname doesn't get broken across lines?
    mixsupp_order_fname: str = mixsupp_parts2order_fname(mixsupp_col)

    SYN_DIAG_PANEL: str = 'syn-diag-binaries'

    # TODO TODO TODO add flag to ignore LR cache (and check i can actually still
    # recreate at least the -m ones, if not all -f options that currently worked)
    # (or just use env var for that? need to test either way)
    # TODO add option to skip all model data? all KC data?
    parser = ArgumentParser(description=f'writes outputs under {output_root_name}, '
        'which is created in current directory by default (or under '
        f'{high_capacity_root}, if -s/--save-dynamics passed, and if that directory '
        'exists)'
    )
    parser.add_argument('-r', '--output-root', action='store', default=output_root_name,
        help='name of output root to be created under current directory. Default: '
        f'{output_root_name}\nIf -d passed, will try making it somewhere else (see -d '
        'help or message above), other than current directory.'
    )
    parser.add_argument('-S', '--skip-panels', action='store', default=SYN_DIAG_PANEL,
        nargs='?', help=f"panels to skip (comma separated). Default: '{SYN_DIAG_PANEL}'"
        '\nPass with no arguments to skip no panels, including default skipped.'
    )
    # TODO refactor to share this (and -e/-x) w/ step_model_pn_apl, and
    # natmix_data/analysis.py?
    parser.add_argument('model_output_dirnames', nargs='?', help='comma separated '
        'list of substrings matching model output directory names \n(subdirectories of '
        '<model_output_root>/<panel> directories). \nsee also -e and -x.'
    )
    parser.add_argument('-u', '--use-cache', action='store_true', help='uses cache for '
        'all model outputs, when available. this may mean no models are re-run.'
    )
    parser.add_argument('-o', '--only-analyze-cache', action='store_true', help='will '
        'not run any models. will only load cached model outputs and analyze those. '
        'implies -u/--use-cache.'
    )
    parser.add_argument('-f', '--full-model-params', action='store_true', help='will '
        'run models for each set of parameters in `FULL_MODEL_KW_LIST`, instead of the '
        'much shorter `SHORT_MODEL_TUNE_KWS`'
    )
    parser.add_argument('-e', '--exclude-substrings', action='store', help='comma '
        'separated list of substrings to EXCLUDE model output directory names \n'
        'containing them. complementary to model_output_dirnames.'
    )
    parser.add_argument('-x', '--exact-model-dirnames', action='store_true',
        help='model_output_dirnames is interpreted as a list of exact directory names,'
        '\nrather than a list of substrings contained within some model output \n'
        'directory names.'
    )
    # TODO care to add option to make all those plots here (or do by default), instead
    # of having to call that other script?
    #
    # currently have this option so the model output directories will have dynamics
    # outputs, and can be used as input to natmix_data/analysis.py, to have it plot the
    # dynamics and cell inspecition/debugging plots
    parser.add_argument('-d', '--save-dynamics', action='store_true', help='Will write '
        'NetCDF files with many model dynamic variables (change plot root, since can '
        'take a lot of disk space). Will make output root under high_capacity_root='
        f'{high_capacity_root}, if it exists, but will fall back to default current '
        'directory.'
    )
    parser.add_argument('-i', '--ignore-existing', action='store_true', help='will '
        'overwrite existing hierarchichal clustering and plots of ORN mix responses '
        'constructed from component ORN data'
    )
    parser.add_argument('-m', '--max-supp-models-only', action='store_true', help='Will'
        ' only analyze the model with the most average kiwi/control (binary+5comp) mix '
        'suppression (i.e. the most negative), within each combination of '
        f"{first_for_each} (excluding 'connectome_apl' if -M/--simplify-models passed)."
        f'{mixsupp_order_fname} must exist under plot_root from a previous run, which '
        'defines the order based on mixture suppression calculated in that run.\n'
        'NOTE: -l/--logistic-scaling-param-index other than default 0 add a suffix to '
        'the name of the file this order is written to (and read from), as logistic '
        'scaling is relevant to mixture suppression calculated on scaled spike counts.'
    )
    parser.add_argument('-M', '--simplify-models', action='store_true', help='Will '
        'exclude from analysis all connectome-APL models, as well as all prat-bouton '
        '/ nonclaw (e.g. wd20) models (as neither made a huge difference, at least in '
        'terms of average kiwi/control mix suppression), to simplify plots for thesis '
        '(and thus the writing about it).'
    )
    parser.add_argument('-O', '--old-model-kws', action='store_true', help='will use '
        'older definition OLD_SHORT_MODEL_TUNE_KWS, which had '
        'use_connectome_APL_weights=True variants of each of several models, and did '
        'not include parameters now known to help improve mixture suppression'
    )
    parser.add_argument('-N', '--norm-across-models', action='store_true',
        help='Restores old behavior where (for intensity point plots) model values are '
        'scaled such that the max single (model, odor) value, across all models, was '
        'set to 1. New behavior is to scale each model variant separately, so each will'
        ' have max odor response set to 1.'
    )
    parser.add_argument('-l', '--logistic-scaling-param-index', action='store',
        type=int, default=0, help='Which element to select from '
        'LOGISTIC_SCALING_PARAM_LIST (default: 0)\nIf >0, model outputs will be saved '
        'under a subdirectory including logistic params in name.'
    )
    # NOTE: will currently need to recompute models outputs (i.e. no -u or -o CLI flags)
    # if cached versions were generated with the other value for this flag
    # TODO (delete) provide means of renaming odors in cached model outputs, to avoid
    # need for regenerating? (doesn't seem possible in general tho... will currently
    # just err if cached odor names don't match what would be the model input names)
    parser.add_argument('-L', '--leave-concs', action='store_true', help='if passed, '
        'will leave processed concentrations in odor (component and mix) names. If '
        'not passed, certain analyses may fail if trying to analyze data where any one '
        'panel contains >=1 odor(s) at multiple concentrations. If using cached model '
        'outputs (via either -u or -o args), will err if cache values were created with'
        'a different value for this. Re-run without -u or -o to regenerate model '
        'responses.'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='print more')

    # also sets appropriate al_util module variables for the -c/-P flags to work
    args = add_check_outputs_unchanged_CLI_flag_and_parse_args(parser)

    output_root = args.output_root
    skip_panels = args.skip_panels
    model_output_dirnames = args.model_output_dirnames
    use_cache = args.use_cache
    only_analyze_cache = args.only_analyze_cache
    full_model_params = args.full_model_params
    exclude_substrings = args.exclude_substrings
    exact_model_dirnames = args.exact_model_dirnames
    save_dynamics = args.save_dynamics
    ignore_existing = args.ignore_existing
    max_supp_models_only = args.max_supp_models_only
    simplify_models = args.simplify_models
    norm_across_models = args.norm_across_models
    logistic_scaling_param_index = args.logistic_scaling_param_index
    old_model_kws = args.old_model_kws
    leave_concs_in_odors = args.leave_concs
    verbose = args.verbose
    quiet = not verbose
    # TODO (still true? delete?) set in fit_and_plot_mb_model instead? (currently errs
    # if try_cache=False [default] and cache_only=True [not default])
    if only_analyze_cache:
        use_cache = True

    if skip_panels is not None:
        skip_panels_str = str(skip_panels)
        skip_panels = [x for x in skip_panels.split(',')]
        assert all(x.strip() == x for x in skip_panels)
        skip_panels = set(skip_panels)
        # TODO if 'diag-binaries' is included, have that also skip any of the others
        # ones that start with that?

    output_root_name: str = str(args.output_root).strip()

    if max_supp_models_only and not full_model_params:
        warn('-m implies -f, but -f not passed. setting full_model_params=True')
        full_model_params = True

    if model_output_dirnames is not None:
        model_output_dirnames = model_output_dirnames.split(',')

    if exclude_substrings is not None:
        exclude_substrings = exclude_substrings.split(',')
        assert len(exclude_substrings) == len(set(exclude_substrings))

    if old_model_kws:
        assert not full_model_params, '-f incompatible with -O'
        # TODO support? prob not
        assert not max_supp_models_only, '-m incompatible with -O'

    had_str_model_filtering = model_output_dirnames or exclude_substrings

    # TODO instead of RHS, check whether we skipped any models (for reasons other
    # than missing cache, which i think are counted separately anyway)? (by checking
    # after loop loading models)
    # TODO actually, just add assertion(s) that doing that would be consistent with this
    # def now? since i think i'd like to keep this simpler def up top now
    unrestricted_full_model_params = (
        full_model_params and not (max_supp_models_only or simplify_models or
            had_str_model_filtering
        )
    )

    # currently asserting that we actually do only have one specific model per PNKC
    # class below, if this is True
    one_model_per_pnkc_class = simplify_models and (
        max_supp_models_only or not full_model_params
    )

    if not full_model_params:
        if not old_model_kws:
            model_tune_kws = SHORT_MODEL_TUNE_KWS
        else:
            model_tune_kws = OLD_SHORT_MODEL_TUNE_KWS
    else:
        # TODO TODO see 2026-05-25_yang_err.txt for an invalid_argument olfsysm error i
        # hadn't seen before, in prat-claws_True__prat-boutons_True__pn-claw-to-apl__
        # True__allow-net-inh-per-claw_True__connectome-APL_True case
        # (can i repro? fix?)
        model_tune_kws = FULL_MODEL_KW_LIST
        warn(f'running models on all {len(FULL_MODEL_KW_LIST)} elements in '
            'FULL_MODEL_KW_LIST, because -f/--full-model-params passed!'
        )

    if simplify_models:
        len_before = len(model_tune_kws)
        model_tune_kws = [x for x in model_tune_kws
            if not (x.get('prat_boutons') or x.get('use_connectome_APL_weights') or
                # TODO TODO working? why in chosen_modeldirs CSV output tho (the
                # -m one only, but still)?
                # weight_divisor should be present in all 'nonclaw' cases, so now should
                # just be left w/ 'uniform' and 'claw'
                x.get('weight_divisor')
            )
        ]
        n_dropped = len_before - len(model_tune_kws)
        # should be true no matter the other options
        assert n_dropped > 0
        warn(f'dropped {n_dropped}/{len_before} models with use_connectome_APL_weights='
            'True or prat_boutons=True, as -M/--simplify-models passed'
        )

        first_for_each = [x for x in first_for_each if x != 'connectome_apl']

    if verbose:
        print()
        print('model_tune_kws (each element has parameters for a separate model run):')
        pprint(model_tune_kws)
        print()

    # by default, create output root under current directory
    root_parent: Path = Path('.')

    if save_dynamics:
        if high_capacity_root.is_dir():
            root_parent = high_capacity_root
            warn(f'writing to {output_root_name} under {high_capacity_root} instead of '
                'under usual current directory, b/c saving (potentially large) dynamics'
                ' outputs (-s/--save-dynamics passed)'
            )
        else:
            warn('-s/--save_dynamics passed but high_capacity_root='
                f'{high_capacity_root} did not exist! falling back to saving outputs '
                'under current directory. hardcode your own high_capacity_root if you '
                'do not wish to save dynamics (which can be many GB) under current '
                'directory'
            )

    model_root = (root_parent / output_root_name).resolve()
    model_root.mkdir(exist_ok=True)

    # directories under this used to just be under model_root, but especially when
    # running on FULL_MODEL_KW_LIST, it was getting pretty cluttered
    tune_root = model_root2tune_root(model_root)
    tune_root.mkdir(exist_ok=True)

    simplify_models_fname_part = 'no-connectomeAPL-wd20-or-boutons'
    # NOTE: -m implies -f, so should just need these two fname parts
    max_supp_models_only_fname_part = 'max-mixsupp-only'

    old_model_kws_fname_part = 'old-model-kws'

    subdirname_parts = []
    if simplify_models:
        subdirname_parts.append(simplify_models_fname_part)
    if max_supp_models_only:
        subdirname_parts.append(max_supp_models_only_fname_part)
    if old_model_kws:
        subdirname_parts.append(old_model_kws_fname_part)

    if unrestricted_full_model_params:
        assert not (simplify_models or max_supp_models_only or old_model_kws)
        subdirname_parts = ['full-param-sweep']

    # whether each parameterization of model will have it's max odor response (w/in
    # panel[+mix]) scaled to 1 (=True), or whether all models (w/in panel[+mix]) will be
    # scaled such that max odor response across all models is 1 (=False).
    NORM_PER_MODEL: bool = not norm_across_models
    if NORM_PER_MODEL:
        subdirname_parts.append('norm-per-model')

    logistic_scaling_params = LOGISTIC_SCALING_PARAM_LIST[logistic_scaling_param_index]
    # TODO only do if verbose?
    print_logistic_scaling_effect(**logistic_scaling_params)
    print()
    #
    logistic_scaling_fname_part = ''
    logistic_scaling_title_str = ''
    if logistic_scaling_param_index != 0:
        warn(f'using logistic scaling params at index {logistic_scaling_param_index}:\n'
            f'{pformat(logistic_scaling_params)}\n(change via `-l <index>`. see '
            'LOGISTIC_SCALING_PARAM_LIST)'
        )
        # TODO TODO check i'm also writing order parquets and plots with this too
        # (should be)
        logistic_scaling_fname_part = ('_logistic-scaling_'
            + '_'.join(f'{k}-{v:.1f}' for k, v in logistic_scaling_params.items())
        )
        subdirname_parts.append(logistic_scaling_fname_part)

        prefix = mixsupp_order_fname2shared_prefix(mixsupp_order_fname)
        mixsupp_order_fname = (
            f'{prefix}{logistic_scaling_fname_part}{mixsupp_order_fname_suffix}'
        )
        # TODO factor to mb_model? also used in print_logistic_scaling_effect
        # e.g. {'k': 0.5, 'x0': 2.0, 'L': 3.0} -> 'k=0.5, x0=2.0, L=3.0'
        logistic_scaling_str = format_params(logistic_scaling_params, sort=False,
            float_format='.1f'
        )
        logistic_scaling_title_str = '\nlogistic scaling: ' + logistic_scaling_str

    # TODO rename these two, to be more clear that plot_root is just used for plots that
    # can vary w/ e.g. simplify_model (so i can have two parallel subdirs of them),
    # whereas model_root currently has both plots and model output dirs
    plot_root = model_root
    if len(subdirname_parts) > 0:
        subdir_name = '_'.join(subdirname_parts)
        warn(f'saving outputs that contain model data under {model_root.name}/'
            f'{subdir_name} subdirectory, because of some non-default CLI args'
        )
        plot_root = model_root / subdir_name
        plot_root.mkdir(exist_ok=True)


    def plot_all_comparisons_for(plot_fn: Callable, df: pd.DataFrame, **kwargs) -> None:
        # TODO would re-ordering signature of these fns so plot_dir is first make
        # things easier? matter? (going to try treating all as keyword arguments, for
        # now)
        """
        Takes a function with arguments:
            - `df`: a DataFrame with 'source_type' column (and most/all other data in
              separate columns, like after `.reset_index()`)

            - `plot_dir`: a directory `Path`, in which to save plots

            - `source_types`: a sequence of 'source_type' (e.g. 'model', 'KCs', ORNs')
              values to analyze together

        ...and calls it for each combination we want to analyze in thesis:
        - model vs KCs
        - KCs only
        - model only
        - KCs vs ORNs

        Expects each `plot_fn` to change output filenames to be unique in each call.
        """
        source_types_list = [
            ('model', 'KCs'),
            ('KCs',),
            ('model',),
            # TODO TODO plot ORN only too, for sanity checking of normalization / axis
            # labelling on the KC vs ORN plots
            ('KCs', 'ORNs'),
        ]
        # TODO add flag to control whether to fail / skip if we are missing any of the
        # source types (particularly 'ORNs', but also 'KCs'. maybe even 'model')?
        for source_types in source_types_list:
            plot_dir = model_root
            # plot_root is a subdir of model_root (or same directory). if subdir, it is
            # named with model-specific processing choices, to keep separate analysis
            # choices in parallel, while not cluttering with multiple fnames
            if 'model' in source_types:
                plot_dir = plot_root

            try:
                plot_fn(df=df, plot_dir=plot_dir, source_types=source_types, **kwargs)
            except MissingRequestedSourcesError as err:
                warn(err)
                warn(f'skipping {source_types=} for this odor panel')


    # otherwise we currently won't see the names of plots being saved printed in blue
    al_util.verbose = True

    if max_supp_models_only:
        assert order_by_mean_mixsupp_from in MODEL_STATS
        mixsupp_order_parquet = model_root / mixsupp_order_fname
        if not mixsupp_order_parquet.exists():
            # TODO or warn and don't order?
            raise IOError(f'{order_by_mean_mixsupp_from=} requested, but '
                f'{mixsupp_order_parquet} did not exist! script must run through to '
                'part at end that writes these parquet files (and write such a file for'
                ' the requested stat)'
            )

    # TODO CLI flag to set drop_exp_type=False/True? (and same flag to control
    # similar debug info for Remy KC experiments, if i include any of that)
    # TODO add CLI flag to skip processing KC data?
    # TODO TODO check drop_expt_type=False doesn't break anything (+ fix if not)
    # (seems ok, but still want to compare plots visually w/ and w/o)
    yang_df, yang_bin_df = load_yang_kc_data(drop_exp_type=False, verbose=verbose)

    model_strs = [format_model_params(x) for x in model_tune_kws]

    unique_pnkc_classes = {model_pnkc_class(x) for x in model_strs}
    actually_have_one_model_per_pnkc_class = (
        len(model_tune_kws) == len(unique_pnkc_classes)
    )
    if not one_model_per_pnkc_class:
        if actually_have_one_model_per_pnkc_class:
            assert model_tune_kws == SHORT_MODEL_TUNE_KWS, ('this was only case i was '
                'currently expecting to only have one model per PNKC class (except '
                'those where one_model_per_pnkc_class is already True), and only '
                'for current hardcoded version, which has no connectome-APL versions'
            )
            one_model_per_pnkc_class = True
    else:
        assert actually_have_one_model_per_pnkc_class, (f'{one_model_per_pnkc_class=} '
            'bool (computed from CLI args above), but actually have multiple models '
            'in some classes here. bool computation above must be wrong! or other bug.'
        )
    del unique_pnkc_classes, actually_have_one_model_per_pnkc_class

    if verbose:
        model_str2abbrev = {m: abbrev_model_id(m) for m in model_strs}
        # TODO delete? now that i'm mostly not using the abbrevs anyway (for now, at
        # least, after mostly using model_pnkc_class now)
        #
        # these abbreviations are also applied (via abbrev_model_id fn) elsewhere
        print('model ID -> abbrev:')
        pprint(model_str2abbrev)
        print()

    df = megamat_orn_deltas(drop_diags=False)

    # no other panels in output of fn above
    #
    # indexing differently for `mdf`, since some of code using it below expects 'panel'
    # column level to be there still, but `df.loc[:, 'megamat']` would drop that level
    mdf = df.loc[:, df.columns.get_level_values('panel') == 'megamat']
    diags = df.loc[:, df.columns.get_level_values('panel') == 'glomeruli_diagnostics']

    tune_df = mdf
    del mdf

    series_list = []
    # make binary mixtures of synthetic diagnostics (150Hz spike delta to each of the
    # two glomeruli, for all combinations of them)
    gloms_to_mix = ['DM4', 'VM5d', 'DC3']
    syndiag_comp_names = set()
    for x in gloms_to_mix:
        name = f'{x}-300'
        # don't need to add names in loop below, because that loop only adds mixes,
        # and we just need a set of the string exactly as they appear in component name
        # part of odor str
        syndiag_comp_names.add(name)
        ser = pd.Series(index=df.index.copy(), name=f'{name} @ 0', data=0.0)
        ser.loc[x] = 300.0
        ser.name = (SYN_DIAG_PANEL, ser.name)
        series_list.append(ser)

    glom_combos = list(combinations(gloms_to_mix, 2))
    for x, y in glom_combos:
        # TODO also sort x and y alphabetically here?
        # (less important prob, since presumably won't be comparing this directly to any
        # real data, at least not without transforming to odor names that best
        # approximate these?)
        # TODO problem that i switched '/' delim to ' + '?
        mser = pd.Series(index=df.index.copy(), name=f'{x}-150 + {y}-150 @ 0', data=0.0)
        mser.loc[x] = 150.0
        mser.loc[y] = 150.0
        mser.name = (SYN_DIAG_PANEL, mser.name)
        series_list.append(mser)

    # make binary mixtures by combining the real diagnostic data for each of these
    # three odors (should target same glomeruli as above). again all pairwise combos.
    diag_subset = diags.loc[:, diags.columns.get_level_values('odor').isin(
        ('2h @ -6', 'farn @ -2', 'ma @ -7')
    )]
    # TODO delete (or add CLI flag to do this for yang, and still compare the current
    # diag_subset to KC data [in a way that doesn't break w/ -L? or warn it breaks
    # because of -L, if that's passed?])
    #diag_subset = diags.copy()
    #

    # TODO use elsewhere?
    ODOR_COLS: List[str] = ['panel', 'odor']

    diag_odors = diag_subset.columns.to_frame(index=False)
    assert list(diag_odors.columns) == ODOR_COLS, f'{diag_odors.columns.names=}'
    diag_odors['name'] = diag_odors.odor.map(parse_odor_name)
    diag_odors_at_multiple_concs = diag_odors.name.duplicated(keep=False)

    if not leave_concs_in_odors and diag_odors_at_multiple_concs.any():
        warn('taking mean across the following odors at multiple concentrations:\n'
            f'{diag_odors[diag_odors_at_multiple_concs].to_string(index=False)}\n'
            '...since -L/--leave-concs not passed'
        )
        diag_subset.columns = pd.MultiIndex.from_frame(diag_odors)
        diag_subset = diag_subset.groupby(level=['panel', 'name'], sort=False,
            axis='columns').mean().droplevel('name', axis='columns')

    DUMMY_CONC_FOR_MODEL: str = ' @ 0'
    # NOTE: going back to only removing the concentrations for mixtures (adding
    # DUMMY_CONC_FOR_MODEL suffix), and leaving concentrations of components until
    # *after* running/loading model outputs. otherwise, have more issues telling odors
    # apart in outputs (still would have that issue if `leave_concs_in_odors=False` and
    # we had [in one input to model] any mixtures where the combination of components
    # are not always the same concentrations])
    #
    # TODO TODO assert no duplicate odor strs in input to model, after
    # process_odor_str_for_model calls (wrap this fn with one processing a series /
    # index, and put those assertions in there? many calls wouldn't be able to use
    # that...)
    #
    # NOTE: will not affect behavior of `-L` (leave_concs_in_odors=True) case, where we
    # will mangle component concentrations either way, to be consistent w/ mix, e.g.
    # '2h @ -6' -> '2h-6 @ 0'
    ALWAYS_LEAVE_CONCS_FOR_COMPONENTS: bool = True
    def process_odor_str_for_model(odor: str, *, delim: str = '+',
        output_delim: str = ' + ', is_component: bool = False,
        always_leave_concs_for_components: bool = ALWAYS_LEAVE_CONCS_FOR_COMPONENTS,
        _recursing_from_mix_call: bool = False
        ) -> str:
        """
        Args:
            is_component: only used if True, to check input does not already have
                `delim='+'` present (e.g. as some validation panel odors might? in
                abbrevs too?)
        """
        if not leave_concs_in_odors:
            output_delim = output_delim.strip()

        n_delims = odor.count(delim)
        if is_component:
            assert n_delims == 0, (f'{is_component=} was set, but {odor} already '
                f'contained {delim=}, which should only be used to separate components '
                'of a mix. preprocess this odor name to remove this character.'
            )

        if n_delims == 0:
            if leave_concs_in_odors:
                # TODO share w/ this delim w/ hong2p.olf?
                conc_delim: str = ' @ '
                assert odor.count(conc_delim) <= 1, \
                    f'{odor=} had multiple {conc_delim=}'
                # e.g. 'ea @ -4' -> 'ea-4'
                odor = odor.replace(conc_delim, '')
                if not _recursing_from_mix_call:
                    odor += DUMMY_CONC_FOR_MODEL
                return odor
            else:
                if always_leave_concs_for_components and not _recursing_from_mix_call:
                    # and this case will leave the component odor strs as-is
                    # (e.g. '2h @ -6'), rather than mangling concs like sometimes done
                    # (or done for mix) (e.g. 'ea @ -4.2 + eb @ -3.5' ->
                    # 'ea-4.2+eb-3.5 @ 0') or stripping them
                    # (e.g. 'ea @ -4.2 + eb @ -3.5' -> 'ea+eb @ 0'
                    return odor
                else:
                    # TODO require_conc=True? (would have to allow to override of
                    # default i set in this scripts wrapper of olf.parse_odor_name)
                    return parse_odor_name(odor)
        else:
            assert not _recursing_from_mix_call, 'did not expect nested mixes'

        assert n_delims == 1, (f'expected only one {delim=}, separating components '
            'of a binary mixture. does a component name have delim in it? '
            'preprocess to remove it, if so. {odor=}'
        )
        comps = [process_odor_str_for_model(x.strip(), _recursing_from_mix_call=True)
            for x in odor.split(delim)
        ]
        unique_comps = set(comps)
        assert len(comps) == len(unique_comps), ('some components were duplicated, at '
            f'least after processing:\n{comps=}\n{unique_comps=}'
        )
        mix_str = output_delim.join(comps)
        # always need to add this suffix for mix, whether
        # leave_concs_in_odors=True/False, and never want to add to components (which
        # should only have conc stripped after running through model)
        return f'{mix_str}{DUMMY_CONC_FOR_MODEL}'


    odor2glom = {
        '2h @ -6': 'VM5d',
        'farn @ -2': 'DC3',
        'ma @ -7': 'DM4',
    }
    assert diag_subset.index.equals(df.index)

    new_panels = ('diag-binaries_max',)
    for panel in new_panels:
        # replacing 'glomeruli_diagnostics' panel w/ each of these new ones, so
        # single components will appear in all panel specific plots
        without_panel = diag_subset.droplevel('panel', axis='columns')

        assert without_panel.columns.name == 'odor'
        without_panel.columns = without_panel.columns.map(lambda x:
            # is_component=True just checks odor doesn't already contain the mix
            # component delim (='+'), otherwise would need to preprocess odor names to
            # remove it
            process_odor_str_for_model(x, is_component=True)
        )
        comp_df = addlevel(without_panel, 'panel', panel, axis='columns')
        # technically not just a list of series anymore... but should still work
        series_list.append(comp_df)

    step_desc = (
        'constructing in-silico binary mixtures from max ORN component responses'
    )
    if verbose:
        print(step_desc)

    for x, y in combinations(diag_subset.columns.get_level_values('odor'), 2):
        # would be easiest to process any names w/ + in them to exclude that, to
        # minimize changes to downstream code (there is at least one in validation panel
        # i think)
        assert not any('+' in o for o in (x, y)), ('+ must not be in odor names, as '
            'downstream code assumes that separates binary mixture components. '
            f'pre-process odor names to remove this character.\n{x=} {y=}'
        )

        # just to sort alphabetically on component names, to be consistent w/ other code
        # (mainly code loading and processing odors in yang's data, which uses olf fns
        # that do that)
        x, y = sorted([x, y])

        d1 = diag_subset.loc[:, (slice(None), x)].squeeze()
        d2 = diag_subset.loc[:, (slice(None), y)].squeeze()
        both = pd.concat([d1, d2], axis='columns', verify_integrity=True)
        both.columns.names = ODOR_COLS

        max_ser = both.max(axis='columns')
        max_ser.name = 'max'
        # TODO delete
        mean_ser = both.mean(axis='columns')
        mean_ser.name = 'mean'
        #

        max_zerod_ser = None
        if x in odor2glom and y in odor2glom:
            g1 = odor2glom[x]
            g2 = odor2glom[y]
            max_zerod_ser = max_ser.copy()
            max_zerod_ser[~max_zerod_ser.index.isin((g1, g2))] = 0
            max_zerod_ser.name = 'max, non-cognate gloms 0d'
        else:
            warn('could not construct max-rest0d series since both odors not in '
                'odor2glom. add odors for any missing diagnostics, if you care'
            )

        # TODO that hack actually necessary? test?
        # main function of this is a hack to add ' @ 0' when necessary, so some of the
        # string processing code inside modelling doesn't err (with concentrations set
        # elsewhere in string then, if would not be preserved otherwise). ideally we'd
        # just have mix_name=f'{x} + {y}'.
        ca = process_odor_str_for_model(x)
        cb = process_odor_str_for_model(y)
        # TODO share this delim w/ hong2p.olf?
        mix_name = process_odor_str_for_model(f'{x} + {y}')
        mix_suffix = get_odor_fname_suffix(f'{ca}_and_{cb}')

        fname_prefix = f'diags_vs_constructed-mixtures{mix_suffix}'
        constructed_mix_plot = model_root / f'{fname_prefix}.{plot_fmt}'
        if ignore_existing or not constructed_mix_plot.exists():

            constructed_sers = [max_ser, mean_ser]
            if max_zerod_ser is not None:
                constructed_sers.append(max_zerod_ser)

            to_plot = pd.concat([
                    both.droplevel('panel', axis='columns'),
                    mean_ser, max_ser, max_zerod_ser
                ], axis='columns', verify_integrity=True
            )
            plot_responses(to_plot, model_root, fname_prefix,
                cbar_label='est. ORN firing rate delta (Hz)'
            )
        else:
            warn(f'skipping creation of {constructed_mix_plot.name} b/c already '
                'existed. pass CLI arg -i/--ignore-existing to regenerate'
            )

        # first element of each of these should be in `new_panels` defined before this
        # loop (otherwise single components will not be added correctly)
        max_ser.name = ('diag-binaries_max', mix_name)
        series_list.append(max_ser)

    if verbose:
        print(f'done {step_desc}')

    test_df = pd.concat(series_list, axis='columns', verify_integrity=True)
    assert not test_df.isna().any().any()
    test_df.columns.names = ODOR_COLS

    # TODO TODO how different are thr and APL scale params if tuning on natmix_df
    # instead of megamat_df?
    # TODO TODO what about if we tune on one of the diag-subset dfs?

    natmix_df = natmix_orn_deltas()
    assert not natmix_df.isna().any().any()
    # TODO actually need to do any processing on odors in natmix_orn_deltas()?
    # (don't think so, currently, at least if not -L [and not currently processing
    # anything natmix_df or comparable ORN/KC data in -L case anyway])

    # NOTE: calling fillna(0) on test_df below, to remove the NaNs added because of
    # these index differences:
    # ipdb> natmix_df.index.difference(test_df.index)
    # Index(['DA1', 'DA4l', 'DA4m', 'V', 'VA1d', 'VA1v'], ...
    # ipdb> test_df.index.difference(natmix_df.index)
    # Index(['VA4'], dtype='object', name='glomerulus')
    #
    # this verify_integrity=True should already gaurantee no duplicate odors within an
    # panel
    test_df = pd.concat([test_df, natmix_df], axis='columns', verify_integrity=True)

    # some code below that currently assumes this? subset_second_to_odors_in_first may
    # currently assume that, but could be modified. maybe some other places? odor
    # sorting that tries to guess panel? other component / mix finding?
    test_odors = get_odors(test_df, axis='columns')
    if test_odors.duplicated().any():
        warn('when ignoring panel column index level, have some duplicates in test_df '
            'odors. some code may currently assume odor names are unique even across '
            'panels, but should fix if so.'
        )
    del test_odors

    if skip_panels is not None:
        test_df_panel = test_df.columns.get_level_values('panel')
        test_df_panelset = set(test_df_panel)
        assert skip_panels - test_df_panelset == set(), ('requested to skip (via -S CLI'
            ' arg) following panels not in current test_df model input:\n'
            f'{skip_panels - test_df_panelset=}'
        )

        panels_to_skip = test_df_panel.isin(skip_panels)
        if panels_to_skip.sum() > 0:

            msg = (f'skipping model panels {skip_panels} because of '
                f'`-S {skip_panels_str}`'
            )
            if skip_panels_str == SYN_DIAG_PANEL:
                msg += ' (default)'
            msg += ', with the following odors:\n'

            panel2odors = test_df.columns.to_frame().droplevel('odor')
            # TODO refactor this def?
            assert list(panel2odors.columns) == ['panel', 'odor'], \
                f'{panel2odors.columns=}'

            panel2odors = panel2odors.drop(columns='panel')
            for panel in skip_panels:
                msg += (
                    f'{panel=}\n{panel2odors.loc[panel].to_string(index=False)}\n\n'
                )

            msg = msg.strip()
            warn(msg)
            test_df = test_df.loc[:, ~panels_to_skip]

    # TODO want to infer_dtypes or anything after this? check dtypes before/after?
    test_df = test_df.fillna(0.0)
    test_df = test_df.sort_index().sort_index(axis='columns')
    del df

    # TODO rename this pretuned_panels or something? nontuned_panels?
    # this does NOT include megamat (or megamat-tuned)
    panels = list(test_df.columns.get_level_values('panel').unique())

    # saw 0.212 on some kiwi/control stuff (tuned on megamat)
    # now .2228 on something
    #response_rate_plot_max = 0.23
    # need at least this for comparison to real KC data
    # TODO switch between maxes depending on whether we are comparing to real KC
    # data or not ?
    # TODO delete? even still used? make lower?
    #response_rate_plot_max: float = 0.32
    response_rate_plot_max: float = 0.15

    # by this point, model_cols should all be present, and match those we would have
    # below, where these mixture-suppression sorted models are saved
    if not simplify_models:
        # TODO move this to module level, and subset to version below based on columns
        # we have (or based on has_connectome_apl(...)?)?
        model_cols = [PNKC_CLASS_COL, 'connectome_apl', 'source']
        assert model_cols == MODEL_COLS
    else:
        model_cols = [PNKC_CLASS_COL, 'source']

    # TODO use elsewhere too
    withinpanel_odor_cols = ['pair_dilution_factor', 'odor']
    odor_cols = PANEL_COLS + withinpanel_odor_cols

    model_order = None
    max_supp_models = None
    chosen_modeldirs = None
    model_ids = None
    # TODO handle case where this doesn't exist after loop (where model_order still
    # None)?
    if max_supp_models_only and mixsupp_order_parquet.exists():
        # comparing parameters across models sorted by mean natmix mixsupp)
        model_order = read_parquet(mixsupp_order_parquet)

        if simplify_models and 'connectome_apl' in model_order.columns:
            assert not model_order.connectome_apl.isna().any()
            # TODO (delete) ok, yea source is screwed up after this (ok i think it was
            # before too tho...) (i think source was always supposed  to be missing
            # '_True', it's just the model_dirname values what weren't. was there ever
            # actualy a problem, or was i just confusing the two?)
            assert model_order.index.names == [None]
            assert isinstance(model_order.index, pd.RangeIndex)
            # this was also droppping all the bouton cases, because those currently
            # happen to only support connectome-APL
            #
            # RangeIndex before reset_index() too, just re-numbering
            model_order = model_order[model_order.connectome_apl == False].reset_index(
                drop=True
            )
            model_order = model_order.drop(columns='connectome_apl')

            assert not model_order[PNKC_CLASS_COL].str.startswith('bouton').any(), (
                'would need to manually drop these, if not dropped as a side-effect '
                'of dropping connectome_apl != False above'
            )

        # TODO refactor to share w/ other def after loop?
        # will take a couple seconds as-is (when run w/ -f [the "full" model list])
        # TODO assert there are no duplicates instead? (len assertion below already
        # sort of doing that)
        model_ids = model_order[model_cols + ['model_dirname']].drop_duplicates()

        # TODO assert no change in length (w/ drop_duplicates() above)? why was i
        # calling drop_duplicates() above and not asserting something like this before?
        # are there actually duplicates to be dropped? model_dirname alone should
        # guarantee there are not, no?
        assert len(model_ids) == len(model_order)

        # this should just mean that we can use model_cols to align model_order and
        # model_roi_odor_df (I initially was not writing the order parquets with full
        # model_dirname)
        assert (
            len(model_ids[model_cols].drop_duplicates()) ==
            model_ids.model_dirname.nunique()
        )
        assert not model_ids.isna().any().any()
        model_ids = model_ids.set_index(model_cols, verify_integrity=True).squeeze()

        # NOTE: would need something other than .first() to get >1 in each group
        gb = model_order.groupby(first_for_each, sort=False)
        first = gb[mixsupp_col].first()
        last = gb[mixsupp_col].last()

        ordered_mixsupp = model_order[mixsupp_col]
        first_indices = np.searchsorted(ordered_mixsupp, first)
        assert np.array_equal(ordered_mixsupp.iloc[first_indices], first)

        model_order = model_order.set_index(model_cols, verify_integrity=True)

        print()
        print(f'using {order_by_mean_mixsupp_from=} as the response stat to define '
            'mixture suppression below'
        )
        print('lowest (most suppression) avg kiwi/control (binary+5comp) mix '
            f'suppression, for each {first_for_each} combo:\n{first.to_string()}'
        )
        print()
        print('will analyze just the models above with the most suppression, but for '
            'reference...'
        )
        print('highest (least suppression) avg kiwi/control (binary+5comp) mix '
            f'suppression, for each {first_for_each} combo:\n{last.to_string()}'
        )
        print()

        max_supp_models = set(model_ids.iloc[first_indices].values)
        warn(f'skipping all models except those in:\n{pformat(max_supp_models)}')

        chosen_modeldirs = model_ids.iloc[first_indices].reset_index(drop=True)

    # metadata in index, and column for each of:
    # responded,num_spikes,logistic_scaled_num_spikes
    model_roi_odor_dfs = []
    tuned_dfs = []
    failing_kws_and_tracebacks = []
    seen_model_strs = set()
    failed_model_strs = []
    # only populated if model_output_dirs (positiona) / -e / -x CLI args used to
    # restrict which model param combos we analyze
    skipped_model_strs = []
    # only populated if -o/--only-analyze-cache
    model_strs_without_cache = []
    warned_about_clustering = False
    for i, kws in enumerate(tqdm(model_tune_kws, unit='model (on all panels)')):
        # TODO only print this if verbose? or if we don't end up skipping this dir?
        #print(f'{kws=}')

        # TODO rename plot_dirname? explicitly pass it as dirname? don't think it
        # should be necessary
        #
        # asserting that this matches created dirname right after fit_and_plot_mb_model
        # call below
        model_str = format_model_params(kws)
        assert model_str not in seen_model_strs, f'already saw {model_str=}'
        seen_model_strs.add(model_str)

        # TODO keep a list of what we skipped?
        skip = False
        if max_supp_models is not None:
            if model_str not in max_supp_models:
                skip = True

        # TODO should exact also apply here?
        # TODO err if model_output_dirnames is None/[] and exact...=True?
        if (exclude_substrings is not None and
            any(s in model_str for s in exclude_substrings)):
            warn(f'skipping {model_str} because it contained an exclude substring')
            skip = True

        if model_output_dirnames is not None:
            if (not exact_model_dirnames and
                not any(s in model_str for s in model_output_dirnames)):

                warn(f'skipping {model_str} because it did not match any substring '
                    'in model_output_dirnames'
                )
                skip = True

            elif (exact_model_dirnames and
                not any(s == model_str for s in model_output_dirnames)):
                warn(f'skipping {model_str} because it did not match any element of '
                    'model_output_dirnames'
                )
                skip = True

        if skip:
            # NOTE: will not warn about stale model dirs in this case
            # (i.e. those that exist but aren't in list of current parameters)
            # will also only do that if full_model_params
            skipped_model_strs.append(model_str)
            continue

        if not full_model_params or verbose:
            print(model_str)

        # TODO delete eventually (after everyone using model updates and runs this code,
        # or manually moves stuff)
        old_tune_model_dir = model_root / model_str
        new_tune_model_dir = tune_root / model_str
        if old_tune_model_dir.exists():
            assert old_tune_model_dir.is_dir()
            assert not new_tune_model_dir.exists()
            warn(f'moving megamat-tuned dir {model_str} from old location under '
                f'{model_root.name} to under {model_root.name}/{tune_root.name}'
            )
            shutil.move(old_tune_model_dir, new_tune_model_dir)
            assert not old_tune_model_dir.is_dir()
            assert new_tune_model_dir.is_dir()
        #

        # TODO why is this seemingly not using LR cache in home? it's still tuning
        # on megamat, so it should be the same, no? (true?)
        # TODO TODO also check for dirs w/ params in diff order, for cache purposes?
        # probably better to normalize order before defining model_dirnames...
        try:
            tuned_params = fit_and_plot_mb_model(tune_root, orn_deltas=tune_df,
                try_cache=use_cache, cache_only=only_analyze_cache, quiet=quiet,
                return_dynamics=save_dynamics,
                # TODO change max_iters to 200?
                response_rate_plot_max=response_rate_plot_max, max_iters=100, **kws
            )
        except NoCachedModelOutputsError as err:
            if new_tune_model_dir.exists():
                # TODO maybe w/ flag to do/not do this?
                warn(f'deleting {new_tune_model_dir} that did not have minimum cache '
                    'requirements (probably error during creation)'
                )
                shutil.rmtree(new_tune_model_dir)

            if max_supp_models_only:
                # TODO also err if model_output_dirnames passed?
                raise IOError(f'{err}\n...and -o and -m both set! would skip if not -m')

            # TODO change error messages (between here and above)?
            if not full_model_params:
                raise IOError(f'{err}\nwould skip if -f')

            for p in panels:
                panel_dir = model_root / p
                matching_dir = model_str2matching_pretuned_dir(model_str, panel_dir)
                if matching_dir is not None:
                    raise RuntimeError(f'matching pre-tuned directory {matching_dir} '
                        'existed (and presumably had cached outputs? perhaps not?), '
                        f'despite no cached outputs under {new_tune_model_dir}. may '
                        'want to delete, or check you can regenerate'
                    )

            # we will already fail below (in other catch of NoCachedModelOutputsError),
            # if these cached outputs exist, but they don't exist for any pre-tuned
            # panel

            warn(f'{err}\n...and -o/--only-analyze-cache set. skipping!')
            model_strs_without_cache.append(model_str)
            continue

        except AssertionError as err:
            if new_tune_model_dir.exists():
                # TODO maybe w/ flag to do/not do this?
                warn(f'deleting {new_tune_model_dir} because encountered error during '
                    'call to populate it'
                )
                shutil.rmtree(new_tune_model_dir)

            msg = traceback.format_exc()
            # TODO print these at end (+ fix assertion issue and remove this, ideally)
            failing_kws_and_tracebacks.append((kws, msg))
            warn(f'modelling call failed with the following message:\n{msg}\n'
                'skipping for now!\n'
            )
            failed_model_strs.append(model_str)
            continue

        assert tuned_params['output_dir'] == model_str, (
            f"{tuned_params['output_dir']=} != {model_str=}"
        )

        thr_and_apl_kws = get_thr_and_APL_weights(tuned_params, kws)
        # TODO TODO is it a mistake that i'm getting a vector for wAPLKC for some
        # of these? i.e. prat-claws_True. model at least working properly?
        # just a formatting mistake?
        if verbose:
            # TODO say more, like that we tuned on megamat and where those outputs are?
            print(f'tuned thr and APL weights: {pformat(thr_and_apl_kws)}')

        tuned_model_output_dir = tune_root / tuned_params['output_dir']
        trs = read_parquet(tuned_model_output_dir / 'responses.parquet')
        tss = read_parquet(tuned_model_output_dir / 'spike_counts.parquet')

        wPNKC = read_parquet(tuned_model_output_dir / 'wPNKC.parquet')
        raw_wPNKC = wPNKC.copy()
        # TODO only store wPNKC (+ later save) for a unique set of wPNKC params?
        # (most model params don't affect wPNKC. way too cluttered when saving under
        # root, esp when full_model_params=True)
        save_kc_glom_counts: bool = False
        if save_kc_glom_counts:
            if kws.get('one_row_per_claw', False):
                wPNKC = wPNKC.groupby(KC_ID).sum()

                if kws.get('prat_boutons', False):
                    wPNKC = wPNKC.droplevel(
                        [x for x in wPNKC.columns.names if x != 'glomerulus'],
                        axis='columns'
                    )
                    wPNKC = wPNKC.groupby('glomerulus', axis='columns').sum()

            kc_glom_combo_counts = wPNKC[gloms_to_mix].value_counts().sort_index()
            kc_glom_combo_counts.name = 'n_kcs'

            to_csv(kc_glom_combo_counts,
                model_root / f'kc-glom-combo-counts_{model_str}.csv'
            )
            to_parquet(kc_glom_combo_counts,
                model_root / f'kc-glom-combo-counts_{model_str}.parquet'
            )

        # TODO TODO plot binarized version, just counting # KCs getting any amount of
        # input from each combo? or some kind of dist of total amount of input per
        # combo? (separate line for those that only get input from one?)
        # TODO best way to plot this? for uniform model, can get value_counts like:
        # ipdb> wPNKC[gloms_to_mix].value_counts().sort_index()
        # DM4  VM5d  DC3
        # 0.0  0.0   0.0    1223
        #            1.0     178
        #            2.0       9
        #            3.0       1
        #      1.0   0.0     153
        #            1.0      15
        #            2.0       3
        #      2.0   0.0       7
        # 1.0  0.0   0.0     186
        #            1.0      17
        #      1.0   0.0      17
        #            1.0       1
        #      2.0   0.0       2
        # 2.0  0.0   0.0      14
        #            1.0       1
        #      1.0   0.0       1
        # dtype: int64

        mean_num_spikes = addlevel(tss.mean(), 'model', model_str)
        mean_num_spikes.name = 'mean_num_spikes'

        mean_response_rate = addlevel(trs.mean(), 'model', model_str)
        mean_response_rate.name = 'mean_response_rate'

        stats = pd.concat([mean_num_spikes, mean_response_rate], axis='columns',
            verify_integrity=True
        )
        tuned_dfs.append(stats)

        # TODO one tqdm that updates for each iteration of this too? copy another script
        # i have that does that?
        for panel in panels:
            panel_dir = model_root / panel
            panel_dir.mkdir(exist_ok=True)
            panel_df = test_df.loc[:,test_df.columns.get_level_values('panel') == panel]

            try:
                params = fit_and_plot_mb_model(panel_dir, orn_deltas=panel_df,
                    try_cache=use_cache, cache_only=only_analyze_cache, quiet=quiet,
                    return_dynamics=save_dynamics,
                    response_rate_plot_max=response_rate_plot_max, **kws,
                    **thr_and_apl_kws
                )
            # NOTE: we are also NOT catching AssertionError like tuning above does
            except NoCachedModelOutputsError as err:
                raise RuntimeError(f'when running pre-tuned model on {panel=}:\n{err}\n'
                    '...and was successful when tuning on megamat!'
                )

            model_output_dir = panel_dir / params['output_dir']
            rs = read_parquet(model_output_dir / 'responses.parquet')
            ss = read_parquet(model_output_dir / 'spike_counts.parquet')

            curr_odors = panel_df.columns.droplevel('panel')
            cached_odors = rs.columns
            if not curr_odors.equals(cached_odors):
                # If False, odors must match exactly. If True, as long as
                # `olf.parse_odor_name(x, require_conc=False)` (applied to each
                # sequence) provides the same set of (non-duplicated!) odor names,
                # continue (and replace cached odor strings with current ones (including
                # any concentrations)
                allow_matching_odor_names_only: bool = True

                err_msg = ('cached model odors different from current! current '
                    'vs cached probably run with a different choice of whether to pass '
                    '-L CLI flag! Re-run without -u or -o args, to regenerate model '
                    f'responses.\n\n{curr_odors=}\n{cached_odors=}'
                )
                # TODO TODO actually strip the conc parts like in -L's '2h-6' or
                # '2h-6 + ma-7'? (would have to check no '-' in odor names earlier, and
                # maybe exclude syn diag panel?)
                #
                # otherwise, this code was mainly to fix a temporary mistake i made,
                # where i was also replacing component concs withi ' @ 0' in
                # leave_concs_in_odors=False case, which i do not want (in that case,
                # concs should be stripped after, not before, running model)
                curr_names = [parse_odor_name(x) for x in curr_odors]
                cached_names = [parse_odor_name(x) for x in cached_odors]
                if allow_matching_odor_names_only and curr_names == cached_names:
                    warn(f'{err_msg}\n...but replacing cached_odor with curr_odors, '
                        'since allow_matching_odor_names_only=True and name order in '
                        'each matched'
                    )
                else:
                    if allow_matching_odor_names_only:
                        err_msg += (f'\n...and current and cached name orders did not '
                            f'match either!\n{curr_names=}\n{cached_names=}'
                        )
                    raise RuntimeError(err_msg)

            wPNKC2 = read_parquet(model_output_dir / 'wPNKC.parquet')
            assert raw_wPNKC.equals(wPNKC2), 'wPNKC should not change across tuned/not'

            def add_metadata(data):
                data = addlevel(data, 'model', model_str)
                return addlevel(data, 'panel', panel)

            assert len(ss.squeeze().shape) == 2
            assert not ss.isna().any().any()
            assert ss.columns.name == 'odor'
            size_before = ss.size
            ss = ss.stack().rename('num_spikes')
            assert len(ss) == size_before
            assert isinstance(ss, pd.Series)
            assert not ss.isna().any()
            ss = add_metadata(ss)

            assert len(rs.squeeze().shape) == 2
            assert not rs.isna().any().any()
            assert rs.columns.name == 'odor'
            size_before = rs.size
            rs = rs.stack().rename('responded')
            assert len(rs) == size_before
            assert isinstance(rs, pd.Series)
            assert not rs.isna().any()
            rs = add_metadata(rs)

            # from some output of:
            # ```
            # ./analysis.py -r /mnt/d0/PNAPL_stepping/extra_panels/ pn-claw-to-apl_True,pn-claw-to-apl_False -x -m
            # ```
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=1.69, x0=2.10, L=2.98
            # final cost: 0.210
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.08
            # n_spikes=1: 0.40
            # n_spikes=2: 1.36
            # n_spikes=3: 2.45
            # n_spikes=4: 2.87
            # n_spikes=5: 2.96
            # ...
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=1.47, x0=2.28, L=2.94
            # final cost: 0.213
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.10
            # n_spikes=1: 0.39
            # n_spikes=2: 1.18
            # n_spikes=3: 2.19
            # n_spikes=4: 2.72
            # n_spikes=5: 2.89
            #
            # from some output of: (on old, outdated outputs committed in natmix_data/)
            # ```
            # ./analysis.py -m
            # ```
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=2.10, x0=1.78, L=2.97
            # final cost: 0.220
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.07
            # n_spikes=1: 0.48
            # n_spikes=2: 1.82
            # n_spikes=3: 2.76
            # n_spikes=4: 2.94
            # n_spikes=5: 2.96
            # ...
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=2.16, x0=1.78, L=3.07
            # final cost: 0.219
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.06
            # n_spikes=1: 0.48
            # n_spikes=2: 1.89
            # n_spikes=3: 2.86
            # n_spikes=4: 3.04
            # n_spikes=5: 3.07
            # ...
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=2.01, x0=1.81, L=2.78
            # final cost: 0.217
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.07
            # n_spikes=1: 0.46
            # n_spikes=2: 1.66
            # n_spikes=3: 2.55
            # n_spikes=4: 2.74
            # n_spikes=5: 2.77
            # ...
            # input params: drop_silent_frac=False, method=logistic, metric=wasserstein,
            #   fit_L=True, kc_quantile_for_L=1
            # fit params: k=2.15, x0=1.79, L=3.13
            # final cost: 0.219
            # # spikes -> output of logistic scaling fn:
            # n_spikes=0: 0.07
            # n_spikes=1: 0.49
            # n_spikes=2: 1.92
            # n_spikes=3: 2.92
            # n_spikes=4: 3.11
            # n_spikes=5: 3.13
            #
            # these seem like reasonable enough (round) defaults, from looking at
            # some fit outputs i was getting (in comments above)
            logistic_scaled_num_spikes = logistic(ss, **logistic_scaling_params)

            logistic_scaled_num_spikes = logistic_scaled_num_spikes.rename(
                'logistic_scaled_num_spikes'
            )

            # TODO don't pass under certain circumstances? don't pass ever?
            kc_spont_in = read_parquet(model_output_dir / 'kc_spont_in.parquet')

            nonsilent_cells: Optional[pd.Index] = None
            to_cluster_list = [ss, logistic_scaled_num_spikes]
            stat2to_cluster_nosilent = dict()
            keep_binary: bool = True
            for j, to_cluster in enumerate(to_cluster_list):
                df = to_cluster.unstack(['panel', 'odor']).droplevel('model')

                # TODO TODO TODO wait, so was model run with mix dilutions? (oh, yea.
                # nice). actually analyze?
                warn_ = ((i == len(model_tune_kws) - 1) and (panel == panels[-1]) and
                    (j == len(to_cluster_list) - 1)
                )
                # do currently need the internal .T to get this fn to work, and the
                # external one for drop_silent_model_cells to currently work
                df = drop_binaries_mixdilutions_and_pfo(df.T, keep_binary=keep_binary,
                    warn_=warn_
                ).T
                if warn_:
                    warn('also dropped odors mentioned above (or equivalent set per '
                        'panel) in all *pre-hierarchichal-clustering only* model data '
                        'processing before this too. only warning on last model '
                        'directory'
                    )

                n_total = len(df)

                stat = to_cluster.name
                # so loop must hit num_spikes first, so it can use that to define silent
                # cells for rest (since logistic scaling doesn't necessarily sent 0->0
                # now, and 0 is what is counted as a non-response in
                # drop_silent_model_cells)
                if stat == 'num_spikes':
                    with_silent = df.copy()
                    df = drop_silent_model_cells(df)
                    assert with_silent.columns.equals(df.columns)
                    del with_silent
                    nonsilent_cells = df.index.copy()
                else:
                    assert nonsilent_cells is not None
                    df = df.loc[nonsilent_cells]

                stat2to_cluster_nosilent[stat] = df

            # TODO assert this is the same on each iteration? it should be
            assert len(set(x.shape for x in to_cluster_list)) == 1
            assert len(set(x.shape for x in stat2to_cluster_nosilent.values())) == 1
            # can use the values from last iteration, b/c assertion their shapes are all
            # the same
            n_silent = n_total - len(df)
            assert n_silent > 0, 'expected at least some silent cells'
            title = f'{model_str}\ndropped {n_silent}/{n_total} silent model cells'

            stat_order = ['num_spikes', 'logistic_scaled_num_spikes']
            assert set(stat_order) == set(stat2to_cluster_nosilent.keys())

            cluster_on_logistic_scaled = True
            if cluster_on_logistic_scaled:
                # reversing list, so logistic scaled will be encountered first, and thus
                # the one clustered on
                stat_order = stat_order[::-1]

            row_linkage = None
            for stat in stat_order:
                df = stat2to_cluster_nosilent[stat]
                df = sort_odors(df.T)
                df = df.droplevel('panel')

                curr_title = str(title)
                if (stat == 'logistic_scaled_num_spikes' and
                    not cluster_on_logistic_scaled):

                    curr_title += '\nclustered on raw spike counts'

                elif stat == 'num_spikes' and cluster_on_logistic_scaled:
                    curr_title += '\nclustered on logistic-scaled spike counts'

                fname_suffix = stat.replace("_", "-")
                if stat == 'logistic_scaled_num_spikes' or cluster_on_logistic_scaled:
                    # both '' if default, or str with logistic params otherwise
                    fname_suffix += logistic_scaling_fname_part
                    curr_title += logistic_scaling_title_str

                if not cluster_on_logistic_scaled:
                    # adding suffix in this case, since it seems to be worse
                    fname_suffix += '_clustered-on-raw-spikes'

                if keep_binary:
                    fname_suffix += '_with-binary'

                if unrestricted_full_model_params:
                    if not warned_about_clustering:
                        warn('not running hierarchichal clustering, since unrestricted '
                            '-f'
                        )
                        warned_about_clustering = True
                    continue

                if not ignore_existing:
                    hierarch_clust_plot_path = get_hierarch_clust_plot_path(
                        model_output_dir, fname_suffix
                    )
                    exists = get_hierarch_clust_plot_exists_and_warn_if_so(
                        hierarch_clust_plot_path, show_n_dirs=2,
                        warn_=(not unrestricted_full_model_params)
                    )
                    # TODO delete? now that i'm skipping in -f case anyway? maybe delete
                    # all of this get_hierarch_clust_plot_exists_and_warn_if_so code?
                    if exists:
                        if (unrestricted_full_model_params and
                            not warned_about_clustering):
                            warn('one (or more) model hierarchichal clustering plots '
                                'already existed and not being regenerated. more '
                                'warnings suppressed since -f and no other parameters '
                                'restricted set of models analyzed. pass -i to remake.'
                            )
                            warned_about_clustering = True
                        continue

                # TODO want logistic scaling in cbar label, if already in title? in
                # title here?
                cbar_label = get_model_stat_label(stat, logistic_scaling_title_str)
                # TODO plot one version of these with and one without row_colors?
                # currently defaulting all to no row colors
                # TODO flag to cluster each independently? (as before)
                ret = plot_hierarch_clustered_rois(model_output_dir, df,
                    fname_suffix, ignore_existing=ignore_existing, warn_=False,
                    # when row_linkage=None (on first iteration, for 'num_spikes'
                    # currently, or whichever i put first), rows will actually be
                    # clustered
                    return_linkages=(row_linkage is None), row_linkage=row_linkage,
                    wPNKC=wPNKC, kc_spont_in=kc_spont_in, cbar_label=cbar_label,
                    title=curr_title
                )

                # ret will always be None if ignore_existing, but it doesn't matter, b/c
                # nothing should be generated (assuming if one thing is a cache hit, all
                # will be, for my sanity now)
                # also assuming we don't care to check on the fresh run, when they are
                # generated regardless of ignore_existing.
                if ignore_existing and row_linkage is None:
                    assert ret is not None

                if ret is not None:
                    # TODO assert row_linkage is None here? (before def below)
                    row_linkage = ret

            del stat, title

            model_roi_odor_sers = [ss, rs, logistic_scaled_num_spikes]
            m0_index = model_roi_odor_sers[0].index
            assert all(x.index.equals(m0_index)
                for x in model_roi_odor_sers[1:]
            )
            model_roi_odor_df = pd.concat(model_roi_odor_sers, axis='columns',
                verify_integrity=True
            )
            assert model_roi_odor_df.index.equals(m0_index)
            assert len(model_roi_odor_df.columns) == len(model_roi_odor_sers)
            model_roi_odor_dfs.append(model_roi_odor_df)

        if not full_model_params or verbose:
            print()

    if not full_model_params:
        warn('will not summarize model dirs present in panel dirs but absent from '
            'current model params, because not -f/--full-model-params'
        )
    else:
        # TODO TODO use / delete these?
        unused_model_strs = set(skipped_model_strs) | set(model_strs_without_cache)
        # NOTE: "seen" here just means it's part of *tuned* directory names that would
        # be produced by current FULL_MODEL_KW_LIST (since we already can assume -f
        # here), not that it wasn't "skipped" or that it was re-run or anything else
        used_model_strs = seen_model_strs - unused_model_strs
        #

        panel_dirs = [tune_root]
        for panel in panels:
            panel_dirs.append(model_root / panel)

        for panel_dir in panel_dirs:
            # for now, just assuming all directories are complete model dirs (could use
            # some fn for that, if i have/make one)
            panel_model_dirs = [x for x in panel_dir.glob('*/') if x.is_dir()]

            # TODO TODO -> also use this fn (below loop) to calculate stale dirs
            matching_dir = model_str2matching_pretuned_dir(model_str, panel_dir)

            if panel_dir == tune_root:
                stale_dirs = [
                    x for x in panel_model_dirs if x.name not in seen_model_strs
                ]
            else:
                seen_dirs = set(panel_model_dirs)
                expected_pretuned_dirs = {model_str2matching_pretuned_dir(m, panel_dir)
                    for m in seen_model_strs
                }
                # TODO TODO warn about any of these? or just ignore any that we
                # aren't currently analyzing? was it actually an error just in running
                # the pre-tuned model (for any of these)? (or were they just not
                # generated for some reason, and only older for megamat-tuned)?
                # TODO assert same set in each panel? matter?
                nomatch = {
                    m for m in seen_model_strs
                    if model_str2matching_pretuned_dir(m, panel_dir) is None
                }
                # TODO delete
                #print()
                #print('nomatch:')
                #pprint(nomatch)
                #
                # TODO delete. not true apparently (re-evaluate...)
                # we should be able to assume there are no None here, b/c above we
                # err if any pre-tuned panels don't have cached outputs that tuning
                # panel does (what about in cases where we aren't strictly using cache
                # outputs tho? still gauranteed?)
                #assert None not in expected_pretuned_dirs
                #
                # works whether or not None is already in there
                expected_pretuned_dirs -= {None}

                stale_dirs = list(seen_dirs - expected_pretuned_dirs)
                # TODO TODO seems i want to fail earlier if any of these directories
                # have multiple directories sharing same prefix before tune_params_delim
                # (may have manually deleted all those cases now, but might happen
                # again)

            if len(stale_dirs) == 0:
                continue

            # just for nicer formatting below
            stale_dirs = [str(x) for x in stale_dirs]

            warn(f'in panel dir {panel_dir.name}, had {len(stale_dirs)} old model '
                'directories that are not referenced in current FULL_MODEL_KW_LIST:\n'
                f'{pformat(stale_dirs)}'
            )

    if len(model_roi_odor_dfs) == 0:
        raise RuntimeError('no model data actually loaded / returned! maybe filtering '
            'skipping too much in loop? check CLI args?'
        )

    stat_names = model_roi_odor_dfs[0].columns
    for x in model_roi_odor_dfs[1:]:
        assert x.columns.equals(stat_names)
    stat_names = list(stat_names)

    if full_model_params:
        print('concatenating model dfs...', end='', flush=True)

    # need the .reset_index() since KC_TYPE not present for uniform model (but is for
    # all others), so can't concat based on indices, which produces bad output when not
    # all have same level names
    # TODO TODO add CLI option to load this instead of computing via loop above (esp to
    # shortcut testing stuff in -f case) (would need to cache w/ CLI options in name,
    # and no caching or loading cache if any str based inclusion/exclusion)
    model_roi_odor_df = pd.concat([x.reset_index() for x in model_roi_odor_dfs])
    assert not model_roi_odor_df[['panel', 'model', 'odor', 'kc_id']].duplicated(
        ).any()
    del model_roi_odor_dfs

    if chosen_modeldirs is None and not unrestricted_full_model_params:
        # TODO move all this code below, and define from unique_model_ids (somewhat
        # duplicated effort...)
        chosen_modeldirs = model_roi_odor_df.model.drop_duplicates().rename(
            'model_dirname').reset_index(drop=True)

    if chosen_modeldirs is not None:
        chosen_modeldirs_prefix = 'chosen_modeldirs'
        # TODO TODO TODO why does
        # chosen_modeldirs_no-connectomeAPL-wd20-or-boutons_max-mixsupp-only.csv
        # have a 'weight_divisor....' line?
        # chosen_modeldirs_no-connectomeAPL-wd20-or-boutons.csv does not, so that seems
        # to be working at least

        # don't want the extra choice-specific suffix here, as want to be able to always
        # use this link when passing to e.g. natmix_data/analysis.py
        chosen_modeldirs_link = model_root / f'last_{chosen_modeldirs_prefix}.csv'

        if simplify_models:
            chosen_modeldirs_prefix += f'_{simplify_models_fname_part}'

        if max_supp_models_only:
            chosen_modeldirs_prefix += f'_{max_supp_models_only_fname_part}'

        if old_model_kws_fname_part:
            chosen_modeldirs_prefix += f'_{old_model_kws_fname_part}'

        chosen_modeldirs_fname = f'{chosen_modeldirs_prefix}.csv'
        chosen_modeldirs_csv = model_root / chosen_modeldirs_fname

        # with index=False & header=False, there will only be one line per
        # model_dirname, and no commas / column names / anything else
        to_csv(chosen_modeldirs, chosen_modeldirs_csv, index=False, header=False)

        # so this is how it can be loaded
        assert pd.read_csv(chosen_modeldirs_csv, header=None).squeeze().rename(
            'model_dirnames').equals(chosen_modeldirs)

        symlink(chosen_modeldirs_csv, chosen_modeldirs_link, replace=True)


    unique_model_ids = model_roi_odor_df.model.unique()
    if full_model_params:
        print('done', flush=True)

        skip_str = ''
        if len(skipped_model_strs) > 0:
            n_nonskipped_model_strs = len(FULL_MODEL_KW_LIST) - len(skipped_model_strs)
            # TODO count simplify_models=True filtering as "skipped", for counting here?
            # (just move to skipping in loop, rather than pre-filtering?)
            skip_str = f' (/{n_nonskipped_model_strs} non-skipped)'

        warn(f'{len(unique_model_ids)}/{len(FULL_MODEL_KW_LIST)}{skip_str} model params'
            ' successfully run'
        )

    def param_str_parts(model_str: str) -> frozenset:
        parts = model_str.split('__')
        assert len(parts) == len(set(parts))
        return frozenset(parts)

    def compare_missing_to_present(missing_strs: List[str], desc: str = 'missing'
        ) -> None:

        printed_header = False
        working_only_intersection2missing_dirs = defaultdict(list)
        working_only_intersection2working_dirs = defaultdict(list)
        # TODO delete if not necessary
        #missing_only_intersection2missing_dirs = defaultdict(list)
        #missing_only_intersection2working_dirs = defaultdict(list)
        #
        missing_parts_list = []
        for model_str in missing_strs:
            missing_parts = param_str_parts(model_str)
            missing_parts_list.append(missing_parts)

            largest_overlap = frozenset()
            # in case there are multiple distinct sets w/ same amount of overlap.
            # not sure if that will/could happen.
            n_overlapping2overlap_set = defaultdict(set)
            n_overlapping2working_model_strs = defaultdict(list)
            n_overlapping2working_parts = defaultdict(list)

            for working_model_str in unique_model_ids:
                assert working_model_str != model_str, 'model both working and not...'

                working_parts = param_str_parts(working_model_str)

                overlap = missing_parts & working_parts
                if len(overlap) >= len(largest_overlap):
                    largest_overlap = overlap

                if len(overlap) > 0:
                    n_overlapping2overlap_set[len(overlap)].add(overlap)
                    # TODO union/intersection across all these later? or
                    # union/intersection of differnces from missing_parts, in both
                    # directions?
                    n_overlapping2working_parts[len(overlap)].append(
                        working_parts
                    )
                    n_overlapping2working_model_strs[len(overlap)].append(
                        working_model_str
                    )

            if len(largest_overlap) == 0:
                continue

            # converting from frozenset to set just for nicer printing
            largest_overlaps = [
                set(x) for x in n_overlapping2overlap_set[len(largest_overlap)]
            ]
            # not sure there will ever be any cases when this len is NOT 1.
            # could assert, but don't really require it.
            if len(largest_overlaps) == 1:
                largest_overlaps = largest_overlaps[0]
                largest_overlaps_str = str(largest_overlaps)
            else:
                largest_overlaps_str = pformat(largest_overlaps)
            del largest_overlaps

            if not printed_header:
                # TODO update for accuracy/delete
                #print('missing model ID -> largest param set overlap(s) across working '
                #    'model IDs:'
                #)
                printed_header = True

            working_parts_list = n_overlapping2working_parts[len(largest_overlap)]
            # TODO pick some subset of these?
            working_only_union = set()
            missing_only_union = set()
            working_only_intersection = None
            missing_only_intersection = None
            for working_parts in working_parts_list:
                missing_only = missing_parts - working_parts
                working_only = working_parts - missing_parts
                working_only_union |= set(working_only)
                missing_only_union |= set(missing_only)

                if working_only_intersection is None:
                    working_only_intersection = set(working_only)
                else:
                    working_only_intersection &= set(working_only)

                if missing_only_intersection is None:
                    missing_only_intersection = set(missing_only)
                else:
                    missing_only_intersection &= set(missing_only)

            # TODO TODO why are these always empty? bug? or do i just not have any
            # parameters that always fail?
            assert len(missing_only_intersection) == 0, f'{missing_only_intersection=}'
            assert len(missing_only_union) == 0, f'{missing_only_union=}'

            # NOTE: this is NOT largest_overlapS, which was intermediate for printing
            working_model_strs = n_overlapping2working_model_strs[len(largest_overlap)]

            working_only_intersection2missing_dirs[frozenset(working_only_intersection)
                ].append(model_str)
            working_only_intersection2working_dirs[frozenset(working_only_intersection)
                ].extend(working_model_strs)

            # TODO delete if not necessary
            #missing_only_intersection2missing_dirs[frozenset(missing_only_intersection)
            #    ].append(model_str)
            #missing_only_intersection2working_dirs[frozenset(missing_only_intersection)
            #    ].extend(working_model_strs)
            #

            # TODO TODO try grouping by working/missing_only_intersection, and listing
            # working and non-working dirs for each of those? how many distinct such
            # sets?
            if verbose:
                # TODO hide this even if verbose? too much...
                print(f'{model_str}:\nlargest overlap(s): {largest_overlaps_str}')
                print('working models with this overlap:\n'
                    f'{pformat(working_model_strs)}'
                )
                # TODO delete (/pick subset)
                #print()
                #print(f'{working_only_union=}')
                #print(f'{working_only_intersection=}')
                ##print(f'{missing_only_union=}')
                ##print(f'{missing_only_intersection=}')
                #
                print()

        # TODO delete

        if verbose:
            print()

        # TODO does summing len(missing_dirs) across all these give us all the
        # missing dirs (and they are unique across lists, right?)
        seen_missing_dirs = set()
        for working_only, missing_dirs in working_only_intersection2missing_dirs.items():
            assert not any(x in seen_missing_dirs for x in missing_dirs)
            assert len(missing_dirs) == len(set(missing_dirs))
            seen_missing_dirs.update(missing_dirs)
            if not verbose:
                continue

            # TODO TODO TODO this working? being reached? (or are extra params to fix
            # thr and APL weight scale(s) currently interfering w/ expected function?)

            # TODO reword "some working versions"?
            print(f'params shared by some working versions: {set(working_only)}')
            # TODO TODO print just the common subset of below instead? would reveal
            # prat_boutons is the common denominator for connectome APL case...
            # TODO TODO warn instead?
            print(f'missing model directories:')
            # TODO TODO print d.name instead (or model_str, which should also exclude
            # the fixed thr / APL weight suffix in the pretuned [e.g. kiwi/control]
            # cases)
            for d in missing_dirs:
                print(d)
            # TODO are there any failed directories that have some of the
            # working_only? (prob only care if they have *all*?)
            # TODO and are there any working directories that have any params
            # unique to the failing side?
            print()

        # TODO TODO fix (still an issue? was it just a matter of skipped stuff?)
        # ipdb> len(seen_missing_dirs)
        # 44
        # ipdb> len(unique_model_ids)
        # 91
        # ipdb> len(model_tune_kws)
        # 140
        # ipdb> len(seen_missing_dirs) + len(unique_model_ids)
        # 135
        # (was when i was manually excluding at least 2 dirs, but maybe more matched
        # those substrings)
        assert (
            len(seen_missing_dirs) + len(unique_model_ids) + len(skipped_model_strs) ==
            len(model_tune_kws)
        )
        # TODO delete/use
        #working_parts_list = [param_str_parts(x) for x in unique_model_ids]
        # TODO TODO also print any missing dirs that contain these params (will there be
        # any?)

        # TODO delete
        #print()
        #print('missing_only_intersection2missing_dirs:')
        #pprint(dict(missing_only_intersection2missing_dirs))
        #

        # TODO maybe something simple (like set diff) makes sense within each
        # model_pnkc_class?


    # only populated if model_output_dirs (positiona) / -e / -x CLI args used to
    # restrict which model param combos we analyze
    if len(skipped_model_strs) > 0:
        # TODO only summarize (/print at all) if not -m?
        # TODO also say that all were still in FULL_MODEL_KW_LIST?
        warn(f'skipped_model_strs (b/c CLI args):\n{pformat(skipped_model_strs)}\n')

    if len(failed_model_strs) > 0:
        warn('failed_model_strs (AssertionError during this run):\n'
            f'{pformat(failed_model_strs)}\n'
        )
        compare_missing_to_present(failed_model_strs, 'failed')

    # only populated if -o
    if len(model_strs_without_cache) > 0:
        # TODO populate this even without -o?
        warn(f'{len(model_strs_without_cache)} model_strs_without_cache (model probably'
            ' failed to run or converge, potentially because of a bug or some '
            'unsupported combination of parameters) (only populated b/c '
            f'-o/--only-analyze-cache):\n{pformat(model_strs_without_cache)}\n'
        )
        compare_missing_to_present(model_strs_without_cache, 'without cache')
    # TODO also warn about any cached models that we would not be currently generating
    # (so i can delete them, e.g. before running natmix_data/analysis.py on these model
    # outputs) (i guess that's what model_strs_without_cache is? rename?)

    for_n_variants = unique_model_ids
    if model_order is not None:
        # so that plotting using this palette doesn't fail, since the #-variants part of
        # strings will be from previous run that defined the model_order parquet, with
        # counts out of the full model params
        for_n_variants = model_order.model_dirname

    pnkc2n_models = Counter([model_pnkc_class(x) for x in for_n_variants])
    # TODO TODO check this works now that i also think i don't want the suffix in
    # the max_supp_models_only=True case (including w/ saved model_order that either
    # does or does not still have it)
    # TODO TODO also exclude in `not full_model_params` case?
    if not (full_model_params and not max_supp_models_only):
        pnkc2n_models = None

    # max_supp_models=True is the only case model_order will ever be non-None
    if simplify_models and max_supp_models_only:
        # TODO TODO move below downstream def of model_ids (if model_ids is None
        # here), and still fix counts there (matter? counts only important if -f, right?
        # really care if `-f -M` [and NOT -m])

        # should have levels [PNKC_CLASS_COL, 'source']
        # (or at least that's what it seems to have now)
        for_index = model_order.index.to_frame(index=False)

        # fixing the (now wrong) counts at the end of the model_pnkc_class strs, since
        # we dropped some of the variants above. for example:
        # 'claw (54 variants)' -> 'claw (26 variants)'
        classes_with_fixed_counts = model_order.model_dirname.map(
            lambda x: model_pnkc_class(x, pnkc2n_models=pnkc2n_models)
        )
        # TODO assert they all start with same splitting on ' ('? (as before redefing)
        # (not a huge concern... delete)
        for_index[PNKC_CLASS_COL] = classes_with_fixed_counts.reset_index(drop=True)

        new_index = pd.MultiIndex.from_frame(for_index)
        assert model_order.index.equals(model_ids.index)
        model_order.index = new_index
        model_ids.index = new_index

    model_roi_odor_df[PNKC_CLASS_COL] = model_roi_odor_df.model.map(
        lambda x: model_pnkc_class(x, pnkc2n_models=pnkc2n_models)
    )
    model_roi_odor_df['model_dirname'] = model_roi_odor_df.model.copy()
    model_roi_odor_df.model = model_roi_odor_df.model.apply(abbrev_model_id)

    # can keep this if leave_concs_in_odors=True, because all components and mixes
    # should have concentrations mangled in that case, so they should have the conc in
    # the "name" part (e.g. '2h-6 @ 0' instead of '2h @ -6' or '2h-6 + ma-4 @ 0'
    model_roi_odor_df.odor = model_roi_odor_df.odor.map(parse_odor_name)

    # TODO (delete? already have in natmix_data/analysis.py i think. that using a
    # function shared here?) print # of spikes -> values, for first few # of spikes, and
    # print max before and after (after should approach L, right?)

    model_roi_odor_df = model_roi_odor_df.rename(columns={
        'model': source_col,
        'kc_id': 'roi',
    })

    # just since old df didn't have it. could keep?
    model_roi_odor_df = model_roi_odor_df.drop(columns='kc_type')
    # would need to subset to exclude KC_TYPE if we were not dropping above
    assert not model_roi_odor_df.isna().any().any()

    id_cols = ['panel', source_col, PNKC_CLASS_COL, 'model_dirname', 'odor', 'roi']
    assert set(id_cols) | set(stat_names) == set(model_roi_odor_df.columns)
    tidy = pd.melt(model_roi_odor_df, id_vars=id_cols, value_vars=stat_names,
        var_name='stat'
    )
    assert tidy['value'].size == model_roi_odor_df[stat_names].size
    assert not tidy.isna().any().any()
    model_roi_odor_df = tidy
    del tidy

    # TODO (delete?) also handle this from per-roi stuff, like all other outputs now
    # assuming only one panel ('megamat'), so it's ok this doesn't have a panel column
    tdf = pd.concat(tuned_dfs, verify_integrity=True)
    tdf = tdf.reset_index()
    tdf[PNKC_CLASS_COL] = tdf.model.map(
        lambda x: model_pnkc_class(x, pnkc2n_models=pnkc2n_models)
    )
    tdf.model = tdf.model.apply(abbrev_model_id)
    tdf.odor = tdf.odor.map(parse_odor_name)
    tdf = pd.melt(tdf, id_vars=['model', PNKC_CLASS_COL, 'odor'],
        value_vars=['mean_num_spikes', 'mean_response_rate'], var_name='stat'
    )
    tdf = tdf.rename(columns={'model': source_col})

    if not simplify_models:
        tdf['connectome_apl'] = tdf[source_col].str.contains('_connectome-APL')
        tdf[source_col] = tdf[source_col].str.replace('_connectome-APL', '')
    # TODO assert no '_connectome-APL' in any of the strs if simplify_models=True?

    # TODO (delete?) print tdf (/ use to set / check ylim below)

    model_roi_odor_df = model_roi_odor_df[~(
        model_roi_odor_df.odor.str.contains('mix-') |
        model_roi_odor_df.odor.str.contains('(air mix)', regex=False
    ))].copy()

    model_roi_odor_df.odor = model_roi_odor_df.odor.replace(
        {'kmix0': 'kmix', 'cmix0': 'cmix'}
    )

    # NOTE: sorting column index of yang_[bin_]df here did not fix odor order in
    # combined model + KC plots, as subsequent processing seemed to screw up order.
    # now doing similar sort_values call in get_yang_panel_means fn.
    model_roi_odor_df = model_roi_odor_df.sort_values(by='odor', kind='stable',
        key=odor_sort_fn
    )
    model_roi_odor_df = model_roi_odor_df.reset_index(drop=True)

    if not simplify_models:
        model_roi_odor_df['connectome_apl'] = model_roi_odor_df[source_col
            ].str.contains('_connectome-APL')

        model_roi_odor_df[source_col] = model_roi_odor_df[source_col].str.replace(
            '_connectome-APL', ''
        )

    unique_model_pnkc_classes = model_roi_odor_df[PNKC_CLASS_COL].unique()
    assert 'KCs' not in unique_model_pnkc_classes, \
        f'{unique_model_pnkc_classes=}'
    source_palette = pnkc_classes2source_palette(unique_model_pnkc_classes)

    # TODO need to check if we actually load KC data? prob doesn't matter...
    source_palette['KCs'] = kc_color

    # TODO could use tab: red/purple/cyan/gray for one instead?
    # TODO like cyan?
    source_palette['ORNs'] = 'tab:cyan'
    #source_palette['ORNs'] = 'tab:brown'
    # TODO try for orn? didn't like cause ugly i think, but want something distinct from
    # KC ='m', and not sure any of purple/brown/gray good for that
    #source_palette['EAG'] = 'tab:olive'
    source_palette['EAG'] = 'tab:gray'

    if model_ids is None:
        # will take a couple seconds as-is (when run w/ -f [the "full" model list])
        model_ids = model_roi_odor_df[model_cols + ['model_dirname']].drop_duplicates()
        # this should just mean that we can use model_cols to align model_order and
        # model_roi_odor_df (I initially was not writing the order parquets with full
        # model_dirname)
        assert (
            len(model_ids[model_cols].drop_duplicates()) ==
            model_ids.model_dirname.nunique()
        )
        assert not model_ids.isna().any().any()
        model_ids = model_ids.set_index(model_cols, verify_integrity=True).squeeze()

    if max_supp_models_only and model_order is not None:
        # subsetting should be done effectively in loop above now.
        # could delete these assertions eventually.
        #
        # TODO i assume this would also fail if there was mismatch between saved and
        # current models (shouldn't matter if i end up moving this plotting code to
        # where parquets are saved below, as planned)
        assert model_ids.loc[model_order.index].equals(model_ids), \
            'expected subsetting to all be effectively above now'
        # TODO this .loc would err if model_order has stuff not in current models,
        # right? want more explicit error message there?
        first_model_dirnames = set(model_ids.loc[model_order.iloc[first_indices].index])
        assert model_roi_odor_df.model_dirname.isin(first_model_dirnames).all(), \
            'expected subsetting to all be effectively above now'
        del first_model_dirnames

        # to check we don't even need to re-order model_order/model_ids, and can use
        # pass just model_order to plotting fn
        assert model_order.drop(columns=[
            c for c in model_order.columns if c != 'model_dirname'
        ]).squeeze().equals(model_ids)

    if NORM_PER_MODEL:
        model_norm_desc = 'within EACH model'
    else:
        model_norm_desc = 'across ALL models'

    if not NORM_PER_FLY:
        NORM_DESC = f'{model_norm_desc} and within KCs'
    else:
        # TODO ? delete? not used anyway
        NORM_DESC = f'{model_norm_desc} and per KC fly'

    compare_normalized = {
        # TODO separate mapping indicating what we should call this shared
        # thing? or maybe i should keep both names, esp if i want to compare
        # multiple normalized response strength metrics (i.e. adding a logistic
        # scaled version of # spikes)
        'mean_num_spikes': KC_RESP_COL,
        'mean_logistic_scaled_num_spikes': KC_RESP_COL,
    }

    assert 'roi' in id_cols
    assert 'connectome_apl' not in id_cols
    id_cols = [x for x in id_cols if x != 'roi']
    if not simplify_models:
        id_cols += ['connectome_apl', 'roi']
    else:
        id_cols += ['roi']

    df = model_roi_odor_df.groupby([x for x in id_cols if x != 'roi'] + ['stat'],
        sort=False).value.mean().reset_index()

    df.stat = df.stat.map({
        'logistic_scaled_num_spikes': 'mean_logistic_scaled_num_spikes',
        'num_spikes': 'mean_num_spikes',
        # this makes sense b/c mean above, presumably?
        'responded': 'mean_response_rate',
    })

    # above doesn't screw up odor sorting
    assert df.sort_values(by='odor', kind='stable', key=odor_sort_fn).equals(df)

    COL_ORDER = [
        'mean_Fc_zscore',
        'mean_logistic_scaled_num_spikes',
        'mean_num_spikes',
        'mean_response_rate',
    ]
    mean_stat_names = list(df.stat.unique())
    # TODO delete (no longer true now that COL_ORDER can have both kc-only and
    # model-only stats)
    #assert set(COL_ORDER) == set(mean_stat_names)
    assert len(set(mean_stat_names)) == len(mean_stat_names)

    # TODO TODO set values into global dicts (like model_marker_kws, so this can
    # actually influence other things, like in plots fns defined at module level?)
    model_markersize = MODEL_MARKERSIZE
    model_linewidth = MODEL_LINEWIDTH
    model_alpha = MODEL_ALPHA
    # TODO delete (or set into some global thing if i really care)
    #if unrestricted_full_model_params:
    #    # TODO TODO still use default alpha (and/or plot last) for uniform, in this
    #    # case, since there's only one point liek that and it's hard to find
    #    # TODO dodge/jitter more? other changes?
    #    # will be set to model_alpha_for_legend=0.7 in add_fixed_legend
    #    model_alpha = 0.15
    #    # TODO use this for everything? (markers included? comparable to
    #    # markeredgewidth?)
    #    model_markersize = 6.0

    #    # NOTE: different from (what I think can be constant linewidth for
    #    # model_marker_kws. this is used for actual line plots
    #    model_linewidth = 0.5

    # TODO which plot call was this comment referring to? seem to be using something now
    # that expects different set of args...
    #
    # do need linewidth=1.5 (or something nonzero, to see anything for '+' marker.
    # edgecolor='face' is not enough). do need edgecolor='face' (TODO for which plot
    # call? and which call(s) is this currently for? not currently using that here at
    # least) or else seaborn puts black/grey border. also need to make points not
    # filled. i prefer edgecolor='none' to edgecolor='face' (w/ marker='.' at least)
    #
    # TODO check i still need linewidth=1.5 and edgecolor='none'?
    #
    # only used as input to stripplot calls (including when passed to
    # plot_response_class_summary). don't need in add_fixed_legend.
    model_marker_kws = dict(size=model_markersize, linewidth=1.5, edgecolor='none')

    # TODO rename stats->intensity? remove panel?
    def plot_panel_stats_across_models(df: pd.DataFrame, panel: str, suffix: str = ''
        ) -> None:

        # TODO delete? even relevant anymore? maybe if data sits within these limits?
        stat2ymax = {
            'mean_response_rate': response_rate_plot_max,
            'mean_num_spikes': 0.4,
            'mean_logistic_scaled_num_spikes': 0.3,
        }
        if 'model_stat' in df.columns:
            max_model_stats = df[['model_stat','value']].groupby('model_stat').max(
                ).squeeze()

            for k, v in max_model_stats.items():
                if k.startswith(NORM_PREFIX):
                    continue

                # TODO warn/err if not startswith MEAN_PREFIX?
                assert k.startswith(MEAN_PREFIX)
                v2 = stat2ymax[k]
                if v > v2:
                    # TODO TODO is this even triggering? why is scale still broken?
                    warn(f'model max {k} ({v:.1f}) exceeded hardcoded stat2ymax value '
                        f'({v2:.1f}). replacing value in stat2ymax!'
                    )
                    stat2ymax[k] = v

        def plot_fn(data, **kwargs):
            is_normalized = False
            if 'is_normalized' in data.columns:
                is_normalized = data.is_normalized.unique()
                assert len(is_normalized) == 1, f'{is_normalized=}'
                is_normalized = is_normalized[0]

            have_model = has_model(data)

            ax = plt.gca()
            plotted_kc_data = False
            # real KC data will only have connectome_apl=False
            # TODO as will model now, if simplify_models=True. that a problem?
            for connectome_apl in (False, True):
                if 'connectome_apl' in data.columns:
                    assert not simplify_models
                    # doing it this way, rather than initial groupby, so that point
                    # markers get plotted last, so all i have to do is add a separate
                    # colorless '+' marker to legend to handle that (was mix of '.' and
                    # '+' in legend, since uniform doesn't have connectome APL case.
                    gdf = data[data.connectome_apl == connectome_apl]
                else:
                    # since there are not actually multiple (or any) values of
                    # connectome_apl in here
                    if connectome_apl:
                        break

                    gdf = data.copy()

                marker = (
                    uniform_apl_marker if not connectome_apl else connectome_apl_marker
                )

                # TODO just move the kc stuff before / after the loop?
                ylabel = None
                from_kcs = gdf[source_col] == 'KCs'
                # intentionally plotting KC stuff first, since it's busier and i want
                # model stuff to be plotted on top, to stand out better
                if from_kcs.any():
                    assert not plotted_kc_data, ('should only be present for '
                        'connectome_apl=False case'
                    )
                    plotted_kc_data = True

                    kc_stats = data['kc_stat'].dropna().unique()
                    assert len(kc_stats) == 1
                    kc_stat = kc_stats[0]

                    if have_model:
                        model_stats = data['model_stat'].dropna().unique()
                        assert len(model_stats) == 1
                        model_stat = model_stats[0]

                    # model_stat != kc_stat to exclude case where BOTH are not
                    # normalized i.e. top right stat=mean_response_rate for both.
                    # TODO may still want to try handling that case here, esp if i don't
                    # end up also normalizing mean_response_rate (scales pretty
                    # different)
                    # TODO TODO want to still do any of this if only KC data? don't
                    # think so...
                    if have_model and not is_normalized and (model_stat != kc_stat):
                        kc_ax = ax.twinx()
                        kc_ax.spines['top'].set_visible(False)
                        # TODO leave this one, but change color (+alpha?) to kc_color
                        #kc_ax.spines['right'].set_visible(False)
                        kc_ax.set_ylabel(f'KC {kc_stat}', color=kc_color)
                        kc_ax.tick_params(axis='y', color=kc_color)
                        for text in kc_ax.yaxis.get_ticklabels():
                            text.set_color(kc_color)

                        assert model_stat != 'response_strength', ('should not have '
                            'name of normalized stat ("response_strength") here'
                        )

                        model_ymax = None
                        try:
                            model_ymax = stat2ymax[model_stat]
                        except KeyError:
                            warn(f'{model_stat=} missing from stat2ymax. could not '
                                'align unnormalized model vs KC response strength.\n'
                                f'{gdf[~from_kcs].value.max()=}'
                            )

                        if model_ymax is not None:
                            # TODO TODO (still true?) why is this so screwed up
                            # again in -f case?  kiwi numspikes and logistic numspikes
                            # barely have any KC CIs on the axes....
                            # TODO TODO is it that the other ax changes scale
                            # later?
                            # TODO TODO calculate model stat max in here, rather
                            # than use stat2ymax dict?
                            model_mean = gdf[~from_kcs].value.mean()
                            scale_relative_to_ymax = model_mean / model_ymax
                            # taking mean w/in odor first, then mean, so we don't weight
                            # odors w/ more flies more. shouldn't matter too much.
                            kc_mean = gdf[from_kcs].groupby('odor').value.mean().mean()
                            kc_ymax = kc_mean / scale_relative_to_ymax
                            assert (gdf[from_kcs].value >= 0).all(), 'set diff ymin'
                            kc_ax.set_ylim([0.0, kc_ymax])
                    else:
                        kc_ax = ax

                    # this is definitely not empty in at least some calls
                    # (has e.g. x='odor', y='value')
                    kwargs_without_color = {
                        k: v for k, v in kwargs.items() if k != 'color'
                    }

                    # TODO when we have pair_dilution_factor (which will only be for a
                    # KC-only case), add groups and lines separating dilution factors,
                    # to replace current str suffixes for the /10 and /100 cases

                    # TODO assert mean (of CI? need to get that from artists,
                    # to really know?) is 1.0 (after normalization) (seems like it might
                    # be 1.2, so maybe calculation is accidentally using margin or
                    # something? see pink lines in control_vs_kcs_*.pdf plots)
                    #
                    # delete this comment. this was just b/c i was concatenating 5comp
                    # and binary stuff, each normalized separately, together. not sure
                    # there is any great way to combine those outputs into one analysis,
                    # at least not with assuming the experimental scales are the same
                    # [which they seem they might not be?], and probably no great way to
                    # normalize across the two [could pin odors? don't want to].

                    # TODO TODO TODO still mean not 1.0 in simplify_models
                    # control_binary_vs_kcs.pdf? seems slightly above it...
                    # same for logistic scaled... same for 5comp...
                    # (delete? fixed now, right?)

                    # TODO just leave color to palette again?
                    # TODO could maybe remove dodge=False if i use palette and hardcoded
                    # (or input hue level? not sure we have in input...)
                    #
                    # need dodge=False here as long as we only have one hue level here,
                    # or else will get ZeroDivisionError
                    #
                    # legend=False here hides legend inside this plot, but then there is
                    # also no marker next to 'KCs' in outside legend (will try to add
                    # handling for that outside)
                    sns.pointplot(gdf[from_kcs], ax=kc_ax, color=kc_color, dodge=False,
                        legend=False, **kc_pointplot_kws, **kwargs_without_color
                    )

                    # hack to get max of all errorbars (in data coords)
                    # there is one initial element that does not seem like an errorbar
                    # line
                    # TODO why are there so many lines tho? this was the control binary
                    # data, no? why not just 3 lines? have_model=False
                    # ipdb> pp [x.get_data() for x in kc_ax.lines]
                    # [(array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
                    #   array([0.26, 0.27, 0.2 , 0.31, 0.27, 0.18, 0.46, 0.24, 0.23])),
                    #  (array([0, 0]), array([0.19, 0.38])),
                    #  (array([1, 1]), array([0.17, 0.43])),
                    #  (array([2, 2]), array([0.13, 0.27])),
                    #  (array([3, 3]), array([0.12, 0.42])),
                    #  (array([4, 4]), array([0.17, 0.45])),
                    #  (array([5, 5]), array([0.06, 0.26])),
                    #  (array([6, 6]), array([0.19, 0.67])),
                    #  (array([7, 7]), array([0.14, 0.37])),
                    #  (array([8, 8]), array([0.09, 0.31]))]
                    if have_model:
                        y_errmax = max([
                            x.get_data()[-1].max() for x in kc_ax.lines
                            if len(x.get_data()[0]) == 2
                        ])
                        kc_ymin, kc_ymax = kc_ax.get_ylim()
                        # TODO use same margin var here? * (1 + margin)?
                        kc_ymax2 = y_errmax * 1.1
                        if kc_ymax < kc_ymax2:
                            warn(f'setting kc_ax ymax={kc_ymax2:.1f} so all errorbars '
                                'should be visible'
                            )
                            kc_ax.set_ylim([kc_ymin, kc_ymax2])
                        del kc_ymax, kc_ymax2

                    plot_individual_flies = not have_model
                    stripplot_kws = perfly_stripplot_kws
                    # TODO TODO hue/marker = 'mix'? (to debug intensity differences
                    # across 5comp/binary natmix recordings. pick a response threshold
                    # that gives same average response rate for overlapping odors [top 2
                    # components] in each mix=?)
                    if plot_individual_flies:
                        _debug_exp_type = False
                        # TODO rename this flag to be more generic, and then also use
                        # hue to show '5comp' vs 'binary' in remy kiwi/control cases?
                        if _debug_exp_type and 'exp_type' in gdf.columns:
                            # TODO also try fly_id? different plot fn, w/ lines
                            # connecting points for one fly, across odors? multiple
                            # calls, each w/ diff marker (instead of hue)?
                            assert 'palette' not in kwargs
                            unique_exp_types = gdf[from_kcs].exp_type.unique()
                            exp_type_palette = dict(zip(
                                sorted(unique_exp_types),
                                sns.color_palette('hls', len(unique_exp_types))
                            ))
                            stripplot_kws = dict(hue='exp_type',
                                palette=exp_type_palette, alpha=0.6, size=5.0
                            )

                        # TODO delete? like color=kc_color here? (for consistency w/ ORN
                        # vs KC plot) (and if that works, can i just to
                        # palette=source_palette?)
                        #stripplot(gdf[from_kcs], ax=kc_ax, color='k',
                        stripplot(gdf[from_kcs], ax=kc_ax, color=kc_color,
                            **stripplot_kws, **kwargs_without_color
                        )
                    else:
                        if verbose:
                            warn('not plotting individual flies, b/c '
                                f'{plot_individual_flies=}'
                            )

                if have_model:
                    # can't use float dodge w/ striplot unfortunately
                    # TODO ig i could make figure wider?
                    # jitter=1.0 is too much. 0.3 too much too, esp w/ aspect=1.1
                    # TODO move marker and model_alpha into model_marker_kws?
                    stripplot(gdf[~from_kcs], ax=ax, hue=PNKC_CLASS_COL, jitter=0.13,
                        dodge=False, palette=source_palette, marker=marker,
                        alpha=model_alpha, **model_marker_kws, **kwargs
                    )


        def plot_stats(df: pd.DataFrame, extra_suffix: str = '') -> None:
            assert len(df) > 0, f'empty df with {extra_suffix=}'

            # TODO TODO for bottom left axes here (for model vs KCs version of this
            # plot), show KC ax w/ ticks reflecting original KC data range (even tho
            # plottted KC data actually normalized max of means -> 1)
            # TODO and if NORM_PER_MODEL=False, could also do same on left axis for
            # model data, but it will probably be True moving forward, so can't

            from_model = df[source_col] != 'KCs'
            have_model = from_model.any()

            # TODO use has_model fn? (esp if i start analyzing ORN data too...)
            # try using has_model fn if this fails
            if have_model:
                model_vals = df[source_col].dropna()
                if len(model_vals) > 0:
                    assert not model_vals.str.lower().str.startswith('orn').any()
                del model_vals
            #

            kc_response_stat = None
            kc_normed_response_stat = None
            if from_model.all() or not have_model:
                df = df[~ df.stat.str.startswith(NORM_PREFIX)].copy()
                # TODO TODO (still true? delete?) how is this getting set to just
                # 'mean_response_rate'???
                # TODO change def to not need to manually add new things to COL_ORDER
                # (or at least be clear that's what needs to happen if assert below
                # fails, about set of col_order, after this if/else)
                col_order = [x for x in COL_ORDER if x in df.stat.unique()]
                facet_kws = dict()

                # TODO TODO TODO need to do anything else for `not have_model` case?
                # (i initially tried handling that in else below, but now all that code
                # is commented out)
                if not have_model:
                    df['kc_stat'] = df.stat.copy()

                    kc_stats = set(df.stat.unique())
                    # TODO refactor to share w/ assertion hardcoding
                    # 'mean_response_rate' below?
                    nonnormed_kc_stats = {x for x in kc_stats
                        if x != 'mean_response_rate' and not x.startswith(NORM_PREFIX)
                    }
                    assert len(nonnormed_kc_stats) == 1, f'{nonnormed_kc_stats=}'
                    kc_response_stat = nonnormed_kc_stats.pop()
            else:
                # TODO remove this copy? already copying below now too
                df = df.copy()
                kc_stats = set(df[~from_model].stat.unique())
                kc_normed_response_stats = {
                    x for x in kc_stats if x.startswith(NORM_PREFIX)
                }
                assert len(kc_normed_response_stats) == 1, \
                    f'{kc_normed_response_stats=}'
                kc_normed_response_stat = kc_normed_response_stats.pop()

                model_stats = set(df[from_model].stat.unique())

                assert not any(x.startswith(NORM_PREFIX) for x in model_stats)
                model_stats_to_norm = set(compare_normalized.keys()) & model_stats
                assert len(model_stats_to_norm) == 1, f'{model_stats_to_norm=}'
                model_response_stat = model_stats_to_norm.pop()
                raw_df = df[from_model & (df.stat == model_response_stat)]
                # TODO delete not the same, b/c this expects to currently always takes
                # mean within each odor (in this case, across models), before
                # normalizing (b/c NORM_TO_FLYMEAN_MAX=True)
                #
                #normed_df2 = normalize_one_panel(raw_df)
                #print(f'{pd_allclose(normed_df, normed_df2, equal_nan=True)=}')
                #print(f'{normed_df.equals(normed_df2)=}')
                #
                normed_df = raw_df.copy()
                # TODO strip MEAN_PREFIX before prepending NORM_PREFIX? (don't think
                # so?) (well i'm stripping for the other case below...) (maybe this one
                # already have MEAN_PREFIX removed when defined tho?) (assert not both
                # prefixes in any stat name? fn for that?)
                model_normed_response_stat = f'{NORM_PREFIX}{model_response_stat}'
                normed_df['stat'] = model_normed_response_stat
                if not NORM_PER_MODEL:
                    normed_df['value'] = normed_df.value / normed_df.value.max()
                else:
                    assert not raw_df.model_dirname.isna().any()
                    model2max = normed_df.groupby('model_dirname', sort=False
                        ).value.max()
                    for x in raw_df.model_dirname.unique():
                        normed_df.loc[normed_df.model_dirname == x, 'value'] /= \
                            model2max[x]

                    assert np.allclose(normed_df.groupby('model_dirname').value.max(),1)

                model_stats.add(model_normed_response_stat)

                # TODO also need to relax this if using a normalized
                # mean_response_rate column
                assert model_normed_response_stat != kc_normed_response_stat, \
                    f'{model_normed_response_stat=} == {kc_normed_response_stat}'

                assert model_stats == {'mean_response_rate',
                    model_normed_response_stat, model_response_stat
                }

                # saving original stat names, so we can access these later (instead
                # of just 'response_strength', after replacing all below
                df.loc[~from_model, 'kc_stat'] = df.stat

                # NOTE: normed_df here only contains model data. any normalized KC data
                # already in df.
                # TODO this ignore_index=True cause any problems?
                df = pd.concat([df, normed_df], ignore_index=True)
                # since index changed w/ concat
                from_model = df[source_col] != 'KCs'

                df.loc[from_model, 'model_stat'] = df.stat

                # TODO delete? always define now it is in else now?
                kc_response_stat = kc_normed_response_stat[len(NORM_PREFIX):]
                assert kc_response_stat in kc_stats

                # TODO may need to update if i include logistic scaled stuf in same plot
                # TODO fix (kc_normed_response_stat not defined here in KC-only case, or
                # None now actually)
                expected_stats = {'mean_response_rate', kc_normed_response_stat,
                    kc_response_stat
                }
                assert kc_stats == expected_stats

                df['is_normalized'] = df.stat.str.startswith(NORM_PREFIX)

                shared_norm_stat_name = 'response_strength'
                replace_dict = {
                    kc_response_stat: shared_norm_stat_name,
                    kc_normed_response_stat: shared_norm_stat_name,
                }
                if kc_normed_response_stat is not None:
                    replace_dict['kc_normed_response_stat'] = shared_norm_stat_name

                replace_dict.update({
                    # want same stat name across rows (i.e. whether normalized or
                    # not), so stat can be col_order and then row can be
                    # is_normalized
                    model_response_stat: shared_norm_stat_name,
                    model_normed_response_stat: shared_norm_stat_name,
                })
                # TODO TODO why doing this after setting unmodified model_response_stat
                # in dict above? make sense?
                if model_response_stat.startswith(MEAN_PREFIX):
                    model_response_stat = model_response_stat[len(MEAN_PREFIX):]
                #

                df['stat'] = df.stat.replace(replace_dict)

                col_order = [shared_norm_stat_name, 'mean_response_rate']
                row_order = [False, True]
                facet_kws = dict(row='is_normalized', row_order=row_order)

            if (kc_response_stat is not None and
                kc_response_stat.startswith(MEAN_PREFIX)):
                kc_response_stat = kc_response_stat[len(MEAN_PREFIX):]

            assert set(df.stat.unique()) == set(col_order), \
                f'{set(df.stat.unique())=} != {set(col_order)=} {extra_suffix=}'

            g = sns.FacetGrid(data=df, col='stat', col_order=col_order, sharey=False,
                aspect=1.1, **facet_kws
            )
            g.map_dataframe(plot_fn, x='odor', y='value')

            if have_model:
                add_fixed_legend(g, df, source_palette, lines=False)

            assert g.col_names == col_order, (
                f'{g.col_names=} != {col_order=} {extra_suffix=}'
            )
            # if only columns, axes.shape[0] should be 1 (w/ number of cols in shape[1])
            assert len(g.axes.shape) == 2, f'{g.axes.shape=} {extra_suffix=}'

            all_suffix = f'{suffix}{extra_suffix}'
            plot_dir = plot_root
            if 'kc-only' in all_suffix:
                plot_dir = model_root
            # TODO share this intensity prefix with the kc vs orn [vs eag] plot?
            # currently hardcoded both places
            fname = f'intensity_{panel}{all_suffix}'

            for (i, j, hue), gdf in g.facet_data():
                ax = g.axes[i, j]
                # hue is index into g.hue_names. ig i'm not actually managing hue
                # through the FacetGrid, at least for this one? so i can just assert
                # this
                assert g.hue_names is None and hue == 0, ('hue was not previously '
                    'managed by FacetGrid. would need to update code'
                )

                if len(gdf) == 0:
                    # should be non-normalized mean_response_rate only
                    # (or KC-only case, where there is also no normalized 'Fc_zscore'
                    # data currently)
                    # TODO should i change to never even specify row='is_normalized', if
                    # there is only one value of that in input? could potentially
                    # simplify... (and allow me to use row= for something else there)
                    assert g.row_names[i] == True

                    # TODO can i also do this in the mean_response_rate case?
                    #
                    # mean_response_rate ax currently removed below, and not yet sure if
                    # doing so here would cause issues (it may still be used to get
                    # ticklabels for other axes?)
                    if g.col_names[j] != 'mean_response_rate':
                        ax.remove()

                    continue

                is_normed = False
                if 'is_normalized' in gdf.columns:
                    is_normed = gdf.is_normalized
                    if is_normed.all():
                        is_normed = True
                    else:
                        assert not is_normed.any()
                        is_normed = False

                assert not gdf[source_col].isna().any()
                from_kcs = gdf[source_col] == 'KCs'

                # TODO delete? is 'kc_stat' always in columns?
                stat_col = 'stat'
                if have_model and from_kcs.any():
                    assert 'model_stat' in gdf.columns
                    stat_col = 'model_stat'
                # TODO this always gonna be in columns? assert that?
                elif 'kc_stat' in gdf.columns:
                    stat_col = 'kc_stat'

                # dropna should only be dropping model_stat rows for source == 'KCs'
                df_stat_cols = gdf[stat_col].dropna().unique()
                assert len(df_stat_cols) == 1, f'{df_stat_cols=}'
                stat_col = df_stat_cols[0]

                if have_model:
                    # TODO care to do anything other than just skip this assertion in
                    # this case? change handling more broadly?
                    #
                    # this should always contain the unnormalized value
                    assert stat_col != 'response_strength', f'{stat_col=}'

                    sser = gdf[~from_kcs].value
                else:
                    sser = gdf.value

                assert len(sser) > 0

                # how much is needed to make sure at least KC confidence
                # intervals are fully shown? adjust based on that automatically?
                # 0.2 currently seems to barely work in all the cases i care about
                # (kiwi/control/diag-binaries_max)
                margin = 0.2

                ymin = 0
                if not is_normed:
                    try:
                        ymax = stat2ymax[stat_col]
                    # TODO should i add KC-only stats to stat2ymax now (for KC-only
                    # plots?) or include in same?
                    except KeyError as err:
                        # TODO refactor to share w/ below? (or just set None and also do
                        # below in that case?)
                        dmax = sser.max()
                        ymax = dmax + margin * dmax
                        #
                        # (err just contains name of missing key)
                        # TODO (delete. should be handled fine by other code now, if not
                        # this?) need to address this? (and similar)
                        # ```
                        # Warning: stat2ymax missing stat_col='mean_Fc_zscore'
                        # setting ymax=0.467
                        # ```
                        # still, why not falling back to something reasonable?
                        warn(f'stat2ymax missing stat_col={err}\nsetting {ymax=:.3f}')
                else:
                    # only other place that uses stat2ymax does not deal with any
                    # normalized data, so we can hardcode this here, and exclude this
                    # from stat2ymax.
                    #
                    # just adding a tiny bit of margin on top of 1.0, to clip points
                    # less mainly. all normalized (typically max=1.0) data currently
                    # takes the name 'response_strength'
                    ymax = 1.0 + margin

                # TODO should i be checking have_model here? or just always if
                # from_kcs.any()?
                if is_normed and have_model and from_kcs.any():
                    assert 'kc_stat' in gdf.columns

                    kdf = gdf[from_kcs]
                    assert np.isclose(kdf.groupby('odor').value.mean().max(), 1), \
                        f"{kdf.groupby('odor').value.mean().max()=} != 1"

                    kser = kdf.value
                    # TODO option to only set range to include error bars, not all
                    # points? (currently am further expanding to errorbars below i
                    # believe. not sure if i need to disable here... i should have to,
                    # but do i actually? something up?)
                    # intentionally not increasing margin if KC max only within margin
                    # above KC data (checking ymax not prev dmax)
                    dmax = kser.max()
                    if dmax > ymax:
                        # TODO also say which stat this is, or something else?
                        warn(f'KC data exceeded initial model {ymax=:.3f}! setting to '
                            f'{dmax=:.3f}'
                        )
                        ymax = dmax + dmax * margin

                dmax = sser.max()
                abs_margin = dmax * margin
                assert dmax > 0, ('relative margin calculation would need to change if '
                    'this were to fail. in particular for ymin.'
                )
                try:
                    # used to also allow exact equality, but i decided i always want
                    # some margin
                    assert sser.max() < ymax, f'{stat_col=} {sser.max()=} > {ymax=}'
                except AssertionError as err:
                    # TODO refactor to share w/ above (or just do all here by changing
                    # conditional logic)
                    ymax = dmax + abs_margin
                    #
                    warn(f'{err}\nsetting {ymax=:.3f}')

                try:
                    # used to also allow exact equality, but i decided i always want
                    # some margin
                    assert sser.min() > ymin, f'{stat_col=} {sser.min()=} < {ymin=}'
                except AssertionError as err:
                    dmin = sser.min()
                    ymin = dmin - abs_margin
                    warn(f'{err}\nsetting {ymin=:.3f}')

                ax.set_ylim([ymin, ymax])
                if is_normed:
                    assert g.row_names[i] == True
                else:
                    assert g.row_names == [] or g.row_names[i] == False

                if have_model:
                    # TODO this true? just delete? was previously testing RHS instead of
                    # is_normed when picking ylabel below, but that didn't work w/
                    # KC-only data, where column is just 'response_strength' (for *both*
                    # normed and unnormed, it seems)
                    # TODO TODO fix, so it says Fc_zscore or whatever in KC-only case
                    assert is_normed == stat_col.startswith(NORM_PREFIX)

                if j == 0 or i == 0:
                    if is_normed:
                        assert stat_col.startswith(NORM_PREFIX), f'{stat_col=}'
                        # TODO delete?
                        #ylabel = f'normalized\n{model_stat[len(NORM_PREFIX):]}'

                        # TODO like this better than in suptitle?
                        ylabel = f'normalized\n({NORM_DESC})'
                    else:
                        # TODO delete? restore?
                        #ylabel = stat_col
                        ylabel = 'raw'

                    ax.set_ylabel(ylabel)

                if i == 0:
                    title = stat_col
                    if title.startswith(MEAN_PREFIX):
                        title = title[len(MEAN_PREFIX):]
                    # TODO assert it doesn't start with NORM_PREFIX?

                    if (panel in NATMIX_PANELS and 'response_rate' in stat_col and
                        not from_model.all()):

                        assert kc_response_stat is not None
                        title += ('\nKC threshold: '
                            f'{kc_response_stat}>={NATMIX_KC_THRESH:.1f}'
                        )

                    # TODO what's good here? 7 too small. may need ~10?
                    ax.set_title(title, fontsize=10)
                else:
                    # TODO or None?
                    ax.set_title('')

            ncols = g.axes.shape[1]
            assert len(g.axes.shape) == 2 and ncols > 1, (
                f'{g.axes.shape=} {extra_suffix=} {df.stat.unique()=}'
            )
            nrows = g.axes.shape[0]
            labels = None
            if g.row_names != []:
                assert nrows > 1
                # seems that xticklabels not currently defined for anything but last
                # row, but still not showing up for last row for some reason
                for i, col_name in zip(range(ncols), g.col_names):
                    ax = g.axes[-1, i]
                    # TODO check other cols if labels are defined (not just [-1,i])?
                    if labels is None:
                        labels = ax.get_xticklabels()
                    else:
                        l2 = ax.get_xticklabels()
                        # == doesn't seem to return True when i'd want for Text objects
                        # reprs are like "Text(0, 0, '2h')", and include both position
                        # of label and text
                        assert all(repr(x) == repr(y) for x, y in zip(labels, l2))
                        assert all(
                            x.get_text() == y.get_text() for x, y in zip(labels, l2)
                        )

                    # TODO can i just do this above, when gdf is empty? prob not...
                    if col_name == 'mean_response_rate':
                        ax.remove()

                # trying to make more space for right y-label (for twinx overlay on top
                # left axes)
                # .05 is less than default. .15 too it seems
                # TODO g.fig is same as g.figure, right? (was using latter before here)
                # (docs say figure should be preferred moving forward)
                # TODO could use move_legend, but don't really care
                g.fig.subplots_adjust(wspace=0.75, hspace=0.4)
            else:
                assert nrows == 1

            for ax in g.axes.flatten():
                ax.tick_params(labelbottom=True)

            # labels=None here still produces the labels I want for nrows=1 case
            g.set_xticklabels(labels=labels, rotation=90)

            suptitle = panel
            # TODO delete
            #if (~ from_model).any():
            #    assert df.is_normalized.any()

            g.fig.suptitle(suptitle, y=1.04)

            savefig(g, plot_dir, fname, normalize_fname=False)

        if 'connectome_apl' in df.columns:
            assert not simplify_models
            df = df.copy()
            df.connectome_apl = df.connectome_apl.fillna(False)

        # TODO try to combine logistic scaled into same plot...?
        is_model = df[source_col] != 'KCs'
        logistic_scaled = df.stat.str.contains('logistic_scaled')
        raw_num_spikes = df.stat.str.contains('num_spikes') & ~logistic_scaled

        # TODO delete?
        model_vals = df[is_model][source_col].dropna()
        if len(model_vals) > 0:
            assert not model_vals.str.lower().str.startswith('orn').all()
        #

        # TODO use has_model fn? (esp if i start analyzing ORN data too...)
        # try using has_model fn if assertion above fails (unlikely...)
        # TODO actually check we have some matching EXPECTED_MODEL_PNKC_CLASSES?
        have_model = is_model.any()
        if not have_model:
            if 'pair_dilution_factor' in df.columns:
                # TODO TODO also pass an extra arg to split pair_dilution_factor apart
                # by row or something? hue? extra grouped xticks?
                # TODO maybe exclude (/don't add) normalized column value here,
                # so that row can be this, instead of that?
                df = df.copy()
                # TODO TODO try to get xticklabel groups to work instead of this
                dilution_factors = np.power(10.0, df.pair_dilution_factor)
                if np.allclose(dilution_factors, dilution_factors.astype(int)):
                    dilution_factors = dilution_factors.astype(int)

                dilution_strs = dilution_factors.astype(str)
                # TODO some different formatting for this? use latex?
                dilution_strs = dilution_factor_delim + dilution_strs

                replace_str = f'{dilution_factor_delim}1'
                if np.issubdtype(dilution_factors.dtype, float):
                    # '1.0' seems to be the default formatting for float 1.0
                    replace_str += '.0'

                # will leave the ' / 10' and ' / 100' entries as-is
                dilution_strs = dilution_strs.replace(replace_str, '')
                df.odor = df.odor + dilution_strs

                df = df.sort_values(by='odor', kind='stable', key=odor_sort_fn)

            # should already have '_kc-only[_pair-dilutions]' in overall suffix
            # TODO TODO assert that? is that really the only case that reaches this?
            plot_stats(df, '')
        else:
            # TODO use stat2fname_part for all these (would have to refactor, or at
            # least loop below, instead of using those hardcoded masks)
            #stat_part = stat2fname_part(stat_col)

            assert is_model.any()

            # TODO change if i start using this for ORN data too?
            # (for an ORN vs KC plot) (or happy w/ existing plot for that?)
            # TODO assert source is just 'KCs' within these then?
            from_kcs = ~is_model
            # same as assertion above but w/e
            assert not from_kcs.all()

            # TODO define all these w/o all the negation? (loop over stats, and use
            # stat2fname_part?)

            any_logistic_scaled = logistic_scaled.any()
            if from_kcs.any():
                assert any_logistic_scaled, ('expected logisitic scaled model outputs'
                    ' in call comparing to KC data'
                )
                plot_stats(df[from_kcs | ~raw_num_spikes], '_logistic-scaled')
                plot_stats(df[from_kcs | ~logistic_scaled], '_num-spikes')

            # TODO just megamat that doesn't have logistic scaled?
            if any_logistic_scaled:
                plot_stats(df[is_model & ~raw_num_spikes],'_logistic-scaled_model-only')

            plot_stats(df[is_model & ~logistic_scaled], '_num-spikes_model-only')
            # TODO is this just missing from megamat tdf or what?

        # TODO or just compare all pairs of normalized things (across model vs
        # fly)? will only be one pair initially anyway

    # TODO TODO avoid calling this on copy of natmix_*data|df used to define test_df
    # kiwi/control subset, if leave_concs_in_odors=True? care?
    # (that should be only case where model input handling needs to change. other places
    # the mix names are constructed from binary mixtures of ORN component data, or
    # completely synthetic odors)
    def strip_concs(odors: Union[pd.Index, pd.Series]) -> Union[pd.Index, pd.Series]:
        # TODO TODO still avoid calling on KC data in leave_concs_in_odors=True case,
        # and instead rename each element to put the concs in diff format, as loop
        # constructing diag binaries does?
        return odors.map(olf.strip_concs_from_odor_str
            # component_delim (= ' + ') -> '+'
            ).str.replace(component_delim, '+', regex=False)


    def process_yang_odors(odors: Union[pd.Series, pd.Index]) -> Union[
            pd.Series, pd.Index
        ]:
        if not leave_concs_in_odors:
            # TODO TODO this actually what i want always?
            return strip_concs(odors)
        else:
            odors = odors.str.replace('farn @ -2.5', 'farn @ -3', regex=False)
            # TODO assertions after doing this?
            odors = odors.map(process_odor_str_for_model)
            breakpoint()
            # TODO assert no more duplicates? or else expand rename dict?
            return odors
        # TODO TODO assert odors not duplicated before / after (or at least unique odors
        # before not duplicated after)


    yang_panel = yang_df.index.get_level_values('panel')
    assert yang_panel.equals(yang_bin_df.index.get_level_values('panel'))
    def get_yang_panel_ser(panel: str, df: Optional[pd.DataFrame] = None) -> pd.Series:
        if df is None:
            df = yang_df

        flyroi_ser = df.loc[yang_panel == panel].dropna(how='all', axis='columns'
            ).stack()
        # after .stack(), Series index should be `fly_cols + ['roi' ,'odor]`, and there
        # should be no NaN (but could be varying # of ROIs per fly, and potentially
        # different odors for different flies)
        assert not flyroi_ser.isna().any()
        return flyroi_ser


    def process_yang_df_odors(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        without_concs = strip_concs(df.odor)

        nonvalue_cols = None
        if without_concs.nunique() < df.odor.nunique():
            nonvalue_cols = [x for x in df.columns if x != 'value']
            assert not df[nonvalue_cols].duplicated().any()

        # ipdb> df.odor.value_counts()
        # 2h @ -7                  24
        # 2h @ -7 + farn @ -3      24
        # 2h @ -7 + ma @ -7        24
        # farn @ -3                24
        # farn @ -3 + ma @ -7      24
        # ma @ -7                  24
        # 2h @ -7 + farn @ -2.5    10
        # farn @ -2.5              10
        # farn @ -2.5 + ma @ -7    10
        #
        # either way, will treat farn -2.5 and farn -3.0, but will rename the farn -2.5
        # stuff to farn -3.0 in leave_concs_in_odors=True case
        # TODO could drop instead? meh
        if not leave_concs_in_odors:
            # TODO include more detailed info, similar to some stuff currently
            # output in load/preprocess yang data fns?
            warn('averaging over multiple concs for some odors in KC data! see load'
                '/preprocess output above for more details.'
            )
            df['odor'] = without_concs
        else:
            warn(f'renaming farn -2.5 to farn -3.0, since {leave_concs_in_odors=}. '
                'This is somewhat consistent with leave_concs_in_odors=False path, '
                'where concs are stripped and they will end up treated the same anyway.'
            )
            df['odor'] = process_yang_odors(df.odor)

            name_and_odor = pd.concat([df.odor, without_concs], axis='columns')
            name_and_odor.columns = ['odor', 'name']
            name_and_odor = name_and_odor[name_and_odor.columns].drop_duplicates()
            name2nconcs = name_and_odor.groupby('name').size()

            assert (name2nconcs == 1).all(), (f'{name2nconcs[name2nconcs > 1]=}\n'
                f'{name_and_odor.to_string(index=False)}'
            )

        # TODO assert not None?
        if nonvalue_cols is not None:
            assert df[nonvalue_cols].duplicated().any()
            df = df.groupby(nonvalue_cols, sort=False).value.mean().reset_index()

        # necessary for plots to still have odors in order i want
        df = df.sort_values(by='odor', kind='stable', key=odor_sort_fn)
        # TODO delete? shouldn't matter
        df = df.reset_index(drop=True)
        #
        return df


    # TODO factor to module level? (would then also have to pass in dataframes from
    # load_* fn)
    def get_yang_panel_means(panel: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # TODO what are the possible value for panel here?

        flyroi_ser = get_yang_panel_ser(panel)
        assert isinstance(flyroi_ser, pd.Series)

        # averaging over ROIs, within each (fly, odor)
        group_cols = ['panel', 'odor'] + fly_cols
        assert all(x in flyroi_ser.index.names for x in group_cols)
        # this will be the case iff load_yang_kc_data had drop_exp_type=False arg
        if 'exp_type' in flyroi_ser.index.names:
            group_cols += ['exp_type']

        # could define group_cols in one step like this, but might want to have the
        # columns listed above in the earlier order positions that should enforce
        group_cols += [x for x in flyroi_ser.index.names
            if x not in group_cols and x != 'roi'
        ]
        # we are just intending to average over ROIs, within unique combos of all other
        # index variables
        assert set(flyroi_ser.index.names) - {'roi'} == set(group_cols)

        # we have already computed mean (of per-trial, within-window Fc_zscore mean
        # from 2s after onset of a 2s odor pulse to 8s after onset (same as Remy's
        # maybe? / similar?)
        # TODO why did yang not start averaging within odor pulse? really that slow?
        flyavg_ser = flyroi_ser.groupby(group_cols).mean().rename(KC_RESP_COL)
        assert isinstance(flyavg_ser, pd.Series)

        flyroi_bin_ser = get_yang_panel_ser(panel, yang_bin_df)
        # (delete) does yang's analysis pipeline already (effectively?) exclude or
        # not-pull-out non-responding ROIs? some estimate of what fraction of ROIs fall
        # in that category if so? (i.e. how many double counts does she already have, if
        # she doesn't attempt to resolve them. then denominator should just be total #
        # expected KCs). she believes analysis should pull out non-responders, there
        # should be no bias favoring responders (in double counting or identification),
        # and that there has been no similar filtering at this point.
        flyavg_bin_ser = flyroi_bin_ser.groupby(group_cols).mean().rename(
            'mean_response_rate'
        )
        assert isinstance(flyavg_bin_ser, pd.Series)
        assert flyavg_ser.index.equals(flyavg_bin_ser.index)

        flyavg_df = pd.concat([flyavg_ser, flyavg_bin_ser], axis='columns',
            verify_integrity=True
        )
        assert flyavg_df.index.equals(flyavg_ser.index)
        # now the existing index will be duplicated, once for each existing column name
        # (which will be the values of the new 'stat' column, with the values in new
        # 'value' columns, similar to model data in `df` outside)
        flyavg_df = flyavg_df.melt(ignore_index=False, var_name='stat')
        flyavg_df = flyavg_df.reset_index()

        # TODO TODO factor out odor processing, to share w/ preprocessing for mix supp
        # calc too? (and check whether i can do it before above processing, to same
        # effect? if so, i could move it into get_yang_panel_ser)
        # (prob can't do in get_yang_panel_ser, b/c i think some of the code above is
        # currently changing order of odors, so need to at least sort them after? or
        # would at least need to change code above to preserve order, if possible)
        flyavg_df = process_yang_df_odors(flyavg_df)

        # TODO normalize yang's stuff outside here, so we can do it across panels?
        # even want to do it across panels? i don't think so...
        return flyavg_df


    # TODO also do something with the other panel in yang_dfs? (natmix-top2-dilute)
    # (i think those concs are too low to be able to compare to any orn data we really
    # have currently... since the ramp experiments i did were not of good quality)
    # TODO could at least lump the data for that one mix that overlaps between that
    # panel and diag-binaries into diag-binaries (would probably want to do when
    # initially loading yang data, and maybe also exclude from natmix-top2-dilute at the
    # same time)
    # TODO rename flyodor->flyavg_odor?
    panel2kc_flyodor_stats: Dict[str, pd.DataFrame] = dict()
    panel2kc_mix_supp: Dict[str, pd.DataFrame] = dict()
    # stats like max intensity for each (ROI, odor), to make e.g. distributions of odor
    # intensity for mixes vs components
    panel2kc_flyroi_odor_stats: Dict[str, pd.DataFrame] = dict()

    panel2orn_mix_supp: Dict[str, pd.DataFrame] = dict()
    panel2orn_flyroi_odor_stats: Dict[str, pd.DataFrame] = dict()

    for panel in ('diag-binaries',):
        if skip_panels is not None and panel in skip_panels:
            continue

        yang_flyavg_df = get_yang_panel_means(panel)
        panel2kc_flyodor_stats[panel] = yang_flyavg_df

        flyroi_ser = get_yang_panel_ser(panel)
        # TODO make sure this doesn't cause issues w/ calculating mix suppression (it
        # will make farn -2.5 and -3, both comps and in mixes, look the same,
        # potentially w/o ROIs having same meaning across them? see _check_farn code)
        # TODO maybe drop to only farn -3 instead? seems all flies should have that
        # (from _check_farn code) (compare output from current method to dropping farn
        # -2.5)
        flyroi_df = process_yang_df_odors(flyroi_ser.rename('value').reset_index())
        del flyroi_ser

        # TODO TODO is this consistent w/ how i'll handle model diag mixes tho
        # (prob not?)? also computed across all components/diag mixes, or just within
        # each?
        # TODO TODO instead, store responder mask alongside (in?) whatever i'll use
        # to calculate mix_suppression, and then calculate which "non-responders" to
        # drop down there?
        panel_responded = yang_bin_df.loc[panel].dropna(how='all', axis='columns')

        # TODO this work in both leave_concs_in_odors=True/False cases
        panel_responded.columns = process_yang_odors(panel_responded.columns)

        panel_responders = panel_responded.T.any()
        panel_responders = addlevel(panel_responders, 'panel', panel)

        # TODO need to groupby farn / farn-mixes and take union of whether it
        # responded to any now? i assume so. doing it now
        # TODO change mix_supp_list2... below to not care if 'stat' already in df, and
        # then don't copy here?
        panel_responded = panel_responded.stack()
        assert not panel_responded.isna().any()
        assert set(panel_responded.unique()) == {0, 1}
        panel_responded = panel_responded.astype(bool)
        # seems it's already sorted anyway
        panel_responded = panel_responded.groupby(level=panel_responded.index.names,
            sort=False).any()

        panel_responded = panel_responded.rename('responded')

        flyroi_odor_stats = flyroi_df.copy()
        flyroi_odor_stats['stat'] = KC_RESP_COL

        flyroi_odor_stats = flyroi_odor_stats.set_index(panel_responded.index.names)
        flyroi_odor_stats['responded'] = panel_responded.loc[flyroi_odor_stats.index]
        flyroi_odor_stats = flyroi_odor_stats.reset_index()

        is_responder = flyroi_odor_stats.pivot(columns=fly_cols + ['roi'], index='odor',
            values='responded').any()
        assert not is_responder.all(), ('panel2kc_flyroi_odor_stats elements should all'
            ' still contain non-responders'
        )
        del is_responder
        assert flyroi_odor_stats.notna().all().all()

        # TODO TODO (delete? done?) also group odors with a 'mix' level, as for
        # kiwi/control data? am i already doing that in some places? move handling
        # earlier to share w/ per-ROI clustering yang_df? (or even care to cluster each
        # separately? i guess i might need to depending on which flies have what?)
        # NOTE: currently unused
        # TODO TODO use this too? (in addition to just for kiwi/control panels)
        panel2kc_flyroi_odor_stats[panel] = flyroi_odor_stats

        # TODO TODO does a responder rate this low make sense???
        # ipdb> yang_bin_df.loc[panel].dropna(how='all', axis='columns').T.any().sum()
        # 4851
        # ipdb> yang_bin_df.loc[panel].dropna(how='all', axis='columns').T.any(
        #   ).sum() / 20442
        # 0.2373055474024068

        # TODO delete?
        responded = addlevel(panel_responded, 'panel', panel)

        mixes = [x for x in flyroi_df.odor.unique() if '+' in x]
        # ['2h+farn', '2h+ma', 'farn+ma']
        mix_supp_dfs = []
        for mix in mixes:
            c1, c2 = [x.strip() for x in mix.split('+')]
            mix_df = flyroi_df[flyroi_df.odor.isin((c1, c2, mix))]
            assert set(mix_df.odor.unique()) == {c1, c2, mix}

            mix_ser = mix_df.set_index([x for x in mix_df.columns if x != 'value'],
                verify_integrity=True).squeeze()

            # dropping responders per-mix instead of per-panel, as i'm not sure how many
            # of yang's flies have all or most of the mixes, and this is currently how
            # model stuff is handled below
            mix_responded = responded[
                responded.index.get_level_values('odor').isin((c1, c2, mix))
            ]
            non_odor_levels = [x for x in mix_responded.index.names if x != 'odor']
            mix_responders = mix_responded.groupby(level=non_odor_levels, sort=False
                ).any()

            if MIX_SUPP_IN_RESPONDERS_ONLY:
                mix_ser = mix_ser[mix_responders]

            mix_df = mix_ser.unstack('odor')
            assert not mix_df.isna().any().any()

            mix_df = mix_df.sort_index(level='odor', key=odor_sort_fn, axis='columns',
                kind='stable').T

            # TODO delete. was old per-panel responder dropping, inconsistent w/ how
            # model data is currently handled
            # TODO + delete unused panel_responders code above now?
            #assert panel_responders.index.equals(mix_df.columns)
            #if MIX_SUPP_IN_RESPONDERS_ONLY:
            #    mix_df = mix_df.loc[:, panel_responders]

            plot_hierarch_clustered_rois(model_root, mix_df,
                f'{panel}_{mix.replace("+", "-")}', title=mix,
                ignore_existing=ignore_existing
                # TODO TODO restore this if we no longer are dropping fly colors for
                # yang stuff? restore this and check if i still want to drop fly colors?
                # (would want to first check if i can concat across mixtures maybe?)
                #, row_coarsen_factor=KC_ROW_COARSEN, row_coarsen_by=fly_cols
            )

            # TODO also group into response classes here, before calculating mix
            # suppression? (changing that code to work w/ 2 components in process)

            mix_supp = calc_mix_suppression(mix_df)
            del mix_df
            mix_supp['mix'] = mix
            mix_supp = mix_supp.set_index('mix', append=True)
            mix_supp_dfs.append(mix_supp)

        mix_supp = mix_supp_list2flystat_df(mix_supp_dfs)
        panel2kc_mix_supp[panel] = mix_supp

    # contents:
    # remy_kiwi-control_5comp_Fc_zscore.csv
    # remy_kiwi-control_5comp_Fc_zscore.parquet
    # remy_kiwi-control_5comp_fly2n-total-rois.csv
    # remy_kiwi-control_5comp_fly2n-total-rois.parquet
    # remy_kiwi-control_binary_Fc_zscore.csv
    # remy_kiwi-control_binary_Fc_zscore.parquet
    # remy_kiwi-control_binary_fly2n-total-rois.parquet
    # remy_kiwi-control_binary_fly2n-total-rois.csv
    remy_dir = data_root / 'internal/remy_natmix_kcs'
    assert remy_dir.is_dir()

    # TODO TODO factor out all the natmix loading/preprocessing to a fn, mainly to
    # declutter main()
    #
    prefix = 'remy_kiwi-control_'
    response_suffix = '_Fc_zscore.parquet'
    fly2n_total_rois_suffix = '_fly2n-total-rois.parquet'
    mdf = read_parquet(remy_dir / f'{prefix}5comp{response_suffix}')
    fly2n_total_rois_5comp = read_parquet(
        remy_dir / f'{prefix}5comp{fly2n_total_rois_suffix}'
    )
    # TODO TODO maybe i want to analyze these actually? at least for correlation /
    # breadth? just exclude from most existing analyses (+ sort towards end, from
    # highest to lowest conc, w/in existing sort key fn, if using)
    # TODO TODO if i'm not calling this on ORN data, what is dropping this from
    # kiwi/control model input (/output)?
    # (it's only dropped in here when i process the model output. i could analyze it
    # them if i wanted)
    mdf = drop_mix_dilutions(mdf)

    # TODO TODO does haoru actually have binary mix at steps down (he may have binary
    # mix, but probably not at lower concs?) (either way, include his data too?)
    # TODO sanity check this isn't all the same exact data for ea/eb (or is it
    # pinned to that mix or something? why spread so small on activation strength for
    # that particular [top] concentration step?) (still true?)
    bdf = read_parquet(remy_dir / f'{prefix}binary{response_suffix}')
    fly2n_total_rois_binary = read_parquet(
        remy_dir / f'{prefix}binary{fly2n_total_rois_suffix}'
    )

    # TODO also have this (or fn factored out of this?) return the
    # (date,fly_num)->n_rois series, to check each element is <= that of the total #
    # ROIs series i'm loading here
    print()
    print("Remy's kiwi/control KC data:")
    for desc, data, total_rois in [
            ('binary', bdf, fly2n_total_rois_binary),
            ('5-component', mdf, fly2n_total_rois_5comp),
        ]:
        print(f'{desc}:')
        # this will also print a summary of the # of ROIs per fly, unless explicitly
        # verbose=False
        n_flies, n_rois = count_flies_and_rois(data)
        assert len(total_rois) == n_flies
        assert total_rois.sum() > n_rois
        print()

    # TODO better names for these than mdf/bdf
    mdf = preprocess_natmix_df(mdf)
    bdf = preprocess_natmix_df(bdf)

    # both should be {'control', kiwi'}
    natmix_panels = set(mdf.columns.get_level_values('panel').unique())
    assert natmix_panels == set(bdf.columns.get_level_values('panel').unique())
    assert natmix_panels == set(NATMIX_PANELS)
    del natmix_panels

    # TODO move up somewhere? CLI flags? (don't want to rely on these inside any
    # plotting fns inside main tho, and currently don't)
    ANALYZE_ORN: bool = True
    ANALYZE_EAG: bool = False

    eag_intensity = None
    if ANALYZE_EAG:
        # TODO move this file to data/ subdir of al_analysis, for consistency?
        eag = pd.read_csv('mean_min_eag.csv')
        eag['name'] = eag.odorname.map(olf.abbrev)

        assert eag.name.notna().all()
        assert not (eag.name == eag.odorname).any()
        all_natmix_abbrevs = set()
        # NOTE: no 'pfo' (in this CSV at least), so on odors should be in >1 of the
        # panels
        eag['panel'] = np.nan
        for panel in NATMIX_PANELS:
            panel_odors = set(panel2name_order[panel])
            eag.loc[eag.name.isin(panel_odors), 'panel'] = panel
            all_natmix_abbrevs |= panel_odors

        assert eag.panel.notna().all()
        assert eag.name.isin(all_natmix_abbrevs).all()
        eag = eag.drop(columns='odorname')

        eag = eag[(eag.panel == 'kiwi') | (
            (eag.panel == 'control') & (eag.air_dilution == 0.2)
        )].copy()
        # each panel should only have one air_dilution (standard 0.1 for kiwi, and
        # weaker 0.2 for control)
        assert len(eag[['panel', 'air_dilution']].drop_duplicates()) == 2
        eag = eag.drop(columns='air_dilution')

        eag['odor'] = eag[['name', 'log10_conc_vv']].rename(
            columns={'log10_conc_vv': 'log10_conc'}).apply(format_odor, axis='columns')

        eag = eag.drop(columns=['name', 'log10_conc_vv'])

        assert (eag.min_eag_mv < 0).all()
        # should make it easier to compare to other data where more positive = more
        # intense
        eag_stat = 'abs_min_eag_mv'
        eag[eag_stat] = eag.min_eag_mv.abs()
        eag = eag.drop(columns='min_eag_mv')

        eag.odor = strip_concs(eag.odor)

        eag = eag.set_index(['panel','odor','fly_id'], verify_integrity=True).squeeze()
        assert isinstance(eag, pd.Series)

        eag_df = eag.to_frame()
        eag_df['stat'] = eag_stat
        eag_df = eag_df.rename(columns={eag_stat: 'value'})

        # TODO refactor to share w/ orn processing below
        normed_eag_dfs = []
        for panel in NATMIX_PANELS:
            pdf = eag_df[eag_df.index.get_level_values('panel') == panel]
            normed_pdf = normalize_one_panel(pdf)
            normed_eag_dfs.append(normed_pdf)

        normed_eag_df = pd.concat(normed_eag_dfs, verify_integrity=True)

        curr_levels = list(eag_df.index.names)
        level_order = curr_levels + ['is_normalized']
        normed_eag_df = addlevel(normed_eag_df, 'is_normalized', True).reorder_levels(
            level_order)

        eag_df = addlevel(eag_df, 'is_normalized', False).reorder_levels(level_order)
        eag_intensity = pd.concat([eag_df, normed_eag_df], verify_integrity=True)
        eag_intensity['source'] = 'EAG'
        #

    if ANALYZE_ORN:
        orn_df = load_natmix_dff()

        group_cols = ['panel', 'odor1']
        orn_df = orn_df.groupby(level=group_cols, sort=False).mean()
        orn_df = orn_df.loc[
            ~orn_df.index.get_level_values('odor1').str.startswith('pfo')
        ]
        orn_df = drop_mix_dilutions(orn_df)
        # NOTE: there is already not air mix in here, and odors already sorted as i want
        # within each panel (w/ binary mix at end, right before full mix)

        print('\nORN data:')
        n_flies, n_rois = count_flies_and_rois(orn_df)
        del n_rois

        n_str = f'\nn={n_flies}'

        # TODO why am i calling this separately from preprocess_natmix_df call again?
        # cause odor level diff? (needed diff data format up here, but having renamed,
        # anyway?)
        orn_df = rename_natmix_odors(orn_df, 'odor1')

        check_odor_index_sorted_within_panel(orn_df)

        # TODO seems maybe we were relying on odors already being sorted in output
        # of load_natmix_dff? assert sorted or re-sort here, to ensure?

        orn_mean = orn_df.groupby(level='roi', axis='columns').mean()
        orn_mean = fill_to_hemibrain(orn_mean)

        # would just have to apply this sorting if it failed. checking none of the calls
        # between loading and here re-ordered odors

        check_odor_index_sorted_within_panel(orn_mean)

        plot_responses(orn_mean.T, model_root, 'orn_mean_responses',
            # default title pad=6.0
            title=f'mean ORN responses{n_str}', title_kws=dict(fontsize=6, pad=10.0),
            vline_level_fn=format_panel,
            vline_group_text=True,
            # default: 0.08
            vgroup_label_offset=0.13,
            group_fontsize=6.0,
            xticklabels=format_mix_from_strs,
            levels_from_labels=False,
            linecolor='k',
            yticklabels=True,
            cbar_label=mean_response_desc,
            cbar_shrink=0.4,
            # happy with this? default was probably -.2 or so (from mean data i assume)
            vmin=ORN_VMIN,
            # TODO move cbar closer to main ax?
        )

        # dropping diagnostics
        orn_df = orn_df[orn_df.index.get_level_values('panel').isin(NATMIX_PANELS)
            ].copy()

        assert set(orn_df.index.get_level_values('panel')) == set(NATMIX_PANELS)
        # TODO lines around binary mix?
        for panel in sorted(NATMIX_PANELS):
            if skip_panels is not None and panel in skip_panels:
                continue
            panel_orn_df = orn_df.loc[panel]
            orn_corr = mean_of_fly_corrs(panel_orn_df)
            plot_corr(orn_corr, model_root, f'orn_corr_{panel}',
                title=f'mean ORN correlation{n_str}',
            )

        # TODO and plot corrs for KC data i care about? (mean of fly)
        # across the diagonal concentrations for the kiwi stuff?
        # (or in model_yang_mixtures?)

        # calling this after plot_responses b/c it reshapes and adds NaN. single rows
        # would not show for both kiwi and control panels.
        # now duplicating the odor renaming from this earlier, so plots above use those
        # names.
        orn_df = preprocess_natmix_df(orn_df)

        raw_orn = orn_df.groupby(level=['panel'] + fly_cols, sort=False,
            axis='columns').mean().unstack().dropna()
        raw_orn = raw_orn.rename('value').to_frame()

        names_before = list(raw_orn.index.names)
        raw_orn = raw_orn.reset_index()
        raw_orn.odor = strip_concs(raw_orn.odor)
        raw_orn = raw_orn.set_index(names_before, verify_integrity=True)

        orn_stat = 'mean_peak_dff'
        raw_orn['stat'] = orn_stat

        # TODO just start by defining it w/ this name, rather than renaming, once i
        # delete commented normalization code above
        orn_intensity = raw_orn
        #
        orn_intensity['source'] = 'ORNs'
        del raw_orn

    # only the highest concs (the concs that are also the components in the 5-component
    # mixtures) of the binary ramp experiment should be in both natmix_df and bdf
    natmix_odors = get_odors(natmix_df, axis='columns')
    # TODO TODO need to do this check differently depending on -L? or nah?
    # (not currently processing any of the natmix stuff depending on it... and that
    # might be fine?)
    binary_comps = natmix_df.columns[natmix_odors.isin(bdf.index)]
    # TODO delete
    # TODO TODO assert binary_comps is len 2 or what? len 6? 4?
    #print()
    #print('these depend on -L?')
    #print(f'{binary_comps=}')
    #print()
    #print(f'{bdf.index=}')
    #breakpoint()
    #
    binary_comps_and_mixes = []
    # get list of only the top-concentration binary mixture and its components, to
    # subset bdf to that
    panel_diag_dfs = []
    for panel, odf in binary_comps.to_frame(index=False).groupby('panel'):
        panel_comps_and_mix = []
        assert len(odf) == 2, 'expected 2 top-components per panel'
        assert odf.odor.nunique() == 2
        # NOTE: these are just top concentration for each component
        sorted_comps = sorted(odf.odor)
        panel_comps_and_mix.extend(sorted_comps)

        # TODO ever need to worry about '+' being the delimiter instead of ' + '?
        mix_str = component_delim.join(sorted_comps)
        assert mix_str in bdf.index, f'{mix_str=} not in index:\n{bdf.index}'
        panel_comps_and_mix.append(mix_str)

        # TODO refactor to share below? probably just skipping below for now
        mix_str_rev = component_delim.join(sorted_comps[::-1])
        assert mix_str_rev not in bdf.index, (f'did not expect {mix_str_rev=} in index:'
            f'\n{bdf.index}'
        )
        binary_comps_and_mixes.extend(panel_comps_and_mix)

        # index will now be a single 'odor' level with only the odors (at all pairwise
        # mixtures, and alone) for the current panel
        pdf = bdf.loc[:, panel].dropna(how='all')
        panel_comps_allconcs = pdf.index[
            ~pdf.index.str.contains(component_delim, regex=False)
        ]
        # TODO work? need to group into lists first?
        # bit inefficient but whatever
        compname2concs = {
            # without sorting value lists, seem to be in high->low (e.g. -3, -4...)
            # order by default. shouldn't really matter, but should be consistent across
            # the two. sorted(...) puts them in reverse order (e.g. -5, -4...)
            parse_odor_name(x): sorted([
                parse_log10_conc(y) for y in panel_comps_allconcs
                if parse_odor_name(y) == parse_odor_name(x)
            ]) for x in panel_comps_allconcs
        }
        err_suffix = f'got:\n{pformat(compname2concs)}'
        assert len(compname2concs) == 2, ('expected 2 components for pair ramp '
            f'experiment. {err_suffix}'
        )
        n_steps = 3
        assert all(len(concs) == n_steps for concs in compname2concs.values()), (
            'expected 3 concentration steps for each component in pair experiment.'
            f' {err_suffix}'
        )
        # TODO also assert highest conc is as above?
        for concs in compname2concs.values():
            # since sorted(...) above puts in order like -5, -4... diff should all be 1
            diff = np.diff(concs)
            assert np.allclose(diff, 1), ('expected all concentration steps to be in '
                f'powers of 10 (difference of 1 on log scale). {err_suffix}'
            )

        sorted_comp_names = [parse_odor_name(x) for x in sorted_comps]
        n1, n2 = sorted_comp_names
        odors = pdf.index
        # this should just get the diagonal
        diag_dfs = []
        for i, (c1, c2) in enumerate(zip(compname2concs[n1], compname2concs[n2])):
            # TODO modify format_odor to allow taking name/log10_conc kwargs directly,
            # as an option instead of passing in a dict?
            s1 = format_odor(dict(name=n1, log10_conc=c1))
            s2 = format_odor(dict(name=n2, log10_conc=c2))

            # the fact that the concentration diffs were all 1 above means that we can
            # define dilution factor from this loop variable. since we start at lowest
            # concs (highest dilution factor), need to subtract i from n_steps
            log10_dilution = (n_steps - 1) - i

            # NOTE: if this ever fails, may need to parse odor dicts from each element
            # in index, and check equality from there, instead of just checking string
            # equality (e.g. what if code writing this didn't use same value for
            # cast_int_concs)
            assert (s1 == odors).sum() == 1
            assert (s2 == odors).sum() == 1
            # TODO also check reverse order mix is not in index, as above?

            mix = component_delim.join([s1, s2])
            assert (mix == odors).sum() == 1

            step_df = addlevel(pdf.loc[[s1, s2, mix]], 'panel', panel, axis='columns')
            del mix

            # TODO better name? this is different betweeen current and top conc (both
            # log10)
            step_df = addlevel(step_df, 'pair_dilution_factor', log10_dilution)

            diag_dfs.append(step_df)

        panel_diag_df = pd.concat(diag_dfs, verify_integrity=True)
        panel_diag_dfs.append(panel_diag_df)

    del binary_comps

    # TODO axis even matter here? (don't think so, or at least this seems fine)
    diag_df = pd.concat(panel_diag_dfs, verify_integrity=True)
    assert diag_df.columns.equals(bdf.columns)
    # only other level diag_df has in its index is pair_dilution_factor: [2,1,0]
    diag_odors = diag_df.index.get_level_values('odor')
    assert not diag_odors.duplicated().any()
    assert diag_odors.isin(bdf.index).all()
    del diag_odors

    # TODO refactor to share across the two (binary/5comp) cases (and w/ diag stuff?)?
    flyroi_binary_ser = diag_df.T.stack(diag_df.index.names).rename(KC_RESP_COL)
    assert isinstance(flyroi_binary_ser, pd.Series)
    assert not flyroi_binary_ser.isna().any()
    assert len(flyroi_binary_ser) == diag_df.notna().sum().sum()
    flyroi_binary_ser = addlevel(flyroi_binary_ser, 'mix', 'binary')
    #

    # TODO check this isn't changing set of flies? some w/ all NaN or something?
    # (it's not)
    # ipdb> c1 = count_flies_and_rois(mdf)
    # ipdb> c2 = count_flies_and_rois(flyroi_5comp_ser.to_frame().T)
    #
    # (#-flies, #-ROIs) for each
    # ipdb> c1
    # (11, 25055)
    # ipdb> c2
    # (11, 25055)
    #
    flyroi_5comp_ser = mdf.T.stack().rename(KC_RESP_COL)
    # TODO append ORN data here too (no, prob wanna keep separate for now, since e.g.
    # natmix_stat_ser created from this is thresholded w/ KC thresh)? if i'm only adding
    # source='KCs' below (or whatever), do that here instead, to add 'ORNs' too?
    # TODO refactor this processing then?
    assert isinstance(flyroi_5comp_ser, pd.Series)
    assert not flyroi_5comp_ser.isna().any()
    assert len(flyroi_5comp_ser) == mdf.notna().sum().sum()
    flyroi_5comp_ser = addlevel(flyroi_5comp_ser, 'mix', '5comp')

    assert (set(flyroi_binary_ser.index.names) - set(flyroi_5comp_ser.index.names) ==
        {'pair_dilution_factor'}
    )
    # TODO or rename to dilution_factor and use to separate out the mix dilutions
    # too, if i ever want to analyze those w/ the pair data?
    flyroi_5comp_ser = addlevel(flyroi_5comp_ser, 'pair_dilution_factor', 0
        ).reorder_levels(flyroi_binary_ser.index.names)

    natmix_stat_ser = pd.concat([flyroi_5comp_ser, flyroi_binary_ser],
        verify_integrity=True
    )

    nonroi_levels = [x for x in natmix_stat_ser.index.names if x != 'roi']
    # TODO TODO pick a threshold that gives us same average response rate as model
    # data? (or at least say what that would be, and what average response rate is with
    # whatever threshold we choose, and what it is on the model data)?
    #
    # TODO TODO retune on just kiwi/control, to get higher response rate? do for
    # all panels? would prob also be fine to tune on kiwi/control separately?
    #
    # ipdb> df[df.panel.isin(NATMIX_PANELS) & (df.stat == 'mean_response_rate')
    #   ].value.mean()
    # 0.08937741802973936
    #
    # ipdb> df[df.panel.isin(NATMIX_PANELS) & (df.stat == 'mean_response_rate')
    #   ].groupby('source').value.mean()
    # source
    # prat-claws                   0.089080
    # prat-claws_connectome-APL    0.103431
    # uniform                      0.083425
    # wd20                         0.084480
    # wd20_connectome-APL          0.086472
    #
    # TODO TODO so maybe i should use higher threshold in natmix_data/analysis.py?
    # i think i use .8 there. yang uses 1.0, but also requires > (>=?) 50% of trials,
    # so probably more stringent than my approach
    # (and also none of this is using the corrected higher denominator, that tries to
    # add back count of previously excluded ROIs)
    # ipdb> (natmix_stat_ser >= 0.8).groupby('odor').mean().mean()
    # 0.1652843599867842
    # ipdb> (natmix_stat_ser >= 1.0).groupby('odor').mean().mean()
    # 0.13511062134858848
    # ipdb> (natmix_stat_ser >= 1.25).groupby('odor').mean().mean()
    # 0.10669566477411459
    # ipdb> (natmix_stat_ser >= 1.4).groupby('odor').mean().mean()
    # 0.09302150916533318
    # ipdb> (natmix_stat_ser >= 1.5).groupby('odor').mean().mean()
    # 0.08485341122613663
    #
    # natmix_stat_ser just defined from pd.concat([flyroi_5comp_ser, flyroi_binary_ser])
    # TODO (should be done? for all?) include this in suptitle of plots that use it
    # TODO TODO include in fname too? plot for a few thresh values?
    # TODO TODO need to try using something lower, to be more consistent w/ what i
    # was using (0.8) before in natmix_data/analysis.py, unless i wanna use the
    # natmix_data/analysis.py plots w/ the higher threshold, but then not many KCs
    # passing. maybe that's good? just plot differently (log scale)?
    # (well, at least with the ROIs i have at this point, excluding some that may have
    # already been dropped as nonresponsive essentially, and not calculating with those
    # including in the denominator, i have about ~7% response rate now. what is it with
    # the corrected [higher] denominator?)
    #
    # should currently be 1.5 (but defined in mb_model, to share w/
    # natmix_data/analysis.py)
    #
    # TODO TODO make a new plot_root for case where NATMIX_KC_THRESH !=
    # REMY_KC_RESPONSE_THRESHOLD
    NATMIX_KC_THRESH: float = REMY_KC_RESPONSE_THRESHOLD
    # TODO also make some plots showing effect of this cutoff (like the ones in
    # natmix_data/analysis.py, where i send subthreshold values to 0 in a plot of
    # clustered responses)

    # TODO why no panel info here? need to regen with that (doesn't seem so, but still
    # confused on why some flies in these but not the natmix_n_responding below. see
    # comments below)?
    nrois_5comp = addlevel(fly2n_total_rois_5comp, 'mix', '5comp')
    nrois_binary = addlevel(fly2n_total_rois_binary, 'mix', 'binary')
    natmix_nrois = pd.concat([nrois_5comp, nrois_binary], verify_integrity=True)

    assert isinstance(natmix_stat_ser, pd.Series)
    assert set(natmix_stat_ser.index.names) == set(odor_cols + flyroi_cols), \
        f'{natmix_stat_ser.index.names=}'

    # TODO delete (unless it's really easier to calculate responders here and save than
    # it is to do just before mix suppression analysis)
    #natmix_responders = (natmix_stat_ser >= NATMIX_KC_THRESH).groupby(
    #    PANEL_COLS + flyroi_cols
    #).any()

    # ipdb> NATMIX_KC_THRESH
    # 1.5
    # ipdb> natmix_responders.sum() / len(natmix_responders)
    # 0.3555670417968658
    # ipdb> (natmix_stat_ser >= NATMIX_KC_THRESH).sum() / len(natmix_stat_ser)
    # 0.0743385882521062
    # ipdb> natmix_stat_ser.groupby(PANEL_COLS + fly_cols).ngroups
    # 17
    # ipdb> natmix_stat_ser.groupby(fly_cols).ngroups
    # 15
    # ipdb> len(natmix_stat_ser)
    # 307902
    # ipdb> len(natmix_responders)
    # 42563
    # TODO add some assertion this is in the right range (>1500, <~3000) (it is, at
    # least on average across flies)
    # ipdb> len(natmix_responders) / 17
    # 2503.705882352941
    #assert set(natmix_responders.index.names) == set(PANEL_COLS + flyroi_cols)
    #assert set(natmix_responders.index.names) & set(withinpanel_odor_cols) == set()
    #natmix_n_responding = natmix_responders.groupby(PANEL_COLS + fly_cols).sum()

    # TODO delete comment? i've regenerated them with panel, right?
    #
    # checking that no fly has either mix (5comp/binary) type for both panels
    # (kiwi/control), otherwise could not index fly2n_total_roi data sources without
    # panel (which they don't currently have)
    assert (
        len(natmix_stat_ser.index.to_frame(index=False)[['mix'] + fly_cols
            ].drop_duplicates()) ==
        len(natmix_stat_ser.index.to_frame(index=False)[PANEL_COLS + fly_cols
            ].drop_duplicates())
    )

    # total # of ROIs responding, per odor
    n_responding_per_odor = (natmix_stat_ser >= NATMIX_KC_THRESH).groupby(
        level=nonroi_levels, sort=False
    ).sum()
    natmix_flyavg_bin_ser = n_responding_per_odor / natmix_nrois
    assert natmix_flyavg_bin_ser.notna().all()

    # TODO try to sort odors in name order (w/in panel) at higher priority than
    # pair_dilution_factor sort? or opposite order?
    nonfly_levels = [x for x in natmix_flyavg_bin_ser.index.names if x not in fly_cols]
    print()
    print(f"thresholding Remy's KC data at mean Fc_zscore of {NATMIX_KC_THRESH:.2f}, "
        'we get mean response rates of:'
    )
    perodor_means = natmix_flyavg_bin_ser.groupby(level=nonfly_levels).mean(
        ).sort_index(level='pair_dilution_factor', kind='stable')
    print(perodor_means.to_string())

    # TODO TODO redo analysis comparing everything to a threshold that gives us
    # more like 10% response rate? (at least now my model response rates are also that
    # on average, for the ones w/ target_sparsity=0.05 that i'm actually using)
    print('across all analyzed odors (including binary mix data and dilutions): '
        f'{perodor_means.mean():.3g}'
    )
    kc_5comp_mean_resp_rate = perodor_means.loc["5comp"].mean()
    print('across all 5-component panel odors (excluding binary mix / dilutions): '
        f'{kc_5comp_mean_resp_rate:.3g}'
    )
    # TODO TODO include this one in titles? (+ compare to similarly computed for models)
    kc_no_dilution_mean_resp_rate = perodor_means[
        perodor_means.index.get_level_values('pair_dilution_factor') == 0
    ].mean()
    print('only excluding binary mix dilutions (5comp mix dilutions always excluded):'
        f' {kc_no_dilution_mean_resp_rate:.3g}'
    )
    # across all analyzed odors (including binary mix data and dilutions): 0.0471
    # across all 5-component panel odors (excluding binary mix / dilutions): 0.053
    # only excluding binary mix dilutions (5comp mix dilutions always excluded): 0.0518
    print()

    # TODO (done, right? delete?) do i also need to filter mix_minus_maxcomp stuff to
    # only responders in diag-binaries case (if i'm not already)? rename thresh to be
    # consistent? or ig i already have yang's threshold calls there? (doc how that
    # differs in my thesis)

    # TODO use other col defs than nonroi_levels now?
    # TODO at least check index names after now, against new col defs?
    natmix_flyavg_ser = natmix_stat_ser.groupby(level=nonroi_levels).mean()

    natmix_flyavg_bin_ser = natmix_flyavg_bin_ser.reorder_levels(
        natmix_flyavg_ser.index.names
    ).sort_index().rename('mean_response_rate')
    natmix_flyavg_ser = natmix_flyavg_ser.sort_index()

    # TODO refactor to share w/ processing of yang data above?
    natmix_stat_df = pd.concat([natmix_flyavg_ser, natmix_flyavg_bin_ser],
        axis='columns', verify_integrity=True
    )
    assert natmix_stat_df.index.equals(natmix_flyavg_ser.index)
    # now the existing index will be duplicated, once for each existing column name
    # (which will be the values of the new 'stat' column, with the values in new
    # 'value' columns, similar to model data in `df` outside)
    natmix_stat_df = natmix_stat_df.melt(ignore_index=False, var_name='stat')
    natmix_stat_df = natmix_stat_df.reset_index()
    #

    # TODO also skip this if -L? maybe still need to map process_odor_str_for_model
    # tho? (and not using -L for any natmix data currently, so not now)
    natmix_stat_df['odor'] = strip_concs(natmix_stat_df.odor)
    natmix_stat_df = natmix_stat_df.sort_values(by='odor', kind='stable',
        key=odor_sort_fn
    )
    natmix_stat_df = natmix_stat_df.reset_index(drop=True)

    # TODO refactor to share w/ above?
    if ANALYZE_ORN:
        orn_flyroi_ser = orn_df.T.stack().rename(KC_RESP_COL)
        assert isinstance(orn_flyroi_ser, pd.Series)
        assert not orn_flyroi_ser.isna().any()
        assert len(orn_flyroi_ser) == orn_df.notna().sum().sum()
        assert (
            orn_flyroi_ser.index.names ==
            flyroi_binary_ser.index.droplevel(['pair_dilution_factor', 'mix']).names
        )
    #

    # TODO describe purpose of this loop
    for panel in sorted(NATMIX_PANELS):
        if skip_panels is not None and panel in skip_panels:
            continue

        # TODO also need this for orn data? (if just for intensity plots, nah. and
        # yea i think that might be all)
        panel2kc_flyodor_stats[panel] = natmix_stat_df[natmix_stat_df.panel == panel]

        panel_mdf = mdf.loc[:, mdf.columns.get_level_values('panel') == panel
            ].dropna(how='all')
        assert not panel_mdf.isna().any().any()
        panel_mdf = addlevel(panel_mdf, 'mix', '5comp', axis='columns')

        if ANALYZE_ORN:
            orn_panel_df = orn_flyroi_ser.loc[
                orn_flyroi_ser.index.get_level_values('panel') == panel
            ].unstack('odor').T

            # split_mixes will currently fail otherwise, b/c component names (split from
            # e.g. 'ea+eb') will not be in index
            orn_panel_df.index = orn_panel_df.index.map(parse_odor_name)
            assert not orn_panel_df.index.duplicated().any()

            assert 'mix' not in orn_panel_df.columns.names
            assert 'mix' not in orn_panel_df.index.names

            binary_mix_list, orn_panel_mdf = split_mixes(orn_panel_df)
            assert len(binary_mix_list) == 1
            orn_panel_bdf = binary_mix_list[0]

            mdf_mix = get_single_unique(orn_panel_mdf.columns.get_level_values('mix'))
            bdf_mix = get_single_unique(orn_panel_bdf.columns.get_level_values('mix'))
            assert mdf_mix == '5comp'
            assert bdf_mix == 'binary'

            del orn_panel_df, binary_mix_list

        panel_bdf = diag_df.loc[:, bdf.columns.get_level_values('panel') == panel
            ].dropna(how='all')
        assert not panel_bdf.isna().any().any()
        panel_bdf = addlevel(panel_bdf, 'mix', 'binary', axis='columns')

        # TODO this cause issues w/ calc_mix_suppression below? (no, but it was getting
        # dropped by that call last i tried, so now adding below)
        #panel_mdf = addlevel(panel_mdf, 'pair_dilution_factor', 0)

        if MIX_SUPP_IN_RESPONDERS_ONLY:
            panel_mdf_only_responders = panel_mdf.loc[:,
                (panel_mdf >= NATMIX_KC_THRESH).any()
            ].copy()
            panel_bdf_only_responders = panel_bdf.loc[:,
                (panel_bdf >= NATMIX_KC_THRESH).any()
            ].copy()

            if ANALYZE_ORN:
                orn_panel_mdf_only_responders = orn_panel_mdf.loc[:,
                    (orn_panel_mdf >= NATMIX_ORN_RESPONSE_THRESH).any()
                ].copy()
                orn_panel_bdf_only_responders = orn_panel_bdf.loc[:,
                    (orn_panel_bdf >= NATMIX_ORN_RESPONSE_THRESH).any()
                ].copy()

        # TODO factor out one KC row_coarsen_factor (have that now), or special
        # case for each dataset, or based on target number of ROIs to display (and
        # current size of input data?)? (either way, have this fn include the chosen
        # value in title prob)
        plot_hierarch_clustered_rois(model_root, panel_mdf_only_responders,
            f'{panel}_5comp', title=panel, ignore_existing=ignore_existing,
            row_coarsen_factor=KC_ROW_COARSEN, row_coarsen_by=fly_cols
        )
        plot_hierarch_clustered_rois(model_root, panel_bdf_only_responders,
            f'{panel}_binary', title=panel, ignore_existing=ignore_existing,
            row_coarsen_factor=KC_ROW_COARSEN, row_coarsen_by=fly_cols
        )

        # TODO group ROIs into response classes (both here and for model cells),
        # before computing mix suppression, so i can also plot average mix suppression
        # for each [alongside fraction of overall population]?  (and also do for
        # diag-binaries above)?

        # this currently drops the pair_dilution_factor = 0 index level (which comes
        # before odor, which is only other level), so we do need to add after
        mix_supp_5comp = calc_mix_suppression(panel_mdf_only_responders)
        # TODO even matter? delete?
        mix_supp_5comp = addlevel(mix_supp_5comp, 'pair_dilution_factor', 0)

        # calc_mix_suppression only works when input has a single mix (and its
        # components, with mix at end)
        mix_supp_binary = panel_bdf_only_responders.groupby('pair_dilution_factor',
            sort=False).apply(calc_mix_suppression)
        assert mix_supp_5comp.index.names == mix_supp_binary.index.names

        if ANALYZE_ORN:
            orn_mix_supp_5comp = calc_mix_suppression(orn_panel_mdf_only_responders)
            # no pair_dilution_factor in ORN data (only highest was presented in my
            # useable experiments)
            orn_mix_supp_binary = calc_mix_suppression(orn_panel_bdf_only_responders)
            assert orn_mix_supp_5comp.index.names == orn_mix_supp_binary.index.names

        # TODO delete
        # at least for orn_panel_mdf, the following two were equivalent:
        # ipdb> s1 = orn_panel_mdf.T.stack(orn_panel_df.index.names).rename(ORN_RESP_COL)
        # ipdb> s2 = orn_panel_mdf.unstack().rename(ORN_RESP_COL)
        # ipdb> s1.equals(s2)
        # True
        # replace body of list comp w/ above simpler expr? (no, it only works for orn &
        # mdf (which have only one level in index), but not bdf (which has two)
        # (panel_bdf.unstack(panel_df.index.names) does work, but requires .dropna(),
        # and [for better or worse, maybe actually preserving odor order better?] has a
        # different odor order than current)
        #

        # converts dataframes with indices that are [['pair_dilution_factor', ], 'odor']
        # and columns `['mix', 'panel'] + fly_cols` to Series with the odor levels at
        # the end of a single index
        flyroi_odor_sers = [
            x.T.stack(x.index.names).rename(KC_RESP_COL) for x in [panel_mdf, panel_bdf]
        ]
        panel_mser, panel_bser = flyroi_odor_sers
        # TODO still necessary? (yes) aren't i adding this above? this isn't adding a
        # second one, is it? (no, it's apparently not added to whatever creates this, or
        # it was lost)
        panel_mser = addlevel(panel_mser, 'pair_dilution_factor', 0
            ).reorder_levels(panel_bser.index.names)

        flyroi_odor_stats = pd.concat([panel_mser, panel_bser], verify_integrity=True
            ).reset_index()
        del panel_mser, panel_bser
        assert flyroi_odor_stats.notna().all().all()

        flyroi_odor_stats.odor = strip_concs(flyroi_odor_stats.odor)
        flyroi_odor_stats = flyroi_odor_stats.sort_values(by='odor', kind='stable',
            key=odor_sort_fn
        )

        assert KC_RESP_COL in flyroi_odor_stats.columns
        assert 'value' not in flyroi_odor_stats.columns
        flyroi_odor_stats = flyroi_odor_stats.rename(columns={KC_RESP_COL: 'value'})
        assert 'stat' not in flyroi_odor_stats.columns
        flyroi_odor_stats['stat'] = KC_RESP_COL

        # TODO what all are these being used for? doc here. cause it's already been
        # subset to responders (to any odor in panel, either across 5comp or binary)
        # above, so shouldn't be used for response rates. i think it might just be used
        # for selecting cells for comp-vs-mix response strength dists tho, so should be
        # fine
        # NOTE: no longer subsetting to just non-responders above anymore
        flyroi_odor_stats['responded'] = flyroi_odor_stats.value >= NATMIX_KC_THRESH

        if ANALYZE_ORN:
            orn_flyroi_odor_sers = [
                x.unstack().rename(ORN_RESP_COL) for x in [orn_panel_mdf, orn_panel_bdf]
            ]
            # TODO refactor to share w/ above?
            orn_flyroi_odor_stats = pd.concat(orn_flyroi_odor_sers,
                verify_integrity=True).reset_index()
            assert orn_flyroi_odor_stats.notna().all().all()

            orn_flyroi_odor_stats.odor = strip_concs(orn_flyroi_odor_stats.odor)
            orn_flyroi_odor_stats = orn_flyroi_odor_stats.sort_values(by='odor',
                kind='stable', key=odor_sort_fn
            )
            assert ORN_RESP_COL in orn_flyroi_odor_stats.columns
            assert 'value' not in orn_flyroi_odor_stats.columns
            orn_flyroi_odor_stats = orn_flyroi_odor_stats.rename(
                columns={ORN_RESP_COL: 'value'}
            )
            assert 'stat' not in orn_flyroi_odor_stats.columns
            orn_flyroi_odor_stats['stat'] = ORN_RESP_COL
            orn_flyroi_odor_stats['responded'] = (
                orn_flyroi_odor_stats.value >= NATMIX_ORN_RESPONSE_THRESH
            )
            #
            panel2orn_flyroi_odor_stats[panel] = orn_flyroi_odor_stats

            orn_panel_mix_supp = mix_supp_list2flystat_df(
                [orn_mix_supp_5comp, orn_mix_supp_binary], stat=ORN_RESP_COL
            )
            panel2orn_mix_supp[panel] = orn_panel_mix_supp

        panel2kc_flyroi_odor_stats[panel] = flyroi_odor_stats

        # TODO what does this function do exactly? doc?
        panel_mix_supp = mix_supp_list2flystat_df([mix_supp_5comp, mix_supp_binary])
        panel2kc_mix_supp[panel] = panel_mix_supp


    # these should be the same across all model panels
    unique_stats = set(df.stat.unique())

    mean_model_response_rate_list = []
    suffix = ''
    for panel in panel2kc_flyodor_stats:
        kc_df = panel2kc_flyodor_stats[panel].copy()
        # TODO assert only one panel in kc_df? not sure i care? (and might even want to
        # pass both kiwi/control in that case, to normalize across both?) prob wouldn't
        # want to compute same thing twice tho...

        unique_kc_stats = set(kc_df.stat.unique())
        shared_stats = unique_stats & unique_kc_stats
        assert all(
            k in unique_stats and k not in shared_stats
            for k in compare_normalized.keys()
        )
        assert all(
            v in unique_kc_stats and v not in shared_stats
            for v in compare_normalized.values()
        )
        # TODO (delete. should always have mean_response_rate now) how to handle when
        # mean_response_rate is NOT in both (and thus not in shared_stats here), as
        # currently for natmix data? just fix that by thresholding it to get a binarized
        # version (have done that now)?
        expected_model_stats = set(compare_normalized.keys()) | shared_stats
        assert df.stat.isin(expected_model_stats).all(), \
            f'{df.stat.unique()=} {expected_model_stats=}'

        expected_kc_stats = set(compare_normalized.values()) | shared_stats
        assert kc_df.stat.isin(expected_kc_stats).all(), \
            f'{kc_df.stat.unique()=} {expected_kc_stats=}'

        normed_dfs = []
        # TODO add flag to also add normalized versions for any w/ name matching
        # exactly? (i.e. mean_response_rate)?
        # TODO TODO also try retuning on her odors, with her mean response rate
        # (and how much do fixed_thr and wAPLKC_scale differ between that and tuning on
        # megamat [and for each model, including bouton versions]?)
        for x in set(compare_normalized.values()):
            raw_df = kc_df[kc_df.stat == x]
            if panel in NATMIX_PANELS:
                kc_mix_types = set(raw_df.mix.unique())
                assert kc_mix_types == set(NATMIX_MIX_TYPES), (f'{kc_mix_types=} were '
                    f'not the same set as {NATMIX_MIX_TYPES=}'
                )

                for mix_type in NATMIX_MIX_TYPES:
                    raw_mix = raw_df[raw_df.mix == mix_type]
                    if mix_type == 'binary':
                        # no model data to comapare against at the lower dilution
                        # factors, so need to exclude them from normalization, for the
                        # *_vs_kcs* plots to have KC mean at one in these cases
                        raw_mix = raw_mix[raw_mix.pair_dilution_factor == 0]

                    # TODO TODO TODO also duplicate + separately normalize model
                    # data here (w/in same odors as current KC subset)? or do in
                    # plotting?

                    normed_df = normalize_one_panel(raw_mix)
                    assert np.isclose(normed_df.groupby('odor').value.mean().max(), 1)
                    normed_dfs.append(normed_df)
            else:
                # TODO or assert only one unique value if so
                assert 'mix' not in raw_df.columns
                normed_df = normalize_one_panel(raw_df)
                normed_dfs.append(normed_df)
            #

        kc_df = pd.concat([kc_df] + normed_dfs, ignore_index=True)
        kc_df = add_source_and_class_cols(kc_df, 'KCs')
        panel2kc_flyodor_stats[panel] = kc_df

    model_response_strengths = model_roi_odor_df.pivot(columns='stat', values='value',
        index=[x for x in model_roi_odor_df.columns if x not in ('stat','value')]
    )
    assert not model_response_strengths.isna().any().any()
    assert set(model_response_strengths['responded'].unique()) == {0, 1}
    model_response_strengths['responded'] = model_response_strengths.responded.astype(
        bool
    )
    # TODO rename all of above too, to avoid confusion (or rename var w/ same name in
    # loop below)
    all_model_response_strengths = model_response_strengths.reset_index()
    all_model_response_strengths['source_type'] = 'model'
    # TODO any reason to calculate these *model_response_strengths, instead of
    # calculating directly from model_roi_df (/df, which just has them averaged over
    # ROIs)

    natmix_panel_class_frac_list = []

    comps_to_drop = [
        'fur', 'ms', 'va', 'EtOH', 'IAol', 'IaA'
    ]
    model_mean_mix_supp_sers = []
    mean_mixsupp_list = []
    # TODO TODO is the change from 2026-05-10 outputs to those on 2026-05-20 just
    # the change in tuning convergence? or what else? is it only the wd20 and prat-claws
    # stuff moving? ignore LR cache and regen?
    # NOTE: seems like it might *just* be the wd20 case moving around (neither uniform
    # nor prat-claws [nor either APL case for latter] seem to have changed)
    # TODO TODO if so, use smaller sp_acc for everything, to minimize tuning
    # related noise? otherwise, what is the difference?
    diff_col = None
    for panel in panels:
        # TODO remove this copy? could help w/ memory issues...
        pdf = df[df.panel == panel].copy()

        kc_panel = None
        if panel2kc_panel(panel) == 'diag-binaries':
            kc_panel = 'diag-binaries'

        elif panel in panel2kc_flyodor_stats:
            kc_panel = panel

        if kc_panel is not None:
            assert kc_panel in panel2kc_flyodor_stats
            assert kc_panel in panel2kc_mix_supp
            assert kc_panel in panel2kc_flyroi_odor_stats

            for i, x in enumerate([
                    panel2kc_flyodor_stats,
                    panel2kc_mix_supp,
                    panel2kc_flyroi_odor_stats,
                ]):
                assert panel2kc_mix_supp[kc_panel] is not None, \
                    f'{panel=} {kc_panel=} {i=}'

            if ANALYZE_ORN and kc_panel in NATMIX_PANELS:
                # NOTE: ORN data does not have panel2orn_flyodor_stats, as it is only
                # used for intensity plots, which are handled separately in ORN case
                assert kc_panel in panel2orn_mix_supp
                assert kc_panel in panel2orn_flyroi_odor_stats
                for i, x in enumerate([
                        panel2orn_mix_supp, panel2orn_flyroi_odor_stats
                    ]):
                    assert panel2orn_mix_supp[kc_panel] is not None, \
                        f'{panel=} {kc_panel=} {i=}'
        else:
            assert panel not in panel2kc_flyodor_stats
            assert panel not in panel2kc_mix_supp
            assert panel not in panel2kc_flyroi_odor_stats

        if not simplify_models:
            pivot_cols = ['source', 'connectome_apl', PNKC_CLASS_COL, 'roi']
        else:
            pivot_cols =  ['source', PNKC_CLASS_COL, 'roi']

        # TODO assert only two stats in model_roi_odor_df here, and use the other one
        # rather than `!= 'responded'` here (oh, this still includes both 'num_spikes'
        # and 'logistic_scaled_num_spikes' tho. mix supp just should be computed within
        # stat, as i believe it is)
        # TODO rename model_...?
        model_responses = model_roi_odor_df[
            (model_roi_odor_df.panel == panel) & (model_roi_odor_df.stat != 'responded')
        ].pivot(index='odor', values='value', columns=pivot_cols + ['stat'])

        # TODO delete? refactor (do before pivot?)?
        model_responded = model_roi_odor_df[
            (model_roi_odor_df.panel == panel) & (model_roi_odor_df.stat == 'responded')
        ].pivot(index='odor', values='value', columns=pivot_cols)

        assert set(model_responded.values.flat) == {0, 1}
        model_responded = model_responded.astype(bool)

        model_responses = model_responses.sort_index(kind='stable', key=odor_sort_fn)
        # just for current response class defining code to work (below) (that actually
        # still matter?)
        model_responded = model_responded.sort_index(kind='stable', key=odor_sort_fn)

        assert pivot_cols == model_responded.columns.names
        nonroi_pivot_cols = [x for x in pivot_cols if x != 'roi']
        responded_cols = model_responded.columns.to_frame(index=False)
        assert PNKC_CLASS_COL in nonroi_pivot_cols
        other_pivot_levels = [
            x for x in nonroi_pivot_cols if x != PNKC_CLASS_COL
        ]
        # want this one first for easier .loc to get response rate of interest
        nonroi_pivot_cols = [PNKC_CLASS_COL] + other_pivot_levels

        # otherwise, could not just groupby source to summarize response rate of each
        # model (probably won't anyway tho...)
        # with `-o -M` (2026-07-15):
        #
        # model_pnkc_class      claw   uniform
        # odor
        # 1o3ol             0.091224  0.092998
        # 1o3ol+2h          0.080254  0.075492
        # 2h                0.016166  0.020241
        # cmix              0.054850  0.100109
        # fur               0.029446  0.053611
        # ms                0.030023  0.039934
        # va                0.025982  0.036105
        #
        # model_pnkc_class
        # claw       0.046849
        # uniform    0.059784
        perodor_model_response_rate = model_responded.groupby(level=nonroi_pivot_cols,
            axis='columns').mean().droplevel(other_pivot_levels, axis='columns')
        # TODO say how many models we are averaging over to get the response rates
        # in title, for the cases where one_model_per_pnkc_class=True but
        # not just one model per PNKC_CLASS_COL value (e.g. `-m` or default without
        # `-M`)
        mean_model_response_rate = perodor_model_response_rate.mean()

        if mean_model_response_rate.index.duplicated().any():
            assert not one_model_per_pnkc_class
            # TODO also print value_counts() of them?
            warn(f'averaging over all model variants within a given {PNKC_CLASS_COL}, '
                'to get one mean response rate per PNKC class (rather than for each '
                'single model)'
            )
            mean_model_response_rate = mean_model_response_rate.groupby(
                level=PNKC_CLASS_COL, sort=False).mean()
        else:
            assert one_model_per_pnkc_class

        if one_model_per_pnkc_class:
            print(f'mean response rate per model (across all {panel=} odors analyzed):')
            print(mean_model_response_rate.to_string())

        mean_model_response_rate = addlevel(mean_model_response_rate, 'panel', panel)
        mean_model_response_rate_list.append(mean_model_response_rate)

        # does it? change handling of ORN stuff?
        split_mixes_from = model_responses.copy()
        # TODO sort ROIs into response classes first? (maybe after grouping by mix, esp
        # in diag-binaries case?) (meh)
        model_mix_supp = None
        all_model_mix_resps = None
        # TODO should i just continue if panel is not in one of these?
        if panel in NATMIX_PANELS or kc_panel == 'diag-binaries':

            binary_mix_list, full_mix_df = split_mixes(split_mixes_from,
                model_responded, drop_nonresponders_per_mix=MIX_SUPP_IN_RESPONDERS_ONLY,
                # input here doesn't also have panel in columns, like ORN stuff does,
                # (nor in index) so passing in panel
                panel=kc_panel
            )

            binary_mix_odors = None
            if panel in NATMIX_PANELS:
                assert len(binary_mix_list) == 1
                binary_mix_odors = list(binary_mix_list[0].index)

            # TODO TODO rename these variables to include model in them, to be clear
            # that the KC data is coming from elsewhere
            mix_supp_binary = pd.concat(
                [calc_mix_suppression(x) for x in binary_mix_list],
                verify_integrity=True
            )

            if kc_panel == 'diag-binaries':
                model_mix_supp = mix_supp_binary

            elif kc_panel in NATMIX_PANELS:
                mix_supp_5comp = calc_mix_suppression(full_mix_df)
                model_mix_supp = pd.concat([mix_supp_binary, mix_supp_5comp],
                    verify_integrity=True
                )

            # TODO TODO also make response strength [comp vs mix] dist plots, i.e. most
            # of conditional below) for diag-binaries case too? (for each binary mix?)
            if kc_panel in NATMIX_PANELS:
                full_mix_mask = is_natmix_full_mix(full_mix_df.index)
                assert full_mix_mask.sum() == 1
                full_mix = full_mix_df.index[full_mix_mask][0]
                del full_mix_mask

                full_mix_odors = list(full_mix_df.index)

                panel2top_component = {
                    'control': '1o3ol',
                    # ea is actually the highest on average:
                    # TODO maybe i should go back whatever is highest in real KCs tho?
                    # unless it's iaa...
                    # ipdb> panel_response_strengths.groupby('odor'
                    # )[['logistic_scaled_num_spikes','num_spikes']].mean()
                    # stat   logistic_scaled_num_spikes  num_spikes
                    # odor
                    # EtOH                     0.846472    1.547826
                    # IAol                     0.887554    1.754902
                    # IaA                      1.148865    3.132588
                    # ea                       1.637313    7.843023
                    # eb                       1.152635    3.088710
                    # kmix0                    0.785062    1.594416
                    #
                    'kiwi': 'ea',
                }
                top_component = panel2top_component[panel]

                # TODO delete? or keep calc up here? check against below?
                # can i replace w/ part of full_mix_df? (would have to change
                # indexing so that i drop all nonresponding (KC, odor) pairs, which
                # isn't that easy in current format) (still true?)
                panel_allodor_response_strengths = all_model_response_strengths[
                    (all_model_response_strengths.panel == panel)
                ]
                assert not panel_allodor_response_strengths.responded.all(), \
                    'should still have non-responding (KC, odor) pairs here'

                model_allodor_resp_strengths_5comp = panel_allodor_response_strengths[
                    panel_allodor_response_strengths.odor.isin(full_mix_odors)
                ].copy()
                model_allodor_resp_strengths_5comp['mix'] = '5comp'

                model_allodor_resp_strengths_binary = panel_allodor_response_strengths[
                    panel_allodor_response_strengths.odor.isin(binary_mix_odors)
                ].copy()
                model_allodor_resp_strengths_binary['mix'] = 'binary'

                # TODO leave the responded column to be able to make this decision
                # later? / put behind flag similar to MIX_SUPP_IN_RESPONDERS_ONLY?
                panel_response_strengths = panel_allodor_response_strengths[
                    panel_allodor_response_strengths.responded
                ]
                # NOTE: the below also changes l1 above (unique combos of those columns
                # that dont include odor... make sense?)
                # TODO TODO note that top component differs depending on metric. going
                # to just hardcode which component i want to compare to for now
                # (1o3ol for control, and probably eb for kiwi. need to check there)
                #
                # ipdb> panel_response_strengths.groupby('odor')[[
                #   'logistic_scaled_num_spikes', 'num_spikes']].mean()
                # stat   logistic_scaled_num_spikes  num_spikes
                # odor
                # 1o3ol                    0.815560    1.579604
                # 2h                       0.751807    1.574113
                # cmix0                    0.587520    1.209181
                # fur                      0.721496    1.465144
                # ms                       0.872448    1.795571
                # va                       1.006529    2.084291
                # TODO move subsetting into plot_response_strength_dists fn?
                # (as well as subsetting to responding (KC, odor) pairs too, if so)
                # (well, at least i have the *_allodor_* versions now, which i'm using
                # for new plot, tho still drop to responding (unit, odor) pairs for the
                # version i like...)

                # this is for the response strength dist plot that just includes (unit,
                # odor) responders, of top component and mix. not used for general
                # per-odor response strength / sparsity plot (which is based on pdf),
                # nor the per-odor response distribution plots (which uses *allodor*
                # variant)
                model_response_strengths_5comp = panel_response_strengths[
                    # TODO refactor to share w/ other mask defs?
                    panel_response_strengths.odor.isin((top_component, full_mix))
                ].copy()
                model_response_strengths_5comp['mix'] = '5comp'

                # TODO fix how columns.name = 'stat' for model_response_strengths (and
                # derived vars)?  shouldn't matter but still

                unique_odors = panel_response_strengths.odor.unique()
                assert all(x in unique_odors for x in (top_component, full_mix))

                binaries = [x for x in unique_odors if '+' in x]
                assert len(binaries) == 1, f'{binaries=}'
                binary_mix = binaries[0]
                del unique_odors

                # TODO TODO do this subsetting later, so i can plot dists across all
                # components more easily
                model_response_strengths_binary = panel_response_strengths[
                    panel_response_strengths.odor.isin((top_component, binary_mix))
                ].copy()
                model_response_strengths_binary['mix'] = 'binary'

                model_allodor_resp_strengths = pd.concat([
                        model_allodor_resp_strengths_5comp,
                        model_allodor_resp_strengths_binary
                    ], ignore_index=True
                )
                all_model_mix_resps = pd.concat(
                    [model_response_strengths_binary, model_response_strengths_5comp],
                    # need ignore_index so that we can duplicate the top_component
                    # responses for each mix= value (or at least can't use
                    # verify_integrity=True for that reason)
                    ignore_index=True
                )

        # TODO TODO fit gaussian to non-responders in yang's labelled non-responders, as
        # well as in my thresholded Remy data (for a few thresholds?)
        # TODO eventually incorporate as preprocessing step in fn that fits scaling fn
        # to match spike count distribution to real data?

        # TODO TODO TODO are there really no plots made befoer this? refactor to move
        # the stuff that doesn't require KC data above?
        if kc_panel is None:
            warn(f'skipping rest of loop analysis for {panel=}, because no '
                f'corresponding kc_panel'
            )
            continue

        # NOTE: don't need to rename panel in this dataframe (no matter if it's diff
        # from current `panel` in loop, because plotting fn does not actually use
        # panel in input data, just the input panel str for naming stuff)
        # TODO rename flyodor_stats?
        kc_df = panel2kc_flyodor_stats[kc_panel]

        # TODO why offset: (just b/c diags and yang's odors are at slightly diff
        # concs...)
        # ipdb> model_odors - kc_odors
        # {'2h-6 + farn-2', 'farn-2 + ma-7', '2h-6 + ma-7', 'farn-2', '2h-6'}
        # ipdb> kc_odors - model_odors
        # {'2h-7', '2h-7 + farn-3', '2h-7 + ma-7', 'farn-3 + ma-7', 'farn-3'}
        # ipdb> kc_odors & model_odors
        # {'ma-7'}
        model_odors = set(pdf.odor.unique())
        kc_odors = set(kc_df.odor.unique())
        # TODO maybe for some other panels (besides diag-binaries, where there is an
        # exact match) i will just want to check s2 is a subset of s1?
        # TODO (delete) why does pdf have a mix of stripped concs and not now?
        # (was probably a bug that was since fixed. was only for one directory that
        # i deleted)
        # > s1
        # {'2h-6 + farn-2', 'ma-7 + farn-2', 'ma-7 + 2h-6', 'farn', 'ma', '2h+ma',
        # 'farn+ma', '2h', '2h+farn'}
        # > s2
        # {'farn', 'ma', '2h+ma', 'farn+ma', '2h', '2h+farn'}
        #
        # may be difficult/impossible to get not-stripping concs to work w/ yang's data,
        # unless i pick certain concs to exclude? (diag data was collected at diff concs
        # too!)
        # TODO and still do KC-only analyses despite this? shouldn't super matter...
        if model_odors != kc_odors:
            msg = (f'{panel=} {kc_panel=}\n{model_odors=}\n{kc_odors=}\neven though '
                'kc_panel was non-None, model and KC odors did not match! This is '
                'probably because of a bug. Skipping rest of analysis for this panel.'
            )
            if leave_concs_in_odors:
                msg += (' It may also work if you re-run without any of the -L/-o/-u '
                    'options.'
                )
            warn(msg)
            continue

        # TODO make sure odors are sorted in same order, so that we don't
        # screw up xticklabel order for plots that include KC data (and will that
        # fix it? how did i sort odors for model stuff again? do here or when
        # loading yang data?)
        # (they should be now, but assert that here?)

        assert model_mix_supp is not None, ('model mix suppression not calculated '
            f'for {panel=} {kc_panel=}'
        )
        model_mix_supp = model_mix_supp.reset_index()
        # TODO or leave as model panel? (doing that for now. many fns using it will
        # normalize to kc panel anyway, and any code that currently doesn't prob should)
        model_mix_supp['panel'] = panel
        # TODO already have this or no?
        model_mix_supp['source_type'] = 'model'

        kc_mix_supp = panel2kc_mix_supp[kc_panel]
        kc_mix_supp = add_source_and_class_cols(kc_mix_supp, 'KCs')

        mix_supp_list = [model_mix_supp, kc_mix_supp]
        orn_mix_supp = None
        if ANALYZE_ORN and kc_panel in panel2orn_mix_supp:
            orn_mix_supp = panel2orn_mix_supp[kc_panel]
            orn_mix_supp = add_source_and_class_cols(orn_mix_supp, 'ORNs')
            mix_supp_list.append(orn_mix_supp)

        mix_supp = pd.concat(mix_supp_list, ignore_index=True)

        # TODO why can i not always do this anyway? shouldn't it only be NaN for
        # non-model stuff regardless? add assertion whether it is NaN for any models
        # data?
        if not simplify_models:
            mix_supp.connectome_apl = mix_supp.connectome_apl.fillna(False)

        # TODO TODO (also?) serialize mix_supp here to parquet, if i can
        # TODO TODO save to plot_root instead?
        to_pickle(mix_supp, model_root / f'mix_supp_{panel}.p', verbose=True)

        diff_col = get_diff_col(mix_supp)

        # TODO TODO how was my pd_allclose returning False if (x == y).all().all() is
        # True? fix bug!
        # ipdb> pd_allclose(k2.reset_index()[kc_mix_supp.columns], kc_mix_supp)
        # False
        # ipdb> pd_allclose(k2.reset_index()[kc_mix_supp.columns], kc_mix_supp,
        #   equal_nan=True)
        # False
        # ipdb> pd_isclose(k2.reset_index()[kc_mix_supp.columns], kc_mix_supp)
        # *** TypeError: ufunc 'isfinite' not supported for the input types, and the
        # inputs could not be safely coerced to any supported types according to the
        # casting rule ''safe''
        # ipdb> np.isclose(k2.reset_index()[kc_mix_supp.columns], kc_mix_supp)
        # *** TypeError: ufunc 'isfinite' not supported for the input types, and the
        # inputs could not be safely coerced to any supported types according to the
        # casting rule ''safe''
        # ipdb> (k2.reset_index()[kc_mix_supp.columns] == kc_mix_supp).all().all()
        # True
        # TODO delete
        #breakpoint()
        # TODO delete (def above should be fine)
        #from_kcs = mix_supp.source == 'KCs'
        #kc_mix_supp = mix_supp[from_kcs]
        #

        if 'pair_dilution_factor' in kc_mix_supp.columns:
            # TODO TODO factor this into separate plotting fn?

            # TODO do something to specifically silence:
            # `UserWarning: The figure layout has changed to tight` from these?
            # what else emitting them?
            g = sns.FacetGrid(data=kc_mix_supp[kc_mix_supp.mix == 'binary'],
                # despite col='panel' only currently expecting one panel.
                # just so we have something, in case it matters.
                col='panel', hue='pair_dilution_factor',
                # TODO pretty good, but try husl? neon green a bit hard to see
                #palette=sns.color_palette('hls', 3)
                # TODO improve? blue and green are a bit too similar here, but prob
                # better than above
                palette=sns.color_palette('husl', 3)
            )

            # TODO filter to only responders first (for at least one version?) if i
            # haven't already? (should already be doing that. restore at least that
            # part of suptitle? remove current facet col_name title and use whole
            # thing?)
            #
            # TODO say what stat (Fc_zscore) went into diff_col somewhere (and
            # above, probably)
            # TODO try a version w/o common_norm (nah), to get a sense of how many
            # responders left for each? (or put # cells total + # responders in plot
            # somewhere?)
            # 0.35 was ok for non-filled hists, if 0.45 isn't good
            alpha = 0.6 if _USE_KDEPLOT else 0.45
            g.map_dataframe(plot_one_dist_per_fly, x=diff_col, alpha=alpha,
                linewidth=1.0
            )

            # TODO TODO set ax.title to say # of flies? (for each mix, if multiple)
            # (pass g and use facet_data to do so? share code w/
            # natmix_data/analysis.py fn doing something similar?)
            # (for this, as well as the similar facetgrids below)

            # TODO refactor all FacetGrid postprocessing to share w/ above?
            # TODO want fixed limits?
            #g.set(xlim=(-6, 6))
            # TODO delete. don't like suptitle here (and currently has panel, which
            # is redundant w/ col_name = title)
            #g.fig.suptitle(suptitle, y=1.10)
            g.set_titles('{col_name}')
            g.set_xlabels(diff_col2desc(diff_col))
            g.set_ylabels('density across KCs')
            # TODO happy with this?
            g.add_legend(title='$\log_{10}$ dilution factor')
            # TODO refactor to share 'kc-only_pair-dilutions_'
            # TODO TODO change sparsity ylim on this one, so we can we more
            # easily? [or maybe need to change threshold anyway? currently between
            # about 0.02 and 0.1 response rate, for kiwi])
            # TODO TODO fix how 'Warning: The figure layout has changed to tight'
            # gets emitted for this (and many other figures...) (which line(s) are
            # actually doing it?)
            savefig(g, model_root,
                f'mixsupp_dists_kc-only_pair-dilutions_{panel}'
            )
            #

            # so the binary mix plot below doesn't average across multiple dilution
            # factors (as concentrations are stripped at this point)
            kc_mix_supp = kc_mix_supp[kc_mix_supp.pair_dilution_factor == 0].copy()

        # TODO TODO delete this (and replace w/ subsetting model from whatever overall
        # mean mixsupp thing i create, after loop)?
        model_nonroi_levels = [
            x for x in model_mix_supp.columns if x not in ('roi', diff_col)
        ]
        # TODO (still an issue?) preserve model_dirname (prob not lost here. where
        # above is it?) (or just add back below?)
        model_mean_mix_supp = model_mix_supp.groupby(model_nonroi_levels, sort=False
            )[diff_col].mean()
        # NOTE: there is no 'mix' level here, as we are averaging over all the
        # currently-still-analyzed model odors, which include 5comp odors as well as
        # the undiluted binary mix. should be comparable to the KC mean response
        # rate calculated above (the one used towards end, for part of response
        # class plot titles)
        model_mean_mix_supp = addlevel(model_mean_mix_supp, 'panel', panel)
        model_mean_mix_supp_sers.append(model_mean_mix_supp)
        #

        if 'pair_dilution_factor' in mix_supp.columns:
            warn('dropping pair_dilution_factor other than 0/NaN for all remaining '
                'mixture suppression analysis in this panel (or across panels)'
            )
            mix_supp = mix_supp[~mix_supp.pair_dilution_factor.isin((1,2))].copy()
            remaining_vals = set(mix_supp.pair_dilution_factor.dropna().unique())
            assert remaining_vals == {0}, f'{remaining_vals=}'
            mix_supp = mix_supp.drop(columns='pair_dilution_factor')

        group_cols = [x for x in mix_supp.columns if x not in ('roi', diff_col)]

        put_first = ['panel', 'mix', 'source_type', 'model_pnkc_class', 'source']
        assert all(x in group_cols for x in put_first)
        rest = [x for x in group_cols if x not in put_first]
        group_cols = put_first + rest
        del put_first, rest

        # need dropna=False to not drop model data (where fly_cols=['date', 'fly_num']
        # will be NaT / NaN)
        panel_mean_mixsupp = mix_supp.groupby(group_cols, dropna=False)[diff_col].mean()
        mean_mixsupp_list.append(panel_mean_mixsupp)
        del group_cols

        # in case i were to accidentally try to use below (for analyses that should
        # exclude `pair_dilution_factor != 0`
        del kc_mix_supp

        # TODO TODO factor this into a separate plotting fn too?
        facet_kws = dict()
        # TODO are there any panels where this isn't true? which?
        # TODO why is there NaN in mix here? add dropna flag to this? (still?)
        multiple_mixes = has_multiple_mixes(mix_supp)
        if multiple_mixes:
            # TODO use row like above, or hue here?
            #
            # separates the binary and 5comp versions of the mix
            # TODO was thinking of just gray=binary, black=5comp, but ideally need
            # something that also works for all 3 yang's binary mixes
            facet_kws = dict(
                # TODO don't re-use husl(3)? (where else is it?)
                hue='mix', palette='cubehelix' if panel in NATMIX_PANELS else 'husl'
            )

        # need to redef kc_mix_supp like this now that we've dropped
        # pair_dilution_factor != 0 in mix_supp, but not kc_mix_supp
        kc_mix_supp = mix_supp[mix_supp.source == 'KCs']
        for data, fname_part in zip(
                [kc_mix_supp, orn_mix_supp], ['kc-only', 'orn-only']
            ):
            if data is None:
                assert panel not in NATMIX_PANELS
                assert fname_part == 'orn-only'
                warn(f'{panel=}: skipping mixsupp_per-fly_dists_* plot b/c no ORN data')
                continue

            # TODO refactor to share below w/ *_kc-only_pair-dilutions.pdf stuff above?
            # TODO say i'm dropping non-responders in all mixsupp plots somewhere
            g = sns.FacetGrid(data=data, col='panel', **facet_kws)
            # TODO just put each mix on a different row again, if hue is going to be too
            # messy? (for yang's data, at least?)
            g.map_dataframe(plot_one_dist_per_fly, x=diff_col,
                # was using 0.6 for natmix w/ KDE, but want lower w/ stepwise hist
                alpha=0.4 if panel in NATMIX_PANELS else 0.3,
                linewidth=1.0 if panel in NATMIX_PANELS else 0.75
            )
            g.set_titles('{col_name}')
            g.set_xlabels(diff_col2desc(diff_col))
            source = get_single_unique(data.source)
            unit_str = source
            if source == 'ORNs':
                unit_str = 'glomeruli'
            g.set_ylabels(f'density across {unit_str}')
            # TODO keep all lines kc_color, and use linestyle for binary/5comp
            # (dashed for binary)? (that's also used elsewhere tho... for mix vs top
            # component response strength dists)
            g.add_legend(title='mix')
            # TODO change sparsity ylim on this one, so we can we more easily (more
            # easily what? delete?)? [or maybe need to change threshold anyway?
            # currently between about 0.02 and 0.1 response rate, for kiwi])
            savefig(g, model_root, f'mixsupp_per-fly_dists_{panel}_{fname_part}')


        # TODO in each facet title, say how many nonresponders were dropped
        # (or in suptitle/legend for KCs?) would probably need a CSV for models...

        # TODO TODO figure out how to show num_spikes and KC computed data on same
        # scale (percentile? zscore?) (currently just not plotting those two against
        # each other, but removing col='stat' option from call plotting both model
        # and KC data)

        plot_all_comparisons_for(plot_mixsupp_dists, df=mix_supp,
            palette=source_palette
        )
        # TODO TODO for diag-binaries KC-only case, make a version where it's a
        # diff hue per mean line, rather than a separate row? (+ try to share palette
        # with the per-fly version of the plot, where they are already on one facet)
        #
        # this model vs KCs one is the only one we wanted to try with kde=True, so
        # handling outside of plot_all_comparisons_for
        plot_mixsupp_dists(mix_supp, plot_root, source_palette, kde=True,
            fname_suffix='_kde'
        )

        analyze_resp_strengths_and_classes_within = []
        if panel in NATMIX_PANELS:
            # TODO define from something else in a dataframe we already have here,
            # rather than hardcoding?
            analyze_resp_strengths_and_classes_within = list(NATMIX_MIX_TYPES)

        for mix in analyze_resp_strengths_and_classes_within:
            flyroi_odor_stats = panel2kc_flyroi_odor_stats[kc_panel]
            assert flyroi_odor_stats.notna().all().all()

            kc_mix_df = flyroi_odor_stats[flyroi_odor_stats.mix == mix]
            if 'pair_dilution_factor' in kc_mix_df.columns:
                # TODO just assert the whole flyroi_odors_stats is not NaN above?
                assert not kc_mix_df.pair_dilution_factor.isna().any()
                # only one we have model data to compare against
                kc_mix_df = kc_mix_df[kc_mix_df.pair_dilution_factor == 0]

            kc_responses, kc_responded = tidy2responses_and_response_mask(kc_mix_df)
            kc_class_means, kc_class_sizes = response_class_means_and_perfly_counts(
                kc_responded, kc_responses
            )
            # below plot_means_and_counts_call, kc_responded is also not really used
            # (just to compare to model_responded format)
            del kc_responses

            n_filtered_kcs = kc_responded.columns.to_frame().groupby(level=fly_cols
                ).size()
            assert n_filtered_kcs.sum() == len(kc_responded.columns)

            n_flies = len(n_filtered_kcs)

            # we can drop panel level here b/c no flies in that KC dataset have both
            # panels (as established earlier)
            curr_natmix_nrois = natmix_nrois.loc[mix].droplevel('panel')
            n_total_kcs = curr_natmix_nrois.loc[n_filtered_kcs.index].copy()

            assert (n_total_kcs >= n_filtered_kcs).all()
            n_filtered_eq_total = n_total_kcs == n_filtered_kcs
            if n_filtered_eq_total.any():
                n_sus_flies = n_filtered_eq_total.sum()
                warn(f'{kc_panel}/{mix} had the following {n_sus_flies}/{n_flies} '
                    'flies with (seemingly) no KC ROIs ever excluded:\n'
                    f'{n_total_kcs[n_filtered_eq_total]}'
                )

            del n_filtered_kcs, n_filtered_eq_total

            # no need to do this for model stuff, since that data has not had the
            # silent cells dropped here
            kc_class_sizes = add_missing_cells_to_nonresponders(kc_class_sizes,
                n_total_kcs
            )
            # TODO fix how this add_missing_cells... call is changing shape
            # from series -> dataframe (or did i need that for some other things? i
            # think i do...) (still an issue?)
            assert kc_class_sizes.sum().equals(n_total_kcs)

            ser_class_sizes = kc_class_sizes.stack(fly_cols)
            assert ser_class_sizes.notna().all()

            mix_fname_suffix = f'_{panel}_{mix}'
            fname_suffix = f'{mix_fname_suffix}_kc'
            title_suffix = f'\n{panel}/{mix}'
            title = (
                f'KCs: n={n_flies} flies ({n_total_kcs.sum()} ROIs){title_suffix}'
            )
            # just so tiny values don't look really blue (when default vmin from
            # data is like -0.05 or something)
            # TODO TODO refactor to share this as KC_VMIN (-> w/ hierarch clust plots,
            # etc)
            vmin = -0.25
            mmin = kc_class_means.min().min()
            assert mmin > vmin, f'{vmin=} {mmin=}'
            kc_stat = get_single_unique(kc_mix_df.stat)
            # TODO compare plot outputs if moved before add_missing_cells_... ?
            # (w/ n_total_kcs= still passed) (should be same as output that had
            # already had add_missing_cells... called)
            plot_means_and_counts(kc_class_means, ser_class_sizes, model_root,
                'response-class', fname_suffix=fname_suffix, title=title,
                n_total_rois=n_total_kcs, cbar_label=kc_stat, cmap=diverging_cmap,
                # TODO this is the default class_size_frac_thresh currently. change
                # default to None? or remove this?
                class_size_frac_thresh=CLASS_SIZE_FRAC_THRESH, vmin=vmin
            )
            # TODO or keep this and use below, for agging class sizes?
            del ser_class_sizes

            if ANALYZE_ORN and kc_panel in panel2orn_flyroi_odor_stats:
                orn_flyroi_odor_stats = panel2orn_flyroi_odor_stats[kc_panel]
                assert orn_flyroi_odor_stats.notna().all().all()
                orn_mix_df = orn_flyroi_odor_stats[orn_flyroi_odor_stats.mix == mix]
                orn_responses, orn_responded = tidy2responses_and_response_mask(
                    orn_mix_df
                )
                orn_class_means, orn_class_sizes = \
                    response_class_means_and_perfly_counts(orn_responded, orn_responses)

                fname_suffix = f'{mix_fname_suffix}_orn'
                # input .columns.names must have flyroi_cols, as orn_responses does
                n_flies, n_rois = count_flies_and_rois(orn_responses, verbose=False)
                title = (
                    f'ORNs: n={n_flies} flies ({n_total_kcs.sum()} ROIs){title_suffix}'
                )
                vmin = ORN_VMIN
                mmin = orn_class_means.min().min()
                assert mmin > vmin, f'{vmin=} {mmin=}'
                plot_means_and_counts(orn_class_means, orn_class_sizes, model_root,
                    'response-class', fname_suffix=fname_suffix, title=title,
                    cbar_label=orn_response_desc, cmap=diverging_cmap, vmin=vmin,
                    warn_=False,
                    # nothing was being dropped with the thresh used for KCs/model KCs,
                    # so setting None to silence the warning (while not changing
                    # anything, since still no small classes will be dropped)
                    class_size_frac_thresh=None,
                    # necessary b/c max responder class has more than non-responder
                    # class
                    break_axes_for_nonresponders=False
                )

            if mix == '5comp':
                to_drop_mask = model_responded.index == binary_mix
                assert to_drop_mask.sum() == 1
                to_drop = model_responded.index[to_drop_mask]
            else:
                assert mix == 'binary'
                comps = [x.strip() for x in binary_mix.split('+')]
                to_keep = comps
                to_keep.append(binary_mix)
                to_keep_mask = model_responded.index.isin(to_keep)
                assert to_keep_mask.sum() == 3
                to_drop = model_responded.index[~to_keep_mask]

            curr_model_responded = model_responded.drop(index=to_drop)
            curr_model_responses = model_responses.drop(index=to_drop)
            # TODO delete. not true b/c curr_model_responses still has 'stat' as
            # last level of column index (after ['source', PNKC_CLASS_COL, 'roi']),
            # so column indices won't be equal
            #assert pd_indices_equal(curr_model_responded, curr_model_responses)
            #
            assert curr_model_responded.index.equals(
                curr_model_responses.index
            )
            assert curr_model_responses.columns.droplevel('stat').drop_duplicates(
                ).equals(curr_model_responded.columns)
            assert (
                len(curr_model_responses.columns) >
                len(curr_model_responded.columns)
            ), ('will be true as long as curr_model_responses has multiple unique '
                'values in stat level of columns MultiIndex'
            )

            assert curr_model_responded.index.equals(kc_responded.index)
            del kc_responded

            # any point grouping by source first? yes, currently source column will
            # not be preserved otherwise. fly_cols seemed to be handled
            # automatically however.
            # TODO modify summarize_response_classes to automatically group on other
            # column index levels (and summarize for each), or take a kwarg to
            # specify which levels to treat like fly_cols?)
            # TODO (delete? still an issue?) are there any other places i'm
            # mistakenly assuming source still encodes APL connectome/uniform? was
            # initially thinking that here, and not including connectome_apl in
            # group levels
            #
            # TODO use model_cols (and subset from there? or care to have
            # them in this diff order, w/ source first?)
            if not simplify_models:
                model_id_cols = ['source', 'connectome_apl', PNKC_CLASS_COL]
            else:
                model_id_cols = ['source', PNKC_CLASS_COL]

            gb_model = curr_model_responded.groupby(level=model_id_cols,
                axis='columns', sort=False
            )

            # partially as another check that we will have non-responders, in inputs
            # used to compute class_sizes
            model2n_kcs = gb_model.apply(lambda x: len(x.columns)).rename('n_kc'
                ).reset_index()
            assert (len(model2n_kcs[[PNKC_CLASS_COL, 'n_kc']].drop_duplicates()) ==
                model2n_kcs[PNKC_CLASS_COL].nunique()
            )
            unique_n_kcs = model2n_kcs['n_kc'].unique()
            model2n_kcs = model2n_kcs.set_index(model_id_cols, verify_integrity=True
                ).squeeze()
            assert len(unique_n_kcs) <= 2, ('expected one value for uniform/nonclaw'
                f' and another for claw/bouton\ngot: {unique_n_kcs=}'
            )
            # TODO assert sizes are all 1828 (for uniform or probably also
            # wd20) or 1732 for claw/bouton stuff?

            assert gb_model.apply(lambda x: (x == False).all().any()).all(), (
                'not every model had non-responders prior to response class '
                'calculation'
            )

            # should halve this same set for all single models too
            # (i.e. in all iterations of loop over gb_model below)
            unique_stats = curr_model_responses.columns.get_level_values('stat'
                ).unique()

            gb_stat = curr_model_responses.groupby(level='stat', sort=False,
                axis='columns'
            )
            if unrestricted_full_model_params:
                warn('skipping response class mean+count plotting, since '
                    'unrestricted -f'
                )
                gb_stat = []

            # TODO move this analysis of model data before `kc_panel is None` check?
            for stat, stat_df in gb_stat:
                for gn, onemodel_responded in gb_model:
                    # TODO deal w/ PerformanceWarning: indexing past lexsort depth
                    # may impact performance. (from this loc[:, gn])?
                    onemodel_responses = stat_df.loc[:, gn].droplevel('stat',
                        axis='columns'
                    )

                    curr_rois = onemodel_responded.columns.get_level_values('roi')
                    assert not curr_rois.duplicated().any()
                    assert len(onemodel_responded.columns.to_frame(index=False)[[
                        x for x in onemodel_responded.columns.names if x != 'roi'
                    ]].drop_duplicates()) == 1
                    onemodel_responded = onemodel_responded.copy()
                    onemodel_responded.columns = curr_rois

                    assert pd_indices_equal(onemodel_responded, onemodel_responses)

                    g_class_sizes, g_class_means = summarize_response_classes(
                        onemodel_responded, responses=onemodel_responses,
                        verbose=False
                    )

                    metadata = dict(zip(model_id_cols, gn))
                    source = metadata['source']
                    title = str(source)

                    pnkc_class = metadata[PNKC_CLASS_COL]
                    assert pnkc_class_is_model(pnkc_class), f'{pnkc_class=}'

                    fname_suffix = f'_{panel}_{mix}'
                    # source still has info redundant w/ pnkc_class, so don't want
                    # to add that to fname_suffix currently (well, not if we need source
                    # and not just pnkc_class)
                    if one_model_per_pnkc_class:
                        fname_suffix += f'_{pnkc_class}'

                    fname_suffix += f'_{stat2fname_part(stat)}'

                    fname_suffix = (
                        f'_{panel}_{mix}_{pnkc_class}_{stat2fname_part(stat)}'
                    )

                    # otherwise we just use pnkc_class above
                    if not one_model_per_pnkc_class:
                        fname_suffix += f'__{source}'

                    if one_model_per_pnkc_class:
                        title += f'\n{pnkc_class}'

                    if 'connectome_apl' in metadata:
                        title += f'\nconnectome-APL={metadata["connectome_apl"]}'
                        fname_suffix += '_connectome-APL'

                    # TODO also want mix + panel in title, like for KC one? (meh. busy
                    # already)

                    n_total_rois = model2n_kcs[gn]

                    cbar_label = get_model_stat_label(stat,
                        logistic_scaling_title_str
                    )
                    # NOTE: normalize_fname=False already hardcoded in savefig in this
                    plot_means_and_counts(g_class_means, g_class_sizes, plot_root,
                        'response-class', fname_suffix=fname_suffix, title=title,
                        n_total_rois=n_total_rois, norm=None, cbar_label=cbar_label,
                        cmap=diverging_cmap_tophalf,
                        # TODO this is the default currently. change default to
                        # None? or remove this?
                        class_size_frac_thresh=CLASS_SIZE_FRAC_THRESH,
                    )
                del stat

            # TODO replace this w/ something from calls below, that also pass
            # responses (to not compute twice)?
            model_class_sizes = gb_model.apply(lambda x:
                # not passing responses= to this call, and just selecting first
                # returned value (second would be None w/o responses= anyway)
                summarize_response_classes(x, verbose=False, warn_=False)[0
                    ].to_frame()
            )
            assert model_class_sizes.columns.names == model_id_cols + [None]
            assert (
                model_class_sizes.columns.get_level_values(-1) == 'n_rois'
            ).all()
            model_class_sizes = model_class_sizes.droplevel(level=-1,axis='columns')
            model_class_sizes = model_class_sizes.fillna(0)
            assert np.allclose(model_class_sizes, model_class_sizes.astype(int))
            model_class_sizes = model_class_sizes.astype(int)
            assert model_class_sizes.sum().equals(
                curr_model_responded.columns.to_frame().groupby(level=model_id_cols,
                    sort=False).size()
            )

            assert model_class_sizes.index.names == kc_class_sizes.index.names
            model_class_sizes = model_class_sizes.sort_index()
            assert kc_class_sizes.equals(kc_class_sizes.sort_index())
            if not model_class_sizes.index.equals(kc_class_sizes.index):
                # TODO also include a bit about ORN stuff? move all this after agg tho?
                warn(f'{panel=} response classes only in either model or KC:\n'
                    f'{model_class_sizes.index.difference(kc_class_sizes.index)=}\n'
                    f'{kc_class_sizes.index.difference(model_class_sizes.index)=}'
                )

            # TODO do i really need to reindex until after the agg_within... anyway?
            # don't think it should matter
            shared_index = kc_class_sizes.index.union(model_class_sizes.index)

            assert kc_class_sizes.index.sort_values().equals(
                model_class_sizes.index.sort_values()
            ) or (
                len(shared_index) > len(kc_class_sizes) or
                len(shared_index) > len(model_class_sizes)
            )

            # TODO move other conditional w/ same condition from above to here, to
            # de-dedupe?
            if ANALYZE_ORN and kc_panel in panel2orn_flyroi_odor_stats:
                orn_class_sizes = orn_class_sizes.unstack(fly_cols).fillna(0
                    ).astype(int)
                assert orn_class_sizes.index.names == kc_class_sizes.index.names
                assert orn_class_sizes.columns.names == kc_class_sizes.columns.names

                shared_index = shared_index.union(orn_class_sizes.index)

            # TODO also check it has all consecutive categories? prob don't care
            # that much...
            assert shared_index.equals(shared_index.sort_values())

            # TODO assert these reindex calls aren't changing stuff in index before
            # and/or changing sums (just to sanity check)
            kc_class_sizes = reindex(kc_class_sizes, shared_index, fill_value=0)

            # TODO move analysis of these before `kc_panel is None` check
            model_class_sizes = reindex(model_class_sizes, shared_index,
                fill_value=0
            )

            # TODO already code in natmix_data/analysis.py for this?  refactor to
            # share?
            group_cols = ['mix_resp', 'n_comps']
            def agg_within_mixresp_and_ncomps(df: pd.DataFrame) -> pd.Series:
                assert df.index.names == group_cols + ['max_comp_idx']
                ser = df.groupby(level=group_cols).sum().T.stack(group_cols)
                ser = ser.rename('frac_response_class')
                return ser

            model_class_fracs = model_class_sizes / model_class_sizes.sum()
            model_class_fracs = agg_within_mixresp_and_ncomps(model_class_fracs)
            model_class_fracs = model_class_fracs.reset_index()
            model_class_fracs['source_type'] = 'model'

            # TODO move this class fracs stuff out from depending on
            # model_mix_resps? (i mean, may still not want to analyze for
            # diag-binaries panel, which is only panel it's not currently defined
            # for, but should be more clear)
            kc_class_fracs = kc_class_sizes / kc_class_sizes.sum()
            assert np.isclose(kc_class_fracs.sum(), 1).all()
            assert kc_class_fracs.sum().index.names == fly_cols
            kc_class_fracs = agg_within_mixresp_and_ncomps(kc_class_fracs)
            kc_class_fracs = kc_class_fracs.reset_index()
            kc_class_fracs = add_source_and_class_cols(kc_class_fracs, 'KCs')

            class_frac_list = [kc_class_fracs, model_class_fracs]
            if ANALYZE_ORN and kc_panel in panel2orn_flyroi_odor_stats:
                orn_class_sizes = reindex(orn_class_sizes, shared_index, fill_value=0)

                orn_class_fracs = orn_class_sizes / orn_class_sizes.sum()
                assert np.isclose(orn_class_fracs.sum(), 1).all()
                assert orn_class_fracs.sum().index.names == fly_cols
                orn_class_fracs = agg_within_mixresp_and_ncomps(orn_class_fracs)

                orn_class_fracs = orn_class_fracs.reset_index()
                orn_class_fracs = add_source_and_class_cols(orn_class_fracs, 'ORNs')
                class_frac_list.append(orn_class_fracs)

            # TODO expand both indices to full possibility of response classes (up to
            # max observed), and fill any missing values w/ 0. or do at end?
            # (delete? already doing somewhere, where i'm warning about difference
            # in class indices i think)

            panel_class_fracs = pd.concat([x for x in class_frac_list],
                ignore_index=True
            )
            # TODO do unconditionally?
            if not simplify_models:
                panel_class_fracs.connectome_apl = \
                    panel_class_fracs.connectome_apl.fillna(False)

            panel_class_fracs = panel_class_fracs.set_index(
                ['source_type'] + model_id_cols + fly_cols + group_cols,
                verify_integrity=True
            ).squeeze()
            panel_class_fracs = addlevel(panel_class_fracs, 'mix', mix)
            panel_class_fracs = addlevel(panel_class_fracs, 'panel', panel)
            natmix_panel_class_frac_list.append(panel_class_fracs)

            # TODO TODO move/duplicate def of this before `kc_panel is None` check
            # (and move the model only calls up there too)
            model_mix_resps = all_model_mix_resps[all_model_mix_resps.mix == mix]

            model_mix_allodor_resps = model_allodor_resp_strengths[
                model_allodor_resp_strengths.mix == mix
            ]
            assert not model_mix_allodor_resps.responded.all()
            assert model_mix_allodor_resps.odor.nunique() > 2

            # currently we are filtering to only have responding (KC, odor) pairs
            # above (but since this is just for one limited plot of
            # within-responder response strength, and not the main plots of response
            # strength and response rate for all odors, that's ok)
            assert model_mix_resps.responded.all()
            model_mix_resps = model_mix_resps.drop(columns='responded')

            # TODO use earlier hardcoded def? (use MODEL_STATS?)
            model_stats = [x for x in model_mix_resps.columns if 'num_spikes' in x]
            model_mix_resps = model_mix_resps.melt(
                value_vars=model_stats, var_name='stat',
                # TODO share id_vars def w/ processing of *allodor* version?
                # should be same, no? (no, one below needs ['responded'] as well.
                # already dropped from this one)
                id_vars=[x for x in model_mix_resps.columns if x not in model_stats]
            )
            model_mix_allodor_resps = model_mix_allodor_resps.melt(
                value_vars=model_stats, var_name='stat', id_vars=[
                    x for x in model_mix_allodor_resps.columns if x not in model_stats
                ]
            )

            kc_mix_allodor_resps = flyroi_odor_stats[flyroi_odor_stats.mix == mix]
            assert kc_mix_allodor_resps.responded.any()
            assert (
                set(kc_mix_allodor_resps.odor.unique()) ==
                set(model_mix_allodor_resps.odor.unique())
            )
            kc_mix_allodor_resps = add_source_and_class_cols(kc_mix_allodor_resps,
                'KCs'
            )

            # TODO better (more specific) name for these (to not need the allodor
            # specifier in less filtered version)
            kc_mix_resps = kc_mix_allodor_resps[
                kc_mix_allodor_resps.responded &
                # this is subsetting down to just mix (binary OR 5comp) + (hardcoded)
                # "top" component, consistent w/ what i'm currently doing on model data
                # above
                kc_mix_allodor_resps.odor.isin(model_mix_resps.odor.unique())
            ]
            kc_mix_resps = kc_mix_resps.drop(columns='responded')

            mix_resp_list = [kc_mix_resps, model_mix_resps]
            mix_allodor_resp_list = [kc_mix_allodor_resps, model_mix_allodor_resps]
            if ANALYZE_ORN and kc_panel in panel2kc_flyroi_odor_stats:
                orn_mix_allodor_resps = orn_flyroi_odor_stats[
                    orn_flyroi_odor_stats.mix == mix
                ]
                assert orn_mix_allodor_resps.responded.any()
                assert (
                    set(orn_mix_allodor_resps.odor.unique()) ==
                    set(model_mix_allodor_resps.odor.unique())
                )
                orn_mix_allodor_resps = add_source_and_class_cols(orn_mix_allodor_resps,
                    'ORNs'
                )
                mix_allodor_resp_list.append(orn_mix_allodor_resps)

                orn_mix_resps = orn_mix_allodor_resps[
                    orn_flyroi_odor_stats.responded &
                    orn_flyroi_odor_stats.odor.isin(model_mix_resps.odor.unique())
                ]
                orn_mix_resps = orn_mix_resps.drop(columns='responded')
                mix_resp_list.append(orn_mix_resps)

            # TODO delete this concatenation, if i'm just going to plot model vs KC
            # stuff separately below?
            response_strengths = pd.concat(mix_resp_list, ignore_index=True)

            allodor_response_strengths = pd.concat(mix_allodor_resp_list,
                ignore_index=True
            )

            # TODO refactor to share w/ other filling?
            # TODO do i not want to unconditionally do this? (maybe i didn't for model
            # params ordered by avg mixsupp plot?)
            if not simplify_models:
                response_strengths['connectome_apl'] = \
                    response_strengths.connectome_apl.fillna(False)
                # TODO do for allodor_* too?

            to_pickle(allodor_response_strengths,
                model_root / f'allodor_response_strengths_{panel}_{mix}.p', verbose=True
            )

            # TODO TODO serialize these (and other concatenated outputs) (to at least
            # pickle, if not parquet)
            # TODO TODO save to plot_root instead?
            to_pickle(response_strengths,
                model_root / f'response_strengths_{panel}_{mix}.p', verbose=True
            )
            # TODO fix whatever dtype error i have trying to write this one with my
            # to_parquet wrapper? (and also stock <df>.to_parquet)?

            # TODO TODO zscore these? or how to align? based on min/max?
            # (since we are limiting all KC data to responses, might make sense to
            # have min->0) (should i even need to do anything in logistic scaled
            # case? currently subtracting KC threshold (min, really) when comparing
            # against raw model 'num_spikes', inside plotting fn)

            if not unrestricted_full_model_params:
                facet_kws = dict()

                unique_odors = allodor_response_strengths.odor.unique()
                # odors seem already sorted. assuming that for now.
                # (and with these assertions, only issue could be component ordering)
                assert is_mix(unique_odors[-1])
                assert not any(is_mix(c) for c in unique_odors[:-1])

                odor_palette = sns.color_palette('Set2', n_colors=len(unique_odors))
                odor_palette = dict(zip(unique_odors, odor_palette))
                # TODO what in here is emitting this warning? any issue? (or in some of
                # the code above related to this plot)
                # Warning: Boolean Series key will be reindexed to match DataFrame index

                # if not model_only, then KC & ORN only (so we can have col=stat for
                # version with model, and not have to remove empty axes)
                for flags in all_bool_combos_of_length_n(3):
                    include_nonresponses, model_only, log_yscale = flags

                    # only want to try log_scale for this one case
                    if log_yscale and not (model_only and include_nonresponses):
                        continue

                    data = allodor_response_strengths

                    if not include_nonresponses:
                        data = data[data.responded]

                    fname = f'perodor-response-strengths_{panel}_{mix}'

                    kws = dict(facet_kws)
                    if model_only:
                        data = data[data.source_type == 'model']
                        kws['col'] = 'stat'
                        if one_model_per_pnkc_class and simplify_models:
                            # TODO also define for other cases?
                            kws['row_order'] = ['uniform', 'claw']

                        unit_str = 'KC'
                        plot_dir = plot_root
                    else:
                        data = data[data.source_type.isin(('KCs', 'ORNs'))]
                        unit_str = 'KC|glomerulus'
                        plot_dir = model_root
                        kws['row_order'] = ['ORNs', 'KCs']

                    # TODO share this panel/mix stuff w/ other titles?
                    suptitle = (
                        f'{panel}/{mix}\nper-odor response strength distributions'
                    )
                    if not include_nonresponses:
                        suptitle += f'\nresponder ({unit_str}, odor) pairs only'
                    else:
                        suptitle += (
                            f'\nnon-responding ({unit_str}, odor) pairs included too'
                        )
                        fname += '_with-nonresponses'

                    if model_only:
                        fname += '_model-only'
                    else:
                        fname += '_kc-orn'

                    if log_yscale:
                        fname += '_logy'

                    fg = sns.FacetGrid(data=data, row=PNKC_CLASS_COL, hue='odor',
                        palette=odor_palette, sharex=False, sharey=False, **kws
                    )
                    def plot_one_odor_dist(**kwargs) -> None:
                        # TODO TODO fill=True if log_yscale?
                        return distplot(fill=False, **kwargs)

                    fg.map_dataframe(plot_one_odor_dist, x='value', alpha=0.7,
                        log_scale=(False, log_yscale)
                    )

                    for (i, j, hue), gdf in fg.facet_data():
                        ax = fg.axes[i, j]
                        if hue != 0:
                            continue

                        source_type = get_single_unique(gdf.source_type)
                        if source_type in ('KCs', 'model'):
                            unit_str = 'KCs'
                        else:
                            assert source_type == 'ORNs'
                            unit_str = 'ORNs'
                        ax.set_ylabel(f'density across {unit_str}')

                        stat = get_single_unique_stat(gdf, strip_mean_prefix=False)
                        ax.set_xlabel(stat)

                        xmin = None
                        xmax = None
                        if stat == 'num_spikes':
                            xmin = FIXED_NUM_SPIKES_XMIN
                            xmax = FIXED_NUM_SPIKES_XMAX
                            # TODO also include message like in other case using these?
                            # for responder only version maybe?

                        elif stat == 'mean_Fc_zscore':
                            curr_xmin, curr_xmax = ax.get_xlim()

                            # TODO maybe 4.5?
                            xmax = 5.0
                            if curr_xmax < xmax:
                                xmax = curr_xmax

                            if include_nonresponses:
                                xmin = -0.75
                                if curr_xmin > xmin:
                                    xmin = curr_xmin
                            else:
                                xmin = curr_xmin

                        elif stat == 'mean_peak_dff':
                            ax.set_xlabel(orn_response_desc)
                            xmin, curr_xmax = ax.get_xlim()
                            if curr_xmax > 3:
                                xmax = 3.0

                        if xmin is not None:
                            assert xmax is not None
                            ax.set_xlim([xmin, xmax])

                    fg.set_titles('{row_name}')
                    fg.add_legend()

                    # hspace: 0.2 very much not enough. 0.6 a bit too much. 0.4 good
                    #
                    # right: 0.85 good? for mode_only [=2 columns] case, yes, but it's
                    # thoroughly overlapping plot axes for 1 column case? trying to make
                    # it so legend isn't touching right. 0.7 for 1 col? just barely.
                    right = 0.82 if model_only else 0.675
                    fg.fig.subplots_adjust(hspace=0.4, right=right)

                    # fontsize=8 was still maybe a bit too small
                    fg.fig.suptitle(suptitle, y=1.05, fontsize=9)
                    savefig(fg, plot_dir, fname)

                # TODO try to move this call before `kc_panel is None` check
                # (or at least the subset of calls that does not involve model data)
                #
                # TODO TODO TODO maybe append all in loop, and then at end loop over
                # panels/mixes and plot all? (for all the main things plotted) could
                # then also serialize one master version of each analyzed quantity
                plot_all_comparisons_for(plot_response_strength_dists,
                    df=response_strengths, palette=source_palette
                )
                # only want to true separate row for this one
                plot_response_strength_dists(df=response_strengths, plot_dir=model_root,
                    palette=source_palette, source_types=('KCs', 'ORNs'),
                    orn_kc_on_diff_rows=True
                )
            else:
                # TODO memory profile? am i doing something stupid?
                # TODO actually check if _USE_KDEPLOT=True solves it? or never used
                # -f w/ new variant of the plot, and something else is intensive
                # about it?
                warn('skipping plot_response_strength_dists calls, because '
                    '-f/--full-model-params, and no other args restricting models '
                    'to a subset of that. would probably get killed b/c of OOM w/ '
                    'current implementation (i.e. "Terminated")'
                )

        # TODO TODO still regenerate for final thesis models, with -M models only, and
        # no other CLI args (should be done again now on 2026-07-28. just need to check
        # outputs)
        # TODO TODO (done, right? in outputs on /mnt/d0? was it using same
        # threshold?) + diagnostic(? meaning the claw dynamics and weights + other
        # KC/claw metadata, right?) plots for a few particular model variants?  (and
        # use same threshold as here!, or at least as one variant!)

        # TODO assert we have this column for both kiwi/control (and nothing else?)?
        if 'pair_dilution_factor' in kc_df.columns:
            # TODO just plot earlier, and never include this data in what gets put
            # into dicts to be analyzed later? idk
            # TODO could keep 5comp mixes if i update dilution factor to be accurate
            # for mix dilutions (and if i don't drop mix dilutions)
            diag_df = kc_df[kc_df.mix == 'binary'].copy()
            assert set(diag_df.pair_dilution_factor.unique()) == {0, 1, 2}

            # don't really get anything else out of plotting the normalized version
            # here, since not comparing to model data
            diag_df = diag_df[~diag_df.stat.str.startswith(NORM_PREFIX)].copy()

            # TODO make sure these are saved to model_root (not plot root) too
            plot_panel_stats_across_models(diag_df, panel,
                f'{suffix}_kc-only_pair-dilutions'
            )

            # TODO assert only things filtered here are in mix='binary' case?
            # doesn't really matter
            # TODO double check that (before we strip concs bove), 0 really is
            # the one w/ the highest concs (i did, but add assertion?)
            kc_df = kc_df[kc_df.pair_dilution_factor == 0]

        pdf = pd.concat([pdf, kc_df], ignore_index=True)
        del kc_df

        kc_pdf = pdf[pdf.source == 'KCs']
        # TODO maybe some / all of below should be moved into `if kc_panel is not None`
        # conditional immediately above?
        if panel not in NATMIX_PANELS:
            if kc_panel is not None:
                assert len(kc_pdf) > 0
                plot_panel_stats_across_models(kc_pdf, panel, f'{suffix}_kc-only')

            # TODO TODO is there not a model-only version of this call? if so, move
            # before `kc_panel is None` check (it's within a subcall to this. still
            # would like to reorganize such that it gets made when `kc_panel is None`)
            plot_panel_stats_across_models(pdf, panel, suffix)
        else:
            assert kc_panel is not None
            assert len(kc_pdf) > 0
            assert set(kc_pdf.mix.unique()) == {'binary', '5comp'}

            # TODO subsetting to 0 already done?
            # pair_dilution_factor is NaN for model data (and only model data)
            pdf = pdf[(pdf.pair_dilution_factor == 0) | pdf.pair_dilution_factor.isna()
                ].copy()

            model_pdf = pdf[pdf.source != 'KCs'].copy()
            # grouping by mix will drop model stuff, because it can't easily be assigned
            # just one value for that (would need to duplicate stuff shared between the
            # two)
            for mix, kc_mdf in pdf.groupby('mix'):
                assert (kc_mdf.source == 'KCs').all()
                plot_panel_stats_across_models(kc_mdf, panel, f'{suffix}_{mix}_kc-only')

                # TODO TODO is there not a model-only version of this call? if so,
                # move before `kc_panel is None` check (there is but i think it's buried
                # within this, not a separate call of plot_panel_stats.... fix)
                model_mdf = model_pdf[model_pdf.odor.isin(kc_mdf.odor.unique())]
                mdf = pd.concat([kc_mdf, model_mdf], verify_integrity=True)
                plot_panel_stats_across_models(mdf, panel, f'{suffix}_{mix}')

                kc_mdf = kc_mdf[kc_mdf.stat == 'mean_Fc_zscore'].copy()
                plot_kc_orn_eag_intensity_comparison(model_root, source_palette,
                    kc_mdf, orn_intensity, eag_intensity if mix == '5comp' else None
                )

    mean_model_response_rate = pd.concat(mean_model_response_rate_list,
        verify_integrity=True
    )

    mean_mixsupp = pd.concat(mean_mixsupp_list, verify_integrity=True).reset_index()
    # TODO TODO serialize this to parquet too
    to_pickle(mean_mixsupp, plot_root / 'mixsupp-avg.p', verbose=True)

    def add_model_suffix(x: str) -> x:
        if x in EXPECTED_NONMODEL_PNKC_VALS:
            return x
        else:
            return f'{x} model'

    for_avg_mixsupp_plot = mean_mixsupp.copy()

    # TODO define this from data, if i have something other than that
    pnkc_class_order = ['ORNs', 'uniform', 'claw', 'bouton', 'KCs']
    for_avg_mixsupp_plot = for_avg_mixsupp_plot.sort_values(by=PNKC_CLASS_COL,
        kind='stable', key=lambda x: x.map(pnkc_class_order.index)
    )

    for_avg_mixsupp_plot[PNKC_CLASS_COL] = for_avg_mixsupp_plot[PNKC_CLASS_COL
        ].map(add_model_suffix)

    avg_mixsupp_palette = {add_model_suffix(k): v for k, v in source_palette.items()}

    # TODO TODO want to plot the 'diag-binaries' stuff separately or nah? (and need to
    # normalize (w/ panel2kc_panel, if so?)?
    mean_mixsupp_natmix_only = for_avg_mixsupp_plot[
        for_avg_mixsupp_plot.panel.isin(NATMIX_PANELS)
    ]
    # initially tried one version across mixes, w/ row='mix', but didn't like
    # (would need work to get xticklabels right, and also would only want to sharey
    # within mix anyway)
    for mix in NATMIX_MIX_TYPES:
        mix_mean_mixsupp = mean_mixsupp_natmix_only[mean_mixsupp_natmix_only.mix == mix]
        for model_stat in MODEL_STAT_ORDER:
            # TODO TODO or just skip model_stat == 'num_spikes'? (it is clear that, esp
            # for claw model, that doesn't really make sense to use)
            mix_mean_mixsupp_onemodelstat = mix_mean_mixsupp[
                mix_mean_mixsupp.source.isin(('KCs', 'ORNs')) |
                (mix_mean_mixsupp.stat == model_stat)
            ]
            fname_suffix = f'_{mix}_model-{stat2fname_part(model_stat)}'

            title = f'{mix}'
            if model_stat == 'num_spikes':
                title += f'\n{model_stat=}'

            plot_avg_mixsupp(mix_mean_mixsupp_onemodelstat, plot_root, title=title,
                fname_suffix=fname_suffix, x=PNKC_CLASS_COL, hue=PNKC_CLASS_COL,
                palette=avg_mixsupp_palette, legend=True
            )

    # don't want to overwrite these outputs unless currently analyzing all models
    if unrestricted_full_model_params:
        # TODO TODO define model_mean_mix_supp from mean_mixsupp (-> delete def of that
        # above, after checking the two defs are equiv)
        model_mean_mix_supp = pd.concat(model_mean_mix_supp_sers, verify_integrity=True)
        assert (
            len(model_mean_mix_supp.shape) == 1 and model_mean_mix_supp.name == diff_col
        )

        # values should now be the diff_col values for each stat
        model_mean_mix_supp = model_mean_mix_supp.unstack('stat')
        assert set(model_mean_mix_supp.columns) == set(MODEL_STATS)
        # to indicate it's the mixture suppression (mix - max(components)) derived from
        # each stat, not just the raw stat itself
        model_mean_mix_supp.columns = 'mixsupp_' + model_mean_mix_supp.columns
        mixsupp_cols = list(model_mean_mix_supp.columns)
        # to override 'stat' we get from unstack, which is then confusing after
        # reset_index
        model_mean_mix_supp.columns.name = None

        # TODO rename non_mixsupp_cols now
        nonstat_cols = list(model_mean_mix_supp.index.names)
        model_mean_mix_supp = model_mean_mix_supp.reset_index()
        assert not model_mean_mix_supp.isna().any().any()

        model_acrosspanel_mean_mix_supp = model_mean_mix_supp[
            # not including diag-binaries* panels in mean, b/c they don't have nearly
            # the mean mixture suppression in real data as the natmix panels do
            # TODO maybe even sort by the difference between natmix and those? (or focus
            # on min few with diag-binaries mix suppression sufficiently close to 0?)
            model_mean_mix_supp.panel.isin(NATMIX_PANELS)
        ].groupby(model_cols)[mixsupp_cols].mean()

        model_mean_mix_supp = model_mean_mix_supp.set_index(model_cols)

        # TODO (delete) also include mean megamat 2h-whatever correlation (from
        # tuned model dir) for each of these (to see if any of these solve the
        # general problem? and fraction of segmenting cells?)
        # TODO (delete) or at least manually check what those outputs look like for
        # the best models here (not dramatically improved, at least as currently
        # calculated, or w/ spike counts + gaussian noise)

        for stat in mixsupp_cols:
            # TODO just sort by kiwi/control (doing that)? and another version by all?
            # one one version sorted just by binary stuff?
            sorted_by_stat = model_acrosspanel_mean_mix_supp.sort_values(by=stat,
                kind='stable'
            )
            model_dirnames = sorted_by_stat.index.map(model_ids)
            sorted_by_stat['model_dirname'] = model_dirnames

            sorted_mixsupp = model_mean_mix_supp.loc[sorted_by_stat.index].reset_index()
            # TODO why .reset_index()? (i always set it back on load anyway, right?)
            sorted_by_stat = sorted_by_stat.reset_index()

            curr_mixsupp_order_fname = mixsupp_parts2order_fname(stat,
                logistic_scaling_fname_part
            )
            fname_prefix = mixsupp_order_fname2shared_prefix(curr_mixsupp_order_fname)
            to_csv(sorted_mixsupp, model_root / f'{fname_prefix}.csv', index=False)
            to_parquet(sorted_mixsupp, model_root / f'{fname_prefix}.parquet')

            to_csv(sorted_by_stat, model_root / f'{fname_prefix}_panelmean.csv',
                index=False
            )
            mixsupp_order_parquet = model_root / f'{fname_prefix}_panelmean.parquet'
            to_parquet(sorted_by_stat, mixsupp_order_parquet)

            # TODO TODO just save plots if order is by same stat as hardcoded above?
            # or also save for `stat == mixsupp_prefix + 'num_spikes'`?
            if curr_mixsupp_order_fname != mixsupp_order_fname:
                # TODO TODO why am i not seeing this warning?
                warn('not making model param (ordered by mean mix supp.) plots, b/c '
                    f'current {stat=} not in {mixsupp_order_fname}'
                )
                continue

            model_order = sorted_by_stat
            for flags in all_bool_combos_of_length_n(3):
                show_per_panel_mixsupp, separate_binary_cbars, model_dirname_yticks = \
                    flags

                # would raise AssertionError in plot fn in this case (and if not, would
                # fail b/c attemping creation of duplicate plots, since
                # separate_binary_cbars doesn't do anything if
                # show_per_panel_mixsupp=False)
                if not show_per_panel_mixsupp and separate_binary_cbars:
                    continue

                # this will currently fail in at least one case...
                assert model_order is not None
                plot_model_params_ordered_by_avg_mixsupp(model_order, model_root,
                    title_suffix=logistic_scaling_title_str,
                    show_per_panel_mixsupp=show_per_panel_mixsupp,
                    separate_binary_cbars=separate_binary_cbars,
                    model_dirname_yticks=model_dirname_yticks,
                    # order already defined in model_order. this is only used if
                    # show_per_panel_mixsupp=True, to process the name of this and load
                    # similar CSVs with mixture suppression computed on different data
                    # subsets (or w/ diff scaling/stat)
                    mixsupp_order_parquet=mixsupp_order_parquet,
                )
    else:
        # TODO just define diff_col as a const anyway? (or once somewhere?)
        if diff_col is None:
            if mix_supp is not None:
                diff_col = get_diff_col(mix_supp)
            else:
                diff_col = 'mix_minus_comp-max'

        warn(f'not writing model order, sorted by mean {diff_col}, because need '
            '-f and NOT -m for that'
        )

    if len(natmix_panel_class_frac_list) > 0:
        # contains both model and KC 5comp kiwi/control data
        class_fracs = pd.concat(natmix_panel_class_frac_list, verify_integrity=True)

        # TODO also try to save to parquet
        to_pickle(class_fracs, model_root / 'class_fracs.p', verbose=True)

        # TODO TODO second log-yscale version of this? (easier. trying this)
        # TODO or use similar approach to other plot (plot_mean_and_counts) to
        # break Y axis (difficult b/c FacetGrid. going w/ log_yscale actually)
        #
        # TODO maybe, for myself, show one w/ two thresholds 0.8 / 1.5 on remy's data?
        #
        # TODO do for yang's data too somehow? maybe w/ a separate row for each mix?

        # TODO define Fc_zscore part from getting KC stat, not hardcoding
        title = (f'observed KC mean response rate: {kc_no_dilution_mean_resp_rate:.2g}'
            f'\n(with a mean Fc_zscore threshold of {NATMIX_KC_THRESH:.2f})'
        )
        title_y = 1.07

        title_for_model_kc = str(title)
        title_y_for_model_kc = title_y
        if one_model_per_pnkc_class:
            natmix_mean_model_response_rates = mean_model_response_rate.loc[
                list(NATMIX_PANELS)].groupby(level=PNKC_CLASS_COL, sort=False).mean()

            model_rr_strs = ['\nmean model response rates:']
            for pnkc_class, mean_resp_rate in natmix_mean_model_response_rates.items():
                model_rr_strs.append(f'{pnkc_class}={mean_resp_rate:.2g}')
            model_rr_str = ' '.join(model_rr_strs)
            title_for_model_kc += model_rr_str
            title_y_for_model_kc += 0.08

        assert 'mix' in class_fracs.index.names
        # TODO just assert the whole thing is not NaN?
        assert class_fracs.index.get_level_values('mix').notna().any()

        shared_kws = dict(
            hue=PNKC_CLASS_COL, palette=source_palette, alpha=model_alpha, ci=CI,
            # NOTE: model_marker_kws only goes to the stripplot calls inside this fn
            facet_kws=dict(height=4, aspect=1.2), model_marker_kws=model_marker_kws
        )
        for mix, mix_df in class_fracs.groupby(level='mix', sort=False):
            # TODO TODO say N flies (+ROIs?) (for each panel) somewhere. share code for
            # that from natmix_data/analysis.py?
            mix_suffix = f'_{mix}'
            source_type = mix_df.index.get_level_values('source_type')
            mix_df_model_and_kc = mix_df.loc[source_type.isin(('KCs', 'model'))]
            mix_df_kconly = mix_df.loc[source_type == 'KCs']
            mix_df_kc_orn = mix_df.loc[source_type.isin(('KCs', 'ORNs'))]

            # plot_response_class_summary will append '_logy' to fname_suffix if this is
            # True
            for log_yscale in (False, True):
                plot_response_class_summary(mix_df_model_and_kc, plot_root,
                    fname_suffix=mix_suffix, title=title_for_model_kc,
                    title_y=title_y_for_model_kc, log_yscale=log_yscale, **shared_kws
                )

                fname_suffix = f'{mix_suffix}_kc-only'
                plot_response_class_summary(mix_df_kconly, model_root,
                    fname_suffix=fname_suffix, title=title, title_y=title_y,
                    log_yscale=log_yscale, **shared_kws
                )

            # don't need a log_yscale=True version of this?
            fname_suffix = f'{mix_suffix}_kc-orn'
            plot_response_class_summary(mix_df_kc_orn, model_root,
                fname_suffix=fname_suffix, title=title, title_y=title_y,
                **shared_kws
            )
    else:
        warn('had no natmix panel class frac data! presumably because kiwi,control were'
            ' included in `-S <skip-panels>`'
        )
        assert set(NATMIX_PANELS) - skip_panels == set()

    plot_panel_stats_across_models(tdf, 'megamat', suffix)


if __name__ == '__main__':
    main()

