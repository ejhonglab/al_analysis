
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

from importlib_resources import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
# TODO replace w/ conditional import of seaborn / scipy as needed?
# if trying to run tests w/ valgrind, this import currently seems to raise:
# SystemError: initialization of beta_ufunc raised unreported exception
# from a from scipy.stats._boost.beta_ufunc import
# TODO can i assert we are not TYPE_CHECKING here, and then put the imports that would
# fail (for typing hinting commented below) behind that flag? and maybe define the
# types for type hinting conditional on PYTHONMALLOC and/or TYPE_CHECKING?
if os.getenv('PYTHONMALLOC') != 'malloc':
    import seaborn as sns
else:
    warnings.warn('could not import scipy (and dependent packages), because '
        'PYTHONMALLOC=malloc'
    )

from hong2p import olf, util, viz
from hong2p import io as hong2p_io
from hong2p.olf import format_mix_from_strs, solvent_str
from hong2p.roi import is_ijroi_named
from hong2p.pandas import addlevel, pd_allclose
from hong2p.viz import (dff_latex, map_each_series_to_rgb, rgb_tuple_df_to_3d_array,
    Ticklabels, is_cmap_diverging
)
from hong2p.types import (Axis, Color, CMap, Pathlike, ParamDict, DataFrameOrSeries,
    MplRotation
)

# Shared with al_analysis's model code (mb_model), so moved to hong2p (2026-11), and
# re-exported here so existing code importing these from al_util keeps working.
from hong2p.olf import (diag_panel_str, panel2name_order, panel_order,
    abbrev_hallem_odor_index
)
from hong2p.olf import sort_odors_by_panel as sort_odors
from hong2p.util import (read_json, bootstrap_seed, fly_cols, flyroi_cols,
    formatwarning_msg_only, in_pytest, is_pytest_capturing_stderr, warn, format_mtime,
    print_curr_mem_usage, n_choose_2, n_multichoose_k, corr_triangular,
    invert_corr_triangular, mean_of_fly_corrs
)
from hong2p.pandas import stack_keep_na, stack_drop_na
from hong2p.viz import (diverging_cmap, light_gray, diverging_cmap_tophalf,
    diverging_cmap_kwargs, plot_corr, _plot_responses_helper, plot_responses,
    plot_responses_err, plot_responses_and_corr, cluster_rois, rotate_xticklabels
)
from hong2p.io import (add_check_outputs_unchanged_CLI_flag_and_parse_args, CodeContext,
    produces_output, MultipleSavesPerRunException, _output_change_prompt_or_err,
    caller_info, to_csv, read_parquet, to_parquet, to_pickle, read_pickle, to_json,
    np_save, drop_old_odor_index_levels, read_csv, text_diff,
    _check_output_would_not_change, should_ignore_existing, makedirs, savefig,
    curr_proc_start_time, written_since_proc_start
)

# Settings and state of the output helpers (now in hong2p.io), which code sets from
# outside (e.g. `al_util.verbose = True`, or the -c flag setting
# `check_outputs_unchanged`). These are not copied into this module, but forwarded to
# hong2p.io on both read and assignment (see `_AlUtilModule`), so there is only ever one
# value. New code should set them on hong2p.io directly.
_forwarded_to_hong2p_io = frozenset({
    'verbose',
    'check_outputs_unchanged',
    # only defined once add_check_outputs_unchanged_CLI_flag_and_parse_args has run
    # with check_nonmain_option=True
    'check_nonmain_outputs_unchanged',
    'prompt_if_changed',
    'ignore_existing',
    'ignore_if_explicitly_requested',
    'also_ignore_if_ignore_existing_true',
    'ignore_existing_options',
    'save_figs',
    'plot_fmt',
    'exit_after_saving_fig_containing',
    'response_calc_params_json_name',
    'required_index_levels',
    '_consider_as_main',
    '_dirs_to_delete_if_empty',
    '_fn_name2seen_inputs',
    '_fn_name2wrapped_fn',
    '_all_seen_inputs',
    '_saved_path2last_save_code_context',
    '_savefig_seen_paths',
    '_save_fn_name2diff_fn',
    '_SERIES_NAME_PLACEHOLDER',
})


class _AlUtilModule(ModuleType):
    """Module type of `al_util`, forwarding `_forwarded_to_hong2p_io` names to hong2p.io.
    """
    def __getattr__(self, name: str) -> Any:
        # only called for names not found in this module's own namespace
        if name in _forwarded_to_hong2p_io:
            return getattr(hong2p_io, name)

        raise AttributeError(f'module {self.__name__!r} has no attribute {name!r}')

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _forwarded_to_hong2p_io:
            setattr(hong2p_io, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in _forwarded_to_hong2p_io:
            delattr(hong2p_io, name)
        else:
            super().__delattr__(name)

    def __dir__(self) -> List[str]:
        return sorted(set(super().__dir__()) | _forwarded_to_hong2p_io)


sys.modules[__name__].__class__ = _AlUtilModule


# TODO do just in main? (/ after some call?)
warnings.formatwarning = formatwarning_msg_only


# TODO test
# TODO type hint
def sign_preserving_maxabs(x):
    """Returns value with maximum absolute value (keeping its sign), for each column.

    Args:
        x: DataFrame (e.g. timepoints X ROIs)

    Returns 1D array with one value per column of `x`.
    """
    idxmax = np.abs(x).idxmax(axis=0)
    # TODO update w/ sam's version that should also work w/ >2D input?
    # not sure it also works for 2D input though... here's the code he used to make
    # dF/F images using [what should be an equivalent] calculation:
    # newshape = dff_response.shape[1:]
    # oldshape = [1] + [i for i in newshape]
    # maxabs_idxs = np.abs(dff_response).argmax(axis=0).reshape(oldshape)
    # maxabs_dff = np.take_along_axis(dff_response, maxabs_idxs, axis=0).reshape(newshape)
    # (and i think I'm fine leaving those images as computed with mean, instead of this
    # type of calculation)
    df = pd.DataFrame([
        x.loc[absmaxidx, idx] for idx, absmaxidx in zip(idxmax.index.values, idxmax)
    ])
    # TODO keep as pandas type (old stat=mean seemed to, as last bit of _checks=True
    # code in trial_response_traces seemed to assume Series types)?
    # TODO why .flatten? replace w/ .squeeze() at least, if that's the issue?
    return df.values.flatten()


# NOTE: need to manually switch between these two values depending on whether you want
# to use `al-analysis`'s `-R/--repro-paper-models` CLI argument
#
# use this value by default (all new analysis; except when using -R)
response_stat_fn = sign_preserving_maxabs
#
# use this value for -R
#
# mean was what I had used for a while (also with n_volumes_for_response=2, I believe),
# including to generate Remy-paper outputs, and inputs to modelling for that.
#response_stat_fn = np.mean

if response_stat_fn != sign_preserving_maxabs:
    warn('al_util.response_stat_fn is not the standard sign_preserving_maxabs!')

# TODO refactor to fn(s), so i can recompute this/these if al-analysis called w/ -R (to
# use old stat)? pretty much always compute only as needed, recomputing each time?
# TODO TODO probably just need to refactor most of this back to al_analysis? seems like
# some things use roi_plot_kws (like load_antennal_csv, which might not all want to
# import al_analysis?), but almost all variables defined between response_stat_fn and
# that are only used by al_analysis.py
if response_stat_fn.__name__ == 'mean':
    # TODO want to avoid just 'mean', to avoid confusion w/ 'mean ' prefixes that might
    # get prepended, once we start dealing with means over either trials or flies?
    # would get pretty verbose...
    # (currently, this just refers to mean over timepoints in a response window, used to
    # get one number for a given unit [e.g. glomerulus, pixel, etc] and trial)
    trial_stat_desc = 'mean'

# currently, np.max seems to have __name__ of amax, but checking both in case that isn't
# always true.
elif response_stat_fn.__name__ in ('max', 'amax'):
    trial_stat_desc = 'peak'

elif response_stat_fn.__name__ == 'sign_preserving_maxabs':
    trial_stat_desc = '±peak'

else:
    assert False, f'unrecognized {response_stat_fn.__name__=}'

# TODO TODO delete (output does not make that much sense. also already have per-panel
# trace zscoring imlemented now, from cached raw traces. output there also doesn't make
# much sense)
#
# NOTE: currently will warn + exit before running model, if this is True. assuming for
# now I probably don't want to run model with these inputs. (but can't currently tell if
# cached responses were computed that way...)
zscore_traces_per_recording: bool = False
#

if not zscore_traces_per_recording:
    response_desc: str = f'{trial_stat_desc} {dff_latex}'
else:
    # TODO TODO also mention the additional baseline subtraction step? add delta symbol?
    response_desc: str = f'{trial_stat_desc} Z-scored F'

# never saying "mean mean <x>", just specifying outer mean separately when
# trial_stat_desc == 'max'
if trial_stat_desc != 'mean':
    # going to make no effort to clarify whether it's a mean over trials or flies.
    # should be clear from plot, and will get too verbose + create too many variables
    mean_response_desc: str = f'mean {response_desc}'
else:
    mean_response_desc: str = response_desc


# TODO can i also set weight_divisor=20, _drop_glom_with_plus=False here, or need
# to not do either of those for uniform case? check latter in uniform case?
# (probably can't. uniform test at least loops over the two values of _drop_glom...
# currently, though maybe doesn't need to)
paper_repro_kws = dict(
    target_sparsity=0.0915, drop_kcs_with_no_input=False, hardcode_initial_sp=True,
    sp_lr_coeff=10.0, max_iters=10,
    # TODO rename hardcode_initial_sp as repro...<something> and then have it imply both
    # of these?
    binary_search_on_overshoot=False, linear_lr_falloff=False
)
hemibrain_paper_repro_kws = dict(
    # I would not normally recommend you hardcode any of these except perhaps
    # weight_divisor=20. The defaults target_sparsity=0.1 and
    # _drop_glom_with_plus=True should be fine.
    weight_divisor=20, _drop_glom_with_plus=False, **paper_repro_kws
)

# TODO adapt -> share w/ (at least) drop_redone_odors?
# TODO type hint Mapping? can it be Series or Dict?
# TODO define my own hong2p.types.DictOrSeries?
def format_panel(x) -> str:
    """Returns `x['panel']`, formatted for plot labels (e.g. as `vline_level_fn`).

    Returns '' if `x` has no (or an empty) 'panel', and 'diagnostics' for
    `diag_panel_str`.
    """
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


# TODO also take series ever? share type hint w/ format_panel, if i add one?
def roi_label(index_dict) -> str:
    """Returns `index_dict['roi']` if it is a named ROI, otherwise '' (e.g. numbered).
    """
    roi = index_dict['roi']
    if is_ijroi_named(roi):
        return roi
    # Don't want numbered ROIs to be grouped together in plots, as the numbers
    # aren't meanginful across flies.
    return ''


# TODO rename now that i'm also allowing input w/o date/fly_num attributes?
def fly_roi_id(row: pd.Series, *, fly_only: bool = False) -> str:
    """Returns ID str for plot labels, e.g. '<m-dd>/<fly_num>/<roi>'.

    Uses `row.fly_id` in place of date and fly_num, if `row` has a non-null 'fly_id'.

    Args:
        row: should have 'roi', and either 'fly_id' or 'date' and 'fly_num'

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


def _have_fly_cols(data: Union[pd.DataFrame, pd.Series]) -> bool:
    """Returns whether `data` has all `fly_cols` as index level names.

    For DataFrame input, also checks column level names and column labels.
    """
    if isinstance(data, pd.DataFrame):
        names_lists = [data.columns.names, data.columns, data.index.names]
    else:
        assert isinstance(data, pd.Series)
        names_lists = [data.index.names]

    return any(all(x in names for x in fly_cols) for names in names_lists)


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

    # TODO can the following values (from load_antennal_csv.py) also work here?
    # i assume not, at least not for hgroup_label_offset.
    #hgroup_label_offset=0.12,
    #vgroup_label_offset=0.03,

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


# TODO delete? sort_odors below not do what i wanted in some pair data stuff?
def sort_concs(df: pd.DataFrame) -> pd.DataFrame:
    return olf.sort_odors(df, sort_names=False)


# TODO flag to select whether ROI or (date, fly) take priority?
# TODO move to hong2p + test
def sort_fly_roi_cols(df: pd.DataFrame, flies_first: bool = False, sort_first_on=None
    ) -> pd.DataFrame:
    # TODO delete key if i can do w/o it (by always just sorting a second time when i
    # want some outer level)
    """Sorts column MultiIndex with `flyroi_cols` levels.

    Named ROIs (see `hong2p.roi.is_ijroi_named`) are sorted before numbered/autonamed
    ROIs, within each group of any higher priority sort keys.

    Args:
        df: data to sort by fly/ROI column values

        flies_first: if True, sorts on `fly_cols` columns primarily, followed by 'roi'
            ROI names. Otherwise, sorts on ROI names first.

        sort_first_on: sequence of same length as `df.columns` (or callable, called with
            a dict of column level names -> values, for each column), used as the
            highest priority sort key. Within each value of this key, sorts as without
            it.
    """
    # shallow copy, so setting temporary column levels below doesn't modify input
    # (it previously did)
    df = df.copy(deep=False)

    index_names = df.columns.names
    # TODO replace w/ def from flyroi_cols? or just also assert it's in flyroi_cols?
    assert 'roi' in index_names or 'roi' == df.columns.name

    levels = ['not_named', 'roi']
    # TODO replace date/fly_num usage w/ all fly_cols
    if 'date' in index_names and 'fly_num' in index_names:
        if not flies_first:
            levels = levels + fly_cols
        else:
            levels = fly_cols + levels

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

        if isinstance(sort_first_on, Callable):
            sort_first_on = [
                sort_first_on(dict(zip(df.columns.names, x))) for x in df.columns
            ]

        assert len(sort_first_on) == len(df.columns), (
            f'{len(sort_first_on)=} {len(df.columns)=}'
        )
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


megamat_odor_names = set(panel2name_order['megamat'])

def odor_is_megamat(odor: str) -> bool:
    """Takes odor str (with conc) like 'va @ -3' to whether it's among 17 megamat odors.
    """
    return olf.parse_odor_name(odor, require_conc=False) in megamat_odor_names


# TODO replace some use of this w/ diverging_cmap_kwargs?
# (e.g. in response matrix plots)
#cmap = 'plasma'
# to match remy
cmap: CMap = 'magma'


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

    Returns DataFrame indexed by ['date', 'fly'], with normalized (e.g. lowercase)
    column names. Missing 'driver' / 'indicator' values are filled with 'pebbled' /
    '6f'.
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


# TODO delete odor_min_max_scale if i don't end up using
# TODO TODO rename this, and similar w/ 'roi_' (any others?), to exclude that?
# what else would i be plotting responses of? this is the main type of response i'd want
# to plot...
# TODO TODO always(/option to) break each of these into a number of plots such that we
# can always see the xticklabels (at the top), without having to scroll up?
# TODO add option to chop off odor concentrations in odor matshow xticklabels IF it's
# all the same on the input data? maybe default to that?
def plot_all_roi_mean_responses(trial_df: pd.DataFrame, *, title: Optional[str] = None,
    roi_sort: bool = True, sort_rois_first_on=None, odor_sort: Optional[bool] = True,
    keep_panels_separate: bool = True, roi_min_max_scale: bool = False,
    odor_min_max_scale: bool = False, use_diverging_cmap: bool = True,

    # TODO delete hack! (?)
    yticklabels: Ticklabels = None,

    # TODO keep?
    avg_repeats: bool = True,

    single_fly: bool = False,
    odor_glomerulus_combos_to_highlight: Optional[List[Dict]] = None, **kwargs
    ) -> Tuple[Figure, pd.DataFrame]:
    # TODO rename odor_sort -> conc_sort (or delete altogether)
    """Plots odor x ROI data displayed with odors as columns and ROI means as rows.

    Args:
        trial_df: ['odor1', 'odor2', 'repeat'] index names and a column for each ROI.
            ['odor2', 'repeat'] are optional, and 'odor' may be used in place of
            'odor1'.

        roi_sort: whether to sort columns

        sort_rois_first_on: passed to sort_fly_roi_cols's sort_first_on kwarg

        odor_sort: whether to sort odors by concentration (via `sort_concs`)

        keep_panels_separate: if 'panel' is among trial_df index level names, and there
            are any odors shared by multiple panels, this will prevent data from
            different panels from being averaged together

        roi_min_max_scale: if True, scales data within each ROI to [0, 1].
            if `cbar_label` is in `kwargs`, will append '[0,1] scaled per ROI'.

        odor_min_max_scale: if True, scales data within each odor to [0, 1].
            if `cbar_label` is in `kwargs`, will append '[0,1] scaled per odor'.

        use_diverging_cmap: if True, uses `diverging_cmap_kwargs` (unless either
            min-max scaling option is used, where `al_util.cmap` is used instead)

        yticklabels: if None, ROI labels are from `fly_roi_id` (or just ROI names, if
            `single_fly=True`)

        avg_repeats: if False, keeps separate rows for each 'repeat' (rather than
            averaging over trials)

        single_fly: if True, labels rows with just ROI names (input should only have
            data from one fly)

        odor_glomerulus_combos_to_highlight: list of dicts with 'odor' and 'glomerulus'
            keys. cells where `odor1` matches 'odor' (with no odor in `odor2`) and `roi`
            matches 'glomerulus' will have a gray box drawn around them.

        **kwargs: passed thru to hong2p.viz.matshow

    Returns figure and DataFrame of plotted (trial-averaged) data, with ROIs as rows
    and odors as columns.
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
        mean_df = trial_df.groupby(level=avg_levels, sort=False).mean()

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
            if 'hline_level_fn' in kwargs and not kwargs.get('levels_from_labels',True):
                if all([x in trial_df.columns.names for x in fly_cols]):
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

    # TODO detect (using viz.is_cmap_diverging?)?
    if use_diverging_cmap:
        kwargs = {**diverging_cmap_kwargs, **kwargs}
        # center of diverging cmap should be white, so we'll use black lines here
        kwargs['linecolor'] = 'k'

    kws = dict(
        xticklabels=xticklabels, yticklabels=yticklabels,
        hline_group_text=hline_group_text, vline_group_text=vline_group_text,
    )
    kwargs = {**kws, **kwargs}
    fig, _ = viz.matshow(mean_df, title=title, **kwargs)

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
            # TODO also check odor_glomerulus_combos_to_highlight for 'odor' vs 'odor1'?
            # assuming for now it will always be the former
            odor = combo['odor']
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


def count_n_per_odor_and_glom(df: pd.DataFrame, *, count_zero: bool = True
    ) -> pd.DataFrame:
    """Returns number of non-NaN values for each row and ROI (e.g. # of flies).

    Args:
        df: rows (e.g. odors) X columns with a 'roi' level (e.g. fly-ROIs)

        count_zero: if False, exact 0.0 values are not counted (e.g. if they are just
            fill values). if True, they are counted, but warns if there are any.

    Returns DataFrame with same row index as `df`, and one column per ROI.
    """
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
    n_per_odor_and_glom = df.notna().T.groupby(level='roi', sort=False).sum().T

    return n_per_odor_and_glom


def get_n_glomeruli_per_fly(df: pd.DataFrame) -> pd.Series:
    """Returns # of glomeruli (ROIs w/ any non-NaN data) per fly, indexed by `fly_cols`.

    Args:
        df: should have `flyroi_cols` column levels
    """
    assert set(flyroi_cols) - set(df.columns.names) == set(), f'{df.columns.names=}'

    # to remove all-NaN columns added by fill_to_hemibrain, which should not be counted
    df = df.dropna(how='all', axis='columns')

    def count_glom_per_fly(x: pd.DataFrame) -> int:
        gloms = x.index.get_level_values('roi')
        assert not gloms.duplicated().any()
        return gloms.nunique()

    # (groupby axis= kwarg removed in pandas 3.0, hence the transposing)
    return df.T.groupby(level=fly_cols, sort=False).apply(count_glom_per_fly)


def glomeruli_per_fly_summary(df: pd.DataFrame) -> str:
    """Returns str like '30 to 38 (mean 34.5) glomeruli per fly', for `df` data.
    """
    n_glom_per_fly = get_n_glomeruli_per_fly(df)
    gmin = n_glom_per_fly.min()
    gmax = n_glom_per_fly.max()
    gmean = n_glom_per_fly.mean()
    return f'{gmin} to {gmax} (mean {gmean:.1f}) glomeruli per fly'


# TODO share default cmap/zero_color w/ other fns that pass them in to this?
def plot_n(counts: pd.DataFrame, *, cmap: CMap = 'cividis', zero_color: Color = 'white',
    from1: bool = True, roi_sort: bool = False, **kwargs) -> Figure:
    """Plots matrix of integer counts (e.g. # of flies), w/ a discrete colormap.

    Args:
        counts: as input to `plot_all_roi_mean_responses` (e.g. odors X ROIs)

        zero_color: color for counts below colormap range (e.g. 0)

        from1: if True, colormap starts from 1 (rather than min count), and min count
            is added to xlabel

        **kwargs: passed to `plot_all_roi_mean_responses` (overriding defaults from
            `roimean_plot_kws`)
    """
    # TODO if from1=True especially, put min N in text somewhere? xlabel?
    # (unless i change how cbar is displayed, to only start it from min n)

    # TODO even allow NaN?
    counts_0filled = counts.fillna(0)
    assert pd_allclose(counts_0filled, counts_0filled.astype(int)), (
        'input was not int-like'
    )

    # TODO allow passing in fixed max_n (as long as it's >= data max_n)?
    max_n = counts.max().max()
    assert max_n > 0, 'expected some non-zero counts'
    # discrete colormap: https://stackoverflow.com/questions/14777066
    cmap = plt.get_cmap(cmap, max_n)
    # want to display 0 as distinct (white rather than dark blue)
    cmap.set_under(zero_color)

    n_roi_plot_kws = dict(roimean_plot_kws)
    n_roi_plot_kws['cbar_label'] = 'number of flies (n)'

    # TODO (delete? still relevant?) why isn't 0 in the bar tho? if the data had 0,
    # would there be? vmin has to be > 0, so that zero_color set correctly via cmap's
    # set_under
    cbar_kws = kwargs.pop('cbar_kws', dict())
    n_roi_plot_kws.update(kwargs)

    min_n = counts.min().min()
    if from1:
        xlabel = n_roi_plot_kws.get('xlabel', None)
        min_n_str = f'min n={min_n}'
        if xlabel is not None:
            xlabel += f'\n{min_n_str}'
        else:
            xlabel = min_n_str
        n_roi_plot_kws['xlabel'] = xlabel

        min_n = 1

    # these three should override input
    n_roi_plot_kws.update(cmap=cmap, vmin=(min_n - 0.5), vmax=(max_n + 0.5))
    cbar_kws.update(ticks=np.arange(min_n, max_n + 1))

    # TODO more elegant solution (than use_diverging_cmap=False flag) -> delete (detect
    # whether cmap diverging inside plot_all*?)
    fig, _ = plot_all_roi_mean_responses(counts, use_diverging_cmap=False,
        cbar_kws=cbar_kws, roi_sort=roi_sort, **n_roi_plot_kws
    )

    return fig


def plot_n_per_odor_pair(counts: pd.DataFrame, **kwargs) -> Figure:
    """Plots square odor X odor matrix of counts (e.g. # of flies), via `plot_n`.

    Args:
        counts: row and column index should be equal, and both named 'odor'

        **kwargs: currently unused
    """
    # (nvm, currently just expecting precomputed square matrix of counts)
    # TODO allow either datefly or date,fly_num input, and mean over everything
    # else (if any other column levels)
    # TODO should i allow single levels that just *seem* like datefly too, w/o
    # explicitly being named as such? (as in current process_remy_tables.py code)

    # code below currently assumes this
    assert counts.index.name == 'odor'
    assert counts.columns.name == 'odor'

    assert counts.index.equals(counts.columns), 'expected square matrix'

    fig = plot_n(counts, yticklabels=lambda x: x.odor, hline_level_fn=lambda x: x)
    return fig


# TODO rename (either this or others) to be consistent about "plot_*" fns either saving
# outputs or not? use decorator to add save (option?) to plot fns that dont save
# (having all either just return fig, or at least having fig as first var returned?)?
# TODO TODO add arg to collapse over trials (+ also share that w/ odor-pair fn)
def plot_n_per_odor_and_glom(df: pd.DataFrame, *, input_already_counts: bool = False,
    count_zero: bool = True, try_collapse_odors: bool = False, title: bool = True,
    title_prefix: str = '', **kwargs) -> Tuple[Figure, pd.DataFrame]:
    """Plots number of non-NaN values (e.g. # of flies) for each odor and glomerulus.

    Args:
        df: data to count (via `count_n_per_odor_and_glom`), w/ odors as rows and a
            'roi' column level, or precomputed counts (if `input_already_counts=True`)

        count_zero: passed to `count_n_per_odor_and_glom`. if False, exact 0.0 values
            (e.g. fill values) are not counted.

        try_collapse_odors: if True, and if all odors have the same counts (across
            glomeruli), will plot just one column instead of a full matrix

        title: if True, adds title like 'sample size (n) per (glomerulus X odor)'
            (and a summary of glomeruli per fly, if counts not precomputed)

        title_prefix: added before title (requires `title=True`)

        **kwargs: passed to `plot_n`

    Returns figure and DataFrame of counts that were plotted.
    """
    if not input_already_counts:
        n_per_odor_and_glom = count_n_per_odor_and_glom(df, count_zero=count_zero)
    else:
        n_per_odor_and_glom = df

    # TODO at least for panels below, show min N for each glomerulus?
    # (maybe as a separate single-column matshow w/ it's own colorbar?)
    # (only relevant in plots that take mean across flies)

    if len(title_prefix) > 0:
        # TODO assert it doesn't already end in \n (or strip()?)
        title_prefix = f'{title_prefix}\n'

    unit_str = '(glomerulus X odor)'
    collapsed_odors = False
    if try_collapse_odors:
        first_row = n_per_odor_and_glom.iloc[0]
        assert first_row.index.name == 'roi', f'{first_row.index.name=}'

        if (first_row == n_per_odor_and_glom).all().all():
            collapsed_odors = True
            title_prefix += 'all odors had same # glomeruli\n'
            unit_str = 'glomerulus'
            n_per_odor_and_glom = first_row.to_frame(name='all_odors').T
            # currently need this so plot_all_roi_mean_responses won't complain
            n_per_odor_and_glom.index.name = 'odor1'

    if title:
        title = f'{title_prefix}sample size (n) per {unit_str}'

        # TODO try to fix this case? don't think we have what we need...
        # (would prob need to change input to not pass in pre-computed counts, if i
        # cared) (al_analysis.main all_final_pebbled_data...=True case has a plot_n...
        # call that hits this)
        if not input_already_counts:
            title += f'\n{glomeruli_per_fly_summary(df)}'

        # TODO de-dupe title w/ cbar_label? just title_prefix?
    else:
        assert len(title_prefix) == 0
        title = None

    fig = plot_n(n_per_odor_and_glom, title=title, roi_sort=True, **kwargs)

    if collapsed_odors:
        # there should be two axes, w/ second the colorbar axes w/ label='<colorbar>'
        ax = fig.axes[0]
        # TODO work?
        ax.xaxis.set_visible(False)

    # TODO delete
    # (was to draw numbers over the center of each cell, with the count)
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


# TODO TODO any reason to not just move all this extra behavior into viz fn?
# (was originally made in natmix_data/analysis.py, but then moved here to share w/
# model_yang_mixtures.py)
def add_legends_and_colorbars(*args, **kwargs) -> None:
    """Calls `hong2p.viz.add_legends_and_colorbars`, w/ default `legend_locations`.

    Default `legend_locations` are chosen for clustermaps with many `row_colors`.
    """
    # 'upper center' / 'center' generally won't work, and 'lower right' may overlap w/
    # clustermap cbar. these locations in priority order, chosen with current clustermap
    # plots [the ones with many row_colors] in mind.
    legend_locations = ['upper right', 'lower right', 'lower left', 'upper left',
        'center right', 'lower center'
    ]
    if 'legend_locations' not in kwargs:
        kwargs['legend_locations'] = legend_locations

    viz.add_legends_and_colorbars(*args, **kwargs)


# TODO TODO try to move this (+ additional special casing above) into hong2p.viz
def plot_cols_with_diff_colormaps(df: pd.DataFrame, fig_kws: Optional[Dict] = None,
    legend_and_cbar_kws: Optional[Dict] = None, **kwargs) -> Tuple[Figure, Axes]:
    """Plots `df` as an image, with values in each column mapped to colors separately.

    Args:
        df: data to map columns to different colormaps (some of which can be shared).
            see `map_each_series_to_rgb` for options and more details, in particular
            `name2palette`, which can be used to define colormaps to use for particular
            columns.

        fig_kws: passed to `plt.subplots()` call creating the figure

        legend_and_cbar_kws: passed to `add_legends_and_colorbars`. has internal
            default, if not passed.

        **kwargs: passed to `map_each_series_to_rgb`

    Returns a created Figure and the Axes with the main `imshow` data.
    """
    color_df, for_curr_legends, for_curr_cbars = map_each_series_to_rgb(df, **kwargs)

    if fig_kws is None:
        fig_kws = dict()

    fig, ax = plt.subplots(**fig_kws)

    # TODO factor this and the plotting below into a fn for plotting these types of
    # dataframes? (returning ax at least, maybe just the next two calls? maybe also
    # calling add_legends_and_colorbars [that need any of intermediate stuff?]?)
    #
    # since imshow/pcolormesh/similar aren't able to take 2D arrays of 3-tuples, but
    # need (x, y, c) 3D arrays
    arr = rgb_tuple_df_to_3d_array(color_df)

    # TODO add some of the viz.matshow features to viz.imshow, to also allow specifying
    # xticklabels in a similar manner?
    ax.imshow(arr)

    # TODO refactor the ticks->ticklabels thing? (to hong2p?) (or just combine by
    # passing labels as 2nd arg of set_*ticks() calls, which should be same?)
    # TODO assert color_df.columns is a single level str index?
    ax.set_xticks(np.arange(len(color_df.columns)))
    # these should already all be str variable names
    ax.set_xticklabels([x for x in color_df.columns])

    ax.tick_params('x', rotation=90)

    # width of colorbars, and hmargin largely to leave enough space between colorbars,
    # for the label on each to not overlap the next colorbar
    default_legend_and_cbar_kws = dict(width=0.01, hmargin=0.09)
    if legend_and_cbar_kws is None:
        legend_and_cbar_kws = dict()
    legend_and_cbar_kws = {**default_legend_and_cbar_kws, **legend_and_cbar_kws}

    # default cbar_fontsize is effectively 10
    add_legends_and_colorbars(fig, for_curr_legends, for_curr_cbars,
        **legend_and_cbar_kws
    )
    return fig, ax


# TODO maybe some of below (stuff re: remy's kc data) should be moved into
# mb_model, rather than here in al_util? (al_analysis can import from mb_model, mb_model
# just can't import from al_analysis)

# using importlib_resources now
# TODO move to top (+ caps)
data_root: Path = files('al_analysis.data')

sent_to_remy: Path = data_root / 'sent_to_remy'

# TODO TODO regenerate these (and recommit in new dir, leaving old data), so we also
# have newer parquet / response_calc_json / etc outputs?
# TODO TODO add tests we can re-run al_analysis (w/ appropriate response calcs) and
# regenerate this as well as similar paper outputs (w/ older response calc)
#
# Contains subdirs [kiwi_control|megamat|validation2]_signed-max, each with dF/F
# CSV/pickle outputs and plots committed.
signedmax_orn_dff_dir: Path = sent_to_remy / '2025-09-30_tom_orn_data_signed-max'

# TODO TODO add tests that we can successfully load data with all these fns (both dff
# and est spike deltas). will be mainly important when trying to move data handling from
# stuff in an editable repo on disk to files provided by importlib_resources.
#
# TODO add similar fns for megamat/control (and both for paper response calc and newer
# signedmax one)
# TODO use in tests to check we can repro mean_est_spike_deltas for all, and everything
# downstream
def load_natmix_dff(**kwargs) -> pd.DataFrame:
    """Loads committed kiwi/control (+ diagnostic panel) ORN dF/F (signed max response).

    Args:
        **kwargs: passed to `read_csv`
    """
    data_dir = signedmax_orn_dff_dir / 'kiwi_control_signed-max'

    # TODO also load parquet (and prefer that, if we have it). need to regen and commit
    # new outputs first
    # TODO factor out this name to share w/ mb_model/al_analysis
    csv_path = data_dir / 'ij_certain-roi_stats.csv'
    df = read_csv(csv_path, **kwargs)

    assert set(df.index.get_level_values('panel').unique()) == {
        diag_panel_str, 'kiwi', 'control'
    }
    return df


# TODO assert matches subset of
# data/internal/for_dff_to_spiking_fn/ij_certain-roi_stats.parquet?
# TODO use in tests
# TODO TODO add flag to load old stat computation version of this (for recreating
# outputs), or just leave that to the spike delta version of this fn? (can't pass into
# model_mb_responses tho..., at least without changing that)
# (that fn `mb_model.megamat_orn_deltas` also does not have a flag to load old data.
# see hemibrain repro test, which i assume references some committed version, presumably
# of the ORN deltas. oh nvm i have the paper_megamat_orn_deltas fn for that. still want
# a corresponding fn / flag here, which should be able to reproduce those)
# TODO TODO TODO presumably i have some output committed under data/ which is based to
# point to as "the paper version" of the dF/F. what is that version? or should i just
# recompute w/ mean stat? (again see scripts/repro_remy_paper_modeling.py, which was
# never completely successful. i think
# dasta/sent_to_remy/2023-10-29/pebbled_ij_certain-roi_stats.csv should do)
def load_megamat_dff(**kwargs) -> pd.DataFrame:
    """Loads committed megamat (+ diagnostic panel) ORN dF/F (signed max response).

    Args:
        **kwargs: passed to `read_csv`
    """
    data_dir = signedmax_orn_dff_dir / 'megamat_signed-max'
    csv_path = data_dir / 'ij_certain-roi_stats.csv'
    df = read_csv(csv_path, **kwargs)
    assert set(df.index.get_level_values('panel').unique()) == {
        diag_panel_str, 'megamat'
    }
    return df


# TODO assert matches subset of
# data/internal/for_dff_to_spiking_fn/ij_certain-roi_stats.parquet?
# TODO use in tests
def load_validation2_dff(**kwargs) -> pd.DataFrame:
    """Loads committed validation2 (+ diagnostic panel) ORN dF/F (signed max response).

    Args:
        **kwargs: passed to `read_csv`
    """
    data_dir = signedmax_orn_dff_dir / 'validation2_signed-max'
    csv_path = data_dir / 'ij_certain-roi_stats.csv'
    df = read_csv(csv_path, **kwargs)
    assert set(df.index.get_level_values('panel').unique()) == {
        diag_panel_str, 'validation2'
    }
    return df


# TODO implement this one too (may need to recalc and commit? or instead have flags for
# validation / megamat load fns?
# TODO rename load_paper_megamat_dff?
#def load_remypaper_dff(**kwargs) -> pd.DataFrame:


# TODO also anchor path to script dir? would only be to support running from elsewhere,
# which i prob don't care about
remy_data_dir: Path = data_root / 'from_remy'

# NOTE: contains sparsities in top level CSVs, as well as individual fly binarized
# responses in subdirectories
#
# downloaded from Dropbox folder:
# Remy/odor_space_collab/analysis_outputs/multistage/multiregion_data/\
#     response_breadth/by_trialavg_ref_stim/megamat
# (Remy sent me a link on Slack 2024-04-04)
remy_sparsity_dir: Path = (remy_data_dir /
    'response_rates/refstim__ep_at_-3.0__median__0.120-0.130'
)

# TODO refactor to share?
remy_date_col: str = 'date_imaged'
# she uses 'fly_num' same as I do, to number flies within each day (i.e. the numbers not
# unique across days). these two can generally be used to compute/lookup values for
# remy_fly_id.
remy_fly_cols: List[str] = [remy_date_col, 'fly_num']

# 0, 1, ..., 3 (or all 0 in one of the CSVs, by accident, but that CSV redundant
# anyway)
remy_fly_id: str = 'acq'

# e.g. '1-5ol @ -3.0'
remy_odor_col = 'stim'

remy_conc_str = ' @ -3.0'
# TODO refactor to share w/ some similar fns?
def _strip_remy_concs(x: Union[pd.Series, pd.Index]) -> Union[pd.Series, pd.Index]:
    assert x.str.endswith(remy_conc_str).all()
    return x.str.replace(remy_conc_str, '', regex=False)


# contains CSVs remy made from the pickles she sent earlier under 'by_acq'
# (which i couldn't load)
remy_binary_response_dir: Path = remy_sparsity_dir / 'by_acq_csvs'

# looks like the top-level sparsity CSVs are computed from peak amplitude alone
# (rather than options involving "std"), so only going to load these pickles
# (out of the 3 options in each fly_dir)
remy_fly_binary_response_fname: str = 'df_stim_responders_from_peak.csv'


# TODO add verbose kwarg and set False when calling for debug purposes from kc response
# loading code
def load_remy_fly_binary_responses(fly_sparsity_path: Path,
    acq_ledger: Optional[pd.DataFrame] = None, *, reset_index: bool = True,
    _seen_date_fly_combos=None) -> pd.DataFrame:
    """Loads single fly (tab separated) CSV to boolean (cell X odor) DataFrame.

    Args:
        fly_sparsity_path: CSV in a directory named like '<date>__fly<NN>__...'

        acq_ledger: if passed in, has `index.names == remy_fly_cols`, and has
            `remy_fly_id in df.columns`

            maps ['date', 'fly_num'] to int 'acq' (`remy_fly_id`) (0-indexed), which is
            added as a row index level.

        reset_index: if True, only return int 'cell' index. otherwise, retains all
            metadata columns in source CSV (i.e. row index names in the DataFrame)

        _seen_date_fly_combos: internal use only (set of (date, fly_num) loaded so far,
            to check no fly is loaded twice. updated in place.)
    """
    fly_dirname = fly_sparsity_path.parent.name

    if hong2p_io.verbose:
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
        fly_df = addlevel(fly_df, remy_fly_id, curr_acq)

        msg = f'{fly_dirname} ({remy_fly_id}={curr_acq})'
    else:
        msg = f'{fly_dirname}'

    if hong2p_io.verbose:
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
    """Loads Remy's binarized megamat KC responses, for each fly.

    Result is cached, so later calls return the same object (which should not be
    modified).

    Returns boolean DataFrame of cells X odors, with `remy_fly_id` ('acq') and 'cell'
    row index levels, and megamat odor names (without concentrations) as columns.
    """
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

    if hong2p_io.verbose:
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
    assert binary_responses.shape[1] == len(megamat_odor_names)

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
    # TODO delete? ideally, i'd use more flies, but i'm not sure if she calculated the
    # sparsity (esp not pegged to one odor, if this was) for the other partial-panel
    # flies she's now using for all megamat corr calculations in paper
    n_final_megamat_kc_flies = 4
    assert df[remy_fly_id].nunique() == n_final_megamat_kc_flies
    #
    assert set(x.stim.nunique() for _, x in df.groupby(remy_fly_id)) == {
        len(megamat_odor_names)
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

    # TODO TODO TODO save this again + compare to model response rates
    # (or did i have another plot directly making that comparison?)
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


# (not in this module's namespace, or reads would not reach hong2p.io)
assert not (_forwarded_to_hong2p_io & set(globals())), \
    sorted(_forwarded_to_hong2p_io & set(globals()))
