#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from pprint import pformat, pprint
from itertools import combinations, product
import shutil
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import (LogNorm, SymLogNorm, LinearSegmentedColormap,
    ListedColormap
)
from matplotlib.collections import PolyCollection
from matplotlib.container import BarContainer
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm

from hong2p.util import (pd_allclose, addlevel, add_group_id, reindex,
    subset_same_in_all_dicts
)
from hong2p import olf
from hong2p.olf import (parse_odor_name, parse_log10_conc, solvent_str, component_delim,
    format_odor, format_mix_from_strs, sort_odors
)
from hong2p.viz import dff_latex, add_group_labels_and_lines, map_each_series_to_rgb
from hong2p.util import symlink, is_scalar
from natmix import drop_mix_dilutions

from al_analysis import al_util
from al_analysis.al_util import (savefig, plot_responses, read_parquet, to_csv,
    to_parquet, data_root, fly_cols, flyroi_cols, warn, cluster_rois, _have_fly_cols,
    add_legends_and_colorbars, diverging_cmap, ParamDict, plot_cols_with_diff_colormaps,
    panel2name_order, load_natmix_dff, plot_corr, mean_of_fly_corrs, format_panel,
    mean_response_desc, plot_fmt
)
from al_analysis.mb_model import (megamat_orn_deltas, fit_and_plot_mb_model,
    megamat_orn_deltas, natmix_orn_deltas, get_thr_and_APL_weights, format_model_params,
    get_odor_fname_suffix, KC_ID, CLAW_ID, dict_seq_product, abbrev_model_id,
    read_params, exclude_params, calc_mix_suppression, get_diff_col, diff_col2desc,
    FULL_MODEL_KW_LIST, NoCachedModelOutputsError, logistic, summarize_response_classes,
    add_missing_cells_to_nonresponders, format_response_class, kc_type_hue_order,
    plot_response_class_summary, get_fly_color_series, KC_TYPE, count_flies_and_rois,
    get_fitmbmodel_default, TRY_ALL_MODELS_WITH, TRY_NONCLAW_MODELS_WITH,
    TRY_CLAW_MODELS_WITH, TRY_BOUTON_MODELS_WITH, drop_binaries_mixdilutions_and_pfo,
    drop_silent_model_cells, analyze_spatial_claws, model_pnkc_class,
    EXPECTED_MODEL_PNKC_CLASSES, REMY_KC_RESPONSE_THRESHOLD, NATMIX_ORN_RESPONSE_THRESH
)
from al_analysis.al_analysis import fill_to_hemibrain


# TODO use pre-existing const kw list vars in mb_model to replace some of what i have
# here now?
'''
test_with_connectome_vs_uniform_apl = [
    dict(weight_divisor=20),
    dict(one_row_per_claw=True, prat_claws=True),
    dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True),
]
# passing the CLI arg -f will use FULL_MODEL_KW_LIST (currently len 137) instead of this
# TODO TODO TODO replace these w/ something chosen from -M output
# (n_spikes_for_response=2 & target_sparsity=0.05, mainly), and add flag to restore this
# original variations w/ APL options and everything
SHORT_MODEL_TUNE_KWS = [
    # comparison for all other model cases, to see to what extent changes to PN>KC
    # weight matrix (and potentially other changes) matter
    dict(pn2kc_connections='uniform', n_claws=7),
] + dict_seq_product(test_with_connectome_vs_uniform_apl,
    [dict(), dict(use_connectome_APL_weights=True)]
)
del test_with_connectome_vs_uniform_apl
'''
# TODO TODO add flag to restore commented above?
# TODO TODO TODO check these are set i want to use from looking at -m output
nonshared_model_kws = [
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
SHORT_MODEL_TUNE_KWS = dict_seq_product(nonshared_model_kws,
    [dict(target_sparsity=0.05, n_spikes_for_response=2)]
)
# TODO assert the above are a subset of FULL_MODEL_KW_LIST? they are, right?

# TODO delete
# TODO this even help? (maybe? regen all w ~10. worked ok for 5comp control?
# wouldn't want much lower)?
#KC_ROW_COARSEN: int = 10
#
KC_ROW_COARSEN: int = None

# TODO TODO compare to 'auto' again?
#N_BINS: int = 50
# TODO TODO halve 50 for logistic_scaled_num_spikes stuff? how (w/o making distplot too
# context specific?). just make the plots w/ diff stats w/ separate calls?
N_BINS: int = 30

# TODO delete this flag if it wasn't the difference between plotting fns that was
# causing plot_response_strength_dists to get Terminated in `-f` case
# (ever make this plots w/ -f before? anything i can do differently?)
_USE_KDEPLOT: bool = False
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

    if 'x' in kwargs:
        assert isinstance(data, pd.DataFrame)
        x = kwargs['x']
        values = data[x]
    else:
        if isinstance(x, pd.DataFrame):
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


# TODO really not have something like this elsewhere? move to hong2p.olf (+ use
# elsewhere, if not)?
# NOTE: outside of this script, component_delim='+' (instead of imported
# component_delim=' + ') would also be in single odors like (+)-carene or whatever, so
# may want to default to one with spaces if refactoring
# may also need to check hardcoded list of full mix names (e.g. 'kiwi approx.'), if
# refactoring...
def is_mix(odor: str, *, component_delim: str = '+') -> bool:
    if 'mix' in odor or component_delim in odor:
        return True
    return False


# TODO use this str elsewhere?
PNKC_CLASS_COL: str = 'model_pnkc_class'
EXPECTED_NONMODEL_PNKC_VALS: Set[str] = {'KCs', 'ORNs', 'EAG'}

# TODO use this elsewhere this value is currently hardcoded
N_VARIANTS_DELIM: str = ' ('
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
        x.split(N_VARIANTS_DELIM)[0] in EXPECTED_MODEL_PNKC_CLASSES
    )

    if not is_model:
        assert x in EXPECTED_NONMODEL_PNKC_VALS, f'{x=}'

    return is_model


def has_model(df: pd.DataFrame) -> bool:
    """Checks `PNKC_CLASS_COL` is in `df.columns`, and whether it contains model values.

    Values matching or starting with one of `EXPECTED_MODEL_PNKC_CLASSES` count as model
    entries.

    Raises ValueError if unexpected values in this column, but NaN currently allowed
    (assumed non-model). Besides model values and NaN, only values in
    `EXPECTED_MODEL_PNKC_CLASSES` are allowed.
    """
    assert PNKC_CLASS_COL not in df.index.names, 'handle?'

    if PNKC_CLASS_COL not in df.columns:
        # TODO ever any cases where i'd want to check something else?
        return False

    # TODO delete. there is NaN sometimes currently (e.g. from some plot_fn calls)
    # TODO assert no NaN in this column? i have it filled pretty much everywhere,
    # right? (could just drop if it's an issue)
    #assert not df[PNKC_CLASS_COL].isna().any(), ('expected NaN values to be filled '
    #    'with "KCs" or something, for non-model data. could dropna here instead'
    #)

    # doing the dropna() since I could not always currently assert no NaN
    # (assuming NaN would only ever be for non-model stuff tho)
    vals = df[PNKC_CLASS_COL].dropna().unique()
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


def has_connectome_apl(df: pd.DataFrame) -> bool:
    # TODO also need to check index? maybe if this assertion fails
    assert 'connectome_apl' not in df.index.names, ('guess i need to check index '
        'too'
    )
    return 'connectome_apl' in df.columns and df.connectome_apl.any()


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

# whether each parameterization of model will have it's max odor response (w/in
# panel[+mix]) scaled to 1 (=True), or whether all models (w/in panel[+mix]) will be
# scaled such that max odor response across all models is 1 (=False).
NORM_PER_MODEL: bool = True

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
    # TODO TODO try NORM_PER_FLY again?
    if not NORM_PER_FLY:
        normed_df['value'] = normed_df.value / normed_df.value.max()
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

    if NORM_TO_FLYMEAN_MAX:
        normed_df['value'] /= calc_max_flymean(normed_df)
        assert np.isclose(calc_max_flymean(normed_df), 1)
        # TODO if i ever restore NORM_PER_FLY=True, would have to move the assertion
        # below into a conditional probably
        # TODO well at least it doesn't matter that i'm already doing the
        # `not NORM_PER_FLY` step above
        assert pd_allclose(raw_df.value / calc_max_flymean(raw_df), normed_df.value)

    normed_df['stat'] = f'{NORM_PREFIX}{stat}'
    return normed_df


# TODO TODO refactor to share w/ natmix_data/analysis.py i copied this from (though
# there is a lot more model specific code currently in there, and right now i just want
# to use this for real KC data here) (and was buried in cluster_rois_and_plot there, not
# a separate fn)
# TODO type hint for linkage?
def plot_hierarch_clustered_rois(plot_dir: Path, df: pd.DataFrame, fname_suffix: str, *,
    ignore_existing: bool = False, title: str = '', cbar_label: Optional[str] = None,
    dendrogram: bool = False, row_colors: bool = False, optimal_ordering: bool = True,
    row_coarsen_factor: Optional[int] = None, wPNKC: Optional[pd.DataFrame] = None,
    kc_spont_in: Optional[pd.Series] = None, **kwargs) -> Optional:

    hierarch_clust_fname = f'clust_hierarch_{fname_suffix}'

    # TODO TODO also put in titles, not just fnames
    if row_coarsen_factor is not None:
        hierarch_clust_fname += f'_row-downsample{row_coarsen_factor}'

    hierarch_clust_plot_path = plot_dir / f'{hierarch_clust_fname}.{plot_fmt}'

    if not (ignore_existing or not hierarch_clust_plot_path.exists()):
        # TODO include plot_dir.name?
        warn(f'not remaking {hierarch_clust_fname}.{plot_fmt} b/c existed and '
            f'{ignore_existing=}'
        )
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
    if fname_suffix == 'num-spikes':
        assert discrete
    else:
        assert not discrete, f'{fname_suffix=}'

    if fname_suffix == 'logistic-scaled-num-spikes':
        # may not be true? but i think it should be
        assert nonnegative
    #

    if nonnegative:
        # .N is the # of colors in list (256)
        # making a new colormap that is just the top (red) half of the old one
        colors = diverging_cmap(np.linspace(0.5, 1.0, diverging_cmap.N // 2))
        cmap = LinearSegmentedColormap.from_list(f'top_half_{diverging_cmap.name}',
            colors
        )
        # set_bad is all that's needed w/ LogNorm at least (0 values counted as bad)
        cmap.set_bad('w')
        cmap.set_under('w')
    else:
        cmap = diverging_cmap

    if not discrete:
        # TODO TODO what norm_kws do i want for nonnegative but not discrete (i.e.
        # logistic scaled spike counts)
        if not nonnegative:
            norm_kws = dict(norm='two-slope', vmin=vmin, vmax=vmax)
        else:
            # TODO TODO OK???
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
    # TODO TODO try SymLogNorm for raw spike count data? (wouldn't have to deal with
    # negative values there either, so could just do LogNorm? no need for diverging cmap
    # either. could do just the red half)
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

    # TODO either add a save= kwarg to disable saving (returning plot), or
    # refactor to have one fn that does most and returns plot?
    # or add kwarg to pass in a fn to call before saving?
    # (to avoid baking in the special case stuff to this fn that could otherwise be
    # shared w/ natmix_data/analysis.py /etc)
    savefig(cg, plot_dir, hierarch_clust_fname)

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


def main():
    # these definitions before CLI setup are here so doc for -m option has more info
    sorted_mixsupp_fname_prefix = 'model-mix-supps_sorted-by-mean-natmix_'
    order_by_mean_mixsupp_from = 'logistic_scaled_num_spikes'
    first_for_each = ['model_pnkc_class', 'connectome_apl']
    # TODO share w/ below? (bottom, near parquet saving)
    mixsupp_prefix = 'mixsupp_'
    def stat2fname_part(stat: str) -> str:
        return stat.replace('_', '-')

    mixsupp_col = f'{mixsupp_prefix}{order_by_mean_mixsupp_from}'
    stat_fname_part = stat2fname_part(mixsupp_col)
    # the versions of the CSV/parquet *without* '_panelmean' suffix include the mix
    # suppression values for all panels, for each model, although still all models
    # are ordered by mean natmix mixsupp (so excluding e.g. diag-binaries)
    # TODO change formatter so that fname doesn't get broken across lines?
    mixsupp_order_fname = (
        f'{sorted_mixsupp_fname_prefix}{stat_fname_part}_panelmean.parquet'
    )

    # TODO TODO TODO add flag to ignore LR cache (and check i can actually still
    # recreate at least the -m ones, if not all -f options that currently worked)
    # (or just use env var for that? need to test either way)
    # TODO TODO add option like -c/-C from al_analysis.py, to check outputs would be
    # same, without overwriting them?
    # TODO TODO add option to skip all model data?
    parser = ArgumentParser()
    # TODO refactor to share this (and -e/-x) w/ step_model_pn_apl, and
    # natmix_data/analysis.py?
    parser.add_argument('model_output_dirnames', nargs='?', help='comma separated '
        'list of substrings matching model output directory names \n(subdirectories of '
        '<model_output_root>/<panel> directories). \nsee also -e and -x.'
    )
    parser.add_argument('-c', '--use-cache', action='store_true', help='uses cache for '
        'all model outputs, when available. this may mean no models are re-run.'
    )
    parser.add_argument('-o', '--only-analyze-cache', action='store_true', help='will '
        'not run any models. will only load cached model outputs and analyze those. '
        'implies -c/--use-cache.'
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
    parser.add_argument('-d', '--save-dynamics', action='store_true', help='will write '
        'NetCDF files with many model dynamic variables (change plot root, since can '
        'take a lot of disk space)'
    )
    parser.add_argument('-i', '--ignore-existing', action='store_true', help='will '
        'overwrite existing hierarchichal clustering'
    )
    parser.add_argument('-m', '--max-supp-models-only', action='store_true', help='Will'
        ' only analyze the model with the most average kiwi/control (binary+5comp) mix '
        'suppression (i.e. the most negative), within each combination of '
        f"{first_for_each} (excluding 'connectome_apl' if -M/--simplify-models passed)."
        f'{mixsupp_order_fname} must exist under plot_root from a previous run, which '
        'defines the order based on mixture suppression calculated in that run.'
    )
    parser.add_argument('-M', '--simplify-models', action='store_true', help='Will '
        'exclude from analysis all connectome-APL models, as well as all prat-bouton '
        '/ nonclaw (e.g. wd20) models (as neither made a huge difference, at least in '
        'terms of average kiwi/control mix suppression), to simplify plots for thesis '
        '(and thus the writing about it).'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='print more')
    args = parser.parse_args()
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
    verbose = args.verbose
    quiet = not verbose
    # TODO (still true? delete?) set in fit_and_plot_mb_model instead? (currently errs
    # if try_cache=False [default] and cache_only=True [not default])
    if only_analyze_cache:
        use_cache = True

    if max_supp_models_only and not full_model_params:
        warn('-m implies -f, but -f not passed. setting full_model_params=True')
        full_model_params = True

    if model_output_dirnames is not None:
        model_output_dirnames = model_output_dirnames.split(',')

    if exclude_substrings is not None:
        exclude_substrings = exclude_substrings.split(',')
        assert len(exclude_substrings) == len(set(exclude_substrings))

    # TODO instead of RHS, check whether we skipped any models (for reasons other
    # than missing cache, which i think are counted separately anyway)? (by checking
    # after loop loading models)
    # TODO actually, just add assertion(s) that doing that would be consistent with this
    # def now? since i think i'd like to keep this simpler def up top now
    unrestricted_full_model_params = (
        full_model_params and not (max_supp_models_only or simplify_models or
            model_output_dirnames or exclude_substrings
        )
    )

    # calculate instead based on whether we have n-variant suffices?
    # can't! b/c why don't we have n_variants=2 for the two here w/ and
    # w/o APL:
    # ipdb> print(responded_cols[[PNKC_CLASS_COL, 'connectome_apl']
    #   ].drop_duplicates().to_string(index=False))
    # model_pnkc_class  connectome_apl
    #          uniform           False
    #          nonclaw            True
    #          nonclaw           False
    #             claw           False
    #             claw            True
    #           bouton            Tru
    only_analyzing_few_models = simplify_models and not (
        full_model_params and not max_supp_models_only
    )

    if not full_model_params:
        model_tune_kws = SHORT_MODEL_TUNE_KWS
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
                # TODO TODO TODO working? why in chosen_modeldirs CSV output tho (the
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

    if save_dynamics:
        model_root = Path('/mnt/d0/yang_mix_outputs').resolve()
        warn(f'writing to {model_root=} instead of usual under current directory, b/c '
            'saving (potentially large) dynamics outputs'
        )
    else:
        model_root = Path('yang_mix_outputs').resolve()

    model_root.mkdir(exist_ok=True)

    # directories under this used to just be under model_root, but especially when
    # running on FULL_MODEL_KW_LIST, it was getting pretty cluttered
    tune_root = model_root / 'megamat-tuned'
    tune_root.mkdir(exist_ok=True)

    simplify_models_fname_part = 'no-connectomeAPL-wd20-or-boutons'
    # NOTE: -m implies -f, so should just need these two fname parts
    max_supp_models_only_fname_part = 'max-mixsupp-only'

    subdirname_parts = []
    if simplify_models:
        subdirname_parts.append(simplify_models_fname_part)
    if max_supp_models_only:
        subdirname_parts.append(max_supp_models_only_fname_part)

    if unrestricted_full_model_params:
        assert not (simplify_models or max_supp_models_only)
        subdirname_parts = ['full-param-sweep']

    if NORM_PER_MODEL:
        subdirname_parts.append('norm-per-model')

    # TODO rename these two, to be more clear that plot_root is just used for plots that
    # can vary w/ e.g. simplify_model (so i can have two parallel subdirs of them),
    # whereas model_root currently has both plots and model output dirs
    plot_root = model_root
    if len(subdirname_parts) > 0:
        subdir_name = '_'.join(subdirname_parts)
        plot_root = model_root / subdir_name
        plot_root.mkdir(exist_ok=True)

    # otherwise we currently won't see the names of plots being saved printed in blue
    al_util.verbose = True

    # TODO move to module level, to share w/ CLI input checking (-> allow passing
    # order_by_mean_mixsupp_from as CLI arg, along w/ N to include?)
    MODEL_STAT_COLS = ['logistic_scaled_num_spikes', 'num_spikes']
    if max_supp_models_only:
        assert order_by_mean_mixsupp_from in MODEL_STAT_COLS
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
    model_str2abbrev = {m: abbrev_model_id(m) for m in model_strs}
    if verbose:
        # TODO delete? now that i'm mostly not using the abbrevs anyway (for now, at
        # least, after mostly using model_pnkc_class now)
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

    natmix_df = natmix_orn_deltas()

    tune_df = mdf
    del mdf

    series_list = []
    # make binary mixtures of synthetic diagnostics (150Hz spike delta to each of the
    # two glomeruli, for all combinations of them)
    gloms_to_mix = ['DM4', 'VM5d', 'DC3']
    for x in gloms_to_mix:
        ser = pd.Series(index=df.index.copy(), name=f'{x}-300 @ 0', data=0.0)
        ser.loc[x] = 300.0
        ser.name = ('syn-diag-binaries', ser.name)
        series_list.append(ser)

    glom_combos = list(combinations(gloms_to_mix, 2))
    for x, y in glom_combos:
        # TODO also sort x and y alphabetically here?
        # (less important prob, since presumably won't be comparing this directly to any
        # real data, at least not without transforming to odor names that best
        # approximate these?)
        mser = pd.Series(index=df.index.copy(), name=f'{x}-150/{y}-150 @ 0', data=0.0)
        mser.loc[x] = 150.0
        mser.loc[y] = 150.0
        mser.name = ('syn-diag-binaries', mser.name)
        series_list.append(mser)

    # make binary mixtures by combining the real diagnostic data for each of these
    # three odors (should target same glomeruli as above). again all pairwise combos.
    diag_subset = diags.loc[:, diags.columns.get_level_values('odor').isin(
        ('2h @ -6', 'farn @ -2', 'ma @ -7')
    )]
    odor2glom = {
        '2h @ -6': 'VM5d',
        'farn @ -2': 'DC3',
        'ma @ -7': 'DM4',
    }
    assert diag_subset.index.equals(df.index)

    new_panels = ('diag-binaries_mean', 'diag-binaries_max', 'diag-binaries_max-rest0')
    for panel in new_panels:
        # replacing 'glomeruli_diagnostics' panel w/ each of these new ones, so
        # single components will appear in all panel specific plots
        comp_df = addlevel(diag_subset.droplevel('panel', axis='columns'), 'panel',
            panel, axis='columns'
        )
        if panel == 'diag-binaries_max-rest0':
            comp_df.loc[~comp_df.index.isin(odor2glom.values())] = 0

        # technically not just a list of series anymore... but should still work
        series_list.append(comp_df)

    # TODO also have this control whether concs appear (in a consistent manner) for
    # components (+rename)
    leave_concs_in_mix = False
    for x, y in combinations(diag_subset.columns.get_level_values('odor'), 2):
        # just to sort alphabetically on component names, to be consistent w/ other code
        # (mainly code loading and processing odors in yang's data, which uses olf fns
        # that do that)
        x, y = sorted([x, y])
        # hack so some of the string processing code inside modelling doesn't err
        # ideally we'd just have mix_name=f'{x} + {y}'
        # TODO that hack actually necessary? test?
        if leave_concs_in_mix:
            mix_name = f'{x.replace(" @ ", "")} + {y.replace(" @ ", "")} @ 0'
        else:
            # TODO want space on either side of +?
            mix_name = f'{parse_odor_name(x)}+{parse_odor_name(y)} @ 0'

        d1 = diag_subset.loc[:, (slice(None), x)].squeeze()
        d2 = diag_subset.loc[:, (slice(None), y)].squeeze()
        both = pd.concat([d1, d2], axis='columns', verify_integrity=True)
        both.columns.names = ['panel', 'odor']

        mean_ser = both.mean(axis='columns')
        max_ser = both.max(axis='columns')

        g1 = odor2glom[x]
        g2 = odor2glom[y]
        max_zerod_ser = max_ser.copy()
        max_zerod_ser[~max_zerod_ser.index.isin((g1, g2))] = 0

        mean_ser.name = 'mean'
        max_ser.name = 'max'
        max_zerod_ser.name = 'max, non-cognate gloms 0d'
        to_plot = pd.concat([
                both.droplevel('panel', axis='columns'),
                mean_ser, max_ser, max_zerod_ser
            ], axis='columns', verify_integrity=True
        )
        mix_suffix = get_odor_fname_suffix(f'{x}_and_{y}')
        plot_responses(to_plot, model_root,
            f'diags_vs_constructed-mixtures{mix_suffix}',
            cbar_label='est. ORN firing rate delta (Hz)'
        )

        # first element of each of these should be in `new_panels` defined before this
        # loop (otherwise single components will not be added correctly)
        mean_ser.name = ('diag-binaries_mean', mix_name)
        series_list.append(mean_ser)
        max_ser.name = ('diag-binaries_max', mix_name)
        series_list.append(max_ser)
        max_zerod_ser.name = ('diag-binaries_max-rest0', mix_name)
        series_list.append(max_zerod_ser)

    test_df = pd.concat(series_list, axis='columns', verify_integrity=True)
    test_df.columns.names = ['panel', 'odor']
    assert not test_df.isna().any().any()

    # TODO delete! this is dropping the panels:
    # 'syn-diag-binaries', 'diag-binaries_mean', 'diag-binaries_max-rest0'
    print('remove dropping of other synthetic panels? (meh)')
    test_df = test_df.loc[:,
        test_df.columns.get_level_values('panel') == 'diag-binaries_max'
    ]
    #

    # TODO TODO how different are thr and APL scale params if tuning on natmix_df
    # instead of megamat_df?
    # TODO TODO what about if we tune on one of the diag-subset dfs?

    # TODO want to do anything about his other than fillna(0)? prob not
    # ipdb> natmix_df.index.difference(test_df.index)
    # Index(['DA1', 'DA4l', 'DA4m', 'V', 'VA1d', 'VA1v'], ...
    # ipdb> test_df.index.difference(natmix_df.index)
    # Index(['VA4'], dtype='object', name='glomerulus')
    test_df = pd.concat([test_df, natmix_df], axis='columns', verify_integrity=True)
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
    # TODO TODO switch between maxes depending on whether we are comparing to real KC
    # data or not ?
    response_rate_plot_max = 0.32

    # by this point, model_cols should all be present, and match those we would have
    # below, where these mixture-suppression sorted models are saved
    if not simplify_models:
        model_cols = ['model_pnkc_class', 'connectome_apl', 'source']
    else:
        model_cols = ['model_pnkc_class', 'source']

    # TODO use elsewhere too
    # TODO rename kc_panel_cols or something? 'mix' mostly (only?) used for kc case, at
    # least for 5comp + binary, right?
    panel_cols = ['panel', 'mix']
    withinpanel_odor_cols = ['pair_dilution_factor', 'odor']
    odor_cols = panel_cols + withinpanel_odor_cols

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

            assert not model_order.model_pnkc_class.str.startswith('bouton').any(), (
                'would need to manually drop these, if not dropped as a side-effect '
                'of dropping connectome_apl != False above'
            )

        # TODO refactor to share w/ other def after loop?
        # will take a couple seconds as-is (when run w/ -f [the "full" model list])
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
    for kws in tqdm(model_tune_kws, unit='model (on all panels)'):
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
            logistic_scaled_num_spikes = logistic(ss, k=2.0, x0=2.0, L=3.0)
            logistic_scaled_num_spikes = logistic_scaled_num_spikes.rename(
                'logistic_scaled_num_spikes'
            )

            # TODO don't pass under certain circumstances? don't pass ever?
            kc_spont_in = read_parquet(model_output_dir / 'kc_spont_in.parquet')

            nonsilent_cells: Optional[pd.Index] = None
            to_cluster_list = [ss, logistic_scaled_num_spikes]
            stat2to_cluster_nosilent = dict()
            for to_cluster in to_cluster_list:
                df = to_cluster.unstack(['panel', 'odor']).droplevel('model')

                # do currently need the internal .T to get this fn to work, and the
                # external one for drop_silent_model_cells to currently work
                # TODO TODO don't have this warn, at least not if we aren't going to
                # plot (or maybe always only warn on the first time? or never warn?)
                # TODO TODO TODO wait, so was model run with mix dilutions? (oh, yea.
                # nice)
                df = drop_binaries_mixdilutions_and_pfo(df.T).T

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
                df = sort_odors(df.T, panel2name_order=panel2name_order, warn=False)
                df = df.droplevel('panel')

                curr_title = str(title)
                if (stat == 'logistic_scaled_num_spikes' and
                    not cluster_on_logistic_scaled):

                    curr_title += '\nclustered on raw spike counts'

                elif stat == 'num_spikes' and cluster_on_logistic_scaled:
                    curr_title += '\nclustered on logistic-scaled spike counts'

                fname_suffix = stat.replace("_", "-")
                if not cluster_on_logistic_scaled:
                    # adding suffix in this case, since it seems to be worse
                    fname_suffix += '_clustered-on-raw-spikes'

                # TODO plot one version of these with and one without row_colors?
                # currently defaulting all to no row colors
                # TODO flag to cluster each independently? (as before)
                ret = plot_hierarch_clustered_rois(model_output_dir, df,
                    fname_suffix, ignore_existing=ignore_existing,
                    # when row_linkage=None (on first iteration, for 'num_spikes'
                    # currently, or whichever i put first), rows will actually be
                    # clustered
                    return_linkages=(row_linkage is None), row_linkage=row_linkage,
                    wPNKC=wPNKC, kc_spont_in=kc_spont_in, cbar_label=stat,
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

            # TODO TODO TODO -> also use this fn (below loop) to calculate stale dirs
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

    stat_names = model_roi_odor_dfs[0].columns
    for x in model_roi_odor_dfs[1:]:
        assert x.columns.equals(stat_names)
    stat_names = list(stat_names)

    if full_model_params:
        print('concatenating model dfs...', end='', flush=True)

    # need the .reset_index() since KC_TYPE not present for uniform model (but is for
    # all others), so can't concat based on indices, which produces bad output when not
    # all have same level names
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

        # TODO does simplify_models actually depend on -f? seems it might not? matter?
        if simplify_models:
            chosen_modeldirs_prefix += f'_{simplify_models_fname_part}'

        if max_supp_models_only:
            chosen_modeldirs_prefix += f'_{max_supp_models_only_fname_part}'

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

        # should have levels ['model_pnkc_class', 'source']
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
        for_index['model_pnkc_class'] = classes_with_fixed_counts.reset_index(drop=True)

        new_index = pd.MultiIndex.from_frame(for_index)
        assert model_order.index.equals(model_ids.index)
        model_order.index = new_index
        model_ids.index = new_index

    model_roi_odor_df['model_pnkc_class'] = model_roi_odor_df.model.map(
        lambda x: model_pnkc_class(x, pnkc2n_models=pnkc2n_models)
    )
    model_roi_odor_df['model_dirname'] = model_roi_odor_df.model.copy()
    model_roi_odor_df.model = model_roi_odor_df.model.apply(abbrev_model_id)

    # TODO need to do this in advance for mixes (or process here to strip conc,
    # leaving in format suitable for my conc stripping fn until here?)
    # i think there was concern the modelling code might still not handle odor strs
    # formatted as my code typically expects for mixtures? is that still true? fix that?
    # as a result, i'm processing diag-binaries_* mixes above to look like:
    # 'ma-7 + farn-2 @ 0' here, instead of 'ma @ -7 + farn @ -2'
    # (currently will just do in advance for mixtures, and continue processing like
    # above)
    model_roi_odor_df.odor = model_roi_odor_df.odor.map(parse_odor_name)

    # TODO (delete? already have in natmix_data/analysis.py i think. that using a
    # function shared here?) print # of spikes -> values, for first few # of spikes, and
    # print max before and after (after should approach L, right?)

    source_col = 'source'
    model_roi_odor_df = model_roi_odor_df.rename(columns={
        'model': source_col,
        'kc_id': 'roi',
    })

    # just since old df didn't have it. could keep?
    model_roi_odor_df = model_roi_odor_df.drop(columns='kc_type')
    # would need to subset to exclude KC_TYPE if we were not dropping above
    assert not model_roi_odor_df.isna().any().any()

    id_cols = ['panel', source_col, 'model_pnkc_class', 'model_dirname', 'odor', 'roi']
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
    tdf['model_pnkc_class'] = tdf.model.map(
        lambda x: model_pnkc_class(x, pnkc2n_models=pnkc2n_models)
    )
    tdf.model = tdf.model.apply(abbrev_model_id)
    tdf.odor = tdf.odor.map(parse_odor_name)
    tdf = pd.melt(tdf, id_vars=['model', 'model_pnkc_class', 'odor'],
        value_vars=['mean_num_spikes', 'mean_response_rate'], var_name='stat'
    )
    tdf = tdf.rename(columns={'model': source_col})

    if not simplify_models:
        tdf['connectome_apl'] = tdf[source_col].str.contains('_connectome-APL')
        tdf[source_col] = tdf[source_col].str.replace('_connectome-APL', '')
    # TODO assert no '_connectome-APL' in any of the strs if simplify_models=True?

    # TODO (delete?) print tdf (/ use to set / check ylim below)

    dilution_factor_delim: str = ' / '
    # TODO also sort components in fixed order? matter (prob not?)?
    def odor_sort_fn(x):
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

        return key

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

    # TODO TODO sort, to sanity check?
    unique_model_pnkc_classes = model_roi_odor_df['model_pnkc_class'].unique()
    assert 'KCs' not in unique_model_pnkc_classes, \
        f'{unique_model_pnkc_classes=}'

    # TODO TODO try to keep blue=uniform, orange=nonclaw, green=claw, red=bouton?
    # (even if simplify_models also removes nonclaw)
    source_palette = dict(zip(
        unique_model_pnkc_classes,
        sns.color_palette(n_colors=len(unique_model_pnkc_classes))
    ))

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
        # TODO TODO delete now? aren't we already subsetting effectively in load loop
        # above now? assert that?
        # TODO also say how many we are dropping (of each class?)
        warn('subsetting model_roi_odor_df to only the lowest avg kiwi/control mix '
            f'suppression model, within each combination of {first_for_each}!'
        )
        # TODO i assume this would also fail if there was mismatch between saved and
        # current models (shouldn't matter if i end up moving this plotting code to
        # where parquets are saved below, as planned)
        model_ids = model_ids.loc[model_order.index]
        # TODO this .loc would err if model_order has stuff not in current models,
        # right? want more explicit error message there?
        first_model_dirnames = set(model_ids.loc[model_order.iloc[first_indices].index])

        model_roi_odor_df = model_roi_odor_df[
            model_roi_odor_df.model_dirname.isin(first_model_dirnames)
        ].copy()

        model_params_list = []
        for model_id, model_dirname in model_ids.items():
            model_dir = tune_root / model_dirname
            assert model_dir.is_dir(), f'{model_dir=} did not exist'
            params = read_params(model_dir)
            # TODO any i don't actually wanna filter here? should be ok
            params = {k: v for k, v in params.items() if k not in exclude_params}
            model_params_list.append(params)

        model_params = pd.DataFrame(model_params_list,
            index=pd.Index(model_ids)
        )
        # since we will drop several separate columns below, which (across them)
        # contain the same info as this
        model_params['model_pnkc_class'] = model_ids.index.get_level_values(
            'model_pnkc_class'
        )

        # so we don't need to include both in plot
        assert model_params['one_row_per_claw'].equals(model_params['prat_claws'])

        # should just be 7 (for the single pn2kc_connections='uniform' case)
        assert model_params['n_claws'].dropna().nunique() == 1

        # only =20 for the 6 nonclaw models currently used
        assert model_params['weight_divisor'].dropna().nunique() == 1

        exclude = {'one_row_per_claw', 'fixed_thr', 'wAPLKC', 'n_claws',
            'weight_divisor'
        }

        # since these should all be contained in model_pnkc_class
        if not simplify_models:
            included_in_pnkc_class = {'prat_claws', 'prat_boutons', 'pn2kc_connections'}
        else:
            included_in_pnkc_class = {'prat_claws', 'pn2kc_connections'}

        exclude.update(included_in_pnkc_class)

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

        if not simplify_models:
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

        # ipdb> unique_model_pnkc_classes
        # array(['uniform (1 variant)', 'nonclaw (6 variants)',
        #        'claw (54 variants)', 'bouton (28 variants)'], dtype=object)
        #
        # getting the class values that contain the ' (<n> variants)' suffixes, just
        # from knowing the prefixes i want.
        claw_and_bouton_classes = set()
        if simplify_models:
            classes_to_check = ('claw',)
        else:
            classes_to_check = ('claw', 'bouton')

        for prefix in classes_to_check:
            # exact equality when no ' (<n> variants)' suffixes (e.g. when -m, now)
            mask = [x == prefix or x.startswith(f'{prefix} ')
                for x in unique_model_pnkc_classes
            ]
            assert sum(mask) == 1, f'{prefix=}: {sum(mask)=} != 1'
            claw_and_bouton_classes.add(unique_model_pnkc_classes[mask][0])

        claw_and_bouton_mask = model_params.model_pnkc_class.map(
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
        # ipdb> model_params.loc[model_params.model_pnkc_class.str.startswith('nonclaw')
        #   ].reset_index(drop=True).drop(columns='model_pnkc_class').T
        #                                0     1    2     3     4    5
        # pn_claw_to_apl               NaN   NaN  NaN   NaN   NaN  NaN
        # claw_dynamics                NaN   NaN  NaN   NaN   NaN  NaN
        # use_connectome_APL_weights  True   NaN  NaN  True  True  NaN
        # target_sparsity             0.05  0.05  NaN   NaN   NaN  NaN
        # n_spikes_for_response        2.0   2.0  2.0   2.0   NaN  NaN
        # allow_net_inh_per_claw       NaN   NaN  NaN   NaN   NaN  NaN

        name2palette = dict()

        # since source_palette doesn't currently have ' (<n> variants)' suffices here,
        # and want to use the suffices with the counts as they were in saved
        # model_params
        key_update = dict(zip(
            model_params[PNKC_CLASS_COL].str.split().apply(lambda x: x[0]),
            model_params[PNKC_CLASS_COL]
        ))
        new_source_palette = {key_update[k]: v for k, v in source_palette.items()}
        name2palette['model_pnkc_class'] = new_source_palette
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
            colors = [cm(0.0), cm(1.0)]
            ret = ListedColormap(colors, name=f'{len(colors)}colors_from_{name}')
            return ret

        name2palette['target_sparsity'] = name2listed_cmap('coolwarm')
        name2palette['n_spikes_for_response'] = name2listed_cmap('viridis')

        # without setting this index, assignment to create the new col seems to just
        # create all NaN
        ordered_mixsupp.index = model_params.index.copy()
        model_params[mixsupp_col] = ordered_mixsupp
        assert not model_params[mixsupp_col].isna().any()

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
        assert set(model_params.columns) - set(to_corr.columns) == {'model_pnkc_class'}
        for_order = to_corr.corr().loc[mixsupp_col].drop(mixsupp_col).abs(
            ).sort_values()
        col_order = list(for_order.index) + ['model_pnkc_class', mixsupp_col]
        model_params = model_params[col_order].copy()

        logistic_palette = 'Reds_r'
        name2palette[mixsupp_col] = logistic_palette
        # TODO use same cmap (sharing cbar) for other panel/mix values w/ same logistic
        # scaling based calculation, but use diff cbar (again shared across same things)
        # for calculation based on raw spike counts

        # TODO TODO try computing megamat corr, both adding noise and also logistic
        # scaling
        # TODO TODO factor into a fn for calculating evaluation metrics of model,
        # and share w/ other stuff in step_pn_apl_weights

        # (to what extent do these all line up? some models that are more consistently
        # better than others?)
        # TODO and w/ frac segmenting cells?
        #
        # TODO refactor to be able to make this type of plot for arbitrary inputs?

        # TODO refactor plotting so i can loop over combos of these three flags, and
        # make a plot for each?
        model_dirname_yticks = False
        show_per_panel_mixsupp = False
        # whether 5comp and binary mixtures will have their own colorbars (each using
        # the same palette though), each with a different range. only relevant if
        # show_per_panel_mixsupp=True.
        separate_binary_cbars = True

        extra_suffix = ''
        if model_dirname_yticks:
            extra_suffix = '_with-dirnames'

        if show_per_panel_mixsupp:
            extra_suffix += '_sep-stats'
            if separate_binary_cbars:
                extra_suffix += '_sep-binary-cbars'

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

        # TODO TODO plot binary mixsupp on its own scale? maybe also weight that
        # differently when taking mean? is there even as strong an ordering?
        if show_per_panel_mixsupp:
            # # of rows / columns / data only determined by whether '_panelmean' in
            # parquet fname suffix, and otherwise it's just the sort order that varies,
            # so don't need to load both of the non-panelmean versions to access all
            # per-panel/mix data (each contains mix supp calcs based on both stats)
            per_panel_mixsupp = read_parquet(
                model_root / mixsupp_order_fname.replace('_panelmean', '')
            )
            per_panel_mixsupp = per_panel_mixsupp.set_index(
                list(model_order.index.names) + panel_cols).rename_axis(columns='stat'
                ).unstack(level=panel_cols).loc[model_order.index]

            stat_metadata = per_panel_mixsupp.columns.to_frame(index=False)
            stat_metadata.stat = stat_metadata.stat.replace({
                'mixsupp_logistic_scaled_num_spikes': 'logistic',
                'mixsupp_num_spikes': 'n_spikes',
            })

            # TODO define natmix_panels earlier, and use that here?
            #
            # should just be excluding 'diag-binaries_max', which is probably not
            # particularly useful info (doesn't seem to vary much, and don't really care
            # that much if it does. should all be lower than mix suppression in
            # kiwi/control stuff anyway)
            to_drop = ~ stat_metadata.panel.isin(('kiwi', 'control'))

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
                assert panel in ('kiwi', 'control')
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

        plot_fname, rest = mixsupp_order_fname.split('_panelmean.parquet')
        assert rest == ''

        if simplify_models:
            warn(f'not making {plot_fname} plot, because -M. must run with -m '
                '(but still not -M), after generating order parquet from a run with '
                '-f (and no subsetting. unrestricted_full_model_params must be True)'
            )
        else:
            fig, ax = plot_cols_with_diff_colormaps(model_params,
                name2palette=name2palette,
                legend_and_cbar_kws=dict(
                    legend_locations=['lower right', pnkc_class_legend_loc],
                    width=cbar_width, hmargin=cbar_hmargin, left=cbar_left
                ),
                fig_kws=dict(figsize=figsize)
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
            #other_legends = {'model_pnkc_class', 'target_sparsity'}
            other_legends = {'model_pnkc_class',}
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
                f'mix suppression\ncalculated using {order_by_mean_mixsupp_from}',
                y=1.04
            )

            # TODO add lines separating stats from other things?

            savefig(fig, model_root, f'{plot_fname}{extra_suffix}', bbox_inches='tight')

            # TODO also make a separate plot w/ params for models that could not run
            # successfully, or were skipped (w/o mix suppression or other things we'd
            # need outputs for, obviously), just to quickly see which model types not
            # successfully being run

            # TODO and if i'm  not going to move this plotting to below, maybe i should
            # at least assert order would not change if recalculated below (even if not
            # re-saving CSVs/parquets? would have to copy relevant data instead of
            # redefining as subset tho)?

            # (delete. everything seems to check out...)
            # TODO how are the (now quite shifted) distributions in mix_minus_max_comp
            # stuff below consistent w/ the (seemingly unchanged) response strengths
            # plots. is the right data being fed into that? we haven't already
            # calculated that across all models or something (or the wrong set of
            # models?)? or is really just some weird disconnect between the analyses

    if NORM_PER_MODEL:
        model_norm_desc = 'within EACH model'
    else:
        model_norm_desc = 'across ALL models'

    if not NORM_PER_FLY:
        NORM_DESC = f'{model_norm_desc} and within KCs'
    else:
        # TODO ? delete? not used anyway
        NORM_DESC = f'{model_norm_desc} and per KC fly'

    # TODO rename KC_RESPONSE_COL or something?
    resp_col: str = 'mean_Fc_zscore'
    # TODO refactor?
    orn_resp_col: str = 'mean_peak_dff'

    compare_normalized = {
        # TODO separate mapping indicating what we should call this shared
        # thing? or maybe i should keep both names, esp if i want to compare
        # multiple normalized response strength metrics (i.e. adding a logistic
        # scaled version of # spikes)
        'mean_num_spikes': resp_col,
        'mean_logistic_scaled_num_spikes': resp_col,
    }

    kc_color = 'm'
    # TODO need to check if we actually load KC data? prob doesn't matter...
    source_palette['KCs'] = kc_color

    # TODO could use tab: red/purple/cyan/gray for one instead?
    source_palette['ORNs'] = 'tab:brown'
    #source_palette['EAG'] = 'tab:olive'
    source_palette['EAG'] = 'tab:gray'

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

    kc_err_alpha = 0.5
    # TODO TODO working? is color='k'? still seemed pretty light w/ 0.7
    kc_point_alpha = 0.7

    #  will be set to model_alpha_for_legend=0.7 in add_fixed_legend
    if not _USE_KDEPLOT:
        # TODO TODO which plot do i actually want this 0.3 for? want high alpha (maybe
        # even 1, for things that i'm using fill=False for now), esp things like
        # response-strength_dists_control.pdf
        #model_alpha = 0.3
        model_alpha = 0.5
    else:
        model_alpha = 0.5

    # 1.5 seems to be pretty much the same as the default being used for KC line in
    # mixsupp distplot
    model_linewidth = 1.5
    # 8.0 seems too small now for some reason. related all the "The figure layout has
    # changed to tight" warnings i'm seeing?
    # for some reason this needs to be much larger than the KC stripplot markersize
    # values
    # 15.0 was a bit too much
    model_markersize = 13.0

    marker_kws = dict(
        # default markeredgewidth seems ~2
        # TODO TODO set even lower markeredgewidth/markersize if full_model_params?
        # (would want to copy this dict to model_marker_kws, since this is
        # currently also used for KC data)
        # TODO TODO as long as i'm using strip plot and not pointplot, i don't think i
        # need this
        linestyle='none',
        # TODO take out of here? use model_markersize (always the largest one?)
        markersize=model_markersize
    )

    if unrestricted_full_model_params:
        # TODO TODO still use default alpha (and/or plot last) for uniform, in this
        # case, since there's only one point liek that and it's hard to find
        # TODO dodge/jitter more? other changes?
        # will be set to model_alpha_for_legend=0.7 in add_fixed_legend
        model_alpha = 0.15
        # TODO use this for everything? (markers included? comparable to
        # markeredgewidth?)
        model_markersize = 6.0

        # NOTE: different from (what I think can be constant linewidth for
        # model_marker_kws. this is used for actual line plots
        model_linewidth = 0.5

    # do need linewidth=1.5 (or something nonzero, to see anything for '+' marker.
    # edgecolor='face' is not enough). do need edgecolor='face' or else seaborn puts
    # black/grey border. also need to make points not filled.
    # i prefer edgecolor='none' to edgecolor='face' (w/ marker='.' at least)
    # TODO TODO TODO maybe i need to go back to markersize=model_markersize instead of
    # size=?
    # TODO TODO check i still need linewidth=1.5 and edgecolor='none?
    #model_marker_kws = dict(size=model_markersize, linewidth=1.5, edgecolor='none')
    model_marker_kws = dict(size=model_markersize, linewidth=1.5, edgecolor='none')

    def add_fixed_legend(g: sns.axisgrid.FacetGrid, df: pd.DataFrame,
        lines: bool = True, odor_linestyle: bool = False) -> None:

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
            df_pnkc_classes = set(df[PNKC_CLASS_COL].unique())
            # TODO do i never currently have variants here for some reason?
            # (no, we do w/ `-f` alone, but seemingly not with `-m -M` or `-m`. that
            # right? try w/ no args [other than `-o`]?)
            # and seems like if we have variants there, we also have them in
            # unique_model_pnkc_classes, so maybe there is no issue
            df_only_classes = df_pnkc_classes - set(unique_model_pnkc_classes)
            for x in df_only_classes:
                # stuff only in df should be non-model values like 'KCs'
                assert not pnkc_class_is_model(x)

            # maintains order in unique_model_pnkc_classes, in case i might care
            # TODO sort into a particular order? (should already be in order i want,
            # with uniform first and more complex models moving down) (if so, so it at
            # unique_model_pnkc_classes def anyway, not here)
            label_order = [x for x in unique_model_pnkc_classes if x in df_pnkc_classes]

            model_alpha_for_legend: float = 0.7

            for k in label_order:
                color = source_palette[k]
                assert type(color) is tuple and len(color) == 3, (f'{color=} was '
                    f'not a RGB tuple for {k=} (in source_palette)'
                )
                artist = None
                curr_line_kws = dict(line_kws)
                if k in legend_data:
                    existing = legend_data[k]
                    rgb, _ = artist_rgb_and_alpha(existing)

                    # TODO maybe need to check w/ np.allclose? (doesn't seem so. delete
                    # comment)
                    assert rgb == color, f'existing artist {rgb=} != desired {color=}'

                    # TODO anything else?
                    if isinstance(existing, (BarContainer, PolyCollection)):
                        # TODO TODO ever want line artist for PolyCollection? check
                        # get_facecolor() is not white / transparent or something?
                        # (and use line if so)
                        # TODO actually need label=k?
                        artist = Patch(facecolor=color, alpha=model_alpha_for_legend,
                            label=k
                        )
                    else:
                        assert isinstance(existing, Line2D), f'{type(existing)=}'

                        # TODO linewidth too? anything else?
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

            # TODO TODO TODO use this `if` code path for everything, after adapting to
            # handle stuff already present
            '''
            if not all(x in legend_data for x in label_order):
                assert not any(x in legend_data for x in label_order), 'handle?'
                # TODO (still?) fix how kc-only version of this plot is getting one
                # line for "uniform" model (w/ all model_pnkc_classes in legend),
                # instead of the one KC color w/ two odor linestyles intended (no odor
                # linestyles currently showing in legend there)
                # TODO actually, it's the odor-rows version (which i don't want anyway),
                # so missing the linestyle is expected, but still why do i have model
                # data?
                # TODO and it really just looks like the KC data offset by the
                # threshold... suspicious. doesn't look like the model data

                for k in label_order:
                    color = source_palette[k]
                    assert type(color) is tuple and len(color) == 3, (f'{color=} was '
                        f'not a RGB tuple for {k=} (in source_palette)'
                    )
                    artist = line_artist(alpha=model_alpha_for_legend, color=color,
                        **line_kws
                    )
                    legend_data[k] = artist
            else:
                for k, artist in legend_data.items():
                    # should skip 'KCs'/'ORNs'/'EAG'/etc
                    if not pnkc_class_is_model(k):
                        continue

                    # TODO will this cause me any issues in odor_linestyle=True case
                    # now? (ig i'm not using label for that, so it shouldn't)
                    assert k in label_order, f'{k=} not in model keys ({label_order})'

                    assert_msg = ('expected either separate alpha w/ RGB, or '
                        'get_alpha()=None with RGBA color'
                    )

                    # TODO TODO TODO is setting color/alpha here actually changing it in
                    # one of the axes too (and not just something in legend?) maybe i
                    # should make all myself, as above then? seems it is changing bottom
                    # right model dists to have higher alpha, in
                    # mix_minus_comp-max_dists_diag-binaries_max.pdf. only an issue for
                    # certain artist types?
                    # TODO TODO TODO can i just copy the artist first?
                    # TODO TODO TODO work? not for PolyCollection, at least...
                    # TODO TODO TODO was this also an issue w/ KDE plots before?
                    #artist = artist.copy()
                    warn('will probably screw up alpha on last axes... find other '
                        'solution!'
                    )

                    # TODO TODO TODO and the control/kiwi ones too...
                    set_color_fn = lambda x: artist.set_color(x)
                    set_alpha_fn = lambda x: artist.set_alpha(x)
                    try:
                        color = artist.get_color()
                        # assume that this one will always be available if first works?
                        alpha = artist.get_alpha()

                    # TODO maybe just skip these artists anyway? or some of them?
                    except AttributeError:
                        if isinstance(artist, BarContainer):
                            # should all be matplotlib.patches.Rectangle
                            rects = artist.get_children()

                            def get_single(name: str) -> Any:
                                unique = set(getattr(x, f'get_{name}')() for x in rects)
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
                            set_alpha_fn = lambda x: (r.set_alpha(x) for r in rects)

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
                        set_alpha_fn(model_alpha_for_legend)
                    else:
                        assert len(color) == 4, assert_msg
                        rgb = color[:-1]
                        set_color_fn(rgb + (model_alpha_for_legend,))
            '''

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

        # TODO only do this if i need to (i.e. if something is coming next?)?
        # (or is something always coming next?)
        empty_line = ''
        legend_data[empty_line] = title_proxy
        # TODO can i repeat an entry in label_order, or will i need to define something
        # like empty_line2 = ' '?
        label_order.append(empty_line)

        # since they are handled w/ a separate plot call that just plots KCs, and we
        # set legend=False on that
        # this even being hit? (yes)
        # (and commenting it did get rid of erroneous KCs line in model-only legends)
        # TODO why didn't it cause assertion below (about label_order matching
        # legend_data.keys()) to fail, when i commented it
        #'''
        if 'KCs' in df[source_col].unique() and 'KCs' not in legend_data:
            # TODO delete
            # TODO TODO TODO does df[source_col] really have 'KCs' when i don't want it
            # in legend (seems so)
            print('adding KCs line to legend')
            #
            # TODO pass in kc_alpha (but use this as default)?
            legend_data['KCs'] = line_artist(color=kc_color, alpha=kc_err_alpha,
                **line_kws
            )
            # TODO also do one for KC points? (if any. add flag?)
        #'''

        # TODO what's purpose of this? why put empty_line here?
        if 'KCs' in legend_data:
            label_order = ['KCs', empty_line] + label_order

        replace = {
            'bouton': 'bouton (no uniform APL)',
        }
        if not lines:
            replace.update({
                'KCs': 'KCs (95% CI on mean)',
            })

        legend_data = {
            (replace[k] if k in replace else k): v for k, v in legend_data.items()
        }
        label_order = [replace[k] if k in replace else k for k in label_order]

        have_connectome_apl = has_connectome_apl(df)
        if odor_linestyle:
            assert not have_connectome_apl, 'must drop the connectome_apl data'

            odor_artist_kws = dict(color='k', alpha=0.5)
            odor_artist_kws.update(line_kws)

            mix_kws = dict(linestyle=mix_linestyle)
            component_kws = dict(linestyle=component_linestyle)

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


    mean_prefix = 'mean_'

    # TODO rename stats->intensity? remove panel?
    def plot_panel_stats_across_models(df: pd.DataFrame, panel: str, suffix: str = ''
        ) -> None:

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

                # TODO warn/err if not startswith mean_prefix?
                assert k.startswith(mean_prefix)
                v2 = stat2ymax[k]
                if v > v2:
                    # TODO TODO is this even triggering? why is scale still broken?
                    warn(f'model max {k} ({v:.1f}) exceeded hardcoded stat2ymax value '
                        f'({v2:.1f}). replacing value in stat2ymax!'
                    )
                    stat2ymax[k] = v

        def plot_fn(data, *cols, **kwargs):
            is_normalized = False
            if 'is_normalized' in data.columns:
                is_normalized = data.is_normalized.unique()
                assert len(is_normalized) == 1, f'{is_normalized=}'
                is_normalized = is_normalized[0]

            # NOTE: previous code checking model_pnkc_class here did check whether the
            # column was all NaN, so has_model might need to deal with that actually
            # (assertion would trip), if that code was ever actually hit
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

                    kwargs_without_color = {
                        k: v for k, v in kwargs.items() if k != 'color'
                    }

                    # TODO when we have pair_dilution_factor (which will only be for a
                    # KC-only case), add groups and lines separating dilution factors,
                    # to replace current str suffixes for the /10 and /100 cases

                    err_kws = dict(linewidth=1.5, alpha=kc_err_alpha)

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

                    # TODO just leave color to palette again?
                    sns.pointplot(gdf[from_kcs], *cols, color=kc_color,
                        # need dodge=False here as long as we only have one hue level
                        # here, or else will get ZeroDivisionError
                        dodge=False, marker='o', markerfacecolor='none',
                        linestyle='none', seed=1, err_kws=err_kws,
                        # TODO move errorbar= def out ot module level / somewhere else,
                        # so i can share w/ text generating description of it? this
                        # should be default too
                        #$
                        # legend=False here hides legend inside this plot, but then
                        # there is also no marker next to 'KCs' in outside legend
                        # (will try to add handling for that outside)
                        errorbar=('ci', 95), capsize=0, ax=kc_ax, legend=False,
                        **kwargs_without_color
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
                    # TODO TODO hue/marker = 'mix'? (to debug intensity differences
                    # across 5comp/binary natmix recordings. pick a response threshold
                    # that gives same average response rate for overlapping odors [top 2
                    # components] in each mix=?)
                    if plot_individual_flies:
                        # TODO jitter/dodge more (than default)?
                        stripplot_kws = dict(color='k', legend=False,
                            # TODO use model_markersize here? (+ rename
                            # [point|dot]_markersize?) (no. for whatever reason, seems
                            # we need way less here than there)
                            # 3.5 was too small
                            alpha=kc_point_alpha, size=5.0
                        )
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

                        # TODO set jitter=False?
                        sns.stripplot(gdf[from_kcs], *cols, ax=kc_ax, **stripplot_kws,
                            **kwargs_without_color
                        )
                    else:
                        if verbose:
                            warn('not plotting individual flies, b/c '
                                f'{plot_individual_flies=}'
                            )
                    # TODO TODO say somewhere what the errorbars are 95% CI (on mean) by
                    # default

                if have_model:
                    # can't use float dodge w/ striplot unfortunately
                    # TODO ig i could make figure wider?
                    # jitter=1.0 is too much. 0.3 too much too, esp w/ aspect=1.1
                    sns.stripplot(gdf[~from_kcs], *cols, hue='model_pnkc_class',
                        # TODO move marker and model_alpha into model_marker_kws?
                        jitter=0.13, dodge=False, palette=source_palette, marker=marker,
                        alpha=model_alpha, ax=ax, **model_marker_kws, **kwargs
                    )


        def plot_stats(df: pd.DataFrame, extra_suffix: str = '') -> None:
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
                # TODO TODO TODO how is this getting set to just 'mean_response_rate'???
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
                # TODO strip 'mean_' before prepending NORM_PREFIX? (don't think so?)
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
                expected_stats = {'mean_response_rate',
                    kc_normed_response_stat, kc_response_stat
                }
                assert kc_stats == expected_stats
                # TODO delete
                #if have_model:
                #    expected_stats = {'mean_response_rate',
                #        kc_normed_response_stat, kc_response_stat
                #    }
                #    assert kc_stats == expected_stats
                #else:
                #    # TODO what is correct here?
                #    print('what is right here?')
                #    #assert expected_stats - kc_stats <= {kc_normed_res}

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
                if model_response_stat.startswith(mean_prefix):
                    model_response_stat = model_response_stat[len(mean_prefix):]

                df['stat'] = df.stat.replace(replace_dict)

                col_order = [shared_norm_stat_name, 'mean_response_rate']
                row_order = [False, True]
                facet_kws = dict(row='is_normalized', row_order=row_order)

            if (kc_response_stat is not None and
                kc_response_stat.startswith(mean_prefix)):
                kc_response_stat = kc_response_stat[len(mean_prefix):]

            assert set(df.stat.unique()) == set(col_order), \
                f'{set(df.stat.unique())=} != {set(col_order)=}'

            g = sns.FacetGrid(data=df, col='stat', col_order=col_order, sharey=False,
                aspect=1.1, **facet_kws
            )
            g.map_dataframe(plot_fn, x='odor', y='value')

            if have_model:
                add_fixed_legend(g, df, lines=False)

            assert g.col_names == col_order, f'{g.col_names=} != {col_order=}'

            # if only columns, axes.shape[0] should be 1 (w/ number of cols in shape[1])
            assert len(g.axes.shape) == 2, f'{g.axes.shape=}'

            all_suffix = f'{suffix}{extra_suffix}'
            plot_dir = plot_root
            if 'kc-only' in all_suffix:
                plot_dir = model_root
            fname = f'{panel}{all_suffix}'

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
                    if title.startswith(mean_prefix):
                        title = title[len(mean_prefix):]
                    # TODO assert it doesn't start with NORM_PREFIX?

                    if (panel in natmix_panels and 'response_rate' in stat_col and
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
            assert len(g.axes.shape) == 2 and ncols > 1, f'{g.axes.shape=}'
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

            # normalize_fname=False to not convert '__' -> '_'
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
                # (sorting screwed up as-is, and probably still need to solve that
                # separately either way)
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
            # TODO TODO assert that? is that really the only case that reaches this part?
            plot_stats(df, '')
        else:
            if (~ is_model).any():
                plot_stats(df[~is_model | ~raw_num_spikes], '_vs_kcs_logistic-scaled')
                plot_stats(df[~is_model | ~logistic_scaled], '_vs_kcs')

            if logistic_scaled.any():
                plot_stats(df[is_model & ~raw_num_spikes], '_logistic-scaled')

            plot_stats(df[is_model & ~logistic_scaled])
            # TODO is this just missing from megamat tdf or what?

        # TODO or just compare all pairs of normalized things (across model vs
        # fly)? will only be one pair initially anyway


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


    def strip_concs(odors: Union[pd.Index, pd.Series]) -> Union[pd.Index, pd.Series]:
        return odors.map(olf.strip_concs_from_odor_str
            ).str.replace(component_delim, '+', regex=False)


    def process_yang_odors(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # TODO also use that leave_concs_in_mix flag? (which currently doesn't support
        # leaving concs in components anyway...)
        without_concs = strip_concs(df.odor)

        nonvalue_cols = None
        if without_concs.nunique() < df.odor.nunique():
            # TODO include more detailed info, similar to some stuff currently
            # output in load/preprocess yang data fns?
            warn('averaging over multiple concs for some odors in KC data! see load/'
                'preprocess output above for more details.'
            )
            nonvalue_cols = [x for x in df.columns if x != 'value']
            assert not df[nonvalue_cols].duplicated().any()

        df['odor'] = without_concs
        # TODO assert not None?
        if nonvalue_cols is not None:
            assert df[nonvalue_cols].duplicated().any()
            df = df.groupby(nonvalue_cols, sort=False).value.mean(
                ).reset_index()

        # necessary for plots to still have odors in order i want
        df = df.sort_values(by='odor', kind='stable', key=odor_sort_fn)
        # TODO delete? shouldn't matter
        df = df.reset_index(drop=True)
        #
        return df


    # TODO factor to module level? (would then also have to pass in dataframes from
    # load_* fn)
    def get_yang_panel_means(panel: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        flyavg_ser = flyroi_ser.groupby(group_cols).mean().rename(resp_col)
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
        flyavg_df = process_yang_odors(flyavg_df)

        # TODO normalize yang's stuff outside here, so we can do it across panels?
        # even want to do it across panels? i don't think so...
        return flyavg_df


    # TODO move to module level?
    def mix_supp_list2flystat_df(mix_supp_dfs: List[pd.DataFrame], stat: str = resp_col
        ) -> pd.DataFrame:
        mix_supp = pd.concat(mix_supp_dfs, verify_integrity=True).reset_index()
        diff_col = get_diff_col(mix_supp)

        assert 'stat' not in mix_supp.columns
        # TODO problem that it isn't previxed w/ 'mean_'
        # (will probably have to change handling of model ones anyway, for same reason,
        # if theres an issue)
        mix_supp['stat'] = stat

        assert 'value' not in mix_supp.columns
        # TODO delete
        #return mix_supp.rename(columns={diff_col: 'value'})
        return mix_supp

    MIX_SUPP_IN_RESPONDERS_ONLY: bool = True

    # TODO also do something with the other panel in yang_dfs? (natmix-top2-dilute)
    # (i think those concs are too low to be able to compare to any orn data we really
    # have currently... since the ramp experiments i did were not of good quality)
    # TODO could at least lump the data for that one mix that overlaps between that
    # panel and diag-binaries into diag-binaries (would probably want to do when
    # initially loading yang data, and maybe also exclude from natmix-top2-dilute at the
    # same time)
    # TODO rename flyodor->flyavg_odor?
    # TODO TODO might i also want to store fly odor stats that are also per ROI?
    # TODO TODO TODO want to do any preprocess for that (storing stuff behind a new
    # panel2kc_flyroi_odor_stats, or just calc directly from yang_df/mdf?)
    panel2kc_flyodor_stats: Dict[str, pd.DataFrame] = dict()
    # TODO rename fly->flyroi?
    panel2kc_fly_stats: Dict[str, pd.DataFrame] = dict()
    # stats like max intensity for each (ROI, odor), to make e.g. distributions of odor
    # intensity for mixes vs components
    panel2kc_flyroi_odor_stats: Dict[str, pd.DataFrame] = dict()

    panel2orn_fly_stats: Dict[str, pd.DataFrame] = dict()
    panel2orn_flyroi_odor_stats: Dict[str, pd.DataFrame] = dict()

    for panel in ('diag-binaries',):
        yang_flyavg_df = get_yang_panel_means(panel)
        panel2kc_flyodor_stats[panel] = yang_flyavg_df

        flyroi_ser = get_yang_panel_ser(panel)
        # TODO make sure this doesn't cause issues w/ calculating mix
        # suppression (it will make farn -2.5 and -3, both comps and in mixes, look the
        # same, potentially w/o ROIs having same meaning across them? see _check_farn
        # code)
        # TODO maybe drop to only farn -3 instead? seems all flies should have that
        # (from _check_farn code) (compare output from current method to dropping farn
        # -2.5)
        flyroi_df = process_yang_odors(flyroi_ser.rename('value').reset_index())

        # TODO TODO is this consistent w/ how i'll handle model diag mixes tho
        # (prob not?)? also computed across all components/diag mixes, or just within
        # each?
        # TODO TODO instead, store responder mask alongside (in?) whatever i'll use
        # to calculate mix_suppression, and then calculate which "non-responders" to
        # drop down there?
        panel_responded = yang_bin_df.loc[panel].dropna(how='all', axis='columns')
        panel_responders = panel_responded.T.any()
        panel_responders = addlevel(panel_responders, 'panel', panel)

        panel_responded.columns = strip_concs(panel_responded.columns)
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
        flyroi_odor_stats['stat'] = resp_col

        flyroi_odor_stats = flyroi_odor_stats.set_index(panel_responded.index.names)
        # TODO how is this used later? delete?
        flyroi_odor_stats['responded'] = panel_responded.loc[flyroi_odor_stats.index]
        flyroi_odor_stats = flyroi_odor_stats.reset_index()
        # TODO TODO (deletet? done?) also group odors with a 'mix' level, as for
        # kiwi/control data? am i already doing that in some places? move handling
        # earlier to share w/ per-ROI clustering yang_df? (or even care to cluster each
        # separately? i guess i might need to depending on which flies have what?)
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
            c1, c2 = mix.split('+')
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

            # TODO TODO also group into response classes here, before calculating
            # mix suppression? (changing that code to work w/ 2 components in process)

            mix_supp = calc_mix_suppression(mix_df)
            mix_supp['mix'] = mix
            mix_supp = mix_supp.set_index('mix', append=True)
            mix_supp_dfs.append(mix_supp)

        mix_supp = mix_supp_list2flystat_df(mix_supp_dfs)
        panel2kc_fly_stats[panel] = mix_supp

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
    # TODO TODO (delete? done?) also use this as denominator for sparsity (and at least
    # do that for the binary mix one too, if not for any response class stuff)
    fly2n_total_rois_5comp = read_parquet(
        remy_dir / f'{prefix}5comp{fly2n_total_rois_suffix}'
    )
    # TODO TODO TODO maybe i want to analyze these actually? at least for correlation /
    # breadth? just exclude from most existing analyses (+ sort towards end, from
    # highest to lowest conc, w/in existing sort key fn, if using)
    # TODO TODO TODO if i'm not calling this on ORN data, what is dropping this from
    # kiwi/control model input (/output)?
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
    # TODO TODO TODO also do response class analysis on the binary data
    # (including lower concentrations in diag_df)

    def rename_natmix_odors(df: pd.DataFrame, level='odor') -> pd.DataFrame:
        # TODO refactor to share processing of odor names that has happened to
        # natmix_df contents (that's precomputed, right? what computed it?)
        # (eh, whatever. i've already done what i need here)
        # TODO find solution. don't want to plot the '@ 0' for the mixes anywhere,
        # unless also showing dilutions (would have to handle at plot time tho, unless
        # the data being processed will never have the dilutions...)
        # (maybe it's ok to omit the '@ 0' tho..., when comparing too)
        # TODO need to restore for some reason?
        #return df.rename({'cmix @ 0': 'cmix0 @ 0', 'kmix @ 0': 'kmix0 @ 0'},
        return df.rename({'cmix @ 0': 'cmix', 'kmix @ 0': 'kmix',
                'ea+eb @ 0': 'ea+eb', '1o3ol+2h @ 0': '1o3ol+2h',
            }, level=level
        )

    def preprocess_natmix_df(df: pd.DataFrame) -> pd.DataFrame:
        """Move 'panel' level from row to column index, but does not change data.
        """
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
        return ret

    # TODO better names for these than mdf/bdf
    mdf = preprocess_natmix_df(mdf)
    bdf = preprocess_natmix_df(bdf)

    # both should be {'control', kiwi'}
    natmix_panels = set(mdf.columns.get_level_values('panel').unique())
    assert natmix_panels == set(bdf.columns.get_level_values('panel').unique())

    analyze_eag = False
    if analyze_eag:
        # TODO move this file to data/ subdir of al_analysis, for consistency?
        eag = pd.read_csv('mean_min_eag.csv')
        eag['name'] = eag.odorname.map(olf.abbrev)

        assert eag.name.notna().all()
        assert not (eag.name == eag.odorname).any()
        all_natmix_abbrevs = set()
        # NOTE: no 'pfo' (in this CSV at least), so on odors should be in >1 of the
        # panels
        eag['panel'] = np.nan
        for panel in natmix_panels:
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
        for panel  in natmix_panels:
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

    analyze_orn = True
    if analyze_orn:
        orn_mdf = load_natmix_dff()
        orn_mdf = orn_mdf.groupby(['panel','odor1'], sort=False).mean()
        orn_mdf = orn_mdf.loc[
            ~orn_mdf.index.get_level_values('odor1').str.startswith('pfo')
        ]
        orn_mdf = drop_mix_dilutions(orn_mdf)
        # NOTE: there is already not air mix in here, and odors already sorted as i want
        # within each panel (w/ binary mix at end, right before full mix)

        print('\nORN data:')
        n_flies, n_rois = count_flies_and_rois(orn_mdf)
        del n_rois

        n_str = f'\nn={n_flies}'

        orn_mdf = rename_natmix_odors(orn_mdf, 'odor1')

        orn_mean = orn_mdf.groupby(level='roi', axis='columns').mean()
        orn_mean = fill_to_hemibrain(orn_mean)
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
            # TODO move cbar closer to main ax?
        )

        # dropping diagnostics
        orn_mdf = orn_mdf[orn_mdf.index.get_level_values('panel').isin(natmix_panels)
            ].copy()

        assert set(orn_mdf.index.get_level_values('panel')) == natmix_panels
        # TODO lines around binary mix?
        for panel in sorted(natmix_panels):
            panel_orn_mdf = orn_mdf.loc[panel]
            orn_corr = mean_of_fly_corrs(panel_orn_mdf)
            plot_corr(orn_corr, model_root, f'orn_corr_{panel}',
                title=f'mean ORN correlation{n_str}',
            )

        # TODO and plot corrs for KC data i care about? (mean of fly)
        # across the diagonal concentrations for the kiwi stuff?
        # (or in model_yang_mixtures?)

        # calling this after plot_responses b/c it reshapes and adds NaN. single rows
        # would not show for both kiwi and control panels
        # TODO move the odor renaming from this earlier tho?
        orn_mdf = preprocess_natmix_df(orn_mdf)

        raw_orn = orn_mdf.groupby(level=['panel'] + fly_cols, sort=False,
            axis='columns').mean().unstack().dropna()
        raw_orn = raw_orn.rename('value').to_frame()

        names_before = list(raw_orn.index.names)
        raw_orn = raw_orn.reset_index()
        raw_orn.odor = strip_concs(raw_orn.odor)
        raw_orn = raw_orn.set_index(names_before, verify_integrity=True)

        orn_stat = 'mean_peak_dff'
        raw_orn['stat'] = orn_stat

        # TODO refactor to share w/ eag processing above
        normed_orn_dfs = []
        for panel  in natmix_panels:
            pdf = raw_orn[raw_orn.index.get_level_values('panel') == panel]
            normed_pdf = normalize_one_panel(pdf)
            normed_orn_dfs.append(normed_pdf)

        normed_orn_df = pd.concat(normed_orn_dfs, verify_integrity=True)

        curr_levels = list(raw_orn.index.names)
        level_order = curr_levels + ['is_normalized']
        normed_orn_df = addlevel(normed_orn_df, 'is_normalized', True).reorder_levels(
            level_order)

        raw_orn = addlevel(raw_orn, 'is_normalized', False).reorder_levels(level_order)
        orn_intensity = pd.concat([raw_orn, normed_orn_df], verify_integrity=True)
        orn_intensity['source'] = 'ORNs'

        # TODO TODO TODO also process orn data below

    # only the highest concs (the concs that are also the components in the 5-component
    # mixtures) of the binary ramp experiment should be in both natmix_df and bdf
    binary_comps = natmix_df.columns[
        natmix_df.columns.get_level_values('odor').isin(bdf.index)
    ]
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
    flyroi_binary_ser = diag_df.T.stack(diag_df.index.names).rename(resp_col)
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
    flyroi_5comp_ser = mdf.T.stack().rename(resp_col)
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
    # ipdb> df[df.panel.isin(natmix_panels) & (df.stat == 'mean_response_rate')
    #   ].value.mean()
    # 0.08937741802973936
    #
    # ipdb> df[df.panel.isin(natmix_panels) & (df.stat == 'mean_response_rate')
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
    # TODO TODO include this in suptitle of plots that use it
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
    # TODO TODO TODO also refactor to share ORN thresh w/ that script
    #
    # TODO TODO TODO make a new plot_root for case where NATMIX_KC_THRESH !=
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
    #    panel_cols + flyroi_cols
    #).any()

    # ipdb> NATMIX_KC_THRESH
    # 1.5
    # ipdb> natmix_responders.sum() / len(natmix_responders)
    # 0.3555670417968658
    # ipdb> (natmix_stat_ser >= NATMIX_KC_THRESH).sum() / len(natmix_stat_ser)
    # 0.0743385882521062
    # ipdb> natmix_stat_ser.groupby(panel_cols + fly_cols).ngroups
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
    #assert set(natmix_responders.index.names) == set(panel_cols + flyroi_cols)
    #assert set(natmix_responders.index.names) & set(withinpanel_odor_cols) == set()
    #natmix_n_responding = natmix_responders.groupby(panel_cols + fly_cols).sum()

    # checking that no fly has either mix (5comp/binary) type for both panels
    # (kiwi/control), otherwise could not index fly2n_total_roi data sources without
    # panel (which they don't currently have)
    # TODO can i just regen them with panel tho?
    assert (
        len(natmix_stat_ser.index.to_frame(index=False)[['mix'] + fly_cols
            ].drop_duplicates()) ==
        len(natmix_stat_ser.index.to_frame(index=False)[panel_cols + fly_cols
            ].drop_duplicates())
    )
    # technically redundant with assertion above
    #assert not natmix_n_responding.index.droplevel('panel').duplicated().any()

    # did i not need to reorder levels before and after this division? matter?
    # (does not matter)
    # TODO assert set of these indices (natmix_n_responding and natmix_nrois) are now
    # the same, at least after reordering levels)
    # TODO TODO rename frac_responding or something more clear?
    #
    # TODO TODO TODO is this calculation actually right? (no. delete!) isn't it just
    # *other* places i want to compute this per ROI across odors, and here i actually do
    # want it just per (cell, odor) pair?
    # TODO TODO TODO isn't it currently *just* for clustering + plotting and diff_col
    # mean/distribution that i currently want to filter out *rois* that are
    # non-responding?
    #natmix_flyavg_bin_ser = natmix_n_responding / natmix_nrois

    # NOTE: kiwi/binary/2022-04-10/1 might be one fly without any filtering applied on
    # ROIs
    # ipdb> natmix_stat_ser.groupby(panel_cols + fly_cols).apply(lambda x:
    #   x.index.get_level_values('roi').nunique()) -
    #   natmix_nrois.reorder_levels(panel_cols + fly_cols).sort_index()
    # panel    mix     date        fly_num
    # control  5comp   2022-03-29  2         -1592
    #                  2022-04-04  1         -2573
    #                  2022-07-20  2         -2198
    #                  2022-07-25  2         -1794
    #          binary  2022-03-29  1          -225
    #                              2         -2346
    #                  2022-04-04  1          -294
    # kiwi     5comp   2022-07-01  1         -1544
    #                  2022-07-02  1         -1379
    #                              2         -1752
    #                  2022-07-11  1         -1259
    #                              2         -2477
    #                  2022-07-12  3         -1207
    #                              5         -1367
    #          binary  2022-04-09  1          -312
    #                              2          -213
    #                  2022-04-10  1             0
    # dtype: int64
    # ipdb> natmix_stat_ser.groupby(panel_cols + fly_cols).apply(lambda x:
    #   x.index.get_level_values('roi').nunique())
    # panel    mix     date        fly_num
    # control  5comp   2022-03-29  2          2623
    #                  2022-04-04  1          2687
    #                  2022-07-20  2          1106
    #                  2022-07-25  2          1814
    #          binary  2022-03-29  1          3634
    #                              2           734
    #                  2022-04-04  1          3652
    # kiwi     5comp   2022-07-01  1          2605
    #                  2022-07-02  1          2584
    #                              2          1764
    #                  2022-07-11  1          2655
    #                              2          3107
    #                  2022-07-12  3          2297
    #                              5          1813
    #          binary  2022-04-09  1          3352
    #                              2          3002
    #                  2022-04-10  1          3134

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

    # TODO TODO TODO redo analysis comparing everything to a threshold that gives us
    # more like 10% response rate?
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

    # TODO also add fly_id to this at some point?

    natmix_stat_df['odor'] = strip_concs(natmix_stat_df.odor)
    natmix_stat_df = natmix_stat_df.sort_values(by='odor', kind='stable',
        key=odor_sort_fn
    )
    natmix_stat_df = natmix_stat_df.reset_index(drop=True)

    # TODO refactor to share w/ above?
    if analyze_orn:
        orn_flyroi_5comp_ser = orn_mdf.T.stack().rename(resp_col)
        assert isinstance(orn_flyroi_5comp_ser, pd.Series)
        assert not orn_flyroi_5comp_ser.isna().any()
        assert len(orn_flyroi_5comp_ser) == orn_mdf.notna().sum().sum()
        orn_flyroi_5comp_ser = addlevel(orn_flyroi_5comp_ser, 'mix', '5comp')

        assert (
            orn_flyroi_5comp_ser.index.names ==
            flyroi_binary_ser.index.droplevel('pair_dilution_factor').names
        )
    #

    # TODO describe purpose of this loop
    for panel in sorted(natmix_panels):
        panel2kc_flyodor_stats[panel] = natmix_stat_df[natmix_stat_df.panel == panel]

        panel_mdf = mdf.loc[:, mdf.columns.get_level_values('panel') == panel
            ].dropna(how='all')
        assert not panel_mdf.isna().any().any()
        panel_mdf = addlevel(panel_mdf, 'mix', '5comp', axis='columns')

        if analyze_orn:
            orn_panel_df = orn_flyroi_5comp_ser.loc[
                orn_flyroi_5comp_ser.index.get_level_values('panel') == panel
            ].unstack('odor').T

            # TODO TODO refactor to share w/ elsewhere (and also pull in logic to get
            # components and form a mask of components + mix from there to here)
            binary_mix_mask = orn_panel_df.index.str.contains('+', regex=False)
            assert binary_mix_mask.sum() == 1
            #

            orn_panel_mdf = orn_panel_df.loc[~binary_mix_mask]
            # TODO TODO construct orn_panel_bdf by subsetting to just top 2
            # components and binary mix? (just wouldn't have pair_dilution_factor > 0)

        panel_bdf = diag_df.loc[:, bdf.columns.get_level_values('panel') == panel
            ].dropna(how='all')
        assert not panel_bdf.isna().any().any()
        panel_bdf = addlevel(panel_bdf, 'mix', 'binary', axis='columns')

        if MIX_SUPP_IN_RESPONDERS_ONLY:
            panel_mdf = panel_mdf.loc[:, (panel_mdf >= NATMIX_KC_THRESH).any()]
            if analyze_orn:
                orn_panel_mdf = orn_panel_mdf.loc[:,
                    (orn_panel_mdf >= NATMIX_ORN_RESPONSE_THRESH).any()
                ]

            # TODO TODO delete? this does not seem used to calculate mix suppression in
            # binary case? is appropriate subsetting to panel-any-responders still
            # happening tho, for natmix binary cases?
            panel_bdf = panel_bdf.loc[:, (panel_bdf >= NATMIX_KC_THRESH).any()]

        # TODO factor out one KC row_coarsen_factor (have that now), or special
        # case for each dataset, or based on target number of ROIs to display (and
        # current size of input data?)? (either way, have this fn include the chosen
        # value in title prob)
        plot_hierarch_clustered_rois(model_root, panel_mdf, f'{panel}_5comp',
            title=panel, ignore_existing=ignore_existing,
            row_coarsen_factor=KC_ROW_COARSEN, row_coarsen_by=fly_cols
        )
        plot_hierarch_clustered_rois(model_root, panel_bdf, f'{panel}_binary',
            title=panel, ignore_existing=ignore_existing,
            row_coarsen_factor=KC_ROW_COARSEN, row_coarsen_by=fly_cols
        )

        # TODO group ROIs into response classes (both here and for model cells),
        # before computing mix suppression, so i can also plot average mix suppression
        # for each [alongside fraction of overall population]?  (and also do for
        # diag-binaries above)?

        mix_supp_5comp = calc_mix_suppression(panel_mdf)
        if analyze_orn:
            orn_mix_supp_5comp = calc_mix_suppression(orn_panel_mdf)

        # TODO delete
        #mix_supp_binary = calc_mix_suppression(panel_bdf)
        # calc_mix_suppression only works when input has a single mix (and its
        # components, with mix at end)
        mix_supp_binary = panel_bdf.groupby('pair_dilution_factor', sort=False).apply(
            calc_mix_suppression
        )

        mix_supp_5comp = addlevel(mix_supp_5comp, 'pair_dilution_factor', 0)
        assert mix_supp_5comp.index.names == mix_supp_binary.index.names

        # at least for orn_panel_mdf, the following two were equivalent:
        # ipdb> s1 = orn_panel_mdf.T.stack(orn_panel_df.index.names).rename(orn_resp_col)
        # ipdb> s2 = orn_panel_mdf.unstack().rename(orn_resp_col)
        # ipdb> s1.equals(s2)
        # True
        # TODO TODO replace body of list comp w/ above simpler expr?
        #
        # TODO what is this doing? doc
        flyroi_odor_sers = [
            x.T.stack(x.index.names).rename(resp_col) for x in [panel_mdf, panel_bdf]
        ]
        panel_mser, panel_bser = flyroi_odor_sers
        panel_mser = addlevel(panel_mser, 'pair_dilution_factor', 0
            ).reorder_levels(panel_bser.index.names)

        flyroi_odor_stats = pd.concat([panel_mser, panel_bser], verify_integrity=True
            ).reset_index()

        flyroi_odor_stats.odor = strip_concs(flyroi_odor_stats.odor)
        flyroi_odor_stats = flyroi_odor_stats.sort_values(by='odor', kind='stable',
            key=odor_sort_fn
        )

        assert resp_col in flyroi_odor_stats.columns
        assert 'value' not in flyroi_odor_stats.columns
        flyroi_odor_stats = flyroi_odor_stats.rename(columns={resp_col: 'value'})
        assert 'stat' not in flyroi_odor_stats.columns
        flyroi_odor_stats['stat'] = resp_col

        # TODO what all are these being used for? doc here. cause it's already been
        # subset to responders (to any odor in panel, either across 5comp or binary)
        # above, so shouldn't be used for response rates. i think it might just be used
        # for selecting cells for comp-vs-mix response strength dists tho, so should be
        # fine
        # TODO TODO TODO what are class_fracs computed from tho? that calculation
        # correct?
        flyroi_odor_stats['responded'] = flyroi_odor_stats.value >= NATMIX_KC_THRESH

        if analyze_orn:
            # TODO refactor to share w/ above
            orn_flyroi_odor_stats = orn_panel_mdf.unstack().rename(orn_resp_col
                ).reset_index()
            orn_flyroi_odor_stats.odor = strip_concs(orn_flyroi_odor_stats.odor)
            orn_flyroi_odor_stats = orn_flyroi_odor_stats.sort_values(by='odor',
                kind='stable', key=odor_sort_fn
            )
            assert orn_resp_col in orn_flyroi_odor_stats.columns
            assert 'value' not in orn_flyroi_odor_stats.columns
            orn_flyroi_odor_stats = orn_flyroi_odor_stats.rename(
                columns={orn_resp_col: 'value'}
            )
            assert 'stat' not in orn_flyroi_odor_stats.columns
            orn_flyroi_odor_stats['stat'] = orn_resp_col
            orn_flyroi_odor_stats['responded'] = (
                orn_flyroi_odor_stats.value >= NATMIX_KC_THRESH
            )
            #
            panel2orn_flyroi_odor_stats[panel] = orn_flyroi_odor_stats

            # TODO TODO worth splitting out "binary" version of ORN stuff?
            # (that's something to do above anyway)
            # i'm not separately analyzing that w/ response class analysis, so lower
            # utility rn
            orn_panel_mix_supp = mix_supp_list2flystat_df([orn_mix_supp_5comp],
                stat=orn_resp_col
            )
            panel2orn_fly_stats[panel] = orn_panel_mix_supp

        # TODO TODO TODO just make parallel versions of each of these for ORN data
        # TODO TODO TODO TODO this is used as input to summarize_response_classes! is it
        # an issue that we've dropped panel non-responders above?
        # TODO TODO TODO TODO check by recomputer # / frac of non-responders above, and
        # comparing to that computed in summarize_response_classes (or after any
        # adjustment to add back non-responders, if there is any)
        print('CHECK RESPONDER DROPPING NOT AFFECTING RESPONSE CLASS CALCS')
        panel2kc_flyroi_odor_stats[panel] = flyroi_odor_stats
        # TODO what does this function do exactly? doc?
        # TODO need to update to work w/ pair_dilution_factor level? (prob not)
        panel_mix_supp = mix_supp_list2flystat_df([mix_supp_5comp, mix_supp_binary])
        # TODO rename panel2kc_mix_supp? (no other stats in here)
        panel2kc_fly_stats[panel] = panel_mix_supp

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

        kc_df['source'] = 'KCs'

        normed_dfs = []
        # TODO add flag to also add normalized versions for any w/ name matching
        # exactly? (i.e. mean_response_rate)?
        # TODO TODO also try retuning on her odors, with her mean response rate
        # (and how much do fixed_thr and wAPLKC_scale differ between that and tuning on
        # megamat [and for each model, including bouton versions]?)
        for x in set(compare_normalized.values()):
            raw_df = kc_df[kc_df.stat == x]
            if panel in natmix_panels:
                mix_vals = ['binary', '5comp']
                assert set(raw_df.mix.unique()) == set(mix_vals)

                for mix_type in mix_vals:
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
    # TODO any reason to calculate these *model_response_strengths, instead of
    # calculating directly from model_roi_df (/df, which just has them averaged over
    # ROIs)

    natmix_panel_class_frac_list = []

    comps_to_drop = [
        'fur', 'ms', 'va', 'EtOH', 'IAol', 'IaA'
    ]
    model_mean_mix_supp_sers = []
    # TODO TODO is the change from 2026-05-10 outputs to those on 2026-05-20 just
    # the change in tuning convergence? or what else? is it only the wd20 and prat-claws
    # stuff moving? ignore LR cache and regen?
    # NOTE: seems like it might *just* be the wd20 case moving around (neither uniform
    # nor prat-claws [nor either APL case for latter] seem to have changed)
    # TODO TODO if so, use smaller sp_acc for everything, to minimize tuning
    # related noise? otherwise, what is the difference?
    for panel in panels:
        # TODO remove this copy? could help w/ memory issues...
        pdf = df[df.panel == panel].copy()

        kc_panel = None
        if panel.startswith('diag-binaries_'):
            kc_panel = 'diag-binaries'

        elif panel in panel2kc_flyodor_stats:
            kc_panel = panel

        if kc_panel is not None:
            assert kc_panel in panel2kc_flyodor_stats
            assert kc_panel in panel2kc_fly_stats
            assert kc_panel in panel2kc_flyroi_odor_stats
            # TODO and assert they are all not-None?
            for i, x in enumerate([
                    panel2kc_fly_stats, panel2kc_fly_stats, panel2kc_flyroi_odor_stats
                ]):
                assert panel2kc_fly_stats[kc_panel] is not None, \
                    f'{panel=} {kc_panel=} {i=}'
        else:
            assert panel not in panel2kc_flyodor_stats
            assert panel not in panel2kc_fly_stats
            assert panel not in panel2kc_flyroi_odor_stats

        if not simplify_models:
            pivot_cols = ['source', 'connectome_apl', 'model_pnkc_class', 'roi']
        else:
            pivot_cols =  ['source', 'model_pnkc_class', 'roi']

        for_mix_supp = model_roi_odor_df[
            (model_roi_odor_df.panel == panel) & (model_roi_odor_df.stat != 'responded')
        ].pivot(index='odor', values='value', columns=pivot_cols + ['stat'])
        # TODO delete? refactor (do before pivot?)?
        responded = model_roi_odor_df[
            # TODO TODO why am i precomputed 'responded' again? recompute here and make
            # sure it makes sense?
            # TODO is n_spikes_for_response=2 handled correctly? (it should be, since
            # it's coming from the masks returned by modeling code. could assert w/in
            # tolerance in each model here? or on megamat at least, since not tuned
            # here...)
            (model_roi_odor_df.panel == panel) & (model_roi_odor_df.stat == 'responded')
        ].pivot(index='odor', values='value', columns=pivot_cols)

        assert pivot_cols == responded.columns.names
        nonroi_pivot_cols = [x for x in pivot_cols if x != 'roi']
        responded_cols = responded.columns.to_frame(index=False)
        assert PNKC_CLASS_COL in nonroi_pivot_cols
        other_pivot_levels = [
            x for x in nonroi_pivot_cols if x != PNKC_CLASS_COL
        ]
        # want this one first for easier .loc to get response rate of interest
        nonroi_pivot_cols = [PNKC_CLASS_COL] + other_pivot_levels

        # otherwise, could not just groupby source to summarize response rate of each
        # model (probably won't anyway tho...)
        # NOTE: this assertion lets us droplevel other_pivot_levels below
        # TODO delete. this is no longer true if 'connectome_apl' is in levels, b/c
        # that info is stripped from 'source' level. only care to print (or include mean
        # model response rates in titles) if -M, which ignores connectome_apl, and also
        # if not -f
        #n_unique_models = len(responded_cols[nonroi_pivot_cols].drop_duplicates())
        #assert n_unique_models == responded_cols.source.nunique(), (
        #    f'{nonroi_pivot_cols=}\n'
        #    f'{n_unique_models=} != {responded_cols.source.nunique()=}'
        #)
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
        perodor_model_response_rate = responded.groupby(level=nonroi_pivot_cols,
            axis='columns').mean().droplevel(other_pivot_levels, axis='columns')
        # TODO print this if we PNKC_CLASS_COL values don't have n-variant suffices
        # (mainly if -M and either -m or no -f)
        mean_model_response_rate = perodor_model_response_rate.mean()
        mean_model_response_rate = addlevel(mean_model_response_rate, 'panel', panel)
        mean_model_response_rate_list.append(mean_model_response_rate)

        if only_analyzing_few_models:
            print(f'mean response rate per model (across all {panel=} odors analyzed):')
            print(mean_model_response_rate.to_string())

        # TODO put in title/fname too
        # TODO remove this step? we are already subsetting in both binary and 5comp
        # cases below, so should be redundant... this is vestigal code i think
        if MIX_SUPP_IN_RESPONDERS_ONLY:
            for_mix_supp = for_mix_supp.loc[:, responded.any()]
        #

        # TODO sort ROIs into response classes first? (maybe after grouping by mix, esp
        # in diag-binaries case?)
        model_mix_supp = None
        model_mix_resps = None
        if panel in natmix_panels or kc_panel == 'diag-binaries':
            binary_mix_mask = for_mix_supp.index.str.contains('\+')

            for_binary_list = []
            for binary_mix in for_mix_supp.index[binary_mix_mask]:
                ca, cb = binary_mix.split('+')
                assert all((for_mix_supp.index == x).sum() == 1 for x in (ca, cb))

                binary_and_comp_mask = for_mix_supp.index.isin((ca, cb, binary_mix))
                # TODO work?
                assert binary_and_comp_mask.sum() == 3
                for_binary = for_mix_supp[binary_and_comp_mask]

                if MIX_SUPP_IN_RESPONDERS_ONLY:
                    for_binary = for_binary.loc[:,
                        responded[binary_and_comp_mask].any()
                    ]

                if kc_panel == 'diag-binaries':
                    for_binary = addlevel(for_binary, 'mix', binary_mix, axis='columns')

                for_binary_list.append(for_binary)

            if panel in natmix_panels:
                assert len(for_binary_list) == 1
                for_mix_supp_binary = for_binary_list[0]
                assert 'mix' not in for_mix_supp_binary.columns.names
                # TODO use name of mix instead? (would also have to change KC handling)
                for_mix_supp_binary = addlevel(for_mix_supp_binary, 'mix', 'binary',
                    axis='columns'
                )
                # TODO TODO rename these variables to include model in them, to be clear
                # that the KC data is coming from elsewhere
                mix_supp_binary = calc_mix_suppression(for_mix_supp_binary)
            else:
                assert len(for_binary_list) > 1
                mix_supp_binary = pd.concat(
                    [calc_mix_suppression(x) for x in for_binary_list],
                    verify_integrity=True
                )

            full_mix_mask = for_mix_supp.index.str.contains('mix')
            if kc_panel == 'diag-binaries':
                assert full_mix_mask.sum() == 0
                model_mix_supp = mix_supp_binary
            else:
                # TODO define below, closer to where used?
                assert full_mix_mask.sum() == 1
                full_mix = for_mix_supp.index[full_mix_mask][0]
                #

                full_mix_and_comp_mask = ~binary_mix_mask
                assert full_mix_and_comp_mask.sum() == 6
                for_mix_supp_5comp = for_mix_supp[full_mix_and_comp_mask]

                if MIX_SUPP_IN_RESPONDERS_ONLY:
                    for_mix_supp_5comp = for_mix_supp_5comp.loc[:,
                        responded[full_mix_and_comp_mask].any()
                    ]

                # TODO use name of mix instead?
                for_mix_supp_5comp = addlevel(for_mix_supp_5comp, 'mix', '5comp',
                    axis='columns'
                )
                mix_supp_5comp = calc_mix_suppression(for_mix_supp_5comp)

                model_mix_supp = pd.concat([mix_supp_binary, mix_supp_5comp],
                    verify_integrity=True
                )

                # TODO delete? or keep calc up here? check against below?
                # can i replace w/ part of for_mix_supp_5comp? (would have to change
                # indexing so that i drop all nonresponding (KC, odor) pairs, which
                # isn't that easy in current format)
                panel_response_strengths = all_model_response_strengths[
                    (all_model_response_strengths.panel == panel) &
                    ~all_model_response_strengths.odor.str.contains('\+')
                ]
                # TODO put behind flag?
                # TODO leave the responded column to be able to make this decision
                # later?
                panel_response_strengths = panel_response_strengths[
                    panel_response_strengths.responded
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
                # TODO what is this line doing?
                # (oh, this was for a plot that was just the response strength within
                # responders, of top component and mix ig... not used for general
                # per-odor response strength / sparsity plot, which is based on pdf)
                panel_response_strengths = panel_response_strengths[
                    # TODO assert both present alone?
                    panel_response_strengths.odor.isin((top_component, full_mix))
                ]

                # TODO TODO TODO define for binary too
                panel_response_strengths['mix'] = '5comp'
                # TODO delete?
                model_mix_resps = panel_response_strengths.copy()

            model_mix_supp = model_mix_supp.reset_index()

        # TODO TODO fit gaussian to non-responders in yang's labelled non-responders, as
        # well as in my thresholded Remy data (for a few thresholds?)
        # TODO eventually incorporate as preprocessing step in fn that fits scaling fn
        # to match spike count distribution to real data?

        if kc_panel is not None:
            # NOTE: don't need to rename panel in this dataframe (no matter if it's diff
            # from current `panel` in loop, because plotting fn does not actually use
            # panel in input data, just the input panel str for naming stuff)
            # TODO rename flyodor_stats?
            kc_df = panel2kc_flyodor_stats[kc_panel]

            s1 = set(pdf.odor.unique())
            s2 = set(kc_df.odor.unique())
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
            assert s1 == s2, f'{panel=} {s1 - s2=}'

            # TODO make sure odors are sorted in same order, so that we don't
            # screw up xticklabel order for plots that include KC data (and will that
            # fix it? how did i sort odors for model stuff again? do here or when
            # loading yang data?)
            # (they should be now, but assert that here?)

            fly_stats = panel2kc_fly_stats[kc_panel]
            assert model_mix_supp is not None, ('model mix suppression not calculated '
                f'for {panel=} {kc_panel=}'
            )
            mix_supp = pd.concat([model_mix_supp, fly_stats], ignore_index=True)

            # TODO set these in code populating panel2kc_fly_stats above? any other
            # panel2* data currently missing this for KCs? add both even earlier in KC
            # data?
            mix_supp[source_col] = mix_supp[source_col].fillna('KCs')
            mix_supp['model_pnkc_class'] = mix_supp['model_pnkc_class'].fillna('KCs')

            if not simplify_models:
                mix_supp.connectome_apl = mix_supp.connectome_apl.fillna(False)
            #

            multiple_mixes = 'mix' in mix_supp.columns and mix_supp.mix.nunique() > 1

            # TODO delete. should be filled earlier now
            # TODO (still? delete?) well it's not, and neither is 'source' (why???)
            #mix_supp.model_pnkc_class = mix_supp.model_pnkc_class.fillna('KCs')

            diff_col = get_diff_col(mix_supp)

            from_kcs = mix_supp.source == 'KCs'
            kc_mix_supp = mix_supp[from_kcs]

            def plot_one_dist_per_fly(data: pd.DataFrame, flies_share_bins: bool = True,
                **kwargs) -> None:

                if flies_share_bins and _USE_KDEPLOT:
                    warn('plot_one_dist_per_fly: can not use flies_share_bins=True'
                        f' when {_USE_KDEPLOT=}. ignoring!'
                    )

                if flies_share_bins and not _USE_KDEPLOT:
                    assert 'x' in kwargs
                    x = kwargs['x']
                    values = data[x]
                    binrange = (values.min(), values.max())
                    # TODO delete
                    #print(f'plot_one_dist_per_fly: {binrange=}')
                    #

                # TODO or CI across flies? possible?
                for gn, gdf in data.groupby(fly_cols, sort=False):
                    # TODO delete
                    #sns.kdeplot(data=gdf, **kwargs)
                    # TODO or try low alpha instead of fill=False?
                    distplot(data=gdf, fill=False, **kwargs)

            if 'pair_dilution_factor' in kc_mix_supp.columns:
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
                    f'{diff_col}_dists_kc-only_pair-dilutions_{panel}'
                )
                #

                # so the binary mix plot below doesn't average across multiple dilution
                # factors (as concentrations are stripped at this point)
                kc_mix_supp = kc_mix_supp[kc_mix_supp.pair_dilution_factor == 0].copy()

            facet_kws = dict()
            # TODO are there any panels where this isn't true? which?
            if multiple_mixes:
                # TODO use row like above, or hue here?
                #
                # separates the binary and 5comp versions of the mix
                # TODO was thinking of just gray=binary, black=5comp, but ideally need
                # something that also works for all 3 yang's binary mixes
                facet_kws = dict(
                    hue='mix',
                    # TODO don't re-use husl(3)?
                    palette='cubehelix' if panel in natmix_panels else 'husl'
                )

            # TODO TODO refactor to share below w/ *_kc-only_pair-dilutions.pdf stuff
            # above
            # TODO TODO TODO also plot one line per all KCs, similar to model_only path
            # plotting model dists (or finish implementing kc_only=True there)
            # TODO say i'm dropping non-responders in all mixsupp plots somewhere
            g = sns.FacetGrid(data=kc_mix_supp, col='panel', **facet_kws)
            # TODO just put each mix on a different row again, if hue is going to be too
            # messy? (for yang's data, at least?)
            g.map_dataframe(plot_one_dist_per_fly, x=diff_col,
                # was using 0.6 for natmix w/ KDE, but want lower w/ stepwise hist
                alpha=0.4 if panel in natmix_panels else 0.3,
                linewidth=1.0 if panel in natmix_panels else 0.75
            )
            g.set_titles('{col_name}')
            g.set_xlabels(diff_col2desc(diff_col))
            g.set_ylabels('density across KCs')
            # TODO TODO keep all lines kc_color, and use linestyle for binary/5comp
            # (dashed for binary)?
            g.add_legend(title='mix')
            # TODO change sparsity ylim on this one, so we can we more easily (more
            # easily what? delete?)? [or maybe need to change threshold anyway?
            # currently between about 0.02 and 0.1 response rate, for kiwi])
            savefig(g, model_root, f'{diff_col}_dists_kc-only_{panel}_per-fly')

            model_nonroi_levels = [
                x for x in model_mix_supp.columns if x not in ('roi', diff_col)
            ]
            # TODO (still an issue?) preserve model_dirname (prob not lost here. where
            # above is it?) (or just add back below?)
            model_mean_mix_supp = model_mix_supp.groupby(model_nonroi_levels, sort=False
                )[diff_col].mean()

            model_mean_mix_supp = addlevel(model_mean_mix_supp, 'panel', panel)
            model_mean_mix_supp_sers.append(model_mean_mix_supp)

            # TODO use use prior def of model_mix_supp? does code below actually need
            # anything added to mix_supp?
            model_mix_supp = mix_supp[~from_kcs]

            # TODO in each facet title, say how many nonresponders were dropped
            # (or in suptitle/legend for KCs?) would probably need a CSV for models...

            model_stat_order = ['logistic_scaled_num_spikes', 'num_spikes']

            # TODO TODO should everything be z-scored before computing mix - max?
            # or how to make more comparable (at least, in terms of the expected offset
            # from 0)? i suppose even just making logistic ceiling higher would do that?
            # TODO TODO at least plot input distributions for each of this (hist
            # for spike counts) and maybe kde/hist for logistic scaled spike counts
            def plot_one_dist_per_model(data, model_only: bool = False, **kwargs
                ) -> None:
                pnkc_classes = data[PNKC_CLASS_COL].unique()
                assert len(pnkc_classes) == 1, ('expected data from only one hue='
                    f'{PNKC_CLASS_COL} level to be passed in at a time'
                )
                pnkc_class = pnkc_classes[0]

                assert 'alpha' not in kwargs, 'we set this manually below'

                # it can be NaN some places still, right? hopefully not here tho
                assert pd.notna(pnkc_class)

                model_input = pnkc_class_is_model(pnkc_class)
                if model_only:
                    assert model_input

                if not simplify_models:
                    # make below conditional if this fails
                    assert 'connectome_apl' in data.columns

                assert not data.mix.isna().any()
                mix = data.mix.unique()
                assert len(mix) == 1
                # either 'binary'/'5comp' (for control/kiwi KC data), or
                # '2h+farn'/'farn+ma'/'2h+ma' (for Yang's diag-binaries data)
                # TODO still or NaN for model? must not be now?
                mix = mix[0]

                if not model_input:
                    # TODO refactor to a kc_alpha const?
                    alpha = 1.0
                    distplot(data=data, fill=False, alpha=alpha, **kwargs)
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
                if not simplify_models:
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
                    if not simplify_models:
                        source, connectome_apl = gn
                    else:
                        source = gn
                        connectome_apl = False

                    # TODO (delete? presumably this was fixed?) fix legend so it shows
                    # both linestyles (and uses that to show connectome vs uniform APL,
                    # w/o color, like i do w/ markers for other legend. refactor?)
                    linestyle = '--' if connectome_apl else '-'
                    # TODO only label on first one or something legend screwed up
                    # otherwise?
                    #sns.kdeplot(data=gdf, linewidth=model_linewidth,
                    # I think I do like (default) fill=True here, right?
                    distplot(data=gdf, linewidth=model_linewidth, linestyle=linestyle,
                        alpha=model_alpha, **kwargs
                    )

            # TODO TODO figure out how to show num_spikes and KC computed data on same
            # scale (percentile? zscore?) (currently just not plotting those two against
            # each other, but removing col='stat' option from call plotting both model
            # and KC data)
            def plot_mixsupp_dists(model_only: bool = False, kc_only: bool = False,
                fname_suffix: str = '', **kwargs) -> None:
                if model_only:
                    assert not kc_only

                facet_kws = dict(sharex=False, sharey=False, hue=PNKC_CLASS_COL,
                    palette=source_palette
                )

                model_stat = None
                kc_stat = None
                plot_kws = dict(x=diff_col)
                if not kc_only:
                    plot_kws.update(dict(model_only=model_only))

                    if model_only:
                        facet_kws.update(dict(
                            col='stat', col_order=model_stat_order
                        ))
                        data = model_mix_supp
                    else:
                        assert facet_kws.get('col') != 'stat'
                        # this wouldn't work if we didn't also want to subset to just
                        # one KC stat here, because we have col='stat', and since KCs
                        # stat is different from the two model ones, would never be
                        # plotted on same Axes (and would not be plotted at all if not
                        # in col_order)
                        data = mix_supp

                        warn('plot_mixsupp_dists: dropping raw num_spikes for '
                            'comparison between model and KC data! see model_only=True'
                            ' version of plot for the distribution of those values'
                        )
                        data = data[data.stat != 'num_spikes'].copy()

                        model_stat = 'logistic_scaled_num_spikes'
                        kc_stat = 'Fc_zscore'
                        unique_stats = set(data.stat.unique())
                        # TODO TODO TODO what did i change (wasn't i just changing ORN
                        # handling?) that led to this now failing:
                        # AssertionError:
                        # unique_stats={'mean_Fc_zscore', 'logistic_scaled_num_spikes'}
                        assert unique_stats == {kc_stat, model_stat}, f'{unique_stats=}'
                else:
                    data = kc_mix_supp
                    facet_kws.update(dict(col='stat'))

                if multiple_mixes:
                    facet_kws['row'] = 'mix'
                    # will put 'binary' on top, and '5comp' on bottom. not sure i care
                    # about order for other mixtures (e.g. yang's diag binaries), but
                    # would have to do something else if i did
                    facet_kws['row_order'] = sorted(data.mix.unique())[::-1]

                # TODO TODO TODO remove right column model+KC plot (num_spikes), unless
                # i can find a way to get scales in line (prob just try that on a
                # separate plot anyway, if at all)
                #g = sns.FacetGrid(data=data, palette=palette, **facet_kws)
                g = sns.FacetGrid(data=data, **facet_kws)
                g.map_dataframe(plot_one_dist_per_model, **plot_kws, **kwargs)

                if not (model_only or kc_only):
                    # TODO update ranges to be wider for even the one plot i still want
                    # to keep comparing logistic scaled model and KC data?
                    if kc_panel != 'diag-binaries':
                        xlim = (-6, 4)
                    else:
                        xlim = (-6, 6)
                    warn(f'plot_mixsupp_dists: hardcoding {xlim=} for model vs KC plot')
                    g.set(xlim=xlim)

                suptitle = f'{panel}\ndistribution of "mixture suppression" across KCs'
                suptitle_y = 1.10
                if MIX_SUPP_IN_RESPONDERS_ONLY:
                    suptitle += '\nsilent cells dropped'
                    if not (model_only or kc_only):
                        suptitle += ' (both model and real KCs)'
                else:
                    suptitle += '\nall cells included'

                if g.col_names != []:
                    # this case is just when col= is not in facet_kws
                    assert model_stat is None
                    assert model_only or kc_only
                    g.set_titles('{col_name}')
                else:
                    assert not (model_only or kc_only)
                    assert model_stat is not None
                    g.set_titles('')
                    stat_part = ('\n\nmix suppression computed on:\n'
                        f'{model_stat} for model\n{kc_stat} for observed'
                    )
                    suptitle += stat_part
                    suptitle_y += 0.1

                g.fig.suptitle(suptitle, y=suptitle_y)

                # TODO what's lines=True doing for us?
                add_fixed_legend(g, data, lines=True)

                # TODO put odor (components + mix? just mix name?) in this actually
                # (instead of ylabel), and show for both rows?
                g.set_xlabels(diff_col2desc(diff_col))

                assert len(g.row_names) > 1
                assert g.axes.shape[0] == len(g.row_names)
                for (i, j, hue), gdf in g.facet_data():
                    if hue != 0:
                        continue

                    ax = g.axes[i, j]
                    ylabel = ''
                    if j == 0:
                        ylabel = f'{g.row_names[i]} mix\ndensity across KCs'

                    if i != 0:
                        ax.set_title('')

                    ax.set_ylabel(ylabel)

                fname = f'{diff_col}_dists_{panel}'
                plot_dir = plot_root
                if model_only:
                    fname += '_model-only'
                if kc_only:
                    fname += '_kc-only'
                    plot_dir = model_root

                savefig(g, plot_dir, f'{fname}{fname_suffix}')

            # TODO delete. like cut-3 (sns default cut=) better
            #plot_mixsupp_dists(kde=True, cut=0, fname_suffix='_kde_cut-0')
            #
            plot_mixsupp_dists(kde=True, fname_suffix='_kde')
            plot_mixsupp_dists()
            plot_mixsupp_dists(model_only=True)
            plot_mixsupp_dists(kc_only=True)

            # TODO TODO TODO also one [pointplot?] plot of means (+CI) diff_col values?
            # (comparing across KCs/ORNs and KCs/models?)
            # (w/ panel/mix combos all separate, but otherwise combining data)

            # TODO TODO compare mix response amplitude across KCs and models? or
            # how do i want to handle that? (already sufficiently captured between the
            # mean response amplitude + mean sparsity plot?)
            # TODO TODO maybe just add distributions of response amplitude for each
            # odor, across all KCs? could have one hue (line) per odor, and one facet
            # per KCs and for each model variant? or just have two linestyles (one for
            # all components, one for 5component mix, and then use hue for KCs vs model
            # variants? and normalize each KDE so component one doesn't swamp mix
            # responders)
            if model_mix_resps is None:
                # TODO TODO delete? still true? and was this for yang mixtures only?
                # TODO TODO TODO fix if still true
                warn('model_mix_resps not currently defined for binary mixture case!')
            else:
                flyroi_odor_stats = panel2kc_flyroi_odor_stats[kc_panel]

                kc_responded = flyroi_odor_stats[flyroi_odor_stats.mix == '5comp'
                    ].pivot(
                    columns=fly_cols + ['roi'], index='odor', values='responded'
                )
                kc_responded = kc_responded.sort_index(kind='stable', key=odor_sort_fn)

                n_filtered_kcs = kc_responded.columns.to_frame().groupby(level=fly_cols
                    ).size()
                assert n_filtered_kcs.sum() == len(kc_responded.columns)
                # we can drop panel level here b/c no flies in that KC dataset have both
                # panels (as established earlier)
                n_total_kcs = fly2n_total_rois_5comp.droplevel('panel').loc[
                    n_filtered_kcs.index
                ]
                assert (n_total_kcs > n_filtered_kcs).all()

                # TODO TODO TODO duplicate for ORN data
                # TODO need to groupby fly? ifl prob not. compare w/ model below
                kc_class_sizes = summarize_response_classes(kc_responded)
                assert kc_class_sizes.sum() == len(kc_responded.columns)

                # no need to do this for model stuff, since that data has not had the
                # silent cells dropped here
                kc_class_sizes = add_missing_cells_to_nonresponders(kc_class_sizes,
                    n_total_kcs
                )
                assert kc_class_sizes.sum().equals(n_total_kcs)

                # just for current response class defining code to work (below)
                model_responded = responded.sort_index(kind='stable', key=odor_sort_fn)

                # TODO also calculate something like these w/ just binary mix and
                # components?
                # TODO can i use some other var defined with current binary mix already?
                bmix = model_responded.index[model_responded.index.str.contains('\+')]
                model_responded = model_responded.drop(index=bmix)
                # TODO TODO why does model still have cmix0 here? cache? defined from
                # something other than df?
                assert model_responded.index.equals(kc_responded.index)
                model_responded = model_responded.astype(bool)

                # any point grouping by source first? yes, currently source column will
                # not be preserved otherwise. fly_cols seemed to be handled
                # automatically however.
                # TODO modify summarize_resposne_classes to automatically group on other
                # column index levels (and summarize for each), or take a kwarg to
                # specify which levels to treat like fly_cols?)
                # TODO TODO (delete? still an issue?) are there any other places i'm
                # mistakenly assuming source still encodes APL connectome/uniform? was
                # initially thinking that here, and not including connectome_apl in
                # group levels
                if not simplify_models:
                    model_id_cols = ['source', 'connectome_apl', 'model_pnkc_class']
                else:
                    model_id_cols = ['source', 'model_pnkc_class']

                model_class_sizes = model_responded.groupby(level=model_id_cols,
                    axis='columns', sort=False).apply(lambda x:
                    summarize_response_classes(x, verbose=False).to_frame()
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
                    model_responded.columns.to_frame().groupby(level=model_id_cols,
                        sort=False).size()
                )

                assert model_class_sizes.index.names == kc_class_sizes.index.names
                model_class_sizes = model_class_sizes.sort_index()
                assert kc_class_sizes.equals(kc_class_sizes.sort_index())
                if not model_class_sizes.index.equals(kc_class_sizes.index):
                    warn(f'{panel=} response classes only in either model or KC:\n'
                        f'{model_class_sizes.index.difference(kc_class_sizes.index)=}\n'
                        f'{kc_class_sizes.index.difference(model_class_sizes.index)=}'
                    )

                shared_index = kc_class_sizes.index.union(model_class_sizes.index)
                # TODO also check it has all consecutive categories? prob don't care
                # that much...
                assert shared_index.equals(shared_index.sort_values())

                kc_class_sizes = reindex(kc_class_sizes, shared_index, fill_value=0)
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

                kc_class_fracs = kc_class_sizes / kc_class_sizes.sum()
                assert np.isclose(kc_class_fracs.sum(), 1).all()
                assert kc_class_fracs.sum().index.names == fly_cols

                model_class_fracs = model_class_sizes / model_class_sizes.sum()

                kc_class_fracs = agg_within_mixresp_and_ncomps(kc_class_fracs)
                model_class_fracs = agg_within_mixresp_and_ncomps(model_class_fracs)
                # TODO expand both indices to full posibility of response classes (up to
                # max observed), and fill any missing values w/ 0. or do at end?
                panel_class_fracs = pd.concat(
                    [x.reset_index() for x in [kc_class_fracs, model_class_fracs]],
                    ignore_index=True
                )
                if not simplify_models:
                    panel_class_fracs.connectome_apl = \
                        panel_class_fracs.connectome_apl.fillna(False)

                panel_class_fracs.source = panel_class_fracs.source.fillna('KCs')
                panel_class_fracs.model_pnkc_class = \
                    panel_class_fracs.model_pnkc_class.fillna('KCs')

                panel_class_fracs = panel_class_fracs.set_index(
                    model_id_cols + fly_cols + group_cols,
                    verify_integrity=True
                ).squeeze()
                panel_class_fracs = addlevel(panel_class_fracs, 'panel', panel)
                natmix_panel_class_frac_list.append(panel_class_fracs)

                flyroi_odor_stats = flyroi_odor_stats[
                    # TODO TODO am i also only selecting (cell, odor) *pairs* that
                    # count as responding, not just *cells* that respond to any odor???
                    flyroi_odor_stats.responded &
                    # this is subsetting down to just full mix + (hardcoded) "top"
                    # component, consistent w/ what i'm currently doing on model data
                    # above
                    flyroi_odor_stats.odor.isin(model_mix_resps.odor.unique())
                ]
                flyroi_odor_stats = flyroi_odor_stats.drop(columns='responded')

                # currently we are filtering to only have responding (KC, odor) pairs
                # above (but since this is just for one limited plot of
                # within-responder response strength, and not the main plots of response
                # strength and response rate for all odors, that's ok)
                assert model_mix_resps.responded.all()
                model_mix_resps = model_mix_resps.drop(columns='responded')

                # TODO use earlier hardcoded def?
                model_stats = [x for x in model_mix_resps.columns if 'num_spikes' in x]
                model_mix_resps = model_mix_resps.melt(
                    value_vars=model_stats, var_name='stat',
                    id_vars=[x for x in model_mix_resps.columns if x not in model_stats]
                )

                # TODO delete this concatenation, if i'm just going to plot model vs KC
                # stuff separately below?
                response_strengths = pd.concat(
                    [flyroi_odor_stats, model_mix_resps], ignore_index=True
                )
                # TODO refactor to share w/ other filling
                if not simplify_models:
                    response_strengths['connectome_apl'] = \
                        response_strengths.connectome_apl.fillna(False)

                response_strengths['source'] = response_strengths.source.fillna('KCs')
                response_strengths['model_pnkc_class'] = \
                    response_strengths.model_pnkc_class.fillna('KCs')
                #
                # TODO TODO zscore these? or how to align? based on min/max?
                # (since we are limiting all KC data to responses, might make sense to
                # have min->0) (should i even need to do anything in logistic scaled
                # case? currently subtracting KC threshold (min, really) when comparing
                # against raw model 'num_spikes', inside plotting fn)

                # TODO rename either this or the var before the loop
                # (currently renaming one before loop at last second to all_*)
                # NOTE: (delete) still have boutons here
                # TODO TODO TODO am i accidentally unconditionally dropping
                # connectome-APL=False stuff below (and thus dropping boutons that way)
                # (even in `-o -f` case)?
                model_response_strengths = response_strengths[
                    response_strengths.source != 'KCs'
                ]
                kc_response_strengths = response_strengths[
                    response_strengths.source == 'KCs'
                ]

                def plot_response_strength_dist_per_model(data,
                    model_only: bool = False, **kwargs) -> None:

                    if not simplify_models:
                        # make below conditional if this fails
                        assert 'connectome_apl' in data.columns

                    odor_linestyle = kwargs.pop('odor_linestyle', False)
                    have_model = has_model(data)
                    if model_only:
                        assert have_model

                    model_stat = None
                    if have_model:
                        model_stats = data.stat.unique()
                        assert len(model_stats) == 1, f'{model_stats=}'
                        model_stat = model_stats[0]

                    ax = plt.gca()

                    if not model_only:
                        if have_model:
                            # (delete) this inheriting the logscale (if present) from
                            # parent ax? fix if not! (compare to kc-only one?)
                            # (yea, seems to be)
                            kc_ax = ax.twiny()
                        else:
                            kc_ax = ax

                        assert not data.odor.isna().any()
                        odors = data.odor.unique()
                        if odor_linestyle:
                            assert len(odors) == 2
                        else:
                            assert len(odors) == 1

                        kdf = kc_response_strengths[
                            kc_response_strengths.odor.isin(odors)
                        ].copy()
                        assert len(kdf) > 0

                        # TODO refactor to share this unique-stat-getting w/ below (in
                        # calling fn too) (including mean_prefix stripping)
                        kc_stats = kdf.stat.unique()
                        assert len(kc_stats) == 1, f'{kc_stats=}'
                        kc_stat = kc_stats[0]
                        # TODO delete eventually? just to check we aren't also doing the
                        # threshold subtracting outside now
                        assert kc_stat == 'mean_Fc_zscore'

                        # TODO do this back outside (on whole KC df, w/
                        # .stat.str.replace)? matter? previously was doing this only
                        # when subtracting threshold (min), but also was unconditionally
                        # doing that at the time
                        kc_stat = kc_stat.replace(mean_prefix, '')

                        # TODO TODO even make sense in this case? pretty sure it doesn't
                        # when we are comparing against the logistic scaled data, right?
                        # TODO still add flag to decide whether to do this, even for
                        # num_spikes?
                        if model_stat == 'num_spikes':
                            # TODO add assertion kc_min is thresh (at least if using
                            # remy's data, where it's one single thresh. couldn't do
                            # same for yang's)?
                            # TODO in yang's case, maybe i should be subtracting
                            # something other than just min tho? can the threshold
                            # really be reduced to one number in her case? matter?
                            kc_min = kdf.value.min()
                            assert not np.isclose(kc_min, 0)
                            # TODO warn we are doing this? what was reason i thought it
                            # made sense again?
                            kdf.value -= kc_min
                            assert np.isclose(kdf.value.min(), 0)
                            kc_stat += ' - threshold'

                    def assert_all_mix_after_component(df: pd.DataFrame) -> None:
                        odors = df.odor.unique()
                        # only should be called in odor_linestyle=True case
                        assert len(odors) == 2, f'{odors=}'
                        mix = None
                        comp = None
                        for x in odors:
                            if is_mix(x):
                                assert mix is None, 'multiple odors w/ is_mix(x)=True'
                                mix = x
                            else:
                                comp = x
                        assert mix is not None, 'no odor w/ is_mix(x)=True'
                        assert comp is not None
                        last_component_idx = (df.odor == comp)[::-1].idxmax()
                        first_mix_idx = (df.odor == mix).idxmax()
                        assert last_component_idx < first_mix_idx, 'need to sort!'

                    if not model_only:
                        exclude_kws = ('color', 'label')
                        kc_kws = {
                            k: v for k, v in kwargs.items() if k not in exclude_kws
                        }
                        # TODO is area under curve really same for all these (probably
                        # was, but that's cause units of x axes were pretty diff, w/
                        # #-spikes on more visually different one, so easier to get
                        # integral of 1 earlier.... what's solution? something w/
                        # percentiles? other normalization options in histplot i'm using
                        # now?)? do i need to share some axes i'm not for that to make
                        # sense? common_norm=False not doing it? (all kdeplot calls, for
                        # KC and model below, should have that, right?) (see
                        # response-strength_dists_kiwi.pdf. control too honestly)
                        kc_kws.update(dict(ax=kc_ax, color=kc_color, fill=False,
                            label='KCs'
                        ))
                        if not odor_linestyle:
                            # TODO delete
                            #sns.kdeplot(data=kdf, **kc_kws)
                            distplot(data=kdf, **kc_kws)
                        else:
                            assert_all_mix_after_component(kdf)
                            for odor, gdf in kdf.groupby('odor', sort=False):
                                # TODO refactor to share w/ below? (+ legend fixing)
                                linestyle = (mix_linestyle if is_mix(odor) else
                                    component_linestyle
                                )
                                # TODO TODO change label to include bit about mix? (+
                                # component) (and to hide KCs label if that's the only
                                # color being plotted? via removing label='KCs' from
                                # kc_kws above)
                                # (or just put mix + comp info in title?)
                                # TODO delete
                                #sns.kdeplot(data=gdf, linestyle=linestyle, **kc_kws)
                                distplot(data=gdf, linestyle=linestyle, **kc_kws)

                        kc_ax.set_xlabel(f'KC {kc_stat}', color=kc_color)
                        # TODO refactor to share w/ twinx mb_model APL Vm plotting?
                        # alpha not supported here
                        kc_ax.tick_params(axis='x', color=kc_color)
                        for text in kc_ax.xaxis.get_ticklabels():
                            # alpha also not supported here
                            text.set_color(kc_color)
                        #

                        # TODO set color of top spine / ticks, like in APL Vm plotting
                        # elsewhere?
                        kc_ax.spines['right'].set_visible(False)
                        kmin = kdf.value.min()
                        kmax = kdf.value.max()
                        kc_ax.set_xlim([kmin - 0.5, kmax + 0.5])

                    if not have_model:
                        return

                    group_cols = ['source']

                    have_connectome_apl = has_connectome_apl(data)
                    if simplify_models:
                        assert not have_connectome_apl

                    if odor_linestyle:
                        assert not have_connectome_apl

                        # TODO this fix assertion failure below? (nope, well not
                        # w/o reset_index(drop=True). with reset_index, yes)
                        # TODO even need sort, or just the reset_index?
                        data = data.sort_values(by='odor', kind='stable',
                            key=odor_sort_fn
                        ).reset_index(drop=True)

                        # TODO and why does this matter again? might just not assign to
                        # linestyles properly? am i still assuming a certain order for
                        # that?
                        assert_all_mix_after_component(data)
                        group_cols.append('odor')

                    if have_connectome_apl:
                        # so connectome_apl=False comes second, which is hopefully what
                        # makes it into legend (so i don't have the dotted lines there,
                        # just the hue)
                        data = data.sort_values(by='connectome_apl', ascending=False,
                            kind='stable'
                        )
                        group_cols.append('connectome_apl')

                    # to avoid a FutureWarning
                    if len(group_cols) == 1:
                        group_cols = group_cols[0]

                    # TODO TODO make sure we are skipping any KC data here
                    # (or just make sure we would have already returned)
                    for gn, gdf in data.groupby(group_cols, sort=False):
                        if odor_linestyle:
                            source, odor = gn
                            # TODO refactor to share w/ above?
                            linestyle = (
                                mix_linestyle if is_mix(odor) else component_linestyle
                            )
                        else:
                            if have_connectome_apl:
                                source, connectome_apl = gn
                            else:
                                source = gn
                                connectome_apl = False

                            linestyle = (
                                connectome_apl_linestyle if connectome_apl else
                                uniform_apl_linestyle
                            )

                        # TODO delete
                        #sns.kdeplot(data=gdf, ax=ax, linewidth=model_linewidth,
                        # TODO fill=False? (for at least some [which?], yes.
                        # everything?)
                        distplot(data=gdf, ax=ax, fill=False, linewidth=model_linewidth,
                            linestyle=linestyle, **kwargs
                        )

                # TODO TODO want distribution for each odor, or just for top component
                # vs mix? (currently just picking a top component for each, and only
                # analyzing 5comp case at all)
                # TODO TODO TODO both (+ pick diff palette) for one-one-per-odor plot
                # and only plot either KC or model data in one facet
                def plot_response_strength_dists(log_yscale: bool = True,
                    kc_only: bool = False, model_only: bool = False,
                    odor_rows: bool = False) -> None:
                    # TODO TODO delete odor_rows=True path
                    assert not odor_rows
                    if kc_only:
                        assert not model_only

                    # TODO (delete? fine to just have separate plots for each
                    # panel/mix?) col='mix', and loop over stat, making multiple plots
                    # w/ diff suffixes? (for now i'm only analyzing the 5comp mixes tho)
                    # TODO TODO TODO also analyze binary mixes if i'm not already? need
                    # any changes? (this comment duplicated)

                    kws = dict()
                    if odor_rows:
                        warn('odor_rows=True should be deleted! do not use')
                        kws['row'] = 'odor'

                    if not kc_only:
                        # mapped plotting fn finds and plots the relevant KC data, so
                        # model_only=True is set as kwarg to that call below, if we want
                        # to have it not do that
                        data = model_response_strengths
                        if not odor_rows and has_connectome_apl(data):
                            # TODO make a separate plot with the connectome_apl=True
                            # data in this case? prob don't really care to...
                            warn('plot_response_strength_dist: subsetting data to '
                                'connectome_apl=False, for odor_rows=False version of '
                                'the plot. linestyle would be ambiguous between '
                                'uniform-vs-connectome APL and mix-vs-component'
                            )
                            data = data[data.connectome_apl == False]

                        kws.update(dict(
                            col='stat', col_order=model_stat_order,
                            hue='model_pnkc_class', palette=source_palette,
                        ))
                    else:
                        # TODO need anything in kws? assert odor_linestyle (or only make
                        # that version when calling outside, instead of all pairwise
                        # combos of all 3 flags?)
                        data = kc_response_strengths

                    # TODO sharey=True if odor_rows and not log_yscale?
                    # see comment about handling ylim where it is set below
                    g = sns.FacetGrid(data=data, sharey=False, sharex=False, **kws)

                    plot_kws = dict()
                    if not odor_rows:
                        plot_kws['odor_linestyle'] = True

                    g.map_dataframe(plot_response_strength_dist_per_model, x='value',
                        alpha=model_alpha, log_scale=(False, log_yscale),
                        model_only=model_only, **plot_kws
                    )

                    # TODO TODO what to set this to if not log_yscale? leave it?
                    # want sharey=True then?
                    if log_yscale:
                        # need anything slightly above 1 (since log_scale=True)? yes
                        # (should be <10 tho)
                        g.set(ylim=(1e-5, 5))
                    # TODO delete
                    else:
                        print('how to handle yscale in non-log case? sharey=True?')
                    #

                    g.fig.subplots_adjust(hspace=0.6)

                    # TODO TODO version of this plot including nonresponding values too
                    # (mainly for sanity checking)?
                    suptitle = (f'{panel}\nactivation strengths across KCs\n'
                        'responder (KC, odor) pairs only'
                    )
                    g.fig.suptitle(suptitle, y=1.10)

                    add_fixed_legend(g, data, lines=True, odor_linestyle=not odor_rows)

                    g.set_titles('')

                    if odor_rows:
                        assert len(g.row_names) > 1
                        assert g.axes.shape[0] == len(g.row_names)
                    else:
                        # it should be this value when row= not specified in FacetGrid
                        # init
                        assert g.row_names == []
                        assert g.axes.shape[0] == 1

                    # TODO delete
                    print()
                    print('plot_response_strength_dist_per_model:')
                    print(f'{kc_only=} {model_only=} {log_yscale=}')
                    #
                    for (i, j, hue), gdf in g.facet_data():
                        if hue != 0:
                            continue

                        ax = g.axes[i, j]
                        ylabel = ''
                        if j == 0:
                            ylabel = 'density across KCs'
                            if odor_rows:
                                ylabel = f'{g.row_names[i]}\n{ylabel}'

                        if i != 0:
                            ax.set_title('')

                        ax.set_ylabel(ylabel)
                        ax.tick_params(labelbottom=True)

                        if g.col_names == []:
                            assert kc_only
                            # TODO refactor all this w/ other def of kc_stat?
                            stats = data.stat.unique()
                            assert len(stats) == 1
                            stat_name = stats[0]
                            stat_name = stat_name.replace(mean_prefix, '')
                            #
                        else:
                            assert not kc_only
                            stat_name = g.col_names[j]
                        ax.set_xlabel(stat_name)

                        mmin = gdf.value.min()
                        mmax = gdf.value.max()
                        # TODO 0.05 instead of 0.25?
                        # TODO TODO does this also work for the KC-only plots? or is
                        # facet data not set up right for some reason?
                        # TODO delete
                        print(f'{stat_name=}')
                        print(f'{mmax=}')
                        print(f'{mmin=}')
                        #
                        margin = (mmax - mmin) * 0.05
                        ax.set_xlim([mmin - margin, mmax + margin])
                    # TODO delete
                    print()
                    #

                    fname = f'response-strength_dists_{panel}'

                    plot_dir = plot_root
                    if kc_only:
                        # will help declutter each model-subsetting-option-specific
                        # subdir (and these would be same across each of those anyway)
                        plot_dir = model_root
                        fname += '_kc-only'

                    if model_only:
                        fname += '_model-only'

                    if odor_rows:
                        fname += '_odor-rows'

                    if log_yscale:
                        fname += '_logy'

                    savefig(g, plot_dir, fname)


                # TODO TODO TODO also analyze binary mixes if i'm not already? need
                # any changes? (this comment duplicated)
                # TODO TODO TODO versions (one for KC, one for best model. necessarily
                # on diff axes? unless diff linestyle/something?), where it's all
                # components and mix, w/ hue for that? define distinct palette for that
                if not unrestricted_full_model_params:
                    bools = [False, True]
                    for log_yscale, kc_only in product(*([bools] * 2)):
                        plot_response_strength_dists(log_yscale=log_yscale,
                            kc_only=kc_only
                        )
                        # TODO TODO keep? mainly to check bounds of model dists are not
                        # clipped/whatever too badly on kc+model plot
                        # TODO TODO make sure xlim set to include all model data (or
                        # just not set?), to sanity check KDE bounds, etc?
                        if kc_only:
                            continue
                        plot_response_strength_dists(log_yscale=log_yscale,
                            model_only=True
                        )
                        #
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

            # TODO TODO TODO (?) regen response class mean/count plots (do in here?
            # satisfied leaving that to natmix_data/analysis.py [just make sure it says
            # what threshold it used in plots, and that it's same as what i'm using
            # here, for both model and KC stuff])

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
                # TODO TODO double check that (before we strip concs bove), 0 really is
                # the one w/ the highest concs
                kc_df = kc_df[kc_df.pair_dilution_factor == 0]

            pdf = pd.concat([pdf, kc_df], ignore_index=True)

            # TODO still want to (try?) normalizing mean_response_rate for each, even
            # though in theory those could be directly comparable?

        # TODO TODO also include ORNs too ideally (in plots w/ KCs vs model)? for at
        # least some plots? maybe do all in natmix_data/analysis? (maybe just vs KCs)?

        kc_pdf = pdf[pdf.source == 'KCs']
        # TODO maybe some / all of below should be moved into `if kc_panel is not None`
        # conditional immediately above?
        if panel not in natmix_panels:
            if kc_panel is not None:
                assert len(kc_pdf) > 0
                plot_panel_stats_across_models(kc_pdf, panel, f'{suffix}_kc-only')

            plot_panel_stats_across_models(pdf, panel, suffix)
        else:
            assert kc_panel is not None
            assert len(kc_pdf) > 0

            # TODO subsetting to 0 already done?
            # pair_dilution_factor is NaN for model data (and only model data)
            pdf = pdf[(pdf.pair_dilution_factor == 0) | pdf.pair_dilution_factor.isna()
                ].copy()

            model_pdf = pdf[pdf.source != 'KCs'].copy()

            assert set(kc_pdf.mix.unique()) == {'binary', '5comp'}
            # grouping by mix will drop model stuff, because it can't easily be assigned
            # just one value for that (would need to duplicate stuff shared between the
            # two)
            for mix, kc_mdf in pdf.groupby('mix'):
                assert (kc_mdf.source == 'KCs').all()
                plot_panel_stats_across_models(kc_mdf, panel, f'{suffix}_{mix}_kc-only')

                model_mdf = model_pdf[model_pdf.odor.isin(kc_mdf.odor.unique())]
                mdf = pd.concat([kc_mdf, model_mdf], verify_integrity=True)
                plot_panel_stats_across_models(mdf, panel, f'{suffix}_{mix}')


            if analyze_orn or analyze_eag:
                assert analyze_orn, 'some code below currently assumes this'
                df1 = kc_pdf[
                    ~kc_pdf.stat.str.startswith(NORM_PREFIX) & (kc_pdf.mix == '5comp') &
                    (kc_pdf.stat == 'mean_Fc_zscore')
                ].copy()
                df1 = add_group_id(df1, fly_cols, 'fly_id')

            if analyze_orn:
                df2 = orn_intensity[
                    (orn_intensity.index.get_level_values('panel') == panel) &
                    # TODO remove separate calculations of EAG/ORN normalization above?
                    # seems some stuff got screwed up doing that before subsetting odors
                    # again here (or something like that)
                    ~orn_intensity.index.get_level_values('is_normalized')
                ].reset_index()
                df2 = df2.drop(columns='is_normalized')
                df2 = add_group_id(df2, fly_cols, 'fly_id')
                df2 = df2.drop(columns=fly_cols)
                # df1/3 don't have this binary mixture
                df2 = df2[~df2.odor.str.contains('+', regex=False)]

                df1 = df1.drop(columns=list(set(df1.columns) - set(df2.columns)))
                assert set(df1.columns) == set(df2.columns)

                dfs = [df1, df2]
                fname_parts = ['kc', 'orn']

            if analyze_eag:
                df3 = eag_intensity[
                    (eag_intensity.index.get_level_values('panel') == panel) &
                    ~eag_intensity.index.get_level_values('is_normalized')
                ].reset_index()
                df3 = df3.drop(columns='is_normalized')
                assert set(df2.columns) == set(df3.columns)

                dfs.append(df3)
                fname_parts.append('eag')

            if analyze_orn or analyze_eag:
                # TODO assert df1/2/3 all have same set of odors
                # (they should now anyway)
                dfs = [normalize_one_panel(x) for x in dfs]

                dfs_with_n = []
                new_palette = dict()
                for curr_df in dfs:
                    assert not curr_df.fly_id.isna().any()
                    n_flies = curr_df.fly_id.nunique()
                    n_suffix = f' (n={n_flies})'
                    # TODO clean up?
                    source = curr_df.source.unique()[0]
                    with_n = source + n_suffix
                    new_palette[with_n] = source_palette[source]

                    curr_df['source'] = curr_df.source + n_suffix
                    dfs_with_n.append(curr_df)

                intensity_comp_df = pd.concat(dfs_with_n, ignore_index=True)

                intensity_comp_df = sort_odors(intensity_comp_df,
                    panel2name_order=panel2name_order
                )

                # TODO dedupe w/ where i copied from?
                err_kws = dict(linewidth=1.5)
                fig, ax = plt.subplots()
                # TODO something more distinct than gray (wrt brown)
                sns.pointplot(intensity_comp_df, ax=ax, x='odor', y='value',
                    # TODO use float dodge? seems ok as is? may want a bit more?
                    hue='source', palette=new_palette, marker='o', dodge=True,
                    markerfacecolor='none', linestyle='none', seed=1, err_kws=err_kws,
                    errorbar=('ci', 95),
                    # TODO need any other kwargs that were passed in before?
                    capsize=0, legend=True
                )
                ax.set_ylabel('normalized intensity (max mean -> 1)'
                    '\n(with 95% CI on mean)'
                )
                ax.set_title(f'intensity comparison\n{panel}')
                savefig(fig, model_root, f'{"-".join(fname_parts)}_intensity_{panel}')
            else:
                warn('not comparing intensity to EAG/ORN data!')

    mean_model_response_rate = pd.concat(mean_model_response_rate_list,
        verify_integrity=True
    )

    # don't want to overwrite these outputs unless currently analyzing all models
    if unrestricted_full_model_params:
        model_mean_mix_supp = pd.concat(model_mean_mix_supp_sers, verify_integrity=True)
        assert (
            len(model_mean_mix_supp.shape) == 1 and model_mean_mix_supp.name == diff_col
        )

        # values should now be the diff_col values for each stat
        model_mean_mix_supp = model_mean_mix_supp.unstack('stat')
        assert set(model_mean_mix_supp.columns) == set(MODEL_STAT_COLS)
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
            model_mean_mix_supp.panel.isin(natmix_panels)
        ].groupby(model_cols)[mixsupp_cols].mean()

        # TODO delete. isn't sufficient to have first 3 index levels be the same =(
        #
        # re-ordering levels, so that we can index from  model_acrosspanel_mean_mix_supp
        # more easily
        #model_mean_mix_supp = model_mean_mix_supp.set_index(model_cols + [
        #    x for x in nonstat_cols if x not in model_cols
        #])
        model_mean_mix_supp = model_mean_mix_supp.set_index(model_cols)

        for stat in mixsupp_cols:
            # TODO just sort by kiwi/control (doing that)? and another version by all?
            # one one version sorted just by binary stuff?
            sorted_by_stat = model_acrosspanel_mean_mix_supp.sort_values(by=stat,
                kind='stable'
            )
            model_dirnames = sorted_by_stat.index.map(model_ids)
            sorted_by_stat['model_dirname'] = model_dirnames

            sorted_mixsupp = model_mean_mix_supp.loc[sorted_by_stat.index].reset_index()
            stat_for_fname = stat.replace('_', '-')
            fname_prefix = f'{sorted_mixsupp_fname_prefix}{stat_for_fname}'
            to_csv(sorted_mixsupp, model_root/ f'{fname_prefix}.csv', index=False)
            to_parquet(sorted_mixsupp, model_root / f'{fname_prefix}.parquet')

            # TODO (delete) also include mean megamat 2h-whatever correlation (from
            # tuned model dir) for each of these (to see if any of these solve the
            # general problem? and fraction of segmenting cells?)
            # TODO (delete) or at least manually check what those outputs look like for
            # the best models here (not dramatically improved, at least as currently
            # calculated, or w/ spike counts + gaussian noise)

            sorted_by_stat = sorted_by_stat.reset_index()
            to_csv(sorted_by_stat, model_root / f'{fname_prefix}_panelmean.csv',
                index=False
            )
            to_parquet(sorted_by_stat, model_root / f'{fname_prefix}_panelmean.parquet')

            # TODO move plotting code down here? (from max_supp_models_only
            # case above) (once it's finalized, at least)
    else:
        warn(f'not writing model order, sorted by mean {diff_col}, because need '
            '-f and NOT -m for that'
        )

    # contains both model and KC 5comp kiwi/control data
    class_fracs = pd.concat(natmix_panel_class_frac_list, verify_integrity=True)

    # TODO second log-yscale version of this?
    #
    # TODO TODO maybe, for myself, show one w/ two thresholds 0.8 / 1.5 on remy's data?
    # TODO maybe change black fly points to same color as KCs/ORNs/whatever estimate, to
    # add ORN data with fly points as well
    #
    # TODO TODO do for yang's data too somehow? maybe w/ a separate row for each
    # mix?

    # TODO TODO TODO also compare ORNs vs KCs (sharing ORN thresh w/
    # natmix_data/analysis.py) (still need to compute thresholded ORN data)

    # TODO define Fc_zscore part from getting KC stat, not hardcoding
    title = (f'observed KC mean response rate: {kc_no_dilution_mean_resp_rate:.3g}\n'
        f'(with a mean Fc_zscore threshold of {NATMIX_KC_THRESH:.2f}'
    )
    if only_analyzing_few_models:
        natmix_mean_model_response_rates = mean_model_response_rate.loc[
            list(natmix_panels)].groupby(level=PNKC_CLASS_COL, sort=False).mean()

        model_rr_strs = ['\nmean model response rates:']
        for pnkc_class, mean_resp_rate in natmix_mean_model_response_rates.items():
            model_rr_strs.append(f'{pnkc_class}={mean_resp_rate:.3g}')
        model_rr_str = ' '.join(model_rr_strs)
        title += model_rr_str

    # TODO TODO fix so these plots also show source_palette value for KCs? is
    # source NaN in class_fracs or something? (doesn't seem so)
    plot_response_class_summary(class_fracs, plot_root, title=title, hue=PNKC_CLASS_COL,
        palette=source_palette, model_marker_kws=model_marker_kws, alpha=model_alpha,
        facet_kws=dict(height=4, aspect=1.2), jitter=0.3,
        # TODO TODO work? (also do for calls in natmix_data analysis, or do
        # cg.add_legend() by default [or if legend=True?]?)
        # TODO TODO why is this added twice now? (without this i don't think there
        # is any legend) (it seems it might just be the model points duplicated)
        #call_on_grids_before_save=lambda x: add_fixed_legend(x,
        #    class_fracs.reset_index(), lines=False)
    )

    class_fracs_kconly = class_fracs.loc[
        class_fracs.index.get_level_values('source') == 'KCs'
    ]
    # TODO TODO say N flies (+ROIs?) (for each panel) somewhere. share code for that
    # from natmix_data/analysis.py?
    plot_response_class_summary(class_fracs_kconly, model_root, title=title,
        hue=PNKC_CLASS_COL, palette=source_palette, model_marker_kws=model_marker_kws,
        alpha=model_alpha, facet_kws=dict(height=4, aspect=1.2), fname_suffix='_kc-only'
    )

    plot_panel_stats_across_models(tdf, 'megamat', suffix)


if __name__ == '__main__':
    main()

