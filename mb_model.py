#!/usr/bin/env python3
"""
Wrappers to facilitate running `olfsysm` MB models, mainly:
- `model_mb_responses`: highest-level function. takes multiple-fly ORN dF/F data and
  runs multiple parameterizations of the model (as well as saving plots and other
  outputs to subdirs).

- `fit_and_plot_mb_model`: takes mean ORN data (already scaled into units of spike rate
  deltas), and runs one parameterization of the model (making multiple `fit_mb_model`
  calls only when multiple seeds are needed), saving plots/etc in a created directory

- `fit_mb_model`: the loosest wrapper around `olfsysm`. returns model outputs, but
  typically does not make any plots (though there is some debug code for that).
  when needed, it:
  - fills glomeruli to intersection of those in hemibrain and Task et al. 2022
  - imputes mean Hallem SFR

See also docstring for `connectome_wPNKC` below.repo_root
"""

from ast import literal_eval
import filecmp
import itertools
from io import StringIO
import inspect
import json
import os
from os.path import getmtime
from pathlib import Path
from pprint import pprint, pformat
import re
import sys
import shutil
from tempfile import NamedTemporaryFile
import time
import traceback
from typing import (Any, Dict, Callable, Hashable, List, Literal, Optional, Set,
    Sequence, Tuple, Union, Iterable
)
import warnings

import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgba
import matplotlib as mpl
# TODO add CLI arg to switch this off?
mpl.use('Agg')
from sklearn.cluster import DBSCAN
from scipy.stats import spearmanr, pearsonr, f_oneway, percentileofscore
from scipy.interpolate import interp1d
from sklearn.preprocessing import maxabs_scale as sk_maxabs_scale
from sklearn.preprocessing import minmax_scale as sk_minmax_scale
import statsmodels.api as sm
import seaborn as sns
import colorcet as cc
from tqdm import tqdm
from termcolor import cprint
# for type hinting
from statsmodels.regression.linear_model import RegressionResultsWrapper
from rastermap import Rastermap

from drosolf import orns
from hong2p import olf, util, viz
from hong2p.olf import solvent_str
from hong2p.viz import dff_latex, no_constrained_layout
from hong2p.util import (num_notnull, num_null, pd_allclose, format_date, date_fmt_str,
    reindex, is_scalar, pd_index_equal
)
from hong2p.types import Pathlike, DataFrameOrSeries, CMap, KwargDict
from natmix import drop_mix_dilutions
import olfsysm as osm

# TODO what's purpose? needed? delete?
import faulthandler, sys
faulthandler.enable(file=sys.stderr, all_threads=True)
#

# NOTE: can't import from al_analysis w/o causing circular import issues, so need to
# factor shared stuff to al_util
from al_util import (savefig, abbrev_hallem_odor_index, sort_odors, panel2name_order,
    diag_panel_str, warn, should_ignore_existing, to_csv, to_pickle, read_pickle,
    produces_output, makedirs, corr_triangular, invert_corr_triangular, n_choose_2,
    diverging_cmap, diverging_cmap_kwargs, bootstrap_seed, mean_of_fly_corrs, plot_corr,
    plot_responses_and_corr, rotate_xticklabels, cluster_rois, odor_is_megamat,
    megamat_odor_names, remy_data_dir, remy_binary_response_dir, remy_megamat_sparsity,
    remy_date_col, remy_fly_cols, remy_fly_id, remy_fly_binary_response_fname,
    load_remy_fly_binary_responses, load_remy_megamat_kc_binary_responses,
    n_final_megamat_kc_flies, MultipleSavesPerRunException, print_curr_mem_usage
)
import al_util


# TODO move to hong2p.types? already have something like that there?
ParamDict = Dict[str, Any]

warnings.filterwarnings('error', category=pd.errors.SettingWithCopyWarning)
# TODO does this not actually trigger in pytest? if not, why not (and is there a
# workaround?)?
warnings.filterwarnings('error', category=FutureWarning)
# (doesn't work to catch numpy percentile interpolation= warning anyway,
# at least from pytest w/ or w/o pytest ignoring warnings...)
# TODO restore
#warnings.filterwarnings('error', category=DeprecationWarning)
# also doesn't seem to work
# warnings.filterwarnings('error', message='the `interpolation=` argument.*')
#

# TODO add fn to load megamat fixed_thr / wAPLKC params used for model in paper?
# (-> try using in some other contexts, like for yang's diagnostic mixture model)
# TODO + include test checking output against params currently loaded but not used in
# hemibrain repro test? or don't, if i load from there in the first place... then would
# be a tautology

repo_root = Path(__file__).parent

# e.g. before calculating correlations across model KC populations.
#
# Remy generally DOES drop "bad" cells, which are largely silent cells, but that isn't
# controlled by this flag. my analysis of her data generally also drops the same cells
# she does.
drop_silent_model_kcs: bool = True

KC_ID: str = 'kc_id'
# TODO use (+ in test_mb_model.py and elsewhere) (replacing hardcoded str w/ same value)
KC_TYPE: str = 'kc_type'

CLAW_ID: str = 'claw_id'
PN_ID: str = 'pn_id'
BOUTON_ID: str = 'bouton_id'

glomerulus_col: str = 'glomerulus'

# in all Prat's "v5" outputs that deal with PN data
bouton_col: str = 'anatomical_bouton'

# bouton_id are only unique within each pn_id
bouton_cols = [PN_ID, BOUTON_ID]

# TODO rename to hemibrain or something? presumably this is specfic to that?
# TODO move into the one place that uses it (tianpei's part of connectome_wPNKC)
# (or keep module level and share w/ his script PNKC_claw_plots_dif_color.py?)
PIXEL_TO_UM = 8/1000


# TODO support arbitrary # of inputs? (how to type hint all elements of e.g. *args are
# of same type?)
# TODO move to hong2p.util?
def dict_seq_product(dict_seq1: Sequence[dict], dict_seq2: Sequence[dict], *,
    allow_key_overlap: bool = False) -> List[dict]:
    """Returns a sequence of dicts, by combining product of input dict sequences.

    Args:
        allow_key_overlap: if False, will raise ValueError if inputs share keys,
            otherwise value from last input should be used.

    >>> dict_seq_product([dict(), dict(a=1)], [dict(b=1), dict()])
    [{'b': 1}, {}, {'a': 1, 'b': 1}, {'a': 1}]
    """
    output_seq = []
    for ds in itertools.product(dict_seq1, dict_seq2):
        combined = dict()
        for d in ds:
            if not allow_key_overlap and any(k in combined.keys() for k in d.keys()):
                # TODO say which keys overlapped?
                raise ValueError('inputs had overlapping keys')
            combined.update(d)
        output_seq.append(combined)
    return output_seq


# TODO move to hong2p?
# TODO type hint for class (if returning err raised)?
def eval_and_check_err(v: str) -> Tuple[Any, bool]:
    assert isinstance(v, str)
    # TODO doc
    err = None
    try:
        ev = literal_eval(v)
        assert not isinstance(ev, str), f'{ev=}\n{type(ev)}'
        raised_err = False

    # to catch:
    # ValueError: malformed node or string: <_ast.Name object at ...
    # which (so far) I've only seen with string input (e.g. 'x', but not
    # repr('x') = "'x'")
    #
    # but also SyntaxError's about unexpected EOF's, which might be unavoidable?
    # maybe just without repr?
    # TODO TODO how did i get this error? how to fix? (repr?)
    #
    # test/test_mb_model.py:1327: in test_fixed_inh_params_fitandplot
    #     assert_param_csv_matches_returned(plot_root, params2,
    # test/test_mb_model.py:1247: in assert_param_csv_matches_returned
    #     p2 = read_param_csv(model_output_dir)
    # mb_model.py:300: in read_param_csv
    #     params = read_series_csv(model_output_dir / 'params.csv', convert_dtypes=True)
    # mb_model.py:230: in read_series_csv
    #     ev = literal_eval(v)
    # one-row-per-claw_True__prat-claws_True__pn-claw-to-APL_True__connectome-APL_True__fixed-thr_207__wAPLKC_5.44
    # SyntaxError: unexpected EOF while parsing
    #
    # TODO diff err code for SyntaxError? return int 0,1,2 instead? or
    # None/err-type?
    except (ValueError, SyntaxError) as err:
        raised_err = True
        # TODO can i just always include the repr call?
        # expecting this not to raise? maybe it will sometimes though?
        ev = literal_eval(repr(v))
        # TODO any other cases besides str that raise (a similar) initial
        # ValueError?
        assert isinstance(ev, str), f'{ev=}\n{type(ev)}'

    # TODO delete if this always works. checking whether we can always call
    # repr first, and then maybe even avoid error handling above.
    if not raised_err:
        ev2 = literal_eval(repr(v))
        # TODO need more general than == here?
        # TODO will this only ever be true if ev is a str? or maybe just fails
        # in list(/iterable) cases, for some reason? failed on:
        # ev=[174.7258591209665, 172.12329035345402]
        # ...!=...ev2='[174.7258591209665, 172.12329035345402]'
        #assert ev == ev2, f'{ev=}\n...!=...{ev2=}'
        if isinstance(ev2, str):
            assert ev != ev2, f'{ev=}\n{ev2=}\n{type(ev)=}\n{type(ev2)=}'
        else:
            # TODO TODO include other cases besides str in above, if needed, or
            # remove this `if not raised_err` check altogether
            assert ev == ev2, f'{ev=}\n{ev2=}\n{type(ev)=}\n{type(ev2)=}'
    #

    # TODO prob return error type or None instead of raised_err
    # TODO or figure out what i want to do in here (and if i need to ever test
    # this outside, and just keep internal)
    return ev, raised_err


def eval_and_check_compatible(v: str, v2: str) -> Tuple[Any, Any]:
    # TODO doc
    ev, v_raised_err = eval_and_check_err(v)
    ev2, v2_raised_err = eval_and_check_err(v2)

    if v_raised_err:
        assert v2_raised_err
    else:
        assert not v2_raised_err

    if isinstance(ev, str):
        assert isinstance(ev2, str)
    else:
        assert not isinstance(ev2, str)
        # TODO also check same type at output? or one a subclass of other?

    return ev, ev2


# TODO move to hong2p?
def eval_and_check_equal(v: str, v2: str) -> Any:
    """Asserts `literal_eval` outputs of two strings are equal, returns value.
    """
    ev, ev2 = eval_and_check_compatible(v, v2)

    # TODO use `equals` instead? or (probably) want to only be evaling simple python
    # types
    if not isinstance(ev, str):
        # handles (at least) cases like lists-of-floats
        assert ev2 == ev, f'{ev2=} != {ev}'
    else:
        # TODO delete? this work? just to justify not having to special case
        # assignment (where this is used in read_pd_series). even need this
        # (don't think so?)?
        assert v == ev, f'{v=} != {ev=} (do not need this assertion)'
        assert v2 == ev2, f'{v2=} != {ev2=} (do not need this assertion)'
        #
        assert v2 == v, f'{v2=} != {v}'

    # TODO may need to return v instead sometimes, if unclear assertions in else branch
    # above fail
    return ev


# TODO test convert_dtypes=True default doesn't break anything, and make it default
def read_series_csv(csv: Pathlike, convert_dtypes: bool = False, **kwargs) -> pd.Series:
    # TODO doc
    """
    Args:
        convert_dtypes: writes loaded Series to temporary DataFrame, re-reads that (so
            that default `read_csv` type inference does what we typically want), and
            converts back to a Series. If not set True, values should all be str.
    """
    # TODO add option to drop a header (after using for name of series?)
    # (-> use for prat clean KC ID csv, used in prat_claws condition)
    # (where/how to repro exactly?)
    df = pd.read_csv(csv, header=None, index_col=0, **kwargs)
    assert df.shape[1] == 1
    ser = df.iloc[:, 0].copy()

    # these should both be autogenerated, and not actually in CSV
    assert ser.index.name == 0
    assert ser.name == 1
    ser.index.name = None
    ser.name = None

    # [Series|DataFrame].[infer_objects|convert_dtypes] nor pd.to_numeric did not seem
    # to be able to convert strings like I wanted. not sure there is a more idiomatic
    # way to infer dtypes nicely for a one column input (where that column will have
    # mixed types).
    if convert_dtypes:
        # the reason I added this convert_dtypes code in the first place
        # TODO tho maybe missing value could still show up as None? or empty str?
        assert all(type(x) is str for x in ser.values), f'{ser.values=}'

        # If we write as a CSV where each key is a column rather than a row, default
        # type inference of read_csv should do all/most of what I want. Just need to
        # convert back to a Series after.
        # TODO want (any of?) **kwargs passed to this read_csv call too? presumably
        # passing all would tend to cause issues, and not sure i care to filter out some
        # set we should pass through. could add separate optional dict for kwargs here,
        # if really needed
        ser2 = pd.read_csv(StringIO(ser.to_frame().T.to_csv(index=False)),
            # default None (='high') (and presumably also 'legacy' low precision)
            # converter would cause repr checks below to fail (just in very last of many
            # digits in floats. not really a meaningful difference.)
            float_precision='round_trip'
        ).iloc[0]
        # index.name already None
        ser2.name = None

        assert ser.keys().equals(ser2.keys())

        # mostly just have the list calls for paranoia surrounding changing something
        # I'm iterating over
        for k, v, v2 in zip(list(ser.keys()), list(ser.values), list(ser2.values)):
            # otherwise we will get repr('x') == "'x'" != 'x' below
            if isinstance(v2, str):
                ev = eval_and_check_equal(v, v2)

                # TODO need to duplicate this isinstance check w/ eval_and_check_equal?
                # could just unconditionally assign value in to series, assuming other
                # assertions i added (that `v == ev` and `v2 == ev2`) pass (unclear)
                if not isinstance(ev, str):
                    ser2.at[k] = ev

                continue

            # TODO also want something like this in eval_and_check_equal? separate fn
            # wrapping more of the body of this loop (eval_and_check_equal factored out
            # from here)?
            #
            # anything more we'd want to check?
            assert repr(v2) == v, f'(converted) {repr(v2)=} != (initial str) {v=}'

        ser = ser2

    return ser


def read_param_csv(model_output_dir: Pathlike) -> ParamDict:
    # TODO to what extent will it differ? will we need to dip into the param pickle for
    # anything? anything missing from both?
    """Reads+returns `fit_and_plot_mb_model`'s 'params.csv' in same format it returns

    Args:
        model_output_dir: directory created by (and with outputs from) a single
            `fit_and_plot_mb_model` call. should contain at least 'params.csv'.
    """
    # values would all have type str w/o convert_dtypes=True
    params = read_series_csv(model_output_dir / 'params.csv', convert_dtypes=True)
    params = params.to_dict()
    return params


# TODO narrow output type? (can i change tianpei's one-row-per-claw code to also give us
# a float here? would prob be involved)
# TODO TODO change to never return wKCAPL (to check whether we actually need to specify
# that on top of wAPLKC, for one_row_per_claw=True case. may want to try to remove that
# need if it is present)
def get_APL_weights(param_dict: ParamDict, kws: ParamDict) -> Dict[str,
    Union[float, np.ndarray]]:
    """Returns dict with kwargs necessary to fix APL weights to match prior call.

    Will always contain 'wAPLKC' key, and will only currently contain 'wKCAPL' for
    one-row-per-claw case, where code may currently require this passed separately.
    """
    # TODO share defaults w/ fit_mb_model?
    one_row_per_claw = kws.get('one_row_per_claw', False)

    # TODO refactor
    variable_n_claws = kws.get('pn2kc_connections') in variable_n_claw_options

    # TODO share defaults w/ fit_mb_model?
    use_connectome_APL_weights = kws.get('use_connectome_APL_weights', False)

    # TODO just test whether we have *_scale params or not, rather than requiring kws to
    # be passed in?

    apl_params = dict()
    if use_connectome_APL_weights:
        # TODO so this assumes that wAPLKC is from connectome inside fit_mb_model, and
        # there is no current support for passing in vector wAPLKC, right?
        wAPLKC = param_dict['wAPLKC_scale']
        apl_params['wAPLKC'] = wAPLKC

        # TODO assert we are in `one_row_per_claw and not prat_claws` case
        # (tianpei=True, defined below currently) (for either of below) (if we have both
        # "" and ""_scale)?
        # TODO or will i always have wAPLKC/wKCAPL, for all calls of this fn? (remove
        # conditionals?)
        # TODO delete this assertion and let new code serializing param_dict outputs
        # (which currently asserts no ndarray) handle it?
        if 'wAPLKC' in param_dict:
            assert isinstance(param_dict['wAPLKC'], pd.Series)
        if 'wKCAPL' in param_dict:
            assert isinstance(param_dict['wKCAPL'], pd.Series)
        #
        wKCAPL = param_dict['wKCAPL_scale']
        #
    else:
        assert 'wAPLKC_scale' not in param_dict
        assert 'wKCAPL_scale' not in param_dict

        # TODO how are both this and wAPLKC_scale (and both for wKCAPL)
        # missing in one_row_per_claw=True and prat_claws=True case?
        # (b/c popped in fit_and_plot... between fit_mb_model call and where wAPLKC
        # cached shortly after) (added hack to not pop in that case, for now)
        wAPLKC = param_dict['wAPLKC']
        apl_params['wAPLKC'] = wAPLKC

        wKCAPL = param_dict['wKCAPL']

    # TODO factor out, to also use w/ fixed_thr? would need to apass variable_n_claws
    # and maybe one other flag?
    def check_weight_type(weights, _hack=True) -> None:
        # added _hack flag b/c don't want to check that for wAPLPN/wPNAPL vectors,
        # which will always have the *_scale factors, so shouldn't need this.
        # also, unlikely that code will ever really be used w/o connectome APL anyway.

        # TODO delete
        if _hack and one_row_per_claw and not use_connectome_APL_weights:
            # happens whether prat_claws=True/False, but still don't really care much
            # about these cases
            assert isinstance(weights, pd.Series)
        #
        # TODO restore
        #if not variable_n_claws:
        # TODO delete
        elif not variable_n_claws:
            #
            assert is_scalar(weights), f'{type(weights)=}'
        else:
            # TODO only check first part if n_seeds=1, and only second otherwise?
            assert is_scalar(weights) or (
                isinstance(weights, list) and all(is_scalar(x) for x in weights)
            ), f'{type(weights)=}'


    check_weight_type(wAPLKC, _hack=True)

    # TODO only do this in restricted circumstances?
    # TODO do i always have wKCAPL (will i need to default to None above, and only set
    # when we have it?)
    check_weight_type(wKCAPL, _hack=True)

    # TODO remove this bit if i can confirm tianpei's code only needs wAPLKC
    # passed in, or if i can change it to work that way (then could just return
    # single value for wAPLKC, rather than a dict) (that was more for passing output
    # to new calls, not [i think] checking output of calls [where wKCAPL included
    # regardless of whether prat_claws=True/False])
    # TODO and if only want this for tianpei path, also check prat_claws=False here?
    # TODO update prat_claws defaults here if i do in fit_mb_model too
    # (currently default = False)
    # TODO delete. need more cases than this.
    #tianpei = one_row_per_claw and not kws.get('prat_claws', False)

    # TODO delete. fixed by dividing by kc.N in olfsysm def of wKCAPL_scale from
    # wAPLKC_scale
    # TODO also include tianpei here at least?
    #if one_row_per_claw and use_connectome_APL_weights:
    #    warn('need to pass wKCAPL (on top of wAPLKC), in order to reproduce this '
    #        'output while skipping APL tuning! wKCAPL calc may be wrong in this path '
    #        'in general!'
    #    )
    #    # TODO TODO does need for this mean my calc is wrong? currently assume i
    #    # can just pass wAPLKC scalar to fit_mb_model, and derive other by dividing by
    #    # # KCs
    #    apl_params['wKCAPL'] = wKCAPL
    #

    if kws.get('prat_boutons', False) and not kws.get('per_claw_pn_apl_weights', False):
        # wAPLPN/wPNAPL will be of length # boutons (which should be # cols of wPNKC)
        # TODO will these sometimes be popped? just assert type if they are present, but
        # don't require?
        if 'wAPLPN' in param_dict:
            assert 'wPNAPL' in param_dict
            wAPLPN = param_dict['wAPLPN']
            assert isinstance(wAPLPN, pd.Series)
            wPNAPL = param_dict['wPNAPL']
            assert isinstance(wPNAPL, pd.Series)
        #
        assert 'wAPLPN_scale' in param_dict
        assert 'wPNAPL_scale' in param_dict

        wAPLPN = param_dict['wAPLPN_scale']
        wPNAPL = param_dict['wPNAPL_scale']

        check_weight_type(wAPLPN)
        check_weight_type(wPNAPL)

        apl_params['wAPLPN'] = wAPLPN
        # TODO can i get away with just setting wAPLPN (where wPNAPL will be
        # defined from it in fit_mb_model, by dividing i think by # boutons), or do i
        # need to (sometimes?) also set wPNAPL into apl_params? could get away with it
        # (as currently set up) w/ wAPLKC w/o wKCAPL
        #apl_params['wPNAPL'] = wPNAPL
    else:
        # NOTE: per_claw_pn_apl_weights=True case currently only returns wAPLKC/wKCAPL
        # (and *_scale parameters), which were either:
        # - the per-claw addition of PN<>APL and KC<>APL weights (in add_...=True case)
        # - replaced by the PN<>APL weights (in the replace...=True case)
        #
        # code would get more complex if I wanted to still return PN<>APL weights from
        # fit_mb_model in those cases.
        assert 'wAPLPN' not in param_dict
        assert 'wPNAPL' not in param_dict
        assert 'wAPLPN_scale' not in param_dict
        assert 'wPNAPL_scale' not in param_dict
    #

    return apl_params


# TODO what to type hint for output? does tianpei have np.ndarrays for both of his?
# and both 1d, or (N, 1) vs (1, N) depending on wAPLKC vs wKCAPL? may want to change
# his code to work w/ floats, or at least squeeze input
# TODO narrow output type (want same one for 'fixed_thr' vs 'wAPLKC'/'wKCAPL'?)
def get_thr_and_APL_weights(param_dict: ParamDict, kws: ParamDict) -> ParamDict:
    """Returns kwargs to fix thresholds and APL weights to match prior call.

    Tuning should be skipped on future `fit_mb_model` or `fit_and_plot_mb_model` calls
    using these keyword arguments.

    Args:
        kws: keyword arguments to prior call (which may be shared with future calls),
            some of which determine which parts of `param_dict` should be used
    """
    fixed_thr = param_dict['fixed_thr']

    variable_n_claws = kws.get('pn2kc_connections') in variable_n_claw_options
    # NOTE: get_APL_weights makes similar checks on wAPLKC
    # TODO refactor to share w/ it
    if not variable_n_claws:
        # TODO just ndarray, or series too (/only?)?
        assert is_scalar(fixed_thr) or isinstance(fixed_thr, np.ndarray)
    else:
        # TODO only check first part if n_seeds=1, and only second otherwise?
        assert is_scalar(fixed_thr) or (
            isinstance(fixed_thr, list) and all(is_scalar(x) for x in fixed_thr)
        )
        #

    params = {'fixed_thr': fixed_thr}

    apl_params = get_APL_weights(param_dict, kws)
    params.update(apl_params)

    return params


def handle_multiglomerular_receptors(df: pd.DataFrame, *, drop: bool = True,
    verbose: bool = False) -> pd.DataFrame:

    # TODO if this is false, could add to each glomerulus? doubt it would matter much at
    # all. this fn should only be handling 'DM3+DM5' (where 'DM3' and 'DM5' are also in
    # hallem data)
    if not drop:
        raise NotImplementedError

    assert df.index.name == glomerulus_col
    # glomeruli should only contain '+' delimiter (e.g. 'DM3+DM5') if the
    # measurement was originally from a receptor that is expressed in each
    # of the glomeruli in the string.
    multiglomerular_receptors = df.index.str.contains('+', regex=False)
    if multiglomerular_receptors.any():
        # 'DM3+DM5' should correspond to Or33b coreceptor Hallem adata
        expected_mg_receptors = {'DM3+DM5'}
        mg_receptors = set(df.index[multiglomerular_receptors])
        # in case some of my weird names for uncertain glomeruli have slipped thru to
        # here by accident (none of the calls to this use my glom names tho...)
        assert mg_receptors == expected_mg_receptors, (
            'had unexpected multiglomerular receptors:\n'
            f'{pformat(mg_receptors - expected_mg_receptors)}'
        )

        # defaulting to verbose=False, b/c it seems all calls to this are not on my
        # input data (just on hallem, or in an unused path where my input is given w/
        # receptor names, which are then converted to glomeruli names)
        if verbose:
            warn(f'dropping multiglomerular receptors: {mg_receptors}')

        df = df.loc[~multiglomerular_receptors, :].copy()

    return df


def _print_response_rates(responses: pd.DataFrame, *, verbose: bool = False) -> None:
    """Prints mean response rates for bool responded/not data of shape (# KCs, # odors)

    If input index has a `KC_TYPE` level, also prints response rate within each KC
    subtype, and how many cells of each type.
    """
    # TODO refactor sparsity float formatting?
    print(f'mean response rate: {responses.mean().mean():.4f}', end='')

    if KC_TYPE not in responses.index.names:
        # TODO still breakdown by odor when we don't have KC type?
        # (should have a plot for that, so idk...)
        print()
        return

    # ab         802
    # g          612
    # a'b'       336
    # unknown     80
    n_kcs_by_type = responses.index.get_level_values(KC_TYPE).value_counts()

    response_rate_by_type = responses.groupby(KC_TYPE).sum().T / n_kcs_by_type

    print('. by KC type:')
    avg_response_rate_by_type = response_rate_by_type.mean(
        ).to_frame(name='avg_response_rate')

    avg_response_rate_by_type['n_kcs_of_type'] = n_kcs_by_type

    # mean response rate: 0.1074. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.074230            336
    # ab                0.106718            802
    # g                 0.131200            612
    # unknown           0.072059             80
    print(avg_response_rate_by_type.to_string())

    # TODO don't print this last one by default, but add flag for extra verbosity?
    # (and print if CLI -v?)
    if verbose:
        print()
        print('mean response rate by odor and KC type:')
        print(response_rate_by_type.to_string())


# TODO factor to hong2p.util?
def delim_sep_part(ser: pd.Series, *, sep: str = '_', i: int = 0) -> pd.Series:
    return ser.str.split(sep).apply(lambda x: x[i])

def first_delim_sep_part(ser, *, sep: str = '_') -> pd.Series:
    return delim_sep_part(ser, sep=sep, i=0)
#


expected_kc_types = {'g', 'ab', "a'b'"}
# TODO delete? still needed anywhere (newest [and prob final] v3 prat outputs shouldn't
# have incomplete/gap for any types)
#kc_type_hue_order = sorted(expected_kc_types) + ['unknown', 'incomplete', 'gap']
kc_type_hue_order = sorted(expected_kc_types) + ['unknown']
del expected_kc_types

def add_kc_type_col(df: pd.DataFrame, type_col: str) -> pd.DataFrame:
    # TODO replace NaN w/ 'unknown' here, instead of adding it after the fact
    # potentially inconsistently
    # TODO + provide example of str processing
    """Returns new `df` with added 'kc_type' column

    New column values should be a subset of `kc_type_hue_order` values.
    """
    # TODO also use for fafb inputs (not currently doing so)? or need diff processing
    # there?
    df = df.copy()

    # TODO provide example of input str format (-> adapt to new prat outputs), where at
    # least this assertion isn't working. from tianpei's one-row-per-claw case:
    # ipdb> df[type_col].unique()
    # array(['KCg-m', nan, 'KCg-s2', 'KCab-m', 'KCab-c', 'KCg-t', "KCa'b'-m",
    #        "KCa'b'-ap1", 'KCg-d', 'KCab-s', "KCa'b'-ap2", 'KCab-p', 'KCg-s3',
    #        'KCg-s4', 'KCg-s1'], dtype=object)
    #
    # from a more recent prat output, when trying to get prat_claws=True working (but
    # not absolute latest PN->KC output from him):
    # ipdb> df['kc_subtype'].unique()
    # array(['KCab-c', 'KCab-m', 'KCab-s', 'KCg-m', "KCa'b'-m", "KCa'b'-ap2",
    #        'KC part due to gap', "KCa'b'-ap1", 'KCg-t', 'KCg-d', 'KCg-s4',
    #        'KC(incomplete?)', 'KCab-p', 'KCg-s2(super)'], dtype=object)

    # TODO TODO at least warn, if doing this...
    # TODO TODO don't do this? this is why unknown type has 0 weight downstream...
    kc_types = df[type_col].dropna()

    # after this, should all be one of {"g", "ab", "a'b'"}
    # (or NaN after assigning back into `df`)
    kc_types = kc_types.str.strip('KC ').replace({
        'part due to gap': 'gap',
        # TODO TODO clarify which subtype these actually belong to (known? all
        # gamma?)
        '(incomplete?)': 'incomplete',
    })

    sep_count = kc_types.str.count('-')
    multiple_sep = sep_count > 1
    assert not multiple_sep.any()

    kc_types = first_delim_sep_part(kc_types, sep='-')

    # TODO also check no 'unknown' already in there?
    # TODO fix + restore in new prat_claws=True connectome_APL_weights call? currently
    # getting {'incomplete', 'y(half)', 'gap'} as difference of these sets there
    #assert len(set(kc_types) - set(kc_type_hue_order)) == 0, \
    #    f'unique kc_types:\n{pformat(set(kc_types))}'

    assert KC_TYPE not in df.columns
    # from hemibrain wPNKC input:
    # ipdb> df[KC_TYPE].str.strip('KC').value_counts(dropna=False)
    # g       4639
    # ab      3759
    # a'b'    1469
    # NaN      238
    df[KC_TYPE] = kc_types

    # important this happens after line above, which will often be what introduces the
    # NaN (by assigning back into df)
    # TODO add separate module-level variable for 'unknown'?
    df[KC_TYPE] = df[KC_TYPE].fillna('unknown')

    return df


def agg_synapses_to_claws(syns: pd.DataFrame, claw_cols: List[str],
    cols_to_avg: List[str], extra_cols_to_keep: List[str], *,
    check_unique_per_claw: Optional[List[str]] = None) -> pd.DataFrame:
    # TODO doc form of non-unique extra_cols_to_keep iterable in output
    """Returns data aggregated per claw, and 'n_synapses' counting input rows per claw.

    Args:
        syns: dataframe containing at least the columns in remaining arguments

        claw_cols: levels of output row index. output will have a number of rows equal
            to the number of unique combinations input has in these columns.

        cols_to_avg: numeric columns which will be averaged over all synapses in the
            claw

        extra_cols_to_keep: if only a single unique value per claw, that column will
            only contain those single values. otherwise, will contain iterable of unique
            values for all elements of column (even if a given row only has one unique
            value)

        check_unique_per_claw: if None, will default to `extra_cols_to_keep`. will
            assert that these columns have only one unique value per unique combination
            of `claw_cols` values (per row in output).

    Notes:
    Will start by asserting no NaN in any `claw_cols`
    """
    assert 'n_synapses' not in syns.columns, 'could not add n_synapses. already present'
    assert syns[claw_cols].notna().all().all()

    if check_unique_per_claw is None:
        check_unique_per_claw = list(extra_cols_to_keep)

    # TODO sort=False? (would probably change some of current outputs...)
    by_claw = syns.groupby(claw_cols)

    agg_dict = {
        **{c: 'mean' for c in cols_to_avg},
        # TODO this includes NaN by default, right? would have to handle if not
        **{c: 'unique' for c in extra_cols_to_keep}
    }
    if len(agg_dict) > 0:
        claws = by_claw.agg(agg_dict)
        assert (
            cols_to_avg + extra_cols_to_keep == list(claws.columns)
        )

        unique_per_claw = claws[extra_cols_to_keep].apply(
            lambda x: (x.str.len() == 1).all()
        )
        cols_unique_per_claw = list(unique_per_claw[unique_per_claw].index)

        assert set(cols_unique_per_claw) >= set(check_unique_per_claw), (
            'expected the following columns to be unique per claw '
            '(pass explicit check_unique_per_claw excluding them, if otherwise):\n'
            f'{set(check_unique_per_claw) - set(cols_unique_per_claw)=}'
        )

        claws[cols_unique_per_claw] = claws[cols_unique_per_claw].applymap(
            lambda x: x[0]
        )

    n_synapses_per_claw = by_claw.size()
    assert n_synapses_per_claw.sum() == len(syns)
    assert n_synapses_per_claw.index.names == claw_cols

    if len(agg_dict) > 0:
        assert n_synapses_per_claw.index.equals(claws.index)
        claws['n_synapses'] = n_synapses_per_claw
        return claws
    else:
        return n_synapses_per_claw.to_frame(name='n_synapses')


# TODO delete (after checking i don't want to revert any of the changes tianpei made in
# his version below) (or delete his version)
# TODO rename to plot_n_synapse_hist?
def _plot_connectome_raw_weight_hist(weights: DataFrameOrSeries, *,
    discrete: bool = True, **kwargs) -> Tuple[Figure, Axes]:

    # weight_ser is just for check below. weights will always be a df (required for
    # sns.histplot now, it seems, which also preserves any input cols [as may be needed
    # for hue=<col> in kwargs])
    if isinstance(weights, pd.DataFrame):
        assert 'x' in kwargs, 'must pass x= if weights are a DataFrame'
        x = kwargs.pop('x')
        weight_ser = weights[x]
    else:
        weight_ser = weights.copy()

        x = weights.name
        # should be a meaningful str name (e.g. 'n_synapses')
        assert x is not None
        weights = weights.reset_index()

    if discrete:
        # discrete=True wouldn't make sense otherwise, right?
        # TODO multiply back to int first (easy to compute here?) (currently doing that
        # in connectome_APL_weights code that was having issues here)? assert input has
        # int (synapse count) dtype? (or disable discrete if can't?)
        assert (weight_ser.astype(int) == weight_ser.astype(float)).all()
        del weight_ser

    fig, ax = plt.subplots()
    sns.histplot(weights, x=x, discrete=discrete, ax=ax, **kwargs)
    return fig, ax


# TODO also use these elsewhere (e.g. in connectome_wPNKC)
presynapse_cols = ['bodyId_pre', 'x_pre', 'y_pre', 'z_pre']
postsynapse_cols = ['bodyId_post', 'x_post', 'y_post', 'z_post']
def check_polyadic_synapses(df: pd.DataFrame) -> None:
    """Raises AssertionError if `df` has any multiple-presynapses:1-postsynapse

    Also raises AssertionError if:
        - `df` has no cases of 1-presynapse:multiple-postsynapses
        - any NaN in `presynapse_cols` or `postsynapse_cols`

    Args:
        df: should contain `presynapse_cols` to identify pre-synapses, and
            `postsynapse_cols` to identify post-synapses.

    Pratyush said (paraphrasing): "polyadic should always be one T-bar going to many
    PSDs, never multiple T-bars going to one PSD"

    He did find some multiple-pre:1-post violations in 2025-12-05 v5 outputs, but the
    last outputs he sent on 2025-12-09 should have fixed this issue, by filtering of >=
    0.35 on confidence of some of these duplicates. Filtering was only applied to PN->KC
    synapses with the unexpected duplicates, not to all PN->KC synapses.
    """
    assert not df[presynapse_cols + postsynapse_cols].isna().any().any()

    # implies there can not be multiple pre -> 1 post, since all postsynapses only
    # listed once
    assert len(df) == len(df[postsynapse_cols].drop_duplicates()), (
        f'had duplicate {postsynapse_cols=} combinations (some multiple-pre-synapses '
        '[aka T-bars] going to one post-synapse [aka PSD]. not the established type '
        'of polyadic synapse.)'
    )

    # TODO add flag to allow no 1-pre:multiple-post (would want to warn instead, at
    # least)? only if this ever fails
    #
    # in combination with above, implies there are at least some
    # 1-presynapse:multiple-postsynapse cases
    assert len(df[presynapse_cols].drop_duplicates()) < len(df), (
        'had no presynapses that connect to multiple postsynapses (the expected '
        'direction of polyadic synapses)'
    )


variable_n_claw_options: Set[str] = {'uniform', 'caron', 'hemidraw'}

# TODO add 'hemibrain-matt' here, and replace _use_matt_wPNKC w/ that?
# NOTE: FAFB = FlyWire
connectome_options: Set[str] = {'hemibrain', 'fafb-left', 'fafb-right'}

pn2kc_connections_options: Set[str] = set(variable_n_claw_options)
pn2kc_connections_options.update(connectome_options)

def get_connectome(pn2kc_connections: Optional[str]) -> str:
    # TODO also (/only) err here, if pn2kc_connections not in pn2kc_connections_options?
    # (as opposed to erring in fit_mb_model intro)

    # NOTE: this means that if pn2kc_connections == 'hemidraw', it will use marginal
    # probabilities from 'hemibrain' connectome. no current support for using either
    # fafb data source for that.
    return pn2kc_connections if pn2kc_connections in connectome_options else 'hemibrain'


from_prat = repo_root / 'data/from_pratyush'

# hemibrain PN/KC/APL weights (PN->KC, APL<->KC, APL<->PN), split by claw (using Prat's
# dendritic-tree based claw determination) but not currently by bouton. no MB-C1 yet.
# includes distances to "root", also from Prat's dendritic-tree based analysis, but
# haven't yet found a good way to use those.
#
# TODO delete v3
# "v3"
prat_hemibrain_seg_v3_dir = from_prat / '2025-09-24'
#
#
# "v5", which should also be final version of these hemibrain outputs (he never handed
# over a "v4", if I recall correctly, though he may have had one himself)
prat_hemibrain_seg_dir = from_prat / '2025-12-05'

claw_coord_cols = [f'claw_{d}' for d in ('x', 'y', 'z')]

def center_each_claw_coord(wPNKC: pd.DataFrame) -> pd.DataFrame:
    """Returns dataframe like input, with each of `claw_coord_cols` separately centered.
    """
    for_index = wPNKC.index.to_frame(index=False)
    for c in claw_coord_cols:
        vals = for_index[c]
        val_min = vals.min()
        val_range = vals.max() - val_min
        # centering range of values on 0
        for_index[c] = vals - (val_min + val_range/2)

    wPNKC2 = wPNKC.copy()
    wPNKC2.index = pd.MultiIndex.from_frame(for_index)
    return wPNKC2


# TODO remove roi columns after this?
# TODO + make similar fn that filters based in pre/post synapse confidence (or not?
# seems we may not want to for *at least* either the KC>APL or APL>KC synapses. i forget
# which. one seemed it would filter more out thatn we'd want, if using prat's typical
# cutoffs) (and then also removes those input cols)
def filter_synapses_to_roi(df: pd.DataFrame, roi: str, *,
    assert_some_dropped: bool = False) -> pd.DataFrame:
    """Filters synapses to only rows with 'roi_pre' and 'roi_post' matching `roi`.

    Args:
        assert_some_dropped: if True, will raise AssertionError if all input rows match
            `roi`
    """
    # NOTE: NaN ROI values (at least in v5 PN->KC data) should be real, and can also be
    # considered != to a particular ROI (particularly != 'CA(R)', the main ROI of
    # interest)
    len_before = len(df)
    df = df[(df.roi_pre == roi) & (df.roi_post == roi)]

    assert len(df) > 0, 'dropped everything!'

    n_dropped = len_before - len(df)
    if assert_some_dropped:
        assert n_dropped > 0

    if n_dropped > 0:
        warn(f'dropped {n_dropped} ({n_dropped/len_before:.3f} of total) synapses with '
            f'either pre/post synaptic ROI != "{roi}"'
        )

    return df


# TODO refactor to share w/ other places that redef these (now that module level)
# TODO cache any of this (fn to instatiate on first call)? worth it?
glomerulus2receptors = orns.task_glomerulus2receptors()
task_gloms = set(glomerulus2receptors.keys())

# TODO better name
glomerulus_renames = {'VC3l': 'VC3', 'VC3m': 'VC3'}
assert all(x in task_gloms for x in glomerulus_renames.values())
# wouldn't necessarily need to be true, if we were shuffling names around, but we
# currently aren't...
assert not any(x in task_gloms for x in glomerulus_renames.keys())

def add_glomerulus_col_from_hemibrain_type(df: pd.DataFrame, pn_type_col: str,
    # TODO doc what kc_id_col is even used for. why not optional?
    # TODO double check doc about kc_id_col (that those are only two cases where
    # KCs would be dropped) is true
    kc_id_col: str, *, check_no_multi_underscores: bool = False,
    # TODO remove '_' prefix on _drop_glom_with_plus
    _drop_glom_with_plus: bool = True, drop_kcs_with_no_input: bool = True
    ) -> Tuple[pd.DataFrame, Set[int]]:
    """Returns `df` with added glomerulus_col column, and KCs dropped for no input.

    Returned `df` will have certain glomeruli (and thus certain KCs typically) dropped.

    Glomerulus parsed from `df[pn_type_col]`.

    kc_id_col should only ever be used to summarize how many KCs were dropped (if
    any) b/c of check_no_multi_underscores=True / _drop_glom_with_plus=True
    """
    assert glomerulus_col not in df.columns

    df = df.copy()

    assert not df[kc_id_col].isna().any()
    def _get_n_kcs(df) -> int:
        return df[kc_id_col].nunique()

    n_kcs_initial = _get_n_kcs(df)
    def _kc_drop_message(n_kcs_dropped):
        return (f'(also means that {n_kcs_dropped}/{n_kcs_initial} KCs only '
            'connected to these "glomeruli" dropped)'
        )

    assert not df[pn_type_col].isna().any()

    # TODO were these just in v3? or am i using wrong pn_type_col for v5? not getting
    # error these were supposedly causing in add_glom..., when i disabled dropping these
    # (which should have been b/c they had no underscore)
    #
    # only 4 rows in df here (/ 173222) should have these values (2
    # each, with all having suffix of '_R' in instance_pre column. these should
    # be the only rows that would cause check_no_multi_underscores=True to fail.
    #
    # what glomeruli even do these correspond to? (they don't. go to "wedge"
    # region. just drop) and it's only 4 rows in all of df, so def feel safe
    # dropping
    pn_types_to_drop = ('WEDPN12', 'WEDPN4')

    in_pn_types_to_drop = df[pn_type_col].isin(pn_types_to_drop)
    n_in_pn_types_to_drop = in_pn_types_to_drop.sum()
    if n_in_pn_types_to_drop > 0:
        warn(f'dropping {n_in_pn_types_to_drop} synapses with PN types in '
            f'{pn_types_to_drop}'
        )
        df = df[~ in_pn_types_to_drop].copy()

    # askprat: are there specific PN types (RHS after '_') that i should
    # categorically be dropping? (prob not. what are prefixes anyway? all lineage
    # info, or anything closer to what i actually care about?).
    # Prat: no. keep all.
    #
    # a.type should all be roughly of form: <glomerulus-str>_<PN-group>, where
    # PN-group are distributed as follows (w/ connectome='hemibrain' data):
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
    try:
        assert (df[pn_type_col].str.count('_') == 1).all()

    except AssertionError:
        # can do this for hemibrain input, but not fafb (b/c this type column in
        # fafb comes from aligning to hemibrain types, which are better annotated,
        # and i believe this multiple-underscore cases are all from a small number
        # of instances where this alignment process is ambiguous
        if check_no_multi_underscores:
            # TODO also say something separate if it's just a 0-underscore string
            # causing it, vs one with multiple? (or move raising of error in
            # some-0-undescore case before this, since that would fail [with better
            # message] despite check_no_multi_underscores=True/False?)
            raise

        no_us = df.loc[df[pn_type_col].str.count('_') == 0, pn_type_col].unique()

        assert (df[pn_type_col].str.count('_') >= 1).all(), \
            f'{pn_type_col} unique values with no underscore:\n{no_us}'

        # askprat: are any of below MG PNs, or something i want to include? what
        # are these weird glomeruli names? trailing +?
        # Prat: their alignment produced duplicates. not MG PNs. he thinks 'M' is to
        # indicate multiglomerular (or at least that it goes to multiple areas,
        # perhaps including things other than glomeruli), but not easy to get info
        # on which they are, without more work. he was saying if we really cared, we
        # could use the previously defined glomerular boundaries to count and figure
        # out which.
        #
        # connectome='hemibrain' has no rows w/ multiple '_'
        #
        # connectome='fafb-left'
        # M_adPNm4,M_adPNm5                120
        # VP1m+VP2_lvPN1,VP1m+VP2_lvPN2     23
        # M_lPNm12,M_lPNm13                 20
        # M_ilPNm90,M_ilPN8t91               4
        #
        # connectome='fafb-right'
        # M_adPNm4,M_adPNm5                86
        # M_lPNm12,M_lPNm13                12
        # VP1m+VP2_lvPN1,VP1m+VP2_lvPN2    12
        # M_ilPNm90,M_ilPN8t91              5
        multi_rows = df[pn_type_col].str.count('_') >= 2
        n_multi_rows = multi_rows.sum()
        assert n_multi_rows > 0

        df_no_multi = df[~multi_rows].copy()
        n_kcs_dropped = _get_n_kcs(df) - df_no_multi[kc_id_col].nunique()

        # NOTE: not currently hitting this for prat_claws=True v5 data (assuming I'm
        # using correct PN type column)
        # TODO also mention # claws dropped, when appropriate?
        warn(f'dropping {n_multi_rows} synapses w/ multiple underscores in PN type:\n' +
            df[pn_type_col][multi_rows].value_counts().to_string() + '\n' +
            _kc_drop_message(n_kcs_dropped) + '\n'
        )
        df = df_no_multi

    # TODO replace this kwarg w/ one talking about dropping glomeruli w/ no
    # task-glomeruli inputs instead? could it be equiv, or no? or otherwise rename
    # it to be more clear about what it's currently dropping (multiglomerular PNs?
    # anything else?)?
    if _drop_glom_with_plus:
        has_plus = df[pn_type_col].str.contains('+', regex=False)
        if has_plus.any():
            # askprat: what is meaning when it finishes w/ '+', w/o that
            # being a separator between two glomerulus names?
            # Prat: (re: VP3+ does go somewhere else, but either "not dense enough
            # (in other place(s) it goes to)" or not going to somewhere in hemibrain
            # volume, but that could still be in AL...)
            #
            # what is 'Z'? (Prat: Z=SEZ. it also goes there.)
            #
            # connectome='hemibrain'
            # VP1d+VP4_l2PN1    250
            # VP3+VP1l_ivPN      76
            # VP1m+VP5_ilPN      64
            # VP5+Z_adPN         17
            # VP3+_vPN           17
            # VP1m+VP2_lvPN2     10
            # VP1m+VP2_lvPN1      1
            #
            # connectome='fafb-left'
            # VP1d+VP4_l2PN1                   285
            # VP3+VP1l_ivPN                    117
            # VP1m+VP5_ilPN                     92
            # VP5+Z_adPN                        47
            # VP1m+VP2_lvPN1,VP1m+VP2_lvPN2     23
            # VP3+_vPN                          20
            # VP1m+_lvPN                         6
            # VP1l+VP3_ilPN                      2
            # VP2+_adPN                          1
            #
            # connectome='fafb-right'
            # VP1d+VP4_l2PN1                   303
            # VP3+VP1l_ivPN                    112
            # VP1m+VP5_ilPN                    106
            # VP5+Z_adPN                        42
            # VP3+_vPN                          20
            # VP1m+VP2_lvPN1,VP1m+VP2_lvPN2     12
            # VP2+_adPN                          5
            # VP1m+_lvPN                         4
            # VP1l+VP3_ilPN                      3
            df_no_plus = df[~has_plus].copy()
            n_kcs_dropped = _get_n_kcs(df) - df_no_plus[kc_id_col].nunique()
            # TODO also mention # claws dropped, when appropriate?
            warn(f'dropping {has_plus.sum()}/{len(df)} synapses w/ "+" in PN type, with'
                ' the following types:\n' +
                df[pn_type_col][has_plus].value_counts().to_string() + '\n' +
                _kc_drop_message(n_kcs_dropped) + '\n'
            )
            df = df_no_plus
        else:
            warn(f"not dropping glomeruli with '+' in their name, because "
                f'{_drop_glom_with_plus=}. this preserves old hemibrain model '
                'behavior exactly, but should probably be removed moving forward.'
            )

    # TODO also print count of unique PN bodyids within each glom_strs value?
    # (here might not be the place anymore, if i even still care about this...)

    glom_strs = first_delim_sep_part(df[pn_type_col], sep='_')
    assert glom_strs.notna().all()

    def _format_type_counts(values: pd.Series) -> str:
        # expects Series of pn_type_col (str) values as input
        counts = values.value_counts()
        counts.index.name = pn_type_col
        counts.name = 'n_synapses'
        return counts.to_frame().reset_index().to_string(index=False)

    to_rename = glom_strs.isin(glomerulus_renames)
    if to_rename.any():
        old_names = glom_strs[to_rename]
        warn(f'renaming glomeruli as {glomerulus_renames}:\n' +
            _format_type_counts(old_names) + '\n'
        )

    # TODO actually check it doesn't matter whether i do this before vs after pivot?
    # (or at least, that i intend for current behavior) (not super worried about it)
    glom_strs = glom_strs.replace(glomerulus_renames)
    assert glom_strs.notna().all()

    connectome_gloms = set(glom_strs)

    non_task_gloms = connectome_gloms - task_gloms
    if len(non_task_gloms) > 0:
        # connectome='hemibrain'|'fafb-right'
        # ['M']
        #
        # connectome='fafb-left'
        # ['M', 'MZ']
        #
        # connectome='hemibrain' (v5 prat_claws=True)
        # ['M']
        warn('dropping glomeruli in connectome but NOT task: '
            f'{sorted(non_task_gloms)}\n' +
            _format_type_counts(glom_strs[glom_strs.isin(non_task_gloms)]) + '\n'
        )
        glom_strs = glom_strs[glom_strs.isin(task_gloms)]

    non_connectome_gloms = task_gloms - connectome_gloms
    if len(non_connectome_gloms) > 0:
        # connectome='hemibrain' (including v5) |'fafb-left'|'fafb-right'
        # ['VM6l', 'VM6m', 'VM6v', 'VP1l', 'VP3', 'VP5']
        # TODO TODO why VC3 printed in prior prat_claws=True outputs (before moving
        # code here, but not here?) now getting just: ['VM6l', 'VM6m', 'VM6v', 'VP1l',
        # 'VP3', 'VP5'] (which were also all there before refactor)
        # TODO TODO and why did VC3 get dropped before, when that step seemed to
        # happen after glomerulus renames? was set defined before renames?
        warn(f'glomeruli in Task but NOT connectome: {sorted(non_connectome_gloms)}')

    kcs_before = set(df[kc_id_col].unique())

    # for prat_claws=True v5 (these sum to value that is currently reported as just 'M'
    # above). don't think we need extra warning with all these subtypes.
    #
    # ipdb> df[~ df.index.isin(glom_strs.index)][pn_type_col].value_counts()
    # M_adPNm4     777
    # M_lPNm11D    266
    # M_l2PNm16    120
    # M_adPNm5      39
    # M_smPNm1       1
    # M_lPNm11C      1
    # M_lvPNm46      1
    df = df.loc[glom_strs.index]
    df[glomerulus_col] = glom_strs

    kcs_after = set(df[kc_id_col].unique())
    kcs_without_input = kcs_before - kcs_after
    n_kcs_without_input = len(kcs_without_input)

    if n_kcs_without_input > 0:
        # TODO also update message to "claws" without input, when appropriate currently
        # getting:
        # "Warning: 875/12298 KCs without input, after dropping non-Task glomeruli"
        # in prat_claws=True case
        msg = (f'{n_kcs_without_input}/{len(kcs_before)} KCs without input, after '
            f'dropping non-Task glomeruli:\n{sorted(kcs_without_input)}'
        )

        if drop_kcs_with_no_input:
            msg = f'dropping (b/c {drop_kcs_with_no_input=}) {msg}'
        else:
            msg = f'NOT dropping (b/c {drop_kcs_with_no_input=}) {msg}'

        warn(msg)

    # easiest way I could think of to check for RangeIndex
    # if this fails, need to change code to preserve index information at least.
    # could just not reset_index() then.
    assert df.index.name is None and df.index.dtype == int
    df = df.reset_index(drop=True)

    assert df[glomerulus_col].notna().all()
    assert (df[glomerulus_col].str.len() > 0).all()

    # TODO delete
    # cases causing failure in prat_claws=True v5:
    # ipdb> df[[glomerulus_col, pn_type_col]].drop_duplicates().groupby(glomerulus_col
    #   ).filter(lambda x: len(x) > 1)
    #       glomerulus   type_pre
    # 35013        VC3  VC3m_lvPN
    # 37187        VC3  VC3l_adPN
    # 42318        VM4   VM4_adPN
    # 53574        VM4   VM4_lvPN
    #
    # nevermind, this wasn't actually true in prat_claws=True v5 case. 56 combos (&
    # types), but only 54 unique gloms
    #
    # probably don't particularly care if this fails (could remove), but does seem to be
    # passing for prat_claws=True v5 at least. also, type_pre and instance_pre are
    # interchangeable in that context (same # of unique values, w/ latter only having
    # some unnecessary extra suffix info)
    #n_type_glom_combos = len(df[[pn_type_col, glomerulus_col]].drop_duplicates())
    #n_types = df[pn_type_col].nunique()
    #n_gloms = df[glomerulus_col].nunique()
    #assert n_type_glom_combos == n_types == n_gloms, \
    #    f'{n_type_glom_combos=} {n_types=} {n_gloms=}'
    #

    # TODO or just always track sets of KCs before/after as needed (w/ metadata too,
    # like kc_type)? (-> remove 2nd return val)
    return df, kcs_without_input


def assert_one_glom_per_pn(df: pd.DataFrame, *, pn_id_col: str = PN_ID) -> None:
    """Raises AssertionError if >1 glomerulus for any one PN ID.

    Assumes `df` has `glomerulus_col` and `pn_id_col` in columns, and no NaN in either.
    """
    assert df[[pn_id_col, glomerulus_col]].notna().all().all(), \
        f'{df[[pn_id_col, glomerulus_col]].isna().sum()=}'

    assert (
        len(df[[pn_id_col, glomerulus_col]].drop_duplicates()) ==
        df[pn_id_col].nunique()
    ), f'duplicate combinations of {pn_id_col=} and {glomerulus_col=}!'


# TODO compare to what ann had been using (does she have her own fully formed connectome
# based PN->KC matrix, or is it all random draws with certain connectome inspired
# probabilities?)
# TODO (delete?) add parameter for thresholding PN-KC pairs above the existing cutoffs
# of >=4 (which was enforced by neuprint query that produced the hemibrain data we have,
# or we do manually here for the fafb datasets) (-> try to get total # of claws closer
# to reported ~5.[2-6?] mean claws in Davi Bock paper, through varying both this and
# weight_divisor. currently weight_divisor=20 brings us way above their reported mean #
# of claws, given threshold of >4 we are stuck w/ in hemibrain case. maybe Bock # of
# claws is partially just b/c that brain is abnormal [i.e. many more KCs, etc]?)
#
# TODO TODO get prat's code he used to produce outputs i'm loading (or at least latest
# ones)
# TODO make prat_claws default to true later? (here and in other places)
def connectome_wPNKC(connectome: str = 'hemibrain', *, prat_claws: bool = False,
    # TODO TODO TODO separate kwarg to enable prat PN->APL connections, w/o also
    # splitting PNs into boutons? olfsysm currently support that? (and/or for
    # connectome_APL_weights)
    prat_boutons: bool = False, dist_weight: Optional[str] = None,
    weight_divisor: Optional[float] = None, plot_dir: Optional[Path] = None,
    _use_matt_wPNKC: bool = False, drop_kcs_with_no_input: bool = True,
    _drop_glom_with_plus: bool = True,
    # TODO reconsider handling of synapse_*_path kwargs, and DBSCAN related param kws
    synapse_con_path: Optional[Path] = None, synapse_loc_path: Optional[Path] = None,
    cluster_eps: float = 1.9, cluster_min_samples: int = 3, Btn_separate: bool = False,
    # NOTE: Tianpei's commit that added some bouton stuff (a786cec2) initially had an
    # unused Btn_divide_per_glom (bool) param
    # TODO still want something like that, or never really a case where there is a
    # choice to be made?
    # TODO is this actually Optional? what happens if it's None
    Btn_num_per_glom: Optional[int] = 10) -> pd.DataFrame:
    # TODO doc possible contents of row/column index in returned wPNKC dataframe
    """
    Args:
        connectome: which connectome of hemibrain/fafb-left/fafb-right to use.
            should be one of `connectome_options`.

        prat_boutons: requires `prat_claws=True`, then also stores separate bouton ID
            for each claw (as Pratyush defined anatomically, for his v5 hemibrain
            outputs)

        dist_weight: only relevant if `prat_claws=True`, then selects method for using
            `1 / dist_to_root` (mean per-claw) to scale `wPNKC` values (from 0.1x to
            10x)

        weight_divisor: if None, one model claw is added for each PN-KC pair exceeding
            minimum total number of unitary synapses (currently de facto >=4 for
            hemibrain, b/c of query that pulled hemibrain data, and also hardcoded to
            that in this function for FAFB datasets).

            If float, each PN-KC pair gets `ceil(total_synapses / weight_divisor)`
            claws.

            Note that glomeruli have different numbers of cognate PNs, and that only in
            the `weight_divisor=<float>` case can there be multiple claws assigned to
            one particular PN (for a given KC).

        plot_dir: if passed, saves plots histograms of: 1) PN-KC connectome weights, and
            2) #-claws-per-model-KC (#2 depends on `weight_divisor`)

    Returns dataframe of shape (#-KCs [in selected connectome], #-glomeruli).
    """
    one_row_per_claw = False
    if (prat_claws or
        (synapse_loc_path is not None and synapse_con_path is not None)):
        one_row_per_claw = True

    assert connectome in connectome_options
    if _use_matt_wPNKC:
        assert not one_row_per_claw
        assert connectome == 'hemibrain'
        assert weight_divisor is None, \
            'weight_divisor must be None when _use_matt_wPNKC=True'
        # TODO other assertions?

    if one_row_per_claw:
        assert weight_divisor is None

        if not prat_claws:
            # need both
            assert synapse_loc_path is not None
            assert synapse_con_path is not None

            assert dist_weight is None

    if prat_claws:
        assert connectome == 'hemibrain'
        assert synapse_con_path is None and synapse_loc_path is None

    if prat_boutons:
        assert prat_claws

    def _underscore_part(ser, i=0):
        return ser.str.split('_').apply(lambda x: x[i])

    def _first_underscore_part(ser):
        return _underscore_part(ser, i=0)

    def add_compartment_index(wPNKC: pd.DataFrame, shape: int) -> pd.DataFrame:
        # TODO delete shape!=0 path? (currently unused) or will it be in some of his
        # other attempts at defining APL compartments? what are other values shape=
        # might take? (doc + validate)
        """
        Compute compartment IDs and attach them as an INDEX LEVEL named 'compartment'.
        Returns a new DataFrame with the same rows, same glomerulus columns, and
        an extra index level 'compartment'.
        """
        # to ensure we don't accidentally changed input dataframe
        wPNKC = wPNKC.copy()

        # Coordinates must be present as index levels
        idx_names = list(wPNKC.index.names)
        need = claw_coord_cols
        missing = [n for n in need if n not in idx_names]
        assert not missing, f"Index missing levels required for compartments: {missing}"

        coords = wPNKC.index.to_frame(index=False)[need]

        if shape == 0:
            # shell over sphere
            center = coords.mean().to_numpy()
            # TODO if verbose, print center
            distances = np.linalg.norm(coords.to_numpy() - center, axis=1)
            shell_r = distances.max()
            # TODO TODO allow overriding sphere_r w/ kwarg? or (also) define to equalize
            # across two? or allow passing in absolute radius?
            sphere_r = 0.5 * shell_r
            compartment_ids = np.where(distances <= sphere_r, 0, 1).astype(np.int32)
            # TODO delete (/ put behind verbose)
            #print(f'shell over sphere {center=}')
        else:
            # TODO refactor to loop over claw_coord_cols
            # 3×3×3 grid
            x_edges = np.linspace(coords['claw_x'].min(), coords['claw_x'].max(), 4)
            y_edges = np.linspace(coords['claw_y'].min(), coords['claw_y'].max(), 4)
            z_edges = np.linspace(coords['claw_z'].min(), coords['claw_z'].max(), 4)
            for edges in (x_edges, y_edges, z_edges):
                edges[0]  -= 1e-6
                edges[-1] += 1e-6
            ix = np.digitize(coords['claw_x'], x_edges) - 1
            iy = np.digitize(coords['claw_y'], y_edges) - 1
            iz = np.digitize(coords['claw_z'], z_edges) - 1
            compartment_ids = (ix * 9 + iy * 3 + iz).astype(np.int32)

            if (compartment_ids > 26).any():
                print("Some compartment_ids > 26 detected!")
                print(wPNKC.index[(compartment_ids > 26)])
            elif (compartment_ids < 0).any():
                print("Some compartment_ids < 0 detected!")
                print(wPNKC.index[(compartment_ids < 0)])
            else:
                print("All compartment_ids normal")

        # Safety: length match
        assert len(compartment_ids) == len(wPNKC), "compartment id length mismatch"

        # Remove any existing compartment column/level to avoid duplicates
        if 'compartment' in wPNKC.columns:
            wPNKC = wPNKC.drop(columns=['compartment'])

        # TODO delete. should never trigger
        assert 'compartment' not in wPNKC.index.names
        #

        # Attach as an index level (appended at the end)
        tmp = pd.Series(compartment_ids, index=wPNKC.index, name='compartment')
        wPNKC = wPNKC.set_index(tmp, append=True)
        assert wPNKC.index.names[-1] == 'compartment'

        # Reorder so 'compartment' sits right after 'claw_z' if those exist
        names = list(wPNKC.index.names)
        if set(claw_coord_cols).issubset(names):
            names_no_comp = [n for n in names if n != 'compartment']
            insert_pos = names_no_comp.index('claw_z') + 1
            new_order = (
                names_no_comp[:insert_pos] + ['compartment'] +
                names_no_comp[insert_pos:]
            )
            wPNKC = wPNKC.reorder_levels(new_order)

        return wPNKC


    def expand_wPNKC_to_boutons(wPNKC: pd.DataFrame, boutons_per_glom: int = 10,
        check_reconstruction: bool = True) -> pd.DataFrame:
        """
        Column expansion with two header rows:
        level-0 = glomerulus (string), level-1 = bouton id (1..B).
        Treats ONLY numeric columns in `wPNKC` as glomeruli; object/list cols are
        ignored. Each bouton sub-column = original / B.
        """
        if boutons_per_glom <= 0:
            raise ValueError('boutons_per_glom must be a positive integer')

        B = boutons_per_glom
        # choose glomerulus columns: numeric dtype only (skip metadata like
        # 'pre_cell_ids')
        # TODO TODO which columns is this actually pulling out? still working as
        # expected?
        gloms = [c for c in wPNKC.columns if pd.api.types.is_numeric_dtype(wPNKC[c])]
        if not gloms:
            # numeric b/c all 0/1 values, unlike some other columns which were at one
            # point here. should currently probably be no non glomerulus columns
            # though...
            raise ValueError('No numeric glomerulus columns found in wPNKC.')

        W = wPNKC.loc[:, gloms].astype(float).to_numpy()
        R, G = W.shape

        # of shape (R, G*B)
        expanded = np.repeat(W / B, repeats=B, axis=1)
        cols = pd.MultiIndex.from_product([gloms, range(1, B+1)],
            names=[glomerulus_col, BOUTON_ID]
        )
        out = pd.DataFrame(expanded, index=wPNKC.index, columns=cols)

        # Safety: sum across bouton level recreates original
        if check_reconstruction:
            # TODO TODO this actually unique? i assume not? make so?
            # (will trigger an assertion in reindex, if not)_
            glom_index = pd.Index(gloms, name=glomerulus_col)
            recon = reindex(out.groupby(level=glomerulus_col, axis='columns').sum(),
                glom_index, axis='columns'
            )
            if not np.allclose(recon.to_numpy(), W, atol=1e-9):
                bad = [g for i, g in enumerate(gloms)
                    if not np.allclose(recon.iloc[:, i].to_numpy(), W[:, i], atol=1e-9)
                ]
                raise AssertionError(
                    f'Reconstruction check failed for glomeruli: {bad[:5]}'
                )

        return out

    n_prat_claws = None
    df_noinput_kcs = None

    if connectome == 'hemibrain':
        if _use_matt_wPNKC:
            matt_data_dir = repo_root / 'data/from_matt/hemibrain'

            # TODO which was that other CSV (that maybe derived these?) that was full
            # PN->KC connectome matrix?
            #
            # NOTE: gkc-halfmat[-wide].csv have 22 columns for glomeruli (/receptors).
            # This should be the number excluding 2a and 33b.
            gkc_wide = pd.read_csv(matt_data_dir / 'halfmat/gkc-halfmat-wide.csv')

            # TODO see if above include more than hallem glomeruli (and find
            # scripts that generated these -> figure out how to regen w/ more than
            # hallem glomeruli)
            # TODO process gkc_wide to have consistent glomerulus/receptor labels
            # where possible (consistent w/ what i hope to also return in random
            # connectivity cases, etc) (presumbly if it's already a subset, should be
            # possible for all of that subset?)

            # All other columns are glomerulus names.
            assert gkc_wide.columns[0] == 'bodyid'

            wPNKC = gkc_wide.set_index('bodyid', verify_integrity=True)
            wPNKC.columns.name = glomerulus_col
            assert wPNKC.columns.isin(task_gloms).all()

            # TODO where are values >1 coming from in here:
            # ipdb> mdf.w.value_counts()
            # 1    9218
            # 2     359
            # 3      15
            # 4       1
            mdf = pd.read_csv(matt_data_dir / 'glom-kc-cxns.csv')

            mdf.glom = mdf.glom.replace(glomerulus_renames)

            # NOTE: if we do this, mdf_wide.max() is only >1 for VC3 (and it's 2 there,
            # from merging VC3l and VC3m)
            #mdf.loc[mdf.w > 1, 'w'] = 1

            # TODO are all >1 weights below coming from 'w' values that are already >1
            # before this sum? set all to 1, recompute, and see? (seems so?)
            # TODO replace groupby->pivot w/ pivot_table (aggfunc='count'/'sum')?
            # seemed possible w/ pratyush input (but 'weight' input there was max 1...)
            # TODO factor out similar pivoting (w/ pivot_table) below -> share w/ here?
            mcounts = mdf.groupby(['glom', 'bodyid']).sum('w').reset_index()
            mdf_wide = mcounts.pivot(columns='glom', index='bodyid', values='w').fillna(
                0).astype(int)
            # TODO uncomment
            #del mcounts

            # TODO try to remove need for orns.orns + handle_multiglomerular_receptors
            # in here?
            # TODO refactor to share w/ code calling connectome_wPNKC? or get from a
            # module level const in drosolf (maybe add one)?
            hallem_orn_deltas = orns.orns(add_sfr=False, drop_sfr=False,
                columns=glomerulus_col).T

            hallem_glomeruli = handle_multiglomerular_receptors(hallem_orn_deltas,
                drop=True
            ).index
            del hallem_orn_deltas

            mdf_wide = mdf_wide[[x for x in hallem_glomeruli if x != 'DA4m']].copy()
            del hallem_glomeruli

            mdf_wide = mdf_wide[mdf_wide.sum(axis='columns') > 0].copy()

            # TODO move creation of mdf_wide + checking against wPNKC to model_test.py /
            # similar
            assert wPNKC.columns.equals(mdf_wide.columns)
            assert set(mdf_wide.index) == set(wPNKC.index)
            mdf_wide = mdf_wide.loc[wPNKC.index].copy()
            assert mdf_wide.equals(wPNKC)
            del mdf_wide

            # TODO still define one of these as df (+ any other variables i need
            # to define? some column names?), so i can make the same histograms below?
            # or never want them in the _use_matt_wPNKC case?
            # (won't have pn_id_col, b/c not in any of matt's outputs i'm loading [at
            # least], so would have to work w/o that)

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
            # NOTE: pn-kc-cxns.csv above should also be what matt uses to generate
            # distribution of # claws per KC (in matt-hemibrain/docs/mb-claws.html)

            kc_id_col = 'bodyid'
            # NOTE: no pn_id_col available in any current outputs loaded above (or any
            # of his I've ever loaded)

        elif prat_claws:
            # TODO delete (+ loading of outputs that are only in this old dir, all of
            # which [that i care about] should be replaced by a combination of newer
            # outputs)
            prat_hemibrain_PNKC_claw_dir = from_prat / '2025-08-20'

            data_path = (prat_hemibrain_seg_dir /
                # TODO check equiv to prior 1-2 versions he sent me around the same
                # time? should be. i just misunderstood data i think, and output
                # shouldn't have changed
                'PNbouton-to-KCclaw_Connectivity_2025-12-09-19-47-36.parquet'
            )

            # TODO delete (or combine w/ switch to re-run w/ old v3 outputs, which also
            # uses diff def for prat_hemibrain_seg_dir?)
            # TODO would any of my old outputs have changed, if i had been
            # filtering based roi_[pre|post] to 'CA(R)'? was i really not effectively
            # doing this (maybe some of the other filtering, e.g. on anatomical_claw,
            # also excluded stuff out of 'CA(R)'? NO)?
            old = pd.read_parquet(prat_hemibrain_seg_v3_dir /
                'PN-to-KC_with_Synapses_v3.parquet'
            )
            #
            syns = pd.read_parquet(data_path)

            if 'weight' in syns.columns:
                # in v3, but not in v5
                assert (syns['weight'] == 1).all()

            # different versions may not have all of these.
            # none of these are in v5 (at least not in the PNbouton-to-KCclaw* file from
            # v5)
            #
            # 'type' here is "connector" type, and not meaningful.
            to_drop = ['node_id_post', 'connector_id_post', 'connector_type', 'weight',
                'connector_id', 'node_id', 'type'
            ]
            syns = syns.drop(columns=[c for c in to_drop if c in syns.columns])

            # v5 notes:
            # ipdb> syns.dtypes
            # bodyId_pre                     int64
            # bodyId_post                    int64
            # roi_pre                       object
            # roi_post                      object
            # x_pre                          int32
            # y_pre                          int32
            # z_pre                          int32
            # x_post                         int32
            # y_post                         int32
            # z_post                         int32
            # confidence_pre               float32
            # confidence_post              float32
            # instance_pre                  object
            # type_pre                      object
            # instance_post                 object
            # type_post                     object
            # anatomical_claw_corrected    float64
            # anatomical_bouton            float64
            #
            # ipdb> syns.isna().sum()
            # bodyId_pre                      0
            # bodyId_post                     0
            # roi_pre                      2663
            # roi_post                     2690
            # x_pre                           0
            # y_pre                           0
            # z_pre                           0
            # x_post                          0
            # y_post                          0
            # z_post                          0
            # confidence_pre                  0
            # confidence_post                 0
            # instance_pre                    0
            # type_pre                        0
            # instance_post                   0
            # type_post                    3447
            # anatomical_claw_corrected       6
            # anatomical_bouton              26
            #
            # ipdb> syns.roi_pre.value_counts()
            # CA(R)     187913
            # SLP(R)      6921
            # SCL(R)       436
            # SMP(R)         2
            # Name: roi_pre, dtype: int64
            # ipdb> syns.roi_post.value_counts()
            # CA(R)     187844
            # SLP(R)      6925
            # SCL(R)       473
            # SMP(R)         3
            #
            # ipdb> syns.instance_pre.value_counts()
            # DP1m_adPN_R    8498
            # DM1_lPN_R      6665
            # DC1_adPN_R     6449
            # VL2p_adPN_R    5699
            # DM4_adPN_R     5595
            #                ...
            # M_lPNm12_R        2
            # M_lPNm11A_R       1
            # M_l2PNm15_R       1
            # M_lvPNm45_R       1
            # M_ilPNm90_R       1
            # Name: instance_pre, Length: 83, dtype: int64
            # ipdb> syns.instance_post.value_counts()
            # KCg-m_R               99483
            # KCab-m_R              28822
            # KCab-s_R              20871
            # KCab-c_R              12647
            # KCa'b'-ap2_R           9595
            # KCa'b'-m_R             8690
            # KCa'b'-ap1_R           7963
            # KCg-s2(super)_R        4637
            # KC part due to gap     3045
            # KCg-t_R                1000
            # KCg-d_R                 456
            # KCy(half)               242
            # KC(incomplete?)         160
            # KCg-s3_R                152
            # KCab-p_R                121
            # KCg-s4_R                 34
            # KCg-s1(super)_R          17
            #

            # NOTE: this filtering was NOT applied to v3 (prior to updating this code
            # to load v5), but it probably should have been
            syns = filter_synapses_to_roi(syns, 'CA(R)', assert_some_dropped=True)
            # (end filtering only added for v5 processing)

            # work here? yes! didn't work before restricting to 'CA(R)', even on last
            # output from 2025-12-09...
            check_polyadic_synapses(syns)

            # on v5, after filtering by ROI above:
            # ipdb> syns.isna().sum()
            # bodyId_pre                      0
            # ...
            # instance_post                   0
            # type_post                    3358
            # anatomical_claw_corrected       5
            # anatomical_bouton               1
            # dtype: int64
            # ipdb> syns.type_post.value_counts(dropna=False)
            # KCg-m         96653
            # KCab-m        28635
            # KCab-s        20632
            # KCab-c        12556
            # KCa'b'-ap2     9437
            # KCa'b'-m       8653
            # KCa'b'-ap1     5808
            # None           3358
            # KCg-t           909
            # KCg-d           415
            # KCg-s2          384
            # KCg-s3          130
            # KCab-p           78
            # KCg-s4           17
            # KCg-s1            2

            # TODO update/delete (was for v3 or earlier)
            # ipdb> syns.bodyId_pre.unique().shape
            # (158,)
            # ipdb> syns.bodyId_post.unique().shape
            # (1786,)

            assert (len(syns[['type_pre', 'instance_pre']].drop_duplicates()) ==
                syns.type_pre.nunique() == syns.instance_pre.nunique()
            )

            # v3 and older should have had -1 claw IDs dropped, but v5 (final) outputs
            # can end up with -1 corresponding to actual meaningful claws (and maybe for
            # most/all?), so pratyush recommended using them all as any other ID.
            # other IDs (e.g. APL units / boutons) in v5 outputs should still drop -1
            # values in their IDs, as reason some were meaningful was only b/c of a
            # correction step applied only to the KC claw IDs.
            drop_negative_claw_id = True
            if 'anatomical_claw_corrected' in syns.columns:
                drop_negative_claw_id = False
                assert 'anatomical_claw' not in syns.columns
                syns = syns.rename(
                    columns={'anatomical_claw_corrected': 'anatomical_claw'}
                )

            # TODO factor out this ID handling/validation
            assert syns.anatomical_claw.dropna().min() == -1

            # i believe these should all be from stuff clipped by his cuttiing volume?
            # still true (assuming it was for pre-v5...)?
            null_claw_ids = syns.anatomical_claw.isna()
            n_null_claw_ids = null_claw_ids.sum()
            # TODO remove this assertion? (make warning conditional on # > 0 if so)
            assert n_null_claw_ids > 0
            warn(f'dropping {n_null_claw_ids} PN->KC synapses with null claw ID')
            syns = syns[~ null_claw_ids]

            if drop_negative_claw_id:
                negative_claw_ids = syns.anatomical_claw == -1
                n_negative_claw_ids = negative_claw_ids.sum()
                assert n_negative_claw_ids > 0
                warn(f'dropping {n_negative_claw_ids} PN->KC synapses with -1 '
                    '(unassigned) claw ID'
                )
                syns = syns[~ negative_claw_ids]

            # should be numbered within each KC, from 0. -1 for synapses not assigned to
            # any claw (but sorta assigned to a "real" claw if enough from one PN ID, in
            # v5)
            claw_ids = syns['anatomical_claw']
            assert not claw_ids.isna().any()
            assert pd_allclose(claw_ids, claw_ids.astype(int))
            syns['anatomical_claw'] = claw_ids.astype(int)
            del claw_ids

            # TODO assert instance_post are all 1:1 with these (or at least after
            # stripping some maybe irrelevant suffix info?)
            kc_type_col = 'type_post'

            # TODO (done? delete?) use these for types later, instead of just the
            # coarser ones? provide both?
            # ipdb> syns.instance_post.unique()
            # array(['KCab-m', 'KCg-m', 'KCab-c', 'KCab-s', "KCa'b'-m", "KCa'b'-ap2",
            #        'KCg-d', 'KCg-t', "KCa'b'-ap1", 'KCy(half)', 'KCg-s4',
            #        'KCg-s2(super)', 'KCab-p', 'KC(incomplete?)', 'KCg-s1(super)',
            #        'KCg-s3'], dtype=object)
            #
            # ipdb> syns[kc_type_col].unique()
            # array(['KCab', 'KCg', "KCa'b'", 'KCy(half)', 'KC(incomplete?)'],
            #       dtype=object)

            # TODO delete? (replace w/ some assertions, and check still pass on v5?)
            #
            # (this was before dropping anatomical_claw == -1)
            # ipdb> syns[kc_type_col].value_counts()
            # KCg                99193
            # KCab               61270
            # KCa'b'             24969
            # KCy(half)            242
            # KC(incomplete?)      141
            #
            # TODO TODO restore? delete? how many of these am i currently actually left
            # with at end? any? already filtered out somewhere else?
            # TODO try with and without this?
            # TODO any justification for keeping one or the other of these?
            # TODO warn about these
            # TODO no longer drop KC(incomplete?) (why again? still want?) (+ check they
            # all have claws now on 2nd version [or greater] of his PN->KC outputs)
            ##kc_types_to_drop = ('KCy(half)', 'KC(incomplete?)')
            #kc_types_to_drop = ('KCy(half)',)
            #syns = syns[~ syns[kc_type_col].isin(kc_types_to_drop)].copy()
            # (will actually be less now that we are also dropping NaN claw IDs)
            # len(syns)=173222
            #

            # NOTE: redefined to PN_ID by end of this prat_claws=True conditional
            pn_id_col = 'bodyId_pre'
            # NOTE: redefined to KC_ID by end of this prat_claws=True conditional
            kc_id_col = 'bodyId_post'

            # TODO factor out these ID cols (+ PN ID one), as needed (esp after
            # factoring all this new code to an appropriate home)
            syns, kcs_without_input = add_glomerulus_col_from_hemibrain_type(syns,
                'type_pre', kc_id_col, check_no_multi_underscores=True,
                _drop_glom_with_plus=_drop_glom_with_plus,
                drop_kcs_with_no_input=drop_kcs_with_no_input
            )

            assert syns[[pn_id_col, glomerulus_col]].notna().all().all()
            # each PN ID only has a single unique glomerulus, across all rows of synapse
            # dataframe
            assert (syns[[pn_id_col, glomerulus_col]].drop_duplicates().groupby(
                pn_id_col).size() == 1
            ).all(), 'some PN IDs w/ inconsistent glomerulus str across rows'

            # TODO make plot for this? # PNs per glomerulus:
            # ipdb> syns[[pn_id_col,glomerulus_col]].drop_duplicates().groupby(
            #   glomerulus_col).size().value_counts()
            # 1    25
            # 2    15
            # 3     8
            # 5     3
            # 7     1
            # 4     1
            # 6     1

            # TODO (care? delete?) assert glomerulus 1:1 w/ type (or after all dropping
            # below, if it's only then that it becomes true. it is currently true after
            # all the dropping, where syn is len 164986). (add_glomerulus_col* already
            # do something like that?)
            # (will actually be less now that we are also dropping NaN claw IDs)
            # len after: 165909 (w/ _drop_glom_with_plus=True)

            # TODO re: infs check one neuron not overrepresented, and check all
            # stragglers (or most)
            finite_dist_to_root = np.isfinite(syns.dist_to_root)
            warn(f'dropping {(~ finite_dist_to_root).sum()}/{len(syns)} PN->KC synapses'
                ' with inf dist_to_root'
            )
            syns = syns[finite_dist_to_root].copy()
            assert syns.dist_to_root.notna().all()

            # newest one doesn't have. only older, slightly bad versions
            if 'dist_influence' in syns.columns:
                assert pd_allclose(1 / syns.dist_to_root, syns.dist_influence)

            # from Prat:
            # dist_influence is sum of 1 / dist_to_root (nm), for each (PN, KC) pair
            # total_weight is sum of dist_influence, per KC
            # scaled_weight is dist_influence / total_weight

            kc_type_isna = syns[kc_type_col].isna()
            n_null_kc_type = kc_type_isna.sum()
            # true on at least v3 and v5 (w/ similar #, and presumably interpretation,
            # in both cases)
            assert n_null_kc_type > 0
            warn(f'dropping {n_null_kc_type}/{len(syns)} PN->KC synapses, for KCs with '
                'null type'
            )
            # TODO now that i'm dropping these outside, rather than previously where it
            # would be dropped in [at least part of] add_kc_type_col, assert no null
            # values in KC type col in add_kc_type_col?
            # TODO move these assertions before all filtering on syns?
            expected_instance_vals_for_null_kc_types = {
                'KC part due to gap', 'KCy(half)', 'KC(incomplete?)'
            }
            assert (
                set(syns[kc_type_isna].instance_post.unique()) ==
                expected_instance_vals_for_null_kc_types
            )
            assert syns[
                syns.instance_post.isin(expected_instance_vals_for_null_kc_types)
            ][kc_type_col].isna().all()
            #
            syns = syns[~ kc_type_isna]

            if bouton_col in syns.columns:
                # TODO refactor to share w/ dropping [both null and -1] above?
                # (+ dropping i'll prob want to add for APL units...)

                null_bouton_ids = syns[bouton_col].isna()
                n_null_bouton_ids = null_bouton_ids.sum()
                assert n_null_bouton_ids == 0, f'{n_null_bouton_ids=}'
                # TODO delete this code? (assuming assertion above doesn't fail on any
                # tests)
                #if n_null_bouton_ids > 0:
                #    warn(f'dropping {n_null_bouton_ids}/{len(syns)} PN->KC synapses '
                #        'with null bouton IDs'
                #    )
                #    syns = syns[~ null_bouton_ids]
                #

                negative_bouton_ids = syns[bouton_col] == -1
                n_negative_bouton_ids = negative_bouton_ids.sum()
                # could convert this assertion to conditional if fails. non-essential.
                assert n_negative_bouton_ids > 0
                # TODO TODO also print how many KCs / claws are dropped by this
                # (including breakdown of how many of those claws had ID -1). refactor
                # to share this reporting code w/ connectome_APL_weights?
                #breakpoint()
                warn(f'dropping {n_negative_bouton_ids}/{len(syns)} PN->KC synapses '
                    'with -1 (unassigned) bouton ID'
                )
                syns = syns[~ negative_bouton_ids]

                # TODO also assert all float values are close to int values, before
                # replacing? (share w/ some code elsewhere that does that already?)
                #
                # was type float before
                syns = syns.astype({bouton_col: int})

            # can only do this now that we are also dropping null kc_type_col here
            # (rather than as part of add_kc_type_col below), since both v3 and v5 data
            # have synapses (rows) w/ null KC type here.
            assert not syns.isna().any().any()

            # TODO any reason to experiment w/ claws that receive input from multiple
            # diff PNs? and how many are there of those really? meaningful? any patterns
            # in which combinations?
            #
            # Initially, I thought I did NOT want to include PN ID in claw cols, but I
            # think I do, mainly since Prat's approach has some claws w/ significant
            # input from mulitple PNs, and I want to [for now] split them out and
            # pretend they are two claws).
            # TODO refactor to share 'anatomical_claw' col def?
            claw_cols = [kc_id_col, 'anatomical_claw', pn_id_col]

            if prat_boutons:
                assert bouton_col in syns.columns, (f'must have {bouton_col} '
                    'column in order to use prat_boutons=True'
                )
                assert syns[bouton_col].notna().all()
                # no -1's to worry about here
                assert syns[bouton_col].min() == 0, f'{syns[bouton_col].min()=}'

            # each of these is expected to only have one unique value within a
            # combination of claw_cols argument to agg_synapses_to_claws. will err if
            # that's not true. not currently supported to get an iterable of unique
            # values for any column (within a unique combination of claw_cols values)
            # TODO remove instance_post? (maybe just after checking it's redundant w/
            # kc_type_col earlier, around where we first process kc_type_col. should
            # just have some irrelevant extra suffix info on top of kc_type_col, and may
            # even already be 1:1)
            extra_cols_to_keep = [glomerulus_col, kc_type_col, 'instance_post']
            check_unique_per_claw = None
            if prat_boutons:
                # everything besides bouton_col
                check_unique_per_claw = list(extra_cols_to_keep)
                extra_cols_to_keep.append(bouton_col)

            # TODO subset to only those present in data (only would be needed to load
            # old versions, which didn't have coords) (or assert all in data?)
            cols_to_avg = ['dist_to_root', 'x_post', 'y_post', 'z_post']

            # NOTE: old (v3) data seems it did NOT have roi_[pre|post] != 'CA(R)' rows
            # effectively dropped by here, so old outputs probably not 100% correct?
            # TODO check to what extent it would have changed outputs below?

            check_polyadic_synapses(syns)

            # TODO why do we not have a call like this in tianpei's case? could i fix
            # some of his issues by adding one on the right input?
            #
            # this fn also starts by asserting no NaN in any claw_cols
            claws = agg_synapses_to_claws(syns, claw_cols, cols_to_avg,
                extra_cols_to_keep, check_unique_per_claw=check_unique_per_claw
            )
            # TODO also preserve n_synapses as index level, for analysis later? (when
            # constructing wPNKC, alongside other metadata i'd like to add, like PN
            # [+bouton(s)] ID(s))

            # TODO rename dist_to_root to include '_nm' suffix?
            renames = {
                kc_id_col: KC_ID,

                # NOTE: numbering is arbitrary, may be missing some integers in range
                # (potentially b/c "claws" w/ only APL/etc synapses get that ID), and
                # may [at least in v5] include -1 as well.
                'anatomical_claw': CLAW_ID,

                pn_id_col: PN_ID,

                bouton_col: BOUTON_ID,

                # more fine grained than the 3 values for KC_TYPE we wil have here
                'instance_post': 'kc_subtype',
            }
            claws = claws.reset_index().rename(columns=renames)
            claw_cols = [renames.get(c, c) for c in claw_cols]

            assert len(claws[claw_cols].drop_duplicates()) == len(claws)

            # TODO de-dedupe w/ code that does this below (id_cols_to_check)?
            id_cols = list(claw_cols)
            if prat_boutons:
                assert BOUTON_ID not in claw_cols
                assert BOUTON_ID in claws.columns
                id_cols.append(BOUTON_ID)

                # len(ids_only)=12094 (when bouton_col was in intial claw_cols)
                # (and after exploding BOUTON_ID col, when it was in
                # extra_cols_to_keep, instead of claw_cols)
                #
                # if trying to drop_duplicates before explode, get:
                # TypeError: unhashable type: 'numpy.ndarray'
                # (and order shouldn't matter otherwise)
                ids_only = claws[id_cols].explode(BOUTON_ID).drop_duplicates()

                # using ids_only, this is equiv to:
                # ids_only.groupby(claw_cols).apply(lambda x:
                #   len(x[[PN_ID, 'bouton_id']].drop_duplicates())
                # )
                # ...but much after. also equiv if only using 'bouton_id' in commented
                # call above, since PN_ID already in claw_cols.
                #
                # ipdb> n_boutons_per_claw.value_counts()
                # 1    11386
                # 2      342
                # 3        8
                # TODO this is currently redefined below. delete one or the other?
                n_boutons_per_claw = ids_only.groupby(claw_cols).size()

                # TODO delete (/put behind checks=True path)
                n_boutons_per_claw2 = claws.set_index(claw_cols, verify_integrity=True
                    )[BOUTON_ID].str.len()
                assert n_boutons_per_claw.equals(n_boutons_per_claw2)
                del n_boutons_per_claw2
                #

                # TODO plot distribution of these (esp # claws per bouton?)
                # (already have somewhere?)
                # TODO and also worth trying:
                # .apply(lambda x: len(x[[KC_ID, CLAW_ID]].drop_duplicates()))
                # ...instead of just claws that include PN in def? prob not here...
                n_claws_per_bouton = ids_only.groupby(bouton_cols).size()

                # TODO TODO look into cases where this is >1? and do check again
                # below, after filtering/dropping PNs? maybe *only* do this check
                # there?

                # TODO TODO check no n-boutons:1-claw (but check we do have
                # 1-bouton:n-claws)
                # TODO need to do any of these checks after dropping the existing
                # (small # of) cases where there are already multiple PNs per claw?
                # (dropped below)? (hopefully can still check here if including PN in
                # claw_cols?)

                # TODO TODO even want BOUTON_ID to be included in duplicate check?
                # (don't think so) what's nature of duplicates? what do duplicates this
                # adds mean?
                # TODO TODO or is this a matter of not having filtered bouton_ids
                # sufficiently yet?
                # TODO are duplicates (w/ boutons) all claw_id -1? just drop those
                # before this check?
                # TODO bouton weights?
                # TODO delete. would fail, and don't think i want this (comments above
                # were referring to when BOUTON_ID was part of check below [after
                # conditional])
                #assert len(ids_only.drop_duplicates()) == len(ids_only)

            assert claws[id_cols].notna().all().all()

            n_kcs_before = claws[KC_ID].nunique()
            # TODO include this parameter (synapse_per_1pn_claw_thresh) in downstream
            # stuff? (or thread thru at least?)
            # TODO diff cutoff if we are using prat_boutons=True?
            #
            # is it actually accurate that it is just 1pn per claw at this point?
            # per [KC_ID, CLAW_ID], no, but per-row (which is same as per
            # [KC_ID, CLAW_ID, PN_ID], yes.
            # with trying [2, 10, 5] values for synapse_per_1pn_claw_thresh:
            # Warning: dropping 1153 (0.097) (single PN) claws with <2 synapses
            # Warning: dropping 3893 (0.326) (single PN) claws with <10 synapses
            # Warning: dropping 1956 (0.164) (single PN) claws with <5 synapses
            for synapse_per_1pn_claw_thresh in [5]:
                claws_below_synapse_thresh = (
                    claws.n_synapses < synapse_per_1pn_claw_thresh
                )
                n_claws_below_synapse_thresh = claws_below_synapse_thresh.sum()
                if n_claws_below_synapse_thresh > 0:
                    n_syns_dropped = claws.n_synapses[claws_below_synapse_thresh].sum()
                    total_syns = claws.n_synapses.sum()

                    frac_claws_dropped = n_claws_below_synapse_thresh / len(claws)
                    frac_syns_dropped = n_syns_dropped / total_syns

                    # TODO refactor to share w/ some of reindex_to_wPNKC reporting?
                    warn(f'dropping {n_claws_below_synapse_thresh}/{len(claws)} '
                        f'({frac_claws_dropped:.3f}) (single PN) claws with '
                        f'<{synapse_per_1pn_claw_thresh} synapses. '
                        f'this drops {n_syns_dropped}/{total_syns} '
                        f'({frac_syns_dropped:.3f}) synapses.'
                    )

            # would like to ignore the (few) cases in claws_with_multiple_PNs, that I'll
            # already be dropping below, but need to do this before subsetting claws to
            # only those >= synapse thresh (subtract those ones later, before reporting
            # the # ambiguous)
            n_multiple_valid_minus_one = 0
            multiple_valid_minus_one_ids = set()
            n_valid_and_invalid_minus_one = 0
            valid_and_invalid_minus_one_ids = set()
            for (kc, claw), claw_df in claws[claws[CLAW_ID] == -1].groupby(
                [KC_ID, CLAW_ID]):

                valid = claw_df.n_synapses >= synapse_per_1pn_claw_thresh
                n_valid = valid.sum()
                if n_valid == 0:
                    continue

                elif n_valid > 1:
                    n_multiple_valid_minus_one += 1
                    multiple_valid_minus_one_ids.add((kc, claw))

                if (~ valid).any():
                    # (and there are some cases where n_valid == 1 here, potentially
                    # also some of the claws_with_multiple_PNs cases below)
                    n_valid_and_invalid_minus_one += 1
                    valid_and_invalid_minus_one_ids.add((kc, claw))

            # NOTE: there is no spatial clustering to be expected in the -1 claw IDs,
            # other than that those synapses did not fall into any of the other spatial
            # clusters. could still make sense to try dropping -1 claws in v5 in all
            # data sources. Prat also said his outputs have (to some extent) had
            # [potentially distant] -1 claw synapses (that share same PN ID w/ some
            # non- -1 claw ID) grouped into that claw ID. This seems to have assumed to
            # some extent that there aren't multiple claws from any one PN to one
            # particular KC, which doesn't quite seem true (see below), but probably
            # doesn't matter much.
            #
            # There are cases where there are multiple claws from one PN to one KC:
            # ipdb> nnclaws = claws[claws.claw_id >= 0]
            # ipdb> nnclaws[['kc_id','claw_id',PN_ID]].drop_duplicates().shape
            # (9805, 3)
            # ipdb> nnclaws[['kc_id',PN_ID]].drop_duplicates().shape
            # (9093, 2)

            # TODO maybe count all these synapses as part of non-claw (/"spine")
            # synapses though? return separately? store in separate per-KC metadata
            # somewhere? could put as a multiindex level (would be unique per KC ID),
            # to not have to change what is returned?
            # (prob don't care enough. matters more [and probably only] in the APL<>KC
            # case)
            claws = claws[~ claws_below_synapse_thresh]

            n_kcs_after = claws[KC_ID].nunique()
            if n_kcs_after < n_kcs_before:
                warn(f'lost {n_kcs_before - n_kcs_after} KCs by filtering claws with '
                    'subthreshold # of synapses'
                )

            # redundant w/ check below that there are some PNs with multiple claws.
            # and would be great if this were not true actually, but it is...
            assert (
                len(claws[[KC_ID, CLAW_ID]].drop_duplicates()) <
                len(claws)
            )

            claws_with_multiple_PNs = claws.groupby([KC_ID, CLAW_ID]
                ).filter(lambda x: len(x) > 1)

            assert (multiple_valid_minus_one_ids ==
                set(tuple(x) for x in claws_with_multiple_PNs[['kc_id','claw_id']
                    ].itertuples(index=False)
                )
            )
            # each of these two (kc, claw) cases happen to also have invalid (i.e.
            # insufficient synapses from one PN) -1 claw IDs too
            assert (multiple_valid_minus_one_ids & valid_and_invalid_minus_one_ids ==
                multiple_valid_minus_one_ids
            )
            # (after dropping claws_with_multiple_PNs)
            n_invalid_valid_ambiguous_after_dropping = len(
                valid_and_invalid_minus_one_ids - multiple_valid_minus_one_ids
            )
            # TODO maybe flag (in new index level) the -1's that have the
            # valid_and_invalid amgibuity? (unless it's not an issue when merging w/
            # other data...) (prob not worth)
            if n_invalid_valid_ambiguous_after_dropping > 0:
                # v5: 26
                warn(f'{n_invalid_valid_ambiguous_after_dropping} KCs where claw ID of '
                    '-1 is ambiguous as to whether it is a "valid" claw or not '
                    '(validity based on # synapses from a single PN)'
                )

            n_claws_with_multiple_PNs = claws_with_multiple_PNs.groupby([KC_ID, CLAW_ID]
                ).ngroups
            # both v3 and v5 prat outputs should have claws with multiple PNs here.
            # and would be great if this were not true actually, but it is...
            assert 0 < n_claws_with_multiple_PNs < len(claws)

            # NOTE: none of these (KC, PN) combos have any other claw IDs associated
            # with them.
            #      kc_id  claw_id       pn_id  ...  n_synapses  dist_influence
            #  754538763       -1   850375847  ...           5        0.000080
            #  754538763       -1  1914823337  ...           5        0.000101
            # 5812983451       -1  5901222910  ...           5        0.000234
            # 5812983451       -1   606090268  ...           5        0.000099
            #
            # can not increase synapse_per_1pn_claw_thresh from 5 to 6 to fix this,
            # because with 5 we get:
            # ```
            # dropping 1865 (0.159 of 11736) (single PN) claws with <5 synapses
            # lost 15 KCs by filtering claws with subthreshold # of synapses
            # dropping 2/9871 claws with multiple PNs
            # ```
            # ...and with 6 we get (losing many more claws):
            # ```
            # dropping 2145 (0.183 of 11736) (single PN) claws with <6 synapses
            # lost 20 KCs by filtering claws with subthreshold # of synapses
            # dropping 0/9591 claws with multiple PNs
            # ```
            warn(f'dropping {n_claws_with_multiple_PNs}/{len(claws)} claws with '
                f'multiple PNs ({claws_with_multiple_PNs.n_synapses.sum()} synapses)'
            )
            # TODO anything to do other than drop these (would prob be easiest to drop
            # all, rather than trying to sort or something, could be part to align with
            # other things)
            claws = claws.drop(index=claws_with_multiple_PNs.index)

            n_prat_claws = len(claws)

            # no longer need to include PN_ID as part of claw definition
            assert len(claws) == len(claws[[KC_ID, CLAW_ID]].drop_duplicates())

            # TODO TODO keep BOUTON_ID if needed? (in prat_boutons=True case?)
            # (prob want as column index level if anything? is that what i started
            # changing tianpei's code to do for pn_id?)
            claws = claws.set_index([KC_ID, CLAW_ID], verify_integrity=True)

            # TODO move before add_glomerulus... to be consistent w/ handling in other
            # cases (so we can add back these IDs with correct type metadata, etc)
            # (mostly just relevant for hemibrain_paper_repro test tho)
            #
            # adds KC_TYPE column, which we will then use instead of kc_type_col
            claws = add_kc_type_col(claws, kc_type_col)
            # TODO assert 318 'unknown' at output? (there are 318 in a manual check of
            # KC_TYPE output) (# None entries in value_counts(dropna=False). no
            # unknown/incomplete/gap in type_post at this point, w/ latest (v3) outputs
            claws = claws.drop(columns=kc_type_col)

            # NOTE: this is probably where much of ordering of claws was going from in
            # natmix_data/analysis.py, since including glomerulus in sort here
            #
            # sorting here probably won't be enough to be able to use new claw IDs
            # rather than these ones, to align to APL<->KC weights, since I expect we
            # will often have different sets of claw IDs available there.
            claws = claws.sort_values([KC_ID, CLAW_ID, glomerulus_col])

            for d in ['x', 'y', 'z']:
                new_col = f'claw_{d}'
                assert new_col not in claws.columns
                # using average of KC side of synapse ("post") as claw coordinate
                claws[new_col] = claws[f'{d}_post'] * PIXEL_TO_UM

            claws['dist_influence'] = 1 / claws.dist_to_root

            # TODO update comment w/ new values (/delete)
            # ipdb> qs = [0, 0.0005, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99,
            #   0.999, 0.9995, 1]
            # ipdb> claws.dist_influence.quantile(q=qs)
            # 0.0000    0.000051
            # 0.0005    0.000057
            # 0.0010    0.000059
            # 0.0100    0.000071
            # 0.1000    0.000096
            # 0.2500    0.000125
            # 0.5000    0.000191
            # 0.7500    0.000299
            # 0.9000    0.000469
            # 0.9900    0.002861
            # 0.9990    0.008632
            # 0.9995    0.010624
            # 1.0000    0.037253
            #
            # TODO are these not just getting swamped by large ones frequently tho?
            # since they will often vary over orders of magnitude (and presumably within
            # many individual KCs, not just across them).
            # TODO matter? want to normalize in some way other than straight
            # average of this?
            # ipdb> claws.groupby(KC_ID).dist_influence.sum(
            #   ).quantile(q=qs)
            # 0.0000    0.000061
            # 0.0005    0.000064
            # 0.0010    0.000065
            # 0.0100    0.000264
            # 0.1000    0.000556
            # 0.2500    0.000860
            # 0.5000    0.001633
            # 0.7500    0.002802
            # 0.9000    0.004127
            # 0.9900    0.009087
            # 0.9990    0.017528
            # 0.9995    0.022459
            # 1.0000    0.039409
            #
            # TODO plot subtype distributions of these diff quantities? per claw
            # and per KC at least


            # NOTE: had previously tried argsort instead of percentileofscore, but
            # couldn't get results to match up. deleted some old committed comment about
            # that. not 100% current method is reasonable.
            claws['dist_influence_percentile'] = percentileofscore(
                claws.dist_influence, claws.dist_influence) / 100

            assert (
                claws.dist_influence.max() ==
                claws.dist_influence.loc[claws.dist_influence_percentile.idxmax()]
            )
            assert (
                claws.dist_influence.min() ==
                claws.dist_influence.loc[claws.dist_influence_percentile.idxmin()]
            )
            assert (claws.loc[claws.dist_influence == claws.dist_influence.min(),
                'dist_influence_percentile'].squeeze() ==
                claws.dist_influence_percentile.min()
            )
            assert (claws.loc[claws.dist_influence == claws.dist_influence.max(),
                'dist_influence_percentile'].squeeze() ==
                claws.dist_influence_percentile.max()
            )
            assert np.isclose(claws.dist_influence_percentile.max(), 1)
            # NOTE: min is not quite 0, but pretty close (would need to tweak
            # isclose atol/rtol to get assertion to work, or maybe diff kind=?)
            # ipdb> claws.dist_influence_percentile.min()
            # 9.889240506329115e-05
            # ipdb> np.isclose(claws.dist_influence_percentile.min(), 0)
            # False

            # NOTE: had also tried kind='cubic'/'quadratic', and didn't prefer their
            # outputs, at least for 'percentile' case
            kind = 'linear'
            output_range = [0.1, 1, 10.0]

            dist_weight_col = 'dist_weight'
            if dist_weight == 'percentile':
                # TODO try from raw dist_influence rather than from
                # percentiles? (maybe after some non-linear scaling?)
                # TODO scale in a way where mean change is 1 (within KC?)?
                # (possible to do via changing way we call interp1d? add point for
                # mean?) (or just postprocess to achieve this?)
                #
                # TODO delete. below does seem to be an improvement.
                #interpolator = interp1d([0, 1], output_range, kind=kind)
                #claws['dist_weight'] = interpolator(claws.dist_influence_percentile)

                # TODO share input range def w/ what i do below (move above)
                interpolator = interp1d([0, 0.5, 1], output_range, kind=kind)
                claws['dist_weight'] = interpolator(claws.dist_influence_percentile)

            elif dist_weight == 'raw':
                interpolator = interp1d(
                    claws.dist_influence.quantile(q=[0, 0.5, 1]), output_range,
                    kind=kind
                )
                claws['dist_weight'] = interpolator(claws.dist_influence)
            else:
                assert dist_weight is None
                dist_weight_col = 'dist_influence'

            index_cols = [KC_ID, CLAW_ID]
            extra_cols = [KC_TYPE] + claw_coord_cols

            # TODO exclude [pn_id, bouton_id] for now, then save outputs for
            # repro checks, then check we can repro w/ including in extra_cols
            # (prob fine, since output of pivot same as long as those levels are
            # dropped)
            if prat_boutons:
                # TODO want to add to pivot columns instead (would change wPNKC shape
                # though... that might require more changes overall)
                # TODO check this doesn't break anything below (+ fix if so)
                extra_cols.extend([PN_ID, BOUTON_ID])

                # making them hashable, so set_index call below won't fail
                # (would also need hashable if I wanted to exclude from set_index call,
                # but include in columns= kwarg to subsequent pivot call)
                claws[BOUTON_ID] = claws[BOUTON_ID].apply(lambda x: tuple(sorted(x)))
            #

            # adding [PN_ID, BOUTON_ID] (at least, at end of index levels) doesn't
            # change output of this pivot call, after dropping those two levels:
            #
            # ipdb> d1 = claws.reset_index().set_index(index_cols + extra_cols +
            #   [PN_ID, BOUTON_ID]).pivot(columns=glomerulus_col,
            #   values=dist_weight_col
            # )
            # ipdb> d2 = claws.reset_index().set_index(index_cols +
            #   extra_cols).pivot(columns=glomerulus_col, values=dist_weight_col)
            #
            # ipdb> d1.droplevel([PN_ID, BOUTON_ID]).equals(d2)
            # True
            dist_weights = claws.reset_index().set_index(index_cols + extra_cols
                ).pivot(columns=glomerulus_col, values=dist_weight_col)

            del extra_cols

            assert np.isclose(
                dist_weights.sum().sum(), claws[dist_weight_col].sum()
            )

            # (from when dist_weight_col='dist_influence')
            # ipdb> dist_influence.T.sum()
            # kc_id       claw_id  kc_type
            # 300968622   0        ab         0.000077
            #             1        ab         0.000083
            #             2        ab         0.000087
            # 301309622   0        ab         0.000142
            #             1        ab         0.000125
            #                                   ...
            # 5901225361  1        g          0.000113
            #             2        g          0.000121
            #             3        g          0.000148
            #             4        g          0.000110
            #             5        g          0.000117
            # Length: 10112, dtype: float64
            # ipdb> dist_influence.T.sum().quantile(q=[0,0.01,0.1, 0.5, 0.9,0.99,1])
            # 0.00    0.000048
            # 0.01    0.000063
            # 0.10    0.000077
            # 0.50    0.000106
            # 0.90    0.000200
            # 0.99    0.001444
            # 1.00    0.004267
            #
            # ipdb> dist_influence.replace(0, np.nan).stack().quantile(q=[0, 0.01,
            #   0.1, 0.5, 0.9, 0.99, 1])
            # 0.00    0.000048
            # 0.01    0.000063
            # 0.10    0.000077
            # 0.50    0.000106
            # 0.90    0.000200
            # 0.99    0.001444
            # 1.00    0.004267
            # dtype: float64
            # ipdb> dist_influence.stack().shape
            # (10112,)

            if dist_weight is None:
                wPNKC = dist_weights.fillna(0).astype(bool).astype(int)
            else:
                # TODO need to scale so mean is similar to above? see comments above.
                wPNKC = dist_weights.fillna(0)

            # TODO some version with weight components from both synapse count as
            # well as dist_influence?

            # TODO try diff radius than the 0.5 * max this currently hardcodes within?
            # TODO summarize how many claws are in/out of this?
            wPNKC = add_compartment_index(wPNKC, shape=0)

            # TODO also apply in prat_claws=False case (tianpei already have a
            # separate version of these cols, centered or similar?)
            wPNKC2 = center_each_claw_coord(wPNKC)

            # add_compartment_index should drop this regardless. just repeating here for
            # clarity about what we are testing.
            wPNKC2 = wPNKC2.droplevel('compartment')

            wPNKC2 = add_compartment_index(wPNKC2, shape=0)

            # TODO delete? (+ rename all wPNKC2 -> wPNKC above)
            # checking the comparments don't change if we pre-center coords
            assert wPNKC.index.get_level_values('compartment').equals(
                wPNKC2.index.get_level_values('compartment')
            )
            # want claw coords to have their ranges centered
            wPNKC = wPNKC2
            #

            # TODO check against what pratyush had (was his not a bit lower?
            # differences in dropping account for it?)
            #
            # ok this seems somewhat reasonable
            # ipdb> claws.groupby(KC_ID).size().mean()
            # 6.899596076168494

            # TODO delete (along w/ rest of loading + processing of unused and/or old
            # data versions)
            prefix = 'PN-KC_Connectivity_'

            # TODO assert set of IDs are same as in my previous hemibrain stuff, or
            # summarize differences?

            df = claws
            # TODO move earlier + use in code in this branch of conditional (before
            # renamed to KC_ID/etc)?
            pn_id_col = PN_ID
            kc_id_col = KC_ID

            weight_col = 'n_synapses'
            # TODO delete? make separate hists of this one in this conditional?
            # (allow weight_col to be list of cols, then loop over below?)
            #weight_col = 'dist_influence'

            # TODO (still an issue?) why were hists below taking so much longer than in
            # tianpei case? fix!

        elif synapse_con_path is not None and synapse_loc_path is not None:
            # used in title of histograms later
            # TODO or does the other one contain more of the important info? use both?
            # just list the containing dir here?
            data_path = synapse_con_path

            #  load & clean connectivity CSV
            df = pd.read_csv(synapse_con_path)

            df['type_pre'] = df['type_pre'].str.replace(
                r'(WED)(PN\d+)', r'\1_\2', regex=True
            )
            df = df.rename(columns={
                'bodyId_pre': 'a.bodyId',
                'bodyId_post':'b.bodyId',
                # TODO TODO what are these? why not all 1? are rows not synapses?
                # cant even clearly find script that generated these. not obviously the
                # committed PN2KC_DBSCAN_Clustering.py i have.
                # (just going to ignore for now. don't care much about
                # non-prat_claws=True outputs now anyway)
                'weight':     'c.weight',
                'type_pre':   'a.type',
                'type_post':  'b.type'
            })
            pn_id_col         = 'a.bodyId'
            kc_id_col         = 'b.bodyId'
            weight_col        = 'c.weight'
            hemibrain_pn_type = 'a.type'
            claw_col          = 'claw'

            df = add_kc_type_col(df, 'b.type')

            # drop funky '+' / multi-underscore types & extract glomerulus
            df, kcs_without_input = add_glomerulus_col_from_hemibrain_type(
                df, hemibrain_pn_type, kc_id_col, check_no_multi_underscores=False,
                _drop_glom_with_plus=_drop_glom_with_plus,
                drop_kcs_with_no_input=drop_kcs_with_no_input
            )

            # TODO delete glom filtering since add_glom_... now should do it?
            #
            # keep only task-et-al glomeruli so no empty claws later.
            df = df[df[glomerulus_col].isin(task_gloms)].copy()
            if isinstance(task_gloms, (list, tuple, pd.Index)):
                # preserve caller’s order if it’s ordered
                glom_order = list(task_gloms)
            else:
                # fallback to deterministic order
                glom_order = sorted(set(task_gloms))

            loc = pd.read_csv(synapse_loc_path)
            loc = loc.rename(columns={
                'bodyId_pre': pn_id_col,
                'bodyId_post': kc_id_col,
            })
            df = df.merge(loc, on=[pn_id_col, kc_id_col], how='inner')

            input_coord_cols = ['x_pre', 'y_pre', 'z_pre']
            for x in input_coord_cols:
                df[x] *= PIXEL_TO_UM

            # DBSCAN cluster into pre_claw
            clustered = []
            for kc, grp in df.groupby(kc_id_col, sort=False):
                pts = grp[input_coord_cols].values

                labels = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples
                    ).fit_predict(pts)

                g = grp.copy()
                g['pre_claw'] = labels
                clustered.append(g)

            df = pd.concat(clustered, ignore_index=True)
            # TODO refactor to share 'pre_claw' throughout
            df = df[df.pre_claw != -1]   # drop noise

            column_order = (
                [kc_id_col, claw_col, KC_TYPE, 'n_syn', glomerulus_col] +
                # TODO factor out pre_cell_ids to shared var
                # TODO TODO where does this column actually come from?
                claw_coord_cols + ['pre_cell_ids']
            )

            # TODO delete (and always do =True)? actually want to preserve old behavior
            # for any reason?  maybe in case some tests only pass that way, to
            # understand why?
            fix_tianpei_pn_to_glom_assignment = True

            if fix_tianpei_pn_to_glom_assignment:
                warn('will drop glomeruli (and their synapses), other than that with'
                    ' the most synapses, for each "claw" from clustering! hack to fix '
                    'how synapses (and PN IDs) from all glomeruli were counted towards '
                    'the max-synapses glomerulus. better fix would involve splitting '
                    'initial clustering "claws" by PN ID.'
                )
                n_minor_pns_dropped = 0
                n_minor_pn_syns_dropped = 0

            # For each (KC,claw), pick the single glomerulus with most synapses
            claws = []
            grouped = df.groupby([kc_id_col, 'pre_claw'], sort=False)
            for (kc, claw), grp in grouped:
                # TODO also make this only run in !fix... case, if fix code will only
                # deal w/ PN IDs, not gloms?
                #
                # count synapses per glomerulus
                counts = grp[glomerulus_col].value_counts()
                chosen_gl = counts.idxmax()
                #

                # find the PNs in this cluster
                if fix_tianpei_pn_to_glom_assignment:
                    counts = grp[pn_id_col].value_counts()
                    chosen_pn = counts.idxmax()

                    grp = grp.loc[grp[pn_id_col] == chosen_pn]

                    gloms = grp[glomerulus_col].unique()
                    assert len(gloms) == 1
                    chosen_gl = gloms[0]

                    # TODO delete
                    #grp = grp.loc[grp[glomerulus_col] == chosen_gl]

                    # TODO or keep track of for each? then report average at end?
                    # same with synapses? report average # (/%) of synapses (+ total)
                    # dropped per claw?
                    n_minor_pns_dropped += (len(counts) - 1)
                    n_minor_pn_syns_dropped += counts[counts.index != chosen_pn].sum()
                    # TODO delete
                    #n_minor_pn_syns_dropped += counts[counts.index != chosen_gl].sum()

                # TODO assert len 1 in fix... case?
                pre_cells = grp[pn_id_col].unique().tolist()

                # compute centroid of the kept synapses
                centroid  = grp[input_coord_cols].mean()

                assert list(centroid.index) == input_coord_cols
                # e.g. 'x_pre' -> 'claw_x'
                centroid = centroid.rename(lambda x: f"claw_{x.split('_')[0]}")
                assert list(centroid.index) == claw_coord_cols

                types = grp[KC_TYPE].unique()
                assert len(types) == 1
                kc_type = types[0]

                claw_dict = {
                    kc_id_col: kc,
                    claw_col: claw,
                    KC_TYPE: kc_type,
                    'n_syn': len(grp),
                    glomerulus_col: chosen_gl,
                    'pre_cell_ids': pre_cells,
                }
                claw_dict.update(centroid.to_dict())

                claws.append(claw_dict)

            claw_df = pd.DataFrame(claws)
            assert set(claw_df.columns) == set(column_order)
            claw_df = claw_df[column_order]

            if fix_tianpei_pn_to_glom_assignment:
                assert_one_glom_per_pn(claw_df.explode('pre_cell_ids'),
                    pn_id_col='pre_cell_ids'
                )

                assert (len(claw_df) ==
                    len(claw_df[[kc_id_col, claw_col]].drop_duplicates())
                )
                n_claws = len(claw_df)

                assert len(df) == claw_df['n_syn'].sum() + n_minor_pn_syns_dropped
                warn(f'dropped {n_minor_pn_syns_dropped}/{len(df)} synapses from '
                    'glomeruli other than one with most synapses (per initial claw)'
                )
                n_total_claws = n_claws + n_minor_pns_dropped
                warn(f'dropped {n_minor_pns_dropped}/{n_total_claws} (claw, glom) '
                    'combos from glomeruli other than one with most synapses (per '
                    'initial claw)'
                )

            min_synapses = 3
            claw_df = claw_df[claw_df['n_syn'] >= min_synapses]

            avg_claws = claw_df.groupby(kc_id_col)[claw_col].nunique().mean()
            print(f'Average claws per KC before merging: {avg_claws:.2f}')

            merge_thresh = 3  # µm
            merged_claws = []

            if fix_tianpei_pn_to_glom_assignment:
                n_minor_pns_dropped = 0
                n_minor_pn_syns_dropped = 0

            # TODO doc what this loop is doing
            for kc, grp in claw_df.groupby(kc_id_col, sort=False):
                # grab the 3D centroids
                coords = grp[claw_coord_cols].values

                # cluster centroids within merge_thresh; min_samples=1 => every point
                # belongs
                merge_labels = DBSCAN(eps=merge_thresh, min_samples=1
                    ).fit_predict(coords)

                grp = grp.assign(merge_id=merge_labels)

                # now collapse each merge-cluster into one “super-claw”
                for merge_id, sub in grp.groupby('merge_id', sort=False):

                    # TODO also make this only run in !fix... case, if fix code will
                    # only deal w/ PN IDs, not gloms?
                    #
                    # pick the glomerulus with the most synapses (sum over sub-claws)
                    gl_counts = sub.groupby(glomerulus_col)['n_syn'].sum()
                    chosen_gl = gl_counts.idxmax()
                    #

                    if fix_tianpei_pn_to_glom_assignment:
                        # at one point i wasn't expecting this, but seems true...
                        assert (sub['pre_cell_ids'].str.len() == 1).all()

                        sub = sub.explode('pre_cell_ids')

                        counts = sub['pre_cell_ids'].value_counts()
                        chosen_pn = counts.idxmax()

                        sub = sub.loc[sub['pre_cell_ids'] == chosen_pn]

                        gloms = sub[glomerulus_col].unique()
                        assert len(gloms) == 1
                        chosen_gl = gloms[0]

                        # TODO delete
                        #sub = sub.loc[sub[glomerulus_col] == chosen_gl]

                        n_minor_pns_dropped += (len(counts) - 1)
                        n_minor_pn_syns_dropped += counts[counts.index != chosen_pn
                            ].sum()

                        # TODO or put back into list here, for .explode later? or just
                        # delete .explode later (and keep as single int here)?
                        all_pre_ids = [chosen_pn]
                    else:
                        # union of all pre_cell_ids
                        all_pre_ids = sorted(
                            {pid for lst in sub['pre_cell_ids'] for pid in lst}
                        )

                    total_n_syn = sub['n_syn'].sum()

                    # centroid weighted by n_syn
                    weighted = (sub[claw_coord_cols].multiply(sub['n_syn'], axis=0)
                                    .sum(axis=0) / total_n_syn)

                    types = sub[KC_TYPE].unique()
                    assert len(types) == 1
                    kc_type = types[0]

                    claw_dict = {
                        kc_id_col: kc,
                        claw_col: merge_id,
                        KC_TYPE: kc_type,
                        'n_syn': total_n_syn,
                        glomerulus_col: chosen_gl,
                        'pre_cell_ids': all_pre_ids,
                    }
                    assert set(weighted.keys()) == set(claw_coord_cols)
                    claw_dict.update(weighted)
                    merged_claws.append(claw_dict)

            # overwrite claw_df with the merged result
            claw_df_merged = pd.DataFrame(merged_claws)

            if fix_tianpei_pn_to_glom_assignment:
                assert_one_glom_per_pn(claw_df_merged.explode('pre_cell_ids'),
                    pn_id_col='pre_cell_ids'
                )

                n_claws = len(claw_df_merged)
                assert (n_claws ==
                    len(claw_df_merged[[kc_id_col, claw_col]].drop_duplicates())
                )

                n_total_syn = claw_df['n_syn'].sum()
                # TODO (delete) what accounts for this difference? not huge, at least...
                # (it's the actual merging, i assume...)
                # ipdb> claw_df['n_syn'].sum()
                # 173077
                # ipdb> claw_df_merged['n_syn'].sum()
                # 173029
                # ipdb> claw_df_merged['n_syn'].sum() + n_minor_pn_syns_dropped
                # 173038
                #assert (n_total_syn ==
                #    claw_df_merged['n_syn'].sum() + n_minor_pn_syns_dropped
                #)
                warn(f'(while merging) dropped {n_minor_pn_syns_dropped}/{n_total_syn}'
                    ' synapses from glomeruli other than one with most synapses (per '
                    'initial claw)'
                )
                n_total_claws = n_claws + n_minor_pns_dropped
                warn(f'(while merging) dropped {n_minor_pns_dropped}/{n_total_claws} '
                    '(claw, glom) combos from glomeruli other than one with most '
                    'synapses (per initial claw)'
                )

            claw_df = claw_df_merged

            assert set(claw_df.columns) == set(column_order)
            claw_df = claw_df[column_order]

            avg_claws = claw_df.groupby(kc_id_col)[claw_col].nunique().mean()
            print(f'Average claws per KC after merging: {avg_claws:.2f}')

            # after explode call, # rows should be expanded to include one for each
            # TODO rename to pre_cell_id or pn_col_id after this?
            claw_df = claw_df.explode('pre_cell_ids')
            # TODO check calculations of # avg claws (/similar) from wPNKC later match
            # what he printed above? assuming above prints are even correct, and match
            # final processing...

            claw_cols = [kc_id_col, claw_col, 'pre_cell_ids']

            # TODO only do behind fix_tianpei... flag? (including explode above?)
            # TODO TODO renumber claws, splitting so that each PN ID (within a (KC,
            # claw) combo) gets its own ID
            # TODO +refactor fn for doing this, since i keep forgetting how
            # TODO + refactor assumption that there is only one PN per claw, and have
            # that trigger before whatever else was triggering first? or ig it was
            # basically doing that anyway (be more explicit if it wasn't already
            # referencing pn_id_col tho)
            # NOTE: want to keep distinct claw IDs, if already have separate PN ID, but
            # want to split claw IDs associated with multiple PN IDs
            #
            # TODO delete (shouldn't be hit now, since splitting by PNs above?)
            # TODO TODO or am i not actually renumbering above, like i might need
            # to be? (matter, if i'm dropping more?)
            #
            # TODO TODO compare this to after changing to use PN above (may
            # actually end up just wanting to renumber here instead? as long as synapses
            # counted per PN above, and not per glom?)?
            # from when above was still using glomerulus_col instead of PN ID:
            # Warning: will drop glomeruli (and their synapses), other than that with
            # the most synapses, for each "claw" from clustering! hack to fix how
            # synapses (and PN IDs) from all glomeruli were counted towards the
            # max-synapses glomerulus. better fix would involve splitting initial
            # clustering "claws" by PN ID.
            # Warning: dropped 7298/180923 synapses from glomeruli other than one with
            # most synapses (per initial claw)
            # Warning: dropped 1172/12434 (claw, glom) combos from glomeruli other than
            # one with most synapses (per initial claw)
            # ...
            # Average claws per KC before merging: 6.16
            # Warning: (while merging) dropped 38/173605 synapses from glomeruli other
            # than one with most synapses (per initial claw)
            # Warning: (while merging) dropped 8/11051 (claw, glom) combos from
            # glomeruli other than one with most synapses (per initial claw)
            #
            # TODO TODO replace w/ assertion first? (at least done in fix_...=True
            # case)
            # TODO delete
            '''
            claw_df = claw_df.sort_values(claw_cols).reset_index(drop=True)
            for kc, kdf in claw_df.groupby(kc_id_col, sort=False):
                if (len(kdf[claw_cols].drop_duplicates()) >
                    len(kdf[claw_cols[:-1]].drop_duplicates()) ):
                    print()
                    print(f'{kc=}')
                    print('kdf:')
                    print(kdf[[x for x in kdf.columns if x not in claw_coord_cols]].to_string())
                    breakpoint()
                #
            '''

            claw_levels = [kc_id_col, claw_col, KC_TYPE] + claw_coord_cols

            # TODO share below w/ prat_claws=True? is pivot[_table] call below not used
            # in this case?
            # pivot to a (KC,claw,x,y,z)×glomerulus matrix, values = 1
            wPNKC_counts = claw_df.pivot_table(
                # TODO (delete? assert if i care?) can i make this work? (wPNKC.T.sum()
                # == 1).all() still?
                index=claw_levels + ['pre_cell_ids'],
                columns=glomerulus_col,
                values='n_syn',
                fill_value=0
            ).astype(int)

            present = [g for g in glom_order if g in wPNKC_counts.columns]
            wPNKC_counts = wPNKC_counts.loc[:, present]
            # TODO this even needed anymore? (yes, i think counts still counts #
            # synapses w/in each claw?)
            wPNKC = (wPNKC_counts > 0).astype(int)

            pn_id_col = PN_ID

            wPNKC = wPNKC.rename_axis(index={'pre_cell_ids': pn_id_col})

            # TODO TODO where is (/was) 'pre_cell_ids' column removed from wPNKC later?
            # is it in returned + run wPNKC? (should prob be in index level to avoid
            # that, if so, or drop here. used outside this fn?)

            # TODO just change claw_col to CLAW_ID above, for both?
            df = claw_df.rename(columns={claw_col: CLAW_ID, 'pre_cell_ids': pn_id_col})
            weight_col = 'n_syn'

            wPNKC = wPNKC.rename_axis(index={claw_col: CLAW_ID})

            wPNKC = add_compartment_index(wPNKC, shape=0)

        else:
            data_path = repo_root / 'data/PNtoKC_connections_raw.xlsx'
            df = pd.read_excel(data_path)

            pn_id_col = 'a.bodyId'
            kc_id_col = 'b.bodyId'
            # TODO assert this is all 1? is it? or all integer # synapses here?
            weight_col = 'c.weight'

            hemibrain_pn_type = 'a.type'

            df_with_types = add_kc_type_col(df, 'b.type')

            # TODO move this call into `not _use_matt_wPNKC` case below (to share w/
            # connectome='fafb-[left|right]' cases below)?
            df, kcs_without_input = add_glomerulus_col_from_hemibrain_type(
                df_with_types.copy(), hemibrain_pn_type, kc_id_col,
                check_no_multi_underscores=True,
                _drop_glom_with_plus=_drop_glom_with_plus,
                drop_kcs_with_no_input=drop_kcs_with_no_input
            )

            # TODO use later (to restore to wPNKC)
            df_noinput_kcs = df_with_types[
                df_with_types[kc_id_col].isin(kcs_without_input)
            ].copy()
    else:
        fafb_dir = from_prat / '2024-09-13'

        if connectome == 'fafb-left':
            data_path = fafb_dir / 'FlyWire_PNKC_Left.csv'
        else:
            assert connectome == 'fafb-right'
            data_path = fafb_dir / 'FlyWire_PNKC_Right.csv'

        df = pd.read_csv(data_path)

        cols_with_nan = df.isna().any()
        assert set(cols_with_nan[cols_with_nan].index) == {
            'source_cell_type', 'target_cell_type'
        }

        assert (df.source_cell_class == 'ALPN').all()
        assert (df.target_cell_class == 'Kenyon_Cell').all()

        if connectome == 'fafb-left':
            assert (df.source_side == 'left').all()

            # askprat: drop those w/ target side == 'right'?
            # Prat: eh, any of these options could work. up to me.
            # (prob doesn't matter much anyway, looking at which glomeruli it is...)
            #
            # TODO or just use them too? could turn to be roughly equiv to using values
            # from other csv (appending two w/ same 'target_side' together)?
            # TODO or load both csvs and add both together, based on target
            # side (prob doesn't matter hugely w/ how many fewer connections there are)?
            # TODO is there anything else special about these PNs that cross midline?
            # (maybe we'd exclude already anyway, for some other reason?)
            #
            # ipdb> df.target_side.value_counts()
            # left     13774
            # right      275
            #
            # Prat: below mostly/all the bilateral PNs he was excited about before, that
            # i had helped him image
            #
            # ipdb> df.loc[df.target_side == 'right', 'source_hemibrain_type'].value_counts()
            # V_ilPN                  87
            # VP3+VP1l_ivPN           65
            # VP1m+VP5_ilPN           45
            # VP1d_il2PN              42
            # VL1_ilPN                29
            # M_ilPNm90,M_ilPN8t91     3
            # VP1l+VP3_ilPN            2
            # M_smPNm1                 1
            # M_smPN6t2                1
        else:
            assert (df.source_side == 'right').all()
            # ipdb> df.target_side.value_counts()
            # right    13485
            # left       314
            #
            # ipdb> df.loc[df.target_side == 'left', 'source_hemibrain_type'].value_counts()
            # V_ilPN                  145
            # VP1m+VP5_ilPN            60
            # VP3+VP1l_ivPN            48
            # VP1d_il2PN               30
            # VL1_ilPN                 25
            # M_ilPNm90,M_ilPN8t91      3
            # M_smPN6t2                 2
            # VP1l+VP3_ilPN             1

        pn_id_col = 'source'
        kc_id_col = 'target'
        weight_col = 'weight'

        assert df['target_hemibrain_type'].notna().all()
        # ipdb> df.target_hemibrain_type.value_counts()
        # KCg-m         5435
        # KCab-m        1718
        # KCab-s        1586
        # KCab-c         952
        # KCa'b'-m       483
        # KCa'b'-ap2     350
        # KCa'b'-ap1     297
        # KCg-d           39
        # KCab-p           1
        # KCg-s2           1
        df = add_kc_type_col(df, 'target_hemibrain_type')

        hemibrain_pn_type = 'source_hemibrain_type'
        df, kcs_without_input = add_glomerulus_col_from_hemibrain_type(df,
            hemibrain_pn_type, kc_id_col, _drop_glom_with_plus=_drop_glom_with_plus,
            drop_kcs_with_no_input=drop_kcs_with_no_input
        )

        # this should be the same as the min_weight from hemibrain, where in that case
        # prat's query is what filtered stuff w/ smaller weight
        df = df[df[weight_col] >= 4].copy()

        # TODO delete
        # fafb_types = df.source_cell_type.dropna()
        # # true for at least fafb-left
        # assert (fafb_types.str.count('_') == 1).all()
        # fafb_gloms = first_delim_sep_part(fafb_types, sep='_')
        # odf = df.dropna(subset=['source_cell_type'])
        # assert odf.index.equals(fafb_gloms.index)
        # print(pd.concat([fafb_gloms, odf.glomerulus], axis=1).drop_duplicates())
        #
        # (askprat) want to change handling of any of these? have been
        # merging VC3l and VC3m into "VC3" (w/ old hemibrain stuff, at least). should
        # glomerulus_renames reflect this?
        # Prat: just completely ignore the source_cell_type values. almost certainly not
        # meaningful corrections made by the flywire people.
        #
        # why does task not split them? same receptor or something?
        # (task doesn't refer to VC3l/m, but does list diff receptors/etc for VC3 and
        # VC5. they also split VM6 into VM6v/m/l [all w/ at least mostly same
        # receptors]. they might also be saying that the "canonical" VM6 was VM6v?)
        #
        # (same combinations for both left/right)
        #       source_cell_type glomerulus
        # 630                VC3       VC3l
        # 9868               VC5       VC3m
        # 10678              VM6        VC5
        #
        # ipdb> 'VM6' in set(df.glomerulus)
        # False

        # askprat: do anything w/ 'target_hemibrain_type'? e.g. 'KCab-s', 'KCg-m',
        # etc (prob not, at least not categorically filtering out any of them)
        # Prat: final part after dash is from clustering on connectome. no reason to
        # exclude any of this.
        #
        # connectome='fafb-left'
        # ipdb> df.target_hemibrain_type.value_counts()
        # KCg-m         6463
        # KCab-m        1893
        # KCab-s        1874
        # KCab-c        1221
        # KCa'b'-m       765
        # KCa'b'-ap2     553
        # KCa'b'-ap1     421
        # KCg-d           66
        # KCab-p          44
        # KCg-s2           6
        # KCg-s3           4
        # KCg-s1           2
        #
        # connectome='fafb-right'
        # ipdb> df.target_hemibrain_type.value_counts()
        # KCg-m         6555
        # KCab-s        1800
        # KCab-m        1542
        # KCab-c        1357
        # KCa'b'-m       745
        # KCa'b'-ap2     613
        # KCa'b'-ap1     346
        # KCg-d           73
        # KCab-p          51
        # KCg-s2           5
        # KCg-s3           1
        # KCg-s1           1

    # keeping mostly to check _use_matt_wPNKC=True outputs now. assertion below, for
    # other cases, should be mostly redundant
    def _get_kcs_without_input(wPNKC):
        kcs_without_input = (wPNKC == 0).T.all()
        n_kcs_without_input = kcs_without_input.sum()
        return kcs_without_input, n_kcs_without_input

    if _use_matt_wPNKC:
        _, n_kcs_without_input = _get_kcs_without_input(wPNKC)
        # so we can skip the code that would handle them below
        assert n_kcs_without_input == 0

        # TODO also need to sort_index(axis='index') in this case? i assume no?
        # TODO try to move all of this type of sorting of wPNKC up here (rather than
        # near other return, for other cases), and check outputs don't change
        # (so i can just have one call above this conditional)
        wPNKC = wPNKC.sort_index(axis='columns')

        wPNKC = wPNKC.rename_axis(index={'bodyid': KC_ID})

        return wPNKC

    kc_types = None
    id_cols_to_check = [kc_id_col, pn_id_col]
    if one_row_per_claw:
        id_cols_to_check.append(CLAW_ID)

    #if prat_boutons:
    # TODO how to handle bouton_id here? actually want to include in this
    # check? (sufficiently checked elsewhere already i assume? delete?)
    assert df.reset_index()[id_cols_to_check].notna().all().all()

    df_reset = df.reset_index()

    # TODO also add bouton ID col as needed?
    unique_cols = [pn_id_col, kc_id_col]
    if one_row_per_claw:
        unique_cols.append(CLAW_ID)

    # TODO make another version of wPNKC, where pn_id_col is another column level
    # beyond (what is currently just glomerulus), to help clarify some of the issues
    # below? (for prat_claws=True cases) (maybe also where there's an additional
    # bouton ID, when available) (return that way by default?)

    assert len(df_reset[unique_cols].drop_duplicates()) == len(df)

    # askprat: so does this mean prat has already excluded multiglomeruli PNs
    # (intentionally or not), or are they all contained w/in stuff dropped above?
    #
    # Prat: doesn't think it's likely any of his queries would have missed MG PNs
    # (and as other comments say 'M' in PN type str probably means multiglomerular,
    # or at least includes them)
    #
    # seems True for both hemibrain and fafb (at least as long as we are dropping
    # stuff w/ multiple '_' or '+' in PN types above...)
    assert_one_glom_per_pn(df, pn_id_col=pn_id_col)

    # TODO prob define this outside conditional (if defined separately in
    # prat_claws=True and tianpei cases [and some other]. could assert same
    # here, and maybe remove those defs above?). may differ in cases where some
    # claws/KCs have 0 input, if that's ever still true? dropping KCs w/ no
    # input should only hapen below
    n_kcs = df_reset[kc_id_col].nunique()
    if one_row_per_claw:
        n_claws = len(df_reset[[kc_id_col, CLAW_ID]].drop_duplicates())

    # TODO if tianpei's outputs failing, check to see if 'pre_claw' being used
    # instead of CLAW_ID would fix it (then redef above?)? otherwise delete this
    # old commented code
    #assert (
    #    len(df[[pn_id_col, kc_id_col, 'pre_claw']].drop_duplicates()) == len(df)
    #)

    # both of these assertions also work in prat_claws=True case (as of 2025-09-18)
    assert not df[weight_col].isna().any()
    min_weight = df[weight_col].min()
    assert min_weight > 0
    # TODO delete
    # was true b/c Prat's query in hemibrain case, and b/c subsetting above in
    # fafb cases
    #assert min_weight == 4

    if not one_row_per_claw:
        if weight_divisor is None:
            # TODO refactor pivoting to share across branches of this conditional,
            # and with above processing of matt's CSVs
            wPNKC = pd.pivot_table(df, values=weight_col, index=kc_id_col,
                columns=glomerulus_col, aggfunc='count').fillna(0).astype(int)

            # TODO delete? (/ move to unit test)
            df_bin = df.copy()
            assert (df[weight_col] > 0).all()
            df_bin[weight_col] = (df_bin[weight_col] > 0).astype(int)
            assert (df_bin[weight_col] == 1).all()
            wb = pd.pivot_table(df_bin, values=weight_col, index=kc_id_col,
                columns=glomerulus_col, aggfunc='sum').fillna(0).astype(int)
            assert wb.equals(wPNKC)
            del df_bin, wb
            #
        else:
            assert weight_divisor > 0

            using_count = pd.pivot_table(df, values=weight_col, index=kc_id_col,
                columns=glomerulus_col, aggfunc='count').fillna(0).astype(int)

            wdf = df.copy()
            wdf[weight_col] = np.ceil(wdf[weight_col] / weight_divisor)

            wPNKC = pd.pivot_table(wdf, values=weight_col, index=kc_id_col,
                columns=glomerulus_col, aggfunc='sum').fillna(0).astype(int)
            del wdf

            assert (wPNKC >= using_count).all().all()
            # if weight_divisor is too large, this could fail
            # TODO maybe check it does (i.e. that wPNKC.equals(using_count), for
            # high enough weight_divisor)?
            assert (wPNKC > using_count).any().any()
            del using_count

    # TODO TODO check plot outputs (and others) same w/ or w/o this in prat_claws=True
    # case, then go back to skipping this in prat_claws=True case (+ try to avoid need
    # for it in other cases too, mainly tianpei path)
    # TODO TODO and actually just maintain KC_TYPE earlier for prat_claws=False stuff
    # (->delete this conditional?)
    #
    # TODO delete? or actually don't want in prat_claws=True case?
    #if KC_TYPE in df.columns and not prat_claws:
    # TODO test if we can also skip (or if even works + is used below, for
    # one_row_per_claw=False cases)
    if KC_TYPE in df.columns and not one_row_per_claw:
        # need the df.reset_index() for at least prat_claws=True df here

        # TODO delete?
        kc_ids_and_types = wPNKC.index.to_frame(index=False).merge(
            df.reset_index()[[kc_id_col, KC_TYPE]].drop_duplicates(), on=kc_id_col
        )
        # TODO what is this checking? kinda seems like a merge above isn't getting
        # us anything...
        # TODO replace w/ `if not prat_claws`?
        if synapse_loc_path is None and synapse_con_path is None:
            assert np.array_equal(
                kc_ids_and_types[kc_id_col], wPNKC.index.get_level_values(kc_id_col)
            )

        # just to handle prat_claws=True case, which probably already has totally fine
        # KC_TYPE, and thus doesn't need this outer conditional at all...
        if KC_TYPE not in kc_ids_and_types.columns:
            tx = f'{KC_TYPE}_x'
            ty = f'{KC_TYPE}_y'
            assert all(x in kc_ids_and_types.columns for x in (tx, ty))
            assert kc_ids_and_types[tx].equals(kc_ids_and_types[ty])
            kc_ids_and_types[KC_TYPE] = kc_ids_and_types[tx]
            kc_ids_and_types = kc_ids_and_types.drop(columns=[tx, ty])

        kc_types = kc_ids_and_types[KC_TYPE]
    #

    if not one_row_per_claw:
        assert CLAW_ID not in wPNKC.index.names
        n_kcs2 = len(wPNKC)
        assert n_kcs2 == wPNKC.index.get_level_values(kc_id_col).nunique()
    else:
        assert wPNKC.index.get_level_values(CLAW_ID).notna().all()

        n_claws2 = len(wPNKC)
        assert (n_claws2 == len(wPNKC.index.to_frame(index=False)[[kc_id_col, CLAW_ID]
            ].drop_duplicates())
        )

        # this is just to check a prior calculation in prat_claws=True case. pass?
        if n_prat_claws is not None:
            assert n_claws2 == n_prat_claws

        kc_ids = wPNKC.index.get_level_values(kc_id_col)
        # TODO also similar checks (that hits all branches of code that have PN ID /
        # bouton ID), that check all those are all non-NaN too
        assert kc_ids.notna().all()
        # excludes NaN by default, but we already checked we don't have any of those
        n_kcs2 = kc_ids.nunique()

        if dist_weight is None:
            # TODO will this still be true if some claws have no input? is that
            # possible here? (dropping KCs with no input happens below, but not sure
            # if that ever applies on a per-claw basis?)
            assert wPNKC.sum().sum() == n_claws2
            assert set(np.unique(wPNKC.values.flat)) == {0, 1}
        else:
            assert (wPNKC > 0).sum().sum() == n_claws2, ('possibly 0 weight for some '
                'claws with synapses? make sense?'
            )

    # checking calculations from long-form df and wide wPNKC match
    assert n_kcs == n_kcs2, f'{n_kcs=} (from df) != {n_kcs2} (from wPNKC)'
    del n_kcs2
    if one_row_per_claw:
        assert n_claws == n_claws2, f'{n_claws=} (from df) != {n_claws2} (from wPNKC)'
        del n_claws2

    assert 1000 < n_kcs < 2800, f'unexpected # of KCs: {n_kcs}'
    if one_row_per_claw:
        # 11043 in Tianpei case, after adding my own fix_...=True code to split on PNs.
        # I previously had the upper bound at 10000
        assert 4000 < n_claws < 12000, f'unexpected # of claws: {n_claws}'

    # TODO delete this conditional altogether? can i rewrite to avoid need?
    # (just need to check the one_row_per_claw=False cases)
    if kc_types is not None:
        # TODO assert indices are same, instead (stronger)?
        assert len(kc_types) == len(wPNKC), f'{len(kc_types)} != {len(wPNKC)=}'

        # TODO delete special casing?
        if 'compartment' in wPNKC.index.names:
            for_index = wPNKC.index.to_frame(index=False)
            for_index[KC_TYPE] = kc_types
            kc_index = pd.MultiIndex.from_frame(for_index)
        else:
            assert len(wPNKC.index.names) == 1, 'otherwise, also need from_frame here'
            kc_index = pd.MultiIndex.from_arrays([wPNKC.index, kc_types])

        # TODO fix for one_row_per_claw=True cases (/assert kc_types only used for other
        # ones, or delete `kc_types` code altogether)
        assert kc_index.to_frame(index=False).equals(kc_ids_and_types)
    else:
        kc_index = wPNKC.index.copy()

    if isinstance(kc_index, pd.MultiIndex):
        # .rename seemed to have same effect with dict in 1.5.0, but docs don't seem to
        # mention it, and never seem to mention [MultIndex|Index].rename accepts
        # dict-like as one option
        kc_index = kc_index.set_names({kc_id_col: KC_ID})
    else:
        assert kc_index.name == kc_id_col
        kc_index = kc_index.set_names(KC_ID)

    # (for hemibrain)
    # ipdb> kc_index.get_level_values(KC_TYPE).value_counts(dropna=False)
    # ab      802
    # g       612
    # a'b'    336
    # NaN      80
    wPNKC.index = kc_index

    _, n_without_input = _get_kcs_without_input(wPNKC)
    assert n_without_input == 0
    #

    # TODO even possible w/ pre-filtering glomeruli now? test? (may need to move
    # assertion into add_glom... now?)
    assert not (wPNKC == 0).all().any(), 'had Task glomeruli providing no input to KCs'

    if not drop_kcs_with_no_input:
        if df_noinput_kcs is None:
            # currently only implemented for non-one_row_per_claw hemibrain case
            # (for hemibrain_paper_repro test, which is all that really uses
            # drop_kcs_with_no_input=False, I believe)
            raise NotImplementedError('must define df_noinput_kcs in path above, to '
                'support drop_kcs_with_no_input=False'
            )

        # TODO do earlier?
        df_noinput_kcs = df_noinput_kcs.rename(columns={kc_id_col: KC_ID})

        noinput_kc_metadata = df_noinput_kcs[wPNKC.index.names].drop_duplicates()
        assert len(noinput_kc_metadata) == len(kcs_without_input)

        wPNKC_noinput = pd.DataFrame(data=0, columns=wPNKC.columns,
            index=pd.MultiIndex.from_frame(noinput_kc_metadata)
        )
        wPNKC = pd.concat([wPNKC, wPNKC_noinput], verify_integrity=True)

        # TODO something implicitly/excplicitly sorting rows above anyway?
        # needed to repro paper outputs
        wPNKC = wPNKC.sort_index(axis='index')

        n_kcs = len(wPNKC)

    if plot_dir is not None:
        # TODO also run in tianpei case, assuming we can get a similar variable to
        # claws, consistently organized and defined in all one_row_per_claw=True cases
        if prat_claws:
            # TODO TODO delete these plots? redundant w/ below now that it's after
            # filtering?
            fig, ax = plt.subplots()
            sns.histplot(claws, ax=ax, x='n_synapses', discrete=True, hue=KC_TYPE,
                hue_order=kc_type_hue_order
            )
            # TODO show # (/frac) of filtered claws (e.g. based on
            # synapse_per_1pn_claw_thresh) in title/etc for any of these plots?
            # TODO better title? (# claws defined this way?)
            #ax.set_title(f'input from each PN counted separately\n{data_path.name}')
            # NOTE: this plot was originally before filtering based on
            # synapse_per_1pn_claw_thresh (want a version before that? prob not)
            savefig(fig, plot_dir, 'wPNKC_prat_syns-per-1pn-claw',
                bbox_inches='tight'
            )

            fig, ax = plt.subplots()
            # adding KC_TYPE here should not change # of unique groups (it's just to
            # access for hue in plot below)
            syns_per_claw = claws.groupby([KC_ID, CLAW_ID, KC_TYPE]
                ).n_synapses.sum().reset_index()

            assert len(syns_per_claw) == len(claws)

            sns.histplot(syns_per_claw, ax=ax, x='n_synapses', discrete=True,
                hue=KC_TYPE, hue_order=kc_type_hue_order
            )
            # TODO better title? (# claws defined this way?)
            # TODO delete (no longer after, now that this is after filtering)
            ##ax.set_title(f'claws can contain multiple PNs\n{data_path.name}')
            #
            #ax.set_title(f'# synapses per claw\n(only 1 PN per claw)\n{data_path.name}')

            # TODO drop unused types from legend
            # NOTE: this plot was originally before filtering based on
            # synapse_per_1pn_claw_thresh (want a version before that? prob not)
            savefig(fig, plot_dir, 'wPNKC_prat_syns-per-claw', bbox_inches='tight')

            # TODO separate hists for dist_influence / dist_to_root?
            # (in prat_claws=True case)

        # at least in connectome='hemibrain' & _drop_glom_with_plus=True, wPNKC here has
        # two KCs with no input connections, which are not in reprocessed df.
        # doesn't matter tho.
        df = df[df[glomerulus_col].isin(wPNKC.columns)]

        n_unit_str = f'\n{n_kcs=}'
        if CLAW_ID in wPNKC.index.names:
            n_unit_str += f'\n{n_claws=}'

        data_str = f'{connectome} PN->KC weights\n{min_weight=}\n{data_path.name}'

        fig, ax = _plot_connectome_raw_weight_hist(df[weight_col])
        ax.set_title(f'{data_str}{n_unit_str}')
        # bbox_inches='tight' necessary for title to not be cut off
        savefig(fig, plot_dir, f'wPNKC_hist_{connectome}', bbox_inches='tight')

        # temporarily comment out (come back to check later)
        fig, ax = _plot_connectome_raw_weight_hist(df, x=weight_col, hue=KC_TYPE,
            hue_order=kc_type_hue_order
        )
        ax.set_title(f'{data_str}{n_unit_str}')
        savefig(fig, plot_dir, f'wPNKC_hist_{connectome}_by-kc-type',
            bbox_inches='tight'
        )

        # NOTE: mean of this w/ connectome='hemibrain' is 5.44 (NOT n_claws=7 used
        # by uniform) (w/ what weight_divisor? or binarizing? delete comment?)
        n_inputs_per_kc = wPNKC.T.sum()

        if one_row_per_claw:
            # all other levels are specific to each claw
            n_inputs_per_kc = n_inputs_per_kc.groupby(level=[KC_ID, KC_TYPE], sort=False
                ).sum()

        assert len(n_inputs_per_kc) == n_kcs

        # relevant for picking appropriate n_claws for uniform/hemidraw cases, or for
        # picking weight_divisor that produces closest avg to the n_claws=7 we had
        # already been using
        avg_n_inputs_per_kc = n_inputs_per_kc.mean()

        # Reset the index, ensuring to fill NaNs in the kc_type level
        kcs_with_type_and_nclaws = n_inputs_per_kc.reset_index(name='n_claws')

        # Now, use .fillna() on the 'kc_type' column to replace any NaNs
        kcs_with_type_and_nclaws[KC_TYPE].fillna('unknown', inplace=True)

        # TODO label y-axis for two below (and *differently* for those above)?

        fig, ax = plt.subplots()
        sns.histplot(n_inputs_per_kc, discrete=True, ax=ax)
        ax.set_xlabel('# "claws" per KC\n(after processing connectome weights)')

        ax.set_title(f'total inputs per KC\n{connectome=}\n{weight_divisor=}\n'
            f'{prat_claws=}{n_unit_str}\nmean inputs per KC: '
            f'{avg_n_inputs_per_kc:.2f}'
        )
        # TODO why these look so different for fafb inputs (vs hemibrain)? for similar
        # avg_n_inputs_per_kc (adjusting fafb weight_divisor to roughly match the value
        # for hemibrain w/ weight_divisor=20), we get much more of the right lobe in the
        # fafb plots. are there less (PN, KC) pairs for some reason (what should be the
        # only lobe w/ weight_divisor=None, or the left lobe in others)?
        savefig(fig, plot_dir, f'wPNKC_nclaws-sum-per-KC_hist_{connectome}',
            bbox_inches='tight'
        )

        # TODO refactor to share w/ above?
        #
        # same as above, just also breaking down by kc_type, with separate hues
        fig, ax = plt.subplots()
        sns.histplot(data=kcs_with_type_and_nclaws, discrete=True, ax=ax, x='n_claws',
            hue=KC_TYPE, hue_order=kc_type_hue_order
        )
        ax.set_xlabel('# "claws" per KC\n(after processing connectome weights)')
        ax.set_title(f'total inputs per KC\n{connectome=}\n{weight_divisor=}'
            f'{n_unit_str}\nmean inputs per KC: {avg_n_inputs_per_kc:.2f}'
        )
        savefig(fig, plot_dir, f'wPNKC_nclaws-sum-per-KC_hist_{connectome}_by-kc-type',
            bbox_inches='tight'
        )

        # TODO also plot sums within glomeruli?
        # TODO delete
        #print('also plot hists per glomerulus / PN?')

        # TODO also plot (hierarchichally clustered) wPNKC (w/ plot+colorscale as in
        # natmix_data/analysis.py?)?

    if CLAW_ID not in wPNKC.index.names:
        assert len(wPNKC) == n_kcs
    else:
        assert len(wPNKC) == n_claws

    # TODO see if i can also sort output in uniform case (b/c if not, might suggest
    # there is other order-of-glomeruli dependence in fit_mb_model THAT THERE SHOULD NOT
    # BE)
    # TODO (still true? test described actually relevant? delete?) i can not seem to
    # recreate uniform output (by sorting wPNKC post-hoc), but i'm not sure that's
    # actually a problem. maybe input *should* always just be in a particular order, and
    # shouldn't necessarily matter that it's this one...
    wPNKC = wPNKC.sort_index(axis='columns')

    if Btn_separate:
        warn(f'expanding wPNKC to fixed # of boutons ({Btn_num_per_glom=}) per PN')
        wPNKC_btn = expand_wPNKC_to_boutons(wPNKC=wPNKC,
            boutons_per_glom=Btn_num_per_glom
        )
        # TODO delete
        # ipdb> one_row_per_claw
        # True
        # ipdb> wPNKC_btn.iloc[:2, :2]
        # glomerulus                                D
        # bouton_id                                 1    2
        # kc_id     claw_id kc_type ... pn_id
        # 300968622 0       ab      ... 1851389175  0.0  0.0
        #           1       ab      ... 635062078   0.0  0.0
        # ipdb> wPNKC.stack().value_counts()
        # 0    585279
        # 1     11043
        # dtype: int64
        # ipdb> wPNKC_btn.stack().stack().value_counts(dropna=False)
        # 0.0    5852790
        # 0.1     110430
        wPNKC = wPNKC_btn

    # TODO TODO TODO reformat (either his or mine), so boutons are in consistent
    # spot for both prat_boutons=True (+ whatever other flag actually enables
    # distinct ones, if i require one) and tianpei's Btn_separate=True
    #
    # TODO TODO + also want to divide weight be # boutons in this case?
    # or should i be counting and storing weight separately from the very beginning?
    # (and same question in PN<>APL context. what's currently happening there?)
    if prat_boutons:
        # need to redef here, b/c currently also has PN_ID at end above
        claw_cols = [KC_ID, CLAW_ID]

        # TODO TODO refactor to also have this available outside? compute outside,
        # before running sims (to hopefully not get killed...) (or optionally return?)
        # TODO can i already compute something close enough from wPNKC at output or nah?
        #
        # stack() is to get a glomerulus index level from columns (which gives us a
        # Series with all 1 for values, that we don't need, so just use the index
        # from there)
        claw2bouton = wPNKC.replace(0, np.nan).stack().index.to_frame(
            index=False).set_index(claw_cols).explode(BOUTON_ID).reset_index()

        # TODO maybe drop -1 (+ NaN, if any) BOUTON_ID first (from apl<>pn stuff!),
        # before merge?  and then also merge in a way that doesn't add NaNs (if that
        # doesn't make things harder)? just if it helps clarify overall # of
        # synapses before vs after
        # (should be asserting all id_cols *in wPNKC.index*, which include
        # BOUTON_ID, are not NaN here, if not already established)

        id_cols = [KC_ID, CLAW_ID, PN_ID, BOUTON_ID]
        assert not claw2bouton[id_cols].duplicated().any()

        assert claw2bouton[BOUTON_ID].min() == 0, \
            'wPNKC index should already have had invalid (-1) bouton IDs dropped'

        # checking that our glomerulus column->index transformation (as part of
        # claw2bouton def above) didn't screw anything up
        assert_one_glom_per_pn(claw2bouton)
        claw2bouton2 = wPNKC.index.to_frame(index=False).explode(BOUTON_ID)
        # TODO also assert not duplicated on claw_cols either now?
        # (would fail if i wasn't dropping cases of n-boutons:1-claw earlier now
        # [in connectome_wPNKC], unless i also added handling to dedupe them)
        assert not claw2bouton[id_cols].duplicated().any()
        assert len(claw2bouton) == len(claw2bouton2)
        del claw2bouton2

        # currently 10216 (update. prob no longer true after dropping multibouton)
        n_claw_bouton_combos = len(claw2bouton)

        # 9867
        n_claws = len(wPNKC)
        # NOTE: if claws that were associated with multiple boutons (either 2 or 3)
        # had been dropped / split by this  point, would have to change this
        # assertion to equality (dropped after this line though)
        assert n_claws < n_claw_bouton_combos, f'{n_claws=} >= {n_claw_bouton_combos=}'
        # TODO delete  this is now before the de-duping tho... (restore, after
        # de-duping?)
        #assert n_claws == n_claw_bouton_combos, ('assuming we have dropped or '
        #    'de-duped (merged/split) any boutons, to eliminate claws connecting '
        #    'to multiple boutons'
        #)

        # number after all the filtering in connectome_wPNKC (i.e. dropping
        # non-olfactory glomeruli, dropping claws without sufficient synapses [which
        # may end up dropping boutons, if they are only paired with those claws],
        # etc). n_boutons=392
        # TODO delete one def? (or assert equal to redef below?)
        n_boutons = len(claw2bouton[bouton_cols].drop_duplicates())

        # TODO this is currently redefined above. delete one or the other?
        # already sorted on claw_cols in claw2bouton anyway
        n_boutons_per_claw = claw2bouton.groupby(claw_cols, sort=False).size()
        n_boutons_per_claw_counts = n_boutons_per_claw.value_counts()
        if (n_boutons_per_claw_counts.index > 1).any():
            # TODO TODO how to deal w/ these?
            # TODO TODO TODO to what extent is it explained by adjacent "boutons" of
            # same PN (seems like they are *all* the same PN, when there are
            # duplicates), which are perhaps split too aggressively? merge those?
            # 1    9526
            # 2     333
            # 3       8
            warn('some boutons with multiple claws! #-boutons-per-claw -> '
                f'#-claws:\n{n_boutons_per_claw_counts.to_string()}'
            )
        #

        multibouton_claw_counts = n_boutons_per_claw_counts[
            n_boutons_per_claw_counts.index > 1
        ]
        # 333*2 + 8*3 = 690
        n_multibouton_dupes = (
            multibouton_claw_counts.index.values * multibouton_claw_counts.values
        ).sum()

        # TODO + redef n_boutons/etc, if i end up merging here
        #
        # TODO TODO need to store all bouton IDs for merging in all datasets
        # later? or merge in all 3 here? (prob can't pre-merge/split in
        # connectome_wPNKC, unless storing all bouton IDs there, since we may need
        # all of those IDs to know which data to align here)
        multibouton = claw2bouton.set_index(claw_cols).loc[
            n_boutons_per_claw[n_boutons_per_claw > 1].index
        ]
        assert len(multibouton) == n_multibouton_dupes

        # as long as only one unique PN for each claw, even if given multiple bouton
        # IDs, should be fine to merge all of those bouton IDs together (as long as
        # we do it in a way that is consistent across PN>KC, PN>APL, and APL>PN)
        assert (multibouton.groupby(claw_cols)[PN_ID].unique().str.len() == 1
            ).all(), (
            'there actually were claws that receive input from multiple boutons, '
            'where not all boutons were from same PN'
        )

        multibouton = multibouton.reset_index()
        assert (
            len(multibouton[claw_cols].drop_duplicates()) ==
            len(multibouton[claw_cols + [PN_ID]].drop_duplicates())
        ), 'should be same as assertion above (only 1 PN per multibouton-claw)'
        assert (
            len(multibouton[claw_cols + [BOUTON_ID]].drop_duplicates()) ==
            len(multibouton)
        )

        # TODO delete assertion? prob don't need to be sorted for anything
        #assert claw2bouton[claw_cols + [BOUTON_ID]].equals(
        #    claw2bouton[claw_cols + [BOUTON_ID]].sort_values(claw_cols +[BOUTON_ID])
        #)
        # TODO also assert all other columns same, for a given bouton ID? should be,
        # no?

        # TODO TODO come up with some way to merge the duplicates instead?
        #
        # just dropping these (for now, at least), since Btn_to_pn is
        # only vector<int> in model code currently (w/ pn_to_Btns
        # vector<vector<int>>)? don't think i want to add complexity in C++ code if
        # i can avoid it...
        # TODO skip this if per_claw_pn_apl_weights=True (prob not. currently will give
        # counts > 1 in wPNKC, which might also not matter, but was unexecpected)?
        # matter?
        if len(multibouton) > 0:
            # TODO remove the 'in claws2bouton' part of message?
            warn('for each claw with more than one bouton (in above warning) will '
                'drop all but the first bouton for each claw! ideally we would instead '
                'merge these duplicate boutons (which, for each claw, are all from the '
                'same PN)'
            )
            claw2bouton = claw2bouton.drop_duplicates(subset=claw_cols)

        # TODO TODO do i only want to merge within a given claw? or OK to also
        # merge these bouton IDs in all contexts? what to check first? how many
        # other claws also connected to these multibouton things?
        # TODO TODO need to make sure i merge in a way that is consisent
        # across claws. e.g. if claw 1 has pn ID 1 w/ boutons [2,3] and claw 2 has
        # pn ID 1 w/ boutons [1,2], would currently get a diff min ID for each claw.
        # need to compute across claws, it seems. how??? possible? merge and split?
        # NOTE: bouton_cols value pairs in multibouton subset are also very much
        # present in the non-multibouton subset of claw2bouton
        # TODO TODO does one bouton ever go to two diff claws, where answer to
        # min ID would be different across any of the claws? yes, fuck:
        # ipdb> claw2bouton.groupby(bouton_cols).merged_bouton_id.unique()
        # pn_id       bouton_id
        # 542634818   0                  [0]
        #             1                  [1]
        #             2                  [2]
        #             3               [3, 0]
        #             4            [4, 3, 0]
        #                            ...
        # 5901222910  0                  [0]
        #             1               [1, 0]
        #             2                  [2]
        #             3                  [3]
        #             5               [5, 2]
        # TODO ...and i can't just take the min of above right? cause still
        # wouldn't guarntee we actually merge when we are supposed to having two
        # (pn, bouton) combos actually pointing to same (consistent) new ID, right?
        #
        # TODO TODO maybe i should get the set of (pn, bouton) IDs in
        # multibouton, and then find min bouton ID across all (pn, claw) combos
        # (in full claw2bouton) with any boutons in that set? work?
        # don't think i can just:
        # claw2bouton.groupby([PN_ID] + claw_cols)
        # ...though, b/c adding [PN_ID] there doesn't even change # of groups, so
        # wouldn't change output of below
        # TODO TODO TODO can i just ignore all this stuff? even matter after
        # agg-ing down to claw<>APL weights? (or will they be glom<>APL weights?
        # need to be claw, no [unless i want to change glomerulus dynamics, without
        # actually having separate PNs / boutons, which i'm not sure i really have
        # now]?)
        # TODO delete
        #claw2merged_bouton = claw2bouton.groupby(claw_cols)[BOUTON_ID].min()

        # TODO delete
        # could also exclude PN_ID from here. shouldn't matter
        #non_bouton_cols = [
        #    x for x in claw2bouton.columns if x not in claw_cols + bouton_cols
        #]

        # TODO delete. prob can't use this strategy anyway
        #claw2bouton = claw2bouton.set_index(claw_cols)
        ## NOTE:  temporarily assigning
        ## into column of different name, to make easier to make (pn, bouton) ->
        ## merged-bouton map. will eventually overwrite BOUTON_ID with these merged
        ## values.
        #claw2bouton[f'merged_{BOUTON_ID}'] = claw2bouton.index.map(
        #    claw2merged_bouton
        #)
        # TODO TODO also need a separate (pn, bouton) -> merged-bouton-id map
        # for the PN<>APL data? (and prob need to compute before merging
        # claw2bouton below)

        # TODO TODO is it expected that the # of duplicates only barely changes below?
        #
        # ipdb> claw2bouton.reset_index().set_index(bouton_cols
        #   ).merged_bouton_id.reset_index()[bouton_cols].drop_duplicates().shape
        # (392, 2)
        #
        # ipdb> claw2bouton.reset_index().set_index(bouton_cols
        #   ).merged_bouton_id.reset_index()[[PN_ID, 'merged_bouton_id']
        #   ].drop_duplicates().shape
        # (389, 2)

        if plot_dir is not None:
            n_claws_per_bouton = claw2bouton.groupby(bouton_cols).size()
            n_claws_per_bouton.name = 'n_claws_per_bouton'
            fig, ax = plt.subplots()
            sns.histplot(n_claws_per_bouton, ax=ax, discrete=True)
            ax.set_title('# claws per bouton')
            savefig(fig, plot_dir, 'n_claws_per_bouton_hist')

            # TODO TODO also plot bar graph of # boutons per glom, with gloms
            # labelled (and sorted). (change anything if i start w/ output that is just
            # length # boutons, rather than length # claws [as here]? should just be
            # careful with calculation, to drop dupes w/in claw
            n_boutons_per_glom = claw2bouton.groupby(glomerulus_col).apply(
                lambda x: len(x[bouton_cols].drop_duplicates())
            )
            # TODO check 54 matches # unique gloms later (i think it does?)
            assert len(n_boutons_per_glom) == 54

            # TODO matshow this instead? (as some of the plots in
            # connectome_APL_weights)
            n_gloms = len(n_boutons_per_glom)
            n_boutons_per_glom.name = 'n_boutons_per_glom'
            fig, ax = plt.subplots()
            # TODO TODO make sure xlabels are shown (make smaller, or plot larger, and
            # rotate 90 degrees)
            sns.barplot(n_boutons_per_glom, ax=ax)

            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels, rotation=90, fontsize=6)

            ax.set_title(f'# boutons per glomerulus\n({n_gloms} glomeruli)')
            savefig(fig, plot_dir, 'n_boutons_per_glom')

            # TODO TODO # claws per glom too? the point any different than # boutons per
            # glom? (and latter would just be a scaling factor on top of # boutons,
            # essentially, right? check?)
            # (or happy with plots i currently have? move those here b/c they
            # don't actually need the PN<>APL weights, right?)

            # TODO TODO summarize (in connectome_APL_weights) any gloms w/ 0 weight in
            # either PN<>APL direction?

        # TODO assert we can actually restore after inverting this, and check
        # that we understand the nature of any dupes added
        # (might just need to explode bouton_id in wPNKC in a step above, before
        # dropping dupes from it?)
        #
        # moving PN_ID (int) and BOUTON_ID (a tuple-of-ints) from last index levels to
        # column levels to new int (for BOUTON_ID too) column (after glomerulus), which
        # is has a separate column for each bouton
        row_per_claw_and_bouton = wPNKC.reset_index(level=BOUTON_ID).explode(BOUTON_ID
            ).set_index(BOUTON_ID, append=True)

        if len(multibouton) > 0:
            index_names = list(row_per_claw_and_bouton.index.names)
            row_per_claw_and_bouton = row_per_claw_and_bouton.reset_index().set_index(
                claw_cols
            )
            assert row_per_claw_and_bouton.index.duplicated().any()
            assert len(row_per_claw_and_bouton) > len(wPNKC)

            claw2bouton = claw2bouton.set_index(claw_cols)
            assert set(row_per_claw_and_bouton.index) == set(claw2bouton.index)
            assert not claw2bouton.index.duplicated().any()

            row_per_claw_and_bouton = row_per_claw_and_bouton.loc[
                # keeps first row for each set of duplicated values
                ~row_per_claw_and_bouton.index.duplicated()
            ]
            assert not row_per_claw_and_bouton.index.duplicated().any()
            assert len(row_per_claw_and_bouton) == len(wPNKC)
            row_per_claw_and_bouton = row_per_claw_and_bouton.reset_index().set_index(
                index_names
            )

        wPNKC3 = row_per_claw_and_bouton.unstack(bouton_cols).fillna(0)
        # TODO delete eventually
        # (no all NaN rows/cols before fillna(0), nor any pre-existing 0)
        #inverted = wPNKC3.replace(0, np.nan).stack(bouton_cols).fillna(0)
        # passes
        #assert row_per_claw_and_bouton.index.equals(inverted.index)
        #

        assert not (wPNKC3 == 0).T.all().any()
        wPNKC3 = wPNKC3.loc[:, ~(wPNKC3 == 0).all()].copy()

        n_boutons = len(row_per_claw_and_bouton.index.to_frame(index=False)[bouton_cols
            ].drop_duplicates()
        )
        assert len(wPNKC3.columns) == n_boutons

        # currently 392
        assert 350 < n_boutons < 600, f'{n_boutons=} seems inconsistent w/ # boutons'

        # if i didn't drop the duplicate boutons above, might not be true?
        assert not wPNKC3.index.duplicated().any()

        assert wPNKC.index.droplevel(bouton_cols).equals(wPNKC3.index)
        assert not wPNKC3.columns.duplicated().any()
        # TODO assert # columns in correct range for # boutons? >300 and < 600?

        # TODO renumber boutons so they are unique per glomerulus (instead of
        # unique per PN), consistent w/ tianpei's case? or just keep the PN_ID level?
        # (no actually, as that would make aligning in connectome_APL_weights (to
        # the APL<>PN stuff impossible)

        # TODO maybe convert dtype back to int?

        # TODO move this assertion to all cases?
        assert set(wPNKC3.T.sum().values) == {1.0}

        wPNKC = wPNKC3

    # TODO add assertion (for both tianpei case and mine) that no bouton
    # (columns) have all 0 values? (currently doing in prat_boutons=True case above)

    # TODO option to also return a version of (long-form) df, so i can use for a test
    # comparing old + new hemibrain version, as i'm currently doing w/ sloppy code
    # above?
    return wPNKC


# TODO move to hong2p.util
def get_param_subset(kwargs: ParamDict, fn: Callable) -> ParamDict:
    """Returns subset of `kwargs` matching argument names in `callable`
    """
    sig = inspect.signature(fn)
    param_names = set(sig.parameters)
    kwarg_subset = {k: v for k, v in kwargs.items() if k in param_names}
    return kwarg_subset


# TODO keep (/move to test code?)? only currently can get use of this in
# test_connectome_wPNKC_repro, b/c in fit_mb_model, params are all split out into
# separate names, rather than condensed in one kwargs dict. would need a further layer
# of magic to condense back to kwargs...
def get_connectome_wPNKC_params(kwargs: ParamDict) -> ParamDict:
    """Takes params as `fit_mb_model`, returns subset in `connectome_wPNKC` arguments.

    Should also work with params serialized to 'params.csv' under each output dir, if
    they are deseralized with `read_series_csv(<path>, convert_dtypes=True)`.

    If there were ever differences in default values between `fit_mb_model` and
    `connectome_wPNKC`, for any parameter of one name, using params this function
    returns may then not reproduce `connectome_wPNKC` behavior (as it would be when
    called from within `fit_mb_model`). Also true if `fit_mb_model` were to change any
    value away from this matching default before `connectome_wPNKC` call.
    """
    # TODO also assert defaults same for params shared between fit_mb_model and
    # connectome_wPNKC, to avoid issue mentioned at end of doc above?
    # (still have issue that values could change before connectome_wPNKC call, in
    # fit_mb_model...)

    # TODO actually want to switch anything based on one_row_per_claw (which will
    # not be in wPNKC kwargs below)? prob fine leaving that validation to other things
    wPNKC_kwargs = get_param_subset(kwargs, connectome_wPNKC)

    # TODO test that some less-used params (e.g. drop_kcs_with_no_input /
    # _drop_glom_with_plus / Btn_separate) still make it into params.csv (and thus
    # kwargs)? (or else might need to also inspect fit_mb_model defaults?). should be
    # fine.

    connectome = get_connectome(kwargs.get('pn2kc_connections'))
    assert 'connectome' not in wPNKC_kwargs, ('expected this specified as '
        'pn2kc_connections, in serialized params.csv contents (if present)'
    )
    wPNKC_kwargs['connectome'] = connectome
    return wPNKC_kwargs


# TODO use in connectome_wPNKC?
def fmt_frac(x: int, y: int, *, den: bool = True, dec: bool = True) -> str:
    msg = f'{x}'
    if den:
        msg += f'/{y}'
    if dec:
        msg += f' ({x/y:.3f})'
    return msg


# TODO TODO TODO add option to actually return outputs of length # of boutons (probably
# w/ index same as wPNKC.columns there), as one option in prat_boutons=True case
def connectome_APL_weights(connectome: str = 'hemibrain', *, prat_claws: bool = False,
    # TODO TODO TODO implement per_claw_pn_apl_weights=False
    prat_boutons: bool = False, per_claw_pn_apl_weights: bool = False,
    # TODO delete
    #pn_apl_scale_factor: float = 1.0,
    #
    wPNKC: Optional[pd.DataFrame] = None, kc_types: Optional[pd.Series] = None,
    kc_to_claws: Optional[List[List[int]]] = None, _drop_glom_with_plus: bool = True,
    plot_dir: Optional[Path] = None) -> Tuple[pd.Series, pd.Series, Optional[pd.Series],
    Optional[pd.Series]]:
    # TODO add param to support imputing something non-zero (min_weight?) for KC IDs in
    # wPNKC but not APL-KC data
    # TODO actually want to support weight_divisor (prob not, especially if i'm
    # scaling everything anyway? and prob wouldn't same to be same as for PN->KC data
    # anyway...)?
    # TODO param for min_weight? maybe add smilar one for connectome_wPNKC, and share
    # min_weight value across these two fns? (even if doesn't make sense to share
    # weight_divisor)?
    # TODO why do i need both wPNKC and kc_types? just always pass index (which may
    # or may not include kc_types, but don't otherwise pass wPNKC?)
    # TODO TODO doc shape of w[APLPN|PNAPL], and in case of each flag choice (if i
    # implement an actual separate pn/bouton one, instead of just APL>claw and claw>APL
    # (each determined by averaging over glomeruli in each claws boutons)
    # TODO TODO update doc for prat_boutons when done (+ add & doc flag to control
    # whether we return something of shape # claws vs shape # boutons)
    # TODO ever want something of shape # glomeruli or nah? prob not
    """Returns wAPLKC, wKCAPL, wAPLPN, wPNAPL. Each KC[+claw] index, # synapse values.

    wAPLPN and wPNAPL will be None unless `prat_boutons=True`.

    Args:
        wPNKC: dataframe of shape (# KCs, # glomeruli), with index being KC body IDs
            from same connectome as requested. If passed, KCs with IDs not in this index
            will be dropped from output, and KC IDs in `wPNKC.index` but not in APL-KC
            data will have weight set to 0.

            In one-row-per-claw cases (`prat_claws=True` or `kc_to_claws not None`),
            assumed that `wPNKC` has one row per-claw (and additional claw metadata in
            row index levels), instead of one row per-KC.

            Bouton information may also be in column index, instead of row index, and
            `prat_boutons=True` input is expected to have both `BOUTON_ID` and `PN_ID`
            in column index level names (on top of already expected `glomerulus_col`).

        prat_boutons: requires `prat_claws=True`, and `wPNKC` passed in to have bouton
            information in column index (from `prat_boutons=True` to `connectome_wPNKC`
            call that made it). See `per_claw_pn_apl_weights`.

        per_claw_pn_apl_weights: if True, will compute PN<>APL weights per claw
            (assigning bouton weight to all claws connected to it). Assumes the
            small number of claws erroneously labelled as connecting to multiple
            boutons have been dropped in `connectome_wPNKC`. `wAPLPN` and `wPNAPL`
            returned will be of length equal to # of claws (same row index as input
            `wPNKC`), collapsing over the bouton[/pn]/glomerulus columns of `wPNKC`.

            If False, will return `wAPLPN` and `wPNAPL` of length equal to # of bouton
            in `wPNKC`. There will be no per-claw metadata in these outputs.

            Ignored if `prat_boutons=False`.

        connectome: same as for `connectome_wPNKC`

        plot_dir: same as for `connectome_wPNKC`
    """
    assert connectome in connectome_options

    if connectome != 'hemibrain':
        raise NotImplementedError('currently only have data for APL-KC connections'
            " for connectome='hemibrain'"
        )

    # TODO work? (latter currently also non-None in prat_claws case, but not used. will
    # try to add a check that it makes sense in prat_claws=True case)
    one_row_per_claw = prat_claws or kc_to_claws is not None

    # TODO also trying splitting out by apl unit (include in claw_cols for agg_* fn,
    # rather than extra_cols_to_keep?)? (not unless we add an actual compartmented
    # apl implementation in C++ code)
    claw_cols = [KC_ID, CLAW_ID]

    wAPLPN = None
    wPNAPL = None
    if prat_boutons:
        assert prat_claws
        assert wPNKC is not None, ('need this to merge to get (kc, claw) IDs for '
            'PN<>APL data'
        )

    if prat_claws:
        # TODO delete (or combine into flag to load old v3 outputs)
        # TODO TODO was old v3 handling of APL weights maintaining correct
        # association of claws, despite renumbering claws [via cumcount in
        # connectome_wPNKC]? try removing all the renumbering (done), and check outputs
        # same?
        old_a2k = pd.read_parquet(prat_hemibrain_seg_v3_dir /
            'APL-to-KC_with_Synapses_v3.parquet'
        )
        old_k2a = pd.read_parquet(prat_hemibrain_seg_v3_dir /
            'KC-to_APL_with_Synapses_v3.parquet'
        )
        #

        apl2kc_data = (prat_hemibrain_seg_dir /
            # TODO check that it doesn't matter which of these i use. IDs should
            # have also been correct in 12-09-19 data
            'APLunit-to-KCclaw_Connectivity_2025-12-14-21-12-48.parquet'
            # TODO delete (check equiv? should be. i just misunderstood data i think,
            # and output shouldn't have changed)
            #'APLunit-to-KCclaw_Connectivity_2025-12-09-19-47-36.parquet'
        )
        kc2apl_data = (prat_hemibrain_seg_dir /
            # TODO check that it doesn't matter which of these i use. IDs should
            # have also been correct in 12-09-19 data
            'KCclaw-to-APLunit_Connectivity_2025-12-14-21-12-48.parquet'
            # TODO delete (check equiv? should be. i just misunderstood data i think,
            # and output shouldn't have changed)
            #'KCclaw-to-APLunit_Connectivity_2025-12-09-19-47-36.parquet'
        )

        apl2pn_data = (prat_hemibrain_seg_dir /
            'APLunit-to-PNbouton_Connectivity_2025-12-05-13-39-22.parquet'
        )
        pn2apl_data = (prat_hemibrain_seg_dir /
            'PNbouton-to-APLunit_Connectivity_2025-12-05-13-39-22.parquet'
        )

        # v5 notes:
        # ipdb> apl2kc_df.dtypes
        # bodyId_pre                     int64
        # bodyId_post                    int64
        # roi_pre                       object
        # roi_post                      object
        # x_pre                          int32
        # y_pre                          int32
        # z_pre                          int32
        # x_post                         int32
        # y_post                         int32
        # z_post                         int32
        # confidence_pre               float32
        # confidence_post              float32
        # instance_pre                  object
        # type_pre                      object
        # instance_post                 object
        # type_post                     object
        # anatomical_claw_corrected    float64
        # anatomical_unit              float64
        # dtype: object
        # ipdb> kc2apl_df.dtypes
        # bodyId_pre                     int64
        # bodyId_post                    int64
        # roi_pre                       object
        # roi_post                      object
        # x_pre                          int32
        # y_pre                          int32
        # z_pre                          int32
        # x_post                         int32
        # y_post                         int32
        # z_post                         int32
        # confidence_pre               float32
        # confidence_post              float32
        # instance_pre                  object
        # type_pre                      object
        # instance_post                 object
        # type_post                     object
        # anatomical_claw_corrected    float64
        # anatomical_unit              float64
        #
        # ipdb> apl2kc_df.isna().sum().replace(0, np.nan).dropna()
        # roi_pre                        423.0
        # roi_post                       434.0
        # type_post                      787.0
        # anatomical_claw_corrected    65220.0
        # anatomical_unit              65075.0
        # dtype: float64
        # ipdb> kc2apl_df.isna().sum().replace(0, np.nan).dropna()
        # roi_pre                        252.0
        # roi_post                       249.0
        # type_pre                       933.0
        # anatomical_claw_corrected    76727.0
        # anatomical_unit              76600.0
        apl2kc_df = pd.read_parquet(apl2kc_data)
        kc2apl_df = pd.read_parquet(kc2apl_data)
        assert set(apl2kc_df.columns) == set(kc2apl_df.columns)

        apl2pn_df = pd.read_parquet(apl2pn_data)
        pn2apl_df = pd.read_parquet(pn2apl_data)
        assert set(apl2pn_df.columns) == set(pn2apl_df.columns)

        # TODO add assertion about how pn<>apl and kc<>apl columns differ?

        # v3 outputs seemed prefiltered on these (roi_[pre|post]), but v5 not
        # TODO assert assert_some_dropped=True to all calls then? (or delete if causes
        # issue)
        apl2kc_df = filter_synapses_to_roi(apl2kc_df, 'CA(R)', assert_some_dropped=True)
        kc2apl_df = filter_synapses_to_roi(kc2apl_df, 'CA(R)', assert_some_dropped=True)

        apl2pn_df = filter_synapses_to_roi(apl2pn_df, 'CA(R)', assert_some_dropped=True)
        pn2apl_df = filter_synapses_to_roi(pn2apl_df, 'CA(R)', assert_some_dropped=True)

        assert (apl2kc_df.instance_pre == 'APL_R').all()
        assert (kc2apl_df.instance_post == 'APL_R').all()
        assert (apl2kc_df.type_pre == 'APL').all()
        assert (kc2apl_df.type_post == 'APL').all()

        assert (apl2pn_df.instance_pre == 'APL_R').all()
        assert (pn2apl_df.instance_post == 'APL_R').all()
        assert (apl2pn_df.type_pre == 'APL').all()
        assert (pn2apl_df.type_post == 'APL').all()

        assert not apl2kc_df.instance_post.isna().any()
        assert not kc2apl_df.instance_pre.isna().any()
        apl2kc_df = add_kc_type_col(apl2kc_df, 'instance_post')
        kc2apl_df = add_kc_type_col(kc2apl_df, 'instance_pre')

        # TODO TODO double check how KC type col is used in these cols, and that it also
        # makes sense for APL handling
        # TODO TODO cache current connectome_wPNKC output (and it's internal plots?
        # check them w/ -c/-C? care?), and check moving this to add_glom* doesn't change
        # anything
        apl2pn_df, _ = add_glomerulus_col_from_hemibrain_type(apl2pn_df, 'type_post',
            # TODO TODO 'bodyId_pre' (which is APL, not KC) work for kc_type_col
            # here? adapt / refactor, if not?
            'bodyId_pre', check_no_multi_underscores=True,
            _drop_glom_with_plus=_drop_glom_with_plus
            # TODO also want drop_kcs_with_no_input=drop_kcs_with_no_input here?
        )
        pn2apl_df, _ = add_glomerulus_col_from_hemibrain_type(pn2apl_df, 'type_pre',
            # TODO TODO 'bodyId_post' (which is APL, not KC) work for kc_type_col
            # here? adapt / refactor, if not?
            'bodyId_post', check_no_multi_underscores=True,
            _drop_glom_with_plus=_drop_glom_with_plus
            # TODO also want drop_kcs_with_no_input=drop_kcs_with_no_input here?
        )

        # TODO TODO if prat_boutons (or even if we want to use PN data at all, probably,
        # which will still require merging w/ [pn_id, bouton_id], assert we have wPNKC
        # with the bouton col (and that the APL<>PN data has the same bouton col)

        # TODO refactor to share (at least v5_claw_col='anatomical_claw_corrected' def)
        # w/ wPNKC loading
        v5_claw_col = 'anatomical_claw_corrected'
        v3_claw_col = 'KC_anatomical_claw'
        v5_apl_unit_col = 'anatomical_unit'
        v3_apl_unit_col = 'APL_anatomical_unit'
        if v5_claw_col in apl2kc_df.columns:
            assert v5_claw_col in kc2apl_df.columns
            assert v5_apl_unit_col in apl2kc_df.columns
            assert v5_apl_unit_col in kc2apl_df.columns

            # TODO assert bouton + PN ID cols are are also in these dfs
            # (though PN ID would be a diff col in each of these here, until renamed to
            # be consistent. pre in one, post in other. TODO rename to be consistent at
            # this point?)
            assert v5_apl_unit_col in apl2pn_df.columns
            assert v5_apl_unit_col in pn2apl_df.columns

            # TODO TODO TODO will need a flag to differentiate APL>PN being done in
            # terms of claws (counting all synapses onto a [pn_id, bouton_id] towards
            # each claw that receives input from that bouton), or truly onto boutons
            # with separate dynamics (not yet implemented in model, or at least haven't
            # really checked tianpei's code there + would probably need to heavily
            # change it)
            # TODO TODO should prat_boutons be that flag? if so, will need
            # connectome_wPNKC to return bouton_id regardless of prat_boutons, so we can
            # still use PN data here (and then would need separate flag to indicate
            # whether there should be direct PN<>APL interactions, whether thru claws or
            # thru actual boutons)

            # bouton_ids are not globally unique, just unique within PN

            assert v3_claw_col not in apl2kc_df.columns
            assert v3_claw_col not in kc2apl_df.columns
            assert v3_apl_unit_col not in apl2kc_df.columns
            assert v3_apl_unit_col not in kc2apl_df.columns

            orig_claw_col = v5_claw_col
            apl_unit_col = v5_apl_unit_col
            drop_negative_claw_id = False
        else:
            assert v3_claw_col in apl2kc_df.columns
            assert v3_claw_col in kc2apl_df.columns
            assert v3_apl_unit_col in apl2kc_df.columns
            assert v3_apl_unit_col in kc2apl_df.columns
            orig_claw_col = v3_claw_col
            apl_unit_col = v3_apl_unit_col
            drop_negative_claw_id = True

        apl2kc_df = apl2kc_df.rename(columns={
            'bodyId_pre': 'apl_id',
            'bodyId_post': KC_ID,
            orig_claw_col: CLAW_ID,
        })
        kc2apl_df = kc2apl_df.rename(columns={
            'bodyId_pre': KC_ID,
            'bodyId_post': 'apl_id',
            orig_claw_col: CLAW_ID,
        })

        # TODO refactor
        assert all(KC_ID in x.columns for x in [apl2kc_df, kc2apl_df])
        assert all(CLAW_ID in x.columns for x in [apl2kc_df, kc2apl_df])
        assert not bouton_col in kc2apl_df.columns
        assert not bouton_col in apl2kc_df.columns
        assert not apl2kc_df.apl_id.isna().any() and apl2kc_df.apl_id.nunique() == 1
        assert not kc2apl_df.apl_id.isna().any() and kc2apl_df.apl_id.nunique() == 1

        if prat_boutons:
            assert orig_claw_col not in apl2pn_df.columns
            assert orig_claw_col not in pn2apl_df.columns
            apl2pn_df = apl2pn_df.rename(columns={
                'bodyId_pre': 'apl_id',
                'bodyId_post': PN_ID,
                bouton_col: BOUTON_ID,
            })
            pn2apl_df = pn2apl_df.rename(columns={
                'bodyId_pre': PN_ID,
                'bodyId_post': 'apl_id',
                bouton_col: BOUTON_ID,
            })
            # TODO refactor each to one line? (if not refactoring more generally)
            assert all(BOUTON_ID in x.columns for x in [apl2pn_df, pn2apl_df])
            assert all(PN_ID in x.columns for x in [apl2pn_df, pn2apl_df])
            #
            assert not apl2pn_df.apl_id.isna().any() and apl2pn_df.apl_id.nunique() == 1
            assert not pn2apl_df.apl_id.isna().any() and pn2apl_df.apl_id.nunique() == 1

        # TODO refactor to share w/ wPNKC handling?
        def drop_missing_claws(df: pd.DataFrame, name: str,
            # TODO delete 3rd arg? ever gonna be something other than [KC_ID] now?
            # what about for PN<>APL stuff?
            count_dropped_syns_within: Optional[List[str]] = None) -> pd.DataFrame:

            # TODO handle some more elegant way, w/o copying first?
            orig = df.copy()

            claw_isna = df[CLAW_ID].isna()
            if claw_isna.any():
                warn(f'{name}: dropping {claw_isna.sum()} (/{len(df)}) synapses with '
                    'NaN claws'
                )
                df = df[~claw_isna].copy()

            # TODO delete comment. shouldn't need to handle here, after already
            # filtering in wPNKC, and then filtering to that here
            # (so long as wPNKC filtering below also contributes to non-claw synapses
            # counted here, within each KC)
            #
            # still consider as invalid (even within one KC), if not >= (>?) 5 synapses
            # FROM ONE PN (according to Prat). he's currently just assuming there won't
            # be that much collective input (across distributed spine synapses) from one
            # PN, but we will have to check that (and how?)
            if drop_negative_claw_id:
                claw_unassigned = df[CLAW_ID] == -1
                assert claw_unassigned.any()
                warn(f'{name}: dropping {claw_unassigned.sum()} (/{len(df)}) synapses '
                    'with unassigned claws'
                )
                df = df[~claw_unassigned].copy()

            int_claws = df[CLAW_ID].astype(int)
            assert pd_allclose(df[CLAW_ID], int_claws)
            df[CLAW_ID] = int_claws

            # TODO want this to be optional? also want for both pn<>apl things?
            if count_dropped_syns_within is None:
                return df

            # TODO TODO maybe just don't groupby->size()? or return raw subset as well?
            # (if i still want to do some per-synapse analysis on this dropped stuff)
            n_dropped_syns_per_kc = orig.loc[orig.index.difference(df.index)
                ].groupby(count_dropped_syns_within).size()

            # TODO TODO TODO and maybe factor wPNKC filtering into here too (or have
            # each return a similarly-indexed series, then add them?) (would prob prefer
            # the latter)
            return df, n_dropped_syns_per_kc

        # TODO TODO (done, at least on a per-kc basis) make sure we are still keeping
        # track of missing claws (ID -1 or something not in wPNKC, and lump all together
        # (no matter whether it was b/c -1 or just something not in wPNKC), per KC [or
        # per PN] for each cell type).
        # TODO TODO also keep track of all the synapses, for analysis not just on a
        # per-kc basis (e.g. radius of them)
        # TODO still need to pass 3rd arg, now that always [KC_ID]? move def into fn +
        # delete?
        apl2kc_df, n_nonclaw_apl2kc_syns_per_kc = drop_missing_claws(apl2kc_df,
            'apl2kc', [KC_ID]
        )
        kc2apl_df, n_nonclaw_kc2apl_syns_per_kc = drop_missing_claws(kc2apl_df,
            'kc2apl', [KC_ID]
        )

        # TODO check that tianpei's path thru here currently does something sensible in
        # Btn_separate=True (prat_claws=False) case too (matters less)
        if prat_boutons:
            # so any NaNs added by merging
            assert all(x.notna().all().all() for x in [apl2pn_df, pn2apl_df])

            if per_claw_pn_apl_weights:
                # TODO delete path we are not planning to support eventually (-> just
                # make this one assertion)
                if all(x in wPNKC.columns.names for x in bouton_cols):
                    assert not any(x in wPNKC.index.names for x in bouton_cols)
                    # TODO assert index not duplicated on claw_cols before next line
                    # (to just clarify it's that operation that is adding them [back? i
                    # suppose not clear it's same dupes as before? or is it?])
                    # TODO delete
                    #orig = wPNKC.copy()
                    #

                    # replacing 0 w/ NaN seems to make stack significantly faster (very
                    # slow w/ bouton_cols otherwise, and many rows in output), and gives
                    # us far fewer rows in initial output. output does not have any all
                    # NaN row/columns
                    # TODO not sure we actually get anything out of the fillna(0), in
                    # terms of use of wPNKC in the remainder of this fn...
                    wPNKC = wPNKC.replace(0, np.nan).stack(bouton_cols).fillna(0)

                    # TODO delete
                    # TODO (no longer true) wish i could figure out how to make this not
                    # happen, or at least to explain why it is unavoidable
                    #assert len(wPNKC.index) > len(orig.index)
                    #

                    assert not (wPNKC == 0).all().any()
                    assert not (wPNKC == 0).T.all().any()

                    index_names = wPNKC.index.names
                    # TODO TODO TODO need to check this is same as at similar step in
                    # connectome_wPNKC? any way to avoid re-introducing these dupes in
                    # the first place?
                    wPNKC = wPNKC.reset_index().drop_duplicates(subset=claw_cols
                        ).set_index(index_names)

                # stack() is to get a glomerulus index level from columns (which gives
                # us a Series with all 1 for values, that we don't need, so just use the
                # index from there)
                claw2bouton = wPNKC.replace(0, np.nan).stack().index.to_frame(
                    index=False).set_index(claw_cols).explode(BOUTON_ID).reset_index()

                # TODO TODO remove as much of claw2bouton stuff below as i can. no
                # longer need most (now that it's all duped to, and may only be needed
                # in connectome_wPNKC)

                # TODO maybe drop -1 (+ NaN, if any) BOUTON_ID first (from apl<>pn
                # stuff!), before merge?  and then also merge in a way that doesn't add
                # NaNs (if that doesn't make things harder)? just if it helps clarify
                # overall # of synapses before vs after
                # (should be asserting all id_cols *in wPNKC.index*, which include
                # BOUTON_ID, are not NaN here, if not already established)

                id_cols = [KC_ID, CLAW_ID, PN_ID, BOUTON_ID]
                assert not claw2bouton[id_cols].duplicated().any()

                assert claw2bouton[BOUTON_ID].min() == 0, ('wPNKC index should already '
                    'have had invalid (-1) bouton IDs dropped'
                )

                # checking that our glomerulus column->index transformation (as part of
                # claw2bouton def above) didn't screw anything up
                assert_one_glom_per_pn(claw2bouton)
                claw2bouton2 = wPNKC.index.to_frame(index=False).explode(BOUTON_ID)
                # TODO also assert not duplicated on claw_cols either now?
                # (would fail if i wasn't dropping cases of n-boutons:1-claw earlier now
                # [in connectome_wPNKC], unless i also added handling to dedupe them)
                assert not claw2bouton[id_cols].duplicated().any()
                assert len(claw2bouton) == len(claw2bouton2)
                del claw2bouton2

                # currently 10216 (update. prob no longer true after dropping
                # multibouton)
                n_claw_bouton_combos = len(claw2bouton)

                # 9867
                n_claws = len(wPNKC)
                # NOTE: if claws that were associated with multiple boutons (either 2 or
                # 3) had been dropped / split by this  point, would have to change this
                # assertion to equality
                #assert n_claws < n_claw_bouton_combos, \
                #    f'{n_claws=} >= {n_claw_bouton_combos=}'
                assert n_claws == n_claw_bouton_combos, ('assuming we have dropped or '
                    'de-duped (merged/split) any boutons, to eliminate claws connecting'
                    ' to multiple boutons'
                )

                # number after all the filtering in connectome_wPNKC (i.e. dropping
                # non-olfactory glomeruli, dropping claws without sufficient synapses
                # [which may end up dropping boutons, if they are only paired with those
                # claws], etc). n_boutons=392
                n_boutons = len(claw2bouton[bouton_cols].drop_duplicates())

                # TODO assert merging w/ this same as merging pn2apl_df (below, and
                # potentially needing to dropna after merge)? not worth?
                only_assigned_bouton_pn2apl = pn2apl_df[pn2apl_df[BOUTON_ID] != -1]
                only_assigned_bouton_apl2pn = apl2pn_df[apl2pn_df[BOUTON_ID] != -1]

                if plot_dir is not None:

                    # TODO plot this (in connectome_wPNKC, if not already plotted there)
                    #
                    # NOTE: prior filtering is mostly done on a claw basis (requiring
                    # min # of synapses for each), but should also end up dropping some
                    # boutons
                    wPNKC_boutons_per_glom = claw2bouton.groupby(glomerulus_col).apply(
                        lambda x: len(x[bouton_cols].drop_duplicates())
                    ).rename('# (filtered) PN>KC boutons')

                    # NOTE: these are also dropped in connectome_wPNKC, so would also be
                    # filtered at step where we intersect with bouton_cols values in
                    # there
                    unassigned_bouton_pn2apl = pn2apl_df[pn2apl_df[BOUTON_ID] == -1]
                    unassigned_bouton_apl2pn = apl2pn_df[apl2pn_df[BOUTON_ID] == -1]

                    for pn2apl, apl2pn, title_suffix, fname_suffix in (
                            # TODO want to keep all 3 of these? (at least filtered and
                            # one other...)
                            (pn2apl_df, apl2pn_df, '', ''),
                            (
                                unassigned_bouton_pn2apl,
                                unassigned_bouton_apl2pn,
                                ('\nunassigned boutons IDs (-1) only (dropped later)\n'
                                 f'{len(unassigned_bouton_apl2pn)}/{len(apl2pn_df)} '
                                 'APL>PN synapses\n'
                                 f'{len(unassigned_bouton_pn2apl)}/{len(pn2apl_df)} '
                                 'PN>APL synapses'
                                ),
                                '_only-invalid-bouton-ids'
                            ),

                            # since we assert above that claw2bouton has no -1 bouton
                            # IDs, all of these will be absent from wPNKC index (and
                            # thus from claw2bouton), although there are probably also
                            # other boutons missing for other reasons
                            (
                                only_assigned_bouton_pn2apl,
                                only_assigned_bouton_apl2pn,
                                ('\nunassigned boutons IDs (-1) dropped\n'
                                # TODO be more clear '...remaining'?
                                 f'{len(only_assigned_bouton_apl2pn)}/{len(apl2pn_df)} '
                                 'APL>PN synapses\n'
                                 f'{len(only_assigned_bouton_pn2apl)}/{len(pn2apl_df)} '
                                 'PN>APL synapses'
                                ),
                                '_dropped-invalid-bouton-ids'
                            ),
                        ):

                        cbar_shrink = 0.2

                        # TODO refactor this method of getting # boutons? (also used
                        # above, and prob below)
                        pnapl_boutons_per_glom = pn2apl.groupby(glomerulus_col).apply(
                            lambda x: len(x[bouton_cols].drop_duplicates())
                        ).rename('# PN>APL boutons')
                        aplpn_boutons_per_glom = apl2pn.groupby(glomerulus_col).apply(
                            lambda x: len(x[bouton_cols].drop_duplicates())
                        ).rename('# APL>PN boutons')
                        apl_and_pn_boutons_per_glom = pd.concat(
                            [pnapl_boutons_per_glom, aplpn_boutons_per_glom],
                            axis='columns'
                        )

                        # TODO also move this plotting after loop (along w/ plots in
                        # similar conditional below?)?
                        if fname_suffix == '_dropped-invalid-bouton-ids':
                            apl_and_pn_boutons_per_glom = pd.concat(
                                [apl_and_pn_boutons_per_glom, wPNKC_boutons_per_glom],
                                axis='columns'
                            )

                        fig, ax = plt.subplots()
                        viz.matshow(apl_and_pn_boutons_per_glom.T, ax=ax,
                            cbar_label='# boutons', cbar_shrink=cbar_shrink
                        )
                        # TODO TODO also include # total boutons in title (for each
                        # connection type?)
                        if fname_suffix != '_dropped-invalid-bouton-ids':
                            ax.set_title('total raw PN<>APL boutons per glomerulus'
                                f'\ntop: PN>APL, bottom: APL>PN{title_suffix}'
                            )
                        else:
                            ax.set_title('total raw PN<>APL boutons per glomerulus'
                                '\ntop: PN>APL, middle: APL>PN\n'
                                f'bottom: (filtered) PN>KC{title_suffix}'
                            )
                            # TODO also change fname here?

                        # TODO replace version below (similar name) w/ this?
                        # then remove '_raw' here?
                        savefig(fig, plot_dir,
                            f'pn_and_apl_boutons_per_glom_raw{fname_suffix}',
                            bbox_inches='tight'
                        )

                        pnapl_syns_per_glom = pn2apl.groupby(glomerulus_col).size(
                            ).rename('# PN>APL synapses')
                        aplpn_syns_per_glom = apl2pn.groupby(glomerulus_col).size(
                            ).rename('# APL>PN synapses')
                        apl_and_pn_syns_per_glom = pd.concat(
                            [pnapl_syns_per_glom, aplpn_syns_per_glom], axis='columns'
                        )

                        fig, ax = plt.subplots()
                        # TODO TODO add row for (filtered) wPNKC stuff? (could only do #
                        # synapses here, if wPNKC had n_synapses for values [or another
                        # column/row index level, which i also processed]). currently
                        # wPNKC just contains 0/1 in values, and # synapse info is lost
                        # (though we do know each claw has at least 5 synapses from PN)
                        viz.matshow(apl_and_pn_syns_per_glom.T, ax=ax,
                            cbar_label='# synapses', cbar_shrink=cbar_shrink
                        )
                        ax.set_title('total raw PN<>APL synapses per glomerulus'
                            f'\ntop: PN>APL, bottom: APL>PN{title_suffix}'
                        )
                        # TODO replace version (far) below (similar name) w/ this?
                        # then remove '_raw' here?
                        savefig(fig, plot_dir,
                            f'pn_and_apl_syns_per_glom_raw{fname_suffix}',
                            bbox_inches='tight'
                        )

                        # TODO move all this after loop? (just using
                        # only_assigned_bouton_apl2pn)
                        if fname_suffix == '_dropped-invalid-bouton-ids':
                            # TODO (minor) combine these two into one call, w/ column
                            # for two facets? (so title_suffix referring to weights in
                            # both directions actually makes sense)
                            fig, ax = plt.subplots()
                            aplpn_synapses_per_bouton = apl2pn.groupby(bouton_cols
                                ).size()
                            assert aplpn_synapses_per_bouton.sum() == len(apl2pn)
                            sns.histplot(aplpn_synapses_per_bouton, ax=ax,
                                discrete=True
                            )
                            ax.set_title('# APL>PN synapses per bouton\n'
                                f'(across all glomeruli){title_suffix}'
                            )
                            savefig(fig, plot_dir,
                                f'wAPLPN_syns_per_bouton_hist{fname_suffix}'
                            )

                            fig, ax = plt.subplots()
                            pnapl_synapses_per_bouton = pn2apl.groupby(bouton_cols
                                ).size()
                            assert pnapl_synapses_per_bouton.sum() == len(pn2apl)
                            sns.histplot(pnapl_synapses_per_bouton, ax=ax,
                                discrete=True
                            )
                            ax.set_title('# PN>APL synapses per bouton\n'
                                f'(across all glomeruli){title_suffix}'
                            )
                            savefig(fig, plot_dir,
                                f'wPNAPL_syns_per_bouton_hist{fname_suffix}'
                            )

                            aplpn_synapses_per_bouton_and_glom = apl2pn.groupby(
                                bouton_cols + [glomerulus_col]).size(
                                ).rename('n_synapses').reset_index()
                            assert (len(apl2pn) ==
                                aplpn_synapses_per_bouton_and_glom.n_synapses.sum()
                            )
                            # NOTE: by default displot will share x and y (and seem to
                            # need to use facet_kws to disable. not super easy.)
                            g = sns.displot(aplpn_synapses_per_bouton_and_glom,
                                discrete=True, x='n_synapses', col=glomerulus_col,
                                col_wrap=7
                            )
                            # 15 min that could be OK for this font size
                            g.set_titles('{col_name}', size=17)
                            # y=1.05 a bit more than needed (at least, w/ default
                            # fontsize, which should be increased tbh. also a bit of
                            # space w/ size=15)
                            g.fig.suptitle('# APL>PN synapses per bouton\n'
                                f'(within each glomerulus){title_suffix}', y=1.05,
                                size=17
                            )
                            savefig(g, plot_dir,
                                f'wAPLPN_syns_per_bouton_by-glom_hists{fname_suffix}'
                            )

                            pnapl_synapses_per_bouton_and_glom = pn2apl.groupby(
                                bouton_cols + [glomerulus_col]).size(
                                ).rename('n_synapses').reset_index()
                            assert (len(pn2apl) ==
                                pnapl_synapses_per_bouton_and_glom.n_synapses.sum()
                            )
                            g = sns.displot(pnapl_synapses_per_bouton_and_glom,
                                discrete=True, x='n_synapses', col=glomerulus_col,
                                col_wrap=7
                            )
                            # TODO also increase font size of xlabel
                            g.set_titles('{col_name}', size=17)
                            g.fig.suptitle('# PN>APL synapses per bouton\n'
                                f'(within each glomerulus){title_suffix}', y=1.05,
                                size=17
                            )
                            savefig(g, plot_dir,
                                f'wPNAPL_syns_per_bouton_by-glom_hists{fname_suffix}'
                            )

                        # TODO delete
                        # TODO matshow where there is one row per bouton (value # of
                        # synapses, maybe using clustermap, or similar wrapper code to
                        # what i do to plot claw activities in
                        # natmix_data/analysis.py?)? (prob fine to not, if i can show
                        # fraction / count of zero entries)
                        # TODO maybe also show a row/col color (to side, like clustermap
                        # can) for those w/ invalid claw ID / bouton ID / missing (PN,
                        # bouton) from wPNKC?
                        #

                # TODO keep track of these -1's, so that we see them dropped when
                # reindexing later? (prob gonna be dropped as NaN in step right after
                # merge actually...) (should end up being counted same as a non-claw
                # synapse, no?) how to do that?
                #
                # (these are both preserved thru current how='left' merge)
                # ipdb> (apl2pn_df[BOUTON_ID] == -1).sum()
                # 358
                # ipdb> (pn2apl_df[BOUTON_ID] == -1).sum()
                # 203

                # TODO TODO have we already asserted n:1 / 1:n only in direction we
                # expect (between claws and boutons?) do so here, if not. or again, to
                # check (would fail, as-is, b/c of cases w/ 2 or 3 boutons for one claw.
                # still need to figure out how to handle those, and their nature)
                # TODO could postprocess to remove those dupes (splitting or dropping),
                # and then assert?

                # TODO delete. just to sanity check merging.
                premerge_apl2pn = apl2pn_df.copy()
                premerge_pn2apl = pn2apl_df.copy()
                print()
                print(f'{premerge_apl2pn.shape=}')
                print(f'{premerge_pn2apl.shape=}')
                print(f'{claw2bouton.shape=}')
                print()
                #
                # ipdb> len(claw2bouton)
                # 10216
                # ipdb> len(wPNKC)
                #
                # ipdb> claw2bouton[[KC_ID, CLAW_ID]].duplicated().sum()
                # 349
                # ipdb> 333 + 8
                # 341
                # ipdb> len(claw2bouton)
                # 10216
                # ipdb> len(wPNKC)
                # 9867
                # ipdb> len(claw2bouton) - len(wPNKC)
                # 349
                # 9867
                #
                # 349 is explained by the (somewhat rare) n-boutons:1-claw cases warned
                # about below in this fn (333 w/ count 2 [1 dupe], 8 w/ count 3 [2
                # dupes].  333 + 8*2 = 349

                # TODO TODO count # synapses before merge (compare w/ # synapses after)?
                # TODO TODO and check against result dividing by claws per KC?
                # (should only be off by maybe the dupes?)
                #
                # TODO TODO TODO and also just manually inspect a few boutons/claws that
                # cover all the cases (n:1, 1:n, and that should be it, right? missing
                # in left? right?)
                # TODO TODO + reason about how many there should be (+compute that in
                # advance separately), and assert we have that many rows/whatever after

                # TODO if each claw only had one bouton, would that simplify merge?
                # how= only matter for the small number of duplicate cases?

                # TODO maybe i should try merging after agg_synapses_to_claws (w/in
                # bouton_cols, for PN<>APL data)? (check equiv to current output, if so)
                # not sure it should matter when, and would be easier to think about and
                # prob faster
                #
                # TODO (delete?) want how='right' or (default) how='inner'? inner prob
                # fine
                #
                # need how='left' actually? or how else will i drop (+ count non-claw
                # synapses, more importantly) stuff not in wPNKC later? handle that
                # here, for these two? well, won't have KC ID, so could only count
                # synapses missing claws per PN (/bouton) or total per APL (or per APL
                # unit, if we cared about that) (at least drop_missing_claws should
                # report that then?)
                # TODO skip this step (done), and related dropping of claws w/ multiple
                # boutons (not done, and not sure i want), if
                # per_claw_pn_apl_weights=False?
                apl2pn_df = apl2pn_df.merge(claw2bouton, on=bouton_cols, how='left')
                pn2apl_df = pn2apl_df.merge(claw2bouton, on=bouton_cols, how='left')
                # NOTE: merge adds 206 NaN (kc_id, claw_id) in pn2apl data, and 366 in
                # apl2pn data (w/ how='left').
                # TODO want diff how=? matter? should be dropped in step below anyway...

                for df in [apl2pn_df, pn2apl_df]:
                    # NOTE: should be no other columns that were in both inputs (and
                    # thus no others w/ '_x' / '_y' suffices here) (except 'claw_x',
                    # 'claw_y' which were already in input, that is currently true)
                    g1 = df[df.glomerulus_y.notna()].glomerulus_x
                    g2 = df[df.glomerulus_y.notna()].glomerulus_y
                    assert g1.equals(g2)

                apl2pn_df = apl2pn_df.drop(columns='glomerulus_y').rename(columns={
                    'glomerulus_x': glomerulus_col
                })
                pn2apl_df = pn2apl_df.drop(columns='glomerulus_y').rename(columns={
                    'glomerulus_x': glomerulus_col
                })

                # TODO delete? no reason to think should fail here really
                assert_one_glom_per_pn(apl2pn_df)
                assert_one_glom_per_pn(pn2apl_df)
                #

                # TODO delete
                print(f'{pn2apl_df.shape=}')
                print(f'{apl2pn_df.shape=}')
                print()
                #print(f'{pn2apl_df.isna().sum()=}')
                #print(f'{apl2pn_df.isna().sum()=}')
                #

                # TODO TODO check that size after merging is at least as much larger as
                # we expect (at least # boutons * # claws before [at least for
                # overlapping set of claw_cols]. could be slightly higher from the
                # current existance of a few claws associated w/ multiple boutons.
                #
                # TODO want how='cross'? (prob not?)
                # TODO double check that (after this merging), we have more rows
                # (one for each claw) (should be fine)
                # TODO delete
                print('double check PN<>APL merging w/ claw index')
                #breakpoint()
                #

                # TODO try a how= option that doesn't add NaNs, if that can not affect
                # outputs, in order to preserve int dtypes (for claw_cols)? or just
                # restore them later? prob doesn't matter anyway

                # TODO want count_dropped_syns_within (3rd arg) == [KC_ID] here too? if
                # so, just hardcode above, and delete that arg? change output or just
                # reporting?
                # TODO TODO also report how many boutons are getting dropped here?
                #
                # if we were maybe merging w/ a less filtered version of wPNKC, could
                # maybe report some amount of claws getting filtered, before subsetting
                # down to filtered wPNKC? not doing that though. and in general, may not
                # always be able to meaningfully report # of "claws" that should be in
                # the data dropped here, since we only get those IDs by merging.
                apl2pn_df, n_nonclaw_apl2pn_syns_per_kc = drop_missing_claws(apl2pn_df,
                    'APL>PN', [KC_ID]
                )
                pn2apl_df, n_nonclaw_pn2apl_syns_per_kc = drop_missing_claws(pn2apl_df,
                    'PN>APL', [KC_ID]
                )
                # TODO TODO move below assertions on n_nonclaw_* values to here

                # TODO assert reindex_to_wPNKC wouldn't change numerator/denominator at
                # all for these plots below [that i'm now not really planning on making,
                # but still might want the assertion] (all it should be doing is adding
                # claws w/ 0 PN>APL or APL>PN weight, which is what i've been seeing in
                # warn output)
                #
                # TODO delete
                if plot_dir is not None:
                    # TODO use wPNKC_boutons_per_glom as denominator for plot showing
                    # fraction of boutons w/ 0 PN<>APL weight
                    # (actually care, now that i have PN>KC # boutons in plot above?)
                    # TODO matshow fraction of boutons that have non-zero weight, for
                    # each glomerulus?
                    pass
                #


        # NOTE: on v5, -1 is no longer a "missing" claw! (only NaN missing there, at
        # least enough to -1 as valid generally) (still means invalid for all IDs other
        # than claw IDs)
        # NOTE: many synapses are not assigned to claws!!!
        # TODO (delete? how would i even do? not sure there's much/any need on v5)
        # should i lump synapses without claw into nearest claw or what?

        # TODO look at how many apl_unit_col values there are there are, and how
        # they map to claws (make plots to help?)
        # NOTE: -1 should be dropped for these, whether v5 or not, unlikle claw IDs in
        # v5, which are still meaningful
        #
        # ipdb> (a2k.APL_anatomical_unit == -1).sum()
        # 62
        # ipdb> (k2a.APL_anatomical_unit == -1).sum()
        # 39
        # ipdb> a2k.APL_anatomical_unit.isna().sum()
        # 1
        # ipdb> k2a.APL_anatomical_unit.isna().sum()
        # 0
        #
        # 508/509 (including -1, and maybe NaN)

        # could avg pre/post coords if wanted here
        cols_to_avg = []

        extra_cols_to_keep = [apl_unit_col, 'kc_type']

        # since there will be multiple apl_unit_col per claw, only checking this element
        # of extra_cols_to_keep
        check_unique_per_claw = ['kc_type']

        # TODO may want to add 'APL_anatomical_unit' to claw_cols in future, but
        # currently essentially ignoring that column
        apl2kc_df = agg_synapses_to_claws(apl2kc_df, claw_cols, cols_to_avg,
            extra_cols_to_keep, check_unique_per_claw=check_unique_per_claw
        )
        apl2kc_df = apl2kc_df.reset_index()
        apl2kc_df = apl2kc_df.rename(columns={
            'n_synapses': 'weight',
        })
        apl2kc_df = apl2kc_df.drop(columns=[apl_unit_col])

        # TODO remove APL from claw_cols? should only be 1 anyway, so shouldn't change
        # anything? should be able to just use a more global [KC_ID, CLAW_ID] def,
        # after renaming those

        # TODO refactor to share (move into reindex?)
        kc2apl_df = agg_synapses_to_claws(kc2apl_df, claw_cols, cols_to_avg,
            extra_cols_to_keep, check_unique_per_claw=check_unique_per_claw
        )
        kc2apl_df = kc2apl_df.reset_index()
        kc2apl_df = kc2apl_df.rename(columns={
            'n_synapses': 'weight',
        })
        kc2apl_df = kc2apl_df.drop(columns=[apl_unit_col])
        #

        if prat_boutons:
            if per_claw_pn_apl_weights:
                # TODO (fixed? delete?) where was i getting KC_TYPE here before? why not
                # have it now?  (i assume b/c merge isn't pulling it in now, for some
                # reason?)

                # TODO refactor
                apl2pn_df = agg_synapses_to_claws(apl2pn_df, claw_cols, cols_to_avg,
                    extra_cols_to_keep, check_unique_per_claw=check_unique_per_claw
                )

                # TODO refactor
                pn2apl_df = agg_synapses_to_claws(pn2apl_df, claw_cols, cols_to_avg,
                    extra_cols_to_keep, check_unique_per_claw=check_unique_per_claw
                )
                # TODO TODO also check n_claw_bouton_combos in this branch?
                # TODO just remove from extra_cols_to_keep then? should be same as what
                # we currently get after dropping, right?
                apl2pn_df = apl2pn_df.drop(columns=[apl_unit_col])
                pn2apl_df = pn2apl_df.drop(columns=[apl_unit_col])

                # n_claw_bouton_combos is not defined or relevant in
                # per_claw_pn_apl_weights=False case
                # TODO TODO (delete? not sure what i can do) check that this # of unique
                # combos is preserved after agg_synapses_to_claws (would it be? or just
                # check all key combos are still there? may be a subset now?)
                #
                # ipdb> pn2apl_df.n_synapses.sum()
                #81567
                #ipdb> apl2pn_df.n_synapses.sum()
                #81748
                #
                #ipdb> len(apl2pn_df)
                #9365
                #ipdb> len(pn2apl_df)
                #9413
                #ipdb> n_claw_bouton_combos
                #9867
                s1 = {tuple(x) for x in claw2bouton[claw_cols].itertuples(index=False)}
                assert not apl2pn_df.reset_index()[claw_cols].duplicated().any()
                s2 = {
                    tuple(x) for x in
                    apl2pn_df.reset_index()[claw_cols].itertuples(index=False)
                }
                assert s2 - s1 == set()
                assert not pn2apl_df.reset_index()[claw_cols].duplicated().any()
                s3 = {
                    tuple(x) for x in
                    pn2apl_df.reset_index()[claw_cols].itertuples(index=False)
                }
                assert s3 - s1 == set()
            else:
                bouton_levels = list(wPNKC.columns.names)
                apl2pn_df = agg_synapses_to_claws(apl2pn_df, bouton_levels, [], [])
                pn2apl_df = agg_synapses_to_claws(pn2apl_df, bouton_levels, [], [])

            apl2pn_df = apl2pn_df.reset_index()
            apl2pn_df = apl2pn_df.rename(columns={
                'n_synapses': 'weight',
            })

            pn2apl_df = pn2apl_df.reset_index()
            pn2apl_df = pn2apl_df.rename(columns={
                'n_synapses': 'weight',
            })

        # TODO if prat_claws=False (or one-row-per-claw=False), return version
        # of per-claw KC<>APL weights summed across claws per KC too (prob don't want
        # any path returning old data. everything should be derived from this new data)

    # TODO TODO delete this old data loading (replacing all w/ some processing of
    # prat v3 data above), once that's working
    else:
        # not really used, just so it's defined below for renaming levels
        claw_col = 'anatomical_claw'

        apl_data_dir = from_prat / '2025-04-03'

        apl2kc_data = apl_data_dir / 'APL2KC_Connectivity.csv'
        # columns: bodyId_pre, type_pre, instance_pre, bodyId_post, type_post,
        # instance_post, roi, weight
        apl2kc_df = pd.read_csv(apl2kc_data)
        # ipdb> apl2kc_df.instance_pre.unique()
        # array(['APL_R', 'APL fragment?', 'APL or DPM', 'APL fragment_L'],

        # TODO refactor to share assertion + subsetting w/ kc2apl_df processing below
        #
        # unique roi values before filtering:
        # [gL(R), CA(R), b'L(R), a'L(R), aL(R), bL(R), PED(R), SLP(R), PLP(R), SCL(R),
        # b'L(L), NotPrimary, ICL(R), SIP(R), CRE(R)]
        assert {'CA(R)'} == set(
            apl2kc_df.roi[apl2kc_df.roi.str.contains('CA')].unique()
        )
        # TODO TODO what fraction of rows are 'CA(R)'? what are other big components,
        # if any? (and same for kc2apl. put in comment if don't already have)
        apl2kc_df = apl2kc_df[apl2kc_df.roi == 'CA(R)'].copy()

        # filtering pratyush recommended. he doesn't think the other fragments (which
        # may or may not have been from the right APL as well, but may also have been
        # DPM, left APL, or something else) represent much of the data, or are worth
        # using.
        # TODO actually see what fraction of calyx connections are excluded by this
        # filtering?
        apl2kc_df = apl2kc_df[apl2kc_df.instance_pre == 'APL_R'].copy()

        # TODO why are there NaN in type post (how many? 80/1951 rows) (seems they are
        # all for uncertain / fragment KCs)? just drop those i assume?
        #
        # ipdb> apl2kc_df[apl2kc_df.type_post.isna()].instance_post.unique()
        # array(['KCy(half)', 'KC part due to gap', 'KC(incomplete?)'], dtype=object)
        #
        # ipdb> apl2kc_df[~ apl2kc_df.type_post.isna()].instance_post.unique()
        # array(['KCg-m_R', 'KCg-t_R', 'KCg-d_R', 'KCab-m_R', 'KCab-c_R',
        #        "KCa'b'-m_R", "KCa'b'-ap2_R", 'KCg-s3_R', 'KCab-s_R',
        #        'KCg-s2(super)_R', 'KCab-p_R', "KCa'b'-ap1_R", 'KCg-s4_R'],
        #       dtype=object)
        apl2kc_df = apl2kc_df[apl2kc_df.type_post.notna()].copy()
        assert not apl2kc_df.isna().any().any()

        apl2kc_df = add_kc_type_col(apl2kc_df, 'instance_post')
        assert not apl2kc_df[KC_TYPE].isna().any()
        # TODO want any other assertions on type (gap/incomplete?)?
        # I guess inputs are missing what I would need to independently define type for
        # these (or that this is the reality / this data is not available)
        assert not (apl2kc_df[KC_TYPE] == 'unknown').any()

        kc2apl_data = apl_data_dir / 'KC2APL_Connectivity.csv'
        # columns same as above
        kc2apl_df = pd.read_csv(kc2apl_data)

        # TODO refactor to share most filtering w/ above (almost exactly the same)
        assert {'CA(R)'} == set(
            kc2apl_df.roi[kc2apl_df.roi.str.contains('CA')].unique()
        )
        kc2apl_df = kc2apl_df[kc2apl_df.roi == 'CA(R)'].copy()

        # NOTE: using instance_post here instead of instance_pre above
        kc2apl_df = kc2apl_df[kc2apl_df.instance_post == 'APL_R'].copy()
        # NOTE: seems like only 79 NaN here, despite seemingly 80 at this point in
        # filtering for apl2kc_df above. almost certainly doesn't matter.
        # NOTE: using type_pre, instead of type_post for apl2kc_df above
        kc2apl_df = kc2apl_df[kc2apl_df.type_pre.notna()].copy()
        assert not kc2apl_df.isna().any().any()

        kc2apl_df = add_kc_type_col(kc2apl_df, 'instance_pre')
        # TODO move these assertions into add_kc_type_col (/delete. prob already have
        # these assertions in that fn...)?
        assert not kc2apl_df[KC_TYPE].isna().any()
        # I guess inputs are missing what I would need to independently define type for
        # these (or that this is the reality / this data is not available)
        assert not (kc2apl_df[KC_TYPE] == 'unknown').any()

        # TODO refactor to share? if keeping this else branch at all...
        #
        # claw_col will just be ignored if not in columns.
        # should already be done for PN<>APL stuff.
        # TODO do way earlier (and similar in all other datatypes too)
        kc2apl_df = kc2apl_df.rename(columns={'bodyId_pre': KC_ID, claw_col: CLAW_ID})
        apl2kc_df = apl2kc_df.rename(columns={'bodyId_post': KC_ID, claw_col: CLAW_ID})
    #

    apl2kc_ids = set(apl2kc_df[KC_ID])
    kc2apl_ids = set(kc2apl_df[KC_ID])

    # TODO move to end, w/ other plotting? or input change, in a way i don't want, by
    # then?
    if plot_dir is not None:
        # TODO TODO do these plots before normalizing (or invert / keep a second copy?)
        # (just want to make plots look like comb-like, while still preserving any
        # relevant upstream filtering)
        # TODO TODO TODO why these plots still look comb-like? normalizing shouldn't
        # have happened yet, no?
        # TODO TODO include n_claws separate from n_kcs in titles, as recent change in
        # wPNKC plotting code

        min_weight = apl2kc_df['weight'].min()
        # TODO (for per-claw weights, mainly) change to >= 0? (/ remove?)
        # TODO still warn about 0 weights?
        # (actually no claws w/ 0 weights. prob need to fill in 0 for claw_ids w/ no
        # weight. will that happen automatically below? make these plots after doing
        # that?)
        assert min_weight > 0
        fig, ax = _plot_connectome_raw_weight_hist(apl2kc_df['weight'])
        ax.set_title(f'APL->KC weights\n{min_weight=}\n{apl2kc_data.name}'
            f'\nn_kcs={len(apl2kc_ids)}'
        )
        savefig(fig, plot_dir, 'wAPLKC_hist', bbox_inches='tight')

        # TODO refactor
        min_weight = kc2apl_df['weight'].min()
        # TODO (for per-claw weights, mainly) change to >= 0? (/ remove?)
        # TODO still warn about 0 weights?
        # (actually no claws w/ 0 weights [very untrue after filling, but may be true
        # here]. prob need to fill in 0 for claw_ids w/ no weight. will that happen
        # automatically below? make these plots after doing that?)
        assert min_weight > 0
        fig, ax = _plot_connectome_raw_weight_hist(kc2apl_df['weight'])
        ax.set_title(f'KC->APL weights\n{min_weight=}\n{kc2apl_data.name}'
            f'\nn_kcs={len(kc2apl_ids)}'
        )
        savefig(fig, plot_dir, 'wKCAPL_hist', bbox_inches='tight')

        # TODO even want these two?
        if prat_boutons:
            # TODO refactor
            fig, ax = _plot_connectome_raw_weight_hist(apl2pn_df['weight'])
            # TODO want other into in here? # PNs? # boutons? could get # KCs w/ KC_ID
            # for this... (could also get # claws here too)
            ax.set_title('APL->PN weights')
            savefig(fig, plot_dir, 'wAPLPN_hist', bbox_inches='tight')

            # TODO refactor
            fig, ax = _plot_connectome_raw_weight_hist(pn2apl_df['weight'])
            # TODO want other into in here? # PNs? # boutons? could get # KCs w/ KC_ID
            # for this... (could also get # claws here too)
            ax.set_title('PN->APL weights')
            savefig(fig, plot_dir, 'wPNAPL_hist', bbox_inches='tight')

    # TODO probably change so wPNKC/kc_types not passed in typically tho...
    # shouldn't need really (probably want to merge outside either of these fns anyway,
    # to extent needed)
    # TODO test/delete (`wPNKC is None` currently unused)
    if wPNKC is None:
        # TODO assert not kc_to_claws here? or add/check/test support for that case?

        # TODO test filling below works w/ this branch. expecting to mainly use branch
        # below (where wPNKC is passed in)
        index = pd.Index((apl2kc_ids | kc2apl_ids), name=KC_ID)
    #
    else:
        index = wPNKC.index.copy()
        unique_kc_ids = index.get_level_values(KC_ID).unique()

        n_kcs = len(unique_kc_ids)
        if one_row_per_claw:
            n_claws = len(wPNKC)

        assert unique_kc_ids.equals(unique_kc_ids.sort_values())

    # TODO refactor this part to share? (just pass in level to rename to KC_ID and
    # *_df?)

    index_cols = [KC_ID]
    if prat_claws:
        # in this case, [apl2kc|kc2apl]_df *should* also have CLAW_ID, unlike in case
        # where we add CLAW_ID after next few statements
        index_cols.append(CLAW_ID)

    if prat_boutons:
        if per_claw_pn_apl_weights:
            bouton_index_cols = list(index_cols)
        else:
            # TODO refactor to share w/ above agg calls??
            bouton_index_cols = list(wPNKC.columns.names)

    assert not apl2kc_df[index_cols].duplicated().any()
    apl2kc_weights = apl2kc_df[index_cols + ['weight']].set_index(index_cols,
        verify_integrity=True).squeeze()
    assert apl2kc_weights.index.names == index_cols

    # TODO refactor
    assert not kc2apl_df[index_cols].duplicated().any()
    kc2apl_weights = kc2apl_df[index_cols + ['weight']].set_index(index_cols,
        verify_integrity=True).squeeze()
    assert kc2apl_weights.index.names == index_cols

    if prat_boutons:
        # TODO refactor
        assert not apl2pn_df[bouton_index_cols].duplicated().any()
        apl2pn_weights = apl2pn_df[bouton_index_cols + ['weight']].set_index(
            bouton_index_cols, verify_integrity=True).squeeze()
        assert apl2pn_weights.index.names == bouton_index_cols

        # TODO refactor
        assert not pn2apl_df[bouton_index_cols].duplicated().any()
        pn2apl_weights = pn2apl_df[bouton_index_cols + ['weight']].set_index(
            bouton_index_cols, verify_integrity=True).squeeze()
        assert pn2apl_weights.index.names == bouton_index_cols

    if one_row_per_claw and not prat_claws:
        # NOTE: this needs to happen after use of index_cols above
        # in (one_row_per_claw and not prat_claws) case, APL<>KC data does not have
        # CLAW_ID, but wPNKC does.
        index_cols.append(CLAW_ID)

    if not one_row_per_claw:
        # TODO also check nothing with 'claw' in lower()? where else did i do that?
        assert CLAW_ID not in index.names

    assert all(x in index.names for x in index_cols)
    assert not index.to_frame(index=False)[index_cols].duplicated().any(), \
        f'duplicates in input wPNKC index, within {index_cols=}'

    assert apl2kc_weights.index.names == kc2apl_weights.index.names

    if prat_boutons:
        if per_claw_pn_apl_weights:
            assert apl2kc_weights.index.names == apl2pn_weights.index.names
            assert pn2apl_weights.index.names == apl2pn_weights.index.names
        else:
            # TODO this what i want? anything else?
            assert apl2pn_weights.index.names == bouton_index_cols
            assert pn2apl_weights.index.names == bouton_index_cols

    only_in_wPNKC_index = [x for x in index.names
        if x not in apl2kc_weights.index.names
    ]
    # We only have claw IDs for APL weights in the outputs from Pratyush. With Tianpei's
    # inputs, we just have KC IDs, and we assign claw IDs from the PN>KC data.
    if one_row_per_claw and not prat_claws:
        assert CLAW_ID in only_in_wPNKC_index
        only_in_wPNKC_index = [x for x in only_in_wPNKC_index if x != CLAW_ID]

    # TODO rename var, now that we are special casing above (so not actually always
    # "shared")?
    # TODO TODO also remove CLAW_ID from this?
    index_sharedlevels = index.droplevel(only_in_wPNKC_index)

    # NOTE: reindex(index_nodupes) seems to work (in contrast to call where
    # reindex(...) argument has duplicates in it) (still need to restore
    # remaining metadata values in index, including for rows where [KC_ID,
    # CLAW_ID] are duplicated, perhaps via merge)
    #
    # TODO TODO refactor all other uses of reindex (in all my code. couple
    # calls at least elsewhere in this repo, and in natmix_data/analysis.py) to
    # assert no duplicates in input index? to check for similar bugs.
    # TODO merge simpler than reindex? or useful to restore something with
    # the duplicates?
    # TODO try drop_duplicates(subset=[KC_ID, CLAW_ID]) instead of subsetting
    # to columns first (see if output still passes assertions) (easier to work
    # from this, rather than having to add back the missing info? would still
    # have to add back missing info for duplicates... so maybe easier to rethink
    # whole strategy?)
    index_nodupes2 = pd.MultiIndex.from_frame(index.to_frame(index=False)[
        index_cols].drop_duplicates()
    )
    # because the .equals check below will fail for a MultiIndex with one level (vs and
    # Index) even if the content of that single level matches that of the Index
    if index_nodupes2.nlevels == 1:
        assert index_nodupes2.names == [KC_ID]
        index_nodupes2 = index_nodupes2.get_level_values(KC_ID)

    # TODO (this comment still accurate for prat_claws=True case? can't be. there is
    # only CLAW_ID now [which are now prat's originally ID's, not some renumbered/split
    # thing of mine])
    # unit is (KC, claw) pair, NOT (KC, CLAW_ID) pair (which already splits out
    # separate PNs, removing these duplicates)
    #
    # unit is KC in one_row_per_claw=False case
    n_pns_per_unit = index.to_frame(index=False).groupby(index_cols, sort=False
        ).size()
    index_nodupes = n_pns_per_unit.index
    assert index_nodupes2.equals(index_nodupes)

    # it's ok that this index will have some duplicates. merge below still works
    # as intended, as checked by assertions beneath it.
    # NOTE: doesn't seem to have dupes anymore (I forget what changed. comment
    # nearby may explain it?)
    # TODO add verify_integrity=True, if i expect that to remain the case?
    index_df_with_dupes = index.to_frame(index=False).set_index(index_cols)

    # the "with_dupes" one used to be defined from claw_col
    # (previously ='anatomical_claw'), where preprocesing (in connectome_wPNKC) was also
    # not dropping a small number of duplicates. some of the no-dupe versions
    # may have been using the old renumbered CLAW_ID, not that it matters now.
    # presence of duplicates was causing bad output of .reindex(...) call below.
    assert index_nodupes.equals(index_df_with_dupes.index)
    assert not index_nodupes.duplicated().any()

    # TODO delete (or fix first later, but not prioritizing
    # `one_row_per_claw and not prat_claws` cases for now, until after PN<>APL basics
    '''
    if one_row_per_claw and not prat_claws:
        # TODO TODO define index / function / something to use below to expand
        # index on each weights argument to reindex_to_wPNKC
        #
        # TODO something like this useful?
        # index.to_frame(index=False).groupby(KC_ID).apply(lambda x: x[CLAW_ID].unique())
        breakpoint()
    '''

    # TODO TODO try a version where we sum weights for all claws w/ one bouton
    # [i.e. one microglomerulus], and use that weight (or some normalized
    # version of it) for all claws associated with the bouton?
    # (assuming maybe gap junctions could account for coupling between KCs
    # within one microglomerulus?)

    # TODO TODO TODO do similar for all PN>APL and APL>PN?

    # TODO TODO option to (or by default) handle all non-wPNKC stuff as
    # -1 (and maybe option to lump all that into each KC, probably dividing
    # between all claws)

    # TODO can i make this work here (for all cases)? copied from within prat_claws=True
    # conditional below
    def reindex_to_wPNKC(desc: str, weights: pd.Series) -> pd.Series:
        """Warns about how counts+fractions of synapses & claws dropped

        Also warns about 0-weight KCs added.

        Args:
            desc: unique description of input data (e.g. 'APL>KC')

            weights: Series with (KC_ID, CLAW_ID) index and # synapse values,
                (KC_ID, CLAW_ID) combos may have been filtered, but not yet
                subset to only those in wPNKC

        Returns:
            reindexed: similar to weights (though will also contain index levels
                only in wPNKC.index), but has been reindexed to also contain
                (KC_ID, CLAW_ID) combos only in wPNKC. will typically have KCs with
                0-weight have been added by reindexing.
        """
        # TODO TODO move earlier plotting (/etc) in here?

        assert not weights.isna().any()
        assert not (weights <= 0).any()

        # TODO assert earlier that all of index_nodupes is subset of pairs in wPNKC
        # index? (would also be pretty tautological...)
        # TODO delete? should be tautological, at least so long as reindex calls
        # above have argument that is an index without dupes
        # TODO TODO add claws for each KC up here (in that case), and maybe remove
        # code below that might be doing that? (or check equiv?)
        # TODO prob need to adapt messages below to say we are dropping at least that
        # many claws in that case (or an unknown #, for the KCs not in wPNKC)
        #
        # TODO TODO fix how this is all NaN in `one_row_per_claw and not prat_claws`
        # cases (or will i need to do some special casing after all? what path did that
        # take through this fn, before my refactoring into reindex_to_wPNKC?)
        # TODO TODO use level= arg to reindex, to "broadcast matching Index values on
        # the passed MultiIndex level"? for `one_row_per_claw and not prat_claws` case?
        reindexed = reindex(weights, index_nodupes)

        assert not reindexed.isna().all(), ('reindexing failed. index levels mistmatch?'
            f'\n{index_nodupes.names=}\n{weights.index.names=}'
        )

        assert reindexed.index.equals(index_nodupes)
        # TODO TODO probably need to redef both index_sharedlevels and
        # weights.index, for `one_row_per_claw and not prat_claws` case
        input_and_wPNKC = set(index_sharedlevels.intersection(weights.index))
        # TODO (done?) check again after processing to restore metadata for dupes?
        assert input_and_wPNKC == set(reindexed.dropna().index)

        msg_prefix = f'{desc} subsetting (reindexing) to wPNKC:'
        msg = str(msg_prefix)

        # seems this is actually KCs, not claws, in one_row_per_claw=False case
        # TODO TODO why now causing issue when weights.index.names == ['kc_id']
        # and index_sharedlevels.names == claw_cols?? shouldn't names be same anyway?
        # TODO TODO assert that here? (would fail in prat_claws=False
        # one_row_per_claw=True case currently, but a bunch of other stuff above also
        # currently doesn't make sense there)
        #assert weights.index.names == index_sharedlevels.index.names, \
        #    f'{weights.index.names=} != {index_sharedlevels.index.names=}'
        only_input = weights.index.difference(index_sharedlevels)

        have_claws = CLAW_ID in index_nodupes.names
        if have_claws:
            assert index_nodupes.names == claw_cols
        else:
            assert index_nodupes.names == [KC_ID]

        only_input_noclaw = None
        if have_claws and len(only_input) > 0:
            # TODO TODO TODO need to sum up over KCs, and add to n_dropped_syns below?
            # or is one included in the other? (probably below is included in this?)

            # no way of associating any of these -1's with any claw IDs from
            # PN>KC data (including the "valid" -1's there, which are only
            # called that b/c they have enough inputs from one PN ID)
            only_input_noclaw = only_input[
                only_input.get_level_values(CLAW_ID) == -1
            ]
            n_dropped_claws = len(only_input)
            n_dropped_noclaw_claws = len(only_input_noclaw)

            # divisor is # input weight "claws" (including unassigned, for each
            # KC) and synapses, not claws/synapses in wPNKC
            n_input_claws = len(weights)

            msg += (f'\n - drops {fmt_frac(n_dropped_claws, n_input_claws)} claws, '
                f'{fmt_frac(n_dropped_noclaw_claws, n_input_claws, den=False)} of '
                'which were unassigned (-1)'
            )

        n_dropped_syns = weights[only_input].sum()
        if n_dropped_syns > 0:
            n_input_syns = weights.sum()
            msg += f'\n - drops {fmt_frac(n_dropped_syns, n_input_syns)} synapses'

            if only_input_noclaw is not None:
                n_dropped_noclaw_syns = weights[only_input_noclaw].sum()
                msg += (f', {fmt_frac(n_dropped_noclaw_syns, n_input_syns, den=False)} '
                    'with unassigned claws'
                )

        input_kcs = set(weights.index.get_level_values(KC_ID).unique())
        # TODO maybe use unique_kc_ids instead? (and work w/ Index objects instead of
        # sets?) (or one of the index objects [from wPNKC] i already use in here?)
        pn2kc_ids = set(unique_kc_ids)
        n_dropped_kcs = len(input_kcs - pn2kc_ids)

        if not have_claws:
            assert n_dropped_kcs == len(only_input)

        n_dropped_syns_per_kc = weights[only_input].groupby(KC_ID).sum()
        n_dropped_kc_syns = 0
        if n_dropped_kcs > 0:
            n_dropped_kc_syns = n_dropped_syns_per_kc[
                # length of this difference same as n_dropped_kcs
                n_dropped_syns_per_kc.index.difference(unique_kc_ids)
            ].sum()
            msg += (f'\n - drops {fmt_frac(n_dropped_kcs, len(input_kcs))} KCs'
                # TODO keep?
                f' (and their {n_dropped_kc_syns} synapses)'
            )

        sum_before = n_dropped_syns_per_kc.sum()
        n_dropped_syns_per_kc = reindex(n_dropped_syns_per_kc, unique_kc_ids).fillna(0
            ).astype(int)

        sum_after = n_dropped_syns_per_kc.sum()
        assert sum_before - sum_after == n_dropped_kc_syns

        # TODO delete assertion? redundant w/ asserting reindexed.index.equals(index)
        assert set(reindexed.index.get_level_values(KC_ID)) == pn2kc_ids
        n_0weight_kcs_added = len(pn2kc_ids -
            set(reindexed.dropna().index.get_level_values(KC_ID))
        )
        if n_0weight_kcs_added > 0:
            msg += (f'\n - adds {n_0weight_kcs_added}/{len(pn2kc_ids)} 0-weight '
                'KCs (IDs only in wPNKC)'
            )

        if have_claws:
            # NaNs only added for wPNKC claws that weren't in input, as we assert input
            # has no NaN (at start of this fn). also don't have any existing 0-weight
            # values in input data.
            n_0weight_claws_added = reindexed.isna().sum()
            if n_0weight_claws_added > 0:
                msg += (f'\n - adds {n_0weight_claws_added}/{len(reindexed)} 0-weight '
                    'claws (IDs only in wPNKC)'
                )

        msg += '\n'
        if len(msg) > len(msg_prefix):
            warn(msg)

        # TODO delete? somewhat of a tautology
        #
        # would expect to fail if n_pns_per_unit > 1 entry did not have an even
        # number of synapses in APL>KC and KC>APL weights, but it does in both cases
        # (for single duplicate at this point in v5)
        assert weights[list(input_and_wPNKC)].sort_index().astype(float
            ).equals( reindexed.dropna().sort_index() )

        assert reindexed.index.equals(n_pns_per_unit.index)

        # TODO rename n_pns_per_unit? make sense this is always all 1 in this case?
        # or just name back to n_pns_per_claw, and otherwise special case
        # one_row_per_claw=False cases (not using this var)?
        if index_cols == [KC_ID]:
            assert (n_pns_per_unit == 1).all()

        sum_before = reindexed.sum()
        # n_pns_per_unit has same index as each of these, and thus these operations
        # should not change index
        # NOTE: n_pns_per_unit is all 1 in one_row_per_claw=False case, so as long as
        # index matches reindexed.index, this shouldn't be doing anything
        reindexed = reindexed / n_pns_per_unit

        # since division above seems to remove the .name, which is required for
        # merge calls below
        reindexed.name = 'weight'

        # TODO still neeed?
        # restoring index levels (from wPNKC.index) i had to drop to remove
        # duplicates. since this adds duplicates back, if it weren't for dividing by
        # n_pns_per_unit above, these merges would change .sum()
        reindexed = index_df_with_dupes.merge(reindexed, left_index=True,
            right_index=True
        )
        assert reindexed.index.equals(index_df_with_dupes.index)

        assert reindexed['weight'].sum() == sum_before
        reindexed = reindexed.reset_index().set_index(index.names).squeeze()
        assert reindexed.index.equals(index)

        # TODO TODO move plotting and most other stuff here too
        # (continue factoring into this fn, so we can simplify calls across data
        # types)

        return reindexed, n_dropped_syns_per_kc

    # TODO TODO fix reindex_to_wPNKC to also work for this (tianpei) case?
    wAPLKC = None
    wKCAPL = None
    if not (one_row_per_claw and not prat_claws):
        wAPLKC, n_nonclaw_apl2kc_syns_per_kc2 = reindex_to_wPNKC('APL>KC',
            apl2kc_weights
        )
        wKCAPL, n_nonclaw_kc2apl_syns_per_kc2 = reindex_to_wPNKC('KC>APL',\
            kc2apl_weights
        )
        # so yea, basically negligible:
        # ipdb> n_nonclaw_apl2kc_syns_per_kc.reindex(unique_kc_ids).fillna(0).sum()
        # 1.0
        # ipdb> n_nonclaw_kc2apl_syns_per_kc.reindex(unique_kc_ids).fillna(0).sum()
        # 3.0
        # ipdb> n_nonclaw_pn2apl_syns_per_kc.reindex(unique_kc_ids).fillna(0).sum()
        # 0.0
        # ipdb> n_nonclaw_apl2pn_syns_per_kc.reindex(unique_kc_ids).fillna(0).sum()
        # 0.0

        # ipdb> n_nonclaw_apl2kc_syns_per_kc2.sum()
        # 10049
        # ipdb> n_nonclaw_kc2apl_syns_per_kc2.sum()
        # 9104

        # TODO delete (replace w/ appropriate summing, but really insignificant, so this
        # is fine for now)
        n_nonclaw_apl2kc_syns_per_kc = n_nonclaw_apl2kc_syns_per_kc2
        n_nonclaw_kc2apl_syns_per_kc = n_nonclaw_kc2apl_syns_per_kc2

    if prat_boutons:
        if per_claw_pn_apl_weights:
            # TODO TODO double check that each of these really has all their (kc, claw)
            # combos present in the wPNKC data
            wAPLPN, n_nonclaw_apl2pn_syns_per_kc2 = reindex_to_wPNKC('APL>PN',
                apl2pn_weights
            )
            wPNAPL, n_nonclaw_pn2apl_syns_per_kc2 = reindex_to_wPNKC('PN>APL',
                pn2apl_weights
            )

            # TODO maybe it's just a consequence of how i'm doing the merges
            # above? look back into that?
            # TODO TODO any boutons not paired w/ claws? summarize?
            #
            # so if these calculations are correct, don't need to worry about or try to
            # analyze any of these. can just focus on non-claw KC<>APL stuff.
            #
            # TODO TODO move these top assertions above (and/or change to [/ supplement
            # with] assertions on NaN right after merging?)
            assert n_nonclaw_apl2pn_syns_per_kc.sum() == 0
            assert n_nonclaw_pn2apl_syns_per_kc.sum() == 0
            #
            assert n_nonclaw_apl2pn_syns_per_kc2.sum() == 0
            assert n_nonclaw_pn2apl_syns_per_kc2.sum() == 0

            # TODO TODO ok, am i doing something wrong? surely either this or the non
            # *2 version should have something?
            # TODO TODO don't i actually need to count on a (kc, claw) basis, not just
            # a KC basis? (was doing that actually tho, i think?)
            # ipdb> n_nonclaw_apl2pn_syns_per_kc2.sum()
            # 0.0
            # ipdb> n_nonclaw_pn2apl_syns_per_kc2.sum()
            # 0.0

            # TODO TODO make sure i was computing each the right way (should be a lot of
            # ovoerlapping KC IDs between two, right? or at least, a lot of overlap
            # between the KC IDs in the *2 versions and the wAPLKC/etc KC IDs, no?)

            # TODO add above # dropped per KC to others?
            # (could probably mostly ignore above contribution. second seems much bigger
            # in [i think?] all cases?) (yea, see comment above comparing magnitudes)
            # TODO + make sure index stays same as *2 versions after adding, at least
        else:
            # TODO convert dtype of bouton_id to int earlier (or prevent from becoming
            # float in first place) (there are no NaNs here, and should be asserted
            # above anyway)
            wAPLPN = apl2pn_df.astype({BOUTON_ID: int}).set_index(bouton_index_cols
                ).squeeze()
            wPNAPL = pn2apl_df.astype({BOUTON_ID: int}).set_index(bouton_index_cols
                ).squeeze()

            # TODO delete
            # ipdb> wAPLPN
            # glomerulus  pn_id       bouton_id
            # D           1536947502   0           2.730784
            #                          1           6.826961
            #             5813038889   0           5.461569
            #                          1           4.778873
            #             5813055184   3           4.778873
            #                                        ...
            # VP2         1975878958  -1           0.682696
            #                          1           0.682696
            #                          5           1.365392
            #                          6           1.365392
            #                          9           0.682696
            # Name: weight, Length: 439, dtype: float64
            # ipdb> wPNAPL
            # glomerulus  pn_id       bouton_id
            # D           1536947502  -1            0.676827
            #                          0            2.030481
            #                          1            3.384134
            #             5813038889   0            2.030481
            #                          1            4.737788
            #                                        ...
            # VP1m        5813056072   0           12.182884
            # VP2         1975878958   1            0.676827
            #                          5            1.353654
            #                          6            4.737788
            #                          9            2.707308
            # Name: weight, Length: 420, dtype: float64
            # ipdb> wPNKC.columns.duplicated().any()
            # False

            assert not wPNKC.columns.duplicated().any()

            wAPLPN = wAPLPN.reindex(wPNKC.columns)
            assert wAPLPN.index.equals(wPNKC.columns)

            wPNAPL = wPNAPL.reindex(wPNKC.columns)
            assert wPNAPL.index.equals(wPNKC.columns)

    # NOTE: wPNKC CLAW_ID (when present) is no longer some renumbered ID I came up with.
    # It is raw anatomical_claw[_corrected] values from Pratyush, so should be able to
    # use to align with the same values in here. The only remaining issue is regarding
    # -1 handling, where -1 can probably never really be assigned to a particular claw
    # in here, as the -1 claw IDs in PN>KC data are also formed from synapses without
    # particular spatial clustering, just happened to have enough synapses from one PN
    # ID.

    # TODO TODO (done?) plot distributions of apl<->kc weights (within each claw) here
    # (and within all claws per bouton). make sense there are so many (majority) <=3 in
    # apl2kc data (yes)?

    # TODO so kc_to_claws was never actually used in here? replace w/ some boolean flag?
    # (don't think it ever was used here... checked some old commits from tianpei)
    if kc_to_claws is not None:
        # TODO use pandas vectorized filling instead of looping over all kc_id
        # values. replace this loop w/ more idiomatic code.
        # Iterate through each KC to distribute the weights to its claws
        if not prat_claws:
            wAPLKC2 = pd.Series(index=index, dtype=float)
            wKCAPL2 = pd.Series(index=index, dtype=float)

            for kc_id in index.get_level_values(KC_ID).unique():
                # Get the original weights for the current KC
                apl_weight = apl2kc_weights.get(kc_id, 0)
                kca_weight = kc2apl_weights.get(kc_id, 0)

                # Get the sub-index for all claws belonging to this KC
                kc_claws_index = wPNKC.loc[kc_id].index

                # TODO this accurate? test! len(kc_claws_index) should be # of claws for
                # the current KC
                if len(kc_claws_index) > 0:
                    num_claws = len(kc_claws_index)
                    distributed_apl_weight = apl_weight / num_claws
                    distributed_kca_weight = kca_weight / num_claws

                    # TODO this is setting for all claws, right?
                    # Assign the distributed weight to the appropriate rows in the
                    # Series
                    wAPLKC2.loc[kc_id] = distributed_apl_weight
                    wKCAPL2.loc[kc_id] = distributed_kca_weight

            # None b/c still havnen't fixed reindex_to_wPNKC (above) to work in this
            # case
            if wAPLKC is None:
                assert wKCAPL is None
                wAPLKC = wAPLKC2
                wKCAPL = wKCAPL2
            else:
                # TODO maybe break the dividing by num_claws into separate step, and
                # have assertion before that?
                # TODO get to work (after also implementing this case in
                # reindex_to_wPNKC above)
                breakpoint()
                assert wAPLKC.equals(wAPLKC2)
                assert wKCAPL.equals(wKCAPL2)

    if one_row_per_claw:
        # TODO TODO scale all to sum of one instead (help make things more comparable
        # between KC and bouton cases?), and then rework all downstream code from there?

        # TODO refactor
        wAPLKC = wAPLKC.fillna(0)
        assert wAPLKC.sum() > 0
        assert (wAPLKC >= 0).all()
        wAPLKC_normalization_factor = n_kcs / wAPLKC.sum()
        wAPLKC = wAPLKC * wAPLKC_normalization_factor
        assert np.isclose(wAPLKC.sum() / n_kcs, 1)
        assert np.isclose(wAPLKC.groupby('kc_id').sum().mean(), 1)

        # TODO TODO can't just let filling happen below? does NaN cause issues in this
        # branch?
        # TODO move these below filling below (outside of this conditional) (/delete)?
        # TODO is this not duplicated below now? refactor to share?
        # Fill any remaining NaNs (for KCs with no claws in the data)
        wKCAPL = wKCAPL.fillna(0)
        assert wKCAPL.sum() > 0
        assert (wKCAPL >= 0).all()
        # TODO why not doing in non-one-row-per-claw case?
        # (b/c it's down outside of this fn i think? move all handling in to one place
        # or the other?)
        # TODO TODO should this be # claws on top instead?
        # TODO TODO TODO or if not, maybe scale boutons within each glomerulus (in
        # python, below), similar to how we are scaling claws within KCs here?
        # TODO or don't scale claws w/in KCs?
        wKCAPL_normalization_factor = n_kcs / wKCAPL.sum()
        # TODO TODO use to invert for plotting (if i can't figure out how to do it
        # automatically, without saving these factors)
        wKCAPL = wKCAPL * wKCAPL_normalization_factor
        # TODO is this actually a property we want to enforce tho?
        assert np.isclose(wKCAPL.sum() / n_kcs, 1)
        # why do we not want assertions like these outside of one_row_per_claw
        # conditional? (because one_row_per_claw=False path currently
        # does NOT normalize in here. currently happens outside [after] this function.)
        # TODO may want to change to be consistent at some point in the future...
        #
        # TODO replace some of the assertions above w/ these?
        assert np.isclose(wKCAPL.groupby('kc_id').sum().mean(), 1)

        # TODO delete
        print()
        print('connectome_APL_weights:')
        print(f'{wAPLKC.mean()=}')
        print(f'{wKCAPL.mean()=}')
        #

        # TODO even need this in any prat_boutons case? don't think there is still
        # any NaN? do probably need the normalization though, or at least some kind
        # TODO TODO and also to vary APL>PN vs PN>APL
        if prat_boutons:
            # TODO TODO better to normalize sum to same thing (could just normalize
            # sum of PN<>APL stuff to sum of whatever the mean-normalized KC<>APL stuff
            # is), to keep PN<>APL and KC<>APL weight scales comparable (so i can
            # hopefully use one, or sweep around the KC<>APL scale factor)?

            # TODO TODO TODO scale boutons w/in glomeruli (like claws w/in KCs above)?
            # TODO or don't scale claws w/in KCs above?

            n_boutons = len(wAPLPN)
            assert len(wPNAPL) == n_boutons

            # TODO even necessary?
            wAPLPN = wAPLPN.fillna(0)
            assert wAPLPN.sum() > 0
            assert (wAPLPN >= 0).all()
            # TODO just always divide length of vector by sum() (no, for some
            # reason, i think i might need to use n_kcs here for some tests to pass?
            # (and why not the other way around??)
            # TODO TODO or was this initial scales the reason olfsysm tests were
            # only passing with a certain init? try moving all this weight vector
            # scaling to olfsysm, to avoid doing it wrong outside?
            if not per_claw_pn_apl_weights:
                assert len(wAPLPN) == n_boutons
                wAPLPN_normalization_factor = n_boutons / wAPLPN.sum()
            else:
                # TODO TODO try reverting to n_claws while also changing init of this
                # (and wKCAPL, in olfsysm, to use length of vector rather than # KCs
                # there)
                #assert len(wAPLPN) == n_claws
                #wAPLPN_normalization_factor = n_claws / wAPLPN.sum()
                wAPLPN_normalization_factor = n_kcs / wAPLPN.sum()

            wPNAPL = wPNAPL.fillna(0)
            assert wPNAPL.sum() > 0
            assert (wPNAPL >= 0).all()
            if not per_claw_pn_apl_weights:
                assert len(wPNAPL) == n_boutons
                wPNAPL_normalization_factor = n_boutons / wPNAPL.sum()
            else:
                # TODO TODO try reverting to n_claws while also changing init of this
                # (and wKCAPL, in olfsysm, to use length of vector rather than # KCs
                # there)
                #assert len(wPNAPL) == n_claws
                #wPNAPL_normalization_factor = n_claws / wPNAPL.sum()
                wPNAPL_normalization_factor = n_kcs / wPNAPL.sum()

            # TODO delete
            print()
            print('before')
            print(f'{wAPLPN.mean()=}')
            print(f'{wAPLPN.sum()=}')
            print(f'{wAPLPN_normalization_factor=}')
            print(f'{wPNAPL.mean()=}')
            print(f'{wPNAPL.sum()=}')
            print(f'{wPNAPL_normalization_factor=}')
            #

            # TODO delete
            #wAPLPN_normalization_factor = (n_kcs / wAPLPN.sum()) / pn_apl_scale_factor
            wAPLPN = wAPLPN * wAPLPN_normalization_factor
            # TODO delete
            #wPNAPL_normalization_factor = (n_kcs / wPNAPL.sum()) / pn_apl_scale_factor
            wPNAPL = wPNAPL * wPNAPL_normalization_factor

            # TODO delete
            print()
            print('after')
            print(f'{wAPLPN.mean()=}')
            print(f'{wAPLPN.sum()=}')
            print(f'{wPNAPL.mean()=}')
            print(f'{wPNAPL.sum()=}')
            print()
            #

            # TODO delete
            '''
            if pn_apl_scale_factor != 1:
                # TODO TODO redef above, so scale factor isn't so arbitrary here (to
                # remove n_kcs from numerator, or at least also ref # boutons). then
                # update what any hardcoded scale factors (e.g. in pn_claw_to_APL=False
                # branch in test code, currently =200) to what new values should be
                warn(f'dividing PN<>APL weights all by {pn_apl_scale_factor=} (to '
                    'scale relative to KC<>APL weights)'
                )
            '''

            # TODO same assertions, but grouping by glomeruli for the
            # `not per_claw_pn_apl_weights` case?
            if per_claw_pn_apl_weights:
                # NOTE: not true if we use n_claws in scale factors above (which might
                # want to try along with an olfsysm change, if trying to move weight
                # normalization in there)
                assert np.isclose(wAPLPN.sum() / n_kcs, 1)
                assert np.isclose(wAPLPN.groupby('kc_id').sum().mean(), 1)

                assert np.isclose(wPNAPL.sum() / n_kcs, 1)
                assert np.isclose(wPNAPL.groupby('kc_id').sum().mean(), 1)

        # TODO (delete? not sure tianpeis stuff that useable...) compare distribution of
        # weights from prat_claws=True/False paths in here (# dists of # claws per KC,
        # if i haven't already checked those are [essentially] same in both cases)
    else:
        # shouldn't need to define normalization factors here. these should currently be
        # normalized outside of connectome_APL_weights in this path, and still be raw
        # int synapse counts here.
        wAPLKC_normalization_factor = None
        wKCAPL_normalization_factor = None

        # TODO delete this comparison eventually? (just use the non-2-suffixed version
        # of each, and remove this else conditional)
        wAPLKC2 = pd.Series(index=index, data=apl2kc_weights)
        wKCAPL2 = pd.Series(index=index, data=kc2apl_weights)
        assert wAPLKC.equals(wAPLKC2)
        assert wKCAPL.equals(wKCAPL2)
        #

    # TODO move into reindex_to_wPNKC (if not already there?) or still want after
    # post-processing (that currently exists only for one_row_per_claw=True branch,
    # right?)
    # TODO keep?
    assert wAPLKC.index.equals(index)
    assert wKCAPL.index.equals(index)
    #

    # TODO move into reindex_to_wPNKC?
    wAPLKC.name = 'weight'
    wKCAPL.name = 'weight'

    # TODO TODO can i move this into reindex_to_wPNKC? does it matter if this happens
    # before/after some of the ot her processing (as long as fill value is 0, good
    # chance it doesn't?)? (test, now that i have repro test?)
    # (or can i at least move this [+ plotting below] before normalizing above, so i
    # don't need to ever invert that?)
    #
    # TODO would it be better to impute some non-zero min value, so those cells activity
    # could still be scaled (add param for that?)?
    fill_weight = 0
    # we now warn about this filling above
    wAPLKC = wAPLKC.fillna(fill_weight)
    wKCAPL = wKCAPL.fillna(fill_weight)

    assert not wAPLKC.isna().any()
    assert not wKCAPL.isna().any()

    # TODO TODO TODO is there actually still any NaN here? and at least for
    # non-per-claw case, already name 'weight' as name here. this all duplicated?
    if prat_boutons:
        # TODO refactor all this into reindex_to_wPNKC?

        # TODO isn't this already guaranteed above? delete?
        if per_claw_pn_apl_weights:
            assert wAPLPN.index.equals(index)
            assert wPNAPL.index.equals(index)

        # TODO isn't this already true?
        wAPLPN.name = 'weight'
        wPNAPL.name = 'weight'
        wAPLPN = wAPLPN.fillna(fill_weight)
        wPNAPL = wPNAPL.fillna(fill_weight)
        assert not wAPLPN.isna().any()
        assert not wPNAPL.isna().any()

    if kc_types is not None:
        # explore whether subtypes have diff weights in either (esp a'/b' vs others),
        # along lines of inada/kazama paper (they don't, but can i force them to, so
        # that effect is similar? presumably?). anything else to look at?

        # index should be same as wAPLKC and wKCAPL indices
        assert len(kc_types) == len(index)

        # is it consistent w/ expectation that a/b (802) are more numerous than gamma
        # (612)? (betty thought they counts across all 3 seemed reasonable)
        type2count = kc_types.value_counts()
        old_type_index = type2count.index.copy()
        type2count.index = type2count.index + ' (' + type2count.astype(str).values + ')'

        # converting to Series since Index doesn't have a .replace method
        kc_types = pd.Series(kc_types)

        type2type_with_count = dict(zip(old_type_index, type2count.index))
        kc_types = kc_types.replace(type2type_with_count)

        # do need this for some reason, or else adding kc_type col to
        # w[APLKC|KCAPL]_with_types will only add an all-NaN kc_type column
        kc_types = pd.Index(kc_types)

        # TODO move plotting before normalization, to avoid need for all this inverting
        # of it?
        # TODO do i ever actually *want* to plot the normalized values (maybe for the
        # scaling that is just to divide by # of claws? ever really care about those
        # paths now? prob not...)?
        wAPLKC_for_plots = wAPLKC
        wKCAPL_for_plots = wKCAPL
        if wAPLKC_normalization_factor is not None:
            inverted_normed_wAPLKC = wAPLKC / wAPLKC_normalization_factor
            inverted_int_wAPLKC = inverted_normed_wAPLKC.round().astype(int)
            if not (one_row_per_claw and not prat_claws):
                assert pd_allclose(inverted_int_wAPLKC, inverted_normed_wAPLKC)
                wAPLKC_for_plots = inverted_int_wAPLKC
            else:
                # because different KCs have different # of claws, and weights divided
                # within each KC by that, we don't start with int weights above (when
                # computing normalization factor), nor could we have a single one
                # (computed same way)
                assert not pd_allclose(inverted_int_wAPLKC, inverted_normed_wAPLKC)
                # TODO TODO just plot raw wAPLKC instead?
                # TODO TODO TODO or per KC int version (rather than per-claw)?
                wAPLKC_for_plots = inverted_normed_wAPLKC

            assert wKCAPL_normalization_factor is not None
            # TODO refactor
            inverted_normed_wKCAPL = wKCAPL / wKCAPL_normalization_factor
            inverted_int_wKCAPL = inverted_normed_wKCAPL.round().astype(int)
            if not (one_row_per_claw and not prat_claws):
                assert pd_allclose(inverted_int_wKCAPL, inverted_normed_wKCAPL)
                wKCAPL_for_plots = inverted_int_wKCAPL
            else:
                assert not pd_allclose(inverted_int_wKCAPL, inverted_normed_wKCAPL)
                wKCAPL_for_plots = inverted_normed_wKCAPL

            if prat_boutons:
                assert wAPLPN_normalization_factor is not None
                assert wPNAPL_normalization_factor is not None

                # TODO refactor
                inverted_normed_wAPLPN = wAPLPN / wAPLPN_normalization_factor
                inverted_int_wAPLPN = inverted_normed_wAPLPN.round().astype(int)
                assert pd_allclose(inverted_int_wAPLPN, inverted_normed_wAPLPN)
                wAPLPN_for_plots = inverted_int_wAPLPN

                # TODO refactor
                inverted_normed_wPNAPL = wPNAPL / wPNAPL_normalization_factor
                inverted_int_wPNAPL = inverted_normed_wPNAPL.round().astype(int)
                assert pd_allclose(inverted_int_wPNAPL, inverted_normed_wPNAPL)
                wPNAPL_for_plots = inverted_int_wPNAPL
        else:
            assert not prat_boutons
        #

        wAPLKC_with_types = wAPLKC_for_plots.to_frame()
        wKCAPL_with_types = wKCAPL_for_plots.to_frame()
        if prat_boutons:
            wAPLPN_with_types = wAPLPN_for_plots.to_frame()
            wPNAPL_with_types = wPNAPL_for_plots.to_frame()

        # TODO delete? can i avoid need for this by fixing some code duplication
        # earlier? still need to pass in kc_types now? just use something based on
        # existing KC_TYPE level we seem to have here in index now?
        if KC_TYPE in wAPLKC_with_types.index.names:
            assert KC_TYPE in wKCAPL_with_types.index.names
            # and all these *_with_types vars should have same index, so shouldn't need
            # to check each
            assert wAPLKC_with_types.index.get_level_values(KC_TYPE).equals(
                kc_types.str.split().map(lambda x: x[0])
            )

            # NOTE: not changing wAPLKC/wKCAPL that get returned
            wAPLKC_with_types = wAPLKC_with_types.droplevel(KC_TYPE)
            wKCAPL_with_types = wKCAPL_with_types.droplevel(KC_TYPE)

            if prat_boutons and per_claw_pn_apl_weights:
                wAPLPN_with_types = wAPLPN_with_types.droplevel(KC_TYPE)
                wPNAPL_with_types = wPNAPL_with_types.droplevel(KC_TYPE)
        #

        # adding types w/ suffixes that include the number in each type (so I don't have
        # to add that to any legends myself). should preserve order of actual type part
        # of the labels (e.g. 'ab' -> 'ab (3735)'
        # TODO rename kc_types -> kc_types_and_count_strs?
        wAPLKC_with_types[KC_TYPE] = kc_types
        wKCAPL_with_types[KC_TYPE] = kc_types
        if prat_boutons and per_claw_pn_apl_weights:
            wAPLPN_with_types[KC_TYPE] = kc_types
            wPNAPL_with_types[KC_TYPE] = kc_types

        # TODO care to dupe for apl<>pn stuff?
        wAPLKC_max_weight_by_type = wAPLKC_with_types.groupby(KC_TYPE).weight.max()
        wKCAPL_max_weight_by_type = wKCAPL_with_types.groupby(KC_TYPE).weight.max()

        # TODO prob delete
        # TODO TODO at least would need to actually check unknown type here, rather
        # than just checking if any type has 0 weight (?)
        wAPLKC_0weight_types = wAPLKC_max_weight_by_type == 0
        wAPLKC_unknown_type_has_0weight = False
        if wAPLKC_0weight_types.any():
            assert len(wAPLKC_max_weight_by_type.index[wAPLKC_0weight_types]) == 1
            assert wAPLKC_max_weight_by_type.index[wAPLKC_0weight_types].str.startswith(
                'unknown (').all()
            wAPLKC_unknown_type_has_0weight = True

        wKCAPL_0weight_types = wKCAPL_max_weight_by_type == 0
        wKCAPL_unknown_type_has_0weight = False
        if wKCAPL_0weight_types.any():
            assert len(wKCAPL_max_weight_by_type.index[wKCAPL_0weight_types]) == 1
            assert wKCAPL_max_weight_by_type.index[wKCAPL_0weight_types].str.startswith(
                'unknown (').all()
            wKCAPL_unknown_type_has_0weight = True

        n_unknown_type = None
        if wAPLKC_unknown_type_has_0weight and wKCAPL_unknown_type_has_0weight:
            wAPLKC_unknown_type = wAPLKC_with_types[KC_TYPE].str.startswith('unknown (')
            wAPLKC_with_types = wAPLKC_with_types[~wAPLKC_unknown_type]

            wKCAPL_unknown_type = wKCAPL_with_types[KC_TYPE].str.startswith('unknown (')
            wKCAPL_with_types = wKCAPL_with_types[~wKCAPL_unknown_type]

            n_unknown_type = wAPLKC_unknown_type.sum()
            assert n_unknown_type == wKCAPL_unknown_type.sum()
        else:
            # TODO TODO but maybe still assert no NaN type? do that outside of this
            # if/else?
            assert not (wAPLKC_0weight_types.any() or wKCAPL_0weight_types.any())

        if plot_dir is not None:
            # TODO also only show present types for the other plots (in
            # connectome_wPNKC/fit_mb_model) that use kc_type_hue_order?
            #
            # otherwise type='unknown'/etc will show up in legend, without a
            # correponding histogram element (b/c no rows have that type here, and they
            # should also all be of weight 0 anyway, unless there was a bug w/ how I was
            # getting the types from wPNKC...), which is confusing
            hue_order = [
                type2type_with_count[x] for x in kc_type_hue_order
                if x in type2type_with_count
            ]

            # TODO retore if i can fix reindex_to_wPNKC for `one_row_per_claw and not
            # prat_claws` case (n_nonclaw_apl2kc_syns_per_kc currently only defined
            # there, and in separate prat_claws=True branch above)
            #if one_row_per_claw:
            if prat_claws:
                # TODO duped w/ something else?
                # TODO use one w/ counts from earlier, to also be able to use hue_order,
                # and have that in legend?
                kc2type = index.to_frame(index=False)[[KC_ID, KC_TYPE]].drop_duplicates(
                    ).set_index(KC_ID).squeeze()

                nonclaw_apl2kc_df = n_nonclaw_apl2kc_syns_per_kc.to_frame()
                nonclaw_apl2kc_df[KC_TYPE] = nonclaw_apl2kc_df.index.map(kc2type)
                nonclaw_apl2kc_df['claw_weight'] = wAPLKC_with_types.groupby(KC_ID
                    ).weight.sum()

                nonclaw_kc2apl_df = n_nonclaw_kc2apl_syns_per_kc.to_frame()
                nonclaw_kc2apl_df[KC_TYPE] = nonclaw_kc2apl_df.index.map(kc2type)
                nonclaw_kc2apl_df['claw_weight'] = wKCAPL_with_types.groupby(KC_ID
                    ).weight.sum()

            def agg_per_unit(df, unit_cols):
                agg_fns = {
                    'weight': 'sum',

                    'radius': 'mean',

                    # without the str(...) squeeze output is a str-like np.ndarray,
                    # which can not be indexed w/ [0] to get a single str (it seemed)
                    # like: array('ab (3735)', dtype=object)
                    KC_TYPE: lambda x: str(x.unique().squeeze())
                }
                agg_fns = {k: v for k, v in agg_fns.items() if k in df.columns}
                # adding other levels will typically just make the groupby return one
                # agg output per claw, even if unit_cols are [KC_ID]. no particularly
                # useful info there, and could handle separately if needed.
                return df.groupby(unit_cols).agg(agg_fns)

            # TODO TODO TODO need to add non-claw weights first, for per-KC plot i
            # wanted? (fine w/ existing plot for that, that doesn't include claw
            # weights. no, prob not?)
            # TODO TODO TODO do separate (per-KC-only) version like that, with both
            # non-claw and claw weights, per KC
            hist_versions = [
                ([KC_ID], 'per-kc'),
            ]
            # TODO or maybe only if prat_claws?
            if one_row_per_claw:
                hist_versions.append((claw_cols, 'per-claw'))

            for unit_cols, desc in hist_versions:

                unit_str = f'\n{n_kcs=}'
                if unit_cols == [KC_ID]:
                    unit_str += '\ncounts=KCs'
                else:
                    assert unit_cols == claw_cols
                    unit_str += f'\n{n_claws=}\ncounts=claws'

                # TODO for per-KC plots, disable discrete=True on these? or kde?  plots
                # kinda rough looking (for just some outputs? PN<>APL stuff or what?)
                aplkc_by_unit = agg_per_unit(wAPLKC_with_types, unit_cols)

                kws = dict()
                if one_row_per_claw and not prat_claws:
                    weight = aplkc_by_unit['weight']
                    if unit_cols == [KC_ID]:
                        int_weights = weight.round().astype(int)
                        assert pd_allclose(int_weights, weight)
                        # otherwise current assertion in plotting fn will trip
                        aplkc_by_unit['weight'] = int_weights
                    else:
                        assert not pd_allclose(weight.round().astype(int), weight)
                        # TODO warn?
                        # TODO TODO just skip plot in this case (should basically be
                        # redundant w/ per-KC anyway...)
                        kws = dict(discrete=False)

                fig, ax = _plot_connectome_raw_weight_hist(aplkc_by_unit, x='weight',
                    hue=KC_TYPE, hue_order=hue_order, **kws
                )
                ax.set_title(f'APL->KC weights{unit_str}')
                savefig(fig, plot_dir, f'wAPLKC_hist_{desc}_by-type',
                    bbox_inches='tight'
                )


                # TODO refactor
                kcapl_by_unit = agg_per_unit(wKCAPL_with_types, unit_cols)

                if one_row_per_claw and not prat_claws:
                    weight = kcapl_by_unit['weight']
                    if unit_cols == [KC_ID]:
                        int_weights = weight.round().astype(int)
                        assert pd_allclose(int_weights, weight)
                        # otherwise current assertion in plotting fn will trip
                        kcapl_by_unit['weight'] = int_weights
                    else:
                        assert not pd_allclose(weight.round().astype(int), weight)

                fig, ax = _plot_connectome_raw_weight_hist(kcapl_by_unit, x='weight',
                    hue=KC_TYPE, hue_order=hue_order, **kws
                )
                ax.set_title(f'KC->APL weights{unit_str}')
                savefig(fig, plot_dir, f'wKCAPL_hist_{desc}_by-type',
                    bbox_inches='tight'
                )

                # TODO (delete. not sure i actually care) do a version (of above two)
                # merging in non-claw weights, when unit_cols == [KC_ID]?
                #if one_row_per_claw and unit_cols == ['KC_ID']:
                #    breakpoint()

                # TODO TODO want to plot anything in non-per-claw case?
                # (can prob leave those plots to connectome_wPNKC?)
                if prat_boutons and per_claw_pn_apl_weights:
                    assert len(kws) == 0

                    # TODO refactor
                    fig, ax = _plot_connectome_raw_weight_hist(
                        agg_per_unit(wAPLPN_with_types, unit_cols), x='weight',
                        hue=KC_TYPE, hue_order=hue_order
                    )
                    ax.set_title(f'APL->PN weights{unit_str}')
                    savefig(fig, plot_dir, f'wAPLPN_hist_{desc}_by-type',
                        bbox_inches='tight'
                    )

                    fig, ax = _plot_connectome_raw_weight_hist(
                        agg_per_unit(wPNAPL_with_types, unit_cols), x='weight',
                        hue=KC_TYPE, hue_order=hue_order
                    )
                    ax.set_title(f'PN->APL weights{unit_str}')
                    savefig(fig, plot_dir, f'wPNAPL_hist_{desc}_by-type',
                        bbox_inches='tight'
                    )

            if one_row_per_claw:
                title_suffix = '\nclaw weight vs radius (from calyx center)'

                coords = wAPLKC_with_types.index.to_frame(index=False)[claw_coord_cols]
                center = coords.mean().to_numpy()
                distances = np.linalg.norm(coords.to_numpy() - center, axis=1)
                wAPLKC_with_types['radius'] = distances

                # TODO refactor to share radius calc + addition
                #
                # TODO or just use 0,0,0 for center? already centered?
                coords = wKCAPL_with_types.index.to_frame(index=False)[claw_coord_cols]
                center = coords.mean().to_numpy()
                distances = np.linalg.norm(coords.to_numpy() - center, axis=1)
                wKCAPL_with_types['radius'] = distances

                fig, ax = plt.subplots()
                # TODO violin plot (to still have hues?) (that only work if still
                # x='weight' & y='radius'?)
                # NOTE: these radius vs weight plots probably only make sense on a
                # per-claw basis, so no need to also groupby KCs and do that
                # TODO need alpha=0.3 or something? (for hue to work) try lower?
                # TODO + jitter? (don't see a dodge or jitter option for scatterplot...)
                sns.scatterplot(wAPLKC_with_types, ax=ax, x='radius', y='weight',
                    hue=KC_TYPE, alpha=0.3
                )
                ax.set_title(f'APL->KC{title_suffix}')
                savefig(fig, plot_dir, 'wAPLKC_weight_vs_radius')

                fig, ax = plt.subplots()
                # TODO some kind of distribution plot instead? 2d hist (meh)? jitter?
                sns.scatterplot(wKCAPL_with_types, ax=ax, x='radius', y='weight',
                    hue=KC_TYPE, alpha=0.3
                )
                ax.set_title(f'KC->APL{title_suffix}')
                savefig(fig, plot_dir, 'wKCAPL_weight_vs_radius')

                # TODO compare (normalize against?) all these APL<>KC plots to
                # similar types of plots from PN->KC data? (how?)
                # just plot against radius hists of all syns / [non-0-weight] claws, per
                # type?

                # TODO TODO version (at least for non-claw synapses), counting each
                # synapse instead? (would need to use earlier version of data,
                # pre-agg_synapses_to_claws, and would need to store the non-claw stuff
                # separately) (could also do a version of claw synapses, counting each
                # synapse)
                # TODO TODO version weighted by weight, for claw synapses?
                fig, ax = plt.subplots()
                # TODO want kde (vs hist?). can overlay both w/ displot or histplot(...,
                # kde=True).
                sns.kdeplot(wAPLKC_with_types[wAPLKC_with_types.weight > 0], ax=ax,
                    x='radius', hue=KC_TYPE, hue_order=hue_order
                )
                ax.set_title(f'APL>KC claw radius distribution\n'
                    'claws with 0 APL>KC weight excluded'
                )
                savefig(fig, plot_dir, 'wAPLKC_nonzero-claw_radius_by-type')

                fig, ax = plt.subplots()
                sns.kdeplot(wKCAPL_with_types[wKCAPL_with_types.weight > 0], ax=ax,
                    x='radius', hue=KC_TYPE, hue_order=hue_order
                )
                ax.set_title(f'KC>APL claw radius distribution\n'
                    'claws with 0 KC>APL weight excluded'
                )
                savefig(fig, plot_dir, 'wKCAPL_nonzero-claw_radius_by-type')


                # TODO delete?
                fig, ax = plt.subplots()
                # TODO discrete=True?
                sns.histplot(wAPLKC_with_types, ax=ax, x='radius', y='weight')
                ax.set_title(f'APL->KC{title_suffix}')
                savefig(fig, plot_dir, 'wAPLKC_weight_vs_radius_hist')

                fig, ax = plt.subplots()
                sns.histplot(wKCAPL_with_types, ax=ax, x='radius', y='weight')
                ax.set_title(f'KC->APL{title_suffix}')
                savefig(fig, plot_dir, 'wKCAPL_weight_vs_radius_hist')
                #

                # TODO TODO hists of radius of non-claw synapses by type?
                # TODO TODO would have to preserve radius (or whole non-claw subset
                # of synapse data) earlier. currently just
                # have KC -> total # non-claw synapses

                # TODO make unconditional on prat_claws if i can fix defs above in
                # prat_claws=False case
                if prat_claws:
                    nonclaw_apl2kc_df = nonclaw_apl2kc_df.reset_index()
                    nonclaw_kc2apl_df = nonclaw_kc2apl_df.reset_index()

                    fig, ax = plt.subplots()
                    # TODO care about hue_order here? would need to process to something
                    # w/o the ' (<count>)' suffices current hue_order contains (or
                    # process these kc_type values to include them)
                    sns.histplot(nonclaw_apl2kc_df, ax=ax, x='weight', hue=KC_TYPE,
                        discrete=True, alpha=0.3
                    )
                    ax.set_title('APL->KC\nnon-claw synapses, per KC')
                    savefig(fig, plot_dir, 'wAPLKC_nonclaw_syn_hist_by-type')

                    fig, ax = plt.subplots()
                    sns.histplot(nonclaw_kc2apl_df, ax=ax, x='weight', hue=KC_TYPE,
                        discrete=True, alpha=0.3
                    )
                    ax.set_title('KC->APL\nnon-claw synapses, per KC')
                    savefig(fig, plot_dir, 'wKCAPL_nonclaw_syn_hist_by-type')


                    fig, ax = plt.subplots()
                    # TODO care about hue_order here? would need to process to something
                    # w/o the ' (<count>)' suffices current hue_order contains (or
                    # process these kc_type values to include them)
                    # TODO separate 2d hist (scatterplot) for each subtype?
                    # TODO or jitter and/or lower alpha further?
                    sns.scatterplot(nonclaw_apl2kc_df, ax=ax, x='claw_weight',
                        y='weight', hue=KC_TYPE, alpha=0.3
                    )
                    ax.set_ylabel('# non-claw synapses')
                    ax.set_xlabel('# claw synapses')
                    ax.set_title('APL->KC\none row per KC')
                    savefig(fig, plot_dir, 'wAPLKC_claw_vs_nonclaw_weight')

                    fig, ax = plt.subplots()
                    sns.scatterplot(nonclaw_kc2apl_df, ax=ax, x='claw_weight',
                        y='weight', hue=KC_TYPE, alpha=0.3
                    )
                    ax.set_ylabel('# non-claw synapses')
                    ax.set_xlabel('# claw synapses')
                    ax.set_title('KC->APL\none row per KC')
                    savefig(fig, plot_dir, 'wKCAPL_claw_vs_nonclaw_weight')

                if prat_boutons and per_claw_pn_apl_weights:
                    pn2glom = wPNKC.replace(0, np.nan).stack().index.to_frame(
                        index=False)[[PN_ID, glomerulus_col]].drop_duplicates(
                        ).set_index(PN_ID).squeeze()

                    PNAPL_df = wPNAPL_for_plots.to_frame().reset_index()
                    PNAPL_df[glomerulus_col] = PNAPL_df[PN_ID].map(pn2glom)

                    APLPN_df = wAPLPN_for_plots.to_frame().reset_index()
                    APLPN_df[glomerulus_col] = APLPN_df[PN_ID].map(pn2glom)

                    pnapl_syns_per_glom = PNAPL_df[[glomerulus_col, 'weight']].groupby(
                        glomerulus_col).sum().squeeze().rename('# PN>APL synapses')
                    aplpn_syns_per_glom = APLPN_df[[glomerulus_col, 'weight']].groupby(
                        glomerulus_col).sum().squeeze().rename('# APL>PN synapses')

                    # TODO TODO TODO look more into earlier thing where not dropping
                    # claws / syns in pn<>apl data
                    # TODO delete
                    print('double check my + prat filtering of PN<>APL data (+ my '
                        'merging above)'
                    )
                    #
                    # TODO is it weird that this is true? or just have 0's for some?
                    assert pnapl_syns_per_glom.index.equals(aplpn_syns_per_glom.index)

                    apl_and_pn_syns_per_glom = pd.concat(
                        [pnapl_syns_per_glom, aplpn_syns_per_glom], axis='columns'
                    )
                    fig, ax = plt.subplots()
                    cbar_shrink = 0.2
                    viz.matshow(apl_and_pn_syns_per_glom.T, ax=ax,
                        cbar_label='# synapses', cbar_shrink=cbar_shrink
                    )
                    ax.set_title('total PN<>APL synapses per glomerulus'
                        '\ntop: PN>APL, bottom: APL>PN'
                    )
                    savefig(fig, plot_dir, 'pn_and_apl_syns_per_glom',
                        bbox_inches='tight'
                    )

                    # ipdb> (APLPN_df.weight == 0).sum()
                    # 490
                    # ipdb> (PNAPL_df.weight == 0).sum()
                    # 432
                    # ipdb> len(APLPN_df)
                    # 9867
                    # ipdb> len(PNAPL_df)
                    # 9867
                    # (shouldn't be dropping a huge amount)
                    APLPN_df = APLPN_df[APLPN_df.weight != 0]
                    PNAPL_df = PNAPL_df[PNAPL_df.weight != 0]

                    # TODO just delete this plot? almost every bouton seems to have some
                    # weights, no? or need to make another hist (on raw weights?) to
                    # make that point?
                    # TODO TODO TODO or i suppose we need separate dropping of "boutons"
                    # which have no PN>APL or APL>PN weights. what would that look like?
                    '''
                    aplpn_boutons_per_glom = APLPN_df.groupby(glomerulus_col).apply(
                        lambda x: x[BOUTON_ID].str.len().sum()
                    ).rename('# APL>PN boutons')
                    # TODO TODO TODO should add up to total # of boutons. what does this
                    # even have to do w/ APL? just compute from wPNKC [glomerulus_col] +
                    # bouton_cols, right?
                    # TODO TODO TODO try dropping duplicate [pn, bouton_ids]? matter?
                    pnapl_boutons_per_glom = PNAPL_df.groupby(glomerulus_col).apply(
                        lambda x: x[BOUTON_ID].str.len().sum()
                    ).rename('# PN>APL boutons')
                    apl_and_pn_boutons_per_glom = pd.concat(
                        [pnapl_boutons_per_glom, aplpn_boutons_per_glom], axis='columns'
                    )
                    # TODO delete
                    #breakpoint()
                    #
                    fig, ax = plt.subplots()
                    viz.matshow(apl_and_pn_boutons_per_glom.T, ax=ax,
                        cbar_label='# boutons', cbar_shrink=cbar_shrink
                    )
                    ax.set_title('total PN<>APL boutons per glomerulus'
                        '\ntop: PN>APL, bottom: APL>PN'
                        '\nexcluded any with 0 weight'
                    )
                    savefig(fig, plot_dir, 'pn_and_apl_boutons_per_glom',
                        bbox_inches='tight'
                    )
                    '''

                    # will be less than total # of claws in wPNKC, because (for each
                    # direction) we are dropping the claws w/ 0 weight.
                    aplpn_claws_per_glom = APLPN_df.groupby(glomerulus_col).apply(
                        lambda x: len(x[claw_cols].drop_duplicates())
                    ).rename('# APL>PN claws')
                    pnapl_claws_per_glom = PNAPL_df.groupby(glomerulus_col).apply(
                        lambda x: len(x[claw_cols].drop_duplicates())
                    ).rename('# PN>APL claws')
                    apl_and_pn_claws_per_glom = pd.concat(
                        [pnapl_claws_per_glom, aplpn_claws_per_glom], axis='columns'
                    )
                    fig, ax = plt.subplots()
                    viz.matshow(apl_and_pn_claws_per_glom.T, ax=ax,
                        cbar_label='# claws', cbar_shrink=cbar_shrink
                    )
                    ax.set_title('total PN<>APL claws per glomerulus'
                        '\ntop: PN>APL, bottom: APL>PN'
                        '\nexcluded any with 0 weight'
                    )
                    savefig(fig, plot_dir, 'pn_and_apl_claws_per_glom',
                        bbox_inches='tight'
                    )
                    # TODO TODO why aren't counts for these more like 20 * bouton counts
                    # (if avg 20 claws per bouton). am i not merging correctly above?
                    # doing something wrong here?
                    # (or maybe should be lower in bouton-per-glom plot? PROBABLY JUST
                    # THIS.  BOUTON COUNTS WERE PROBABLY ESSENTIALLY DUPLICATING CLAW
                    # COUNTS, with how i was merging + grouping for counts here)
                    # TODO TODO assert sum of claws is reasonable though (probably going
                    # to just delete bouton plot here, or normalize first, or else would
                    # want to do something similar there)
                    # TODO delete
                    #breakpoint()


    # TODO fill in unknown KC type (NaN here) w/ mean weight of other type? or
    # drop? (and prob do one by default) (at least it's just 80/1830 cells)
    # (what is latest fraction w/ v3 pratyush data?)

    # TODO TODO TODO actually save wAPLPN/wPNAPL (in fit_and_plot... maybe actually loop
    # over objects in param_dict, and pop all Series, and use that to replace existing
    # hardcode code for wAPLKC/wKCAPL?)
    # TODO add flags to control shape of wAPLPN/wPNAPL (length # claws make sense? #
    # glomeruli? # boutons?)?
    return wAPLKC, wKCAPL, wAPLPN, wPNAPL


# TODO doc expected input format (i.e. what are rows / cols)
def drop_silent_model_cells(responses: pd.DataFrame) -> pd.DataFrame:
    # .any() checks for any non-zero, so should also work for spike counts
    # (or 1.0/0.0, instead of True/False)
    nonsilent_cells = responses.T.any()

    # TODO maybe restore as warning, or behind a checks flag or something.
    # (now that i should be using KC_ID='kc_id' everywhere, rather than a mix of
    # 'model_kc' and 'bodyid'/similar connectome stuff, might be more realistic)
    #
    # (also works if index.name == KC_ID)
    # (commented cause was failing b/c index name was diff in natmix_data/analysis.py
    # script, though i could have changed that...)
    #assert KC_ID in nonsilent_cells.index.names

    return responses.loc[nonsilent_cells].copy()


def _get_silent_cell_suffix(responses_including_silent, responses_without_silent=None
    ) -> str:
    if responses_without_silent is None:
        responses_without_silent = drop_silent_model_cells(responses_including_silent)

    assert responses_without_silent.shape[1] == responses_including_silent.shape[1]
    n_total_cells = len(responses_including_silent)
    n_silent_cells = n_total_cells - len(responses_without_silent)

    # TODO could relax to assert >=, if ever fails
    # (same as asserting len(responses_without_silent) > 0)
    assert n_total_cells > n_silent_cells

    # TODO delete. prob no longer true (matter?)
    # titles these will be appended to should already end w/ '\n'
    #
    # TODO make sure this isn't adding one more newline than i want in plots made from
    # fit_mb_model (don't seem to be generated in current paths...)
    title_suffix = f'\n(dropped {n_silent_cells}/{n_total_cells} silent cells)'
    # TODO also return n_[total|silent]_cells? only if useful in some existing calling
    # code
    return title_suffix


# TODO move to hong2p.util?
def _single_unique_val(arr: Union[np.ndarray, pd.Series], *, exact: bool = True
    ) -> float:
    """Returns single unique value from array.

    Raises AssertionError if array has more than one unique value (including NaNs).
    """
    unique_vals = set(np.unique(arr))
    if exact:
        assert len(unique_vals) == 1, f'{unique_vals=}'
        return unique_vals.pop()
    else:
        # TODO support NaN in this branch? (prob not...)
        v0 = unique_vals.pop()
        for v in unique_vals:
            assert np.isclose(v, v0)

        return v0


# TODO TODO increase n_PCs/n_clusters here (and just set lower in wrapper specifically
# for model 0/1 spiking data)?
def cluster_timeseries(df: pd.DataFrame, *, n_PCs: int = 10, n_clusters: int = 8,
    verbose: bool = False, **kwargs) -> Optional[pd.DataFrame]:
    """
    Args:
        **kwargs: passed to `Rastermap` constructor
    """
    # TODO check again if this is actually an issue? spike plotting fn does currently
    # drop all rows like this though, and unlikely to get in any other input (unless as
    # a result of filling in missing data w/ 0)
    assert not (df == 0).all(axis=1).any()

    # rastermap call below gets unhappy if we omit .values (leaving df a DataFrame)
    arr = df.values.astype('float32')
    # TODO maybe try converting to spike times in seconds, and using
    # `rastermap.io.load_spike_times(<spike-time-npy>, <spike-cluster-npy>,
    # st_bin=100)` (or whatever it does)?
    # TODO TODO or just try re-binning my spike time matrix into one with a bin of
    # 100ms (from st_bin above)? do we ever have two spikes within one of those
    # bins? how does rastermap fn handle that? add? is output binary or not?
    assert not np.isnan(arr).any(), 'rastermap will err if input has NaN'
    assert np.isfinite(arr).all(), 'rastermap will err if input is not all finite'

    # TODO does time lag actually matter? not clearly better w/ it set vs default, when
    # using n_clusters=5, n_PCs=100 (and locality default/0, which i think are same)
    # TODO how does this compare to version w/ locality + time_lag_window set above?
    # (should be pretty much the same) prefer either? try diff (lower?)
    # n_PCs/n_clusters?
    #
    # remy came up with this calculation. not immediately clear to me (from rastermap
    # docs) that this is the appropriate value.
    # dt=0.0005 is from the olfsysm time_dt setting.
    #dt = 0.0005
    #time_lag_window_seconds = 0.1
    #time_lag_window = int(time_lag_window_seconds / dt)

    # TODO try smoothing parameters / diff binning?
    #
    # works after dropping non-responders (see scripts/test_rastermap.py for some
    # discussion on choice of parameters, including kwarg defaults in plot_spike_rasters
    # def)
    with warnings.catch_warnings():
        # this warning also corresponds to nothing being plotted
        warnings.filterwarnings('error', category=UserWarning,
            message='Singular matrix'
        )
        try:
            # TODO need to seed this? can i?
            model = Rastermap(n_PCs=n_PCs, n_clusters=n_clusters, verbose=verbose,
                **kwargs
            ).fit(arr)

        # TODO can i still repro ValueError & LinAlgError? w/ 1.0?
        # TODO TODO this warning filter work to catch singular matrix?
        except (ValueError, np.linalg.LinAlgError, UserWarning) as err:
            msg = str(err)

            # TODO delete? possible to get this just based on data, if data doesn't have
            # NaN/inf (asserted above we don't have either)?
            if isinstance(err, ValueError):
                if msg != 'array must not contain infs or NaNs':
                    raise
            #

            # TODO reword? one/both can sometimes need increasing, if already ~1-2.
            # maybe in other regimes too, but that's only case i've seen on lower end so
            # far.
            # TODO print traceback too?
            warn(f'Rastermap() failed with: {type(err).__name__}: {err}\n'
                '...try decreasing n_PCs or n_clusters?'
            )
            return None

    # TODO what is this? delete?
    # neurons x 1
    # min/max seem to range on [0, <=n_clusters] (but continuous, not sure if can equal
    # n_clusters)?
    #
    # "array, shape (n_samples, 1) embedding of each neuron / voxel"
    #
    # TODO TODO return as extra index level? or plot that way (i.e. w/ row_colors, after
    # rounding)?
    #y = model.embedding

    # TODO TODO round to nearest value [why was remy saying she rounded? just to
    # get discrete clusters out?] (values ranging from [0, n_clusters], if i understand
    # remy), and use that to order? or just order based on this? assert already ordered?
    # TODO TODO what does remy mean she does her own clustering?

    # TODO worth plotting this? does it capture the overall gestalt any better?
    # TODO TODO and if i don't plot this, would i want to do my own downsampling across
    # time at least? i assume this includes some of that?
    # TODO keep_norm_X=True useful for just getting the downsampling?
    # (no, docs say it still has first dim of len n_samples)
    #
    # "array, shape (n_samples//bin_size, n_features) normalized data binned across
    # samples (if compute_X_embedding is True)"
    #X_embedding = model.X_embedding

    # "array, shape (n_samples,) sorting along first dimension of input matrix - use
    # this to get neuron / voxel sorting"
    isort = model.isort

    isort_set = set(isort)
    assert isort_set == set(range(len(df)))
    assert len(isort_set) == len(isort)
    # TODO return model instead/too?
    return df.iloc[isort]


# TODO want diff default cmap here?
def cluster_timeseries_and_plot(df: pd.DataFrame, *, ax: Optional[Axes] = None,
    cmap: Optional[CMap] = 'gray_r', imshow_kws: Optional[KwargDict] = None,
    vmin: Optional[float] = None, vmax: Optional[float] = None, verbose: bool = False,
    **kwargs) -> Optional[Figure]:
    """
    Args:
        **kwargs: passed to `cluster_timeseries`
    """
    xlim = None
    # TODO use calculated dt to scale / set time lag params (if any) in rastermap
    # appropriately? (or any other timescale related params, if important)
    time_index = df.columns
    if time_index.dtype == float:
        assert time_index.is_monotonic_increasing
        assert not time_index.isna().any()
        xlim = (time_index.min(), time_index.max())

    clustered = cluster_timeseries(df)
    # errors/warnings will be handled inside cluster_timeseries, returning None in all
    # those cases
    if clustered is None:
        return None

    if ax is None:
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
    else:
        # TODO this what i want? just don't return anything (either in general, or in
        # this case)? just getting this so i can consistently return the relevant Figure
        # object [whether we created it or not]
        fig = ax.figure

    # TODO what is purpose of this? doc. necessary to have imshow xlim labelled in other
    # coordinates (than default [0, n])?
    extent = None
    if xlim is not None:
        # TODO need to swap top and bottom? (seems fine as is)
        top = len(df) + 0.5
        bottom = -0.5

        # TODO what to add/subtract, (if anything and) if not -0.5 (half of dt?)?
        left, right = xlim

        # https://matplotlib.org/stable/users/explain/artists/imshow_extent.html
        # exent: (left, right, bottom, top)
        # TODO use a colormap w/o white=0 to test this?
        extent = (left, right, bottom, top)

    if imshow_kws is None:
        imshow_kws = dict()

    # TODO TODO (/option to?) clustermap (instead of imshow) (for row colors at least.
    # no dendrograms)

    # TODO TODO want to downsample, if not plotting X_embedding (from model)?

    ax.imshow(clustered, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto',
        extent=extent, **imshow_kws
    )
    return fig


# for some more recent data (and maybe old too), lower n_PCs seems to make it less
# likely we get the "Singular matrix" warning (which seems to line up with nothing being
# plotted). can also get it again if we decrease n_clusters below 3. not sure if too
# many clusters alone can also produce it. still getting some of that warning down to
# n_PCs=6 (and may encounter below that). also got it w/ n_PCs=4, n_clusters=3...
# TODO TODO TODO is my output the same each time w/ current:
# one-row-per-claw_True__prat-claws_True__prat-boutons_True__connectome-APL_True__pn-apl-scale-factor_1000
# code? cause it feels like it might not be, if i need to keep tweaking this...
def plot_spike_rasters(spks: pd.DataFrame, *, n_PCs: int = 4, n_clusters: int = 3,
    **kwargs) -> Optional[Figure]:
    # TODO also silence warnings about not finding enough clusters, at least when
    # verbose=False?
    """
    Args:
        spks: of shape (#-neurons, #-timepoints), with values all 0 or 1

        **kwargs: passed to `cluster_timeseries_and_plot`
    """
    # ipdb> spks.shape
    # (1830, 5500)
    # ipdb> set(spks.flat)
    # {0.0, 1.0}

    # seems to help avoid some rastermap errors (w/ spiking inputs)
    # TODO when refactoring, leave this in wrapper to handle spiking data, but still
    # assert (in called timeseries clustering fn) that data has no rows w/ all 0
    spks = spks[(spks != 0).any(axis=1)]
    if len(spks) == 0:
        raise ValueError('no responders in input!')

    return cluster_timeseries_and_plot(spks, n_PCs=n_PCs, n_clusters=n_clusters,
        # TODO (delete) actually need +.5 on vmax for spiking data to plot well? just
        # let inner fn (or mpl) pick vmin/vmax if not (prob not anymore now that
        # plotting actual data instead of X_embedded? or want to plot the embedding
        # instead?)
        #vmin=0.0, vmax=1.5,
        **kwargs
    )


# TODO use in other places too (sens analysis dir naming / similar?)
def weight_debug_suffix(param_dict: Dict[str, Any]) -> str:
    # TODO TODO want to support non-vector weights for any of this?
    wAPLKC = param_dict['wAPLKC'].mean()
    wKCAPL = param_dict['wKCAPL'].mean()
    # TODO refactor to share .2g format in a var
    # TODO use scientific notation instead?
    suffix = f'\nmean weights: wAPLKC={wAPLKC:.2g} wKCAPL={wKCAPL:.2g}'
    if 'wAPLPN' in param_dict:
        assert 'wPNAPL' in param_dict
        wAPLPN = param_dict['wAPLPN'].mean()
        wPNAPL = param_dict['wPNAPL'].mean()
        suffix += f' wAPLPN={wAPLPN:.2g} wPNAPL={wPNAPL:.2g} '

    # TODO put behind flag?
    if 'odor_stats' in param_dict:
        odor_stats = param_dict['odor_stats']
        # rows = odors, columns = different variables tracked and set in olfsysm
        # sim_KC_layer. to add new ones, have to manually define in C++ + thread thru
        # binding code.
        assert isinstance(odor_stats, pd.DataFrame)
        for x in odor_stats.columns:
            # TODO too many lines? condense?
            #
            # averaging over odors, for each stat
            suffix += '\n{x}={odor_stats[x].mean():.2g}'

    return suffix


# TODO delete all these? or re-organize? want to minimize how much mb_model stuff
# assumes a certain output folder structure
#
# TODO also use for wPNKC(s)? anything else?
data_outputs_root = Path('data')
hallem_csv_root = data_outputs_root / 'preprocessed_hallem'
hallem_delta_csv = hallem_csv_root / 'hallem_orn_deltas.csv'
hallem_sfr_csv = hallem_csv_root / 'hallem_sfr.csv'
#

# TODO delete? the series is just a hack to support one case i believe (one-row +
# no-connectome-apl)
#
# Either a pd.Series (KC/claw/bouton length, as appropriate), with matching index levels
# to what `connectome_wPNKC` and/or `connectome_APL_weights` would spit out for
# corresponding variable, or a float to scale the same outputs of those functions.
# If None, will be chosen inside `fit_mb_model`.
Weights = Optional[Union[float, pd.Series]]
#

# TODO delete
#_seen_plot_dirs = set()
#
_seen_olfsysm_logs = set()
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
# TODO check new default for tune_on_hallem didn't break any of my existing calling code
def fit_mb_model(orn_deltas: Optional[pd.DataFrame] = None, sim_odors=None, *,
    tune_on_hallem: bool = False, pn2kc_connections: str = 'hemibrain',
    use_connectome_APL_weights: bool = False, weight_divisor: Optional[float] = None,
    prat_claws: bool = False, prat_boutons: bool = False,
    # TODO change default to this to True? or have this be default behavior w/
    # prat_boutons but nothing else? shouldn't need 4 flags for only 3 distinct possible
    # behaviors... prat_boutons currently needs one of these other flags to be True
    per_claw_pn_apl_weights: bool = False,
    #
    add_PNAPL_to_KCAPL: bool = False, replace_KCAPL_with_PNAPL: bool = False,
    dist_weight: Optional[str] = None, _wPNKC: Optional[pd.DataFrame] = None,
    _wAPLKC: Optional[pd.Series] = None, _wKCAPL: Optional[pd.Series] = None,
    one_row_per_claw: bool = False, allow_net_inh_per_claw: bool = False,
    n_claws: Optional[int] = None, drop_multiglomerular_receptors: bool = True,
    drop_receptors_not_in_hallem: bool = False, seed: int = 12345,
    target_sparsity: Optional[float] = None,
    target_sparsity_factor_pre_APL: Optional[float] = None,
    max_iters: Optional[int] = None, sp_acc: Optional[float] = None,
    # TODO TODO change default sp_lr_coeff here (or in olfsysm?)? will need to restore
    # old value (=1) for some repro tests to continue passing
    sp_lr_coeff: Optional[float] = None, hardcode_initial_sp: bool = False,
    # TODO delete
    #pn_apl_scale_factor: float = 1.0,
    #
    APL_coup_const: Optional[float] = None,
    Btn_separate: bool = False, Btn_num_per_glom: int = 10,
    _use_matt_wPNKC: bool = False,
    drop_kcs_with_no_input: bool = True, _drop_glom_with_plus: bool = True,
    _add_back_methanoic_acid_mistake=False, equalize_kc_type_sparsity: bool = False,
    ab_prime_response_rate_target: Optional[float] = None,
    homeostatic_thrs: bool = False,
    fixed_thr: Optional[Union[float, np.ndarray]] = None,
    wAPLKC: Weights = None, wKCAPL: Weights = None,
    wAPLPN: Optional[float] = None, wPNAPL: Optional[float] = None,
    # pn_claw_to_APL: if it's true, in sim_KC_layer, we take pn_t to calculate
    # claw_to_apl drive instead of inheriting from KC_acitivity
    pn_claw_to_APL: bool = False, n_claws_active_to_spike: Optional[int] = None,
    #
    # TODO TODO TODO replace return_dynamics w/ save_dynamics, then use new olfsysm code
    # (to save .npy files for everything) and alongside save parquet(+csv) indices from
    # python in here
    print_olfsysm_log: Optional[bool] = None, return_dynamics: bool = False,
    plot_dir: Optional[Path] = None, make_plots: bool = True,
    plot_example_dynamics: bool = False, title: str = '',
    drop_silent_cells_before_analyses: bool = drop_silent_model_kcs,
    repro_preprint_s1d: bool = False, return_olfsysm_vars: bool = False,
    # TODO TODO implement
    # TODO rename 'broadcell_...'? (actually just rename to something generic, then add
    # other kwargs to control what defines the cells to boost)
    # TODO TODO TODO add other options for defining cells to boost APL on. want
    # multiresponders as one option [could keep them defined by mask saved by
    # natmix_data/analysis.py], but also want options for:
    # 1) KCs w/ more community PN input (or more input from some input set of
    #    glomeruli?)
    # 2) KCs w/ more input from periphery
    multiresponder_APL_boost: Optional[float] = None,
    _multiresponder_mask: Optional[pd.Series] = None,
    boost_wKCAPL: Literal[False, True, 'only'] = False,
    # TODO this return signature still accurate?
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], ParamDict]:
    # TODO doc point of sim_odors. do we need to pass them in (not typically, no)?
    # (even when neither tuning nor running on any hallem data?)
    # TODO does matt's code support float wPNKC? can i just directly normalize wPNKC and
    # pass that in, rather than dealing w/ weight_divisor=<float>?
    # TODO TODO have responses [/spike_counts] returned w/ row indices using same
    # connectome cell IDs are wPNKC row index ('bodyid' as called in hemibrain data prat
    # assembled) (or have wPNKC also use same sequential int index)
    # TODO allow passing in wPNKC matrix directly? mainly thinking of using for tests
    # now... would have to override pn2kc_connections (or take as option to that param)
    # TODO TODO doc equalize_kc_types
    """
    Args:
        orn_deltas: dataframe of shape (# glomeruli, # odors). values should be in units
            of change in spike deltas (`model_mb_responses` will do this scaling for
            you). Input should only contain one value per (glomerulus, odor) pair, so
            take a mean across flies (if applicable) before passing.

            Odor (column) index should also not have multiple repeats of the same odor.
            Odor strings (in at least one index level) should be in format like:
            '<odor-name-abbreviation> @ <log10-conc>', e.g. 'eb @ -3'.
            Odor index may also have a 'panel' level, to group odors by experiment,
            though output `responses` and `spike_counts` currently do not preserve
            this level [could implement].

            Only glomeruli in intersection of connectome (hemibrain, unless otherwise
            specified) and Task et al 2022 glomeruli will be included in model.
            Additional input glomeruli will be dropped, and any of these glomeruli
            missing from input will have their spike-change-deltas imputed as 0.
            Glomeruli will use SFRs reported in Hallem, when cognate OR is in Hallem
            dataset, or mean Hallem SFR otherwise.

        target_sparsity: target mean response rate (across tuned odors, which is all
            odors by default). By default, model is only tuned until mean response
            rate is within +/- 10% (sp_acc) of target. If passed, `fixed_thr` and
            `wAPLKC` should not be.

        fixed_thr: added to each KCs spontaneous firing rate (SFR) to produce the spike
            threshold. `olfsysm` has options to not add this, but not currently exposed
            by this wrapper. expected that only either this and `wAPLKC` OR
            `target_sparsity` are specified.

        wAPLKC: weight from APL to every KC. `wKCAPL` defined from this, unless both are
            passed. If `use_connectome_APL_weights=True`, used to set
            `rv.kc.wAPLKC_scale` instead (which is multiplied by
            `connectome_APL_weights` output, that has been scaled so its mean is 1).

        wKCAPL: used in same manner of `wAPLKC`. If this is passed, `wAPLKC` must
            also be non-None. If only `wAPLKC` is passed, this is defined from it (as
            `wAPLKC` / <#-of-KCs>).

        wAPLPN, wPNAPL: similar to corresponding KC weights, but can also be of length #
            boutons if `per_claw_pn_apl_weights=False` (othewise length claws). requires
            `prat_boutons=True`

        use_connectome_APL_weights: if True, `connectome_APL_weights` called (with
            `connectome=pn2kc_connections`) to set `rv.kc.w[APLKC|KCAPL]`.
            `connectome_APL_weights` output is scaled to have a mean of 1 before being
            used to set these model variables.

        pn2kc_connections: string specifying method for forming PN->KC connection
            matrix. Must be one of `connectome_options` or `variable_n_claw_options`.

            If among the former, passed directly to `connectome_wPNKC` `connectome=`,
            otherwise `connectome_wPNKC` called with `connectome='hemibrain'`.

        weight_divisor: passed to `connectome_wPNKC` (only relevant for connectome
            `pn2kc_connections` options)

        drop_kcs_with_no_input|_use_matt_wPNKC: passed to `connectome_wPNKC`

        _drop_glom_with_plus: passed to `connectome_wPNKC` and `connectome_APL_weights`

        seed: if using one of `pn2kc_connections` options that uses RNG to generate the
            PN->KC weight matrix (e.g. 'uniform', 'hemidraw', NOT 'hemibrain'), this
            seeds that RNG. The model (as configured by default, and as presented by
            this wrapper) is otherwise deterministic. NOTE: seeded by default.

        n_claws: for relevant `pn2kc_connections` (e.g. 'uniform', 'hemidraw', NOT
            'hemibrain') this sets that number of PNs each KC draws (i.e. # of claws for
            each KC)

        return_dynamics: if True, returns all available model internals, all as new
            elements of already-returned `param_dict`. This includes model ORN & PN
            firing rates, KC membrane potentials, KC spikes, APL membrane potential, and
            mean current from KCs to the APL, and potentially some other variables.

        plot_dir: if passed, will save some model-internals plots under this directory
            (which should be unique across all calls, within one run), and will copy the
            `olfsysm` log to `plot_dir / 'olfsysm_log.txt'` (log file will have suffix
            for seed if variable # of claws)

        drop_silent_cells_before_analyses: only relevant if `make_plots=True`

        title: (internal use only) only used if plot_dir passed. used as prefix for
            titles of plots.

        repro_preprint_s1d: (internal use only) whether to add fake odors + return data
            to allow reproduction of preprint figure S1D (showing model response rates
            to fake CO2, fake ms, and real eb)

        boost_wKCAPL: if boosting APL via `multiresponder_APL_boost=<float>`, will only
            boost `wAPLKC` if this is False.

            If True, will boost both `wAPLKC` and `wKCAPL` (by the same
            `multiresponder_APL_boost` factor).

            If 'only', will NOT boost `wAPLKC` and will only boost `wKCAPL` for these
            cells.

    Returns:
        responses: `spike_counts`, but binarized so any amount of spikes = True, else
            False.

        spike_counts: dataframe of shape (# KCs, # odors). If input had 'panel' level in
            odor (column) index, this currently does not preserve that level.
            If `repro_preprint_s1d=True` (which is NOT the default), extra
            (synthetic) odors for that plot will be included in output.

        wPNKC: dataframe of shape (# KCs, # glomeruli).

            Currently, I've only tested Matt's code with integer values, which can be
            interpreted as # of claws between PNs (of each glomerulus) to each KC.

        param_dict: dict containing tuned parameters (e.g. 'fixed_thr', 'wAPLKC'),
            certain model intermediates (e.g. 'kc_spont_in'), and certain parameters
            relevant for reproducibility (e.g. 'sp_acc', 'max_iters').

            If `use_connectome_APL_weights=True`, output will have `wAPLKC_scale` and
            `wKCAPL_scale`, which are scalars that could be used as input to `wAPLKC` /
            `wKCAPL` kwargs on another `fit_mb_model` call, where they will then also be
            interpreted to scale unit-mean connectome APL weight vectors. The other call
            should have `use_connectome_APL_weights=True` as well.
    """
    if _wPNKC is not None:
        # just a hacky way to check pn2kc_connections is unset (== default 'hemibrain',
        # unless i move default out of kwarg def to be able to detect unset more easily)
        assert pn2kc_connections == 'hemibrain'

    if _wAPLKC is not None:
        assert _wKCAPL is not None, '_wKCAPL and _wAPLKC must be passed together'
        assert use_connectome_APL_weights
        assert pn2kc_connections == 'hemibrain'
    else:
        assert _wKCAPL is None, '_wKCAPL and _wAPLKC must be passed together'

    if pn2kc_connections not in pn2kc_connections_options:
        raise ValueError(f'{pn2kc_connections=} not in {pn2kc_connections_options}')

    if pn2kc_connections == 'caron':
        # TODO support (isn't this default olfsysm behavior?)? may need for comparisons
        # to ann's model (but hopefully there's a determinstic version of her model,
        # like our 'hemibrain', if i really care about that? not sure i could get same
        # RNG wPNKC in Matt's code vs hers...)?
        raise NotImplementedError

    # TODO rename? there is a fixed number of claws, just that we can set them w/
    # n_claws for these models, as opposed to wPNKC determining it (from whatever
    # connectome) in other cases.
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

    if not one_row_per_claw:
        if APL_coup_const is not None:
            raise ValueError('non-None APL_coup_const only valid if '
                'one_row_per_claw=True'
            )

    if prat_boutons and not per_claw_pn_apl_weights:
        # bouton length wPNAPL and wAPLPN. currently only tested in this case, and some
        # code below would need to be changed otherwise.
        assert use_connectome_APL_weights, 'not supported. code below may not work.'
        # probably will need this too, but asserting now for simplicity (should also
        # imply one_row_per_claw)
        assert prat_claws
        assert one_row_per_claw
    else:
        assert wPNAPL is None
        assert wAPLPN is None

    # TODO rename hallem_input to only_run_on_hallem (or something better)?
    hallem_input = False
    if orn_deltas is None:
        hallem_input = True
        # TODO just load orn_deltas here?
    else:
        # TODO delete
        orn_deltas = orn_deltas.copy()
        #

        # TODO switch to requiring glomerulus_col (and in the one test that passes hallem
        # as input explicitly, process to convert to glomeruli before calling this fn)?
        valid_orn_index_names = ('receptor', glomerulus_col)
        if orn_deltas.index.name not in valid_orn_index_names:
            raise ValueError(f"{orn_deltas.index.name=} not in {valid_orn_index_names}")

        # TODO delete this path? shouldn't really be used...
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
            orn_deltas.index.name = glomerulus_col

            # should drop any input glomeruli w/ '+' in name (e.g. 'DM3+DM5')
            orn_deltas = handle_multiglomerular_receptors(orn_deltas,
                drop=drop_multiglomerular_receptors
            )
        #

        # TODO if orn_deltas.index.name == glomerulus_col, assert all input are in
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

    # see comment in olfsysm.cpp for meaning of this. may need true to reproduce certain
    # outputs, but otherwise want false.
    # TODO also warn in python, if this is true (similar to what is logged in olfsysm
    # now)
    mp.kc.hardcode_initial_sp = hardcode_initial_sp

    # may or may not care to relax this later
    # (so that we can let either be defined from one, or to try varying separately)
    # TODO TODO ever want to allow at least PN<>APL weights to be set by themselves?
    # and maybe even arbtirary subsets of the 4, so long as remaining can still be
    # tuned? or interpret as scale factors to apply to initial values before tuning or
    # something (may need diff param for this to avoid ambiguity...)?
    if wAPLKC is None and any(x is not None for x in (wKCAPL, wAPLPN, wPNAPL)):
        raise NotImplementedError('other weights can currently only be specified if '
            'wAPLKC is too'
        )

    if fixed_thr is not None:
        # TODO TODO (still an issue?) probably still allow non-None target_sparsity if
        # there is vector fixed_thr (why?)? (currently just also hardcoding wAPLKC from
        # call in test_vector_thr)
        assert target_sparsity is None
        assert target_sparsity_factor_pre_APL is None
        #
        assert wAPLKC is not None, 'for now, assuming both passed if either is'

        if prat_boutons:
            if not per_claw_pn_apl_weights:
                # currently will also assume we have both of these in this case
                assert wAPLPN is not None

            # TODO do i even want to support skipping tuning in
            # per_claw_pn_apl_weights=True cases? (accept on wAPLKC probably? need to do
            # anything special for that below? don't think so?)
            else:
                assert wAPLKC is not None

        # TODO still support varying apl activity here? (for a limited sens analysis)
        assert not homeostatic_thrs

        # TODO need to support int type too (in both of the two isinstance calls below)?
        # isinstance(<int>, float) is False
        if isinstance(fixed_thr, float):
            mp.kc.fixed_thr = fixed_thr
        else:
            # NOTE: will set rv.kc.thr below in this case, and mp.kc.fixed_thr should
            # not be used
            # TODO check olfsysm actually not using fixed_thr in this case
            # TODO TODO replace w/ Series (indexed same as other things w/ KCs)
            # (currently only used in limited test vector thr code, and maybe for some
            # attempts at changing multiresponder responses in kiwi/control data)
            assert isinstance(fixed_thr, np.ndarray)
            # TODO assert length matches other relevant dimensions (and if i change
            # type to Series from ndarray, also check metadata matches?)

        # except for "homeostatic" variants, Ann also sets "KC thresholds to a fixed
        # amount above [the time-averaged spontaneous PN input they receive]"
        # (from her thesis)
        mp.kc.add_fixed_thr_to_spont = True

        # actually do need this. may or may not need thr_type='fixed' too
        mp.kc.use_fixed_thr = True
        mp.kc.thr_type = 'fixed'
    else:
        if not homeostatic_thrs:
            mp.kc.thr_type = 'uniform'
        else:
            assert not equalize_kc_type_sparsity

            mp.kc.thr_type = 'hstatic'

            mp.kc.add_fixed_thr_to_spont = False
            mp.kc.use_fixed_thr = False

            # may or may not need this
            mp.kc.use_homeostatic_thrs = True

    if target_sparsity is not None:
        assert wAPLKC is None and wKCAPL is None
        # TODO TODO let these be set anyway?
        assert wAPLPN is None and wPNAPL is None
        #
        mp.kc.sp_target = target_sparsity

    if sp_acc is not None:
        mp.kc.sp_acc = sp_acc

    # TODO TODO TODO delete hack (fix in general [in terms of step size
    # calculation?] or change [default?] learning rate? change initial scales?
    # ratio of PN and KC APL scales?
    # TODO only do if tuning APL weights / thrs (would cause some minor issues if trying
    # to repro outputs w/ fixed thr / APL weights, which i'm not currently)
    if prat_boutons and not per_claw_pn_apl_weights:
        # down from 10
        #sp_lr_coeff = 2.0
        # this converge faster in
        # one-row-per-claw_True__prat-claws_True__prat-boutons_True__connectome-APL_True
        # ??? getting 48 iterations w/ 2.0 (including iterations just to decrease step)
        # 27 iterations w/ 1.5
        #sp_lr_coeff = 1.5
        # 21 iters w/ 1.3
        #sp_lr_coeff = 1.3
        #sp_lr_coeff = .087
        #
        # gets to sp=0.00998506 in one iteration in test:
        # params_fitandplot[one-row-per-claw_True__prat-claws_True__prat-boutons_True__connectome-APL_True]
        sp_lr_coeff = 6.577712
        print(f'HARDCODING {sp_lr_coeff=} (for prat_boutons case)!')
        # this one we could set more broadly w/o changing output, but it would trip
        # some test_fitandplot_repro tests, at least unless i add this param to an
        # exclude list (which i could, as long as max not reached?)
        max_iters = 200
        print(f'HARDCODING {max_iters=} (for prat_boutons case) (matters less than '
            'learning rate, unless it is reached)!'
        )
        #
    #
    if sp_lr_coeff is not None:
        mp.kc.sp_lr_coeff = sp_lr_coeff

    if max_iters is not None:
        mp.kc.max_iters = max_iters

    # target_sparsity_factor_pre_APL=2 would preserve old default behavior, where KC
    # threshold set to achieve 2 * sp_target, then APL tuned to bring down to sp_target
    if target_sparsity_factor_pre_APL is not None:
        # since APL should only be able to decrease response rate from where we set it
        # by picking KC spike threshold
        assert target_sparsity_factor_pre_APL >= 1

        if target_sparsity is not None:
            sp_target = target_sparsity
        else:
            # should be the olfsysm default (get from mp.kc?)
            sp_target = .1
        assert sp_target * target_sparsity_factor_pre_APL <= 1.0
        del sp_target

        mp.kc.sp_factor_pre_APL = target_sparsity_factor_pre_APL

        # TODO (move to test_mb_model.py) test that if this is 1.0, then APL is
        # kept off (or will one iteration of tuning loop still happen + change things?),
        # or at least doesn't change responses/spike_counts
        # TODO what's max value of this (min that forces threshold to do nothing,
        # with only APL bringing activity down? or does that not max sense?) maybe i
        # should change olfsysm to use something with a more sensible max (so i can go
        # between two extremes of all-threshold vs all-APL more easily)?
        # (probably just `1 / target_sparsity`? maybe accounting for how `sp_acc` would
        # factor into that)
        # TODO add unit test that my `1 / target_sparsity` guess above correct

    # NOTE: I committed olfsysm/hc_data.csv under al_analysis/data, since I couldn't
    # find a nice mechanism to install that CSV as part of olfsysm setup. This should be
    # the same as the olfsysm CSV.
    hc_data_csv = repo_root / 'data/hc_data.csv'
    # get crypic `ValueError: stod` in `osm.load_hc_data` below, if this doesn't exist
    assert hc_data_csv.exists()

    # just assuming olfsysm is at the path I would typically clone it to. this check
    # isn't super important. just establishing that the hc_data.csv committed in this
    # repo matches where we copied it from.
    olfsysm_repo = Path('~/src/olfsysm').expanduser()
    if olfsysm_repo.exists():
        olfsysm_hc_data_csv = olfsysm_repo / 'hc_data.csv'
        assert olfsysm_hc_data_csv.exists()

        # checking files are exactly the same
        unchanged = filecmp.cmp(hc_data_csv, olfsysm_hc_data_csv, shallow=False)
        assert unchanged

        del olfsysm_hc_data_csv
    del olfsysm_repo

    # TODO can i change to not load this when i'm just passing in my own data? necessary
    # even then? (add unit test to check?) (shouldn't be, if we are ourselves setting or
    # not using the 3 variables mentioned in comment below)
    #
    # initializes:
    # - mp.orn.data.delta to (#-hallem-gloms, #-hallem-odors)
    # - mp.orn.data.spont to (#-hallem-gloms, 1)
    # - mp.kc.cxn_distrib to values currently hardcoded inside load_hc_data
    #   (from Caron, presumably?)
    #
    # load_hc_data does not use the supplemental-odor section of the CSV, despite them
    # being present in the olfsysm CSV (and this one we copied+committed to this repo).
    # just the 110 main-text odors + sfr_col.
    osm.load_hc_data(mp, str(hc_data_csv))

    hallem_orn_deltas = orns.orns(add_sfr=False, drop_sfr=False, columns=glomerulus_col).T

    checks = True
    if checks:
        # columns: [glomeruli, receptors]
        hc_data = pd.read_csv(hc_data_csv, header=[0,1], index_col=[0,1])
        # [odor "class" (int: [0, 12]), odor name (w/ conc suffix for supplemental)]
        hc_data.index.names = ['class', 'odor']
        hc_data.index = hc_data.index.droplevel('class')
        hc_data.columns.names = [glomerulus_col, 'receptor']
        hc_data = hc_data.T

        coreceptor_idx = hc_data.index.get_level_values('receptor').get_loc('33b')
        # get_loc can return other type if there isn't just one match
        assert isinstance(coreceptor_idx, int)
        hc_data.index = hc_data.index.droplevel('receptor')
        hc_data_gloms = list(hc_data.index)
        # renaming from duplicate 'dm3'
        hc_data_gloms[coreceptor_idx] = 'dm3+dm5'
        hc_data.index = pd.Index(data=hc_data_gloms, name=glomerulus_col)
        assert not hc_data.index.duplicated().any()

        # hc_data.columns[110:-1] are all the supplemental odors, which are not in
        # current hallem_orn_deltas. columns[-1] is sfr_col for both hc_data and
        # hallem_orn_deltas. first 110 should be same main-text Hallem odors (in same
        # order) for each, just formatted slightly differently for these few.
        #
        # {'2 3-butanediol': '2,3-butanediol',
        #  '2 3-butanedione': '2,3-butanedione',
        #  'ammoniumhydroxide': 'ammonium hydroxide',
        #  'ethylcinnamate': 'ethyl cinnamate',
        #  'linoleum acid': 'linoleic acid',
        #  'methanoicacid': 'methanoic acid',
        #  'nonionic acid': 'nonanoic acid',
        #  'pyretic acid': 'pyruvic acid'}
        hc_data_odor_renames = {
            k: v for k, v in zip(hc_data.columns[:110], hallem_orn_deltas.columns[:110])
            if k != v
        }
        hc_data.columns = hc_data.columns.map(lambda x: hc_data_odor_renames.get(x, x))
        assert not hc_data.columns.isna().any()
        assert not hc_data.columns.duplicated().any()

        hc_supp_odors = hc_data.columns[110:-1]
        assert not hc_supp_odors.isin(hallem_orn_deltas.columns).any()
        hc_data = hc_data.drop(columns=hc_supp_odors)

        assert hc_data.columns.equals(hallem_orn_deltas.columns)
        assert hallem_orn_deltas.index.str.lower().equals(hc_data.index)
        assert np.array_equal(hallem_orn_deltas, hc_data)

        del hc_data_csv, hc_data

    assert not hallem_orn_deltas.index.duplicated().any()
    # Or33b ('DM3+DM5' in my drosolf output, a duplicate 'dm3' in Matt's olfsysm CSV) is
    # the co-receptor for both DM3 (Or47a) and DM5 (Or85a)
    assert all(x in hallem_orn_deltas.index for x in ('DM3', 'DM5'))
    # should drop 'DM3+DM5'
    hallem_orn_deltas = handle_multiglomerular_receptors(hallem_orn_deltas,
        drop=drop_multiglomerular_receptors
    )

    # how to handle this for stuff not in hallem? (currently imputing mean Hallem sfr)
    sfr_col = 'spontaneous firing rate'
    sfr = hallem_orn_deltas[sfr_col]
    assert hallem_orn_deltas.columns[-1] == sfr_col
    hallem_orn_deltas = hallem_orn_deltas.iloc[:, :-1].copy()
    n_hallem_odors = hallem_orn_deltas.shape[1]
    assert n_hallem_odors == 110
    # TODO refactor
    hallem_orn_deltas = abbrev_hallem_odor_index(hallem_orn_deltas, axis='columns')

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
        deltas_from_csv = pd.read_csv(hallem_delta_csv, index_col=glomerulus_col)
        sfr_from_csv = pd.read_csv(hallem_sfr_csv, index_col=glomerulus_col)

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
        if data_outputs_root.is_dir():
            # (subdirectory of data_outputs_root)
            hallem_csv_root.mkdir(exist_ok=True)

            # TODO assert columns of the two match here (so i don't need to check from
            # loaded versions, and so i can only check one against wPNKC, not both)
            to_csv(hallem_orn_deltas_for_csv, hallem_delta_csv)
            to_csv(sfr_for_csv, hallem_sfr_csv)

            # TODO delete? unused
            #deltas_from_csv = hallem_orn_deltas_for_csv.copy()
            #sfr_from_csv = sfr_for_csv.copy()
            #

        # TODO warn if data_outputs_root does not exist

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

    # TODO should tune_on_hallem be set True if input is already hallem? prob?
    # (i.e. if orn_deltas not passed)

    # TODO (delete?) implement means of getting threshold from hallem input + hallem
    # glomeruli only -> somehow applying that threshold [+APL inh?] globally (and
    # running subsequent stuff w/ all glomeruli, including non-hallem ones) (even
    # possible?)
    # TODO (delete?) now that i can just hardcode the 2 params, can i make plots where i
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

    connectome = get_connectome(pn2kc_connections)

    kc_types = None
    if _wPNKC is None:
        # TODO check that nothing else depends on order of columns (glomeruli) in these
        # (add a unit test permuting columns via _wPNKC kwarg, and delete this comment?)

        shared_wPNKC_kws = dict(
            connectome=connectome,
            drop_kcs_with_no_input=drop_kcs_with_no_input,
            _drop_glom_with_plus=_drop_glom_with_plus,
        )

        if one_row_per_claw:
            # TODO delete? or still need? (do still seem to need, at least for
            # tests_btn_expansion. maybe also prat_claws=True stuff typically?)
            #'''
            # initially, Tianpei had hardcoded these mp.kc.* values in his olfsysm
            # compilation, but want to keep old values
            old_max_iters = mp.kc.max_iters
            old_sp_lr_coeff = mp.kc.sp_lr_coeff
            # TODO assert old values actually were both 10 (depends on me reverting
            # olfsysm correctly and recompiling)

            default_change_msg = (
                'hardcoding new defaults for one_row_per_claw=True case:\n'
            )
            defaults_changed = False
            # was default of 10 before
            if max_iters is None:
                mp.kc.max_iters = 10
                default_change_msg += f'{mp.kc.max_iters=} (was {old_max_iters})\n'
                defaults_changed = True

            # TODO TODO fix underlying issue? still seems this test was recently
            # passing with sp_lr_coeff=10 (somewhere in the 20bd55c73 - 66eff37e0 range,
            # or at least at one point [without changing these] uncommitted nearby end
            # of that range). but after adding `bouton_dynamics` code, and trying to
            # homogenize bouton handling across Tianpei/Prat cases, have now
            # (2026-02-12) felt the temptation to decrease this again (b/c not
            # converging within max iterations, often b/c oscillating and wasting
            # iterations to decrease step size).
            # compare steps to before? or was it also reliant on hardcoding intial
            # sparsity?
            #
            # was default of 10.0 before
            # it seems he had at one point also tried 1.0, but i'm assuming 1.5 is the
            # latest value he intended to use (why?)
            if sp_lr_coeff is None:
                # TODO TODO TODO did 10 ever really make sense w/o the
                # hardcode_initial_sp code (or what is not confined to the =true path of
                # that flag)? it
                # would probably force a smaller delta on first step
                # TODO TODO TODO revert to 10? (esp since i'm already passing in in
                # test_btn_expansion test that i'm first noticeing new-code issues with)
                mp.kc.sp_lr_coeff = 10.0
                default_change_msg += f'{mp.kc.sp_lr_coeff=} (was {old_sp_lr_coeff})\n'
                defaults_changed = True

            # TODO also want these for prat_claws=True?
            if defaults_changed:
                warn(default_change_msg)

            # these are only used for Tianpei's outputs that had grouped synapses into
            # KC claws. not used for (more typical) case, where we are loading Prat's
            # outputs where he did the same.
            synapse_con_path = None
            synapse_loc_path = None
            if not prat_claws:
                # TODO move all this inside relevant branch of connectome_wPNKC?
                synapse_data_dir = from_prat / '2025-04-24'
                assert synapse_data_dir.is_dir()

                # TODO TODO what actually generated these? trying to find how he's
                # defining the 'weight' column? is one row a synapse or not? if so, then
                # what is the weight?
                synapse_con_path = synapse_data_dir / 'PN2KC_Connectivity.csv'
                synapse_loc_path = synapse_data_dir / 'PN2KC_Synapse_Locations.csv'
                assert synapse_con_path.exists()
                assert synapse_loc_path.exists()
                #

            wPNKC = connectome_wPNKC(
                # TODO what am i actually ruling out here and why?
                # TODO move the pn2kc_connections check into connectome_wPNKC itself?
                plot_dir=(
                    plot_dir
                    if (make_plots and pn2kc_connections in connectome_options)
                    else None
                ),

                prat_claws=prat_claws,
                prat_boutons=prat_boutons,
                dist_weight=dist_weight,

                # TODO move these both into the prat_claws=False + one-row-per-claw=True
                # branch of connectome_wPNKC (-> remove from kwargs to this fn)?
                synapse_con_path=synapse_con_path,
                synapse_loc_path=synapse_loc_path,

                Btn_separate=Btn_separate,
                Btn_num_per_glom=Btn_num_per_glom,

                **shared_wPNKC_kws
            )
            # TODO delete
            # TODO TODO try to recompute claw2bouton, for use in model dynamics plotting
            # below (does take a long time [and probably memory]. return instead?
            # compute some other way ad-hoc for plotting [will try this first]?)
            #breakpoint()
            #
            if 'compartment' in wPNKC.index.names:
                claw_comp = wPNKC.index.get_level_values('compartment').to_numpy(
                    np.int32, copy=True
                )
                assert claw_comp.size == len(wPNKC), 'compartment length mismatch'

            # TODO prob delete
            if plot_dir is not None:
                # TODO need to handle multiple saves to same path?
                to_csv(wPNKC.reset_index(), plot_dir / 'test_spatial_wPNKC.csv',
                    index=True
                )
            #
        else:
            wPNKC = connectome_wPNKC(
                weight_divisor=weight_divisor,
                _use_matt_wPNKC=_use_matt_wPNKC,

                # TODO TODO doc why we even need to call connectome_wPNKC in
                # pn2kc_connections like 'uniform' (and is _drop_glom_with_plus relevant
                # there?) (will check it matters w/ uniform repro test i'm working on)
                # (or is it just for hemidraw really?)
                #
                # disabling plot_dir here b/c models that are run w/ multiple seeds
                # (handled in code that calls this fn, not within here), would end up
                # trying to make the same plots for each seed (which would trigger
                # savefig assertion that we aren't writing to same path more than once)
                plot_dir=None if variable_n_claws else plot_dir,

                Btn_separate=Btn_separate,
                Btn_num_per_glom=Btn_num_per_glom,

                **shared_wPNKC_kws
            )
    else:
        wPNKC = _wPNKC.copy()

    if not one_row_per_claw:
        # no claw_index needed here
        kc_index = wPNKC.index.copy()
    else:
        claw_index = wPNKC.index.copy()

        # TODO delete? (/ move def below, near where used?)
        if 'compartment' not in claw_index.names:
            claw_comp = np.zeros(len(wPNKC), dtype=np.int64)
        #

        # TODO TODO rewrite to not require new metadata variables (new index levels in
        # wPNKC) to always require separately hardcoding those possible variables here
        # (drop all but a certain set of variables for KCs instead? anything that isn't
        # 1:1 w/ KC ID?)
        # TODO was this causing the RunVars init error when initially adding pn_id +
        # BOUTON_ID levels to wPNKC index? (yes)
        to_drop = [x for x in [CLAW_ID] + claw_coord_cols + ['compartment',
                'anatomical_claw', PN_ID, BOUTON_ID
            ] if x in claw_index.names
        ]
        assert len(to_drop) > 0
        kc_index = claw_index.droplevel(to_drop).drop_duplicates()

        # TODO also (/instead) assert kc_index is of length equal to # of KCs?
        assert len(claw_index) > len(kc_index), ('some metadata variable(s) in '
            'claw_index still need to be dropped, so that drop_duplicates() brings '
            'length down to # of KCs'
        )

        if pn_claw_to_APL:
            mp.kc.pn_claw_to_APL = True

        if APL_coup_const is not None:
            # 0 = distinct comparments, but uncoupled
            # NOTE: coupling implementation (>0) currently (2026-01-19) doesn't make
            # sense
            assert APL_coup_const == -1 or APL_coup_const >= 0
            mp.kc.apl_coup_const = APL_coup_const

            # TODO implement + delete
            if not allow_net_inh_per_claw:
                warn('setting allow_net_inh_per_claw=True (away from default), since '
                    'olfsysm currently only supports =False in the '
                    '`APL_coup_const == -1` (None, as passed to `fit_mb_model`) case!\n'
                    'output may not make the most sense!'
                )
                allow_net_inh_per_claw = True
                # TODO delete
                #raise NotImplementedError('olfsysm currently does not have the '
                #    'allow_net_inh_per_claw=False code in the APL_coup_const != -1 '
                #    'path. olfsysm would raise error on run_KC_sims(mp, rv, True) call,'
                #    ' from a `check` statement inside olfsysm which looks for this same'
                #    ' set of parameters.'
                #)

        if n_claws_active_to_spike is not None:
            assert n_claws_active_to_spike > 0
            assert type(n_claws_active_to_spike) is int
            mp.kc.n_claws_active_to_spike = n_claws_active_to_spike

    # TODO check kc_index here instead of wPNKC.index?
    if KC_TYPE in wPNKC.index.names:
        # TODO TODO at least doc why we still need to drop this level (only to
        # re-add later), or remove
        # TODO delete? currently tempted to assign this kc_type col back into a
        # level of wPNKC index below (am doing that now) (and want all outputs w/ a
        # KC index to have them consistent)
        kc_types = kc_index.get_level_values(KC_TYPE)
        kc_index = kc_index.droplevel(KC_TYPE)
        # TODO delete?
        # claw_index = claw_index.droplevel(KC_TYPE)

        # TODO TODO even if we do need to drop in kc_index, do we need to drop in
        # wPNKC.index? either way, all code should probably be changed to never need to
        # drop this level
        if not one_row_per_claw:
            wPNKC.index = kc_index.copy()

        # TODO delete / restore similar, that also works in one-row-per-claw case
        #assert wPNKC.index.names == [KC_ID]

    # TODO TODO is there some code currently behind this flag, that still needs to
    # run (at least, so long as wPNKC still has boutons in columns, and hasn't been
    # condensed down. all of it? need new flag? check?)
    bouton_dynamics = (Btn_separate or
        (prat_boutons and not (add_PNAPL_to_KCAPL or replace_KCAPL_with_PNAPL))
    )
    # TODO rename? boutons_in_wPNKC_columns? (and prob actually want to indicate this is
    # just for the collapsed PN<>APL into KC<>APL claw-length vectors... might no longer
    # be used for initially intended purpose)
    sep_boutons = Btn_separate or prat_boutons

    bouton_index = None
    # TODO restore? consolidate flags?
    #if bouton_dynamics:
    if sep_boutons:
        # TODO still want to de-duplicate? there will be (Btn_num_per_glom=10) adjacent
        # copies of each glomerulus currently (or in prat_claws=True case, some
        # arbitrary # for each glomerulus) (de-duplicating would also be consistent w/
        # diff between kc_index and claw_index, right?)
        glomerulus_index = wPNKC.columns.get_level_values(glomerulus_col)

        # mainly for prat_claws=True case
        assert glomerulus_index.sort_values().equals(glomerulus_index)

        # TODO try to replace all uses of glomerulus index w/ this? (then maybe remove
        # glomerulus_index, and rename this to that, if i can replace all)
        glomerulus_index_unique = glomerulus_index.unique()

        bouton_index = wPNKC.columns.copy()
        # TODO assert dtype int?
        # TODO assert not in columns?
        assert BOUTON_ID in bouton_index.names
    else:
        glomerulus_index = wPNKC.columns
        # TODO delete
        assert not glomerulus_index.duplicated().any()
        glomerulus_index_unique = glomerulus_index
        #

    if not hallem_input:
        zero_filling = (~ glomerulus_index_unique.isin(orn_deltas.index))
        if zero_filling.any():
            msg = ('zero filling spike deltas for glomeruli not in data: '
                f'{sorted(glomerulus_index_unique[zero_filling])}'
            )
            warn(msg)

        # TODO (?) if i add 'uniform' draw path, make sure zero filling is keeping
        # glomeruli that would implicitly be dropped in hemibrain (/ hemidraw / caron)
        # cases (as we don't need wPNKC info in 'uniform' case, as all glomeruli are
        # sampled equally, without using any explicit connectivity / distribution)
        # (don't warn there either)
        #
        # Any stuff w/ '+' in name (e.g. 'DM3+DM5' in Hallem) should already have been
        # dropped.
        input_glomeruli = set(orn_deltas.index)
        glomeruli_missing_in_wPNKC = input_glomeruli - set(glomerulus_index_unique)
        if len(glomeruli_missing_in_wPNKC) > 0:
            # TODO assert False? seems we could do that at least for megamat data...
            warn('dropping glomeruli not in wPNKC (while zero-filling): '
                f'{glomeruli_missing_in_wPNKC}'
            )

        if tune_on_hallem:
            # TODO make sure we aren't writing wPNKC in this case (and maybe not other
            # hallem CSVs? they are probably fine either way...)
            hallem_not_in_wPNKC = (
                set(hallem_orn_deltas.index) - set(glomerulus_index_unique)
            )
            assert len(hallem_not_in_wPNKC) == 0 or hallem_not_in_wPNKC == {'DA4m'}, \
                f'unexpected {hallem_not_in_wPNKC=}'

            if len(hallem_not_in_wPNKC) > 0:
                warn(f'dropping glomeruli not in wPNKC {hallem_not_in_wPNKC} from '
                    'Hallem data to be used for tuning'
                )

            # this will be concatenated with orn_deltas below, and we don't want to add
            # back the glomeruli not in wPNKC
            hallem_orn_deltas = hallem_orn_deltas.loc[
                [c for c in hallem_orn_deltas.index if c in glomerulus_index_unique]
            ].copy()

        orn_deltas_pre_filling = orn_deltas.copy()

        # TODO simplify this. not a pandas call for it? reindex_like seemed to not
        # behave as expected, but maybe it's for something else / i was using it
        # incorrectly
        # TODO just do w/ pd.concat? or did i want shape to match hallem exactly in that
        # case? matter?
        # TODO reindex -> fillna?
        if sep_boutons:
            # TODO delete this whole block eventually (run through all tests first? or
            # at least tianpei path too?) so far, assertion that it's equiv w/ simpler
            # single reindex call below has passed
            row_lookup = {g: orn_deltas.loc[g].to_numpy() for g in orn_deltas.index}
            zero_row = np.zeros(orn_deltas.shape[1], dtype=float)
            rows = [row_lookup.get(g, zero_row) for g in glomerulus_index]
            orn_deltas = pd.DataFrame(rows, index=glomerulus_index,
                columns=orn_deltas.columns
            )
            if not orn_deltas.index.is_unique:
                orn_deltas = orn_deltas.groupby(level=glomerulus_col, sort=False
                    ).first()
            orn_deltas = reindex(orn_deltas, glomerulus_index_unique).fillna(0)
            #

            # TODO replace above with this, after passing enough tests w/o this
            # assertion failing
            orn_deltas2 = reindex(orn_deltas_pre_filling, glomerulus_index_unique,
                fill_value=0
            )
            assert orn_deltas2.equals(orn_deltas)
            #
        else:
            # TODO TODO also simplify (try a single reindex call) like above
            print('also simplify this orn_deltas reindexing')
            # TODO delete
            #orn_deltas1 = orn_deltas_pre_filling.copy()
            #breakpoint()
            #
            orn_deltas = pd.DataFrame([
                    orn_deltas.loc[x].values if x in orn_deltas.index
                    # TODO correct? after concat across odors in tune_on_hallem=True
                    # case?
                    else np.zeros(len(orn_deltas.columns))
                    # TODO also use glomerulus_index here (instead of wPNKC.columns) for
                    # consistency w/ above?
                    for x in wPNKC.columns
                ], index=glomerulus_index, columns=orn_deltas.columns
            )

        # TODO need to be int (doesn't seem so)?
        mean_sfr = sfr.mean()

        non_hallem_gloms = sorted(set(glomerulus_index) - set(sfr.index))
        if len(non_hallem_gloms) > 0:
            warn(f'imputing mean Hallem SFR ({mean_sfr:.2f}) for non-Hallem glomeruli:'
                f' {non_hallem_gloms}'
            )
        sfr = pd.Series(index=glomerulus_index,
            data=[(sfr.loc[g] if g in sfr else mean_sfr) for g in glomerulus_index]
        )
        # sfr: Series indexed by glomerulus with duplicates
        sfr = sfr[~sfr.index.duplicated(keep="first")]
        sfr.index.name = glomerulus_col
        assert sfr.index.equals(orn_deltas.index)
    #
    odor_index = orn_deltas.columns
    n_input_odors = orn_deltas.shape[1]

    extra_orn_deltas: Optional[pd.DataFrame] = None
    # TODO delete/comment after i'm done?
    #'''
    eb_mask = orn_deltas.columns.get_level_values('odor').str.startswith('eb @')
    # should be true for megamat and hallem
    if repro_preprint_s1d and eb_mask.sum() == 1:
        # TODO finish support for "extra" odors (to be simmed, but not tuned on)
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

    # TODO can i remove need for this (and just use odor_index), if i remove the extra
    # odors asap? prob not...
    # I'm currently assuming that pks (but not responses
    # TODO rename to odor_index_noextras or something? may depend on whether i use for
    # much beyond pks (where i think it is the tuned odors that are needed)
    tuning_odor_index = odor_index.copy()

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

        # TODO TODO TODO either maintain original odor_index, and use that below where
        # appropriate, or quickly restore this after dropping those odors (may need the
        # original odor_index in some places regardless...)
        # TODO TODO actually need this version of odor_index anywhere?
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
            # TODO (if i want to keep this) do i want to use odor_index or
            # tuning_odor_index (the former also includes "odors" from extra_orn_deltas,
            # if that's not None)?
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

    # TODO TODO probably still support just one .name == 'odor' tho...
    # (esp for calls w/ just hallem input, either old ones here or model_test.py?)
    # (could just check 'odor' in .names that works even if single level)
    # TODO do i only want to allow [the possibility of] a single other 'panel' level, or
    # allow arbitrary other levels?
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
    # TODO TODO only drop DA4m if it's not in wPNKC (which should only be if
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
    # currently getting tripped by model_test.py case that passes in hallem orn_deltas
    #print(f'{have_DA4m=}')
    #if not have_DA4m:
    #    print()
    #    print('did not have DA4m in sfr.index. add comment explaining current input')
    #    import ipdb; ipdb.set_trace()
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

    assert sfr.index.equals(orn_deltas.index)

    # TODO just assert sfr.index is wPNKC.index (/.columns)? maybe after sorting 1/both?
    # TODO delete (replace w/ assertion?)
    _wPNKC_shape_changed = False
    if wPNKC.shape != wPNKC[sfr.index].shape:
        print()
        print(f'wPNKC shape BEFORE subsetting to sfr.index: {wPNKC.shape}')
        _wPNKC_shape_changed = True
    #

    # TODO TODO also need to subset glomerulus_index here now? just always use
    # wPNKC.columns and remove glomerulus_index?
    # TODO just add assertion that wPNKC shape unchanged by this? i haven't seen the
    # surrounding debug prints trigger in a while, and i'm not sure if we path we care
    # about can reproduce them...

    wPNKC = wPNKC[sfr.index].copy()

    # TODO delete (replace w/ assertion?)
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
    # TODO need to remove DA4m (2a) from wPNKC first too (already out, it seems)?
    # don't see matt doing it in hemimat-modeling... (i don't think i need to.
    # rv.pn.pn_sims below had receptor-dim length of 22)

    if variable_n_claws:
        # TODO is seed actually only used in variable_n_claws=True cases?
        # (seems so, and doesn't seem to matter it is set right before KC sims)
        # TODO should seed be Optional?
        mp.kc.seed = seed
        mp.kc.nclaws = n_claws

    if not one_row_per_claw:
        # TODO also take an optional parameter to control this number?
        # (for variable_n_claws cases mainly)
        # TODO or if always gonna use wPNKC, option to use # from [one of] fafb data
        # sources (2482 in left), instead of hemibrain?
        mp.kc.N = len(wPNKC)
    else:
        # this should also work if values are all True/False, or all float 1.0/0.0.
        # NOTE: should be OK if some claws receive no input (should only be for KCs with
        # no claws with input, and thus should only be in a single claw for each such
        # KC, with claw_id=0)
        if not (prat_claws and dist_weight is not None):
            # TODO move this assertion to include one_row_per_claw=False too?
            # (also dividing values in Btn_separate=True case there, which only used for
            # test)
            if Btn_separate:
                assert set(wPNKC.values.flat) == {1 / Btn_num_per_glom, 0}
            else:
                # TODO if this assertion moved outside `one_row_per_claw` case, still
                # assert non-negative int, as long as no dist_weights?
                assert set(wPNKC.values.flat) == {1, 0}

        # TODO still some assertions on values in an else case? min 0? all finite?
        # (don't really care about `dist_weight is not None` case though...)

        assert CLAW_ID in wPNKC.index.names

        # TODO assert coords all positive or have some other properties?
        assert all(c in wPNKC.index.names for c in claw_coord_cols)

        assert KC_ID in wPNKC.index.names
        assert not wPNKC.index.to_frame(index=False)[[KC_ID, CLAW_ID]].duplicated(
            ).any()

        # TODO TODO restore for at least everything but prat_claws + dist_weight != None
        # cases?
        '''
        checks = True
        # TODO factor out these checks and share w/ end of connectome_wPNKC (in a
        # new branch shared by all one-row-per-claw cases, which currently would need to
        # be made a more complex conditional)
        if checks:
            claws_without_input = (wPNKC.T == 0).all()
            wPNKC_only_kcs_with_input = wPNKC.loc[~ claws_without_input]

            wPNKC_only_kcs_without_input = wPNKC.loc[claws_without_input]
            assert (wPNKC_only_kcs_without_input.index.get_level_values(CLAW_ID) == 0
                ).all()

            kcs_without_input2 = wPNKC_only_kcs_without_input.index.get_level_values(
                KC_ID
            )
            assert not kcs_without_input2.duplicated().any()
            kcs_with_input = set(
                wPNKC_only_kcs_with_input.index.get_level_values(KC_ID).unique()
            )
            assert not any(x in kcs_with_input for x in kcs_without_input2)

            assert (wPNKC_only_kcs_with_input.T.sum() == 1).all()
            if Btn_separate:
                assert (wPNKC_only_kcs_with_input.T.max() == 1/Btn_num_per_glom).all()
            else:
                assert (wPNKC_only_kcs_with_input.T.max() == 1).all()
        '''

        # TODO will i end up wanting to transpose this, to have it's shape in olfsysm
        # consistent w/ some other stuff in there? (see wAPLKC vs wKCAPL, for one
        # current case were olfsysm wants some things in either row or column vectors)
        # TODO try w/o .values after getting working with it?
        kc_ids = kc_index.get_level_values(KC_ID).values
        kc_ids_per_claw = claw_index.get_level_values(KC_ID).values
        # TODO delete (not used right? delete kc_ids too?)
        n_kcs = len(set(kc_ids))
        #
        mp.kc.N = len(kc_index)

        # TODO TODO delete (+ replace usage in C++ w/ kc.N). only used for .size() in
        # model
        # TODO move assignment of these down by other IDs (the claw<>KC ID maps) below?
        # or need here (to init stuff properly?)? if so, maybe if i add bouton/PN IDs,
        # also set those up here
        mp.kc.kc_ids = kc_ids_per_claw
        #

        # TODO also rename olfsysm stuff from wPNKC_one_row_per_claw to
        # one_row_per_claw? maybe not
        mp.kc.wPNKC_one_row_per_claw = True

        # olfsysm default should also be False, but need =True for
        # test_spatial_wPNKC_equiv to pass exactly (but differences w/ =False shouldn't
        # be too large, and that test should soon be updated to also check that)
        mp.kc.allow_net_inh_per_claw = allow_net_inh_per_claw

    if bouton_dynamics:
        # NOTE: Tianpei's initial code will rely on this (so long as I have removed
        # Btn_num_per_glom=<int> from model code, as I have so far, which this
        # replaces need for), but that initial code also did not actually have
        # distinct PN dynamics, or at least it never implemented the weights, nor
        # the interaction in the right place (so reproducing those outputs will fail
        # at some point, if we actually introduce the dynamics and PN<>APL
        # interaction in that case too. will need to be able to not have an
        # additional non-linearity [at least], and also to be able to disable the
        # PN<>APL interactions, to still be able to reproduce that path, if I care.
        # Same thing if I want to keep the test test_btn_expansion)
        mp.pn.n_total_boutons = len(bouton_index)

    if pn2kc_connections in connectome_options:
        mp.kc.preset_wPNKC = True

    elif pn2kc_connections == 'hemidraw':
        # TODO version like hemidraw, but by pre-generating wPNKC, to have similar
        # distribution to whatever real wPNKC we end up settling on from connectome (but
        # otherwise keeping draws of each input indep)
        # try just taking each cell in real connectome, taking # of connections from
        # that cell, then drawing from whatever distribution up to that number of
        # connections? (in contrast to current hemidraw/uniform cases which both have a
        # fixed number of "claws" per model KC)

        # TODO support using wPNKC from fafb-left/fafb-right (currently just hemibrain,
        # w/ _use_matt_wPNKC=False. i.e. using the newer data from prat's query)?
        # (and/or also support arbitrary _wPNKC input)

        # TODO check index (glomeruli) is same as sfr/etc (all other things w/ glomeruli
        # that model uses)
        # TODO just set directly into mp.kc.cxn_distrib?
        # (and in other places that set this)
        #
        # TODO have olfsysm use same appropriate to internally normalize (to mean of 1)
        # connectome wAPLKC/wKCAPL (as i currently do in here) (already doing in here
        # for connectome APL weights)
        #
        # TODO add unit test confirming we don't need to pre-normalize, and then delete
        # uncertain language
        #
        # TODO add this to param dict if it's set? (whether via this code, or future
        # code using other wPNKC inputs)
        #
        # should be normalized (to mean of 1? check) inside olfsysm
        cxn_distrib = wPNKC.sum()

        # TODO delete?
        if hallem_input:
            # TODO compute this from something?
            n_hallem_glomeruli = 23
            assert mp.kc.cxn_distrib.shape == (1, n_hallem_glomeruli)
        #

        # TODO TODO what currently happens if using # glomeruli other > hallem?
        # seems like it may already be broken? (and also in uniform case. not sure if
        # this is why tho) (? delete?)
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

    # TODO add additional kwargs, to allow setting and/or scaling only one of these at
    # at time?
    if use_connectome_APL_weights:
        # TODO consolidate w/ code actually setting the initial vectors into the
        # corresponding variables (one conditional)?
        mp.kc.preset_wAPLKC = True
        mp.kc.preset_wKCAPL = True

    if return_dynamics or plot_example_dynamics:
        # TODO also warn if input odor size is above some threshold? make a formula to
        # roughly calculate memory usage we'd expect?
        if hallem_input:
            # will probably be killed by system OOM killer
            warn('plotting/returning model dynamics likely to crash with hallem input, '
                'b/c many odors'
            )

        # NOTE: if these are left to default of False, seems the rv.kc.vm_sims (or
        # whatever cognate output variable) will seem like an empty list here.
        #
        # These will all save output to `rv.kc.<var-name>` for save flag like
        # `mp.kc.save_<var-name>`.
        mp.kc.save_vm_sims = True
        mp.kc.save_spike_recordings = True
        mp.kc.save_inh_sims = True
        mp.kc.save_Is_sims = True
        # if `mp.kc.ves_p == 0`, the "vesicle depletion" part of olfsysm is disabled.
        # see related comments around use of nves_sims below.
        if mp.kc.ves_p != 0:
            # TODO try setting this True even with `ves_p == 0`, to at least assert it's
            # all 1s, ideally for a unit test
            mp.kc.save_nves_sims = True

        # should have no effect if not in one_row_per_claw mode anyway
        mp.kc.save_claw_sims = True

    # TODO catch likely cause of:
    # ```
    # ValueError: libolfsysm/src/olfsysm.cpp:424 in `KC` check `nextIndex ==
    # int(p.kc.N)` failed
    # ```
    # ...and provide better error message earlier. too many index levels or something?
    # has happened before, and getting again after adding pn_id (int ID) and BOUTON_ID
    # (tuple of ID) levels
    rv = osm.RunVars(mp)

    kc_to_claws = None
    if one_row_per_claw:
        # TODO only even set this compartment stuff if APL_coup_const != -1 (to be
        # clear?)
        rv.kc.claw_compartments = claw_comp
        # 2) build compartment -> claws (list[list[int]])
        C = int(claw_comp.max()) + 1 if claw_comp.size else 0
        compartment_to_claws = [[] for _ in range(C)]
        # stable order: original claw index order is preserved within each compartment
        for claw_in_comp_id, comp_val in enumerate(claw_comp.tolist()):
            compartment_to_claws[comp_val].append(int(claw_in_comp_id))
        rv.kc.compartment_to_claws = compartment_to_claws
        mp.kc.comp_num = C
        # TODO assertions that check above?

        # TODO factor out construction of kc_to_claws to separate fn (+ test it)
        kc_ids_per_claw = np.asarray(kc_ids_per_claw, dtype=np.int64)

        # TODO replace this w/ more idiomatic / simply numpy/pandas code?
        # Compact body IDs to 0..N-1 (first-appearance order)
        id2compact = {}
        claw_to_kc = np.empty(kc_ids_per_claw.shape[0], dtype=np.int32)
        next_idx = 0
        for i, bid in enumerate(kc_ids_per_claw):
            b = int(bid)
            idx = id2compact.get(b)
            if idx is None:
                idx = next_idx
                id2compact[b] = idx
                next_idx += 1
            claw_to_kc[i] = idx
        N = int(next_idx)

        assert len(claw_to_kc) == len(wPNKC)
        wPNKC_kc_ids = wPNKC.index.get_level_values(KC_ID)
        assert wPNKC_kc_ids.sort_values().equals(wPNKC_kc_ids)
        assert np.array_equal(claw_to_kc, np.sort(claw_to_kc))

        # TODO delete two loops w/ dicts below? this should replace them
        for i, x in enumerate(wPNKC_kc_ids.unique()):
            assert np.array_equal(claw_to_kc == i, wPNKC_kc_ids == x)

        # TODO refactor (+ check another way? something more idiomatic?)
        id2seen_compact = dict()
        for x, y in zip(wPNKC_kc_ids, claw_to_kc):
            if x in id2seen_compact:
                s = id2seen_compact[x]
                assert y == s, f'multiple seen claw_to_kc IDs for wPNKC ID {x}'
            else:
                id2seen_compact[x] = y
        #
        id2seen_wPNKC = dict()
        for x, y in zip(claw_to_kc, wPNKC_kc_ids):
            if x in id2seen_wPNKC:
                s = id2seen_wPNKC[x]
                assert y == s, f'multiple seen wPNKC IDs for claw_to_kc ID {x}'
            else:
                id2seen_wPNKC[x] = y
        #

        # TODO add some assertions here that IDs created are correct?

        # Ensure params agree (or update them)
        # Overwrite mapping at runtime; a vector of kc ids for each claw
        # TODO TODO isn't there C++ initialization code that already sets this? just use
        # that instead (+ refactor to share w/ APL<>PN stuff?) check consistent w/ this
        # code, first
        rv.kc.claw_to_kc = claw_to_kc  # len=num_claws

        # TODO replace w/ something more idiomatic. should be able to get numbers from
        # just a reset_index() or something on wPNKC_kc_ids
        # something starting with: wPNKC_kc_ids.to_series().reset_index(drop=True).index
        kc_to_claws = [[] for _ in range(N)]
        for claw_idx, kc_idx in enumerate(claw_to_kc):
            kc_to_claws[int(kc_idx)].append(claw_idx)

        assert len(claw_to_kc) > len(kc_to_claws)
        assert len(kc_to_claws) == len(wPNKC_kc_ids.unique())
        kc_to_claws_flat = [item for sublist in kc_to_claws for item in sublist]
        assert kc_to_claws_flat == sorted(kc_to_claws_flat)
        assert kc_to_claws_flat == list(
            wPNKC_kc_ids.to_series().reset_index(drop=True).index
        )

        assert min(claw_to_kc) == 0, 'claw_to_kc did not start from index 0!'
        assert min(min(xs) for xs in kc_to_claws) == 0, \
            'kc_to_claws did not start from index 0!'

        # TODO del kc_to_claws here, if i can avoid the need to pass to
        # connectome_APL_weights below [already don't need for prat_claws=True case]? or
        # at least check consistent w/ wPNKC index data [want to do that here or in
        # connectome_APL_weights??
        rv.kc.kc_to_claws = kc_to_claws

    # make a map for PN->Btn map; so each glom, what Btn index are associated with that;
    # and vector of length total boutons, and each position
    # stores the glomeruli index.
    if bouton_dynamics:
        # TODO delete eventually
        # TODO move earlier?
        assert all((type(g) is str) and ('#' not in g) for g in glomerulus_index_unique)
        #

        assert isinstance(bouton_index, pd.MultiIndex)
        assert all(x in bouton_index.names for x in (glomerulus_col, BOUTON_ID))

        # TODO replace w/ glomerulus_index (or glomerulus_index_unique, if i can)
        # (/delete, if i can)
        gloms_all = bouton_index.get_level_values(glomerulus_col)

        # ordered unique glomeruli and mapping to indices
        glom_list   = list(pd.unique(gloms_all))
        glom_to_idx = {g: i for i, g in enumerate(glom_list)}

        # TODO what is this doing?
        # (A) bouton -> glomerulus index vector
        btn_to_glom_idx = np.fromiter((glom_to_idx[g] for g in gloms_all),
                                    dtype=np.int32, count=len(gloms_all))

        # TODO TODO refactor all code in here to share w/ kc<>claw stuff above (so i
        # don't have to check each separately as much)

        # (B) glomerulus index -> list of bouton column indices
        G = len(glom_list)
        glom_to_btn = [[] for _ in range(G)]
        for j, g in enumerate(gloms_all):
            glom_to_btn[glom_to_idx[g]].append(j)

        # TODO refactor (or delete after refactoring above)
        assert len(btn_to_glom_idx) > len(glom_to_btn)
        assert len(btn_to_glom_idx) == len(bouton_index)
        wPNKC_glom_ids = bouton_index.get_level_values(glomerulus_col)
        assert wPNKC_glom_ids.sort_values().equals(wPNKC_glom_ids)
        assert np.array_equal(btn_to_glom_idx, np.sort(btn_to_glom_idx))

        for i, x in enumerate(wPNKC_glom_ids.unique()):
            assert np.array_equal(btn_to_glom_idx == i, wPNKC_glom_ids == x)

        assert len(glom_to_btn) == len(wPNKC_glom_ids.unique())
        glom_to_btn_flat = [item for sublist in glom_to_btn for item in sublist]
        assert glom_to_btn_flat == sorted(glom_to_btn_flat)
        assert glom_to_btn_flat == list(
            wPNKC_glom_ids.to_series().reset_index(drop=True).index
        )
        #

        assert min(btn_to_glom_idx) == 0, 'btn_to_glom_idx did not start from index 0!'
        assert min(min(xs) for xs in glom_to_btn) == 0, \
            'glom_to_btn did not start from index 0!'

        rv.pn.Btn_to_pn = btn_to_glom_idx
        rv.pn.pn_to_Btns = glom_to_btn

    # TODO need to support int type too (and in all similar isinstance calls)?
    # isinstance(<int>, float) is False
    if fixed_thr is not None and not isinstance(fixed_thr, float):
        mp.kc.use_vector_thr = True

        assert len(fixed_thr.shape) == 1
        # TODO necessary? not sure (run test_vector_thr again after finishing vector
        # fixed_thr support -> try to see if required)
        thr = np.expand_dims(fixed_thr, 1)
        #thr = fixed_thr

        rv.kc.thr = thr
        del thr

    wAPLKC_scale = None
    wKCAPL_scale = None
    wAPLPN_scale = None
    wPNAPL_scale = None
    if wAPLKC is not None:
        assert target_sparsity_factor_pre_APL is None
        assert fixed_thr is not None, 'for now, assuming both passed if either is'

        mp.kc.tune_apl_weights = False

        if wKCAPL is None:
            # TODO assert this on save? b/c seems some cases violated this at some
            # point, maybe just when there were olfsysm bugs (the ones that
            # get_APL_weights needed to explicitly set this for)
            # TODO TODO try dividing by #-claws instead (along w/ change to olfsysm
            # init), and check we can then pass fixed_inh_params test for #-claw length
            # outputs there (might help move weight normalization to olfsysm, and
            # simplify by always using length of vectors in factor, instead of a
            # collection of random other constants)
            wKCAPL = wAPLKC / mp.kc.N

        if use_connectome_APL_weights or not one_row_per_claw:
            # NOTE: expected a single float passed in here, interpret as in same manner
            # as w[APLKC|KCAPL]_scale floats output by prior calls
            # TODO use is_scalar instead?
            assert isinstance(wAPLKC, float)
            assert isinstance(wKCAPL, float)
        else:
            # currently seems one-row-per-claw=True cases always have non-float
            # input here (in contrast to how i might originally have implemented
            # it)
            assert isinstance(wAPLKC, pd.Series)
            assert isinstance(wKCAPL, pd.Series)

        if use_connectome_APL_weights:
            # TODO TODO add additional kwargs, to allow setting and/or scaling only one
            # of these at at time (would also want to change block above setting
            # mp.kc.preset_w[APLKC|KCAPL] above)?
            # TODO TODO TODO just change init values by some scale factor? or/also allow
            # hardcoding some, and letting tuning vary the others?
            wAPLKC_scale = wAPLKC
            wKCAPL_scale = wKCAPL
            rv.kc.wAPLKC_scale = wAPLKC_scale
            rv.kc.wKCAPL_scale = wKCAPL_scale
        else:
            # NOTE: min/max for these should all be the same. they are essentially
            # scalars, at least as tuned before
            # rv.kc.wKCAPL.shape=(1, 1630)
            # rv.kc.wKCAPL.max()=0.002386503067484662
            # rv.kc.wAPLKC.shape=(1630, 1)
            # rv.kc.wAPLKC.max()=3.8899999999999992
            if one_row_per_claw:
                assert pd_index_equal(wAPLKC, wPNKC, only_check_shared_levels=True)
                assert pd_index_equal(wKCAPL, wPNKC, only_check_shared_levels=True)

                # TODO use diff var than `claw_to_kc.size` (n_claws [no, that's used as
                # kwarg for variable_n_claws cases])?
                #
                # expand_dims will strip off index info w/o having to specify .values
                assert len(wAPLKC) == claw_to_kc.size
                assert len(wKCAPL) == claw_to_kc.size
                # TODO delete? should only need for wKCAPL?
                rv.kc.wAPLKC = np.expand_dims(wAPLKC, 1)
                #
                rv.kc.wKCAPL = np.expand_dims(wKCAPL, 0)
            else:
                assert isinstance(wAPLKC, float)
                assert isinstance(wKCAPL, float)
                rv.kc.wAPLKC = np.ones((mp.kc.N, 1)) * wAPLKC
                rv.kc.wKCAPL = np.ones((1, mp.kc.N)) * wKCAPL

            # TODO try setting wAPLKC = 1 (or another reasonable constant), and only
            # vary wKCAPL? (also considering adding an olfsysm param to vary ratio
            # between the two, as mentioned in another comment)
            # (or probably vice versa, where wKCAPL = 1 / mp.kc.N, and wAPLKC varies)

    if wAPLPN is not None:
        assert isinstance(wAPLPN, float)
        if wPNAPL is not None:
            assert isinstance(wPNAPL, float)
        # TODO TODO can this also be set from wAPLPN, in same manner as above?
        # make sense? or get to what i want more easily by setting separately?
        else:
            # TODO delete
            # compare to wAPLKC/wKCAPL situation
            warn(f'setting wPNAPL from {wAPLPN=} / #-boutons. may not make sense?')
            #
            assert mp.pn.n_total_boutons > 1
            wPNAPL = wAPLPN / mp.pn.n_total_boutons

        wAPLPN_scale = wAPLPN
        wPNAPL_scale = wPNAPL
        rv.pn.wAPLPN_scale = wAPLPN_scale
        rv.pn.wPNAPL_scale = wPNAPL_scale

    # TODO delete
    '''
    if not (prat_boutons and not per_claw_pn_apl_weights):
        # only relevant in this case. don't  want it polluting param_dict otherwise
        if pn_apl_scale_factor != 1:
            # TODO TODO TODO working correctly?
            # TODO fix + remove need for hack
            warn('setting pn_apl_scale_factor=1 because of other params')
            #
            pn_apl_scale_factor = 1
    '''

    if use_connectome_APL_weights:
        # TODO can i delete this (and from connectome_APL_weights)? don't i already have
        # this info in wPNKC still?
        if not one_row_per_claw:
            for_kc_types = kc_types
        else:
            for_kc_types = claw_index.get_level_values(KC_TYPE)
        #

        # TODO assert none of the APL<>PN args set if `_wAPLKC is not None`?
        # (+ other args only used in below?)
        if _wAPLKC is None:
            assert _wKCAPL is None
            # TODO even need kc_to_claws? can i not use wPNKC to associate the two
            # (and maybe still create + set this for olfsysm [both of which are done
            # above], if needed)
            wAPLKC, wKCAPL, wAPLPN, wPNAPL = connectome_APL_weights(
                connectome=connectome, wPNKC=wPNKC, prat_claws=prat_claws,
                prat_boutons=prat_boutons,
                per_claw_pn_apl_weights=per_claw_pn_apl_weights,
                # TODO delete
                #pn_apl_scale_factor=pn_apl_scale_factor,
                #
                kc_types=for_kc_types, kc_to_claws=kc_to_claws,
                _drop_glom_with_plus=_drop_glom_with_plus,
                plot_dir=plot_dir if make_plots else None
            )

            # TODO assert anything about _wAPLKC/_wKCAPL if either of these flags is
            # True?
            if add_PNAPL_to_KCAPL or replace_KCAPL_with_PNAPL:
                assert prat_boutons
                assert per_claw_pn_apl_weights

                assert wAPLPN is not None and wPNAPL is not None
                # should be established elsewhere that wKCAPL and wAPLKC have same index
                assert wAPLPN.index.equals(wKCAPL.index)
                assert wPNAPL.index.equals(wKCAPL.index)

            # TODO still return (in params?) and save both KC<>APL weights as well
            # as new PN<>APL weights, in all cases
            if add_PNAPL_to_KCAPL:
                assert not replace_KCAPL_with_PNAPL
                # TODO TODO try different scales? currently, they should both be mean 1
                wAPLKC = wAPLKC + wAPLPN
                wKCAPL = wKCAPL + wPNAPL

            elif replace_KCAPL_with_PNAPL:
                wAPLKC = wAPLPN
                wKCAPL = wPNAPL

            # TODO and what's happening in one-row-per-claw case? is there any
            # similar scaling? try to do in both cases? (test_spatial_wPNKC_equiv
            # already tests both w/ and w/o connectome APL weights, so prob OK)
            # NOTE: also not doing this in advance if _wAPLKC & _wKCAPL hardcoded (for
            # tests), so we can test w/ 0 vector input there
            if not one_row_per_claw:
                wAPLKC = wAPLKC / wAPLKC.mean()
                wKCAPL = wKCAPL / wKCAPL.mean()
                assert wAPLPN is None and wPNAPL is None
        else:
            assert isinstance(_wAPLKC, pd.Series)
            assert isinstance(_wKCAPL, pd.Series)
            wAPLKC = _wAPLKC.copy()
            wKCAPL = _wKCAPL.copy()
            # TODO make this some other kwarg for testing? like _tune_apl_weights?
            # first test i want to use _wAPLKC/etc for is to check thr/pks are same from
            # first step in fit_sparseness, whether or not wAPLKC/wKCAPL are the
            # 0-vector or not
            warn('hardcoding mp.kc.tune_apl_weights=False, b/c using _wAPLKC & _wKCAPL')
            mp.kc.tune_apl_weights = False

        assert wAPLKC.index.equals(wKCAPL.index)
        if not one_row_per_claw:
            assert wPNKC.index.equals(wAPLKC.index)
        else:
            assert pd_index_equal(wPNKC, wAPLKC, only_check_shared_levels=True)
            # TODO delete
            #assert len(wPNKC) == len(wAPLKC)

        # TODO delete (this was all before scaling, presumably)
        # (min is 1 for both of these)
        # ipdb> wAPLKC.max()
        # 38.0
        # ipdb> wKCAPL.max()
        # 27.0

        # TODO delete
        # TODO don't warn in one-row-per-claw case, if we don't do that there...
        # (move warning to where scaling happens, at least, to keep more accurate)
        warn(f'scaling {connectome=} wAPLKC & wKCAPL, each to mean of 1')

        wAPLKC_arr = np.expand_dims(wAPLKC, 1)
        if one_row_per_claw:
            # TODO refactor len(kc_ids_per_claw) to some shared n_claws var
            assert wAPLKC_arr.shape == (len(kc_ids_per_claw), 1)
        else:
            assert wAPLKC_arr.shape == (mp.kc.N, 1)
        rv.kc.wAPLKC = wAPLKC_arr.copy()

        wKCAPL_arr = np.expand_dims(wKCAPL, 0)
        if one_row_per_claw:
            assert wKCAPL_arr.shape == (1, len(kc_ids_per_claw))
        else:
            assert wKCAPL_arr.shape == (1, mp.kc.N)
        rv.kc.wKCAPL = wKCAPL_arr.copy()

        n_zero_input_wAPLKC = (wAPLKC == 0).sum()
        n_zero_input_wKCAPL = (wKCAPL == 0).sum()

        # TODO also define these for tianpei's one-row-per-claw cases now too?
        # (or probably rather change so they aren't passed as Series...)
        input_wAPLKC = wAPLKC.copy()
        input_wKCAPL = wKCAPL.copy()

        if prat_boutons and not per_claw_pn_apl_weights:
            n_zero_input_wAPLPN = (wAPLPN == 0).sum()
            n_zero_input_wPNAPL = (wPNAPL == 0).sum()
            input_wAPLPN = wAPLPN.copy()
            input_wPNAPL = wPNAPL.copy()

            # will set wAPLPN/wPNAPL into rv.pn.wAPLPN/wPNAPL below
            mp.pn.preset_wAPLPN = True
            mp.pn.preset_wPNAPL = True

            wAPLPN_arr = np.expand_dims(wAPLPN, 1)
            assert wAPLPN_arr.shape == (mp.pn.n_total_boutons, 1)
            rv.pn.wAPLPN = wAPLPN_arr.copy()

            wPNAPL_arr = np.expand_dims(wPNAPL, 0)
            assert wPNAPL_arr.shape == (1, mp.pn.n_total_boutons)
            rv.pn.wPNAPL = wPNAPL_arr.copy()
        else:
            assert not mp.pn.preset_wAPLPN
            assert not mp.pn.preset_wPNAPL

    # TODO implement (+ delete similar comment above if so)
    # TODO TODO restore this assertion? still relevant for multiresponder_APL_boost
    # block that exists below now? test w/o connectome APL weights?
    # (test_multiresponder_APL_boost currently only seems to test
    # use_connectome_APL_weights=True case)
    """
    if not use_connectome_APL_weights:
        assert multiresponder_APL_boost is None, 'not supported'
    """

    if pn2kc_connections in connectome_options or _wPNKC is not None:
        wPNKC_for_model = wPNKC
        if add_PNAPL_to_KCAPL or replace_KCAPL_with_PNAPL:
            warn('summing wPNKC across bouton columns, within glomeruli, before '
                'plugging in to model, since C++ boutons not currently supported in '
                'per_claw_pn_apl_weights=True case'
            )

            # will no longer have bouton_cols in columns after
            # TODO want to do anything to move them back to index first? should be ok
            # w/o them, right?
            # TODO refactor to avoid need for this separate variable?
            # TODO .mean() instead?
            wPNKC_for_model = wPNKC.groupby(glomerulus_col, axis='columns').sum()

            # since still expecting {0, 1} for values below
            # (could probably also replace .sum() above w/ .any()?)
            wPNKC_for_model = (wPNKC_for_model > 0).astype(int).copy()

        # TODO TODO check that any add_PNAPL_to_KCAPL or replace_KCAPL_with_PNAPL
        # results are unchanged if wPNKC actually only has # columns for glomeruli
        # (seems i currently probably need to reduce wPNKC down to that shape anyway,
        # and not sure that will change...)
        # if `_wPNKC is not None`, its contents were already copied into wPNKC above
        rv.kc.wPNKC = wPNKC_for_model

    # TODO need delete=False?
    # TODO need to set delete_on_close=False too?
    # TODO don't delete this by default? or add arg to not delete it?
    # ig it's only deleted when copied to a model output dir anyway? but still...
    temp_log_file = NamedTemporaryFile(suffix='.olfsysm.log', delete=False)

    # also includes directory (e.g. '/tmp/tmp5lhlb2n0.olfsysm.log')
    # a tempfile is nice here, in case the program runs away and generates a huge log.
    # at least on a reboot, if not sooner, the temp file should be cleared. may not be
    # cleared before that if you had to kill the process by sending SIGSEGV, like i
    # often do.
    temp_log_path: str = temp_log_file.name
    try:
        # it seems to just append to this file, if it already exists (should no longer
        # an issue now that I'm making temp files)
        rv.log.redirect(temp_log_path)

    # TODO just `raise` (/ remove try/except)? shouldn't need to support olfsysm
    # versions this old anymore
    #
    # just so i can experiment w/ reverting to old olfsysm, before i added this
    except AttributeError:
        pass

    # TODO only do this if verbose? if some other flag?
    tee_olfsysm = False
    # TODO put behind al_util.verbose?
    try:
        # will write to stdout as well as the file in the redirect(...) call above.
        # this must be called after redirect.
        rv.log.tee()
        tee_olfsysm = True
    except AttributeError:
        pass

    # TODO delete? can use tee now, which also gets output in real time. only minor
    # disadvantage is it's not in a diff color (but could hardcode that in C++?).
    # duplicate output is annoying.
    if not tee_olfsysm and print_olfsysm_log is None:
        # TODO add new verbose kwarg in here (put all unconditional prints inside there
        # too)
        print_olfsysm_log = al_util.verbose

    temp_log_dir = Path(temp_log_path).parent
    latest_log_link = temp_log_dir / 'olfsysm.log'
    # NOTE: exists() will be False for a symlink that exists, but has non-existant
    # target (but if the link exists, is_symlink() will be True)
    if latest_log_link.is_symlink():
        latest_log_link.unlink()

    assert not latest_log_link.exists()
    latest_log_link.symlink_to(temp_log_path)
    assert latest_log_link.is_symlink()

    # TODO delete?
    if tee_olfsysm or print_olfsysm_log:
        # TODO maybe print this regardless though? logging regardless...
        print(f'writing olfsysm log to {temp_log_path} '
            f'(and linking {latest_log_link} to it)'
        )

    osm.run_ORN_LN_sims(mp, rv)
    osm.run_PN_sims(mp, rv)
    before_any_tuning = time.time()

    # This is the only place where build_wPNKC and fit_sparseness (both functions in
    # olfsysm, the C++ code) are called, and they are only called if the 3rd parameter
    # (regen=) is True.
    # TODO TODO fix:
    # test_fixed_inh_params[one-row-per-claw_True__prat-claws_True__APL-coup-const_0__connectome-APL_True]
    # libolfsysm/src/olfsysm.cpp:2093 in `run_KC_sims` check `p.kc.allow_net_inh_per_claw`
    osm.run_KC_sims(mp, rv, True)

    tuning_time_s = time.time() - before_any_tuning

    # TODO is it all zeros after the n_hallem odors?
    # TODO do responses to first n_hallem odors stay same after changing sim_only and
    # re-running below?
    # Of shape (n_kcs, n_odors). odors as columns, as elsewhere.
    responses = rv.kc.responses.copy()
    responses_after_tuning = responses.copy()

    # TODO clarify whether this is just from stim_start:stim_end
    # (bit hard to tell from quick look at olfsysm. possible this is over all time
    # points...? that might not be what i want...).
    #
    # ok, well at least these are true (from hemibrain run on megamat data):
    # ipdb> spike_recordings[:, :, :stim_start_idx].sum()
    # 0.0
    # ipdb> spike_recordings[:, :, stim_end_idx:].sum()
    # 2.0
    #
    # yes, it does seem (the 2) spikes after stim_end_idx are also counted
    # ipdb> spike_counts.sum().sum()
    # 5468.0
    # ipdb> spike_recordings.sum()
    # 5468.0
    spike_counts = rv.kc.spike_counts.copy()

    # TODO if tune_apl_weights=False, assert wAPLKC/wKCAPL (and *_scale counterparts,
    # for preset_*=True cases) do not change from initialization to end?
    # TODO also check we aren't *too* far off in homeostatic_thrs case (when current
    # call may not have it True, but using vector thr from such a call)
    #
    # could be True even if `target_sparsity is None` (if olfsysm default of 0.1 is
    # used), but scalar fixed_thr/wAPLKC/wKCAPL should not be passed then
    if mp.kc.tune_apl_weights:
        # TODO assert that things mentioned in comment above are actually None (or
        # vector, not scalar, for at least fixed_thr)?

        if len(mp.kc.tune_from) > 0:
            # mp.kc.tune_from is an empty list if not explicitly set
            sp_actual = responses[:, mp.kc.tune_from].mean()
        else:
            sp_actual = responses.mean()

        # NOTE: if this fails, may want to check if
        # (rv.kc.tuning_iters == mp.kc.max_iters)
        # (and increase if needed [/ add a check we haven't reached max_iters])
        #
        # matt's tuning loop runs while:
        # (abs(sp - p.kc.sp_target) > (p.kc.sp_acc * p.kc.sp_target)
        abs_sp_diff = abs(sp_actual - mp.kc.sp_target)
        rel_sp_diff = abs_sp_diff / mp.kc.sp_target

        # TODO raise some kind of custom convergence failure error instead, with message
        # including which parameters can be changed to try to get it to converge
        # NOTE: do not remove/comment this assertion
        assert rel_sp_diff <= mp.kc.sp_acc, (f'{rel_sp_diff=} > {mp.kc.sp_acc=}'
            f'\n{sp_actual=}\n{mp.kc.sp_target=}'
        )

        del sp_actual, abs_sp_diff, rel_sp_diff

    if variable_n_claws:
        assert mp.kc.seed == seed

        wPNKC = rv.kc.wPNKC.copy()
        # TODO should these not also be the case in variable_n_claws == False case?
        # move these two assertions out?
        assert wPNKC.shape[1] == len(glomerulus_index)
        assert len(wPNKC) == len(responses)

        # TODO don't define from responses.index (that's just a default range index w/
        # name KC_ID anyway) (-> move earlier -> define kc_types from wPNKC index
        # -> define all other KC indices from that)
        wPNKC = pd.DataFrame(data=wPNKC, columns=glomerulus_index)
        wPNKC.index.name = KC_ID

    # TODO skip this re-adding if i change above to not drop kc_type level from wPNKC
    # index? any code in between that would actually err w/o the dropping?
    if kc_types is not None:
        assert not kc_types.isna().any()

        # TODO move this outside of these conditionals?
        assert KC_TYPE not in kc_index.names

        if not variable_n_claws:
            # TODO need to special case? just always use to_frame()/etc?
            if len(kc_index.names) > 1:
                for_index = kc_index.to_frame(index=False)
                for_index[KC_TYPE] = kc_types
                kc_index = pd.MultiIndex.from_frame(for_index)
            else:
                kc_index = pd.MultiIndex.from_arrays([kc_index, kc_types])

            assert KC_TYPE in kc_index.names

        if not one_row_per_claw:
            wPNKC.index = kc_index
        # KC_type was not dropped from claw_index

    # TODO TODO maybe correlate spont_in against raw (or claw) PN->KC weight for that
    # glomerulus? any better than (Caron-wPNKC-based) 3B/C point in ann's preprint?
    # maybe use average PN->KC weight per-glomerulus to fill in missing spont_in values
    # (for glomeruli not in hallem)?
    try:
        # line that would trigger the AttributeError
        spont_in = rv.kc.spont_in.copy()

    # to allow trying older versions of olfsysm, that didn't have rv.kc.spont_in
    # (which is what this would be doing, if i WAS still `pass`-ing instead of
    # `raise`-ing below)
    except AttributeError:
        # was previously just `pass`-ing here, but don't think i need to support this
        # again moving forward.
        raise

    type2target_response_rate = None
    if fixed_thr is not None:
        # this currently only works by adjusting thresholds per type, so can't have any
        # kind of prespecified thresholds
        assert not equalize_kc_type_sparsity

        # TODO need to support int type too (in both of the two isinstance calls below)?
        # isinstance(<int>, float) is False
        #
        # just checking what we set above hasn't changed
        if isinstance(fixed_thr, float):
            assert mp.kc.fixed_thr == fixed_thr

        assert mp.kc.add_fixed_thr_to_spont == True
        # actually do need this (or else what? it's tuned?). may or may not need
        # thr_type='fixed' too
        assert mp.kc.use_fixed_thr == True
        assert mp.kc.thr_type == 'fixed'
        # TODO some assertion w/ spont_in here? should we be able to calculate fixed_thr
        # same way?

    elif not homeostatic_thrs:
        thr = rv.kc.thr

        # TODO add assertion checking i can recalculate spont_in (or at least make sure
        # i understand how it's computed in the process of trying...)

        unique_thrs_and_counts = pd.Series((thr - 2*spont_in).squeeze()).value_counts()
        unique_fixed_thrs = unique_thrs_and_counts.index
        # TODO try min/max instead of value w/ max count? any possibility of
        # avoiding the numerical issue causing this?
        # TODO take this kind of strategy by default for _single_unique_val?
        # seems we couldn't use current implementation of that here
        # (but haven't proven that this strategy is an improvement...)
        #
        # pick one w/ biggest count (assuming that is least likely to have been affected
        # by numerical issues...). values are the counts (w/ largest count at -1).
        # index contains the thresholds.
        #
        # this should correspond to the thr_const variable inside
        # olfsysm.choose_KC_thresh_uniform (and can be set by passing as the fixed_thr
        # kwarg to this function, which will also set mp.kc.add_fixed_thr_to_spont=True)
        fixed_thr = unique_thrs_and_counts.sort_values().index[-1]
        assert np.allclose(fixed_thr, unique_fixed_thrs)

        # TODO put behind verbose kwarg (/delete)
        print(f'fixed_thr: {fixed_thr}')

        # TODO move below into a unit test (that also hardcodes
        # tune_apl_weights=False)? can i use it to figure out if there's a strategy than
        # can consistently pick which of numerically-slightly-diff thresholds to use to
        # exactly recreate what responses would have been pre-APL?
        #
        # ***w/ mp.kc.tune_apl_weights hardcoded above rv def to False*** :
        # ipdb> np.array_equal(pks >= unique_fixed_thrs.max(), responses)
        # True
        #
        # ipdb> (pks > unique_fixed_thrs.min()).mean()
        # 0.20003214400514305
        # ipdb> (pks >= unique_fixed_thrs.min()).mean()
        # 0.20003214400514305
        #
        # ipdb> unique_fixed_thrs.min()
        # 256.8058676548658
        # ipdb> unique_fixed_thrs.max()
        # 256.8058676548659
        pks = pd.DataFrame(data=rv.kc.pks, index=kc_index, columns=tuning_odor_index)

        # TODO summarize+delete most of comment block below
        #
        # ipdb> pks.stack().iloc[:, 0].quantile(q=[0, 0.2, 0.5, 0.8, 1])
        # 0.0    -228.153484
        # 0.2       9.505707
        # 0.5     113.635038
        # 0.8     256.812860
        # 1.0    1232.700436
        # Name: megamat, dtype: float64
        # ipdb> pks.stack().iloc[:, 0].sort_values()
        # kc_id       kc_type  odor
        # 415852518   g        aa @ -3       -228.153484
        # 5813020132  g        B-cit @ -3    -211.070698
        # 5812981441  g        2-but @ -3    -191.946836
        # 415852518   g        va @ -3       -191.910623
        # 5812982766  g        2-but @ -3    -182.479178
        #                                       ...
        # 693500652   g        eb @ -3        970.448394
        #                      2h @ -3        971.923691
        #                      1-6ol @ -3    1122.538275
        #                      IaA @ -3      1184.759164
        #                      pa @ -3       1232.700436
        #
        # ipdb> pks.stack().squeeze().quantile(q=1 - mp.kc.sp_target *
        #    mp.kc.sp_factor_pre_APL, interpolation='higher')
        # 256.84082743722297
        # ipdb> fixed_thr
        # 256.8058676548658
        # ipdb> pks.stack().squeeze().quantile(
        #    q=1 - mp.kc.sp_target * mp.kc.sp_factor_pre_APL, interpolation='lower')
        # 256.80586765486584
        # ipdb> mp.kc.sp_target * mp.kc.sp_factor_pre_APL
        # 0.2
        # ipdb> lower = pks.stack().squeeze().quantile(q=1 - mp.kc.sp_target *
        #    mp.kc.sp_factor_pre_APL, interpolation='lower')
        # ipdb> higher = pks.stack().squeeze().quantile(q=1 - mp.kc.sp_target *
        #    mp.kc.sp_factor_pre_APL, interpolation='higher')
        # ipdb> (pks >= lower).mean()
        # panel    odor
        # megamat  2h @ -3       0.350273
        #          IaA @ -3      0.276503
        #          pa @ -3       0.343169
        #          2-but @ -3    0.208743
        #          eb @ -3       0.289617
        #          ep @ -3       0.248634
        #          aa @ -3       0.045902
        #          va @ -3       0.110929
        #          B-cit @ -3    0.009836
        #          Lin @ -3      0.055191
        #          6al @ -3      0.248087
        #          t2h @ -3      0.249727
        #          1-8ol @ -3    0.142077
        #          1-5ol @ -3    0.301093
        #          1-6ol @ -3    0.329508
        #          benz @ -3     0.156831
        #          ms @ -3       0.034426
        # dtype: float64
        # ipdb> (pks >= lower).mean().mean()
        # 0.20003214400514305
        # ipdb> (pks > lower).mean().mean()
        # 0.19999999999999998
        # ipdb> (pks > higher).mean().mean()
        # 0.19996785599485697
        # ipdb> (pks >= higher).mean().mean()
        # 0.19999999999999998
        pks = pks.stack(pks.columns.names).squeeze()
        assert isinstance(pks, pd.Series)

        # olfsysm defaults: mp.kc.sp_target=0.1, mp.kc.sp_factor_pre_APL=2.0
        target_response_rate_pre_apl = mp.kc.sp_target * mp.kc.sp_factor_pre_APL

        # TODO delete (/ turn into tests).
        # almost entire purpose of this was to justify my interpolation method was
        # consistent w/ how threshold was calculated inside olfsysm.
        """
        #interpolation_for_thr = 'midpoint'
        # fails sooner than 'lower' (w/ >=)?
        #interpolation_for_thr = 'higher'
        # initially this seemed right (w/ >=). now it seems inconsistent? maybe that's
        # the best i can do?
        #interpolation_for_thr = 'lower'
        for interpolation_for_thr in ('lower', 'higher'):

            # NOTE: updating from pandas 1.3.1 to 1.5.0 seemed to fix a DeprecationWarning
            # here (about interpolation= being replaced w/ method=, despite all pandas
            # versions seemingly using interpolation= for Series.quantile). numpy==1.24.4
            fixed_thr2 = pks.quantile(q=1 - target_response_rate_pre_apl,
                interpolation=interpolation_for_thr
            )

            # TODO delete
            print()
            print(f'{interpolation_for_thr=} (to define fixed_thr2)')
            print()
            print(f'{fixed_thr=}')
            print(f'{fixed_thr2=}')
            print()
            print(f'{(pks >= fixed_thr).equals(pks >= fixed_thr2)=}')
            print(f'{(pks > fixed_thr).equals(pks > fixed_thr2)=}')
            print()
            #

            #print(pd.Series((thr - 2*spont_in).squeeze()).value_counts())
            # 195.742763    821
            # 195.742763    508
            # 195.742763    501
            #
            #print(pd.Series((thr - 2*spont_in).squeeze()).value_counts().index)
            # Float64Index([195.7427633158587, 195.74276331585867, 195.74276331585872],
            #   dtype='float64')
            #
            # ipdb> pd.Series((thr - 2*spont_in).squeeze()).max()
            # 195.74276331585872
            # ipdb> pd.Series((thr - 2*spont_in).squeeze()).min()
            # 195.74276331585867
            #print()

            # TODO TODO unit test taking selected fixed_thr, setting that into olfsysm
            # fixed_thr, and re-running model (or running equiv new model) ->
            # checking/verifying which one reproduces output spike_counts exactly?
            #
            # TODO delete (try to move to unit test)
            #mp.kc.use_fixed_thr = True
            #mp.kc.thr_type = 'fixed'
            #mp.kc.add_fixed_thr_to_spont = True
            #mp.kc.tune_apl_weights = False

            #mp.kc.fixed_thr = fixed_thr2
            ## looks like i do need regen=True to get thr change to take effect
            #osm.run_KC_sims(mp, rv, True)
            #spike_counts2 = rv.kc.spike_counts.copy()
            #print(f'{np.array_equal(spike_counts, spike_counts2)=}')
            #print()

            for fthr in sorted(unique_fixed_thrs):
                print(f'{fthr=}')
                print(f'{unique_thrs_and_counts[fthr]=}')
                print(f'{(pks >= fthr).equals(pks >= fixed_thr2)=}')
                print(f'{(pks > fthr).equals(pks > fixed_thr2)=}')

                #mp.kc.fixed_thr = fthr
                # looks like i do need regen=True to get thr change to take effect
                #osm.run_KC_sims(mp, rv, True)
                #spike_counts2 = rv.kc.spike_counts.copy()
                # TODO TODO why was this always true despite thresholding on pks being
                # different (for diff choice of threshold)?
                #print(f'{np.array_equal(spike_counts, spike_counts2)=}')

                print()

            # so at least we can actually get diff output for different-enough
            # fixed_thr...
            #
            # ipdb> mp.kc.fixed_thr
            # 195.74276331585872
            # ipdb> mp.kc.fixed_thr = 195.7
            # ipdb> osm.run_KC_sims(mp, rv, True)
            # ipdb> spike_counts2 = rv.kc.spike_counts.copy()
            # ipdb> print(f'{np.array_equal(spike_counts, spike_counts2)=}')
            # np.array_equal(spike_counts, spike_counts2)=False

            import ipdb; ipdb.set_trace()
            #
            #

        #assert np.isclose(fixed_thr, fixed_thr2)
        try:
            # TODO i can't assert it's exactly equal can i? aren't i just picking a
            # value from a list tho? why can i not get it exact?
            assert np.isclose(fixed_thr, fixed_thr2)
        except AssertionError:
            warn(f'{fixed_thr=} != {fixed_thr2=}')
            # TODO TODO TODO why broken now (sometimes?)?
            # ipdb> fixed_thr
            # 268.0375322649455
            # ipdb> fixed_thr2
            # 268.0081239925118
            import ipdb; ipdb.set_trace()

        # TODO TODO try other interpolation methods (+ combos w/ other choices? what?)?
        # maybe i picked the wrong approach?
        #
        # TODO delete?
        #
        # seems this will also fail if above fails
        #assert (pks >= fixed_thr).equals(pks >= fixed_thr2)
        try:
            # TODO TODO restore?
            #assert (pks >= fixed_thr).equals(pks >= fixed_thr2)
            #
            assert (pks > fixed_thr).equals(pks > fixed_thr2)
        # TODO TODO TODO fix
        except AssertionError:
            warn(f'fixed_thr and fixed_thr2 produce diff thresholded pks!!!')
            import ipdb; ipdb.set_trace()
        """
        #

        interpolation_for_thr = 'lower'
        # TODO also compute this in fixed_thr branch above, and use to save a
        # parameter for what the target_sparsity_factor_pre_APL should be?
        # (does olfsysm even compute pks in that case? might not...)
        # (when stepping thr and wAPLKC separately will likely lead to values != 2)
        fixed_thr2 = pks.quantile(q=1 - target_response_rate_pre_apl,
            interpolation=interpolation_for_thr
        )

        # TODO also assert we can recreate responses by using this threshold?
        # (prob more important) (would need to be in a context where responses don't
        # have APL used to compute them tho)

        # TODO do something with this? at least put in param dict
        #
        # "reponse_rate" same units/meaning as "target_sparsity", but better name
        pre_apl_response_rate = (pks >= fixed_thr2).mean().mean()

        # TODO in future, could add a param that's a type->target_rate dict, but would
        # be more complicated(/longer) in filenames and stuff, so some extra
        # clutter/complexity (already have such a dict below. would just need to expose
        # it as a new kwarg)

        # TODO TODO i think if i want a certain response rate at output, i really might
        # have to modify olfsysm to run the APL tuning probably jointly on all types...

        if equalize_kc_type_sparsity:
            # TODO assert no one-row-per-claw here (/test)

            # TODO need to support int type too (in both of the two isinstance calls
            # below)? isinstance(<int>, float) is False
            #
            # if we already had vector thresholds, then they won't be particularly
            # meaningful after we overwrite those outputs with those using cell_thrs
            # below.
            assert fixed_thr is None or isinstance(fixed_thr, float)
            assert not mp.kc.use_vector_thr

            # probably don't care to implement this, but would need to rethink current
            # ordering of olfsysm calls if so
            if tune_on_hallem:
                raise NotImplementedError

            # to compare against the output of the call that will happen below this
            # conditional (before and after attempt at equalizing KC response rates)
            print('response rates BEFORE attempting to equalize pre-APL response rates'
                ' across KC types:'
            )
            pre_responses = pd.DataFrame(responses, index=kc_index, columns=odor_index)
            if extra_orn_deltas is not None:
                # this subset of odors is not tuned, so we don't care about it here
                # (and as a result values seem not even properly initialized here)
                pre_responses = pre_responses.drop(columns=extra_orn_deltas.columns)

            _print_response_rates(pre_responses)

            print()

            kc_type_set = set(kc_types)
            type2target_response_rate = {
                t: r for t, r in
                zip(sorted(kc_type_set), [mp.kc.sp_target] * len(kc_type_set))
            }
            if ab_prime_response_rate_target is not None:
                assert 0 <= ab_prime_response_rate_target <= 1

                # otherwise, either we have the wrong input data or the "a'b'" KCs are
                # called something else.
                assert "a'b'" in type2target_response_rate

                type2target_response_rate["a'b'"] = ab_prime_response_rate_target

            # kc_type
            # a'b'       213.766788
            # ab         247.624665
            # g          304.647391
            # unknown    150.569856
            # Name: megamat, dtype: float64
            #
            # x.name will be a str kc_type
            thr_by_type = pks.groupby(KC_TYPE).apply(lambda x: x.quantile(
                # TODO see how the final rates of each class differ after tuning
                q=1 - type2target_response_rate[x.name] * mp.kc.sp_factor_pre_APL,
                interpolation=interpolation_for_thr
            ))

            # TODO TODO TODO add support for sensitivity analysis in this case?
            cell_thrs = kc_types.map(thr_by_type)

            # TODO also include thr_by_type in param_dict?

            # TODO keep this print (/warn) for a verbose branch?
            print('thr_by_type:')
            print(thr_by_type.to_string())

            # TODO check that these thresholds (if APL is skipped) produce
            # sp-factor * type-target in each type?

            mp.kc.use_vector_thr = True
            mp.kc.add_fixed_thr_to_spont = True
            # actually do need this. may or may not need thr_type='fixed' too
            mp.kc.use_fixed_thr = True
            mp.kc.thr_type = 'fixed'

            # TODO either use a different type or otherwise fix formatting of this into
            # params.csv and related (currently show up w/ middle values truncated, like
            # `[1 2 3 ... 98 99 100]`)
            #
            # overwriting fixed_thr since otherwise it is the float scalar threshold
            # value from *before* picking a threshold for each subtype (to "equalize"
            # the response rate across subtypes), so no longer really meaningful.
            fixed_thr = cell_thrs.values.copy()

            # TODO necessary? not sure (run test_vector_thr again after finishing vector
            # fixed_thr support -> try to see if required)
            thr = np.expand_dims(fixed_thr, 1)
            #thr = fixed_thr

            rv.kc.thr = thr

            # TODO TODO probably also expose the whole type->thr (/fraction) dict below,
            # as does seem (for some reason...) we will need to tweak gamma-KC response
            # fraction up (after APL brings it down to 0.07, below 0.1 target)
            # (under which circumstances did it do this again? connectome APL?)
            #
            # TODO expose this as kwarg? does =False actually make sense?
            retune_apl_post_equalized_thrs = True

            if use_connectome_APL_weights:
                # TODO duplicate (/refactor to share) assertions below on output
                # wAPLKC/wKCAPL (so that we can also check here before overwriting
                # them)?
                # TODO could also change olfsysm so rv.kc.w[APLKC|KCAPL] store the
                # unscaled versions? not sure i care to though.
                #
                # since current rv.kc.w[APLKC|KCAPL] matrices are scaled by the
                # corresponding rv.kc.w[APLKC|KCAPL]_scale scalars before
                # osm.fit_sparseness returns, we need to restart them at a mean of 1
                # before re-scaling, so that the *_scale variables are interpretable
                # in the same manner.
                #
                # TODO check that these values are then scaled by the (still updated)
                # rv.kc.w[APLKC|KCAPL]_scale vars (inside fit_sparseness)
                # TODO assert w[APLKC|wKCAPL]_scale != (default of) 1?
                rv.kc.wAPLKC = wAPLKC_arr
                rv.kc.wKCAPL = wKCAPL_arr
            #

            if not retune_apl_post_equalized_thrs:
                mp.kc.tune_apl_weights = False

            checks = True
            # TODO move/dupe these checks to a unit test
            if checks:
                # TODO check that another run_KC_sims call before changing rv gives us
                # same outputs? prob not worth it...

                # NOTE: .copy() is necessary
                old_spike_counts = spike_counts.copy()

                osm.run_KC_sims(mp, rv, False)
                spike_counts_type_thresh = rv.kc.spike_counts.copy()
                assert not np.array_equal(old_spike_counts, spike_counts_type_thresh,
                    # need equal_nan=True b/c NaN for extra_orn_deltas odors, if any
                    # (b/c those odors are excluded from tuning, and only run in a later
                    # step)
                    equal_nan=True
                )

                # TODO compare to a recursive fit_mb_model call output (passing
                # fixed_thr=thr (or cell_thrs.values, if having already adding singleton
                # dim at end is an issue w/ fixed_thr [which it very well could be, esp
                # since another one will probably get added in recursive call])?
                # TODO how to get all kwargs nicely tho? refactor this fn to pass them
                # all as a dict (basically just to make recursive calls easier)? would
                # come at the cost of having them all explicitly in kwargs tho, unless
                # maybe i broke out a separate fn to initialize the defaults, w/ all
                # kwargs there?

                osm.run_KC_sims(mp, rv, False)
                spike_counts_type_thresh2 = rv.kc.spike_counts.copy()
                assert np.array_equal(
                    spike_counts_type_thresh, spike_counts_type_thresh2, equal_nan=True
                )

            osm.run_KC_sims(mp, rv, True)
            # TODO also exclude any extra_orn_deltas odors here? above?
            # (the values are garbage, yes, b/c sim_only excludes them)
            responses = rv.kc.responses.copy()
            spike_counts = rv.kc.spike_counts.copy()

            # just for check in extra_orn_deltas case below
            responses_after_tuning = responses.copy()

            if checks:
                assert not np.array_equal(spike_counts, old_spike_counts,
                    equal_nan=True
                )
                assert not np.array_equal(spike_counts, spike_counts_type_thresh2,
                    equal_nan=True
                )

            wPNKC2 = rv.kc.wPNKC.copy()
            assert np.array_equal(wPNKC, wPNKC2)

            spont_in2 = rv.kc.spont_in.copy()
            assert np.array_equal(spont_in, spont_in2)

            del thr, spont_in2, wPNKC2

    else:
        assert homeostatic_thrs
        fixed_thr = rv.kc.thr.squeeze().copy()

        # ideally so this works w/ another call w/ add_fixed_thr_to_spont=True, as other
        # vector fixed_thr outputs. necessary b/c of different handling in olfsysm.
        # test_homeostatic_thrs confirms this is correct.
        fixed_thr = fixed_thr - 2 * spont_in.squeeze()

    if extra_orn_deltas is not None:
        # TODO add unit test to confirm (some way of) simming just last bit is equiv to
        # re-simming all and subsetting to last bit (then replace code that re-sims all
        # w/ code that just sims extra odors)
        # TODO maybe just sim the last bit and concat to existing responses,
        # instead of re-running all (check equiv tho)
        #mp.sim_only = range(n_input_odors, n_input_odors + n_extra_odors)
        mp.sim_only = range(n_input_odors + n_extra_odors)

        osm.run_ORN_LN_sims(mp, rv)
        osm.run_PN_sims(mp, rv)
        # Don't want to do either build_wPNKC or fit_sparseness here (after tuning)
        osm.run_KC_sims(mp, rv, False)

        responses = rv.kc.responses.copy()
        spike_counts = rv.kc.spike_counts.copy()

        assert np.array_equal(
            responses_after_tuning[:, :n_input_odors], responses[:, :n_input_odors]
        )

    if tune_on_hallem and not hallem_input:
        assert (responses[:, n_hallem_odors:] == 0).all()

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

        responses = rv.kc.responses.copy()
        spike_counts = rv.kc.spike_counts.copy()

        assert np.array_equal(
            responses_after_tuning[:, :n_hallem_odors], responses[:, :n_hallem_odors]
        )

        # TODO also test where appended stuff has slightly diff number of odors than
        # hallem (maybe missing [one random/first/last] row?)
        responses = responses[:, n_hallem_odors:]
        spike_counts = spike_counts[:, n_hallem_odors:]
    #

    # TODO TODO check this multiresponder_APL_boost path also works in
    # use_connectome_APL_weights=False case. (add to unit test, if not already there)
    # (+ probably support if not)
    #
    # TODO TODO modify so we can pass vector wAPLKC, and move this earlier?
    # (may even work to have passed vector wAPLKC take place of connectome_wAPLKC
    # outputs, where it is scaled to mean of 1 and then scaled via wAPLKC_scale)
    # (had initially tried this boost pre-tuning, but didn't get it working there.
    # commented block with that code exists in first commit w/ multiresponder_APL_boost)
    #
    # NOTE: currently need to implement in all final calls (after any pre-tuning calls),
    # since we don't have a way to pass vector wAPLKC into subsequent calls. this is
    # somewhat limiting (can't actually tune overall response rate w/ these boosted
    # wAPLKC values).
    if multiresponder_APL_boost is not None:
        # TODO assert no one-row-per-claw here (/test)

        if _multiresponder_mask is None:
            # NOTE: just assuming we want to union all the per-panel masks in this dir
            # (as long as this branch is only hit in the pre-tuning on control+kiwi
            # data, that should be ok)
            mask_dir = Path(
                '~/src/natmix_data/pdf/scaled_model_versions/final_scaling'
            ).expanduser()

            mask_paths = list(mask_dir.glob('multiresponder_*.p'))
            assert len(mask_paths) == 2

            mask = pd.Series(index=wAPLKC.index, data=False)
            # TODO TODO compare multiresponders across panels
            # (+ plot clustering of responses across both panels?)
            for mask_path in mask_paths:
                panel_mask = pd.read_pickle(mask_path).droplevel(KC_TYPE)
                assert mask.index.names == panel_mask.index.names
                # indices can differ b/c cells were already dropped in natmix_data, so
                # we can't directly union the Series
                mask[panel_mask[panel_mask].index] = True

                # TODO TODO warn about how many cells we have for each, how many
                # overlap, and how many we have at the end
        else:
            mask = _multiresponder_mask

        # TODO assert index of mask matches something else?

        # TODO or delete all these None defs, and define these + print whether we are
        # boosting each specific var or not?
        mask_mean_wAPLKC_before = None
        nonmask_mean_wAPLKC = None
        mask_mean_wAPLKC_after = None

        mask_mean_wKCAPL_before = None
        nonmask_mean_wKCAPL = None
        mask_mean_wKCAPL_after = None

        boosting_wAPLKC = False
        boosting_wKCAPL = False
        #

        wAPLKC_arr = rv.kc.wAPLKC.copy()

        # for boost_wKCAPL=True, both wAPLKC and wKCAPL are boosted.
        # for boost_wKCAPL=False, only wAPLKC boosted.
        if boost_wKCAPL in (False, True):
            if boost_wKCAPL == False:
                vars_being_boosted = 'wAPLKC (NOT wKCAPL)'
            else:
                vars_being_boosted = 'wAPLKC AND wKCAPL'
                boosting_wKCAPL = True

            boosting_wAPLKC = True

            mask_mean_wAPLKC_before = wAPLKC_arr[mask].mean()
            nonmask_mean_wAPLKC = wAPLKC_arr[~mask].mean()

            wAPLKC_arr[mask] *= multiresponder_APL_boost

            mask_mean_wAPLKC_after = wAPLKC_arr[mask].mean()
            rv.kc.wAPLKC = wAPLKC_arr
        else:
            assert boost_wKCAPL == 'only'
            vars_being_boosted = 'ONLY wKCAPL (not wAPLKC)'
            boosting_wKCAPL = True

        if boost_wKCAPL in (True, 'only'):
            wKCAPL_arr = rv.kc.wKCAPL.copy()

            assert wKCAPL_arr.shape[0] == 1 and len(wKCAPL_arr.shape) == 2
            assert len(wAPLKC_arr) == wKCAPL_arr.shape[1]

            mask_mean_wKCAPL_before = wKCAPL_arr[:, mask].mean()
            nonmask_mean_wKCAPL = wKCAPL_arr[:, ~mask].mean()

            wKCAPL_arr[:, mask] *= multiresponder_APL_boost

            mask_mean_wKCAPL_after = wKCAPL_arr[:, mask].mean()

            rv.kc.wKCAPL = wKCAPL_arr

        warn(f'scaling {vars_being_boosted} by {multiresponder_APL_boost=} for '
            f'{mask.sum()} cells (hack to try to remove multi-responders)!'
        )

        mp.kc.tune_apl_weights = False

        osm.run_KC_sims(mp, rv, False)

        spike_counts_before = spike_counts.copy()

        responses = rv.kc.responses.copy()
        spike_counts = rv.kc.spike_counts.copy()

        boost_msg = f'boosting multiresponder {vars_being_boosted}:\n'
        # TODO move boost_msg def above, and remove boosting_w[APLKC|KCAPL] vars (only
        # used for this)?
        if boosting_wAPLKC:
            boost_msg += (
                f' mean non-multiresponder wAPLKC:    {nonmask_mean_wAPLKC:.3f}\n'
                f' mean multiresponder wAPLKC before: {mask_mean_wAPLKC_before:.3f}\n'
                f' mean multiresponder wAPLKC after:  {mask_mean_wAPLKC_after:.3f}\n'
            )
        if boosting_wKCAPL:
            boost_msg += (
                f' mean non-multiresponder wKCAPL:    {nonmask_mean_wKCAPL:.3f}\n'
                f' mean multiresponder wKCAPL before: {mask_mean_wKCAPL_before:.3f}\n'
                f' mean multiresponder wKCAPL after:  {mask_mean_wKCAPL_after:.3f}\n'
            )
        #

        # TODO turn some of this into (also?) assertions (/tests? happy w/ tests
        # as-is?) (that responses actually decreased in class w/ wAPLKC boosted, as long
        # as it wasn't just wKCAPL that was boosted)?
        warn(boost_msg +

            ' total multiresponder spikes before:     '
                f'{spike_counts_before[mask.values].sum()}\n'

            ' total non-multiresponder spikes before: '
                f'{spike_counts_before[~mask.values].sum()}\n'

            ' total multiresponder spikes after:      '
                f'{spike_counts[mask.values].sum()}\n'

            ' total non-multiresponder spikes after:  '
                f'{spike_counts[~mask.values].sum()}'
        )

        del (vars_being_boosted, spike_counts_before, nonmask_mean_wAPLKC,
            mask_mean_wAPLKC_after, mask_mean_wAPLKC_before, nonmask_mean_wKCAPL,
            mask_mean_wKCAPL_after, mask_mean_wKCAPL_before, boosting_wAPLKC,
            boosting_wKCAPL, wAPLKC_arr
        )
    #

    # want this to be after last olfsysm call. this is also currently deleting it b/c of
    # default delete_on_close=True (may want to change)
    temp_log_file.close()

    # TODO TODO replace w/ wrappers for each osm call, that prints right after each
    # call? (or some combined approach?) (seeing these outputs interleaved w/ python
    # debug prints would prob make some debugging easier)
    # TODO or read lines after each call, and only print new ones? (so we can still have
    # one master log too, if we want that...)
    if print_olfsysm_log:
        print('olfsysm log:')
        log_txt = Path(temp_log_path).read_text()
        cprint(log_txt, 'light_yellow')

    # TODO assert 0 tuning iters otherwise?
    if mp.kc.tune_apl_weights:
        print()
        print(f'tuning time: {tuning_time_s:.1f}s')
        # TODO also print # of iterations? (move print of tuning params below to
        # incorporate here?)

    # TODO warn that we aren't copying log, if plot_dir is None?
    if plot_dir is not None:
        # TODO fix/delete. encountering again from uniform model repro test (w/
        # n_seeds=2) (well, was before checking log instead of plot dir. also not force
        # making plots in variable_n_claws=True case now)
        # TODO delete
        # TODO replace _seen_plot_dirs+usage w/ copy2/to_txt wrapper that will check for
        # me whether we have already to this path?
        #assert plot_dir not in _seen_plot_dirs
        #

        # shutil.copy2 will fail anyway if this doesn't exist, but the error message is
        # a bit less clear
        assert plot_dir.is_dir()

        if not variable_n_claws:
            olfsysm_log = plot_dir / 'olfsysm_log.txt'
        else:
            # TODO want to make separate plot subdirs for seeds rather than just
            # suffixing log file? or just skip all the other outputs?
            olfsysm_log = plot_dir / f'olfsysm_log.seed{seed}.txt'

        assert olfsysm_log not in _seen_olfsysm_logs

        # TODO need to take care to not overwrite if -c/-C? (shouldn't be a huge deal
        # either way. this should mainly be for debugging while something is actively
        # changing anyway)

        # will overwrite `dst`, if it already exists.
        shutil.copy2(temp_log_path, olfsysm_log)

        _seen_olfsysm_logs.add(olfsysm_log)
        # TODO delete
        #_seen_plot_dirs.add(plot_dir)

    # TODO also unlink() the symlink if we get here?
    Path(temp_log_path).unlink()

    # TODO delete if i can get both of these (+ wPNKC) to preserve kc_type level, if
    # wPNKC ever has it (currently just hemibrain)
    #
    # TODO why does it seem kc_type has already been dropped from the .index of each
    # of these? just don't do that (rather than adding it back)
    # (would need to stat by not dropping it from wPNKC, which is then passed to
    # connectome_APL_weights(). could prob then remove separate kc_types= kwarg to
    # that fn)
    kc_ids = kc_index.get_level_values(KC_ID)
    if not one_row_per_claw:
        # TODO maybe these should be asserting against kc_index instead of
        # kc_ids? (or at least w[APLKC|KCAPL].index.get_level_values(KC_ID)?)
        if use_connectome_APL_weights:
            # TODO TODO is it even possible for wAPLKC to be None here? mean to check
            # *_scale instead or something? just delete?
            assert wAPLKC is not None
            #
            if wAPLKC is not None:
                assert kc_ids.equals(wAPLKC.index)
                assert kc_ids.equals(input_wAPLKC.index)

            # TODO TODO is it even possible for wKCAPL to be None here? mean to check
            # *_scale instead or something? just delete?
            assert wKCAPL is not None
            #
            if wKCAPL is not None:
                assert kc_ids.equals(wKCAPL.index)
                assert kc_ids.equals(input_wKCAPL.index)

        apl_weight_index = kc_index
    else:
        apl_weight_index = claw_index

    if not use_connectome_APL_weights:
        # these should either be the same as any hardcoded (scalar) wAPLKC [+ wKCAPL]
        # inputs, or the values chosen by the tuning process. _single_unique_val will
        # raise AssertionError if the input arrays contain more than one unique value.
        # For wAPLKC (column vector, shape Nx1)
        # TODO why not doing in one-row-per-claw case too?
        # (b/c dividing by # of claws, inside the C++ code [which does not use the
        # preset_w[APLKC|KCAPL]=true paths the connectome APL stuff does)
        # TODO some easy change to implementation to not divide by # of claws (maybe
        # move dividing into model), but still preserve behavior? if not, yea can't do
        # these assertions in one-row-per-claw cases
        if not one_row_per_claw:
            rv_scalar_wAPLKC = _single_unique_val(rv.kc.wAPLKC)
            rv_scalar_wKCAPL = _single_unique_val(rv.kc.wKCAPL)

        if wAPLKC is not None:
            # TODO delete? just checking what we set above hasn't changed
            assert mp.kc.tune_apl_weights == False

            # this should now be defined whenever wAPLKC is, whether passed in or not...
            assert wKCAPL is not None

            if not one_row_per_claw:
                assert rv_scalar_wAPLKC == wAPLKC, f'{rv_scalar_wAPLKC=}\n{wAPLKC=}'
                assert rv_scalar_wKCAPL == wKCAPL, f'{rv_scalar_wKCAPL=}\n{wKCAPL=}'
                del rv_scalar_wAPLKC, rv_scalar_wKCAPL
        else:
            if not one_row_per_claw:
                # TODO delete prints? (at least put behind a verbose kwarg)
                # TODO try to have similar prints in other cases, if keeping
                print(f'wAPLKC: {rv_scalar_wAPLKC}')
                print(f'wKCAPL: {rv_scalar_wKCAPL}')
                #
                wAPLKC = rv_scalar_wAPLKC
                wKCAPL = rv_scalar_wKCAPL
                del rv_scalar_wAPLKC, rv_scalar_wKCAPL
            else:
                wAPLKC = pd.Series(index=apl_weight_index, data=rv.kc.wAPLKC.squeeze())
                wKCAPL = pd.Series(index=apl_weight_index, data=rv.kc.wKCAPL.squeeze())
    else:
        # TODO delete? currently unused below (only in commented code)
        input_wAPLKC.index = apl_weight_index
        input_wKCAPL.index = apl_weight_index
        #

        # TODO (outside this fn?) make histograms of these scaled values somewhere,
        # similar histograms of connectome weights?
        # TODO any point copying here?
        wAPLKC = pd.Series(index=apl_weight_index, data=rv.kc.wAPLKC.squeeze())
        wKCAPL = pd.Series(index=apl_weight_index, data=rv.kc.wKCAPL.squeeze())

        # TODO share below w/ one-row-per-claw=True & connectome_APL_weights=False case
        # above?

        # TODO TODO care to do any of same for PN<>APL weights here? delete this?
        n_zero_tuned_wAPLKC = (wAPLKC == 0).sum()
        n_zero_tuned_wKCAPL = (wKCAPL == 0).sum()
        # olfsysm should not be adding any new 0s to any of these vectors
        # TODO warn if either scale is 0 (should only encounter w/ sensitivity analysis
        # sweep, and only as steps are currently configured, where 0 is the lower
        # bound)?
        if wAPLKC_scale is not None and wAPLKC_scale > 0:
            assert n_zero_input_wAPLKC == n_zero_tuned_wAPLKC

        if wKCAPL_scale is not None and wKCAPL_scale > 0:
            assert n_zero_input_wKCAPL == n_zero_tuned_wKCAPL

        # TODO only do this if we didn't already have wAPLKC_scale/wKCAPL_scale defined
        # (not None) before this? or check output against those pre-existing vars
        # separately, if we have them?
        wAPLKC_scale = rv.kc.wAPLKC_scale
        wKCAPL_scale = rv.kc.wKCAPL_scale

        # TODO move outside containing conditional?
        if prat_boutons and not per_claw_pn_apl_weights:
            # TODO refactor (to ideally one fn per weight vector. above and below)
            wAPLPN = pd.Series(index=bouton_index, data=rv.pn.wAPLPN.squeeze())
            wPNAPL = pd.Series(index=bouton_index, data=rv.pn.wPNAPL.squeeze())

            n_zero_tuned_wAPLPN = (wAPLPN == 0).sum()
            n_zero_tuned_wPNAPL = (wPNAPL == 0).sum()
            if wAPLPN_scale is not None and wAPLPN_scale > 0:
                assert n_zero_input_wAPLPN == n_zero_tuned_wAPLPN
            if wPNAPL_scale is not None and wPNAPL_scale > 0:
                assert n_zero_input_wPNAPL == n_zero_tuned_wPNAPL

            wAPLPN_scale = rv.pn.wAPLPN_scale
            wPNAPL_scale = rv.pn.wPNAPL_scale
        #

        # TODO how to repro again? any cases i currently care about?
        # TODO TODO just move the APL boosting after this conditional?
        # TODO TODO fix + restore (at least enable if not in the case that was the
        # issue?)
        #'''
        # do need exact=False on both of these calls (first call doesn't always trip w/o
        # it, but there still are cases where it does)
        wAPLKC_scale_recomputed = _single_unique_val(
            wAPLKC[input_wAPLKC > 0] / input_wAPLKC[input_wAPLKC > 0], exact=False
        )
        try:
            assert np.isclose(wAPLKC_scale, wAPLKC_scale_recomputed)
        # TODO fix
        # TODO or just relax this assertion? (but under what circumstances? seems
        # use_connectome_APL_weights=True & retune_apl_post_equalized_thrs=False)
        except AssertionError:
            print()
            print(f'{wAPLKC_scale=}')
            print(f'{wAPLKC_scale_recomputed=}')
            import ipdb; ipdb.set_trace()
        #

        wKCAPL_scale_recomputed = _single_unique_val(
            wKCAPL[input_wKCAPL > 0] / input_wKCAPL[input_wKCAPL > 0], exact=False
        )
        assert np.isclose(wKCAPL_scale, wKCAPL_scale_recomputed)

        # TODO also care to do for PN<>APL weights? or delete?
        if prat_boutons and not per_claw_pn_apl_weights:
            wAPLPN_scale_recomputed = _single_unique_val(
                wAPLPN[input_wAPLPN > 0] / input_wAPLPN[input_wAPLPN > 0], exact=False
            )
            assert np.isclose(wAPLPN_scale, wAPLPN_scale_recomputed)
            wPNAPL_scale_recomputed = _single_unique_val(
                wPNAPL[input_wPNAPL > 0] / input_wPNAPL[input_wPNAPL > 0], exact=False
            )
            assert np.isclose(wPNAPL_scale, wPNAPL_scale_recomputed)
        #'''

    assert responses.shape[1] == (n_input_odors + n_extra_odors)
    responses = pd.DataFrame(responses, index=kc_index, columns=odor_index)

    assert spike_counts.shape[1] == (n_input_odors + n_extra_odors)
    assert len(responses) == len(spike_counts)
    spike_counts = pd.DataFrame(spike_counts, index=kc_index, columns=odor_index)

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
        # (would need to test extra_orn_deltas code to be sure...)
        '''
        if make_plots:
            # causes errors re: duplicate ticklabels in some of the orn_deltas plots
            # currently (would need to remove 'eb' from all of those plots, but also
            # prob want to remove all extra_orn_deltas odors for them. still need to
            # keep in returned responses tho)
            warn('fit_mb_model: setting make_plots=False since not currently supported '
                'in extra_orn_deltas case'
            )
        # TODO restore? needed to keep this true to test some code below
        # (in plot_example_dynamics, but also interaction w/ this case)
        #make_plots = False
        '''

    if n_claws_active_to_spike is None:
        spont_in = pd.Series(index=kc_index, data=spont_in.squeeze())
    else:
        assert one_row_per_claw
        spont_in = pd.Series(index=claw_index, data=spont_in.squeeze())

    param_dict = {
        'fixed_thr': fixed_thr,

        # still want these here in the use_connectome_APL_weights=True case?
        # (yes, but they will be popped from output and serialized separately in that
        # case, like kc_spont_in always is. they would otherwise interfere w/ how i'm
        # currently formatted + saving params into CSVs). They are also saved as
        # separate pickles when they are popped.
        'wAPLKC': wAPLKC,
        'wKCAPL': wKCAPL,

        # TODO rename to just 'spont_in' (mainly now that it can sometimes be of len #
        # claws now, rather than just # KCs)
        'kc_spont_in': spont_in,
    }
    # TODO care to change how these (+ KC<>APL weights) are handled in
    # per_claw_pn_apl_weights=True case?
    if prat_boutons and not per_claw_pn_apl_weights:
        assert wAPLPN is not None and wPNAPL is not None
        param_dict['wAPLPN'] = wAPLPN
        # TODO TODO this causing problems? why do i have in some outputs (w/ fixed inh
        # params) w/ scales, but not others (call before fixed inh one, that was tuned)?
        param_dict['wPNAPL'] = wPNAPL

        assert wAPLPN_scale is not None and wPNAPL_scale is not None
        param_dict.update({'wAPLPN_scale': wAPLPN_scale, 'wPNAPL_scale': wPNAPL_scale})

        try:
            odor_stats = rv.kc.odor_stats

            # should be odor length list of 4-element vectors,
            # with the elements being these column values (in order)
            odor_stats = pd.DataFrame(odor_stats, index=orn_deltas.columns, columns=[
                'max_kc_apl_drive', 'avg_kc_apl_drive', 'max_bouton_apl_drive',
                'avg_bouton_apl_drive'
            ])

            param_dict['odor_stats'] = odor_stats
        except AttributeError:
            pass

        # called again to modify title in fit_and_plot..., after this fn returns
        title += weight_debug_suffix(param_dict)

    # TODO delete
    '''
    if pn_apl_scale_factor != 1:
        # TODO TODO TODO replace w/ one factor for each (of 4 instead of of 2) weight,
        # rather than one ratio between two of them?
        param_dict['pn_apl_scale_factor'] = pn_apl_scale_factor
    '''
    #

    # NOTE: currently will probably not be able to achieve these after APL tuning
    # (until olfsysm is modified to tune within subtypes), but (if not None) per-type
    # thresholds will be picked to achieve sparsities here scaled by
    # mp.kc.sp_factor_pre_APL (default=2.0).
    if equalize_kc_type_sparsity:
        assert type2target_response_rate is not None
        param_dict['type2target_response_rate'] = type2target_response_rate

        # both of these should also be defined if equalize_kc_type_sparsity=True
        param_dict['type2thr'] = thr_by_type.to_dict()
        param_dict['retune_apl_post_equalized_thrs'] = retune_apl_post_equalized_thrs

    if use_connectome_APL_weights:
        assert wAPLKC_scale is not None and wKCAPL_scale is not None
        param_dict.update({'wAPLKC_scale': wAPLKC_scale, 'wKCAPL_scale': wKCAPL_scale})

    tuning_dict = {
        # TODO expose at least these first two (sp_acc, max_iters) as kwargs.
        # prob also sp_lr_coeff.
        # TODO TODO + maybe default to smaller tolerance (+ more iterations if
        # needed). what currently happens if tolerance not reached in max_iters?
        # add my own assertion (in this script) that we are w/in sp_acc?
        #
        # parameters relevant to model threshold + APL tuning process
        # default=0.1 (fraction +/- sp_target)
        'sp_acc': mp.kc.sp_acc,

        # default=10
        'max_iters': mp.kc.max_iters,

        'sp_lr_coeff': mp.kc.sp_lr_coeff,
        'apltune_subsample': mp.kc.apltune_subsample,

        # should be how many iterations it took to tune,
        'tuning_iters': rv.kc.tuning_iters,

        # removed tuning_time_s from this, because it would cause -c checks to fail
        # TODO fix -c/whatever to not check this in the first place? kinda nice to
        # have...
    }

    if mp.kc.tune_apl_weights:
        print('tuning parameters:')
        pprint(tuning_dict)
        print()

    param_dict = {**param_dict, **tuning_dict}

    # TODO expose (+rename?) extra_verbose?
    extra_verbose = False
    # TODO put prints below behind a verbose flag? (this whole conditional basically)
    _print_response_rates(responses, verbose=extra_verbose)

    # TODO can i not just assert `KC_TYPE in responses.index.names` now?
    if extra_verbose and KC_TYPE in responses.index.names:
        n_kcs_by_type = responses.index.get_level_values(KC_TYPE).value_counts()

        silent_kcs = (responses == 0).T.all()
        silent_frac_by_type = silent_kcs.groupby(KC_TYPE).sum() / n_kcs_by_type
        # TODO still (at least if verbose) print this (one number) if no type
        print()
        print('silent_frac_by_type:')
        print(silent_frac_by_type.to_string())
        print()

        # TODO check that if i use uniform/whatever instead of hemibrain wPNKC, this
        # bias favoring heavily responsive gamma KCs goes down?
        # TODO TODO should wPNKC be set in a way that normalizes within type, to
        # avoid this bias introduced? or should i address it in some other way instead?
        # in a separate step than what i end up using to implement the Inada paper a'/b'
        # vs other cell type responsiveness (which might just be threshold in their
        # paper?)
        # TODO TODO should i test that i can actually use non-int wPNKC now?

    assert (mp.time_pre_start < mp.time_start < mp.time_stim_start < mp.time_stim_end <
        mp.time_end
    )
    # from default parameters:
    # p.time.pre_start  = -2.0;
    # p.time.start      = -0.5;
    # p.time.end        = 0.75;
    # p.time.stim.start = 0.0;
    # p.time.stim.end   = 0.5;
    # p.time.dt         = 0.5e-3;
    delete_pretime = True
    # pre_start and start are both before odor onset (mp.time_stim_start)
    if not delete_pretime:
        t0 = mp.time_pre_start
        t1 = mp.time_end
    else:
        t0 = mp.time_start
        t1 = mp.time_end

        # TODO delete
        #print_curr_mem_usage(end='')
        #print(', before osm.remove_all_pretime()')
        #

        # TODO change this to also remove even more at start/end? don't need after
        # stim_end, right?
        #
        # huge memory savings from this
        osm.remove_all_pretime(mp, rv)

        # TODO delete
        # TODO need to time.sleep() first to get an accurate reading?
        # (not sure, but definitely did see it drop with this)
        #time.sleep(0.2)
        #print_curr_mem_usage(end='')
        #print(', after osm.remove_all_pretime()')
        #

    # So that i can use linspace instead of arange. arange takes step, but last point
    # seems numerically slightly off the even last point I get with linspace.
    n_samples = int(round((t1 - t0) / mp.time_dt))
    ts = pd.Series(name='seconds', data=np.linspace(t0, t1, num=n_samples))

    # time_start = "start of KC stimulation" [from spont PN activity]
    # (it's before stimulus comes on at time_stim_start)
    start_idx = np.searchsorted(ts, mp.time_start)
    stim_start_idx = np.searchsorted(ts, mp.time_stim_start)
    stim_end_idx = np.searchsorted(ts, mp.time_stim_end)

    # TODO why is this seemingly a list of arrays, while the equiv kc variable seems to
    # be an array immediately? binding code seems similar...
    orn_sims = np.array(rv.orn.sims)
    assert orn_sims.shape[-1] == n_samples

    # orn_sims.shape=(110, 22, 5500)
    # also a list out of the box
    # pn_sims.shape=(110, 22, 5500)
    pn_sims = np.array(rv.pn.pn_sims)

    # orn_sims is of shape (n_odors, n_glomeruli, n_timepoints)
    assert pn_sims.shape[-1] == n_samples

    if bouton_dynamics:
        bouton_sims = np.array(rv.pn.bouton_sims)
        assert bouton_sims.shape[-1] == n_samples

    if tune_on_hallem and not hallem_input:
        orn_sims = orn_sims[n_hallem_odors:]
        pn_sims = pn_sims[n_hallem_odors:]
        bouton_sims = bouton_sims[n_hallem_odors:]

        if plot_example_dynamics:
            # TODO also subset other dynamics vars, at least if we are gonna return them
            # (or use them for plot_example_dynamics)?
            raise NotImplementedError('prob need to subset those vars below')

    # TODO rename (+ move rv to param) -> move out of fit_mb_model? just to
    # declutter this fn...
    # TODO worth looking in to no-copy ways we might be able to access these
    # variables via pybind11 + eigen? claw_sims especially takes up a lot of
    # memory (~2.5GiB for 6 odors...). would probably require at least changing
    # olfsysm pybind11 C++ binding code.
    # see: https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html
    # (may need to change so these values aren't all std::vector<Matrix|Row>,
    # but are instead an array with one more dimension for # odors. seems
    # like we may have to roll much more of our own binding code to support objects
    # that are not one simple Eigen object.
    # see: https://stackoverflow.com/questions/63159807 seems complicated...)
    # TODO may be easier to just have C++ write it, and make a python fn to read it
    # into a numpy array (e.g. via fromfile)?
    # TODO delete? was this really not working?
    def _get_sim_var(name, cpp_obj=rv.kc) -> np.ndarray:
        # all appear as #-odor-length lists w/o np.array(...) applied
        arr = np.array(getattr(cpp_obj, name))

        # TODO add option to slice all to exclude time periods that will only
        # contain 0/simliar? (to save on size of outputs, would just somewhat
        # complicate loading [as we'd have to either invert the process, or save in
        # a format that facilitates aligning all the different outputs])

        _debug = False
        if not _debug:
            return arr

        # TODO delete below?

        print(f'{name=}')

        # this should be a vector of length equal to #-timepoints
        all0 = (arr == 0).all(axis=(0,1))

        # NOTE: only not true for nves_sims, which isn't all 0 anywhere
        # (it's all 1 everywhere. prob disabled. see below.)
        #print(f'{all0[:(start_idx + 1)].all()=}')
        # only vm_sims has this as False (excluding nves_sims)
        # (others still all 0 at +3 too)
        #print(f'{all0[:(start_idx + 2)].all()=}')

        # TODO use this (to get mean/min/max/etc within)?
        after_start = arr[:, :, start_idx:]
        # TODO also print min/max/mean/[np.quantile 0.5] for each?
        # maybe only within non-zero part? and/or only within stim window?
        if all0.any():
            first_non_all0_idx = np.argwhere(~all0)[0][0]
            print(f'{first_non_all0_idx=}')

            # last index all 0
            last_all0_idx = np.argwhere(all0)[-1][0]
            print(f'{last_all0_idx=}')

            all0_to_last = (arr[:, :, :(last_all0_idx + 1)] == 0).all()
            print(f'{all0_to_last=}')
        else:
            print('not all0 anywhere!')
        print()

        return arr

    if return_dynamics or plot_example_dynamics:
        # TODO TODO only process and serialize these one at time (saving to plot_dir in
        # here, rather than returning), to try to avoid memory issues? i assume making a
        # dataarray object (or even returning as a numpy array?) will create a copy
        # (increasing amount of used memory)? will prob require slight change to
        # plotting code below, either interleaving plotting code, or defining dataframe
        # from single example odor up here (latter prob preferable)
        #
        # (whether for that reason, or perhaps only b/c additional processing I was
        # trying was using enough extra memory to cause issues, this section had been
        # killed before, when trying to process claw_sims)
        #
        # TODO maybe serialize as some kind of format i can load in a memory mapped
        # manner, to not need to change any of plotting code below?

        # TODO which of these are all 0 before stim_start_idx?
        #
        # these first 3 are of shape (#-odors, #-KCs, #-timepoints)
        #
        # "Membrane voltage"
        vm_sims = _get_sim_var('vm_sims')

        # presumably 0/1 recording which time bins have spikes in them?
        spike_recordings = _get_sim_var('spike_recordings')
        assert set(spike_recordings.flat) == {0, 1}, (f'{set(spike_recordings.flat)=}\n'
            f'{spike_recordings.shape=}'
        )

        # this has been 0 by default my whole time working with the model. presumably
        # matt also decided pretty quickly to not use this, tho not sure why.
        #
        # I assume this is either not in Ann's model, or primarily not used there?
        # TODO TODO is this true? what might we get from adding this back in to the
        # model?
        #
        # if `mp.kc.ves_p == 0`, the "vesicle depletion" part of olfsysm is disabled
        if mp.kc.ves_p != 0:
            # "vesicle depletion factor"
            nves_sims = _get_sim_var('nves_sims')
            # this is probably  only the case if we get it when `mp.kc.ves_p != 0`,
            # which i haven't tested, and not sure we care to
            #assert (nves_sims == 1).all()
        else:
            # TODO TODO assert that (nves_sims == 1).all() here. that variable does
            # seem to be used in C++ tho... is it just that the math works out to leave
            # it all 1? or is it maybe not properly returned?
            warn('nves_sims are disabled, as has typically been the case for olfsysm')

        # TODO TODO TODO confirm these both have correct shape and are populated
        # correctly in multicompart APL versions of model (+ make plot to show)
        #
        # these two are of shape (#-odors, 1, #-timepoints), with the length-1 component
        # removed by squeeze. both properties of scalar APL (hence 1 vs #-KCs).
        #
        # "APL potential" (mV?)
        inh_sims = _get_sim_var('inh_sims').squeeze()
        # "KC->APL synapse current"
        # TODO so this must be the current across all KCs, right?
        Is_sims = _get_sim_var('Is_sims').squeeze()
        #

        # NOTE: just chose 'stim' for that dim name b/c xarray complained that
        # 'odor' overlapped with the odor_index level of the same name.
        #
        # wasn't also an issue with glomerulus_col being the .name of
        # glomerulus_index, presumably b/c not multiple levels there.
        coords = {'stim': odor_index, 'time_s': ts}

        al_dims = ['stim', glomerulus_col, 'time_s']
        glom_coords = {**coords, glomerulus_col: glomerulus_index_unique}

        orn_sims = xr.DataArray(data=orn_sims, dims=al_dims, coords=glom_coords)
        pn_sims = xr.DataArray(data=pn_sims, dims=al_dims, coords=glom_coords)
        if bouton_dynamics:
            bouton_sims = xr.DataArray(data=bouton_sims,
                dims=['stim', 'bouton', 'time_s'],
                coords={**coords, 'bouton': bouton_index}
            )

        kc_dims = ['stim', 'kc', 'time_s']
        kc_coords = {**coords, 'kc': kc_index}
        vm_sims = xr.DataArray(data=vm_sims, dims=kc_dims, coords=kc_coords)

        spike_recordings = xr.DataArray(data=spike_recordings, dims=kc_dims,
            coords=kc_coords
        )
        if mp.kc.ves_p != 0:
            nves_sims = xr.DataArray(data=nves_sims, dims=kc_dims, coords=kc_coords)
            # TODO TODO assert all 1? or plot to get a sense of what's happening?

        # may change if we end up having multiple APL compartments
        apl_dims = ['stim', 'time_s']
        apl_coords = coords
        inh_sims = xr.DataArray(data=inh_sims, dims=apl_dims, coords=apl_coords)
        Is_sims = xr.DataArray(data=Is_sims, dims=apl_dims, coords=apl_coords)

        if one_row_per_claw:
            # TODO TODO test to see whether _get_sim_var('claw_sims') or defining
            # DataArray from it below are increasing memory usage (-> more motivation to
            # loop over and process vars one at a time, if so)

            # TODO what are proper units? matter? rename (here and in olfsysm) to
            # include proper units?
            claw_sims = _get_sim_var('claw_sims')

            claw_dims = ['stim', 'claw', 'time_s']
            claw_coords = {**coords, 'claw': claw_index}

            # NOTE: the creation of this DataArray does not (alone) seem to increase the
            # memory usage (at least not the RSS)
            claw_sims = xr.DataArray(data=claw_sims, dims=claw_dims, coords=claw_coords)

            # TODO maybe don't overwrite... (or do for all by default? maybe
            # starting from a bit before stim_start_idx and ending a bit after
            # stim_end_idx?)
            # TODO TODO how different is sum within this slice (across time)?
            # vs some across time in full thing?
            # TODO TODO maybe i should add more of a return to baseline per claw?
            # time constant like Vm calculation has?
            #
            # https://stackoverflow.com/questions/50009978
            # NOTE: .copy() needed to actually free memory from region other than slice.
            # might consume more memory while it's happening? not sure.
            claw_sims = claw_sims.isel(
                time_s=slice(stim_start_idx, stim_end_idx)
            ).copy()

            # <xarray.DataArray (stim: 18, claw: 9472)>
            claw_sims_sums = claw_sims.sum(dim='time_s')
            param_dict['claw_sims_sums'] = claw_sims_sums.to_pandas()

            claw_sims_maxs = claw_sims.max(dim='time_s')
            param_dict['claw_sims_maxs'] = claw_sims_maxs.to_pandas()

        # TODO TODO save each one at a time in here (before getting the next var from
        # olfsysm, which copies), rather than returning all, to help save on memory
        # (otherwise total usage can easily get to low 20GiBs w/ even 6 odors)
        if return_dynamics:
            dynamics_dict = {
                'orn_sims': orn_sims,
                'pn_sims': pn_sims,

                'inh_sims': inh_sims,
                'Is_sims': Is_sims,

                'vm_sims': vm_sims,
                'spike_recordings': spike_recordings,
            }
            if mp.kc.ves_p != 0:
                dynamics_dict['nves_sims'] = nves_sims

            if one_row_per_claw:
                dynamics_dict['claw_sims'] = claw_sims

            if bouton_dynamics:
                dynamics_dict['bouton_sims'] = bouton_sims

            assert not any(k in param_dict for k in dynamics_dict.keys())
            param_dict.update(dynamics_dict)

    if plot_example_dynamics:
        # TODO remove make_plots part of this? (assuming i can't easily get those plots
        # to work w/ extra_orn_deltas, but i can easily get those to work w/ this)
        # TODO relax assertion to test some of this code?
        assert plot_dir is not None and make_plots
        assert plot_dir.exists(), f'{plot_dir=} did not exist'

        odor_values = odor_index.get_level_values('odor')
        for odor in ('t2h @ -3', 'kmix0 @ 0'):
            if odor in odor_values:
                break
            # for if we don't find one of the preferred example odors
            odor = None

        if odor is None:
            odor = odor_values[-1]
            warn(f'picking last odor {odor} for example dynamics plots')

        # TODO try to replace usage w/ sel(odor=odor) below. seems like it should work
        example_odor_idx = odor_values.get_loc(odor)

        # TODO still keep one version of plot showing full timecourse, in case
        # understanding the ramp up ~"start" time (< "stim_start") is important?
        # (just set xlim differently and re-save?)

        # TODO factor out this plotting (so someone could run on saved dynamics
        # outputs)? (may be hard to do that if i still want to only process one var at
        # a time... unless i'm loading each file that we already saved, within the
        # consolidated plotting fn)

        # seem to need constrained layout to get fig.legend() reliably outside bounds of
        # Axes (at least w/o manual positioning...)?
        # TODO TODO TODO separate ax for claws too? (refactor natmix_data/analysis.py
        # dynamics plotting code to start from?)
        if prat_boutons and not per_claw_pn_apl_weights:
            fig, (ax, bouton_ax, spike_raster_ax) = plt.subplots(nrows=3, layout='constrained',
                sharex=True, figsize=(10, 15)
            )
        else:
            fig, (ax, spike_raster_ax) = plt.subplots(nrows=2, layout='constrained',
                sharex=True, figsize=(10, 10)
            )

        # TODO make fn (/find a library for) plotting a stimlus bar along some time axis
        # (-> use that here instead)?
        #
        # used to have these are 'stim start' / 'stim end' in legend, but legend is
        # already too busy as-is.
        ax.axvline(mp.time_stim_start, color='k')
        ax.axvline(mp.time_stim_end, color='k')

        def _plot_normed(xs, **kwargs) -> None:
            # xs should probably be Series, but could also be some xarray type with a
            # .max() method
            ax.plot(ts, xs / xs.max(), **kwargs)

        # TODO is there some reason orn_sims.min() is not sfr.min()?
        # min seems to be 17, if i'm computing it right...
        # is it just that orn_sims is not in firing rate units?
        # (was it the sign that concerned me? or that it's larger than sfr.min()? what
        # is the ordering of them? delete comment?)

        # TODO always plot whichever glomerulus has the biggest response (and say so)?
        # would make this code useful for more input data...
        # TODO change to work if DL5 not in input (it will be, but it may not respond
        # meaningfully to panel odors)?
        glom = 'DL5'
        # TODO try to replace usage w/ sel(glomerulus=glom) below?
        example_glom_idx = glomerulus_index_unique.get_loc(glom)

        # TODO delete? after resolving glomerulus_index[_unique] (/bouton_index) issue
        should_be_glom = orn_sims[example_odor_idx, example_glom_idx].glomerulus
        assert should_be_glom.size == 1
        assert should_be_glom.item(0) == glom
        # TODO also want to add anything similar for check on bouton_index?
        #

        # TODO TODO use new natmix plotting code to plot dynamics here? maybe alongside
        # existing plots? have i already refactored that code?

        # TODO directly index odor/glom by name, now that we are using xarray?
        #
        # TODO add units for these (firing rate in Hz?) (via y-axis label?)
        #
        # units seems to be firing rates (absolute i think. actually, there are some
        # negative values, even in hallem_input=True [w/ matt config] case. that's not a
        # mistake though, is it?)
        _plot_normed(orn_sims[example_odor_idx, example_glom_idx], label=f'{glom} ORN')
        _plot_normed(pn_sims[example_odor_idx, example_glom_idx], label=f'{glom} PN')

        # TODO TODO leave a copy of [glomerulus, pn_id, bouton_id] levels in wPNKC rows
        # too (or precompute claw->glom mapping, for plotting)? trying to stack() all
        # wPNKC.columns.names to index triggers OOM killer (at least, while already near
        # limit in this code, b/c claw_sims)

        # TODO limit all data to this in advance (to not need to only use in bouton
        # plotting below)?
        xlim = [-0.05, 0.7]
        xmin, xmax = xlim

        if bouton_dynamics:
            # TODO try to replace other (int based) indexing of DataArrays w/ calls like
            # this. seems this might throw away the selected metadata, if i ever care
            # about that... not sure if it's avoidable
            #
            # squeeze() here to remove length 1 odor dimension (would we have that if we
            # didn't have 'panel' level in index? squeeze still keeps that metadata, but
            # removes from index and .sizes)
            #
            # NOTE: slicing bouton this way (rather than glomerulus=glom) preserves all
            # the levels of the bouton index, and the 'bouton' dimension/coordinate
            # name (so we can always do sizes['bouton'], no matter whether the bouton
            # index has PN_ID in the levels or not).
            glom_boutons = bouton_sims.sel(odor=odor,
                bouton=(bouton_sims.glomerulus == glom)
            ).squeeze()

            n_glom_boutons = glom_boutons.sizes['bouton']

            if prat_boutons and not per_claw_pn_apl_weights:
                # TODO pretty up?
                for x in sorted(glom_boutons.bouton_id.values):
                    # TODO TODO offset/whatever needed to make clear where the
                    # overlapping lines are (share code w/ claw handling in
                    # natmix_data/analysis.py? not sure i have a good solution there yet
                    # though)
                    bs = glom_boutons.sel(bouton_id=x).squeeze()

                    # otherwise the .max() for boutons that are fully 0 within odor
                    # window seems to occur before odor onset (from spont input,
                    # presumably)
                    bs = bs[(bs.time_s >= xmin) & (bs.time_s <= xmax)]

                    # TODO also say which glomerulus in legend (for comparison to
                    # glomerulus / claw axes) (it's all the same in this case. would
                    # only matter if we were also using the plotting code somewhere
                    # else, where we are actually looking at input w/ multiple boutons
                    # from diff glomeruli)
                    # TODO want PN ID in legend?
                    # TODO make separate axes legend (rather than one fig-level one also
                    # shared w/ above `ax`)
                    bouton_ax.plot(bs.time_s, bs,
                        # TODO TODO include weight (unscaled?) too (apl at least)
                        # TODO delete max from legend if i have good enough alternative
                        # way to see where the overlapping lines are
                        # TODO TODO TODO why is max apparently not in window plotted?
                        # take make just within that (seeing 16 for all that appear
                        # completely 0 in plot)
                        label=f'{x} (max={bs.max().item(0):.0f})'
                    )

                # TODO TODO TODO need to make the APL>PN inhibition more like LN>PN
                # inhibition in any way (but still dynamic? is LN>PN?), since it's
                # operating on different units? how are scales of the two values (KC
                # claw Vm vs bouton "firing rates") actually different?
                bouton_ax.set_ylabel('bouton "firing rate"')

                # TODO happy with?
                bouton_ax.legend(loc='upper right', title='bouton ID (max in odor):')
            else:
                bouton_sum = glom_boutons.sum('bouton')
                assert bouton_sum.dims == ('time_s',)

                # TODO why was Tianpei adding 1e-9 here? don't use plot_normed here, so
                # i can still do that here if needed? (or delete commented line)
                #ax.plot(ts, sum_activity / (sum_activity.max() + 1e-9),
                _plot_normed(bouton_sum,
                    label=f'{glom} PN (sum of {n_glom_boutons} boutons)'
                )

            # TODO maybe just label if boutons are all the same? (and only plot one
            # curve? shouldnt matter)
            # TODO also label w/ PN ID, when bouton_index has it

            # TODO TODO instead plot separate plot(s)[/lines] for each bouton (maybe
            # offset slightly or something else to allow showing multiple, even if they
            # are the exact same? diff line markers? alpha? either/or + legend? small
            # random x offsets? diff facets?)
            # (especially after truly adding PN<>APL dynamics)

            # TODO TODO + add test that establishes how scale of bouton activity in
            # dynamics relates (for one bouton, or summed across them, vs in
            # non-separate-bouton case)

        # TODO TODO drop non-responding KCs before all plots using them
        # (both dynamics plots and whichever spike raster things i ultimately use)
        # (only matter for mean Vm [+ by type] stuff? anything else?)
        # TODO also say in title (below) if we drop non-responders, if we do

        # plotting these before the KC mean[+types] now, since that part of the plot
        # can vary (in terms of # of lines), so having this earlier fixes the colors for
        # these
        # TODO TODO adapt to work w/ multiple compartments, when available
        # (especially when it's a small #, want to plot all. otherwise want to pick some
        # compartments that are more different to plot, or maybe plot all as a matrix?)
        _plot_normed(inh_sims[example_odor_idx], label='APL Vm')
        _plot_normed(Is_sims[example_odor_idx], label='APL current')

        # TODO TODO or maybe i don't want to always drop non-responders from vm_sims
        # plots? at least have a version not dropping them? (would need to get indices
        # from determination on spike_recordings / example_odor_spikes anyway)

        # TODO label plot with how many responders there are out of how many total
        # cells
        # TODO per kc_type labels w/ how many silent in each? too noisy? separate
        # plot/something for that instead?
        # ipdb> example_odor_spikes[example_odor_spikes.T.sum() > 0].shape
        # (214, 5500)
        # ipdb> example_odor_spikes.shape
        # (1830, 5500)
        example_odor_spikes = spike_recordings[example_odor_idx].to_pandas()

        # TODO maybe plot clusters? per cell-type clusters?
        # TODO or random sample a few and plot on axis by itself (w/ original scale. not
        # normalized)?
        example_odor_vm_sims = vm_sims[example_odor_idx]

        # taking mean over KCs, giving us a series of length equal to #-timepoints
        mean_vm_sims = example_odor_vm_sims.mean('kc')
        _plot_normed(mean_vm_sims, label='mean KC Vm')

        # TODO TODO make a version of all these plots w/ each quantity on a separate
        # facet (or at least, only those w/ comparable units sharing a facet), so that
        # axes can all have the right units

        # TODO TODO modify fit_mb_model to accept arbitrary wAPLKC/wKCAPL, like i
        # had for _wPNKC (-> use to play around w/ per-subtype scales from here)
        # (how equiv to code i've been adding to set thr by type? prob not? not sure i
        # need per-cell wAPLKC/wKCAPL once i get per-cell thr)
        # TODO or should the cell type variations happen by effectively scaling
        # spontaneous input to each cell (where fixed_thr currently added to that
        # anyway)? that feels much more likely to be equiv to just scaling the
        # thresholds/fixed_thr values per cell type... (scaling wPNKC could probably do
        # that?)
        # TODO TODO what was the issue with having a constant fixed_thr again (the
        # reason that ann/matt added to spontaneous firing rate)? ann explain in her
        # thesis / preprint?

        # TODO hline for threshold (per-KC, ofc)? to sanity check at least?
        # TODO also for spont_in

        # could also use `example_odor_vm_sims.coords['kc'].to_index()`, which was
        # defined from `kc_index`, and should be equiv. not sure if there's a more
        # idiomatic xarray way to count these w/o converting to a pandas Index
        # ipdb> kc_index.to_frame(index=False)[KC_TYPE].value_counts()
        # ab         802
        # g          612
        # a'b'       336
        # unknown     80
        vm_sims_by_type = example_odor_vm_sims.groupby(KC_TYPE).mean()

        # normalizing to max of 1, as elsewhere
        vm_sims_by_type = vm_sims_by_type / vm_sims_by_type.max('time_s')
        kc_type_labels = [f'mean {x}-KC Vm' for x in vm_sims_by_type[KC_TYPE].values]

        # normalized within each type above, so no need for _plot_normed
        ax.plot(ts, vm_sims_by_type.T, label=kc_type_labels)

        # TODO restore something like my spikes -> spike times -> sns.rugplot for
        # best responding cell (never committed, unfortunately), or just directly use
        # imshow/similar (like rastermap does)
        # TODO TODO -> use to show spiking against threshold/spont_in context for an
        # example cell? or do that in separate plot(s) specifically for it?

        # NOTE: plot_spike_rasters currently drops silent cells itself anyway
        # TODO but modify it to have it say somewhere how many it dropped?
        # or like a kwarg flag to enabled putting that info in y-label or something?
        # TODO natmix_data/analysis.py already have method to deal with this?
        # (not sure it does...)
        # TODO at least remove this axes if singular?
        plot_spike_rasters(example_odor_spikes, ax=spike_raster_ax)

        # applies to both ax and spike_raster_ax, b/c sharex=True above
        ax.set_xlim(xlim)
        ax.set_xlabel('time (s)')

        ax.set_ylabel('normalized response (max=1)')

        ax.set_title(f'{title}\nmodel {glom} (& downstream pop.) response to {odor}')
        # TODO happy with?
        ax.legend(loc='upper right')
        # TODO delete? trying to replace w/ per axis ones
        #fig.legend(loc='outside right center')

        # TODO move under model_internals stuff saved below?
        # TODO get ' @ ' from hong2p.olf?
        savefig(fig, plot_dir, f'model_dynamics_{odor.replace(" @ ", "_")}')

        # TODO TODO matshow clustered timeseries responses of all these?
        # (that even work on raw timeseries, without taking too long / too much memory?
        # what library to use?)
        # TODO row color for which PN? / which KC / whatever?
        # TODO move above, and use to plot claws on sep axes
        glom_claws = wPNKC.loc[:, wPNKC.columns.get_level_values(glomerulus_col) == glom
            ].replace(0, np.nan).dropna(how='all').stack(wPNKC.columns.names)
        # TODO TODO TODO plot w/ cluster_timeseries_and_plot

        # TODO delete
        print('finish some kind of per-claw plot')
        #breakpoint()
        #

    # TODO so should i average starting a bit after stim start? seems it still take
    # a bit to peak. what did matt do?
    # in (DL5, t2h) case, PN peak is ~0.047, and ORN plateaus after ~0.077

    # plot_example_dynamics=True code above does analyze the DataArrays that include
    # time, without averaging over it
    orn_df = pd.DataFrame(index=odor_index, columns=glomerulus_index_unique,
        data=orn_sims[:, :, stim_start_idx:stim_end_idx].mean(axis=-1)
    )

    if sep_boutons and not bouton_dynamics:
        # in these cases (currently only add_PNAPL_to_KCAPL or replace_KCAPL_with_PNAPL)
        # wPNKC will have had it's separate boutons aggregated over, right before wPNKC
        # is assigned into olfsysm variable for it, so it actually just has one column
        # per glomerulus (and no PN / bouton info in row metadata [or otherwise across
        # rows])
        bouton_index = glomerulus_index_unique
    del glomerulus_index

    pn_df = pd.DataFrame(index=odor_index, columns=glomerulus_index_unique,
        # TODO what is this mean over? time, it seems? rename these vars to be more
        # clear about that?
        data=pn_sims[:, :, stim_start_idx:stim_end_idx].mean(axis=-1)
    )

    if bouton_dynamics:
        # TODO even want this? at least plot below?
        bouton_df = pd.DataFrame(index=odor_index, columns=bouton_index,
            data=bouton_sims[:, :, stim_start_idx:stim_end_idx].mean(axis=-1)
        )

    if extra_orn_deltas is not None:
        orn_df = orn_df.drop(index=extra_orn_deltas.columns)
        pn_df = pn_df.drop(index=extra_orn_deltas.columns)

        # TODO TODO also subset bouton_df (/ claw stuff? below?)

        # TODO (delete? still relevant? some commented assertion that was failing?) also
        # drop from orn_deltas? assertion failing below b/c not doing so (or is that how
        # they get returned? maybe just drop for purpose of assertion or within whatever
        # plotting, if important)

    if sim_odors is not None:
        # TODO have parse_odor_name return input (rather than current ValueError), if
        # input doesn't have '@' in it (or add in model_test.py r1 call that currently
        # is failing b/c of this)
        input_odor_names = {olf.parse_odor_name(x) for x in sim_odors}
    else:
        # TODO need to exclude extra_orn_deltas from this?
        # (given i'm checking subset below, prob fine. but COULD also prob exclude extra
        # odors, if that helps simplify code)
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

    # print(f"megamat_odor_names = {megamat_odor_names}")
    # print(f"input_odor_names  = {input_odor_names}")

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

            # TODO is it a problem that i'm now also only are only doing all below if
            # `not hallem_input`? (change any outputs in my main al_analysis.py
            # remy-paper analyses?) changed to fix impact of sorting on check in
            # model_test.py, but could also add kwarg to disable this sorting...

            orn_deltas = sort_odors(orn_deltas, panel=panel, warn=False)

            # TODO maybe only do this one on a copy we don't return? probably don't
            # really care if it's already sorted tho...
            # TODO even care to do this? if just for a corr diff thing, even need?
            responses = sort_odors(responses, panel=panel, warn=False)
            spike_counts = sort_odors(spike_counts, panel=panel, warn=False)

            orn_df = sort_odors(orn_df, panel=panel, warn=False)
            pn_df = sort_odors(pn_df, panel=panel, warn=False)

    # TODO still want?
    if 'panel' in responses.columns.names:
        assert 'panel' in spike_counts.columns.names
        responses = responses.droplevel('panel', axis='columns')
        spike_counts = spike_counts.droplevel('panel', axis='columns')

    if plot_dir is not None and make_plots:
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

        # TODO rename to be clear it's just orn/pn stuff?
        def _plot_internal_responses_and_corrs(subset_fn=lambda x: x, suffix='',
            **plot_kws):

            if extra_orn_deltas is not None:
                def _subset_fn(x):
                    try:
                        # will currently cause ValueError in some plotting calls below,
                        # if we don't drop (at least on megamat data, b/c eb is both in
                        # extra_orn_deltas and input odors there)
                        ret = x.drop(columns=extra_orn_deltas.columns)

                    except KeyError:
                        ret = x

                    return ret

                subset_fn = _subset_fn

            # TODO maybe subset these down to things that are still in
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
            # hack to fix all iteams are single element data array
            if isinstance(orn_df.values[0,0], xr.DataArray):
                orn_corr = plot_responses_and_corr(subset_fn(orn_df.applymap(lambda x: x.item())), plot_dir,
                    f'orns{suffix}', title=title, **plot_kws
                )
                plot_responses_and_corr(subset_fn(pn_df.applymap(lambda x: x.item())), plot_dir, f'pns{suffix}',
                    title=title, **plot_kws
                )
            else:
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
            def subset_sim_odors(df):
                # TODO delete? still necessary? seems we currently do this earlier
                if megamat:
                    df = sort_odors(df, panel='megamat', warn=False)
                #

                if sim_odors is None:
                    return df
                else:
                    return df.loc[:, [x in sim_odors for x in df.columns]]

            # assuming we won't be passing sim_odors for other cases (other than what?
            # which case is this? elaborate in comment), for now
            if sim_odors is not None:
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
        # TODO can i share silent cell handling w/ plot_example_dynamics stuff that
        # also wants to drop silent KCs (might be tricky...)?
        if drop_silent_cells_before_analyses:
            title += _get_silent_cell_suffix(responses)
            model_kc_corr = drop_silent_model_cells(responses).corr()
            # drop_silent_model_cells should also work w/ spike count input
            spike_count_corr = drop_silent_model_cells(spike_counts).corr()
        else:
            model_kc_corr = responses.corr()
            spike_count_corr = spike_counts.corr()

        if extra_orn_deltas is not None:
            # TODO TODO why do we not seem to always still have them here?
            try:
                # TODO drop earlier on a copy of responses/spike_counts (still need to
                # return in main version of those variables)
                model_kc_corr = model_kc_corr.drop(index=extra_orn_deltas.columns
                    ).drop(columns=extra_orn_deltas.columns)
            except KeyError:
                pass

            try:
                spike_count_corr = spike_count_corr.drop(index=extra_orn_deltas.columns
                    ).drop(columns=extra_orn_deltas.columns)
            except KeyError:
                pass
            # TODO also need to drop from orn_delta_corr? anything else?

        # for sanity checking some of the diffs. should also be saving this outside.
        plot_corr(model_kc_corr, plot_dir, f'kcs_corr{suffix}', title=title)
        #
        # TODO also sanity check by extra plot_corr calls w/ orn_[delta_]corr
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
    # TODO why checking sim_odors is not None if i'm just using hallem_sim_odors here?
    # this a mistake?
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
        # for more glomeruli / diff runs?)a

    # TODO delete this (/ make it "private" w/ underscore prefix)?
    # (any code actually using it currently?)
    # TODO doc example of how to use these correctly?
    if return_olfsysm_vars:
        # NOTE: these objects not currently pickleable, and may (or may not) use a lot
        # of memory to keep them around.
        warn('also returning olfsysm vars in param dict! may need to avoid keeping '
            'references to too many of these objects. not supported by '
            'fit_and_plot_mb_model!'
        )
        param_dict['mp'] = mp
        param_dict['rv'] = rv

    # TODO also return model if i can make it pickle-able (+ verify that. it's possible,
    # but not likely, that it can already be [de]serialized)
    #
    # TODO maybe in wPNKC index name clarify which connectome they came from (or
    # something similarly appropriate for each type of random draws)
    return responses, spike_counts, wPNKC, param_dict


# NOTE: this does control n_seeds used in calls made from model_mb_responses (currently
# explicitly referenced in most entries in model_kw_list), but does not currently set
# the default n_seeds for fit_and_plot_mb_model
# TODO change other code that uses this to compute # seeds rather than using this
# hardcoded value? currently may not get expected outputs sometimes
N_SEEDS = 100
# TODO just add cli flag for this, at this point?
# (for testing code faster, in a way that includes n_seeds > 1 model cases)
#N_SEEDS = 3

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
# NOTE: trying this again 2025-04-14 to see if Betty likes it now for fig 3B.
# IQR (i.e. 25th - 75th percentile)
# TODO TODO TODO special case this one for 3B (similar to how i have a separate param
# for 2E), if they go with it
# TODO TODO how do they like this one for 3B?
#seed_errorbar = ('pi', 50)
#
# TODO restore? for things other than 3B at least?
seed_errorbar = ('ci', 95)

# was at B'c request to make new 2E versions using ('ci', 95), taking the first 20
# (/100) seeds.
#
# TODO TODO TODO update paper methods. only mention in fig2E legend that it was first 20
# seeds, and remove sentence about subsetting to first 20 seeds from end of that methods
# section
# TODO TODO wait, which have i been using? what are current paper plots using:
# (pretty sure all plots say in title if they are only using first 20 seeds, and only 2E
# says that in modeling.svg)
# - 3B?
# - S1C?
# - S1D?
# - validation/megamat sparsity plots?
# NOTE: fig 2E has this configured separately (via `fig2e_n_first_seeds`, currently 20)
# TODO clarify. not sure yet if she wants me to handle other seed_errorbar plots
# this way too... (don't think we do)
# TODO (still want to revert? idk... only if betty asks) revert to 20 (but maybe ignore
# for 3B scatterplots [and S1C?])
#n_first_seeds_for_errorbar = 20
n_first_seeds_for_errorbar = None

# TODO warn for anything using n_first_seeds other than module-level
# n_first_seeds_for_errorbar (have enough context for meaningful message? inspect?)?
# TODO also explain (in another line of text?) what seed=('pi', 50) means (+ maybe
# similar confusing ones). ('pi', 50) means "percentile interval" for 25th - 75th
# percentiles.
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
        err_text += (f'\nonly analyzing first {n_first_seeds}/{N_SEEDS} '
            'seeds'
        )

    return err_text, fname_suffix


seed_err_text, seed_err_fname_suffix = _get_seed_err_text_and_fname_suffix()

# TODO factor to hong2p.util
# TODO use in other places that do something similar?
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

    # TODO use # of seeds actually in df instead of N_SEEDS?
    #
    # wouldn't play nice if there were ever e.g. a diff number of cells per seed, but
    # that's not how it is now. this assertion isn't super important though, just a
    # sanity check.
    assert np.isclose(len(subset) / len(df), min(n_first_seeds, N_SEEDS) / N_SEEDS)

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
        n_odors_per_cell_counts = reindex(n_odors_per_cell_counts, n_odor_index
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
        # TODO some way to get groupby->apply to preserve .name _n_odors2frac_per_cell
        # sets? (so we don't have to duplicate that here)
        frac_responding_to_n_odors.name = frac_responding_col

        # TODO reset_index necessary?
        frac_responding_to_n_odors = frac_responding_to_n_odors.reset_index()

    # should only happen for modelling inputs w/ hemibrain wPNKC (as the other wPNKC
    # options should have a 'seed' level)
    else:
        frac_responding_to_n_odors = _n_odors2frac_per_cell(n_odors_per_cell)

    # NOTE: did just downgrade to matplotlib==3.4.3, so this is not currently an issue.
    # hopefully seaborn fixes it soon, so I can upgrade back to >=3.7.3
    # (see comments below)
    #
    # TODO did this actually have negative values? (just in the way seaborn calculates
    # the err, at least up to 0.13.2)
    #
    # ...
    # pebbled_6f/pdf/ijroi/mb_modeling/megamat/dff_scale-to-avg-max__data_pebbled__hallem-tune_False__pn2kc_uniform__n-claws_7__drop-plusgloms_False__target-sp_0.0915__n-seeds_100/sparsity_per_odor.pdf
    # Uncaught exception
    # Traceback (most recent call last):
    #   File "./al_analysis.py", line 13234, in <module>
    #     main()
    #   File "./al_analysis.py", line 13223, in main
    #     model_mb_responses(consensus_df, across_fly_ijroi_dir,
    #   File "/home/tom/src/al_analysis/mb_model.py", line 8232, in model_mb_responses
    #     params_for_csv = fit_and_plot_mb_model(panel_plot_dir,
    #   File "/home/tom/src/al_analysis/mb_model.py", line 5222, in fit_and_plot_mb_model
    #     plot_n_odors_per_cell(responses_including_silent, ax,
    #   File "/home/tom/src/al_analysis/mb_model.py", line 3280, in plot_n_odors_per_cell
    #     sns.lineplot(frac_responding_to_n_odors, label=label, color=color,
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/seaborn/relational.py", line 507, in lineplot
    #     p.plot(ax, kwargs)
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/seaborn/relational.py", line 354, in plot
    #     ebars = ax.errorbar(
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/__init__.py", line 1446, in inner
    #     return func(ax, *map(sanitize_sequence, args), **kwargs)
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/matplotlib/axes/_axes.py", line 3636, in errorbar
    #     raise ValueError(
    # ValueError: 'yerr' must not contain negative values
    try:
        sns.lineplot(frac_responding_to_n_odors, label=label, color=color,
            markeredgecolor=color, **lineplot_kws
        )

    except ValueError as err:
        # TODO TODO raise if message doesn't match:
        # "'yerr' must not contain negative values"

        # TODO TODO maybe it wasn't still a problem after upgrading seaborn? error i'm
        # getting now is still a ValueError, but with a diff message. more basic cause
        # it seems? try to repro w/ full number of seeds
        # ValueError: Could not interpret value `frac_responding_to_n_odors` for `y`. An
        # entry with this name does not appear in `data`.
        #
        # still a problem after upgrading from seaborn 0.13.0 to 0.13.2 (which does
        # appear to be latest seaborn, as of 2025-04-14). could probably downgrade to
        # 3.4.3 (https://github.com/sktime/pytorch-forecasting/issues/1145) but not sure
        # that's a great long term solution. see also:
        # https://github.com/matplotlib/matplotlib/issues/26187
        #
        # ipdb> pp lineplot_kws
        # {'ax': <Axes: >,
        #  'err_style': 'bars',
        #  'errorbar': ('pi', 50),
        #  'legend': False,
        #  'linestyle': '-',
        #  'marker': 'o',
        #  'markerfacecolor': 'white',
        #  'seed': 1337,
        #  'x': 'n_odors',
        #  'y': 'frac_responding_to_n_odors'}
        #
        # ipdb> frac_responding_to_n_odors
        #        seed  n_odors  frac_responding_to_n_odors
        # 0     94895        0                    0.589548
        # 1     94895        1                    0.107784
        # 2     94895        2                    0.076756
        # 3     94895        3                    0.053892
        # 4     94895        4                    0.039194
        # ...     ...      ...                         ...
        # 1795  94994       13                    0.004899
        # 1796  94994       14                    0.000544
        # 1797  94994       15                    0.000544
        # 1798  94994       16                    0.000000
        # 1799  94994       17                    0.000000
        # [1800 rows x 3 columns]
        #
        # ipdb> frac_responding_to_n_odors.min()
        # seed                          94895.0
        # n_odors                           0.0
        # frac_responding_to_n_odors        0.0
        # dtype: float64
        # ipdb> frac_responding_to_n_odors.max()
        # seed                          94994.00000
        # n_odors                          17.00000
        # frac_responding_to_n_odors        0.63963
        # dtype: float64
        # ipdb> frac_responding_to_n_odors.isna().any()
        # seed                          False
        # n_odors                       False
        # frac_responding_to_n_odors    False
        import ipdb; ipdb.set_trace()

    if log_yscale:
        # TODO increase if needed (i.e. if we ever use fafb wPNKC / cell #s, which have
        # 2482 in fafb-left, and probably similar # in right)
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
# TODO try to fix corr calc to not re-order stuff (was that the issue?) -> delete
# this?
# NOTE: need to re-sort since corr_triangular necessarily (why? can't i have it sort
# to original order or something?) sorts internally
def _resort_corr(corr, add_panel, **kwargs):
    return sort_odors(corr, panel=add_panel, **kwargs)


# TODO factor to hong2p.viz
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
    # TODO TODO be clear about what unit we are bootstrapping over (+ what test is being
    # used to compute p-value). update doc to reflect that it's not just spearman this
    # can work with (also pearson)
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

        # (we would have raised an error already if method wasn't in one of two options
        # above)
        # pylint: disable=possibly-used-before-assignment
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


# TODO test w/ Series too (-> update type hint ["Arraylike"?] if we can get it to work)?
def step_around(center: Union[float, np.ndarray], param_lim_factor: float,
    param_name: Optional[str] = None, *, n_steps: int = 3, drop_negative: bool = True,
    drop_zero: bool = False) -> np.ndarray:
    # TODO doc

    assert param_lim_factor > 0

    # TODO check that center is positive? not sure some of the stuff below makes
    # sense if it's not (assumes the <=0 elements will be at start of list)

    # TODO rename this? it's not the size of the single steps, now is it? (no, it's the
    # total extent we'll step in either direction, though could be limited by 0)
    step_size = np.abs(center * param_lim_factor)

    # if center is a vector w/ shape (N,), and param_steps would have
    # length M w/ scalar center, param_steps will now be of shape (M, N).
    # len(param_steps) will be the same regardless.
    param_steps = np.linspace(center - step_size, center + step_size,
        num=n_steps
    )

    if drop_zero:
        assert drop_negative

    prefix = '' if param_name is None else f'{param_name=}: '
    if drop_negative:
        param_steps[param_steps < 0] = 0

        # TODO factor out? allow passing arbitrary value instead of just 0?
        def _last_zero_idx(xs):
            """Returns index of last 0 element in vector, or NaN if there isn't one.
            """
            assert xs.ndim == 1
            is_zero = xs == 0
            if not is_zero.any():
                return np.nan

            return np.argwhere(is_zero)[-1, 0]


        if (param_steps == 0).any():
            if isinstance(center, float):
                last_zero_idx = _last_zero_idx(param_steps)
            else:
                # TODO just do this call in both cases (should work for float too)?
                last_zero_idx = np.apply_along_axis(_last_zero_idx, 0, param_steps)

            # if we chose steps in such a way that (for 2d input), we could have some
            # columns without 0 values while other columns have them, would need to
            # handle this case. current method should produce them in all the same place
            # though.
            # TODO try removing this assertion (-> filling NaN below, to 0, so
            # first_idx_to_use uses all data for those columns) (would that even work
            # tho? cause then output cols would be of different length. might need to
            # postprocess instead?)
            assert not np.isnan(last_zero_idx).any()

            first_idx_to_use = last_zero_idx
            if drop_zero:
                first_idx_to_use += 1

            if param_steps.ndim > 1:
                assert param_steps.ndim == 2
                # this should also only list NaN once (if any), even if there are
                # multiple in input
                unique_first_indices = np.unique(first_idx_to_use)
                # if we chose steps differently (rather than being relative within
                # each column), could be possible to get different
                # first-nonzero-element indices. for now, it shouldn't be.
                # would just need to think about what i want the warning message to
                # be in that case.
                assert len(unique_first_indices) == 1

                # unless i want to slice within each column below (currently no
                # point for same reason i can rely on a single unique value above),
                # and then somehow combine together (w/ potentially diff lengths),
                # need to just get a single integer to slice with.
                #
                # NOTE: warning below also assumes it's a single integer
                first_idx_to_use = unique_first_indices[0]

            if first_idx_to_use > 0:
                warn(f'{prefix}setting lowest {first_idx_to_use} steps (negative) to 0')

            param_steps = param_steps[first_idx_to_use:]

            if not drop_zero:
                # TODO actually guaranteed step sizes are uneven now? maybe we could
                # choose values where that's not true? add to unit test? (and then maybe
                # check explicitly that the step sizes are uneven here?)
                warn(f'{prefix}step sizes uneven after setting negative values to 0!')

    # TODO actually check we have something less than center if we care about that.
    # otherwise delete. currently can get past this if i just have enough steps for the
    # stuff >= center to satisfy this
    #
    #assert len(param_steps) >= 3, (f'{prefix}must have at least 1 step on '
    #    'either side of center'
    #)

    assert sum(np.allclose(row, center) for row in param_steps) == 1, \
        f'{prefix}center not in steps'

    if drop_negative:
        if drop_zero:
            assert (param_steps > 0).all()
        else:
            assert (param_steps >= 0).all()

    return param_steps


# TODO factor to share this type w/ somewhere else?
# TODO and maybe share the float | list-of-float part w/ fit_and_plot... kwargs, then
# union further w/ series if keeping that here?
def format_weights(weights: Optional[Union[float, pd.Series, List[float]]], name: str
    ) -> str:
    # TODO exclude the ', ' prefix from output?

    if weights is None:
        return ''

    # TODO share this w/ format_model_params + elsewhere?
    float_fmt = '.2f'

    # TODO share some code w/ fn to put some weight info in titles ("debug suffix")?

    if isinstance(weights, float):
        weight = weights

    elif isinstance(weights, pd.Series):
        # TODO delete (only had access to this when defined inside
        # format_model_params)
        # TODO could i have also asserted
        # `one_row_per_claw and use_connectome_APL_weights==False`?
        # (i might have also not preserved logic exactly when reorganizing conditionaal
        # into what is now that commented assert...)
        #assert model_kws.get('one_row_per_claw')
        #
        weight = weights.mean()
    else:
        # TODO delete (only had access to this when defined inside
        # format_model_params)
        #assert n_seeds > 1
        #
        assert isinstance(weights, list)
        assert all(isinstance(x, float) for x in weights)
        weight = np.mean(weights)

    # TODO delete (only had access to this when defined inside format_model_params)
    #if isinstance(weights, (float, pd.Series)):
    #    assert n_seeds == 1
    #
    param_str = f', {name}={weight:{float_fmt}}'
    return param_str


def filter_params_for_csv(params: ParamDict) -> ParamDict:
    # TODO update doc pickle->parquet where appropriate, when making that change for
    # caching pandas objects
    """Returns dict like input, but without values that would screw up CSV formatting

    Excludes arrays / Series / etc, which should each be saved at least to a separate
    pickle in `fit_and_plot_mb_model` (but each currently saved manually, so would have
    to do same for any new outputs with one of these types, or change handling).

    Python lists of simple types (those that themselves would have `include_in_csv`
    return True) are not filtered.
    """
    def include_in_csv(x):
        # (for test_fitandplot_repro[pn2kc_uniform__n-claws_7__n-seeds_2])
        if isinstance(x, list):
            x0 = x[0]
            assert not isinstance(x0, list)
            # TODO assert all elements would also pass? currently just assuming if first
            # element does, they all would
            #
            # to handle lists appended across seeds, in n_seeds>1 case.
            # should filter out kc_spont_in list-of-Series there.
            return include_in_csv(x0)

        # could also explicitly include types like
        # from typing import get_args
        # instance(x, get_args(Union[np.ndarray, pd.Series])) (+ xarray whatever)
        # ...but then I'd have to specify too many types to cover all cases I might want
        if hasattr(x, 'shape'):
            # to include data like arr[0] (wherer arr is 1D, arr[0] is a scalar, but
            # this scalar will both 1) have .shape [== ()] and 2) no longer be
            # isinstance(<scalar>, np.ndarray
            return len(x.shape) == 0

        return True

    return {k: v for k, v in params.items() if include_in_csv(v)}


# TODO replace (/combine w/) util.format_params?
def normalize_param_str(param_str: str) -> str:
    """Processes `format_model_params` output to make more friendly for code + filenames
    """
    return param_str.strip(', ').replace('_','-').replace(', ','__').replace('=','_')


# in fit_and_plot_mb_model, excluded from param_str and params_for_csv.
# also used to exclude params from test IDs in paramterized tests.
exclude_params = (
    'orn_deltas',
    'title',
    'repro_preprint_s1d',
    'make_plots',
    'plot_example_dynamics',
    'return_dynamics',
    '_multiresponder_mask',
    '_wPNKC',
    '_wAPLKC',
    '_wKCAPL',
    # should only currently show up in one_row_per_claw case. i might be able
    # to remove need for it, and should have enough info to distinguish output
    # dir/plots w/ just wAPLKC anyway.
    'wKCAPL',
    # TODO i assume i'll want to remove this one too?
    'wPNAPL',
)
# TODO also skip stuff equal to default in fit_mb_model (so that if we explicitly
# specify it for one test later, still matches old repro outputs, at least assuming
# defaults didn't change...)?
def format_model_params(model_kws: ParamDict, *, human: bool = False) -> str:
    """Returns str unique within most important model parameters.

    By default `exclude_params` and (internal) `exclude_if_value_matches` are used to
    exclude certain (generally less relevant, or less compressible to a string)
    parameters from the output.

    Returns 'default' for an empty dict (or input with all params excluded), to always
    return a non-empty string.

    Args:
        model_kws: arguments to `fit_mb_model`, or also `n_seeds` from
            `fit_and_plot_mb_model`. If using `model_kws` defined inside
            `fit_and_plot_mb_model` as input, should also add the variables:
              - 'fixed_thr'
              - 'wAPLKC'
              - 'n_seeds'

            ...and their values (that function will manually pass the first two through
            to `fit_mb_model` as well, and 'n_seeds' is a relevant parameter specific to
            `fit_and_plot_mb_model`).

        human: if False, `normalize_param_str` will be called on output before return.
            If True, will be more suitable for human readable titles/labels/etc.
    """
    model_kws = dict(model_kws)

    fixed_thr = model_kws.pop('fixed_thr', None)
    wAPLKC = model_kws.pop('wAPLKC', None)
    wAPLPN = model_kws.pop('wAPLPN', None)
    # TODO TODO also support wPNAPL and wKCAPL (or *_scale) if present? (may end up
    # using to control each separately? or may use new param for that)
    n_seeds = model_kws.pop('n_seeds', 1)

    # responses_to handled below, circa def of param_dir
    param_abbrevs = {
        'tune_on_hallem': 'hallem-tune',
        'pn2kc_connections': 'pn2kc',
        # TODO TODO why is target_sp (for cases where it's default of 0.1) still in
        # model output dir names? making kiwi/control modeling dir names too long (did i
        # ever remove it? either way, do so now)
        'target_sparsity': 'target_sp',
        'target_sparsity_factor_pre_APL': 'sp_factor_pre_APL',
        '_drop_glom_with_plus': 'drop-plusgloms',
        'use_connectome_APL_weights': 'connectome-APL',
    }

    # TODO just use default values (from fit_mb_model def, via introspection? how else?)
    # for these instead?
    #
    # will be excluded from param_str (and thus created param_dir names) if values match
    # those in this dict. to remove clutter in these names. keys should be full param
    # names, not the abbreviated values in param_abbrevs.
    exclude_if_value_matches = {
        'tune_on_hallem': False,
    }

    # TODO sort params first? (so changing order in code doesn't cause
    # cache miss...)
    # TODO why adding [''] again? why not just prepend ', ' to output if i want?
    # TODO TODO limit precision of all floats by default?
    param_str = ', '.join([''] + [
        f'{param_abbrevs[k] if k in param_abbrevs else k}={v}'
        for k, v in model_kws.items() if k not in exclude_params and
        (k not in exclude_if_value_matches or v != exclude_if_value_matches[k])
    ])

    if n_seeds > 1:
        param_str += f', n_seeds={n_seeds}'

    if fixed_thr is not None or wAPLKC is not None:
        assert fixed_thr is not None and wAPLKC is not None

        # TODO maybe only do if _in_sens_analysis. don't think i actually want in
        # hemibrain case (other than within sens analysis subcalls)
        #
        # TODO need to support int type too (in both of the two isinstance calls below)?
        # isinstance(<int>, float) is False
        if n_seeds == 1:
            if isinstance(fixed_thr, float):
                # TODO check that variable_n_claws case (w/ n_seeds=1) ends up here, and
                # not with a list-of-float (or change logic of conditional to handle
                # n_seeds=1 case below too)
                fixed_thr_str = f'fixed_thr={fixed_thr:.0f}'

            # TODO or format type2thr if that is passed? (and if it is, should be
            # used to define vector fixed_thr. initial fixed_thr should be None if in
            # that case)
            # TODO is it really an array sometimes and not Series? fix that?
            elif isinstance(fixed_thr, np.ndarray):
                # TODO or just format type2thr dict? was just trying to still be useful
                # for sensitivity analysis dir names, without making them too long
                fixed_thr_str = f'mean_thr={fixed_thr.mean():.0f}'

            # TODO at least do something for vector/dict fixed_thr (assuming i continue
            # to need/support both)? this only used for dir names, right?
            else:
                raise ValueError('unexpected fixed_thr type: {type(fixed_thr)}')

            param_str += f', {fixed_thr_str}'
        else:
            variable_n_claws = model_kws['pn2kc_connections'] in variable_n_claw_options
            assert variable_n_claws

            # fixed_thr and APL weights should all be list-of-float here.
            # I don't believe connectome APL weights are supported in any of these cases
            # (in which case it would be list of Series).
            #
            # TODO refactor
            assert isinstance(fixed_thr, list)
            assert all(isinstance(x, float) for x in fixed_thr)

            # TODO refactor
            fixed_thr_str = f'mean_thr={np.mean(fixed_thr):.0f}'
            param_str += f', {fixed_thr_str}'

    param_str += format_weights(wAPLKC, 'wAPLKC')
    param_str += format_weights(wAPLPN, 'wAPLPN')

    # simplifies some things to always have a non-empty string output (creation of
    # directories, etc)
    if len(param_str) == 0:
        param_str = 'default'

    if not human:
        param_str = normalize_param_str(param_str)

    return param_str


# TODO use parquet instead (more portable?) something else (other than CSV)?
# (for most pickles, except params maybe. something else for that? json? leave as
# pickle?)
model_responses_cache_name = 'responses.p'
model_spikecounts_cache_name = 'spike_counts.p'

# TODO refactor to use these in test code + natmix
# (or to use fns that load these w/ dir as input, not needing to use name
# separately)
#
# TODO rename this to have "fit"/"tuned" in name or something (and change
# model_mb_responses `tuned` var/outputs to not), since this is all (true?) tuned, and
# latter is a mix (stuff from this, but also stuff hardcoded from above / output
# statistics)
# TODO TODO rename to be less confusing. the variable called params_for_csv is not saved
# to this. param_dict is saved to this.
param_cache_name = 'params_for_csv.p'
# TODO delete these, and just save to name of the variable?
wPNKC_cache_name = 'wPNKC.p'
extra_responses_cache_name = 'extra_responses.p'
extra_spikecounts_cache_name = 'extra_spikecounts.p'
#

# TODO also rename to include "fit"/"tuned" in name?
def read_param_cache(model_output_dir: Path) -> ParamDict:
    # TODO doc how contents expected to differ from that of params.csv in same dir
    # (across code paths)
    """Returns contents of `param_cache_name` from input directory.
    """
    # TODO if update to json, will still need to fallback to pickle for hemibrain paper
    # repro (+ uniform paper repro test), and maybe some others
    return read_pickle(model_output_dir / param_cache_name)


def save_and_remove_from_param_dict(param_dict: ParamDict, param_dir: Path, *,
    save_dynamics: bool = True, keys_not_to_remove: Iterable[str] = tuple()) -> None:
    """Removes keys from param_dict, saving most corresponding values to single files.

    Modifies input inplace.
    """
    # NOTE: might break some of new vector-fixed_thr code if i start popping that
    # (but might want to still save separately there, as not being saved correctly
    # in params.csv/similar there)
    for k in list(param_dict.keys()):
        v = param_dict[k]
        # TODO also do for np.ndarray / Series / DataFrame?
        # TODO TODO especially if trying to get all serialized formats somewhat more
        # portable (and/or backwards compatible). may want all pandas objects as
        # parquet, all xarray either netcdf (or OK being not portable), and
        # everything else [mainly, parameter outputs, including the pickle, to the
        # extent that the CSV isn't enough] either using as simple types as possible
        # w/ pickle (to improve chances for portability / etc) (or json/similar for
        # params currently in pickle, but still that probably can't do
        # numpy/pandas/etc)
        pickle_path = param_dir / f'{k}.p'
        if isinstance(v, xr.DataArray):
            if save_dynamics:
                # TODO TODO move saving of all/most of these into fit_mb_model, so
                # each can be removed from memory as soon as possible? (and each
                # before making a copy of other variables, only processing [and
                # plotting if needed] one at at time). many of these DataArray are
                # very large, especially claw_sims, which can be ~2.5GiB for just 6
                # odors
                #
                # for just panel=control, the biggest of these files
                # (spike_recordings.p / vm_sims.p) are just under a GB (768M)
                # (this was before claw_sims, which is probably bigger...)
                to_pickle(v, pickle_path, verbose=True)

                # TODO switch to netcdf? would just need to reformat the index
                # surrounding IO (prob don't wanna do both given the sizes, but
                # could. ig it's just a factor of 2...)
                # (and may NOT want to save as both, considering the aggregate size)

            # TODO add flag to still enable saving here?
            else:
                warn('not saving dynamics for sensitivity analysis sub-calls,'
                    ' to save on storage space'
                )

            del param_dict[k]

        elif isinstance(v, (pd.DataFrame, pd.Series)):
            # TODO TODO TODO replace w/ to_parquet (after verifying current internal
            # check in to_pickle, which is calling to_parquet, passes in all tests?)
            # TODO + CSV (alongside parquet) just for posterity, and a backup option
            # for people who may not want to use parquet?
            #
            # expecting thr and/or kc_spont_in may be an issue here, in some cases?
            to_pickle(v, pickle_path, verbose=True)

            # TODO delete hack eventually
            if k in keys_not_to_remove:
                warn(f'not removing non-json serializable value for {k=} from '
                    'param_dict! (hack to preserve way to get APL weights in '
                    'fit_and_plot_mb_model output, in this specific case)'
                )
                continue
            #

            del param_dict[k]

        # TODO delete
        # TODO TODO kc_spont_in and/or thr are in at least some cases, right? maybe
        # just thr (and maybe only currently in unused vector thr test code?)?
        # already dealt with somewhere?
        # TODO TODO yup, test_equalize_kc_type_sparsity can trigger this (anything else?
        # try vector_thr again?) still just fix so both deal w/ series instead of
        # arrays?
        elif isinstance(v, np.ndarray):
            print()
            print(f'{k=}')
            print(f'{type(k)=}')
            print(f'{v=}')
            print('was a numpy array! pop + save separately?')
            breakpoint()
            # TODO raise ValueError or something?
        #

        else:
            # try dumping json for each object, because presumably a dict of those
            # will also be json serializable (so we can write them all in one call
            # later, if we aren't popping + handling separately above, rather than
            # needing pickle or something less portable).
            # presumably these will just be nested groups of simple python objects?
            # this is probably the easiest way to check that too?
            # https://stackoverflow.com/questions/42033142
            try:
                # TODO TODO now actually save rest of param_dict to one pickle later
                # (+ replace pickle w/ what)
                json.dumps(v)
            except (TypeError, OverflowError) as err:
                # TODO delete
                print(f'{k=}')
                print(f'{type(v)=}')
                print(f'{v=}')
                print(f'{err=}')
                print('was not json serializable!')
                breakpoint()
                #

                # TODO raise diff err?
                raise


_fit_and_plot_seen_param_dirs = set()
# TODO why is sim_odors an explicit kwarg? just to not have included in strs describing
# model params? i already special case orn_deltas to exclude it. why not do something
# like that (if i keep the param at all)?
# TODO try to get [h|v]lines between components, 2-component mixes, and 5-component
# mix for new kiwi/control data (at least for responses and correlation matrices)
# (could just check for '+' character, to handle all cases)
# TODO add flag to toggle showing all KC subtype mean response rates in all titles?
def fit_and_plot_mb_model(plot_dir: Path, sensitivity_analysis: bool = False,
    sens_analysis_kws: Optional[ParamDict] = None, try_cache: bool = True,
    # TODO rename comparison_responses to indicate it's only used for sensitivity
    # analysis stuff? (and to be more clear how it differs from comparison_[kcs|orns])
    comparison_responses: Optional[pd.DataFrame] = None,
    # TODO default to n_seeds=None and then in code set it 1, warning to explicitly set
    # it (prob)? or default to N_SEEDS (don't want to accidentally make slow calls we
    # don't reallyy want tho...)?
    n_seeds: int = 1, restrict_sparsity: bool = False,
    min_sparsity: float = 0.03, max_sparsity: float = 0.25,
    _in_sens_analysis: bool = False,
    # TODO just use model_kws for fixed_thr/wAPLKC?
    # (may now make sense, if i'm gonna add a flag to indicate whether we are in a
    # sensitivity analysis subcall)
    fixed_thr: Optional[Union[float, np.ndarray]] = None,
    # TODO TODO would also have to pop any wKCAPL/wPNAPL floats/lists  we happen to get,
    # if any. test those cases? having them in exclude_params sufficient for that?
    # TODO check we can actually input list of floats (presumably to repro
    # variable_n_claws cases w/ multiple seeds)
    wAPLKC: Optional[Union[float, List[float]]] = None,
    wAPLPN: Optional[Union[float, List[float]]] = None,
    drop_silent_cells_before_analyses: bool = drop_silent_model_kcs,
    _add_combined_plot_legend=False, sim_odors=None, comparison_orns=None,
    comparison_kc_corrs=None, _strip_concs_comparison_kc_corrs=False,
    param_dir_prefix: str = '', title_prefix: str= '',
    extra_params: Optional[dict] = None, _only_return_params: bool = False, **model_kws
    ) -> Optional[ParamDict]:
    # TODO doc which extra plots made by each of comparison* inputs (or which plots are
    # changed, if no new ones)
    """
    Args:
        plot_dir: parent to directory that will be created to contain model outputs and
            plots. created model output directories will have names incuding key
            parameters.

        sensitivity_analysis: if True, will run multiple versions of the model
            (saving output for each to a separte subdirectory), with `fixed_thr` and
            `wAPLKC` stepped around tuned values from primary model outputs.
            use `sens_analysis_kws` to control steps.

        try_cache: set False to force* any model cache to be ignored. Calls via
            `al_analysis.py` CLI with `-i model` will have the same effect.

        min_sparsity: (internal use only) only used for models parameterized with fixed
            `fixed_thr` and `wAPLKC` (typically in context of sensitivity analysis).
            return before generating plots if output sparsity is outside these bounds.

        max_sparsity: (internal use only) see min_sparsity.

        extra_params: saved alongside internal params in cache pickle/CSV
            (for keeping tracking of important external parameters, for reproducibility)

        **model_kws: passed to `fit_mb_model` (see its docstring, particularly for key
            inputs, such as `orn_deltas`)

    Returns dict with key model parameters (some tuned), any input `extra_params`, as
    well as 'output_dir' with name of created directory (which contains model outputs
    and plots).

    Notes:
    The contents of the `orn_deltas.csv` files copied to each model output directory
    should reflect the `model_kws['orn_deltas']` input to this function.
    """
    assert n_seeds >= 1

    # TODO delete restrict_sparsity? currently used?
    # TODO delete [min|max]_sparsity too?
    if not restrict_sparsity:
        min_sparsity = 0
        max_sparsity = 1

    # TODO delete. isn't responses_to just overwritten w/ 'pebbled' below
    # (before being used, right?)
    # TODO use one of newer strs in al_analysis.py for this (-> move to al_util?)?
    # might this ever be 'Z-scored F' instead of dF/F?
    my_data = f'pebbled {dff_latex}'

    if 'orn_deltas' in model_kws:
        # TODO fix how this might misrepresent stuff if i pass hallem data in manually?
        # (allow passing in description of the data, overriding this?)
        responses_to = my_data
    else:
        responses_to = 'hallem'
    #

    # TODO also try tuning on remy's subset of hallem odors?
    # (did i already? is it clear that what's in preprint was not done this way?)

    # TODO share default w/ fit_mb_model somehow?
    tune_on_hallem = model_kws.get('tune_on_hallem', False)
    if tune_on_hallem:
        tune_from = 'hallem'
    else:
        tune_from = my_data
    del my_data

    # TODO share defaults w/ fit_mb_model somehow?
    pn2kc_connections = model_kws.get('pn2kc_connections', 'hemibrain')
    variable_n_claws = pn2kc_connections in variable_n_claw_options
    use_connectome_APL_weights = model_kws.get('use_connectome_APL_weights', False)
    one_row_per_claw = model_kws.get('one_row_per_claw', False)

    param_str = format_model_params({**model_kws, **{
            'fixed_thr': fixed_thr, 'wAPLKC': wAPLKC, 'wAPLPN': wAPLPN,
            'n_seeds': n_seeds
        }}, human=True
    )

    # TODO clean up / refactor. hack to make filename not atrocious when these are
    # 'pebbled_\$\\Delta_F_F\$'
    if responses_to.startswith('pebbled'):
        responses_to = 'pebbled'

    if tune_from.startswith('pebbled'):
        tune_from = 'pebbled'

    # this way it will also be included in params_for_csv, and we won't need to manually
    # pass to all fit_mb_model calls
    model_kws['drop_silent_cells_before_analyses'] = drop_silent_cells_before_analyses

    # TODO refactor so param_str defined from this (prob not, now that i have
    # format_model_parmams... maybe include these other things in call to that? still
    # want it to be consistent when called from test code tho...), and then f-str below
    # (+ for_dirname def) doesn't separately specify {responses_to=}?
    params_for_csv = {
        'responses_to': responses_to,
        'tune_from': tune_from,
    }
    params_for_csv.update(
        {k: v for k, v in model_kws.items() if k not in exclude_params}
    )

    # prefix defaults to empty str
    title = title_prefix
    del title_prefix

    if _in_sens_analysis:
        # TODO also assert wAPLKC is a (scalar) float (and not a vector float array from
        # use_connectome_APL_weight=True code)?
        assert fixed_thr is not None and wAPLKC is not None

        # TODO if `prat_boutons and not per_claw_pn_apl_weights`, assert wAPLPN is not
        # None?

        # TODO what plot_dir_prefix? update comment below
        #
        # assumed to be passed in (but not created by) sensitivity analysis calls
        # (recursive calls below)
        #
        # the parent directory of this should have plot_dir_prefix in it, and don't feel
        # the need to also include here.
        param_dir = plot_dir

        # TODO refactor to share more of this w/ format_model_params (/delete)
        #
        # TODO delete? should always be redefed below...
        # (if so, then why is this code even here?)
        # TODO refactor this thr str handling?
        if isinstance(fixed_thr, float):
            title += f'thr={fixed_thr:.2f}'
        else:
            assert isinstance(fixed_thr, np.ndarray)
            title += f'mean_thr={fixed_thr.mean():.2f}, wAPLKC={wAPLKC:.2f}'

        title += format_weights(wAPLKC, 'wAPLKC')
        if wAPLPN is not None:
            title += format_weights(wAPLPN, 'wAPLPN')
        #

        # TODO TODO replace w/ title defined after fit_mb_model call (so we can add
        # information based on those outputs to title, like mean weights)?
        title_including_silent_cells = title
    else:
        # TODO one hardcode flag to control whether these are dropped or not?
        # TODO TODO start excluding parts from dirname unless they are different from
        # some default? parts to consider excluding for default values (hardly change):
        # - dff_scale-to-avg-max
        # - data_pebbled
        # - hallem-tune_False
        # - target-sp_0.0915

        # TODO refactor for_dirname handling to not specialcase responses_to/others?
        # possible to have simple code not split by fixed_thr/wAPLKC None or not?
        # TODO need to pass thru util.to_filename / simliar normalization myself now
        # (since i'm putting this in a dirname now, not the final filename of the plot)
        for_dirname = ''
        # NOTE: responses_to set to 'pebbled' above if orn_deltas is passed, so could be
        # confusing if hallem data passed in as orn_deltas
        if responses_to != 'pebbled':
            for_dirname = f'data_{responses_to}'

        if len(param_str) > 0:
            if len(for_dirname) > 0:
                for_dirname += '__'

            for_dirname += normalize_param_str(param_str)

        # TODO rename plot_dir + this to be more clear?
        # plot_dir contains all modelling (mb_modeling)
        # param_dir contains outputs from model run w/ specific choice of params
        # (and only contains stuff downstream of dF/F -> spiking model
        # creation/application)
        param_dir_name = f'{param_dir_prefix}{for_dirname}'

        param_dir = plot_dir / param_dir_name

        if fixed_thr is not None or wAPLKC is not None:
            assert fixed_thr is not None and wAPLKC is not None
            # TODO assert target_sparsity_factor_pre_APL is None?

            # in n_seeds > 1 case, fixed_thr/wAPLKC will be lists of floats, and will be
            # too cumbersome to format into this
            #
            # TODO need to support int type too? isinstance(<int>, float) is False
            if n_seeds == 1:
                if isinstance(fixed_thr, float):
                    # TODO delete?
                    #title += f'fixed_thr={fixed_thr:.0f}, wAPLKC={wAPLKC:.2f}\n'
                    pass
                else:
                    assert isinstance(fixed_thr, np.ndarray)
                    # TODO delete?
                    #title += f'mean_thr={fixed_thr.mean():.0f}, wAPLKC={wAPLKC:.2f}\n'
        else:
            if 'target_sparsity' in model_kws:
                assert model_kws['target_sparsity'] is not None
                target_sparsity = model_kws['target_sparsity']
            else:
                target_sparsity = 0.1
                # TODO replace .3g w/ a format_sparsity fn? (doing .3g [or .2g?] if
                # there are 0s after decimal point, and maybe .2f otherwise? some way to
                # accomplish something like that within f-str syntax?)
                warn(f'using default target_sparsity of {target_sparsity:.3g}')

            # TODO include something else for case where fixed_thr is vector (which
            # currently requires wAPLKC set, to a float as before)?
            assert fixed_thr is None and wAPLKC is None and wAPLPN is None

            # .3g will show up to 3 sig figs (regardless of their position wrt decimal
            # point), but also strip any trailing 0s (0.0915 -> '0.0915', 0.1 -> '0.1')
            #title += f'target_sparsity: {target_sparsity:.3g}\n'

        # TODO delete old title code above (commented lines) if i like this. or maybe
        # combine strategies, if there are certain params i still want to special case
        # for titles?
        title += '\n'.join(for_dirname.split('__')).replace('_', ': ').replace('-', '_')
        del for_dirname

        # NOTE: this is for analyses that either always include or always drop silent
        # cells, regardless of value of `drop_silent_cells_before_analyses`
        # (e.g. should be used for analyses using `responses_including_silent_cells`)
        # TODO move this below too? (to where title is updated w/ silent cells suffix)
        # TODO TODO replace w/ title defined after fit_mb_model call (so we can add
        # information based on those outputs to title, like mean weights)?
        title_including_silent_cells = title

        # to save plots of internal ORN / PN matrices (and their correlations, etc),
        # exactly as used to run model
        model_kws['plot_dir'] = param_dir
        model_kws['title'] = title
    #

    params_for_csv['output_dir'] = param_dir.name

    # TODO also rename to include "fit"/"tuned" in name?
    param_dict_cache = param_dir / param_cache_name

    # TODO replace all these cache names (where possible) w/ just saving to name of
    # variable (+ '.p' / '.parquet' whatever)
    model_responses_cache = param_dir / model_responses_cache_name
    model_spikecounts_cache = param_dir / model_spikecounts_cache_name

    extra_responses_cache = param_dir / extra_responses_cache_name
    extra_responses = None

    extra_spikecounts_cache = param_dir / extra_spikecounts_cache_name
    extra_spikecounts = None

    use_cache = try_cache and (not should_ignore_existing('model')) and (
        # checking both since i had previously only been returning+saving the 1st
        model_responses_cache.exists() and model_spikecounts_cache.exists()
    )

    made_param_dir = False

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
            # TODO replace below w/ this? or too convoluted since still checking
            # param_dict_cache.exists() outside?
            #param_dict = read_param_cache(param_dir)
            param_dict = read_pickle(param_dict_cache)

            cached_APL_weights = get_APL_weights(param_dict, model_kws)
            cached_wAPLKC = cached_APL_weights['wAPLKC']

            # TODO TODO also handle wAPLPN?

            # np.array_equal works with both float and list-of-float inputs
            if (not np.array_equal(fixed_thr, param_dict['fixed_thr']) or
                not np.array_equal(wAPLKC, cached_wAPLKC)
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

    # TODO give better explanation as to why this is here.
    # to make sure we are accounting for all parameters we might vary in filename
    if param_dir in _fit_and_plot_seen_param_dirs:
        # otherwise, param_dir being in seen set would indicate an error
        assert _only_return_params, f'{param_dir=} already seen!'
        use_cache = True

    _fit_and_plot_seen_param_dirs.add(param_dir)

    # NOTE: this currently will cause -c/-C checks to fail
    # TODO TODO want to fix that (i.e. remove this from that output, but still keep long
    # enough to use for what i wanted? possible?)?
    params_for_csv['used_model_cache'] = use_cache

    make_plots = model_kws.pop('make_plots', True)

    print()
    # TODO TODO default to also skipping any plots made before returning? maybe add
    # another ignore-existing option ('model-plots'?) if i really want to be able to
    # remake plots w/o changing model outputs? takes a lot of time to make plots on all
    # the model outputs...
    if use_cache:
        print(f'loading model responses (+params) from cache {model_responses_cache}')
        # TODO why using my read_pickle wrapper for only some of these?
        responses = pd.read_pickle(model_responses_cache)
        spike_counts = pd.read_pickle(model_spikecounts_cache)
        param_dict = read_pickle(param_dict_cache)

        if extra_responses_cache.exists():
            extra_responses = pd.read_pickle(extra_responses_cache)

        if extra_spikecounts_cache.exists():
            extra_spikecounts = pd.read_pickle(extra_spikecounts_cache)
    else:
        # doesn't necessarily matter if it already existed. will be deleted if sparsity
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
            assert wAPLPN is None or type(wAPLPN) is list

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
            kc_spont_in_list = []

            _fixed_thr = None
            _wAPLKC = None

            # TODO do i really need these for tests? if so, can i define them from what
            # i already have available here (or from files i can load from here)?
            tuning_seeds = None
            tuning_wPNKC = None

            if fixed_thr is not None:
                # this branch should not run in any sensitivity analyis subcalls, as
                # currently only doing that for n_seeds=1 (i.e. hemibrain) case
                # (otherwise, we would need to test if tuning_output_dir is None / etc)
                assert not _in_sens_analysis

                assert len(fixed_thr) == len(wAPLKC) == n_seeds
                if wAPLPN is not None:
                    assert len(wAPLPN) == n_seeds

                # TODO delete?
                #assert tuning_output_dir is not None
                if tuning_output_dir is not None:
                    tuning_wPNKC_cache = tuning_output_dir / wPNKC_cache_name
                    assert tuning_wPNKC_cache.exists()

                    tuning_wPNKC = read_pickle(tuning_wPNKC_cache)
                    # assuming all entries of a given seed are at adjacent indices in
                    # the seed level values (should never be False given how i'm
                    # implementing things)
                    tuning_seeds = tuning_wPNKC.index.get_level_values('seed').unique()
                else:
                    # TODO or can i easily define from what i already have available
                    # here? (-> remove this warning)
                    warn('assuming tuning seeds and wPNKC are same as those for calls '
                        'below (with fixed fixed_thr/wAPLKC)'
                    )
                #

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
                    # TODO assume, at least for test_[fitandplot_]fixed_inh_params that
                    # seed sequences will be the same? (that's what i'm hoping i can do
                    # now)
                    #
                    # would need to use same seed sequence if this ever failed
                    if tuning_seeds is not None:
                        assert seed == tuning_seeds[i]

                responses, spike_counts, wPNKC, param_dict = fit_mb_model(
                    # TODO or can i handle fixed_thr/wAPLKC thru model_kws (prob not)?
                    # (maybe i will soon be able to, if i'm gonna replace some of their
                    # usage with a flag to indicate whether we are in a sensitivity
                    # analysis subcall...)
                    sim_odors=sim_odors, fixed_thr=_fixed_thr, wAPLKC=_wAPLKC,
                    wAPLPN=wAPLPN, seed=seed,
                    # ORN/PN plots would be redundant, and overwrite each other.
                    # currently those are the only plots I'm making in here (no longer
                    # true, but still probably don't want for each seed).
                    make_plots=make_plots if (i == 0) else False,
                    **model_kws
                )

                # TODO return as separate variable then, if we can't handle certain
                # types below w/o explicitly popping? or make handling of things like
                # this automatic, based on variable name (saved to appropriate file,
                # **with seed also handled correctly, for pandas stuff**, similar to
                # DataArray handling [but also handling seeds])
                kc_spont_in = param_dict.pop('kc_spont_in')

                if fixed_thr is not None:
                    if tuning_wPNKC is not None:
                        # could prob delete. should be sufficient to check the seeds
                        # equal, as we are doing above
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

                kc_spont_in = util.addlevel(kc_spont_in, 'seed', seed)

                if first_param_dict is None:
                    first_param_dict = param_dict
                else:
                    assert param_dict.keys() == first_param_dict.keys()

                responses_list.append(responses)
                spikecounts_list.append(spike_counts)
                wPNKC_list.append(wPNKC)
                kc_spont_in_list.append(kc_spont_in)
                param_dict_list.append(param_dict)

            responses = pd.concat(responses_list, verify_integrity=True)
            spike_counts = pd.concat(spikecounts_list, verify_integrity=True)
            wPNKC = pd.concat(wPNKC_list, verify_integrity=True)
            kc_spont_in = pd.concat(kc_spont_in_list, verify_integrity=True)

            param_dict = {
                k: [x[k] for x in param_dict_list] for k in first_param_dict.keys()
            }
            param_dict['kc_spont_in'] = kc_spont_in
        else:
            # TODO need to support int type too? (in both of the two isinstance calls
            # below) isinstance(<int>, float) is False
            # isinstance works w/ both float and scalar np.float64 (but not int)
            # TODO update fixed_thr check to include ndarray
            #assert fixed_thr is None or isinstance(fixed_thr, float)

            # NOTE: wKCAPL is set from wAPLKC, if only wAPLKC is passed. The reverse is
            # not true though, so if only one is passed, it must be wAPLKC.
            assert wAPLKC is None or (
                # NOTE: currently one-row-per-claw case has wAPLKC as a Series
                isinstance(wAPLKC, float) or isinstance(wAPLKC, pd.Series)
            )
            assert wAPLPN is None or (
                # NOTE: currently one-row-per-claw case has wAPLPN as a Series
                isinstance(wAPLPN, float) or isinstance(wAPLPN, pd.Series)
            )

            # TODO rename param_dict everywhere -> tuned_params?
            responses, spike_counts, wPNKC, param_dict = fit_mb_model(
                sim_odors=sim_odors, fixed_thr=fixed_thr, wAPLKC=wAPLKC, wAPLPN=wAPLPN,
                make_plots=make_plots, **model_kws
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

        # TODO TODO also pass in + save a copy of full input ORN data (same as in
        # ij_certain-roi_stats.[csv|p], maybe just load that in here, to not need to
        # pass? assuming mtime is since the start of run?). or just shutil copy
        # ij_certain-roi_stats.[csv+p]?
        if orn_deltas is not None:
            # TODO TODO use parquet instead (+ check can round trip w/o change)
            #
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
        # TODO TODO use parquet instead (+ check can round trip w/o change)
        to_pickle(responses, model_responses_cache)
        to_pickle(spike_counts, model_spikecounts_cache)

        wPNKC_cache = param_dir / wPNKC_cache_name
        to_pickle(wPNKC, wPNKC_cache)

        # currently just assuming that both will be in same format
        # (both either np.arrays/pd.Series [depending on which i end up settling on
        # for implementation] in only use_connectome_APL_weights=True case)
        wAPLKC = param_dict.get('wAPLKC', None)
        # TODO need to support int type too (in all isinstance calls below)?
        # isinstance(<int>, float) is False
        # TODO use get_APL_weights to replace some of below? (at least also call it, to
        # check it doesn't fail anywhere?)
        get_APL_weights(param_dict, model_kws)

        if wAPLKC is not None and not is_scalar(wAPLKC):
            if not variable_n_claws:
                # TODO delete?
                # Tianpei currently has vector APL weights divided by # claws for each
                # KC [in the C++ code], and he does not (seem to) use the
                # preset_w[APLKC|KCAPL]=true path for that (at least, not for the
                # use_connectome_APL_weights=False case, where he still needs diff
                # weights for diff KCs, b/c they have diff # of claws)

                wKCAPL = param_dict['wKCAPL']
                assert isinstance(wAPLKC, pd.Series)
                assert isinstance(wKCAPL, pd.Series)

                # TODO move some/all assertions into get_APL_weights? (/delete)
                assert hasattr(wAPLKC, 'shape') and len(wAPLKC.shape) == 1
                assert wKCAPL.shape == wAPLKC.shape
                if not one_row_per_claw:
                    assert 'wAPLKC_scale' in param_dict
                    assert 'wKCAPL_scale' in param_dict
                # TODO (delete? test this case, and whether i can discard?) seems it
                # *does* have wAPLKC_scale in use_connectome_APL_weights=True case (but
                # also wAPLKC, for prat_claws=False at least, right? need that too?).
                # can i just discard here tho (scale already applied)?
                #else:
                #    # NOTE: there will now be *neither* wAPLKC nor wAPLKC_scale in
                #    # param_dict (and same for wKCAPL)
                #    # TODO TODO should i not pop instead? (in general, but maybe
                #    # specifically b/c of this?)
                #    assert not 'wAPLKC_scale' in param_dict
                #    assert not 'wKCAPL_scale' in param_dict

                # NOTE: these two only used for vector values (specifically, just the
                # use_connectome_APL_weights=True case, which should always produce such
                # values, unlike any other current path). If they are a scalar
                # (including a whole vector with just one value repeated), these two
                # params will be saved as elements of usual param pickle+CSV outputs.
                #
                # TODO doc what happens in variable_n_claws cases (prob going to also
                # pop + pickle like in connectome APL weights case)
            else:
                assert not use_connectome_APL_weights

                assert 'wAPLKC_scale' not in param_dict
                assert 'wKCAPL_scale' not in param_dict

                # TODO or go back to handling list-of-floats same as array of floats
                # (like in `not variable_n_claws` case above) regardless of fact that
                # (at least in some circumstances), keeping lists of floats seemed to
                # allow serialization to CSV? prob not w/o some reason...
                # TODO (delete?) switch handling? does de-serialize as a string, which
                # needs eval'd...
                assert type(wAPLKC) is list and all(is_scalar(x) for x in wAPLKC)
                wKCAPL = param_dict['wKCAPL']
                assert type(wKCAPL) is list and all(is_scalar(x) for x in wKCAPL)
        else:
            # in this branch, wAPLKC/wKCAPL should be in a format (e.g. single
            # scalar floats) where they don't cause issues if saved as part of
            # param_dict, so we don't need to separately save to w[APLKC|KCAPL]_cache

            wKCAPL = param_dict['wKCAPL']
            # TODO need to support int type too (in both of the two isinstance calls
            # below)? isinstance(<int>, float) is False
            # TODO TODO probably remove conditional before the check (reverting to old
            # behavior), or only have conditional to narrow
            # one-row-per-claw=True/whatever case this currently applies to)
            wKCAPL_size_check_temp = np.asarray(wKCAPL)
            if wKCAPL_size_check_temp.size == 1:
                assert wKCAPL is None or isinstance(wKCAPL, float)

        if model_kws.get('prat_boutons'):
            # TODO work? need to support any non-vector cases?
            title += weight_debug_suffix(param_dict)
            # TODO return processed title suffix too (from fit_mb_model), and just pop +
            # use that, rather than recomputing out here?

        # TODO delete this hack eventually
        #
        # these two should get filtered from CSV, but still returned
        # TODO also want to filter from saving to param pickle?
        # why even not del-ing them here then? can't we just load from disk
        # if we really need them? or was it b/c get_APL_weights downstream
        # of this then didn't have any of the APL weights it needed (i think
        # so?)? (was easier to get test code working again this way...)
        # TODO manually store and pass thru to get_APL_weights (or wrap call
        # below, handling this case?)? not sure if that works w/ all the
        # other places i use it though...
        keys_not_to_remove = tuple()
        if one_row_per_claw and not use_connectome_APL_weights:
            keys_not_to_remove = ('wAPLKC', 'wKCAPL')
        #
        save_dynamics = not _in_sens_analysis

        # modifies param_dict, removing keys (and saving most of their values to single
        # files)
        save_and_remove_from_param_dict(param_dict, param_dir,
            save_dynamics=save_dynamics, keys_not_to_remove=keys_not_to_remove
        )

        # TODO update comment (/fix code). not always all scalars now, at least b/c
        # Series wAPLKC in some one-row-per-claw cases (unless i meant only in
        # variable_n_claw n_seeds=1 cases)
        #
        # In n_seeds=1 case, param_dict keys are:
        # 'fixed_thr', 'wAPLKC', and 'wKCAPL' (all w/ scalar values)
        # ...as well as several other values relevant for APL tuning.
        #
        # In use_connectome_APL_weights=True case:
        # 1. param_dict also contains ['wAPLKC_scale', 'wKCAPL_scale'] which are scalars
        #    that were used to scale wAPLKC/wKCAPL during APL tuning.
        #
        # 2. wAPLKC/wKCAPL are scaled connectome weight vectors now, which will be
        #    popped from params below (before params saved to CSV/pickle), and pickled
        #    separately.
        #
        # in n_seeds > 1 case, should be same keys, but list values (of length equal to
        # n_seeds)
        # TODO TODO also save as json, and eventually replace this w/ that
        # (should be guaranteed that all types saveable as json now, unless some other
        # test cases [vector thr? will prob stay unused...] hits currently unhandled
        # ndarray case in param_dict saving+popping. could just convert it to Series to
        # have it handled there.)
        # TODO test that values read back from json match this (may need isclose for
        # some floats?)
        to_pickle(param_dict, param_dict_cache)

        # TODO don't save in sensitivity analysis subcalls? as this should not change
        # across those (+ make a list of files to exclude and pass to automated saving
        # above, based on dtypes in param_dict, and exclude based on list)
        # TODO TODO make CSVs for all (or a list of) outputs saved in loop above?
        # (then delete here?)
        to_csv(wPNKC, param_dir / 'wPNKC.csv', verbose=(not _in_sens_analysis))

        # saving after all the other things, so that (if script run w/ -c) checks
        # against old/new outputs have an opportunity to trip and fail before this is
        # written
        if extra_responses is not None:
            assert extra_spikecounts is not None
            # TODO TODO use parquet instead (+ check can round trip w/o change)
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

    # param_dict should include 'fixed_thr', 'wAPLKC' and 'wKCAPL' parameters, as they
    # are at the end of the model run (either tuned or hardcoded-from-the-beginning).
    #
    # In use_connectome_APL_weights=True case, wAPLKC/wKCAPL were vector, and have
    # already been popped from param_dict and pickled above. w[APLKC|KCAPL]_scale are
    # scalars in that case (used to multiply by unit-mean vector wAPLKC/wKCAPL), which
    # can be used/interpreted similarly to scalar wAPLKC/wKCAPL we get from other cases.
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
    # TODO check no duplicates, so that it's not just one of the megamat odors repeated
    # 17 times?
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

    # TODO do we not want to also filter what we are returning?
    # TODO delete filter_params_for_csv if we no longer have anything to remove?
    # (by virtue of more consistently popping stuff in param_dict. currently only one
    # exception in there, to not pop Series wAPLKC/wKCAPL for one particular case)
    filtered_params = filter_params_for_csv(params_for_csv)
    s1 = set(params_for_csv.keys())
    s2 = set(param_dict.keys())
    assert s2 - s1 == set()
    # so the CSV actually has as lot more stuff:
    # - responses_to
    # - tune_from
    # - model_kws (except keys in exclude_params)
    # - output_dir name
    # - used_model_cache
    # - sparsity
    # - megamat_sparsity (if we have all 17 megamat odors)
    #
    # ipdb> s1 - s2
    # {'responses_to', 'weight_divisor', 'tune_from', 'sparsity', 'hardcode_initial_sp',
    # 'used_model_cache', 'megamat_sparsity', 'drop_silent_cells_before_analyses',
    # 'output_dir'}
    #
    # NOTE: 'pearson' is also added to params_for_csv below, and i think that's the last
    # thing.
    param_series = pd.Series(filtered_params)
    try:
        # just to manually inspect all relevant parameters for outputs in a given
        # param_dir
        to_csv(param_series, param_dir / 'params.csv', header=False,
            # NOTE: only ignoring b/c adding the use_cache=True/False component of these
            # is now causing -c/-C checks to fail when I otherwise don't really want
            # them too.
            # TODO also build in a way to only ignore the changes if the change matches
            # what we expect (i.e. just a change in this use_cache=True/False line)?
            # TODO maybe replace w/ just saving the use_cache value to its own file?
            ignore_output_change_check='warn',
            #
            verbose=(not _in_sens_analysis)
        )

    # TODO change code to avoid this happening in the first place?
    # (should only happen on second call used for getting inh params on a panel set, to
    # then run a model with a single panel and those inh params later)
    except MultipleSavesPerRunException:
        # TODO delete? not sure i want to support this case anyway
        if _only_return_params:
            return params_for_csv
        #
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

    # TODO fix (when called from model_banana_iaa_concs.py). need a panel level?
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

    # NOTE: 'pearson' is added after this
    # TODO if i want it, will need to return later (and if i want in CSV in this
    # directory, will need to save that [currently above] after as well)
    if _only_return_params or not make_plots:
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
    def plot_sparsity_per_odor(sparsity_per_odor, comparison_sparsity_per_odor, suffix,
        *, ylim=None) -> Tuple[Figure, Axes]:

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

        # TODO assert no NaN (closer to start of fn? why not def this up there?)?
        response_rates = sparsity_per_odor[sparsity_col]

        if ylim is not None:
            ymin, ymax = ylim
            assert (response_rates <= ymax).all(), f'{response_rates.max()=}'
        else:
            ymin = 0
            ymax = response_rates.max()

        assert (response_rates >= ymin).all(), f'{response_rates.min()=}'

        if comparison_sparsity_per_odor is not None:
            comparison_response_rates = comparison_sparsity_per_odor[sparsity_col]

            if ylim is None:
                ymax = max(ymax, comparison_response_rates.max())
            else:
                assert (comparison_response_rates >= ymin).all(), \
                    f'{comparison_response_rates.min()=}'

                # TODO just warn and increase max in cases where this would fail
                assert (comparison_response_rates <= ymax).all(), \
                    f'{comparison_response_rates.max()=}'

        ax.set_ylim([ymin, ymax])

        savefig(fig, param_dir, f'sparsity_per_odor{suffix}')
        return fig, ax


    repro_preprint_s1d = model_kws.get('repro_preprint_s1d', False)

    eb_mask = responses.columns.get_level_values(odor_col).str.startswith('eb @')
    try:
        assert eb_mask.sum() <= 1
    # TODO TODO fix. sum is 2, but why? or why isn't it always in
    # repro_preprint_s1d=True case?
    #
    # ipdb> responses.shape
    # (1630, 110)
    # ipdb> responses.columns.get_level_values(odor_col)[eb_mask]
    # Index(['eb @ -2', 'eb @ -2'], dtype='object', name='odor1')
    except AssertionError:
        import ipdb; ipdb.set_trace()

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
        # TODO in n_seeds > 1 case, use old suffix w/o saying total numbers of cells?
        # (or some other format?)
        title += _get_silent_cell_suffix(responses_including_silent, responses)

    # parameters relevant for tuning:
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
        # TODO move to similar section of fit_mb_model? (near check of output vs target
        # sparsity)
        target_sparsity = model_kws['target_sparsity']
        print()
        print(f'target_sparsity={target_sparsity:.3g}')
        print(f'sparsity={sparsity:.3g}')

        adiff = sparsity - target_sparsity
        rdiff = adiff / target_sparsity
        print(f'{(rdiff * 100):.1f}% rel sparsity diff')
        print(f'{adiff:.2g} abs sparsity diff')
        #

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

        # TODO rename to 'odor_a', 'odor_b'? (here and in *corr_triangular?)? assuming
        # 'odor2' here isn't the for-mixtures 'odor2' i often have in odor
        # multiindices...
        odor_levels = ['odor1', 'odor2']
        mean_pearson_ser = seed_corr_ser.groupby(level=odor_levels, sort=False).mean()

        # TODO below 2 comments still an issue?
        # TODO TODO fix! (only in hallem/uniform, after no longer only passing
        # megamat odors as sim_odors)
        # TODO TODO check at input of this what is reducing length of
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

    pearson = _resort_corr(pearson, panel,
        warn=False if responses_to == 'hallem' else True
    )

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

        # TODO TODO fix how this now trips from scripts/model_banana_iaa_concs.py
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

            raise
            # (not currently an issue since i added the hack to move conc info to name
            # part for those odors)
            # TODO fix for new kiwi vs control data
            # (seems to be caused by dilutions of mixture. just drop those first? could
            # call the natmix fn for that)
            #import ipdb; ipdb.set_trace()

        assert len(set(df.columns)) == len(df.columns)
        return df

    try:
        pearson = _strip_index_and_col_concs(pearson)
    except AssertionError:
        warn('_strip_index_and_col_concs failed with AssertionError! probably have '
            'multiple concs for some odors. skipping rest of plots.'
        )
        return params_for_csv

    if _in_sens_analysis:
        assert fixed_thr is not None and wAPLKC is not None

        if use_connectome_APL_weights:
            wAPLKC = params_for_csv['wAPLKC_scale']
            assert wAPLKC is not None

            wKCAPL = params_for_csv['wKCAPL_scale']
            assert wKCAPL is not None

        # TODO TODO do anything w/ PN<>APL weights here, in relevant cases?

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
            # TODO refactor all this thr str handling? duplicated a fair bit now...
            if isinstance(fixed_thr, float):
                thr_str = f'thr={fixed_thr:.2f}'
            else:
                assert isinstance(fixed_thr, np.ndarray)
                thr_str = f'mean_thr={fixed_thr.mean():.2f}'

            title = (
                f'{thr_str}{format_weight(wAPLKC, "wAPLKC")} (sparsity={sparsity:.3g})'
            )
        else:
            title = f'sparsity={sparsity:.3g}'

    plot_corr(pearson, param_dir, 'corr', xlabel=title)

    if responses_to == 'hallem':
        # TODO factor to use len(megamat_odor_names) / something instead of 17...
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
        # maybe adding an assertion columns.name == glomerulus_col if not? might make it
        # nicer to refactor into plot_corr (for deciding whether to call
        # mean_of_fly_corrs)
        if comparison_orns.columns.name == glomerulus_col:
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

        def _save_kc_vs_orn_corr_scatterplot(metric_name: str) -> None:
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
                ax=ax, data=df, x=orn_col, y=model_col, linestyle='None',
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
            panel, warn=False if responses_to == 'hallem' else True
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

    row_colors = None
    if KC_TYPE in to_cluster.index.names:
        response_rate_by_type = to_cluster.groupby(KC_TYPE).apply(
            lambda x: x.mean().mean()
        )
        fig, ax = plt.subplots()
        # TODO drop unknown?
        # TODO use colors consistent w/ elsewhere?
        # TODO fixed scale?
        sns.barplot(response_rate_by_type, ax=ax)
        savefig(fig, param_dir, 'response_rate_by_type')

        # TODO need to do anything to get palette consistent w/ histograms?
        # (seems so, if i care [still?]). all types that should only be in one dataset
        # or another ['unknown'/'incomplete'/'gap'] should all be at end of hue order
        # now... still want to drop currently-missing types, like connectome APL fn
        # does?
        kc_type_palette = sns.color_palette(n_colors=len(kc_type_hue_order))

        type2color = dict(zip(kc_type_hue_order, kc_type_palette))

        row_colors = to_cluster.index.get_level_values(KC_TYPE).to_series(
            ).map(type2color)

        # seems this might be necessary
        row_colors.index = to_cluster.index

        # TODO can i add labels here (as opposed to separtely passing dict in legend
        # call below)?
        handles = [Patch(facecolor=color) for type, color in type2color.items()]

    if (~ silent_cells).sum() == 1:
        # cluster_rois below would fail (in the sns.clustermap call) with:
        # ValueError: The number of observations cannot be determined on an empty
        # distance matrix.
        # ...if we did not return here.
        warn('only one model cell had any responses! returning before generating '
            'further plots!'
        )
        return params_for_csv

    # ~30" height worked for ~1837 cells, but don't need all that when not plotting
    # silent cells
    cg = cluster_rois(to_cluster[~ silent_cells].T, odor_sort=False,
        figsize=(7, 12), row_colors=row_colors
    )

    if KC_TYPE in to_cluster.index.names:
        # TODO include counts in parens by default?
        # TODO factor all this legend creation into cluster_rois (-> also use for
        # fly_colors stuff)?
        cg.fig.legend(handles, type2color, title='KC subtype', loc='center right')

    cg.fig.suptitle(f'{title_including_silent_cells}\n\n'
        # TODO just define n_silent_cells alongside responses_including_silent/responses
        # def, then remove separate use of `silent_cells` here (+ use responses instead
        # of responses_including_silent)?
        # (doing it earlier would be complicated/impossible in n_seeds > 1 case
        # though...)
        f'{silent_cells.sum()} silent cells / {len(to_cluster)} total'
    )
    savefig(cg, param_dir, f'responses_nosilent{clust_suffix}')

    # TODO TODO TODO also plot in order of clustered ROIs from pre-tuned control+kiwi
    # panel, so that we can concat side-by-side after boosting APL, to show effect on
    # responses across both panels at once (can't boost APL while doing the pre-tuning,
    # so we won't have responses on both panels as currently run, unless we explicitly
    # added a call to run on both panels [in a separate step after pre-tuning])

    # TODO TODO also version that shows any mask used for apl boosting, for row colors
    # (maybe in addition to KC_TYPE colors, if easily possible, in which case it
    # wouldn't need to be in separate plot(s))

    # TODO TODO also make a plot that is clustering on spike_counts, and showing
    # subtypes when available, just as natmix_data does (could maybe just move that code
    # here? don't need in two places)

    # TODO also plot wPNKC (clustered?) for matts + my stuff?
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

    panel2sparsity_ylims = {
        # TODO add one for megamat? (only sensitivity analysis currently has these figs
        # in paper for megamat, but maybe betty will end up wanting these plots alone
        # anyway? in both cases, just need to find a range that works).
        #
        # 0.21 not enough for some.
        'validation2': [0, 0.22],
        # could prob do [0, 2], but might as well keep same as validation2. could just
        # hardcode this in general (or at least as long as the data is within limit?)?
        # TODO TODO fix (+ update validation?) actually some stuff is past this
        # apparently... (oh, it was actually on hallem data that it was failing, in the
        # uniform model)
        # TODO TODO fix so we fall back to no scale set (w/ warning), or so hallem isn't
        # considered megamat here (almost certainly former)?
        #'megamat': [0, 0.22],
    }
    # ylim=None will let plot_sparsity_per_odor set it
    ylim = panel2sparsity_ylims.get(panel)
    combined_fig, sparsity_ax = plot_sparsity_per_odor(sparsity_per_odor,
        comparison_sparsity_per_odor, '', ylim=ylim
    )

    #sparsity_ylim_max = 0.5
    # to exceed .706 in (fixed_thr=120.85, wAPLKC=0.0) param case
    sparsity_ylim_max = 0.71
    # TODO TODO are these the ylims used for validation2 modelling plots in current
    # (2025-02-18) modeling.svg for paper? are any other plots in paper using this?
    # maybe for sensitivity analysis?
    sparsity_ax.set_ylim([0, sparsity_ylim_max])
    # TODO TODO fix sensitivity analysis to not give us stuff outside this
    # comparison_sparsity_per_odor isn't being pushed to the same extreme, and should be
    # well within this limit.
    #assert not (sparsity_per_odor[sparsity_col] > sparsity_ylim_max).any()

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
        # TODO delete?
        # if assertion fails, will also need to subset comparison_responses to megamat
        # odors (in commented code below)
        #if comparison_responses is not None:
        #    plot_n_odors_per_cell(comparison_responses, ax, label_suffix=' (tuned)',
        #        color='gray', title=title_including_silent_cells
        #    )

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
        # TODO (still an issue?) where are dirs like
        # <panel>/tuned-on_control-kiwi__dff_scale-to-avg-max__data_pebbled__hallem-tune_False__pn2kc_hemibrain__fixed-thr_0__wAPLKC_0.00
        # coming from??? something not working right?

        shared_model_kws = {k: v for k, v in model_kws.items() if k not in (
            # plot_dir would conflict with first positional arg of
            # fit_and_plot_mb_model, and we don't want plots for sensitivity analysis
            # subcalls anyway (could fix if we did want those plots).
            'plot_dir',
            # excluding target_sparsity b/c that is mutually exclusive w/ fixing
            # threshold and KC<->APL inhibition, as all these calls will.
            'target_sparsity',

            # this factor only relevant (+ should only be passed if) we are also passing
            # target_sparsity.
            'target_sparsity_factor_pre_APL',

            'equalize_kc_type_sparsity',
            'ab_prime_response_rate_target',

            # TODO actually support varying just APL in this case? or still varying
            # vector thrs too?
            'homeostatic_thrs',

            # will default to False (via fit_mb_model default) once I remove this, which
            # is what I want
            'repro_preprint_s1d',

            # TODO add return_dynamics/plot_example_dynamics? (to make default to
            # False)?
        )}

        # TODO TODO how to support vector fixed_thr?
        # (also, probably remove line above setting vector fixed_thr=None in
        # equalize*=True case, so this branch also works on initial equalize*=True
        # calls)
        tuned_fixed_thr = param_dict['fixed_thr']

        APL_weights = get_APL_weights(param_dict, model_kws)
        del param_dict
        assert isinstance(APL_weights, dict)
        # TODO need to test we don't actually need wKCAPL in the cases of tianpei's
        # code that also separately defined this (which is prob also the only reason
        # get_APL_weights is returning a dict with both?)
        tuned_wAPLKC = APL_weights['wAPLKC']

        # TODO modify fit_mb_model+olfsysm code in this case to use
        # wAPLKC_scale+preset_wAPLKC (and to have connectome_APL_weights return
        # consistently scaled version of the vector, so we can use the wAPLKC_scale
        # output?), and remove this special casing here
        if isinstance(tuned_wAPLKC, pd.Series):
            # this should be the only case that has pd.Series here
            assert not use_connectome_APL_weights and one_row_per_claw
            raise NotImplementedError('need to use wAPLKC_scale in this case too, in '
                'order to support sensitivity analysis'
            )
        #

        # TODO assert wKCAPL (/ wKCAPL_scale) is ~ (wAPLKC / <#-KCs>)?
        # (or otherwise handle case where we also have that separately?)

        assert tuned_fixed_thr is not None and tuned_wAPLKC is not None
        # TODO TODO also add support for wAPLPN, at least to still repro sweep of other
        # two parameters for those cases?

        checks = True
        # TODO want to keep after i finish test_mb_model.test_fixed_inh_params?
        # probably, just to be sure it would work in this context (args might be
        # slightly diff)
        if checks:
            print('checking we can recreate responses by hardcoding tuned '
                'fixed_thr/wAPLKC...', end=''
            )
            # TODO figure out why this wasn't working on some runs of
            # kiwi/control data (re-running w/ `-i model` seemed to fix it, but unclear
            # on why that would be. using cache after break it again?) (doesn't seem
            # like it...) (encountered again 2025-02-20. going to re-run w/ `-i model`,
            # but do have a recent backup of whole mb_modeling dir, if i want to compare
            # outputs) (likely this has been resolved. delete comment.)
            # TODO why *was* responses2.sum().sum() == 0 in kiwi/control case???
            # (prob same reason sens analysis failing there)
            # (not sure i can repro, same as w/ failing assertion below)
            #
            # TODO silence output here? makes surrounding prints hard to follow
            # TODO also check spike_counts (2nd / 4 returned values)?
            responses2, _, _, _ = fit_mb_model(sim_odors=sim_odors,
                fixed_thr=tuned_fixed_thr, wAPLKC=tuned_wAPLKC, **shared_model_kws
            )
            # NOTE: this can fail if loading responses from cache (where values are the
            # same, but new output also has KC_TYPE in index)
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
            # TODO TODO switch all mean_fixed_thr stuff below back to rel_fixed_thr?
            # that code was a lot simpler... and some interpretability benefits too (w/
            # tradeoffs)
            #tried = pd.DataFrame(columns=['rel_fixed_thr', 'wAPLKC', 'sparsity'])
            if isinstance(tuned_fixed_thr, float):
                tried = pd.DataFrame(columns=['fixed_thr', 'wAPLKC', 'sparsity'])
            else:
                assert isinstance(tuned_fixed_thr, np.ndarray)
                tried = pd.DataFrame(columns=['mean_fixed_thr', 'wAPLKC', 'sparsity'])

        sens_analysis_kw_defaults = dict(
            n_steps=3,
            fixed_thr_param_lim_factor=0.5,
            wAPLKC_param_lim_factor=5.0,
            drop_nonpositive_fixed_thr=True,
            drop_negative_wAPLKC=True,
        )
        if sens_analysis_kws is None:
            sens_analysis_kws = dict()
        else:
            assert all(k in sens_analysis_kw_defaults for k in sens_analysis_kws.keys())

        sens_analysis_kws = {**sens_analysis_kw_defaults, **sens_analysis_kws}

        # TODO TODO try implementing alternative means of specifying bounds of sens
        # analysis: try sweeping each parameter until we reach some 2nd target response
        # rate (e.g. 2-5%, << typical target response rate when tuning both params of
        # ~10%) (then go the same distance [relative?] the other way? or what? not sure
        # this works...)
        #
        # want to avoid trial-and-error setting of below parameters such that corners of
        # grid each see similarly low/extreme response rates
        # TODO TODO TODO easier to implement such a thing by just slightly modifying
        # olfsysm C++ code?

        # needs to be odd so grid has tuned values (that produce outputs used elsewhere
        # in paper) as center.
        n_steps = sens_analysis_kws['n_steps']
        assert n_steps % 2 == 1, f'n_steps must be odd (got {n_steps=})'

        # TODO might need to be <1 to have lower end be reasonable?  but that prob won't
        # be enough (meaning? delete comment or elaborate) for upper ends...
        #
        # had previously tried up to at least 3, as well (not sure how it compared now
        # though...)
        #
        # for getting upper left 3x3 from original 4x4 (which was generated w/ this
        # param doubled, and n_steps=5 rather than 3)
        fixed_thr_param_lim_factor = sens_analysis_kws['fixed_thr_param_lim_factor']

        # TODO try seeing if we can push this high enough to start getting missing
        # correlations. 1000? why only getting those for high fixed_thr? and where
        # exactly do they come from?
        #
        # NOTE: was not seemingly able to get odors w/ no cells responding to them by
        # increasing this param (up to 100.0)...
        # 10.0 was used for most of early versions of this
        wAPLKC_param_lim_factor = sens_analysis_kws['wAPLKC_param_lim_factor']

        drop_nonpositive_fixed_thr = sens_analysis_kws['drop_nonpositive_fixed_thr']
        drop_negative_wAPLKC = sens_analysis_kws['drop_negative_wAPLKC']

        # TODO try 0 for min of fixed_thr_steps (remove drop_zero=True, and tweak step
        # size to get at least 1 <=0) (current steps not clipped by it)?
        # is 0 the lowest value that maximizes response rate? or are negative vals
        # meaningful given how this is implemented?
        # (would allow me to simplify code slightly if this could be handled as wAPLKC)
        fixed_thr_steps = step_around(tuned_fixed_thr, fixed_thr_param_lim_factor,
            'fixed_thr', n_steps=n_steps, drop_negative=True,
            drop_zero=drop_nonpositive_fixed_thr
        )

        # TODO delete
        ## TODO TODO also get this to work for scalar input case (+ switch all outputs /
        ## plots to using these relative descriptions?) (test whether it already does?)
        ## TODO also handle wAPLKC steps this way, for consistency?
        #rel_fixed_thr_steps = (fixed_thr_steps / tuned_fixed_thr)

        #if fixed_thr_steps.ndim > 1:
        #    firstcell_rel_fixed_thr_steps = rel_fixed_thr_steps[:,0]
        #    assert np.allclose(
        #        np.stack([firstcell_rel_fixed_thr_steps] * len(tuned_fixed_thr)).T,
        #        rel_fixed_thr_steps
        #    )
        #    rel_fixed_thr_steps = firstcell_rel_fixed_thr_steps

        wAPLKC_steps = step_around(tuned_wAPLKC, wAPLKC_param_lim_factor, 'wAPLKC',
            n_steps=n_steps, drop_negative=drop_negative_wAPLKC
        )

        print(f'{tuned_fixed_thr=}')
        print(f'{tuned_wAPLKC=}')
        print()
        print('parameter steps around sparsity-tuned values (above):')

        if isinstance(tuned_fixed_thr, float):
            print(f'{fixed_thr_steps=}')
        else:
            assert isinstance(tuned_fixed_thr, np.ndarray)
            # e.g. from:
            # array([[129.39, 129.39, 129.39, ..., 129.39, 129.39, 158.97],
            #        [258.79, 258.79, 258.79, ..., 258.79, 258.79, 317.94],
            #        [388.18, 388.18, 388.18, ..., 388.18, 388.18, 476.91]])
            # ...to:
            # array([133.81, 267.61, 401.42])
            mean_fixed_thr_steps = fixed_thr_steps.mean(axis=1)
            print(f'{mean_fixed_thr_steps=}')

        print(f'{wAPLKC_steps=}')
        # TODO delete
        #print(f'(relative to tuned) {rel_fixed_thr_steps=}')
        #print(f'(absolute) {wAPLKC_steps=}')
        print()

        step_choice_param_dict = {
            'tuned_fixed_thr': tuned_fixed_thr,
            'tuned_wAPLKC': tuned_wAPLKC,

            'n_steps': n_steps,
            'fixed_thr_param_lim_factor': fixed_thr_param_lim_factor,
            'wAPLKC_param_lim_factor': wAPLKC_param_lim_factor,
            'drop_negative_wAPLKC': drop_negative_wAPLKC,
            'drop_nonpositive_fixed_thr': drop_nonpositive_fixed_thr,
        }


        if isinstance(tuned_fixed_thr, float):
            step_choice_param_dict['fixed_thr_steps'] = fixed_thr_steps
        else:
            assert isinstance(tuned_fixed_thr, np.ndarray)
            # should be fully determined by above, but just including for easier
            # inspection
            #step_choice_param_dict['rel_fixed_thr_steps'] = rel_fixed_thr_steps
            step_choice_param_dict['mean_fixed_thr_steps'] = mean_fixed_thr_steps

        step_choice_param_dict['wAPLKC_steps'] = wAPLKC_steps
        step_choice_params = pd.Series(step_choice_param_dict)

        to_csv(step_choice_params, parent_output_dir / 'step_choices.csv',
            # so column name '0' doesn't get added (also added if doing ser.to_frame())
            header=False
        )

        # TODO delete
        #tried_wide = pd.DataFrame(data=float('nan'), columns=rel_fixed_thr_steps,
        #    index=wAPLKC_steps
        #)
        #tried_wide.columns.name = 'rel_fixed_thr'
        if isinstance(tuned_fixed_thr, float):
            try:
                # TODO TODO add unit test covering this failing case
                # TODO TODO fix:
                # (can't seem to repro in prat_claws=True & connectome_APL=True case.
                # was it only an issue w/ prat_claws=True & connectome_APL=False case?
                # or what?)
                # (ok, yea seems like only an issue for connectome_APL=False case?
                # also true w/ prat_claws=False? affect all one-row-per-claw cases?)
                # ValueError: Index data must be 1-dimensional
                tried_wide = pd.DataFrame(data=float('nan'), columns=fixed_thr_steps,
                    index=wAPLKC_steps
                )
            # TODO delete try/except. what error was this to catch anyway?
            except ValueError:
                breakpoint()
            #
        else:
            assert isinstance(tuned_fixed_thr, np.ndarray)
            tried_wide = pd.DataFrame(data=float('nan'), columns=mean_fixed_thr_steps,
                index=wAPLKC_steps
            )

        tried_wide.columns.name = 'mean_fixed_thr'
        tried_wide.index.name = 'wAPLKC'

        # should be something that won't appear in actual computed values. NaN may
        # appear in computed values. after loop, we check that we no longer have any of
        # these.
        corr_placeholder = 10

        # TODO delete
        #row_index = pd.MultiIndex.from_product([rel_fixed_thr_steps, wAPLKC_steps],
        #    names=['rel_fixed_thr', 'wAPLKC']
        #)
        if isinstance(tuned_fixed_thr, float):
            row_index = pd.MultiIndex.from_product([fixed_thr_steps, wAPLKC_steps],
                names=['fixed_thr', 'wAPLKC']
            )
        else:
            assert isinstance(tuned_fixed_thr, np.ndarray)
            row_index = pd.MultiIndex.from_product([mean_fixed_thr_steps, wAPLKC_steps],
                names=['mean_fixed_thr', 'wAPLKC']
            )

        col_index = corr_triangular(pearson).index
        pearson_at_each_param_combo = pd.DataFrame(index=row_index, columns=col_index,
            data=corr_placeholder
        )
        del row_index, col_index

        # TODO any point in having this if we are deleting root of all these above?
        # delete?
        ignore_existing = True

        _add_combined_plot_legend = True

        for fixed_thr, wAPLKC in tqdm(itertools.product(fixed_thr_steps, wAPLKC_steps),
            total=len(fixed_thr_steps) * len(wAPLKC_steps),
            unit='fixed_thr+wAPLKC combos'):

            print(f'{fixed_thr=}')
            # TODO delete
            #rel_fixed_thr = _single_unique_val(fixed_thr / tuned_fixed_thr, exact=False)

            #print(f'{rel_fixed_thr=}')
            #
            print(f'{wAPLKC=}')

            # TODO rename 'thr' prefix to 'rthr' or something, now that it's relative?
            # or otherwise distinguish outputs? (+ relabel plots and stuff)
            # TODO delete
            #dirname = f'thr{rel_fixed_thr:.2f}_wAPLKC{wAPLKC:.2f}'
            mean_fixed_thr = None
            if isinstance(fixed_thr, float):
                dirname = f'thr{fixed_thr:.2f}_wAPLKC{wAPLKC:.2f}'
            else:
                mean_fixed_thr = fixed_thr.mean()
                dirname = f'mean-thr{mean_fixed_thr:.2f}_wAPLKC{wAPLKC:.2f}'

            # NOTE: created by inner fit_and_plot... call below
            output_dir = parent_output_dir / dirname

            if output_dir.exists() and not ignore_existing:
                print(f'{output_dir} already existed. skipping!')
                continue

            # TODO delete? (if i want to restore, and don't want to use rel instead of
            # mean fixed thr, would prob need an outer conditional here, and would get
            # even uglier)
            '''
            #
            #if ((tried.rel_fixed_thr <= rel_fixed_thr) & (tried.wAPLKC <= wAPLKC) &
            if ((tried.mean_fixed_thr <= mean_fixed_thr) & (tried.wAPLKC <= wAPLKC) &
                (tried.sparsity < min_sparsity)).any():

                print(f'sparsity would be < {min_sparsity=}')
                continue

            #elif ((tried.rel_fixed_thr >= rel_fixed_thr) & (tried.wAPLKC >= wAPLKC) &
            elif ((tried.mean_fixed_thr >= mean_fixed_thr) & (tried.wAPLKC >= wAPLKC) &
                    (tried.sparsity > max_sparsity)).any():

                print(f'sparsity would be > {max_sparsity=}')
                continue
            '''
            #

            curr_params = fit_and_plot_mb_model(output_dir,
                comparison_responses=responses_including_silent,
                sim_odors=sim_odors, sensitivity_analysis=False, _in_sens_analysis=True,
                # this should be the only place we use fixed_thr instead of
                # [rel|mean]_fixed_thr, inside this loop (b/c fixed_thr can now
                # sometimes be a vector of length equal to # of KCs, which makes it
                # mostly unusable for plot/column labels / etc)
                fixed_thr=fixed_thr, wAPLKC=wAPLKC,
                _add_combined_plot_legend=_add_combined_plot_legend,
                # not passing param_dir_prefix here, b/c that should be in parent
                # directory, and should be easy enough to keep track of from that
                extra_params=extra_params,
                **shared_model_kws
            )
            # shouldn't need to call _write_inputs_for_reproducibility here, as should
            # also be in parent dir

            # should only be None in first_seed_only=True case, but not doing any
            # multi-seed runs w/ sensitivity analysis. only doing sensitivity analysis
            # for hemibrain + pebbled case currently.
            assert curr_params is not None

            # (added only if wAPLKC/fixed_thr passed)
            pearson = curr_params['pearson']

            pearson_ser = corr_triangular(pearson)
            assert pearson_ser.index.equals(pearson_at_each_param_combo.columns)
            # TODO need to assert this thr value is IN index already?
            if mean_fixed_thr is None:
                pearson_at_each_param_combo.loc[fixed_thr, wAPLKC] = pearson_ser
            else:
                pearson_at_each_param_combo.loc[mean_fixed_thr, wAPLKC] = pearson_ser
            # TODO delete
            #pearson_at_each_param_combo.loc[rel_fixed_thr, wAPLKC] = pearson_ser

            sparsity = curr_params['sparsity']
            print(f'sparsity={sparsity:.3g}')

            if output_dir.exists():
                # TODO need to assert this thr value is IN index already?
                if mean_fixed_thr is None:
                    tried_wide.loc[wAPLKC, fixed_thr] = sparsity
                else:
                    tried_wide.loc[wAPLKC, mean_fixed_thr] = sparsity
                # TODO delete
                #tried_wide.loc[wAPLKC, rel_fixed_thr] = sparsity

                # only want for first plot
                if _add_combined_plot_legend:
                    _add_combined_plot_legend = False

            if mean_fixed_thr is None:
                for_tried = {
                    # TODO delete
                    #'rel_fixed_thr': rel_fixed_thr, 'wAPLKC': wAPLKC,
                    'fixed_thr': fixed_thr, 'wAPLKC': wAPLKC, 'sparsity': sparsity
                }
            else:
                for_tried = {
                    # TODO delete
                    #'rel_fixed_thr': rel_fixed_thr, 'wAPLKC': wAPLKC,
                    'mean_fixed_thr': mean_fixed_thr,
                    'wAPLKC': wAPLKC, 'sparsity': sparsity
                }
            # replaced old .append w/ this hacky concat usage, to silence future warning
            # about append being removed
            tried = pd.concat([tried, pd.Series(for_tried).to_frame().T],
                ignore_index=True
            )
            # TODO delete
            #tried = tried.sort_values(['rel_fixed_thr', 'wAPLKC'])
            if mean_fixed_thr is None:
                tried = tried.sort_values(['fixed_thr', 'wAPLKC'])
            else:
                tried = tried.sort_values(['mean_fixed_thr', 'wAPLKC'])

            # can't use my to_csv as it currently errs if same CSV would get written >1
            # time in a given run
            tried.to_csv(tried_param_cache, index=False)

            print()

        # to make rows=rel_fixed_thr, cols=wAPLKC (consistent w/ how i had been laying
        # out the figure grids)
        tried_wide = tried_wide.T

        # (this, but not columns.name, makes it into CSV. it will be top left element.)
        # TODO delete
        #tried_wide.index.name = 'rows=rel_fixed_thr, cols=wAPLKC'
        if isinstance(tuned_fixed_thr, float):
            tried_wide.index.name = 'rows=fixed_thr, cols=wAPLKC'
        else:
            assert isinstance(tuned_fixed_thr, np.ndarray)
            tried_wide.index.name = 'rows=mean_fixed_thr, cols=wAPLKC'

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

        group_text = True

        if isinstance(tuned_fixed_thr, float):
            level_fn = lambda d: d['fixed_thr']
        else:
            assert isinstance(tuned_fixed_thr, np.ndarray)
            level_fn = lambda d: d['mean_fixed_thr']

        format_fixed_thr = lambda x: f'{x:.0f}'
        # TODO delete
        #level_fn = lambda d: d['rel_fixed_thr']
        #format_rel_fixed_thr = lambda x: f'{x:.2f}'

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
            # TODO delete
            #vgroup_formatter=format_rel_fixed_thr,
            #hgroup_formatter=format_rel_fixed_thr
        )
        fig.suptitle("Spearman of odor X odor Pearson correlations")
        savefig(fig, parent_output_dir, 'spearman_of_pearsons')

    return params_for_csv


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


@produces_output(verbose=False)
# TODO correct type hint for model?
# (statsmodels.regression.linear_model.RegressionResultsWrapper, but maybe something
# more general / not the wrapper?)
def save_model(model, path: Path) -> None:
    model.save(path)


spike_delta_col: str = 'delta_spike_rate'
est_spike_delta_col: str = f'est_{spike_delta_col}'

# for histograms of dF/F, transformed versions, or estimated spike deltas derived from
# one of the former
n_bins: int = 50

# TODO add some kwargs to explicitly control whether we fit dF/F->est-spiking fn?
def scale_dff_to_est_spike_deltas_using_hallem(plot_dir: Path, certain_df: pd.DataFrame,
    roi_depths: Optional[pd.DataFrame] = None, *, dff_col: str = 'delta_f_over_f'
    ) -> Tuple[pd.DataFrame, ParamDict, str, pd.DataFrame, List[Path]]:
    # TODO doc

    # TODO w/ a verbose flag, say which odors / glomeruli overlapped w/ hallem

    # I think deltas make more sense to fit than absolute rates, as both can go negative
    # and then we could better filter out points from non-responsive (odor, glomerulus)
    # combinations, if we wanted to.
    hallem_delta = orns.orns(columns=glomerulus_col, add_sfr=False)
    hallem_delta = abbrev_hallem_odor_index(hallem_delta)

    #our_odors = {olf.parse_odor_name(x) for x in certain_df.index.unique('odor1')}

    # TODO delete?
    # TODO or print intersection of these w/ stuff here (or at least stuff
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

    # TODO (prob delete. handling prob can not be improved) match glomeruli up to hallem
    # names (may need to make some small decisions)
    # (basically, are there any that are currently unmatched that can be salveaged?)

    # TODO TODO also check which of our_odors are in hallem lower conc data
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

    assert hallem_delta.columns.name == glomerulus_col
    hallem_glomeruli = set(hallem_delta.columns)

    # TODO delete. (after actually checking...)
    # TODO check no naming issues
    # {'DM3+DM5', 'DA4m' (2a), 'VA1d' (88a), 'DA4l' (43a), 'DA3' (23a), 'VA1v' (47b),
    # 'DL3' (65a, 65b, 65c), 'DL4' (49a, 85f)}
    print(f'{(hallem_glomeruli - our_glomeruli)=}')
    #

    # TODO delete. (after actually checking...)
    # TODO TODO check pdf receptor names matches what i get from drosolf w/o passing
    # columns=glomerulus_col, then use drosolf receptors to check these
    # TODO TODO check receptors of all these are not in hallem
    # TODO TODO print this out and check again. not clear on why DM3 was ever here...
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
    del hallem_glomeruli, hallem_glomeruli_in_task

    our_glomeruli = np.array(sorted(our_glomeruli))
    our_glomeruli_in_task = np.array([
        x in glomerulus2receptors.keys() for x in our_glomeruli
    ])
    # True for now, but may not always be?
    assert our_glomeruli_in_task.all()
    del our_glomeruli, our_glomeruli_in_task

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

    # TODO replace w/ call just renaming 'roi'->glomerulus_col
    assert 'fly_id' == fly_mean_df.columns.names[0]
    # TODO also assert len of names and/or names[1] is 'roi'?
    fly_mean_df.columns.names = ['fly_id', glomerulus_col]

    assert num_notnull(fly_mean_df) == n_before
    assert fly_mean_df.shape == shape_before
    assert set(fly_mean_df.dtypes) == {np.dtype('float64')}

    # TODO delete? here and elsewhere? (was before fly_mean_df code)
    mean_df = fly_mean_df.groupby(glomerulus_col, axis='columns').mean()

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
        assert fly_mean_df.columns.get_level_values(glomerulus_col).equals(
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
            on=['panel', 'fly_id', glomerulus_col]
        )

        fly_mean_df = fly_mean_df.set_index(index_levels_before)

        assert fly_mean_df.shape[0] == shape_before[0]
        assert fly_mean_df.shape[-1] == (shape_before[-1] + 1)

    assert num_notnull(fly_mean_df[dff_col]) == n_notnull_before
    assert num_null(fly_mean_df[dff_col]) == n_null_before

    # TODO what is this actually dropping? doc
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


    qs = [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]
    # quantile 0 = min, 1 = max. after unstacking, columns will be quantiles.
    fly_quantiles = fly_mean_df.groupby('fly_id')[dff_col].quantile(qs).unstack()

    avg_flymin = fly_quantiles[0].mean()
    avg_flymax = fly_quantiles[1].mean()
    # TODO assert all flies have same panels? (or otherwise might need to compute
    # per-panel?) shouldn't be an issue w/ kiwi+control tho, as all those flies should
    # have had both panels (+ diags)

    # NOTE: seems to be more variation in upper end than in inhibitory values
    # maybe i should be scaling the two sides diff then? (didn't seem worth it)
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
    # TODO use one of newer strs in al_analysis.py for this (-> move to al_util?)?
    # might this ever be 'Z-scored F' instead of dF/F?
    dff_desc = f'mean glomerulus {dff_latex}'
    # TODO refactor to preprend dff_desc inside loop (rather than manually for each
    # of these)
    scaling_method2desc = {
        None: dff_desc,

        #'minmax': f'{dff_desc}\n[0,1] scaled within fly',

        'zscore': f'{dff_desc}\nZ-scored within fly',

        'maxabs': dff_desc + '\n$fly_{max} \\rightarrow 1$, 0-preserved',

        # TODO check adding escaped '\\' in front of 'overline' (still? did it ever
        # w/o escaping the '\'?) produces what i want
        'to-avg-max':
            dff_desc + '\n$fly_{max} \\rightarrow \\overline{fly_{max}}$, 0-preserved',

        'split-minmax':
            # TODO latex working yet?
            f'{dff_desc}\n$+ \\rightarrow [0, 1]$\n$- \\rightarrow [-1, 0]$',

        'split-minmax-to-avg': (dff_desc +
            # TODO latex working yet?
            '\n$+ \\rightarrow [0, \\overline{fly_{max}}]$\n'
            '$- \\rightarrow [\\overline{fly_{min}}, 0]$'
        ),
    }

    # TODO factor out?
    # TODO rename to "add_scaled_dff_col" or something?
    def scale_one_fly(gdf: pd.DataFrame, method: str = 'to-avg-max'):
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
                # TODO TODO apply same scales to data that still has trials split out?
                # (-> save that)
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
        fly_mean_df = fly_mean_df.groupby('fly_id', sort=False, group_keys=False).apply(
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

    merged_dff_and_hallem = fly_mean_df.reset_index().merge(hallem_delta,
        left_on=['odor', glomerulus_col], right_on=['odor', glomerulus_col]
    ).reset_index()

    assert not merged_dff_and_hallem[spike_delta_col].isna().any()

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
        # TODO TODO require both this and other condition?
        # TODO delete
        '''
        if len(certain_df.loc['megamat'].dropna(axis='columns', how='all'
            ).columns.to_frame(index=False)[['date','fly_num']].drop_duplicates()
            ) >= 7:
        '''

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

    # TODO what exacty triggers this? doc in comment
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
            # TODO why this (and others) getting overwritten when being run w/ -C?
            # same w/ -c now (yes)? -P matter (don't think so)? seems to be a font
            # spacing issue for the most part? not sure why i'm just seeing it now
            # (2025)
            savefig(fig, plot_dir, f'hist_pebbled_{curr_dff_col}')

            # TODO also hist megamat subset of each of these? or at least of the pebbled
            # itself?
            # TODO or just loop over panels? easier below?


    # TODO iterate over options (just None and 'to-avg-max'? others worse) and verify
    # that what i'm using is actually the best (or not far off)?
    #scaling_method_to_use = None
    # 'to-avg-max'/'split-minmax-to-avg'/None all produce extremely visually similar
    # megamat/est_orn_spike_deltas*.pdf plots (including the correlation plots)
    # (as expected, since they keep 0)
    scaling_method_to_use = 'to-avg-max'

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
    # TODO move outside this fn
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


    # TODO move outside this fn
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
            linestyle='None', linewidth=0, edgecolor='white'
        )
        ax.plot(xs, est_df[est_spike_delta_col], color=color)
        ax.fill_between(xs, est_df[ci_lower_col], est_df[ci_upper_col], color=color,
            **fill_between_kws
        )
        if separate_inh_model:
            inh_color = 'red'
            # pylint: disable=possibly-used-before-assignment
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


    # TODO move def of these to module level? (maybe prepending plot_dir inside usage?)
    copy_to_model_dirs = []

    dff_to_spiking_model_path = plot_dir / 'dff2spiking_fit.p'
    copy_to_model_dirs.append(dff_to_spiking_model_path)

    if separate_inh_model:
        dff_to_spiking_inh_model_path = plot_dir / 'dff2spiking_inh_fit.p'
        copy_to_model_dirs.append(dff_to_spiking_inh_model_path)

    # TODO TODO TODO make a (matrix, probably) plot with just this data too (and at
    # least say in text which fraction of odors we are using, and N [+fly IDs?] for
    # each?)
    dff_to_spiking_data_csv = plot_dir / 'dff2spiking_model_input.csv'
    copy_to_model_dirs.append(dff_to_spiking_data_csv)

    # TODO move all fitting + plotting up above (~where current seaborn plotting
    # is), so i can do for all scaling choices, like w/ current seaborn plots (still
    # just doing modelling w/ one scaling choice)

    # so we have a record of which scaling choice we made (modelling plots already show
    # many parameters and this one isn't going to vary across modelling outputs from a
    # given run)
    # TODO save this and other non-plot outputs we need to load saved dF/F->spiking
    # fit outside of plot dirs, so i can swap between png and pdf w/o issue...
    dff_to_spiking_model_choices_csv = plot_dir / 'dff2spiking_model_choices.csv'
    copy_to_model_dirs.append(dff_to_spiking_data_csv)
    #

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

    # TODO also save + check response_calc_params match (currently module level variable
    # in al_analysis.py) (and do same for all data being fed thru model.  response calc
    # should match that for data used to compute dF/F -> est spike delta fn) (or
    # sufficient that we are saving input? and could then figure out which response stat
    # fn that input was using, if needed)

    if not use_saved_dff_to_spiking_model:
        to_csv(dff_to_spiking_model_choices, dff_to_spiking_model_choices_csv,
            header=False
        )
        # to check no more dtype issues (and just that we saved correctly)
        saved = read_dff_to_spiking_model_choices()
        assert saved.equals(dff_to_spiking_model_choices)
        del saved

        # TODO also add depth col (when available) here?
        cols_to_save = ['fly_id', 'odor', glomerulus_col, dff_col]

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

        # NOTE: ['odor', 'fly_id', glomerulus_col] would be unique if not for those few
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
        save_model(model, dff_to_spiking_model_path)

        if separate_inh_model:
            cprint(
                f'saving separate inhibition model to {dff_to_spiking_inh_model_path}',
                _cprint_color
            )
            save_model(model, dff_to_spiking_inh_model_path)

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
    del params_for_suffix

    if not use_saved_dff_to_spiking_model:
        plot_fname = f'dff_vs_hallem__{dff2spiking_choices_str}'

        fig, _ = plt.subplots()
        # TODO factor into same fn that fits model?
        plot_dff2spiking_fit(merged_dff_and_hallem, model)
        # normalize_fname=False to prevent '__' from getting replaced w/ '_'
        # TODO TODO (still an issue?) fix -c/-C failure here!
        # TODO add a to_csv(merged_dff_and_hallem, <some-path>) call first, to
        # sanity check input data not changing (pretty sure it's not)?
        # TODO need to change tolerance values in savefig (-c/-C) check of output
        # equivalence? or possible to make this completely deterministic (can i repro
        # this -c/-C failure on adjacent runs? maybe something actually did change?)?
        # sns.scatterplot and mpl.scatterplot (which former calls) both seem
        # deterministic though... or at least don't seem to have any seed kwargs / etc.
        # this was also the part of the plot that seemed (from the diff) like it had
        # changed...
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

            assert group_col in (glomerulus_col, 'depth_bin')
            if group_col == glomerulus_col:
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

            # pylint: disable-next=possibly-used-before-assignment
            plot_dff2spiking_fit(df, model, inh_model, title_prefix=title_prefix)


        if roi_depths is not None:
            avg_depth_per_glomerulus = merged_dff_and_hallem.groupby(glomerulus_col)[
                roi_depth_col].mean()
            avg_depth_per_glomerulus.name = f'avg_{roi_depth_col}'
            merged_dff_and_hallem = merged_dff_and_hallem.merge(
                avg_depth_per_glomerulus, left_on=glomerulus_col, right_index=True
            )

            merged_dff_and_hallem = merged_dff_and_hallem.sort_values(
                f'avg_{roi_depth_col}').reset_index(drop=True)

        else:
            merged_dff_and_hallem = merged_dff_and_hallem.sort_values(glomerulus_col
                ).reset_index(drop=True)

        col = glomerulus_col
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


    # TODO TODO TODO also get per-fly scaled data that preserves trials (-> save that)
    # TODO TODO TODO or save per-fly scales separately, so it could be multiplied by
    # separate copies of the data w/ separate trials?
    #breakpoint()
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

    # TODO test whether downstream code works fine w/o stopping here (at least
    # check equiv in megamat case. may want to hardcode a skip of the validation2 panel
    # by default anyway [w/ a flag])
    # TODO would checking megamat subset of fly_mean_df is same between two runs (w/ all
    # data vs just megamat flies) get us most/all of the way there?
    #
    # TODO delete hack (see corresponding hack above where this flag is defined)
    # (plus now rest of modeling code loops over panels anyway, no?)
    if not use_saved_dff_to_spiking_model:
        print('EXITING EARLY AFTER HAVING SAVED MODEL ON ALL DATA (analyze specific '
            'panels with additional al_analysis runs, restricting date range to only '
            'one panel)!'
        )
        sys.exit()
    #

    # TODO plot histogram of fly_mean_df[est_spike_delta_col] (maybe resting on the x
    # axis in the same kind of scatter plot of (x=dF/F, y=delta spike rate,
    # hue=fly_id)?)

    # TODO rename? it's a series here, not a df (tho that should change in next
    # re-assignment, the one where RHS is unstacked...)
    mean_est_df = fly_mean_df.reset_index().groupby(['panel', 'odor', glomerulus_col],
        sort=False)[est_spike_delta_col].mean()
    del fly_mean_df

    # then odors will be columns and glomeruli will be rows, which is same as
    # orns.orns().T
    mean_est_df = mean_est_df.unstack(['panel', 'odor'])

    mean_est_df = sort_odors(mean_est_df)

    # TODO TODO save something like this, but keeping flies + trials too?
    #breakpoint()
    # TODO save w/ panel(s)/similar in name, or under panel dirs? at least as duplicate?
    # (then remove ignore_output_change_check=True, if so)
    # do we actually need anything at this path exactly (prob not)?
    # (or do i actually often want this to contain multiple panels? maybe? really lose
    # anything if i switch to per-panel? yes, prob cause diagnostics will differ
    # depending of context of other panels they are presented with)
    #
    # can fail -C/-c, cause this saves directly under:
    # <driver>_<indicator>/<plot_fmt>/ijroi/mb_modeling (instead of any panel-specific
    # subdirectory of `mb_modeling`), so multiple runs with different panels (typically
    # defined by start/end date args to `al_analysis.py`) will have different data here.
    to_csv(mean_est_df, plot_dir / 'mean_est_spike_deltas.csv',
        ignore_output_change_check=True
    )

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

    tidy_hallem = hallem_delta_wide.T.stack()

    # just to rename the second level from 'odor1'->'odor', to be consistent w/
    # above
    tidy_hallem.index.names = [glomerulus_col, 'odor']
    tidy_hallem.name = spike_delta_col
    tidy_hallem = tidy_hallem.reset_index()
    fig, ax = plt.subplots()
    sns.histplot(data=tidy_hallem, x=spike_delta_col, bins=n_bins, ax=ax)
    ax.set_title('all Hallem')
    # TODO TODO fix -c failure (just a tolerance thing?)
    savefig(fig, plot_dir, 'hist_hallem')

    # TODO move inside new loop over panels (doing for every panel, not just megamat)?
    tidy_hallem_megamat = tidy_hallem.loc[
        tidy_hallem.odor.apply(odor_is_megamat)
    ].copy()

    fig, ax = plt.subplots()
    sns.histplot(data=tidy_hallem_megamat, x=spike_delta_col, bins=n_bins, ax=ax)
    ax.set_title('Hallem megamat')
    savefig(fig, plot_dir, 'hist_hallem_megamat')
    #

    return (mean_est_df, dff_to_spiking_model_choices, dff2spiking_choices_str,
        hallem_delta_wide, copy_to_model_dirs
    )


# TODO modify to accept csv path for certain_df too (if that makes it easier to get some
# basic versions of the model runnable for other people)? either way, want to commit
# some example AL data to use.
# TODO rename certain_df to just df (or something less loaded)
# TODO TODO add kwarg to hardcode a dF/F -> spike delta scaling factor (or a
# general fn), and have it bypass the computation of this fn / related plotting/cache.
# (-> use for simple example uses of this fn, hardcoding scale factor = 127 from
# Remy-paper)
def model_mb_responses(certain_df: pd.DataFrame, parent_plot_dir: Path, *,
    roi_depths=None, skip_sensitivity_analysis: bool = False,
    skip_models_with_seeds: bool = False, skip_model_dynamics_saving: bool = False,
    skip_hallem_models: bool = False, first_model_kws_only: bool = False) -> None:
    # TODO when is it ok for certain_df to have NaNs? does seem current input has
    # some NaNs, which are only for some odors [for which no odors is NaN for all
    # fly-glomeruli]. any restrictions (if none, why was sam having issues?)
    # TODO allow passing in model_kw_list?
    """Passes input through dF/F -> est spike delta function, runs through MB model.

    Calls `fit_and_plot_mb_model` for each of several parameter options for the model,
    currently defined below in `model_kw_list`. Each call will have a directory created
    with model outputs (under `parent_plot_dir / 'mb_modeling/<panel>/<param_dir>'`).

    Some outputs under `parent_plot_dir / 'mb_modeling'` and `<panel>` sub-directories
    describe dF/F -> est spike delta function, parameters, and outputs.

    Args:
        certain_df: dataframe of shape (# odors [including repeats], # fly-glomeruli),
            with dF/F values from [potentially multiple] flies.

            Column index names should be ['date', 'fly_num', 'roi' (i.e. glomerulus
            name)].

            Row index names should be ['panel', 'is_pair', 'odor1', 'odor2', 'repeat']
            (possible that not all are required, but 'panel' and 'odor1' should be)

        parent_plot_dir: where a 'mb_modeling' subdirectory will be created to contain
            any model output directories (and other across-model plots).

            In typical use from `al_analysis.py`, this might be 'pebbled_6f/pdf/ijroi'.

        first_model_kws_only: only runs the model with the first set of parameters (in
            `model_kw_list` in this function), to more quickly test changes to model.
    """
    # TODO delete
    # TODO why `memory usage (MiB): rss=4136.91 vms=7756.15` already here???
    # (seems [from memory_profiler] that it's likely all from stuff loaded in
    # process_recording in al_analysis.main. can i reduce that at all? still need all
    # the big things?)
    #print_curr_mem_usage(end='')
    #print(', at start of model_mb_responses')
    #

    # TODO (not using roi_depths really, so not a big deal) why do roi_depths seem to
    # have one row per panel, rather than per recording?  current input has column
    # levels ['date', 'fly_num', 'roi'] and row levels ['panel', 'is_pair'])

    # TODO delete. for debugging.
    global _spear_inputs2dfs
    #

    # TODO TODO refactor to whare w/ natmix_data/analysis.py (copied from there)
    def drop_unused_model_odors(df: pd.DataFrame) -> pd.DataFrame:
        odor_strs = df.index.get_level_values('odor1')
        to_drop = (
            # hack to remove stuff like 'cmix-1 @ 0'
            odor_strs.str.contains('x-', regex=False) |

            # to remove 2 component mixtures (whether in-vial or air mixture)
            odor_strs.str.contains('+', regex=False) |

            # TODO replace w/ using parsing to get name or match solvent_str or
            # something
            # TODO could delete this one. already dropped (just a few lines) below.
            odor_strs.str.startswith('pfo')
        )
        if 'odor2' in df.index.names:
            to_drop = to_drop | (df.index.get_level_values('odor2') != solvent_str)

        df = df.loc[~to_drop].copy()

        df = drop_mix_dilutions(df)
        return df

    # TODO TODO delete (/make conditional) hack to remove odors we don't want to
    # analyze, before running kiwi/control data thru model (prob won't affect
    # megamat/validation anyway?)
    print('TOSSING ODORS WE WILL NOT ANALYZE IN NATMIX_DATA')
    certain_df = drop_unused_model_odors(certain_df)

    # TODO make and use a subdir in plot_dir (for everything in here, including
    # fit_and_plot... calls)

    # TODO have this taken in, rather than hardcoding 'mb_modeling'?
    plot_dir = parent_plot_dir / 'mb_modeling'

    # written below in _write_inputs_for_reproducibility.
    # not otherwise used.
    unmodified_orn_dff_input_df = certain_df.copy()

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
    if 'odor2' in certain_df.index.names:
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
    # (may no longer be true? may not matter)
    hack_strs_to_fix_mix_dilutions = hack_strs_to_fix_mix_dilutions + ' @ 0'

    index_df.loc[odor1_names.str.endswith('mix'), 'odor1'] = \
        hack_strs_to_fix_mix_dilutions

    certain_df.index = pd.MultiIndex.from_frame(index_df)
    del index_df

    # after above two hacks, for kiwi/control data, sort_odors should order odors as:
    # pfo, components, 2-component mix, 2-component air mix, 5-component mix, dilutions
    # of 5-component mix (with more dilute mixtures further towards the end)
    certain_df = sort_odors(certain_df)

    ret = scale_dff_to_est_spike_deltas_using_hallem(plot_dir, certain_df, roi_depths)

    (mean_est_df, dff_to_spiking_model_choices, dff2spiking_choices_str,
        hallem_delta_wide, copy_to_model_dirs
    ) = ret

    def _write_inputs_for_reproducibility(plot_root: Path, param_dict: ParamDict
        ) -> None:
        # TODO doc. param_dict really just used for output_dir and used_model_cache,
        # right?

        # plot_root should be the panel plot dir,
        # e.g. pebbled_6f/pdf/ijroi/mb_modeling/megamat
        assert plot_root.is_dir()

        output_dir = plot_root / param_dict['output_dir']
        assert output_dir.is_dir()

        used_model_cache = param_dict.get('used_model_cache', True)
        # should be ok to write these files regardless of `-c`
        # (check_outputs_unchanged) flag value, because we must have made it
        # thru fit_and_plot... above w/o it tripping anything
        if used_model_cache:
            return

        if al_util.verbose:
            print(f'copying model inputs to {output_dir.name}:')

        for to_copy in copy_to_model_dirs:
            # would need to check copy2 behavior in this case, if wanted to support
            assert not to_copy.is_symlink()

            dst = output_dir / to_copy.name

            if al_util.verbose:
                print(f'copying {to_copy}')

            # TODO or load/save, using diff fns for this depending on ext? not sure i
            # gain anything from that...
            #
            # should preserve as much file metadata as possible (e.g. modification
            # time). will overwrite `dst`, if it already exists.
            shutil.copy2(to_copy, dst)

        # TODO refactor to share date_format= part w/ consensus_df saving i copied this
        # from? / delete? not sure it's even relevant if date is in index...
        to_csv(unmodified_orn_dff_input_df, output_dir / 'full_orn_dff_input.csv',
            date_format=date_fmt_str
        )
        # TODO TODO use parquet instead (+ check can round trip w/o change)
        to_pickle(unmodified_orn_dff_input_df, output_dir / 'full_orn_dff_input.p')

    # TODO explicitly compare fitting on hallem vs hallem-megamat vs remy's megamat data
    # passed thru fn to get est spike deltas

    # TODO also include actual equation for scaling in this (just one constant w/
    # current choices)? currently just includes choices that influence that, but since
    # we aren't including full data in there (though we are copying it to each model dir
    # now), we can't quickly tell which scaling factor was used
    # (could refactor part of plotting code that gets `model_eq` for title, and use that
    # str equation here too? might want a bit more precision on some params, so maybe
    # include them [e.g. `model.params`] as well?)
    #
    # will be saved alongside later params, inside each model output dir
    # (for reproducibility)
    extra_params = {
        f'dff2spiking_{k}': v for k, v in dff_to_spiking_model_choices.to_dict().items()
    }

    # slightly nicer number (less sig figs) that is almost exactly the same as the
    # sparsity computed on remy's data (from binarized outputs she gave me on
    # 2024-04-03)
    remy_sparsity = 0.0915

    # TODO only do this if trying to repro paper (or at least only if panel is megamat
    # or something)? (calling fn below doesn't cause a meaningful change in memory
    # currently used, but just to remove unnecessary prints and processing)
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

    # TODO TODO break out a new script that just loads kiwi/control consensus df and
    # runs it through model_mb_responses (then move special casing for kiwi/control
    # panel below to some parameters passed in from there [/similar]?)
    # (+ prob put script in natmix_data repo)

    # TODO expose this as kwarg (+ maybe thread thru from CLI arg?)
    # TODO unit test model_mb_responses w/ this True, checking we can recreate committed
    # outputs
    # TODO set this True by default if input just has megamat/validation2 (+diags)?
    # TODO add test based on the =True branch of this? would be a bit redundant w/ some
    # existing tests, but still
    repro_remy_paper = False

    if not repro_remy_paper:
        # TODO allow overriding this w/ kwarg (or entries in kwarg list passed in?
        # both?)?
        # TODO default to None instead (or not specifying?) (letting inner fns default
        # apply)
        target_sparsity = 0.1
    else:
        # TODO move this special casing into def of input_model_kw_list below (in
        # repro_remy_paper=True conditional)?
        target_sparsity = remy_sparsity
        warn(f'since {repro_remy_paper=} using {target_sparsity=}')

    # TODO move to module level (+ rename?) (could be hacky alternative to passing in,
    # and could share w/ some tests then...)
    # TODO accept this as optional input kwarg?
    # TODO use dict_seq_product to define as much of this as i can
    # TODO modify al_analysis CLI -M arg to accept optional int (# of first entries
    # to use)?
    input_model_kw_list = [
        # TODO TODO TODO need another flag beyond prat_boutons, to also have distinct
        # boutons w/ diff dynamics in model (+ need to implement that too), rather than
        # counting all synapses on a bouton towards each claw on that bouton, which is
        # what current behavior should be
        dict(prat_claws=True, one_row_per_claw=True,
            # TODO TODO rename pn_claw_to_APL to just claw_to_APL? don't want to
            # cause confusion when also adding direct PN->APL inputs (and it is a claw
            # input right? or was i interpreting it wrong so far? what's distinction
            # anyway? just whether APL had a chance to act on bouton yet?)
            use_connectome_APL_weights=True, pn_claw_to_APL=True, prat_boutons=True
        ),

        # TODO delete? this code reasonable at all (well vm_sims are
        # NOTE: n_claws_active_to_spike code in olfsysm has not yet implemented
        # threshold picking for that case, so I was initially manually hardcoding a
        # threshold in olfsysm (and recompiling, iterating manually until sparsity
        # withihn tolerance).
        # TODO move manual thersholds out of olfsysm and in to here (+ check outputs
        # same) (+ have olfsysm err if thr_type is not 'fixed')
        # probably not, but that's probably ok. can just ignore them)?
        #dict(n_claws_active_to_spike=3, prat_claws=True, one_row_per_claw=True,
        #    use_connectome_APL_weights=True
        #),
        #dict(n_claws_active_to_spike=2, prat_claws=True, one_row_per_claw=True,
        #    use_connectome_APL_weights=True
        #),
        #
        ##dict(n_claws_active_to_spike=2, prat_claws=True, one_row_per_claw=True,
        ##    use_connectome_APL_weights=True, pn_claw_to_APL=True
        ##),
        #

        # TODO decide which of these one-row-per-claw variants i want to keep here

        # pn_claw_to_APL=true means APL activity depends directly on claw activity, and
        # does not require KC spiking (which would then provide APL activation through
        # all claws of that KC equally, regardless of whether that claw had any input
        # before KC spiking)
        dict(prat_claws=True, one_row_per_claw=True,
            # TODO TODO rename pn_claw_to_APL to just claw_to_APL? don't want to
            # cause confusion when also adding direct PN->APL inputs (and it is a claw
            # input right? or was i interpreting it wrong so far? what's distinction
            # anyway? just whether APL had a chance to act on bouton yet?)
            use_connectome_APL_weights=True, pn_claw_to_APL=True
        ),
        dict(prat_claws=True, one_row_per_claw=True, pn_claw_to_APL=True),

        # TODO sens analysis really only failing on this 2nd one (i.e. the one
        # WITHOUT use_connectome_APL_weights=True) (yes, b/c that path doesn't use
        # wAPLKC_scale, so doesn't support sensitivity analysis currently)?
        #
        # sensitivity analysis works for this
        dict(prat_claws=True, one_row_per_claw=True,
            use_connectome_APL_weights=True
        ),
        # TODO TODO fix sens analysis for this (-> run kiwi/control data thru it, just
        # like for above)
        dict(prat_claws=True, one_row_per_claw=True),

        # TODO restore + regen these w/ signed maxabs responses (and sensitivity
        # analysis for connectome APL weights one at least, or both if i fix it for
        # other case)
        #dict(one_row_per_claw=True),
        #dict(one_row_per_claw=True, use_connectome_APL_weights=True),

        # TODO restore after checking (+ fixing, if needed) implementation of APL
        # compartments
        #
        #dict(one_row_per_claw=True, APL_coup_const=0),
        # NOTE: APL_coup_const=0 should enable different activity in different APL
        # compartments (but without any coupling between them)
        # TODO give it a better name!
        # TODO TODO check these outputs are actually diff from w/ default
        # APL_coup_const=None, where there should only be one APL (no compartments)
        # TODO TODO try multiple radii (need to expose as kwarg)
        #dict(prat_claws=True, one_row_per_claw=True, APL_coup_const=0),
        #

        # TODO TODO (still an issue?) drop multiglomerular PNs are re-run all
        # prat_claws=True variants (hardcode inside that branch of connectome_wPNKC.
        # never want them)

        # TODO delete?
        # TODO try these again after scaling wPNKC within each KC (to have same
        # mean as before?)?
        #dict(prat_claws=True, one_row_per_claw=True, dist_weight='percentile'),
        #dict(prat_claws=True, one_row_per_claw=True, dist_weight='raw'),
        #

        # TODO remove these entries w/ equalize_kc_type_sparsity=True? (or do i actually
        # want this to be standard?)
        dict(
            weight_divisor=20,
            use_connectome_APL_weights=True,
            equalize_kc_type_sparsity=True,
        ),
        dict(
            weight_divisor=20,
            equalize_kc_type_sparsity=True,
        ),
        #

        # TODO delete? if i'm satisfied to always equalize_kc_type_sparsity=True now...
        # not sure i am.
        dict(
            weight_divisor=20,
            use_connectome_APL_weights=True,
        ),
        dict(
            weight_divisor=20,
        ),
        #

        # NOTE: uniform/hemibrain models currently use # of KCs from hemibrain
        # connectome (1837 if _drop_glom_with_plus=False [= old behavior], or 1830
        # otherwise). model would default to 2000 otherwise. fafb data more cells
        # (2482 in left, probably similar in right).
        dict(
            # TODO just move n_seeds=N_SEEDS either into defaults, or set in loop below,
            # if pn2kc_connections in a variable_n_claws case?
            pn2kc_connections='uniform', n_claws=7, n_seeds=N_SEEDS,
        ),
    ]

    deduped_model_kw_list = []
    seen_model_kws = set()
    for model_kws in input_model_kw_list:
        # will err if not all model_kws.values() are Hashable, and I don't intend to
        # support non-hashable values in these dicts
        curr_kws = frozenset(model_kws.items())
        if curr_kws not in seen_model_kws:
            deduped_model_kw_list.append(model_kws)
        else:
            warn(f'{model_kws=} duplicated in input_model_kw_list! only keeping first')
        seen_model_kws.add(curr_kws)

    input_model_kw_list = deduped_model_kw_list

    paper_sens_analysis_kws = dict(
        n_steps=3,
        fixed_thr_param_lim_factor=0.5,
        wAPLKC_param_lim_factor=5.0,
        # TODO why not keeping 0? that not work? actually matter (what
        # happened?)?
        drop_nonpositive_fixed_thr=True,
        drop_negative_wAPLKC=True,
    )

    # these values were for trying to find better combinations for
    # use on newer kiwi/control data (for analysis in
    # natmix_data/analysis.py). not 100% happy with outputs yet.
    natmix_sens_analysis_kws = dict(
        n_steps=7,
        fixed_thr_param_lim_factor=0.75,

        # TODO just leave these last 3 at default (that's what these
        # values are)?
        wAPLKC_param_lim_factor=5.0,
        drop_nonpositive_fixed_thr=True,
        drop_negative_wAPLKC=True,
    )

    # TODO just explicitly pass nothing and use defaults of inner fns? same as this?
    # TODO or default to values i'm now using for kiwi stuff?
    default_sens_analysis_kws = paper_sens_analysis_kws

    if repro_remy_paper:
        warn('replacing input_model_kw_list with list to recreate Remy-paper outputs, '
            f'since {repro_remy_paper=}'
        )
        input_model_kw_list = [
            dict(
                weight_divisor=20,
                # may need to explicitly set use_connectome_APL_weights=False, if
                # default for that ever changes

                _drop_glom_with_plus=False,

                # TODO still set this False in loop below, if panel == 'validation2'?
                # (i.e. not 'megamat')
                sensitivity_analysis=True,
            ),

            dict(
                # TODO be consistent about 100 vs N_SEEDS (maybe switch preprint repro
                # stuff below to 100? or have a var shared between repro cases that is
                # also defined to 100?)
                pn2kc_connections='uniform', n_claws=7, n_seeds=100,

                _drop_glom_with_plus=False,
            ),
        ]

    # TODO hardcode list of panels to NOT run sensitivity analysis on (just
    # validation2?)? (and only if repro_remy_paper?)

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
        # spike counts from each) (not seeing this script... move to test if i find it?)
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

        assert panel_est_df.equals(sort_odors(panel_est_df))

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
            # TODO use one of newer strs in al_analysis.py for this (-> move to
            # al_util?)? might this ever be 'Z-scored F' instead of dF/F?
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
            # TODO use one of newer strs in al_analysis.py for this (-> move to
            # al_util?)? might this ever be 'Z-scored F' instead of dF/F?
            xlabel=f'ORN {dff_latex}\nHallem glomeruli only'
        )
        plot_corr(mean_fly_dff_hallem_corr, panel_plot_dir,
            'orn_dff_hallem-subset_corr-dist',
            # TODO use one of newer strs in al_analysis.py for this (-> move to
            # al_util?)? might this ever be 'Z-scored F' instead of dF/F?
            xlabel=f'ORN {dff_latex}\nHallem glomeruli only', as_corr_dist=True
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
        scaling_method_to_use = dff_to_spiking_model_choices['scaling_method_to_use']
        est_corr = plot_responses_and_corr(est_df.T, panel_plot_dir,
            f'est_orn_spike_deltas_{dff2spiking_choices_str}',
            # TODO maybe borrow final part from scaling_method2desc (but current strs
            # there have more info than i want)
            xlabel=('est. ORN spike deltas\n'
                # TODO use one of newer strs in al_analysis.py for this (-> move to
                # al_util?)? might this ever be 'Z-scored F' instead of dF/F?
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
            zerofilled_hallem = reindex(hallem_megamat, panel_est_df.index,
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
        # TODO TODO why did the xticks seem to change (comparing old vs new version of
        # dff_scale-to-avg-max_hist_est-spike-delta_validation2.pdf, highlighted by -C)?
        # shape of rest seems the same. something meaningful? diff input subset or
        # something?
        sns.histplot(data=tidy_est, x=est_spike_delta_col, bins=n_bins, ax=ax)
        ax.set_title(f'pebbled {panel}')
        # TODO or save in panel dir? this consistent w/ saving of hallem megamat stuff
        # above tho...
        savefig(fig, plot_dir,
            f'{dff2spiking_choices_str}__hist_est-spike-delta_{panel}'
        )
        del tidy_est

        pebbled_input_df = panel_est_df

        comparison_orns = None
        comparison_kc_corrs = None

        # TODO also only do if repro_remy_paper=True?
        if panel == 'megamat':
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

            plot_corr(comparison_kc_corrs, panel_plot_dir, 'remy_kc_corr',
                xlabel='observed KCs'
            )
            plot_corr(comparison_kc_corrs, panel_plot_dir, 'remy_kc_corr-dist',
                xlabel='observed KCs', as_corr_dist=True
            )

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

            # TODO refactor to share w/ where i copied from
            fig, ax = plt.subplots()
            add_unity_line(ax)
            lineplot_kws = dict(
                ax=ax, data=merged_corrs, x=orn_col, y=kc_col, linestyle='None',
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
            # TODO use one of newer strs in al_analysis.py for this (-> move to
            # al_util?)? might this ever be 'Z-scored F' instead of dF/F?
            ax.set_xlabel(f'{metric_name} of raw ORN {dff_latex} (observed)')
            ax.set_ylabel(f'{metric_name} of KCs (observed)')

            metric_max = max(merged_corrs[kc_col].max(), merged_corrs[orn_col].max())
            metric_min = min(merged_corrs[kc_col].min(), merged_corrs[orn_col].min())

            plot_max = 1
            plot_min = -.5
            assert metric_max <= plot_max, f'{metric_max=} > {plot_max=}'
            assert metric_min >= plot_min, f'{metric_min=} < {plot_min=}'

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

        model_kw_list = []
        for model_kws in input_model_kw_list:
            model_kws = dict(model_kws)

            assert 'orn_deltas' not in model_kws
            model_kws['orn_deltas'] = pebbled_input_df

            assert 'comparison_orns' not in model_kws
            model_kws['comparison_orns'] = comparison_orns

            # TODO share 'hemibrain' default w/ two other places this defined
            pn2kc_connections = model_kws.get('pn2kc_connections', 'hemibrain')

            one_row_per_claw = model_kws.get('one_row_per_claw', False)
            if one_row_per_claw:
                # NOTE: hallem cases should be all added below (preprint repro stuff),
                # so shouldn't have to worry about dynamics trying to be saved for them
                # (would require way too much memory + space)
                #
                # if could be specified per-element in input_model_kw_list, would need
                # to del here if skip flag set
                assert 'return_dynamics' not in model_kws
                # now defaulting to saving these dynamics (which do take a lot of
                # storage space), unless `-s model-dynamics` are among skip option, when
                # called via `al_analysis.py` CLI [which sets this flag True])
                if not skip_model_dynamics_saving:
                    model_kws['return_dynamics'] = True
            else:
                if not skip_model_dynamics_saving:
                    # natmix_data/analysis.py is currently only thing that analyzing
                    # these, and only really for the one-row-per-claw outputs.
                    # takes up too much disk space to justify saving for everything,
                    # until needed.
                    warn('not saving model dynamics b/c one_row_per_claw=True '
                        'not in model_kws'
                    )

            if 'sensitivity_analysis' not in model_kws:
                # TODO variable_n_claw cases really not supported? fix that (-> update
                # code here)?
                if pn2kc_connections not in variable_n_claw_options:
                    # TODO warn we are defaulting to this (+ say you can disable by
                    # setting False explicitly?)?
                    model_kws['sensitivity_analysis'] = True

            if model_kws.get('sensitivity_analysis', False):
                if 'sens_analysis_kws' not in model_kws:
                    model_kws['sens_analysis_kws'] = dict(default_sens_analysis_kws)

            model_kw_list.append(model_kws)


        # TODO do in separate script, regardless of panel? not even using that panel's
        # data, right? (i suppose for KC comparison i still am, and using remy's data
        # there?)
        if panel == 'megamat':
            hallem_for_comparison = hallem_delta_wide.copy()
            assert hallem_for_comparison.index.str.contains(' @ -3').all()
            # so things line up in comparison_orns path (fit_mb_model hallem data has '@
            # -2' for each conc)
            hallem_for_comparison.index = hallem_for_comparison.index.str.replace(
                ' @ -3', ' @ -2'
            )
            # TODO delete? actually needed by anything?
            hallem_for_comparison.index.name = 'odor1'

            # parameter combinations to recreate preprint figures, using same Hallem
            # data as input (that Matt did when making those figures, before we had our
            # own ORN outputs)
            preprint_repro_model_kw_list = [
                dict(
                    pn2kc_connections='hemibrain',

                    # TODO delete/comment
                    # need to fix breakpoint hit in fit_mb_model (currently just
                    # commented it...)
                    _use_matt_wPNKC=True,

                    comparison_orns=hallem_for_comparison,

                    # TODO don't require this passed in! (do unconditionally)
                    # TODO or move to model_kws postprocesing in tune_on_hallem=True
                    # case
                    _strip_concs_comparison_kc_corrs=True,
                ),
                dict(
                    pn2kc_connections='uniform', n_claws=7, n_seeds=N_SEEDS,

                    # TODO delete/comment
                    # need to fix breakpoint hit in fit_mb_model
                    _use_matt_wPNKC=True,

                    # TODO probably also _add_back_methanoic_acid_mistake=True?
                    # shouldn't matter...
                    #_add_back_methanoic_acid_mistake=True,

                    comparison_orns=hallem_for_comparison,

                    # TODO don't require this passed in! (do unconditionally)
                    #
                    # since outputs of model will have ' @ -2' when using Hallem input,
                    # but KC comparison data has ' @ -3'. this will strip conc from all
                    # odor strings, when comparing data from these variables.
                    # NOTE: comparison_orns path currently strips unconditionally...
                    _strip_concs_comparison_kc_corrs=True,
                ),
                dict(
                    pn2kc_connections='hemidraw', n_claws=7, n_seeds=N_SEEDS,
                    _use_matt_wPNKC=True,
                    comparison_orns=hallem_for_comparison,
                    _strip_concs_comparison_kc_corrs=True,
                ),
            ]
            if not skip_hallem_models:
                # all entries in preprint_repro_model_kw_list use hallem data, and all
                # entries in model_kw_list (before line below) should not use hallem
                # data
                model_kw_list = model_kw_list + preprint_repro_model_kw_list
            else:
                warn('skipping all models using Hallem data (rather than our measured '
                    'ORN data) as input (because `-s model-hallem` CLI option)'
                )

        # hack to skip long running models, if I want to test something on pebbled and
        # hallem cases w/o re-running many seeds before getting an answer on the test.
        if skip_models_with_seeds:
            old_len = len(model_kw_list)
            model_kw_list = [x for x in model_kw_list if 'n_seeds' not in x]

            n_skipped = old_len - len(model_kw_list)
            warn(f'currently skipping {n_skipped} models with seeds! (because '
                '`-s model-seeds` CLI option)'
            )

        if repro_remy_paper:
            # probably don't want to use some of the other choices of args in cases
            # other than repro_remy_paper=True (e.g. _drop_glom_with_plus=False)
            assert panel in ('megamat', 'validation2'), ('repro_remy_paper=True '
                'probably does not make sense for use on new data'
            )


        for model_kws in model_kw_list:

            model_kws = dict(model_kws)

            # TODO only do if not already in model_kws
            model_kws['target_sparsity'] = target_sparsity

            model_kws['comparison_kc_corrs'] = comparison_kc_corrs

            if repro_remy_paper and panel == 'megamat':
                model_kws['repro_preprint_s1d'] = True

            do_sensitivity_analysis = False
            if model_kws.get('sensitivity_analysis', False):
                do_sensitivity_analysis = True

            if skip_sensitivity_analysis or not do_sensitivity_analysis:
                try:
                    # assumes the default is False. could also set False explicitly,
                    # but not passing that in explicitly for other model_kws
                    # iterated over.
                    del model_kws['sensitivity_analysis']
                except KeyError:
                    pass

            else:
                # TODO TODO also skip anything other than 1 or 2 hemibrain calls for
                # these panels? move calls to natmix_data/analysis.py itself?
                if panel in ('kiwi', 'control'):
                    # TODO also check whether sensitivity_analysis=True? or
                    # sens_analysis_kws just ignored anyway, if it's False(/which i
                    # assume is default?)?
                    warn('using diff sens_analysis_kws for kiwi/control panels:\n' +
                        pformat(natmix_sens_analysis_kws)
                    )
                    model_kws['sens_analysis_kws'] = natmix_sens_analysis_kws

            # used to have this like 'dff_scale-<scaling-method>', for pebbled
            # input, but didn't like how it cluttered up dir names (since almost all
            # of them have this part of their name, and I only ever scale my pebbled
            # data one way these days, so it was always 'dff_scale-to-avg-max__')
            param_dir_prefix = ''

            # TODO (delete?) is this loop working as expected? in run on
            # kiwi+control data, i feel like i've seen more progress bars than i
            # expected...
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

                tuning_model_kws = {k: v for k, v in model_kws.items()
                    if k not in (
                        'orn_deltas', 'comparison_kc_corrs', 'comparison_orns',

                        # NOTE: would need to move these APL boost params out of
                        # here, if I were to move boosting to pre-tuning
                        'multiresponder_APL_boost', '_multiresponder_mask',
                        'boost_wKCAPL', 'return_dynamics'
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

                # this call is to pre-tune model on specified panels, and then calls
                # below will run the current panel(s) through this pre-tuned model
                param_dict = fit_and_plot_mb_model(tuning_panels_plot_dir,
                    extra_params=extra_params,

                    # NOTE: this is intended to prevent any sensitivity analysis
                    # recursive calls from running, and also skips most/all plotting
                    # TODO still plot clustering (of spike_counts) across both
                    # panels, to get a sense of overlap in multi-responders across
                    # the two (if we want to tune APL within them, which would then
                    # affect both panels, if we were to do it here)
                    _only_return_params=True,

                    **tuning_model_kws
                )
                _write_inputs_for_reproducibility(tuning_panels_plot_dir,
                    param_dict
                )

                fixed_thr = param_dict['fixed_thr']

                # if we were still popping the list-of-float wAPLKC/wKCAPL in the
                # variable_n_claw=True cases (in fit_and_plot*), then would need to load
                # wAPLKC/wKCAPL from separate pickles here
                apl_params = get_APL_weights(param_dict, model_kws)
                wAPLKC = apl_params['wAPLKC']

                equalize_kc_type_sparsity = model_kws.get(
                    'equalize_kc_type_sparsity', False
                )
                if equalize_kc_type_sparsity:
                    # TODO TODO fixed_thr should be array now. just pass that
                    # directly instead of below? (wrote below when i was returning
                    # fixed_thr=None)

                    # TODO union [some of] this param_dict into one below? (e.g. for
                    # type2thr)

                    # this must be in param_dict if equalize_kc_type_sparsity=True
                    type2thr = param_dict['type2thr']

                    tuning_output_dir = (
                        tuning_panels_plot_dir / param_dict['output_dir']
                    )
                    # TODO assert this has been written since run start?
                    #
                    # using this to get the kc type for each cell (nothing in
                    # param_dict has it, and vector things in there tend to screw up
                    # some outputs, as currently implemented [e.g. the CSVs
                    # summarizing parameters for different runs])
                    wPNKC = pd.read_pickle(tuning_output_dir / 'wPNKC.p')

                    # TODO (delete/) refactor this part to share w/
                    # test_fixed_inh_params2 (copied this there)?
                    kc_types = wPNKC.index.get_level_values(KC_TYPE)
                    assert not kc_types.isna().any()
                    assert set(kc_types) == set(type2thr.keys())
                    cell_thrs = kc_types.map(type2thr)
                    fixed_thr = cell_thrs.values.copy()

                    del kc_types, cell_thrs, type2thr, wPNKC

                    # TODO replace recomputed fixed_thr w/ one from param_dict
                    # (-> delete code to recompute)
                    assert np.array_equal(fixed_thr, param_dict['fixed_thr'])
                else:
                    assert 'type2thr' not in param_dict

                assert fixed_thr is not None
                assert wAPLKC is not None

                # TODO share 'hemibrain' default w/ two other places this defined
                pn2kc_connections = model_kws.get('pn2kc_connections', 'hemibrain')
                if pn2kc_connections not in connectome_options:
                    # should be one list element per seed.
                    # all elements should be float (shouldn't need to support vector
                    # fixed_thr there).
                    assert type(fixed_thr) is list and type(wAPLKC) is list
                    assert len(fixed_thr) == len(wAPLKC)

                    # NOTE: currently relying on the pre-tuning + actual modelling
                    # calls using the same sequences of seeds (which they do, b/c
                    # initial seed currently hardcoded, and i always increment
                    # following seeds by one from there), so that applying the
                    # sequence of inh params across the two makes sense

                model_kws['title_prefix'] = f'tuning panels: {tuning_panels_str}\n'

                _extra_params['tuning_panels'] = tuning_panels_str
                _extra_params['tuning_output_dir'] = param_dict['output_dir']

                tuning_param_str = ''

                target_sp = model_kws.get('target_sparsity')
                del model_kws['target_sparsity']
                if target_sparsity is not None:
                    # TODO skip if it's some default value (0.1?)?
                    # TODO refactor formatting?
                    tuning_param_str += f'_target-sp_{target_sp:.3g}'

                target_sparsity_factor_pre_APL = model_kws.get(
                    'target_sparsity_factor_pre_APL'
                )
                if target_sparsity_factor_pre_APL is not None:
                    tuning_param_str += (
                        '_target-sp-factor-pre-APL_'
                        f'{target_sparsity_factor_pre_APL:.1f}'
                    )

                homeostatic_thrs = model_kws.get('homeostatic_thrs', False)
                if homeostatic_thrs:
                    tuning_param_str += '_hstatic-thrs_True'
                    del model_kws['homeostatic_thrs']
                    # TODO also assert fixed_thr is a vector that varies across the
                    # cells?

                if equalize_kc_type_sparsity:
                    tuning_param_str += '_equalize-types_True'

                    # need to remove equalize_kc_type_sparsity=True from calls below
                    # (w/ fixed fixed_thr and wAPLKC), since it operates by tuning
                    # threshold within each KC type (which we already did above.
                    # wouldn't make sense to do again).
                    del model_kws['equalize_kc_type_sparsity']

                    ab_prime_response_rate_target = model_kws.get(
                        'ab_prime_response_rate_target'
                    )
                    if ab_prime_response_rate_target is not None:
                        tuning_param_str += (
                            f'_ab-prime_{ab_prime_response_rate_target:.3g}'
                        )
                        # same reason we are removing this one. it's only used
                        # inside the equalize_kc_type_sparsity=True path.
                        del model_kws['ab_prime_response_rate_target']

                del equalize_kc_type_sparsity

                # TODO TODO variable_n_claws cases actually supported here? should
                # fixed_thr / wAPLKC (or means) not also be appending to model output
                # dir names? (not currently for uniform n_claws=7 outputs generated
                # 2025-09-10) (if not, care to support?)

                # TODO any other params that influence tuning?

                tuning_prefix = f'tuned-on_{tuning_panels_str}{tuning_param_str}__'
                param_dir_prefix = f'{tuning_prefix}{param_dir_prefix}'

            params_for_csv = fit_and_plot_mb_model(panel_plot_dir,
                param_dir_prefix=param_dir_prefix, extra_params=_extra_params,
                fixed_thr=fixed_thr, wAPLKC=wAPLKC, **model_kws
            )
            _write_inputs_for_reproducibility(panel_plot_dir, params_for_csv)

            # should only be the case if first_seed_only=True inside fit_and_plot...
            # (just used to regen model internal plots, which are made on first seed
            # for multi-seed runs. no downstream plots/caches are updated by
            # fit_and_plot... in that case).
            if params_for_csv is None:
                continue

            if first_model_kws_only:
                # TODO also say we aren't writing CSV here too? (or don't say in
                # similar conditional below?)
                warn('breaking after first model_kw_list entry, because '
                    f'{first_model_kws_only=}!'
                )
                break

            if skip_models_with_seeds:
                warn(f'not writing to {model_param_csv} (b/c '
                    'skip_models_with_seeds=True)!'
                )
                continue

            # should only be added if wAPLKC/fixed_thr passed, which should not
            # be the case in any of these calls
            assert 'pearson' not in params_for_csv

            params_for_csv = filter_params_for_csv(params_for_csv)

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

        # TODO TODO how much of stuff below should i factor out of model_mb_responses
        # (into something al_analysis calls after model_mb_responses is done. would
        # probably want to return something from this fn)?

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

        # TODO comment explaining what all is done in this loop (/ below)?
        for desc, mask in (('pebbled', pebbled_mask), ('hallem', ~ pebbled_mask)):

            if mask.sum() == 0:
                warn(f'no {desc} data in current model runs. skipping 2E/S1C/etc!')
                continue

            # one row per model run
            curr_model_params = model_params.loc[mask]

            curr_model_params = curr_model_params.sort_values('pn2kc_connections',
                kind='stable', key=lambda x: x.map(_sort_pn2kc)
            )

            try:
                # TODO just drop_duplicates to keep first row for each, and warn we are
                # doing that instead?
                #
                # since we'll use this for line labels (e.g. 'hemibrain', 'uniform')
                assert not curr_model_params.pn2kc_connections.duplicated().any()
            except AssertionError:
                warn(f'duplicate pn2kc_connections values across {desc} models! '
                    'skipping 2E/S1C/etc!\n\ncomment/remove model_kw_list values to '
                    'remove these duplicates, to generate skipped plots.'
                )
                continue

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

            # TODO comment explaining what all is done in this loop (/ below)?
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
                    assert len(corrs) == N_SEEDS
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
                # TODO need to check if we actually have odor2 (or diff meaning here?)?
                #
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
                    assert n_odors == len(megamat_odor_names) == len(model_odors)

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

                assert (
                    len(responses_including_silent.columns) == len(megamat_odor_names)
                )
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

                # compare mean-ORN / mean-KC / model Pearson's correlations, using
                # Spearman's correlation
                for method in ('spearman', 'pearson'):
                    corr_of_pearsons = merged_corrs.corr(method=method)
                    xlabel = f"{method.title()}s-of-Pearsons"

                    output_name_without_ci = f'{method}_of_pearsons'

                    plot_corr(corr_of_pearsons, panel_plot_dir, output_name_without_ci,
                        overlay_values=True, xlabel=xlabel
                    )

                    for ci in (90, 95):
                        ci_str = f'{ci:.0f}'
                        ci_title_str = f'{ci_str}% CI'
                        # TODO do something other than average over 100 seeds?

                        output_name = f'{output_name_without_ci}_ci{ci_str}'

                        corrs = []
                        lower_cis = []
                        upper_cis = []
                        pvals = []
                        for x, y in col_pairs:
                            # TODO add one extra sigfig (in spear_text, the first
                            # returned arg from bootstrapped_corr) for the correlation
                            # and CI part (from 2 sigfigs -> 3)
                            _, corr, ci_lower, ci_upper, pval = bootstrapped_corr(
                                merged_corrs, x, y, method=method, ci=ci
                            )
                            corrs.append(corr)
                            lower_cis.append(ci_lower)
                            upper_cis.append(ci_upper)
                            pvals.append(pval)

                        corr_df = pd.DataFrame({
                                'corr': corrs, 'lower': lower_cis, 'upper': upper_cis,
                                'pval': pvals
                            }, index=index
                        )
                        to_csv(corr_df, panel_plot_dir / f'{output_name}.csv')

                        square_corr = invert_corr_triangular(corr_df['corr'], name=None)
                        assert pd_allclose(square_corr, corr_of_pearsons)

                        # are CI's symmetric (no) (i.e. can i get one measure of error
                        # for each matrix element, or will it make more sense to have 2
                        # extra matrix plots, one for lower CI and one for upper CI?)
                        square_lower = invert_corr_triangular(corr_df['lower'],
                            name=None
                        )
                        square_upper = invert_corr_triangular(corr_df['upper'],
                            name=None
                        )

                        plot_corr(square_lower, panel_plot_dir,
                            f'{method}_of_pearsons_lower_ci{ci_str}',
                            overlay_values=True,
                            xlabel=f'{xlabel}\nlower side of {ci_title_str}'
                        )
                        plot_corr(square_upper, panel_plot_dir,
                            f'{method}_of_pearsons_upper_ci{ci_str}',
                            overlay_values=True,
                            xlabel=f'{xlabel}\nupper side of {ci_title_str}'
                        )

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


# TODO try to move all fns below to al_analysis wrapping model_mb_responses output?
# (or otherwise separate out a simplest-possible version of running model code, that
# doesn't involve all the plotting for paper w/ remy)
# (or at least move to al_util, to declutter this file?)

n_megamat_odors = 17
assert len(megamat_odor_names) == n_megamat_odors

# TODO put in docstring which files we are loading from
def _load_remy_megamat_kc_responses(drop_nonmegamat: bool = True, drop_pfo: bool = True
    ) -> pd.DataFrame:

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

    verbose = al_util.verbose
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

    names = np.array([x['name'] for x in odors])
    assert megamat_odor_names - set(names) == set(), 'missing some megamat odors'

    # cast_int_concs=True to convert '-3.0' to '-3', to be consistent w/ mine
    odor_strs = [olf.format_odor(x, cast_int_concs=True) for x in odors]
    mean_responses.columns = odor_strs

    # so al_util.mean_of_fly_corrs works
    mean_responses.columns.name = 'odor1'

    # when also loading old (prior to final 4 flies) megamat data:
    # (set(names) - megamat_odor_names)={'PropAc', '1p3ol', 'pfo', 'g-6lac', 'eug',
    # 'd-dlac', 'MethOct', 'ECin'}
    if drop_nonmegamat:
        megamat_mask = [x in megamat_odor_names for x in names]
        megamat_mean_responses = mean_responses.loc[:, megamat_mask]

        if verbose:
            nonmegamat_odors = mean_responses.columns.difference(
                megamat_mean_responses.columns
            )
            warn('dropping the following non-megamat odors:\n'
                f'{pformat(list(nonmegamat_odors))}'
            )

        mean_responses = megamat_mean_responses

    # elif because we will have already dropped pfo if drop_nonmegamat=True
    # (pfo is not considered part of megamat)
    elif drop_pfo:
        pfo_mask = names == 'pfo'
        assert pfo_mask.sum() > 0
        mean_responses = mean_responses.loc[:, ~pfo_mask]

    return mean_responses


# TODO rename all of these fns to remove '_megamat' (unless i actually drop down to just
# megamat, but i don't think i want that?)? or just do it anyway to shorten these names?
def _remy_megamat_flymean_kc_corrs(ordered_pairs=None, **kwargs) -> pd.DataFrame:
    mean_responses = _load_remy_megamat_kc_responses(**kwargs)

    # TODO move some functionality like this into al_util.mean_of_fly_corrs (to average within
    # fly across recordings first)?
    recording_corrs = mean_responses.groupby(level=['datefly', 'thorimage'], sort=False
        ).apply(lambda x: corr_triangular(x.corr(), ordered_pairs=ordered_pairs))

    fly_corrs = recording_corrs.groupby(level='datefly', sort=False).mean()

    checks = True
    if checks:
        old_megamat_root = remy_data_dir / '2024-11-12'
        # TODO also load + check against stim_rdms__iscell_good_xid0__correlation.xlsx?
        # (it should have been generated from this .nc, w/ remy providing a script to
        # generate this xlsx file from it, but still...)
        corr_file_for_anoop = (
            old_megamat_root / 'xrda_stim_rdm_concat__iscell_good_xid0__correlation.nc'
        )

        # adapted from the example script Remy emailed (2024-11-07) Anoop alongside this
        # data (demo_megamat_new_and_old_by_fly_17_kc_soma_nls.py, from OdorSpaceShare
        # repo)
        da_stim_rdm_concat = xr.load_dataarray(corr_file_for_anoop)

        # otherwise .to_index() call below doesn't give me what i want
        da_stim_rdm_concat = da_stim_rdm_concat.set_index({
            'acq': remy_fly_cols + ['thorimage_name']
        })

        #            date_imaged  fly_num              thorimage_name
        # 0   2022-10-10        1                    megamat0
        # 1   2022-10-10        2                    megamat0
        # 2   2022-10-11        1                    megamat0
        # 3   2022-11-10        1            megamat0__dsub03
        # 4   2018-10-21        1         _002+_003+_004+_005
        # 5   2019-03-06        3                        _002
        # 6   2019-03-06        4                        _003
        # 7   2019-03-07        2                  _003+_0057
        # 8   2019-04-26        4                     fn_0002
        # 9   2019-05-09        4             fn_0001+fn_0003
        # 10  2019-05-09        5             fn_0001+fn_0002
        # 11  2019-05-23        2                     fn_0001
        # 12  2019-05-24        1                     fn_0003
        # 13  2019-05-24        3                     fn_0001
        # 14  2019-05-24        4             fn_0001+fn_0002
        # 15  2019-07-19        2  movie001+movie002+movie003
        # 16  2019-09-12        1             fn_0002+fn_0003
        # 17  2019-09-12        2             fn_0001+fn_0002
        fly_metadata = da_stim_rdm_concat.coords['acq'].to_index().to_frame(index=False)
        # we already have combined data across recordings (for flies that have multiple.
        # see rows w/ '+' in thorimage_name in comment above)
        assert len(fly_metadata) == len(fly_metadata[remy_fly_cols].drop_duplicates())

        assert fly_corrs.index.name == 'datefly'
        recalced_flies = set(fly_corrs.index)

        datefly_strs = fly_metadata[remy_fly_cols].astype(str).agg('/'.join, axis=1)
        flies_for_anoop = set(datefly_strs)
        assert recalced_flies == flies_for_anoop

        da_stim_rdm_concat = da_stim_rdm_concat.assign_coords(
            {'datefly': ('acq', datefly_strs)}).set_index({'acq': 'datefly'}
        )

        fly_corrs_has_dropped_non_megamat = kwargs.get('drop_nonmegamat', True)

        if fly_corrs_has_dropped_non_megamat:
            # (to compare to the single fly corrs in the .nc file Remy gave Anoop in
            # November 2024, which also included old megamat data, in addition to the
            # final 4 flies we had been using)
            corrs_to_compare_to_anoop_data = fly_corrs
        else:
            megamat_pairs = fly_corrs.columns.to_frame().applymap(odor_is_megamat
                ).all(axis='columns')
            corrs_to_compare_to_anoop_data = fly_corrs.loc[:, megamat_pairs]

            n_megamat_only_pairs = n_choose_2(n_megamat_odors)
            assert len(corrs_to_compare_to_anoop_data.columns) == n_megamat_only_pairs


        for datefly in corrs_to_compare_to_anoop_data.index:
            fly_corr = corrs_to_compare_to_anoop_data.loc[datefly]
            fly_corr = fly_corr.dropna()

            # need to go from '1-5ol @ -3.0' format row/col index odors have here,
            # to '1-5ol @ -3' as in fly_corr index
            fly_da_stim_rdm = da_stim_rdm_concat.sel(acq=datefly).to_pandas()

            odor_strs = [
                # TODO refactor this parsing -> formatting, to a fn just for normalizing
                # repr of conc part of str (to int)?
                olf.format_odor(olf.parse_odor(x), cast_int_concs=True)
                for x in fly_da_stim_rdm.index
            ]
            assert fly_da_stim_rdm.columns.equals(fly_da_stim_rdm.index)
            fly_da_stim_rdm.index = odor_strs
            fly_da_stim_rdm.columns = odor_strs

            # TODO delete if i remove assertion in corr_triangular that index/columns
            # names have to start with 'odor'
            fly_da_stim_rdm.index.name = 'odor'
            fly_da_stim_rdm.columns.name = 'odor'

            anoop_fly_corr = corr_triangular(1 - fly_da_stim_rdm,
                # TODO TODO do we actually need this tho? or was it the other changes
                # that made a diff?
                ordered_pairs=fly_corr.index
            )

            # TODO work? (comment what intention / effect is?)
            anoop_fly_corr = anoop_fly_corr.dropna()

            # seems we don't need to pass particular pairs into corr_triangular
            assert pd_allclose(fly_corr, anoop_fly_corr, equal_nan=True)

    # TODO delete
    # TODO check above equiv to this, at least if we no longer load old data?
    # TODO check this works if there are multiple thorimage level values (i.e.
    # recordings) for one pair (e.g. 1-6ol, 2-but) for any fly. should average the corrs
    # first, then compute average across flies.
    #mean_corr = al_util.mean_of_fly_corrs(mean_responses.T, id_cols=['datefly'])

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

    sns.pointplot(data=df, order=pair_order, linestyle='none', ax=g.ax,
        **_fig2e_shared_plot_kws, **kwargs, **marker_kws
    )


# TODO move this (and related) to mb_model.py?
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
    _repo_root = Path.home() / 'src/OdorSpaceShare'
    assert _repo_root.is_dir()

    preprint_data_folder = _repo_root / 'preprint/data/figure-02/02e'

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
        data_folder = _repo_root / 'manuscript/data/figure-02/02e'
        # df_obs.csv in the same folder was one of her earlier attempts to get me a
        # newer version of this data, but was not completely consistent w/ format of old
        # CSV (and also had 'correlation' col that was actually correlation distance).
        # df_obs.csv should not be used.
        csv_name = 'df_obs_for_tom.csv'

    assert data_folder.is_dir()

    csv_path = data_folder.joinpath(csv_name)
    if al_util.verbose:
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

    # TODO inspect with remy
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
        # TODO refactor w/ place copied from in model_mb...?
        remy_pairs = set(list(zip(df_obs.abbrev_row, df_obs.abbrev_col)))

        # TODO does it actually matter? does df_obs have all the corrs i would compute
        # for old flies anyway? maybe just expand checks below to also check those
        # flies?
        #
        # TODO TODO can i switch things to using corrs from
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

        # TODO move this dropna into above fn? this even doing anything? why would a
        # column be all NaN (and is that the right interpretation of axis='columns'?)?
        flymean_corrs = flymean_corrs.dropna(how='all', axis='columns')
        corrs = flymean_corrs

        n_megamat_only_pairs = n_choose_2(n_megamat_odors)
        # TODO delete? already relaxed from == to >=
        assert len(corrs.columns) >= n_megamat_only_pairs

        for datefly in corrs.index:
            fly_df = df_obs[df_obs.datefly == datefly]

            remy_2e_csv_ser = fly_df[['abbrev_row', 'abbrev_col',
                'correlation_distance']].set_index(['abbrev_row', 'abbrev_col'])

            # just to convert from shape (n, 1) to (n,)
            remy_2e_csv_ser = remy_2e_csv_ser.iloc[:, 0]

            remy_2e_csv_ser.index.names = ['odor1', 'odor2']

            # convert from correlation distance to correlation (to match what we have in
            # corrs)
            remy_2e_csv_ser = 1 - remy_2e_csv_ser
            remy_2e_csv_ser.name = 'correlation'

            recalced_ser = corrs.loc[datefly]

            # since corrs is of shape (<n_flies>, <n_total_odor_pairs>), this will drop
            # the pairs down to those actually measured in this fly
            recalced_ser = recalced_ser.dropna()
            assert not remy_2e_csv_ser.isna().any()

            recalced_pair_set = set(recalced_ser.index)
            # neither index should have any duplicate pairs
            assert len(recalced_pair_set) == len(recalced_ser)
            assert len(recalced_pair_set) == len(remy_2e_csv_ser)

            assert recalced_pair_set == set(remy_2e_csv_ser.index)

            # above assertion justifies indexing one by the other, as it's just the
            # order that is different, not that either series has any different pairs
            assert pd_allclose(recalced_ser, remy_2e_csv_ser.loc[recalced_ser.index])

        df_megamat = df_obs[
            df_obs.abbrev_row.isin(megamat_odor_names) &
            df_obs.abbrev_col.isin(megamat_odor_names)
        ]

        # TODO (delete? satisfied?) are all of the old megamat flies that i'm now
        # supposed to use a subset of these? do the correlations match what i would
        # compute?
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
            # TODO elaborate on why?
            warn('could not plot 2e square matrix corr plots')
    #

    return df_obs


def main():
    # TODO print names of plots we are saving (by default, and prob unconditionally),
    # as if -v/--verbose were passed to al_analysis.py

    # TODO refactor to share loading of this megamat orn_deltas w/ tests that do the
    # same (move to fn in this module)
    paper_hemibrain_model_output_dir = Path('data/sent_to_remy/2025-03-18/'
        'dff_scale-to-avg-max__data_pebbled__hallem-tune_False__pn2kc_hemibrain__'
        'weight-divisor_20__drop-plusgloms_False__target-sp_0.0915'
    ).resolve()

    # TODO TODO commit + use kiwi/control data for one? (could just temporarily change
    # this path to one for kiwi/control data) (might be more complicated, since i
    # typically tune on both panels there, but i could skip that?)
    #
    # panel          megamat   ...
    # odor           2h @ -3   ...   benz @ -3    ms @ -3
    # glomerulus               ...
    # D            40.363954   ...   42.445274  41.550370
    # DA2          15.144943   ...   12.363544   3.856004
    # ...
    # VM7d        108.535394   ...   58.686294  20.230297
    # VM7v         59.896953   ...   13.250292   8.446418
    # orn_deltas = pd.read_csv(model_output_dir1 / 'orn_deltas.csv', header=[0,1],
    #     index_col=0
    # )

    # df = pd.read_csv('mean_est_spike_deltas.csv', header=[0, 1], index_col=0)
    # orn_deltas = df.loc[:, df.columns.get_level_values('panel') == 'control']
    orn_deltas = pd.read_csv(paper_hemibrain_model_output_dir / 'orn_deltas.csv',
        header=[0,1], index_col=0
    )

    assert orn_deltas.columns.names == ['panel', 'odor']
    assert orn_deltas.index.names == [glomerulus_col]

    # currently only way to enable olfsysm log prints
    al_util.verbose = True

    plot_root = Path('model_mb_example').resolve()

    kws = dict()
    # TODO delete this product thing, and just switch to a kws_list?
    # TODO define some of this w/ dict_seq_product?
    try_each_with_kws = [
        dict(one_row_per_claw=True, APL_coup_const=0),
        # NOTE: APL_coup_const=0 should enable different activity in different APL
        # compartments (but without any coupling between them)
        # TODO give it a better name!
        # TODO TODO check these outputs are actually diff from w/ default
        # APL_coup_const=None, where there should only be one APL (no compartments)
        # (add test for that? already exist? test we have activity timecourses for each
        # compartment separately too?)
        # TODO try multiple radii? (need to expose as kwarg) (may be bigger priority to
        # try other compartmentalizations, whether that's a grid or something else)
        dict(prat_claws=True, one_row_per_claw=True, APL_coup_const=0),

        # TODO TODO (done?) drop multiglomerular PNs and re-run all prat_claws=True
        # variants (hardcode inside that branch of connectome_wPNKC. never want them)
        # TODO add test comparing glomeruli set wd20 vs tianpei vs prat_claws?
        # (and explaining any differences in comments) (assert all in task? some
        # exceptions?)

        dict(prat_claws=True, one_row_per_claw=True,
            use_connectome_APL_weights=True
        ),
        dict(prat_claws=True, one_row_per_claw=True),

        # TODO delete? both of these seemed to just make main issues w/ model worse
        # TODO TODO try these again after scaling wPNKC within each KC (to have same
        # mean as before?) (have i already? would this really matter?)?
        dict(prat_claws=True, one_row_per_claw=True, dist_weight='percentile'),
        dict(prat_claws=True, one_row_per_claw=True, dist_weight='raw'),
        #

        dict(one_row_per_claw=True),
        dict(one_row_per_claw=True, use_connectome_APL_weights=True),

        dict(weight_divisor=20),
        dict(weight_divisor=20, use_connectome_APL_weights=True),
    ]

    for extra_kws in try_each_with_kws:
        # extra_kws will override kws without warning, if they have common keys
        param_dict = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
            try_cache=False,
            # TODO disable plot_example_dynamics (resource intensive)?
            #plot_example_dynamics=True,
            **{**kws, **extra_kws}
        )
        output_dir = (plot_root / param_dict['output_dir']).resolve()
        assert output_dir.is_dir()
        assert output_dir.parent == plot_root

        #           2h @ -3  IaA @ -3  pa @ -3  ...  1-6ol @ -3  benz @ -3  ms @ -3
        # kc_id         ...
        # 0             0.0       0.0      0.0  ...         0.0        0.0      0.0
        # 1             0.0       0.0      0.0  ...             0.0        0.0      0.0
        # ...           ...       ...      ...  ...         ...        ...      ...
        # 1835          1.0       1.0      2.0  ...         1.0        0.0      0.0
        # 1836          0.0       0.0      0.0  ...         0.0        0.0      0.0
        #
        # NOTE: would probably need to remove KC_TYPE level if using e.g.
        # pn2kc_connections='uniform' (where we are not using hemibrain data, and thus
        # do not have subtypes per cell)
        df = pd.read_csv(output_dir / 'spike_counts.csv', index_col=[KC_ID, KC_TYPE])

    # TODO TODO also include an example w/ kiwi/control data (as used by natmix_data)
    # (either committing data in al_analysis too, or moving this whole example to
    # another repo)
    # TODO + add test i can reproduce those outputs (use outputs i already sent someone
    # from my kiwi/control data? sent to ruoyi? or someone else?)
    # TODO TODO add tests in test_al_analysis to test i can repro other outputs too (or
    # at least characterize extent of current mismatch, with an explanation as to why
    # each difference exists)

    # TODO also include equalize_kc_type_sparsity=False for all below
    # TODO TODO pick from kwargs described in comments below
    #
    # TODO why do gamma KCs have a LOWER than average response rate after this?
    #
    # use_connectome_APL_weights=True
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=None
    # retune_apl_post_equalized_thrs=False
    # mean response rate: 0.1158. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.124475            336
    # ab                0.121534            802
    # g                 0.092561            612
    # unknown           0.200000             80
    #
    # use_connectome_APL_weights=True
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=0.2
    # retune_apl_post_equalized_thrs=False
    # mean response rate: 0.1377. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.294293            336
    # ab                0.111706            802
    # g                 0.077566            612
    # unknown           0.200000             80
    #
    # use_connectome_APL_weights=True
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=0.15
    # retune_apl_post_equalized_thrs=False
    # mean response rate: 0.1279. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.212885            336
    # ab                0.117207            802
    # g                 0.085736            612
    # unknown           0.200 000             80
    #
    #
    # use_connectome_APL_weights=False
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=None
    # retune_apl_post_equalized_thrs=False
    # mean response rate: 0.0951. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.101716            336
    # ab                0.087355            802
    # g                 0.099097            612
    # unknown           0.114706             80
    #
    # use_connectome_APL_weights=False
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=0.2
    # retune_apl_post_equalized_thrs=False
    # mean response rate: 0.1024. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.239146            336
    # ab                0.065645            802
    # g                 0.077278            612
    # unknown           0.089706             80
    #
    # use_connectome_APL_weights=False
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=0.15
    # retune_apl_post_equalized_thrs=False
    # mean response rate: 0.1004. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.174020            336
    # ab                0.077527            802
    # g                 0.089869            612
    # unknown           0.101471             80
    #
    #
    # use_connectome_APL_weights=False
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=None
    # retune_apl_post_equalized_thrs=True
    # mean response rate: 0.1096. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.113270            336
    # ab                0.103125            802
    # g                 0.114379            612
    # unknown           0.123529             80
    #
    # use_connectome_APL_weights=False
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=0.2
    # retune_apl_post_equalized_thrs=True
    # mean response rate: 0.0930. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.223389            336
    # ab                0.058163            802
    # g                 0.068627            612
    # unknown           0.080147             80
    #
    # use_connectome_APL_weights=False
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=0.15
    # retune_apl_post_equalized_thrs=True
    # mean response rate: 0.0961. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.168242            336
    # ab                0.074153            802
    # g                 0.085160            612
    # unknown           0.097794             80
    #
    #
    # use_connectome_APL_weights=True
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=None
    # retune_apl_post_equalized_thrs=True
    # mean response rate: 0.0996. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.109769            336
    # ab                0.105252            802
    # g                 0.073626            612
    # unknown           0.200000             80
    #
    # use_connectome_APL_weights=True
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=0.2
    # retune_apl_post_equalized_thrs=True
    # mean response rate: 0.0984. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.241772            336
    # ab                0.071586            802
    # g                 0.041619            612
    # unknown           0.200000             80
    #
    # use_connectome_APL_weights=True
    # equalize_kc_type_sparsity=True
    # ab_prime_response_rate_target=0.15
    # retune_apl_post_equalized_thrs=True
    # mean response rate: 0.0996. by KC type:
    #          avg_response_rate  n_kcs_of_type
    # a'b'              0.178046            336
    # ab                0.087575            802
    # g                 0.059208            612
    # unknown           0.200000             80

    kws = dict(
        weight_divisor=20,
        # TODO just make this default?
        use_connectome_APL_weights=True,
    )

    # TODO TODO try dropping all kc_type == 'unknown' cells before running
    # (in fit_mb_model)?

    plot_root = Path('model_mb_example').resolve()
    # updated to keep things clean;
    #plot_root = Path("model_mb_example") / f"data_pebbled_target-sp_{target_sparsity:.4f}"

    # TODO modify this fn so dirname includes all same params by default (rather than
    # just e.g. param_dir='data_pebbled'), as the ones i'm currently manually creating
    # by calls in model_mb_... (prob behaving diff b/c e.g.
    # pn2kc_connections='hemibrain' is explicitly passed there)
    # param_dict = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
    #    try_cache=False, print_olfsysm_log = True, **kws
    # )

    # TODO delete this product thing, and just switch to a kws_list?
    '''
    try_each_with_kws = [
        # TODO TODO (still relevant?) make sure output dirs have something in name
        # for use_vector_thr=True case (i.e. case where fixed_thr vector here)
        # (just pass via suffix? or extra params? might need to restore the code for
        # that...) (param_dir_prefix? still also add something to extra params to
        # include in plot titles hopefully (or those not used that way?)?)
        # TODO maybe mean response rate per subtype? or thr for each?
        # TODO TODO at least include thr multiplier for each as a parameter?
        # (and maybe include in titles?)
        #
        # probably always want `kws` unmodified too. that's what this empty dict is for.
        dict(),

        dict(use_connectome_APL_weights=True),
    ]
    '''
    #   APL_coup_const = 0.8,
    # APL_coup_const = [0.5, 0.5],
    # APL_coup_const = 0.8,
    APL_coup_const = 0.00
    Btn_separate = True
    pn_claw_to_APL = True
    try_each_with_kws = [
        dict(one_row_per_claw = True, pn_claw_to_APL = True, APL_coup_const = 0.00, Btn_separate = False)
    ]

    for i, kws in enumerate(try_each_with_kws):
        row_per_claw   = bool(kws.get('one_row_per_claw'))
        pn_claw_to_APL = bool(kws.get('pn_claw_to_APL'))

        if pn_claw_to_APL and not row_per_claw:
            raise ValueError(
                f"Config #{i}: pn_claw_to_APL=True requires one_row_per_claw=True."
            )

    for extra_kws in try_each_with_kws:
        # extra_kws will override kws without warning, if they have common keys
        param_dict = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
            # TODO disable plot_example_dynamics (resource intensive)?
            try_cache=False, plot_example_dynamics=True, **{**kws, **extra_kws}
        )
        output_dir = (plot_root / param_dict['output_dir']).resolve()
        assert output_dir.is_dir()
        assert output_dir.parent == plot_root

        #           2h @ -3  IaA @ -3  pa @ -3  ...  1-6ol @ -3  benz @ -3  ms @ -3
        # kc_id         ...
        # 0             0.0       0.0      0.0  ...         0.0        0.0      0.0
        # 1             0.0       0.0      0.0  ...             0.0        0.0      0.0
        # ...           ...       ...      ...  ...         ...        ...      ...
        # 1835          1.0       1.0      2.0  ...         1.0        0.0      0.0
        # 1836          0.0       0.0      0.0  ...         0.0        0.0      0.0
        df = pd.read_csv(output_dir / 'spike_counts.csv', index_col=KC_ID)

    # TODO TODO also include an example w/ kiwi/control data (as used by natmix_data)
    # (either committing data in al_analysis too, or moving this whole example to
    # another repo)
    # TODO + add test i can reproduce those outputs (use outputs i already sent someone
    # from my kiwi/control data? sent to ruoyi? or someone else?)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

