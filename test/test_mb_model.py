#!/usr/bin/env python3

from itertools import product
import os
from pathlib import Path
from pprint import pformat
# TODO delete?
import traceback
#
from typing import Hashable, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# TODO does importing this before hong2p.equals allow that warning (optional from within
# `equals`) to be hooked in same way (from `al_util` module-level warning format
# handling change)? import order matter? (delete if not)
import al_util
#
from hong2p.util import pd_allclose, equals
from hong2p.xarray import coords_equal, series2xarray_like, move_all_coords_to_index
import olfsysm as osm

from al_util import warn, diag_panel_str, fly_cols, load_natmix_dff, data_root
from mb_model import (fit_mb_model, fit_and_plot_mb_model, connectome_wPNKC,
    connectome_APL_weights, KC_ID, CLAW_ID, BOUTON_ID, KC_TYPE, step_around,
    read_param_csv, read_params, read_tuned_params, get_thr_and_APL_weights,
    variable_n_claw_options, dict_seq_product, get_connectome_wPNKC_params,
    format_model_params, eval_and_check_compatible, glomerulus_col, ParamDict,
    format_weights, megamat_orn_deltas, paper_megamat_orn_deltas,
    paper_hemibrain_output_dir, get_dynamics, get_time_index,
    fit_dff2spiking_from_remypaper_flies_and_hallem,
    scale_dff_to_est_spike_deltas_using_hallem, remypaper_dff2spiking_data_dir,
    written_since_proc_start, dff_to_spiking_model_choices_csv_name,
    dff_to_spiking_data_csv_name, read_parquet, MODEL_KW_LIST, QUICK_MODEL_KW_LIST,
    BOUTON_MODEL_KW_LIST, get_fitandplot_model_kw_list, model_mb_responses
)

# TODO better way?
from conftest import test_data_dir


# TODO test i can recreate committed megamat_orn_deltas() contents. prob need to recalc
# dF/F w/ old mean (n_volumes=2) response calc? or use old outputs, but check that
# was the response calc for them too?

# You can set these either 0/1 in prefix before pytest command.
#
# TODO do allow certain caches being used if another flag (or this?) is set? would
# need centrally somewhere though... want to skip some of the longer tuning cases
#
# currently this replaces [FITANDPLOT_]MODEL_KW_LIST w/ the QUICK* versions below,
# and disables some plotting, if True.
QUICK: bool = bool(int(os.environ.get('QUICK', False)))

# Can be slightly faster, and can avoid some of the main potential memory issues by
# skipping plotting code, since main issue is loading claw_sims into fit_mb_model, for
# plot_example_dynamics=True path.  If this is not set, will plot if QUICK=False (and
# not otherwise).
_plot = os.environ.get('PLOT')
if _plot is None:
    if QUICK:
        PLOT: bool = False
    else:
        PLOT: bool = True
else:
    PLOT: bool = bool(int(_plot))

if PLOT:
    # TODO keep return_dynamics=True here? (will also lead to them being saved, at least
    # temporarily)
    PLOT_KWS = dict(make_plots=True, plot_example_dynamics=True, return_dynamics=True)
else:
    # worth monkey patching savefig at least (or support None as plot_dir?)? or is
    # most of the time in the steps before savefig anyway?
    PLOT_KWS = dict(make_plots=False, plot_example_dynamics=False)

# TODO any way to get pytest to output captured warnings inline? --capture=no doesn't do
# it. that's why i currently have a hack in al_util.warn that also prints them to
# stdout, if it detects pytest
# NOTE: may eventually want to revert to per-test-fn marks (via
# `@pytest.mark.filterwarnings(...)` decorator), but many modelling wrapper calls will
# emit many warnings, hence the current module-level ignore
# TODO change to only ignore warnings generated from one of my files (or those in this
# repo)? easily possible?
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

# TODO TODO add test that none of options in MODEL_KW_LIST give same outputs?
# TODO test that relevant subset of these all have diff wPNKC (those w/ diff
# connectome_wPNKC args. could use the fn that collects those args to tell which subset
# to test?)

# TODO some mechanism in conftest.py to add CLI arg that can swap this in for all
# tests that use MODEL_KW_LIST (and same for corresponding two FITANDPLOT vars)?
# (-> delete this hack) (maybe i'm fine just using an env var for this? any issues?)
if QUICK:
    print()
    print('USING QUICK[_FITANDPLOT]_MODEL_KW_LIST (because QUICK=True)!!!')
    print()
    MODEL_KW_LIST = QUICK_MODEL_KW_LIST

if not PLOT:
    print()
    print('NOT TESTING PLOTTING CODE (because PLOT=False)!')
    print()
#

# TODO predefine both this and QUICK* version of it in mb_model? or just always call
# get_fitandplot_model_kw_list on one of [QUICK_]MODEL_KW_LIST?
N_TEST_SEEDS: int = 2
FITANDPLOT_MODEL_KW_LIST: List[ParamDict] = get_fitandplot_model_kw_list(MODEL_KW_LIST,
    N_TEST_SEEDS
)

def mark_kw_list_entries_xfail(model_kw_list: List[ParamDict]) -> List[ParamDict]:
    """Wraps appropriate entries in model_kw_list w/ pytest xfail wrapper
    """
    fitandplot_model_kw_list = []
    for kws in model_kw_list:
        if isinstance(kws, dict):
            pass
            # 'APL_coup_const' doesn't exist anymore. just leaving this as an example of
            # how to xfail something here based on parameters.
            #if 'APL_coup_const' in kws:
            #    assert kws['APL_coup_const'] == 0
            #    kws = pytest.param(kws, marks=pytest.mark.xfail(
            #        reason='APL_coup_const C++ code broken', run=False
            #    ))
        else:
            # TODO assert it's already a pytest.param (ParameterSet)? still add mark if
            # not already marked for xfail?
            pass
        fitandplot_model_kw_list.append(kws)
    return fitandplot_model_kw_list

FITANDPLOT_MODEL_KW_LIST = mark_kw_list_entries_xfail(FITANDPLOT_MODEL_KW_LIST)

# TODO set `al_util.verbose = True` for all these (at least, so long as that's only
# way to get olfsysm log output printed?) (or add a new way to configure that, and use
# that) (doing so for now)
al_util.verbose = True

# TODO move to model_mb? (or hong2p.types?)
FitMBModelOutputs = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ParamDict]

@pytest.fixture(scope='session')
# the name of this function is the name of the variable made accessible to test using
# this fixture. the name of the returned variable is not important.
#
# TODO rename to megamat_orn_deltas or something? (/ at least doc where this data came
# from)
# TODO add similar to load kiwi+control data (-> use [in tests that loop over
# MODEL_KW_LIST, in additional to curr megamat orn_deltas?])
# TODO do something to check output of this is never mutated (copy to module level, and
# check against that? too expensive?) (check on completion of all tests, that this
# fixture is still giving same value as if computed anew?)
def orn_deltas() -> pd.DataFrame:
    return megamat_orn_deltas()


# TODO cache outputs for same args+kwargs? can i do this as a fixture? or just some
# generic mechanism? (+ using indirect=True on parametrize calls, passing the args to
# this? that sufficient to cache?)
# maybe something that uses functools cache, but sorts args + kwargs and converts them
# all to hashable first? seems it requires them to be hashable, and order to be same
# for using functools cache w/ dict args: https://stackoverflow.com/questions/6358481
# prob more trouble than worth... (frozendict [3rd party] even work w/ DataFrame values,
# like orn_deltas? or can i assume orn_deltas will not change, and then exclude certain
# things?)
# TODO use format_model_params as key for caching these fns (w/in run, to share
# across tests, at least in many circumstances. may want to exclude certain tests from
# using the caches)?
def _fit_mb_model(*args, **kwargs) -> FitMBModelOutputs:
    # can i pytest.xfail APL-coup-const stuff from in here, or does it have to
    # be from within one of the test fns? yup, this works.
    #
    # stuff calling _fit_and_plot_mb_model should be handled by the xfail marks added by
    # get_fitandplot_model_kw_list. just leaving this as an example of how to xfail
    # within test.
    #if 'APL_coup_const' in kwargs:
    #    pytest.xfail('APL_coup_const C++ code broken')
    #

    # TODO move this prints into fit_mb_model, under a verbose flag?
    print('running fit_mb_model...', flush=True)
    # TODO still include **PLOT_KWS if we have plot_dir (non-None)?
    #ret = fit_mb_model(*args, **kwargs, **PLOT_KWS)
    ret = fit_mb_model(*args, **kwargs)
    print('done', flush=True)
    return ret


# TODO after finding pytest compatible caching mechanism (fixture?) for _fit_mb_model,
# also do that here?
def _fit_and_plot_mb_model(*args, **kwargs) -> ParamDict:
    return fit_and_plot_mb_model(*args, try_cache=False, **kwargs, **PLOT_KWS)


# TODO TODO why does test_fixed_inh_params_fitandplot[pn2kc_uniform__n-claws_7__n-seeds_2]
# seem to not have seed as a level of either [spike_counts|responses|wPNKC].csv?
# pickles also wrong?
# TODO TODO and why n_seeds not in params.csv for that case?
def _read_spike_counts(output_dir: Path, *, index_col=None) -> pd.DataFrame:

    retry_index_col = False
    if index_col is None:
        index_col = [KC_ID, KC_TYPE]
        retry_index_col = True

    spike_counts_csv = output_dir / 'spike_counts.csv'

    try:
        return pd.read_csv(spike_counts_csv, index_col=index_col)

    except ValueError as err:
        msg = str(err)
        if msg != 'Index kc_type invalid' or not retry_index_col:
            raise err

    # TODO factor this def out somewhere?
    # default levels for variable_n_claw outputs
    index_col = ['seed', KC_ID]
    return pd.read_csv(spike_counts_csv, index_col=index_col)


# TODO move to hong2p.util?
def diff_sets(a: Iterable[Hashable], b: Iterable[Hashable]) -> str:
    """Reports difference bewetween two sets (or similar, like dict.keys())
    """
    sa = set(a)
    sb = set(b)
    if sa == sb:
        return ''
    return f'a - b: {pformat(sa - sb)}\nb - a: {pformat(sb - sa)}'


# TODO move to hong2p.util?
# TODO use/delete
def diff_dicts(a: Mapping, b: Mapping) -> str:
    """Reports difference bewetween two dicts (or other Mappings)
    """
    msg = diff_sets(a, b)
    for k, v in a.items():
        if k not in b:
            continue

        v2 = b[k]
        # TODO implement more complicated logic than just `==` here, refactoring to
        # share code switching between that and .equals / np.array_equal / pd_allclose /
        # etc (w/ some existing code in this file and prob elsewhere)
        if v != v2:
            msg += f'{k}: {v} != {v2}\n'

    return msg


def is_tuning_output_param(x: str) -> bool:
    return (
        # for wAPLKC/wKCAPL matches or ""_scale
        x.startswith('wAPLKC') or x.startswith('wKCAPL') or
        x in ('fixed_thr', 'sparsity', 'megamat_sparsity')
    )


def assert_param_dicts_equal(params: ParamDict, params2: ParamDict, *,
    # TODO only check these wKCAPL params w/ allclose when explicitly requested by
    # particular tests? (check none w/ allclose by default) (or am i going to need to
    # specify wKCAPL enough that i shouldn't...?)
    check_with_allclose=('wKCAPL','wKCAPL_scale'),
    only_check_overlapping_keys: bool = False, ignore_tuning_iters: bool = False,
    expected_missing_keys: Iterable[str] = tuple(),
    check_tuning_outputs: bool = True) -> None:
    # TODO doc if can come from a serialized output, which (and how to load, and whether
    # that's equiv to checking output returned from a call directly)
    # TODO and are param dicts from these two fns the same? doc how differ, if not
    """Asserts param dicts (as from `fit_mb_model` / `fit_and_plot_mb_model`) equivalent

    Args:
        only_check_overlapping_keys: if False, will err if keys are not the same

        expected_missing_keys: keys that `params2` is expected to be missing, but
            `params` is expected to have
    """
    # params that we'd expect to be different between the fixed thr/APL-weights call
    # and the call that picked those values (tuning_iters), and other special cases
    exclude_params = ('rv', 'mp', 'output_dir')
    if ignore_tuning_iters:
        exclude_params += ('tuning_iters',)

    if len(expected_missing_keys) > 0:
        missing_keys = set(params.keys()) - set(params2.keys())
        assert missing_keys <= set(expected_missing_keys), \
            f'{missing_keys=} > {expected_missing_keys=}'

        params = {k: v for k, v in params.items()
            if (k not in expected_missing_keys) or k in params2
        }

    if not only_check_overlapping_keys:
        k1 = {k for k in params.keys() if k not in exclude_params}
        k2 = {k for k in params2.keys() if k not in exclude_params}
        assert k1 == k2, (f'{diff_sets(k1, k2)}\n'
            'maybe set only_check_overlapping_keys=True?'
        )
    else:
        key_overlap = set(params.keys()) & set(params2.keys())
        params = {k: v for k, v in params.items() if k in key_overlap}
        params2 = {k: v for k, v in params2.items() if k in key_overlap}

    # TODO ignore diff types if values equal? then maybe just skip this altogether, and
    # integrete all into loop below?
    #
    # to simplify logic in loop below, where we actually check values equal
    for k in params.keys():
        if k in exclude_params:
            continue

        t1 = type(params[k])
        t2 = type(params2[k])
        if t1 != t2:
            # TODO need flag to disallow this sometimes?
            #
            # should just be when stuff read from CSV is a pure <class 'float'> and
            # returned value is <class 'numpy.float64'>
            #
            # e.g.
            # fixed_thr: <class 'numpy.float64'> != <class 'float'>
            # sparsity: <class 'numpy.float64'> != <class 'float'>
            # megamat_sparsity: <class 'numpy.float64'> != <class 'float'>
            assert issubclass(t1, t2) or issubclass(t2, t1), (
                f'{k}: neither type {t1=} {t2=} is subclass of other\n'
                f'{params[k]=} {params2[k]=}'
            )

    # currently just checking non-tuning outputs again (shouldn't add much time), on
    # subsequent calls that set check_tuning_outputs=False
    if check_tuning_outputs:
        keys_to_check = sorted(params.keys(),
            key=lambda x: (is_tuning_output_param(x), x)
        )
    else:
        keys_to_check = [x for x in params.keys() if not is_tuning_output_param(x)]
        # just sorting for consistency w/ above. shouldn't be necessary here.
        keys_to_check = sorted(keys_to_check)


    def check_key_values(k: str) -> None:
        v = params[k]
        v2 = params2[k]

        # TODO is it a problem that we didn't need to check wKCAPL w/ allclose
        # before, and we do now? (that list is growing now...)
        # NOTE: despite needing to check wKCAPL this way, didn't seem to need the
        # same for wAPLKC

        # TODO check for issues now that this is not just in check_with_allclose branch
        if type(v) is str:
            assert type(v2) is str, f'{k=}: type({v2=}) != str'
            # doing this rather than just float(v) for each, since sometimes we have
            # list-of-floats here
            # TODO TODO does this just raise AssertionError (if anything), or need to
            # handle other errors in check_key_values handling now?
            v, v2 = eval_and_check_compatible(v, v2)
        #

        # TODO delete check_with_allclose if i'm happy w/ this as a replacement
        assert equals(v, v2, check_float_with_allclose=True, equal_nan=True), \
            f'{k=}: {v=} not equals {v2=}'

        # TODO move initial else branch checks into pd_allclose if they have any value?
        # (/delete)
        '''
        if k not in check_with_allclose:
            # TODO (delete? can i still repro?) how are we getting this err here:
            # ValueError: The truth value of a Series is ambiguous. Use a.empty,
            # a.bool(), a.item(), a.any()
            # for test_fitandplot_repro[pn2kc_uniform__n-claws_7__n-seeds_2]
            # (oh, it's a list-of-Series for kc_spont_in)
            assert equals(v, v2), f'{k=}: {v=} not equals {v2=}'

        else:
            # TODO delete this, now that i have convert_dtypes=True path of
            # read_series_csv (and read_param_csv that wraps that)?
            # to handle input loaded from CSV into series, where many values still float
            # TODO can i change how i load to cast at load-time when possible?
            if type(v) is str:
                assert type(v2) is str, f'{k=}: type({v2=}) != str'
                # doing this rather than just float(v) for each, since sometimes we have
                # list-of-floats here
                v, v2 = eval_and_check_compatible(v, v2)

            # otherwise, would have to specify equal_nan=True below (and expect no
            # inputs should have NaN anyway). these lines work for both float and
            # ndarray input (.any() available on output of isnan for both), and should
            # also work for Series input.
            assert not np.isnan(v).any(), f'{k=}: {v=} had NaN'
            assert not np.isnan(v2).any(), f'{k=}: {v2=} had NaN'
            #

            # TODO is this Series case still hit? are they filtered out before saving to
            # params.csv? (if not, is serialization round trip param checking equipped
            # to parse Series element values?)
            if isinstance(v, pd.Series):
                assert isinstance(v2, pd.Series), \
                    f'{k=}: {v2=} ({type(v2)=}) is not a Series instance'

                # TODO modify pd_allclose to work w/ two float+ndarray inputs too
                # (if it doesn't already) -> simplify this code a bit (removing separate
                # branch calling np.allclose below)
                assert pd_allclose(v, v2), f'{k=}: pd.allclose({v=}, {v2=}) failed!'
            else:
                # NOTE: isinstance(x, <np.float64-scalar>) is True, and allclose works
                # with lists of floats/ints
                assert (
                    # TODO refactor handling to single call to check either float or
                    # int? some builtin / numpy way to do that already?
                    (isinstance(v, float) and isinstance(v2, float)) or
                    (isinstance(v, int) and isinstance(v2, int)) or

                    # NOTE: this should just be necessary for current wAPLKC/wKCAPL in
                    # one_row_per_claw=True output, but I may change type of those to
                    # Series
                    (isinstance(v, np.ndarray) and isinstance(v2, np.ndarray)) or

                    # should mostly (exclusively?) be for handling stuff concatenated
                    # across seeds, when running variable_n_claw=True cases through
                    # fit_and_plot_mb_model (w/ n_seeds > 1)
                    ((type(v) is list and type(v2) is list) and (
                            all(isinstance(x, float) for x in v) and
                            all(isinstance(x, float) for x in v2)
                        ) or (
                            all(isinstance(x, int) for x in v) and
                            all(isinstance(x, int) for x in v2)
                        )
                    )
                ), (f'{k=}: {type(v)=} and {type(v2)=} were not both float|int|ndarray|'
                    'list-of-[float|int]'
                )
                assert np.allclose(v, v2), f'{k=}: np.allclose({v=}, {v2=}) failed!'
        '''

    # TODO factor above value checking to (new, already existing) hong2p.util.equal (or
    # use that?)?
    #
    # sorting fixed_thr/wAPLKC/wKCAPL[_scale] to end, so we detect changes in parameters
    # that influence tuning first (before detecting changes in output of tuning i.e.
    # these parameters)
    msgs = []
    k_mismatch = []
    for k in keys_to_check:
        if k in exclude_params:
            continue

        # unless assertions above about keys()+types being equal above fail,
        # assuming keys are present in both and types equal.
        try:
            check_key_values(k)

        except AssertionError as err:
            msgs.append(str(err))
            k_mismatch.append(k)

    if len(msgs) > 0:
        # TODO don't print values longer than x lines? or no dataframes period? (handle
        # by changing assertion messages above, if so)
        warn(f'following keys had mismatched values:\n{k_mismatch}\n' + '\n'.join(msgs))
        assert False


def assert_fit_outputs_equal(ret: FitMBModelOutputs, ret2: FitMBModelOutputs, **kwargs
    ) -> None:
    """Checks outputs of two `fit_mb_model` calls are equal.

    Args:
        **kwargs: passed to `assert_param_dicts_equal`
    """
    responses, spike_counts, wPNKC, params = ret
    responses2, spike_counts2, wPNKC2, params2 = ret2

    # TODO could move back below param check, if i break out the part of
    # assert_param_dicts_equal that tests wAPLKC/fixed_thr/etc [anything that depends on
    # olfsysm tuning process], and do that after wPNKC check
    assert wPNKC.equals(wPNKC2)

    # intentionally checking responses/spike_counts last, as differences in params/wPNKC
    # will often help explain why there are differences in these (and differences in
    # these are often harder to interpret)
    assert_param_dicts_equal(params, params2, **kwargs)
    assert responses.equals(responses2)
    assert spike_counts.equals(spike_counts2)


def assert_fit_and_plot_outputs_equal(plot_root: Path, params: ParamDict,
    params2: Optional[ParamDict] = None , *, plot_root2: Optional[Path] = None,
    **kwargs) -> None:
    # TODO doc which outputs it checks
    """Asserts spike_counts, w[APLKC|KCAPL] weights, (most) params, and more are equal.

    Also asserts set of CSVs and pickles are the same in the two directories, and checks
    contents of all same-named pickled match across the two directories. Currently does
    NOT check content of all same-named CSVs like this.

    Args:
        plot_root: see `plot_root2`

        params2: if not passed, comparison params will be loaded from
            `plot_root2 / params['output_dir']`

        plot_root2: if NOT passed, assumes 'output_dir' value in both input dicts is a
            name of a directory under `plot_root`. If passed, `params2['output_dir']`
            (or `params['output_dir']`, if `params2=None`) will be under `plot_root2`

        **kwargs: passed thru to first `assert_param_dicts_equal` call, that directly
            compares `params` and `params2` inputs.
    """
    assert plot_root.is_dir(), f'{plot_root=} not a directory'

    dirname = params['output_dir']
    output_dir = (plot_root / dirname).resolve()
    # TODO TODO TODO fix how generate_... is triggering this now, in
    # check_against_last=True case
    assert output_dir.is_dir(), f'{output_dir=} not a directory'

    if params2 is None:
        assert plot_root2 is not None, 'must pass either params2 or plot_root2'

    if plot_root2 is None:
        plot_root2 = plot_root
    else:
        assert plot_root2.is_dir(), f'{plot_root2=} not a directory'

    if params2 is not None:
        dirname2 = params2['output_dir']
    else:
        dirname2 = dirname
    output_dir2 = (plot_root2 / dirname2).resolve()
    assert output_dir2.is_dir(), f'{output_dir2=} not a directory'

    assert output_dir != output_dir2

    # TODO TODO replace all read_tuned_params w/ read_params (done?), and then fix any
    # issues (/add exclusions) as needed (need to test)
    if params2 is None:
        # TODO TODO any point to read csv too? that ever have keys this doesn't? YES!!!
        # use that instead/too!!! tuned_params only has tuned params set in
        # fit_mb_model/olfsysm, but missing many in CSV. only stuff filtered before
        # saving CSV [if any] is in this but not CSV, and now we also should have either
        # json/pickle param_cache_name?  also load it and assert it doesn't?
        params2 = read_params(output_dir2)
        # TODO delete
        #params2 = read_tuned_params(output_dir2)

    def check_pickle_and_parquet_outputs(name):
        # TODO this even possible? shouldn't be if we define set of filenames to check
        # from intersection instead of union. any reason to not change def like that?
        if name in exclude_pickles:
            return

        # TODO use something other than pd.read_pickle? may want if i ever pickle
        # xarray, but it does work w/ np.ndarray, which is only other thing i think i
        # currently have in pickles (except for dict in params_for_csv.p, which should
        # be excluded anyway)
        if name.endswith('.p'):
            p1 = pd.read_pickle(output_dir / name)
            p2 = pd.read_pickle(output_dir2 / name)
        else:
            assert name.endswith('.parquet')
            # TODO use my own read_parquet wrapper instead
            p1 = pd.read_parquet(output_dir / name)
            p2 = pd.read_parquet(output_dir2 / name)

        # TODO delete?
        assert not isinstance(p1, np.ndarray)

        # parquet files should all be one of these two pandas types
        assert isinstance(p1, (pd.DataFrame, pd.Series))

        # TODO TODO TODO this working?
        # TODO use check_float_with_allclose elsewhere too
        assert equals(p1, p2, check_float_with_allclose=True), \
            f'{name=}\n{p1=}\nnot equals\n{p2=}'

    def filenames_with_ext(output_dir: Path, ext: str) -> Set[str]:
        return {x.name for x in output_dir.glob(f'*.{ext}')}

    # TODO allow some mismatch here? kwarg for that? e.g. odor_stats or claw_sims_sums.p
    # (/other dynamics outputs?)
    # TODO assert pickles has at least some minimum set of pickles?
    output_dir_pickles = filenames_with_ext(output_dir, 'p')
    output_dir2_pickles = filenames_with_ext(output_dir2, 'p')

    # TODO factor out?
    def filter_dynamics(fnames: Set[str]) -> Set[str]:
        # filters e.g. {'pn_sims.p', 'spike_recordings.p', 'Is_sims.p', 'vm_sims.p',
        # 'orn_sims.p', 'inh_sims.p'}
        return {n for n in fnames
            if not (n.endswith('_sims.p') or n in ('spike_recordings.p',
                'Is_from_kcs.p', 'Is_from_pns.p'
            ))
        }

    # TODO option to also compare these? (default to False tho)
    output_dir_pickles = filter_dynamics(output_dir_pickles)
    output_dir2_pickles = filter_dynamics(output_dir2_pickles)

    # TODO expose as kwarg (+ default to False)
    allow_old_cache_name = True
    old_tuned_param_name = 'params_for_csv.p'
    # NOTE: old name can only ever be in 1st output, and must not be in 2nd output
    assert old_tuned_param_name not in output_dir2_pickles
    if allow_old_cache_name and old_tuned_param_name in output_dir_pickles:
        warn(f'{output_dir=} used old tuned param cache name (={old_tuned_param_name})')
        # new name for same thing (new 'params.p' is "all" params, not just the "tuned"
        # params)
        new_tuned_param_name = 'tuned_params.p'
        assert new_tuned_param_name not in output_dir_pickles

        assert output_dir_pickles - output_dir2_pickles <= {old_tuned_param_name}
        # params.p ("all" params, which should be more than "tuned" params) should now
        # also exist, in addition to params.csv (which still has the same name as it did
        # before)
        assert output_dir2_pickles - output_dir_pickles <= {
            new_tuned_param_name, 'params.p'
        }
    else:
        # TODO is kc_spont_in.p still getting saved? still want it to be?
        assert output_dir_pickles == output_dir2_pickles, \
            f'{diff_sets(output_dir_pickles, output_dir2_pickles)}'

    output_dir_parquets = filenames_with_ext(output_dir, 'parquet')
    output_dir2_parquets = filenames_with_ext(output_dir2, 'parquet')
    assert output_dir_parquets == output_dir2_parquets, \
        f'{diff_sets(output_dir_parquets, output_dir2_parquets)}'

    # TODO delete eventually
    parquet_names = {x[:-len('.parquet')] for x in output_dir_parquets}
    pickle_names = {x[:-len('.p')] for x in output_dir_pickles}
    # TODO update to remove params_for_csv, after updating/renaming all old outputs to
    # have that one as 'tuned_params.p' instead
    assert pickle_names - parquet_names <= {'params_for_csv', 'params', 'tuned_params'}
    #

    # TODO assert some minimum set of CSVs?
    output_dir_csvs = filenames_with_ext(output_dir, 'csv')
    output_dir2_csvs = filenames_with_ext(output_dir2, 'csv')
    assert output_dir_csvs == output_dir2_csvs
    # TODO TODO also check contents of CSVs, like pickles?

    # TODO move check on wPNKC (currently done in loop below only, right?) before
    # assert_param_dicts_equal? (unless i'm going to break out the part of
    # assert_param_dicts_equal that tests wAPLKC/fixed_thr/etc [anything that depends on
    # olfsysm tuning process], and do that after wPNKC check)
    #
    # params_for_csv.p is loaded and checked above (we need to ignore a subset and
    # may need to special case allclose checking on some there)
    # TODO TODO just define from what is only in pickle (not parquet)
    # (or update to remove params_for_csv, after updating/renaming all old outputs to
    # have that one as 'tuned_params.p' instead, as above)
    exclude_pickles = ('params_for_csv.p', 'params.p', 'tuned_params.p')

    output_file_names_to_check = output_dir_pickles | output_dir_parquets

    # TODO could move back below param check, if i break out the part of
    # assert_param_dicts_equal that tests wAPLKC/fixed_thr/etc [anything that depends on
    # olfsysm tuning process], and do that after wPNKC check
    check_before_params = ('wPNKC',)
    for name in check_before_params:
        for path in (f'{name}.p', f'{name}.parquet'):
            assert path in output_file_names_to_check
            check_pickle_and_parquet_outputs(path)
            output_file_names_to_check.remove(path)

    # TODO ok that we are doing this before first two assert_param_dicts_equal calls
    # now? (delete commented below, if so)
    # TODO maybe don't add to expected_missing_keys, and just replace it?
    expected_missing_keys = set(kwargs.pop('expected_missing_keys', tuple()))
    # should no longer be serializing any of these in pickle, so as long as the 2nd arg
    # is the older output, should be ok here
    expected_missing_keys |= set(k for k, v in params.items()
        if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series)
    )
    #

    wAPLKC2 = None
    # TODO assert neither of these would be Series in output_dir2 / params2? (tho
    # separate files containing just the Series should match)
    if 'wAPLKC' in params:
        # in these cases, should be excluded via expected_missing_keys defined above,
        # for all assert_param_dicts_equal calls
        # TODO also want to pop from dicts?
        if isinstance(params['wAPLKC'], pd.Series):
            wAPLKC2 = params['wAPLKC']

    wKCAPL2 = None
    if 'wKCAPL' in params:
        if isinstance(params['wKCAPL'], pd.Series):
            wKCAPL2 = params['wKCAPL']

    # TODO TODO if wAPLKC/wKCAPL exist and are series, check that they match
    # wAPLKC/wKCAPL.p files in same dir, then exclude from param dict checking call
    # below (probably only applies to limited `one_row_per_claw and not
    # use_connectome_APL` cases)
    # TODO don't just thread kwargs thru all these kwargs? only manually pass
    # expected_missing_keys and ignore others for now? (esp since only doing on this
    # first call)
    assert_param_dicts_equal(params, params2,
        expected_missing_keys=expected_missing_keys, **kwargs
    )

    # comparing two CSVs, we should not have to worry about expected_missing_csv_keys.
    # assuming passed in expected_missing_keys (in kwargs) are still relevant.
    a2 = read_params(output_dir)
    b2 = read_params(output_dir2)

    # TODO update to some new assertion?
    # (no longer true, now that we are saving stuff to params_for_csv.p that get
    # filtered out by filter_params_for_csv before writing params.csv. e.g. stuff with
    # .shape, like kc_spont_in, wAPLKC, wKCAPL)
    #assert (a1.keys() - a2.keys()) == set()
    #assert (b1.keys() - b2.keys()) == set()

    # TODO check contents same for overlap too? could require type conversion in some
    # cases, and prob not worth...

    # TODO ok to also pass kwargs here? (may need to separately expose just
    # ignore_tuning_params, if not)
    assert_param_dicts_equal(a2, b2, expected_missing_keys=expected_missing_keys,
        **kwargs
    )

    # if loading the params again, why passing in? can these differ at all from input?
    # (yes, the passed in params will contain the most, with the CSVs often containing
    # almost all of those [only missing likely some minor ones, like 'pearson', or
    # 'wAPLKC'/'wKCAPL' in a special subset of the cases where they are Series cached to
    # their own files]).
    # TODO delete
    # TODO TODO delete? should all be in read_params output above now, certainly if
    # using output loaded from pickle/json and not CSV (should be using that in
    # read_params() now, though CSV should also be loaded and checked against that)
    a1 = read_tuned_params(output_dir)
    b1 = read_tuned_params(output_dir2)
    #
    # TODO delete (/restore) (moved above)
    # TODO maybe don't add to expected_missing_keys, and just replace it?
    #expected_missing_keys = set(kwargs.pop('expected_missing_keys', tuple()))
    ## should no longer be serializing any of these in pickle, so as long as the 2nd arg
    ## is the older output, should be ok here
    ## TODO TODO TODO was this an error, tha it was b1 and not a1? dict checking fn says
    ## 1 but not 2 should have it, no?
    #expected_missing_keys |= set(k for k, v in b1.items()
    #    if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series)
    #)
    #
    # TODO ok to also pass kwargs here? (may need to separately expose just
    # ignore_tuning_params, if not)
    # TODO can i delete only_check_overlapping_keys=True now?
    assert_param_dicts_equal(a1, b1, expected_missing_keys=expected_missing_keys,
        only_check_overlapping_keys=True, **kwargs
    )
    # TODO could maybe check the other way around? but we will always expect more in
    # params than either of these pickle outputs
    #assert (a1.keys() - params.keys()) == set()
    #assert (b1.keys() - params.keys()) == set()

    assert len(a2.keys() - a1.keys()) > 0
    assert len(b2.keys() - b1.keys()) > 0

    for name in output_file_names_to_check:
        check_pickle_and_parquet_outputs(name)

    # other than these and params.csv (dealt with above), seems only CSVs currently are:
    # {'responses.csv', 'wPNKC.csv', 'orn_deltas.csv'} (though this may change,
    # especially if I move/duplicate something currently only in pickles to CSVs)
    # TODO TODO pass something thru so i can read w/ `index_col=[KC_ID, 'seed']` for
    # stuff w/ variable_n_claws (and n_seeds > 1)
    df = _read_spike_counts(output_dir)
    df2 = _read_spike_counts(output_dir2)
    assert df.equals(df2)

    if 'wAPLKC' not in params:
        assert 'wKCAPL' not in params

        # TODO also check for CSVs if i also write to those in the future?
        # (+ check equiv to pickles, if so)
        # TODO TODO replace check of these pickles w/ just the parquet versions, after
        # checking i can replace w/ those
        assert 'wAPLKC.p' in output_dir_pickles
        assert 'wKCAPL.p' in output_dir_pickles

        assert 'wAPLKC.parquet' in output_dir_parquets
        assert 'wKCAPL.parquet' in output_dir_parquets

        # TODO TODO read the parquet instead
        #
        # we don't need to read from both dirs, b/c we already checked pickles of same
        # name have same contents (across the two dirs) above
        wAPLKC = pd.read_pickle(output_dir / 'wAPLKC.p')
        wKCAPL = pd.read_pickle(output_dir / 'wKCAPL.p')

        # TODO should i assert these are not None? or only have in some cases?
        if wAPLKC2 is not None:
            assert wAPLKC.equals(wAPLKC2)

        if wKCAPL2 is not None:
            assert wKCAPL.equals(wKCAPL2)

        assert isinstance(wAPLKC, pd.Series)
        assert isinstance(wKCAPL, pd.Series)
        # TODO TODO may need to fix some one-row-per-claw=True cases (which currently
        # may save these as ndarrays instead of Series), but prob want that
        assert wAPLKC.index.equals(wKCAPL.index)
        assert not wAPLKC.isna().any()
        assert not wKCAPL.isna().any()

        if not (params.get('one_row_per_claw', False) and
                params.get('use_connectome_APL_weights', False)):

            assert wAPLKC.index.equals(df.index)
        else:
            # TODO double check values still make sense here. are weights averaged
            # wAPLKC.index longer and has claw levels. df.index only KCs.
            assert len(df.index) < len(wAPLKC.index)
    # TODO delete this path now? (always popping above, before assert_param_dicts_equal
    # calls? or should i only be doing that in certain cases?)
    else:
        assert 'wKCAPL' in params
        assert params['wAPLKC'] is not None
        assert params['wKCAPL'] is not None
    #

    # TODO warn about any files (that are not plots) that we are not checking?
    # (excluding *.pdf, model_internals/ (which should only have *.pdf), and
    # olfsysm_log.txt [or renamed logs if multiple seeds]) should leave us only pickles
    # and CSVs, right?


@pytest.mark.xfail(
    reason='test unfinished. supporting code may or may not already be working',
    run=False
)
def test_vector_thr(orn_deltas):
    """
    Tests that `mp.kc.use_vector_thr` + hardcoded `rv.kc.thr` allows changing
    responsiveness of individual KCs.

    These variables aim to support changing KC-subtype response rates, both to:
    1. Make gamma KCs less responsive, despite more synapses [and thus more "claws"] in
       wPNKC (when using `pn2kc_connections='hemibrain'`, or anything connectome data
       where this is true)

    2. Inflate the responsiveness of a'/b' KCs, in line with some results from Inada,
       Tschimoto, and Kazama (2017)
    """
    connectome = 'hemibrain'
    wPNKC_kws = dict(
        weight_divisor=20,
    )
    # TODO add fixture to share this connectome_wPNKC call output (+ maybe wPNKC_kws,
    # perhaps as default_wPNKC_kws?)?
    wPNKC = connectome_wPNKC(connectome=connectome, **wPNKC_kws)

    assert len(wPNKC.index.names) > 1
    assert KC_TYPE in wPNKC.index.names

    kc_types = wPNKC.index.get_level_values(KC_TYPE)
    # should all be filled in w/ 'unknown', for now
    assert not kc_types.isna().any()

    _, hb_spike_counts, hb_wPNKC, hb_params = _fit_mb_model(orn_deltas=orn_deltas,
        pn2kc_connections=connectome, **wPNKC_kws
    )
    assert wPNKC.equals(hb_wPNKC)
    assert len(hb_spike_counts) == len(wPNKC)

    # TODO add some assertion that *appropriately*-weighted (by #-KCs-per-type) average
    # of per-KC response rates equals (approx) the overall avg response rate?
    # not super important (or delete n_kcs_per_type?)
    #
    # ab         802
    # g          612
    # a'b'       336
    # unknown     80
    #n_kcs_per_type = kc_types.value_counts()
    #

    hb_responses = hb_spike_counts > 0
    hb_avg_response_rate = hb_responses.mean().mean()
    hb_avg_response_rate_by_type = hb_responses.groupby(KC_TYPE).mean().T.mean()
    assert (
        hb_avg_response_rate_by_type.loc['g'] >
        (1.5 * hb_avg_response_rate_by_type[hb_avg_response_rate_by_type.index != 'g'])
    ).all()

    # TODO assertion that ~70th-90th percentile (whatever is appropriate) of # claws is
    # much bigger in hemibrain case than in uniform (by virtue of gamma KCs existing)?

    _, u7_spike_counts, u7_wPNKC, _ = _fit_mb_model(orn_deltas=orn_deltas,
        pn2kc_connections='uniform', n_claws=7, seed=12345
    )
    # TODO assert u7_wPNKC actually more even across types (if we just force kc_type
    # values from wPNKC above into index on u7_spike_counts, at least to get # of each
    # cell the same)? (already checked this about response rate below. may not care to
    # also check wPNKC for each)

    assert len(u7_spike_counts) == len(wPNKC)

    u7_responses = u7_spike_counts > 0
    u7_avg_response_rate = u7_responses.mean().mean()
    # TODO is it suspicious / lucky that these are exactly the same? same for any w/
    # same number of tuning iterations (w/ same tuning step size / etc params), i
    # suppose, so prob not suspicious.
    assert np.isclose(hb_avg_response_rate, u7_avg_response_rate)
    # NOTE: 'unknown' type has lower response rate, but probably some component of
    # smaller sample size. may also be b/c these cells are less complete in general
    #
    # ipdb> u7_avg_response_rate_by_type
    # kc_type
    # a'b'       0.097864
    # ab         0.099017
    # g          0.098904
    # unknown    0.073529
    # TODO TODO now dropping KC_TYPE from variable_n_claw=True cases (like this one), so
    # would need to add back if i wanted
    # TODO TODO what is purpose of comparison to this anyway tho?
    u7_avg_response_rate_by_type = u7_responses.groupby(KC_TYPE).mean().T.mean()
    assert np.isclose(
        u7_avg_response_rate_by_type['g'], u7_avg_response_rate_by_type, rtol=0.35
    ).all()
    assert np.isclose(
        u7_avg_response_rate_by_type['g'], u7_avg_response_rate_by_type.drop('unknown'),
        rtol=0.02
    ).all()

    # TODO is it suspicious that these are seemingly exactly the same?
    # (and not just each within 10% of target 0.1 response rate)
    # (some properties do differ between u7_responses and hb_responses tho, so doesn't
    # seem like duplicated. e.g. <x>.sum() of each is a unique vector)
    # ipdb> u7_responses.mean().mean()
    # 0.09765348762455801
    # ipdb> hb_responses.mean().mean()
    # 0.09765348762455801

    # TODO use get_thr_and_APL_weights here
    # 256.8058676548658
    hb_thr = hb_params['fixed_thr']

    _, spike_counts2, _, _= _fit_mb_model(orn_deltas=orn_deltas,
        pn2kc_connections=connectome, fixed_thr=hb_thr, wAPLKC=hb_params['wAPLKC'],
        **wPNKC_kws
    )
    assert np.array_equal(hb_spike_counts, spike_counts2)

    _, spike_counts, _, params = _fit_mb_model(orn_deltas=orn_deltas,
        pn2kc_connections=connectome, fixed_thr=np.array([hb_thr] * len(kc_types)),
        wAPLKC=hb_params['wAPLKC'],
        **wPNKC_kws
    )
    assert np.array_equal(hb_spike_counts, spike_counts)

    # TODO TODO TODO some kind of principled way to find these values? even iteratively?
    # (can i compute from pre-APL rv.kc.pks? i think so, but i might need to add a
    # "private" flag to return them? i don't see why i need post-APL rv.kc.pks
    # anyway...)
    #
    # TODO TODO modify fit_mb_model to take a kwarg of this name, rather than passing in
    # unlabelled ndarray (require scalar fixed_thr also passed in that case, and
    # multiply by that? not sure if anything else would make sense... but then that'd
    # also pretty much require a prior call...)
    type2thr_factor_even_rates = {
        # avg_response_rate_by_type:
        # kc_type
        # a'b'       0.103817
        # ab         0.101511
        # g          0.101019
        # unknown    0.101471
        "a'b'": 0.8,
        'ab': 0.9,
        # TODO is it not surprising that just a 15% change to threshold account for all
        # the difference in g-KC response rate (from ~5% to ~15%!)?
        'g': 1.15,
        'unknown': 0.6,
    }
    type2thr_factor_prime_boosted = {
        #"a'b'": 0.7,
        # 0.174545

        "a'b'": 0.6,
        'ab': 0.9,
        'g': 1.15,
        'unknown': 0.6,
    }

    # TODO TODO print out per-type threshold to compare to what i'm currently getting in
    # equalize_kc_type_sparsity=True code

    # TODO TODO can i modify fit_mb_model to calculate the starting threshold, upon
    # which to apply the per-type scale factors (maybe w/ a separate osm call?)

    type2thr_even_rates = {k: f * hb_thr for k, f in type2thr_factor_even_rates.items()}
    # TODO need to convert to numpy array (via .values)? (may want to change vector type
    # mb_model fns support to Series from ndarray, to keep metadata and check it against
    # internal)
    kc_thrs_even_rates = kc_types.map(type2thr_even_rates).values

    _, even_spike_counts, _, params = _fit_mb_model(orn_deltas=orn_deltas,
        pn2kc_connections=connectome, fixed_thr=kc_thrs_even_rates,
        # TODO for now, maybe also hardcode wAPLKC to output from hb_params?
        # (to require less code changes in mb_model.fit_mb_model) (doing so)
        wAPLKC=hb_params['wAPLKC'],
        #
        **wPNKC_kws
    )
    assert len(even_spike_counts) == len(wPNKC)
    even_responses = even_spike_counts > 0
    # TODO TODO is this even within +/-10% of target (default = 0.1)? if so, maybe just
    # b/c lucky? (if tuning APL, then should be, although target_sparsity_factor_pre_APL
    # might be violated, since response rates from threshold along [pre-APL] are not
    # currently controlled by olfsysm in the use_vector_thr=True case)
    # TODO add assertion it is?
    even_avg_response_rate = even_responses.mean().mean()
    #
    even_avg_response_rate_by_type = even_responses.groupby(KC_TYPE).mean().T.mean()

    # TODO then check that, with another call using use_vector_thr=True (and
    # setting an appropriate vector rv.kc.thr), we can even gamma KC response rate out,
    # despite having larger average wPNKC there
    # (pretty much have this now, w/ current type2thr_factor values. just need to add an
    # assertion w/ appropriate rtol. ideally same as for similar check above on u7 data)
    # TODO TODO actually add the assertion for comment above

    # TODO TODO TODO this code work w/ use_connectome_APL_weights=True? i assume it
    # would currently fail for same reason my equalize_kc_response_rates=True code is
    # failing w/ use_connectome_APL_weights=True (where response rates are ~2x what they
    # should be in that case)?

    # TODO delete
    print()
    print(f'{hb_params["wAPLKC"]=}')
    from pprint import pprint
    print()
    print()
    print('evened-subtypes + wAPLKC fixed to hemibrain tuned value (prob not ideal):')
    print('type2thr_even_rates:')
    pprint(type2thr_even_rates)
    print()
    # TODO this also w/in tolerance of hb avg response rate? assert?
    print(f'{even_avg_response_rate=}')
    print('even_avg_response_rate_by_type:')
    print(even_avg_response_rate_by_type)
    breakpoint()
    #
    # TODO delete comment block below
    # TODO try using breakpoint instead of this? work better (/ same, w/ less
    # configuration required in pyproject.toml? may just require --pdb?)
    # pytest.set_trace()?
    # breakpoint() does seem to provide much nicer output than ipdb.set_trace() (at
    # least when relying on --pdb instead of -s), even though breakpoint is now
    # configured to also give us ipdb (it seems)
    # TODO actually true pdb.set_trace() disables capturing automatically (yes, -s not
    # needed. breakpoint() seems to do same)? some other part of my config interfering
    # with that? or is it really special and neither breakpoint() (w/ my current
    # debugger class) or ipdb.set_trace() can replace that?
    #import pdb; pdb.set_trace()
    # TODO try? nah (why not referenced in docs? deprecated?)
    #pytest.set_trace()
    #

    # TODO TODO TODO (maybe in a script, not this test) compare consequences of using
    # threshold to even subtype response rates vs using changes to wPNKC (and mabye
    # wAPLKC+wKCAPL) to do the same? (maybe only using subtype thresholds to boost
    # a'b'-KCs, after taking perhaps another strategy to even things out first)

    type2thr_prime_boosted = {
        k: f * hb_thr for k, f in type2thr_factor_prime_boosted.items()
    }
    # TODO need to convert to numpy array (via .values)? (may want to change vector type
    # mb_model fns support to Series from ndarray, to keep metadata and check it against
    # internal)
    kc_thrs_prime_boosted = kc_types.map(type2thr_prime_boosted).values

    _, pboost_spike_counts, _, params = _fit_mb_model(orn_deltas=orn_deltas,
        pn2kc_connections=connectome, fixed_thr=kc_thrs_prime_boosted,
        # TODO for now, maybe also hardcode wAPLKC to output from hb_params?
        # (to require less code changes in mb_model.fit_mb_model) (doing so)
        wAPLKC=hb_params['wAPLKC'],
        #
        **wPNKC_kws
    )
    assert len(pboost_spike_counts) == len(wPNKC)
    pboost_responses = pboost_spike_counts > 0
    # TODO TODO is this even within +/-10% of target (default = 0.1)? if so, maybe just
    # b/c lucky? (if tuning APL, then should be, although target_sparsity_factor_pre_APL
    # might be violated, since response rates from threshold along [pre-APL] are not
    # currently controlled by olfsysm in the use_vector_thr=True case)
    pboost_avg_response_rate = pboost_responses.mean().mean()
    #
    pboost_avg_response_rate_by_type = pboost_responses.groupby(KC_TYPE).mean().T.mean()

    # TODO delete
    print()
    print('prime-boosted + wAPLKC fixed to hemibrain tuned value (prob not ideal):')
    # at least we are within tolerance here (w/ g-boosted values above)
    # 0.10610736097717777
    print(f'{pboost_avg_response_rate=}')
    print('pboost_avg_response_rate_by_type:')
    print(pboost_avg_response_rate_by_type)
    breakpoint()
    #

    # TODO TODO TODO can i get vector thr to work with connectome APL weights?

    # TODO TODO TODO can i get it to work w/o also having to pass in wAPLKC?

    # TODO TODO also test fit_and_plot... both w/ and w/o wAPLKC set?
    # mainly to check that they run, and save to diff dirs at least.
    # (both in non-connectome-APL case, but might want to test that too)


    # TODO check whether scaling wPNKC within subtypes can have same effect as changing
    # thresh? not sure, but think it should be possible (assumes float wPNKC supported,
    # which i think it should be, at least for non-Tianpei code)


def test_homeostatic_thrs(orn_deltas):
    ret = _fit_mb_model(orn_deltas=orn_deltas, homeostatic_thrs=True)
    responses, spike_counts, wPNKC, params = ret

    # TODO replace w/ get_thr_and_APL_weights?
    thrs = params['fixed_thr']

    # TODO TODO update so it's a Series here too? is this currently the only case where
    # thrs is returned as a numpy ndarray?
    assert isinstance(thrs, np.ndarray)

    assert len(thrs) == len(spike_counts)
    # actual range of thresholds seems to be [-57, 574] in current test, so atol could
    # be as much as this difference (but keeping smaller in case i ever make thresholds
    # more meaningful, where the range of differences should be smaller)
    assert not np.allclose(thrs[0], thrs, atol=3)
    assert not np.isnan(thrs).any()

    wAPLKC = params['wAPLKC']
    # also works for np.float64 scalars (what it actually seems to be)
    assert isinstance(wAPLKC, float)
    #

    ret2 =  _fit_mb_model(orn_deltas=orn_deltas, fixed_thr=thrs, wAPLKC=wAPLKC)
    assert_fit_outputs_equal(ret, ret2, ignore_tuning_iters=True)


# TODO add test that allow_net_inh_per_claw=True (the default) produces some
# negative claw activities (and that there are none w/ =False)
# TODO + test that orn activities (after adding spont) can't be negative (but seems like
# C++ code might allow them to be? fix that?

# TODO TODO TODO fix (broken as of 2026-02-13)
# TODO TODO TODO try just removing the one assertion on relative sparsity diff (w/ and
# w/o allow_net_inh...) and see if there are any other issues
# TODO TODO TODO why is such a high wAPLKC scale seemingly needed now (bug in olfsysm
# probably? did i change scale factor used in this case? check old logs / outputs?)
#@pytest.mark.xfail(
#    reason='broke in Feb 2026. fix!', run=False
#)
def test_spatial_wPNKC_equiv(orn_deltas):
    """
    Tests that one-row-per-claw wPNKC can recreate one-row-per-KC hemibrain outputs,
    where latter previously had a count of #-of-claws in each wPNKC matrix element, if
    olfsysm is modified and configured to interpret input rows as claws, but still
    having olfsysm ignore claw coordinate information.
    """
    # TODO TODO also add checks that we can use either tianpei/prat one-row-per-claw
    # variants, and then get same output if we sum over the claws there (rather than
    # splitting apart an existing wPNKC matrix here)
    # TODO TODO may need to rewrite test to create wPNKC by summing over prat_claws
    # version, rather than splitting apart, if i still want a similar test in the
    # meantime
    for kws in dict_seq_product(
            # for allow_net_inh_per_claw=True cases, outputs should match exactly.
            # for new default of =False, outputs should be close but not match exactly.
            [dict(allow_net_inh_per_claw=True), dict()],
            # TODO delete (was skipping allow_net_inh_per_claw case above. still
            # helpful? i assume this test still broken?)
            #[dict()],

            [dict(use_connectome_APL_weights=True), dict()]
        ):

        print(f'{kws=}')

        connectome = 'hemibrain'
        wPNKC_kws = dict(
            weight_divisor=20,
        )
        wPNKC = connectome_wPNKC(connectome=connectome, **wPNKC_kws)

        assert not wPNKC.index.duplicated().any()
        assert not wPNKC.index.get_level_values(KC_ID).duplicated().any()

        one_hot_claw_series_list = []
        for (kc_id, kc_type), n_claws_per_glom in wPNKC.iterrows():
            claw_id = 0

            # if we don't add these elements, the two KCs with no claws will be in
            # wPNKC's index, but not final wPNKC_one_row_per_claw index
            if n_claws_per_glom.sum() == 0:
                one_hot_claw_series_list.append(pd.Series(index=wPNKC.columns,
                    data=False, name=(kc_id, kc_type, claw_id)
                ))

            for glom, n_claws_from_glom in n_claws_per_glom.items():
                while n_claws_from_glom > 0:
                    # glom index -> 1 for one glomerulus (indicating one claw from that
                    # glomerulus to this KC), 0 for all others. name= contents will form
                    # 2-level MultiIndex w/ names=['kc_id','claw_id'], after loop.
                    one_hot_claw_series = pd.Series(index=wPNKC.columns, data=False,
                        name=(kc_id, kc_type, claw_id)
                    )
                    one_hot_claw_series.at[glom] = True
                    one_hot_claw_series_list.append(one_hot_claw_series)
                    n_claws_from_glom -= 1
                    claw_id += 1

        n_claws = wPNKC.sum().sum()
        assert sum((x > 0).any() for x in one_hot_claw_series_list) == n_claws
        assert len(one_hot_claw_series_list) == n_claws

        # values will be True if there is a claw for the (kc,glom) pair, otherwise False
        wPNKC_one_row_per_claw = pd.concat(one_hot_claw_series_list, axis='columns',
            verify_integrity=True
        )
        # values are the elements of the 2-tuples from the .name of each concatenated
        # Series
        wPNKC_one_row_per_claw.columns.names = [KC_ID, KC_TYPE, CLAW_ID]

        # AFTER .T, rows will be [KC_ID, CLAW_ID] and columns will be 'glomerulus',
        # with claw_id values going from [0, <#-claws-per-(kc,glom)-pair> - 1]
        wPNKC_one_row_per_claw = wPNKC_one_row_per_claw.T.copy()

        # TODO move/copy assertion that there are no duplicate [KC_ID, CLAW_ID]
        # combos (from one_row_per_claw=True code in mb_model) here? refactor
        # most checks to share?

        # TODO replace x/y/z/ ranges w/ those from actual hemibrain data
        # (in micrometers)
        x_min, y_min, z_min = (0, 0, 0)
        x_max, y_max, z_max = (100, 100, 100)

        # .names = [KC_ID, CLAW_ID]
        index = wPNKC_one_row_per_claw.index

        # best practice would in theory be passing around np.random.Generator objects (a
        # la https://stackoverflow.com/questions/68222756), but this global seed+rng
        # should be fine here.
        # NOTE: the issue is that it will also seed any numpy RNG in tests that happen
        # after, whether they were intended to be seeded or not
        np.random.seed(1)
        claw_coords = pd.DataFrame({
            # TODO change to have input that are ints in same range as neuprint input
            # (if that data is in ints... is it?), which are then multiplied by same
            # 8/1000 factor to get units of micrometers
            'claw_x': np.random.uniform(x_min, x_max, len(index)),
            'claw_y': np.random.uniform(y_min, y_max, len(index)),
            'claw_z': np.random.uniform(z_min, z_max, len(index)),
        })
        for_claw_index = pd.concat([index.to_frame(index=False), claw_coords],
            axis='columns', verify_integrity=True
        )
        claw_index = pd.MultiIndex.from_frame(for_claw_index)
        # .names now [KC_ID, CLAW_ID, 'claw_x', 'claw_y', 'claw_z'] with the x/y/z/
        # coord values randomly generated (deterministically)
        wPNKC_one_row_per_claw.index = claw_index

        # checking we didn't drop any claws through the one-hot-encoding process
        assert wPNKC_one_row_per_claw.groupby([KC_ID, KC_TYPE]).sum().equals(wPNKC)

        _, spike_counts, wPNKC2, _ = _fit_mb_model(orn_deltas=orn_deltas, **kws,
            pn2kc_connections=connectome, **wPNKC_kws
        )
        assert wPNKC.equals(wPNKC2)

        # just establishing new path allowing us to hardcode _wPNKC works
        # TODO move to separate test? and also add support for + test hardcoding of
        # wAPLKC and wKCAPL there?
        _, spike_counts2, _, _ = _fit_mb_model(orn_deltas=orn_deltas, **kws,
            _wPNKC=wPNKC
        )
        assert spike_counts.equals(spike_counts2)
        del spike_counts2

        _, spike_counts2, _, _ = _fit_mb_model(orn_deltas=orn_deltas,
            **kws, _wPNKC=wPNKC_one_row_per_claw, one_row_per_claw=True
        )
        if kws.get('allow_net_inh_per_claw', False):
            assert spike_counts.equals(spike_counts2)
        else:
            # could be possible to still match under some circumstances, but that's not
            # what we see if current tests here (so something would be up if they did
            # match)
            assert not spike_counts.equals(spike_counts2)

            # (outdated) currently getting 0.0172 w/ NO connectome APL weights, and
            # 0.0139 with them.
            #
            # (outdated) connectome_APL_weights=False (default):
            # (spike_counts - spike_counts2).abs().sum().sum()=99.0
            # spike_counts.sum().sum()=5752.0
            # spike_counts2.sum().sum()=5801.0
            # rel_abs_change=0.017211404728789986
            rel_abs_change = (spike_counts - spike_counts2).abs().sum().sum() / (
                spike_counts.sum().sum()
            )
            # TODO TODO TODO fix (broken as of 2026-02-13)
            # assert 0.19137422105608395 < 0.0175
            # (.16 if denominator is the larger spike_counts2.sum().sum())
            # sparsity of spike_counts is .0979212
            # sparsity of spike_counts2 is .109988
            #
            # the extent of the difference after changing C++ code to sum excitation and
            # inhibition within each claw, before then summing over claws within KC
            # (to support returning claw sims):
            # connectome_APL_weights=True:
            # ipdb> (spike_counts - spike_counts2).abs().sum().sum()
            # 97.0
            # ipdb> spike_counts.sum().sum()
            # 6976.0
            # ipdb> spike_counts2.sum().sum()
            # 7021.0
            # ipdb> rel_abs_change
            # 0.013904816513761468
            # TODO need hardcode_initial_sp to repro? maybe just relax
            # tolerance? (currently getting 0.19137 for
            # dict(use_connectome_APL_weights=True), though... isn't that a bit high?)
            # TODO or just increase sp_acc a bit here?
            # (only other case here [dict()] seems to not fail here though.
            # rel_abs_change=0.01406 there)
            # TODO if we didn't tune separately, and we just used the same
            # thr/weights, would that make outputs any more consistent? worth trying?
            # TODO warn instead? delete warning even?
            rel_abs_change_thr = 0.02
            # TODO is current value actually an issue (tuning is still converging
            # in each separate case, right? so prob not...)
            if rel_abs_change > rel_abs_change_thr:
                warn(f'response rate changed by >{rel_abs_change_thr} across '
                    'allow_net_inh_per_claw=False/True cases!'
                )
            # TODO delete?
            #assert rel_abs_change < 0.0175

        # TODO add test like fit_mb_model call above, but shuffling order of wPNKC rows.
        # order of rows should not matter.
        # TODO also shuffling claw_id's within each KC (that also shouldn't matter,
        # and is probably a more important test than shuffling rows)


# TODO move to mb_model? (+ get indices from some other output in params, rather than
# having to pass spike_counts in for that?)
def get_pks_and_thr(params: ParamDict, spike_counts: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], pd.Series]:
    """
    Returned pks will be None if model had fixed threshold pre-specified, rather than
    tuning to find threshold.
    """
    pks = params['rv'].kc.pks
    if len(pks) == 0:
        pks = None
    else:
        pks = pd.DataFrame(data=pks, index=spike_counts.index,
            columns=spike_counts.columns
        )

    thr = params['rv'].kc.thr
    thr = pd.Series(thr.squeeze(), index=spike_counts.index, name='thr')
    return pks, thr


# some way to get --collect-only to not print docstring for each separately
# parameterized version of these tests? (yes, pass -q/--quiet)
@pytest.mark.parametrize('kws', MODEL_KW_LIST, ids=format_model_params)
def test_fixed_inh_params(tmp_path, orn_deltas, kws):
    # TODO regarding last part of doc: are we really only "typically" hardcoding thr for
    # those calls, or always?
    """
    Tests that outputs of `fit_mb_model` calls, where KC thresholds and APL<->KC weights
    are tuned to a target response rate, can be recapitulated by similar calls that also
    set the APL<->KC weight parameters (and typically also the KC threshold parameters,
    either `mp.kc.fixed_thr` or `rv.kc.thr`).

    Only calls `fit_mb_model`
    """
    # TODO (? delete?) move some/all of the commented stuff in mb_model regarding
    # picking which (of slightly numerically different) thr values for fixed_thr into
    # here? (or a sep test)

    # TODO does use_connectome_APL_weights=True case take multiple iterations
    # to terminate? (no! current one terminates immediately!) want at least one test
    # case that does (and probably another that terminates immediately, or after 1
    # iteration)

    # TODO delete (move to script to generate test data)
    return_olfsysm_vars = False
    # (using to get pks & thr, to compare to after new code which might resolve
    # possible bug in code that picks thresholds (wAPLKC/wKCAPL may not be 0 when
    # they are supposed to be. for some reason APL activity [inh & Is] still seem to
    # be 0 as desired during sim_KC_layer calls to pick thresholds...)
    #if kws == dict('use_connectome_APL_weights'=True):
    #    return_olfsysm_vars = True
    #

    print(f'{kws=}')
    ret = _fit_mb_model(orn_deltas=orn_deltas, return_olfsysm_vars=return_olfsysm_vars,
        **kws
    )
    responses, spike_counts, wPNKC, params = ret
    # does 0 actually imply no tuning happened? yes, it's incremented to 1 on first
    # iteration.
    assert params['tuning_iters'] > 0

    # TODO remove this? to support vector thr case?
    fixed_thr = params['fixed_thr']
    assert isinstance(fixed_thr, float)
    del fixed_thr
    #

    # TODO instead return new kws that add thr_and_apl_kws to them (while also
    # checking no key overlap)? or change to check no key overlap here (would then
    # want to duplicate that a bunch of places tho, hence preference for former)
    #
    # TODO in one_row_per_claw=True case, where does wAPLKC_scale get set (in
    # use_connectome_APL_weights=True case), if tianpei's one-row-per-claw path
    # through olfsysm does not use preset_wAPLKC=true (which is only path in olfsysm
    # that uses wAPLKC_scale, no?)? (i was probably missing/misunderstanding
    # something)
    # TODO is it necessary for wKCAPL[_scale] to be set explicitly for
    # tianpei's test to pass? (try without)
    # if so, could code be modified to just require wAPLKC[_scale], as w/ my code?
    # (if not, why not?)
    # TODO try removing wAPLKC/wKCAPL for repro calls where we also have
    # w[APLKC|KCAPL]_scale, and make sure still passes (should never need to
    # explicitly pass in) (just remove those keys currently added in
    # get_thr_and_APL_weights?)
    thr_and_apl_kws = get_thr_and_APL_weights(params, kws)

    ret2 = _fit_mb_model(orn_deltas=orn_deltas,
        return_olfsysm_vars=return_olfsysm_vars, **{**thr_and_apl_kws, **kws}
    )
    responses2, spike_counts2, wPNKC2, params2 = ret2

    # does 0 actually imply no tuning happened? yes, it's incremented to 1 on first
    # iteration.
    assert params2['tuning_iters'] == 0

    assert_fit_outputs_equal(ret, ret2, ignore_tuning_iters=True)

    # TODO delete? move to a script to generate test data (along w/ related code
    # above)?
    if return_olfsysm_vars:
        del params['mp']
        pks, thr = get_pks_and_thr(params, spike_counts)
        del params['rv']

        del params2['mp']
        pks2, thr2 = get_pks_and_thr(params2, spike_counts2)
        del params2['rv']

        # expected when using pre-specified fixed threshold
        assert pks2 is None
        del pks2

        assert thr.equals(thr2)
        del thr2

        save_pks_and_thr = False
        if save_pks_and_thr:
            print('saving thr.csv and pks.csv for test_connectome_APL_repro!')
            print('set save_pks_and_thr=False and return')

            thr.to_csv('thr.csv')
            pks.to_csv('pks.csv')
            # TODO could handle this one by checking all outputs of fit_and_plot...
            # dir against contents of an old dir. would need to add a test that does
            # that though, and commit current outputs w/ such a call in this case
            spike_counts.to_csv('spike_counts.csv')

            thr3 = pd.read_csv('thr.csv', index_col=[KC_ID, KC_TYPE]).squeeze()
            assert pd_allclose(thr, thr3)

            pks3 = pd.read_csv('pks.csv', index_col=[KC_ID, KC_TYPE])
            assert pd_allclose(pks, pks3)
    #

    print()


# TODO delete all this? missing between what? CSV has way more keys in general actually
# TODO TODO maybe it makes sense if only used to compare CSV to returned (but not CSV to
# pickle)
def get_expected_missing_csv_keys(model_kws: ParamDict) -> Tuple[str]:
    # TODO doc (w/ explanation as to why each expected to be missing)
    # TODO go based on type of value for each? shouldn't need any other params. these
    # should all be np.arrays or series. probably all series?
    # TODO and assert all other values are simple types (that can be round tripped
    # through simply json or something? could help move towards a non-pickle format for
    # that, which might be last pickle holdout, if i convert all pandas stuff to
    # parquet)
    expected_missing_csv_keys = tuple()
    # TODO delete. this should now always be saved in a separate file.
    #expected_missing_csv_keys = ('kc_spont_in',)

    # TODO does one_row_per_claw actually matter at all? (maybe only if
    # `use_connectome_APL_weights=False and prat_claws=False`)?
    if (model_kws.get('one_row_per_claw') or
        model_kws.get('use_connectome_APL_weights')):

        # TODO these the appropriate cases? also for vector_thr=True cases? any other?
        expected_missing_csv_keys += ('wAPLKC', 'wKCAPL')

    # TODO move into conditional above?
    if model_kws.get('prat_boutons') and not model_kws.get('per_claw_pn_apl_weights'):
        expected_missing_csv_keys += ('wAPLPN', 'wPNAPL')

    # TODO put all dataframe/series in here? (oh, don't have output params, just input)

    return expected_missing_csv_keys


def assert_param_csv_matches_returned(model_output_root: Path, param_dict: ParamDict, *,
    expected_missing_csv_keys: Iterable[str] = ('kc_spont_in',)
    ) -> None:
    """Asserts 'params.csv' file contents matches dict `fit_and_plot_mb_model` returns

    Args:
        model_output_root: as passed to `fit_and_plot_mb_model`

        param_dict: as returned from `fit_and_plot_mb_model`
    """
    assert model_output_root.is_dir()

    model_output_dir = model_output_root / param_dict['output_dir']
    assert model_output_dir.is_dir()
    p2 = read_params(model_output_dir)

    returned_keys = set(param_dict.keys())

    csv_keys = set(p2.keys())
    # TODO to what extent will this always be the case?
    assert csv_keys - returned_keys == set(), f'{csv_keys - returned_keys=}'

    # TODO delete
    # TODO fix other code to remove need for this hack (should only be b/c these are
    # DataFrames returned now that i'm hardcoding plot_example_dynamics in some places.
    # could prevent this from being in param_dict output, probably globally?)
    # TODO or replace all get_expected_missing_csv_keys usage w/ this?
    extra_missing_keys = {k for k, v in param_dict.items()
        if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series)
    }
    if len(extra_missing_keys) > 0:
        # TODO warn?
        expected_missing_csv_keys = set(expected_missing_csv_keys) | extra_missing_keys
    #

    # TODO delete try/except?
    try:
        # TODO do i want to load wAPLKC/wKCAPL (/kc_spont_in?) from somewhere else
        # and also check they matches? (can prob leave to fn checking more than just
        # params?)
        # TODO only equality in tianpei case?
        assert returned_keys - csv_keys <= set(expected_missing_csv_keys), \
            f'{returned_keys - csv_keys=} > (!= {expected_missing_csv_keys=})'

        assert_param_dicts_equal(param_dict, p2, only_check_overlapping_keys=True)

    except AssertionError as err:
        warn(traceback.format_exc())
        # TODO delete
        breakpoint()
        #
        raise

    # TODO remove this eventually?
    # TODO also assert values match what i'm loading from CSV (in case my parsing fn is
    # missing some cases, perhaps just in the type handling?)
    #
    # the only things that seem to be in here, but not in CSV, are things that should be
    # saved in separate files anyway ('kc_spont_in', and when they are Series 'wKCAPL'+
    # 'wAPLKC). shouldn't need to check this.
    from_tuned = read_tuned_params(model_output_dir)
    tuned_keys = set(from_tuned.keys())
    # this could only happen in the single current special case in param_dict
    # non-serializable output saving loop (that pops + saves DataArray/DataFrames/etc),
    # where wAPLKC/wKCAPL are not popped for one particular set of parameters
    # (`one_row_per_claw and not use_connectome_APL_weights`)
    assert tuned_keys - csv_keys <= set(expected_missing_csv_keys)
    # "all" params in CSV will always have many more entries than the "tuned" params.
    # model_kws/etc.
    assert len(csv_keys - tuned_keys) > 0, f'{csv_keys - tuned_keys=}'


# TODO also want a test checking output of fit_mb_model and fit_and_plot_mb_model calls
# are equiv, for same input? prob not too important
# TODO still want separate test that checks we can load output for all of these?
# (at least all used by downstream in model_mb_responses or natmix_data/analysis.py)
# (or prob just incorporate into test below that we can load and also check those
# equiv? could prob only check params, or whatever else was included in there? other
# outputs are just in terms of serialized data files, so nothing to check against,
# right?)
@pytest.mark.parametrize('kws', FITANDPLOT_MODEL_KW_LIST, ids=format_model_params)
def test_fixed_inh_params_fitandplot(tmp_path, orn_deltas, kws):
    """
    Like test_fixed_inh_params, but calling (+ checking outputs of)
    fit_and_plot_mb_model instead of fit_mb_model.
    """
    plot_root = tmp_path

    params = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, **kws)
    # are dynamics in params here, when return_dynamics=True is in PLOT_KWS? no, doesn't
    # seem so.
    # TODO so anything i can do about current memory issue killing one test in second
    # call here, in prat_boutons case? (ok, well doesn't seem to be a big enough deal
    # that it is always getting killed, with ~21GB available)

    # TODO need to define this separately for fixed-inh vs not case? likely...
    # (but may only need to check once b/c we are already checking params from the two
    # calls against each other, right? unless the checks against CSV are somehow *more*
    # comprehensive than checks against output of these two calls, but i doubt that)
    expected_missing_csv_keys = get_expected_missing_csv_keys(kws)
    assert_param_csv_matches_returned(plot_root, params,
        expected_missing_csv_keys=expected_missing_csv_keys
    )

    # TODO try removing wAPLKC/wKCAPL for repro calls where we also have
    # w[APLKC|KCAPL]_scale (aka use_connectome_APL_weights=True cases), and make sure
    # still passes (should never need to explicitly pass in) (just remove those keys
    # currently added in get_thr_and_APL_weights?)
    thr_and_apl_kws = get_thr_and_APL_weights(params, kws)

    # get_thr_and_APL_weights should rename any *_scale parameters to just their prefix,
    # overwriting any existing keys w/ that name (e.g. 'wAPLKC_scale' -> 'wAPLKC')
    assert 'wAPLKC_scale' not in thr_and_apl_kws
    assert 'wAPLKC' in thr_and_apl_kws
    wAPLKC = thr_and_apl_kws['wAPLKC']

    # TODO TODO true? maybe only in one of the branches below?
    assert 'wKCAPL' not in thr_and_apl_kws
    #
    if (kws.get('one_row_per_claw', False) and
        not kws.get('use_connectome_APL_weights', False)):
        # TODO should there also have been some scalar *_scale parameter too?
        # another mistake? (doesn't seem so. test passing.)
        # (no scalar values, nor *_scale scalars here. just a single series wAPLKC
        # currently, though maybe it should also have been a series wKCAPL, and mabye at
        # some point the *_scale params too?)
        assert isinstance(wAPLKC, pd.Series)
    else:
        # TODO refactor to share w/ existing mb_model code
        assert isinstance(wAPLKC, float) or (
            isinstance(wAPLKC, list) and all(isinstance(x, float) for x in wAPLKC)
        )

    params2 = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
        **thr_and_apl_kws, **kws
    )
    # TODO just do one of these?
    assert_param_csv_matches_returned(plot_root, params2,
        # TODO need to define this separately for fixed-inh vs not case? likely...
        # (but may only need to check once b/c we are already checking params from the
        # two calls against each other, right? unless the checks against CSV are somehow
        # *more* comprehensive than checks against output of these two calls, but i
        # doubt that)
        expected_missing_csv_keys=expected_missing_csv_keys
    )

    assert_fit_and_plot_outputs_equal(plot_root, params, params2,
        ignore_tuning_iters=True
    )

    # TODO also test we can load all the things that either downstream stuff in
    # mb_model (e.g. in model_mb_responses) or natmix_data/analysis.py uses (and
    # that all in same type / format?) (or separate test for that?) (or just all
    # things we can reasonably load?)
    # (should be doing most of this now? delete? maybe also separate
    # [non/less-parametrized] test that checks we can save and load dynamics outputs?)


reference_output_root = test_data_dir / 'reference_outputs_for_repro'
def get_latest_reference_output_dir() -> Path:
    """Returns path to latest directory created by script to generate reference outputs

    The script to generate reference outputs is
    `generate_reference_outputs_for_repro.py`
    """
    assert reference_output_root.is_dir(), \
        'was generate_reference_outputs_for_repro.py run?'

    def sort_key(path: Path) -> Tuple[str, int]:
        assert path.is_dir()

        name = path.name
        parts = name.split('.')
        assert 0 < len(parts) <= 2

        # should be in YYYY-MM-DD format
        # TODO assert date format?
        date_str = parts[0]
        assert len(date_str) == len('YYYY-MM-DD'), \
            f'{date_str=} not in expected YYYY-MM-DD date format'

        if len(parts) == 1:
            n = -1
        else:
            # starting from 0, when this part present, counting up from there.
            # non-padded integer.
            n = int(parts[-1])

        return date_str, n

    # NOTE: trailing slash isn't sufficient to exclude directories, not with pathlib at
    # least
    subdirs = list(reference_output_root.glob('*/'))
    subdirs = [x for x in subdirs if x.is_dir()]
    assert len(subdirs) > 0, (f'no outputs under {reference_output_root=}\n'
        'try running generate_reference_outputs_for_repro.py again?'
    )
    subdirs = sorted(subdirs, key=sort_key)
    reference_output_dir = subdirs[-1]
    # probably redundant...
    assert reference_output_dir.is_dir()
    print(f'latest reference output dir:\n{reference_output_dir}')
    print()
    # TODO also assert it has some subdirs? or leave to whatever uses this? (leaning
    # towards latter...)
    return reference_output_dir


# TODO add some options to FITANDPLOT_MODEL_KW_LIST, just for repro check (def new
# var), and use that both in generate script and here? want to add one case of
# `Btn_separate=True + Btn_num_per_glom=5`, but don't necessarily want to test that in
# other tests? or just test elsewhere too?
#
# TODO possible to make similar test for fit_mb_model, or too much work to implement
# serialization, such that easiest is just to call fit_and_plot_mb_model?
# TODO use something other than parametrize, to generate only tests for the data we have
# available? or xfail all the stuff missing? just fail?
@pytest.mark.parametrize('kws', FITANDPLOT_MODEL_KW_LIST, ids=format_model_params)
# TODO is request (magic variable w/ metadata about context test called from) name
# available by default, or need to mark fn as a @pytest.fixture to get it?
def test_fitandplot_repro(tmp_path, orn_deltas, kws, request):
    """
    NOTE: input (reference prior outputs) for a given parametrization of this test
    should be a directory named the same as the part of the test ID within brackets (the
    part generated by `format_model_params`).

    All reference outputs are saved under a date-stamped (and numbered within-day, if
    needed, to avoid duplicates) subdirectory of `reference_output_root`, generated by a
    prior run of `./.generate_reference_outputs_for_repro.py`. This test will use only
    outputs from the latest subdirectory of `reference_output_root`.
    """
    plot_root = tmp_path

    # TODO get this from a fixture (some global one, to not have to specify explicitly?
    # or just precompute at module level?)
    reference_output_dir = get_latest_reference_output_dir()

    # TODO why is there still a subdir called 'default'? does that correspond to
    # empty dict in MODEL_KW_LIST? does format_model_params do that explicitly, or what
    # triggers it?

    # https://stackoverflow.com/questions/56466111
    test_id = request.node.callspec.id
    # TODO remove this check? or only compute this way, rather than using magic request
    # fixture?
    assert format_model_params(kws) == test_id

    ref_model_output_dir = reference_output_dir / test_id

    # this works for symlink ref_model_output_dir too, where is_dir() will only return
    # True if the target exists and is a directory then (no .resolve() needed for that).
    # all below currently also works when this is a symlink.
    if not ref_model_output_dir.is_dir():
        pytest.xfail(f'{ref_model_output_dir=} did not exist\n'
            'maybe this case failed (or was not included) when generate script run?'
        )

    # TODO delete? like the pytest.xfail above better? (i assume so)
    #assert ref_model_output_dir.is_dir(), (f'{ref_model_output_dir=} did not exist\n'
    #    'maybe this case failed (or was not included) when generate script run?'
    #)

    # TODO also need to deal with marks here (where else do i already?)? could get from
    # request if not in kws (but probably is in kws)

    # TODO want to predetect dirs w/ only partial output (probably...), or just let fail
    # as we get there? (currently one-row-per-claw_True is one such dir)
    # TODO already have some fn for asserting we have some minimum set of files?
    # refactor to make one, if not?

    # TODO also first check just wPNKC, via call to connectome_WPNKC, or just handle all
    # via call to fit_and_plot_mb_model (will prob at least start with latter...)

    expected_missing_csv_keys = get_expected_missing_csv_keys(kws)

    # can hardcode this to include extra kwargs needed to repro certain cases, if needed
    extra_repro_kws = dict()
    expected_missing_keys = tuple()
    if len(extra_repro_kws) > 0:
        warn(f'hardcoding {extra_repro_kws=} to reproduce fit_and_plot_mb_model '
            'outputs! regen outputs after removing need for this hardcode!'
        )
        expected_missing_keys = tuple(extra_repro_kws.keys())

    params = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, **kws,
        **extra_repro_kws
    )
    output_dir = (plot_root / params['output_dir']).resolve()

    # TODO keep?
    assert_param_csv_matches_returned(plot_root, params,
        expected_missing_csv_keys=expected_missing_csv_keys
    )

    params2 = read_params(ref_model_output_dir)

    assert_fit_and_plot_outputs_equal(plot_root, params, params2,
        plot_root2=reference_output_dir,
        # TODO not actually sure this works...
        # TODO TODO keep all these?
        expected_missing_keys=expected_missing_keys + expected_missing_csv_keys
    )


# TODO care to understand the difference between the output i had then and what i'm
# getting now? meaningfully different?
#
# TODO maybe delete anyway, once i get generate_reference_outputs_for_repro.py +
# corresponding repro test in here established?
# TODO also delete any data i might have committed / added for this
'''
# TODO xfail this one until i get can it to work?
#@pytest.mark.xfail(
#    # TODO what's reason actually? not sure...
#    #reason='`one_row_per_claw and not prat_claws` path currently broken'
#    run=False
#)
def test_connectome_wPNKC_repro():
    # TODO save outputs + test repro for other param options (could prob be handled as
    # part of fit_mb_model tests, if i add output saving + repro verifying tests for
    # that...)

    wPNKC_dir = test_data_dir / ('prat-claws_True__-wPNKC-one-row-per-claw_True__'
        'connectome-APL_True__pn-claw-to-APL_True__target-sp_0.1_wPNKC_ref'
    )
    # TODO use w/ other assertion fns that check param dicts? (maybe only if enough in
    # these to be worth checking [i.e. also not too much missing...]?)
    params = read_params(wPNKC_dir)

    # TODO delete? still need after get_connectome_wPNKC_params?
    if params.get('one_row_per_claw'):
        prat_claws = params.get('prat_claws')
        assert prat_claws, ("otherwise would need to specify Tianpei's synapse_con_path"
            f'+synapse_loc_path, which prob would not be in params?\n{params=}'
        )

    wPNKC_params = get_connectome_wPNKC_params(params)

    wPNKC = connectome_wPNKC(**wPNKC_params)

    # assuming this will be portable enough to not need to use a CSV instead (which
    # would complicate [de]serialization. equal to a pickle serialized version, so just
    # using this.
    w1 = pd.read_parquet(wPNKC_dir / 'wPNKC.parquet')

    # TODO delete
    w2 = wPNKC.loc[:, w1.columns]
    w2 = w2[~ (w2 == 0).T.all()]
    assert w2.shape == w1.shape
    assert w1.columns.equals(w2.columns)

    # TODO TODO TODO what's mismatch? (shapes at least were same when initially getting
    # this mismatch, but may have changed more things since)
    # ipdb> w1.index.equals(w2.index)
    # False
    # ipdb> w1.sort_index().index.equals(w2.sort_index().index)
    # False

    # ipdb> w2.sort_index().index.to_frame(index=False)[['kc_id','claw_id']].equals(
    #    w1.sort_index().index.to_frame(index=False)[['kc_id','claw_id']])
    # False
    # ipdb> w2.sort_index().index.to_frame(index=False)[['kc_id','claw_id']
    #  ].index.equals(
    #  w1.sort_index().index.to_frame(index=False)[['kc_id','claw_id']].index
    # )
    # True

    # TODO delete
    breakpoint()
    #

    # TODO TODO will need to exlude renumbered CLAW_ID from comparison (which i'm
    # dropping in future versions, as it would cause confusion and prevent alignment
    # with other datasets w/ claw IDs [e.g. APL<>KC stuff]) (should rename
    # 'anatomical_claw'->CLAW_ID for future versions)

    # TODO TODO TODO why is this failing? was it that certain stuff happened
    # before/after glomerulus renames in one_row_per_claw=True cases? something simple?
    # responses same, despite this?
    # (yea, seems VC3 is present in wPNKC after refactor, but wasn't before. probably
    # should be there, so probably was a mistake earlier)
    assert w1.equals(wPNKC)
'''


# TODO TODO TODO either pass hardcode_initial_sp=True (which should fix failure, or
# regen output) (or just let *_fitandplot_repro tests replace this? any reason not to?)
@pytest.mark.xfail(
    reason='would need to regen outputs with hardcode_initial_sp=True at least. may '
    'remove test (should be replaced by some of *_fitandplot_repro cases?)',
    run=False
)
def test_connectome_APL_repro(orn_deltas):
    # TODO doc purpose of test
    # TODO doc how to generate test data for this (+ move to script, along w/ other test
    # data generation)

    kws = dict(use_connectome_APL_weights=True)

    responses, spike_counts, wPNKC, params = _fit_mb_model(orn_deltas=orn_deltas,
        return_olfsysm_vars=True, **kws
    )
    pks, thr = get_pks_and_thr(params, spike_counts)

    # NOTE: these outputs were saved by part of test_fixed_inh_params above, by enabling
    # some commented/hardcode-disabled code in there.
    # unscaled_per-KC_w[APLKC|KCAPL].csv were saved outputs of a connectome_APL_weights
    # call, w/ same arguments (before scaling mean to 1 or anything).
    data_dir = test_data_dir / 'connectome_APL_repro'

    pks2 = pd.read_csv(data_dir / 'pks.csv', index_col=[KC_ID, KC_TYPE])
    assert pd_allclose(pks, pks2)

    thr2 = pd.read_csv(data_dir / 'thr.csv', index_col=[KC_ID, KC_TYPE]).squeeze()
    assert pd_allclose(thr, thr2)

    # TODO replace check of this one by saving output of a fit_and_plot* call, and a
    # check against that output
    spike_counts2 = pd.read_csv(data_dir / 'spike_counts.csv',
        index_col=[KC_ID, KC_TYPE]
    )
    assert pd_allclose(spike_counts, spike_counts2)

    # these should have the same index, so technically we could just load one, since
    # planning to replace all values with 0 anyway...
    wAPLKC0 = pd.read_csv(data_dir / 'unscaled_per-KC_wAPLKC.csv', index_col=KC_ID)
    wKCAPL0 = pd.read_csv(data_dir / 'unscaled_per-KC_wKCAPL.csv', index_col=KC_ID)

    wAPLKC0 = pd.Series(data=0, index=wAPLKC0.index)
    wKCAPL0 = pd.Series(data=0, index=wKCAPL0.index)

    # TODO add separate test that we can use _wAPLKC/_wKCAPL to repro output, if we pass
    # in (mean->1 scaled?) wAPLKC/wKCAPL ouput Series?
    responses, spike_counts, wPNKC, params = _fit_mb_model(orn_deltas=orn_deltas,
        return_olfsysm_vars=True, _wAPLKC=wAPLKC0, _wKCAPL=wKCAPL0, **kws
    )
    assert params['wAPLKC_scale'] == 1
    assert params['wKCAPL_scale'] == 1
    pks, thr = get_pks_and_thr(params, spike_counts)
    assert pd_allclose(pks, pks2)
    assert pd_allclose(thr, thr2)
    # should actually be able to check spike_counts, as apl tuning should have been
    # disabled in fit_mb_model when _wAPLKC & _wKCAPL were hardcoded, and also the
    # weights were actually different (all 0, instead of actual mean-1-scaled connectome
    # APL weights)
    # TODO assert response rate is still ~2x (or whatever factor) target?
    assert not pd_allclose(spike_counts, spike_counts2)


def test_equalize_kc_type_sparsity(tmp_path, orn_deltas):
    """
    Tests equalize_kc_type_sparsity=True output can be reproduced by setting
    fixed_thr to appropriate vector (with a threshold for each KC) and wAPLKC as before.
    """
    # TODO TODO also test that per-kc_type sparsities are all within target now?
    # (or just the pre-APL ones? so are there any real guarantees i can assert here?)
    # (or else maybe rename this test to include '_fixed_inh_params' as suffix,
    # indicating narrower scope of test [/combine w/ another, more general, test of
    # fixed inh params])

    # TODO also test w/ below + use_connectome_APL_weights=True? (prob not super
    # critical. shouldn't interact much w/ this)
    kws = dict(
        equalize_kc_type_sparsity=True, ab_prime_response_rate_target=0.2
    )
    plot_root = tmp_path

    param_dict = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, **kws)
    output_dir = (plot_root / param_dict['output_dir']).resolve()

    #           2h @ -3  IaA @ -3  pa @ -3  ...  1-6ol @ -3  benz @ -3  ms @ -3
    # kc_id         ...
    # 0             0.0       0.0      0.0  ...         0.0        0.0      0.0
    # 1             0.0       0.0      0.0  ...         0.0        0.0      0.0
    # ...           ...       ...      ...  ...         ...        ...      ...
    # 1835          1.0       1.0      2.0  ...         1.0        0.0      0.0
    # 1836          0.0       0.0      0.0  ...         0.0        0.0      0.0
    df = _read_spike_counts(output_dir)

    # TODO delete, i've since change this. assert it's an array w/ more than one unique
    # value?
    #
    # not much probably relies on this, so could change. currently it's set to None
    # mainly to be obvious that we no longer have the scalar float fixed_thr here
    # (after tuning thr w/in each kc_type)
    #assert param_dict['fixed_thr'] is None

    # TODO delete
    #
    # this must be in param_dict if equalize_kc_type_sparsity=True
    type2thr = param_dict['type2thr']

    wPNKC = pd.read_pickle(output_dir / 'wPNKC.p')
    kc_types = wPNKC.index.get_level_values(KC_TYPE)
    assert not kc_types.isna().any()
    assert set(kc_types) == set(type2thr.keys())
    cell_thrs = kc_types.map(type2thr)
    fixed_thr = cell_thrs.values.copy()
    #

    # TODO delete above code. seems this assertion is passing.
    thr_and_apl_params = get_thr_and_APL_weights(param_dict, kws)
    assert np.array_equal(thr_and_apl_params['fixed_thr'], fixed_thr)

    fixed_kws = {k: v for k, v in kws.items()
        if k not in ('equalize_kc_type_sparsity', 'ab_prime_response_rate_target')
    }

    param_dict2 = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
        **{**thr_and_apl_params, **fixed_kws}
    )
    # TODO any assertions on contents of param_dict2 [vs param_dict?]? (e.g. on
    # fixed_thr)

    output_dir2 = (plot_root / param_dict2['output_dir']).resolve()
    assert output_dir2.is_dir()
    # TODO replace w/ just checking param_dict2['output_dir'] is a str with no path sep?
    # (+ refactor to share w/ other places that deal w/ these)
    assert output_dir2.parent == plot_root
    df2 = _read_spike_counts(output_dir2)

    assert df.equals(df2)


# TODO TODO TODO fix (broken as of 2026-02-13)
# assert (sc1[~mask.values] <= sc2[~mask.values]).all().all()
def test_multiresponder_APL_boost(orn_deltas):
    # ipdb> mask
    # kc_id
    # 300968622     False
    # 301309622     False
    # 301314150     False
    # 301314154     False
    # 301314208     False
    #               ...
    # 5901202076    False
    # 5901202102    False
    # 5901205770    False
    # 5901207528    False
    # 5901225361    False
    # Length: 1830, dtype: bool
    #
    # ipdb> mask.sum()
    # 139
    mask = pd.read_csv(test_data_dir / 'kiwi-control-multiresponder-mask.csv'
        ).set_index(KC_ID).squeeze()
    mask.name = None
    # TODO also assert that we initially have some cells responding to multiple odors in
    # here? or at least some spikes at all?
    assert mask.sum() > 0

    # NOTE: first implementation only supports use_connectome_APL_weights=True case
    kws = dict(use_connectome_APL_weights=True, drop_kcs_with_no_input=False)

    _, sc1, _, params1 = _fit_mb_model(orn_deltas=orn_deltas, **kws)
    assert sc1.index.get_level_values(KC_ID).equals(mask.index)

    boost_factor = 3.0
    _, sc2, _, params2 = _fit_mb_model(orn_deltas=orn_deltas,
        multiresponder_APL_boost=boost_factor, _multiresponder_mask=mask, **kws
    )

    # TODO TODO any zero values inside mask? still, can that explain how that one cell
    # got increased response?
    assert params1['wAPLKC'][~mask.values].equals(params2['wAPLKC'][~mask.values])

    # NOTE: if i moved boosting pre-tuning, would not be true that
    # wAPLKC_scale/wKCAPL_scale would be the same. if i did add suport for that, may
    # still want to leave a path that keeps current post-tuning-APL-boost behavior.
    #
    # NOTE: if boost_wKCAPL=True/'only' (not default =False), then wKCAPL Series would
    # be expected to change (only in mask, as w/ wAPLKC)
    should_be_unchanged = ['wAPLKC_scale', 'wKCAPL_scale', 'wKCAPL']
    # TODO refactor this check? don't have a fn for this already (dict_equal?)?
    for k in should_be_unchanged:
        v1 = params1[k]
        v2 = params2[k]
        assert equals(v1, v2)

    # TODO also test including new boost_wKCAPL (w/ at least =False and =True values)
    assert params1['wAPLKC'][~mask.values].equals(params2['wAPLKC'][~mask.values])
    assert (params1['wAPLKC'][mask.values] * boost_factor).equals(
        params2['wAPLKC'][mask.values]
    )

    # TODO TODO TODO fix:
    # ...was that (cell, odor) pair just already non-responding or something?
    # ipdb> (sc1[~mask.values] > sc2[~mask.values]).sum()
    # TODO TODO TODO plot dynamics for this cell (for both calls) -> try to figure out
    # why it's responding with more spikes in second call
    # odor
    # 2h @ -3       0
    # IaA @ -3      0
    # pa @ -3       0
    # 2-but @ -3    0
    # eb @ -3       0
    # ep @ -3       0
    # aa @ -3       0
    # va @ -3       0
    # B-cit @ -3    0
    # Lin @ -3      1
    # 6al @ -3      0
    # t2h @ -3      0
    # 1-8ol @ -3    0
    # 1-5ol @ -3    0
    # 1-6ol @ -3    0
    # benz @ -3     0
    # ms @ -3       0
    #
    # TODO TODO is it really true that we would never expect any of this?
    # limit max # allowed instead? (just 1 here)
    # TODO TODO TODO was APL weight for that cell already non-zero?
    #
    # responses in other cells might get slightly elevated, b/c of decreased APL
    # activity caused by decreased activity among the multiresponders. seemed to be
    # pretty subtle.
    assert (sc1[~mask.values] <= sc2[~mask.values]).all().all()

    # TODO check at least some fraction has had spike counts reduced (or was already 0
    # for that (odor, cell) pair)?
    # NOTE: not all (cell, odor) pairs w/ non-zero spikes have had # of spikes reduced
    # (but at least none should be increased)
    assert (sc1[mask.values] > sc2[mask.values]).any().any()

    assert (sc1[mask.values].sum().sum() * .75) > sc2[mask.values].sum().sum()

    # TODO TODO also test w/o connectome APL? work in that case, given how currently
    # implemented? (if not, restore assertion triggering in !connectome_APL & boost_APL
    # case, in fit_mb_model)


# TODO TODO test that confirms scaling wPNKC doesn't change output (not sure it's even
# true...) (w/ appropriate tolerance on output check + tuning params) (to maybe justify
# using all synapses for wPNKC, instead of trying to group/cluster into claws)
# TODO what modifications would i need to just pass in wPNKC like this? just use the
# fit_mb_model call directly? olfsysm directly (maybe trying to serialize all other
# parameters from a call invoked normally?)

# TODO TODO move much/all of al_analysis/model_test.py into (separate) tests in here
# (started to do that w/ test_hemibrain_matt_repro, but need to finish)
# (anything i still care about in there?)

# TODO test that the parameter i added to control what fraction of sparsity comes from
# threshold vs APL limits work as expected

# TODO test showing that fixed_thr only depends on odor panel, and that wAPLKC (at
# least, if tuning converges within same number of steps) only depends on:
# (initial value, mp.kc.sp_lr_coeff, and number of tuning iterations)?
# show that we can change final wAPLKC (e.g. across
# use_connectome_APL_weights=True/False) if we set sp_acc such that they take a
# different number of iterations to finish tuning?

# TODO add a test that (at least in cases like hemibrain one, not
# variable_n_claw=True) order of glomeruli in wPNKC do not matter
# (nor order of KCs in any of those inputs). could use _wPNKC= to hardcode those for
# that test?

# TODO TODO test that (at least some things, and at least compared to matt's model) are
# same between outputs from ann's matlab model and (some versions of our model, w/
# hopefully minor config changes). might only be able to easily test hemibrain, and then
# would still need to check how ann is making wPNKC (and maybe changing our code to
# allow it to behave like hers). not sure i could as easily test any non-deterministic
# stuff (e.g. pn2kc_connections='uniform', where RNG to form wPNKC prob would behave
# diff in matlab vs olfsysm), but could have ann's code generate the wPNKCs and then
# pass easy via new _wPNKC kwarg to fit_mb_model?

# TODO also check we can repro 2025-03-19 validation2 (hemibrain) outputs?
# 2025-02-19/validation2_hemibrain_model*.csv(s)? what are the CSVs i should check
# against?
def test_hemibrain_paper_repro(tmp_path):
    """
    Tests that, starting from committed estimated-ORN-spike-deltas, we can reproduce
    paper hemibrain model outputs (at least in terms of spike counts, which are among
    committed outputs).

    `tmp_path` is a pytest fixture for a temporary path. Will be saved under a subdir of
    `/tmp` that pytest creates (currently under `/tmp/pytest-of-<user>` for me).
    Contents would by default be preserved (until `/tmp` is cleared by restarting), but
    I've configured pytest (via `pyproject.toml`) to remove each of these after each
    test finishes (as contents can be quite large in total; enough to cause disk space
    issues).
    """
    orn_deltas = paper_megamat_orn_deltas()

    kws = dict(
        # I would not normally recommend you hardcode any of these except perhaps
        # weight_divisor=20. The defaults target_sparsity=0.1 and
        # _drop_glom_with_plus=True should be fine.
        target_sparsity=0.0915, weight_divisor=20, _drop_glom_with_plus=False,
        # TODO (delete? should be fixed now) are drop_kcs_with_no_input and
        # _drop_glom_with_plus still doing what they were before? currently getting 1828
        # (instead of paper 1837) cells in wPNKC)
        drop_kcs_with_no_input=False,
        # TODO check we actually need hardcode_intial_sp=True
        hardcode_initial_sp=True
    )

    plot_root = tmp_path

    # TODO modify this fn so dirname includes all same params by default (rather than
    # just e.g. param_dir='data_pebbled'), as the ones i'm currently manually creating
    # by calls in model_mb_... (prob behaving diff b/c e.g.
    # pn2kc_connections='hemibrain' is explicitly passed there)
    param_dict = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, **kws)

    output_dir = plot_root / param_dict['output_dir']
    assert output_dir.is_dir()
    assert output_dir.parent == plot_root

    # TODO convert to loading the CSVs instead (to future proof this test). should also
    # be committed.
    paper_wPNKC = pd.read_pickle(paper_hemibrain_output_dir / 'wPNKC.p')
    wPNKC = pd.read_pickle(output_dir / 'wPNKC.p')
    #

    paper_wPNKC = paper_wPNKC.rename_axis(index={'bodyid': KC_ID})
    assert paper_wPNKC.index.names == [KC_ID]

    assert (paper_wPNKC == 0).T.all().sum() == 9
    n_kcs_no_input = (wPNKC == 0).T.all().sum()
    assert n_kcs_no_input == 9, (f"{n_kcs_no_input=} != paper_wPNKC's 9. if 0, is "
        'drop_kcs_with_no_input=False flag working?'
    )

    assert set(wPNKC.columns) == set(paper_wPNKC.columns)
    assert wPNKC.columns.equals(paper_wPNKC.columns)

    wPNKC2 = wPNKC.droplevel(KC_TYPE)
    assert wPNKC2.index.equals(paper_wPNKC.index)

    assert wPNKC2.equals(paper_wPNKC)

    # {'fixed_thr': 268.0375322649455, 'wAPLKC': 4.622950819672131, 'wKCAPL':
    # 0.0025165763852325156, 'sp_acc': 0.1, 'max_iters': 10, 'sp_lr_coeff': 10.0,
    # 'apltune_subsample': 1, 'tuning_iters': 1}
    a1 = read_tuned_params(paper_hemibrain_output_dir)

    # {'fixed_thr': 268.0375322649456, 'wAPLKC': 4.306010928961749, 'wKCAPL':
    # 0.002344045143691752, 'sp_acc': 0.1, 'max_iters': 10, 'sp_lr_coeff': 10.0,
    # 'apltune_subsample': 1, 'tuning_iters': 1}
    b1 = read_tuned_params(output_dir)

    # NOTE: does not seem to matter that we are not using allclose to check wKCAPL here
    # (since check_with_allclose currently overwrites default, rather than adding to it)
    # (so may want to stop checking wKCAPL that way by default, and go back to
    # explicitly specifying in which tests/cases that is necessary)
    check_with_allclose = ('fixed_thr',)
    assert_param_dicts_equal(a1, b1, check_with_allclose=check_with_allclose,
        # TODO what were new keys that now exist that caused me to now need to set this
        # flag here (previously only needed below)? should be fine tho. nbd.
        only_check_overlapping_keys=True
    )

    # TODO TODO replace all read_param_csv w/ read_params? (here and elsewhere)
    # (just need to make sure it can load old CSV, when that's all there is)
    a2 = read_param_csv(paper_hemibrain_output_dir)
    b2 = read_params(output_dir)
    # TODO TODO am i no longer including params for dff2spiking_*? is that a mistake?
    # as of 2025-10-12:
    # ipdb> a2.index.difference(b2.index)
    # Index(['dff2spiking_add_constant', 'dff2spiking_scaling_method',
    #        'dff2spiking_separate_inh_model', 'pn2kc_connections',
    #        'tune_on_hallem'],
    #       dtype='object')
    # ipdb> b2.index.difference(a2.index)
    # Index(['drop_kcs_with_no_input'], dtype='object')

    assert_param_dicts_equal(a2, b2, only_check_overlapping_keys=True,
        check_with_allclose=check_with_allclose
    )

    # NOTE: despite using read_pickle, this is a np.ndarray (shape (1837, 1))
    spont1 = pd.read_pickle(paper_hemibrain_output_dir / 'kc_spont_in.p')

    # this one is now a pd.Series, w/ similar KC index to new wPNKC
    spont2 = pd.read_pickle(output_dir / 'kc_spont_in.p')

    assert np.allclose(spont1.squeeze(), spont2)

    #           2h @ -3  IaA @ -3  pa @ -3  ...  1-6ol @ -3  benz @ -3  ms @ -3
    # kc_id         ...
    # 0             0.0       0.0      0.0  ...         0.0        0.0      0.0
    # 1             0.0       0.0      0.0  ...         0.0        0.0      0.0
    # ...           ...       ...      ...  ...         ...        ...      ...
    # 1835          1.0       1.0      2.0  ...         1.0        0.0      0.0
    # 1836          0.0       0.0      0.0  ...         0.0        0.0      0.0
    df = _read_spike_counts(output_dir)

    # TODO rename 'model_kc' -> KC_ID (='kc_id') in those committed outputs?
    # NOTE: older output still used 'model_kc' instead of KC_ID, for KC ID column
    idf = pd.read_csv(paper_hemibrain_output_dir / 'spike_counts.csv',
        index_col='model_kc'
    )

    # can't compare KC indices while paper_hemibrain_output_dir contents still has [0,
    # N-1] 'model_kc' index, instead of KC_ID(='kc_id') IDs from connectome
    # (so also can't just do one x.equals(y) check)
    assert len(idf) == len(df)
    assert not idf.index.duplicated().any()
    assert not idf.index.isna().any()

    df_ids = df.index.get_level_values(KC_ID)
    assert not df_ids.duplicated().any()
    assert not df_ids.isna().any()

    df1 = idf.rename_axis(index={'model_kc': KC_ID})
    df2 = df.reset_index(drop=True).rename_axis(index=KC_ID)
    #

    # odors are in same order
    assert df1.columns.equals(df2.columns)

    # KC index is same [0, N-1] after processing new index to match that of saved paper
    # data
    assert df1.index.equals(df2.index)

    assert df1.equals(df2)


# TODO also check against 2025-02-19/validation2_uniform_model*.csv(s)? (no megamat data
# under 2025-02-19)
def test_uniform_paper_repro(tmp_path):
    """Similar purpose to `test_hemibrain_paper_repro`, but for uniform wPNKC outputs.
    """
    orn_deltas = paper_megamat_orn_deltas()

    # TODO need .resolve() call? pytest only ever going to be called from repo root?
    sent_to_anoop = Path('data/sent_to_anoop').resolve()

    # TODO delete this one + use below
    # v2 dir outputs should never really have been used, and (from data/README.md)
    # "still had the offset in the dF/F -> est. spike delta fn, and thus still had
    # unnecessarily high KC correlations"
    may26_dir = sent_to_anoop / 'v2'
    #

    # TODO also check uniform_model_wPNKC_n-seeds_100.csv /
    # megamat_uniform_model_params.csv in same folder?
    #
    # contains close-to but probably not final hemibrain (wd20 came after), but what
    # should be final uniform responses.
    may29_dir = sent_to_anoop / '2024-05-16'

    # NOTE: it is actually 2024-05-16 that I can repro (at least for first 2 seeds, and
    # w/ _drop_gloms_with_plus=False, which does matters), and not v2.
    # _drop_gloms_with_plus=True does not work to reproduce either.
    #
    uniform_response_csv_name = 'megamat_uniform_model_responses_n-seeds_100.csv'
    paper_uniform_responses = may29_dir / uniform_response_csv_name
    assert paper_uniform_responses.exists()

    # TODO rename 'model_kc' -> KC_ID (='kc_id') in those committed outputs?
    # NOTE: older output still used 'model_kc' instead of KC_ID, for KC ID column
    pdf = pd.read_csv(paper_uniform_responses, index_col=['model_kc', 'seed'])
    # TODO share w/ n_megamat_odors elsewhere?
    assert len(pdf.columns) == 17

    params = read_param_csv(may29_dir / 'megamat_uniform_model_params.csv')

    kws = dict(
        pn2kc_connections='uniform', n_claws=7, target_sparsity=0.0915,
        # TODO check we actually need hardcode_intial_sp=True
        drop_kcs_with_no_input=False, hardcode_initial_sp=True,
    )

    plot_root = tmp_path

    # NOTE: this product() should give us all n_seeds=2 cases before the =100 ones
    for n_seeds, _drop_glom_with_plus in product([2], [False, True]):
        print(f'{n_seeds=}')
        print(f'{_drop_glom_with_plus=}')

        if _drop_glom_with_plus and n_seeds > 2:
            # neither committed sent_to_anoop output matches in
            # _drop_glom_with_plus=True case, when looking at first 2 seeds. no need to
            # run all the seeds there.
            continue

        param_dict = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
            # TODO actually need tune_on_hallem=False for some reason?
            n_seeds=n_seeds, _drop_glom_with_plus=_drop_glom_with_plus, **kws
        )
        output_dir = plot_root / param_dict['output_dir']
        assert output_dir.is_dir()
        assert output_dir.parent == plot_root

        # TODO TODO load dict pickle and use assert_param_dicts_equal? modify that to
        # also work w/ this series input (or convert to dict), if doesn't already, and
        # use that fn to check these?

        # if i do end up switching wAPLKC/wKCAPL handling to also pop them in n_seeds>1
        # case (similar to when dealing w/ vector APL weights in non-variable_n_claw
        # cases), will need to compare wAPLKC/wKCAPL to paper params from loading them
        # separately. currently trying to keep these list-of-floats in CSV, as seemed to
        # work for committed paper outputs.
        params2 = read_params(output_dir)

        if not _drop_glom_with_plus:
            # TODO replace w/ eval_and_check_[err|compatible|equal]? (or a wrapper that
            # also works w/ non-str input?) (eval requires str input) (/delete)
            # TODO switch handling? does de-serialize as a string, which needs eval'd...
            # (to array, or what?)
            wAPLKC = params['wAPLKC']
            wAPLKC2 = params2['wAPLKC']
            assert wAPLKC[:len(wAPLKC2)] == wAPLKC2

            wKCAPL = params['wKCAPL']
            wKCAPL2 = params2['wKCAPL']
            assert wKCAPL[:len(wKCAPL2)] == wKCAPL2

            # TODO TODO check other parts of params (vs params2) as well
            # (use assert_param_dicts_equal?)
            # TODO delete
            #breakpoint()
            #

            # TODO TODO move this check before check of APL<>KC weights, as might
            # catch root cause earlier
            #
            # TODO actually check this against 2024-05-16 dir contents? that seems like
            # it might be dir to use anyway? (/delete)
            # (checking subset of responses below should be sufficient to check these
            # are also equal, at least under the vast majority of cases)
            #wPNKC = pd.read_pickle(output_dir / 'wPNKC.p')
            #

        # TODO change fit_and_plot... so there is still a seed level in n_seeds=1 case?
        # seed still available elsewhere for repro (param dict?)? prob not?
        df = _read_spike_counts(output_dir, index_col=[KC_ID, 'seed'])

        first_seeds = set(df.index.unique('seed'))
        assert len(first_seeds) == n_seeds

        # these committed responses were binarized. no spike counts in those two
        # sent_to_anoop dirs (v2 / 2024-05-16)
        df[df > 0] = 1

        pdf2 = pdf[pdf.index.get_level_values('seed').isin(first_seeds)]

        i2 = pd.MultiIndex.from_frame(
            df.groupby(level='seed', sort=False).cumcount().rename('model_kc'
                ).reset_index().drop(columns=KC_ID)
        ).reorder_levels(['model_kc', 'seed'])
        if not _drop_glom_with_plus:
            assert i2.equals(pdf2.index)

        # renumbering KC_ID (which are from connectome still in df, but not pdf) to
        # match [0, N-1] 'model_kc' IDs for pdf (within each seed)
        df.index = i2

        if not _drop_glom_with_plus:
            assert pdf2.equals(df)
        else:
            assert not pdf2.equals(df)

        print()


def test_hemibrain_matt_repro():
    # TODO delete (may want to commit some other things from here first tho)
    #matt_data_dir = Path('../matt/matt-modeling/data')
    #
    matt_data_dir = Path('data/from_matt')

    # NOTE: was NOT also committed under data/from_matt (like
    # hemibrain/halfmat/responses.csv below was)
    # TODO either delete this or also commit this file?
    #
    # TODO fix code that generated hemimatrix.npy / delete
    # (to remove effect of hc_data.csv methanoic acid bug that persisted in many copies
    # of this csv) (won't be equal to `wide` until fixed)
    # (what was hemimatrix.npy exactly tho? and why would odor responses matter for it?
    # is it not just connectivity?)
    #
    # Still not sure which script of Matt's wrote this (couldn't find by grepping his
    # code on hal), but we can compare it to the same matrix reformatted from
    # responses.csv (which is written in hemimat-modeling.html)
    #hemi = np.load(matt_data_dir / 'reference/hemimatrix.npy')

    # I regenerated this, using Matt's account on hal, by manually running all the
    # relevant code from matt-modeling/docs/hemimat-modeling.html, because it seemed the
    # previous version was affected by the hc_data.csv methanoic acid error.
    # After regenerating it, my outputs computed in this script are now equal.
    df = pd.read_csv(matt_data_dir / 'hemibrain/halfmat/responses.csv')

    # The Categoricals are just to keep order of odors and KC body IDs the same as in
    # input. https://stackoverflow.com/questions/57177605/
    df['ordered_odors'] = pd.Categorical(df.odor, categories=df.odor.unique(),
        ordered=True
    )
    df['ordered_kcs'] = pd.Categorical(df.kc, categories=df.kc.unique(), ordered=True)
    wide = df.pivot(columns='ordered_odors', index='ordered_kcs', values='r')
    del df

    # TODO delete? (has been commented basically since creation of this test)
    #assert np.array_equal(hemi, wide.values)
    #del hemi

    # TODO rename to run_model? or have a separate fn for that? take `mp` (and `rv` too,
    # or are even fit thresholds in mp?) as input (and return from fit_model?)?
    # TODO modify so i don't need to return gkc_wide here (or at least be more clear
    # about what it is, both in docs and in name)?
    responses, _, gkc_wide, _ = _fit_mb_model(tune_on_hallem=True,
        pn2kc_connections='hemibrain', _use_matt_wPNKC=True,
        # TODO check we actually need hardcode_intial_sp=True
        hardcode_initial_sp=True
    )
    assert gkc_wide.index.name == KC_ID
    assert np.array_equal(wide.index, gkc_wide.index)
    assert np.array_equal(responses, wide)


def test_step_around():
    # TODO do i want absolute step size the same for each or (what i assume is current
    # behavior of) same relative step size? something else?  (prob happy enough w/
    # "relative" as it is)

    # TODO do another call matching fixed_thr steps?
    #
    # checking i can recreate wAPLKC steps used in sens analysis for paper w/ remy
    s1 = step_around(4.6, param_lim_factor=5.0, n_steps=3,
        drop_negative=True
    )
    paper_wAPLKC_steps = [0, 4.6, 27.6]
    assert np.allclose(paper_wAPLKC_steps, s1)

    s2 = step_around(np.array([4.6, 4.6]), param_lim_factor=5.0, n_steps=3,
        drop_negative=True
    )
    col1 = s2[:, 0]
    col2 = s2[:, 1]
    assert np.array_equal(col1, col2)
    assert np.allclose(s1, col1)

    # TODO delete?
    s1 = step_around(4.6, param_lim_factor=1, n_steps=5,
        drop_negative=True, drop_zero=True
    )
    s2 = step_around(np.array([4.6, 1.3]), param_lim_factor=1, n_steps=5,
        drop_negative=True, drop_zero=True
    )
    assert (s1 > 0).all()
    assert (s2 > 0).all()
    #

    n_steps = 5
    kws = dict(n_steps=n_steps, drop_negative=True)

    x = 249.665

    eps = 1e-4
    nz_steps = step_around(x, 1.0 - eps, **kws)
    assert len(nz_steps) == n_steps
    assert nz_steps[0] > 0

    # first param_lim_factor[=2.0] should produce 2 <=0 values w/ n_steps=5
    for param_lim_factor in (2.0, 1.0):

        raw_steps = step_around(x, param_lim_factor, n_steps=n_steps,
            drop_negative=False
        )
        steps = step_around(x, param_lim_factor, **kws)
        # TODO similar check that works for 2d steps (for vector center)
        assert len(steps) + (raw_steps < 0).sum() == n_steps

        if param_lim_factor >= 1:
            # as long as drop_zero=False, which is default
            assert steps[0] == 0

        if param_lim_factor == 1:
            assert np.isclose(steps[-1], x * 2)

        # TODO similar cases w/ vector input? already covered well enough above?


# TODO TODO add test (that hits claw_sims code in olfsysm, and checks those dynamic
# outputs) that order of odors doesnt matter to claw_sims, for at least 2 odors
# (to double check claw_sims re-use [i.e. lack of odor indexing] in fit_sparseness isn't
# a problem)

@pytest.mark.parametrize('kws', [
    dict(),
    dict(use_connectome_APL_weights=True),
    dict(one_row_per_claw=True),
    # at least this case is now actually failing b/c different values.
    # TODO TODO can it (and others that will likely fail) be saved by either C++ code
    # tweaks or hardcoding fixed_thr/wAPLKC from prev call?
    dict(one_row_per_claw=True, use_connectome_APL_weights=True),
], ids=format_model_params)
# TODO TODO also hardcode fixed_thr/wAPLKC (from first call) to shortcut second call's
# tuning? or would that degrade test? or do i actually need that for test to work
# (hopefully not?)?
def test_btn_expansion(tmp_path, orn_deltas, kws):
    # TODO doc

    # TODO delete this? should be default anyway
    connectome = 'hemibrain'

    # TODO possible to replace tianpei's path w/ prat_claws=True here? not sure i care.
    # may want separate test boutons are placed in separate format (in wPNKC) across
    # prat_boutons=True and Btn_separate=True cases tho?

    # TODO delete
    print()
    print(f'{kws=}')
    #

    # TODO TODO set this to new default of sp_lr_coeff after settling? (then hardcode
    # old default for select repro tests)
    # 0.5 is far too low. 1.0 also too low. default was 10 actually. 2.0 still too low
    # (I think? check again?). oscillating again at 5, and 3.0 (which didn't terminate
    # after 15 iterations)
    kws = dict(orn_deltas=orn_deltas, pn2kc_connections=connectome, sp_lr_coeff=2.5,
        max_iters=20, **kws
    )

    Btn_num_per_glom = 3

    # TODO run (a slow version of?) this test w/ this + plot_example_dynamics=True?
    # (currently getting killed saving claw_sims on 2nd call tho... maybe only after
    # adapting to subset early, and/or only work off an olfsysm-created disk copy)
    #plot_dir1 = tmp_path / 'Btn-separate_True'
    #plot_dir1.mkdir()
    plot_dir1 = None
    responses1, spike_counts1, wPNKC1, param_dict1 = _fit_mb_model(
        Btn_separate=True, Btn_num_per_glom=Btn_num_per_glom, plot_dir=plot_dir1, **kws
    )

    #plot_dir2 = tmp_path / 'Btn-separate_False'
    #plot_dir2.mkdir()
    plot_dir2 = None
    responses2, spike_counts2, wPNKC2, param_dict2 = _fit_mb_model(
        plot_dir=plot_dir2, **kws
    )

    # TODO TODO also save return (limited?) dynamics, and assert they are of correct
    # shape? or did tianpei never actually have bouton dynamics saved (or even
    # simmed) separately? (want to test there is actually something internal in
    # model of this shape, other than wPNKC...)
    # TODO add option to dynamics/saving stuff to only do so for first 1-2 cells /
    # odors, to save on space?

    # the responses and spike_count matrices should be the same
    assert responses1.equals(responses2)
    assert spike_counts1.equals(spike_counts2)

    # wPNKC is not the same, but row-wise sum should be the same.
    # test row wise sum is the same;
    # (use allclose since wPNKC1 have doubles and wPNKC2 have integers)
    assert np.allclose(wPNKC1.sum(axis='columns'), wPNKC2.sum(axis='columns'))

    summed_wPNKC1 = wPNKC1.groupby(level='glomerulus', axis='columns').sum()
    assert np.allclose(summed_wPNKC1, wPNKC2)

    # TODO TODO may need to update this, if i end up moving his bouton IDs to
    # another component of row index instead (if that ends up being easiest way to
    # get useful output for prat_boutons=True outputs through model)
    # TODO TODO probably at least change his name for this level to BOUTON_ID
    # (something will change, when refactoring to make his path and
    # prat_boutons=True paths consistent, and working to get olfsysm going with
    # those outputs)
    assert wPNKC1.columns.names == [glomerulus_col, BOUTON_ID]
    assert wPNKC2.columns.names == [glomerulus_col]
    assert wPNKC1.index.equals(wPNKC2.index)
    assert len(wPNKC1.columns) == Btn_num_per_glom * len(wPNKC2.columns)

    # TODO how are wAPLKC/wKCAPL handled in this case? are they also expanded?
    # if not, how is that handled? ig they wouldn't be (since PN<>APL stuff wansn't
    # really implemented at the time [well it was, but not in the way i wanted at
    # least])?


# TODO TODO add test of new olfsysm dynamics saving code (check equiv to what i'm
# currently getting [for single odors, at least] through pybind11) (-> then update to
# save all odors into single files, if possible)

# TODO add test that we can actually run sensitivity analysis path of fit_and_plot*
# (for all MODEL_KW_LIST?)

# TODO TODO add test that the panel2tuning_panel section of model_mb_responses
# works for most param combos (including some i haven't gotten working yet there, like
# one-row-per-claw variants) (required to run model on kiwi/control, given i've been
# tuning model on both panels, and then running separately on each for a while now)
# (what do i need beyond checking what test_fixed_inh_params_fitandplot does? just add
# anything else i might need into that test?)
# TODO TODO will probably need to factor out some of that section, so i can test it, but
# would prob be good anyway

# TODO TODO add test that inh/Is (if we set return_dynamics=True) are non-zero for claw
# & apl_coup_const variants of model (and for all elements, or at least all within odor
# window [perhaps off by ~1])? unclear to me that tianpei's current code is initializing
# those variables correctly (though the overall calculation probably is correct, at
# least insofar as we currently have tests covering those variants)

# TODO TODO also test that inh/Is are distinct for apl_coup_const versions of model
# (and that returned values have a component of shape equal to # of compartments there)

# TODO TODO add test that we can recreate use_connectome_APL_weights=False path by
# hardcoding all 1 (or all some constant. may need to use const def from olfsysm?)
# for _wAPLKC and _wKCAPL

# TODO add test that target_sparsity_factor_pre_APL is working, both w/ default of 2.0,
# as well as maybe 1-2 other reasonable values (two calls, one w/
# `mp.kc.tune_apl_weights = False`? might be easiest way unless i end up implementing
# sparsity calculation + saving both before and after APL)
# (see some other comments circa sp_factor_pre_apl code in mb_model.fit_mb_model)

# TODO TODO add test we can recreate connectome_APL_weights=False output w/
# connectome_APL_weights=True path (really, the preset_wAPLKC/similar paths in olfsysm)
# if we set all of the preset weights to be either 1 or all uniform w/ mean of 1
# (and which of those two options is correct?)

# TODO add test we can manually pass in hallem inputs and get same results as using some
# of the calls that don't pass in orn_deltas (may need to specify tune_on_hallem=True?
# or not if orn_deltas not passed?)

# TODO do variable_n_claws cases support sensitivity_analysis? is it just a matter of
# tracking some of the metadata getting cumbersome, as to why i decided not to support
# it? add test confirming failure is at least as expected (presumably w/ reasonably
# information error message), if i try sensitivity_analysis=True in one of those cases?

# TODO add tests checking the key parts of example scripts for sam / ruoyi / george /
# yang all still work (how to organize?)

# TODO test we can start tuning at arbitrary tuning iters (w/ weights + scale params
# set appropriately), and still get to same final output

# TODO need scope='function' here? can still share params (w/ one cached output for
# each? if not?)
@pytest.fixture(scope='function')
# TODO rename since this is actually mainly model outputs (but also should have weight
# vectors initialized in params)
# TODO separate fixture for just the weights later?
def apl_weights(orn_deltas, request):
    kws = request.param

    # TODO TODO TODO move this intial part (defining weights at least, probably not
    # initial rv? maybe it's fine as long as only one downstream test ever uses that
    # rv?) to a fixture (-> share w/ two new tests from remaining section)
    precalc_weights = False
    # TODO implement + test
    warn('fix precalc_weights=True (to speed up this test in general, and mainly to '
        'fix whatever underlying consistency issue!'
    )
    #precalc_weights = True
    if precalc_weights:
        # TODO delete after saving/committing/replacing these with use of those outputs?
        # unless flag set, maybe?
        wPNKC_params = get_connectome_wPNKC_params(kws)
        # NOTE: one_row_per_claw is not an argument to connectome_wPNKC, nor is
        # use_connectome_APL_weights
        # TODO TODO change how this adds a connectome='hemibrain' we don't have as
        # input? why can't i just let that be handled by default? (any other uses
        # currently depend on that? just check no tests fail after removing [once less
        # tests in general are failing...])
        #assert all(x in kws for x in wPNKC_params)
        wPNKC = connectome_wPNKC(**wPNKC_params)

        assert kws['use_connectome_APL_weights']

        # TODO neither fn below explicitly takes this, so could delete if i end up
        # changing so it's implied by e.g. prat_claws=True (or even if it's the default)
        assert kws['one_row_per_claw']
        # TODO delete
        print(f'{wPNKC_params=}')
        #

        # TODO would probably be easier to save all these w/ just a fit_and_plot... call
        # TODO TODO TODO still need a way to return these weights if precalc...=True
        # (currently only returning params) (just separate fixture? prob used by this
        # one, if so)
        wAPLKC, wKCAPL, wAPLPN, wPNAPL = connectome_APL_weights(wPNKC=wPNKC,
            **wPNKC_params
        )
        # TODO delete
        print('NOT CURRENTLY ACTUALLY RETURNING THESE WEIGHTS! FIX!')
        breakpoint()
        #

        # TODO TODO TODO fix
        # TODO TODO TODO why does this not work, but below ret2 call (using
        # get_thr_and_APL_weights output) does work? compare all weight matrices and
        # other params between the two calls?
        ret = _fit_mb_model(orn_deltas=orn_deltas, return_olfsysm_vars=True,
            _wPNKC=wPNKC, _wAPLKC=wAPLKC, _wKCAPL=wKCAPL, _wAPLPN=wAPLPN,
            _wPNAPL=wPNAPL, **kws
        )
        # TODO TODO TODO wtf is the problem here? why is nothing matching?
        # TODO delete (after getting working)
        ret2 = _fit_mb_model(orn_deltas=orn_deltas, return_olfsysm_vars=True,
             **kws
        )
        assert_fit_outputs_equal(ret, ret2)
        breakpoint()
        #
    else:
        ret = _fit_mb_model(orn_deltas=orn_deltas, return_olfsysm_vars=True,
            **kws
        )

    return ret, kws


# TODO (below should be start of these. delete)
#
# TODO add test that we can vary each of wAPLKC/wKCAPL/wPNAPL/wAPLPN individually, all
# having different effects but all in the direction we expect
# TODO add test that current APL<>PN weight tuning procedure has PN<>APL weights doing a
# reasonable amount (bouton dynamics should not just like like PN, and setting weights
# to 0 after should increase response rate) (not all in direction i thought...  not sure
# if bug or not yet)
#
@pytest.mark.parametrize('apl_weights', BOUTON_MODEL_KW_LIST, ids=format_model_params,
    indirect=['apl_weights']
)
def test_apl_weights_osm(apl_weights):
    # TODO make sure this test disables all plotting in calls it makes, or at least any
    # that uses sns/scipy, so we can remove that import (to the extent it remains
    # incompatible w/ setting PYTHONMALLOC=malloc for certain debug tools)
    ret, kws = apl_weights

    params = ret[-1]
    wAPLKC = params['wAPLKC']
    wKCAPL = params['wKCAPL']
    wAPLPN = params['wAPLPN']
    wPNAPL = params['wPNAPL']

    rv = params['rv']
    mp = params['mp']

    # TODO TODO TODO check we can get same effects on other calls setting each of the
    # respective scalar parameters to 0?
    # TODO TODO TODO so should i combine this test back with the *_fitmbmodel one?
    #
    # TODO TODO TODO also [maybe more importantly], test we can get same outcomes by
    # setting either these >0 vectors [or *_scale params? may not work...] vs scalar
    # wAPLKC/etc params in here. test we can hardcode each individually.
    # (-> adapt above test into sweep checking for cases w/ more KC-like odor-odor
    # correlations)
    sp0 = np.mean(rv.kc.responses)

    # TODO (delete) calculated values below ever very slightly above (or equal to) this?
    # (don't think so)
    #
    # TODO TODO also include this factor (what values / which direction to step?) in a
    # sweep?
    # default sp_factor_pre_apl=2.0
    sp_max = mp.kc.sp_factor_pre_apl * mp.kc.sp_target
    # TODO delete
    print()
    print(f'{wAPLKC.mean()=}')
    print(f'{wKCAPL.mean()=}')
    print(f'{wAPLPN.mean()=}')
    print(f'{wPNAPL.mean()=}')
    #

    ak_scale = rv.kc.wAPLKC_scale
    ka_scale = rv.kc.wKCAPL_scale
    ap_scale = rv.pn.wAPLPN_scale
    pa_scale = rv.pn.wPNAPL_scale
    # TODO delete?
    print()
    print(f'{ak_scale=}')
    print(f'{ka_scale=}')
    print(f'{ap_scale=}')
    print(f'{pa_scale=}')
    #
    thr_and_apl_kws = get_thr_and_APL_weights(params, kws)
    # TODO TODO also assert for wKCAPL and wPNAPL?
    # (would need to recalc wKCAPL and wPNAPL from other two, as currently calculated,
    # and not sure if i want to move that functionality into
    # get_APL_weights/get_thr_and... (maybe behind a flag?))
    assert ak_scale == thr_and_apl_kws['wAPLKC']
    assert ap_scale == thr_and_apl_kws['wAPLPN']

    # need to check all wAPLKC/etc.values all match values in rv currently
    # (otherwise, attempts at "restoring" won't work)
    ak = rv.kc.wAPLKC
    ka = rv.kc.wKCAPL
    ap = rv.pn.wAPLPN
    pa = rv.pn.wPNAPL
    # TODO delete
    print()
    print(f'{ak.mean()=}')
    print(f'{ka.mean()=}')
    print(f'{ap.mean()=}')
    print(f'{pa.mean()=}')
    #

    # TODO need copy here? (would *= below change orig if not?) try removing?
    # NOTE: this would be different if the code were using scaled wAPKC/etc weight
    # vectors (was written originally right after run_KC_sims calls in fit_mb_model,
    # where these were still unscaled vectors) ("unscaled" still has the normalization
    # to mean [with some current additional scaling for some, that i might want to
    # remove, based loosely on unit #], but NOT the tuning-based additional scale
    # factor)
    ak_arr = wAPLKC.values.reshape(ak.shape).copy()
    ka_arr = wKCAPL.values.reshape(ka.shape).copy()
    ap_arr = wAPLPN.values.reshape(ap.shape).copy()
    pa_arr = wPNAPL.values.reshape(pa.shape).copy()
    del ak, ka, ap, pa
    # TODO delete
    print()
    print(f'{ak_arr.mean()=}')
    print(f'{ka_arr.mean()=}')
    print(f'{ap_arr.mean()=}')
    print(f'{pa_arr.mean()=}')
    # TODO delete
    print()
    print(f'{rv.kc.wAPLKC_unscaled.mean()=}')
    print(f'{rv.kc.wKCAPL_unscaled.mean()=}')
    print(f'{rv.pn.wAPLPN_unscaled.mean()=}')
    print(f'{rv.pn.wPNAPL_unscaled.mean()=}')
    #

    # TODO delete
    print(f'{kws=}')
    #

    assert np.allclose(rv.kc.wAPLKC / rv.kc.wAPLKC_scale, rv.kc.wAPLKC_unscaled)
    assert np.allclose(rv.kc.wKCAPL / rv.kc.wKCAPL_scale, rv.kc.wKCAPL_unscaled)
    assert np.allclose(rv.pn.wAPLPN / rv.pn.wAPLPN_scale, rv.pn.wAPLPN_unscaled)
    assert np.allclose(rv.pn.wPNAPL / rv.pn.wPNAPL_scale, rv.pn.wPNAPL_unscaled)

    # TODO delete. already was scaled.
    #ak_arr *= ak_scale
    #ka_arr *= ka_scale
    #ap_arr *= ap_scale
    #pa_arr *= pa_scale
    ## TODO delete
    #print()
    #print('after multiplying each by corresponding *_scale parameters:')
    #print(f'{ak_arr.mean()=}')
    #print(f'{ka_arr.mean()=}')
    #print(f'{ap_arr.mean()=}')
    #print(f'{pa_arr.mean()=}')
    #print()
    #

    assert np.allclose(ak_arr, rv.kc.wAPLKC)
    assert np.allclose(ka_arr, rv.kc.wKCAPL)
    assert np.allclose(ap_arr, rv.pn.wAPLPN)
    assert np.allclose(pa_arr, rv.pn.wPNAPL)

    assert np.allclose(ak_arr.squeeze(), wAPLKC)
    assert np.allclose(ka_arr.squeeze(), wKCAPL)
    assert np.allclose(ap_arr.squeeze(), wAPLPN)
    assert np.allclose(pa_arr.squeeze(), wPNAPL)

    # TODO TODO reorganize C++ to avoid need to set this true to avoid segfault?
    # TODO TODO TODO why did we get sample_PN_spont pn.sims[i].block ... check failing
    # w/ regen=True (in first call below)?
    #regen = True
    regen = False
    if regen:
        mp.kc.tune_apl_weights = False

    # TODO TODO refactor to share w/ fit_mb_model usage (have fn get_time_index now. use
    # that.)
    t0 = mp.time_pre_start
    t1 = mp.time_end
    n_samples = int(round((t1 - t0) / mp.time_dt))
    ts = pd.Series(name='seconds', data=np.linspace(t0, t1, num=n_samples))
    # TODO delete use of these int indices (use new label based slicing like in
    # mb_model)
    stim_start_idx = np.searchsorted(ts, mp.time_stim_start)
    stim_end_idx = np.searchsorted(ts, mp.time_stim_end)
    #

    # TODO TODO also plot non-normed APL? or assert APL inh/whatever is in
    # direction we expect too? breadth across KCs (vs max) (something else that might be
    # diff across KCs to account for sparsity diff?)?
    boutons = np.array(rv.pn.bouton_sims).copy()
    print(f'{boutons[:, :, stim_start_idx:stim_end_idx].mean()=}')

    # TODO delete. replaced by code that was moved to test_apl_weights_fitmbmodel (which
    # should be broken off into a script anyway. scripts/step_model_pn_apl.py)
    '''
    # starts at 1.937 in pn_claw_to_apl=True case
    print()
    print('stepping wAPLPN around tuned value:')
    for ap in [0.1, 20, 0.5, 10]:
        rv.pn.wAPLPN = ap * ap_arr.copy()
        # TODO TODO reword (and work from unscaled, so that numbers in list above could
        # directly be used as *_scaled param?)?
        warn(f'setting wAPLPN to {ap} * original vector')
        osm.run_KC_sims(mp, rv, regen)
        sp_ap_step = np.mean(rv.kc.responses)
        print(f'{sp_ap_step=}')
        print()
    rv.pn.wAPLPN = ap_arr.copy()

    # TODO TODO TODO also do for PNAPL
    print()
    print('stepping wPNAPL around tuned value:')
    for pa in [0.1, 20, 0.5, 10]:
        rv.pn.wPNAPL = pa * pa_arr.copy()
        # TODO TODO reword (and work from unscaled, so that numbers in list above could
        # directly be used as *_scaled param?)?
        warn(f'setting wPNAPL to {pa} * original vector')
        osm.run_KC_sims(mp, rv, regen)
        sp_pa_step = np.mean(rv.kc.responses)
        print(f'{sp_pa_step=}')
        print()
    rv.pn.wPNAPL = pa_arr.copy()
    # TODO delete
    breakpoint()
    '''
    #

    # TODO TODO TODO test that w/ high enough scale factor, we can indeed change
    # response rate with this one (in both pn_claw_to_apl=True/False)
    # TODO TODO TODO ...we can, just not in the direction i expected. why??? explain.
    warn('setting wAPLPN to 0')
    rv.pn.wAPLPN = np.zeros(rv.pn.wAPLPN.shape)
    osm.run_KC_sims(mp, rv, regen)
    sp_no_APLPN = np.mean(rv.kc.responses)
    print(f'{sp_no_APLPN=}')
    # TODO TODO TODO test that bouton_sims at least goes down (w/in odor window?)
    # TODO TODO get APL dynamics inh/Is, and check at least that is in direction we
    # expect? or can i not really place expectations on that either?
    # TODO delete
    if sp_no_APLPN <= sp0:
        print('SPARSITY STAYS SAME OR GETS SMALLER WITH APL>PN=0! FIX/EXPLAIN!')
    #

    # TODO assert these are all individually (elementwise) higher than `boutons` above?
    # or never less than, and all >=? (the mean in odor window is indeed larger, despite
    # lower KC response rate)
    boutons_no_APLPN = np.array(rv.pn.bouton_sims).copy()
    print(f'{boutons_no_APLPN[:, :, stim_start_idx:stim_end_idx].mean()=}')
    # TODO delete
    breakpoint()
    #
    print()
    # TODO TODO TODO why is this smaller? (not sure it is anymore. don't think it's
    # doing anything at all now...)
    # TODO TODO TODO sanity check w/ plots at least?
    # (in an earlier version of this code, i did get 0.138738 here, but maybe i made a
    # mistake there?) (or maybe that was w/ or w/o pn-claw-to-APL? check?)
    # (sp_no_APLPN=0.06483494090476838)
    # TODO TODO TODO is this implying that *more* overall PN activity leads to less KC
    # activity (presumably b/c broader APL input)???
    # TODO TODO make some plots of dynamics to sanity check this?
    # 2026-02-22: once again smaller, now:
    # sp_no_APLPN=0.05077435131096318 (in !pn-claw-to-APL case)
    # sp_no_APLPN=0.06802744192365168  (in pn-claw-to-APL case)
    # TODO TODO TODO fix
    #assert sp0 < sp_no_APLPN < sp_no_APL
    #
    rv.pn.wAPLPN = ap_arr.copy()

    warn('setting both wAPLKC and wAPLPN to 0')
    rv.kc.wAPLKC = np.zeros(rv.kc.wAPLKC.shape)
    rv.pn.wAPLPN = np.zeros(rv.pn.wAPLPN.shape)
    osm.run_KC_sims(mp, rv, regen)
    sp_no_APL = np.mean(rv.kc.responses)
    print(f'{sp_no_APL=} (same for both no APL cases)')
    print()
    rv.kc.wAPLKC = ak_arr.copy()
    rv.pn.wAPLPN = ap_arr.copy()

    warn('setting both wKCAPL and wPNAPL to 0 (restored weights from APL)')
    rv.kc.wKCAPL = np.zeros(rv.kc.wKCAPL.shape)
    rv.pn.wPNAPL = np.zeros(rv.pn.wPNAPL.shape)
    osm.run_KC_sims(mp, rv, regen)
    sp_no_APL2 = np.mean(rv.kc.responses)
    # if APL has no input, should have same effect as if all weights from APL are 0
    assert sp_no_APL == sp_no_APL2, (f'{sp_no_APL=} (0 weights FROM APL) != '
        f'{sp_no_APL2=} (0 weights TO APL)'
    )
    # these two numbers need atol=1e-4 for isclose to work (1e-5 doesn't)
    # ipdb> sp_no_APL
    # 0.1999728297785627
    # ipdb> sp_max
    # 0.2
    assert np.isclose(sp_no_APL, sp_max, atol=1e-4), \
        f'{sp_no_APL=} not close to {sp_max=}'
    rv.kc.wKCAPL = ka_arr.copy()
    rv.pn.wPNAPL = pa_arr.copy()

    # TODO (delete. no longer having seg fault issue that made me want to try this) try
    # w/ regen=True and setting tune_apl_weights=False?  is current =False 3rd arg
    # actually being respected anyway? i think my usual init of APL weights is actually
    # expected to happen in fit_sparseness. should probably move it out of there
    warn('setting wAPLKC to 0 (restored others)')
    # TODO this not working? dtype/shape issue? (doesn't seem like that)
    rv.kc.wAPLKC = np.zeros(rv.kc.wAPLKC.shape)
    # can this print to cout still? (yes)
    #
    # TODO TODO am i even able to re-run after changing weights like this?
    # TODO TODO TODO maybe if i regen but set tune_APL_weights=False (and configure so
    # weights now appear as fixed? any point to not just doing separate calls then?
    # speed, i guess? would that then be duplicate w/ other test?)
    osm.run_KC_sims(mp, rv, regen)
    # TODO TODO TODO TODO does it actually make sense that this is so high? can any
    # PN<>APL stuff alone change response rate at all? if not, why not?
    sp_no_APLKC = np.mean(rv.kc.responses)
    # sp_no_APLKC=0.1999728297785627
    print(f'{sp_no_APLKC=}')
    print()
    # TODO TODO TODO assert this one is less than sp_no_APL though (currently it's not,
    # it's equal, at least for pn_claw_to_apl=True case)
    # TODO delete
    if sp_no_APL == sp_no_APLKC:
        print('APL>KC=0 HAD SAME EFFECT AS DISABLING APL ENTIRELY (so APL>PN '
            'irrelevant)! FIX!'
        )
    #
    #assert sp0 < sp_no_APLKC < sp_no_APL
    rv.kc.wAPLKC = ak_arr.copy()

    # TODO TODO TODO should i try to tweak PN<>APL scale(s) relative to KC<>APL
    # scale(s), s.t. removing either has similar effect on sparsity?
    # TODO if so, assert change in sparsity is close to some tolerance?
    warn('setting wPNAPL to 0 (restored others)')
    rv.pn.wPNAPL = np.zeros(rv.pn.wPNAPL.shape)
    osm.run_KC_sims(mp, rv, regen)
    sp_no_PNAPL = np.mean(rv.kc.responses)
    # sp_no_PNAPL=0.13153783453335144 (!pn-claw-to-APL)
    # sp_no_PNAPL= (pn-claw-to-APL) (is this also the .1133337 i get in other test
    # here?)
    print(f'{sp_no_PNAPL=}')
    print()
    assert sp0 < sp_no_PNAPL < sp_no_APL, f'{sp0=} {sp_no_PNAPL=} {sp_no_APL}'
    rv.pn.wPNAPL = pa_arr.copy()

    warn('setting wKCAPL to 0 (restored others)')
    rv.kc.wKCAPL = np.zeros(rv.kc.wKCAPL.shape)
    osm.run_KC_sims(mp, rv, regen)
    sp_no_KCAPL = np.mean(rv.kc.responses)
    # sp_no_KCAPL=0.12780192908572205 (!pn-claw-to-APL)
    # sp_no_KCAPL= (pn-claw-to-APL)
    print(f'{sp_no_KCAPL=}')
    print()
    assert sp0 < sp_no_KCAPL < sp_no_APL
    rv.kc.wKCAPL = ka_arr.copy()
    #'''

    warn('restoring all weights and re-running')
    rv.kc.wAPLKC = ak_arr.copy()
    rv.pn.wAPLPN = ap_arr.copy()
    osm.run_KC_sims(mp, rv, regen)
    spr = np.mean(rv.kc.responses)
    assert spr == sp0, 'did not get same sparsity after restoring weights'

    # TODO delete
    breakpoint()
    #


# TODO TODO parametrize on at least two cases, one w/o boutons?
# TODO also one w/o claws, ideally
# (can probabably always test w/ connectome APL weights)
def test_dynamics_indexing(orn_deltas):
    # shouldn't matter which of the two (pn_claw_to_apl=True/False) we use
    # NOTE: 1st currently has pn_claw_to_apl=True, and 2nd is same but with it False
    # (2nd omits it, and False is default)
    kws = BOUTON_MODEL_KW_LIST[0]

    responses, _, wPNKC1, params = _fit_mb_model(orn_deltas=orn_deltas,
        return_olfsysm_vars=True, delete_pretime=True, return_dynamics=True,
        # TODO TODO TODO set weights TO APL to nonzero. should still work, and then
        # could test (some of?) inh / Is / Is_from_kcs calcs, right? would then just
        # need to figure out how to test alignment of weights FROM APL (just enable weak
        # amount? or just check first timepoint or two? necessary?)?
        # TODO TODO TODO separate call after most checks below, where we just try to
        # check alignment of weights FROM APL?
        # setting threshold high enough that no KCs should spike
        # (so that claw_sims should sum, within each KC, to an entry KC vm_sims)
        fixed_thr=1e6, wAPLKC=0.0, wAPLPN=0.0, wKCAPL=1.0, wPNAPL=1.0, **kws
    )
    # TODO TODO check i can recreate dot product to get [inh diff?] at least, prob also
    # accounting for dt & appropriate tau there)
    # TODO TODO check i can recreate Is_from_kcs diff w/ wKCAPL weights and either
    # spiking or inh?
    # TODO TODO TODO more important to check in pn_claw_to_apl=True case (and also if i
    # add anything where inh can be shape #-KCs/claws instead of scalar)

    assert params['wAPLKC'].sum() == 0

    # shouldn't be able to re-run model, since I set delete_pretime=True above (to save
    # memory), but should still be able to get parameters from these without issue
    # TODO also assert wAPLKC in this is all 0?
    rv = params['rv']
    mp = params['mp']

    # TODO get dt from diffing time index instead (/check against that?)?
    dt = mp.time_dt
    kc_tau = mp.kc.taum

    # first timepoint (after dropping pretime which is all NaN) is still NaN, as
    # sim_KC_layer loop starts at index 3001 (and time start index is 3000).
    # vm_sims is also 0 for first entry (and only first), so these all probably are.
    boutons = params['bouton_sims']

    n_samples0 = boutons.sizes['time_s']
    # TODO need the .to_index() calls?
    time_s_before = boutons.time_s.to_index()
    # default how='any'. should be no NaN left.
    boutons = boutons.dropna('time_s')

    time_s_after = boutons.time_s.to_index()
    dropped_timepoints = time_s_before.difference(time_s_after)
    if len(dropped_timepoints) == 0:
        warn('bouton NaN was probably already dropped? where exactly though? make '
            'more explicit?'
        )
        assert boutons.sizes['time_s'] == 2499
        # just checking we already dropped them in everything
        assert {
            x.sizes['time_s'] for x in params.values() if isinstance(x, xr.DataArray)
        } == {2499}
    else:
        assert len(dropped_timepoints) == 1, \
            'dropped more than one NaN bouton timepoint (pretime deleted?)'
        t0 = dropped_timepoints[0]
        assert t0 == time_s_before.min(), \
            'dropped timepoint was not first (pretime deleted?)'
        warn(f'dropped first timepoint (={t0}) after pretime '
            '(= start time < odor onset = 0)'
        )
        assert boutons.sizes['time_s'] == n_samples0 - 1

    # TODO could skip the .sel stuff, as it seems we are now in the case above where we
    # have already dropped that one timepoint from everything
    claws = params['claw_sims'].sel(time_s=boutons.time_s)
    kcs = params['vm_sims'].sel(time_s=boutons.time_s)
    # orns/pns are the only ones not 0 for first timepoint (the one dropped from
    # boutons b/c it was/is initialied to NaN instead of 0)
    pns = params['pn_sims'].sel(time_s=boutons.time_s)
    orns = params['orn_sims'].sel(time_s=boutons.time_s)

    wPNKC = xr.DataArray(wPNKC1, dims=['claw', 'bouton'])

    # TODO also some kind of check against orn_deltas / wPNKC indices?
    # TODO and kc/claw indices vs those of APL<>KC weights?
    # TODO and bouton indices vs those of APL<>PN weights?
    assert coords_equal(orns, pns)

    # TODO check i can recalc orns from orn_deltas? or some ordering at least?
    # index at least?
    # TODO get orn_sims and check no negative?
    # TODO check orns and pns have the same ordering?
    # TODO or recalc pns from orns and ln stuff? (need to add the LN dynamics
    # getting in fit_mb_model i assume?)

    # TODO summarize stuff in orns glomeruli but not orn_deltas index (/columns?)?
    # (and reverse, if either)

    # TODO provide these in normal model outputs, as Series w/ appropriate index? (in
    # params, from fit_mb_model, and also pop + save in fit_and_plot...)
    orn_spont = mp.orn.data.spont.squeeze()
    filled_deltas = np.array(mp.orn.data.delta)
    assert len(orn_spont.shape) == 1
    assert len(filled_deltas.shape) == 2
    assert len(orn_spont) == len(filled_deltas)
    assert filled_deltas.shape[1] == orns.sizes['stim']

    # TODO also check we can recreate kc spont (in one-row-per-claw + bouton case)
    # w/ this + wPNKC?

    glom_index = orns[glomerulus_col].to_index()
    # should also be checked in coords_equal(orns, pns) above
    assert glom_index.equals(pns[glomerulus_col].to_index())

    orn_spont = pd.Series(index=glom_index, data=orn_spont)
    filled_deltas = pd.DataFrame(index=glom_index, columns=orns.stim.to_index(),
        data=filled_deltas
    )

    # TODO (?) as well as filled (/processed, if any) orn_deltas, in case any glomeruli
    # are dropped/renamed (+ doc whether either/both are possibilities)

    # this should be the filled spont value:
    # ipdb> orn_spont.value_counts()
    # 13.26087    31
    # 8.00000      2
    # ...

    # NOTE: `pns` has a 'glomerulus' component of sizes, but not anything of # PNs
    # (since all PNs with input from one glomerulus can probably reasonably be thought
    # of as the same, until we get to the boutons, where then things can be moduled
    # separately if that code is enabled). This means we will need to group `claws` by
    # glomerulus, not PN ID, to compare to `pns`.
    n_gloms = pns.sizes['glomerulus']
    for i in range(n_gloms):
        for j in range(n_gloms):
            gi = pns.isel(glomerulus=i)
            gj = pns.isel(glomerulus=j)
            ni = gi.glomerulus.item()
            nj = gj.glomerulus.item()

            values_allclose = np.allclose(gi.values, gj.values)
            if i == j:
                assert values_allclose, 'did not match itself'
            else:
                if orn_spont[ni] == orn_spont[nj] and (
                        (filled_deltas.loc[ni] == 0).all() and
                        (filled_deltas.loc[nj] == 0).all()
                    ):
                    # TODO + summarize glomeruli w/ 0 response below? also the
                    # equivalence classes that also have same spont (prob don't care)?
                    continue

                assert not values_allclose, f'gloms {ni} and {nj} had same responses!'

    bouton_glom_mins = boutons.groupby('glomerulus').min()
    assert bouton_glom_mins.identical(boutons.groupby('glomerulus').max())
    assert bouton_glom_mins.identical(pns)

    # TODO TODO convert wPNKC to arr and check we can recreate claw from bouton?

    claw_glom_mins = claws.groupby('glomerulus').min()
    assert claw_glom_mins.identical(claws.groupby('glomerulus').max())
    # TODO worth a test that if we have non-0/1 wPNKC we can get the expected divergence
    # between claw_sims and bouton_sims?
    #
    # one implication of this is that claw_sims are not 0 in the [start, stim_start]
    # period, b/c they are just as influenced by the PN spontaneous activity as
    # bouton_sims are
    assert claw_glom_mins.identical(pns)

    # pretty slow. just do for ~2 odors?
    # TODO delete. too slow (and tests on it might get killed)
    #claws2 = wPNKC.dot(boutons)
    for oi in [0, 3]:
        odor_claws = claws.isel(stim=oi).squeeze(drop=True)

        odor_boutons = boutons.isel(stim=oi).squeeze(drop=True)
        odor_claws2 = wPNKC.dot(odor_boutons)

        odor_claws = move_all_coords_to_index(
            odor_claws.reset_index('claw').drop_vars(['glomerulus', 'stim'])
        )
        # TODO why does drop=True not remove this 'stim' var? call my fn to remove
        # scalar coords?
        odor_claws2 = odor_claws2.drop_vars('stim')
        # the coords check is pretty slow (~5-10s). allclose is faster.
        assert coords_equal(odor_claws2, odor_claws)
        assert np.allclose(odor_claws2, odor_claws)

    # TODO some way to not drop other metadata? want to preserve what we have in `kcs`,
    # but prob nbd. just kc_type missing it seems (may actually want to drop some
    # metadata, if [as i assume] we have more claw metadata than kc metadata)
    # NOTE: these are used in fn below
    claw_sums = claws.groupby('kc_id').sum()
    assert np.array_equal(claw_sums.kc_id, kcs.kc_id)
    # TODO assert coords are only diff by kc_type (b/c kcs has it but claw_sums
    # doesn't), or also resolve that and assert coords equal?
    #
    # TODO also check other odors? currently hardcoding isel(stim=0)
    def check_we_can_recreate_kc_vm_from_claw_sums(i: int, j: Optional[int] = None
        ) -> float:
        """i, j are int KC indices to be used for isel on claw_sums/kcs
        """
        if j is None:
            j = i

        # TODO fix so these have same KC coords before subtracting?
        # ipdb> kcs.isel(stim=0, kc=0)
        # <xarray.DataArray (time_s: 2499)>
        # array([ 3.51,  6.84, 10.01, ..., 44.12, 44.14, 44.15])
        # Coordinates:
        #     stim     object ('megamat', '2h @ -3')
        #   * time_s   (time_s) float64 -0.4995 -0.499 -0.4985 ... 0.749 0.7495 0.75
        #     kc       object (300968622, 'ab')
        # ipdb> claw_sums.isel(stim=0, kc_id=0)
        # <xarray.DataArray (time_s: 2499)>
        # array([70.18, 70.18, 70.18, ..., 44.4 , 44.42, 44.43])
        # Coordinates:
        #     stim     object ('megamat', '2h @ -3')
        #   * time_s   (time_s) float64 -0.4995 -0.499 -0.4985 ... 0.749 0.7495 0.75
        #     kc_id    int64 300968622
        # maybe there is an off-by-one 1 this? (maybe just either claw_sums or kcs
        # part?) really doesn't look like it. neither is sufficiently small:
        # ```
        # fc.values=array([3.51, 3.51, 3.51, ..., 2.22, 2.22, 2.22])
        # fc.values[0]=3.5088591241801677
        # fc.values[1]=3.508825509897477
        # fk.values=array([-0.18, -0.34, -0.5 , ..., -2.21, -2.21, -2.21])
        # fk.values[0]=-0.17544295620900838
        # fk.values[1]=-0.3421120838934318
        #
        # arr1.values=array([3.33, 3.17, 3.01, ..., 0.01, 0.01, 0.01])
        # arr1.values[0]=3.3334161679711594
        # arr1.values[1]=3.1667134260040455
        # arr2.values=array([3.33, 3.17, 3.01, ..., 0.01, 0.01, 0.01])
        # arr2.values[0]=3.3333825536884683
        # arr2.values[1]=3.1666798548860093
        # ```
        # TODO any (/ how much?) of numerical issue from way i compute factor at
        # end? change?
        # TODO recalc by first multiplying whole tensor by dt, then dividing by
        # kc_tau, and checking diff vs this calculation? also try multiplying by
        # 1/kc_tau?
        fc = claw_sums.isel(stim=0, kc_id=i) * (dt/kc_tau)
        fk = kcs.isel(stim=0, kc=i) * (-dt/kc_tau)

        # TODO fix coords of above so they are the same, so i can use the same code for
        # each here?
        # need a KC ID, since I can't use isel to get KC in claws, since it has kc_id
        # within claw dimension MultiIndex
        k1 = fk.kc.item()[0]
        k2 = fc.kc_id.item()
        # comparison of kc_id across claw_sums and kcs (before this fn) should guarantee
        # this
        assert k1 == k2

        # this adds a NaN dropped below, when shifting
        arr1 = (claw_sums.isel(stim=0, kc_id=i) - kcs.isel(stim=0, kc=i).shift(time_s=1)
            ) * (dt/kc_tau)

        # TODO try removing this now that i'm shifting kcs above?
        arr1 = arr1.isel(time_s=slice(None, -1))

        kcs0 = kcs.isel(stim=0, kc=j)
        # label='upper' is the default
        arr2 = kcs0.diff('time_s', label='lower')

        assert np.array_equal(arr1.time_s, arr2.time_s)

        # TODO maybe shift of +1 isn't right? seems i have values offset by 1 again
        # now... some other fix? (am i actually checking all the values i could be?)
        # should just be dropping NaN added at start of arr1, from shift by 1
        arr1 = arr1.dropna('time_s')
        arr2 = arr2.sel(time_s=arr1.time_s)

        # TODO fix xarray indexing above to not need this? maybe go back to
        # label='upper'? (would at least require other changes)
        # TODO use another shift call (or two?) to fix this?
        assert np.allclose(arr1.values[1:], arr2.values[:-1])

    n_kcs = kcs.sizes['kc']
    for i in range(n_kcs):
        check_we_can_recreate_kc_vm_from_claw_sums(i)
        # TODO still check no other rows, at least not w/ actual responses on input,
        # have equal arr1/arr2? (+ tqdm at that point?) (maybe there would be some rows
        # equal though?)

    wKCAPL = series2xarray_like(params['wKCAPL'], claws)
    wPNAPL = series2xarray_like(params['wPNAPL'], boutons)

    # TODO TODO TODO prob need separate call w/o APL output disabled to actually check
    # these?
    wAPLKC = series2xarray_like(params['wAPLKC'], claws)
    wAPLPN = series2xarray_like(params['wAPLPN'], boutons)

    Is = params['Is_sims'].sel(time_s=boutons.time_s)
    Is_from_kcs = params['Is_from_kcs'].sel(time_s=boutons.time_s)
    if mp.kc.pn_claw_to_apl:
        assert coords_equal(claws.dot(wKCAPL), Is_from_kcs)
        assert np.allclose(claws.dot(wKCAPL), Is_from_kcs)
        # TODO this doesn't matter, right? explain why i dont want to use this var for
        # Is_from_kcs here? i suppose cause this has dynamics and Is_from_kcs wouldn't
        # here (i.e. it's own decay)?
        assert (Is == 0).all()
    else:
        # TODO also check we can recreate Is here
        # TODO TODO TODO what to test here? use spike_recordings?
        breakpoint()

    # TODO TODO TODO then just need to verify wAPLPN indexing (need to enable some
    # inhibition for that?)
    Is_from_pns = params['Is_from_pns'].sel(time_s=boutons.time_s)
    # this should verify wPNAPL indexing in model (and pn>bouton indexing should already
    # have been confirmed above). not .equals exactly, but allclose.
    assert coords_equal(boutons.dot(wPNAPL), Is_from_pns)
    assert np.allclose(boutons.dot(wPNAPL), Is_from_pns)

    # TODO TODO TODO use (/ delete) all below
    # TODO i assert something about Is shifted relative to Is_from_kcs in
    # mb_model.plot_dynamics, right? move that here (too?)?
    # TODO TODO TODO also test this one here (already have some code for this in
    # mb_model?) (will handle later. higher priority to check APL>(KC,PN) weights below)
    inh = params['inh_sims'].sel(time_s=boutons.time_s)

    assert mp.pn.preset_wPNAPL
    # TODO TODO TODO why does inh blow up towards the end, but this doesn't?
    # also happen w/ more realistic setups? just test w/ all weights enabled, below?
    inh2 = (Is_from_pns + Is_from_kcs - inh.shift(time_s=1)) * (dt/mp.kc.apl_taum)
    print('finish checking inh2 vs inh (finding why so diff...)')
    #breakpoint()
    #
    # TODO TODO fix so below won't get killed. del all above? separate test?


# TODO combine back into one test? rename at least
def test_dynamics_indexing2(orn_deltas):
    # shouldn't matter which of the two (pn_claw_to_apl=True/False) we use
    # NOTE: 1st currently has pn_claw_to_apl=True, and 2nd is same but with it False
    # (2nd omits it, and False is default)
    kws = BOUTON_MODEL_KW_LIST[0]

    # TODO TODO TODO separate tests w/ only either wAPLKC or wAPLPN = 1 (other 0),
    # to see if outputs make more sense for either of those?
    _, _, wPNKC, params = _fit_mb_model(orn_deltas=orn_deltas,
        return_olfsysm_vars=True, delete_pretime=True, return_dynamics=True,
        # refactor to share w/ above. only difference here is wAPLKC & wAPLPN
        # are now also 1.0 (instead of 0.0 before)
        # TODO TODO TODO restore
        #fixed_thr=1e6, wAPLKC=1.0, wAPLPN=1.0, wKCAPL=1.0, wPNAPL=1.0, **kws
        #fixed_thr=1e6, wAPLKC=0.0, wAPLPN=0.0, wKCAPL=1.0, wPNAPL=1.0, **kws
        # TODO TODO TODO actually, i think i need to test w/ either wKCAPL or wPNAPL 0
        # (-> see if we can recreate inh any better)
        fixed_thr=1e6, wAPLKC=0.0, wAPLPN=0.0, wKCAPL=1.0, wPNAPL=0.0, **kws
    )
    # shouldn't be able to re-run model, since I set delete_pretime=True above (to save
    # memory), but should still be able to get parameters from these without issue
    # TODO also assert wAPLKC in this is all 0?
    rv = params['rv']
    mp = params['mp']

    # TODO get dt from diffing time index instead (/check against that?)?
    dt = mp.time_dt

    # we should have set the threshold high enough to avoid spiking
    assert (params['spike_recordings'] == 0).all()

    # TODO delete. was only for when this was part of test above
    # don't need to redefine the DataArray wPNKC (constructed from wPNKC1), if this
    # passes
    #assert wPNKC1.equals(wPNKC2)
    # TODO also remove the time_s subsetting in above. shouldn't be needed anymore
    # TODO subset all down to one odor first? or change code below?
    pns = params['pn_sims']
    boutons = params['bouton_sims']
    claws = params['claw_sims']
    inh = params['inh_sims']

    Is_from_kcs = params['Is_from_kcs']
    Is_from_pns = params['Is_from_pns']
    wAPLKC_all0 = (rv.kc.wKCAPL == 0).all()
    wAPLPN_all0 = (rv.pn.wPNAPL == 0).all()
    # may also want to assert only one is not all0 (or at least focus on those cases for
    # now. could integrate into one test w/ both enabled in one call later)
    assert not (wAPLKC_all0 and wAPLPN_all0)

    if wAPLKC_all0:
        assert (Is_from_kcs == 0).all()

    if wAPLPN_all0:
        assert (Is_from_pns == 0).all()

    assert mp.pn.preset_wPNAPL
    # TODO TODO TODO why does inh blow up towards the end, but this doesn't?
    # also happen w/ more realistic setups? just test w/ all weights enabled, below?
    # TODO TODO does it still here (like w/ two weight scales 0 above)?
    inh2 = (Is_from_pns + Is_from_kcs - inh.shift(time_s=1)) * (dt/mp.kc.apl_taum)
    # TODO TODO TODO fix still large divergence between inh/inh2 in case only
    # Is_from_pns nonzero

    # NOTE: much of below copied from WIP code in plot_apl_dynamics don't need fn like
    # above for DataFrames. DataArray constructor handles those fine.
    wPNKC_arr = xr.DataArray(data=wPNKC, dims=['claw', 'bouton'])
    assert np.array_equal(wPNKC_arr.values, wPNKC.values)

    claw_index = wPNKC_arr.get_index('claw')
    assert claw_index.equals(wPNKC.index)
    claw_index2 = claws.get_index('claw')
    assert glomerulus_col in claw_index2.names
    assert claw_index2.droplevel(glomerulus_col).equals(claw_index)

    # TODO TODO TODO is it problematic that this index is not equal to itself
    # sorted? could that be part of indexing issue? (if there is one...)
    bouton_index = wPNKC_arr.get_index('bouton')
    assert bouton_index.equals(wPNKC.columns)
    assert boutons.get_index('bouton').equals(bouton_index)
    wPNKC = wPNKC_arr

    # TODO subset to single odor from here? or change code below?
    # TODO why am i needing tot manually drop 'stim' on all these?
    pns = pns.isel(stim=0).squeeze(drop=True).drop_vars('stim')
    boutons = boutons.isel(stim=0).squeeze(drop=True).drop_vars('stim')
    claws = claws.isel(stim=0).squeeze(drop=True).drop_vars('stim')
    #

    # ah, it's failing when we are not also analyzing just a single odor.
    # ipdb> odor
    # slice(None, None, None)
    # ipdb> pns.dims
    # ('panel', 'glomerulus', 'time_s')
    assert pns.dims == ('glomerulus', 'time_s'), f'{pns.dims=}'
    # drop_vars('glomerulus') is just to remove this leftover 'glomerulus' level
    # outside of the bouton index (first row in output below):
    # Coordinates:
    #   glomerulus  (bouton) object 'D' 'D' 'D' 'D' 'D' ... 'VP2' 'VP2' 'VP2' 'VP4'
    # * time_s      (time_s) float64 -0.4995 -0.499 -0.4985 ... 0.749 0.7495 0.75
    # * bouton      (bouton) MultiIndex
    # - glomerulus  (bouton) object 'D' 'D' 'D' 'D' 'D' ... 'VP2' 'VP2' 'VP2' 'VP4'
    # - pn_id       (bouton) int64 1536947502 5813055184 ... 1975878958 634759240
    # - bouton_id   (bouton) int64 1 3 1 0 0 0 1 1 0 2 1 0 ... 1 3 3 0 2 9 6 8 5 1 0
    #
    # expanding each PN up to # of times it appears in boutons (and in same order as
    # there)
    boutons_no_inh = pns.loc[dict(glomerulus=boutons.glomerulus)].drop_vars(
        'glomerulus'
    )
    assert boutons_no_inh.dims == ('bouton', 'time_s')
    assert boutons_no_inh.groupby('glomerulus').min().equals(pns)

    # (it is True now that i've actually removed the NaN by here)
    # > (boutons >= 0).all()
    # False
    boutons_min = boutons.min().item()
    # should == 0 almost certainly, at least if there is any meaningful PN>APL
    # inhibition enabled, with APL actually getting activity
    # TODO assert it's >0 otherwise?
    assert boutons_min >= 0, f'had some bouton activities < 0. min={boutons_min}'

    # TODO see if we get any performance benefit to using any scipy.sparse or Sparse
    # (https://sparse.pydata.org/en/stable/) representation of data while
    # constructing wPNKC
    # TODO any other way to get sparse DataArrays?
    #
    # need to start w/ boutons that have already have inh applied here.
    # takes a couple seconds, but haven't gotten killed yet.
    # .dot returns something that had dot product computed over shared dimensions,
    # which is just 'bouton' between these two, so we will be left with dims
    # ('time_s', 'claw')
    claws_no_inh = wPNKC.dot(boutons)

    # removing the 'glomerulus' level from the 'claw' MultiIndex, since wPNKC (and
    # thus claws_no_inh, does not have it)
    claws = claws.reset_index('claw').drop_vars('glomerulus')
    # restoring the 'claw' dim MultiIndex
    breakpoint()
    claws = move_all_coords_to_index(claws)

    # TODO this all work?
    assert coords_equal(boutons, boutons_no_inh)
    assert coords_equal(claws, claws_no_inh)

    # NOTE: would probably also fail if there was any chance of boutons (or anything
    # really) having NaN still
    #
    # ok, this one is true at least
    assert (boutons_no_inh >= boutons).all()
    # TODO TODO TODO FIX. at least claws one is failing
    print('FIX CLAWS_NO_INH < CLAWS (bouton indexing issue?)')
    #assert (claws_no_inh >= claws).all()


    # TODO TODO TODO also test alignment of APL<>KC and PN<>APL weights
    # (at least when claw & bouton lengths, respectively)
    # TODO TODO TODO need some model run(s) w/ APL>KC and APL>PN not disabled, to test
    # indexing of those? how to set up?

    # TODO delete
    breakpoint()
    #

    # TODO TODO TODO also check that we can recalculate Is_sims / inh_sims from
    # claw_sims (or pn_sims, if wPNKC_one_row_per_claw=False)? or at least
    # Is_from_[kcs|pns]?
    # TODO TODO TODO TODO need to add separate variable to olfsysm, to be able to tell
    # for sure we are recreating these values? otherwise would probably have to recreate
    # claw/bouton activities, by applying this recalculated inhibition
    # TODO TODO TODO should need separate test to check Is_from_kcs, at least if KC>APL
    # input requires spiking (currently threshold set above to disable spiking)

    # TODO worth also shuffling things by int index, to make sure we are actually
    # relying on DataArray/pandas indexing, and not existing order of entries?
    # prob not (b/c would need to re-order indices then anyway, to compare, no?)?


@pytest.mark.parametrize('apl_weights', BOUTON_MODEL_KW_LIST, ids=format_model_params,
    indirect=['apl_weights']
)
def test_apl_weights_fitmbmodel(apl_weights, orn_deltas):
    # NOTE: apl_weights fixture currently also returns kws, for the convenience of this
    # test
    # TODO otherwise, pass BOUTON_MODEL_KW_LIST twice in parametrize, for apl_weights
    # and kws? or better way to have a param be both direct and indirect?
    ret, kws = apl_weights
    params = ret[-1]

    thr_and_apl_kws = get_thr_and_APL_weights(params, kws)
    # TODO delete
    print()
    print(f'{[k for k in kws if "APL" in k and k.startswith("w")]=}')
    #

    print(f'{thr_and_apl_kws=}')

    # TODO should i move this to fixture? what happens if a test fixture fails? do all
    # selected tests using it also fail?
    #
    # working for both pn_claw_to_apl=True/False cases
    ret2 = _fit_mb_model(orn_deltas=orn_deltas, **{**thr_and_apl_kws, **kws})
    assert_fit_outputs_equal(ret, ret2, ignore_tuning_iters=True)

    responses = ret[0]
    # ipdb> responses.mean().mean()
    #0.10728841190055699

    rv = params['rv']
    # these will not currently be in thr_and_apl_kws (assumed each can be calculated
    # from the from-APL weights), so need to get separately
    wKCAPL_scale = rv.kc.wKCAPL_scale
    wPNAPL_scale = rv.pn.wPNAPL_scale

    wAPLPN_scale = thr_and_apl_kws['wAPLPN']
    wAPLKC_scale = thr_and_apl_kws['wAPLKC']

    # TODO TODO TODO restore after back into test def commented w/ docstring above
    no_APLPN = dict(thr_and_apl_kws)
    # TODO important this is actually a double? add some pybind11 magic to automatically
    # cast to double from int, if so?
    no_APLPN['wAPLPN'] = 0.0
    # TODO TODO work?
    # need to specify this separately, otherwise it will be computed from wAPLPN
    no_APLPN['wPNAPL'] = wPNAPL_scale
    ret_no_APLPN = _fit_mb_model(orn_deltas=orn_deltas, **{**no_APLPN, **kws})
    responses_no_APLPN, spike_counts_no_APLPN, _, params_no_APLPN = \
        ret_no_APLPN

    # with no weights hardcoded, as calls above...
    # with pn_claw_to_apl=True:
    # average (across odor) values in odor_stats:
    # max_kc_apl_drive: 68.547
    # avg_kc_apl_drive: 32.4625
    # max_bouton_apl_drive: 21.0145
    # avg_bouton_apl_drive: 5.52647
    # avg_kc_pre_inh: 508970
    # avg_bouton_pre_inh: 19749.2
    # avg_kc_inh: 197132
    # avg_bouton_inh: 44275.1
    #
    # with pn_claw_to_apl=False (default):
    # average (across odor) values in odor_stats:
    # max_kc_apl_drive: 0.0205206
    # avg_kc_apl_drive: 0.0005523
    # max_bouton_apl_drive: 43.9507
    # avg_bouton_apl_drive: 13.189
    # avg_kc_pre_inh: 508970
    # avg_bouton_pre_inh: 19749.2
    # avg_kc_inh: 122444
    # avg_bouton_inh: 27500.4

    # with pn_claw_to_apl=True:
    # average (across odor) values in odor_stats:
    # max_kc_apl_drive: 54.1662
    # avg_kc_apl_drive: 21.1573
    # max_bouton_apl_drive: 89.5107
    # avg_bouton_apl_drive: 55.9184
    # avg_kc_pre_inh: 508970
    # avg_bouton_pre_inh: 19749.2
    # avg_kc_inh: 412409
    # avg_bouton_inh: 0
    #
    # with pn_claw_to_apl=False (default):
    # average (across odor) values in odor_stats:
    # max_kc_apl_drive: 0.0092385
    # avg_kc_apl_drive: 0.000203146
    # max_bouton_apl_drive: 121.734
    # avg_bouton_apl_drive: 76.0487
    # avg_kc_pre_inh: 508970
    # avg_bouton_pre_inh: 19749.2
    # avg_kc_inh: 580218
    # avg_bouton_inh: 0
    #
    # TODO TODO TODO this also misbehaving like in other test? (yes) why???
    # responses_no_APLPN.mean().mean()=0.06802744192365166 (pn_claw_to_apl=True)
    # responses_no_APLPN.mean().mean()=0.05077435131096319 (pn_claw_to_apl=False)
    print(f'{responses_no_APLPN.mean().mean()=}')
    # TODO TODO add similar sparsity assertions to in other test (that would pass here,
    # for reason i'm not clear on...)
    # TODO delete
    #

    # pn_claw_to_apl=True:
    # average (across odor) values in odor_stats:
    # max_kc_apl_drive: 71.7594
    # avg_kc_apl_drive: 34.177
    # max_bouton_apl_drive: 0
    # avg_bouton_apl_drive: 0
    # avg_kc_pre_inh: 508970
    # avg_bouton_pre_inh: 19749.2
    # avg_kc_inh: 177800
    # avg_bouton_inh: 39933.2
    #
    # pn_claw_to_apl=False:
    # average (across odor) values in odor_stats:
    # max_kc_apl_drive: 0.02932
    # avg_kc_apl_drive: 0.000796943
    # max_bouton_apl_drive: 0
    # avg_bouton_apl_drive: 0
    # avg_kc_pre_inh: 508970
    # avg_bouton_pre_inh: 19749.2
    # avg_kc_inh: 36376.3
    # avg_bouton_inh: 8169.96
    no_PNAPL = dict(thr_and_apl_kws)
    no_PNAPL['wAPLPN'] = wAPLPN_scale
    no_PNAPL['wPNAPL'] = 0.0
    ret_no_PNAPL = _fit_mb_model(orn_deltas=orn_deltas, **{**no_PNAPL, **kws})
    responses_no_PNAPL, spike_counts_no_PNAPL, _, params_no_PNAPL = \
        ret_no_PNAPL
    print(f'{responses_no_PNAPL.mean().mean()=}')

    # TODO restore?
    '''
    no_APLKC = dict(thr_and_apl_kws)
    no_APLKC['wAPLKC'] = 0.0
    no_APLKC['wKCAPL'] = wKCAPL_scale
    ret_no_APLKC = _fit_mb_model(orn_deltas=orn_deltas, **{**no_APLKC, **kws})
    responses_no_APLKC, spike_counts_no_APLKC, _, params_no_APLKC = \
        ret_no_APLKC
    print(f'{responses_no_APLKC.mean().mean()=}')

    no_KCAPL = dict(thr_and_apl_kws)
    no_KCAPL['wAPLKC'] = wAPLKC_scale
    no_KCAPL['wKCAPL'] = 0.0
    ret_no_KCAPL = _fit_mb_model(orn_deltas=orn_deltas, **{**no_KCAPL, **kws})
    responses_no_KCAPL, spike_counts_no_KCAPL, _, params_no_KCAPL = \
        ret_no_KCAPL
    print(f'{responses_no_KCAPL.mean().mean()=}')
    '''

    # TODO delete
    breakpoint()
    #


# TODO TODO test orn deltas can't actually ever go negative (+ fix if so), since scaling
# fn is just a factor w/ no offset, must be relying on spontaneous firing rate to be
# higher than scaled negative dF/F (-> negative est spike rate delta)
# TODO + add assertion for this on average values in fit_mb_model? does that imply
# dynamic orn firing rates never goes below 0 too (prob not?)?

@pytest.mark.parametrize('scaling_method', ['to-avg-max', None])
def test_scale_dff2spiking(tmp_path, scaling_method):
    # TODO also (/only) test under tmp_path? currently will write to dir of committed
    # outputs (prob don't want, especially in a test)
    # TODO actually check anything on output?
    dir1 = tmp_path / 'dff_to_spiking_scale'
    dir1.mkdir()

    # TODO TODO commit outputs from this, and use in model_mb_responses test
    # TODO TODO + also make example script / fn using this (and/or hardcoding
    # scale, and not loading this data at all)
    ret = fit_dff2spiking_from_remypaper_flies_and_hallem(dir1,
        scaling_method=scaling_method
    )
    ret1, response_calc_params, roi_depths = ret
    del ret

    dff_to_spiking_choices_csv = dir1 / dff_to_spiking_model_choices_csv_name
    assert written_since_proc_start(dff_to_spiking_choices_csv)

    dff_to_spiking_data_csv = dir1 / dff_to_spiking_data_csv_name
    assert written_since_proc_start(dff_to_spiking_data_csv)

    # NOTE: currently imporotant this is different from plot_dir in call above, or else
    # plots + mean_est_spike_deltas.csv will raise errors when call below tries to write
    # them to same paths as call above. In real usage, if call generating cache occurred
    # on a previous run of whatever script, this would not be an issue (but the outputs
    # would be [at least partially] overwritten by subsequent runs).
    dir2 = tmp_path / 'using_cached_scale'
    dir2.mkdir()

    data_dir = remypaper_dff2spiking_data_dir
    df = read_parquet(data_dir / 'ij_certain-roi_stats.parquet')

    # TODO need to only pass one panel's data thru at a time? (scaling fn actually errs
    # if we do that, if we index in a way that drops panel level) any code still have
    # that restriction, even if not this fn?

    response_calc_params2 = dict(response_calc_params)
    # TODO restore? test both. shouldn't matter
    #response_calc_params2 = None

    # TODO loop over both? shouldn't matter
    #roi_depths2 = None
    roi_depths2 = roi_depths

    # TODO test both w/ and w/o doing this? shouldn't really matter
    # TODO parametrize test w/ this?
    subset_to_megamat = True
    #subset_to_megamat = False

    if subset_to_megamat:
        # TODO (delete? check again i really do need diags to get same output [since
        # scaling within each fly]?) care to fix df.loc['megamat'] (so that scale_dff...
        # doesn't require panel level)? currently fails if we index in a way that drops
        # panel level
        # TODO delete (think i need diags too, or else min/max per fly is different)
        #df = df.loc[df.index.get_level_values('panel') == 'megamat']
        panels = (diag_panel_str, 'megamat')
        df = df.loc[df.index.get_level_values('panel').isin(panels)]

        # if one ROI in a fly has NaN value for one odor (in megamat panel), all ROIs
        # will have NaN for that fly. first .all() is whether all odors (in a given
        # panel) are NaN, across all fly_cols + ['roi'] columns.
        all_rois_nan = df.loc['megamat'].isna().all().groupby(fly_cols).all()
        any_rois_nan = df.loc['megamat'].isna().all().groupby(fly_cols).any()
        assert all_rois_nan.equals(any_rois_nan)
        megamat_flies = any_rois_nan[~ any_rois_nan].index

        # TODO skip this subsetting, now that i'm having to .loc['megamat'] in
        # pd_allclose call below anyway?

        # dropping flies that only had validation+diagnostics, and not
        # megamat+diagnostics. no flies should just have diagnostics.
        megamat_fly_col_mask = df.columns.droplevel('roi').isin(megamat_flies)

        if roi_depths2 is not None:
            assert roi_depths2.columns.equals(df.columns)

        df = df.loc[:, megamat_fly_col_mask]

        if roi_depths2 is not None:
            roi_depths2 = roi_depths2.loc[:, megamat_fly_col_mask]
            assert roi_depths2.columns.equals(df.columns)

    ret2 = scale_dff_to_est_spike_deltas_using_hallem(dir2, df, roi_depths2,
        model_dir=dir1, use_cache=True, response_calc_params=response_calc_params2,
        scaling_method=scaling_method
    )
    (mean_est_df1, _, _, hallem_delta_wide1, _) = ret1
    # hallem_delta_wide would be None from this cached call (not true anymore, since
    # model_mb_responses needed it unconditionally)
    (mean_est_df2, _, _, _, _) = ret2

    # TODO TODO also test w/ load_data_and_refit_dff2spiking_model? in same test?

    # TODO delete
    # doesn't seem like the panel specific issue is because where the min/maxs are:
    # ipdb> fly_mean_df.groupby('fly_id')[col_to_fit].idxmin()
    # fly_id
    # 1     (glomeruli_diagnostics, p-cre @ -3)
    # 2                      (megamat, pa @ -3)
    # 3      (glomeruli_diagnostics, aphe @ -5)
    # 4                      (megamat, pa @ -3)
    # 5                      (megamat, va @ -3)
    # 6                     (megamat, Lin @ -3)
    # 7                   (megamat, 2-but @ -3)
    # 8                     (megamat, Lin @ -3)
    # 9      (glomeruli_diagnostics, e3hb @ -6)
    # 10    (glomeruli_diagnostics, p-cre @ -3)
    # 11     (glomeruli_diagnostics, elac @ -7)
    # 12     (glomeruli_diagnostics, 3mtp @ -5)
    # 13    (glomeruli_diagnostics, p-cre @ -3)
    # 14      (glomeruli_diagnostics, HCl @ -1)
    # Name: to-avg-max_scaled_delta_f_over_f, dtype: object
    # ipdb> fly_mean_df.groupby('fly_id')[col_to_fit].idxmax()
    # fly_id
    # 1                   (megamat, 1-6ol @ -3)
    # 2       (glomeruli_diagnostics, t2h @ -6)
    # 3       (glomeruli_diagnostics, HCl @ -1)
    # 4                      (megamat, pa @ -3)
    # 5       (glomeruli_diagnostics, HCl @ -1)
    # 6     (glomeruli_diagnostics, p-cre @ -3)
    # 7       (glomeruli_diagnostics, HCl @ -1)
    # 8                      (megamat, eb @ -3)
    # 9      (glomeruli_diagnostics, e3hb @ -6)
    # 10       (glomeruli_diagnostics, 2h @ -6)
    # 11     (glomeruli_diagnostics, elac @ -7)
    # 12       (glomeruli_diagnostics, 2h @ -6)
    # 13       (glomeruli_diagnostics, ms @ -3)
    # 14       (glomeruli_diagnostics, 2h @ -3

    if subset_to_megamat:
        # TODO delete (think i need diags too, or else min/max per fly is different)
        #mean_est_df1 = mean_est_df1.loc[
        #    mean_est_df1.index.get_level_values('panel') == 'megamat'
        #]
        #mean_est_df2 = mean_est_df2.loc[
        #    mean_est_df2.index.get_level_values('panel') == 'megamat'
        #]
        mean_est_df1 = mean_est_df1.loc[
            # NOTE: transposed wrt input dataframe
            :, mean_est_df1.columns.get_level_values('panel').isin(panels)
        ]
        mdf1 = mean_est_df1

        mean_est_df2 = mean_est_df2.loc[
            :, mean_est_df2.columns.get_level_values('panel').isin(panels)
        ]
        mdf2 = mean_est_df2
        assert not mdf2.isna().any().any()

        assert len(mdf2.index.difference(mdf1.index)) == 0
        assert len(mdf2.columns.difference(mdf1.columns)) == 0

        mdf1 = mdf1.loc[mdf2.index, mdf2.columns]
        # there would be NaN before subsetting to mdf2 index/columns above
        assert not mdf1.isna().any().any()

        # NOTE: there is no way to get the diagnostics panel to match up, if we are also
        # averaging over the diagnostics for the validation2 flies in the first call (as
        # we are)
        assert pd_allclose(mdf1.loc[:, 'megamat'], mdf2.loc[:, 'megamat'])
    else:
        assert pd_allclose(mean_est_df1, mean_est_df2, equal_nan=True)

    # TODO just call model_mb_responses here? ideally would want in separate
    # test, but would like to share setup of fit_dff2spiking_from... dir w/ this test
    # (do have separate test below, which currently does not share any setup)
    # TODO if i add a test for model_mb_responses, also commit+load+test w/ kiwi-control
    # data (have under differen't committed dir, 2025-09-30_tom_orn_data_signed-max)
    # TODO can fixtures use tmp_path? or how to do that?


# TODO mark slow
def test_model_mb_responses(tmp_path):
    # TODO delete here (but use in some other test. this one needs dF/F, not spike
    # delta input tho)
    #orn_deltas = natmix_orn_deltas()

    # TODO also run megamat/validation2 data thru w/ whatever repro_paper flag set
    # (-> check against saved outputs?)

    # kiwi/control signed absmax response calc data, using Sam's ROIs
    df = load_natmix_dff()

    # contains committed files:
    # - dff2spiking_model_choices.csv
    # - dff2spiking_model_input.parquet
    # - dff2spiking_model_input.csv (probably not used, in favor of parquet)
    # which together should be sufficient to recreate dF/F -> spiking model I would
    # typically use
    model_dir = data_root / 'internal'

    # TODO TODO either skip sensitivity analysis, or force it to do dramatically fewer
    # steps
    model_mb_responses(df, tmp_path, dff2spiking_cache_dir=model_dir)


# TODO commit some (dramatically subset, perhaps also compressed) example model
# dynamics, and use to test all dynamics plotting fns (so might want test data with and
# without boutons and/or claws)

