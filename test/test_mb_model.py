#!/usr/bin/env python3

from itertools import product
import os
from pathlib import Path
from pprint import pformat
import math
# TODO delete?
import traceback
#
from typing import Hashable, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytest
import math
from pathlib import Path

# TODO does importing this before hong2p.equals allow that warning (optional from within
# `equals`) to be hooked in same way (from `al_util` module-level warning format
# handling change)? import order matter?
import al_util
from hong2p.util import pd_allclose, equals

from al_util import read_pickle, warn
from mb_model import (fit_mb_model, fit_and_plot_mb_model, connectome_wPNKC, KC_ID,
    CLAW_ID, BOUTON_ID, KC_TYPE, step_around, read_series_csv, read_param_csv,
    read_param_cache, get_thr_and_APL_weights, variable_n_claw_options,
    dict_seq_product, get_connectome_wPNKC_params, format_model_params,
    eval_and_check_compatible, glomerulus_col, ParamDict
)


# You can set these either 0/1 in prefix before pytest command.
#
# Can be slightly faster, and can avoid some of the main potential memory issues by
# skipping plotting code, since main issue is loading claw_sims into fit_mb_model, for
# plot_example_dynamics=True path.  If this is not set, will plot if QUICK=False (and
# not otherwise).
PLOT: bool = bool(int(os.environ.get('PLOT', True)))
# TODO do allow certain caches being used if another flag (or this?) is set? would
# need centrally somewhere though... want to skip some of the longer tuning cases
#
# currently this replaces [FITANDPLOT_]MODEL_KW_LIST w/ the QUICK* versions below,
# and disables some plotting, if True.
QUICK: bool = bool(int(os.environ.get('QUICK', False)))

if not QUICK and PLOT:
    PLOT_KWS = dict(make_plots=True, plot_example_dynamics=True)
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
# (could then use that to also test APL_coup_const=0 vs -1 case)
# TODO test that relevant subset of these all have diff wPNKC (those w/ diff
# connectome_wPNKC args. could use the fn that collects those args to tell which subset
# to test?)
#
# NOTE: IDs need to be unique, so if a parameter I'd otherwise want to include in the
# *MODEL_KW_LIST is not in `format_model_params` output (e.g. b/c in
# `mb_model.exclude_params`), would currently have to make a separate test for that (or
# manually assign ID to that case, maybe)
#
# should cover all the main paths in olfsysm, but also in fit_mb_model/etc
MODEL_KW_LIST: List[ParamDict] = dict_seq_product(
    [
        # pn_claw_to_apl=True: no-spiking required; direct claw>APL input
        dict(one_row_per_claw=True, prat_claws=True, pn_claw_to_APL=True),

        # TODO test n_claws_active_to_spike=2/3? (betty didn't care much about that
        # code)

        dict(one_row_per_claw=True, prat_claws=True),
        # TODO keep? (tianpei's version. may eventually need to specify
        # prat_claws=False)
        dict(one_row_per_claw=True),

        # TODO maybe pick only either this or dict() to do w/ both
        # use_connectome_APL_weights=True/False. prob don't need both for each?
        dict(weight_divisor=20),
        # TODO move to top of this list, after debugging prat_claws=True case?
        dict(),

        dict(one_row_per_claw=True, prat_claws=True, APL_coup_const=0),

        # TODO keep? just want to check output not changing when starting to rework
        # model bouton implementation (for one, to remove need to specify
        # Btn_num_per_glom explicitly, so I can start to actually plug in arbitrary
        # PN>bouton wPNKC and PN<>APL weights)
        dict(one_row_per_claw=True, Btn_separate=True, Btn_num_per_glom=3),

        # TODO add entry where fixed_thr is vector? do i even want to support that?
        # (yea, currently using as part of equalize_kc*=True implementation, when tuned
        # in prior step)
    ],
    # will test the connectome APL version first
    [dict(use_connectome_APL_weights=True), dict()]

) + dict_seq_product([
        # TODO (separate) test that prat_boutons=True without
        # per_claw_pn_apl_weights=False / replace_*/add_*=True gives ValueError
        dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
            use_connectome_APL_weights=True
        ),
        # TODO delete these two eventually. don't do anything really (moves things in
        # direction of uniform [i.e. non-connectome-APL] model, but no real change)
        dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
            use_connectome_APL_weights=True, replace_KCAPL_with_PNAPL=True,
            per_claw_pn_apl_weights=True
        ),
        dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
            use_connectome_APL_weights=True, add_PNAPL_to_KCAPL=True,
            per_claw_pn_apl_weights=True
        ),
        #
    ],
    # TODO delete
    # NOTE: pn_apl_scale_factor will get set back to 1 for all cases other than
    # prat_boutons=True alone (w/ default per_claw_pn_apl_weights=False)
    # TODO TODO TODO fix underlying scaling to not manually need this factor for
    # pn_claw_to_APL=False case? (don't want polluting dir / test names for things it's
    # not relevant for)
    # TODO TODO find sufficient value. 200 wasn't enough despite old log having ratio of
    # max_bouton_apl_drive / max_kc_apl_drive of ~500 typically
    # TODO add an extra 0?
    # TODO TODO maybe log mean PN weight(s) somewhere, for reference (in case how i
    # specify scale factor changes)? in output params? (+ in plots?)
    #[dict(pn_claw_to_APL=True), dict(pn_apl_scale_factor=1000)]
    #
    # TODO want both?
    [dict(pn_claw_to_APL=True), dict()]
) + [

    dict(pn2kc_connections='uniform', n_claws=7),

    # TODO also test 'caron'? 'hemidraw'? (though neither currently used, and would want
    # to re-implement hemidraw to fix some issues anyway...)

    # TODO delete one?
    dict(pn2kc_connections='fafb-left', weight_divisor=12),
    dict(pn2kc_connections='fafb-right', weight_divisor=12),

    # TODO also test some version(s) w/ compartmented APL? the only version of that code
    # that so far (2026-01-19) ever made sense, was the path where the different APL
    # comparments had no coupling between them

    # TODO add _use_matt_wPNKC=True? (or leave to it's own test?)
]
# TODO support pytest.param objects in dict_seq_product, xfail stuff as needed?
# (had previously post-processed this list, when needed, to xfail prat_claws=False
# one_row_per_claw=True cases)

N_TEST_SEEDS: int = 2
# TODO change type hint of output, now that we sometimes have pytest.param wrappers
# (if keeping...) (ParameterSet, but not sure best way to get that type. type(...)
# returns _pytest.mark.structures.ParameterSet currently)
def get_fitandplot_model_kw_list(model_kw_list: List[ParamDict]) -> List[ParamDict]:
    """Processes `MODEL_KW_LIST` into arg list suitable to test `fit_and_plot_mb_model`
    """
    fitandplot_model_kw_list = []
    for kws in model_kw_list:
        if isinstance(kws, dict):
            pn2kc_connections = kws.get('pn2kc_connections')
            assert 'n_seeds' not in kws
            if pn2kc_connections in variable_n_claw_options:
                kws = dict(kws)
                # NOTE: not setting this for test that calls fit_mb_model, since that
                # call only ever returns output of one seed
                kws['n_seeds'] = N_TEST_SEEDS
        else:
            # TODO assert it's already a pytest.param (ParameterSet)? still add mark if
            # not already marked for xfail?
            pass

        fitandplot_model_kw_list.append(kws)
    return fitandplot_model_kw_list

FITANDPLOT_MODEL_KW_LIST: List[ParamDict] = get_fitandplot_model_kw_list(MODEL_KW_LIST)

QUICK_MODEL_KW_LIST: List[ParamDict] = dict_seq_product(
    [dict(one_row_per_claw=True, prat_claws=True), dict(weight_divisor=20),],
    # will test the connectome APL version first
    [dict(use_connectome_APL_weights=True), dict()]
)[::-1] + [
    # TODO delete?
    #dict(one_row_per_claw=True),
    #
    dict(pn2kc_connections='uniform', n_claws=7),
    # this one is quite slow actually, w/ current learning rate params...
    # TODO TODO TODO also, why currently failing fixed_inh_params_fitandplot with:
    # following keys had mismatched values:
    # k='megamat_sparsity': v=0.09068061404700448 not equals v2=0.009985056378209482
    # assert False
    #  +  where False = equals(0.09068061404700448, 0.009985056378209482)
    # k='sparsity': v=0.09068061404700448 not equals v2=0.009985056378209482
    # assert False
    #  +  where False = equals(0.09068061404700448, 0.009985056378209482)
    # FAILED
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> traceback
    # test/test_mb_model.py:1602: in test_fixed_inh_params_fitandplot
    #     ???
    # test/test_mb_model.py:685: in assert_fit_and_plot_outputs_equal
    #     assert_param_dicts_equal(params, params2, **kwargs)
    # test/test_mb_model.py:547: in assert_param_dicts_equal
    #     assert False
    # E   assert False
    # TODO TODO TODO is it just that i'm not setting the PN<>APL stuff in python?
    dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
        use_connectome_APL_weights=True
    ),
]
QUICK_FITANDPLOT_MODEL_KW_LIST: List[ParamDict] = get_fitandplot_model_kw_list(
    QUICK_MODEL_KW_LIST
)

# TODO some mechanism in conftest.py to add CLI arg that can swap this in for all
# tests that use MODEL_KW_LIST (and same for corresponding two FITANDPLOT vars)?
# (-> delete this hack) (maybe i'm fine just using an env var for this? any issues?)
if QUICK:
    print()
    print('USING QUICK[_FITANDPLOT]_MODEL_KW_LIST (because QUICK=True)!!!')
    print()
    MODEL_KW_LIST = QUICK_MODEL_KW_LIST
    FITANDPLOT_MODEL_KW_LIST = QUICK_FITANDPLOT_MODEL_KW_LIST

if not PLOT:
    print()
    print('NOT TESTING PLOTTING CODE (because PLOT=False)!')
    print()
#

# so can work w/ pytest called from repo root, but also w/ scripts like
# generate_reference_outputs_for_repro.py, which I've been calling from this directory.
test_dir = Path(__file__).resolve().parent
sent_to_remy = test_dir.parent / 'data/sent_to_remy'

paper_hemibrain_output_dir = sent_to_remy / ('2025-03-18/'
    'dff_scale-to-avg-max__data_pebbled__hallem-tune_False__pn2kc_hemibrain__'
    'weight-divisor_20__drop-plusgloms_False__target-sp_0.0915'
)

# TODO refactor this handling of test data path? also used in test_al_analysis.py
test_data_dir = test_dir / 'test_data'

# TODO set `al_util.verbose = True` for all these (at least, so long as that's only
# way to get olfsysm log output printed?) (or add a new way to configure that, and use
# that) (doing so for now)
al_util.verbose = True

# TODO move to model_mb? (or hong2p.types?)
FitMBModelOutputs = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ParamDict]

def _orn_deltas() -> pd.DataFrame:
    """Gives same output as the `orn_deltas` fixture, but can be called directly
    (in `generate_reference_outputs_for_repro.py`), unlike the fixture-wrapper version,
    which will cause an error.
    """
    # panel          megamat   ...
    # odor           2h @ -3   ...   benz @ -3    ms @ -3
    # glomerulus               ...
    # D            40.363954   ...   42.445274  41.550370
    # DA2          15.144943   ...   12.363544   3.856004
    # ...
    # VM7d        108.535394   ...   58.686294  20.230297
    # VM7v         59.896953   ...   13.250292   8.446418
    orn_deltas = pd.read_csv(paper_hemibrain_output_dir / 'orn_deltas.csv', header=[0,1],
        index_col=0
    )
    assert orn_deltas.columns.names == ['panel', 'odor']
    assert orn_deltas.index.names == ['glomerulus']
    return orn_deltas


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
    return _orn_deltas()


# TODO cache outputs for same args+kwargs? can i do this as a fixture? or just some
# generic mechanism? (+ using indirect=True on parametrize calls, passing the args to
# this? that sufficient to cache?)
# maybe something that uses functools cache, but sorts args + kwargs and converts them
# all to hashable first? seems it requires them to be hashable, and order to be same
# for using functools cache w/ dict args: https://stackoverflow.com/questions/6358481
# prob more trouble than worth... (frozendict [3rd party] even work w/ DataFrame values,
# like orn_deltas? or can i assume orn_deltas will not change, and then exclude certain
# things?)
def _fit_mb_model(*args, **kwargs) -> FitMBModelOutputs:
    # TODO move this prints into fit_mb_model, under a verbose flag?
    print('running fit_mb_model...', flush=True)
    # TODO TODO still include **PLOT_KWS if we have plot_dir (non-None)
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
        assert params.keys() == params2.keys(), (f'{diff_sets(params, params2)}\n'
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
            # TODO keep this, not that i have convert_dtypes=True path of
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


# TODO option (or diff fn?) to not pass in param dicts, if we are just checking two
# directories on disk against each other? make params[2] Optional, and load if not
# passed?
def assert_fit_and_plot_outputs_equal(params: ParamDict, params2: ParamDict,
    plot_root: Path, *, plot_root2: Optional[Path] = None, **kwargs) -> None:
    # TODO doc which outputs it checks
    """Asserts spike_counts, w[APLKC|KCAPL] weights, (most) params, and more are equal.

    Also asserts set of CSVs and pickles are the same in the two directories, and checks
    contents of all same-named pickled match across the two directories. Currently does
    NOT check content of all same-named CSVs like this.

    Args:
        plot_root: if passed, assumes 'output_dir' value in both input dicts is a
            name of a directory under this

        plot_root2: if NOT passed, assumes 'output_dir' value in both input dicts is a
            name of a directory under `plot_root`. If passed, `params2['output_dir']`
            will be under `plot_root2`

        **kwargs: passed thru to first `assert_param_dicts_equal` call, that directly
            compares `params` and `params2` inputs.
    """
    assert plot_root.is_dir(), f'{plot_root=} not a directory'

    output_dir = (plot_root / params['output_dir']).resolve()
    assert output_dir.is_dir(), f'{output_dir=} not a directory'

    if plot_root2 is None:
        plot_root2 = plot_root
    else:
        assert plot_root2.is_dir(), f'{plot_root2=} not a directory'

    output_dir2 = (plot_root2 / params2['output_dir']).resolve()
    assert output_dir2.is_dir(), f'{output_dir2=} not a directory'

    assert output_dir != output_dir2

    def check_pickle_and_parquet_outputs(name):
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

    # TODO assert pickles has at least some minimum set of pickles?
    output_dir_pickles = filenames_with_ext(output_dir, 'p')
    output_dir2_pickles = filenames_with_ext(output_dir2, 'p')
    # TODO TODO is kc_spont_in.p still getting saved? still want it to be?
    # (failed for uniform case)
    assert output_dir_pickles == output_dir2_pickles, \
        f'diff_sets(output_dir_pickles, output_dir2_pickles)'

    output_dir_parquets = filenames_with_ext(output_dir, 'parquet')
    output_dir2_parquets = filenames_with_ext(output_dir2, 'parquet')
    assert output_dir_parquets == output_dir2_parquets, \
        f'diff_sets(output_dir_parquets, output_dir2_parquets)'

    # TODO delete eventually
    parquet_names = {x[:-len('.parquet')] for x in output_dir_parquets}
    pickle_names = {x[:-len('.p')] for x in output_dir_pickles}
    # TODO anything else?
    assert pickle_names - parquet_names == {'params_for_csv'}
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
    # TODO just define from what is only in pickle (not parquet)
    exclude_pickles = ('params_for_csv.p',)

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

    # TODO don't just thread kwargs thru all these kwargs? only manually pass
    # expected_missing_keys and ignore others for now? (esp since only doing on this
    # first call)
    assert_param_dicts_equal(params, params2, **kwargs)

    # comparing two CSVs, we should not have to worry about expected_missing_csv_keys.
    # assuming passed in expected_missing_keys (in kwargs) are still relevant.
    a2 = read_series_csv(output_dir / 'params.csv')
    b2 = read_series_csv(output_dir2 / 'params.csv')
    a2 = a2.to_dict()
    b2 = b2.to_dict()

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
    assert_param_dicts_equal(a2, b2, **kwargs)

    # if loading the params again, why passing in? can these differ at all from input?
    # (yes, the passed in params will contain the most, with the CSVs often containing
    # almost all of those [only missing likely some minor ones, like 'pearson', or
    # 'wAPLKC'/'wKCAPL' in a special subset of the cases where they are Series cached to
    # their own files]).
    a1 = read_param_cache(output_dir)
    b1 = read_param_cache(output_dir2)
    #
    # TODO maybe don't add to expected_missing_keys, and just replace it?
    expected_missing_keys = set(kwargs.pop('expected_missing_keys', tuple()))
    # should no longer be serializing any of these in pickle, so as long as the 2nd arg
    # is the older output, should be ok here
    expected_missing_keys |= set(k for k, v in b1.items()
        if isinstance(v, pd.DataFrame) or isinstance(v, pd.Series)
    )
    # TODO ok to also pass kwargs here? (may need to separately expose just
    # ignore_tuning_params, if not)
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
    else:
        assert 'wKCAPL' in params
        assert params['wAPLKC'] is not None
        assert params['wKCAPL'] is not None

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
# negative claw activities (and that there are none w/ =False), and also cover
# APL_coup_const != -1 case, where i currently haven't implemented
# allow_net_inh_per_claw=False (but xfail that here for now, and then get it to work
# later, by refactoring olfsysm)
# TODO + test that orn activities (after adding spont) can't be negative (but seems like
# C++ code might allow them to be? fix that?

# TODO TODO TODO fix (broken as of 2026-02-13)
@pytest.mark.xfail(
    reason='broke in Feb 2026. fix!', run=False
)
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
            # TODO TODO need hardcode_initial_sp to repro? maybe just relax
            # tolerance? (currently getting 0.19137 for
            # dict(use_connectome_APL_weights=True), though... isn't that a bit high?)
            # TODO TODO or just increase sp_acc a bit here?
            #
            # (only other case here [dict()] seems to not fail here though.
            # rel_abs_change=0.01406 there)
            # TODO TODO if we didn't tune separately, and we just used the same
            # thr/weights, would that make outputs any more consistent? worth trying?
            assert rel_abs_change < 0.0175

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
    p2 = read_param_csv(model_output_dir)

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
    from_pickle = read_param_cache(model_output_dir)
    pickle_keys = set(from_pickle.keys())

    # this could only happen in the single current special case in param_dict
    # non-serializable output saving loop (that pops + saves DataArray/DataFrames/etc),
    # where wAPLKC/wKCAPL are not popped for one particular set of parameters
    # (`one_row_per_claw and not use_connectome_APL_weights`)
    assert pickle_keys - csv_keys <= set(expected_missing_csv_keys)
    # CSV will always have many more entries than the pickle. model_kws/etc
    assert len(csv_keys - pickle_keys) > 0


# TODO also want a test checking output of fit_mb_model and fit_and_plot_mb_model calls
# are equiv, for same input? prob not too important
# TODO TODO still want separate test that checks we can load output for all of these?
# (at least all used by downstream in model_mb_responses or natmix_data/analysis.py)
# (or prob just incorporate into test below that we can load and also check those
# equiv? could prob only check params, or whatever else was included in there? other
# outputs are just in terms of serialized data files, so nothing to check against,
# right?)
# TODO also test return_dynamics=True + make_plots=True + plot_example_dynamics=True
# paths, for all, and check that they also all work w/ input data not from megamat
# (fit_mb_model had for a while only been running a large part of make_plots code if
# panel was megamat...)
#
# TODO TODO TODO fix:
# (same issue for `test_fixed_inh_params`. no other cases currently failing to converge)
# one-row-per-claw_True__prat-claws_True__prat-boutons_True__pn-claw-to-APL_True__connectome-APL_True
# TODO TODO TODO wait, why is this converging now? i wouldn't have thought anything
# changed?
# TODO TODO TODO is total_n_boutons set in this code currently? is it actually doing
# anything (should err or something, if not add- / replace- / bouton dynamics flags
# set?)?
# ValueError: libolfsysm/src/olfsysm.cpp:1172 in `sparsity_nonconvergence_failure` check
# `rv.kc.tuning_iters <= p.kc.max_iters` failed
@pytest.mark.parametrize('kws', FITANDPLOT_MODEL_KW_LIST, ids=format_model_params)
def test_fixed_inh_params_fitandplot(tmp_path, orn_deltas, kws):
    """
    Like test_fixed_inh_params, but calling (+ checking outputs of)
    fit_and_plot_mb_model instead of fit_mb_model.
    """
    plot_root = tmp_path

    params = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, **kws)

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

    assert_fit_and_plot_outputs_equal(params, params2, plot_root,
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
    # TODO TODO already have some fn for asserting we have some minimum set of files?
    # refactor to make one, if not?


    # TODO also first check just wPNKC, via call to connectome_WPNKC, or just handle all
    # via call to fit_and_plot_mb_model (will prob at least start with latter...)

    expected_missing_csv_keys = get_expected_missing_csv_keys(kws)

    # TODO update outputs + restore this to dict()
    # TODO need anything else to repro current outputs? hopefully things are still
    # reproducible w/o too much effort...
    extra_repro_kws = dict(hardcode_initial_sp=True)
    #extra_repro_kws = dict()
    #
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

    params2 = read_param_csv(ref_model_output_dir)

    assert_fit_and_plot_outputs_equal(params, params2, plot_root,
        plot_root2=reference_output_dir,
        # TODO not actually sure this works...
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
    params = read_param_csv(wPNKC_dir)

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

# TODO TODO test like this for paper uniform outputs (hemidraw prob less important b/c
# it's flawed and i think we'll likely leave out of submitted version, unless we end up
# making it match new wPNKC (from weight_divisor=20) better)
#
# TODO also check we can repro 2025-03-19 validation2 (hemibrain) outputs?
# 2025-02-19/validation2_hemibrain_model*.csv(s)? what are the CSVs i should check
# against?
def test_hemibrain_paper_repro(tmp_path, orn_deltas):
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

    # TODO which to use?
    #
    # {'fixed_thr': 268.0375322649455, 'wAPLKC': 4.622950819672131, 'wKCAPL':
    # 0.0025165763852325156, 'sp_acc': 0.1, 'max_iters': 10, 'sp_lr_coeff': 10.0,
    # 'apltune_subsample': 1, 'tuning_iters': 1}
    a1 = read_param_cache(paper_hemibrain_output_dir)

    # {'fixed_thr': 268.0375322649456, 'wAPLKC': 4.306010928961749, 'wKCAPL':
    # 0.002344045143691752, 'sp_acc': 0.1, 'max_iters': 10, 'sp_lr_coeff': 10.0,
    # 'apltune_subsample': 1, 'tuning_iters': 1}
    b1 = read_param_cache(output_dir)

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

    a2 = read_series_csv(paper_hemibrain_output_dir / 'params.csv')
    b2 = read_series_csv(output_dir / 'params.csv')
    # TODO TODO am i no longer including params for dff2spiking_*? is that a mistake?
    # as of 2025-10-12:
    # ipdb> a2.index.difference(b2.index)
    # Index(['dff2spiking_add_constant', 'dff2spiking_scaling_method_to_use',
    #        'dff2spiking_separate_inh_model', 'pn2kc_connections',
    #        'tune_on_hallem'],
    #       dtype='object')
    # ipdb> b2.index.difference(a2.index)
    # Index(['drop_kcs_with_no_input'], dtype='object')

    a2 = a2.to_dict()
    b2 = b2.to_dict()
    assert_param_dicts_equal(a2, b2, only_check_overlapping_keys=True,
        check_with_allclose=check_with_allclose
    )
    #

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
# TODO mark this test as slow (~10min) (or a variant of it that actually has a call that
# generates all 100 seeds. part configuring that below is currently commented. just
# checking we can recreate outputs for first 2 seeds)
def test_uniform_paper_repro(tmp_path, orn_deltas):
    """Similar purpose to `test_hemibrain_paper_repro`, but for uniform wPNKC outputs.
    """
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

    params = read_series_csv(may29_dir / 'megamat_uniform_model_params.csv')

    kws = dict(
        pn2kc_connections='uniform', n_claws=7, target_sparsity=0.0915,
        # TODO check we actually need hardcode_intial_sp=True
        drop_kcs_with_no_input=False, hardcode_initial_sp=True,
    )

    plot_root = tmp_path

    # TODO move n_seeds=100 to separate case? or only if called separate, explicit way?
    # NOTE: this product() should give us all n_seeds=2 cases before the =100 ones
    #for n_seeds, _drop_glom_with_plus in product([2, 100], [False, True]):
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
        params2 = read_series_csv(output_dir / 'params.csv')

        if not _drop_glom_with_plus:
            # TODO replace w/ eval_and_check_[err|compatible|equal]? (or a wrapper that
            # also works w/ non-str input?) (eval requires str input) (/delete)
            # TODO switch handling? does de-serialize as a string, which needs eval'd...
            # (to array, or what?)
            wAPLKC = eval(params['wAPLKC'])
            wAPLKC2 = eval(params2['wAPLKC'])
            assert wAPLKC[:len(wAPLKC2)] == wAPLKC2

            wKCAPL = eval(params['wKCAPL'])
            wKCAPL2 = eval(params2['wKCAPL'])
            assert wKCAPL[:len(wKCAPL2)] == wKCAPL2

            # TODO TODO check other parts of params (vs params2) as well
            # (use assert_param_dicts_equal?)
            # TODO delete
            #breakpoint()
            #

            # TODO TODO TODO move this check before check of APL<>KC weights, as might
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

# TODO TODO TODO add test that we can recreate use_connectome_APL_weights=False path by
# hardcoding all 1 (or all some constant. may need to use const def from olfsysm?)
# for _wAPLKC and _wKCAPL

# TODO add test that target_sparsity_factor_pre_APL is working, both w/ default of 2.0,
# as well as maybe 1-2 other reasonable values (two calls, one w/
# `mp.kc.tune_apl_weights = False`? might be easiest way unless i end up implementing
# sparsity calculation + saving both before and after APL)
# (see some other comments circa sp_factor_pre_APL code in mb_model.fit_mb_model)

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

# TODO (delete. not sure i care to revisit this code) add test that APL_coup_const=0
# (which should have 2 separate *uncoupled* APL's, according to default center/surround
# compartmentalization) has different activity than w/o it (should be covered if i add a
# test that checks all elements in list below have distinct responses)
# (and test that it's in some way as expected? which compartment has more APL
# inhibition, if either? or more total input? either inhibited more? [not so sure it's
# easy to reason about balance of excitation and inhibition...])

# TODO test we can start tuning at arbitrary tuning iters (w/ weights + scale params
# set appropriately), and still get to same final output
