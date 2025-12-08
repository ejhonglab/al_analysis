#!/usr/bin/env python3

from itertools import product
from pathlib import Path
import math
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytest
import math
from pathlib import Path


from hong2p.util import pd_allclose

import al_util
from mb_model import (fit_mb_model, fit_and_plot_mb_model, connectome_wPNKC, KC_ID,
    CLAW_ID, KC_TYPE, step_around, read_series_csv, get_thr_and_APL_weights,
    get_APL_weights, variable_n_claw_options, dict_seq_product
)


# TODO set `al_util.verbose = True` for all these (at least, so long as that's only
# way to get olfsysm log output printed?) (or add a new way to configure that, and use
# that) (doing so for now)
al_util.verbose = True

# NOTE: may eventually want to revert to per-test-fn marks (via
# `@pytest.mark.filterwarnings(...)` decorator), but many modelling wrapper calls will
# emit many warnings, hence the current module-level ignore
# TODO change to only ignore warnings generated from one of my files (or those in this
# repo)? easily possible?
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

# TODO need .resolve() call? pytest only ever going to be called from repo root?
sent_to_remy = Path('data/sent_to_remy').resolve()

# TODO rename to indicate hemibrain / paper
model_output_dir1 = sent_to_remy / ('2025-03-18/'
    'dff_scale-to-avg-max__data_pebbled__hallem-tune_False__pn2kc_hemibrain__'
    'weight-divisor_20__drop-plusgloms_False__target-sp_0.0915'
)

# TODO refactor this handling of test data path? also used in test_al_analysis.py
test_data_dir = Path(__file__).resolve().parent / 'test_data'

n_test_seeds = 2

# should cover all the main paths in olfsysm, but also in fit_mb_model/etc
model_kw_list = dict_seq_product(
    [
        # TODO keep both of these?
        dict(_wPNKC_one_row_per_claw=True, prat_claws=True),
        dict(_wPNKC_one_row_per_claw=True),

        # TODO move to top of this list, after debugging prat_claws=True case
        dict(),
    ],
    [dict(), dict(use_connectome_APL_weights=True)]
) + [
    dict(pn2kc_connections='uniform', n_claws=7),
    # TODO also test some version(s) w/ compartmented APL
]

@pytest.fixture(scope='session')
# the name of this function is the name of the variable made accessible to test using
# this fixture. the name of the returned variable is not important.
#
# TODO rename to megamat_orn_deltas or something? (/ at least doc where this data came
# from)
# TODO add similar to load kiwi+control data (-> use [in tests that loop over
# model_kw_list, in additional to curr megamat orn_deltas?])
def orn_deltas() -> pd.DataFrame:
    # panel          megamat   ...
    # odor           2h @ -3   ...   benz @ -3    ms @ -3
    # glomerulus               ...
    # D            40.363954   ...   42.445274  41.550370
    # DA2          15.144943   ...   12.363544   3.856004
    # ...
    # VM7d        108.535394   ...   58.686294  20.230297
    # VM7v         59.896953   ...   13.250292   8.446418
    orn_deltas = pd.read_csv(model_output_dir1 / 'orn_deltas.csv', header=[0,1],
        index_col=0
    )
    assert orn_deltas.columns.names == ['panel', 'odor']
    assert orn_deltas.index.names == ['glomerulus']
    return orn_deltas


# TODO move these type defs to model_mb?
ParamDict = Dict[str, Any]
FitMBModelOutputs = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ParamDict]

def _fit_mb_model(*args, **kwargs) -> FitMBModelOutputs:
    # TODO move this prints into fit_mb_model, under a verbose flag?
    print('running fit_mb_model...', flush=True)
    ret = fit_mb_model(*args, **kwargs)
    print('done', flush=True)
    return ret


def _fit_and_plot_mb_model(*args, **kwargs) -> ParamDict:
    return fit_and_plot_mb_model(*args, try_cache=False, **kwargs)


def _read_spike_counts(output_dir: Path, *, index_col=[KC_ID, KC_TYPE]) -> pd.DataFrame:
    # TODO may need to special case reading ones with seeds (factor to mb_model at that
    # point, if gonna add more complicated logic to read header and determine from
    # there?)
    return pd.read_csv(output_dir / 'spike_counts.csv', index_col=index_col)


def assert_param_dicts_equal(params: ParamDict, params2: ParamDict, *,
    # TODO only check these wKCAPL params w/ allclose when explicitly requested by
    # particular tests? (check none w/ allclose by default) (or am i going to need to
    # specify wKCAPL enough that i shouldn't...?)
    check_with_allclose=('wKCAPL','wKCAPL_scale'),
    only_check_overlapping_keys: bool = False) -> None:
    """Asserts param dicts (as output by fit_mb_model/fit_and_plot_mb_model) equivalent

    Args:
        only_check_overlapping_keys: if False, will err if keys are not the same
    """
    # params that we'd expect to be different between the fixed thr/APL-weights call
    # and the call that picked those values (tuning_iters), and other special cases
    exclude_params = ('tuning_iters', 'rv', 'mp', 'output_dir')

    # NOTE: may currently fail on some outputs of fit_and_plot..., which may be popping
    # some things from param dict before returning
    # TODO TODO probably change fit_and_plot... to not pop for returned param_dict, but
    # only pop prior to saving CSV (which should be why we were popping), if so
    if not only_check_overlapping_keys:
        assert params.keys() == params2.keys()
    else:
        key_overlap = set(params.keys()) & set(params2.keys())
        params = {k: v for k, v in params.items() if k in key_overlap}
        params2 = {k: v for k, v in params2.items() if k in key_overlap}

    param_types = {k: type(v) for k, v in params.items()}
    param_types2 = {k: type(v) for k, v in params2.items()}
    # to simplify logic in loop below, where we actually check values equal
    assert param_types == param_types2

    # TODO factor to a util dict_equal fn? already have one (maybe in hong2p)?
    for k in params.keys():
        if k in exclude_params:
            continue

        # unless assertions above about keys()+types being equal above fail,
        # assuming keys are present in both and types equal.
        v = params[k]
        v2 = params2[k]

        # TODO is it a problem that we didn't need to check wKCAPL w/ allclose
        # before, and we do now?
        # NOTE: despite needing to check wKCAPL this way, didn't seem to need the
        # same for wAPLKC

        if k in check_with_allclose:
            # to handle input loaded from CSV into series, where many values still float
            # TODO can i change how i load to cast at load-time when possible?
            if type(v) is str:
                assert type(v2) is str
                # TODO TODO if this fails, may need to eval/similar (for
                # array/list-of-float), or would need to preprocess outside in those
                # cases
                v = float(v)
                v2 = float(v2)

            # otherwise, would have to specify equal_nan=True below (and expect no
            # inputs should have NaN anyway). these lines work for both float and
            # ndarray input (.any() available on output of isnan for both), and should
            # also work for Series input.
            assert not np.isnan(v).any()
            assert not np.isnan(v2).any()

            if isinstance(v, pd.Series):
                assert isinstance(v2, pd.Series)
                # TODO modify pd_allclose to work w/ two float+ndarray inputs too
                # (if it doesn't already) -> simplify this code a bit (removing separate
                # branch calling np.allclose below)
                assert pd_allclose(v, v2)
            else:
                assert (
                    (isinstance(v, float) and isinstance(v2, float)) or
                    # NOTE: this should just be necessary for current wAPLKC/wKCAPL in
                    # one_row_per_claw=True output, but I may change type of those to
                    # Series
                    (isinstance(v, np.ndarray) and isinstance(v2, np.ndarray))
                )
                assert np.allclose(v, v2)

        elif hasattr(v, 'equals'):
            assert v.equals(v2), f'{k=}'
        elif isinstance(v, np.ndarray):
            assert np.array_equal(v, v2), f'{k=}'
        else:
            assert v == v2, f'{k=}'


def assert_fit_outputs_equal(ret: FitMBModelOutputs, ret2: FitMBModelOutputs) -> None:
    responses, spike_counts, wPNKC, params = ret
    responses2, spike_counts2, wPNKC2, params2 = ret2

    # intentionally checking responses/spike_counts last, as differences in params/wPNKC
    # will often help explain why there are differences in these (and differences in
    # these are often harder to interpret)
    assert_param_dicts_equal(params, params2)
    assert wPNKC.equals(wPNKC2)
    assert responses.equals(responses2)
    assert spike_counts.equals(spike_counts2)


def assert_fit_and_plot_outputs_equal(params: ParamDict, params2: ParamDict, *,
    plot_root: Optional[Path] = None) -> None:
    # TODO need alternative to plot_root (manually passing in paths of input dirs?)?
    # (just additional Optional plot_root2?)
    """
    Args:
        plot_root: if passed, assumes 'output_dir' value in both input dicts is a
            name of a directory under this
    """
    if plot_root is not None:
        output_dir = (plot_root / params['output_dir']).resolve()
        assert output_dir.is_dir(), f'{output_dir}'

        output_dir2 = (plot_root / params2['output_dir']).resolve()
        assert output_dir2.is_dir(), f'{output_dir2}'

    assert output_dir != output_dir2

    assert_param_dicts_equal(params, params2)

    a1 = pd.read_pickle(output_dir / 'params_for_csv.p')
    b1 = pd.read_pickle(output_dir2 / 'params_for_csv.p')
    #check_with_allclose = ('fixed_thr',)
    assert_param_dicts_equal(a1, b1) #, check_with_allclose=check_with_allclose)

    assert (a1.keys() - params.keys()) == set()
    assert (b1.keys() - params.keys()) == set()

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

    assert len(a2.keys() - a1.keys()) > 0
    assert len(b2.keys() - b1.keys()) > 0
    # TODO check contents same for overlap too? could require type conversion in some
    # cases, and prob not worth...

    # TODO ever need only_check_overlapping_keys=True here?  would prob need to thread
    # thru all assert_param_dicts_equal calls in this fn if i do need here
    assert_param_dicts_equal(a2, b2)
    #assert_param_dicts_equal(a2, b2, only_check_overlapping_keys=True) #,
    #    check_with_allclose=check_with_allclose
    #)
    #

    def filenames_with_ext(output_dir: Path, ext: str) -> Set[str]:
        return {x.name for x in output_dir.glob(f'*.{ext}')}

    # TODO TODO assert pickles has at least some minimum set of pickles?
    output_dir_pickles = filenames_with_ext(output_dir, 'p')
    output_dir2_pickles = filenames_with_ext(output_dir2, 'p')
    assert output_dir_pickles == output_dir2_pickles

    # TODO assert some minimum set of CSVs?
    output_dir_csvs = filenames_with_ext(output_dir, 'csv')
    output_dir2_csvs = filenames_with_ext(output_dir2, 'csv')
    assert output_dir_csvs == output_dir2_csvs

    # params_for_csv.p is loaded and checked above (we need to ignore a subset and
    # may need to special case allclose checking on some there)
    exclude_pickles = ('params_for_csv.p',)
    for name in output_dir_pickles:
        if name in exclude_pickles:
            continue

        # TODO use something other than pd.read_pickle? may want if i ever pickle
        # xarray, but it does work w/ np.ndarray, which is only other thing i think i
        # currently have in pickles (except for dict in params_for_csv.p, which should
        # be excluded anyway)
        p1 = pd.read_pickle(output_dir / name)
        p2 = pd.read_pickle(output_dir2 / name)

        # TODO delete
        if isinstance(p1, np.ndarray):
            breakpoint()
        #
        # TODO TODO use np.array_equal if type is ndarray?
        # TODO refactor to share def w/ check_with_allclose, rather than hardcoding this
        # one pickle to use allclose for
        if name == 'wKCAPL.p':
            assert pd_allclose(p1, p2), f'{name=}'
        else:
            assert p1.equals(p2), f'{name=}'

    # other than these and params.csv (dealt with above), seems only CSVs currently are:
    # {'responses.csv', 'wPNKC.csv', 'orn_deltas.csv'} (though this may change,
    # especially if I move/duplicate something currently only in pickles to CSVs)
    df = _read_spike_counts(output_dir)
    df2 = _read_spike_counts(output_dir2)
    assert df.equals(df2)

    if 'wAPLKC' not in params:
        assert 'wKCAPL' not in params

        # TODO also check for CSVs if i also write to those in the future?
        # (+ check equiv to pickles, if so)
        assert 'wAPLKC.p' in output_dir_pickles
        assert 'wKCAPL.p' in output_dir_pickles

        # we don't need to read from both dirs, b/c we already checked pickles of same
        # name have same contents (across the two dirs) above
        wAPLKC = pd.read_pickle(output_dir / 'wAPLKC.p')
        wKCAPL = pd.read_pickle(output_dir / 'wKCAPL.p')
        # TODO TODO may need to fix some one-row-per-claw=True cases (which currently
        # may save these as ndarrays instead of Series), but prob want that
        assert wAPLKC.index.equals(wKCAPL.index)
        assert not wAPLKC.isna().any()
        assert not wKCAPL.isna().any()

        assert wAPLKC.index.equals(df.index)
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
    # TODO TODO after tianpei's code, how need to sort both to get consistent order
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
    assert_fit_outputs_equal(ret, ret2)


# TODO TODO add test that allow_net_inh_per_claw=True (the default) produces some
# negative claw activities (and that there are none w/ =False), and also cover
# APL_coup_const != -1 case, where i currently haven't implemented
# allow_net_inh_per_claw=False (but xfail that here for now, and then get it to work
# later, by refactoring olfsysm)

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
    for kws in dict_seq_product(
            # for allow_net_inh_per_claw=True cases, outputs should match exactly.
            # for new default of =False, outputs should be close but not match exactly.
            [dict(allow_net_inh_per_claw=True), dict()],

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
        # combos (from _wPNKC_one_row_per_claw=True code in mb_model) here? refactor
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
            **kws, _wPNKC=wPNKC_one_row_per_claw, _wPNKC_one_row_per_claw=True
        )
        if kws.get('allow_net_inh_per_claw', False):
            assert spike_counts.equals(spike_counts2)
        else:
            # could be possible to still match under some circumstances, but that's not
            # what we see if current tests here (so something would be up if they did
            # match)
            assert not spike_counts.equals(spike_counts2)

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
            #
            # connectome_APL_weights=False (default):
            # (spike_counts - spike_counts2).abs().sum().sum()=99.0
            # spike_counts.sum().sum()=5752.0
            # spike_counts2.sum().sum()=5801.0
            # rel_abs_change=0.017211404728789986
            rel_abs_change = (spike_counts - spike_counts2).abs().sum().sum() / (
                spike_counts.sum().sum()
            )
            # currently getting 0.0172 w/ NO connectome APL weights, and 0.0139 with
            # them.
            assert rel_abs_change < 0.0175

        # TODO add test like fit_mb_model call above, but shuffling order of wPNKC rows.
        # order of rows should not matter.
        # TODO also shuffling claw_id's within each KC (that also shouldn't matter,
        # and is probably a more important test than shuffling rows)


# TODO move to mb_model? (+ get indices from some other output in params, rather than
# having to pass spike_counts in for that?)
def get_pks_and_thr(params: Dict[str, Any], spike_counts: pd.DataFrame
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


def test_fixed_inh_params(orn_deltas):
    """
    Tests that outputs of calls where APL<->KC weights are tuned to a target response
    rate can be recapitulated by similar calls setting the APL<->KC weight parameters
    (and typically also the KC threshold parameters, either `mp.kc.fixed_thr` or
    `rv.kc.thr`).

    Only calls `fit_mb_model`
    """
    # TODO (? delete?) move some/all of the commented stuff in mb_model regarding
    # picking which (of slightly numerically different) thr values for fixed_thr into
    # here? (or a sep test)

    # TODO does use_connectome_APL_weights=True case take multiple iterations
    # to terminate? (no! current one terminates immediately!) want at least one test
    # case that does (and probably another that terminates immediately, or after 1
    # iteration)

    # TODO add model_kw_list entry where fixed_thr is vector? do i even want to support
    # that (yea, currently using as part of equalize_kc*=True implementation, when tuned
    # in prior step)?
    for kws in model_kw_list:

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
        ret = _fit_mb_model(orn_deltas=orn_deltas,
            return_olfsysm_vars=return_olfsysm_vars, **kws
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
        # TODO TODO try removing wAPLKC/wKCAPL for repro calls where we also have
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

        assert_fit_outputs_equal(ret, ret2)

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


# TODO also want a test checking output of fit_mb_model and fit_and_plot_mb_model calls
# are equiv, for same input? prob not too important
# TODO TODO still want separate test that checks we can load output for all of these?
# (at least all used by downstream in model_mb_responses or natmix_data/analysis.py)
# TODO also test return_dynamics=True + make_plots=True + _plot_example_dynamics=True
# paths, for all, and check that they also all work w/ input data not from megamat
# (fit_mb_model had for a while only been running a large part of make_plots code if
# panel was megamat...)
def test_fixed_inh_params_fitandplot(tmp_path, orn_deltas):
    """
    Like test_fixed_inh_params, but calling (+ checking outputs of)
    fit_and_plot_mb_model instead of fit_mb_model.
    """
    plot_root = tmp_path

    for kws in model_kw_list:
        pn2kc_connections = kws.get('pn2kc_connections')
        assert 'n_seeds' not in kws
        if pn2kc_connections in variable_n_claw_options:
            kws = dict(kws)
            kws['n_seeds'] = n_test_seeds

            # TODO TODO TODO delete after fixing current failure in fit_and_plot...
            # (which happens on second iteration b/c param_dir was already seen,
            # tripping assertion. why didn't this happen before, when called from
            # model_mb_responses?)
            import warnings
            warnings.warn('CURRENTLY SKIPPING VARIABLE_N_SEEDS CASE! RESTORE!')
            continue
            #

        params = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, **kws)

        # TODO TODO TODO try removing wAPLKC/wKCAPL for repro calls where we also have
        # w[APLKC|KCAPL]_scale, and make sure still passes (should never need to
        # explicitly pass in) (just remove those keys currently added in
        # get_thr_and_APL_weights?)
        thr_and_apl_kws = get_thr_and_APL_weights(params, kws)

        params2 = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
            **{**thr_and_apl_kws, **kws}
        )

        assert_fit_and_plot_outputs_equal(params, params2, plot_root=plot_root)

    # TODO TODO also test we can load all the things that either downstream stuff in
    # mb_model (e.g. in model_mb_responses) or natmix_data/analysis.py uses (and that
    # all in same type / format?) (or separate test for that?)


def test_connectome_APL_repro(orn_deltas):
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
        if hasattr(v1, 'equals'):
            assert v1.equals(v2)
        else:
            # shouldn't need np.isclose b/c it should be the same unchanged value
            assert v1 == v2

    # TODO also test including new boost_wKCAPL (w/ at least =False and =True values)
    assert params1['wAPLKC'][~mask.values].equals(params2['wAPLKC'][~mask.values])
    assert (params1['wAPLKC'][mask.values] * boost_factor).equals(
        params2['wAPLKC'][mask.values]
    )

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
        drop_kcs_with_no_input=False
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

    # TODO TODO which to use?
    #
    # {'fixed_thr': 268.0375322649455, 'wAPLKC': 4.622950819672131, 'wKCAPL':
    # 0.0025165763852325156, 'sp_acc': 0.1, 'max_iters': 10, 'sp_lr_coeff': 10.0,
    # 'apltune_subsample': 1, 'tuning_iters': 1}
    a1 = pd.read_pickle(model_output_dir1 / 'params_for_csv.p')

    # {'fixed_thr': 268.0375322649456, 'wAPLKC': 4.306010928961749, 'wKCAPL':
    # 0.002344045143691752, 'sp_acc': 0.1, 'max_iters': 10, 'sp_lr_coeff': 10.0,
    # 'apltune_subsample': 1, 'tuning_iters': 1}
    b1 = pd.read_pickle(output_dir / 'params_for_csv.p')

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

    a2 = read_series_csv(model_output_dir1 / 'params.csv')
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

    # TODO TODO add earlier check that wPNKC is the same, so that fails first (giving us
    # info), if it's only that and not some olfsysm difference that is the issue

    # TODO convert to loading the CSVs instead (to future proof this test). should also
    # be committed.
    paper_wPNKC = pd.read_pickle(model_output_dir1 / 'wPNKC.p')
    wPNKC = pd.read_pickle(output_dir / 'wPNKC.p')
    #

    paper_wPNKC = paper_wPNKC.rename_axis(index={'bodyid': KC_ID})
    assert paper_wPNKC.index.names == [KC_ID]

    assert set(wPNKC.columns) == set(paper_wPNKC.columns)
    assert wPNKC.columns.equals(paper_wPNKC.columns)

    wPNKC2 = wPNKC.droplevel(KC_TYPE)
    assert wPNKC2.index.equals(paper_wPNKC.index)

    assert wPNKC2.equals(paper_wPNKC)

    # NOTE: despite using read_pickle, this is a np.ndarray (shape (1837, 1))
    spont1 = pd.read_pickle(model_output_dir1 / 'kc_spont_in.p')

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
    idf = pd.read_csv(model_output_dir1 / 'spike_counts.csv', index_col='model_kc')

    # TODO delete if i end up modifying model_output_dir1 contents to use current KC
    # index conventions
    #
    # can't compare KC indices while model_output_dir1 contents still has [0, N-1]
    # 'model_kc' index, instead of KC_ID(='kc_id') IDs from connectome
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
        drop_kcs_with_no_input=False
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


# def test_hemibrain_matt_repro():
#     # TODO delete (may want to commit some other things from here first tho)
#     #matt_data_dir = Path('../matt/matt-modeling/data')
#     #
#     matt_data_dir = Path('data/from_matt')

#     # NOTE: was NOT also committed under data/from_matt (like
#     # hemibrain/halfmat/responses.csv below was)
#     # TODO either delete this or also commit this file?
#     #
#     # TODO fix code that generated hemimatrix.npy / delete
#     # (to remove effect of hc_data.csv methanoic acid bug that persisted in many copies
#     # of this csv) (won't be equal to `wide` until fixed)
#     # (what was hemimatrix.npy exactly tho? and why would odor responses matter for it?
#     # is it not just connectivity?)
#     #
#     # Still not sure which script of Matt's wrote this (couldn't find by grepping his
#     # code on hal), but we can compare it to the same matrix reformatted from
#     # responses.csv (which is written in hemimat-modeling.html)
#     #hemi = np.load(matt_data_dir / 'reference/hemimatrix.npy')

#     # I regenerated this, using Matt's account on hal, by manually running all the
#     # relevant code from matt-modeling/docs/hemimat-modeling.html, because it seemed the
#     # previous version was affected by the hc_data.csv methanoic acid error.
#     # After regenerating it, my outputs computed in this script are now equal.
#     df = pd.read_csv(matt_data_dir / 'hemibrain/halfmat/responses.csv')

#     # The Categoricals are just to keep order of odors and KC body IDs the same as in
#     # input. https://stackoverflow.com/questions/57177605/
#     df['ordered_odors'] = pd.Categorical(df.odor, categories=df.odor.unique(),
#         ordered=True
#     )
#     df['ordered_kcs'] = pd.Categorical(df.kc, categories=df.kc.unique(), ordered=True)
#     wide = df.pivot(columns='ordered_odors', index='ordered_kcs', values='r')
#     del df

#     # TODO delete?
#     #assert np.array_equal(hemi, wide.values)
#     #del hemi

#     # TODO rename to run_model? or have a separate fn for that? take `mp` (and `rv` too,
#     # or are even fit thresholds in mp?) as input (and return from fit_model?)?
#     # TODO modify so i don't need to return gkc_wide here (or at least be more clear
#     # about what it is, both in docs and in name)?
#     responses, _, gkc_wide, _ = _fit_mb_model(tune_on_hallem=True,
#         pn2kc_connections='hemibrain', _use_matt_wPNKC=True
#     )
#     assert gkc_wide.index.name == KC_ID
#     assert np.array_equal(wide.index, gkc_wide.index)
#     assert np.array_equal(responses, wide)


def test_hemibrain_matt_repro():
    # Paths
    matt_data_dir = Path('data/from_matt')

    # Load Matt’s reference (hemibrain/halfmat) responses as a wide table
    df = pd.read_csv(matt_data_dir / 'hemibrain/halfmat/responses.csv')

    # Keep odor/KC order identical to file order
    df['ordered_odors'] = pd.Categorical(df.odor, categories=df.odor.unique(), ordered=True)
    df['ordered_kcs']   = pd.Categorical(df.kc,   categories=df.kc.unique(),   ordered=True)

    wide = df.pivot(columns='ordered_odors', index='ordered_kcs', values='r')
    del df

    # TODO delete?
    #assert np.array_equal(hemi, wide.values)
    #del hemi

    # TODO why again did i not need _add_back_methanoic_acid_mistake=True here?
    # delete that code, if not needed to reproduce anything i currently care about?
    # (was there some reason it only mattered for uniform case, and not hemibrain?)

    # TODO also check we can recreate his uniform model outputs?
    # (something from old test script we can move in here for that?)

    # TODO rename to run_model? or have a separate fn for that? take `mp` (and `rv` too,
    # or are even fit thresholds in mp?) as input (and return from fit_model?)?
    # TODO modify so i don't need to return gkc_wide here (or at least be more clear
    # about what it is, both in docs and in name)?
    responses, _, gkc_wide, _ = _fit_mb_model(tune_on_hallem=True,
        pn2kc_connections='hemibrain', _use_matt_wPNKC=True
    )

    # KC row order must match
    assert gkc_wide.index.name == KC_ID
    assert np.array_equal(wide.index, gkc_wide.index)

    # -------- Normalize column labels to match Matt's names --------
    resp = responses.copy()
    import ipdb;ipdb.set_trace()
    # Strip concentration suffix like " @ -2"
    resp.columns = resp.columns.to_series().str.replace(r"\s*@\s*-2$", "", regex=True)

    # Expand abbreviations / naming differences observed between our outputs and Matt's CSV
    ABBREV = {
        # Stereo / specific naming
        "(-)-alpha-pinene": "a-pinene",
        "(-)-beta-caryophyllene": "(-)-trans-caryophyllene",
        "(1S)-(+)-carene": "(1S)-(+)-3-carene",

        # Shorthands for alcohols/ketones/esters/etc.
        "1-5ol": "1-pentanol",
        "1-6ol": "1-hexanol",
        "1-8ol": "1-octanol",
        "1o3ol": "1-octen-3-ol",
        "2,3-b": "2,3-butanedione",
        "2-but": "2-butanone",
        "2h": "2-heptanone",
        "t2h": "E2-hexenal",  # trans-2-hexenal

        # Common abbreviations
        "EtOH": "ethanol",
        "ace": "acetone",
        "but": "butanal",
        "ea": "ethyl acetate",
        "eb": "ethyl butyrate",
        "ep": "ethyl propionate",
        "e3hb": "ethyl 3-hydroxybutyrate",
        "fur": "furfural",
        "ha": "hexyl acetate",
        "IaA": "isopentyl acetate",
        "Lin": "linalool",
        "ma": "methyl acetate",
        "ms": "methyl salicylate",
        "pa": "pentanoic acid",
        "va": "pentanoic acid",  # aka valeric acid

        # Minor spacing / letter variants
        "4-ethylguaiacol": "4-ethyl guaiacol",
        "o-cresol": "2-methylphenol",
        "alpha-terpineol": "a-terpineol",
        "beta-myrcene": "b-myrcene",
        "benz": "benzaldehyde",
        "B-cit": "b-citronellol",
        "6al": "hexanal",
    }

    resp.columns = resp.columns.to_series().replace(ABBREV).astype(str)
    resp.columns.name = "ordered_odors"  # match wide’s column name

    # Make Matt’s table float and remove CategoricalIndex wrapper for easy alignment
    wide_f = wide.copy()
    wide_f.columns = pd.Index(wide_f.columns.astype(str), name="ordered_odors")
    wide_f = wide_f.astype(float)

    # Align rows (KC ids) explicitly
    resp = resp.reindex(index=wide_f.index)

    # Check column sets now match
    missing_in_wide = set(resp.columns) - set(wide_f.columns)
    missing_in_resp = set(wide_f.columns) - set(resp.columns)
    import ipdb;ipdb.set_trace()
    if missing_in_wide or missing_in_resp:
        # Print a readable diff to help extend ABBREV if needed
        print("Only in responses:", sorted(missing_in_wide))
        print("Only in wide:", sorted(missing_in_resp))
        raise AssertionError("Column mismatch after normalization")

    # Align exact column order
    resp = resp.loc[:, wide_f.columns]

    # Final equality check (float vs float)
    assert np.array_equal(resp.to_numpy(), wide_f.to_numpy())



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


def test_btn_expansion(orn_deltas):
    connectome = "hemibrain"
    for kws in [dict(_wPNKC_one_row_per_claw=True, use_connectome_APL_weights=True),
                dict(_wPNKC_one_row_per_claw=True),
                dict(use_connectome_APL_weights=True),
                dict()]:
        # with bouton separation
        responses1, spike_counts1, wPNKC1, param_dict1 = _fit_mb_model(
            orn_deltas=orn_deltas,
            pn2kc_connections=connectome,
            Btn_separate=True,
            preset_Btn_coord=False,
            Btn_divide_per_glom=True,
            Btn_num_per_glom=10,
            **kws
        )

        responses2, spike_counts2, wPNKC2, param_dict2 = _fit_mb_model(
            orn_deltas=orn_deltas,
            pn2kc_connections=connectome,
            Btn_separate=False,
            preset_Btn_coord=False,
            Btn_divide_per_glom=True,
            Btn_num_per_glom=10,
            **kws
        )

        # the responses and spike_count matrices should be the same
        assert responses1.equals(responses2)
        assert spike_counts1.equals(spike_counts2)

        # wPNKC is not the same, but row-wise sum should be the same.
        # test row wise sum is the same;
        # (use allclose since wPNKC1 have doubles and wPNKC2 have integers)
        assert np.allclose(wPNKC1.sum(axis=1), wPNKC2.sum(axis=1))

        summed_wPNKC1 = wPNKC1.groupby(level='glomerulus', axis=1).sum()
        assert np.allclose(summed_wPNKC1, wPNKC2)


# TODO TODO add test that we can actually run sensitivity analysis path of fit_and_plot*
# (for all model_kw_list?)

# TODO TODO add a test that loads saved model outputs, for all in model_kw_list,
# and checks we can repro those. replace some current tests w/ this?
# TODO TODO and add a script that saves current outputs for all of those (or do that in
# mb_model.main?)
# TODO prioritize getting some one-row-per-claw outputs for that test (including w/
# prat_claws=True/False)

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

