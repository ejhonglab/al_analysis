#!/usr/bin/env python3

from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import pytest
import math

from hong2p.util import pd_allclose

import al_util
from mb_model import (fit_mb_model, fit_and_plot_mb_model, connectome_wPNKC, KC_ID,
    KC_TYPE, step_around, read_series_csv
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

# TODO add tests checking the key parts of example scripts for sam / ruoyi / george /
# yang all still work (how to organize?)

@pytest.fixture(scope='session')
# the name of this function is the name of the variable made accessible to test using
# this fixture. the name of the returned variable is not important.
#
# TODO rename to megamat_orn_deltas or something? (/ at least doc where this data came
# from)
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


def _fit_mb_model(*args, **kwargs):
    # TODO move this prints into fit_mb_model, under a verbose flag?
    print('running fit_mb_model...', flush=True)
    ret = fit_mb_model(*args, **kwargs)
    print('done', flush=True)
    return ret


def _fit_and_plot_mb_model(*args, **kwargs):
    return fit_and_plot_mb_model(*args, try_cache=False, **kwargs)


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
    _, spike_counts, _, params = _fit_mb_model(orn_deltas=orn_deltas,
        homeostatic_thrs=True
    )

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

    _, spike_counts2, _, params2 = _fit_mb_model(orn_deltas=orn_deltas, fixed_thr=thrs,
        wAPLKC=wAPLKC
    )
    assert spike_counts.equals(spike_counts2)


def test_spatial_wPNKC_equiv(orn_deltas):
    """
    Tests that one-row-per-claw wPNKC can recreate one-row-per-KC hemibrain outputs,
    where latter previously had a count of #-of-claws in each wPNKC matrix element, if
    olfsysm is modified and configured to interpret input rows as claws, but still
    having olfsysm ignore claw coordinate information.
    """
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

        # if we don't add these elements, the two KCs with no claws will be in wPNKC's
        # index, but not final wPNKC_one_row_per_claw index
        if n_claws_per_glom.sum() == 0:
            one_hot_claw_series_list.append(pd.Series(index=wPNKC.columns, data=False,
                name=(kc_id, kc_type, claw_id)
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
    # since two KCs receive no input (in output of connectome_wPNKC call above, w/
    # essentially default args)
    assert len(one_hot_claw_series_list) - 2 == n_claws

    # values will be True if there is a claw for the (kc,glom) pair, otherwise False
    wPNKC_one_row_per_claw = pd.concat(one_hot_claw_series_list, axis='columns',
        verify_integrity=True
    )
    # values are the elements of the 2-tuples from the .name of each concatenated Series
    wPNKC_one_row_per_claw.columns.names = ['kc_id', 'kc_type', 'claw_id']

    # AFTER .T, rows will be ['kc_id', 'claw_id'] and columns will be 'glomerulus',
    # with claw_id values going from [0, <#-claws-per-(kc,glom)-pair> - 1]
    wPNKC_one_row_per_claw = wPNKC_one_row_per_claw.T.copy()

    # TODO move/copy assertion that there are no duplicate ['kc_id', 'claw_id']
    # combos (from _wPNKC_one_row_per_claw=True code in mb_model) here? refactor most
    # checks to share?

    # TODO replace x/y/z/ ranges w/ those from actual hemibrain data
    # (in micrometers)
    x_min, y_min, z_min = (0, 0, 0)
    x_max, y_max, z_max = (100, 100, 100)

    # .names = ['kc_id', 'claw_id']
    index = wPNKC_one_row_per_claw.index

    # best practice would in theory be passing around np.random.Generator objects (a la
    # https://stackoverflow.com/questions/68222756), but this global seed+rng should be
    # fine here.
    # NOTE: the issue is that it will also seed any numpy RNG in tests that happen
    # after, whether they were intended to be seeded or not
    np.random.seed(1)
    claw_coords = pd.DataFrame({
        # TODO change to have input that are ints in same range as neuprint input (if
        # that data is in ints... is it?), which are then multiplied by same 8/1000
        # factor to get units of micrometers
        'claw_x': np.random.uniform(x_min, x_max, len(index)),
        'claw_y': np.random.uniform(y_min, y_max, len(index)),
        'claw_z': np.random.uniform(z_min, z_max, len(index)),
    })
    for_claw_index = pd.concat([index.to_frame(index=False), claw_coords],
        axis='columns', verify_integrity=True
    )
    claw_index = pd.MultiIndex.from_frame(for_claw_index)
    # .names now ['kc_id', 'claw_id', 'claw_x', 'claw_y', 'claw_z'] with the x/y/z/
    # coord values randomly generated (deterministically)
    wPNKC_one_row_per_claw.index = claw_index

    # checking we didn't drop any claws through the one-hot-encoding process
    assert wPNKC_one_row_per_claw.groupby(['kc_id', 'kc_type']).sum().equals(wPNKC)

    _, spike_counts, wPNKC2, _ = _fit_mb_model(orn_deltas=orn_deltas, use_connectome_APL_weights = True,
        pn2kc_connections=connectome, **wPNKC_kws
    )
    assert wPNKC.equals(wPNKC2)

    # just establishing new path allowing us to hardcode _wPNKC works
    _, spike_counts2, _, _ = _fit_mb_model(orn_deltas=orn_deltas, use_connectome_APL_weights = True, _wPNKC=wPNKC)
    assert spike_counts.equals(spike_counts2)
    del spike_counts2

    # NOTE: currently will fail. need to modify this test to configure olfsysm to
    # interpret input as claws (and then need to modify olfsysm to support this new
    # configuration we decide on)
    _, spike_counts2, _, _ = _fit_mb_model(orn_deltas=orn_deltas,
        # TODO may need wPNKC_one_row_per_claw.astype(int) (or may need to modify
        # fit_mb_model to accept dtype=bool input)?
        # TODO TODO TODO implement _wPNKC_one_row_per_claw (and similar [+other required
        # changes] in olfsysm)
        use_connectome_APL_weights = True,
        _wPNKC=wPNKC_one_row_per_claw, _wPNKC_one_row_per_claw=True
    )
    assert spike_counts.equals(spike_counts2)

    # TODO add test like fit_mb_model call above, but shuffling order of wPNKC rows.
    # order of rows should not matter.
    # TODO TODO also shuffling claw_id's within each KC (that also shouldn't matter, and
    # is probably a more important test than shuffling rows)


def _read_spike_counts(output_dir: Path, *, index_col=[KC_ID, KC_TYPE]) -> pd.DataFrame:
    # TODO may need to special case reading ones with seeds (factor to mb_model at that
    # point, if gonna add more complicated logic to read header and determine from
    # there?)
    return pd.read_csv(output_dir / 'spike_counts.csv', index_col=index_col)


# TODO TODO add test that use_connectome_APL_weights=True path w/ those vectors set to
# all 1 [would have to add way of hardcoding them] (the normalized weight vectors)
# produces same effect as use_connectome_APL_weights=False (any reason it shouldn't?)?

def test_fixed_inh_params_claw(orn_deltas):
    connectome = 'hemibrain'
    # for kws in [dict(_wPNKC_one_row_per_claw = True, use_connectome_APL_weights=True), 
    #             dict(_wPNKC_one_row_per_claw = True)]:
    for kws in [dict(_wPNKC_one_row_per_claw = True, use_connectome_APL_weights=True)]:
        print(f'{kws=}')
        responses, spike_counts, wPNKC, params = _fit_mb_model(
            orn_deltas=orn_deltas, pn2kc_connections=connectome, **kws
        )
        fixed_thr = params['fixed_thr']
        assert isinstance(fixed_thr, float)

        if kws.get('use_connectome_APL_weights', False):
            # TODO so this assumes that wAPLKC is from connectome inside fit_mb_model,
            # and there is no current support for passing in vector wAPLKC, right?
            wAPLKC = params['wAPLKC_scale']
            wKCAPL = params['wKCAPL_scale']
        else:
            wAPLKC = params['wAPLKC']
            wKCAPL = params['wKCAPL']
        assert isinstance(wAPLKC, float)

        responses2, spike_counts2, wPNKC2, params2 = _fit_mb_model(
            orn_deltas=orn_deltas, fixed_thr=fixed_thr, wAPLKC=wAPLKC, wKCAPL = wKCAPL, **kws
        )

        # TODO factor all the checking below into a fn to share w/ other tests?
        
        assert responses.equals(responses2)
        assert spike_counts.equals(spike_counts2)
        assert wPNKC.equals(wPNKC2)
        print("passed the responses, spike_counts, wPNKC compare point")
        # TODO always true? also in connectome APL case?
        assert params.keys() == params2.keys()

        param_types = {k: type(v) for k, v in params.items()}
        param_types2 = {k: type(v) for k, v in params2.items()}
        # to simplify logic in loop below, where we actually check values equal
        assert param_types == param_types2

        assert params2['tuning_iters'] == 0

        # params that we'd expect to be different between the fixed thr/APL-weights call
        # and the call that picked those values
        exclude_params = ('tuning_iters',)

        # TODO factor to a util dict_equal fn? already have one (maybe in hong2p)?
        for k in params.keys():
            if k in exclude_params:
                continue

            print(f'{k=}')

            # unless assertions above about keys()+types being equal above fail,
            # assuming keys are present in both and types equal.
            v = params[k]
            v2 = params2[k]
            if hasattr(v, 'equals'):
                assert v.equals(v2)
            elif isinstance(v, np.ndarray):
                assert np.array_equal(v, v2)
            else:
                if isinstance(v, float):
                    assert math.isclose(v, v2, rel_tol=1e-14, abs_tol=0.0)
                elif isinstance(v, np.floating):
                    assert np.isclose(v, v2, rtol=1e-14, atol=0.0)

        print()


def test_fixed_inh_params(orn_deltas):
    """
    Tests that outputs of calls where APL<->KC weights are tuned to a target response
    rate can be recapitulated by similar calls setting the APL<->KC weight parameters
    (and typically also the KC threshold parameters, either `mp.kc.fixed_thr` or
    `rv.kc.thr`).
    """
    # TODO TODO move some/all of the commented stuff in mb_model regarding picking which
    # (of slightly numerically different) thr values for fixed_thr into here?
    # (or a sep test)

    # TODO does use_connectome_APL_weights=True case take multiple iterations
    # to terminate? (no! current one terminates immediately!) want at least one test
    # case that does (and probably another that terminates immediately, or after 1
    # iteration)

    # TODO one where fixed_thr is vector? do i even want to support that (yea, currently
    # using as part of equalize_kc*=True implementation, when tuned in prior step)?
    for kws in [dict(), dict(use_connectome_APL_weights=True)]:
        print(f'{kws=}')
        responses, spike_counts, wPNKC, params = _fit_mb_model(orn_deltas=orn_deltas,
            **kws
        )
        fixed_thr = params['fixed_thr']
        assert isinstance(fixed_thr, float)

        if kws.get('use_connectome_APL_weights', False):
            # TODO so this assumes that wAPLKC is from connectome inside fit_mb_model,
            # and there is no current support for passing in vector wAPLKC, right?
            wAPLKC = params['wAPLKC_scale']
        else:
            wAPLKC = params['wAPLKC']
        assert isinstance(wAPLKC, float)

        responses2, spike_counts2, wPNKC2, params2 = _fit_mb_model(
            orn_deltas=orn_deltas, fixed_thr=fixed_thr, wAPLKC=wAPLKC, **kws
        )

        # TODO factor all the checking below into a fn to share w/ other tests?

        assert responses.equals(responses2)
        assert spike_counts.equals(spike_counts2)
        assert wPNKC.equals(wPNKC2)

        # TODO always true? also in connectome APL case?
        assert params.keys() == params2.keys()

        param_types = {k: type(v) for k, v in params.items()}
        param_types2 = {k: type(v) for k, v in params2.items()}
        # to simplify logic in loop below, where we actually check values equal
        assert param_types == param_types2

        assert params2['tuning_iters'] == 0

        # params that we'd expect to be different between the fixed thr/APL-weights call
        # and the call that picked those values
        exclude_params = ('tuning_iters',)

        # TODO factor to a util dict_equal fn? already have one (maybe in hong2p)?
        for k in params.keys():
            if k in exclude_params:
                continue

            print(f'{k=}')

            # unless assertions above about keys()+types being equal above fail,
            # assuming keys are present in both and types equal.
            v = params[k]
            v2 = params2[k]

            # TODO is it a problem that we didn't need to check wKCAPL w/ allclose
            # before, and we do now?
            # NOTE: despite needing to check wKCAPL this way, didn't seem to need the
            # same for wAPLKC
            to_check_with_allclose = ('wKCAPL', 'wKCAPL_scale')
            # TODO check this way regardless of whether it's a Series?
            if k in to_check_with_allclose:
                if isinstance(v, pd.Series):
                    assert isinstance(v2, pd.Series)

                    # otherwise would just need equal_nan=True to assertion, but also
                    # currently expect no NaN
                    assert not v.isna().any().any()
                    assert not v2.isna().any().any()

                    # TODO modify pd_allclose to work w/ two float inputs too ->
                    # simplify this code a bit
                    assert pd_allclose(v, v2)
                else:
                    assert isinstance(v, float)
                    assert isinstance(v2, float)

                    assert np.allclose(v, v2)

            elif hasattr(v, 'equals'):
                assert v.equals(v2)
            elif isinstance(v, np.ndarray):
                assert np.array_equal(v, v2)
            else:
                assert v == v2

        print()


# TODO combine w/ above? /rename (to something w/ "equalize", at least?)?
def test_fixed_inh_params2(tmp_path, orn_deltas):
    """
    Tests equalize_kc_type_sparsity=True output can be reproduced by setting
    fixed_thr to appropriate vector (with a threshold for each KC) and wAPLKC as before.
    """
    # TODO also test w/ below + use_connectome_APL_weights=True? (prob not super
    # critical. shouldn't interact much w/ this)
    kws = dict(
        equalize_kc_type_sparsity=True, ab_prime_response_rate_target=0.2
    )
    plot_root = tmp_path / 'equalize_fixed_inh_test'

    param_dict = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, **kws)

    output_dir = (plot_root / param_dict['output_dir']).resolve()
    assert output_dir.is_dir()
    assert output_dir.parent == plot_root

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

    # this must be in param_dict if equalize_kc_type_sparsity=True
    type2thr = param_dict['type2thr']

    wPNKC = pd.read_pickle(output_dir / 'wPNKC.p')
    kc_types = wPNKC.index.get_level_values(KC_TYPE)
    assert not kc_types.isna().any()
    assert set(kc_types) == set(type2thr.keys())
    cell_thrs = kc_types.map(type2thr)
    fixed_thr = cell_thrs.values.copy()

    fixed_kws = {k: v for k, v in kws.items()
        if k not in ('equalize_kc_type_sparsity', 'ab_prime_response_rate_target')
    }

    if kws.get('use_connectome_APL_weights', False):
        assert 'wAPLKC' not in param_dict
        # TODO so this assumes that wAPLKC is from connectome inside fit_mb_model, and
        # there is no current support for passing in vector wAPLKC, right?
        wAPLKC = param_dict['wAPLKC_scale']
    else:
        assert 'wAPLKC_scale' not in param_dict
        wAPLKC = param_dict['wAPLKC']

    param_dict2 = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
        fixed_thr=fixed_thr, wAPLKC=wAPLKC, **fixed_kws
    )
    # TODO any assertions on contents of param_dict2 [vs param_dict?]? (e.g. on
    # fixed_thr)

    output_dir2 = (plot_root / param_dict2['output_dir']).resolve()
    assert output_dir2.is_dir()
    assert output_dir2.parent == plot_root
    df2 = _read_spike_counts(output_dir2)

    assert df.equals(df2)


def test_multiresponder_APL_boost(orn_deltas):
    # TODO refactor this handling of test data path? also used in test_al_analysis.py
    data_dir = Path(__file__).resolve().parent / 'test_data'

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
    mask = pd.read_csv(data_dir / 'kiwi-control-multiresponder-mask.csv'
        ).set_index('kc_id').squeeze()
    mask.name = None
    # TODO also assert that we initially have some cells responding to multiple odors in
    # here? or at least some spikes at all?
    assert mask.sum() > 0

    # NOTE: first implementation only supports use_connectome_APL_weights=True case
    kws = dict(use_connectome_APL_weights=True)

    _, sc1, _, params1 = _fit_mb_model(orn_deltas=orn_deltas, **kws)
    assert sc1.index.get_level_values('kc_id').equals(mask.index)

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
        target_sparsity=0.0915, weight_divisor=20, _drop_glom_with_plus=False
    )

    # TODO resolve actually needed?
    # TODO this causing issues i'm doing resolve here and not below now?
    # (maybe only in some weird cases like running two pytest processes at once?)
    plot_root = (tmp_path / 'hemibrain_paper_repro').resolve()

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

    assert a1.keys() == b1.keys()
    for k in a1.keys():
        # could remove this continue-ing after fixing source of issue
        if k in ('wAPLKC', 'wKCAPL'):
            continue
        #

        assert np.isclose(a1[k], b1[k])

    # TODO TODO TODO fix how (only) wAPLKC and wKCAPL are not isclose

    # responses_to                                                                   pebbled
    # tune_from                                                                      pebbled
    # tune_on_hallem                                                                   False
    # pn2kc_connections                                                            hemibrain
    # weight_divisor                                                                      20
    # _drop_glom_with_plus                                                             False
    # target_sparsity                                                                 0.0915
    # drop_silent_cells_before_analyses                                                 True
    # output_dir                           dff_scale-to-avg-max__data_pebbled__hallem-tun...
    # used_model_cache                                                                 False
    # fixed_thr                                                            268.0375322649455
    # wAPLKC                                                               4.622950819672131
    # wKCAPL                                                           0.0025165763852325156
    # sp_acc                                                                             0.1
    # max_iters                                                                           10
    # sp_lr_coeff                                                                       10.0
    # apltune_subsample                                                                    1
    # tuning_iters                                                                         1
    # sparsity                                                           0.09334272631208172
    # megamat_sparsity                                                   0.09334272631208172
    # dff2spiking_scaling_method_to_use                                           to-avg-max
    # dff2spiking_add_constant                                                         False
    # dff2spiking_separate_inh_model                                                   False
    # dtype: object
    a2 = read_series_csv(model_output_dir1 / 'params.csv')

    # responses_to                                                                   pebbled
    # tune_from                                                                      pebbled
    # target_sparsity                                                                 0.0915
    # weight_divisor                                                                      20
    # _drop_glom_with_plus                                                             False
    # drop_silent_cells_before_analyses                                                 True
    # output_dir                           target-sp_0.0915__weight-divisor_20__drop-plus...
    # used_model_cache                                                                 False
    # fixed_thr                                                            268.0375322649456
    # wAPLKC                                                                             NaN
    # wKCAPL                                                                             NaN
    # sp_acc                                                                             0.1
    # max_iters                                                                          100
    # sp_lr_coeff                                                                        1.0
    # apltune_subsample                                                                    1
    # tuning_iters                                                                         6
    # sparsity                                                           0.08248743155400429
    # megamat_sparsity                                                   0.08248743155400429
    # dtype: object
    b2 = read_series_csv(output_dir / 'params.csv')
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

    # TODO TODO TODO fix
    assert df1.equals(df2)


def test_one_row_per_claw(tmp_path, orn_deltas):

    kws = dict(
        # I would not normally recommend you hardcode any of these except perhaps
        # weight_divisor=20. The defaults target_sparsity=0.1 and
        # _drop_glom_with_plus=True should be fine.
        weight_divisor=20, use_connectome_APL_weights=True, _wPNKC_one_row_per_claw=True
    )

    plot_root = tmp_path / 'one_row_per_claw'

    param_dict_defined = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, fixed_thr=232.34284608546403, wAPLKC=9.5578, **kws)

    # TODO modify this fn so dirname includes all same params by default (rather than
    # just e.g. param_dir='data_pebbled'), as the ones i'm currently manually creating
    # by calls in model_mb_... (prob behaving diff b/c e.g.
    # pn2kc_connections='hemibrain' is explicitly passed there)
    param_dict = _fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, **kws)

    output_dir = (plot_root / param_dict['output_dir']).resolve()
    assert output_dir.is_dir()
    assert output_dir.parent == plot_root

    #           2h @ -3  IaA @ -3  pa @ -3  ...  1-6ol @ -3  benz @ -3  ms @ -3
    # kc_id         ...
    # 0             0.0       0.0      0.0  ...         0.0        0.0      0.0
    # 1             0.0       0.0      0.0  ...         0.0        0.0      0.0
    # ...           ...       ...      ...  ...         ...        ...      ...
    # 1835          1.0       1.0      2.0  ...         1.0        0.0      0.0
    # 1836          0.0       0.0      0.0  ...         0.0        0.0      0.0
    df = _read_spike_counts(output_dir)

    df_ids = df.index.get_level_values(KC_ID)
    assert not df_ids.duplicated().any()
    assert not df_ids.isna().any()


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

    # TODO delete
    #pdf2 = pd.read_csv(may26_dir / uniform_response_csv_name,
    #    index_col=['model_kc', 'seed']
    #)
    #assert pdf.index.equals(pdf2.index)
    #assert pdf.columns.equals(pdf2.columns)
    #assert set(np.unique(pdf.values.flat)) == set(np.unique(pdf2.values.flat)) == {
    #    0.0, 1.0
    #}
    #assert pdf.dtypes.equals(pdf2.dtypes)
    ## TODO check first 2 don't happen two match (below)
    #assert not pdf.equals(pdf2)
    #

    kws = dict(
        pn2kc_connections='uniform', n_claws=7, target_sparsity=0.0915,
    )

    # TODO resolve actually needed?
    plot_root = (tmp_path / 'uniform_paper_repro').resolve()
    assert not plot_root.exists()

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
        assert plot_root.is_dir()
        output_dir = plot_root / param_dict['output_dir']
        assert output_dir.is_dir()
        assert output_dir.parent == plot_root

        # TODO actually check this against  2024-05-16 dir contents? that seems like it
        # might be dir to use anyway? (/delete)
        wPNKC = pd.read_pickle(output_dir / 'wPNKC.p')
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

    # TODO delete?
    #assert np.array_equal(hemi, wide.values)
    #del hemi

    # TODO rename to run_model? or have a separate fn for that? take `mp` (and `rv` too,
    # or are even fit thresholds in mp?) as input (and return from fit_model?)?
    # TODO modify so i don't need to return gkc_wide here (or at least be more clear
    # about what it is, both in docs and in name)?
    responses, _, gkc_wide, _ = _fit_mb_model(tune_on_hallem=True,
        pn2kc_connections='hemibrain', _use_matt_wPNKC=True
    )
    assert gkc_wide.index.name == KC_ID
    assert np.array_equal(wide.index, gkc_wide.index)
    assert np.array_equal(responses, wide)


# TODO add test that target_sparsity_factor_pre_APL is working, both w/ default of 2.0,
# as well as maybe 1-2 other reasonable values (two calls, one w/
# `mp.kc.tune_apl_weights = False`? might be easiest way unless i end up implementing
# sparsity calculation + saving both before and after APL)
# (see some other comments circa sp_factor_pre_APL code in mb_model.fit_mb_model)


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

