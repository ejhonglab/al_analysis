#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mb_model import fit_mb_model, fit_and_plot_mb_model, connectome_wPNKC


# NOTE: may eventually want to revert to per-test-fn marks (via
# `@pytest.mark.filterwarnings(...)` decorator), but many modelling wrapper calls will
# emit many warnings, hence the current module-level ignore
# TODO keep?
# TODO change to only ignore warnings generated from one of my files (or those in this
# repo)? easily possible?
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

model_output_dir1 = Path('data/sent_to_remy/2025-03-18/'
    'dff_scale-to-avg-max__data_pebbled__hallem-tune_False__pn2kc_hemibrain__'
    'weight-divisor_20__drop-plusgloms_False__target-sp_0.0915'
).resolve()

@pytest.fixture(scope='session')
# the name of this function is the name of the variable made accessible to test using
# this fixture. the name of the returned variable is not important.
def orn_deltas() -> pd.DataFrame:
    orn_deltas = pd.read_csv(model_output_dir1 / 'orn_deltas.csv', header=[0,1],
        index_col=0
    )
    assert orn_deltas.columns.names == ['panel', 'odor']
    assert orn_deltas.index.names == ['glomerulus']
    return orn_deltas


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

    one_hot_claw_series_list = []
    for kc_id, n_claws_per_glom in wPNKC.iterrows():
        claw_id = 0

        # if we don't add these elements, the two KCs with no claws will be in wPNKC's
        # index, but not final wPNKC_one_row_per_claw index
        if n_claws_per_glom.sum() == 0:
            one_hot_claw_series_list.append(pd.Series(index=wPNKC.columns, data=False,
                name=(kc_id, claw_id)
            ))

        for glom, n_claws_from_glom in n_claws_per_glom.items():
            while n_claws_from_glom > 0:
                # glom index -> 1 for one glomerulus (indicating one claw from that
                # glomerulus to this KC), 0 for all others. name= contents will form
                # 2-level MultiIndex w/ names=['kc_id','claw_id'], after loop.
                one_hot_claw_series = pd.Series(index=wPNKC.columns, data=False,
                    name=(kc_id, claw_id)
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
    wPNKC_one_row_per_claw.columns.names = ['kc_id', 'claw_id']

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
    assert wPNKC_one_row_per_claw.groupby('kc_id').sum().equals(wPNKC)

    # TODO refactor prints
    print('running fit_mb_model...', end='', flush=True)
    _, spike_counts, wPNKC2, _ = fit_mb_model(orn_deltas=orn_deltas,
        pn2kc_connections=connectome, **wPNKC_kws
    )
    print('done', flush=True)
    assert wPNKC.equals(wPNKC2)

    # just establishing new path allowing us to hardcode _wPNKC works
    print('running fit_mb_model...', end='', flush=True)
    _, spike_counts2, _, _ = fit_mb_model(orn_deltas=orn_deltas, _wPNKC=wPNKC)
    print('done', flush=True)
    assert spike_counts.equals(spike_counts2)
    del spike_counts2

    # NOTE: currently will fail. need to modify this test to configure olfsysm to
    # interpret input as claws (and then need to modify olfsysm to support this new
    # configuration we decide on)
    print('running fit_mb_model...', end='', flush=True)
    _, spike_counts2, _, _ = fit_mb_model(orn_deltas=orn_deltas,
        # TODO may need wPNKC_one_row_per_claw.astype(int) (or may need to modify
        # fit_mb_model to accept dtype=bool input)?
        # TODO TODO TODO implement _wPNKC_one_row_per_claw (and similar [+other required
        # changes] in olfsysm)
        _wPNKC=wPNKC_one_row_per_claw, _wPNKC_one_row_per_claw=True
    )
    print('done', flush=True)
    assert spike_counts.equals(spike_counts2)

    # TODO add test like fit_mb_model call above, but shuffling order of wPNKC rows.
    # order of rows should not matter.
    # TODO TODO also shuffling claw_id's within each KC (that also shouldn't matter, and
    # is probably a more important test than shuffling rows)


# TODO TODO add test that wAPLKC/wKCAPL work to repro fit_mb_model output, in
# use_connectome_APL_weights=True case (as well as in regular case, as i believe there
# is currently some `if checks == True` code for this in fit_and_plot..., surrounding
# sensitivity analysis)

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

def test_hemibrain_paper_repro(tmp_path):
    """
    Tests that, starting from committed estimated-ORN-spike-deltas, we can reproduce
    paper hemibrain model outputs (at least in terms of spike counts, which are among
    committed outputs).

    `tmp_path` is a pytest fixture for a temporary path.
    """
    model_output_dir1 = Path('data/sent_to_remy/2025-03-18/'
        'dff_scale-to-avg-max__data_pebbled__hallem-tune_False__pn2kc_hemibrain__'
        'weight-divisor_20__drop-plusgloms_False__target-sp_0.0915'
    ).resolve()

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

    kws = dict(
        # I would not normally recommend you hardcode any of these except perhaps
        # weight_divisor=20. The defaults target_sparsity=0.1 and
        # _drop_glom_with_plus=True should be fine.
        target_sparsity=0.0915, weight_divisor=20, _drop_glom_with_plus=False
    )

    plot_root = tmp_path / 'hemibrain_paper_repro'

    # TODO modify this fn so dirname includes all same params by default (rather than
    # just e.g. param_dir='data_pebbled'), as the ones i'm currently manually creating
    # by calls in model_mb_... (prob behaving diff b/c e.g.
    # pn2kc_connections='hemibrain' is explicitly passed there)
    param_dict = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
        try_cache=False, **kws
    )

    output_dir = (plot_root / param_dict['output_dir']).resolve()
    assert output_dir.is_dir()
    assert output_dir.parent == plot_root

    #           2h @ -3  IaA @ -3  pa @ -3  ...  1-6ol @ -3  benz @ -3  ms @ -3
    # model_kc                              ...
    # 0             0.0       0.0      0.0  ...         0.0        0.0      0.0
    # 1             0.0       0.0      0.0  ...         0.0        0.0      0.0
    # ...           ...       ...      ...  ...         ...        ...      ...
    # 1835          1.0       1.0      2.0  ...         1.0        0.0      0.0
    # 1836          0.0       0.0      0.0  ...         0.0        0.0      0.0
    df = pd.read_csv(output_dir / 'spike_counts.csv', index_col='model_kc')

    idf = pd.read_csv(model_output_dir1 / 'spike_counts.csv', index_col='model_kc')
    assert idf.equals(df)


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
    responses, _, gkc_wide, _ = fit_mb_model(tune_on_hallem=True,
        pn2kc_connections='hemibrain', _use_matt_wPNKC=True
    )
    # (i might decide to change this index name, inside fit_mb_model...)
    assert gkc_wide.index.name == 'bodyid'
    assert np.array_equal(wide.index, gkc_wide.index)
    assert np.array_equal(responses, wide)
    #print("hemibrain (halfmat) responses equal to Matt's (uniform tuning)\n")

