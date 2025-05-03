#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import pytest

from mb_model import fit_and_plot_mb_model


# NOTE: may eventually want to revert to per-test-fn marks (via
# `@pytest.mark.filterwarnings(...)` decorator), but many modelling wrapper calls will
# emit many warnings, hence the current module-level ignore
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

# TODO TODO TODO add test that confirms output same whether wPNKC spatial or not
# (split non-spatial wPNKC into one-row/col-per-claw + add fake coords -> run through
# modified model that accepts this data)
# + try w/ rows/cols shuffled too

# TODO TODO test that confirms scaling wPNKC doesn't change output (not sure it's even
# true...) (w/ appropriate tolerance on output check + tuning params) (to maybe justify
# using all synapses for wPNKC, instead of trying to group/cluster into claws)
# TODO what modifications would i need to just pass in wPNKC like this? just use the
# fit_mb_model call directly? olfsysm directly (maybe trying to serialize all other
# parameters from a call invoked normally?)

# TODO TODO move much/all of al_analysis/model_test.py into (separate) tests in here

# TODO test that the parameter i added to control what fraction of sparsity comes from
# threshold vs APL limits work as expected


def test_mb_model_hemibrain_paper_repro(tmp_path):
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

    plot_root = tmp_path / 'mb_model_hemibrain_paper_repro'

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

