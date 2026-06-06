#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

from hong2p.util import pd_allclose, pd_indices_equal

from al_analysis import al_util
from al_analysis.al_util import (data_root, read_csv, read_parquet, load_megamat_dff,
    flyroi_cols, fly_cols, diag_panel_str
)
from al_analysis.mb_model import (model_mb_responses, paper_megamat_orn_deltas,
    paper_hemibrain_output_dir
)


def main():
    plot_root = Path('paper_repro')
    plot_root.mkdir(exist_ok=True)

    # TODO avoid need for this ffs...
    al_util.verbose = True

    # TODO TODO TODO directly load the model data i sent remy (or even the corrs she
    # computed from them), and use that to replot 2e sorted KC vs model corrs for betty?
    # (current outputs seem to differ slightly, b/c some ROI redrawing for one example
    # fly i think, but probably don't differ noticeably/meaningfully, if using
    # `al-analysis -R` to generate them)

    # TODO refactor to share w/ test code i copied model_dir from?
    #
    # contains committed files:
    # - dff2spiking_model_choices.csv
    # - dff2spiking_model_input.parquet
    # - dff2spiking_model_input.csv (probably not used, in favor of parquet)
    # which together should be sufficient to recreate dF/F -> spiking model I would
    # typically use
    model_dir = data_root / 'internal'
    #

    # this directory contains both [pebbled|GH146]_ij_certain-roi_stats.csv
    old_dff_dir = data_root / 'sent_to_remy/2023-10-29'
    old_dff_csv = old_dff_dir / 'pebbled_ij_certain-roi_stats.csv'

    # TODO also pass to other dF/F loading fn? i assume that gets get to read_csv?
    # expecting this to avoid dropping is_pair level
    # TODO may then need to manually drop 'odor2' to match input from al-analysis?
    # (yup. what is dropping it in al-analysis? refactor + run that here? is it just
    # that it's only ever constructed with that level now?)
    drop_old_odor_levels = False
    dff = read_csv(old_dff_csv, drop_old_odor_levels=drop_old_odor_levels)
    assert dff.columns.names == flyroi_cols == ['date', 'fly_num', 'roi'], \
        f'{dff.columns.names=}'

    if drop_old_odor_levels:
        odor_levels = ['panel', 'odor1', 'repeat']
    else:
        dff = dff.droplevel('odor2')
        odor_levels = ['panel', 'is_pair', 'odor1', 'repeat']
    assert dff.index.names == odor_levels, f'{dff.index.names=}'

    # no flag (/alternate fn) for load_megamat_dff to load paper version
    dff2 = load_megamat_dff(drop_old_odor_levels=drop_old_odor_levels)

    assert dff.columns.equals(dff2.columns)
    assert dff.index.equals(dff2.index)
    assert dff.isna().equals(dff2.isna())

    if not drop_old_odor_levels:
        drop_ispair = False
        # al-analysis currently seems to leave this level in what it provides to
        # model_mb_responses
        if drop_ispair:
            dff = dff.droplevel('is_pair')
            dff2 = dff2.droplevel('is_pair')

    zero_fill_msg = 'dF/F expected not already 0-filled anywhere'
    assert not (dff == 0).any().any(), zero_fill_msg
    assert not (dff2 == 0).any().any(), zero_fill_msg

    # TODO avg/max diff? (eh, whatever)
    # ipdb> (dff - dff2).abs().max().max()
    # 1.4806037159186896
    #
    # ipdb> (dff - dff2).abs().mean().mean()
    # 0.05367705568917362
    #
    # ipdb> dff.abs().mean().mean()
    # 0.19532796668691751
    # ipdb> dff2.abs().mean().mean()
    # 0.24512203203145502
    #
    # ipdb> dff2.abs().max().max()
    # 4.124954272855258
    # ipdb> dff.abs().max().max()
    # 3.5434047600572

    # TODO TODO TODO why am i still unable to repro mean_est_spike_deltas, e.g.
    # pebbled_6f/pdf/ijroi/mb_modeling/mean_est_spike_deltas.parquet
    # current model seems to be `spike_rate = 127.0 * normed_dF/F` (is that not what it
    # was before? or is that the value from using sign_preserving_maxabs? is it the
    # input data, rather than the input scale that's the issue? or some cache being used
    # when it shouldn't be? recompute scale factor from (normed) dF/F and est spike rate
    # deltas [don't have the per-fly scaling nicely factored out tho, do i?]?)

    # TODO TODO TODO check current ORN input vs:
    # paper_hemibrain_output_dir / 'full_orn_dff_input.csv'
    # TODO and/or dff2spiking_model_input.csv / dff2spiking_fit.p?

    # TODO TODO TODO concat this and 2025-03-19/<dirname>/"".csv (validation2 data), and
    # pass that as input to a model_mb_responses call for fitting dF/F -> spike delta
    # fn? (then use just dff3 as input to a call for generating megamat responses)
    old2 = paper_hemibrain_output_dir / 'full_orn_dff_input.csv'

    assert paper_hemibrain_output_dir.parent.name == '2025-03-18'
    old2validation = (
        paper_hemibrain_output_dir.parent.parent / '2025-03-19' /
        '/'.join(old2.parts[-2:])
    )

    dff3 = read_csv(old2, drop_old_odor_levels=drop_old_odor_levels)
    dff3 = dff3.droplevel('odor2')

    repo_dir = Path('~/src/al_analysis').expanduser()
    curr_output_dir = repo_dir / 'pebbled_6f/pdf/ijroi/mb_modeling'
    df4 = pd.read_csv(curr_output_dir / 'mean_est_spike_deltas_from-max-n2.csv',
        index_col=0, header=[0, 1]
    )
    dff4 = read_csv(curr_output_dir / 'full_orn_dff_input_from-max-n2.csv',
        drop_old_odor_levels=drop_old_odor_levels
    )

    # also read the *_from-mean-n2.csv outputs i copied in there (ones since then would
    # be named *_from-max-n2.csv)
    dff5 = read_csv(curr_output_dir / 'full_orn_dff_input_from-mean-n2.csv',
        drop_old_odor_levels=drop_old_odor_levels
    )
    # TODO don't use *my* read_csv for this
    #df5 = read_csv(curr_output_dir / 'mean_est_spike_deltas_from-mean-n2.csv')
    df5 = pd.read_csv(curr_output_dir / 'mean_est_spike_deltas_from-mean-n2.csv',
        index_col=0, header=[0, 1]
    )

    def drop_flies_without_megamat(df: pd.DataFrame) -> pd.DataFrame:
        megamat_and_diag = df.loc[
            df.index.get_level_values('panel').isin((diag_panel_str, 'megamat'))
        ]
        flies_dropped = megamat_and_diag.loc[:, ~megamat_and_diag.loc['megamat'].isna(
            ).all()
        ]
        # now just need to drop any odors all NaN (should just be subset of diagnostic
        # odors)
        # TODO actually need to do this, or ok to pass right to model_mb_responses (for
        # both calls?)?
        return flies_dropped.loc[~flies_dropped.isna().T.all()].copy()

    # TODO or try just re-ordering and doing the not_in_old drop first (then
    # comparisons to dff3 concat, then drop_flies...)?
    dff4b = dff4.copy()
    dff5b = dff5.copy()
    dff4 = drop_flies_without_megamat(dff4)
    dff5 = drop_flies_without_megamat(dff5)

    assert dff4.index.equals(dff5.index)
    # ipdb> not_in_old
    # MultiIndex([('glomeruli_diagnostics', False, '2h @ -3', 0),
    #             ('glomeruli_diagnostics', False, '2h @ -3', 1),
    #             ('glomeruli_diagnostics', False, '2h @ -3', 2)],
    #            names=['panel', 'is_pair', 'odor1', 'repeat'])
    # TODO TODO TODO is it just the presence of this in the fitting of the dF/F -> est
    # spike delta fn that is changing things slightly?
    # TODO TODO does this automatically get included if we concat the validation2 data
    # too? (e.g. to dff3 below) (seems so?)
    not_in_old = dff5.index.difference(dff3.index)

    # shouldn't be needed now that drop_flies_without_megamat is also dropping any odors
    # all NaN in what it would return
    assert len(not_in_old) == 0

    # TODO delete
    #dff4 = dff4.drop(not_in_old)
    #dff5 = dff5.drop(not_in_old)
    assert pd_allclose(dff3, dff5, equal_nan=True)

    # TODO so it was just some small changes in ROIs? matter?
    # TODO TODO TODO which of dff3 vs dff better matches what remy is computing
    # correlations from? use that one for [load_]paper_megamat_dff fn?
    # ipdb> dff.columns[((dff3 - dff).max() != 0)]
    # MultiIndex([('2023-05-10', 1,  'DC3'),
    #             ('2023-05-10', 1,  'DC4'),
    #             ('2023-05-10', 1, 'DP1m'),
    #             ('2023-05-10', 1,  'VA6'),
    #             ('2023-05-10', 1, 'VA7l'),
    #             ('2023-05-10', 1, 'VL2a'),
    #             ('2023-05-10', 1,  'VM2'),
    #             ('2023-05-10', 1, 'VM5v')],
    #            names=['date', 'fly_num', 'roi'])

    for d2 in [dff2, dff3, dff4, dff5]:
        assert pd_indices_equal(dff, d2)
        assert dff.isna().equals(d2.isna())
        assert not (d2 == 0).any().any()

    dff3v = read_csv(old2validation, drop_old_odor_levels=drop_old_odor_levels)
    dff3v = dff3v.droplevel('odor2')
    dff3f = pd.concat([dff3, dff3v], verify_integrity=True, axis='columns')

    assert dff3f.columns.equals(dff5b.columns)
    # neither already has their index sorted (at least, in this manner. maybe odors in
    # my own custom sort order in one? hopefully/presumably order doesn't matter tho,
    # for model_mb_responses?)
    assert dff5b.index.sort_values().equals(dff3f.index.sort_values())

    assert pd_allclose(dff5b.loc[dff3f.index], dff3f, equal_nan=True)

    # TODO TODO do i also need to drop the diagnostic '2h @ -3' in the one validation2
    # fly has, in order for fit (and outputs) to better match those in the paper?
    # matter?
    allpanel_dff = dff5b

    # aiming to just fit dF/F -> spiking fn
    # TODO TODO handle via some way other than sys.exit tho... so i don't need two calls
    # to this script (or two calls to al-analysis)
    # TODO at least make CLI flag or something (or only call this if need to generate
    # it, and warn otherwise?)
    #model_mb_responses(allpanel_dff, plot_root)

    # TODO TODO TODO why not allclose again? matter (could be the difference, right?)?
    # ipdb> pd_indices_equal(dff, dff5)
    # True
    # ipdb> pd_allclose(dff, dff5, equal_nan=True)
    # False

    # TODO TODO TODO and why is same problem there for dff vs dff3?
    # dff and dff5 closer (well, still not allclose...)?

    # TODO delete
    #breakpoint()
    #

    # TODO drop diag or no? no, right?
    #
    # this fn currently just takes dF/F input,m not ORN spike rate delta input
    # TODO TODO also need roi_depths? no, right?
    # TODO skip_sensitivity_analysis=True by default, even here?
    # TODO skip_hallem_models?
    # TODO TODO dff2spiking_cache_dir? prob necessary, or need to change code.
    # currently getting:
    # No such file or directory: 'paper_repro/dff2spiking_model_choices.csv'
    # TODO response_calc_params?
    # TODO TODO TODO need to hardcode all default-changign convergence params to old
    # values? and hardcode_intial_sp=True, etc? anything else?
    # TODO TODO TODO oh, it's actualy on the first element of extra_orn_deltas, which is
    # the 18th odor
    #model_mb_responses(dff2, plot_root, repro_remy_paper=True,
    # dff or dff5 should be the same input here, right?
    # TODO any option to load existing model outputs? (seems to be doing that
    # by default. add CLI option to ignore those caches?)
    model_mb_responses(dff5, plot_root, repro_remy_paper=True,
        skip_hallem_models=True, skip_sensitivity_analysis=True,
        # TODO TODO TODO i assume this isn't the cached dF/F -> spike delta fn from
        # the paper tho? either commit + use that, or go back to trying to work from
        # spike deltas instead (would require factoring out the 2e plotting from
        # model_mb_responses...)
        # TODO TODO TODO why am i getting:
        # ```
        # ValueError: libolfsysm/src/olfsysm.cpp:995 in `sample_PN_spont` check
        # `(rv.pn.sims[i].block(0, sp_t1, row_dim, sp_t2-sp_t1).rowwise().mean().array()
        # == rv.pn.sims[0].block(0, sp_t1, row_dim,
        # sp_t2-sp_t1).rowwise().mean().array() ).all()` failed
        # ```
        # ...on first call? b/c wrong dff2spiking_cache_dir (NOPE! getting w/ dff2 as
        # input too!)? still something worth giving a better error message for?
        skip_model_dynamics_saving=True
    )

    # TODO compare to some saved outputs under model_mb_responses dir?
    # TODO delete?
    df = paper_megamat_orn_deltas()

    breakpoint()


if __name__ == '__main__':
    main()

