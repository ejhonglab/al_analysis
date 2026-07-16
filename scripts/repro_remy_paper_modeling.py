#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from hong2p.util import pd_allclose, pd_indices_equal

from al_analysis import al_util
from al_analysis.al_util import (data_root, read_csv, read_parquet, load_megamat_dff,
    flyroi_cols, fly_cols, diag_panel_str
)
from al_analysis.mb_model import (model_mb_responses, paper_megamat_orn_deltas,
    paper_hemibrain_output_dir, paper_uniform_model_responses, read_orn_deltas,
    read_spike_counts, KC_ID
)


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--ignore-existing', action='store_true', help='will '
        're-run all models, regardless of whether outputs exist for them'
    )
    args = parser.parse_args()
    ignore_existing = args.ignore_existing
    # TODO TODO also check against what i'm loading in step05b-run-resampling.py? or
    # easier to just save (as kc_mean_megamat_corrdist.parquet written by 2E code in
    # model_mb_responses already does) and load in that step05-... script, and do the
    # check there?

    plot_root = Path('paper_repro')
    plot_root.mkdir(exist_ok=True)
    # TODO (delete) should this be erring (like `al-analysis ... -R` does), if
    # `al_util.response_stat_fn` != `np.mean`? don't think so, b/c data should be loaded
    # from precomputed outputs referenced in here, not the stuff under the uncommitted
    # analysis output directory. maybe print something about which file(s) used which
    # response stat tho? or at least doc in here?

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
    # this matches the pickle in the same directory (according to read_csv default
    # check), and the md5 of that pickle matches the md5 of the two
    # pebbled_ij_certain-roi_stats.p files I have committed now under the scripts/
    # corr_minus_orn_vs_vcf_repro directory (4c77005283d5e31c22b7bec41ed8adae).
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
    # (above still true?)

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
    df4 = read_orn_deltas(curr_output_dir / 'mean_est_spike_deltas_from-max-n2.csv')
    dff4 = read_csv(curr_output_dir / 'full_orn_dff_input_from-max-n2.csv',
        drop_old_odor_levels=drop_old_odor_levels
    )

    # TODO TODO commit both of these somewhere

    # also read the *_from-mean-n2.csv outputs i copied in there (ones since then would
    # be named *_from-max-n2.csv)
    dff5 = read_csv(curr_output_dir / 'full_orn_dff_input_from-mean-n2.csv',
        drop_old_odor_levels=drop_old_odor_levels
    )
    # TODO don't use *my* read_csv for this
    #df5 = read_csv(curr_output_dir / 'mean_est_spike_deltas_from-mean-n2.csv')
    # TODO TODO TODO assert this one matches committed one at least, if i'm unsure
    # whether current scale function is correct? (especially if i don't commit +
    # hardcode relevant files there)
    df5 = read_orn_deltas(curr_output_dir / 'mean_est_spike_deltas_from-mean-n2.csv')

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

    dff2spiking_committed_dir = data_root / 'internal/should_match_paper_dff2spiking'

    # TODO anything to even check against here? megamat subset still needs to be
    # generated separately, because of the way some of the normalization works, right?
    #committed_allpanel_deltas = read_orn_deltas(
    #    dff2spiking_committed_dir / 'mean_est_spike_deltas.csv'
    #)

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
    # correlations from (dff should match exactly. just need to confirm by regenerating
    # outputs using relevant section of her tom.py script)? use that one for
    # [load_]paper_megamat_dff fn?
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
    # ipdb> (dff3 - dff).loc[:, (dff3 - dff).max() != 0].abs().mean().mean()
    # 0.019030484580113798
    # ipdb> dff3.loc[:, (dff3 - dff).max() != 0].abs().mean().mean()
    # 0.2719786327060094
    # ipdb> dff.loc[:, (dff3 - dff).max() != 0].abs().mean().mean()
    # 0.27571867336565425

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
    # TODO TODO assert there are there expected # of both megamat and validation
    # flies, and that both have diags first (before thios is used to fit dF/F -> spiking
    # fn)
    allpanel_dff = dff5b

    allpanel_dff2 = read_csv(dff2spiking_committed_dir / 'full_orn_dff_input.csv')
    assert pd_allclose(allpanel_dff.droplevel('is_pair'), allpanel_dff2, equal_nan=True)


    old_dff2spiking_data = pd.read_csv(
        paper_hemibrain_output_dir / 'dff2spiking_model_input.csv'
    )
    new_dff2spiking_data = read_parquet(
        dff2spiking_committed_dir / 'dff2spiking_model_input.parquet'
    )
    index_cols = fly_cols + ['fly_id', 'odor', 'glomerulus']
    # old data duplicated within index_cols b/c it didn't have panel level
    # new data is not duplicated within ['panel'] + index_cols
    #
    # ok so to-avg-max_scaled_delta_f_over_f is seemingly the same in each, after
    # sorting at least
    old_dff = old_dff2spiking_data['to-avg-max_scaled_delta_f_over_f (X_train)']
    new_dff = new_dff2spiking_data['to-avg-max_scaled_delta_f_over_f']
    assert np.isclose(old_dff.sort_values(), new_dff.sort_values()).all()
    # TODO TODO is order really what matters? that seems unlikely
    print('does order in these really matter?')
    #breakpoint()

    # TODO TODO TODO if i pass allpanel_dff in instead of just megamat subset, does that
    # repro better (don't really think it should matter?)? or what about dropping
    # glomeruli_diags first?

    # aiming to just fit dF/F -> spiking fn
    # TODO TODO handle via some way other than sys.exit tho... so i don't need two calls
    # to this script (or two calls to al-analysis)
    # TODO at least make CLI flag or something (or only call this if need to generate
    # it, and warn otherwise?)
    # TODO TODO TODO call this if doesn't exist, or w/ CLI arg (and use committed data
    # as input)
    #model_mb_responses(allpanel_dff, plot_root)

    # TODO TODO TODO why not allclose again? matter (could be the difference, right?)?
    # ipdb> pd_indices_equal(dff, dff5)
    # True
    # ipdb> pd_allclose(dff, dff5, equal_nan=True)
    # False

    # TODO TODO and why is same problem there for dff vs dff3?
    # dff and dff5 closer (well, still not allclose...)?

    # TODO compare to some saved outputs under model_mb_responses dir?
    # TODO TODO why these two so diff??? df5 actualy matter? it's dff5 that should match
    # (or at least did at one point, though dff5 input uncommitted) dff3 (paper dff
    # input) which should actually match
    #df5_megamat_only = df5.loc[:, df5.columns.get_level_values('panel') == 'megamat']
    paper_deltas = paper_megamat_orn_deltas()
    # TODO TODO check something against this? after model_mb_responses call that should
    # generate it?

    # TODO drop diag or no? no, right?
    #
    # this fn currently just takes dF/F input, not ORN spike rate delta input
    # dff or dff5 should be the same input here, right? (well 3 and 5 are, and 3 should
    # be full committed paper inputs. both were also checked against more recent
    # committed all-panel inputs (recalculated in way to match, with old response calc
    # and everything))
    # TODO any option to load existing model outputs? (seems to be doing that
    # by default. add CLI option to ignore those caches?)
    # TODO TODO TODO check this is remaking 3Di/ii too (and that those correlations are
    # also dropping the correct flies, and only those. should be using >4 flies total,
    # and only dropping those on dates in remy_dates_with_little_megamat, if those are
    # loaded at all)
    # TODO TODO try to replace input w/ something committed
    # TODO TODO TODO try to fix still? giving up and add
    # input_is_already_est_spike_deltas flag, to skip all the scaling
    #model_mb_responses(dff5, plot_root, repro_remy_paper=True,
    # TODO even work?
    model_mb_responses(paper_deltas, plot_root, input_is_already_est_spike_deltas=True,
        repro_remy_paper=True, dff2spiking_cache_dir=dff2spiking_committed_dir,
        try_cache=not ignore_existing, skip_hallem_models=True,
        skip_sensitivity_analysis=True, skip_model_dynamics_saving=True
    )

    hemibrain_dirname = ('weight-divisor_20__drop-plusgloms_False__target-sp_0.09__'
        'drop-kcs-with-no-input_False__hardcode-initial-sp_True'
    )
    # TODO assert it was populated since run start too? at least if not using cache?
    curr_hemibrain_dir = plot_root / 'megamat' / hemibrain_dirname
    curr_hb_spike_counts = read_spike_counts(curr_hemibrain_dir)
    # this one just has model_kc->int [0, n-1] as index, and just odors as columns,
    # unlike newer outputs w/ actual KC IDs and other metadata
    paper_hb_spike_counts = pd.read_csv(
        paper_hemibrain_output_dir / 'spike_counts.csv', index_col='model_kc'
    )
    assert np.array_equal(curr_hb_spike_counts, paper_hb_spike_counts)

    uniform_dirname = ('pn2kc_uniform__n-claws_7__drop-plusgloms_False__target-sp_0.09'
        '__drop-kcs-with-no-input_False__hardcode-initial-sp_True__n-seeds_100'
    )
    curr_uniform_dir = plot_root / 'megamat' / uniform_dirname
    # don't actually have spike count CSV committed in uniform paper dir, just
    # responses, so will compare those
    curr_u7_responses = pd.read_csv(curr_uniform_dir / 'responses.csv',
        index_col=[KC_ID, 'seed']
    )
    paper_u7_responses = paper_uniform_model_responses()
    assert np.array_equal(curr_u7_responses, paper_u7_responses)

    curr_deltas = read_orn_deltas(plot_root / 'mean_est_spike_deltas.csv')
    # exclude the diagnostic data
    curr_deltas = curr_deltas.loc[:,
        curr_deltas.columns.get_level_values('panel') == 'megamat'
    ]
    # TODO TODO fuck, why are these still diff from paper deltas?

    # TODO TODO from loading old committed dF/F->spiking model fit pickle:
    model = pd.read_pickle(data_root / (
        'sent_to_remy/2025-03-18/dff_scale-to-avg-max__data_pebbled__hallem-tune_'
        'False__pn2kc_hemibrain__weight-divisor_20__drop-plusgloms_False__target-sp'
        '_0.0915/dff2spiking_fit.p'
    ))
    # coef below matches the 127 shown in june 3rd paper_repro/dff_vs_hallem*.pdf
    # plots...
    # TODO TODO TODO so what's issue now?
    # TODO TODO TODO new one also seems to match 127... so idk
    # ipdb> model.summary()
    # <class 'statsmodels.iolib.summary.Summary'>
    #                                  OLS Regression Results
    # =======================================================================================
    # Dep. Variable:       delta_spike_rate   R-squared (uncentered):                   0.589
    # Model:                            OLS   Adj. R-squared (uncentered):              0.589
    # Method:                 Least Squares   F-statistic:                              4534.
    # Date:                Tue, 30 Jun 2026   Prob (F-statistic):                        0.00
    # Time:                        15:29:36   Log-Likelihood:                         -17078.
    # No. Observations:                3169   AIC:                                  3.416e+04
    # Df Residuals:                    3168   BIC:                                  3.416e+04
    # Df Model:                           1
    # Covariance Type:            nonrobust
    #  ====================================================================================================
    #                                        coef    std err          t      P>|t|      [0.025      0.975]
    # ----------------------------------------------------------------------------------------------------
    # to-avg-max_scaled_delta_f_over_f   126.9782      1.886     67.338      0.000     123.281     130.675
    # ==============================================================================
    # Omnibus:                      555.402   Durbin-Watson:                   0.605
    # Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1343.654
    # Skew:                           0.975   Prob(JB):                    1.70e-292
    # Kurtosis:                       5.525   Cond. No.                         1.00
    # ==============================================================================

    breakpoint()


if __name__ == '__main__':
    main()

