#!/usr/bin/env python3

from pathlib import Path
from pprint import pformat, pprint

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from hong2p.util import pd_allclose, addlevel
from hong2p.olf import parse_odor_name

import al_util
from al_util import warn, savefig, read_csv
from mb_model import (megamat_orn_deltas, fit_and_plot_mb_model, megamat_orn_deltas,
    get_thr_and_APL_weights, format_model_params, read_orn_deltas,
    scale_dff_to_est_spike_deltas_using_hallem
)


model_tune_kws = [
    # comparison for all other model cases, to see to what extent changes to PN>KC
    # weight matrix (and potentially other changes) matter
    dict(pn2kc_connections='uniform', n_claws=7),

    dict(weight_divisor=20),
    dict(one_row_per_claw=True, prat_claws=True),
    dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
        use_connectome_APL_weights=True
    ),
]


def main():
    plot_root = Path('kai_grant_model_outputs').resolve()
    plot_root.mkdir(exist_ok=True)

    # TODO avoid need to set this to see plots being saved
    al_util.verbose = True

    df = megamat_orn_deltas(drop_diags=False)

    # no other panels in output of fn above
    #
    # indexing differently for `mdf`, since some of code using it below expects 'panel'
    # column level to be there still, but `df.loc[:, 'megamat']` would drop that level
    mdf = df.loc[:, df.columns.get_level_values('panel') == 'megamat']
    diags = df.loc[:, 'glomeruli_diagnostics']

    # run validation2 data thru dF/F -> spiking model (after scaling each fly), to run
    # 'geraniol @ -2' data thru model too? (do now have that available behind
    # mb_model.validation2_orn_deltas())

    repo_root = Path('~/src/al_analysis').expanduser().resolve()
    pebbled_dir = repo_root / 'pebbled_6f/pdf/ijroi/mb_modeling'
    megamat_dir = (pebbled_dir / 'megamat' /
        'prat-claws_True__one-row-per-claw_True__connectome-APL_True__pn-claw-to-APL_'
        'True__prat-boutons_True__target-sp_0.1/'
    )
    if megamat_dir.exists():
        print('also checking megamat_orn_deltas(drop_diags=False) output against local'
            f' analysis outputs under {pebbled_dir}'
        )
        # ipdb> df.columns.get_level_values('panel').value_counts()
        # glomeruli_diagnostics    26
        # megamat                  17
        most_recent_arbitrary_panels_deltas_csv = (
            pebbled_dir / 'mean_est_spike_deltas.csv'
        )
        df2 = read_orn_deltas(most_recent_arbitrary_panels_deltas_csv)

        # NOTE: my current (2026-04-03) copy of this, which presumably was also used to
        # generate plots I sent to Betty, does slightly differ from latest committed
        # est spike deltas (loaded above via megamat_orn_deltas).
        #
        # should be same data but using a slightly different response calculation. was
        # previously mean of 2 volumes, now it should be the sign-preserving max over
        # the same 2 volumes. my 3s odor pulses generally last about 2 volumes, at most
        # 3, sometimes slightly less than 2 volumes, at my volumetric framerates
        # downstairs.
        #
        # the main effect of the change should be that some positive values get larger,
        # and that inhibition is better preserved
        #
        # ipdb> (df.loc[:, 'megamat'] - mdf2).abs().mean().mean()
        # 5.5496534563418605
        megamat_specific_deltas_csv = megamat_dir / 'orn_deltas.csv'
        mdf2 = read_orn_deltas(megamat_specific_deltas_csv)

        # TODO also assert model choices / something indicate signed maxabs response
        # calc?  where do i have that? (+ that that output was saved around the time
        # that the mean_est_spike_deltas.csv was)

        # TODO also generate (have committed now [that's not the spike deltas tho, just
        # raw data. nvm, should have est spike deltas now too]. still check?) + add
        # check for validation outputs? (delete?)
        try:
            assert pd_allclose(df2.loc[:, 'megamat'], mdf2.loc[:, 'megamat'])
        except KeyError:
            warn(f'could not check {most_recent_arbitrary_panels_deltas_csv} against '
                f'megamat specific deltas in {megamat_specific_deltas_csv}, probably '
                'because most recent al_analysis.py run (that produced model outputs, '
                'or fit dF/F -> spiking) did not include megamat data'
            )
    #

    # TODO also add this to test_df, just to check outputs are the same (should be,
    # tho)?
    tune_df = mdf

    # TODO CO2? (would need to regen output not dropping that, if i wanted, i think)
    # (oh, no. the signed maxabs stuff has it, BUT V dropped, so might as well not have
    # it. would still need to regen [actually, could work from existing
    # ij_roi_stats.[p|csv], which still has V for that one megamat fly i have CO2 data
    # for)

    gloms = ['VA6', 'DA1', 'VA1d', 'VA1v', 'V', 'VM7d', 'DL5', 'DM1', 'DM2', 'DM4',
        'DM5'
    ]

    gloms_to_mix_with_va6 = ['DM2', 'DM4', 'DM5', 'VM7d']
    assert all(x  in gloms for x in gloms_to_mix_with_va6)

    series_list = []
    # TODO delete after checking
    series_list2 = []
    #
    for x in gloms:
        gser = pd.Series(index=df.index.copy(), name=f'{x}-300 @ 0', data=0.0)
        gser.loc[x] = 300.0
        # TODO TODO split VA1d/v and rerun extraction to get them both for diags?
        if x not in df.index:
            warn(f'glomerulus {x} not in existing ORN deltas!')
        else:
            # TODO (delete?) append glomerulus to end if not (loc already should)? need
            # to check columns handled correctly when concatenating then (check rest
            # same as if we skipped those extra glomeruli?)
            # TODO delete after checking
            gser2 = gser.copy()
            assert gser2.index.equals(df.index)
            series_list2.append(gser2)
            #

        gser.name = ('private', gser.name)
        series_list.append(gser)

    for x in gloms_to_mix_with_va6:
        # TODO '+' instead of '/'
        mser = pd.Series(index=df.index.copy(), name=f'VA6-150/{x}-150 @ 0', data=0.0)
        mser.loc['VA6'] = 150.0
        mser.loc[x] = 150.0
        mser.name = ('private-plus-VA6', mser.name)
        series_list.append(mser)

    syn_df = pd.concat(series_list, axis='columns', verify_integrity=True)
    # TODO any way to set this names w/ arg to concat call?
    # names/keys/levels=['panel','odor'] didn't seem to work..
    syn_df.columns.names = ['panel', 'odor']

    # TODO TODO assert all glomeruli in this actually run through model (how to do
    # that? need to be internal to fit_mb_model? add flag to check no input glomeruli
    # dropped?)
    syn_df = syn_df.fillna(0.0).sort_index()
    # TODO delete after checking
    o2 = pd.concat(series_list2, axis='columns', verify_integrity=True)
    # ipdb> syn_df.shape
    # (42, 11)
    # ipdb> o2.shape
    # (38, 7)
    # ipdb> set(o2.index) - set(syn_df.index)
    # set()
    # ipdb> set(syn_df.index) - set(o2.index)
    # {'VA1d', 'DA1', 'V', 'VA1v'}
    assert syn_df.droplevel('panel', axis='columns').loc[o2.index, o2.columns].equals(o2)
    #

    # TODO factor above into fn to generate test data, to make main more concise
    del df

    # dropping CO2 since lack of V in data would make this one confusing
    # TODO any other diags w/o cognate glomeruli in output? (also drop them, if so)
    #
    # dropping 'aphe @ -5', since it was the older concentration of aphe, and it is the
    # only things that needs to be dropped to remove all duplicate odor names (after
    # stripping concs). may or may not have been needed to fix some current plots in
    # fit_and_plot... / fit_mb_model (was getting a warning, at least)
    diags = diags.drop(columns=['CO2 @ 0', 'aphe @ -5'])
    # TODO or just don't drop this level w/ .loc above?
    diags = addlevel(diags, 'panel', 'glomeruli_diagnostics', axis='columns')

    # TODO add diff panel for each (private, mix, diags), to break apart in plots later?
    test_df = pd.concat([syn_df, diags], axis='columns', verify_integrity=True)
    test_df = test_df.fillna(0.0).sort_index()

    panels = list(test_df.columns.get_level_values('panel').unique())

    # TODO need any lower, for all things not to fail? (would fail if any odor response
    # rates were higher than this) (nope, seems fine. delete comment)
    response_rate_plot_max = 0.2

    dfs = []
    # TODO delete one of these, if i don't end up using both
    tuned_stat_means = []
    tuned_dfs = []
    for kws in model_tune_kws:
        print(f'{kws=}')

        # set False to regen plots from tuning output dirs (and to re-tune)
        try_cache = True
        # TODO delete. should have a hack to fix now (added keys_not_to_remove kwarg to
        # write_tuned_params, and passing it in fit_and_plot_mb_model)
        # TODO fix wAPLKC/wKCAPL saving in this case (also add hack to save these
        # params to pickle, like i had in save_... before) (-> delete this hack)
        # TODO does param_dict not have wAPLKC in
        # one-row-per-claw_True__prat-claws_True case? recent issue?
        # TODO how was recently added code to test_fixed_inh_params_fitandplot
        # not failing (the one that asserted about wAPLKC params in the two cases)?
        # refactoring of param loading/saving after break it?
        #if (kws.get('prat_claws', False) and
        #    not kws.get('use_connectome_APL_weights', False)):
        #    try_cache = False
        #

        tuned_params = fit_and_plot_mb_model(plot_root, orn_deltas=tune_df,
            try_cache=try_cache, response_rate_plot_max=response_rate_plot_max, **kws
        )
        thr_and_apl_kws = get_thr_and_APL_weights(tuned_params, kws)
        print(f'tuned thr and APL weights: {pformat(thr_and_apl_kws)}')
        # TODO + get old code to sim extra odors working (+ add test it's equiv to using
        # tow calls like this)?

        model_str = format_model_params(kws)

        tuned_model_output_dir = plot_root / tuned_params['output_dir']
        trs = pd.read_pickle(tuned_model_output_dir / 'responses.p')
        tss = pd.read_pickle(tuned_model_output_dir / 'spike_counts.p')

        # TODO delete one of these? (tuned_stat_means vs tuned_dfs)
        tuned_stat_means.append({
            'model': model_str,
            'mean_response_rate': trs.mean().mean(),
            'mean_num_spikes': tss.mean().mean(),
        })
        #
        mean_num_spikes = addlevel(tss.mean(), 'model', model_str)
        mean_num_spikes.name = 'mean_num_spikes'

        mean_response_rate = addlevel(trs.mean(), 'model', model_str)
        mean_response_rate.name = 'mean_response_rate'

        stats = pd.concat([mean_num_spikes, mean_response_rate], axis='columns',
            verify_integrity=True
        )
        tuned_dfs.append(stats)
        #

        # when trying to run all at once, currently getting:
        # Warning: returning from fit_and_plot_model before making plots, because
        # input had multiple panels (currently unsupported)
        #fit_and_plot_mb_model(plot_root, orn_deltas=test_df, **kws, **thr_and_apl_kws)
        # ...so having to loop over panels below, w/ a separate call for each
        # TODO fix to not require that? would complicate plotting tho...

        # TODO actually an issue for any plots i care about?
        # TODO fix: (fixed by dropping earlier aphe concentration from diags above)
        # Warning: _strip_index_and_col_concs failed with AssertionError! probably have multiple concs for some odors. skipping rest of plots.
        # (prob by dropping diags w/ dupe concs first)
        # TODO one plot (/axes) per panel
        for panel in panels:
            panel_dir = plot_root / panel
            panel_dir.mkdir(exist_ok=True)
            panel_df = test_df.loc[:,test_df.columns.get_level_values('panel') == panel]
            # TODO important to preserve panel column level, rather than using .loc?
            # prob/hopefully not? (seems like it might be. fix!!! got:
            # ValueError: coordinate stim has dimensions ('odor',), but these are not a
            # subset of the DataArray dimensions ['stim', 'glomerulus', 'time_s']
            # ...that seemed to go away when preserving panel level
            # TODO restore try_cache=False?
            params = fit_and_plot_mb_model(panel_dir, orn_deltas=panel_df,
                try_cache=True, response_rate_plot_max=response_rate_plot_max, **kws,
                **thr_and_apl_kws
            )
            model_output_dir = panel_dir / params['output_dir']
            rs = pd.read_pickle(model_output_dir / 'responses.p')
            ss = pd.read_pickle(model_output_dir / 'spike_counts.p')

            def add_metadata(series):
                series = addlevel(series, 'model', model_str)
                return addlevel(series, 'panel', panel)

            # TODO do just w/in responders? (eh, B would probably not want that)
            mean_num_spikes = add_metadata(ss.mean())
            mean_num_spikes.name = 'mean_num_spikes'

            mean_response_rate = add_metadata(rs.mean())
            mean_response_rate.name = 'mean_response_rate'

            stats = pd.concat([mean_num_spikes, mean_response_rate], axis='columns',
                verify_integrity=True
            )
            dfs.append(stats)

        print()

    df = pd.concat(dfs, verify_integrity=True)
    stat_names = list(df.columns)
    df = df.reset_index()

    model_str2abbrev = {
        'weight-divisor_20': 'wd20',
        'pn2kc_uniform__n-claws_7': 'uniform',
        'prat-claws_True': 'prat-claws',
        'prat-claws_True__prat-boutons_True__connectome-APL_True':
            'prat-claws-boutons-APL'
        ,
    }
    assert df.model.isin(model_str2abbrev).all(), \
        f'{df.model[~df.model.isin(model_str2abbrev)].unique()=}'

    df.model = df.model.map(model_str2abbrev)

    # TODO sort odors, by glomerulus name for synthetic odors named like 'VA6-300'? and
    # maybe don't sort for diags (or alphabetical / existing sort order)?
    df.odor = df.odor.map(parse_odor_name)

    assert not df.isna().any().any()

    # TODO TODO are per-odor response rates / mean # spikes as variable for megamat
    # (tuned) as they are for private/etc? (store per-odor above, rather than just
    # getting a mean across all odors?)
    # TODO print this (/ use to set ylim for mean_num_spikes axes)
    # TODO delete one of these?
    mtdf = pd.DataFrame.from_dict(tuned_stat_means)
    mtdf.model = mtdf.model.map(model_str2abbrev)

    tdf = pd.concat(tuned_dfs, verify_integrity=True)
    tdf = tdf.reset_index()
    # TODO print tdf (/ use to set / check ylim below)
    tdf.model = tdf.model.map(model_str2abbrev)
    tdf.odor = tdf.odor.map(parse_odor_name)
    tdf = pd.melt(tdf, id_vars=['model', 'odor'], value_vars=stat_names,
        var_name='stat'
    )

    # TODO TODO TODO huh, so some models have private odors giving more total spikes?
    # TODO TODO is this all from a small number of cells spiking a lot, or what's the
    # nature of it?
    # ipdb> df.mean_num_spikes.max()
    # 0.819284064665127
    # ipdb> tdf.mean_num_spikes.max()
    # 0.5409930715935335
    # ipdb> df[['model', 'odor', 'mean_num_spikes']].sort_values('mean_num_spikes').tail()
    #                       model             odor  mean_num_spikes
    # 2                      wd20         VA1d-300         0.462254
    # 128  prat-claws-boutons-APL  VA6-150/DM2-150         0.534642
    # 85               prat-claws          DM1-300         0.580254
    # 125  prat-claws-boutons-APL          DM2-300         0.600462
    # 124  prat-claws-boutons-APL          DM1-300         0.819284

    tidy = pd.melt(df, id_vars=['panel', 'model', 'odor'], value_vars=stat_names,
        var_name='stat'
    )
    assert tidy['value'].size == df[stat_names].size
    assert not tidy.isna().any().any()
    df = tidy

    col_order = ['mean_num_spikes', 'mean_response_rate']
    assert set(col_order) == set(stat_names)
    assert len(set(stat_names)) == len(stat_names)


    def plot_panel_stats_across_models(df: pd.DataFrame, panel: str, suffix: str = ''
        ) -> None:
        g = sns.catplot(data=df, col='stat', col_order=col_order, sharey=False,
            hue='model', x='odor', y='value', kind='point', linestyle='none', alpha=0.3
        )
        # could probably just get col order from col_names, and not enforce with
        # col_order, once i've established well enough they match order of axes in
        # axes.flat
        assert g.col_names == col_order, f'{g.col_names=} != {col_order=}'
        for ax, n in zip(g.axes.flat, g.col_names):
            ymin = 0
            if n == 'mean_response_rate':
                ymax = response_rate_plot_max

            elif n == 'mean_num_spikes':
                # TODO drop the ~3 outliers over tdf.max() here (see comment above)?
                # we do have up to ~0.82 here...
                ymax = 1.0
            else:
                assert False, f'unexpected stat name {n=}'

            sser = df.loc[df.stat == n, 'value']
            assert sser.min() >= ymin, f'{sser.min()=} < {ymin=}'
            assert sser.max() <= ymax, f'{sser.max()=} > {ymax=}'

            ax.set_ylim([ymin, ymax])

            # TODO keep?
            ax.set_ylabel(n)

        g.set_titles('{col_name}')
        g.set_xticklabels(rotation=90)
        g.fig.suptitle(panel, y=1.04)

        # normalize_fname=False to not convert '__' -> '_'
        savefig(g, plot_root, f'{panel}{suffix}', normalize_fname=False)


    exclude_models = []
    # TODO delete. was just to generate different plot options for betty.
    #exclude_models = ['prat-claws-boutons-APL']
    #exclude_models = ['prat-claws-boutons-APL', 'prat-claws']
    #exclude_models = ['prat-claws-boutons-APL', 'wd20']
    if len(exclude_models) > 0:
        warn(f'dropping model variants {exclude_models}! hardcode exclude_models to '
            'empty, to plot all model variants'
        )
        # TODO assert each element of exclude_models is in these dataframes
        tdf = tdf[~tdf.model.isin(exclude_models)]
        df = df[~df.model.isin(exclude_models)]
        suffix = '__exclude_' + '_'.join(exclude_models)
    else:
        suffix = ''

    plot_panel_stats_across_models(tdf, 'megamat', suffix)

    for panel in panels:
        pdf = df[df.panel == panel]
        plot_panel_stats_across_models(pdf, panel, suffix)


if __name__ == '__main__':
    main()

