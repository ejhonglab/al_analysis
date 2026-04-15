#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat, pprint

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from hong2p.util import pd_allclose, addlevel
from hong2p.olf import parse_odor_name

from al_analysis import al_util
from al_analysis.al_util import warn, savefig, read_csv, read_parquet, to_csv, data_root
from al_analysis.mb_model import (megamat_orn_deltas, fit_and_plot_mb_model,
    megamat_orn_deltas, get_thr_and_APL_weights, format_model_params, abbrev_model_id,
    dict_seq_product, KC_TYPE
)


# TODO refactor to some standard list in mb_model? (-> share w/ model_yang_mixtures.py)
test_with_connectome_vs_uniform_apl = [
    dict(weight_divisor=20),
    dict(one_row_per_claw=True, prat_claws=True),
]
model_tune_kws = [
    # comparison for all other model cases, to see to what extent changes to PN>KC
    # weight matrix (and potentially other changes) matter
    dict(pn2kc_connections='uniform', n_claws=7),
] + dict_seq_product(test_with_connectome_vs_uniform_apl,
    [dict(), dict(use_connectome_APL_weights=True)]
)
# TODO delete? try again after finding more reasonable APL<>PN scales (that do more)
#+ [
#    dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
#        use_connectome_APL_weights=True
#    ),
#]

def check_local_megamat_spike_deltas_match(df: pd.DataFrame) -> None:
    # TODO factor out this checking (-> share w/ other scripts, if i want)
    # TODO TODO also fix it first to take df and actually check that input against below
    repo_root = Path('~/src/al_analysis').expanduser().resolve()
    pebbled_dir = repo_root / 'pebbled_6f/pdf/ijroi/mb_modeling'
    # TODO or find the most recent one?
    megamat_dir = (pebbled_dir / 'megamat' /
        'prat-claws_True__one-row-per-claw_True__connectome-APL_True__pn-claw-to-APL_'
        'True__prat-boutons_True__target-sp_0.1/'
    )
    if megamat_dir.exists():
        # TODO TODO TODO actually restore code in loop to check df or mdf against one of
        # below? currently below just checks against stuff it loads
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
            # TODO TODO TODO also actually check df.loc[:, 'megamat'] against same
            # subset of df2
        except KeyError:
            warn(f'could not check {most_recent_arbitrary_panels_deltas_csv} against '
                f'megamat specific deltas in {megamat_specific_deltas_csv}, probably '
                'because most recent al_analysis.py run (that produced model outputs, '
                'or fit dF/F -> spiking) did not include megamat data'
            )
    #


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--use-cache', action='store_true', help='uses cache for '
        'all model outputs, when available. this may mean no models are re-run.'
    )
    args = parser.parse_args()
    use_cache = args.use_cache

    plot_root = Path('george_model_outputs').resolve()
    plot_root.mkdir(exist_ok=True)

    # TODO avoid need to set this to see plots being saved
    al_util.verbose = True

    model_strs = [format_model_params(x) for x in model_tune_kws]
    model_str2abbrev = {m: abbrev_model_id(m) for m in model_strs}
    print('model ID -> abbrev:')
    pprint(model_str2abbrev)
    print()

    df = megamat_orn_deltas(drop_diags=False)

    # TODO restore?
    #check_local_megamat_spike_deltas_match(df)

    # indexing like this, since some of code using it below expects 'panel' column level
    # to be there still, but `df.loc[:, 'megamat']` would drop that level
    mdf = df.loc[:, df.columns.get_level_values('panel') == 'megamat']

    tune_df = mdf

    # TODO scale george's data (use diags to sanity check that scaling? other
    # overlapping odors?)
    #
    # there are also diags in diag_AL_panel_george_fly_trial.csv, if i wanted to use
    # those to calibrate scale against my data
    george_csv = data_root / 'from_george/AL_panel_george_fly_trial.csv'
    george_df = pd.read_csv(george_csv)

    george_df = george_df.set_index(['fly','trial','odor'], verify_integrity=True
        ).groupby('odor').mean()

    george_df.columns.name = 'glomerulus'

    cols_before = set(george_df.columns)
    george_df = george_df.dropna(how='all', axis='columns')
    dropped_gloms = cols_before - set(george_df.columns)
    warn(f'dropped {len(dropped_gloms)} glomeruli NaN for all flies:\n'
        f'{pformat(dropped_gloms)}'
    )
    # ipdb> s1 = set(df.index)
    # ipdb> s2 = set(george_df.index)
    # ipdb> s1 - s2
    # {'VA4', 'VL1'}
    # ipdb> s2 - s1
    # {'V'}
    # ipdb> len(s1 & s2)
    # 36

    assert not george_df.isna().any().any()
    assert george_df.index.equals(george_df.index.sort_values())
    assert george_df.columns.equals(george_df.columns.sort_values())
    george_panel = 'george-valence'
    george_df = addlevel(george_df, 'panel', george_panel).T.copy()

    # 808.7 (calculated as below) was seeming to take the max mean-#-spikes (across all
    # cells) too far outside the range I had been seeing before, so just gonna try a
    # slightly lower factor
    #scale_factor = df.max().max() / george_df.max().max()
    scale_factor = 600
    warn("scaling George's data so it has same max (across all odor x glom pairs) as my"
        ' mean est spike deltas (from megamat + diags) (by a factor of '
        f'{scale_factor:.1f})'
    )
    # TODO TODO probably scale negative part separately. his seems more intense than
    # mine (or try w/o doing so?)
    # TODO TODO implement
    #scale_inh_separately: bool = True
    scale_inh_separately: bool = False
    if not scale_inh_separately:
        # ipdb> george_df.droplevel('panel', axis='columns').agg([
        #   'min', 'mean', 'median', 'max']).T
        #                                min       mean     median         max
        # odor
        # 1,4 diaminobutane       -15.686692  -1.287167  -2.815787   23.134347
        # 1-hexanol               -21.980512  25.450344   8.853375  104.923907
        # 2,5-dimethylpyrazine    -34.004606  20.602278  17.756209   94.841622
        # 2-heptanone             -37.556967  64.662846  56.110527  195.014477
        # 3-methylthio-1-propanol -25.231738  21.516676  14.118277  101.768748
        # 3-octanol               -38.129037  60.935646  37.217463  228.288033
        # 4-methylcyclohexanol    -38.110991  18.815094  19.541303  107.436199
        # ACV 10%                 -52.252661  26.219046  23.531394  212.991984
        # CO2                     -81.802141 -13.987014 -22.936996   71.619080
        # acetic acid             -24.163128  25.112478  14.447277  167.351317
        # banana                  -53.295959  61.560176  56.686461  182.549227
        # ethanol 15%              -6.915643  16.093438  10.970025   67.894409
        # geosmin                 -21.274935  18.782088   9.328346  154.777360
        # geranyl acetate         -16.197290  28.764844  20.052027  178.402641
        # isoamyl acetate         -41.078022  43.152696  38.823003  190.081291
        # menthone                -28.239560  11.362024   7.228528  134.496508
        # methyl acetate          -39.837362  14.180682   5.219238  120.710805
        # methyl salicylate       -55.627517  16.418834  15.481360   72.786232
        # mint                    -38.292462  50.508233  41.911761  234.413498
        # p-cresol                -55.680928   9.913007  11.283514   83.689968
        # pfo                      -8.804892   2.451305   0.913936   17.681082
        # trans-2-hexenal         -53.259340  40.809197  43.194558  122.085339
        # water                   -12.298079   4.512435   2.832206   26.633905
        # wine                    -40.743134  21.582140   1.682331  118.164168
        # ipdb> mdf.agg(['min', 'mean', 'median', 'max']).T
        #                   min       mean     median         max
        # odor
        # 2h @ -3     -2.641954  52.671143  39.520532  159.201547
        # IaA @ -3     1.337206  46.563853  29.780338  188.007608
        # pa @ -3      1.205800  54.351983  30.039942  234.413498
        # 2-but @ -3  -6.431876  28.687480  13.940463  136.770253
        # eb @ -3      2.709834  49.290627  28.526569  199.173427
        # ep @ -3     -5.465991  41.168997  25.518597  166.692619
        # aa @ -3    -23.214124  21.424584  16.218057   78.246530
        # va @ -3    -15.742539  34.400630  20.846082  172.133416
        # B-cit @ -3   0.166586  16.496191   9.241352   63.777612
        # Lin @ -3   -22.335144  19.391615  11.559970  111.769464
        # 6al @ -3     1.716270  41.591401  30.148626  128.007319
        # t2h @ -3     0.218776  38.913864  29.225480  142.599943
        # 1-8ol @ -3  -2.815028  28.541136  17.994794  104.092717
        # 1-5ol @ -3   0.900240  41.769616  32.326551  145.836397
        # 1-6ol @ -3  -1.673497  56.186094  36.328795  205.769287
        # benz @ -3   -3.187387  38.767337  28.563463  147.175484
        # ms @ -3     -6.507243  19.872096  13.066179   79.290078
        george_df = george_df * scale_factor
        warn('scaling inhibition (negative values) same as positive values. look at '
            'summary statistics in comment to see why this might not make sense. mins '
            "are much larger in George's data after this scaling"
        )
    else:
        pass

    test_df = george_df
    panels = list(test_df.columns.get_level_values('panel').unique())

    # TODO need any lower, for all things not to fail? (would fail if any odor response
    # rates were higher than this) (nope, seems fine. delete comment)
    response_rate_plot_max = 0.2

    dfs = []
    dfs_with_kc_types = []
    tuned_dfs = []
    for kws in tqdm(model_tune_kws, unit='model'):
        print(f'{kws=}')

        tuned_params = fit_and_plot_mb_model(plot_root, orn_deltas=tune_df,
            try_cache=use_cache, response_rate_plot_max=response_rate_plot_max,
            max_iters=100, **kws
        )
        thr_and_apl_kws = get_thr_and_APL_weights(tuned_params, kws)
        # TODO only print if verbose? or say more, like that we tuned on megamat and
        # where those outputs are?
        # TODO TODO is it a mistake that i'm getting a vector for wAPLKC for some
        # of these? i.e. prat-claws_True. model at least working properly?
        # just a formatting mistake?
        print(f'tuned thr and APL weights: {pformat(thr_and_apl_kws)}')

        model_str = format_model_params(kws)

        tuned_model_output_dir = plot_root / tuned_params['output_dir']
        trs = read_parquet(tuned_model_output_dir / 'responses.parquet')
        tss = read_parquet(tuned_model_output_dir / 'spike_counts.parquet')

        mean_num_spikes = addlevel(tss.mean(), 'model', model_str)
        mean_num_spikes.name = 'mean_num_spikes'

        mean_response_rate = addlevel(trs.mean(), 'model', model_str)
        mean_response_rate.name = 'mean_response_rate'

        stats = pd.concat([mean_num_spikes, mean_response_rate], axis='columns',
            verify_integrity=True
        )
        tuned_dfs.append(stats)

        for panel in panels:
            panel_dir = plot_root / panel
            panel_dir.mkdir(exist_ok=True)
            panel_df = test_df.loc[:,test_df.columns.get_level_values('panel') == panel]

            params = fit_and_plot_mb_model(panel_dir, orn_deltas=panel_df,
                try_cache=use_cache, response_rate_plot_max=response_rate_plot_max,
                **kws, **thr_and_apl_kws
            )
            model_output_dir = panel_dir / params['output_dir']
            rs = pd.read_pickle(model_output_dir / 'responses.p')
            ss = pd.read_pickle(model_output_dir / 'spike_counts.p')

            def add_metadata(series):
                series = addlevel(series, 'model', model_str)
                return addlevel(series, 'panel', panel)

            mean_num_spikes = add_metadata(ss.mean())
            mean_num_spikes.name = 'mean_num_spikes'

            mean_response_rate = add_metadata(rs.mean())
            mean_response_rate.name = 'mean_response_rate'

            stats = pd.concat([mean_num_spikes, mean_response_rate], axis='columns',
                verify_integrity=True
            )

            if kws.get('pn2kc_connections') != 'uniform':
                assert rs.index.equals(ss.index)
                assert KC_TYPE in rs.index.names
                kc_types = rs.index.get_level_values(KC_TYPE)
                assert not kc_types.isna().any()

                # TODO assert this is the same for all? it's not tho, right? what's the
                # diffs?
                # TODO for some reason we have 1828 cells instead of 1732. change that?
                # should match!) (oh, ig it matches wd20?)
                type_counts = kc_types.value_counts()
                if 'unknown' in type_counts:
                    # TODO should i drop these from all analyses, not just this?
                    warn(f'{abbrev_model_id(model_str)}: dropping '
                        f'{type_counts["unknown"]}/{len(rs)} KCs with type="unknown"'
                    )
                    rs = rs.drop(level=KC_TYPE, labels='unknown')
                    ss = ss.drop(level=KC_TYPE, labels='unknown')

                if panel == panels[-1]:
                    print(f'\n{model_str} KC subtype counts:\n'
                        f'{type_counts.to_string()}'
                    )

                mean_num_spikes_by_type = add_metadata(
                    ss.groupby(KC_TYPE).mean().stack()
                )
                mean_num_spikes_by_type.name = 'mean_num_spikes'

                mean_response_rate_by_type = add_metadata(
                    rs.groupby(KC_TYPE).mean().stack()
                )
                mean_response_rate_by_type.name = 'mean_response_rate'

                stats_by_type = pd.concat(
                    [mean_num_spikes_by_type, mean_response_rate_by_type],
                    axis='columns', verify_integrity=True
                )
                dfs_with_kc_types.append(stats_by_type)

            dfs.append(stats)

        print()

    df = pd.concat(dfs, verify_integrity=True)
    stat_names = list(df.columns)

    def postprocess_and_tidy_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index()
        df.model = df.model.apply(abbrev_model_id)

        odor_has_conc_delim = df.odor.str.contains(' @ ')
        # so it will be skipped on george's data
        if odor_has_conc_delim.any():
            assert odor_has_conc_delim.all()
            df.odor = df.odor.map(parse_odor_name)

        assert not df.isna().any().any()

        id_vars = ['model', 'odor']
        if 'panel' in df.columns:
            id_vars = ['panel'] + id_vars

        if KC_TYPE in df.columns:
            id_vars.append(KC_TYPE)

        tidy = pd.melt(df, id_vars=id_vars, value_vars=stat_names, var_name='stat')
        assert tidy['value'].size == df[stat_names].size
        assert not tidy.isna().any().any()

        return tidy


    df = postprocess_and_tidy_df(df)

    df_with_kc_types = pd.concat(dfs_with_kc_types, verify_integrity=True)
    df_with_kc_types = postprocess_and_tidy_df(df_with_kc_types)

    # assuming only one panel ('megamat'), so it's ok this doesn't have a panel column
    tdf = pd.concat(tuned_dfs, verify_integrity=True)
    tdf = postprocess_and_tidy_df(tdf)

    col_order = ['mean_num_spikes', 'mean_response_rate']
    assert set(col_order) == set(stat_names)
    assert len(set(stat_names)) == len(stat_names)

    george12_order = [
        'CO2',
        'p-cresol',
        'menthone',
        'isoamyl acetate',
        'mint',
        '1-hexanol',
        '2-heptanone',
        'geranyl acetate',
        '4-methylcyclohexanol',
        'banana',
        'wine',
        'ACV 10%',
    ]
    george24_order = [
        'CO2',
        'p-cresol',
        'menthone',
        'isoamyl acetate',
        '1,4 diaminobutane',
        'mint',
        '3-octanol',
        '1-hexanol',
        'pfo',
        'methyl salicylate',
        '2-heptanone',
        '2,5-dimethylpyrazine',
        'water',
        'geosmin',
        'methyl acetate',
        'trans-2-hexenal',
        'geranyl acetate',
        '3-methylthio-1-propanol',
        'acetic acid',
        '4-methylcyclohexanol',
        'ethanol 15%',
        'banana',
        'wine',
        'ACV 10%',
    ]
    s1 = set(df.odor)
    s2 = set(george12_order)
    s3 = set(george24_order)
    assert s2 - s3 == set()
    assert s1 == s3

    g2 = [x for x in george24_order if x in george12_order]
    # ok, so we can just sort by index in george24_order
    assert g2 == george12_order

    def odor_sort_fn(ser):
        return ser.map(george24_order.index)

    def sort_odors(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by='odor', kind='stable', key=odor_sort_fn)

    df = sort_odors(df)
    df_with_kc_types = sort_odors(df_with_kc_types)

    def plot_panel_stats_across_models(df: pd.DataFrame, panel: str, suffix: str = '',
        title_suffix: str = '', mean_num_spikes_max: float = 1.5,
        mean_response_rate_max: float = response_rate_plot_max) -> None:

        df = df.copy()
        df['connectome_apl'] = df.model.str.contains('_connectome-APL')
        df['model'] = df.model.str.replace('_connectome-APL', '')

        unique_model_ids = df.model.unique()
        palette = dict(zip(
            unique_model_ids, sns.color_palette(n_colors=len(unique_model_ids))
        ))
        # this is only taken a param of pointplot (which catplot should pass it to),
        # but pointplot only defines it per hue level, so would prob need to either
        # duplicate colors (and have same len list of colors and markers) or two calls?
        #
        uniform_apl_marker = '.'
        connectome_apl_marker = '+'

        marker_fill_kws = dict(
            # default markeredgewidth seems ~2
            linestyle='none', fillstyle='none', markeredgewidth=1.0,
            markersize=8.0
        )
        def plot_fn(data, *cols, **kwargs):
            for connectome_apl in (True, False):
                # doing it this way, rather than initial groupby, so that point markers
                # get plotted last, so all i have to do is add a separate colorless '+'
                # marker to legend to handle that (was mix of '.' and '+' in legend,
                # since uniform doesn't have connectome APL case.
                gdf = data[data.connectome_apl == connectome_apl]
                marker = (
                    uniform_apl_marker if not connectome_apl else connectome_apl_marker
                )
                # TODO assert order of odors in gdf?
                sns.pointplot(gdf, *cols, hue='model', palette=palette, dodge=True,
                    marker=marker, alpha=0.5, **marker_fill_kws, **kwargs
                )

        facet_kws = dict(height=4.5, aspect=1.5)
        if KC_TYPE in df.columns:
            facet_kws['row'] = KC_TYPE
            # i think height is per row, so we want smaller actually
            facet_kws['height'] = 2.5
            # TODO bigger aspect (wider) for 24 odor panel?
            facet_kws['aspect'] = 1.5

        g = sns.FacetGrid(data=df, col='stat', col_order=col_order, sharey=False,
            **facet_kws
        )
        g.map_dataframe(plot_fn, x='odor', y='value')

        legend_data = dict(g._legend_data)
        artist_kws = dict(color='k', alpha=0.5, **marker_fill_kws)
        legend_data['uniform-APL'] = Line2D([0],[0], marker=uniform_apl_marker,
            **artist_kws
        )
        legend_data['connectome-APL'] = Line2D([0],[0], marker=connectome_apl_marker,
            **artist_kws
        )
        label_order = list(unique_model_ids) + ['uniform-APL', 'connectome-APL']
        g.add_legend(legend_data=legend_data, label_order=label_order,
            title='model variant'
        )
        assert g.col_names == col_order, f'{g.col_names=} != {col_order=}'
        for ti, row_axes in enumerate(g.axes):
            if KC_TYPE in df.columns:
                curr_type = g.row_names[ti]

            row_axes = row_axes.squeeze()
            for ax, n in zip(row_axes, g.col_names):
                ymin = 0
                if n == 'mean_response_rate':
                    ymax = mean_response_rate_max

                elif n == 'mean_num_spikes':
                    # was `ymax = 1.0` for other scripts (Yang's at least, and prob only
                    # needed this high for megamat odors? not sure).
                    #
                    # with original calculated (max0 / max1) scale factor (=800.7) was
                    # getting 1.236 on some of his odors here. also got 0.24942.  some
                    # others may have even been more. not sure. decreased scale factor
                    # to 600 after that.
                    ymax = mean_num_spikes_max
                else:
                    assert False, f'unexpected stat name {n=}'

                sser = df.loc[df.stat == n, 'value']
                assert sser.min() >= ymin, f'{sser.min()=} < {ymin=}'
                assert sser.max() <= ymax, f'{sser.max()=} > {ymax=}'

                ax.set_ylim([ymin, ymax])
                if KC_TYPE in df.columns:
                    ax.set_ylabel(f'KC subtype: {curr_type}')
                else:
                    ax.set_ylabel(n)

        g.set_titles('{col_name}')

        if KC_TYPE in df.columns:
            # TODO assert it's actually the order? and that it matches some property of
            # the sns object?
            g.set_xticklabels(list(df.odor.unique()), rotation=90)
        else:
            g.set_xticklabels(rotation=90)

        g.fig.suptitle(f'{panel}{title_suffix}', y=1.04)

        # normalize_fname=False to not convert '__' -> '_'
        savefig(g, plot_root, f'{panel}{suffix}', normalize_fname=False)


    csvname_prefix = 'george_model_mean_response_stats'
    to_csv(df, plot_root / f'{csvname_prefix}.csv')
    to_csv(df_with_kc_types, plot_root / f'{csvname_prefix}_by-type.csv')

    plot_panel_stats_across_models(tdf, 'megamat')
    for desc, odors in [('g12', george12_order), ('g24', george24_order)]:
        suffix = f'_{desc}'

        sdf = df[df.odor.isin(odors)]
        plot_panel_stats_across_models(sdf, george_panel, suffix)

        suffix = f'{suffix}_by-type'
        sdf_by_type = df_with_kc_types[df_with_kc_types.odor.isin(odors)]
        plot_panel_stats_across_models(sdf_by_type, george_panel, suffix,
            title_suffix=f'\n{len(odors)} odor panel',
            # hit at least 1.5844 here (for # spikes), and at least .256 (for response
            # rate)
            mean_num_spikes_max=1.75, mean_response_rate_max=0.3
        )


if __name__ == '__main__':
    main()

