#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat, pprint
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from hong2p.util import pd_allclose, addlevel
from hong2p.olf import parse_odor_name

import al_util
from al_util import savefig, plot_responses, read_parquet, to_csv, to_parquet
from mb_model import (megamat_orn_deltas, fit_and_plot_mb_model, megamat_orn_deltas,
    natmix_orn_deltas, get_thr_and_APL_weights, format_model_params,
    get_odor_fname_suffix, KC_ID
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
    parser = ArgumentParser()
    parser.add_argument('-c', '--use-cache', action='store_true', help='uses cache for '
        'all model outputs, when available. this may mean no models are re-run.'
    )
    args = parser.parse_args()
    use_cache = args.use_cache

    plot_root = Path('yang_mix_outputs').resolve()
    plot_root.mkdir(exist_ok=True)

    al_util.verbose = True

    df = megamat_orn_deltas(drop_diags=False)

    # no other panels in output of fn above
    #
    # indexing differently for `mdf`, since some of code using it below expects 'panel'
    # column level to be there still, but `df.loc[:, 'megamat']` would drop that level
    mdf = df.loc[:, df.columns.get_level_values('panel') == 'megamat']
    diags = df.loc[:, df.columns.get_level_values('panel') == 'glomeruli_diagnostics']

    natmix_df = natmix_orn_deltas()

    tune_df = mdf

    series_list = []
    # make binary mixtures of synthetic diagnostics (150Hz spike delta to each of the
    # two glomeruli, for all combinations of them)
    gloms_to_mix = ['DM4', 'VM5d', 'DC3']
    for x in gloms_to_mix:
        ser = pd.Series(index=df.index.copy(), name=f'{x}-300 @ 0', data=0.0)
        ser.loc[x] = 300.0
        ser.name = ('syn-diag-binaries', ser.name)
        series_list.append(ser)

    glom_combos = list(combinations(gloms_to_mix, 2))
    for x, y in glom_combos:
        mser = pd.Series(index=df.index.copy(), name=f'{x}-150/{y}-150 @ 0', data=0.0)
        mser.loc[x] = 150.0
        mser.loc[y] = 150.0
        mser.name = ('syn-diag-binaries', mser.name)
        series_list.append(mser)

    # make binary mixtures by combining the real diagnostic data for each of these
    # three odors (should target same glomeruli as above). again all pairwise combos.
    diag_subset = diags.loc[:, diags.columns.get_level_values('odor').isin(
        ('2h @ -6', 'farn @ -2', 'ma @ -7')
    )]
    odor2glom = {
        '2h @ -6': 'VM5d',
        'farn @ -2': 'DC3',
        'ma @ -7': 'DM4',
    }
    assert diag_subset.index.equals(df.index)

    new_panels = ('diag-binaries_mean', 'diag-binaries_max', 'diag-binaries_max-rest0')
    for panel in new_panels:
        # replacing 'glomeruli_diagnostics' panel w/ each of these new ones, so
        # single components will appear in all panel specific plots
        comp_df = addlevel(diag_subset.droplevel('panel', axis='columns'), 'panel',
            panel, axis='columns'
        )
        # technically not just a list of series anymore... but should still work
        series_list.append(comp_df)

    for x, y in combinations(diag_subset.columns.get_level_values('odor'), 2):
        # hack so some of the string processing code inside modelling doesn't err
        # ideally we'd just have mix_name=f'{x} + {y}'
        mix_name = f'{x.replace(" @ ", "")} + {y.replace(" @ ", "")} @ 0'
        d1 = diag_subset.loc[:, (slice(None), x)].squeeze()
        d2 = diag_subset.loc[:, (slice(None), y)].squeeze()
        both = pd.concat([d1, d2], axis='columns', verify_integrity=True)
        both.columns.names = ['panel', 'odor']

        mean_ser = both.mean(axis='columns')
        max_ser = both.max(axis='columns')

        g1 = odor2glom[x]
        g2 = odor2glom[y]
        max_zerod_ser = max_ser.copy()
        max_zerod_ser[~max_zerod_ser.index.isin((g1, g2))] = 0

        mean_ser.name = 'mean'
        max_ser.name = 'max'
        max_zerod_ser.name = 'max, non-cognate gloms 0d'
        to_plot = pd.concat([
                both.droplevel('panel', axis='columns'),
                mean_ser, max_ser, max_zerod_ser
            ], axis='columns', verify_integrity=True
        )
        mix_suffix = get_odor_fname_suffix(f'{x}_and_{y}')
        plot_responses(to_plot, plot_root, f'diags_vs_constructed-mixtures{mix_suffix}',
            cbar_label='est. ORN firing rate delta (Hz)'
        )

        # first element of each of these should be in `new_panels` defined before this
        # loop (otherwise single components will not be added correctly)
        mean_ser.name = ('diag-binaries_mean', mix_name)
        series_list.append(mean_ser)
        max_ser.name = ('diag-binaries_max', mix_name)
        series_list.append(max_ser)
        max_zerod_ser.name = ('diag-binaries_max-rest0', mix_name)
        series_list.append(max_zerod_ser)


    test_df = pd.concat(series_list, axis='columns', verify_integrity=True)
    test_df.columns.names = ['panel', 'odor']
    assert not test_df.isna().any().any()

    # TODO want to do anything about his other than fillna(0)? prob not
    # ipdb> natmix_df.index.difference(test_df.index)
    # Index(['DA1', 'DA4l', 'DA4m', 'V', 'VA1d', 'VA1v'], ...
    # ipdb> test_df.index.difference(natmix_df.index)
    # Index(['VA4'], dtype='object', name='glomerulus')
    test_df = pd.concat([test_df, natmix_df], axis='columns', verify_integrity=True)
    test_df = test_df.fillna(0.0)

    test_df = test_df.sort_index().sort_index(axis='columns')
    del df

    panels = list(test_df.columns.get_level_values('panel').unique())

    # saw 0.212 on some kiwi/control stuff (tuned on megamat)
    response_rate_plot_max = 0.22

    dfs = []
    tuned_dfs = []
    for kws in tqdm(model_tune_kws, unit='model (on all panels)'):
        print(f'{kws=}')

        tuned_params = fit_and_plot_mb_model(plot_root, orn_deltas=tune_df,
            try_cache=use_cache, response_rate_plot_max=response_rate_plot_max, **kws
        )
        thr_and_apl_kws = get_thr_and_APL_weights(tuned_params, kws)
        print(f'tuned thr and APL weights: {pformat(thr_and_apl_kws)}')

        model_str = format_model_params(kws)

        tuned_model_output_dir = plot_root / tuned_params['output_dir']
        trs = read_parquet(tuned_model_output_dir / 'responses.parquet')
        tss = read_parquet(tuned_model_output_dir / 'spike_counts.parquet')

        wPNKC = read_parquet(tuned_model_output_dir / 'wPNKC.parquet')
        raw_wPNKC = wPNKC.copy()
        if kws.get('one_row_per_claw', False):
            wPNKC = wPNKC.groupby(KC_ID).sum()

            if kws.get('prat_boutons', False):
                wPNKC = wPNKC.droplevel(
                    [x for x in wPNKC.columns.names if x != 'glomerulus'],
                    axis='columns'
                )
                wPNKC = wPNKC.groupby('glomerulus', axis='columns').sum()

        kc_glom_combo_counts = wPNKC[gloms_to_mix].value_counts().sort_index()
        kc_glom_combo_counts.name = 'n_kcs'
        to_csv(kc_glom_combo_counts,
            plot_root / f'kc-glom-combo-counts_{model_str}.csv'
        )
        to_parquet(kc_glom_combo_counts,
            plot_root / f'kc-glom-combo-counts_{model_str}.parquet'
        )
        # TODO TODO plot binarized version, just counting # KCs getting any amount of
        # input from each combo? or some kind of dist of total amount of input per
        # combo? (separate line for those that only get input from one?)
        # TODO best way to plot this? for uniform model, can get value_counts like:
        # ipdb> wPNKC[gloms_to_mix].value_counts().sort_index()
        # DM4  VM5d  DC3
        # 0.0  0.0   0.0    1223
        #            1.0     178
        #            2.0       9
        #            3.0       1
        #      1.0   0.0     153
        #            1.0      15
        #            2.0       3
        #      2.0   0.0       7
        # 1.0  0.0   0.0     186
        #            1.0      17
        #      1.0   0.0      17
        #            1.0       1
        #      2.0   0.0       2
        # 2.0  0.0   0.0      14
        #            1.0       1
        #      1.0   0.0       1
        # dtype: int64

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
            rs = read_parquet(model_output_dir / 'responses.parquet')
            ss = read_parquet(model_output_dir / 'spike_counts.parquet')

            wPNKC2 = read_parquet(model_output_dir / 'wPNKC.parquet')
            assert raw_wPNKC.equals(wPNKC2), 'wPNKC should not change across tuned/not'

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
    df.odor = df.odor.map(parse_odor_name)
    assert not df.isna().any().any()
    tidy = pd.melt(df, id_vars=['panel', 'model', 'odor'], value_vars=stat_names,
        var_name='stat'
    )
    assert tidy['value'].size == df[stat_names].size
    assert not tidy.isna().any().any()
    df = tidy

    # assuming only one panel ('megamat'), so it's ok this doesn't have a panel column
    tdf = pd.concat(tuned_dfs, verify_integrity=True)
    tdf = tdf.reset_index()
    # TODO print tdf (/ use to set / check ylim below)
    tdf.model = tdf.model.map(model_str2abbrev)
    tdf.odor = tdf.odor.map(parse_odor_name)
    tdf = pd.melt(tdf, id_vars=['model', 'odor'], value_vars=stat_names,
        var_name='stat'
    )

    col_order = ['mean_num_spikes', 'mean_response_rate']
    assert set(col_order) == set(stat_names)
    assert len(set(stat_names)) == len(stat_names)

    def odor_sort_fn(x):
        v1 = 1 * (x.str.contains('+', regex=False)) | (x.str.contains('/', regex=False))
        # to put the cmix0/kmix 0 at end
        v2 = 2 * x.str.contains('mix0', regex=False)
        return v1 + v2

    df = df[~(df.odor.str.contains('mix-') | df.odor.str.contains('(air mix)'))].copy()

    df = df.sort_values(by='odor', kind='stable', key=odor_sort_fn)

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
            ax.set_ylabel(n)

        g.set_titles('{col_name}')
        g.set_xticklabels(rotation=90)
        g.fig.suptitle(panel, y=1.04)

        # normalize_fname=False to not convert '__' -> '_'
        savefig(g, plot_root, f'{panel}{suffix}', normalize_fname=False)

    suffix = ''
    plot_panel_stats_across_models(tdf, 'megamat', suffix)

    comps_to_drop = [
        'fur', 'ms', 'va', 'EtOH', 'IAol', 'IaA'
    ]
    for panel in panels:
        pdf = df[df.panel == panel]
        plot_panel_stats_across_models(pdf, panel, suffix)

        if panel in ('kiwi', 'control'):
            pdf_nocomps = pdf[~pdf.odor.isin(comps_to_drop)]
            plot_panel_stats_across_models(pdf_nocomps, panel, f'{suffix}_nocomps')


if __name__ == '__main__':
    main()

