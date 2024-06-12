#!/usr/bin/env python3

import argparse
from os.path import getmtime
from pathlib import Path
from pprint import pformat, pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hong2p.types import Pathlike

from al_analysis import (format_mtime, roi_plot_kws, roimean_plot_kws,
    plot_all_roi_mean_responses, plot_n_per_odor_and_glom
)



fly_cols = ['date', 'fly_num']
col_levels = fly_cols + ['roi']
index_levels = ['panel', 'is_pair', 'odor1', 'odor2', 'repeat']


# TODO provide fn to invert zero filling i had done for some new outputs (dropping
# glomeruli w/ all 0s or NaN)

def drop_old_odor_index_levels(df: pd.DataFrame) -> pd.DataFrame:
    # we can drop these. metadata intended for binary mixture experiments
    # (not what we are dealing with here)
    assert set(df.index.get_level_values('is_pair')) == {False}
    assert set(df.index.get_level_values('odor2')) == {'solvent'}
    df.index = df.index.droplevel(['is_pair', 'odor2'])
    return df


def read_csv(csv: Pathlike, *, drop_old_odor_levels: bool = True,
    check_vs_pickle: bool = True, verbose: bool = True) -> pd.DataFrame:

    csv = Path(csv)
    assert csv.exists()

    if verbose:
        print(f'loading {csv}')
        print(f'modified {format_mtime(getmtime(csv), year=True)}')
        print()

    # TODO specify by name instead? possible? might be easier to also optionally
    # support 'fly_id' instead/plus ['date', 'fly_num'] levels
    df = pd.read_csv(csv,
        # index level names: ['panel', 'is_pair', 'odor1', 'odor2', 'repeat']
        #
        # can't pass list of names here when specifying column MultiIndex via header.
        # trying raises ValueError to that effect.
        index_col=list(range(5)),

        # column level names: ['date', 'fly_num', 'roi']
        #
        # doesn't take list of str, and names= seems to only set column level names,
        # rather than finding them by name.
        header=list(range(3))
    )

    assert df.columns.names == col_levels
    assert df.index.names == index_levels

    assert df.columns.names[0] == fly_cols[0]
    df.columns = df.columns.set_levels(pd.to_datetime(df.columns.levels[0]),
        level=0, verify_integrity=True
    )

    assert df.columns.names[1] == fly_cols[1]
    df.columns = df.columns.set_levels(df.columns.levels[1].astype(int), level=1,
        verify_integrity=True
    )

    if check_vs_pickle:
        # just some checking i was doing against a parallel pickle version i had, mainly
        # to make sure i was loading CSV correctly (with same dtype info)
        pickle_path = csv.with_suffix('.p')
        if pickle_path.exists():
            pdf = pd.read_pickle(pickle_path)

            assert df.index.equals(pdf.index)
            assert df.columns.equals(pdf.columns)

            # TODO factor this isna + isclose checking to hong2p.util fn?
            # (maybe w/ col/index check above too?)

            isna = df.isna()
            assert isna.equals(pdf.isna())

            isclose = np.isclose(df, pdf)
            assert np.logical_xor(isna, isclose).all().all()

            if verbose:
                print(f'CSV data matches pickle {pickle_path}\n')
        else:
            if verbose:
                print(f'no pickle at {pickle_path}. could not check against CSV.\n')

    if drop_old_odor_levels:
        df = drop_old_odor_index_levels(df)

    return df


def summarize_antennal_data(df: pd.DataFrame, verbose: bool = True) -> None:
    # TODO also fly_id if present?
    unique_flies = df.columns.to_frame(index=False)[fly_cols].drop_duplicates()

    n_flies = len(unique_flies)
    print(f'{n_flies=}')
    print(unique_flies.to_string(index=False))
    print()

    # a glomerulus is one of ~50 named regions in the antennal lobe. each receives
    # input from all olfactory receptor neurons (ORNs) expressing a corresponding
    # type of receptor. in each glomerulus, all ORNs of one type synapse onto all
    # projection neurons (PNs) of a corresponding type. the PNs then provide input
    # to the Kenyon cells (KCs).
    #
    # the Hallem and Carlson (2006) data you might be familiar with measured signals
    # from ~half of these receptor types, with a different type of measurement.
    unique_glomeruli = sorted(set(df.columns.get_level_values('roi')))
    print(f'unique glomeruli ({len(unique_glomeruli)})', end='')
    if verbose:
        print(f':\n{pformat(unique_glomeruli)}')
    print()

    n_glom_per_fly = df.groupby(fly_cols, axis='columns').apply(
        lambda x: x.columns.nunique('roi')
    )
    avg_n_glom = n_glom_per_fly.mean()
    print(f'number of glomeruli per fly (avg: {avg_n_glom:.1f}):\n'
        f'{n_glom_per_fly.to_string()}'
    )
    print()

    # TODO print number of repeats (if we have)
    n_repeats_per_odor = df.index.to_frame(index=False).groupby(['panel', 'odor1']
        ).nunique('repeat')

    assert n_repeats_per_odor.shape[1] == 1
    n_repeats_per_odor = n_repeats_per_odor.iloc[:, 0]
    max_repeats = n_repeats_per_odor.max()
    assert (n_repeats_per_odor == max_repeats).all()
    print(f'number of trials ("repeats", 0-indexed) per odor: {max_repeats}')

    print('number of unique odors (diff concs counted separately) for each panel:')
    print(df.groupby('panel').apply(lambda x: len(x.index.unique('odor1'))
        ).to_string(header=False)
    )
    if verbose:
        print()
        for panel, pdf in df.groupby('panel'):
            print(f'{panel=} odors:')
            # TODO odor sorting from al_analysis instead? no sorting?
            #pprint(sorted(pdf.index.unique('odor1')))
            pprint(list(pdf.index.unique('odor1')))
            print()

    print()


sep_line = '#' * 88

def summarize_old_panel_csvs(*, verbose=True):
    csv_dir = Path('pebbled_6f')

    # TODO TODO TODO why only 6 flies in this validation2 panel csv? check what i gave
    # remy / anoop! (prob just old csv. still...)
    for i, panel in enumerate(('megamat', 'validation2')):
        if i != 0:
            print(sep_line)

        # TODO TODO TODO update to now just load ij_certain-roi_stats.csv (and maybe
        # also other csvs i'm now saving). NO LONGER saving these csvs w/ panel prefix
        # in al_analysis.py!
        csv_name = f'{panel}_ij_certain-roi_stats.csv'
        csv = csv_dir / csv_name
        print(f'{panel=}')

        df = read_csv(csv)

        unique_panels = set(df.index.get_level_values('panel'))
        assert unique_panels == {'glomeruli_diagnostics', panel}

        summarize_antennal_data(df, verbose=verbose)

        # example code:
        '''
        assert df.index.names[0] == 'panel'
        # dropping 'glomeruli_diagnostic' panel, as you probably don't want to analyze
        # that. it's just a series of narrowly activating odors intended to help me
        # identify particular glomeruli
        df = df.loc[panel].copy()

        # averaging over the 3 trials for each odor
        mean_df = df.groupby('odor1', sort=False).mean()
        '''


def main():
    parser = argparse.ArgumentParser()
    # TODO include message about what happens by default if not passed
    # (though i might change what that is now...)
    parser.add_argument('csv', nargs='?', type=Path, help='path to CSV to summarize')
    parser.add_argument('-q', '--quiet', action='store_true', help='prints less info')
    parser.add_argument('-p', '--plot', action='store_true', help='plots individual '
        'fly-ROI and mean-ROI matrices and saves alongside input CSV'
    )
    args = parser.parse_args()
    csv = args.csv
    plot = args.plot
    verbose = not args.quiet

    if csv is not None:
        df = read_csv(csv)
        summarize_antennal_data(df, verbose=verbose)

        if plot:
            plot_fmt = 'pdf'
            def savefig(fig, suffix='', **kwargs):
                fig_name = f'{csv.stem}{suffix}.{plot_fmt}'
                fig_path = csv.parent / fig_name
                print(f'saving figure {fig_path}')
                fig.savefig(fig_path, bbox_inches='tight', **kwargs)

            trialmean_df = df.groupby(level=['panel', 'odor1'], sort=False).mean()

            # TODO also work back in al_analysis? why the diff?
            roi_plot_kws['vgroup_label_offset'] = 0.03
            roi_plot_kws['hgroup_label_offset'] = 0.12

            fig, _ = plot_all_roi_mean_responses(trialmean_df, **roi_plot_kws)
            savefig(fig)

            mean_df = trialmean_df.groupby(level='roi', sort=False, axis='columns'
                ).mean()
            vmin = mean_df.min().min()
            vmax = mean_df.max().max()

            fig, _ = plot_all_roi_mean_responses(mean_df, vmin=vmin, vmax=vmax,
                **roimean_plot_kws
            )
            savefig(fig,  '_mean')

            stddev_df = trialmean_df.groupby(level='roi', sort=False, axis='columns'
                ).std()
            fig, _ = plot_all_roi_mean_responses(stddev_df, vmin=0, vmax=vmax,
                **roimean_plot_kws
            )
            savefig(fig,  '_stddev')

            fig, _ = plot_n_per_odor_and_glom(trialmean_df, title=False)
            savefig(fig, '_n-per-odor-and-glom') #, bbox_inches='tight')
    else:
        # TODO support plot here too? prob not...
        assert not plot, 'plot only supported if CSV path provided'
        summarize_old_panel_csvs(verbose=verbose)


if __name__ == '__main__':
    main()

