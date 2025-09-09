#!/usr/bin/env python3

import argparse
from os.path import getmtime
from pathlib import Path
from pprint import pformat, pprint
from typing import Dict, Any

import pandas as pd
import numpy as np

from hong2p.olf import add_mix_str_index_level, solvent_str, mix_col
from hong2p.roi import certain_roi_indices
from hong2p.types import Pathlike

from al_util import format_mtime, warn, sort_odors
from al_analysis import (roi_plot_kws, roimean_plot_kws, plot_all_roi_mean_responses,
    plot_n_per_odor_and_glom, get_gsheet_metadata
)


fly_cols = ['date', 'fly_num']
col_levels = fly_cols + ['roi']

# TODO maybe don't require is_pair? panel?
#
# can have more 'odor<x>' row index levels if there are multiple odors presented at once
# (via air mixing). will have a number of levels equal to maximum presented at once.
required_index_levels = ['panel', 'is_pair', 'odor1', 'repeat']

# TODO provide fn to invert zero filling i had done for some new outputs (dropping
# glomeruli w/ all 0s or NaN)

def drop_old_odor_index_levels(df: pd.DataFrame) -> pd.DataFrame:
    # for dropping metadata intended for binary mixture experiments
    to_drop = []

    if 'is_pair' in df.index.names:
        if set(df.index.get_level_values('is_pair')) == {False}:
            to_drop.append('is_pair')
        else:
            warn('index had some is_pair=True entries! not dropping is_pair level!')

    if 'odor2' in df.index.names:
        if set(df.index.get_level_values('odor2')) == {solvent_str}:
            to_drop.append('odor2')
        else:
            warn(f'index had some odor2 != {solvent_str} entries! not dropping odor2 '
                'level!'
            )

    if len(to_drop) > 0:
        df.index = df.index.droplevel(to_drop)

    return df


def read_csv(csv: Pathlike, *, drop_old_odor_levels: bool = True,
    check_vs_pickle: bool = True, verbose: bool = True) -> pd.DataFrame:
    # TODO doc output format (w/ example str repr)

    csv = Path(csv)
    assert csv.exists(), f'CSV {csv} did not exist!'

    if verbose:
        print(f'loading {csv}')
        print(f'modified {format_mtime(getmtime(csv), year=True)}')
        print()

    # this line will start with the level names for the row index, e.g.
    # ['panel', 'is_pair', 'odor1', 'repeat', nan, ...]
    # (for data with a maximum of one odor component mixed-in-air)
    #
    # if there are additional components, there will be additional 'odor<N>' index
    # levels, up to the maximum # of components presented at once (via air mixing. odors
    # mixed in vial are represented by their own unique name, not via separate levels).
    # most I've ever used (via air mixing) is 2.
    #
    # not sure if there might be data after these level names, or if it will always be
    # NaN for everything else in that row (but doesn't matter for this).
    index_row = pd.read_csv(csv, skiprows=3, nrows=1, header=None).squeeze()
    # 'repeat' should always be the last level, so if we find that, we know we only have
    # to read index_col up to there (in next read_csv call)
    eq_repeat = index_row == 'repeat'
    assert eq_repeat.sum() == 1
    # index starts at 0, so we will need to read 1 more index_col level past this
    repeat_idx = eq_repeat.idxmax()
    n_index_col_levels = repeat_idx + 1

    df = pd.read_csv(csv,
        # can't pass list of names here when specifying column MultiIndex via header.
        # trying raises ValueError to that effect.
        index_col=list(range(n_index_col_levels)),

        # doesn't take list of str, and names= seems to only set column level names,
        # rather than finding them by name.
        header=list(range(len(col_levels)))
    )
    # this is assumed in section above that determines how many index levels there are
    assert df.index.names[-1] == 'repeat'

    assert df.columns.names == col_levels
    assert set(required_index_levels) - set(df.index.names) == set()

    # the only row index levels not in required_index_levels should be extra air-mix
    # components (with names in 'odor<N>' form)
    prefix = 'odor'
    for x in set(df.index.names) - set(required_index_levels):
        assert x.startswith(prefix)
        try:
            component_num = int(x[len(prefix):])
        # only intending to catch int parsing failure like:
        # ValueError: invalid literal for int() with base 10: ...
        except ValueError:
            # TODO include better message about malformed index level name
            raise

        # TODO assert all contiguous?
        assert component_num >= 1

    # TODO refactor?
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


def get_unique_flies(df: pd.DataFrame) -> pd.DataFrame:
    # TODO also fly_id if present?
    return df.columns.to_frame(index=False)[fly_cols].drop_duplicates()


def summarize_antennal_data(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Prints (if `verbose=True`):
    - # of unique glomeruli, and a list of their names

    - # of glomeruli per fly (and on average)

    - # of flies, and their 'date' + 'fly_num' identifiers (numbers should be unique
      within each day), with a separate list of any flies that are now marked for
      exclusion in metadata Google Sheet.

    - # of unique odors per panel, and a list of odors in each

    - # of trials (assumed to be same for each odor?)

    Prints somewhat less if `verbose=False`.
    """
    df = add_mix_str_index_level(df)

    # TODO TODO also switch away from nunique here? see note below
    n_repeats_per_odor = df.index.to_frame(index=False).groupby(['panel', mix_col]
        )['repeat'].nunique()

    max_repeats = n_repeats_per_odor.max()
    assert (n_repeats_per_odor == max_repeats).all()
    print(f'number of trials ("repeats", 0-indexed) per odor: {max_repeats}')

    print('number of unique odors (diff concs counted separately) for each panel:')
    print(df.groupby('panel').apply(lambda x: len(x.index.unique(mix_col))
        ).to_string(header=False)
    )
    if verbose:
        print()
        for panel, pdf in df.groupby('panel'):
            print(f'{panel=} odors:')
            pprint(list(pdf.index.unique(mix_col)))
            print()
    print()

    # a glomerulus is one of ~50 named regions in the antennal lobe. each receives
    # input from all olfactory receptor neurons (ORNs) expressing a corresponding
    # type of receptor. in each glomerulus, all ORNs of one type synapse onto all
    # projection neurons (PNs) of a corresponding type. the PNs then provide input
    # to the Kenyon cells (KCs).
    #
    # the Hallem and Carlson (2006) data you might be familiar with measured signals
    # from ~half of these receptor types, with a different type of measurement.
    def print_n_glom_per_fly(df, desc=''):
        n_glom_per_fly = df.groupby(fly_cols, axis='columns').apply(
            # NOTE: using len(... .unique(...)) instead of nunique b/c at least in one
            # case it seems nunique can be diff, and not what i expected (MUCH larger).
            #
            # ipdb> df.index.nunique('panel')
            # 192
            # ipdb> df.index.unique('panel')
            # Index(['glomeruli_diagnostics', 'megamat', 'validation2'], ...)
            lambda x: len(x.columns.unique('roi'))
        )
        avg_n_glom = n_glom_per_fly.mean()
        print(f'number of {desc}glomeruli per fly (avg: {avg_n_glom:.1f}):\n'
            f'{n_glom_per_fly.to_string()}'
        )
        print()

    certain_rois = certain_roi_indices(df)
    certain_df = df.loc[:, certain_rois]

    # TODO TODO also include sam's _t0 / _t1 suffixes as part of uncertainty
    # determination
    if certain_rois.all():
        print('no glomeruli names indicate uncertainty in identity')
    else:
        uncertain_df = df.loc[:, ~certain_rois]
        print_n_glom_per_fly(uncertain_df, 'UNCERTAIN ')

    unique_glomeruli = sorted(set(certain_df.columns.get_level_values('roi')))
    print(f'unique glomeruli ({len(unique_glomeruli)})', end='')
    if verbose:
        print(f':\n{pformat(unique_glomeruli)}')
    print()

    print_n_glom_per_fly(certain_df)
    print()
    del certain_df, certain_rois

    def print_flies(df):
        unique_flies = get_unique_flies(df)
        n_flies = len(unique_flies)
        print(f'{n_flies=}')
        print(unique_flies.to_string(index=False))
        print()
        return unique_flies

    unique_flies = print_flies(df)

    if len(df.index.unique('panel')) > 1:
        print('flies by panel:')
        for panel, pdf in df.groupby(level='panel'):
            print(f'{panel=}')
            print_flies(pdf.dropna(how='all', axis='columns'))

    # TODO cache gsheet at module level? change in al_analysis so it does that by
    # default (w/ a module-level cache there)?
    gsheet = get_gsheet_metadata()

    not_in_gsheet = []
    now_excluded = []
    for _, row in unique_flies.iterrows():
        try:
            excluded = gsheet.loc[(row.date, row.fly_num), 'exclude']
        except KeyError:
            not_in_gsheet.append(row)
            continue

        if excluded:
            now_excluded.append(row)

    if len(not_in_gsheet) > 0:
        print('flies not currently in my metadata Google Sheet (not my data? bug?):')
        print(pd.concat([x.to_frame().T for x in not_in_gsheet]).to_string(index=False))
        print()

    # to highlight stuff like 2024-01-05/4 in
    # data/sent_to_anoop/v1/validation2_ij_certain-roi_stats.csv
    if len(now_excluded) > 0:
        print('flies now marked for exclusion in my metadata Google Sheet:')
        print(pd.concat([x.to_frame().T for x in now_excluded]).to_string(index=False))
        print()


sep_line = '#' * 88

def summarize_old_panel_csvs(*, verbose=True):
    csv_dir = Path('pebbled_6f/old')

    for i, panel in enumerate(('megamat', 'validation2')):
        if i != 0:
            print(sep_line)

        csv_name = f'{panel}_ij_certain-roi_stats.csv'
        csv = csv_dir / csv_name
        print(f'{panel=}')

        if i == 0:
            if not csv.exists():
                raise IOError('pass input CSV to script or change dir. can not find '
                    f'first default CSV {csv}'
                )

        df = read_csv(csv)

        unique_panels = set(df.index.get_level_values('panel'))
        assert unique_panels == {'glomeruli_diagnostics', panel}

        summarize_antennal_data(df, verbose=verbose)


def csvinfo_cli():
    # TODO does editable install work? doc either way
    """
    CLI install specified via `pyproject.toml`
    """
    parser = argparse.ArgumentParser()
    # TODO include message about what happens by default if not passed
    # (though i might change what that is now...)
    parser.add_argument('csv', nargs='?', type=Path, help='path to CSV to summarize')
    parser.add_argument('-q', '--quiet', action='store_true', help='prints less info')
    parser.add_argument('-p', '--plot', nargs='?', const='', help='plots individual '
        'fly-ROI and mean-ROI matrices and saves alongside input CSV. can pass optional'
        ' (comma-separated) str of plotting kwargs (e.g. -p vmax=1.5), assuming all '
        'values will be floats.'
    )
    args = parser.parse_args()
    csv = args.csv
    plot = args.plot
    verbose = not args.quiet

    def parse_cli_plot_kw_str(plot_kw_str: str) -> Dict[str, Any]:
        parts = plot.split(',')

        cli_plot_kws = dict()
        for p in parts:
            p = p.strip()
            # mainly to handle case where -p passed alone (-> plot_kw_str='')
            if p == '':
                continue

            assert p.count('=') == 1
            var_name, value = p.split('=')

            var_name = var_name.strip()
            value = value.strip()
            assert len(var_name) > 0
            assert len(value) > 0

            assert var_name not in cli_plot_kws
            # assuming all values will be floats for now
            cli_plot_kws[var_name] = float(value)

        return cli_plot_kws

    if plot is not None:
        if csv is None and plot.lower().endswith('.csv'):
            csv = Path(plot)
            cli_plot_kws = dict()
        else:
            cli_plot_kws = parse_cli_plot_kw_str(plot)

        plot = True
    else:
        plot = False

    if csv is not None:
        df = read_csv(csv)

        # TODO should summarize_antennal_data be doing this internally? calling it from
        # anywhere else?
        df = sort_odors(df)

        summarize_antennal_data(df, verbose=verbose)

        if plot:
            plot_fmt = 'pdf'
            def savefig(fig, suffix='', **kwargs):
                fig_name = f'{csv.stem}{suffix}.{plot_fmt}'
                fig_path = csv.parent / fig_name
                print(f'saving figure {fig_path}')
                fig.savefig(fig_path, bbox_inches='tight', **kwargs)

            df = add_mix_str_index_level(df)

            # NOTE: mix_col index level added by add_mix_str_index_level
            trialmean_df = df.groupby(level=['panel', mix_col], sort=False).mean()

            # TODO also work back in al_analysis? why the diff?
            roi_plot_kws['vgroup_label_offset'] = 0.03
            roi_plot_kws['hgroup_label_offset'] = 0.12

            fig, _ = plot_all_roi_mean_responses(trialmean_df, **roi_plot_kws)
            savefig(fig)

            mean_df = trialmean_df.groupby(level='roi', sort=False, axis='columns'
                ).mean()

            vmin = cli_plot_kws.pop('vmin', mean_df.min().min())
            vmax = cli_plot_kws.pop('vmax', mean_df.max().max())

            roimean_plot_kws.update(cli_plot_kws)

            # TODO add args to calls below s.t. warnings get triggered if (many) values
            # are above/below vmin/vmax limits (should already have such a param
            # exposed [/nearly]?)

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

            # NOTE: if input has some uncertain glomeruli, this plot will currently
            # overrepresent the true counts (none of final data I'm working with now
            # should be affected, just older outputs, like
            # data/sent_to_remy/v1/2023-01-06)
            fig, _ = plot_n_per_odor_and_glom(trialmean_df, title=False)
            savefig(fig, '_n-per-odor-and-glom') #, bbox_inches='tight')

            # TODO TODO version of all of the above, but filling to hemibrain glomeruli?
    else:
        # TODO TODO update to now just load ij_certain-roi_stats.csv (and maybe also
        # other csvs i'm now saving). NO LONGER saving these csvs w/ panel prefix in
        # al_analysis.py!
        # (below only loads <panel>_ij_certain-roi_stats.csv files, which al_analysis.py
        # hasn't been writing since maybe 2023-05 or so)

        # TODO support plot here too? prob not...
        assert not plot, 'plot only supported if CSV path provided'
        summarize_old_panel_csvs(verbose=verbose)


# TODO TODO just call csvinfo on each and diff the text output? i did that manually and
# it can be helpful. if i take this approach, may want to add csvinfo option to suppress
# output of lines i don't care to see in the diff (e.g. 'loading data <x>', 'CSV data
# matches <x>', etc)
def csvdiff_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv1', type=Path, help='path to CSV to summarize')
    parser.add_argument('csv2', type=Path, help='path to CSV to summarize')
    args = parser.parse_args()
    csv1 = args.csv1
    csv2 = args.csv2

    df1 = read_csv(csv1)
    df2 = read_csv(csv2)

    # TODO test on (among other things) data/sent_to_remy/2023-10-29/pebbled* vs similar
    # csv i sent here a few hours earlier on same day (on slack. shouldn't have been
    # used, but should only really differ in terms of ~1 non-consensus glomerulus being
    # dropped)

    unique_flies1 = get_unique_flies(df1)
    unique_flies2 = get_unique_flies(df2)

    # TODO delete/refactor (copied from above)
    '''
    def print_flies(df):
        unique_flies = get_unique_flies(df)
        n_flies = len(unique_flies)
        print(f'{n_flies=}')
        print(unique_flies.to_string(index=False))
        print()
        return unique_flies
    '''

    # TODO TODO some of code i added to al_analysis.main (for same/similar purpose)
    # useful here?
    # TODO TODO TODO implement
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    csvinfo_cli()

