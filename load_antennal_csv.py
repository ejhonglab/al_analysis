#!/usr/bin/env python3

from pathlib import Path
from pprint import pformat

import pandas as pd
import numpy as np


def main():
    csv_dir = Path('pebbled_6f')

    for panel in ('megamat', 'validation2'):
        csv_name = f'{panel}_ij_certain-roi_stats.csv'
        csv = csv_dir / csv_name
        print(f'loading {panel=} data from {csv}')

        df = pd.read_csv(csv,
            # index level names: ['panel', 'is_pair', 'odor1', 'odor2', 'repeat']
            index_col=list(range(5)),
            # column level names: ['date', 'fly_num', 'roi']
            header=list(range(3))
        )

        assert df.columns.names[0] == 'date'
        df.columns = df.columns.set_levels(pd.to_datetime(df.columns.levels[0]),
            level=0, verify_integrity=True
        )

        assert df.columns.names[1] == 'fly_num'
        df.columns = df.columns.set_levels(df.columns.levels[1].astype(int), level=1,
            verify_integrity=True
        )


        # just some checking i was doing against a parallel pickle version i had, mainly
        # to make sure i was loading CSV correctly (with same dtype info)
        pickle_path = csv.with_suffix('.p')
        if pickle_path.exists():
            pdf = pd.read_pickle(pickle_path)

            assert df.index.equals(pdf.index)
            assert df.columns.equals(pdf.columns)

            isna = df.isna()
            assert isna.equals(pdf.isna())

            isclose = np.isclose(df, pdf)
            assert np.logical_xor(isna, isclose).all().all()


        unique_flies = df.columns.to_frame(index=False)[['date', 'fly_num']
            ].drop_duplicates()

        n_flies = len(unique_flies)
        print(f'{n_flies=}')

        # a glomerulus is one of ~50 named regions in the antennal lobe. each receives
        # input from all olfactory receptor neurons (ORNs) expressing a corresponding
        # type of receptor. in each glomerulus, all ORNs of one type synapse onto all
        # projection neurons (PNs) of a corresponding type. the PNs then provide input
        # to the Kenyon cells (KCs).
        #
        # the Hallem and Carlson (2006) data you might be familiar with measured signals
        # from ~half of these receptor types, with a different type of measurement.
        unique_glomeruli = set(df.columns.get_level_values('roi'))
        print(f'unique glomeruli:\n{pformat(unique_glomeruli)}')

        unique_panels = set(df.index.get_level_values('panel'))
        assert unique_panels == {'glomeruli_diagnostics', panel}

        assert df.index.names[0] == 'panel'
        # dropping 'glomeruli_diagnostic' panel, as you probably don't want to analyze
        # that. it's just a series of narrowly activating odors intended to help me
        # identify particular glomeruli
        df = df.loc[panel].copy()

        # we can drop these. metadata intended for binary mixture experiments
        # (not what we are dealing with here)
        assert set(df.index.get_level_values('is_pair')) == {False}
        assert set(df.index.get_level_values('odor2')) == {'solvent'}
        df.index = df.index.droplevel(['is_pair', 'odor2'])

        # averaging over the 3 trials for each odor
        mean_df = df.groupby('odor1', sort=False).mean()

        print()
        import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

