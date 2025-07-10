#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

from hong2p.util import pd_allclose, pd_isclose, pd_indices_equal, shorten_path

from al_util import data_root
from load_antennal_csv import read_csv


def main():
    # the following files in this directory should be commited to this repo:
    # GH146_ij_certain-roi_stats.csv
    # GH146_ij_certain-roi_stats.p
    # pebbled_ij_certain-roi_stats.csv
    # pebbled_ij_certain-roi_stats.p
    paper_csv_dir = data_root / 'sent_to_remy' / '2023-10-29'

    script_dir = Path(__file__).resolve().parent

    # this is where outputs of al_analysis.py are typically saved (the same directory
    # al_analysis.py is in, and is typically run from)
    output_root = script_dir.parent
    orn_dir = output_root / 'pebbled_6f'
    pn_dir = output_root / 'GH146_6f'

    csv_name = 'ij_certain-roi_stats.csv'

    for i, curr_dir in enumerate([orn_dir, pn_dir]):
        if i != 0:
            print()
            print()

        driver = curr_dir.name.split('_')[0]
        paper_csv = paper_csv_dir / f'{driver}_{csv_name}'

        curr_csv = curr_dir / csv_name
        print(f'checking {driver} data...')
        print(f'paper CSV:       {shorten_path(paper_csv,n=4)}')
        print(f'current outputs: {shorten_path(curr_csv,n=3)}')
        print()

        d1 = read_csv(paper_csv)
        d2 = read_csv(curr_csv)

        if pd_allclose(d1, d2, equal_nan=True):
            print('match!')
        else:
            print('did NOT match!')
            # GH146 matches exactly
            assert driver == 'pebbled'

            assert pd_indices_equal(d1, d2)

            mismatch_cols = d1.columns[(~ pd_isclose(d1, d2, equal_nan=True).all())]
            # all odors except the two rare diagnostics not in this fly will thus
            # mismatch slightly
            print()
            print('fly-rois with any mismatch:')
            print(mismatch_cols)

            # on pebbled data (vs 2025-07-09 pebbled outputs, which should have been the
            # same for many months now, if not >1 year [shortly after i sent outputs to
            # remy when editing ROIs slightly for ROI example plot, at B's request]):
            expected_pebbled_mismatch_cols = pd.MultiIndex.from_tuples([
                ('2023-05-10', 1,  'DC3'),
                ('2023-05-10', 1,  'DC4'),
                ('2023-05-10', 1, 'DP1m'),
                ('2023-05-10', 1,  'VA6'),
                ('2023-05-10', 1, 'VA7l'),
                ('2023-05-10', 1, 'VL2a'),
                ('2023-05-10', 1,  'VM2'),
                ('2023-05-10', 1, 'VM5v')],
               names=['date', 'fly_num', 'roi']
            )
            # these are probably the only odors the fly above doesn't have (yup, see
            # below)
            # ipdb> d1.index[pd_isclose(d1, d2, equal_nan=True).T.all()]
            # MultiIndex([('glomeruli_diagnostics', 'aphe @ -5', 0),
            #             ('glomeruli_diagnostics', 'aphe @ -5', 1),
            #             ('glomeruli_diagnostics', 'aphe @ -5', 2),
            #             ('glomeruli_diagnostics',   'CO2 @ 0', 0),
            #             ('glomeruli_diagnostics',   'CO2 @ 0', 1),
            #             ('glomeruli_diagnostics',   'CO2 @ 0', 2)],
            #            names=['panel', 'odor1', 'repeat'])

            # can't directly compare (via .equals()) these MultiIndex b/c diff dtypes:
            # (really just date column is off, b/c it's string in hardcoded var)
            #
            # ipdb> expected_pebbled_mismatch_cols.to_frame(index=False).dtypes
            # date       object
            # fly_num     int64
            # roi        object
            #
            # ipdb> mismatch_cols.dtypes
            # date       datetime64[ns]
            # fly_num             int64
            idf1 = mismatch_cols.to_frame(index=False)
            idf2 = expected_pebbled_mismatch_cols.to_frame(index=False)
            assert idf2.astype(idf1.dtypes).equals(idf1)
            del idf1, idf2

            md1 = d1.loc[:, mismatch_cols]
            md2 = d2.loc[:, mismatch_cols]
            #
            # ipdb> md1.isna().sum().sum()
            # 48
            #
            # ipdb> md1.isna().equals(md2.isna())
            # True
            #
            # ipdb> pd_allclose(md1, md2, equal_nan=True)
            # False
            #
            # ipdb> pd_isclose(md1, md2, equal_nan=True).all()
            # date        fly_num  roi
            # 2023-05-10  1        DC3     False
            #                      DC4     False
            #                      DP1m    False
            #                      VA6     False
            #                      VA7l    False
            #                      VL2a    False
            #                      VM2     False
            #                      VM5v    False
            #
            # ipdb> pd_isclose(md1, md2, equal_nan=True).T.all().any()
            # True
            # ipdb> pd_isclose(md1, md2, equal_nan=True).T.all().sum()
            # 6
            # ipdb> md1.loc[pd_isclose(md1, md2, equal_nan=True).T.all()]
            # date                                   2023-05-10
            # fly_num                                         1
            # roi                                           DC3 DC4 DP1m VA6 VA7l VL2a VM2 VM5v
            # panel                 odor1     repeat
            # glomeruli_diagnostics aphe @ -5 0             NaN NaN  NaN NaN  NaN  NaN NaN  NaN
            #                                 1             NaN NaN  NaN NaN  NaN  NaN NaN  NaN
            #                                 2             NaN NaN  NaN NaN  NaN  NaN NaN  NaN
            #                       CO2 @ 0   0             NaN NaN  NaN NaN  NaN  NaN NaN  NaN
            #                                 1             NaN NaN  NaN NaN  NaN  NaN NaN  NaN
            #                                 2             NaN NaN  NaN NaN  NaN  NaN NaN  NaN
            # ipdb> md1.loc[pd_isclose(md1, md2, equal_nan=True).T.all()].size
            # 48

            print()
            print(f'{d1.max().max()=}')
            print(f'{d2.max().max()=}')
            print()
            # ipdb> md1.max().max()
            # 2.7357987516075406

            # TODO TODO turn these into assertions on difference + warning
            # the differences are not very large at all, and only affect a few glomeruli
            # in one fly:
            max_absdiff = (md1 - md2).abs().max().max()
            print(f'{max_absdiff=}')
            # 0.4670860902171855
            assert max_absdiff <= 0.46709

            mean_absdiff_in_mismatch_flyrois = (md1 - md2).abs().mean().mean()
            print(f'{mean_absdiff_in_mismatch_flyrois=}')
            # 0.019030484580113798
            assert mean_absdiff_in_mismatch_flyrois <= 0.01904

            mean_absdiff_overall = (d1 - d2).abs().mean().mean()
            print(f'{mean_absdiff_overall=}')
            # 0.00046700575656720975
            assert mean_absdiff_overall <= 0.000468


if __name__ == '__main__':
    main()

