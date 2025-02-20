#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


# remy sent this to me on slack 2025-02-05
inchi2abbrev = {
    '1S/C5H12O/c1-2-3-4-5-6/h6H,2-5H2,1H3': '1-5ol',
    '1S/C6H14O/c1-2-3-4-5-6-7/h7H,2-6H2,1H3': '1-6ol',
    '1S/C8H18O/c1-2-3-4-5-6-7-8-9/h9H,2-8H2,1H3': '1-8ol',
    '1S/C4H8O/c1-3-4(2)5/h3H2,1-2H3': '2-but',
    '1S/C7H14O/c1-3-4-5-6-7(2)8/h3-6H2,1-2H3': '2h',
    '1S/C6H12O/c1-2-3-4-5-6-7/h6H,2-5H2,1H3': '6al',
    '1S/C10H20O/c1-9(2)5-4-6-10(3)7-8-11/h5,10-11H,4,6-8H2,1-3H3': 'B-cit',
    '1S/C7H14O2/c1-6(2)4-5-9-7(3)8/h6H,4-5H2,1-3H3': 'IaA',
    '1S/C10H18O/c1-5-10(4,11)8-6-7-9(2)3/h5,7,11H,1,6,8H2,2-4H3': 'Lin',
    '1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)': 'aa',
    '1S/C7H6O/c8-6-7-4-2-1-3-5-7/h1-6H': 'benz',
    '1S/C6H12O2/c1-3-5-6(7)8-4-2/h3-5H2,1-2H3': 'eb',
    '1S/C5H10O2/c1-3-5(6)7-4-2/h3-4H2,1-2H3': 'ep',
    '1S/C8H8O3/c1-11-8(10)6-4-2-3-5-7(6)9/h2-5,9H,1H3': 'ms',
    '1S/C7H14O2/c1-3-4-5-6-9-7(2)8/h3-6H2,1-2H3': 'pa',
    '1S/C6H10O/c1-2-3-4-5-6-7/h4-6H,2-3H2,1H3/b5-4+': 't2h',
    '1S/C5H10O2/c1-2-3-4-5(6)7/h2-4H2,1H3,(H,6,7)': 'va'
}

# probably already have a version of this in al_analysis, but re-implementing here to
# make this script self contained
def drop_silent_cells(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df > 0).T.any()].copy()


def df_triu(df: pd.DataFrame) -> pd.Series:
    # NOTE: w/ k=1 (in contrast to default k=0) this excludes the diagonal.
    # diagonal and below will now have values replaced w/ NaN
    triu = df.where(np.triu(np.ones(df.shape), k=1).astype(bool))

    # this will dropna too
    ser = triu.stack()

    assert ser.index.names == ['odor', 'odor']
    ser.index.names = ['odor1', 'odor2']

    return ser


def main():
    # seems that anoop did NOT do this, at least for computing the correlations i'm
    # loading below
    #DROP_SILENT_CELLS = False
    DROP_SILENT_CELLS = True

    this_script_dir = Path(__file__).parent

    # should contain the final model data (newer than sent_to_anoop/[v1,v2] dirs)
    data_dir = Path(this_script_dir / '../data/sent_to_anoop/2024-05-16')

    # first CSV column is 'model_kc' (w/ kc number from 0) and rest are odor
    # abbreviations w/ '@ -3' suffix. responses are binary (0/1)
    hb = pd.read_csv(data_dir / 'megamat_hemibrain_model_responses.csv',
        index_col='model_kc'
    )
    hb.columns.name = 'odor'
    hb.columns = hb.columns.str.replace(' @ -3', '')

    # first column is 'seed', and following columns are as above
    u7 = pd.read_csv(data_dir / 'megamat_uniform_model_responses_n-seeds_100.csv',
        index_col=['seed', 'model_kc']
    )
    u7.columns.name = 'odor'
    u7.columns = u7.columns.str.replace(' @ -3', '')

    if DROP_SILENT_CELLS:
        hb = drop_silent_cells(hb)
        u7 = drop_silent_cells(u7)

    hb_corr_dist = 1 - hb.corr()

    seed_corrs = []
    for seed, sdf in u7.groupby('seed', sort=False):
        seed_corr = sdf.corr()

        seed_corr_ser = seed_corr.stack()
        seed_corr_ser.index.names = ['odor1', 'odor2']

        # prepending a 'seed' level to index, so we can concat after and average over
        # this level
        seed_corr_ser = pd.concat([seed_corr_ser], names=['seed'], keys=[seed])

        seed_corrs.append(seed_corr_ser)

    u7_mean_corr = pd.concat(seed_corrs).groupby(['odor1','odor2'], sort=False).mean()

    # now the pairs are sorted, unlike above, and my version of pandas doesn't have a
    # sort=False arg to [Series|DataFrame].unstack(), so will need to re-order to
    # compare to hemibrain corrs
    u7_corr = u7_mean_corr.unstack()
    # on top of re-ordering odors to match hemibrain corr, this also seems to rename
    # index/columns to 'odor' (from 'odor1','odor2' they were before)
    u7_corr = u7_corr.loc[hb_corr_dist.index, hb_corr_dist.index].copy()
    u7_corr_dist = 1 - u7_corr

    hb_corr_dist_ser = df_triu(hb_corr_dist)
    u7_corr_dist_ser = df_triu(u7_corr_dist)


    anoop_dir = Path(this_script_dir / '../data/from_anoop/megamat_dmats')

    # both of these contain correlation distances, and the rows seem to be in the same
    # order as the columns, as diagonal has 0 correlation distance (rows unlabeled)
    anoop_hb = pd.read_csv(anoop_dir / 'KC_mod_hb_dmat.csv')
    orig_anoop_cols = anoop_hb.columns.copy()
    assert anoop_hb.columns.isin(inchi2abbrev).all()
    anoop_hb.columns = anoop_hb.columns.map(inchi2abbrev)
    anoop_hb.columns.name = 'odor'
    anoop_hb.index = anoop_hb.columns

    anoop_u7 = pd.read_csv(anoop_dir / 'KC_mod_un_dmat.csv')
    assert anoop_u7.columns.equals(orig_anoop_cols)
    assert anoop_u7.columns.isin(inchi2abbrev).all()
    anoop_u7.columns = anoop_u7.columns.map(inchi2abbrev)
    anoop_u7.columns.name = 'odor'
    anoop_u7.index = anoop_u7.columns


    if not DROP_SILENT_CELLS:
        # remy said she noticed he scales everything to have a max distance of 2 (one
        # factor across the whole matrix)
        my_hb = 2 * (hb_corr_dist / hb_corr_dist.max().max())
        my_u7 = 2 * (u7_corr_dist / u7_corr_dist.max().max())

        assert np.allclose(my_hb, anoop_hb)
        print("successfully recreated Anoop's hemibrain correlation distances")

        assert np.allclose(my_u7, anoop_u7)
        print("successfully recreated Anoop's uniform correlation distances")

        # just as another sanity check (to compare to his CSVs). could delete.
        my_hb.columns = orig_anoop_cols
        my_u7.columns = orig_anoop_cols
        my_hb.to_csv('KC_mod_hb_dmat_AS-ANOOP.csv', index=False)
        my_u7.to_csv('KC_mod_un_dmat_AS-ANOOP.csv', index=False)
        #
    else:
        print('writing CSVs with correlation distances computed NOW DROPPING SILENT '
            'CELLS, which Anoop did not'
        )
        assert anoop_hb.columns.equals(hb_corr_dist.columns)
        assert anoop_u7.columns.equals(u7_corr_dist.columns)
        assert hb_corr_dist.columns.equals(u7_corr_dist.columns)

        hb_corr_dist.columns = orig_anoop_cols
        u7_corr_dist.columns = orig_anoop_cols

        # TODO save under script dir?
        hb_corr_dist.to_csv('KC_mod_hb_dmat_SILENT-CELLS-DROPPED.csv', index=False)
        u7_corr_dist.to_csv('KC_mod_un_dmat_SILENT-CELLS-DROPPED.csv', index=False)


if __name__ == '__main__':
    main()

