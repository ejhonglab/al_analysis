#!/usr/bin/env python3
"""
Plots over-convergence of PNs onto KCs (courtesy of Matt Bauer) vs correlation of the
responses to their cognate ORNs in the Hallem data. No real effect seen.

Data used in this script (MB-Convergence-Data_01-14-23) can be downloaded from Hong lab
Dropbox at:
'HongLab @ Caltech/MB Model/Matt connectome analysis/MB-Convergence-Data_01-14-23'

(excluded from repo b/c it's ~0.5GB)
"""

from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from hong2p import olf
from hong2p.util import melt_symmetric
from drosolf import orns

# TODO refactor so we don't need to import from al_analysis (which still runs too much
# outside of __main__ check) (/ fix al_analysis to not do that)
from al_analysis import panel2name_order


orn_df = orns.orns(columns='glomerulus')

# TODO refactor (duped from al_analysis.py)
orn_df = orn_df.rename(index={
    'b-citronellol': 'B-citronellol',
    'isopentyl acetate': 'isoamyl acetate',
    'E2-hexenal': 'trans-2-hexenal',
})
orn_df = orn_df.rename(index=olf.odor2abbrev)

# TODO TODO get all pairwise combinations of glomeruli -> compute correlation

# TODO do i want to focus on this subset? prob not, at least for initial version
#remy_df = orn_df.loc[panel2name_order['megamat']]

# TODO deal w/ DM3 / 'DM3+DM5' (and any other issues)
hallem_glom_names = set(orn_df.columns)


matt_convergence_data_dir = Path('MB-Convergence-Data_01-14-23')

# TODO rename if i actually wanna keep using pvalues directly (instead of z-scores)
def load_pn2kc_overconvergence_zscores(data_str, data_desc) -> pd.DataFrame:
    data_dir = matt_convergence_data_dir / data_str

    # From README.txt:
    # "Pairwise convergence z-scores and p-values for the observed matrix."
    csv_path = data_dir / f'{data_str}-result.csv'

    # Example data from one of the CSVs:
    # A	B	obs	diff	diff_sd	p
    # D	D	160	0	NA	1
    # D	DA1	49	25.6649	6.08029778983244	0
    # ...
    df = pd.read_csv(csv_path)

    # TODO TODO TODO check both CSVs use same naming scheme for glomeruli
    # (and that it is consistent w/ what i'm deriving from hallem, as much as possible)
    # TODO homogenize glomeruli naming if not

    # TODO if we have A=x, B=y, do we have duplicate row at A=y, B=x?
    # (prob de-dupe if so) (looks like we do, but i didn't check we have *all* such
    # duplicate combos)
    df = df.set_index(['A', 'B'], verify_integrity=True)

    glom_names = set(df.index.get_level_values('A'))
    glom_names2 = set(df.index.get_level_values('B'))
    assert glom_names == glom_names2

    #import ipdb; ipdb.set_trace()

    # TODO TODO TODO TODO check this is what we want!
    # (check matt's thesis + compare CSVs i'm using to other CSVs)
    '''
    zscores = df['diff'] / df['diff_sd']
    # TODO TODO is 'diff' a z-score? or do i need to divide by 'diff_sd' or something?
    #zscores = df['diff']

    df = zscores.to_frame(name=f'{data_desc.lower()}_zscore')
    '''

    pvalues = df['p']
    df = pvalues.to_frame(name=f'{data_desc.lower()}_pvalue')

    # TODO TODO TODO prob move this stuff back to main (and assert names between fafb
    # and hemibrain are same there, which is seems they are)
    # (can then move hallem orn stuff back to main too)
    glom_names = set(df.index.get_level_values('A'))

    print(f'Glomerulus names in Hallem but not {data_desc}:')
    # TODO deal w/ 'DM3+DM5'
    pprint(hallem_glom_names - glom_names)

    # TODO TODO TODO check that none of these should be processed into some name in
    # hallem (at least check if any of these share prefixes with stuff in hallem)
    # (might just be VC3l + VC3m in hemibrain that need to be resolved w/ VC3 in hallem)
    print(f'Glomerulus names in {data_desc} but not Hallem:')
    pprint(glom_names - hallem_glom_names)
    print()
    #

    return df


def main():
    # glomeruli x glomeruli
    orn_corr = orn_df.corr()

    # length: (24 choose 2) + 24
    orn_tidy_corr = melt_symmetric(orn_corr)
    # to be consistent w/ outputs of load_pn2kc_overconvergence_zscores
    orn_tidy_corr.index.names = ['A', 'B']

    # since there isn't a Series.merge method (maybe pd.merge would work tho?)
    orn_tidy_corr = orn_tidy_corr.to_frame(name='hallem_corr')


    hemibrain_str = 'mb'
    fafb_str = 'fafb-mb'

    #hemibrain_zscores = load_pn2kc_overconvergence_zscores(hemibrain_str, 'Hemibrain')
    hemibrain_pvals = load_pn2kc_overconvergence_zscores(hemibrain_str, 'Hemibrain')

    # TODO also merge in fafb
    # TODO TODO chase down size diff (prob just glom names...).
    # orn_tidy_corr was length 300, merged is length 253.
    #df = orn_tidy_corr.merge(hemibrain_zscores, left_index=True, right_index=True)
    df = orn_tidy_corr.merge(hemibrain_pvals, left_index=True, right_index=True)

    #fafb_zscores = load_pn2kc_overconvergence_zscores(fafb_str, 'FAFB')
    fafb_pvals = load_pn2kc_overconvergence_zscores(fafb_str, 'FAFB')

    # TODO delete after sanity checking
    pre_fafb_merge = df.copy()
    #

    #df = df.merge(fafb_zscores, left_index=True, right_index=True)
    df = df.merge(fafb_pvals, left_index=True, right_index=True)
    #import ipdb; ipdb.set_trace()

    # TODO probably drop all identity elements (is zscore 0 for all of them?  hallem
    # corr will be 1 for all of them)
    #
    # brings it down to length 231 (when input was length 243 from only merge w/
    # hemibrain)
    df = df[df.index.get_level_values('A') != df.index.get_level_values('B')].copy()

    '''
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='hallem_corr', y='hemibrain_zscore', ax=ax)
    # TODO refactor plotting + saving to reuse y= str
    fig.savefig('hallem_orn_corr_vs_hemibrain_zscore.png')

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='hallem_corr', y='fafb_zscore', ax=ax)
    fig.savefig('hallem_orn_corr_vs_fafb_zscore.png')
    '''

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='hallem_corr', y='hemibrain_pvalue', ax=ax)
    # TODO refactor plotting + saving to reuse y= str
    fig.savefig('hallem_orn_corr_vs_hemibrain_pvalue.png')

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='hallem_corr', y='fafb_pvalue', ax=ax)
    fig.savefig('hallem_orn_corr_vs_fafb_pvalue.png')

    # TODO try restricting just to remy's odors?

    # TODO does context of rest of z-score data matter (for glomeruli [pairs] w/o data
    # in Hallem)?

    plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

