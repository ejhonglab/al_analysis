#!/usr/bin/env python3
#
# run from repo root

from al_analysis import print_dataframe_diff
from load_antennal_csv import read_csv


def main():
    df1 = read_csv('data/sent_to_anoop/v1/megamat_ij_certain-roi_stats.csv')

    # copied from /tmp output that -c generated. should match above
    df2 = read_csv('would-be-new.csv')

    # to remove 'repeat' level, that isn't useful for comparing
    df1 = df1.groupby(['panel', 'odor1']).mean()
    df2 = df2.groupby(['panel', 'odor1']).mean()

    print_dataframe_diff(df1, df2)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

