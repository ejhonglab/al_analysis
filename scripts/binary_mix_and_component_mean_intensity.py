#!/usr/bin/env python3

import pandas as pd

from hong2p.olf import solvent_str, add_mix_str_index_level, mix_col
from load_antennal_csv import read_csv


def main():
    df = read_csv('2024_kiwi_control_roi_data.csv', verbose=True)

    # adds `hong2p.olf.mix_str` (currently 'odor') level, with str formatted from all
    # index levels with component odor strs
    df = add_mix_str_index_level(df)

    binary_mix_df = df[df.index.get_level_values('odor2') != solvent_str].copy()

    # getting components in any of the binary mixtures
    odor1 = set(binary_mix_df.index.get_level_values('odor1'))
    odor2 = set(binary_mix_df.index.get_level_values('odor2'))
    components = odor1 | odor2

    # mix str of any odor presented alone should equal the component str
    mix_and_comps = df[
        df.index.get_level_values(mix_col).isin(components) |
        (df.index.get_level_values('odor2') != solvent_str)
    ].copy()

    # panel    odor
    # kiwi     ea @ -4.2                0.138599
    #          ea @ -4.2 + eb @ -3.5    0.314664
    #          eb @ -3.5                0.348867
    # control  2h @ -5                  0.194156
    #          1o3ol @ -3               0.448030
    #          1o3ol @ -3 + 2h @ -5     0.394453
    # dtype: float64
    print(mix_and_comps.groupby(['panel', mix_col], sort=False).mean(
        ).mean(axis='columns')
    )
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

