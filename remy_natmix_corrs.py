#!/usr/bin/env python3

import matplotlib.pyplot as plt
import xarray as xr

from hong2p.xarray import move_all_coords_to_index
from hong2p.viz import matshow


def main():
    # TODO document (at least) when i got this data, which flies of hers it contains.
    # (probably in data/README.md)
    da = xr.load_dataarray('from_remy/natmix/da_corrs.nc')

    codors = ['MS @ -3','VA @ -3','FUR @ -4','2H @ -5','OCT @ -3','control mix @ 0']
    kodors = ['EtOH @ -2','IAol @ -3.6','IAA @ -3.7','EA @ -4.2','EB @ -3.5','~kiwi @ 0']

    #da.where(da.odor1.isin(kodors), drop=True).where(da.odor1_b.isin(kodors), drop=True)

    kda = da[da.panel == 'kiwi']

    # TODO why is this size (10, 18, 18) and indexing just on odor1 (before panel) gives
    # (15, 18, 18)? pfo?
    kda = kda.where(kda.odor1.isin(kodors), drop=True).where(kda.odor1_b.isin(kodors),
        drop=True
    )
    kdf = move_all_coords_to_index(kda.mean('fly_panel')).to_pandas()
    mkdf = kdf.groupby('odor1').mean().groupby('odor1_b', axis='columns').mean()

    cda = da[da.panel == 'control']
    cda = cda.where(cda.odor1.isin(codors), drop=True).where(cda.odor1_b.isin(codors),
        drop=True
    )
    cdf = move_all_coords_to_index(cda.mean('fly_panel')).to_pandas()
    mcdf = cdf.groupby('odor1').mean().groupby('odor1_b', axis='columns').mean()


    #kda.dropna('odor', how='all').dropna('fly_panel', how='all').dropna('odor_b', how='all')
    #kdf = move_all_coords_to_index(kda.mean('fly_panel')).to_pandas()

    # (just to inspect N. 4-5 for cont and 7-10 for kiwi, it seems)
    #kda.notnull().sum('fly_panel')
    #move_all_coords_to_index(kda.notnull().sum('fly_panel')).to_pandas()
    #move_all_coords_to_index(cda.notnull().sum('fly_panel')).to_pandas()


    matshow(mkdf.loc[kodors, kodors])


    plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

