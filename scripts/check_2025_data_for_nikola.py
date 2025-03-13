#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

from hong2p.olf import parse_odor_name
from load_antennal_csv import read_csv


def main():
    data_dir = Path('~/src/al_analysis/data/sent_to_nikola/2025-02-25').expanduser()

    mdf = read_csv(data_dir / 'megamat_ij_certain-roi_stats.csv')
    vdf = read_csv(data_dir / 'validation2_ij_certain-roi_stats.csv')

    # ['glomerulus', 'sensillum', 'receptors', ...]
    tdf = pd.read_csv(data_dir / 'task22_table3.csv')

    task_gloms = set(tdf.glomerulus)

    def get_gloms(df):
        return set(df.columns.get_level_values('roi'))

    vgloms = get_gloms(vdf)
    mgloms = get_gloms(mdf)
    assert vgloms - task_gloms == set()
    assert mgloms - task_gloms == set()


    # ['name', 'abbrev', 'cid', 'solvent', 'pubchem_url']
    odf = pd.read_csv(data_dir / 'odor_metadata.csv')
    abbrevs = set(odf.abbrev)

    def get_odors(df):
        return {parse_odor_name(x) for x in df.index.get_level_values('odor1')}

    # ipdb> vodors - abbrevs
    # {'1-3ol', 'paa', 'HCl'}
    # ipdb> modors - abbrevs
    # {'HCl', 'paa', 'CO2'}
    vodors = get_odors(vdf)
    modors = get_odors(mdf)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

