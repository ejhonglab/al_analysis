#!/usr/bin/env python3
"""
Copied just the ORN subset of the analysis in:
`step-02-format-remy-neural-distances.ipynb`
(which otherwise only does similar loading/processing/saving of KC data)
...to make it easier for me to run and test reproducing her 'orn_remy' distances.
"""

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# TODO refactor to share? have all data i need?
data_folder = Path('data')
# TODO delete
#data_folder = Path("/home/remy/PycharmProjects/OdorSpaceShare/manuscript/data/figure-04/04cde")

# TODO refactor to share?
kc_ord = ['2h', 'IaA', 'pa',
          '2-but', 'eb', 'ep',
          'aa', 'va',
          'B-cit', 'Lin',
          '6al', 't2h',
          '1-8ol', '1-5ol', '1-6ol',
          'benz', 'ms'
          ]
# TODO delete
# ['uniform_dat', 'hemibrain_dat', 'uniform_emb', 'hemibrain_emb']

label_map = {
    'chem_rdkit': 'rdkit',
    'chem_fcfp': 'fcfp',
    'chem_ecfp': 'ecfp',
    'chem_pattern': 'pattern',
    'vcf_dat': 'VCF distance',
    'vcf_emb': 'VCF hyp. distance',
    'orn_dat': 'ORN distance',
    'orn_emb': 'ORN distance (embedded)',
    'orn_remy': 'ORN distance (remy)',
    'kc_dat': 'KC distance',
    'kc_emb': 'KC distance (embedded)',
    'kc_remy': 'KC distance (remy)',
    'uniform_dat': "modeled KCs (uniform)",
    'uniform_emb': "modeled KCs (uniform, emb.)",
    'hemibrain_dat': "modeled KCs (hemibrain)",
    'hemibrain_emb': "modeled KCs (hemibrain, emb.)",
}


def preprocess_remy_distance_dataframe(df_distance):
    df_new = df_distance.rename(index=lambda x: x.split(" @ ")[0],
                                columns=lambda x: x.split(" @ ")[0]
                                )
    df_new = df_new.rename_axis(None, axis=0).rename_axis(None, axis=1)
    return df_new


# is `display()` some ipython notebook thing? just print instead? (yea seems so)
# using this function to replace it:
# TODO are all inputs to this actually DataFrame? any DataArray? handle same way?
def display(x: pd.DataFrame) -> None:
    print(x.to_string())
    print()


def main():
    # Load inchi --> abbrev map
    with open(data_folder.joinpath('anoop_inchi_2_abbrev.json'), 'r') as f:
        anoop_inchi_2_abbrev = json.load(f)

    # Also Make abbrev --> inchi map
    anoop_abbrev_2_inchi = {v: k for k, v in anoop_inchi_2_abbrev.items()}

    anoop_inchis = list(anoop_inchi_2_abbrev.keys())
    inchi_pairs = list(combinations(anoop_inchis, 2))
    abbrev_pairs = list(combinations(kc_ord, 2))


    ## Load ORN distances
    panel_name = 'megamat17'
    '''
    da_stim_rdm_concat_mm_orn = xr.load_dataarray(data_folder /
        # TODO TODO TODO can i get tom.py to recreate this file?
        # (or now, should be using update_orn_rdms.py instead)
        #
        # i have a few versions currently committed, none of which have the same md5 at
        # least (haven't checked to what extent contents actually differ):
        # $ find . -name xrda_orn_stim_rdm_concat_mm.nc -exec md5sum "{}" \;
        # 8aa41d515d78ff1a41474b962ca660c7  ./xrda_orn_stim_rdm_concat_mm.nc (80K)
        # 0aed58b9b1057836c01a50da997d5865
        # ./2023-10-29_tom-data_processed-by-remy/orn_terminals/xrda_orn_stim_rdm_concat_mm.nc
        # (75K)
        # 4c3627a18494ba8a1274eed90421d0e0
        # ./2023-10-29_tom-data_processed-by-remy/xrda_orn_stim_rdm_concat_mm.nc (76K)
        #
        # i just copied (+ committed to this repo) the .../by_imaging_panel/... version
        # off tensor-nightly, and its md5 *DOES* match 8aa41d515d78ff1a41474b962ca660c7
        # (of the xrda_orn_stim_rdm_concat_mm.nc at the same level as this script)
        f"by_imaging_panel/{panel_name}/orn_terminals/xrda_orn_stim_rdm_concat_mm.nc"
    )
    '''
    da_stim_rdm_concat_mm_orn = xr.load_dataarray('update-orn-rdms_output/'
        'processed_by_remy/orn_terminals/xrda_orn_stim_rdm_concat_mm.nc'
    )

    da_stim_rdm_concat_mm_orn = (da_stim_rdm_concat_mm_orn
                             .set_index(acq=['date_imaged', 'fly_num'])
                             # .set_index(rdms=['cell_mask_coord', 'metric'])
                             )

    # compute "raw" mean correlation distances
    da_stim_rdm_orn = (da_stim_rdm_concat_mm_orn.sel(metric='correlation'))
    da_stim_rdm_orn_mean = da_stim_rdm_orn.mean(dim='acq')
    df_stim_rdm_orn_mean = da_stim_rdm_orn_mean.to_pandas()

    # compute scaled distances (mean, then scaled)
    da_stim_rdm_orn_mean_scaled = (2.0 / da_stim_rdm_orn_mean.max
        (dim=['stim_row', 'stim_col'])) * da_stim_rdm_orn_mean

    df_stim_rdm_orn_mean_scaled = da_stim_rdm_orn_mean_scaled.to_pandas()

    # compute scaled distances (scaled, then mean)
    da_stim_rdm_orn_scaled = (2.0 / da_stim_rdm_orn.max(dim=['stim_row', 'stim_col'])
        ) * da_stim_rdm_orn
    da_stim_rdm_orn_scaled_mean = da_stim_rdm_orn_scaled.mean(dim='acq')
    df_stim_rdm_orn_scaled_mean = da_stim_rdm_orn_scaled_mean.to_pandas()

    orn_neural_dists = dict(
        orn_remy=df_stim_rdm_orn_mean,
        orn_remy_mean_scaled=df_stim_rdm_orn_mean_scaled,
        orn_remy_scaled_mean=df_stim_rdm_orn_scaled_mean,
    )

    print('megamat17 (9 flies only): `orn_remy`')
    # NOTE: current values from running this script (using .nc file copied from
    # tensor-nightly; the one that is equiv to the one Remy sent me directly over slack
    # recently) match those in comment below
    #
    # copied from notebook output in .ipynb remy sent me:
    # stim_col 	        1-5ol @ -3.0 	1-6ol @ -3.0 ...  t2h @ -3.0	va @ -3.0
    # stim_row
    # 1-5ol @ -3.0 	0.000000 	0.223439     ...  0.447456 	0.798579
    # 1-6ol @ -3.0 	0.223439 	0.000000     ...  0.482146 	0.902255
    # 1-8ol @ -3.0 	0.429620 	0.419934     ...  0.520386 	0.861899
    # 2-but @ -3.0 	0.498254 	0.643171     ...  0.625406 	0.725107
    # 2h @ -3.0 	0.421255 	0.417559     ...  0.586664 	0.701311
    # 6al @ -3.0 	0.362927 	0.426926     ...  0.203800 	0.547210
    # B-cit @ -3.0 	0.775826 	0.662061     ...  0.584890 	0.916365
    # IaA @ -3.0 	0.520031 	0.451292     ...  0.862226 	0.891023
    # Lin @ -3.0 	0.827723 	0.672271     ...  0.636398 	0.871642
    # aa @ -3.0 	0.875689 	0.922147     ...  0.867532 	0.593447
    # benz @ -3.0 	0.840561 	0.902641     ...  0.778112 	0.844455
    # eb @ -3.0 	0.386791 	0.397104     ...  0.765170 	0.687628
    # ep @ -3.0 	0.393776 	0.458814     ...  0.750686 	0.717029
    # ms @ -3.0 	0.951336 	0.988172     ...  0.875812 	0.962242
    # pa @ -3.0 	0.449823 	0.382417     ...  0.744054 	0.905194
    # t2h @ -3.0 	0.447456 	0.482146     ...  0.000000 	0.704344
    # va @ -3.0 	0.798579 	0.902255     ...  0.704344 	0.000000
    display(orn_neural_dists['orn_remy'])

    print('megamat17 (4 flies only): `orn_remy_mean_scaled`')
    # copied from notebook output in .ipynb remy sent me:
    # stim_col 	        1-5ol @ -3.0 	1-6ol @ -3.0 ... t2h @ -3.0 	va @ -3.0
    # stim_row
    # 1-5ol @ -3.0 	0.000000 	0.412306     ... 0.825677 	1.473593
    # 1-6ol @ -3.0 	0.412306 	0.000000     ... 0.889689 	1.664904
    # 1-8ol @ -3.0 	0.792765 	0.774891     ... 0.960254 	1.590437
    # 2-but @ -3.0 	0.919413 	1.186824     ... 1.154043 	1.338018
    # 2h @ -3.0 	0.777329 	0.770509     ... 1.082553 	1.294109
    # 6al @ -3.0 	0.669699 	0.787793     ... 0.376066 	1.009749
    # B-cit @ -3.0 	1.431608 	1.221682     ... 1.079281 	1.690940
    # IaA @ -3.0 	0.959597 	0.832756     ... 1.591040 	1.644177
    # Lin @ -3.0 	1.527372 	1.240522     ... 1.174326 	1.608415
    # aa @ -3.0 	1.615883 	1.701611     ... 1.600830 	1.095070
    # benz @ -3.0 	1.551062 	1.665617     ... 1.435826 	1.558248
    # eb @ -3.0 	0.713733 	0.732764     ... 1.411945 	1.268859
    # ep @ -3.0 	0.726622 	0.846635     ... 1.385217 	1.323112
    # ms @ -3.0 	1.755471 	1.823444     ... 1.616109 	1.775597
    # pa @ -3.0 	0.830045 	0.705663     ... 1.372981 	1.670327
    # t2h @ -3.0 	0.825677 	0.889689     ... 0.000000 	1.299705
    # va @ -3.0 	1.473593 	1.664904     ... 1.299705 	0.000000
    display(orn_neural_dists['orn_remy_mean_scaled'])

    print('megamat17 (4 flies only): `orn_remy_scaled_mean`')
    # copied from notebook output in .ipynb remy sent me:
    # stim_col 	    1-5ol @ -3.0 	1-6ol @ -3.0 ... t2h @ -3.0 	va @ -3.0
    # stim_row
    # 1-5ol @ -3.0 	0.000000 	0.375549     ... 0.745758 	1.292177
    # 1-6ol @ -3.0 	0.375549 	0.000000     ... 0.805640 	1.469973
    # 1-8ol @ -3.0 	0.724386 	0.699511     ... 0.862057 	1.394749
    # 2-but @ -3.0 	0.828608 	1.071963     ... 1.043522 	1.176157
    # 2h @ -3.0 	0.701923 	0.693697     ... 0.980699 	1.140276
    # 6al @ -3.0 	0.605481 	0.711184     ... 0.338547 	0.884068
    # B-cit @ -3.0 	1.289033 	1.098661     ... 0.971697 	1.485610
    # IaA @ -3.0 	0.865809 	0.750946     ... 1.437915 	1.452749
    # Lin @ -3.0 	1.381301 	1.117493     ... 1.060322 	1.412421
    # aa @ -3.0 	1.412166 	1.483642     ... 1.402224 	0.961287
    # benz @ -3.0 	1.394403 	1.496369     ... 1.291642 	1.372804
    # eb @ -3.0 	0.640916 	0.665082     ... 1.273670 	1.119643
    # ep @ -3.0 	0.654050 	0.769182     ... 1.248311 	1.164675
    # ms @ -3.0 	1.576222 	1.640945     ... 1.456091 	1.559601
    # pa @ -3.0 	0.750748 	0.635269     ... 1.242110 	1.478217
    # t2h @ -3.0 	0.745758 	0.805640     ... 0.000000 	1.137801
    # va @ -3.0 	1.292177 	1.469973     ... 1.137801 	0.000000
    display(orn_neural_dists['orn_remy_scaled_mean'])


    ## Combine and save ORN distances
    tidy_orn_dists_abbrev = pd.concat(
        [v.stack().rename(k) for k, v in orn_neural_dists.items()], axis=1
    )
    tidy_orn_dists_abbrev = tidy_orn_dists_abbrev.rename(
        index=lambda x: x.split(" @ ")[0]).rename_axis(index=[None, None])

    # copied from notebook output in .ipynb remy sent me:
    #  	                orn_remy 	orn_remy_mean_scaled 	orn_remy_scaled_mean
    # 1-5ol 	1-5ol 	0.000000 	0.000000        	0.000000
    # 1-6ol      	0.223439 	0.412306        	0.375549
    # 1-8ol     	0.429620 	0.792765        	0.724386
    # 2-but     	0.498254 	0.919413        	0.828608
    # 2h        	0.421255 	0.777329        	0.701923
    # ...       	... 	... 	... 	       ...
    # va 	ep 	0.717029 	1.323112        	1.164675
    # ms 	        0.962242 	1.775597        	1.559601
    # pa        	0.905194 	1.670327        	1.478217
    # t2h        	0.704344 	1.299705        	1.137801
    # va 	        0.000000 	0.000000        	0.000000
    #
    # 289 rows × 3 columns
    display(tidy_orn_dists_abbrev)

    tidy_orn_dists_inchi = tidy_orn_dists_abbrev.rename(index=anoop_abbrev_2_inchi)
    display(tidy_orn_dists_inchi)

    neural_dir = data_folder / 'neural_remy'
    if neural_dir.exists():
        print(f'{neural_dir} already exists! will not overwrite contents! delete it if '
            'you wish to regenerate outputs.'
        )
        return

    neural_dir.mkdir()

    tidy_orn_dists_inchi.to_pickle(neural_dir / 'tidy_orn_dists_inchi.pkl')
    tidy_orn_dists_inchi.to_csv(neural_dir / 'tidy_orn_dists_inchi.tsv', sep='\t')
    print("`tidy_orn_dists_inchi` saved.")

    # well step05b_... loads a similar named tidy_abbrev_dists.pkl, which
    # presumably has ORN subset from this?
    # check ORN subset of that matches (it does) this (there is an assertion in that
    # step05b...py script now)
    tidy_orn_dists_abbrev.to_pickle(neural_dir / 'tidy_orn_dists_abbrev.pkl')
    tidy_orn_dists_abbrev.to_csv(neural_dir / 'tidy_orn_dists_abbrev.tsv', sep='\t')
    print("`tidy_kc_dists_abbrev` saved.")


if __name__ == '__main__':
    main()

