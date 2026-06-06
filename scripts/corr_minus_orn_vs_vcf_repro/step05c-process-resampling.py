#!/usr/bin/env python3
"""
date created: 2025-05-28
last run: 2025-05-28
odor panel: validation2

Compute summary stats for resampling analysis.
Generate a dataframe with the following statistics:
ci_0.025
ci_0.050
ci_0.500
ci_0.950
ci_0.975
mean

Index columns:
- resampling_fraction: 0.1 - 1.0
- space_row
- space_col
- stat: should be 'ci_*' or 'mean'
- space_metric: 'pearson' or 'spearman'

Input files:
===========
See `figure-06/all_odor_spaces/resampled/with_replacement/{}_odor_pairs/`
    - xrds_resampled_136_iter01000_no_replacement.nc
    - etc.

Ran resampling for `n_iter = 1000` and `n_iter = 5000`.

Output files:
=============
Save to `figure-06/all_odor_spaces/resampled/df_resampled_summary_stats_niter{}.pkl`

Each dataframe should contain the results from different `n_odor_pairs` and `with_replacement`.

"""

import json
from itertools import combinations, product
# from tqdm.notebook import tqdm
from pathlib import Path
from typing import List, Tuple

import matplotlib_inline
import pandas as pd
import xarray as xr
from tqdm import tqdm

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

#data_folder = Path("/home/remy/PycharmProjects/OdorSpaceShare/manuscript/data/"
#                   "figure-04/04cde")
data_folder = Path('data')

kc_ord = ['2h', 'IaA', 'pa',
          '2-but', 'eb', 'ep',
          'aa', 'va',
          'B-cit', 'Lin',
          '6al', 't2h',
          '1-8ol', '1-5ol', '1-6ol',
          'benz', 'ms'
          ]

# Load inchi --> abbrev map
with open(data_folder.joinpath('anoop_inchi_2_abbrev.json'), 'r') as f:
    anoop_inchi_2_abbrev = json.load(f)

# Also Make abbrev --> inchi map
anoop_abbrev_2_inchi = {v: k for k, v in anoop_inchi_2_abbrev.items()}

anoop_inchis = list(anoop_inchi_2_abbrev.keys())
inchi_pairs = list(combinations(anoop_inchis, 2))
abbrev_pairs = list(combinations(kc_ord, 2))


def get_resampling_filename(n_iter: int, n_odor_pairs: int, with_replacement: bool,
                            filetype: str) -> str:
    if filetype == 'xarray':
        filename = (f"xrds_resampled_{n_odor_pairs:d}_iter{n_iter:05d}_"
                    f"{'with_replacement' if with_replacement else 'no_replacement'}"
                    f".nc"
                    )
    elif filetype == 'pandas':
        filename = (f"df_resampled_{n_odor_pairs:d}_iter{n_iter:05d}_"
                    f"{'with_replacement' if with_replacement else 'no_replacement'}"
                    f".parquet"
                    )
    else:
        raise ValueError(f"Unknown resampling file type: {filetype}")
    return filename


def get_resampling_folder(n_iter: int, n_odor_pairs: int, with_replacement: bool) -> Path:
    return (data_folder / "resampled3/"
            f"{'with_replacement' if with_replacement else 'no_replacement'}/"
            f"{n_odor_pairs:d}_odor_pairs"
    )


def get_resampling_file(n_iter: int, n_odor_pairs: int, with_replacement: bool,
                        filetype: str) -> Path:
    return (get_resampling_folder(n_iter, n_odor_pairs, with_replacement) /
            get_resampling_filename(n_iter, n_odor_pairs, with_replacement, filetype)
            )


def get_summary_filename(n_iter: int, n_odor_pairs: int, with_replacement: bool):
    return (f"df_resampled_summary_stats_{n_odor_pairs:d}_iter{n_iter:05d}_"
            f"{'with_replacement' if with_replacement else 'no_replacement'}"
            f".pkl"
            )


def get_summary_file(n_iter: int, n_odor_pairs: int, with_replacement: bool):
    return (get_resampling_folder(n_iter, n_odor_pairs, with_replacement) /
            get_summary_filename(n_iter, n_odor_pairs, with_replacement))


def load_resampling(n_iter: int, n_odor_pairs: int, with_replacement: bool) -> xr.Dataset:
    file = get_resampling_file(n_iter, n_odor_pairs, with_replacement, filetype='xarray')
    xr.load_dataset(file)


def load_summary_stats(n_iter: int, n_odor_pairs: int, with_replacement: bool) -> pd.DataFrame:
    # TODO is this actually a valid way to return stuff (now? ever?)
    # (doesn't seem so in at least my 3.8.12)
    pd.read_pickle(get_summary_file(n_iter, n_odor_pairs, with_replacement))


def compute_summary_stats(df_resampled: pd.DataFrame) -> pd.DataFrame:
    """Compute confidence intervals and mean for resampled space correlations

    Args:
        df_resampled (pd.DataFrame): Has columns ['pearson', 'spearman', 'n_overlap_pairs_sampled',]
    Returns:
        df_resampled_stats (pd.DataFrame):
            MultiIndex names: ['resampling_fraction', 'space_row', 'space_col', 'stat']
            Columns: ['pearson', 'spearman']
    """
    quantiles = [0.025, 0.05, 0.5, 0.95, 0.975]
    df_resampled_stats_ci = (df_resampled
                             .groupby(['resampling_fraction', 'space_row', 'space_col'])
                             .quantile(quantiles, interpolation='midpoint')
                             .rename_axis(index=['resampling_fraction',
                                                 'space_row',
                                                 'space_col',
                                                 'stat']
                                          )
                             .rename(lambda x: f"ci_{x:.3f}", level=3)
                             )
    df_resampled_stats_mean = (df_resampled
                               .groupby(['resampling_fraction', 'space_row', 'space_col'])
                               .mean()
                               .assign(stat='mean').set_index('stat', append=True)
                               )
    df_resampled_stats = pd.concat([df_resampled_stats_ci, df_resampled_stats_mean],
                                   axis=0)
    return df_resampled_stats


def load_pooled_resampling_summary_stats(n_iters: int) -> pd.DataFrame:
    """Load summary stats.

    Args:
        n_iters: # of resampling iterations
    Returns:
          df_pooled_summary_stats (pd.DataFrame):
            MultiIndex names:
                ['n_odor_pairs', 'with_replacement', 'resampling_fraction', 'space_row',
                'space_col', 'stat']
            Columns: ['pearson', 'spearman']
    """
    pass


# %%

if __name__ == "__main__":
    # %%

    n_iter = 5000

    for n_odor_pairs, with_replacement in product([128, 136], [True, False]):
        print(n_odor_pairs, with_replacement)
        parquet_file = get_resampling_file(n_iter, n_odor_pairs, with_replacement,
                                           filetype='pandas')

        print(f"\tsaving to parquet {parquet_file}")
        resampling_file = get_resampling_file(n_iter, n_odor_pairs, with_replacement,
                                              filetype='xarray')

        # each file ~1.9GB in memory (~1.3GB on disk)
        df = xr.load_dataset(resampling_file).to_dataframe()

        check = True
        # TODO restore False
        check_only = True
        if check:
            # the file of hers I originally copied off tensor-nightly path she mentioned
            parts = resampling_file.parts
            assert parts[0] == 'data'
            assert parts[1] == 'resampled3'
            parts = list(parts)
            parts[1] = 'resampled3_remy_backup'
            remy_resampling_file = Path('/'.join(parts))

            print("checking my recomputed outputs equal to Remy's...", flush=True,
                end=''
            )
            df2 = xr.load_dataset(remy_resampling_file).to_dataframe()
            assert df.equals(df2)
            print('done', flush=True)

            if check_only:
                continue

        df.to_parquet(parquet_file, engine='pyarrow')
        print(f"\tDone!")

    # %% Make summary statistics dataframe
    #   Quantiles for ci at 90% (0.05, 0.95) and 95% (0.024, 0.975)
    n_iter = 5000

    for n_odor_pairs, with_replacement in tqdm(product([128, 136], [True, False])):
        print(n_odor_pairs, with_replacement)

        # Load
        # ----
        parquet_file = get_resampling_file(n_iter, n_odor_pairs, with_replacement,
                                           filetype='pandas')
        # TODO print paths to these
        #
        df_resampled = pd.read_parquet(parquet_file, engine='pyarrow')

        # Compute stats
        # ---------------
        df_summary_stats = compute_summary_stats(df_resampled)

        summary_file = get_summary_file(n_iter, n_odor_pairs, with_replacement)
        print(summary_file)
        df_summary_stats.to_pickle(summary_file)

        # # Compute difference tables
        # space_info = [
        #     ('orn_remy', 'pn_dendrites-correlation', 'PN dendrites - ORN'),
        #     ('orn_remy', 'pn_boutons_F_zscore', 'PN boutons - ORN'),
        #     ('orn_remy', 'kc_claws_Fc_zscore', 'KC claws - ORN'),
        #     ('orn_remy', 'kc_remy_combo', 'KC - ORN'),
        #     ]
        #
    # %%
    # %%
