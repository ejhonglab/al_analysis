#!/usr/bin/env python3
"""
Analyzes outputs made via scripts/model_banana_iaa_concs.py
"""

import pandas as pd
import matplotlib.pyplot as plt

from al_util import cluster_rois
from mb_model import drop_silent_model_cells


def main():
    df = pd.read_csv('model_banana_iaa_concs/weight-divisor_20/spike_counts.csv',
        index_col=0
    )
    df.columns.name = 'odor'

    # binarize reponses, so each element of matrix is True if the model KC (row) spiked
    # at all to that odor (column)
    df = (df > 0)

    print(df.mean())

    df = drop_silent_model_cells(df)
    cluster_rois(df.T)
    plt.show()
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()

