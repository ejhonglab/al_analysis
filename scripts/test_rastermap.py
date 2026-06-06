#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap


def main():
    # ipdb> spks.shape
    # (1830, 5500)
    # ipdb> set(spks.flat)
    # {0.0, 1.0}
    spks = np.load('spks.npy')

    # TODO maybe try converting to spike times in seconds, and using
    # `rastermap.io.load_spike_times(<spike-time-npy>, <spike-cluster-npy>,
    # st_bin=100)` (or whatever it does)?
    # TODO TODO or just try re-binning my spike time matrix into one with a bin of
    # 100ms (from st_bin above)? do we ever have two spikes within one of those
    # bins? how does rastermap fn handle that? add? is output binary or not?
    assert not np.isnan(spks).any()

    # getting responders only. leaves 214 of the initial 1830 cells.
    spks = spks[(spks == 1).any(axis=1)]

    # before dropping non-responders and also changing some other rastermap config
    # (decrease n_PCs and n_clusters, tho maybe both not needed?) was getting this
    # error:
    #   File "./mb_model.py", line 3296, in fit_mb_model
    #     model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75,
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/rastermap/rastermap.py", line 327, in fit
    #     Usv_valid = SVD(X[igood][:, itrain] if itrain is not None else X[igood],
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/rastermap/svd.py", line 33, in SVD
    #     U = TruncatedSVD(n_components=nmin,
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/sklearn/utils/_set_output.py", line 140, in wrapped
    #     data_to_wrap = f(self, X, *args, **kwargs)
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/sklearn/base.py", line 1151, in wrapper
    #     return fit_method(estimator, *args, **kwargs)
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/sklearn/decomposition/_truncated_svd.py", line 246, in fit_transform
    #     U, Sigma, VT = randomized_svd(
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/sklearn/utils/extmath.py", line 450, in randomized_svd
    #     Q = randomized_range_finder(
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/sklearn/utils/extmath.py", line 279, in randomized_range_finder
    #     Q, _ = linalg.lu(safe_sparse_dot(A.T, Q), permute_l=True)
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/scipy/linalg/_decomp_lu.py", line 213, in lu
    #     a1 = asarray_chkfinite(a)
    #   File "/home/tom/src/al_analysis/venv/lib/python3.8/site-packages/numpy/lib/function_base.py", line 628, in asarray_chkfinite
    #     raise ValueError(
    # ValueError: array must not contain infs or NaNs
    #
    # ipdb> spks.isna().any().any()
    # False
    # still fails after dropping non-responders
    #model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75,
    #    time_lag_window=5
    #).fit(spks)
    #
    # still fails after dropping non-responders
    #model = Rastermap().fit(spks)
    #
    # remy was saying this works. did not work for me. need the other params? (at
    # least some of them, yes)
    # still fails after dropping non-responders
    #model = Rastermap(n_clusters=50).fit(spks)
    #
    # works after dropping non-responders
    #model = Rastermap(n_PCs=100, n_clusters=50, locality=0.75,
    #    time_lag_window=5
    #).fit(spks)
    #
    # works after dropping non-responders
    # TODO how does this compare to version w/ locality + time_lag_window set above.
    # prefer either? try diff (lower?) n_PCs/n_clusters?
    # TODO try smoothing parameters / diff binning?
    #model = Rastermap(n_PCs=100, n_clusters=50).fit(spks)

    dt = 0.0005
    time_lag_window_seconds = 0.1
    time_lag_window = int(time_lag_window_seconds / dt)

    kws_list = [
        #dict(
        #    n_PCs=100, n_clusters=50, locality=0.75, time_lag_window=5
        #),

        # TODO delete
        #dict(
        #    n_PCs=100, n_clusters=50, locality=0.75, time_lag_window=time_lag_window
        #),
        #dict(
        #    n_PCs=100, n_clusters=50, locality=0.0, time_lag_window=time_lag_window
        #),
        #

        # TODO restore?
        #dict(
        #    n_PCs=100, n_clusters=5, locality=0.75, time_lag_window=time_lag_window
        #),

        # these two should look same. they do. delete one/both
        #dict(
        #    n_PCs=100, n_clusters=5, locality=0, time_lag_window=time_lag_window
        #),
        dict(
            n_PCs=100, n_clusters=5, time_lag_window=time_lag_window
        ),
        #

        # these two look the same. i think locality=0 might be the default?
        #dict(
        #    n_PCs=100, n_clusters=5, locality=0
        #),
        dict(
            n_PCs=100, n_clusters=5
        ),

        # TODO TODO does time lag actually matter? not clearly better w/ it set vs
        # default, when using n_clusters=5, n_PCs=100 (and locality default/0, which i
        # think are same)

        # TODO delete. too many clusters (worse than w/ n_clusters=5, and nothing else
        # changed)
        #dict(
        #    n_PCs=100, n_clusters=50
        #),
        #
    ]
    for kws in kws_list:
        print(f'{kws=}')
        model = Rastermap(**kws).fit(spks)

        y = model.embedding # neurons x 1
        isort = model.isort

        # visualize binning over neurons
        X_embedding = model.X_embedding

        # plot
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
        ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")

        # TODO add assertion there are no spikes after 4600 (there shouldn't be on curr
        # data)
        # 4000 should be ~index of odor onset (at t=0s)
        ax.set_xlim([4000, 4600])

        ax.set_title(str(kws))

        print()

    plt.show()
    breakpoint()


if __name__ == '__main__':
    main()

