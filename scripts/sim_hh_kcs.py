#!/usr/bin/env python3

from itertools import product
from pathlib import Path
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from hong2p.xarray import load_dataarray

import al_util
from al_util import warn, savefig
from mb_model import (plot_spike_rasters, summarize_isi, REFRACTORY_PERIOD,
    get_odor_fname_suffix
)


def main():
    al_util.verbose = True

    script_path = Path(__file__).resolve().parent

    plot_dir = script_path / 'hh_output'
    plot_dir.mkdir(exist_ok=True)

    # pn firing rate at each time point
    pn_path = script_path / 'example_pn_sims.nc'

    pns = load_dataarray(pn_path)
    ts = pns.get_index('time_s')
    # TODO TODO am i screwing up how i'm defining the time indices now?
    # maybe some off-by-one? use np.arange rather than np.linspace (or vice versa?)
    #
    # ipdb> np.diff(ts[:2])[0]
    # 0.0005002000800319872
    # ipdb> np.isclose(np.diff(ts[:2])[0], 0.0005)
    # False
    # ipdb> np.diff(ts)
    # array([0., 0., 0., ..., 0., 0., 0.])
    # ipdb> np.unique(np.diff(ts))
    # array([0., 0., 0., 0.])
    # ipdb> for x in np.unique(np.diff(ts)):
    #     print(x)
    #
    # 0.0005002000800318207
    # 0.0005002000800319317
    # 0.0005002000800319872
    # 0.0005002000800320427
    dt = 0.0005

    rng = np.random.default_rng(seed=0)
    for_spikes = rng.uniform(size=pns.shape)

    # TODO TODO want to enforce a refractory period? some efficient way to do that? that
    # also hopefully doesn't break statistics?
    # TODO TODO use time bins of size equal to refractory period, then sample back to
    # time i want (putting spike at start of each bin, if there is one)?
    # TODO <= vs < matter?
    spikes = (for_spikes <= (pns * dt))
    n_spikes = spikes.sum().item()
    print(f'# spikes (generated at original dt): {n_spikes}')

    summarize_isi(spikes)

    # TODO TODO replace whatever is generating current time indices w/ arange like this
    # this is more well behaved than ts above.
    # TODO TODO but last point is not time end. just accept that, to have dt match
    # better? shouldn't care about points out there. it's 0.749499799919968 instead of
    # 0.75. ig issues is just if there is an offset earlier that matters (prob not)?
    # ipdb> np.unique(np.diff(np.arange(ts2[0], ts2[-1], dt)))
    # array([0., 0., 0., 0.])
    # ipdb> for x in np.unique(np.diff(np.arange(ts2[0], ts2[-1], dt))):
    #     print(x)
    # 0.0004999999999999449
    # 0.0005000000000000004
    # 0.000500000000000056
    # 0.000500000000000167
    ts2 = pd.Index(np.arange(ts[0], ts[-1], dt), name='time_s')

    ts_down = pd.Index(np.arange(ts[0], ts[-1], REFRACTORY_PERIOD), name='time_s')
    pns_down = pns.reindex(time_s=ts_down, method='nearest')
    # if method were default =None, this would not be true
    assert not pns_down.isnull().any().item()

    for_spikes_down = rng.uniform(size=pns_down.shape)
    # TODO <= vs < matter?
    spikes_down = (for_spikes_down <= (pns_down * REFRACTORY_PERIOD))
    n_spikes_down = spikes_down.sum().item()
    print(f'# spikes (first downsampled to dt={REFRACTORY_PERIOD=}): {n_spikes_down}')
    summarize_isi(spikes_down)

    # in initial test, only failed on rtol=0.0005 (didn't test anything between 0.001
    # and that)
    assert np.isclose(n_spikes, n_spikes_down, rtol=0.001)

    spikes_up = xr.zeros_like(spikes)
    for stim, glom in product(spikes_down.stim, spikes_down.glomerulus):
        ss_down = spikes_down.sel(stim=stim, glomerulus=glom).squeeze()
        spikes_only = ss_down[ss_down]
        spike_up_indices = ts.get_indexer(spikes_only.time_s.values, method='nearest')
        assert len(spikes_only) == len(spike_up_indices)
        # times selected from upsampled (=original) time index
        spike_times = ts[spike_up_indices]
        assert np.abs((spike_times - spikes_only.time_s.values).values).max() < dt
        indexer_dict = dict(stim=stim, glomerulus=glom, time_s=spike_times)
        assert len(spikes_up.loc[indexer_dict]) == len(spikes_only)
        # b/c current values are True/False. may change to 0/1 later, but nbd
        spikes_up.loc[indexer_dict] = True

    assert spikes_up.sum().item() == spikes_down.sum().item()

    print('after upsampling the spikes generated with dt=REFRACTORY_PERIOD:')
    # output is the same
    # TODO assert that (by calling individual fns? or having this wrap a fn returning a
    # str, and asserting strs equal?)
    summarize_isi(spikes_up)

    odor = 't2h @ -3'
    odor_str = get_odor_fname_suffix(odor)

    fig, ax = plt.subplots()
    plot_spike_rasters(spikes.sel(odor=odor).squeeze(), ax=ax)
    ax.set_title(f'PN poisson spiking\n{odor=}')
    savefig(fig, plot_dir, f'pn_poisson_spikes{odor_str}')

    # NOTE: this one looks darker, presumably because more spikes relative to number of
    # cells, despite covering same amount of time (but does plotting code care about
    # that? prob not).
    # also, the clustering behaves differently on the two versions below, so need to fix
    # order to one, in order to have output look comparable
    fig, ax = plt.subplots()
    _, order = plot_spike_rasters(spikes_down.sel(odor=odor).squeeze(), ax=ax)
    ax.set_title('PN poisson spiking\ngenerated by downsampling to refractory period\n'
        f'{odor=}'
    )
    savefig(fig, plot_dir, f'pn_poisson_spikes{odor_str}_downsampled')

    fig, ax = plt.subplots()
    plot_spike_rasters(spikes_up.sel(odor=odor).squeeze(), fixed_order=order, ax=ax)
    ax.set_title('PN poisson spiking\ngenerated by downsampling to refractory period, '
        f'then upsampling back\n{odor=}'
    )
    savefig(fig, plot_dir, f'pn_poisson_spikes{odor_str}_generated-at-refractory-dt')

    # TODO TODO try simulating like turner/bazhenov/laurent 2007?
    # TODO TODO but with my observed wPNKC? (w/ and w/o?)

    breakpoint()


if __name__ == '__main__':
    main()

