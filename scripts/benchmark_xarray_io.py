#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
from socket import gethostname
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

from hong2p.util import is_path_on_hdd, file_size_bytes, format_nbytes
from hong2p.xarray import save_dataarray, load_dataarray, NETCDF_ENGINE

from mb_model import (fit_mb_model, MODEL_KW_LIST, megamat_orn_deltas,
    format_model_params
)


def main():
    parser = ArgumentParser(description='will save/load model claw_sims to NetCDF file '
        'in current path, to benchmark xarray IO, including with different compression '
        'options'
    )
    args = parser.parse_args()

    model_kws = None
    for x in MODEL_KW_LIST:
        # just want first element where this is true, since we know we will have
        # claw_sims in output
        if x.get('prat_claws'):
            model_kws = x
            break
    assert model_kws is not None, 'found no model parameter entry w/ prat_claws=True'

    # model_kws:
    # {'one_row_per_claw': True,
    #  'pn_claw_to_apl': True,
    #  'prat_boutons': True,
    #  'prat_claws': True,
    #  'use_connectome_APL_weights': True}
    print('model_kws:')
    pprint(model_kws)
    print()

    orn_deltas = megamat_orn_deltas()

    def get_claw_sims(**kwargs):
        with warnings.catch_warnings():
            # to silence all my many informational UserWarnings in fit_mb_model and
            # things it calls
            warnings.filterwarnings('ignore', category=UserWarning)

            # fixed_thr=200 & wAPLKC=2.2 produce average response rate of 0.108783, for
            # current model_kws (in comment above).
            _, _, _, params = fit_mb_model(orn_deltas=orn_deltas, fixed_thr=200,
                wAPLKC=2.2, return_dynamics=True, silent=True, **model_kws, **kwargs
            )

        arr = params['claw_sims']
        del params
        return arr

    # TODO use temp file? care about actual io though, and won't that probably just be
    # going to memory? still a meaningful benchmark? maybe better than writing to some
    # random device, which could be SSD or HDD? detect which and print that in benchmark
    # results?

    # TODO also try to benchmark memory usage while saving? that an issue for any of
    # these fns? any make a copy in process? (excluding check=True on save_dataarray,
    # where obviously it would)

    path = Path('.')
    # same path used for everything
    nc_path = path / 'test.nc'

    final_pbar_position = 2
    def benchmark_write(arr: xr.DataArray, *, n_writes: int = 3, **kwargs) -> float:
        print(f'benchmarking writes {kwargs=}')
        last = time.time()
        write_times = []
        for i in tqdm(range(n_writes), unit='save call', position=final_pbar_position,
            leave=False):

            save_dataarray(arr, nc_path, check=False, **kwargs)
            now = time.time()
            curr_write_time_s = now - last
            write_times.append(curr_write_time_s)
            last = now

        mean_write_time = np.mean(write_times)
        print(f'mean write time: {mean_write_time:.1f}s')
        return mean_write_time


    def benchmark_load(use_xr_load: bool = False, n_loads: int = 3,
        silent: bool = False) -> float:
        if not silent:
            if not use_xr_load:
                print('benchmarking load with my hong2p.xarray.load_dataarray '
                    '(which may be deferring all of the actual loading)'
                )
            else:
                print('benchmarking load with xr.load_dataarray')

        load_iter = range(n_loads)
        if not silent:
            load_iter = tqdm(load_iter, unit='load call', position=final_pbar_position,
                leave=False
            )

        last = time.time()
        load_times = []
        for i in load_iter:
            # use xr.load_datarray instead, for benchmarking? or test whether there is
            # even a difference? open_dataarray doesn't load it all into memory, and
            # that is deferred until some later point when it's needed (but next call
            # may force that anyway?)
            if not use_xr_load:
                # this one actually calls xr.open_dataarray, which doesn't necessarily
                # load all into memory (and then makes one call after:
                # move_all_coords_to_index, which might or might not force it all into
                # memory)
                arr = load_dataarray(nc_path)
            else:
                # this call *does* always load all into memory
                arr = xr.load_dataarray(nc_path)
            del arr

            now = time.time()
            curr_load_time_s = now - last
            load_times.append(curr_load_time_s)
            last = now

        mean_load_time = np.mean(load_times)
        if not silent:
            print(f'mean load time: {mean_load_time:.1f}s')

        return mean_load_time

    # TODO just do my own processing in here that is equiv to delete_pretime=True, to
    # only need the one call (w/ delete_pretime=False)?

    # so we know which computer the benchmark was run on
    hostname = gethostname()

    # will print device / partition current path is on, as well as whether it's on HDD
    hdd = is_path_on_hdd(path, verbose=True)
    print(f'using hong2p.xarray.{NETCDF_ENGINE=}')
    print()

    zlib_at_diff_complevels = [dict(encoding=dict(zlib=True, complevel=x))
        # >=4 and things start to get slow enough I don't think it's worth it
        # i think 1 is the min and 9 is the max
        # TODO are there other args that might speed it up? (thought i saw something
        # about "shuffle" once, maybe in netcdf4 docs? not seeing in xarray docs now)
        for x in (1, 4, 9)
    ]
    # TODO TODO define this from product of current contents and (current engine,
    # 'h5netcdf')? and try compression='gzip' & compression_opts=9 there (LZF better?)
    # TODO TODO or only product of default dict() w/ diff engines, and then specify
    # engine specific compression options?
    write_kw_list = [dict()] + zlib_at_diff_complevels

    # TODO restore?
    # TODO add CLI arg to test both?
    #model_kw_list = [dict(delete_pretime=False), dict(delete_pretime=True)]
    model_kw_list = [dict(delete_pretime=True)]

    # TODO look better if i reverse order of all positions, so outer is 2, and innermost
    # is 0? would be complicated
    next_position = 0
    if len(model_kw_list) == 1:
        model_iter = model_kw_list
    else:
        # TODO omitting leave=False improve anything? i think so?
        model_iter = tqdm(model_kw_list, unit='model', position=next_position)
        next_position += 1

    bench_data_list = []
    for kws in model_iter:
        print(f'extra kwargs to fit_mb_model: {kws}')

        print('calling fit_mb_model...', end='', flush=True)
        # TODO how did i seem to get olfsysm tee output in one random call after having
        # seemingly previously successfully silenced all output (including that) via
        # new silent=True path?
        arr = get_claw_sims(**kws)
        print('done')

        print(f'arr size in memory: {format_nbytes(arr.nbytes)}')

        # TODO do i have a more generic named format_params or format_dict somewhere?
        # use that instead?
        #
        # need exclude=False to ignore the default mb_model.exclude_params exclusion of
        # delete_pretime
        extra_kws_str = format_model_params(kws, exclude=False)

        # TODO TODO try h5netcdf engine w/ compression=gzip, compression_opts=9 (not
        # supported by netcdf4 engine)? and compare it's default performance?
        # docs mention LZF as one option supported by h5py (and thus presumably
        # h5netcdf), and this alg seems well suited to fast reading/writing, so might be
        # attractive. try that?

        for write_kws in tqdm(write_kw_list, unit='encoding-args',
                position=next_position,
            ):
            # TODO some way to get this print to not clobber pbar over write_kws? try
            # leave=True? (i think that helped?) add another newline print before this?
            print(f'additional hong2p.xarray.save_dataarray args: {write_kws}')

            engine = write_kws.pop('engine', NETCDF_ENGINE)

            assert len(write_kws) <= 1
            assert len(write_kws) == 0 or 'encoding' in write_kws

            encoding_kws = dict()
            if 'encoding' in write_kws:
                encoding_kws = write_kws['encoding']

            # TODO do i have a more generic named format_params or format_dict
            # somewhere? use that instead?
            encoding_str = format_model_params(encoding_kws)

            before = time.time()
            # just doing one call w/ check=True, to make sure my save/load scheme is
            # working before we benchmark anything
            print('writing arr and checking loaded value matches...', end='')
            save_dataarray(arr, nc_path, check=True, **write_kws)
            print('done')
            intiial_write_time_s = time.time() - before
            print(f'initial write took {intiial_write_time_s:.1f}s')

            nbytes = file_size_bytes(nc_path)
            size_str = format_nbytes(nbytes)
            print()
            print(f'{nc_path} size on disk: {size_str}')
            compression_ratio = nbytes / arr.nbytes
            print(f'compression ratio: {compression_ratio:.3f}')
            print()

            single_write_only = False
            if len(encoding_kws) > 0 and encoding_kws['complevel'] >= 5:
                # ~60s per write, or more, at complevel=9
                # even at complevel=1 it is taking ~40s per call now, w/
                # delete_pretime=False
                print('skipping multiple-write benchmarking, because very slow in this '
                    'case'
                )
                mean_write_time_s = intiial_write_time_s
                single_write_only = True
            else:
                mean_write_time_s = benchmark_write(arr, **write_kws)
            print()

            # NOTE: my current implementation does not actually seem to force dataarray
            # (from xr.open_datarray) to be loaded into memory, because all of the
            # hong2p.xarray.load_dataarray times essentially instant. therefore, can
            # skip benchmarking use_xr_load=False for now
            deferred_load_time_s = benchmark_load(use_xr_load=False, n_loads=1,
                silent=True
            )
            # highest seen so far (on a HDD): 0.0551
            assert deferred_load_time_s < 0.1, (f'{deferred_load_time_s} greater than '
                'expected. may need to include use_xr_load=False in benchmarking.'
            )

            mean_load_time_s = benchmark_load(use_xr_load=True)
            print()

            curr_bench_data = {
                'extra_model_kws': extra_kws_str,
                'encoding_kws': encoding_str,
                'compression_ratio': compression_ratio,
                'mean_write_time': mean_write_time_s,
                'mean_load_time': mean_load_time_s,
                'single_write_only': single_write_only,

                'engine': engine,

                # TODO actually test on multiple drive types in one run?
                'hdd': hdd,

                'host': hostname,
            }
            bench_data_list.append(curr_bench_data)

        del arr
        print()

    print(f'deleting test NetCDF file {nc_path}')
    nc_path.unlink()

    df = pd.DataFrame.from_records(bench_data_list)
    print()
    print('benchmark summary:')
    print(df.to_string(index=False))
    print()

    csv_name = 'xarray_netcdf_io_bench.csv'
    print(f'writing benchmark summary to {csv_name}')
    df.to_csv(csv_name, index=False)


if __name__ == '__main__':
    main()

