#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import xarray as xr


def main():
    nc_name = 'xrda_orn_stim_rdm_concat_mm.nc'
    # <xarray.DataArray (metric: 3, acq: 9, stim_row: 17, stim_col: 17)>
    # ...
    # Coordinates:
    #   * stim_row     (stim_row) object '1-5ol @ -3.0' '1-6ol @ -3.0' ... 'va @ -3.0'
    #   * stim_col     (stim_col) object '1-5ol @ -3.0' '1-6ol @ -3.0' ... 'va @ -3.0'
    #     panel        <U7 'megamat'
    #     date_imaged  (acq) object '2023-04-22' '2023-04-22' ... '2023-06-22'
    #     fly_num      (acq) int64 2 3 2 3 1 3 1 1 1
    #     datefly      (acq) object '2023-04-22/2' '2023-04-22/3' ... '2023-06-22/1'
    #   * metric       (metric) object 'correlation' 'cosine' 'euclidean'
    #     abbrev_row   (stim_row) object '1-5ol' '1-6ol' '1-8ol' ... 'pa' 't2h' 'va'
    #     conc_row     (stim_row) float64 -3.0 -3.0 -3.0 -3.0 ... -3.0 -3.0 -3.0 -3.0
    #     abbrev_col   (stim_col) object '1-5ol' '1-6ol' '1-8ol' ... 'pa' 't2h' 'va'
    #     conc_col     (stim_col) float64 -3.0 -3.0 -3.0 -3.0 ... -3.0 -3.0 -3.0 -3.0
    # Dimensions without coordinates: acq
    # Attributes:
    #     region:       orn
    #     date_shared:  2023-10-29
    #     rdm.metric:   correlation
    d1 = xr.load_dataarray(nc_name)

    regen_dir = Path('update-orn-rdms_output/processed_by_remy/orn_terminals')
    # <xarray.DataArray (metric: 3, acq: 9, stim_row: 17, stim_col: 17)>
    # ...
    # Coordinates:
    #   * stim_row     (stim_row) object '1-5ol @ -3.0' '1-6ol @ -3.0' ... 'va @ -3.0'
    #   * stim_col     (stim_col) object '1-5ol @ -3.0' '1-6ol @ -3.0' ... 'va @ -3.0'
    #     datefly      (acq) object '2023-04-22/2' '2023-04-22/3' ... '2023-06-22/1'
    #     date_imaged  (acq) object '2023-04-22' '2023-04-22' ... '2023-06-22'
    #     fly_num      (acq) int32 2 3 2 3 1 3 1 1 1
    #     panel        object 'megamat'
    #   * metric       (metric) object 'correlation' 'cosine' 'euclidean'
    # Dimensions without coordinates: acq
    # Attributes:
    #     region:       orn_terminals
    #     date_shared:  2023-10-29
    d2 = xr.load_dataarray(regen_dir / nc_name)

    assert np.allclose(d1, d2, equal_nan=True)

    # TODO TODO also assert relevant indices / coords are equal?
    breakpoint()


if __name__ == '__main__':
    main()

