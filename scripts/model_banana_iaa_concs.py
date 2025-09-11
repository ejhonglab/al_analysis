#!/usr/bin/env python3
"""
Runs basic MB model(s) and saves outputs for each in their own subdir of `plot_root`
(see below. currently `./model_banana_iaa_concs/`). Subdirs are named with some
important parameters of the model.

`analyze_model_banana_iaa_concs.py` is a short example script analyzing outputs from one
of these directories (currently `model_banana_iaa_concs/weight-divisor_20/`).
"""

from pathlib import Path

import pandas as pd

# TODO delete
import al_util
#
from al_util import warn
from al_analysis import process_sam_glomeruli_names
from mb_model import fit_and_plot_mb_model


# subdirectories will be created, each with output from one parameterization of the MB
# model. this directory added to .gitignore.
plot_root = Path('model_banana_iaa_concs').resolve()

def main():
    # sent to me by Sam, over Slack, on 2025-05-22
    # TODO TODO transpose after process_sam_glomeruli_names, so that can be changed to
    # always expect ROIs in columns
    sam_dff = pd.read_csv('mono_nat_ramp_cond_IaA_Ban.csv', header=0, index_col=0)

    sam_dff.columns.name = 'glomerulus'
    # renaming from 'odor1'. might not be necessary.
    sam_dff.index.name = 'odor'

    assert not sam_dff.isna().any().any()

    # TODO have process_sam* automatically work w/ either 'glomerulus' or 'roi', and
    # then remove 2nd arg here?
    sam_dff = process_sam_glomeruli_names(sam_dff, 'glomerulus')

    sam_dff = sam_dff.T

    # dF/F -> est spike delta scaling constant from my fit on subset of
    # megamat/validation2/glomeruli_diagnostics data that ~overlaps with Hallem
    dff_to_spike_delta_const = 127.0

    # TODO tho check dF/F input data range has similar max[/min] to what i had fit on to
    # produce this constant (seems similar enough for now, with max ~2 dF/F)
    # NOTE: comments below was from before I had processed sam_dff glomeruli names.
    #
    # ipdb> sam_dff.max().max()
    # 2.197755622134672
    #
    # ipdb> sam_dff.max()
    # odor
    # IaA @ -5    0.787015
    # IaA @ -4    0.815997
    # IaA @ -3    1.137053
    # IaA @ -2    1.503476
    # IaA @ -1    2.197756
    # ban @ -4    0.593618
    # ban @ -3    0.516919
    # ban @ -2    0.749566
    # ban @ -1    1.531915
    # ban @ -0    1.671884
    #
    # ipdb> sam_dff.T.max().T
    # roi
    # D               0.564991
    # DA1_t0          0.074277
    # DA2             0.209837
    # DA4l_t1         0.488529
    # DA4m_t1         0.664142
    # DC1             0.574208
    # DC2_t1          1.032803
    # DC3             0.423443
    # DC4             1.007472
    # DL1             0.967578
    # DL2d            0.473850
    # DL2v            0.513464
    # DL5             0.986328
    # DM1             1.418777
    # DM2             1.376647
    # DM3             1.503476
    # DM4             1.343254
    # DM5             1.442196
    # DM6             1.429545
    # DP1l_t1         0.450964
    # DP1m            1.667867
    # VA1d/VA1v_t0    0.065824
    # VA2             0.869086
    # VA3             0.550332
    # VA5             0.493813
    # VA6             0.784694
    # VA7l            0.315879
    # VA7m            0.367000
    # VC1             2.197756
    # VC2             1.429383
    # VC3             1.419319
    # VC4             1.555665
    # VC5_t1          0.815997
    # VL1_t0          0.043421
    # VL2a            0.173683
    # VL2p            0.147811
    # VM2             1.545166
    # VM3             1.752416
    # VM5d            1.671884
    # VM5v            1.101949
    # VM7d            1.724047
    # VM7v_t1         1.501515
    # V_t1            1.157807
    orn_deltas = (sam_dff * dff_to_spike_delta_const)

    # TODO TODO fix mb_model.py code to work w/o adding this panel level
    # (assuming this does actually fix anything...)
    # ...
    # writing /home/tom/src/al_analysis/scripts/model_banana_iaa_concs/weight-divisor_20/params.csv
    # Traceback (most recent call last):
    #   File "./model_banana_iaa_concs.py", line 182, in <module>
    #     main()
    #   File "./model_banana_iaa_concs.py", line 170, in main
    #     param_dict = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
    #   File "/home/tom/src/al_analysis/mb_model.py", line 5255, in fit_and_plot_mb_model
    #     assert len(spike_counts.columns.shape) == 1 and spike_counts.columns.name == 'odor'
    # AssertionError
    #
    #
    # TODO TODO TODO fix how sort_odors fails w/ missing panels
    # (or at least add a hack to add this panel to that order...)
    # ...
    # writing /home/tom/src/al_analysis/scripts/model_banana_iaa_concs/weight-divisor_20/params.csv
    # Traceback (most recent call last):
    #   File "./model_banana_iaa_concs.py", line 167, in <module>
    #     main()
    #   File "./model_banana_iaa_concs.py", line 155, in main
    #     param_dict = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
    #   File "/home/tom/src/al_analysis/mb_model.py", line 5280, in fit_and_plot_mb_model
    #     responses = sort_odors(responses, panel=panel, warn=False)
    #   File "/home/tom/src/al_analysis/al_util.py", line 793, in sort_odors
    #     return olf.sort_odors(df, panel_order=panel_order,
    #   File "/home/tom/src/hong2p/hong2p/olf.py", line 585, in sort_odors
    #     assert panel in panel2name_order
    # AssertionError
    panel = 'ban_iaa'
    orn_deltas.columns = pd.MultiIndex.from_arrays([
        pd.Index(data=[panel] * len(orn_deltas.columns), name='panel'),
        orn_deltas.columns
    ])
    # TODO delete (or restore if i want to get outputs again, before fixing sort_odors
    # issue)
    # hack to get sort_odors call in fit_and_plot_... not fail
    #al_util.panel2name_order[panel] = ['IaA', 'ban']
    #

    kws = dict(
        weight_divisor=20,
    )

    # TODO delete this product thing, and just switch to a kws_list?
    try_each_with_kws = [
        # TODO delete? or replace w/ something they might care about more, like varying
        # target_sparsity
        #dict(use_connectome_APL_weights=True),

        # probably always want `kws` unmodified too. that's what this empty dict is for
        dict(),
    ]

    for extra_kws in try_each_with_kws:
        # extra_kws will override kws without warning, if they have common keys
        param_dict = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
            try_cache=False, **{**kws, **extra_kws}
        )
        output_dir = (plot_root / param_dict['output_dir']).resolve()
        assert output_dir.is_dir()
        assert output_dir.parent == plot_root

        # you could read the output like this
        #df = pd.read_csv(output_dir / 'spike_counts.csv', index_col='model_kc')


if __name__ == '__main__':
    main()
