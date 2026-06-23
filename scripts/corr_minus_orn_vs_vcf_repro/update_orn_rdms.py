#!/usr/bin/env python3
# Example script from Remy, that I modified somewhat (mainly commenting stuff).
# Still need to apply a change she suggested on Slack or otherwise fix a bug towards the
# end of this script, though most of it runs.

from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import external
import xrsa
import stimuli

# TODO delete. not used
#tom_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/analysis_outputs/from_tom")

# megamat17 panel
########################
date_shared = "2023-10-29"

# %% ORN responses
# TODO delete
#df_orn_ori = pd.read_pickle("/local/matrix/Remy-Data/projects/odor_space_collab/"
#                            "analysis_outputs/from_tom/2023-10-29/pebbled_ij_certain-roi_stats.p")
#
df_orn_ori = pd.read_pickle("pebbled_ij_certain-roi_stats.p")
da_orn_ori = external.tom.convert.fix_da_ori(
        external.tom.convert.df_ori_2_dataarray(df_orn_ori)
        )
# fix stimuli
da_orn_ori['odor1'] = ('row', stimuli.fix.fix_stim(da_orn_ori['odor1'].to_numpy(),
                                                   conc_as_float=True))
da_orn = external.tom.convert.reshape_responses(da_orn_ori)
da_orn = da_orn.assign_attrs({'region': 'orn_terminals',
                              'date_shared': date_shared})
# %% PN responses

'''
df_pn_ori = pd.read_pickle("/local/matrix/Remy-Data/projects/odor_space_collab/"
                           "analysis_outputs/from_tom/2023-10-29/GH146_ij_certain-roi_stats.p")
da_pn_ori = external.tom.convert.fix_da_ori(
        external.tom.convert.df_ori_2_dataarray(df_pn_ori),
        )
da_pn_ori['odor1'] = ('row', stimuli.fix.fix_stim(da_pn_ori['odor1'].to_numpy(),
                                                  conc_as_float=True))
da_pn = external.tom.convert.reshape_responses(da_pn_ori)
da_pn = da_pn.assign_attrs({'region': 'pn_dendrites',
                            'date_shared': date_shared})
'''

# %% Make multiindex, and sort
da_orn_with_glomeruli_diagnostics = (da_orn
                                     .set_index(trials=['panel', 'stim', 'stim_occ'])
                                     .set_index(acq=['date_imaged', 'fly_num', 'datefly'])
                                     ).copy(deep=True)
'''
da_pn_with_glomeruli_diagnostics = (da_pn
                                    .set_index(trials=['panel', 'stim', 'stim_occ'])
                                    .set_index(acq=['date_imaged', 'fly_num', 'datefly'])
                                    ).copy(deep=True)
'''
# %%
da_orn_wd_trial_rdm_multimetric = external.tom.multimetric.make_multimetric_trial_rdms(
        da_orn_with_glomeruli_diagnostics,
        metric_list=['correlation', 'cosine', 'euclidean'])

da_orn_wd_stim_rdm_multimetric = external.tom.multimetric.make_multimetric_stim_rdms(
        da_orn_with_glomeruli_diagnostics.groupby('stim').mean(dim='trials'),
        metric_list=['correlation', 'cosine', 'euclidean']
        )
da_orn_wd_blockavg_rdm_multimetric = xrsa.qc.compute_trial_rdm_blockavg(
        da_orn_wd_trial_rdm_multimetric)

# %%
# TODO rename (->output_dir?). not actually used to load input data, and only used for
# outputs / plots
data_dir = Path('update-orn-rdms_output')
# TODO delete
#data_dir = Path(
#        "/local/matrix/Remy-Data/projects/odor_space_collab/analysis_outputs/from_tom").joinpath(
#        date_shared)


data_dir.mkdir(exist_ok=True)

data_dir.joinpath('processed_by_remy').mkdir(exist_ok=True)
data_dir.joinpath('processed_by_remy', 'orn_terminals').mkdir(exist_ok=True)
#data_dir.joinpath('processed_by_remy', 'pn_dendrites').mkdir(exist_ok=True)

'''
(da_orn_with_glomeruli_diagnostics
 .reset_index('trials')
 .reset_index('acq')
 ).to_netcdf(data_dir.joinpath('processed_by_remy',
                               'orn_terminals_with_diagnostics',
                               'xrda_orn_with_glomeruli_diagnostics.nc'))
'''

'''
(da_pn_with_glomeruli_diagnostics
 .reset_index('trials')
 .reset_index('acq')
 ).to_netcdf(data_dir.joinpath('processed_by_remy',
                               'pn_dendrites_with_diagnostics',
                               'xrda_pn_with_glomeruli_diagnostics.nc'))
'''

# %%
# save trial RDMs
'''
((da_orn_trial_rdm_multimetric
  .reset_index('acq')
  .reset_index('trial_row')
  .reset_index('trial_col')).to_netcdf(
        data_dir.joinpath(
                'processed_by_remy',
                'orn_terminals',
                'xrda_orn_trial_rdm_concat_mm.nc'))
)
'''

# %%
da_orn = (da_orn
          .set_index(trials=['panel', 'stim', 'stim_occ'])
          .set_index(acq=['date_imaged', 'fly_num', 'datefly'])
          .sel(panel='megamat')
          )
da_orn = da_orn.sortby('trials')

'''
da_pn = (da_pn
         .set_index(trials=['panel', 'stim', 'stim_occ'])
         .set_index(acq=['date_imaged', 'fly_num', 'datefly'])
         .sel(panel='megamat')
         )

da_pn = da_pn.sortby('trials')
'''
# %% Compute trial ORN RDM

# make ORN dataarrays
da_orn_trial_rdm_multimetric = external.tom.multimetric.make_multimetric_trial_rdms(
        da_orn, metric_list=['correlation', 'cosine', 'euclidean'])

da_orn_stim_rdm_multimetric = external.tom.multimetric.make_multimetric_stim_rdms(
        da_orn.groupby('stim').mean(dim='trials'),
        metric_list=['correlation', 'cosine', 'euclidean']
        )
da_orn_blockavg_rdm_multimetric = xrsa.qc.compute_trial_rdm_blockavg(da_orn_trial_rdm_multimetric)

# %% Make PN dataarrays
'''
da_pn_trial_rdm_multimetric = external.tom.multimetric.make_multimetric_trial_rdms(
        da_pn, metric_list=['correlation', 'cosine', 'euclidean'])

da_pn_stim_rdm_multimetric = external.tom.multimetric.make_multimetric_stim_rdms(
        da_pn.groupby('stim').mean(dim='trials'),
        metric_list=['correlation', 'cosine', 'euclidean']
        )
da_pn_blockavg_rdm_multimetric = xrsa.qc.compute_trial_rdm_blockavg(da_pn_trial_rdm_multimetric)
'''
# %% Save to from_tom/{{date_shared}}/processed_by_remy

save_netcdf = True

if save_netcdf:
    # save responses
    (da_orn
     .reset_index('trials')
     .reset_index('acq')
     ).to_netcdf(data_dir.joinpath('processed_by_remy', 'orn_terminals', 'xrda_orn.nc'))

    '''
    (da_pn
     .reset_index('trials')
     .reset_index('acq')
     ).to_netcdf(data_dir.joinpath('processed_by_remy', 'pn_dendrites', 'xrda_pn.nc'))
    '''

    # save trial RDMs
    '''
    ((da_orn_trial_rdm_multimetric
      .reset_index('acq')
      .reset_index('trial_row')
      .reset_index('trial_col')).to_netcdf(
            data_dir.joinpath(
                    'processed_by_remy',
                    'orn_terminals',
                    'xrda_orn_trial_rdm_concat_mm.nc'))
    )
    '''

    '''
    (da_pn_trial_rdm_multimetric
     .reset_index('acq')
     .reset_index('trial_row')
     .reset_index('trial_col')).to_netcdf(
            data_dir.joinpath(
                    'processed_by_remy',
                    'pn_dendrites',
                    'xrda_pn_trial_rdm_concat_mm.nc'))
    '''

    # save stim RDMs
    (da_orn_stim_rdm_multimetric
     .reset_index('acq')
     .to_netcdf(data_dir.joinpath('processed_by_remy',
                                  'orn_terminals',
                                  'xrda_orn_stim_rdm_concat_mm.nc'))
     )

    '''
    (da_pn_stim_rdm_multimetric
    .reset_index('acq').to_netcdf(
            data_dir.joinpath('processed_by_remy',
                              'pn_dendrites',
                              'xrda_pn_stim_rdm_concat_mm.nc')
            )
    )

    # save blockavg.
    (da_orn_blockavg_rdm_multimetric.reset_index('acq')
     .to_netcdf(data_dir.joinpath('processed_by_remy',
                                  'orn_terminals',
                                  'xrda_orn_blockavg_rdm_concat_mm.nc')))
    (da_pn_blockavg_rdm_multimetric.reset_index('acq')
     .to_netcdf(data_dir.joinpath('processed_by_remy',
                                  'pn_dendrites',
                                  'xrda_pn_blockavg_rdm_concat_mm.nc')))
     '''

# %% KC order

# kc_ord = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep', 'aa', 'va', 'B-cit', 'Lin', '6al', 't2h',
#           '1-8ol', '1-5ol', '1-6ol', 'benz', 'ms']
# kc_ord = [f"{item} @ -3.0" for item in kc_ord]
#
# isort = pd.Categorical(da_orn_trial_rdm_concat['row_stim'].to_numpy(),
#                        categories=kc_ord, ordered=True).argsort()
# # %% Plot ORN RDMs
# da_orn_trial_rdm_concat = ds_orn_trial_rdm_mm.sel(metric='correlation')
# da_orn_stim_rdm_concat = ds_orn_stim_rdm_mm.sel(metric='correlation')
# da_orn_blockavg_rdm_concat = ds_orn_blockavg_rdm_mm.sel(metric='correlation')
#
# agg_orn_plotter = external.tom.plot.AggregateOrnPlotter(
#         trial_rsm_concat=1 - da_orn_trial_rdm_concat.isel(trial_row=isort, trial_col=isort),
#         stim_rsm_concat=1 - da_orn_stim_rdm_concat.reindex(stim_row=kc_ord, stim_col=kc_ord),
#         trial_rsm_concat_blockavg=1 - da_orn_blockavg_rdm_concat.reindex(trial_row=kc_ord,
#                                                                          trial_col=kc_ord),
#         heatmap_kws=dict(cmap='RdBu_r', vmin=-1, vmax=1),
#         )
#
# fig_trials = agg_orn_plotter.plot_individual_and_mean_rsms('trial')
# fig_trials.suptitle(f'ORN trial RDMs (metric=correlation)\ndate={date_shared}')
# plt.show()
#
# fig_stim = agg_orn_plotter.plot_individual_and_mean_rsms('stim')
# fig_stim.suptitle(f'ORN stim RDMs (metric=correlation)\ndate={date_shared}')
# plt.show()
#
# fig_blockavg = agg_orn_plotter.plot_individual_and_mean_rsms('blockavg')
# fig_blockavg.suptitle(f'ORN blockavg. RDMs (metric=correlation)\ndate={date_shared}')
#
# fig_summary, g_summary = agg_orn_plotter.plot_trial_and_stim_summary_rsms()
# fig_summary.suptitle(f'ORN summary RDMs (metric=correlation)\ndate={date_shared}', fontsize=10)
#
# plt.show()
# # %% Save ORN RDM plots
# plot_dir = data_dir.joinpath('plots')
# plot_dir.mkdir(exist_ok=True, parents=True)
#
# with (PdfPages(plot_dir.joinpath(
#         f'orn_RDMs__correlation__mean_and_individual__kc_ord__{date_shared}.pdf')) as pdf):
#     pdf.savefig(fig_trials)
#     pdf.savefig(fig_stim)
#     pdf.savefig(fig_blockavg)
#
# fig_summary.savefig(plot_dir.joinpath(f'orn_summary_RDMs__correlation__kc_ord__{date_shared}.pdf'))
#
# plot_dir.joinpath('orn_pngs').mkdir(exist_ok=True)
#
# fig_trials.savefig(plot_dir.joinpath('orn_pngs',
#                                      f'orn_trial_rdms__correlation__mean_and_individual'
#                                      f'__kc_ord__{date_shared}.png'))
#
# fig_stim.savefig(plot_dir.joinpath('orn_pngs',
#                                    f'orn_stim_rdms__correlation__mean_and_individual__kc_ord__'
#                                    f'{date_shared}.png'))
# fig_blockavg.savefig(plot_dir.joinpath('orn_pngs',
#                                        f'orn_blockavg_rdms__correlation__mean_and_individual__kc_ord_'
#                                        f'_{date_shared}.png'))
# fig_summary.savefig(plot_dir.joinpath('orn_pngs',
#                                       f'orn_summary_rdms__correlation__kc_ord__{date_shared}.png'))
#
# # %%
# plot_dir = data_dir.joinpath('plots')
# plot_dir.mkdir(exist_ok=True, parents=True)
#
# da_pn_trial_rdm_concat = ds_pn_trial_rdm_mm.sel(metric='correlation')
# da_pn_stim_rdm_concat = ds_pn_stim_rdm_mm.sel(metric='correlation')
# da_pn_blockavg_rdm_concat = ds_pn_blockavg_rdm_mm.sel(metric='correlation')
#
# agg_pn_plotter = external.tom.plot.AggregateOrnPlotter(
#         trial_rsm_concat=1 - da_pn_trial_rdm_concat.isel(trial_row=isort, trial_col=isort),
#         stim_rsm_concat=1 - da_pn_stim_rdm_concat.reindex(stim_row=kc_ord, stim_col=kc_ord),
#         trial_rsm_concat_blockavg=1 - da_pn_blockavg_rdm_concat.reindex(trial_row=kc_ord,
#                                                                         trial_col=kc_ord),
#         heatmap_kws=dict(cmap='RdBu_r', vmin=-1, vmax=1),
#         )
#
# fig_trials = agg_pn_plotter.plot_individual_and_mean_rsms('trial')
# fig_trials.suptitle(f'PN trial RDMs (metric=correlation)\ndate={date_shared}')
# plt.show()
#
# fig_stim = agg_pn_plotter.plot_individual_and_mean_rsms('stim')
# fig_stim.suptitle(f'PN stim RDMs (metric=correlation)\ndate={date_shared}')
# plt.show()
#
# fig_blockavg = agg_pn_plotter.plot_individual_and_mean_rsms('blockavg')
# fig_blockavg.suptitle(f'PN blockavg. RDMs (metric=correlation)\ndate={date_shared}')
# plt.show()
#
# fig_summary, g_summary = agg_pn_plotter.plot_trial_and_stim_summary_rsms()
# fig_summary.suptitle(f'PN summary RDMs (metric=correlation)\ndate={date_shared}', fontsize=10)
# plt.show()
# # %%
# with (PdfPages(plot_dir.joinpath(
#         f'pn_RDMs__correlation__mean_and_individual__kc_ord__{date_shared}.pdf')) as pdf):
#     pdf.savefig(fig_trials)
#     pdf.savefig(fig_stim)
#     pdf.savefig(fig_blockavg)
#
# fig_summary.savefig(plot_dir.joinpath(
#         f'pn_summary_RDMs__correlation__kc_ord__{date_shared}.pdf'))
#
# plot_dir.joinpath('pn_pngs').mkdir(exist_ok=True)
#
# fig_trials.savefig(plot_dir.joinpath('pn_pngs',
#                                      f'pn_trial_rdms__correlation__mean_and_individual'
#                                      f'__kc_ord__{date_shared}.png'))
#
# fig_stim.savefig(plot_dir.joinpath('pn_pngs',
#                                    f'pn_stim_rdms__correlation__mean_and_individual__kc_ord__'
#                                    f'{date_shared}.png'))
# fig_blockavg.savefig(plot_dir.joinpath('pn_pngs',
#                                        f'pn_blockavg_rdms__correlation__mean_and_individual__kc_ord_'
#                                        f'_{date_shared}.png'))
# fig_summary.savefig(plot_dir.joinpath('pn_pngs',
#                                       f'pn_summary_rdms__correlation__kc_ord__{date_shared}.png'))
#
#
# # %%
# def compute_orn_rdms(da_orn_trials_, da_orn_stim_, metric_list=None):
#     if metric_list is None:
#         metric_list = ['correlation', 'cosine', 'euclidean']
#
#     trial_rdm_data_ = {}
#     stim_rdm_data_ = {}
#     trial_rdm_blockavg_data_ = {}
#
#     for metric_ in metric_list:
#         da_orn_trial_rdm_concat_ = xr.concat(
#                 [xrsa.rdm.compute_trial_respvec_rdm(
#                         da_.dropna(dim='cells', how='all').dropna(dim='trials', how='all'),
#                         metric=metric_)
#                     for _, da_ in da_orn_trials_.groupby('datefly')],
#                 dim='acq').rename(metric_)
#
#         da_orn_stim_rdm_concat_ = xr.concat(
#                 [xrsa.rdm.compute_rdm(
#                         da_.dropna(dim='cells', how='all').dropna(dim='stim', how='all'),
#                         input_dim_ord=['stim', 'cells'],
#                         metric=metric_)
#                     for _, da_ in da_orn_stim_.groupby('datefly')],
#                 dim='acq').rename(metric_)
#
#         da_orn_trial_rdm_concat_blockavg_ = xrsa.qc.compute_trial_rdm_blockavg(
#                 da_orn_trial_rdm_concat_).rename(metric_)
#
#         # add to data vars
#         trial_rdm_data_[metric_] = da_orn_trial_rdm_concat_
#         stim_rdm_data_[metric_] = da_orn_stim_rdm_concat_
#         trial_rdm_blockavg_data_[metric_] = da_orn_trial_rdm_concat_blockavg_
#
#     ds_orn_trial_rdm_concat_ = xr.Dataset(data_vars=trial_rdm_data_, attrs=da_orn_trials_.attrs)
#     ds_orn_stim_rdm_concat_ = xr.Dataset(data_vars=stim_rdm_data_, attrs=da_orn_stim_.attrs)
#     ds_orn_trial_rdm_concat_blockavg_ = xr.Dataset(data_vars=trial_rdm_blockavg_data_,
#                                                    attrs=da_orn_trials_.attrs)
#
#     return ds_orn_trial_rdm_concat_, ds_orn_stim_rdm_concat_, ds_orn_trial_rdm_concat_blockavg_
#
#
# # %%
# # da_orn_reshaped = (da_orn
# #                    .set_index(col=['date', 'fly_num', 'roi'])
# #                    .unstack('col')
# #                    .stack(acq=['date', 'fly_num']))
# # da_orn_reshaped = da_orn_reshaped.where(~da_orn_reshaped.isnull().all(dim=['row', 'roi']),
# #                                         drop=True)
# #
# # da_orn_reshaped = da_orn_reshaped.rename(
# #         {'odor1': 'stim',
# #          'repeat': 'stim_occ',
# #          'date': 'date_imaged',
# #          'row': 'trials',
# #          })
# #
# # # add datefly
# # #############
# # datefly = [f"{a}/{b}" for a, b, in zip(da_orn_reshaped['date_imaged'].to_numpy(),
# #                                        da_orn_reshaped['fly_num'].to_numpy())]
# # da_orn_reshaped = da_orn_reshaped.assign_coords(datefly=('acq', datefly))
#
# # add attrs
# ###########
# da_orn_reshaped.attrs['date_shared'] = date_shared
# da_orn_reshaped.attrs['region'] = 'orn'
# #                          'rois_finalized': ('certain' in date_2_filename[date_shared]) * 1}
# # print(da_orn_reshaped)
#
# # save to netcdf
# #################
# # da_orn_reshaped.reset_index('acq').to_netcdf(data_dir.joinpath('xrda_orn_with_datefly.nc'))
# # %% make individual dataarrays for ORNs
# #########################################
#
# orn_trial_respvecs_by_datefly = []
# orn_stim_respvecs_by_datefly = []
#
# for (imaging_date, fly_num), da in da_orn.set_xindex(['date', 'fly_num']).groupby('col'):
#     print(imaging_date)
#     print(fly_num)
#     print(da)
#
#     da = (da
#           .drop(['date', 'fly_num', 'panel'])
#           .rename({'roi': 'cells'})
#           .rename({'col': 'cells',
#                    'row': 'trials',
#                    'odor1': 'stim',
#                    'repeat': 'occ'}
#                   )
#           .set_xindex('cells')
#           .set_xindex(['stim', 'occ'])
#           )
#
#     da = da.assign_coords(imaging_date=imaging_date,
#                           fly_num=fly_num,
#                           datefly=f"{imaging_date}/{fly_num}"
#                           )
#     da.attrs = dict(imaging_date=imaging_date,
#                     fly_num=fly_num,
#                     datefly=f"{imaging_date}/{fly_num}")
#
#     orn_trial_respvecs_by_datefly.append(da)
#     orn_stim_respvecs_by_datefly.append(da.reset_index('trials').set_xindex('stim')
#                                         .groupby('stim').mean(skipna=True)
#                                         )
# # %% make concatenated RDMs
#
# orn_trial_rdms = [xrsa.rdm.compute_trial_respvec_rdm(item)
#                   for item in orn_trial_respvecs_by_datefly]
#
# da_orn_trial_rdm_concat = xr.concat(orn_trial_rdms, 'acq')
#
# orn_stim_rdms = [xrsa.rdm.compute_rdm(item, input_dim_ord=['stim', 'cells'])
#                  for item in orn_stim_respvecs_by_datefly
#                  ]
#
# da_orn_stim_rdm_concat = xr.concat(orn_stim_rdms, 'acq')
# # %%
# da_orn_trial_rdm_concat.reset_index('trial_row').reset_index('trial_col') \
#     .to_netcdf(data_dir.joinpath('xrda_orn_trial_rdm_concat.nc'))
#
# da_orn_stim_rdm_concat.to_netcdf(data_dir.joinpath('xrda_orn_stim_rdm_concat.nc'))
# # %%
