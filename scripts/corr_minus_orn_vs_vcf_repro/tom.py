"""Functions for making natural mixture data compatible with tom's analysis code.

Required inputs:
- 'pin_odor_mixture_list.json'
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn import preprocessing
import xrsa
import stimuli
import external.tom
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import json
from itertools import combinations

from external.tom.plot import AggregateOrnPlotter

plt.rcParams.update({'pdf.fonttype': 42,
                     'text.usetex': False})

tom_abbrevs = {'kiwi approx.': '~kiwi',
               'ethyl acetate': 'EA',
               'ethyl butyrate': 'EB',
               'isoamyl alcohol': 'IAol',
               'isoamyl acetate': 'IAA',
               'ethanol': 'EtOH',
               '1-octen-3-ol': 'OCT',
               '2-heptanone': '2H',
               'methyl salicylate': 'MS',
               'valeric acid': 'VA',
               'furfural': 'FUR',
               'control mix': 'control mix',
               'paraffin': 'pfo',
               'pfo': 'pfo',
               # 'trans-2-hexenal': 'T2H',
               # '3-methylthio-1-propanol': '3MT1P'
               }

kc_odor_ord = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep', 'aa', 'va', 'B-cit', 'Lin', '6al', 't2h',
               '1-8ol', '1-5ol', '1-6ol', 'benz', 'ms']

kc_stim_ord = [f"{item} @ -3" for item in kc_odor_ord]


# %%

def make_dataframes_from_xlsx(file):
    # file = Path("/local/matrix/Remy-Data/projects/odor_space_collab"
    #             "/analysis_outputs/from_tom/data_from_tom(4).xlsx")

    df_orns_ = pd.read_excel(file, sheet_name='orn_terminals', header=[0, 1, 2],
                             index_col=[0, 1, 2, 3, 4])
    df_orns_.columns.names = ['date', 'fly_num', 'roi']
    df_pns_ = pd.read_excel(file, sheet_name='pn_boutons', header=[0, 1, 2],
                            index_col=[0, 1, 2, 3, 4])
    df_pns_.columns.names = ['date', 'fly_num', 'roi']
    return df_orns_, df_pns_


def df_ori_2_dataarray(df_ori):
    # fixed_index = df_ori.index.to_frame().convert_dtypes()
    # mi_row = pd.MultiIndex.from_frame(fixed_index)
    #
    # fixed_columns = df_ori.columns.to_frame().convert_dtypes()
    # mi_col = pd.MultiIndex.from_frame(fixed_columns)

    da_ori = xr.DataArray(df_ori.to_numpy(),
                          dims=['row', 'col'],
                          coords=dict(
                                  row=df_ori.index,
                                  col=df_ori.columns
                                  ))
    da_ori = da_ori.reset_index('row')
    da_ori = da_ori.reset_index('col')
    return da_ori


def fix_da_ori(da_ori, panel_name):
    da_fixed = da_ori.drop_vars(['is_pair', 'odor2'])

    da_fixed = da_fixed.where(da_fixed.panel == panel_name, drop=True)
    for coord_name in ['panel', 'odor1', 'roi']:
        da_fixed[coord_name] = da_fixed[coord_name].astype(str)
    da_fixed['repeat'] = da_fixed['repeat'].astype('int')
    da_fixed['fly_num'] = da_fixed['fly_num'].astype('int')
    da_fixed['date'] = da_fixed['date'].dt.date
    da_fixed['date'] = da_fixed['date'].astype(str)

    return da_fixed


def convert_xlsx_to_netcdfs(file):
    df_orn_ori, df_pn_ori = make_dataframes_from_xlsx(file)

    da_orn_ori = df_ori_2_dataarray(df_orn_ori)
    da_pn_ori = df_ori_2_dataarray(df_pn_ori)

    da_orn_fixed = fix_da_ori(da_orn_ori)
    da_pn_fixed = fix_da_ori(da_pn_ori)

    da_orn_fixed.name = 'orn_terminals'
    da_pn_fixed.name = 'pn_dendrites'

    # da_orn_fixed.to_netcdf(natmixconfig.NAS_PRJ_DIR.joinpath(
    #         'analysis_outputs',
    #         'by_imaging_type',
    #         'orn_terminals',
    #         'xrds_orn_terminals.nc'
    #         ))
    # da_pn_fixed.to_netcdf(natmixconfig.NAS_PRJ_DIR.joinpath(
    #         'analysis_outputs',
    #         'by_imaging_type',
    #         'pn_dendrites',
    #         'xrds_pn_dendrites.nc'
    #         ))
    return da_orn_fixed, da_pn_fixed


def compute_orn_rdms(da_orn_trials_, da_orn_stim_, metric_list=None):
    if metric_list is None:
        metric_list = ['correlation', 'cosine', 'euclidean']

    trial_rdm_data_ = {}
    stim_rdm_data_ = {}
    trial_rdm_blockavg_data_ = {}

    for metric_ in metric_list:
        da_orn_trial_rdm_concat_ = xr.concat(
                [xrsa.rdm.compute_trial_respvec_rdm(
                        da_.dropna(dim='cells', how='all').dropna(dim='trials', how='all'),
                        metric=metric_)
                    for _, da_ in da_orn_trials_.groupby('datefly')],
                dim='acq').rename(metric_)

        da_orn_stim_rdm_concat_ = xr.concat(
                [xrsa.rdm.compute_rdm(
                    da_.dropna(dim='cells', how='all').dropna(dim='stim', how='all'),
                    input_dim_ord=['stim', 'cells'],
                    metric=metric_)
                 for _, da_ in da_orn_stim_.groupby('datefly')],
                dim='acq').rename(metric_)

        da_orn_trial_rdm_concat_blockavg_ = xrsa.qc.compute_trial_rdm_blockavg(
                da_orn_trial_rdm_concat_).rename(metric_)

        # add to data vars
        trial_rdm_data_[metric_] = da_orn_trial_rdm_concat_
        stim_rdm_data_[metric_] = da_orn_stim_rdm_concat_
        trial_rdm_blockavg_data_[metric_] = da_orn_trial_rdm_concat_blockavg_

    ds_orn_trial_rdm_concat_ = xr.Dataset(data_vars=trial_rdm_data_, attrs=da_orn_trials_.attrs)
    ds_orn_stim_rdm_concat_ = xr.Dataset(data_vars=stim_rdm_data_, attrs=da_orn_stim_.attrs)
    ds_orn_trial_rdm_concat_blockavg_ = xr.Dataset(data_vars=trial_rdm_blockavg_data_,
                                                   attrs=da_orn_trials_.attrs)

    return ds_orn_trial_rdm_concat_, ds_orn_stim_rdm_concat_, ds_orn_trial_rdm_concat_blockavg_


def invert_euc_dist_abs(da_euc, inf_2_nan=True):
    dims = [d for d in da_euc.dims if d != 'acq']
    da_inv = 1 / (da_euc / da_euc.max(dim=dims))

    if inf_2_nan:
        da_inv = da_inv.where(da_inv.map_blocks(np.isfinite))
    return da_inv


def invert_euc_dist_rbf(da_euc, gamma):
    return da_euc.map_blocks(lambda x: np.exp(-1 * x / gamma)),


# %% load Tom's AL data
########################
# da_orn should have dims (row, col)
# coords along row:
#  - panel
#  - odor1
#  - repeat
#
# coords along col:
#  - date
#  - fly_num
#  - roi

# validation2.py panel
########################
date_shared =
# date_shared = "2024-01-12" # finalized roi stats
# date_shared = '2024-01-29'

date_2_filename = {
    '2024-01-12': 'validation2_ij_certain-roi_stats.p',
    '2024-01-29': 'validation2_ij_roi_stats.p',
    }

data_dir = Path(
        "/local/matrix/Remy-Data/projects/odor_space_collab/analysis_outputs/from_tom").joinpath(
        date_shared)
# %%
df_orn_ori = pd.read_pickle(data_dir.joinpath(date_2_filename[date_shared]))
# %%
da_orn_ori = external.tom.convert.df_ori_2_dataarray(df_orn_ori)
da_orn = external.tom.convert.fix_da_ori(da_orn_ori)
u_rois = np.unique(da_orn['roi'])
# %%
da_orn_reshaped = (da_orn
                   .set_index(col=['date', 'fly_num', 'roi'])
                   .unstack('col')
                   .stack(acq=['date', 'fly_num']))
da_orn_reshaped = da_orn_reshaped.where(~da_orn_reshaped.isnull().all(dim=['row', 'roi']),
                                        drop=True)

da_orn_reshaped = da_orn_reshaped.rename(
        {'odor1': 'stim',
         'repeat': 'stim_occ',
         'date': 'date_imaged',
         'row': 'trials',
         })

# fix stimuli
############
abbrev_to_replace = {'1-3ol': '1-prop',
                     'paa': 'PAA'}

stim_list = da_orn['odor1'].to_numpy()
fixed_stim_list = stimuli.fix.fix_stim(stim_list,
                                       abbrev_to_replace=abbrev_to_replace,
                                       conc_as_float=True)
da_orn_reshaped = da_orn_reshaped.assign_coords(
        stim=('trials', fixed_stim_list)
        )

# add datefly
#############
datefly = [f"{a}/{b}" for a, b, in zip(da_orn_reshaped['date_imaged'].to_numpy(),
                                       da_orn_reshaped['fly_num'].to_numpy())]
da_orn_reshaped = da_orn_reshaped.assign_coords(datefly=('acq', datefly))

# add attrs
###########
da_orn_reshaped.attrs = {'date_shared': date_shared,
                         'rois_finalized': ('certain' in date_2_filename[date_shared]) * 1}
print(da_orn_reshaped)

# save to netcdf
#################
da_orn_reshaped.reset_index('acq').to_netcdf(data_dir.joinpath('xrda_orn_with_datefly.nc'))

# # %% fix stimuli
#
# abbrev_to_replace = {'1-3ol': '1-prop',
#                      'paa': 'PAA'}
#
# stim_list = da_orn['odor1'].to_numpy()
# fixed_stim_list = stimuli.fix.fix_stim(stim_list,
#                                        abbrev_to_replace=abbrev_to_replace,
#                                        conc_as_float=True)
# da_orn = da_orn.assign_coords(
#         odor1=('row', fixed_stim_list)
#         )
#
# # rename coords
# da_orn = da_orn.rename({'odor1': 'stim',
#                         'repeat': 'stim_occ',
#                         'date': 'date_imaged',
#                         }
#                        )
#
# # %% combine into a trial x roi dataarray
#
# da_list = []
#
# # for grp, da0 in da_orn.set_xindex(['date_imaged', 'fly_num']).groupby('col'):
# for grp, da0 in da_orn.set_index(col=['date_imaged', 'fly_num']).groupby('col'):
#     datefly = f"{grp[0]}/{grp[1]}"
#     print(datefly)
#     da = da0.copy(deep=True)
#     da = da.drop(['date_imaged', 'fly_num'])
#     da = da.assign_coords(datefly=datefly, date_imaged=grp[0], fly_num=grp[1])
#     da = da.set_index(col='roi').rename(col='roi')
#     da = da.rename(row='trials')
#     # da = da.reindex(roi=u_rois).rename(col='roi').set_index(roi='roi')
#     da_list.append(da)
#
# da_list_aligned = xr.align(*da_list, join='outer')
#
# # %% combine into trial x roi dataarray
#
# da_orn_with_datefly = xr.concat(da_list_aligned, dim='acq')

# %%
da_orn_with_datefly = xr.load_dataarray(data_dir.joinpath('xrda_orn_with_datefly.nc'))

da_orn_with_datefly = da_orn_with_datefly.set_index(trials=['panel', 'stim', 'stim_occ'])
da_orn_with_datefly = da_orn_with_datefly.set_index(acq=['date_imaged', 'fly_num'])

da_orn_trials = da_orn_with_datefly.sel(trials=dict(panel='validation2')).rename(roi='cells')
da_orn_stim = da_orn_trials.groupby('stim').mean(dim='trials')

da_orn_trials.attrs['preprocessing'] = 'None'
da_orn_stim.attrs['preprocessing'] = 'None'

# preprocesses respvec arrays
da_orn_trials_proc = xr.apply_ufunc(preprocessing.maxabs_scale,
                                    da_orn_trials,
                                    input_core_dims=[['trials', 'cells']],
                                    output_core_dims=[['trials', 'cells']],
                                    vectorize=True)
da_orn_trials_proc.attrs['preprocessing'] = 'maxabs_scale'

da_orn_stim_proc = xr.apply_ufunc(preprocessing.maxabs_scale,
                                  da_orn_stim,
                                  input_core_dims=[['stim', 'cells']],
                                  output_core_dims=[['stim', 'cells']],
                                  vectorize=True, )
da_orn_stim_proc.attrs['preprocessing'] = 'maxabs_scale'

# %% save respvecs
data_dir.joinpath('respvec').mkdir(exist_ok=True)

# save unprocessed
da_orn_trials.reset_index('trials').reset_index('acq').to_netcdf(
        data_dir.joinpath('respvec', 'xrda_orn_trials.nc'))

da_orn_stim.reset_index('acq').to_netcdf(
        data_dir.joinpath('respvec', 'xrda_orn_stim.nc'))

# save maxabs_scale
da_orn_trials_proc.reset_index('trials').reset_index('acq').to_netcdf(
        data_dir.joinpath('respvec', 'xrda_orn_trials__maxabs_scale.nc'))

da_orn_stim_proc.reset_index('acq').to_netcdf(
        data_dir.joinpath('respvec', 'xrda_orn_stim__maxabs_scale.nc'))

# %%


ds_orn_trial_rdm_concat, ds_orn_stim_rdm_concat, ds_orn_trial_rdm_concat_blockavg = \
    compute_orn_rdms(da_orn_trials, da_orn_stim)

ds_orn_proc_trial_rdm_concat, ds_orn_proc_stim_rdm_concat, ds_orn_proc_trial_rdm_concat_blockavg = \
    compute_orn_rdms(da_orn_trials_proc, da_orn_stim_proc)
# %% save datasets w/ multiple distance metrics
data_dir.joinpath('rdm').mkdir(exist_ok=True)

# unprocessed
ds_orn_trial_rdm_concat.reset_index('acq').reset_index('trial_row').reset_index(
        'trial_col').to_netcdf(
        data_dir.joinpath('rdm', f'xrds_orn_trial_rdm_concat.nc')
        )

ds_orn_stim_rdm_concat.reset_index('acq').to_netcdf(
        data_dir.joinpath('rdm', f'xrds_orn_stim_rdm_concat.nc')
        )

ds_orn_trial_rdm_concat_blockavg.reset_index('acq').to_netcdf(
        data_dir.joinpath('rdm', f'xrds_orn_trial_rdm_concat_blockavg.nc')
        )

# save processed
ds_orn_proc_trial_rdm_concat.reset_index('acq').reset_index('trial_row').reset_index(
        'trial_col').to_netcdf(
        data_dir.joinpath('rdm',
                          'xrds_orn_trial_rdm_concat__maxabs_scale.nc'
                          )
        )

ds_orn_proc_stim_rdm_concat.reset_index('acq').to_netcdf(
        data_dir.joinpath('rdm', f'xrds_orn_stim_rdm_concat__maxabs_scale.nc')
        )

ds_orn_proc_trial_rdm_concat_blockavg.reset_index('acq').to_netcdf(
        data_dir.joinpath('rdm', f'xrds_orn_trial_rdm_concat_blockavg__maxabs_scale.nc')
        )
# %%
ds_orn_trial_rdm_concat = xr.load_dataset(
        data_dir.joinpath('rdm', 'xrds_orn_trial_rdm_concat.nc'))
ds_orn_trial_rdm_concat = (ds_orn_trial_rdm_concat
                           .set_index(trial_row=['row_stim', 'row_stim_occ'])
                           .set_index(trial_col=['col_stim', 'col_stim_occ'])
                           )

ds_orn_trial_rdm_concat_blockavg = xr.load_dataset(
        data_dir.joinpath('rdm', 'xrds_orn_trial_rdm_concat_blockavg.nc')
        )
ds_orn_stim_rdm_concat = xr.load_dataset(
        data_dir.joinpath('rdm', 'xrds_orn_stim_rdm_concat.nc')
        )

# %% Plot maxabs_scale RDMs

for metric in ['correlation', 'cosine']:
    agg_orn_plotter = AggregateOrnPlotter(
            trial_rsm_concat=1 - ds_orn_proc_trial_rdm_concat,
            stim_rsm_concat=1 - ds_orn_proc_stim_rdm_concat,
            trial_rsm_concat_blockavg=1 - ds_orn_proc_trial_rdm_concat_blockavg,
            metric=metric,
            title_str=f'ORNs (Pebbled), shared {date_shared}'
                      f'\npreprocessing=maxabs_scale, metric={metric}',
            heatmap_kws=dict(cmap='RdBu_r', vmin=-1, vmax=1),
            )

    # plot and save RDMs

    fig_trials = agg_orn_plotter.plot_individual_and_mean_rsms('trial')
    plt.show()

    fig_stim = agg_orn_plotter.plot_individual_and_mean_rsms('stim')
    plt.show()

    fig_blockavg = agg_orn_plotter.plot_individual_and_mean_rsms('blockavg')
    plt.show()

    plot_dir = data_dir.joinpath('plots')
    plot_dir.mkdir(exist_ok=True, parents=True)

    with (PdfPages(plot_dir.joinpath(
            f'orn_RDMs__maxabs_scale__{metric}__mean_and_individual__default_ord.pdf'))
    as pdf):
        pdf.savefig(fig_trials)
        pdf.savefig(fig_stim)
        pdf.savefig(fig_blockavg)

# %% Plot RDMs from unscaled trials/stim

for metric in ['correlation', 'cosine']:
    agg_orn_plotter = AggregateOrnPlotter(
            trial_rsm_concat=1 - ds_orn_trial_rdm_concat,
            stim_rsm_concat=1 - ds_orn_stim_rdm_concat,
            trial_rsm_concat_blockavg=1 - ds_orn_trial_rdm_concat_blockavg,
            metric=metric,
            title_str=f'ORNs (Pebbled), shared {date_shared}'
                      f'\npreprocessing=None, metric={metric}',
            heatmap_kws=dict(cmap='RdBu_r', center=0, robust=True),
            )

    # plot and save RDMs
    fig_trials = agg_orn_plotter.plot_individual_and_mean_rsms('trial')
    plt.show()

    fig_stim = agg_orn_plotter.plot_individual_and_mean_rsms('stim')
    plt.show()

    fig_blockavg = agg_orn_plotter.plot_individual_and_mean_rsms('blockavg')
    plt.show()

    plot_dir = data_dir.joinpath('plots')
    plot_dir.mkdir(exist_ok=True, parents=True)

    with (PdfPages(plot_dir.joinpath(
            f'orn_RDMs__raw__{metric}__mean_and_individual__default_ord.pdf'))
    as pdf):
        pdf.savefig(fig_trials)
        pdf.savefig(fig_stim)
        pdf.savefig(fig_blockavg)

# %% Plot summary RDMs

with (PdfPages(
        plot_dir.joinpath(f'orn_RDMs__raw__summary_mean_std__default_ord.pdf')) as pdf):
    for metric in ['correlation', 'cosine']:
        agg_orn_plotter = AggregateOrnPlotter(
                trial_rsm_concat=1 - ds_orn_trial_rdm_concat,
                stim_rsm_concat=1 - ds_orn_stim_rdm_concat,
                trial_rsm_concat_blockavg=1 - ds_orn_trial_rdm_concat_blockavg,
                metric=metric,
                title_str=f'ORNs (Pebbled), shared {date_shared}'
                          f'\npreprocessing=None, metric={metric}',
                heatmap_kws=dict(cmap='RdBu_r', center=0, robust=True),
                )

        fig_summary, _ = agg_orn_plotter.plot_trial_and_stim_summary_rsms()
        plt.show()

        pdf.savefig(fig_summary)

with (PdfPages(
        plot_dir.joinpath(f'orn_RDMs__maxabs_scale__summary_mean_std__default_ord.pdf')) as pdf):
    for metric in ['correlation', 'cosine']:
        agg_orn_plotter = AggregateOrnPlotter(
                trial_rsm_concat=1 - ds_orn_proc_trial_rdm_concat,
                stim_rsm_concat=1 - ds_orn_proc_stim_rdm_concat,
                trial_rsm_concat_blockavg=1 - ds_orn_proc_trial_rdm_concat_blockavg,
                metric=metric,
                title_str=f'ORNs (Pebbled), shared {date_shared}'
                          f'\npreprocessing=maxabs_scale, metric={metric}',
                heatmap_kws=dict(cmap='RdBu_r', vmin=-1, vmax=1),
                )

        fig_summary, _ = agg_orn_plotter.plot_trial_and_stim_summary_rsms()
        plt.show()

        pdf.savefig(fig_summary)
# %% save mean and ind. plots as pdfs
plot_dir.mkdir(exist_ok=True)

if do_qc_thresh:
    pdf_name = (f"{agg.agg_name}__mean_and_individual_stim_rdms__qcthresh_"
                f"{agg.sim_threshold:0.02f}__default_ord__allstim__vlag.pdf")
else:
    pdf_name = f"{agg.agg_name}__mean_and_individual_stim_rdms__default_ord__allstim__vlag.pdf"

pdf_file = plot_dir.joinpath(pdf_name)

with PdfPages(pdf_file) as pdf:
    pdf.savefig(fig_trials)
    pdf.savefig(fig_stim)
    pdf.savefig(fig_blockavg)

# %% plot trials/stim/blockavg RDMs

fig_trial_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(
        # invert_euc_dist_abs(ds_orn_stim_rdm_concat['correlation']),
        1 - ds_orn_proc_stim_rdm_concat['correlation'],
        row_coord='stim_row',
        col_coord='stim_col',
        title_coord='datefly',
        col_wrap=3,
        heatmap_kws=dict(vmin=-0.5,
                         vmax=1.5,
                         center=0, robust=True, cmap='RdBu_r')
        ,
        )
plt.show()

# %% plot response heatmaps

sns.set(font_scale=0.6)

cmap = 'RdBu_r'
robust = True

with (sns.axes_style('ticks')):
    with PdfPages(data_dir.joinpath('plots',
                                    f'peak_amp_responses_by_fly__{cmap}__robust{robust}.pdf')) as pdf:
        for use_maxabs_scale in [False, True]:

            if use_maxabs_scale:
                da_stim_plot = da_orn_stim_proc
            else:
                da_stim_plot = da_orn_stim

            fig, axarr = plt.subplots(1, da_orn_stim_proc.sizes['acq'], figsize=(11, 8.5))

            for iacq, ax in zip(range(da_stim_plot.sizes['acq']), axarr.flat):
                sns.heatmap(da_stim_plot.isel(acq=iacq).to_pandas().T,
                            cmap=cmap,
                            center=0,
                            robust=robust,
                            square=True,
                            ax=ax,
                            xticklabels=True,
                            yticklabels=True,
                            cbar_kws=dict(shrink=0.5)
                            )
                ax.set_title(da_orn_with_datefly.isel(acq=iacq)['datefly'].item())
                ax.set_facecolor("0.8")

            fig.suptitle(f"Responses by fly (use_maxabs_scale={use_maxabs_scale})")

            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            plt.show()

            pdf.savefig(fig)
# %%
ds_orn_stim_rdm_concat_highconc = xr.load_dataset(
        data_dir.with_name('2024-01-12').joinpath('rdm', 'xrds_orn_stim_rdm_concat.nc')
        )
ds_orn_stim_rdm_concat_highconc = stimuli.split_stim_coord(ds_orn_stim_rdm_concat_highconc,
                                                           ['stim_row', 'stim_col'],
                                                           substr_to_replace='stim')
print(ds_orn_stim_rdm_concat_highconc)
# %%
ds_orn_stim_rdm_concat = stimuli.split_stim_coord(ds_orn_stim_rdm_concat,
                                                  ['stim_row', 'stim_col'],
                                                substr_to_replace='stim')
# %%
tidy_stim_rdm_low = (ds_orn_stim_rdm_concat['correlation'].mean(dim='acq')
                       .set_index(stim_row='abbrev_row')
                       .set_index(stim_col='abbrev_col')
                       .to_pandas()
                       .stack()
                       .rename('low_conc')
                       )
tidy_stim_rdm_high = (ds_orn_stim_rdm_concat_highconc['correlation'].mean(dim='acq')
                       .set_index(stim_row='abbrev_row')
                       .set_index(stim_col='abbrev_col')
                       .to_pandas()
                       .stack()
                       .rename('high_conc')
                       )
tidy_stim_rdm_concs = pd.concat([tidy_stim_rdm_low, tidy_stim_rdm_high], axis=1)
print(tidy_stim_rdm_concs)
# %%

u_abbrev_pairs = list(combinations(ds_orn_stim_rdm_concat_highconc['abbrev_row'].to_numpy().tolist(), 2))
sns.scatterplot(tidy_stim_rdm_concs.loc[u_abbrev_pairs, :], x='low_conc', y='high_conc')
plt.show()

# %% Compute RDMs

# make stim, trial respvecs
da_multifly_orn_trials = (da_orn_with_datefly
                          .where(da_orn_with_datefly['panel'] == 'validation2', drop=True)
                          .reset_index('trials')
                          .drop('panel')
                          .rename(roi='cells')
                          # .set_index(trials=['stim', 'stim_occ'])
                          )

da_multifly_orn_stim = da_multifly_orn_trials.groupby('stim').mean(dim='trials')
da_multifly_orn_trials = da_multifly_orn_trials.set_index(trials=['stim', 'stim_occ'])

orn_trial_rdms = []
orn_stim_rdms = []

# trial RDMs
for iacq, da in da_multifly_orn_trials.groupby('acq'):
    orn_trial_rdms.append(xrsa.rdm.compute_trial_respvec_rdm(da.dropna(dim='cells')))
da_orn_trial_rdm_concat = xr.concat(orn_trial_rdms, 'acq')

# stim RDMs
for iacq, da in da_multifly_orn_stim.groupby('acq'):
    orn_stim_rdms.append(
            xrsa.rdm.compute_rdm(da.dropna(dim='cells'), input_dim_ord=['stim', 'cells']))
da_orn_stim_rdm_concat = xr.concat(orn_stim_rdms, 'acq')

da_orn_trial_rdm_concat_blockavg = xrsa.qc.compute_trial_rdm_blockavg(da_orn_trial_rdm_concat)

# %% sort RDMs (alphabetical, default order)
da_orn_trial_rdm_concat = (da_orn_trial_rdm_concat
                           .sortby('col_stim')
                           .sortby('row_stim'))

da_orn_trial_rdm_concat_blockavg = (da_orn_trial_rdm_concat_blockavg
                                    .sortby('trial_row')
                                    .sortby('trial_col'))

da_orn_stim_rdm_concat = (da_orn_stim_rdm_concat
                          .sortby('stim_row')
                          .sortby('stim_col')
                          )
# %% save RDMs to data folder
#
# files include:
# --------------
#   - xrda_orn_trial_rdm_concat.nc
#   - xrda_orn_trial_rdm_concat_blockavg.nc
#   - xrda_orn_stim_rdm_concat.nc

da_orn_trial_rdm_concat.reset_index('trial_row').reset_index('trial_col') \
    .to_netcdf(data_dir.joinpath('xrda_orn_trial_rdm_concat.nc'))

da_orn_stim_rdm_concat.to_netcdf(data_dir.joinpath('xrda_orn_stim_rdm_concat.nc'))

da_orn_trial_rdm_concat_blockavg.to_netcdf(data_dir.joinpath('da_orn_trial_rdm_concat_blockavg.nc'))

# %% Plot mean and individual RDMs - default order

fig_trial_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(1 - da_orn_trial_rdm_concat,
                                                            title_coord='datefly',
                                                            col_wrap=3,
                                                            fig_size_calculator_kws=dict(
                                                                    top_margin=1),
                                                            heatmap_kws=dict(cmap='viridis', vmin=0,
                                                                             vmax=1)
                                                            )
fig_trial_rdms.suptitle('ORNs (Pebbled)\ndata shared {date_shared}')
plt.tight_layout()
plt.show()

fig_stim_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(1 - da_orn_stim_rdm_concat,
                                                           row_coord='stim_row',
                                                           col_coord='stim_col',
                                                           title_coord='datefly',
                                                           col_wrap=3,
                                                           fig_size_calculator_kws=dict(
                                                                   top_margin=1),
                                                           heatmap_kws=dict(cmap='viridis', vmin=0,
                                                                            vmax=1)
                                                           )
fig_stim_rdms.suptitle(f'ORNs (Pebbled)\ndata shared {date_shared}')
plt.tight_layout()
plt.show()

fig_blockavg_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(1 - da_orn_trial_rdm_concat_blockavg,
                                                               row_coord='trial_row',
                                                               col_coord='trial_col',
                                                               title_coord='datefly',
                                                               col_wrap=3,
                                                               fig_size_calculator_kws=dict(
                                                                       top_margin=1),
                                                               heatmap_kws=dict(cmap='viridis',
                                                                                vmin=0, vmax=1)
                                                               )
fig_blockavg_rdms.suptitle(f'ORNs (Pebbled)\ndata shared {date_shared}')
plt.tight_layout()
plt.show()
# %%
da_orn_trial_rdm_concat_blockavg.get_index()
# %%
# save to PDF
# with PdfPages(data_dir.joinpath('plots', 'orn_RDMs__mean_and_individual__default_ord.pdf')) as
# pdf:
with PdfPages(
        data_dir.joinpath('plots',
                          'orn_RDMs__mean_and_individual__default_ord__20240122.pdf')) as pdf:
    pdf.savefig(fig_trial_rdms)
    pdf.savefig(fig_stim_rdms)
    pdf.savefig(fig_blockavg_rdms)
# %%

fig_trial_rdms.savefig(
        data_dir.joinpath('plots', 'orn_RDMs__mean_and_individual__default_ord__trials.png'))
fig_stim_rdms.savefig(
        data_dir.joinpath('plots', 'orn_RDMs__mean_and_individual__default_ord__stim.png'))
fig_blockavg_rdms.savefig(
        data_dir.joinpath('plots', 'orn_RDMs__mean_and_individual__default_ord__blockavg.png'))
# %% Compute stimulus cluster order

da_mean_stim_rdm = da_orn_stim_rdm_concat.mean(dim='acq')

# compute distance matrix
D = da_mean_stim_rdm.to_numpy()
D_condensed = squareform(D, force='tovector')

# compute linkage, leaf orders
orn_ord = {}
default_ord = da_mean_stim_rdm['stim_row'].to_numpy()
for method in ['complete', 'average', 'single']:
    Z_stim = hierarchy.linkage(D_condensed, method=method, optimal_ordering=True)
    leaf_ord = hierarchy.leaves_list(Z_stim)
    stim_ord = default_ord[leaf_ord].tolist()
    orn_ord[method] = stim_ord

# save to json file
with open(data_dir.joinpath('orn_ord.json'), 'w') as f:
    json.dump(orn_ord, f, indent=4)
# %%
with PdfPages(data_dir.joinpath('plots', 'orn_RDMs__mean_and_individual__clust_ord.pdf')) as pdf:
    for method in ['complete', 'average', 'single']:
        # reorder RDMs
        trial_rdm_concat_sorted = xrsa.rdm.sort_trial_rdm_by_stim_ord(da_orn_trial_rdm_concat,
                                                                      orn_ord[method])
        stim_rdm_concat_sorted = xrsa.rdm.sort_stim_rdm_by_stim_ord(da_orn_stim_rdm_concat,
                                                                    orn_ord[method])

        trial_rdm_concat_blockavg_sorted = xrsa.rdm.sort_stim_rdm_by_stim_ord(
                da_orn_trial_rdm_concat_blockavg,
                orn_ord[method],
                row_coord='trial_row',
                col_coord='trial_col')

        # plot trial RDMs
        ###################
        fig_trial_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(1 - trial_rdm_concat_sorted,
                                                                    title_coord='datefly')
        fig_trial_rdms.suptitle('ORNs (Pebbled)\ndata shared {date_shared}'
                                f'\nmethod={method}')
        plt.show()

        # plot stim RDMs
        #################
        fig_stim_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(1 - stim_rdm_concat_sorted,
                                                                   row_coord='stim_row',
                                                                   col_coord='stim_col',
                                                                   title_coord='datefly'
                                                                   )
        fig_stim_rdms.suptitle(f'ORNs (Pebbled)\ndata shared {date_shared}'
                               f'\nmethod={method}')
        plt.show()

        # plot blockavg RDMs
        ######################
        fig_blockavg_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(
                1 - trial_rdm_concat_blockavg_sorted,
                row_coord='trial_row',
                col_coord='trial_col',
                title_coord='datefly',
                )
        fig_blockavg_rdms.suptitle(f'ORNs (Pebbled)\ndata shared {date_shared}'
                                   f'\nmethod={method}')
        plt.show()

        # save to pdf
        pdf.savefig(fig_trial_rdms)
        pdf.savefig(fig_stim_rdms)
        pdf.savefig(fig_blockavg_rdms)

######
# END
######
# %%
# orn_stim_rdms = [xrsa.rdm.compute_rdm(item, input_dim_ord=['stim', 'cells'])
#                  for item in orn_stim_respvecs_by_datefly
#                  ]
#
# da_orn_stim_rdm_concat = xr.concat(orn_stim_rdms, 'acq')

# %% compute validation2 heatmaps

da_orn_with_datefly.sel(panel='validation2').drop('panel').rename(roi='cells')

# %%
sns.clustermap(df_plot.groupby('stim').mean(),
               cmap='magma',
               # center=0,
               robust=True,
               metric='correlation',
               method='average'
               )
# %%
da_orn_reshaped = (da_orn_with_datefly
                   .set_index(col=['datefly', 'roi'])
                   .unstack('col')
                   # .stack(acq=['date_imaged', 'fly_num'])
                   )

# %%
# average across repeats
import seaborn as sns

df_orn_stim = (df_orn_ori
               .droplevel(['is_pair', 'odor2'])
               .query("panel=='validation2.py'")
               .groupby('odor1')
               .mean()
               )
for (date_imaged, fly_num), df in df_orn_stim.T.groupby(['date', 'fly_num']):
    print(df)
    sns.clustermap(df.droplevel(['date', 'fly_num']),
                   square=True,
                   cmap='vlag',
                   center=0,
                   standard_scale=1,
                   robust=False
                   )
    # plt.tight_layout()
    plt.show()
# %%
panel_name = 'validation2.py'

data_dir = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
                "analysis_outputs/from_tom/2023-11-28/")

df_orn_ori = pd.read_pickle("/local/matrix/Remy-Data/projects/odor_space_collab/"
                            "analysis_outputs/from_tom/2023-11-28/ij_roi_stats.p")
da_orn = fix_da_ori(df_ori_2_dataarray(df_orn_ori), panel_name=panel_name)
# %%
# da_orn, da_pn = convert_xlsx_to_netcdfs(al_file)
# df_orn_ori = pd.read_pickle("/local/matrix/Remy-Data/projects/odor_space_collab/"
#                             "analysis_outputs/from_tom/pebbled_ij_roi_stats.p")

df_orn_ori = pd.read_pickle("/local/matrix/Remy-Data/projects/odor_space_collab/"
                            "analysis_outputs/from_tom/2023-10-29/pebbled_ij_certain-roi_stats.p")
da_orn = fix_da_ori(df_ori_2_dataarray(df_orn_ori), panel_name)

# %%
df_pn_ori = pd.read_pickle("/local/matrix/Remy-Data/projects/odor_space_collab/"
                           "analysis_outputs/from_tom/2023-10-29/GH146_ij_certain-roi_stats.p")
da_pn = fix_da_ori(df_ori_2_dataarray(df_pn_ori))

# %% make individual dataarrays for ORNs
#########################################

orn_trial_respvecs_by_datefly = []
orn_stim_respvecs_by_datefly = []

for (imaging_date, fly_num), da in da_orn.set_xindex(['date', 'fly_num']).groupby('col'):
    print(imaging_date)
    print(fly_num)
    print(da)

    da = (da
          .drop(['date', 'fly_num', 'panel'])
          .rename({'roi': 'cells'})
          .rename({'col': 'cells',
                   'row': 'trials',
                   'odor1': 'stim',
                   'repeat': 'occ'}
                  )
          .set_xindex('cells')
          .set_xindex(['stim', 'occ'])
          )

    da = da.assign_coords(imaging_date=imaging_date,
                          fly_num=fly_num,
                          datefly=f"{imaging_date}/{fly_num}"
                          )
    da.attrs = dict(imaging_date=imaging_date,
                    fly_num=fly_num,
                    datefly=f"{imaging_date}/{fly_num}")

    orn_trial_respvecs_by_datefly.append(da)
    orn_stim_respvecs_by_datefly.append(da.reset_index('trials').set_xindex('stim')
                                        .groupby('stim').mean(skipna=True)
                                        )
# %% make concatenated RDMs

orn_trial_rdms = [xrsa.rdm.compute_trial_respvec_rdm(item)
                  for item in orn_trial_respvecs_by_datefly]

da_orn_trial_rdm_concat = xr.concat(orn_trial_rdms, 'acq')

orn_stim_rdms = [xrsa.rdm.compute_rdm(item, input_dim_ord=['stim', 'cells'])
                 for item in orn_stim_respvecs_by_datefly
                 ]

da_orn_stim_rdm_concat = xr.concat(orn_stim_rdms, 'acq')
# %%
da_orn_trial_rdm_concat.reset_index('trial_row').reset_index('trial_col') \
    .to_netcdf(data_dir.joinpath('xrda_orn_trial_rdm_concat.nc'))

da_orn_stim_rdm_concat.to_netcdf(data_dir.joinpath('xrda_orn_stim_rdm_concat.nc'))
# %% pick stim ord
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

Z_odor = hierarchy.linkage(squareform(da_orn_stim_rdm_concat.mean(dim='acq'), force='tovector'),
                           metric='correlation',
                           method='average')
leaf_ord_odor = hierarchy.leaves_list(Z_odor)
stim_clust_ord = da_orn_stim_rdm_concat['stim_row'].to_numpy()[leaf_ord_odor].tolist()
# %%
da_orn_trial_rdm_concat = xr.load_dataarray("/local/matrix/Remy-Data/projects/odor_space_collab/"
                                            "analysis_outputs/from_tom/2023-11-28/xrda_orn_trial_rdm_concat.nc")
da_orn_stim_rdm_concat = xr.load_dataarray("/local/matrix/Remy-Data/projects/odor_space_collab/"
                                           "analysis_outputs/from_tom/2023-11-28/xrda_orn_stim_rdm_concat.nc")

stim_to_replace = {'1-3ol @ -3': '1-prop @ -3',
                   'paa @ -4': 'PAA @ -4'}
stim_row = da_orn_stim_rdm_concat['stim_row'].to_numpy().tolist()
stim_col = da_orn_stim_rdm_concat['stim_col'].to_numpy().tolist()

trial_stim_row = da_orn_trial_rdm_concat['row_stim'].to_numpy()
trial_stim_col = da_orn_trial_rdm_concat['col_stim'].to_numpy()

for k, v in stim_to_replace.items():
    stim_row[stim_row.index(k)] = v
    stim_col[stim_col.index(k)] = v

    trial_stim_row[trial_stim_row == k] = v
    trial_stim_col[trial_stim_col == k] = v

da_orn_stim_rdm_concat['stim_row'] = stim_row
da_orn_stim_rdm_concat['stim_col'] = stim_col

da_orn_trial_rdm_concat['row_stim'] = ('trial_row', trial_stim_row)
da_orn_trial_rdm_concat['col_stim'] = ('trial_col', trial_stim_col)

da_orn_trial_rdm_concat = da_orn_trial_rdm_concat.set_xindex(['row_stim', 'row_occ']).set_xindex([
    'col_stim',
    'col_occ'])
# %%

ecfp_ord = ['+pul',
            'menth',
            'long',
            'sab',
            '-bCar',
            '-aPine',
            'euc',
            '2-mib',
            'geos',
            'guai',
            'mchav',
            'PEA',
            'PAA',
            'bbenz',
            'B-myr',
            'ger',
            '1-prop',
            '1p3one',
            '1o3one',
            'EtOct',
            '2-mba',
            'Z4-7al']

ecfp_stim_ord = ['+pul @ -2',
                 'menth @ -3',
                 'long @ -2',
                 'sab @ -3',
                 '-bCar @ -3',
                 '-aPine @ -1.5',
                 'euc @ -1.75',
                 '2-mib @ -4',
                 'geos @ -4',
                 'guai @ -3',
                 'mchav @ -3',
                 'PEA @ -4',
                 'PAA @ -4',
                 'bbenz @ -2',
                 'B-myr @ -2',
                 'ger @ -2',
                 '1-prop @ -3',
                 '1p3one @ -5',
                 '1o3one @ -4',
                 'EtOct @ -2',
                 '2-mba @ -3',
                 'Z4-7al @ -4']
# %%
stim_parts = {}
for item in trial_stim_row:
    abbrev, conc = item.split(" @ ")
    stim_parts[abbrev] = conc
# %%
ecfp_stim_ord = [f"{item} @ {stim_parts[item]}" for item in ecfp_ord]
# %%
trial_stim_row = da_orn_trial_rdm_concat['row_stim'].to_numpy()
trial_stim_col = da_orn_trial_rdm_concat['col_stim'].to_numpy()

trial_abbrev_row = [item.split(" @ ")[0] for item in trial_stim_row]
trial_abbrev_col = [item.split(" @ ")[0] for item in trial_stim_col]

stim_row = da_orn_stim_rdm_concat['stim_row'].to_numpy()
stim_col = da_orn_stim_rdm_concat['stim_col'].to_numpy()

abbrev_row = [item.split(" @ ")[0] for item in stim_row]
abbrev_col = [item.split(" @ ")[0] for item in stim_col]
# %%
da_orn_trial_rdm_concat = da_orn_trial_rdm_concat.assign_coords(
        row_abbrev=('row_stim', trial_abbrev_row),
        col_abbrev=('row_stim', trial_abbrev_col),

        )

da_orn_stim_rdm_concat = da_orn_stim_rdm_concat.assign_coords(
        abbrev_row=('stim_row', abbrev_row),
        abbrev_col=('stim_col', abbrev_col),

        )
# %%
trial_idx = [trial_abbrev_row.index(item) for item in da_orn_trial_rdm_concat[
    'row_abbrev'].to_numpy()]

# %%
kc_odor_ord = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep', 'aa', 'va', 'B-cit', 'Lin', '6al', 't2h',
               '1-8ol', '1-5ol', '1-6ol', 'benz', 'ms']

kc_stim_ord = [f"{item} @ -3" for item in kc_odor_ord]

# %% force order - trial RDMs
da_orn_trial_rdm_concat = (da_orn_trial_rdm_concat
                           .reset_index('trial_row')
                           .reset_index('trial_col')
                           .rename({'row_occ': 'row_stim_occ',
                                    'col_occ': 'col_stim_occ'})
                           )
da_orn_trial_rdm_concat = xrsa.rdm.sort_trial_rdm_by_stim_ord(da_orn_trial_rdm_concat,
                                                              stim_ord=ecfp_stim_ord)
da_orn_trial_rdm_concat = (da_orn_trial_rdm_concat
                           .set_xindex(['row_stim', 'row_stim_occ'])
                           .set_xindex(['col_stim', 'col_stim_occ'])
                           )
# %% force order - stim RDMs
da_orn_stim_rdm_concat = xrsa.rdm.sort_stim_rdm_by_stim_ord(da_orn_stim_rdm_concat,
                                                            stim_ord=ecfp_stim_ord)
# %% plot KC ordered RDMs

fig_trial_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(1 - da_orn_trial_rdm_concat)
fig_trial_rdms.suptitle('ORNs (Pebbled)')
plt.show()

fig_stim_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(1 - da_orn_stim_rdm_concat,
                                                           row_coord='stim_row',
                                                           col_coord='stim_col',
                                                           )
fig_stim_rdms.suptitle('ORNs (Pebbled)')
plt.show()
# %%
save_folder = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
                   "analysis_outputs/from_tom/2023-11-28")

with PdfPages(save_folder.joinpath('orn_RDMs__mean_and_individual__ecfp_ord.pdf')) as pdf:
    pdf.savefig(fig_trial_rdms)
    pdf.savefig(fig_stim_rdms)

# %% make individual dataarrays for PNs
#########################################

pn_trial_respvecs_by_datefly = []
pn_stim_respvecs_by_datefly = []

for (imaging_date, fly_num), da in da_pn.set_xindex(['date', 'fly_num']).groupby('col'):
    print(imaging_date)
    print(fly_num)
    print(da)

    da = (da
          .drop(['date', 'fly_num', 'panel'])
          .rename({'roi': 'cells'})
          .rename({'col': 'cells',
                   'row': 'trials',
                   'odor1': 'stim',
                   'repeat': 'occ'}
                  )
          .set_xindex('cells')
          .set_xindex(['stim', 'occ'])
          )

    da = da.assign_coords(imaging_date=imaging_date,
                          fly_num=fly_num,
                          datefly=f"{imaging_date}/{fly_num}"
                          )
    da.attrs = dict(imaging_date=imaging_date,
                    fly_num=fly_num,
                    datefly=f"{imaging_date}/{fly_num}")

    pn_trial_respvecs_by_datefly.append(da)
    pn_stim_respvecs_by_datefly.append(da.reset_index('trials').set_xindex('stim')
                                       .groupby('stim').mean(skipna=True)
                                       )
# %% make concatenated RDMs

pn_trial_rdms = [xrsa.rdm.compute_trial_respvec_rdm(item)
                 for item in pn_trial_respvecs_by_datefly]

pn_stim_rdms = [xrsa.rdm.compute_rdm(item, input_dim_ord=['stim', 'cells'])
                for item in pn_stim_respvecs_by_datefly
                ]

da_pn_trial_rdm_concat = xr.concat(pn_trial_rdms, 'acq')

da_pn_stim_rdm_concat = xr.concat(pn_stim_rdms, 'acq')
# %%

kc_odor_ord = ['2h', 'IaA', 'pa', '2-but', 'eb', 'ep', 'aa', 'va', 'B-cit', 'Lin', '6al', 't2h',
               '1-8ol', '1-5ol', '1-6ol', 'benz', 'ms']

kc_stim_ord = [f"{item} @ -3" for item in kc_odor_ord]

# %% force order - trial RDMs
da_pn_trial_rdm_concat = (da_pn_trial_rdm_concat
                          .reset_index('trial_row')
                          .reset_index('trial_col')
                          .rename({'row_occ': 'row_stim_occ',
                                   'col_occ': 'col_stim_occ'})
                          )
da_pn_trial_rdm_concat = xrsa.rdm.sort_trial_rdm_by_stim_ord(da_pn_trial_rdm_concat,
                                                             stim_ord=kc_stim_ord)

da_pn_trial_rdm_concat = (da_pn_trial_rdm_concat
                          .set_xindex(['row_stim', 'row_stim_occ'])
                          .set_xindex(['col_stim', 'col_stim_occ'])
                          )
# %% force order - stim RDMs
da_pn_stim_rdm_concat = xrsa.rdm.sort_stim_rdm_by_stim_ord(da_pn_stim_rdm_concat,
                                                           stim_ord=kc_stim_ord)
# %% plot KC ordered RDMs

fig_pn_trial_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(1 - da_pn_trial_rdm_concat)
fig_pn_trial_rdms.suptitle('PNs (GH146)')
plt.show()

fig_pn_stim_rdms = xrsa.vis.rdm.plot_individual_and_mean_rdms(1 - da_pn_stim_rdm_concat,
                                                              row_coord='stim_row',
                                                              col_coord='stim_col',
                                                              )
fig_pn_stim_rdms.suptitle('PNs (GH146)')
plt.show()
# %%
save_folder = Path("/local/matrix/Remy-Data/projects/odor_space_collab/"
                   "analysis_outputs/from_tom/2023-10-29")

with PdfPages(save_folder.joinpath('pn_RDMs__mean_and_individual.pdf')) as pdf:
    pdf.savefig(fig_pn_stim_rdms)
    pdf.savefig(fig_pn_stim_rdms)
# %%
