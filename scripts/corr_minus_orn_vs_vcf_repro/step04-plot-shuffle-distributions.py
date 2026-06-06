#!/usr/bin/env python3
"""Plot the results of resampled space correlations

Plot types:
==========
- pointplots: plot the neural spaces (cols) against chemical/natural spaces (rows)
    - For viewing multiple space cols against multiple space rows.
- split_violins: plot chemical/natural space corrs along neural spaces (rows) as split violins.
    - i.e. different neural cols are arranged as rows. For each neural space, the chemical and
    natural source space corr. distributions are shown as split violins.
    - Space cols: multiple
    - Space rows: up to 2
- diff_violins: plot the distribution of differences for 2 reference spaces, with ci and mean.
    - i.e. Violin of corr(ref_space_f, space_col) - corr(ref_space_i, space_col)
"""
from itertools import product
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import yaml
from attrs import define, field

# %%
plt.style.reload_library()
plt.style.use('remy-default')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# set seaborn style
sns.axes_style('ticks')
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

# Set the matplotlib font sizes
plt.rcParams.update({
    'figure.titlesize': 12,
    'axes.titlesize': 10,  # Title font size
    'axes.labelsize': 10,  # Axis label font size
    'xtick.labelsize': 8,  # X-tick label font size
    'ytick.labelsize': 8  # Y-tick label font size
    })

# TODO do i even have all these? matter? how to install it matters, and i don't have?
# Add the preferred fonts
font_files = ['/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf',
              '/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf',
              '/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf',
              '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf',
              '/usr/share/fonts/truetype/msttcorefonts/Arial_Black.ttf']

for item in font_files:
    fm.fontManager.addfont("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
plt.rcParams['font.sans-serif'] = ['Arial', 'Verdana', 'DejaVu Sans']

# set the default marker style
marker_style = dict(marker='o', markersize=10, markeredgewidth=0)

hues = dict(chem='#1b9e77',  # green
            vcf='#7570b3',  # purple
            )

data_folder = Path("/home/remy/PycharmProjects/OdorSpaceShare/manuscript/data/"
                   "figure-04/04cde")

figure_folder = Path("/home/remy/PycharmProjects/OdorSpaceShare/manuscript/figures/"
                     "figure-04/04cde")


# TODO refactor to share w/ other definitions of these in other scripts
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
    return Path("/home/remy/PycharmProjects/OdorSpaceShare/manuscript/data/figure-04/04cde/"
                "all_odor_spaces/resampled3/"
                f"{'with_replacement' if with_replacement else 'no_replacement'}/"
                f"{n_odor_pairs:d}_odor_pairs")


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
    return pd.read_parquet(get_resampling_file(n_iter, n_odor_pairs, with_replacement,
                                               filetype='pandas'))


def load_summary_stats(n_iter: int, n_odor_pairs: int, with_replacement: bool) -> pd.DataFrame:
    return pd.read_pickle(get_summary_file(n_iter, n_odor_pairs, with_replacement))


def compute_resampled_space_corr_diffs(df_resampled: pd.DataFrame,
                                       diff_space_info: List[Tuple],
                                       space_metric) -> pd.DataFrame:
    """Return a dataframe of resampled space corr differences.
    Args:
        df_resampled (pd.DataFrame): resampling data
        diff_space_info (List[tuple]): List of (ref_space_i, ref_space_f, final_space_name)
            tuples.
            ref_space_i (str): initial space name
            ref_space_f (str): final space name
            final_space_name (str): what to name the new column (ref_space_f - ref_space_i).
        space_metric (str): 'pearson' or 'spearman'
    Returns:
        df_resampled_diffs (pd.DataFrame):
            Columns: All the `final_space_name` values in diff_space_mapping.
            Index names: ['resampling_fraction', 'iter', 'space_row']
    """
    diffs = []

    for space_i, space_f, space_diff_col in diff_space_info:
        plot_cols = [space_i, space_f]

        # Pearson
        # -------
        df_diff = (df_resampled
                   .query("space_col in @plot_cols")[space_metric]
                   .unstack('space_col'))
        diffs.append((df_diff[space_f] - df_diff[space_i])
                     .rename(space_diff_col)
                     )

    df_diffs = pd.concat(diffs, axis=1)
    return df_diffs


@define
class ResamplingLoader:
    """Load resampled space correlations and/or summary statistics."""
    n_iter: int
    n_odor_pairs: int
    with_replacement: bool

    def load_resampling(self):
        return load_resampling(self.n_iter, self.n_odor_pairs, self.with_replacement)

    def load_summary_stats(self):
        return load_summary_stats(self.n_iter, self.n_odor_pairs, self.with_replacement)


# ['new_pn_boutons_zscore', 'new_kc_claws_zscore', 'new_kc_soma_nls_zscore']


# %%
n_odor_pairs = 136
n_iter = 5000
with_replacement = True

rs_loader = ResamplingLoader(n_iter, n_odor_pairs, with_replacement)

df_resampled = rs_loader.load_resampling()


# df_pearson_diffs = compute_resampled_space_corr_diffs(df_resampled,
#                                                       diff_space_info,
#                                                       'pearson')


# %%
@define
class ResampledDiffPlotter:
    n_iter: int
    n_odor_pairs: int
    with_replacement: bool
    diff_space_info: List[Tuple]
    df_resampled: pd.DataFrame = field(init=False, default=None)
    df_pearson_diff: pd.DataFrame = field(init=False, default=None)
    df_spearman_diff: pd.DataFrame = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.df_resampled = load_resampling(self.n_iter, self.n_odor_pairs, self.with_replacement)

        self.df_pearson_diff = compute_resampled_space_corr_diffs(self.df_resampled,
                                                                  self.diff_space_info,
                                                                  'pearson')

        self.df_spearman_diff = compute_resampled_space_corr_diffs(self.df_resampled,
                                                                   self.diff_space_info,
                                                                   'spearman')

    def plot(self, resampling_fraction, plot_rows, diff_col_to_plot, space_metric, ci=90):
        if space_metric == 'pearson':
            df_diffs = self.df_pearson_diff
        elif space_metric == 'spearman':
            df_diffs = self.df_spearman_diff

        df_plot_diff = (df_diffs.xs(resampling_fraction,
                                    level='resampling_fraction')
                        .query("space_row in @plot_rows")
                        .reset_index()
                        )

        fig_, ax_ = plt.subplots(1, 1, figsize=(6, 4), dpi=200)

        sns.violinplot(data=df_plot_diff,
                       y="space_row",
                       x=diff_col_to_plot,
                       order=plot_rows,
                       # hue="space_col",
                       linewidth=0,
                       split=True,
                       gap=0.1,
                       ax=ax_,
                       alpha=0.5,
                       color='.2'
                       )

        boxplot_alpha = 1

        sns.boxplot(data=df_plot_diff,
                    y="space_row",
                    x=diff_col_to_plot,
                    # linewidth=0,
                    # inner='box',
                    # split=True,
                    gap=0.1,
                    width=0.1,
                    ax=ax_,
                    whis=(((100 - ci) / 2), 100 - ((100 - ci) / 2)),
                    boxprops=dict(alpha=boxplot_alpha),
                    capprops=dict(alpha=boxplot_alpha),
                    whiskerprops=dict(alpha=boxplot_alpha, linewidth=1),
                    medianprops=dict(alpha=boxplot_alpha),
                    showfliers=False,
                    showcaps=False,
                    meanline=True,
                    # color='0.7',
                    # inner_kws=dict(box_width=5, whis_width=1, color=".5", whis=(2.5, 97.5)),
                    )

        ax_.axvline(0, ls='--', color='0.4', lw=1, zorder=-1)

        for artist in ax_.get_children():
            if hasattr(artist, "set_clip_on"):
                artist.set_clip_on(False)

        return fig_, ax_


# %%
plot_rows = (['chem_ecfp',
              'chem_fcfp',
              'chem_rdkit',
              'chem_pattern',
              # 'uniform_tom',
              # 'hemibrain_wd20_tom'
              ] +
             ['vcf5q_ND030_D03_seed08', ]
             )

diff_space_info = [
    ('orn_remy', 'pn_dendrites-correlation', 'PN dendrites - ORN'),
    ('orn_remy', 'pn_boutons_F_zscore', 'PN boutons - ORN'),
    ('orn_remy', 'new_pn_boutons_zscore', 'PN boutons (new zscored) - ORN'),
    ('orn_remy', 'new_pn_boutons_dff', 'PN boutons (new dF/F) - ORN'),
    ('orn_remy', 'kc_claws_Fc_zscore', 'KC claws - ORN'),
    ('orn_remy', 'new_kc_claws_zscore', 'KC claws (new zscored) - ORN'),
    ('orn_remy', 'new_kc_claws_dff', 'KC claws (new dF/F) - ORN'),
    ('orn_remy', 'kc_remy_combo', 'KC - ORN'),
    ('orn_remy', 'new_kc_soma_nls_zscore', 'KC soma (new zscored) - ORN'),
    ('orn_remy', 'new_kc_soma_nls_dff', 'KC soma (new dF/F) - ORN'),
    ('orn_remy', 'uniform_tom', 'uniform - ORN'),
    ('orn_remy', 'hemibrain_wd20_tom', 'hemibrain_wd20 - ORN'),
    ]
resampled_diff_plotter = ResampledDiffPlotter(n_iter, n_odor_pairs,
                                              with_replacement=with_replacement,
                                              diff_space_info=diff_space_info,
                                              )
# %%
pearson_diff_figs = []

for diff_col in [
    'PN boutons - ORN',
    'PN boutons (new zscored) - ORN',
    'PN boutons (new dF/F) - ORN',
    'KC claws - ORN',
    'KC claws (new zscored) - ORN',
    'KC claws (new dF/F) - ORN',
    'KC - ORN',
    'KC soma (new zscored) - ORN',
    'KC soma (new dF/F) - ORN',
    'uniform - ORN',
    'hemibrain_wd20 - ORN'
    ]:
    fig, ax = resampled_diff_plotter.plot(1.0,
                                          plot_rows,
                                          diff_col,
                                          'pearson',
                                          ci=90)
    plt.xlim(-0.57, 0.57)
    ax.set_title('pearson', fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.show()
    pearson_diff_figs.append(fig)
# %%
pdf_filename = (f"resampled_split_violins_"
                f"{n_odor_pairs}_iter{n_iter:05d}_"
                f"{'with_replacement' if with_replacement else 'no_replacement'}"
                f"_diff__pearson__models.pdf")

with PdfPages(figure_folder / f"resampled3/{pdf_filename}") as pdf:
    for fig in pearson_diff_figs:
        pdf.savefig(fig, bbox_inches='tight')
# %%
spearman_diff_figs = []

for diff_col in [
    'PN boutons - ORN',
    'PN boutons (new zscored) - ORN',
    'PN boutons (new dF/F) - ORN',
    'KC claws - ORN',
    'KC claws (new zscored) - ORN',
    'KC claws (new dF/F) - ORN',
    'KC - ORN',
    'KC soma (new zscored) - ORN',
    'KC soma (new dF/F) - ORN',
    'uniform - ORN',
    'hemibrain_wd20 - ORN'
    ]:
    fig, ax = resampled_diff_plotter.plot(1.0,
                                          plot_rows,
                                          diff_col,
                                          'spearman',
                                          ci=95)
    plt.xlim(-0.57, 0.57)
    ax.set_title('spearman', fontsize=10)
    plt.tight_layout()
    sns.despine()
    plt.show()
    spearman_diff_figs.append(fig)
# %%
pdf_filename = (f"resampled_split_violins_"
                f"{n_odor_pairs}_iter{n_iter:05d}_"
                f"{'with_replacement' if with_replacement else 'no_replacement'}"
                f"_diff__spearman__models.pdf")

with PdfPages(figure_folder / f"resampled3/{pdf_filename}") as pdf:
    for fig in spearman_diff_figs:
        pdf.savefig(fig, bbox_inches='tight')
# %%
diff_col_to_plot = 'PN boutons (new zscored) - ORN'
ci = 95

df_plot_diff = (df_pearson_diffs.xs(1.0, level='resampling_fraction')
                .query("space_row in @plot_rows").reset_index())

sns.despine()
plt.tight_layout()
plt.show()

# %%
with open(data_folder / "all_odor_spaces" / "column_groups.yaml", 'r') as f:
    column_groups = yaml.safe_load(f)

column_groups['vcf5a_cols'] = ['vcf5a_emb_seed01',
                               'vcf5a_emb_seed02',
                               'vcf5a_emb_seed03',
                               'vcf5a_emb_seed04',
                               'vcf5a_emb_seed05',
                               'vcf5a_emb_seed06',
                               'vcf5a_emb_seed07',
                               'vcf5a_emb_seed08',
                               'vcf5a_emb_seed09',
                               'vcf5a_emb_seed10']

column_groups['vcf5b_cols'] = ['vcf5b_emb_seed01',
                               'vcf5b_emb_seed02',
                               'vcf5b_emb_seed03',
                               'vcf5b_emb_seed04',
                               'vcf5b_emb_seed05',
                               'vcf5b_emb_seed06',
                               'vcf5b_emb_seed07',
                               'vcf5b_emb_seed08',
                               'vcf5b_emb_seed09',
                               'vcf5b_emb_seed10', ]

all_columns = [col for grp in column_groups.values() for col in grp] + ['vcf5q_dat']

resampling_cols_128 = (column_groups['chem_cols'] +
                       # ['orn_remy', 'kc_remy_combo'] +
                       ['orn_remy',
                        'pn_dendrites-correlation',
                        'pn_boutons_F_zscore',
                        'kc_claws_Fc_zscore',
                        'kc_remy_combo',
                        'uniform_tom',
                        'hemibrain_wd20_tom'
                        ] +
                       column_groups['model_cols'] +
                       column_groups['vcf5a_cols'] +
                       column_groups['vcf5b_cols'] +
                       column_groups['vcf5q_D03_cols'] +
                       ['vcf5q_dat']
                       )

resampling_cols_136 = (column_groups['chem_cols'] +
                       ['orn_remy',
                        'pn_dendrites-correlation',
                        'pn_boutons_F_zscore',
                        'kc_claws_Fc_zscore',
                        'kc_remy_combo',
                        'uniform_tom',
                        'hemibrain_wd20_tom'
                        ] +
                       # ['orn_remy', 'kc_remy_combo'] +
                       column_groups['model_cols'] +
                       column_groups['vcf5a_cols'] +
                       column_groups['vcf5b_cols'] +
                       column_groups['vcf5q_D03_cols'] +
                       ['vcf5q_dat']
                       )

# %%
label_map = {
    **{item: item.replace('chem-', '') for item in column_groups['chem_cols']},
    'orn_remy': 'ORs',
    'kc_remy_combo': 'KCs',
    'uniform_tom': "modeled KCs uniform",
    'hemibrain_tom': "modeled KCs hemibrain",
    'hemibrain_wd20_tom': "modeled KCs hemibrain wd20",
    }


# %%
def plot_resampled_split_violins(df_resampled_to_plot, space_metric, ci=95, palette='muted',
                                 row_order=None):
    """Plot split violins.

    There should be only 2 values in level 'space_col'. All spaces in 'space_row' will be plotted.

    df_resampled_to_plot should only contain one `resampling_fraction`.

    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)

    sns.violinplot(data=df_resampled_to_plot,
                   y="space_row",
                   x=space_metric,
                   hue="space_col",
                   # density_norm='count',
                   linewidth=0,
                   split=True,
                   gap=0.4,
                   ax=ax,
                   alpha=0.5,
                   palette=palette,
                   order=row_order,
                   # inner_kws=dict(box_width=5, whis_width=1, color=".5", whis=(2.5, 97.5)),
                   )
    # sns.pointplot(
    #         data=df_resampled_to_plot,
    #         y="space_row",
    #         x=space_metric,
    #         hue="space_col",
    #         dodge=.4,
    #         ax=ax,
    #         linestyle="none",
    #         errorbar=('pi', ci),
    #         markeredgewidth=0,
    #         lw=1,
    #         palette=palette,
    #         )
    boxplot_alpha = 1
    sns.boxplot(data=df_resampled_to_plot,
                y="space_row",
                x=space_metric,
                hue="space_col",
                # linewidth=0,
                # inner='box',
                # split=True,
                gap=0.4,
                width=0.2,
                ax=ax,
                whis=(((100 - ci) / 2), 100 - ((100 - ci) / 2)),
                boxprops=dict(alpha=boxplot_alpha),
                capprops=dict(alpha=boxplot_alpha),
                whiskerprops=dict(alpha=boxplot_alpha, linewidth=1),
                medianprops=dict(alpha=boxplot_alpha),
                showfliers=False,
                showcaps=False,
                meanline=True,
                palette=palette
                # inner_kws=dict(box_width=5, whis_width=1, color=".5", whis=(2.5, 97.5)),
                )
    sns.despine()
    try:
        sns.move_legend(ax, "upper left",
                        bbox_to_anchor=(1.0, 1), frameon=False, fontsize=6)
    except:
        print('no legend')
    ax.tick_params(which='both', labelsize=8)
    for artist in ax.get_children():
        if hasattr(artist, "set_clip_on"):
            artist.set_clip_on(False)

    return fig, ax


# %% Load resampling data

n_iter = 5000
n_odor_pairs = 136
with_replacement = True

df_resampled = load_resampling(n_iter, n_odor_pairs, with_replacement)
df_summary_stats = load_summary_stats(n_iter, n_odor_pairs, with_replacement)

# %% Plot split violins

viol_figs = []

plot_rows = ['orn_remy',
             # 'pn_dendrites-correlation',
             'pn_boutons_F_zscore',
             'kc_claws_Fc_zscore',
             'kc_remy_combo',
             'uniform_tom',
             'hemibrain_wd20_tom'
             ]

plot_cols = ['chem_ecfp',
             'vcf5q_ND030_D03_seed08',
             # 'vcf5q_ND030_D04_seed02'
             # 'vcf5a_emb_seed09',
             # 'vcf5a_emb_seed10',
             # 'vcf5b_emb_seed02',
             ]

for resampling_fraction, space_metric, ci in product(
        np.arange(0.1, 1.001, 0.1).round(2),
        ['pearson', 'spearman'],
        [90, 95]):
    with sns.axes_style('ticks'):
        title = (
            f"n_odor_pairs={n_odor_pairs}, with_replacement={with_replacement}, n_iter={n_iter},"
            f"\nmetric={space_metric}, resampling_fraction={resampling_fraction:.2f}, ci={ci:d}%")

        fig_viol, ax = plot_resampled_split_violins(
                (df_resampled
                 .xs(resampling_fraction, level='resampling_fraction')
                 .query("space_row in @plot_rows and space_col in @plot_cols")
                 ),
                space_metric,
                ci=ci,
                palette='Set2',
                row_order=plot_rows,
                )
        ax.set_title(title, fontsize=8)

        plt.tight_layout()
        plt.show()
        viol_figs.append(fig_viol)
# %%
pdf_file = (figure_folder /
            (f"resampled3/resampled_split_violins_{n_odor_pairs}_iter{n_iter:05d}_"
             f"{'with_replacement' if with_replacement else 'no_replacement'}.pdf"))

with PdfPages(pdf_file) as pdf:
    for fig in viol_figs:
        pdf.savefig(fig, bbox_inches='tight')

# %% 2. Plot split violins (split = observed)

viol_figs = []

plot_rows = (['chem_ecfp',
              'chem_fcfp',
              'chem_rdkit',
              'chem_pattern',
              'uniform_tom',
              # 'hemibrain_tom',
              'hemibrain_wd20_tom'] +
             ['vcf5q_ND030_D03_seed08', ])

plot_cols = ['orn_remy',
             # 'pn_dendrites-correlation',
             # 'pn_boutons_F_zscore',
             # 'kc_claws_Fc_zscore',
             'kc_remy_combo',
             # 'uniform_tom',
             # 'hemibrain_wd20_tom'
             ]

for resampling_fraction, space_metric, ci in product(
        np.arange(0.1, 1.00001, 0.1).round(2),
        ['pearson', 'spearman'],
        [90, 95]):
    with sns.axes_style('ticks'):
        title = (
            f"n_odor_pairs={n_odor_pairs}, with_replacement={with_replacement}, n_iter={n_iter},"
            f"\nmetric={space_metric}, resampling_fraction={resampling_fraction:.2f}, ci={ci:d}%")

        fig_viol, ax = plot_resampled_split_violins(
                (df_resampled
                 .xs(resampling_fraction, level='resampling_fraction')
                 .query("space_row in @plot_rows and space_col in @plot_cols")
                 ),
                space_metric,
                ci=ci,
                palette='Set2',
                row_order=plot_rows, )
        ax.set_title(title, fontsize=8)

        plt.tight_layout()
        plt.show()
        viol_figs.append(fig_viol)
# %%
pdf_file = (figure_folder /
            (f"resampled3/resampled_split_violins_{n_odor_pairs}_iter{n_iter:05d}_"
             f"{'with_replacement' if with_replacement else 'no_replacement'}_obs"
             f".pdf"))

with PdfPages(pdf_file) as pdf:
    for fig in viol_figs:
        pdf.savefig(fig, bbox_inches='tight')

# %% Compute stats of differences (KC - ORN)

# %%
plot_rows = ['uniform_tom',
             'hemibrain_tom',
             'hemibrain_wd20_tom',
             'chem_ecfp',
             ] + [
                # 'vcf5q_ND030_D03_seed04',
                'vcf5q_ND030_D03_seed08'
                # 'vcf5a_emb_seed09',
                # 'vcf5b_emb_seed02',
                # 'vcf5q_ND030_D04_seed02'
                ]

space_info = [
    ('orn_remy', 'pn_dendrites-correlation', 'PN dendrites - ORN'),
    ('orn_remy', 'pn_boutons_F_zscore', 'PN boutons - ORN'),
    ('orn_remy', 'kc_claws_Fc_zscore', 'KC claws - ORN'),
    ('orn_remy', 'kc_remy_combo', 'KC - ORN'),
    ]
# space_i = 'orn_remy'
# space_f = 'kc_remy_combo'
# space_diff_col = 'KC - ORN'

pearson_diffs = []
spearman_diffs = []

for space_i, space_f, space_diff_col in space_info:
    plot_cols = [space_i, space_f]

    # Pearson
    # -------
    df_pearson_diff = (df_resampled
                       .query("space_col in @plot_cols")['pearson']
                       .unstack('space_col'))
    pearson_diffs.append((df_pearson_diff[space_f] - df_pearson_diff[space_i])
                         .rename(space_diff_col)
                         )
    # df_pearson_diff[space_diff_col] = (df_pearson_diff[space_f] - df_pearson_diff[space_i])
    # df_pearson_qt = (df_pearson_diff
    #                  .groupby(['resampling_fraction', 'space_row'])
    #                  .quantile([0.025, 0.05, 0.95, 0.975])
    #                  .rename_axis(['resampling_fraction', 'space_row', 'quantile'], axis=0)
    #                  )
    # df_pearson_mean = df_pearson_diff.groupby(['resampling_fraction', 'space_row']).mean()

    # Spearman
    # --------
    df_spearman_diff = (df_resampled.query("space_col in @plot_cols")['spearman']
                        .unstack('space_col'))
    spearman_diffs.append((df_spearman_diff[space_f] - df_spearman_diff[space_i])
                          .rename(space_diff_col)
                          )

    # df_spearman_diff[space_diff_col] = (df_spearman_diff[space_f] - df_spearman_diff[space_i])
    # df_spearman_qt = (df_spearman_diff
    #                   .groupby(['resampling_fraction', 'space_row'])
    #                   .quantile([0.025, 0.05, 0.95, 0.975])
    #                   .rename_axis(['resampling_fraction', 'space_row', 'quantile'], axis=0)
    #                   )
    # df_spearman_mean = df_spearman_diff.groupby(['resampling_fraction', 'space_row']).mean()

df_pearson_diff = pd.concat(pearson_diffs, axis=1)
df_pearson_qt = (df_pearson_diff
                 .groupby(['resampling_fraction', 'space_row'])
                 .quantile([0.025, 0.05, 0.95, 0.975])
                 .rename_axis(['resampling_fraction', 'space_row', 'quantile'], axis=0)
                 )

df_spearman_diff = pd.concat(spearman_diffs, axis=1)
df_spearman_qt = (df_spearman_diff
                  .groupby(['resampling_fraction', 'space_row'])
                  .quantile([0.025, 0.05, 0.95, 0.975])
                  .rename_axis(['resampling_fraction', 'space_row', 'quantile'], axis=0)
                  )
# %% 3. Plot split diff violins (split = observed)
# ['vcf5a_emb_seed09',
#  'vcf5a_emb_seed10',
#  'vcf5b_emb_seed02',
#  'vcf5b_emb_seed07', ] +
# [
#     'vcf5q_ND030_D03_seed04',
#     'vcf5q_ND030_D03_seed08',
#     'vcf5q_ND030_D04_seed02',
#     'vcf5q_ND030_D04_seed05',
#     'vcf5q_dat'
#     ]
viol_figs = []

plot_rows = (['chem_ecfp',
              'chem_fcfp',
              'chem_rdkit',
              'chem_pattern',
              'uniform_tom',
              'hemibrain_tom',
              'hemibrain_wd20_tom'] +
             ['vcf5q_ND030_D03_seed08', ]

             # ['vcf5a_emb_seed09', 'vcf5a_emb_seed10']
             #              [
             #                  # 'vcf5q_ND030_D03_seed04',
             #                  # 'vcf5q_ND030_D03_seed08'
             #                  # 'vcf5a_emb_seed09',
             #                  # 'vcf5b_emb_seed02',
             #                  'vcf5q_ND030_D04_seed02'
             #                  ]
             )
# plot_rows = column_groups['vcf5q_D04_cols']

# # plot_cols = ['KC - ORN']
# plot_cols = [
#     'PN dendrites - ORN',
#     'PN boutons - ORN',
#     'KC claws - ORN',
#     'KC - ORN'
#     ]
# plot_diff = 'PN dendrites - ORN'
# plot_diff = 'PN boutons - ORN'
# plot_diff = 'KC claws - ORN'
plot_diff = 'PN boutons - ORN'

for resampling_fraction, space_metric, ci in product(
        np.arange(0.1, 1.001, 0.1).round(2),
        ['pearson', 'spearman'],
        [90, 95]):

    if space_metric == 'spearman':
        df_plot_diff = df_spearman_diff
    elif space_metric == 'pearson':
        df_plot_diff = df_pearson_diff

    df_plot_diff = (df_plot_diff.xs(resampling_fraction, level='resampling_fraction')
                    .query("space_row in @plot_rows").reset_index())

    with sns.axes_style('ticks'):
        title = (
            f"n_odor_pairs={n_odor_pairs}, with_replacement={with_replacement}, n_iter={n_iter},"
            f"\nmetric={space_metric}, resampling_fraction={resampling_fraction:.2f}, ci={ci:d}%")

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)

        sns.violinplot(data=df_plot_diff,
                       y="space_row",
                       x=plot_diff,
                       order=plot_rows,
                       # hue="space_col",
                       linewidth=0,
                       split=True,
                       gap=0.1,
                       ax=ax,
                       alpha=0.5,
                       color='.2'
                       )

        boxplot_alpha = 1

        sns.boxplot(data=df_plot_diff,
                    y="space_row",
                    x=plot_diff,
                    # linewidth=0,
                    # inner='box',
                    # split=True,
                    gap=0.1,
                    width=0.1,
                    ax=ax,

                    whis=(((100 - ci) / 2), 100 - ((100 - ci) / 2)),
                    boxprops=dict(alpha=boxplot_alpha),
                    capprops=dict(alpha=boxplot_alpha),
                    whiskerprops=dict(alpha=boxplot_alpha, linewidth=1),
                    medianprops=dict(alpha=boxplot_alpha),
                    showfliers=False,
                    showcaps=False,
                    meanline=True,
                    # color='0.7',
                    # inner_kws=dict(box_width=5, whis_width=1, color=".5", whis=(2.5, 97.5)),
                    )
        sns.despine()
        ax.set_title(title, fontsize=8)
        ax.tick_params(which='both', labelsize=8)
        ax.axvline(0, ls='--', color='0.4', lw=1, zorder=-1)
        for artist in ax.get_children():
            if hasattr(artist, "set_clip_on"):
                artist.set_clip_on(False)
        plt.tight_layout()
        plt.show()
        viol_figs.append(fig)
# %%
pdf_file = (figure_folder /
            (f"resampled3/resampled_split_violins_{n_odor_pairs}_iter{n_iter:05d}_"
             f"{'with_replacement' if with_replacement else 'no_replacement'}_diff"
             f"__{plot_diff}.pdf"))

with PdfPages(pdf_file) as pdf:
    for fig in viol_figs:
        pdf.savefig(fig, bbox_inches='tight')
# %%

diff_figs = []
for resampling_fraction in np.arange(0.1, 1.0001, 0.1).round(2):
    for space_metric in ['pearson', 'spearman']:
        for ci in [90, 95]:

            if ci == 90:
                lq = 0.05
                uq = 0.95
            elif ci == 95:
                lw = 0.025
                uq = 0.975

            if space_metric == 'pearson':
                df_displot = (df_pearson_diff.query("space_row in @plot_rows")
                              .xs(resampling_fraction, level='resampling_fraction')
                              .rename_axis([None], axis=1)
                              .reset_index()
                              )
                df_qt = df_pearson_qt
                df_mean = df_pearson_mean
            elif space_metric == 'spearman':
                df_displot = (df_spearman_diff
                              .query("space_row in @plot_rows")
                              .xs(resampling_fraction, level='resampling_fraction')
                              .rename_axis([None], axis=1)
                              .reset_index()
                              )
                df_qt = df_spearman_qt
                df_mean = df_spearman_mean

            with sns.axes_style('ticks'):
                g = sns.displot(df_displot,
                                x='KC - ORN',
                                col='space_row',
                                col_order=plot_rows,
                                col_wrap=3,
                                height=1.75,
                                aspect=1.5,
                                bins=100,
                                fill=True,
                                linewidth=0,
                                kind='hist',
                                element='step',
                                stat='count',
                                kde=True,
                                line_kws=dict(linewidth=0.5, zorder=-1, ),
                                facet_kws=dict(sharex=True, sharey=True, )
                                )

                g.set_titles("{col_name}", fontsize=8)
                for space, ax in g.axes_dict.items():
                    ax.axvline(0, color='0.7', lw=1, ls='-', zorder=-1)

                    ax.axvline(df_qt.at[(resampling_fraction, space, lq), 'KC - ORN'],
                               color='r', lw=0.5, ls='-', zorder=1)
                    ax.axvline(df_qt.at[(resampling_fraction, space, uq), 'KC - ORN'],
                               color='r', lw=0.5, ls='-', zorder=1)
                    ax.axvline(df_mean.at[(resampling_fraction, space), 'KC - ORN'],
                               color='r', lw=0.5, ls='--', zorder=1)
                    for artist in ax.get_children():
                        if hasattr(artist, "set_clip_on"):
                            artist.set_clip_on(False)
                g.figure.suptitle(
                        f"n_odor_pairs={n_odor_pairs}, with_replacement={with_replacement}, "
                        f"n_iter={n_iter}"
                        f"\nmetric={space_metric}, resampling_fraction="
                        f"{resampling_fraction:.2f}, ci={ci:d}%",
                        y=0.999, fontsize=10, va='top'
                        )

                plt.tight_layout()
                plt.show()
                diff_figs.append(g.figure)
    # %%

    pdf_file = (figure_folder /
                (f"resampled3/resampled_diff_dists_{n_odor_pairs}_iter{n_iter:05d}_"
                 f"{'with_replacement' if with_replacement else 'no_replacement'}.pdf"))

    with PdfPages(pdf_file) as pdf:
        for fig in diff_figs:
            pdf.savefig(fig, bbox_inches='tight')
    # %%
