#!/usr/bin/env python3
"""Compares random-bouton models to the latest uniform, claw, and bouton models.

The random-bouton model (`pn2kc_connections='uniform-boutons'`, see `fit_mb_model`) has
the same KCs and boutons as the hemibrain bouton model (`prat_boutons=True`), but each KC
instead has 7 claws, each from a random bouton (each glomerulus drawn with probability
proportional to its # of boutons). Boutons and claws are still separate, so it supports
the same options as the bouton model (PN<>APL weights, `pn_claw_to_apl`,
`claw_dynamics`). KC<>APL weights are the same for each claw, and PN<>APL weights are
from the connectome.

As in `model_yang_mixtures.py` (and with the same `fit_and_plot_mb_model` calls), each
model is tuned on megamat odors, and the tuned thresholds and APL weights are then used
(without retuning) on each of the kiwi and control panels. Then compares, across models:
- response rate to each odor (and by KC type)
- mixture suppression (mix response minus max component response, in # spikes), for the
  binary and 5-component mixtures, among KCs responding to the mix or a component
- how well model odor-odor correlations (megamat) match those of Remy's measured KCs
  (Pearson and Spearman correlation between the two, across odor pairs)
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from hong2p.olf import parse_odor_name
from hong2p.types import ParamDict

from al_analysis.al_util import savefig, read_parquet
from al_analysis.mb_model import (megamat_orn_deltas, natmix_orn_deltas,
    fit_and_plot_mb_model, get_thr_and_APL_weights, format_model_params,
    calc_mix_suppression, plot_avg_mixsupp, load_remy_megamat_mean_kc_corrs,
    drop_silent_model_cells, bootstrapped_corr, MIX_NAME_PREFIXES, KC_TYPE
)


# added to all models in model_yang_mixtures.SHORT_MODEL_TUNE_KWS
SHARED_KWS: ParamDict = dict(target_sparsity=0.05, n_spikes_for_response=2)

# (key order as in model_yang_mixtures.py, so model output directory names match)
NAME2MODEL_KWS: Dict[str, ParamDict] = {
    # uniform and claw models in model_yang_mixtures.SHORT_MODEL_TUNE_KWS
    'uniform': dict(pn2kc_connections='uniform', n_claws=7, **SHARED_KWS),
    'claw': dict(one_row_per_claw=True, prat_claws=True, pn_claw_to_apl=True,
        claw_dynamics=True, **SHARED_KWS
    ),
    # bouton model with the most mixture suppression, of those run by
    # `model_yang_mixtures.py -f -m` (yang_mix_outputs/chosen_modeldirs_max-mixsupp-only.csv)
    'bouton': dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
        pn_claw_to_apl=True, claw_dynamics=True, use_connectome_APL_weights=True,
        **SHARED_KWS
    ),
    # same options as bouton model, except for how claws are defined
    'uniform-bouton': dict(pn2kc_connections='uniform-boutons', one_row_per_claw=True,
        prat_boutons=True, pn_claw_to_apl=True, claw_dynamics=True,
        use_connectome_APL_weights=True, **SHARED_KWS
    ),
}

# as in model_yang_mixtures.py
RESPONSE_RATE_PLOT_MAX: float = 0.15

TUNE_PANEL: str = 'megamat'

MIXSUPP_COL: str = 'mix_minus_comp-max'


def get_model_specs(models: List[str], seeds: List[int]
    ) -> List[Tuple[str, ParamDict]]:
    """Returns (model name, `fit_mb_model` kwargs) for each model to run.

    Args:
        models: keys of `NAME2MODEL_KWS` to run

        seeds: seeds for random-bouton model (one model run per seed). If empty, runs
            once with `fit_mb_model` default seed.
    """
    specs = []
    for name in models:
        kws = NAME2MODEL_KWS[name]
        if kws.get('pn2kc_connections') == 'uniform-boutons' and len(seeds) > 0:
            specs.extend((f'{name}_seed{s}', {**kws, 'seed': s}) for s in seeds)
        else:
            specs.append((name, kws))
    return specs


def fit_pretuned(output_root: Path, kws: ParamDict, tune_orn_deltas: pd.DataFrame,
    panel2orn_deltas: Dict[str, pd.DataFrame], *, use_cache: bool
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Tunes model on `tune_orn_deltas`, then runs it (w/o retuning) on each panel.

    Model outputs are saved under `output_root / 'megamat-tuned'` (tuning) and
    `output_root / <panel>`, as in model_yang_mixtures.py.

    Returns dict of panel -> (responses, spike_counts), including `TUNE_PANEL` for the
    tuning outputs.
    """
    def read_outputs(model_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (read_parquet(model_dir / 'responses.parquet'),
            read_parquet(model_dir / 'spike_counts.parquet')
        )

    tune_root = output_root / f'{TUNE_PANEL}-tuned'
    tune_root.mkdir(exist_ok=True)

    tuned_params = fit_and_plot_mb_model(tune_root, orn_deltas=tune_orn_deltas,
        try_cache=use_cache, response_rate_plot_max=RESPONSE_RATE_PLOT_MAX,
        max_iters=100, **kws
    )
    thr_and_apl_kws = get_thr_and_APL_weights(tuned_params, kws)

    panel2outputs = {
        TUNE_PANEL: read_outputs(tune_root / tuned_params['output_dir'])
    }
    for panel, panel_df in panel2orn_deltas.items():
        panel_dir = output_root / panel
        panel_dir.mkdir(exist_ok=True)

        params = fit_and_plot_mb_model(panel_dir, orn_deltas=panel_df,
            try_cache=use_cache, response_rate_plot_max=RESPONSE_RATE_PLOT_MAX, **kws,
            **thr_and_apl_kws
        )
        panel2outputs[panel] = read_outputs(panel_dir / params['output_dir'])

    return panel2outputs


def find_mixes(odors: pd.Index) -> Dict[str, List[str]]:
    """Returns dict of 'binary' / '5comp' -> [mix odor, component odors...].

    Expects odors of one kiwi/control panel, as in `natmix_orn_deltas`. Air mixes and
    5-component mix dilutions are not used.
    """
    is_5comp = odors.str.startswith(MIX_NAME_PREFIXES)
    is_binary = odors.str.contains('+', regex=False)
    components = odors[~ (is_5comp | is_binary)]

    binary_mixes = [o for o in odors[is_binary] if 'air mix' not in o]
    assert len(binary_mixes) == 1, f'{binary_mixes=}'
    binary_mix = binary_mixes[0]
    binary_comp_names = parse_odor_name(binary_mix).split('+')
    binary_comps = [o for o in components if parse_odor_name(o) in binary_comp_names]
    assert len(binary_comps) == 2, f'{binary_comps=}'

    # undiluted 5-component mix (e.g. 'kmix0 @ 0', vs dilutions 'kmix-1 @ 0', ...)
    mixes_5comp = [o for o in odors[is_5comp] if parse_odor_name(o).endswith('mix0')]
    assert len(mixes_5comp) == 1, f'{mixes_5comp=}'
    assert len(components) == 5, f'{components=}'

    return {
        'binary': [binary_mix] + binary_comps,
        '5comp': mixes_5comp + list(components),
    }


def mix_suppression(responses: pd.DataFrame, spike_counts: pd.DataFrame
    ) -> pd.DataFrame:
    """Returns mixture suppression of each KC responding to a mix or its components.

    Mixture suppression is mix # spikes minus max # spikes across components. Returns
    long-form DataFrame, with 'mix' ('binary' / '5comp') and `MIXSUPP_COL` columns.
    """
    dfs = []
    for mix_type, odors in find_mixes(spike_counts.columns).items():
        responders = responses[odors].any(axis='columns')
        mixsupp = calc_mix_suppression(spike_counts.loc[responders, odors].T)
        mixsupp['mix'] = mix_type
        dfs.append(mixsupp.reset_index())

    return pd.concat(dfs, ignore_index=True)


def odor_pair_corrs(responses: pd.DataFrame, kc_corr: pd.DataFrame) -> pd.DataFrame:
    """Returns model and measured KC correlation for each pair of megamat odors.

    Model correlations are Pearson correlations between odors of binarized responses,
    excluding silent KCs (as in `fit_and_plot_mb_model` correlation plots). Measured
    correlations (`kc_corr`) should be from `load_remy_megamat_mean_kc_corrs`.

    Returns DataFrame with 'model_corr' and 'kc_corr' columns, one row per odor pair.
    """
    model_corr = drop_silent_model_cells(responses).corr()
    model_corr = model_corr.loc[kc_corr.index, kc_corr.columns]

    upper = np.triu_indices(len(kc_corr), k=1)
    return pd.DataFrame({
        'odor1': kc_corr.index[upper[0]],
        'odor2': kc_corr.columns[upper[1]],
        'model_corr': model_corr.values[upper],
        'kc_corr': kc_corr.values[upper],
    })


def corr_of_corrs(pair_corrs: pd.DataFrame) -> pd.Series:
    """Returns Pearson and Spearman correlation (w/ 90% CI) of model vs KC correlations.

    Args:
        pair_corrs: from `odor_pair_corrs`
    """
    stats = dict()
    for method in ('pearson', 'spearman'):
        _, corr, ci_lower, ci_upper, pval = bootstrapped_corr(pair_corrs, 'model_corr',
            'kc_corr', method=method
        )
        stats.update({method: corr, f'{method}_ci_lower': ci_lower,
            f'{method}_ci_upper': ci_upper, f'{method}_pval': pval
        })
    return pd.Series(stats)


def plot_response_rates(response_rates: pd.DataFrame, plot_dir: Path) -> None:
    """Plots response rate to each odor, for each model, with one facet per panel.
    """
    panels = response_rates['panel'].unique()
    fig, axs = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5),
        squeeze=False, sharey=True, layout='constrained'
    )
    for ax, panel in zip(axs[0], panels):
        sns.pointplot(response_rates[response_rates['panel'] == panel], x='odor',
            y='response_rate', hue='model', linestyles='none', dodge=0.5, ax=ax,
            legend=(ax is axs[0, -1])
        )
        ax.set_title(panel)
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelrotation=90)

    axs[0, 0].set_ylabel('response rate')
    sns.move_legend(axs[0, -1], 'upper left', bbox_to_anchor=(1, 1))
    savefig(fig, plot_dir, 'response_rates', bbox_inches='tight')


def plot_model_vs_kc_corrs(pair_corrs: pd.DataFrame, corr_stats: pd.DataFrame,
    plot_dir: Path) -> None:
    """Plots model vs measured KC correlation for each megamat odor pair, per model.
    """
    models = pair_corrs['model'].unique()
    fig, axs = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.6),
        squeeze=False, sharex=True, sharey=True, layout='constrained'
    )
    lims = [
        min(pair_corrs['model_corr'].min(), pair_corrs['kc_corr'].min()) - 0.05,
        max(pair_corrs['model_corr'].max(), pair_corrs['kc_corr'].max()) + 0.05,
    ]
    for ax, model in zip(axs[0], models):
        model_pairs = pair_corrs[pair_corrs['model'] == model]
        ax.plot(lims, lims, color='gray', linewidth=0.8, zorder=0)
        ax.scatter(model_pairs['kc_corr'], model_pairs['model_corr'], s=10, alpha=0.6,
            color='black'
        )
        stats = corr_stats.loc[model]
        ax.set_title(f"{model}\npearson={stats['pearson']:.2f}, "
            f"spearman={stats['spearman']:.2f}", fontsize=9
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.set_xlabel(f"Remy's KC correlation\n({TUNE_PANEL} odor pairs)")

    axs[0, 0].set_ylabel('model KC correlation')
    savefig(fig, plot_dir, f'{TUNE_PANEL}_model_vs_kc_corrs', bbox_inches='tight')


def main():
    """Tunes each model on megamat, runs it on kiwi/control, and compares outputs.

    See module docstring and CLI `--help` for options.
    """
    parser = ArgumentParser(description=__doc__,
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument('-s', '--seeds', nargs='+', type=int, default=[],
        help='seeds for random-bouton model, which is run once per seed (default: '
        'once, with fit_mb_model default seed)'
    )
    parser.add_argument('-m', '--models', nargs='+', choices=list(NAME2MODEL_KWS),
        default=list(NAME2MODEL_KWS),
        help=f'models to run (default: all of {list(NAME2MODEL_KWS)})'
    )
    parser.add_argument('-i', '--ignore-cache', action='store_true',
        help='re-runs models even if cached outputs exist'
    )
    parser.add_argument('-o', '--output-dir', type=Path,
        default=Path('uniform_bouton_outputs'), help='where to save model outputs and '
        'plots (default: ./uniform_bouton_outputs)'
    )
    args = parser.parse_args()

    output_root = args.output_dir.resolve()
    output_root.mkdir(exist_ok=True, parents=True)
    plot_dir = output_root / 'comparison'
    plot_dir.mkdir(exist_ok=True)

    tune_orn_deltas = megamat_orn_deltas()
    assert set(tune_orn_deltas.columns.get_level_values('panel')) == {TUNE_PANEL}

    natmix_df = natmix_orn_deltas().sort_index().sort_index(axis='columns')
    panels = natmix_df.columns.get_level_values('panel')
    panel2orn_deltas = {p: natmix_df.loc[:, panels == p] for p in panels.unique()}

    kc_corr = load_remy_megamat_mean_kc_corrs()

    model_specs = get_model_specs(args.models, args.seeds)
    print('models to run:')
    pprint({name: format_model_params(kws) for name, kws in model_specs})
    print()

    rate_dfs = []
    type_rate_dfs = []
    mixsupp_dfs = []
    pair_corr_dfs = []
    name2corr_stats = dict()
    for name, kws in tqdm(model_specs, unit='model'):
        panel2outputs = fit_pretuned(output_root, kws, tune_orn_deltas,
            panel2orn_deltas, use_cache=not args.ignore_cache
        )
        for panel, (responses, spike_counts) in panel2outputs.items():
            ids = dict(model=name, panel=panel)

            rates = responses.mean().rename('response_rate').reset_index()
            rate_dfs.append(rates.assign(**ids))

            # (models w/ pn2kc_connections='uniform' do not have KC types)
            if KC_TYPE in responses.index.names:
                type_rates = responses.groupby(KC_TYPE).mean().mean(axis='columns')
                type_rates = type_rates.rename('response_rate').reset_index()
                type_rate_dfs.append(type_rates.assign(**ids))

            if panel == TUNE_PANEL:
                pair_corrs = odor_pair_corrs(responses, kc_corr)
                pair_corr_dfs.append(pair_corrs.assign(model=name))
                name2corr_stats[name] = corr_of_corrs(pair_corrs)
            else:
                mixsupp_dfs.append(
                    mix_suppression(responses, spike_counts).assign(**ids)
                )

    response_rates = pd.concat(rate_dfs, ignore_index=True)
    type_response_rates = pd.concat(type_rate_dfs, ignore_index=True)
    mixsupp = pd.concat(mixsupp_dfs, ignore_index=True)
    pair_corrs = pd.concat(pair_corr_dfs, ignore_index=True)
    corr_stats = pd.DataFrame(name2corr_stats).T
    corr_stats.index.name = 'model'

    mean_mixsupp = mixsupp.groupby(['panel', 'mix', 'model'], sort=False)[MIXSUPP_COL
        ].agg(['mean', 'size']).rename(columns={'size': 'n_responders'})

    response_rates.to_csv(plot_dir / 'response_rates.csv', index=False)
    type_response_rates.to_csv(plot_dir / 'response_rates_by_kc_type.csv', index=False)
    mixsupp.to_csv(plot_dir / 'mix_suppression.csv', index=False)
    mean_mixsupp.to_csv(plot_dir / 'mean_mix_suppression.csv')
    pair_corrs.to_csv(plot_dir / f'{TUNE_PANEL}_model_vs_kc_corrs.csv', index=False)
    corr_stats.to_csv(plot_dir / f'{TUNE_PANEL}_model_vs_kc_corr_stats.csv')

    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print('\nmean response rate:')
        print(response_rates.pivot_table(index='model', columns='panel',
            values='response_rate', sort=False
        ).to_string(float_format='{:.4f}'.format))
        print('\nmean response rate by KC type:')
        print(type_response_rates.pivot_table(index='model',
            columns=['panel', KC_TYPE], values='response_rate', sort=False
        ).to_string(float_format='{:.4f}'.format))
        print('\nmean mixture suppression (# spikes, among responders):')
        print(mean_mixsupp.to_string(float_format='{:.3f}'.format))
        print(f'\n{TUNE_PANEL} model vs KC odor-odor correlations (90% CI):')
        print(corr_stats.to_string(float_format='{:.3g}'.format))

    plot_response_rates(response_rates, plot_dir)
    for mix_type, mix_df in mixsupp.groupby('mix', sort=False):
        plot_avg_mixsupp(mix_df, plot_dir, x='model', lin_col=MIXSUPP_COL,
            fname_suffix=f'_{mix_type}', title=f'{mix_type} mixtures'
        )
    plot_model_vs_kc_corrs(pair_corrs, corr_stats, plot_dir)

    print(f'\noutputs under {output_root}')


if __name__ == '__main__':
    main()
