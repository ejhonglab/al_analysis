#!/usr/bin/env python3
"""Optimizes claw model KC threshold / APL weights against Remy's megamat KC data.

Standard tuning (as in `model_mb_responses`) picks a KC threshold and APL weights that
hit a target mean response rate. Here, those tuned values are the starting point for
`optimize_mb_model_params`, which minimizes a loss vs Remy's measured per-odor KC
response rates and odor-odor correlations. Both models are run through
`fit_and_plot_mb_model` (for the usual outputs and plots), and their parameters,
losses, response rates, and correlations are compared.
"""

from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from hong2p import olf
from hong2p.types import ParamDict

from al_analysis import al_util
from al_analysis.al_util import savefig, to_csv, warn
from al_analysis.mb_model import (fit_and_plot_mb_model, optimize_mb_model_params,
    make_kc_data_loss, remy_megamat_kc_targets, megamat_orn_deltas,
    optimizable_model_params, drop_silent_model_cells, model_responses_cache_name,
    model_spikecounts_cache_name
)


# claw model, w/ connectome APL<>KC weights (required for one_row_per_claw=True by
# make_fixed_param_kc_simulator, which optimize_mb_model_params uses)
MODEL_KWS: ParamDict = dict(one_row_per_claw=True, prat_claws=True,
    use_connectome_APL_weights=True
)

methods = ('Nelder-Mead', 'Powell', 'differential_evolution', 'dual_annealing',
    'direct'
)

remy_color = 'black'
tuned_color = '#2a78d6'
optimized_color = '#eb6834'


def strip_concs(odors: pd.Index) -> pd.Index:
    """Returns odor names without concentrations (e.g. 'eb @ -3' -> 'eb').
    """
    return pd.Index([olf.parse_odor_name(x, require_conc=False) for x in odors],
        name='odor'
    )


def model_corr(responses: pd.DataFrame) -> pd.DataFrame:
    """Returns odor X odor correlation of model responses, as computed for loss.

    Silent cells are dropped first, and odor names have concentrations stripped.
    """
    corr = drop_silent_model_cells(responses).corr()
    return corr.set_axis(strip_concs(corr.index), axis='index').set_axis(
        strip_concs(corr.columns), axis='columns'
    )


def plot_response_rates(plot_dir: Path, rates: pd.DataFrame,
    losses: Dict[str, Dict[str, float]]) -> None:
    """Plots per-odor response rates of Remy's data and each model.

    Args:
        rates: of shape (# odors, # sources), with columns 'remy', 'tuned', and
            'optimized'

        losses: model name -> dict of losses (from `make_kc_data_loss` loss function)
    """
    rates = rates.sort_values('remy')
    x = np.arange(len(rates))

    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    for col, label, color, marker in [
            ('remy', 'Remy (measured)', remy_color, 'o'),
            ('tuned', f"tuned (loss={losses['tuned']['response_rate_loss']:.3f})",
                tuned_color, 's'
            ),
            ('optimized',
                f"optimized (loss={losses['optimized']['response_rate_loss']:.3f})",
                optimized_color, '^'
            ),
        ]:
        ax.plot(x, rates[col], marker=marker, color=color, linewidth=1.5,
            markersize=6, label=label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(rates.index, rotation=90)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('response rate')
    ax.set_title('per-odor response rates')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3)
    savefig(fig, plot_dir, 'response_rates_per_odor')


def plot_corrs_vs_remy(plot_dir: Path, remy_corr: pd.DataFrame,
    corrs: Dict[str, pd.DataFrame], losses: Dict[str, Dict[str, float]]) -> None:
    """Plots model vs Remy odor-odor correlations, one panel per model.

    Args:
        remy_corr: odor X odor correlations (odors without concentrations)

        corrs: model name ('tuned' / 'optimized') -> odor X odor correlations (same
            odors as `remy_corr`)

        losses: model name -> dict of losses (from `make_kc_data_loss` loss function)
    """
    triu = np.triu_indices(len(remy_corr), k=1)
    remy_values = remy_corr.values[triu]

    fig, axs = plt.subplots(ncols=len(corrs), figsize=(4 * len(corrs), 4),
        sharex=True, sharey=True, layout='constrained'
    )
    for ax, (name, corr) in zip(axs, corrs.items()):
        corr = corr.loc[remy_corr.index, remy_corr.columns]
        values = corr.values[triu]
        color = tuned_color if name == 'tuned' else optimized_color
        ax.scatter(remy_values, values, s=16, color=color, alpha=0.8,
            edgecolors='white', linewidths=0.5
        )
        r = np.corrcoef(remy_values, np.nan_to_num(values))[0, 1]
        ax.set_title(f"{name}\ncorr loss={losses[name]['corr_loss']:.3f}, r={r:.2f}")
        ax.set_xlabel('Remy KC correlation')
        ax.set_aspect('equal')

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    for ax in axs:
        ax.plot(lims, lims, color='gray', linewidth=1, linestyle='--', zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    axs[0].set_ylabel('model KC correlation')
    savefig(fig, plot_dir, 'corrs_vs_remy')


def plot_loss_history(plot_dir: Path, history: pd.DataFrame) -> None:
    """Plots each loss evaluated during optimization, and the best loss so far.
    """
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    ax.scatter(history.index, history['loss'], s=12, color='gray', alpha=0.6,
        label='evaluated'
    )
    ax.plot(history.index, history['loss'].cummin(), color=optimized_color,
        linewidth=2, label='best so far'
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('evaluation')
    ax.set_ylabel('loss (response rate + correlation)')
    ax.set_title('optimization loss history')
    ax.legend(frameon=False)
    savefig(fig, plot_dir, 'loss_history')


def main():
    """Tunes claw model as usual, then optimizes threshold / APL weights vs Remy's data.

    See module docstring, and CLI `--help` for options.
    """
    parser = ArgumentParser()
    parser.add_argument('-m', '--method', default='Nelder-Mead', choices=methods,
        help='optimization method (see optimize_mb_model_params). default: '
        '%(default)s'
    )
    parser.add_argument('-n', '--maxfev', default=200, type=int, help='maximum number '
        'of loss evaluations (each one KC simulation). default: %(default)s'
    )
    parser.add_argument('-t', '--target-sparsity', default=0.1, type=float,
        help='target mean response rate for standard tuning (default: %(default)s, as '
        'in model_mb_responses)'
    )
    parser.add_argument('-i', '--ignore-cache', action='store_true', help='re-runs '
        'standard tuned model, even if cached outputs exist (optimized model is always '
        're-run)'
    )
    args = parser.parse_args()
    method = args.method
    maxfev = args.maxfev
    target_sparsity = args.target_sparsity
    ignore_cache = args.ignore_cache

    plot_root = Path('optimize_megamat_claw_model')
    plot_root.mkdir(exist_ok=True)

    al_util.verbose = True

    orn_deltas = megamat_orn_deltas()
    odors = orn_deltas.columns.get_level_values('odor')

    remy_rates, remy_corr = remy_megamat_kc_targets()
    # same order as model odors, as model_mb_responses does
    remy_corr = remy_corr.loc[odors, odors]

    loss_fn = make_kc_data_loss(target_response_rates=remy_rates,
        target_corr=remy_corr
    )

    print('standard tuning:')
    tuned = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
        target_sparsity=target_sparsity, comparison_kc_corrs=remy_corr,
        try_cache=not ignore_cache, **MODEL_KWS
    )
    tuned_dir = plot_root / tuned['output_dir']
    tuned_params = {p: float(tuned[p]) for p in optimizable_model_params if p in tuned}
    print('tuned params:')
    pprint(tuned_params)

    tuned_responses = pd.read_pickle(tuned_dir / model_responses_cache_name)
    tuned_spike_counts = pd.read_pickle(tuned_dir / model_spikecounts_cache_name)

    print(f'\noptimizing with {method=}, {maxfev=}:')
    ret = optimize_mb_model_params(orn_deltas, loss_fn=loss_fn,
        initial_params=tuned_params, method=method, maxfev=maxfev, **MODEL_KWS
    )
    if not ret['initial_params'] == tuned_params:
        warn('initial parameters of optimization did not match tuned parameters')

    losses = {
        'tuned': loss_fn(tuned_responses, tuned_spike_counts),
        'optimized': ret['losses'],
    }
    # if tuned params did not round trip exactly, optimization would not have started
    # from the same model as the plotted tuned outputs
    if not np.isclose(losses['tuned']['loss'], ret['initial_losses']['loss']):
        warn(f"tuned model loss ({losses['tuned']['loss']:.5g}) != loss at start of "
            f"optimization ({ret['initial_losses']['loss']:.5g})"
        )

    optimized_dirname = f'optimized_{method}'
    print(f'\noptimized model outputs ({optimized_dirname}):')
    fit_and_plot_mb_model(plot_root, plot_dirname=optimized_dirname,
        orn_deltas=orn_deltas, comparison_kc_corrs=remy_corr, try_cache=False,
        **ret['fit_mb_model_kws']
    )

    comparison_dir = plot_root / f'tuned_vs_optimized_{method}'
    comparison_dir.mkdir(exist_ok=True)

    params = pd.DataFrame({'tuned': ret['initial_params'],
        'optimized': ret['params']
    })
    params.index.name = 'param'
    params['optimized/tuned'] = params['optimized'] / params['tuned']

    loss_df = pd.DataFrame(losses)
    loss_df.index.name = 'loss'

    rates = pd.DataFrame({
        'remy': remy_rates,
        'tuned': tuned_responses.mean().set_axis(strip_concs(tuned_responses.columns)),
        'optimized': ret['responses'].mean().set_axis(
            strip_concs(ret['responses'].columns)
        ),
    })
    rates.index.name = 'odor'
    assert not rates.isna().any().any()

    to_csv(params, comparison_dir / 'params.csv')
    to_csv(loss_df, comparison_dir / 'losses.csv')
    to_csv(rates, comparison_dir / 'response_rates.csv')
    to_csv(ret['history'], comparison_dir / 'history.csv')

    remy_corr_stripped = remy_corr.set_axis(strip_concs(remy_corr.index),
        axis='index'
    ).set_axis(strip_concs(remy_corr.columns), axis='columns')
    corrs = {
        'tuned': model_corr(tuned_responses),
        'optimized': model_corr(ret['responses']),
    }
    plot_response_rates(comparison_dir, rates, losses)
    plot_corrs_vs_remy(comparison_dir, remy_corr_stripped, corrs, losses)
    plot_loss_history(comparison_dir, ret['history'])

    print(f'\ntuned ({tuned_dir.name}) vs optimized ({optimized_dirname}):')
    print(params.to_string(float_format='{:.5g}'.format))
    print()
    print(loss_df.to_string(float_format='{:.5g}'.format))
    print(f'\ncomparison outputs saved under {comparison_dir}')


if __name__ == '__main__':
    main()
