#!/usr/bin/env python3
"""Optimizes claw model KC threshold / APL weights against Remy's megamat KC data.

Standard tuning (as in `model_mb_responses`) picks a KC threshold and APL weights that
hit a target mean response rate. Here, those tuned values are the starting point for
`optimize_mb_model_params`, which minimizes a loss vs Remy's measured per-odor KC
response rates and odor-odor correlations. Both models are run through
`fit_and_plot_mb_model` (for the usual outputs and plots), and their parameters,
losses, response rates, and correlations are compared.

This is done for each of these cases (see `CASES`):
- 'claw': claw model, optimizing threshold and wAPLKC_scale, with wKCAPL_scale tied to
  wAPLKC_scale (`optimize_mb_model_params` default)
- 'claw_separate': claw model, optimizing threshold and each APL weight scale
  separately
- 'bouton_separate': claw model with PN boutons (and PN<>APL weights), optimizing
  threshold and each of the 4 APL weight scales separately
- 'claw_units': as 'claw', but with the APL split into coupled spatial units (see
  `apl_unit_coupling` in `fit_mb_model`), also optimizing the coupling between units
- 'bouton_units_separate': as 'bouton_separate', but with APL units, also optimizing
  the coupling between units

Models with APL units are tuned with `apl_unit_coupling=1` (`INITIAL_APL_UNIT_COUPLING`),
and coupling is optimized within `-k/--coupling-bounds` (by default, from nearly
uncoupled to the strongest coupling olfsysm can simulate).

Model outputs only depend on products of APL input and output weight scales (e.g.
wKCAPL_scale * wAPLKC_scale), so individual scales from separate optimization are not
unique, but these products are (and are included in parameter comparisons).
"""

from argparse import ArgumentParser
import inspect
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional, Tuple

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
    optimizable_model_params, drop_silent_model_cells, read_model_output,
    max_apl_unit_coupling
)


# claw model, w/ connectome APL<>KC weights (required for one_row_per_claw=True by
# make_fixed_param_kc_simulator, which optimize_mb_model_params uses)
CLAW_MODEL_KWS: ParamDict = dict(one_row_per_claw=True, prat_claws=True,
    use_connectome_APL_weights=True
)
# as above, but also w/ PN boutons, and connectome PN<>APL weights
BOUTON_MODEL_KWS: ParamDict = dict(one_row_per_claw=True, prat_claws=True,
    prat_boutons=True, use_connectome_APL_weights=True
)

# apl_unit_coupling for standard tuning of models w/ APL units (as in the MODEL_KW_LIST
# entries w/ APL units), which optimization of coupling starts from
INITIAL_APL_UNIT_COUPLING: float = 1.0

# case name -> (model kwargs, `params_to_optimize` for optimize_mb_model_params). with
# APL units, apl_unit_coupling is optimized in both cases (None and 'all').
CASES: Dict[str, Tuple[ParamDict, Optional[str]]] = {
    'claw': (CLAW_MODEL_KWS, None),
    'claw_separate': (CLAW_MODEL_KWS, 'all'),
    'bouton_separate': (BOUTON_MODEL_KWS, 'all'),
    'claw_units': (
        dict(CLAW_MODEL_KWS, apl_unit_coupling=INITIAL_APL_UNIT_COUPLING), None
    ),
    'bouton_units_separate': (
        dict(BOUTON_MODEL_KWS, apl_unit_coupling=INITIAL_APL_UNIT_COUPLING), 'all'
    ),
}

# (APL input weight scale, APL output weight scale), for each path through the APL.
# products of these (and the threshold) determine model outputs.
APL_PATH2SCALES: Dict[str, Tuple[str, str]] = {
    'KC>APL>KC': ('wKCAPL_scale', 'wAPLKC_scale'),
    'KC>APL>PN': ('wKCAPL_scale', 'wAPLPN_scale'),
    'PN>APL>KC': ('wPNAPL_scale', 'wAPLKC_scale'),
    'PN>APL>PN': ('wPNAPL_scale', 'wAPLPN_scale'),
}

methods = ('Nelder-Mead', 'Powell', 'differential_evolution', 'dual_annealing',
    'direct'
)
default_bound_factor: float = inspect.signature(optimize_mb_model_params).parameters[
    'bound_factor'].default

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


def param_table(ret: ParamDict) -> pd.DataFrame:
    """Returns tuned and optimized parameters, and products of APL weight scales.

    Args:
        ret: output of `optimize_mb_model_params`, with tuned parameters as initial
            parameters

    Returns DataFrame with columns 'tuned', 'optimized', 'optimized/tuned', and 'fit'
    (whether each parameter was optimized, tied to wAPLKC_scale, or fixed, or 'product'
    for rows with products of APL input and output weight scales).
    """
    params = pd.DataFrame({'tuned': ret['initial_params'], 'optimized': ret['params']})
    for path, (input_scale, output_scale) in APL_PATH2SCALES.items():
        if input_scale in params.index and output_scale in params.index:
            params.loc[f'{input_scale}*{output_scale} ({path})'] = (
                params.loc[input_scale] * params.loc[output_scale]
            )

    params.index.name = 'param'
    params['optimized/tuned'] = params['optimized'] / params['tuned']

    def fit_type(param: str) -> str:
        if param in ret['params_to_optimize']:
            return 'optimized'
        elif param in ret['tied_params']:
            return 'tied to wAPLKC_scale'
        elif param in ret['params']:
            return 'fixed'
        return 'product'

    params['fit'] = [fit_type(p) for p in params.index]
    return params


def plot_response_rates(plot_dir: Path, rates: pd.DataFrame,
    losses: Dict[str, Dict[str, float]], case: str) -> None:
    """Plots per-odor response rates of Remy's data and each model.

    Args:
        rates: of shape (# odors, # sources), with columns 'remy', 'tuned', and
            'optimized'

        losses: model name -> dict of losses (from `make_kc_data_loss` loss function)

        case: name of case (key of `CASES`), for title
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
    ax.set_title(f'{case}: per-odor response rates')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3)
    savefig(fig, plot_dir, 'response_rates_per_odor')


def plot_corrs_vs_remy(plot_dir: Path, remy_corr: pd.DataFrame,
    corrs: Dict[str, pd.DataFrame], losses: Dict[str, Dict[str, float]], case: str
    ) -> None:
    """Plots model vs Remy odor-odor correlations, one panel per model.

    Args:
        remy_corr: odor X odor correlations (odors without concentrations)

        corrs: model name ('tuned' / 'optimized') -> odor X odor correlations (same
            odors as `remy_corr`)

        losses: model name -> dict of losses (from `make_kc_data_loss` loss function)

        case: name of case (key of `CASES`), for title
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
    fig.suptitle(case)
    savefig(fig, plot_dir, 'corrs_vs_remy')


def plot_loss_history(plot_dir: Path, history: pd.DataFrame, case: str) -> None:
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
    ax.set_title(f'{case}: optimization loss history')
    ax.legend(frameon=False)
    savefig(fig, plot_dir, 'loss_history')


def run_case(plot_root: Path, case: str, orn_deltas: pd.DataFrame,
    remy_rates: pd.Series, remy_corr: pd.DataFrame, *, method: str, maxfev: int,
    bound_factor: float, coupling_bounds: Tuple[float, float], target_sparsity: float,
    ignore_cache: bool, output_suffix: str) -> Dict[str, Dict[str, float]]:
    """Tunes model for `case` as usual, optimizes from there, and compares the two.

    Saves comparison CSVs and plots under `<plot_root>/tuned_vs_optimized_<case>_
    <output_suffix>`, alongside `fit_and_plot_mb_model` outputs for both models.

    Args:
        case: key of `CASES`

        remy_rates/remy_corr: from `remy_megamat_kc_targets` (correlations with odors
            in same order as `orn_deltas`)

        method/maxfev/bound_factor: passed to `optimize_mb_model_params`

        coupling_bounds: (lower, upper) bounds for `apl_unit_coupling` (only used for
            cases w/ APL units)

        output_suffix: appended to names of optimized model and comparison output
            directories

    Returns dict with 'tuned' and 'optimized' losses (each a dict, from
    `make_kc_data_loss` loss function).
    """
    model_kws, params_to_optimize = CASES[case]

    loss_fn = make_kc_data_loss(target_response_rates=remy_rates,
        target_corr=remy_corr
    )

    print(f'{case}: standard tuning:')
    tuned = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
        target_sparsity=target_sparsity, comparison_kc_corrs=remy_corr,
        try_cache=not ignore_cache, **model_kws
    )
    tuned_dir = plot_root / tuned['output_dir']
    tuned_params = {p: float(tuned[p]) for p in optimizable_model_params if p in tuned}
    bounds = None
    if 'apl_unit_coupling' in model_kws:
        tuned_params['apl_unit_coupling'] = float(model_kws['apl_unit_coupling'])
        bounds = {'apl_unit_coupling': coupling_bounds}

    print('tuned params:')
    pprint(tuned_params)

    tuned_responses = read_model_output(tuned_dir / 'responses')
    tuned_spike_counts = read_model_output(tuned_dir / 'spike_counts')

    print(f'\n{case}: optimizing with {method=}, {maxfev=}, {params_to_optimize=}:')
    ret = optimize_mb_model_params(orn_deltas, loss_fn=loss_fn,
        initial_params=tuned_params, params_to_optimize=params_to_optimize,
        method=method, maxfev=maxfev, bounds=bounds, bound_factor=bound_factor,
        **model_kws
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

    optimized_dirname = f'optimized_{case}_{output_suffix}'
    print(f'\n{case}: optimized model outputs ({optimized_dirname}):')
    fit_and_plot_mb_model(plot_root, plot_dirname=optimized_dirname,
        orn_deltas=orn_deltas, comparison_kc_corrs=remy_corr, try_cache=False,
        **ret['fit_mb_model_kws']
    )

    comparison_dir = plot_root / f'tuned_vs_optimized_{case}_{output_suffix}'
    comparison_dir.mkdir(exist_ok=True)

    params = param_table(ret)

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
    plot_response_rates(comparison_dir, rates, losses, case)
    plot_corrs_vs_remy(comparison_dir, remy_corr_stripped, corrs, losses, case)
    plot_loss_history(comparison_dir, ret['history'], case)

    print(f'\n{case}: tuned ({tuned_dir.name}) vs optimized ({optimized_dirname}):')
    print(params.to_string(float_format='{:.5g}'.format))
    print()
    print(loss_df.to_string(float_format='{:.5g}'.format))
    print(f'\ncomparison outputs saved under {comparison_dir}\n')

    return losses


def main():
    """Tunes models as usual, then optimizes threshold / APL weights vs Remy's data.

    See module docstring, and CLI `--help` for options.
    """
    parser = ArgumentParser()
    parser.add_argument('-c', '--cases', nargs='+', default=list(CASES),
        choices=list(CASES), help='which cases to run (see module docstring). default: '
        'all'
    )
    parser.add_argument('-m', '--method', default='Nelder-Mead', choices=methods,
        help='optimization method (see optimize_mb_model_params). default: '
        '%(default)s'
    )
    parser.add_argument('-n', '--maxfev', default=200, type=int, help='maximum number '
        'of loss evaluations (each one KC simulation), per case. default: %(default)s'
    )
    parser.add_argument('-b', '--bound-factor', default=default_bound_factor,
        type=float, help='each optimized parameter is searched between its tuned value '
        'divided and multiplied by this (see optimize_mb_model_params). default: '
        '%(default)s'
    )
    parser.add_argument('-k', '--coupling-bounds', nargs=2, type=float,
        default=None, metavar=('LOWER', 'UPPER'), help='bounds for apl_unit_coupling, '
        'for cases with APL units. default: 0.01 to the strongest coupling olfsysm can '
        'simulate (see max_apl_unit_coupling)'
    )
    parser.add_argument('-t', '--target-sparsity', default=0.1, type=float,
        help='target mean response rate for standard tuning (default: %(default)s, as '
        'in model_mb_responses)'
    )
    parser.add_argument('-i', '--ignore-cache', action='store_true', help='re-runs '
        'standard tuned models, even if cached outputs exist (optimized models are '
        'always re-run)'
    )
    args = parser.parse_args()
    cases = args.cases
    method = args.method
    maxfev = args.maxfev
    bound_factor = args.bound_factor
    target_sparsity = args.target_sparsity
    ignore_cache = args.ignore_cache

    output_suffix = method
    if bound_factor != default_bound_factor:
        output_suffix += f'_bound-factor-{bound_factor:g}'

    if args.coupling_bounds is None:
        coupling_bounds = (0.01, max_apl_unit_coupling())
    else:
        coupling_bounds = tuple(args.coupling_bounds)
        output_suffix += '_coupling-bounds-{:g}-{:g}'.format(*coupling_bounds)

    plot_root = Path('optimize_megamat_claw_model')
    plot_root.mkdir(exist_ok=True)

    al_util.verbose = True

    orn_deltas = megamat_orn_deltas()
    odors = orn_deltas.columns.get_level_values('odor')

    remy_rates, remy_corr = remy_megamat_kc_targets()
    # same order as model odors, as model_mb_responses does
    remy_corr = remy_corr.loc[odors, odors]

    case2losses = dict()
    for case in cases:
        case2losses[case] = run_case(plot_root, case, orn_deltas, remy_rates,
            remy_corr, method=method, maxfev=maxfev, bound_factor=bound_factor,
            coupling_bounds=coupling_bounds, target_sparsity=target_sparsity,
            ignore_cache=ignore_cache, output_suffix=output_suffix
        )

    summary = pd.concat(
        {case: pd.DataFrame(losses).T for case, losses in case2losses.items()},
        names=['case', 'model']
    )
    to_csv(summary, plot_root / f'summary_{output_suffix}.csv')
    print('losses for each case:')
    print(summary.to_string(float_format='{:.5g}'.format))


if __name__ == '__main__':
    main()
