#!/usr/bin/env python3
"""Compares APL unit models across coupling strengths, without differences from tuning.

Standard tuning stops once the response rate is within `sp_acc` (10%) of its target, so
separately tuned models can have quite different response rates. Here, for each base
model (see `BASES`), all models instead share one threshold, and:
- a single APL model has its APL weights scaled to hit the target number of responses
  (as closely as the discrete number of responses allows)
- 'fixed' sweep: models with APL units at each coupling strength use exactly the same
  threshold and APL weights as the single APL model, to show how units and coupling
  change responses at the same parameters
- 'matched' sweep: models with APL units at each coupling strength have their APL
  weights scaled to hit the same target number of responses, to show which KCs respond
  differently without any difference in overall response rate

APL unit potentials are then computed on a few odors (with weights from the 'matched'
sweep), to measure how much they vary across units, and to plot them in space with shared
color scales and times.

Outputs are saved under `<output-dir>/<base>/`. `sweep.csv` has response comparisons to
the single APL model (see `response_changes`), and losses vs Remy's megamat KC data (see
`make_kc_data_loss`). `apl_unit_cv.csv` has the coefficient of variation across units of
mean potential during each odor.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import gc
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from hong2p.types import ParamDict

from al_analysis.al_util import savefig, to_csv, to_parquet
from al_analysis.mb_model import (megamat_orn_deltas, fit_mb_model,
    make_fixed_param_kc_simulator, fixed_params_to_fit_mb_model_kws, make_kc_data_loss,
    remy_megamat_kc_targets, plot_apl_activity_sections, format_model_params, APL_UNIT
)
from model_spatial_apl import (NAME2BASE_KWS, summarize_responses, response_changes,
    plot_unit_potential_maps, STIM_START_S, STIM_END_S
)


Simulator = Callable[[ParamDict], Tuple[pd.DataFrame, pd.DataFrame]]

# base model name -> (model kwargs, target mean response rate)
BASES: Dict[str, Tuple[ParamDict, float]] = {
    'claw_10pct': (NAME2BASE_KWS['claw'], 0.1),
    'claw_5pct': (dict(NAME2BASE_KWS['claw'], n_spikes_for_response=2), 0.05),
    'bouton_10pct': (NAME2BASE_KWS['bouton'], 0.1),
}

# 15 is near strongest allowed w/ default olfsysm time constants (see
# `max_apl_unit_coupling`)
DEFAULT_COUPLINGS: List[float] = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0]

# subset of couplings to plot potentials in space for
PLOT_COUPLINGS: List[float] = [0.0, 1.0, 10.0]
SECTION_TIMES_S: Tuple[float, ...] = (0.03, 0.05, 0.07, 0.11, 0.15)


def scale_apl_weights(params: ParamDict, factor: float) -> ParamDict:
    """Returns `params` with all APL weight scales multiplied by `factor`.

    As in standard tuning, which changes all APL weight scales by the same factor.
    """
    return {k: v * factor if k.endswith('_scale') else v for k, v in params.items()}


def match_response_count(simulate: Simulator, params: ParamDict, n_target: int, *,
    initial_factor: float = 1.0, tol: int = 1, max_evals: int = 30
    ) -> Tuple[float, pd.DataFrame, pd.DataFrame, int]:
    """Finds factor to scale APL weights by, so # of responses is closest to `n_target`.

    Assumes # of responses decreases with stronger APL weights.

    Args:
        simulate: from `make_fixed_param_kc_simulator`
        params: parameters to simulate with, before scaling APL weights
        n_target: target number of KC-odor pairs with a response
        initial_factor: first factor to try
        tol: stops once # of responses is within this of `n_target`
        max_evals: stops after this many simulations

    Returns best `factor`, its `responses` and `spike_counts`, and # of simulations run.
    """
    n_evals = 0
    best = None

    def n_responses(factor: float) -> int:
        nonlocal n_evals, best
        responses, spike_counts = simulate(scale_apl_weights(params, factor))
        n_evals += 1
        n = int((responses != 0).values.sum())
        if best is None or abs(n - n_target) < abs(best[0] - n_target):
            best = (n, factor, responses, spike_counts)
        return n

    def done() -> bool:
        return abs(best[0] - n_target) <= tol or n_evals >= max_evals

    # bracketing target, with (log factor, # responses) on each side
    log_factor = np.log(initial_factor)
    n = n_responses(initial_factor)
    step = 0.03 if n > n_target else -0.03
    prev_log_factor, prev_n = log_factor, n
    while (prev_n > n_target) == (n > n_target) and not done():
        prev_log_factor, prev_n = log_factor, n
        log_factor += step
        step *= 2
        n = n_responses(np.exp(log_factor))

    (log_low, n_low), (log_high, n_high) = sorted(
        [(prev_log_factor, prev_n), (log_factor, n)]
    )
    while not done() and log_high - log_low > 1e-7:
        width = log_high - log_low
        # false position, kept away from ends of bracket, as responses are discrete
        frac = np.clip((n_low - n_target) / max(n_low - n_high, 1), 0.1, 0.9)
        log_factor = log_low + frac * width
        n = n_responses(np.exp(log_factor))
        if n > n_target:
            log_low, n_low = log_factor, n
        else:
            log_high, n_high = log_factor, n

    n, factor, responses, spike_counts = best
    return factor, responses, spike_counts, n_evals


def compare_to_single(responses: pd.DataFrame, spike_counts: pd.DataFrame,
    single_responses: pd.DataFrame, single_spike_counts: pd.DataFrame) -> ParamDict:
    """Returns how responses differ from single APL model, including odor correlations.

    `jaccard_floor` is lowest `jaccard_distance` possible given the numbers of responses
    (if one model's responses were a subset of the other's).
    """
    n_responses = int((responses != 0).values.sum())
    n_single = int((single_responses != 0).values.sum())
    changes = response_changes(responses, single_responses)

    corr = spike_counts.corr().values
    single_corr = single_spike_counts.corr().values
    upper = np.triu_indices(len(corr), k=1)
    return {
        **changes,
        'n_lost': int(round(changes['frac_lost'] * n_single)),
        'n_gained': int(round(changes['frac_gained'] * n_single)),
        'jaccard_floor': 1 - min(n_responses, n_single) / max(n_responses, n_single),
        'odor_corr_rms_diff': np.sqrt(np.nanmean((corr[upper] - single_corr[upper]) ** 2)),
    }


def summarize_run(responses: pd.DataFrame, spike_counts: pd.DataFrame,
    loss_fn: Callable[[pd.DataFrame, pd.DataFrame], Dict[str, float]]) -> ParamDict:
    """Returns response rates, correlations, and losses vs measured KC data.
    """
    return {
        **summarize_responses(responses, spike_counts).to_dict(),
        'n_responses': int((responses != 0).values.sum()),
        'frac_silent_kcs': float((responses == 0).all(axis='columns').mean()),
        **loss_fn(responses, spike_counts),
    }


def unit_dynamics(orn_deltas: pd.DataFrame, kws: ParamDict, params: ParamDict
    ) -> Tuple[xr.DataArray, pd.DataFrame]:
    """Returns APL unit potentials (stim X apl_unit X time_s) and responses for `params`.
    """
    responses, _, _, param_dict = fit_mb_model(orn_deltas, return_dynamics=True,
        plot_example_dynamics=False, make_plots=False,
        **{**kws, **fixed_params_to_fit_mb_model_kws(params)}
    )
    return param_dict['apl_unit_inh_sims'], responses


def dynamics_odor_subset(orn_deltas: pd.DataFrame, n_odors: int) -> pd.DataFrame:
    """Returns `orn_deltas` for odors evenly spaced in rank by mean ORN response.

    Strongest and weakest odors are included. Too many odors at once can use a lot of
    memory for dynamics (> 1GB per odor).
    """
    order = orn_deltas.mean().sort_values(ascending=False).index
    odors = order[np.linspace(0, len(order) - 1, n_odors).round().astype(int)]
    return orn_deltas.loc[:, orn_deltas.columns.isin(odors)]


def plot_potentials(base: str, coupling2unit_inh: Dict[float, xr.DataArray],
    plot_dir: Path) -> None:
    """Plots APL unit potentials in space, comparing couplings on shared color scales.

    Saves posterior sections for the odor with the largest potentials, and maps of unit
    potentials relative to mean across units.
    """
    couplings = list(coupling2unit_inh.keys())
    first = coupling2unit_inh[couplings[0]]
    example_stim = int(first.mean([APL_UNIT, 'time_s']).argmax('stim'))
    odor = first.get_index('stim')[example_stim]
    odor = odor[-1] if isinstance(odor, tuple) else odor

    section_inh = {k: v.isel(stim=example_stim).sel(time_s=list(SECTION_TIMES_S),
            method='nearest'
        ) for k, v in coupling2unit_inh.items()
    }
    vmax = float(np.percentile(
        np.concatenate([v.values.ravel() for v in section_inh.values()]), 99
    ))
    for coupling, unit_inh in coupling2unit_inh.items():
        fig = plot_apl_activity_sections(unit_inh.isel(stim=example_stim),
            times=SECTION_TIMES_S, vmax=vmax,
            title=f'{base}, coupling={coupling:g}\n{odor}'
        )
        savefig(fig, plot_dir, f'sections_coupling_{coupling:g}', bbox_inches='tight')

    coupling2potentials = {k: mean_during_odor(v.isel(stim=[example_stim]))
        for k, v in coupling2unit_inh.items()
    }
    plot_unit_potential_maps(base, coupling2potentials, plot_dir)


def mean_during_odor(unit_inh: xr.DataArray) -> pd.DataFrame:
    """Returns odor X APL unit mean potential during odor.
    """
    during_odor = unit_inh.sel(time_s=slice(STIM_START_S, STIM_END_S))
    return during_odor.mean('time_s').transpose('stim', APL_UNIT).to_pandas()


def run_base(base: str, orn_deltas: pd.DataFrame, loss_fn, couplings: Sequence[float],
    plot_root: Path, *, n_dynamics_odors: int) -> None:
    """Runs sweeps for one base model (see module docstring), saving outputs as they go.
    """
    kws, target_sparsity = BASES[base]
    output_dir = plot_root / base
    responses_dir = output_dir / 'responses'
    responses_dir.mkdir(exist_ok=True, parents=True)

    rows = []

    def save_run(mode: str, coupling: float, params: ParamDict, factor: float,
        n_evals: int, responses: pd.DataFrame, spike_counts: pd.DataFrame,
        comparison: ParamDict) -> None:
        name = mode if coupling is None else f'{mode}_coupling_{coupling:g}'
        to_parquet(responses, responses_dir / f'responses_{name}.parquet')
        to_parquet(spike_counts, responses_dir / f'spike_counts_{name}.parquet')
        rows.append({'base': base, 'mode': mode, 'coupling': coupling, **params,
            'weight_factor': factor, 'n_evals': n_evals,
            **summarize_run(responses, spike_counts, loss_fn), **comparison
        })
        to_csv(pd.DataFrame(rows), output_dir / 'sweep.csv', index=False,
            multiple_saves_per_run_ok=True
        )
        print(f'{base} {name}: {rows[-1]["n_responses"]} responses, '
            f'{factor=:.5f}, {n_evals=}', flush=True
        )

    simulate, tuned_params = make_fixed_param_kc_simulator(orn_deltas,
        target_sparsity=target_sparsity, max_iters=100, **kws
    )
    n_target = int(round(target_sparsity * orn_deltas.shape[1] * (
        simulate(tuned_params)[0].shape[0]
    )))
    factor, single_responses, single_spike_counts, n_evals = match_response_count(
        simulate, tuned_params, n_target
    )
    single_params = scale_apl_weights(tuned_params, factor)
    save_run('single', None, single_params, factor, n_evals, single_responses,
        single_spike_counts, compare_to_single(single_responses, single_spike_counts,
            single_responses, single_spike_counts
        )
    )
    del simulate
    gc.collect()

    # (coupling here is only the initial value. all simulated below.)
    simulate, _ = make_fixed_param_kc_simulator(orn_deltas, initial_params=single_params,
        apl_unit_coupling=couplings[0], **kws
    )
    coupling2matched_params = dict()
    coupling2matched_responses = dict()
    factor = 1.0
    for coupling in couplings:
        params = {**single_params, 'apl_unit_coupling': coupling}

        responses, spike_counts = simulate(params)
        save_run('fixed', coupling, params, 1.0, 1, responses, spike_counts,
            compare_to_single(responses, spike_counts, single_responses,
                single_spike_counts
            )
        )

        # (starting from factor for previous coupling, which should be close)
        factor, responses, spike_counts, n_evals = match_response_count(simulate, params,
            n_target, initial_factor=factor
        )
        params = scale_apl_weights(params, factor)
        coupling2matched_params[coupling] = params
        coupling2matched_responses[coupling] = responses
        save_run('matched', coupling, params, factor, n_evals, responses, spike_counts,
            compare_to_single(responses, spike_counts, single_responses,
                single_spike_counts
            )
        )

    del simulate
    gc.collect()

    if n_dynamics_odors <= 0:
        return

    odor_subset = dynamics_odor_subset(orn_deltas, n_dynamics_odors)
    cvs = []
    coupling2unit_inh = dict()
    for coupling in couplings:
        unit_inh, dyn_responses = unit_dynamics(odor_subset, kws,
            coupling2matched_params[coupling]
        )
        # simulator and fit_mb_model should agree
        matched_responses = coupling2matched_responses[coupling]
        if not np.array_equal(dyn_responses.values,
                matched_responses.loc[:, dyn_responses.columns].values):
            raise RuntimeError(f'{base=}, {coupling=}: responses from fit_mb_model did '
                'not match simulator'
            )

        potentials = mean_during_odor(unit_inh)
        unit_cv = potentials.std(axis='columns') / potentials.mean(axis='columns')
        for odor, cv in unit_cv.items():
            cvs.append({'base': base, 'coupling': coupling,
                'odor': odor[-1] if isinstance(odor, tuple) else odor, 'unit_cv': cv,
                'mean_potential': potentials.loc[[odor]].values.mean()
            })
        to_csv(pd.DataFrame(cvs), output_dir / 'apl_unit_cv.csv', index=False,
            multiple_saves_per_run_ok=True
        )

        if coupling in PLOT_COUPLINGS:
            # (only need the potentials, not the rest of the dynamics)
            coupling2unit_inh[coupling] = unit_inh.copy(deep=True)

        del unit_inh, dyn_responses
        gc.collect()

    plot_potentials(base, coupling2unit_inh, output_dir)


def main():
    """Runs sweeps for each base model (see module docstring, and CLI `--help`).
    """
    parser = ArgumentParser(description=__doc__,
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument('-b', '--bases', nargs='+', choices=list(BASES),
        default=list(BASES), help='base models to run (default: all)'
    )
    parser.add_argument('-k', '--couplings', nargs='+', type=float,
        default=DEFAULT_COUPLINGS,
        help=f'apl_unit_coupling values to try (default: {DEFAULT_COUPLINGS})'
    )
    parser.add_argument('-d', '--n-dynamics-odors', type=int, default=4,
        help='# of odors to compute APL unit potentials on. 0 to skip. (default: 4)'
    )
    parser.add_argument('-o', '--output-dir', type=Path,
        default=Path('apl_coupling_sweep'), help='where to save outputs '
        '(default: ./apl_coupling_sweep)'
    )
    args = parser.parse_args()

    plot_root = args.output_dir.resolve()
    plot_root.mkdir(exist_ok=True, parents=True)

    orn_deltas = megamat_orn_deltas()
    odors = orn_deltas.columns.get_level_values('odor')

    remy_rates, remy_corr = remy_megamat_kc_targets()
    # same order as model odors, as optimize_megamat_claw_model.py does
    remy_corr = remy_corr.loc[odors, odors]
    loss_fn = make_kc_data_loss(target_response_rates=remy_rates, target_corr=remy_corr)

    for base in args.bases:
        print(f'{base}: {format_model_params(BASES[base][0])}', flush=True)
        run_base(base, orn_deltas, loss_fn, args.couplings, plot_root,
            n_dynamics_odors=args.n_dynamics_odors
        )

    print(f'\noutputs under {plot_root}')


if __name__ == '__main__':
    main()
