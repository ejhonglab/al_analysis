#!/usr/bin/env python3
"""Demonstrates spatial APL models (APL split into coupled units), vs single APL models.

With `apl_unit_coupling=<float>` (see `fit_mb_model`), the APL is split into units
(spatial regions of the hemibrain APL arbor in the calyx), each with its own potential,
coupled to neighboring units. Each claw (and bouton) gives input to, and is inhibited by,
each unit in proportion to its # of synapses with that unit.

This tunes claw and bouton models (with options as used in `model_yang_mixtures.py`)
with a single APL compartment, and with APL units at a few coupling strengths, and then:
- compares response rates (overall and by KC type), mean correlation between odors, and
  the fraction of KC responses that differ from the single APL model
- plots the spatial pattern of APL unit potentials during a few example odors, and how
  much they vary across units, for each coupling strength
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hong2p.types import ParamDict

from al_analysis.al_util import savefig, read_parquet
from al_analysis.mb_model import (megamat_orn_deltas, fit_and_plot_mb_model,
    fit_mb_model, get_thr_and_APL_weights, format_model_params,
    load_apl_unit_centroids, KC_TYPE, APL_UNIT
)


# single APL models to compare against. all with connectome APL weights, which models
# with APL units require. options as in model_yang_mixtures.py.
NAME2BASE_KWS: Dict[str, ParamDict] = {
    'claw': dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True
    ),
    'claw-dynamics': dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True, pn_claw_to_apl=True, claw_dynamics=True
    ),
    'bouton': dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
        use_connectome_APL_weights=True
    ),
    'bouton-pn-claw-to-apl': dict(one_row_per_claw=True, prat_claws=True,
        prat_boutons=True, use_connectome_APL_weights=True, pn_claw_to_apl=True
    ),
}
DEFAULT_BASES: List[str] = ['claw', 'bouton']

# as in model_yang_mixtures.SHORT_MODEL_TUNE_KWS
SHARED_TUNE_KWS: ParamDict = dict(target_sparsity=0.05, n_spikes_for_response=2)

# 0 is uncoupled units. 10 is near strongest allowed w/ default olfsysm time constants
# and n_neighbors=None (see `get_apl_unit_coupling`).
DEFAULT_COUPLINGS: List[float] = [0.0, 1.0, 10.0]

# (olfsysm default stimulus window)
STIM_START_S: float = 0.0
STIM_END_S: float = 0.5


def variant_name(coupling: Optional[float]) -> str:
    """Returns short label for model variant (single APL, or APL units w/ coupling).
    """
    if coupling is None:
        return 'single APL'
    return f'units, coupling={coupling:g}'


def summarize_responses(responses: pd.DataFrame, spike_counts: pd.DataFrame
    ) -> pd.Series:
    """Returns response rates (overall and by KC type) and mean odor correlation.

    Args:
        responses: KC X odor binarized responses
        spike_counts: KC X odor spike counts
    """
    corr = spike_counts.corr().values
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]

    type_response_rates = responses.groupby(KC_TYPE).mean().mean(axis='columns')
    type_response_rates = type_response_rates.drop('unknown', errors='ignore')

    return pd.Series({
        'response_rate': responses.mean().mean(),
        'mean_spikes': spike_counts.mean().mean(),
        'mean_odor_corr': np.nanmean(off_diagonal),
        **{f'response_rate_{t}': r for t, r in type_response_rates.items()},
    })


def mean_unit_potentials(orn_deltas: pd.DataFrame, kws: ParamDict,
    tuned_params: ParamDict) -> pd.DataFrame:
    """Returns odor X APL unit mean potential (during odor), from re-run of tuned model.

    Model is re-run (with thresholds and APL weights fixed to tuned values) only on
    `orn_deltas`, to limit memory needed for dynamics.
    """
    fixed_kws = get_thr_and_APL_weights(tuned_params, kws)
    _, _, _, param_dict = fit_mb_model(orn_deltas=orn_deltas, return_dynamics=True,
        plot_example_dynamics=False, make_plots=False, **kws, **fixed_kws
    )
    unit_inh = param_dict['apl_unit_inh_sims']
    during_odor = unit_inh.sel(time_s=slice(STIM_START_S, STIM_END_S))
    return during_odor.mean('time_s').transpose('stim', APL_UNIT).to_pandas()


def plot_response_summary(summary: pd.DataFrame, plot_dir: Path) -> None:
    """Plots response rates by KC type, and mean odor correlation, for each model.
    """
    bases = list(summary['base'].unique())
    type_cols = [c for c in summary.columns if c.startswith('response_rate_')]

    fig, axs = plt.subplots(2, len(bases), figsize=(4.5 * len(bases), 7),
        squeeze=False, sharey='row', layout='constrained'
    )
    for base, (rate_ax, corr_ax) in zip(bases, axs.T):
        base_summary = summary[summary['base'] == base]
        x = np.arange(len(base_summary))
        labels = base_summary['variant']

        for col in type_cols:
            rate_ax.plot(x, base_summary[col], marker='o',
                label=col[len('response_rate_'):]
            )
        rate_ax.plot(x, base_summary['response_rate'], marker='o', color='k',
            label='all KCs'
        )
        rate_ax.set_title(base)
        rate_ax.set_ylabel('response rate')

        corr_ax.plot(x, base_summary['mean_odor_corr'], marker='o', color='k')
        corr_ax.set_ylabel('mean correlation between odors\n(spike counts)')

        for ax in (rate_ax, corr_ax):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha='right')

    axs[0, -1].legend(title=KC_TYPE, loc='upper left', bbox_to_anchor=(1, 1))
    savefig(fig, plot_dir, 'response_summary', bbox_inches='tight')


def plot_unit_potential_maps(base: str, coupling2potentials: Dict[float, pd.DataFrame],
    plot_dir: Path) -> None:
    """Plots each APL unit's mean potential (relative to mean across units) in space.

    Units are placed at their centroids, projected onto the two principal axes of all
    centroids. One row per odor, one column per coupling strength.
    """
    centroids = load_apl_unit_centroids()
    centered = centroids.values - centroids.values.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xy = pd.DataFrame(centered @ vt[:2].T, index=centroids.index, columns=['pc1', 'pc2'])

    couplings = list(coupling2potentials.keys())
    odors = coupling2potentials[couplings[0]].index

    relative = {c: p.div(p.mean(axis='columns'), axis='index')
        for c, p in coupling2potentials.items()
    }
    vmax = np.nanpercentile(np.concatenate([r.values.ravel() for r in relative.values()]),
        99
    )

    fig, axs = plt.subplots(len(odors), len(couplings),
        figsize=(3.6 * len(couplings), 3.2 * len(odors)), squeeze=False,
        layout='constrained'
    )
    for col, coupling in enumerate(couplings):
        for row, odor in enumerate(odors):
            ax = axs[row, col]
            unit_values = relative[coupling].loc[odor].reindex(xy.index)
            points = ax.scatter(xy['pc1'], xy['pc2'], c=unit_values, s=9, vmin=0,
                vmax=vmax, cmap='viridis'
            )
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(variant_name(coupling))
            if col == 0:
                odor_str = odor[-1] if isinstance(odor, tuple) else odor
                ax.set_ylabel(odor_str)

    fig.colorbar(points, ax=axs, shrink=0.6,
        label='APL unit potential during odor\n(relative to mean across units)'
    )
    fig.suptitle(f'{base}: APL unit potentials (unit centroids, top 2 PCs)')
    savefig(fig, plot_dir, f'apl_unit_potential_maps_{base}', bbox_inches='tight')


def plot_unit_spread(spread: pd.DataFrame, plot_dir: Path) -> None:
    """Plots coefficient of variation of unit potentials vs coupling, for each model.
    """
    fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
    for base, base_spread in spread.groupby('base'):
        ax.plot(base_spread['coupling'], base_spread['unit_cv'], marker='o', label=base)

    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlabel('apl_unit_coupling')
    ax.set_ylabel('CV of APL unit potentials during odor\n(mean across example odors)')
    ax.legend()
    savefig(fig, plot_dir, 'apl_unit_cv_vs_coupling', bbox_inches='tight')


def main():
    """Tunes single APL and APL unit models, and compares their outputs.

    See module docstring and CLI `--help` for options.
    """
    parser = ArgumentParser(description=__doc__,
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument('-b', '--bases', nargs='+', choices=list(NAME2BASE_KWS),
        default=DEFAULT_BASES, help=f'model variants to run (default: {DEFAULT_BASES})'
    )
    parser.add_argument('-k', '--couplings', nargs='+', type=float,
        default=DEFAULT_COUPLINGS,
        help=f'apl_unit_coupling values to try (default: {DEFAULT_COUPLINGS})'
    )
    parser.add_argument('-n', '--n-neighbors', type=int, default=None,
        help='apl_unit_n_neighbors (# of nearest units each unit is coupled to). '
        'default: neighbors in minimum spanning tree of unit path distances'
    )
    parser.add_argument('-d', '--n-dynamics-odors', type=int, default=2,
        help='# of odors (with largest mean ORN response) to re-run each APL unit model '
        'on, to get APL unit potentials (default: 2)'
    )
    parser.add_argument('-c', '--use-cache', action='store_true',
        help='uses cached model outputs, when available'
    )
    parser.add_argument('-o', '--output-dir', type=Path,
        default=Path('spatial_apl_outputs'), help='where to save model outputs and '
        'plots (default: ./spatial_apl_outputs)'
    )
    args = parser.parse_args()

    plot_root = args.output_dir.resolve()
    plot_root.mkdir(exist_ok=True, parents=True)
    plot_dir = plot_root / 'comparison'

    orn_deltas = megamat_orn_deltas()
    dynamics_odors = orn_deltas.mean().nlargest(args.n_dynamics_odors).index
    dynamics_orn_deltas = orn_deltas.loc[:, orn_deltas.columns.isin(dynamics_odors)]

    unit_kws = dict()
    if args.n_neighbors is not None:
        unit_kws['apl_unit_n_neighbors'] = args.n_neighbors

    model_specs = []
    for base in args.bases:
        base_kws = {**NAME2BASE_KWS[base], **SHARED_TUNE_KWS}
        model_specs.append((base, None, base_kws))
        for coupling in args.couplings:
            model_specs.append((base, coupling,
                {**base_kws, 'apl_unit_coupling': coupling, **unit_kws}
            ))

    print('models to run:')
    pprint([format_model_params(kws) for _, _, kws in model_specs])
    print()

    summaries = []
    spreads = []
    base2responses = dict()
    base2coupling2potentials = {b: dict() for b in args.bases}
    for base, coupling, kws in tqdm(model_specs, unit='model'):
        params = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
            try_cache=args.use_cache, max_iters=100, **kws
        )
        output_dir = plot_root / params['output_dir']
        responses = read_parquet(output_dir / 'responses.parquet')
        spike_counts = read_parquet(output_dir / 'spike_counts.parquet')

        summary = summarize_responses(responses, spike_counts)
        summary['base'] = base
        summary['variant'] = variant_name(coupling)
        summary['coupling'] = coupling
        summary['wAPLKC_scale'] = params.get('wAPLKC_scale')

        if coupling is None:
            base2responses[base] = responses
            summary['frac_responses_changed'] = 0.0
        else:
            single_apl_responses = base2responses[base]
            assert responses.index.equals(single_apl_responses.index)
            summary['frac_responses_changed'] = (
                responses != single_apl_responses
            ).mean().mean()

            potentials = mean_unit_potentials(dynamics_orn_deltas, kws, params)
            base2coupling2potentials[base][coupling] = potentials
            unit_cv = potentials.std(axis='columns') / potentials.mean(axis='columns')
            spreads.append({'base': base, 'coupling': coupling,
                'unit_cv': unit_cv.mean()
            })

        summaries.append(summary)

    summary = pd.DataFrame(summaries)
    spread = pd.DataFrame(spreads)

    plot_dir.mkdir(exist_ok=True)
    summary.to_csv(plot_dir / 'response_summary.csv', index=False)
    spread.to_csv(plot_dir / 'apl_unit_cv.csv', index=False)

    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print('\nresponse summary:')
        print(summary.to_string(index=False, float_format='{:.4g}'.format))
        print('\ncoefficient of variation of APL unit potentials during odor:')
        print(spread.to_string(index=False, float_format='{:.4g}'.format))

    plot_response_summary(summary, plot_dir)
    for base, coupling2potentials in base2coupling2potentials.items():
        plot_unit_potential_maps(base, coupling2potentials, plot_dir)
    plot_unit_spread(spread, plot_dir)

    print(f'\noutputs under {plot_root}')


if __name__ == '__main__':
    main()
