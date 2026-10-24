#!/usr/bin/env python3
"""Demonstrates `apl_weights_kc_mean`, which changes how KC<>APL weights are normalized.

With connectome APL weights (`use_connectome_APL_weights=True`), olfsysm multiplies each
APL weight vector by one factor before tuning (see `init_apl_weights` in olfsysm.cpp):
- default (`apl_weights_kc_mean=False`): each vector has a mean of 1 over its own
  entries. For one-row-per-claw models, entries are claws, so each KC's total (summed
  over its claws) has a mean of (# claws / # KCs).
- `apl_weights_kc_mean=True`: per-claw KC<>APL vectors are instead scaled so each KC's
  total has a mean of 1, as for one-row-per-KC models, and for the uniform weights used
  without connectome APL weights.

Either way, relative weights across claws are the same (only one factor changes), and
PN<>APL (per-bouton) weights are normalized the same way. Tuning steps all APL weight
scales together (see `add_to_apl_weight_scales` in olfsysm.cpp), so:
- in claw models, tuned scales absorb the factor, and outputs only differ within tuning
  tolerance
- in bouton models, KC<>APL weights shrink relative to PN<>APL weights, shifting APL
  input and output from KCs towards PNs

This tunes models (options as in model_yang_mixtures.py) with each setting, and then:
- prints / saves means of normalized and tuned weights (per claw, per KC, per bouton),
  tuned scales, and response summaries
- plots distributions of normalized per-claw KC<>APL weights and their per-KC totals
- for bouton models, plots the ratio of total APL>KC to APL>PN (and KC>APL to PN>APL)
  weight, and the fraction of APL drive from KCs (vs PNs) during a few odors (from
  re-running tuned models with dynamics)
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from hong2p.types import ParamDict

from al_analysis.al_util import savefig, read_parquet
from al_analysis.mb_model import (megamat_orn_deltas, fit_and_plot_mb_model,
    fit_mb_model, get_thr_and_APL_weights, format_model_params, KC_ID, KC_TYPE
)


# all with connectome APL weights and one row per claw, which apl_weights_kc_mean
# requires. options as in model_yang_mixtures.py.
NAME2BASE_KWS: Dict[str, ParamDict] = {
    'claw': dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True
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

KC_MEAN2LABEL: Dict[bool, str] = {
    False: 'per-claw mean 1\n(default)',
    True: 'per-KC mean 1\n(apl_weights_kc_mean)',
}

# (olfsysm default stimulus window)
STIM_START_S: float = 0.0
STIM_END_S: float = 0.5


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


def load_normalized_weights(output_dir: Path, params: ParamDict
    ) -> Dict[str, pd.Series]:
    """Returns APL weight name -> normalized (pre-tuning) weights, for one model.

    Normalized weights are the tuned weights `fit_and_plot_mb_model` saves, divided by
    the corresponding tuned scale.
    """
    name2normed = dict()
    for n in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL'):
        weight_path = output_dir / f'{n}.parquet'
        if weight_path.exists():
            name2normed[n] = read_parquet(weight_path) / params[f'{n}_scale']
    return name2normed


def summarize_weights(name2normed: Dict[str, pd.Series], params: ParamDict
    ) -> pd.Series:
    """Returns means of normalized weights, tuned scales, and ratios of tuned totals.
    """
    summary = dict()
    for n, normed in name2normed.items():
        summary[f'{n}_scale'] = params[f'{n}_scale']
        if n in ('wAPLKC', 'wKCAPL'):
            summary[f'{n}_mean_per_claw'] = normed.mean()
            summary[f'{n}_mean_per_kc'] = normed.groupby(level=KC_ID).sum().mean()
        else:
            summary[f'{n}_mean_per_bouton'] = normed.mean()

    if 'wAPLPN' in name2normed:
        def tuned_total(n):
            return params[f'{n}_scale'] * name2normed[n].sum()

        summary['APL>KC / APL>PN'] = tuned_total('wAPLKC') / tuned_total('wAPLPN')
        summary['KC>APL / PN>APL'] = tuned_total('wKCAPL') / tuned_total('wPNAPL')

    return pd.Series(summary)


def kc_fraction_of_apl_drive(orn_deltas: pd.DataFrame, kws: ParamDict,
    tuned_params: ParamDict) -> pd.Series:
    """Returns fraction of APL drive from KCs (vs PN boutons) during each odor.

    Model is re-run (with thresholds and APL weights fixed to tuned values) only on
    `orn_deltas`, to limit memory needed for dynamics. KC and bouton drive are summed
    into the APL potential's derivative in olfsysm (`Is_from_kcs` / `Is_from_pns`).
    """
    fixed_kws = get_thr_and_APL_weights(tuned_params, kws)
    _, _, _, param_dict = fit_mb_model(orn_deltas=orn_deltas, return_dynamics=True,
        plot_example_dynamics=False, make_plots=False, **kws, **fixed_kws
    )
    during_odor = dict(time_s=slice(STIM_START_S, STIM_END_S))
    from_kcs = param_dict['Is_from_kcs'].sel(during_odor).mean('time_s')
    from_pns = param_dict['Is_from_pns'].sel(during_odor).mean('time_s')
    return (from_kcs / (from_kcs + from_pns)).to_pandas()


def plot_weight_distributions(kc_mean2normed: Dict[bool, pd.Series], base: str,
    plot_dir: Path) -> None:
    """Plots normalized wAPLKC per claw, and per-KC totals, for each setting.

    Dashed lines are means.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    per_claw = kc_mean2normed
    per_kc = {k: v.groupby(level=KC_ID).sum() for k, v in kc_mean2normed.items()}
    for ax, kc_mean2values, xlabel in zip(axs, (per_claw, per_kc),
            ('per claw', 'per KC (sum over its claws)')
        ):

        bins = np.histogram_bin_edges(np.concatenate(
            [v.values for v in kc_mean2values.values()]), bins=60
        )
        for i, (kc_mean, values) in enumerate(kc_mean2values.items()):
            color = f'C{i}'
            ax.hist(values, bins=bins, histtype='step', color=color,
                label=KC_MEAN2LABEL[kc_mean].replace('\n', ' ')
            )
            ax.axvline(values.mean(), color=color, linestyle='--')

        ax.set_yscale('log')
        ax.set_xlabel(f'normalized wAPLKC, {xlabel}')
        ax.set_ylabel('count')

    axs[0].legend()
    fig.suptitle(f'{base}: APL>KC weights before tuning')
    savefig(fig, plot_dir, f'wAPLKC_distributions_{base}', bbox_inches='tight')


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
        labels = [KC_MEAN2LABEL[k] for k in base_summary['apl_weights_kc_mean']]

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
            ax.set_xticklabels(labels)
            ax.set_xlim(-0.5, len(x) - 0.5)

    axs[0, -1].legend(title=KC_TYPE, loc='upper left', bbox_to_anchor=(1, 1))
    savefig(fig, plot_dir, 'response_summary', bbox_inches='tight')


def plot_apl_balance(summary: pd.DataFrame, plot_dir: Path) -> None:
    """Plots KC vs PN share of APL weights and APL drive, for bouton models.
    """
    if 'APL>KC / APL>PN' not in summary.columns:
        return

    summary = summary[summary['APL>KC / APL>PN'].notna()]
    bases = list(summary['base'].unique())

    cols = ['APL>KC / APL>PN', 'KC>APL / PN>APL', 'kc_frac_of_apl_drive']
    ylabels = [
        'total tuned APL>KC weight /\ntotal tuned APL>PN weight',
        'total tuned KC>APL weight /\ntotal tuned PN>APL weight',
        'fraction of APL drive from KCs\n(during odor, mean across odors)',
    ]
    fig, axs = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4),
        layout='constrained'
    )
    x = np.arange(len(bases))
    width = 0.38
    for ax, col, ylabel in zip(axs, cols, ylabels):
        for i, kc_mean in enumerate((False, True)):
            values = summary[summary['apl_weights_kc_mean'] == kc_mean].set_index(
                'base').reindex(bases)[col]
            ax.bar(x + (i - 0.5) * width, values, width=width, color=f'C{i}',
                label=KC_MEAN2LABEL[kc_mean].replace('\n', ' ')
            )
        ax.set_xticks(x)
        ax.set_xticklabels(bases)
        ax.set_ylabel(ylabel)

    axs[-1].set_ylim(0, 1)
    axs[-1].legend(loc='upper right')
    savefig(fig, plot_dir, 'apl_kc_vs_pn_balance', bbox_inches='tight')


def main():
    """Tunes models with each apl_weights_kc_mean setting, and compares their outputs.

    See module docstring and CLI `--help` for options.
    """
    parser = ArgumentParser(description=__doc__,
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument('-b', '--bases', nargs='+', choices=list(NAME2BASE_KWS),
        default=DEFAULT_BASES, help=f'model variants to run (default: {DEFAULT_BASES})'
    )
    parser.add_argument('-d', '--n-dynamics-odors', type=int, default=2,
        help='# of odors (with largest mean ORN response) to re-run each bouton model '
        'on, to get fraction of APL drive from KCs (default: 2)'
    )
    parser.add_argument('-c', '--use-cache', action='store_true',
        help='uses cached model outputs, when available'
    )
    parser.add_argument('-o', '--output-dir', type=Path,
        default=Path('apl_weights_kc_mean_outputs'), help='where to save model outputs '
        'and plots (default: ./apl_weights_kc_mean_outputs)'
    )
    args = parser.parse_args()

    plot_root = args.output_dir.resolve()
    plot_root.mkdir(exist_ok=True, parents=True)
    plot_dir = plot_root / 'comparison'
    plot_dir.mkdir(exist_ok=True)

    orn_deltas = megamat_orn_deltas()
    dynamics_odors = orn_deltas.mean().nlargest(args.n_dynamics_odors).index
    dynamics_orn_deltas = orn_deltas.loc[:, orn_deltas.columns.isin(dynamics_odors)]

    model_specs = []
    for base in args.bases:
        for kc_mean in (False, True):
            kws = {**NAME2BASE_KWS[base], **SHARED_TUNE_KWS}
            # only passing when True, so default model output dirs match other scripts
            if kc_mean:
                kws['apl_weights_kc_mean'] = True
            model_specs.append((base, kc_mean, kws))

    print('models to run:')
    pprint([format_model_params(kws) for _, _, kws in model_specs])
    print()

    rows = []
    base2default_responses = dict()
    base2kc_mean2normed = {b: dict() for b in args.bases}
    for base, kc_mean, kws in tqdm(model_specs, unit='model'):
        params = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas,
            try_cache=args.use_cache, max_iters=100, **kws
        )
        output_dir = plot_root / params['output_dir']
        responses = read_parquet(output_dir / 'responses.parquet')
        spike_counts = read_parquet(output_dir / 'spike_counts.parquet')

        name2normed = load_normalized_weights(output_dir, params)
        base2kc_mean2normed[base][kc_mean] = name2normed['wAPLKC']

        row = pd.concat([
            pd.Series({'base': base, 'apl_weights_kc_mean': kc_mean,
                'tuning_iters': params.get('tuning_iters')
            }),
            summarize_responses(responses, spike_counts),
            summarize_weights(name2normed, params),
        ])

        if not kc_mean:
            base2default_responses[base] = responses
        default_responses = base2default_responses[base]
        assert responses.index.equals(default_responses.index)
        row['frac_responses_changed'] = (responses != default_responses).mean().mean()

        if 'wAPLPN' in name2normed:
            row['kc_frac_of_apl_drive'] = kc_fraction_of_apl_drive(dynamics_orn_deltas,
                kws, params
            ).mean()

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(plot_dir / 'summary.csv', index=False)

    id_cols = ['base', 'apl_weights_kc_mean']
    apl_cols = [c for c in summary.columns if c.startswith('w') or ' / ' in c or
        c in ('tuning_iters', 'kc_frac_of_apl_drive')
    ]
    response_cols = [c for c in summary.columns
        if c not in id_cols and c not in apl_cols
    ]
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        for name, cols in (('APL', apl_cols), ('responses', response_cols)):
            print(f'\n{name}:')
            print(summary[id_cols + cols].set_index(id_cols).T.to_string(
                float_format='{:.4g}'.format
            ))

    plot_weight_distributions(base2kc_mean2normed[args.bases[0]], args.bases[0],
        plot_dir
    )
    plot_response_summary(summary, plot_dir)
    plot_apl_balance(summary, plot_dir)

    print(f'\noutputs under {plot_root}')


if __name__ == '__main__':
    main()
