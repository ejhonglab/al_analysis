#!/usr/bin/env python3
"""Views APL activity in space, for models with the APL split into coupled units.

For each odor, saves 2D projections of APL activity, in sections along the
anterior-posterior axis (as viewed from posterior), at several times (see
`plot_apl_activity_sections`), and a 3D view at the middle of the odor presentation. Then
shows an interactive 3D view, with sliders for time and odor (see
`view_apl_activity_3d`), unless `--no-show`.

APL unit potentials either come from dynamics that `fit_and_plot_mb_model` saved in a
model output directory (with `return_dynamics=True`, for a model with
`apl_unit_coupling`), or from a model tuned on megamat odors (re-using cached outputs, if
any) and re-run on a few odors.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from typing import Dict

import xarray as xr

from hong2p.types import ParamDict

from al_analysis import al_util
from al_analysis.al_util import savefig
from al_analysis.mb_model import (megamat_orn_deltas, fit_and_plot_mb_model,
    fit_mb_model, get_thr_and_APL_weights, load_dynamics, plot_apl_activity_sections,
    view_apl_activity_3d, format_model_params
)


# options as in model_spatial_apl.py
NAME2BASE_KWS: Dict[str, ParamDict] = {
    'claw': dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True
    ),
    'bouton': dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
        use_connectome_APL_weights=True
    ),
}
SHARED_TUNE_KWS: ParamDict = dict(target_sparsity=0.05, n_spikes_for_response=2)


def fit_unit_dynamics(plot_root: Path, kws: ParamDict, n_odors: int) -> xr.DataArray:
    """Returns APL unit potentials (stim X apl_unit X time_s) for model w/ `kws`.

    Model is tuned on megamat odors (via `fit_and_plot_mb_model`, using cached outputs
    if available), then re-run with tuned thresholds and APL weights fixed, on the
    `n_odors` odors with the largest mean ORN response.
    """
    orn_deltas = megamat_orn_deltas()
    params = fit_and_plot_mb_model(plot_root, orn_deltas=orn_deltas, try_cache=True,
        max_iters=100, **kws
    )
    fixed_kws = get_thr_and_APL_weights(params, kws)

    odors = orn_deltas.mean().nlargest(n_odors).index
    _, _, _, param_dict = fit_mb_model(
        orn_deltas=orn_deltas.loc[:, orn_deltas.columns.isin(odors)],
        return_dynamics=True, plot_example_dynamics=False, make_plots=False, **kws,
        **fixed_kws
    )
    return param_dict['apl_unit_inh_sims']


def main():
    """Plots and shows APL activity (see module docstring and CLI `--help`).
    """
    parser = ArgumentParser(description=__doc__,
        formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument('model_dir', type=Path, nargs='?', default=None,
        help='fit_and_plot_mb_model output directory, with saved apl_unit_inh_sims.nc '
        'dynamics. if not passed, fits a model (see -b, -k, -n, -d)'
    )
    parser.add_argument('-b', '--base', choices=list(NAME2BASE_KWS), default='claw',
        help='model to fit, if no model_dir (default: %(default)s)'
    )
    parser.add_argument('-k', '--coupling', type=float, default=1.0,
        help='apl_unit_coupling of model to fit (default: %(default)s)'
    )
    parser.add_argument('-n', '--n-neighbors', type=int, default=None,
        help='apl_unit_n_neighbors of model to fit (default: neighbors in minimum '
        'spanning tree)'
    )
    parser.add_argument('-d', '--n-odors', type=int, default=2,
        help='# of odors (with largest mean ORN response) to re-run fit model on '
        '(default: %(default)s)'
    )
    parser.add_argument('-s', '--n-sections', type=int, default=4,
        help='# of sections along anterior-posterior axis (default: %(default)s)'
    )
    parser.add_argument('-t', '--n-times', type=int, default=5,
        help='# of times (evenly spaced while APL is active) to show for each section '
        '(default: %(default)s)'
    )
    parser.add_argument('-r', '--relative', action='store_true',
        help='shows potentials relative to mean across APL units (at each time)'
    )
    parser.add_argument('-p', '--projection', choices=['mean', 'max'], default='mean',
        help='how potentials of APL nodes within each pixel of section projections are '
        'combined (default: %(default)s)'
    )
    parser.add_argument('--pixel-um', type=float, default=2.0,
        help='pixel size of section projections, in um (default: %(default)s)'
    )
    parser.add_argument('--max-points', type=int, default=None,
        help='only show a random subset of this many APL nodes in 3D (faster to redraw)'
    )
    parser.add_argument('--no-show', action='store_true',
        help='only save plots, without showing interactive 3D view'
    )
    parser.add_argument('-o', '--output-dir', type=Path,
        default=Path('apl_activity_outputs'), help='where to save fit model outputs '
        'and plots (default: ./%(default)s)'
    )
    args = parser.parse_args()

    al_util.verbose = True

    plot_root = args.output_dir.resolve()
    plot_root.mkdir(exist_ok=True, parents=True)

    if args.model_dir is not None:
        unit_inh = load_dynamics(args.model_dir)['apl_unit_inh_sims']
        name = args.model_dir.resolve().name
        title = name
    else:
        kws = {**NAME2BASE_KWS[args.base], **SHARED_TUNE_KWS,
            'apl_unit_coupling': args.coupling
        }
        if args.n_neighbors is not None:
            kws['apl_unit_n_neighbors'] = args.n_neighbors

        unit_inh = fit_unit_dynamics(plot_root, kws, args.n_odors)
        name = format_model_params(kws)
        title = format_model_params(kws, human=True)

    plot_dir = plot_root / 'apl_activity' / name
    suffix = '_relative' if args.relative else ''
    for i, odor in enumerate(unit_inh.get_index('stim')):
        odor_str = odor[-1] if isinstance(odor, tuple) else odor
        odor_inh = unit_inh.isel(stim=i)

        fig = plot_apl_activity_sections(odor_inh, n_sections=args.n_sections,
            n_times=args.n_times, projection=args.projection, pixel_um=args.pixel_um,
            relative=args.relative, title=f'{title}\n{odor_str}'
        )
        savefig(fig, plot_dir, f'sections_{odor_str}{suffix}', bbox_inches='tight')

        # (keeping 'stim' dim, so odor is in title)
        fig = view_apl_activity_3d(unit_inh.isel(stim=[i]), relative=args.relative,
            max_points=args.max_points, title=title, show=False
        )
        # (PDF would have a vector point for each APL node)
        savefig(fig, plot_dir, f'3d_{odor_str}{suffix}', fmt='png', dpi=150)

    if not args.no_show:
        view_apl_activity_3d(unit_inh, relative=args.relative,
            max_points=args.max_points, title=title
        )


if __name__ == '__main__':
    main()
