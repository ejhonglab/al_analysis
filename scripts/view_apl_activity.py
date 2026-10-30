#!/usr/bin/env python3
"""Views APL activity in space, for models with the APL split into coupled units.

For each odor, saves 2D projections of APL activity, in sections along the
anterior-posterior axis (as viewed from posterior), at several times (see
`plot_apl_activity_sections`), and a 3D view at the middle of the odor presentation. Then
shows an interactive 3D view, with sliders for time and odor (see
`view_apl_activity_3d`), unless `--no-show`. Exits with an error before anything else
(in particular, before fitting any model) if that view could not be shown (e.g. no
display).

APL unit potentials either come from dynamics that `fit_and_plot_mb_model` saved in a
model output directory (with `return_dynamics=True`, for a model with
`apl_unit_coupling`), or from a model tuned on megamat odors (re-using cached outputs, if
any) and re-run on a few odors.
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from hong2p.types import ParamDict

from al_analysis import al_util
from al_analysis.al_util import savefig
from al_analysis.mb_model import (APL_UNIT, megamat_orn_deltas, fit_and_plot_mb_model,
    fit_mb_model, get_thr_and_APL_weights, load_apl_points, load_dynamics,
    plot_apl_activity_sections, view_apl_activity_3d, format_model_params
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


def find_interactive_backend() -> str:
    """Returns name of an interactive matplotlib backend that can show figures here.

    Uses backend from MPLBACKEND or matplotlibrc, if either specifies one, otherwise the
    first that works. Agg is selected on return (as it is after importing `mb_model`,
    which is why `plt.show()` would not otherwise show anything).

    Raises `ImportError` or `RuntimeError` if figures could not be shown (e.g. no
    display or GUI toolkit).
    """
    plt.switch_backend(os.environ.get('MPLBACKEND') or mpl.rcParamsOrig['backend'])
    backend = mpl.get_backend()

    # (also creates the GUI application, which can fail. figure not shown w/o plt.show)
    fig = plt.figure()
    is_interactive = fig.canvas.required_interactive_framework is not None
    plt.close(fig)
    plt.switch_backend('agg')

    # automatic backend selection falls back to Agg, rather than raising
    if not is_interactive:
        raise ImportError(f'{backend=} is not interactive')

    return backend


def check_plots(section_kws: ParamDict, view_kws: ParamDict) -> None:
    """Raises error if plots would fail with these options, regardless of model outputs.

    Plots placeholder APL unit potentials, to find this out without fitting a model.

    Args:
        section_kws: passed to `plot_apl_activity_sections`
        view_kws: passed to `view_apl_activity_3d`
    """
    units = load_apl_points()[APL_UNIT].unique()
    unit_inh = xr.DataArray(np.ones((2, len(units), 2)),
        dims=['stim', APL_UNIT, 'time_s'],
        coords={'stim': ['a', 'b'], APL_UNIT: units, 'time_s': [0.0, 1.0]}
    )
    plt.close(plot_apl_activity_sections(unit_inh.isel(stim=0), **section_kws))
    plt.close(view_apl_activity_3d(unit_inh, show=False, **view_kws))


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

    backend = None
    if not args.no_show:
        try:
            backend = find_interactive_backend()
        except (ImportError, RuntimeError) as err:
            parser.error(f'can not show interactive 3D view: {err}\n'
                'pass --no-show to only save plots'
            )

    al_util.verbose = True

    plot_root = args.output_dir.resolve()
    plot_root.mkdir(exist_ok=True, parents=True)

    section_kws = dict(n_sections=args.n_sections, n_times=args.n_times,
        projection=args.projection, pixel_um=args.pixel_um, relative=args.relative
    )
    view_kws = dict(relative=args.relative, max_points=args.max_points)

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

        check_plots(section_kws, view_kws)
        unit_inh = fit_unit_dynamics(plot_root, kws, args.n_odors)
        name = format_model_params(kws)
        title = format_model_params(kws, human=True)

    plot_dir = plot_root / 'apl_activity' / name
    suffix = '_relative' if args.relative else ''
    for i, odor in enumerate(unit_inh.get_index('stim')):
        odor_str = odor[-1] if isinstance(odor, tuple) else odor
        odor_inh = unit_inh.isel(stim=i)

        fig = plot_apl_activity_sections(odor_inh, title=f'{title}\n{odor_str}',
            **section_kws
        )
        savefig(fig, plot_dir, f'sections_{odor_str}{suffix}', bbox_inches='tight')

        # (keeping 'stim' dim, so odor is in title)
        fig = view_apl_activity_3d(unit_inh.isel(stim=[i]), title=title, show=False,
            **view_kws
        )
        # (PDF would have a vector point for each APL node)
        savefig(fig, plot_dir, f'3d_{odor_str}{suffix}', fmt='png', dpi=150)

    if backend is not None:
        plt.switch_backend(backend)
        view_apl_activity_3d(unit_inh, title=title, **view_kws)


if __name__ == '__main__':
    main()
