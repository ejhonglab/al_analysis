"""Characterization tests of the MB model (`al_analysis.mb_model`, olfsysm).

Freezes outputs of the current code for the model's main configurations and the
variants added since the repro tests' reference outputs were generated
(`test_mb_model.py::test_fitandplot_repro` covers the main configurations through
`fit_and_plot_mb_model`): tuned spike counts, responses, the tuned parameters and the
PN>KC wiring, from `fit_mb_model` on the committed megamat ORN deltas. Also covers the
outputs `fit_and_plot_mb_model` writes and the `plot-dynamics` command.

Planned to move with the model code (to the olfsysm repository, as pyolfsysm), with its
frozen outputs (`test_data/characterization/model/`) and `characterize.py`.

Frozen outputs come from the code at the commit they were added in. Regenerate with
`CHARACTERIZE_REGEN=1 QUICK=0`. Cases on the full claw / bouton connectome (tens of s to
a few min each) only run with QUICK=0. Spike counts and responses are compared exactly,
so the tests are meant to compare code versions on one machine and olfsysm build (other
compilers / CPUs could change tuning in the last bits).
"""

import os
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from hong2p.types import ParamDict
import numpy as np
import olfsysm as osm
import pandas as pd
import pytest
import xarray as xr

from al_analysis.mb_model import (fit_mb_model, fit_and_plot_mb_model,
    megamat_orn_deltas, connectome_wPNKC,
    load_and_plot_dynamics_cli, load_dynamics, plot_dynamics,
    default_model_odor_startstop_times, read_params, read_tuned_params, KC_ID,
    glomerulus_col, FitMBModelOutputs
)
from al_analysis.wpnkc_shuffles import shuffle_wPNKC, shuffled_wPNKC

from .characterize import check_frozen, frame_summary, to_jsonable
from .conftest import test_data_dir


QUICK: bool = bool(int(os.environ.get('QUICK', False)))

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

slow = pytest.mark.skipif(QUICK, reason='full claw / bouton connectome (QUICK=1)')


@lru_cache(maxsize=None)
def _orn_deltas() -> pd.DataFrame:
    return megamat_orn_deltas()


def orn_deltas() -> pd.DataFrame:
    return _orn_deltas().copy()


@lru_cache(maxsize=None)
def _reduced_bouton_wPNKC() -> pd.DataFrame:
    """Same subset as test_mb_model.py `reduced_bouton_system` (copied, to keep this
    module independent of it): boutons of every 3rd glomerulus, claws of every 4th KC.
    """
    wPNKC = connectome_wPNKC(prat_claws=True, prat_boutons=True)
    gloms = wPNKC.columns.get_level_values(glomerulus_col)
    wPNKC = wPNKC.loc[:, gloms.isin(gloms.unique()[::3])]
    wPNKC = wPNKC[(wPNKC != 0).any(axis='columns')]
    kc_ids = wPNKC.index.get_level_values(KC_ID)
    return wPNKC[kc_ids.isin(kc_ids.unique()[::4])]


@lru_cache(maxsize=None)
def _reduced_kc_wPNKC() -> pd.DataFrame:
    """Returns one-row-per-KC hemibrain wPNKC (`weight_divisor=20`), every 4th KC."""
    wPNKC = connectome_wPNKC(**WD20)
    return wPNKC.iloc[::4]


def reduced_bouton_system() -> Tuple[pd.DataFrame, pd.DataFrame]:
    wPNKC = _reduced_bouton_wPNKC().copy()
    gloms = wPNKC.columns.get_level_values(glomerulus_col)
    orn = orn_deltas()
    return orn[orn.index.isin(gloms)], wPNKC


WD20: ParamDict = dict(weight_divisor=20)
BOUTON: ParamDict = dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
    use_connectome_APL_weights=True
)
CLAWS: ParamDict = dict(one_row_per_claw=True, prat_claws=True)

_default_mp = osm.ModelParams()

# (case ID, base, model kwargs, whether slow). base 'reduced_bouton' runs `BOUTON`
# kwargs on `reduced_bouton_system`; 'full' runs on all ORN deltas / connectome.
CASES: List[Tuple[str, str, ParamDict, bool]] = [
    # one row per KC, hemibrain
    ('hemibrain', 'full', dict(), False),
    ('wd20', 'full', dict(WD20), False),
    ('wd20_connAPL', 'full', dict(WD20, use_connectome_APL_weights=True), False),
    ('wd20_sp0.05', 'full', dict(WD20, target_sparsity=0.05), False),
    ('wd20_nspikes2', 'full', dict(WD20, n_spikes_for_response=2), False),
    ('wd20_kckc', 'full', dict(WD20, kc_kc_inh=10.0), False),
    ('wd20_kckc_tau', 'full', dict(WD20, kc_kc_inh=10.0, kc_kc_tau=0.2), False),
    ('wd20_refractory_adapt', 'full', dict(WD20, kc_refractory_period=3e-3,
        kc_adapt_inc=0.3, kc_adapt_tau=0.2), False
    ),
    ('wd20_divisive', 'full', dict(WD20, apl_inh_divisive=True, max_iters=1000), False),
    ('wd20_apl_taum', 'full', dict(WD20, apl_taum=0.25), False),
    ('wd20_stim_end', 'full', dict(WD20, apl_taum=0.25, time_stim_end=2.0,
        time_end=2.25), False
    ),
    ('wd20_pn_ln_inh', 'full', dict(WD20, pn_inhsc=2 * _default_mp.pn.inhsc,
        ln_inhsc=2 * _default_mp.ln.inhsc), False
    ),
    ('wd20_homeostatic', 'full', dict(WD20, homeostatic_thrs=True), False),
    ('wd20_ext_inh_kc', 'full', dict(WD20, ext_inh_kc='odor0:50.0'), False),
    ('wd20_ext_inh_kc_scale', 'full', dict(WD20, ext_inh_kc='odor0:50.0',
        ext_inh_scale=0.5), False
    ),
    ('wd20_shuffle_curveball', 'full', dict(WD20, _wPNKC='curveball'), False),
    ('wd20_shuffle_claw_permutation_within_type', 'full', dict(WD20,
        _wPNKC='claw_permutation_within_type'), False
    ),
    ('connAPL_multiresponder_boost', 'full', dict(use_connectome_APL_weights=True,
        drop_kcs_with_no_input=False, multiresponder_APL_boost=3.0), False
    ),
    ('equalize_kc_type_sparsity', 'full', dict(equalize_kc_type_sparsity=True,
        ab_prime_response_rate_target=0.2), False
    ),
    ('uniform', 'full', dict(pn2kc_connections='uniform', n_claws=7), False),
    ('uniform_seed', 'full', dict(pn2kc_connections='uniform', n_claws=7, seed=94895),
        False
    ),
    ('fafb_left', 'full', dict(pn2kc_connections='fafb-left', weight_divisor=12),
        False
    ),

    # one row per KC, every 4th KC (for the dynamics outputs)
    ('wd20_reduced', 'reduced_kc', dict(), False),

    # reduced claw + bouton system (every 4th KC, every 3rd glomerulus)
    ('rb', 'reduced_bouton', dict(), False),
    ('rb_pn_claw_to_apl', 'reduced_bouton', dict(pn_claw_to_apl=True), False),
    ('rb_net_inh_per_claw', 'reduced_bouton', dict(allow_net_inh_per_claw=True), False),
    ('rb_claw_dynamics', 'reduced_bouton', dict(pn_claw_to_apl=True,
        claw_dynamics=True), False
    ),
    ('rb_claw_sat', 'reduced_bouton', dict(claw_sat=150.0), False),
    ('rb_claw_sat_hard', 'reduced_bouton', dict(claw_sat=150.0, claw_sat_hard=True),
        False
    ),
    ('rb_claw_thr', 'reduced_bouton', dict(claw_thr=70.0), False),
    ('rb_claw_thr_width', 'reduced_bouton', dict(claw_thr=70.0, claw_thr_width=20.0),
        False
    ),
    ('rb_claw_thr_subgain', 'reduced_bouton', dict(claw_thr=70.0,
        claw_thr_subgain=0.5), False
    ),
    ('rb_claw_pow', 'reduced_bouton', dict(claw_thr=70.0, claw_pow=2.0), False),
    ('rb_claw_pow_ref', 'reduced_bouton', dict(claw_sat=150.0, claw_pow=2.0,
        claw_pow_ref=200.0), False
    ),
    ('rb_bouton_ves_p', 'reduced_bouton', dict(bouton_ves_p=0.01), False),
    ('rb_bouton_ves_p_tau_r', 'reduced_bouton', dict(bouton_ves_p=0.01,
        bouton_tau_r=0.5), False
    ),
    ('rb_kckc', 'reduced_bouton', dict(kc_kc_inh=10.0), False),
    ('rb_divisive', 'reduced_bouton', dict(apl_inh_divisive=True,
        bouton_inh_divisive=True, max_iters=1000), False
    ),
    ('rb_ext_inh_bouton', 'reduced_bouton', dict(ext_inh_bouton='odor0:20.0'), False),
    ('rb_refractory_adapt', 'reduced_bouton', dict(kc_refractory_period=3e-3,
        kc_adapt_inc=0.3, kc_adapt_tau=0.2), False
    ),
    ('rb_apl_unit_coupling', 'reduced_bouton', dict(apl_unit_coupling=1.0), False),
    ('rb_apl_taum', 'reduced_bouton', dict(apl_taum=0.25), False),
    ('rb_shuffle_claw_permutation', 'reduced_bouton', dict(
        _wPNKC='claw_permutation'), False
    ),

    # full claw / bouton connectome (slow)
    ('claws', 'full', dict(CLAWS), True),
    ('claws_connAPL', 'full', dict(CLAWS, use_connectome_APL_weights=True), True),
    ('claws_connAPL_pn_claw_to_apl', 'full', dict(CLAWS,
        use_connectome_APL_weights=True, pn_claw_to_apl=True), True
    ),
    ('claws_connAPL_kc_mean', 'full', dict(CLAWS, use_connectome_APL_weights=True,
        apl_weights_kc_mean=True), True
    ),
    ('claws_connAPL_unit_coupling', 'full', dict(CLAWS,
        use_connectome_APL_weights=True, apl_unit_coupling=1.0), True
    ),
    ('claws_shuffle_claw_permutation', 'full', dict(CLAWS,
        use_connectome_APL_weights=True, _wPNKC='claw_permutation'), True
    ),
    ('bouton', 'full', dict(BOUTON), True),
    ('bouton_pn_claw_to_apl', 'full', dict(BOUTON, pn_claw_to_apl=True), True),
    ('bouton_unit_coupling', 'full', dict(BOUTON, apl_unit_coupling=1.0), True),
    ('uniform_boutons', 'full', dict(pn2kc_connections='uniform-boutons',
        one_row_per_claw=True, prat_boutons=True, use_connectome_APL_weights=True),
        True
    ),
    ('one_row_per_claw', 'full', dict(one_row_per_claw=True), True),
    ('one_row_per_claw_Btn_separate', 'full', dict(one_row_per_claw=True,
        Btn_separate=True, Btn_num_per_glom=3), True
    ),
    ('fafb_right', 'full', dict(pn2kc_connections='fafb-right', weight_divisor=12),
        True
    ),
]
CASE_IDS = [c[0] for c in CASES]
assert len(set(CASE_IDS)) == len(CASE_IDS)

# wPNKC frozen once per distinct wiring (first case using it)
FREEZE_WPNKC_CASES = {'hemibrain', 'wd20', 'wd20_reduced', 'wd20_shuffle_curveball',
    'wd20_shuffle_claw_permutation_within_type', 'uniform', 'uniform_seed', 'fafb_left',
    'rb', 'rb_shuffle_claw_permutation', 'claws', 'claws_shuffle_claw_permutation',
    'bouton', 'uniform_boutons', 'one_row_per_claw', 'one_row_per_claw_Btn_separate',
    'fafb_right'
}


def _ext_inh_odor0(index: pd.Index, odors: pd.Index, value: float) -> pd.DataFrame:
    df = pd.DataFrame(0.0, index=index, columns=odors)
    df.iloc[:, 0] = value
    return df


def _model_inputs(case: str) -> Tuple[pd.DataFrame, ParamDict]:
    """Returns ORN deltas and `fit_mb_model` kwargs for a case (resolving placeholders).
    """
    _, base, kws, _ = CASES[CASE_IDS.index(case)]
    kws = dict(kws)
    if base == 'reduced_bouton':
        orn, wPNKC = reduced_bouton_system()
        kws = {**BOUTON, **kws}
    elif base == 'reduced_kc':
        orn = orn_deltas()
        wPNKC = _reduced_kc_wPNKC().copy()
        kws = {**WD20, **kws}
    else:
        orn = orn_deltas()
        wPNKC = None

    shuffle = kws.pop('_wPNKC', None)
    if shuffle is not None:
        if wPNKC is None:
            kws['_wPNKC'] = shuffled_wPNKC(shuffle, seed=0, **kws)
        else:
            kws['_wPNKC'] = shuffle_wPNKC(wPNKC, shuffle, seed=0)
    elif wPNKC is not None:
        kws['_wPNKC'] = wPNKC

    for k in ('ext_inh_kc', 'ext_inh_bouton'):
        if isinstance(kws.get(k), str):
            value = float(kws[k].split(':')[1])
            base_case = 'wd20' if base == 'full' else 'rb'
            responses, _, base_wPNKC, _ = fit_case(base_case)
            index = responses.index if k == 'ext_inh_kc' else base_wPNKC.columns
            kws[k] = _ext_inh_odor0(index, responses.columns, value)

    if kws.get('multiresponder_APL_boost') is not None:
        mask = pd.read_csv(test_data_dir / 'kiwi-control-multiresponder-mask.csv'
            ).set_index(KC_ID).squeeze()
        mask.name = None
        kws['_multiresponder_mask'] = mask

    return orn, kws


_fit_cache: Dict[str, FitMBModelOutputs] = dict()

def fit_case(case: str) -> FitMBModelOutputs:
    """Returns (cached) `fit_mb_model` outputs for a case. Do not modify outputs."""
    if case not in _fit_cache:
        orn, kws = _model_inputs(case)
        _fit_cache[case] = fit_mb_model(orn, make_plots=False, **kws)
    return _fit_cache[case]


def _is_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, bool, int, float, np.generic))


def freeze_params(prefix: str, params: ParamDict, *, skip=()) -> None:
    """Freezes scalar params as one JSON, and array / Series / frame params separately.
    """
    scalars = dict()
    for k, v in sorted(params.items()):
        if k in skip:
            continue

        if isinstance(v, xr.DataArray):
            check_frozen(f'{prefix}/dynamics/{k}', dataarray_summary(v))

        elif isinstance(v, (pd.DataFrame, pd.Series)):
            check_frozen(f'{prefix}/param_{k}', v)

        elif isinstance(v, np.ndarray) and v.ndim > 0:
            check_frozen(f'{prefix}/param_{k}', v)

        elif isinstance(v, np.ndarray):
            scalars[k] = v.item()

        elif _is_scalar(v) or (isinstance(v, (list, tuple)) and
                all(_is_scalar(x) for x in v)):
            scalars[k] = v
        else:
            scalars[k] = f'<{type(v).__name__}>'

    check_frozen(f'{prefix}/params', scalars)


def dataarray_summary(arr: xr.DataArray) -> Dict[str, Any]:
    """Returns dims / shape / coordinate ranges and value sums of a dynamics array."""
    values = np.asarray(arr.values, dtype=np.float64)
    finite = np.where(np.isfinite(values), values, 0.0)
    summary = dict(dims=list(arr.dims), shape=list(arr.shape),
        sum=float(finite.sum()), abs_sum=float(np.abs(finite).sum()),
        sq_sum=float((finite ** 2).sum()), max=float(finite.max()),
        min=float(finite.min()), n_nan=int(np.isnan(values).sum()),
    )
    time_dims = [d for d in arr.dims if 'time' in str(d)]
    if len(time_dims) == 1:
        time_dim = time_dims[0]
        times = np.asarray(arr[time_dim].values, dtype=np.float64)
        summary['time'] = [float(times[0]), float(times[-1]), len(times)]
        # sum over everything but time, so the time course is frozen too
        other = [d for d in arr.dims if d != time_dim]
        per_time = arr.fillna(0).sum(dim=other).values
        # (every 50th time point, to keep this compact)
        summary['sum_by_time'] = [float(x) for x in per_time[::50]]

    return summary


def wPNKC_nonzero(wPNKC: pd.DataFrame) -> pd.Series:
    """Returns nonzero wPNKC entries (long format), a compact frozen form of wPNKC."""
    # (stack drops all-NaN, not 0. pandas 1.5 and 3 differ in stack defaults)
    values = wPNKC.to_numpy()
    rows, cols = np.nonzero(values)
    index = pd.MultiIndex.from_arrays(
        [wPNKC.index.get_level_values(i)[rows] for i in range(wPNKC.index.nlevels)] +
        [wPNKC.columns.get_level_values(i)[cols] for i in range(wPNKC.columns.nlevels)],
        names=list(wPNKC.index.names) + [f'col_{n}' for n in wPNKC.columns.names]
    )
    return pd.Series(values[rows, cols], index=index, name='weight')


@pytest.mark.parametrize('case', [pytest.param(c[0], marks=slow) if c[3] else c[0]
    for c in CASES
])
def test_fit_mb_model_case(case):
    """Tests `fit_mb_model` outputs (tuned) match frozen outputs, for each case.
    """
    responses, spike_counts, wPNKC, params = fit_case(case)
    prefix = f'model/fit_mb_model/{case}'

    check_frozen(f'{prefix}/spike_counts', spike_counts)
    check_frozen(f'{prefix}/responses', responses)
    # (wPNKC row / column labels, shape and sums, for every case)
    check_frozen(f'{prefix}/wPNKC_summary', dict(frame_summary(wPNKC),
        index_names=[str(x) for x in wPNKC.index.names],
        columns_names=[str(x) for x in wPNKC.columns.names],
        row_sums=[float(x) for x in wPNKC.sum(axis=1).values[::25]],
    ))
    if case in FREEZE_WPNKC_CASES:
        check_frozen(f'{prefix}/wPNKC_nonzero', wPNKC_nonzero(wPNKC))

    freeze_params(prefix, params)

    rate = responses.mean().mean()
    print(f'{case}: response rate {rate:.4f}, {int(spike_counts.values.sum())} spikes, '
        f'{responses.shape[0]} KCs'
    )


def test_fit_mb_model_dynamics():
    """Tests `return_dynamics=True` outputs (ORN / PN / KC / APL time courses) on the
    reduced claw + bouton system, against frozen summaries.
    """
    orn, kws = _model_inputs('rb')
    tuned = fit_case('rb')
    responses, spike_counts, _, params = fit_mb_model(orn, make_plots=False,
        return_dynamics=True, **kws
    )
    assert spike_counts.equals(tuned[1])
    dynamics = {k: v for k, v in params.items() if isinstance(v, xr.DataArray)}
    assert len(dynamics) > 0
    check_frozen('model/dynamics_rb/keys', sorted(dynamics))
    for k, v in sorted(dynamics.items()):
        check_frozen(f'model/dynamics_rb/{k}', dataarray_summary(v))


# outputs `fit_and_plot_mb_model` writes that vary between runs
_VOLATILE_PARAMS = ('output_dir', 'plot_dir', 'model_provenance')


def _output_listing(root: Path) -> List[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())


# (case, whether `plot-dynamics` is expected to fail on its outputs)
FITANDPLOT_CASES = ['wd20_reduced', 'rb']


@pytest.fixture(scope='module', params=FITANDPLOT_CASES)
def fitandplot_output(request, tmp_path_factory) -> Tuple[str, Path, ParamDict]:
    """Runs `fit_and_plot_mb_model` (w/ plots and saved dynamics) for a case.

    Returns (case, plot root, returned params).
    """
    case = request.param
    plot_root = tmp_path_factory.mktemp(f'fitandplot_{case}')
    orn, kws = _model_inputs(case)
    params = fit_and_plot_mb_model(plot_root, orn_deltas=orn, try_cache=False,
        try_lr_cache=False, make_plots=True, return_dynamics=True, **kws
    )
    return case, plot_root, params


def test_fit_and_plot_mb_model_outputs(fitandplot_output):
    """Tests which files `fit_and_plot_mb_model` writes, and the tables in them.
    """
    case, plot_root, params = fitandplot_output
    output_dir = (plot_root / params['output_dir']).resolve()
    assert output_dir.is_dir()
    prefix = f'model/fit_and_plot/{case}'

    listing = _output_listing(output_dir)
    check_frozen(f'{prefix}/output_files', listing)

    tuned = fit_case(case)
    spike_counts = pd.read_parquet(output_dir / 'spike_counts.parquet')
    assert np.array_equal(spike_counts.values, tuned[1].values)
    responses = pd.read_parquet(output_dir / 'responses.parquet')
    assert np.array_equal(responses.values, tuned[0].values)

    for name, read_fn in (('params', read_params), ('tuned_params', read_tuned_params)):
        data = {k: v for k, v in read_fn(output_dir).items()
            if k not in _VOLATILE_PARAMS
        }
        check_frozen(f'{prefix}/{name}', data)

    returned = {k: v for k, v in params.items() if k not in _VOLATILE_PARAMS}
    check_frozen(f'{prefix}/returned_keys', sorted(returned))


_PLOT_DYNAMICS_XFAIL_REASON = ('plot-dynamics fails with the current xarray: '
    'load_and_plot_dynamics calls plot_dynamics w/ odor=slice(None) (and '
    'glom=slice(None), used for claw_sims), and xarray no longer supports slice '
    'selection on a multi-index level (odor / glomerulus) (issue #9)'
)


def _saved_dynamics_dir(fitandplot_output) -> Path:
    case, plot_root, params = fitandplot_output
    output_dir = (plot_root / params['output_dir']).resolve()
    nc_files = sorted(p.name for p in output_dir.glob('*.nc'))
    check_frozen(f'model/plot_dynamics/{case}/nc_files', nc_files)
    assert len(nc_files) > 0
    return output_dir


@pytest.mark.xfail(strict=True, raises=ValueError, reason=_PLOT_DYNAMICS_XFAIL_REASON)
def test_plot_dynamics_cli(fitandplot_output, monkeypatch):
    """Tests `plot-dynamics` on a `fit_and_plot_mb_model` output dir (saved dynamics).
    """
    import matplotlib
    matplotlib.use('agg')

    case = fitandplot_output[0]
    output_dir = _saved_dynamics_dir(fitandplot_output)

    # CLI, as installed `plot-dynamics` script (writes under <model_dir>/dynamics)
    monkeypatch.setattr(sys, 'argv', ['plot-dynamics', str(output_dir)])
    load_and_plot_dynamics_cli()
    check_frozen(f'model/plot_dynamics/{case}/plot_files',
        _output_listing(output_dir / 'dynamics')
    )


def test_load_and_plot_saved_dynamics(fitandplot_output, tmp_path, request):
    """Tests loading saved dynamics (as `plot-dynamics` does), and the first of its
    `plot_dynamics` calls (all glomeruli, default odor), which works on one-row-per-KC
    models.
    """
    import matplotlib
    matplotlib.use('agg')

    case = fitandplot_output[0]
    if case == 'rb':
        request.applymarker(pytest.mark.xfail(strict=True, raises=ValueError,
            reason=_PLOT_DYNAMICS_XFAIL_REASON
        ))

    output_dir = _saved_dynamics_dir(fitandplot_output)
    prefix = f'model/plot_dynamics/{case}'

    dynamics = load_dynamics(output_dir)
    check_frozen(f'{prefix}/loaded_keys', sorted(dynamics))
    for k, v in sorted(dynamics.items()):
        check_frozen(f'{prefix}/loaded/{k}', dataarray_summary(v))

    # plot_dynamics reads other outputs (e.g. wPNKC) from the plot dir's parent, so
    # plotting into a copy of the model output dir (w/o the dynamics themselves)
    import shutil
    model_dir = tmp_path / 'model'
    shutil.copytree(output_dir, model_dir, ignore=shutil.ignore_patterns('*.nc',
        'dynamics'
    ))
    plot_dir = model_dir / 'dynamics'
    plot_dir.mkdir()

    stim_start, stim_end = default_model_odor_startstop_times()
    var2range = plot_dynamics(plot_dir, dynamics, stim_start=stim_start,
        stim_end=stim_end, title=case, glom=slice(None)
    )
    check_frozen(f'{prefix}/plot_files', _output_listing(plot_dir))
    check_frozen(f'{prefix}/var2range', {k: list(v) if v is not None else None
        for k, v in var2range.items()
    })
