#!/usr/bin/env python3
"""Tests for the `fit_mb_model` odor pulse / simulation length kwargs (`TIME_PARAMS`), and
`make_fixed_param_kc_simulator`'s `spike_times` option.
"""

from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import olfsysm as osm

from hong2p.types import ParamDict

from al_analysis.mb_model import (fit_mb_model, megamat_orn_deltas, connectome_wPNKC,
    make_fixed_param_kc_simulator, fixed_params_to_fit_mb_model_kws, format_model_params,
    onestep_lr_cache_key, get_time_index, assert_fit_outputs_equal, glomerulus_col,
    KC_ID, TIME_PARAMS, time_param_dict, _get_fixed_params, FitMBModelOutputs
)

from .conftest import test_data_dir


pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

DEFAULTS = dict(time_stim_end=0.5, time_end=0.75)
LONG = dict(time_stim_end=1.0, time_end=1.25)

_quiet = dict(make_plots=False, silent=True)

# one-row-per-KC hemibrain model (the olfsysm `sim_KC_layer` branch w/o claws), and a
# reduced claw + bouton model w/ connectome APL weights, KC-KC inhibition, a refractory
# period and adaptation (the claw branch, and the kernels a longer simulation runs on)
TIMING_MODEL_KWS = [
    dict(weight_divisor=20),
    dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
        use_connectome_APL_weights=True, kc_kc_inh=50.0, kc_kc_tau=0.05,
        kc_refractory_period=3e-3, kc_adapt_inc=0.1, kc_adapt_tau=1.0
    ),
]


@lru_cache(maxsize=None)
def _reduced_bouton_wPNKC() -> pd.DataFrame:
    """Hemibrain claw+bouton wPNKC reduced as test_mb_model's `reduced_bouton_system`:
    the boutons of every 3rd glomerulus, and the claws (from those) of every 4th KC.
    """
    wPNKC = connectome_wPNKC(prat_claws=True, prat_boutons=True)
    gloms = wPNKC.columns.get_level_values(glomerulus_col)
    wPNKC = wPNKC.loc[:, gloms.isin(gloms.unique()[::3])]
    wPNKC = wPNKC[(wPNKC != 0).any(axis='columns')]
    kc_ids = wPNKC.index.get_level_values(KC_ID)
    return wPNKC[kc_ids.isin(kc_ids.unique()[::4])]


def _model_inputs(orn_deltas: pd.DataFrame, kws: ParamDict
    ) -> Tuple[pd.DataFrame, ParamDict]:
    kws = dict(kws, **_quiet)
    if not kws.get('prat_boutons', False):
        return orn_deltas, kws
    wPNKC = _reduced_bouton_wPNKC().copy()
    gloms = wPNKC.columns.get_level_values(glomerulus_col)
    return orn_deltas[orn_deltas.index.isin(gloms)], {**kws, '_wPNKC': wPNKC}


@pytest.fixture(scope='module')
def orn_deltas() -> pd.DataFrame:
    return megamat_orn_deltas()


def _id(kws: ParamDict) -> str:
    return format_model_params({k: v for k, v in kws.items()})


@pytest.fixture(scope='module', params=TIMING_MODEL_KWS, ids=_id)
def timing_model(request, orn_deltas
    ) -> Tuple[pd.DataFrame, ParamDict, FitMBModelOutputs]:
    """Returns `orn_deltas`, model kwargs and tuned outputs w/o `TIME_PARAMS` passed (do
    not modify).
    """
    orn_deltas, kws = _model_inputs(orn_deltas, request.param)
    return orn_deltas, kws, fit_mb_model(orn_deltas, **kws)


def _pop_time_params(ret: FitMBModelOutputs, expected: ParamDict) -> FitMBModelOutputs:
    responses, spike_counts, wPNKC, params = ret
    params = dict(params)
    for k, v in expected.items():
        x = params.pop(k)
        assert type(x) is float and x == v
    return responses, spike_counts, wPNKC, params


def test_time_defaults_unchanged(timing_model):
    """Tests that passing olfsysm's default pulse / simulation end explicitly changes no
    tuned output (and records them), and that they are not recorded when not passed.
    """
    orn_deltas, kws, ret = timing_model
    assert not any(k in ret[3] for k in TIME_PARAMS)
    mp = osm.ModelParams()
    assert (mp.time_stim_end, mp.time_end) == (0.5, 0.75)

    ret2 = fit_mb_model(orn_deltas, **DEFAULTS, **kws)
    assert_fit_outputs_equal(ret, _pop_time_params(ret2, DEFAULTS))

    # (each alone)
    for k, v in DEFAULTS.items():
        ret3 = fit_mb_model(orn_deltas, **{k: v}, **kws)
        assert_fit_outputs_equal(ret, _pop_time_params(ret3, {k: v}))


@pytest.mark.parametrize('case, kws', [('default', dict()),
    ('connectome-APL_True', dict(use_connectome_APL_weights=True))], ids=['default',
    'connectome-APL_True']
)
def test_time_defaults_repro_committed(case, kws):
    """Tests models w/ olfsysm's default pulse / simulation end passed explicitly still
    reproduce committed reference outputs (the latest
    `test_data/reference_outputs_for_repro` run with the case, as `test_fitandplot_repro`
    uses), exactly.
    """
    root = test_data_dir / 'reference_outputs_for_repro'
    ref_dir = sorted(d for d in root.iterdir() if (d / case).is_dir())[-1] / case
    committed = pd.read_parquet(ref_dir / 'spike_counts.parquet')
    orn_deltas = pd.read_parquet(ref_dir / 'orn_deltas.parquet')

    responses, spike_counts, _, params = fit_mb_model(orn_deltas, **DEFAULTS, **kws,
        **_quiet
    )
    assert all(params[k] == v for k, v in DEFAULTS.items())
    assert spike_counts.equals(committed)
    assert responses.equals(pd.read_parquet(ref_dir / 'responses.parquet'))


def _odor_columns(orn_deltas: pd.DataFrame, odors: pd.Index) -> pd.DataFrame:
    model_odors = orn_deltas.columns
    if 'panel' in model_odors.names:
        model_odors = model_odors.droplevel('panel')
    cols = model_odors.get_indexer(odors)
    assert (cols >= 0).all()
    return orn_deltas.iloc[:, sorted(cols)]


def _spikes_before(spikes: pd.DataFrame, max_step: int) -> pd.DataFrame:
    """The spikes (kc, odor, step; sorted as `simulate` returns them) before `max_step`.
    """
    return spikes.loc[spikes.step < max_step, ['kc', 'odor', 'step']].reset_index(
        drop=True
    )


def test_longer_pulse(timing_model):
    """Tests a longer pulse (1 s, the simulation to 1.25 s) runs, w/ fixed tuned
    parameters: olfsysm's timeseries have the new length, the PN input stays on after
    the default pulse's end, the simulator matches `fit_mb_model`, and (the model being
    causal) the PN rates and every spike before the default pulse's end are the same as
    w/ the default pulse, APL on and off.
    """
    orn_deltas, kws, ret = timing_model
    params = _get_fixed_params(ret[3])
    # (odors w/ the largest spike counts, to keep the recordings small)
    sub = _odor_columns(orn_deltas, ret[1].sum().sort_values(ascending=False).index[:3])
    fixed_kws = {**kws, **fixed_params_to_fit_mb_model_kws(params)}

    r_long, s_long, _, p_long = fit_mb_model(sub, return_olfsysm_vars=True, **LONG,
        **fixed_kws
    )
    mp, rv = p_long['mp'], p_long['rv']
    assert (mp.time_stim_end, mp.time_end) == (1.0, 1.25)
    n_steps = int(round((1.25 - mp.time_pre_start) / mp.time_dt))
    assert rv.orn.sims[0].shape[-1] == n_steps
    ts = get_time_index(mp, rv)
    assert len(ts) == n_steps and ts.iloc[0] == mp.time_pre_start and ts.iloc[-1] == 1.25

    # the input is on longer: PN rates the same up to the default pulse's end, and
    # higher from 0.55 s to 0.75 s (the odor still on), for every odor
    _, _, _, p0 = fit_mb_model(sub, return_olfsysm_vars=True, **fixed_kws)
    rv0 = p0['rv']
    stim_end_step = int(round((0.5 - mp.time_pre_start) / mp.time_dt))
    late = slice(int(round((0.55 - mp.time_pre_start) / mp.time_dt)),
        int(round((0.75 - mp.time_pre_start) / mp.time_dt))
    )
    for pn0, pn1 in zip(rv0.pn.pn_sims, rv.pn.pn_sims):
        assert np.array_equal(pn0[:, :stim_end_step], pn1[:, :stim_end_step])
        assert pn1[:, late].mean() > pn0[:, late].mean()
    del p0, rv0

    simulate0, initial = make_fixed_param_kc_simulator(sub, initial_params=params, **kws)
    simulate, _ = make_fixed_param_kc_simulator(sub, initial_params=params, **LONG, **kws)
    r1, s1 = simulate(dict())
    assert r1.equals(r_long) and s1.equals(s_long)

    apl_off = {k: 0.0 for k in initial if k != 'fixed_thr'}
    for q in (dict(), apl_off):
        _, c0, st0 = simulate0(q, spike_times=True)
        _, c1, st1 = simulate(q, spike_times=True)
        assert st0.step.max() < int(round((0.75 - mp.time_pre_start) / mp.time_dt))
        assert st1.step.max() < n_steps
        before0 = _spikes_before(st0, stim_end_step)
        assert len(before0) > 0
        assert before0.equals(_spikes_before(st1, stim_end_step))
        # (models w/ strong adaptation can respond only at onset, so not more spikes in
        # general)
        assert c1.values.sum() == len(st1) and c0.values.sum() == len(st0)
        print(f"{'APL off' if q else 'APL on'}: {c0.values.sum():.0f} spikes w/ the "
            f'0.5 s pulse, {c1.values.sum():.0f} w/ the 1 s pulse'
        )


def test_spike_times(timing_model):
    """Tests `simulate(..., spike_times=True)`: same responses / spike counts as w/o it
    (before and after), spikes that add up to the spike counts at the times of their
    olfsysm steps, the `kc_spiking` and `kc_thr_factor` paths, and one-odor runs
    (`mp.sim_only`).
    """
    orn_deltas, kws, ret = timing_model
    params = _get_fixed_params(ret[3])
    sub = _odor_columns(orn_deltas, ret[1].sum().sort_values(ascending=False).index[:3])
    simulate, initial = make_fixed_param_kc_simulator(sub, initial_params=params, **kws)

    out = simulate(dict())
    assert len(out) == 2
    r0, c0 = out
    r1, c1, st = simulate(dict(), spike_times=True)
    assert r1.equals(r0) and c1.equals(c0)
    assert simulate(dict())[1].equals(c0)
    assert list(st.columns) == ['kc', 'odor', 'step', 'time_s']

    counts = np.zeros(c0.shape)
    np.add.at(counts, (st.kc.to_numpy(), st.odor.to_numpy()), 1)
    assert np.array_equal(counts, c0.to_numpy(dtype=float))
    mp = osm.ModelParams()
    assert np.allclose(st.time_s, mp.time_pre_start + st.step * mp.time_dt, atol=1e-9)
    assert (st.time_s > mp.time_start).all() and (st.time_s < mp.time_end).all()
    assert not st.duplicated(['kc', 'odor', 'step']).any()

    # (threshold factors, and per-simulation refractory period / adaptation)
    cells = {k: c for k, c in zip(simulate.__code__.co_freevars, simulate.__closure__)}
    factor = np.linspace(0.8, 1.2, cells['spont_in'].cell_contents.size)
    _, cf = simulate(dict(), factor)
    _, cf2, stf = simulate(dict(), factor, spike_times=True)
    assert cf2.equals(cf) and len(stf) == int(cf.values.sum())
    assert not cf.equals(c0)
    spiking = dict(kc_refractory_period=5e-3, kc_adapt_inc=0.3, kc_adapt_tau=0.2)
    _, cs = simulate(dict(), kc_spiking=spiking)
    _, cs2, sts = simulate(dict(), kc_spiking=spiking, spike_times=True)
    assert cs2.equals(cs) and len(sts) == int(cs.values.sum())
    if len(sts) > 1:
        g = sts.groupby(['kc', 'odor']).step.diff().dropna()
        assert (g >= int(round(5e-3 / mp.time_dt))).all()

    # one-odor runs (as E77's PerOdorRunner): only that odor's spikes
    mp_sim = cells['mp'].cell_contents
    order = np.asarray(cells['odor_order'].cell_contents)
    try:
        mp_sim.sim_only = [int(order[1])]
        _, c_one, st_one = simulate(dict(), spike_times=True)
    finally:
        mp_sim.sim_only = []
    assert set(st_one.odor) <= {1}
    assert len(st_one) == int(c_one.iloc[:, 1].sum()) == int(c0.iloc[:, 1].sum())


def test_time_params_invalid(orn_deltas):
    """Tests invalid pulse / simulation ends raise ValueError.
    """
    for bad in (dict(time_stim_end=0.0), dict(time_stim_end=-0.1),
            dict(time_stim_end=0.75), dict(time_stim_end=1.0),
            dict(time_end=0.5), dict(time_stim_end=1.0, time_end=1.0),
            dict(time_end=np.nan), dict(time_stim_end=np.inf, time_end=np.inf),
            # (not on a time step)
            dict(time_stim_end=0.5003), dict(time_stim_end=0.5, time_end=1.00002)
        ):
        with pytest.raises(ValueError, match='time_'):
            time_param_dict(**bad)
        with pytest.raises(ValueError, match='time_'):
            fit_mb_model(orn_deltas, weight_divisor=20, **bad, **_quiet)

    assert time_param_dict() == dict()
    assert time_param_dict(time_stim_end=2, time_end=2.25) == dict(time_stim_end=2.0,
        time_end=2.25
    )


def test_time_params_model_id():
    """Tests model IDs / LR cache keys only change when `TIME_PARAMS` are passed.
    """
    kws = dict(weight_divisor=20)
    none_kws = {**kws, **{k: None for k in TIME_PARAMS}}
    assert format_model_params(none_kws) == format_model_params(kws)
    assert onestep_lr_cache_key(none_kws) == onestep_lr_cache_key(kws)

    variants = (dict(), dict(time_stim_end=0.5), dict(time_end=0.75), DEFAULTS, LONG,
        dict(time_stim_end=2.0, time_end=2.25)
    )
    ids = {format_model_params({**kws, **x}) for x in variants}
    assert len(ids) == len(variants)
    lr_keys = {onestep_lr_cache_key({**kws, **x}) for x in variants}
    assert len(lr_keys) == len(variants)
