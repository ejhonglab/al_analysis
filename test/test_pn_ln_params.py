#!/usr/bin/env python3
"""Tests for `fit_mb_model` antennal lobe inhibition kwargs (`AL_INH_PARAMS`).
"""

import numpy as np
import pandas as pd
import pytest
import olfsysm as osm

from al_analysis.mb_model import (fit_mb_model, megamat_orn_deltas, AL_INH_PARAMS,
    format_model_params, make_fixed_param_kc_simulator,
    fixed_params_to_fit_mb_model_kws, onestep_lr_cache_key, get_time_index
)


pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

# plain hemibrain model (one row per KC, from QUICK_MODEL_KW_LIST), for speed
MODEL_KWS = dict(weight_divisor=20, make_plots=False, silent=True)


def al_inh_values(mp: osm.ModelParams):
    """Returns dict of `AL_INH_PARAMS` kwarg -> value of corresponding `mp` parameter.
    """
    return {
        'pn_inhsc': mp.pn.inhsc,
        'pn_inhadd': mp.pn.inhadd,
        'ln_inhsc': mp.ln.inhsc,
        'ln_inhadd': mp.ln.inhadd,
    }


def olfsysm_defaults():
    return al_inh_values(osm.ModelParams())


@pytest.fixture(scope='module')
def orn_deltas() -> pd.DataFrame:
    return megamat_orn_deltas()


@pytest.fixture(scope='module')
def default_fit(orn_deltas):
    """Returns `fit_mb_model` outputs (tuned, w/ `return_olfsysm_vars=True`) for
    `MODEL_KWS`, w/o any `AL_INH_PARAMS` passed (do not modify).
    """
    return fit_mb_model(orn_deltas, return_olfsysm_vars=True, **MODEL_KWS)


def fixed_param_kws(param_dict):
    """Returns `fit_mb_model` kwargs fixing threshold / APL weights from `param_dict`.
    """
    return {k: param_dict[k] for k in ('fixed_thr', 'wAPLKC', 'wKCAPL')}


def sims_equal(sims1, sims2) -> bool:
    """Returns whether two olfsysm lists of per-odor timeseries are all equal.
    """
    assert len(sims1) == len(sims2)
    return all(np.array_equal(x, y) for x, y in zip(sims1, sims2))


def mean_pn_odor_rate(mp, rv) -> float:
    """Returns mean PN firing rate during odor pulse (across glomeruli and odors).
    """
    ts = get_time_index(mp, rv).values
    mask = (ts >= mp.time_stim_start) & (ts < mp.time_stim_end)
    return float(np.mean([x[:, mask].mean() for x in rv.pn.pn_sims]))


def test_olfsysm_defaults_unchanged(orn_deltas, default_fit):
    """Tests passing olfsysm defaults explicitly does not change (tuned) outputs.
    """
    responses, spike_counts, wPNKC, param_dict = default_fit
    assert not any(k in param_dict for k in AL_INH_PARAMS)
    defaults = olfsysm_defaults()
    assert al_inh_values(param_dict['mp']) == defaults

    responses2, spike_counts2, wPNKC2, param_dict2 = fit_mb_model(orn_deltas,
        **defaults, **MODEL_KWS
    )
    assert responses2.equals(responses)
    assert spike_counts2.equals(spike_counts)
    assert wPNKC2.equals(wPNKC)
    assert param_dict2['fixed_thr'] == param_dict['fixed_thr']
    assert param_dict2['wAPLKC'] == param_dict['wAPLKC']
    assert param_dict2['wKCAPL'] == param_dict['wKCAPL']
    assert param_dict2['tuning_iters'] == param_dict['tuning_iters']

    assert all(type(param_dict2[k]) is float for k in AL_INH_PARAMS)
    assert {k: param_dict2[k] for k in AL_INH_PARAMS} == defaults


@pytest.mark.parametrize('inh_kws', [
    # 2x PN input gain
    dict(pn_inhsc=2 * olfsysm_defaults()['pn_inhsc']),
    # PN input gain much less dependent on LN activity
    dict(pn_inhadd=10 * olfsysm_defaults()['pn_inhadd']),
    dict(ln_inhsc=2 * olfsysm_defaults()['ln_inhsc']),
    dict(ln_inhadd=0.5 * olfsysm_defaults()['ln_inhadd']),
], ids=format_model_params)
def test_al_inh_params_change_outputs(orn_deltas, default_fit, inh_kws):
    """Tests each AL inhibition kwarg sets olfsysm parameter, changing PN / KC outputs.

    Uses threshold / APL weights from `default_fit`, so differences in KC outputs are
    only from the AL changes.
    """
    _, default_spike_counts, _, default_params = default_fit
    default_mp = default_params['mp']
    default_rv = default_params['rv']

    _, spike_counts, _, param_dict = fit_mb_model(orn_deltas,
        return_olfsysm_vars=True, **fixed_param_kws(default_params), **inh_kws,
        **MODEL_KWS
    )
    mp = param_dict['mp']
    rv = param_dict['rv']

    (k, v), = inh_kws.items()
    layer, param = k.split('_')
    assert getattr(getattr(mp, layer), param) == v
    assert al_inh_values(mp) == {**olfsysm_defaults(), k: v}

    assert {x: param_dict[x] for x in AL_INH_PARAMS if x in param_dict} == inh_kws

    # ORN layer is upstream of all of these, and LN layer of the PN parameters
    assert sims_equal(rv.orn.sims, default_rv.orn.sims)
    inhA_changed = not sims_equal(rv.ln.ln_inhA.sims, default_rv.ln.ln_inhA.sims)
    inhB_changed = not sims_equal(rv.ln.ln_inhB.sims, default_rv.ln.ln_inhB.sims)
    assert inhA_changed == inhB_changed == (layer == 'ln')

    assert not sims_equal(rv.pn.pn_sims, default_rv.pn.pn_sims)
    assert not np.array_equal(rv.kc.spont_in, default_rv.kc.spont_in)
    assert not spike_counts.equals(default_spike_counts)

    # only increasing PN gain (pn_inhsc) should increase PN rates. others increase
    # LN activity, or how much of it is needed to reduce PN gain.
    rate = mean_pn_odor_rate(mp, rv)
    default_rate = mean_pn_odor_rate(default_mp, default_rv)
    if k == 'pn_inhsc':
        assert rate > default_rate
    else:
        assert rate < default_rate


def test_negative_al_inh_params(orn_deltas):
    with pytest.raises(ValueError, match='pn_inhadd'):
        fit_mb_model(orn_deltas, pn_inhadd=-1.0, **MODEL_KWS)


def test_fixed_param_kc_simulator_al_inh_params(orn_deltas, default_fit):
    """Tests `make_fixed_param_kc_simulator` w/ AL kwarg equals `fit_mb_model` w/ same.
    """
    _, default_spike_counts, _, default_params = default_fit
    initial_params = {'fixed_thr': default_params['fixed_thr'],
        'wAPLKC_scale': default_params['wAPLKC'],
        'wKCAPL_scale': default_params['wKCAPL'],
    }
    inh_kws = dict(pn_inhadd=2 * olfsysm_defaults()['pn_inhadd'])

    simulate, initial_params2 = make_fixed_param_kc_simulator(orn_deltas,
        initial_params=initial_params, **inh_kws, **MODEL_KWS
    )
    assert initial_params2 == initial_params

    _, spike_counts = simulate(dict())
    # same threshold / APL weights as default_fit, so only differs b/c of PN changes
    assert not spike_counts.equals(default_spike_counts)

    # thr lower and APL weights higher, as in test_fixed_param_kc_simulator
    params = {p: v * (0.9 if p == 'fixed_thr' else 1.3)
        for p, v in initial_params.items()
    }
    responses, spike_counts = simulate(params)

    responses2, spike_counts2, _, param_dict2 = fit_mb_model(orn_deltas,
        **fixed_params_to_fit_mb_model_kws(params), **inh_kws, **MODEL_KWS
    )
    assert responses.equals(responses2)
    assert spike_counts.equals(spike_counts2)
    assert param_dict2['pn_inhadd'] == inh_kws['pn_inhadd']


def test_al_inh_params_model_ids():
    """Tests AL kwargs only change model IDs / LR cache keys when passed (non-None).
    """
    kws = dict(weight_divisor=20)
    none_kws = {**kws, **{k: None for k in AL_INH_PARAMS}}

    model_id = format_model_params(kws)
    assert format_model_params(none_kws) == model_id
    assert format_model_params(none_kws, human=True) == format_model_params(kws,
        human=True
    )
    assert onestep_lr_cache_key(none_kws) == onestep_lr_cache_key(kws)

    model_ids = {model_id}
    lr_keys = {onestep_lr_cache_key(kws)}
    for k in AL_INH_PARAMS:
        k_kws = {**kws, k: 2 * olfsysm_defaults()[k]}
        k_id = format_model_params(k_kws)
        assert k.replace('_', '-') in k_id
        assert k_id.startswith(model_id)
        model_ids.add(k_id)
        lr_keys.add(onestep_lr_cache_key(k_kws))

    assert len(model_ids) == len(lr_keys) == len(AL_INH_PARAMS) + 1

    assert format_model_params({**kws, 'pn_inhsc': 737.3262}, human=True) == (
        'weight_divisor=20, pn_inhsc=737.33'
    )
