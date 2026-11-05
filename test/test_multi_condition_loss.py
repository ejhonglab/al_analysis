#!/usr/bin/env python3
"""Tests for cosine similarity option of `make_kc_data_loss`, and for fitting one set of
model parameters to data from several conditions (e.g. APL intact and blocked), via
`make_multi_condition_loss` and `optimize_mb_model_params` `condition_param_overrides`.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from hong2p.types import ParamDict

from al_analysis import mb_model
from al_analysis.mb_model import (make_kc_data_loss, make_multi_condition_loss,
    optimize_mb_model_params, megamat_orn_deltas, drop_silent_model_cells,
    format_model_params, QUICK_MODEL_KW_LIST
)


# model calls emit many warnings (as in test_mb_model.py)
pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')


@pytest.fixture(scope='module')
def orn_deltas() -> pd.DataFrame:
    """Returns committed megamat ORN spike deltas (see `megamat_orn_deltas`).
    """
    return megamat_orn_deltas()


def _random_responses(n_cells: int = 500, n_odors: int = 6, seed: int = 0
    ) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    odors = pd.Index([f'o{i}' for i in range(n_odors)], name='odor')
    responses = pd.DataFrame((rng.random((n_cells, n_odors)) < 0.1).astype(float),
        columns=odors
    )
    # some silent cells
    responses.iloc[:50] = 0.0
    return responses


def test_kc_data_loss_cosine():
    """Tests similarity='cosine' option of make_kc_data_loss against hand-computed
    values, and that default (Pearson) behavior is unchanged.
    """
    odors = pd.Index(['a', 'b', 'c'], name='odor')
    # cells X odors. |a|=3, |b|=2, |a and b|=1, and no cell responds to c. last cell is
    # silent.
    responses = pd.DataFrame([
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], columns=odors, dtype=float
    )
    cos_ab = 1 / np.sqrt(3 * 2)

    def target(ab: float, ac: float = 0.0, bc: float = 0.0) -> pd.DataFrame:
        return pd.DataFrame([[np.nan, ab, ac], [ab, np.nan, bc], [ac, bc, np.nan]],
            index=odors, columns=odors
        )

    # (similarities w/ c are undefined, and treated as 0)
    for drop_silent_cells in (True, False):
        losses = make_kc_data_loss(target_corr=target(cos_ab), similarity='cosine',
            drop_silent_cells=drop_silent_cells)(responses, responses)
        assert np.isclose(losses['corr_loss'], 0, rtol=0, atol=1e-12)
        assert losses['loss'] == losses['corr_loss']

    losses = make_kc_data_loss(target_corr=target(0.5, ac=0.2), similarity='cosine')(
        responses, responses
    )
    assert np.isclose(losses['corr_loss'],
        np.sqrt(((cos_ab - 0.5)**2 + 0.2**2 + 0) / 3)
    )

    # non-binary (spike counts)
    spike_counts = pd.DataFrame({'a': [2, 1, 0], 'b': [1, 1, 1], 'c': [0, 0, 3]},
        dtype=float
    ).rename_axis(columns='odor')
    cos_ab = (2 + 1) / np.sqrt(5 * 3)
    cos_ac = 0.0
    cos_bc = 3 / np.sqrt(3 * 9)
    losses = make_kc_data_loss(target_corr=target(cos_ab, cos_ac, cos_bc),
        similarity='cosine', corr_from='spike_counts')(spike_counts * 0, spike_counts)
    assert np.isclose(losses['corr_loss'], 0, rtol=0, atol=1e-12)

    responses = _random_responses()
    # 1 - cosine distance, as targets from measured data are computed
    cosine_target = 1 - squareform(pdist(responses.T.values, 'cosine'))
    np.fill_diagonal(cosine_target, np.nan)
    cosine_target = pd.DataFrame(cosine_target, index=responses.columns,
        columns=responses.columns
    )
    cosine_loss_fn = make_kc_data_loss(target_corr=cosine_target, similarity='cosine')
    assert np.isclose(cosine_loss_fn(responses, responses)['corr_loss'], 0, rtol=0,
        atol=1e-12
    )

    pearson_target = drop_silent_model_cells(responses).corr()
    pearson_target.iloc[1, 2] = 0.3
    pearson_target.iloc[2, 1] = 0.3

    # recomputing loss as make_kc_data_loss did before similarity kwarg was added
    model_corr = drop_silent_model_cells(responses).corr().values
    triu_indices = np.triu_indices(len(model_corr), k=1)
    expected_pearson_loss = np.sqrt(np.mean(
        (np.nan_to_num(model_corr[triu_indices], nan=0.0) -
        pearson_target.values[triu_indices])**2
    ))
    assert expected_pearson_loss > 0
    for kws in (dict(), dict(similarity='pearson')):
        losses = make_kc_data_loss(target_corr=pearson_target, **kws)(responses,
            responses
        )
        assert losses['corr_loss'] == expected_pearson_loss
        assert losses['loss'] == expected_pearson_loss

    # cosine and Pearson similarities differ (cosine all >= 0 for binary data)
    assert cosine_loss_fn(responses, responses)['corr_loss'] != make_kc_data_loss(
        target_corr=cosine_target)(responses, responses)['corr_loss']

    with pytest.raises(ValueError):
        make_kc_data_loss(target_corr=cosine_target, similarity='spearman')


def test_multi_condition_loss():
    """Tests make_multi_condition_loss component naming, weights, and total loss.
    """
    responses = _random_responses()
    tnt_responses = _random_responses(seed=1)
    corr = drop_silent_model_cells(responses).corr()

    ctrl_loss_fn = make_kc_data_loss(target_response_rates=0.05)
    tnt_loss_fn = make_kc_data_loss(target_response_rates=0.2, target_corr=corr)
    ctrl_losses = ctrl_loss_fn(responses, responses)
    tnt_losses = tnt_loss_fn(tnt_responses, tnt_responses)

    condition_outputs = {'ctrl': (responses, responses),
        'tnt': (tnt_responses, tnt_responses)
    }
    loss_fn = make_multi_condition_loss({'ctrl': ctrl_loss_fn, 'tnt': tnt_loss_fn})
    losses = loss_fn(condition_outputs)
    assert set(losses) == {'loss', 'ctrl_loss', 'ctrl_response_rate_loss',
        'ctrl_mean_response_rate', 'tnt_loss', 'tnt_response_rate_loss',
        'tnt_mean_response_rate', 'tnt_corr_loss'
    }
    for k, v in ctrl_losses.items():
        assert losses[f'ctrl_{k}'] == v
    for k, v in tnt_losses.items():
        assert losses[f'tnt_{k}'] == v
    assert np.isclose(losses['loss'], ctrl_losses['loss'] + tnt_losses['loss'])
    assert ctrl_losses['loss'] > 0 and tnt_losses['loss'] > 0

    # conditions not in weights have weight 1
    weighted = make_multi_condition_loss({'ctrl': ctrl_loss_fn, 'tnt': tnt_loss_fn},
        weights={'tnt': 2.5})(condition_outputs)
    assert np.isclose(weighted['loss'], ctrl_losses['loss'] + 2.5 * tnt_losses['loss'])
    assert weighted['tnt_loss'] == tnt_losses['loss']

    # loss functions returning floats
    float_losses = make_multi_condition_loss({
            'ctrl': lambda r, s: 1.0, 'tnt': lambda r, s: 2.0
        }, weights={'ctrl': 3.0})(condition_outputs)
    assert float_losses == {'ctrl_loss': 1.0, 'tnt_loss': 2.0, 'loss': 5.0}

    with pytest.raises(ValueError):
        loss_fn({'ctrl': condition_outputs['ctrl']})

    with pytest.raises(ValueError):
        loss_fn({**condition_outputs, 'other': condition_outputs['ctrl']})

    with pytest.raises(ValueError):
        make_multi_condition_loss({'ctrl': ctrl_loss_fn}, weights={'tnt': 1.0})

    with pytest.raises(ValueError):
        make_multi_condition_loss(dict())


def _fake_simulator_factory(calls: List[ParamDict]):
    """Returns stand-in for make_fixed_param_kc_simulator, w/o running the model.

    Response rate decreases w/ threshold and product of APL weight scales. Each
    `simulate` call appends its (complete) params to `calls`.
    """
    odors = pd.Index([f'o{i}' for i in range(4)], name='odor')
    n_cells = 1000

    def fake_make_fixed_param_kc_simulator(orn_deltas, *, initial_params=None,
        **model_kws):

        initial = {'fixed_thr': 1.0, 'wAPLKC_scale': 2.0, 'wKCAPL_scale': 3.0}

        def simulate(params: ParamDict) -> Tuple[pd.DataFrame, pd.DataFrame]:
            unknown = set(params) - set(initial)
            if len(unknown) > 0:
                raise ValueError(f'parameters {unknown} not in {set(initial)}')
            params = {**initial, **params}
            calls.append(params)

            rate = 0.5 / (params['fixed_thr'] *
                (1 + params['wAPLKC_scale'] * params['wKCAPL_scale'])
            )
            n_responders = int(round(rate * n_cells))
            responses = pd.DataFrame(0.0, index=range(n_cells), columns=odors)
            for i in range(len(odors)):
                responses.iloc[i:(i + n_responders), i] = 1.0
            return responses, responses.copy()

        return simulate, dict(initial)

    return fake_make_fixed_param_kc_simulator


def test_condition_param_overrides(monkeypatch):
    """Tests optimize_mb_model_params(condition_param_overrides=...) applies overrides
    after tying APL weights, does not simulate identical conditions twice, and returns
    per-condition parameters / outputs. Uses a fake KC simulator (no model runs).
    """
    calls = []
    monkeypatch.setattr(mb_model, 'make_fixed_param_kc_simulator',
        _fake_simulator_factory(calls)
    )
    model_kws = dict(weight_divisor=20)

    overrides = {'ctrl': {}, 'ctrl_copy': {}, 'tnt': {'wAPLKC_scale': 0.0}}
    loss_fn = make_multi_condition_loss({
        'ctrl': make_kc_data_loss(target_response_rates=0.1),
        'ctrl_copy': make_kc_data_loss(target_response_rates=0.1),
        'tnt': make_kc_data_loss(target_response_rates=0.2),
    })
    maxfev = 6
    ret = optimize_mb_model_params(None, loss_fn=loss_fn,
        condition_param_overrides=overrides,
        params_to_optimize=['fixed_thr', 'wAPLKC_scale'], maxfev=maxfev,
        verify=False, verbose=False, **model_kws
    )
    history = ret['history']
    assert len(history) == maxfev
    assert ret['tied_params'] == ['wKCAPL_scale']

    # one simulation per distinct condition parameters ('ctrl' == 'ctrl_copy')
    assert len(calls) == 2 * len(history)
    for i in range(len(history)):
        candidate, tnt = calls[2 * i], calls[2 * i + 1]
        assert history.iloc[i][list(candidate)].to_dict() == candidate
        # override applied after tying wKCAPL_scale to (candidate) wAPLKC_scale
        assert tnt == {**candidate, 'wAPLKC_scale': 0.0}
        assert np.isclose(candidate['wKCAPL_scale'] / candidate['wAPLKC_scale'], 1.5)

    assert history.wAPLKC_scale.nunique() > 1
    assert (history.wAPLKC_scale > 0).all()

    for c in overrides:
        for k in ('loss', 'response_rate_loss', 'mean_response_rate'):
            assert f'{c}_{k}' in history.columns
    assert np.allclose(history.loss,
        history.ctrl_loss + history.ctrl_copy_loss + history.tnt_loss
    )
    assert (history.tnt_mean_response_rate > history.ctrl_mean_response_rate).all()
    assert history.loss.min() == ret['loss']
    assert ret['losses']['loss'] == ret['loss']

    params = ret['params']
    assert ret['condition_param_overrides'] == overrides
    assert ret['condition_params'] == {'ctrl': params, 'ctrl_copy': params,
        'tnt': {**params, 'wAPLKC_scale': 0.0}
    }
    tnt_kws = ret['condition_fit_mb_model_kws']['tnt']
    assert tnt_kws == {**model_kws, 'fixed_thr': params['fixed_thr'], 'wAPLKC': 0.0,
        'wKCAPL': params['wKCAPL_scale']
    }
    assert ret['condition_fit_mb_model_kws']['ctrl'] == ret['fit_mb_model_kws']

    condition_outputs = ret['condition_outputs']
    assert set(condition_outputs) == set(overrides)
    assert condition_outputs['ctrl_copy'] is condition_outputs['ctrl']
    assert loss_fn(condition_outputs) == ret['losses']
    assert 'responses' not in ret

    # override keys must be optimizable parameters (wAPLKC is the fit_mb_model kwarg).
    # checked before any simulation.
    n_calls = len(calls)
    with pytest.raises(ValueError, match='overrides'):
        optimize_mb_model_params(None, loss_fn=loss_fn,
            condition_param_overrides={**overrides, 'tnt': {'wAPLKC': 0.0}},
            maxfev=2, verify=False, verbose=False
        )
    assert len(calls) == n_calls

    # (not applicable to this model, so rejected by simulator)
    with pytest.raises(ValueError):
        optimize_mb_model_params(None, loss_fn=loss_fn,
            condition_param_overrides={**overrides, 'tnt': {'wAPLPN_scale': 0.0}},
            maxfev=2, verify=False, verbose=False
        )

    with pytest.raises(ValueError, match='loss_fn'):
        optimize_mb_model_params(None, target_response_rates=0.1,
            condition_param_overrides=overrides, maxfev=2, verify=False
        )

    with pytest.raises(ValueError):
        optimize_mb_model_params(None, loss_fn=loss_fn, condition_param_overrides={},
            maxfev=2, verify=False
        )


# smallest/fastest QUICK_MODEL_KW_LIST models (hemibrain, one row per KC, no dynamics)
_quick_kws = [dict(weight_divisor=20, use_connectome_APL_weights=True),
    dict(weight_divisor=20)
]
assert all(kws in QUICK_MODEL_KW_LIST for kws in _quick_kws)

@pytest.mark.parametrize('kws', _quick_kws, ids=format_model_params)
def test_optimize_mb_model_params_apl_knockout(orn_deltas, kws):
    """Tests fitting model to control and APL output blocked conditions together, w/
    wAPLKC_scale=0 in the latter.

    verify=True checks fit_mb_model (w/ wAPLKC=0 for blocked condition) reproduces
    fast simulator outputs in each condition.
    """
    odors = orn_deltas.columns.get_level_values('odor')
    target_cosine = np.full((len(odors), len(odors)), 0.3)
    np.fill_diagonal(target_cosine, np.nan)
    target_cosine = pd.DataFrame(target_cosine, index=odors, columns=odors)

    loss_fn = make_multi_condition_loss({
        'ctrl': make_kc_data_loss(target_response_rates=0.1),
        'tnt': make_kc_data_loss(target_response_rates=0.2, target_corr=target_cosine,
            similarity='cosine'
        ),
    })
    overrides = {'ctrl': {}, 'tnt': {'wAPLKC_scale': 0.0}}
    maxfev = 3
    ret = optimize_mb_model_params(orn_deltas, loss_fn=loss_fn,
        condition_param_overrides=overrides, params_to_optimize=['fixed_thr'],
        maxfev=maxfev, verify=True, **kws
    )
    history = ret['history']
    assert 1 < len(history) <= maxfev
    for c in overrides:
        for k in ('loss', 'response_rate_loss', 'mean_response_rate'):
            assert f'{c}_{k}' in history.columns
    assert 'tnt_corr_loss' in history.columns
    assert np.allclose(history.loss, history.ctrl_loss + history.tnt_loss)
    assert history.fixed_thr.nunique() == len(history)

    # APL output blocked should increase response rate
    assert (history.tnt_mean_response_rate > history.ctrl_mean_response_rate).all()

    condition_outputs = ret['condition_outputs']
    assert set(condition_outputs) == set(overrides)
    ctrl_responses = condition_outputs['ctrl'][0]
    tnt_responses = condition_outputs['tnt'][0]
    assert tnt_responses.values.mean() > ctrl_responses.values.mean()
    assert np.isclose(ctrl_responses.mean().mean(),
        ret['losses']['ctrl_mean_response_rate']
    )

    params = ret['params']
    assert params['wAPLKC_scale'] > 0
    assert ret['condition_params']['tnt'] == {**params, 'wAPLKC_scale': 0.0}
    assert ret['condition_fit_mb_model_kws']['tnt']['wAPLKC'] == 0.0

    # verify=True outputs (fit_mb_model w/ best params fixed) are for first condition
    assert ret['responses'].equals(ctrl_responses)
    assert ret['spike_counts'].equals(condition_outputs['ctrl'][1])
