"""Tests of `connectome_wPNKC(multibouton_claws=..., claw_weight=...)`: how claws whose
synapses come from more than one bouton of their PN are placed in the claw x bouton
matrix, and weighting claws by their PN->KC synapse counts.

The default ('first', 'presence') must reproduce the construction that existed before
these options (each claw in the lowest-ID of its boutons, weight 1).
"""
import numpy as np
import pandas as pd
import pytest

from al_analysis.mb_model import (connectome_wPNKC, claw2bouton_from_wPNKC, KC_ID, CLAW_ID,
    PN_ID, BOUTON_ID
)

pytestmark = pytest.mark.filterwarnings('ignore')
KWS = dict(prat_claws=True, prat_boutons=True)


@pytest.fixture(scope='module')
def default():
    return connectome_wPNKC(**KWS)


@pytest.fixture(scope='module')
def argmax():
    return connectome_wPNKC(multibouton_claws='argmax', **KWS)


@pytest.fixture(scope='module')
def split():
    return connectome_wPNKC(multibouton_claws='split', **KWS)


@pytest.fixture(scope='module')
def synapses():
    return connectome_wPNKC(claw_weight='synapses', **KWS)


@pytest.fixture(scope='module')
def split_synapses():
    return connectome_wPNKC(multibouton_claws='split', claw_weight='synapses', **KWS)


@pytest.fixture(scope='module')
def synapses_raw():
    return connectome_wPNKC(claw_weight='synapses_raw', **KWS)


def claw_keys(w: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays([w.index.get_level_values(KC_ID),
        w.index.get_level_values(CLAW_ID)], names=[KC_ID, CLAW_ID])


def bouton_of(w: pd.DataFrame) -> pd.Series:
    """(KC, claw) -> the one non-zero (pn_id, bouton_id) column of a one-bouton-per-claw matrix."""
    nz = (w > 0)
    assert (nz.sum(axis='columns') == 1).all()
    cols = w.columns.to_frame(index=False)
    idx = nz.to_numpy().argmax(axis=1)
    return pd.Series(list(zip(cols[PN_ID].to_numpy()[idx], cols[BOUTON_ID].to_numpy()[idx])),
        index=claw_keys(w))


def boutons_by_share(split: pd.DataFrame):
    """(KC, claw) -> list of ((pn_id, bouton_id), share) from the split matrix, by share desc."""
    cols = split.columns.to_frame(index=False)
    out = {}
    arr = split.to_numpy()
    for i, key in enumerate(claw_keys(split)):
        nz = np.flatnonzero(arr[i] > 0)
        items = sorted(((cols[PN_ID].iat[j], cols[BOUTON_ID].iat[j]), arr[i, j]) for j in nz)
        out[key] = sorted(items, key=lambda x: (-x[1], x[0]))
    return out


def test_default_is_first_bouton_presence(default, split):
    """Each claw sits in the lowest-ID of its boutons (the boutons `split` uses), weight 1."""
    assert set(np.unique(default)) == {0, 1}
    assert ((default > 0).sum(axis='columns') == 1).all()
    assigned = bouton_of(default)
    by_share = boutons_by_share(split)
    lowest = pd.Series({k: min(b for b, _ in v) for k, v in by_share.items()})
    assert (assigned == lowest.reindex(assigned.index)).all()


def test_argmax_differs_only_on_multibouton_claws(default, argmax, split):
    # columns: the boutons that are some claw's dominant bouton (not the default's set)
    assert argmax.index.equals(default.index)
    assert set(argmax.columns) <= set(split.columns)
    assert set(np.unique(argmax)) == {0, 1}
    assert ((argmax > 0).sum(axis='columns') == 1).all()
    by_share = boutons_by_share(split)
    n_boutons = pd.Series({k: len(v) for k, v in by_share.items()}).reindex(claw_keys(default))
    dominant = pd.Series({k: v[0][0] for k, v in by_share.items()}).reindex(claw_keys(default))
    assigned = bouton_of(argmax)
    assert (assigned == dominant).all()
    cols = argmax.columns.union(default.columns)
    changed = ((argmax > 0).reindex(columns=cols, fill_value=False)
        != (default > 0).reindex(columns=cols, fill_value=False)).any(axis='columns').to_numpy()
    assert changed.any(), 'argmax should move some claws to their dominant bouton'
    assert (n_boutons.to_numpy()[changed] > 1).all(), 'only multibouton claws may change'
    assert changed.sum() < 0.05 * len(default)


def test_split_rows_sum_to_one(default, split):
    # split has every bouton with a claw: a superset of the default's columns (boutons
    # that are never a claw's lowest-ID bouton are absent from the default)
    assert split.index.equals(default.index)
    assert set(default.columns) <= set(split.columns)
    assert len(split.columns) - len(default.columns) < 20
    nz = (split > 0).sum(axis='columns')
    assert (nz >= 1).all() and (nz > 1).any() and (nz > 1).mean() < 0.1
    assert np.allclose(split.sum(axis='columns'), 1.0)
    # the bouton the default used is always one of the split boutons
    d = (default > 0).reindex(columns=split.columns, fill_value=False)
    assert ((split > 0) | ~d).all().all()
    assert (split[nz > 1].max(axis='columns') < 1).all()


def test_synapse_weights_are_integer_counts(default, synapses_raw):
    assert synapses_raw.shape == default.shape and synapses_raw.index.equals(default.index)
    assert ((synapses_raw > 0) == (default > 0)).all().all(), 'same bouton as default'
    tot = synapses_raw.sum(axis='columns')
    assert np.allclose(tot, np.round(tot)) and (tot >= 5).all()   # loader threshold
    assert 10 < tot.median() < 25


def test_synapse_weights_normalized_to_mean_one(default, synapses, synapses_raw):
    assert synapses.shape == default.shape and synapses.index.equals(default.index)
    assert ((synapses > 0) == (default > 0)).all().all(), 'same bouton as default'
    tot = synapses.sum(axis='columns')
    assert np.isclose(tot.mean(), 1.0) and (tot > 0).all()
    assert 0.3 < tot.std() < 0.6   # the spread of claw strength (CV ~0.43)
    raw = synapses_raw.sum(axis='columns')
    assert np.allclose(tot.to_numpy(), (raw / raw.mean()).to_numpy())


def test_split_synapses(split, synapses, split_synapses):
    # mean 1 over the per-connection entries, so row totals are proportional to the
    # per-claw totals of the unsplit synapse weights (whose mean is over claws)
    tot = split_synapses.sum(axis='columns')
    nz = split_synapses.to_numpy()[split_synapses.to_numpy() > 0]
    assert np.isclose(nz.mean(), 1.0)
    per_claw = synapses.sum(axis='columns')
    assert np.allclose((tot / tot.sum()).to_numpy(), (per_claw / per_claw.sum()).to_numpy())
    assert ((split_synapses > 0) == (split > 0)).all().all()
    shares = split_synapses.div(tot, axis='index')
    assert np.allclose(shares.to_numpy(), split.to_numpy())


def test_claw2bouton_on_each(default, argmax, split, synapses, split_synapses):
    for w, weighted, multirow in ((default, False, False), (argmax, False, False),
            (split, True, True), (synapses, True, False), (split_synapses, True, True)):
        c = claw2bouton_from_wPNKC(w)
        assert ('weight' in c.columns) == weighted
        assert len(c) == int((w > 0).sum().sum())
        assert c.index.duplicated().any() == multirow
        if weighted:
            assert np.allclose(c.groupby([KC_ID, CLAW_ID]).weight.sum().to_numpy(),
                w.sum(axis='columns').groupby([KC_ID, CLAW_ID]).sum().to_numpy())


def test_option_validation():
    with pytest.raises(ValueError):
        connectome_wPNKC(prat_claws=True, prat_boutons=False, multibouton_claws='argmax')
    with pytest.raises(ValueError):
        connectome_wPNKC(prat_claws=False, claw_weight='synapses')
    with pytest.raises(ValueError):
        connectome_wPNKC(multibouton_claws='nonsense', **KWS)


BOUTON_MODEL = dict(one_row_per_claw=True, prat_claws=True, prat_boutons=True,
    use_connectome_APL_weights=True, pn_claw_to_apl=True, n_spikes_for_response=2
)


@pytest.mark.parametrize('kws', [
        dict(multibouton_claws='argmax'),
        dict(multibouton_claws='split'),
        dict(claw_weight='synapses'),
        dict(multibouton_claws='split', claw_weight='synapses'),
    ], ids=lambda d: ','.join(f'{k}={v}' for k, v in d.items())
)
def test_fit_mb_model_runs_with_options(kws):
    """`fit_mb_model` simulates the bouton model with each non-default option (fixed
    threshold and APL weights, so nothing is tuned; 4 megamat odors), and the wiring it
    returns has the option's properties. Guards the one-row-per-claw weight checks inside
    `fit_mb_model`, which the `connectome_wPNKC` tests above do not reach.
    """
    from al_analysis.mb_model import fit_mb_model, megamat_orn_deltas

    orn = megamat_orn_deltas().iloc[:, :4]
    responses, spike_counts, wPNKC, params = fit_mb_model(orn, fixed_thr=100.0, wAPLKC=0.1,
        make_plots=False, silent=True, **BOUTON_MODEL, **kws
    )
    assert responses.shape[1] == 4 and spike_counts.shape == responses.shape
    assert len(responses) == wPNKC.index.get_level_values(KC_ID).nunique()
    assert (wPNKC >= 0).all().all() and np.isfinite(wPNKC.to_numpy()).all()
    row_sums = wPNKC.sum(axis='columns')
    if kws.get('claw_weight', 'presence') == 'synapses':
        nz = wPNKC.to_numpy()[wPNKC.to_numpy() > 0]
        assert np.isclose(nz.mean(), 1.0) and 0.3 < row_sums.std() < 0.6
    else:
        assert np.allclose(row_sums, 1.0)
    n_nonzero = (wPNKC > 0).sum(axis='columns')
    if kws.get('multibouton_claws', 'first') == 'split':
        assert (n_nonzero > 1).any() and (n_nonzero > 1).mean() < 0.1
    else:
        assert (n_nonzero == 1).all()
    for k, v in kws.items():
        assert params.get(k, v) == v
