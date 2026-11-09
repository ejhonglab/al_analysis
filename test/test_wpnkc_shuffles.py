#!/usr/bin/env python3
"""Tests for shuffled PN->KC connectivity (`al_analysis.wpnkc_shuffles`).
"""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from al_analysis.mb_model import (connectome_wPNKC, fit_mb_model, megamat_orn_deltas,
    KC_ID, KC_TYPE, CLAW_ID
)
from al_analysis.wpnkc_shuffles import (shuffle_wPNKC, curveball, wPNKC_shuffle_summary,
    shuffled_wPNKC, SHUFFLE_METHODS, NAMED_SHUFFLES
)


pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

WPNKC_FORMATS = ('kc', 'claw')


@pytest.fixture(scope='module')
def wPNKCs():
    """Returns dict of format -> hemibrain wPNKC (do not modify).

    'kc' is one row per KC with claw counts (`weight_divisor=20`), and 'claw' one row
    per claw with 0/1 values (`prat_claws=True`).
    """
    return {
        'kc': connectome_wPNKC(weight_divisor=20),
        'claw': connectome_wPNKC(prat_claws=True),
    }


@pytest.fixture(scope='module')
def orn_deltas() -> pd.DataFrame:
    return megamat_orn_deltas()


def kc_counts(wPNKC: pd.DataFrame) -> pd.DataFrame:
    """Returns # of claws per (KC, glomerulus), with [KC_ID, KC_TYPE] index.
    """
    if CLAW_ID in wPNKC.index.names:
        return wPNKC.groupby(level=[KC_ID, KC_TYPE], sort=False).sum()
    return wPNKC


def test_curveball_margins():
    rng = np.random.default_rng(1)
    binary = (rng.random((40, 12)) < 0.3).astype(np.int64)
    orig = binary.copy()

    shuffled = curveball(binary, np.random.default_rng(0), 200)
    assert np.array_equal(binary, orig)
    assert shuffled.dtype == binary.dtype
    assert not np.array_equal(shuffled, binary)
    assert np.array_equal(shuffled.sum(axis=0), binary.sum(axis=0))
    assert np.array_equal(shuffled.sum(axis=1), binary.sum(axis=1))

    shuffled_bool = curveball(binary.astype(bool), np.random.default_rng(0), 200)
    assert shuffled_bool.dtype == bool
    assert np.array_equal(shuffled_bool, shuffled.astype(bool))

    assert np.array_equal(curveball(binary, rng, 0), binary)
    assert np.array_equal(curveball(binary[:1], rng, 10), binary[:1])

    with pytest.raises(ValueError):
        curveball(binary * 2, rng, 10)

    with pytest.raises(ValueError):
        curveball(binary[0], rng, 10)


def test_curveball_uniform():
    """Tests Curveball samples all matrices with the same margins, roughly uniformly.
    """
    row_sums = np.array([2, 2, 1])
    col_sums = np.array([2, 1, 1, 1])
    n_rows = len(row_sums)
    n_cols = len(col_sums)

    matrices = []
    for bits in product([0, 1], repeat=n_rows * n_cols):
        m = np.array(bits).reshape(n_rows, n_cols)
        if (np.array_equal(m.sum(axis=1), row_sums) and
                np.array_equal(m.sum(axis=0), col_sums)):
            matrices.append(m)
    assert len(matrices) > 5

    keys = [m.tobytes() for m in matrices]
    start = matrices[0]
    rng = np.random.default_rng(0)
    n_samples = 50 * len(matrices)
    observed = pd.Series(
        [curveball(start, rng, 30).tobytes() for _ in range(n_samples)]
    ).value_counts().reindex(keys, fill_value=0)

    assert observed.sum() == n_samples
    assert (observed > 0).all()
    assert chisquare(observed.values).pvalue > 0.001


def check_shuffle_invariants(orig: pd.DataFrame, shuffled: pd.DataFrame, method: str,
    within_kc_type: bool) -> None:

    assert shuffled.index.equals(orig.index)
    assert shuffled.index.names == orig.index.names
    assert shuffled.columns.equals(orig.columns)
    assert shuffled.dtypes.equals(orig.dtypes)
    assert not shuffled.equals(orig)

    if CLAW_ID in orig.index.names:
        assert np.isin(shuffled.values, (0, 1)).all()
        # each claw row still has exactly one 1 (and any all-0 rows stay all-0)
        assert shuffled.sum(axis='columns').equals(orig.sum(axis='columns'))

    orig_counts = kc_counts(orig)
    counts = kc_counts(shuffled)
    assert counts.index.equals(orig_counts.index)
    assert counts.sum(axis='columns').equals(orig_counts.sum(axis='columns'))

    if method == 'claw_permutation':
        to_compare = [(orig_counts, counts)]
    else:
        to_compare = [(orig_counts > 0, counts > 0)]
        assert (counts > 0).sum(axis='columns').equals(
            (orig_counts > 0).sum(axis='columns')
        )
        # multiplicity pattern of each KC's claws preserved
        assert np.array_equal(np.sort(counts.values, axis=1),
            np.sort(orig_counts.values, axis=1)
        )

    for orig_x, x in to_compare:
        # claws (claw_permutation) or KCs (curveball) per glomerulus
        assert x.sum().equals(orig_x.sum())

        per_type = x.groupby(level=KC_TYPE).sum()
        orig_per_type = orig_x.groupby(level=KC_TYPE).sum()
        if within_kc_type:
            assert per_type.equals(orig_per_type)
        else:
            assert not per_type.equals(orig_per_type)

    summary = wPNKC_shuffle_summary(orig, shuffled)
    assert summary['n_claws'] == orig_counts.values.sum()
    assert summary['kc_n_claws_preserved']
    assert summary['frac_claws_changed'] > 0.5
    if method == 'claw_permutation':
        assert summary['max_abs_glom_n_claws_change'] == 0
    else:
        assert summary['kc_n_glomeruli_preserved']
        assert summary['max_abs_glom_n_kcs_change'] == 0

    assert summary['coinput_corr'] < 0.97
    # with all KCs shuffled together, nothing beyond glomerulus frequencies should be
    # left. within KC types, a fair amount of co-input structure is (~0.4 for hemibrain)
    if within_kc_type:
        assert 0.1 < summary['coinput_resid_corr'] < 0.8
    else:
        assert abs(summary['coinput_resid_corr']) < 0.2


@pytest.mark.parametrize('within_kc_type', [False, True])
@pytest.mark.parametrize('method', SHUFFLE_METHODS)
@pytest.mark.parametrize('fmt', WPNKC_FORMATS)
def test_shuffle_invariants(wPNKCs, fmt, method, within_kc_type):
    wPNKC = wPNKCs[fmt]
    orig = wPNKC.copy()

    shuffled = shuffle_wPNKC(wPNKC, method, within_kc_type=within_kc_type)
    assert wPNKC.equals(orig)
    check_shuffle_invariants(wPNKC, shuffled, method, within_kc_type)


@pytest.mark.parametrize('method', SHUFFLE_METHODS)
@pytest.mark.parametrize('fmt', WPNKC_FORMATS)
def test_shuffle_all0_rows(wPNKCs, fmt, method):
    """Tests all-0 rows (KCs / claws without input) are left as is.
    """
    wPNKC = wPNKCs[fmt].copy()
    all0 = np.zeros(len(wPNKC), dtype=bool)
    all0[::50] = True
    wPNKC.iloc[all0] = 0

    shuffled = shuffle_wPNKC(wPNKC, method)
    assert (shuffled.iloc[all0] == 0).all().all()
    check_shuffle_invariants(wPNKC, shuffled, method, False)


def test_shuffle_summary_unshuffled(wPNKCs):
    for wPNKC in wPNKCs.values():
        summary = wPNKC_shuffle_summary(wPNKC, wPNKC.copy())
        assert summary['frac_claws_changed'] == 0
        assert summary['kc_n_claws_preserved']
        assert summary['kc_n_glomeruli_preserved']
        assert summary['max_abs_glom_n_claws_change'] == 0
        assert summary['max_abs_glom_n_kcs_change'] == 0
        assert np.isclose(summary['coinput_corr'], 1)
        assert np.isclose(summary['coinput_resid_corr'], 1)


@pytest.mark.parametrize('method', SHUFFLE_METHODS)
@pytest.mark.parametrize('fmt', WPNKC_FORMATS)
def test_shuffle_seeds(wPNKCs, fmt, method):
    wPNKC = wPNKCs[fmt]
    shuffled = shuffle_wPNKC(wPNKC, method, seed=3)
    assert shuffle_wPNKC(wPNKC, method, seed=3).equals(shuffled)
    assert not shuffle_wPNKC(wPNKC, method, seed=4).equals(shuffled)

    shuffled = shuffle_wPNKC(wPNKC, method, seed=3, within_kc_type=True)
    assert shuffle_wPNKC(wPNKC, method, seed=3, within_kc_type=True).equals(shuffled)
    assert not shuffle_wPNKC(wPNKC, method, seed=4, within_kc_type=True
        ).equals(shuffled)


@pytest.mark.parametrize('fmt', WPNKC_FORMATS)
def test_curveball_n_iters(wPNKCs, fmt):
    wPNKC = wPNKCs[fmt]
    orig_binary = kc_counts(wPNKC) > 0

    # each KC's glomeruli are still randomly re-assigned to its claws, but the binary
    # KC x glomerulus matrix is unchanged
    shuffled = shuffle_wPNKC(wPNKC, 'curveball', n_iters=0)
    assert (kc_counts(shuffled) > 0).equals(orig_binary)
    assert not shuffled.equals(wPNKC)

    few = wPNKC_shuffle_summary(wPNKC,
        shuffle_wPNKC(wPNKC, 'curveball', n_iters=len(orig_binary) // 10)
    )
    default = wPNKC_shuffle_summary(wPNKC, shuffle_wPNKC(wPNKC, 'curveball'))
    assert few['coinput_resid_corr'] > default['coinput_resid_corr']


@pytest.mark.parametrize('method', SHUFFLE_METHODS)
def test_shuffle_bool_claws(wPNKCs, method):
    wPNKC = wPNKCs['claw'].astype(bool)
    shuffled = shuffle_wPNKC(wPNKC, method)
    assert (shuffled.dtypes == bool).all()
    assert shuffled.astype(np.int64).equals(
        shuffle_wPNKC(wPNKCs['claw'], method)
    )


def test_shuffled_wPNKC(wPNKCs):
    claw_kws = dict(one_row_per_claw=True, prat_claws=True,
        use_connectome_APL_weights=True, n_spikes_for_response=2
    )
    for name, (method, within_kc_type) in NAMED_SHUFFLES.items():
        for fmt, kws in [('kc', dict(weight_divisor=20)), ('claw', claw_kws)]:
            expected = shuffle_wPNKC(wPNKCs[fmt], method, within_kc_type=within_kc_type,
                seed=2
            )
            assert shuffled_wPNKC(name, seed=2, **kws).equals(expected)

    assert shuffled_wPNKC('curveball', pn2kc_connections='hemibrain', weight_divisor=20
        ).equals(shuffle_wPNKC(wPNKCs['kc'], 'curveball'))

    with pytest.raises(ValueError):
        shuffled_wPNKC('not_a_shuffle', weight_divisor=20)

    with pytest.raises(ValueError):
        shuffled_wPNKC('curveball', pn2kc_connections='fafb-left', weight_divisor=12)


def test_shuffle_errors(wPNKCs):
    wPNKC = wPNKCs['kc']

    with pytest.raises(ValueError):
        shuffle_wPNKC(wPNKC, 'not_a_method')

    with pytest.raises(ValueError):
        shuffle_wPNKC(wPNKC, 'claw_permutation', n_iters=10)

    with pytest.raises(ValueError):
        shuffle_wPNKC(wPNKC.droplevel(KC_TYPE), within_kc_type=True)

    # (without within_kc_type=True, KC_TYPE level is not needed)
    shuffle_wPNKC(wPNKC.droplevel(KC_TYPE))

    with pytest.raises(ValueError):
        shuffle_wPNKC(wPNKC / 2)

    with pytest.raises(ValueError):
        shuffle_wPNKC(wPNKC.astype(bool))

    claw_wPNKC = wPNKCs['claw'].copy()
    claw_wPNKC.iloc[0] = 1
    with pytest.raises(ValueError):
        shuffle_wPNKC(claw_wPNKC)


def test_shuffled_wPNKC_model_one_row_per_kc(wPNKCs, orn_deltas):
    """Tests `fit_mb_model` runs with shuffled one-row-per-KC wPNKC (`_wPNKC`).
    """
    wPNKC = wPNKCs['kc']
    target_sparsity = 0.1
    kws = dict(target_sparsity=target_sparsity, make_plots=False)

    responses, _, wPNKC_used, _ = fit_mb_model(orn_deltas, _wPNKC=wPNKC, **kws)
    assert wPNKC_used.equals(wPNKC)

    shuffled = shuffle_wPNKC(wPNKC, 'curveball')
    shuffled_responses, _, shuffled_wPNKC_used, _ = fit_mb_model(orn_deltas,
        _wPNKC=shuffled, **kws
    )
    assert shuffled_wPNKC_used.equals(shuffled)

    assert shuffled_responses.index.equals(responses.index)
    assert shuffled_responses.columns.equals(responses.columns)
    assert not shuffled_responses.equals(responses)
    assert np.isclose(shuffled_responses.mean().mean(), target_sparsity, rtol=0.15)


@pytest.mark.parametrize('kws', [
    dict(),
    # "claw-dynamics" model (thesis chapter 3)
    dict(pn_claw_to_apl=True, claw_dynamics=True),
])
def test_shuffled_wPNKC_model_one_row_per_claw(wPNKCs, orn_deltas, kws):
    """Tests `fit_mb_model` runs with shuffled hemibrain claw wPNKC (`_wPNKC`).

    Connectome APL weights stay with their claws (rows), whose glomeruli change.
    """
    wPNKC = wPNKCs['claw']
    shuffled = shuffle_wPNKC(wPNKC, 'claw_permutation')

    target_sparsity = 0.05
    responses, _, wPNKC_used, _ = fit_mb_model(orn_deltas, _wPNKC=shuffled,
        one_row_per_claw=True, prat_claws=True, use_connectome_APL_weights=True,
        target_sparsity=target_sparsity, n_spikes_for_response=2, make_plots=False,
        **kws
    )
    assert wPNKC_used.equals(shuffled)

    kc_index = wPNKC.groupby(level=[KC_ID, KC_TYPE], sort=False).size().index
    assert responses.index.equals(kc_index)
    assert responses.columns.get_level_values('odor').equals(
        orn_deltas.columns.get_level_values('odor')
    )
    assert np.isclose(responses.mean().mean(), target_sparsity, rtol=0.15)


def _small_claw_wPNKC(multi_columns: bool) -> pd.DataFrame:
    """3 KCs: KC0 claws A, B; KC1 claws A, B, B; KC2 claws B, C (type g, g, ab)."""
    rows = [(0, 0, 'g'), (0, 1, 'g'), (1, 0, 'g'), (1, 1, 'g'), (1, 2, 'g'), (2, 0, 'ab'),
        (2, 1, 'ab')
    ]
    gloms = ['A', 'B', 'A', 'B', 'B', 'B', 'C']
    if multi_columns:
        columns = pd.MultiIndex.from_tuples([('A', 1, 0), ('B', 2, 0), ('B', 2, 1), ('C', 3, 0)],
            names=['glomerulus', 'pn_id', 'bouton_id']
        )
        # KC1's two B claws on different boutons of B
        col_of = {('A', 0): 0, ('A', 1): 0, ('B', 0): 1, ('B', 1): 1, ('B', 2): 2, ('C', 2): 3}
        cols = [col_of[(g, kc)] for g, (kc, _, _) in zip(gloms, rows)]
    else:
        columns = pd.Index(['A', 'B', 'C'], name='glomerulus')
        cols = [columns.get_loc(g) for g in gloms]
    values = np.zeros((len(rows), len(columns)), dtype=np.int64)
    values[np.arange(len(rows)), cols] = 1
    index = pd.MultiIndex.from_tuples(rows, names=[KC_ID, CLAW_ID, KC_TYPE])
    return pd.DataFrame(values, index=index, columns=columns)


@pytest.mark.parametrize('multi_columns', [False, True])
def test_coinput_counts_small(multi_columns):
    from al_analysis.wpnkc_shuffles import coinput_counts

    co = coinput_counts(_small_claw_wPNKC(multi_columns))
    expected = pd.DataFrame([[2, 2, 0], [2, 3, 1], [0, 1, 1]], index=['A', 'B', 'C'],
        columns=['A', 'B', 'C']
    )
    assert co.index.tolist() == ['A', 'B', 'C']
    assert np.array_equal(co.to_numpy(), expected.to_numpy())


def test_coinput_counts_hemibrain(wPNKCs):
    from al_analysis.wpnkc_shuffles import coinput_counts

    wPNKC = wPNKCs['claw']
    co = coinput_counts(wPNKC)
    assert np.array_equal(co.to_numpy(), co.to_numpy().T)
    # the diagonal is each glomerulus' number of KCs
    n_kcs = (kc_counts(wPNKC) > 0).sum()
    assert np.array_equal(np.diag(co.to_numpy()), n_kcs.reindex(co.index).to_numpy())
    # a claw permutation keeps the diagonal roughly and the shuffle summary's co-input
    # correlation is computed from the same counts
    co_kc = coinput_counts(wPNKCs['kc'])
    assert co_kc.index.equals(wPNKCs['kc'].columns)


@pytest.mark.parametrize('within_kc_type', [True, False])
def test_lesion_coinput(wPNKCs, within_kc_type):
    from al_analysis.wpnkc_shuffles import coinput_counts, lesion_coinput

    wPNKC = wPNKCs['claw']
    co = coinput_counts(wPNKC)
    off = co.to_numpy().copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    top = (co.index[i], co.index[j])
    # a second pair to reduce, and one to protect
    i2, j2 = np.unravel_index(np.argsort(off.ravel())[-4], off.shape)
    second = (co.index[i2], co.index[j2])
    i3, j3 = np.unravel_index(np.argsort(off.ravel())[-9], off.shape)
    protect = (co.index[i3], co.index[j3])
    targets = {top: int(off[i, j]) - 6, second: int(off[i2, j2]) - 3}

    lesioned, swaps = lesion_coinput(wPNKC, targets, seed=1, within_kc_type=within_kc_type,
        protect=[protect]
    )
    assert lesioned.index.equals(wPNKC.index) and lesioned.columns.equals(wPNKC.columns)
    assert lesioned.dtypes.equals(wPNKC.dtypes)
    co2 = coinput_counts(lesioned)
    # the listed pairs reached their targets (a later pair's swaps may take an earlier one
    # lower, never higher) and the protected pair did not grow
    assert co2.loc[top] <= targets[top] and co2.loc[second] <= targets[second]
    assert co2.loc[top] >= targets[top] - 3
    assert co2.loc[protect] == co.loc[protect]
    # every margin kept: claws per KC, claws per glomerulus (per type if asked)
    assert (lesioned.groupby(level=KC_ID).sum().sum(axis=1)
        == wPNKC.groupby(level=KC_ID).sum().sum(axis=1)).all()
    assert lesioned.sum().equals(wPNKC.sum())
    if within_kc_type:
        assert lesioned.groupby(level=KC_TYPE).sum().equals(wPNKC.groupby(level=KC_TYPE).sum())
    summary = wPNKC_shuffle_summary(wPNKC, lesioned)
    assert summary['kc_n_claws_preserved'] and summary['max_abs_glom_n_claws_change'] == 0
    # the swaps table accounts for every changed claw
    changed = (lesioned.to_numpy() != wPNKC.to_numpy()).any(axis=1)
    assert set(np.flatnonzero(changed)) == set(swaps.claw_row) | set(swaps.partner_row)
    assert len(swaps) > 0 and (swaps.moved != swaps.received).all()
    # reproducible
    again, _ = lesion_coinput(wPNKC, targets, seed=1, within_kc_type=within_kc_type,
        protect=[protect]
    )
    assert again.equals(lesioned)
    # a target at the current count changes nothing
    same, no_swaps = lesion_coinput(wPNKC, {top: int(off[i, j])}, seed=1)
    assert same.equals(wPNKC) and no_swaps.empty


def test_lesion_coinput_errors(wPNKCs):
    from al_analysis.wpnkc_shuffles import lesion_coinput

    with pytest.raises(ValueError, match='one-row-per-claw'):
        lesion_coinput(wPNKCs['kc'], {('DM1', 'DM4'): 0})

    with pytest.raises(ValueError, match='not two glomeruli'):
        lesion_coinput(wPNKCs['claw'], {('DM1', 'no_such_glomerulus'): 0})

    with pytest.raises(ValueError, match='twice'):
        lesion_coinput(wPNKCs['claw'], {('DM1', 'DM1'): 0})
