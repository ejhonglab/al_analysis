#!/usr/bin/env python3
"""Tests for the repeat statistics of the KC-remapping harness (the flies' variability across the three
repeats, E81: scripts/claude_pair_remapping_experiments/kc_remapping_harness.py): single-trial distances by
kind of trial pair, the same-repeat gap, the noise-free distances from products of different repeats and the
parts of a response's variability, on synthetic recordings whose structure is known."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
H = pytest.importorskip('kc_remapping_harness')

ODORS = list(H.APL16_ODORS)


def _signal(seed=0, n_units=3000, n_odors=16, p_resp=0.12):
    """A KC-like noise-free signal (units x odors): sparse positive responses, odors sharing responders through
    a few latent groups, so that the pair distances span a range."""
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 4, n_odors)
    unit_pref = rng.random((n_units, 4)) ** 6
    p = np.clip(p_resp * unit_pref[:, groups] / unit_pref.mean(), 0, 1)   # odors of a group share their responders
    resp = rng.random((n_units, n_odors)) < p
    amp = 2.0 + rng.gamma(2.0, 0.6, (n_units, n_odors))
    return np.where(resp, amp, 0.0)


def _true_distances(m, odors=ODORS):
    return H.pair_distances(pd.DataFrame(m, columns=odors), 'cosine')


def _recording(m, *, seed=1, n_rep=3, sd_indep=0.6, sd_shared=0.0, sd_gain=0.0, return_parts=False):
    """Trials (n_rep, units, odors): x = g[r, o] * m + u[r, k] + e[r, k, o], with g a gain per presentation
    (1 + sd_gain * N(0, 1)), u a state shared by one unit's odors within a repeat, e independent noise; with
    `return_parts` also (g, u, e) as drawn."""
    rng = np.random.default_rng(seed)
    U, O = m.shape
    g = 1.0 + sd_gain * rng.standard_normal((n_rep, 1, O))
    u = sd_shared * rng.standard_normal((n_rep, U, 1))
    e = sd_indep * rng.standard_normal((n_rep, U, O))
    X = g * m[None] + u + e
    return (X, g, u, e) if return_parts else X


def _dev_var(x):
    """Per-trial variance of the deviations from the mean over repeats (axis 0), unbiased."""
    return ((x - x.mean(axis=0)) ** 2).sum(axis=0) / (x.shape[0] - 1)


def test_trial_values_and_cosines():
    X = np.array([[[0.5, 2.0], [3.0, -1.0], [0.0, 1.5]]])           # 1 repeat, 3 units, 2 odors
    assert np.array_equal(H.trial_values(X, 'thresholded', 1.0), [[[0, 2.0], [3.0, 0], [0, 1.5]]])
    assert np.array_equal(H.trial_values(X, 'binary', 1.0), [[[0, 1.0], [1.0, 0], [0, 1.0]]])
    C = H.trial_cosines(X)
    assert C.shape == (1, 2, 1, 2)
    a, b = X[0, :, 0], X[0, :, 1]
    assert np.isclose(C[0, 0, 0, 1], a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    Z = np.zeros((2, 3, 2))
    Z[0, 0, 0] = 1.0
    C = H.trial_cosines(Z)
    assert np.isnan(C[1, 0, 0, 0]) and np.isclose(C[0, 0, 0, 0], 1.0)


def test_distances_by_kind_layout_and_kinds():
    m = _signal(n_units=500)
    X = _recording(m, sd_indep=0.5)
    bk = H.distances_by_kind(X, ODORS)
    assert list(bk.columns) == list(H.REPEAT_KINDS)
    assert len(bk) == 120 + 16
    same = bk.index.get_level_values(0) == bk.index.get_level_values(1)
    assert bk.loc[same, 'same_repeat'].isna().all()
    assert bk.loc[~same].notna().all().all()
    # by hand, one pair: different repeats = mean of the six ordered (i != j) single-trial distances
    i_a, i_b = ODORS.index('2h'), ODORS.index('eb')
    cos = lambda x, y: x @ y / np.linalg.norm(x) / np.linalg.norm(y)
    diff = np.mean([1 - cos(X[i, :, i_a], X[j, :, i_b]) for i in range(3) for j in range(3) if i != j])
    adj = np.mean([1 - cos(X[i, :, i_a], X[j, :, i_b]) for i in range(3) for j in range(3) if abs(i - j) == 1])
    same_r = np.mean([1 - cos(X[i, :, i_a], X[i, :, i_b]) for i in range(3)])
    assert np.isclose(bk.loc[('2h', 'eb'), 'different_repeat'], diff)
    assert np.isclose(bk.loc[('2h', 'eb'), 'adjacent'], adj)
    assert np.isclose(bk.loc[('2h', 'eb'), 'same_repeat'], same_r)
    assert np.isclose(H.same_repeat_gap(bk, [('eb', '2h')]), diff - same_r)
    # two repeats: no first-against-third
    bk2 = H.distances_by_kind(X[:2], ODORS)
    assert bk2['first_third'].isna().all()


def test_independent_noise_only_agrees_with_correction_in_use():
    """With noise independent between odors and repeats, the same-repeat gap is about 0, and the noise-free
    estimate, the correction in use and the true distances agree within sampling error."""
    m = _signal(seed=2)
    true = _true_distances(m)
    X = _recording(m, seed=3, sd_indep=1.0)
    tab = H.repeat_pair_table(X, ODORS)
    gap = H.same_repeat_gap(H.distances_by_kind(X, ODORS))
    assert abs(gap) < 0.005
    # the noise inflates the trial-mean distances well beyond the sampling error
    assert (tab['trial_mean'] - true).mean() > 0.025
    for col in ('noise_free', 'corrected', 'noise_free_adjacent', 'noise_free_first_third'):
        err = tab[col] - true.reindex(tab.index)
        assert abs(err.mean()) < 0.01, col
        assert np.sqrt((err ** 2).mean()) < 0.03, col
    assert abs((tab['noise_free'] - tab['corrected']).mean()) < 0.01
    assert np.sqrt(((tab['noise_free'] - tab['corrected']) ** 2).mean()) < 0.03
    nf = H.noise_free_pair_distances(X, ODORS)
    assert np.allclose(nf.to_numpy(), tab['noise_free'].to_numpy())


def test_shared_state_only_new_estimate_returns_true_distances():
    """With a state shared by one unit's odors within a repeat, two odors of the same repeat are closer than
    two odors of different repeats (a positive gap), the correction in use is biased low, and the noise-free
    estimate from products of different repeats still returns the true distances."""
    m = _signal(seed=4)
    true = _true_distances(m)
    X = _recording(m, seed=5, sd_indep=0.5, sd_shared=0.5)
    tab = H.repeat_pair_table(X, ODORS)
    gap = H.same_repeat_gap(H.distances_by_kind(X, ODORS))
    assert gap > 0.02
    err_nf = tab['noise_free'] - true.reindex(tab.index)
    err_c = tab['corrected'] - true.reindex(tab.index)
    assert abs(err_nf.mean()) < 0.01
    assert np.sqrt((err_nf ** 2).mean()) < 0.03
    assert err_c.mean() < -0.03                                  # the correction makes the odors too alike
    # a gain per presentation, which a cosine does not see, changes neither estimate's agreement
    Xg = _recording(m, seed=5, sd_indep=0.5, sd_shared=0.5, sd_gain=0.2)
    err_g = H.noise_free_pair_distances(Xg, ODORS) - true
    assert abs(err_g.mean()) < 0.01


def test_drift_cancels_in_noise_free_ratio():
    """A drift of each odor's response pattern over the repeats (a fixed step per repeat, its own direction for
    each odor) acts on the same-odor and the different-odor products alike: the noise-free estimates from
    adjacent repeats and from the first against the third agree."""
    rng = np.random.default_rng(6)
    m = _signal(seed=6)
    U, O = m.shape
    # each odor's response pattern changes by a fixed step per repeat (a random direction per odor)
    step = 0.15 * np.where(m > 0, 1.0, 0.0) * rng.standard_normal((U, O))
    X = np.stack([m + r * step for r in (-1, 0, 1)]) + 0.5 * rng.standard_normal((3, U, O))
    tab = H.repeat_pair_table(X, ODORS)
    assert abs((tab['noise_free_adjacent'] - tab['noise_free_first_third']).mean()) < 0.02


def test_variance_parts_return_known_shares():
    """A synthetic recording with a gain per presentation, a state shared by a unit's odors within a repeat and
    independent noise, of known sizes: the parts come back."""
    m = _signal(seed=7, n_units=6000)
    sd_g, sd_u, sd_e = 0.15, 0.3, 0.5
    X, g, u, e = _recording(m, seed=8, n_rep=3, sd_indep=sd_e, sd_shared=sd_u, sd_gain=sd_g, return_parts=True)
    vp = H.repeat_variance_parts(X, ODORS, thresh=1.0, n_amp_bins=3)
    assert {'all', 'responders', 'silent', 'responders_amp1', 'responders_amp3'} <= set(vp.index)
    # the parts as drawn (48 gains, 18000 shared states: the realized sizes differ from sd_g, sd_u by sampling)
    v_gain = _dev_var(g * m[None])                               # per entry
    v_u = float(_dev_var(u).mean())
    v_e = float(_dev_var(e).mean())
    s = vp.loc['silent']
    assert np.isclose(s.var_shared, v_u, rtol=0.1)
    assert np.isclose(s.var_total, v_u + v_e, rtol=0.05)
    assert s.share_gain < 0.02
    assert np.isclose(s.corr_shared, v_u / (v_u + v_e), rtol=0.1)
    r = vp.loc['responders']
    resp = X.mean(axis=0) > 1.0
    assert np.isclose(r.var_gain, v_gain[resp].mean(), rtol=0.1)
    assert np.isclose(r.var_shared, v_u, rtol=0.15)
    assert np.isclose(r.var_total, v_gain[resp].mean() + v_u + v_e, rtol=0.05)
    assert np.isclose(r.share_gain + r.share_shared + r.share_rest, 1.0)
    # the gain part scales with the amplitude, the shared part does not
    assert vp.loc['responders_amp3', 'var_gain'] > 2 * vp.loc['responders_amp1', 'var_gain']
    assert np.isclose(vp.loc['responders_amp3', 'var_shared'], vp.loc['responders_amp1', 'var_shared'], rtol=0.3)
    # without a shared state, the shared part is about 0
    vp0 = H.repeat_variance_parts(_recording(m, seed=9, sd_indep=sd_e), ODORS, thresh=1.0)
    assert abs(vp0.loc['silent', 'corr_shared']) < 0.02
    assert abs(vp0.loc['responders', 'corr_shared']) < 0.05


def test_coresponder_deviation_corr():
    m = _signal(seed=10, n_units=6000)
    sd_u, sd_e = 0.4, 0.4
    X = _recording(m, seed=11, sd_indep=sd_e, sd_shared=sd_u)
    c = H.coresponder_deviation_corr(X, ODORS, thresh=1.0)
    assert len(c) == 120
    assert np.isclose(np.nanmedian(c), sd_u ** 2 / (sd_u ** 2 + sd_e ** 2), atol=0.08)
    c0 = H.coresponder_deviation_corr(_recording(m, seed=12, sd_indep=sd_e), ODORS, thresh=1.0)
    assert abs(np.nanmedian(c0)) < 0.08
    # a gain common to the whole repeat correlates co-responders' deviations; removing the gain part removes it
    rng = np.random.default_rng(13)
    g = (1 + 0.3 * rng.standard_normal(3))[:, None, None]
    Xg = g * m[None] + sd_e * rng.standard_normal((3,) + m.shape)
    assert np.nanmedian(H.coresponder_deviation_corr(Xg, ODORS, thresh=1.0)) > 0.3
    assert abs(np.nanmedian(H.coresponder_deviation_corr(Xg, ODORS, thresh=1.0, remove_gain=True))) < 0.1


def test_repeat_response_stats():
    X = np.zeros((3, 10, 2))
    X[0, :4, 0] = 2.0
    X[2, :1, 1] = 3.0
    st = H.repeat_response_stats(X, ['2h', 'eb'], thresh=1.0)
    assert st.loc[(0, '2h'), 'rate'] == 0.4 and st.loc[(0, '2h'), 'amplitude'] == 2.0
    assert st.loc[(2, 'eb'), 'n_resp'] == 1 and np.isnan(st.loc[(1, 'eb'), 'amplitude'])


# ---------------------------------------------------------------------------
# part A6, the stimulus order: matching the order to the trials, the separation of two presentations, the
# same-repeat gap's separation and block parts, and the random orders of the null
E81 = pytest.importorskip('E81_repeat_structure')
STIM = ODORS + ['pfo']


def _order(seed=0, rec=('2024-01-01', 1, 'L'), genotype='TNT3c', n_blocks=3, within=140, across=170):
    """A synthetic order table of one recording: per block the 16 odors in a random order, the solvent last;
    onsets `within` volumes apart in a block and `across` volumes across a block boundary."""
    rng = np.random.default_rng(seed)
    rows, onset, p = [], 30, 0
    for b in range(n_blocks):
        for pos, o in enumerate(list(rng.permutation(ODORS)) + ['pfo']):
            if p:
                onset += across if pos == 0 else within
            rows.append(dict(recording=f'{rec[0]}_{rec[1]}{rec[2]}', genotype=genotype, block=b, position=pos,
                             odor=f'{o} @ -3', date=rec[0], fly_num=rec[1], hemibrain=rec[2], presentation=p,
                             odor_abbrev=o, repeat=b, onset_volume=onset))
            p += 1
    return pd.DataFrame(rows)


def _keys(order):
    k = order[['genotype', 'date', 'fly_num', 'hemibrain', 'odor', 'block']].rename(columns={'block': 'repeat'})
    return k.sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_match_order_synthetic():
    order = pd.concat([_order(0), _order(1, rec=('2024-01-02', 3, 'R'), genotype='TNT')], ignore_index=True)
    keys = _keys(order)
    m, s = E81.match_order(order, keys)
    assert s['matched'] == s['order_presentations'] == s['olfdb_trials'] == 102
    assert s['order_only'] == s['olfdb_only'] == s['order_repeat_not_block'] == s['genotype_disagrees'] == 0
    assert s['olfdb_trials_matched_more_than_once'] == 0
    assert s['recordings_both'] == 2 and s['each_block_each_stimulus_once']
    assert (m['olfdb_repeat'] == m['block']).all()
    # one olfdb trial with a repeat the order does not have: one presentation and one trial unmatched
    k2 = keys.copy()
    i = k2.index[(k2.odor == 'eb @ -3') & (k2.repeat == 2) & (k2.genotype == 'TNT')][0]
    k2.loc[i, 'repeat'] = 3
    m2, s2 = E81.match_order(order, k2)
    assert s2['matched'] == 101 and s2['order_only'] == 1 and s2['olfdb_only'] == 1
    assert s2['presentations_same_count'] == 102                 # still three trials of eb in that recording
    # an odor presented twice in one block and not in the next: its count of earlier presentations is not the block
    o3 = _order(2).copy()
    a = o3.index[(o3.block == 1) & (o3.odor_abbrev == '2h')][0]
    b = o3.index[(o3.block == 0) & (o3.odor_abbrev == 'IaA')][0]
    o3.loc[b, ['odor', 'odor_abbrev']] = ['2h @ -3', '2h']
    o3.loc[a, ['odor', 'odor_abbrev']] = ['IaA @ -3', 'IaA']
    o3['repeat'] = o3.groupby('odor').cumcount()
    m3, s3 = E81.match_order(o3, _keys(_order(2)))
    assert not s3['each_block_each_stimulus_once']
    assert s3['order_repeat_not_block'] > 0
    # the two presentations of 2h in block 0 take its first trial, its second is left (likewise IaA)
    assert s3['olfdb_only'] == 2 and s3['olfdb_trials_matched_more_than_once'] == 2 and s3['order_only'] == 0


def test_presentation_pairs_separations_and_distances():
    order = _order(3)
    o = order.rename(columns={'odor': 'odor_name'}).rename(columns={'odor_abbrev': 'odor'})
    o['repeat'] = (o['block'] + 1) % 3                           # the trial index need not be the block
    rng = np.random.default_rng(4)
    X = rng.standard_normal((3, 200, 16))
    t = H.presentation_pairs(X, ODORS, o, volume_rate=2.0)
    assert len(t) == 48 * 47 // 2 - 16 * 3
    assert t.groupby(['odor_a', 'odor_b']).size().eq(9).all()
    assert t.groupby(['odor_a', 'odor_b'])['same_block'].sum().eq(3).all()
    assert all(ODORS.index(a) < ODORS.index(b) for a, b in zip(t.odor_a, t.odor_b))
    by_p = o.set_index('presentation')
    for _, r in t.sample(40, random_state=0).iterrows():
        pa, pb = by_p.loc[r.presentation_a], by_p.loc[r.presentation_b]
        assert pa.odor == r.odor_a and pb.odor == r.odor_b
        assert r.separation == abs(r.presentation_a - r.presentation_b)
        assert r.same_block == (pa.block == pb.block) and r.block_a == pa.block
        assert np.isclose(r.separation_s, abs(pa.onset_volume - pb.onset_volume) / 2.0)
        x, y = X[pa['repeat'], :, ODORS.index(r.odor_a)], X[pb['repeat'], :, ODORS.index(r.odor_b)]
        assert np.isclose(r.distance, 1 - x @ y / np.linalg.norm(x) / np.linalg.norm(y))
    # the solvent closes every block: two odors are 1 to 15 apart within a block and at least 2 across a boundary
    assert t.loc[t.same_block, 'separation'].between(1, 15).all()
    assert t.loc[~t.same_block, 'separation'].min() == 2
    # across a boundary the onsets are 30 volumes further apart than within a block at the same separation
    w = t[t.same_block & (t.separation == 2)].separation_s.iloc[0]
    acr = t[~t.same_block & (t.separation == 2)].separation_s.iloc[0]
    assert np.isclose(w, 140.0) and np.isclose(acr - w, 15.0)
    with pytest.raises(ValueError):
        H.presentation_pairs(X, ODORS, o.assign(repeat=0))      # an (odor, repeat) given twice


def _state_trials(order, m, *, kind, seed, sd_state=0.6, sd_indep=0.4, phi=0.9):
    """Trials (3, units, 16) of one recording whose presentations share a per-unit state: 'time', an AR(1) process
    over the 51 presentations (no block structure); 'block', one state per block (none of time within it)."""
    rng = np.random.default_rng(seed)
    U = m.shape[0]
    o = order.sort_values('presentation')
    u = np.zeros((len(o), U))
    if kind == 'time':
        u[0] = rng.standard_normal(U)
        for p in range(1, len(o)):
            u[p] = phi * u[p - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(U)
    else:
        v = rng.standard_normal((3, U))
        u = v[o['block'].to_numpy()]
    X = np.zeros((3, U, 16))
    for p, (od, rep) in enumerate(zip(o['odor'], o['repeat'])):
        if od in ODORS:
            X[rep, :, ODORS.index(od)] = m[:, ODORS.index(od)] + sd_state * u[p] + sd_indep * rng.standard_normal(U)
    return X


@pytest.mark.parametrize('kind', ['time', 'block'])
def test_gap_decomposition_separation_vs_block(kind):
    """A state that decays with the time between presentations makes a same-repeat gap carried by the separation (no
    block part); a state constant within a block makes one carried by the block (no separation part)."""
    m = _signal(seed=20, n_units=2000)
    tabs = {'g': {}}
    for i in range(6):
        o = _order(30 + i).rename(columns={'odor': 'odor_name'}).rename(columns={'odor_abbrev': 'odor'})
        X = _state_trials(o, m, kind=kind, seed=40 + i)
        t = H.presentation_pairs(X, ODORS, o, volume_rate=3.0)
        t['k_bin'] = E81.k_bin(t['separation'])
        tabs['g'][i] = t
    lv = E81.separation_levels(tabs, 'positions')
    parts = [E81.decomposition_parts(t, 'positions', lv) for t in tabs['g'].values()]
    d = E81.gap_decomposition(parts)
    assert abs(d['check']) < 1e-10
    # the gap equals E81's: different-repeat minus same-repeat single-trial distances, mean over recordings
    gaps = [t[~t.same_block].distance.mean() - t[t.same_block].distance.mean() for t in tabs['g'].values()]
    assert np.isclose(d['gap'], np.mean(gaps))
    assert d['gap'] > 0.02
    if kind == 'time':
        assert d['separation_part'] > 0.85 * d['gap'] and abs(d['block_part']) < 0.15 * d['gap']
    else:
        assert d['block_part'] > 0.85 * d['gap'] and abs(d['separation_part']) < 0.15 * d['gap']
    # the same in seconds (knots every onset interval): a block state stays in the block part
    lvs = E81.separation_levels(tabs, 'seconds')
    ds = E81.gap_decomposition([E81.decomposition_parts(t, 'seconds', lvs) for t in tabs['g'].values()])
    assert np.isclose(ds['gap'], d['gap']) and abs(ds['check']) < 1e-10
    if kind == 'block':
        assert ds['block_part'] > 0.85 * ds['gap']
    # with a level for each presentation: the parts still sum to the gap, and each state stays where it was
    dp = E81.gap_decomposition([E81.decomposition_parts(t, 'positions', lv, presentations=True) for t in tabs['g'].values()])
    assert abs(dp['check']) < 1e-10 and np.isclose(dp['gap'], d['gap'])
    big = 'block_part' if kind == 'block' else 'separation_part'
    assert dp[big] > 0.85 * dp['gap']


def test_gap_decomposition_noisy_block_edges_make_no_block_part():
    """Noisier trials at the edges of the blocks (the first two and the last two odors of each block; no state shared
    between presentations) make neither a gap nor a block part at equal separation, with or without a level for
    each presentation."""
    m = _signal(seed=21, n_units=2000)
    tabs = {'g': {}}
    for i in range(6):
        o = _order(50 + i).rename(columns={'odor': 'odor_name'}).rename(columns={'odor_abbrev': 'odor'})
        rng = np.random.default_rng(60 + i)
        X = np.zeros((3, m.shape[0], 16))
        for _, r in o[o.odor.isin(ODORS)].iterrows():
            sd = 1.2 if r.position in (0, 1, 14, 15) else 0.3
            X[r['repeat'], :, ODORS.index(r.odor)] = m[:, ODORS.index(r.odor)] + sd * rng.standard_normal(m.shape[0])
        t = H.presentation_pairs(X, ODORS, o)
        t['k_bin'] = E81.k_bin(t['separation'])
        tabs['g'][i] = t
    lv = E81.separation_levels(tabs, 'positions')
    d = E81.gap_decomposition([E81.decomposition_parts(t, 'positions', lv) for t in tabs['g'].values()])
    dp = E81.gap_decomposition([E81.decomposition_parts(t, 'positions', lv, presentations=True) for t in tabs['g'].values()])
    assert abs(d['gap']) < 0.005
    assert abs(d['block_part']) < 0.005 and abs(dp['block_part']) < 0.005
    assert abs(dp['check']) < 1e-10


def test_pair_separation_stats_and_null_orders():
    P = np.arange(16)[None, None, :].repeat(2, axis=1)           # 1 recording, 2 blocks, odors in ODORS order
    P[0, 1] = P[0, 1][::-1]
    st = E81.pair_separation_stats(P)
    pairs = [tuple(p) for p in H.PAIRS16]
    i01, i0f = pairs.index((ODORS[0], ODORS[1])), pairs.index((ODORS[0], ODORS[15]))
    assert st['sep'][0, i01] == 1 and st['adj'][0, i01] == 2 and st['within3'][0, i01] == 2
    assert st['sep'][0, i0f] == 15 and st['adj'][0, i0f] == 0
    # the null keeps the design's reuse of block orders, and each block a permutation of its positions
    rng = np.random.default_rng(0)
    P2 = np.stack([rng.permutation(16) for _ in range(4)]).reshape(2, 2, 16)
    P2[1, 1] = P2[0, 0]
    nP = E81.null_positions(P2, np.random.default_rng(1), 200)
    assert nP.shape == (200, 2, 2, 16)
    assert (nP[:, 1, 1] == nP[:, 0, 0]).all()
    assert (np.sort(nP, axis=-1) == np.arange(16)).all()
    assert (nP[:, 0, 1] != nP[:, 0, 0]).any(axis=-1).mean() > 0.99
    # the null's mean adjacency count per pair and block is 2/16
    assert np.isclose(E81.pair_separation_stats(nP)['adj'].mean() / 2, 2 / 16, atol=0.01)
    # the rank correlation helper is Spearman's
    from scipy.stats import spearmanr
    x, y = rng.standard_normal((5, 30)), rng.standard_normal(30)
    assert np.allclose(E81._rank_corr(x, y), [spearmanr(a, y)[0] for a in x])
