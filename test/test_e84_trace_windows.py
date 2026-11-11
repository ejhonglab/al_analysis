#!/usr/bin/env python3
"""Tests for the trace windows and the indicator kernel of the KC-remapping harness (the time course of the APL
effect in the flies' traces, E84: scripts/claude_pair_remapping_experiments/kc_remapping_harness.py), on synthetic
traces only: the target's window reproduces Yang's per-trial responses, the noise-free distances of windowed trials
are those of the per-trial values, and a known kernel and drive are recovered from a noisy mean trace."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
H = pytest.importorskip('kc_remapping_harness')

# the APL16 recordings' volume rate (59.273 frames / s, 20 frames per volume) and Yang's volume counts at it
RATE = 59.273 / 20
B, S, P = round(5 * RATE), round(2 * RATE), round(8 * RATE)
ODORS = list(H.APL16_ODORS)
# numpy >= 2 renamed trapz (numpy 1.x, for the old Python 3.8 environment, has only trapz)
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def _session(seed=0, n_units=40, n_trials=12, gap=139, n_before=15, n_after=119):
    """Synthetic traces (units, volumes) with a slow drift, noise and responses after each onset, and the onsets."""
    rng = np.random.default_rng(seed)
    T = n_before + n_trials * gap + n_after
    onsets = n_before + 5 + gap * np.arange(n_trials)
    t = np.arange(T)
    x = 100 + 20 * rng.random((n_units, 1)) + 0.002 * t[None] * rng.standard_normal((n_units, 1))
    x = x + rng.standard_normal((n_units, T))
    for n in onsets:
        amp = rng.gamma(1.0, 2.0, (n_units, 1)) * (rng.random((n_units, 1)) < 0.3)
        u = (t - n) / RATE
        x = x + amp * H.indicator_kernel(u, 1.0, 10.0)[None]
    return x, onsets


def test_window_volumes():
    assert H.window_volumes((2, 8), RATE) == (S, P) == (6, 24)
    assert H.window_volumes((2.0, 8.1), RATE) == (6, 24)
    assert H.window_volumes((0, 2), RATE) == (0, 6)
    assert H.window_volumes((-2, -1), RATE) == (-6, -3)
    with pytest.raises(ValueError):
        H.window_volumes((1.0, 1.1), RATE)


def test_target_window_reproduces_yang_responses():
    from scipy.stats import zscore
    x, onsets = _session()
    z = zscore(x, axis=1)
    yang = np.stack([z[:, n + S:n + P].mean(axis=1) - z[:, n - B:n].mean(axis=1) for n in onsets], axis=1)
    ex = H.trace_excerpts(z, onsets, B, 119)
    assert ex.shape == (len(onsets), x.shape[0], B + 119)
    assert np.allclose(ex[:, :, :B].mean(axis=2), 0, atol=1e-12)
    ours = H.windowed_responses(ex, [(S, P)], first_volume=-B)[..., 0].T
    assert np.max(np.abs(ours - yang)) < 1e-12
    # the same window given in seconds, and the excerpts of the unscaled trace divided by the whole-trace SD
    sec = H.windowed_responses(ex, [(2.0, 8.1)], first_volume=-B, volume_rate=RATE)[..., 0].T
    assert np.array_equal(sec, ours)
    exr = H.trace_excerpts(x, onsets, B, 119) / x.std(axis=1)[None, :, None]
    assert np.max(np.abs(H.windowed_responses(exr, [(S, P)], first_volume=-B)[..., 0].T - yang)) < 1e-10
    # several windows at once, and windows outside the excerpt refused
    two = H.windowed_responses(ex, [(0, 6), (S, P)], first_volume=-B)
    assert two.shape == (len(onsets), x.shape[0], 2) and np.array_equal(two[..., 1].T, ours)
    with pytest.raises(ValueError):
        H.windowed_responses(ex, [(-20, 0)], first_volume=-B)
    with pytest.raises(ValueError):
        H.trace_excerpts(x, [3], B, 119)


def test_noise_free_of_windowed_trials():
    """The noise-free distances of windowed excerpts are those of the per-trial values (E81's function)."""
    rng = np.random.default_rng(3)
    n_units, n_rep = 300, 3
    signal = np.where(rng.random((n_units, len(ODORS))) < 0.15, rng.gamma(2.0, 1.5, (n_units, len(ODORS))), 0.0)
    # excerpts: (repeats * odors) trials whose window means are signal + noise
    ex = rng.standard_normal((n_rep, len(ODORS), n_units, B + 30)) * 0.5
    ex[..., B + S:B + 24] += signal.T[None, :, :, None]
    ex = ex - ex[..., :B].mean(axis=-1, keepdims=True)
    w = H.windowed_responses(ex, [(S, 24)], first_volume=-B)[..., 0]          # (repeats, odors, units)
    trials = w.transpose(0, 2, 1)
    direct = np.stack([[ex[r, o, :, B + S:B + 24].mean(axis=-1) for o in range(len(ODORS))]
                       for r in range(n_rep)]).transpose(0, 2, 1)
    assert np.allclose(trials, direct, atol=1e-12)
    a = H.noise_free_pair_distances(trials, ODORS)
    b = H.noise_free_pair_distances(direct, ODORS)
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)
    assert list(a.index) == [tuple(p) for p in H.PAIRS16]


def test_kernel_functions():
    u = np.linspace(-1, 30, 6201)
    k = H.indicator_kernel(u, 1.3, 20.0)
    assert np.all(k[u <= 0] == 0) and k.max() < 1
    # the integral against a numerical one, and the limits
    K = H.indicator_kernel_integral(u, 1.3, 20.0)
    num = np.concatenate([[0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(u))])
    assert np.max(np.abs(K - num)) < 1e-5
    assert np.allclose(H.indicator_kernel(u, 0.0, np.inf), (u > 0).astype(float))
    assert np.allclose(H.indicator_kernel_integral(u, 0.0, np.inf), np.maximum(u, 0))
    # the onset drive is the kernel; the steady drive a box convolution; a one-bin histogram the steady drive
    t = np.arange(-15, 119) / RATE
    assert np.allclose(H.kernel_signal(t, 1.3, 20.0, 'onset'), H.indicator_kernel(t, 1.3, 20.0))
    s = np.linspace(0, 2, 4001)
    box = np.array([_trapz(H.indicator_kernel(tt - s, 0.8, 10.0), s) / 2.0 for tt in t])
    st = H.kernel_signal(t, 0.8, 10.0, 'steady', pulse=2.0)
    assert np.max(np.abs(st - box)) < 1e-6
    hist = H.kernel_signal(t, 0.8, 10.0, 'histogram', edges=[0.0, 2.0], weights=[5.0])
    assert np.allclose(hist, st)
    two = H.kernel_signal(t, 0.8, 10.0, 'histogram', edges=[0.0, 1.0, 2.0], weights=[1.0, 1.0])
    assert np.allclose(two, st)
    # the area inside the flies' window of a spike's kernel is what kernel_window_weights weighs it by
    dense = np.linspace(2.0, 8.1, 20001)
    for spike in (0.05, 1.0, 3.0):
        area = _trapz(H.indicator_kernel(dense - spike, 0.5, 3.0), dense)
        ref = _trapz(H.indicator_kernel(dense - 2.0, 0.5, 3.0), dense)
        assert abs(area / ref - H.kernel_window_weights([spike], 0.5, 3.0)[0]) < 1e-6
    with pytest.raises(ValueError):
        H.kernel_signal(t, 1.0, 5.0, 'ramp')


@pytest.mark.parametrize('drive,rise,decay', [('onset', 1.3, 20.0), ('steady', 0.8, 10.0), ('onset', 0.5, 6.0)])
def test_fit_recovers_known_kernel(drive, rise, decay):
    rng = np.random.default_rng(7)
    t = np.arange(-15, 119) / RATE
    amp = 4.0
    clean = amp * H.kernel_signal(t, rise, decay, drive, pulse=2.0)
    y = clean + 0.01 * amp * rng.standard_normal(len(t))
    f = H.fit_indicator_kernel(t, y, drive, pulse=2.0)
    assert f['drive'] == drive and f['n'] == len(t)
    assert abs(f['rise'] / rise - 1) < 0.1
    assert abs(f['decay'] / decay - 1) < 0.1
    assert abs(f['amplitude'] / amp - 1) < 0.05
    assert f['rms'] < 0.02 * amp
    assert not f['rise_at_bound'] and not f['decay_at_bound']
    # the amplitude alone with the kernel fixed
    g = H.fit_indicator_kernel(t, y, drive, pulse=2.0, fixed=(rise, decay))
    assert abs(g['amplitude'] / amp - 1) < 0.02 and g['rise'] == rise and g['decay'] == decay


def test_fit_tells_the_drives_apart_without_noise():
    """A steady drive's trace is fitted worse by an onset burst than by the steady drive, and the other way round."""
    t = np.arange(-15, 119) / RATE
    y_steady = 3.0 * H.kernel_signal(t, 0.8, 10.0, 'steady', pulse=2.0)
    y_onset = 3.0 * H.kernel_signal(t, 1.3, 20.0, 'onset')
    for y, right, wrong in ((y_steady, 'steady', 'onset'), (y_onset, 'onset', 'steady')):
        good = H.fit_indicator_kernel(t, y, right, pulse=2.0)
        bad = H.fit_indicator_kernel(t, y, wrong, pulse=2.0)
        assert good['rms'] < 1e-4 and bad['rms'] > 10 * good['rms']
    # a histogram drive (firing in two bins) is recovered as such
    y_h = 2.0 * H.kernel_signal(t, 1.0, 12.0, 'histogram', edges=[0.0, 0.5, 2.0], weights=[3.0, 1.0])
    f = H.fit_indicator_kernel(t, y_h, 'histogram', edges=[0.0, 0.5, 2.0], weights=[3.0, 1.0])
    assert abs(f['rise'] - 1.0) < 1e-3 and abs(f['decay'] - 12.0) < 1e-2


def test_e84_label_parsing():
    """E84's records: the label of a recording's hemisphere and the construct, as written in the YAML files."""
    E84 = pytest.importorskip('E84_trace_time_course')
    cases = [('APL labeled, immediate response.', 'L', 'label'), ('APL not labeled, immediate response.', 'L', 'nolabel'),
             ('both labeled.', 'R', 'label'), ('Both labeled.', 'L', 'label'), ('R labeled.', 'R', 'label'),
             ('R labeled.', 'L', 'nolabel'), ('L labeled. Manual dissection.', 'L', 'label'),
             ('R weakly labeled.', 'R', 'label'), ('Not labeled.', 'L', 'nolabel'), ('', 'L', 'unknown')]
    for note, hemi, want in cases:
        assert E84.label_as_recorded(note, hemi) == want, (note, hemi)
    assert E84.construct_as_recorded('NPGH/TubPGal80ts,UAS-TNT(84);MBnls6s') == 'TNT'
    assert E84.construct_as_recorded('NPGH/TubPGal80ts,UAS-TNT-in(36);MBnls6s') == 'TNTin'
    assert E84.construct_as_recorded('NPGH/TubPGal80ts,UAS-TNTin(36);MBnls6s') == 'TNTin'
    # the queue's notes' groups: 2024-07-10 fly 2's side decides the third TNTin-nolabel recording
    assert E84.notes_group('TNT3c', ('2024-07-10', 2, 'L'), 'L') == 'TNTin-nolabel'
    assert E84.notes_group('TNT3c', ('2024-07-10', 2, 'L'), 'R') == 'TNTin-label'
    assert E84.notes_group('TNT3c', ('2024-08-30', 3, 'R'), 'L') == 'TNT-nolabel'
    assert E84.notes_group('TNT', ('2024-07-17', 3, 'L'), 'L') == 'TNT-label'
    # the reader refuses any recording outside the 19 (without touching a drive)
    YRN = pytest.importorskip('yang_response_normalization')
    with pytest.raises(ValueError):
        YRN.read_suite2p('TNT', '20240101_1FLfun1', Path('/nonexistent'))
    with pytest.raises(ValueError):
        E84.load_cache('TNT3c', '20240717_2FLfun1')
