#!/usr/bin/env python3
"""Tests for E10_joint_fit.py's options for the flies' recording protocol (E82; scripts/claude_pair_remapping_experiments):
--stim-end / --sim-end / --apl-taum (`protocol_kws`, the tag suffix, `fit_one`'s protocol, `bouton_base`'s timing for the
breadth), --count-kernel (`make_loss`' count_kernel, `apl_effect_terms`' responders_on, the spike times through
`fit_one`) and the options' checks in `main`; all off leave every kwarg, term and statistic as before. No simulations,
except the exact starts (E82_E10_EXACT_START=1, about 3 min per fit: E10 from the optima of C2, E72's P1 fit and E76's
fit 0, with the options off and with every option at its identity setting, each first evaluation against the fit's
saved history). Needs the shared directory (E10 reads the data-target registry at import); skipped without it."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

EXP_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'claude_pair_remapping_experiments'
sys.path.insert(0, str(EXP_DIR))
H = pytest.importorskip('kc_remapping_harness')
try:
    H.data_targets_path().read_bytes()
except (FileNotFoundError, OSError):
    pytest.skip('the shared directory (data-target registry) is not available', allow_module_level=True)
E10 = pytest.importorskip('E10_joint_fit')
if not hasattr(H, 'kernel_weighted_counts'):
    pytest.skip('kc_remapping_harness has no kernel_weighted_counts', allow_module_level=True)

ODORS = list(H.APL16_ODORS)
LABELS = [f'{o} @ -3' for o in ODORS]
# the APL-effect statistics (`apl_effect_terms`) and the terms and loss that use them
APL_KEYS = ('apl_raw_', 'apl_cal_', 'block_cal', 'mean_dist_cal_', 'term_apl', 'loss')


def _outputs(seed=0, n_units=900, p_on=0.07, p_off=0.15):
    """Synthetic model outputs, as test_e10_band_units makes them."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(0.6, 1.0, (n_units, len(ODORS))) * rng.gamma(2.0, 0.5, (n_units, 1))
    out = {}
    for c, p in (('on', p_on), ('off', p_off)):
        thr = np.quantile(base, 1 - p)
        n = np.where(base > thr, np.rint(2 + 6 * (base - thr) / base.std()), (base > 0.8 * thr).astype(float))
        s = pd.DataFrame(n, columns=LABELS)
        out[c] = ((s >= 2).astype(float), s)
    return out


def _spike_times(s: pd.DataFrame, seed: int = 0, t_max: float = 0.75) -> pd.DataFrame:
    """Spike times adding up to the counts `s` (each KC x odor's spikes at random times in (0, t_max), sorted as the
    simulator sorts them)."""
    rng = np.random.default_rng(seed)
    a = s.to_numpy(dtype=int)
    kc, od = np.nonzero(a)
    n = a[kc, od]
    kc, od = np.repeat(kc, n), np.repeat(od, n)
    t = rng.uniform(0.0, t_max, len(kc))
    df = pd.DataFrame(dict(kc=kc.astype(np.int32), odor=od.astype(np.int32), time_s=t))
    return df.sort_values(['odor', 'kc', 'time_s'], kind='stable', ignore_index=True)


def _with_spike_times(o: dict, t_max: float = 0.75) -> dict:
    return {c: (r, s, _spike_times(s, i, t_max)) for i, (c, (r, s)) in enumerate(o.items())}


@pytest.fixture(scope='module')
def ctx():
    data = {'raw': {g: H.yang_mean_rdm(g, 'cosine') for g in ('TNT3c', 'TNT')}}
    data_rates = {g: H.yang_response_fractions(g).mean() for g in ('TNT3c', 'TNT')}
    return dict(data=data, data_rates=data_rates, orn_pairs=H.orn16_consensus_pairs(), drive=H.consensus_orn_drive(),
        ref=H.reference_pairs(), rate=H.data_target('ctrl_frac'))


def _loss(ctx, **kw):
    kw.setdefault('variant', 'apl')
    if kw['variant'] == 'apl':
        kw.setdefault('apl_readout', 'both')
    return E10.make_loss(ctx['data'], ctx['data_rates'], ctx['orn_pairs'], ctx['drive'], ref=ctx['ref'], structure=True,
        rate_target=ctx['rate'], **kw)


def _same(a: dict, b: dict, keys=None) -> list:
    keys = sorted(set(a) | set(b)) if keys is None else keys
    return [k for k in keys if not (k in a and k in b and (a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k]))))]


def test_protocol_kws_and_tag():
    assert E10.protocol_kws() == {} and E10.protocol_tag({}) == '' and E10.protocol_tag(None) == ''
    p = E10.protocol_kws(2, 2.25)
    assert p == dict(time_stim_end=2.0, time_end=2.25) and E10.protocol_tag(p) == '_pulse2'
    assert E10.protocol_tag(E10.protocol_kws(2, 2.5)) == '_pulse2end2.5'
    assert E10.protocol_tag(E10.protocol_kws(0.6)) == '_pulse0.6end0.75'
    assert E10.protocol_tag(E10.protocol_kws(0.5, 0.75)) == '_pulse0.5'
    assert E10.protocol_tag(E10.protocol_kws(None, 1.0)) == '_end1'
    assert E10.protocol_kws(apl_taum=0.25) == dict(apl_taum=0.25)
    assert E10.protocol_tag(E10.protocol_kws(apl_taum=0.25)) == '_apltau250'
    assert E10.protocol_tag(E10.protocol_kws(2, 2.25, 0.01), (0.5, 3.0)) == '_pulse2_apltau10_kr0.5d3'
    assert E10.protocol_tag({}, (0.0, np.inf)) == '_kr0dinf'
    assert E10.timing_kws(E10.protocol_kws(2, 2.25, 0.01)) == dict(time_stim_end=2.0, time_end=2.25)
    assert E10.timing_kws(E10.protocol_kws(apl_taum=0.01)) == {} and E10.timing_kws(None) == {}
    assert E10.timing_label({}) == '' and E10.timing_label(dict(time_stim_end=2.0, time_end=2.25)) == 'pulse2end2.25'
    assert E10.timing_label(dict(time_end=1.0)) == 'pulse0.5end1'
    for bad in (dict(stim_end=0.75), dict(stim_end=2.0), dict(sim_end=0.4), dict(stim_end=2.0003, sim_end=2.25),
            dict(apl_taum=0.0), dict(apl_taum=1e-4), dict(apl_taum=np.nan)):
        with pytest.raises(ValueError):
            E10.protocol_kws(**bad)


def test_kernel_counts():
    like = pd.DataFrame(np.zeros((2, 2)), index=['a', 'b'], columns=['x', 'y'])
    st = pd.DataFrame(dict(kc=[0, 0, 1, 1], odor=[0, 1, 1, 1], time_s=[0.05, 2.0, 1.0, 0.5]))
    w = E10.kernel_counts(st, like, (0.0, np.inf))
    assert w.index.equals(like.index) and w.columns.equals(like.columns)
    assert w.to_numpy().tolist() == [[1.0, 1.0], [0.0, 2.0]]
    w = E10.kernel_counts(st, like, (0.5, 3.0))
    # (a spike at the window's start weighs 1; earlier ones less, by the entry's 0.62 at 50 ms for rise 0.5, decay 3)
    assert w.loc['a', 'y'] == 1.0 and abs(w.loc['a', 'x'] - 0.62) < 0.01
    assert np.isclose(w.loc['b', 'y'], H.kernel_window_weights([1.0, 0.5], 0.5, 3.0).sum())


def test_apl_effect_terms_responders_on_raw_counts(ctx):
    """The responders given as the counts' own two-spike entries: every statistic as without (to the last digit)."""
    o = _outputs(1)
    s_on, s_off = o['on'][1], o['off'][1]
    on, off = H.pair_distances(s_on, 'cosine'), H.pair_distances(s_off, 'cosine')
    data, target = H.yang_apl_effect(), H.data_target('ctrl_participation_ratio_resp')
    a = E10.apl_effect_terms(on, off, s_on, s_off, data, target)
    b = E10.apl_effect_terms(on, off, s_on, s_off, data, target, responders_on=s_on >= 2)
    assert set(a) == set(b) and not _same(a, b)


@pytest.mark.parametrize('readout', ['both', 'raw', 'cal'])
def test_count_kernel_identity_unchanged(ctx, readout):
    """A kernel weighing every spike 1 (rise 0, decay inf, every spike before the window): every term and statistic of
    the loss is the one without the kernel, to the last digit."""
    o = _outputs(2)
    l0 = _loss(ctx, apl_readout=readout)(o)
    lk_fn = _loss(ctx, apl_readout=readout, count_kernel=(0.0, np.inf))
    assert lk_fn.count_kernel == (0.0, np.inf) and lk_fn.apl['count_kernel']['rise'] == 0.0
    lk = lk_fn(_with_spike_times(o))
    assert set(lk) == set(l0) and not _same(lk, l0)
    assert _loss(ctx, apl_readout=readout).count_kernel is None
    assert 'count_kernel' not in _loss(ctx, apl_readout=readout).apl


@pytest.mark.parametrize('variant', ['structure', 'apl'])
def test_spike_times_without_kernel_unchanged(ctx, variant):
    """Outputs carrying their spike times (--save-spike-times without --count-kernel): the loss is the one without."""
    o = _outputs(4)
    fn = _loss(ctx, variant=variant)
    l0 = fn(o)
    l1 = fn(_with_spike_times(o))
    assert set(l1) == set(l0) and not _same(l1, l0)


def test_count_kernel_changes_the_apl_terms_only(ctx):
    """A kernel of the flies' protocol (rise 0.5, decay 3): the APL-effect statistics are those of the kernel-weighted
    counts (the calibration's responders the two-spike entries), every other statistic as without it."""
    o = _outputs(3)
    ot = _with_spike_times(o)
    l0 = _loss(ctx)(o)
    lk = _loss(ctx, count_kernel=(0.5, 3.0))(ot)
    assert set(lk) == set(l0)
    differ = _same(lk, l0)
    assert differ and all(k.startswith(APL_KEYS) for k in differ)
    assert 'apl_raw_rmse' in differ and 'apl_cal_K' in differ
    w = {c: E10.kernel_counts(ot[c][2], ot[c][1], (0.5, 3.0)) for c in ('on', 'off')}
    ref = E10.apl_effect_terms(H.pair_distances(w['on'], 'cosine'), H.pair_distances(w['off'], 'cosine'), w['on'], w['off'],
        H.yang_apl_effect(), H.data_target('ctrl_participation_ratio_resp'), responders_on=o['on'][1] >= 2)
    assert not _same(lk, ref, list(ref))


def test_count_kernel_checks(ctx):
    with pytest.raises(ValueError, match='count_kernel'):
        _loss(ctx, variant='structure', count_kernel=(0.5, 3.0))
    with pytest.raises(ValueError, match='count_kernel'):
        _loss(ctx, apl_readout='zcell', count_kernel=(0.5, 3.0))
    with pytest.raises(ValueError):
        _loss(ctx, count_kernel=(-1.0, 3.0))


def _fake_base(n=12):
    idx = pd.Index([f'kc{i}' for i in range(n)], name=H.KC_ID_LEVEL)
    wnorm = {k: pd.Series(np.linspace(1, 2, n), index=idx) for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
    breadth = pd.Series(np.arange(n, dtype=float), index=idx)
    initial = dict(fixed_thr=100.0, wAPLKC_scale=0.1, wKCAPL_scale=1e-4, wAPLPN_scale=0.1, wPNAPL_scale=1e-3)
    return initial, wnorm, breadth


@pytest.mark.parametrize('protocol, spike_times', [(None, False), ({}, False), (dict(time_stim_end=2.0, time_end=2.25), False),
    (dict(apl_taum=0.25), False), (None, True), (dict(time_stim_end=2.0, time_end=2.25, apl_taum=0.01), True)])
def test_fit_one_passes_the_protocol(monkeypatch, protocol, spike_times):
    import al_analysis.mb_model as mb
    seen = {}

    def fake(orn, **kw):
        seen.update(kw)
        return dict(history=pd.DataFrame(), params={}, loss=0.0)

    monkeypatch.setattr(mb, 'optimize_mb_model_params', fake)
    initial, wnorm, breadth = _fake_base()
    kw = dict(kckc0=60.0, maxfev=1, method='Nelder-Mead', thr0_factor=0.7)

    def loss_fn(o):
        return {}

    E10.fit_one(2.0, None, initial, wnorm, breadth, loss_fn, **kw)
    plain = dict(seen)
    seen.clear()
    E10.fit_one(2.0, None, initial, wnorm, breadth, loss_fn, **kw,
        **({} if protocol is None else dict(protocol=protocol)), **({} if not spike_times else dict(spike_times=True)))
    for k in ('time_stim_end', 'time_end', 'apl_taum'):
        assert (seen.get(k) == (protocol or {})[k]) if k in (protocol or {}) else k not in seen
    assert (seen.get('condition_spike_times') == ('on', 'off')) if spike_times else 'condition_spike_times' not in seen
    # everything else as without them
    new = set(protocol or {}) | ({'condition_spike_times'} if spike_times else set())
    rest = {k: v for k, v in seen.items() if k not in new}
    assert set(rest) == set(plain)
    for k, v in plain.items():
        assert rest[k] is v or (isinstance(v, (pd.Series, pd.DataFrame)) and rest[k].equals(v)) or rest[k] == v, k


def test_fit_one_protocol_checks(monkeypatch):
    import al_analysis.mb_model as mb
    monkeypatch.setattr(mb, 'optimize_mb_model_params', lambda orn, **kw: dict(history=pd.DataFrame(), params={}, loss=0.0))
    initial, wnorm, breadth = _fake_base()
    kw = dict(kckc0=60.0, maxfev=1, method='Nelder-Mead', thr0_factor=0.7)
    with pytest.raises(ValueError, match='protocol'):
        E10.fit_one(2.0, None, initial, wnorm, breadth, lambda o: {}, **kw, protocol=dict(claw_thr=40.0))
    proxy = dict(sigma=0.1, thr_ratio=1.0, n_recordings=1, n_trials=1, seed=0, passes=1, relax=False)
    for extra in (dict(protocol=dict(apl_taum=0.25)), dict(spike_times=True)):
        with pytest.raises(NotImplementedError):
            E10.fit_one(2.0, None, initial, wnorm, breadth, lambda o: {}, **kw, block_proxy=proxy, **extra)


@pytest.mark.parametrize('timing', [None, dict(time_stim_end=2.0, time_end=2.25)])
def test_bouton_base_timing(monkeypatch, tmp_path, timing):
    """The breadth run (APL off) gets the timing and a cache name with its label; the base's own call does not; without a
    timing, the calls are as before (hal's saved breadth, no breadth run)."""
    import al_analysis.mb_model as mb
    calls = []
    n = 6
    idx = pd.MultiIndex.from_arrays([[f'kc{i}' for i in range(n)]], names=[H.KC_ID_LEVEL])

    def fake(orn, **kw):
        calls.append(kw)
        s = pd.DataFrame(np.arange(n * 2, dtype=float).reshape(n, 2), index=idx, columns=['a', 'b'])
        pd0 = {}
        for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL'):
            pd0[k] = pd.Series(np.ones(n), index=idx)
            pd0[f'{k}_scale'] = 1.0
        return (s >= 2).astype(float), s, pd.DataFrame(np.ones((n, 1)), index=idx, columns=['g']), pd0

    monkeypatch.setattr(mb, 'fit_mb_model', fake)
    orn = pd.DataFrame(np.ones((1, 2)), index=['g'], columns=['a', 'b'])
    out = E10.bouton_base(orn, tmp_path, None, **({} if timing is None else dict(timing=timing)))
    assert len(calls) == (1 if timing is None else 2)
    assert not any(k in calls[0] for k in E10.TIMING_KEYS)
    if timing is not None:
        assert calls[1]['wAPLKC'] == 0.0 and calls[1]['wAPLPN'] == 0.0
        assert all(calls[1][k] == v for k, v in timing.items())
        assert (tmp_path / 'pulse2end2.25_base_acute_off_spike_counts.parquet').exists()
        assert out[3].to_numpy().tolist() == [1.0, 5.0, 9.0, 13.0, 17.0, 21.0]
    with pytest.raises(ValueError):
        E10.bouton_base(orn, tmp_path, None, timing=dict(apl_taum=0.25))


def _main(monkeypatch, tmp_path, argv):
    orig_r, orig_o = H.results_dir, H.outputs_dir
    monkeypatch.setattr(H, 'results_dir', lambda e: (tmp_path / 'results') if e == 'E82_check' else orig_r(e))
    monkeypatch.setattr(H, 'outputs_dir', lambda e: (tmp_path / 'outputs') if e == 'E82_check' else orig_o(e))
    for d in ('results', 'outputs'):
        (tmp_path / d).mkdir(exist_ok=True)
    monkeypatch.setattr(sys, 'argv', ['E10_joint_fit.py', '--experiment', 'E82_check', *argv])
    E10.main()


@pytest.mark.parametrize('argv, message', [
    (['--count-kernel', '0.5', '3'], '--count-kernel needs --loss apl'),
    (['--loss', 'apl', '--apl-readout', 'zcell', '--rate-target', 'ctrl_frac', '--count-kernel', '0.5', '3'],
        '--count-kernel is not defined for --apl-readout zcell'),
    (['--loss', 'apl', '--count-kernel', '-1', '3'], 'RISE must be'),
    (['--loss', 'apl', '--count-kernel', '0.5', '0'], 'RISE must be'),
    (['--apl-taum', '0'], 'apl_taum must be'),
    (['--stim-end', '0.8'], 'time_stim_end'),
    (['--sim-end', '0.4'], 'time_end'),
    (['--stim-end', '2', '--sim-end', '2.25', '--apl-drive', 'spikes_only'], 'spikes_only'),
    (['--save-spike-times', '--block-proxy', '1', '--jitter-sigma', '0.1'], 'with --block-proxy is not supported'),
], ids=['kernel_without_apl', 'kernel_zcell', 'kernel_rise', 'kernel_decay', 'apl_taum', 'stim_end', 'sim_end',
    'spikes_only', 'block_proxy'])
def test_main_refuses(monkeypatch, tmp_path, capsys, argv, message):
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, tmp_path, argv)
    assert e.value.code == 2
    assert message in capsys.readouterr().err


# the fits whose best evaluation an exact start with the options off must reproduce (E82's last-digit checks): C2 (the
# breadth-2 APL-effect fit, E67), E72's part-B fit on P1 and E76's fit 0 (the firing-ceiling fit), with their E10 options
COMMON = ['-k', '2', '--loss', 'apl', '--rate-target', 'ctrl_frac', '--aplpn-max', '3', '--maxfev', '1', '--init-from-exact',
    '--apl-readout', 'both']
EXACT_FITS = {
    'C2': ('E67_apl_effect_fits', 'rate0.068_aplboth_k2_cap3', ['--tag', '_cap3']),
    'P1': ('E72_wiring_population', 'wiring-claw_permutation_within_type-seed1_rate0.068_aplboth_k2_cap3',
        ['--tag', '_cap3', '--wiring', 'claw_permutation_within_type', '--wiring-seed', '1']),
    'fit0': ('E76_data_side_target', 'nu_rate0.068_aplboth_refr3_adapt0.1114t1000_k2',
        ['--band-units', 'noise', '--kc-refractory-period', '0.003', '--kc-adapt-inc', '0.1113973859994802',
        '--kc-adapt-tau', '1.0']),
}
# every option at its identity setting: the pulse's and the simulation's ends and the APL's time constant at olfsysm's
# defaults, a kernel weighing every spike (all before the flies' window) exactly 1
IDENTITY = ['--stim-end', '0.5', '--sim-end', '0.75', '--apl-taum', '0.05', '--count-kernel', '0', 'inf']


def _e10_exact(root: Path, fit: str, extra) -> Tuple[pd.Series, Dict[str, pd.DataFrame], Path]:
    """One exact-start evaluation of `fit` (E78_claw_knee's runner: results and outputs under `root`); (its first history
    row, its saved counts and spike times by file stem, its history path)."""
    import E78_claw_knee as E78
    exp, tag, opts = EXACT_FITS[fit]
    args = COMMON + ['--experiment', E78.CHECK_SCRATCH, '--init-from', tag, '--init-from-experiment', exp] + opts + list(extra)
    p = subprocess.run([sys.executable, '-u', str(EXP_DIR / 'E78_claw_knee.py'), '--run-e10', str(root), '--', *args],
        cwd=EXP_DIR, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    hist = sorted((root / 'results').glob('history_*.csv'))
    assert len(hist) == 1
    out = {f.stem: pd.read_parquet(f) for f in (root / 'outputs').glob('*_spike_*.parquet')}
    return pd.read_csv(hist[0], float_precision='round_trip').iloc[0], out, hist[0]


@pytest.mark.skipif(os.environ.get('E82_E10_EXACT_START') != '1', reason='model fits (minutes): set E82_E10_EXACT_START=1')
@pytest.mark.parametrize('fit', list(EXACT_FITS))
def test_exact_start(tmp_path, fit):
    """E10 from a fit's optimum (exact start, one evaluation), with the options off and with every option at its identity
    setting: the saved counts are the fit's own, the first evaluation's statistics are its best evaluation's in its saved
    history (exactly when the history was saved on this host; within 1e-12 relative otherwise: fit 0's was saved on
    atlas, and E10 at the commit before these options gives the same last-digit differences on viki), and the two runs'
    first evaluations are identical to the last digit."""
    E78 = pytest.importorskip('E78_claw_knee')
    import E72_wiring_population as E72
    exp, tag, _ = EXACT_FITS[fit]
    hist0 = H.results_dir(exp) / f'history_{tag}.csv'
    saved = E72.load_fit(exp, tag)
    if not hist0.exists() or saved is None:
        pytest.skip(f"{fit}'s saved history or outputs are not available")
    rows, outs = {}, {}
    for name, extra in (('off', []), ('identity', IDENTITY)):
        rows[name], outs[name], path = _e10_exact(tmp_path / name, fit, extra)
        r = E78.compare_history(path, exp, tag)
        assert not r['missing'] and not r['extra'], r
        for c, d in r['differing_statistics'].items():
            assert d['abs'] <= 1e-12 * max(abs(d['saved']), 1.0), (c, d)
        for c, f in (('on', 'on'), ('off', 'acute_off')):
            s = outs[name][f'{path.stem[len("history_"):]}_{f}_spike_counts']
            assert s.index.equals(saved[c][1].index) and (s.to_numpy(dtype=float) == saved[c][1].to_numpy(dtype=float)).all()
    cols = [c for c in rows['off'].index if c not in ('eval', 'time_s')]
    assert list(rows['identity'].index) == list(rows['off'].index)
    assert all(rows['identity'][c] == rows['off'][c] or (pd.isna(rows['identity'][c]) and pd.isna(rows['off'][c]))
        for c in cols)
    # (the identity run saves its spike times, which add up to its counts)
    tag_id = [k for k in outs['identity'] if k.endswith('_on_spike_times')][0][:-len('_on_spike_times')]
    for f in ('on', 'acute_off'):
        assert len(outs['identity'][f'{tag_id}_{f}_spike_times']) == int(outs['identity'][f'{tag_id}_{f}_spike_counts'].to_numpy().sum())
    assert not any(k.endswith('_spike_times') for k in outs['off'])
