#!/usr/bin/env python3
"""Tests for E78 (the high per-claw knee without a dead zone; scripts/claude_pair_remapping_experiments): the plumbing of
E10_joint_fit.py's --claw-thr-subgain / --claw-thr-width (`claw_transfer_kws`, `base_kws`' transfer, `fit_one`'s and
`bouton_base`'s claw_transfer; the options off leave the kwargs as before), and E78_claw_knee.py's gate on synthetic
rows, its point list and the odor-relative knee's inhibition matrix. No simulations. Needs the shared directory (E10
reads the data-target registry at import); skipped without it."""

import json
import sys
from pathlib import Path

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
E78 = pytest.importorskip('E78_claw_knee')

ODORS = list(H.APL16_ODORS)


def test_claw_transfer_kws_and_tag():
    assert E10.claw_transfer_kws() == {}
    assert E10.claw_transfer_kws(0.25) == {'claw_thr_subgain': 0.25}
    assert E10.claw_transfer_kws(None, 15) == {'claw_thr_width': 15.0}
    assert E10.claw_transfer_kws(0, 0) == {'claw_thr_subgain': 0.0, 'claw_thr_width': 0.0}
    assert E10.claw_transfer_tag({}) == '' and E10.claw_transfer_tag(None) == ''
    assert E10.claw_transfer_tag(E10.claw_transfer_kws(0.25)) == 'sg0.25'
    assert E10.claw_transfer_tag(E10.claw_transfer_kws(0.1, 15)) == 'sg0.1w15'
    assert E10.claw_transfer_tag(E10.claw_transfer_kws(None, 15)) == 'w15'


def test_base_kws_transfer():
    for thr in (None, 40.0, 100.0):
        assert E10.base_kws(thr) == E10.base_kws(thr, 'claws', None) == E10.base_kws(thr, transfer={})
    k = E10.base_kws(100.0, transfer=E10.claw_transfer_kws(0.25, 15))
    assert k['claw_thr'] == 100.0 and k['claw_thr_subgain'] == 0.25 and k['claw_thr_width'] == 15.0
    assert {kk: v for kk, v in k.items() if kk not in ('claw_thr_subgain', 'claw_thr_width')} == E10.base_kws(100.0)
    assert 'claw_thr_subgain' not in E10.base_kws(100.0)
    with pytest.raises(ValueError):
        E10.base_kws(None, transfer={'claw_thr_subgain': 0.25})
    with pytest.raises(ValueError):
        E10.base_kws(100.0, transfer={'claw_pow': 2.0})


def _fake_base(n=12):
    idx = pd.Index([f'kc{i}' for i in range(n)], name=H.KC_ID_LEVEL)
    wnorm = {k: pd.Series(np.linspace(1, 2, n), index=idx) for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL')}
    breadth = pd.Series(np.arange(n, dtype=float), index=idx)
    initial = dict(fixed_thr=100.0, wAPLKC_scale=0.1, wKCAPL_scale=1e-4, wAPLPN_scale=0.1, wPNAPL_scale=1e-3)
    return initial, wnorm, breadth


@pytest.mark.parametrize('transfer', [None, {'claw_thr_subgain': 0.25}, {'claw_thr_subgain': 0.1, 'claw_thr_width': 15.0}])
def test_fit_one_passes_the_transfer(monkeypatch, transfer):
    import al_analysis.mb_model as mb
    seen = {}

    def fake(orn, **kw):
        seen.update(kw)
        return dict(history=pd.DataFrame(), params={}, loss=0.0)

    monkeypatch.setattr(mb, 'optimize_mb_model_params', fake)
    initial, wnorm, breadth = _fake_base()
    E10.fit_one(2.0, None, initial, wnorm, breadth, lambda o: {}, kckc0=60.0, maxfev=1, method='Nelder-Mead', thr0_factor=0.7,
        claw_thr=100.0, **({} if transfer is None else dict(claw_transfer=transfer)))
    assert seen['claw_thr'] == 100.0
    for key in ('claw_thr_subgain', 'claw_thr_width'):
        if transfer and key in transfer:
            assert seen[key] == transfer[key]
        else:
            assert key not in seen
    # everything but the transfer as without it
    base = {k: v for k, v in E10.base_kws(100.0).items()}
    assert all(seen[k] == v for k, v in base.items())


def test_fit_one_refuses_transfer_with_proxy():
    initial, wnorm, breadth = _fake_base()
    with pytest.raises(NotImplementedError):
        E10.fit_one(2.0, None, initial, wnorm, breadth, lambda o: {}, kckc0=60.0, maxfev=1, method='Nelder-Mead',
            thr0_factor=0.7, claw_thr=100.0, block_proxy=dict(sigma=0.1), claw_transfer={'claw_thr_subgain': 0.25})


def test_bouton_base_uses_the_transfer_prepass(monkeypatch, tmp_path):
    """bouton_base passes the transfer and the transfer-free run to `transfer_prepass` and takes the breadth from what
    it returns; without a transfer it is not called."""
    import al_analysis.mb_model as mb
    n = 8
    kcs = pd.MultiIndex.from_tuples([(f'kc{i}', 'g') for i in range(n)], names=[H.KC_ID_LEVEL, H.KC_TYPE_LEVEL])
    W = pd.DataFrame(np.ones((n, 2)), index=kcs, columns=pd.MultiIndex.from_tuples([('DM1', 1, 1), ('DM2', 2, 1)],
        names=['glomerulus', 'pn_id', 'bouton_id']))
    pd0 = {}
    for k in ('wAPLKC', 'wKCAPL', 'wAPLPN', 'wPNAPL'):
        pd0[k], pd0[f'{k}_scale'] = pd.Series(np.ones(n), index=kcs), 1.0
    off = pd.DataFrame(np.arange(n * 2, dtype=float).reshape(n, 2), index=kcs, columns=['a', 'b'])

    def fake_fit(orn, **kw):
        return off.astype(bool), off, W, pd0

    monkeypatch.setattr(mb, 'fit_mb_model', fake_fit)
    calls = []

    def fake_prepass(orn, kws, initial, off_spk_ref, transfer, off_path, info=None, label=None):
        calls.append(dict(kws=kws, transfer=transfer, ref=off_spk_ref, off_path=off_path))
        if info is not None:
            info['claw_transfer_prepass'] = dict(thr_factor=1.5)
        return off_spk_ref * 2

    monkeypatch.setattr(E10, 'transfer_prepass', fake_prepass)
    _, _, _, b0 = E10.bouton_base(pd.DataFrame(), tmp_path, 100.0)
    assert calls == []
    info = {}
    _, _, _, b1 = E10.bouton_base(pd.DataFrame(), tmp_path, 100.0, claw_transfer={'claw_thr_subgain': 0.25}, info=info)
    assert len(calls) == 1 and calls[0]['transfer'] == {'claw_thr_subgain': 0.25}
    assert 'claw_thr_subgain' not in calls[0]['kws'] and calls[0]['kws']['claw_thr'] == 100.0
    assert np.array_equal(b1.to_numpy(), 2 * b0.to_numpy())
    assert info['claw_transfer_prepass'] == dict(thr_factor=1.5)
    with pytest.raises(ValueError):
        E10.bouton_base(pd.DataFrame(), tmp_path, None, claw_transfer={'claw_thr_subgain': 0.25})


# ---------------------------------------------------------------------------
# E78_claw_knee.py
def _ref_row(**kw):
    r = dict(point_id='identity', start='C2', cal_err_ester=-0.069, raw_six_effect=0.085, cal_loss=0.056, rate_rho_on=0.60,
        low_2h_m5_frac_on=0.010, realism_flag_noise='')
    for od in ODORS:
        r[f'frac_on_{od}'], r[f'frac_off_{od}'] = 0.05, 0.10
    r.update(kw)
    return r


def _row(**kw):
    r = _ref_row(point_id='knee100_g0.25', cal_err_ester=-0.030, raw_six_effect=0.130, cal_loss=0.058, rate_rho_on=0.55,
        low_2h_m5_frac_on=0.006)
    for od in ODORS:
        r[f'frac_on_{od}'], r[f'frac_off_{od}'] = 0.04, 0.08
    r.update(kw)
    return r


def test_gate_point_pass():
    g = E78.gate_point(_row(), _ref_row())
    assert g['i'] and g['ii'] and g['iii'] and g['iv'] is True and g['bands'] and g['pass'] and g['pass_without_iv']
    assert g['ester_reduction'] == pytest.approx(1 - 0.030 / 0.069)
    assert g['six_raise'] == pytest.approx(0.045)
    assert g['worst_frac_ratio'] == pytest.approx(0.8)


@pytest.mark.parametrize('change, cond', [
    (dict(cal_err_ester=-0.050), 'i'),              # reduced by less than a third
    (dict(raw_six_effect=0.100), 'i'),              # six pairs raised by less than 0.03
    (dict(cal_loss=0.0605), 'ii'),                  # more than 0.004 above the start
    (dict(rate_rho_on=0.45), 'iii'),                # response rate rho below 0.50
    (dict(frac_off_ms=0.04), 'iii'),                # ms keeps less than half its APL-off fraction
    (dict(low_2h_m5_frac_on=0.004), 'iv'),          # 2h at 10^-5 below half
])
def test_gate_point_fails(change, cond):
    g = E78.gate_point(_row(**change), _ref_row())
    assert g[cond] is False and not g['pass']
    others = {'i', 'ii', 'iii', 'iv'} - {cond}
    assert all(g[c] is True for c in others)


def test_gate_point_edges():
    ref = _ref_row()
    # the error's magnitude counts: a sign flip of the same size is no reduction, a small overshoot is one
    assert not E78.gate_point(_row(cal_err_ester=+0.069), ref)['i']
    assert E78.gate_point(_row(cal_err_ester=+0.020), ref)['i']
    # the firing ceiling's points get 0.006
    assert not E78.gate_point(_row(cal_loss=0.0605), ref)['ii']
    assert E78.gate_point(_row(cal_loss=0.0605), ref, ceiling=True)['ii']
    # an odor the start does not recruit is skipped
    assert E78.gate_point(_row(frac_on_ms=0.0), _ref_row(frac_on_ms=0.0))['iii']
    # not yet run at the low concentrations: (iv) pending, not passed
    g = E78.gate_point(_row(low_2h_m5_frac_on=np.nan), ref)
    assert g['iv'] is None and g['pass_without_iv'] and not g['pass']
    # outside a band
    g = E78.gate_point(_row(realism_flag_noise='APL-on response rate 0.100 (band 0.045-0.085)'), ref)
    assert not g['bands'] and not g['pass'] and not g['pass_without_iv']


def _write_point(out: Path, row: dict):
    d = out / 'points' / row['start']
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{row['point_id']}.json").write_text(json.dumps(row))


def test_gate_selects(tmp_path):
    base = dict(start='C2', family='knee', claw_thr=100.0, claw_thr_subgain=0.25, claw_thr_width=0.0, rel=np.nan,
        start_experiment='E67', start_tag='x', k=2.0)
    _write_point(tmp_path, dict(_ref_row(), **dict(base, family='identity'), tag='C2_identity'))
    _write_point(tmp_path, dict(_row(point_id='knee100_g0.25', cal_loss=0.058), **base, tag='C2_knee100_g0.25'))
    _write_point(tmp_path, dict(_row(point_id='knee130_g0.25', cal_loss=0.057), **base, tag='C2_knee130_g0.25'))
    _write_point(tmp_path, dict(_row(point_id='knee115_g0', cal_loss=0.050, rate_rho_on=0.4), **base, tag='C2_knee115_g0'))
    _write_point(tmp_path, dict(_row(point_id='rel0.7', cal_loss=0.059), **dict(base, family='rel'), tag='C2_rel0.7'))
    res = E78.gate(tmp_path)
    assert res['n_points'] == 4 and res['n_pass'] == 3
    assert [s['tag'] for s in res['selected']] == ['C2_knee130_g0.25', 'C2_rel0.7']
    assert (tmp_path / 'gate.json').exists() and (tmp_path / 'sweep.csv').exists()


def test_point_specs():
    pts = E78.point_specs()
    ids = [p['point_id'] for p in pts]
    assert ids[:2] == ['identity', 'identity_held'] and len(ids) == len(set(ids)) == 2 + 12 + 2 + 3
    assert 'knee115_g0.5' in ids and 'knee70_g0.25' in ids and 'rel0.85' in ids
    k = next(p for p in pts if p['point_id'] == 'knee130_g0')
    assert k['transfer'] == {'claw_thr': 130.0}
    k = next(p for p in pts if p['point_id'] == 'knee100_g0.1')
    assert k['transfer'] == {'claw_thr': 100.0, 'claw_thr_subgain': 0.1}
    sw = pd.DataFrame([dict(start='C2', family='knee', point_id=p['point_id'], claw_thr=p['claw_thr'],
        claw_thr_subgain=p['claw_thr_subgain'], cal_loss=0.05 + 0.001 * i, realism_flag_noise='')
        for i, p in enumerate(pts) if p['family'] == 'knee'])
    soft = [p for p in E78.point_specs(sw) if p['family'] == 'knee_soft']
    assert [p['point_id'] for p in soft] == ['knee100_g0_w15', 'knee100_g0.1_w15']
    assert soft[1]['transfer'] == {'claw_thr': 100.0, 'claw_thr_subgain': 0.1, 'claw_thr_width': 15.0}


def test_odor_relative_inhibition():
    W = pd.DataFrame(np.ones((3, 4)), columns=pd.MultiIndex.from_tuples([('DM1', 1, i) for i in range(4)],
        names=['glomerulus', 'pn_id', 'bouton_id']))
    orn = pd.DataFrame(np.zeros((2, 3)), columns=pd.MultiIndex.from_tuples([('megamat', f'{o} @ -3') for o in ODORS[:3]],
        names=['panel', 'odor']))
    top = pd.Series({o: 100.0 + i for i, o in enumerate(ODORS)})
    m = E78.odor_relative_inhibition(W, orn, top, 0.7)
    assert m.index.equals(W.columns) and m.columns.equals(orn.columns)
    assert np.allclose(m.to_numpy(), np.tile([70.0, 70.7, 71.4], (4, 1)))


def _fake_outputs(seed, n=600):
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_tuples([('megamat', f'{o} @ -3') for o in ODORS], names=['panel', 'odor'])
    idx = pd.MultiIndex.from_tuples([(f'kc{i}', 'ab') for i in range(n)], names=[H.KC_ID_LEVEL, H.KC_TYPE_LEVEL])
    out = {}
    for c, p in (('on', 0.07), ('off', 0.15)):
        base = rng.gamma(0.6, 1.0, (n, len(ODORS))) * rng.gamma(2.0, 0.5, (n, 1))
        thr = np.quantile(base, 1 - p)
        s = pd.DataFrame(np.where(base > thr, np.rint(2 + 6 * (base - thr) / base.std()), 0.0), index=idx, columns=cols)
        out[c] = ((s >= 2).astype(float), s)
    return out


def test_sweep_writes_and_aggregates(monkeypatch, tmp_path):
    """sweep's per-point bookkeeping on a fake model (no simulations): the identity and a knee point are scored, written
    per point, aggregated into sweep.csv and the harness-readable tables (the identity left out of those), and the gate
    reads them."""
    ref, pt_out = _fake_outputs(0), _fake_outputs(1)

    class FakeModel:
        def __init__(self, start, out_dir):
            self.start, self.exp, self.tag, self.wiring, self.info, self.check = start, 'E67', 'x', {}, {}, {}
            self.p = dict(fixed_thr=100.0, wAPLKC_scale=0.1, wKCAPL_scale=1e-4, wAPLPN_scale=0.3, wPNAPL_scale=1e-3, wKCKC_scale=26.0)
            self.k, self.kws, self.ref = 2.0, dict(kc_kc_tau=0.05), ref
            self.n_on, self.n_off = int(ref['on'][0].to_numpy().sum()), int(ref['off'][0].to_numpy().sum())
            self.orn = H.apl16_orn_deltas()

        def hold(self, pt):
            return dict(self.p, fixed_thr=50.0), pt_out, dict(point_id=pt['point_id'], start=self.start, breadth_from='held',
                thr_factor=0.5, apl_factor=1.2, thr_at_bound=False, apl_at_bound=False, n_responses_on=1, n_responses_off=2,
                target_n_on=1, target_n_off=2, n_sims=3, seconds=1.0, bisections=[], breadth_sha256='x', breadth_corr_vs_start=0.9,
                runner_kws={})

    orig_r, orig_o = H.results_dir, H.outputs_dir
    monkeypatch.setattr(H, 'results_dir', lambda e: tmp_path / 'results' if e == E78.EXPERIMENT else orig_r(e))
    monkeypatch.setattr(H, 'outputs_dir', lambda e: tmp_path / 'outputs' if e == E78.EXPERIMENT else orig_o(e))
    monkeypatch.setattr(E78, 'KneeModel', FakeModel)
    monkeypatch.setattr(H, 'model_provenance', lambda **kw: {})
    out = H.results_dir(E78.EXPERIMENT)
    out.mkdir(parents=True)
    df = E78.sweep(out, 'C2', ['knee130_g0.25'])
    assert set(df.point_id) == {'identity', 'identity_held', 'knee130_g0.25'}
    k = df.set_index('point_id').loc['knee130_g0.25']
    assert k['claw_thr'] == 130.0 and k['claw_thr_subgain'] == 0.25 and k['fixed_thr'] == 50.0 and k['breadth_from'] == 'held'
    for col in ('cal_loss', 'raw_six_effect', 'cal_err_ester', 'frac_on_ms', 'realism_flag_noise', 'rate_rho_on'):
        assert col in df.columns
    sv = pd.read_csv(out / 'stats_variants.csv')
    assert 'C2_identity' not in set(sv.tag) and 'C2_knee130_g0.25' in set(sv.tag)
    mp = pd.read_csv(out / 'model_pair_distances_long.csv')
    assert set(mp.tag) == {'C2_identity_held', 'C2_knee130_g0.25'} and len(mp) == 2 * 2 * 120
    assert (tmp_path / 'outputs' / 'C' / 'C2_knee130_g0.25_on_spike_counts.parquet').exists()
    g = E78.gate(out)
    assert g['n_points'] == 2
