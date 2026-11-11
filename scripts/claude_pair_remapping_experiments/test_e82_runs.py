"""Tests of E82 part B (E82_protocol_runs): the point list, its order and prerequisites, the kwargs of each point, the
models' sources and the spike-frame comparison. No simulation, except the tests marked by E82_SIM_TEST=1 (opt-in: each
builds a model at the 0.5 s pulse, about half a minute at 8 threads, 17 in all), which check that every family's
identity (its option at the fit's own value, depletion off) reproduces its model's saved spike counts and responses
exactly, with its breadth pre-pass equal to the model's own."""
import os
import re

import numpy as np
import pandas as pd
import pytest

import E82_protocol_runs as R


def _by_id():
    return {s['point_id']: s for s in R.point_specs()}


def test_point_counts_and_ids():
    specs = R.point_specs()
    ids = [s['point_id'] for s in specs]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r'[A-Za-z0-9_.-]+', i) for i in ids)
    held = [s for s in specs if not s['identity']]
    by_part = {p: [s for s in held if s['part'] == p] for p in ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')}
    # 7 new B1 points plus E80's two re-held 2 s points; then 5, 4, 4, 5; B6 the literature setting's four wirings at 3 s;
    # B7 fit0 at 3 and 10 s, B6's four wirings at 3 s and D1 fixed and re-held, all at 0.5 s
    assert {p: len(v) for p, v in by_part.items()} == dict(B1=9, B2=5, B3=4, B4=4, B5=5, B6=4, B7=8)
    assert len([s for s in by_part['B1'] if not s['reproduces']]) == 7
    assert sorted(s['model'] for s in by_part['B1']) == sorted(R.MODELS)
    idents = [s for s in specs if s['identity']]
    # one identity per B1 model, and one per model of each other family (B6: four; B7: D1 only, its other points read
    # against the B2 / B6 identities)
    assert len(idents) == 9 + 2 + 2 + 2 + 2 + 4 + 1
    by = {s['point_id']: s for s in specs}
    for s in held:
        if s['part'] == 'B7':
            i = by[s['reference']]
            assert i['identity'] and i['model'] == s['model'] and s['reference'] in s['after'], s['point_id']
            continue
        assert any(i['part'] == s['part'] and i['model'] == s['model'] for i in idents), s['point_id']


def test_identity_kwargs_are_the_fits_own():
    by = _by_id()
    for s in by.values():
        if not s['identity']:
            continue
        # D1 was fitted at the flies' pulse, so its identity is at 2 s
        pulse = (2.0, 2.0, 2.25) if s['point_id'] == 'I_B7_D1' else (0.5, 0.5, 0.75)
        assert (s['pulse'], s['time_stim_end'], s['time_end'], s['held']) == pulse + (False,)
        assert s['after'] == [] and s['reference'] == '' and s['reproduces'] == ''
    assert by['I_B2_fit0']['opts'] == dict(kc_adapt_tau=1.0) == by['I_B2_fit1']['opts']
    for m in ('C2', 'fit0'):
        assert by[f'I_B3_{m}']['opts'] == dict(bouton_ves_p=0.0, bouton_tau_r=1.0) == by[f'I_B3_{m}']['prepass']
        assert by[f'I_B4_{m}']['kc_kc_tau'] == 0.05 and by[f'I_B4_{m}']['opts'] == {}
        assert by[f'I_B5_{m}']['opts'] == dict(apl_taum=0.05) and by[f'I_B5_{m}']['requires'] == 'apl_taum'
    for m in R.MODELS:
        s = by[f'I_B1_{m}']
        assert s['opts'] == {} and s['prepass'] == {} and s['kc_kc_tau'] == 0.05


def test_held_point_kwargs():
    by = _by_id()
    for s in by.values():
        if s['identity'] or s['part'] == 'B7':
            continue
        assert (s['pulse'], s['time_stim_end'], s['time_end'], s['held']) == (2.0, 2.0, 2.25, True)
        assert f"I_{s['part']}_{s['model']}" in s['after']
        if s['part'] != 'B1':
            assert s['reference'] == f"B1_{s['model']}" and s['reference'] in by
            assert set(R.REPRODUCTIONS) <= set(s['after'])
    assert by['B1_C2']['reproduces'] == 'C2_pulse2_held' and by['B1_fit0']['reproduces'] == 'fit0_pulse2_held'
    assert by['B1_C2']['after'] == ['I_B1_C2'] and by['B1_fit0']['after'] == ['I_B1_fit0']
    for m in R.MODELS:
        s = by[f'B1_{m}']
        assert s['opts'] == {} and s['prepass'] == {} and s['kc_kc_tau'] == 0.05
        if m not in ('C2', 'fit0'):
            assert set(R.REPRODUCTIONS) <= set(s['after'])
    assert {(s['model'], s['value']) for s in by.values() if s['part'] == 'B2' and not s['identity']} == {
        ('fit0', 0.3), ('fit0', 3.0), ('fit0', 10.0), ('fit1', 3.0), ('fit1', 10.0)}
    for s in by.values():
        if s['identity']:
            continue
        if s['part'] == 'B2':
            assert s['opts'] == dict(kc_adapt_tau=s['value']) and s['prepass'] == {}
        if s['part'] == 'B3':
            assert s['opts'] == dict(bouton_ves_p=s['value'], bouton_tau_r=1.0) == s['prepass']
            assert s['model'] in ('C2', 'fit0') and s['value'] in (0.01, 0.03)
        if s['part'] == 'B4':
            assert s['opts'] == {} and s['kc_kc_tau'] == s['value'] and s['value'] in (0.2, 1.0)
        if s['part'] == 'B5':
            assert s['opts'] == dict(apl_taum=s['value']) and s['requires'] == 'apl_taum' and s['prepass'] == {}
    assert sorted(s['value'] for s in by.values() if s['part'] == 'B5' and s['model'] == 'fit0' and not s['identity']) == [
        0.01, 0.25, 1.0]
    assert sorted(s['value'] for s in by.values() if s['part'] == 'B5' and s['model'] == 'C2' and not s['identity']) == [
        0.01, 0.25]


def test_b7_points():
    """B7: every point at the model's 0.5 s pulse; fit0 at 3 and 10 s and B6's four wirings at 3 s re-held after
    their B2 / B6 identities (population 'fc_tau3' at 0.5 s); D1 after its 2 s identity, fixed and re-held."""
    by = _by_id()
    b7 = {p: s for p, s in by.items() if s['part'] == 'B7'}
    assert set(b7) == {'B7_fit0_tau3', 'B7_fit0_tau10', 'B7_fit1_tau3', 'B7_P1_fc_tau3', 'B7_P2_fc_tau3',
                       'B7_M1_fc_tau3', 'I_B7_D1', 'B7_D1', 'B7_D1_held'}
    for p, s in b7.items():
        assert s['option'] == 'kc_adapt_tau' and s['opts'] == dict(kc_adapt_tau=s['value']) and s['prepass'] == {}
        assert s['kc_kc_tau'] == 0.05
        if p != 'I_B7_D1':
            assert (s['pulse'], s['time_stim_end'], s['time_end']) == (0.5, 0.5, 0.75)
    assert [b7[f'B7_fit0_tau{t}']['value'] for t in (3, 10)] == [3.0, 10.0]
    for t in (3, 10):
        s = b7[f'B7_fit0_tau{t}']
        assert s['held'] and s['reference'] == 'I_B2_fit0' and s['population'] == ''
    for m in R.B6_MODELS:
        s = b7[f'B7_{m}_tau3']
        assert s['held'] and s['value'] == 3.0 and s['reference'] == f'I_B6_{m}' and s['population'] == 'fc_tau3'
        assert set(R.REPRODUCTIONS) <= set(s['after'])
    assert b7['I_B7_D1']['identity'] and b7['I_B7_D1']['value'] == R.D1_TAU == 3.0
    assert not b7['B7_D1']['held'] and b7['B7_D1']['after'] == ['I_B7_D1']
    assert b7['B7_D1_held']['held'] and b7['B7_D1_held']['reference'] == 'I_B7_D1'
    assert R.model_kws(b7['B7_D1'], dict(kc=dict(kc_refractory_period=0.003, kc_adapt_inc=0.11, kc_adapt_tau=3.0))) == dict(
        kc_refractory_period=0.003, kc_adapt_inc=0.11, kc_adapt_tau=3.0, time_stim_end=0.5, time_end=0.75)


def test_order_and_prerequisites():
    specs = R.point_specs()
    ids = [s['point_id'] for s in specs]
    assert ids[:4] == ['I_B1_C2', 'I_B1_fit0', 'B1_C2', 'B1_fit0']
    pos = {p: i for i, p in enumerate(ids)}
    for s in specs:
        assert all(pos[a] < pos[s['point_id']] for a in s['after']), s['point_id']
    assert R.with_prerequisites(specs, {'B2_fit1_tau3'}) == {
        'B2_fit1_tau3', 'I_B2_fit1', 'B1_C2', 'B1_fit0', 'I_B1_C2', 'I_B1_fit0'}
    assert R.with_prerequisites(specs, {'I_B3_C2'}) == {'I_B3_C2'}


def test_populations():
    by = _by_id()
    assert set(R.POPULATIONS) == {'b2', 'fc'}
    for fam, members in R.POPULATIONS.items():
        assert len(members) == 4 and set(members) <= set(R.MODELS)
        for m in members:
            assert by[f'B1_{m}']['population'] == fam
    assert by['B1_fit0']['population'] == ''


def test_model_kws_override_and_timing():
    by = _by_id()
    m = dict(kc=dict(kc_refractory_period=0.003, kc_adapt_inc=0.04, kc_adapt_tau=1.0))
    k = R.model_kws(by['B2_fit1_tau10'], m)
    assert k == dict(kc_refractory_period=0.003, kc_adapt_inc=0.04, kc_adapt_tau=10.0, time_stim_end=2.0, time_end=2.25)
    assert R.model_kws(by['I_B1_C2'], dict(kc={})) == dict(time_stim_end=0.5, time_end=0.75)
    k = R.model_kws(by['B3_C2_ves0.03'], dict(kc={}))
    assert k == dict(bouton_ves_p=0.03, bouton_tau_r=1.0, time_stim_end=2.0, time_end=2.25)


def test_build_simulator_takes_kc_kc_tau_separately():
    with pytest.raises(ValueError):
        R.build_simulator(dict(kc_kc_tau=0.2), wPNKC=None, breadth=pd.Series(dtype=float), params={})


def test_spikes_equal():
    a = pd.DataFrame(dict(kc=[0, 3, 3], odor=[0, 0, 1], step=[10, 12, 40], time_s=[0.0, 0.001, 0.015]))
    assert R.spikes_equal(a, a.copy())
    b = a.copy()
    b.loc[2, 'step'] = 41
    assert not R.spikes_equal(a, b)
    assert not R.spikes_equal(a, a.iloc[:2])


def _shared():
    try:
        import kc_remapping_harness as H
        return (H.shared_dir() / 'results' / 'E72_wiring_population' / 'fits.csv').exists()
    except Exception:
        return False


@pytest.mark.skipif(not _shared(), reason='needs the shared directory')
@pytest.mark.parametrize('model', [m for m in R.MODELS if m not in ('C2', 'fit0')])
def test_model_sources(model):
    folder, tag, wiring = R.model_source(model)
    r = R.fit_row(folder, tag, wiring)
    if model.endswith('_fc') or model == 'fit1':
        assert 'refr3_adapt0.04142t1000' in tag
        assert float(r['kc_adapt_tau']) == R.FIT_ADAPT_TAU and abs(float(r['kc_adapt_inc']) - 0.0414) < 1e-3
    else:
        assert pd.isna(r['kc_refractory_period'])
    assert wiring == (None if model == 'fit1' else model.split('_')[0])


@pytest.mark.skipif(not _shared(), reason='needs the shared directory')
def test_d1_parts():
    """D1 is part D's fit at 2 s with the 3 s adaptation on the real wiring, and its breadth is the 2 s base run's,
    which the breadth pre-pass of its identity point reproduces."""
    folder, tag, wiring = R.model_source('D1')
    r = R.fit_row(folder, tag, wiring)
    assert wiring is None and float(r['kc_adapt_tau']) == 3.0 and float(r['time_stim_end']) == 2.0
    m = R.model_parts('D1')
    assert m['kc']['kc_adapt_tau'] == 3.0 and set(m['kc']) == set(R.KC_SPIKING)
    assert m['target_on'] == int((m['saved']['on'][0].to_numpy() > 0.5).sum())
    b = R.point_breadth(_by_id()['I_B7_D1'], m)
    assert b.index.equals(m['breadth'].index) and np.array_equal(b.to_numpy(), m['breadth'].to_numpy())


@pytest.mark.skipif(not _shared(), reason='needs the shared directory')
def test_population_rows_match_e72(tmp_path):
    """The populations at 0.5 s, from the four saved fits written as identity points, equal E72 part G's (the
    breadth-2 family 'none', the firing ceiling '40'; the same wirings in another order)."""
    import kc_remapping_harness as H
    outputs = tmp_path / 'runs'
    outputs.mkdir()
    pids = []
    for m in R.POPULATIONS['b2'][1:] + R.POPULATIONS['fc']:
        folder, tag, _ = R.model_source(m)
        import E72_wiring_population as E72
        saved = E72.load_fit(folder, tag)
        R._save_outputs(outputs, f'I_B1_{m}', saved, {'on': pd.DataFrame(), 'off': pd.DataFrame()})
        pids.append(f'I_B1_{m}')
    import E77_input_screen as S
    c2 = E72.load_fit(*S.C2)
    R._save_outputs(outputs, 'I_B1_C2', c2, {'on': pd.DataFrame(), 'off': pd.DataFrame()})
    pd.DataFrame(dict(point_id=pids + ['I_B1_C2'])).to_csv(tmp_path / 'runs.csv', index=False)
    df = R.population_rows(tmp_path, outputs).set_index('population')
    assert list(df.pulse) == [0.5, 0.5]
    ref = pd.read_csv(H.results_dir('E72_wiring_population') / 'partG_population.csv').set_index('family')
    for fam, g in (('b2', 'none'), ('fc', '40')):
        for c in ('raw_loss', 'raw_mean', 'raw_pattern', 'cal_loss', 'cal_pattern', 'binary_loss', 'rate_on', 'ratio'):
            assert np.isclose(df.loc[fam, c], ref.loc[g, c], rtol=1e-9, atol=0), (fam, c, df.loc[fam, c], ref.loc[g, c])


SIM = os.environ.get('E82_SIM_TEST') == '1'


@pytest.mark.skipif(not SIM, reason='builds a model at 0.5 s (about half a minute each); set E82_SIM_TEST=1')
@pytest.mark.parametrize('pid', [s['point_id'] for s in R.point_specs() if s['identity']])
def test_identity_reproduces_saved(pid):
    spec = _by_id()[pid]
    if spec['requires'] == 'apl_taum' and not R.apl_taum_supported():
        pytest.skip("mb_model has no apl_taum option yet")
    m = R.model_parts(spec['model'])
    o, res, spikes, rec = R.run_spec(spec)
    ok, chk = R.check_point(spec, m, o, res, spikes, rec['breadth'])
    assert ok, chk
    for c in ('on', 'off'):
        counts = o[c][1].to_numpy(dtype=float)
        assert len(spikes[c]) == int(counts.sum())
