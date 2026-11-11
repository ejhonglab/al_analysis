#!/usr/bin/env python3
"""Tests for E83 (the input and APL's glomerulus profile fitted on several wirings at once;
scripts/claude_pair_remapping_experiments/E83_cross_wiring_fit.py): the identity checks the entry asks for before any run.
Without simulations: the population objective at the fits equals the probe's values (S's pattern part 0.0569 against
half A, 0.0552 against half B), the targets, part A1's profiles equal E75's, inputs at size 0 are the lab's exactly and
E77's searched inputs at full size are its inputs. With simulations (about 1 min per wiring on 16 threads): every one of
the eight wirings' simulators reproduces its saved fit exactly at factor 1 and with zero scores, and on the real wiring
E75's two archived profile points are reproduced exactly. Needs the shared directory; skipped without it."""

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
E83 = pytest.importorskip('E83_cross_wiring_fit')


@pytest.mark.parametrize('half', ['A', 'B'])
def test_population_objective_at_identity(half):
    v = E83.PopulationObjective(E83.half_target(half))({})
    assert v['n_pairs'] == 120
    assert round(v['rms'], 4) == E83.IDENTITY_OBJECTIVE[half]
    assert v['sse'] == pytest.approx(v['rms'] ** 2 * 120)
    # the saved counts passed explicitly give the same value to the last digit
    on = {w: E83.saved_fit(w)['saved']['on'][1] for w in E83.S_WIRINGS}
    off = {w: E83.saved_fit(w)['saved']['off'] for w in E83.S_WIRINGS}
    assert E83.PopulationObjective(E83.half_target(half))(on, off)['sse'] == v['sse']


def test_populations_reproduce_the_probe():
    d = {w: E83.pair_distances(E83.saved_fit(w)['saved'], 'cal') for w in E83.ALL_WIRINGS}
    probe = {'S': (0.0569, 0.0552, 0.0500, 0.0499), 'T': (0.0565, 0.0538, 0.0486, 0.0487),
             'all8': (0.0558, 0.0535, 0.0482, 0.0482)}
    for name, vals in probe.items():
        sc = E83.effect_scores(E83.population([d[w] for w in E83.POPULATIONS[name]]))
        got = tuple(round(sc[f'{t}_pattern'], 4) for t in ('halfA', 'halfB', 'z8', 'z6'))
        assert got == vals, name
    # the eight wirings' error against all recordings is E72's common error
    import E75_glomerulus_apl_profile as E75
    c = E83._cs(E75.common_error('cal'))
    e = E83.pattern_error(E83.population([d[w] for w in E83.ALL_WIRINGS]), E83.targets()['z8']).reindex(c.index)
    assert float(np.max(np.abs(e.to_numpy() - c.to_numpy()))) < 1e-9


def test_targets():
    t = E83.targets()
    assert tuple(t) == E83.TARGETS
    for v in t.values():
        assert len(v) == 120 and not v.isna().any()
    assert float(t['z8'].mean()) == pytest.approx(0.178, abs=0.001)


def test_a1_profiles_are_e75s():
    m = pd.read_csv(H.results_dir('E75_glomerulus_apl_profile') / 'sensitivity_map.csv').set_index('point_id')
    for name, pid in E83.E75_POINT_IDS.items():
        assert json.loads(m.loc[pid].scores) == E83.e75_profile(name), name
    assert E83.scale_scores({'DM1': 0.5}, 0.6) == {'DM1': 0.3}


def test_inputs_at_size():
    orn = H.apl16_orn_deltas()
    for X in ('A', 'B'):
        F = E83.read_factors(H.shared_dir() / 'results' / 'E77_input_side_common_error' / f'oracle_factors_{X}.csv')
        assert E83.input_at(F, 0.0).equals(orn.astype(float))
        full = E83.a2_input(X, 1.0)
        part = E83.a2_input(X, E83.GATE_SIZE)
        lo = np.log(np.abs(part.to_numpy()) + 1e-12) - np.log(np.abs(orn.to_numpy()) + 1e-12)
        hi = np.log(np.abs(full.to_numpy()) + 1e-12) - np.log(np.abs(orn.to_numpy()) + 1e-12)
        nz = np.abs(orn.to_numpy()) > 1e-6
        assert np.allclose(lo[nz], E83.GATE_SIZE * hi[nz], atol=1e-9)
    assert E83.input_tag_for(orn).startswith('E83_') and E83.input_tag_for(orn) == E83.input_tag_for(orn.copy())


@pytest.mark.parametrize('label', list(E83.ALL_WIRINGS))
def test_wiring_reproduces_its_fit(label):
    """Factor 1 (checked in the constructor) and zero scores reproduce the wiring's saved counts exactly."""
    m = E83.WiringModel(label, check=True)
    m.set_profile('cell', {g: 0.0 for g in set(m.W.columns.get_level_values('glomerulus'))})
    r, s = m.run_on(1.0)
    assert np.array_equal(s.to_numpy(dtype=float), m.saved['on'][1].to_numpy(dtype=float))
    assert np.array_equal(r.to_numpy(dtype=float), m.saved['on'][0].to_numpy(dtype=float))
    m.set_profile(None, {})
    assert np.array_equal(m._kc, m._kc0) and np.array_equal(m._pn, m._pn0)


def test_e75_archived_points_on_the_real_wiring():
    m = E83.WiringModel('real', check=True)
    sm = pd.read_csv(H.results_dir('E75_glomerulus_apl_profile') / 'sensitivity_map.csv').set_index('point_id')
    for name, tag in E83.E75_ARCHIVED.items():
        rd = E83.profile_reading(m, E83.e75_profile(name), 'cell', label=name)
        assert rd['apl_factor'] == float(sm.loc[E83.E75_POINT_IDS[name], 'factor'])
        assert E83._same({'on': rd['on'], 'off': rd['off']}, E83._e75_archived(tag)) == (0, 0), tag
