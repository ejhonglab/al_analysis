"""E43_model_trial_noise.fit_model refuses a fit made at another odor pulse or APL time constant (E10's --stim-end /
--sim-end / --apl-taum, E82) instead of rebuilding it at olfsysm's defaults; a row at the defaults or without the
columns gets past that check. No simulation: the row is a stand-in and the build stops at the next check."""
import numpy as np
import pandas as pd
import pytest

import E43_model_trial_noise as E43


class _Stop(Exception):
    pass


def _row(**kw) -> pd.Series:
    return pd.Series(dict(dict(apl_drive='claws', base_params=np.nan, claw_thr=np.nan), **kw))


@pytest.fixture
def fake_fit(monkeypatch):
    rows = {}

    def fit_row(experiment, tag):
        return rows[tag]

    monkeypatch.setattr(E43, 'fit_row', fit_row)
    monkeypatch.setitem(E43.FIT_MODELS, 'timing_test', ('E82_flies_protocol', 'timing_test'))
    return rows


@pytest.mark.parametrize('kw', [dict(time_stim_end=2.0, time_end=2.25), dict(time_end=1.0), dict(apl_taum=0.25)])
def test_refuses_other_timing(fake_fit, kw):
    fake_fit['timing_test'] = _row(**kw)
    with pytest.raises(NotImplementedError, match='another pulse or APL time constant'):
        E43.fit_model('timing_test', pd.DataFrame(), None)


@pytest.mark.parametrize('kw', [dict(), dict(time_stim_end=np.nan, time_end=np.nan, apl_taum=np.nan),
                                dict(time_stim_end=0.5, time_end=0.75, apl_taum=0.05)])
def test_defaults_pass(fake_fit, monkeypatch, kw):
    import olfsysm as osm
    mp = osm.ModelParams()
    assert (mp.time_stim_end, mp.time_end, mp.kc.apl_taum) == (0.5, 0.75, 0.05)
    fake_fit['timing_test'] = _row(**kw)
    # the next step after the checks reads the fit's claw threshold; stop there
    monkeypatch.setattr(E43, '_given', _stop_at(E43._given, 'claw_thr'))
    with pytest.raises(_Stop):
        E43.fit_model('timing_test', pd.DataFrame(), None)


def _stop_at(given, col):
    def g(row, c):
        if c == col:
            raise _Stop
        return given(row, c)
    return g
