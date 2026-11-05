"""Recording spikes from an already-tuned olfsysm model, by setting a `save_*` flag after
the `RunVars` were constructed and re-running only the KC layer.

Before the olfsysm fix (allocate recording buffers on demand in `run_KC_sims`), this
indexed an empty vector inside the parallel region (`IndexError: vector::_M_range_check`).
"""
import numpy as np
import pytest

from al_analysis.mb_model import fit_mb_model, megamat_orn_deltas, get_sim_var


pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')


def test_save_spike_recordings_after_construction():
    import olfsysm as osm

    orn_deltas = megamat_orn_deltas()
    responses, spike_counts, _, param_dict = fit_mb_model(orn_deltas,
        pn2kc_connections='hemibrain', weight_divisor=20, target_sparsity=0.1,
        make_plots=False, silent=True, return_olfsysm_vars=True
    )
    mp, rv = param_dict['mp'], param_dict['rv']
    assert len(rv.kc.spike_recordings) == 0 and len(rv.kc.vm_sims) == 0

    mp.kc.save_spike_recordings = True
    mp.kc.save_vm_sims = True
    rv.log.disable()
    rv.verbose = False
    osm.run_KC_sims(mp, rv, False)

    n_odors = len(responses.columns)
    assert len(rv.kc.spike_recordings) == n_odors
    assert len(rv.kc.vm_sims) == n_odors

    spikes = get_sim_var(rv.kc, 'spike_recordings')
    assert spikes.shape[:2] == (n_odors, len(responses))
    # same thresholds and weights, so re-running reproduces the tuned model's outputs,
    # and the recorded spikes sum to the returned spike counts
    totals = np.nan_to_num(spikes).sum(axis=-1).T
    assert np.array_equal(totals, spike_counts.to_numpy())
    assert np.array_equal(np.array(rv.kc.responses), responses.to_numpy().astype(float))

    # a second run with the flag turned off again must not touch the saved buffers
    mp.kc.save_spike_recordings = False
    osm.run_KC_sims(mp, rv, False)
    assert len(rv.kc.spike_recordings) == n_odors
