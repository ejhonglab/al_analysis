
from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd
import pytest

from al_analysis import trial_response_traces, delta_f_over_f, compute_trial_stats


# TODO better way to store (+ reference paths to) test_data for pytest?
# TODO refactor (put where?)? also used in test_mb_model.test_multiresponder_APL_boost
# now
data_dir = Path(__file__).resolve().parent / 'test_data'

@pytest.fixture(scope='session')
def odor_and_frame_metadata():
    """Returns odor_lists, bounding_frames. Metadata from one recording.
    """
    odor_lists = json.loads(Path(data_dir / 'odor_lists.json').read_text())

    bounding_frames = json.loads(Path(data_dir / 'bounding_frames.json').read_text())

    last_end_frame = None
    # TODO clarify if first_odor_frame is first frame after onset, or that contains
    # onset, or what (but not here... in some hong2p unit test, if not already there)?
    for start_frame, first_odor_frame, end_frame in bounding_frames:
        assert start_frame < first_odor_frame < end_frame

        # this is more a test of something else, but should probably always be true
        if last_end_frame is not None:
            assert start_frame == last_end_frame + 1
        else:
            assert start_frame == 0

        last_end_frame = end_frame

    assert len(bounding_frames) == len(odor_lists)
    return odor_lists, bounding_frames


@pytest.fixture(scope='session')
def trace_data(odor_and_frame_metadata):
    """Returns traces (frames X ROIs), bounding_frames, odor_lists.

    Data from one recording.
    """
    odor_lists, bounding_frames = odor_and_frame_metadata

    # this data is output of extract_traces_bool_masks (raw F values within each of 87
    # ROIs, from a pebbled 6f diagnostic recording). 675 frames (=len(traces)).
    #
    # index.name should be 'frame', and columns.name should be 'roi' (both int
    # RangeIndices here. yes, no glomerulus names at this point).
    traces = pd.read_csv(data_dir / 'traces.csv', index_col=0)
    traces.columns.name = 'roi'
    assert traces.index.name == 'frame'
    assert not traces.isna().any().any()
    #

    return traces, odor_lists, bounding_frames


# TODO actually test compute_trial_stats

def test_trial_response_traces(trace_data):
    # trial_response_traces doesn't actually use odor_lists (and its current use of
    # optional odor_index= kwarg is limited)
    traces, odor_lists, bounding_frames = trace_data

    # TODO test w/ input other than DataFrame? np.ndarray / xarray.DataArray?

    # TODO test w/ odor_index=<index-from-odor_lists>? (only currently used for unused
    # one_baseline_per_odor branch, so not that important)

    # TODO test that n_volumes_for_baseline=None output is same as setting it to
    # explicitly number of frames we have available (get from bounding frames, but may
    # need to also check this test data has same # of pre-odor frames on each trial)
    # TODO TODO also, how does n_volumes_for_baseline work if # pre-odor frames is
    # variable? that a real concern? should it be using the same # of frames in each
    # case?
    # TODO test w/ n_volume_for_baseline > # available? should fail probably?

    # explicitly setting this just b/c al_analysis.py sets kwarg defaults (for these fns
    # being tested) to this value. so now the tests code won't depend on whether that
    # module-level default change in al_analysis (but it probably will stay `= None`,
    # which uses all frames)
    kws = dict(n_volumes_for_baseline=None)

    # TODO force al_analysis.stat=np.mean, or also test all combinations w/ that?
    # (trial_response_traces has some _checks=True code [in zscore=True path] that only
    # worked for that, and is currently broken for new default of
    # stat=sign_preserving_maxabs, but don't think it matters)
    for zscore, keep_pre_odor in product((False, True), (False, True)):

        trial_trace_iter = trial_response_traces(traces, bounding_frames, zscore=zscore,
            keep_pre_odor=keep_pre_odor, _checks=True, **kws
        )
        trial_traces_list = list(trial_trace_iter)

        assert len(trial_traces_list) == len(bounding_frames)

        for trial_traces, frames in zip(trial_traces_list, bounding_frames):
            start_frame, first_odor_frame, end_frame = frames

            assert trial_traces.index.name == 'frame'
            if not keep_pre_odor:
                assert trial_traces.index[0] == first_odor_frame
            else:
                assert trial_traces.index[0] == start_frame

            assert trial_traces.index[-1] == end_frame
            # we have indices of all frames between where we start and where we end
            # (b/c they only ever change by +1)
            assert set(np.diff(trial_traces.index)) == {1}

            assert trial_traces.columns.equals(traces.columns)

            assert not trial_traces.isna().any().any()

        # trial_response_traces output should be same as delta_f_over_f, when
        # zscore=False
        if not zscore:
            trace_list2 = list(delta_f_over_f(traces, bounding_frames,
                keep_pre_odor=keep_pre_odor, **kws
            ))
            assert len(trial_traces_list) == len(trace_list2)

            for x, y in zip(trial_traces_list, trace_list2):
                assert x.equals(y)

