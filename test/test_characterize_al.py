"""Characterization tests of the AL side: the `al-analysis`, `plot-roi` and `csvinfo`
commands (and the response calculation under them), with outputs frozen under
`test_data/characterization/al/` (see `characterize.py`).

`al-analysis` is run end-to-end on real recordings from the NAS (skipped if
`/local/tensor/Tom/2p_data` is not mounted): one recording under QUICK=1 (~1.5 min), and
two whole flies (with the across-fly steps, `-A`) under QUICK=0 (~10 min). The NAS inputs
are only read: the test builds its own data tree (symlinks to the raw data and motion
corrected TIFFs, copies of the ImageJ ROIs), and checks afterwards that nothing under the
NAS paths it used changed. Each run is a subprocess (`al_cli_driver.py`, which lists the
workarounds it applies).

Not covered (by any test here):
- suite2p registration (`register_all_fly_recordings_together`) and the raw -> TIFF
  conversion: switched off in `al_cli_driver.py`, and the motion corrected TIFFs from
  the NAS used instead.
- the model step of `al-analysis` (`-s` skips it, as by default).
- `plot-roi`'s client / server layer (`plot_roi.main`, a socket server for ImageJ) and
  interactive plotting. `plot_roi_util.load_and_plot` is driven directly instead.
- the Google Sheet download (`get_gsheet_metadata`), replaced by a constructed frame.
- running on GH146 (PN) data: the ORN run reads committed GH146 outputs instead.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import pytest

from .conftest import test_dir, test_data_dir, repo_root
from .characterize import check_frozen


QUICK: bool = bool(int(os.environ.get('QUICK', False)))

al_data_root: Path = repo_root / 'al_analysis' / 'data'

nas_2p_data: Path = Path('/local/tensor/Tom/2p_data')

# two flies of the final megamat ORN data (in the sent_to_remy CSVs, and the model
# input). the across-fly steps need at least 2 flies, and flies after 2023-04-22 (VA / AA
# are nulled for flies up to then, and with only such flies the across-fly checks of
# odor consensus fail).
E2E_DATE: str = '2023-04-26'
E2E_FLY: int = 2
E2E_FLIES: Tuple[int, ...] = (2, 3)
E2E_RECORDINGS: Tuple[str, ...] = ('diagnostics1', 'diagnostics2', 'diagnostics3',
    'megamat1', 'megamat2'
)
# for the QUICK=1 (single recording) run, on fly E2E_FLY
E2E_QUICK_RECORDING: str = 'diagnostics1'
# the one recording (of the whole-fly run) whose frame <-> odor assignment is computed
# from ThorSync data (the others use the NAS caches, as reading the ThorSync files
# takes ~1 min per recording)
E2E_THORSYNC_RECORDING: Tuple[int, str] = (2, 'diagnostics3')

# a CSV / pickle pair of consensus ORN responses (megamat panel, 9 flies)
megamat_csv_dir: Path = (al_data_root / 'sent_to_remy' /
    '2025-09-30_tom_orn_data_signed-max' / 'megamat_signed-max'
)


def _nas_available() -> bool:
    try:
        return (nas_2p_data / 'raw_data' / E2E_DATE / str(E2E_FLY)).is_dir()
    except OSError:
        return False


requires_nas = pytest.mark.skipif(not _nas_available(),
    reason=f'NAS data not reachable ({nas_2p_data} not mounted?)'
)


###################################################################################
# response calculation, from committed traces (no NAS needed)
###################################################################################

@pytest.fixture(scope='module')
def trace_data() -> Tuple[pd.DataFrame, list, list]:
    """Returns (traces, odor_lists, bounding_frames) committed for one recording.

    Same data as in `test_al_analysis.py`: raw F of 87 ImageJ ROIs (675 frames) from a
    pebbled 6f diagnostic recording.
    """
    traces = pd.read_csv(test_data_dir / 'traces.csv', index_col=0)
    traces.columns.name = 'roi'
    odor_lists = json.loads((test_data_dir / 'odor_lists.json').read_text())
    bounding_frames = json.loads((test_data_dir / 'bounding_frames.json').read_text())
    return traces, odor_lists, bounding_frames


@pytest.mark.parametrize('variant', ['default', 'zscore', 'mean', 'max-all-frames'])
def test_compute_trial_stats(trace_data, variant):
    """Freezes `compute_trial_stats` output (trials x ROIs) for a few response calcs.

    'default' uses the module defaults (`al_util.response_stat_fn`, currently
    `sign_preserving_maxabs`, over the first `n_volumes_for_response` volumes), 'mean'
    the mean over 2 volumes (as for the paper outputs, `-R`), 'zscore' Z-scored traces.
    """
    from al_analysis.al_analysis import compute_trial_stats

    traces, odor_lists, bounding_frames = trace_data
    kws = dict()
    if variant == 'zscore':
        kws = dict(zscore=True)
    elif variant == 'mean':
        kws = dict(stat=np.mean, n_volumes_for_response=2)
    elif variant == 'max-all-frames':
        kws = dict(stat=np.max, n_volumes_for_response=None)

    trial_stats = compute_trial_stats(traces, bounding_frames, odor_lists, **kws)
    assert trial_stats.shape == (len(bounding_frames), traces.shape[1])
    check_frozen(f'al/compute_trial_stats/{variant}', trial_stats)


def test_delta_f_over_f(trace_data):
    """Freezes per-trial dF/F traces (`delta_f_over_f`, pre-odor frames kept)."""
    from al_analysis.al_analysis import delta_f_over_f

    traces, _, bounding_frames = trace_data
    dff = pd.concat(list(delta_f_over_f(traces, bounding_frames, keep_pre_odor=True,
        n_volumes_for_baseline=None
    )))
    assert dff.shape == traces.shape
    check_frozen('al/delta_f_over_f', dff)


###################################################################################
# csvinfo
###################################################################################

def _fake_csvinfo_gsheet(csv: Path):
    """Returns a stand-in for `get_gsheet_metadata`, for the flies in `csv`.

    The first fly is left out of the sheet and the last is marked excluded, so both of
    the csvinfo messages about those are exercised.
    """
    from al_analysis.al_util import read_csv

    df = read_csv(csv, verbose=False, warn_=False)
    flies = df.columns.to_frame(index=False)[['date', 'fly_num']].drop_duplicates()
    flies = flies.rename(columns={'fly_num': 'fly'}).iloc[1:]
    index = pd.MultiIndex.from_frame(flies)
    gsheet = pd.DataFrame({'exclude': [False] * len(index)}, index=index)
    gsheet.iloc[-1, 0] = True

    def get_gsheet_metadata():
        return gsheet.copy()

    return get_gsheet_metadata


def _normalize_output(text: str, tmp_path: Path) -> str:
    """Strips machine / run specific parts (temp paths, file mtimes) from printed text.
    """
    lines = []
    for line in text.replace(str(tmp_path), '<tmp>').splitlines():
        if line.startswith('modified '):
            continue
        lines.append(line.rstrip())
    return '\n'.join(lines) + '\n'


@pytest.fixture
def megamat_csv(tmp_path) -> Path:
    """Returns path to a copy (in `tmp_path`) of committed megamat consensus CSV (+ .p).
    """
    for name in ('ij_certain-roi_stats.csv', 'ij_certain-roi_stats.p'):
        shutil.copy2(megamat_csv_dir / name, tmp_path / name)

    return tmp_path / 'ij_certain-roi_stats.csv'


@pytest.mark.parametrize('quiet', [False, True], ids=['verbose', 'quiet'])
def test_csvinfo(monkeypatch, capsys, tmp_path, megamat_csv, quiet):
    """Freezes what `csvinfo <csv> [-q]` prints for a committed consensus CSV.
    """
    import al_analysis.load_antennal_csv as lac

    monkeypatch.setattr(lac, 'get_gsheet_metadata', _fake_csvinfo_gsheet(megamat_csv))
    argv = ['csvinfo', str(megamat_csv)]
    if quiet:
        argv.append('-q')
    monkeypatch.setattr(sys, 'argv', argv)

    capsys.readouterr()
    lac.csvinfo_cli()
    out = _normalize_output(capsys.readouterr().out, tmp_path)

    # (read_csv also checks the CSV against the pickle next to it)
    assert 'CSV data matches pickle' in out
    check_frozen(f'al/csvinfo/megamat_{"quiet" if quiet else "verbose"}', out)


def test_csvinfo_plots(monkeypatch, capsys, tmp_path, megamat_csv):
    """`csvinfo <csv> -p vmax=1.5` writes its 4 PDFs next to the CSV.

    Also freezes the frames it plots (trial means, mean and SD over flies, and counts).
    """
    import matplotlib
    matplotlib.use('agg')
    import al_analysis.load_antennal_csv as lac

    plotted: List[pd.DataFrame] = []
    orig_plot = lac.plot_all_roi_mean_responses
    orig_plot_n = lac.plot_n_per_odor_and_glom

    def plot_all_roi_mean_responses(df, *args, **kwargs):
        plotted.append(df.copy())
        return orig_plot(df, *args, **kwargs)

    def plot_n_per_odor_and_glom(df, *args, **kwargs):
        plotted.append(df.copy())
        return orig_plot_n(df, *args, **kwargs)

    monkeypatch.setattr(lac, 'plot_all_roi_mean_responses', plot_all_roi_mean_responses)
    monkeypatch.setattr(lac, 'plot_n_per_odor_and_glom', plot_n_per_odor_and_glom)
    monkeypatch.setattr(lac, 'get_gsheet_metadata', _fake_csvinfo_gsheet(megamat_csv))
    monkeypatch.setattr(sys, 'argv', ['csvinfo', str(megamat_csv), '-q', '-p',
        'vmax=1.5'
    ])
    lac.csvinfo_cli()
    capsys.readouterr()

    stem = megamat_csv.stem
    for suffix in ('', '_mean', '_stddev', '_n-per-odor-and-glom'):
        pdf = tmp_path / f'{stem}{suffix}.pdf'
        assert pdf.is_file() and pdf.stat().st_size > 0, f'{pdf} not written'

    assert len(plotted) == 4
    # (counts are plotted from the same trial means)
    assert plotted[3].equals(plotted[0])
    for name, df in zip(('trialmean', 'mean', 'stddev'), plotted[:3]):
        check_frozen(f'al/csvinfo/megamat_plot_{name}', df)


###################################################################################
# plot-roi (plot_roi_util), from cached responses (no NAS needed)
###################################################################################

def _load_and_plot_args(**kwargs):
    """Returns `argparse.Namespace` as `plot_roi.main` would send to `load_and_plot`.

    debug=True, so plotting is called directly rather than in a new Process.
    """
    import argparse
    defaults = dict(roi_strs=[], analysis_dir=None, roi_index=None, roiset_path=None,
        no_compare=False, pairs=False, other_odors=False, add=False, hallem=False,
        debug=True
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture
def plot_roi_util(monkeypatch):
    """Returns `al_analysis.plot_roi_util`, with plotting recorded instead of shown.

    `plot` (which would block in `plt.show()`) is replaced by a function recording its
    inputs to the returned module's `_recorded_plot_calls`, and module-level state from
    previous calls is reset.
    """
    import matplotlib
    matplotlib.use('agg')
    import al_analysis.plot_roi_util as pru

    calls = []

    def plot(df, **kwargs):
        calls.append((df, kwargs))

    monkeypatch.setattr(pru, 'plot', plot)
    monkeypatch.setattr(pru, '_recorded_plot_calls', calls, raising=False)
    for name, value in (('newly_analyzed_dfs', []), ('newly_analyzed_roi_names', []),
            ('newly_analyzed_roi_strs', []), ('keep_comparison_to_cache', False),
            ('most_recent_plot_proc', None), ('plotting_processes', [])):
        monkeypatch.setattr(pru, name, value)

    return pru


def test_plot_roi_cached(plot_roi_util, monkeypatch, tmp_path):
    """`plot-roi DM2 VA2` (no -d): the frame it would plot, from cached ROI responses.

    Reads `ij_roi_stats.p` (`al_analysis.ij_roi_responses_cache`) from the current
    directory, here a copy of committed megamat ORN responses (9 flies).
    """
    shutil.copy2(megamat_csv_dir / 'ij_roi_stats.p', tmp_path / 'ij_roi_stats.p')
    monkeypatch.chdir(tmp_path)

    plot_roi_util.load_and_plot(_load_and_plot_args(roi_strs=['DM2', 'VA2']))

    calls = plot_roi_util._recorded_plot_calls
    assert len(calls) == 1
    df, kwargs = calls[0]
    assert kwargs == {}
    assert set(df.columns.get_level_values('roi')) == {'DM2', 'VA2'}
    check_frozen('al/plot_roi/cached_DM2_VA2', df)


def test_plot_roi_plot_fn(monkeypatch, tmp_path):
    """`plot_roi_util.plot` makes its figure (Agg backend, so `plt.show` returns).
    """
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import al_analysis.plot_roi_util as pru

    shutil.copy2(megamat_csv_dir / 'ij_roi_stats.p', tmp_path / 'ij_roi_stats.p')
    monkeypatch.chdir(tmp_path)

    calls = []
    monkeypatch.setattr(pru, 'plot', lambda df, **kwargs: calls.append(df))
    for name, value in (('newly_analyzed_dfs', []), ('newly_analyzed_roi_names', []),
            ('newly_analyzed_roi_strs', []), ('plotting_processes', [])):
        monkeypatch.setattr(pru, name, value)
    pru.load_and_plot(_load_and_plot_args(roi_strs=['DM2']))
    monkeypatch.undo()

    figs_before = set(plt.get_fignums())
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    pru.plot(calls[0])
    new_figs = [plt.figure(n) for n in set(plt.get_fignums()) - figs_before]
    assert len(new_figs) == 1
    fig = new_figs[0]
    assert fig.df.equals(calls[0])
    plt.close(fig)


###################################################################################
# al-analysis end-to-end (NAS)
###################################################################################

def _snapshot(paths: List[Path]) -> Dict[str, Tuple[float, int]]:
    """Returns {path: (mtime, size)} for the files under each of `paths` (following
    directory symlinks at the top level only), to check that nothing was written there.
    """
    snapshot = dict()
    for p in paths:
        p = p.resolve()
        files = [p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()]
        for f in files:
            st = f.stat()
            snapshot[str(f)] = (st.st_mtime, st.st_size)
    return snapshot


def _build_e2e_tree(root: Path, fly2recordings: Dict[int, Tuple[str, ...]], *,
    fly_concat: bool, thorsync_recordings: Tuple[Tuple[int, str], ...] = ()
    ) -> Tuple[Path, List[Path]]:
    """Builds a hong2p data tree for NAS flies (all imaged on `E2E_DATE`) under `root`.

    Raw ThorImage / ThorSync directories, the stimulus YAMLs and the motion corrected
    TIFFs are symlinks to the NAS (inputs only read). Each fly's ImageJ RoiSet.zip is
    copied (the other recordings link to it, as on the NAS).

    Args:
        fly2recordings: fly number -> names of its recordings to include

        fly_concat: whether to also link each fly's `mocorr_concat.tif` (all recordings
            concatenated), which makes the later recordings' ROIs link to the first
            recording's, and makes `al-analysis` compute an all-frame average from it.

        thorsync_recordings: (fly, recording) whose frame <-> odor assignment is
            computed from ThorSync data. For all others, the NAS
            `trial_bounding_frames.yaml` caches are copied.

    Returns (data root, NAS paths used).
    """
    data = root / 'data'
    nas_used = [nas_2p_data / 'stimulus_data_files']
    (data / 'raw_data' / E2E_DATE).mkdir(parents=True)
    (data / 'stimulus_data_files').symlink_to(nas_2p_data / 'stimulus_data_files')

    for fly, recordings in fly2recordings.items():
        nas_raw_fly = nas_2p_data / 'raw_data' / E2E_DATE / str(fly)
        nas_fly = nas_2p_data / 'analysis_intermediates' / E2E_DATE / str(fly)
        nas_used.append(nas_raw_fly)

        raw_fly = data / 'raw_data' / E2E_DATE / str(fly)
        raw_fly.mkdir()
        for d in sorted(nas_raw_fly.iterdir()):
            (raw_fly / d.name).symlink_to(d)

        fly_dir = data / 'analysis_intermediates' / E2E_DATE / str(fly)
        (fly_dir / 'suite2p').mkdir(parents=True)
        for rec in recordings:
            rec_dir = fly_dir / rec
            rec_dir.mkdir()

            nas_tiff = nas_fly / 'suite2p' / f'{rec}.tif'
            nas_used.append(nas_tiff)
            (fly_dir / 'suite2p' / f'{rec}.tif').symlink_to(nas_tiff)
            # (al_analysis asserts this is a symlink)
            (rec_dir / 'mocorr.tif').symlink_to(Path('..') / 'suite2p' / f'{rec}.tif')

            nas_roiset = nas_fly / rec / 'RoiSet.zip'
            if nas_roiset.is_symlink():
                (rec_dir / 'RoiSet.zip').symlink_to(os.readlink(nas_roiset))
            elif nas_roiset.exists():
                shutil.copy2(nas_roiset, rec_dir / 'RoiSet.zip')
                nas_used.append(nas_roiset)

            if (fly, rec) not in thorsync_recordings:
                bounding_frames = nas_fly / rec / 'trial_bounding_frames.yaml'
                shutil.copy2(bounding_frames, rec_dir / bounding_frames.name)
                nas_used.append(bounding_frames)

        if fly_concat:
            nas_concat = nas_fly / 'suite2p' / 'mocorr_concat.tif'
            nas_used.append(nas_concat)
            (fly_dir / 'suite2p' / 'mocorr_concat.tif').symlink_to(nas_concat)
            (fly_dir / 'mocorr_concat.tif').symlink_to(
                Path('suite2p') / 'mocorr_concat.tif'
            )

    return data, nas_used


def _gsheet_rows(flies) -> List[dict]:
    """Returns metadata Google Sheet rows for `flies` (as for these flies in Tom's).
    """
    # (side as their flipped.tif were made with)
    return [dict(date=E2E_DATE, fly=fly, driver='pebbled', indicator='6f',
        exclude=False, exclusion_reason=None, side='right', all_labelled=False
        ) for fly in flies
    ]


# GH146 (PN) consensus ROI responses of the 7 final megamat GH146 flies. The across-fly
# steps on ORN data read GH146 outputs (`GH146_<indicator>/ij_roi_stats.p`, written by an
# earlier `al-analysis -d GH146` run) from the current directory, to subset glomeruli
# (`get_gh146_glomeruli`), and fail if there is no `GH146_*` directory there (issue
# #11).
gh146_ij_roi_stats: Path = (al_data_root / 'sent_to_remy' / '2023-10-29' /
    'GH146_ij_certain-roi_stats.p'
)


def _run_al_analysis(root: Path, data: Path, args: List[str], flies, test_id: str, *,
    with_gh146_outputs: bool = False) -> Path:
    """Runs `al-analysis <args>` (via `al_cli_driver.py`) with cwd `<root>/out`.

    Args:
        with_gh146_outputs: if True, puts a copy of `gh146_ij_roi_stats` where a
            previous GH146 run would have written its `ij_roi_stats.p`

    Returns the output directory.
    """
    out = root / 'out'
    out.mkdir()
    if with_gh146_outputs:
        (out / 'GH146_6f').mkdir()
        shutil.copy2(gh146_ij_roi_stats, out / 'GH146_6f' / 'ij_roi_stats.p')

    gsheet_json = root / 'gsheet.json'
    gsheet_json.write_text(json.dumps(_gsheet_rows(flies)))

    env = dict(os.environ)
    env.update(HONG2P_DATA=str(data), HONG2P_FAST_DATA=str(data),
        XDG_DATA_HOME=str(root / 'xdg_data'), MPLBACKEND='agg',
        AL_ANALYSIS_DATA_AUDIT_TEST=test_id,
        AL_CLI_DRIVER_OLF_CONFIG_DIR=str(test_data_dir / 'tom_olfactometer_configs'),
    )
    for var in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT'):
        env.setdefault(var, '16')
    env.pop('PYTEST_CURRENT_TEST', None)

    log = root / 'al-analysis.log'
    with open(log, 'w') as f:
        proc = subprocess.run([sys.executable, '-u', str(test_dir / 'al_cli_driver.py'),
            str(gsheet_json)] + args, cwd=out, env=env, stdout=f,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=3600
        )

    if proc.returncode != 0:
        tail = '\n'.join(log.read_text().splitlines()[-40:])
        raise AssertionError(f'al-analysis {" ".join(args)} failed '
            f'(exit {proc.returncode}). end of log:\n{tail}'
        )

    return out


def _output_listing(out: Path, data: Path) -> Dict[str, List[str]]:
    """Returns relative paths of files (and links) written to outputs / intermediates.
    """
    def listing(root: Path) -> List[str]:
        return sorted(str(p.relative_to(root)) for p in root.rglob('*')
            if p.is_file() or p.is_symlink()
        )

    return dict(outputs=listing(out), analysis_intermediates=listing(
        data / 'analysis_intermediates'
    ))


def _read_output_csv(csv: Path) -> pd.DataFrame:
    """Reads an `al-analysis` output CSV for freezing.

    Per-panel consensus response CSVs (`<panel>_consensus.csv`) are parsed as numbers
    (so they can be compared within tolerance). Other CSVs (methods, glomerulus counts)
    are compared cell by cell as strings, except the 'unique_vals' row of the methods
    summaries, which holds reprs of pandas arrays (which differ across pandas versions).
    """
    if csv.stem.endswith('_consensus'):
        return pd.read_csv(csv, header=[0, 1], index_col=[0, 1])

    df = pd.read_csv(csv, header=None, dtype=str, keep_default_na=False)
    if csv.stem.startswith('methods_summary'):
        df = df[df.iloc[:, 0] != 'unique_vals']
    return df


def _drop_unstable_listing(paths: List[str]) -> List[str]:
    # logs are named / located by the run, and nothing else here should vary
    return [p for p in paths if not p.endswith('.log')]


@pytest.fixture(scope='module')
def e2e_one_recording(tmp_path_factory, request):
    """Runs `al-analysis` on one NAS recording (QUICK=1 version of the end-to-end test).

    Bounding frames come from the NAS caches (no ThorSync reading), and only the per
    recording steps run (no -A, so `al-analysis` exits after processing).

    Returns (root, data root, output dir, NAS snapshot before, NAS paths).
    """
    if not _nas_available():
        pytest.skip(f'NAS data not reachable ({nas_2p_data} not mounted?)')

    root = tmp_path_factory.mktemp('al_e2e_one')
    data, nas_used = _build_e2e_tree(root, {E2E_FLY: (E2E_QUICK_RECORDING,)},
        fly_concat=False
    )
    before = _snapshot(nas_used)
    out = _run_al_analysis(root, data, ['-t', E2E_DATE, '-e', E2E_DATE,
        f'{E2E_DATE}/{E2E_FLY}/{E2E_QUICK_RECORDING}'], flies=[E2E_FLY],
        test_id=f'{request.node.nodeid} (e2e_one_recording fixture)'
    )
    return root, data, out, before, nas_used


def _read_pickle(path: Path):
    return pd.read_pickle(path)


def _check_matches_nas_cache(rec_dir: Path, nas_rec_dir: Path) -> None:
    """Checks ImageJ ROI responses equal the NAS cache of the same recording, if that
    was computed with the same response calculation parameters.

    (an independent check of the ROI analysis, against outputs the code made when the
    data was analyzed; skipped if the NAS cache was made with other parameters)
    """
    nas_params = nas_rec_dir / 'response_calc_params.p'
    nas_cache = nas_rec_dir / 'ij_trial_df_cache.p'
    if not (nas_params.exists() and nas_cache.exists()):
        return

    if _read_pickle(nas_params) != _read_pickle(rec_dir / 'response_calc_params.p'):
        warnings.warn(f'{nas_rec_dir} responses computed with other parameters. not '
            'comparing'
        )
        return

    computed = _read_pickle(rec_dir / 'ij_trial_df_cache.p')
    cached = _read_pickle(nas_cache)
    assert list(computed.columns) == list(cached.columns)
    assert list(computed.index) == list(cached.index)
    np.testing.assert_allclose(computed.to_numpy(dtype=float),
        cached.to_numpy(dtype=float), rtol=1e-6, atol=1e-9, equal_nan=True,
        err_msg=f'responses differ from NAS cache {nas_cache}'
    )


@requires_nas
def test_al_analysis_one_recording(e2e_one_recording):
    """`al-analysis` on one recording: ImageJ ROI responses and the files written.

    Also checks the responses equal those the NAS cache has for this recording.
    """
    root, data, out, before, nas_used = e2e_one_recording

    assert _snapshot(nas_used) == before, 'NAS inputs changed!'

    rec_dir = (data / 'analysis_intermediates' / E2E_DATE / str(E2E_FLY) /
        E2E_QUICK_RECORDING
    )
    ij_trial_df = _read_pickle(rec_dir / 'ij_trial_df_cache.p')
    check_frozen(f'al/e2e/one_recording/{E2E_QUICK_RECORDING}_ij_trial_df',
        ij_trial_df
    )
    full_traces = _read_pickle(rec_dir / 'full_traces.p')
    check_frozen(f'al/e2e/one_recording/{E2E_QUICK_RECORDING}_full_traces',
        full_traces
    )
    _check_matches_nas_cache(rec_dir, nas_2p_data / 'analysis_intermediates' /
        E2E_DATE / str(E2E_FLY) / E2E_QUICK_RECORDING
    )

    listing = _output_listing(out, data)
    listing = {k: _drop_unstable_listing(v) for k, v in listing.items()}
    check_frozen('al/e2e/one_recording/written_files', listing)


@requires_nas
@pytest.mark.skipif(QUICK, reason='QUICK=1 (the 2-fly al-analysis run takes ~10 min)')
def test_al_analysis_two_flies(tmp_path, request):
    """`al-analysis -A` on two whole flies (3 diagnostic + 2 megamat recordings each).

    Covers frame <-> odor assignment from ThorSync data (for one recording), ImageJ ROI
    responses of each recording (ROIs linked to each fly's first recording), and the
    across-fly steps: methods CSVs, merging / consensus ROI tables, mean / SD outputs,
    odor metadata, and the files written.
    """
    fly2recordings = {fly: E2E_RECORDINGS for fly in E2E_FLIES}
    data, nas_used = _build_e2e_tree(tmp_path, fly2recordings, fly_concat=True,
        thorsync_recordings=(E2E_THORSYNC_RECORDING,)
    )
    before = _snapshot(nas_used)
    out = _run_al_analysis(tmp_path, data, ['-A', '-t', E2E_DATE, '-e', E2E_DATE] +
        [f'{E2E_DATE}/{fly}' for fly in E2E_FLIES], flies=E2E_FLIES,
        test_id=request.node.nodeid, with_gh146_outputs=True
    )
    assert _snapshot(nas_used) == before, 'NAS inputs changed!'

    import yaml
    for fly, recordings in fly2recordings.items():
        fly_dir = data / 'analysis_intermediates' / E2E_DATE / str(fly)
        nas_fly = nas_2p_data / 'analysis_intermediates' / E2E_DATE / str(fly)
        for rec in recordings:
            rec_dir = fly_dir / rec
            check_frozen(f'al/e2e/two_flies/{fly}_{rec}_ij_trial_df',
                _read_pickle(rec_dir / 'ij_trial_df_cache.p')
            )
            _check_matches_nas_cache(rec_dir, nas_fly / rec)

            if (fly, rec) == E2E_THORSYNC_RECORDING:
                # frame <-> odor assignment from ThorSync should match the NAS cache
                # from when the recording was first analyzed
                computed = yaml.safe_load(
                    (rec_dir / 'trial_bounding_frames.yaml').read_text()
                )
                cached = yaml.safe_load(
                    (nas_fly / rec / 'trial_bounding_frames.yaml').read_text()
                )
                assert computed == cached, \
                    f'{fly}/{rec}: bounding frames differ from NAS cache'

    output_root = out / 'pebbled_6f'
    for name in ('ij_roi_stats.p', 'ij_certain-roi_stats.p', 'roi_best_plane_depths.p'):
        check_frozen(f'al/e2e/two_flies/{Path(name).stem}', _read_pickle(
            output_root / name
        ))

    # (same data as the pickles above)
    skip_csvs = ('ij_roi_stats.csv', 'ij_certain-roi_stats.csv')
    csvs = [x for x in sorted(output_root.glob('*.csv')) if x.name not in skip_csvs]
    csvs.append(out / 'GH146_6f' / 'GH146_consensus_glomeruli.csv')
    for csv in csvs:
        check_frozen(f'al/e2e/two_flies/csv/{csv.stem}', _read_output_csv(csv))

    check_frozen('al/e2e/two_flies/response_calc_params',
        json.loads((output_root / 'response_calc_params.json').read_text())
    )

    listing = _output_listing(out, data)
    listing = {k: _drop_unstable_listing(v) for k, v in listing.items()}
    check_frozen('al/e2e/two_flies/written_files', listing)


###################################################################################
# plot-roi -d (analyzing a newly drawn ROI), on the end-to-end tree (NAS)
###################################################################################

@pytest.fixture
def e2e_hong2p_roots(e2e_one_recording, monkeypatch):
    """Points hong2p / al_analysis (in this process) at the one-recording e2e tree.
    """
    from hong2p import util
    import al_analysis.al_analysis as ala

    root, data, out, _, _ = e2e_one_recording
    monkeypatch.setattr(util, '_data_root', data)
    monkeypatch.setattr(util, '_fast_data_root', data)
    monkeypatch.setattr(ala, 'analysis_intermediates_root',
        data / 'analysis_intermediates'
    )
    return e2e_one_recording


# index of an ImageJ ROI in the recording's RoiSet.zip (0-indexed, as in ImageJ's ROI
# manager)
PLOT_ROI_INDEX: int = 0


@requires_nas
def test_plot_roi_extract_ij_responses(e2e_hong2p_roots):
    """`extract_ij_responses` (what `plot-roi -d <dir> -i <index>` computes): trial
    responses of one ImageJ ROI from the motion corrected movie.
    """
    import al_analysis.plot_roi_util as pru

    root, data, out, _, _ = e2e_hong2p_roots
    rec_dir = (data / 'analysis_intermediates' / E2E_DATE / str(E2E_FLY) /
        E2E_QUICK_RECORDING
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        df = pru.extract_ij_responses(rec_dir, PLOT_ROI_INDEX)

    assert df.shape[1] == 1
    check_frozen(f'al/plot_roi/extract_ij_responses_{E2E_QUICK_RECORDING}_roi'
        f'{PLOT_ROI_INDEX}', df
    )


@requires_nas
@pytest.mark.xfail(strict=True, raises=AttributeError, reason='load_and_plot calls '
    '`al.is_ijroi_named` on al_analysis.al_analysis, which no longer has that name '
    '(moved to hong2p.roi), so `plot-roi -d ...` always fails (issue #6)'
)
def test_plot_roi_analysis_dir(e2e_hong2p_roots, plot_roi_util):
    """`plot-roi -d <recording dir> -i <index> -n`: the frame it would plot.
    """
    root, data, out, _, _ = e2e_hong2p_roots
    rec_dir = (data / 'analysis_intermediates' / E2E_DATE / str(E2E_FLY) /
        E2E_QUICK_RECORDING
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        plot_roi_util.load_and_plot(_load_and_plot_args(analysis_dir=str(rec_dir),
            roi_index=PLOT_ROI_INDEX, no_compare=True
        ))

    calls = plot_roi_util._recorded_plot_calls
    assert len(calls) == 1
    check_frozen(f'al/plot_roi/analysis_dir_{E2E_QUICK_RECORDING}_roi{PLOT_ROI_INDEX}',
        calls[0][0]
    )


###################################################################################
# known breakage in the current venv (documented, so a fix is noticed)
###################################################################################

def _py314_default_start_method_not_fork() -> bool:
    return sys.platform.startswith('linux') and sys.version_info >= (3, 14)


_mp_logging_xfail = pytest.mark.xfail(_py314_default_start_method_not_fork(),
    strict=True, reason="init_logger installs multiprocessing_logging's handler, which "
    "asserts the 'fork' start method (Python >= 3.14 defaults to 'forkserver' on "
    'Linux), so the command fails before parsing its arguments (issue #10)'
)


@pytest.mark.parametrize('module, fn, cmd', [
    pytest.param('al_analysis.al_analysis', 'main', 'al-analysis',
        marks=_mp_logging_xfail
    ),
    pytest.param('al_analysis.plot_roi', 'main', 'plot-roi', marks=_mp_logging_xfail),
    ('al_analysis.load_antennal_csv', 'csvinfo_cli', 'csvinfo'),
], ids=['al-analysis', 'plot-roi', 'csvinfo'])
def test_cli_help(tmp_path, module, fn, cmd):
    """`<cmd> -h` runs (in a new Python, as the installed entry point would, without
    the workarounds of `al_cli_driver.py`).
    """
    env = dict(os.environ)
    env.update(HONG2P_DATA=str(tmp_path), HONG2P_FAST_DATA=str(tmp_path),
        XDG_DATA_HOME=str(tmp_path / 'xdg_data'), MPLBACKEND='agg'
    )
    proc = subprocess.run([sys.executable, '-c', 'import sys; '
        f"sys.argv = ['{cmd}', '-h']; "
        f'from {module} import {fn}; {fn}()'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600,
        stdin=subprocess.DEVNULL
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert 'usage:' in proc.stdout


@pytest.mark.xfail(strict=True, reason='with newer xarray, rois2best_planes_only\'s '
    '`rois.assign_coords(roi=rois.roi_name)` leaves the ROI MultiIndex in place (with a '
    'FutureWarning), so its output is not indexed by ROI name, and ij_traces fails an '
    'assertion (al_cli_driver.py works around this; hong2p issue #8)'
)
def test_rois2best_planes_only_roi_names():
    """`hong2p.roi.rois2best_planes_only` output 'roi' coordinate has the ROI names.
    """
    import xarray as xr
    from hong2p.roi import rois2best_planes_only

    index = pd.MultiIndex.from_arrays([[0, 1, 2], ['DM1', 'DM2', 'DM1'], [0, 0, 1],
        ['', '', ''], ['', '', '']], names=['roi_num', 'roi_name', 'roi_z',
        'ijroi_prefix', 'ijroi_suffix']
    )
    try:
        coords = xr.Coordinates.from_pandas_multiindex(index, 'roi')
    # (for old xarray, as with python 3.8, which has no Coordinates class)
    except AttributeError:
        coords = {'roi': index}

    rois = xr.DataArray(np.ones((3, 2, 2), dtype=bool), dims=['roi', 'y', 'x'],
        coords=coords
    )
    roi_quality = pd.Series([0.1, 0.5, 0.9])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        roi_indices, best = rois2best_planes_only(rois, roi_quality)

    assert sorted(roi_indices) == [1, 2]
    assert sorted(best.roi.values.tolist()) == ['DM1', 'DM2']
