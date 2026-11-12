
import atexit
import csv
import functools
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Tuple

import pytest


# so can work w/ pytest called from repo root, but also w/ scripts like
# generate_reference_outputs_for_repro.py, which I've been calling from this directory.
test_dir: Path = Path(__file__).resolve().parent

# TODO define in conftest or something?
# TODO refactor this handling of test data path? also used in test_al_analysis.py
test_data_dir: Path = test_dir / 'test_data'

repo_root: Path = test_dir.parent


def _use_tmp_hong2p_data_if_missing() -> None:
    """Points HONG2P_DATA / HONG2P_FAST_DATA at an empty temp dir, if set to a missing dir.

    Importing `al_analysis.al_analysis` creates directories under these (and hong2p
    warns if they are set to missing directories), so test collection failed on hosts
    where they point to an unmounted drive. Tests that need real data under these roots
    set them up themselves.
    """
    tmp_root = None
    for var in ('HONG2P_DATA', 'HONG2P_FAST_DATA'):
        value = os.environ.get(var)
        if value is None or Path(value).is_dir():
            continue

        if tmp_root is None:
            tmp_root = tempfile.mkdtemp(prefix='al_analysis_test_hong2p_data_')
            atexit.register(shutil.rmtree, tmp_root, True)

        print(f'conftest: {var}={value} is not a directory. using {tmp_root} for tests')
        os.environ[var] = tmp_root


# before any test module imports hong2p (which reads these at import)
_use_tmp_hong2p_data_if_missing()


# Opt-in data audit: AL_ANALYSIS_DATA_AUDIT=<csv path> records every file opened (and
# every directory listed) under al_analysis/data and test/test_data, per test, to that
# CSV (appending, so several pytest runs can share one CSV). Summarize it with
# `test/data_audit_summary.py`. Only this process (and processes forked from it) is
# audited, not subprocesses that start a new Python.
DATA_AUDIT_ENV_VAR = 'AL_ANALYSIS_DATA_AUDIT'

# modules in the al_analysis package, by which side of the planned split they are on.
# al_util is shared, so it does not decide the side on its own.
_model_modules = ('mb_model', 'wpnkc_shuffles')
_al_modules = ('al_analysis', 'plot_roi', 'plot_roi_util', 'load_antennal_csv')

_audit_state = dict(current='<session>', writer=None, file=None, seen=None, roots=None,
    in_hook=False
)


def _audit_roots() -> Tuple[Tuple[str, str], ...]:
    roots = [str(repo_root / 'al_analysis' / 'data'), str(test_data_dir)]
    labels = ['data', 'test_data']
    # the editable install should point at the same tree, but in case the package is
    # installed elsewhere
    try:
        import importlib.util
        spec = importlib.util.find_spec('al_analysis')
        if spec is not None and spec.submodule_search_locations is not None:
            for loc in spec.submodule_search_locations:
                data = os.path.join(os.path.realpath(str(loc)), 'data')
                if os.path.isdir(data) and data not in roots:
                    roots.append(data)
                    labels.append('data')
    except Exception:
        pass

    return tuple((r.rstrip(os.sep) + os.sep, label) for r, label in zip(roots, labels))


def _caller_chain() -> Tuple[str, str]:
    """Returns (al_analysis package modules on the stack, innermost repo frame).
    """
    chain = []
    innermost = ''
    frame = sys._getframe(1)
    repo = str(repo_root) + os.sep
    conftest = os.path.join('test', 'conftest.py')
    venv = 'venv' + os.sep
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename.startswith(repo):
            rel = filename[len(repo):]
            if rel != conftest and not rel.startswith(venv):
                if not innermost:
                    innermost = f'{rel}:{frame.f_code.co_name}'

                if rel.startswith('al_analysis' + os.sep):
                    module = os.path.splitext(os.path.basename(rel))[0]
                    if not chain or chain[-1] != module:
                        chain.append(module)

        frame = frame.f_back

    # outermost first
    return '>'.join(reversed(chain)), innermost


def _side(chain: str) -> str:
    modules = chain.split('>') if chain else []
    if any(m in _model_modules for m in modules):
        return 'model'

    if any(m in _al_modules for m in modules):
        return 'al'

    if 'al_util' in modules:
        return 'al_util'

    return 'test'


def _match_root(path: str):
    for root, label in _audit_state['roots']:
        if path.startswith(root) or path + os.sep == root:
            return root, label
    return None


def _record(event: str, path) -> None:
    state = _audit_state
    if state['in_hook'] or state['writer'] is None:
        return

    if isinstance(path, int):
        return

    state['in_hook'] = True
    try:
        try:
            path = os.fsdecode(os.fspath(path))
        except TypeError:
            return

        abspath = os.path.abspath(path)
        match = _match_root(abspath)
        # only paying for realpath on paths that might be links into the data dirs
        if match is None and ('data' in abspath):
            abspath = os.path.realpath(abspath)
            match = _match_root(abspath)

        if match is None:
            return

        root, label = match
        rel = abspath[len(root):] if abspath.startswith(root) else ''
        folder = rel.split(os.sep)[0] if rel else '.'

        chain, innermost = _caller_chain()
        key = (state['current'], event, rel, label, chain)
        if key in state['seen']:
            return

        state['seen'].add(key)
        state['writer'].writerow([state['current'], event, label, folder, rel,
            _side(chain), chain, innermost, os.getpid()
        ])
        state['file'].flush()

    finally:
        state['in_hook'] = False


def _audit_hook(event: str, args) -> None:
    if _audit_state['writer'] is None:
        return

    if event == 'open':
        _record('open', args[0])

    elif event in ('os.listdir', 'os.scandir'):
        _record('list', args[0] if len(args) > 0 and args[0] is not None else '.')

    elif event == 'glob.glob':
        _record('glob', args[0])


def _wrap_reader(module, attr: str, event: str) -> None:
    """Wraps `module.<attr>` to record its path argument.

    For readers that open files in C (not seen by the 'open' audit event).
    """
    fn = getattr(module, attr, None)
    if fn is None or getattr(fn, '_data_audit_wrapped', False):
        return

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if len(args) > 0:
            _record(event, args[0])
        else:
            for k in ('source', 'filename_or_obj', 'path', 'path_or_paths'):
                if k in kwargs:
                    _record(event, kwargs[k])
                    break

        return fn(*args, **kwargs)

    wrapped._data_audit_wrapped = True
    setattr(module, attr, wrapped)


def _wrap_c_readers() -> None:
    # pyarrow (parquet) and netCDF4 open files without Python's open
    try:
        import pyarrow.parquet as pq
        _wrap_reader(pq, 'read_table', 'open-parquet')
        _wrap_reader(pq, 'ParquetFile', 'open-parquet')
    except ImportError:
        pass

    try:
        import xarray as xr
        import xarray.backends.api as xr_api
        for mod in (xr, xr_api):
            for attr in ('open_dataset', 'open_dataarray', 'load_dataset',
                    'load_dataarray'):
                _wrap_reader(mod, attr, 'open-netcdf')
    except ImportError:
        pass


def pytest_configure(config) -> None:
    audit_csv = os.environ.get(DATA_AUDIT_ENV_VAR)
    if not audit_csv:
        return

    audit_csv = Path(audit_csv).resolve()
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not audit_csv.exists() or audit_csv.stat().st_size == 0
    f = open(audit_csv, 'a', newline='')
    writer = csv.writer(f)
    if new:
        writer.writerow(['test', 'event', 'root', 'folder', 'path', 'side', 'chain',
            'caller', 'pid'
        ])
        f.flush()

    _audit_state.update(writer=writer, file=f, seen=set(), roots=_audit_roots())
    _wrap_c_readers()
    # audit hooks can not be removed. this one does nothing once pytest_unconfigure
    # clears the writer.
    sys.addaudithook(_audit_hook)


def pytest_unconfigure(config) -> None:
    f = _audit_state.get('file')
    if f is not None:
        _audit_state.update(writer=None, file=None)
        f.close()


def pytest_collectstart(collector) -> None:
    if _audit_state['writer'] is not None:
        _audit_state['current'] = f'<collect> {collector.nodeid}'


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if _audit_state['writer'] is not None:
        _audit_state['current'] = item.nodeid

    yield

    if _audit_state['writer'] is not None:
        _audit_state['current'] = '<between tests>'
