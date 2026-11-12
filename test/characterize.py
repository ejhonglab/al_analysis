"""Frozen-output helpers for the characterization tests (`test_characterize_*.py`).

The characterization tests freeze outputs of the current code (tables, tuned
parameters, printed summaries) under `test_data/characterization/`, so that later
refactors (moving the model code to olfsysm, the output helpers to hong2p) can be checked
against them. This module has no dependency on al_analysis, so a copy of it can move with
the tests that use it.

To (re)generate the frozen outputs, run the tests with `CHARACTERIZE_REGEN=1`, e.g.:
```
CHARACTERIZE_REGEN=1 QUICK=0 ./venv/bin/pytest test/test_characterize_model.py
```
Regeneration only rewrites files whose contents would change (by the same comparison the
tests make), so unchanged outputs are not rewritten.

Frozen tables are saved as `.npz` (one array per column, labels as strings), rather than
parquet / pickle, so that they read the same under Python 3.8 / pandas 1.5 and newer
versions. Other values are saved as JSON (`.json`) or text (`.txt`).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytest

from .conftest import test_data_dir


frozen_root: Path = test_data_dir / 'characterization'

REGEN: bool = os.environ.get('CHARACTERIZE_REGEN', '0') not in ('', '0')

# default tolerances for float comparisons. exact (0) would be fine on the machine and
# build the outputs were generated with, but compilers / BLAS / CPUs can differ in the
# last bits.
DEFAULT_RTOL: float = 1e-6
DEFAULT_ATOL: float = 1e-9


def _label_strs(index: pd.Index) -> Dict[str, np.ndarray]:
    """Returns dict of '<i>' -> str array of level i values (and 'names')."""
    if isinstance(index, pd.MultiIndex):
        levels = [index.get_level_values(i) for i in range(index.nlevels)]
    else:
        levels = [index]

    out = {str(i): np.array([str(x) for x in level], dtype=str)
        for i, level in enumerate(levels)
    }
    out['names'] = np.array([str(x) for x in index.names], dtype=str)
    return out


def _column_array(ser: pd.Series) -> np.ndarray:
    """Returns a numeric / bool array for numeric columns, a str array otherwise."""
    if pd.api.types.is_bool_dtype(ser.dtype):
        return ser.to_numpy(dtype=bool)

    if pd.api.types.is_integer_dtype(ser.dtype) and not ser.isna().any():
        return ser.to_numpy(dtype=np.int64)

    if pd.api.types.is_numeric_dtype(ser.dtype):
        return ser.to_numpy(dtype=np.float64, na_value=np.nan)

    return np.array([str(x) for x in ser], dtype=str)


def _frame_to_arrays(obj: Union[pd.DataFrame, pd.Series]) -> Dict[str, np.ndarray]:
    if isinstance(obj, pd.Series):
        kind = 'series'
        df = obj.to_frame(name=obj.name if obj.name is not None else 0)
    else:
        kind = 'frame'
        df = obj

    arrays = {'kind': np.array(kind)}
    for k, v in _label_strs(df.index).items():
        arrays[f'index_{k}'] = v

    for k, v in _label_strs(df.columns).items():
        arrays[f'columns_{k}'] = v

    for j in range(df.shape[1]):
        arrays[f'col_{j}'] = _column_array(df.iloc[:, j])

    return arrays


def _assert_arrays_close(actual: np.ndarray, expected: np.ndarray, what: str, *,
    rtol: float, atol: float) -> None:

    assert actual.shape == expected.shape, \
        f'{what}: shape {actual.shape} != frozen {expected.shape}'

    if expected.dtype.kind in 'US' or actual.dtype.kind in 'US':
        mismatch = actual.astype(str) != expected.astype(str)
        if mismatch.any():
            idx = np.argwhere(mismatch)[:5].tolist()
            raise AssertionError(f'{what}: {mismatch.sum()} values differ from frozen '
                f'(first at {idx}: {actual[mismatch][:5]} != {expected[mismatch][:5]})'
            )
        return

    if expected.dtype.kind in 'biu' and actual.dtype.kind in 'biu':
        if not np.array_equal(actual, expected):
            mismatch = actual != expected
            raise AssertionError(f'{what}: {mismatch.sum()} of {mismatch.size} values '
                f'differ from frozen (actual {actual[mismatch][:5]}, frozen '
                f'{expected[mismatch][:5]})'
            )
        return

    np.testing.assert_allclose(actual.astype(np.float64), expected.astype(np.float64),
        rtol=rtol, atol=atol, equal_nan=True, err_msg=what
    )


def _compare_arrays(actual: Dict[str, np.ndarray], expected: Dict[str, np.ndarray],
    name: str, *, rtol: float, atol: float) -> None:

    assert set(actual.keys()) == set(expected.keys()), (f'{name}: arrays '
        f'{sorted(set(actual) ^ set(expected))} only in one of output / frozen'
    )
    for k in sorted(expected.keys()):
        _assert_arrays_close(actual[k], expected[k], f'{name}[{k}]', rtol=rtol,
            atol=atol
        )


def _compare_json(actual: Any, expected: Any, where: str, *, rtol: float,
    atol: float) -> None:

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f'{where}: {type(actual)} != dict'
        assert set(actual) == set(expected), (f'{where}: keys only in one of output / '
            f'frozen: {sorted(set(actual) ^ set(expected))}'
        )
        for k in expected:
            _compare_json(actual[k], expected[k], f'{where}.{k}', rtol=rtol, atol=atol)

    elif isinstance(expected, list):
        assert isinstance(actual, list), f'{where}: {type(actual)} != list'
        assert len(actual) == len(expected), \
            f'{where}: length {len(actual)} != frozen {len(expected)}'

        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare_json(a, e, f'{where}[{i}]', rtol=rtol, atol=atol)

    elif isinstance(expected, float) or (isinstance(actual, float) and
            isinstance(expected, int) and not isinstance(expected, bool)):

        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), \
            f'{where}: {actual!r} is not a number (frozen {expected!r})'

        if np.isnan(expected):
            assert np.isnan(actual), f'{where}: {actual!r} != frozen nan'
        else:
            assert abs(actual - expected) <= atol + rtol * abs(expected), \
                f'{where}: {actual!r} != frozen {expected!r}'
    else:
        assert actual == expected, f'{where}: {actual!r} != frozen {expected!r}'


def to_jsonable(obj: Any) -> Any:
    """Converts numpy / pandas scalars and containers to JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())

    if isinstance(obj, (pd.Series, pd.Index)):
        return to_jsonable(obj.tolist())

    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)

    if isinstance(obj, (int, np.integer)):
        return int(obj)

    if isinstance(obj, (float, np.floating)):
        return float(obj)

    if obj is None or isinstance(obj, str):
        return obj

    return str(obj)


def _json_text(obj: Any) -> str:
    # allow_nan so NaN can be frozen (as JSON extension 'NaN', which json reads back)
    return json.dumps(to_jsonable(obj), indent=1, sort_keys=True, allow_nan=True) + '\n'


def frozen_path(name: str, suffix: str) -> Path:
    return frozen_root / f'{name}{suffix}'


def check_frozen(name: str, obj: Any, *, rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL) -> None:
    """Asserts `obj` matches the frozen copy saved under `name`, or saves it if REGEN.

    Args:
        name: path of the frozen file relative to `frozen_root`, without suffix (e.g.
            'model/uniform/spike_counts'). The suffix is chosen from the type of `obj`.

        obj: a DataFrame or Series (saved as .npz; labels compared as strings, numeric
            columns within tolerance, int / bool columns exactly), an ndarray (.npz), a
            str (.txt, compared exactly), or anything else JSON-serializable after
            `to_jsonable` (.json, floats compared within tolerance).

        rtol, atol: tolerances for float comparisons.
    """
    if isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray)):
        if isinstance(obj, np.ndarray):
            arrays = {'kind': np.array('ndarray'), 'values': obj}
        else:
            arrays = _frame_to_arrays(obj)

        path = frozen_path(name, '.npz')

        def load():
            with np.load(path, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}

        def compare():
            _compare_arrays(arrays, load(), name, rtol=rtol, atol=atol)

        def write():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)

    elif isinstance(obj, str):
        path = frozen_path(name, '.txt')

        def compare():
            expected = path.read_text()
            if obj != expected:
                import difflib
                diff = ''.join(list(difflib.unified_diff(expected.splitlines(True),
                    obj.splitlines(True), 'frozen', 'output'
                ))[:60])
                raise AssertionError(f'{name}: text differs from frozen:\n{diff}')

        def write():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(obj)
    else:
        path = frozen_path(name, '.json')
        text = _json_text(obj)

        def compare():
            _compare_json(json.loads(text), json.loads(path.read_text()), name,
                rtol=rtol, atol=atol
            )

        def write():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    if not path.exists():
        if REGEN:
            write()
            return

        pytest.fail(f'frozen output {path} missing. run with CHARACTERIZE_REGEN=1 to '
            'create it (from the current code)'
        )

    if REGEN:
        try:
            compare()
        except AssertionError:
            write()
        return

    compare()


def frame_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Returns a compact JSON-able summary of a numeric frame (shape, sums, counts).

    For frozen copies of large frames where the full values would be too big to commit.
    """
    values = df.to_numpy(dtype=np.float64)
    finite = np.where(np.isfinite(values), values, 0.0)
    return dict(
        shape=list(df.shape),
        sum=float(finite.sum()),
        abs_sum=float(np.abs(finite).sum()),
        sq_sum=float((finite ** 2).sum()),
        n_nonzero=int((finite != 0).sum()),
        n_nan=int(np.isnan(values).sum()),
        col_sums=[float(x) for x in finite.sum(axis=0)],
    )
