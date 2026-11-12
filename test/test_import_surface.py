"""Tests every name other code imports from al_analysis's main modules is importable.

The names are frozen in `test_data/import_surface.json` by `import_surface.py` (run it to
regenerate), from al_analysis's scripts / tests / package and the experiment queue
repository (frozen before the split). Since the model code moved to olfsysm (pyolfsysm)
and the shared output helpers to hong2p, `al_analysis.mb_model` and `al_analysis.al_util`
still provide all of these, forwarded there (the model's names only with the `model`
extra; without it, those are skipped).
"""

import importlib
import json
import sys
from typing import Dict, List

import pytest

from .import_surface import frozen_path, MODULES, scan, default_queue_root


# names some code imports, but that are already missing from the current code (bugs in
# that code, not in the module). listed so the tests still catch any new missing name.
KNOWN_MISSING: Dict[str, Dict[str, str]] = {
    'al_analysis.al_analysis': {
        # moved to hong2p.roi in b199229 (2023), but plot_roi_util.load_and_plot (its
        # -d/--analysis-dir path) still calls `al.is_ijroi_named` (issue #6)
        'is_ijroi_named': 'plot_roi_util.py still uses al_analysis.is_ijroi_named',
        # moved to mb_model. only used by scripts/compare_matt_and_new_hemibrain_wPNKC.py
        'connectome_wPNKC': 'stale import in compare_matt_and_new_hemibrain_wPNKC.py',
    },
}


# modules that forward to pyolfsysm (the model), and pyolfsysm's module behind each
MODEL_MODULES: Dict[str, str] = {
    'al_analysis.mb_model': 'pyolfsysm.mb_model',
    'al_analysis.wpnkc_shuffles': 'pyolfsysm.wpnkc_shuffles',
}


def _have_pyolfsysm() -> bool:
    try:
        importlib.import_module('pyolfsysm.mb_model')
        return True
    except ImportError:
        return False


def _frozen() -> Dict:
    return json.loads(frozen_path.read_text())


def _missing(module_name: str, names: List[str]) -> List[str]:
    if module_name in MODEL_MODULES and not _have_pyolfsysm():
        pytest.skip(f'{module_name} needs pyolfsysm (the `model` extra)')

    module = importlib.import_module(module_name)
    if module_name == 'al_analysis.al_util' and not _have_pyolfsysm():
        # forwarded to pyolfsysm.util, only available with the `model` extra
        names = [n for n in names if n not in module._forwarded_to_pyolfsysm]

    known = KNOWN_MISSING.get(module_name, dict())
    missing = [n for n in names if n != '*' and not hasattr(module, n)]
    fixed = [n for n in known if hasattr(module, n)]
    assert not fixed, f'remove {fixed} from KNOWN_MISSING (now in {module_name})'
    return [n for n in missing if n not in known]


@pytest.mark.parametrize('module_name', MODULES)
def test_frozen_names_importable(module_name):
    """Tests names in the frozen import surface are attributes of their module.
    """
    names = sorted(_frozen()['modules'][module_name])
    assert len(names) > 0
    missing = _missing(module_name, names)
    assert not missing, (f'{module_name} lacks {len(missing)} names other code imports '
        f'(see {frozen_path.name}): {missing}'
    )


@pytest.mark.parametrize('module_name', MODULES)
def test_assigned_names_exist(module_name):
    """Tests module attributes that other code assigns exist in that module.

    Assigning these (e.g. `al_util.verbose = True`, `monkeypatch.setattr(mb_model,
    'osm', ...)`) only has an effect if the code reading them reads them from the same
    module, so any re-export module (after the split) has to forward these explicitly.
    """
    names = sorted(_frozen()['assigned'][module_name])
    missing = _missing(module_name, names)
    assert not missing, f'{module_name} lacks assigned names {missing}'


def test_current_names_importable():
    """Tests names found by scanning the current sources (not just frozen ones) exist.

    Also prints names added / no longer used since the frozen list was generated (only
    informative; regenerate the frozen list with `import_surface.py`).
    """
    current = scan()
    frozen = _frozen()
    for module_name in MODULES:
        if module_name in MODEL_MODULES and not _have_pyolfsysm():
            print(f'{module_name} not checked: needs pyolfsysm (the `model` extra)')
            continue

        names = sorted(current['modules'][module_name])
        missing = _missing(module_name, names)
        assert not missing, (f'{module_name} lacks names current sources import: '
            f'{missing}'
        )

        added = sorted(set(names) - set(frozen['modules'][module_name]))
        removed = sorted(set(frozen['modules'][module_name]) - set(names))
        if added or removed:
            print(f'{module_name}: names not in frozen list: {added}; frozen names no '
                f'longer imported: {removed}'
            )

    if not default_queue_root.is_dir():
        print(f'{default_queue_root} not found, so its imports were only checked via '
            'the frozen list'
        )


@pytest.mark.parametrize('module_name', sorted(MODEL_MODULES))
def test_model_module_forwards_to_pyolfsysm(module_name, monkeypatch):
    """Tests reads, assignments, deletions and `import *` reach the pyolfsysm module.
    """
    target = pytest.importorskip(MODEL_MODULES[module_name])
    module = importlib.import_module(module_name)

    public = [n for n in vars(target) if not n.startswith('_')]
    assert len(public) > 0
    for name in public:
        assert getattr(module, name) is getattr(target, name), name
    assert set(dir(target)) <= set(dir(module))
    # the forwarding module's own namespace holds no names that would shadow the target's
    assert not [n for n in vars(module) if not n.startswith('_')]

    # e.g. `mb_model.STRICT_CLEAN_OLFSYSM = True`, or monkeypatching in tests
    value = object()
    monkeypatch.setattr(module, '_forwarding_test_attr', value, raising=False)
    assert getattr(target, '_forwarding_test_attr') is value
    assert getattr(module, '_forwarding_test_attr') is value
    monkeypatch.undo()
    assert not hasattr(target, '_forwarding_test_attr')

    namespace = dict()
    exec(f'from {module_name} import *', namespace)
    for name in public:
        assert namespace[name] is getattr(target, name), name


@pytest.mark.parametrize('module_name', sorted(MODEL_MODULES))
def test_model_module_without_pyolfsysm(module_name, monkeypatch):
    """Tests importing a forwarding module without pyolfsysm says how to install it.
    """
    # as if pyolfsysm were not installed
    monkeypatch.setitem(sys.modules, MODEL_MODULES[module_name], None)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    with pytest.raises(ModuleNotFoundError, match=r'\[model\]'):
        importlib.import_module(module_name)
