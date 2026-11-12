"""Forwarding of al_analysis's former model modules to pyolfsysm.

The model (`mb_model`, `wpnkc_shuffles` and the model-only part of `al_util`) moved to the
olfsysm repository, as the package pyolfsysm (2026-11). `al_analysis.mb_model` and
`al_analysis.wpnkc_shuffles` remain as modules that forward every attribute read,
assignment and deletion to their pyolfsysm counterparts (`forward_module`), so code that
imports from them, assigns their settings (`mb_model.STRICT_CLEAN_OLFSYSM = True`) or
monkeypatches them keeps working. pyolfsysm is installed with the `model` extra
(`al_analysis[model]`).
"""
import importlib
import sys
from types import ModuleType
from typing import Any, List


MODEL_EXTRA_MESSAGE: str = ("the model is in pyolfsysm (part of olfsysm), installed with "
    "al_analysis's `model` extra: `pip install -e '.[model]'` from an al_analysis checkout "
    "(see its README), or `pip install -e <olfsysm checkout>`"
)


def import_pyolfsysm(name: str) -> ModuleType:
    """Imports module `name` of pyolfsysm, with installation instructions if that fails.
    """
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise ModuleNotFoundError(f'could not import {name} ({e}): {MODEL_EXTRA_MESSAGE}',
            name=getattr(e, 'name', None)
        ) from e


_TARGET_ATTR = '_pyolfsysm_target_module'


def _is_dunder(name: str) -> bool:
    return name.startswith('__') and name.endswith('__')


class ForwardingModule(ModuleType):
    """Module type forwarding (non-dunder) attribute access to another module.

    The forwarding module's own namespace holds only its dunder attributes (its name, file,
    docstring, `__all__`, ...) and the target module, so every other name is read from,
    assigned on and deleted from the target, and there is only ever one value.
    """
    def __getattr__(self, name: str) -> Any:
        # only called for names not in this module's own namespace
        target = self.__dict__.get(_TARGET_ATTR)
        if target is None or _is_dunder(name):
            raise AttributeError(f'module {self.__name__!r} has no attribute {name!r}')
        return getattr(target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if _is_dunder(name) or name == _TARGET_ATTR:
            super().__setattr__(name, value)
        else:
            setattr(self.__dict__[_TARGET_ATTR], name, value)

    def __delattr__(self, name: str) -> None:
        if _is_dunder(name) or name == _TARGET_ATTR:
            super().__delattr__(name)
        else:
            delattr(self.__dict__[_TARGET_ATTR], name)

    def __dir__(self) -> List[str]:
        return sorted(set(dir(self.__dict__[_TARGET_ATTR])) | set(super().__dir__()))


def forward_module(module_name: str, target_name: str) -> None:
    """Makes module `module_name` (being imported) forward to module `target_name`.

    Raises ModuleNotFoundError (with how to install pyolfsysm) if the target can not be
    imported. Call at the end of the forwarding module, which should define nothing else
    (any other name it defined would shadow the target's).
    """
    target = import_pyolfsysm(target_name)
    module = sys.modules[module_name]
    module.__dict__[_TARGET_ATTR] = target
    # for `from <module_name> import *` (which reads only `__all__`, or the module's own
    # namespace)
    module.__dict__['__all__'] = getattr(target, '__all__',
        [n for n in vars(target) if not n.startswith('_')]
    )
    # names the forwarding module itself bound (e.g. `forward_module`, imported to call this)
    for name in [n for n in module.__dict__ if not _is_dunder(n) and n != _TARGET_ATTR]:
        del module.__dict__[name]
    module.__class__ = ForwardingModule
