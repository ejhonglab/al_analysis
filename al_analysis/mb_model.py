"""The MB model, now `pyolfsysm.mb_model` (in the olfsysm repository, since 2026-11).

This module forwards every attribute read, assignment and deletion there, so existing code
(`from al_analysis.mb_model import fit_mb_model`, `mb_model.STRICT_CLEAN_OLFSYSM = True`,
monkeypatching in tests) keeps working. New code should import `pyolfsysm.mb_model`.
Requires pyolfsysm, installed with al_analysis's `model` extra (`al_analysis[model]`).
"""
from al_analysis._pyolfsysm_shim import forward_module
forward_module(__name__, 'pyolfsysm.mb_model')
