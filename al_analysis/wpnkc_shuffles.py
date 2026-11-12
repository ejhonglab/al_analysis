"""Shuffled PN->KC wiring, now `pyolfsysm.wpnkc_shuffles` (olfsysm repository, 2026-11).

This module forwards every attribute read, assignment and deletion there (see
`al_analysis.mb_model`). New code should import `pyolfsysm.wpnkc_shuffles`. Requires
pyolfsysm, installed with al_analysis's `model` extra (`al_analysis[model]`).
"""
from al_analysis._pyolfsysm_shim import forward_module
forward_module(__name__, 'pyolfsysm.wpnkc_shuffles')
