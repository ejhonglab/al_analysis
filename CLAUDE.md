This project depends heavily on hong2p and olfsysm, so changes may also be necessary in
those repositories. The MB model (formerly mb_model.py, wpnkc_shuffles.py and the model-only
part of al_util.py here, with the data it reads, its scripts and its tests) moved to the
olfsysm repository as the package pyolfsysm (2026-11); al_analysis installs it with its
optional `model` extra, and al_analysis.mb_model / wpnkc_shuffles (and al_util's model-only
names) only forward there. Model changes go in olfsysm (~/src/olfsysm).

Other Python packages that are installed in editable mode, if they were installed from a
directory under ~/src, should also be considered and possibly changed.

When running pytest, run with the environment variable QUICK=1 set.

Tests for the C++ code in olfsysm, and for the model's Python code (pyolfsysm), are in the
olfsysm repository (its test/test_mb_model.py and the other model tests; they were here,
under test/, until 2026-11); its CLAUDE.md says how to run them. The tests here cover the
AL analysis, the output helpers, and the forwarding of the old model module names
(test/test_import_surface.py).

For now, try to preserve support for Python 3.8 (with last pandas 1.5), as well as the
current ./venv/ python (should be 3.14 with pandas 3). Do this by either
asking-for-forgiveness-not-permission (e.g. try/except handling), or explicit checking
of versions if necessary. If code only exists to support old pandas 1.5 (or other old py
thon 3.8 stuff), add a comment saying which old thing that part of the code is for.
