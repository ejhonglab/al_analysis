This project depends heavily on hong2p and olfsysm, so changes may also be necessary in
those repositories. Other Python packages that are installed in editable mode, if they
were installed from a directory under ~/src, should also be considered and possibly
changed.

When running pytest, run with the environment variable QUICK=1 set, unless extensively
testing changes specifically to the model code in mb_model (or the associated code in
olfsysm), to skip most of the different model cases.

Tests for the C++ code in olfsysm are currently written in Python in this al_analysis
repo, under the test/test_mb_model.py file. Any changes to the model should preserve
existing behavior, and write additional tests to verify this unless there seems to be
some bug or misunderstanding in the current implementation. Bring my attention to any
such cases when evaluating current tests or implementing new ones.

For now, try to preserve support for Python 3.8 (with last pandas 1.5), as well as the
current ./venv/ python (should be 3.14 with pandas 3). Do this by either
asking-for-forgiveness-not-permission (e.g. try/except handling), or explicit checking
of versions if necessary. If code only exists to support old pandas 1.5 (or other old py
thon 3.8 stuff), add a comment saying which old thing that part of the code is for.
