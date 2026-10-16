All of my repositories are organized under ~/src.
This project depends heavily on hong2p and olfsysm, so changes may also be necessary in
those repositories. Other Python packages that are installed in editable mode, if they
were installed from a directory under ~/src, should also be considered and possibly
changed.

Write commit messages in a broadly similar style to existing, with at least one line per
file changed (and more if necessary, if there were e.g. many changes to one function, or
several similar changes). If a set of small changes across many files makes more sense
to describe in terms of the high-level impact of that change, rather than to enumerate
file-by-file, prefer a style like that.

Keep Python docstrings in a similar style, which is similar to the Google docstring
style. Do not add as many comments to the code otherwise: only where potentially
confusing.

When running pytest, run with the environment variable QUICK=1 set, unless extensively
testing changes specifically to the model code in mb_model (or the associated code in
olfsysm), to skip most of the different model cases.

Tests for the C++ code in olfsysm are currently written in Python in this al_analysis
repo, under the test/test_mb_model.py file. Any changes to the model should preserve
existing behavior, and write additional tests to verify this unless there seems to be
some bug or misunderstanding in the current implementation. Bring my attention to any
such cases when evaluating current tests or implementing new ones.
