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

BUGS.md lists the known bugs and questionable choices in the ROI response calculation
(`ij_traces`, `compute_trial_stats`, `delta_f_over_f`, `sign_preserving_maxabs`, and
hong2p's `ijroi_masks` / `ijroi2mask` / `rois2best_planes_only`), each with where it is, how
much it was measured to matter, and a possible fix. None is fixed yet. In short: an ROI's
best plane is chosen separately in each recording, by one signed maximum, from the
responses then reported; masks are half a pixel off ImageJ's and depend on the direction an
outline was drawn in; F0 is the mean of 2 to 5 time points (deliberate: every trial gets
the 7 s before its odor that the first one has; a small cost in noise, not a bug); the
±peak statistic is biased away from zero; the two-volume window misses some peaks, and at
0.67 volumes/s can leave out a volume the 3 s pulse reaches into (a longer pulse, more:
the window that holds a pulse is floor(pulse / volume + 1/2) + 1 volumes, which hong-data
takes by default); no background is subtracted; odor
abbreviations depend on hong2p's cache. It also records that the frames assigned to each
odor were checked against ThorSync across years, rigs and ThorImage versions, and were
consistent. Remind me of the items that bear on what I ask for
whenever a request touches that code or interprets its outputs (response sizes, inhibition,
comparisons across a fly's recordings), before doing the work. When one is fixed, update
BUGS.md, and say that hong-data's `app/extract.py` (which reproduces the current numbers
exactly with its response window set to 2 volumes, and is tested to) must change with it.
