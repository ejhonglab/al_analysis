We are following up on a report you made, in the markdown:
"APL and the ORN→KC remapping of odor–odor correlations.md"

The report may reference the analysis and outputs in apl_pair_analysis/, which you used
in construction of the report (and provided alongside for download). The two 16x16 KC
correlation-distance matrices the report analyzed (TNT3c_dist_apl16.csv [control] and
TNT_dist_apl16.csv [APL output blocked], both from Yang) are not currently in
apl_pair_analysis/. Until they are added, reconstruct them from the `ctrl` and `tnt`
columns of apl_pair_analysis/all120_pairs.csv.

An earlier, related Claude analysis (done with Yang, Oct 2026) is on tensor under
Tom/claude_APL_modeling_YZ4TC/Claude_APL_analysis/. It has APL synapse tables from four
connectomes, a connectome-APL model of the TNT vs TNT3c change, and checks on Yang's
per-recording TNT / TNT3c KC data (8 TNT recordings from 7 flies, 11 TNT3c recordings
from 8 flies). Read apl_synapses_markdown_combined_2026-10-23.md there before
duplicating any of that work. The per-recording data itself is not in that folder.

A local copy of my thesis is at ~/src/oconnell_thomas_2026_thesis (clone it from
github.com/tom-f-oconnell/oconnell_thomas_2026_thesis if missing). The .tex source is
easier to search than the PDF.


## Code changes

When implementing experiments, modify ~/src/olfsysm (the model) or the library code in
this repo or ~/src/hong2p, and then make scripts in this folder as needed. If either is
not on this machine, clone it to that location from the ejhonglab Github. The
al_analysis venv normally installs both from Github (non-editable), so for changes to
either to take effect, install them editable into ./venv (after `git pull`-ing those
repos first, as the venv copy may be newer than a stale checkout):
```
./venv/bin/pip install -e ~/src/hong2p[all]
./venv/bin/pip install -e ~/src/olfsysm
```
olfsysm is a C++ extension, so re-run its `pip install -e` after any C++ change. Run the
tests (with QUICK=1, see the al_analysis CLAUDE.md) before committing model changes.

If multiple experiments would benefit from an overlapping set of changes to any library
code, start with some of those code changes, regardless of which experiments you are
running first. You may spawn up to 3 subagents (Opus 5.5 extra) to work on code changes
necessary for any future experiments (prioritize code changes for experiments with
higher priority), so they can work in parallel with ongoing experiments (which may take
a non-negligible amount of simulation time). Commit and push as code changes are done,
so another instance of claude working on another computer may benefit from the changes
too. If the remote has commits you do not (for this repo, olfsysm or hong2p), rebase
your commits on top of them (`git pull --rebase`), make sure nothing broke, and then
push. Never force push.

Scripts in this folder should write their large outputs (model output directories, etc)
under ./outputs/E<n>_<short-description>/, which is gitignored. Compact results (plots,
summary CSVs) go to the shared directory, see below.


## System resources

Run as many parallel experiments / simulations as you can to utilize the system's
resources (CPU cores / memory), stopping just short of loading the system to the point
of causing major problems. These machines are shared with other lab members: check
`uptime`, `nproc` and `free -g` before launching anything big, keep the total CPU use of
your simulations to roughly half the cores unless the machine is otherwise idle, and run
simulations under `nice -n 10`.


## Shared directory on tensor

Write markdown notes about current priorities / experiment interpretation to the NAS
tensor, in the Tom/claude_kc_remapping directory (will refer to this as the "shared
directory"). On hal, tensor is automounted at /local/tensor, so the shared directory is
/local/tensor/Tom/claude_kc_remapping (`ls` it to trigger the mount). On other hosts,
find the mount with `mount | grep tensor`. The shared directory already contains copies
of the report, apl_pair_analysis/, from_remy/ and from_yang/.

In the shared directory, create/maintain/read at least the following:
- experiment_priorities.md

- <hostname-running-claude>_local_data.md (e.g. hal_local_data.md)

- nas_data.md

- chat.md: for communicating with agents running on other computers (let me know if
  there is a better way to do this that you'd like me to set up, but only if it would
  fundamentally improve the workflow). Append-only, with each entry starting with a
  line like `## 2026-11-04 17:05 hal`, so entries from different hosts never overwrite
  each other.

In the shared directory, you may also create/edit:
- arbitrary markdown files to summarize findings / keep notes. Notes that are more
  for you than for me should be organized under a notes/ subdirectory.

- proposed_experiments.md: add new experiments not already in experiment_priorities.md
  here, and if some of these should be higher priority than something in
  experiment_priorities.md, talk me through it and then move those items into
  experiment_priorities.md after getting my OK.

Several agents (on different hosts) edit these files concurrently over NFS, so always
re-read a shared file immediately before editing it, and keep each edit small. Before
editing experiment_priorities.md, take a lock by `mkdir`-ing experiment_priorities.lock
in the shared directory (atomic on NFS) and `rmdir` it right after; a lock older than
10 minutes can be assumed stale and removed. Claim an experiment by writing its
"running from <host>" status line (see below) BEFORE starting it, so two hosts do not
run the same experiment.

File-change notifications (inotify) do not work over NFS, so monitor chat.md by polling
(e.g. a background loop that compares its mtime/checksum every ~60s and exits when it
changes, so you get woken up). Keep such a process running whenever you are not
otherwise busy, so you can copy data that other agents may request (in addition to
other collaboration or whatever you think might be useful there).

experiment_priorities.md should contain a numbered list of experiments to run (in
priority order), and each entry (describing one computational experiment) should have
the following:
- a stable experiment ID, E<n>, used in script and note filenames (see naming below).
  Start from the report's E1-E10, and give new experiments the next unused number. Never
  reuse or renumber IDs: the list position changes with priority, the ID does not.

- a short description of experiment and goals, including possible outcomes and what
  their interpretations would be (including how those interpretations might
  constrain/inform future experiments)

- things that are required to run it (e.g. code changes, data that is likely to exist
  in the lab but may not be accessible on the current machine)

- references to any relevant additional markdown note files written

- current status, if you have started working on it, but have not yet completed it.
  When the main experiment scripts are started, add a line saying that it is running
  (and from which host, when it was started, and expected duration). If an experiment
  you were running crashed or was unsuccessful for some reason (e.g. if code needs to be
  changed and re-run), update status as appropriate. Also say if you are blocked waiting
  on certain data to be copied here.

<hostname>_local_data.md should summarize uncommitted data on this machine, that might
be useful to other agents working on these experiments on other computers, but might
not be accessible on these other machines. Focus on analysis intermediates or processed
datasets in this repository and scratch/script outputs you made, but also (just once
when setting this file up) summarize datasets that might be useful in other repos /
non-NAS drives on this computer.

If experiment_priorities.md items list data that you think may be accessible on this
machine only, copy them to the shared directory on tensor if it will take less than a
couple minutes to do so, otherwise just maintain a description of the data in the
<hostname>_local_data.md file (as for all datasets, whether copied or not).
If you really need to share stuff like >~10GB model outputs or things like that, ask me
before copying that data to tensor. All of this shared data should reside under
Tom/claude_kc_remapping/data, which you can create if it doesn't exist. Compact
experiment outputs (plots, summary CSVs; well under 1GB per experiment) that the
interpretation markdown refers to should go under
Tom/claude_kc_remapping/results/E<n>_<short-description>/ instead, with large model
outputs staying local (and listed in <hostname>_local_data.md).
If there is some data that you need, but is not accessible on your computer, check any
<hostname>_local_data.md files that other agents (on other hosts) may have created in
the shared directory, and request they copy data via the chat.md file (or ask me to deal
with it, if that's not working).

nas_data.md: create this only if it's not already there, otherwise just reference it as
needed to see whether potentially useful datasets might exist in other paths on tensor.
When creating this file, search tensor (and tensor-archive) for the same type of
potentially useful processed data / analysis intermediates / reports / plots / slide
packs, and summarize the items in a bulleted list in this file. tensor holds ~90TB,
mostly raw imaging data, so do not `find` all of it blindly: restrict to relevant file
types (md, pptx, pdf, csv, tsv, parquet, pkl, html; skip tif / raw movies), limit the
depth (e.g. 4), and start with the Tom, Remy, Yang, Remy_Yang, MB_Model, MB_team and
"Lab Meetings" directories.


## Completing an experiment

When you complete an experiment and its interpretation:
1. write a new markdown file in the shared folder with interpretation (longer form than
   in the summary in the experiment priorities entries above)

2. edit the priorities file to add additional entry text referring to that additional
   markdown, then move the experiment entry to another numbered list in a "Completed"
   section at the bottom of the file (renumber remaining entries in priority list as you
   do this, but not their E<n> IDs). Entries in this list should be ordered with
   experiments we learned more from presented earlier.

3. make an artifact summarizing the findings of the experiment (with goals up top, and
   next steps summarized at the bottom) (this could just be taken directly from or
   lightly processed from your markdown from #1). Record the artifact URL in the
   markdown from #1 and in the Completed entry.

Before deciding to start an experiment that will take more than a couple minutes,
re-read experiment_priorities.md (and chat.md) in the shared directory to see if
priorities have changed.


## Naming and choosing experiments

When naming markdown notes relevant to a single experiment only, or naming scripts to
run single experiments, name them as E<n>_<short-description>.<ext> (n = the stable
experiment ID), with <short-description> all lowercase and underscore-separated [ascii
only]. Try to keep similar naming between scripts and these files that summarize goals
/ interpretation of the experiments they run.

Group experiments together if it would save time to run them simultaneously, but
otherwise make one script per experiment.

When deciding which experiments to test when, consider the following factors (more
important first):

1. How informative will it be in picking future simulation experiments? Consider how
   much we would be able to guide or rule out future experiments based on how long it
   will take to execute a given experiment until there is enough evidence to make a
   conclusion or future choices.

2. The degree to which the mechanism-under-test is likely to move the ORN odor-odor
   representation distances towards those of the KCs (or how likely it is to move the
   model representations towards those of the KCs). If this is easier to evaluate than
   1, you may swap priority of 1 & 2.

3. How long it will take to run. Very long running parameter searches, particularly
   those that are not (for other reasons) very high priority, should be de-prioritized,
   except when increasing their priority would make use of otherwise idle system
   resources that are not soon going to be used by other processes.


## Data from Remy and Yang

If they are useful, there should be copies of Remy's KC data and some of Yang's KC data
(the latter to lower concentrations of odors, expected to produce more distinct spatial
activity patterns in the APL). Remy's megamat KC data is loaded by
`load_remy_megamat_kc_responses` in al_analysis/mb_model.py and Yang's by
`load_yang_kc_data` in scripts/model_yang_mixtures.py. The data itself is mostly under
al_analysis/data/from_remy and al_analysis/data/from_yang/20260415_YZtransfer.

I asked Remy for her current analysis (on the same paper, one version of which was
presented in chapter 2 of my thesis), and she sent me all the contents of from_remy/.
On first run, make a markdown file in the shared NAS folder (as notes/remy-summary.md)
that summarizes what Remy sent me, in particular as relevant to these computational
experiments.

Make a similar notes/yang-summary.md file to describe anything Yang sent me
(also only on first run), or if I add / tell you about new stuff from either of them.
Some of her slides summarizing her data are in from_yang/ (these powerpoints are >100MB
each, so they are gitignored and only in the shared directory's copy of from_yang/).

Followup on questions you had in your initial markdown report:
When I asked Yang about whether her APL16 odors were all at the same concentrations as
Remy's panel 1 (megamat) and 2 (validation2) panels, she said:
"Actually, no. the 4 overlapping w/ valid 2 are not. the conc for PAA is -3 instead of -4, other 3 are all at -2 instead of things like -1.75 etc in valid2."
That is, paa was at 10^-3 (not -4), and -aPine, euc and B-myr were all at 10^-2 (vs
-1.5, -1.75 and -2 in the lab's validation2 ORN data), while the 12 megamat odors were at
the same concentrations as Remy's. Check whether the lab has ORN data at Yang's
concentrations before using the validation2 ORN responses as model input for those four
odors, and if not, say which approximation you used in the experiment notes.
