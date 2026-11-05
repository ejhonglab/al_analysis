We are following up on a report you made, in the markdown:
"APL and the ORN→KC remapping of odor–odor correlations.md"

The report may reference the analysis and outputs in apl_pair_analysis/, which you used
in construction of the report (and provided alongside for download). The two 16x16 KC
distance matrices the report analyzed (TNT3c_dist_apl16.csv [control] and
TNT_dist_apl16.csv [APL output blocked], both from Yang) are regenerated there from
Yang's per-recording data (see below). Note they are 1 - cosine similarity over all KCs
of a recording, averaged over recordings, not 1 - Pearson as the report says; see
data/yang_apl16_kc_olfdb/README.md in the shared directory.

Yang's per-recording APL16 KC data (TNT: olfdb dataset 19, TNT3c: dataset 20; per trial,
per hemisphere) is in the lab's odor response database olfdb (the hong-data repo,
deployed at /opt/olfdb on hal, served at http://hal:8080; HTTP API listed in its
design.md, e.g. `/api/datasets`, `/api/download?ids=<id>&format=long`). It also has
Remy's kiwi/control-mixture KC data and my megamat and validation2 ORN glomerulus data.
Copies of Yang's two datasets, plus per-recording RDMs (Pearson and cosine) and per-odor
response fractions computed from them, are in the shared directory under
data/yang_apl16_kc_olfdb/ (made by yang_apl16_kc_data.py in this folder; re-run it to
regenerate).

Yang had her own Claude develop an independent static MB (KC+APL) model. That work (Oct
2026) is on tensor under Tom/claude_APL_modeling_YZ4TC/Claude_APL_analysis/, with APL
synapse tables from four connectomes and a comparison of her model against her
per-recording TNT / TNT3c KC data (8 TNT recordings from 7 flies, 11 TNT3c recordings
from 8 flies). It is separate work and still in progress, and it is unclear how much we
gain from the model itself, but the reports may be worth reading (start with
apl_synapses_markdown_combined_2026-10-23.md). The per-recording data itself is not in
that folder.

A local copy of my thesis is at ~/src/oconnell_thomas_2026_thesis (clone it from
github.com/tom-f-oconnell/oconnell_thomas_2026_thesis if missing). The .tex source is
easier to search than the PDF.


## Code changes

When implementing experiments, modify ~/src/olfsysm (the model) or the library code in
this repo or ~/src/hong2p, and then make scripts in this folder as needed. If either is
not on this machine, clone it to that location from the ejhonglab Github. Changes to
either only take effect if the al_analysis venv has it installed editable from that
checkout, which is not guaranteed (on hal, olfsysm is editable from ~/src/olfsysm, but
hong2p is a plain install from Github). Check with `./venv/bin/pip show <pkg>` (an
editable install reports "Editable project location") and, if needed, after
`git pull`-ing the checkout (the venv copy may be newer than a stale checkout), run:
```
./venv/bin/pip install -e ~/src/hong2p[all]
./venv/bin/pip install -e ~/src/olfsysm
```
olfsysm is a C++ extension, so re-run its `pip install -e` after any C++ change. Run the
tests (with QUICK=1, see the al_analysis CLAUDE.md) before committing model changes.

If multiple experiments would benefit from an overlapping set of changes to any library
code, start with some of those code changes, regardless of which experiments you are
running first. You may spawn up to 3 subagents (Opus 5.5, at xhigh effort) to work on
code changes necessary for any future experiments (prioritize code changes for
experiments with higher priority), so they can work in parallel with ongoing experiments
(which may take a non-negligible amount of simulation time). Commit and push as code
changes are done,
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
of causing major problems. These machines are shared with other lab members, so check
`uptime`, `nproc` and `free -g` before launching anything big, run simulations under
`nice -n 10`, and aim for roughly half the cores unless the machine is otherwise idle.
olfsysm is compiled with OpenMP (so OMP_NUM_THREADS sets its thread count, and it uses
all cores by default) and parallelizes over odors, so give each job as many threads as
it has odors (16 for the APL panel, 17 for megamat) and pick the number of concurrent
jobs from that: on hal (48 cores), two 17-odor jobs (34 threads) at once is fine. On
machines with fewer cores (atlas), run fewer jobs at once rather than fewer threads per
job, and run one at a time if the machine is busy.


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

- <hostname>_status.md: heartbeat of what this host is doing right now (see "Claims,
  probes and status" below)

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

### Claims, probes and status

Experiments are claimed with their Status line, but on the first day shared code (the
harness and a parallel data module), shared statistics, and one-off probes of another
experiment's variants were done twice. So, in addition:

- Work claims. experiment_priorities.md has a "Work claims" table above the priority
  list. Before starting any shared-code module, data pull, or other task expected to
  take more than a few minutes that another agent could plausibly also start, add a
  row (item, host, started, ETA, scope / stub) under the usual lock; remove the row
  when the work is pushed or abandoned (say so in chat.md if abandoned). The table is
  for anything that is not itself an experiment entry.

- Interface first. Whoever writes shared code commits and pushes a stub (the module
  with function signatures and docstrings; bodies may raise NotImplementedError)
  within the first few minutes, before implementing, and names it in the claims row.
  Others build on that interface, or ask in chat.md for additions, rather than writing
  a parallel module.

- Probes. Any run of a variant that belongs to another experiment's design (a one-off
  check while working on something else, including one that crashed) gets a one-line
  note in that experiment's Status: host, time, what was run, result, and where the
  output is (or that it was not saved). The experiment's owner runs its variants; do
  not run them in bulk without taking over the claim.

- Heartbeat. Keep <hostname>_status.md in the shared directory: what this host is
  doing right now (start time, expected finish), what it will do next, and what is
  blocked on whom. Overwrite it at every change of activity, and at least every ~30
  minutes during long runs. Claims say what is reserved; this file says what is
  actually happening.

- Re-read before starting. Re-read chat.md, the claims table and the experiment
  statuses before starting any task expected to take more than a couple of minutes
  (not only before experiments), and again when one finishes.

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

Before starting any task that will take more than a couple of minutes (an experiment,
a shared-code change, a data pull), re-read experiment_priorities.md (the work claims
table and the experiment statuses) and chat.md in the shared directory to see if
priorities or claims have changed.


## Report artifact

The report markdown in this folder ("APL and the ORN→KC remapping of odor–odor
correlations.md") doubles as the running list of in silico experiments. Whenever new
experiments are added to experiment_priorities.md (E11 onwards), also:
1. append them to the report's "In silico experiments" numbered list, in the same style as
   E1–E10 (bold `E<n>, title (mechanism)` then goal / design / readout / prediction in one
   paragraph), after the `*Added ...*` note (extend that note with when and from what the
   new ones were conceived; keep E1–E10 unchanged);
2. rebuild the page with `./venv/bin/python build_report_artifact.py` and republish it to the
   existing report artifact (same slug, so the link stays valid; the slug and the URL are
   recorded at the top of experiment_priorities.md), then commit and push the markdown;
3. mention the republish in chat.md.
Only the experiment list changes this way; findings go in the per-experiment markdown
files, not in the report.

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
the same concentrations as Remy's. No ORN data exist at paa -3, euc -2 or -aPine -2.
Rule: when no exact concentration match is available, ORN data within 1 log10 v/v unit
(inclusive, so -4 may stand in for -3) may be used, and likewise when comparing across
datasets (e.g. Remy's vs Yang's KC data). Every time outputs affected by such a
substitution are discussed, note which results are affected (here: anything involving
paa, euc or -aPine model input).
