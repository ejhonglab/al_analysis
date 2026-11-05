We are following up on a report you made, in the markdown:
"apl_orn_to_kc_remapping_report.md"

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


## Model provenance

Model outputs must be fully explained by the model parameters plus the commits of our code.
olfsysm bakes its commit hash, a clean / dirty flag and a hash of its compiled sources into
the extension at build time (`olfsysm.build_info()`), and every `pip install -e` recompiles
everything, so there are no stale objects. `fit_mb_model` refuses to run unless the
installed olfsysm is a clean build matching its checkout (`mb_model.olfsysm_is_clean()`);
importing `kc_remapping_harness` makes that strict (`AL_ANALYSIS_ALLOW_DIRTY_OLFSYSM` is
ignored), so for the experiments here: commit the olfsysm change, `pip install -e
~/src/olfsysm`, run the tests, then run the experiment. Every experiment's provenance.json
includes `H.model_provenance(orn_deltas=...)` (olfsysm build, al_analysis / hong2p commit
and dirty state with diff, CPU, package versions, input hashes), and
`fit_and_plot_mb_model` outputs carry `model_provenance.json`. The CPU is recorded for
information only; nothing compares it.

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
  here (standard entry text plus "proposed by <host>, <date time>"), and if some of these
  should be higher priority than something in experiment_priorities.md, talk me through it
  and then move those items into experiment_priorities.md after getting my OK. This file
  holds OPEN proposals only: moving a proposal means cutting it from here (one line in the
  file's "Moved log"), pasting its full text into the experiment_priorities.md entry with
  the "proposed by / when" and "approved by / when" provenance, and adding its row to the
  experiment index (below). Exception, so that no host sits idle waiting for me: if your
  host would otherwise be idle (nothing runnable in the priority list is unclaimed and
  suited to it), move proposed experiments into experiment_priorities.md yourself, without
  my approval: give them the next E<n>, place them where their stated rationale puts them
  (write that rationale in the entry; I will reorder if I disagree), append them to the
  report markdown and republish the report artifact (see "Report artifact" below), say in
  chat.md what you moved and why, then claim one and start it. Proposals that would
  displace work another host is about to start still go through me.

- The experiment index: experiment_priorities.md starts with a table with one row per
  experiment ID (name, state: proposed / approved / running / completed, host, when and by
  whom it was proposed, when and how it was approved or moved into the list, when it was
  started, when it was completed and its position in the Completed list). Update the row at
  every state change, under the lock, in the same edit that changes the entry or the
  status line; it is the record of what happened when (for the reports) and the one place
  to see what is running and what awaits my OK.

Several agents (on different hosts) edit these files concurrently over NFS, so always
re-read a shared file immediately before editing it, and keep each edit small. Before
editing experiment_priorities.md, take a lock by `mkdir`-ing experiment_priorities.lock
in the shared directory (atomic on NFS) and `rmdir` it right after; a lock older than
10 minutes can be assumed stale and removed. Claim an experiment by writing its
"running from <host>" status line (see below) BEFORE starting it, so two hosts do not
run the same experiment.

### How the queue works (and what not to ask Tom)

The priority list is a queue. An idle host takes the next unclaimed item that it can run
(claims it with its Status line and index row, then starts it); if nothing runnable is
unclaimed, it moves a proposal into the list under the idle-host rule and starts that. A
blocked item (data not on hand, code missing) says in its Status line exactly what unblocks
it, and an idle host may take the unblocking itself as a work claim (a data pull, a
connectome query, a code change); items whose unblocking needs something only Tom can
provide (an access token, a dataset, a decision between proposals that displace other
hosts' work) say so, and that is what gets raised with him. Consequently, do not end replies
to Tom with offers or choices about queued work ("E9 or the coverage check?"): what runs next
is determined by the list and these rules, and the interface turns such closing lines into
suggestions he does not need to see. Raise only decisions that are genuinely his.

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


## Notes to keep for the final summary report

The shared directory has an `add_to_reports/` folder (README there). When I ask for a note
to be saved for the report (e.g. my questions about a parameter and your answer, a methods
caveat, a cross-experiment synthesis), write it as a self-contained markdown file in that
folder, with the same content as any artifact you made for it and the artifact URL in its
header, and add a row to the folder's README table. Keep working notes in `notes/`.

When I later ask you to summarize all findings (the final report over all experiments),
include every file in `add_to_reports/`, together with the per-experiment `E<n>_*.md`
files and the Completed list in experiment_priorities.md.

## Writing results (notes, chat, reports, replies to me)

Every statistic quoted in prose must carry its context: the baseline it is compared with,
described as a model configuration (what the model had and did not have) together with its
number, never the number alone; the configuration the new number belongs to, likewise
described; the direction of the change and whether that direction is an improvement; and
the value in the KC data (the target) or the noise ceiling. "Per-odor rate agreement 0.68
to 0.57" is ambiguous (range or change? which configuration is the baseline? better or
worse? how far from the data?); write "per-odor rate agreement fell from 0.68 for the bouton
model with connectome APL weights and no per-claw threshold to 0.57 once each claw's input
was thresholded at 70 Hz (worse; the model-to-data ceiling is ~0.95)".

In reports, and also in replies to me in the terminal chat, prefer tables (or plots) that
show the baseline, each variant and the KC data side by side, and keep most numbers out of
the text: the text should say what moved, which way,
and whether it reached the data. Spelling out the full context for every number in prose is
too verbose, so put the numbers in the table and the judgement in the sentence.

Name Spearman-based statistics so the text itself says they are rank correlations: the
per-odor response-rate comparison is "rate rank agreement" (or "rate rank match"), and
likewise "rank agreement with the control matrix". Dropping "rank" because "Spearman" is
mentioned nearby is allowed but is not the default.

Describe experiments in plain English ("the per-claw threshold sweep", "the KC-KC lateral
inhibition experiment"), not by ID. Polished reports (the summary artifact, the final report)
use no IDs in prose. Preliminary and experiment-specific notes (the per-experiment markdown,
chat.md, replies to me) use the plain-English description first and then the ID in
parentheses: "the per-claw threshold sweep (E16)". The same applies when referring to
experiments in the terminal chat with me, including when proposing them or asking whether to
run them: a short plain-English description first, then the ID in parentheses, never the
bare ID ("the per-glomerulus drive normalization sweep (E20)", not "E20").

### Summary reports (the all-experiments summary and the final report)

- **No experiment IDs in the text**, not even in parentheses: refer to experiments by the
  canonical short names below (same wording every time). The one place IDs appear is a
  collapsed "Experiment key" right under the title (a `??? Experiment key ...` / `???`
  block in the markdown, which md_to_artifact.py renders as a details element, collapsed by
  default): one bullet per experiment, `E<n>: <canonical name> (completed / running / not
  started)`, so a reader can match the page to the per-experiment notes. Per-experiment
  notes and chat keep "plain-English name (E<n>)".
- **Background first.** Before any model number, a section with a figure and a table that
  shows what the data are: the ORN input matrix, the control KC matrix and the APL-blocked
  KC matrix for the 16 odors; what the ORN->KC remapping looks like in control flies (which
  pairs move, how much, rank agreement with the ORN input); and what blocking APL output
  changes (mean distance, response fractions, the monotone on-vs-off relation and its one
  pair-specific exception). Only then the model baseline and the mechanisms.
- **Pages are generated, not hand-converted.** Write the summary as markdown, convert it with
  `md_to_artifact.py` (handles `###`, list items and paragraphs wrapped over several lines,
  tables, figures embedded from `![caption](path)`), and do not publish while its lint
  reports anything (leftover `###`, `**`, backticks, link syntax, line breaks inside list
  items). Read the produced HTML's structure (headings, list items, tables) once before
  publishing; the artifact renders exactly that file.
- Numbers in tables with the data value and the model baseline alongside (see above); prose
  says what moved and whether it reached the data.
- **No provenance chatter.** Summary reports never mention which host or which model / agent
  ran, wrote or requested anything, do not open with a status list of what is completed,
  running or started, and have no byline or "Artifact: ..." line: the page starts with the
  title and the first content section.
- **Pending work goes at the bottom, with context.** Experiments still running or not yet run
  are described near the end, each with what it tests, the possible outcomes and what each
  outcome would mean for the picture, not as a bare list.
- **Every statistic in plain English.** Say what a number is of (e.g. "the distance between
  two odors' KC population responses, one minus their cosine similarity, averaged over the
  120 odor pairs"), define each statistic once in plain words where it first appears (a
  short definitions list after the first figure works), and avoid shorthand such as
  "split-half ceiling" or "iso R2" without that explanation; "the agreement between two halves
  of the recordings, the most any model could reach" is the register to use.

- **Odor names.** Odors are always written by their abbreviations only (2h, IaA, pa, 2-but,
  eb, ep, 6al, t2h, 1-8ol, 1-5ol, 1-6ol, ms, paa, -aPine, euc, B-myr; ASCII hyphen in -aPine),
  in prose, tables and figures alike, with no full names anywhere, not even a key. If an odor
  has no abbreviation on hand, look it up in olfdb's odors table (http://hal:8080, see its
  design.md for the API) rather than writing the full name.
- **Genotype labels.** In every figure title, axis label, legend entry, table header and
  row label the two genotypes are "control flies" and "APL output blocked (TNT)", never "APL
  blocked" or "TNT" alone (prose may say "with APL output blocked" once the label has been
  introduced). Model conditions stay "APL on" / "APL off".
- **Distance-matrix figures.** Panel titles name only what is shown ("ORN input", "KCs,
  control flies", "KCs, APL output blocked (TNT)"); the distance function appears once per
  panel, in the colorbar label, by its name ("cosine distance (1 - cosine similarity) across
  glomerular responses"), never as a bare "similarity"; scatter axes say "...: cosine distance".
  Long subplot titles are split over two lines, the statistic in parentheses on the second.
- **Opaque terms and methods.** A technical term the main text needs but should not stop to
  explain (isotonic, ...) gets a hover definition: put a `%%% hover` block (lines
  `term: definition`, closed by `%%%`) anywhere in the markdown and `md_to_artifact.py` wraps
  every whole-word use in paragraphs, lists and tables with a dotted underline that shows the
  definition on hover. A method that deserves a real explanation (how it is done, why it was
  chosen, what the alternatives were and why not) goes in a "Footnotes" section at the very
  end of the page (`## Footnotes {#footnotes}`, one `### ... {#fn-<id>}` per note) and the
  phrase in the text that uses it is linked there with `[phrase](#fn-<id>)`. Headings take an
  explicit `{#id}`; without one they get a slug of their text.

Canonical experiment names for summary reports:

| ID | name used in summary reports |
|---|---|
| E1 | the APL knockout baseline |
| E2 | local APL units |
| E3 | APL inhibition of PN boutons (drive-dependent APL gain) |
| E4 | the PN->KC wiring substrate |
| E5 | KC heterogeneity |
| E6 | the readout window and time constants |
| E7 | the data-only confound controls |
| E8 | KC-KC lateral inhibition |
| E9 | structured centrifugal input to the calyx |
| E10 | the joint fit to both matrices |
| E11 | claw coincidence with claw-level inhibition |
| E12 | breadth-weighted APL |
| E13 | the bouton base with claw coincidence |
| E14 | oracle co-responder suppression |
| E15 | the breadth x shared-input APL rule |
| E16 | per-claw saturation and the per-claw threshold |
| E17 | panel-general glomerulus scores and the APL-synapse anatomy check |
| E18 | bouton vesicle depletion |
| E19 | the threshold x breadth x KC-KC grid |
| E20 | per-glomerulus drive normalization |
| E21 | the APL-blocked sharing diagnostic |
| E22 | the softer per-claw knee |
| E23 | the weak-odor responder analysis |
| E24 | input scaling for the terpenes and 1-octanol |
| E25 | the joint fit with the structure-beyond-ORN targets in the loss |
| E26 | the supralinear per-claw transfer |
| E27 | the oracle glomerulus gain for the weak-input odors |

## Model-output archive (data/model_outputs/ in the shared directory)

Every experiment's model outputs are copied to the shared directory under
`data/model_outputs/E<n>_<short-description>/` (the experiment's results / outputs folder
name), organized by experiment and not by host: the KC x odor spike-count and response
matrices of every variant and condition, the variants' parameters (`params.csv` built from
the stats tables, plus `tuned_params.json` or equivalent where the run wrote one), the
`provenance.json` from results/ (script, commit hashes, base kwargs, sweep settings), the run
logs, and any wiring or gain matrices the run saved; `data/model_outputs/inputs/` holds what
all experiments share (the ORN input matrix, the standard PN->KC wiring matrices). Run
`./venv/bin/python archive_model_outputs.py E<n>_<desc>` from the repo folder when an
experiment completes (and again if more variants are added); it also upserts the row in
`data/model_outputs/README.md`. Record in every new script the parameters needed to
regenerate each variant (tuned threshold and APL scales as absolute values, not only factors,
in the stats table) and the olfsysm commit in the provenance.

## File and folder names

Everything we produce (files, folders, figure names, archive and report names) uses ASCII
characters only: letters, digits, `_`, `-`, `.`; no en or em dashes, arrows, multiplication
signs or other characters that need a Unicode input method. This applies to names only;
the contents of notes keep normal typography.

## Report artifact

The report markdown in this folder ("APL and the ORN→KC remapping of odor–odor
correlations.md") doubles as the running list of in silico experiments. Whenever new
experiments are added to experiment_priorities.md (E11 onwards; by me, or by an otherwise
idle host under the exception above), also:
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

## Report copies (reports/ in the shared directory)

Every artifact page we publish (per-experiment summaries, the report page, summary and
add_to_reports pages) must also exist as a self-contained file under
`<shared>/reports/`, the same document the artifact's share > export download gives. The
artifact server serves the HTML we upload unchanged, so:
1. build every artifact page as a local file and publish it with `publish_files` from that
   file, never from inline content. Prefer a single `index.html` with figures embedded as
   data URIs (inline base64, no files alongside): its export is one .html file. Upload files
   alongside `index.html` only when the page needs them (then the export is a .zip of the
   folder), and never upload a figure that is also embedded;
2. after every publish or republish, run (from this folder)
   `./venv/bin/python save_report_copy.py <index.html or directory> <name> <artifact url>`,
   which copies the file to `<shared>/reports/<name>.html`, or, when the artifact has files
   alongside `index.html`, pass the directory holding exactly the published files and it
   writes `<name>.zip` (index.html plus those files, as the export does); it upserts the
   page's row in `reports/README.md` (name, page title, URL, when and from which host). Use the
   same `<name>` on republish so the folder holds the latest version only: `E<n>_<short>` for
   experiment pages, `report_<short>` for the report, `summary_<date>` for summaries,
   `note_<short>` for add_to_reports pages.
Record the artifact URL where the scheme already asks for it (the experiment markdown, the
Completed entry, the add_to_reports README); the reports/ copy is in addition, not instead.

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
