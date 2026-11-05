We are following up on a report you made, in the markdown:
"APL and the ORN\342\206\222KC remapping of odor\342\200\223odor correlations.md"

The report may reference the analysis and outputs in apl_pair_analysis/, which you used
in construction of the report (and provided alongside for download).

A local copy of my thesis is at ~/src/oconnell_thomas_2026_thesis.

When implementing experiments, modify ~/src/olfsysm (the model) or the library code in
this repo or ~/src/hong2p, and then make scripts in this folder as needed. If either is
not on this machine, you may clone them to those locations, from the ejhonglab Github.
If multiple experiments would benefit from an overlapping set of changes to any library
code, start with some of those code changes, regardless of which experiments you are
necessarily running first. You may spawn up to 3 subagents (Opus 5.5 extra) to work on
code changes necessary for any future experiments (prioritize code changes for
experiments with higher priority), so they can work in parallel with ongoing experiments
(which may take a non-negligible amount of simulation time). Commit and push as code
changes are done, so so another instance of claude working on another computer may
benefit from the changes too. Rebase your commit on top of any future commits, making sure
nothing broke, if the remote has changes you do not (for this repo or olfsysm / hong2p)
(and then push).

Run as many parallel experiments / simulations as you can to utililze the systems
resources (CPU cores / memory), stopping just short of loading the system to the point
of causing major problems.

Write markdown notes about current priorities / experiment interpretation to the NAS tensor,
in the Tom/claude_kc_remapping directory (will refer to this as the "shared directory").
In the shared directory, create/maintain/read at least the following:
- experiment_priorities.md

- <hostname-running-claude>_local_data.md

- nas_data.md

In the shared directory, you may also create/edit:
- arbitrary markdown files to summarize findings / keep notes. notes that are more
  for you than for me should be organized under a notes/ subdirectory.

- a shared chat.md file, to be used for communicating with agents running on
  other computers (let me know if there is a better way to do this that you'd like me to
  set up, but only if it would fundamentally improve the workflow)

- proposed_experiments.md: add new experiments not already in experiment_priorities.md
  here, and if some of these should be higher priority than something in
  experiment_priorities.md, talk me through it and then move those items into
  experiment_priorities.md after getting my OK.

experiment_priorities.md should contain a numbered list of experiments to run (in priority order),
and each entry (describng one computational experiment) should have the following:
- a short description of experiment and goals, including possible outcomes and what
  their interpretations would be (including how those interpretations might
  constrain/inform future experiments)

- things that are required to run it (i.e. code changes, data that is likely to exist
  in the lab but may not be accessible on the current machine)

- references to any relevant additional markdown note files written

- current status, if you have started working on it, but have not yet completed it.
  when the main experiment scripts are started, add a line saying that it is running
  (and from which host, when it was started, and expected duration). if you an
  experiment you were running crashed or was unsuccesful for some reason (i.e. if code
  needs to be changed and re-run), update status as appropriate. also say if you are
  blocked waiting on certain data to be copied here.

<hostname>_local_data.md should summarize uncommitted data on this machine, that might
be useful to other agents working on these experiments on other computers, but might
not be on accessible on these other machines. Focus on analysis intermediates or
processed datasets in this repository and scratch/script outputs you made, but also
(just once when setting this file up) summarize datasets that might be useful in other repos / non-NAS drives on this computer.

If experiment_priorities.md items list data that you think may be accessible on this
machine only, copy them to the shared directory on tensor if it will take less than a
couple minutes to do so, otherwise just maintain a description of the data in the
<hostname>_local_data.md file (as for all datasets, whether copied or not).
If you really need to share stuff like >~10GB model outputs or things like that, ask me
before copying that data to tensor. All of this shared data should reside under
Tom/claude_kc_remapping/data, which you can create if it doesn't exist.
If there is some data that you need, but is not accessible on your computer, check any
<hostname>_local_data.md files that other agents (on other hosts) may have created in
the shared directory, and request they copy data via the chat.md file (or ask me to deal
with it, if that's not working). Keep a background process monitoring the chat.md file
for changes, so you can copy data that other agents may request (in addition to other
collaboration or whatever you think might be useful there).

nas_data.md: create this only if it's not already there, otherwise just reference it as
needed to see whether potentially useful datasets might exist in other paths on tensor.
When creating this file, search all of tensor for the same type of potentially useful
processed data / analysis intermediates / reports / plots / slide packs, and summarize
the items in a bulleted list in this file.

When you complete an experiment and its interpretation:
1. write a new markdown file in the shared folder with interpretation (longer form than
   in the summary in the experiment priorities entries above)

2. edit the priorities file to add additional entry text referring to that additional markdown,
   then move the experiment entry to another numbered list in a "Completed" section at the
   bottom of the file (renumber remaining entries in priority list as you do this).
   entries in this list should be ordered with experiments we learned more from
   presented earlier.

3. make an an artifact summarizing the findings of the experiment (with goals up top, and
   next steps summarized at the bottom) (this could just be taken directly from or
   lightly processed from your markdown from #1)

Before deciding to start an experiment that will take more than a couple minutes, check
this file to see if priorities (in the shared markdown file) have changed.

When naming markdown notes relevant to a single experiment only, or naming scripts to
run single experiments, name them as E<n>_<short-description>.<ext>, with
<short-description> all lowercase and underscore-separated [ascii only]. Try to keep
similar naming between scripts and these files that summarize goals / interpretation of
the experiments they run.

Group experiments together if it would save time to run them simultaneously, but
otherwise make one script per experiment.

When deciding which experiments to test when, consider the following factors (more important first):

1. How informative will it be in picking future simulation experiments? Consider how
   much we would be able to guide or rule out future experiments based on how long it
   will take to execute a given experiment until there is enough evidence to make a
   conclusion or future choices.

2. The degree to which the mechanism-under-test is likely to move the ORN odor-odor
   representation distances towards those of the KCs (or how likely it is to move the
   model representations towards those of the KCs). If this is easier you evaluate than
   1, you may swap priority of 1 & 2.

3. How long it will take to run? Very long running parameter searches, particularly
   those that are not (for other reasons) very high priority, should be de-prioritized,
   except when increasing their priority would make use of otherwise idle system
   resources that are not soon going to be used by other processes

If they are useful, there should be copies of Remy's KC data and some of Yang's KC data
(to lower concentrations of odors expected to produce more distinct spatial activity
patterns in the APL), both of which I believe are currently loaded in
al_analysis/scripts/model_yang_mixtures.py, though the data is probably mostly under
al_analysis/al_analysis/data.

I asked Remy for her current analysis (on the same paper which one version of was
presented in chapter 2 of my thesis), and she sent me all the contents of from_remy/.
On first run, make a markdown file in the shared NAS folder (as notes/remy-summary.md)
that summarizes what Remy sent me, in particular as relevant to these computational
experiments.

Make a similar notes/yang-summary.md file to describe anything Yang sent me
(also only on first run), or if I add / tell you about new stuff from either of them.
Some of her slides summarizing her data are in from_yang/

Followup on questions you had in your intial markdown report:
When I asked Yang about whether her APL16 odors were all at the same concentrations as
Remy's panel 1 (megamat) and 2 (validation2) panels, she said:
"Actually, no. the 4 overlapping w/ valid 2 are not. the conc for PAA is -3 instead of -4, other 3 are all at -2 instead of things like -1.75 etc in valid2."

