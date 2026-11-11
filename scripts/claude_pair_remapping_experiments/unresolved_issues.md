# Unresolved issues

Errors and bugs that affect, could affect, or have affected analysis outputs in this folder
(statistics, figures, reports). One entry per issue: what is wrong, what it touches, what has been
done, what remains. Move an entry to "Resolved" (with the commit) once every affected output has
been regenerated or annotated. Keep entries short; long investigations go in the shared
directory's notes/ and are linked from here.

## Open

### The noise correction of the KC distances assumes noise independent between odors, which the recordings' repeats contradict (found 2026-11-11, atlas's planning subagent, probe)
- **What.** `noise_floor_trial_distances.py` corrects each recording's odor-odor distances on the assumption that trial-to-trial noise adds
  nothing to the dot product between different odors. In the recordings two different odors of the same repeat of the panel are more alike
  than two odors of different repeats, in all 19 recordings (single-trial cosine distance gap 0.006 to 0.053 in control flies, 0.041 to 0.113
  with APL output blocked (TNT); a planning subagent's read-only probe, single pass, no intervals), so part of the noise is shared between the
  odors of a repeat and the correction is biased low: by the probe, mean distances 0.79 / 0.58 (control flies / APL output blocked (TNT))
  where an estimate from products of different repeats gives about 0.81 / 0.62, the mean APL effect +0.212 where it gives +0.186, and the six
  pairs with near-identical ORN input 0.30 / 0.12 where it gives about 0.36 / 0.22.
- **Touches.** results/noise_floor/; the registry's noise-corrected values (the mean distances, the block's +0.072, the effect's +0.212) and
  the summary rows that quote them; the "corrected for trial-to-trial noise" rows of E43, E48 and E68 and of E76's unit table; the "trial
  noise removed" column and the "at zero" sentence of E78's entry and report item 78.
- **Done.** Filed; the flies' variability across the three repeats (E81, atlas) estimates the noise-free distances with intervals (part A4).
- **Done (E81 parts A and B, atlas, 2026-11-11).** The shared part is confirmed with intervals (E81_repeat_structure.py,
  results/E81_repeat_structure/): the same-repeat gap is 0.025 (90 % interval over resampled recordings 0.019 to 0.033) in control
  flies and 0.070 (0.055 to 0.085) with APL output blocked (TNT), positive in all 19 recordings. The noise-free estimate from cosines
  of different repeats (`kc_remapping_harness.noise_free_pair_distances`; registry `noise_free_*`) gives mean distances 0.811
  (0.782 to 0.835) / 0.625 (0.569 to 0.677) where the correction in use gives 0.789 / 0.577, a mean APL effect of +0.186 (0.127 to
  0.244) for +0.212, the six pairs 0.365 / 0.222 for 0.298 / 0.122 and the block 0.067 for 0.072. With APL output blocked (TNT) the
  estimate from the first against the third repeat is 0.030 (0.019 to 0.040) below that from adjacent repeats (control flies
  +0.001), so the drift does not cancel completely there. In the models' pipeline trials, whose variability is all shared by the
  odors of a trial, the correction in use takes the six pairs to about 0 or below it at the matched levels (C2: 0.012 / 0.005; the
  other archived models -0.65 to +0.10), where the noise-free estimate keeps them near their trial means (C2: 0.213 / 0.145; the
  others 0.07 to 0.28; saved_trials_structure.csv).
- **Remains.** Annotation of every quoted noise-corrected value (the list under Touches), or the correction replaced by the
  noise-free estimate; for E81's write-up.

### E75's part B named the archive's least-common-error variants wrongly (found 2026-11-10 18:15, viki's planning subagent for E77; checked by viki)
- **What.** The glomerulus profile of APL inhibition (E75) part B ranks every archived variant by how much of the error common to
  all wirings its pattern error carries (archive_vs_common_error.csv; the numbers are right). The write-up named the variants by
  their experiment's canonical name, not by what they are: E3's `hemibrain-wd20_inhadd*` were called "APL inhibition added at the
  PN boutons", but they are E3's part A, the antennal lobe's divisive normalization weakened (pn_inhadd raised with pn_inhsc,
  E3_drive_dependent_gain.py lines 207-211) on the one-row-per-KC wd20 base, whose default (inhadd 31.4) already projects 0.52.
  E3's bouton arm (`bouton_pnapl_x0` to `x2`) projects 0.80 to 0.82, as the rest of the archive. "The per-claw saturation at
  130 Hz (E16)" is `bouton_thr130_tanh50`, a 130 Hz per-claw threshold with a tanh cap at 50 Hz (the saturation-only variants
  project 0.79 to 1.04). "The uniform and hemidraw substrates, 0.43 to 0.53" are the lowest three of 34 random draws (median
  0.68). Their pattern errors (0.056 to 0.074 under the readout; other E4 draws 0.052 to 0.055) are not below C2's 0.056.
- **Affects.** E75_glomerulus_apl_profile.md (summary, part B, interpretation, next step 1) and its page; E75's Status of 15:38
  and its Completed entry in experiment_priorities.md; viki's chat posts of 14:50 and 17:51; the E77 brief (corrected by E77's
  planner, whose entry gives the right names). The conclusion "the variants least aligned with the common error change what
  reaches the KCs" stands; "bouton inhibition" is not one of them.
- **Done.** Note corrected in place with a dated correction line; page rebuilt; Status and Completed entry annotated; the summary
  passage for E75 uses the corrected names (its planner had the correction).
- **Remains.** Nothing once the summary passage is in (the page was republished by hal at 18:23, version 3).

### The realism bands compared the models' two-spike responses with the KC data's whole-recording z-score responses (found 2026-11-10, hal's planning subagent; decided by Tom 16:10)
- **What.** Since the bands were set (2026-11-09), the model side counted a response at two or more spikes and the data side at a
  trial mean above 1 z of Yang's whole-recording z-score, whose unit grows with how much a KC responds (1.4 to 3.0 times its baseline
  noise). The unit change alone moves the flies' own odors-per-KC distribution by 0.064 (control flies) / 0.078 (APL output blocked
  (TNT)) against bands of 0.167 / 0.196, so many odors-per-KC flags (116 archived variants carry only an APL-off one between 0.196 and
  0.25) were the criterion's, not the model's.
- **Affects.** Every realism flag and band verdict written since 2026-11-09 (results/apl_effect/by_variant.csv `realism_flag`, the
  Realism sections of the per-experiment notes, the summary's best-models realism row and the "Realism flag:" posts), most visibly the
  reference model's and its fits' APL-off odors-per-KC flags (0.200-0.212 against 0.196).
- **Done.** Tom's decision (CLAUDE.md, "The main goal for new experiments"; notes/response_criterion_decision.md): the data side in
  baseline-noise units (threshold 1.97 noise SDs). The code (cbdebc2: 54 `noise_*` registry entries, `apl_effect_scores.py`'s new
  `realism_flag` with the old one kept as `realism_flag_zcrit`, E73's fixed-unit columns, the progress plot's filter; 5100063: E10's
  `--band-units noise`), by an Opus code subagent of hal's session. The archive re-scored (results/apl_effect/by_variant.csv, 2026-11-10
  16:34): 423 of 916 variants flagged under the new rule against 604 under the old (24 newly flagged, 205 cleared); 3 of the summary's 14
  best-models columns against 10. The summary's footnote on baseline-noise units, the band paragraph, its table and the best-models
  realism rows (e83c8bb). E73's note reports both rules.
- **Remains.** The summary template's other band rows and prose (rows that pair the re-scored, noise-unit distances with the z-unit
  bands, and experiments' own z-unit tables; a planning subagent of hal's is on it), then a rebuild; earlier notes annotated (not
  rerun) where a flag changes (E44, E45, E64, E66, E67, E69, E70, E71, E72, E74, E75 Realism sections); whether the response rate ρ
  band is re-anchored on the reference model's ρ in noise units (0.55, giving 0.45 instead of 0.37), Tom's call. Hosts running
  build_summary.py or apl_effect_scores.py from code older than cbdebc2 rewrite by_variant.csv with the old flags (chat 16:39).

### The KC data's amplitudes and response criterion are relative to each cell's whole-recording spread (found 2026-11-10, hal's planning subagent; confirmed by Yang)
- **What.** Yang's per-trial KC values (files suffixed `zscored_full`) look z-scored per cell over the whole recording, not against a
  pre-odor baseline: in every one of the 19 APL16 recordings the per-KC mean of squared trial values has a sharp upper edge (99.9th
  percentile within 0.6 of the maximum; maxima 3.64 to 4.99 in control flies, 4.90 to 5.83 with APL output blocked (TNT)), the largest
  trial mean is 7.3 to 7.4 in both genotypes, and responders' mean amplitude is about 2 z whether a KC responds to one odor or to eleven;
  the low-concentration panel (diag_mix) shows the same edge. An inference from the values (read-only probes, nothing saved); notes/
  yang-summary.md lists the z-score's definition as unknown.
- **Affects (if it holds).** Every comparison of amplitudes with the model (E68's z conversion and its amplitude statistics; E69's APL-off
  amplitudes and the "strong tail" readings; E65's calibration targets), the data's response criterion (trial mean above 1 z) against the
  model's two-spike criterion, so the response rates, their ratio, the per-odor rates and the odors-per-KC distributions (all four realism
  bands, every fit's operating point), and the low-concentration fractions (E49, E72's part F).
- **Done.** The probes (in the E73 entry of experiment_priorities.md); confirmed by Yang (2026-11-10 10:00, in viki's session) and from her
  pipeline: each KC's suite2p Fc = F - 0.7 Fneu is z-scored over the whole recording (~7,194 volumes), and a trial's response is the mean z
  2-8 s after onset minus the mean over the 5 s before (add_to_reports/yang_response_definition.md, viki; a z unit is 1.4-3.0 baseline-noise
  SDs, rising with the KC's breadth). The per-KC normalization of the recordings (E73) is claimed by hal to read every archived model the
  same way (part B).
- **Done (E73, hal, 2026-11-10).** Parts A to C (E73_per_kc_normalization.md, page https://claude.ai/artifact/5gnZXoz2fkK3oXrUnQZWQU):
  every archived variant read through the per-KC session z-score (results/E73_per_kc_normalization/by_variant.csv); the best family
  re-fitted under that reading; the bands' criterion decided by Tom (the entry above). viki's per-KC scales
  (notes/yang_response_normalization/kc_scales.csv) give every KC's responses in baseline-noise units.
- **Remains.** Annotating the affected outputs (E65, E68, E69 amplitudes; E49 and E72's part F fractions, which E73 read within their own
  panel's session) with E73's readings; whether the main target's distances should also be taken in baseline-noise units (E73's next
  step 4).

### Experiment provenance records the checkout at the end of a run, not the code that ran (found 2026-11-07, atlas)

- **What.** Several experiment scripts call `model_provenance()` (al_analysis commit, dirty flag and
  diff) when they write provenance.json at the end of the run. A `git pull` while the run goes on
  (routine with several hosts pushing) makes the file name a later commit than the one whose code was
  imported. Found in the oracle-gain decomposition (E39): its two provenance files named bfd49c6 and
  bf314a6 (dirty) for runs that imported b431aa1 and bfd49c6.
- **Affects.** Only the recorded al_analysis commit of runs during which the checkout moved; the
  olfsysm build info is baked into the extension and is exact. No statistic changes.
- **Done.** E39's provenance files carry a `code_that_ran` annotation with the commits that ran;
  E39_gain_decomposition.py now records provenance at the start.
- **Done (viki).** E38's run (18:22-18:46) pulled at 18:24, so its provenance.json names 3160ed6 for code
  that ran from 7974f21 (the commits between touch only CLAUDE.md and join_queue.sh); annotated in the
  archive README and results/E38_hallem_input/PROVENANCE_NOTE.md, and E38_hallem_input.py now records
  provenance at the start.
- **Done (hal).** E37's stage 0 (18:54-19:26) pulled at 19:04 and 19:12, so its variants record
  7c61cd4, bfd49c6 and 00bbfa1 for code that ran from 7c61cd4 (the commits between touch only
  CLAUDE.md, the summary template and the progress plot); annotated in
  results/E37_spike_only_apl/PROVENANCE_NOTE.md, and E37_spike_only_apl.py now records provenance at
  the start. E37's stage-1 fits ran with no pull and are exact.
- **Remains.** The same fix (record at start) in the other scripts that call `model_provenance` at the
  end (grep `model_provenance(orn_deltas`); until then, avoid pulling in a checkout with a run going.

### The 16-odor ORN reference matrix counts unimaged glomeruli as zero responses (found 2026-11-06)

- **What.** `kc_remapping_harness.orn16_consensus_pairs()` reads `apl_pair_analysis/
  orn16_consensus_corrdist.csv`, which the original report's analysis (`apl_pair_analysis/
  apl_analysis.py`, section C) computed as 1 - Pearson r across glomeruli from
  `al_analysis/data/sent_to_vlad/2024-06-21/consensus_mean.csv`. It took the union of the megamat
  and validation2 panels' glomeruli (40 responding to any of the 16 odors), and the consensus file
  has 0, not NaN, where a panel did not image a glomerulus: DL4 and VM4 for the 12 megamat odors,
  VA4 and VL1 for paa, -aPine, euc and B-myr. Unmeasured glomeruli therefore count as zero
  responses. Against olfdb dataset 1 (Tom's megamat ORN data) the 66 megamat pairs differ by up to
  0.067 (mean 0.025); restricted to the 38 glomeruli imaged in megamat flies the same consensus
  means agree within 0.007.
- **Affects.** Everything that uses the 16-odor ORN reference: the ρ of the KC matrices with the
  ORNs (background figure and table of the summary report; data 0.65 / 0.75), `compare_on_off`'s
  ORN statistics (spearman_orn_on / _off, gap_closed, the linear-ORN residual statistics) and
  `compare_model_to_data`'s deviation-from-ORN statistics ("vs control (dev)" columns) in every
  experiment's stats table. Not affected: the model's ORN input (`apl16_orn_deltas()`, the 36
  glomeruli both panels imaged, no zero-filling), the joint-fit losses, and the 66-pair
  structure-beyond-ORN statistics (Remy's per-fly matrix of Tom's megamat data, `orn_remy`).
- **Done.** `orn16_distances_from_olfdb.py` rebuilds the matrix from olfdb (datasets 1 and 4) over
  the 36 shared glomeruli: within-panel pairs as the mean over flies of per-fly 1 - r
  (`al_util.mean_of_fly_corrs`, which reproduces `orn_remy` exactly on dataset 1), cross-panel pairs
  as 1 - r of fly-mean vectors (no fly saw both panels). Written to the shared directory's
  `data/model_outputs/inputs/orn16_corrdist_olfdb_36glom{.csv,_pairs.csv,.json}`. Effect on the
  headline numbers is small: ρ of the KCs with the ORNs 0.655 to 0.648 (control flies), 0.748 to
  0.737 (APL output blocked (TNT)); individual pair distances move by 0.05-0.07 on average, up to
  0.15.
- **Done (2026-11-06, later).** `orn16_consensus_pairs()` now reads the rebuilt matrix (the old one
  stays available as `orn16_consensus_pairs_zero_filled()`); the summary's background figures,
  data table (ORN column: mean distance 0.67 -> 0.73, range 0.05-1.13 -> 0.11-1.16, pairs > 0.8
  apart 47 -> 52, within-group 0.28 -> 0.35, fruity x small-ester 0.51 -> 0.56; ρ with the ORNs
  0.65 / 0.75 -> 0.65 / 0.74) and text are regenerated.
- **Remains.** The ORN-based columns already written in earlier experiments' stats tables
  (spearman_orn_on / _off, gap_closed, the *_dev columns) were computed with the old matrix; re-run
  their scoring from the archived outputs or annotate the tables. Scripts re-run from now on use
  the new matrix.

### The data's APL-off / APL-on response-rate ratio "1.9" was never computed (found 2026-11-06)

- **What.** First typed as a literal "(data 1.9)" in `E8_kc_kc_inhibition.py`'s figure (11-04),
  probably the model base's own 1.9 "by construction"; later labelled "by mean rate" and copied
  into many notes and the summary as "1.9–2.2". The computed values are 2.20 (fraction of KCs
  responding) and 1.87 (mean response size, negatives set to 0); `summary_background_figure.py`
  `rate_ratios()`. Trace: the shared directory's `notes/report_value_provenance.md`.
- **Affects.** Report text and tables (the summary now reads "1.9 (mean response ratio) / 2.2");
  per-experiment notes E8, E10, E19, E25, E28, E30, E33 still say "1.9–2.2" or "1.9 by mean rate".
  The joint fits' loss used 2.2 throughout, so no fit is affected.
- **Done (2026-11-07, hal).** The value registry exists (data_targets.py; scripts read it with
  `H.data_target`). The E8 and E19 figures draw the data's ratio from it (2.2, fraction of KCs
  responding; redrawn with `--figure-only`), and the E8 and E19 notes and pages say 2.2 with a dated
  correction note (pages republished, reports/ copies updated).
- **Remains.** Notes E10, E25, E28, E30, E33 still say "1.9–2.2" or "1.9 by mean rate" (annotate
  or leave).

### Hand-rounded numbers in the summary report differ from their sources (found 2026-11-06)

- **What.** Building the summary from the template (reports/summary_template.md, fill_report.py
  --compare) found the typed values that differ from what their sources give. Pattern: values
  first printed to 3 decimals and then rounded again by hand to 2 (0.8245 -> 0.825 -> 0.83).
  KC data: the wild-type structure target 0.83 (is 0.82, three tables); the smallest pair distance
  with APL output blocked 0.19 (is 0.18). Models: the KC-spike-driven no-gain fit's ρ on
  odor-odor distances 0.57 (0.56); the breadth-2 joint fit's APL-on block excess −0.16 (−0.15);
  breadth 3 with KC–KC 100: response rate ρ 0.73 (0.72) and structure beyond the ORN input +0.04
  (+0.03). The response-rate ρ ceiling was given as one value "~0.97"; it is 0.98 for control
  flies and 0.97 for APL output blocked (TNT). Full list: the shared directory's
  results/templated_summary_preview/comparison.md.
- **Related.** `E10_joint_fit.py` hard-codes the wild-type structure target as 0.83
  (`WT_DEV_TARGET`; data 0.8245); every fit's wt term is off by about 0.007, uniformly, so fit
  rankings are unaffected. The background table labels the 0.98 / 0.97 "measurement ceiling (ρ
  between halves of the recordings)"; the ρ between halves is 0.91 / 0.90, and 0.98 / 0.97 is its
  Spearman-Brown-corrected square root (the expected ρ of a noise-free model).
- **Done (2026-11-06 14:08).** The published summary is generated from the template
  (build_summary.py): the values above and the ceiling label / definition are corrected there.
- **Done (2026-11-07, hal, between experiments).** E10_joint_fit.py reads its five data targets
  from the registry (structure vs wild type 0.8245, natural-source alignment 0.3612, block 0.0554,
  isotonic R² 0.9391, excess difference 0.0237; were 0.83, 0.36, 0.055, 0.94, 0.02) and records them
  and the registry's sha256 in each fit's provenance, so joint fits from now on use a loss that
  differs from the earlier fits' by at most 0.006 in any target. summary_progress_plot.py reads its
  six targets from the registry (the APL-on response rate ρ target was 0.97, the APL-off ceiling;
  it is 0.98; the wild-type structure 0.83 is 0.82). In the template, the progress-plot paragraph
  and 29 other KC-data values (the wild-type structure "0.83" in four places, the natural-source
  alignment 0.44 / 0.36, the block, the ceilings) are placeholders now; summary republished.
- **Remains.** Other KC-data columns in the template's tables are still typed (not all checked
  against their sources); the per-experiment notes keep their hand-typed values.

## Resolved

### E78's gate.csv gave C2's readout loss as the start's for the points on P1 and the firing-ceiling fit (found 2026-11-11 02:45, hal's session)
- **What.** In results/E78_claw_knee/gate.csv (the high per-claw knee without a dead zone, E78), the column `cal_loss_start` held C2's
  readout loss (0.0561) for the points repeated on P1 and on E69's fit 1, while condition (ii)'s limit (`loss_limit`) used each start's own
  loss (0.0638 + 0.004 on P1, 0.0567 + 0.006 on fit 1); `ester_error_start` and `six_effect_start` likewise held C2's values, which
  conditions (i), (iii) and (iv) use for every start, as planned at 21:39.
- **Touches.** gate.csv and gate.json only. comparison_table.md printed `loss_limit`, so the limits shown and every pass / fail were right;
  nothing passed (i) on any start.
- **Done.** `gate_point` takes the start's own identity row: `*_start` now hold the point's own start, `*_ref` C2's reference, and
  `ester_reduction_vs_start` / `six_raise_vs_start` read (i) like for like against the start (not gated); the comparison table prints that
  column. The gate (both runs) and the tables were regenerated 2026-11-11 02:50 (al_analysis, with this entry).

### E76's note gave three fit-start parameters from the wrong rows (found 2026-11-10 23:40, atlas's planning subagent)
- **What.** In E76_data_side_target.md (the data side of the main target), the fits table's row "somatic threshold / APL→KC /
  APL→bouton / KC–KC" gave E69's fit 2 (the start of fits 0 and 1) APL→bouton 0.27 and KC–KC 20.8, which are fit 0's values; E69's
  fits.csv and the first evaluation of fit 0's history give 0.245 and 21.2. It gave E69's fit 3 (the start of fit 2) KC–KC 10.9, the
  value of that fit's own start, not its optimum (12.2); the Realism section repeated 20.8 / 10.9 as the starts.
- **Touches.** The note and its page only; the summary's passage reads its parameters from E76's fits.csv; chat and E76's Status
  compared fit 1 with fit 0 (whose values were right).
- **Done.** The note is corrected (dated line in its header), its page rebuilt and republished, reports/E76_data_side_target.html
  saved again (2026-11-10 23:50).

### The summary's low-concentration table read the wrong noisy fraction (found 2026-11-10, viki's planning subagent)
- **What.** In "The per-claw threshold at low concentration" the summary template fills the columns "APL on, through the matched noise"
  from E49's `fractions.csv` column `frac_trial` (each trial's response fraction, averaged). E49's note and that subsection's prose describe
  the recordings' rule, a mean over the three trials of at least one spike, which is `frac_count1`; with `frac_trial` the noisy columns
  barely differ from the noiseless ones (e.g. 2h @ -7, reference model: 0.0137 against 0.0139).
- **Touches.** That table of the summary (six rows, two models); nothing else reads `frac_trial` (the APL-effect fit on re-drawn wirings
  uses `frac_count1`).
- **Done.** E49's own comparison table (E49_low_conc_kc.py) prints `frac_count1` in that column, and the subsection's prose ("roughly
  doubles") describes it, so the template's twelve placeholders now read `frac_count1` (al_analysis, with this entry); the summary
  rebuilt and saved 2026-11-10.

### The progress plot counted re-drawn wirings as models (found 2026-11-10, viki)
- **What.** `summary_progress_plot.py` meant to leave out random, shuffled and re-drawn wirings (its comment and CLAUDE.md), but its pattern
  caught only `_seed<N>`; the joint fit's re-drawn wirings are tagged `wiring-<method>-seed<N>_...` (and `thr40_wiring-...`), so the
  re-drawn, shuffled, lesioned and transplanted wirings of E30, E33, E42, E46, E47, E50, E54, E59 and E72 were counted, and some were the
  running best (E30's draws on several statistics; E72's P3 would have been the best ρ on the per-pair APL effect).
- **Touches.** The progress plot and its table of best models per statistic on the summary page, since 2026-11-06.
- **Done.** The pattern also matches `wiring-` and E72's `popfit_<P|M><n>` (al_analysis, with the E72 summary passage); the summary
  rebuilt and saved 2026-11-10 (E30's draw points gone, e.g. the best ρ on odor-odor distances with APL off 0.771 -> 0.769).

### The claw-count thresholds of the KC heterogeneity experiment (E5) were not applied (found 2026-11-10, atlas's planning subagent)
- **What.** E5_kc_heterogeneity.py's `thr_nclaws_0.5` / `thr_nclaws_1` variants pass a per-KC vector as `fixed_thr` to the simulator of
  `make_fixed_param_kc_simulator`, whose `simulate` computes `thr = params['fixed_thr'] + spont_in * 2.0` with `spont_in` a column
  (checked in al_analysis/mb_model.py by atlas, 2026-11-10 04:55): the vector broadcasts to a KC × KC matrix, and by the archived counts every
  KC got the first KC's value. Evidence (the planning subagent's read-only probe of the archive): against the base, 88 and 181 of the 27,712
  spike counts differ with APL on (190 and 377 with APL off), none by more than one spike, although the nominal thresholds have a coefficient
  of variation of 0.22 and 0.41; the first KC's value times the bisected scale is 169.9 and 169.5 against the base's common 170.3, which is the
  whole change; per-type response rates are identical to the base's. hal's probe (notes/E5_hal_probe_claw_dynamics.md, "thr ∝ claws^k") used
  the same simulator and reports statistics identical to the base to three decimals. `kc_thr_factor` (reshaped to `spont_in`'s shape) is the
  working route.
- **Affects.** E5's result 1 and interpretation as far as they concern per-KC thresholds by claw count (the per-type thresholds, set inside
  `fit_mb_model`, are not affected); hal's probe note; the two archived variants in results/E5_kc_heterogeneity/ and
  data/model_outputs/E5_kc_heterogeneity/ (they are the base with a common threshold 0.2 to 0.5 % lower); the summary's "What did not matter"
  bullet on KC heterogeneity. Nothing that uses `kc_thr_factor` (the trial-noise jitter, E43 onwards).
- **Done.** Confirmed by E71's code subagent (2026-11-10 05:59): in the claw base's `thr_nclaws_1`, `simulate` set `rv.kc.thr` to a
  1732 × 1732 matrix whose column 0 (what olfsysm reads) is `thr[0] + 2·spont_in`, so every KC had the first KC's `fixed_thr` (253.16, a
  5-claw KC, against the intended mean 288.44, SD 117.6); re-running the call reproduces the archived counts exactly (0 of 27,712 differ, APL
  on and off) and equals a scalar run at `thr[0]`; the intended thresholds through `kc_thr_factor` change 3,673 of the 27,712 APL-on counts by
  up to 52 spikes (APL-on rate 0.0531 against 0.0507). Archived variants against E5's base (counts differing, each by at most one spike, APL
  on / off): claw thr_nclaws_0.5 53 / 131, thr_nclaws_1 99 / 254; claw-dynamics thr_nclaws_0.5 88 / 190, thr_nclaws_1 181 / 377. The guard
  (`simulate` refuses a non-scalar `fixed_thr`) and its test: al_analysis 1d7c5e5 (E5_kc_heterogeneity.py, lines 144 and 202, would now
  raise; no other script passes a vector `fixed_thr`). Annotated 2026-11-10 06:04 by atlas: E5's note (a correction at the top and in result
  1), its page (v2, https://claude.ai/artifact/1cZNDeiWRca44KMRJGURaM, and reports/ copy), hal's probe note, the archive README of
  data/model_outputs/E5_kc_heterogeneity/; the summary's KC-heterogeneity bullet corrected (template 5d6af1b, summary v110). The
  input-scaled KC threshold experiment (E71) runs the test.
- **Remains.** The info.json `thr_mean`, `thr_sd` and `thr_mean_by_type` of the 20 archived thr_nclaws files and the matching rows of
  results/E5_kc_heterogeneity/ (E5_summary*.csv, model_pair_distances_long.csv, blocks_* and per_odor_* thr_nclaws files) describe the
  intended vector; they are marked only through the archive README. Move to Resolved once E71 reports what such thresholds do.
- **Resolved 2026-11-10 12:35 (atlas).** The input-scaled KC threshold experiment (E71; note E71_input_scaled_threshold.md, page
  https://claude.ai/artifact/Y2Ki5cgRNKS4cA1U9Jb3ft) ran the test properly: thresholds scaled to claw input change the representation
  a lot (the release by claw count reverses) but cost the APL effect in every family. The 20 archived thr_nclaws files and E5's results
  rows still describe the intended vector; they are marked through the archive README, E5's note and page.

### The bounded-input oracle's (E64) second pass searched doubly scaled inputs (found 2026-11-09, hal, from a Fable planning subagent)

- **What.** `E64_input_oracle.py`'s `search()` built each generation's candidates as `candidate_orn(orn_now, ...)`. On an odor's second
  search `orn_now` already held the lab's response times the first pass's factors, and the candidates were factor vectors around
  those same factors, so every pass-2 candidate was the lab's response times f1 times a vector near f1 (effective factors 0.25 to 4,
  outside the bounds), while the input recorded at the end of the pass was the lab's response times the winning vector, which was
  never evaluated. Seen in search_history.csv (pass-2 offspring scores far below the parent's and falling as sigma shrinks) and in
  the whole-panel re-simulation (pass 1 and the controls reproduce the search's incumbent to six decimals, pass 2 does not; 2-but's
  response fractions 0.079 / 0.154, outside the ±15 % band). Pass 1 and the controls are unaffected (an odor's first search starts
  from the lab's column).
- **Affects.** E64's pass-2 rows (I_searched_pass2 in stats_variants.csv, apl_effect.csv, model_pair_distances_long.csv and
  per_odor_response_fractions.csv; per_odor_gains.csv pass 2; factors_weak_pass2.csv, orn_input_searched_pass2.csv; the archive's
  I_searched_pass2 outputs and params row; results/apl_effect/by_variant.csv's E64 rows), and E64's note and page as first published
  (they reported pass 2 as the result: 0.70 / 0.78).
- **Done.** The call is `candidate_orn(orn, ...)` now (this commit); E64's result is pass 1 (the note, page, Completed entry and summary
  use it); the pass-2 rows are removed from results/E64_input_oracle/ (stats_variants, apl_effect, model_pair_distances_long, per_odor_gains,
  per_odor_response_fractions; the tables as first written and the pass-2 factor and input files kept in its invalid_pass2/
  folder with a README), the archive's pass-2 outputs renamed *.parquet.invalid and its params row removed (backup in
  params_backup/), the tables and figure regenerated (E64_input_oracle.py --summary), the note and page corrected (version 3), the
  Completed entry annotated, the summary built from pass 1 (version 100), results/apl_effect/by_variant.csv rebuilt without it;
  summary_progress_plot.py now leaves E62 and E64 out as oracles. A fixed second pass was not rerun (it would refine paa, B-myr and
  2-but given the others' searched input; about 25 min at 16 threads).

### A fit seeded with `--init-from` does not start at its seed's optimum and runs with a different bouton->APL / KC->APL input balance (found 2026-11-08, atlas)

- **What.** `init_from_fit` in E10_joint_fit.py builds the starting point from the seed row's factors
  relative to the base (threshold, APL, APL->bouton) but resets the bouton->APL input scale
  (`wPNAPL_scale`, tied, never fitted) to the base's value while the KC->APL scale gets the seed's APL
  factor. The tie keeps every tied scale at its starting ratio to the APL->KC scale for the whole fit,
  so a chained fit (i) does not start at its seed's optimum (E44's 40 Hz reference model at 6.8 %
  starts at block +0.050 and APL-on response rate 0.070 instead of its fitted +0.054 and 0.068) and
  (ii) runs with a bouton->APL vs KC->APL input balance (wPNAPL / wKCAPL) that differs from its
  seed's: 5.57 for E44's 40 Hz fit vs 4.23 for its +0.072-target re-fit in E48; 10.58 vs 7.54 at
  70 Hz; 7.24 for E45's eight at 1.25 at 70 Hz vs 2.57 for its +0.072 and +0.10 re-fits (the tuned
  base: 4.45). Found by the code subagent writing the noise-aware loss (E51).
- **Affects.** Every fit started with `--init-from`: E48's re-fits (from E44 / E45's optima), E45's
  gain fits (from the nearest 5 % optima; their `initial` in provenance has the base's wPNAPL_scale),
  and the re-drawn-wiring fits of E46 and E47 (from E44 / E45's optima; viki, 2026-11-08 18:39: each
  draw's fit was compared with the real wiring's fit after both converged in their own runs, so their
  conclusions stand, but the notes need the annotation).
  The block target through the noise (E48) compared each re-fit with a counterpart that was meant to
  differ only in the target; they also differ in this unfitted balance, so the "raising the target
  cannot raise the block" reading rests on a confounded comparison. Comparisons across fits of the
  family that were seeded differently (E45's gain 1.25 against E44's gain 1.5, which started from
  E19's grid rows) differ in it too. No statistic of any single fit is wrong: each fit is what its
  recorded `initial` and tie say it is (provenance records `initial`, not the seed tag).
  Also every re-fit on a re-drawn or shuffled wiring chained from E44 / E45's optima (E46's draw-2 fits, E47's part B,
  E50's draws): their class comparisons are like-for-like (all chained the same way), but their comparison with the
  real-wiring fit carries the balance difference besides the wiring (noted in E47's note; E46's and E50's notes to follow).
- **Done.** `--init-from-exact` (al_analysis 8b3820d) starts at the seed's absolute parameters
  (checked against the row at the first evaluation); the noise-robust block's fits (E51) use it. The
  default `--init-from` is unchanged.
- **Done.** E48's note and page and the summary template's block-target paragraph carry the caveat
  (atlas, 2026-11-08 18:39; template 23977e5).
- **Done.** E10_joint_fit.py's provenance now always records the seed fit (`init_from`,
  `init_from_experiment`) and the start mode (`init_from_exact`).
- **Done.** E46's note carries the caveat (viki, 2026-11-08 18:41; its caveats list; page republished from one of Tom's hosts).
- **Done.** E47's note carries the caveat (hal, 2026-11-08; its caveats list, on the part B refits; page version 4).
- **Done.** E50's note carries the caveat (hal, 2026-11-09 02:16; its caveats list, second bullet; page and reports copy built from it).
- **Done.** E45's note carries the caveat (atlas, 2026-11-09 05:10; its caveats list; page version 4, reports copy updated).
- **Resolved.** E48's three re-fits repeated with `--init-from-exact` (tags `..._cap3_exact` in E48's fits.csv; the fits
  2026-11-08 18:49-23:02, their pipeline runs 2026-11-09 00:51-07:46, atlas) give the same answer as the warm-start re-fits
  (deterministic block +0.002 to +0.007 against their seeds, through the noise -0.002 to +0.002); E48's note carries the
  addendum and the summary's block-target passage the exact-start rows (atlas, 2026-11-09 07:50), and every affected note is
  annotated (E45, E46, E47, E48, E50).

### Trial-noise pipeline runs described as calibrated on 6 trials were calibrated on 3 (found 2026-11-09, hal)

- **What.** `E43_model_trial_noise.py --cal-trials` is part A's option (the PN-rate-noise calibration);
  parts B / C calibrate on their threshold-tuning trials, `--tune-trials` (default 3), and ignored
  `--cal-trials`. Every part-B run that passed `--cal-trials 6` (the noise-robust block's part C: hal's
  run on E44's 40 Hz model, 18:43-00:57; atlas's run on viki's eight at 1.25 noise-aware fit, 23:05-00:50)
  calibrated on 3 trials, i.e. with E48's settings exactly. hal noticed because its run's rows equal E48's
  rows for the same model to six digits; the runs' provenance records `tune_trials` 3 throughout.
- **Affects.** Only descriptions: the E51 entry's design (C) and Status lines, atlas's chat posts of
  2026-11-09 00:52 and 01:03 ("6 calibration trials"), the comments in the run scripts, and the E48 note's
  next-steps line recommending `--cal-trials 6`. No number is wrong, and the runs compare directly with
  E48's partB rows (which therefore serve as the deterministic counterparts in E51's comparison).
- **Done.** The pipeline refuses `--cal-trials` in parts B / C unless it equals `--tune-trials`, and its help
  says which option is which; E51_noise_robust_block.py's docstring and E51's run scripts say 3 tuning
  trials; the E51 entry's text and a chat correction (atlas, 2026-11-09 ~02:40).
- **Resolved.** E48's note names `--tune-trials 6` in its next steps and says in its addendum which option the
  pipeline runs' calibration follows (atlas, 2026-11-09 07:50; page republished, reports copy updated).

### The KC->APL drive per spike is proportional to the time step in spike-driven APL models (found 2026-11-07, viki)

- **What.** In olfsysm's KC layer a KC spike adds `wKCAPL * 1e4` to dIs/dt for one step, so the
  increment of the KC->APL current per spike is `wKCAPL * 1e4 * dt / tau_apl2kc`: it halves when
  the step halves (the KC-KC term adds `1/kc_kc_tau` per spike and is dt-invariant; the
  claw-driven and bouton-driven APL inputs are rates and dt-invariant). A step-halving test at
  fixed parameters (numerics_dt_check.py; results/numerics_dt_check/; reports/
  diffeq_stability_and_numerics.html) shows the spike-driven 70 Hz base's APL peak falling from
  55.5 to 36.9 to 25.2 and its APL-on response rate rising from 5.2 % to 6.0 % to 6.7 % at
  0.5 / 0.25 / 0.125 ms, with 14-22 % of its responses flipping, while the claw-driven base
  converges (3-4 % of responses flip, every statistic within 0.01).
- **Affects.** Every spike-driven APL model (E35's spike variants, E36, E37, E40's fits seeded from
  E36's optima): their tuned wKCAPL scales are dt = 0.5 ms numbers, the "x baseline" factors
  that compare them with claw-driven scales are not like for like, and the spike-driven /
  bouton-driven balance of APL's input is dt-specific. Not affected: the odor-odor geometry
  (rho with the 0.5 ms matrices 0.998 across steps; the agreement with both KC matrices within
  0.002), the claw-driven models, and the APL-off condition.
- **Resolved.** olfsysm ae6869c (2026-11-08, hal) scales each spike's drive by
  `p.kc.apl_spike_ref_dt / dt` (`apl_spike_ref_dt` 0.5 ms by default, the default step; 0
  restores the old drive), for the single APL and for APL units, so a spike adds the same charge
  to the KC->APL current at any step and outputs at 0.5 ms are bit-identical: no stored result
  changes and nothing needs re-tuning (the tuned scales now mean the same at every step; the "x
  baseline" factors against claw-driven scales still compare a spike drive with a rate drive).
  Tested by test_mb_model.py::test_apl_spike_drive_dt_invariant (al_analysis 7391c6a). The time-step
  check re-run with the fix (numerics_dt_check.py --out-name numerics_dt_check_spike_fix;
  results/numerics_dt_check_spike_fix/) gives the spike-driven 70 Hz base an APL peak of 55.5 /
  54.8 / 54.3 and an APL-on response rate of 5.2 % / 5.0 % / 5.0 % at 0.5 / 0.25 / 0.125 ms, with
  2.5-3.9 % of its responses flipping, the same as the claw-driven base (3.1-4.1 %); the 0.5 ms rows
  equal the first run's exactly. The E37 note's caveat and the summary's step footnote say so
  (summary template, this commit; E37 page republished).

### Two figures label the data's isotonic R² as 0.75 (found 2026-11-06)

- **What.** `E8_kc_kc_inhibition.py` and `E19_thr_breadth_kckc.py` draw a reference line
  "isotonic R2 on vs off (data 0.75)"; the data give 0.94. Unsourced like the 1.9 above.
- **Affects.** The E8 and E19 figures (results/ and their artifacts).
- **Resolved.** The literal was corrected in 4fe1fde and both figures redrawn on 2026-11-05; the
  E8 note and page still quoted the data's 0.75 in their text (calling the model's 0.88 "preserved",
  though it is below the data's 0.94) and were corrected on 2026-11-07, when both figures were
  redrawn again with every data reference read from the registry (pages republished).

### Artifact publishing unavailable on both hosts (found 2026-11-06 14:34)

- **What.** The claude-artifacts tools stopped working in hal's and atlas's sessions ("not found:
  the artifact tools on this server aren't available to this session"); they come back only after
  a Claude Code restart. Nothing was published or changed on the server.
- **Affects.** The published summary artifact (https://claude.ai/artifact/DCsKxo7Y2XT1MhiL4MfJoe)
  is the 14:08 version, without the footnote on how "structure beyond the ORN input" is computed
  (and anything added to the template since). The shared directory's reports/summary_2026-11-06.html
  (and current-summary.html) is kept current from build_summary.py's page, so it has the content
  the artifact would have (Tom: that suffices meanwhile).
- **Was to do.** At the start of the next session (once, without probing the tools repeatedly):
  tell Tom, and if the tools work, run build_summary.py, publish
  outputs/_artifacts/summary/index.html to the summary slug with the new title "Model experiments to
  improve fit to KC remapping, with APL on/off" (the artifact title is set at publish) and run
  save_report_copy.py. Until
  then, keep updating the reports/ copy with save_report_copy.py after each build.
- **Resolved** 2026-11-07 17:31 (hal): the tools work again in a new session; build_summary.py was
  re-run (356 values, lint clean), the page published to the same artifact (version 50, title
  "Model experiments to improve fit to KC remapping, with APL on/off") and copied to
  reports/summary_2026-11-06.html with save_report_copy.py. No other page lagged (every reports/
  row has its artifact).

### The archive's params.csv did not record what drives APL, the divisive flags or the fit bounds (found 2026-11-06, atlas)

- **What.** `archive_model_outputs.py` kept only the columns in its PARAM_COLS list, which lacked
  `apl_drive`, `pn_claw_to_apl`, `apl_inh_divisive`, `bouton_inh_divisive`, `base_params`,
  `init_params` and the bound settings (`aplpn_max`, `aplpn_fixed`, `aplkc_min`), and it copied
  only `provenance.json`, not the joint-fit scripts' per-fit `provenance_<tag>.json`.
- **Affected.** The shared archive (`data/model_outputs/`): E35's params.csv could not tell the
  spike-driven or divisive variants from the others (the final parameters alone do not say what
  drives APL or how it inhibits), E36 stage 2 likewise; E30 / E31 / E33 / E34 lacked their bounds;
  the joint-fit folders (E10, E25, the E28 joint-fit arm, E30, E31, E33, E34, E36 stage 2) lacked
  their per-fit provenance. No statistic or figure was affected.
- **Resolved** in 6ba1063: the columns are archived, the per-fit provenance files copied (also by
  `--params-only`), and the nine folders re-run with `--params-only` (old columns unchanged).

### `--aplpn-fixed` did not fix the APL->bouton scale in the first run of the bound control (found 2026-11-06, atlas)

- **What.** `optimize_mb_model_params(tie_apl_weights=True)` ties every APL scale, including the
  non-optimized bouton scales, to the APL->KC factor, so E31's "APL->bouton fixed at 1x" fit moved
  the bouton scale with the APL->KC scale (it ended at 2.4x).
- **Affected.** E31's first fixed-scale run only.
- **Resolved** in a7b611f (`tie_apl_weights` accepts the subset of scales to tie; `--aplpn-fixed`
  ties only the KC->APL scale); the fixed variant was re-run, and the mis-run is kept as the
  "tied to APL->KC" variant, annotated in E31_aplpn_bound.md and in its stats row.
