# Unresolved issues

Errors and bugs that affect, could affect, or have affected analysis outputs in this folder
(statistics, figures, reports). One entry per issue: what is wrong, what it touches, what has been
done, what remains. Move an entry to "Resolved" (with the commit) once every affected output has
been regenerated or annotated. Keep entries short; long investigations go in the shared
directory's notes/ and are linked from here.

## Open

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
- **Done.** Documented in the numerics report with the measured bound; nothing re-run.
- **Remains.** Make the spike impulse dt-invariant in olfsysm (add `wKCAPL / tau_apl2kc` per
  spike, or add to Is directly at the spike), re-tune the spike-driven bases and re-state their
  scales; until then, quote spike-driven scales with the step they were tuned at.

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
