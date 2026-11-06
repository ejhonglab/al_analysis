# Unresolved issues

Errors and bugs that affect, could affect, or have affected analysis outputs in this folder
(statistics, figures, reports). One entry per issue: what is wrong, what it touches, what has been
done, what remains. Move an entry to "Resolved" (with the commit) once every affected output has
been regenerated or annotated. Keep entries short; long investigations go in the shared
directory's notes/ and are linked from here.

## Open

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
- **Remains.** Point `orn16_consensus_pairs()` (or a new harness function) at the rebuilt matrix,
  regenerate the summary's background figure / table, and decide whether to recompute the
  deviation-from-ORN columns of earlier experiments or annotate them.

### The data's APL-off / APL-on response-rate ratio "1.9" was never computed (found 2026-11-06)

- **What.** First typed as a literal "(data 1.9)" in `E8_kc_kc_inhibition.py`'s figure (11-04),
  probably the model base's own 1.9 "by construction"; later labelled "by mean rate" and copied
  into many notes and the summary as "1.9–2.2". The computed values are 2.20 (fraction of KCs
  responding) and 1.87 (mean response size, negatives set to 0); `summary_background_figure.py`
  `rate_ratios()`. Trace: the shared directory's `notes/report_value_provenance.md`.
- **Affects.** Report text and tables (the summary now reads "1.9 (mean response ratio) / 2.2");
  per-experiment notes E8, E10, E19, E25, E28, E30, E33 still say "1.9–2.2" or "1.9 by mean rate".
  The joint fits' loss used 2.2 throughout, so no fit is affected.
- **Remains.** Annotate or leave the old notes; the value-registry proposal in the provenance note.

### Two figures label the data's isotonic R² as 0.75 (found 2026-11-06)

- **What.** `E8_kc_kc_inhibition.py` and `E19_thr_breadth_kckc.py` draw a reference line
  "isotonic R2 on vs off (data 0.75)"; the data give 0.94. Unsourced like the 1.9 above.
- **Affects.** The E8 and E19 figures (results/ and their artifacts).
- **Remains.** Correct the literal and redraw when those figures are next regenerated.

## Resolved
