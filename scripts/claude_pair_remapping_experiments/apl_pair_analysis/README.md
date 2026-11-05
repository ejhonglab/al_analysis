Analysis scripts and outputs behind the Nov 3, 2026 report
"apl_orn_to_kc_remapping_report.md" (made in a web Claude session).

`apl_analysis.py` reads `TNT3c_dist_apl16.csv` (control) and `TNT_dist_apl16.csv` (APL
output blocked) from the working directory. The originals Yang sent were not saved with
the rest; the copies here were regenerated (2026-11-04) from her per-recording data in
olfdb by `../yang_apl16_kc_data.py`, as 1 - cosine similarity over all KCs of a recording
(trial-mean responses), averaged over recordings, which matches the originals (as
preserved in `all120_pairs.csv` columns `ctrl` / `tnt`) to 4 decimals. Note the report
calls these 1 - Pearson r; they are cosine distances.
