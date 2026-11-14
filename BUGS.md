# Bugs and questionable choices in the ROI response calculation

What is wrong, or open to question, in how al_analysis turns a movie and a `RoiSet.zip`
into each glomerulus' responses: `ij_traces` and `compute_trial_stats` in
`al_analysis/al_analysis.py`, with hong2p's `ijroi_masks`, `extract_traces_bool_masks` and
`rois2best_planes_only` (`hong2p/roi.py`). Written 2026-11-13. **Nothing below has been
changed here or in hong2p**: outputs are as they always were.

Found by reading the code (al_analysis 478b56b, hong2p ae2abda), then measured on three
flies of `2p_data/analysis_intermediates`: 2023-05-08/1 (5 recordings, 100 outlines),
2023-11-21/2 (6 recordings, 97 outlines) and 2024-01-05/1 (6 recordings, 117 outlines).
The measuring was done with hong-data's `app/extract.py` (the odor response database's
*Extract signals*), which reproduces this calculation exactly: for 9 recordings of 4 flies,
every trace in `full_traces`, every plane chosen in `best_plane_traces` and every response
in `ij_trial_df_cache` came out identical to the last bit (where it differs from
al_analysis otherwise, in labels and in what it takes and leaves undone, is the last
section here). The scripts that made the numbers
are `notes/trace_extraction_flaws.py` and `notes/trace_extraction_baseline.py` (finding 4)
in hong-data (a copy of this note is beside them).

## Summary

| # | What | Where | How much it matters | Basis |
|---|------|-------|---------------------|-------|
| 1 | A glomerulus' plane is chosen separately in each recording | `ij_traces` | 74–85% of multi-plane ROIs change plane between a fly's recordings; single ROIs' responses 0.33× to 1.75× | measured |
| 2 | The plane is chosen by one signed maximum | `ij_traces`: `roi_quality = trial_dff.max()` | one trial decides; an ROI that is only inhibited gets the plane that responds least | read in the code |
| 3 | Masks are half a pixel off ImageJ's, and depend on which way round an outline was drawn | hong2p `ijroi2mask` | 9% of an outline's pixels differ; responses over 0.2 differ by about 1% (median) | measured |
| 4 | F0 is the mean of very few time points (a deliberate choice, not a bug) | `delta_f_over_f`; imaging is triggered per trial | 2–5 volumes; no bias; another baseline could cut a response's s.d. between repeats by 11–16% at most, and only by assuming F0 steady from trial to trial, which it is not | measured |
| 5 | The ±peak statistic is biased away from zero | `sign_preserving_maxabs` | no odor: mean magnitude 0.063, against 0.037 for the mean of the same two time points | measured (one fly) |
| 6 | Two time points do not always hold the peak | `n_volumes_for_response = 2` | 86–95% of strong responses peak in them; for 4–10% they hold under 80% of the peak | measured |
| 7 | The plane is chosen by the responses then reported | `ij_traces` | adds 2–5% to an ROI's largest response | measured |
| 8 | No background is subtracted before dF/F, and F0 is not kept from zero | `delta_f_over_f` | not measured | read in the code |
| 9 | Odor abbreviations depend on what was loaded before on the computer | hong2p `olf.odor2abbrev` and its cache | 2 odors' labels differ from the YAMLs' own (`1-3ol` / `1-prop`, `paa` / `PAA`) | observed |
| 10 | Smaller: square frames assumed; an ROI named twice on a plane | hong2p `ijroi_masks`, `rois2best_planes_only` | not measured | read in the code |

Line numbers below are of the commits above.

## 1. A glomerulus' plane is chosen separately in each recording

`ij_traces` (`al_analysis.py:3239`) is called once per recording, and picks each ROI's best
plane from that recording's trials alone (`rois2best_planes_only(full_rois, roi_quality)`,
line 3287). So the same glomerulus is read on one plane in `diagnostics1` and maybe another
in `megamat1`: different pixels, at a different gain.

Of the ROIs drawn on several planes (about half: 27 to 34 a fly), 74 to 85% were read on
different planes in different recordings of the same fly, on up to 4 planes. Over all of
them the choice makes no difference in the median (responses on a recording's own plane are
1.00× those on the plane that is best over the whole fly), but for single ROIs they are
0.33× to 1.75×, and once of the other sign.

The code says so itself: the TODO above line 3287 ("refactor so all this calculation is
done across all recordings within each fly"), and the one in `process_recording` ("compare
best_plane_rois across recordings").

*A fix:* choose the plane once per fly, from the largest of each plane's `roi_quality` over
all of the fly's recordings. *Extract signals* offers this ("its best plane, the same one in
every recording of the fly").

## 2. The plane is chosen by one signed maximum

`roi_quality = trial_dff.max()` (line 3281) is the largest single response of a plane over
the recording's trials, with its sign.

- One trial decides, so one artefact (a motion jump, a bright neighbour on one trial) can
  choose the plane.
- For a glomerulus that is only ever inhibited, the largest response is the least negative
  one: its "best" plane is the one that responds least, or is noisiest.

Not measured. *A fix:* choose by something that many trials share and that does not care
about sign, such as the largest magnitude of the mean response per odor, or a plane's
response variance across odors over its baseline noise. Or do not choose: see 7.

## 3. Masks are not the pixels ImageJ shows inside an outline

hong2p's `ijroi2mask` (`roi.py:311`) asks matplotlib's `Path.contains_points` whether the
point (x, y) is in the polygon, for whole numbers x and y. ImageJ takes a pixel when its
centre, (x + 0.5, y + 0.5), is inside. Two consequences:

- **Every mask is that of the outline moved half a pixel right and down.**
- **Pixels whose corner is exactly on the outline go in or out by the direction the outline
  was drawn in.** matplotlib counts crossings of a ray, and which side of an edge a point on
  it falls depends on the edge's direction. ImageJ's ROIs here have points on quarter
  pixels, so many edges pass exactly through pixel corners. (It also depends on hong2p
  handing matplotlib the image's rows as x: with rows as y, 65 of a fly's 100 masks differ.)

Small examples: a square drawn over 3 × 3 pixels, from (2, 2) to (5, 5), comes out as
2 × 3 pixels, one row short and one column to the right. A diamond with corners at (4, 1),
(7, 4), (4, 7), (1, 4) is 14 pixels drawn one way round and 24 the other; its area is 18.

On the flies: 9% of an outline's pixels (median; up to 32%) are in one of hong2p's and
ImageJ's masks and not the other, and 64 to 67% of outlines would give another mask if drawn
the other way round (by 1 to 2 pixels). It matters little to the responses: with ImageJ's
masks they correlate with al_analysis' at r ≥ 0.997 in every recording, and responses over
0.2 differ by 1.0 to 1.4% in the median and 5 to 7.5% at the 95th percentile. The worst
differ by twice their size.

*A fix:* test pixel centres (`points + 0.5` in `ijroi2mask`), which also takes nearly every
pixel off the outlines. It changes every mask a little, so every cached trace. al-diag-seg
rasterizes the same outlines with scikit-image's `draw.polygon`, a third rule: if this is
fixed, make it the same there.

## 4. F0 is the mean of very few time points (a deliberate choice, and a small cost)

`delta_f_over_f` (`al_analysis.py:2374`) takes F0 as the mean of all of a trial's time
points before the odor (`n_volumes_for_baseline = None`, line 303). Imaging is triggered
by the stimulus, a block per trial (`stimulusTriggering="1"` in the recordings'
`Experiment.xml`), so those are all the time points there are next to the odor: 4 or 5
volumes in the 2023-05-08 fly (6 to 7.5 s, at 0.67 volumes/s), and 2 or 3 in the two later
flies (4.5 to 6.8 s, at 0.44 volumes/s). The time points before a trial's `start_frame` are
the end of the block before, across the time between trials when nothing was imaged.

This is not a bug. It is the same calculation for every trial, the first included; it
assumes nothing about how fast a response decays or how long imaging goes on after an odor;
and the response is itself of one or two volumes. It is here for what it costs, which is
noise and not bias, and that is small:

- **The noise it adds.** One time point's noise is 4 to 5.5% of F0 (median over ROIs). The
  variance of a response R read from one time point is about σ²(1 + (1 + R)² / n) with n
  time points in F0: 1.5 σ² for a small response at n = 2, 3 σ² at R = 1.
- **The most another baseline could gain.** With F0 replaced by the median of five
  neighbouring trials' F0, the s.d. of a response between an odor's repeats (responses over
  0.3; their median 0.56) falls from 0.069 to 0.120 (by fly) to 0.056 to 0.100: by 11 to
  16%. With each trial's own F0 still subtracted, and only the division by that median: 7
  to 8%. The rest of the variation between repeats is not the baseline's.
- **A trial's own F0 follows something real.** F0 moves between successive trials with 1.4
  to 1.6× the variance its noise alone would give (2 to 4× over a whole recording, whose
  last three trials' F0 are 0.91 to 1.08 of its first three's), so a baseline pooled over
  trials would trade this noise for a bias.
- **No bias.** After a response over 0.5, F0 over all of a trial's time points is within
  0.4% (median; +3% at the 90th percentile) of F0 over its last two. The last two time
  points of the block before are within 0.2% (median; 0.6% after a response over 0.5) of
  the next trial's F0, with a spread (4.9 to 6.6%) a little over what noise alone would give
  (4.2 to 5.0%): the response before has decayed by the end of its block, but assuming so
  buys only two more time points, and none for the first trial.

*If more precision is ever wanted:* image one or two more volumes before each odor (the
term is 1 / n). Nothing in analysis is worth doing. The 7 to 8% and 11 to 16% above come
from an F0 smoothed over neighbouring trials, which assumes F0's scale does not jump from
one block to the next. It can (the scope is off between blocks: drift in z, motion,
bleaching and its recovery), and it does: for a tenth of the ROIs, F0's variance between
successive trials is over 3× what its noise would give. Those medians are over ROIs whose
F0 is mostly steady; where it jumps, a smoothed F0 is wrong by the jump.

## 5. The ±peak statistic is biased away from zero

`sign_preserving_maxabs` (`al_util.py:182`, the default `response_stat_fn`) returns the
larger in magnitude of the first two dF/F values. Where there is no response, the larger of
two noise values is never near zero.

Taken of two time points with no odor (the last two before it, against those before them,
in the fly with enough of them: 2023-05-08/1), it has a mean magnitude of 0.063 and a
standard deviation of 0.090; the mean of the same two has 0.037 and 0.058. So a glomerulus
that does not respond is about 1.6× as far from zero as it need be, with either sign.

*A fix:* the mean of the two (the statistic before this one, still there for `-R`) keeps the
sign too. If the peak is what is wanted, report it together with a noise floor from the
same statistic on time points without odor.

## 6. Two time points do not always hold the peak

`n_volumes_for_response = 2` (line 298). Of responses whose |dF/F| passes 0.3 anywhere in
their trial, 86 to 95% peak in the first two time points from onset and 3 to 9% later than
the third; for 4 to 10% of them the first two hold under 80% of the peak.

The comment there asks for the window in seconds: two volumes are 3.0 s in the 2023-05-08
fly (0.67 volumes/s) and 4.5 s in the two later ones (0.44 volumes/s, 11 planes), by their
`Experiment.xml`. With the slower volumes, more peaks fall in the first time point.

## 7. The plane is chosen by the responses then reported

Choosing the plane with the largest response, then reporting that plane's responses,
favours whichever plane's noise went up. Measured by choosing on two of three repeats and
reading the third: an ROI's largest response on its chosen plane is 1.13 to 1.16× the mean
of its planes' on the trials it was chosen by, and 1.10 to 1.11× on the others. So 2 to 5%
of an ROI's largest response is the choosing. Small beside 1.

*A fix:* do not choose. *Extract signals* offers the mean over all of an ROI's voxels, and
its planes averaged with weights 1 / the variance of each plane's dF/F before the odors.

## 8. No background is subtracted before dF/F

F0 holds the detector's offset and out-of-focus light, so dF/F is smaller by that share,
differently in dim and bright glomeruli (the TODO at line 2486). F0 is not kept from being
near zero either (the commented-out `baseline + 10.` below it). Not measured: the movies do
not say what their dark level is (the lowest value in the first volumes of one fly's first
recording is 116, of the other two flies' 0).

## 9. Odor abbreviations depend on what was loaded before

`odor_lists_to_multiindex` abbreviates through `hong2p.olf.odor2abbrev`, which is the
hardcoded table plus a pickle of every abbreviation seen in a YAML on that computer
(`odor2abbrev_cache`), an existing abbreviation winning over a YAML's. So the labels in
`ij_trial_df_cache` depend on the history of the computer that made them. In 9 recordings,
two odors' labels were not the ones their own YAMLs give now: `1-3ol` for `1-prop`, and
`paa` for `PAA`. `my_abbrev2remy` (line 12336, which `convert_my_abbrevs_to_remy` applies)
changes exactly these later, under a TODO to correct `odor2abbrev` instead.

## 10. Smaller

- `ijroi_masks` asserts `x == y` (`roi.py:842`, "not tested in case x != y"): rows and
  columns have never been told apart by a test.
- Two ROIs of one name on the same plane count as two "planes" in `rois2best_planes_only`,
  and one of them is dropped.

## The measurements, by fly

| | 2023-05-08/1 | 2023-11-21/2 | 2024-01-05/1 |
|---|---|---|---|
| Recordings; outlines; median pixels an outline | 5; 100; 410 | 6; 97; 371 | 6; 117; 372 |
| Pixels in one of hong2p's and ImageJ's masks and not the other (median; most) | 9.1%; 23.0% | 8.9%; 31.1% | 9.3%; 32.3% |
| Outlines whose mask depends on the direction drawn (by how many pixels, median) | 65% (2) | 64% (1) | 67% (2) |
| Responses with ImageJ's masks against al_analysis': r, over recordings | 0.9987–0.9996 | 0.9967–0.9994 | 0.9983–0.9996 |
| … difference of responses over 0.2 (median; 95th percentile; most) | 1.0%; 5.2%; 203% | 1.2%; 6.4%; 201% | 1.4%; 7.5%; 198% |
| ROIs drawn on several planes, of all | 29 of 59 | 27 of 49 | 34 of 62 |
| … read on different planes in different recordings | 79% | 74% | 85% |
| … responses on the recording's plane over those on the fly's (median; range) | 1.00; −0.07 to 1.75 | 1.00; 0.37 to 1.33 | 1.00; 0.33 to 1.53 |
| Largest response on the chosen plane over the planes' mean: trials chosen by; others | 1.13; 1.11 | 1.16; 1.11 | 1.15; 1.10 |
| Peak at time point 0; 1; 2; later (responses passing 0.3) | 36%; 50%; 5%; 9% | 52%; 43%; 1%; 3% | 48%; 43%; 2%; 8% |
| First two time points hold under 80% of the peak | 10% | 4% | 8% |
| Volumes per second (planes) | 0.67 (8) | 0.44 (11) | 0.44 (11) |
| Time points before the odor | 4 or 5 | 2 or 3 | 2 or 3 |
| F0 over all of them against the last two, after a response over 0.5 (median; 90th percentile) | −0.2%; +2.8% | −0.3%; +2.8% | −0.4%; +3.2% |
| Noise of one time point, of F0 (median over ROIs) | 4.1% | 4.6% | 5.5% |
| Variance of F0 over what its noise would give: between successive trials; over a recording | 1.6×; 4.1× | 1.4×; 2.2× | 1.5×; 2.0× |
| The last two time points of the block before, against the next F0 (median; after a response over 0.5) | −0.1%; −0.5% | −0.1%; −0.6% | −0.2%; −0.6% |
| S.d. of a response between repeats: own F0; median F0 of 5 trials; own subtracted, median divided by | 0.069; 0.056; 0.060 | 0.102; 0.087; 0.093 | 0.120; 0.100; 0.103 |
| No odor: ±peak of two time points (s.d.; mean magnitude) against their mean | 0.090; 0.063 against 0.058; 0.037 | too few time points | too few time points |

## Where hong-data's Extract signals differs from al_analysis

With its default (each ROI's best plane, chosen in each recording), *Extract signals*
(`app/extract.py`) gives al_analysis' numbers: every trace, every plane chosen and every
response was identical, to the last bit, in the 9 recordings checked. It differs in what it
labels them with, in which recordings and ROIs it takes, and in stopping where
`ij_trial_df` does. So its tables can be put beside al_analysis' per recording, but they
are not al_analysis' final outputs.

### Labels

- **Odor abbreviations** are hong2p's hardcoded ones, else the abbreviation in the
  recording's own `trial_frames_and_odors.json`. al_analysis also uses hong2p's cache of
  every YAML loaded on its computer (finding 9), so two labels of the 9 recordings differ:
  `1-prop` for its `1-3ol`, and `PAA` for its `paa`.
- **A panel** is as `get_panel` names it, for the names it knows. For any other recording
  it is the folder's name less a number at its end, where `get_panel` gives None.
- **`is_pair`** is always False. al_analysis sets it for grids of two odors'
  concentrations (`is_pairgrid`).

### Which recordings and ROIs

- **A recording done again.** A folder `<name>_redo` is taken in place of `<name>`, whole.
  al_analysis processes both, and later blanks, odor by odor, all but the last recording
  of each of a panel's odors (`drop_redone_odors`), whatever the folders are called. So
  where a redo repeated only some odors, the first recording's other odors are missing
  here; and odors given again in a folder not named `_redo` are both kept here, as further
  repeats in the fly's `responses.csv`.
- **Every recording with a movie and its trials is taken**: each folder of the fly with
  `mocorr.tif` (else `flipped.tif`, else `raw.tif`, as al_analysis) and
  `trial_frames_and_odors.json`. What al_analysis leaves out is not left out: the flies
  and dates it is not run on, recordings of one plane, and those it skips for having no
  stimulus YAML or frames that could not be assigned to trials.
- **The ROIs' file** is the recording's `RoiSet.zip`, else one in the fly's folder.
  al_analysis keeps it in the fly's first (diagnostic) recording and links the other
  recordings to that; it does not read one in the fly's folder.
- **ROIs al_analysis' reader refuses are read**: rectangles, ovals and composite ROIs. An
  ROI on no plane is left out, saying so, where al_analysis tries to read a plane from
  its name.

### What is not done

- **Nothing after `ij_trial_df`**: merging ROIs across a fly's recordings
  (`merge_rois_across_recordings`), dropping uncertain ROIs where there is a certain one
  (`drop_superfluous_uncertain_rois`), blanking concentrations that are not a panel's final
  ones and data with the wrong solvent (`setnull_*`), keeping only consensus odors and
  glomeruli, and the means over trials and flies.
- **`responses_glomeruli.csv`** has the ROIs the database's uploads accept (those of its
  list of glomeruli, and names with `_t0`, `_t1`, ... where no other ROI is that
  glomerulus), which is not al_analysis' certain ROIs (`is_ijroi_certain`).
- **The fly's `responses.csv`** is a table al_analysis does not make: every recording's
  trials one under another, repeats counted through a panel's recordings. Each recording's
  own `responses.csv` is the one laid out as `ij_trial_df`.
- **Its settings are al_analysis' current defaults, and fixed**: `sign_preserving_maxabs`
  over 2 volumes, and all of a trial's time points before the odor as its baseline. Not
  the mean of `-R`, z-scored traces, a median baseline, or one baseline per odor.
- **suite2p's ROIs** are not extracted, only ImageJ's.

### By choice

- The three other ways to make one signal of an ROI's planes (its best plane of the whole
  fly, all of its voxels, its planes weighted by their noise) are not al_analysis'. The
  weighted one's `traces.csv` is of F relative to each plane's mean, not F.

### Not checked

- **Recordings with 8 or more time points before the odor.** F0 is summed there by numpy,
  and in al_analysis by pandas, which may not add in the same order, so a last digit could
  differ. Those checked had 2 to 5.
- **Movies that are not 16-bit integers** were checked against al_analysis' steps written
  out (hong-data's tests), not against its own outputs: all nine recordings were `uint16`.
