
- `from_prat`
  - `2024-09-13`
    - `weight` # of T-bar-to-PSD connections between two neurons (same as for hemibrain
       input, though maybe called `x.weight` there). generally threshold at >= 5 (not
       sure if this is what I did previously)

  - `PNtoKC_connections_raw.xlsx` (currently under root `data/`. need to move to subdir
     here) was received 2023-08-08 on Slack. contains hemibrain data. his neuprint query
     already filtered stuff where `c.weight > 5` (well, according to the code snippet
     Pratyush gave me, but `df['c.weight'].min() == 4`, so maybe it was `>= 4`?)


- `sent_to_remy`
  - `2023-01-06`
    These 4 files (2 CSVs, and 2 pickle files with same data) should contain the data
    Remy used to generate the original ORN / PN correlation matrices in the preprint.

    The preprint was posted 2023-02-16 originally, and this was the only data I had
    seemed to send to Remy (on Slack) recently before this (on 2023-01-06). I didn't
    send anything to her via email during the same time frame.

  - `2023-10-29`
    - should be final megamat data (pebbled and GH146)
      (`pebbled_ij_certain-roi_stats.csv` here is exactly the same as
      `sent_to_anoop/v1/megamat_ij_certain-roi_stats.csv`)

    - sent on: Slack

  - `2024-01-12`
    - should be final validation2 panel data (just pebbled, we never did PN imaging for
      this panel)

    - sent on: Slack

  - `2025-03-18`
    - contains final megamat hemibrain model responses
      - wd20, but no connectome APL weights


- `sent_to_anoop`
  - `v1`
    - megamat and validation2 ORN data (should be pretty much, if not exactly, as
      final). non-consensus glomeruli may not have been dropped, or may have been
      handled differently than what I do now (e.g. for megamat data I sent to DePasquale
      lab people more recently)?

  - `v2`
     - model KC outputs that still had the offset in the dF/F -> est. spike delta fn,
       and thus still had unnecessarily high KC correlations. Anoop never really used.

  - `2024-05-16`
    - final megamat uniform model responses (binarized) (+ parameters & wPNKC)
      (these responses also sent to Remy on Slack 2025-02-05).

      `test_uniform_paper_repro` checks we can reproduce.

    - also contains megamat hemibrain responses, but those are probably not quite the
      final versions, as I believe the wd20 code came later (that included some
      influence of synapse counts, rather than just counting each PN-KC pair as a claw).

- `from_remy`
  - `megamat17`
    - downloaded from Dropbox folder `Remy/odor_space_collab/for_mendy` on 2024-07-05
      (though shouldn't have changed in a while, and should be same data Anoop is
      loading). only copied the KC soma subset of the data from that (large) Dropbox
      folder.

    - `per_fly`
      - copied + renamed from `for_mendy/data/megamat17`
      - used by:
        - `al_analysis.py`:
          - to make figs (#?) comparing model vs real KC correlations
          - any other figs? check.

        - this is also the subset of the Dropbox data that Anoop is loading

    - `respvec_concat`
      - copied from `for_mendy/concatenated_respvecs_by_imaging_type/kc_soma/megamat17/respvec_concat`

    - `README.html`
      - some of the folder / file structure descriptions may be wrong now, given my
        re-organization of the above


  - `response_rates`
    - was used (by `al_analysis.py`) to calculate new 0.0915 response rate target now
      used for all modelling (was previously using 0.1 target)

    - `refstim__ep_at_-3.0__median__0.120-0.130`
      - downloaded from Dropbox folder:
        `Remy/odor_space_collab/analysis_outputs/multistage/multiregion_data/response_breadth/by_trialavg_ref_stim/megamat`

      - received: 2024-04-04, via a Dropbox link on Slack (files may have been on
        Dropbox for longer)

    - `old_megamat_sparsities`
      - received: 2024-04-03, via Slack

      - only used to verify one of the files match the corresponding file from folder
        above (they did).


  - `2024-06-05`
    - odor metadata (full names, PubChem CIDs, etc) for megamat and validation2 panel
      odors
    - renamed `megamat17.xlsx` to `megamat.xlsx`


  - `2024-11-12`
    - old megamat data (from prior to final 4 flies I had been using until now). these
      experiments each only have a subset of the 17 megamat odors. Betty asked us to
      include all the megamat data for all paper KC correlations now, so I'll use these
      in combination with the 4 flies I had already been using

      - from `matrix/Remy-Data/projects/odor_space_collab/outputs/megamat`

    - `stim_rdms__iscell_good_xid0__correlation.xlsx` and `xrda_stim_rdm_concat__iscell_good_xid0__correlation.nc`
      - these are the two files Remy asked Anoop to use in an email the week before she
        oriented me to the old megamata data. I copied these in to check against
        correlations I compute from the old data.

      - from Remy's private repo `ejhonglab/OdorSpaceShare`, under `manuscript/data/by_imaging_panel/megamat_new_and_old_by_fly_17/kc_soma_nls`


- `from_matt`:
  - copied from my local copy of his code (`~/src/matt/matt-modeling/data/hemibrain`),
    which I had previously copied from his code on hal.

  - removed some larger files and some I don't use.


- `ann_model_outputs`
  - contents:
    - CSVs with ORN (Hallem) / KC responses and correlations.

  - used by:
    - `analyze_ann_outputs.py`

  - generated by my script `save_ann_outputs.m`, which is currently under
    `~/Documents/MATLAB`. may commit here or (more likely) under my copy of her
    `mushroomBody` repo.


# TODO

- add other files in initial 2024-04-05 Dropbox folder `hong_depasquale_collab`
  (under `sent_to_grant/2024-04-05`)
