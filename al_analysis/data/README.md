The AL data and records of data sent to collaborators. The data the model reads moved
with it to pyolfsysm (in the olfsysm repository, `pyolfsysm/data`, with its history,
2026-11): `from_pratyush`, `PNtoKC_connections_raw.xlsx`, `from_remy` (except
`2024-06-05`), `internal`, `from_yang`, `from_matt`, `from_george`, `from_anoop`,
`ann_model_outputs`, `hc_data.csv`, `preprocessed_hallem`, `sent_to_anoop/2024-05-16`
and `sent_to_remy/2025-03-18`, `2025-03-19` and `2025-09-30_tom_orn_data_signed-max`;
their notes are in its `data/README.md`.

- `sent_to_remy`
  - `2023-01-06`
    These 4 files (2 CSVs, and 2 pickle files with same data) should contain the data
    Remy used to generate the original ORN / PN correlation matrices in the preprint.

    The preprint was posted 2023-02-16 originally, and this was the only data I had
    seemed to send to Remy (on Slack) recently before this (on 2023-01-06). I didn't
    send anything to her via email during the same time frame.

  - `2023-10-29`
    - should be final (no longer true, right? old response calc? is that only
      difference?) megamat data (pebbled and GH146) (`pebbled_ij_certain-roi_stats.csv`
      here is exactly the same as `sent_to_anoop/v1/megamat_ij_certain-roi_stats.csv`)

    - sent on: Slack

  - `2024-01-12`
    - should be final validation2 panel data (just pebbled, we never did PN imaging for
      this panel)

    - sent on: Slack


- `sent_to_anoop`
  - `v1`
    - megamat and validation2 ORN data (should be pretty much, if not exactly, as
      final). non-consensus glomeruli may not have been dropped, or may have been
      handled differently than what I do now (e.g. for megamat data I sent to DePasquale
      lab people more recently)?

  - `v2`
     - model KC outputs that still had the offset in the dF/F -> est. spike delta fn,
       and thus still had unnecessarily high KC correlations. Anoop never really used.


- `from_remy`
  - `2024-06-05`
    - odor metadata (full names, PubChem CIDs, etc) for megamat and validation2 panel
      odors
    - renamed `megamat17.xlsx` to `megamat.xlsx`


# TODO

- add other files in initial 2024-04-05 Dropbox folder `hong_depasquale_collab`
  (under `sent_to_grant/2024-04-05`)
