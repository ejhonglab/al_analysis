
- `rdm_concat__kc_soma__megamat__mean_peak__correlation__trialavg.nc`
  - contents:
    - correlations from Remy's final 4 megamat flies (17x17 odors).
      should be same correlations as shown throughout paper.

  - received from: Remy (include date + medium received from)

  - used by:
    - `al_analysis.py`:
      - to make fig (#?) comparing model KC vs real KC (check this true)
      - any other figs? check.


- `sent_to_anoop`
  - `v1`
    - megamat and validation2 ORN data (should be pretty much, if not exactly, as
      final). non-consensus glomeruli may not have been dropped, or may have been
      handled differently than what I do now (e.g. for megamat data I sent to DePasquale
      lab people more recently)?

  - `v2`
     - model KC outputs that still had the offset in the dF/F -> est. spike delta fn,
       and thus still had unnecessarily high KC correlations. Anoop never really used.
