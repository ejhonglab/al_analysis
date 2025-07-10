TODO:
- also talk about old kiwi/control data
  this work?
  `./al_analysis.py -d pebbled -n 6f -v -t 2022-02-07 -e 2022-04-03 -s model`

- mention which data are good candidates for diagnostic examples
  - aphe was bumped from -5 to -4 starting with fly 2023-05-10, so it only was at final
    concentration for the last two megamat pebbled flies (2023-05-10/1 and 2023-06-22/1)

  - va -4 solvent was still water for 2023-04-22 and 2023-04-23
    - 5 megamat/pebbled flies from 2023-04-26/2 - 2023-05-09/1
      (where va -4 solvent was changed, but aphe still at -5)

  - 2023-05-10/1?
    - 2023-06-22/1 better (only other w/ all concs "final")?
    - or another fly enough better to show despite aphe -5?


NOTE: add `-s` to any of the following to ensure no across-fly steps are skipped.
The default skipped steps should be mentioned in the `-s` section of the
`./al_analysis.py -h` output.


megamat:
- all pebbled data (9 flies):
  - 2023-04-22/2
  - 2023-04-22/3
  - 2023-04-26/2
  - 2023-04-26/3
  - 2023-05-08/1
  - 2023-05-08/3
  - 2023-05-09/1
  - 2023-05-10/1
  - 2023-06-22/1

  - 2 flies on 2023-04-22 had water for va/aa, but i had typically been including them.
    first fly after 2023-04-22 was on 2023-04-26. va was still in the diagnostics on
    2023-04-22, but solvent was also water there (and thus also nulled for across-fly
    analyses).

  - TODO why was i not using 2023-04-03 - 2023-04-11 flies, which say
    "pfo for va/aa" in Google Sheet "Exclusion reason"? was it also that diagnostics
    weren't as good then (was still changing them?)? other data quality issues?

  ```
  ./al_analysis.py -d pebbled -n 6f -t 2023-04-22 -e 2023-06-22
  ```


- all GH146>6f data (7 flies):
  - i also had a similar amount of GH146 data w/ 8s/8m (mostly latter), from
    2022-10-18 to 2022-11-30, but may never have analyzed that to same standard,
    b/c we decided to use same indicator as pebbled (6f), to try to keep comparison more
    similar. the flies in the older data also likely had genetic issues (not
    backcrossed, and some balancers may have slipped through).

  ```
  ./al_analysis.py -d GH146 -n 6f -t 2023-06-22 -e 2023-07-28
  ```


validation2:
- pebbled data (at final concentrations only):
  - 5 good flies (after dropping 2024-01-05/4 at B's request, for looking anomalous):
    - 2023-11-19/1
    - 2023-11-21/1
    - 2023-11-21/2
    - 2023-11-21/3
    - 2024-01-05/1

  ```
  ./al_analysis.py -d pebbled -n 6f -t 2023-11-19 -e 2024-01-05
  ```


- all pebbled data (including first 3 flies, after which some odor concs changed):

  ```
  ./al_analysis.py -d pebbled -n 6f -t 2023-10-15 -e 2024-01-05
  ```

  NOTE: the 3 extra flies that would be analyzed here (2023-10-15/1, 10-19/1-2) are now
  marked explicitly for exclusion in Google sheet, so that I can analyze data
  all the megamat and validation2 data together with one command now.


fitting dF/F -> spiking model (analyzing all pebbled data, including megamat and
validation2 panels, as well as glomeruli diagnostics for those flies):
```
./al_analysis.py -d pebbled -n 6f -t 2023-04-22 -e 2024-01-05
```

NOTE: above command (which analyzes both megamat and validation2 data) will currently
cause script to exit after saving dF/F -> spiking model, rather than going on to run
modeling for both of those panels. to actually run the MB models, use commands which
restrict data to only either megamat OR validation2 date ranges (see above), AFTER
running this command to output one dF/F -> spiking model.


newer kiwi/control data:
to get model outputs for `natmix_data/analysis.py`
```
./al_analysis.py -d pebbled -n 6f -t 2024-09-03 -e 2024-10-01 -s model-seeds
```

