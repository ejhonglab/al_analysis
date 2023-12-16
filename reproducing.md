TODO:
- also talk about old kiwi/control data

- mention which data are good candidates for diagnostic examples
  - 2023-05-10/1?


NOTE: add `-s` to any of the following to ensure no across-fly steps are skipped.
The default skipped steps should be mentioned in the `-s` section of the
`./al_analysis.py -h` output.


megamat:
- all pebbled data:
  - 2 flies on 2023-04-22 had water for va/aa, but i had typically been including them.
    first fly after 2023-04-22 was on 2023-04-26.

  - TODO why was i not using 2023-04-03 - 2023-04-11 flies, which say
    "pfo for va/aa" in Google Sheet "Exclusion reason"? was it also that diagnostics
    weren't as good then (was still changing them?)? other data quality issues?

  ```
  ./al_analysis.py -d pebbled -n 6f -t 2023-04-22 -e 2023-06-22
  ```


- all GH146>6f data:
  - i also had a similar amount of GH146 data w/ 8s/8m (mostly latter), from
    2023-10-18 to 2023-11-30, but may never have analyzed that to same standard,
    b/c we decided to use same indicator as pebbled (6f), to try to keep comparison more
    similar.

  ```
  ./al_analysis.py -d GH146 -n 6f -t 2023-06-22 -e 2023-07-28
  ```


validation2:
- pebbled data (at final concentrations only):
  - last date of first 4 flies was 2023-11-21

  ```
  ./al_analysis.py -d pebbled -n 6f -t 2023-11-19
  ```


- all pebbled data (including first 3 flies, after which some odor concs changed):

  ```
  ./al_analysis.py -d pebbled -n 6f -t 2023-10-15
  ```

