
Analysis of antennal lobe (AL) 2-photon imaging data from the Hong lab: the `al-analysis`
command processes ThorImage / ThorSync recordings into ORN / PN glomerulus responses and
plots, and `plot-roi` and `csvinfo` view ROIs and response tables. With the optional
`model` extra, `al-analysis` also runs the mushroom body model (pyolfsysm, in the
[olfsysm](https://github.com/ejhonglab/olfsysm) repository) on those responses.


### Installation (from scratch)

These steps make a new environment (a venv, in `al_analysis/venv`). Installs made before
2026-11, when the model moved to olfsysm, should be replaced this way rather than updated:
delete the old `venv` directory (or conda environment) first.

1. Python 3.14 with its `venv` module (on Ubuntu releases that do not package it, from the
   deadsnakes PPA), and git:
   ```
   sudo apt install python3.14 python3.14-venv git
   ```

2. Get the code and make the venv:
   ```
   mkdir -p ~/src
   cd ~/src
   git clone https://github.com/ejhonglab/al_analysis
   cd al_analysis
   python3.14 -m venv venv
   ```

3. Install, either
   - without the model (the AL analysis only: `al-analysis`, whose model step is skipped
     by default, `plot-roi` and `csvinfo`):
     ```
     ./venv/bin/pip install -e .
     ```
   - or with the model (also the `al-analysis` model step, `al_analysis.mb_model`, and
     the model's `plot-dynamics` and `step_model_pn_apl` commands). This builds the
     olfsysm C++ extension, so it also needs a C++ compiler and the Python headers:
     ```
     sudo apt install build-essential python3.14-dev
     ./venv/bin/pip install -e '.[model]'
     ```
   Without the model, the model step of `al-analysis` (`-s` without `model`) and
   `al_analysis.mb_model` stop with an error saying to install `al_analysis[model]`.

4. Run the commands from the venv (`./venv/bin/al-analysis -h`), or activate it first
   (`source venv/bin/activate`).

Python 3.8 (with pandas 1.5) is also supported: make the venv with `python3.8`, then
`./venv/bin/pip install -r exact-requirements-py38.txt` (which includes the model) and
`./venv/bin/pip install --no-deps -e .`.


### Updating

```
cd ~/src/al_analysis
git pull
# installs requirements specified in exact-requirements.txt (via a reference to that file
# in pyproject.toml); add [model] as above if it was installed with the model
./venv/bin/pip install -e .
```


### For `al-analysis` (the CLI to analyze AL 2p data)

`al-analysis` is a CLI program installed along other things when following the install
instructions above. It looks for data under directories set by the environment variables
discussed below. The data it expects is:

- raw ThorImage and ThorSync outputs (named / organized how? doc!)

- YAML files output by `olf` (under what subdir? doc!)
  [olfactometer control CLI](https://github.com/tom-f-oconnell/olfactometer))

  NOTE: each ThorImage experiment should have the name of the `olf` output YAML file
  (which is copied to the clipboard by `olf`) pasted into the "Notes" ThorImage field
  (and should be on a line by itself), as the analysis will use that to find the odor
  metadata for an experiment.

See environment variables section below for which subdirectories the above outputs are
expected to be in. If doing multiple recordings for one experiment (in one fly), let
ThorImage and ThorSync both increment the name of the recording automatically. ThorImage
recording names should include `diag` if they are part of the ~24-odor panel to
identify/segment glomeruli, and otherwise should include a shared string describing the
odor panel in the prefix. ThorSync output names should not need to manually be changed,
as long as they are auto-incrementing appropriately.

It will also only analyze data also entered appropriately (with fly metadata, etc) in a
Google Sheet, which is at a link set as described below. In the Google Sheet, each fly
should have at least:

- `Date`: should download in `YYYY-MM-DD` format. At least current date validation in my
  Google Sheet enforces this.

- `Fly`: fly number (numbered from 1 within each day)

- `Driver`: abbreviation for the driver (Gal4/etc) (e.g. 'pebbled')

- `Indicator`: abbreviation for the indicator (e.g. '6f')

- `Side`: 'left'/'right'. all 'right' hemisphere data will be flipped into left
  orientation, to keep analysis easy and outputs comparable.

- (optional) `Exclude`: a column with checkbox data validation, which will download as
  boolean. If checked, data from that fly will not be analyzed.


#### TODO
are there other columns currently used? `ImageJ ROIs done`?


#### Environment variable setup
Set the environment variables `HONG2P_DATA` and/or `HONG_NAS` and (optional)
`HONG2P_FAST_DATA` appropriately:

- `$HONG_NAS` should contain a directory `mb_team` directly under it.

  - This `mb_team` directory should contain `raw_data` and `stimulus_data_files`
    directly under it.

  - You may also set `HONG2P_DATA` as a folder that contains `raw_data` and
    `stimulus_data_files` directly under it.

- `$HONG2P_FAST_DATA` should contain a local copy of the data under a `raw_data`
  directory, again immediately under it. Analysis intermediates will be saved to a
  directory `analysis_intermediates`, which will be created under this directory.


#### Setting up link to Google sheet with metadata

For the analysis to use the metadata in my Google Sheet `Hong lab documents/Tom - odor
mixture experiments/pair_grid_data`, open the sheet and copy the link to a file called
`pair_grid_data_gsheet_link.txt` in the top-level of this repository.


#### TODO

- Document expectation on this Google Sheet? Make public read-only template and share
  link? Or commit example export (XLSX?) from it?

- Are `HONG_NAS` and `HONG2P_FAST_DATA` both actually required? isn't there a
  `HONG2P_DATA` too? update above to reflect actual minimum set


#### Running

```
# (or whatever else you named the environment for this project)
conda activate al_analysis

al-analysis <command line arguments ...>

# To display help message describing how to run and different command line arguments
al-analysis -h
```

Plots will be created under a directory such as `pdf`, under the current directory.


#### Drawing ROIs

TODO (+ explain how to setup ImageJ stuff and hotkeys for it, or link to repo that
explains that)


### The model

The model step of `al-analysis` runs pyolfsysm's `model_mb_responses`.
`al_analysis.mb_model` and `al_analysis.wpnkc_shuffles` forward to `pyolfsysm.mb_model` and
`pyolfsysm.wpnkc_shuffles`, and al_util's model-only names (Remy's KC data loaders,
`load_*_dff`, ...) to `pyolfsysm.util`, so older code importing them from al_analysis
still works; new code should import pyolfsysm. The model's data, scripts and tests, its
clean-build check (every model run first checks the installed olfsysm build) and the
provenance it records with each output are documented in olfsysm's README.


### Installation notes

Initially tested on Ubuntu 18.04, mostly on 20.04, and also believe it has worked on at
least 24.04 (not sure if I've tested 26.04 yet). Some things, like symbolic links, may
not work as expected by default on Windows.


### Development

To work on hong2p or olfsysm (the model) alongside this, clone them next to this
checkout and install them editable into this venv, after al_analysis (which installs them
from GitHub):
```
./venv/bin/pip install -e '.[model,dev]'
./venv/bin/pip install -e ../hong2p[all]
./venv/bin/pip install -e ../olfsysm
```

Tests (`QUICK=1` skips the slowest cases; the model's tests, including those of the
olfsysm C++ code, are in olfsysm's `test/`, since 2026-11):
```
QUICK=1 ./venv/bin/python -m pytest
```
