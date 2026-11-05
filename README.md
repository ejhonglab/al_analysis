
### Installation

```
mkdir -p ~/src
cd ~/src

git clone https://github.com/ejhonglab/al_analysis
cd al_analysis

# This will make a conda environment named 'al_analysis'
conda env create -f environment.yml
conda activate al_analysis
```

If instead using a venv made from the system Python, first install a C++ compiler and
the Python headers (needed to build the olfsysm C++ extension), with the `-dev` package
matching the Python version the venv is made with:
```
sudo apt install build-essential python3.14-dev
python3.14 -m venv venv
source venv/bin/activate
pip install -e .
```


### Updating

```
cd ~/src/al_analysis

# activate whatever conda/venv/etc environment you installed al_analysis into,
# if it is not a conda environment named 'al_analysis'
conda activate al_analysis

# this should install requirements specified in exact-requirements.txt, via a reference
# to that file in pyproject.toml. this `pip install .` command is also the only command
# or requirement specified in the environment.yml file (other than specifying the python
# version to [currently] 3.8.12)
pip install .
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


### Model provenance (olfsysm builds)

Every model run (`mb_model.fit_mb_model`) first checks that the installed `olfsysm`
extension is a clean build of a known commit: olfsysm bakes its git hash, a clean / dirty
flag and a hash of its compiled sources into the module at build time, and
`mb_model.olfsysm_is_clean()` also compares the build with the checkout it was installed
from (`mb_model.olfsysm_build_info()` / `olfsysm_checkout_info()` show the details). If the
check fails, the error says why and how to proceed: commit (or stash) the olfsysm changes
and rebuild with `pip install -e ~/src/olfsysm`, or, for development only,
```
export AL_ANALYSIS_ALLOW_DIRTY_OLFSYSM=1
```
to run anyway. `mb_model.model_provenance()` collects the olfsysm build info, the commit
and dirty state (with diff) of `al_analysis` and `hong2p`, the CPU, Python and package
versions, and content hashes of the model inputs; `fit_and_plot_mb_model` writes it as
`model_provenance.json` next to each output's parameter files, and the KC-remapping
experiment scripts include it in their `provenance.json`. The experiment harness
(`scripts/claude_pair_remapping_experiments`) and `test/generate_reference_outputs_for_repro.py`
always require a clean build (the environment variable is ignored there). The repro tests
in `test/test_mb_model.py` do not compare any of this: they check that behavior did not
change across versions.

### Installation notes

Initially tested on Ubuntu 18.04, mostly on 20.04, and also believe it has worked on at
least 24.04 (not sure if I've tested 26.04 yet). Some things, like symbolic links, may
not work as expected by default on Windows.

