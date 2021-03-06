
### Installation

#### Manual

```
mkdir -p ~/src
cd ~/src

# My modified version of suite2p
git clone https://github.com/tom-f-oconnell/suite2p

git clone https://github.com/ejhonglab/hong2p
git clone https://github.com/ejhonglab/al_analysis

# This will make a conda environment named 'suite2p'
conda env create -f suite2p/environment.yml

conda activate suite2p
cd ~/src/al_analysis
pip install -r requirements.txt
```

Then set the environment variables `HONG_NAS` and `HONG2P_FAST_DATA` appropriately:

- `$HONG_NAS` should contain a directory `mb_team` directly under it.

  - This `mb_team` directory should contain `raw_data` and `stimulus_data_files`
    directly under it.

  - You may also set `HONG2P_DATA` as a folder that contains `raw_data` and
    `stimulus_data_files` directly under it.

- `$HONG2P_FAST_DATA` should contain a local copy of the data under a `raw_data`
  directory, again immediately under it. Analysis intermediates will be saved to a
  directory `analysis_intermediates`, which will be created under this directory.


#### Mostly automated

To install as manual install above, but using SSH authentication, you may simply call
`./install.sh` after cloning. If it asks you to set environment variables, you will need
to manually do so.


#### Setting up link to Google sheet with metadata

For the analysis to use the metadata in my Google Sheet `Hong lab documents/Tom - odor
mixture experiments/pair_grid_data`, open the sheet and copy the link to a file called
`pair_grid_data_gsheet_link.txt` in the top-level of this repository.


### Running

```
# (or whatever else you named the environment for this project)
conda activate suite2p

./al_analysis.py
```

Plots will be created under a directory such as `svg`, under the current directory.


### Installation notes

Only tested on Ubuntu 18.04.

Note that despite the fact that `suite2p/environment.yml` includes the line `suite2p`,
my current conda installation (despite `conda clean -a` and deleting all but the base
environment) thinks it already has `suite2p` when it gets to that part of the install
step. This is only really an issue because it suggests some other dependencies may not
be installed correctly, as I would otherwise want to prevent suite2p from being
installed, so that I can install an editable version of my fork without any ambiguity
about other versions possibly being installed.

This is the relevant line:
```
Requirement already satisfied: suite2p in /home/tom/src/suite2p (from -r /home/tom/src/suite2p/      condaenv.npju4l22.requirements.txt (line 15)) (0.10.2.dev11+gc7af998.d20210802)
```

