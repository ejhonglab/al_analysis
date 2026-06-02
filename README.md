
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


### For `al-analysis`
Set the environment variables `HONG_NAS` and `HONG2P_FAST_DATA` appropriately:

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


##### TODO

Document expectation on this Google Sheet? Make public read-only template and share
link?


### Running

```
# (or whatever else you named the environment for this project)
conda activate al_analysis

al-analysis <command line arguments ...>

# To display help message describing how to run and different command line arguments
al-analysis -h
```

Plots will be created under a directory such as `pdf`, under the current directory.


### Installation notes

Initially tested on Ubuntu 18.04, mostly on 20.04, and also believe it has worked on at
least 24.04 (not sure if I've tested 26.04 yet). Some things, like symbolic links, may
not work as expected by default on Windows.

