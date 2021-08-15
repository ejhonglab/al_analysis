#!/usr/bin/env bash

# Assumes the following are already installed:
# - conda
# - git
# - awk

# Defaults to suite2p, but if you call as (for example) `./install.sh al_pair_grids`,
# this will take the value 'al_pair_grids'
conda_env_to_make=${1:-suite2p}

src_path=$HOME/src
mkdir -p ${src_path}
cd ${src_path}

declare -a required_git_repos=(
    # My modified version of suite2p
    "tom-f-oconnell/suite2p"

    "ejhonglab/hong2p"

    # Chances are we are running this script because we have already cloned this, but
    # leaving it in to enable one-line install getting the script via curl or something
    # (would require making this repo public).
    "ejhonglab/al_pair_grids"
)

git_auth_prefix="git@github.com:"

# Assuming everything is suitably up-to-date if already cloned.
for repo in "${required_git_repos[@]}"
do
    # https://stackoverflow.com/questions/17921544
    dir_name=$(echo $repo | awk -F/ '{print $NF}')
    if [ ! -d $dir_name ]; then
        git clone ${git_auth_prefix}${repo}
    fi
done

if conda env list | awk '{print $1}' | grep -q "^${conda_env_to_make}$"; then
    echo "Environment ${conda_env_to_make} already exists! Pass alternate name, e.g. \`./install.sh al_pair_grids\`"
    exit 1
fi

if ! [ -x "$(command -v mamba)" ]; then
    MAMBA_OR_CONDA="conda"
else
    MAMBA_OR_CONDA="mamba"
fi
# This would make a conda environment named 'suite2p' if we didn't pass -n explicitly
${MAMBA_OR_CONDA} env create -f suite2p/environment.yml -n ${conda_env_to_make}

conda activate ${conda_env_to_make}
cd ~/src/al_pair_grids
# TODO test that this actually works (that conda activate call before actually worked in
# the script context. if not, need to do this a different way)
pip install -r requirements.txt

declare -a required_env_vars=(
    # Technically can also set HONG2P_DATA, and it will take precedence over this
    "HONG_NAS"

    "HONG2P_FAST_DATA"
)
declare -a example_values=(
    # Should contain 'mb_team' directory immediately under it.
    "/path/to/nas"

    # Should contain 'raw_data' immediately under it.
    # Data will preferentially be loaded from here, and analysis intermediates will be
    # written to an 'analysis_intermediates' subdirectory that will be created here.
    "/local/copy/of/raw/data"
)
for i in "${!required_env_vars[@]}"
do
    env_var=${required_env_vars[i]}
    if [ -z "${!env_var}" ]; then
        echo "Set environment variable ${env_var} by adding a line to your ~/.bashrc such as:"
        echo "export ${env_var}=\"${example_values[i]}\""
        echo ""
    fi
done

