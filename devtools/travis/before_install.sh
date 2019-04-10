#!/usr/bin/env bash

# Temporarily change directory to $HOME to install software
pushd .
cd $HOME

# Install Miniconda
MINICONDA=Miniconda3-latest-Linux-x86_64.sh
export PYTHON_VER=$TRAVIS_PYTHON_VERSION

MINICONDA_HOME=$HOME/miniconda
MINICONDA_MD5=$(curl -s https://repo.continuum.io/miniconda/ | grep -A3 $MINICONDA | sed -n '4p' | sed -n 's/ *<td>\(.*\)<\/td> */\1/p')
wget -q https://repo.continuum.io/miniconda/$MINICONDA

if [[ $MINICONDA_MD5 != $(md5sum $MINICONDA | cut -d ' ' -f 1) ]]; then
    echo "Miniconda MD5 mismatch"
    exit 1
fi
bash $MINICONDA -b -p $MINICONDA_HOME

# Configure miniconda
export PIP_ARGS="-U"

echo ". $MINICONDA_HOME/etc/profile.d/conda.sh" >> ~/.bashrc  # Source the profile.d file
echo "conda activate" >> ~/.bashrc  # Activate conda
source ~/.bashrc # source file to get new commands

conda config --add channels conda-forge
conda config --set always_yes yes
conda install conda conda-build jinja2 anaconda-client
conda update --quiet --all

# Restore original directory
popd