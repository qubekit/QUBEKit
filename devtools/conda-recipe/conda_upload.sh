#!/usr/bin/env bash

PKG_NAME=qubekit
USER=cringrose
OS=linux-64

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
export VERSION=2.2.0
conda build .
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER $CONDA_BLD_PATH/$OS/$PKG_NAME-2.3.2.tar.bz2 --force
