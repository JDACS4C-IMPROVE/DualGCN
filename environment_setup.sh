#!/bin/bash --login

set -e

# Create a new conda environment:
$CONDA_ENV_NAME=DualGCN36GPU python=3.6 pip --yes
pip install tensorflow==1.13.1 keras==2.1.4
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop

conda install -c conda-forge deepchem=2.4.0 --yes
conda install -c conda-forge rdkit=2021.03.5 --yes
pip install hickle==5.0.2 dill==0.3.4 # dill-0.3.6 h5py-3.8.0 hickle-5.0.2
pip install networkx
pip install scikit-learn pandas
