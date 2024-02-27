"""
author: @cesarasa
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# IMPROVE/CANDLE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

# Check IMPROVE
filepath = Path(__file__).resolve().parent
print(filepath)

# 1. App-specific params (App: monotherapy drug response prediction)
# Note! This list should not be modified (i.e., no params should added or
# removed from the list.
# 
# There are two types of params in the list: default and required
# default:   default values should be used
# required:  these params must be specified for the model in the param file

app_preproc_params = [
    {"name": "y_data_files", # default
     "type": str,
     "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
    },
    {"name": "x_data_canc_files", # required
     "type": str,
     "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    },
    {"name": "x_data_drug_files", # required
     "type": str,
     "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    },
    {"name": "canc_col_name",
     "default": "improve_sample_id", # default
     "type": str,
     "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    },
    {"name": "drug_col_name", # default
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name in the y (response) data file that contains the drug ids.",
    },

]

# NOTE: This will not be used as the graphs are created during the training stage!
# 2. Model-specific params (Model: DualGCN)
# All params in model_preproc_params are optional.
# If no params are required by the model, then it should be an empty list.
model_preproc_params = [
    {"name": "use_lincs",
     "type": frm.str2bool,
     "default": True,
     "help": "Flag to indicate if landmark genes are used for gene selection.",
    },
    {"name": "scaling",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "miabs", "robust"],
     "help": "Scaler for gene expression data.",
    },
    {"name": "ge_scaler_fname",
     "type": str,
     "default": "x_data_gene_expression_scaler.gz",
     "help": "File name to save the gene expression scaler object.",
    },
]

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
preprocess_params = app_preproc_params + model_preproc_params
# ---------------------

params = frm.initialize_parameters(filepath, 
                                   default_model = 'params_cs.txt', 
                                   additional_definitions = preprocess_params, 
                                   required=None)

params = frm.build_paths(params)

frm.create_outdir(outdir = params['ml_data_outdir'])

# [Req] Load X data (feature representations)
    # ------------------------------------------------------
# Use the provided data loaders to load data that is required by the model.
#
# Benchmark data includes three dirs: x_data, y_data, splits.
# The x_data contains files that represent feature information such as
# cancer representation (e.g., omics) and drug representation (e.g., SMILES).
#
# Prediction models utilize various types of feature representations.
# Drug response prediction (DRP) models generally use omics and drug features.
#
# If the model uses omics data types that are provided as part of the benchmark
# data, then the model must use the provided data loaders to load the data files
# from the x_data dir

## --------------------------------------------------------------------------
# NOTE: The omics object and the drug objects are created using the provided
# documentation
print("\nLoads omics data.")
omics_obj = drp.OmicsLoader(params)
print(type(omics_obj))
ge = omics_obj.dfs['cancer_gene_expression.tsv'] # return gene expression
# cn = omics_obj.dfs['cancer_copy_number.tsv']  # return copy number
print("\nLoad drugs data.")
drugs_obj = drp.DrugsLoader(params)
# # print(drugs_obj)
# smi = drugs_obj.dfs['drug_SMILES.tsv']  # return SMILES data
print(ge)
print(type(ge))