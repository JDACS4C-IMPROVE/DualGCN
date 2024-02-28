"""
author: @cesarasa
"""
# NOTE: Order properly these imports. 
import os
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
# Model related:
from model_utils import gene_information as gi
# import rdkit
import deepchem as dc
import hickle as hkl 
from rdkit import Chem

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
# --------------------------- Developer's Notes

# NOTE: The omics object and the drug objects are created using the provided

# NOTE 2: The omics object and the drug object loads ALL the data and we access 
# through it via a dictionary. The same with the SMILES data, in case of having 
# to use more information, this could last a while. 

# NOTE 3: The data are loaded into Pandas Dataframe (how much memory is needed?)
# documentation.


# Load Omics data.
print("\nLoads omics data.")
omics_obj = drp.OmicsLoader(params)
ge = omics_obj.dfs['cancer_gene_expression.tsv'] # return gene expression
cn = omics_obj.dfs['cancer_copy_number.tsv']  # return copy number

# Load Drug Data.
print("\nLoad drugs data.")
drugs_obj = drp.DrugsLoader(params)
smi = drugs_obj.dfs['drug_SMILES.tsv']  # return SMILES data
# Put index in a column and name that column 'improve_chem_id'
smi.reset_index(inplace=True)
smi.rename(columns={'index': 'improve_chem_id'}, inplace=True)
# print(smi)

# Save files on GDSCv1-CCLE/split0
# gi.preprocess_omics_data(params['ml_data_outdir'], ge, cn)

# -------------------------------------------
# Construct ML data for every stage (train, val, test)
# [Req] All models must load response data (y data) using DrugResponseLoader().
# -------------------------------------------
stages = {"train": params["train_split_file"],
          "val": params["val_split_file"],
          "test": params["test_split_file"]}
scaler = None

# Load Responses: 
print("\nLoad responses.")


    
# Save drug graphs
# ---------------------------------- Developer's Notes --------------------------------
# NOTE  : This is being done for testing, afterwards, we can send it to a function in model_utils. 
# NOTE 2: Try to document it properly.
# --------------------------------------------------------------------------------------
def smiles_to_molecules(smiles_list : list) -> list:
    """The following function aims to convert SMILES to RDKit molecules. And save all the molecules in a list.
    """
    molecules = []
    for smile in smiles_list:
        molecule = Chem.MolFromSmiles(smile)
        molecules.append(molecule)
    
    return molecules

def rdkit_mols_to_dc_features(mols: list) -> list:
    """
    RDKit mols to DeepChem features

    The rdkit_mols_to_dc_features function takes a list of RDKit molecules and returns a list of DeepChem features. 
    It uses the ConvMolFeaturizer class from the DeepChem package to extract features from the molecules.
    """
    ft = dc.feat.ConvMolFeaturizer()
    feature_list = ft.featurize(mols)
    return feature_list

def save_hickles(feature_list, drug_id_list, save_dir) -> None:
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for ii in range(len(feature_list)):
        ft = feature_list[ii]
        obj = (ft.atom_features, ft.canon_adj_list, ft.deg_list)
        f_name = os.path.join(save_dir, str(drug_id_list[ii]) + ".hkl")
        hkl.dump(obj, f_name)
    
    return None 
        
for stage, split_file in stages.items():
    dr = drp.DrugResponseLoader(params, split_file = split_file, verbose = True)
    df_response = dr.dfs['response.tsv']
    
    # Keep only the required columns in the dataframe:
    df_response[['improve_sample_id', 'improve_chem_id', 'auc']]
    df_response.dropna(inplace=True)
    list_drugs = df_response['improve_chem_id'].unique().tolist()
    # Get the SMILES for the drugs in the list:
    smi_mask = smi['improve_chem_id'].isin(list_drugs)
    smiles = smi[smi_mask]
    smi_list = smiles['canSMILES'].tolist()
    # print(smi_list)
    mols = smiles_to_molecules(smi_list)
    # print(mols)
    features = rdkit_mols_to_dc_features(mols)
    
    # Save the features in a directory
    save_hickles(features, list_drugs, params['ml_data_outdir'] + '/drug_features/')
    # print shape and head
    # print(f"Stage: {stage}")
    # print(df_response.shape)
    # print(df_response.head())
    
    # frm.save_stage_ydf(df_response, params, stage)