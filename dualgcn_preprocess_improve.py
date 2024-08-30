"""
author: @cesarasa

Code for preprocessing data for the DualGCN model.

Required outputs
----------------
All the outputs from this preprocessing script are saved in params["ml_data_outdir"].

1. Model input data files.
   This script creates three data files corresponding to train, validation,
   and test data. These data files are used as inputs to the ML/DL model in
   the train and infer scripts. The file format is specified by
   params["data_format"].
   For GraphDRP, the generated files:
        train_data.pt, val_data.pt, test_data.pt

2. Y data files.
   The script creates dataframes with true y values and additional metadata.
   Generated files:
        train_y_data.csv, val_y_data.csv, and test_y_data.csv.
        
This script is based on the GraphDRP code made by Dr. Partin.

For this script, my last run in lambda took: 

real    2m59.189s
user    2m32.532s
sys     0m11.997s
"""
# NOTE: Order properly these imports. 
import sys
import warnings
from pathlib import Path
from typing import Dict

# Ignore warnings:
warnings.simplefilter(action='ignore', category=FutureWarning)

# [Req] IMPROVE/CANDLE imports:
from improve import framework as frm
from improve import drug_resp_pred as drp

# Model specific imports:
from model_utils import gene_information as gi
from model_utils.drug_utils import smiles_to_molecules, rdkit_mols_to_dc_features, save_hickles


# Check IMPROVE (Global Variables)
filepath = Path(__file__).resolve().parent
print(filepath)

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

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

# 2. Model-specific params (Model: DualGCN)
# All params in model_preproc_params are optional.
# If no params are required by the model, then it should be an empty list.
model_preproc_params = []

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
preprocess_parameters = app_preproc_params + model_preproc_params
# ---------------------

def run(params: Dict) -> str:
    
    # ------------------------------------------------------
    # [Req] Build paths and create output dir
    # ------------------------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)

    frm.create_outdir(outdir = params['ml_data_outdir'])

    ## --------------------------------------------------------------------------
    # --------------------------- Developer's Notes -----------------------------
    # --------------------------------------------------------------------------
    # NOTE: The omics object and the drug objects are created using the provided functions.

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
    gi.preprocess_omics_data(params['ml_data_outdir'], ge, cn)

    stages = {"train": params["train_split_file"],
            "val": params["val_split_file"],
            "test": params["test_split_file"]}
    # scaler = None

    # Load Responses: 
    print("\nLoad responses.")


        
    # Save drug graphs
    # ---------------------------------- Developer's Notes --------------------------------
    # NOTE  : This is being done for testing, afterwards, we can send it to a function in model_utils. 
    # NOTE 2: Try to document it properly.
    # --------------------------------------------------------------------------------------
            
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
        print(f"Stage: {stage}")
        print(df_response.shape)
        print(df_response.head())

        frm.save_stage_ydf(df_response, params, stage)

    return params['ml_data_outdir']

def main(args): 
    # req:
    additional_definitions = preprocess_parameters
    params = frm.initialize_parameters(filepath, 
                                       default_model = 'params_cs.txt', 
                                       additional_definitions = additional_definitions, 
                                       required=None)
    ml_data_outdir = run(params)
    print(f"\n Data Preprocessing finished. Data saved in {ml_data_outdir}")
    print("\n Finished Data Preprocessing")

if __name__ == "__main__":
    main(sys.argv[1:])