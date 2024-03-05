"""
author: @cesarasa

Code for training for the DualGCN model.

Required outputs
----------------
All the outputs from this train script are saved in params["model_outdir"].

1. Trained model.
   The model is trained with train data and validated with val data. The model
   file name and file format are specified, respectively by
   params["model_file_name"] and params["model_file_format"].
   For GraphDRP, the saved model:
        model.pt

2. Predictions on val data. 
   Raw model predictions calcualted using the trained model on val data. The
   predictions are saved in val_y_data_predicted.csv

3. Prediction performance scores on val data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in val_scores.json
   
This script is based on the GraphDRP code made by Dr. Partin.
"""
import pandas as pd
import warnings
from pathlib import Path

warnings.simplefilter(action='ignore', category=FutureWarning)

# IMPROVE/CANDLE:
from improve import framework as frm
from improve import drug_resp_pred as drp
filepath = Path(__file__).resolve().parent
print(filepath)

app_preproc_params = [
    # # These arg should be specified in the [modelname]_default_model.txt:
    # # y_data_files, x_data_canc_files, x_data_drug_files
    # {"name": "y_data_files", # default
    #  "type": str,
    #  "help": "List of files that contain the y (prediction variable) data. \
    #          Example: [['response.tsv']]",
    # },
    # {"name": "x_data_canc_files", # [Req]
    #  "type": str,
    #  "help": "List of feature files including gene_system_identifer. Examples: \n\
    #          1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
    #          2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
    # },
    # {"name": "x_data_drug_files", # [Req]
    #  "type": str,
    #  "help": "List of feature files. Examples: \n\
    #          1) [['drug_SMILES.tsv']] \n\
    #          2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
    # },
    # {"name": "canc_col_name",
    #  "default": "improve_sample_id", # default
    #  "type": str,
    #  "help": "Column name in the y (response) data file that contains the cancer sample ids.",
    # },
    # {"name": "drug_col_name", # default
    #  "default": "improve_chem_id",
    #  "type": str,
    #  "help": "Column name in the y (response) data file that contains the drug ids.",
    # },

]

# [Req] App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific args for the train script.
app_train_params = []

model_train_params = [
    # {"name": "model_arch",
    #  "default": "GINConvNet",
    #  "choices": ["GINConvNet", "GATNet", "GAT_GCN", "GCNNet"],
    #  "type": str,
    #  "help": "Model architecture to run."},
    # {"name": "log_interval",
    #  "action": "store",
    #  "type": int,
    #  "help": "Interval for saving o/p"},
    # {"name": "cuda_name",
    #  "action": "store",
    #  "type": str,
    #  "help": "Cuda device (e.g.: cuda:0, cuda:1."},
    # {"name": "learning_rate",
    #  "type": float,
    #  "default": 0.0001,
    #  "help": "Learning rate for the optimizer."
    # },
]

# [Req] List of metrics names to be compute performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

additional_definitions = app_train_params + model_train_params
params = frm.initialize_parameters(
        filepath,
        default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
frm.create_outdir(outdir=params["model_outdir"])

modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])
train_data_fname = frm.build_ml_data_name(params, stage="train")  # [Req]
val_data_fname = frm.build_ml_data_name(params, stage="val")  # [Req]

# print("train_data_fname:", train_data_fname)
# print("val_data_fname:", val_data_fname)


df_train = pd.read_csv(params['train_ml_data_dir'] + '/' + 'train_y_data.csv')
df_val = pd.read_csv(params['val_ml_data_dir'] + '/' + 'val_y_data.csv')
df_test = pd.read_csv(params['test_ml_data_dir'] + '/' + 'test_y_data.csv')
