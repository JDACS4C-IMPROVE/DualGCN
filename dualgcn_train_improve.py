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
import os
import hickle as hkl
import pandas as pd
import warnings
from pathlib import Path

warnings.simplefilter(action='ignore', category=FutureWarning)

# IMPROVE/CANDLE:
from improve import framework as frm
from improve import drug_resp_pred as drp

from model_utils.feature_extraction import NormalizeAdj, random_adjacency_matrix, CalculateGraphFeat, CelllineGraphAdjNorm, FeatureExtract

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
def MetadataFramework(params):
    # print("Loading Training Data")swd
    df_train = pd.read_csv(params['train_ml_data_dir'] + '/' + 'train_y_data.csv')
    # print("Loading Validation Data")
    df_val = pd.read_csv(params['val_ml_data_dir'] + '/' + 'val_y_data.csv')
    # print("Loading Testing Data")
    df_test = pd.read_csv(params['test_ml_data_dir'] + '/' + 'test_y_data.csv')

    df_train = df_train[['improve_sample_id', 'improve_chem_id', 'auc']].values.tolist()
    df_test = df_test[['improve_sample_id', 'improve_chem_id', 'auc']].values.tolist()
    df_val = df_val[['improve_sample_id', 'improve_chem_id', 'auc']].values.tolist()

    drug_feature = {}
    for each in os.listdir(params['drug_path']):
        feat_mat,adj_list,degree_list = hkl.load(params['drug_path'] + each)
        # Save the name of the drug "each" as the word for the dictionary
        
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]

    PPI_net = pd.read_csv(params['ppi_path'], sep = '\t', header=None)
    list_omics = os.listdir(params['omics_path'])
    common_genes = []
    for each in list_omics:
        df = pd.read_csv(params['omics_path'] + each, sep = ',', index_col = 0)
        # Give me the index of df
        common_genes.append(df.index.tolist())
    common_genes = list(set.intersection(*map(set, common_genes)))

    # Removing from PPI_net the genes that are not in common_genes
    PPI_net = PPI_net[PPI_net[0].isin(common_genes)]
    PPI_net = PPI_net[PPI_net[1].isin(common_genes)]

    idx_dic = {}
    for index, item in enumerate(common_genes):
        idx_dic[item] = index
    ppi_adj_info = [[] for item in common_genes] 
    # ppi_adj_info = pd.read_csv(ppi_info_file, sep = '\t', header = None)
    # print(ppi_adj_info.shape)
    for line in PPI_net.values.tolist():
        gene1, gene2 = line[0], line[1]
        # print(gene1, gene2)
        if idx_dic[gene1] <= idx_dic[gene2]:
            ppi_adj_info[idx_dic[gene1]].append(idx_dic[gene2])
            ppi_adj_info[idx_dic[gene2]].append(idx_dic[gene1])

    # Concatenate first and second column of PPI_net to get the list of all genes, drop duplicates and reset index
    genes_ppi = pd.concat([PPI_net[0], PPI_net[1]], axis = 0).drop_duplicates().reset_index(drop=True).values.tolist()
    print('Number of common genes: {}'.format(len(common_genes)))
    print('Number of PPI edges: {}'.format(len(ppi_adj_info)))
    print('Number of drugs: {}'.format(len(drug_feature)))
    print('Genes ppi {}'.format(len(genes_ppi)))
    return df_train, df_test, df_val, drug_feature, ppi_adj_info, common_genes



def main(params):
    # MetadataFramework(params)
    data_train_idx, data_test_idx, data_val_idx, drug_feature, ppi_adj_info, common_genes = MetadataFramework(params)
    ppi_adj = CelllineGraphAdjNorm(ppi_adj_info, common_genes)
    X_train_drug_feat, X_train_drug_adj, X_train_cellline_feat, Y_train = FeatureExtract(data_train_idx, drug_feature, common_genes, israndom=False)
    print(X_train_drug_feat.shape)
    print(X_train_drug_adj.shape)
    print(X_train_cellline_feat.shape)
    print(Y_train.shape)
    print(ppi_adj.shape)
    
if __name__ == "__main__":
    main(params)
    print("Done!")