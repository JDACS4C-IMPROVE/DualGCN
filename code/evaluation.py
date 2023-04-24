"""
This python script is being scripted by Cesar Sanchez-Villalobos to 
evaluate the model. The script will be used just to check if the model can be 
uploaded for testing purposes. 

The model is stored in the  folder ../checkpoint/ and is named best_DualGCNmodel.h5, 
therefore, we first need to load the model. 
"""
import numpy as np
from layers.graph import GraphConv
from keras.models import load_model
from dualgcn_baseline_keras2 import MetadataGenerate, DataSplit, CelllineGraphAdjNorm, FeatureExtract, ModelEvaluate

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

model = load_model('../checkpoint/MyBestDualGCNModel.h5', custom_objects={'GraphConv': GraphConv})
# model.summary()

# NOTE: Let's change it to be just one argument for being more user friendly. Getting it from the data folder. 
# 
# We need to put everything on a single file, this way we will have the same files for training and testing, 
# once the new data arrives. 

data_dir = '../data/' # Work in progress (CESAR)
israndom = False 
Drug_info_file = '../data/drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Cell_line_info_file = '../data/CCLE/Cell_lines_annotations_20181226.txt'
Drug_feature_file = '../data/drug/drug_graph_feat'
Cancer_response_exp_file = '../data/CCLE/GDSC_IC50.csv'
PPI_file = "../data/PPI/PPI_network.txt"
selected_info_common_cell_lines = "../data/CCLE/cellline_list.txt"
selected_info_common_genes = "../data/CCLE/gene_list.txt"
celline_feature_folder = "../data/CCLE/omics_data"
TCGA_label_set = ["ALL","BLCA","BRCA","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ','SARC','UCEC','MESO', 'PRAD']

def test_model(model, 
               Drug_info_file, 
               Cell_line_info_file, 
               Drug_feature_file, 
               Cancer_response_exp_file, 
               PPI_file, 
               selected_info_common_cell_lines,
               selected_info_common_genes,
               celline_feature_folder):
    """This function is being scripted to test the model, note that the model is already
    trained and loaded at the beginning of this script, that means that this file needs:
    
    - The h5 model already loaded. 
    - All the data files related. 
    """
    ppi_adj_info, drug_feature, data_idx = MetadataGenerate(Drug_info_file,
                                                            Cell_line_info_file,
                                                            Drug_feature_file,
                                                            PPI_file,
                                                            selected_info_common_cell_lines, 
                                                            selected_info_common_genes)
    
    ppi_adj = CelllineGraphAdjNorm(ppi_adj_info,selected_info_common_genes)
    X_drug_feat, X_drug_adj, X_cellline_feat, Y, cancer_type_list=FeatureExtract(data_idx,
                                                                                 drug_feature,
                                                                                 celline_feature_folder,
                                                                                 selected_info_common_cell_lines, 
                                                                                 selected_info_common_genes)
    
    X_cellline_feat_mean = np.mean(X_cellline_feat, axis=0)
    X_cellline_feat_std  = np.std(X_cellline_feat, axis=0)
    X_cellline_feat = (X_cellline_feat - X_cellline_feat_mean) / X_cellline_feat_std
    X = [X_drug_feat, X_drug_adj, X_cellline_feat, np.tile(ppi_adj, (X_drug_feat.shape[0], 1, 1))]
    cancertype2pcc = ModelEvaluate(model=model,
                                  X_val=X,
                                  Y_val=Y,
                                  cancer_type_test_list=cancer_type_list,
                                  data_test_idx_current=data_idx,
                                  file_path_pcc_log='../log/pcc_DualGCNmodel_inference.log',
                                  file_path_spearman_log='../log/spearman_DualGCNmodel_inference.log',
                                  file_path_rmse_log='../log/rmse_DualGCNmodel_inference.log',
                                  file_path_csv='../log/result_DualGCNmodel_inference.csv',
                                  batch_size=128)

test_model(model, 
           Drug_info_file, 
           Cell_line_info_file, 
           Drug_feature_file, 
           Cancer_response_exp_file, 
           PPI_file, 
           selected_info_common_cell_lines,
           selected_info_common_genes,
           celline_feature_folder)

