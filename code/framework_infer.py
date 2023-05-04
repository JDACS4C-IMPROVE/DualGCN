import numpy as np
import os

from dualgcn_baseline_keras2 import MetadataGenerate, CelllineGraphAdjNorm, FeatureExtract, ModelEvaluate
from get_data import path_function, output_paths
from keras.models import load_model
from layers.graph import GraphConv
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr




def launch(path_dict, log_dict):
    ppi_adj_info, drug_feature, data_idx = MetadataGenerate(**path_dict)
    ppi_adj = CelllineGraphAdjNorm(ppi_adj_info, **path_dict)
    X_drug_feat, X_drug_adj, X_cellline_feat, Y, cancer_type_list=FeatureExtract(data_idx, drug_feature, **path_dict)
    X_cellline_feat_mean = np.mean(X_cellline_feat, axis=0)
    X_cellline_feat_std  = np.std(X_cellline_feat, axis=0)
    X_cellline_feat = (X_cellline_feat - X_cellline_feat_mean) / X_cellline_feat_std
    X = [X_drug_feat, X_drug_adj, X_cellline_feat, np.tile(ppi_adj, (X_drug_feat.shape[0], 1, 1))]
    model = load_model('../checkpoint/MyBestDualGCNModel.h5', custom_objects={'GraphConv': GraphConv})
    cancertype2pcc, cancertype2rmse, cancertype2spearman, overall_pcc, overall_rmse, overall_spearman, Y_pred = ModelEvaluate(model=model,
                                  X_val=X,
                                  Y_val=Y,
                                  cancer_type_test_list=cancer_type_list,
                                  data_test_idx_current=data_idx,
                                  batch_size=128,
                                  **log_dict)
    return Y_pred, Y, cancertype2pcc, cancertype2rmse, cancertype2spearman, overall_pcc, overall_rmse, overall_spearman


def main():
    path_dict = path_function('../data') 
    log_dict = output_paths('../log')
    Y_pred, Y, cancertype2pcc, cancertype2rmse, cancertype2spearman, overall_pcc, overall_rmse, overall_spearman = launch(path_dict, log_dict)
    print('Done inference')

if __name__ == "__main__":
    main()
