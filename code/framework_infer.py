"""
Modified from GraphDRP https://github.com/JDACS4C-IMPROVE/GraphDRP/blob/develop/frm_infer.py 
"""

import numpy as np
import os
import json

import candle
from dualgcn_baseline_keras2 import MetadataGenerate, CelllineGraphAdjNorm, FeatureExtract, ModelEvaluate
import dualgcn_keras as bmk
from get_data import path_function, output_paths

from keras.models import load_model
from layers.graph import GraphConv
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr




def launch(path_dict, log_dict, ckpt_dir):
    ppi_adj_info, drug_feature, data_idx = MetadataGenerate(**path_dict)
    ppi_adj = CelllineGraphAdjNorm(ppi_adj_info, path_dict["selected_info_common_genes"])
    X_drug_feat, X_drug_adj, X_cellline_feat, Y, cancer_type_list=FeatureExtract(data_idx, drug_feature, **path_dict)
    X_cellline_feat_mean = np.mean(X_cellline_feat, axis=0)
    X_cellline_feat_std  = np.std(X_cellline_feat, axis=0)
    X_cellline_feat = (X_cellline_feat - X_cellline_feat_mean) / X_cellline_feat_std
    X = [X_drug_feat, X_drug_adj, X_cellline_feat, np.tile(ppi_adj, (X_drug_feat.shape[0], 1, 1))]
    model = load_model(
        os.join(ckpt_dir, 'MyBestDualGCNModel.h5'),   #TODO: must match with the file name in training func. 
        custom_objects={'GraphConv': GraphConv}
        )
    cancertype2pcc, cancertype2rmse, cancertype2spearman, overall_pcc, overall_rmse, overall_spearman, Y_pred = ModelEvaluate(model=model,
                                  X_val=X,
                                  Y_val=Y,
                                  cancer_type_test_list=cancer_type_list,
                                  data_test_idx_current=data_idx,
                                  batch_size=128,
                                  **log_dict)
    return Y_pred, Y, cancertype2pcc, cancertype2rmse, cancertype2spearman, overall_pcc, overall_rmse, overall_spearman


def run(params):
    print("In Run Function:\n")
    args = params
    # # In GraphDRP it is transformed to another type:
    # args = candle.ArgumentStruct(**params)
    # print("Note: now the args obj has the type", type(args))

    path_dict = path_function('../data')  #TODO: need to match with preprocessing. 
    log_dict = output_paths(args.log_dir)
    ckpt_dir = args.ckpt_directory
    Y_pred, Y, cancertype2pcc, cancertype2rmse, cancertype2spearman, overall_pcc, overall_rmse, overall_spearman = launch(path_dict, log_dict, ckpt_dir)
    #TODO: if they require to save the Y_pred. 

    scores = {"mse": float(overall_rmse) ** 2,
        "rmse": float(overall_rmse),
        "pcc": float(overall_pcc),
        "scc": float(overall_spearman)}

    # Supervisor HPO
    with open(os.join(args.output_dir, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return scores


def initialize_parameters():
    """ Initialize the parameters for the GraphDRP benchmark. """
    print("Initializing parameters\n")
    dualgcn_bmk = bmk.DualGCNBenchmark(
        filepath=bmk.file_path,
        defmodel="dualgcn_default_model.txt",
        framework="keras2",
        prog="DualGCN",
        desc="CANDLE compliant DualGCN",
    )
    gParameters = candle.finalize_parameters(dualgcn_bmk)
    return gParameters


def main():
    gParameters = initialize_parameters()
    print(gParameters)
    scores = run(gParameters)
    print("Done inference.")


if __name__ == "__main__":
    main()
