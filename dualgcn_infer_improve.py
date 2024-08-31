"""
author: @cesarasa

Code for inferencing with the trained DualGCN model.

Required outputs
----------------
All the outputs from this train script are saved in params["model_outdir"].

1. Trained model.
   The model is trained with train data and validated with val data. The model
   file name and file format are specified, respectively by
   params["model_file_name"] and params["model_file_format"].
   For DualGCN, the saved model:
        model.h5

2. Predictions on val data. 
   Raw model predictions calcualted using the trained model on val data. The
   predictions are saved in val_y_data_predicted.csv

3. Prediction performance scores on val data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in val_scores.json
   
This script is based on the GraphDRP code made by Dr. Partin.
"""

import numpy as np
import os
import sys
from pathlib import Path
from keras.models import load_model

# [Req] Model Specific imports
from dualgcn_train_improve import MetadataFramework, ModelEvaluate, train_params, metrics_list
from dualgcn_preprocess_improve import preprocess_parameters
from model_utils.feature_extraction import CelllineGraphAdjNorm, FeatureExtract
from code.layers.graph import GraphConv

# [Req] IMPROVE/CANDLE imports
import improve.framework as frm
from improve.metrics import compute_metrics
from improve import drug_resp_pred as drp

filepath = Path(__file__).resolve().parent # [Req]


# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_infer_params
# 2. model_infer_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params in this script.
app_infer_params = []

# 2. Model-specific params (Model: GraphDRP)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params
# ---------------------
# model_path = os.path.join(params["model_outdir"], f'{params["model_file_name"]}{params["model_file_format"]}')
# print(model_path)
# [Req] 
def run(params):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    batch_size = params["batch_size"]
    print(params)
    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])
    
    # ------------------------------------------------------
    # [Req] Create data names for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")
    
    data_train_idx, data_test_idx, data_val_idx, drug_feature, ppi_adj_info, common_genes = MetadataFramework(params)
    
    ppi_adj = CelllineGraphAdjNorm(ppi_adj_info, common_genes, params)
    
    # Data Extraction
    X_drug_feat, X_drug_adj, X_cellline_feat, Y = FeatureExtract(data_train_idx, 
                                                                 drug_feature, 
                                                                 params, 
                                                                 israndom=False)
    
    # Data Normalization + Type Conversion
    X_cellline_feat_mean = np.mean(X_cellline_feat, axis=0)
    X_cellline_feat_std  = np.std(X_cellline_feat, axis=0)
    X_cellline_feat = (X_cellline_feat - X_cellline_feat_mean) / X_cellline_feat_std
    X_cellline_feat = X_cellline_feat.astype('float16')
    X_drug_feat = X_drug_feat.astype('float16')
    X_drug_adj = X_drug_adj.astype('float16')
    ppi_adj = ppi_adj.astype('float16')
    
    # Data for inference
    X = [X_drug_feat, X_drug_adj, X_cellline_feat, np.tile(ppi_adj, (X_drug_feat.shape[0], 1, 1))]
    
    # Loading the Model
    # Load the best saved model (as determined based on val data)
    model_path = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]
    model = load_model(model_path, custom_objects={'GraphConv': GraphConv})
    
    pcc, rmse, spearman, rsquared, Y_pred = ModelEvaluate(model=model,
                                X_val=X,
                                Y_val=Y,
                                data_test_idx_current=data_test_idx,
                                eval_batch_size=batch_size)
    
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(params,
                             y_true = Y,
                             y_pred = Y_pred,
                             stage = 'test',
                             outdir = params["infer_outdir"])
    
    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(params,
                                                y_true = Y,
                                                y_pred= Y_pred,
                                                stage = 'test',
                                                outdir = params["infer_outdir"],
                                                metrics = metrics_list)

# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_parameters + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")
    return None

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
    