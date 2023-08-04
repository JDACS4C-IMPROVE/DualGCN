import numpy as np
import os
import json

import candle
from training_framework import MetadataGenerate_version_IMPROVE, CelllineGraphAdjNorm, FeatureExtract, ModelEvaluate
import dualgcn_keras as bmk
# from get_data import path_function, output_paths

from keras.models import load_model
from code.layers.graph import GraphConv
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr

data_train_idx, data_test_idx, data_val_idx, drug_feature, ppi_adj_info, common_genes = MetadataGenerate_version_IMPROVE()
ppi_adj = CelllineGraphAdjNorm(ppi_adj_info, common_genes)
ckpt_dir = './checkpoints/'
X_drug_feat, X_drug_adj, X_cellline_feat, Y = FeatureExtract(data_val_idx, drug_feature, common_genes, israndom=False)
# X_drug_feat, X_drug_adj, X_cellline_feat, Y, cancer_type_list=FeatureExtract(data_idx, drug_feature, **path_dict)
X_cellline_feat_mean = np.mean(X_cellline_feat, axis=0)
X_cellline_feat_std  = np.std(X_cellline_feat, axis=0)
X_cellline_feat = (X_cellline_feat - X_cellline_feat_mean) / X_cellline_feat_std
X = [X_drug_feat, X_drug_adj, X_cellline_feat, np.tile(ppi_adj, (X_drug_feat.shape[0], 1, 1))]
model = load_model(
        os.path.join(ckpt_dir, 'MyBestDualGCNModel_new.h5'),   #TODO: must match with the file name in training func. 
        custom_objects={'GraphConv': GraphConv}
        )
cancertype2pcc = ModelEvaluate(model=model,
                                  X_val=X,
                                  Y_val=Y,
                                  data_test_idx_current=data_val_idx,
                                  eval_batch_size=32)