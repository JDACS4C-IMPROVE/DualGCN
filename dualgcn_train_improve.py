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

# Main Python (system) libraries
import json
import os
import random
import warnings
from math import sqrt
from os.path import join as pj
from pathlib import Path

# Third-party Python libraries
import hickle as hkl
import numpy as np
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, History, ModelCheckpoint
from keras.optimizers import Adam
from scipy import stats
# from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score

# [Req] IMPROVE/CANDLE imports
import improve.framework as frm
from improve.metrics import compute_metrics
from improve import drug_resp_pred as drp

# Custom libraries:
from code.model import KerasMultiSourceDualGCNModel
from model_utils.feature_extraction import CelllineGraphAdjNorm, FeatureExtract

# [Req] Imports from preprocess script
from dualgcn_preprocess_improve import preprocess_parameters

warnings.simplefilter(action='ignore', category=FutureWarning)

filepath = Path(__file__).resolve().parent # [Req]


# The following required information is stored in the params dictionary: 


# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_train_params
# 2. model_train_params
# 
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params for this script.
app_train_params   = []
model_train_params = []
train_params = app_train_params + model_train_params

# [Req] List of metrics names to be compute performance scores
metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]  

filepath = Path(__file__).resolve().parent
print(filepath)

# additional_definitions = train_params + preprocess_params

# params = frm.initialize_parameters(
#         filepath,
#         default_model="params_cs.txt",
#         additional_definitions=additional_definitions,
#         required=None,
#     )
# frm.create_outdir(outdir=params["model_outdir"])


def initialize_parameters():
    # preprocess_params is a global variable from the preprocess script
    additional_definitions = preprocess_parameters + train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="params_cs.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    return params

def MetadataFramework(params):
    
    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    frm.create_outdir(outdir=params["model_outdir"])
    
    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])
    
    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_name(params, stage="val")  # [Req]
    
    # DualGCN-specific: build Metadata framework.
    # Load csv files for train, val, and test sets
    df_train = pd.read_csv(params['train_ml_data_dir'] + '/' + 'train_y_data.csv')
    df_val = pd.read_csv(params['val_ml_data_dir'] + '/' + 'val_y_data.csv')
    df_test = pd.read_csv(params['test_ml_data_dir'] + '/' + 'test_y_data.csv')

    # Covnert the dataframes to lists
    df_train = df_train[['improve_sample_id', 'improve_chem_id', 'auc']].values.tolist()
    df_test = df_test[['improve_sample_id', 'improve_chem_id', 'auc']].values.tolist()
    df_val = df_val[['improve_sample_id', 'improve_chem_id', 'auc']].values.tolist()

    drug_feature = {}
    # Save the name of the drug as the word for a dictionary
    for each in os.listdir(params['drug_path']):
        feat_mat,adj_list,degree_list = hkl.load(params['drug_path'] + each)
        # Save the name of the drug "each" as the word for the dictionary
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]

    # Build the PPI network from the original file
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

class MyCallback(Callback):
    def __init__(self, validation_data, result_file_path, improve_score_path, patience):
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.patience = patience
        self.result_file_path = result_file_path
        self.improve_score_path = improve_score_path

    def on_train_begin(self,logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        return
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weight)
        self.model.save(self.result_file_path)
        if self.stopped_epoch > 0 :
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

        # IMPROVE supervisor HPO
        val_loss = self.model.evaluate(self.x_val, self.y_val)
        y_pred_val = self.model.predict(self.x_val)
        
        val_scores = frm.compute_performace_scores(params,
                                                y_true = self.y_val,
                                                y_pred= y_pred_val[:,0],
                                                stage = 'val',
                                                outdir = params["model_outdir"],
                                                metrics = metrics_list)
        

        print("\nIMPROVE_RESULT val_loss:\t{}\n".format(val_scores["val_loss"]))
        print("scores.json saved at", self.improve_score_path)
        with open(self.improve_score_path, "w", encoding="utf-8") as f:
            json.dump(val_scores, f, ensure_ascii=False, indent=4)

        return
    
def on_epoch_begin(self, epoch, logs={}):
        return

def on_epoch_end(self, epoch, logs={}):
    # Note: early stop with pcc val as the criterion.
    y_pred_val = self.model.predict(self.x_val)
    pcc_val = pearsonr(self.y_val, y_pred_val[:,0])[0]
    print('pcc-val: %s' % str(round(pcc_val,4)))
    if pcc_val > self.best:
        self.best = pcc_val
        self.wait = 0
        self.best_weight = self.model.get_weights()
    else:
        self.wait+=1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
    return

def ModelTraining(model, X_train, Y_train, validation_data, 
                  params):
    learn_rate = params["learning_rate"]
    ckpt_directory = params["output_dir"]
    batch_size = params["batch_size"]
    nb_epoch = params["epochs"]
    result_file_path = pj(ckpt_directory, 'best_DualGCNmodel_new.h5')
    result_file_path_callback = pj(ckpt_directory, 'MyBestDualGCNModel_new.h5')
    ckpt_save_best = params["ckpt_save_best"]
    ckpt_save_weights_only = params["ckpt_save_weights_only"]
    ckpt_save_best_metric = params["ckpt_save_best_metric"]
    metrics = params["metrics"]
    loss_func = params["loss"]
    improve_score_path = pj(params["output_dir"], "scores.json")

    optimizer = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics)
    callbacks = [ModelCheckpoint(result_file_path, monitor=ckpt_save_best_metric, save_best_only=ckpt_save_best, save_weights_only=ckpt_save_weights_only), 
                 MyCallback(validation_data=validation_data, result_file_path=result_file_path_callback, improve_score_path=improve_score_path, patience = 15)]
    # By default Keras' model.fit() returns a History callback object.
    history = model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=validation_data, callbacks=None)
    return model, history

def ModelEvaluate(model, X_val, Y_val, data_test_idx_current, eval_batch_size=32):

    Y_pred = model.predict(X_val, batch_size=eval_batch_size)
    
    val_scores = frm.compute_performace_scores(params,
                                                y_true = Y_val,
                                                y_pred= Y_pred[:,0],
                                                stage = 'val',
                                                outdir = params["model_outdir"],
                                                metrics = metrics_list)
    
    pcc = val_scores['pcc']
    rmse = val_scores['rmse']
    mse = val_scores['mse']
    spearman = val_scores['scc']
    rsquared = val_scores['r2']
    val_loss = val_scores['val_loss']
    # Printing: 
    print('Overall PCC: %.4f' % pcc)
    print('Overall RMSE: %.4f' % rmse)
    print('Overall Spearman: %.4f' % spearman)
    print('Overall R2: %.4f' % rsquared)
    print('Overall MSE: %.4f' % mse)
    print('Overall Val Loss: %.4f' % val_loss)
    return pcc, rmse, spearman, rsquared, Y_pred


def main(params):
    print('In main function:\n')
    ckpt_dir = params["ckpt_directory"]
    output_dir = params["output_dir"]
    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])
    print('Where the model is saved: ')
    print(modelpath)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 
    
    # os.makedirs('out_models/GDSCv1/split_4/model.pt')
    # Hyperparameters
    batch_size = params["batch_size"]
    seed = params["rng_seed"]
    n_fold = params["n_fold"]
    assert(n_fold > 1), "Num of split (folds) must be more than 1. "
    assert(params["use_gexpr"] or params["use_cnv"]), "Must use at least one cell line feature."
    
    np.random.seed(seed)
    random.seed(seed)
    # MetadataFramework(params)
    print('Loading Data with the Metadata Framework function')
    data_train_idx, data_test_idx, data_val_idx, drug_feature, ppi_adj_info, common_genes = MetadataFramework(params)
    print('Data loaded')
    print('Extracting Features')
    ppi_adj = CelllineGraphAdjNorm(ppi_adj_info, 
                                   common_genes, params)
    X_train_drug_feat, X_train_drug_adj, X_train_cellline_feat, Y_train = FeatureExtract(data_train_idx, 
                                                                                         drug_feature, 
                                                                                         params, 
                                                                                         israndom=False)
    print('Features extracted')
    X_train_cellline_feat_mean = np.mean(X_train_cellline_feat, axis=0)
    X_train_cellline_feat_std = np.std(X_train_cellline_feat, axis=0)
    X_train_cellline_feat = (X_train_cellline_feat - X_train_cellline_feat_mean) / X_train_cellline_feat_std
    X_train_cellline_feat = X_train_cellline_feat.astype('float16')
    X_train_drug_feat = X_train_drug_feat.astype('float16')
    X_train_drug_adj = X_train_drug_adj.astype('float16')
    ppi_adj = ppi_adj.astype('float16')
    X_train = [X_train_drug_feat, X_train_drug_adj, X_train_cellline_feat,
                np.tile(ppi_adj, (X_train_drug_feat.shape[0], 1, 1))]
    
    print("Training Data: ")
    print(X_train_drug_feat.shape)
    print(X_train_drug_adj.shape)
    print(X_train_cellline_feat.shape)
    print(Y_train.shape)
    
    X_test_drug_feat, X_test_drug_adj, X_test_cellline_feat, Y_test = FeatureExtract(data_val_idx, drug_feature, params, israndom=False)
    X_test_cellline_feat = (X_test_cellline_feat - X_train_cellline_feat_mean) / X_train_cellline_feat_std
    X_test = [X_test_drug_feat, X_test_drug_adj, X_test_cellline_feat, 
                np.tile(ppi_adj, (X_test_drug_feat.shape[0], 1, 1))]

    val_data = [X_test, Y_test]
    
    model = KerasMultiSourceDualGCNModel(
        use_gexpr=params["use_gexpr"],
        use_cn=params["use_cnv"],
        regr=params["regression"]
    ).createMaster(
        X_train[0][0].shape[-1],X_train[2][0].shape[-1],
        drug_gcn_units_list = params["drug_gcn_units_list"], 
        cell_feature_fc_units_list = params["cell_feature_fc_units_list"],
        fc_units_list = params["dense"],
        cell_line_gcn_units_list = params["cell_line_gcn_units_list"],
        universal_dropout = params["universal_dropout"],
        fc_layers_dropout = params["fc_layers_dropout"]
        )
    
    print("... Train the model ...")
    model, history = ModelTraining(model=model,
                          X_train=X_train,
                          Y_train=Y_train,
                          validation_data=val_data,
                          params=params)

    # Save the model
    model.save(pj(output_dir, 'MyBestDualGCNModel_force.h5'))
    
    # Save the history of the model into a txt
    with open(pj(output_dir, 'history.txt'), 'w') as f:
        f.write(str(history.history))
    print('Model trained!')
    print("... Evaluate the model ...")
    cancertype2pcc, overall_rmse, overall_spearman, overall_rsquared, Y_pred = ModelEvaluate(model=model,
                                  X_val=X_test,
                                  Y_val=Y_test,
                                  data_test_idx_current=data_test_idx,
                                  eval_batch_size=batch_size)
    print('Evaluation finished!')
    
    # Reshape Y_test and Y_pred to 1-dimensional
    Y_test = Y_test.reshape(-1, 1).flatten()
    Y_pred = Y_pred.reshape(-1, 1).flatten()
    
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(params,
                             y_true = Y_test,
                             y_pred = Y_pred,
                             stage = 'val',
                             outdir = params["model_outdir"])
    
    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performace_scores(params,
                                                y_true = Y_test,
                                                y_pred= Y_pred,
                                                stage = 'val',
                                                outdir = params["model_outdir"],
                                                metrics = metrics_list)
    
    
    
    # print(val_scores) 
    print("Done!")
    return None
    
    


if __name__ == "__main__":
    params = initialize_parameters()
    main(params)
    print("Done!")