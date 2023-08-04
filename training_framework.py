"""
    Trials for training with the new dataset.
    
    author: @cesarasa
    
    The idea is to use the new dataset for training the model, using the same 
    functions provided by ANL for loading the data. 
    
    NOTE: REMOVE HARD CODED STRINGS.
    """

## Importing base libraries:
import candle
import pandas as pd
import numpy as np
import csv
import os
import hickle as hkl
import improve_utils as iu
import scipy.sparse as sp
import random
import json
import tensorflow as tf

# warnings.simplefilter(action='ignore', category=FutureWarning)

## Auxiliary Libraries.
from dualgcn_keras import DualGCNBenchmark
from math import sqrt
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from os.path import join as pj

## Keras info
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, History
from keras.optimizers import Adam

## Sklear info:
from sklearn.metrics import mean_squared_error, r2_score

## Load model
from code.model import KerasMultiSourceDualGCNModel
from code.layers.graph import GraphLayer, GraphConv

def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mse(y, f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


# Just because the tensorflow warnings are a bit verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
selected_info_common_genes = "./data/IMPROVE_CCLE/gene_list.txt"

file_path = os.path.dirname(os.path.realpath(__file__))
# Param initialization:
def initialize_parameters():
    i_bmk = DualGCNBenchmark(
        file_path,                           # this is the path to this file needed to find default_model.txt
        'dualgcn_default_model.txt',         # name of the default_model.txt file
        'keras',                             # framework, choice is keras or pytorch
        prog='example_baseline',             # basename of the model
        desc='IMPROVE Benchmark'
    )
    #TODO: how to name prog and desc? any conventions? parameter txt in json format?

    gParameters = candle.finalize_parameters(i_bmk)  # returns the parameter dictionary built from 
                                                                # default_model.txt and overwritten by any 
                                                                # matching comand line parameters.

    return gParameters


PARAMS = initialize_parameters() 
Max_atoms = PARAMS["max_atoms"]



def MetadataGenerate_version_IMPROVE(path = './data_new/IMPROVE_CCLE/drug/drug_graph_feat/',
                                     drug_path = './data_new/drug/drug_graph_feat/',
                                     PPI_file = './data_new/IMPROVE_test/PPI/PPI_network_new.txt', 
                                     selected_info_common_genes = './data/IMPROVE_CCLE/gene_list.txt',
                                     source = 'CTRPv2', 
                                     split_file_train = 'CTRPv2_split_0_train.txt',
                                     split_file_test = 'CTRPv2_split_0_test.txt', 
                                     split_file_val = 'CTRPv2_split_0_val.txt', 
                                     target = 'auc',
                                     if_train = True):
    
    
    
    
    df_train = iu.load_single_drug_response_data_v2(source = source, split_file_name = split_file_train, y_col_name=target)
    df_val = iu.load_single_drug_response_data_v2(source = source, split_file_name = split_file_val, y_col_name=target)
    df_test = iu.load_single_drug_response_data_v2(source = source, split_file_name = split_file_test, y_col_name=target)

    
    # Sorting columns and removing source column, to be like the data_idx format
    df_train = df_train[['improve_sample_id', 'improve_chem_id', 'auc1']].values.tolist()
    df_test = df_test[['improve_sample_id', 'improve_chem_id', 'auc1']].values.tolist()
    df_val = df_val[['improve_sample_id', 'improve_chem_id', 'auc1']].values.tolist()
    
    drug_feature = {}
    for each in os.listdir(drug_path):
        feat_mat,adj_list,degree_list = hkl.load(drug_path + each)
        # Save the name of the drug "each" as the word for the dictionary
        
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
        
    # NOTE: I am adding the ppi_adj_info here. However, I am not sure why it is needed. 
    # common_genes = pd.read_csv(selected_info_common_genes, sep = '\t', header = None).values.squeeze().tolist()
    PPI_net = pd.read_csv(PPI_file, sep = '\t', header = None)

    list_omics = os.listdir('./data_new/IMPROVE_test/omics_data/')
    common_genes = []
    for each in list_omics:
        df = pd.read_csv('./data_new/IMPROVE_test/omics_data/' + each, sep = ',', index_col = 0)
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



# df_train, df_test, df_val, drug_feature, ppi_adj_info = MetadataGenerate_version_IMPROVE()
# print(Max_atoms)



# Normalize adjacent matrix D^{-0.5}{T}A^{T}D^{-0.5}
def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix

def CalculateGraphFeat(feat_mat,adj_list,israndom=False):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0]) 
    # print(feat_mat.shape)
    # print(feat.shape)
    feat[:feat_mat.shape[0],:] = feat_mat  
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1 
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2
    return [feat,adj_mat]

def CelllineGraphAdjNorm(ppi_adj_info, common_genes, **kwargs):
    # There is a problem here. IndexError: index 694 is out of bounds for axis 1 with size 689 Debug it with the jupyter Notebook. 
    # I think it is because the ppi_adj_info is not the same as the selected_info_common_genes.
    # with open(selected_info_common_genes) as f:
    #     common_genes = [item.strip() for item in f.readlines()]
    nb_nodes = len(common_genes)
    adj_mat = np.zeros((nb_nodes,nb_nodes),dtype='float32')
    for i in range(len(ppi_adj_info)):
        nodes = ppi_adj_info[i]
        for each in nodes:
            adj_mat[i,each] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    norm_adj = NormalizeAdj(adj_mat)
    return norm_adj 

def FeatureExtract(data_idx, drug_feature, 
                   common_genes, 
                   israndom=False, **kwargs):
    nb_instance = len(data_idx)
    drug_data = [[] for item in range(nb_instance)]
    cell_line_data_feature = [[] for item in range(nb_instance)]
    target = np.zeros(nb_instance, dtype='float32')
    cellline_drug_pair = []
    common_cell_lines = [item[0] for item in data_idx]
    
    # with open(selected_info_common_genes) as f:
    #     common_genes = [item.strip() for item in f.readlines()]
    dic_cell_line_feat = {}
    for each in common_cell_lines:
        dic_cell_line_feat[each] = pd.read_csv('./data_new/IMPROVE_test/omics_data/' + each + '.csv', index_col=0).values 
    for idx in range(nb_instance):
        cell_line_id, pubchem_id, ln_IC50 = data_idx[idx]
        cellline_drug_tmp = cell_line_id + "_" + pubchem_id
        cellline_drug_pair.append(cellline_drug_tmp)
        cell_line_feat_mat =  dic_cell_line_feat[cell_line_id] 
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)] 
        drug_data[idx] = CalculateGraphFeat(feat_mat, adj_list, israndom)
        cell_line_data_feature[idx] = cell_line_feat_mat
        target[idx] = ln_IC50
    drug_feat = np.array([item[0] for item in drug_data])
    drug_adj = np.array([item[1] for item in drug_data])
    return drug_feat, drug_adj, np.array(cell_line_data_feature), target

#%%
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
        pcc_val = pearsonr(self.y_val, y_pred_val[:,0])[0]
        spearman_val = spearman(self.y_val, y_pred_val[:,0])
        rmse_val = rmse(self.y_val, y_pred_val[:,0])
        rsquared = r2_score(self.y_val, y_pred_val[:,0])
        val_scores = {"val_loss": float(val_loss[0]), "pcc": float(pcc_val), 
                      "scc": float(spearman_val), "rmse": float(rmse_val),
                      "rsquared": float(rsquared)}

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
    
#%% Model training and Evaluation:

def ModelTraining(model, X_train, Y_train, validation_data, 
                  params):
    learn_rate = params["learning_rate"]
    ckpt_directory = params["ckpt_directory"]
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
    history = model.fit(x=X_train,y=Y_train,batch_size=batch_size,epochs=nb_epoch, validation_data=validation_data,callbacks=callbacks)
    return model, history

def ModelEvaluate(model, X_val, Y_val, data_test_idx_current, eval_batch_size=32):

    Y_pred = model.predict(X_val, batch_size=eval_batch_size)
    overall_pcc = pearsonr(Y_pred[:,0],Y_val)[0]
    overall_rmse = mean_squared_error(Y_val,Y_pred[:,0],squared=False)
    overall_spearman = spearmanr(Y_pred[:,0],Y_val)[0]
    overall_rsquared = r2_score(Y_val,Y_pred[:,0])
    
    # Printing: 
    print('Overall PCC: %.4f' % overall_pcc)
    print('Overall RMSE: %.4f' % overall_rmse)
    print('Overall Spearman: %.4f' % overall_spearman)
    print('Overall R2: %.4f' % overall_rsquared)
    
    # f_out_pcc = open(file_path_pcc_log,'w')
    # f_out_rmse = open(file_path_spearman_log,'w')
    # f_out_spearman = open(file_path_rmse_log,'w')
    # cancertype2pcc = {}
    # cancertype2rmse = {}
    # cancertype2spearman = {}
    # for each in TCGA_label_set:
    #     ind = [b for a,b in zip(cancer_type_test_list,list(range(len(Y_pred)))) if a==each]
    #     if len(ind)>1:
    #         cancertype2pcc[each] = pearsonr(Y_pred[:,0][ind],Y_val[ind])[0]
    #         f_out_pcc.write('%s\t%d\t%.4f\n'%(each,len(ind),cancertype2pcc[each]))

    #         cancertype2rmse[each] = mean_squared_error(Y_pred[:,0][ind],Y_val[ind],squared=False)
    #         f_out_rmse.write('%s\t%d\t%.4f\n'%(each,len(ind),cancertype2rmse[each]))

    #         cancertype2spearman[each] = spearmanr(Y_pred[:,0][ind],Y_val[ind])[0]
    #         f_out_spearman.write('%s\t%d\t%.4f\n'%(each,len(ind),cancertype2spearman[each]))

    # f_out_pcc.write("AvegePCC\t%.4f\n"%overall_pcc)
    # f_out_rmse.write("AvegeRMSE\t%.4f\n"%overall_rmse)
    # f_out_spearman.write("AvegeSpearman\t%.4f\n"%overall_spearman)
    # f_out_pcc.close()
    # f_out_rmse.close()
    # f_out_spearman.close()

    # f_out = open(file_path_csv,'w')
    # f_out.write('drug_id,cellline_id,cancer_type,IC50,IC50_predicted\n')
    # for i in range(len(cancer_type_test_list)):
    #     drug_ = data_test_idx_current[i][1]
    #     cellline_ = data_test_idx_current[i][0]
    #     predicted_ = Y_pred[i,0]
    #     true_ = Y_val[i]
    #     cancertype_ = cancer_type_test_list[i]
    #     f_out.write('%s,%s,%s,%.4f,%.4f\n'%(drug_,cellline_,cancertype_,true_,predicted_))
    # f_out.close()
    return overall_pcc, overall_rmse, overall_spearman, overall_rsquared, Y_pred


#%% Run routine: 
celline_feature_folder = './data_new/IMPROVE_test/omics_data/'
def run(params):
    print("In Run Function:\n")
    ckpt_directory = params["ckpt_directory"]
    output_dir = params["output_dir"]
    if not os.path.exists(ckpt_directory):
        os.mkdir(ckpt_directory)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    batch_size = params["batch_size"]
    seed = params["rng_seed"]
    n_fold = params["n_fold"]
    assert(n_fold > 1), "Num of split (folds) must be more than 1. "
    assert(params["use_gexpr"] or params["use_cnv"]), "Must use at least one cell line feature."

    np.random.seed(seed)
    random.seed(seed)

    data_train_idx, data_test_idx, data_val_idx, drug_feature, ppi_adj_info, common_genes = MetadataGenerate_version_IMPROVE()
    ppi_adj = CelllineGraphAdjNorm(ppi_adj_info, common_genes)

    # Training Data:
    X_train_drug_feat, X_train_drug_adj, X_train_cellline_feat, Y_train = FeatureExtract(data_train_idx, drug_feature, common_genes, israndom=False)
    X_train_cellline_feat_mean = np.mean(X_train_cellline_feat, axis=0)
    X_train_cellline_feat_std = np.std(X_train_cellline_feat, axis=0)
    X_train_cellline_feat = (X_train_cellline_feat - X_train_cellline_feat_mean) / X_train_cellline_feat_std
    # Reduce the ram usage:
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
    # Validation data: (for some reason, the original model is using validation as test)
    X_test_drug_feat, X_test_drug_adj, X_test_cellline_feat, Y_test = FeatureExtract(data_val_idx, drug_feature, common_genes, israndom=False)
    X_test_cellline_feat = (X_test_cellline_feat - X_train_cellline_feat_mean) / X_train_cellline_feat_std
    # X_test = [X_test_drug_feat,X_test_drug_adj,X_test_cellline_feat,np.array([ppi_adj for i in range(X_test_drug_feat.shape[0])])]
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

    print("... Evaluate the model ...")
    cancertype2pcc = ModelEvaluate(model=model,
                                  X_val=X_test,
                                  Y_val=Y_test,
                                  data_test_idx_current=data_test_idx,
                                  eval_batch_size=batch_size)
    print('Evaluation finished!')

    return False 

def main():
    params = initialize_parameters()
    print(params)
    history = run(params)
    print("Done.")


if __name__ == "__main__":
    main()
    
"""
Overall PCC: 0.5869
Overall RMSE: 0.1360
Overall Spearman: 0.4247
Overall R2: 0.2889
Evaluation finished!
Done.

real    127m6.168s
user    584m57.663s
sys     399m15.146s
"""

