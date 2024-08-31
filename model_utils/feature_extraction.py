
import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
from tqdm import tqdm

Max_atoms = 230
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

def CalculateGraphFeat(feat_mat, adj_list, params, israndom=False):
    assert feat_mat.shape[0] == len(adj_list)
    # print(f'Debugging: {feat_mat.shape[-1]}')
    # print(f'Debugging: {Max_atoms}')
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float16')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float16')
    if israndom:
        feat = np.random.rand(Max_atoms, feat_mat.shape[-1])
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

def CelllineGraphAdjNorm(ppi_adj_info, common_genes, params, **kwargs):
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
                   params, 
                   israndom=False, **kwargs):
    nb_instance = len(data_idx)
    drug_data = [[] for item in range(nb_instance)]
    cell_line_data_feature = [[] for item in range(nb_instance)]
    target = np.zeros(nb_instance, dtype='float32')
    cellline_drug_pair = []
    common_cell_lines = [item[0] for item in data_idx]
    print(type(common_cell_lines))
    # print(common_cell_lines)
    # with open(selected_info_common_genes) as f:
    #     common_genes = [item.strip() for item in f.readlines()]
    dic_cell_line_feat = dict()
    for each in tqdm(common_cell_lines):
        # print(each)
        # print(params['omics_path'])
        # # print(df_vals)
        # break
        dic_cell_line_feat[each] = pd.read_csv(params['omics_path'] + each + '.csv', index_col=0).values 
    for idx in tqdm(range(nb_instance)):
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