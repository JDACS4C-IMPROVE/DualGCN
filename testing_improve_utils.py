import deepchem as dc
import networkx as nx
import pandas as pd
from rdkit import Chem
from improve_utils import load_single_drug_response_data_v2
# # smiles = "CCOc1ccn(-c2ccc(F)cc2)c(=O)c1C(=O)Nc1ccc(Oc2ccnc(N)c2Cl)c(F)c1"
responses_train = load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_train.txt', y_col_name='auc')
# responses_val = load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_val.txt', y_col_name='auc')
# responses_test = load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_test.txt', y_col_name='auc')\

# Load drug_SMILES.tsv
drug_smiles = pd.read_csv("csa_data/raw_data/x_data/drug_SMILES.tsv", sep="\t")
# print(drug_smiles.head())

# Get canSMILES from drug_smiles and match them with the improve_chem_id
responses = responses_train['improve_chem_id'].unique()

smiles_list = []
for response in responses:
    smiles_list.append(drug_smiles[drug_smiles['improve_chem_id'] == response]['canSMILES'].values.tolist()[0])
    
smile_id = smiles_list[15]
ft = dc.feat.ConvMolFeaturizer()
features = ft.featurize(smile_id)[0]
f0, a0, d0 = features.atom_features, features.canon_adj_list, features.deg_list
nn = len(d0)
g0 = nx.Graph()
g0.add_nodes_from(range(nn))
nx.set_node_attributes(g0, {node: features for node, features in zip(g0, f0)}, 'features')
for node, neighbors in enumerate(a0):
    for neighbor in neighbors:
        g0.add_edge(node, neighbor)

print(responses) 


## 

# For the omics data: 
print(len(responses_train['improve_sample_id'].unique()))